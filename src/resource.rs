//! Persistent resource manifests that can be bound to one-shot graphs.
//!
//! Sets declare synchronization only; descriptor sets and recorded Vulkan commands remain separate.
//! Every resource used by those operations must also be present in the corresponding manifest.

use {
    crate::{
        driver::{
            DriverError,
            accel_struct::AccelerationStructure,
            device::Device,
            format_aspect_mask,
            image::{Image, ImageInfo, ImageSetQueue},
        },
        pool::Lease,
    },
    ash::{vk, vk::Handle as _},
    std::{
        collections::{HashMap, HashSet, hash_map::DefaultHasher},
        hash::{Hash, Hasher},
        sync::Arc,
    },
    vk_sync::AccessType,
};

/// The read performed through an [`AccelerationStructureSet`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AccelerationStructureAccessType {
    /// Reads every member while building another acceleration structure, such as a TLAS.
    BuildRead,

    /// Reads every member during ray traversal.
    RayTracingRead,
}

impl AccelerationStructureAccessType {
    pub(crate) const fn access_type(self) -> AccessType {
        match self {
            Self::BuildRead => AccessType::AccelerationStructureBuildRead,
            Self::RayTracingRead => AccessType::RayTracingShaderReadAccelerationStructure,
        }
    }
}

/// A persistent read-only acceleration structure manifest.
///
/// Exact duplicate members are retained once. Cloning a set is constant-time and preserves its
/// identity when bound to a [`Graph`](crate::Graph). A graph that accesses a set cannot directly
/// access one of its members. The default `checked` feature validates this restriction; without it,
/// the caller must uphold the restriction.
#[derive(Clone, Debug)]
pub struct AccelerationStructureSet {
    inner: Arc<AccelerationStructureSetInner>,
}

impl AccelerationStructureSet {
    pub(crate) fn addr(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    #[cfg(feature = "checked")]
    pub(crate) fn assert_device(&self, device: &Device) {
        if let Some(device_identity) = self.inner.device_identity {
            assert_eq!(
                device_identity,
                Device::identity(device),
                "acceleration structure set belongs to a different device"
            );
        }
    }

    /// Returns the immutable membership fingerprint used for diagnostics and caching.
    ///
    /// The fingerprint is process-local because resource identity is pointer-based.
    pub fn fingerprint(&self) -> u64 {
        self.inner.fingerprint
    }

    /// Returns `true` when the set has no members.
    pub fn is_empty(&self) -> bool {
        self.inner.unique_members.is_empty()
    }

    /// Returns the number of unique acceleration structures in the set.
    pub fn len(&self) -> usize {
        self.inner.unique_members.len()
    }

    /// Creates a persistent read-only acceleration structure set.
    ///
    /// Exact duplicate shared resources are deduplicated. Every member must belong to the same
    /// logical device.
    pub fn new<I, M>(members: I) -> Result<Self, DriverError>
    where
        I: IntoIterator<Item = M>,
        M: Into<AccelerationStructureSetMember>,
    {
        let members = members.into_iter();
        let mut device_identity = None;
        let mut member_addrs = HashSet::with_capacity(members.size_hint().0);
        let mut physical_acceleration_structures = HashSet::with_capacity(members.size_hint().0);
        let mut unique_members = Vec::with_capacity(members.size_hint().0);

        for member in members {
            let member = member.into();
            let acceleration_structure = member.acceleration_structure();
            merge_device_identity(
                &mut device_identity,
                Device::identity(&acceleration_structure.buffer.device),
            )?;
            if member_addrs.insert(member.addr()) {
                physical_acceleration_structures
                    .insert(PhysicalAccelerationStructureId::of(acceleration_structure));
                unique_members.push(member);
            }
        }

        let fingerprint = membership_fingerprint(unique_members.iter().map(|member| member.addr()));

        Ok(Self {
            inner: Arc::new(AccelerationStructureSetInner {
                #[cfg(feature = "checked")]
                device_identity,
                fingerprint,
                physical_acceleration_structure_count: physical_acceleration_structures.len(),
                unique_members: unique_members.into_boxed_slice(),
            }),
        })
    }

    pub(crate) fn physical_acceleration_structure_count(&self) -> usize {
        self.inner.physical_acceleration_structure_count
    }

    /// Returns the unique members in first-seen order.
    pub fn unique_members(&self) -> &[AccelerationStructureSetMember] {
        &self.inner.unique_members
    }
}

#[derive(Debug)]
struct AccelerationStructureSetInner {
    #[cfg(feature = "checked")]
    device_identity: Option<usize>,
    fingerprint: u64,
    physical_acceleration_structure_count: usize,
    unique_members: Box<[AccelerationStructureSetMember]>,
}

/// One acceleration structure retained by an [`AccelerationStructureSet`].
#[derive(Clone, Debug)]
pub struct AccelerationStructureSetMember {
    resource: AccelerationStructureSetResource,
}

impl AccelerationStructureSetMember {
    /// Returns the retained acceleration structure.
    pub fn acceleration_structure(&self) -> &AccelerationStructure {
        match &self.resource {
            AccelerationStructureSetResource::Owned(resource) => resource,
            AccelerationStructureSetResource::Pooled(resource) => resource,
        }
    }

    pub(crate) fn addr(&self) -> usize {
        match &self.resource {
            AccelerationStructureSetResource::Owned(resource) => Arc::as_ptr(resource) as usize,
            AccelerationStructureSetResource::Pooled(resource) => Arc::as_ptr(resource) as usize,
        }
    }
}

impl From<Arc<AccelerationStructure>> for AccelerationStructureSetMember {
    fn from(resource: Arc<AccelerationStructure>) -> Self {
        Self {
            resource: AccelerationStructureSetResource::Owned(resource),
        }
    }
}

impl From<&Arc<AccelerationStructure>> for AccelerationStructureSetMember {
    fn from(resource: &Arc<AccelerationStructure>) -> Self {
        Arc::clone(resource).into()
    }
}

impl From<Arc<Lease<AccelerationStructure>>> for AccelerationStructureSetMember {
    fn from(resource: Arc<Lease<AccelerationStructure>>) -> Self {
        Self {
            resource: AccelerationStructureSetResource::Pooled(resource),
        }
    }
}

impl From<&Arc<Lease<AccelerationStructure>>> for AccelerationStructureSetMember {
    fn from(resource: &Arc<Lease<AccelerationStructure>>) -> Self {
        Arc::clone(resource).into()
    }
}

#[derive(Clone, Debug)]
enum AccelerationStructureSetResource {
    Owned(Arc<AccelerationStructure>),
    Pooled(Arc<Lease<AccelerationStructure>>),
}

/// The access performed through an [`ImageSet`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ImageAccessType {
    /// Samples every member through any shader stage.
    SampledRead,
}

impl ImageAccessType {
    pub(crate) const fn access_type(self) -> AccessType {
        match self {
            Self::SampledRead => AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        }
    }
}

/// A persistent sampled read-only image manifest.
///
/// Logical descriptor slots retain their original order while duplicate image and subresource
/// pairs share one unique synchronization member. Cloning a set is constant-time and preserves its
/// identity when bound to a [`Graph`](crate::Graph). A graph that accesses a set cannot directly
/// access one of its members. The default `checked` feature validates this restriction; without it,
/// the caller must uphold the restriction.
#[derive(Clone, Debug)]
pub struct ImageSet {
    inner: Arc<ImageSetInner>,
}

impl ImageSet {
    pub(crate) fn addr(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    #[cfg(feature = "checked")]
    pub(crate) fn assert_device(&self, device: &Device) {
        if let Some(device_identity) = self.inner.device_identity {
            assert_eq!(
                device_identity,
                Device::identity(device),
                "image set belongs to a different device"
            );
        }
    }

    pub(crate) fn exclusive_physical_image_count(&self) -> usize {
        self.inner.exclusive_physical_image_count
    }

    /// Returns the immutable membership fingerprint used for diagnostics and caching.
    ///
    /// The fingerprint includes persistent image identities, exact subresource ranges, slot order,
    /// and duplicate slots. It is process-local because image identity is pointer-based.
    pub fn fingerprint(&self) -> u64 {
        self.inner.fingerprint
    }

    /// Returns `true` when the set has no descriptor slots.
    pub fn is_empty(&self) -> bool {
        self.inner.slot_to_member.is_empty()
    }

    /// Returns the number of logical descriptor slots, including duplicates.
    pub fn len(&self) -> usize {
        self.inner.slot_to_member.len()
    }

    /// Creates a persistent sampled read-only image set.
    ///
    /// Each input item defines one logical descriptor slot. Remaining mip-level and array-layer
    /// counts are resolved before exact duplicate `Image` and subresource-range pairs are
    /// deduplicated. Every member must belong to the same logical device.
    pub fn new<I, M>(slots: I) -> Result<Self, DriverError>
    where
        I: IntoIterator<Item = M>,
        M: Into<ImageSetMember>,
    {
        let slots = slots.into_iter();
        let mut normalized_slots = Vec::with_capacity(slots.size_hint().0);
        let mut device_identity = None;

        for slot in slots {
            let mut member: ImageSetMember = slot.into();
            merge_device_identity(
                &mut device_identity,
                Device::identity(&member.image().device),
            )?;
            member.subresource =
                ImageSetMember::normalize_subresource(member.image().info, member.subresource)?;
            normalized_slots.push(member);
        }

        let (unique_members, slot_to_member) =
            deduplicate(normalized_slots, ImageSetMemberKey::from_member)?;
        let fingerprint = ImageSetInner::membership_fingerprint(&unique_members, &slot_to_member);
        let queue = Arc::new(ImageSetQueue::new());
        let mut physical_images = HashSet::with_capacity(unique_members.len());
        let mut exclusive_physical_images = HashSet::with_capacity(unique_members.len());
        for member in &unique_members {
            let image = member.image();
            let image_id = PhysicalImageId::of(image);
            physical_images.insert(image_id);
            if image.info.sharing_mode != vk::SharingMode::CONCURRENT {
                exclusive_physical_images.insert(image_id);
                image.register_image_set_queue(&queue);
            }
        }

        Ok(Self {
            inner: Arc::new(ImageSetInner {
                #[cfg(feature = "checked")]
                device_identity,
                exclusive_physical_image_count: exclusive_physical_images.len(),
                fingerprint,
                physical_image_count: physical_images.len(),
                queue,
                slot_to_member,
                unique_members,
            }),
        })
    }

    pub(crate) fn physical_image_count(&self) -> usize {
        self.inner.physical_image_count
    }

    pub(crate) fn publish_queue(&self, queue: (u32, u32)) {
        self.inner.queue.publish(queue);
    }

    pub(crate) fn queue(&self) -> Option<(u32, u32)> {
        self.inner.queue.queue()
    }

    /// Returns the unique-member index for every logical descriptor slot.
    pub fn slot_to_member(&self) -> &[u32] {
        &self.inner.slot_to_member
    }

    /// Iterates members in logical descriptor-slot order.
    pub fn slots(&self) -> impl ExactSizeIterator<Item = &ImageSetMember> {
        self.inner
            .slot_to_member
            .iter()
            .map(|&member_idx| &self.inner.unique_members[member_idx as usize])
    }

    /// Returns the unique image and subresource members in first-seen order.
    pub fn unique_members(&self) -> &[ImageSetMember] {
        &self.inner.unique_members
    }
}

#[derive(Debug)]
struct ImageSetInner {
    #[cfg(feature = "checked")]
    device_identity: Option<usize>,
    exclusive_physical_image_count: usize,
    fingerprint: u64,
    physical_image_count: usize,
    queue: Arc<ImageSetQueue>,
    slot_to_member: Box<[u32]>,
    unique_members: Box<[ImageSetMember]>,
}

impl ImageSetInner {
    fn membership_fingerprint(unique_members: &[ImageSetMember], slot_to_member: &[u32]) -> u64 {
        membership_fingerprint(slot_to_member.iter().map(|&member_idx| {
            ImageSetMemberKey::from_member(&unique_members[member_idx as usize])
        }))
    }
}

impl Drop for ImageSetInner {
    fn drop(&mut self) {
        for member in &self.unique_members {
            member.image().unregister_image_set_queue(&self.queue);
        }
    }
}

/// One unique owned or pool-leased image and subresource range retained by an [`ImageSet`].
#[derive(Clone, Debug)]
pub struct ImageSetMember {
    resource: ImageSetResource,
    subresource: vk::ImageSubresourceRange,
}

impl ImageSetMember {
    /// Returns the retained image.
    pub fn image(&self) -> &Image {
        match &self.resource {
            ImageSetResource::Owned(resource) => resource,
            ImageSetResource::Pooled(resource) => resource,
        }
    }

    /// Creates a set member for an image subresource range.
    ///
    /// Remaining mip-level and array-layer counts are resolved when the member is added to a
    /// [`ImageSet`].
    pub fn new(image: Arc<Image>, subresource: vk::ImageSubresourceRange) -> Self {
        Self {
            resource: ImageSetResource::Owned(image),
            subresource,
        }
    }

    pub(crate) fn addr(&self) -> usize {
        match &self.resource {
            ImageSetResource::Owned(resource) => Arc::as_ptr(resource) as usize,
            ImageSetResource::Pooled(resource) => Arc::as_ptr(resource) as usize,
        }
    }

    fn normalize_subresource(
        info: ImageInfo,
        subresource: vk::ImageSubresourceRange,
    ) -> Result<vk::ImageSubresourceRange, DriverError> {
        let image_aspects = format_aspect_mask(info.format);
        if !info.usage.contains(vk::ImageUsageFlags::SAMPLED)
            || subresource.aspect_mask.is_empty()
            || !image_aspects.contains(subresource.aspect_mask)
            || !valid_subresource_count(
                subresource.base_mip_level,
                subresource.level_count,
                info.mip_level_count,
                vk::REMAINING_MIP_LEVELS,
            )
            || !valid_subresource_count(
                subresource.base_array_layer,
                subresource.layer_count,
                info.array_layer_count,
                vk::REMAINING_ARRAY_LAYERS,
            )
        {
            return Err(DriverError::InvalidData);
        }

        Ok(info.resolve_subresource_counts(subresource))
    }

    /// Returns the image subresource range synchronized for this member.
    ///
    /// Members borrowed from an [`ImageSet`] always contain explicit counts.
    pub fn subresource(&self) -> vk::ImageSubresourceRange {
        self.subresource
    }
}

impl From<Arc<Image>> for ImageSetMember {
    fn from(image: Arc<Image>) -> Self {
        let subresource = image.info.into();
        Self::new(image, subresource)
    }
}

impl From<&Arc<Image>> for ImageSetMember {
    fn from(image: &Arc<Image>) -> Self {
        Arc::clone(image).into()
    }
}

impl From<(Arc<Image>, vk::ImageSubresourceRange)> for ImageSetMember {
    fn from((image, subresource): (Arc<Image>, vk::ImageSubresourceRange)) -> Self {
        Self::new(image, subresource)
    }
}

impl From<(&Arc<Image>, vk::ImageSubresourceRange)> for ImageSetMember {
    fn from((image, subresource): (&Arc<Image>, vk::ImageSubresourceRange)) -> Self {
        Self::new(Arc::clone(image), subresource)
    }
}

impl From<Arc<Lease<Image>>> for ImageSetMember {
    fn from(resource: Arc<Lease<Image>>) -> Self {
        let subresource = resource.info.into();
        Self {
            resource: ImageSetResource::Pooled(resource),
            subresource,
        }
    }
}

impl From<&Arc<Lease<Image>>> for ImageSetMember {
    fn from(resource: &Arc<Lease<Image>>) -> Self {
        Arc::clone(resource).into()
    }
}

impl From<(Arc<Lease<Image>>, vk::ImageSubresourceRange)> for ImageSetMember {
    fn from((resource, subresource): (Arc<Lease<Image>>, vk::ImageSubresourceRange)) -> Self {
        Self {
            resource: ImageSetResource::Pooled(resource),
            subresource,
        }
    }
}

impl From<(&Arc<Lease<Image>>, vk::ImageSubresourceRange)> for ImageSetMember {
    fn from((resource, subresource): (&Arc<Lease<Image>>, vk::ImageSubresourceRange)) -> Self {
        (Arc::clone(resource), subresource).into()
    }
}

#[derive(Clone, Debug)]
enum ImageSetResource {
    Owned(Arc<Image>),
    Pooled(Arc<Lease<Image>>),
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct ImageSetMemberKey {
    aspect_mask: u32,
    base_array_layer: u32,
    base_mip_level: u32,
    image_addr: usize,
    layer_count: u32,
    level_count: u32,
}

impl ImageSetMemberKey {
    fn from_member(member: &ImageSetMember) -> Self {
        Self::new(member.addr(), member.subresource)
    }

    fn new(image_addr: usize, subresource: vk::ImageSubresourceRange) -> Self {
        Self {
            aspect_mask: subresource.aspect_mask.as_raw(),
            base_array_layer: subresource.base_array_layer,
            base_mip_level: subresource.base_mip_level,
            image_addr,
            layer_count: subresource.layer_count,
            level_count: subresource.level_count,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PhysicalAccelerationStructureId {
    device: usize,
    handle: u64,
}

impl PhysicalAccelerationStructureId {
    #[cfg(test)]
    pub(crate) const fn from_parts(device: usize, handle: u64) -> Self {
        Self { device, handle }
    }

    pub(crate) fn of(acceleration_structure: &AccelerationStructure) -> Self {
        Self {
            device: Device::identity(&acceleration_structure.buffer.device),
            handle: acceleration_structure.handle.as_raw(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PhysicalImageId {
    device: usize,
    handle: u64,
}

impl PhysicalImageId {
    #[cfg(test)]
    pub(crate) const fn from_parts(device: usize, handle: u64) -> Self {
        Self { device, handle }
    }

    pub(crate) fn of(image: &Image) -> Self {
        Self {
            device: Device::identity(&image.device),
            handle: image.handle.as_raw(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ResourceSet {
    AccelerationStructure(AccelerationStructureSet),
    Image(ImageSet),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResourceSetAccessType {
    AccelerationStructure(AccelerationStructureAccessType),
    Image(ImageAccessType),
}

impl ResourceSetAccessType {
    pub(crate) const COUNT: usize = 3;

    pub(crate) const fn access_type(self) -> AccessType {
        match self {
            Self::AccelerationStructure(access) => access.access_type(),
            Self::Image(access) => access.access_type(),
        }
    }

    pub(crate) const fn acquisition_offset(self) -> usize {
        match self {
            Self::Image(ImageAccessType::SampledRead) => 0,
            Self::AccelerationStructure(AccelerationStructureAccessType::BuildRead) => 1,
            Self::AccelerationStructure(AccelerationStructureAccessType::RayTracingRead) => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct ResourceSetIndex(usize);

impl ResourceSetIndex {
    pub(crate) const fn as_usize(self) -> usize {
        self.0
    }

    pub(crate) const fn new(index: usize) -> Self {
        Self(index)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ResourceSetKey {
    AccelerationStructure(usize),
    Image(usize),
}

#[derive(Debug, Default)]
pub(crate) struct ResourceSetMap {
    addr_index: HashMap<ResourceSetKey, ResourceSetIndex>,
    sets: Vec<ResourceSet>,
}

impl ResourceSetMap {
    pub(crate) fn bind(&mut self, set: &ResourceSet) -> ResourceSetIndex {
        match set {
            ResourceSet::AccelerationStructure(set) => self.bind_acceleration_structure(set),
            ResourceSet::Image(set) => self.bind_image(set),
        }
    }

    pub(crate) fn bind_acceleration_structure(
        &mut self,
        set: &AccelerationStructureSet,
    ) -> ResourceSetIndex {
        let key = ResourceSetKey::AccelerationStructure(set.addr());

        *self.addr_index.entry(key).or_insert_with(|| {
            let index = ResourceSetIndex::new(self.sets.len());
            self.sets
                .push(ResourceSet::AccelerationStructure(set.clone()));
            index
        })
    }

    pub(crate) fn bind_image(&mut self, set: &ImageSet) -> ResourceSetIndex {
        let key = ResourceSetKey::Image(set.addr());

        *self.addr_index.entry(key).or_insert_with(|| {
            let index = ResourceSetIndex::new(self.sets.len());
            self.sets.push(ResourceSet::Image(set.clone()));
            index
        })
    }

    pub(crate) fn get(&self, index: ResourceSetIndex) -> &ResourceSet {
        &self.sets[index.as_usize()]
    }

    pub(crate) fn get_acceleration_structure(
        &self,
        index: ResourceSetIndex,
    ) -> Option<&AccelerationStructureSet> {
        let ResourceSet::AccelerationStructure(set) = self.get(index) else {
            return None;
        };

        Some(set)
    }

    pub(crate) fn get_image(&self, index: ResourceSetIndex) -> Option<&ImageSet> {
        let ResourceSet::Image(set) = self.get(index) else {
            return None;
        };

        Some(set)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }

    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = &ResourceSet> {
        self.sets.iter()
    }

    pub(crate) fn len(&self) -> usize {
        self.sets.len()
    }
}

type Deduplicated<T> = (Box<[T]>, Box<[u32]>);

fn deduplicate<T, K>(
    slots: impl IntoIterator<Item = T>,
    mut key: impl FnMut(&T) -> K,
) -> Result<Deduplicated<T>, DriverError>
where
    K: Eq + Hash,
{
    let slots = slots.into_iter();
    let mut member_indices = HashMap::with_capacity(slots.size_hint().0);
    let mut slot_to_member = Vec::with_capacity(slots.size_hint().0);
    let mut unique_members = Vec::new();

    for member in slots {
        let key = key(&member);
        let member_idx = if let Some(&member_idx) = member_indices.get(&key) {
            member_idx
        } else {
            let member_idx =
                u32::try_from(unique_members.len()).map_err(|_| DriverError::InvalidData)?;
            member_indices.insert(key, member_idx);
            unique_members.push(member);
            member_idx
        };
        slot_to_member.push(member_idx);
    }

    Ok((
        unique_members.into_boxed_slice(),
        slot_to_member.into_boxed_slice(),
    ))
}

fn valid_subresource_count(base: u32, count: u32, total: u32, remaining: u32) -> bool {
    base < total
        && (count == remaining
            || count > 0 && base.checked_add(count).is_some_and(|end| end <= total))
}

fn merge_device_identity(
    set_device_identity: &mut Option<usize>,
    member_device_identity: usize,
) -> Result<(), DriverError> {
    if set_device_identity.is_some_and(|identity| identity != member_device_identity) {
        return Err(DriverError::InvalidData);
    }

    *set_device_identity = Some(member_device_identity);

    Ok(())
}

fn membership_fingerprint<K>(keys: impl ExactSizeIterator<Item = K>) -> u64
where
    K: Hash,
{
    let mut hasher = DefaultHasher::new();
    keys.len().hash(&mut hasher);
    for key in keys {
        key.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod test {
    use {super::*, crate::Graph, std::sync::Arc};

    #[derive(Debug)]
    struct TestMember {
        image: Arc<u8>,
        subresource: vk::ImageSubresourceRange,
    }

    fn test_member_key(member: &TestMember) -> ImageSetMemberKey {
        ImageSetMemberKey::new(Arc::as_ptr(&member.image) as usize, member.subresource)
    }

    fn range(base_mip_level: u32, level_count: u32) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level,
            level_count,
            base_array_layer: 0,
            layer_count: 1,
        }
    }

    fn empty_set() -> ImageSet {
        ImageSet::new(std::iter::empty::<ImageSetMember>()).unwrap()
    }

    fn empty_acceleration_structure_set() -> AccelerationStructureSet {
        AccelerationStructureSet::new(std::iter::empty::<AccelerationStructureSetMember>()).unwrap()
    }

    fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
        if let Some(message) = payload.downcast_ref::<&str>() {
            message
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message
        } else {
            "non-string panic"
        }
    }

    #[test]
    fn image_set_deduplicates_exact_members_and_preserves_slots() {
        let image_a = Arc::new(0);
        let image_b = Arc::new(0);
        let range_0 = range(0, 1);
        let range_1 = range(1, 1);
        let slots = [
            TestMember {
                image: Arc::clone(&image_a),
                subresource: range_0,
            },
            TestMember {
                image: Arc::clone(&image_a),
                subresource: range_0,
            },
            TestMember {
                image: Arc::clone(&image_a),
                subresource: range_1,
            },
            TestMember {
                image: Arc::clone(&image_b),
                subresource: range_0,
            },
            TestMember {
                image: Arc::clone(&image_a),
                subresource: range_0,
            },
        ];

        let (members, slot_to_member) = deduplicate(slots, test_member_key).unwrap();

        assert_eq!(members.len(), 3);
        assert!(Arc::ptr_eq(&members[0].image, &image_a));
        assert_eq!(members[0].subresource.base_mip_level, 0);
        assert!(Arc::ptr_eq(&members[1].image, &image_a));
        assert_eq!(members[1].subresource.base_mip_level, 1);
        assert!(Arc::ptr_eq(&members[2].image, &image_b));
        assert_eq!(&*slot_to_member, &[0, 0, 1, 2, 0]);
    }

    #[test]
    fn image_set_normalizes_remaining_counts_before_deduplication() {
        let image = Arc::new(0);
        let info = ImageInfo::image_2d_array(
            1,
            1,
            3,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
        )
        .into_builder()
        .mip_level_count(4)
        .build();
        let mut remaining = range(0, vk::REMAINING_MIP_LEVELS);
        remaining.layer_count = vk::REMAINING_ARRAY_LAYERS;
        let mut explicit = range(0, 4);
        explicit.layer_count = 3;
        let mut slots = [
            TestMember {
                image: Arc::clone(&image),
                subresource: remaining,
            },
            TestMember {
                image,
                subresource: explicit,
            },
        ];
        for member in &mut slots {
            member.subresource =
                ImageSetMember::normalize_subresource(info, member.subresource).unwrap();
        }

        let (members, slot_to_member) = deduplicate(slots, test_member_key).unwrap();

        assert_eq!(members.len(), 1);
        assert_eq!(members[0].subresource.level_count, 4);
        assert_eq!(members[0].subresource.layer_count, 3);
        assert_eq!(&*slot_to_member, &[0, 0]);
    }

    #[test]
    fn image_set_validates_usage_and_subresource_range() {
        let sampled_info = ImageInfo::image_2d(
            4,
            4,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
        )
        .into_builder()
        .mip_level_count(4)
        .build();

        assert!(ImageSetMember::normalize_subresource(sampled_info, range(3, 1)).is_ok());
        assert_eq!(
            ImageSetMember::normalize_subresource(
                sampled_info,
                range(1, vk::REMAINING_MIP_LEVELS),
            )
            .unwrap()
            .level_count,
            3
        );
        assert!(matches!(
            ImageSetMember::normalize_subresource(sampled_info, range(4, 1)),
            Err(DriverError::InvalidData)
        ));
        assert!(matches!(
            ImageSetMember::normalize_subresource(sampled_info, range(3, 2)),
            Err(DriverError::InvalidData)
        ));

        let mut invalid_array_layer = range(0, 1);
        invalid_array_layer.base_array_layer = 1;
        assert!(matches!(
            ImageSetMember::normalize_subresource(sampled_info, invalid_array_layer),
            Err(DriverError::InvalidData)
        ));

        let mut remaining_array_layers = range(0, 1);
        remaining_array_layers.layer_count = vk::REMAINING_ARRAY_LAYERS;
        assert_eq!(
            ImageSetMember::normalize_subresource(sampled_info, remaining_array_layers)
                .unwrap()
                .layer_count,
            1
        );

        let storage_info = ImageInfo::image_2d(
            4,
            4,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
        );
        assert!(matches!(
            ImageSetMember::normalize_subresource(storage_info, range(0, 1)),
            Err(DriverError::InvalidData)
        ));
    }

    #[test]
    fn image_set_rejects_mixed_device_identities() {
        let mut identity = None;

        assert!(merge_device_identity(&mut identity, 7).is_ok());
        assert!(merge_device_identity(&mut identity, 7).is_ok());
        assert!(matches!(
            merge_device_identity(&mut identity, 8),
            Err(DriverError::InvalidData)
        ));
    }

    #[test]
    fn image_set_fingerprint_tracks_slot_order_and_multiplicity() {
        let image_a = Arc::new(0);
        let image_b = Arc::new(0);
        let key_a = ImageSetMemberKey::new(Arc::as_ptr(&image_a) as usize, range(0, 1));
        let key_b = ImageSetMemberKey::new(Arc::as_ptr(&image_b) as usize, range(0, 1));

        let fingerprint = membership_fingerprint([key_a, key_b, key_a].into_iter());

        assert_eq!(
            fingerprint,
            membership_fingerprint([key_a, key_b, key_a].into_iter())
        );
        assert_ne!(
            fingerprint,
            membership_fingerprint([key_b, key_a, key_a].into_iter())
        );
        assert_ne!(
            fingerprint,
            membership_fingerprint([key_a, key_b].into_iter())
        );
    }

    #[test]
    fn physical_acceleration_structure_id_includes_device_and_handle() {
        let acceleration_structure = PhysicalAccelerationStructureId::from_parts(1, 2);

        assert_eq!(
            acceleration_structure,
            PhysicalAccelerationStructureId::from_parts(1, 2)
        );
        assert_ne!(
            acceleration_structure,
            PhysicalAccelerationStructureId::from_parts(2, 2)
        );
        assert_ne!(
            acceleration_structure,
            PhysicalAccelerationStructureId::from_parts(1, 3)
        );
    }

    #[test]
    fn physical_image_id_includes_device_and_handle() {
        let image = PhysicalImageId::from_parts(1, 2);

        assert_eq!(image, PhysicalImageId::from_parts(1, 2));
        assert_ne!(image, PhysicalImageId::from_parts(2, 2));
        assert_ne!(image, PhysicalImageId::from_parts(1, 3));
    }

    #[test]
    fn acceleration_structure_set_binding_reuses_persistent_identity() {
        let set = empty_acceleration_structure_set();
        let clone = set.clone();
        let mut graph = Graph::new();

        let node = graph.bind_resource(&set);
        let clone_node = graph.bind_resource(&clone);

        assert_eq!(node, clone_node);
        assert!(graph.resources.is_empty());
        assert_eq!(graph.resource_sets.sets.len(), 1);
        assert_eq!(graph.resource(node).addr(), set.addr());
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert!(set.unique_members().is_empty());
    }

    #[test]
    fn acceleration_structure_set_resource_access_preserves_distinct_read_profiles() {
        let mut graph = Graph::new();
        let resource_set = graph.bind_resource(&empty_acceleration_structure_set());

        graph
            .begin_cmd()
            .resource_access(resource_set, AccelerationStructureAccessType::BuildRead)
            .resource_access(resource_set, AccelerationStructureAccessType::BuildRead)
            .resource_access(
                resource_set,
                AccelerationStructureAccessType::RayTracingRead,
            )
            .record_cmd(|_| {})
            .end_cmd();

        let submission = graph.finalize();
        let accesses = &submission.graph().cmds[0].execs[0].resource_set_accesses;

        assert_eq!(accesses.len(), 2);
        assert_eq!(
            accesses[0].access_type,
            ResourceSetAccessType::AccelerationStructure(
                AccelerationStructureAccessType::BuildRead
            )
        );
        assert_eq!(
            accesses[1].access_type,
            ResourceSetAccessType::AccelerationStructure(
                AccelerationStructureAccessType::RayTracingRead
            )
        );
    }

    #[test]
    fn image_set_binding_reuses_persistent_identity() {
        let set = empty_set();
        let clone = set.clone();
        let mut graph = Graph::new();

        let node = graph.bind_resource(&set);
        let clone_node = graph.bind_resource(&clone);

        assert_eq!(node, clone_node);
        assert!(graph.resources.is_empty());
        assert_eq!(graph.resource_sets.sets.len(), 1);
        assert_eq!(graph.resource(node).addr(), set.addr());
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert_eq!(set.unique_members().len(), 0);
        assert!(set.slot_to_member().is_empty());
        assert_eq!(set.slots().len(), 0);

        let submission = graph.finalize();
        assert_eq!(submission.resource(node).addr(), set.addr());
    }

    #[test]
    fn image_set_member_accepts_owned_and_pooled_images() {
        fn assert_member<T: Into<ImageSetMember>>() {}

        assert_member::<Arc<Image>>();
        assert_member::<Arc<Lease<Image>>>();
        assert_member::<(Arc<Image>, vk::ImageSubresourceRange)>();
        assert_member::<(Arc<Lease<Image>>, vk::ImageSubresourceRange)>();
    }

    #[test]
    fn distinct_empty_image_sets_bind_distinct_nodes() {
        let mut graph = Graph::new();

        let lhs = graph.bind_resource(&empty_set());
        let rhs = graph.bind_resource(&empty_set());

        assert_ne!(lhs, rhs);
        assert_eq!(graph.resource_sets.sets.len(), 2);
    }

    #[test]
    fn image_set_resource_access_merges_per_execution_without_expanding_members() {
        let mut graph = Graph::new();
        let resource_set = graph.bind_resource(&empty_set());

        graph
            .begin_cmd()
            .resource_access(resource_set, ImageAccessType::SampledRead)
            .resource_access(resource_set, ImageAccessType::SampledRead)
            .record_cmd(|_| {})
            .resource_access(resource_set, ImageAccessType::SampledRead)
            .record_cmd(|_| {})
            .end_cmd();

        let submission = graph.finalize();
        let graph = submission.graph();

        assert!(graph.resources.is_empty());
        assert_eq!(graph.resource_sets.len(), 1);
        assert_eq!(graph.cmds.len(), 1);
        assert_eq!(graph.cmds[0].execs.len(), 2);
        for exec in &graph.cmds[0].execs {
            assert_eq!(exec.accesses.iter().len(), 0);
            assert_eq!(exec.resource_set_accesses.len(), 1);
            assert_eq!(
                exec.resource_set_accesses[0].resource_set_idx,
                resource_set.index()
            );
        }
    }

    #[test]
    fn image_set_resource_access_preserves_distinct_set_order() {
        let mut graph = Graph::new();
        let lhs = graph.bind_resource(&empty_set());
        let rhs = graph.bind_resource(&empty_set());
        let mut cmd = graph.begin_cmd();

        cmd.set_resource_access(rhs, ImageAccessType::SampledRead);
        cmd.set_resource_access(lhs, ImageAccessType::SampledRead);
        cmd.record_cmd_mut(|_| {});
        cmd.end_cmd();

        let submission = graph.finalize();
        let accesses = &submission.graph().cmds[0].execs[0].resource_set_accesses;

        assert_eq!(accesses.len(), 2);
        assert_eq!(accesses[0].resource_set_idx, rhs.index());
        assert_eq!(accesses[1].resource_set_idx, lhs.index());
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn acceleration_structure_set_rejects_direct_member_access() {
        let device = Device::create(crate::driver::device::DeviceInfo::default()).unwrap();
        let acceleration_structure = Arc::new(
            AccelerationStructure::create(
                &device,
                crate::driver::accel_struct::AccelerationStructureInfo::blas(1),
            )
            .unwrap(),
        );
        let set = AccelerationStructureSet::new([Arc::clone(&acceleration_structure)]).unwrap();
        let mut graph = Graph::new();
        let acceleration_structure_node = graph.bind_resource(acceleration_structure);
        let set_node = graph.bind_resource(&set);

        graph
            .begin_cmd()
            .resource_access(
                acceleration_structure_node,
                AccessType::AccelerationStructureBuildRead,
            )
            .resource_access(set_node, AccelerationStructureAccessType::BuildRead)
            .record_cmd(|_| {})
            .end_cmd();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| graph.finalize()))
            .expect_err("direct member access should be rejected");
        assert!(
            panic_message(panic.as_ref())
                .contains("acceleration structure set member cannot also be accessed directly")
        );
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn image_set_rejects_direct_member_access() {
        let device = Device::create(crate::driver::device::DeviceInfo::default()).unwrap();
        let image = Arc::new(
            Image::create(
                &device,
                ImageInfo::image_2d(
                    1,
                    1,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::SAMPLED,
                ),
            )
            .unwrap(),
        );
        let set = ImageSet::new([Arc::clone(&image)]).unwrap();
        let mut graph = Graph::new();
        let image_node = graph.bind_resource(Arc::clone(&image));
        let set_node = graph.bind_resource(&set);

        graph
            .begin_cmd()
            .resource_access(
                image_node,
                AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
            )
            .resource_access(set_node, ImageAccessType::SampledRead)
            .record_cmd(|_| {})
            .end_cmd();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| graph.finalize()))
            .expect_err("direct member access should be rejected");
        assert!(
            panic_message(panic.as_ref())
                .contains("image set member cannot also be accessed directly")
        );
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn image_set_rejects_distinct_wrappers_for_one_physical_image() {
        let device = Device::create(crate::driver::device::DeviceInfo::default()).unwrap();
        let image = Arc::new(
            Image::create(
                &device,
                ImageInfo::image_2d(
                    1,
                    1,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::SAMPLED,
                ),
            )
            .unwrap(),
        );
        let alias = Arc::new(unsafe { Image::from_raw(&device, image.handle, image.info) });
        let lhs = ImageSet::new([Arc::clone(&image)]).unwrap();
        let rhs = ImageSet::new([Arc::clone(&alias)]).unwrap();
        let mut graph = Graph::new();
        let lhs = graph.bind_resource(&lhs);
        let rhs = graph.bind_resource(&rhs);

        graph
            .begin_cmd()
            .resource_access(lhs, ImageAccessType::SampledRead)
            .resource_access(rhs, ImageAccessType::SampledRead)
            .record_cmd(|_| {})
            .end_cmd();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| graph.finalize()))
            .expect_err("distinct image wrappers should be rejected");
        assert!(
            panic_message(panic.as_ref()).contains("image sets contain incompatible image aliases")
        );
    }

    #[test]
    fn image_set_fixture_export_is_rejected() {
        let mut graph = Graph::new();
        let resource_set = graph.bind_resource(&empty_set());
        graph
            .begin_cmd()
            .resource_access(resource_set, ImageAccessType::SampledRead)
            .record_cmd(|_| {})
            .end_cmd();

        let err = graph
            .export_fixture("unused-resource-set.bin", "unused-resource-set.md")
            .unwrap_err();

        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    }

    #[cfg(feature = "checked")]
    #[test]
    #[should_panic(expected = "node belongs to a different graph")]
    fn acceleration_structure_set_resource_access_rejects_another_graph() {
        let mut source = Graph::new();
        let resource_set = source.bind_resource(&empty_acceleration_structure_set());
        let mut destination = Graph::new();

        destination
            .begin_cmd()
            .set_resource_access(resource_set, AccelerationStructureAccessType::BuildRead);
    }

    #[cfg(feature = "checked")]
    #[test]
    #[should_panic(expected = "node belongs to a different graph")]
    fn image_set_resource_access_rejects_another_graph() {
        let mut source = Graph::new();
        let resource_set = source.bind_resource(&empty_set());
        let mut destination = Graph::new();

        destination
            .begin_cmd()
            .set_resource_access(resource_set, ImageAccessType::SampledRead);
    }

    #[cfg(feature = "checked")]
    #[test]
    #[should_panic(expected = "node belongs to a different graph")]
    fn acceleration_structure_set_node_rejects_another_graph() {
        let mut source = Graph::new();
        let node = source.bind_resource(&empty_acceleration_structure_set());
        let destination = Graph::new();

        destination.resource(node);
    }

    #[cfg(feature = "checked")]
    #[test]
    #[should_panic(expected = "node belongs to a different graph")]
    fn image_set_node_rejects_another_graph() {
        let mut source = Graph::new();
        let node = source.bind_resource(&empty_set());
        let destination = Graph::new();

        destination.resource(node);
    }
}
