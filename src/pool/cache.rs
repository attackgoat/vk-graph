//! Pool wrapper which enables memory-efficient resource caching.

use {
    super::{
        BufferHostMappingCompatibility, Lease, Pool, compatible_buffer_info, compatible_image_info,
    },
    crate::driver::{
        DriverError,
        accel_struct::{AccelerationStructure, AccelerationStructureInfo},
        buffer::{Buffer, BufferInfo},
        image::{Image, ImageInfo},
    },
    log::debug,
    std::{
        collections::HashMap,
        hash::Hash,
        ops::{Deref, DerefMut},
        sync::{Arc, Weak},
    },
};

#[derive(Default)]
struct AliasSet {
    accel_structs: Vec<(
        AccelerationStructureInfo,
        Weak<Lease<AccelerationStructure>>,
    )>,
    buffers: Vec<(BufferInfo, Weak<Lease<Buffer>>)>,
    images: Vec<(ImageInfo, Weak<Lease<Image>>)>,
}

/// A memory-efficient resource cache for any [`Pool`] type.
///
/// Use [`Cache::tag`] to create a tag-scoped view that caches resources independently from other
/// tags. Untagged access still behaves like the default cache wrapper.
///
/// # Examples
///
/// ```no_run
/// # use vk_graph::driver::device::{Device, DeviceInfo};
/// # use vk_graph::driver::image::ImageInfo;
/// # use vk_graph::pool::cache::Cache;
/// # use vk_graph::pool::hash::HashPool;
/// # fn main() {
/// # let device = Device::create(DeviceInfo::default()).unwrap();
/// # let mut cache = Cache::new(HashPool::new(&device));
/// let mut shadow = cache.tag("shadow");
/// let image = shadow
///     .resource(ImageInfo::image_2d(
///         32,
///         32,
///         ash::vk::Format::R8G8B8A8_UNORM,
///         ash::vk::ImageUsageFlags::SAMPLED,
///     ))
///     .unwrap();
/// # let _ = image;
/// # }
/// ```
pub struct Cache<T, Tag = ()> {
    aliases: HashMap<Tag, AliasSet>,
    pool: T,
}

/// A tag-scoped cache view.
pub struct TaggedCache<'a, T, Tag> {
    cache: &'a mut Cache<T, Tag>,
    tag: Tag,
}

impl<T, Tag> Cache<T, Tag>
where
    Tag: Eq + Hash,
{
    /// Creates a new cache wrapper over the given pool.
    pub fn new(pool: T) -> Self {
        Self {
            aliases: Default::default(),
            pool,
        }
    }

    /// Returns a tag-scoped cache view.
    pub fn tag(&mut self, tag: Tag) -> TaggedCache<'_, T, Tag> {
        TaggedCache { cache: self, tag }
    }

    fn alias_set(&mut self, tag: Tag) -> &mut AliasSet {
        self.aliases.entry(tag).or_default()
    }

    fn resource_accel_struct_tagged(
        &mut self,
        tag: Tag,
        info: AccelerationStructureInfo,
    ) -> Result<Arc<Lease<AccelerationStructure>>, DriverError>
    where
        Tag: Clone,
        T: Pool<AccelerationStructureInfo, AccelerationStructure>,
    {
        let mut result = None;

        {
            let state = self.alias_set(tag.clone());
            state
                .accel_structs
                .retain(|(_, item)| item.strong_count() > 0);

            profiling::scope!("check aliases");

            for (item_info, item) in &state.accel_structs {
                if item_info.ty == info.ty
                    && item_info.size >= info.size
                    && let Some(item) = item.upgrade()
                {
                    result = Some(item);
                    break;
                }
            }
        }

        if let Some(item) = result {
            return Ok(item);
        }

        debug!("Leasing new {}", stringify!(AccelerationStructure));

        let item = Arc::new(self.pool.resource(info)?);
        self.alias_set(tag)
            .accel_structs
            .push((info, Arc::downgrade(&item)));

        Ok(item)
    }

    fn resource_buffer_tagged(
        &mut self,
        tag: Tag,
        info: BufferInfo,
    ) -> Result<Arc<Lease<Buffer>>, DriverError>
    where
        Tag: Clone,
        T: Pool<BufferInfo, Buffer>,
    {
        let mut result = None;

        {
            let state = self.alias_set(tag.clone());
            state.buffers.retain(|(_, item)| item.strong_count() > 0);

            profiling::scope!("check aliases");

            for (item_info, item) in &state.buffers {
                if compatible_buffer_info(item_info, &info, BufferHostMappingCompatibility::Exact)
                    && let Some(item) = item.upgrade()
                {
                    result = Some(item);
                    break;
                }
            }
        }

        if let Some(item) = result {
            return Ok(item);
        }

        debug!("Leasing new {}", stringify!(Buffer));

        let item = Arc::new(self.pool.resource(info)?);
        self.alias_set(tag)
            .buffers
            .push((info, Arc::downgrade(&item)));

        Ok(item)
    }

    fn resource_image_tagged(
        &mut self,
        tag: Tag,
        info: ImageInfo,
    ) -> Result<Arc<Lease<Image>>, DriverError>
    where
        Tag: Clone,
        T: Pool<ImageInfo, Image>,
    {
        let mut result = None;

        {
            let state = self.alias_set(tag.clone());
            state.images.retain(|(_, item)| item.strong_count() > 0);

            profiling::scope!("check aliases");

            for (item_info, item) in &state.images {
                if compatible_image_info(item_info, &info)
                    && let Some(item) = item.upgrade()
                {
                    result = Some(item);
                    break;
                }
            }
        }

        if let Some(item) = result {
            return Ok(item);
        }

        debug!("Leasing new {}", stringify!(Image));

        let item = Arc::new(self.pool.resource(info)?);
        self.alias_set(tag)
            .images
            .push((info, Arc::downgrade(&item)));

        Ok(item)
    }
}

impl<T> Cache<T, ()>
where
    T: Pool<AccelerationStructureInfo, AccelerationStructure>
        + Pool<BufferInfo, Buffer>
        + Pool<ImageInfo, Image>,
{
    /// Alias an acceleration structure using the default tag.
    pub fn accel_struct(
        &mut self,
        info: AccelerationStructureInfo,
    ) -> Result<Arc<Lease<AccelerationStructure>>, DriverError> {
        self.resource_accel_struct_tagged((), info)
    }

    /// Alias a buffer using the default tag.
    pub fn buffer(&mut self, info: BufferInfo) -> Result<Arc<Lease<Buffer>>, DriverError> {
        self.resource_buffer_tagged((), info)
    }

    /// Alias an image using the default tag.
    pub fn image(&mut self, info: ImageInfo) -> Result<Arc<Lease<Image>>, DriverError> {
        self.resource_image_tagged((), info)
    }
}

impl<'a, T, Tag> TaggedCache<'a, T, Tag>
where
    Tag: Eq + Hash + Clone,
{
    /// Alias a resource using this cache tag.
    pub fn resource<I>(&mut self, info: I) -> Result<Arc<Lease<I::Item>>, DriverError>
    where
        I: TaggedCacheResource<Tag>,
        T: Pool<I, I::Item>,
    {
        I::resource(self.cache, self.tag.clone(), info)
    }
}

#[allow(private_bounds)]
#[doc(hidden)]
pub trait TaggedCacheResource<Tag>: cache_private::TaggedCacheResourceSealed + Sized {
    type Item;

    fn resource<T>(
        cache: &mut Cache<T, Tag>,
        tag: Tag,
        info: Self,
    ) -> Result<Arc<Lease<Self::Item>>, DriverError>
    where
        Tag: Eq + Hash + Clone,
        T: Pool<Self, Self::Item>;
}

mod cache_private {
    pub trait TaggedCacheResourceSealed {}
}

impl cache_private::TaggedCacheResourceSealed for AccelerationStructureInfo {}

impl cache_private::TaggedCacheResourceSealed for BufferInfo {}

impl cache_private::TaggedCacheResourceSealed for ImageInfo {}

impl<Tag> TaggedCacheResource<Tag> for AccelerationStructureInfo
where
    Tag: Eq + Hash + Clone,
{
    type Item = AccelerationStructure;

    fn resource<T>(
        cache: &mut Cache<T, Tag>,
        tag: Tag,
        info: Self,
    ) -> Result<Arc<Lease<Self::Item>>, DriverError>
    where
        T: Pool<Self, Self::Item>,
    {
        cache.resource_accel_struct_tagged(tag, info)
    }
}

impl<Tag> TaggedCacheResource<Tag> for BufferInfo
where
    Tag: Eq + Hash + Clone,
{
    type Item = Buffer;

    fn resource<T>(
        cache: &mut Cache<T, Tag>,
        tag: Tag,
        info: Self,
    ) -> Result<Arc<Lease<Self::Item>>, DriverError>
    where
        T: Pool<Self, Self::Item>,
    {
        cache.resource_buffer_tagged(tag, info)
    }
}

impl<Tag> TaggedCacheResource<Tag> for ImageInfo
where
    Tag: Eq + Hash + Clone,
{
    type Item = Image;

    fn resource<T>(
        cache: &mut Cache<T, Tag>,
        tag: Tag,
        info: Self,
    ) -> Result<Arc<Lease<Self::Item>>, DriverError>
    where
        T: Pool<Self, Self::Item>,
    {
        cache.resource_image_tagged(tag, info)
    }
}

impl<T, Tag> Deref for Cache<T, Tag> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.pool
    }
}

impl<T, Tag> DerefMut for Cache<T, Tag> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.pool
    }
}
