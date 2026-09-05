//! Opacity micromap resource types.
//!
//! Capture replay is intentionally not exposed. It requires a replayable backing-buffer device
//! address, which [`BufferInfo`] cannot currently request.

use {
    super::{Buffer, BufferInfo, DriverError, device::Device},
    crate::driver::accel_struct::DeviceOrHostAddress,
    ash::vk,
    derive_builder::Builder,
    log::warn,
    std::{
        ffi::c_void,
        fmt::{Debug, Formatter},
        ptr,
        thread::panicking,
    },
    vk_sync::AccessType,
};

#[cfg(feature = "parking_lot")]
use parking_lot::{Mutex, MutexGuard};

#[cfg(not(feature = "parking_lot"))]
use std::sync::{Mutex, MutexGuard};

const SERIALIZATION_ALIGNMENT: usize = 16;

pub(crate) fn micromap_sync_flags_for_access(
    access: AccessType,
) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    let info = vk_sync::get_access_info2(access);
    (info.stage_mask, info.access_mask)
}

fn map_vk_result(result: vk::Result) -> Result<(), DriverError> {
    match result {
        vk::Result::SUCCESS | vk::Result::OPERATION_NOT_DEFERRED_KHR => Ok(()),
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
            Err(DriverError::OutOfMemory)
        }
        vk::Result::ERROR_DEVICE_LOST | vk::Result::ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS => {
            Err(DriverError::InvalidData)
        }
        _ => Err(DriverError::Unsupported),
    }
}

/// Smart pointer handle to a Vulkan micromap and its backing buffer.
#[read_only::cast]
pub struct Micromap {
    accesses: Mutex<Vec<AccessType>>,

    /// The buffer containing the micromap storage.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub buffer: Buffer,

    /// The native Vulkan micromap handle.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub handle: vk::MicromapEXT,

    /// Information used to create this object.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub info: MicromapInfo,
}

impl Micromap {
    /// Creates a micromap and its backing buffer.
    #[profiling::function]
    pub fn create(device: &Device, info: impl Into<MicromapInfo>) -> Result<Self, DriverError> {
        debug_assert!(device.physical.vk_ext_opacity_micromap.is_some());

        let info = info.into();
        if info.micromap_type != vk::MicromapTypeEXT::OPACITY_MICROMAP {
            return Err(DriverError::Unsupported);
        }

        let buffer_info = if info.host_visible {
            BufferInfo::host_mem(info.size, vk::BufferUsageFlags::MICROMAP_STORAGE_EXT)
        } else {
            BufferInfo::device_mem(info.size, vk::BufferUsageFlags::MICROMAP_STORAGE_EXT)
        };
        let buffer = Buffer::create(device, buffer_info)?;
        let create_info = vk::MicromapCreateInfoEXT::default()
            .buffer(buffer.handle)
            .size(info.size)
            .ty(info.micromap_type);

        let ext = Device::expect_vk_ext_opacity_micromap(device);
        let mut handle = vk::MicromapEXT::null();

        let result = unsafe {
            (ext.fp().create_micromap_ext)(ext.device(), &create_info, ptr::null(), &mut handle)
        };

        map_vk_result(result).inspect_err(|err| warn!("unable to create micromap: {err}"))?;

        Ok(Self {
            accesses: Mutex::new(vec![AccessType::Nothing]),
            buffer,
            handle,
            info,
        })
    }

    /// Returns the storage and scratch sizes required for a build.
    #[profiling::function]
    pub fn size_of(device: &Device, info: &OpacityMicromapBuildInfo) -> MicromapSize {
        Self::size_of_build_type(
            device,
            info,
            vk::AccelerationStructureBuildTypeKHR::HOST_OR_DEVICE,
        )
    }

    pub(crate) fn size_of_build_type(
        device: &Device,
        info: &OpacityMicromapBuildInfo,
        build_type: vk::AccelerationStructureBuildTypeKHR,
    ) -> MicromapSize {
        let build_info = info.to_vk(vk::MicromapEXT::null());
        let mut size_info = vk::MicromapBuildSizesInfoEXT::default();

        let ext = Device::expect_vk_ext_opacity_micromap(device);

        unsafe {
            (ext.fp().get_micromap_build_sizes_ext)(
                ext.device(),
                build_type,
                &build_info,
                &mut size_info,
            );
        }

        MicromapSize {
            build_size: size_info.build_scratch_size,
            create_size: size_info.micromap_size,
            discardable: size_info.discardable == vk::TRUE,
        }
    }

    /// Builds this opacity micromap synchronously on the host.
    ///
    /// There is no micromap update mode; this always uses [`vk::BuildMicromapModeEXT::BUILD`].
    ///
    /// # Safety
    ///
    /// All addresses in `info` must be host pointers with Vulkan-required alignment, valid for the
    /// complete call. The triangle range must cover every strided entry described by the usage
    /// counts; each entry's format, subdivision level and data offset must reference initialized
    /// opacity data wholly within the readable input range. Usage counts must match the triangles'
    /// formats and subdivision levels. Scratch must be writable for the build
    /// size, and this micromap's backing storage must cover the create size, from a matching host or
    /// host-or-device size query for these flags and usage counts. Null scratch is allowed only when
    /// the required scratch size is zero. Scratch, destination storage and input ranges must not
    /// overlap, except that read-only inputs may alias each other.
    ///
    /// All memory must remain alive and bound for the call. This micromap must not be in GPU use;
    /// synchronize prior host/device accesses (including mapped-memory flushes/invalidations as
    /// needed), prevent concurrent accesses conflicting with these reads/writes, and synchronize
    /// subsequent device use. Counts, strides, formats, subdivision levels and flags must satisfy
    /// all `vkBuildMicromapsEXT` validity requirements and device limits.
    pub unsafe fn build_host(
        &mut self,
        info: &OpacityMicromapBuildInfo,
    ) -> Result<(), DriverError> {
        self.require_host_commands()?;
        let allow_null_scratch = matches!(info.scratch_data, DeviceOrHostAddress::HostAddress(address) if address.is_null())
            && Self::size_of_build_type(
                &self.buffer.device,
                info,
                vk::AccelerationStructureBuildTypeKHR::HOST,
            )
            .build_size
                == 0;
        if self.info.micromap_type != vk::MicromapTypeEXT::OPACITY_MICROMAP
            || !info.has_host_addresses(allow_null_scratch)
        {
            return Err(DriverError::InvalidData);
        }

        let build_info = info.to_vk(self.handle);

        let ext = Device::expect_vk_ext_opacity_micromap(&self.buffer.device);

        let result = unsafe {
            (ext.fp().build_micromaps_ext)(
                ext.device(),
                vk::DeferredOperationKHR::null(),
                1,
                &build_info,
            )
        };

        self.finish_host_write(result, "build micromap")
    }

    /// Clones `source` into this micromap synchronously on the host.
    ///
    /// # Safety
    ///
    /// `source` must have been successfully constructed, and this micromap's backing storage must
    /// be large enough for a clone of it. Neither micromap may be in use by a device operation.
    /// Their storage must not overlap and must remain alive and bound for the call. Synchronize
    /// prior host/device accesses and subsequent device use, including mapped-memory cache
    /// maintenance, and prevent concurrent host writes to the source or accesses to the destination.
    /// All `vkCopyMicromapEXT` validity requirements must be satisfied.
    pub unsafe fn clone_from_host(&mut self, source: &Self) -> Result<(), DriverError> {
        unsafe { self.copy_from_host(source, vk::CopyMicromapModeEXT::CLONE) }
    }

    /// Copies a compacted representation of `source` into this micromap on the host.
    ///
    /// # Safety
    ///
    /// `source` must have been successfully built with
    /// [`vk::BuildMicromapFlagsEXT::ALLOW_COMPACTION`], and this micromap's backing storage must be
    /// at least `source.compacted_size()` bytes. Neither micromap may be in use by a device
    /// operation.
    /// Their storage must not overlap and must remain alive and bound for the call. Synchronize
    /// prior host/device accesses and subsequent device use, including mapped-memory cache
    /// maintenance, and prevent concurrent host writes to the source or accesses to the destination.
    /// All `vkCopyMicromapEXT` validity requirements must be satisfied.
    pub unsafe fn compact_from_host(&mut self, source: &Self) -> Result<(), DriverError> {
        unsafe { self.copy_from_host(source, vk::CopyMicromapModeEXT::COMPACT) }
    }

    unsafe fn copy_from_host(
        &mut self,
        source: &Self,
        mode: vk::CopyMicromapModeEXT,
    ) -> Result<(), DriverError> {
        self.require_host_commands()?;
        source.require_host_commands()?;
        if self.buffer.device != source.buffer.device
            || self.info.micromap_type != source.info.micromap_type
        {
            return Err(DriverError::InvalidData);
        }

        let info = vk::CopyMicromapInfoEXT::default()
            .src(source.handle)
            .dst(self.handle)
            .mode(mode);

        let ext = Device::expect_vk_ext_opacity_micromap(&self.buffer.device);

        let result = unsafe {
            (ext.fp().copy_micromap_ext)(ext.device(), vk::DeferredOperationKHR::null(), &info)
        };

        map_vk_result(result).inspect_err(|err| warn!("unable to copy micromap: {err}"))?;
        source
            .swap_access(AccessType::MicromapBuildRead)
            .for_each(drop);
        self.swap_access(AccessType::MicromapBuildWrite)
            .for_each(drop);

        Ok(())
    }

    /// Serializes this micromap into host memory.
    ///
    /// `destination` must be at least [`Self::serialization_size`] bytes and its address must be
    /// aligned to 16 bytes as required by Vulkan.
    ///
    /// # Safety
    ///
    /// This micromap must have been successfully constructed and must not be in use by a device
    /// operation.
    /// Its bound storage must remain alive and must not overlap `destination`. Synchronize prior
    /// host/device writes, including mapped-memory cache maintenance, and prevent concurrent writes
    /// to this micromap or accesses to `destination` during the call. Synchronize subsequent device
    /// use of the output. All `vkCopyMicromapToMemoryEXT` validity requirements must be satisfied.
    pub unsafe fn serialize_host(&self, destination: &mut [u8]) -> Result<(), DriverError> {
        self.require_host_commands()?;
        let required = usize::try_from(unsafe { self.serialization_size()? })
            .map_err(|_| DriverError::OutOfMemory)?;
        if destination.len() < required
            || !(destination.as_ptr() as usize).is_multiple_of(SERIALIZATION_ALIGNMENT)
        {
            return Err(DriverError::InvalidData);
        }

        let info = vk::CopyMicromapToMemoryInfoEXT::default()
            .src(self.handle)
            .dst(vk::DeviceOrHostAddressKHR {
                host_address: destination.as_mut_ptr().cast::<c_void>(),
            })
            .mode(vk::CopyMicromapModeEXT::SERIALIZE);

        let ext = Device::expect_vk_ext_opacity_micromap(&self.buffer.device);

        let result = unsafe {
            (ext.fp().copy_micromap_to_memory_ext)(
                ext.device(),
                vk::DeferredOperationKHR::null(),
                &info,
            )
        };

        map_vk_result(result).inspect_err(|err| warn!("unable to serialize micromap: {err}"))?;
        self.swap_access(AccessType::MicromapBuildRead)
            .for_each(drop);

        Ok(())
    }

    /// Deserializes a micromap from host memory into this object.
    ///
    /// # Safety
    ///
    /// `source` must contain a complete serialized micromap representation. Vulkan does not expose
    /// a way to determine its required length before reading it. Its version data must have been
    /// reported compatible by [`Self::compatibility`], this micromap's backing storage must be large
    /// enough for the deserialized object, and this micromap must not be in device use.
    /// The representation must be unmodified output of Vulkan serialization. Source and destination
    /// storage must not overlap and must remain alive (and backing storage bound) for the call.
    /// Synchronize prior host/device accesses, including mapped-memory cache maintenance, prevent
    /// concurrent source writes or destination accesses, and synchronize subsequent device use.
    /// All `vkCopyMemoryToMicromapEXT` validity requirements must be satisfied.
    pub unsafe fn deserialize_host(&mut self, source: &[u8]) -> Result<(), DriverError> {
        self.require_host_commands()?;
        if source.is_empty() || !(source.as_ptr() as usize).is_multiple_of(SERIALIZATION_ALIGNMENT)
        {
            return Err(DriverError::InvalidData);
        }

        let info = vk::CopyMemoryToMicromapInfoEXT::default()
            .src(vk::DeviceOrHostAddressConstKHR {
                host_address: source.as_ptr().cast::<c_void>(),
            })
            .dst(self.handle)
            .mode(vk::CopyMicromapModeEXT::DESERIALIZE);

        let ext = Device::expect_vk_ext_opacity_micromap(&self.buffer.device);

        let result = unsafe {
            (ext.fp().copy_memory_to_micromap_ext)(
                ext.device(),
                vk::DeferredOperationKHR::null(),
                &info,
            )
        };

        self.finish_host_write(result, "deserialize micromap")
    }

    /// Returns the serialized representation size in bytes.
    ///
    /// # Safety
    ///
    /// This micromap must have been successfully constructed and must not be in device use.
    /// Its storage must remain alive and bound, prior writes must be made visible to the host, and
    /// concurrent host writes must be prevented for the call, as required by [`Self::property`].
    pub unsafe fn serialization_size(&self) -> Result<vk::DeviceSize, DriverError> {
        unsafe { self.property(vk::QueryType::MICROMAP_SERIALIZATION_SIZE_EXT) }
    }

    /// Returns the compacted micromap size in bytes for a source built with compaction enabled.
    ///
    /// # Safety
    ///
    /// This micromap must have been successfully built with
    /// [`vk::BuildMicromapFlagsEXT::ALLOW_COMPACTION`] and must not be in device use.
    /// Its storage must remain alive and bound, prior writes must be made visible to the host, and
    /// concurrent host writes must be prevented for the call, as required by [`Self::property`].
    pub unsafe fn compacted_size(&self) -> Result<vk::DeviceSize, DriverError> {
        unsafe { self.property(vk::QueryType::MICROMAP_COMPACTED_SIZE_EXT) }
    }

    /// Writes one 64-bit micromap property synchronously on the host.
    ///
    /// # Safety
    ///
    /// This micromap must have been successfully constructed and must not be in device use. If
    /// `query_type` is [`vk::QueryType::MICROMAP_COMPACTED_SIZE_EXT`], it must have been built with
    /// [`vk::BuildMicromapFlagsEXT::ALLOW_COMPACTION`].
    /// Its storage must remain alive and bound for the call. Synchronize prior host/device writes,
    /// including mapped-memory cache maintenance, and prevent concurrent host writes. All
    /// `vkWriteMicromapsPropertiesEXT` validity requirements must be satisfied.
    pub unsafe fn property(
        &self,
        query_type: vk::QueryType,
    ) -> Result<vk::DeviceSize, DriverError> {
        self.require_host_commands()?;
        if query_type != vk::QueryType::MICROMAP_SERIALIZATION_SIZE_EXT
            && query_type != vk::QueryType::MICROMAP_COMPACTED_SIZE_EXT
        {
            return Err(DriverError::InvalidData);
        }

        let mut value = 0_u64;
        let ext = Device::expect_vk_ext_opacity_micromap(&self.buffer.device);
        let result = unsafe {
            (ext.fp().write_micromaps_properties_ext)(
                ext.device(),
                1,
                &self.handle,
                query_type,
                size_of::<u64>(),
                (&mut value as *mut u64).cast(),
                size_of::<u64>(),
            )
        };

        map_vk_result(result)
            .inspect_err(|err| warn!("unable to query micromap property: {err}"))?;
        self.swap_access(AccessType::MicromapBuildRead)
            .for_each(drop);

        Ok(value)
    }

    /// Reports whether serialized micromap version data is compatible with this device.
    pub fn compatibility(
        device: &Device,
        version_data: &[u8; vk::UUID_SIZE * 2],
    ) -> vk::AccelerationStructureCompatibilityKHR {
        let info = vk::MicromapVersionInfoEXT::default().version_data(version_data);
        let ext = Device::expect_vk_ext_opacity_micromap(device);
        let mut compatibility = vk::AccelerationStructureCompatibilityKHR::default();

        unsafe {
            (ext.fp().get_device_micromap_compatibility_ext)(
                ext.device(),
                &info,
                &mut compatibility,
            );
        }

        compatibility
    }

    fn finish_host_write(&self, result: vk::Result, operation: &str) -> Result<(), DriverError> {
        map_vk_result(result).inspect_err(|err| warn!("unable to {operation}: {err}"))?;
        self.swap_access(AccessType::MicromapBuildWrite)
            .for_each(drop);
        Ok(())
    }

    fn require_host_commands(&self) -> Result<(), DriverError> {
        if !self.info.host_visible {
            return Err(DriverError::InvalidData);
        }

        let supported = self
            .buffer
            .device
            .physical
            .vk_ext_opacity_micromap
            .as_ref()
            .is_some_and(|ext| ext.features.micromap_host_commands);

        supported.then_some(()).ok_or(DriverError::Unsupported)
    }

    fn lock_accesses(&self) -> MutexGuard<'_, Vec<AccessType>> {
        let accesses = self.accesses.lock();

        #[cfg(not(feature = "parking_lot"))]
        let accesses = accesses.expect("poisoned micromap access lock");

        accesses
    }

    /// Records `next_access` and returns accesses requiring synchronization before it.
    #[profiling::function]
    pub(crate) fn swap_access(
        &self,
        next_access: AccessType,
    ) -> impl Iterator<Item = AccessType> + '_ {
        MicromapAccessIter::one(self.lock_accesses(), next_access)
    }

    pub(crate) fn swap_accesses(
        &self,
        next_accesses: &[AccessType],
    ) -> impl Iterator<Item = AccessType> + '_ {
        MicromapAccessIter::many(self.lock_accesses(), next_accesses)
    }

    /// Returns synchronization information for current micromap accesses.
    pub fn sync_info(&self) -> MicromapSyncInfo {
        MicromapSyncInfo::from_accesses(self.lock_accesses().iter().copied())
    }

    /// Sets the debugging name assigned to this micromap.
    pub fn set_debug_name(&self, name: impl AsRef<str>) {
        Device::try_set_debug_utils_object_name(&self.buffer.device, self.handle, &name);
        Device::try_set_private_data_object_name(
            &self.buffer.device,
            vk::ObjectType::MICROMAP_EXT,
            self.handle,
            &name,
        );
    }

    /// Sets a debugging name and returns this object.
    pub fn with_debug_name(self, name: impl AsRef<str>) -> Self {
        self.set_debug_name(name);
        self
    }
}

impl Debug for Micromap {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut result = f.debug_struct(stringify!(Micromap));
        if let Some(name) = Device::private_data_object_name(
            &self.buffer.device,
            vk::ObjectType::MICROMAP_EXT,
            self.handle,
        ) {
            result.field("debug_name", &name);
        }
        result.field("handle", &self.handle).finish_non_exhaustive()
    }
}

impl Drop for Micromap {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        Device::try_clear_private_data_object_name(
            &self.buffer.device,
            vk::ObjectType::MICROMAP_EXT,
            self.handle,
        );
        let ext = Device::expect_vk_ext_opacity_micromap(&self.buffer.device);
        unsafe {
            (ext.fp().destroy_micromap_ext)(ext.device(), self.handle, ptr::null());
        }
    }
}

impl Eq for Micromap {}

impl PartialEq for Micromap {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

/// Information used to create a [`Micromap`].
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct MicromapInfo {
    /// Uses a host-visible backing allocation when true.
    #[builder(default)]
    pub host_visible: bool,

    /// Type of micromap.
    #[builder(default = "vk::MicromapTypeEXT::OPACITY_MICROMAP")]
    pub micromap_type: vk::MicromapTypeEXT,

    /// Size of the backing storage.
    #[builder(default)]
    pub size: vk::DeviceSize,
}

impl MicromapInfo {
    /// Creates a default builder.
    pub fn builder() -> MicromapInfoBuilder {
        MicromapInfoBuilder::default()
    }

    /// Creates an opacity micromap backed by device-local memory.
    pub const fn device_mem(size: vk::DeviceSize) -> Self {
        Self {
            host_visible: false,
            micromap_type: vk::MicromapTypeEXT::OPACITY_MICROMAP,
            size,
        }
    }

    /// Creates an opacity micromap backed by host-visible memory.
    pub const fn host_mem(size: vk::DeviceSize) -> Self {
        Self {
            host_visible: true,
            micromap_type: vk::MicromapTypeEXT::OPACITY_MICROMAP,
            size,
        }
    }

    /// Converts this value into a builder.
    pub fn into_builder(self) -> MicromapInfoBuilder {
        MicromapInfoBuilder {
            host_visible: Some(self.host_visible),
            micromap_type: Some(self.micromap_type),
            size: Some(self.size),
        }
    }
}

impl From<MicromapInfoBuilder> for MicromapInfo {
    fn from(info: MicromapInfoBuilder) -> Self {
        info.build()
    }
}

impl MicromapInfoBuilder {
    /// Builds a micromap description.
    pub fn build(self) -> MicromapInfo {
        self.fallible_build().expect("all fields have defaults")
    }
}

/// Size requirements returned by [`Micromap::size_of`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MicromapSize {
    /// Required build scratch size.
    pub build_size: vk::DeviceSize,

    /// Required micromap storage size.
    pub create_size: vk::DeviceSize,

    /// Whether the micromap may be destroyed after an acceleration-structure build or update.
    ///
    /// When false, the acceleration structure may reference the micromap's storage, so the
    /// micromap must remain alive until ray traversal has concluded. When true, the information
    /// is copied into the acceleration structure and the micromap may be destroyed once that
    /// build or update completes. Micromap-build input and scratch memory may be released after
    /// the micromap build completes, independently of this flag.
    pub discardable: bool,
}

/// Typed opacity micromap usage count.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct OpacityMicromapUsage {
    /// Number of matching micromap triangles.
    pub count: u32,

    /// Opacity format for these triangles.
    pub format: vk::OpacityMicromapFormatEXT,

    /// Subdivision level for these triangles.
    pub subdivision_level: u32,
}

impl OpacityMicromapUsage {
    /// Creates an opacity micromap usage count.
    pub const fn new(
        count: u32,
        subdivision_level: u32,
        format: vk::OpacityMicromapFormatEXT,
    ) -> Self {
        Self {
            count,
            format,
            subdivision_level,
        }
    }
}

impl From<OpacityMicromapUsage> for vk::MicromapUsageEXT {
    fn from(value: OpacityMicromapUsage) -> Self {
        Self::default()
            .count(value.count)
            .subdivision_level(value.subdivision_level)
            .format(value.format.as_raw() as u32)
    }
}

/// Build parameters for an opacity micromap.
///
/// Usage counts are owned so pointers passed to Vulkan remain valid for the complete call.
#[derive(Clone, Debug)]
pub struct OpacityMicromapBuildInfo {
    /// Address containing encoded opacity data.
    pub data: DeviceOrHostAddress,

    /// Additional build behavior.
    pub flags: vk::BuildMicromapFlagsEXT,

    /// Build scratch address.
    pub scratch_data: DeviceOrHostAddress,

    /// Address of the micromap triangle array.
    pub triangle_array: DeviceOrHostAddress,

    /// Byte stride between triangle entries.
    pub triangle_array_stride: vk::DeviceSize,

    usage_counts: Box<[vk::MicromapUsageEXT]>,
}

impl OpacityMicromapBuildInfo {
    /// Creates build parameters with owned usage counts and null addresses.
    pub fn new(usage_counts: impl IntoIterator<Item = OpacityMicromapUsage>) -> Self {
        let usage_counts = usage_counts.into_iter().map(Into::into).collect();
        Self {
            data: DeviceOrHostAddress::DeviceAddress(0),
            flags: vk::BuildMicromapFlagsEXT::empty(),
            scratch_data: DeviceOrHostAddress::DeviceAddress(0),
            triangle_array: DeviceOrHostAddress::DeviceAddress(0),
            triangle_array_stride: size_of::<vk::MicromapTriangleEXT>() as _,
            usage_counts,
        }
    }

    /// Sets the encoded opacity-data address.
    pub fn data(mut self, data: impl Into<DeviceOrHostAddress>) -> Self {
        self.data = data.into();
        self
    }

    /// Sets build flags.
    pub fn flags(mut self, flags: vk::BuildMicromapFlagsEXT) -> Self {
        self.flags = flags;
        self
    }

    /// Sets the scratch address.
    pub fn scratch_data(mut self, scratch_data: impl Into<DeviceOrHostAddress>) -> Self {
        self.scratch_data = scratch_data.into();
        self
    }

    /// Sets the triangle array address and stride.
    pub fn triangle_array(
        mut self,
        triangle_array: impl Into<DeviceOrHostAddress>,
        stride: vk::DeviceSize,
    ) -> Self {
        self.triangle_array = triangle_array.into();
        self.triangle_array_stride = stride;
        self
    }

    /// Returns the typed usage counts supplied at construction.
    pub fn usage_counts(&self) -> impl ExactSizeIterator<Item = OpacityMicromapUsage> + '_ {
        self.usage_counts.iter().map(|usage| OpacityMicromapUsage {
            count: usage.count,
            format: vk::OpacityMicromapFormatEXT::from_raw(usage.format as i32),
            subdivision_level: usage.subdivision_level,
        })
    }

    fn has_host_addresses(&self, allow_null_scratch: bool) -> bool {
        matches!(self.data, DeviceOrHostAddress::HostAddress(address) if !address.is_null())
            && matches!(self.scratch_data, DeviceOrHostAddress::HostAddress(address) if allow_null_scratch || !address.is_null())
            && matches!(self.triangle_array, DeviceOrHostAddress::HostAddress(address) if !address.is_null())
    }

    pub(crate) fn to_vk(&self, destination: vk::MicromapEXT) -> vk::MicromapBuildInfoEXT<'_> {
        vk::MicromapBuildInfoEXT::default()
            .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP)
            .flags(self.flags)
            .mode(vk::BuildMicromapModeEXT::BUILD)
            .dst_micromap(destination)
            .usage_counts(&self.usage_counts)
            .data(self.data.into())
            .scratch_data(self.scratch_data.into())
            .triangle_array(self.triangle_array.into())
            .triangle_array_stride(self.triangle_array_stride)
    }
}

/// Synchronization information for a micromap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MicromapSyncInfo {
    /// Synchronization2 access masks for those stages.
    pub access_mask: vk::AccessFlags2,

    /// Current exclusive queue-family ownership, when known.
    pub queue_family_index: Option<u32>,

    /// Synchronization2 pipeline stages accessing the micromap.
    pub stage_mask: vk::PipelineStageFlags2,
}

impl MicromapSyncInfo {
    fn from_accesses(accesses: impl IntoIterator<Item = AccessType>) -> Self {
        let mut stage_mask = vk::PipelineStageFlags2::empty();
        let mut access_mask = vk::AccessFlags2::empty();

        for access in accesses {
            let (stages, mask) = micromap_sync_flags_for_access(access);
            stage_mask |= stages;
            access_mask |= mask;
        }

        Self {
            access_mask,
            queue_family_index: None,
            stage_mask,
        }
    }
}

struct MicromapAccessIter<'a> {
    accesses: MutexGuard<'a, Vec<AccessType>>,
    index: usize,
    previous_len: usize,
}

impl<'a> MicromapAccessIter<'a> {
    fn one(mut accesses: MutexGuard<'a, Vec<AccessType>>, next: AccessType) -> Self {
        let previous_len = accesses.len();
        accesses.push(next);
        Self {
            accesses,
            index: 0,
            previous_len,
        }
    }

    fn many(mut accesses: MutexGuard<'a, Vec<AccessType>>, next_accesses: &[AccessType]) -> Self {
        if next_accesses.is_empty() {
            return Self::one(accesses, AccessType::Nothing);
        }

        if next_accesses.iter().copied().any(super::is_write_access) {
            let previous_len = accesses.len();
            for &next_access in next_accesses {
                if !accesses[previous_len..].contains(&next_access) {
                    accesses.push(next_access);
                }
            }

            return Self {
                accesses,
                index: 0,
                previous_len,
            };
        }

        if next_accesses
            .iter()
            .all(|next_access| accesses.contains(next_access))
        {
            return Self {
                accesses,
                index: 0,
                previous_len: 0,
            };
        }

        let previous_len = accesses.len();
        accesses.extend_from_within(..);
        for &next_access in next_accesses {
            if !accesses[previous_len..].contains(&next_access) {
                accesses.push(next_access);
            }
        }

        Self {
            accesses,
            index: 0,
            previous_len,
        }
    }
}

impl Iterator for MicromapAccessIter<'_> {
    type Item = AccessType;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.previous_len {
            return None;
        }

        let result = self.accesses[self.index];
        self.index += 1;

        Some(result)
    }
}

impl Drop for MicromapAccessIter<'_> {
    fn drop(&mut self) {
        self.accesses.drain(..self.previous_len);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn micromap_host_null_scratch_requires_zero_scratch_size() {
        let mut byte = 0_u8;
        let address = DeviceOrHostAddress::HostAddress((&mut byte as *mut u8).cast());
        let mut info = OpacityMicromapBuildInfo::new([])
            .data(address)
            .triangle_array(address, size_of::<vk::MicromapTriangleEXT>() as u64)
            .scratch_data(DeviceOrHostAddress::HostAddress(ptr::null_mut()));

        assert!(info.has_host_addresses(true));
        assert!(!info.has_host_addresses(false));
        info.scratch_data = address;
        assert!(info.has_host_addresses(false));
        info.scratch_data = DeviceOrHostAddress::DeviceAddress(0);
        assert!(!info.has_host_addresses(true));
        info.scratch_data = address;
        info.data = DeviceOrHostAddress::HostAddress(ptr::null_mut());
        assert!(!info.has_host_addresses(true));
        info.data = address;
        info.triangle_array = DeviceOrHostAddress::HostAddress(ptr::null_mut());
        assert!(!info.has_host_addresses(true));
    }

    #[test]
    fn info_memory_constructors_and_builder_round_trip() {
        let device = MicromapInfo::device_mem(128);
        let host = MicromapInfo::host_mem(256);

        assert!(!device.host_visible);
        assert!(host.host_visible);
        assert_eq!(device, device.into_builder().build());
        assert_eq!(MicromapInfo::builder().size(64).build().size, 64);
    }

    #[test]
    fn opacity_usage_is_typed_and_owned() {
        let usage = OpacityMicromapUsage::new(7, 3, vk::OpacityMicromapFormatEXT::TYPE_4_STATE);
        let info = OpacityMicromapBuildInfo::new([usage]);

        assert_eq!(info.usage_counts().collect::<Vec<_>>(), [usage]);

        let raw = info.to_vk(vk::MicromapEXT::null());

        assert_eq!(raw.mode, vk::BuildMicromapModeEXT::BUILD);
        assert_eq!(raw.usage_counts_count, 1);
        assert!(!raw.p_usage_counts.is_null());
        assert!(raw.pp_usage_counts.is_null());
    }

    #[test]
    fn raw_results_map_consistently() {
        assert!(map_vk_result(vk::Result::SUCCESS).is_ok());
        assert!(matches!(
            map_vk_result(vk::Result::ERROR_OUT_OF_HOST_MEMORY),
            Err(DriverError::OutOfMemory)
        ));
        assert!(matches!(
            map_vk_result(vk::Result::ERROR_DEVICE_LOST),
            Err(DriverError::InvalidData)
        ));
        assert!(matches!(
            map_vk_result(vk::Result::ERROR_FEATURE_NOT_PRESENT),
            Err(DriverError::Unsupported)
        ));
    }

    #[test]
    fn sync_info_uses_micromap_sync2_bits() {
        let info = MicromapSyncInfo::from_accesses([
            AccessType::MicromapBuildRead,
            AccessType::MicromapBuildWrite,
        ]);

        assert_eq!(info.stage_mask, vk::PipelineStageFlags2::MICROMAP_BUILD_EXT);
        assert!(
            info.access_mask
                .contains(vk::AccessFlags2::MICROMAP_READ_EXT)
        );
        assert!(
            info.access_mask
                .contains(vk::AccessFlags2::MICROMAP_WRITE_EXT)
        );
    }

    #[test]
    fn micromap_buffer_and_input_accesses_use_distinct_sync2_masks() {
        for (access, expected) in [
            (
                AccessType::MicromapBuildInputRead,
                vk::AccessFlags2::SHADER_READ,
            ),
            (
                AccessType::MicromapBuildBufferRead,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            (
                AccessType::MicromapBuildBufferWrite,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
        ] {
            let (stage, mask) = micromap_sync_flags_for_access(access);

            assert_eq!(stage, vk::PipelineStageFlags2::MICROMAP_BUILD_EXT);
            assert_eq!(mask, expected);
        }
    }

    #[test]
    fn access_tracking_elides_repeated_reads_and_retains_reads_until_write() {
        let accesses = Mutex::new(vec![AccessType::Nothing]);
        let read = AccessType::MicromapBuildRead;
        let other_read = AccessType::AccelerationStructureBuildMicromapRead;
        let write = AccessType::MicromapBuildWrite;

        assert_eq!(
            MicromapAccessIter::many(lock_test_accesses(&accesses), &[read]).collect::<Vec<_>>(),
            [AccessType::Nothing]
        );
        assert!(
            MicromapAccessIter::many(lock_test_accesses(&accesses), &[read])
                .collect::<Vec<_>>()
                .is_empty()
        );
        assert_eq!(
            MicromapAccessIter::many(lock_test_accesses(&accesses), &[other_read])
                .collect::<Vec<_>>(),
            [AccessType::Nothing, read]
        );
        assert_eq!(
            MicromapAccessIter::many(lock_test_accesses(&accesses), &[write]).collect::<Vec<_>>(),
            [AccessType::Nothing, read, other_read]
        );
    }

    #[test]
    fn micromap_access_one_and_empty_many_replace_state_on_early_drop() {
        let read = AccessType::MicromapBuildRead;
        let write = AccessType::MicromapBuildWrite;
        let accesses = Mutex::new(vec![
            read,
            AccessType::AccelerationStructureBuildMicromapRead,
        ]);

        let mut iter = MicromapAccessIter::one(lock_test_accesses(&accesses), write);
        assert_eq!(iter.next(), Some(read));
        drop(iter);
        assert_eq!(*lock_test_accesses(&accesses), [write]);

        drop(MicromapAccessIter::many(lock_test_accesses(&accesses), &[]));
        assert_eq!(*lock_test_accesses(&accesses), [AccessType::Nothing]);
    }

    fn lock_test_accesses(accesses: &Mutex<Vec<AccessType>>) -> MutexGuard<'_, Vec<AccessType>> {
        let accesses = accesses.lock();

        #[cfg(not(feature = "parking_lot"))]
        let accesses = accesses.expect("poisoned test access lock");

        accesses
    }
}
