//! Acceleration structure resource types

use {
    super::{
        Buffer, BufferInfo, DriverError, device::Device, is_write_access,
        micromap::OpacityMicromapUsage, pipeline_stage_access_flags,
    },
    ash::vk,
    derive_builder::Builder,
    log::warn,
    smallvec::{SmallVec, smallvec},
    std::{
        fmt::{Debug, Formatter},
        marker::PhantomData,
        mem::size_of_val,
        thread::panicking,
    },
    vk_sync::AccessType,
};

type Accesses = SmallVec<[AccessType; 4]>;

#[cfg(feature = "parking_lot")]
use parking_lot::{Mutex, MutexGuard};

#[cfg(not(feature = "parking_lot"))]
use std::sync::{Mutex, MutexGuard};

fn accel_struct_sync_flags_for_access(
    access: AccessType,
) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    match access {
        AccessType::VertexShaderReadOther => (
            vk::PipelineStageFlags::VERTEX_SHADER,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        ),
        AccessType::TessellationControlShaderReadOther => (
            vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        ),
        AccessType::TessellationEvaluationShaderReadOther => (
            vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        ),
        AccessType::GeometryShaderReadOther => (
            vk::PipelineStageFlags::GEOMETRY_SHADER,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        ),
        AccessType::FragmentShaderReadOther => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        ),
        AccessType::ComputeShaderReadOther => (
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        ),
        AccessType::AnyShaderReadOther => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        ),
        AccessType::RayTracingShaderReadOther => (
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        ),
        AccessType::MeshShaderReadOther => (
            vk::PipelineStageFlags::MESH_SHADER_EXT,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        ),
        AccessType::TaskShaderReadOther => (
            vk::PipelineStageFlags::TASK_SHADER_EXT,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        ),
        _ => pipeline_stage_access_flags(access),
    }
}

fn validate_build_size_counts(geometry_count: usize, primitive_count_len: usize) {
    assert_eq!(
        geometry_count, primitive_count_len,
        "one maximum primitive count is required per geometry"
    );
    u32::try_from(geometry_count).expect("geometry count exceeds u32::MAX");
}

/// Smart pointer handle to an [acceleration structure] object.
///
/// Also contains the backing buffer and information about the object.
///
/// ```no_run
/// # use ash::vk;
/// # use vk_graph::driver::DriverError;
/// # use vk_graph::driver::device::Device;
/// # use vk_graph::driver::accel_struct::{AccelerationStructure, AccelerationStructureInfo};
/// # fn create(device: &Device, storage_size: vk::DeviceSize) -> Result<(), DriverError> {
/// let info = AccelerationStructureInfo::blas(storage_size);
/// let accel_struct = AccelerationStructure::create(device, info)?;
/// let addr = accel_struct.device_address();
///
/// assert_eq!(accel_struct.info, info);
/// assert_ne!(accel_struct.handle, vk::AccelerationStructureKHR::null());
/// # Ok(()) }
/// ```
///
/// See [`VkAccelerationStructureKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureKHR.html).
#[read_only::cast]
pub struct AccelerationStructure {
    // TODO: Replace with single atomicu8
    accesses: Mutex<Accesses>,

    /// The native Vulkan resource handle of the buffer which supports this acceleration structure.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub buffer: Buffer,

    /// The native Vulkan resource handle of this acceleration structure.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub handle: vk::AccelerationStructureKHR,

    /// Information used to create this object.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub info: AccelerationStructureInfo,
}

impl AccelerationStructure {
    /// Creates a new acceleration structure on the given device.
    ///
    /// ```no_run
    /// # use vk_graph::driver::{DriverError, device::Device};
    /// # use vk_graph::driver::accel_struct::{AccelerationStructure, AccelerationStructureInfo};
    /// # fn create(device: &Device, storage_size: u64) -> Result<(), DriverError> {
    /// // Use acceleration_structure_size from a matching build_sizes query.
    /// let info = AccelerationStructureInfo::blas(storage_size);
    /// let accel_struct = AccelerationStructure::create(device, info)?;
    /// assert_eq!(accel_struct.info.size, storage_size);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn create(
        device: &Device,
        info: impl Into<AccelerationStructureInfo>,
    ) -> Result<Self, DriverError> {
        debug_assert!(device.physical.vk_khr_acceleration_structure.is_some());

        let info = info.into();

        let buffer = Buffer::create(
            device,
            BufferInfo::device_mem(
                info.size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            ),
        )?;

        let handle = {
            let create_info = vk::AccelerationStructureCreateInfoKHR::default()
                .ty(info.acceleration_structure_type)
                .buffer(buffer.handle)
                .size(info.size);

            let khr_acceleration_structure = Device::expect_vk_khr_acceleration_structure(device);

            unsafe {
                khr_acceleration_structure
                    .create_acceleration_structure(&create_info, None)
                    .map_err(|err| {
                        warn!("unable to create acceleration structure: {err}");

                        match err {
                            vk::Result::ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS => {
                                warn!(
                                    "invalid acceleration structure opaque capture address: {err}"
                                );
                                DriverError::InvalidData
                            }
                            vk::Result::ERROR_OUT_OF_HOST_MEMORY => DriverError::OutOfMemory,
                            _ => {
                                warn!("unsupported acceleration structure creation: {err}");
                                DriverError::Unsupported
                            }
                        }
                    })?
            }
        };

        Ok(Self {
            accesses: Mutex::new(smallvec![AccessType::Nothing]),
            buffer,
            handle,
            info,
        })
    }

    /// Returns storage and scratch sizes for the given build type and geometry limits.
    ///
    /// Input device addresses are not dereferenced by the query. `transform_data` must still be
    /// nonzero when a transform will be used, and opacity micromap usage metadata must
    /// describe the intended build. Vulkan's geometry, flag and device-limit requirements apply.
    ///
    /// # Safety
    /// Every non-null attached micromap handle must be valid, belong to `device`, and
    /// remain alive through the call. All Vulkan build-size query validity requirements apply.
    ///
    /// # Panics
    /// Panics if the geometry and primitive-count lengths differ or any Vulkan array
    /// length exceeds `u32::MAX`.
    ///
    /// ```no_run
    /// # use ash::vk;
    /// # use vk_graph::driver::{device::Device, accel_struct::*};
    /// # fn query(device: &Device) {
    /// let geometries = [AccelerationStructureGeometry::opaque(
    ///     AccelerationStructureGeometryData::triangles(
    ///         0, vk::IndexType::UINT32, 2, 0, 0, vk::Format::R32G32B32_SFLOAT, 12,
    ///     ),
    /// )];
    /// // SAFETY: No micromap handles are attached; the query inputs satisfy Vulkan requirements.
    /// let sizes = unsafe { AccelerationStructure::build_sizes(
    ///     device, vk::AccelerationStructureBuildTypeKHR::DEVICE,
    ///     vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
    ///     vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
    ///     &geometries, &[1],
    /// ) };
    /// let info = AccelerationStructureInfo::blas(sizes.acceleration_structure_size);
    /// # }
    /// ```
    ///
    /// Querying sizes requires an `unsafe` block:
    ///
    /// ```compile_fail,E0133
    /// # use ash::vk;
    /// # use vk_graph::driver::{device::Device, accel_struct::*};
    /// # fn query(device: &Device, geometries: &[AccelerationStructureGeometry<'_>]) {
    /// let sizes = AccelerationStructure::build_sizes(
    ///     device, vk::AccelerationStructureBuildTypeKHR::DEVICE,
    ///     vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
    ///     vk::BuildAccelerationStructureFlagsKHR::empty(), geometries, &[1],
    /// );
    /// # }
    /// ```
    #[profiling::function]
    pub unsafe fn build_sizes(
        device: &Device,
        build_type: vk::AccelerationStructureBuildTypeKHR,
        acceleration_structure_type: vk::AccelerationStructureTypeKHR,
        flags: vk::BuildAccelerationStructureFlagsKHR,
        geometries: &[AccelerationStructureGeometry<'_>],
        max_primitive_counts: &[u32],
    ) -> AccelerationStructureBuildSizes {
        validate_build_size_counts(geometries.len(), max_primitive_counts.len());
        let geometries = AccelerationStructureGeometryMarshaler::new(geometries);

        let vk_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(acceleration_structure_type)
            .flags(flags)
            .geometries(geometries.geometries());
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        let khr_acceleration_structure = Device::expect_vk_khr_acceleration_structure(device);

        unsafe {
            khr_acceleration_structure.get_acceleration_structure_build_sizes(
                build_type,
                &vk_info,
                max_primitive_counts,
                &mut sizes,
            );
        }

        AccelerationStructureBuildSizes {
            acceleration_structure_size: sizes.acceleration_structure_size,
            build_scratch_size: sizes.build_scratch_size,
            update_scratch_size: sizes.update_scratch_size,
        }
    }

    /// Returns the device address of this object.
    ///
    /// ```no_run
    /// # use vk_graph::driver::accel_struct::AccelerationStructure;
    /// # fn address(accel_struct: &AccelerationStructure) {
    /// let addr = accel_struct.device_address();
    /// assert_ne!(addr, 0);
    /// # }
    /// ```
    #[profiling::function]
    pub fn device_address(&self) -> vk::DeviceAddress {
        let khr_acceleration_structure =
            Device::expect_vk_khr_acceleration_structure(&self.buffer.device);

        unsafe {
            khr_acceleration_structure.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(self.handle),
            )
        }
    }

    /// Helper function which is used to prepare instance buffers.
    pub fn instance_slice(instances: &[vk::AccelerationStructureInstanceKHR]) -> &[u8] {
        use std::slice::from_raw_parts;

        unsafe { from_raw_parts(instances.as_ptr() as *const _, size_of_val(instances)) }
    }

    fn lock_accesses(&self) -> MutexGuard<'_, Accesses> {
        let accesses = self.accesses.lock();

        #[cfg(not(feature = "parking_lot"))]
        let accesses = accesses.expect("poisoned acceleration structure access lock");

        accesses
    }

    /// Sets the debugging name assigned to this acceleration structure.
    pub fn set_debug_name(&self, name: impl AsRef<str>) {
        Device::try_set_debug_utils_object_name(&self.buffer.device, self.handle, &name);
        Device::try_set_private_data_object_name(
            &self.buffer.device,
            vk::ObjectType::ACCELERATION_STRUCTURE_KHR,
            self.handle,
            &name,
        );
    }

    /// Keeps track of a `next_access` which affects this object.
    ///
    /// Returns previous accesses for which a pipeline barrier should be used to prevent data
    /// corruption.
    #[profiling::function]
    pub(crate) fn swap_access(
        &self,
        next_access: AccessType,
    ) -> impl Iterator<Item = AccessType> + '_ {
        AccessIter::one(self.lock_accesses(), next_access)
    }

    pub(crate) fn swap_accesses(
        &self,
        next_accesses: &[AccessType],
    ) -> impl Iterator<Item = AccessType> + '_ {
        AccessIter::many(self.lock_accesses(), next_accesses)
    }

    /// Returns synchronization information for the acceleration structure's current accesses.
    pub fn sync_info(&self) -> AccelerationStructureSyncInfo {
        AccelerationStructureSyncInfo::from_accesses(self.lock_accesses().iter().copied())
    }

    /// Sets the debugging name assigned to this acceleration structure.
    pub fn with_debug_name(self, name: impl AsRef<str>) -> Self {
        self.set_debug_name(name);

        self
    }
}

impl Debug for AccelerationStructure {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut res = f.debug_struct(stringify!(AccelerationStructure));

        if let Some(debug_name) = &Device::private_data_object_name(
            &self.buffer.device,
            vk::ObjectType::ACCELERATION_STRUCTURE_KHR,
            self.handle,
        ) {
            res.field("debug_name", debug_name);
        }

        res.field("handle", &self.handle).finish_non_exhaustive()
    }
}

impl Drop for AccelerationStructure {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        Device::try_clear_private_data_object_name(
            &self.buffer.device,
            vk::ObjectType::ACCELERATION_STRUCTURE_KHR,
            self.handle,
        );

        let khr_acceleration_structure =
            Device::expect_vk_khr_acceleration_structure(&self.buffer.device);

        unsafe {
            khr_acceleration_structure.destroy_acceleration_structure(self.handle, None);
        }
    }
}

impl Eq for AccelerationStructure {}

impl PartialEq for AccelerationStructure {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

/// Size requirements returned by [`AccelerationStructure::build_sizes`].
#[derive(Clone, Copy, Debug)]
pub struct AccelerationStructureBuildSizes {
    /// Required storage size in bytes. Use as `size` in [`AccelerationStructureInfo`] when
    /// calling [`AccelerationStructure::create`].
    pub acceleration_structure_size: vk::DeviceSize,

    /// Required scratch size in bytes for
    /// [`CommandRef::build_acceleration_structures`](crate::cmd::CommandRef::build_acceleration_structures)
    /// with [`vk::BuildAccelerationStructureModeKHR::BUILD`].
    pub build_scratch_size: vk::DeviceSize,

    /// Required scratch size in bytes for
    /// [`CommandRef::build_acceleration_structures`](crate::cmd::CommandRef::build_acceleration_structures)
    /// with [`vk::BuildAccelerationStructureModeKHR::UPDATE`].
    pub update_scratch_size: vk::DeviceSize,
}

/// Geometry to be built into an acceleration structure.
///
/// Metadata is borrowed; primitive counts are supplied separately for each query or build.
///
/// See [`VkAccelerationStructureGeometryKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureGeometryKHR.html).
#[derive(Clone, Copy, Debug)]
pub struct AccelerationStructureGeometry<'a> {
    /// Describes additional properties of how the geometry should be built.
    pub flags: vk::GeometryFlagsKHR,

    /// Specifies acceleration structure geometry data.
    pub geometry: AccelerationStructureGeometryData<'a>,
}

impl<'a> AccelerationStructureGeometry<'a> {
    /// Creates a new acceleration structure geometry instance.
    pub fn new(geometry: AccelerationStructureGeometryData<'a>) -> Self {
        let flags = Default::default();

        Self { flags, geometry }
    }

    /// Creates a new acceleration structure geometry instance with the
    /// [vk::GeometryFlagsKHR::OPAQUE] flag set.
    pub fn opaque(geometry: AccelerationStructureGeometryData<'a>) -> Self {
        Self::new(geometry).flags(vk::GeometryFlagsKHR::OPAQUE)
    }

    /// Sets the instance flags.
    pub fn flags(mut self, flags: vk::GeometryFlagsKHR) -> Self {
        self.flags = flags;

        self
    }
}

/// Specifies acceleration structure geometry data.
///
/// See [`VkAccelerationStructureGeometryKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureGeometryKHR.html).
#[derive(Clone, Copy, Debug)]
pub enum AccelerationStructureGeometryData<'a> {
    /// Axis-aligned bounding box geometry in a bottom-level acceleration structure.
    ///
    /// See [`VkAccelerationStructureGeometryAabbsDataKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureGeometryAabbsDataKHR.html).
    AABBs {
        /// Device address of [vk::AabbPositionsKHR] values describing each axis-aligned
        /// bounding box in the geometry.
        data: vk::DeviceAddress,

        /// Stride in bytes between each entry in data.
        ///
        /// The stride must be a multiple of `8`.
        stride: vk::DeviceSize,
    },

    /// Geometry consisting of instances of other acceleration structures.
    ///
    /// See [`VkAccelerationStructureGeometryInstancesDataKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureGeometryInstancesDataKHR.html).
    Instances {
        /// Whether `data` contains device addresses of instances rather than instance values.
        array_of_pointers: bool,

        /// Either the address of an array of device addresses referencing individual
        /// [`VkAccelerationStructureInstanceKHR`] values if `array_of_pointers` is `true`, or the
        /// address of an array of [`VkAccelerationStructureInstanceKHR`] values.
        ///
        /// Addresses and `VkAccelerationStructureInstanceKHR` values are tightly packed.
        ///
        /// [`VkAccelerationStructureInstanceKHR`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureInstanceKHR.html
        data: vk::DeviceAddress,
    },

    /// A triangle geometry in a bottom-level acceleration structure.
    ///
    /// See [`VkAccelerationStructureGeometryTrianglesDataKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureGeometryTrianglesDataKHR.html).
    Triangles(AccelerationStructureTriangles<'a>),
}

impl<'a> AccelerationStructureGeometryData<'a> {
    /// Specifies acceleration structure geometry data as AABBs.
    pub fn aabbs(data: vk::DeviceAddress, stride: vk::DeviceSize) -> Self {
        Self::AABBs { data, stride }
    }

    /// Creates instance geometry from an array of device addresses of instances.
    pub fn instance_pointers(data: vk::DeviceAddress) -> Self {
        Self::Instances {
            array_of_pointers: true,
            data,
        }
    }

    /// Creates instance geometry from an array of instance values.
    pub fn instances(data: vk::DeviceAddress) -> Self {
        Self::Instances {
            array_of_pointers: false,
            data,
        }
    }

    /// Specifies acceleration structure geometry data as triangles.
    pub fn triangles(
        index_data: vk::DeviceAddress,
        index_type: vk::IndexType,
        max_vertex: u32,
        transform_data: vk::DeviceAddress,
        vertex_data: vk::DeviceAddress,
        vertex_format: vk::Format,
        vertex_stride: vk::DeviceSize,
    ) -> Self {
        Self::Triangles(AccelerationStructureTriangles::new(
            index_data,
            index_type,
            max_vertex,
            transform_data,
            vertex_data,
            vertex_format,
            vertex_stride,
        ))
    }
}

impl<'a> From<AccelerationStructureTriangles<'a>> for AccelerationStructureGeometryData<'a> {
    fn from(value: AccelerationStructureTriangles<'a>) -> Self {
        Self::Triangles(value)
    }
}

/// Owns Vulkan geometry and extension records for the duration of one Vulkan call.
pub(crate) struct AccelerationStructureGeometryMarshaler<'a> {
    _opacity_micromaps: Vec<vk::AccelerationStructureTrianglesOpacityMicromapEXT<'static>>,
    _source: PhantomData<&'a AccelerationStructureGeometry<'a>>,
    _usage_counts: Vec<Vec<vk::MicromapUsageEXT>>,
    geometries: Vec<vk::AccelerationStructureGeometryKHR<'static>>,
}

impl<'a> AccelerationStructureGeometryMarshaler<'a> {
    pub(crate) fn new(
        geometries: impl IntoIterator<Item = &'a AccelerationStructureGeometry<'a>>,
    ) -> Self {
        let source = geometries.into_iter().collect::<Vec<_>>();
        let extension_count = source
            .iter()
            .filter(|geometry| {
                matches!(
                    &geometry.geometry,
                    AccelerationStructureGeometryData::Triangles(triangles)
                        if triangles.opacity_micromap.is_some()
                )
            })
            .count();
        let mut opacity_micromaps = Vec::with_capacity(extension_count);
        let mut usage_counts = Vec::with_capacity(extension_count);

        for geometry in &source {
            if let AccelerationStructureGeometryData::Triangles(triangles) = &geometry.geometry
                && let Some(attachment) = &triangles.opacity_micromap
            {
                let count = u32::try_from(attachment.usage_counts.len())
                    .expect("micromap usage count exceeds u32::MAX");
                let usage = attachment
                    .usage_counts
                    .iter()
                    .copied()
                    .map(Into::into)
                    .collect::<Vec<vk::MicromapUsageEXT>>();
                opacity_micromaps.push(vk::AccelerationStructureTrianglesOpacityMicromapEXT {
                    p_usage_counts: if usage.is_empty() {
                        std::ptr::null()
                    } else {
                        usage.as_ptr()
                    },
                    usage_counts_count: count,
                    ..vk::AccelerationStructureTrianglesOpacityMicromapEXT::default()
                        .index_type(attachment.index_type)
                        .index_buffer(vk::DeviceOrHostAddressConstKHR {
                            device_address: attachment.index_buffer,
                        })
                        .index_stride(attachment.index_stride)
                        .base_triangle(attachment.base_triangle)
                        .micromap(attachment.micromap)
                });
                usage_counts.push(usage);
            }
        }

        let mut geometries_out = Vec::with_capacity(source.len());
        let mut extension_index = 0;
        for geometry in &source {
            let (geometry_type, geometry_data) = match &geometry.geometry {
                AccelerationStructureGeometryData::AABBs { data, stride } => (
                    vk::GeometryTypeKHR::AABBS,
                    vk::AccelerationStructureGeometryDataKHR {
                        aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::default()
                            .data(vk::DeviceOrHostAddressConstKHR {
                                device_address: *data,
                            })
                            .stride(*stride),
                    },
                ),
                AccelerationStructureGeometryData::Instances {
                    array_of_pointers,
                    data,
                } => (
                    vk::GeometryTypeKHR::INSTANCES,
                    vk::AccelerationStructureGeometryDataKHR {
                        instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                            .data(vk::DeviceOrHostAddressConstKHR {
                                device_address: *data,
                            })
                            .array_of_pointers(*array_of_pointers),
                    },
                ),
                AccelerationStructureGeometryData::Triangles(triangles) => {
                    let mut vk_triangles =
                        vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                            .index_data(vk::DeviceOrHostAddressConstKHR {
                                device_address: triangles.index_data,
                            })
                            .index_type(triangles.index_type)
                            .max_vertex(triangles.max_vertex)
                            .transform_data(vk::DeviceOrHostAddressConstKHR {
                                device_address: triangles.transform_data,
                            })
                            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                                device_address: triangles.vertex_data,
                            })
                            .vertex_format(triangles.vertex_format)
                            .vertex_stride(triangles.vertex_stride);

                    if triangles.opacity_micromap.is_some() {
                        // Extension storage is fully allocated before taking pointers. It and
                        // the usage arrays remain unchanged; moving the marshaler preserves
                        // their heap allocations and pointer validity.
                        vk_triangles.p_next = (&opacity_micromaps[extension_index]
                            as *const vk::AccelerationStructureTrianglesOpacityMicromapEXT<'_>)
                            .cast();
                        extension_index += 1;
                    }

                    (
                        vk::GeometryTypeKHR::TRIANGLES,
                        vk::AccelerationStructureGeometryDataKHR {
                            triangles: vk_triangles,
                        },
                    )
                }
            };

            geometries_out.push(
                vk::AccelerationStructureGeometryKHR::default()
                    .flags(geometry.flags)
                    .geometry_type(geometry_type)
                    .geometry(geometry_data),
            );
        }

        debug_assert_eq!(opacity_micromaps.len(), extension_count);

        Self {
            _opacity_micromaps: opacity_micromaps,
            _source: PhantomData,
            _usage_counts: usage_counts,
            geometries: geometries_out,
        }
    }

    pub(crate) fn geometries(&self) -> &[vk::AccelerationStructureGeometryKHR<'_>] {
        &self.geometries
    }
}

/// Information used to create an [`AccelerationStructure`] instance.
///
/// See [`VkAccelerationStructureCreateInfoKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureCreateInfoKHR.html).
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct AccelerationStructureInfo {
    /// Type of acceleration structure.
    #[builder(default = "vk::AccelerationStructureTypeKHR::GENERIC")]
    pub acceleration_structure_type: vk::AccelerationStructureTypeKHR,

    /// The size of the backing buffer that will store the acceleration structure.
    ///
    /// Use [`AccelerationStructure::build_sizes`] to calculate this value.
    #[builder(default)]
    pub size: vk::DeviceSize,
}

impl AccelerationStructureInfo {
    /// Specifies a [`vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL`] acceleration structure of the
    /// given size.
    #[inline(always)]
    pub const fn blas(size: vk::DeviceSize) -> Self {
        Self {
            acceleration_structure_type: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            size,
        }
    }

    /// Creates a default `AccelerationStructureInfoBuilder`.
    pub fn builder() -> AccelerationStructureInfoBuilder {
        Default::default()
    }

    /// Specifies a [`vk::AccelerationStructureTypeKHR::TOP_LEVEL`] acceleration structure of the
    /// given size.
    #[inline(always)]
    pub const fn tlas(size: vk::DeviceSize) -> Self {
        Self {
            acceleration_structure_type: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            size,
        }
    }

    /// Converts an `AccelerationStructureInfo` into an `AccelerationStructureInfoBuilder`.
    pub fn into_builder(self) -> AccelerationStructureInfoBuilder {
        AccelerationStructureInfoBuilder {
            acceleration_structure_type: Some(self.acceleration_structure_type),
            size: Some(self.size),
        }
    }
}

impl From<AccelerationStructureInfoBuilder> for AccelerationStructureInfo {
    fn from(info: AccelerationStructureInfoBuilder) -> Self {
        info.build()
    }
}

impl From<AccelerationStructureInfo> for () {
    fn from(_: AccelerationStructureInfo) -> Self {}
}

impl AccelerationStructureInfoBuilder {
    /// Builds a new `AccelerationStructureInfo`.
    #[inline(always)]
    pub fn build(self) -> AccelerationStructureInfo {
        self.fallible_build().expect("all fields have defaults")
    }
}

/// Opacity micromap data attached to acceleration-structure triangles.
#[derive(Clone, Copy, Debug)]
pub struct AccelerationStructureOpacityMicromap<'a> {
    /// Offset added to non-special micromap indices. With `index_type` set to `NONE_KHR`,
    /// triangle `i` uses micromap triangle `base_triangle + i`.
    pub base_triangle: u32,

    /// Device address of micromap indices for the triangles.
    pub index_buffer: vk::DeviceAddress,

    /// Byte stride between micromap indices.
    pub index_stride: vk::DeviceSize,

    /// Type of each micromap index.
    pub index_type: vk::IndexType,

    /// Native opacity micromap handle.
    pub micromap: vk::MicromapEXT,

    /// Borrowed usage counts for the micromap triangles referenced by this geometry after
    /// index indirection and `base_triangle` are applied, excluding special micromap indices.
    pub usage_counts: &'a [OpacityMicromapUsage],
}

impl<'a> AccelerationStructureOpacityMicromap<'a> {
    /// Creates an opacity micromap attachment borrowing its usage counts.
    pub fn new(micromap: vk::MicromapEXT, usage_counts: &'a [OpacityMicromapUsage]) -> Self {
        Self {
            base_triangle: 0,
            index_buffer: 0,
            index_stride: 0,
            index_type: vk::IndexType::NONE_KHR,
            micromap,
            usage_counts,
        }
    }

    /// Sets the offset added to non-special micromap indices (or the implicit triangle index
    /// when `index_type` is `NONE_KHR`).
    pub fn base_triangle(mut self, base_triangle: u32) -> Self {
        self.base_triangle = base_triangle;

        self
    }

    /// Sets the micromap-index buffer address.
    pub fn index_buffer(mut self, index_buffer: vk::DeviceAddress) -> Self {
        self.index_buffer = index_buffer;

        self
    }

    /// Sets the byte stride between micromap-index elements.
    pub fn index_stride(mut self, index_stride: vk::DeviceSize) -> Self {
        self.index_stride = index_stride;

        self
    }

    /// Sets the micromap-index element type.
    pub fn index_type(mut self, index_type: vk::IndexType) -> Self {
        self.index_type = index_type;

        self
    }
}

/// Synchronization information for an acceleration structure.
#[derive(Clone, Copy, Debug)]
pub struct AccelerationStructureSyncInfo {
    /// Access types performed by those stages.
    pub access_mask: vk::AccessFlags,

    /// Current exclusive queue-family ownership, when relevant.
    pub queue_family_index: Option<u32>,

    /// Pipeline stages that access the acceleration structure.
    pub stage_mask: vk::PipelineStageFlags,
}

impl AccelerationStructureSyncInfo {
    fn from_accesses(accesses: impl IntoIterator<Item = AccessType>) -> Self {
        let mut stage_mask = vk::PipelineStageFlags::empty();
        let mut access_mask = vk::AccessFlags::empty();

        for access in accesses {
            let (stages, mask) = accel_struct_sync_flags_for_access(access);
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

/// Triangle geometry and its optional opacity micromap attachment.
#[derive(Clone, Copy, Debug)]
pub struct AccelerationStructureTriangles<'a> {
    /// Device address of index data, or zero for non-indexed geometry.
    pub index_data: vk::DeviceAddress,

    /// The [`VkIndexType`] of each index element.
    ///
    /// [`VkIndexType`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndexType.html
    pub index_type: vk::IndexType,

    /// Highest vertex index addressed by a build command using this geometry.
    pub max_vertex: u32,

    /// Optional opacity micromap attachment.
    pub opacity_micromap: Option<AccelerationStructureOpacityMicromap<'a>>,

    /// Device address of a [`VkTransformMatrixKHR`] transforming vertices from geometry space
    /// to acceleration structure space, or zero when absent.
    ///
    /// [`VkTransformMatrixKHR`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkTransformMatrixKHR.html
    pub transform_data: vk::DeviceAddress,

    /// Device address of vertex data for this geometry.
    pub vertex_data: vk::DeviceAddress,

    /// The [`VkFormat`] of each vertex element.
    ///
    /// [`VkFormat`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormat.html
    pub vertex_format: vk::Format,

    /// Stride in bytes between vertices.
    pub vertex_stride: vk::DeviceSize,
}

impl<'a> AccelerationStructureTriangles<'a> {
    /// Creates triangle geometry without an opacity micromap attachment.
    pub fn new(
        index_data: vk::DeviceAddress,
        index_type: vk::IndexType,
        max_vertex: u32,
        transform_data: vk::DeviceAddress,
        vertex_data: vk::DeviceAddress,
        vertex_format: vk::Format,
        vertex_stride: vk::DeviceSize,
    ) -> Self {
        Self {
            index_data,
            index_type,
            max_vertex,
            opacity_micromap: None,
            transform_data,
            vertex_data,
            vertex_format,
            vertex_stride,
        }
    }

    /// Attaches an opacity micromap to this triangle geometry.
    pub fn opacity_micromap(
        mut self,
        opacity_micromap: AccelerationStructureOpacityMicromap<'a>,
    ) -> Self {
        self.opacity_micromap = Some(opacity_micromap);

        self
    }
}

struct AccessIter<'a> {
    accesses: MutexGuard<'a, Accesses>,
    idx: usize,
    previous_len: usize,
}

impl<'a> AccessIter<'a> {
    fn many(mut accesses: MutexGuard<'a, Accesses>, next_accesses: &[AccessType]) -> Self {
        if next_accesses.is_empty() {
            let previous_len = accesses.len();
            accesses.push(AccessType::Nothing);

            return Self {
                accesses,
                idx: 0,
                previous_len,
            };
        }

        if next_accesses.iter().copied().any(is_write_access) {
            let previous_len = accesses.len();
            for &next_access in next_accesses {
                if !accesses[previous_len..].contains(&next_access) {
                    accesses.push(next_access);
                }
            }

            return Self {
                accesses,
                idx: 0,
                previous_len,
            };
        }

        if next_accesses
            .iter()
            .all(|next_access| accesses.contains(next_access))
        {
            return Self {
                accesses,
                idx: 0,
                previous_len: 0,
            };
        }

        let previous_len = accesses.len();
        for idx in 0..previous_len {
            let access = accesses[idx];
            accesses.push(access);
        }

        for &next_access in next_accesses {
            if !accesses[previous_len..].contains(&next_access) {
                accesses.push(next_access);
            }
        }

        Self {
            accesses,
            idx: 0,
            previous_len,
        }
    }

    fn one(mut accesses: MutexGuard<'a, Accesses>, next_access: AccessType) -> Self {
        let previous_len = accesses.len();
        accesses.push(next_access);

        Self {
            accesses,
            idx: 0,
            previous_len,
        }
    }
}

impl Iterator for AccessIter<'_> {
    type Item = AccessType;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.previous_len {
            return None;
        }

        let access = self.accesses[self.idx];
        self.idx += 1;

        Some(access)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();

        (len, Some(len))
    }
}

impl ExactSizeIterator for AccessIter<'_> {
    fn len(&self) -> usize {
        self.previous_len - self.idx
    }
}

impl Drop for AccessIter<'_> {
    fn drop(&mut self) {
        self.accesses.drain(..self.previous_len);
    }
}

#[cfg(test)]
mod test {
    use {super::*, ash::vk::Handle};

    type Info = AccelerationStructureInfo;
    type Builder = AccelerationStructureInfoBuilder;

    #[test]
    pub fn accel_struct_info() {
        let info = Info::blas(32);
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn accel_struct_info_builder() {
        let info = Info {
            acceleration_structure_type: vk::AccelerationStructureTypeKHR::GENERIC,
            size: 32,
        };
        let builder = Builder::default().size(32).build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn accel_struct_info_builder_default_size() {
        let info = Info {
            acceleration_structure_type: vk::AccelerationStructureTypeKHR::GENERIC,
            size: 0,
        };

        assert_eq!(Builder::default().build(), info);
    }

    #[test]
    fn access_tracking_retains_write_until_next_write() {
        let accesses = Mutex::new(smallvec![AccessType::Nothing]);
        let write = AccessType::AccelerationStructureBuildWrite;
        let read = AccessType::RayTracingShaderReadAccelerationStructure;

        assert_eq!(swap_accesses(&accesses, &[write]), [AccessType::Nothing]);
        assert_eq!(swap_accesses(&accesses, &[read]), [write]);
        assert!(swap_accesses(&accesses, &[read]).is_empty());
        assert_eq!(swap_accesses(&accesses, &[write]), [write, read]);
        assert_eq!(lock_accesses(&accesses).as_slice(), [write]);
    }

    #[test]
    fn access_tracking_retains_reads_after_inline_capacity_is_exceeded() {
        let accesses = Mutex::new(smallvec![AccessType::Nothing]);
        let reads = [
            AccessType::VertexShaderReadOther,
            AccessType::GeometryShaderReadOther,
            AccessType::FragmentShaderReadOther,
            AccessType::ComputeShaderReadOther,
            AccessType::RayTracingShaderReadAccelerationStructure,
        ];

        for &read in &reads {
            swap_accesses(&accesses, &[read]);
            assert!(swap_accesses(&accesses, &[read]).is_empty());
        }

        let mut expected = vec![AccessType::Nothing];
        expected.extend(reads);
        let write = AccessType::AccelerationStructureBuildWrite;
        assert_eq!(swap_accesses(&accesses, &[write]), expected);
        assert_eq!(lock_accesses(&accesses).as_slice(), [write]);
    }

    #[test]
    fn build_size_count_shapes() {
        validate_build_size_counts(0, 0);
        validate_build_size_counts(1, 1);
        validate_build_size_counts(64, 64);

        assert!(std::panic::catch_unwind(|| validate_build_size_counts(1, 0)).is_err());
        assert!(std::panic::catch_unwind(|| validate_build_size_counts(0, 1)).is_err());
        assert!(std::panic::catch_unwind(|| validate_build_size_counts(2, 1)).is_err());
        if usize::BITS > u32::BITS {
            let oversized = usize::try_from(u64::from(u32::MAX) + 1).unwrap();

            assert!(
                std::panic::catch_unwind(|| validate_build_size_counts(oversized, oversized))
                    .is_err()
            );
        }
    }

    #[test]
    fn geometry_marshaler_emits_opacity_micromap_extension() {
        let usage = [
            OpacityMicromapUsage {
                count: 5,
                format: vk::OpacityMicromapFormatEXT::TYPE_4_STATE,
                subdivision_level: 2,
            },
            OpacityMicromapUsage {
                count: 3,
                format: vk::OpacityMicromapFormatEXT::TYPE_2_STATE,
                subdivision_level: 1,
            },
        ];
        let attachment =
            AccelerationStructureOpacityMicromap::new(vk::MicromapEXT::from_raw(7), &usage)
                .index_buffer(3)
                .index_type(vk::IndexType::UINT16)
                .index_stride(2)
                .base_triangle(11);
        let geometry =
            AccelerationStructureGeometry::new(triangles().opacity_micromap(attachment).into());
        let marshaled = AccelerationStructureGeometryMarshaler::new([&geometry]);
        let marshaled = Box::new(marshaled);
        let triangles = unsafe { marshaled.geometries()[0].geometry.triangles };
        let extension = unsafe {
            &*triangles
                .p_next
                .cast::<vk::AccelerationStructureTrianglesOpacityMicromapEXT<'_>>()
        };

        assert_eq!(extension as *const _, marshaled._opacity_micromaps.as_ptr());
        assert_eq!(unsafe { (*extension.p_usage_counts).count }, 5);
        assert_eq!(extension.usage_counts_count, 2);
        assert_eq!(unsafe { (*extension.p_usage_counts.add(1)).count }, 3);
        assert_eq!(extension.index_type, vk::IndexType::UINT16);
        assert_eq!(extension.index_stride, 2);
        assert_eq!(extension.base_triangle, 11);
        assert_eq!(extension.micromap, vk::MicromapEXT::from_raw(7));
        assert_eq!(unsafe { extension.index_buffer.device_address }, 3);
        assert_eq!(unsafe { (*extension.p_usage_counts).subdivision_level }, 2);
        assert_eq!(
            unsafe { (*extension.p_usage_counts).format },
            vk::OpacityMicromapFormatEXT::TYPE_4_STATE.as_raw() as u32
        );
        assert!(extension.pp_usage_counts.is_null());
        assert!(extension.p_next.is_null());
        assert_eq!(attachment.usage_counts.as_ptr(), usage.as_ptr());
    }

    #[test]
    fn geometry_marshaler_emits_triangles_without_extension() {
        let geometry = AccelerationStructureGeometry::new(triangles().into());
        let marshaled = AccelerationStructureGeometryMarshaler::new([&geometry]);

        assert!(marshaled._opacity_micromaps.is_empty());
        assert_eq!(marshaled.geometries.len(), 1);

        let triangles = unsafe { marshaled.geometries[0].geometry.triangles };

        assert!(triangles.p_next.is_null());
        assert_eq!(triangles.max_vertex, 3);

        unsafe {
            assert_eq!(triangles.index_data.device_address, 1);
            assert_eq!(triangles.vertex_data.device_address, 2);
            assert_eq!(triangles.transform_data.device_address, 0);
        }
    }

    #[test]
    fn geometry_marshaler_empty_usage_and_transform() {
        let empty = AccelerationStructureGeometryMarshaler::new(std::iter::empty());

        assert!(empty.geometries().is_empty());
        assert!(empty._usage_counts.is_empty());

        let geometry = AccelerationStructureGeometry::new(
            AccelerationStructureTriangles::new(
                0,
                vk::IndexType::NONE_KHR,
                2,
                64,
                128,
                vk::Format::R32G32B32_SFLOAT,
                12,
            )
            .opacity_micromap(AccelerationStructureOpacityMicromap::new(
                vk::MicromapEXT::null(),
                &[],
            ))
            .into(),
        );
        let marshaled = AccelerationStructureGeometryMarshaler::new([&geometry]);
        let raw = unsafe { marshaled.geometries()[0].geometry.triangles };

        assert_eq!(unsafe { raw.transform_data.device_address }, 64);

        let extension = unsafe {
            &*raw
                .p_next
                .cast::<vk::AccelerationStructureTrianglesOpacityMicromapEXT<'_>>()
        };
        assert_eq!(extension.usage_counts_count, 0);
        assert!(extension.p_usage_counts.is_null());
        assert!(extension.pp_usage_counts.is_null());
        assert_eq!(extension.index_type, vk::IndexType::NONE_KHR);
    }

    #[test]
    fn geometry_marshaler_keeps_all_extension_pointers_stable() {
        let usages = (0..64)
            .map(|index| {
                [OpacityMicromapUsage {
                    count: index + 1,
                    format: vk::OpacityMicromapFormatEXT::TYPE_2_STATE,
                    subdivision_level: 0,
                }]
            })
            .collect::<Vec<_>>();
        let geometries = (0..64)
            .map(|index| {
                let attachment = AccelerationStructureOpacityMicromap::new(
                    vk::MicromapEXT::from_raw(index as u64 + 1),
                    &usages[index],
                );
                AccelerationStructureGeometry::new(triangles().opacity_micromap(attachment).into())
            })
            .collect::<Vec<_>>();
        let marshaled = AccelerationStructureGeometryMarshaler::new(&geometries);
        let marshaled = Box::new(marshaled);

        for (index, geometry) in marshaled.geometries().iter().enumerate() {
            let triangles = unsafe { geometry.geometry.triangles };

            assert_eq!(
                triangles.p_next,
                (&marshaled._opacity_micromaps[index]
                    as *const vk::AccelerationStructureTrianglesOpacityMicromapEXT<'_>)
                    .cast()
            );
            let extension = &marshaled._opacity_micromaps[index];
            assert_eq!(
                extension.p_usage_counts,
                marshaled._usage_counts[index].as_ptr()
            );
            assert_eq!(
                unsafe { (*extension.p_usage_counts).count },
                index as u32 + 1
            );
            assert_eq!(
                extension.micromap,
                vk::MicromapEXT::from_raw(index as u64 + 1)
            );
        }
    }

    #[test]
    fn geometry_marshaler_mixes_concrete_geometry() {
        let usage = [OpacityMicromapUsage {
            count: 1,
            format: vk::OpacityMicromapFormatEXT::TYPE_2_STATE,
            subdivision_level: 0,
        }];
        let geometries = [
            AccelerationStructureGeometry::opaque(triangles().into()),
            AccelerationStructureGeometry::new(
                triangles()
                    .opacity_micromap(AccelerationStructureOpacityMicromap::new(
                        vk::MicromapEXT::from_raw(7),
                        &usage,
                    ))
                    .into(),
            ),
            AccelerationStructureGeometry::new(AccelerationStructureGeometryData::aabbs(8, 24)),
            AccelerationStructureGeometry::new(AccelerationStructureGeometryData::instances(16)),
            AccelerationStructureGeometry::new(
                AccelerationStructureGeometryData::instance_pointers(32),
            ),
        ];
        let marshaled = AccelerationStructureGeometryMarshaler::new(&geometries);
        assert_eq!(marshaled._opacity_micromaps.len(), 1);
        assert_eq!(marshaled.geometries[0].flags, vk::GeometryFlagsKHR::OPAQUE);
        assert_eq!(
            marshaled
                .geometries()
                .iter()
                .map(|geometry| geometry.geometry_type)
                .collect::<Vec<_>>(),
            [
                vk::GeometryTypeKHR::TRIANGLES,
                vk::GeometryTypeKHR::TRIANGLES,
                vk::GeometryTypeKHR::AABBS,
                vk::GeometryTypeKHR::INSTANCES,
                vk::GeometryTypeKHR::INSTANCES
            ]
        );
        unsafe {
            assert!(marshaled.geometries[0].geometry.triangles.p_next.is_null());
            assert!(!marshaled.geometries[1].geometry.triangles.p_next.is_null());
            assert_eq!(marshaled.geometries[2].geometry.aabbs.stride, 24);
            assert_eq!(
                marshaled.geometries[2].geometry.aabbs.data.device_address,
                8
            );
            assert_eq!(
                marshaled.geometries[3].geometry.instances.array_of_pointers,
                vk::FALSE
            );
            assert_eq!(
                marshaled.geometries[4].geometry.instances.array_of_pointers,
                vk::TRUE
            );
            assert_eq!(
                marshaled.geometries[3]
                    .geometry
                    .instances
                    .data
                    .device_address,
                16
            );
        }
    }

    fn lock_accesses(accesses: &Mutex<Accesses>) -> MutexGuard<'_, Accesses> {
        let accesses = accesses.lock();

        #[cfg(not(feature = "parking_lot"))]
        let accesses = accesses.expect("poisoned acceleration structure access lock");

        accesses
    }

    fn swap_accesses(accesses: &Mutex<Accesses>, next_accesses: &[AccessType]) -> Vec<AccessType> {
        AccessIter::many(lock_accesses(accesses), next_accesses).collect()
    }

    fn triangles<'a>() -> AccelerationStructureTriangles<'a> {
        AccelerationStructureTriangles::new(
            1,
            vk::IndexType::UINT32,
            3,
            0,
            2,
            vk::Format::R32G32B32_SFLOAT,
            12,
        )
    }
}
