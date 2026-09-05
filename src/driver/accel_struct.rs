//! Acceleration structure resource types

use {
    super::{
        Buffer, BufferInfo, DriverError, device::Device, is_write_access,
        pipeline_stage_access_flags,
    },
    ash::vk,
    derive_builder::Builder,
    log::warn,
    std::{
        ffi::c_void,
        fmt::{Debug, Formatter},
        mem::size_of_val,
        thread::panicking,
    },
    vk_sync::AccessType,
};

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

/// Smart pointer handle to an [acceleration structure] object.
///
/// Also contains the backing buffer and information about the object.
///
/// ```no_run
/// # use ash::vk;
/// # use vk_graph::driver::DriverError;
/// # use vk_graph::driver::device::{Device, DeviceInfo};
/// # use vk_graph::driver::accel_struct::{AccelerationStructure, AccelerationStructureInfo};
/// # fn main() -> Result<(), DriverError> {
/// # let device = Device::create(DeviceInfo::default())?;
/// let info = AccelerationStructureInfo::blas(0);
/// let accel_struct = AccelerationStructure::create(&device, info)?;
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
    accesses: Mutex<Vec<AccessType>>,

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
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::accel_struct::{AccelerationStructure, AccelerationStructureInfo};
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// const SIZE: vk::DeviceSize = 1024;
    /// let info = AccelerationStructureInfo::blas(SIZE);
    /// let accel_struct = AccelerationStructure::create(&device, info)?;
    ///
    /// assert_ne!(accel_struct.handle, vk::AccelerationStructureKHR::null());
    /// assert_eq!(accel_struct.info.size, SIZE);
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
            accesses: Mutex::new(vec![AccessType::Nothing]),
            buffer,
            handle,
            info,
        })
    }

    /// Returns the device address of this object.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_sync::AccessType;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::accel_struct::{AccelerationStructure, AccelerationStructureInfo};
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// # const SIZE: vk::DeviceSize = 1024;
    /// # let info = AccelerationStructureInfo::blas(SIZE);
    /// # let my_accel_struct = AccelerationStructure::create(&device, info)?;
    /// let addr = AccelerationStructure::device_address(&my_accel_struct);
    ///
    /// assert_ne!(addr, 0);
    /// # Ok(()) }
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

    fn lock_accesses(&self) -> MutexGuard<'_, Vec<AccessType>> {
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

    /// Returns the size of some geometry info which is then used to create a new
    /// [AccelerationStructure] instance or update an existing instance.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::accel_struct::{
    /// #     AccelerationStructure,
    /// #     AccelerationStructureGeometry,
    /// #     AccelerationStructureGeometryData,
    /// #     AccelerationStructureGeometryInfo,
    /// #     DeviceOrHostAddress,
    /// # };
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// # let my_geom_triangles = AccelerationStructureGeometryData::Triangles {
    /// #     index_addr: DeviceOrHostAddress::DeviceAddress(0),
    /// #     index_type: vk::IndexType::UINT32,
    /// #     max_vertex: 1,
    /// #     transform_addr: None,
    /// #     vertex_addr: DeviceOrHostAddress::DeviceAddress(0),
    /// #     vertex_format: vk::Format::R32G32B32_SFLOAT,
    /// #     vertex_stride: 12,
    /// # };
    /// let my_geom = AccelerationStructureGeometry {
    ///     max_primitive_count: 1,
    ///     flags: vk::GeometryFlagsKHR::OPAQUE,
    ///     geometry: my_geom_triangles,
    /// };
    /// let build_range = vk::AccelerationStructureBuildRangeInfoKHR {
    ///     primitive_count: 1,
    ///     primitive_offset: 0,
    ///     first_vertex: 0,
    ///     transform_offset: 0,
    /// };
    /// let my_info = AccelerationStructureGeometryInfo::blas([(my_geom, build_range)]);
    /// let res = AccelerationStructure::size_of(&device, &my_info);
    ///
    /// assert_eq!(res.create_size, 2432);
    /// assert_eq!(res.build_size, 640);
    /// assert_eq!(res.update_size, 0);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn size_of<D: Clone + Into<AccelerationStructureGeometryDataExt>>(
        device: &Device,
        info: &AccelerationStructureGeometryInfo<impl AsRef<AccelerationStructureGeometry<D>>>,
    ) -> AccelerationStructureSize {
        let geometries =
            AccelerationStructureGeometryMarshaler::new(info.geometries.iter().map(AsRef::as_ref));
        let max_primitive_counts = info
            .geometries
            .iter()
            .map(|geometry| geometry.as_ref().max_primitive_count)
            .collect::<Vec<_>>();

        let vk_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(info.acceleration_structure_type)
            .flags(info.flags)
            .geometries(geometries.geometries());
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        let khr_acceleration_structure = Device::expect_vk_khr_acceleration_structure(device);

        unsafe {
            khr_acceleration_structure.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::HOST_OR_DEVICE,
                &vk_info,
                &max_primitive_counts,
                &mut sizes,
            );
        }

        AccelerationStructureSize {
            build_size: sizes.build_scratch_size,
            create_size: sizes.acceleration_structure_size,
            update_size: sizes.update_scratch_size,
        }
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

/// Structure specifying geometries to be built into an acceleration structure.
///
/// The default data type preserves the legacy `Copy` geometry API. Use
/// [`AccelerationStructureGeometryDataExt`] to mix legacy and micromap geometry.
///
/// See [`VkAccelerationStructureGeometryKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureGeometryKHR.html).
#[derive(Clone, Copy, Debug)]
pub struct AccelerationStructureGeometry<D = AccelerationStructureGeometryData> {
    /// Describes additional properties of how the geometry should be built.
    pub flags: vk::GeometryFlagsKHR,

    /// Specifies acceleration structure geometry data.
    pub geometry: D,

    /// The number of primitives built into each geometry.
    pub max_primitive_count: u32,
}

impl<D> AccelerationStructureGeometry<D> {
    /// Creates a new acceleration structure geometry instance.
    pub fn new(max_primitive_count: u32, geometry: D) -> Self {
        let flags = Default::default();

        Self {
            flags,
            geometry,
            max_primitive_count,
        }
    }

    /// Creates a new acceleration structure geometry instance with the
    /// [vk::GeometryFlagsKHR::OPAQUE] flag set.
    pub fn opaque(max_primitive_count: u32, geometry: D) -> Self {
        Self::new(max_primitive_count, geometry).flags(vk::GeometryFlagsKHR::OPAQUE)
    }

    /// Sets the instance flags.
    pub fn flags(mut self, flags: vk::GeometryFlagsKHR) -> Self {
        self.flags = flags;

        self
    }
}

impl<D, T> AsRef<AccelerationStructureGeometry<D>> for (AccelerationStructureGeometry<D>, T) {
    fn as_ref(&self) -> &AccelerationStructureGeometry<D> {
        &self.0
    }
}

impl<'b> From<&'b AccelerationStructureGeometry> for vk::AccelerationStructureGeometryKHR<'_> {
    fn from(&value: &'b AccelerationStructureGeometry) -> Self {
        value.into()
    }
}

impl From<AccelerationStructureGeometry> for vk::AccelerationStructureGeometryKHR<'_> {
    fn from(value: AccelerationStructureGeometry) -> Self {
        Self::default()
            .flags(value.flags)
            .geometry(value.geometry.into())
            .geometry_type(value.geometry.into())
    }
}

/// Specifies acceleration structure geometry data.
///
/// See [`VkAccelerationStructureGeometryKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureGeometryKHR.html).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AccelerationStructureGeometryData {
    /// Axis-aligned bounding box geometry in a bottom-level acceleration structure.
    ///
    /// See [`VkAccelerationStructureGeometryAabbsDataKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureGeometryAabbsDataKHR.html).
    AABBs {
        /// A device or host address to memory containing [vk::AabbPositionsKHR] structures
        /// containing position data for each axis-aligned bounding box in the geometry.
        addr: DeviceOrHostAddress,

        /// Stride in bytes between each entry in data.
        ///
        /// The stride must be a multiple of `8`.
        stride: vk::DeviceSize,
    },

    /// Geometry consisting of instances of other acceleration structures.
    ///
    /// See [`VkAccelerationStructureGeometryInstancesDataKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureGeometryInstancesDataKHR.html).
    Instances {
        /// Either the address of an array of device addresses referencing individual
        /// [`VkAccelerationStructureInstanceKHR`] values if `array_of_pointers` is `true`, or the
        /// address of an array of [`VkAccelerationStructureInstanceKHR`] values.
        ///
        /// Addresses and `VkAccelerationStructureInstanceKHR` values are tightly packed.
        ///
        /// [`VkAccelerationStructureInstanceKHR`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureInstanceKHR.html
        addr: DeviceOrHostAddress,

        /// Specifies whether data is used as an array of addresses or just an array.
        array_of_pointers: bool,
    },

    /// A triangle geometry in a bottom-level acceleration structure.
    ///
    /// See [`VkAccelerationStructureGeometryTrianglesDataKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureGeometryTrianglesDataKHR.html).
    Triangles {
        /// A device or host address to memory containing index data for this geometry.
        index_addr: DeviceOrHostAddress,

        /// The [`VkIndexType`] of each index element.
        ///
        /// [`VkIndexType`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndexType.html
        index_type: vk::IndexType,

        /// The highest index of a vertex that will be addressed by a build command using this
        /// structure.
        max_vertex: u32,

        /// A device or host address to memory containing an optional reference to a
        /// [`VkTransformMatrixKHR`] structure describing a transformation from the space in which
        /// the vertices in this geometry are described to the space in which the acceleration
        /// structure is defined.
        ///
        /// [`VkTransformMatrixKHR`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkTransformMatrixKHR.html
        transform_addr: Option<DeviceOrHostAddress>,

        /// A device or host address to memory containing vertex data for this geometry.
        vertex_addr: DeviceOrHostAddress,

        /// The [`VkFormat`] of each vertex element.
        ///
        /// [`VkFormat`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormat.html
        vertex_format: vk::Format,

        /// The stride in bytes between each vertex.
        vertex_stride: vk::DeviceSize,
    },
}

impl AccelerationStructureGeometryData {
    /// Specifies acceleration structure geometry data as AABBs.
    pub fn aabbs(addr: impl Into<DeviceOrHostAddress>, stride: vk::DeviceSize) -> Self {
        let addr = addr.into();

        Self::AABBs { addr, stride }
    }

    /// Specifies acceleration structure geometry data as instances.
    pub fn instances(addr: impl Into<DeviceOrHostAddress>) -> Self {
        let addr = addr.into();

        Self::Instances {
            addr,
            array_of_pointers: false,
        }
    }

    /// Specifies acceleration structure geometry data as an array of instance pointers.
    pub fn instance_pointers(addr: impl Into<DeviceOrHostAddress>) -> Self {
        let addr = addr.into();

        Self::Instances {
            addr,
            array_of_pointers: true,
        }
    }

    /// Specifies acceleration structure geometry data as triangles.
    pub fn triangles(
        index_addr: impl Into<DeviceOrHostAddress>,
        index_type: vk::IndexType,
        max_vertex: u32,
        transform_addr: impl Into<Option<DeviceOrHostAddress>>,
        vertex_addr: impl Into<DeviceOrHostAddress>,
        vertex_format: vk::Format,
        vertex_stride: vk::DeviceSize,
    ) -> Self {
        let index_addr = index_addr.into();
        let transform_addr = transform_addr.into();
        let vertex_addr = vertex_addr.into();

        Self::Triangles {
            index_addr,
            index_type,
            max_vertex,
            transform_addr,
            vertex_addr,
            vertex_format,
            vertex_stride,
        }
    }

    /// Attaches an opacity micromap when this value contains triangle geometry.
    ///
    /// # Panics
    /// Panics when called on non-triangle geometry.
    pub fn opacity_micromap(
        self,
        opacity_micromap: AccelerationStructureOpacityMicromap,
    ) -> AccelerationStructureGeometryDataExt {
        match self {
            Self::Triangles {
                index_addr,
                index_type,
                max_vertex,
                transform_addr,
                vertex_addr,
                vertex_format,
                vertex_stride,
            } => AccelerationStructureGeometryDataExt::Triangles(
                AccelerationStructureTriangles::new(
                    index_addr,
                    index_type,
                    max_vertex,
                    transform_addr,
                    vertex_addr,
                    vertex_format,
                    vertex_stride,
                )
                .opacity_micromap(opacity_micromap),
            ),
            _ => panic!("opacity micromaps can only be attached to triangle geometry"),
        }
    }
}

impl From<AccelerationStructureGeometryData> for vk::GeometryTypeKHR {
    fn from(value: AccelerationStructureGeometryData) -> Self {
        match value {
            AccelerationStructureGeometryData::AABBs { .. } => Self::AABBS,
            AccelerationStructureGeometryData::Instances { .. } => Self::INSTANCES,
            AccelerationStructureGeometryData::Triangles { .. } => Self::TRIANGLES,
        }
    }
}

impl From<AccelerationStructureGeometryData> for vk::AccelerationStructureGeometryDataKHR<'_> {
    fn from(value: AccelerationStructureGeometryData) -> Self {
        match value {
            AccelerationStructureGeometryData::AABBs { addr, stride } => Self {
                aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::default()
                    .data(addr.into())
                    .stride(stride),
            },
            AccelerationStructureGeometryData::Instances {
                addr,
                array_of_pointers,
            } => Self {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(array_of_pointers)
                    .data(addr.into()),
            },
            AccelerationStructureGeometryData::Triangles {
                index_addr,
                index_type,
                max_vertex,
                transform_addr,
                vertex_addr,
                vertex_format,
                vertex_stride,
            } => Self {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .index_data(index_addr.into())
                    .index_type(index_type)
                    .max_vertex(max_vertex)
                    .transform_data(transform_addr.map(Into::into).unwrap_or_default())
                    .vertex_data(vertex_addr.into())
                    .vertex_format(vertex_format)
                    .vertex_stride(vertex_stride),
            },
        }
    }
}

/// Geometry data supporting extensions without changing the legacy geometry variants.
///
/// Convert legacy data with [`Into::into`] to mix it with micromap triangles in one build.
#[derive(Clone, Debug)]
pub enum AccelerationStructureGeometryDataExt {
    /// Geometry without an extension attachment.
    Geometry(AccelerationStructureGeometryData),

    /// Triangle geometry with an optional opacity micromap attachment.
    Triangles(AccelerationStructureTriangles),
}

impl From<AccelerationStructureGeometryData> for AccelerationStructureGeometryDataExt {
    fn from(value: AccelerationStructureGeometryData) -> Self {
        Self::Geometry(value)
    }
}

impl From<AccelerationStructureTriangles> for AccelerationStructureGeometryDataExt {
    fn from(value: AccelerationStructureTriangles) -> Self {
        Self::Triangles(value)
    }
}

/// Triangle geometry and its optional opacity micromap attachment.
#[derive(Clone, Debug)]
pub struct AccelerationStructureTriangles {
    /// A device or host address to memory containing index data for this geometry.
    pub index_addr: DeviceOrHostAddress,

    /// The [`VkIndexType`] of each index element.
    ///
    /// [`VkIndexType`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndexType.html
    pub index_type: vk::IndexType,

    /// The highest index of a vertex that will be addressed by a build command using this
    /// structure.
    pub max_vertex: u32,

    /// Optional opacity micromap attachment.
    pub opacity_micromap: Option<AccelerationStructureOpacityMicromap>,

    /// A device or host address to memory containing an optional reference to a
    /// [`VkTransformMatrixKHR`] structure describing a transformation from the space in which
    /// the vertices in this geometry are described to the space in which the acceleration
    /// structure is defined.
    ///
    /// [`VkTransformMatrixKHR`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkTransformMatrixKHR.html
    pub transform_addr: Option<DeviceOrHostAddress>,

    /// A device or host address to memory containing vertex data for this geometry.
    pub vertex_addr: DeviceOrHostAddress,

    /// The [`VkFormat`] of each vertex element.
    ///
    /// [`VkFormat`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormat.html
    pub vertex_format: vk::Format,

    /// The stride in bytes between each vertex.
    pub vertex_stride: vk::DeviceSize,
}

impl AccelerationStructureTriangles {
    /// Creates triangle geometry without an opacity micromap attachment.
    pub fn new(
        index_addr: impl Into<DeviceOrHostAddress>,
        index_type: vk::IndexType,
        max_vertex: u32,
        transform_addr: impl Into<Option<DeviceOrHostAddress>>,
        vertex_addr: impl Into<DeviceOrHostAddress>,
        vertex_format: vk::Format,
        vertex_stride: vk::DeviceSize,
    ) -> Self {
        Self {
            index_addr: index_addr.into(),
            index_type,
            max_vertex,
            opacity_micromap: None,
            transform_addr: transform_addr.into(),
            vertex_addr: vertex_addr.into(),
            vertex_format,
            vertex_stride,
        }
    }

    /// Attaches an opacity micromap to this triangle geometry.
    pub fn opacity_micromap(
        mut self,
        opacity_micromap: AccelerationStructureOpacityMicromap,
    ) -> Self {
        self.opacity_micromap = Some(opacity_micromap);
        self
    }
}

/// Opacity micromap data attached to acceleration-structure triangles.
#[derive(Clone, Debug)]
pub struct AccelerationStructureOpacityMicromap {
    /// Offset added to non-special micromap indices. With `index_type` set to `NONE_KHR`,
    /// triangle `i` uses micromap triangle `base_triangle + i`.
    pub base_triangle: u32,

    /// Address containing micromap indices for the triangles.
    pub index_addr: DeviceOrHostAddress,

    /// Byte stride between micromap indices.
    pub index_stride: vk::DeviceSize,

    /// Type of each micromap index.
    pub index_type: vk::IndexType,

    /// Native opacity micromap handle.
    pub micromap: vk::MicromapEXT,
    usage_counts: Box<[vk::MicromapUsageEXT]>,
}

impl AccelerationStructureOpacityMicromap {
    /// Creates an opacity micromap attachment with owned usage counts.
    pub fn new<I>(micromap: vk::MicromapEXT, usage_counts: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<vk::MicromapUsageEXT>,
    {
        Self {
            base_triangle: 0,
            index_addr: DeviceOrHostAddress::DeviceAddress(0),
            index_stride: 0,
            index_type: vk::IndexType::NONE_KHR,
            micromap,
            usage_counts: usage_counts.into_iter().map(Into::into).collect(),
        }
    }

    /// Sets the micromap-index buffer address.
    pub fn index_addr(mut self, index_addr: impl Into<DeviceOrHostAddress>) -> Self {
        self.index_addr = index_addr.into();
        self
    }

    /// Sets the micromap-index element type.
    pub fn index_type(mut self, index_type: vk::IndexType) -> Self {
        self.index_type = index_type;
        self
    }

    /// Sets the byte stride between micromap-index elements.
    pub fn index_stride(mut self, index_stride: vk::DeviceSize) -> Self {
        self.index_stride = index_stride;
        self
    }

    /// Sets the offset added to non-special micromap indices (or the implicit triangle index
    /// when `index_type` is `NONE_KHR`).
    pub fn base_triangle(mut self, base_triangle: u32) -> Self {
        self.base_triangle = base_triangle;
        self
    }

    /// Returns the Vulkan usage records owned by this attachment.
    pub fn usage_counts(&self) -> &[vk::MicromapUsageEXT] {
        &self.usage_counts
    }
}

/// Owns Vulkan geometry and extension records for the duration of one Vulkan call.
pub(crate) struct AccelerationStructureGeometryMarshaler {
    _opacity_micromaps: Vec<vk::AccelerationStructureTrianglesOpacityMicromapEXT<'static>>,
    _source: Vec<AccelerationStructureGeometry<AccelerationStructureGeometryDataExt>>,
    geometries: Vec<vk::AccelerationStructureGeometryKHR<'static>>,
}

impl AccelerationStructureGeometryMarshaler {
    pub(crate) fn new<'a, D: Clone + Into<AccelerationStructureGeometryDataExt> + 'a>(
        geometries: impl IntoIterator<Item = &'a AccelerationStructureGeometry<D>>,
    ) -> Self {
        let source = geometries
            .into_iter()
            .map(|geometry| AccelerationStructureGeometry {
                flags: geometry.flags,
                geometry: geometry.geometry.clone().into(),
                max_primitive_count: geometry.max_primitive_count,
            })
            .collect::<Vec<_>>();
        let extension_count = source
            .iter()
            .filter(|geometry| {
                matches!(
                    &geometry.geometry,
                    AccelerationStructureGeometryDataExt::Triangles(triangles)
                        if triangles.opacity_micromap.is_some()
                )
            })
            .count();
        let mut opacity_micromaps = Vec::with_capacity(extension_count);

        for geometry in &source {
            if let AccelerationStructureGeometryDataExt::Triangles(triangles) = &geometry.geometry
                && let Some(attachment) = &triangles.opacity_micromap
            {
                // Usage arrays are boxed and owned by `source`; both allocations stay alive
                // until the Vulkan call completes, even if the marshaler is moved.
                opacity_micromaps.push(vk::AccelerationStructureTrianglesOpacityMicromapEXT {
                    usage_counts_count: attachment
                        .usage_counts
                        .len()
                        .try_into()
                        .expect("micromap usage count exceeds u32::MAX"),
                    p_usage_counts: attachment.usage_counts.as_ptr(),
                    ..vk::AccelerationStructureTrianglesOpacityMicromapEXT::default()
                        .index_type(attachment.index_type)
                        .index_buffer(attachment.index_addr.into())
                        .index_stride(attachment.index_stride)
                        .base_triangle(attachment.base_triangle)
                        .micromap(attachment.micromap)
                });
            }
        }

        let mut geometries_out = Vec::with_capacity(source.len());
        let mut extension_index = 0;
        for geometry in &source {
            let (geometry_type, geometry_data) = match &geometry.geometry {
                AccelerationStructureGeometryDataExt::Geometry(data) => {
                    ((*data).into(), (*data).into())
                }
                AccelerationStructureGeometryDataExt::Triangles(triangles) => {
                    let mut vk_triangles =
                        vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                            .index_data(triangles.index_addr.into())
                            .index_type(triangles.index_type)
                            .max_vertex(triangles.max_vertex)
                            .transform_data(
                                triangles.transform_addr.map(Into::into).unwrap_or_default(),
                            )
                            .vertex_data(triangles.vertex_addr.into())
                            .vertex_format(triangles.vertex_format)
                            .vertex_stride(triangles.vertex_stride);

                    if triangles.opacity_micromap.is_some() {
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
            _source: source,
            geometries: geometries_out,
        }
    }

    pub(crate) fn geometries(&self) -> &[vk::AccelerationStructureGeometryKHR<'_>] {
        &self.geometries
    }
}

/// Specifies the geometry data of an acceleration structure.
#[derive(Clone, Debug)]
pub struct AccelerationStructureGeometryInfo<G> {
    /// Type of acceleration structure.
    pub acceleration_structure_type: vk::AccelerationStructureTypeKHR,

    /// Specifies additional parameters of the acceleration structure.
    pub flags: vk::BuildAccelerationStructureFlagsKHR,

    /// A slice of geometry structures.
    pub geometries: Box<[G]>,
}

impl<G> AccelerationStructureGeometryInfo<G> {
    /// A bottom-level acceleration structure containing the AABBs or geometry to be intersected.
    pub fn blas(geometries: impl Into<Box<[G]>>) -> Self {
        let geometries = geometries.into();

        Self {
            acceleration_structure_type: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            flags: Default::default(),
            geometries,
        }
    }

    /// A top-level acceleration structure containing instance data referring to bottom-level
    /// acceleration structures.
    pub fn tlas(geometries: impl Into<Box<[G]>>) -> Self {
        let geometries = geometries.into();

        Self {
            acceleration_structure_type: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            flags: Default::default(),
            geometries,
        }
    }

    /// Sets the flags on this instance.
    pub fn flags(mut self, flags: vk::BuildAccelerationStructureFlagsKHR) -> Self {
        self.flags = flags;
        self
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
    /// Use [`AccelerationStructure::size_of`] to calculate this value.
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

/// Holds the results of the [`AccelerationStructure::size_of`] function.
#[derive(Clone, Copy, Debug)]
pub struct AccelerationStructureSize {
    /// The size of the scratch buffer required when building an acceleration structure using
    /// [`CommandRef::build_accel_struct`](crate::cmd::CommandRef::build_accel_struct).
    pub build_size: vk::DeviceSize,

    /// The value of `size` parameter needed by [`AccelerationStructureInfo`] for use with the
    /// [`AccelerationStructure::create`] function.
    pub create_size: vk::DeviceSize,

    /// The size of the scratch buffer required when updating an acceleration structure using
    /// [`CommandRef::update_accel_struct`](crate::cmd::CommandRef::update_accel_struct).
    pub update_size: vk::DeviceSize,
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

struct AccessIter<'a> {
    accesses: MutexGuard<'a, Vec<AccessType>>,
    idx: usize,
    previous_len: usize,
}

impl<'a> AccessIter<'a> {
    fn one(mut accesses: MutexGuard<'a, Vec<AccessType>>, next_access: AccessType) -> Self {
        let previous_len = accesses.len();
        accesses.push(next_access);

        Self {
            accesses,
            idx: 0,
            previous_len,
        }
    }

    fn many(mut accesses: MutexGuard<'a, Vec<AccessType>>, next_accesses: &[AccessType]) -> Self {
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
        accesses.extend_from_within(..);
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

/// Specifies a constant device or host address.
///
/// See [`VkDeviceOrHostAddressKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceOrHostAddressKHR.html).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeviceOrHostAddress {
    /// An address value returned from [`AccelerationStructure::device_address`].
    DeviceAddress(vk::DeviceAddress),

    /// A host memory address.
    HostAddress(*mut c_void),
}

impl From<vk::DeviceAddress> for DeviceOrHostAddress {
    fn from(device_address: vk::DeviceAddress) -> Self {
        Self::DeviceAddress(device_address)
    }
}

impl From<*mut c_void> for DeviceOrHostAddress {
    fn from(host_address: *mut c_void) -> Self {
        Self::HostAddress(host_address)
    }
}

// Safety: The entire purpose of DeviceOrHostAddress is to share memory with Vulkan
unsafe impl Send for DeviceOrHostAddress {}
unsafe impl Sync for DeviceOrHostAddress {}

impl From<DeviceOrHostAddress> for vk::DeviceOrHostAddressConstKHR {
    fn from(value: DeviceOrHostAddress) -> Self {
        match value {
            DeviceOrHostAddress::DeviceAddress(device_address) => Self { device_address },
            DeviceOrHostAddress::HostAddress(host_address) => Self { host_address },
        }
    }
}

impl From<DeviceOrHostAddress> for vk::DeviceOrHostAddressKHR {
    fn from(value: DeviceOrHostAddress) -> Self {
        match value {
            DeviceOrHostAddress::DeviceAddress(device_address) => Self { device_address },
            DeviceOrHostAddress::HostAddress(host_address) => Self { host_address },
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ash::vk::Handle;

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

    fn triangles() -> AccelerationStructureGeometryData {
        AccelerationStructureGeometryData::triangles(
            1,
            vk::IndexType::UINT32,
            3,
            None,
            2,
            vk::Format::R32G32B32_SFLOAT,
            12,
        )
    }

    #[test]
    fn legacy_geometry_api() {
        fn copy<T: Copy>() {}
        fn value_traits<T: Copy + Eq + std::hash::Hash>() {}
        copy::<AccelerationStructureGeometry>();
        value_traits::<AccelerationStructureGeometryData>();

        let data = AccelerationStructureGeometryData::Triangles {
            index_addr: 1.into(),
            index_type: vk::IndexType::UINT32,
            max_vertex: 3,
            transform_addr: None,
            vertex_addr: 2.into(),
            vertex_format: vk::Format::R32G32B32_SFLOAT,
            vertex_stride: 12,
        };
        assert_eq!(data, triangles());
        let geometry: AccelerationStructureGeometry = AccelerationStructureGeometry {
            flags: vk::GeometryFlagsKHR::OPAQUE,
            geometry: data,
            max_primitive_count: 1,
        };
        let expected_type = match data {
            AccelerationStructureGeometryData::AABBs { .. } => vk::GeometryTypeKHR::AABBS,
            AccelerationStructureGeometryData::Instances { .. } => vk::GeometryTypeKHR::INSTANCES,
            AccelerationStructureGeometryData::Triangles { .. } => vk::GeometryTypeKHR::TRIANGLES,
        };
        let ty: vk::GeometryTypeKHR = data.into();
        let raw_data: vk::AccelerationStructureGeometryDataKHR<'_> = data.into();
        let raw: vk::AccelerationStructureGeometryKHR<'_> = geometry.into();
        let borrowed: vk::AccelerationStructureGeometryKHR<'_> = (&geometry).into();
        assert_eq!(ty, expected_type);
        assert_eq!(raw.geometry_type, ty);
        assert_eq!(borrowed.flags, geometry.flags);
        assert_eq!(unsafe { raw_data.triangles.vertex_stride }, 12);
        let tuple = (
            geometry,
            vk::AccelerationStructureBuildRangeInfoKHR::default(),
        );
        let reference: &AccelerationStructureGeometry = tuple.as_ref();
        assert_eq!(reference.geometry, data);

        // Compile the original generic size-query contract and all command signatures without a GPU.
        fn size<G: AsRef<AccelerationStructureGeometry>>(
            device: &Device,
            info: &AccelerationStructureGeometryInfo<G>,
        ) -> AccelerationStructureSize {
            AccelerationStructure::size_of(device, info)
        }
        fn record(
            cmd: &crate::cmd::CommandRef<'_>,
            build: &[crate::cmd::BuildAccelerationStructureInfo],
            build_indirect: &[crate::cmd::BuildAccelerationStructureIndirectInfo],
            update: &[crate::cmd::UpdateAccelerationStructureInfo],
            update_indirect: &[crate::cmd::UpdateAccelerationStructureIndirectInfo],
        ) {
            cmd.build_accel_struct(build)
                .build_accel_struct_indirect(build_indirect)
                .update_accel_struct(update)
                .update_accel_struct_indirect(update_indirect);
        }
        let _ = size::<(AccelerationStructureGeometry, ())>;
        let _ = record;
    }

    #[test]
    fn geometry_marshaler_emits_triangles_without_extension() {
        let geometry = AccelerationStructureGeometry::new(1, triangles());
        let marshaled = AccelerationStructureGeometryMarshaler::new([&geometry]);

        assert!(marshaled._opacity_micromaps.is_empty());
        assert_eq!(marshaled.geometries.len(), 1);
        let triangles = unsafe { marshaled.geometries[0].geometry.triangles };
        assert!(triangles.p_next.is_null());
        assert_eq!(triangles.max_vertex, 3);
    }

    #[test]
    fn geometry_marshaler_emits_opacity_micromap_extension() {
        let usage = vk::MicromapUsageEXT::default()
            .count(5)
            .subdivision_level(2)
            .format(vk::OpacityMicromapFormatEXT::TYPE_4_STATE.as_raw() as u32);
        let attachment =
            AccelerationStructureOpacityMicromap::new(vk::MicromapEXT::from_raw(7), [usage])
                .index_addr(3)
                .index_type(vk::IndexType::UINT16)
                .index_stride(2)
                .base_triangle(11);
        let geometry =
            AccelerationStructureGeometry::new(1, triangles().opacity_micromap(attachment));
        let marshaled = AccelerationStructureGeometryMarshaler::new([&geometry]);
        drop(geometry);
        let marshaled = Box::new(marshaled);
        let triangles = unsafe { marshaled.geometries[0].geometry.triangles };
        let extension = unsafe {
            &*triangles
                .p_next
                .cast::<vk::AccelerationStructureTrianglesOpacityMicromapEXT<'_>>()
        };

        assert_eq!(extension as *const _, marshaled._opacity_micromaps.as_ptr());
        assert_eq!(unsafe { (*extension.p_usage_counts).count }, 5);
        assert_eq!(extension.usage_counts_count, 1);
        assert_eq!(extension.index_type, vk::IndexType::UINT16);
        assert_eq!(extension.index_stride, 2);
        assert_eq!(extension.base_triangle, 11);
        assert_eq!(extension.micromap, vk::MicromapEXT::from_raw(7));
    }

    #[test]
    fn geometry_marshaler_keeps_all_extension_pointers_stable() {
        let geometries = (0..64)
            .map(|index| {
                let attachment = AccelerationStructureOpacityMicromap::new(
                    vk::MicromapEXT::from_raw(index + 1),
                    [vk::MicromapUsageEXT::default().count(1)],
                );
                AccelerationStructureGeometry::new(1, triangles().opacity_micromap(attachment))
            })
            .collect::<Vec<_>>();
        let marshaled = AccelerationStructureGeometryMarshaler::new(&geometries);

        for (index, geometry) in marshaled.geometries.iter().enumerate() {
            let triangles = unsafe { geometry.geometry.triangles };
            assert_eq!(
                triangles.p_next,
                (&marshaled._opacity_micromaps[index]
                    as *const vk::AccelerationStructureTrianglesOpacityMicromapEXT<'_>)
                    .cast()
            );
        }
    }

    #[test]
    fn geometry_marshaler_mixes_legacy_and_micromap_geometry() {
        let geometries = [
            AccelerationStructureGeometry::opaque(1, triangles().into()),
            AccelerationStructureGeometry::new(
                1,
                triangles().opacity_micromap(AccelerationStructureOpacityMicromap::new(
                    vk::MicromapEXT::from_raw(7),
                    [vk::MicromapUsageEXT::default().count(1)],
                )),
            ),
            AccelerationStructureGeometry::new(
                1,
                AccelerationStructureGeometryData::aabbs(8, 24).into(),
            ),
            AccelerationStructureGeometry::new(
                1,
                AccelerationStructureGeometryData::instances(16).into(),
            ),
        ];
        let marshaled = AccelerationStructureGeometryMarshaler::new(&geometries);
        assert_eq!(marshaled._opacity_micromaps.len(), 1);
        assert_eq!(marshaled.geometries[0].flags, vk::GeometryFlagsKHR::OPAQUE);
        unsafe {
            assert!(marshaled.geometries[0].geometry.triangles.p_next.is_null());
            assert!(!marshaled.geometries[1].geometry.triangles.p_next.is_null());
            assert_eq!(marshaled.geometries[2].geometry.aabbs.stride, 24);
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

    fn lock_accesses(accesses: &Mutex<Vec<AccessType>>) -> MutexGuard<'_, Vec<AccessType>> {
        let accesses = accesses.lock();

        #[cfg(not(feature = "parking_lot"))]
        let accesses = accesses.expect("poisoned acceleration structure access lock");

        accesses
    }

    fn swap_accesses(
        accesses: &Mutex<Vec<AccessType>>,
        next_accesses: &[AccessType],
    ) -> Vec<AccessType> {
        AccessIter::many(lock_accesses(accesses), next_accesses).collect()
    }

    #[test]
    fn access_tracking_retains_write_until_next_write() {
        let accesses = Mutex::new(vec![AccessType::Nothing]);
        let write = AccessType::AccelerationStructureBuildWrite;
        let read = AccessType::RayTracingShaderReadAccelerationStructure;

        assert_eq!(swap_accesses(&accesses, &[write]), [AccessType::Nothing]);
        assert_eq!(swap_accesses(&accesses, &[read]), [write]);
        assert!(swap_accesses(&accesses, &[read]).is_empty());
        assert_eq!(swap_accesses(&accesses, &[write]), [write, read]);
        assert_eq!(*lock_accesses(&accesses), [write]);
    }
}
