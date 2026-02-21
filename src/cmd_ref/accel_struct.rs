use {
    super::Bindings,
    crate::{
        AnyAccelerationStructureNode,
        driver::{
            accel_struct::{
                AccelerationStructureGeometry, AccelerationStructureGeometryInfo,
                DeviceOrHostAddress,
            },
            device::Device,
        },
    },
    ash::vk,
    std::cell::RefCell,
};

/// Recording interface for acceleration structure commands.
///
/// This structure provides a strongly-typed set of methods which allow acceleration structures to
/// be built and updated. An instance of `Acceleration` is provided to the closure parameter of
/// [`PassRef::record_accel_struct`].
///
/// # Examples
///
/// Basic usage:
///
/// ```no_run
/// # use std::sync::Arc;
/// # use ash::vk;
/// # use vk_graph::driver::accel_struct::{AccelerationStructure, AccelerationStructureInfo};
/// # use vk_graph::driver::DriverError;
/// # use vk_graph::driver::device::{Device, DeviceInfo};
/// # use vk_graph::Graph;
/// # use vk_graph::driver::shader::Shader;
/// # fn main() -> Result<(), DriverError> {
/// # let device = Device::new(DeviceInfo::default())?;
/// # let mut my_graph = Graph::default();
/// # let info = AccelerationStructureInfo::blas(1);
/// my_graph.begin_cmd()
///         .record_accel_struct(move |accel_struct, bindings| {
///             // During this closure we have access to the build and update methods
///         });
/// # Ok(()) }
/// ```
pub struct AccelerationStructureRef<'a> {
    pub(super) bindings: Bindings<'a>,
    pub(super) cmd_buf: vk::CommandBuffer,
    pub(super) device: &'a Device,
}

impl AccelerationStructureRef<'_> {
    /// Build acceleration structures.
    ///
    /// There is no ordering or synchronization implied between any of the individual acceleration
    /// structure builds.
    ///
    /// Requires a scratch buffer which was created with the following requirements:
    ///
    /// - Flags must include [`vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS`]
    /// - Size must be equal to or greater than the `build_size` value returned by
    ///   [`AccelerationStructure::size_of`] aligned to `min_accel_struct_scratch_offset_alignment`
    ///   of
    ///   [`PhysicalDevice::accel_struct_properties`](crate::driver::physical_device::PhysicalDevice::accel_struct_properties).
    ///
    ///     TODO: Link to somewhere else for a full example of the scratch buffer steps
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use ash::vk;
    /// # use vk_graph::cmd_ref::BuildAccelerationStructureInfo;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::accel_struct::{AccelerationStructure, AccelerationStructureGeometry, AccelerationStructureGeometryData, AccelerationStructureGeometryInfo, AccelerationStructureInfo, DeviceOrHostAddress};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::Graph;
    /// # use vk_graph::driver::shader::Shader;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::new(DeviceInfo::default())?;
    /// # let mut my_graph = Graph::default();
    /// # let info = AccelerationStructureInfo::blas(1);
    /// # let blas_accel_struct = AccelerationStructure::create(&device, info)?;
    /// # let blas_node = my_graph.bind_node(blas_accel_struct);
    /// # let scratch_buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
    /// # let scratch_buf = Buffer::create(&device, scratch_buf_info)?;
    /// # let scratch_buf = my_graph.bind_node(scratch_buf);
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::INDEX_BUFFER);
    /// # let my_idx_buf = Buffer::create(&device, buf_info)?;
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::VERTEX_BUFFER);
    /// # let my_vtx_buf = Buffer::create(&device, buf_info)?;
    /// # let index_node = my_graph.bind_node(my_idx_buf);
    /// # let vertex_node = my_graph.bind_node(my_vtx_buf);
    /// my_graph.begin_cmd()
    ///         .read_node(index_node)
    ///         .read_node(vertex_node)
    ///         .write_node(blas_node)
    ///         .write_node(scratch_buf)
    ///         .record_accel_struct(move |accel_struct, bindings| {
    ///             let scratch_addr = bindings[scratch_buf].device_address();
    ///             let geom = AccelerationStructureGeometry {
    ///                 max_primitive_count: 64,
    ///                 flags: vk::GeometryFlagsKHR::OPAQUE,
    ///                 geometry: AccelerationStructureGeometryData::Triangles {
    ///                     index_addr: DeviceOrHostAddress::DeviceAddress(
    ///                         bindings[index_node].device_address()
    ///                     ),
    ///                     index_type: vk::IndexType::UINT32,
    ///                     max_vertex: 42,
    ///                     transform_addr: None,
    ///                     vertex_addr: DeviceOrHostAddress::DeviceAddress(
    ///                         bindings[vertex_node].device_address(),
    ///                     ),
    ///                     vertex_format: vk::Format::R32G32B32_SFLOAT,
    ///                     vertex_stride: 12,
    ///                 },
    ///             };
    ///             let build_range = vk::AccelerationStructureBuildRangeInfoKHR {
    ///                 first_vertex: 0,
    ///                 primitive_count: 1,
    ///                 primitive_offset: 0,
    ///                 transform_offset: 0,
    ///             };
    ///             let info = AccelerationStructureGeometryInfo::blas([(geom, build_range)]);
    ///
    ///             accel_struct.build(&[
    ///                 BuildAccelerationStructureInfo::new(blas_node, scratch_addr, info)
    ///             ]);
    ///         });
    /// # Ok(()) }
    /// ```
    pub fn build(&self, infos: &[BuildAccelerationStructureInfo]) -> &Self {
        #[derive(Default)]
        struct Tls {
            geometries: Vec<vk::AccelerationStructureGeometryKHR<'static>>,
            ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
        }

        thread_local! {
            static TLS: RefCell<Tls> = Default::default();
        }

        TLS.with_borrow_mut(|tls| {
            tls.geometries.clear();
            tls.geometries.extend(infos.iter().flat_map(|info| {
                info.build_data.geometries.iter().map(|(geometry, _)| {
                    <&AccelerationStructureGeometry as Into<
                            vk::AccelerationStructureGeometryKHR,
                        >>::into(geometry)
                })
            }));

            tls.ranges.clear();
            tls.ranges.extend(
                infos
                    .iter()
                    .flat_map(|info| info.build_data.geometries.iter().map(|(_, range)| *range)),
            );

            let vk_ranges = {
                let mut start = 0;
                let mut vk_ranges = Vec::with_capacity(infos.len());
                for info in infos {
                    let end = start + info.build_data.geometries.len();
                    vk_ranges.push(&tls.ranges[start..end]);
                    start = end;
                }

                vk_ranges
            };

            let vk_infos = {
                let mut start = 0;
                let mut vk_infos = Vec::with_capacity(infos.len());
                for info in infos {
                    let end = start + info.build_data.geometries.len();
                    vk_infos.push(
                        vk::AccelerationStructureBuildGeometryInfoKHR::default()
                            .ty(info.build_data.ty)
                            .flags(info.build_data.flags)
                            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                            .dst_acceleration_structure(self.bindings[info.accel_struct].handle)
                            .geometries(&tls.geometries[start..end])
                            .scratch_data(info.scratch_addr.into()),
                    );
                    start = end;
                }

                vk_infos
            };

            unsafe {
                Device::expect_accel_struct_ext(self.device).cmd_build_acceleration_structures(
                    self.cmd_buf,
                    &vk_infos,
                    &vk_ranges,
                );
            }
        });

        self
    }

    /// Builds acceleration structures with some parameters provided on the device.
    ///
    /// There is no ordering or synchronization implied between any of the individual acceleration
    /// structure builds.
    ///
    /// `range` is a buffer device address which points to `info.geometry.len()`
    /// [vk::VkAccelerationStructureBuildRangeInfoKHR] structures defining dynamic offsets to the
    /// addresses where geometry data is stored, as defined by `info`.
    pub fn build_indirect(&self, infos: &[BuildAccelerationStructureIndirectInfo]) -> &Self {
        #[derive(Default)]
        struct Tls {
            geometries: Vec<vk::AccelerationStructureGeometryKHR<'static>>,
            max_primitive_counts: Vec<u32>,
            range_bases: Vec<vk::DeviceAddress>,
            range_strides: Vec<u32>,
        }

        thread_local! {
            static TLS: RefCell<Tls> = Default::default();
        }

        TLS.with_borrow_mut(|tls| {
            tls.geometries.clear();
            tls.geometries.extend(infos.iter().flat_map(|info| {
                info.build_data.geometries.iter().map(
                    <&AccelerationStructureGeometry as Into<
                        vk::AccelerationStructureGeometryKHR,
                    >>::into,
                )
            }));

            tls.max_primitive_counts.clear();
            tls.max_primitive_counts
                .extend(infos.iter().flat_map(|info| {
                    info.build_data
                        .geometries
                        .iter()
                        .map(|geometry| geometry.max_primitive_count)
                }));

            tls.range_bases.clear();
            tls.range_strides.clear();
            let (vk_infos, vk_max_primitive_counts) = {
                let mut start = 0;
                let mut vk_infos = Vec::with_capacity(infos.len());
                let mut vk_max_primitive_counts = Vec::with_capacity(infos.len());
                for info in infos {
                    let end = start + info.build_data.geometries.len();
                    vk_infos.push(
                        vk::AccelerationStructureBuildGeometryInfoKHR::default()
                            .ty(info.build_data.ty)
                            .flags(info.build_data.flags)
                            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                            .dst_acceleration_structure(self.bindings[info.accel_struct].handle)
                            .geometries(&tls.geometries[start..end])
                            .scratch_data(info.scratch_data.into()),
                    );
                    vk_max_primitive_counts.push(&tls.max_primitive_counts[start..end]);
                    start = end;

                    tls.range_bases.push(info.range_base);
                    tls.range_strides.push(info.range_stride);
                }

                (vk_infos, vk_max_primitive_counts)
            };

            unsafe {
                Device::expect_accel_struct_ext(self.device)
                    .cmd_build_acceleration_structures_indirect(
                        self.cmd_buf,
                        &vk_infos,
                        &tls.range_bases,
                        &tls.range_strides,
                        &vk_max_primitive_counts,
                    );
            }
        });

        self
    }

    /// Update acceleration structures.
    ///
    /// There is no ordering or synchronization implied between any of the individual acceleration
    /// structure updates.
    ///
    /// Requires a scratch buffer which was created with the following requirements:
    ///
    /// - Flags must include [`vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS`]
    /// - Size must be equal to or greater than the `update_size` value returned by
    ///   [`AccelerationStructure::size_of`] aligned to `min_accel_struct_scratch_offset_alignment`
    ///   of
    ///   [`PhysicalDevice::accel_struct_properties`](crate::driver::physical_device::PhysicalDevice::accel_struct_properties).
    pub fn update(&self, infos: &[UpdateAccelerationStructureInfo]) -> &Self {
        #[derive(Default)]
        struct Tls {
            geometries: Vec<vk::AccelerationStructureGeometryKHR<'static>>,
            ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
        }

        thread_local! {
            static TLS: RefCell<Tls> = Default::default();
        }

        TLS.with_borrow_mut(|tls| {
            tls.geometries.clear();
            tls.geometries.extend(infos.iter().flat_map(|info| {
                info.update_data.geometries.iter().map(|(geometry, _)| {
                    <&AccelerationStructureGeometry as Into<
                            vk::AccelerationStructureGeometryKHR,
                        >>::into(geometry)
                })
            }));

            tls.ranges.clear();
            tls.ranges.extend(
                infos
                    .iter()
                    .flat_map(|info| info.update_data.geometries.iter().map(|(_, range)| *range)),
            );

            let vk_ranges = {
                let mut start = 0;
                let mut vk_ranges = Vec::with_capacity(infos.len());
                for info in infos {
                    let end = start + info.update_data.geometries.len();
                    vk_ranges.push(&tls.ranges[start..end]);
                    start = end;
                }

                vk_ranges
            };

            let vk_infos = {
                let mut start = 0;
                let mut vk_infos = Vec::with_capacity(infos.len());
                for info in infos {
                    let end = start + info.update_data.geometries.len();
                    vk_infos.push(
                        vk::AccelerationStructureBuildGeometryInfoKHR::default()
                            .ty(info.update_data.ty)
                            .flags(info.update_data.flags)
                            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
                            .dst_acceleration_structure(self.bindings[info.dst_accel_struct].handle)
                            .src_acceleration_structure(self.bindings[info.src_accel_struct].handle)
                            .geometries(&tls.geometries[start..end])
                            .scratch_data(info.scratch_addr.into()),
                    );
                    start = end;
                }

                vk_infos
            };

            unsafe {
                Device::expect_accel_struct_ext(self.device).cmd_build_acceleration_structures(
                    self.cmd_buf,
                    &vk_infos,
                    &vk_ranges,
                );
            }
        });

        self
    }

    /// Updates acceleration structures with some parameters provided on the device.
    ///
    /// There is no ordering or synchronization implied between any of the individual acceleration
    /// structure updates.
    ///
    /// `range` is a buffer device address which points to `info.geometry.len()`
    /// [vk::VkAccelerationStructureBuildRangeInfoKHR] structures defining dynamic offsets to the
    /// addresses where geometry data is stored, as defined by `info`.
    pub fn update_indirect(&self, infos: &[UpdateAccelerationStructureIndirectInfo]) -> &Self {
        #[derive(Default)]
        struct Tls {
            geometries: Vec<vk::AccelerationStructureGeometryKHR<'static>>,
            max_primitive_counts: Vec<u32>,
            range_bases: Vec<vk::DeviceAddress>,
            range_strides: Vec<u32>,
        }

        thread_local! {
            static TLS: RefCell<Tls> = Default::default();
        }

        TLS.with_borrow_mut(|tls| {
            tls.geometries.clear();
            tls.geometries.extend(infos.iter().flat_map(|info| {
                info.update_data.geometries.iter().map(
                    <&AccelerationStructureGeometry as Into<
                        vk::AccelerationStructureGeometryKHR,
                    >>::into,
                )
            }));

            tls.max_primitive_counts.clear();
            tls.max_primitive_counts
                .extend(infos.iter().flat_map(|info| {
                    info.update_data
                        .geometries
                        .iter()
                        .map(|geometry| geometry.max_primitive_count)
                }));

            tls.range_bases.clear();
            tls.range_strides.clear();
            let (vk_infos, vk_max_primitive_counts) = {
                let mut start = 0;
                let mut vk_infos = Vec::with_capacity(infos.len());
                let mut vk_max_primitive_counts = Vec::with_capacity(infos.len());
                for info in infos {
                    let end = start + info.update_data.geometries.len();
                    vk_infos.push(
                        vk::AccelerationStructureBuildGeometryInfoKHR::default()
                            .ty(info.update_data.ty)
                            .flags(info.update_data.flags)
                            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
                            .src_acceleration_structure(self.bindings[info.src_accel_struct].handle)
                            .dst_acceleration_structure(self.bindings[info.dst_accel_struct].handle)
                            .geometries(&tls.geometries[start..end])
                            .scratch_data(info.scratch_addr.into()),
                    );
                    vk_max_primitive_counts.push(&tls.max_primitive_counts[start..end]);
                    start = end;

                    tls.range_bases.push(info.range_base);
                    tls.range_strides.push(info.range_stride);
                }

                (vk_infos, vk_max_primitive_counts)
            };

            unsafe {
                Device::expect_accel_struct_ext(self.device)
                    .cmd_build_acceleration_structures_indirect(
                        self.cmd_buf,
                        &vk_infos,
                        &tls.range_bases,
                        &tls.range_strides,
                        &vk_max_primitive_counts,
                    );
            }
        });

        self
    }
}

/// Specifies the information and data used to build an acceleration structure.
///
/// See
/// [VkAccelerationStructureBuildGeometryInfoKHR](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureBuildGeometryInfoKHR.html)
/// for more information.
#[derive(Clone, Debug)]
pub struct BuildAccelerationStructureInfo {
    /// The acceleration structure to be written.
    pub accel_struct: AnyAccelerationStructureNode,

    /// Specifies the geometry data to use when building the acceleration structure.
    pub build_data: AccelerationStructureGeometryInfo<(
        AccelerationStructureGeometry,
        vk::AccelerationStructureBuildRangeInfoKHR,
    )>,

    /// The temporary buffer or host address (with enough capacity per
    /// [AccelerationStructure::size_of]).
    pub scratch_addr: DeviceOrHostAddress,
}

impl BuildAccelerationStructureInfo {
    /// Constructs new acceleration structure build information.
    pub fn new(
        accel_struct: impl Into<AnyAccelerationStructureNode>,
        scratch_addr: impl Into<DeviceOrHostAddress>,
        build_data: AccelerationStructureGeometryInfo<(
            AccelerationStructureGeometry,
            vk::AccelerationStructureBuildRangeInfoKHR,
        )>,
    ) -> Self {
        let accel_struct = accel_struct.into();
        let scratch_addr = scratch_addr.into();

        Self {
            accel_struct,
            build_data,
            scratch_addr,
        }
    }
}

/// Specifies the information and data used to build an acceleration structure with some parameters
/// sourced on the device.
///
/// See
/// [VkAccelerationStructureBuildGeometryInfoKHR](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureBuildGeometryInfoKHR.html)
/// for more information.
#[derive(Clone, Debug)]
pub struct BuildAccelerationStructureIndirectInfo {
    /// The acceleration structure to be written.
    pub accel_struct: AnyAccelerationStructureNode,

    /// Specifies the geometry data to use when building the acceleration structure.
    pub build_data: AccelerationStructureGeometryInfo<AccelerationStructureGeometry>,

    /// A buffer device addresses which points to `data.geometry.len()`
    /// [vk::VkAccelerationStructureBuildRangeInfoKHR] structures defining dynamic offsets to the
    /// addresses where geometry data is stored.
    pub range_base: vk::DeviceAddress,

    /// Byte stride between elements of [range].
    pub range_stride: u32,

    /// The temporary buffer or host address (with enough capacity per
    /// [AccelerationStructure::size_of]).
    pub scratch_data: DeviceOrHostAddress,
}

impl BuildAccelerationStructureIndirectInfo {
    /// Constructs new acceleration structure indirect build information.
    pub fn new(
        accel_struct: impl Into<AnyAccelerationStructureNode>,
        scratch_data: impl Into<DeviceOrHostAddress>,
        build_data: AccelerationStructureGeometryInfo<AccelerationStructureGeometry>,
        range_base: vk::DeviceAddress,
        range_stride: u32,
    ) -> Self {
        let accel_struct = accel_struct.into();
        let scratch_data = scratch_data.into();

        Self {
            accel_struct,
            build_data,
            range_base,
            range_stride,
            scratch_data,
        }
    }
}

/// Specifies the information and data used to update an acceleration structure with some parameters
/// sourced on the device.
///
/// See
/// [VkAccelerationStructureBuildGeometryInfoKHR](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureBuildGeometryInfoKHR.html)
/// for more information.
#[derive(Clone, Debug)]
pub struct UpdateAccelerationStructureIndirectInfo {
    /// The acceleration structure to be written.
    pub dst_accel_struct: AnyAccelerationStructureNode,

    /// A buffer device addresses which points to `data.geometry.len()`
    /// [vk::VkAccelerationStructureBuildRangeInfoKHR] structures defining dynamic offsets to the
    /// addresses where geometry data is stored.
    pub range_base: vk::DeviceAddress,

    /// Byte stride between elements of [range].
    pub range_stride: u32,

    /// The temporary buffer or host address (with enough capacity per
    /// [AccelerationStructure::size_of]).
    pub scratch_addr: DeviceOrHostAddress,

    /// The source acceleration structure to be read.
    pub src_accel_struct: AnyAccelerationStructureNode,

    /// Specifies the geometry data to use when building the acceleration structure.
    pub update_data: AccelerationStructureGeometryInfo<AccelerationStructureGeometry>,
}

impl UpdateAccelerationStructureIndirectInfo {
    /// Constructs new acceleration structure indirect update information.
    pub fn new(
        src_accel_struct: impl Into<AnyAccelerationStructureNode>,
        dst_accel_struct: impl Into<AnyAccelerationStructureNode>,
        scratch_addr: impl Into<DeviceOrHostAddress>,
        update_data: AccelerationStructureGeometryInfo<AccelerationStructureGeometry>,
        range_base: vk::DeviceAddress,
        range_stride: u32,
    ) -> Self {
        let src_accel_struct = src_accel_struct.into();
        let dst_accel_struct = dst_accel_struct.into();
        let scratch_addr = scratch_addr.into();

        Self {
            dst_accel_struct,
            range_base,
            range_stride,
            scratch_addr,
            src_accel_struct,
            update_data,
        }
    }
}

/// Specifies the information and data used to update an acceleration structure.
///
/// See
/// [VkAccelerationStructureBuildGeometryInfoKHR](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureBuildGeometryInfoKHR.html)
/// for more information.
#[derive(Clone, Debug)]
pub struct UpdateAccelerationStructureInfo {
    /// The acceleration structure to be written.
    pub dst_accel_struct: AnyAccelerationStructureNode,

    /// The temporary buffer or host address (with enough capacity per
    /// [AccelerationStructure::size_of]).
    pub scratch_addr: DeviceOrHostAddress,

    /// The source acceleration structure to be read.
    pub src_accel_struct: AnyAccelerationStructureNode,

    /// Specifies the geometry data to use when updating the acceleration structure.
    pub update_data: AccelerationStructureGeometryInfo<(
        AccelerationStructureGeometry,
        vk::AccelerationStructureBuildRangeInfoKHR,
    )>,
}

impl UpdateAccelerationStructureInfo {
    /// Constructs new acceleration structure update information.
    pub fn new(
        src_accel_struct: impl Into<AnyAccelerationStructureNode>,
        dst_accel_struct: impl Into<AnyAccelerationStructureNode>,
        scratch_addr: impl Into<DeviceOrHostAddress>,
        update_data: AccelerationStructureGeometryInfo<(
            AccelerationStructureGeometry,
            vk::AccelerationStructureBuildRangeInfoKHR,
        )>,
    ) -> Self {
        let src_accel_struct = src_accel_struct.into();
        let dst_accel_struct = dst_accel_struct.into();
        let scratch_addr = scratch_addr.into();

        Self {
            dst_accel_struct,
            scratch_addr,
            src_accel_struct,
            update_data,
        }
    }
}
