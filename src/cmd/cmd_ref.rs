use {
    crate::{
        AnyAccelerationStructureNode, AnyResource, Execution, Node,
        driver::{
            accel_struct::{
                AccelerationStructureGeometry, AccelerationStructureGeometryInfo,
                DeviceOrHostAddress,
            },
            device::Device,
        },
    },
    ash::vk,
    log::trace,
    std::{cell::RefCell, ops::Deref},
};

/// Recording interface for general Vulkan commands.
///
/// This structure provides a strongly-typed set of methods which allow acceleration structures to
/// be built and updated.
///
/// # Examples
///
/// Basic usage:
///
/// ```no_run
/// # use ash::vk;
/// # use vk_graph::Graph;
/// # fn main() {
/// # let mut my_graph = Graph::default();
/// my_graph.begin_cmd()
///         .record_cmd(move |cmd| {
///             // Use provided command buffer functions or native calls
///             assert_ne!(cmd.handle, vk::CommandBuffer::null());
///         });
/// # }
/// ```
#[derive(Clone, Copy)]
pub struct CommandRef<'a> {
    cmd: &'a crate::driver::cmd_buf::CommandBuffer,

    #[cfg(feature = "checked")]
    exec: &'a Execution,

    #[cfg(feature = "checked")]
    graph_id: crate::GraphId,

    node_map: Option<&'a [usize]>,
    resources: &'a [AnyResource],
}

impl<'a> CommandRef<'a> {
    pub(crate) fn new(
        cmd: &'a crate::driver::cmd_buf::CommandBuffer,
        resources: &'a [AnyResource],
        exec: &'a Execution,
        #[cfg(feature = "checked")] graph_id: crate::GraphId,
    ) -> Self {
        Self {
            cmd,
            node_map: exec.node_map.as_deref(),
            resources,

            #[cfg(feature = "checked")]
            exec,

            #[cfg(feature = "checked")]
            graph_id: exec.stream_graph_id.unwrap_or(graph_id),
        }
    }

    /// Build acceleration structures.
    ///
    /// There is no ordering or synchronization implied between any of the individual acceleration
    /// structure builds.
    ///
    /// Requires a scratch buffer which was created with the following requirements:
    ///
    /// - Flags must include [`vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS`]
    /// - Size must be equal to or greater than the `build_size` value returned by
    ///   `AccelerationStructure::size_of`, aligned to `min_accel_struct_scratch_offset_alignment`
    ///   of `PhysicalDevice::vk_khr_acceleration_structure`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use ash::vk;
    /// # use vk_graph::cmd::BuildAccelerationStructureInfo;
    /// # use vk_sync::AccessType;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::accel_struct::{
    /// #     AccelerationStructure,
    /// #     AccelerationStructureGeometry,
    /// #     AccelerationStructureGeometryData,
    /// #     AccelerationStructureGeometryInfo,
    /// #     AccelerationStructureInfo,
    /// #     DeviceOrHostAddress,
    /// # };
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::Graph;
    /// # use vk_graph::driver::shader::Shader;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// # let mut my_graph = Graph::default();
    /// # let info = AccelerationStructureInfo::blas(1);
    /// # let blas_accel_struct = AccelerationStructure::create(&device, info)?;
    /// # let blas_node = my_graph.bind_resource(blas_accel_struct);
    /// # let scratch_buf_info =
    /// #     BufferInfo::device_mem(8, vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
    /// # let scratch_buf = Buffer::create(&device, scratch_buf_info)?;
    /// # let scratch_buf = my_graph.bind_resource(scratch_buf);
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::INDEX_BUFFER);
    /// # let my_idx_buf = Buffer::create(&device, buf_info)?;
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::VERTEX_BUFFER);
    /// # let my_vtx_buf = Buffer::create(&device, buf_info)?;
    /// # let index_buf = my_graph.bind_resource(my_idx_buf);
    /// # let vertex_buf = my_graph.bind_resource(my_vtx_buf);
    /// my_graph.begin_cmd()
    ///         .resource_access(index_buf, AccessType::IndexBuffer)
    ///         .resource_access(vertex_buf, AccessType::VertexBuffer)
    ///         .resource_access(scratch_buf, AccessType::AccelerationStructureBufferWrite)
    ///         .resource_access(blas_node, AccessType::AccelerationStructureBuildWrite)
    ///         .record_cmd(move |cmd| {
    ///             let scratch_addr = cmd.resource(scratch_buf).device_address();
    ///             let geom = AccelerationStructureGeometry {
    ///                 max_primitive_count: 64,
    ///                 flags: vk::GeometryFlagsKHR::OPAQUE,
    ///                 geometry: AccelerationStructureGeometryData::Triangles {
    ///                     index_addr: DeviceOrHostAddress::DeviceAddress(
    ///                         cmd.resource(index_buf).device_address()
    ///                     ),
    ///                     index_type: vk::IndexType::UINT32,
    ///                     max_vertex: 42,
    ///                     transform_addr: None,
    ///                     vertex_addr: DeviceOrHostAddress::DeviceAddress(
    ///                         cmd.resource(vertex_buf).device_address(),
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
    ///             cmd.build_accel_struct(&[
    ///                 BuildAccelerationStructureInfo::new(blas_node, scratch_addr, info)
    ///             ]);
    ///         });
    /// # Ok(()) }
    /// ```
    ///
    /// See also:
    ///
    /// - [`examples/ray_omni.rs`](/examples/ray_omni.rs)
    /// - [`examples/ray_tracing.rs`](/examples/ray_tracing.rs)
    /// - [`examples/rt_triangle.rs`](/examples/rt_triangle.rs)
    pub fn build_accel_struct(&self, infos: &[BuildAccelerationStructureInfo]) -> &Self {
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
                            .ty(info.build_data.acceleration_structure_type)
                            .flags(info.build_data.flags)
                            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                            .dst_acceleration_structure(self.resource(info.accel_struct).handle)
                            .geometries(&tls.geometries[start..end])
                            .scratch_data(info.scratch_addr.into()),
                    );
                    start = end;
                }

                vk_infos
            };

            let khr_acceleration_structure =
                Device::expect_vk_khr_acceleration_structure(&self.cmd.device);

            unsafe {
                khr_acceleration_structure.cmd_build_acceleration_structures(
                    self.cmd.handle,
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
    /// Each [`BuildAccelerationStructureIndirectInfo::range_base`] is a buffer device address which
    /// points to an array of [`vk::AccelerationStructureBuildRangeInfoKHR`] structures defining
    /// dynamic offsets to the addresses where geometry data is stored.
    pub fn build_accel_struct_indirect(
        &self,
        infos: &[BuildAccelerationStructureIndirectInfo],
    ) -> &Self {
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
                            .ty(info.build_data.acceleration_structure_type)
                            .flags(info.build_data.flags)
                            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                            .dst_acceleration_structure(self.resource(info.accel_struct).handle)
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

            let khr_acceleration_structure =
                Device::expect_vk_khr_acceleration_structure(&self.cmd.device);

            unsafe {
                khr_acceleration_structure.cmd_build_acceleration_structures_indirect(
                    self.cmd.handle,
                    &vk_infos,
                    &tls.range_bases,
                    &tls.range_strides,
                    &vk_max_primitive_counts,
                );
            }
        });

        self
    }

    pub(crate) fn clone_resource_at(&self, node_idx: usize) -> AnyResource {
        self.resources[node_idx].clone()
    }

    pub(crate) fn cmd_push_constants(
        &self,
        layout: vk::PipelineLayout,
        push_consts: &[vk::PushConstantRange],
        offset: u32,
        data: &[u8],
    ) {
        for push_const in push_consts {
            let push_const_end = push_const.offset + push_const.size;
            let data_end = offset + data.len() as u32;
            let end = data_end.min(push_const_end);
            let start = offset.max(push_const.offset);

            if end > start {
                trace!(
                    "      push constants {:?} {}..{}",
                    push_const.stage_flags, start, end
                );

                unsafe {
                    self.device.cmd_push_constants(
                        self.handle,
                        layout,
                        push_const.stage_flags,
                        start,
                        &data[(start - offset) as usize..(end - offset) as usize],
                    );
                }
            }
        }
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
    ///   `AccelerationStructure::size_of`, aligned to `min_accel_struct_scratch_offset_alignment`
    ///   of `PhysicalDevice::vk_khr_acceleration_structure`.
    pub fn update_accel_struct(&self, infos: &[UpdateAccelerationStructureInfo]) -> &Self {
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
                            .ty(info.update_data.acceleration_structure_type)
                            .flags(info.update_data.flags)
                            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
                            .dst_acceleration_structure(self.resource(info.dst_accel_struct).handle)
                            .src_acceleration_structure(self.resource(info.src_accel_struct).handle)
                            .geometries(&tls.geometries[start..end])
                            .scratch_data(info.scratch_addr.into()),
                    );
                    start = end;
                }

                vk_infos
            };

            let khr_acceleration_structure =
                Device::expect_vk_khr_acceleration_structure(&self.cmd.device);

            unsafe {
                khr_acceleration_structure.cmd_build_acceleration_structures(
                    self.cmd.handle,
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
    /// Each [`UpdateAccelerationStructureIndirectInfo::range_base`] is a buffer device address
    /// which points to an array of [`vk::AccelerationStructureBuildRangeInfoKHR`] structures
    /// defining dynamic offsets to the addresses where geometry data is stored.
    pub fn update_accel_struct_indirect(
        &self,
        infos: &[UpdateAccelerationStructureIndirectInfo],
    ) -> &Self {
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
                            .ty(info.update_data.acceleration_structure_type)
                            .flags(info.update_data.flags)
                            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
                            .src_acceleration_structure(self.resource(info.src_accel_struct).handle)
                            .dst_acceleration_structure(self.resource(info.dst_accel_struct).handle)
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

            let khr_acceleration_structure =
                Device::expect_vk_khr_acceleration_structure(&self.cmd.device);

            unsafe {
                khr_acceleration_structure.cmd_build_acceleration_structures_indirect(
                    self.cmd.handle,
                    &vk_infos,
                    &tls.range_bases,
                    &tls.range_strides,
                    &vk_max_primitive_counts,
                );
            }
        });

        self
    }

    /// Returns a borrow of the original Vulkan resource (buffer, image or acceleration structure)
    /// which the given bound resource node represents.
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: Node,
    {
        #[cfg(feature = "checked")]
        resource_node.assert_owner(self.graph_id);

        let mut node_idx = resource_node.index();
        if let Some(node_map) = self.node_map {
            node_idx = node_map[node_idx];
        }

        /*
        You must have called an access function for this node on this execution before borrowing
        the resource!

        Code that attempts to access this function is attempting to get access to the Vulkan
        resource (buffer, image, or acceleration structure). In order to access any resources the
        access type must first be specified so the correct barriers may be added.

        See: https://attackgoat.github.io/vk-graph/pipeline_sync.html
        */
        #[cfg(feature = "checked")]
        assert!(
            self.exec.accesses.contains(node_idx),
            "unexpected node access: call an access function first"
        );

        resource_node.borrow_at(self.resources, node_idx)
    }
}

impl<'a> Deref for CommandRef<'a> {
    type Target = crate::driver::cmd_buf::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        self.cmd
    }
}

/// Specifies the information and data used to build an acceleration structure.
///
/// See [`vkCmdBuildAccelerationStructuresKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBuildAccelerationStructuresKHR.html).
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
    /// `AccelerationStructure::size_of`).
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
/// See [`vkCmdBuildAccelerationStructuresKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBuildAccelerationStructuresKHR.html).
#[derive(Clone, Debug)]
pub struct BuildAccelerationStructureIndirectInfo {
    /// The acceleration structure to be written.
    pub accel_struct: AnyAccelerationStructureNode,

    /// Specifies the geometry data to use when building the acceleration structure.
    pub build_data: AccelerationStructureGeometryInfo<AccelerationStructureGeometry>,

    /// A buffer device address which points to `data.geometry.len()`
    /// [vk::AccelerationStructureBuildRangeInfoKHR] structures defining dynamic offsets to the
    /// addresses where geometry data is stored.
    pub range_base: vk::DeviceAddress,

    /// Byte stride between elements of [`Self::range_base`].
    pub range_stride: u32,

    /// The temporary buffer or host address (with enough capacity per
    /// `AccelerationStructure::size_of`).
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
/// See [`vkCmdBuildAccelerationStructuresKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBuildAccelerationStructuresKHR.html).
#[derive(Clone, Debug)]
pub struct UpdateAccelerationStructureIndirectInfo {
    /// The acceleration structure to be written.
    pub dst_accel_struct: AnyAccelerationStructureNode,

    /// A buffer device address which points to `data.geometry.len()`
    /// [vk::AccelerationStructureBuildRangeInfoKHR] structures defining dynamic offsets to the
    /// addresses where geometry data is stored.
    pub range_base: vk::DeviceAddress,

    /// Byte stride between elements of [`Self::range_base`].
    pub range_stride: u32,

    /// The temporary buffer or host address (with enough capacity per
    /// `AccelerationStructure::size_of`).
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
#[derive(Clone, Debug)]
pub struct UpdateAccelerationStructureInfo {
    /// The acceleration structure to be written.
    pub dst_accel_struct: AnyAccelerationStructureNode,

    /// The temporary buffer or host address (with enough capacity per
    /// `AccelerationStructure::size_of`).
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
