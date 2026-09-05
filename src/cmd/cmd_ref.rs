use {
    crate::{
        AnyAccelerationStructureNode, AnyMicromapNode, AnyResource, Execution, ResourceNode,
        driver::{
            accel_struct::{
                AccelerationStructureGeometry, AccelerationStructureGeometryData,
                AccelerationStructureGeometryDataExt, AccelerationStructureGeometryInfo,
                AccelerationStructureGeometryMarshaler, DeviceOrHostAddress,
            },
            device::Device,
            micromap::OpacityMicromapBuildInfo,
        },
        private::ResourceNodeIndex,
        resource::{ResourceSetIndex, ResourceSetMap},
    },
    ash::vk,
    log::trace,
    std::{cell::RefCell, ops::Deref},
};

/// Recording interface for general Vulkan commands.
///
/// Provides typed acceleration-structure and micromap commands alongside raw Vulkan access.
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
    resource_set_map: Option<&'a [ResourceSetIndex]>,
    resource_sets: &'a ResourceSetMap,
    resources: &'a [AnyResource],
}

impl<'a> CommandRef<'a> {
    pub(crate) fn new(
        cmd: &'a crate::driver::cmd_buf::CommandBuffer,
        resources: &'a [AnyResource],
        resource_sets: &'a ResourceSetMap,
        exec: &'a Execution,
        #[cfg(feature = "checked")] graph_id: crate::GraphId,
    ) -> Self {
        Self {
            cmd,
            node_map: exec.node_map.as_deref(),
            resource_set_map: exec.resource_set_map.as_deref(),
            resource_sets,
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
    ///         .resource_access(index_buf, AccessType::AccelerationStructureBuildInputRead)
    ///         .resource_access(vertex_buf, AccessType::AccelerationStructureBuildInputRead)
    ///         .resource_access(
    ///             scratch_buf,
    ///             AccessType::AccelerationStructureBuildScratchReadWrite,
    ///         )
    ///         .resource_access(blas_node, AccessType::AccelerationStructureBuildWrite)
    ///         .record_cmd(move |cmd| {
    ///             let scratch_addr = cmd.resource(scratch_buf).device_address();
    ///             let geom = AccelerationStructureGeometry {
    ///                 max_primitive_count: 64,
    ///                 flags: vk::GeometryFlagsKHR::OPAQUE,
    ///                 geometry: AccelerationStructureGeometryData::triangles(
    ///                     cmd.resource(index_buf).device_address(),
    ///                     vk::IndexType::UINT32,
    ///                     42,
    ///                     None,
    ///                     cmd.resource(vertex_buf).device_address(),
    ///                     vk::Format::R32G32B32_SFLOAT,
    ///                     12,
    ///                 ),
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
    pub fn build_accel_struct<D: Clone + Into<AccelerationStructureGeometryDataExt>>(
        &self,
        infos: &[BuildAccelerationStructureInfo<D>],
    ) -> &Self {
        #[derive(Default)]
        struct Tls {
            ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
        }

        thread_local! {
            static TLS: RefCell<Tls> = Default::default();
        }

        TLS.with_borrow_mut(|tls| {
            let geometries =
                AccelerationStructureGeometryMarshaler::new(infos.iter().flat_map(|info| {
                    info.build_data
                        .geometries
                        .iter()
                        .map(|(geometry, _)| geometry)
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
                            .geometries(&geometries.geometries()[start..end])
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
    pub fn build_accel_struct_indirect<D: Clone + Into<AccelerationStructureGeometryDataExt>>(
        &self,
        infos: &[BuildAccelerationStructureIndirectInfo<D>],
    ) -> &Self {
        #[derive(Default)]
        struct Tls {
            max_primitive_counts: Vec<u32>,
            range_bases: Vec<vk::DeviceAddress>,
            range_strides: Vec<u32>,
        }

        thread_local! {
            static TLS: RefCell<Tls> = Default::default();
        }

        TLS.with_borrow_mut(|tls| {
            let geometries = AccelerationStructureGeometryMarshaler::new(
                infos
                    .iter()
                    .flat_map(|info| info.build_data.geometries.iter()),
            );

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
                            .geometries(&geometries.geometries()[start..end])
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

    /// Builds a batch of opacity micromaps on the device.
    ///
    /// Micromaps do not have an update mode. Every entry is recorded with
    /// [`vk::BuildMicromapModeEXT::BUILD`]. There is no ordering or synchronization implied between
    /// entries in the batch.
    ///
    /// Graph users must separately declare [`vk_sync::AccessType::MicromapBuildWrite`] for each
    /// destination micromap, [`vk_sync::AccessType::MicromapBuildInputRead`] for the buffers backing
    /// `data` and `triangle_array`, and
    /// [`vk_sync::AccessType::MicromapBuildScratchReadWrite`] for the scratch buffer. Device
    /// addresses do not identify graph resources, so declaring only the micromap access is not
    /// sufficient.
    ///
    /// See [`vkCmdBuildMicromapsEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBuildMicromapsEXT.html).
    ///
    /// # Safety
    ///
    /// `infos` must satisfy all `vkCmdBuildMicromapsEXT` validity requirements. Through execution,
    /// addresses must reference live, bound memory on this device with the required usage flags,
    /// alignment, and capacity. Null scratch is valid only when the required scratch size is zero.
    /// Input data, usage counts, strides, formats, and flags must describe a valid build within device
    /// limits. Scratch and destination sizes must cover a matching size query; their ranges must not
    /// overlap inputs, each other, or another build's scratch or destination storage.
    ///
    /// Declare the accesses above and synchronize other host/device accesses and queue ownership.
    /// Keep inputs unchanged during reads. Recording does not retain address-only resources or
    /// insert dependencies between builds in the batch.
    ///
    /// ```compile_fail,E0133
    /// # use vk_graph::cmd::{BuildMicromapInfo, CommandRef};
    /// fn record(cmd: &CommandRef<'_>, infos: &[BuildMicromapInfo]) {
    ///     cmd.build_micromaps(infos); // Execution-time memory validity requires an unsafe call.
    /// }
    /// ```
    pub unsafe fn build_micromaps(&self, infos: &[BuildMicromapInfo]) -> &Self {
        let info_count = vk_count(infos.len(), "micromap build info count");

        #[cfg(feature = "checked")]
        assert!(
            !infos.is_empty(),
            "micromap build info count must be nonzero"
        );

        let vk_infos = infos
            .iter()
            .map(|info| {
                let destination = self.resource(info.micromap);

                #[cfg(feature = "checked")]
                self.validate_micromap_build(destination.info.micromap_type, &info.build_data);

                info.build_data.to_vk(destination.handle)
            })
            .collect::<Vec<_>>();
        let ext = Device::expect_vk_ext_opacity_micromap(&self.cmd.device);

        unsafe {
            (ext.fp().cmd_build_micromaps_ext)(self.cmd.handle, info_count, vk_infos.as_ptr());
        }

        self
    }

    /// Clones or compacts one micromap into another.
    ///
    /// Graph users must declare [`vk_sync::AccessType::MicromapBuildRead`] on `source` and
    /// [`vk_sync::AccessType::MicromapBuildWrite`] on `destination`.
    ///
    /// # Safety
    ///
    /// `source` must have been successfully constructed and `destination` must have enough backing
    /// storage for the result. For [`MicromapCopyMode::Compact`], `source` must have been built with
    /// [`vk::BuildMicromapFlagsEXT::ALLOW_COMPACTION`]. These conditions must hold at execution time;
    /// clone storage must cover the source's storage size, or compact storage its queried compacted
    /// size. Both micromaps must belong to this device and have nonoverlapping bound storage that
    /// remains alive through execution. Declare the accesses above and synchronize all other
    /// host/device accesses and queue ownership transfers. All `vkCmdCopyMicromapEXT` validity
    /// requirements must be satisfied.
    ///
    /// See [`vkCmdCopyMicromapEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyMicromapEXT.html).
    pub unsafe fn copy_micromap(&self, info: &CopyMicromapInfo) -> &Self {
        let source = self.resource(info.source);
        let destination = self.resource(info.destination);

        #[cfg(feature = "checked")]
        {
            assert_opacity_micromap(source.info.micromap_type, "source");
            assert_opacity_micromap(destination.info.micromap_type, "destination");
        }

        let vk_info = marshal_copy_micromap(source.handle, destination.handle, info.mode);
        let ext = Device::expect_vk_ext_opacity_micromap(&self.cmd.device);
        unsafe {
            (ext.fp().cmd_copy_micromap_ext)(self.cmd.handle, &vk_info);
        }

        self
    }

    /// Serializes a micromap to a buffer device address.
    ///
    /// Graph users must declare [`vk_sync::AccessType::MicromapBuildRead`] on `source` and
    /// [`vk_sync::AccessType::MicromapBuildBufferWrite`] on the destination buffer separately. The
    /// destination address must have enough storage for the serialization-size property.
    ///
    /// # Safety
    ///
    /// `source` must have been successfully constructed. `destination` must identify a live,
    /// 256-byte-aligned buffer range large enough for the complete serialized representation.
    /// At execution time, the range must be writable, bound on this device, have
    /// `SHADER_DEVICE_ADDRESS` usage, and not overlap the source storage. Both resources and their
    /// memory must remain alive through execution. Declare the accesses above and synchronize all
    /// other host/device accesses and queue ownership transfers. All
    /// `vkCmdCopyMicromapToMemoryEXT` validity requirements must be satisfied.
    ///
    /// See [`vkCmdCopyMicromapToMemoryEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyMicromapToMemoryEXT.html).
    pub unsafe fn serialize_micromap(&self, info: &SerializeMicromapInfo) -> &Self {
        #[cfg(feature = "checked")]
        assert_device_address(info.destination, 256, "micromap serialization destination");

        let source = self.resource(info.source);
        #[cfg(feature = "checked")]
        assert_opacity_micromap(source.info.micromap_type, "source");

        let vk_info = marshal_serialize_micromap(source.handle, info.destination);
        let ext = Device::expect_vk_ext_opacity_micromap(&self.cmd.device);
        unsafe {
            (ext.fp().cmd_copy_micromap_to_memory_ext)(self.cmd.handle, &vk_info);
        }

        self
    }

    /// Deserializes a buffer device address into a micromap.
    ///
    /// Graph users must declare [`vk_sync::AccessType::MicromapBuildBufferRead`] on the source
    /// buffer and [`vk_sync::AccessType::MicromapBuildWrite`] on `destination` separately.
    ///
    /// # Safety
    ///
    /// `source` must identify a live, 256-byte-aligned buffer range containing a complete serialized
    /// representation reported compatible by [`crate::driver::micromap::Micromap::compatibility`].
    /// The destination micromap must have enough backing storage for that representation.
    /// The representation must be unmodified output of Vulkan serialization.
    /// These conditions must hold at execution time. The source range must be readable, bound on
    /// this device with `SHADER_DEVICE_ADDRESS` usage, and not overlap the destination storage.
    /// Both resources and their memory must remain alive through execution, and serialized data
    /// must remain unchanged during reads. Declare the accesses above and synchronize all other
    /// host/device accesses and queue ownership transfers. All `vkCmdCopyMemoryToMicromapEXT`
    /// validity requirements must be satisfied.
    ///
    /// See [`vkCmdCopyMemoryToMicromapEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyMemoryToMicromapEXT.html).
    pub unsafe fn deserialize_micromap(&self, info: &DeserializeMicromapInfo) -> &Self {
        #[cfg(feature = "checked")]
        assert_device_address(info.source, 256, "serialized micromap source");

        let destination = self.resource(info.destination);
        #[cfg(feature = "checked")]
        assert_opacity_micromap(destination.info.micromap_type, "destination");

        let vk_info = marshal_deserialize_micromap(info.source, destination.handle);
        let ext = Device::expect_vk_ext_opacity_micromap(&self.cmd.device);
        unsafe {
            (ext.fp().cmd_copy_memory_to_micromap_ext)(self.cmd.handle, &vk_info);
        }

        self
    }

    /// Writes micromap properties to consecutive query-pool slots.
    ///
    /// Graph users must declare [`vk_sync::AccessType::MicromapBuildRead`] for every micromap. The
    /// query pool is a raw Vulkan object and is not tracked by the graph; it must match `query_type`,
    /// have enough unavailable slots starting at `first_query`, and be reset as required by Vulkan.
    ///
    /// # Safety
    ///
    /// Every micromap must have been successfully constructed. For
    /// [`MicromapQueryType::CompactedSize`], every micromap must have been built with
    /// [`vk::BuildMicromapFlagsEXT::ALLOW_COMPACTION`]. `query_pool` must remain valid until command
    /// execution completes and satisfy the documented query type, capacity, and reset requirements.
    /// These conditions must hold at execution time; the micromaps and pool must belong to this
    /// device, and all storage must remain alive and bound through execution. The micromap list
    /// must be nonempty. Declare the accesses above and synchronize all other host/device accesses,
    /// query-pool resets, result reads, and queue ownership transfers. All
    /// `vkCmdWriteMicromapsPropertiesEXT` validity requirements must be satisfied.
    ///
    /// See [`vkCmdWriteMicromapsPropertiesEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdWriteMicromapsPropertiesEXT.html).
    pub unsafe fn write_micromaps_properties(&self, info: &WriteMicromapsPropertiesInfo) -> &Self {
        let micromap_count = vk_count(info.micromaps.len(), "micromap property query count");

        #[cfg(feature = "checked")]
        {
            assert!(
                !info.micromaps.is_empty(),
                "micromap property query count must be nonzero"
            );
            assert!(
                info.first_query.checked_add(micromap_count).is_some(),
                "micromap property query range overflows u32"
            );
            assert_ne!(
                info.query_pool,
                vk::QueryPool::null(),
                "micromap property query pool must be non-null"
            );
        }

        let micromaps = info
            .micromaps
            .iter()
            .map(|&node| {
                let micromap = self.resource(node);
                #[cfg(feature = "checked")]
                assert_opacity_micromap(micromap.info.micromap_type, "queried micromap");
                micromap.handle
            })
            .collect::<Vec<_>>();
        let query_type = info.query_type.into();

        #[cfg(feature = "checked")]
        assert!(
            matches!(
                query_type,
                vk::QueryType::MICROMAP_COMPACTED_SIZE_EXT
                    | vk::QueryType::MICROMAP_SERIALIZATION_SIZE_EXT
            ),
            "unsupported micromap query type"
        );

        let ext = Device::expect_vk_ext_opacity_micromap(&self.cmd.device);
        unsafe {
            (ext.fp().cmd_write_micromaps_properties_ext)(
                self.cmd.handle,
                micromap_count,
                micromaps.as_ptr(),
                query_type,
                info.query_pool,
                info.first_query,
            );
        }

        self
    }

    #[cfg(feature = "checked")]
    fn validate_micromap_build(
        &self,
        micromap_type: vk::MicromapTypeEXT,
        info: &OpacityMicromapBuildInfo,
    ) {
        assert_opacity_micromap(micromap_type, "build destination");
        let scratch_alignment =
            self.cmd
                .device
                .physical
                .vk_khr_acceleration_structure
                .as_ref()
                .expect("VK_EXT_opacity_micromap requires VK_KHR_acceleration_structure")
                .properties
                .min_accel_struct_scratch_offset_alignment as vk::DeviceSize;

        assert_device_or_host_address(info.data, 256, "opacity data", false);
        let allow_null_scratch = matches!(info.scratch_data, DeviceOrHostAddress::DeviceAddress(0))
            && crate::driver::micromap::Micromap::size_of_build_type(
                &self.cmd.device,
                info,
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
            )
            .build_size
                == 0;
        assert_device_or_host_address(
            info.scratch_data,
            scratch_alignment,
            "micromap build scratch",
            allow_null_scratch,
        );
        assert_device_or_host_address(info.triangle_array, 256, "micromap triangle array", false);
        assert!(
            info.triangle_array_stride >= std::mem::size_of::<vk::MicromapTriangleEXT>() as u64
                && info
                    .triangle_array_stride
                    .is_multiple_of(std::mem::align_of::<vk::MicromapTriangleEXT>() as u64),
            "micromap triangle array stride must fit and align VkMicromapTriangleEXT"
        );
        let _ = vk_count(info.usage_counts().len(), "micromap usage count");
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
    pub fn update_accel_struct<D: Clone + Into<AccelerationStructureGeometryDataExt>>(
        &self,
        infos: &[UpdateAccelerationStructureInfo<D>],
    ) -> &Self {
        #[derive(Default)]
        struct Tls {
            ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
        }

        thread_local! {
            static TLS: RefCell<Tls> = Default::default();
        }

        TLS.with_borrow_mut(|tls| {
            let geometries =
                AccelerationStructureGeometryMarshaler::new(infos.iter().flat_map(|info| {
                    info.update_data
                        .geometries
                        .iter()
                        .map(|(geometry, _)| geometry)
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
                            .geometries(&geometries.geometries()[start..end])
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
    pub fn update_accel_struct_indirect<D: Clone + Into<AccelerationStructureGeometryDataExt>>(
        &self,
        infos: &[UpdateAccelerationStructureIndirectInfo<D>],
    ) -> &Self {
        #[derive(Default)]
        struct Tls {
            max_primitive_counts: Vec<u32>,
            range_bases: Vec<vk::DeviceAddress>,
            range_strides: Vec<u32>,
        }

        thread_local! {
            static TLS: RefCell<Tls> = Default::default();
        }

        TLS.with_borrow_mut(|tls| {
            let geometries = AccelerationStructureGeometryMarshaler::new(
                infos
                    .iter()
                    .flat_map(|info| info.update_data.geometries.iter()),
            );

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
                            .geometries(&geometries.geometries()[start..end])
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

    /// Returns a borrow of the resource or persistent resource set represented by the given node.
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: ResourceNode,
    {
        #[cfg(feature = "checked")]
        resource_node.assert_owner(self.graph_id);

        let index = match resource_node.resource_node_index() {
            ResourceNodeIndex::Resource(mut node_idx) => {
                if let Some(node_map) = self.node_map {
                    node_idx = node_map[node_idx];
                }

                #[cfg(feature = "checked")]
                assert!(
                    self.exec.accesses.contains(node_idx),
                    "unexpected node access: call an access function first"
                );

                ResourceNodeIndex::Resource(node_idx)
            }
            ResourceNodeIndex::ResourceSet(mut resource_set_idx) => {
                if let Some(resource_set_map) = self.resource_set_map {
                    resource_set_idx = resource_set_map[resource_set_idx.as_usize()];
                }

                #[cfg(feature = "checked")]
                assert!(
                    self.exec
                        .resource_set_accesses
                        .iter()
                        .any(|access| access.resource_set_idx == resource_set_idx),
                    "unexpected resource set access: call an access function first"
                );

                ResourceNodeIndex::ResourceSet(resource_set_idx)
            }
        };

        resource_node.borrow_at(self.resources, self.resource_sets, index)
    }
}

impl<'a> Deref for CommandRef<'a> {
    type Target = crate::driver::cmd_buf::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        self.cmd
    }
}

fn vk_count(len: usize, description: &str) -> u32 {
    #[cfg(feature = "checked")]
    return u32::try_from(len)
        .unwrap_or_else(|_| panic!("{description} ({len}) exceeds Vulkan's u32 limit"));

    #[cfg(not(feature = "checked"))]
    {
        let _ = description;
        len as u32
    }
}

#[cfg(feature = "checked")]
fn assert_device_address(address: vk::DeviceAddress, alignment: vk::DeviceSize, description: &str) {
    assert_ne!(address, 0, "{description} device address must be nonzero");
    assert!(
        address.is_multiple_of(alignment.max(1)),
        "{description} device address must be aligned to {} bytes",
        alignment.max(1)
    );
}

#[cfg(feature = "checked")]
fn assert_device_or_host_address(
    address: DeviceOrHostAddress,
    alignment: vk::DeviceSize,
    description: &str,
    allow_null: bool,
) {
    let DeviceOrHostAddress::DeviceAddress(address) = address else {
        panic!("{description} must be a device address for a GPU micromap command");
    };
    if address != 0 || !allow_null {
        assert_device_address(address, alignment, description);
    }
}

#[cfg(feature = "checked")]
fn assert_opacity_micromap(micromap_type: vk::MicromapTypeEXT, description: &str) {
    assert_eq!(
        micromap_type,
        vk::MicromapTypeEXT::OPACITY_MICROMAP,
        "{description} must be an opacity micromap"
    );
}

fn marshal_copy_micromap(
    source: vk::MicromapEXT,
    destination: vk::MicromapEXT,
    mode: MicromapCopyMode,
) -> vk::CopyMicromapInfoEXT<'static> {
    vk::CopyMicromapInfoEXT::default()
        .src(source)
        .dst(destination)
        .mode(mode.to_vk())
}

fn marshal_serialize_micromap(
    source: vk::MicromapEXT,
    destination: vk::DeviceAddress,
) -> vk::CopyMicromapToMemoryInfoEXT<'static> {
    vk::CopyMicromapToMemoryInfoEXT::default()
        .src(source)
        .dst(vk::DeviceOrHostAddressKHR {
            device_address: destination,
        })
        .mode(vk::CopyMicromapModeEXT::SERIALIZE)
}

fn marshal_deserialize_micromap(
    source: vk::DeviceAddress,
    destination: vk::MicromapEXT,
) -> vk::CopyMemoryToMicromapInfoEXT<'static> {
    vk::CopyMemoryToMicromapInfoEXT::default()
        .src(vk::DeviceOrHostAddressConstKHR {
            device_address: source,
        })
        .dst(destination)
        .mode(vk::CopyMicromapModeEXT::DESERIALIZE)
}

/// Specifies one opacity micromap build in a device batch.
///
/// The build data owns its usage-count array, keeping `pUsageCounts` stable while Vulkan consumes
/// the command. Buffer resources corresponding to its device addresses are deliberately not stored
/// here and must be declared to the graph separately; see [`CommandRef::build_micromaps`].
#[derive(Clone, Debug)]
pub struct BuildMicromapInfo {
    /// Owned opacity build parameters and usage counts.
    pub build_data: OpacityMicromapBuildInfo,

    /// Destination opacity micromap.
    pub micromap: AnyMicromapNode,
}

impl BuildMicromapInfo {
    /// Creates one device opacity-micromap build description.
    pub fn new(micromap: impl Into<AnyMicromapNode>, build_data: OpacityMicromapBuildInfo) -> Self {
        let micromap = micromap.into();

        Self {
            build_data,
            micromap,
        }
    }
}

/// Valid modes for a device-to-device micromap copy.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MicromapCopyMode {
    /// Copies the source without changing its representation.
    Clone,

    /// Writes a compacted representation of a source built with
    /// [`vk::BuildMicromapFlagsEXT::ALLOW_COMPACTION`].
    Compact,
}

impl MicromapCopyMode {
    const fn to_vk(self) -> vk::CopyMicromapModeEXT {
        match self {
            Self::Clone => vk::CopyMicromapModeEXT::CLONE,
            Self::Compact => vk::CopyMicromapModeEXT::COMPACT,
        }
    }
}

/// Specifies a clone or compact operation between two micromaps.
#[derive(Clone, Copy, Debug)]
pub struct CopyMicromapInfo {
    /// Destination micromap written by the copy.
    pub destination: AnyMicromapNode,

    /// Whether to clone or compact the source.
    pub mode: MicromapCopyMode,

    /// Source micromap read by the copy.
    pub source: AnyMicromapNode,
}

impl CopyMicromapInfo {
    /// Creates a constrained micromap copy description.
    pub fn new(
        source: impl Into<AnyMicromapNode>,
        destination: impl Into<AnyMicromapNode>,
        mode: MicromapCopyMode,
    ) -> Self {
        let source = source.into();
        let destination = destination.into();

        Self {
            destination,
            mode,
            source,
        }
    }

    /// Creates a clone operation.
    pub fn new_clone(
        source: impl Into<AnyMicromapNode>,
        destination: impl Into<AnyMicromapNode>,
    ) -> Self {
        Self::new(source, destination, MicromapCopyMode::Clone)
    }

    /// Creates a compaction operation. The source must have been built with
    /// [`vk::BuildMicromapFlagsEXT::ALLOW_COMPACTION`].
    pub fn compact(
        source: impl Into<AnyMicromapNode>,
        destination: impl Into<AnyMicromapNode>,
    ) -> Self {
        Self::new(source, destination, MicromapCopyMode::Compact)
    }
}

/// Specifies serialization of a micromap to a device address.
#[derive(Clone, Copy, Debug)]
pub struct SerializeMicromapInfo {
    /// 256-byte-aligned destination buffer device address.
    pub destination: vk::DeviceAddress,

    /// Source micromap.
    pub source: AnyMicromapNode,
}

impl SerializeMicromapInfo {
    /// Creates a device serialization description.
    pub fn new(source: impl Into<AnyMicromapNode>, destination: vk::DeviceAddress) -> Self {
        let source = source.into();

        Self {
            destination,
            source,
        }
    }
}

/// Specifies deserialization from a device address into a micromap.
#[derive(Clone, Copy, Debug)]
pub struct DeserializeMicromapInfo {
    /// Destination micromap.
    pub destination: AnyMicromapNode,

    /// 256-byte-aligned source buffer device address.
    pub source: vk::DeviceAddress,
}

impl DeserializeMicromapInfo {
    /// Creates a device deserialization description.
    pub fn new(source: vk::DeviceAddress, destination: impl Into<AnyMicromapNode>) -> Self {
        let destination = destination.into();

        Self {
            destination,
            source,
        }
    }
}

/// Micromap properties supported by Vulkan query commands.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MicromapQueryType {
    /// Size required for a compacted copy. The queried micromap must have been built with
    /// [`vk::BuildMicromapFlagsEXT::ALLOW_COMPACTION`].
    CompactedSize,

    /// Size required for a serialized representation.
    SerializationSize,
}

impl From<MicromapQueryType> for vk::QueryType {
    fn from(value: MicromapQueryType) -> Self {
        match value {
            MicromapQueryType::CompactedSize => Self::MICROMAP_COMPACTED_SIZE_EXT,
            MicromapQueryType::SerializationSize => Self::MICROMAP_SERIALIZATION_SIZE_EXT,
        }
    }
}

/// Owned input for [`CommandRef::write_micromaps_properties`].
#[derive(Clone, Debug)]
pub struct WriteMicromapsPropertiesInfo {
    /// First query-pool slot written by the command.
    pub first_query: u32,

    /// Micromaps whose properties are written to consecutive query slots.
    pub micromaps: Box<[AnyMicromapNode]>,

    /// Query pool created for `query_type`.
    pub query_pool: vk::QueryPool,

    /// Property written for every micromap. A compacted-size query requires every micromap to have
    /// been built with [`vk::BuildMicromapFlagsEXT::ALLOW_COMPACTION`].
    pub query_type: MicromapQueryType,
}

impl WriteMicromapsPropertiesInfo {
    /// Creates an owned micromap property-query description.
    pub fn new(
        micromaps: impl IntoIterator<Item = impl Into<AnyMicromapNode>>,
        query_type: MicromapQueryType,
        query_pool: vk::QueryPool,
        first_query: u32,
    ) -> Self {
        let micromaps = micromaps.into_iter().map(Into::into).collect();

        Self {
            first_query,
            micromaps,
            query_pool,
            query_type,
        }
    }
}

/// Specifies the information and data used to build an acceleration structure.
///
/// See [`vkCmdBuildAccelerationStructuresKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBuildAccelerationStructuresKHR.html).
#[derive(Clone, Debug)]
pub struct BuildAccelerationStructureInfo<D = AccelerationStructureGeometryData> {
    /// The acceleration structure to be written.
    pub accel_struct: AnyAccelerationStructureNode,

    /// Specifies the geometry data to use when building the acceleration structure.
    pub build_data: AccelerationStructureGeometryInfo<(
        AccelerationStructureGeometry<D>,
        vk::AccelerationStructureBuildRangeInfoKHR,
    )>,

    /// The temporary buffer or host address (with enough capacity per
    /// `AccelerationStructure::size_of`).
    pub scratch_addr: DeviceOrHostAddress,
}

impl<D> BuildAccelerationStructureInfo<D> {
    /// Constructs new acceleration structure build information.
    pub fn new(
        accel_struct: impl Into<AnyAccelerationStructureNode>,
        scratch_addr: impl Into<DeviceOrHostAddress>,
        build_data: AccelerationStructureGeometryInfo<(
            AccelerationStructureGeometry<D>,
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
pub struct BuildAccelerationStructureIndirectInfo<D = AccelerationStructureGeometryData> {
    /// The acceleration structure to be written.
    pub accel_struct: AnyAccelerationStructureNode,

    /// Specifies the geometry data to use when building the acceleration structure.
    pub build_data: AccelerationStructureGeometryInfo<AccelerationStructureGeometry<D>>,

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

impl<D> BuildAccelerationStructureIndirectInfo<D> {
    /// Constructs new acceleration structure indirect build information.
    pub fn new(
        accel_struct: impl Into<AnyAccelerationStructureNode>,
        scratch_data: impl Into<DeviceOrHostAddress>,
        build_data: AccelerationStructureGeometryInfo<AccelerationStructureGeometry<D>>,
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
pub struct UpdateAccelerationStructureIndirectInfo<D = AccelerationStructureGeometryData> {
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
    pub update_data: AccelerationStructureGeometryInfo<AccelerationStructureGeometry<D>>,
}

impl<D> UpdateAccelerationStructureIndirectInfo<D> {
    /// Constructs new acceleration structure indirect update information.
    pub fn new(
        src_accel_struct: impl Into<AnyAccelerationStructureNode>,
        dst_accel_struct: impl Into<AnyAccelerationStructureNode>,
        scratch_addr: impl Into<DeviceOrHostAddress>,
        update_data: AccelerationStructureGeometryInfo<AccelerationStructureGeometry<D>>,
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
pub struct UpdateAccelerationStructureInfo<D = AccelerationStructureGeometryData> {
    /// The acceleration structure to be written.
    pub dst_accel_struct: AnyAccelerationStructureNode,

    /// The temporary buffer or host address (with enough capacity per
    /// `AccelerationStructure::size_of`).
    pub scratch_addr: DeviceOrHostAddress,

    /// The source acceleration structure to be read.
    pub src_accel_struct: AnyAccelerationStructureNode,

    /// Specifies the geometry data to use when updating the acceleration structure.
    pub update_data: AccelerationStructureGeometryInfo<(
        AccelerationStructureGeometry<D>,
        vk::AccelerationStructureBuildRangeInfoKHR,
    )>,
}

impl<D> UpdateAccelerationStructureInfo<D> {
    /// Constructs new acceleration structure update information.
    pub fn new(
        src_accel_struct: impl Into<AnyAccelerationStructureNode>,
        dst_accel_struct: impl Into<AnyAccelerationStructureNode>,
        scratch_addr: impl Into<DeviceOrHostAddress>,
        update_data: AccelerationStructureGeometryInfo<(
            AccelerationStructureGeometry<D>,
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::driver::micromap::OpacityMicromapUsage;
    use ash::vk::Handle;

    #[cfg(feature = "checked")]
    #[test]
    fn micromap_device_null_scratch_requires_zero_scratch_size() {
        let validate = |address, allow_null| {
            assert_device_or_host_address(address, 256, "micromap build scratch", allow_null);
        };
        validate(DeviceOrHostAddress::DeviceAddress(0), true);
        validate(DeviceOrHostAddress::DeviceAddress(256), false);
        for (address, allow_null) in [
            (DeviceOrHostAddress::DeviceAddress(0), false),
            (DeviceOrHostAddress::DeviceAddress(1), true),
            (DeviceOrHostAddress::HostAddress(std::ptr::null_mut()), true),
        ] {
            assert!(std::panic::catch_unwind(|| validate(address, allow_null)).is_err());
        }
    }

    #[test]
    fn opacity_build_marshaling_keeps_owned_usage_pointer_stable() {
        let build = OpacityMicromapBuildInfo::new([
            OpacityMicromapUsage::new(3, 1, vk::OpacityMicromapFormatEXT::TYPE_2_STATE),
            OpacityMicromapUsage::new(5, 2, vk::OpacityMicromapFormatEXT::TYPE_4_STATE),
        ])
        .data(0x100_u64)
        .scratch_data(0x200_u64)
        .triangle_array(
            0x300_u64,
            std::mem::size_of::<vk::MicromapTriangleEXT>() as u64,
        );
        let destination = vk::MicromapEXT::from_raw(7);

        let before_move = build.to_vk(destination).p_usage_counts;
        let builds = [build];
        let marshaled = builds
            .iter()
            .map(|build| build.to_vk(destination))
            .collect::<Vec<_>>();

        assert_eq!(marshaled[0].mode, vk::BuildMicromapModeEXT::BUILD);
        assert_eq!(marshaled[0].ty, vk::MicromapTypeEXT::OPACITY_MICROMAP);
        assert_eq!(marshaled[0].usage_counts_count, 2);
        assert_eq!(marshaled[0].p_usage_counts, before_move);
        assert!(marshaled[0].pp_usage_counts.is_null());
    }

    #[test]
    fn copy_modes_marshal_only_clone_and_compact() {
        let source = vk::MicromapEXT::from_raw(11);
        let destination = vk::MicromapEXT::from_raw(13);

        let cloned = marshal_copy_micromap(source, destination, MicromapCopyMode::Clone);
        let compacted = marshal_copy_micromap(source, destination, MicromapCopyMode::Compact);

        assert_eq!(cloned.src, source);
        assert_eq!(cloned.dst, destination);
        assert_eq!(cloned.mode, vk::CopyMicromapModeEXT::CLONE);
        assert_eq!(compacted.mode, vk::CopyMicromapModeEXT::COMPACT);
    }

    #[test]
    fn micromap_clone_constructor_does_not_shadow_clone_trait() {
        use crate::Node;

        #[cfg(feature = "checked")]
        let graph_id = crate::GraphId::next();
        let source = crate::MicromapNode::new(
            0,
            #[cfg(feature = "checked")]
            graph_id,
        );
        let destination = crate::MicromapNode::new(
            1,
            #[cfg(feature = "checked")]
            graph_id,
        );
        let info = CopyMicromapInfo::new_clone(source, destination);
        #[allow(clippy::clone_on_copy)]
        let cloned = info.clone();

        assert_eq!(cloned.source.index(), 0);
        assert_eq!(cloned.destination.index(), 1);
        assert_eq!(cloned.mode, MicromapCopyMode::Clone);
        assert_eq!(
            CopyMicromapInfo::compact(source, destination).mode,
            MicromapCopyMode::Compact
        );
    }

    #[test]
    fn device_memory_copy_marshaling_uses_required_modes_and_addresses() {
        let micromap = vk::MicromapEXT::from_raw(17);
        let serialized = marshal_serialize_micromap(micromap, 0x100);
        let deserialized = marshal_deserialize_micromap(0x200, micromap);

        assert_eq!(serialized.src, micromap);
        assert_eq!(serialized.mode, vk::CopyMicromapModeEXT::SERIALIZE);
        assert_eq!(deserialized.dst, micromap);
        assert_eq!(deserialized.mode, vk::CopyMicromapModeEXT::DESERIALIZE);
        unsafe {
            assert_eq!(serialized.dst.device_address, 0x100);
            assert_eq!(deserialized.src.device_address, 0x200);
        }
    }

    #[test]
    fn property_query_types_marshal_to_supported_vulkan_values() {
        assert_eq!(
            vk::QueryType::from(MicromapQueryType::CompactedSize),
            vk::QueryType::MICROMAP_COMPACTED_SIZE_EXT
        );
        assert_eq!(
            vk::QueryType::from(MicromapQueryType::SerializationSize),
            vk::QueryType::MICROMAP_SERIALIZATION_SIZE_EXT
        );
        assert_eq!(vk_count(4, "test count"), 4);
    }
}
