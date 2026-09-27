use {
    crate::{
        AnyAccelerationStructureNode, AnyMicromapNode, AnyResource, Execution, ResourceNode,
        driver::{
            accel_struct::{AccelerationStructureGeometry, AccelerationStructureGeometryMarshaler},
            device::Device,
            micromap::OpacityMicromapUsage,
        },
        private::ResourceNodeIndex,
        resource::{ResourceSetIndex, ResourceSetMap},
        stream::StreamValueArg,
    },
    ash::vk,
    log::trace,
    std::ops::Deref,
};

/// Device build parameters shared by direct and indirect acceleration-structure commands.
///
/// Geometry and usage-count slices are borrowed only during recording, not GPU execution.
/// Device-address resources must be declared separately; see
/// [`CommandRef::build_acceleration_structures`].
///
/// ```compile_fail,E0308
/// # use vk_graph::cmd::AccelerationStructureBuildGeometryInfo;
/// fn host_pointer(info: &mut AccelerationStructureBuildGeometryInfo<'_>) {
///     info.scratch_data = std::ptr::null_mut::<std::ffi::c_void>();
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct AccelerationStructureBuildGeometryInfo<'a> {
    /// Acceleration structure type to build; must not be `GENERIC`.
    pub acceleration_structure_type: vk::AccelerationStructureTypeKHR,

    /// Destination, which may equal the source for an in-place update.
    pub dst_acceleration_structure: AnyAccelerationStructureNode,

    /// Build flags, compatible with the source's original build when updating.
    pub flags: vk::BuildAccelerationStructureFlagsKHR,

    /// Geometry descriptions in the same order as build ranges or maximum primitive counts.
    pub geometries: &'a [AccelerationStructureGeometry<'a>],

    /// `BUILD` or `UPDATE`, selected independently for each batch entry.
    pub mode: vk::BuildAccelerationStructureModeKHR,

    /// Device address of scratch storage sized for the selected mode.
    pub scratch_data: vk::DeviceAddress,

    /// Required for `UPDATE`; ignored for `BUILD`.
    pub src_acceleration_structure: Option<AnyAccelerationStructureNode>,
}

impl<'a> AccelerationStructureBuildGeometryInfo<'a> {
    /// Creates a `BUILD` description with no source acceleration structure.
    pub fn build(
        acceleration_structure_type: vk::AccelerationStructureTypeKHR,
        flags: vk::BuildAccelerationStructureFlagsKHR,
        dst_acceleration_structure: impl Into<AnyAccelerationStructureNode>,
        geometries: &'a [AccelerationStructureGeometry<'a>],
        scratch_data: vk::DeviceAddress,
    ) -> Self {
        Self {
            acceleration_structure_type,
            dst_acceleration_structure: dst_acceleration_structure.into(),
            flags,
            geometries,
            mode: vk::BuildAccelerationStructureModeKHR::BUILD,
            scratch_data,
            src_acceleration_structure: None,
        }
    }

    /// Creates an `UPDATE` description. The source must have been built with `ALLOW_UPDATE`.
    pub fn update(
        acceleration_structure_type: vk::AccelerationStructureTypeKHR,
        flags: vk::BuildAccelerationStructureFlagsKHR,
        src_acceleration_structure: impl Into<AnyAccelerationStructureNode>,
        dst_acceleration_structure: impl Into<AnyAccelerationStructureNode>,
        geometries: &'a [AccelerationStructureGeometry<'a>],
        scratch_data: vk::DeviceAddress,
    ) -> Self {
        Self {
            mode: vk::BuildAccelerationStructureModeKHR::UPDATE,
            src_acceleration_structure: Some(src_acceleration_structure.into()),
            ..Self::build(
                acceleration_structure_type,
                flags,
                dst_acceleration_structure,
                geometries,
                scratch_data,
            )
        }
    }
}

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
    stream_values: Option<&'a crate::stream::StreamValues>,
}

impl<'a> CommandRef<'a> {
    pub(crate) fn new(
        cmd: &'a crate::driver::cmd_buf::CommandBuffer,
        resources: &'a [AnyResource],
        resource_sets: &'a ResourceSetMap,
        exec: &'a Execution,
        stream_values: Option<&'a crate::stream::StreamValues>,
        #[cfg(feature = "checked")] graph_id: crate::GraphId,
    ) -> Self {
        Self {
            cmd,
            node_map: exec.node_map.as_deref(),
            resource_set_map: exec.resource_set_map.as_deref(),
            resource_sets,
            resources,
            stream_values: exec.stream_values.as_deref().or(stream_values),

            #[cfg(feature = "checked")]
            exec,

            #[cfg(feature = "checked")]
            graph_id: exec.stream_graph_id.unwrap_or(graph_id),
        }
    }

    /// Builds or updates a batch of acceleration structures, with one range per geometry.
    ///
    /// Entries may mix `BUILD` and `UPDATE`. No ordering or synchronization is implied between entries.
    /// Slice lengths and whether Vulkan counts fit in `u32` are always checked.
    ///
    /// # Safety
    ///
    /// All `vkCmdBuildAccelerationStructuresKHR` validity requirements must hold. Through execution,
    /// geometry inputs, referenced acceleration structures and micromaps, destinations, and scratch
    /// must remain alive and bound on this device with the required usage, alignment, and capacity.
    /// Scratch must cover the matching `DEVICE` or `HOST_OR_DEVICE` query's build or update scratch size.
    /// Inputs must remain unchanged while read. `UPDATE` requires a successfully built source with
    /// `ALLOW_UPDATE` and Vulkan-compatible flags, geometry, primitive counts, and micromap state.
    /// Source and destination may be identical for an in-place update; otherwise their storage must
    /// not overlap. Scratch must not overlap inputs or acceleration-structure storage; destinations
    /// and scratch ranges of separate entries must not overlap or depend on another entry's writes.
    ///
    /// Explicitly declare graph accesses: `AccelerationStructureBuildInputRead` for input buffers,
    /// `AccelerationStructureBuildRead` for source and referenced structures, `AccelerationStructureBuildWrite`
    /// for destinations, `AccelerationStructureBuildScratchReadWrite` for scratch, and
    /// `AccelerationStructureBuildMicromapRead` for attached micromaps. Synchronize other accesses and
    /// queue ownership. Address-only resources are not retained or inferred by recording. A micromap
    /// may only be discarded when Vulkan permits it, including the matching size query's discardable
    /// result; otherwise keep it alive for subsequent acceleration-structure use.
    ///
    /// ```compile_fail,E0133
    /// # use vk_graph::cmd::{AccelerationStructureBuildGeometryInfo, CommandRef};
    /// # use ash::vk;
    /// fn record(cmd: &CommandRef<'_>, infos: &[AccelerationStructureBuildGeometryInfo<'_>],
    ///           ranges: &[&[vk::AccelerationStructureBuildRangeInfoKHR]]) {
    ///     cmd.build_acceleration_structures(infos, ranges);
    /// }
    /// ```
    pub unsafe fn build_acceleration_structures(
        &self,
        infos: &[AccelerationStructureBuildGeometryInfo<'_>],
        build_range_infos: &[&[vk::AccelerationStructureBuildRangeInfoKHR]],
    ) -> &Self {
        validate_build_shape(infos, build_range_infos, None);
        let geometries = AccelerationStructureGeometryMarshaler::new(
            infos.iter().flat_map(|info| info.geometries.iter()),
        );
        let vk_infos =
            marshal_acceleration_structure_builds(infos, geometries.geometries(), |node| {
                self.resource(node).handle
            });
        let ext = Device::expect_vk_khr_acceleration_structure(&self.cmd.device);

        unsafe {
            ext.cmd_build_acceleration_structures(self.cmd.handle, &vk_infos, build_range_infos);
        }

        self
    }

    /// Builds or updates acceleration structures using device-resident build ranges.
    ///
    /// Each address points to one strided `AccelerationStructureBuildRangeInfoKHR` per geometry.
    /// Entries may mix `BUILD` and `UPDATE`; slice lengths and Vulkan counts are always checked.
    ///
    /// # Safety
    ///
    /// The safety requirements of [`Self::build_acceleration_structures`] also apply here, along
    /// with all `vkCmdBuildAccelerationStructuresIndirectKHR` validity requirements. The
    /// `accelerationStructureIndirectBuild` feature must be enabled. Indirect buffers must remain
    /// alive, bound, and unchanged during reads, with `INDIRECT_BUFFER` and `SHADER_DEVICE_ADDRESS` usage. Addresses and
    /// strides must be multiples of four, and each range's primitive count must not exceed its
    /// corresponding maximum. Declare [`vk_sync::AccessType::AccelerationStructureBuildIndirectRead`]
    /// for range buffers and synchronize their producers. This uses `ACCELERATION_STRUCTURE_BUILD_KHR`
    /// and `INDIRECT_COMMAND_READ`; `IndirectBuffer` uses the wrong stage for these reads.
    /// `General` remains a valid conservative fallback (`ALL_COMMANDS`, `MEMORY_READ | MEMORY_WRITE`).
    /// Size queries must cover the supplied maximum primitive counts.
    ///
    /// ```compile_fail,E0133
    /// # use vk_graph::cmd::{AccelerationStructureBuildGeometryInfo, CommandRef};
    /// fn record(cmd: &CommandRef<'_>, infos: &[AccelerationStructureBuildGeometryInfo<'_>]) {
    ///     cmd.build_acceleration_structures_indirect(infos, &[], &[], &[]);
    /// }
    /// ```
    pub unsafe fn build_acceleration_structures_indirect(
        &self,
        infos: &[AccelerationStructureBuildGeometryInfo<'_>],
        indirect_device_addresses: &[vk::DeviceAddress],
        indirect_strides: &[u32],
        max_primitive_counts: &[&[u32]],
    ) -> &Self {
        validate_build_shape(
            infos,
            max_primitive_counts,
            Some((indirect_device_addresses.len(), indirect_strides.len())),
        );
        #[cfg(feature = "checked")]
        for (&address, &stride) in indirect_device_addresses.iter().zip(indirect_strides) {
            assert_device_address(address, 4, "indirect build ranges");
            assert!(
                stride.is_multiple_of(4),
                "indirect build stride must be a multiple of four"
            );
        }

        let geometries = AccelerationStructureGeometryMarshaler::new(
            infos.iter().flat_map(|info| info.geometries.iter()),
        );
        let vk_infos =
            marshal_acceleration_structure_builds(infos, geometries.geometries(), |node| {
                self.resource(node).handle
            });
        let ext = Device::expect_vk_khr_acceleration_structure(&self.cmd.device);

        unsafe {
            ext.cmd_build_acceleration_structures_indirect(
                self.cmd.handle,
                &vk_infos,
                indirect_device_addresses,
                indirect_strides,
                max_primitive_counts,
            );
        }

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
    /// # use vk_graph::cmd::{MicromapBuildInfo, CommandRef};
    /// fn record(cmd: &CommandRef<'_>, infos: &[MicromapBuildInfo<'_>]) {
    ///     cmd.build_micromaps(infos); // Execution-time memory validity requires an unsafe call.
    /// }
    /// ```
    ///
    /// Host build descriptors cannot be recorded as device commands:
    ///
    /// ```compile_fail,E0308
    /// # use vk_graph::{cmd::CommandRef, driver::micromap::HostMicromapBuildInfo};
    /// fn record(cmd: &CommandRef<'_>, infos: &[HostMicromapBuildInfo<'_>]) {
    ///     unsafe { cmd.build_micromaps(infos); }
    /// }
    /// ```
    pub unsafe fn build_micromaps(&self, infos: &[MicromapBuildInfo<'_>]) -> &Self {
        let info_count = vk_count(infos.len(), "micromap build info count");

        #[cfg(feature = "checked")]
        assert!(
            !infos.is_empty(),
            "micromap build info count must be nonzero"
        );

        let usages = infos
            .iter()
            .map(|info| {
                vk_count(info.usage_counts.len(), "micromap usage count");
                info.usage_counts
                    .iter()
                    .copied()
                    .map(Into::into)
                    .collect::<Vec<vk::MicromapUsageEXT>>()
            })
            .collect::<Vec<_>>();
        let vk_infos = infos
            .iter()
            .zip(&usages)
            .map(|(info, usages)| {
                let destination = self.resource(info.dst_micromap);

                #[cfg(feature = "checked")]
                self.validate_micromap_build(destination.info.micromap_type, info);

                marshal_micromap_build(info, destination.handle, usages)
            })
            .collect::<Vec<_>>();
        let ext = Device::expect_vk_ext_opacity_micromap(&self.cmd.device);

        unsafe {
            (ext.fp().cmd_build_micromaps_ext)(self.cmd.handle, info_count, vk_infos.as_ptr());
        }

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

    /// Borrows the resource or persistent resource set represented by the node.
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

    #[cfg(feature = "checked")]
    fn validate_micromap_build(
        &self,
        micromap_type: vk::MicromapTypeEXT,
        info: &MicromapBuildInfo<'_>,
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

        assert_device_address(info.data, 256, "opacity data");
        assert_micromap_device_scratch(info.scratch_data, scratch_alignment, || {
            crate::driver::micromap::Micromap::build_sizes(
                &self.cmd.device,
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                info.flags,
                info.usage_counts,
            )
            .build_scratch_size
        });
        assert_device_address(info.triangle_array, 256, "micromap triangle array");
    }

    /// Returns this invocation's copied stream constant.
    ///
    /// Also available in compute, graphics, and ray tracing callbacks through `Deref`. The handle
    /// must belong to this stream, even without `checked`.
    ///
    /// # Panics
    ///
    /// Panics outside a stream invocation, for an invalid value index, or for a type mismatch.
    /// With `checked`, also panics if the handle belongs to another stream.
    pub fn value<T: Copy + Send + Sync + 'static>(&self, arg: StreamValueArg<T>) -> T {
        self.stream_values
            .expect("missing command stream values")
            .value(arg)
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
}

impl<'a> Deref for CommandRef<'a> {
    type Target = crate::driver::cmd_buf::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        self.cmd
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
    /// Creates a micromap copy description with the given mode.
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
fn assert_micromap_device_scratch(
    address: vk::DeviceAddress,
    alignment: vk::DeviceSize,
    required_size: impl FnOnce() -> vk::DeviceSize,
) {
    if address == 0 {
        assert_eq!(
            required_size(),
            0,
            "null micromap scratch requires zero scratch size"
        );
    } else {
        assert_device_address(address, alignment, "micromap build scratch");
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

fn marshal_acceleration_structure_builds<'a>(
    infos: &[AccelerationStructureBuildGeometryInfo<'_>],
    geometries: &'a [vk::AccelerationStructureGeometryKHR<'a>],
    mut handle: impl FnMut(AnyAccelerationStructureNode) -> vk::AccelerationStructureKHR,
) -> Vec<vk::AccelerationStructureBuildGeometryInfoKHR<'a>> {
    let mut start = 0;

    infos
        .iter()
        .map(|info| {
            let end = start + info.geometries.len();
            let source = if info.mode == vk::BuildAccelerationStructureModeKHR::UPDATE {
                info.src_acceleration_structure
                    .map(&mut handle)
                    .unwrap_or_default()
            } else {
                vk::AccelerationStructureKHR::null()
            };
            let raw = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(info.acceleration_structure_type)
                .flags(info.flags)
                .mode(info.mode)
                .src_acceleration_structure(source)
                .dst_acceleration_structure(handle(info.dst_acceleration_structure))
                .geometries(&geometries[start..end])
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: info.scratch_data,
                });
            start = end;

            raw
        })
        .collect()
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

fn marshal_micromap_build<'a>(
    info: &MicromapBuildInfo<'_>,
    destination: vk::MicromapEXT,
    usages: &'a [vk::MicromapUsageEXT],
) -> vk::MicromapBuildInfoEXT<'a> {
    vk::MicromapBuildInfoEXT::default()
        .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP)
        .mode(vk::BuildMicromapModeEXT::BUILD)
        .dst_micromap(destination)
        .flags(info.flags)
        .usage_counts(usages)
        .data(vk::DeviceOrHostAddressConstKHR {
            device_address: info.data,
        })
        .triangle_array(vk::DeviceOrHostAddressConstKHR {
            device_address: info.triangle_array,
        })
        .triangle_array_stride(info.triangle_array_stride)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: info.scratch_data,
        })
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

/// Specifies one opacity micromap build in a device batch.
///
/// Usage counts are borrowed only during recording and converted to temporary Vulkan arrays.
/// Address-only resources must be declared separately; see [`CommandRef::build_micromaps`].
///
/// ```compile_fail,E0308
/// # use vk_graph::cmd::MicromapBuildInfo;
/// fn host_pointer(info: &mut MicromapBuildInfo<'_>) {
///     info.data = std::ptr::null::<std::ffi::c_void>();
/// }
/// ```
///
/// Borrowed metadata cannot outlive its storage:
///
/// ```compile_fail,E0515
/// # use vk_graph::{AnyMicromapNode, cmd::MicromapBuildInfo, driver::micromap::OpacityMicromapUsage};
/// # use ash::vk;
/// fn escaping_metadata(dst_micromap: AnyMicromapNode) -> MicromapBuildInfo<'static> {
///     let usage_counts = [OpacityMicromapUsage::new(1, 0, vk::OpacityMicromapFormatEXT::TYPE_2_STATE)];
///     MicromapBuildInfo {
///         data: 256,
///         dst_micromap,
///         flags: vk::BuildMicromapFlagsEXT::empty(),
///         scratch_data: 512,
///         triangle_array: 768,
///         triangle_array_stride: 8,
///         usage_counts: &usage_counts,
///     }
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MicromapBuildInfo<'a> {
    /// Device address of opacity data.
    pub data: vk::DeviceAddress,

    /// Destination opacity micromap.
    pub dst_micromap: AnyMicromapNode,

    /// Flags used for the matching size query.
    pub flags: vk::BuildMicromapFlagsEXT,

    /// Device scratch address; zero is permitted only when no scratch is required.
    pub scratch_data: vk::DeviceAddress,

    /// Device address of the triangle description array.
    pub triangle_array: vk::DeviceAddress,

    /// Byte stride between triangle descriptions.
    pub triangle_array_stride: vk::DeviceSize,

    /// Triangle counts grouped by subdivision level and format.
    pub usage_counts: &'a [OpacityMicromapUsage],
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

fn validate_build_shape<T>(
    infos: &[AccelerationStructureBuildGeometryInfo<'_>],
    per_geometry: &[&[T]],
    indirect_lengths: Option<(usize, usize)>,
) {
    vk_count(infos.len(), "acceleration structure build info count");

    assert_eq!(
        infos.len(),
        per_geometry.len(),
        "one range/count slice is required per build info"
    );
    if let Some((addresses, strides)) = indirect_lengths {
        assert_eq!(
            infos.len(),
            addresses,
            "one indirect address is required per build info"
        );
        assert_eq!(
            infos.len(),
            strides,
            "one indirect stride is required per build info"
        );
    }

    #[cfg(feature = "checked")]
    assert!(
        !infos.is_empty(),
        "acceleration structure build info count must be nonzero"
    );
    for (info, values) in infos.iter().zip(per_geometry) {
        vk_count(
            info.geometries.len(),
            "acceleration structure geometry count",
        );

        assert_eq!(
            info.geometries.len(),
            values.len(),
            "one range/count is required per geometry"
        );
        #[cfg(feature = "checked")]
        {
            assert!(
                matches!(
                    info.mode,
                    vk::BuildAccelerationStructureModeKHR::BUILD
                        | vk::BuildAccelerationStructureModeKHR::UPDATE
                ),
                "build mode must be BUILD or UPDATE"
            );
            assert!(
                info.mode != vk::BuildAccelerationStructureModeKHR::UPDATE
                    || info.src_acceleration_structure.is_some(),
                "UPDATE requires a source acceleration structure"
            );
            assert!(
                matches!(
                    info.acceleration_structure_type,
                    vk::AccelerationStructureTypeKHR::TOP_LEVEL
                        | vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL
                ),
                "build type must be TOP_LEVEL or BOTTOM_LEVEL"
            );
        }
    }
}

fn vk_count(len: usize, description: &str) -> u32 {
    u32::try_from(len)
        .unwrap_or_else(|_| panic!("{description} ({len}) exceeds Vulkan's u32 limit"))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::driver::accel_struct::AccelerationStructureGeometryData;
    use ash::vk::Handle;

    #[test]
    #[cfg(target_pointer_width = "64")]
    #[should_panic(expected = "exceeds Vulkan's u32 limit")]
    fn counts_are_checked_without_the_checked_feature() {
        vk_count(u32::MAX as usize + 1, "test count");
    }

    #[test]
    fn direct_and_indirect_build_shapes_and_mixed_modes() {
        use crate::{
            Node,
            driver::accel_struct::{
                AccelerationStructureOpacityMicromap, AccelerationStructureTriangles,
            },
        };

        #[cfg(feature = "checked")]
        let graph_id = crate::GraphId::next();
        let nodes = [0, 1, 2, 3].map(|index| {
            crate::AccelerationStructureNode::new(
                index,
                #[cfg(feature = "checked")]
                graph_id,
            )
        });
        let usages = [OpacityMicromapUsage::new(
            1,
            0,
            vk::OpacityMicromapFormatEXT::TYPE_2_STATE,
        )];
        let micromap = vk::MicromapEXT::from_raw(20);
        let triangles = AccelerationStructureTriangles::new(
            0,
            vk::IndexType::NONE_KHR,
            2,
            0,
            0x100,
            vk::Format::R32G32B32_SFLOAT,
            12,
        );
        let unattached = AccelerationStructureGeometry::new(
            AccelerationStructureGeometryData::Triangles(triangles),
        );
        let attached =
            AccelerationStructureGeometry::new(AccelerationStructureGeometryData::Triangles(
                triangles
                    .opacity_micromap(AccelerationStructureOpacityMicromap::new(micromap, &usages)),
            ));
        let geometries = [
            attached, unattached, attached, unattached, attached, unattached,
        ];
        let flags = vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE;
        let infos = [
            AccelerationStructureBuildGeometryInfo::build(
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                flags,
                nodes[0],
                &geometries[..2],
                0x200,
            ),
            AccelerationStructureBuildGeometryInfo::update(
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                flags,
                nodes[1],
                nodes[2],
                &geometries[2..3],
                0x300,
            ),
            AccelerationStructureBuildGeometryInfo::update(
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                flags,
                nodes[3],
                nodes[3],
                &geometries[3..],
                0x400,
            ),
        ];
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count: 1,
            ..Default::default()
        }; 3];
        validate_build_shape(&infos, &[&ranges[..2], &ranges[..1], &ranges], None);
        validate_build_shape(&infos, &[&[1, 1], &[1], &[1, 1, 1]], Some((3, 3)));
        for (lengths, counts) in [
            (None, vec![&[][..], &[1][..], &[1, 1, 1][..]]),
            (None, vec![&[1, 1][..], &[1][..]]),
            (Some((2, 3)), vec![&[1, 1][..], &[1][..], &[1, 1, 1][..]]),
            (Some((3, 2)), vec![&[1, 1][..], &[1][..], &[1, 1, 1][..]]),
            (Some((3, 3)), vec![&[1, 1][..], &[][..], &[1, 1, 1][..]]),
            (Some((3, 3)), vec![&[1, 1][..], &[1][..]]),
        ] {
            assert!(
                std::panic::catch_unwind(|| validate_build_shape(&infos, &counts, lengths))
                    .is_err()
            );
        }

        let marshaler = AccelerationStructureGeometryMarshaler::new(
            infos.iter().flat_map(|info| info.geometries.iter()),
        );
        // Moving the owner must not relocate the geometry extension or usage arrays.
        let extension_pointers = marshaler
            .geometries()
            .iter()
            .map(|geometry| unsafe { geometry.geometry.triangles.p_next })
            .collect::<Vec<_>>();
        let usage_pointers = extension_pointers
            .iter()
            .map(|&extension| {
                if extension.is_null() {
                    std::ptr::null()
                } else {
                    unsafe {
                        (*extension
                            .cast::<vk::AccelerationStructureTrianglesOpacityMicromapEXT<'_>>())
                        .p_usage_counts
                    }
                }
            })
            .collect::<Vec<_>>();
        let marshalers = [marshaler];
        let marshaler = &marshalers[0];
        let handle = |node: AnyAccelerationStructureNode| {
            vk::AccelerationStructureKHR::from_raw(node.index() as u64 + 9)
        };
        let raw = marshal_acceleration_structure_builds(&infos, marshaler.geometries(), handle);

        assert_eq!(raw[0].mode, vk::BuildAccelerationStructureModeKHR::BUILD);
        assert_eq!(
            raw[0].src_acceleration_structure,
            vk::AccelerationStructureKHR::null()
        );
        assert_eq!(raw[1].mode, vk::BuildAccelerationStructureModeKHR::UPDATE);
        assert_eq!(raw[1].src_acceleration_structure, handle(nodes[1].into()));
        assert_ne!(
            raw[1].src_acceleration_structure,
            raw[1].dst_acceleration_structure
        );
        assert_eq!(raw[2].mode, vk::BuildAccelerationStructureModeKHR::UPDATE);
        assert_eq!(
            raw[2].src_acceleration_structure,
            raw[2].dst_acceleration_structure
        );
        let mut start = 0;
        for (index, raw) in raw.iter().enumerate() {
            assert_eq!(
                raw.dst_acceleration_structure,
                handle(infos[index].dst_acceleration_structure)
            );
            assert_eq!(raw.flags, flags);
            assert_eq!(raw.geometry_count as usize, infos[index].geometries.len());
            assert_eq!(raw.p_geometries, &marshaler.geometries()[start]);
            assert!(raw.pp_geometries.is_null());

            unsafe {
                assert_eq!(raw.scratch_data.device_address, infos[index].scratch_data);
                for offset in 0..raw.geometry_count as usize {
                    let geometry = &*raw.p_geometries.add(offset);
                    assert_eq!(geometry.geometry_type, vk::GeometryTypeKHR::TRIANGLES);
                    let extension = geometry.geometry.triangles.p_next;
                    assert_eq!(extension, extension_pointers[start + offset]);
                    if (start + offset).is_multiple_of(2) {
                        assert!(!extension.is_null());
                        let attachment = &*extension
                            .cast::<vk::AccelerationStructureTrianglesOpacityMicromapEXT<'_>>(
                        );
                        assert_eq!(attachment.micromap, micromap);
                        assert_eq!(attachment.usage_counts_count, 1);
                        assert!(attachment.pp_usage_counts.is_null());
                        assert!(!attachment.p_usage_counts.is_null());
                        assert_eq!(attachment.p_usage_counts, usage_pointers[start + offset]);
                        assert_eq!((*attachment.p_usage_counts).count, 1);
                        assert_eq!((*attachment.p_usage_counts).subdivision_level, 0);
                        assert_eq!(
                            (*attachment.p_usage_counts).format,
                            vk::OpacityMicromapFormatEXT::TYPE_2_STATE.as_raw() as u32
                        );
                    } else {
                        assert!(extension.is_null());
                    }
                }
            }

            start += raw.geometry_count as usize;
        }

        assert_eq!(start, geometries.len());
    }

    #[test]
    fn general_access_conservatively_covers_indirect_build_ranges() {
        let (src_stage, dst_stage, legacy) = vk_sync::get_memory_barrier(&vk_sync::GlobalBarrier {
            previous_accesses: &[vk_sync::AccessType::General],
            next_accesses: &[vk_sync::AccessType::General],
        });

        assert_eq!(src_stage, vk::PipelineStageFlags::ALL_COMMANDS);
        assert_eq!(dst_stage, vk::PipelineStageFlags::ALL_COMMANDS);
        assert_eq!(
            legacy.src_access_mask,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
        );
        assert_eq!(
            legacy.dst_access_mask,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
        );
        let sync2 = vk_sync::get_access_info2(vk_sync::AccessType::General);

        assert_eq!(sync2.stage_mask, vk::PipelineStageFlags2::ALL_COMMANDS);
        assert_eq!(
            sync2.access_mask,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE
        );
    }

    #[cfg(feature = "checked")]
    #[test]
    fn micromap_device_scratch_checks_size_lazily_and_alignment() {
        let queries = std::cell::Cell::new(0);
        assert_micromap_device_scratch(0, 256, || {
            queries.set(queries.get() + 1);
            0
        });
        assert_eq!(queries.get(), 1);
        assert!(std::panic::catch_unwind(|| assert_micromap_device_scratch(0, 256, || 1)).is_err());
        assert_micromap_device_scratch(256, 256, || panic!("non-null scratch must not query"));
        let queries = std::cell::Cell::new(0);

        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                assert_micromap_device_scratch(1, 256, || {
                    queries.set(queries.get() + 1);
                    0
                });
            }))
            .is_err()
        );
        assert_eq!(queries.get(), 0);
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
    fn opacity_build_marshaling_keeps_temporary_usage_pointer_stable() {
        let usages = [
            OpacityMicromapUsage::new(3, 1, vk::OpacityMicromapFormatEXT::TYPE_2_STATE),
            OpacityMicromapUsage::new(5, 2, vk::OpacityMicromapFormatEXT::TYPE_4_STATE),
        ];
        let node = crate::MicromapNode::new(
            0,
            #[cfg(feature = "checked")]
            crate::GraphId::next(),
        );
        let build = MicromapBuildInfo {
            data: 0x100,
            dst_micromap: node.into(),
            flags: vk::BuildMicromapFlagsEXT::ALLOW_COMPACTION,
            scratch_data: 0x200,
            triangle_array: 0x300,
            triangle_array_stride: 0,
            usage_counts: &usages,
        };
        let destination = vk::MicromapEXT::from_raw(7);
        let raw_usages = usages.into_iter().map(Into::into).collect::<Vec<_>>();
        let raw = marshal_micromap_build(&build, destination, &raw_usages);

        assert_eq!(raw.mode, vk::BuildMicromapModeEXT::BUILD);
        assert_eq!(raw.ty, vk::MicromapTypeEXT::OPACITY_MICROMAP);
        assert_eq!(raw.dst_micromap, destination);
        assert_eq!(raw.flags, build.flags);
        assert_eq!(raw.usage_counts_count, 2);
        assert_eq!(raw.p_usage_counts, raw_usages.as_ptr());
        assert!(raw.pp_usage_counts.is_null());
        assert_eq!(raw.triangle_array_stride, 0);

        unsafe {
            assert_eq!((*raw.p_usage_counts.add(1)).count, 5);
            assert_eq!(raw.data.device_address, 0x100);
            assert_eq!(raw.scratch_data.device_address, 0x200);
            assert_eq!(raw.triangle_array.device_address, 0x300);
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
