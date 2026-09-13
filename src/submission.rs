//! Submission and recording types.
//!
//! This module contains the execution-facing types produced by [`Graph::finalize`].
//!
//! Typical usage starts with a [`Submission`], which represents a finalized graph that has not yet
//! been bound to a command buffer:
//!
//! - Use [`Submission::queue_submit`] for the one-shot path that allocates, records, and submits a
//!   command buffer internally.
//! - Use [`Submission::record`] with a [`RecordSelection`] to bind the submission to an existing
//!   command buffer and obtain a [`Recording`].
//!
//! A [`Recording`] keeps the remaining graph work paired with the command buffer it was
//! recorded into. This typestate prevents recording with one command buffer and accidentally
//! submitting with another.
//!
//! [`Graph::finalize`]: crate::Graph::finalize

use {
    super::{
        AnyResource, Attachment, CommandData, ExecutionAccess, ExecutionPipeline, Graph, LoadOp,
        Node, NodeIndex, ResourceNode, ResourceSetAccess, TimestampQueryData,
        TimestampQueryPlacement,
        cmd::{SubresourceAccess, SubresourceRange},
    },
    crate::{
        StoreOp, TimestampQuery,
        cmd::CommandRef,
        driver::{
            AttachmentInfo, AttachmentRef, Descriptor, DescriptorInfo, DriverError,
            FramebufferAttachmentImageInfo, FramebufferInfo, RawDescriptorSet, SharingMode,
            SubpassDependency, SubpassInfo,
            accel_struct::AccelerationStructure,
            buffer::{Buffer, BufferSubresourceRange},
            cmd_buf::{CommandBuffer, CommandBufferInfo},
            descriptor_set::{DescriptorPool, DescriptorPoolInfo, DescriptorSet},
            device::Device,
            fence::{Fence, FenceDroppable},
            format_aspect_mask,
            graphics::{DepthStencilInfo, GraphicsPipeline},
            image::{
                DenseMap, Image, ImageAccessSet, ImageInfo, SampleCount, access_type_to_layout,
                image_subresource_range_contains, image_subresource_range_intersection,
            },
            initial_image_layout_access, is_read_access, is_write_access,
            micromap::{Micromap, micromap_sync_flags_for_access},
            physical_device::Vulkan10Limits,
            pipeline_stage_access_flags,
            query_pool::{QueryPool, QueryPoolInfo},
            render_pass::{RenderPass, RenderPassInfo},
        },
        lazy_str,
        node::AnyNode,
        pool::{Lease, Pool, SubmissionPool, hash::HashPool},
        resource::{
            ImageAccessType, PhysicalImageId, ResourceSet, ResourceSetAccessType, ResourceSetIndex,
            ResourceSetMap,
        },
    },
    ash::vk::{self, QueueFamilyProperties},
    fixedbitset::FixedBitSet,
    log::{
        Level::{Debug, Trace},
        debug, log_enabled, trace, warn,
    },
    smallvec::SmallVec,
    std::{
        cell::RefCell,
        cmp::Reverse,
        collections::{BTreeMap, BTreeSet, HashMap, hash_map::Entry},
        iter::repeat_n,
        mem::take,
        ops::Range,
        slice,
        sync::{Arc, Mutex},
        time::Duration,
    },
    vk_sync::{
        AccessType, BufferBarrier, GlobalBarrier, ImageBarrier, ImageLayout,
        get_image_memory_barrier, get_memory_barrier,
    },
};

#[cfg(feature = "checked")]
use super::GraphId;

#[cfg(not(feature = "checked"))]
use std::hint::unreachable_unchecked;

thread_local! {
    static SUBMIT: RefCell<SubmitScratch> = Default::default();
    static SUBPASS_DEPENDENCY: RefCell<SubpassDependencyScratch> = Default::default();
}

#[derive(Clone, Copy, Debug)]
enum AccessClass {
    AttachmentLocal(Attachment),
    ReadOnlyBuffer,
    ReadOnlyImage(vk::ImageLayout),
    Other,
}

impl AccessClass {
    fn new(
        exec: &crate::Execution,
        node: NodeIndex,
        origin: SubpassAccessOrigin<'_>,
        attachment_nodes: &FixedBitSet,
    ) -> Self {
        if attachment_nodes.contains(node) {
            if let SubpassAccessOrigin::Attachment(att) = origin
                && att.sample_count != SampleCount::Type1
            {
                return AccessClass::Other;
            }

            // Resolve roles are not sample-local proofs, even for a single-sample destination.
            let resolves_node = exec.attachments.color_attachments().any(|(_, state)| {
                state.resolve.as_ref().is_some_and(|resolve| {
                    state.attachment.target == node
                        || resolve.attachment.target == node
                        || exec
                            .attachments
                            .color_attachment(resolve.src_attachment_idx)
                            .is_some_and(|source| source.attachment.target == node)
                })
            }) || exec.attachments.depth_stencil_attachment().is_some_and(
                |state| {
                    state.resolve.as_ref().is_some_and(|resolve| {
                        state.attachment.target == node || resolve.attachment.target == node
                    })
                },
            );
            if resolves_node {
                return AccessClass::Other;
            }

            let mut matched = None;
            let roles = exec
                .attachments
                .color_attachments()
                .filter(|(_, state)| state.attachment.target == node)
                .map(|(_, state)| (&state.attachment, state.is_attachment, state.is_input))
                .chain(
                    exec.attachments
                        .depth_stencil_attachment()
                        .filter(|state| state.attachment.target == node)
                        .map(|state| (&state.attachment, state.is_attachment, false)),
                );
            for (attachment, output, input) in roles {
                if attachment.sample_count != SampleCount::Type1 {
                    return AccessClass::Other;
                }
                let ordinary = match origin {
                    SubpassAccessOrigin::Attachment(source) => {
                        (output || input) && Submission::attachments_are_exact(*source, *attachment)
                    }
                    SubpassAccessOrigin::Explicit(access) => {
                        let role_matches = if attachment.aspect_mask == vk::ImageAspectFlags::COLOR
                        {
                            (output
                                && matches!(
                                    access.access,
                                    AccessType::ColorAttachmentRead
                                        | AccessType::ColorAttachmentWrite
                                        | AccessType::ColorAttachmentReadWrite
                                ))
                                || (input
                                    && access.access
                                        == AccessType::FragmentShaderReadColorInputAttachment)
                        } else {
                            (input
                            && access.access
                                == AccessType::FragmentShaderReadDepthStencilInputAttachment)
                            || (output
                                && (matches!(
                                    access.access,
                                    AccessType::DepthStencilAttachmentRead
                                        | AccessType::DepthStencilAttachmentWrite
                                        | AccessType::DepthStencilAttachmentReadWrite
                                ) || (attachment.aspect_mask == vk::ImageAspectFlags::DEPTH
                                    && access.access
                                        == AccessType::DepthAttachmentWriteStencilReadOnly)
                                    || (attachment.aspect_mask == vk::ImageAspectFlags::STENCIL
                                        && access.access
                                            == AccessType::StencilAttachmentWriteDepthReadOnly)))
                        };
                        role_matches
                            && matches!(access.subresource, SubresourceRange::Image(range)
                            if ImageOwnershipTransfer::ranges_equal(range, vk::ImageSubresourceRange {
                                aspect_mask: attachment.aspect_mask,
                                base_array_layer: attachment.base_array_layer,
                                layer_count: attachment.array_layer_count,
                                base_mip_level: attachment.base_mip_level,
                                level_count: attachment.mip_level_count,
                            }))
                    }
                };
                if !ordinary {
                    return AccessClass::Other;
                }
                let class = AccessClass::AttachmentLocal(*attachment);
                matched =
                    Some(matched.map_or(class, |previous: AccessClass| previous.merge(class)));
            }
            if let Some(class) = matched {
                return class;
            }
        }
        if let SubpassAccessOrigin::Explicit(access) = origin
            && PipelineStageAccessFlags::is_read_only_graphics_access(access.access)
        {
            return match access.subresource {
                SubresourceRange::Buffer(_) => AccessClass::ReadOnlyBuffer,
                // Only sampled readers accumulate outgoing scopes in ImageAccessSet.
                SubresourceRange::Image(_)
                    if ImageAccessSet::from_access(access.access).is_sampled_read() =>
                {
                    access_type_to_layout(access.access)
                        .filter(|&layout| {
                            layout != vk::ImageLayout::UNDEFINED
                                && layout != vk::ImageLayout::PREINITIALIZED
                        })
                        .map_or(AccessClass::Other, AccessClass::ReadOnlyImage)
                }
                _ => AccessClass::Other,
            };
        }
        AccessClass::Other
    }

    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::AttachmentLocal(lhs), Self::AttachmentLocal(rhs))
                if Submission::attachments_are_exact(lhs, rhs) =>
            {
                self
            }
            (Self::ReadOnlyBuffer, Self::ReadOnlyBuffer) => self,
            (Self::ReadOnlyImage(lhs), Self::ReadOnlyImage(rhs)) if lhs == rhs => self,
            _ => Self::Other,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BufferQueueOwnershipTransfer {
    range: BufferSubresourceRange,
    dst_queue_family_index: u32,
    src_queue_family_index: u32,
}

impl BufferQueueOwnershipTransfer {
    fn barriers<'a>(
        buffer: vk::Buffer,
        prev_access: &'a AccessType,
        next_access: &'a AccessType,
        range: BufferSubresourceRange,
        transfers: &'a [BufferQueueOwnershipTransfer],
    ) -> impl Iterator<Item = BufferBarrier<'a>> + 'a {
        struct BufferBarrierIter<'a> {
            buffer: vk::Buffer,
            cuts: SmallVec<[vk::DeviceSize; 4]>,
            cut_idx: usize,
            next_access: &'a AccessType,
            prev_access: &'a AccessType,
            transfers: &'a [BufferQueueOwnershipTransfer],
        }

        impl<'a> Iterator for BufferBarrierIter<'a> {
            type Item = BufferBarrier<'a>;

            fn next(&mut self) -> Option<Self::Item> {
                while self.cut_idx + 1 < self.cuts.len() {
                    let range = BufferSubresourceRange {
                        start: self.cuts[self.cut_idx],
                        end: self.cuts[self.cut_idx + 1],
                    };
                    self.cut_idx += 1;

                    if range.start == range.end {
                        continue;
                    }

                    let transfer = self
                        .transfers
                        .iter()
                        .find(|transfer| transfer.range.contains(range));

                    trace!(
                        "    buffer {:?} {:?} {:?}->{:?}",
                        self.buffer,
                        range.start..range.end,
                        self.prev_access,
                        self.next_access,
                    );

                    return Some(BufferBarrier {
                        next_accesses: slice::from_ref(self.next_access),
                        previous_accesses: slice::from_ref(self.prev_access),
                        src_queue_family_index: transfer
                            .map_or(vk::QUEUE_FAMILY_IGNORED, |transfer| {
                                transfer.src_queue_family_index
                            }),
                        dst_queue_family_index: transfer
                            .map_or(vk::QUEUE_FAMILY_IGNORED, |transfer| {
                                transfer.dst_queue_family_index
                            }),
                        buffer: self.buffer,
                        offset: range.start as _,
                        size: (range.end - range.start) as _,
                    });
                }

                None
            }
        }

        let mut cuts = SmallVec::<[vk::DeviceSize; 4]>::with_capacity(
            transfers.len().saturating_mul(2).saturating_add(2),
        );
        cuts.extend([range.start, range.end]);

        for transfer in transfers {
            if let Some(overlap) = range.intersection(transfer.range) {
                cuts.push(overlap.start);
                cuts.push(overlap.end);
            }
        }

        cuts.sort_unstable();
        cuts.dedup();

        BufferBarrierIter {
            buffer,
            cuts,
            cut_idx: 0,
            next_access,
            prev_access,
            transfers,
        }
    }

    fn consume_pending(
        transfers: &mut Vec<BufferQueueOwnershipTransfer>,
        range: BufferSubresourceRange,
    ) -> bool {
        transfers.retain(|transfer| {
            !BufferQueueOwnershipTransfer::ranges_intersect(transfer.range, range)
        });
        transfers.is_empty()
    }

    fn ranges_intersect(lhs: BufferSubresourceRange, rhs: BufferSubresourceRange) -> bool {
        lhs.start < rhs.end && lhs.end > rhs.start
    }
}

#[derive(Clone, Default)]
struct CommandAccessIndex {
    cmds_by_node: Vec<Vec<usize>>,
    accessed_nodes_by_cmd: Vec<Vec<usize>>,
    cmds_by_resource_set: Vec<Vec<usize>>,
    accessed_resource_sets_by_cmd: Vec<Vec<ResourceSetIndex>>,
}

impl CommandAccessIndex {
    #[profiling::function]
    fn read_nodes_for_cmd(&self, cmd_idx: usize) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.accessed_nodes_by_cmd[cmd_idx].iter().copied()
    }

    fn read_resource_sets_for_cmd(
        &self,
        cmd_idx: usize,
    ) -> impl ExactSizeIterator<Item = ResourceSetIndex> + '_ {
        self.accessed_resource_sets_by_cmd
            .get(cmd_idx)
            .map(Vec::as_slice)
            .unwrap_or_default()
            .iter()
            .copied()
    }

    fn update(&mut self, graph: &Graph, end_cmd_idx: usize) {
        let binding_count = graph.resources.len();
        let resource_set_count = graph.resource_sets.len();
        let cmds = &graph.cmds[0..end_cmd_idx];
        self.update_from_cmds(cmds, binding_count, resource_set_count);
    }

    fn update_from_cmds(
        &mut self,
        cmds: &[CommandData],
        binding_count: usize,
        resource_set_count: usize,
    ) {
        self.cmds_by_node.clear();
        self.cmds_by_node.resize_with(binding_count, Vec::new);

        self.cmds_by_resource_set.clear();
        self.cmds_by_resource_set
            .resize_with(resource_set_count, Vec::new);

        self.accessed_nodes_by_cmd.clear();
        self.accessed_nodes_by_cmd.resize_with(cmds.len(), Vec::new);

        self.accessed_resource_sets_by_cmd.clear();
        self.accessed_resource_sets_by_cmd
            .resize_with(cmds.len(), Vec::new);

        thread_local! {
            static SEEN_RESOURCES: RefCell<(
                FixedBitSet,
                FixedBitSet,
                FixedBitSet,
                FixedBitSet,
            )> = Default::default();
        }

        SEEN_RESOURCES.with_borrow_mut(
            |(seen_nodes, seen_accesses, seen_resource_sets, seen_resource_set_accesses)| {
                seen_nodes.clear();
                seen_nodes.grow(binding_count);

                seen_accesses.clear();
                seen_accesses.grow(binding_count);

                seen_resource_sets.clear();
                seen_resource_sets.grow(resource_set_count);

                seen_resource_set_accesses.clear();
                seen_resource_set_accesses.grow(resource_set_count);

                for (cmd_idx, cmd) in cmds.iter().enumerate() {
                    let accessed_nodes = &mut self.accessed_nodes_by_cmd[cmd_idx];

                    for (node_idx, _) in cmd.execs.iter().flat_map(|exec| exec.accesses.iter()) {
                        if !seen_nodes.put(node_idx) {
                            self.cmds_by_node[node_idx].push(cmd_idx);
                        }

                        if !seen_accesses.put(node_idx) {
                            accessed_nodes.push(node_idx);
                        }
                    }

                    let accessed_resource_sets = &mut self.accessed_resource_sets_by_cmd[cmd_idx];
                    for access in cmd
                        .execs
                        .iter()
                        .flat_map(|exec| &exec.resource_set_accesses)
                    {
                        let resource_set_idx = access.resource_set_idx;
                        let index = resource_set_idx.as_usize();

                        debug_assert!(index < resource_set_count);

                        if !seen_resource_sets.put(index) {
                            self.cmds_by_resource_set[index].push(cmd_idx);
                        }

                        if !seen_resource_set_accesses.put(index) {
                            accessed_resource_sets.push(resource_set_idx);
                        }
                    }

                    seen_nodes.clear();
                    seen_nodes.grow(binding_count);
                    seen_accesses.clear();
                    seen_accesses.grow(binding_count);

                    seen_resource_sets.clear();
                    seen_resource_sets.grow(resource_set_count);
                    seen_resource_set_accesses.clear();
                    seen_resource_set_accesses.grow(resource_set_count);
                }
            },
        );
    }
}

struct CommandBufferDebugLabel<'a> {
    cmd_buf: &'a CommandBuffer,
}

impl Drop for CommandBufferDebugLabel<'_> {
    fn drop(&mut self) {
        let _ = Device::end_debug_utils_label(&self.cmd_buf.device, self.cmd_buf.handle);
    }
}

impl<'a> CommandBufferDebugLabel<'a> {
    fn begin(cmd_buf: &'a CommandBuffer, name: impl AsRef<str>) -> Option<Self> {
        Device::begin_debug_utils_label(&cmd_buf.device, cmd_buf.handle, name)
            .ok()
            .map(|_| Self { cmd_buf })
    }
}

#[derive(Debug, Default)]
struct CommandRecordingResources {
    descriptor_pool: Option<Lease<DescriptorPool>>,
    descriptor_sets: Vec<Vec<RecordingDescriptorSet>>,
    exec_subpasses: Box<[u32]>,
    render_pass: Option<Lease<RenderPass>>,
}

impl CommandRecordingResources {
    /// # Panics
    ///
    /// Panics if the physical pass has no render pass.
    fn expect_render_pass_mut(&mut self) -> &mut Lease<RenderPass> {
        self.render_pass.as_mut().expect("missing render pass")
    }
}

impl Drop for CommandRecordingResources {
    fn drop(&mut self) {
        self.descriptor_sets.clear();
        self.descriptor_pool = None;
    }
}

// Only reflection and rasterization metadata are needed to plan physical subpasses.
#[derive(Clone, Copy)]
struct GraphicsExecutionInfo<'a> {
    input_attachments: &'a [u32],
    sample_count: SampleCount,
}

#[derive(Debug)]
enum ImageOwnership {
    Whole,
    DualAspect(vk::ImageAspectFlags),
    Dense(DenseMap<bool>),
}

impl ImageOwnership {
    fn new(info: ImageInfo, range: vk::ImageSubresourceRange) -> Self {
        if info.is_full_subresource_range(range) {
            return Self::Whole;
        }

        let aspect_mask = format_aspect_mask(info.format);
        if aspect_mask.as_raw().count_ones() == 2
            && info.array_layer_count == 1
            && info.mip_level_count == 1
        {
            return Self::DualAspect(range.aspect_mask);
        }

        let mut claimed = DenseMap::new(info, false);
        claimed.swap(true, range).for_each(drop);

        Self::Dense(claimed)
    }

    fn claim(
        &mut self,
        info: ImageInfo,
        mut range: vk::ImageSubresourceRange,
    ) -> SmallVec<[vk::ImageSubresourceRange; 4]> {
        match self {
            Self::Whole => SmallVec::new(),
            Self::DualAspect(claimed) => {
                let unclaimed = range.aspect_mask & !*claimed;
                *claimed |= range.aspect_mask;
                let whole = claimed.contains(format_aspect_mask(info.format));

                if whole {
                    *self = Self::Whole;
                }

                if unclaimed.is_empty() {
                    return SmallVec::new();
                }

                range.aspect_mask = unclaimed;

                SmallVec::from_slice(&[range])
            }
            Self::Dense(claimed) => {
                let unclaimed = claimed
                    .swap(true, range)
                    .filter_map(|(claimed, range)| (!claimed).then_some(range))
                    .collect();

                if info.is_full_subresource_range(range) {
                    *self = Self::Whole;
                }

                unclaimed
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ImageOwnershipLayouts {
    old: vk::ImageLayout,
    new: vk::ImageLayout,
}

impl ImageOwnershipLayouts {
    fn new(
        current_layout: Option<vk::ImageLayout>,
        next_access: AccessType,
        discard_contents: bool,
    ) -> ImageOwnershipLayouts {
        ImageOwnershipLayouts {
            old: if discard_contents {
                vk::ImageLayout::UNDEFINED
            } else {
                current_layout.unwrap_or(vk::ImageLayout::UNDEFINED)
            },
            new: access_type_to_layout(next_access).unwrap_or(vk::ImageLayout::UNDEFINED),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ImageOwnershipTransfer {
    dst_queue_family_index: u32,
    layouts: ImageOwnershipLayouts,
    range: vk::ImageSubresourceRange,
    src_queue_family_index: u32,
    src_queue_index: u32,
}

impl ImageOwnershipTransfer {
    fn aspect_mask_for_span(base_aspect: u32, start: u32, end: u32) -> vk::ImageAspectFlags {
        let mut mask = vk::ImageAspectFlags::empty();

        for ordinal in start..end {
            mask |= vk::ImageAspectFlags::from_raw(1 << (base_aspect + ordinal));
        }

        mask
    }

    fn barrier_ranges<'a>(
        transfers: &'a [ImageOwnershipTransfer],
        range: vk::ImageSubresourceRange,
    ) -> impl Iterator<
        Item = (
            vk::ImageSubresourceRange,
            Option<&'a ImageOwnershipTransfer>,
        ),
    > + 'a {
        thread_local! {
            static IMAGE_TRANSFER: RefCell<ImageTransferScratch> = Default::default();
        }

        struct ImageBarrierTransferIter<'a> {
            transfers: &'a [ImageOwnershipTransfer],
            overlaps: Vec<(usize, vk::ImageSubresourceRange)>,
            aspect_cuts: Vec<u32>,
            layer_cuts: Vec<u32>,
            mip_cuts: Vec<u32>,
            base_aspect: u32,
            range: vk::ImageSubresourceRange,
            aspect_idx: usize,
            layer_idx: usize,
            mip_idx: usize,
            yielded_empty: bool,
        }

        impl Drop for ImageBarrierTransferIter<'_> {
            fn drop(&mut self) {
                IMAGE_TRANSFER.with_borrow_mut(|tls| {
                    tls.overlaps = take(&mut self.overlaps);
                    tls.aspect_cuts = take(&mut self.aspect_cuts);
                    tls.layer_cuts = take(&mut self.layer_cuts);
                    tls.mip_cuts = take(&mut self.mip_cuts);
                });
            }
        }

        impl<'a> Iterator for ImageBarrierTransferIter<'a> {
            type Item = (
                vk::ImageSubresourceRange,
                Option<&'a ImageOwnershipTransfer>,
            );

            fn next(&mut self) -> Option<Self::Item> {
                if self.overlaps.is_empty() {
                    return if self.yielded_empty {
                        None
                    } else {
                        self.yielded_empty = true;
                        Some((self.range, None))
                    };
                }

                let aspect_windows = self.aspect_cuts.len().saturating_sub(1);
                let layer_windows = self.layer_cuts.len().saturating_sub(1);
                let mip_windows = self.mip_cuts.len().saturating_sub(1);

                while self.aspect_idx < aspect_windows {
                    let aspect_start = self.aspect_cuts[self.aspect_idx];
                    let aspect_end = self.aspect_cuts[self.aspect_idx + 1];
                    if aspect_start == aspect_end {
                        self.aspect_idx += 1;
                        self.layer_idx = 0;
                        self.mip_idx = 0;
                        continue;
                    }

                    let aspect_mask = ImageOwnershipTransfer::aspect_mask_for_span(
                        self.base_aspect,
                        aspect_start,
                        aspect_end,
                    );

                    while self.layer_idx < layer_windows {
                        let layer_start = self.layer_cuts[self.layer_idx];
                        let layer_end = self.layer_cuts[self.layer_idx + 1];
                        if layer_start == layer_end {
                            self.layer_idx += 1;
                            self.mip_idx = 0;
                            continue;
                        }

                        while self.mip_idx < mip_windows {
                            let mip_start = self.mip_cuts[self.mip_idx];
                            let mip_end = self.mip_cuts[self.mip_idx + 1];
                            self.mip_idx += 1;
                            if mip_start == mip_end {
                                continue;
                            }

                            let subrange = vk::ImageSubresourceRange {
                                aspect_mask,
                                base_array_layer: self.range.base_array_layer + layer_start,
                                layer_count: layer_end - layer_start,
                                base_mip_level: self.range.base_mip_level + mip_start,
                                level_count: mip_end - mip_start,
                            };

                            let transfer = self
                                .overlaps
                                .iter()
                                .find(|(_, overlap)| {
                                    image_subresource_range_contains(*overlap, subrange)
                                })
                                .map(|(transfer_idx, _)| &self.transfers[*transfer_idx]);

                            return Some((subrange, transfer));
                        }

                        self.layer_idx += 1;
                        self.mip_idx = 0;
                    }

                    self.aspect_idx += 1;
                    self.layer_idx = 0;
                    self.mip_idx = 0;
                }

                None
            }
        }

        #[derive(Default)]
        struct ImageTransferScratch {
            overlaps: Vec<(usize, vk::ImageSubresourceRange)>,
            aspect_cuts: Vec<u32>,
            layer_cuts: Vec<u32>,
            mip_cuts: Vec<u32>,
        }

        IMAGE_TRANSFER.with_borrow_mut(|tls| {
            let mut overlaps = take(&mut tls.overlaps);
            let mut aspect_cuts = take(&mut tls.aspect_cuts);
            let mut layer_cuts = take(&mut tls.layer_cuts);
            let mut mip_cuts = take(&mut tls.mip_cuts);

            overlaps.clear();
            aspect_cuts.clear();
            layer_cuts.clear();
            mip_cuts.clear();

            overlaps.extend(
                transfers
                    .iter()
                    .enumerate()
                    .filter_map(|(transfer_idx, transfer)| {
                        image_subresource_range_intersection(transfer.range, range)
                            .map(|intersection| (transfer_idx, intersection))
                    }),
            );

            let base_aspect = range.aspect_mask.as_raw().trailing_zeros();

            if overlaps.is_empty() {
                // Yield the whole range once when there is no overlapping transfer
            } else {
                let aspect_count = range.aspect_mask.as_raw().count_ones();

                aspect_cuts.extend([0, aspect_count]);
                layer_cuts.extend([0, range.layer_count]);
                mip_cuts.extend([0, range.level_count]);

                for (_, overlap) in &overlaps {
                    let aspect_start = overlap.aspect_mask.as_raw().trailing_zeros() - base_aspect;
                    let aspect_end = aspect_start + overlap.aspect_mask.as_raw().count_ones();
                    aspect_cuts.push(aspect_start);
                    aspect_cuts.push(aspect_end);

                    let layer_start = overlap.base_array_layer - range.base_array_layer;
                    let layer_end = layer_start + overlap.layer_count;
                    layer_cuts.push(layer_start);
                    layer_cuts.push(layer_end);

                    let mip_start = overlap.base_mip_level - range.base_mip_level;
                    let mip_end = mip_start + overlap.level_count;
                    mip_cuts.push(mip_start);
                    mip_cuts.push(mip_end);
                }

                aspect_cuts.sort_unstable();
                aspect_cuts.dedup();
                layer_cuts.sort_unstable();
                layer_cuts.dedup();
                mip_cuts.sort_unstable();
                mip_cuts.dedup();
            }

            ImageBarrierTransferIter {
                transfers,
                overlaps,
                aspect_cuts,
                layer_cuts,
                mip_cuts,
                base_aspect,
                range,
                aspect_idx: 0,
                layer_idx: 0,
                mip_idx: 0,
                yielded_empty: false,
            }
        })
    }

    fn consume_pending(
        transfers: &mut Vec<ImageOwnershipTransfer>,
        range: vk::ImageSubresourceRange,
    ) -> bool {
        transfers.retain(|transfer| {
            image_subresource_range_intersection(transfer.range, range).is_none()
        });
        transfers.is_empty()
    }

    fn ranges_equal(lhs: vk::ImageSubresourceRange, rhs: vk::ImageSubresourceRange) -> bool {
        lhs.aspect_mask == rhs.aspect_mask
            && lhs.base_array_layer == rhs.base_array_layer
            && lhs.layer_count == rhs.layer_count
            && lhs.base_mip_level == rhs.base_mip_level
            && lhs.level_count == rhs.level_count
    }
}

impl PartialEq for ImageOwnershipTransfer {
    fn eq(&self, other: &Self) -> bool {
        self.dst_queue_family_index == other.dst_queue_family_index
            && self.layouts.old == other.layouts.old
            && self.layouts.new == other.layouts.new
            && self.src_queue_family_index == other.src_queue_family_index
            && self.src_queue_index == other.src_queue_index
            && ImageOwnershipTransfer::ranges_equal(self.range, other.range)
    }
}

#[derive(Clone, Copy, Debug)]
struct ImageQueueOwnershipRelease {
    image: vk::Image,
    layouts: ImageOwnershipLayouts,
    range: vk::ImageSubresourceRange,
}

impl ImageQueueOwnershipRelease {
    fn memory_barrier(
        release: ImageQueueOwnershipRelease,
        src_queue_family_index: u32,
        dst_queue_family_index: u32,
    ) -> vk::ImageMemoryBarrier<'static> {
        vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::empty())
            .old_layout(release.layouts.old)
            .new_layout(release.layouts.new)
            .src_queue_family_index(src_queue_family_index)
            .dst_queue_family_index(dst_queue_family_index)
            .image(release.image)
            .subresource_range(release.range)
    }
}

struct ImageSubresourceRangeDebug(vk::ImageSubresourceRange);

impl std::fmt::Debug for ImageSubresourceRangeDebug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.aspect_mask.fmt(f)?;

        f.write_str(" array: ")?;

        let array_layers = self.0.base_array_layer..self.0.base_array_layer + self.0.layer_count;
        array_layers.fmt(f)?;

        f.write_str(" mip: ")?;

        let mip_levels = self.0.base_mip_level..self.0.base_mip_level + self.0.level_count;
        mip_levels.fmt(f)
    }
}

#[derive(Debug)]
struct NodeIndexedScratch<T> {
    entries: Vec<NodeIndexedScratchEntry<T>>,
    indices: Vec<NodeIndex>,
}

impl<T> NodeIndexedScratch<T> {
    fn clear(&mut self) {
        for &node_idx in self.indices.iter() {
            let Some(entry) = self.entries.get_mut(node_idx) else {
                continue;
            };

            entry.occupied = false;
            entry.values.clear();
        }

        self.indices.clear();
    }

    fn get(&self, node_idx: NodeIndex) -> &[T] {
        self.entries
            .get(node_idx)
            .filter(|entry| entry.occupied)
            .map_or_else(Default::default, |entry| entry.values.as_slice())
    }

    fn push(&mut self, node_idx: NodeIndex, value: T) {
        if self.entries.len() <= node_idx {
            self.entries
                .resize_with(node_idx.saturating_add(1), Default::default);
        }

        let entry = &mut self.entries[node_idx];

        if !entry.occupied {
            entry.occupied = true;
            self.indices.push(node_idx);
        }

        entry.values.push(value);
    }
}

impl<T> Default for NodeIndexedScratch<T> {
    fn default() -> Self {
        Self {
            entries: Default::default(),
            indices: Default::default(),
        }
    }
}

#[derive(Debug)]
struct NodeIndexedScratchEntry<T> {
    occupied: bool,
    values: Vec<T>,
}

impl<T> Default for NodeIndexedScratchEntry<T> {
    fn default() -> Self {
        Self {
            occupied: false,
            values: Default::default(),
        }
    }
}

#[derive(Debug)]
struct PendingTransferNode<H, T> {
    handle: H,
    transfers: Vec<T>,
}

#[derive(Debug)]
struct PendingTransferNodes<H, T> {
    entries: Vec<Option<PendingTransferNode<H, T>>>,
    indices: Vec<NodeIndex>,
}

impl<H, T> PendingTransferNodes<H, T>
where
    H: Copy,
{
    fn new(node_count: usize) -> Self {
        let mut entries = Vec::with_capacity(node_count);
        entries.resize_with(node_count, || None);

        Self {
            entries,
            indices: Vec::new(),
        }
    }

    fn contains(&self, node_idx: NodeIndex) -> bool {
        self.entries[node_idx].is_some()
    }

    fn get(&self, node_idx: NodeIndex) -> Option<(H, &[T])> {
        self.entries[node_idx]
            .as_ref()
            .map(|entry| (entry.handle, entry.transfers.as_slice()))
    }

    fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    fn iter(&self) -> impl Iterator<Item = (NodeIndex, H, &[T])> + '_ {
        self.indices.iter().filter_map(|&node_idx| {
            self.entries[node_idx]
                .as_ref()
                .map(|entry| (node_idx, entry.handle, entry.transfers.as_slice()))
        })
    }

    fn push_transfer(&mut self, node_idx: NodeIndex, handle: H, transfer: T) -> bool {
        let inserted = self.entries[node_idx].is_none();

        if inserted {
            self.indices.push(node_idx);
            self.entries[node_idx] = Some(PendingTransferNode {
                handle,
                transfers: vec![transfer],
            });
        } else {
            let entry = self.entries[node_idx]
                .as_mut()
                .expect("missing pending transfer node");

            entry.handle = handle;
            entry.transfers.push(transfer);
        }

        inserted
    }

    fn remove_where<F>(&mut self, mut remove: F)
    where
        F: FnMut(NodeIndex, H, &mut Vec<T>) -> bool,
    {
        let mut pending_idx = 0;

        while pending_idx < self.indices.len() {
            let node_idx = self.indices[pending_idx];

            let Some(entry) = self.entries[node_idx].as_mut() else {
                self.indices.swap_remove(pending_idx);
                continue;
            };

            if remove(node_idx, entry.handle, &mut entry.transfers) {
                self.entries[node_idx] = None;
                self.indices.swap_remove(pending_idx);
            } else {
                pending_idx += 1;
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
struct PipelineStageAccessFlags {
    access_flags: vk::AccessFlags,
    stage_flags: vk::PipelineStageFlags,
}

impl PipelineStageAccessFlags {
    fn new(access: AccessType) -> Self {
        let (stage_flags, access_flags) = pipeline_stage_access_flags(access);

        Self {
            access_flags,
            stage_flags,
        }
    }

    fn buffer_source_scope(
        access: AccessType,
        queue_flags: vk::QueueFlags,
    ) -> (vk::PipelineStageFlags, vk::AccessFlags) {
        let (stages, accesses) = pipeline_stage_access_flags(access);
        let mut supported = vk::PipelineStageFlags::TOP_OF_PIPE
            | vk::PipelineStageFlags::BOTTOM_OF_PIPE
            | vk::PipelineStageFlags::HOST
            | vk::PipelineStageFlags::ALL_COMMANDS;
        if queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            supported |= Submission::GRAPHICS_STAGES | vk::PipelineStageFlags::ALL_GRAPHICS;
        }
        if queue_flags.contains(vk::QueueFlags::COMPUTE) {
            supported |= vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::DRAW_INDIRECT
                | vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                | vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR;
        }
        if queue_flags.intersects(
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
        ) {
            supported |= vk::PipelineStageFlags::TRANSFER;
        }
        if supported.contains(stages) {
            (stages, accesses)
        } else {
            // Concurrent sharing can retain a producer from a different queue family.
            // Outside a render pass ALL_COMMANDS is queue-safe; do not discard writes.
            (
                vk::PipelineStageFlags::ALL_COMMANDS,
                if is_write_access(access) {
                    vk::AccessFlags::MEMORY_WRITE
                } else {
                    vk::AccessFlags::empty()
                },
            )
        }
    }

    fn is_micromap_access(access: AccessType) -> bool {
        matches!(
            access,
            AccessType::MicromapBuildRead
                | AccessType::MicromapBuildWrite
                | AccessType::MicromapBuildInputRead
                | AccessType::MicromapBuildScratchReadWrite
                | AccessType::MicromapBuildBufferRead
                | AccessType::MicromapBuildBufferWrite
                | AccessType::AccelerationStructureBuildMicromapRead
        )
    }

    fn is_read_only_graphics_access(access: AccessType) -> bool {
        use AccessType::*;

        matches!(
            access,
            IndirectBuffer
                | IndexBuffer
                | VertexBuffer
                | VertexShaderReadUniformBuffer
                | VertexShaderReadSampledImageOrUniformTexelBuffer
                | VertexShaderReadOther
                | TessellationControlShaderReadUniformBuffer
                | TessellationControlShaderReadSampledImageOrUniformTexelBuffer
                | TessellationControlShaderReadOther
                | TessellationEvaluationShaderReadUniformBuffer
                | TessellationEvaluationShaderReadSampledImageOrUniformTexelBuffer
                | TessellationEvaluationShaderReadOther
                | GeometryShaderReadUniformBuffer
                | GeometryShaderReadSampledImageOrUniformTexelBuffer
                | GeometryShaderReadOther
                | FragmentShaderReadUniformBuffer
                | FragmentShaderReadSampledImageOrUniformTexelBuffer
                | FragmentShaderReadOther
                | AnyShaderReadUniformBuffer
                | AnyShaderReadUniformBufferOrVertexBuffer
                | AnyShaderReadSampledImageOrUniformTexelBuffer
                | AnyShaderReadOther
                | TaskShaderReadUniformBuffer
                | TaskShaderReadSampledImageOrUniformTexelBuffer
                | TaskShaderReadOther
                | MeshShaderReadUniformBuffer
                | MeshShaderReadSampledImageOrUniformTexelBuffer
                | MeshShaderReadOther
        )
    }

    fn record_external_accesses(history: &mut [PipelineStageAccessFlags], cmd: &CommandData) {
        for exec in &cmd.execs {
            for (node_idx, accesses) in exec.accesses.iter() {
                for access in accesses {
                    history[node_idx].union(PipelineStageAccessFlags::new(access.access));
                }
            }
        }
    }

    fn union(&mut self, other: Self) {
        self.stage_flags |= other.stage_flags;
        self.access_flags |= other.access_flags;
    }

    fn with_external_render_pass_accesses<T>(
        node_count: usize,
        f: impl FnOnce(&mut [PipelineStageAccessFlags]) -> T,
    ) -> T {
        thread_local! {
            static ACCESSES: RefCell<Vec<PipelineStageAccessFlags>> = const { RefCell::new(Vec::new()) };
        }

        // Pool callbacks may reenter leasing; do not hold the TLS borrow while they run.
        let mut accesses = ACCESSES.with_borrow_mut(take);
        accesses.clear();
        accesses.resize(node_count, PipelineStageAccessFlags::default());
        let result = f(&mut accesses);
        ACCESSES.with_borrow_mut(|cached| {
            if accesses.capacity() >= cached.capacity() {
                *cached = accesses;
            }
        });
        result
    }
}

#[derive(Debug, Default)]
pub(crate) struct PreparedStreamRecording {
    resources: Mutex<Vec<CommandRecordingResources>>,
}

#[derive(Debug)]
struct QueueOwnershipRelease {
    _cmd_buf: Lease<CommandBuffer>,
    _fence: Fence,
    semaphore: vk::Semaphore,
}

impl QueueOwnershipRelease {
    /// Builds and submits a release barrier command buffer for each release group, calling
    /// `submit_release` to perform the final queue submission.
    fn submit<P>(
        pool: &mut P,
        release_groups: &[QueueOwnershipReleaseGroup],
        target_queue_family_index: u32,
        submit_release: impl Fn(
            &Device,
            vk::Queue,
            vk::CommandBuffer,
            vk::Fence,
            vk::Semaphore,
        ) -> Result<(), DriverError>,
    ) -> Result<Vec<QueueOwnershipRelease>, DriverError>
    where
        P: Pool<CommandBufferInfo, CommandBuffer>,
    {
        let mut releases = Vec::new();

        if !release_groups.is_empty() {
            for group in release_groups {
                let mut release_cmd =
                    pool.resource(CommandBufferInfo::new(group.src_queue_family_index as _))?;
                let mut release_fence = Fence::create(&release_cmd.device, false)?;

                #[cfg(feature = "checked")]
                {
                    release_fence.wait()?;
                    release_fence.reset()?;
                }

                let semaphore = release_cmd.release_semaphore()?;

                release_cmd.set_debug_name(lazy_str!(
                    "queue ownership release qf{}:{} -> qf{}",
                    group.src_queue_family_index,
                    group.src_queue_index,
                    target_queue_family_index
                ));

                Device::begin_command_buffer(
                    &release_cmd.device,
                    release_cmd.handle,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;

                {
                    let _ = CommandBufferDebugLabel::begin(
                        &release_cmd,
                        lazy_str!(
                            "queue ownership release qf{}:{} -> qf{}",
                            group.src_queue_family_index,
                            group.src_queue_index,
                            target_queue_family_index
                        ),
                    );

                    SUBMIT.with_borrow_mut(|tls| {
                        let _ =
                            CommandBufferDebugLabel::begin(&release_cmd, "queue ownership barrier");

                        tls.release_image_barriers.clear();
                        tls.release_buffer_barriers.clear();
                        tls.release_buffer_barriers.reserve(group.buffers.len());
                        tls.release_image_barriers.reserve(group.images.len());

                        tls.release_buffer_barriers.extend(group.buffers.iter().map(
                            |&(handle, range)| {
                                vk::BufferMemoryBarrier::default()
                                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                                    .dst_access_mask(vk::AccessFlags::empty())
                                    .src_queue_family_index(group.src_queue_family_index)
                                    .dst_queue_family_index(target_queue_family_index)
                                    .buffer(handle)
                                    .offset(range.start)
                                    .size(range.end - range.start)
                            },
                        ));

                        tls.release_image_barriers
                            .extend(group.images.iter().copied().map(|release| {
                                ImageQueueOwnershipRelease::memory_barrier(
                                    release,
                                    group.src_queue_family_index,
                                    target_queue_family_index,
                                )
                            }));

                        unsafe {
                            release_cmd.device.cmd_pipeline_barrier(
                                release_cmd.handle,
                                vk::PipelineStageFlags::ALL_COMMANDS,
                                vk::PipelineStageFlags::ALL_COMMANDS,
                                vk::DependencyFlags::empty(),
                                &[],
                                tls.release_buffer_barriers.as_slice(),
                                tls.release_image_barriers.as_slice(),
                            );
                        }
                    });

                    Device::with_queue(
                        &release_cmd.device,
                        group.src_queue_family_index,
                        group.src_queue_index,
                        |queue| {
                            Device::end_command_buffer(&release_cmd.device, release_cmd.handle)?;
                            submit_release(
                                &release_cmd.device,
                                queue,
                                release_cmd.handle,
                                release_fence.handle,
                                semaphore,
                            )?;

                            release_fence.mark_queued();

                            Ok::<_, DriverError>(())
                        },
                    )?;
                }

                releases.push(QueueOwnershipRelease {
                    _cmd_buf: release_cmd,
                    _fence: release_fence,
                    semaphore,
                });
            }
        }

        Ok(releases)
    }
}

#[derive(Debug)]
struct QueueOwnershipReleaseGroup {
    buffers: Vec<(vk::Buffer, BufferSubresourceRange)>,
    images: Vec<ImageQueueOwnershipRelease>,
    src_queue_family_index: u32,
    src_queue_index: u32,
}

impl QueueOwnershipReleaseGroup {
    fn get_or_insert(
        groups: &mut Vec<QueueOwnershipReleaseGroup>,
        src_queue_family_index: u32,
        src_queue_index: u32,
    ) -> &mut QueueOwnershipReleaseGroup {
        if let Some(group_idx) = groups.iter().position(|group| {
            group.src_queue_family_index == src_queue_family_index
                && group.src_queue_index == src_queue_index
        }) {
            return &mut groups[group_idx];
        }

        groups.push(QueueOwnershipReleaseGroup {
            buffers: Vec::new(),
            images: Vec::new(),
            src_queue_family_index,
            src_queue_index,
        });
        groups.last_mut().expect("missing ownership release group")
    }
}

#[derive(Clone, Copy, Debug)]
struct QueueOwnershipReleaseWait {
    semaphore: vk::Semaphore,
    stage_mask: vk::PipelineStageFlags2,
    value: u64,
    device_index: u32,
}

/// Submission payload for [`RecordedSubmission::queue_submit`].
#[derive(Clone, Copy, Debug)]
pub enum QueueSubmitInfo<'a> {
    /// Submit using `vkQueueSubmit`.
    ///
    /// See [`vkQueueSubmit`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueSubmit.html).
    QueueSubmit {
        /// Semaphores to wait on before execution begins.
        waits: &'a [SemaphoreSubmitInfo],

        /// Semaphores to signal after execution completes.
        signals: &'a [SemaphoreSubmitInfo],
    },

    /// Submit using `vkQueueSubmit2`.
    ///
    /// See [`vkQueueSubmit2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueSubmit2.html).
    QueueSubmit2 {
        /// Semaphores to wait on before execution begins.
        waits: &'a [SemaphoreSubmit2Info],

        /// Semaphores to signal after execution completes.
        signals: &'a [SemaphoreSubmit2Info],
    },
}

impl QueueSubmitInfo<'static> {
    /// A `vkQueueSubmit` payload with no waits or signals.
    pub const QUEUE_SUBMIT: Self = Self::QueueSubmit {
        waits: &[],
        signals: &[],
    };

    /// A `vkQueueSubmit2` payload with no waits or signals.
    pub const QUEUE_SUBMIT2: Self = Self::QueueSubmit2 {
        waits: &[],
        signals: &[],
    };
}

impl<'a> QueueSubmitInfo<'a> {
    /// Creates a `vkQueueSubmit` payload.
    pub fn queue_submit(
        waits: &'a [SemaphoreSubmitInfo],
        signals: &'a [SemaphoreSubmitInfo],
    ) -> Self {
        Self::QueueSubmit { waits, signals }
    }

    /// Creates a `vkQueueSubmit2` payload.
    pub fn queue_submit2(
        waits: &'a [SemaphoreSubmit2Info],
        signals: &'a [SemaphoreSubmit2Info],
    ) -> Self {
        Self::QueueSubmit2 { waits, signals }
    }
}

impl<'a> From<(&'a [SemaphoreSubmitInfo], &'a [SemaphoreSubmitInfo])> for QueueSubmitInfo<'a> {
    fn from((waits, signals): (&'a [SemaphoreSubmitInfo], &'a [SemaphoreSubmitInfo])) -> Self {
        Self::QueueSubmit { waits, signals }
    }
}

impl<'a> From<(&'a [SemaphoreSubmit2Info], &'a [SemaphoreSubmit2Info])> for QueueSubmitInfo<'a> {
    fn from((waits, signals): (&'a [SemaphoreSubmit2Info], &'a [SemaphoreSubmit2Info])) -> Self {
        Self::QueueSubmit2 { waits, signals }
    }
}

/// Selects which pending work from a [`Submission`] should be recorded.
#[derive(Clone, Copy, Debug)]
pub enum RecordSelection<'a> {
    /// Record all remaining work.
    All,

    /// Record prerequisite work, excluding commands that directly access the target node.
    Dependencies(AnyNode),

    /// Record work required by the target node.
    Node(AnyNode),

    /// Record work required by all of the target nodes.
    ///
    /// Nodes are processed sequentially in slice order against the same evolving submission state.
    Nodes(&'a [AnyNode]),
}

impl<'a> RecordSelection<'a> {
    /// Creates a selection that records prerequisite work for `node` without recording commands that
    /// directly access it.
    pub fn dependencies(node: impl Into<AnyNode>) -> Self {
        Self::Dependencies(node.into())
    }

    /// Creates a selection that records work required by `node`.
    pub fn node(node: impl Into<AnyNode>) -> Self {
        Self::Node(node.into())
    }

    /// Creates a selection that records work required by all `nodes`.
    ///
    /// Nodes are processed in slice order.
    pub fn nodes(nodes: &'a [AnyNode]) -> Self {
        Self::Nodes(nodes)
    }
}

impl<'a> From<AnyNode> for RecordSelection<'a> {
    fn from(node: AnyNode) -> Self {
        Self::Node(node)
    }
}

macro_rules! record_selection_from_node {
    ($node:ty) => {
        impl<'a> From<$node> for RecordSelection<'a> {
            fn from(node: $node) -> Self {
                Self::Node(node.into())
            }
        }
    };
}

record_selection_from_node!(crate::node::AnyAccelerationStructureNode);
record_selection_from_node!(crate::node::AnyBufferNode);
record_selection_from_node!(crate::node::AnyImageNode);
record_selection_from_node!(crate::node::AnyMicromapNode);
record_selection_from_node!(crate::node::AccelerationStructureNode);
record_selection_from_node!(crate::node::AccelerationStructureLeaseNode);
record_selection_from_node!(crate::node::BufferNode);
record_selection_from_node!(crate::node::BufferLeaseNode);
record_selection_from_node!(crate::node::ImageNode);
record_selection_from_node!(crate::node::ImageLeaseNode);
record_selection_from_node!(crate::node::MicromapNode);
record_selection_from_node!(crate::node::MicromapLeaseNode);
record_selection_from_node!(crate::node::SwapchainImageNode);

/// Graph-side recorded payload for a command buffer that has already been recorded.
#[derive(Debug)]
#[read_only::cast]
pub struct RecordedSubmission<Cb> {
    cmd_buf: Cb,
    queue_ownership_release_waits: Vec<QueueOwnershipReleaseWait>,
    state: Arc<Mutex<RecordedSubmissionState>>,
}

impl<Cb> RecordedSubmission<Cb>
where
    Cb: AsRef<CommandBuffer>,
{
    fn attach_locked(
        state: &mut RecordedSubmissionState,
        cmd_buf: &CommandBuffer,
        queue_index: u32,
    ) -> Option<SubmittedTimestampQueries> {
        let queue_family_index = cmd_buf.info.queue_family_index;

        for (node_idx, ranges) in &state.submission.exclusive_buffer_ranges {
            if let Some(resource) = state.submission.graph.resources[*node_idx].as_buffer() {
                resource.set_sharing_ranges(
                    SharingMode::Exclusive(Some((queue_family_index, queue_index))),
                    ranges.as_slice(),
                );
            }
        }

        for (node_idx, ranges) in &state.submission.exclusive_image_ranges {
            if let Some(resource) = state.submission.graph.resources[*node_idx].as_image() {
                resource.set_sharing_ranges(
                    SharingMode::Exclusive(Some((queue_family_index, queue_index))),
                    ranges.as_slice(),
                );
            }
        }

        let queue = (queue_family_index, queue_index);

        for resource_set_idx in state.submission.touched_image_sets.ones() {
            let resource_set = state
                .submission
                .graph
                .resource_sets
                .get_image(ResourceSetIndex::new(resource_set_idx))
                .expect("touched resource set is not an image set");
            if resource_set.queue() == Some(queue) {
                continue;
            }

            for member in resource_set.unique_members() {
                if member.image().info.sharing_mode != vk::SharingMode::CONCURRENT {
                    member.image().set_sharing_ranges(
                        SharingMode::Exclusive(Some(queue)),
                        slice::from_ref(&member.subresource()),
                    );
                }
            }
        }

        // Member queue writes can invalidate overlapping set queues, so publish after all writes.
        for resource_set_idx in state.submission.touched_image_sets.ones() {
            state
                .submission
                .graph
                .resource_sets
                .get_image(ResourceSetIndex::new(resource_set_idx))
                .expect("touched resource set is not an image set")
                .publish_queue(queue);
        }

        state.submission.query_pool_results.take()
    }

    /// Submits this recorded submission using either `vkQueueSubmit` or `vkQueueSubmit2`.
    pub fn queue_submit<'a>(
        &mut self,
        fence: &mut Fence,
        queue_index: u32,
        submit_info: impl Into<QueueSubmitInfo<'a>>,
    ) -> Result<(), DriverError> {
        #[cfg(feature = "checked")]
        if fence.queued.get() {
            fence.wait()?;
            fence.reset()?;
        }

        let command_buffer = self.cmd_buf.as_ref();
        let device = &command_buffer.device;
        let queue_family_index = command_buffer.info.queue_family_index;

        match submit_info.into() {
            QueueSubmitInfo::QueueSubmit { waits, signals } => {
                SemaphoreSubmitInfo::check_args(waits, signals)?;

                let extra_waits = self.queue_ownership_release_waits.as_slice();
                let wait_count = waits.len() + extra_waits.len();

                Device::with_queue(device, queue_family_index, queue_index, |queue| {
                    SUBMIT.with_borrow_mut(|tls| {
                        tls.wait_semaphores.clear();
                        tls.wait_stage_masks.clear();
                        tls.signal_semaphores.clear();
                        tls.wait_semaphores.reserve(wait_count);
                        tls.wait_stage_masks.reserve(wait_count);
                        tls.signal_semaphores.reserve(signals.len());

                        tls.wait_semaphores
                            .extend(waits.iter().map(|wait| wait.semaphore));
                        tls.wait_stage_masks.extend(
                            waits.iter().map(|wait| {
                                SemaphoreSubmitInfo::stage_mask_legacy(wait.stage_mask)
                            }),
                        );
                        tls.wait_semaphores
                            .extend(extra_waits.iter().map(|wait| wait.semaphore));
                        tls.wait_stage_masks.extend(
                            extra_waits.iter().map(|wait| {
                                SemaphoreSubmitInfo::stage_mask_legacy(wait.stage_mask)
                            }),
                        );
                        tls.signal_semaphores
                            .extend(signals.iter().map(|signal| signal.semaphore));

                        let mut submit_info = vk::SubmitInfo::default()
                            .command_buffers(slice::from_ref(&command_buffer.handle))
                            .signal_semaphores(tls.signal_semaphores.as_slice());

                        if !tls.wait_semaphores.is_empty() {
                            submit_info = submit_info
                                .wait_semaphores(tls.wait_semaphores.as_slice())
                                .wait_dst_stage_mask(tls.wait_stage_masks.as_slice());
                        }

                        #[cfg(test)]
                        test::fail_queue_submit()?;

                        Device::queue_submit(
                            device,
                            queue,
                            slice::from_ref(&submit_info),
                            fence.handle,
                        )?;

                        Ok::<(), DriverError>(())
                    })
                })?;
                fence.mark_queued();
            }
            QueueSubmitInfo::QueueSubmit2 { waits, signals } => {
                SemaphoreSubmit2Info::check_args(device, waits, signals)?;

                let extra_waits = self.queue_ownership_release_waits.as_slice();
                let wait_count = waits.len() + extra_waits.len();

                Device::with_queue(device, queue_family_index, queue_index, |queue| {
                    SUBMIT.with_borrow_mut(|tls| {
                        tls.wait_infos.clear();
                        tls.signal_infos.clear();
                        tls.wait_infos.reserve(wait_count);
                        tls.signal_infos.reserve(signals.len());

                        tls.wait_infos.extend(waits.iter().map(|wait| {
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(wait.semaphore)
                                .stage_mask(wait.stage_mask)
                                .value(wait.value)
                                .device_index(wait.device_index)
                        }));
                        tls.wait_infos.extend(extra_waits.iter().map(|wait| {
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(wait.semaphore)
                                .stage_mask(wait.stage_mask)
                                .value(wait.value)
                                .device_index(wait.device_index)
                        }));
                        tls.signal_infos.extend(signals.iter().map(|signal| {
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(signal.semaphore)
                                .stage_mask(signal.stage_mask)
                                .value(signal.value)
                                .device_index(signal.device_index)
                        }));

                        let command_buffer_info = vk::CommandBufferSubmitInfo::default()
                            .command_buffer(command_buffer.handle);
                        let mut submit_info = vk::SubmitInfo2::default()
                            .command_buffer_infos(slice::from_ref(&command_buffer_info));

                        if !tls.wait_infos.is_empty() {
                            submit_info =
                                submit_info.wait_semaphore_infos(tls.wait_infos.as_slice());
                        }

                        if !tls.signal_infos.is_empty() {
                            submit_info =
                                submit_info.signal_semaphore_infos(tls.signal_infos.as_slice());
                        }

                        #[cfg(test)]
                        test::fail_queue_submit()?;

                        Device::queue_submit2(
                            device,
                            queue,
                            slice::from_ref(&submit_info),
                            fence.handle,
                        )?;

                        Ok::<(), DriverError>(())
                    })
                })?;
                fence.mark_queued();
            }
        }

        let mut state = self
            .state
            .lock()
            .expect("poisoned recorded submission state");

        #[cfg(feature = "checked")]
        let timestamp_query_graph_id = state.submission.graph.graph_id();

        let submitted_timestamps = Self::attach_locked(&mut state, command_buffer, queue_index);
        // Only commands actually recorded into this submission reached the successful
        // queue submit above. Unselected graph commands remain pending.
        for command in &state.submission.submit_retained {
            command.cmd.tracking.signal_submitted();
        }
        drop(state);

        #[cfg(feature = "checked")]
        fence.set_timestamps(TimestampQueryPool::pending(timestamp_query_graph_id));

        #[cfg(not(feature = "checked"))]
        fence.set_timestamps(TimestampQueryPool::pending());

        if let Some(submitted_timestamps) = submitted_timestamps {
            fence.drop_fence_droppable(submitted_timestamps);
        } else {
            fence.drop_fence_droppable(TimestampQueryCompletion);
        }

        fence.drop_fence_droppable(RecordedSubmissionDrop(self.state.clone()));
        self.queue_ownership_release_waits.clear();

        Ok(())
    }
}

#[derive(Debug)]
struct RecordedSubmissionDrop(Arc<Mutex<RecordedSubmissionState>>);

impl FenceDroppable for RecordedSubmissionDrop {
    fn fence_signaled(&mut self, _fence: &Fence) {
        self.0
            .lock()
            .expect("poisoned recorded submission state")
            .signal_executed();
    }
}

#[derive(Debug)]
struct RecordedSubmissionState {
    _releases: Vec<QueueOwnershipRelease>,
    executed: bool,
    submission: Submission,
}

impl RecordedSubmissionState {
    fn signal_executed(&mut self) {
        if self.executed {
            return;
        }

        self.executed = true;
        self.submission.signal_executed();
    }
}

/// A [`Submission`] bound to a specific command buffer for explicit recording and submission.
#[derive(Debug)]
#[read_only::cast]
pub struct Recording<'p, P, Cb> {
    /// The command buffer bound to this recording.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub cmd_buf: Cb,

    /// The pool used to allocate resources used during recording.
    ///
    /// _Note:_ This field may be mutated in between calls to `record`. The updated pool will be
    /// used for future calls to record.
    #[readonly]
    pub resource_pool: &'p mut P,

    ownership: RecordingOwnership,
    submission: Submission,
}

impl<'p, P, Cb> Recording<'p, P, Cb>
where
    P: SubmissionPool,
    Cb: AsRef<CommandBuffer>,
{
    /// Records any remaining graph commands into this submission's command buffer.
    ///
    /// When `selection` is [`RecordSelection::Nodes`], nodes are processed sequentially in the
    /// provided slice order and each step mutates the remaining submission state.
    #[profiling::function]
    pub fn record<'s>(
        &mut self,
        selection: impl Into<RecordSelection<'s>>,
    ) -> Result<(), DriverError> {
        self.submission.record_selection_impl(
            self.resource_pool,
            self.cmd_buf.as_ref(),
            selection.into(),
            &mut self.ownership,
        )
    }
}

impl<'p, P, Cb> Recording<'p, P, Cb>
where
    Cb: AsRef<CommandBuffer>,
{
    /// Finalizes recording into a recorded submission for a caller-owned command buffer.
    pub fn finish(self) -> Result<RecordedSubmission<Cb>, DriverError>
    where
        P: Pool<CommandBufferInfo, CommandBuffer>,
    {
        let Self {
            ownership: _,
            cmd_buf,
            resource_pool,
            submission,
        } = self;

        let queue_family_index = cmd_buf.as_ref().info.queue_family_index;
        let releases = QueueOwnershipRelease::submit(
            resource_pool,
            &submission.queue_ownership_release_groups,
            queue_family_index,
            |device, queue, cmd_handle, fence, semaphore| {
                let submit_info = vk::SubmitInfo::default()
                    .command_buffers(slice::from_ref(&cmd_handle))
                    .signal_semaphores(slice::from_ref(&semaphore));
                Device::queue_submit(device, queue, slice::from_ref(&submit_info), fence)
            },
        )?;
        let waits = releases
            .iter()
            .map(|release| QueueOwnershipReleaseWait {
                semaphore: release.semaphore,
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                value: 0,
                device_index: 0,
            })
            .collect();

        Ok(submission.into_recorded_submission(cmd_buf, releases, waits))
    }

    /// Returns `true` when this submission contains no more commands to record.
    pub fn is_empty(&self) -> bool {
        self.submission.is_empty()
    }

    /// Returns a borrow of the resource or persistent resource set represented by the given node.
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: ResourceNode,
    {
        self.submission.resource(resource_node)
    }
}

#[derive(Debug)]
enum RecordingDescriptorSet {
    Automatic(RawDescriptorSet),
    Supplied(DescriptorSet),
}

impl RecordingDescriptorSet {
    fn handle(&self) -> vk::DescriptorSet {
        match self {
            Self::Automatic(descriptor_set) => **descriptor_set,
            Self::Supplied(descriptor_set) => descriptor_set.handle(),
        }
    }
}

#[derive(Debug, Default)]
struct RecordingOwnership {
    // These ranges are effectively owned by this recording, but global ownership is not updated
    // until its command buffer is submitted successfully.
    buffers: HashMap<usize, Vec<BufferSubresourceRange>>,
    images: HashMap<usize, ImageOwnership>,
    image_set_images: HashMap<PhysicalImageId, ImageOwnership>,
}

impl RecordingOwnership {
    fn claim_buffer(
        &mut self,
        node_idx: usize,
        range: BufferSubresourceRange,
    ) -> SmallVec<[BufferSubresourceRange; 4]> {
        let claimed = self.buffers.entry(node_idx).or_default();
        let mut unclaimed = SmallVec::<[BufferSubresourceRange; 4]>::from_slice(&[range]);

        for &claimed_range in claimed.iter() {
            let mut remaining = SmallVec::<[BufferSubresourceRange; 4]>::new();

            for range in unclaimed.drain(..) {
                let Some(overlap) = range.intersection(claimed_range) else {
                    remaining.push(range);
                    continue;
                };

                if range.start < overlap.start {
                    remaining.push(BufferSubresourceRange {
                        start: range.start,
                        end: overlap.start,
                    });
                }
                if overlap.end < range.end {
                    remaining.push(BufferSubresourceRange {
                        start: overlap.end,
                        end: range.end,
                    });
                }
            }

            unclaimed = remaining;
            if unclaimed.is_empty() {
                break;
            }
        }

        claimed.extend(unclaimed.iter().copied());
        unclaimed
    }

    fn claim_image(
        &mut self,
        node_idx: usize,
        info: ImageInfo,
        range: vk::ImageSubresourceRange,
    ) -> SmallVec<[vk::ImageSubresourceRange; 4]> {
        let range = info.resolve_subresource_counts(range);

        match self.images.entry(node_idx) {
            Entry::Occupied(mut entry) => entry.get_mut().claim(info, range),
            Entry::Vacant(entry) => {
                entry.insert(ImageOwnership::new(info, range));

                SmallVec::from_slice(&[range])
            }
        }
    }

    fn claim_image_set_image(
        &mut self,
        image_id: PhysicalImageId,
        info: ImageInfo,
        range: vk::ImageSubresourceRange,
    ) -> SmallVec<[vk::ImageSubresourceRange; 4]> {
        let range = info.resolve_subresource_counts(range);

        match self.image_set_images.entry(image_id) {
            Entry::Occupied(mut entry) => entry.get_mut().claim(info, range),
            Entry::Vacant(entry) => {
                entry.insert(ImageOwnership::new(info, range));

                SmallVec::from_slice(&[range])
            }
        }
    }

    fn exclusive_transfer_source(
        sharing: SharingMode,
        queue_family_index: u32,
    ) -> Option<(u32, u32)> {
        let SharingMode::Exclusive(Some((src_queue_family_index, src_queue_index))) = sharing
        else {
            return None;
        };

        (src_queue_family_index != queue_family_index)
            .then_some((src_queue_family_index, src_queue_index))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResourceSetSynchronization {
    DeferredToOuterBoundary,
    Enabled,
}

#[derive(Default)]
struct Schedule {
    access_index: CommandAccessIndex,
    cmds: Vec<usize>,
    local_of_global: Vec<usize>,
    successors: Vec<Vec<usize>>,
    predecessor_counts: Vec<usize>,
    remaining_predecessors: Vec<usize>,
    ready: BTreeSet<(usize, Reverse<usize>)>,
    reordered: Vec<usize>,
    node_schedule: ScheduleScratch,
}

impl Schedule {
    #[inline(always)]
    fn add_resource_chain(
        resource_cmds: &[usize],
        chain_count: usize,
        local_of_global: &[usize],
        successors: &mut [Vec<usize>],
        predecessor_counts: &mut [usize],
        omitted_predecessor_counts: &mut [usize],
    ) {
        let mut previous = Option::<usize>::None;

        for &cmd_idx in resource_cmds {
            let Some(&local_idx) = local_of_global.get(cmd_idx) else {
                continue;
            };

            if local_idx == usize::MAX {
                continue;
            }

            if let Some(previous_idx) = previous {
                successors[previous_idx].push(local_idx);
                predecessor_counts[local_idx] += chain_count;
                if chain_count > 1 {
                    omitted_predecessor_counts[local_idx] += chain_count - 1;
                }
            }

            previous = Some(local_idx);
        }
    }

    #[profiling::function]
    fn reorder_cmds(&mut self, end_cmd_idx: usize) {
        if self.cmds.len() < 3 {
            return;
        }

        let cmd_count = self.cmds.len();

        self.local_of_global.resize(end_cmd_idx, usize::MAX);
        self.local_of_global.fill(usize::MAX);

        for (local_idx, &cmd_idx) in self.cmds.iter().enumerate() {
            self.local_of_global[cmd_idx] = local_idx;
        }

        for successors in &mut self.successors {
            successors.clear();
        }
        self.successors.resize_with(cmd_count, Vec::new);
        self.predecessor_counts.resize(cmd_count, 0);
        self.predecessor_counts.fill(0);
        self.remaining_predecessors.resize(cmd_count, 0);

        // A successful prior reorder consumes every stored incoming edge.
        debug_assert!(self.remaining_predecessors.iter().all(|&count| count == 0));

        // Consecutive selected users of each resource form a dependency chain. This preserves the
        // original relative order for every shared-resource pair while still allowing unrelated
        // command chains to be grouped for locality.
        // Comparing whole use chains only pays off for resource-heavy schedules such as bindless
        // arrays. Sparse schedules retain the original construction path.
        let group_resource_chains =
            self.access_index.cmds_by_node.len() >= cmd_count.saturating_mul(4);
        if group_resource_chains {
            let mut resource_idx = 0;
            let mut compressed_resource_chains = false;
            while resource_idx < self.access_index.cmds_by_node.len() {
                let resource_cmds = &self.access_index.cmds_by_node[resource_idx];
                let mut run_end = resource_idx + 1;
                while self.access_index.cmds_by_node.get(run_end) == Some(resource_cmds) {
                    run_end += 1;
                }
                let chain_count = run_end - resource_idx;
                compressed_resource_chains |= chain_count > 1;

                Schedule::add_resource_chain(
                    resource_cmds,
                    chain_count,
                    &self.local_of_global,
                    &mut self.successors,
                    &mut self.predecessor_counts,
                    &mut self.remaining_predecessors,
                );
                resource_idx = run_end;
            }

            for resource_set_cmds in &self.access_index.cmds_by_resource_set {
                Schedule::add_resource_chain(
                    resource_set_cmds,
                    1,
                    &self.local_of_global,
                    &mut self.successors,
                    &mut self.predecessor_counts,
                    &mut self.remaining_predecessors,
                );
            }

            if compressed_resource_chains {
                for (remaining, &predecessor_count) in self
                    .remaining_predecessors
                    .iter_mut()
                    .zip(&self.predecessor_counts)
                {
                    debug_assert!(*remaining <= predecessor_count);
                    *remaining = predecessor_count - *remaining;
                }
            } else {
                self.remaining_predecessors
                    .clone_from(&self.predecessor_counts);
            }
        } else {
            for resource_cmds in &self.access_index.cmds_by_node {
                Schedule::add_resource_chain(
                    resource_cmds,
                    1,
                    &self.local_of_global,
                    &mut self.successors,
                    &mut self.predecessor_counts,
                    &mut self.remaining_predecessors,
                );
            }

            for resource_set_cmds in &self.access_index.cmds_by_resource_set {
                Schedule::add_resource_chain(
                    resource_set_cmds,
                    1,
                    &self.local_of_global,
                    &mut self.successors,
                    &mut self.predecessor_counts,
                    &mut self.remaining_predecessors,
                );
            }

            self.remaining_predecessors
                .clone_from(&self.predecessor_counts);
        }

        self.ready.clear();

        for local_idx in self
            .remaining_predecessors
            .iter()
            .enumerate()
            .filter_map(|(idx, remaining)| (*remaining == 0).then_some(idx))
        {
            self.ready.insert((0, Reverse(local_idx)));
        }

        self.reordered.clear();
        self.reordered.reserve(cmd_count);

        while let Some((_, Reverse(local_idx))) = self.ready.pop_last() {
            self.reordered.push(self.cmds[local_idx]);

            for &successor_idx in &self.successors[local_idx] {
                let remaining = &mut self.remaining_predecessors[successor_idx];

                debug_assert!(*remaining > 0);

                *remaining -= 1;

                if *remaining == 0 {
                    self.ready.insert((
                        self.predecessor_counts[successor_idx],
                        Reverse(successor_idx),
                    ));
                }
            }
        }

        assert_eq!(
            self.reordered.len(),
            cmd_count,
            "command dependency cycle detected"
        );

        self.cmds.clear();
        self.cmds.append(&mut self.reordered);
    }

    fn schedule_dependency_cmds_before_target_access(
        target_node_idx: usize,
        first_target_cmd_idx: usize,
        schedule: &mut Schedule,
    ) {
        let required_node_prefixes = schedule
            .access_index
            .read_nodes_for_cmd(first_target_cmd_idx)
            .filter(|&node_idx| node_idx != target_node_idx)
            .map(|node_idx| (node_idx, first_target_cmd_idx))
            .collect::<SmallVec<[_; 8]>>();
        let required_resource_set_prefixes = schedule
            .access_index
            .read_resource_sets_for_cmd(first_target_cmd_idx)
            .map(|resource_set_idx| (resource_set_idx, first_target_cmd_idx))
            .collect::<SmallVec<[_; 2]>>();

        schedule.schedule_required_prefixes(required_node_prefixes, required_resource_set_prefixes);
    }

    fn schedule_required_node_prefixes(
        &mut self,
        required_prefixes: impl IntoIterator<Item = (usize, usize)>,
    ) {
        self.schedule_required_prefixes(
            required_prefixes,
            std::iter::empty::<(ResourceSetIndex, usize)>(),
        );
    }

    fn schedule_required_prefixes(
        &mut self,
        required_node_prefixes: impl IntoIterator<Item = (usize, usize)>,
        required_resource_set_prefixes: impl IntoIterator<Item = (ResourceSetIndex, usize)>,
    ) {
        self.cmds.clear();
        self.node_schedule.covered_resource_set_prefixes.clear();
        self.node_schedule
            .covered_resource_set_prefixes
            .resize(self.access_index.cmds_by_resource_set.len(), 0);
        self.node_schedule.covered_node_prefixes.clear();
        self.node_schedule
            .covered_node_prefixes
            .resize(self.access_index.cmds_by_node.len(), 0);
        self.node_schedule.pending_cmds.clear();
        self.node_schedule.selected_cmds.clear();
        self.node_schedule.selected_cmds.grow(
            self.access_index
                .accessed_nodes_by_cmd
                .len()
                .max(self.access_index.accessed_resource_sets_by_cmd.len()),
        );

        for (node_idx, end_cmd_idx) in required_node_prefixes {
            ScheduleScratch::schedule_node_prefix(
                &self.access_index,
                &mut self.cmds,
                &mut self.node_schedule,
                node_idx,
                end_cmd_idx,
            );
        }

        for (resource_set_idx, end_cmd_idx) in required_resource_set_prefixes {
            ScheduleScratch::schedule_resource_set_prefix(
                &self.access_index,
                &mut self.cmds,
                &mut self.node_schedule,
                resource_set_idx,
                end_cmd_idx,
            );
        }

        while let Some(cmd_idx) = self.node_schedule.pending_cmds.pop() {
            for node_idx in self.access_index.read_nodes_for_cmd(cmd_idx) {
                ScheduleScratch::schedule_node_prefix(
                    &self.access_index,
                    &mut self.cmds,
                    &mut self.node_schedule,
                    node_idx,
                    cmd_idx + 1,
                );
            }

            for resource_set_idx in self.access_index.read_resource_sets_for_cmd(cmd_idx) {
                ScheduleScratch::schedule_resource_set_prefix(
                    &self.access_index,
                    &mut self.cmds,
                    &mut self.node_schedule,
                    resource_set_idx,
                    cmd_idx + 1,
                );
            }
        }

        self.cmds.sort_unstable();
    }
}

#[derive(Default)]
struct ScheduleScratch {
    covered_resource_set_prefixes: Vec<usize>,
    covered_node_prefixes: Vec<usize>,
    pending_cmds: Vec<usize>,
    selected_cmds: FixedBitSet,
}

impl ScheduleScratch {
    fn schedule_node_prefix(
        access_index: &CommandAccessIndex,
        schedule: &mut Vec<usize>,
        scratch: &mut ScheduleScratch,
        node_idx: usize,
        end_cmd_idx: usize,
    ) {
        let node_cmds = &access_index.cmds_by_node[node_idx];
        let end_prefix = node_cmds.partition_point(|&cmd_idx| cmd_idx < end_cmd_idx);
        let start_prefix = scratch.covered_node_prefixes[node_idx];

        if end_prefix <= start_prefix {
            return;
        }

        scratch.covered_node_prefixes[node_idx] = end_prefix;

        // Selecting any user of a resource requires the complete preceding resource prefix.
        for &cmd_idx in &node_cmds[start_prefix..end_prefix] {
            if !scratch.selected_cmds.put(cmd_idx) {
                schedule.push(cmd_idx);
                scratch.pending_cmds.push(cmd_idx);
            }
        }
    }

    fn schedule_resource_set_prefix(
        access_index: &CommandAccessIndex,
        schedule: &mut Vec<usize>,
        scratch: &mut ScheduleScratch,
        resource_set_idx: ResourceSetIndex,
        end_cmd_idx: usize,
    ) {
        let index = resource_set_idx.as_usize();
        let resource_set_cmds = &access_index.cmds_by_resource_set[index];
        let end_prefix = resource_set_cmds.partition_point(|&cmd_idx| cmd_idx < end_cmd_idx);
        let start_prefix = scratch.covered_resource_set_prefixes[index];

        if end_prefix <= start_prefix {
            return;
        }

        scratch.covered_resource_set_prefixes[index] = end_prefix;

        for &cmd_idx in &resource_set_cmds[start_prefix..end_prefix] {
            if !scratch.selected_cmds.put(cmd_idx) {
                schedule.push(cmd_idx);
                scratch.pending_cmds.push(cmd_idx);
            }
        }
    }
}

/// Semaphore information used during `queue_submit2` submission.
///
/// Requires Vulkan 1.3 core or the `VK_KHR_synchronization2` extension. Using a non-zero
/// [`value`](Self::value) additionally requires the [`timeline_semaphore`] feature.
///
/// See [`VkSemaphoreSubmitInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSemaphoreSubmitInfo.html).
///
/// [`timeline_semaphore`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceTimelineSemaphoreFeatures.html
#[derive(Clone, Copy, Debug, Default)]
pub struct SemaphoreSubmit2Info {
    /// Semaphore to wait on or signal.
    ///
    /// Defaults to [`vk::Semaphore::null`].
    pub semaphore: vk::Semaphore,

    /// Stages blocked by this wait, or stages after which the semaphore is signaled.
    ///
    /// Defaults to [`vk::PipelineStageFlags2::empty`].
    pub stage_mask: vk::PipelineStageFlags2,

    /// Timeline value to wait for or signal, or `0` for binary semaphores.
    pub value: u64,

    /// Device index for device-group submissions.
    pub device_index: u32,
}

impl SemaphoreSubmit2Info {
    fn check_args(
        device: &Device,
        waits: &[SemaphoreSubmit2Info],
        signals: &[SemaphoreSubmit2Info],
    ) -> Result<(), DriverError> {
        if !device.physical.vk_khr_synchronization2 {
            return Err(DriverError::Unsupported);
        }

        if (waits.iter().any(|wait| wait.value != 0)
            || signals.iter().any(|signal| signal.value != 0))
            && !SemaphoreSubmit2Info::supports_timeline_semaphores(device)
        {
            return Err(DriverError::Unsupported);
        }

        Ok(())
    }

    fn supports_timeline_semaphores(device: &Device) -> bool {
        device.physical.features_v1_2.timeline_semaphore
    }
}

/// Semaphore information used during submission.
///
/// Used for both waits and signals. The legacy `vkQueueSubmit` path only supports binary
/// semaphores and coarse stage masks: [`value`](Self::value) must be `0`, and
/// [`stage_mask`](Self::stage_mask) must be [`vk::PipelineStageFlags2::ALL_COMMANDS`] or
/// [`vk::PipelineStageFlags2::NONE`]. Use [`SemaphoreSubmit2Info`] with
/// [`QueueSubmitInfo::QueueSubmit2`] when a more precise stage mask is required.
///
/// See [`VkSubmitInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubmitInfo.html).
#[derive(Clone, Copy, Debug, Default)]
pub struct SemaphoreSubmitInfo {
    /// Semaphore to wait on or signal.
    ///
    /// Defaults to [`vk::Semaphore::null`].
    pub semaphore: vk::Semaphore,

    /// Stages blocked by this wait, or stages after which the semaphore is signaled.
    ///
    /// Defaults to [`vk::PipelineStageFlags2::empty`].
    pub stage_mask: vk::PipelineStageFlags2,

    /// Timeline value to wait for or signal, or `0` for binary semaphores.
    pub value: u64,
}

impl SemaphoreSubmitInfo {
    fn check_args(
        waits: &[SemaphoreSubmitInfo],
        signals: &[SemaphoreSubmitInfo],
    ) -> Result<(), DriverError> {
        waits
            .iter()
            .chain(signals.iter())
            .all(SemaphoreSubmitInfo::is_supported_legacy_submit)
            .then_some(())
            .ok_or(DriverError::Unsupported)
    }

    fn is_supported_legacy_submit(&self) -> bool {
        self.value == 0
            && matches!(
                self.stage_mask,
                vk::PipelineStageFlags2::ALL_COMMANDS | vk::PipelineStageFlags2::NONE
            )
    }

    fn stage_mask_legacy(stage_mask: vk::PipelineStageFlags2) -> vk::PipelineStageFlags {
        match stage_mask {
            vk::PipelineStageFlags2::NONE => vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags2::ALL_COMMANDS => vk::PipelineStageFlags::ALL_COMMANDS,
            _ => {
                #[cfg(feature = "checked")]
                panic!("invalid legacy submit wait stage mask: {stage_mask:?}");

                #[cfg(not(feature = "checked"))]
                {
                    vk::PipelineStageFlags::ALL_COMMANDS
                }
            }
        }
    }
}

/// A finalized graph execution plan.
///
/// `Submission` owns the remaining commands of a [`Graph`] after [`Graph::finalize`] has ended the
/// graph-building phase. It supports two execution styles:
///
/// - [`Submission::queue_submit`] for a one-shot submission path.
/// - [`Submission::record`] with a [`RecordSelection`] for explicit command-buffer recording,
///   returning a [`Recording`].
#[derive(Debug)]
pub struct Submission {
    exclusive_buffer_ranges: HashMap<usize, Vec<BufferSubresourceRange>>,
    exclusive_image_ranges: HashMap<usize, Vec<vk::ImageSubresourceRange>>,
    graph: Graph,
    pending_buffer_transfer_nodes:
        Option<PendingTransferNodes<vk::Buffer, BufferQueueOwnershipTransfer>>,
    pending_image_transfer_nodes: Option<PendingTransferNodes<vk::Image, ImageOwnershipTransfer>>,
    pending_image_set_transfers: HashMap<PhysicalImageId, Vec<ImageOwnershipTransfer>>,
    queue_ownership_release_groups: Vec<QueueOwnershipReleaseGroup>,
    query_pool_results: Option<SubmittedTimestampQueries>,
    query_pool_reset: bool,
    recorded_commands: Vec<CommandRecordingResources>,
    submit_retained: Vec<SubmittedCommand>,
    touched_image_sets: FixedBitSet,
}

impl Submission {
    const GRAPHICS_STAGES: vk::PipelineStageFlags = vk::PipelineStageFlags::from_raw(
        vk::PipelineStageFlags::DRAW_INDIRECT.as_raw()
            | vk::PipelineStageFlags::VERTEX_INPUT.as_raw()
            | vk::PipelineStageFlags::VERTEX_SHADER.as_raw()
            | vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER.as_raw()
            | vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER.as_raw()
            | vk::PipelineStageFlags::GEOMETRY_SHADER.as_raw()
            | vk::PipelineStageFlags::FRAGMENT_SHADER.as_raw()
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS.as_raw()
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS.as_raw()
            | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.as_raw()
            | vk::PipelineStageFlags::TASK_SHADER_EXT.as_raw()
            | vk::PipelineStageFlags::MESH_SHADER_EXT.as_raw(),
    );

    pub(super) fn new(graph: Graph) -> Self {
        let recorded_commands = Vec::with_capacity(graph.cmds.len());
        let touched_image_sets = FixedBitSet::with_capacity(graph.resource_sets.len());

        Self {
            exclusive_buffer_ranges: HashMap::new(),
            exclusive_image_ranges: HashMap::new(),
            pending_buffer_transfer_nodes: None,
            graph,
            queue_ownership_release_groups: Vec::new(),
            query_pool_results: None,
            query_pool_reset: false,
            recorded_commands,
            pending_image_transfer_nodes: None,
            pending_image_set_transfers: HashMap::new(),
            submit_retained: Vec::new(),
            touched_image_sets,
        }
    }

    #[profiling::function]
    fn allow_merge_passes(lhs: &CommandData, rhs: &CommandData) -> bool {
        let lhs_pipeline = Submission::first_graphic_pipeline(lhs);
        if lhs_pipeline.is_none() {
            trace!("  {} is not graphics", lhs.name());

            return false;
        }

        let rhs_pipeline = Submission::first_graphic_pipeline(rhs);
        if rhs_pipeline.is_none() {
            trace!("  {} is not graphics", rhs.name());

            return false;
        }

        let lhs_pipeline = unsafe { lhs_pipeline.unwrap_unchecked() };
        let rhs_pipeline = unsafe { rhs_pipeline.unwrap_unchecked() };

        // Must be same general rasterization modes
        let lhs_info = lhs_pipeline.inner.info;
        let rhs_info = rhs_pipeline.inner.info;
        if lhs_info.blend != rhs_info.blend
            || lhs_info.cull_mode != rhs_info.cull_mode
            || lhs_info.front_face != rhs_info.front_face
            || lhs_info.polygon_mode != rhs_info.polygon_mode
            || lhs_info.samples != rhs_info.samples
        {
            trace!("  different rasterization modes",);

            return false;
        }

        let rhs = rhs.execs.first();

        // Commands without executions are filtered before scheduling.
        debug_assert!(rhs.is_some());

        let rhs = unsafe { rhs.unwrap_unchecked() };

        let mut common_color_attachment = false;
        let mut common_depth_attachment = false;

        // Now we need to know what the subpasses (we may have prior merges) wrote
        for lhs in lhs.execs.iter().rev() {
            // Multiview subpasses cannot be combined with non-multiview subpasses
            if Submission::is_multiview(lhs.view_mask) != Submission::is_multiview(rhs.view_mask) {
                trace!("  incompatible multiview");

                return false;
            }

            // Compare individual color attachments for compatibility
            for (attachment_idx, lhs_attachment) in lhs.attachments.color_attachments() {
                let rhs_attachment = rhs
                    .attachments
                    .color_attachment(attachment_idx)
                    .map(|state| state.attachment);

                if !Attachment::are_compatible(Some(lhs_attachment.attachment), rhs_attachment) {
                    trace!("  incompatible color attachments");

                    return false;
                }

                common_color_attachment = true;
            }

            // Compare depth/stencil attachments for compatibility
            let lhs_depth_stencil = lhs
                .attachments
                .depth_stencil_attachment()
                .map(|state| state.attachment);

            let rhs_depth_stencil = rhs
                .attachments
                .depth_stencil_attachment()
                .map(|state| state.attachment);

            if !Attachment::are_compatible(lhs_depth_stencil, rhs_depth_stencil) {
                trace!("  incompatible depth/stencil attachments");

                return false;
            }

            common_depth_attachment |= lhs_depth_stencil.is_some() && rhs_depth_stencil.is_some();
        }

        // Keep color and depth on tile
        if common_color_attachment || common_depth_attachment {
            trace!("  merging due to common image");

            return true;
        }

        // Keep input on tile
        if !rhs_pipeline.inner.input_attachments.is_empty() {
            trace!("  merging due to subpass input");

            return true;
        }

        trace!("  not merging");

        // No reason to merge, so don't
        false
    }

    #[cfg(feature = "checked")]
    pub(crate) fn assert_reusable_commands(&self) {
        for cmd in &self.graph.cmds {
            for exec in &cmd.execs {
                assert!(
                    exec.func
                        .as_ref()
                        .is_some_and(crate::CommandFunction::is_reusable),
                    "command stream contains a one-shot callback"
                );
            }
        }
    }

    fn attachment_layout(
        aspect_mask: vk::ImageAspectFlags,
        is_random_access: bool,
        is_input: bool,
    ) -> vk::ImageLayout {
        if aspect_mask.contains(vk::ImageAspectFlags::COLOR) {
            if is_input {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            }
        } else if aspect_mask.contains(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
        {
            if is_random_access {
                if is_input {
                    vk::ImageLayout::GENERAL
                } else {
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                }
            } else {
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
            }
        } else if aspect_mask.contains(vk::ImageAspectFlags::DEPTH) {
            if is_random_access {
                if is_input {
                    vk::ImageLayout::GENERAL
                } else {
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
                }
            } else {
                vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL
            }
        } else if aspect_mask.contains(vk::ImageAspectFlags::STENCIL) {
            if is_random_access {
                if is_input {
                    vk::ImageLayout::GENERAL
                } else {
                    vk::ImageLayout::STENCIL_ATTACHMENT_OPTIMAL
                }
            } else {
                vk::ImageLayout::STENCIL_READ_ONLY_OPTIMAL
            }
        } else {
            vk::ImageLayout::UNDEFINED
        }
    }

    fn attachment_read_stage(aspect_mask: vk::ImageAspectFlags) -> vk::PipelineStageFlags {
        match aspect_mask {
            mask if mask.contains(vk::ImageAspectFlags::COLOR) => {
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            }
            mask if mask
                .intersects(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL) =>
            {
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
            }
            _ => vk::PipelineStageFlags::ALL_GRAPHICS,
        }
    }

    fn attachment_read_write_access(
        aspect_mask: vk::ImageAspectFlags,
    ) -> (vk::AccessFlags, vk::AccessFlags) {
        match aspect_mask {
            mask if mask.contains(vk::ImageAspectFlags::COLOR) => (
                vk::AccessFlags::COLOR_ATTACHMENT_READ,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ),
            mask if mask
                .intersects(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL) =>
            {
                (
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
            }
            _ => (
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            ),
        }
    }

    fn attachment_stage(aspect_mask: vk::ImageAspectFlags) -> vk::PipelineStageFlags {
        match aspect_mask {
            mask if mask.contains(vk::ImageAspectFlags::COLOR) => {
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            }
            mask if mask
                .intersects(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL) =>
            {
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
            }
            _ => vk::PipelineStageFlags::ALL_GRAPHICS,
        }
    }

    fn attachments_are_exact(lhs: Attachment, rhs: Attachment) -> bool {
        // The command merger's identity check does not include the view aspect.
        Attachment::are_identical(lhs, rhs) && lhs.aspect_mask == rhs.aspect_mask
    }

    fn barriers_require_sync2(
        global_barrier: Option<&GlobalBarrier<'_>>,
        micromap_barrier: Option<&GlobalBarrier<'_>>,
        buffer_barriers: &[BufferBarrier<'_>],
        image_barriers: &[TrackedImageBarrier],
    ) -> bool {
        micromap_barrier.is_some()
            || global_barrier.is_some_and(|barrier| {
                barrier
                    .previous_accesses
                    .iter()
                    .chain(barrier.next_accesses)
                    .copied()
                    .any(PipelineStageAccessFlags::is_micromap_access)
            })
            || buffer_barriers.iter().any(|barrier| {
                barrier
                    .previous_accesses
                    .iter()
                    .chain(barrier.next_accesses)
                    .copied()
                    .any(PipelineStageAccessFlags::is_micromap_access)
            })
            || image_barriers.iter().any(|barrier| {
                PipelineStageAccessFlags::is_micromap_access(barrier.next_access)
                    || barrier
                        .previous_accesses
                        .iter()
                        .any(PipelineStageAccessFlags::is_micromap_access)
            })
    }

    #[profiling::function]
    fn begin_render_pass(
        cmd_buf: &CommandBuffer,
        bindings: &[AnyResource],
        pass: &CommandData,
        recorded_command: &mut CommandRecordingResources,
        render_area: vk::Rect2D,
    ) -> Result<(), DriverError> {
        trace!("  begin render pass");

        let CommandRecordingResources {
            exec_subpasses,
            render_pass,
            ..
        } = recorded_command;
        let render_pass = render_pass.as_mut().expect("missing render pass");
        let attachment_count = render_pass.info.attachments.len();

        let mut attachments = Vec::with_capacity(attachment_count);
        attachments.resize(
            attachment_count,
            FramebufferAttachmentImageInfo {
                flags: vk::ImageCreateFlags::empty(),
                usage: vk::ImageUsageFlags::empty(),
                width: 0,
                height: 0,
                layer_count: 0,
                view_formats: vec![],
            },
        );

        thread_local! {
            static CLEARS_VIEWS: RefCell<(
                Vec<vk::ClearValue>,
                Vec<vk::ImageView>,
            )> = Default::default();
        }

        CLEARS_VIEWS.with_borrow_mut(|(clear_values, image_views)| {
            clear_values.resize_with(attachment_count, vk::ClearValue::default);
            image_views.resize(attachment_count, vk::ImageView::null());

            for (exec, &subpass_idx) in pass.execs.iter().zip(exec_subpasses.iter()) {
                let subpass = &render_pass.info.subpasses[subpass_idx as usize];
                for (attachment_idx, state) in exec.attachments.color_attachments() {
                    let attachment = state.attachment;
                    let attachment_image = &mut attachments[attachment_idx as usize];
                    if let Err(idx) = attachment_image
                        .view_formats
                        .binary_search(&attachment.format)
                    {
                        if let LoadOp::Clear(clear_value) = state.load {
                            clear_values[attachment_idx as usize] = vk::ClearValue {
                                color: vk::ClearColorValue {
                                    float32: clear_value,
                                },
                            };
                        }

                        let image = Self::expect_attachment_image(bindings, &attachment);

                        attachment_image.flags = image.info.flags;
                        attachment_image.usage = image.info.usage;
                        attachment_image.width = image.info.width >> attachment.base_mip_level;
                        attachment_image.height = image.info.height >> attachment.base_mip_level;
                        attachment_image.layer_count = attachment.array_layer_count;
                        attachment_image.view_formats.insert(idx, attachment.format);

                        image_views[attachment_idx as usize] =
                            Image::view(image, attachment.image_view_info(image.info))?;
                    }
                }

                if let Some(state) = exec.attachments.depth_stencil_attachment()
                    && state.is_attachment
                {
                    let attachment = state.attachment;
                    let attachment_idx = subpass
                        .depth_stencil_attachment
                        .expect("missing depth stencil attachment reference")
                        .attachment as usize;
                    let attachment_image = &mut attachments[attachment_idx];
                    if let Err(idx) = attachment_image
                        .view_formats
                        .binary_search(&attachment.format)
                    {
                        if let LoadOp::Clear(depth_stencil) = state.load {
                            clear_values[attachment_idx] = vk::ClearValue { depth_stencil };
                        }

                        let image = Self::expect_attachment_image(bindings, &attachment);

                        attachment_image.flags = image.info.flags;
                        attachment_image.usage = image.info.usage;
                        attachment_image.width = image.info.width >> attachment.base_mip_level;
                        attachment_image.height = image.info.height >> attachment.base_mip_level;
                        attachment_image.layer_count = attachment.array_layer_count;
                        attachment_image.view_formats.insert(idx, attachment.format);

                        image_views[attachment_idx] =
                            Image::view(image, attachment.image_view_info(image.info))?;
                    }
                }

                if let Some(state) = exec
                    .attachments
                    .depth_stencil_attachment()
                    .and_then(|state| state.resolve)
                {
                    let attachment_idx = subpass
                        .depth_stencil_resolve_attachment
                        .expect("missing depth stencil resolve attachment reference")
                        .0
                        .attachment as usize;
                    let attachment_image = &mut attachments[attachment_idx];
                    if let Err(idx) = attachment_image
                        .view_formats
                        .binary_search(&state.attachment.format)
                    {
                        let image = Self::expect_attachment_image(bindings, &state.attachment);

                        attachment_image.flags = image.info.flags;
                        attachment_image.usage = image.info.usage;
                        attachment_image.width =
                            image.info.width >> state.attachment.base_mip_level;
                        attachment_image.height =
                            image.info.height >> state.attachment.base_mip_level;
                        attachment_image.layer_count = state.attachment.array_layer_count;
                        attachment_image
                            .view_formats
                            .insert(idx, state.attachment.format);

                        image_views[attachment_idx] =
                            Image::view(image, state.attachment.image_view_info(image.info))?;
                    }
                }
            }

            let framebuffer =
                RenderPass::framebuffer(render_pass, FramebufferInfo { attachments })?;

            unsafe {
                cmd_buf.device.cmd_begin_render_pass(
                    cmd_buf.handle,
                    &vk::RenderPassBeginInfo::default()
                        .render_pass(render_pass.handle)
                        .framebuffer(framebuffer)
                        .render_area(render_area)
                        .clear_values(clear_values)
                        .push_next(
                            &mut vk::RenderPassAttachmentBeginInfoKHR::default()
                                .attachments(image_views),
                        ),
                    vk::SubpassContents::INLINE,
                );
            }

            Ok(())
        })
    }

    #[profiling::function]
    fn bind_descriptor_sets(
        cmd_buf: &CommandBuffer,
        pipeline: &ExecutionPipeline,
        recorded_command: &CommandRecordingResources,
        exec_idx: usize,
    ) {
        if let Some(exec_descriptor_sets) = recorded_command.descriptor_sets.get(exec_idx) {
            thread_local! {
                static DESCRIPTOR_SETS: RefCell<Vec<vk::DescriptorSet>> = Default::default();
            }

            if exec_descriptor_sets.is_empty() {
                return;
            }

            DESCRIPTOR_SETS.with_borrow_mut(|descriptor_sets| {
                descriptor_sets.clear();
                descriptor_sets.extend(
                    exec_descriptor_sets
                        .iter()
                        .map(RecordingDescriptorSet::handle),
                );

                trace!("    bind descriptor sets {:?}", descriptor_sets);

                unsafe {
                    cmd_buf.device.cmd_bind_descriptor_sets(
                        cmd_buf.handle,
                        pipeline.bind_point(),
                        pipeline.layout(),
                        0,
                        descriptor_sets,
                        &[],
                    );
                }
            });
        }
    }

    #[profiling::function]
    fn bind_pipeline(
        cmd_buf: &CommandBuffer,
        recorded_command: &mut CommandRecordingResources,
        exec_idx: usize,
        pipeline: &mut ExecutionPipeline,
        depth_stencil: Option<DepthStencilInfo>,
    ) -> Result<(), DriverError> {
        if log_enabled!(Trace) {
            let (pipeline_kind, name, vk_pipeline) = match pipeline {
                ExecutionPipeline::Compute(pipeline) => (
                    "compute",
                    Device::private_data_object_name(
                        pipeline.device(),
                        vk::ObjectType::PIPELINE,
                        pipeline.handle(),
                    ),
                    pipeline.handle(),
                ),
                ExecutionPipeline::Graphics(pipeline) => (
                    "graphics",
                    Device::private_data_object_name(
                        pipeline.device(),
                        vk::ObjectType::PIPELINE_LAYOUT,
                        pipeline.inner.layout,
                    ),
                    vk::Pipeline::null(),
                ),
                ExecutionPipeline::RayTracing(pipeline) => (
                    "ray tracing",
                    Device::private_data_object_name(
                        pipeline.device(),
                        vk::ObjectType::PIPELINE,
                        pipeline.handle(),
                    ),
                    pipeline.handle(),
                ),
            };
            if let Some(name) = name {
                trace!("    bind {pipeline_kind} pipeline {name} ({vk_pipeline:?})");
            } else {
                trace!("    bind {pipeline_kind} pipeline {vk_pipeline:?}");
            }
        }

        // We store a shared reference to this pipeline inside the command buffer!
        let bind_point = pipeline.bind_point();
        let pipeline = match pipeline {
            ExecutionPipeline::Compute(pipeline) => pipeline.handle(),
            ExecutionPipeline::Graphics(pipeline) => {
                let subpass_idx = recorded_command.exec_subpasses[exec_idx];
                RenderPass::pipeline_handle(
                    recorded_command.expect_render_pass_mut(),
                    pipeline,
                    depth_stencil,
                    subpass_idx,
                )?
            }
            ExecutionPipeline::RayTracing(pipeline) => pipeline.handle(),
        };

        unsafe {
            cmd_buf
                .device
                .cmd_bind_pipeline(cmd_buf.handle, bind_point, pipeline);
        }

        Ok(())
    }

    fn buffer_memory_barrier<'a>(
        barrier: &BufferBarrier<'a>,
        queue_flags: vk::QueueFlags,
    ) -> (
        vk::PipelineStageFlags,
        vk::PipelineStageFlags,
        vk::BufferMemoryBarrier<'a>,
    ) {
        let (mut src, mut dst, mut memory) = vk_sync::get_buffer_memory_barrier(barrier);
        if barrier.src_queue_family_index != barrier.dst_queue_family_index {
            // This recorder emits acquires; releases are recorded on the owning queue separately.
            // The ignored source scope must still be legal on the destination queue.
            src = vk::PipelineStageFlags::TOP_OF_PIPE;
            memory.src_access_mask = vk::AccessFlags::empty();
            dst = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
            memory.dst_access_mask = vk::AccessFlags::empty();
            for &access in barrier.next_accesses {
                let (stages, accesses) = pipeline_stage_access_flags(access);
                dst |= stages;
                memory.dst_access_mask |= accesses;
            }
        } else if barrier.previous_accesses.iter().any(|&access| {
            PipelineStageAccessFlags::buffer_source_scope(access, queue_flags)
                != pipeline_stage_access_flags(access)
        }) {
            src = vk::PipelineStageFlags::empty();
            memory.src_access_mask = vk::AccessFlags::empty();
            for &access in barrier.previous_accesses {
                let source = PipelineStageAccessFlags::buffer_source_scope(access, queue_flags);
                if source == pipeline_stage_access_flags(access) {
                    // Keep vk-sync's exact scope for supported producers, including serialization.
                    let (stages, _, barrier) = get_memory_barrier(&GlobalBarrier {
                        previous_accesses: slice::from_ref(&access),
                        next_accesses: &[],
                    });
                    src |= stages;
                    memory.src_access_mask |= barrier.src_access_mask;
                } else {
                    src |= source.0;
                    if is_write_access(access) {
                        memory.src_access_mask |= source.1;
                    }
                }
            }
            if src.is_empty() {
                src = vk::PipelineStageFlags::TOP_OF_PIPE;
            }
        }
        (src, dst, memory)
    }

    fn buffer_memory_barrier2(
        barrier: &BufferBarrier<'_>,
        queue_flags: vk::QueueFlags,
    ) -> vk::BufferMemoryBarrier2<'static> {
        let mut sync = Submission::memory_barrier2(GlobalBarrier {
            previous_accesses: &[],
            next_accesses: barrier.next_accesses,
        });
        let acquire = barrier.src_queue_family_index != barrier.dst_queue_family_index;
        if !acquire {
            for &access in barrier.previous_accesses {
                let source = PipelineStageAccessFlags::buffer_source_scope(access, queue_flags);
                let (stages, accesses) = if source != pipeline_stage_access_flags(access)
                    || (PipelineStageAccessFlags::is_micromap_access(access)
                        && !queue_flags.contains(vk::QueueFlags::COMPUTE))
                {
                    (
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        if is_write_access(access) {
                            vk::AccessFlags2::MEMORY_WRITE
                        } else {
                            vk::AccessFlags2::empty()
                        },
                    )
                } else {
                    // Preserve synchronization2-only stages for producers supported by this queue.
                    micromap_sync_flags_for_access(access)
                };
                sync.src_stage_mask |= stages;
                sync.src_access_mask |= accesses;
            }
        }
        vk::BufferMemoryBarrier2::default()
            .src_stage_mask(if acquire {
                vk::PipelineStageFlags2::NONE
            } else {
                sync.src_stage_mask
            })
            .src_access_mask(if acquire {
                vk::AccessFlags2::NONE
            } else {
                sync.src_access_mask
            })
            .dst_stage_mask(sync.dst_stage_mask)
            .dst_access_mask(sync.dst_access_mask)
            .src_queue_family_index(barrier.src_queue_family_index)
            .dst_queue_family_index(barrier.dst_queue_family_index)
            .buffer(barrier.buffer)
            .offset(barrier.offset as _)
            .size(barrier.size as _)
    }

    fn build_general_subpass_dependencies(
        pass: &CommandData,
        external_access_history: &[PipelineStageAccessFlags],
        exec_subpasses: &[u32],
    ) -> Vec<SubpassDependency> {
        // Planning only reads finalized metadata; no pool or user callbacks can reenter this TLS.
        // RefCell releases the borrow on unwind, and reset discards any partial plan on reuse.
        SUBPASS_DEPENDENCY.with_borrow_mut(|scratch| {
            let subpass_count = exec_subpasses.last().map_or(0, |&idx| idx as usize + 1);
            scratch.reset(external_access_history.len(), subpass_count);
            for exec in &pass.execs {
                for attachment in exec
                    .attachments
                    .color_attachments()
                    .flat_map(|(_, state)| {
                        // All color states are visited, including input-only and resolve sources.
                        std::iter::once(&state.attachment)
                            .chain(state.resolve.as_ref().map(|resolve| &resolve.attachment))
                    })
                    .chain(
                        exec.attachments
                            .depth_stencil_attachment()
                            .into_iter()
                            .flat_map(|state| {
                                std::iter::once(&state.attachment).chain(
                                    state.resolve.as_ref().map(|resolve| &resolve.attachment),
                                )
                            }),
                    )
                {
                    scratch.attachment_nodes.insert(attachment.target);
                }
            }

            for (exec, &subpass_idx) in pass.execs.iter().zip(exec_subpasses) {
                let subpass_idx = subpass_idx as usize;
                if scratch.offsets.len() == subpass_idx {
                    scratch.offsets.push(scratch.groups.len());
                }
                let group_start = scratch.offsets[subpass_idx];
                Self::visit_subpass_scopes(exec, |node_idx, scope, origin| {
                    // Ranges and attachment roles can differ even for the same node/access type.
                    let class = AccessClass::new(exec, node_idx, origin, &scratch.attachment_nodes);
                    let scope = if scope.stage_flags.is_empty() {
                        PipelineStageAccessFlags::default()
                    } else {
                        scope
                    };
                    let buffer_writer = match origin {
                        SubpassAccessOrigin::Explicit(access)
                            if scope.access_flags == vk::AccessFlags::SHADER_WRITE
                                && !scope.stage_flags.is_empty()
                                && Self::GRAPHICS_STAGES.contains(scope.stage_flags) =>
                        {
                            match access.subresource {
                                SubresourceRange::Buffer(range) if range.start < range.end => {
                                    Some((access.access, range))
                                }
                                _ => None,
                            }
                        }
                        _ => None,
                    };
                    let image_reader = match origin {
                        SubpassAccessOrigin::Explicit(access)
                            if PipelineStageAccessFlags::is_read_only_graphics_access(
                                access.access,
                            ) && access_type_to_layout(access.access)
                                == Some(vk::ImageLayout::GENERAL)
                                && scope.access_flags == vk::AccessFlags::SHADER_READ =>
                        {
                            match access.subresource {
                                SubresourceRange::Image(range) => Some((access.access, range)),
                                _ => None,
                            }
                        }
                        _ => None,
                    };
                    let pass_class = &mut scratch.pass_classes[node_idx];
                    if pass_class.is_none() {
                        scratch.touched_nodes.push(node_idx);
                    }
                    *pass_class = Some(pass_class.map_or(class, |old| old.merge(class)));
                    let group_idx = scratch.group_lookup[node_idx];
                    if group_idx != usize::MAX && group_idx >= group_start {
                        let current = &mut scratch.groups[group_idx].2;
                        current.masks.union(scope);
                        current.class = current.class.merge(class);
                        if current.buffer_writer != buffer_writer {
                            current.buffer_writer = None;
                        }
                        if !current.image_reader.zip(image_reader).is_some_and(
                            |((lhs_access, lhs), (rhs_access, rhs))| {
                                lhs_access == rhs_access
                                    && ImageOwnershipTransfer::ranges_equal(lhs, rhs)
                            },
                        ) {
                            current.image_reader = None;
                        }
                    } else {
                        scratch.group_lookup[node_idx] = scratch.groups.len();
                        scratch.groups.push((
                            node_idx,
                            subpass_idx,
                            SubpassAccess {
                                masks: scope,
                                class,
                                buffer_writer,
                                image_reader,
                            },
                        ));
                    }
                });
                // Even attachment metadata with no implicit scope vetoes pass-wide image pruning.
                for attachment in exec
                    .attachments
                    .color_attachments()
                    .flat_map(|(_, state)| {
                        std::iter::once(&state.attachment)
                            .chain(state.resolve.as_ref().map(|resolve| &resolve.attachment))
                    })
                    .chain(
                        exec.attachments
                            .depth_stencil_attachment()
                            .into_iter()
                            .flat_map(|state| {
                                std::iter::once(&state.attachment).chain(
                                    state.resolve.as_ref().map(|resolve| &resolve.attachment),
                                )
                            }),
                    )
                {
                    if scratch.pass_classes[attachment.target].is_none() {
                        scratch.touched_nodes.push(attachment.target);
                    }
                    scratch.pass_classes[attachment.target] = Some(AccessClass::Other);
                    let group_idx = scratch.group_lookup[attachment.target];
                    if group_idx != usize::MAX && group_idx >= group_start {
                        let access = &mut scratch.groups[group_idx].2;
                        access.buffer_writer = None;
                        access.image_reader = None;
                    }
                }
            }
            scratch.offsets.push(scratch.groups.len());

            let mut dependencies = Vec::with_capacity(subpass_count);
            for subpass_idx in 0..subpass_count {
                for group_idx in scratch.offsets[subpass_idx]..scratch.offsets[subpass_idx + 1] {
                    let (node_idx, _, current) = scratch.groups[group_idx];
                    if current.masks.stage_flags.is_empty() {
                        continue;
                    }
                    // Node-wide history is not range-complete. Retain the broad external fallback
                    // for untracked producers, including writers preceding an intermediate read.
                    let previous = external_access_history[node_idx];
                    let external = PipelineStageAccessFlags {
                        // ALL_COMMANDS means ALL_GRAPHICS in a subpass dependency. Preserve explicit
                        // non-graphics producers, but canonicalize redundant graphics/access bits.
                        stage_flags: vk::PipelineStageFlags::ALL_COMMANDS
                            | (previous.stage_flags & !Self::GRAPHICS_STAGES),
                        access_flags: vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                    };
                    scratch.record_dependency(
                        vk::SUBPASS_EXTERNAL as usize,
                        subpass_idx,
                        external,
                        current.masks,
                        vk::DependencyFlags::empty(),
                    );

                    // No internal edges are needed when every use has a read-only proof.
                    if matches!(
                        scratch.pass_classes[node_idx],
                        Some(AccessClass::ReadOnlyBuffer | AccessClass::ReadOnlyImage(_))
                    ) {
                        continue;
                    }

                    // Keep earlier writers, not just the most recent reader or matching stage.
                    // These are physical summaries: later logical readers already contribute here.
                    let buffer_reader = matches!(current.class, AccessClass::ReadOnlyBuffer);
                    let history_len = scratch.history[node_idx].len();
                    let reader_len = if buffer_reader {
                        0
                    } else {
                        scratch.buffer_reader_history[node_idx].len()
                    };

                    // Readers skip the entire reader history; writers still depend on both lists.
                    for history_idx in 0..history_len + reader_len {
                        let previous_group = if history_idx < history_len {
                            scratch.history[node_idx][history_idx]
                        } else {
                            scratch.buffer_reader_history[node_idx][history_idx - history_len]
                        };
                        let (_, previous_idx, previous) = scratch.groups[previous_group];
                        let flags = match (previous.class, current.class) {
                            (
                                AccessClass::AttachmentLocal(lhs),
                                AccessClass::AttachmentLocal(rhs),
                            ) if Submission::attachments_are_exact(lhs, rhs) => {
                                vk::DependencyFlags::BY_REGION
                            }
                            _ => vk::DependencyFlags::empty(),
                        };
                        scratch.record_dependency(
                            previous_idx,
                            subpass_idx,
                            previous.masks,
                            current.masks,
                            flags,
                        );
                    }

                    if buffer_reader {
                        scratch.buffer_reader_history[node_idx].push(group_idx);
                        continue;
                    }

                    let previous_groups = &mut scratch.history[node_idx];
                    if let Some(last) = previous_groups.last_mut()
                        // Separating readers must not retire a writer across an intervening read.
                        && scratch.buffer_reader_history[node_idx]
                            .last()
                            .is_none_or(|reader| reader < last)
                        && ((current.buffer_writer.is_some()
                            && current.buffer_writer == scratch.groups[*last].2.buffer_writer)
                            || current
                                .image_reader
                                .zip(scratch.groups[*last].2.image_reader)
                                .is_some_and(|((lhs_access, lhs), (rhs_access, rhs))| {
                                    lhs_access == rhs_access
                                        && ImageOwnershipTransfer::ranges_equal(lhs, rhs)
                                }))
                    {
                        // The emitted global edge chains identical complete scopes/ranges: buffer
                        // WAW visibility or storage-image read/read execution before a later write.
                        // Never retire a writer through a reader, or through mixed/local scopes.
                        *last = group_idx;
                    } else {
                        previous_groups.push(group_idx);
                    }
                }
                scratch.flush_dependencies(&mut dependencies);
            }

            dependencies.sort_unstable_by_key(|dependency| {
                (dependency.src_subpass, dependency.dst_subpass)
            });
            dependencies
        })
    }

    fn build_render_pass_info(
        pass: &CommandData,
        external_access_history: &[PipelineStageAccessFlags],
        graphics: &[GraphicsExecutionInfo<'_>],
    ) -> (RenderPassInfo, Box<[u32]>) {
        assert_eq!(pass.execs.len(), graphics.len());
        let (mut color_attachment_count, mut depth_stencil_attachment_count) = (0, 0);
        for exec in &pass.execs {
            color_attachment_count = color_attachment_count.max(exec.attachments.color.len());

            let depth_stencil = exec.attachments.depth_stencil_attachment();
            let has_depth_stencil_attachment =
                depth_stencil.is_some_and(|state| state.is_attachment);
            let has_depth_stencil_resolve = depth_stencil.and_then(|state| state.resolve).is_some();

            depth_stencil_attachment_count = depth_stencil_attachment_count
                .max(has_depth_stencil_attachment as usize + has_depth_stencil_resolve as usize);
        }

        let attachment_count = color_attachment_count + depth_stencil_attachment_count;
        let mut attachments = Vec::with_capacity(attachment_count);
        attachments.resize_with(attachment_count, AttachmentInfo::default);

        let mut subpasses = Vec::<SubpassInfo>::with_capacity(pass.execs.len());

        {
            let mut color_set = FixedBitSet::with_capacity(attachment_count);
            color_set.grow(attachment_count);
            let mut depth_stencil_set = false;

            // Add load op attachments using the first executions
            for exec in &pass.execs {
                for (attachment_idx, state) in exec.attachments.color_attachments() {
                    let attachment_idx = attachment_idx as usize;
                    if color_set.put(attachment_idx) {
                        continue;
                    }

                    let attachment = &mut attachments[attachment_idx];
                    attachment.format = state.attachment.format;
                    attachment.sample_count = state.attachment.sample_count;
                    attachment.initial_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                    attachment.load_op = match state.load {
                        LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
                        LoadOp::Load => vk::AttachmentLoadOp::LOAD,
                        LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
                    };
                }

                if !depth_stencil_set {
                    if let Some(state) = exec
                        .attachments
                        .depth_stencil_attachment()
                        .filter(|state| state.is_attachment)
                    {
                        let attachment = &mut attachments[color_attachment_count];
                        attachment.format = state.attachment.format;
                        attachment.sample_count = state.attachment.sample_count;
                        let is_load = matches!(state.load, LoadOp::Load);
                        attachment.initial_layout =
                            if state.attachment.aspect_mask.contains(
                                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                            ) {
                                attachment.load_op = match state.load {
                                    LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
                                    LoadOp::Load => vk::AttachmentLoadOp::LOAD,
                                    LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
                                };
                                attachment.stencil_load_op = match state.load {
                                    LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
                                    LoadOp::Load => vk::AttachmentLoadOp::LOAD,
                                    LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
                                };

                                if is_load {
                                    vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                } else {
                                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                }
                            } else if state
                                .attachment
                                .aspect_mask
                                .contains(vk::ImageAspectFlags::DEPTH)
                            {
                                attachment.load_op = match state.load {
                                    LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
                                    LoadOp::Load => vk::AttachmentLoadOp::LOAD,
                                    LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
                                };

                                if is_load {
                                    vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL
                                } else {
                                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
                                }
                            } else {
                                attachment.stencil_load_op = match state.load {
                                    LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
                                    LoadOp::Load => vk::AttachmentLoadOp::LOAD,
                                    LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
                                };

                                if is_load {
                                    vk::ImageLayout::STENCIL_READ_ONLY_OPTIMAL
                                } else {
                                    vk::ImageLayout::STENCIL_ATTACHMENT_OPTIMAL
                                }
                            };
                        depth_stencil_set = true;
                    } else if exec.attachments.depth_stencil_attachment().is_some() {
                        depth_stencil_set = true;
                    }
                }
            }
        }

        {
            let mut color_set = FixedBitSet::with_capacity(attachment_count);
            color_set.grow(attachment_count);
            let mut depth_stencil_set = false;
            let mut depth_stencil_resolve_set = false;

            // Add store op attachments using the last executions
            for exec in pass.execs.iter().rev() {
                for (attachment_idx, state) in exec.attachments.color_attachments() {
                    let attachment_idx = attachment_idx as usize;
                    if color_set.put(attachment_idx) {
                        continue;
                    }

                    let attachment = &mut attachments[attachment_idx];
                    attachment.format = state.attachment.format;
                    attachment.sample_count = state.attachment.sample_count;
                    attachment.store_op = if state.store == StoreOp::Store {
                        vk::AttachmentStoreOp::STORE
                    } else {
                        vk::AttachmentStoreOp::DONT_CARE
                    };
                    attachment.final_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                }

                if !depth_stencil_set
                    && let Some(state) = exec
                        .attachments
                        .depth_stencil_attachment()
                        .filter(|state| state.is_attachment)
                {
                    let attachment = &mut attachments[color_attachment_count];
                    attachment.format = state.attachment.format;
                    attachment.sample_count = state.attachment.sample_count;
                    attachment.final_layout = if state
                        .attachment
                        .aspect_mask
                        .contains(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                    {
                        attachment.store_op = if state.store == StoreOp::Store {
                            vk::AttachmentStoreOp::STORE
                        } else {
                            vk::AttachmentStoreOp::DONT_CARE
                        };
                        attachment.stencil_store_op = if state.store == StoreOp::Store {
                            vk::AttachmentStoreOp::STORE
                        } else {
                            vk::AttachmentStoreOp::DONT_CARE
                        };

                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                    } else if state
                        .attachment
                        .aspect_mask
                        .contains(vk::ImageAspectFlags::DEPTH)
                    {
                        attachment.store_op = if state.store == StoreOp::Store {
                            vk::AttachmentStoreOp::STORE
                        } else {
                            vk::AttachmentStoreOp::DONT_CARE
                        };

                        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
                    } else {
                        attachment.stencil_store_op = if state.store == StoreOp::Store {
                            vk::AttachmentStoreOp::STORE
                        } else {
                            vk::AttachmentStoreOp::DONT_CARE
                        };

                        vk::ImageLayout::STENCIL_ATTACHMENT_OPTIMAL
                    };
                    depth_stencil_set = true;
                }

                if let Some(state) = exec
                    .attachments
                    .depth_stencil_attachment()
                    .and_then(|state| state.resolve)
                {
                    let attachment = &mut attachments[color_attachment_count + 1];
                    // Resolve APIs have no store option: retain the resolved aspects for later use.
                    if state.depth_mode.is_some()
                        && state
                            .attachment
                            .aspect_mask
                            .contains(vk::ImageAspectFlags::DEPTH)
                    {
                        attachment.store_op = vk::AttachmentStoreOp::STORE;
                    }
                    if state.stencil_mode.is_some()
                        && state
                            .attachment
                            .aspect_mask
                            .contains(vk::ImageAspectFlags::STENCIL)
                    {
                        attachment.stencil_store_op = vk::AttachmentStoreOp::STORE;
                    }
                    if depth_stencil_resolve_set {
                        continue;
                    }
                    attachment.format = state.attachment.format;
                    attachment.sample_count = state.attachment.sample_count;
                    attachment.final_layout = if state
                        .attachment
                        .aspect_mask
                        .contains(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                    {
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                    } else if state
                        .attachment
                        .aspect_mask
                        .contains(vk::ImageAspectFlags::DEPTH)
                    {
                        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
                    } else {
                        vk::ImageLayout::STENCIL_ATTACHMENT_OPTIMAL
                    };
                    depth_stencil_resolve_set = true;
                }
            }
        }

        for attachment in &mut attachments {
            if attachment.load_op == vk::AttachmentLoadOp::DONT_CARE {
                attachment.initial_layout = vk::ImageLayout::UNDEFINED;
            } else if attachment.store_op == vk::AttachmentStoreOp::DONT_CARE
                && attachment.stencil_store_op == vk::AttachmentStoreOp::DONT_CARE
            {
                attachment.final_layout = attachment.initial_layout;
            }
        }

        // Add subpasses
        for (exec, graphics) in pass.execs.iter().zip(graphics) {
            let mut subpass_info = SubpassInfo::with_capacity(attachment_count);

            // Add input attachments
            for attachment_idx in graphics.input_attachments {
                let exec_attachment = exec
                    .attachments
                    .color_attachment(*attachment_idx)
                    .expect("missing input attachment");
                debug_assert!(
                    !matches!(exec_attachment.load, LoadOp::Clear(_)),
                    "cannot clear color attachment {attachment_idx} because it uses subpass input",
                );

                let is_random_access = exec_attachment.store == StoreOp::Store;
                subpass_info.input_attachments.push(AttachmentRef {
                    attachment: *attachment_idx,
                    aspect_mask: exec_attachment.attachment.aspect_mask,
                    layout: Self::attachment_layout(
                        exec_attachment.attachment.aspect_mask,
                        is_random_access,
                        true,
                    ),
                });
            }

            // Set color attachments to defaults
            for attachment_idx in 0..color_attachment_count as u32 {
                let is_input = subpass_info
                    .input_attachments
                    .iter()
                    .any(|input| input.attachment == attachment_idx);
                subpass_info.color_attachments.push(AttachmentRef {
                    attachment: vk::ATTACHMENT_UNUSED,
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    layout: Self::attachment_layout(vk::ImageAspectFlags::COLOR, true, is_input),
                });
            }

            for (attachment_idx, state) in exec.attachments.color_attachments() {
                if state.is_attachment {
                    subpass_info.color_attachments[attachment_idx as usize].attachment =
                        attachment_idx;
                }
            }

            // Set depth/stencil attachment
            if let Some(state) = exec
                .attachments
                .depth_stencil_attachment()
                .filter(|state| state.is_attachment)
            {
                let is_random_access = matches!(state.load, LoadOp::Clear(_))
                    || matches!(state.load, LoadOp::Load)
                    || state.store == StoreOp::Store;
                subpass_info.depth_stencil_attachment = Some(AttachmentRef {
                    attachment: color_attachment_count as u32,
                    aspect_mask: state.attachment.aspect_mask,
                    layout: Self::attachment_layout(
                        state.attachment.aspect_mask,
                        is_random_access,
                        false,
                    ),
                });
            }

            // Set color resolves to defaults
            subpass_info.color_resolve_attachments.extend(repeat_n(
                AttachmentRef {
                    attachment: vk::ATTACHMENT_UNUSED,
                    aspect_mask: vk::ImageAspectFlags::empty(),
                    layout: vk::ImageLayout::UNDEFINED,
                },
                color_attachment_count,
            ));

            // Set any used color resolve attachments now
            for (dst_attachment_idx, state) in exec.attachments.color_attachments() {
                let Some(state) = state.resolve else {
                    continue;
                };

                let is_input = subpass_info
                    .input_attachments
                    .iter()
                    .any(|input| input.attachment == dst_attachment_idx);
                subpass_info.color_resolve_attachments[state.src_attachment_idx as usize] =
                    AttachmentRef {
                        attachment: dst_attachment_idx,
                        aspect_mask: state.attachment.aspect_mask,
                        layout: Self::attachment_layout(
                            state.attachment.aspect_mask,
                            true,
                            is_input,
                        ),
                    };
            }

            if let Some(state) = exec
                .attachments
                .depth_stencil_attachment()
                .and_then(|state| state.resolve)
            {
                // Merging may add color slots ahead of the execution's original depth index.
                trace!(
                    "depth stencil attachment {} maps to {}",
                    state.dst_attachment_idx, color_attachment_count,
                );
                subpass_info.depth_stencil_resolve_attachment = Some((
                    AttachmentRef {
                        attachment: color_attachment_count as u32 + 1,
                        aspect_mask: state.attachment.aspect_mask,
                        layout: Self::attachment_layout(state.attachment.aspect_mask, true, false),
                    },
                    state.depth_mode,
                    state.stencil_mode,
                ))
            }

            subpass_info.view_mask = exec.view_mask;
            subpass_info.correlated_view_mask = exec.correlated_view_mask;

            subpasses.push(subpass_info);
        }

        let exec_subpasses = Self::coalesce_subpasses(pass, graphics, &mut subpasses);
        Self::rebuild_preserve_attachments(&mut subpasses);
        let dependencies =
            Self::build_subpass_dependencies(pass, external_access_history, &exec_subpasses);

        (
            RenderPassInfo {
                attachments,
                dependencies,
                subpasses,
            },
            exec_subpasses,
        )
    }

    fn build_subpass_dependencies(
        pass: &CommandData,
        external_access_history: &[PipelineStageAccessFlags],
        exec_subpasses: &[u32],
    ) -> Vec<SubpassDependency> {
        assert!(Submission::valid_exec_subpasses(
            pass.execs.len(),
            exec_subpasses
        ));
        if exec_subpasses.last() == Some(&0) {
            // All node summaries feed the same external edge; union scopes directly, without
            // node maps or history. Repeated visits are idempotent, including producer stages.
            let mut dependency = SubpassDependency::new(vk::SUBPASS_EXTERNAL, 0);
            for exec in &pass.execs {
                Self::visit_subpass_scopes(exec, |node_idx, scope, _| {
                    if scope.stage_flags.is_empty() {
                        return;
                    }
                    dependency.src_stage_mask |= vk::PipelineStageFlags::ALL_COMMANDS
                        | (external_access_history[node_idx].stage_flags & !Self::GRAPHICS_STAGES);
                    dependency.src_access_mask |=
                        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
                    dependency.dst_stage_mask |= scope.stage_flags;
                    dependency.dst_access_mask |= scope.access_flags;
                });
            }
            return if dependency.dst_stage_mask.is_empty() {
                Vec::new()
            } else {
                vec![dependency]
            };
        }
        Self::build_general_subpass_dependencies(pass, external_access_history, exec_subpasses)
    }

    fn coalesce_subpasses(
        pass: &CommandData,
        graphics: &[GraphicsExecutionInfo<'_>],
        subpasses: &mut Vec<SubpassInfo>,
    ) -> Box<[u32]> {
        assert_eq!(pass.execs.len(), subpasses.len());
        assert_eq!(pass.execs.len(), graphics.len());
        if pass.execs.len() <= 1 {
            return vec![0; pass.execs.len()].into_boxed_slice();
        }
        let mut exec_subpasses = Vec::with_capacity(pass.execs.len());
        let mut previous_eligible = false;
        let mut group_image_layouts = BTreeMap::new();

        for (exec_idx, mut subpass) in take(subpasses).into_iter().enumerate() {
            let exec = &pass.execs[exec_idx];
            subpass.preserve_attachments.clear();
            let mut attachment_nodes = SmallVec::<[(NodeIndex, Attachment); 9]>::new();
            let mut eligible = pass.stream_scope_id.is_none()
                && graphics[exec_idx].input_attachments.is_empty()
                && subpass.input_attachments.is_empty()
                && subpass
                    .color_resolve_attachments
                    .iter()
                    .all(|attachment| attachment.attachment == vk::ATTACHMENT_UNUSED)
                && subpass.depth_stencil_resolve_attachment.is_none();

            for (_, state) in exec.attachments.color_attachments() {
                eligible &= state.is_attachment
                    && !state.is_input
                    && state.resolve.is_none()
                    && state.attachment.aspect_mask == vk::ImageAspectFlags::COLOR;
                eligible &= !attachment_nodes
                    .iter()
                    .any(|&(node, _)| node == state.attachment.target);
                attachment_nodes.push((state.attachment.target, state.attachment));
            }
            if let Some(state) = exec.attachments.depth_stencil_attachment() {
                let aspects = state.attachment.aspect_mask;
                eligible &= state.is_attachment
                    && state.resolve.is_none()
                    && !aspects.is_empty()
                    && (aspects & !(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL))
                        .is_empty();
                eligible &= !attachment_nodes
                    .iter()
                    .any(|&(node, _)| node == state.attachment.target);
                attachment_nodes.push((state.attachment.target, state.attachment));
            }
            eligible &= !attachment_nodes.is_empty();

            // Every execution in the group must independently pass this access whitelist.
            // No dependency-edge or stage-overlap test can prove the absence of a hazard.
            let mut image_layouts = SmallVec::<[_; 8]>::new();
            for (node_idx, accesses) in exec.accesses.iter() {
                let attachment = attachment_nodes
                    .iter()
                    .rev()
                    .find(|&&(node, _)| node == node_idx);
                let mut image_layout = None;
                for access in accesses {
                    if let Some((_, attachment)) = attachment {
                        let ordinary_access = if attachment.aspect_mask
                            == vk::ImageAspectFlags::COLOR
                        {
                            matches!(
                                access.access,
                                AccessType::ColorAttachmentRead
                                    | AccessType::ColorAttachmentWrite
                                    | AccessType::ColorAttachmentReadWrite
                            )
                        } else {
                            matches!(
                                access.access,
                                AccessType::DepthStencilAttachmentRead
                                    | AccessType::DepthStencilAttachmentWrite
                                    | AccessType::DepthStencilAttachmentReadWrite
                            ) || (attachment.aspect_mask == vk::ImageAspectFlags::DEPTH
                                && access.access == AccessType::DepthAttachmentWriteStencilReadOnly)
                                || (attachment.aspect_mask == vk::ImageAspectFlags::STENCIL
                                    && access.access
                                        == AccessType::StencilAttachmentWriteDepthReadOnly)
                        };
                        let attachment_range = vk::ImageSubresourceRange {
                            aspect_mask: attachment.aspect_mask,
                            base_array_layer: attachment.base_array_layer,
                            layer_count: attachment.array_layer_count,
                            base_mip_level: attachment.base_mip_level,
                            level_count: attachment.mip_level_count,
                        };
                        eligible &= ordinary_access
                            && matches!(access.subresource, SubresourceRange::Image(range)
                                if ImageOwnershipTransfer::ranges_equal(range, attachment_range));
                    } else {
                        eligible &=
                            PipelineStageAccessFlags::is_read_only_graphics_access(access.access);
                        if matches!(access.subresource, SubresourceRange::Image(_)) {
                            // Other image reads replace outgoing scopes. Keep an execution
                            // dependency between them so a later writer waits for every reader.
                            eligible &=
                                ImageAccessSet::from_access(access.access).is_sampled_read();
                            if let Some(layout) = access_type_to_layout(access.access) {
                                eligible &= image_layout
                                    .replace(layout)
                                    .is_none_or(|previous| previous == layout);
                            } else {
                                eligible = false;
                            }
                        }
                    }
                }
                if let Some(layout) = image_layout {
                    image_layouts.push((node_idx, layout));
                }
            }
            // One layout per node, not per range. Sorting also handles duplicate remapped nodes.
            image_layouts.sort_unstable_by_key(|&(node, _)| node);
            eligible &= image_layouts
                .windows(2)
                .all(|pair| pair[0].0 != pair[1].0 || pair[0].1 == pair[1].1);
            image_layouts.dedup_by_key(|entry| entry.0);
            // Set members cannot also be accessed as individual nodes (Graph's existing contract).
            eligible &= exec.resource_set_accesses.iter().all(|access| {
                PipelineStageAccessFlags::is_read_only_graphics_access(
                    access.access_type.access_type(),
                )
            });
            eligible &= exec
                .bindings
                .values()
                .all(|(node_idx, _)| !attachment_nodes.iter().any(|(node, _)| node == node_idx));

            let append = eligible && previous_eligible && exec_idx > 0 && {
                let previous = &pass.execs[exec_idx - 1];
                graphics[exec_idx].sample_count == graphics[exec_idx - 1].sample_count
                    && subpasses.last() == Some(&subpass)
                    && exec.attachments.color.len() == previous.attachments.color.len()
                    && exec
                        .attachments
                        .color
                        .iter()
                        .zip(&previous.attachments.color)
                        .all(|(current, previous)| match (current, previous) {
                            (None, None) => true,
                            (Some(current), Some(previous)) => {
                                matches!(current.load, LoadOp::Load)
                                    && Submission::attachments_are_exact(
                                        current.attachment,
                                        previous.attachment,
                                    )
                            }
                            _ => false,
                        })
                    && match (
                        exec.attachments.depth_stencil_attachment(),
                        previous.attachments.depth_stencil_attachment(),
                    ) {
                        (None, None) => true,
                        (Some(current), Some(previous)) => {
                            matches!(current.load, LoadOp::Load)
                                && Submission::attachments_are_exact(
                                    current.attachment,
                                    previous.attachment,
                                )
                        }
                        _ => false,
                    }
                    && image_layouts.iter().all(|(node, layout)| {
                        group_image_layouts
                            .get(node)
                            .is_none_or(|previous| previous == layout)
                    })
            };

            if append {
                group_image_layouts.extend(image_layouts);
            } else {
                subpasses.push(subpass);
                group_image_layouts.clear();
                group_image_layouts.extend(image_layouts);
            }
            previous_eligible = eligible;
            exec_subpasses.push(u32::try_from(subpasses.len() - 1).expect("too many subpasses"));
        }

        debug_assert!(Submission::valid_exec_subpasses(
            pass.execs.len(),
            &exec_subpasses
        ));
        exec_subpasses.into_boxed_slice()
    }

    fn color_attachment_is_read(load: LoadOp<[f32; 4]>) -> bool {
        matches!(load, LoadOp::Load)
    }

    fn depth_stencil_attachment_is_read(load: LoadOp<vk::ClearDepthStencilValue>) -> bool {
        matches!(load, LoadOp::Load)
    }

    fn depth_stencil_attachment_is_write(
        load: LoadOp<vk::ClearDepthStencilValue>,
        store: StoreOp,
        has_resolve: bool,
    ) -> bool {
        matches!(load, LoadOp::Clear(_)) || store == StoreOp::Store || has_resolve
    }

    fn expect_attachment_image<'a>(
        bindings: &'a [AnyResource],
        attachment: &Attachment,
    ) -> &'a Image {
        bindings[attachment.target]
            .as_image()
            .expect("invalid attachment target image")
    }

    fn first_graphic_pipeline(pass: &CommandData) -> Option<&GraphicsPipeline> {
        pass.execs
            .first()
            .and_then(|exec| exec.pipeline.as_ref().map(ExecutionPipeline::as_graphics))
            .flatten()
    }

    fn for_each_first_resource_set_access<'a>(
        accesses: impl IntoIterator<Item = &'a ResourceSetAccess>,
        resource_set_count: usize,
        acquired_resource_sets: &mut FixedBitSet,
        mut visit: impl FnMut(&'a ResourceSetAccess),
    ) {
        for access in accesses {
            if !acquired_resource_sets.put(access.acquisition_index(resource_set_count)) {
                visit(access);
            }
        }
    }

    pub(crate) fn graph(&self) -> &Graph {
        &self.graph
    }

    fn input_attachment_descriptor(
        pass: &CommandData,
        exec_subpasses: &[u32],
        info: &RenderPassInfo,
        exec_idx: usize,
        attachment_idx: u32,
    ) -> (Attachment, vk::ImageLayout) {
        let subpass_idx = exec_subpasses[exec_idx];
        let current_attachment = pass.execs[exec_idx]
            .attachments
            .color_attachment(attachment_idx)
            .expect("missing input attachment target")
            .attachment;
        let attachment = pass.execs[..exec_idx]
            .iter()
            .zip(&exec_subpasses[..exec_idx])
            .rev()
            .filter(|(_, previous_subpass)| **previous_subpass < subpass_idx)
            .find_map(|(exec, _)| {
                exec.attachments
                    .color_attachment(attachment_idx)
                    .map(|state| state.attachment)
                    .filter(|&attachment| {
                        Submission::attachments_are_exact(current_attachment, attachment)
                    })
            })
            .expect("input attachment not written in a prior physical subpass");
        let layout = info.subpasses[subpass_idx as usize]
            .input_attachments
            .iter()
            .find(|input| input.attachment == attachment_idx)
            .expect("missing physical input attachment")
            .layout;

        (attachment, layout)
    }

    fn into_recorded_submission<Cb>(
        self,
        cmd_buf: Cb,
        releases: Vec<QueueOwnershipRelease>,
        waits: Vec<QueueOwnershipReleaseWait>,
    ) -> RecordedSubmission<Cb>
    where
        Cb: AsRef<CommandBuffer>,
    {
        RecordedSubmission {
            cmd_buf,
            queue_ownership_release_waits: waits,
            state: Arc::new(Mutex::new(RecordedSubmissionState {
                _releases: releases,
                executed: false,
                submission: self,
            })),
        }
    }

    /// Returns `true` when this submission contains no more commands to record.
    pub fn is_empty(&self) -> bool {
        self.graph.cmds.is_empty()
    }

    fn is_multiview(view_mask: u32) -> bool {
        view_mask != 0
    }

    #[allow(clippy::type_complexity)]
    #[profiling::function]
    fn lease_descriptor_pool<P>(
        pool: &mut P,
        pass: &CommandData,
    ) -> Result<Option<Lease<DescriptorPool>>, DriverError>
    where
        P: SubmissionPool,
    {
        let max_sets = pass
            .execs
            .iter()
            .filter_map(|exec| {
                exec.pipeline.as_ref().map(|pipeline| {
                    pipeline
                        .descriptor_info()
                        .layouts
                        .keys()
                        .filter(|set| !exec.descriptor_sets.contains_key(set))
                        .count() as u32
                })
            })
            .sum();
        let mut info = DescriptorPoolInfo {
            max_sets,
            ..Default::default()
        };

        // Find the total count of descriptors per type (there may be multiple pipelines!)
        for pool_size in pass.descriptor_pools_sizes() {
            for (&descriptor_ty, &descriptor_count) in pool_size {
                debug_assert_ne!(descriptor_count, 0);

                match descriptor_ty {
                    vk::DescriptorType::ACCELERATION_STRUCTURE_KHR => {
                        info.acceleration_structure_count += descriptor_count;
                    }
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                        info.combined_image_sampler_count += descriptor_count;
                    }
                    vk::DescriptorType::INPUT_ATTACHMENT => {
                        info.input_attachment_count += descriptor_count;
                    }
                    vk::DescriptorType::SAMPLED_IMAGE => {
                        info.sampled_image_count += descriptor_count;
                    }
                    vk::DescriptorType::SAMPLER => {
                        info.sampler_count += descriptor_count;
                    }
                    vk::DescriptorType::STORAGE_BUFFER => {
                        info.storage_buffer_count += descriptor_count;
                    }
                    vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                        info.storage_buffer_dynamic_count += descriptor_count;
                    }
                    vk::DescriptorType::STORAGE_IMAGE => {
                        info.storage_image_count += descriptor_count;
                    }
                    vk::DescriptorType::STORAGE_TEXEL_BUFFER => {
                        info.storage_texel_buffer_count += descriptor_count;
                    }
                    vk::DescriptorType::UNIFORM_BUFFER => {
                        info.uniform_buffer_count += descriptor_count;
                    }
                    vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                        info.uniform_buffer_dynamic_count += descriptor_count;
                    }
                    vk::DescriptorType::UNIFORM_TEXEL_BUFFER => {
                        info.uniform_texel_buffer_count += descriptor_count;
                    }
                    _ => {
                        warn!(
                            "unsupported descriptor type {:?} for command {}",
                            descriptor_ty,
                            pass.name(),
                        );

                        return Err(DriverError::Unsupported);
                    }
                };
            }
        }

        // It's possible to execute a command-only pipeline or use only supplied descriptor sets.
        if info.max_sets == 0 {
            return Ok(None);
        }

        // Trivially round up the descriptor counts to increase cache coherence
        const ATOM: u32 = 1 << 5;
        info.acceleration_structure_count =
            info.acceleration_structure_count.next_multiple_of(ATOM);
        info.combined_image_sampler_count =
            info.combined_image_sampler_count.next_multiple_of(ATOM);
        info.input_attachment_count = info.input_attachment_count.next_multiple_of(ATOM);
        info.sampled_image_count = info.sampled_image_count.next_multiple_of(ATOM);
        info.sampler_count = info.sampler_count.next_multiple_of(ATOM);
        info.storage_buffer_count = info.storage_buffer_count.next_multiple_of(ATOM);
        info.storage_buffer_dynamic_count =
            info.storage_buffer_dynamic_count.next_multiple_of(ATOM);
        info.storage_image_count = info.storage_image_count.next_multiple_of(ATOM);
        info.storage_texel_buffer_count = info.storage_texel_buffer_count.next_multiple_of(ATOM);
        info.uniform_buffer_count = info.uniform_buffer_count.next_multiple_of(ATOM);
        info.uniform_buffer_dynamic_count =
            info.uniform_buffer_dynamic_count.next_multiple_of(ATOM);
        info.uniform_texel_buffer_count = info.uniform_texel_buffer_count.next_multiple_of(ATOM);

        // Rounded descriptor counts make descriptor pools more reusable across similar pipelines

        // debug!("{:#?}", info);

        Ok(Some(pool.descriptor_pool(info)?))
    }

    #[profiling::function]
    fn lease_render_pass<P>(
        &self,
        pool: &mut P,
        pass_idx: usize,
        external_access_history: &[PipelineStageAccessFlags],
    ) -> Result<(Lease<RenderPass>, Box<[u32]>), DriverError>
    where
        P: SubmissionPool,
    {
        let pass = &self.graph.cmds[pass_idx];
        let graphics = pass
            .execs
            .iter()
            .map(|exec| {
                let pipeline = exec
                    .pipeline
                    .as_ref()
                    .expect("missing graphics pipeline")
                    .expect_graphics();
                GraphicsExecutionInfo {
                    input_attachments: &pipeline.inner.input_attachments,
                    sample_count: pipeline.inner.multisample.rasterization_samples,
                }
            })
            .collect::<Vec<_>>();
        let (info, exec_subpasses) =
            Self::build_render_pass_info(pass, external_access_history, &graphics);

        Ok((pool.render_pass(info)?, exec_subpasses))
    }

    #[profiling::function]
    fn lease_scheduled_resources<P>(
        &mut self,
        pool: &mut P,
        schedule: &[usize],
    ) -> Result<(), DriverError>
    where
        P: SubmissionPool,
    {
        PipelineStageAccessFlags::with_external_render_pass_accesses(
            self.graph.resources.len(),
            |external_accesses| {
                for pass_idx in schedule.iter().copied() {
                    // At the time this function runs the pass will already have been optimized into a
                    // larger pass made out of anything that might have been merged into it - so we
                    // only care about one pass at a time here
                    let pass = &self.graph.cmds[pass_idx];

                    trace!("requesting [{pass_idx}: {}]", pass.name());

                    let descriptor_pool = Self::lease_descriptor_pool(pool, pass)?;
                    let mut descriptor_sets = Vec::with_capacity(pass.execs.len());
                    descriptor_sets.resize_with(pass.execs.len(), Vec::new);
                    for (exec_idx, exec) in pass.execs.iter().enumerate() {
                        let Some(pipeline) = exec.pipeline.as_ref() else {
                            continue;
                        };

                        descriptor_sets[exec_idx] = pipeline
                            .descriptor_info()
                            .layouts
                            .iter()
                            .map(|(&set, descriptor_set_layout)| {
                                if let Some(descriptor_set) = exec.descriptor_sets.get(&set) {
                                    Ok(RecordingDescriptorSet::Supplied(descriptor_set.clone()))
                                } else {
                                    let descriptor_pool = descriptor_pool
                                        .as_ref()
                                        .expect("missing automatic descriptor pool");
                                    DescriptorPool::allocate_descriptor_set(
                                        descriptor_pool,
                                        descriptor_set_layout,
                                    )
                                    .map(RecordingDescriptorSet::Automatic)
                                }
                            })
                            .collect::<Result<_, _>>()?;
                    }

                    /*
                    As a side effect of merging compatible passes, all input passes should be attached to
                    their preceding passes by now. This allows subpasses to use input attachments. If a pass
                    still starts with input-only work here, it cannot be represented correctly.
                    */
                    debug_assert!(!pass.execs.is_empty());
                    debug_assert!(
                        pass.expect_first_exec().pipeline.is_none()
                            || !pass
                                .expect_first_exec()
                                .pipeline
                                .as_ref()
                                .is_some_and(|pipeline| pipeline.is_graphics())
                            || pass
                                .expect_first_exec()
                                .pipeline
                                .as_ref()
                                .expect("missing graphics pipeline")
                                .expect_graphics()
                                .inner
                                .descriptor_info
                                .pool_sizes
                                .iter()
                                .filter(|(set, _)| {
                                    !pass.expect_first_exec().descriptor_sets.contains_key(set)
                                })
                                .filter_map(|(_, pool)| {
                                    pool.get(&vk::DescriptorType::INPUT_ATTACHMENT)
                                })
                                .next()
                                .is_none()
                    );

                    // Also, the render pass may be None if the pass contained no graphics operations.
                    let (render_pass, exec_subpasses) = if pass
                        .expect_first_exec()
                        .pipeline
                        .as_ref()
                        .map(|pipeline| pipeline.is_graphics())
                        .unwrap_or_default()
                    {
                        let (render_pass, exec_subpasses) =
                            self.lease_render_pass(pool, pass_idx, external_accesses)?;
                        (Some(render_pass), exec_subpasses)
                    } else {
                        (None, Box::default())
                    };

                    PipelineStageAccessFlags::record_external_accesses(external_accesses, pass);

                    self.recorded_commands.push(CommandRecordingResources {
                        descriptor_pool,
                        descriptor_sets,
                        exec_subpasses,
                        render_pass,
                    });
                }

                Ok(())
            },
        )
    }

    fn memory_barrier2(barrier: GlobalBarrier<'_>) -> vk::MemoryBarrier2<'static> {
        let (mut src_stage_mask, mut src_access_mask) =
            (vk::PipelineStageFlags2::empty(), vk::AccessFlags2::empty());
        let (mut dst_stage_mask, mut dst_access_mask) =
            (vk::PipelineStageFlags2::empty(), vk::AccessFlags2::empty());

        for access in barrier.previous_accesses {
            let (stages, accesses) = micromap_sync_flags_for_access(*access);
            src_stage_mask |= stages;
            src_access_mask |= accesses;
        }

        for access in barrier.next_accesses {
            let (stages, accesses) = micromap_sync_flags_for_access(*access);
            dst_stage_mask |= stages;
            dst_access_mask |= accesses;
        }

        vk::MemoryBarrier2::default()
            .src_stage_mask(src_stage_mask)
            .src_access_mask(src_access_mask)
            .dst_stage_mask(dst_stage_mask)
            .dst_access_mask(dst_access_mask)
    }

    // Merge contiguous scheduled graphics commands with compatible attachments. Scheduled command
    // order is final during this function.
    #[profiling::function]
    fn merge_scheduled_cmds(&mut self, schedule: &mut Vec<usize>) {
        thread_local! {
            static CMD_SLOTS: RefCell<Vec<Option<CommandData>>> = Default::default();
        }

        CMD_SLOTS.with_borrow_mut(|cmds| {
            debug_assert!(cmds.is_empty());

            let old_cmd_len = self.graph.cmds.len();
            let mut old_to_new_cmd = vec![(0, 0); old_cmd_len + 1];
            cmds.extend(self.graph.cmds.drain(..).map(Some));

            let mut schedule_idx = 0;

            // debug!("attempting to merge {} passes", schedule.len(),);

            while schedule_idx < schedule.len() {
                let first_cmd_idx = schedule[schedule_idx];
                let mut cmd = cmds[schedule[schedule_idx]]
                    .take()
                    .expect("missing scheduled cmd");
                let new_cmd_idx = self.graph.cmds.len();
                old_to_new_cmd[first_cmd_idx] = (new_cmd_idx, 0);

                // Find candidates
                let merge_start = schedule_idx + 1;
                let mut merge_end = merge_start;
                while merge_end < schedule.len() {
                    let other = cmds[schedule[merge_end]]
                        .as_ref()
                        .expect("missing scheduled cmd");

                    debug!(
                        "attempting to merge [{schedule_idx}: {}] with [{merge_end}: {}]",
                        cmd.name(),
                        other.name()
                    );

                    if Self::allow_merge_passes(&cmd, other) {
                        merge_end += 1;
                    } else {
                        break;
                    }
                }

                if log_enabled!(Trace) && merge_start != merge_end {
                    trace!(
                        "merging {} passes into [{schedule_idx}: {}]",
                        merge_end - merge_start,
                        cmd.name()
                    );
                }

                let mut name = cmd.name().to_owned();

                // Grow the merged cmd once, not per merge
                {
                    let mut additional_name_len = 0;
                    let mut additional_exec_count = 0;
                    for merge_idx in merge_start..merge_end {
                        let other = cmds[schedule[merge_idx]]
                            .as_ref()
                            .expect("missing scheduled cmd");
                        additional_name_len += other.name().len() + 3;
                        additional_exec_count += other.execs.len();
                    }

                    name.reserve(additional_name_len);
                    cmd.execs.reserve(additional_exec_count);
                }

                let mut exec_offset = cmd.execs.len();
                for merge_idx in merge_start..merge_end {
                    let old_cmd_idx = schedule[merge_idx];
                    let mut other = cmds[schedule[merge_idx]]
                        .take()
                        .expect("missing scheduled cmd");
                    old_to_new_cmd[old_cmd_idx] = (new_cmd_idx, exec_offset);
                    exec_offset += other.execs.len();
                    name.push_str(" + ");
                    name.push_str(other.name());
                    cmd.tracking.extend(take(&mut other.tracking));
                    cmd.execs.append(&mut other.execs);
                }

                #[cfg(debug_assertions)]
                {
                    cmd.name = Some(name);
                }

                self.graph.cmds.push(cmd);
                schedule_idx += 1 + merge_end - merge_start;
            }

            // Reschedule cmds
            schedule.truncate(self.graph.cmds.len());

            for (idx, cmd_idx) in schedule.iter_mut().enumerate() {
                *cmd_idx = idx;
            }

            // Add the remaining cmds back into the graph for later
            for (old_cmd_idx, cmd) in cmds.drain(..).enumerate() {
                let Some(cmd) = cmd else {
                    continue;
                };

                old_to_new_cmd[old_cmd_idx] = (self.graph.cmds.len(), 0);
                self.graph.cmds.push(cmd);
            }
            old_to_new_cmd[old_cmd_len] = (self.graph.cmds.len(), 0);

            if let Some(timestamp_queries) = &mut self.graph.timestamp_queries {
                for query in timestamp_queries.iter_mut().flatten() {
                    let (command_idx, exec_idx) = old_to_new_cmd[query.command_idx];
                    query.command_idx = command_idx;
                    query.exec_idx += exec_idx;
                }
            }
        });
    }

    fn next_subpass(cmd: &CommandBuffer) {
        trace!("next_subpass");

        unsafe {
            cmd.device
                .cmd_next_subpass(cmd.handle, vk::SubpassContents::INLINE);
        }
    }

    // Avoid vk-sync's barrier-recording allocations; see its implementation for reference.
    fn pipeline_barrier_from_slices<'a>(
        cmd_buf: &CommandBuffer,
        global_barrier: Option<GlobalBarrier<'a>>,
        micromap_barrier: Option<GlobalBarrier<'a>>,
        buffer_barriers: &[BufferBarrier<'a>],
        image_barriers: &[TrackedImageBarrier],
    ) {
        let device = &cmd_buf.device;
        let command_buffer = cmd_buf.handle;
        let queue_flags =
            device.physical.queue_families[cmd_buf.info.queue_family_index as usize].queue_flags;
        let requires_sync2 = Submission::barriers_require_sync2(
            global_barrier.as_ref(),
            micromap_barrier.as_ref(),
            buffer_barriers,
            image_barriers,
        );

        if requires_sync2 {
            thread_local! {
                static BARRIER: RefCell<BarrierScratch<
                    vk::MemoryBarrier2<'static>,
                    vk::BufferMemoryBarrier2<'static>,
                    vk::ImageMemoryBarrier2<'static>,
                >> = Default::default();
            }

            BARRIER.with_borrow_mut(|tls| {
                tls.memory_barriers.clear();
                tls.buffer_barriers.clear();
                tls.image_barriers.clear();

                tls.memory_barriers.extend(
                    global_barrier
                        .into_iter()
                        .chain(micromap_barrier)
                        .map(Submission::memory_barrier2),
                );
                tls.buffer_barriers.extend(
                    buffer_barriers
                        .iter()
                        .map(|barrier| Submission::buffer_memory_barrier2(barrier, queue_flags)),
                );
                tls.image_barriers
                    .extend(image_barriers.iter().copied().map(|barrier| {
                        let previous_accesses = barrier
                            .previous_accesses
                            .iter()
                            .collect::<SmallVec<[AccessType; 10]>>();
                        let next_accesses = [barrier.next_access];
                        let sync = Submission::memory_barrier2(GlobalBarrier {
                            previous_accesses: &previous_accesses,
                            next_accesses: &next_accesses,
                        });
                        let ownership_acquire = barrier.ownership_layouts.is_some();
                        let (_, _, barrier) = TrackedImageBarrier::memory_barrier(barrier);

                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(if ownership_acquire {
                                vk::PipelineStageFlags2::ALL_COMMANDS
                            } else {
                                sync.src_stage_mask
                            })
                            .src_access_mask(if ownership_acquire {
                                vk::AccessFlags2::empty()
                            } else {
                                sync.src_access_mask
                            })
                            .dst_stage_mask(sync.dst_stage_mask)
                            .dst_access_mask(sync.dst_access_mask)
                            .old_layout(barrier.old_layout)
                            .new_layout(barrier.new_layout)
                            .src_queue_family_index(barrier.src_queue_family_index)
                            .dst_queue_family_index(barrier.dst_queue_family_index)
                            .image(barrier.image)
                            .subresource_range(barrier.subresource_range)
                    }));

                Device::cmd_pipeline_barrier2(
                    device,
                    command_buffer,
                    &vk::DependencyInfo::default()
                        .memory_barriers(&tls.memory_barriers)
                        .buffer_memory_barriers(&tls.buffer_barriers)
                        .image_memory_barriers(&tls.image_barriers),
                );
            });

            return;
        }

        #[derive(Default)]
        struct BarrierScratch<M, B, I> {
            memory_barriers: Vec<M>,
            buffer_barriers: Vec<B>,
            image_barriers: Vec<I>,
        }

        thread_local! {
            static BARRIER: RefCell<BarrierScratch<
                vk::MemoryBarrier<'static>,
                vk::BufferMemoryBarrier<'static>,
                vk::ImageMemoryBarrier<'static>,
            >> = Default::default();
        }

        BARRIER.with_borrow_mut(|tls| {
            tls.memory_barriers.clear();
            tls.buffer_barriers.clear();
            tls.image_barriers.clear();

            let mut src_stage_mask = vk::PipelineStageFlags::TOP_OF_PIPE;
            let mut dst_stage_mask = vk::PipelineStageFlags::BOTTOM_OF_PIPE;

            if let Some(ref barrier) = global_barrier {
                let (src_mask, dst_mask, memory_barrier) = get_memory_barrier(barrier);
                src_stage_mask |= src_mask;
                dst_stage_mask |= dst_mask;
                tls.memory_barriers.push(vk::MemoryBarrier {
                    src_access_mask: memory_barrier.src_access_mask,
                    dst_access_mask: memory_barrier.dst_access_mask,
                    ..Default::default()
                });
            }

            for buffer_barrier in buffer_barriers {
                let (src_mask, dst_mask, barrier) =
                    Submission::buffer_memory_barrier(buffer_barrier, queue_flags);
                src_stage_mask |= src_mask;
                dst_stage_mask |= dst_mask;
                tls.buffer_barriers.push(vk::BufferMemoryBarrier {
                    src_access_mask: barrier.src_access_mask,
                    dst_access_mask: barrier.dst_access_mask,
                    src_queue_family_index: barrier.src_queue_family_index,
                    dst_queue_family_index: barrier.dst_queue_family_index,
                    buffer: barrier.buffer,
                    offset: barrier.offset,
                    size: barrier.size,
                    ..Default::default()
                });
            }

            for &image_barrier in image_barriers {
                let (src_mask, dst_mask, barrier) =
                    TrackedImageBarrier::memory_barrier(image_barrier);
                src_stage_mask |= src_mask;
                dst_stage_mask |= dst_mask;
                tls.image_barriers.push(barrier);
            }

            if tls.memory_barriers.is_empty()
                && tls.buffer_barriers.is_empty()
                && tls.image_barriers.is_empty()
            {
                return;
            }

            Submission::with_legacy_barrier_batches(
                src_stage_mask,
                dst_stage_mask,
                &tls.memory_barriers,
                &mut tls.buffer_barriers,
                &mut tls.image_barriers,
                |src, dst, memory, buffers, images| unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        src,
                        dst,
                        vk::DependencyFlags::empty(),
                        memory,
                        buffers,
                        images,
                    );
                },
            );
        });
    }

    pub(crate) fn prepare_command_stream<P>(&mut self, pool: &mut P) -> Result<(), DriverError>
    where
        P: SubmissionPool,
    {
        if self.graph.cmds.is_empty() {
            return Ok(());
        }

        thread_local! {
            static SCHEDULE: RefCell<Schedule> = Default::default();
        }

        SCHEDULE.with_borrow_mut(|schedule| {
            schedule
                .access_index
                .update(&self.graph, self.graph.cmds.len());
            schedule.cmds.clear();
            schedule.cmds.extend(0..self.graph.cmds.len());

            debug_assert!(
                schedule.cmds.windows(2).all(|w| w[0] <= w[1]),
                "Unsorted schedule"
            );

            schedule.reorder_cmds(self.graph.cmds.len());
            self.merge_scheduled_cmds(&mut schedule.cmds);
            self.lease_scheduled_resources(pool, &schedule.cmds)
        })
    }

    fn prepare_timestamp_queries_for_commands(
        &mut self,
        command_indices: &[usize],
        include_final_timestamp_queries: bool,
    ) {
        let Some(timestamp_queries) = &mut self.graph.timestamp_queries else {
            return;
        };
        let Some(query_pool_results) = &mut self.query_pool_results else {
            return;
        };

        let cmds = &self.graph.cmds;
        let command_count = cmds.len();
        let mut scheduled_commands = FixedBitSet::with_capacity(command_count + 1);
        for command_idx in command_indices.iter().copied() {
            scheduled_commands.insert(command_idx);
        }
        if include_final_timestamp_queries {
            scheduled_commands.insert(command_count);
        }

        for timestamp_query in timestamp_queries.iter_mut().flatten() {
            if !scheduled_commands.contains(timestamp_query.command_idx) {
                continue;
            }

            let pool_query = timestamp_query.pool_query.unwrap_or_else(|| {
                let pool_query = query_pool_results.allocate_query(
                    Self::timestamp_query_pool_query_count(cmds, timestamp_query),
                );
                timestamp_query.pool_query = Some(pool_query);

                pool_query
            });

            query_pool_results.set_result_info(
                timestamp_query.query,
                TimestampQueryResultInfo {
                    timestamp_query: pool_query,
                },
            );
        }
    }

    fn prepare_timestamp_query_results(
        &mut self,
        cmd_buf: &CommandBuffer,
    ) -> Result<(), DriverError> {
        let Some(timestamp_queries) = &self.graph.timestamp_queries else {
            return Ok(());
        };

        let query_capacity = cmd_buf
            .device
            .physical
            .properties_v1_1
            .max_multiview_view_count
            .max(1);
        let pending_pool_query_count =
            timestamp_queries.iter().flatten().count() as u32 * query_capacity;
        if pending_pool_query_count == 0 {
            return Ok(());
        }

        let result_info_count = timestamp_queries
            .iter()
            .flatten()
            .map(|timestamp_query| timestamp_query.query.index() + 1)
            .max()
            .unwrap_or_default();

        self.query_pool_results = SubmittedTimestampQueries::create(
            &cmd_buf.device,
            cmd_buf.info.queue_family_index,
            result_info_count,
            1 + pending_pool_query_count,
        )
        .map(Some)?;

        Ok(())
    }

    fn queue_family_supports_timestamp_queries(queue_family: &QueueFamilyProperties) -> bool {
        queue_family.timestamp_valid_bits != 0
            && queue_family
                .queue_flags
                .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
    }

    /// Records and submits all remaining commands using an internally allocated command buffer.
    ///
    /// This legacy submit path only supports binary semaphore behavior. All wait and signal
    /// values must be `0`, and wait and signal stage masks must be `ALL_COMMANDS` or `NONE`.
    pub fn queue_submit<P>(
        self,
        resource_pool: &mut P,
        queue_family_index: u32,
        queue_index: u32,
    ) -> Result<Fence, DriverError>
    where
        P: Pool<CommandBufferInfo, CommandBuffer> + SubmissionPool,
    {
        trace!("queue_submit");

        /*
        Phase 1: Get the main command buffer and record commands. This also discovers any ownership
        transfers required by the scheduled work.
        */
        let cmd_buf = resource_pool.resource(CommandBufferInfo::new(queue_family_index as _))?;
        let mut fence = Fence::create(&cmd_buf.device, false)?;
        cmd_buf.begin(
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        let recording = self.record(resource_pool, cmd_buf, RecordSelection::All)?;
        recording.cmd_buf.end()?;

        let mut recorded = recording.finish()?;
        recorded.queue_submit(&mut fence, queue_index, QueueSubmitInfo::QUEUE_SUBMIT)?;

        fence.drop_when_signaled(recorded);

        Ok(fence)
    }

    fn rebuild_preserve_attachments(subpasses: &mut [SubpassInfo]) {
        for subpass in subpasses.iter_mut() {
            subpass.preserve_attachments.clear();
        }
        if subpasses.len() <= 1 {
            return;
        }
        let mut last_uses = BTreeMap::new();
        for subpass_idx in 0..subpasses.len() {
            let subpass = &subpasses[subpass_idx];
            let used = subpass
                .color_attachments
                .iter()
                .chain(&subpass.input_attachments)
                .chain(&subpass.color_resolve_attachments)
                .chain(subpass.depth_stencil_attachment.iter())
                .chain(
                    subpass
                        .depth_stencil_resolve_attachment
                        .iter()
                        .map(|(attachment, _, _)| attachment),
                )
                .map(|attachment| attachment.attachment)
                .filter(|&attachment| attachment != vk::ATTACHMENT_UNUSED)
                .collect::<BTreeSet<_>>();
            for attachment in used {
                if let Some(previous_idx) = last_uses.insert(attachment, subpass_idx) {
                    for skipped in &mut subpasses[previous_idx + 1..subpass_idx] {
                        skipped.preserve_attachments.push(attachment);
                    }
                }
            }
        }
        for subpass in subpasses {
            subpass.preserve_attachments.sort_unstable();
            subpass.preserve_attachments.dedup();
        }
    }

    /// Records any remaining graph commands into `cmd_buf` and returns a [`Recording`].
    ///
    /// When `selection` is [`RecordSelection::Nodes`], nodes are processed sequentially in the
    /// provided slice order and each step mutates the remaining submission state.
    #[profiling::function]
    pub fn record<'p, 's, P, Cb>(
        mut self,
        resource_pool: &'p mut P,
        cmd_buf: Cb,
        selection: impl Into<RecordSelection<'s>>,
    ) -> Result<Recording<'p, P, Cb>, DriverError>
    where
        P: SubmissionPool,
        Cb: AsRef<CommandBuffer>,
    {
        let mut ownership = RecordingOwnership::default();
        self.record_selection_impl(
            resource_pool,
            cmd_buf.as_ref(),
            selection.into(),
            &mut ownership,
        )?;

        Ok(Recording {
            ownership,
            cmd_buf,
            resource_pool,
            submission: self,
        })
    }

    fn record_cmd_indices(
        &mut self,
        cmd_buf: &CommandBuffer,
        cmd_indices: impl IntoIterator<Item = usize>,
        resource_set_synchronization: ResourceSetSynchronization,
        stream_values: Option<&crate::stream::StreamValues>,
    ) -> Result<(), DriverError> {
        #[cfg(feature = "checked")]
        let graph_id = self.graph.graph_id();
        let resource_set_count = self.graph.resource_sets.len();
        let mut acquired_resource_sets =
            FixedBitSet::with_capacity(resource_set_count * ResourceSetAccessType::COUNT);
        let query_pool = self
            .query_pool_results
            .as_ref()
            .map(SubmittedTimestampQueries::query_pool);
        for cmd_idx in cmd_indices {
            let timestamp_queries = self.take_timestamp_queries_for_command(cmd_idx);
            let cmd = &mut self.graph.cmds[cmd_idx];

            profiling::scope!("Cmd", cmd.name());
            let stream_label = cmd
                .stream_scope_id
                .and_then(|_| CommandBufferDebugLabel::begin(cmd_buf, "command stream boundary"));
            let _cmd_label = CommandBufferDebugLabel::begin(cmd_buf, cmd.name());
            let mut next_timestamp_query_idx = 0;

            if let Some(timestamp_queries) = &timestamp_queries {
                next_timestamp_query_idx = Self::write_timestamp_queries(
                    cmd_buf,
                    query_pool,
                    timestamp_queries,
                    TimestampQueryPlacement::BeforeExec,
                    0,
                    next_timestamp_query_idx,
                );
            }

            let recorded_command = &mut self.recorded_commands[cmd_idx];
            let is_graphics = recorded_command.render_pass.is_some();
            debug_assert!(
                !is_graphics
                    || Submission::valid_exec_subpasses(
                        cmd.execs.len(),
                        &recorded_command.exec_subpasses
                    )
            );

            trace!("recording cmd [{}: {}]", cmd_idx, cmd.name());

            if !recorded_command.descriptor_sets.is_empty() {
                Self::write_descriptor_sets(cmd_buf, &self.graph.resources, cmd, recorded_command)?;
            }

            let (render_area, render_pass_label) = if is_graphics {
                if resource_set_synchronization == ResourceSetSynchronization::Enabled
                    && cmd
                        .execs
                        .iter()
                        .flat_map(|exec| &exec.resource_set_accesses)
                        .any(|&access| {
                            !Self::resource_set_access_is_acquired(
                                access,
                                resource_set_count,
                                &acquired_resource_sets,
                            )
                        })
                {
                    Self::record_resource_set_acquisitions(
                        cmd_buf,
                        &self.graph.resource_sets,
                        cmd.execs
                            .iter()
                            .flat_map(|exec| &exec.resource_set_accesses),
                        &mut acquired_resource_sets,
                        &mut self.pending_image_set_transfers,
                    );
                }
                Self::record_image_layout_transitions(
                    cmd_buf,
                    &mut self.graph.resources,
                    cmd,
                    &mut self.pending_buffer_transfer_nodes,
                    &mut self.pending_image_transfer_nodes,
                );

                let render_area = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: Self::render_extent(&self.graph.resources, cmd),
                };
                let render_pass_label = CommandBufferDebugLabel::begin(
                    cmd_buf,
                    format!("{} / render pass", cmd.name()),
                );

                Self::begin_render_pass(
                    cmd_buf,
                    &self.graph.resources,
                    cmd,
                    recorded_command,
                    render_area,
                )?;

                (Some(render_area), render_pass_label)
            } else {
                (None, None)
            };

            let mut loaded_color_attachments = SmallVec::<[bool; 8]>::new();
            if let Some(render_pass) = &recorded_command.render_pass {
                loaded_color_attachments.resize(render_pass.info.attachments.len(), false);
            }
            let mut loaded_depth_stencil = false;
            for exec_idx in 0..cmd.execs.len() {
                let exec_render_area = if is_graphics {
                    Some(
                        cmd.execs[exec_idx]
                            .render_area
                            .unwrap_or(render_area.expect("missing render area")),
                    )
                } else {
                    None
                };
                let exec_label_name = cmd_buf
                    .device
                    .physical
                    .instance
                    .info
                    .debug
                    .then(|| format!("{} / exec {exec_idx}", cmd.name()));

                let exec = &mut cmd.execs[exec_idx];

                if exec_idx > 0 {
                    if is_graphics
                        && recorded_command.exec_subpasses[exec_idx]
                            != recorded_command.exec_subpasses[exec_idx - 1]
                    {
                        Self::next_subpass(cmd_buf);
                    }

                    if let Some(timestamp_queries) = &timestamp_queries {
                        next_timestamp_query_idx = Self::write_timestamp_queries(
                            cmd_buf,
                            query_pool,
                            timestamp_queries,
                            TimestampQueryPlacement::BeforeExec,
                            exec_idx,
                            next_timestamp_query_idx,
                        );
                    }
                }

                if let Some(render_area) = render_area {
                    // Load operations apply only to the first declaration of each slot.
                    // Later clears must run in their subpass, independently of callback scissor.
                    let clear = |attachment: vk::ClearAttachment, layer_count| unsafe {
                        cmd_buf.device.cmd_clear_attachments(
                            cmd_buf.handle,
                            &[attachment],
                            &[vk::ClearRect::default()
                                .rect(render_area)
                                .layer_count(if exec.view_mask == 0 { layer_count } else { 1 })],
                        );
                    };
                    for (attachment_idx, state) in exec.attachments.color_attachments() {
                        let loaded = std::mem::replace(
                            &mut loaded_color_attachments[attachment_idx as usize],
                            true,
                        );
                        if loaded
                            && state.is_attachment
                            && let LoadOp::Clear(value) = state.load
                        {
                            clear(
                                vk::ClearAttachment::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .color_attachment(attachment_idx)
                                    .clear_value(vk::ClearValue {
                                        color: vk::ClearColorValue { float32: value },
                                    }),
                                state.attachment.array_layer_count,
                            );
                        }
                    }
                    if let Some(state) = exec.attachments.depth_stencil_attachment() {
                        if loaded_depth_stencil
                            && state.is_attachment
                            && let LoadOp::Clear(depth_stencil) = state.load
                        {
                            clear(
                                vk::ClearAttachment::default()
                                    .aspect_mask(state.attachment.aspect_mask)
                                    .clear_value(vk::ClearValue { depth_stencil }),
                                state.attachment.array_layer_count,
                            );
                        }
                        loaded_depth_stencil = true;
                    }
                }

                if let Some(pipeline) = exec.pipeline.as_mut() {
                    Self::bind_pipeline(
                        cmd_buf,
                        recorded_command,
                        exec_idx,
                        pipeline,
                        exec.depth_stencil,
                    )?;

                    if is_graphics {
                        let render_area = exec_render_area.expect("missing render area");

                        // In this case we set the viewport and scissor for the user
                        Self::set_viewport(
                            cmd_buf,
                            render_area.offset.x as _,
                            render_area.offset.y as _,
                            render_area.extent.width as _,
                            render_area.extent.height as _,
                            exec.depth_stencil
                                .map(|depth_stencil| {
                                    let min = depth_stencil.min.0;
                                    let max = depth_stencil.max.0;
                                    min..max
                                })
                                .unwrap_or(0.0..1.0),
                        );
                        Self::set_scissor(
                            cmd_buf,
                            render_area.offset.x,
                            render_area.offset.y,
                            render_area.extent.width,
                            render_area.extent.height,
                        );
                    }

                    Self::bind_descriptor_sets(cmd_buf, pipeline, recorded_command, exec_idx);
                }

                if !is_graphics {
                    if resource_set_synchronization == ResourceSetSynchronization::Enabled
                        && exec.resource_set_accesses.iter().any(|&access| {
                            !Self::resource_set_access_is_acquired(
                                access,
                                resource_set_count,
                                &acquired_resource_sets,
                            )
                        })
                    {
                        Self::record_resource_set_acquisitions(
                            cmd_buf,
                            &self.graph.resource_sets,
                            exec.resource_set_accesses.iter(),
                            &mut acquired_resource_sets,
                            &mut self.pending_image_set_transfers,
                        );
                    }
                    Self::record_execution_barriers(
                        cmd_buf,
                        &mut self.graph.resources,
                        &exec.accesses,
                        &mut self.pending_buffer_transfer_nodes,
                        &mut self.pending_image_transfer_nodes,
                    );
                }

                trace!("    > exec[{exec_idx}]");

                {
                    profiling::scope!("Execute callback");
                    let _exec_label = exec_label_name.as_deref().and_then(|exec_label_name| {
                        CommandBufferDebugLabel::begin(cmd_buf, exec_label_name)
                    });

                    let exec_func = exec.func.take().expect("missing command function");
                    exec.func = exec_func.record(CommandRef::new(
                        cmd_buf,
                        &self.graph.resources,
                        &self.graph.resource_sets,
                        exec,
                        stream_values,
                        #[cfg(feature = "checked")]
                        graph_id,
                    ));
                }

                if let Some(timestamp_queries) = &timestamp_queries {
                    next_timestamp_query_idx = Self::write_timestamp_queries(
                        cmd_buf,
                        query_pool,
                        timestamp_queries,
                        TimestampQueryPlacement::AfterExec,
                        exec_idx,
                        next_timestamp_query_idx,
                    );
                }
            }

            if is_graphics {
                trace!("  end render pass");

                cmd_buf.end_render_pass();
            }

            drop(render_pass_label);
            drop(stream_label);
        }
        Ok(())
    }

    #[profiling::function]
    fn record_execution_barriers<'a>(
        cmd_buf: &CommandBuffer,
        resources: &mut [AnyResource],
        accesses: &'a ExecutionAccess,
        pending_buffer_transfer_nodes: &mut Option<
            PendingTransferNodes<vk::Buffer, BufferQueueOwnershipTransfer>,
        >,
        pending_image_transfer_nodes: &mut Option<
            PendingTransferNodes<vk::Image, ImageOwnershipTransfer>,
        >,
    ) {
        // We store a Barriers in TLS to save an alloc; contents are POD
        thread_local! {
            static BARRIER: RefCell<BarrierScratch> = Default::default();
        }

        struct AccessBarrier<T, A> {
            next_access: AccessType,
            prev_access: A,
            resource: T,
        }

        #[derive(Default)]
        struct BarrierScratch {
            accel_struct_accesses: Vec<AccessType>,
            buffers: Vec<AccessBarrier<BufferBarrierTarget, AccessType>>,
            image_accesses: Vec<(AccessType, vk::ImageSubresourceRange, bool)>,
            images: Vec<AccessBarrier<ImageBarrierTarget, ImageAccessSet>>,
            next_accesses: Vec<AccessType>,
            micromap_accesses: Vec<AccessType>,
            next_micromap_accesses: Vec<AccessType>,
            pending_buffers: NodeIndexedScratch<AccessBarrier<BufferBarrierTarget, AccessType>>,
            pending_images: NodeIndexedScratch<AccessBarrier<ImageBarrierTarget, ImageAccessSet>>,
            prev_accesses: Vec<AccessType>,
            prev_micromap_accesses: Vec<AccessType>,
        }

        struct BufferBarrierTarget {
            buffer: vk::Buffer,
            range: BufferSubresourceRange,
        }

        struct ImageBarrierTarget {
            image: vk::Image,
            range: vk::ImageSubresourceRange,
        }

        BARRIER.with_borrow_mut(|tls| {
            // Initialize TLS from a previous call
            tls.accel_struct_accesses.clear();
            tls.buffers.clear();
            tls.image_accesses.clear();
            tls.images.clear();
            tls.next_accesses.clear();
            tls.micromap_accesses.clear();
            tls.next_micromap_accesses.clear();
            tls.pending_buffers.clear();
            tls.pending_images.clear();
            tls.prev_accesses.clear();
            tls.prev_micromap_accesses.clear();

            // Map remaining accesses into vk_sync barriers (some accesses may have been removed by
            // the render pass request function)

            for (node_idx, node_accesses) in accesses.iter() {
                enum ResourceRef<'a> {
                    AccelerationStructure(&'a AccelerationStructure),
                    Buffer(&'a Buffer),
                    Image(&'a Image),
                    Micromap(&'a Micromap),
                }

                let resource = match &resources[node_idx] {
                    AnyResource::AccelerationStructure(resource) => {
                        ResourceRef::AccelerationStructure(resource)
                    }
                    AnyResource::AccelerationStructureArg(_) => {
                        panic!("unbound command stream acceleration structure argument")
                    }
                    AnyResource::AccelerationStructureLease(resource) => {
                        ResourceRef::AccelerationStructure(resource)
                    }
                    AnyResource::Buffer(resource) => ResourceRef::Buffer(resource),
                    AnyResource::BufferArg(_) => panic!("unbound command stream buffer argument"),
                    AnyResource::BufferLease(resource) => ResourceRef::Buffer(resource),
                    AnyResource::Image(resource) => ResourceRef::Image(resource),
                    AnyResource::ImageArg(_) => panic!("unbound command stream image argument"),
                    AnyResource::ImageLease(resource) => ResourceRef::Image(resource),
                    AnyResource::Micromap(resource) => ResourceRef::Micromap(resource),
                    AnyResource::MicromapArg(_) => {
                        panic!("unbound command stream micromap argument")
                    }
                    AnyResource::MicromapLease(resource) => ResourceRef::Micromap(resource),
                    AnyResource::SwapchainImage(resource) => ResourceRef::Image(resource),
                };

                match resource {
                    ResourceRef::AccelerationStructure(accel_struct) => {
                        let canonical_accesses = Self::whole_resource_canonical_accesses(
                            node_accesses,
                            &mut tls.accel_struct_accesses,
                        );
                        tls.next_accesses.extend(canonical_accesses.iter().copied());
                        tls.prev_accesses
                            .extend(AccelerationStructure::swap_accesses(
                                accel_struct,
                                canonical_accesses,
                            ));
                    }
                    ResourceRef::Buffer(buffer) => {
                        for (next_access, prev_access, range) in Buffer::swap_accesses(
                            buffer,
                            node_accesses.iter().map(
                                |&SubresourceAccess {
                                     access,
                                     subresource,
                                 }| {
                                    let SubresourceRange::Buffer(range) = subresource else {
                                        unreachable!()
                                    };

                                    (access, range)
                                },
                            ),
                        ) {
                            let barrier = AccessBarrier {
                                next_access,
                                prev_access,
                                resource: BufferBarrierTarget {
                                    buffer: buffer.handle,
                                    range,
                                },
                            };

                            if pending_buffer_transfer_nodes
                                .as_ref()
                                .is_some_and(|pending| pending.contains(node_idx))
                            {
                                tls.pending_buffers.push(node_idx, barrier);
                            } else {
                                tls.buffers.push(barrier);
                            }
                        }
                    }
                    ResourceRef::Image(image) => {
                        let transfers = pending_image_transfer_nodes
                            .as_ref()
                            .and_then(|pending| pending.get(node_idx))
                            .map_or_else(Default::default, |(_, transfers)| transfers);
                        let mut image_accesses = take(&mut tls.image_accesses);
                        image_accesses.clear();

                        for &SubresourceAccess {
                            access,
                            subresource,
                        } in node_accesses
                        {
                            let SubresourceRange::Image(range) = subresource else {
                                unreachable!()
                            };
                            let range = image.info.resolve_subresource_counts(range);

                            image_accesses.extend(
                                ImageOwnershipTransfer::barrier_ranges(transfers, range).map(
                                    move |(range, transfer)| {
                                        (
                                            access,
                                            range,
                                            transfer.is_none()
                                                && ImageAccessSet::from_access(access)
                                                    .is_sampled_read(),
                                        )
                                    },
                                ),
                            );
                        }

                        for (next_access, prev_access, range) in
                            Image::swap_accesses(image, image_accesses.iter().copied())
                        {
                            let barrier = AccessBarrier {
                                next_access,
                                prev_access,
                                resource: ImageBarrierTarget {
                                    image: image.handle,
                                    range,
                                },
                            };

                            if pending_image_transfer_nodes
                                .as_ref()
                                .is_some_and(|pending| pending.contains(node_idx))
                            {
                                tls.pending_images.push(node_idx, barrier);
                            } else {
                                tls.images.push(barrier);
                            }
                        }

                        tls.image_accesses = image_accesses;
                    }
                    ResourceRef::Micromap(micromap) => {
                        debug_assert!(node_accesses.iter().all(|access| matches!(
                            access.subresource,
                            SubresourceRange::Micromap
                        )));
                        let canonical_accesses = Self::whole_resource_canonical_accesses(
                            node_accesses,
                            &mut tls.micromap_accesses,
                        );
                        tls.next_micromap_accesses
                            .extend(canonical_accesses.iter().copied());
                        tls.prev_micromap_accesses
                            .extend(Micromap::swap_accesses(micromap, canonical_accesses));
                    }
                }
            }

            let global_barrier = if !tls.next_accesses.is_empty() {
                // No resource attached - we use a global barrier for these
                trace!(
                    "    global {:?}->{:?}",
                    tls.next_accesses, tls.prev_accesses
                );

                Some(GlobalBarrier {
                    next_accesses: tls.next_accesses.as_slice(),
                    previous_accesses: tls.prev_accesses.as_slice(),
                })
            } else {
                None
            };
            let micromap_barrier =
                (!tls.next_micromap_accesses.is_empty()).then_some(GlobalBarrier {
                    next_accesses: tls.next_micromap_accesses.as_slice(),
                    previous_accesses: tls.prev_micromap_accesses.as_slice(),
                });

            let mut buffer_barriers = Vec::new();
            for AccessBarrier {
                next_access,
                prev_access,
                resource,
            } in tls.buffers.iter()
            {
                let BufferBarrierTarget { buffer, range, .. } = *resource;

                buffer_barriers.push(BufferBarrier {
                    next_accesses: slice::from_ref(next_access),
                    previous_accesses: slice::from_ref(prev_access),
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    buffer,
                    offset: range.start as _,
                    size: (range.end - range.start) as _,
                });
            }

            if let Some(pending_buffer_transfer_nodes) = pending_buffer_transfer_nodes.as_ref() {
                for (node_idx, _buffer, transfers) in pending_buffer_transfer_nodes.iter() {
                    for AccessBarrier {
                        next_access,
                        prev_access,
                        resource,
                    } in tls.pending_buffers.get(node_idx)
                    {
                        buffer_barriers.extend(BufferQueueOwnershipTransfer::barriers(
                            resource.buffer,
                            prev_access,
                            next_access,
                            resource.range,
                            transfers,
                        ));
                    }
                }
            }

            let mut image_barriers = Vec::new();
            for AccessBarrier {
                next_access,
                prev_access,
                resource,
            } in tls.images.iter()
            {
                let ImageBarrierTarget { image, range, .. } = *resource;

                let barrier = TrackedImageBarrier {
                    next_access: *next_access,
                    previous_accesses: *prev_access,
                    next_layout: TrackedImageBarrier::access_layout(*next_access),
                    previous_layout: TrackedImageBarrier::access_set_layout(*prev_access),
                    ownership_layouts: None,
                    discard_contents: TrackedImageBarrier::execution_discard_contents(*prev_access),
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    range,
                };

                if !TrackedImageBarrier::can_elide_sampled_read(barrier) {
                    image_barriers.push(barrier);
                }
            }

            if let Some(pending_image_transfer_nodes) = pending_image_transfer_nodes.as_ref() {
                for (node_idx, _image, transfers) in pending_image_transfer_nodes.iter() {
                    for AccessBarrier {
                        next_access,
                        prev_access,
                        resource,
                    } in tls.pending_images.get(node_idx)
                    {
                        image_barriers.extend(
                            TrackedImageBarrier::from_transfers(
                                resource.image,
                                *prev_access,
                                *next_access,
                                resource.range,
                                transfers,
                                TrackedImageBarrier::execution_discard_contents(*prev_access),
                            )
                            .filter(|barrier| {
                                !TrackedImageBarrier::can_elide_sampled_read(*barrier)
                            }),
                        );
                    }
                }
            }

            Submission::pipeline_barrier_from_slices(
                cmd_buf,
                global_barrier,
                micromap_barrier,
                &buffer_barriers,
                &image_barriers,
            );

            if let Some(pending) = pending_buffer_transfer_nodes.as_mut() {
                pending.remove_where(|node_idx, _buffer, transfers| {
                    for AccessBarrier { resource, .. } in tls.pending_buffers.get(node_idx) {
                        let range = resource.range;

                        if BufferQueueOwnershipTransfer::consume_pending(transfers, range) {
                            return true;
                        }
                    }

                    false
                });

                if pending.is_empty() {
                    *pending_buffer_transfer_nodes = None;
                }
            }

            if let Some(pending) = pending_image_transfer_nodes.as_mut() {
                pending.remove_where(|node_idx, _image, transfers| {
                    for AccessBarrier { resource, .. } in tls.pending_images.get(node_idx) {
                        let range = resource.range;

                        if ImageOwnershipTransfer::consume_pending(transfers, range) {
                            return true;
                        }
                    }

                    false
                });

                if pending.is_empty() {
                    *pending_image_transfer_nodes = None;
                }
            }
        });
    }

    #[profiling::function]
    fn record_image_layout_transitions(
        cmd_buf: &CommandBuffer,
        resources: &mut [AnyResource],
        pass: &mut CommandData,
        pending_buffer_transfer_nodes: &mut Option<
            PendingTransferNodes<vk::Buffer, BufferQueueOwnershipTransfer>,
        >,
        pending_image_transfer_nodes: &mut Option<
            PendingTransferNodes<vk::Image, ImageOwnershipTransfer>,
        >,
    ) {
        #[cfg(test)]
        test::SubpassFixture::check_incoming_buffer_source_scope(
            PipelineStageAccessFlags::buffer_source_scope,
        );

        struct BufferResourceBarrier {
            buffer: vk::Buffer,
            next_access: AccessType,
            prev_access: AccessType,
            range: BufferSubresourceRange,
        }

        struct ImageResourceBarrier {
            image: vk::Image,
            node_idx: NodeIndex,
            next_access: AccessType,
            prev_access: ImageAccessSet,
            range: vk::ImageSubresourceRange,
        }

        #[derive(Default)]
        struct LayoutTransitionScratch {
            buffers: Vec<BufferResourceBarrier>,
            images: Vec<ImageResourceBarrier>,
            incoming_buffers: HashMap<vk::Buffer, usize>,
            incoming_ranges: Vec<Vec<(AccessType, BufferSubresourceRange)>>,
            first_layout_uses: HashMap<usize, DenseMap<bool>>,
            pending_buffers: NodeIndexedScratch<BufferResourceBarrier>,
            pending_images: NodeIndexedScratch<ImageResourceBarrier>,
        }

        // We store a LayoutTransitionScratch in TLS to save an alloc; contents are POD
        thread_local! {
            static LAYOUT_TRANSITION: RefCell<LayoutTransitionScratch> = Default::default();
        }

        LAYOUT_TRANSITION.with_borrow_mut(|tls| {
            tls.buffers.clear();
            tls.images.clear();
            tls.incoming_buffers.clear();
            for ranges in &mut tls.incoming_ranges {
                ranges.clear();
            }
            tls.first_layout_uses.clear();
            tls.pending_buffers.clear();
            tls.pending_images.clear();
            let mut buffer_consumers = HashMap::new();

            for (node_idx, accesses) in pass.execs.iter_mut().flat_map(|exec| exec.accesses.iter())
            {
                debug_assert!(resources.get(node_idx).is_some());

                let resource = unsafe {
                    // CommandRef enforces this during push_resource_access
                    resources.get_unchecked(node_idx)
                };

                enum ResourceRef<'a> {
                    AccelerationStructure(&'a AccelerationStructure),
                    Buffer(&'a Buffer),
                    Image(&'a Image),
                    Micromap(&'a Micromap),
                }

                let resource = match resource {
                    AnyResource::AccelerationStructure(resource) => {
                        ResourceRef::AccelerationStructure(resource)
                    }
                    AnyResource::AccelerationStructureArg(_) => {
                        panic!("unbound command stream acceleration structure argument")
                    }
                    AnyResource::AccelerationStructureLease(resource) => {
                        ResourceRef::AccelerationStructure(resource)
                    }
                    AnyResource::Buffer(resource) => ResourceRef::Buffer(resource),
                    AnyResource::BufferArg(_) => panic!("unbound command stream buffer argument"),
                    AnyResource::BufferLease(resource) => ResourceRef::Buffer(resource),
                    AnyResource::Image(resource) => ResourceRef::Image(resource),
                    AnyResource::ImageArg(_) => panic!("unbound command stream image argument"),
                    AnyResource::ImageLease(resource) => ResourceRef::Image(resource),
                    AnyResource::Micromap(resource) => ResourceRef::Micromap(resource),
                    AnyResource::MicromapArg(_) => {
                        panic!("unbound command stream micromap argument")
                    }
                    AnyResource::MicromapLease(resource) => ResourceRef::Micromap(resource),
                    AnyResource::SwapchainImage(resource) => ResourceRef::Image(resource),
                };

                match resource {
                    ResourceRef::AccelerationStructure(accel_struct) => {
                        AccelerationStructure::swap_access(accel_struct, AccessType::Nothing)
                            .for_each(drop);
                    }
                    ResourceRef::Buffer(buffer) => {
                        for subresource_access in accesses {
                            let &SubresourceAccess {
                                access,
                                subresource: SubresourceRange::Buffer(access_range),
                            } = subresource_access
                            else {
                                #[cfg(feature = "checked")]
                                unreachable!();

                                #[cfg(not(feature = "checked"))]
                                unsafe {
                                    // This cannot be reached because command access recording
                                    // preserves the buffer subresource type for this node.
                                    unreachable_unchecked()
                                }
                            };

                            let access_range = access_range.resolve_whole(buffer.info.size);
                            // Internal read/read edges may be pruned, so the last reader's
                            // stage alone cannot represent the pass. Retain writers as General.
                            let tracked_access =
                                if access != AccessType::Nothing && !is_write_access(access) {
                                    AccessType::AnyShaderReadOther
                                } else {
                                    access
                                };
                            // Keep coverage node-local like pending ownership, but share the
                            // pre-pass producer snapshot below by physical buffer handle.
                            let covered = buffer_consumers
                                .entry((node_idx, std::mem::discriminant(&access)))
                                .or_insert_with(SmallVec::<[BufferSubresourceRange; 1]>::new);
                            let first =
                                covered.partition_point(|range| range.end < access_range.start);
                            let end = first
                                + covered[first..]
                                    .partition_point(|range| range.start <= access_range.end);
                            let mut uncovered = SmallVec::<[BufferSubresourceRange; 4]>::new();
                            let mut start = access_range.start;
                            for range in &covered[first..end] {
                                if start < range.start {
                                    uncovered.push(BufferSubresourceRange {
                                        start,
                                        end: range.start.min(access_range.end),
                                    });
                                }
                                start = range.end.max(start);
                            }
                            if start < access_range.end {
                                uncovered.push(BufferSubresourceRange {
                                    start,
                                    end: access_range.end,
                                });
                            }
                            if uncovered.is_empty() {
                                // The pre-pass dependency already covers this consumer.
                                // Still track it: intervening accesses may have changed the writer.
                                Buffer::swap_accesses(buffer, [(tracked_access, access_range)])
                                    .for_each(drop);
                                continue;
                            }
                            let mut union = access_range;
                            if first < end {
                                union.start = union.start.min(covered[first].start);
                                union.end = union.end.max(covered[end - 1].end);
                            }
                            covered.drain(first..end);
                            covered.insert(first, union);
                            let next_idx = tls.incoming_buffers.len();
                            let idx = *tls
                                .incoming_buffers
                                .entry(buffer.handle)
                                .or_insert(next_idx);
                            if idx == tls.incoming_ranges.len() {
                                tls.incoming_ranges.push(Vec::new());
                            }
                            let incoming = &mut tls.incoming_ranges[idx];
                            // Capture history only on first touch of each byte. Later swaps
                            // include this render pass's accesses, which cannot be sources of
                            // a barrier recorded before the render pass executes.
                            let mut unseen = SmallVec::<[BufferSubresourceRange; 4]>::new();
                            let mut start = access_range.start;
                            let first = incoming.partition_point(|(_, range)| range.end <= start);
                            let end = first
                                + incoming[first..]
                                    .partition_point(|(_, range)| range.start < access_range.end);
                            for &(_, range) in &incoming[first..end] {
                                if start < range.start {
                                    unseen.push(BufferSubresourceRange {
                                        start,
                                        end: range.start,
                                    });
                                }
                                start = range.end.min(access_range.end);
                            }
                            if start < access_range.end {
                                unseen.push(BufferSubresourceRange {
                                    start,
                                    end: access_range.end,
                                });
                            }

                            let mut fragments =
                                SmallVec::<[(AccessType, BufferSubresourceRange); 4]>::new();
                            for (_, prev_access, range) in
                                Buffer::swap_accesses(buffer, [(tracked_access, access_range)])
                            {
                                let first =
                                    unseen.partition_point(|unseen| unseen.end <= range.start);
                                for unseen in unseen[first..]
                                    .iter()
                                    .take_while(|unseen| unseen.start < range.end)
                                {
                                    if let Some(range) = range.intersection(*unseen) {
                                        fragments.push((prev_access, range));
                                    }
                                }
                            }
                            if !fragments.is_empty() {
                                // Merge just the affected window, including its neighbours so
                                // equal adjacent intervals coalesce without scanning the prefix.
                                let first = first.saturating_sub(1);
                                let end = (end + 1).min(incoming.len());
                                let mut previous = incoming[first..end].iter().copied().peekable();
                                let mut fragments = fragments.into_iter().peekable();
                                let mut merged =
                                    SmallVec::<[(AccessType, BufferSubresourceRange); 4]>::new();
                                while previous.peek().is_some() || fragments.peek().is_some() {
                                    let next = if fragments.peek().is_some_and(|(_, range)| {
                                        previous
                                            .peek()
                                            .is_none_or(|(_, old)| range.start < old.start)
                                    }) {
                                        fragments.next().unwrap()
                                    } else {
                                        previous.next().unwrap()
                                    };
                                    if let Some(last) = merged.last_mut()
                                        && last.0 == next.0
                                        && last.1.end == next.1.start
                                    {
                                        last.1.end = next.1.end;
                                    } else {
                                        merged.push(next);
                                    }
                                }
                                incoming.splice(first..end, merged);
                            }

                            // Only new coverage needs barriers. Different access types retain
                            // their own coverage; internal hazards remain subpass edges.
                            for access_range in uncovered {
                                let first = incoming
                                    .partition_point(|(_, range)| range.end <= access_range.start);
                                for &(prev_access, range) in incoming[first..]
                                    .iter()
                                    .take_while(|(_, range)| range.start < access_range.end)
                                {
                                    let range = range.intersection(access_range).unwrap();

                                    let barrier = BufferResourceBarrier {
                                        buffer: buffer.handle,
                                        next_access: access,
                                        prev_access,
                                        range,
                                    };
                                    if pending_buffer_transfer_nodes
                                        .as_ref()
                                        .is_some_and(|pending| pending.contains(node_idx))
                                    {
                                        tls.pending_buffers.push(node_idx, barrier);
                                    } else if prev_access != AccessType::Nothing
                                        || is_write_access(access)
                                    {
                                        tls.buffers.push(barrier);
                                    }
                                }
                            }
                        }
                    }
                    ResourceRef::Image(image) => {
                        let first_layout_uses = tls
                            .first_layout_uses
                            .entry(node_idx)
                            .or_insert_with(|| DenseMap::new(image.info, true));

                        for subresource_access in accesses {
                            let &SubresourceAccess {
                                access,
                                subresource: SubresourceRange::Image(access_range),
                            } = subresource_access
                            else {
                                #[cfg(feature = "checked")]
                                unreachable!();

                                #[cfg(not(feature = "checked"))]
                                unsafe {
                                    // This cannot be reached because command access recording
                                    // preserves the image subresource type for this node.
                                    unreachable_unchecked()
                                }
                            };

                            let access_range = image.info.resolve_subresource_counts(access_range);

                            for (is_initial_layout, layout_range) in
                                first_layout_uses.swap(false, access_range)
                            {
                                if is_initial_layout {
                                    for (prev_access, range) in
                                        Image::replace_access(image, access, layout_range)
                                    {
                                        let barrier = ImageResourceBarrier {
                                            image: image.handle,
                                            node_idx,
                                            next_access: initial_image_layout_access(access),
                                            prev_access,
                                            range,
                                        };

                                        if pending_image_transfer_nodes
                                            .as_ref()
                                            .is_some_and(|pending| pending.contains(node_idx))
                                        {
                                            tls.pending_images.push(node_idx, barrier);
                                        } else {
                                            tls.images.push(barrier);
                                        }
                                    }
                                } else {
                                    Image::swap_access(image, access, layout_range).for_each(drop);
                                }
                            }
                        }
                    }
                    ResourceRef::Micromap(micromap) => {
                        debug_assert!(accesses.iter().all(|access| matches!(
                            access.subresource,
                            SubresourceRange::Micromap
                        )));
                        Micromap::swap_access(micromap, AccessType::Nothing).for_each(drop);
                    }
                }
            }

            // Add resolve scopes after all executions have published their final layouts. These
            // scopes must survive later depth tests and remain available to the next graph too.
            for exec in &pass.execs {
                let Some(state) = exec.attachments.depth_stencil_attachment() else {
                    continue;
                };
                let Some(resolve) = state.resolve else {
                    continue;
                };
                let mut aspects = vk::ImageAspectFlags::empty();
                if resolve.depth_mode.is_some() {
                    aspects |= vk::ImageAspectFlags::DEPTH;
                }
                if resolve.stencil_mode.is_some() {
                    aspects |= vk::ImageAspectFlags::STENCIL;
                }
                for (attachment, write) in [(state.attachment, false), (resolve.attachment, true)] {
                    let image = Self::expect_attachment_image(resources, &attachment);
                    let mut range: vk::ImageSubresourceRange =
                        attachment.image_view_info(image.info).into();
                    range.aspect_mask &= aspects;
                    if !range.aspect_mask.is_empty() {
                        image.with_access(range, |access| access.with_depth_stencil_resolve(write));
                    }
                }
            }

            let mut buffer_barriers = Vec::new();
            for barrier in &tls.buffers {
                buffer_barriers.extend(BufferQueueOwnershipTransfer::barriers(
                    barrier.buffer,
                    &barrier.prev_access,
                    &barrier.next_access,
                    barrier.range,
                    &[],
                ));
            }
            if let Some(pending_buffer_transfer_nodes) = pending_buffer_transfer_nodes.as_ref() {
                for (node_idx, _buffer, transfers) in pending_buffer_transfer_nodes.iter() {
                    for BufferResourceBarrier {
                        buffer,
                        next_access,
                        prev_access,
                        range,
                        ..
                    } in tls.pending_buffers.get(node_idx)
                    {
                        buffer_barriers.extend(BufferQueueOwnershipTransfer::barriers(
                            *buffer,
                            prev_access,
                            next_access,
                            *range,
                            transfers,
                        ));
                    }
                }
            }

            let mut image_barriers = Vec::new();
            for ImageResourceBarrier {
                image,
                node_idx,
                next_access,
                prev_access,
                range,
            } in tls.images.iter()
            {
                if pending_image_transfer_nodes
                    .as_ref()
                    .is_some_and(|pending| pending.contains(*node_idx))
                {
                    continue;
                }

                image_barriers.extend(TrackedImageBarrier::from_transfers(
                    *image,
                    *prev_access,
                    *next_access,
                    *range,
                    &[],
                    TrackedImageBarrier::layout_transition_discard_contents(
                        *prev_access,
                        *next_access,
                    ),
                ));
            }

            if let Some(pending_image_transfer_nodes) = pending_image_transfer_nodes.as_ref() {
                for (node_idx, _image, transfers) in pending_image_transfer_nodes.iter() {
                    for ImageResourceBarrier {
                        image,
                        next_access,
                        prev_access,
                        range,
                        ..
                    } in tls.pending_images.get(node_idx)
                    {
                        image_barriers.extend(TrackedImageBarrier::from_transfers(
                            *image,
                            *prev_access,
                            *next_access,
                            *range,
                            transfers,
                            TrackedImageBarrier::layout_transition_discard_contents(
                                *prev_access,
                                *next_access,
                            ),
                        ));
                    }
                }
            }

            {
                // vk-sync uses SHADER_READ for some uniform-buffer stages. Our canonical
                // legacy scopes use UNIFORM_READ (and conservative micromap scopes).
                let mut src_stages = vk::PipelineStageFlags::TOP_OF_PIPE;
                let mut dst_stages = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
                let queue_flags = cmd_buf.device.physical.queue_families
                    [cmd_buf.info.queue_family_index as usize]
                    .queue_flags;
                let mut buffers = buffer_barriers
                    .iter()
                    .filter_map(|barrier| {
                        let ownership_acquire =
                            barrier.src_queue_family_index != vk::QUEUE_FAMILY_IGNORED;
                        // Only Nothing -> read can be elided. The tracker replaces pure
                        // readers, so read/read execution edges carry ordering to later writes.
                        // Split ownership first: acquires also need visibility from Nothing.
                        if !ownership_acquire
                            && barrier.previous_accesses[0] == AccessType::Nothing
                            && !is_write_access(barrier.next_accesses[0])
                        {
                            return None;
                        }
                        let (src, src_access) = PipelineStageAccessFlags::buffer_source_scope(
                            barrier.previous_accesses[0],
                            queue_flags,
                        );
                        let (dst, dst_access) =
                            pipeline_stage_access_flags(barrier.next_accesses[0]);
                        // The release supplies availability; its producer stage may not
                        // even be supported by this queue family.
                        if !ownership_acquire {
                            src_stages |= src;
                        }
                        dst_stages |= dst;
                        let visibility =
                            is_write_access(barrier.previous_accesses[0]) || ownership_acquire;
                        Some(
                            vk::BufferMemoryBarrier::default()
                                .src_access_mask(
                                    if !ownership_acquire
                                        && is_write_access(barrier.previous_accesses[0])
                                    {
                                        src_access
                                    } else {
                                        vk::AccessFlags::empty()
                                    },
                                )
                                .dst_access_mask(if visibility {
                                    dst_access
                                } else {
                                    vk::AccessFlags::empty()
                                })
                                .src_queue_family_index(barrier.src_queue_family_index)
                                .dst_queue_family_index(barrier.dst_queue_family_index)
                                .buffer(barrier.buffer)
                                .offset(barrier.offset as _)
                                .size(barrier.size as _),
                        )
                    })
                    .collect::<Vec<_>>();
                let sync2_images =
                    Submission::barriers_require_sync2(None, None, &[], &image_barriers);
                let mut images = image_barriers
                    .iter()
                    .filter(|_| !sync2_images)
                    .map(|&barrier| {
                        let (src, dst, barrier) = TrackedImageBarrier::memory_barrier(barrier);
                        src_stages |= src;
                        dst_stages |= dst;
                        barrier
                    })
                    .collect::<Vec<_>>();
                if !buffers.is_empty() || !images.is_empty() {
                    Submission::with_legacy_barrier_batches(
                        src_stages,
                        dst_stages,
                        &[],
                        &mut buffers,
                        &mut images,
                        |src, dst, memory, buffers, images| {
                            #[cfg(test)]
                            test::INCOMING_BUFFER_BARRIERS.with_borrow_mut(|captured| {
                                if let Some(captured) = captured {
                                    captured.extend(
                                        buffers.iter().copied().map(|barrier| (src, dst, barrier)),
                                    );
                                }
                            });
                            unsafe {
                                cmd_buf.device.cmd_pipeline_barrier(
                                    cmd_buf.handle,
                                    src,
                                    dst,
                                    vk::DependencyFlags::empty(),
                                    memory,
                                    buffers,
                                    images,
                                );
                            }
                        },
                    );
                }
                if sync2_images {
                    Submission::pipeline_barrier_from_slices(
                        cmd_buf,
                        None,
                        None,
                        &[],
                        &image_barriers,
                    );
                }
            }

            // Bound retained snapshot storage even when no subsequent pass is recorded.
            tls.incoming_ranges.truncate(64);
            tls.incoming_ranges.shrink_to(64);
            for ranges in &mut tls.incoming_ranges {
                if ranges.capacity() > 1024 {
                    *ranges = Vec::new();
                }
            }

            if let Some(pending) = pending_buffer_transfer_nodes.as_mut() {
                pending.remove_where(|node_idx, _buffer, transfers| {
                    for BufferResourceBarrier { range, .. } in tls.pending_buffers.get(node_idx) {
                        if BufferQueueOwnershipTransfer::consume_pending(transfers, *range) {
                            return true;
                        }
                    }

                    false
                });

                if pending.is_empty() {
                    *pending_buffer_transfer_nodes = None;
                }
            }

            if let Some(pending) = pending_image_transfer_nodes.as_mut() {
                pending.remove_where(|node_idx, _image, transfers| {
                    for ImageResourceBarrier { range, .. } in tls.pending_images.get(node_idx) {
                        if ImageOwnershipTransfer::consume_pending(transfers, *range) {
                            return true;
                        }
                    }

                    false
                });

                if pending.is_empty() {
                    *pending_image_transfer_nodes = None;
                }
            }
        });
    }

    #[profiling::function]
    fn record_impl<P>(
        &mut self,
        pool: &mut P,
        cmd_buf: &CommandBuffer,
        ownership: &mut RecordingOwnership,
    ) -> Result<(), DriverError>
    where
        P: SubmissionPool,
    {
        if self.graph.cmds.is_empty() {
            return Ok(());
        }

        thread_local! {
            static SCHEDULE: RefCell<Schedule> = Default::default();
        }

        SCHEDULE.with_borrow_mut(|schedule| {
            schedule
                .access_index
                .update(&self.graph, self.graph.cmds.len());
            schedule.cmds.clear();
            schedule.cmds.extend(0..self.graph.cmds.len());

            self.record_scheduled_cmds(pool, cmd_buf, schedule, self.graph.cmds.len(), ownership)
        })
    }

    fn record_node<P>(
        &mut self,
        resource_pool: &mut P,
        cmd_buf: &CommandBuffer,
        node: AnyNode,
        ownership: &mut RecordingOwnership,
    ) -> Result<(), DriverError>
    where
        P: SubmissionPool,
    {
        match node {
            AnyNode::AccelerationStructure(node) => {
                self.record_resource_impl(resource_pool, cmd_buf, node, ownership)
            }
            AnyNode::Buffer(node) => {
                self.record_resource_impl(resource_pool, cmd_buf, node, ownership)
            }
            AnyNode::Image(node) => {
                self.record_resource_impl(resource_pool, cmd_buf, node, ownership)
            }
            AnyNode::Micromap(node) => {
                self.record_resource_impl(resource_pool, cmd_buf, node, ownership)
            }
        }
    }

    #[profiling::function]
    fn record_node_cmds<P>(
        &mut self,
        pool: &mut P,
        cmd_buf: &CommandBuffer,
        node_idx: usize,
        end_cmd_idx: usize,
        ownership: &mut RecordingOwnership,
    ) -> Result<(), DriverError>
    where
        P: SubmissionPool,
    {
        thread_local! {
            static SCHEDULE: RefCell<Schedule> = Default::default();
        }

        SCHEDULE.with_borrow_mut(|schedule| {
            schedule.access_index.update(&self.graph, end_cmd_idx);
            schedule.cmds.clear();

            self.schedule_node_cmds(node_idx, end_cmd_idx, schedule);
            self.record_scheduled_cmds(pool, cmd_buf, schedule, end_cmd_idx, ownership)
        })
    }

    pub(crate) fn record_prepared_command_stream(
        &mut self,
        cmd_buf: &CommandBuffer,
        resources: crate::ResourceMap,
        recording: &PreparedStreamRecording,
        values: &crate::stream::StreamValues,
    ) -> Result<(), DriverError> {
        let mut recording = recording
            .resources
            .lock()
            .expect("poisoned stream recording");
        std::mem::swap(&mut self.recorded_commands, &mut recording);
        let original_resources = std::mem::replace(&mut self.graph.resources, resources);

        let result = (|| {
            if self.recorded_commands.is_empty() {
                let schedule = (0..self.graph.cmds.len()).collect::<Vec<_>>();
                self.lease_scheduled_resources(&mut HashPool::new(&cmd_buf.device), &schedule)?;
            }

            self.record_prepared_command_stream_inner(cmd_buf, values)
        })();

        self.graph.resources = original_resources;
        if result.is_err() {
            self.recorded_commands.clear();
        }
        std::mem::swap(&mut self.recorded_commands, &mut recording);

        result
    }

    fn record_prepared_command_stream_inner(
        &mut self,
        cmd_buf: &CommandBuffer,
        values: &crate::stream::StreamValues,
    ) -> Result<(), DriverError> {
        let mut ownership = RecordingOwnership::default();

        thread_local! {
            static SCHEDULE: RefCell<Schedule> = Default::default();
        }

        SCHEDULE.with_borrow_mut(|schedule| {
            schedule
                .access_index
                .update(&self.graph, self.graph.cmds.len());
            schedule.cmds.clear();
            schedule.cmds.extend(0..self.graph.cmds.len());
            self.track_pending_transfers(
                schedule,
                cmd_buf.info.queue_family_index,
                &mut ownership,
                ResourceSetSynchronization::DeferredToOuterBoundary,
            );
        });

        self.record_cmd_indices(
            cmd_buf,
            0..self.graph.cmds.len(),
            ResourceSetSynchronization::DeferredToOuterBoundary,
            Some(values),
        )?;

        Ok(())
    }

    #[profiling::function]
    fn record_resource_dependencies_impl<P>(
        &mut self,
        pool: &mut P,
        cmd_buf: &CommandBuffer,
        resource_node: impl Node,
        ownership: &mut RecordingOwnership,
    ) -> Result<(), DriverError>
    where
        P: SubmissionPool,
    {
        self.graph.assert_node_owner(&resource_node);

        let node_idx = resource_node.index();

        debug_assert!(self.graph.resources.get(node_idx).is_some());

        // We record up to but not including the first command which accesses the target node.
        if let Some(end_pass_idx) = self.graph.first_node_access_pass_index(resource_node) {
            thread_local! {
                static SCHEDULE: RefCell<Schedule> = Default::default();
            }

            SCHEDULE.with_borrow_mut(|tls| {
                tls.access_index.update(&self.graph, end_pass_idx + 1);
                Schedule::schedule_dependency_cmds_before_target_access(
                    node_idx,
                    end_pass_idx,
                    tls,
                );
                self.record_scheduled_cmds(pool, cmd_buf, tls, end_pass_idx, ownership)
            })?;
        }

        Ok(())
    }

    #[profiling::function]
    fn record_resource_impl<P>(
        &mut self,
        pool: &mut P,
        cmd_buf: &CommandBuffer,
        resource_node: impl Node,
        ownership: &mut RecordingOwnership,
    ) -> Result<(), DriverError>
    where
        P: SubmissionPool,
    {
        self.graph.assert_node_owner(&resource_node);

        let node_idx = resource_node.index();

        debug_assert!(self.graph.resources.get(node_idx).is_some());

        if self.graph.cmds.is_empty() {
            return Ok(());
        }

        let end_pass_idx = self.graph.cmds.len();
        self.record_node_cmds(pool, cmd_buf, node_idx, end_pass_idx, ownership)
    }

    #[profiling::function]
    fn record_resource_set_acquisitions<'a>(
        cmd_buf: &CommandBuffer,
        resource_sets: &ResourceSetMap,
        accesses: impl IntoIterator<Item = &'a ResourceSetAccess>,
        acquired_resource_sets: &mut FixedBitSet,
        pending_transfers: &mut HashMap<PhysicalImageId, Vec<ImageOwnershipTransfer>>,
    ) {
        #[derive(Default)]
        struct ResourceSetBarrierScratch {
            image_accesses: Vec<(AccessType, vk::ImageSubresourceRange, bool)>,
            image_barriers: Vec<TrackedImageBarrier>,
            next_acceleration_structure_accesses: Vec<AccessType>,
            previous_acceleration_structure_accesses: Vec<AccessType>,
        }

        thread_local! {
            static RESOURCE_SET_BARRIER: RefCell<ResourceSetBarrierScratch> = Default::default();
        }

        RESOURCE_SET_BARRIER.with_borrow_mut(|tls| {
            tls.image_barriers.clear();
            tls.next_acceleration_structure_accesses.clear();
            tls.previous_acceleration_structure_accesses.clear();

            Self::for_each_first_resource_set_access(
                accesses,
                resource_sets.len(),
                acquired_resource_sets,
                |access| match resource_sets.get(access.resource_set_idx) {
                    ResourceSet::AccelerationStructure(resource_set) => {
                        let ResourceSetAccessType::AccelerationStructure(_) = access.access_type
                        else {
                            unreachable!("acceleration structure set access type mismatch")
                        };

                        #[cfg(feature = "checked")]
                        resource_set.assert_device(&cmd_buf.device);

                        let next_access = access.access_type.access_type();
                        if !resource_set.is_empty()
                            && !tls
                                .next_acceleration_structure_accesses
                                .contains(&next_access)
                        {
                            tls.next_acceleration_structure_accesses.push(next_access);
                        }

                        for member in resource_set.unique_members() {
                            for previous_access in AccelerationStructure::swap_accesses(
                                member.acceleration_structure(),
                                slice::from_ref(&next_access),
                            ) {
                                if !tls
                                    .previous_acceleration_structure_accesses
                                    .contains(&previous_access)
                                {
                                    tls.previous_acceleration_structure_accesses
                                        .push(previous_access);
                                }
                            }
                        }
                    }
                    ResourceSet::Image(resource_set) => {
                        let ResourceSetAccessType::Image(_) = access.access_type else {
                            unreachable!("image set access type mismatch")
                        };

                        #[cfg(feature = "checked")]
                        resource_set.assert_device(&cmd_buf.device);

                        let next_access = access.access_type.access_type();

                        for member in resource_set.unique_members() {
                            let image = member.image();
                            let range = member.subresource();

                            if pending_transfers.is_empty() {
                                for (previous_accesses, range) in
                                    Image::swap_access(image, next_access, range)
                                {
                                    let barrier = TrackedImageBarrier::new(
                                        image.handle,
                                        previous_accesses,
                                        next_access,
                                        range,
                                        None,
                                        TrackedImageBarrier::execution_discard_contents(
                                            previous_accesses,
                                        ),
                                    );
                                    if !TrackedImageBarrier::can_elide_sampled_read(barrier) {
                                        tls.image_barriers.push(barrier);
                                    }
                                }

                                continue;
                            }

                            let image_id = PhysicalImageId::of(image);
                            let transfers = pending_transfers
                                .get(&image_id)
                                .map(Vec::as_slice)
                                .unwrap_or_default();

                            let mut image_accesses = take(&mut tls.image_accesses);
                            image_accesses.clear();
                            image_accesses.extend(
                                ImageOwnershipTransfer::barrier_ranges(transfers, range).map(
                                    |(range, transfer)| (next_access, range, transfer.is_none()),
                                ),
                            );

                            for (next_access, previous_accesses, range) in
                                Image::swap_accesses(image, image_accesses.iter().copied())
                            {
                                tls.image_barriers.extend(
                                    TrackedImageBarrier::from_transfers(
                                        image.handle,
                                        previous_accesses,
                                        next_access,
                                        range,
                                        transfers,
                                        TrackedImageBarrier::execution_discard_contents(
                                            previous_accesses,
                                        ),
                                    )
                                    .filter(|barrier| {
                                        !TrackedImageBarrier::can_elide_sampled_read(*barrier)
                                    }),
                                );
                            }

                            tls.image_accesses = image_accesses;

                            let remove_transfers = pending_transfers
                                .get_mut(&image_id)
                                .is_some_and(|transfers| {
                                    ImageOwnershipTransfer::consume_pending(transfers, range)
                                });
                            if remove_transfers {
                                pending_transfers.remove(&image_id);
                            }
                        }
                    }
                },
            );

            let global_barrier = if tls
                .previous_acceleration_structure_accesses
                .iter()
                .copied()
                .any(is_write_access)
            {
                Some(GlobalBarrier {
                    previous_accesses: &tls.previous_acceleration_structure_accesses,
                    next_accesses: &tls.next_acceleration_structure_accesses,
                })
            } else {
                None
            };

            Submission::pipeline_barrier_from_slices(
                cmd_buf,
                global_barrier,
                None,
                &[],
                &tls.image_barriers,
            );
        });
    }

    #[profiling::function]
    fn record_scheduled_cmds<P>(
        &mut self,
        pool: &mut P,
        cmd_buf: &CommandBuffer,
        schedule: &mut Schedule,
        end_cmd_idx: usize,
        ownership: &mut RecordingOwnership,
    ) -> Result<(), DriverError>
    where
        P: SubmissionPool,
    {
        if schedule.cmds.is_empty() {
            return Ok(());
        }

        // // Print some handy details or hit a breakpoint if you set the flag
        // if log_enabled!(Debug) && self.graph.debug {
        //     debug!("resolving the following graph:\n\n{:#?}\n\n", self.graph);
        // }

        debug_assert!(
            schedule.cmds.windows(2).all(|w| w[0] <= w[1]),
            "Unsorted schedule"
        );

        // Optimize the schedule; requesting the required resources it needs
        schedule.reorder_cmds(end_cmd_idx);
        self.merge_scheduled_cmds(&mut schedule.cmds);
        self.lease_scheduled_resources(pool, &schedule.cmds)?;
        self.track_pending_transfers(
            schedule,
            cmd_buf.info.queue_family_index,
            ownership,
            ResourceSetSynchronization::Enabled,
        );

        let has_pending_timestamp_queries = self
            .graph
            .timestamp_queries
            .as_ref()
            .is_some_and(|timestamp_queries| timestamp_queries.iter().any(Option::is_some));
        let include_final_timestamp_queries = schedule.cmds.len() == self.graph.cmds.len();

        if has_pending_timestamp_queries {
            if cmd_buf
                .device
                .physical
                .queue_families
                .get(cmd_buf.info.queue_family_index as usize)
                .is_none_or(|queue_family| {
                    !Self::queue_family_supports_timestamp_queries(queue_family)
                })
            {
                self.graph.timestamp_queries = None;
            } else {
                if self.query_pool_results.is_none() {
                    self.prepare_timestamp_query_results(cmd_buf)?;
                }

                self.prepare_timestamp_queries_for_commands(
                    &schedule.cmds,
                    include_final_timestamp_queries,
                );

                if !self.query_pool_reset {
                    let query_pool_results = self
                        .query_pool_results
                        .as_ref()
                        .expect("missing query pool results");
                    query_pool_results.reset(cmd_buf);
                    query_pool_results.write_epoch(cmd_buf);
                    self.query_pool_reset = true;
                }
            }
        }

        self.record_cmd_indices(
            cmd_buf,
            schedule.cmds.iter().copied(),
            ResourceSetSynchronization::Enabled,
            None,
        )?;

        if include_final_timestamp_queries
            && let Some(timestamp_queries) =
                self.take_timestamp_queries_for_command(self.graph.cmds.len())
        {
            let query_pool = self
                .query_pool_results
                .as_ref()
                .map(SubmittedTimestampQueries::query_pool);

            Self::write_timestamp_queries(
                cmd_buf,
                query_pool,
                &timestamp_queries,
                TimestampQueryPlacement::BeforeExec,
                0,
                0,
            );
        }

        self.remap_timestamp_queries_after_removing_scheduled(&schedule.cmds);

        thread_local! {
            static PASSES: RefCell<Vec<CommandData>> = Default::default();
        }

        PASSES.with_borrow_mut(|passes| {
            debug_assert!(passes.is_empty());

            // We have to keep the bindings and pipelines alive until the gpu is done
            schedule.cmds.sort_unstable();
            while let Some(schedule_idx) = schedule.cmds.pop() {
                debug_assert!(!self.graph.cmds.is_empty());

                while let Some(cmd) = self.graph.cmds.pop() {
                    let cmd_idx = self.graph.cmds.len();

                    if cmd_idx == schedule_idx {
                        // This was a scheduled cmd - store it!

                        self.submit_retained.push(SubmittedCommand {
                            cmd,
                            _resources: self
                                .recorded_commands
                                .pop()
                                .expect("missing recorded command"),
                        });
                        break;
                    } else {
                        debug_assert!(cmd_idx > schedule_idx);

                        passes.push(cmd);
                    }
                }
            }

            debug_assert!(self.recorded_commands.is_empty());

            // Put the other passes back for future resolves
            self.graph.cmds.extend(passes.drain(..).rev());
        });

        log::trace!("Recorded passes");

        Ok(())
    }

    #[profiling::function]
    fn record_selection_impl<'a, P>(
        &mut self,
        resource_pool: &mut P,
        cmd_buf: &CommandBuffer,
        selection: RecordSelection<'a>,
        ownership: &mut RecordingOwnership,
    ) -> Result<(), DriverError>
    where
        P: SubmissionPool,
    {
        let _ = CommandBufferDebugLabel::begin(cmd_buf, "graph submission");

        match selection {
            RecordSelection::All => self.record_impl(resource_pool, cmd_buf, ownership),
            RecordSelection::Dependencies(node) => match node {
                AnyNode::AccelerationStructure(node) => {
                    self.record_resource_dependencies_impl(resource_pool, cmd_buf, node, ownership)
                }
                AnyNode::Buffer(node) => {
                    self.record_resource_dependencies_impl(resource_pool, cmd_buf, node, ownership)
                }
                AnyNode::Image(node) => {
                    self.record_resource_dependencies_impl(resource_pool, cmd_buf, node, ownership)
                }
                AnyNode::Micromap(node) => {
                    self.record_resource_dependencies_impl(resource_pool, cmd_buf, node, ownership)
                }
            },
            RecordSelection::Node(node) => {
                self.record_node(resource_pool, cmd_buf, node, ownership)
            }
            RecordSelection::Nodes(nodes) => {
                for &node in nodes {
                    self.record_node(resource_pool, cmd_buf, node, ownership)?;
                }

                Ok(())
            }
        }
    }

    fn remap_timestamp_queries_after_removing_scheduled(&mut self, schedule: &[usize]) {
        let old_cmd_len = self.graph.cmds.len();
        let mut scheduled = FixedBitSet::with_capacity(old_cmd_len);
        for cmd_idx in schedule.iter().copied() {
            scheduled.insert(cmd_idx);
        }

        let mut old_to_new_cmd_idx = vec![0; old_cmd_len + 1];
        let mut new_cmd_idx = 0;
        for (old_cmd_idx, new_idx) in old_to_new_cmd_idx.iter_mut().enumerate().take(old_cmd_len) {
            *new_idx = new_cmd_idx;
            if !scheduled.contains(old_cmd_idx) {
                new_cmd_idx += 1;
            }
        }

        old_to_new_cmd_idx[old_cmd_len] = new_cmd_idx;

        if let Some(timestamp_queries) = &mut self.graph.timestamp_queries {
            for query in timestamp_queries.iter_mut().flatten() {
                query.command_idx = old_to_new_cmd_idx[query.command_idx];
            }
        }
    }

    #[profiling::function]
    fn render_extent(bindings: &[AnyResource], pass: &CommandData) -> vk::Extent2D {
        // set_render_area was not specified so we're going to guess using the minimum common
        // attachment extents
        let first_exec = pass.expect_first_exec();

        // We must be able to find the render area because render passes require at least one
        // image to be attached
        let (mut width, mut height) = (u32::MAX, u32::MAX);
        for (attachment_width, attachment_height) in first_exec
            .attachments
            .color_attachments()
            .map(|(_, state)| state.attachment)
            .chain(
                first_exec
                    .attachments
                    .depth_stencil_attachment()
                    .into_iter()
                    .filter(|state| state.is_attachment)
                    .map(|state| state.attachment),
            )
            .map(|attachment| {
                let info = Self::expect_attachment_image(bindings, &attachment).info;

                (
                    info.width >> attachment.base_mip_level,
                    info.height >> attachment.base_mip_level,
                )
            })
        {
            width = width.min(attachment_width);
            height = height.min(attachment_height);
        }

        vk::Extent2D { height, width }
    }

    /// Returns a borrow of the resource or persistent resource set represented by the given node.
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: ResourceNode,
    {
        self.graph.resource(resource_node)
    }

    fn resource_set_access_is_acquired(
        access: ResourceSetAccess,
        resource_set_count: usize,
        acquired_resource_sets: &FixedBitSet,
    ) -> bool {
        acquired_resource_sets.contains(access.acquisition_index(resource_set_count))
    }

    /// Mutates a schedule of command indices that are required to be executed, in order, for the
    /// given node.
    #[profiling::function]
    fn schedule_node_cmds(&self, node_idx: usize, end_cmd_idx: usize, schedule: &mut Schedule) {
        trace!("scheduling node {node_idx}");
        schedule.schedule_required_node_prefixes([(node_idx, end_cmd_idx)]);

        if log_enabled!(Debug) {
            if !schedule.cmds.is_empty() {
                debug!(
                    "schedule: {}",
                    schedule
                        .cmds
                        .iter()
                        .copied()
                        .map(|idx| format!("[{}: {}]", idx, self.graph.cmds[idx].name()))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            if log_enabled!(Trace) {
                let unscheduled = (0..end_cmd_idx)
                    .filter(|&cmd_idx| !schedule.node_schedule.selected_cmds.contains(cmd_idx))
                    .collect::<Box<_>>();

                if !unscheduled.is_empty() {
                    trace!(
                        "delaying: {}",
                        unscheduled
                            .iter()
                            .copied()
                            .map(|idx| format!("[{}: {}]", idx, self.graph.cmds[idx].name()))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }

                if end_cmd_idx < self.graph.cmds.len() {
                    trace!(
                        "ignoring: {}",
                        self.graph.cmds[end_cmd_idx..]
                            .iter()
                            .enumerate()
                            .map(|(idx, cmd)| {
                                format!("[{}: {}]", idx + end_cmd_idx, cmd.name())
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }
        }
    }

    fn set_scissor(cmd_buf: &CommandBuffer, x: i32, y: i32, width: u32, height: u32) {
        unsafe {
            cmd_buf.device.cmd_set_scissor(
                cmd_buf.handle,
                0,
                slice::from_ref(&vk::Rect2D {
                    extent: vk::Extent2D { width, height },
                    offset: vk::Offset2D { x, y },
                }),
            );
        }
    }

    fn set_viewport(
        cmd_buf: &CommandBuffer,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        depth: Range<f32>,
    ) {
        unsafe {
            cmd_buf.device.cmd_set_viewport(
                cmd_buf.handle,
                0,
                slice::from_ref(&vk::Viewport {
                    x,
                    y,
                    width,
                    height,
                    min_depth: depth.start,
                    max_depth: depth.end,
                }),
            );
        }
    }

    fn signal_executed(&self) {
        for command in &self.submit_retained {
            command.signal_executed();
        }
    }

    fn subpass_stage_mask(stages: vk::PipelineStageFlags) -> vk::PipelineStageFlags {
        if stages.is_empty() {
            return stages;
        }

        if stages
            .intersects(vk::PipelineStageFlags::ALL_GRAPHICS | vk::PipelineStageFlags::ALL_COMMANDS)
        {
            return vk::PipelineStageFlags::ALL_GRAPHICS;
        }

        let graphics_stages = stages & Self::GRAPHICS_STAGES;
        if graphics_stages.is_empty() {
            vk::PipelineStageFlags::ALL_GRAPHICS
        } else {
            graphics_stages
        }
    }

    pub(crate) fn take_prepared_stream_recording(&mut self) -> PreparedStreamRecording {
        PreparedStreamRecording {
            resources: Mutex::new(take(&mut self.recorded_commands)),
        }
    }

    fn take_timestamp_queries_for_command(
        &mut self,
        command_idx: usize,
    ) -> Option<Box<[TimestampQueryData]>> {
        let Some(graph_timestamp_queries) = &mut self.graph.timestamp_queries else {
            return None;
        };

        let mut timestamp_queries = Vec::new();

        for timestamp_query in graph_timestamp_queries {
            if timestamp_query
                .as_ref()
                .is_some_and(|timestamp_query| timestamp_query.command_idx == command_idx)
            {
                timestamp_queries.push(
                    timestamp_query
                        .take()
                        .expect("missing timestamp query after command match"),
                );
            }
        }

        timestamp_queries.sort_unstable_by_key(|timestamp_query| {
            (
                timestamp_query.exec_idx,
                timestamp_query.placement,
                timestamp_query.query.index(),
            )
        });

        (!timestamp_queries.is_empty()).then(|| timestamp_queries.into_boxed_slice())
    }

    fn timestamp_query_pool_query_count(
        cmds: &[CommandData],
        timestamp_query: &TimestampQueryData,
    ) -> u32 {
        if matches!(
            timestamp_query.placement,
            TimestampQueryPlacement::BeforeExec
        ) && timestamp_query.exec_idx == 0
        {
            return 1;
        }

        cmds.get(timestamp_query.command_idx)
            .and_then(|cmd| cmd.execs.get(timestamp_query.exec_idx))
            .map(|exec| exec.view_mask.count_ones().max(1))
            .unwrap_or(1)
    }

    #[profiling::function]
    fn track_pending_image_set_transfers(
        &mut self,
        resource_set_idx: ResourceSetIndex,
        queue_family_index: u32,
        ownership: &mut RecordingOwnership,
    ) {
        let Some(resource_set) = self
            .graph
            .resource_sets
            .get_image(resource_set_idx)
            .cloned()
        else {
            return;
        };
        let next_access = ImageAccessType::SampledRead.access_type();
        let exclusive_image_count = resource_set.exclusive_physical_image_count();

        if exclusive_image_count == 0 {
            return;
        }

        self.touched_image_sets.insert(resource_set_idx.as_usize());
        if resource_set
            .queue()
            .is_some_and(|(family, _)| family == queue_family_index)
        {
            return;
        }

        for member in resource_set.unique_members() {
            let image = member.image();
            if image.info.sharing_mode == vk::SharingMode::CONCURRENT {
                continue;
            }

            if ownership.image_set_images.is_empty() {
                ownership.image_set_images.reserve(exclusive_image_count);
            }
            let image_id = PhysicalImageId::of(image);
            let unclaimed =
                ownership.claim_image_set_image(image_id, image.info, member.subresource());
            if unclaimed.is_empty() {
                continue;
            }

            for access_range in unclaimed {
                for (subresource, sharing) in image.sync_info_with_sharing_range(access_range) {
                    let Some(range) =
                        image_subresource_range_intersection(subresource.range, access_range)
                    else {
                        continue;
                    };
                    let Some((src_queue_family_index, src_queue_index)) =
                        RecordingOwnership::exclusive_transfer_source(sharing, queue_family_index)
                    else {
                        continue;
                    };
                    let layouts = ImageOwnershipLayouts::new(
                        subresource.layout,
                        next_access,
                        subresource.layout.is_none(),
                    );
                    let transfer = ImageOwnershipTransfer {
                        src_queue_family_index,
                        src_queue_index,
                        dst_queue_family_index: queue_family_index,
                        layouts,
                        range,
                    };

                    QueueOwnershipReleaseGroup::get_or_insert(
                        &mut self.queue_ownership_release_groups,
                        src_queue_family_index,
                        src_queue_index,
                    )
                    .images
                    .push(ImageQueueOwnershipRelease {
                        image: image.handle,
                        layouts,
                        range,
                    });
                    self.pending_image_set_transfers
                        .entry(image_id)
                        .or_default()
                        .push(transfer);
                }
            }
        }
    }

    fn track_pending_transfers(
        &mut self,
        schedule: &Schedule,
        queue_family_index: u32,
        ownership: &mut RecordingOwnership,
        resource_set_synchronization: ResourceSetSynchronization,
    ) {
        let resource_count = self.graph.resources.len();
        let mut seen_resource_sets = FixedBitSet::with_capacity(self.graph.resource_sets.len());

        for cmd_idx in schedule.cmds.iter().copied() {
            let cmd = &self.graph.cmds[cmd_idx];
            let resource_set_indices =
                if resource_set_synchronization == ResourceSetSynchronization::Enabled {
                    cmd.execs
                        .iter()
                        .flat_map(|exec| &exec.resource_set_accesses)
                        .map(|access| access.resource_set_idx)
                        .collect::<SmallVec<[_; 2]>>()
                } else {
                    SmallVec::new()
                };
            let is_graphics = cmd
                .execs
                .first()
                .and_then(|exec| exec.pipeline.as_ref())
                .is_some_and(|pipeline| pipeline.is_graphics());

            for (node_idx, accesses) in cmd.execs.iter().flat_map(|exec| exec.accesses.iter()) {
                if let Some(buffer) = self.graph.resources[node_idx].as_buffer() {
                    if buffer.info.sharing_mode == vk::SharingMode::CONCURRENT {
                        continue;
                    }

                    for access in accesses.iter() {
                        let SubresourceRange::Buffer(access_range) = access.subresource else {
                            continue;
                        };
                        let unclaimed = ownership
                            .claim_buffer(node_idx, access_range.resolve_whole(buffer.info.size));

                        self.exclusive_buffer_ranges
                            .entry(node_idx)
                            .or_default()
                            .extend(unclaimed.iter().copied());

                        for access_range in unclaimed {
                            for (subresource, sharing) in
                                buffer.sync_info_with_sharing_range(access_range)
                            {
                                let Some(range) = subresource.range.intersection(access_range)
                                else {
                                    continue;
                                };
                                let Some((src_queue_family_index, src_queue_index)) =
                                    RecordingOwnership::exclusive_transfer_source(
                                        sharing,
                                        queue_family_index,
                                    )
                                else {
                                    continue;
                                };
                                let transfer = BufferQueueOwnershipTransfer {
                                    src_queue_family_index,
                                    dst_queue_family_index: queue_family_index,
                                    range,
                                };

                                QueueOwnershipReleaseGroup::get_or_insert(
                                    &mut self.queue_ownership_release_groups,
                                    src_queue_family_index,
                                    src_queue_index,
                                )
                                .buffers
                                .push((buffer.handle, range));
                                self.pending_buffer_transfer_nodes
                                    .get_or_insert_with(|| {
                                        PendingTransferNodes::new(resource_count)
                                    })
                                    .push_transfer(node_idx, buffer.handle, transfer);
                            }
                        }
                    }

                    continue;
                }

                let Some(image) = self.graph.resources[node_idx].as_image() else {
                    continue;
                };
                if image.info.sharing_mode == vk::SharingMode::CONCURRENT {
                    continue;
                }

                for access in accesses.iter() {
                    let SubresourceRange::Image(access_range) = access.subresource else {
                        continue;
                    };
                    let unclaimed = ownership.claim_image(
                        node_idx,
                        image.info,
                        image.info.resolve_subresource_counts(access_range),
                    );

                    self.exclusive_image_ranges
                        .entry(node_idx)
                        .or_default()
                        .extend(unclaimed.iter().copied());

                    for access_range in unclaimed {
                        for (subresource, sharing) in
                            image.sync_info_with_sharing_range(access_range)
                        {
                            let Some(range) = image_subresource_range_intersection(
                                subresource.range,
                                access_range,
                            ) else {
                                continue;
                            };
                            let Some((src_queue_family_index, src_queue_index)) =
                                RecordingOwnership::exclusive_transfer_source(
                                    sharing,
                                    queue_family_index,
                                )
                            else {
                                continue;
                            };
                            let next_access = if is_graphics {
                                initial_image_layout_access(access.access)
                            } else {
                                access.access
                            };
                            let discard_contents = subresource.layout.is_none()
                                || is_graphics && !is_read_access(next_access);
                            let layouts = ImageOwnershipLayouts::new(
                                subresource.layout,
                                next_access,
                                discard_contents,
                            );
                            let transfer = ImageOwnershipTransfer {
                                src_queue_family_index,
                                src_queue_index,
                                dst_queue_family_index: queue_family_index,
                                layouts,
                                range,
                            };

                            QueueOwnershipReleaseGroup::get_or_insert(
                                &mut self.queue_ownership_release_groups,
                                src_queue_family_index,
                                src_queue_index,
                            )
                            .images
                            .push(ImageQueueOwnershipRelease {
                                image: image.handle,
                                layouts,
                                range,
                            });
                            self.pending_image_transfer_nodes
                                .get_or_insert_with(|| PendingTransferNodes::new(resource_count))
                                .push_transfer(node_idx, image.handle, transfer);
                        }
                    }
                }
            }

            for resource_set_idx in resource_set_indices {
                if seen_resource_sets.put(resource_set_idx.as_usize()) {
                    continue;
                }

                self.track_pending_image_set_transfers(
                    resource_set_idx,
                    queue_family_index,
                    ownership,
                );
            }
        }
    }

    fn valid_exec_subpasses(exec_count: usize, exec_subpasses: &[u32]) -> bool {
        exec_subpasses.len() == exec_count
            && (exec_count == 0 || exec_subpasses.first() == Some(&0))
            && exec_subpasses
                .windows(2)
                .all(|pair| pair[1].checked_sub(pair[0]).is_some_and(|delta| delta <= 1))
    }

    fn visit_subpass_scopes<'a>(
        exec: &'a crate::Execution,
        mut visit: impl FnMut(NodeIndex, PipelineStageAccessFlags, SubpassAccessOrigin<'a>),
    ) {
        for (node_idx, accesses) in exec.accesses.iter() {
            for access in accesses {
                let mut scope = PipelineStageAccessFlags::new(access.access);
                scope.stage_flags = Self::subpass_stage_mask(scope.stage_flags);
                // The general planner also needs declarations excluded from subpass stages
                // to veto locality/pruning proofs. Consumers decide whether to retain masks.
                visit(node_idx, scope, SubpassAccessOrigin::Explicit(access));
            }
        }

        // Attachment operations also contribute when not present in the explicit access list.
        for (_, state) in exec.attachments.color_attachments() {
            let mut scope = PipelineStageAccessFlags::default();
            if state.is_input {
                scope.union(PipelineStageAccessFlags::new(
                    AccessType::FragmentShaderReadColorInputAttachment,
                ));
            } else if state.is_attachment && Self::color_attachment_is_read(state.load) {
                scope.union(PipelineStageAccessFlags::new(
                    AccessType::ColorAttachmentRead,
                ));
            }
            // DONT_CARE stores still write, including the last use of an input-only attachment.
            if state.is_attachment || state.is_input || state.resolve.is_some() {
                scope.union(PipelineStageAccessFlags::new(
                    AccessType::ColorAttachmentWrite,
                ));
            }
            if !scope.stage_flags.is_empty() {
                visit(
                    state.attachment.target,
                    scope,
                    SubpassAccessOrigin::Attachment(&state.attachment),
                );
            }
        }
        if let Some(state) = exec.attachments.depth_stencil_attachment() {
            let mut scope = PipelineStageAccessFlags::default();
            let (read, write) = Self::attachment_read_write_access(state.attachment.aspect_mask);
            if state.is_attachment && Self::depth_stencil_attachment_is_read(state.load) {
                scope.stage_flags |= Self::attachment_read_stage(state.attachment.aspect_mask);
                scope.access_flags |= read;
            }
            if Self::depth_stencil_attachment_is_write(
                state.load,
                state.store,
                state.resolve.is_some(),
            ) || state.is_attachment
            {
                scope.stage_flags |= Self::attachment_stage(state.attachment.aspect_mask);
                scope.access_flags |= write;
            }
            if !scope.stage_flags.is_empty() {
                visit(
                    state.attachment.target,
                    scope,
                    SubpassAccessOrigin::Attachment(&state.attachment),
                );
            }
            if let Some(resolve) = &state.resolve {
                visit(
                    state.attachment.target,
                    PipelineStageAccessFlags::new(AccessType::ColorAttachmentRead),
                    SubpassAccessOrigin::Attachment(&state.attachment),
                );
                // Fixed-function depth/stencil resolves execute in color attachment output.
                visit(
                    resolve.attachment.target,
                    PipelineStageAccessFlags::new(AccessType::ColorAttachmentReadWrite),
                    SubpassAccessOrigin::Attachment(&resolve.attachment),
                );
            }
        }
    }

    fn whole_resource_canonical_accesses<'a>(
        accesses: &'a [SubresourceAccess],
        scratch: &'a mut Vec<AccessType>,
    ) -> &'a [AccessType] {
        scratch.clear();

        let [access] = accesses else {
            for access in accesses {
                if !scratch.contains(&access.access) {
                    scratch.push(access.access);
                }
            }

            return scratch.as_slice();
        };

        slice::from_ref(&access.access)
    }

    fn with_legacy_barrier_batches<'a>(
        src: vk::PipelineStageFlags,
        dst: vk::PipelineStageFlags,
        memory: &[vk::MemoryBarrier<'a>],
        buffers: &mut [vk::BufferMemoryBarrier<'a>],
        images: &mut [vk::ImageMemoryBarrier<'a>],
        mut record: impl FnMut(
            vk::PipelineStageFlags,
            vk::PipelineStageFlags,
            &[vk::MemoryBarrier<'a>],
            &[vk::BufferMemoryBarrier<'a>],
            &[vk::ImageMemoryBarrier<'a>],
        ),
    ) {
        if !(src | dst).contains(vk::PipelineStageFlags::HOST)
            || (!buffers
                .iter()
                .any(|barrier| barrier.src_queue_family_index != barrier.dst_queue_family_index)
                && !images.iter().any(|barrier| {
                    barrier.src_queue_family_index != barrier.dst_queue_family_index
                }))
        {
            record(src, dst, memory, buffers, images);
            return;
        }

        // Legacy stage masks apply to the entire batch. HOST cannot accompany any
        // ownership transfer, even when the host access belongs to another resource.
        buffers.sort_unstable_by_key(|barrier| {
            barrier.src_queue_family_index != barrier.dst_queue_family_index
        });
        images.sort_unstable_by_key(|barrier| {
            barrier.src_queue_family_index != barrier.dst_queue_family_index
        });
        let (local_buffers, acquired_buffers) =
            buffers.split_at(buffers.partition_point(|barrier| {
                barrier.src_queue_family_index == barrier.dst_queue_family_index
            }));
        let (local_images, acquired_images) = images.split_at(images.partition_point(|barrier| {
            barrier.src_queue_family_index == barrier.dst_queue_family_index
        }));
        if !memory.is_empty() || !local_buffers.is_empty() || !local_images.is_empty() {
            record(src, dst, memory, local_buffers, local_images);
        }
        record(
            (src & !vk::PipelineStageFlags::HOST) | vk::PipelineStageFlags::TOP_OF_PIPE,
            (dst & !vk::PipelineStageFlags::HOST) | vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            &[],
            acquired_buffers,
            acquired_images,
        );
    }

    #[profiling::function]
    fn write_descriptor_sets(
        cmd_buf: &CommandBuffer,
        bindings: &[AnyResource],
        pass: &CommandData,
        recorded_command: &CommandRecordingResources,
    ) -> Result<(), DriverError> {
        #[derive(Default)]
        struct DescriptorScratch<'a> {
            accel_struct_handles: Vec<vk::AccelerationStructureKHR>,
            accel_struct_infos: Vec<vk::WriteDescriptorSetAccelerationStructureKHR<'a>>,
            accel_struct_writes: Vec<IndexedWrite<'static>>,
            buffer_infos: Vec<vk::DescriptorBufferInfo>,
            buffer_writes: Vec<IndexedWrite<'a>>,
            descriptors: Vec<vk::WriteDescriptorSet<'a>>,
            image_infos: Vec<vk::DescriptorImageInfo>,
            image_writes: Vec<IndexedWrite<'a>>,
        }

        #[derive(Clone, Copy)]
        struct IndexedWrite<'a> {
            info_idx: usize,
            write: vk::WriteDescriptorSet<'a>,
        }

        thread_local! {
            static DESCRIPTOR: RefCell<DescriptorScratch<'static>> = Default::default();
        }

        DESCRIPTOR.with_borrow_mut(|tls| {
            tls.accel_struct_handles.clear();
            tls.accel_struct_infos.clear();
            tls.accel_struct_writes.clear();
            tls.buffer_infos.clear();
            tls.buffer_writes.clear();
            tls.descriptors.clear();
            tls.image_infos.clear();
            tls.image_writes.clear();

            for (exec_idx, exec, pipeline) in pass
            .execs
            .iter()
            .enumerate()
            .filter_map(|(exec_idx, exec)| {
                exec.pipeline
                    .as_ref()
                    .map(|pipeline| (exec_idx, exec, pipeline))
            })
            .filter(|(.., pipeline)| !pipeline.descriptor_info().layouts.is_empty())
        {
            let descriptor_sets = &recorded_command.descriptor_sets[exec_idx];

            // Write the manually bound things (access, read, and write functions)
            for (descriptor, (node_idx, view_info)) in exec.bindings.iter() {
                let (descriptor_set_idx, dst_binding, binding_offset) = descriptor.into_tuple();
                let Some((descriptor_info, _)) = pipeline.descriptor_bindings().get(&Descriptor {
                    set: descriptor_set_idx,
                    binding: dst_binding,
                }) else {
                    warn!(
                        "binding {}.{}[{}] not found in shader reflection for command \"{}\"",
                        descriptor_set_idx,
                        dst_binding,
                        binding_offset,
                        pass.name(),
                    );
                    return Err(DriverError::InvalidData);
                };
                if exec.descriptor_sets.contains_key(&descriptor_set_idx) {
                    continue;
                }

                let descriptor_type = descriptor_info.descriptor_type();
                let bound_node = &bindings[*node_idx];
                if let Some(image) = bound_node.as_image() {
                    let mut image_view_info = *view_info.expect_image();

                    // Handle default views which did not specify a particular aspect
                    if image_view_info.aspect_mask.is_empty() {
                        image_view_info.aspect_mask = format_aspect_mask(image.info.format);
                    }

                    let image_view = Image::view(image, image_view_info)?;
                    let image_layout = match descriptor_type {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                        | vk::DescriptorType::SAMPLED_IMAGE => {
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        }
                        vk::DescriptorType::STORAGE_IMAGE => vk::ImageLayout::GENERAL,
                        _ => {
                            warn!(
                                "invalid image descriptor type at binding {}.{}[{}] in command \"{}\"",
                                descriptor_set_idx,
                                dst_binding,
                                binding_offset,
                                pass.name()
                            );

                            return Err(DriverError::InvalidData);
                        }
                    };

                    if binding_offset == 0 {
                        tls.image_writes.push(IndexedWrite {
                            info_idx: tls.image_infos.len(),
                            write: vk::WriteDescriptorSet {
                                dst_set: descriptor_sets[descriptor_set_idx as usize].handle(),
                                dst_binding,
                                descriptor_type,
                                descriptor_count: 1,
                                ..Default::default()
                            },
                        });
                    } else {
                        tls.image_writes
                            .last_mut()
                            .expect("missing image descriptor write")
                            .write
                            .descriptor_count += 1;
                    }

                    tls.image_infos.push(
                        vk::DescriptorImageInfo::default()
                            .image_layout(image_layout)
                            .image_view(image_view),
                    );
                } else if let Some(buffer) = bound_node.as_buffer() {
                    let buffer_view_info = view_info.expect_buffer();

                    if binding_offset == 0 {
                        tls.buffer_writes.push(IndexedWrite {
                            info_idx: tls.buffer_infos.len(),
                            write: vk::WriteDescriptorSet {
                                dst_set: descriptor_sets[descriptor_set_idx as usize].handle(),
                                dst_binding,
                                descriptor_type,
                                descriptor_count: 1,
                                ..Default::default()
                            },
                        });
                    } else {
                        tls.buffer_writes
                            .last_mut()
                            .expect("missing buffer descriptor write")
                            .write
                            .descriptor_count += 1;
                    }

                    tls.buffer_infos.push(
                        vk::DescriptorBufferInfo::default()
                            .buffer(buffer.handle)
                            .offset(buffer_view_info.start)
                            .range(buffer_view_info.end - buffer_view_info.start),
                    );
                } else if let Some(accel_struct) = bound_node.as_accel_struct() {
                    if binding_offset == 0 {
                        tls.accel_struct_writes.push(IndexedWrite {
                            info_idx: tls.accel_struct_handles.len(),
                            write: vk::WriteDescriptorSet::default()
                                .dst_set(descriptor_sets[descriptor_set_idx as usize].handle())
                                .dst_binding(dst_binding)
                                .descriptor_type(descriptor_type)
                                .descriptor_count(1),
                        });
                    } else {
                        tls
                            .accel_struct_writes
                            .last_mut()
                            .expect("missing acceleration structure descriptor write")
                            .write
                            .descriptor_count += 1;
                    }

                    tls.accel_struct_handles.push(accel_struct.handle);
                } else {
                    warn!(
                        "invalid bound resource kind at descriptor {}.{}[{}] in command \"{}\"",
                        descriptor_set_idx,
                        dst_binding,
                        binding_offset,
                        pass.name()
                    );

                    return Err(DriverError::InvalidData);
                }
            }

            if let ExecutionPipeline::Graphics(pipeline) = pipeline {
                // Write graphics render pass input attachments (they're automatic)
                if recorded_command.exec_subpasses[exec_idx] > 0 {
                    for (
                        &Descriptor {
                            set: descriptor_set_idx,
                            binding: dst_binding,
                        },
                        (descriptor_info, _),
                    ) in &pipeline.inner.descriptor_bindings
                    {
                        if exec.descriptor_sets.contains_key(&descriptor_set_idx) {
                            continue;
                        }

                        if let DescriptorInfo::InputAttachment(_, attachment_idx) = *descriptor_info
                        {
                            let (attachment, image_layout) = Self::input_attachment_descriptor(
                                pass,
                                &recorded_command.exec_subpasses,
                                &recorded_command.render_pass.as_ref().expect("missing render pass").info,
                                exec_idx,
                                attachment_idx,
                            );
                            let image_binding = &bindings[attachment.target];
                            let image = image_binding.expect_image();
                            let image_view =
                                Image::view(image, attachment.image_view_info(image.info))?;

                            tls.image_writes.push(IndexedWrite {
                                info_idx: tls.image_infos.len(),
                                write: vk::WriteDescriptorSet {
                                    dst_set: descriptor_sets[descriptor_set_idx as usize].handle(),
                                    dst_binding,
                                    descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                                    descriptor_count: 1,
                                    ..Default::default()
                                },
                            });

                            tls.image_infos.push(vk::DescriptorImageInfo {
                                image_layout,
                                image_view,
                                sampler: vk::Sampler::null(),
                            });
                        }
                    }
                }
            }
        }

        // NOTE: We assign the below pointers after the above insertions so they remain stable!

        let accel_struct_handles = tls.accel_struct_handles.as_ptr();
        for write_idx in 0..tls.accel_struct_writes.len() {
            let IndexedWrite {
                info_idx: handle_idx,
                write,
            } = tls.accel_struct_writes[write_idx];

            unsafe {
                tls.accel_struct_infos.push(
                    vk::WriteDescriptorSetAccelerationStructureKHR {
                        acceleration_structure_count: write.descriptor_count,
                        p_acceleration_structures: accel_struct_handles.add(handle_idx),
                        ..Default::default()
                    },
                );
            }
        }

        let infos = tls.accel_struct_infos.as_ptr();
        for (write_idx, IndexedWrite { mut write, .. }) in
            tls.accel_struct_writes.drain(..).enumerate()
        {
            unsafe {
                write.p_next = infos.add(write_idx) as *const _;
            }

            tls.descriptors.push(write);
        }

        let buffer_infos_ptr = tls.buffer_infos.as_ptr();
        for write_idx in 0..tls.buffer_writes.len() {
            let IndexedWrite {
            info_idx,
            mut write,
            } = tls.buffer_writes[write_idx];
            unsafe {
                write.p_buffer_info = buffer_infos_ptr.add(info_idx);
            }
            tls.descriptors.push(write);
        }

        let image_infos_ptr = tls.image_infos.as_ptr();
        for write_idx in 0..tls.image_writes.len() {
            let IndexedWrite {
            info_idx,
            mut write,
            } = tls.image_writes[write_idx];
            unsafe {
                write.p_image_info = image_infos_ptr.add(info_idx);
            }
            tls.descriptors.push(write);
        }

        if !tls.descriptors.is_empty() {
            trace!(
                "  writing {} descriptors ({} buffers, {} images)",
                tls.descriptors.len(),
                tls.buffer_infos.len(),
                tls.image_infos.len()
            );

            unsafe {
                cmd_buf
                    .device
                    .update_descriptor_sets(tls.descriptors.as_slice(), &[]);
            }
        }

        Ok(())
        })
    }

    fn write_timestamp_queries(
        cmd_buf: &CommandBuffer,
        query_pool: Option<vk::QueryPool>,
        timestamp_queries: &[TimestampQueryData],
        placement: TimestampQueryPlacement,
        exec_idx: usize,
        start_idx: usize,
    ) -> usize {
        let mut query_idx = start_idx;

        while let Some(timestamp_query) = timestamp_queries.get(query_idx) {
            if timestamp_query.exec_idx < exec_idx
                || timestamp_query.exec_idx == exec_idx && timestamp_query.placement < placement
            {
                query_idx += 1;
                continue;
            }

            if timestamp_query.exec_idx > exec_idx
                || timestamp_query.exec_idx == exec_idx && timestamp_query.placement > placement
            {
                break;
            }

            let query_pool = query_pool.expect("missing query pool results");
            let pool_query = timestamp_query
                .pool_query
                .expect("missing timestamp query pool index");

            unsafe {
                cmd_buf.device.cmd_write_timestamp(
                    cmd_buf.handle,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    query_pool,
                    pool_query,
                );
            }

            query_idx += 1;
        }

        query_idx
    }
}

#[derive(Default)]
struct SubmitScratch {
    release_buffer_barriers: Vec<vk::BufferMemoryBarrier<'static>>,
    release_image_barriers: Vec<vk::ImageMemoryBarrier<'static>>,
    signal_infos: Vec<vk::SemaphoreSubmitInfo<'static>>,
    signal_semaphores: Vec<vk::Semaphore>,
    wait_infos: Vec<vk::SemaphoreSubmitInfo<'static>>,
    wait_semaphores: Vec<vk::Semaphore>,
    wait_stage_masks: Vec<vk::PipelineStageFlags>,
}

#[derive(Debug)]
struct SubmittedCommand {
    cmd: CommandData,
    _resources: CommandRecordingResources,
}

impl SubmittedCommand {
    fn signal_executed(&self) {
        self.cmd.tracking.signal_executed();
    }
}

#[derive(Debug)]
pub(crate) struct SubmittedTimestampQueries {
    epoch_query: u32,
    next_query: u32,
    query_pool: QueryPool,
    query_count: u32,
    result_infos: Vec<Option<TimestampQueryResultInfo>>,
    timestamp_period: f32,
    timestamp_valid_bits: u32,
}

impl SubmittedTimestampQueries {
    fn create(
        device: &Device,
        queue_family_index: u32,
        result_info_count: u32,
        query_count: u32,
    ) -> Result<Self, DriverError> {
        let device = device.clone();
        let Vulkan10Limits {
            timestamp_period, ..
        } = device.physical.properties_v1_0.limits;
        let QueueFamilyProperties {
            timestamp_valid_bits,
            ..
        } = device.physical.queue_families[queue_family_index as usize];
        let query_pool = QueryPool::create(&device, QueryPoolInfo::timestamp(query_count))?;

        Ok(Self {
            epoch_query: 0,
            next_query: 1,
            query_pool,
            query_count,
            result_infos: vec![None; result_info_count as usize],
            timestamp_period,
            timestamp_valid_bits,
        })
    }

    fn allocate_query(&mut self, query_count: u32) -> u32 {
        let query_count = query_count.max(1);
        let query = self.next_query;
        self.next_query += query_count;

        assert!(
            self.next_query <= self.query_count,
            "timestamp query pool exhausted while assigning query"
        );

        query
    }

    fn query_pool(&self) -> vk::QueryPool {
        self.query_pool.handle
    }

    fn reset(&self, cmd_buf: &CommandBuffer) {
        self.query_pool.reset(cmd_buf, 0, self.query_count);
    }

    fn set_result_info(&mut self, query: TimestampQuery, result_info: TimestampQueryResultInfo) {
        let index = query.index() as usize;
        if index >= self.result_infos.len() {
            self.result_infos.resize(index + 1, None);
        }

        self.result_infos[index] = Some(result_info);
    }

    fn timestamp_duration_since(
        timestamp: u64,
        earlier: u64,
        timestamp_valid_bits: u32,
        timestamp_period: f32,
    ) -> Duration {
        let mask = if timestamp_valid_bits >= u64::BITS {
            u64::MAX
        } else {
            (1_u64 << timestamp_valid_bits) - 1
        };
        let elapsed_ticks = timestamp.wrapping_sub(earlier) & mask;

        Duration::from_secs_f64(elapsed_ticks as f64 * timestamp_period as f64 / 1_000_000_000.0)
    }

    fn timestamp_results(&self) -> Result<Box<[Option<Duration>]>, DriverError> {
        let epoch =
            self.query_pool
                .results_u64(self.epoch_query, 1, vk::QueryResultFlags::empty())?[0];

        let mut results = Vec::with_capacity(self.result_infos.len());
        for result_info in &self.result_infos {
            let Some(result_info) = result_info else {
                results.push(None);
                continue;
            };

            let timestamp = self.query_pool.results_u64(
                result_info.timestamp_query,
                1,
                vk::QueryResultFlags::empty(),
            )?[0];

            results.push(Some(Self::timestamp_duration_since(
                timestamp,
                epoch,
                self.timestamp_valid_bits,
                self.timestamp_period,
            )));
        }

        Ok(results.into_boxed_slice())
    }

    fn write_epoch(&self, cmd_buf: &CommandBuffer) {
        unsafe {
            cmd_buf.device.cmd_write_timestamp(
                cmd_buf.handle,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool.handle,
                self.epoch_query,
            );
        }
    }
}

impl FenceDroppable for SubmittedTimestampQueries {
    fn fence_signaled(&mut self, fence: &Fence) {
        match self.timestamp_results() {
            Ok(results) => fence.timestamps.set(results),
            Err(err) => {
                warn!("unable to read timestamp query pool results: {err}");
                fence.timestamps.complete_without_results();
            }
        }
    }
}

#[derive(Clone, Copy)]
struct SubpassAccess {
    masks: PipelineStageAccessFlags,
    class: AccessClass,
    // Some only when every declaration in the physical group is this exact buffer writer.
    buffer_writer: Option<(AccessType, BufferSubresourceRange)>,
    // Some only when every declaration is this exact non-attachment storage-image reader.
    image_reader: Option<(AccessType, vk::ImageSubresourceRange)>,
}

#[derive(Clone, Copy)]
enum SubpassAccessOrigin<'a> {
    Explicit(&'a SubresourceAccess),
    Attachment(&'a Attachment),
}

#[derive(Default)]
struct SubpassDependencyScratch {
    attachment_nodes: FixedBitSet,
    dependencies: Vec<SubpassDependency>,
    group_lookup: Vec<usize>,
    // Flat physical summaries: (node, subpass, access), partitioned by offsets.
    groups: Vec<(NodeIndex, usize, SubpassAccess)>,
    // Read-only buffer groups cannot conflict with each other, but must survive for later writers.
    buffer_reader_history: Vec<Vec<usize>>,
    history: Vec<Vec<usize>>,
    offsets: Vec<usize>,
    pass_classes: Vec<Option<AccessClass>>,
    touched_dependencies: FixedBitSet,
    touched_nodes: Vec<NodeIndex>,
    touched_sources: Vec<usize>,
}

impl SubpassDependencyScratch {
    fn flush_dependencies(&mut self, output: &mut Vec<SubpassDependency>) {
        for src in self.touched_sources.drain(..) {
            output.push(self.dependencies[src]);
            self.touched_dependencies.set(src, false);
        }
    }

    // Contributions must share a destination until flushed. The last slot is external.
    fn record_dependency(
        &mut self,
        src_subpass: usize,
        dst_subpass: usize,
        previous: PipelineStageAccessFlags,
        current: PipelineStageAccessFlags,
        flags: vk::DependencyFlags,
    ) {
        let src = if src_subpass == vk::SUBPASS_EXTERNAL as usize {
            self.dependencies.len() - 1
        } else {
            src_subpass
        };
        let dep = &mut self.dependencies[src];
        if !self.touched_dependencies.put(src) {
            self.touched_sources.push(src);
            *dep = SubpassDependency::new(src_subpass as _, dst_subpass as _);
            dep.dependency_flags = flags;
        } else {
            debug_assert_eq!(dep.dst_subpass, dst_subpass as u32);
            dep.dependency_flags &= flags;
        }
        dep.src_stage_mask |= previous.stage_flags;
        dep.src_access_mask |= previous.access_flags;
        dep.dst_stage_mask |= current.stage_flags;
        dep.dst_access_mask |= current.access_flags;
    }

    fn reset(&mut self, node_count: usize, subpass_count: usize) {
        self.attachment_nodes.clear();
        self.attachment_nodes.grow(node_count);
        // Reset only visited slots, including partial plans left by an unwind.
        for src in self.touched_sources.drain(..) {
            self.touched_dependencies.set(src, false);
        }
        for node in self.touched_nodes.drain(..) {
            self.group_lookup[node] = usize::MAX;
            self.pass_classes[node] = None;
            self.history[node].clear();
            self.buffer_reader_history[node].clear();
        }
        self.dependencies
            .resize(subpass_count + 1, SubpassDependency::new(0, 0));
        self.groups.clear();
        // Keep inner allocations even when a later pass has fewer nodes.
        if self.history.len() < node_count {
            self.group_lookup.resize(node_count, usize::MAX);
            self.pass_classes.resize(node_count, None);
            self.history.resize_with(node_count, Vec::new);
            self.buffer_reader_history.resize_with(node_count, Vec::new);
        }
        self.offsets.clear();
        self.touched_dependencies.grow(subpass_count + 1);
    }
}

#[derive(Debug)]
struct TimestampQueryCompletion;

impl FenceDroppable for TimestampQueryCompletion {
    fn fence_signaled(&mut self, fence: &Fence) {
        fence.timestamps.complete_without_results();
    }
}

/// Timestamp query results associated with a completed fence.
#[derive(Clone, Debug)]
pub struct TimestampQueryPool {
    inner: Arc<Mutex<TimestampQueryPoolInner>>,
}

impl TimestampQueryPool {
    pub(crate) fn empty() -> Self {
        Self {
            inner: Arc::new(Mutex::new(TimestampQueryPoolInner {
                got_results: true,
                #[cfg(feature = "checked")]
                graph_id: None,
                timestamps: None,
            })),
        }
    }

    pub(crate) fn pending(#[cfg(feature = "checked")] graph_id: GraphId) -> Self {
        Self {
            inner: Arc::new(Mutex::new(TimestampQueryPoolInner {
                got_results: false,
                #[cfg(feature = "checked")]
                graph_id: Some(graph_id),
                timestamps: None,
            })),
        }
    }

    pub(crate) fn complete_without_results(&self) {
        self.inner
            .lock()
            .expect("timestamp query pool poisoned")
            .got_results = true;
    }

    /// Returns the duration from submission start to `query`, or `None` if results are not available.
    ///
    /// `None` can mean the submission is still pending, timestamps were unsupported for the queue,
    /// or the query point was not part of submitted graph work. Use [`Self::has_results`] to
    /// distinguish pending work from a completed submission with no timestamp for this query.
    ///
    /// When the `checked` feature is enabled, this panics if `query` belongs to a different graph.
    pub fn duration(&self, query: TimestampQuery) -> Option<Duration> {
        let inner = self.inner.lock().expect("timestamp query pool poisoned");

        #[cfg(feature = "checked")]
        assert_eq!(
            inner.graph_id,
            Some(query.graph_id()),
            "timestamp query belongs to a different graph"
        );

        inner
            .timestamps
            .as_ref()
            .and_then(|timestamps| timestamps.get(query.index() as usize).copied().flatten())
    }

    /// Returns `true` once the associated submission has completed.
    ///
    /// A complete pool can still return `None` for a query when timestamps were unsupported,
    /// omitted, or never submitted.
    pub fn has_results(&self) -> bool {
        self.inner
            .lock()
            .expect("timestamp query pool poisoned")
            .got_results
    }

    pub(crate) fn set(&self, timestamps: Box<[Option<Duration>]>) {
        let mut inner = self.inner.lock().expect("timestamp query pool poisoned");
        inner.timestamps = Some(timestamps);
        inner.got_results = true;
    }
}

#[derive(Debug)]
struct TimestampQueryPoolInner {
    got_results: bool,
    #[cfg(feature = "checked")]
    graph_id: Option<GraphId>,
    timestamps: Option<Box<[Option<Duration>]>>,
}

#[derive(Clone, Copy, Debug)]
struct TimestampQueryResultInfo {
    timestamp_query: u32,
}

#[derive(Clone, Copy, Debug)]
struct TrackedImageBarrier {
    previous_accesses: ImageAccessSet,
    next_access: AccessType,
    previous_layout: ImageLayout,
    next_layout: ImageLayout,
    ownership_layouts: Option<ImageOwnershipLayouts>,
    discard_contents: bool,
    src_queue_family_index: u32,
    dst_queue_family_index: u32,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
}

impl TrackedImageBarrier {
    fn new(
        image: vk::Image,
        prev_access: ImageAccessSet,
        next_access: AccessType,
        range: vk::ImageSubresourceRange,
        transfer: Option<&ImageOwnershipTransfer>,
        discard_contents: bool,
    ) -> Self {
        trace!(
            "    image {:?} {:?} {:?}->{:?}",
            image,
            ImageSubresourceRangeDebug(range),
            prev_access,
            next_access,
        );

        Self {
            next_access,
            next_layout: TrackedImageBarrier::access_layout(next_access),
            previous_accesses: prev_access,
            previous_layout: TrackedImageBarrier::access_set_layout(prev_access),
            ownership_layouts: transfer.map(|transfer| transfer.layouts),
            discard_contents,
            src_queue_family_index: transfer.map_or(vk::QUEUE_FAMILY_IGNORED, |transfer| {
                transfer.src_queue_family_index
            }),
            dst_queue_family_index: transfer.map_or(vk::QUEUE_FAMILY_IGNORED, |transfer| {
                transfer.dst_queue_family_index
            }),
            image,
            range,
        }
    }

    fn from_transfers<'a>(
        image: vk::Image,
        prev_access: ImageAccessSet,
        next_access: AccessType,
        range: vk::ImageSubresourceRange,
        transfers: &'a [ImageOwnershipTransfer],
        discard_contents: bool,
    ) -> impl Iterator<Item = TrackedImageBarrier> + 'a {
        ImageOwnershipTransfer::barrier_ranges(transfers, range).map(move |(range, transfer)| {
            TrackedImageBarrier::new(
                image,
                prev_access,
                next_access,
                range,
                transfer,
                discard_contents,
            )
        })
    }

    const fn access_layout(access: AccessType) -> ImageLayout {
        if matches!(access, AccessType::Present | AccessType::ComputeShaderWrite) {
            ImageLayout::General
        } else {
            ImageLayout::Optimal
        }
    }

    fn access_set_layout(access_set: ImageAccessSet) -> ImageLayout {
        access_set
            .non_sampled_access()
            .map_or(ImageLayout::Optimal, TrackedImageBarrier::access_layout)
    }

    fn can_elide_sampled_read(barrier: TrackedImageBarrier) -> bool {
        if !barrier.previous_accesses.is_sampled_read()
            || !ImageAccessSet::from_access(barrier.next_access).is_sampled_read()
            || !barrier
                .previous_accesses
                .contains_sampled_read(barrier.next_access)
            || barrier.ownership_layouts.is_some()
            || barrier.discard_contents
        {
            return false;
        }

        let (_, _, barrier) = TrackedImageBarrier::memory_barrier(barrier);

        barrier.old_layout == barrier.new_layout
            && barrier.src_queue_family_index == barrier.dst_queue_family_index
    }

    fn execution_discard_contents(prev_access: ImageAccessSet) -> bool {
        prev_access.is_nothing()
    }

    fn layout_transition_discard_contents(
        prev_access: ImageAccessSet,
        next_access: AccessType,
    ) -> bool {
        // Read/modify/write accesses must preserve the existing image contents
        // Check for "not-read" here because some accesses both read and write
        // Color Attachment Read/Write (blending) will prevent discarding contents
        prev_access.is_nothing() || !is_read_access(next_access)
    }

    fn memory_barrier(
        barrier: TrackedImageBarrier,
    ) -> (
        vk::PipelineStageFlags,
        vk::PipelineStageFlags,
        vk::ImageMemoryBarrier<'static>,
    ) {
        let ownership_layouts = barrier.ownership_layouts;
        let resolve_access = barrier.previous_accesses.depth_stencil_resolve_access();
        let previous_accesses = barrier
            .previous_accesses
            .without_depth_stencil_resolve()
            .iter()
            .collect::<SmallVec<[AccessType; 10]>>();
        let next_accesses = [barrier.next_access];
        let (mut src_stage_mask, mut dst_stage_mask, barrier) =
            get_image_memory_barrier(&ImageBarrier {
                previous_accesses: previous_accesses.as_slice(),
                next_accesses: &next_accesses,
                previous_layout: barrier.previous_layout,
                next_layout: barrier.next_layout,
                discard_contents: barrier.discard_contents,
                src_queue_family_index: barrier.src_queue_family_index,
                dst_queue_family_index: barrier.dst_queue_family_index,
                image: barrier.image,
                range: barrier.range,
            });

        let mut barrier = vk::ImageMemoryBarrier {
            src_access_mask: barrier.src_access_mask,
            dst_access_mask: barrier.dst_access_mask,
            old_layout: barrier.old_layout,
            new_layout: barrier.new_layout,
            src_queue_family_index: barrier.src_queue_family_index,
            dst_queue_family_index: barrier.dst_queue_family_index,
            image: barrier.image,
            subresource_range: barrier.subresource_range,
            ..Default::default()
        };

        if let Some(resolve_access) = resolve_access {
            // Lower the extra scope separately: feeding a color access into vk-sync's image
            // layout selection would replace/conflict with the depth/stencil layout.
            let resolve_accesses = [resolve_access];
            let (src, dst, resolve) = get_memory_barrier(&GlobalBarrier {
                previous_accesses: &resolve_accesses,
                next_accesses: &next_accesses,
            });
            src_stage_mask |= src;
            dst_stage_mask |= dst;
            barrier.src_access_mask |= resolve.src_access_mask;
            barrier.dst_access_mask |= resolve.dst_access_mask;
        }

        if let Some(layouts) = ownership_layouts {
            // The source scope of an acquire is ignored, but its stage must still be supported by the
            // destination queue family. ALL_COMMANDS is valid for every queue capability.
            src_stage_mask = vk::PipelineStageFlags::ALL_COMMANDS;
            barrier.src_access_mask = vk::AccessFlags::empty();
            barrier.old_layout = layouts.old;
            barrier.new_layout = layouts.new;
        }

        (src_stage_mask, dst_stage_mask, barrier)
    }
}

#[doc(hidden)]
pub mod bench {
    use {
        super::{CommandAccessIndex, Schedule, Submission},
        crate::{Graph, resource::ResourceSetIndex},
    };

    // CPU-only subpass dependency fixtures; no Vulkan objects are created.
    #[cfg(feature = "bench-internals")]
    use {
        super::{GraphicsExecutionInfo, PipelineStageAccessFlags},
        crate::{
            AccessType, Attachment, ColorAttachment, ColorResolve, CommandData, Execution, LoadOp,
            StoreOp,
            cmd::{SubresourceAccess, SubresourceRange},
            driver::{SubpassDependency, image::SampleCount},
        },
        ash::vk,
    };

    /// Reusable benchmark harness for `Schedule::reorder_cmds`.
    pub struct ReorderBenchHarness {
        schedule: Schedule,
        original_cmds: Vec<usize>,
        end_cmd_idx: usize,
    }

    impl ReorderBenchHarness {
        /// Builds a deterministic synthetic schedule for benchmarking.
        pub fn new(spec: ReorderBenchSpec) -> Self {
            assert!(spec.cmd_count > 0, "cmd_count must be greater than zero");
            assert!(
                spec.resource_count > 0,
                "resource_count must be greater than zero"
            );
            assert!(
                spec.short_lived_uses > 0,
                "short_lived_uses must be greater than zero"
            );

            let mut cmds_by_node = vec![Vec::new(); spec.resource_count];
            let mut accessed_nodes_by_cmd = vec![Vec::new(); spec.cmd_count];

            for (node_idx, cmds) in cmds_by_node.iter_mut().enumerate() {
                let is_long_lived = node_idx < spec.long_lived_resource_count;
                let uses = if is_long_lived {
                    spec.long_lived_uses.max(spec.short_lived_uses)
                } else {
                    spec.short_lived_uses
                }
                .min(spec.cmd_count);

                let seed = ReorderBenchHarness::splitmix64(
                    node_idx as u64 ^ ((spec.cmd_count as u64) << 32),
                );
                let stride = ReorderBenchHarness::odd_stride(seed, spec.cmd_count);
                let start = (seed as usize) % spec.cmd_count;
                let cluster_len = uses.max(1).min(spec.cmd_count);

                cmds.reserve(uses);

                for use_idx in 0..uses {
                    let cmd_idx = if is_long_lived {
                        (start + use_idx * stride) % spec.cmd_count
                    } else {
                        (start + use_idx % cluster_len + (use_idx / cluster_len) * stride)
                            % spec.cmd_count
                    };

                    cmds.push(cmd_idx);
                }

                cmds.sort_unstable();
                cmds.dedup();

                for next_cmd in 0..spec.cmd_count {
                    if cmds.len() >= uses {
                        break;
                    }
                    if let Err(idx) = cmds.binary_search(&next_cmd) {
                        cmds.insert(idx, next_cmd);
                    }
                }

                for &cmd_idx in cmds.iter() {
                    accessed_nodes_by_cmd[cmd_idx].push(node_idx);
                }
            }

            for nodes in &mut accessed_nodes_by_cmd {
                nodes.sort_unstable();
                nodes.dedup();
            }

            let cmds = (0..spec.cmd_count).collect::<Vec<_>>();

            Self {
                schedule: Schedule {
                    access_index: CommandAccessIndex {
                        cmds_by_node,
                        accessed_nodes_by_cmd,
                        ..Default::default()
                    },
                    cmds: cmds.clone(),
                    ..Default::default()
                },
                original_cmds: cmds,
                end_cmd_idx: spec.cmd_count,
            }
        }

        /// Builds a scheduler benchmark from a graph, optionally repeating its disconnected
        /// topology with independently remapped command and resource indices.
        pub fn from_graph(graph: &Graph, repeat_count: usize) -> Self {
            assert!(repeat_count > 0, "repeat_count must be greater than zero");

            let base_cmd_count = graph.cmds.len();
            let base_resource_count = graph.resources.len();
            let base_resource_set_count = graph.resource_sets.len();
            assert!(base_cmd_count > 0, "graph must contain commands");
            assert!(
                base_resource_count > 0 || base_resource_set_count > 0,
                "graph must contain resources or resource sets"
            );

            let mut base = CommandAccessIndex::default();
            base.update_from_cmds(&graph.cmds, base_resource_count, base_resource_set_count);

            let cmd_count = base_cmd_count * repeat_count;
            let resource_count = base_resource_count * repeat_count;
            let resource_set_count = base_resource_set_count * repeat_count;
            let mut cmds_by_node = Vec::with_capacity(resource_count);
            let mut cmds_by_resource_set = Vec::with_capacity(resource_set_count);
            let mut accessed_nodes_by_cmd = Vec::with_capacity(cmd_count);
            let mut accessed_resource_sets_by_cmd = Vec::with_capacity(cmd_count);

            for copy_idx in 0..repeat_count {
                let cmd_offset = copy_idx * base_cmd_count;
                let resource_offset = copy_idx * base_resource_count;
                let resource_set_offset = copy_idx * base_resource_set_count;

                cmds_by_node.extend(base.cmds_by_node.iter().map(|cmds| {
                    cmds.iter()
                        .map(|cmd_idx| cmd_offset + cmd_idx)
                        .collect::<Vec<_>>()
                }));
                accessed_nodes_by_cmd.extend(base.accessed_nodes_by_cmd.iter().map(|nodes| {
                    nodes
                        .iter()
                        .map(|node_idx| resource_offset + node_idx)
                        .collect::<Vec<_>>()
                }));
                cmds_by_resource_set.extend(base.cmds_by_resource_set.iter().map(|cmds| {
                    cmds.iter()
                        .map(|cmd_idx| cmd_offset + cmd_idx)
                        .collect::<Vec<_>>()
                }));
                accessed_resource_sets_by_cmd.extend(
                    base.accessed_resource_sets_by_cmd
                        .iter()
                        .map(|resource_sets| {
                            resource_sets
                                .iter()
                                .map(|resource_set_idx| {
                                    ResourceSetIndex::new(
                                        resource_set_offset + resource_set_idx.as_usize(),
                                    )
                                })
                                .collect::<Vec<_>>()
                        }),
                );
            }

            let cmds = (0..cmd_count).collect::<Vec<_>>();
            Self {
                schedule: Schedule {
                    access_index: CommandAccessIndex {
                        cmds_by_node,
                        accessed_nodes_by_cmd,
                        cmds_by_resource_set,
                        accessed_resource_sets_by_cmd,
                    },
                    cmds: cmds.clone(),
                    ..Default::default()
                },
                original_cmds: cmds,
                end_cmd_idx: cmd_count,
            }
        }

        /// Builds a harness from a finalized submission graph.
        pub fn from_submission(submission: &Submission, repeat_count: usize) -> Self {
            Self::from_graph(&submission.graph, repeat_count)
        }

        /// Returns the number of commands reordered by each benchmark iteration.
        pub fn cmd_count(&self) -> usize {
            self.end_cmd_idx
        }

        fn odd_stride(seed: u64, cmd_count: usize) -> usize {
            let stride = ((seed >> 32) as usize % cmd_count.max(2)) | 1;

            stride.min(cmd_count.max(1) - 1).max(1)
        }

        /// Restores the original schedule, reorders it once, and returns a checksum.
        pub fn reorder_once(&mut self) -> u64 {
            self.schedule.cmds.clear();
            self.schedule
                .cmds
                .extend(self.original_cmds.iter().copied());

            self.schedule.reorder_cmds(self.end_cmd_idx);

            self.schedule
                .cmds
                .iter()
                .enumerate()
                .fold(0u64, |checksum, (idx, &pass_idx)| {
                    checksum.wrapping_mul(1_099_511_628_211).wrapping_add(
                        ((idx as u64) << 32) ^ pass_idx as u64 ^ 0x9e37_79b9_7f4a_7c15,
                    )
                })
        }

        fn splitmix64(mut value: u64) -> u64 {
            value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^ (value >> 31)
        }
    }

    /// Synthetic workload description for scheduler benchmarks.
    #[derive(Clone, Copy, Debug)]
    pub struct ReorderBenchSpec {
        /// Number of scheduled cmds.
        pub cmd_count: usize,

        /// Number of resources participating in the schedule.
        pub resource_count: usize,

        /// Typical cmd count for short-lived resources.
        pub short_lived_uses: usize,

        /// Number of long-lived resources shared across many cmds.
        pub long_lived_resource_count: usize,

        /// Typical cmd count for each long-lived resource.
        pub long_lived_uses: usize,
    }

    #[cfg(feature = "bench-internals")]
    /// Dependency shapes exercised by the benchmark matrix.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum SubpassDepsCase {
        /// Each output is the next subpass's input attachment.
        AttachmentChain,
        /// The chain plus four UBOs and four sampled images shared by every subpass.
        SharedReads,
        /// The chain plus a shared storage buffer written by every subpass.
        /// Identical writers require only an adjacent global dependency chain.
        StorageHazards,
        /// The chain plus 16 storage buffers written once, then read by every later subpass.
        WriteOnceReadMany,
        /// The chain plus a shared storage image read identically by every subpass.
        StorageImageReads,
        /// Multisampled producer/input attachments cannot prove single-sample locality.
        Multisample,
        /// Each multisampled output resolves to the next subpass's single-sample input.
        Resolve,
        /// Two-draw writer groups alternate with singleton input consumers and shared reads.
        Grouped,
    }

    #[cfg(feature = "bench-internals")]
    impl SubpassDepsCase {
        /// Stable Criterion case name.
        pub fn name(self) -> &'static str {
            match self {
                Self::AttachmentChain => "attachment_chain",
                Self::SharedReads => "shared_ubo_sampled_reads",
                Self::StorageHazards => "storage_global_hazards",
                Self::WriteOnceReadMany => "write_once_read_many",
                Self::StorageImageReads => "storage_image_read_chain",
                Self::Multisample => "multisample_exclusion",
                Self::Resolve => "resolve_exclusion",
                Self::Grouped => "grouped_logical_executions",
            }
        }
    }

    #[cfg(feature = "bench-internals")]
    /// Immutable, reusable inputs to the private dependency builder.
    pub struct SubpassDepsHarness {
        spec: SubpassDepsSpec,
        pass: CommandData,
        external_history: Vec<PipelineStageAccessFlags>,
        exec_subpasses: Vec<u32>,
    }

    #[cfg(feature = "bench-internals")]
    impl SubpassDepsHarness {
        /// Constructs dense node indices, frozen accesses, and synthetic producer history.
        pub fn new(spec: SubpassDepsSpec) -> Self {
            use SubpassDepsCase::*;
            let n = spec.subpass_count;
            assert!(n > 0 && n < vk::SUBPASS_EXTERNAL as usize);
            let grouped = spec.case == Grouped;
            let resolve = spec.case == Resolve;
            let shared_pairs = if matches!(spec.case, SharedReads | Grouped) {
                4
            } else {
                0
            };
            let storage = matches!(spec.case, StorageHazards | StorageImageReads);
            let shared_buffers = if spec.case == WriteOnceReadMany { 16 } else { 0 };
            let attachment_nodes = if grouped {
                n / 2 + 1
            } else {
                n * if resolve { 2 } else { 1 }
            };
            let node_count =
                attachment_nodes + shared_pairs * 2 + usize::from(storage) + shared_buffers;
            let sample_count = if matches!(spec.case, Multisample | Resolve) {
                SampleCount::Type4
            } else {
                SampleCount::Type1
            };
            let image_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let attachment = |target, sample_count| Attachment {
                array_layer_count: 1,
                aspect_mask: image_range.aspect_mask,
                base_array_layer: 0,
                base_mip_level: 0,
                format: vk::Format::R8G8B8A8_UNORM,
                mip_level_count: 1,
                sample_count,
                target,
            };
            let mut execs = Vec::new();
            let mut exec_subpasses = Vec::new();
            for sp in 0..n {
                let mut exec = Execution::default();
                let writer_group = grouped && sp % 2 == 0;
                // A consumer writes the next writer group's target, keeping a connected chain.
                let output = if grouped { sp.div_ceil(2) } else { sp };
                // Stable attachment indices across the whole render pass.
                exec.attachments.color.resize(attachment_nodes, None);
                exec.attachments.color[output] = Some(ColorAttachment {
                    attachment: attachment(output, sample_count),
                    load: if writer_group && sp > 0 {
                        LoadOp::Load
                    } else {
                        LoadOp::DontCare
                    },
                    store: StoreOp::Store,
                    resolve: None,
                    is_input: false,
                    is_attachment: true,
                });
                exec.accesses.push(
                    output,
                    SubresourceAccess {
                        access: AccessType::ColorAttachmentWrite,
                        subresource: SubresourceRange::Image(image_range),
                    },
                );
                if resolve {
                    let destination = attachment(n + sp, SampleCount::Type1);
                    exec.attachments.color[n + sp] = Some(ColorAttachment {
                        attachment: destination,
                        load: LoadOp::DontCare,
                        store: StoreOp::Store,
                        resolve: Some(ColorResolve {
                            attachment: destination,
                            src_attachment_idx: sp as u32,
                        }),
                        is_input: false,
                        is_attachment: false,
                    });
                }
                if sp > 0 && !writer_group {
                    // Reuse the preceding producer's node, never a fresh input-only image.
                    let input = if grouped {
                        output - 1
                    } else if resolve {
                        n + sp - 1
                    } else {
                        sp - 1
                    };
                    exec.attachments.color[input] = Some(ColorAttachment {
                        attachment: attachment(
                            input,
                            if resolve {
                                SampleCount::Type1
                            } else {
                                sample_count
                            },
                        ),
                        load: LoadOp::Load,
                        store: StoreOp::DontCare,
                        resolve: None,
                        is_input: true,
                        is_attachment: false,
                    });
                    exec.accesses.push(
                        input,
                        SubresourceAccess {
                            access: AccessType::FragmentShaderReadColorInputAttachment,
                            subresource: SubresourceRange::Image(image_range),
                        },
                    );
                }
                for pair in 0..shared_pairs {
                    exec.accesses.push(
                        attachment_nodes + pair * 2,
                        SubresourceAccess {
                            access: AccessType::VertexShaderReadUniformBuffer,
                            subresource: SubresourceRange::Buffer((0..256).into()),
                        },
                    );
                    exec.accesses.push(
                        attachment_nodes + pair * 2 + 1,
                        SubresourceAccess {
                            access: AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                            subresource: SubresourceRange::Image(image_range),
                        },
                    );
                }
                for buffer in 0..shared_buffers {
                    exec.accesses.push(
                        attachment_nodes + buffer,
                        SubresourceAccess {
                            access: if sp == 0 {
                                AccessType::FragmentShaderWrite
                            } else {
                                AccessType::FragmentShaderReadOther
                            },
                            subresource: SubresourceRange::Buffer((0..256).into()),
                        },
                    );
                }
                if storage {
                    exec.accesses.push(
                        attachment_nodes,
                        if spec.case == StorageImageReads {
                            SubresourceAccess {
                                access: AccessType::FragmentShaderReadOther,
                                subresource: SubresourceRange::Image(image_range),
                            }
                        } else {
                            SubresourceAccess {
                                access: AccessType::FragmentShaderWrite,
                                subresource: SubresourceRange::Buffer((0..256).into()),
                            }
                        },
                    );
                }
                exec.accesses.freeze();
                if writer_group {
                    execs.push(exec.clone());
                    exec_subpasses.push(sp as u32);
                    exec.attachments.color[output].as_mut().unwrap().load = LoadOp::Load;
                }
                execs.push(exec);
                exec_subpasses.push(sp as u32);
            }
            Self {
                spec,
                pass: CommandData {
                    execs,
                    #[cfg(debug_assertions)]
                    name: None,
                    stream_scope_id: None,
                    tracking: Default::default(),
                },
                external_history: vec![
                    PipelineStageAccessFlags::new(AccessType::TransferWrite);
                    node_count
                ],
                exec_subpasses,
            }
        }

        /// Runs only the production builder; allocations and output destruction are benchmarked.
        pub fn build_deps_once(&self) -> SubpassDepsResult {
            SubpassDepsResult(Submission::build_subpass_dependencies(
                &self.pass,
                &self.external_history,
                &self.exec_subpasses,
            ))
        }

        /// Checks fixture topology and synchronization semantics outside the timed loop.
        pub fn validate(&self) {
            use SubpassDepsCase::*;
            let n = self.spec.subpass_count;
            let inputs = self
                .pass
                .execs
                .iter()
                .map(|exec| {
                    exec.attachments
                        .color_attachments()
                        .filter_map(|(idx, state)| state.is_input.then_some(idx))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let graphics = self
                .pass
                .execs
                .iter()
                .zip(&inputs)
                .map(|(exec, inputs)| GraphicsExecutionInfo {
                    input_attachments: inputs,
                    sample_count: exec
                        .attachments
                        .color_attachments()
                        .find(|(_, state)| state.is_attachment)
                        .unwrap()
                        .1
                        .attachment
                        .sample_count,
                })
                .collect::<Vec<_>>();
            // Exercise production coalescing without pipelines or other Vulkan objects.
            let (info, actual_mapping) =
                Submission::build_render_pass_info(&self.pass, &self.external_history, &graphics);
            assert_eq!(&*actual_mapping, &self.exec_subpasses, "{:?}", self.spec);
            assert_eq!(info.subpasses.len(), n, "{:?}", self.spec);

            let deps = self.build_deps_once().0;
            let grouped = self.spec.case == Grouped;
            let storage = matches!(self.spec.case, StorageHazards | StorageImageReads);
            let write_once = self.spec.case == WriteOnceReadMany;
            let internal_count = if grouped {
                // A target spans consumer -> writer group -> consumer; keep the first writer too.
                n - 1 + n.saturating_sub(2) / 2
            } else if write_once {
                n - 1 + n.saturating_sub(2)
            } else {
                n - 1
            };
            assert_eq!(deps.len(), n + internal_count, "{:?}", self.spec);
            let external = deps
                .iter()
                .filter(|dep| dep.src_subpass == vk::SUBPASS_EXTERNAL)
                .count();
            assert_eq!(external, n, "{:?}", self.spec);
            for dst in 0..n as u32 {
                let external = deps
                    .iter()
                    .find(|dep| dep.src_subpass == vk::SUBPASS_EXTERNAL && dep.dst_subpass == dst)
                    .expect("one external edge per physical subpass");
                assert!(external.dependency_flags.is_empty());
                assert!(external.src_stage_mask.contains(
                    vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::TRANSFER
                ));
                assert_eq!(
                    external.src_access_mask,
                    vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
                );
                if matches!(self.spec.case, SharedReads | Grouped) {
                    assert!(
                        external
                            .dst_access_mask
                            .contains(vk::AccessFlags::UNIFORM_READ | vk::AccessFlags::SHADER_READ)
                    );
                }
            }
            for src in 0..n as u32 {
                for dst in src + 1..n as u32 {
                    let expected = dst == src + 1
                        || (grouped && src % 2 == 1 && dst == src + 2)
                        || (write_once && src == 0);
                    assert_eq!(
                        deps.binary_search_by_key(&(src, dst), |dep| {
                            (dep.src_subpass, dep.dst_subpass)
                        })
                        .is_ok(),
                        expected,
                        "{:?}: {src} -> {dst}",
                        self.spec
                    );
                }
            }
            for dep in deps
                .iter()
                .filter(|dep| dep.src_subpass != vk::SUBPASS_EXTERNAL)
            {
                assert!(dep.src_subpass < dep.dst_subpass && dep.dst_subpass < n as u32);
                let local = matches!(self.spec.case, AttachmentChain | SharedReads | Grouped)
                    || (write_once && dep.src_subpass != 0);
                assert_eq!(
                    dep.dependency_flags,
                    if local {
                        vk::DependencyFlags::BY_REGION
                    } else {
                        vk::DependencyFlags::empty()
                    },
                    "{:?}: {dep:?}",
                    self.spec
                );
                if write_once && dep.src_subpass == 0 {
                    assert!(dep.src_access_mask.contains(vk::AccessFlags::SHADER_WRITE));
                    assert!(dep.dst_access_mask.contains(vk::AccessFlags::SHADER_READ));
                } else if !storage {
                    if write_once {
                        assert_eq!(dep.dst_subpass, dep.src_subpass + 1);
                    }
                    assert!(
                        dep.src_access_mask
                            .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    );
                    let read = if grouped && dep.dst_subpass % 2 == 0 {
                        vk::AccessFlags::COLOR_ATTACHMENT_READ
                    } else {
                        vk::AccessFlags::INPUT_ATTACHMENT_READ
                    };
                    assert!(dep.dst_access_mask.contains(read));
                    assert!(
                        !dep.dst_access_mask.intersects(
                            vk::AccessFlags::UNIFORM_READ | vk::AccessFlags::SHADER_READ
                        )
                    );
                } else {
                    let access = if self.spec.case == StorageImageReads {
                        vk::AccessFlags::SHADER_READ
                    } else {
                        vk::AccessFlags::SHADER_WRITE
                    };
                    assert!(dep.src_access_mask.contains(access));
                    assert!(dep.dst_access_mask.contains(access));
                }
            }
        }
    }

    #[cfg(feature = "bench-internals")]
    /// Opaque owned builder output, avoiding conversion or checksums inside timing.
    pub struct SubpassDepsResult(Vec<SubpassDependency>);

    #[cfg(feature = "bench-internals")]
    /// Synthetic workload size counts physical subpasses, not logical executions.
    #[derive(Clone, Copy, Debug)]
    pub struct SubpassDepsSpec {
        /// Number of physical subpasses (at least one).
        pub subpass_count: usize,
        /// Resource and execution topology.
        pub case: SubpassDepsCase,
    }

    #[cfg(feature = "bench-internals")]
    impl SubpassDepsSpec {
        /// Size one exercises the single-subpass fast path; larger fan-outs expose history scans.
        pub fn matrix() -> impl Iterator<Item = Self> {
            use SubpassDepsCase::*;
            [
                AttachmentChain,
                SharedReads,
                StorageHazards,
                StorageImageReads,
                Multisample,
                Resolve,
                Grouped,
            ]
            .into_iter()
            .flat_map(|case| {
                [1, 2, 4, 8, 16, 32]
                    .into_iter()
                    .map(move |subpass_count| Self {
                        subpass_count,
                        case,
                    })
            })
            .chain([1, 16, 64, 256, 1024].into_iter().map(|subpass_count| Self {
                subpass_count,
                case: WriteOnceReadMany,
            }))
        }
    }

    #[cfg(all(feature = "bench-internals", test))]
    mod tests {
        use super::*;

        #[test]
        fn fixtures_have_expected_dependencies() {
            for spec in SubpassDepsSpec::matrix() {
                let harness = SubpassDepsHarness::new(spec);
                harness.validate();
                let expected = (0..spec.subpass_count as u32)
                    .flat_map(|sp| {
                        let repeats = if spec.case == SubpassDepsCase::Grouped && sp % 2 == 0 {
                            2
                        } else {
                            1
                        };
                        std::iter::repeat_n(sp, repeats)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(harness.exec_subpasses, expected);
            }
        }
    }
}

#[doc(hidden)]
pub mod fuzz {
    use super::{CommandAccessIndex, Schedule};

    #[derive(Clone, Copy, Debug)]
    pub struct ResourceAccess {
        pub cmd_idx: usize,
        pub write: bool,
    }

    impl ResourceAccess {
        fn assert_hazard_order_preserved(
            reordered: &[usize],
            resource_accesses: &[Vec<ResourceAccess>],
        ) {
            let mut positions = vec![usize::MAX; reordered.len()];
            for (position, &cmd_idx) in reordered.iter().enumerate() {
                positions[cmd_idx] = position;
            }

            for accesses in resource_accesses {
                for (left_idx, left) in accesses.iter().enumerate() {
                    for right in &accesses[(left_idx + 1)..] {
                        if left.write || right.write {
                            assert!(
                                positions[left.cmd_idx] < positions[right.cmd_idx],
                                "hazard order changed for resource accesses {:?} -> {:?}: {:?}",
                                left,
                                right,
                                reordered
                            );
                        }
                    }
                }
            }
        }

        fn build_access_index(
            cmd_count: usize,
            resource_accesses: &[Vec<ResourceAccess>],
        ) -> (CommandAccessIndex, Vec<Vec<ResourceAccess>>) {
            let mut cmds_by_node = Vec::with_capacity(resource_accesses.len());
            let mut accessed_nodes_by_cmd = vec![Vec::new(); cmd_count];
            let mut normalized_accesses = Vec::with_capacity(resource_accesses.len());

            for (node_idx, accesses) in resource_accesses.iter().enumerate() {
                let mut normalized = accesses
                    .iter()
                    .copied()
                    .filter(|access| access.cmd_idx < cmd_count)
                    .collect::<Vec<_>>();
                normalized.sort_unstable_by_key(|access| access.cmd_idx);

                let mut deduped = Vec::<ResourceAccess>::with_capacity(normalized.len());
                for access in normalized {
                    if let Some(prev) = deduped.last_mut()
                        && prev.cmd_idx == access.cmd_idx
                    {
                        prev.write |= access.write;
                        continue;
                    }

                    deduped.push(access);
                }

                for access in &deduped {
                    accessed_nodes_by_cmd[access.cmd_idx].push(node_idx);
                }

                cmds_by_node.push(deduped.iter().map(|access| access.cmd_idx).collect());
                normalized_accesses.push(deduped);
            }

            (
                CommandAccessIndex {
                    cmds_by_node,
                    accessed_nodes_by_cmd,
                    ..Default::default()
                },
                normalized_accesses,
            )
        }

        fn reference_reorder(access_index: CommandAccessIndex, cmd_count: usize) -> Vec<usize> {
            if cmd_count < 3 {
                return (0..cmd_count).collect();
            }

            let mut predecessors = vec![Vec::new(); cmd_count];
            for resource_cmds in &access_index.cmds_by_node {
                for pair in resource_cmds.windows(2) {
                    predecessors[pair[1]].push(pair[0]);
                }
            }

            let mut scheduled = vec![false; cmd_count];
            let mut reordered = Vec::with_capacity(cmd_count);
            while reordered.len() < cmd_count {
                let mut best = None;
                for cmd_idx in 0..cmd_count {
                    if scheduled[cmd_idx]
                        || !predecessors[cmd_idx]
                            .iter()
                            .all(|&predecessor| scheduled[predecessor])
                    {
                        continue;
                    }

                    let score = predecessors[cmd_idx].len();
                    if best.is_none_or(|(best_score, best_idx)| {
                        score > best_score || (score == best_score && cmd_idx < best_idx)
                    }) {
                        best = Some((score, cmd_idx));
                    }
                }

                let (_, best_idx) = best.expect("command dependency cycle detected");
                scheduled[best_idx] = true;
                reordered.push(best_idx);
            }

            reordered
        }
    }

    pub fn check_schedule_reordering(cmd_count: usize, resource_accesses: &[Vec<ResourceAccess>]) {
        let cmd_count = cmd_count.min(256);
        if cmd_count == 0 {
            return;
        }

        let (access_index, normalized_accesses) =
            ResourceAccess::build_access_index(cmd_count, resource_accesses);

        let mut schedule = Schedule {
            access_index: access_index.clone(),
            cmds: (0..cmd_count).collect(),
            ..Default::default()
        };

        schedule.reorder_cmds(cmd_count);

        let reordered = schedule.cmds.clone();

        assert_eq!(reordered.len(), cmd_count, "reordered cmd count changed");

        let mut sorted = reordered.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            (0..cmd_count).collect::<Vec<_>>(),
            "reordered cmds are not a permutation"
        );

        let mut repeat = Schedule {
            access_index: access_index.clone(),
            cmds: (0..cmd_count).collect(),
            ..Default::default()
        };
        repeat.reorder_cmds(cmd_count);
        assert_eq!(reordered, repeat.cmds, "reordering is not deterministic");

        let expected = ResourceAccess::reference_reorder(access_index, cmd_count);
        assert_eq!(
            reordered, expected,
            "reordering diverged from reference implementation"
        );

        ResourceAccess::assert_hazard_order_preserved(&reordered, &normalized_accesses);
    }
}

#[cfg(test)]
mod test {
    // Run validation tests with --test-threads=1: the error counter and logger are process-global.
    use super::{
        BufferQueueOwnershipTransfer, CommandAccessIndex, CommandData, CommandRecordingResources,
        GraphicsExecutionInfo, ImageOwnership, ImageOwnershipLayouts, ImageOwnershipTransfer,
        ImageQueueOwnershipRelease, NodeIndex, PipelineStageAccessFlags, PreparedStreamRecording,
        QueueSubmitInfo, RecordSelection, RecordedSubmission, RecordedSubmissionState,
        RecordingOwnership, ResourceSetSynchronization, Schedule, SemaphoreSubmitInfo, Submission,
        SubresourceAccess, SubresourceRange, fuzz,
    };

    use crate::{
        AnyResource, Attachment, ColorAttachment, ColorResolve, CommandExecution,
        DepthStencilAttachment, DepthStencilResolve, Execution, Graph, LoadOp, Node, StoreOp,
        TimestampQuery,
        cmd::GraphicsCommandRef,
        driver::{
            DriverError, SharingMode,
            accel_struct::{AccelerationStructure, AccelerationStructureInfo},
            ash::vk,
            buffer::{Buffer, BufferInfo, BufferSubresourceRange},
            cmd_buf::{CommandBuffer, CommandBufferInfo},
            compute::{ComputePipeline, ComputePipelineInfo},
            descriptor_set::{DescriptorSet, DescriptorSetInfo, DescriptorSetUpdateInfo},
            device::{Device, DeviceInfo},
            fence::Fence,
            graphics::{DepthStencilInfo, GraphicsPipeline, GraphicsPipelineInfo},
            image::{Image, ImageAccessSet, ImageInfo, SampleCount},
            instance::Instance,
            render_pass::{RenderPassInfo, SubpassDependency, SubpassInfo},
        },
        node::{AnyBufferNode, AnyImageNode, AnyNode, BufferNode},
        pool::{Pool, hash::HashPool},
        resource::{
            AccelerationStructureAccessType, AccelerationStructureSet, ImageAccessType, ImageSet,
            PhysicalImageId, ResourceSetIndex,
        },
        stream::{BufferArg, CommandStream, CommandStreamDraft, ImageArg, StreamValueArg},
    };

    use {
        ash::vk::Handle,
        std::{
            env::set_var,
            ops::Deref,
            sync::{Arc, Mutex, MutexGuard, OnceLock},
            time::{Duration, Instant},
        },
        vk_shader_macros::glsl,
        vk_sync::{AccessType, BufferBarrier, GlobalBarrier},
    };

    thread_local! {
        static FAIL_QUEUE_SUBMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    thread_local! {
        // Inspect emitted scopes as well as pixels: broad external subpass fallbacks can
        // hide a missing pre-pass barrier from synchronization validation on some layers.
        pub(super) static INCOMING_BUFFER_BARRIERS: std::cell::RefCell<Option<Vec<(
            vk::PipelineStageFlags,
            vk::PipelineStageFlags,
            vk::BufferMemoryBarrier<'static>,
        )>>> = const { std::cell::RefCell::new(None) };
    }

    type SubpassArgs = (ImageArg, Vec<(BufferArg, ImageArg)>, StreamValueArg<u32>);

    pub(super) struct SubpassFixture {
        pipelines: [GraphicsPipeline; 2],
        uniforms: Vec<Arc<Buffer>>,
        textures: Vec<Arc<Image>>,
        supplied: Vec<DescriptorSet>,
        target_info: ImageInfo,
    }

    impl SubpassFixture {
        fn new(device: &Device, pool: &mut HashPool, n: usize) -> Result<Self, DriverError> {
            assert!((1..=1024).contains(&n));
            let vertex = glsl!(kind: vert, r#"
            #version 450
            void main() {
                vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
            }
        "#);
            let fragment = glsl!(kind: frag, r#"
            #version 450
            layout(set = 0, binding = 0, std140) uniform Color { uvec4 value; } ubo;
            layout(set = 1, binding = 0) uniform sampler2D tex;
            layout(push_constant) uniform State { uint salt; } state;
            layout(location = 0) out vec4 color;
            void main() {
                uvec3 sampled = uvec3(round(texelFetch(tex, ivec2(0), 0).rgb * 255.0));
                uvec3 rgb = (ubo.value.rgb + sampled
                    + uvec3(state.salt, state.salt * 3, state.salt * 5)) & uvec3(255);
                color = vec4(vec3(rgb) / 255.0, 1.0);
            }
        "#);
            let pipeline = || {
                GraphicsPipeline::create(
                    device,
                    GraphicsPipelineInfo::builder()
                        .cull_mode(vk::CullModeFlags::NONE)
                        .bindless_descriptor_count(1),
                    [vertex.as_slice(), fragment.as_slice()],
                )
            };
            let pipelines = [pipeline()?, pipeline()?];
            let mut uniforms = Vec::with_capacity(n);
            let mut textures = Vec::with_capacity(n);
            let mut supplied = Vec::with_capacity(n);
            let mut upload = Graph::new();
            for index in 0..n {
                let uniform = Arc::new(Buffer::create_from_slice(
                    device,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    bytemuck::cast_slice(&Self::uniform_color(index)),
                )?);
                supplied.push(DescriptorSet::alloc_and_update(
                    &pipelines[0],
                    DescriptorSetInfo { set: 0 },
                    DescriptorSetUpdateInfo::buffer(0, &uniform),
                )?);
                uniforms.push(uniform);
                let texture = Arc::new(Image::create(
                    device,
                    ImageInfo::image_2d(
                        1,
                        1,
                        vk::Format::R8G8B8A8_UNORM,
                        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                    ),
                )?);
                let staging = upload.bind_resource(Buffer::create_from_slice(
                    device,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    &Self::texture_color(index),
                )?);
                let image = upload.bind_resource(&texture);
                upload.copy_buffer_to_image(staging, image);
                textures.push(texture);
            }
            // Complete uploads and sampled-layout transitions outside all measured render work.
            let mut ready = upload.begin_cmd();
            for texture in &textures {
                let node = ready.bind_resource(texture);
                ready.set_resource_access(
                    node,
                    AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                );
            }
            ready.record_cmd(|_| {});
            upload.finalize().queue_submit(pool, 0, 0)?.wait()?;
            let columns = n.min(32);
            Ok(Self {
                pipelines,
                uniforms,
                textures,
                supplied,
                target_info: ImageInfo::image_2d(
                    columns as u32 * 2,
                    n.div_ceil(columns) as u32 * 2,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                ),
            })
        }

        fn assert_attachment_read_stage_mappings(dep: &SubpassDependency) {
            if dep
                .src_access_mask
                .contains(vk::AccessFlags::COLOR_ATTACHMENT_READ)
            {
                assert!(
                    dep.src_stage_mask
                        .contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
                    "COLOR_ATTACHMENT_READ source access should use COLOR_ATTACHMENT_OUTPUT: {dep:?}"
                );
            }

            if dep
                .dst_access_mask
                .contains(vk::AccessFlags::COLOR_ATTACHMENT_READ)
            {
                assert!(
                    dep.dst_stage_mask
                        .contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
                    "COLOR_ATTACHMENT_READ destination access should use COLOR_ATTACHMENT_OUTPUT: {dep:?}"
                );
            }

            let fragment_tests = vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;

            if dep
                .src_access_mask
                .contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
            {
                assert!(
                    dep.src_stage_mask.intersects(fragment_tests),
                    "DEPTH_STENCIL_ATTACHMENT_READ source access should use fragment-test stages: {dep:?}"
                );
            }

            if dep
                .dst_access_mask
                .contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
            {
                assert!(
                    dep.dst_stage_mask.intersects(fragment_tests),
                    "DEPTH_STENCIL_ATTACHMENT_READ destination access should use fragment-test stages: {dep:?}"
                );
            }
        }

        fn assert_no_invalid_attachment_stage_access_pairs(dep: &SubpassDependency) {
            let dst_invalid_color_stages = dep.dst_stage_mask
                & (vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::FRAGMENT_SHADER);
            assert!(
                !dep.dst_access_mask
                    .contains(vk::AccessFlags::COLOR_ATTACHMENT_READ)
                    || dst_invalid_color_stages.is_empty(),
                "COLOR_ATTACHMENT_READ must not be paired with unsupported destination stages: {dep:?}"
            );

            let src_invalid_color_stages = dep.src_stage_mask
                & (vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::FRAGMENT_SHADER);
            assert!(
                !dep.src_access_mask
                    .contains(vk::AccessFlags::COLOR_ATTACHMENT_READ)
                    || src_invalid_color_stages.is_empty(),
                "COLOR_ATTACHMENT_READ must not be paired with unsupported source stages: {dep:?}"
            );

            assert!(
                !(dep
                    .src_access_mask
                    .contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
                    && dep
                        .src_stage_mask
                        .contains(vk::PipelineStageFlags::FRAGMENT_SHADER)),
                "DEPTH_STENCIL_ATTACHMENT_READ must not be paired with FRAGMENT_SHADER in source stages: {dep:?}"
            );
            assert!(
                !(dep
                    .dst_access_mask
                    .contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
                    && dep
                        .dst_stage_mask
                        .contains(vk::PipelineStageFlags::FRAGMENT_SHADER)),
                "DEPTH_STENCIL_ATTACHMENT_READ must not be paired with FRAGMENT_SHADER in destination stages: {dep:?}"
            );
        }

        fn check(&self, output: &Buffer, shift: usize, salt: u32) {
            let n = self.uniforms.len();
            let pixels = Buffer::mapped_slice(output);
            for tile in 0..n {
                let expected = Self::expected(
                    (tile + shift) % n,
                    (tile * 13 + shift * 7) % n,
                    salt + tile as u32,
                );
                for dy in 0..2 {
                    for dx in 0..2 {
                        let x = tile % n.min(32) * 2 + dx;
                        let y = tile / n.min(32) * 2 + dy;
                        let offset = (y * self.target_info.width as usize + x) * 4;
                        assert_eq!(
                            &pixels[offset..offset + 4],
                            &expected,
                            "tile={tile} pixel=({x},{y}) shift={shift} salt={salt}"
                        );
                    }
                }
            }
        }

        pub(super) fn check_incoming_buffer_source_scope(
            scope: fn(AccessType, vk::QueueFlags) -> (vk::PipelineStageFlags, vk::AccessFlags),
        ) {
            static CHECKED: OnceLock<()> = OnceLock::new();
            CHECKED.get_or_init(|| {
                for producer in [
                    AccessType::ComputeShaderWrite,
                    AccessType::AccelerationStructureBuildWrite,
                ] {
                    assert_eq!(
                        scope(producer, vk::QueueFlags::GRAPHICS),
                        (
                            vk::PipelineStageFlags::ALL_COMMANDS,
                            vk::AccessFlags::MEMORY_WRITE
                        ),
                        "nonlocal producer on a graphics-only queue"
                    );
                    assert_eq!(
                        scope(producer, vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE),
                        crate::driver::pipeline_stage_access_flags(producer),
                        "supported same-queue producers must retain their scope"
                    );
                }
                assert_eq!(
                    scope(AccessType::ComputeShaderReadOther, vk::QueueFlags::GRAPHICS),
                    (
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::AccessFlags::empty()
                    ),
                    "a nonlocal WAR hazard still needs an execution scope"
                );
                assert_eq!(
                    scope(AccessType::TransferWrite, vk::QueueFlags::GRAPHICS),
                    (
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::TRANSFER_WRITE
                    ),
                    "graphics queues implicitly support transfer"
                );
            });
        }

        fn color_attachment_exec(load: LoadOp<[f32; 4]>) -> Execution {
            let mut exec = Execution::default();
            exec.attachments.color.push(Some(ColorAttachment {
                attachment: Attachment {
                    array_layer_count: 1,
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_array_layer: 0,
                    base_mip_level: 0,
                    format: vk::Format::R8G8B8A8_UNORM,
                    mip_level_count: 1,
                    sample_count: SampleCount::Type1,
                    target: 1,
                },
                load,
                store: StoreOp::Store,
                resolve: None,
                is_input: false,
                is_attachment: true,
            }));
            exec
        }

        fn depth_attachment_dependencies(
            previous_load: LoadOp<vk::ClearDepthStencilValue>,
            previous_store: StoreOp,
            current_load: LoadOp<vk::ClearDepthStencilValue>,
            current_store: StoreOp,
        ) -> Vec<SubpassDependency> {
            let pass = CommandData {
                execs: vec![
                    SubpassFixture::depth_attachment_exec(previous_load, previous_store),
                    SubpassFixture::depth_attachment_exec(current_load, current_store),
                ],

                #[cfg(debug_assertions)]
                name: None,

                stream_scope_id: None,
                tracking: Default::default(),
            };

            Submission::build_subpass_dependencies(
                &pass,
                &[PipelineStageAccessFlags::default(); 1],
                &[0, 1],
            )
        }

        fn depth_attachment_exec(
            load: LoadOp<vk::ClearDepthStencilValue>,
            store: StoreOp,
        ) -> Execution {
            let mut exec = Execution::default();
            exec.attachments.depth_stencil = Some(DepthStencilAttachment {
                attachment: Attachment {
                    array_layer_count: 1,
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_array_layer: 0,
                    base_mip_level: 0,
                    format: vk::Format::D32_SFLOAT,
                    mip_level_count: 1,
                    sample_count: SampleCount::Type1,
                    target: 0,
                },
                load,
                store,
                resolve: None,
                is_attachment: true,
            });

            exec
        }

        fn draft(&self) -> CommandStreamDraft<SubpassArgs> {
            CommandStream::finalize(|stream| {
                let target = stream.arg(self.target_info);
                let args = (0..self.uniforms.len())
                    .map(|index| {
                        (
                            stream.arg(self.uniforms[index].info),
                            stream.arg(self.textures[index].info),
                        )
                    })
                    .collect::<Vec<_>>();
                let salt = stream.add_value_arg::<u32>();
                let inputs = args
                    .iter()
                    .map(|&(ubo, tex)| (ubo.into(), tex.into()))
                    .collect::<Vec<_>>();
                self.draws(
                    &mut stream.graph,
                    target.into(),
                    &inputs,
                    false,
                    Some(salt),
                    false,
                );
                (target, args, salt)
            })
        }

        fn draw(cmd: GraphicsCommandRef<'_>, tile: usize, columns: usize, salt: u32) {
            let x = (tile % columns * 2) as i32;
            let y = (tile / columns * 2) as i32;
            cmd.set_viewport(
                0,
                &[vk::Viewport {
                    x: x as f32,
                    y: y as f32,
                    width: 2.0,
                    height: 2.0,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            )
            .set_scissor(
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x, y },
                    extent: vk::Extent2D {
                        width: 2,
                        height: 2,
                    },
                }],
            )
            .push_constants(0, &salt.to_ne_bytes())
            .draw(3, 1, 0, 0);
        }

        fn draws(
            &self,
            graph: &mut Graph,
            target: AnyImageNode,
            inputs: &[(AnyBufferNode, AnyImageNode)],
            supplied: bool,
            salt: Option<StreamValueArg<u32>>,
            instrument: bool,
        ) -> (Vec<CommandExecution>, Vec<TimestampQuery>) {
            let columns = self.uniforms.len().min(32);
            let mut tracking = Vec::new();
            let mut timestamps = Vec::new();
            for (tile, &(uniform, texture)) in inputs.iter().enumerate() {
                let mut command = graph.begin_cmd();
                if instrument {
                    tracking.push(command.track_execution());
                    timestamps.push(command.write_timestamp());
                }
                let mut command = command
                    .bind_pipeline(&self.pipelines[if supplied { tile % 2 } else { 0 }])
                    .color_attachment_image(
                        0,
                        target,
                        if tile == 0 {
                            LoadOp::CLEAR_BLACK_ALPHA_ZERO
                        } else {
                            LoadOp::Load
                        },
                        StoreOp::Store,
                    )
                    .shader_resource_access(
                        (1, 0),
                        texture,
                        AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                    );
                if supplied {
                    command = command
                        .bind_descriptor_set(&self.supplied[tile])
                        .resource_access(uniform, AccessType::FragmentShaderReadUniformBuffer);
                } else {
                    command = command.shader_resource_access(
                        (0, 0),
                        uniform,
                        AccessType::FragmentShaderReadUniformBuffer,
                    );
                }
                command.record_stream_mut(move |cmd| {
                    let salt = salt.map_or(17, |arg| cmd.value(arg)) + tile as u32;
                    Self::draw(cmd, tile, columns, salt);
                });
                if instrument {
                    timestamps.push(command.write_timestamp());
                }
            }
            (tracking, timestamps)
        }

        fn exec_with_buffer_access(access: AccessType) -> Execution {
            let mut exec = Execution::default();
            exec.accesses.push(
                0,
                SubresourceAccess {
                    access,
                    subresource: SubresourceRange::Buffer((0..16).into()),
                },
            );

            exec
        }

        fn expected(uniform: usize, texture: usize, salt: u32) -> [u8; 4] {
            let ubo = Self::uniform_color(uniform);
            let tex = Self::texture_color(texture);
            [
                (ubo[0] + u32::from(tex[0]) + salt) as u8,
                (ubo[1] + u32::from(tex[1]) + salt * 3) as u8,
                (ubo[2] + u32::from(tex[2]) + salt * 5) as u8,
                255,
            ]
        }

        fn invoke(
            &self,
            graph: &mut Graph,
            stream: &CommandStream<SubpassArgs>,
            target: &Arc<Image>,
            shift: usize,
            salt: u32,
        ) {
            let n = self.uniforms.len();
            let target = graph.bind_resource(target);
            let inputs = (0..n)
                .map(|tile| {
                    (
                        graph.bind_resource(&self.uniforms[(tile + shift) % n]),
                        graph.bind_resource(&self.textures[(tile * 13 + shift * 7) % n]),
                    )
                })
                .collect::<Vec<_>>();
            let mut run = graph
                .insert_cmd_stream(stream)
                .with_arg(stream.args.0, target)
                .with_value(stream.args.2, salt);
            for (&(ubo_arg, tex_arg), (ubo, tex)) in stream.args.1.iter().zip(inputs) {
                run = run.with_arg(ubo_arg, ubo).with_arg(tex_arg, tex);
            }
            run.finish();
        }

        fn output(&self, device: &Device) -> Result<(Arc<Image>, Arc<Buffer>), DriverError> {
            Ok((
                Arc::new(Image::create(device, self.target_info)?),
                Arc::new(Buffer::create(
                    device,
                    BufferInfo::host_mem(
                        u64::from(self.target_info.width * self.target_info.height * 4),
                        vk::BufferUsageFlags::TRANSFER_DST,
                    ),
                )?),
            ))
        }

        fn plan_subpasses(pass: &CommandData) -> (RenderPassInfo, Box<[u32]>) {
            let inputs = pass
                .execs
                .iter()
                .map(|exec| {
                    exec.attachments
                        .color_attachments()
                        .filter_map(|(idx, state)| state.is_input.then_some(idx))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let graphics = inputs
                .iter()
                .map(|inputs| GraphicsExecutionInfo {
                    input_attachments: inputs,
                    sample_count: SampleCount::Type1,
                })
                .collect::<Vec<_>>();
            let node_count = pass
                .execs
                .iter()
                .flat_map(|exec| {
                    exec.accesses
                        .iter()
                        .map(|(node, _)| node)
                        .chain(
                            exec.attachments
                                .color_attachments()
                                .map(|(_, state)| state.attachment.target),
                        )
                        .chain(
                            exec.attachments
                                .depth_stencil_attachment()
                                .map(|state| state.attachment.target),
                        )
                        .chain(
                            exec.attachments
                                .depth_stencil_attachment()
                                .and_then(|state| state.resolve)
                                .map(|resolve| resolve.attachment.target),
                        )
                })
                .max()
                .map_or(0, |node| node + 1);
            Submission::build_render_pass_info(
                pass,
                &vec![PipelineStageAccessFlags::default(); node_count],
                &graphics,
            )
        }

        fn readback(graph: &mut Graph, target: AnyImageNode, output: &Arc<Buffer>) {
            let output = graph.bind_resource(output);
            graph.copy_image_to_buffer(target, output);
            graph
                .begin_cmd()
                .resource_access(output, AccessType::HostRead)
                .record_cmd(|_| {});
        }

        fn subpass_command(execs: Vec<Execution>) -> CommandData {
            let mut cmd = command_with_accesses(&[]);
            cmd.execs = execs;
            cmd
        }

        fn subpass_counts(submission: &Submission) -> Vec<usize> {
            submission
                .recorded_commands
                .iter()
                .filter_map(|command| command.render_pass.as_ref())
                .map(|render_pass| render_pass.info.subpasses.len())
                .collect()
        }

        fn subpass_dependencies_for_accesses(
            previous: AccessType,
            current: AccessType,
        ) -> Vec<SubpassDependency> {
            let pass = CommandData {
                execs: vec![
                    SubpassFixture::exec_with_buffer_access(previous),
                    SubpassFixture::exec_with_buffer_access(current),
                ],

                #[cfg(debug_assertions)]
                name: None,

                stream_scope_id: None,
                tracking: Default::default(),
            };

            Submission::build_subpass_dependencies(
                &pass,
                &[PipelineStageAccessFlags::default(); 1],
                &[0, 1],
            )
        }

        fn test_input_attachment_pipelines(
            device: &Device,
        ) -> Result<(GraphicsPipeline, GraphicsPipeline), DriverError> {
            let vertex = glsl!(
                r#"
            #version 460 core
            #pragma shader_stage(vertex)

            vec2 POSITIONS[3] = vec2[](
                vec2(-1.0, -1.0),
                vec2(3.0, -1.0),
                vec2(-1.0, 3.0)
            );

            void main() {
                gl_Position = vec4(POSITIONS[gl_VertexIndex], 0.0, 1.0);
            }
            "#
            );
            let pipeline_a = GraphicsPipeline::create(
                device,
                GraphicsPipelineInfo::default(),
                [
                    vertex.as_slice(),
                    glsl!(
                        kind: frag,
                        r#"
                    #version 460 core
                    #pragma shader_stage(fragment)

                    layout(location = 0) out vec4 color_out;

                    void main() {
                        color_out = vec4(0.25, 0.5, 0.75, 1.0);
                    }
                    "#
                    )
                    .as_slice(),
                ],
            )?;
            let pipeline_b = GraphicsPipeline::create(
                device,
                GraphicsPipelineInfo::default(),
                [
                    vertex.as_slice(),
                    glsl!(
                        kind: frag,
                        r#"
                    #version 460 core
                    #pragma shader_stage(fragment)

                    layout(input_attachment_index = 0, binding = 0) uniform subpassInput color_in;
                    layout(location = 0) out vec4 color_out;

                    void main() {
                        color_out = subpassLoad(color_in);
                    }
                    "#
                    )
                    .as_slice(),
                ],
            )?;

            Ok((pipeline_a, pipeline_b))
        }

        fn test_triangle_pipeline(device: &Device) -> Result<GraphicsPipeline, DriverError> {
            GraphicsPipeline::create(
                device,
                GraphicsPipelineInfo::default(),
                [
                    glsl!(
                        r#"
                    #version 460 core
                    #pragma shader_stage(vertex)

                    vec2 POSITIONS[3] = vec2[](
                        vec2(-1.0, -1.0),
                        vec2(3.0, -1.0),
                        vec2(-1.0, 3.0)
                    );

                    void main() {
                        gl_Position = vec4(POSITIONS[gl_VertexIndex], 0.0, 1.0);
                    }
                    "#
                    )
                    .as_slice(),
                    glsl!(
                        r#"
                    #version 460 core
                    #pragma shader_stage(fragment)

                    layout(location = 0) out vec4 vk_Color;

                    void main() {
                        vk_Color = vec4(1.0, 0.0, 0.0, 1.0);
                    }
                    "#
                    )
                    .as_slice(),
                ],
            )
        }

        fn texture_color(index: usize) -> [u8; 4] {
            [
                (index * 29) as u8,
                (index * 7) as u8,
                (index / 256 * 53) as u8,
                255,
            ]
        }

        fn uniform_color(index: usize) -> [u32; 4] {
            [index as u32 & 255, (index as u32 >> 8) * 47, 19, 0]
        }
    }

    #[derive(Debug)]
    struct TestDevice<'a, T = Device, F: Fn() -> usize = fn() -> usize> {
        guard: Option<MutexGuard<'a, ()>>,
        device: Option<T>,
        validation: Option<(F, usize)>,
    }

    impl TestDevice<'static> {
        fn new() -> Result<TestDevice<'static>, DriverError> {
            let guard = TestDevice::lock()
                .lock()
                .expect("poisoned test device lock");

            TestDevice::create(guard, None, || Device::create(DeviceInfo::default()))
        }

        // All validation-enabled unit-test devices must use this lock: the counter is process-global.
        fn new_debug() -> Result<TestDevice<'static>, DriverError> {
            let guard = TestDevice::lock()
                .lock()
                .expect("poisoned test device lock");

            TestDevice::create(guard, Some(Instance::validation_error_count), || {
                Device::create(DeviceInfo::builder().debug(true).build())
            })
        }

        fn init_validation_test_logging() {
            static INIT: OnceLock<()> = OnceLock::new();

            INIT.get_or_init(|| {
                unsafe {
                    if std::env::var_os("RUST_LOG").is_none() {
                        set_var("RUST_LOG", "trace");
                    }
                    set_var("VK_GRAPH_SKIP_VALIDATION_PARK", "1");
                }

                let _ = pretty_env_logger::try_init();
            });
        }

        fn lock() -> &'static Mutex<()> {
            static LOCK: OnceLock<Mutex<()>> = OnceLock::new();

            LOCK.get_or_init(|| Mutex::new(()))
        }
    }

    impl<'a, T, F: Fn() -> usize> TestDevice<'a, T, F> {
        fn create(
            guard: MutexGuard<'a, ()>,
            validation_error_count: Option<F>,
            create: impl FnOnce() -> Result<T, DriverError>,
        ) -> Result<Self, DriverError> {
            let validation = validation_error_count.map(|count| {
                let baseline = count();
                (count, baseline)
            });
            let device = create()?;

            Ok(Self {
                guard: Some(guard),
                device: Some(device),
                validation,
            })
        }
    }

    impl<T, F: Fn() -> usize> Deref for TestDevice<'_, T, F> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            self.device.as_ref().unwrap()
        }
    }

    impl<T, F: Fn() -> usize> Drop for TestDevice<'_, T, F> {
        fn drop(&mut self) {
            let before = self.validation.as_ref().map(|(count, _)| count());
            // Taking the device also prevents a second drop if its destructor panics.
            drop(self.device.take());
            let after = self.validation.as_ref().map(|(count, _)| count());
            // Snapshots and teardown belong to this session; assertions must not poison the lock.
            drop(self.guard.take());

            if !std::thread::panicking()
                && let Some((_, baseline)) = &self.validation
            {
                assert_eq!(before, Some(*baseline), "Vulkan validation ERRORs occurred");
                assert_eq!(
                    after,
                    Some(*baseline),
                    "Vulkan validation ERRORs occurred during teardown"
                );
            }
        }
    }

    fn color_subresource_range(
        array_layers: std::ops::Range<u32>,
        mip_levels: std::ops::Range<u32>,
    ) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_array_layer: array_layers.start,
            layer_count: array_layers.end - array_layers.start,
            base_mip_level: mip_levels.start,
            level_count: mip_levels.end - mip_levels.start,
        }
    }

    fn command_with_accesses(accesses: &[(usize, AccessType)]) -> CommandData {
        let mut exec = Execution::default();

        for &(node_idx, access) in accesses {
            exec.accesses.push(
                node_idx,
                SubresourceAccess {
                    access,
                    subresource: SubresourceRange::Buffer(BufferSubresourceRange {
                        start: 0,
                        end: 1,
                    }),
                },
            );
        }

        CommandData {
            execs: vec![exec],

            #[cfg(debug_assertions)]
            name: None,

            stream_scope_id: None,
            tracking: Default::default(),
        }
    }

    fn command_with_resource_set_accesses(executions: &[&[usize]]) -> CommandData {
        let execs = executions
            .iter()
            .map(|resource_set_indices| {
                let mut exec = Execution::default();

                for &resource_set_idx in *resource_set_indices {
                    exec.push_resource_set_access(crate::ResourceSetAccess {
                        resource_set_idx: ResourceSetIndex::new(resource_set_idx),
                        access_type: crate::resource::ResourceSetAccessType::Image(
                            crate::resource::ImageAccessType::SampledRead,
                        ),
                    });
                }

                exec
            })
            .collect();

        CommandData {
            execs,

            #[cfg(debug_assertions)]
            name: None,

            stream_scope_id: None,
            tracking: Default::default(),
        }
    }

    // Inject at the Vulkan call boundary, before any success publication or fence attachment.
    pub(super) fn fail_queue_submit() -> Result<(), DriverError> {
        if FAIL_QUEUE_SUBMIT.replace(false) {
            Err(DriverError::InvalidData)
        } else {
            Ok(())
        }
    }

    fn pending_buffer_transfer_for_range(
        transfers: &[BufferQueueOwnershipTransfer],
        range: BufferSubresourceRange,
    ) -> Option<&BufferQueueOwnershipTransfer> {
        transfers.iter().find(|transfer| transfer.range == range)
    }

    fn pending_timestamp_query_pool(query: TimestampQuery) -> super::TimestampQueryPool {
        #[cfg(feature = "checked")]
        {
            super::TimestampQueryPool::pending(query.graph_id())
        }

        #[cfg(not(feature = "checked"))]
        {
            let _ = query;
            super::TimestampQueryPool::pending()
        }
    }

    fn pending_transfer_for_node<H: Copy, T>(
        pending: &super::PendingTransferNodes<H, T>,
        node_idx: NodeIndex,
    ) -> Option<(H, &[T])> {
        pending
            .iter()
            .find_map(|(idx, handle, transfers)| (idx == node_idx).then_some((handle, transfers)))
    }

    fn sampled_read_barrier(
        previous_access: AccessType,
        next_access: AccessType,
    ) -> super::TrackedImageBarrier {
        let previous_accesses = ImageAccessSet::from_access(previous_access);

        super::TrackedImageBarrier {
            previous_accesses,
            next_access,
            previous_layout: super::TrackedImageBarrier::access_set_layout(previous_accesses),
            next_layout: super::TrackedImageBarrier::access_layout(next_access),
            ownership_layouts: None,
            discard_contents: false,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: vk::Image::null(),
            range: color_subresource_range(0..1, 0..1),
        }
    }

    fn schedule_with_access_index(
        cmds: &[usize],
        cmds_by_node: &[&[usize]],
        accessed_nodes_by_cmd: &[&[usize]],
    ) -> Schedule {
        Schedule {
            access_index: CommandAccessIndex {
                cmds_by_node: cmds_by_node.iter().map(|cmds| cmds.to_vec()).collect(),
                accessed_nodes_by_cmd: accessed_nodes_by_cmd
                    .iter()
                    .map(|nodes| nodes.to_vec())
                    .collect(),
                ..Default::default()
            },
            cmds: cmds.to_vec(),
            ..Default::default()
        }
    }

    fn simulate_partial_transfer_discovery(
        submission: &mut Submission,
        schedule: &Schedule,
        queue_family_index: u32,
        ownership: &mut RecordingOwnership,
    ) {
        submission.track_pending_transfers(
            schedule,
            queue_family_index,
            ownership,
            ResourceSetSynchronization::Enabled,
        );
        submission.pending_buffer_transfer_nodes = None;
        submission.pending_image_transfer_nodes = None;
        submission.pending_image_set_transfers.clear();
    }

    #[cfg(test)]
    fn sort_image_subresource_ranges(ranges: &mut [vk::ImageSubresourceRange]) {
        ranges.sort_unstable_by_key(|range| {
            (
                range.aspect_mask.as_raw(),
                range.base_array_layer,
                range.layer_count,
                range.base_mip_level,
                range.level_count,
            )
        });
    }

    #[cfg(test)]
    fn sort_image_subresource_sync_infos(
        subresources: &mut [crate::driver::image::ImageSubresourceSyncInfo],
    ) {
        subresources.sort_unstable_by_key(|subresource| {
            (
                subresource.range.aspect_mask.as_raw(),
                subresource.range.base_array_layer,
                subresource.range.layer_count,
                subresource.range.base_mip_level,
                subresource.range.level_count,
            )
        });
    }

    #[cfg(test)]
    fn sort_pending_buffer_transfers(transfers: &mut [BufferQueueOwnershipTransfer]) {
        transfers.sort_unstable_by_key(|transfer| {
            (
                transfer.src_queue_family_index,
                transfer.dst_queue_family_index,
                transfer.range.start,
                transfer.range.end,
            )
        });
    }

    #[cfg(test)]
    fn sort_pending_image_transfers(transfers: &mut [ImageOwnershipTransfer]) {
        transfers.sort_unstable_by_key(|transfer| {
            (
                transfer.src_queue_family_index,
                transfer.src_queue_index,
                transfer.dst_queue_family_index,
                transfer.layouts.old.as_raw(),
                transfer.layouts.new.as_raw(),
                transfer.range.aspect_mask.as_raw(),
                transfer.range.base_array_layer,
                transfer.range.layer_count,
                transfer.range.base_mip_level,
                transfer.range.level_count,
            )
        });
    }

    #[cfg(test)]
    fn sort_queue_ownership_release_groups(groups: &mut [super::QueueOwnershipReleaseGroup]) {
        for group in groups.iter_mut() {
            group
                .buffers
                .sort_unstable_by_key(|(buffer, range)| (buffer.as_raw(), range.start, range.end));

            group.images.sort_unstable_by_key(|release| {
                (
                    release.image.as_raw(),
                    release.layouts.old.as_raw(),
                    release.layouts.new.as_raw(),
                    release.range.aspect_mask.as_raw(),
                    release.range.base_array_layer,
                    release.range.layer_count,
                    release.range.base_mip_level,
                    release.range.level_count,
                )
            });
        }

        groups.sort_unstable_by_key(|group| (group.src_queue_family_index, group.src_queue_index));
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn accel_struct_mixed_accesses_preserve_all_stage_bits() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let mut pool = HashPool::new(&device);
        let mut graph = Graph::new();
        let accel_struct = graph.bind_resource(AccelerationStructure::create(
            &device,
            AccelerationStructureInfo::blas(1024),
        )?);

        graph
            .begin_cmd()
            .debug_name("mixed accel struct accesses")
            .resource_access(accel_struct, AccessType::AccelerationStructureBuildRead)
            .resource_access(
                accel_struct,
                AccessType::RayTracingShaderReadAccelerationStructure,
            )
            .record_cmd(|_| {});

        let submission = graph.finalize();
        let mut cmd_buf = pool.resource(CommandBufferInfo::new(0))?;

        cmd_buf.begin(
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let recording = submission.record(&mut pool, &mut cmd_buf, RecordSelection::All)?;
        let sync_info = recording.resource(accel_struct).sync_info();

        assert!(
            sync_info
                .stage_mask
                .contains(vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR),
            "sync info should preserve build-read stage bits"
        );
        assert!(
            sync_info
                .stage_mask
                .contains(vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR),
            "sync info should preserve ray-tracing-read stage bits"
        );
        assert_eq!(
            sync_info.access_mask,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
            "mixed read-only accesses should stay read-only"
        );

        Ok(())
    }

    #[test]
    fn acceleration_structure_writes_have_source_and_destination_accesses() {
        let (src_stage_mask, dst_stage_mask, barrier) =
            vk_sync::get_memory_barrier(&vk_sync::GlobalBarrier {
                previous_accesses: &[AccessType::AccelerationStructureBuildWrite],
                next_accesses: &[AccessType::RayTracingShaderReadAccelerationStructure],
            });

        assert_eq!(
            src_stage_mask,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
        );
        assert_eq!(
            barrier.src_access_mask,
            vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
        );
        assert_eq!(
            dst_stage_mask,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR
        );
        assert_eq!(
            barrier.dst_access_mask,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
        );
    }

    #[test]
    fn attachment_shader_feedback_is_excluded_even_on_a_disjoint_subresource() {
        for range in [
            color_subresource_range(0..1, 0..1),
            color_subresource_range(0..1, 1..2),
        ] {
            let mut pass = SubpassFixture::subpass_command(vec![
                    SubpassFixture::color_attachment_exec(
                        LoadOp::Load
                    );
                    3
                ]);
            pass.execs[1].accesses.push(
                1,
                SubresourceAccess {
                    access: AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                    subresource: SubresourceRange::Image(range),
                },
            );
            assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 1, 2]);
        }
    }

    #[test]
    fn barrier_transfer_ranges_only_marks_overlapping_ranges() {
        let range_a = color_subresource_range(0..1, 0..1);
        let range_b = color_subresource_range(1..2, 0..1);
        let transfers = [ImageOwnershipTransfer {
            src_queue_family_index: 1,
            src_queue_index: 2,
            dst_queue_family_index: 3,
            layouts: ImageOwnershipLayouts {
                old: vk::ImageLayout::GENERAL,
                new: vk::ImageLayout::GENERAL,
            },
            range: range_a,
        }];

        let ranges =
            ImageOwnershipTransfer::barrier_ranges(&transfers, color_subresource_range(0..2, 0..1))
                .collect::<Vec<_>>();

        assert_eq!(ranges.len(), 2);
        assert!(ImageOwnershipTransfer::ranges_equal(ranges[0].0, range_a));
        assert_eq!(
            ranges[0].1.map(|transfer| (
                transfer.src_queue_family_index,
                transfer.src_queue_index,
                transfer.dst_queue_family_index,
            )),
            Some((1, 2, 3))
        );
        assert!(ImageOwnershipTransfer::ranges_equal(ranges[1].0, range_b));
        assert!(ranges[1].1.is_none());
    }

    #[test]
    fn buffer_acquires_use_queue_safe_sources_and_retain_destination_visibility() {
        let queue_flags = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        for previous in [
            AccessType::Nothing,
            AccessType::FragmentShaderReadOther,
            AccessType::FragmentShaderWrite,
            AccessType::VertexShaderWrite,
            AccessType::HostWrite,
        ] {
            for next in [
                AccessType::TransferRead,
                AccessType::ComputeShaderWrite,
                AccessType::VertexShaderReadUniformBuffer,
            ] {
                for acquire in [false, true] {
                    let barrier = BufferBarrier {
                        previous_accesses: std::slice::from_ref(&previous),
                        next_accesses: std::slice::from_ref(&next),
                        src_queue_family_index: if acquire { 1 } else { vk::QUEUE_FAMILY_IGNORED },
                        dst_queue_family_index: if acquire { 0 } else { vk::QUEUE_FAMILY_IGNORED },
                        buffer: vk::Buffer::from_raw(1),
                        offset: 16,
                        size: 32,
                    };
                    let (src, dst, legacy) =
                        super::Submission::buffer_memory_barrier(&barrier, queue_flags);
                    let sync2 = super::Submission::buffer_memory_barrier2(&barrier, queue_flags);
                    assert_eq!(legacy.buffer, barrier.buffer);
                    assert_eq!((legacy.offset, legacy.size), (16, 32));
                    assert_eq!(sync2.buffer, barrier.buffer);
                    assert_eq!((sync2.offset, sync2.size), (16, 32));
                    assert_eq!(
                        legacy.src_queue_family_index,
                        barrier.src_queue_family_index
                    );
                    assert_eq!(
                        legacy.dst_queue_family_index,
                        barrier.dst_queue_family_index
                    );
                    assert_eq!(sync2.src_queue_family_index, barrier.src_queue_family_index);
                    assert_eq!(sync2.dst_queue_family_index, barrier.dst_queue_family_index);
                    if acquire {
                        assert_eq!(src, vk::PipelineStageFlags::TOP_OF_PIPE);
                        assert!(legacy.src_access_mask.is_empty());
                        let (stages, accesses) = crate::driver::pipeline_stage_access_flags(next);
                        assert_eq!(dst, stages | vk::PipelineStageFlags::BOTTOM_OF_PIPE);
                        assert_eq!(legacy.dst_access_mask, accesses);
                        assert!(sync2.src_stage_mask.is_empty());
                        assert!(sync2.src_access_mask.is_empty());
                    } else {
                        let (old_src, old_dst, old) = vk_sync::get_buffer_memory_barrier(&barrier);
                        assert_eq!((src, dst), (old_src, old_dst));
                        assert_eq!(legacy.src_access_mask, old.src_access_mask);
                        assert_eq!(legacy.dst_access_mask, old.dst_access_mask);
                        let (stages, accesses) =
                            crate::driver::micromap::micromap_sync_flags_for_access(previous);
                        assert_eq!(sync2.src_stage_mask, stages);
                        assert_eq!(sync2.src_access_mask, accesses);
                    }
                    let (stages, accesses) =
                        crate::driver::micromap::micromap_sync_flags_for_access(next);
                    assert_eq!(sync2.dst_stage_mask, stages);
                    assert_eq!(sync2.dst_access_mask, accesses);
                }
            }
        }

        let sync2 = super::Submission::buffer_memory_barrier2(
            &BufferBarrier {
                previous_accesses: &[AccessType::FragmentShaderWrite],
                next_accesses: &[AccessType::MicromapBuildInputRead],
                src_queue_family_index: 0,
                dst_queue_family_index: 1,
                ..Default::default()
            },
            vk::QueueFlags::COMPUTE,
        );
        assert!(sync2.src_stage_mask.is_empty());
        assert!(sync2.src_access_mask.is_empty());
        assert_eq!(
            sync2.dst_stage_mask,
            vk::PipelineStageFlags2::MICROMAP_BUILD_EXT
        );
        assert_eq!(sync2.dst_access_mask, vk::AccessFlags2::SHADER_READ);
    }

    #[test]
    fn buffer_writer_retirement_requires_identical_complete_group_scopes_and_ranges() {
        for case in 0..7 {
            for changed in [1, 2] {
                let mut execs =
                    vec![
                        SubpassFixture::exec_with_buffer_access(AccessType::FragmentShaderWrite);
                        4
                    ];
                let exec = &mut execs[changed];
                match case {
                    0 => {
                        exec.accesses.get_mut(&0).unwrap()[0].access = AccessType::VertexShaderWrite
                    }
                    1 => {
                        exec.accesses.get_mut(&0).unwrap()[0].access =
                            AccessType::FragmentShaderReadOther
                    }
                    2..=4 => {
                        exec.accesses.get_mut(&0).unwrap()[0].subresource =
                            SubresourceRange::Buffer(
                                match case {
                                    2 => 0..8,
                                    3 => 8..24,
                                    _ => 16..32,
                                }
                                .into(),
                            );
                    }
                    5 => exec.accesses.push(
                        0,
                        SubresourceAccess {
                            access: AccessType::FragmentShaderWrite,
                            subresource: SubresourceRange::Buffer((16..32).into()),
                        },
                    ),
                    6 => exec.accesses.push(
                        0,
                        SubresourceAccess {
                            access: AccessType::FragmentShaderReadOther,
                            subresource: SubresourceRange::Buffer((0..16).into()),
                        },
                    ),
                    _ => unreachable!(),
                }
                exec.accesses.freeze();
                let deps = Submission::build_subpass_dependencies(
                    &SubpassFixture::subpass_command(execs),
                    &[PipelineStageAccessFlags::default()],
                    &[0, 1, 1, 2],
                );
                let pairs = deps
                    .iter()
                    .filter(|dep| dep.src_subpass != vk::SUBPASS_EXTERNAL)
                    .map(|dep| (dep.src_subpass, dep.dst_subpass))
                    .collect::<Vec<_>>();
                assert_eq!(
                    pairs,
                    [(0, 1), (0, 2), (1, 2)],
                    "case={case} changed={changed}"
                );
                assert!(deps.iter().all(|dep| dep.dependency_flags.is_empty()));
            }
        }

        // Retiring a writer must neither skip an intervening reader nor discard older stages.
        for (accesses, expected) in [
            (
                vec![
                    AccessType::FragmentShaderWrite,
                    AccessType::FragmentShaderWrite,
                    AccessType::FragmentShaderReadOther,
                    AccessType::FragmentShaderWrite,
                    AccessType::IndexBuffer,
                ],
                vec![(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (3, 4)],
            ),
            (
                vec![
                    AccessType::FragmentShaderWrite,
                    AccessType::VertexShaderWrite,
                    AccessType::VertexShaderWrite,
                    AccessType::IndexBuffer,
                ],
                vec![(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)],
            ),
        ] {
            let mapping = (0..accesses.len() as u32).collect::<Vec<_>>();
            let pass = SubpassFixture::subpass_command(
                accesses
                    .into_iter()
                    .map(SubpassFixture::exec_with_buffer_access)
                    .collect(),
            );
            let deps = Submission::build_subpass_dependencies(
                &pass,
                &[PipelineStageAccessFlags::default()],
                &mapping,
            );
            assert_eq!(
                deps.iter()
                    .filter(|dep| dep.src_subpass != vk::SUBPASS_EXTERNAL)
                    .map(|dep| (dep.src_subpass, dep.dst_subpass))
                    .collect::<Vec<_>>(),
                expected
            );
        }

        // Image writers (including local attachments) do not use the buffer-only proof.
        for attachment in [false, true] {
            let exec = if attachment {
                SubpassFixture::color_attachment_exec(LoadOp::Load)
            } else {
                let mut exec =
                    SubpassFixture::exec_with_buffer_access(AccessType::FragmentShaderWrite);
                exec.accesses.get_mut(&0).unwrap()[0].subresource =
                    SubresourceRange::Image(color_subresource_range(0..1, 0..1));
                exec
            };
            let deps = Submission::build_subpass_dependencies(
                &SubpassFixture::subpass_command(vec![exec; 3]),
                &[PipelineStageAccessFlags::default(); 2],
                &[0, 1, 2],
            );
            assert_eq!(deps.len(), 6);
            for dep in deps
                .iter()
                .filter(|dep| dep.src_subpass != vk::SUBPASS_EXTERNAL)
            {
                assert_eq!(
                    dep.dependency_flags,
                    if attachment {
                        vk::DependencyFlags::BY_REGION
                    } else {
                        vk::DependencyFlags::empty()
                    }
                );
            }
        }
    }

    #[test]
    fn build_subpass_dependencies_includes_later_access_stage_bits() {
        let mut exec = Execution::default();

        exec.accesses.push(
            0,
            SubresourceAccess {
                access: AccessType::IndexBuffer,
                subresource: SubresourceRange::Buffer((0..16).into()),
            },
        );
        exec.accesses.push(
            0,
            SubresourceAccess {
                access: AccessType::FragmentShaderReadOther,
                subresource: SubresourceRange::Buffer((0..16).into()),
            },
        );

        let pass = CommandData {
            execs: vec![exec],

            #[cfg(debug_assertions)]
            name: None,

            stream_scope_id: None,
            tracking: Default::default(),
        };
        let dependencies = Submission::build_subpass_dependencies(
            &pass,
            &[PipelineStageAccessFlags::default(); 1],
            &[0],
        );
        let dep = dependencies
            .iter()
            .find(|dep| dep.src_subpass == vk::SUBPASS_EXTERNAL && dep.dst_subpass == 0)
            .expect("missing external dependency for mixed access slice");

        assert!(
            dep.dst_stage_mask
                .contains(vk::PipelineStageFlags::VERTEX_INPUT),
            "first access stage should be preserved"
        );
        assert!(
            dep.dst_stage_mask
                .contains(vk::PipelineStageFlags::FRAGMENT_SHADER),
            "later access stages should also contribute"
        );
    }

    #[test]
    fn coalesces_read_only_graphics_executions_with_stable_color_and_depth() {
        let execs = (0..32)
            .map(|idx| {
                let mut exec = SubpassFixture::color_attachment_exec(if idx == 0 {
                    LoadOp::Clear([0.0; 4])
                } else {
                    LoadOp::Load
                });
                exec.attachments.depth_stencil = SubpassFixture::depth_attachment_exec(
                    if idx == 0 {
                        LoadOp::CLEAR_ONE_STENCIL_ZERO
                    } else {
                        LoadOp::Load
                    },
                    StoreOp::Store,
                )
                .attachments
                .depth_stencil;
                let depth = exec.attachments.depth_stencil.as_mut().unwrap();
                depth.attachment.target = 2;
                exec.accesses = SubpassFixture::exec_with_buffer_access(
                    AccessType::VertexShaderReadUniformBuffer,
                )
                .accesses;
                for (node, aspect, access) in [
                    (
                        1,
                        vk::ImageAspectFlags::COLOR,
                        AccessType::ColorAttachmentReadWrite,
                    ),
                    (
                        2,
                        vk::ImageAspectFlags::DEPTH,
                        AccessType::DepthStencilAttachmentReadWrite,
                    ),
                ] {
                    let mut range = color_subresource_range(0..1, 0..1);
                    range.aspect_mask = aspect;
                    exec.accesses.push(
                        node,
                        SubresourceAccess {
                            access,
                            subresource: SubresourceRange::Image(range),
                        },
                    );
                }
                if idx % 2 == 0 {
                    exec.accesses.freeze();
                }
                exec.render_area = Some(vk::Rect2D::default().extent(vk::Extent2D {
                    width: idx + 1,
                    height: 16,
                }));
                exec.func = Some(crate::CommandFunction::Reusable(Arc::new(|_| {})));
                exec
            })
            .collect();
        let pass = SubpassFixture::subpass_command(execs);
        let (info, mapping) = SubpassFixture::plan_subpasses(&pass);
        assert_eq!(&*mapping, &[0; 32]);
        assert_eq!(info.subpasses.len(), 1);
        assert!(info.subpasses[0].depth_stencil_attachment.is_some());
        assert_eq!(info.attachments[0].load_op, vk::AttachmentLoadOp::CLEAR);
        assert_eq!(info.attachments[0].store_op, vk::AttachmentStoreOp::STORE);
        assert_eq!(pass.execs.len(), 32);
        assert!(pass.execs.iter().all(|exec| exec.func.is_some()));
        assert_eq!(pass.execs[31].render_area.unwrap().extent.width, 32);
        assert!(
            info.dependencies
                .iter()
                .all(|dep| dep.src_subpass == vk::SUBPASS_EXTERNAL && dep.dst_subpass == 0)
        );

        let depth = SubpassFixture::depth_attachment_exec(LoadOp::Load, StoreOp::Store);
        assert_eq!(
            &*SubpassFixture::plan_subpasses(&SubpassFixture::subpass_command(vec![
                depth.clone(),
                depth
            ]))
            .1,
            &[0, 0]
        );
    }

    #[test]
    fn coalescing_checks_repeated_ranges_without_losing_layout_conflicts() {
        for freeze in [false, true] {
            for conflict in [false, true] {
                let mut pass = SubpassFixture::subpass_command(vec![
                        SubpassFixture::color_attachment_exec(
                            LoadOp::Load
                        );
                        2
                    ]);
                for idx in 0..1024 {
                    pass.execs[0].accesses.push(
                        3,
                        SubresourceAccess {
                            access: if conflict && idx == 1023 {
                                AccessType::FragmentShaderReadOther
                            } else {
                                AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer
                            },
                            subresource: SubresourceRange::Image(color_subresource_range(
                                0..1,
                                0..1,
                            )),
                        },
                    );
                }
                if freeze {
                    pass.execs[0].accesses.freeze();
                }
                assert_eq!(
                    &*SubpassFixture::plan_subpasses(&pass).1,
                    if conflict { &[0, 1] } else { &[0, 0] }
                );
            }
        }
    }

    #[test]
    fn coalescing_rejects_duplicate_layout_conflicts_beyond_inline_capacity() {
        for count in [2, 32] {
            let mut pass = SubpassFixture::subpass_command(vec![
                    SubpassFixture::color_attachment_exec(
                        LoadOp::Load
                    );
                    3
                ]);
            for node in (2..count + 2).rev() {
                for exec_idx in [0, 2] {
                    for _ in 0..2 {
                        pass.execs[exec_idx].accesses.push(
                            node,
                            SubresourceAccess {
                                access:
                                    AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                                subresource: SubresourceRange::Image(color_subresource_range(
                                    0..1,
                                    0..1,
                                )),
                            },
                        );
                    }
                }
            }
            assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 0, 0]);
            pass.execs[2].accesses.get_mut(&2).unwrap()[1].access =
                AccessType::FragmentShaderReadOther;
            assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 0, 1]);
            pass.execs[2].accesses.get_mut(&2).unwrap()[0].access =
                AccessType::FragmentShaderReadOther;
            // No per-execution conflict remains, but the non-adjacent group history still differs.
            assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 0, 1]);
        }
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn color_attachment_load_dependencies_avoid_invalid_stage_access_pairs()
    -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let pipeline = SubpassFixture::test_triangle_pipeline(&device)?;
        let mut graph = Graph::new();
        let image = graph.bind_resource(Image::create(
            &device,
            ImageInfo::image_2d(
                4,
                4,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
        )?);

        graph
            .begin_cmd()
            .debug_name("color attachment writer")
            .bind_pipeline(&pipeline)
            .color_attachment_image(0, image, LoadOp::CLEAR_BLACK_ALPHA_ZERO, StoreOp::Store)
            .record_cmd(|cmd| {
                cmd.draw(3, 1, 0, 0);
            });
        graph
            .begin_cmd()
            .debug_name("color attachment reader")
            .bind_pipeline(&pipeline)
            .color_attachment_image(0, image, LoadOp::Load, StoreOp::Store)
            .record_cmd(|cmd| {
                cmd.draw(3, 1, 0, 0);
            });

        let mut submission = graph.finalize();
        let mut schedule = vec![0, 1];
        submission.merge_scheduled_cmds(&mut schedule);

        let dependencies = Submission::build_subpass_dependencies(
            &submission.graph.cmds[0],
            &vec![PipelineStageAccessFlags::default(); submission.graph.resources.len()],
            &[0, 1],
        );
        let dep = dependencies
            .iter()
            .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 1)
            .expect("missing subpass dependency for color attachment load");

        assert!(
            dep.src_access_mask
                .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
            "source access should include color attachment writes"
        );
        assert!(
            dep.dst_access_mask
                .contains(vk::AccessFlags::COLOR_ATTACHMENT_READ),
            "destination access should include color attachment reads"
        );
        SubpassFixture::assert_no_invalid_attachment_stage_access_pairs(dep);
        SubpassFixture::assert_attachment_read_stage_mappings(dep);

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn color_attachment_read_dependencies_avoid_invalid_stage_access_pairs()
    -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let pipeline = SubpassFixture::test_triangle_pipeline(&device)?;
        let mut graph = Graph::new();
        let image = graph.bind_resource(Image::create(
            &device,
            ImageInfo::image_2d(
                4,
                4,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
        )?);

        graph
            .begin_cmd()
            .debug_name("color attachment first reader")
            .bind_pipeline(&pipeline)
            .color_attachment_image(0, image, LoadOp::Load, StoreOp::DontCare)
            .record_cmd(|cmd| {
                cmd.draw(3, 1, 0, 0);
            });
        graph
            .begin_cmd()
            .debug_name("color attachment second reader")
            .bind_pipeline(&pipeline)
            .color_attachment_image(0, image, LoadOp::Load, StoreOp::DontCare)
            .record_cmd(|cmd| {
                cmd.draw(3, 1, 0, 0);
            });

        let mut submission = graph.finalize();
        let mut schedule = vec![0, 1];
        submission.merge_scheduled_cmds(&mut schedule);

        let dependencies = Submission::build_subpass_dependencies(
            &submission.graph.cmds[0],
            &vec![PipelineStageAccessFlags::default(); submission.graph.resources.len()],
            &[0, 1],
        );
        let dep = dependencies
            .iter()
            .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 1)
            .expect("missing subpass dependency for color attachment read");

        assert!(
            dep.src_access_mask
                .contains(vk::AccessFlags::COLOR_ATTACHMENT_READ),
            "source access should include color attachment reads"
        );
        assert!(
            dep.dst_access_mask
                .contains(vk::AccessFlags::COLOR_ATTACHMENT_READ),
            "destination access should include color attachment reads"
        );
        SubpassFixture::assert_no_invalid_attachment_stage_access_pairs(dep);
        SubpassFixture::assert_attachment_read_stage_mappings(dep);

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn color_attachment_read_to_write_dependencies_avoid_invalid_stage_access_pairs()
    -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let pipeline = SubpassFixture::test_triangle_pipeline(&device)?;
        let mut graph = Graph::new();
        let image = graph.bind_resource(Image::create(
            &device,
            ImageInfo::image_2d(
                4,
                4,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
        )?);

        graph
            .begin_cmd()
            .debug_name("color attachment reader")
            .bind_pipeline(&pipeline)
            .color_attachment_image(0, image, LoadOp::Load, StoreOp::DontCare)
            .record_cmd(|cmd| {
                cmd.draw(3, 1, 0, 0);
            });
        graph
            .begin_cmd()
            .debug_name("color attachment writer")
            .bind_pipeline(&pipeline)
            .color_attachment_image(0, image, LoadOp::CLEAR_BLACK_ALPHA_ZERO, StoreOp::Store)
            .record_cmd(|cmd| {
                cmd.draw(3, 1, 0, 0);
            });

        let mut submission = graph.finalize();
        let mut schedule = vec![0, 1];
        submission.merge_scheduled_cmds(&mut schedule);

        let dependencies = Submission::build_subpass_dependencies(
            &submission.graph.cmds[0],
            &vec![PipelineStageAccessFlags::default(); submission.graph.resources.len()],
            &[0, 1],
        );
        let dep = dependencies
            .iter()
            .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 1)
            .expect("missing subpass dependency for color attachment read to write");

        assert!(
            dep.src_access_mask
                .contains(vk::AccessFlags::COLOR_ATTACHMENT_READ),
            "source access should include color attachment reads"
        );
        assert!(
            dep.dst_access_mask
                .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
            "destination access should include color attachment writes"
        );
        SubpassFixture::assert_no_invalid_attachment_stage_access_pairs(dep);
        SubpassFixture::assert_attachment_read_stage_mappings(dep);

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn color_input_attachment_dependencies_use_fragment_shader_input_reads()
    -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let (pipeline_a, pipeline_b) = SubpassFixture::test_input_attachment_pipelines(&device)?;
        let mut graph = Graph::new();
        let image = graph.bind_resource(Image::create(
            &device,
            ImageInfo::image_2d(
                4,
                4,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::INPUT_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_DST,
            ),
        )?);

        graph
            .begin_cmd()
            .debug_name("input attachment writer")
            .bind_pipeline(&pipeline_a)
            .color_attachment_image(0, image, LoadOp::CLEAR_BLACK_ALPHA_ZERO, StoreOp::Store)
            .record_cmd(|cmd| {
                cmd.draw(3, 1, 0, 0);
            });
        graph
            .begin_cmd()
            .debug_name("input attachment reader")
            .bind_pipeline(&pipeline_b)
            .color_attachment_image(0, image, LoadOp::DontCare, StoreOp::Store)
            .record_cmd(|cmd| {
                cmd.draw(3, 1, 0, 0);
            });

        let mut submission = graph.finalize();
        let mut schedule = vec![0, 1];
        submission.merge_scheduled_cmds(&mut schedule);

        let dependencies = Submission::build_subpass_dependencies(
            &submission.graph.cmds[0],
            &vec![PipelineStageAccessFlags::default(); submission.graph.resources.len()],
            &[0, 1],
        );
        let dep = dependencies
            .iter()
            .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 1)
            .expect("missing subpass dependency for input attachment read");

        assert!(
            dep.src_stage_mask
                .contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
            "source stage should include color attachment output"
        );
        assert!(
            dep.src_access_mask
                .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
            "source access should include color attachment write"
        );
        assert!(
            dep.dst_stage_mask
                .contains(vk::PipelineStageFlags::FRAGMENT_SHADER),
            "destination stage should include fragment shader input attachment reads"
        );
        assert!(
            dep.dst_access_mask
                .contains(vk::AccessFlags::INPUT_ATTACHMENT_READ),
            "destination access should include input attachment reads"
        );

        Ok(())
    }

    #[test]
    fn command_access_index_dedupes_accesses_per_command_and_resets_between_commands() {
        let cmds = vec![
            command_with_accesses(&[
                (0, AccessType::TransferRead),
                (0, AccessType::TransferWrite),
                (1, AccessType::TransferRead),
                (1, AccessType::TransferWrite),
            ]),
            command_with_accesses(&[(0, AccessType::TransferRead), (1, AccessType::TransferRead)]),
        ];
        let mut access_index = CommandAccessIndex::default();

        access_index.update_from_cmds(&cmds, 2, 0);

        assert_eq!(access_index.cmds_by_node[0], vec![0, 1]);
        assert_eq!(access_index.cmds_by_node[1], vec![0, 1]);
        assert_eq!(access_index.accessed_nodes_by_cmd[0], vec![0, 1]);
        assert_eq!(access_index.accessed_nodes_by_cmd[1], vec![0, 1]);
    }

    #[test]
    fn command_access_index_dedupes_resource_sets_across_executions() {
        let cmds = [
            command_with_resource_set_accesses(&[&[0], &[0]]),
            command_with_resource_set_accesses(&[&[0, 1]]),
            command_with_resource_set_accesses(&[&[1]]),
        ];
        let mut access_index = CommandAccessIndex::default();

        access_index.update_from_cmds(&cmds, 0, 2);

        assert!(access_index.cmds_by_node.is_empty());
        assert_eq!(access_index.cmds_by_resource_set[0], vec![0, 1]);
        assert_eq!(access_index.cmds_by_resource_set[1], vec![1, 2]);
        assert_eq!(
            access_index.accessed_resource_sets_by_cmd,
            [
                vec![ResourceSetIndex::new(0)],
                vec![ResourceSetIndex::new(0), ResourceSetIndex::new(1)],
                vec![ResourceSetIndex::new(1)],
            ]
        );
    }

    #[test]
    fn command_access_index_includes_read_and_write_accesses() {
        let cmds = vec![
            command_with_accesses(&[(0, AccessType::TransferRead)]),
            command_with_accesses(&[(1, AccessType::TransferWrite)]),
            command_with_accesses(&[(1, AccessType::TransferRead)]),
            command_with_accesses(&[(1, AccessType::TransferWrite)]),
        ];
        let mut access_index = CommandAccessIndex::default();

        access_index.update_from_cmds(&cmds, 2, 0);

        assert_eq!(access_index.cmds_by_node[0], vec![0]);
        assert_eq!(access_index.cmds_by_node[1], vec![1, 2, 3]);
        assert_eq!(access_index.accessed_nodes_by_cmd[0], vec![0]);
        assert_eq!(access_index.accessed_nodes_by_cmd[1], vec![1]);
        assert_eq!(access_index.accessed_nodes_by_cmd[2], vec![1]);
        assert_eq!(access_index.accessed_nodes_by_cmd[3], vec![1]);
    }

    #[test]
    fn concurrent_buffer_sources_are_queue_safe_in_both_barrier_apis() {
        for queue_flags in [
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            vk::QueueFlags::COMPUTE,
            vk::QueueFlags::TRANSFER,
        ] {
            for previous in [
                AccessType::FragmentShaderReadOther,
                AccessType::FragmentShaderWrite,
                AccessType::ComputeShaderWrite,
            ] {
                let supported =
                    queue_flags.contains(if previous == AccessType::ComputeShaderWrite {
                        vk::QueueFlags::COMPUTE
                    } else {
                        vk::QueueFlags::GRAPHICS
                    });
                // A second, supported producer must not be lost when lowering a foreign stage.
                let barrier = BufferBarrier {
                    previous_accesses: &[previous, AccessType::HostWrite],
                    next_accesses: &[AccessType::TransferRead],
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    buffer: vk::Buffer::from_raw(1),
                    offset: 16,
                    size: 32,
                };
                let writer = crate::driver::is_write_access(previous);
                let expected_src = vk::PipelineStageFlags::HOST
                    | if supported {
                        crate::driver::pipeline_stage_access_flags(previous).0
                    } else {
                        vk::PipelineStageFlags::ALL_COMMANDS
                    };
                let expected_access = vk::AccessFlags::HOST_WRITE
                    | if !writer {
                        vk::AccessFlags::empty()
                    } else if supported {
                        vk::AccessFlags::SHADER_WRITE
                    } else {
                        vk::AccessFlags::MEMORY_WRITE
                    };
                let (src, dst, legacy) = Submission::buffer_memory_barrier(&barrier, queue_flags);
                assert_eq!(src, expected_src);
                assert_eq!(legacy.src_access_mask, expected_access);
                assert_eq!(dst, vk::PipelineStageFlags::TRANSFER);
                assert_eq!(legacy.dst_access_mask, vk::AccessFlags::TRANSFER_READ);

                let sync2 = Submission::buffer_memory_barrier2(&barrier, queue_flags);
                assert_eq!(sync2.src_stage_mask.as_raw(), expected_src.as_raw() as u64);
                assert_eq!(
                    sync2.src_access_mask.as_raw(),
                    (expected_access
                        | if supported && !writer {
                            vk::AccessFlags::SHADER_READ
                        } else {
                            vk::AccessFlags::empty()
                        })
                    .as_raw() as u64
                );
                assert_eq!(sync2.dst_stage_mask, vk::PipelineStageFlags2::TRANSFER);
                assert_eq!(sync2.dst_access_mask, vk::AccessFlags2::TRANSFER_READ);
                assert_eq!(sync2.src_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
                assert_eq!(sync2.dst_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
                assert_eq!((sync2.offset, sync2.size), (16, 32));
            }
        }
        for previous in [
            &[AccessType::AccelerationStructureBufferWrite][..],
            &[
                AccessType::AccelerationStructureBufferWrite,
                AccessType::FragmentShaderWrite,
            ],
        ] {
            let barrier = BufferBarrier {
                previous_accesses: previous,
                next_accesses: &[AccessType::TransferRead],
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                ..Default::default()
            };
            let expected_access = vk::AccessFlags::TRANSFER_WRITE
                | if previous.len() == 2 {
                    vk::AccessFlags::MEMORY_WRITE
                } else {
                    vk::AccessFlags::empty()
                };
            let expected_stages = vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                | if previous.len() == 2 {
                    vk::PipelineStageFlags::ALL_COMMANDS
                } else {
                    vk::PipelineStageFlags::empty()
                };
            let (src, _, legacy) =
                Submission::buffer_memory_barrier(&barrier, vk::QueueFlags::COMPUTE);
            assert_eq!(src, expected_stages);
            assert_eq!(legacy.src_access_mask, expected_access);
            let sync2 = Submission::buffer_memory_barrier2(&barrier, vk::QueueFlags::COMPUTE);
            assert_eq!(
                sync2.src_stage_mask.as_raw(),
                expected_stages.as_raw() as u64
            );
            assert_eq!(
                sync2.src_access_mask.as_raw(),
                expected_access.as_raw() as u64
            );
        }
        for (queue_flags, stages, accesses) in [
            (
                vk::QueueFlags::COMPUTE,
                vk::PipelineStageFlags2::MICROMAP_BUILD_EXT,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            (
                vk::QueueFlags::TRANSFER,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
            ),
        ] {
            let barrier = Submission::buffer_memory_barrier2(
                &BufferBarrier {
                    previous_accesses: &[AccessType::MicromapBuildBufferWrite],
                    next_accesses: &[AccessType::TransferRead],
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    ..Default::default()
                },
                queue_flags,
            );
            assert_eq!(barrier.src_stage_mask, stages);
            assert_eq!(barrier.src_access_mask, accesses);
        }
    }

    #[test]
    fn consume_pending_buffer_transfers_removes_intersecting_ranges() {
        let consumed = BufferSubresourceRange { start: 4, end: 8 };
        let kept = BufferSubresourceRange { start: 8, end: 12 };
        let mut pending = vec![
            BufferQueueOwnershipTransfer {
                dst_queue_family_index: 0,
                range: consumed,
                src_queue_family_index: 1,
            },
            BufferQueueOwnershipTransfer {
                dst_queue_family_index: 0,
                range: kept,
                src_queue_family_index: 1,
            },
        ];

        assert!(!super::BufferQueueOwnershipTransfer::consume_pending(
            &mut pending,
            consumed
        ));

        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].range, kept);
    }

    #[test]
    fn consume_pending_image_transfers_removes_intersecting_ranges() {
        let consumed = color_subresource_range(0..1, 0..1);
        let kept = color_subresource_range(1..2, 0..1);
        let mut pending = vec![
            ImageOwnershipTransfer {
                dst_queue_family_index: 0,
                layouts: ImageOwnershipLayouts {
                    old: vk::ImageLayout::GENERAL,
                    new: vk::ImageLayout::GENERAL,
                },
                range: consumed,
                src_queue_family_index: 1,
                src_queue_index: 0,
            },
            ImageOwnershipTransfer {
                dst_queue_family_index: 0,
                layouts: ImageOwnershipLayouts {
                    old: vk::ImageLayout::GENERAL,
                    new: vk::ImageLayout::GENERAL,
                },
                range: kept,
                src_queue_family_index: 1,
                src_queue_index: 0,
            },
        ];

        assert!(!super::ImageOwnershipTransfer::consume_pending(
            &mut pending,
            consumed
        ));

        assert_eq!(pending.len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            pending[0].range,
            kept
        ));
    }

    #[test]
    fn dependency_selection_dedupes_repeated_read_dependencies() {
        let access_index = CommandAccessIndex {
            /*
            Cmd 1 reads node 0 twice and writes node 1. Dependency selection for node 1 must
            schedule cmd 0 once, not once per repeated read of node 0.
            */
            cmds_by_node: vec![vec![0, 1], vec![1]],
            accessed_nodes_by_cmd: vec![vec![0], vec![0, 0, 1]],
            ..Default::default()
        };
        let mut schedule = Schedule {
            access_index,
            ..Default::default()
        };

        super::Schedule::schedule_dependency_cmds_before_target_access(1, 1, &mut schedule);

        assert_eq!(schedule.cmds, vec![0]);
    }

    #[test]
    fn dependency_selection_revisits_node_at_later_boundary() {
        let access_index = CommandAccessIndex {
            /*
            A is first discovered through cmd 0, then rediscovered through cmd 2. The later
            boundary must extend A's selected prefix to include cmd 1.

            cmd 0: A, B
            cmd 1: A
            cmd 2: A, C
            cmd 3: B, C, T
            */
            cmds_by_node: vec![vec![0, 1, 2], vec![0, 3], vec![2, 3], vec![3]],
            accessed_nodes_by_cmd: vec![vec![0, 1], vec![0], vec![0, 2], vec![1, 2, 3]],
            ..Default::default()
        };
        let mut schedule = Schedule {
            access_index,
            ..Default::default()
        };

        super::Schedule::schedule_dependency_cmds_before_target_access(3, 3, &mut schedule);

        assert_eq!(schedule.cmds, vec![0, 1, 2]);
    }

    #[test]
    fn dependency_selection_schedules_inputs_to_first_target_access() {
        let access_index = CommandAccessIndex {
            /*
            Node 0 is produced by cmd 0 and then read by cmd 1. Node 1 is the target written by
            cmd 1, so dependencies(node 1) should include cmd 0 but not cmd 1.
            */
            cmds_by_node: vec![vec![0, 1], vec![1]],
            accessed_nodes_by_cmd: vec![vec![0], vec![0, 1]],
            ..Default::default()
        };
        let mut schedule = Schedule {
            access_index,
            ..Default::default()
        };

        super::Schedule::schedule_dependency_cmds_before_target_access(1, 1, &mut schedule);

        assert_eq!(schedule.cmds, vec![0]);
    }

    #[test]
    fn dependency_selection_schedules_resource_set_prefix() {
        let resource_set_idx = ResourceSetIndex::new(0);
        let access_index = CommandAccessIndex {
            // Cmd 0 first uses the set. Cmd 1 uses the set and accesses the target node.
            cmds_by_node: vec![vec![1]],
            accessed_nodes_by_cmd: vec![vec![], vec![0]],
            cmds_by_resource_set: vec![vec![0, 1]],
            accessed_resource_sets_by_cmd: vec![vec![resource_set_idx], vec![resource_set_idx]],
        };
        let mut schedule = Schedule {
            access_index,
            ..Default::default()
        };

        super::Schedule::schedule_dependency_cmds_before_target_access(0, 1, &mut schedule);

        assert_eq!(schedule.cmds, vec![0]);
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn depth_attachment_load_dependencies_avoid_invalid_stage_access_pairs()
    -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let pipeline = SubpassFixture::test_triangle_pipeline(&device)?;
        let mut graph = Graph::new();
        let image = graph.bind_resource(Image::create(
            &device,
            ImageInfo::image_2d(
                4,
                4,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ),
        )?);

        graph
            .begin_cmd()
            .debug_name("depth attachment first reader")
            .bind_pipeline(&pipeline)
            .depth_stencil_attachment_image(image, LoadOp::Load, StoreOp::Store)
            .record_cmd(|cmd| {
                cmd.draw(3, 1, 0, 0);
            });
        graph
            .begin_cmd()
            .debug_name("depth attachment second reader")
            .bind_pipeline(&pipeline)
            .depth_stencil_attachment_image(image, LoadOp::Load, StoreOp::Store)
            .record_cmd(|cmd| {
                cmd.draw(3, 1, 0, 0);
            });

        let mut submission = graph.finalize();
        let mut schedule = vec![0, 1];
        submission.merge_scheduled_cmds(&mut schedule);

        let dependencies = Submission::build_subpass_dependencies(
            &submission.graph.cmds[0],
            &vec![PipelineStageAccessFlags::default(); submission.graph.resources.len()],
            &[0, 1],
        );
        let dep = dependencies
            .iter()
            .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 1)
            .expect("missing subpass dependency for depth attachment load");

        assert!(
            dep.src_access_mask
                .contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ),
            "source access should include depth/stencil attachment reads"
        );
        assert!(
            dep.dst_access_mask
                .contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ),
            "destination access should include depth/stencil attachment reads"
        );
        SubpassFixture::assert_no_invalid_attachment_stage_access_pairs(dep);
        SubpassFixture::assert_attachment_read_stage_mappings(dep);

        Ok(())
    }

    #[test]
    fn depth_attachment_read_to_write_dependency_includes_late_read_stage() {
        let dependencies = SubpassFixture::depth_attachment_dependencies(
            LoadOp::Load,
            StoreOp::DontCare,
            LoadOp::CLEAR_ONE_STENCIL_ZERO,
            StoreOp::Store,
        );
        let dep = dependencies
            .iter()
            .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 1)
            .expect("missing subpass dependency for depth attachment read to write");

        assert!(
            dep.src_access_mask
                .contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ),
            "source access should include depth/stencil attachment reads"
        );
        assert!(
            dep.dst_access_mask
                .contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
            "destination access should include depth/stencil attachment writes"
        );
        assert!(
            dep.src_stage_mask
                .contains(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS),
            "source stage should include early fragment tests"
        );
        assert!(
            dep.src_stage_mask
                .contains(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS),
            "source stage should include late fragment tests"
        );
    }

    #[test]
    fn depth_attachment_write_to_write_dependency_uses_write_access() {
        let dependencies = SubpassFixture::depth_attachment_dependencies(
            LoadOp::CLEAR_ONE_STENCIL_ZERO,
            StoreOp::Store,
            LoadOp::CLEAR_ONE_STENCIL_ZERO,
            StoreOp::Store,
        );
        let dep = dependencies
            .iter()
            .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 1)
            .expect("missing subpass dependency for depth attachment write to write");

        assert!(
            dep.src_access_mask
                .contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
            "source access should include depth/stencil attachment writes"
        );
        assert!(
            !dep.src_access_mask
                .contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ),
            "source access should not include depth/stencil attachment reads"
        );
        assert!(
            dep.dst_access_mask
                .contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
            "destination access should include depth/stencil attachment writes"
        );
    }

    #[test]
    fn depth_resolve_dependencies_include_fixed_function_color_output() {
        let mut source = SubpassFixture::depth_attachment_exec(LoadOp::Load, StoreOp::Store);
        let state = source.attachments.depth_stencil.as_mut().unwrap();
        state.resolve = Some(DepthStencilResolve {
            attachment: Attachment {
                target: 1,
                ..state.attachment
            },
            dst_attachment_idx: 0,
            depth_mode: None,
            stencil_mode: None,
        });
        let mut consumer = SubpassFixture::depth_attachment_exec(LoadOp::Load, StoreOp::Store);
        consumer
            .attachments
            .depth_stencil
            .as_mut()
            .unwrap()
            .attachment
            .target = 1;
        let deps = Submission::build_subpass_dependencies(
            &SubpassFixture::subpass_command(vec![source, consumer]),
            &[PipelineStageAccessFlags::default(); 2],
            &[0, 1],
        );
        let dep = deps
            .iter()
            .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 1)
            .unwrap();
        assert!(
            dep.src_stage_mask
                .contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        );
        assert!(
            dep.src_access_mask
                .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        );
    }

    #[test]
    fn every_group_member_must_have_only_known_read_only_non_attachment_accesses() {
        for access in [
            AccessType::Nothing,
            AccessType::General,
            AccessType::FragmentShaderWrite,
            AccessType::VertexShaderWrite,
            AccessType::AnyShaderWrite,
            AccessType::ComputeShaderReadWrite,
            AccessType::TransferRead,
            AccessType::FragmentShaderReadColorInputAttachment,
            AccessType::FragmentShaderReadDepthStencilInputAttachment,
        ] {
            assert!(!PipelineStageAccessFlags::is_read_only_graphics_access(
                access
            ));
            for (bad_idx, expected) in [[0, 1, 1], [0, 1, 2], [0, 0, 1]].iter().enumerate() {
                let mut pass = SubpassFixture::subpass_command(vec![
                        SubpassFixture::color_attachment_exec(
                            LoadOp::Load
                        );
                        3
                    ]);
                // The offending resource need not occur in an adjacent execution.
                pass.execs[bad_idx].accesses =
                    SubpassFixture::exec_with_buffer_access(access).accesses;
                assert_eq!(
                    &*SubpassFixture::plan_subpasses(&pass).1,
                    expected,
                    "{access:?} at {bad_idx}"
                );
            }
        }
        for access in [
            AccessType::IndexBuffer,
            AccessType::IndirectBuffer,
            AccessType::VertexBuffer,
            AccessType::FragmentShaderReadOther,
            AccessType::AnyShaderReadUniformBuffer,
            AccessType::MeshShaderReadOther,
            AccessType::TaskShaderReadUniformBuffer,
        ] {
            let mut pass = SubpassFixture::subpass_command(vec![
                    SubpassFixture::color_attachment_exec(
                        LoadOp::Load
                    );
                    2
                ]);
            pass.execs[0].accesses = SubpassFixture::exec_with_buffer_access(access).accesses;
            assert_eq!(
                &*SubpassFixture::plan_subpasses(&pass).1,
                &[0, 0],
                "{access:?}"
            );
        }
        let sets = command_with_resource_set_accesses(&[&[0], &[0]]);
        let mut pass = SubpassFixture::subpass_command(vec![
            SubpassFixture::color_attachment_exec(
                LoadOp::Load
            );
            2
        ]);
        for (exec, set_exec) in pass.execs.iter_mut().zip(&sets.execs) {
            exec.resource_set_accesses = set_exec.resource_set_accesses.clone();
        }
        assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 0]);
    }

    #[test]
    fn every_micromap_access_is_sync2_only() {
        for access in [
            AccessType::MicromapBuildInputRead,
            AccessType::MicromapBuildScratchReadWrite,
            AccessType::MicromapBuildRead,
            AccessType::MicromapBuildWrite,
            AccessType::MicromapBuildBufferRead,
            AccessType::MicromapBuildBufferWrite,
            AccessType::AccelerationStructureBuildMicromapRead,
        ] {
            assert!(PipelineStageAccessFlags::is_micromap_access(access));
        }
    }

    #[test]
    fn execution_subpass_mapping_invariants_and_prepared_resource_moves() {
        assert!(Submission::valid_exec_subpasses(0, &[]));
        assert!(Submission::valid_exec_subpasses(1, &[0]));
        assert!(Submission::valid_exec_subpasses(5, &[0, 0, 1, 1, 2]));
        for invalid in [&[1, 1][..], &[0, 2], &[1, 0], &[0, u32::MAX], &[0], &[]] {
            assert!(!Submission::valid_exec_subpasses(2, invalid));
        }
        let mut resources = vec![CommandRecordingResources {
            exec_subpasses: vec![0, 0, 1].into_boxed_slice(),
            descriptor_sets: vec![Vec::new(), Vec::new(), Vec::new()],
            descriptor_pool: None,
            render_pass: None,
        }];
        let prepared = PreparedStreamRecording {
            resources: Mutex::new(std::mem::take(&mut resources)),
        };
        std::mem::swap(&mut resources, &mut prepared.resources.lock().unwrap());
        assert_eq!(&*resources[0].exec_subpasses, &[0, 0, 1]);
        assert_eq!(resources[0].descriptor_sets.len(), 3);
    }

    #[test]
    fn external_access_scratch_is_reentrant_and_recovers_after_unwind() {
        PipelineStageAccessFlags::with_external_render_pass_accesses(4, |outer| {
            outer[0] = PipelineStageAccessFlags::new(AccessType::TransferWrite);
            PipelineStageAccessFlags::with_external_render_pass_accesses(32, |inner| {
                assert!(
                    inner
                        .iter()
                        .all(|scope| scope.stage_flags.is_empty() && scope.access_flags.is_empty())
                );
                inner[0] = PipelineStageAccessFlags::new(AccessType::ComputeShaderWrite);
            });
            assert_eq!(outer[0].stage_flags, vk::PipelineStageFlags::TRANSFER);
            assert_eq!(outer[0].access_flags, vk::AccessFlags::TRANSFER_WRITE);
        });
        let result = std::panic::catch_unwind(|| {
            PipelineStageAccessFlags::with_external_render_pass_accesses(32, |accesses| {
                accesses[0] = PipelineStageAccessFlags::new(AccessType::TransferWrite);
                panic!("simulated pool callback panic");
            });
        });
        assert!(result.is_err());
        PipelineStageAccessFlags::with_external_render_pass_accesses(32, |accesses| {
            assert!(
                accesses
                    .iter()
                    .all(|scope| scope.stage_flags.is_empty() && scope.access_flags.is_empty())
            );
        });
    }

    #[test]
    fn external_access_scratch_reuses_allocation_and_resets_between_schedules() {
        let allocation =
            PipelineStageAccessFlags::with_external_render_pass_accesses(8, |accesses| {
                accesses[7] = PipelineStageAccessFlags::new(AccessType::TransferWrite);
                accesses.as_ptr()
            });
        for len in [4, 8] {
            PipelineStageAccessFlags::with_external_render_pass_accesses(len, |accesses| {
                assert_eq!(accesses.len(), len);
                assert_eq!(accesses.as_ptr(), allocation);
                assert!(
                    accesses
                        .iter()
                        .all(|scope| scope.stage_flags.is_empty() && scope.access_flags.is_empty())
                );
                accesses[0] = PipelineStageAccessFlags::new(AccessType::ComputeShaderWrite);
            });
        }
        let result = PipelineStageAccessFlags::with_external_render_pass_accesses(16, |accesses| {
            assert!(
                accesses
                    .iter()
                    .all(|scope| scope.stage_flags.is_empty() && scope.access_flags.is_empty())
            );
            accesses[15] = PipelineStageAccessFlags::new(AccessType::TransferWrite);
            Err::<(), _>(())
        });
        assert!(result.is_err());
        PipelineStageAccessFlags::with_external_render_pass_accesses(16, |accesses| {
            assert!(
                accesses
                    .iter()
                    .all(|scope| scope.stage_flags.is_empty() && scope.access_flags.is_empty())
            );
        });
        PipelineStageAccessFlags::with_external_render_pass_accesses(0, |accesses| {
            assert!(accesses.is_empty())
        });
    }

    #[test]
    fn external_accesses_have_canonical_dependencies() {
        let pass = SubpassFixture::subpass_command(vec![SubpassFixture::exec_with_buffer_access(
            AccessType::VertexShaderReadUniformBuffer,
        )]);
        let mut history = [PipelineStageAccessFlags::default(); 1];
        let before = Submission::build_subpass_dependencies(&pass, &history, &[0]);
        for _ in 0..1024 {
            PipelineStageAccessFlags::record_external_accesses(&mut history, &pass);
        }
        assert_eq!(
            Submission::build_subpass_dependencies(&pass, &history, &[0]),
            before
        );
        PipelineStageAccessFlags::record_external_accesses(
            &mut history,
            &command_with_accesses(&[(0, AccessType::TransferWrite)]),
        );
        let after = Submission::build_subpass_dependencies(&pass, &history, &[0]);
        assert!(
            after[0]
                .src_stage_mask
                .contains(vk::PipelineStageFlags::TRANSFER)
        );
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn external_subpass_dependency_targets_first_subpass_consumer() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let pipeline = SubpassFixture::test_triangle_pipeline(&device)?;
        let mut graph = Graph::new();
        let image = graph.bind_resource(Image::create(
            &device,
            ImageInfo::image_2d(
                4,
                4,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
        )?);

        graph.clear_color_image(image, [0.0, 0.0, 0.0, 1.0]);
        graph
            .begin_cmd()
            .debug_name("dependency inspection render pass")
            .bind_pipeline(&pipeline)
            .color_attachment_image(0, image, LoadOp::Load, StoreOp::Store)
            .record_cmd(|cmd| {
                cmd.draw(3, 1, 0, 0);
            });

        let submission = graph.finalize();
        let mut external_access_history =
            vec![PipelineStageAccessFlags::default(); submission.graph.resources.len()];
        PipelineStageAccessFlags::record_external_accesses(
            &mut external_access_history,
            &submission.graph.cmds[0],
        );

        let dependencies = Submission::build_subpass_dependencies(
            &submission.graph.cmds[1],
            &external_access_history,
            &[0],
        );
        let dep = dependencies
            .iter()
            .find(|dep| dep.src_subpass == vk::SUBPASS_EXTERNAL && dep.dst_subpass == 0)
            .expect("missing external -> first subpass dependency");

        assert_eq!(
            dep.dst_stage_mask,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            "destination stage should describe the first subpass consumer"
        );
        assert_eq!(
            dep.dst_access_mask,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            "destination access should describe the first subpass attachment access"
        );

        Ok(())
    }

    #[test]
    fn first_resource_set_access_is_visited_once_per_recording_batch() {
        let accesses = [
            crate::ResourceSetAccess {
                resource_set_idx: ResourceSetIndex::new(0),
                access_type: crate::resource::ResourceSetAccessType::AccelerationStructure(
                    crate::resource::AccelerationStructureAccessType::BuildRead,
                ),
            },
            crate::ResourceSetAccess {
                resource_set_idx: ResourceSetIndex::new(1),
                access_type: crate::resource::ResourceSetAccessType::Image(
                    crate::resource::ImageAccessType::SampledRead,
                ),
            },
            crate::ResourceSetAccess {
                resource_set_idx: ResourceSetIndex::new(0),
                access_type: crate::resource::ResourceSetAccessType::AccelerationStructure(
                    crate::resource::AccelerationStructureAccessType::BuildRead,
                ),
            },
            crate::ResourceSetAccess {
                resource_set_idx: ResourceSetIndex::new(0),
                access_type: crate::resource::ResourceSetAccessType::AccelerationStructure(
                    crate::resource::AccelerationStructureAccessType::RayTracingRead,
                ),
            },
        ];
        let resource_set_count = 2;
        let mut acquired = fixedbitset::FixedBitSet::with_capacity(
            resource_set_count * crate::resource::ResourceSetAccessType::COUNT,
        );
        let mut visited = Vec::new();

        Submission::for_each_first_resource_set_access(
            &accesses,
            resource_set_count,
            &mut acquired,
            |access| visited.push((access.resource_set_idx.as_usize(), access.access_type)),
        );

        assert_eq!(visited.len(), 3);
        assert_eq!(visited[0], (0, accesses[0].access_type));
        assert_eq!(visited[1], (1, accesses[1].access_type));
        assert_eq!(visited[2], (0, accesses[3].access_type));

        Submission::for_each_first_resource_set_access(
            &accesses,
            resource_set_count,
            &mut acquired,
            |_| panic!("resource set visited twice in one batch"),
        );

        acquired.clear();
        acquired.grow(resource_set_count * crate::resource::ResourceSetAccessType::COUNT);
        Submission::for_each_first_resource_set_access(
            &accesses[..1],
            resource_set_count,
            &mut acquired,
            |access| visited.push((access.resource_set_idx.as_usize(), access.access_type)),
        );

        assert_eq!(visited[3], (0, accesses[0].access_type));
    }

    #[test]
    fn graphics_uniform_scopes_use_uniform_read_access() {
        for access in [
            AccessType::VertexShaderReadUniformBuffer,
            AccessType::TaskShaderReadUniformBuffer,
            AccessType::MeshShaderReadUniformBuffer,
        ] {
            assert_eq!(
                PipelineStageAccessFlags::new(access).access_flags,
                vk::AccessFlags::UNIFORM_READ
            );
        }
    }

    #[test]
    fn identical_buffer_writers_form_linear_dependency_chains() {
        for access in [
            AccessType::VertexShaderWrite,
            AccessType::FragmentShaderWrite,
        ] {
            for n in [1, 4, 1024] {
                for repeats in [1, 2] {
                    let mut execs = Vec::new();
                    let mut mapping = Vec::new();
                    for sp in 0..n {
                        for repeat in 0..repeats {
                            let mut exec = SubpassFixture::exec_with_buffer_access(access);
                            // Identical repeated declarations must not spoil the proof.
                            let duplicate = exec.accesses.get_mut(&0).unwrap()[0];
                            exec.accesses.push(0, duplicate);
                            if repeat == 0 {
                                exec.accesses.freeze();
                            }
                            execs.push(exec);
                            mapping.push(sp);
                        }
                    }
                    // A different-stage reader must still depend on the complete writer chain.
                    execs.push(SubpassFixture::exec_with_buffer_access(
                        AccessType::IndexBuffer,
                    ));
                    mapping.push(n);
                    let deps = Submission::build_subpass_dependencies(
                        &SubpassFixture::subpass_command(execs),
                        &[PipelineStageAccessFlags::new(AccessType::TransferWrite)],
                        &mapping,
                    );
                    assert_eq!(deps.len(), (2 * n + 1) as usize);
                    let internal = deps
                        .iter()
                        .filter(|dep| dep.src_subpass != vk::SUBPASS_EXTERNAL)
                        .collect::<Vec<_>>();
                    assert_eq!(internal.len(), n as usize);
                    for (sp, dep) in internal.into_iter().enumerate() {
                        assert_eq!(
                            (dep.src_subpass, dep.dst_subpass),
                            (sp as u32, sp as u32 + 1)
                        );
                        assert_eq!(
                            dep.src_stage_mask,
                            PipelineStageAccessFlags::new(access).stage_flags
                        );
                        assert_eq!(dep.src_access_mask, vk::AccessFlags::SHADER_WRITE);
                        let next = PipelineStageAccessFlags::new(if dep.dst_subpass == n {
                            AccessType::IndexBuffer
                        } else {
                            access
                        });
                        assert_eq!(dep.dst_stage_mask, next.stage_flags);
                        assert_eq!(dep.dst_access_mask, next.access_flags);
                        assert!(dep.dependency_flags.is_empty());
                    }
                    for dep in deps
                        .iter()
                        .filter(|dep| dep.src_subpass == vk::SUBPASS_EXTERNAL)
                    {
                        assert!(
                            dep.src_stage_mask
                                .contains(vk::PipelineStageFlags::TRANSFER)
                        );
                        assert!(dep.dependency_flags.is_empty());
                    }
                }
            }
        }
    }

    #[test]
    fn identical_storage_image_readers_form_linear_dependency_chains() {
        for access in [
            AccessType::VertexShaderReadOther,
            AccessType::FragmentShaderReadOther,
            AccessType::AnyShaderReadOther,
        ] {
            for n in [1, 4, 1024u32] {
                for repeats in [1, 2] {
                    for producer in [false, true] {
                        let execution = |access| {
                            let mut exec = Execution::default();
                            exec.accesses.push(
                                0,
                                SubresourceAccess {
                                    access,
                                    subresource: SubresourceRange::Image(color_subresource_range(
                                        0..1,
                                        0..1,
                                    )),
                                },
                            );
                            exec
                        };
                        let mut execs = Vec::new();
                        let mut mapping = Vec::new();
                        let first = u32::from(producer);
                        if producer {
                            execs.push(execution(AccessType::FragmentShaderWrite));
                            mapping.push(0);
                        }
                        for sp in first..first + n {
                            for repeat in 0..repeats {
                                let mut exec = execution(access);
                                let duplicate = exec.accesses.get_mut(&0).unwrap()[0];
                                exec.accesses.push(0, duplicate);
                                if repeat == 0 {
                                    exec.accesses.freeze();
                                }
                                execs.push(exec);
                                mapping.push(sp);
                            }
                        }
                        execs.push(execution(AccessType::FragmentShaderWrite));
                        mapping.push(first + n);
                        let deps = Submission::build_subpass_dependencies(
                            &SubpassFixture::subpass_command(execs),
                            &[PipelineStageAccessFlags::new(AccessType::TransferWrite)],
                            &mapping,
                        );
                        let mut internal_count = 0;
                        for dep in &deps {
                            assert!(dep.dependency_flags.is_empty());
                            if dep.src_subpass == vk::SUBPASS_EXTERNAL {
                                assert!(
                                    dep.src_stage_mask
                                        .contains(vk::PipelineStageFlags::TRANSFER)
                                );
                                continue;
                            }
                            internal_count += 1;
                            let source_is_writer = producer && dep.src_subpass == 0;
                            // Keep the original writer's visibility edge to every later consumer.
                            if !source_is_writer {
                                assert_eq!(dep.dst_subpass, dep.src_subpass + 1);
                            }
                            for (sp, stages, accesses) in [
                                (dep.src_subpass, dep.src_stage_mask, dep.src_access_mask),
                                (dep.dst_subpass, dep.dst_stage_mask, dep.dst_access_mask),
                            ] {
                                let writer = (producer && sp == 0) || sp == first + n;
                                let scope = PipelineStageAccessFlags::new(if writer {
                                    AccessType::FragmentShaderWrite
                                } else {
                                    access
                                });
                                assert_eq!(
                                    stages,
                                    Submission::subpass_stage_mask(scope.stage_flags)
                                );
                                assert_eq!(accesses, scope.access_flags);
                            }
                        }
                        assert_eq!(internal_count, (n + first * (n + 1)) as usize);
                        assert_eq!(deps.len(), internal_count + (first + n + 1) as usize);
                    }
                }
            }
        }
    }

    #[test]
    fn image_execution_discard_only_when_previous_access_is_nothing() {
        let access_set = ImageAccessSet::from_access;

        assert!(super::TrackedImageBarrier::execution_discard_contents(
            access_set(AccessType::Nothing)
        ));
        assert!(!super::TrackedImageBarrier::execution_discard_contents(
            access_set(AccessType::TransferRead)
        ));
        assert!(!super::TrackedImageBarrier::execution_discard_contents(
            access_set(AccessType::TransferWrite)
        ));
        assert!(!super::TrackedImageBarrier::execution_discard_contents(
            access_set(AccessType::ColorAttachmentReadWrite)
        ));
    }

    #[test]
    fn image_layout_transition_discard_keeps_attachment_write_policy() {
        let access_set = ImageAccessSet::from_access;

        assert!(
            super::TrackedImageBarrier::layout_transition_discard_contents(
                access_set(AccessType::Nothing),
                AccessType::TransferWrite,
            )
        );
        assert!(
            super::TrackedImageBarrier::layout_transition_discard_contents(
                access_set(AccessType::TransferRead),
                AccessType::TransferWrite,
            )
        );
        assert!(
            !super::TrackedImageBarrier::layout_transition_discard_contents(
                access_set(AccessType::TransferWrite),
                AccessType::ColorAttachmentReadWrite,
            )
        );
    }

    #[test]
    fn image_ownership_barriers_use_matching_layouts_and_queue_safe_source_stage() {
        let range = color_subresource_range(0..1, 0..1);
        let layouts = super::ImageOwnershipLayouts::new(
            Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            AccessType::ComputeShaderWrite,
            false,
        );
        let release = super::ImageQueueOwnershipRelease::memory_barrier(
            ImageQueueOwnershipRelease {
                image: vk::Image::null(),
                layouts,
                range,
            },
            1,
            2,
        );
        let previous_accesses = ImageAccessSet::from_access(
            AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
        );
        let (src_stage_mask, dst_stage_mask, acquire) =
            super::TrackedImageBarrier::memory_barrier(super::TrackedImageBarrier {
                previous_accesses,
                next_access: AccessType::ComputeShaderWrite,
                previous_layout: super::TrackedImageBarrier::access_set_layout(previous_accesses),
                next_layout: super::TrackedImageBarrier::access_layout(
                    AccessType::ComputeShaderWrite,
                ),
                ownership_layouts: Some(layouts),
                discard_contents: false,
                src_queue_family_index: 1,
                dst_queue_family_index: 2,
                image: vk::Image::null(),
                range,
            });

        assert_eq!(src_stage_mask, vk::PipelineStageFlags::ALL_COMMANDS);
        assert_eq!(dst_stage_mask, vk::PipelineStageFlags::COMPUTE_SHADER);
        assert_eq!(acquire.src_access_mask, vk::AccessFlags::empty());
        assert_eq!(release.old_layout, acquire.old_layout);
        assert_eq!(release.new_layout, acquire.new_layout);
        assert_eq!(
            acquire.old_layout,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
        assert_eq!(acquire.new_layout, vk::ImageLayout::GENERAL);

        let discarded = super::ImageOwnershipLayouts::new(
            Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            AccessType::TransferWrite,
            true,
        );

        assert_eq!(discarded.old, vk::ImageLayout::UNDEFINED);
        assert_eq!(discarded.new, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn image_set_attachment_republishes_overlapping_sets() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let image = Arc::new(Image::create(
            &device,
            ImageInfo::image_2d(1, 1, vk::Format::R8_UINT, vk::ImageUsageFlags::SAMPLED),
        )?);
        let range = color_subresource_range(0..1, 0..1);
        image.set_sharing_ranges(SharingMode::Exclusive(Some((0, 1))), &[range]);
        let lhs = ImageSet::new([Arc::clone(&image)])?;
        let rhs = ImageSet::new([Arc::clone(&image)])?;
        lhs.publish_queue((0, 1));
        rhs.publish_queue((0, 1));
        let mut graph = Graph::new();
        let lhs_node = graph.bind_resource(&lhs);
        let rhs_node = graph.bind_resource(&rhs);
        let mut submission = graph.finalize();
        submission
            .touched_image_sets
            .insert(lhs_node.index().as_usize());
        submission
            .touched_image_sets
            .insert(rhs_node.index().as_usize());
        let cmd_buf = CommandBuffer::create(&device, CommandBufferInfo::new(0))?;
        let mut state = RecordedSubmissionState {
            submission,
            _releases: Vec::new(),
            executed: false,
        };

        RecordedSubmission::<CommandBuffer>::attach_locked(&mut state, &cmd_buf, 0);

        assert_eq!(lhs.queue(), Some((0, 0)));
        assert_eq!(rhs.queue(), Some((0, 0)));
        assert!(
            image
                .sync_info_with_sharing_range(range)
                .all(|(_, sharing)| sharing == SharingMode::Exclusive(Some((0, 0))))
        );

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan validation layers and distinct queue families; inspect output"]
    fn image_set_submission_transfers_between_queue_families() -> Result<(), DriverError> {
        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        let Some(destination_queue_family) = device
            .physical
            .queue_families
            .iter()
            .position(|family| {
                family.queue_flags.contains(
                    vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
                )
            })
            .map(|index| index as u32)
        else {
            return Ok(());
        };
        let Some(source_queue_family) = device
            .physical
            .queue_families
            .iter()
            .enumerate()
            .find(|(index, family)| {
                *index != destination_queue_family as usize
                    && family.queue_flags.contains(vk::QueueFlags::TRANSFER)
            })
            .map(|(index, _)| index as u32)
        else {
            return Ok(());
        };
        let mut pool = HashPool::new(&device);
        let image = Arc::new(Image::create(
            &device,
            ImageInfo::image_2d(
                1,
                1,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            ),
        )?);
        let range = color_subresource_range(0..1, 0..1);

        let mut source_graph = Graph::new();
        let source_buffer = source_graph.bind_resource(Buffer::create(
            &device,
            BufferInfo::device_mem(4, vk::BufferUsageFlags::TRANSFER_SRC),
        )?);
        let source_image = source_graph.bind_resource(Arc::clone(&image));
        source_graph.copy_buffer_to_image(source_buffer, source_image);
        let mut source_fence = source_graph
            .finalize()
            .queue_submit(&mut pool, source_queue_family, 0)
            .expect("submit source graph");
        source_fence.wait().expect("wait for source graph");
        assert!(
            image
                .sync_info_with_sharing_range(range)
                .all(|(_, sharing)| {
                    sharing == SharingMode::Exclusive(Some((source_queue_family, 0)))
                })
        );

        let resource_set = ImageSet::new([Arc::clone(&image)])?;
        let mut destination_graph = Graph::new();
        let resource_set_node = destination_graph.bind_resource(&resource_set);
        destination_graph
            .begin_cmd()
            .resource_access(resource_set_node, ImageAccessType::SampledRead)
            .record_cmd(|_| {})
            .end_cmd();
        let mut destination_fence = destination_graph
            .finalize()
            .queue_submit(&mut pool, destination_queue_family, 0)
            .expect("submit destination graph");
        destination_fence
            .wait()
            .expect("wait for destination graph");

        assert!(
            image
                .sync_info_with_sharing_range(range)
                .all(|(_, sharing)| {
                    sharing == SharingMode::Exclusive(Some((destination_queue_family, 0)))
                })
        );
        assert_eq!(resource_set.queue(), Some((destination_queue_family, 0)));

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn image_set_transfer_discovery_claims_declared_range_once() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let image = Arc::new(Image::create(
            &device,
            ImageInfo::image_2d_array(1, 1, 2, vk::Format::R8_UINT, vk::ImageUsageFlags::SAMPLED),
        )?);
        let range_a = color_subresource_range(0..1, 0..1);
        let range_b = color_subresource_range(1..2, 0..1);
        image.set_sharing_ranges(SharingMode::Exclusive(Some((1, 0))), &[range_a]);
        image.set_sharing_ranges(SharingMode::Exclusive(Some((2, 0))), &[range_b]);
        image
            .swap_access(AccessType::TransferRead, range_a)
            .for_each(drop);
        image
            .swap_access(AccessType::TransferRead, range_b)
            .for_each(drop);

        let image_id = PhysicalImageId::of(&image);
        let resource_set = ImageSet::new([(Arc::clone(&image), range_a)])?;
        let mut graph = Graph::new();
        let resource_set_node = graph.bind_resource(&resource_set);
        graph
            .begin_cmd()
            .resource_access(resource_set_node, ImageAccessType::SampledRead)
            .record_cmd(|_| {})
            .end_cmd();

        let mut submission = graph.finalize();
        let schedule = Schedule {
            cmds: vec![0],
            ..Default::default()
        };
        let mut ownership = RecordingOwnership::default();
        submission.track_pending_transfers(
            &schedule,
            3,
            &mut ownership,
            ResourceSetSynchronization::Enabled,
        );

        let transfers = &submission.pending_image_set_transfers[&image_id];
        assert_eq!(transfers.len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            transfers[0].range,
            range_a
        ));
        assert_eq!(transfers[0].src_queue_family_index, 1);
        assert_eq!(transfers[0].dst_queue_family_index, 3);
        assert!(
            submission
                .touched_image_sets
                .contains(resource_set_node.index().as_usize())
        );

        submission.pending_image_set_transfers.clear();
        submission.track_pending_transfers(
            &schedule,
            3,
            &mut ownership,
            ResourceSetSynchronization::Enabled,
        );

        assert!(submission.pending_image_set_transfers.is_empty());
        assert_eq!(
            submission
                .queue_ownership_release_groups
                .iter()
                .flat_map(|group| &group.images)
                .count(),
            1
        );
        assert_eq!(ownership.image_set_images.len(), 1);

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn image_set_transfer_discovery_uses_same_family_cache() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let image = Arc::new(Image::create(
            &device,
            ImageInfo::image_2d(1, 1, vk::Format::R8_UINT, vk::ImageUsageFlags::SAMPLED),
        )?);
        let range = color_subresource_range(0..1, 0..1);
        image.set_sharing_ranges(SharingMode::Exclusive(Some((3, 7))), &[range]);
        let resource_set = ImageSet::new([Arc::clone(&image)])?;
        resource_set.publish_queue((3, 7));
        let mut graph = Graph::new();
        let resource_set_node = graph.bind_resource(&resource_set);
        graph
            .begin_cmd()
            .resource_access(resource_set_node, ImageAccessType::SampledRead)
            .record_cmd(|_| {})
            .end_cmd();

        let mut submission = graph.finalize();
        let mut ownership = RecordingOwnership::default();
        submission.track_pending_transfers(
            &Schedule {
                cmds: vec![0],
                ..Default::default()
            },
            3,
            &mut ownership,
            ResourceSetSynchronization::Enabled,
        );

        assert!(ownership.image_set_images.is_empty());
        assert!(submission.pending_image_set_transfers.is_empty());
        assert!(submission.queue_ownership_release_groups.is_empty());
        assert!(
            submission
                .touched_image_sets
                .contains(resource_set_node.index().as_usize())
        );

        image.set_sharing_ranges(SharingMode::Exclusive(Some((3, 7))), &[range]);
        assert_eq!(resource_set.queue(), None);

        Ok(())
    }

    #[test]
    fn input_descriptors_require_prior_physical_groups_and_use_actual_layouts() {
        let base = SubpassFixture::color_attachment_exec(LoadOp::Load);
        let mut pass = SubpassFixture::subpass_command(vec![base; 3]);
        pass.execs[2].attachments.color[0]
            .as_mut()
            .unwrap()
            .is_input = true;
        let (mut info, mapping) = SubpassFixture::plan_subpasses(&pass);
        assert_eq!(&*mapping, &[0, 0, 1]);
        info.subpasses[1].input_attachments[0].layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let (_, layout) = Submission::input_attachment_descriptor(&pass, &mapping, &info, 2, 0);
        assert_eq!(layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        // A matching logical predecessor inside the current physical group is not a producer.
        pass.execs[0].attachments.color[0]
            .as_mut()
            .unwrap()
            .attachment
            .aspect_mask = vk::ImageAspectFlags::DEPTH;
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Submission::input_attachment_descriptor(&pass, &[0, 1, 1], &info, 2, 0)
            }))
            .is_err()
        );
    }

    #[test]
    fn input_resolve_and_opaque_stream_executions_do_not_coalesce() {
        let base = SubpassFixture::color_attachment_exec(LoadOp::Load);
        let mut input = base.clone();
        input.attachments.color[0].as_mut().unwrap().is_input = true;
        let pass =
            SubpassFixture::subpass_command(vec![base.clone(), input.clone(), input, base.clone()]);
        assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 1, 2, 3]);

        let mut resolve = base.clone();
        let state = resolve.attachments.color[0].as_mut().unwrap();
        state.resolve = Some(ColorResolve {
            attachment: state.attachment,
            src_attachment_idx: 0,
        });
        assert_eq!(
            &*SubpassFixture::plan_subpasses(&SubpassFixture::subpass_command(vec![
                resolve.clone(),
                resolve
            ]))
            .1,
            &[0, 1]
        );

        let mut depth = SubpassFixture::depth_attachment_exec(LoadOp::Load, StoreOp::Store);
        let state = depth.attachments.depth_stencil.as_mut().unwrap();
        state.resolve = Some(DepthStencilResolve {
            attachment: Attachment {
                target: 2,
                ..state.attachment
            },
            dst_attachment_idx: 0,
            depth_mode: None,
            stencil_mode: None,
        });
        assert_eq!(
            &*SubpassFixture::plan_subpasses(&SubpassFixture::subpass_command(vec![
                depth.clone(),
                depth
            ]))
            .1,
            &[0, 1]
        );

        let mut pass = SubpassFixture::subpass_command(vec![base; 3]);
        pass.stream_scope_id = Some(1);
        assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 1, 2]);
    }

    #[test]
    fn later_clear_and_dont_care_start_new_physical_groups() {
        for load in [LoadOp::DontCare, LoadOp::Clear([0.0; 4])] {
            let pass = SubpassFixture::subpass_command(vec![
                SubpassFixture::color_attachment_exec(LoadOp::Load),
                SubpassFixture::color_attachment_exec(load),
                SubpassFixture::color_attachment_exec(LoadOp::Load),
            ]);
            assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 1, 1]);
        }
        for load in [LoadOp::DontCare, LoadOp::CLEAR_ONE_STENCIL_ZERO] {
            let pass = SubpassFixture::subpass_command(vec![
                SubpassFixture::depth_attachment_exec(LoadOp::Load, StoreOp::Store),
                SubpassFixture::depth_attachment_exec(load, StoreOp::Store),
                SubpassFixture::depth_attachment_exec(LoadOp::Load, StoreOp::Store),
            ]);
            assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 1, 1]);
        }
    }

    #[test]
    fn legacy_host_batches_separate_buffer_and_image_ownership_transfers() {
        for host_source in [false, true] {
            for host_destination in [false, true] {
                for transfer in [false, true] {
                    let src = vk::PipelineStageFlags::TOP_OF_PIPE
                        | if host_source {
                            vk::PipelineStageFlags::HOST
                        } else {
                            vk::PipelineStageFlags::TRANSFER
                        };
                    let dst = vk::PipelineStageFlags::BOTTOM_OF_PIPE
                        | if host_destination {
                            vk::PipelineStageFlags::HOST
                        } else {
                            vk::PipelineStageFlags::FRAGMENT_SHADER
                        };
                    let local_buffer = vk::BufferMemoryBarrier::default()
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(vk::Buffer::from_raw(1));
                    let local_image = vk::ImageMemoryBarrier::default()
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(vk::Image::from_raw(1));
                    let mut buffers = vec![local_buffer];
                    let mut images = vec![local_image];
                    if transfer {
                        // Put the acquire first to exercise partitioning, not just slicing.
                        buffers.insert(
                            0,
                            local_buffer
                                .buffer(vk::Buffer::from_raw(2))
                                .src_queue_family_index(1)
                                .dst_queue_family_index(0),
                        );
                        images.insert(
                            0,
                            local_image
                                .image(vk::Image::from_raw(2))
                                .src_queue_family_index(1)
                                .dst_queue_family_index(0),
                        );
                    }
                    let mut calls = 0;
                    let mut counts = [0; 3];
                    super::Submission::with_legacy_barrier_batches(
                        src,
                        dst,
                        &[vk::MemoryBarrier::default()],
                        &mut buffers,
                        &mut images,
                        |src, dst, memory, buffers, images| {
                            calls += 1;
                            counts[0] += memory.len();
                            counts[1] += buffers.len();
                            counts[2] += images.len();
                            assert!(!src.is_empty() && !dst.is_empty());
                            if (src | dst).contains(vk::PipelineStageFlags::HOST) {
                                assert!(
                                    buffers.iter().all(|barrier| barrier.src_queue_family_index
                                        == barrier.dst_queue_family_index)
                                );
                                assert!(
                                    images.iter().all(|barrier| barrier.src_queue_family_index
                                        == barrier.dst_queue_family_index)
                                );
                            } else if host_source || host_destination {
                                assert!(memory.is_empty());
                                assert!(
                                    buffers
                                        .iter()
                                        .all(|barrier| barrier.src_queue_family_index == 1
                                            && barrier.dst_queue_family_index == 0)
                                );
                                assert!(
                                    images
                                        .iter()
                                        .all(|barrier| barrier.src_queue_family_index == 1
                                            && barrier.dst_queue_family_index == 0)
                                );
                            }
                        },
                    );
                    assert_eq!(
                        calls,
                        if transfer && (host_source || host_destination) {
                            2
                        } else {
                            1
                        }
                    );
                    assert_eq!(counts, [1, buffers.len(), images.len()]);
                }
            }
        }
    }

    #[test]
    fn legacy_submit_accepts_all_commands_and_none_wait_masks() {
        let waits = [
            SemaphoreSubmitInfo {
                semaphore: vk::Semaphore::null(),
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                value: 0,
            },
            SemaphoreSubmitInfo {
                semaphore: vk::Semaphore::null(),
                stage_mask: vk::PipelineStageFlags2::NONE,
                value: 0,
            },
        ];
        let signals = [SemaphoreSubmitInfo {
            semaphore: vk::Semaphore::null(),
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            value: 0,
        }];

        assert!(SemaphoreSubmitInfo::check_args(&waits, &signals).is_ok());
    }

    #[test]
    fn legacy_submit_rejects_precise_wait_stage_masks() {
        let waits = [SemaphoreSubmitInfo {
            semaphore: vk::Semaphore::null(),
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            value: 0,
        }];

        assert!(matches!(
            SemaphoreSubmitInfo::check_args(&waits, &[]),
            Err(DriverError::Unsupported)
        ));
    }

    #[test]
    fn legacy_submit_rejects_timeline_values() {
        let waits = [SemaphoreSubmitInfo {
            semaphore: vk::Semaphore::null(),
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            value: 1,
        }];

        assert!(matches!(
            SemaphoreSubmitInfo::check_args(&waits, &[]),
            Err(DriverError::Unsupported)
        ));
    }

    #[test]
    fn locality_buffer_read_pruning_preserves_all_writer_and_external_edges() {
        use {
            super::{is_write_access, pipeline_stage_access_flags},
            AccessType::*,
        };

        // Reuse TLS across a large fan-out, interleaved writers/readers, and a read-only pass.
        for accesses in [
            std::iter::once(FragmentShaderWrite)
                .chain(std::iter::repeat_n(FragmentShaderReadOther, 128))
                .collect::<Vec<_>>(),
            vec![
                FragmentShaderWrite,
                VertexShaderReadUniformBuffer,
                IndexBuffer,
            ],
            vec![
                VertexShaderReadUniformBuffer,
                IndexBuffer,
                FragmentShaderWrite,
            ],
            vec![
                FragmentShaderWrite,
                VertexShaderReadUniformBuffer,
                IndexBuffer,
                FragmentShaderWrite,
                FragmentShaderReadOther,
                VertexShaderWrite,
                IndexBuffer,
            ],
            vec![VertexShaderReadUniformBuffer, IndexBuffer],
        ] {
            let pass = SubpassFixture::subpass_command(
                accesses
                    .iter()
                    .copied()
                    .map(SubpassFixture::exec_with_buffer_access)
                    .collect(),
            );
            let mapping = (0..accesses.len() as u32).collect::<Vec<_>>();
            let deps = Submission::build_subpass_dependencies(
                &pass,
                &[PipelineStageAccessFlags::new(ComputeShaderWrite)],
                &mapping,
            );
            let mut expected = Vec::new();
            for (src, &previous) in accesses.iter().enumerate() {
                for (dst, &current) in accesses.iter().enumerate().skip(src + 1) {
                    if !is_write_access(previous) && !is_write_access(current) {
                        continue;
                    }
                    let mut edge = SubpassDependency::new(src as u32, dst as u32);
                    (edge.src_stage_mask, edge.src_access_mask) =
                        pipeline_stage_access_flags(previous);
                    (edge.dst_stage_mask, edge.dst_access_mask) =
                        pipeline_stage_access_flags(current);
                    expected.push(edge);
                }
            }
            for (dst, &access) in accesses.iter().enumerate() {
                let mut edge = SubpassDependency::new(vk::SUBPASS_EXTERNAL, dst as u32);
                edge.src_stage_mask =
                    vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::COMPUTE_SHADER;
                edge.src_access_mask = vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
                (edge.dst_stage_mask, edge.dst_access_mask) = pipeline_stage_access_flags(access);
                expected.push(edge);
            }
            assert_eq!(deps, expected, "{accesses:?}");
            super::SUBPASS_DEPENDENCY.with_borrow(|scratch| {
                let writers = accesses
                    .iter()
                    .filter(|&&access| is_write_access(access))
                    .count();
                assert_eq!(scratch.history[0].len(), writers);
                assert_eq!(
                    scratch.buffer_reader_history[0].len(),
                    if writers == 0 {
                        0
                    } else {
                        accesses.len() - writers
                    }
                );
            });
        }
    }

    #[test]
    fn locality_checks_every_repeated_access_range_in_either_order() {
        for reverse in [false, true] {
            for grouped in [false, true] {
                let mut execs = vec![SubpassFixture::color_attachment_exec(LoadOp::Load); 3];
                for (index, range) in [0..1, 0..2].into_iter().enumerate() {
                    execs[if grouped { index } else { 0 }].accesses.push(
                        1,
                        SubresourceAccess {
                            access: AccessType::ColorAttachmentWrite,
                            subresource: SubresourceRange::Image(color_subresource_range(
                                range,
                                0..1,
                            )),
                        },
                    );
                }
                if reverse {
                    if grouped {
                        execs.swap(0, 1);
                    } else {
                        execs[0].accesses.get_mut(&1).unwrap().reverse();
                    }
                }
                let pass = SubpassFixture::subpass_command(execs);
                let dependencies = Submission::build_subpass_dependencies(
                    &pass,
                    &[PipelineStageAccessFlags::default(); 2],
                    &[0, 0, 1],
                );
                let edge = dependencies
                    .iter()
                    .find(|edge| edge.src_subpass == 0 && edge.dst_subpass == 1)
                    .unwrap();
                assert!(
                    edge.dependency_flags.is_empty(),
                    "reverse={reverse} grouped={grouped}"
                );
            }
        }
    }

    #[test]
    fn locality_flags_intersect_every_retained_contribution() {
        for flags in [
            [vk::DependencyFlags::BY_REGION; 2],
            [vk::DependencyFlags::BY_REGION, vk::DependencyFlags::empty()],
            [vk::DependencyFlags::empty(), vk::DependencyFlags::BY_REGION],
        ] {
            let mut scratch = super::SubpassDependencyScratch::default();
            scratch.reset(0, 2);
            for flag in flags {
                scratch.record_dependency(
                    0,
                    1,
                    PipelineStageAccessFlags::new(AccessType::ColorAttachmentWrite),
                    PipelineStageAccessFlags::new(
                        AccessType::FragmentShaderReadColorInputAttachment,
                    ),
                    flag,
                );
            }
            let mut dependencies = Vec::new();
            scratch.flush_dependencies(&mut dependencies);
            assert_eq!(dependencies[0].dependency_flags, flags[0] & flags[1]);
        }
    }

    #[test]
    fn locality_image_read_pruning_requires_a_pass_wide_proof() {
        for veto in 0..7 {
            let mut pass = SubpassFixture::subpass_command(vec![Execution::default(); 3]);
            for exec in &mut pass.execs {
                exec.accesses.push(
                    0,
                    SubresourceAccess {
                        access: AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                        subresource: SubresourceRange::Image(color_subresource_range(0..1, 0..1)),
                    },
                );
            }
            match veto {
                0 => {}
                1 => {
                    pass.execs[1].accesses.get_mut(&0).unwrap()[0].access =
                        AccessType::FragmentShaderReadOther
                }
                2 => {
                    pass.execs[1].accesses.get_mut(&0).unwrap()[0].access =
                        AccessType::FragmentShaderWrite
                }
                // Even an operation excluded from subpass scopes must veto the proof.
                3 => {
                    pass.execs[1].accesses.get_mut(&0).unwrap()[0].access = AccessType::TransferRead
                }
                4..=6 => {
                    let mut attachment = SubpassFixture::color_attachment_exec(LoadOp::Load);
                    let state = attachment.attachments.color[0].as_mut().unwrap();
                    state.attachment.target = 0;
                    state.is_attachment = veto == 4;
                    state.is_input = veto == 5;
                    if veto == 6 {
                        state.resolve = Some(ColorResolve {
                            attachment: state.attachment,
                            src_attachment_idx: 0,
                        });
                    }
                    pass.execs[1].attachments = attachment.attachments;
                }
                _ => unreachable!(),
            }
            for (changed, (src, dst)) in [(0, (1, 2)), (1, (0, 2)), (2, (0, 1))] {
                pass.execs.swap(1, changed);
                let deps = Submission::build_subpass_dependencies(
                    &pass,
                    &[PipelineStageAccessFlags::default()],
                    &[0, 1, 2],
                );
                let readers = deps
                    .iter()
                    .find(|dep| dep.src_subpass == src && dep.dst_subpass == dst);
                assert_eq!(
                    readers.is_some(),
                    veto != 0,
                    "veto {veto}, execution {changed}"
                );
                if let Some(readers) = readers {
                    assert!(readers.dependency_flags.is_empty());
                }
                pass.execs.swap(1, changed);
            }
        }
    }

    #[test]
    fn locality_later_group_writes_contaminate_read_summaries_in_either_order() {
        for reverse in [false, true] {
            let mut execs = vec![
                SubpassFixture::exec_with_buffer_access(AccessType::VertexShaderReadUniformBuffer),
                SubpassFixture::exec_with_buffer_access(AccessType::FragmentShaderWrite),
                SubpassFixture::exec_with_buffer_access(AccessType::IndexBuffer),
            ];
            if reverse {
                execs.swap(0, 1);
            }
            let pass = SubpassFixture::subpass_command(execs);
            let deps = Submission::build_subpass_dependencies(
                &pass,
                &[PipelineStageAccessFlags::default()],
                &[0, 0, 1],
            );
            let edge = deps
                .iter()
                .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 1)
                .unwrap();
            assert!(
                edge.src_access_mask
                    .contains(vk::AccessFlags::UNIFORM_READ | vk::AccessFlags::SHADER_WRITE)
            );
            assert!(edge.dependency_flags.is_empty());
        }
    }

    #[test]
    fn locality_prunes_shared_reads_but_intersects_storage_hazards_in_either_order() {
        for attachment_node in [0, 2] {
            for resource in [None, Some(false), Some(true)] {
                for write in [false, true] {
                    let mut pass = SubpassFixture::subpass_command(vec![
                            SubpassFixture::color_attachment_exec(
                                LoadOp::Load
                            );
                            2
                        ]);
                    for (idx, exec) in pass.execs.iter_mut().enumerate() {
                        let state = exec.attachments.color[0].as_mut().unwrap();
                        state.attachment.target = attachment_node;
                        state.is_attachment = idx == 0;
                        state.is_input = idx == 1;
                        state.store = StoreOp::DontCare;
                        // Different multiview masks do not change an exact layer mapping.
                        exec.view_mask = idx as u32 + 1;
                        exec.accesses.push(
                            attachment_node,
                            SubresourceAccess {
                                access: if idx == 0 {
                                    AccessType::ColorAttachmentWrite
                                } else {
                                    AccessType::FragmentShaderReadColorInputAttachment
                                },
                                subresource: SubresourceRange::Image(color_subresource_range(
                                    0..1,
                                    0..1,
                                )),
                            },
                        );
                        if let Some(image) = resource {
                            exec.accesses.push(
                                1,
                                SubresourceAccess {
                                    access: if write && idx == 0 {
                                        AccessType::FragmentShaderWrite
                                    } else if image {
                                        AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer
                                    } else {
                                        AccessType::VertexShaderReadUniformBuffer
                                    },
                                    subresource: if image {
                                        SubresourceRange::Image(color_subresource_range(0..1, 0..1))
                                    } else {
                                        SubresourceRange::Buffer((0..16).into())
                                    },
                                },
                            );
                        }
                    }
                    let deps = Submission::build_subpass_dependencies(
                        &pass,
                        &[PipelineStageAccessFlags::new(AccessType::TransferWrite); 3],
                        &[0, 1],
                    );
                    let edge = deps
                        .iter()
                        .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 1)
                        .unwrap();
                    assert_eq!(
                        edge.dependency_flags,
                        if resource.is_some() && write {
                            vk::DependencyFlags::empty()
                        } else {
                            vk::DependencyFlags::BY_REGION
                        }
                    );
                    assert!(edge.dst_access_mask.contains(
                        vk::AccessFlags::INPUT_ATTACHMENT_READ
                            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    ));
                    if resource.is_some() && !write {
                        assert!(!edge.dst_access_mask.intersects(
                            vk::AccessFlags::UNIFORM_READ | vk::AccessFlags::SHADER_READ
                        ));
                    }
                    for dst in [0, 1] {
                        let external = deps
                            .iter()
                            .find(|dep| {
                                dep.src_subpass == vk::SUBPASS_EXTERNAL && dep.dst_subpass == dst
                            })
                            .unwrap();
                        assert!(external.dependency_flags.is_empty());
                        assert!(
                            external
                                .src_stage_mask
                                .contains(vk::PipelineStageFlags::TRANSFER)
                        );
                        assert!(
                            external
                                .src_access_mask
                                .contains(vk::AccessFlags::MEMORY_WRITE)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn locality_requires_exact_single_sample_ordinary_attachment_roles() {
        let mutations: &[fn(&mut Execution)] = &[
            |exec| {
                exec.attachments.color[0]
                    .as_mut()
                    .unwrap()
                    .attachment
                    .base_array_layer = 1
            },
            |exec| {
                exec.attachments.color[0]
                    .as_mut()
                    .unwrap()
                    .attachment
                    .array_layer_count = 2
            },
            |exec| {
                exec.attachments.color[0]
                    .as_mut()
                    .unwrap()
                    .attachment
                    .base_mip_level = 1
            },
            |exec| {
                exec.attachments.color[0]
                    .as_mut()
                    .unwrap()
                    .attachment
                    .mip_level_count = 2
            },
            |exec| {
                exec.attachments.color[0]
                    .as_mut()
                    .unwrap()
                    .attachment
                    .aspect_mask = vk::ImageAspectFlags::DEPTH
            },
            |exec| {
                exec.attachments.color[0]
                    .as_mut()
                    .unwrap()
                    .attachment
                    .format = vk::Format::R32_SFLOAT
            },
            |exec| {
                exec.attachments.color[0]
                    .as_mut()
                    .unwrap()
                    .attachment
                    .sample_count = SampleCount::Type4
            },
            |exec| {
                let state = exec.attachments.color[0].as_mut().unwrap();
                state.resolve = Some(ColorResolve {
                    attachment: state.attachment,
                    src_attachment_idx: 0,
                });
            },
            |exec| {
                exec.accesses.push(
                    1,
                    SubresourceAccess {
                        access: AccessType::ColorAttachmentWrite,
                        subresource: SubresourceRange::Image(color_subresource_range(0..1, 1..2)),
                    },
                )
            },
            |exec| {
                exec.accesses.push(
                    1,
                    SubresourceAccess {
                        access: AccessType::FragmentShaderReadColorInputAttachment,
                        subresource: SubresourceRange::Image(color_subresource_range(0..1, 0..1)),
                    },
                )
            },
            |exec| {
                exec.accesses.push(
                    1,
                    SubresourceAccess {
                        access: AccessType::TransferWrite,
                        subresource: SubresourceRange::Image(color_subresource_range(0..1, 0..1)),
                    },
                )
            },
        ];
        for (case, mutate) in mutations.iter().enumerate() {
            for changed_exec in [0, 1, 2] {
                let mut pass = SubpassFixture::subpass_command(vec![
                        SubpassFixture::color_attachment_exec(
                            LoadOp::Load
                        );
                        3
                    ]);
                mutate(&mut pass.execs[changed_exec]);
                let deps = Submission::build_subpass_dependencies(
                    &pass,
                    &[PipelineStageAccessFlags::default(); 2],
                    &[0, 0, 1],
                );
                let edge = deps
                    .iter()
                    .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 1)
                    .unwrap();
                assert!(
                    edge.dependency_flags.is_empty(),
                    "case {case}, execution {changed_exec}"
                );
            }
        }
        // Equal multisampled footprints are still not an ordinary single-sample proof.
        let mut pass = SubpassFixture::subpass_command(vec![
            SubpassFixture::color_attachment_exec(
                LoadOp::Load
            );
            2
        ]);
        for exec in &mut pass.execs {
            let state = exec.attachments.color[0].as_mut().unwrap();
            state.is_input = true;
            state.is_attachment = false;
            state.attachment.sample_count = SampleCount::Type4;
        }
        let deps = Submission::build_subpass_dependencies(
            &pass,
            &[PipelineStageAccessFlags::default(); 2],
            &[0, 1],
        );
        assert!(deps.iter().all(|dep| dep.dependency_flags.is_empty()));
    }

    #[test]
    fn locality_resolve_source_roles_are_global_even_without_multisampling() {
        for depth in [false, true] {
            let mut exec = if depth {
                SubpassFixture::depth_attachment_exec(LoadOp::Load, StoreOp::Store)
            } else {
                SubpassFixture::color_attachment_exec(LoadOp::Load)
            };
            if depth {
                let state = exec.attachments.depth_stencil.as_mut().unwrap();
                state.resolve = Some(DepthStencilResolve {
                    attachment: Attachment {
                        target: 2,
                        ..state.attachment
                    },
                    dst_attachment_idx: 1,
                    depth_mode: None,
                    stencil_mode: None,
                });
            } else {
                let mut destination = *exec.attachments.color[0].as_ref().unwrap();
                destination.attachment.target = 2;
                destination.is_attachment = false;
                destination.resolve = Some(ColorResolve {
                    attachment: destination.attachment,
                    src_attachment_idx: 0,
                });
                exec.attachments.color.push(Some(destination));
            }
            let mut consumer = exec.clone();
            consumer.attachments.color.truncate(1);
            if let Some(state) = consumer.attachments.depth_stencil.as_mut() {
                state.resolve = None;
            }
            let pass = SubpassFixture::subpass_command(vec![exec, consumer]);
            let deps = Submission::build_subpass_dependencies(
                &pass,
                &[PipelineStageAccessFlags::default(); 3],
                &[0, 1],
            );
            assert!(deps.iter().all(|dep| dep.dependency_flags.is_empty()));
        }
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn merged_graphics_commands_preserve_execution_tracking() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let pipeline = SubpassFixture::test_triangle_pipeline(&device)?;
        let mut graph = Graph::new();
        let image = graph.bind_resource(Image::create(
            &device,
            ImageInfo::image_2d(
                4,
                4,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
        )?);

        let mut first = graph.begin_cmd();
        let first_execution = first.track_execution();
        first
            .bind_pipeline(&pipeline)
            .color_attachment_image(0, image, LoadOp::CLEAR_BLACK_ALPHA_ZERO, StoreOp::Store)
            .record_cmd(|cmd| {
                cmd.draw(3, 1, 0, 0);
            });

        let mut second = graph.begin_cmd();
        let second_execution = second.track_execution();
        second
            .bind_pipeline(&pipeline)
            .color_attachment_image(0, image, LoadOp::Load, StoreOp::Store)
            .record_cmd(|cmd| {
                cmd.draw(3, 1, 0, 0);
            });

        let mut submission = graph.finalize();
        let mut schedule = vec![0, 1];
        submission.merge_scheduled_cmds(&mut schedule);

        assert_eq!(submission.graph.cmds.len(), 1);
        assert_eq!(first_execution.has_executed(), Ok(false));
        assert_eq!(second_execution.has_executed(), Ok(false));

        submission.graph.cmds[0].tracking.signal_executed();
        assert_eq!(first_execution.has_executed(), Ok(true));
        assert_eq!(second_execution.has_executed(), Ok(true));

        Ok(())
    }

    #[test]
    fn micromap_barrier_preserves_sync2_only_bits() {
        let barrier = Submission::memory_barrier2(GlobalBarrier {
            previous_accesses: &[AccessType::MicromapBuildWrite],
            next_accesses: &[AccessType::AccelerationStructureBuildMicromapRead],
        });

        assert_eq!(
            barrier.src_stage_mask,
            vk::PipelineStageFlags2::MICROMAP_BUILD_EXT
        );
        assert_eq!(
            barrier.src_access_mask,
            vk::AccessFlags2::MICROMAP_WRITE_EXT
        );
        assert_eq!(
            barrier.dst_stage_mask,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
        );
        assert_eq!(barrier.dst_access_mask, vk::AccessFlags2::MICROMAP_READ_EXT);
    }

    #[test]
    fn micromap_buffer_only_barriers_require_sync2() {
        for access in [
            AccessType::MicromapBuildInputRead,
            AccessType::MicromapBuildScratchReadWrite,
            AccessType::MicromapBuildBufferRead,
            AccessType::MicromapBuildBufferWrite,
        ] {
            let next_accesses = [access];
            let barriers = [BufferBarrier {
                previous_accesses: &[AccessType::TransferWrite],
                next_accesses: &next_accesses,
                ..Default::default()
            }];

            assert!(Submission::barriers_require_sync2(
                None,
                None,
                &barriers,
                &[]
            ));
        }
    }

    #[test]
    fn node_indexed_scratch_clear_resets_occupancy_and_reuses_entries() {
        let mut scratch = super::NodeIndexedScratch::default();

        scratch.push(1, 10);
        scratch.clear();

        assert!(scratch.indices.is_empty());
        assert_eq!(scratch.get(1), &[] as &[i32]);

        scratch.push(1, 11);
        scratch.push(1, 12);

        assert_eq!(scratch.indices, vec![1]);
        assert_eq!(scratch.get(1), &[11, 12]);
    }

    #[test]
    fn node_indexed_scratch_resizes_for_high_indices() {
        let mut scratch = super::NodeIndexedScratch::default();

        scratch.push(5, 50);

        assert_eq!(scratch.indices, vec![5]);
        assert_eq!(scratch.get(5), &[50]);
        assert_eq!(scratch.get(4), &[] as &[i32]);
    }

    #[test]
    fn node_indexed_scratch_tracks_each_node_once() {
        let mut scratch = super::NodeIndexedScratch::default();

        scratch.push(2, 20);
        scratch.push(2, 21);
        scratch.push(0, 10);

        assert_eq!(scratch.indices, vec![2, 0]);
        assert_eq!(scratch.get(2), &[20, 21]);
        assert_eq!(scratch.get(0), &[10]);
        assert_eq!(scratch.get(1), &[] as &[i32]);
    }

    #[test]
    fn node_selection_follows_transitive_resource_set_prefix() {
        let resource_set_idx = ResourceSetIndex::new(0);
        let access_index = CommandAccessIndex {
            /*
            Cmd 0: set S
            Cmd 1: set S, node A
            Cmd 2: node A, target T
            */
            cmds_by_node: vec![vec![1, 2], vec![2]],
            accessed_nodes_by_cmd: vec![vec![], vec![0], vec![0, 1]],
            cmds_by_resource_set: vec![vec![0, 1]],
            accessed_resource_sets_by_cmd: vec![
                vec![resource_set_idx],
                vec![resource_set_idx],
                vec![],
            ],
        };
        let mut schedule = Schedule {
            access_index,
            ..Default::default()
        };

        schedule.schedule_required_node_prefixes([(1, 3)]);

        assert_eq!(schedule.cmds, vec![0, 1, 2]);
    }

    #[test]
    fn node_selection_revisits_node_at_later_boundary() {
        let mut graph = Graph::new();
        for _ in 0..4 {
            graph.bind_stream_arg_resource(AnyResource::BufferArg(BufferInfo::device_mem(
                1,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            )));
        }
        graph.cmds = vec![
            command_with_accesses(&[(0, AccessType::TransferRead), (1, AccessType::TransferRead)]),
            command_with_accesses(&[(0, AccessType::TransferRead)]),
            command_with_accesses(&[(0, AccessType::TransferRead), (2, AccessType::TransferRead)]),
            command_with_accesses(&[
                (1, AccessType::TransferRead),
                (2, AccessType::TransferRead),
                (3, AccessType::TransferWrite),
            ]),
        ];

        let submission = Submission::new(graph);
        let mut schedule = Schedule::default();
        schedule.access_index.update(&submission.graph, 4);

        submission.schedule_node_cmds(1, 4, &mut schedule);

        assert_eq!(schedule.cmds, vec![0, 1, 2, 3]);
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn partial_recording_acquires_image_set_for_each_selected_batch() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let mut pool = HashPool::new(&device);
        let image = Arc::new(Image::create(
            &device,
            ImageInfo::image_2d(
                1,
                1,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED,
            ),
        )?);
        let resource_set = ImageSet::new([image])?;
        let mut graph = Graph::new();
        let resource_set_node = graph.bind_resource(&resource_set);
        let lhs = graph.bind_resource(Buffer::create(
            &device,
            BufferInfo::device_mem(4, vk::BufferUsageFlags::TRANSFER_DST),
        )?);
        let rhs = graph.bind_resource(Buffer::create(
            &device,
            BufferInfo::device_mem(4, vk::BufferUsageFlags::TRANSFER_DST),
        )?);

        for output in [lhs, rhs] {
            graph
                .begin_cmd()
                .resource_access(output, AccessType::TransferWrite)
                .resource_access(resource_set_node, ImageAccessType::SampledRead)
                .record_cmd(|_| {})
                .end_cmd();
        }

        let submission = graph.finalize();
        let mut cmd_buf = pool.resource(CommandBufferInfo::new(0))?;
        let mut fence = Fence::create(&device, false)?;
        cmd_buf.begin(
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let mut recording = submission.record(&mut pool, &mut cmd_buf, lhs)?;
        assert!(!recording.is_empty());
        recording.record(rhs)?;
        assert!(recording.is_empty());

        recording.cmd_buf.end()?;
        let mut recorded = recording.finish()?;
        recorded.queue_submit(&mut fence, 0, QueueSubmitInfo::QUEUE_SUBMIT)?;
        fence.wait()?;

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn partial_recording_transfers_only_unclaimed_buffer_overlap() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let mut graph = Graph::new();
        let buffer = graph.bind_resource(Buffer::create(
            &device,
            BufferInfo::device_mem(12, vk::BufferUsageFlags::TRANSFER_DST),
        )?);
        let first = BufferSubresourceRange { start: 0, end: 8 };
        let second = BufferSubresourceRange { start: 4, end: 12 };

        graph.resource(buffer).set_sharing_ranges(
            SharingMode::Exclusive(Some((1, 0))),
            &[BufferSubresourceRange { start: 0, end: 12 }],
        );
        graph
            .begin_cmd()
            .debug_name("touch first overlapping range")
            .subresource_access(buffer, first, AccessType::TransferWrite)
            .record_cmd(|_| {})
            .end_cmd();
        graph
            .begin_cmd()
            .debug_name("touch second overlapping range")
            .subresource_access(buffer, second, AccessType::TransferWrite)
            .record_cmd(|_| {})
            .end_cmd();

        let mut submission = graph.finalize();
        let mut ownership = RecordingOwnership::default();
        simulate_partial_transfer_discovery(
            &mut submission,
            &Schedule {
                cmds: vec![0],
                ..Default::default()
            },
            3,
            &mut ownership,
        );
        simulate_partial_transfer_discovery(
            &mut submission,
            &Schedule {
                cmds: vec![1],
                ..Default::default()
            },
            3,
            &mut ownership,
        );

        let mut released_ranges = submission
            .queue_ownership_release_groups
            .iter()
            .flat_map(|group| group.buffers.iter())
            .map(|(_, range)| *range)
            .collect::<Vec<_>>();
        released_ranges.sort_unstable_by_key(|range| (range.start, range.end));

        assert!(
            released_ranges
                .windows(2)
                .all(|ranges| ranges[0].end <= ranges[1].start),
            "released ranges overlap: {released_ranges:?}"
        );
        assert_eq!(
            released_ranges
                .iter()
                .map(|range| range.end - range.start)
                .sum::<vk::DeviceSize>(),
            12
        );

        Ok(())
    }

    #[test]
    fn pass_wide_read_only_nodes_need_only_external_dependencies() {
        for image in [false, true] {
            let pass = SubpassFixture::subpass_command(
                (0..1024)
                    .map(|sp| {
                        let mut exec = SubpassFixture::exec_with_buffer_access(if sp % 2 == 0 {
                            AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer
                        } else {
                            AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer
                        });
                        if image {
                            exec.accesses.get_mut(&0).unwrap()[0].subresource =
                                SubresourceRange::Image(color_subresource_range(0..1, 0..1));
                        }
                        exec
                    })
                    .collect(),
            );
            let deps = Submission::build_subpass_dependencies(
                &pass,
                &[PipelineStageAccessFlags::default()],
                &(0..1024).collect::<Vec<_>>(),
            );
            assert_eq!(deps.len(), 1024);
            for (sp, dep) in deps.iter().enumerate() {
                assert_eq!(
                    (dep.src_subpass, dep.dst_subpass),
                    (vk::SUBPASS_EXTERNAL, sp as u32)
                );
                assert_eq!(
                    dep.dst_stage_mask,
                    if sp % 2 == 0 {
                        vk::PipelineStageFlags::VERTEX_SHADER
                    } else {
                        vk::PipelineStageFlags::FRAGMENT_SHADER
                    }
                );
                assert_eq!(dep.dst_access_mask, vk::AccessFlags::SHADER_READ);
            }
        }
    }

    #[test]
    fn pending_transfer_nodes_remove_where_drops_stale_indices() {
        let mut pending = super::PendingTransferNodes::new(3);

        pending.push_transfer(1, 11, 21);
        pending.entries[1] = None;
        pending.remove_where(|_, _, _| false);

        assert!(pending.indices.is_empty());
        assert_eq!(pending.iter().count(), 0);
    }

    #[test]
    fn pending_transfer_nodes_remove_where_keeps_partially_consumed_node() {
        let mut pending = super::PendingTransferNodes::new(2);

        pending.push_transfer(1, 11, 20);
        pending.push_transfer(1, 11, 21);

        pending.remove_where(|_, _, transfers| {
            transfers.retain(|&transfer| transfer != 20);
            transfers.is_empty()
        });

        assert!(pending.contains(1));
        assert_eq!(pending_transfer_for_node(&pending, 1).unwrap().1, &[21]);
        assert!(!pending.is_empty());

        pending.remove_where(|_, _, transfers| {
            transfers.retain(|&transfer| transfer != 21);
            transfers.is_empty()
        });

        assert!(!pending.contains(1));
        assert!(pending_transfer_for_node(&pending, 1).is_none());
        assert!(pending.is_empty());
    }

    #[test]
    fn pending_transfer_nodes_remove_where_uses_swap_remove() {
        let mut pending = super::PendingTransferNodes::new(4);

        pending.push_transfer(0, 10, 20);
        pending.push_transfer(1, 11, 21);
        pending.push_transfer(2, 12, 22);

        pending.remove_where(|node_idx, _, _| node_idx == 1);

        assert!(pending_transfer_for_node(&pending, 1).is_none());
        assert_eq!(pending.indices.len(), 2);
        assert!(pending.indices.contains(&0));
        assert!(pending.indices.contains(&2));
        assert_eq!(pending.iter().collect::<Vec<_>>().len(), 2);
    }

    #[test]
    fn pending_transfer_nodes_set_tracks_each_node_once() {
        let mut pending = super::PendingTransferNodes::new(4);

        assert!(pending.push_transfer(2, 10, 20));
        assert!(!pending.push_transfer(2, 11, 21));

        assert!(pending.contains(2));
        let (handle, transfers) = pending_transfer_for_node(&pending, 2).unwrap();
        assert_eq!(handle, 11);
        assert_eq!(pending.indices, vec![2]);
        assert_eq!(transfers, &[20, 21]);
        assert_eq!(pending.iter().count(), 1);
    }

    #[test]
    fn physical_group_dependencies_retain_writers_through_intermediate_reads() {
        let pass = SubpassFixture::subpass_command(vec![
            SubpassFixture::exec_with_buffer_access(AccessType::FragmentShaderWrite),
            SubpassFixture::exec_with_buffer_access(AccessType::FragmentShaderReadOther),
            SubpassFixture::exec_with_buffer_access(AccessType::FragmentShaderReadOther),
            SubpassFixture::exec_with_buffer_access(AccessType::VertexShaderReadUniformBuffer),
        ]);
        let mut history = [PipelineStageAccessFlags::default(); 1];
        PipelineStageAccessFlags::record_external_accesses(
            &mut history,
            &command_with_accesses(&[(0, AccessType::TransferWrite)]),
        );
        PipelineStageAccessFlags::record_external_accesses(
            &mut history,
            &command_with_accesses(&[(0, AccessType::VertexBuffer)]),
        );
        let deps = Submission::build_subpass_dependencies(&pass, &history, &[0, 1, 2, 2]);
        let delayed = deps
            .iter()
            .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 2)
            .unwrap();
        assert!(
            delayed
                .src_access_mask
                .contains(vk::AccessFlags::SHADER_WRITE)
        );
        assert!(delayed.dst_stage_mask.contains(
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::VERTEX_SHADER
        ));
        assert!(
            delayed
                .dst_access_mask
                .contains(vk::AccessFlags::SHADER_READ | vk::AccessFlags::UNIFORM_READ)
        );
        let external = deps
            .iter()
            .find(|dep| dep.src_subpass == vk::SUBPASS_EXTERNAL && dep.dst_subpass == 2)
            .unwrap();
        assert!(
            external
                .src_stage_mask
                .contains(vk::PipelineStageFlags::TRANSFER)
        );
        assert!(
            external
                .src_access_mask
                .contains(vk::AccessFlags::MEMORY_WRITE)
        );
        assert_eq!(external.dst_access_mask, delayed.dst_access_mask);
        assert!(external.dependency_flags.is_empty());
    }

    #[test]
    fn physical_group_incoming_dependencies_union_all_reader_stages_and_access_classes() {
        let pass = SubpassFixture::subpass_command(vec![
            SubpassFixture::exec_with_buffer_access(AccessType::FragmentShaderWrite),
            SubpassFixture::exec_with_buffer_access(AccessType::VertexShaderReadUniformBuffer),
            SubpassFixture::exec_with_buffer_access(AccessType::VertexShaderReadOther),
            SubpassFixture::exec_with_buffer_access(AccessType::IndexBuffer),
            SubpassFixture::exec_with_buffer_access(AccessType::FragmentShaderReadOther),
        ]);
        let deps = Submission::build_subpass_dependencies(
            &pass,
            &[PipelineStageAccessFlags::default(); 1],
            &[0, 1, 1, 1, 1],
        );
        let incoming = deps
            .iter()
            .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 1)
            .unwrap();
        assert!(
            incoming
                .src_access_mask
                .contains(vk::AccessFlags::SHADER_WRITE)
        );
        assert!(incoming.dst_stage_mask.contains(
            vk::PipelineStageFlags::VERTEX_SHADER
                | vk::PipelineStageFlags::VERTEX_INPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER
        ));
        assert!(incoming.dst_access_mask.contains(
            vk::AccessFlags::UNIFORM_READ
                | vk::AccessFlags::INDEX_READ
                | vk::AccessFlags::SHADER_READ
        ));
        assert!(
            deps.iter()
                .all(|dep| dep.src_subpass != dep.dst_subpass && dep.dependency_flags.is_empty())
        );
    }

    #[test]
    fn physical_grouping_matches_actual_roles_layouts_samples_and_multiview_metadata() {
        let pass = SubpassFixture::subpass_command(vec![
            SubpassFixture::color_attachment_exec(
                LoadOp::Load
            );
            2
        ]);
        let (info, _) = SubpassFixture::plan_subpasses(&pass);
        let graphics = [GraphicsExecutionInfo {
            input_attachments: &[],
            sample_count: SampleCount::Type1,
        }; 2];
        let mutations: &[fn(&mut SubpassInfo)] = &[
            |s| s.color_attachments[0].layout = vk::ImageLayout::GENERAL,
            |s| s.color_attachments[0].aspect_mask = vk::ImageAspectFlags::DEPTH,
            |s| s.color_attachments[0].attachment = vk::ATTACHMENT_UNUSED,
            |s| s.input_attachments.push(s.color_attachments[0]),
            |s| s.color_resolve_attachments[0] = s.color_attachments[0],
            |s| s.depth_stencil_attachment = Some(s.color_attachments[0]),
            |s| s.view_mask = 3,
            |s| s.correlated_view_mask = 3,
        ];
        for mutate in mutations {
            let mut logical = vec![info.subpasses[0].clone(); 2];
            mutate(&mut logical[1]);
            assert_eq!(
                &*Submission::coalesce_subpasses(&pass, &graphics, &mut logical),
                &[0, 1]
            );
        }
        let mut changed_graphics = graphics;
        changed_graphics[1].sample_count = SampleCount::Type4;
        assert_eq!(
            &*Submission::coalesce_subpasses(
                &pass,
                &changed_graphics,
                &mut vec![info.subpasses[0].clone(); 2]
            ),
            &[0, 1]
        );
        // Reflection alone is enough to exclude an input execution.
        changed_graphics = graphics;
        changed_graphics[1].input_attachments = &[0];
        assert_eq!(
            &*Submission::coalesce_subpasses(
                &pass,
                &changed_graphics,
                &mut vec![info.subpasses[0].clone(); 2]
            ),
            &[0, 1]
        );

        let mut pass = pass;
        for exec in &mut pass.execs {
            exec.view_mask = 3;
            exec.correlated_view_mask = 3;
        }
        let (info, mapping) = SubpassFixture::plan_subpasses(&pass);
        assert_eq!(&*mapping, &[0, 0]);
        assert_eq!(info.subpasses[0].view_mask, 3);
        assert_eq!(info.subpasses[0].correlated_view_mask, 3);
        pass.execs[1].view_mask = 1;
        assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 1]);
        pass.execs[1].view_mask = 3;
        pass.execs[1].correlated_view_mask = 1;
        assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 1]);
    }

    #[test]
    fn physical_grouping_requires_exact_attachment_identity_and_slots() {
        let mutations: &[fn(&mut Attachment)] = &[
            |a| a.target += 1,
            |a| a.array_layer_count += 1,
            |a| a.base_array_layer += 1,
            |a| a.base_mip_level += 1,
            |a| a.mip_level_count += 1,
            |a| a.format = vk::Format::B8G8R8A8_UNORM,
            |a| a.sample_count = SampleCount::Type4,
            |a| a.aspect_mask = vk::ImageAspectFlags::DEPTH,
        ];
        for mutate in mutations {
            let mut pass = SubpassFixture::subpass_command(vec![
                    SubpassFixture::color_attachment_exec(
                        LoadOp::Load
                    );
                    2
                ]);
            let original = pass.execs[0].attachments.color[0].unwrap().attachment;
            let changed = &mut pass.execs[1].attachments.color[0]
                .as_mut()
                .unwrap()
                .attachment;
            mutate(changed);
            assert!(!Submission::attachments_are_exact(original, *changed));
            assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 1]);
        }
        let mut pass = SubpassFixture::subpass_command(vec![
            SubpassFixture::color_attachment_exec(
                LoadOp::Load
            );
            2
        ]);
        pass.execs[1].attachments.color.insert(0, None);
        assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 1]);
        let mut depth = SubpassFixture::depth_attachment_exec(LoadOp::Load, StoreOp::Store);
        let previous = depth.clone();
        let state = depth.attachments.depth_stencil.as_mut().unwrap();
        state.attachment.aspect_mask = vk::ImageAspectFlags::STENCIL;
        assert!(Attachment::are_identical(
            previous.attachments.depth_stencil.unwrap().attachment,
            state.attachment
        ));
        assert_eq!(
            &*SubpassFixture::plan_subpasses(&SubpassFixture::subpass_command(vec![
                previous, depth
            ]))
            .1,
            &[0, 1]
        );
    }

    #[test]
    fn physical_render_pass_cache_identity_does_not_include_execution_count() {
        use std::hash::{Hash, Hasher};
        let base = SubpassFixture::color_attachment_exec(LoadOp::Load);
        let (expected, _) =
            SubpassFixture::plan_subpasses(&SubpassFixture::subpass_command(vec![base.clone()]));
        for count in [2, 8, 64] {
            let (info, mapping) =
                SubpassFixture::plan_subpasses(&SubpassFixture::subpass_command(vec![
                    base.clone();
                    count
                ]));
            assert_eq!(mapping.len(), count);
            assert_eq!(info, expected);
            let mut lhs = std::collections::hash_map::DefaultHasher::new();
            let mut rhs = std::collections::hash_map::DefaultHasher::new();
            info.hash(&mut lhs);
            expected.hash(&mut rhs);
            assert_eq!(lhs.finish(), rhs.finish());
        }
    }

    #[test]
    fn preserve_lists_and_dependencies_use_physical_subpasses() {
        let mut writer = SubpassFixture::color_attachment_exec(LoadOp::Load);
        writer.attachments.depth_stencil =
            SubpassFixture::depth_attachment_exec(LoadOp::Load, StoreOp::Store)
                .attachments
                .depth_stencil;
        let depth = writer.attachments.depth_stencil.as_mut().unwrap();
        depth.attachment.target = 2;
        let mut gap = SubpassFixture::color_attachment_exec(LoadOp::Load);
        gap.attachments.color[0].as_mut().unwrap().attachment.target = 3;
        gap.attachments.color.insert(0, None);
        let mut input = writer.clone();
        input.attachments.color[0].as_mut().unwrap().is_input = true;
        let pass =
            SubpassFixture::subpass_command(vec![writer.clone(), writer, gap.clone(), gap, input]);
        let (mut info, mapping) = SubpassFixture::plan_subpasses(&pass);
        assert_eq!(&*mapping, &[0, 0, 1, 1, 2]);
        assert_eq!(info.subpasses.len(), 3);
        assert!(info.subpasses[0].preserve_attachments.is_empty());
        assert_eq!(info.subpasses[1].preserve_attachments, vec![0, 2]);
        assert!(info.subpasses[2].preserve_attachments.is_empty());
        let dep = info
            .dependencies
            .iter()
            .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 2)
            .unwrap();
        assert!(
            dep.src_access_mask
                .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        );
        assert!(
            dep.dst_access_mask
                .contains(vk::AccessFlags::INPUT_ATTACHMENT_READ)
        );
        assert!(
            info.dependencies
                .iter()
                .all(|dep| dep.dst_subpass < 3 && dep.src_subpass != dep.dst_subpass)
        );
        assert!(info.dependencies.windows(2).all(|pair| (
            pair[0].src_subpass,
            pair[0].dst_subpass
        ) < (
            pair[1].src_subpass,
            pair[1].dst_subpass
        )));
        let expected = info.subpasses.clone();
        for subpass in &mut info.subpasses {
            subpass.preserve_attachments.extend([0, 0, 2, 99]);
        }
        Submission::rebuild_preserve_attachments(&mut info.subpasses);
        assert_eq!(info.subpasses, expected);

        let (attachment, layout) =
            Submission::input_attachment_descriptor(&pass, &mapping, &info, 4, 0);
        assert_eq!(attachment.target, 1);
        assert_eq!(layout, info.subpasses[2].input_attachments[0].layout);
    }

    #[test]
    fn queue_ownership_release_groups_group_by_source_queue() {
        use super::QueueOwnershipReleaseGroup;

        let mut submission = Submission::new(Graph::new());
        let image = vk::Image::null();

        let range_a = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_array_layer: 0,
            layer_count: 1,
            base_mip_level: 0,
            level_count: 1,
        };
        let range_b = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_array_layer: 1,
            layer_count: 1,
            base_mip_level: 0,
            level_count: 1,
        };

        QueueOwnershipReleaseGroup::get_or_insert(
            &mut submission.queue_ownership_release_groups,
            1,
            2,
        )
        .images
        .push(ImageQueueOwnershipRelease {
            image,
            layouts: ImageOwnershipLayouts {
                old: vk::ImageLayout::GENERAL,
                new: vk::ImageLayout::GENERAL,
            },
            range: range_a,
        });
        QueueOwnershipReleaseGroup::get_or_insert(
            &mut submission.queue_ownership_release_groups,
            1,
            2,
        )
        .images
        .push(ImageQueueOwnershipRelease {
            image,
            layouts: ImageOwnershipLayouts {
                old: vk::ImageLayout::GENERAL,
                new: vk::ImageLayout::GENERAL,
            },
            range: range_b,
        });
        QueueOwnershipReleaseGroup::get_or_insert(
            &mut submission.queue_ownership_release_groups,
            4,
            5,
        )
        .images
        .push(ImageQueueOwnershipRelease {
            image,
            layouts: ImageOwnershipLayouts {
                old: vk::ImageLayout::GENERAL,
                new: vk::ImageLayout::GENERAL,
            },
            range: range_a,
        });

        let mut groups = submission.queue_ownership_release_groups;
        sort_queue_ownership_release_groups(&mut groups);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].images.len(), 2);
        assert_eq!(groups[1].images.len(), 1);
        assert_eq!(groups[0].images[0].image, image);
        assert!(ImageOwnershipTransfer::ranges_equal(
            groups[0].images[0].range,
            range_a
        ));
    }

    #[test]
    fn read_only_image_layouts_must_stay_stable_across_the_entire_group() {
        let mut pass = SubpassFixture::subpass_command(vec![
            SubpassFixture::color_attachment_exec(
                LoadOp::Load
            );
            3
        ]);
        pass.execs[0].accesses.push(
            3,
            SubresourceAccess {
                access: AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                subresource: SubresourceRange::Image(color_subresource_range(0..1, 0..1)),
            },
        );
        pass.execs[2].accesses.push(
            3,
            SubresourceAccess {
                access: AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer,
                subresource: SubresourceRange::Image(color_subresource_range(0..1, 0..1)),
            },
        );
        assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 0, 0]);
        pass.execs[2].accesses.get_mut(&3).unwrap()[0].access = AccessType::FragmentShaderReadOther;
        assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 0, 1]);
        pass.execs[0].accesses = pass.execs[2].accesses.clone();
        assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 1, 2]);
    }

    #[test]
    fn record_selection_from_node_creates_node_variant() {
        let node = BufferNode::new(
            7,
            #[cfg(feature = "checked")]
            crate::GraphId(1),
        );

        let selection = RecordSelection::from(node);

        match selection {
            RecordSelection::Node(AnyNode::Buffer(actual)) => assert_eq!(actual.index(), 7),
            _ => panic!("expected RecordSelection::Node(Buffer)"),
        }
    }

    #[test]
    fn record_selection_nodes_preserves_slice() {
        let lhs = AnyNode::from(BufferNode::new(
            1,
            #[cfg(feature = "checked")]
            crate::GraphId(1),
        ));
        let rhs = AnyNode::from(BufferNode::new(
            2,
            #[cfg(feature = "checked")]
            crate::GraphId(1),
        ));
        let nodes = [lhs, rhs];

        match RecordSelection::nodes(&nodes) {
            RecordSelection::Nodes(actual) => assert_eq!(actual.len(), 2),
            _ => panic!("expected RecordSelection::Nodes"),
        }
    }

    #[test]
    fn record_subpass_dependency_keeps_cross_stage_hazards() {
        let mut scratch = super::SubpassDependencyScratch::default();
        scratch.reset(0, 2);
        let current = PipelineStageAccessFlags {
            stage_flags: vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_flags: vk::AccessFlags::SHADER_READ,
        };

        scratch.record_dependency(
            0,
            1,
            PipelineStageAccessFlags {
                stage_flags: vk::PipelineStageFlags::VERTEX_SHADER,
                access_flags: vk::AccessFlags::SHADER_WRITE,
            },
            current,
            vk::DependencyFlags::empty(),
        );

        let mut dependencies = Vec::new();
        scratch.flush_dependencies(&mut dependencies);
        let dep = &dependencies[0];
        assert_eq!(dep.src_stage_mask, vk::PipelineStageFlags::VERTEX_SHADER);
        assert_eq!(dep.src_access_mask, vk::AccessFlags::SHADER_WRITE);
        assert_eq!(dep.dst_stage_mask, vk::PipelineStageFlags::FRAGMENT_SHADER);
        assert_eq!(dep.dst_access_mask, vk::AccessFlags::SHADER_READ);
        assert!(dep.dependency_flags.is_empty());
    }

    #[test]
    fn record_subpass_dependency_unions_complete_scopes() {
        let mut scratch = super::SubpassDependencyScratch::default();
        scratch.reset(0, 3);
        let current = PipelineStageAccessFlags {
            stage_flags: vk::PipelineStageFlags::VERTEX_SHADER
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_flags: vk::AccessFlags::SHADER_READ,
        };

        scratch.record_dependency(
            0,
            2,
            PipelineStageAccessFlags {
                stage_flags: vk::PipelineStageFlags::VERTEX_SHADER,
                access_flags: vk::AccessFlags::SHADER_READ,
            },
            current,
            vk::DependencyFlags::empty(),
        );
        scratch.record_dependency(
            1,
            2,
            PipelineStageAccessFlags {
                stage_flags: vk::PipelineStageFlags::FRAGMENT_SHADER,
                access_flags: vk::AccessFlags::SHADER_READ,
            },
            current,
            vk::DependencyFlags::empty(),
        );

        let mut dependencies = Vec::new();
        scratch.flush_dependencies(&mut dependencies);
        let dep = dependencies
            .iter()
            .find(|dep| dep.src_subpass == 1 && dep.dst_subpass == 2)
            .expect("missing dependency for later matched stage");
        assert!(
            dep.dst_access_mask.contains(vk::AccessFlags::SHADER_READ),
            "later matched stage should retain destination access mask"
        );
        assert_eq!(dep.dst_stage_mask, current.stage_flags);
        assert!(dep.dependency_flags.is_empty());
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn recorded_submission_attach_updates_only_touched_buffer_ranges() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let mut graph = Graph::new();
        let buffer = graph.bind_resource(Buffer::create(
            &device,
            BufferInfo::device_mem(16, vk::BufferUsageFlags::TRANSFER_DST),
        )?);
        let range_a = BufferSubresourceRange { start: 0, end: 8 };
        let range_b = BufferSubresourceRange { start: 8, end: 16 };

        {
            let buffer_resource = graph.resource(buffer);
            buffer_resource.set_sharing_ranges(SharingMode::Exclusive(Some((1, 0))), &[range_a]);
            buffer_resource.set_sharing_ranges(SharingMode::Exclusive(Some((2, 0))), &[range_b]);

            buffer_resource
                .swap_access(AccessType::TransferRead, range_a)
                .for_each(drop);
            buffer_resource
                .swap_access(AccessType::TransferRead, range_b)
                .for_each(drop);
        }

        let mut submission = graph.finalize();
        submission
            .exclusive_buffer_ranges
            .insert(buffer.index(), vec![range_a]);

        let mut fence = Fence::create(&device, false)?;
        let cmd_buf = CommandBuffer::create(&device, CommandBufferInfo::new(3))?;
        cmd_buf.begin(
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        cmd_buf.end()?;
        let mut recorded = RecordedSubmission {
            cmd_buf,
            queue_ownership_release_waits: Vec::new(),
            state: Arc::new(Mutex::new(RecordedSubmissionState {
                submission,
                _releases: Vec::new(),
                executed: false,
            })),
        };

        recorded.queue_submit(&mut fence, 0, QueueSubmitInfo::QUEUE_SUBMIT)?;

        let state = recorded.state.lock().expect("poisoned recorded state");
        let sync_info = state.submission.graph.resource(buffer).sync_info();
        let mut ranges = sync_info.ranges.into_vec();
        ranges.sort_unstable_by_key(|range| (range.range.start, range.range.end));

        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].queue_family_index, Some(3));
        assert_eq!(ranges[1].queue_family_index, Some(2));

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn recorded_submission_attach_updates_only_touched_subresources() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let mut graph = Graph::new();
        let image = graph.bind_resource(Image::create(
            &device,
            ImageInfo::image_2d_array(1, 1, 2, vk::Format::R8_UINT, vk::ImageUsageFlags::SAMPLED),
        )?);
        let range_a = color_subresource_range(0..1, 0..1);
        let range_b = color_subresource_range(1..2, 0..1);

        {
            let image_resource = graph.resource(image);
            image_resource.set_sharing_ranges(SharingMode::Exclusive(Some((1, 0))), &[range_a]);
            image_resource.set_sharing_ranges(SharingMode::Exclusive(Some((2, 0))), &[range_b]);

            image_resource
                .swap_access(AccessType::TransferRead, range_a)
                .for_each(drop);
            image_resource
                .swap_access(AccessType::TransferRead, range_b)
                .for_each(drop);
        }

        let mut submission = graph.finalize();
        submission
            .exclusive_image_ranges
            .insert(image.index(), vec![range_a]);

        let mut fence = Fence::create(&device, false)?;
        let cmd_buf = CommandBuffer::create(&device, CommandBufferInfo::new(3))?;
        cmd_buf.begin(
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        cmd_buf.end()?;
        let mut recorded = RecordedSubmission {
            cmd_buf,
            queue_ownership_release_waits: Vec::new(),
            state: Arc::new(Mutex::new(RecordedSubmissionState {
                submission,
                _releases: Vec::new(),
                executed: false,
            })),
        };

        recorded.queue_submit(&mut fence, 0, QueueSubmitInfo::QUEUE_SUBMIT)?;

        let state = recorded.state.lock().expect("poisoned recorded state");
        let sync_info = state.submission.graph.resource(image).sync_info();
        let mut subresources = sync_info.subresources.into_vec();
        sort_image_subresource_sync_infos(&mut subresources);

        assert_eq!(subresources.len(), 2);
        assert_eq!(subresources[0].queue_family_index, Some(3));
        assert_eq!(subresources[1].queue_family_index, Some(2));

        Ok(())
    }

    #[test]
    fn recording_ownership_deduplicates_image_set_images_by_physical_identity() {
        let mut ownership = RecordingOwnership::default();
        let image = PhysicalImageId::from_parts(1, 2);
        let aliased_handle = PhysicalImageId::from_parts(3, 2);
        let info =
            ImageInfo::image_2d_array(1, 1, 3, vk::Format::R8_UINT, vk::ImageUsageFlags::SAMPLED);
        let partial = color_subresource_range(0..2, 0..1);
        let whole = color_subresource_range(0..3, 0..1);

        let claimed = ownership.claim_image_set_image(image, info, partial);
        assert_eq!(claimed.len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            claimed[0], partial
        ));
        assert!(
            ownership
                .claim_image_set_image(image, info, partial)
                .is_empty()
        );

        let remaining = ownership.claim_image_set_image(image, info, whole);
        assert_eq!(remaining.len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            remaining[0],
            color_subresource_range(2..3, 0..1)
        ));
        let aliased = ownership.claim_image_set_image(aliased_handle, info, whole);
        assert_eq!(aliased.len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            aliased[0], whole
        ));
    }

    #[test]
    fn recording_ownership_keeps_whole_image_claims_uniform() {
        let mut ownership = RecordingOwnership::default();
        let info =
            ImageInfo::image_2d_array(1, 1, 3, vk::Format::R8_UINT, vk::ImageUsageFlags::SAMPLED)
                .into_builder()
                .mip_level_count(4)
                .build();
        let whole = color_subresource_range(0..3, 0..4);
        let partial = color_subresource_range(1..2, 2..3);

        let claimed = ownership.claim_image(0, info, whole);
        assert_eq!(claimed.len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            claimed[0], whole
        ));
        assert!(matches!(ownership.images[&0], ImageOwnership::Whole));

        assert!(ownership.claim_image(0, info, whole).is_empty());
        assert!(ownership.claim_image(0, info, partial).is_empty());
    }

    #[test]
    fn recording_ownership_only_returns_unclaimed_buffer_ranges() {
        let mut ownership = RecordingOwnership::default();
        let first = BufferSubresourceRange { start: 0, end: 8 };
        let overlap = BufferSubresourceRange { start: 4, end: 12 };

        assert_eq!(ownership.claim_buffer(0, first).as_slice(), &[first]);
        assert_eq!(
            ownership.claim_buffer(0, overlap).as_slice(),
            &[BufferSubresourceRange { start: 8, end: 12 }]
        );
        assert!(ownership.claim_buffer(0, overlap).is_empty());
    }

    #[test]
    fn recording_ownership_only_returns_unclaimed_image_subresources() {
        let mut ownership = RecordingOwnership::default();
        let info =
            ImageInfo::image_2d_array(1, 1, 3, vk::Format::R8_UINT, vk::ImageUsageFlags::SAMPLED);
        let first = color_subresource_range(0..2, 0..1);
        let overlap = color_subresource_range(1..3, 0..1);
        let remaining = color_subresource_range(2..3, 0..1);

        let claimed = ownership.claim_image(0, info, first);
        assert_eq!(claimed.len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            claimed[0], first
        ));

        let claimed = ownership.claim_image(0, info, overlap);
        assert_eq!(claimed.len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            claimed[0], remaining
        ));
        assert!(ownership.claim_image(0, info, overlap).is_empty());
    }

    #[test]
    fn recording_ownership_promotes_dense_claims_to_whole() {
        let mut ownership = RecordingOwnership::default();
        let info =
            ImageInfo::image_2d_array(1, 1, 3, vk::Format::R8_UINT, vk::ImageUsageFlags::SAMPLED);
        let partial = color_subresource_range(0..2, 0..1);
        let whole = color_subresource_range(0..3, 0..1);
        let remaining = color_subresource_range(2..3, 0..1);

        ownership.claim_image(0, info, partial);
        assert!(matches!(ownership.images[&0], ImageOwnership::Dense(_)));

        let claimed = ownership.claim_image(0, info, whole);
        assert_eq!(claimed.len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            claimed[0], remaining
        ));
        assert!(matches!(ownership.images[&0], ImageOwnership::Whole));
        assert!(ownership.claim_image(0, info, partial).is_empty());
    }

    #[test]
    fn recording_ownership_tracks_dual_aspects_without_dense_map() {
        let mut ownership = RecordingOwnership::default();
        let info = ImageInfo::image_2d(
            1,
            1,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        );
        let depth = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_array_layer: 0,
            layer_count: 1,
            base_mip_level: 0,
            level_count: 1,
        };
        let stencil = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::STENCIL,
            ..depth
        };

        let claimed = ownership.claim_image(0, info, depth);
        assert_eq!(claimed.len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            claimed[0], depth
        ));
        assert!(matches!(
            ownership.images[&0],
            ImageOwnership::DualAspect(mask) if mask == vk::ImageAspectFlags::DEPTH
        ));

        assert!(ownership.claim_image(0, info, depth).is_empty());
        let claimed = ownership.claim_image(0, info, stencil);
        assert_eq!(claimed.len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            claimed[0], stencil
        ));
        assert!(matches!(ownership.images[&0], ImageOwnership::Whole));
    }

    #[test]
    fn reorder_scheduled_cmds_allows_unrelated_moves_without_crossing_hazard() {
        fuzz::check_schedule_reordering(
            6,
            &[
                vec![
                    fuzz::ResourceAccess {
                        cmd_idx: 1,
                        write: true,
                    },
                    fuzz::ResourceAccess {
                        cmd_idx: 4,
                        write: true,
                    },
                ],
                vec![
                    fuzz::ResourceAccess {
                        cmd_idx: 0,
                        write: false,
                    },
                    fuzz::ResourceAccess {
                        cmd_idx: 2,
                        write: false,
                    },
                    fuzz::ResourceAccess {
                        cmd_idx: 5,
                        write: false,
                    },
                ],
                vec![fuzz::ResourceAccess {
                    cmd_idx: 3,
                    write: false,
                }],
            ],
        );
    }

    #[test]
    fn reorder_scheduled_cmds_combines_compressed_nodes_with_set_chain() {
        let resource_count = 12;
        let resource_set_idx = ResourceSetIndex::new(0);
        let mut schedule = Schedule {
            access_index: CommandAccessIndex {
                cmds_by_node: vec![vec![0, 1, 2]; resource_count],
                accessed_nodes_by_cmd: vec![(0..resource_count).collect(); 3],
                cmds_by_resource_set: vec![vec![0, 1, 2]],
                accessed_resource_sets_by_cmd: vec![vec![resource_set_idx]; 3],
            },
            cmds: vec![0, 1, 2],
            ..Default::default()
        };

        schedule.reorder_cmds(3);

        assert_eq!(schedule.predecessor_counts, vec![0, 13, 13]);
        assert_eq!(schedule.successors[0], vec![1, 1]);
        assert_eq!(schedule.successors[1], vec![2, 2]);
    }

    #[test]
    fn reorder_scheduled_cmds_groups_duplicate_resource_chains() {
        let resource_count = 12;
        let mut schedule = Schedule {
            access_index: CommandAccessIndex {
                cmds_by_node: vec![vec![0, 1, 2]; resource_count],
                accessed_nodes_by_cmd: vec![(0..resource_count).collect(); 3],
                ..Default::default()
            },
            cmds: vec![0, 1, 2],
            ..Default::default()
        };

        schedule.reorder_cmds(3);

        assert_eq!(schedule.cmds, vec![0, 1, 2]);
        assert_eq!(schedule.predecessor_counts, vec![0, 12, 12]);
        assert_eq!(schedule.successors[0], vec![1]);
        assert_eq!(schedule.successors[1], vec![2]);
    }

    #[test]
    fn reorder_scheduled_cmds_groups_ready_dependency_chain() {
        let mut schedule = schedule_with_access_index(
            &[0, 1, 2, 3],
            &[&[0, 1], &[1, 2], &[1, 3]],
            &[&[0], &[0, 1, 2], &[1], &[2]],
        );

        schedule.reorder_cmds(4);

        assert_eq!(schedule.cmds, vec![0, 1, 2, 3]);
    }

    #[test]
    fn reorder_scheduled_cmds_handles_noncontiguous_global_indices() {
        let mut schedule = schedule_with_access_index(
            &[1, 3, 5, 7],
            &[&[1, 5], &[3, 5, 7]],
            &[&[], &[0], &[], &[1], &[], &[0, 1], &[], &[1]],
        );

        schedule.reorder_cmds(8);

        assert_eq!(schedule.cmds, vec![1, 3, 5, 7]);
    }

    #[test]
    fn reorder_scheduled_cmds_keeps_disconnected_groups_deterministic() {
        let mut schedule = schedule_with_access_index(
            &[0, 1, 2, 3, 4],
            &[&[0, 1, 2], &[3, 4]],
            &[&[0], &[0], &[0], &[1], &[1]],
        );

        schedule.reorder_cmds(5);

        assert_eq!(schedule.cmds, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn reorder_scheduled_cmds_keeps_node_and_set_namespaces_independent() {
        let resource_set_idx = ResourceSetIndex::new(0);
        let mut schedule = Schedule {
            access_index: CommandAccessIndex {
                cmds_by_node: vec![vec![0, 1, 2]],
                accessed_nodes_by_cmd: vec![vec![0]; 3],
                cmds_by_resource_set: vec![vec![0, 1, 2]],
                accessed_resource_sets_by_cmd: vec![vec![resource_set_idx]; 3],
            },
            cmds: vec![0, 1, 2],
            ..Default::default()
        };

        schedule.reorder_cmds(3);

        assert_eq!(schedule.predecessor_counts, vec![0, 2, 2]);
        assert_eq!(schedule.successors[0], vec![1, 1]);
        assert_eq!(schedule.successors[1], vec![2, 2]);
    }

    #[test]
    fn reorder_scheduled_cmds_preserves_both_branches_before_join() {
        let mut schedule =
            schedule_with_access_index(&[0, 1, 2], &[&[0, 2], &[1, 2]], &[&[0], &[1], &[0, 1]]);

        schedule.reorder_cmds(3);

        assert_eq!(schedule.cmds, vec![0, 1, 2]);
    }

    #[test]
    fn reorder_scheduled_cmds_preserves_displaced_write_before_read() {
        fuzz::check_schedule_reordering(
            6,
            &[
                vec![
                    fuzz::ResourceAccess {
                        cmd_idx: 0,
                        write: false,
                    },
                    fuzz::ResourceAccess {
                        cmd_idx: 1,
                        write: true,
                    },
                    fuzz::ResourceAccess {
                        cmd_idx: 2,
                        write: false,
                    },
                    fuzz::ResourceAccess {
                        cmd_idx: 5,
                        write: true,
                    },
                ],
                vec![
                    fuzz::ResourceAccess {
                        cmd_idx: 0,
                        write: false,
                    },
                    fuzz::ResourceAccess {
                        cmd_idx: 3,
                        write: true,
                    },
                    fuzz::ResourceAccess {
                        cmd_idx: 4,
                        write: false,
                    },
                ],
            ],
        );
    }

    #[test]
    fn reorder_scheduled_cmds_preserves_hazards_from_command_access_index_update() {
        let cmds = vec![
            command_with_accesses(&[(0, AccessType::TransferRead)]),
            command_with_accesses(&[(1, AccessType::TransferWrite)]),
            command_with_accesses(&[(1, AccessType::TransferRead)]),
            command_with_accesses(&[(1, AccessType::TransferWrite)]),
            command_with_accesses(&[(0, AccessType::TransferRead)]),
        ];
        let mut access_index = CommandAccessIndex::default();
        access_index.update_from_cmds(&cmds, 2, 0);
        let mut schedule = Schedule {
            access_index,
            cmds: (0..cmds.len()).collect(),
            ..Default::default()
        };

        schedule.reorder_cmds(cmds.len());

        let position = |cmd_idx| {
            schedule
                .cmds
                .iter()
                .position(|&scheduled_cmd_idx| scheduled_cmd_idx == cmd_idx)
                .expect("command was not scheduled")
        };
        assert!(position(1) < position(2), "write-read hazard crossed");
        assert!(position(2) < position(3), "read-write hazard crossed");
    }

    #[test]
    fn reorder_scheduled_cmds_preserves_read_then_write_hazard() {
        fuzz::check_schedule_reordering(
            4,
            &[
                vec![
                    fuzz::ResourceAccess {
                        cmd_idx: 1,
                        write: false,
                    },
                    fuzz::ResourceAccess {
                        cmd_idx: 2,
                        write: true,
                    },
                ],
                vec![fuzz::ResourceAccess {
                    cmd_idx: 0,
                    write: false,
                }],
                vec![fuzz::ResourceAccess {
                    cmd_idx: 3,
                    write: false,
                }],
            ],
        );
    }

    #[test]
    fn reorder_scheduled_cmds_preserves_write_after_write_hazard() {
        fuzz::check_schedule_reordering(
            4,
            &[
                vec![
                    fuzz::ResourceAccess {
                        cmd_idx: 1,
                        write: true,
                    },
                    fuzz::ResourceAccess {
                        cmd_idx: 2,
                        write: true,
                    },
                ],
                vec![fuzz::ResourceAccess {
                    cmd_idx: 0,
                    write: false,
                }],
                vec![fuzz::ResourceAccess {
                    cmd_idx: 3,
                    write: false,
                }],
            ],
        );
    }

    #[test]
    fn reorder_scheduled_cmds_preserves_write_only_order() {
        let mut schedule = schedule_with_access_index(
            &[0, 1, 2, 3],
            /*
            Resource 0 is written by cmd 0 and read by cmd 3. Resource 1 is written by cmds 1 and
            2, so their relative order must be preserved even though neither cmd reads it.
            */
            &[&[0, 3], &[1, 2]],
            &[&[0], &[1], &[1], &[0]],
        );

        schedule.reorder_cmds(4);

        let cmd_1_position = schedule
            .cmds
            .iter()
            .position(|&cmd_idx| cmd_idx == 1)
            .expect("cmd 1 was not scheduled");
        let cmd_2_position = schedule
            .cmds
            .iter()
            .position(|&cmd_idx| cmd_idx == 2)
            .expect("cmd 2 was not scheduled");

        assert!(
            cmd_1_position < cmd_2_position,
            "write-only commands were reordered: {:?}",
            schedule.cmds
        );
    }

    #[test]
    fn reorder_scheduled_cmds_preserves_write_then_read_hazard() {
        fuzz::check_schedule_reordering(
            4,
            &[
                vec![
                    fuzz::ResourceAccess {
                        cmd_idx: 1,
                        write: true,
                    },
                    fuzz::ResourceAccess {
                        cmd_idx: 2,
                        write: false,
                    },
                ],
                vec![fuzz::ResourceAccess {
                    cmd_idx: 0,
                    write: false,
                }],
                vec![fuzz::ResourceAccess {
                    cmd_idx: 3,
                    write: false,
                }],
            ],
        );
    }

    #[test]
    fn reorder_scheduled_cmds_prioritizes_ready_resource_successors() {
        let mut schedule = schedule_with_access_index(
            &[0, 1, 2, 3, 4],
            &[&[0, 1, 3], &[0, 4], &[3]],
            &[&[0, 1], &[0], &[], &[0, 2], &[1]],
        );

        schedule.reorder_cmds(5);

        assert_eq!(schedule.cmds, vec![0, 1, 3, 4, 2]);
    }

    #[test]
    fn reorder_scheduled_cmds_ready_ties_use_original_order() {
        let mut schedule = schedule_with_access_index(
            &[0, 1, 2, 3, 4, 5],
            &[&[1, 2], &[1, 4], &[0, 1, 5]],
            &[&[2], &[0, 1, 2], &[0], &[], &[1], &[2]],
        );

        schedule.reorder_cmds(6);

        assert_eq!(schedule.cmds, vec![0, 1, 2, 4, 5, 3]);
    }

    #[test]
    fn reorder_scheduled_cmds_uses_one_resource_set_chain() {
        let resource_set_idx = ResourceSetIndex::new(0);
        let mut schedule = Schedule {
            access_index: CommandAccessIndex {
                cmds_by_node: vec![],
                accessed_nodes_by_cmd: vec![vec![]; 4],
                cmds_by_resource_set: vec![vec![0, 1, 2, 3]],
                accessed_resource_sets_by_cmd: vec![vec![resource_set_idx]; 4],
            },
            cmds: vec![0, 1, 2, 3],
            ..Default::default()
        };

        schedule.reorder_cmds(4);

        assert_eq!(schedule.cmds, vec![0, 1, 2, 3]);
        assert_eq!(schedule.predecessor_counts, vec![0, 1, 1, 1]);
        assert_eq!(schedule.successors[0], vec![1]);
        assert_eq!(schedule.successors[1], vec![2]);
        assert_eq!(schedule.successors[2], vec![3]);
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn repeated_partial_recording_does_not_duplicate_buffer_ownership_transfer()
    -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let mut graph = Graph::new();
        let buffer = graph.bind_resource(Buffer::create(
            &device,
            BufferInfo::device_mem(16, vk::BufferUsageFlags::TRANSFER_DST),
        )?);
        let range = BufferSubresourceRange { start: 0, end: 16 };

        graph
            .resource(buffer)
            .set_sharing_ranges(SharingMode::Exclusive(Some((1, 0))), &[range]);
        graph
            .begin_cmd()
            .debug_name("touch shared range")
            .subresource_access(buffer, range, AccessType::TransferWrite)
            .record_cmd(|_| {})
            .end_cmd();

        let mut submission = graph.finalize();
        let schedule = Schedule {
            cmds: vec![0],
            ..Default::default()
        };
        let mut ownership = RecordingOwnership::default();

        simulate_partial_transfer_discovery(&mut submission, &schedule, 3, &mut ownership);
        simulate_partial_transfer_discovery(&mut submission, &schedule, 3, &mut ownership);

        let released_ranges = submission
            .queue_ownership_release_groups
            .iter()
            .flat_map(|group| group.buffers.iter())
            .map(|(_, range)| *range)
            .collect::<Vec<_>>();
        assert_eq!(released_ranges, vec![range]);
        assert_eq!(
            submission.exclusive_buffer_ranges[&buffer.index()],
            vec![range]
        );

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn repeated_partial_recording_does_not_duplicate_image_ownership_transfer()
    -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let mut graph = Graph::new();
        let image = graph.bind_resource(Image::create(
            &device,
            ImageInfo::image_2d_array(1, 1, 2, vk::Format::R8_UINT, vk::ImageUsageFlags::SAMPLED),
        )?);
        let range = color_subresource_range(0..2, 0..1);

        graph
            .resource(image)
            .set_sharing_ranges(SharingMode::Exclusive(Some((1, 0))), &[range]);
        graph
            .begin_cmd()
            .debug_name("touch shared image range")
            .subresource_access(image, range, AccessType::TransferWrite)
            .record_cmd(|_| {})
            .end_cmd();

        let mut submission = graph.finalize();
        let schedule = Schedule {
            cmds: vec![0],
            ..Default::default()
        };
        let mut ownership = RecordingOwnership::default();

        simulate_partial_transfer_discovery(&mut submission, &schedule, 3, &mut ownership);
        let first_released_ranges = submission
            .queue_ownership_release_groups
            .iter()
            .flat_map(|group| group.images.iter())
            .map(|release| release.range)
            .collect::<Vec<_>>();
        simulate_partial_transfer_discovery(&mut submission, &schedule, 3, &mut ownership);

        let released_ranges = submission
            .queue_ownership_release_groups
            .iter()
            .flat_map(|group| group.images.iter())
            .map(|release| release.range)
            .collect::<Vec<_>>();
        assert_eq!(released_ranges.len(), first_released_ranges.len());
        assert!(
            released_ranges
                .iter()
                .zip(&first_released_ranges)
                .all(|(&lhs, &rhs)| super::ImageOwnershipTransfer::ranges_equal(lhs, rhs)),
            "released ranges changed: {first_released_ranges:?} -> {released_ranges:?}"
        );
        assert_eq!(submission.exclusive_image_ranges[&image.index()].len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            submission.exclusive_image_ranges[&image.index()][0],
            range
        ));

        Ok(())
    }

    #[test]
    fn retired_buffer_edges_leave_independent_attachment_locality_intact() {
        let mut exec = SubpassFixture::color_attachment_exec(LoadOp::Load);
        exec.accesses =
            SubpassFixture::exec_with_buffer_access(AccessType::FragmentShaderWrite).accesses;
        let deps = Submission::build_subpass_dependencies(
            &SubpassFixture::subpass_command(vec![exec; 3]),
            &[PipelineStageAccessFlags::default(); 2],
            &[0, 1, 2],
        );
        assert_eq!(deps.len(), 6);
        for dep in deps
            .iter()
            .filter(|dep| dep.src_subpass != vk::SUBPASS_EXTERNAL)
        {
            // The buffer's global 0 -> 1 -> 2 chain remains even though 0 -> 2 is local.
            let adjacent = dep.dst_subpass == dep.src_subpass + 1;
            assert_eq!(
                dep.dependency_flags,
                if adjacent {
                    vk::DependencyFlags::empty()
                } else {
                    vk::DependencyFlags::BY_REGION
                }
            );
            assert_eq!(
                dep.src_access_mask.contains(vk::AccessFlags::SHADER_WRITE),
                adjacent
            );
            assert_eq!(
                dep.dst_access_mask.contains(vk::AccessFlags::SHADER_WRITE),
                adjacent
            );
        }
    }

    #[test]
    fn sampled_image_reader_scope_control_still_coalesces_and_prunes() {
        let pass = SubpassFixture::subpass_command(
            [
                AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
                AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer,
                AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
            ]
            .into_iter()
            .map(|access| {
                let mut exec = SubpassFixture::color_attachment_exec(LoadOp::Load);
                exec.accesses.push(
                    0,
                    SubresourceAccess {
                        access,
                        subresource: SubresourceRange::Image(color_subresource_range(0..1, 0..1)),
                    },
                );
                exec
            })
            .collect(),
        );
        assert_eq!(&*SubpassFixture::plan_subpasses(&pass).1, &[0, 0, 0]);
        // Force distinct subpasses to exercise pruning independently of coalescing.
        let deps = Submission::build_subpass_dependencies(
            &pass,
            &[PipelineStageAccessFlags::default(); 2],
            &[0, 1, 2],
        );
        let internal = deps
            .iter()
            .filter(|edge| edge.src_subpass != vk::SUBPASS_EXTERNAL)
            .collect::<Vec<_>>();
        assert!(!internal.is_empty(), "attachment dependencies must remain");
        for edge in internal {
            assert_eq!(edge.dependency_flags, vk::DependencyFlags::BY_REGION);
            assert!(
                !edge
                    .src_access_mask
                    .intersects(vk::AccessFlags::SHADER_READ)
            );
            assert!(
                !edge
                    .dst_access_mask
                    .intersects(vk::AccessFlags::SHADER_READ)
            );
        }
    }

    #[test]
    fn sampled_read_barrier_elision_preserves_ownership_split() {
        use AccessType as A;

        let transferred_range = color_subresource_range(0..1, 0..1);
        let full_range = color_subresource_range(0..2, 0..1);
        let transfer = ImageOwnershipTransfer {
            dst_queue_family_index: 2,
            layouts: ImageOwnershipLayouts {
                old: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                new: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            range: transferred_range,
            src_queue_family_index: 1,
            src_queue_index: 0,
        };
        let barriers = super::TrackedImageBarrier::from_transfers(
            vk::Image::null(),
            ImageAccessSet::from_access(A::ComputeShaderReadSampledImageOrUniformTexelBuffer),
            A::ComputeShaderReadSampledImageOrUniformTexelBuffer,
            full_range,
            &[transfer],
            false,
        )
        .collect::<Vec<_>>();

        assert_eq!(barriers.len(), 2);
        let retained = barriers
            .iter()
            .filter(|&&barrier| !super::TrackedImageBarrier::can_elide_sampled_read(barrier))
            .collect::<Vec<_>>();
        let elided = barriers
            .iter()
            .filter(|&&barrier| super::TrackedImageBarrier::can_elide_sampled_read(barrier))
            .collect::<Vec<_>>();

        assert_eq!(retained.len(), 1);
        assert_eq!(elided.len(), 1);
        assert!(retained[0].ownership_layouts.is_some());
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            retained[0].range,
            transferred_range
        ));
        assert!(elided[0].ownership_layouts.is_none());
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            elided[0].range,
            color_subresource_range(1..2, 0..1)
        ));
    }

    #[test]
    fn sampled_read_barrier_elision_requires_existing_reader_stage() {
        use AccessType as A;

        let compute = A::ComputeShaderReadSampledImageOrUniformTexelBuffer;
        let ray_tracing = A::RayTracingShaderReadSampledImageOrUniformTexelBuffer;
        let broad = A::AnyShaderReadSampledImageOrUniformTexelBuffer;

        for barrier in [
            sampled_read_barrier(compute, compute),
            sampled_read_barrier(broad, ray_tracing),
        ] {
            assert!(super::TrackedImageBarrier::can_elide_sampled_read(barrier));
        }

        for barrier in [
            sampled_read_barrier(compute, ray_tracing),
            sampled_read_barrier(ray_tracing, compute),
            sampled_read_barrier(compute, broad),
        ] {
            assert!(!super::TrackedImageBarrier::can_elide_sampled_read(barrier));
        }

        let mut accumulated = sampled_read_barrier(compute, ray_tracing);
        accumulated.previous_accesses = accumulated.previous_accesses.after_access(ray_tracing);

        assert!(super::TrackedImageBarrier::can_elide_sampled_read(
            accumulated
        ));

        assert!(!super::TrackedImageBarrier::can_elide_sampled_read(
            sampled_read_barrier(compute, A::ComputeShaderWrite)
        ));

        let mut layout_transition = sampled_read_barrier(compute, ray_tracing);
        layout_transition.next_layout = vk_sync::ImageLayout::General;

        assert!(!super::TrackedImageBarrier::can_elide_sampled_read(
            layout_transition
        ));

        let mut ownership_transfer = sampled_read_barrier(compute, ray_tracing);
        ownership_transfer.ownership_layouts = Some(super::ImageOwnershipLayouts {
            old: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            new: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        });
        ownership_transfer.src_queue_family_index = 1;
        ownership_transfer.dst_queue_family_index = 2;

        assert!(!super::TrackedImageBarrier::can_elide_sampled_read(
            ownership_transfer
        ));
    }

    #[test]
    fn sampled_reader_barrier_uses_exact_source_stages() {
        let previous_accesses = ImageAccessSet::from_access(
            AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
        )
        .after_access(AccessType::RayTracingShaderReadSampledImageOrUniformTexelBuffer);
        let (src_stage_mask, dst_stage_mask, barrier) =
            super::TrackedImageBarrier::memory_barrier(super::TrackedImageBarrier {
                previous_accesses,
                next_access: AccessType::ComputeShaderWrite,
                previous_layout: super::TrackedImageBarrier::access_set_layout(previous_accesses),
                next_layout: super::TrackedImageBarrier::access_layout(
                    AccessType::ComputeShaderWrite,
                ),
                ownership_layouts: None,
                discard_contents: false,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: vk::Image::null(),
                range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            });

        assert_eq!(
            src_stage_mask,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR
        );
        assert_eq!(dst_stage_mask, vk::PipelineStageFlags::COMPUTE_SHADER);
        assert_eq!(barrier.src_access_mask, vk::AccessFlags::empty());
        assert_eq!(barrier.dst_access_mask, vk::AccessFlags::SHADER_WRITE);
        assert_eq!(
            barrier.old_layout,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
        assert_eq!(barrier.new_layout, vk::ImageLayout::GENERAL);
    }

    #[test]
    fn single_subpass_dependencies_match_general_aggregation() {
        let mut input = SubpassFixture::color_attachment_exec(LoadOp::DontCare);
        let state = input.attachments.color[0].as_mut().unwrap();
        state.is_attachment = false;
        state.is_input = true;
        state.store = StoreOp::DontCare;

        let mut color_resolve = SubpassFixture::color_attachment_exec(LoadOp::DontCare);
        let state = color_resolve.attachments.color[0].as_mut().unwrap();
        state.resolve = Some(ColorResolve {
            attachment: Attachment {
                target: 2,
                ..state.attachment
            },
            src_attachment_idx: 0,
        });
        let mut depth_resolve =
            SubpassFixture::depth_attachment_exec(LoadOp::Load, StoreOp::DontCare);
        let state = depth_resolve.attachments.depth_stencil.as_mut().unwrap();
        state.resolve = Some(DepthStencilResolve {
            attachment: Attachment {
                target: 2,
                ..state.attachment
            },
            dst_attachment_idx: 0,
            depth_mode: None,
            stencil_mode: None,
        });

        let mut repeated =
            SubpassFixture::exec_with_buffer_access(AccessType::VertexShaderReadUniformBuffer);
        repeated.accesses.push(
            0,
            SubresourceAccess {
                access: AccessType::FragmentShaderWrite,
                subresource: SubresourceRange::Buffer((0..16).into()),
            },
        );
        repeated.accesses.push(
            2,
            SubresourceAccess {
                access: AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                subresource: SubresourceRange::Image(color_subresource_range(0..1, 0..1)),
            },
        );
        let sets = command_with_resource_set_accesses(&[&[0, 1]]);
        let mut set_attachment = SubpassFixture::color_attachment_exec(LoadOp::Load);
        set_attachment.resource_set_accesses = sets.execs[0].resource_set_accesses.clone();

        let cases = [
            vec![],
            vec![Execution::default()],
            vec![Execution::default(); 3],
            vec![SubpassFixture::exec_with_buffer_access(
                AccessType::TransferWrite,
            )],
            vec![
                SubpassFixture::exec_with_buffer_access(AccessType::TransferWrite),
                SubpassFixture::exec_with_buffer_access(AccessType::VertexBuffer),
            ],
            vec![SubpassFixture::color_attachment_exec(LoadOp::DontCare)],
            vec![SubpassFixture::depth_attachment_exec(
                LoadOp::DontCare,
                StoreOp::DontCare,
            )],
            vec![input],
            vec![color_resolve],
            vec![depth_resolve],
            vec![sets.execs[0].clone()],
            vec![set_attachment],
            vec![
                repeated.clone(),
                SubpassFixture::exec_with_buffer_access(AccessType::IndexBuffer),
                repeated,
            ],
        ];
        for (case_idx, execs) in cases.into_iter().enumerate() {
            let pass = SubpassFixture::subpass_command(execs);
            let mapping = vec![0; pass.execs.len()];
            for previous in [
                AccessType::Nothing,
                AccessType::TransferWrite,
                AccessType::ComputeShaderWrite,
                AccessType::FragmentShaderWrite,
            ] {
                let mut history = [PipelineStageAccessFlags::new(previous); 3];
                history[2].union(PipelineStageAccessFlags::new(
                    AccessType::ComputeShaderReadOther,
                ));
                let fast = Submission::build_subpass_dependencies(&pass, &history, &mapping);
                let general =
                    Submission::build_general_subpass_dependencies(&pass, &history, &mapping);
                assert_eq!(fast, general, "case {case_idx}, history {previous:?}");
            }
        }
    }

    #[test]
    fn single_subpass_preserve_rebuild_clears_stale_entries() {
        let (mut info, _) = SubpassFixture::plan_subpasses(&SubpassFixture::subpass_command(vec![
            SubpassFixture::color_attachment_exec(LoadOp::Load),
        ]));
        info.subpasses[0].preserve_attachments.extend([0, 0, 99]);
        Submission::rebuild_preserve_attachments(&mut info.subpasses);
        assert!(info.subpasses[0].preserve_attachments.is_empty());
        Submission::rebuild_preserve_attachments(&mut []);
    }

    #[test]
    fn storage_image_reader_retirement_requires_identical_complete_groups() {
        for case in 0..9 {
            for changed in [1, 2] {
                let mut exec = Execution::default();
                exec.accesses.push(
                    0,
                    SubresourceAccess {
                        access: AccessType::FragmentShaderReadOther,
                        subresource: SubresourceRange::Image(color_subresource_range(0..1, 0..1)),
                    },
                );
                let mut execs = vec![exec; 4];
                let exec = &mut execs[changed];
                let access = &mut exec.accesses.get_mut(&0).unwrap()[0];
                match case {
                    0 => access.access = AccessType::VertexShaderReadOther,
                    1 => access.access = AccessType::FragmentShaderWrite,
                    2 => {
                        access.access =
                            AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer
                    }
                    3..=6 => {
                        access.subresource = SubresourceRange::Image(match case {
                            3 => color_subresource_range(0..2, 0..1),
                            4 => color_subresource_range(1..2, 0..1),
                            5 => color_subresource_range(0..1, 1..2),
                            _ => vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::DEPTH,
                                ..color_subresource_range(0..1, 0..1)
                            },
                        });
                    }
                    7 => {
                        let mut extra = *access;
                        extra.subresource =
                            SubresourceRange::Image(color_subresource_range(1..2, 0..1));
                        exec.accesses.push(0, extra);
                    }
                    8 => {
                        let mut attachment = SubpassFixture::color_attachment_exec(LoadOp::Load);
                        let state = attachment.attachments.color[0].as_mut().unwrap();
                        state.attachment.target = 0;
                        // Even metadata without an implicit scope must veto retirement.
                        state.is_attachment = false;
                        exec.attachments = attachment.attachments;
                    }
                    _ => unreachable!(),
                }
                exec.accesses.freeze();
                let deps = Submission::build_subpass_dependencies(
                    &SubpassFixture::subpass_command(execs),
                    &[PipelineStageAccessFlags::default()],
                    &[0, 1, 1, 2],
                );
                assert_eq!(
                    deps.iter()
                        .filter(|dep| dep.src_subpass != vk::SUBPASS_EXTERNAL)
                        .map(|dep| (dep.src_subpass, dep.dst_subpass))
                        .collect::<Vec<_>>(),
                    [(0, 1), (0, 2), (1, 2)],
                    "case={case} changed={changed}",
                );
                assert!(deps.iter().all(|dep| dep.dependency_flags.is_empty()));
            }
        }
    }

    #[test]
    fn storage_image_readers_retain_subpasses_and_global_dependency_chain() {
        for reader_count in [2, 3] {
            for reverse in [false, true] {
                for disjoint in [false, true] {
                    let mut readers = [
                        (
                            AccessType::AnyShaderReadOther,
                            vk::PipelineStageFlags::ALL_GRAPHICS,
                        ),
                        (
                            AccessType::VertexShaderReadOther,
                            vk::PipelineStageFlags::VERTEX_SHADER,
                        ),
                        (
                            AccessType::FragmentShaderReadOther,
                            vk::PipelineStageFlags::FRAGMENT_SHADER,
                        ),
                    ][..reader_count]
                        .to_vec();
                    if reverse {
                        readers.reverse();
                    }
                    let mut pass = SubpassFixture::subpass_command(
                        readers
                            .iter()
                            .enumerate()
                            .map(|(index, &(access, _))| {
                                assert_eq!(
                                    super::access_type_to_layout(access),
                                    Some(vk::ImageLayout::GENERAL)
                                );
                                let mut exec = SubpassFixture::color_attachment_exec(LoadOp::Load);
                                let layer = if disjoint { index as u32 } else { 0 };
                                exec.accesses.push(
                                    0,
                                    SubresourceAccess {
                                        access,
                                        subresource: SubresourceRange::Image(
                                            color_subresource_range(layer..layer + 1, 0..1),
                                        ),
                                    },
                                );
                                exec
                            })
                            .collect(),
                    );
                    let expected_mapping = (0..reader_count as u32).collect::<Vec<_>>();
                    // Check pruning independently, even if production coalescing is broken.
                    let deps = Submission::build_subpass_dependencies(
                        &pass,
                        &[PipelineStageAccessFlags::default(); 2],
                        &expected_mapping,
                    );
                    for src in 0..reader_count - 1 {
                        let edge = deps
                            .iter()
                            .find(|edge| {
                                edge.src_subpass == src as u32 && edge.dst_subpass == src as u32 + 1
                            })
                            .expect("storage read/read execution chain was pruned");
                        assert!(edge.dependency_flags.is_empty());
                        assert!(edge.src_stage_mask.contains(readers[src].1), "{edge:?}");
                        assert!(edge.dst_stage_mask.contains(readers[src + 1].1), "{edge:?}");
                        assert!(edge.src_access_mask.contains(vk::AccessFlags::SHADER_READ));
                        assert!(edge.dst_access_mask.contains(vk::AccessFlags::SHADER_READ));
                    }
                    let (info, mapping) = SubpassFixture::plan_subpasses(&pass);
                    assert_eq!(
                        &*mapping, expected_mapping,
                        "storage readers must not coalesce: reverse={reverse}, disjoint={disjoint}"
                    );
                    assert_eq!(info.dependencies, deps);

                    // A writer must remain ordered after every reader, including the broad
                    // fragment-capable scope that a later vertex-only reader cannot replace.
                    let mut writer = SubpassFixture::color_attachment_exec(LoadOp::Load);
                    writer.accesses.push(
                        0,
                        SubresourceAccess {
                            access: AccessType::FragmentShaderWrite,
                            subresource: SubresourceRange::Image(color_subresource_range(
                                0..3,
                                0..1,
                            )),
                        },
                    );
                    pass.execs.push(writer);
                    let (info, mapping) = SubpassFixture::plan_subpasses(&pass);
                    assert_eq!(&*mapping, (0..=reader_count as u32).collect::<Vec<_>>());
                    for (src, &(_, stages)) in readers.iter().enumerate() {
                        let edge = info
                            .dependencies
                            .iter()
                            .find(|edge| {
                                edge.src_subpass == src as u32
                                    && edge.dst_subpass == reader_count as u32
                            })
                            .expect("storage reader/writer dependency was lost");
                        assert!(edge.dependency_flags.is_empty());
                        assert!(edge.src_stage_mask.contains(stages), "{edge:?}");
                        assert!(
                            edge.dst_stage_mask
                                .contains(vk::PipelineStageFlags::FRAGMENT_SHADER)
                        );
                        assert!(edge.src_access_mask.contains(vk::AccessFlags::SHADER_READ));
                        assert!(edge.dst_access_mask.contains(vk::AccessFlags::SHADER_WRITE));
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn submission_acquires_nonempty_acceleration_structure_set() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let acceleration_structure = Arc::new(AccelerationStructure::create(
            &device,
            AccelerationStructureInfo::blas(1),
        )?);
        let previous_accesses = acceleration_structure
            .swap_access(AccessType::AccelerationStructureBuildWrite)
            .collect::<Vec<_>>();
        assert_eq!(previous_accesses, [AccessType::Nothing]);

        let resource_set = AccelerationStructureSet::new([Arc::clone(&acceleration_structure)])?;
        let mut graph = Graph::new();
        let resource_set_node = graph.bind_resource(&resource_set);

        graph
            .begin_cmd()
            .resource_access(
                resource_set_node,
                AccelerationStructureAccessType::BuildRead,
            )
            .record_cmd(|_| {})
            .end_cmd();
        graph
            .begin_cmd()
            .resource_access(
                resource_set_node,
                AccelerationStructureAccessType::RayTracingRead,
            )
            .record_cmd(|_| {})
            .end_cmd();

        let mut fence = graph
            .finalize()
            .queue_submit(&mut HashPool::new(&device), 0, 0)?;
        fence.wait()?;

        let sync_info = acceleration_structure.sync_info();
        assert!(sync_info.stage_mask.contains(
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                | vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR
        ));
        assert_eq!(
            sync_info.access_mask,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
        );

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn submission_acquires_nonempty_image_set() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let image = Arc::new(Image::create(
            &device,
            ImageInfo::image_2d(
                1,
                1,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED,
            ),
        )?);
        let resource_set = ImageSet::new([image])?;
        let mut graph = Graph::new();
        let resource_set_node = graph.bind_resource(&resource_set);

        graph
            .begin_cmd()
            .resource_access(resource_set_node, ImageAccessType::SampledRead)
            .record_cmd(|_| {})
            .end_cmd();

        let mut fence = graph
            .finalize()
            .queue_submit(&mut HashPool::new(&device), 0, 0)?;
        fence.wait()?;
        assert_eq!(resource_set.queue(), Some((0, 0)));

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan validation layers; inspect validation output"]
    fn submission_external_subpass_dependency_validation_repro() -> Result<(), DriverError> {
        TestDevice::init_validation_test_logging();

        let device = TestDevice::new_debug()?;
        let mut pool = HashPool::new(&device);
        let pipeline = SubpassFixture::test_triangle_pipeline(&device)?;
        let mut graph = Graph::new();
        let image = graph.bind_resource(Image::create(
            &device,
            ImageInfo::image_2d(
                4,
                4,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
        )?);

        // Seed external_access_history with a transfer write so the later render pass relies on
        // the synthesized EXTERNAL -> first subpass dependency
        graph.clear_color_image(image, [0.0, 0.0, 0.0, 1.0]);
        graph
            .begin_cmd()
            .debug_name("validation repro render pass")
            .bind_pipeline(&pipeline)
            .color_attachment_image(0, image, LoadOp::Load, StoreOp::Store)
            .record_cmd(|cmd| {
                cmd.draw(3, 1, 0, 0);
            });

        let submission = graph.finalize();
        let mut cmd_buf = pool.resource(CommandBufferInfo::new(0))?;

        cmd_buf.begin(
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let recorded = submission.record(&mut pool, &mut cmd_buf, RecordSelection::All)?;
        recorded.cmd_buf.end()?;

        let mut fence = Fence::create(&device, false)?;
        let mut recorded = recorded.finish()?;

        recorded.queue_submit(&mut fence, 0, QueueSubmitInfo::QUEUE_SUBMIT)?;
        fence.wait()?;

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn submission_milestone_failed_submit_and_unsubmitted_drop() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let mut pool = HashPool::new(&device);
        for reject_submit in [false, true] {
            let mut graph = Graph::new();
            let buffer = graph.bind_resource(Buffer::create(
                &device,
                BufferInfo::device_mem(4, vk::BufferUsageFlags::TRANSFER_DST),
            )?);
            let mut cmd = graph.begin_cmd();
            let execution = cmd.track_execution();
            cmd.fill_buffer(buffer, 0..4, 7).end_cmd();
            let mut cmd_buf = pool.resource(CommandBufferInfo::new(0))?;
            cmd_buf.begin(&vk::CommandBufferBeginInfo::default())?;
            let recording =
                graph
                    .finalize()
                    .record(&mut pool, &mut cmd_buf, RecordSelection::All)?;
            recording.cmd_buf.end()?;
            let mut recorded = recording.finish()?;
            assert_eq!(execution.has_submitted(), Ok(false));
            if reject_submit {
                let mut fence = Fence::create(&device, false)?;
                let waits = [super::SemaphoreSubmitInfo {
                    semaphore: vk::Semaphore::null(),
                    stage_mask: vk::PipelineStageFlags2::COPY,
                    value: 0,
                }];
                assert!(
                    recorded
                        .queue_submit(
                            &mut fence,
                            0,
                            QueueSubmitInfo::QueueSubmit {
                                waits: &waits,
                                signals: &[]
                            }
                        )
                        .is_err()
                );
                assert!(!fence.is_queued());
                assert_eq!(execution.has_submitted(), Ok(false));
                assert_eq!(execution.has_executed(), Ok(false));
                for submit in [
                    QueueSubmitInfo::QUEUE_SUBMIT,
                    QueueSubmitInfo::QUEUE_SUBMIT2,
                ] {
                    FAIL_QUEUE_SUBMIT.set(true);
                    assert!(recorded.queue_submit(&mut fence, 0, submit).is_err());
                    assert!(
                        !FAIL_QUEUE_SUBMIT.replace(false),
                        "submit did not reach Vulkan boundary"
                    );
                    assert!(!fence.is_queued());
                    assert_eq!(execution.has_submitted(), Ok(false));
                    assert_eq!(execution.has_executed(), Ok(false));
                }
            }
            drop(recorded);
            assert_eq!(
                execution.has_submitted(),
                Err(crate::CommandExecutionAbandoned)
            );
            assert_eq!(
                execution.has_executed(),
                Err(crate::CommandExecutionAbandoned)
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device with timeline semaphores and synchronization2"]
    fn submission_milestone_partial_recording_before_fence() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let mut pool = HashPool::new(&device);
        let mut graph = Graph::new();
        let mut nodes = Vec::new();
        let mut executions = Vec::new();
        for value in [7, 9] {
            let buffer = graph.bind_resource(Buffer::create(
                &device,
                BufferInfo::device_mem(4, vk::BufferUsageFlags::TRANSFER_DST),
            )?);
            nodes.push(buffer);
            let mut cmd = graph.begin_cmd();
            executions.push(cmd.track_execution());
            cmd.fill_buffer(buffer, 0..4, value).end_cmd();
        }
        let mut cmd_buf = pool.resource(CommandBufferInfo::new(0))?;
        cmd_buf.begin(&vk::CommandBufferBeginInfo::default())?;
        let recording = graph.finalize().record(&mut pool, &mut cmd_buf, nodes[0])?;
        recording.cmd_buf.end()?;
        let mut recorded = recording.finish()?;
        let mut fence = Fence::create(&device, false)?;
        let mut timeline =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let semaphore = unsafe {
            device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut timeline),
                None,
            )
        }
        .unwrap();
        let waits = [super::SemaphoreSubmit2Info {
            semaphore,
            value: 1,
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            device_index: 0,
        }];
        // Always release the timeline before dropping a queued fence, including assertion unwind.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            recorded
                .queue_submit(
                    &mut fence,
                    0,
                    QueueSubmitInfo::QueueSubmit2 {
                        waits: &waits,
                        signals: &[],
                    },
                )
                .unwrap();
            assert_eq!(executions[0].has_submitted(), Ok(true));
            assert_eq!(executions[0].has_executed(), Ok(false));
            assert_eq!(executions[1].has_submitted(), Ok(false));
            assert!(!fence.status().unwrap());
            assert_eq!(executions[0].has_executed(), Ok(false));
        }));
        unsafe {
            device.signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(semaphore)
                    .value(1),
            )
        }
        .unwrap();
        if fence.is_queued() {
            fence.wait()?;
        }
        unsafe {
            device.destroy_semaphore(semaphore, None);
        }
        if let Err(error) = result {
            std::panic::resume_unwind(error);
        }
        assert_eq!(executions[0].has_executed(), Ok(true));
        drop(recorded);
        assert_eq!(executions[0].has_submitted(), Ok(true));
        assert_eq!(
            executions[1].has_submitted(),
            Err(crate::CommandExecutionAbandoned)
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn submission_record_all_consumes_single_pass_graph() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let mut pool = HashPool::new(&device);
        let mut graph = Graph::new();
        let buffer = graph.bind_resource(Buffer::create(
            &device,
            BufferInfo::device_mem(16, vk::BufferUsageFlags::TRANSFER_DST),
        )?);

        graph.fill_buffer(buffer, 0..16, 0xdead_beef);

        let submission = graph.finalize();
        let mut cmd_buf = pool.resource(CommandBufferInfo::new(0))?;

        cmd_buf.begin(
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let recorded = submission.record(&mut pool, &mut cmd_buf, RecordSelection::All)?;

        assert!(recorded.is_empty());

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn submission_record_can_be_reused() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let mut pool = HashPool::new(&device);
        let mut graph = Graph::new();
        let buffer = graph.bind_resource(Buffer::create(
            &device,
            BufferInfo::device_mem(16, vk::BufferUsageFlags::TRANSFER_DST),
        )?);

        graph.fill_buffer(buffer, 0..16, 0xdead_beef);

        let submission = graph.finalize();
        let mut cmd_buf = pool.resource(CommandBufferInfo::new(0))?;
        let mut fence = Fence::create(&device, false)?;

        cmd_buf.begin(
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE),
        )?;

        let recorded = submission.record(&mut pool, &mut cmd_buf, RecordSelection::All)?;
        recorded.cmd_buf.end()?;
        let mut replay = recorded.finish()?;
        replay.queue_submit(&mut fence, 0, QueueSubmitInfo::QUEUE_SUBMIT)?;

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn submission_record_nodes_consumes_requested_outputs() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let mut pool = HashPool::new(&device);
        let mut graph = Graph::new();
        let lhs = graph.bind_resource(Buffer::create(
            &device,
            BufferInfo::device_mem(16, vk::BufferUsageFlags::TRANSFER_DST),
        )?);
        let rhs = graph.bind_resource(Buffer::create(
            &device,
            BufferInfo::device_mem(16, vk::BufferUsageFlags::TRANSFER_DST),
        )?);

        graph.fill_buffer(lhs, 0..16, 1);
        graph.fill_buffer(rhs, 0..16, 2);

        let nodes = [AnyNode::from(lhs), AnyNode::from(rhs)];
        let submission = graph.finalize();
        let mut cmd_buf = pool.resource(CommandBufferInfo::new(0))?;

        cmd_buf.begin(
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let recorded =
            submission.record(&mut pool, &mut cmd_buf, RecordSelection::nodes(&nodes))?;

        assert!(recorded.is_empty());

        Ok(())
    }

    #[test]
    fn subpass_dependency_attachment_bitset_tracks_all_roles_and_resets() {
        let mut input = SubpassFixture::color_attachment_exec(LoadOp::DontCare);
        let state = input.attachments.color[0].as_mut().unwrap();
        state.is_attachment = false;
        state.is_input = true;

        let mut color = SubpassFixture::color_attachment_exec(LoadOp::DontCare);
        let state = color.attachments.color[0].as_mut().unwrap();
        state.attachment.target = 64;
        state.resolve = Some(ColorResolve {
            attachment: Attachment {
                target: 65,
                ..state.attachment
            },
            src_attachment_idx: 0,
        });
        let mut depth = SubpassFixture::depth_attachment_exec(LoadOp::DontCare, StoreOp::DontCare);
        let state = depth.attachments.depth_stencil.as_mut().unwrap();
        state.attachment.target = 129;
        state.resolve = Some(DepthStencilResolve {
            attachment: Attachment {
                target: 130,
                ..state.attachment
            },
            dst_attachment_idx: 0,
            depth_mode: None,
            stencil_mode: None,
        });
        let mut metadata = SubpassFixture::color_attachment_exec(LoadOp::DontCare);
        let state = metadata.attachments.color[0].as_mut().unwrap();
        state.attachment.target = 255;
        state.is_attachment = false;
        state.store = StoreOp::DontCare;
        let pass = SubpassFixture::subpass_command(vec![input, color, depth, metadata]);
        let empty = SubpassFixture::subpass_command(vec![Execution::default(); 2]);
        for _ in 0..2 {
            Submission::build_subpass_dependencies(
                &pass,
                &[PipelineStageAccessFlags::default(); 256],
                &[0, 1, 2, 3],
            );
            super::SUBPASS_DEPENDENCY.with_borrow(|scratch| {
                assert_eq!(
                    scratch.attachment_nodes.ones().collect::<Vec<_>>(),
                    [1, 64, 65, 129, 130, 255]
                );
                assert!(!scratch.groups.iter().any(|&(node, _, _)| node == 255));
            });
            assert!(
                Submission::build_subpass_dependencies(
                    &empty,
                    &[PipelineStageAccessFlags::default(); 1],
                    &[0, 1],
                )
                .is_empty()
            );
            super::SUBPASS_DEPENDENCY.with_borrow(|scratch| {
                assert!(scratch.attachment_nodes.is_clear());
                assert!(scratch.pass_classes.iter().all(Option::is_none));
                assert!(scratch.history.iter().all(Vec::is_empty));
            });
        }
    }

    #[test]
    fn subpass_dependency_indexed_accumulator_matches_map_oracle_on_reuse() {
        let mut scratch = super::SubpassDependencyScratch::default();
        for (epoch, count) in [97, 2, 137].into_iter().enumerate() {
            scratch.reset(0, count);
            let mut contributions = std::collections::BTreeMap::<_, Vec<_>>::new();
            let mut actual = Vec::new();
            for dst in [count - 2, count - 1] {
                for round in 0..3 {
                    for src in (0..dst).rev().chain([vk::SUBPASS_EXTERNAL as usize]) {
                        let previous = PipelineStageAccessFlags::new(
                            [
                                AccessType::VertexShaderWrite,
                                AccessType::ColorAttachmentWrite,
                            ][(epoch + dst + round) % 2],
                        );
                        let current = PipelineStageAccessFlags::new(
                            [AccessType::FragmentShaderReadOther, AccessType::IndexBuffer]
                                [(epoch + round) % 2],
                        );
                        let flags = if (src + round) % 3 == 0 && src % 3 != 0 {
                            vk::DependencyFlags::empty()
                        } else {
                            vk::DependencyFlags::BY_REGION
                        };
                        contributions
                            .entry((src as u32, dst as u32))
                            .or_default()
                            .push((previous, current, flags));
                        scratch.record_dependency(src, dst, previous, current, flags);
                    }
                }
                scratch.flush_dependencies(&mut actual);
                let len = actual.len();
                scratch.flush_dependencies(&mut actual);
                assert_eq!(actual.len(), len, "flush must consume touched sources");
            }
            let expected = contributions
                .into_iter()
                .map(|((src, dst), scopes)| {
                    let mut dep = SubpassDependency::new(src, dst);
                    dep.dependency_flags = scopes[0].2;
                    for (previous, current, flags) in scopes {
                        dep.src_stage_mask |= previous.stage_flags;
                        dep.src_access_mask |= previous.access_flags;
                        dep.dst_stage_mask |= current.stage_flags;
                        dep.dst_access_mask |= current.access_flags;
                        dep.dependency_flags &= flags;
                    }
                    dep
                })
                .collect::<Vec<_>>();
            actual.sort_unstable_by_key(|dep| (dep.src_subpass, dep.dst_subpass));
            assert_eq!(actual, expected, "reuse epoch {epoch}");
            // Reset must also discard a partially accumulated destination.
            scratch.record_dependency(
                vk::SUBPASS_EXTERNAL as usize,
                0,
                PipelineStageAccessFlags::new(AccessType::TransferWrite),
                PipelineStageAccessFlags::new(AccessType::TransferRead),
                vk::DependencyFlags::empty(),
            );
        }
    }

    #[test]
    fn subpass_dependency_matches_all_graphics_destination_stage() {
        let dependencies = SubpassFixture::subpass_dependencies_for_accesses(
            AccessType::FragmentShaderWrite,
            AccessType::AnyShaderReadOther,
        );
        let dep = dependencies
            .iter()
            .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 1)
            .expect("missing subpass dependency for ALL_GRAPHICS destination stage");

        assert!(
            dep.src_stage_mask
                .contains(vk::PipelineStageFlags::FRAGMENT_SHADER),
            "source stage should include fragment shader"
        );
        assert!(
            dep.src_access_mask.contains(vk::AccessFlags::SHADER_WRITE),
            "source access should include shader writes"
        );
        assert!(
            dep.dst_stage_mask
                .contains(vk::PipelineStageFlags::ALL_GRAPHICS),
            "destination stage should include ALL_GRAPHICS"
        );
        assert!(
            dep.dst_access_mask.contains(vk::AccessFlags::SHADER_READ),
            "destination access should include shader reads"
        );
    }

    #[test]
    fn subpass_dependency_matches_all_graphics_source_stage() {
        let dependencies = SubpassFixture::subpass_dependencies_for_accesses(
            AccessType::AnyShaderWrite,
            AccessType::FragmentShaderReadOther,
        );
        let dep = dependencies
            .iter()
            .find(|dep| dep.src_subpass == 0 && dep.dst_subpass == 1)
            .expect("missing subpass dependency for ALL_GRAPHICS source stage");

        assert!(
            dep.src_stage_mask
                .contains(vk::PipelineStageFlags::ALL_GRAPHICS),
            "source stage should include ALL_GRAPHICS"
        );
        assert!(
            dep.src_access_mask.contains(vk::AccessFlags::SHADER_WRITE),
            "source access should include shader writes"
        );
        assert!(
            dep.dst_stage_mask
                .contains(vk::PipelineStageFlags::FRAGMENT_SHADER),
            "destination stage should include fragment shader"
        );
        assert!(
            dep.dst_access_mask.contains(vk::AccessFlags::SHADER_READ),
            "destination access should include shader reads"
        );
    }

    #[test]
    fn subpass_dependency_sparse_nodes_and_empty_repeated_groups() {
        let exec = |accesses: &[_]| command_with_accesses(accesses).execs.remove(0);
        let pass = SubpassFixture::subpass_command(vec![
            Execution::default(),
            exec(&[(4095, AccessType::FragmentShaderWrite)]),
            exec(&[(17, AccessType::VertexShaderWrite)]),
            Execution::default(),
            exec(&[(4095, AccessType::FragmentShaderReadOther)]),
            exec(&[
                (4095, AccessType::VertexShaderReadUniformBuffer),
                (17, AccessType::IndexBuffer),
            ]),
            Execution::default(),
        ]);
        let mut history = vec![PipelineStageAccessFlags::default(); 4096];
        history[4095] = PipelineStageAccessFlags::new(AccessType::TransferWrite);
        let deps = Submission::build_subpass_dependencies(&pass, &history, &[0, 1, 1, 2, 3, 3, 4]);
        assert_eq!(
            deps.iter()
                .map(|dep| (dep.src_subpass, dep.dst_subpass))
                .collect::<Vec<_>>(),
            [(1, 3), (vk::SUBPASS_EXTERNAL, 1), (vk::SUBPASS_EXTERNAL, 3)]
        );
        let edge = &deps[0];
        assert_eq!(
            edge.src_stage_mask,
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER
        );
        assert_eq!(edge.src_access_mask, vk::AccessFlags::SHADER_WRITE);
        assert_eq!(
            edge.dst_stage_mask,
            vk::PipelineStageFlags::VERTEX_SHADER
                | vk::PipelineStageFlags::FRAGMENT_SHADER
                | vk::PipelineStageFlags::VERTEX_INPUT
        );
        assert_eq!(
            edge.dst_access_mask,
            vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::UNIFORM_READ
                | vk::AccessFlags::INDEX_READ
        );
        for external in &deps[1..] {
            assert_eq!(
                external.src_stage_mask,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::TRANSFER
            );
            assert_eq!(
                external.src_access_mask,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
            );
        }
        assert!(deps.iter().all(|dep| dep.dependency_flags.is_empty()));
    }

    #[test]
    fn subpass_dependency_tls_recovers_after_unwind() {
        let pass = SubpassFixture::subpass_command(vec![
            command_with_accesses(&[(0, AccessType::FragmentShaderWrite)])
                .execs
                .remove(0),
            command_with_accesses(&[(0, AccessType::VertexShaderReadUniformBuffer)])
                .execs
                .remove(0),
        ]);
        let history = [PipelineStageAccessFlags::default()];
        let expected = Submission::build_subpass_dependencies(&pass, &history, &[0, 1]);
        assert!(
            std::panic::catch_unwind(|| {
                super::SUBPASS_DEPENDENCY.with_borrow_mut(|scratch| {
                    scratch.record_dependency(
                        vk::SUBPASS_EXTERNAL as usize,
                        1,
                        PipelineStageAccessFlags::new(AccessType::HostWrite),
                        PipelineStageAccessFlags::new(AccessType::ColorAttachmentWrite),
                        vk::DependencyFlags::BY_REGION,
                    );
                    panic!("interrupt an unflushed plan");
                });
            })
            .is_err()
        );
        assert_eq!(
            Submission::build_subpass_dependencies(&pass, &history, &[0, 1]),
            expected
        );
    }

    #[test]
    fn subpass_dependency_tls_retains_history_capacity_without_stale_scopes() {
        let mut attachment = SubpassFixture::color_attachment_exec(LoadOp::Load);
        let mut high = *attachment.attachments.color[0].as_ref().unwrap();
        high.attachment.target = 4095;
        attachment.attachments.color.push(Some(high));
        let warm = SubpassFixture::subpass_command(vec![attachment; 70]);
        let mapping = (0..70).collect::<Vec<_>>();
        let history = vec![PipelineStageAccessFlags::new(AccessType::TransferWrite); 4096];
        let expected = Submission::build_subpass_dependencies(&warm, &history, &mapping);
        let capacities = super::SUBPASS_DEPENDENCY.with_borrow(|scratch| {
            [1, 4095].map(|node| {
                assert_eq!(scratch.history[node].len(), 70);
                scratch.history[node].capacity()
            })
        });
        assert_eq!(
            Submission::build_subpass_dependencies(&warm, &history, &mapping),
            expected
        );
        for node_count in [1, 8192] {
            let nodes = if node_count == 1 {
                vec![0]
            } else {
                vec![1, 4095, 8191]
            };
            let pass = SubpassFixture::subpass_command(
                [
                    AccessType::VertexShaderReadUniformBuffer,
                    AccessType::FragmentShaderReadOther,
                ]
                .into_iter()
                .map(|access| {
                    command_with_accesses(
                        &nodes.iter().map(|&node| (node, access)).collect::<Vec<_>>(),
                    )
                    .execs
                    .remove(0)
                })
                .collect(),
            );
            let deps = Submission::build_subpass_dependencies(
                &pass,
                &vec![PipelineStageAccessFlags::default(); node_count],
                &[0, 1],
            );
            let expected = [
                (
                    vk::PipelineStageFlags::VERTEX_SHADER,
                    vk::AccessFlags::UNIFORM_READ,
                ),
                (
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::AccessFlags::SHADER_READ,
                ),
            ]
            .into_iter()
            .enumerate()
            .map(|(dst, (stage, access))| {
                let mut dep = SubpassDependency::new(vk::SUBPASS_EXTERNAL, dst as u32);
                dep.src_stage_mask = vk::PipelineStageFlags::ALL_COMMANDS;
                dep.src_access_mask = vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
                dep.dst_stage_mask = stage;
                dep.dst_access_mask = access;
                dep
            })
            .collect::<Vec<_>>();
            assert_eq!(deps, expected, "node count {node_count}");
            super::SUBPASS_DEPENDENCY.with_borrow(|scratch| {
                assert!(scratch.history.iter().all(Vec::is_empty));
                assert!(scratch.attachment_nodes.is_clear());
                for (node, capacity) in [1, 4095].into_iter().zip(capacities) {
                    assert_eq!(scratch.history[node].capacity(), capacity);
                }
            });
        }
        assert_eq!(
            Submission::build_subpass_dependencies(&warm, &history, &mapping),
            expected
        );
        super::SUBPASS_DEPENDENCY.with_borrow(|scratch| {
            for (node, capacity) in [1, 4095].into_iter().zip(capacities) {
                assert_eq!(scratch.history[node].len(), 70);
                assert_eq!(scratch.history[node].capacity(), capacity);
            }
        });
    }

    #[test]
    fn subpass_fixture_oracle_distinguishes_each_binding() {
        let n = 1024;
        for tile in 0..n {
            let texture = tile * 13 % n;
            let expected = SubpassFixture::expected(tile, texture, 17);
            assert_ne!(
                expected,
                SubpassFixture::expected((tile + 1) % n, texture, 17)
            );
            assert_ne!(
                expected,
                SubpassFixture::expected(tile, (texture + 1) % n, 17)
            );
            assert_ne!(expected, SubpassFixture::expected(tile, texture, 18));
        }
    }

    #[test]
    #[ignore = "manual release benchmark; requires Vulkan device, validation disabled"]
    #[allow(clippy::assertions_on_constants)]
    fn subpass_gpu_benchmark() -> Result<(), DriverError> {
        assert!(!cfg!(debug_assertions), "run this benchmark with --release");
        let device = TestDevice::new()?;
        eprintln!(
            "subpass benchmark device: {}",
            device.physical.properties_v1_0.device_name
        );
        eprintln!(
            "CPU milliseconds; input/output allocation, upload, submit/wait and pixel checks excluded; readback command recording included; cold = fresh pool per mode, not a cleared driver cache"
        );
        eprintln!("N,prepared,physical_subpasses,draft_ms,prepare_ms,sample,build_ms,record_ms");
        for n in [1, 64, 256, 1024] {
            let mut pool = HashPool::new(&device);
            let fixture = SubpassFixture::new(&device, &mut pool, n)?;
            for prepared in [false, true] {
                let mut pool = HashPool::new(&device);
                let start = Instant::now();
                let draft = fixture.draft();
                let draft_ms = start.elapsed().as_secs_f64() * 1000.0;
                let start = Instant::now();
                let stream = if prepared {
                    draft.prepare(&mut pool)?
                } else {
                    draft.into_stream()
                };
                let prepare_ms = start.elapsed().as_secs_f64() * 1000.0;
                let prepared_count = if prepared {
                    SubpassFixture::subpass_counts(&stream.inner.submission.lock().unwrap())
                        .into_iter()
                        .sum()
                } else {
                    0
                };
                let (target, output) = fixture.output(&device)?;
                let budget = Instant::now();
                let mut warm_records = Vec::new();
                for sample in 0..120 {
                    let start = Instant::now();
                    let mut graph = Graph::new();
                    fixture.invoke(&mut graph, &stream, &target, sample, sample as u32 + 17);
                    let image = graph.bind_resource(&target);
                    SubpassFixture::readback(&mut graph, image.into(), &output);
                    let submission = graph.finalize();
                    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
                    let mut cmd_buf = pool.resource(CommandBufferInfo::new(0))?;
                    cmd_buf.begin(
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )?;
                    let start = Instant::now();
                    let recording =
                        submission.record(&mut pool, &mut cmd_buf, RecordSelection::All)?;
                    let record_ms = start.elapsed().as_secs_f64() * 1000.0;
                    if sample >= 20 {
                        warm_records.push(record_ms);
                    }
                    let count = if prepared {
                        prepared_count
                    } else {
                        recording
                            .submission
                            .submit_retained
                            .iter()
                            .filter_map(|command| command._resources.render_pass.as_ref())
                            .map(|render_pass| render_pass.info.subpasses.len())
                            .sum::<usize>()
                    };
                    assert_eq!(count, 1, "eligible benchmark draws must share one subpass");
                    recording.cmd_buf.end()?;
                    let mut recorded = recording.finish()?;
                    let mut fence = Fence::create(&device, false)?;
                    recorded.queue_submit(&mut fence, 0, QueueSubmitInfo::QUEUE_SUBMIT)?;
                    fence.wait()?;
                    fixture.check(&output, sample, sample as u32 + 17);
                    eprintln!(
                        "{n},{prepared},{count},{draft_ms:.6},{prepare_ms:.6},{sample},{build_ms:.6},{record_ms:.6}"
                    );
                    // Finish warm-up and collect at least one sample before capping slow drivers.
                    if sample >= 20 && budget.elapsed() > Duration::from_secs(15) {
                        break;
                    }
                }
                warm_records.sort_unstable_by(f64::total_cmp);
                eprintln!(
                    "summary: N={n} prepared={prepared} warm_record_median_ms={:.3} samples={}",
                    warm_records[warm_records.len() / 2],
                    warm_records.len(),
                );
            }
        }
        Ok(())
    }

    #[test]
    fn subpass_incoming_buffer_barriers_cover_partial_ownership_and_stages() {
        let buffer = vk::Buffer::from_raw(1);
        let transfers = [BufferQueueOwnershipTransfer {
            range: (16..32).into(),
            src_queue_family_index: 1,
            dst_queue_family_index: 0,
        }];
        for producer in [
            AccessType::TransferWrite,
            AccessType::ComputeShaderWrite,
            AccessType::General,
        ] {
            for consumer in [
                AccessType::VertexShaderReadUniformBuffer,
                AccessType::FragmentShaderReadUniformBuffer,
            ] {
                let barriers = super::BufferQueueOwnershipTransfer::barriers(
                    buffer,
                    &producer,
                    &consumer,
                    (0..48).into(),
                    &transfers,
                )
                .collect::<Vec<_>>();
                assert_eq!(barriers.len(), 3);
                for (index, barrier) in barriers.iter().enumerate() {
                    assert_eq!(barrier.offset, index * 16);
                    assert_eq!(barrier.size, 16);
                    assert_eq!(barrier.previous_accesses, &[producer]);
                    assert_eq!(barrier.next_accesses, &[consumer]);
                    // These canonical scopes, not vk-sync's uniform mapping, are used by
                    // record_image_layout_transitions when lowering pre-pass barriers.
                    let (stages, accesses) =
                        crate::driver::pipeline_stage_access_flags(barrier.next_accesses[0]);
                    assert_eq!(accesses, vk::AccessFlags::UNIFORM_READ);
                    assert_eq!(
                        stages,
                        if consumer == AccessType::VertexShaderReadUniformBuffer {
                            vk::PipelineStageFlags::VERTEX_SHADER
                        } else {
                            vk::PipelineStageFlags::FRAGMENT_SHADER
                        }
                    );
                    assert_eq!(
                        (
                            barrier.src_queue_family_index,
                            barrier.dst_queue_family_index
                        ),
                        if index == 1 {
                            (1, 0)
                        } else {
                            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
                        }
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "manual release planning benchmark; no Vulkan device required"]
    #[allow(clippy::assertions_on_constants)]
    fn subpass_planning_benchmark() {
        assert!(!cfg!(debug_assertions), "run this benchmark with --release");
        for n in [1usize, 64, 256, 1024] {
            let pass = SubpassFixture::subpass_command(
                (0..n)
                    .map(|idx| {
                        let mut exec = SubpassFixture::color_attachment_exec(if idx == 0 {
                            LoadOp::Clear([0.0; 4])
                        } else {
                            LoadOp::Load
                        });
                        exec.accesses.push(
                            2 * idx + 2,
                            SubresourceAccess {
                                access: AccessType::FragmentShaderReadUniformBuffer,
                                subresource: SubresourceRange::Buffer((0..16).into()),
                            },
                        );
                        exec.accesses.push(
                            2 * idx + 3,
                            SubresourceAccess {
                                access:
                                    AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                                subresource: SubresourceRange::Image(color_subresource_range(
                                    0..1,
                                    0..1,
                                )),
                            },
                        );
                        exec.accesses.freeze();
                        exec
                    })
                    .collect(),
            );
            let external = vec![PipelineStageAccessFlags::default(); 2 * n + 2];
            let graphics = vec![
                GraphicsExecutionInfo {
                    input_attachments: &[],
                    sample_count: SampleCount::Type1,
                };
                n
            ];
            let batch = (4096 / n).max(1);
            let mut samples = Vec::new();
            for sample in 0..120 {
                let start = Instant::now();
                for _ in 0..batch {
                    let (info, mapping) = std::hint::black_box(Submission::build_render_pass_info(
                        std::hint::black_box(&pass),
                        std::hint::black_box(&external),
                        std::hint::black_box(&graphics),
                    ));
                    assert_eq!(info.subpasses.len(), 1);
                    assert_eq!(mapping.len(), n);
                    std::hint::black_box((info, mapping));
                }
                if sample >= 20 {
                    samples.push(start.elapsed().as_secs_f64() * 1_000_000.0 / batch as f64);
                }
            }
            samples.sort_unstable_by(f64::total_cmp);
            eprintln!(
                "planning: N={n} median_us={:.6} samples={}",
                samples[samples.len() / 2],
                samples.len()
            );
        }
    }

    #[test]
    fn subpass_stage_mask_clamps_non_graphics_stages() {
        assert_eq!(
            Submission::subpass_stage_mask(vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR),
            vk::PipelineStageFlags::ALL_GRAPHICS,
        );
        assert_eq!(
            Submission::subpass_stage_mask(
                vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            ),
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        );
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn supplied_descriptor_set_reuses_compatible_pipeline_layout() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let spirv = glsl!(
            r#"
            #version 460 core
            #pragma shader_stage(compute)

            layout(local_size_x = 1) in;
            layout(set = 0, binding = 0) buffer DataA {
                uint value;
            } data_a;
            layout(set = 1, binding = 0) buffer DataB {
                uint value;
            } data_b;

            void main() {
                data_a.value = 1;
                data_b.value = 2;
            }
            "#
        );
        let pipeline =
            ComputePipeline::create(&device, ComputePipelineInfo::default(), spirv.as_slice())?;
        let compatible_pipeline =
            ComputePipeline::create(&device, ComputePipelineInfo::default(), spirv.as_slice())?;
        let buffer_a = Arc::new(Buffer::create(
            &device,
            BufferInfo::device_mem(4, vk::BufferUsageFlags::STORAGE_BUFFER),
        )?);
        let buffer_b = Arc::new(Buffer::create(
            &device,
            BufferInfo::device_mem(4, vk::BufferUsageFlags::STORAGE_BUFFER),
        )?);
        let descriptor_set = DescriptorSet::alloc_and_update(
            &pipeline,
            DescriptorSetInfo::builder().set(0),
            DescriptorSetUpdateInfo::buffer(0, &buffer_a),
        )?;
        let descriptor_set_a = DescriptorSet::alloc_and_update(
            &pipeline,
            DescriptorSetInfo::builder().set(0),
            DescriptorSetUpdateInfo::copy(&descriptor_set, 0, 0),
        )?;
        let descriptor_set_b = DescriptorSet::alloc_and_update(
            &pipeline,
            DescriptorSetInfo::builder().set(1),
            DescriptorSetUpdateInfo::buffer(0, &buffer_b),
        )?;
        drop(descriptor_set);
        drop(pipeline);

        let mut graph = Graph::new();
        let buffer_a_node = graph.bind_resource(&buffer_a);
        let buffer_b_node = graph.bind_resource(&buffer_b);
        graph
            .begin_cmd()
            .bind_pipeline(&compatible_pipeline)
            .bind_descriptor_set(&descriptor_set_a)
            .resource_access(buffer_a_node, AccessType::ComputeShaderWrite)
            .shader_resource_access((1, 0), buffer_b_node, AccessType::ComputeShaderWrite)
            .record_cmd(|cmd| {
                cmd.dispatch(1, 1, 1);
            })
            .end_cmd();
        graph
            .begin_cmd()
            .bind_pipeline(&compatible_pipeline)
            .bind_descriptor_set(&descriptor_set_a)
            .bind_descriptor_set(&descriptor_set_b)
            .resource_access(buffer_a_node, AccessType::ComputeShaderWrite)
            .resource_access(buffer_b_node, AccessType::ComputeShaderWrite)
            .record_cmd(|cmd| {
                cmd.dispatch(1, 1, 1);
            })
            .end_cmd();

        let mut fence = graph
            .finalize()
            .queue_submit(&mut HashPool::new(&device), 0, 0)?;
        fence.wait()?;

        Ok(())
    }

    #[test]
    fn timestamp_duration_uses_valid_bits_and_wraparound() {
        assert_eq!(
            super::SubmittedTimestampQueries::timestamp_duration_since(1, 14, 4, 1.0),
            Duration::from_nanos(3),
        );
        assert_eq!(
            super::SubmittedTimestampQueries::timestamp_duration_since(20, 4, 64, 2.0),
            Duration::from_nanos(32),
        );
    }

    #[test]
    fn timestamp_queries_require_queue_family_that_can_reset_queries() {
        let mut queue_family = vk::QueueFamilyProperties {
            queue_flags: vk::QueueFlags::TRANSFER,
            timestamp_valid_bits: 64,
            ..Default::default()
        };

        assert!(!Submission::queue_family_supports_timestamp_queries(
            &queue_family
        ));

        queue_family.queue_flags = vk::QueueFlags::COMPUTE;
        assert!(Submission::queue_family_supports_timestamp_queries(
            &queue_family
        ));

        queue_family.queue_flags = vk::QueueFlags::GRAPHICS;
        assert!(Submission::queue_family_supports_timestamp_queries(
            &queue_family
        ));

        queue_family.timestamp_valid_bits = 0;
        assert!(!Submission::queue_family_supports_timestamp_queries(
            &queue_family
        ));
    }

    #[test]
    fn timestamp_query_pool_exposes_only_relative_results() {
        let mut graph = Graph::new();
        let start = graph.write_timestamp();
        let end = graph.write_timestamp();
        let pool = pending_timestamp_query_pool(start);

        assert!(!pool.has_results());

        pool.set(
            vec![
                Some(Duration::from_millis(5)),
                Some(Duration::from_millis(11)),
            ]
            .into_boxed_slice(),
        );

        let result = pool.duration(start).expect("missing timestamp result");
        assert_eq!(result, Duration::from_millis(5));
        assert_eq!(pool.duration(end), Some(Duration::from_millis(11)));
        assert!(pool.has_results());
    }

    #[test]
    fn timestamp_query_pool_returns_none_before_results_are_set() {
        let mut graph = Graph::new();
        let query = graph.write_timestamp();
        let pool = pending_timestamp_query_pool(query);

        assert_eq!(pool.duration(query), None);
        assert!(!pool.has_results());

        pool.complete_without_results();

        assert_eq!(pool.duration(query), None);
        assert!(pool.has_results());
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn track_pending_transfers_keeps_exclusive_owner_without_known_layout()
    -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let mut graph = Graph::new();
        let image = graph.bind_resource(Image::create(
            &device,
            ImageInfo::image_2d_array(1, 1, 2, vk::Format::R8_UINT, vk::ImageUsageFlags::SAMPLED),
        )?);
        let range_a = color_subresource_range(0..1, 0..1);
        let range_b = color_subresource_range(1..2, 0..1);
        let image_handle = graph.resource(image).handle;

        {
            let image_resource = graph.resource(image);
            image_resource.set_sharing_ranges(SharingMode::Exclusive(Some((1, 0))), &[range_a]);
            image_resource.set_sharing_ranges(SharingMode::Exclusive(Some((2, 0))), &[range_b]);
        }

        graph
            .begin_cmd()
            .debug_name("touch first layer only")
            .subresource_access(image, range_a, AccessType::TransferWrite)
            .record_cmd(|_| {})
            .end_cmd();

        let mut submission = graph.finalize();
        let mut ownership = RecordingOwnership::default();
        submission.track_pending_transfers(
            &Schedule {
                cmds: vec![0],
                ..Default::default()
            },
            3,
            &mut ownership,
            ResourceSetSynchronization::Enabled,
        );

        let (handle, transfers) = pending_transfer_for_node(
            submission
                .pending_image_transfer_nodes
                .as_ref()
                .expect("missing pending transfer nodes"),
            image.index(),
        )
        .expect("missing pending transfer for touched subresource");
        assert_eq!(handle, image_handle);
        assert_eq!(
            submission
                .pending_image_transfer_nodes
                .as_ref()
                .expect("missing pending transfer nodes")
                .indices,
            vec![image.index()]
        );
        let mut transfers = transfers.to_vec();
        sort_pending_image_transfers(&mut transfers);

        assert_eq!(transfers.len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            transfers[0].range,
            range_a
        ));
        assert_eq!(transfers[0].layouts.old, vk::ImageLayout::UNDEFINED);
        assert_eq!(
            transfers[0].layouts.new,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL
        );

        let ranges = &submission.exclusive_image_ranges[&image.index()];
        let mut ranges = ranges.clone();
        sort_image_subresource_ranges(&mut ranges);
        assert_eq!(ranges.len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            ranges[0], range_a
        ));

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn track_pending_transfers_only_collects_touched_buffer_ranges() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let mut graph = Graph::new();
        let buffer = graph.bind_resource(Buffer::create(
            &device,
            BufferInfo::device_mem(16, vk::BufferUsageFlags::TRANSFER_DST),
        )?);
        let range_a = BufferSubresourceRange { start: 0, end: 8 };
        let range_b = BufferSubresourceRange { start: 8, end: 16 };
        let buffer_handle = graph.resource(buffer).handle;

        {
            let buffer_resource = graph.resource(buffer);
            buffer_resource.set_sharing_ranges(SharingMode::Exclusive(Some((1, 0))), &[range_a]);
            buffer_resource.set_sharing_ranges(SharingMode::Exclusive(Some((2, 0))), &[range_b]);

            buffer_resource
                .swap_access(AccessType::TransferRead, range_a)
                .for_each(drop);
            buffer_resource
                .swap_access(AccessType::TransferRead, range_b)
                .for_each(drop);
        }

        graph
            .begin_cmd()
            .debug_name("touch first buffer range only")
            .subresource_access(buffer, range_a, AccessType::TransferWrite)
            .record_cmd(|_| {})
            .end_cmd();

        let mut submission = graph.finalize();
        let mut ownership = RecordingOwnership::default();
        submission.track_pending_transfers(
            &Schedule {
                cmds: vec![0],
                ..Default::default()
            },
            3,
            &mut ownership,
            ResourceSetSynchronization::Enabled,
        );

        let (handle, transfers) = pending_transfer_for_node(
            submission
                .pending_buffer_transfer_nodes
                .as_ref()
                .expect("missing pending transfer nodes"),
            buffer.index(),
        )
        .expect("missing pending transfer for touched buffer range");
        assert_eq!(handle, buffer_handle);
        assert_eq!(
            submission
                .pending_buffer_transfer_nodes
                .as_ref()
                .expect("missing pending transfer nodes")
                .indices,
            vec![buffer.index()]
        );
        let mut transfers = transfers.to_vec();
        sort_pending_buffer_transfers(&mut transfers);

        assert_eq!(transfers.len(), 1);
        assert!(pending_buffer_transfer_for_range(&transfers, range_a).is_some());
        assert!(pending_buffer_transfer_for_range(&transfers, range_b).is_none());

        let ranges = &submission.exclusive_buffer_ranges[&buffer.index()];
        let mut ranges = ranges.clone();
        ranges.sort_unstable_by_key(|range| (range.start, range.end));
        assert_eq!(ranges, vec![range_a]);

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn track_pending_transfers_only_collects_touched_subresources() -> Result<(), DriverError> {
        let device = TestDevice::new()?;
        let mut graph = Graph::new();
        let image = graph.bind_resource(Image::create(
            &device,
            ImageInfo::image_2d_array(1, 1, 2, vk::Format::R8_UINT, vk::ImageUsageFlags::SAMPLED),
        )?);
        let range_a = color_subresource_range(0..1, 0..1);
        let range_b = color_subresource_range(1..2, 0..1);
        let image_handle = graph.resource(image).handle;

        {
            let image_resource = graph.resource(image);
            image_resource.set_sharing_ranges(SharingMode::Exclusive(Some((1, 0))), &[range_a]);
            image_resource.set_sharing_ranges(SharingMode::Exclusive(Some((2, 0))), &[range_b]);

            image_resource
                .swap_access(AccessType::TransferRead, range_a)
                .for_each(drop);
            image_resource
                .swap_access(AccessType::TransferRead, range_b)
                .for_each(drop);
        }

        graph
            .begin_cmd()
            .debug_name("touch first layer only")
            .subresource_access(image, range_a, AccessType::TransferWrite)
            .record_cmd(|_| {})
            .end_cmd();

        let mut submission = graph.finalize();
        let mut ownership = RecordingOwnership::default();
        submission.track_pending_transfers(
            &Schedule {
                cmds: vec![0],
                ..Default::default()
            },
            3,
            &mut ownership,
            ResourceSetSynchronization::Enabled,
        );

        let (handle, transfers) = pending_transfer_for_node(
            submission
                .pending_image_transfer_nodes
                .as_ref()
                .expect("missing pending transfer nodes"),
            image.index(),
        )
        .expect("missing pending transfer for touched subresource");
        assert_eq!(handle, image_handle);
        assert_eq!(
            submission
                .pending_image_transfer_nodes
                .as_ref()
                .expect("missing pending transfer nodes")
                .indices,
            vec![image.index()]
        );
        let mut transfers = transfers.to_vec();
        sort_pending_image_transfers(&mut transfers);

        assert_eq!(transfers.len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            transfers[0].range,
            range_a
        ));
        let ranges = &submission.exclusive_image_ranges[&image.index()];
        let mut ranges = ranges.clone();
        sort_image_subresource_ranges(&mut ranges);
        assert_eq!(ranges.len(), 1);
        assert!(super::ImageOwnershipTransfer::ranges_equal(
            ranges[0], range_a
        ));

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and validation layers"]
    fn vulkan_depth_resolve_cross_graph_scopes() -> Result<(), DriverError> {
        use crate::driver::render_pass::ResolveMode;

        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        {
            let info = ImageInfo::image_2d(
                2,
                2,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            );
            if !device
                .physical
                .depth_stencil_resolve_properties
                .supported_depth_resolve_modes
                .contains(vk::ResolveModeFlags::SAMPLE_ZERO)
                || !device
                    .physical
                    .image_format_properties(
                        info.format,
                        info.image_type,
                        info.tiling,
                        info.usage,
                        info.flags,
                    )?
                    .is_some_and(|properties| {
                        properties
                            .sample_counts
                            .contains(vk::SampleCountFlags::TYPE_1 | vk::SampleCountFlags::TYPE_4)
                    })
            {
                eprintln!("skipping cross-graph depth resolve: D32 1x/4x SAMPLE_ZERO unsupported");
                return Ok(());
            }
            let mut pool = HashPool::new(&device);
            let pipeline = GraphicsPipeline::create(
                &device,
                GraphicsPipelineInfo::builder()
                    .cull_mode(vk::CullModeFlags::NONE)
                    .samples(SampleCount::Type4),
                [
                    glsl!(kind: vert, r#"
                    #version 450
                    void main() {
                        vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                        gl_Position = vec4(p * 2.0 - 1.0, 0.25, 1.0);
                    }
                "#)
                    .as_slice(),
                    glsl!(kind: frag, r#"
                    #version 450
                    void main() {}
                "#)
                    .as_slice(),
                ],
            )?;
            let source = Arc::new(Image::create(
                &device,
                info.into_builder().sample_count(SampleCount::Type4),
            )?);
            let resolved = Arc::new(Image::create(&device, info)?);
            let output = Arc::new(Buffer::create(
                &device,
                BufferInfo::host_mem(16, vk::BufferUsageFlags::TRANSFER_DST),
            )?);
            let mut graph = Graph::new();
            let source_node = graph.bind_resource(&source);
            let resolved_node = graph.bind_resource(&resolved);
            for index in 0..3 {
                let mut draw = graph
                    .begin_cmd()
                    .bind_pipeline(&pipeline)
                    .depth_stencil(DepthStencilInfo::DEPTH_WRITE_LESS)
                    .depth_stencil_attachment_image(
                        source_node,
                        if index == 0 {
                            LoadOp::CLEAR_ONE_STENCIL_ZERO
                        } else {
                            LoadOp::Load
                        },
                        StoreOp::Store,
                    );
                if index == 2 {
                    draw = draw.depth_stencil_attachment_resolve_image(
                        0,
                        resolved_node,
                        Some(ResolveMode::SampleZero),
                        None,
                    );
                }
                draw.record_cmd(|cmd| {
                    cmd.set_scissor(
                        0,
                        &[vk::Rect2D::default().extent(vk::Extent2D {
                            width: 1,
                            height: 2,
                        })],
                    )
                    .draw(3, 1, 0, 0);
                });
            }
            let mut producer = graph.finalize().queue_submit(&mut pool, 0, 0)?;
            for (image, access) in [
                (&source, vk::AccessFlags::COLOR_ATTACHMENT_READ),
                (&resolved, vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
            ] {
                let sync = image.sync_info();
                assert!(!sync.subresources.is_empty());
                for subresource in sync.subresources {
                    assert!(
                        subresource
                            .stage_mask
                            .contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    );
                    assert!(subresource.access_mask.contains(access));
                    assert!(matches!(
                        subresource.layout,
                        Some(
                            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                | vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL
                        )
                    ));
                }
            }

            // No host wait or semaphore: the next graph must consume the persisted resolve scopes.
            let mut graph = Graph::new();
            let resolved_node = graph.bind_resource(&resolved);
            let output_node = graph.bind_resource(&output);
            graph
                .begin_cmd()
                .copy_image_to_buffer(
                    resolved_node,
                    output_node,
                    [vk::BufferImageCopy::default()
                        .buffer_row_length(2)
                        .buffer_image_height(2)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: 2,
                            height: 2,
                            depth: 1,
                        })],
                )
                .end_cmd();
            let source_node = graph.bind_resource(&source);
            graph
                .begin_cmd()
                .bind_pipeline(&pipeline)
                .depth_stencil(DepthStencilInfo::DEPTH_WRITE_LESS)
                .depth_stencil_attachment_image(
                    source_node,
                    LoadOp::CLEAR_ONE_STENCIL_ZERO,
                    StoreOp::Store,
                )
                .record_cmd(|cmd| {
                    cmd.draw(3, 1, 0, 0);
                });
            graph
                .begin_cmd()
                .resource_access(output_node, AccessType::HostRead)
                .record_cmd(|_| {});
            graph.finalize().queue_submit(&mut pool, 0, 0)?.wait()?;
            producer.wait()?;
            for (pixel, bytes) in Buffer::mapped_slice(&output).chunks_exact(4).enumerate() {
                assert_eq!(
                    f32::from_ne_bytes(bytes.try_into().unwrap()),
                    if pixel % 2 == 0 { 0.25 } else { 1.0 }
                );
            }
        }
        drop(device);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and validation layers"]
    fn vulkan_late_depth_resolve_public_api() -> Result<(), DriverError> {
        use crate::driver::{graphics::StencilMode, render_pass::ResolveMode};

        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        {
            let mut pool = HashPool::new(&device);
            let pipeline = GraphicsPipeline::create(
                &device,
                GraphicsPipelineInfo::builder()
                    .cull_mode(vk::CullModeFlags::NONE)
                    .samples(SampleCount::Type4),
                [
                    glsl!(kind: vert, r#"
                    #version 450
                    layout(push_constant) uniform Draw { float depth; } params;
                    void main() {
                        vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                        gl_Position = vec4(p * 2.0 - 1.0, params.depth, 1.0);
                    }
                "#)
                    .as_slice(),
                    glsl!(kind: frag, r#"
                    #version 450
                    void main() {}
                "#)
                    .as_slice(),
                ],
            )?;
            for format in [
                vk::Format::D32_SFLOAT,
                vk::Format::S8_UINT,
                vk::Format::D32_SFLOAT_S8_UINT,
            ] {
                let aspects = crate::driver::format_aspect_mask(format);
                let has_depth = aspects.contains(vk::ImageAspectFlags::DEPTH);
                let has_stencil = aspects.contains(vk::ImageAspectFlags::STENCIL);
                let modes = &device.physical.depth_stencil_resolve_properties;
                if (has_depth
                    && !modes
                        .supported_depth_resolve_modes
                        .contains(vk::ResolveModeFlags::SAMPLE_ZERO))
                    || (has_stencil
                        && !modes
                            .supported_stencil_resolve_modes
                            .contains(vk::ResolveModeFlags::SAMPLE_ZERO))
                {
                    eprintln!("skipping late resolve {format:?}: SAMPLE_ZERO unsupported");
                    continue;
                }
                let depth_info = ImageInfo::image_2d(
                    2,
                    2,
                    format,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                );
                let resolve_info = depth_info
                    .into_builder()
                    .usage(
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                            | vk::ImageUsageFlags::TRANSFER_SRC,
                    )
                    .build();
                let color_info = ImageInfo::image_2d(
                    2,
                    2,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT,
                );
                let mut supported = true;
                for (info, samples) in [
                    (depth_info, vk::SampleCountFlags::TYPE_4),
                    (resolve_info, vk::SampleCountFlags::TYPE_1),
                    (color_info, vk::SampleCountFlags::TYPE_4),
                ] {
                    supported &= device
                        .physical
                        .image_format_properties(
                            info.format,
                            info.image_type,
                            info.tiling,
                            info.usage,
                            info.flags,
                        )?
                        .is_some_and(|properties| properties.sample_counts.contains(samples));
                }
                if !supported {
                    eprintln!("skipping late resolve {format:?}: image/sample support unavailable");
                    continue;
                }

                for (resolves, expected_mapping, late_clear) in [
                    (vec![false, false, true], vec![0, 0, 1], false),
                    (vec![true, false, false], vec![0, 1, 1], false),
                    (
                        vec![false, false, true, false, false],
                        vec![0, 0, 1, 2, 2],
                        false,
                    ),
                    (
                        vec![false, true, false, true, false],
                        vec![0, 1, 2, 3, 4],
                        false,
                    ),
                    (vec![false, false, true], vec![0, 0, 1], true),
                ] {
                    for (color_count, late_colors) in [(0, false), (2, false), (2, true)] {
                        let mut graph = Graph::new();
                        let depth = graph.bind_resource(Image::create(
                            &device,
                            depth_info.into_builder().sample_count(SampleCount::Type4),
                        )?);
                        let resolved = graph.bind_resource(Image::create(&device, resolve_info)?);
                        let colors = (0..color_count)
                            .map(|_| {
                                Image::create(
                                    &device,
                                    color_info.into_builder().sample_count(SampleCount::Type4),
                                )
                                .map(|image| graph.bind_resource(image))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        for (index, &resolve) in resolves.iter().enumerate() {
                            let exec_color_count = if late_colors && index == 0 {
                                0
                            } else {
                                color_count
                            };
                            let stencil = StencilMode {
                                pass_op: vk::StencilOp::REPLACE,
                                compare_op: vk::CompareOp::ALWAYS,
                                compare_mask: 255,
                                write_mask: 255,
                                reference: 31 + index as u32,
                                ..StencilMode::IGNORE
                            };
                            let mut draw = graph
                                .begin_cmd()
                                .bind_pipeline(&pipeline)
                                .depth_stencil(
                                    DepthStencilInfo::DEPTH_WRITE_LESS
                                        .into_builder()
                                        .depth_test(has_depth)
                                        .depth_write(has_depth)
                                        .stencil_test(has_stencil)
                                        .front(stencil)
                                        .back(stencil)
                                        .build(),
                                )
                                .depth_stencil_attachment_image(
                                    depth,
                                    if index == 0 || (late_clear && resolve) {
                                        LoadOp::Clear(vk::ClearDepthStencilValue {
                                            depth: if index == 0 { 1.0 } else { 0.875 },
                                            stencil: if index == 0 { 90 } else { 190 },
                                        })
                                    } else {
                                        LoadOp::Load
                                    },
                                    StoreOp::Store,
                                );
                            for (slot, &color) in
                                colors.iter().take(exec_color_count as usize).enumerate()
                            {
                                draw = draw.color_attachment_image(
                                    slot as u32,
                                    color,
                                    if index == 0 || (late_colors && index == 1) {
                                        LoadOp::CLEAR_BLACK_ALPHA_ZERO
                                    } else {
                                        LoadOp::Load
                                    },
                                    StoreOp::Store,
                                );
                            }
                            if resolve {
                                draw = draw.depth_stencil_attachment_resolve_image(
                                    exec_color_count,
                                    resolved,
                                    has_depth.then_some(ResolveMode::SampleZero),
                                    has_stencil.then_some(ResolveMode::SampleZero),
                                );
                            }
                            draw.record_cmd(move |cmd| {
                                // The untouched column verifies the clear, including a later clear
                                // in the subpass where the depth/stencil resolve first appears.
                                cmd.set_scissor(
                                    0,
                                    &[vk::Rect2D::default().extent(vk::Extent2D {
                                        width: 1,
                                        height: 2,
                                    })],
                                )
                                .push_constants(0, &(0.75 - index as f32 * 0.125).to_ne_bytes())
                                .draw(3, 1, 0, 0);
                            });
                        }
                        let mut outputs = Vec::new();
                        for aspect in [vk::ImageAspectFlags::DEPTH, vk::ImageAspectFlags::STENCIL] {
                            if !aspects.contains(aspect) {
                                continue;
                            }
                            let output = Arc::new(Buffer::create(
                                &device,
                                BufferInfo::host_mem(
                                    // The copy helper tracks the full format's texel size, even
                                    // when Vulkan copies just one depth/stencil aspect.
                                    32,
                                    vk::BufferUsageFlags::TRANSFER_DST,
                                ),
                            )?);
                            let output_node = graph.bind_resource(&output);
                            graph
                                .begin_cmd()
                                .copy_image_to_buffer(
                                    resolved,
                                    output_node,
                                    [vk::BufferImageCopy::default()
                                        .buffer_row_length(2)
                                        .buffer_image_height(2)
                                        .image_subresource(
                                            vk::ImageSubresourceLayers::default()
                                                .aspect_mask(aspect)
                                                .layer_count(1),
                                        )
                                        .image_extent(vk::Extent3D {
                                            width: 2,
                                            height: 2,
                                            depth: 1,
                                        })],
                                )
                                .end_cmd();
                            graph
                                .begin_cmd()
                                .resource_access(output_node, AccessType::HostRead)
                                .record_cmd(|_| {});
                            outputs.push((aspect, output));
                        }
                        let mut submission = graph.finalize();
                        submission.prepare_command_stream(&mut pool)?;
                        let recording = &submission.recorded_commands[0];
                        let mut expected_mapping = expected_mapping.clone();
                        if late_colors && !resolves[0] && !resolves[1] {
                            for subpass in &mut expected_mapping[1..] {
                                *subpass += 1;
                            }
                        }
                        assert_eq!(&*recording.exec_subpasses, expected_mapping.as_slice());
                        let info = &recording.render_pass.as_ref().unwrap().info;
                        assert_eq!(info.attachments.len(), color_count as usize + 2);
                        assert_eq!(
                            info.attachments[color_count as usize].load_op,
                            if has_depth {
                                vk::AttachmentLoadOp::CLEAR
                            } else {
                                vk::AttachmentLoadOp::DONT_CARE
                            }
                        );
                        let resolve_attachment = &info.attachments[color_count as usize + 1];
                        assert_eq!(
                            resolve_attachment.store_op,
                            if has_depth {
                                vk::AttachmentStoreOp::STORE
                            } else {
                                vk::AttachmentStoreOp::DONT_CARE
                            }
                        );
                        assert_eq!(
                            resolve_attachment.stencil_store_op,
                            if has_stencil {
                                vk::AttachmentStoreOp::STORE
                            } else {
                                vk::AttachmentStoreOp::DONT_CARE
                            }
                        );
                        for (index, &subpass) in recording.exec_subpasses.iter().enumerate() {
                            let subpass = &info.subpasses[subpass as usize];
                            assert_eq!(
                                subpass.depth_stencil_attachment.unwrap().attachment,
                                color_count
                            );
                            assert_eq!(
                                subpass
                                    .depth_stencil_resolve_attachment
                                    .map(|(attachment, _, _)| attachment.attachment),
                                resolves[index].then_some(color_count + 1)
                            );
                        }
                        // Return inspection leases before exercising the ordinary submission path.
                        submission.recorded_commands.clear();
                        submission.queue_submit(&mut pool, 0, 0)?.wait()?;
                        let last_resolve = resolves.iter().rposition(|&resolve| resolve).unwrap();
                        for (aspect, output) in outputs {
                            let bytes = Buffer::mapped_slice(&output);
                            if aspect == vk::ImageAspectFlags::DEPTH {
                                for (pixel, value) in bytes[..16].chunks_exact(4).enumerate() {
                                    assert_eq!(
                                        f32::from_ne_bytes(value.try_into().unwrap()),
                                        if pixel % 2 == 0 {
                                            0.75 - last_resolve as f32 * 0.125
                                        } else if late_clear {
                                            0.875
                                        } else {
                                            1.0
                                        },
                                        "format={format:?} colors={color_count} resolves={resolves:?} late_clear={late_clear} pixel={pixel}"
                                    );
                                }
                            } else {
                                for (pixel, &value) in bytes[..4].iter().enumerate() {
                                    assert_eq!(
                                        value,
                                        if pixel % 2 == 0 {
                                            31 + last_resolve as u8
                                        } else if late_clear {
                                            190
                                        } else {
                                            90
                                        },
                                        "format={format:?} colors={color_count} resolves={resolves:?} late_clear={late_clear} pixel={pixel}"
                                    );
                                }
                            }
                        }
                    }
                }
                eprintln!(
                    "late resolve {format:?}: appearing/disappearing resolves at depth slots 0 and 2, including color-slot growth and later clears, passed"
                );
            }
        }
        drop(device);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and validation layers"]
    fn vulkan_later_color_clear_public_api() -> Result<(), DriverError> {
        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        {
            let mut pool = HashPool::new(&device);
            let pipeline = GraphicsPipeline::create(
                &device,
                GraphicsPipelineInfo::builder().cull_mode(vk::CullModeFlags::NONE),
                [
                    glsl!(kind: vert, r#"
                    #version 450
                    void main() {
                        vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
                    }
                "#)
                    .as_slice(),
                    glsl!(kind: frag, r#"
                    #version 450
                    layout(location = 0) out uvec4 color;
                    layout(location = 1) out uvec4 extra;
                    void main() {
                        color = uvec4(11, 22, 33, 44);
                        extra = uvec4(55, 66, 77, 88);
                    }
                "#)
                    .as_slice(),
                ],
            )?;
            let info = ImageInfo::image_2d(
                2,
                2,
                vk::Format::R32G32B32A32_UINT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            );
            let initial = [1u32, 2, 3, 4];
            let later = [0x8000_0001u32, 0xffff_ffff, 0x7fc0_1234, 0x0100_0001];
            let added = [101u32, 102, 103, 104];
            for layers in [1u32, 2] {
                if layers == 2
                    && (!device.physical.features_v1_1.multiview
                        || device.physical.properties_v1_1.max_multiview_view_count < 2)
                {
                    eprintln!("skipping two-view later color clear: multiview unsupported");
                    continue;
                }
                let view_mask = if layers == 1 { 0 } else { 0b11 };
                for repeated in [false, true] {
                    for clear_only in [false, true] {
                        let mut graph = Graph::new();
                        let target = graph.bind_resource(Image::create(
                            &device,
                            info.into_builder().array_layer_count(layers),
                        )?);
                        let extra = graph.bind_resource(Image::create(
                            &device,
                            info.into_builder().array_layer_count(layers),
                        )?);
                        let mut command = graph.begin_cmd().bind_pipeline(&pipeline);
                        for index in 0..2 {
                            command = command.multiview(view_mask, 0).color_attachment_image(
                                0,
                                target,
                                LoadOp::Clear(if index == 0 { initial } else { later }.into()),
                                StoreOp::Store,
                            );
                            if index == 1 {
                                // Slot growth requires a new subpass: clear slot 0 there, while
                                // slot 1 still gets its first-declaration render-pass load clear.
                                command = command.color_attachment_image(
                                    1,
                                    extra,
                                    LoadOp::Clear(added.into()),
                                    StoreOp::Store,
                                );
                            }
                            command = command.record_cmd(move |cmd| {
                                cmd.set_scissor(
                                    0,
                                    &[vk::Rect2D::default().extent(vk::Extent2D {
                                        width: 1,
                                        height: 2,
                                    })],
                                );
                                if index == 1 && !clear_only {
                                    cmd.draw(3, 1, 0, 0);
                                }
                            });
                            if index == 0 && !repeated {
                                command.end_cmd();
                                command = graph.begin_cmd().bind_pipeline(&pipeline);
                            }
                        }
                        command.end_cmd();
                        let mut outputs = Vec::new();
                        for image in [target, extra] {
                            let output = Arc::new(Buffer::create(
                                &device,
                                BufferInfo::host_mem(
                                    u64::from(2 * 2 * 16 * layers),
                                    vk::BufferUsageFlags::TRANSFER_DST,
                                ),
                            )?);
                            let output_node = graph.bind_resource(&output);
                            graph
                                .begin_cmd()
                                .copy_image_to_buffer(
                                    image,
                                    output_node,
                                    [vk::BufferImageCopy::default()
                                        .buffer_row_length(2)
                                        .buffer_image_height(2)
                                        .image_subresource(
                                            vk::ImageSubresourceLayers::default()
                                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                                .layer_count(layers),
                                        )
                                        .image_extent(vk::Extent3D {
                                            width: 2,
                                            height: 2,
                                            depth: 1,
                                        })],
                                )
                                .end_cmd();
                            graph
                                .begin_cmd()
                                .resource_access(output_node, AccessType::HostRead)
                                .record_cmd(|_| {});
                            outputs.push(output);
                        }
                        graph.finalize().queue_submit(&mut pool, 0, 0)?.wait()?;
                        for (slot, output) in outputs.iter().enumerate() {
                            for (pixel, bytes) in
                                Buffer::mapped_slice(output).chunks_exact(16).enumerate()
                            {
                                // A callback scissor must not clip either attachment clear.
                                let expected = if slot == 1 {
                                    if !clear_only && pixel % 2 == 0 {
                                        [55, 66, 77, 88]
                                    } else {
                                        added
                                    }
                                } else if !clear_only && pixel % 2 == 0 {
                                    [11, 22, 33, 44]
                                } else {
                                    later
                                };
                                assert_eq!(
                                    bytes,
                                    bytemuck::cast_slice::<_, u8>(&expected),
                                    "layers={layers} repeated={repeated} clear_only={clear_only} slot={slot} pixel={pixel}"
                                );
                            }
                        }
                    }
                }
            }
        }
        drop(device);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and validation layers"]
    fn vulkan_later_color_clear_stream_replay_public_api() -> Result<(), DriverError> {
        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        {
            let mut pool = HashPool::new(&device);
            let pipeline = SubpassFixture::test_triangle_pipeline(&device)?;
            let info = ImageInfo::image_2d(
                2,
                2,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            );
            for prepared in [false, true] {
                let draft = CommandStream::finalize(|stream| {
                    let target = stream.arg(info);
                    for color in [[0.0f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]] {
                        stream
                            .begin_cmd()
                            .bind_pipeline(&pipeline)
                            .color_attachment_image(
                                0,
                                target,
                                LoadOp::Clear(color.into()),
                                StoreOp::Store,
                            )
                            .record_cmd(|cmd| {
                                // Clear-only callbacks still clear outside this smaller scissor.
                                cmd.set_scissor(
                                    0,
                                    &[vk::Rect2D::default().extent(vk::Extent2D {
                                        width: 1,
                                        height: 2,
                                    })],
                                );
                            });
                    }
                    target
                });
                let stream = if prepared {
                    draft.prepare(&mut pool)?
                } else {
                    draft.into_stream()
                };
                for round in 0..2 {
                    let mut graph = Graph::new();
                    let mut outputs = Vec::new();
                    for _ in 0..2 {
                        let target = graph.bind_resource(Image::create(&device, info)?);
                        let output = Arc::new(Buffer::create(
                            &device,
                            BufferInfo::host_mem(16, vk::BufferUsageFlags::TRANSFER_DST),
                        )?);
                        graph
                            .insert_cmd_stream(&stream)
                            .with_arg(stream.args, target)
                            .finish();
                        SubpassFixture::readback(&mut graph, target.into(), &output);
                        outputs.push(output);
                    }
                    graph.finalize().queue_submit(&mut pool, 0, 0)?.wait()?;
                    for (invocation, output) in outputs.iter().enumerate() {
                        assert_eq!(
                            Buffer::mapped_slice(output),
                            [0, 255, 0, 255].repeat(4),
                            "prepared={prepared} round={round} invocation={invocation}"
                        );
                    }
                }
            }
        }
        drop(device);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and validation layers"]
    fn vulkan_later_depth_stencil_clear_public_api() -> Result<(), DriverError> {
        use crate::driver::graphics::StencilMode;

        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        {
            let mut pool = HashPool::new(&device);
            let pipeline = GraphicsPipeline::create(
                &device,
                GraphicsPipelineInfo::builder().cull_mode(vk::CullModeFlags::NONE),
                [
                    glsl!(kind: vert, r#"
                    #version 450
                    void main() {
                        vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                        gl_Position = vec4(p * 2.0 - 1.0, 0.25, 1.0);
                    }
                "#)
                    .as_slice(),
                    glsl!(kind: frag, r#"
                    #version 450
                    void main() {}
                "#)
                    .as_slice(),
                ],
            )?;
            for format in [
                vk::Format::D32_SFLOAT,
                vk::Format::S8_UINT,
                vk::Format::D32_SFLOAT_S8_UINT,
            ] {
                let info = ImageInfo::image_2d(
                    2,
                    2,
                    format,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                );
                if device
                    .physical
                    .image_format_properties(
                        info.format,
                        info.image_type,
                        info.tiling,
                        info.usage,
                        info.flags,
                    )?
                    .is_none()
                {
                    eprintln!("skipping later clear {format:?}: image support unavailable");
                    continue;
                }
                let aspects = crate::driver::format_aspect_mask(format);
                let has_depth = aspects.contains(vk::ImageAspectFlags::DEPTH);
                let has_stencil = aspects.contains(vk::ImageAspectFlags::STENCIL);
                let stencil = StencilMode {
                    pass_op: vk::StencilOp::REPLACE,
                    compare_op: vk::CompareOp::ALWAYS,
                    compare_mask: 255,
                    write_mask: 255,
                    reference: 173,
                    ..StencilMode::IGNORE
                };
                let state = DepthStencilInfo::DEPTH_WRITE_LESS
                    .into_builder()
                    .depth_test(has_depth)
                    .depth_write(has_depth)
                    .stencil_test(has_stencil)
                    .front(stencil)
                    .back(stencil)
                    .build();
                for repeated in [false, true] {
                    for clear_only in [false, true] {
                        let mut graph = Graph::new();
                        let target = graph.bind_resource(Image::create(&device, info)?);
                        let mut command = graph.begin_cmd().bind_pipeline(&pipeline);
                        for index in 0..2 {
                            command = command
                                .depth_stencil(state)
                                .depth_stencil_attachment_image(
                                    target,
                                    LoadOp::Clear(vk::ClearDepthStencilValue {
                                        depth: if index == 0 { 0.0 } else { 0.75 },
                                        stencil: if index == 0 { 91 } else { 37 },
                                    }),
                                    StoreOp::Store,
                                )
                                .record_cmd(move |cmd| {
                                    cmd.set_scissor(
                                        0,
                                        &[vk::Rect2D::default().extent(vk::Extent2D {
                                            width: 1,
                                            height: 2,
                                        })],
                                    );
                                    if index == 1 && !clear_only {
                                        // This passes LESS only after the later depth clear.
                                        cmd.draw(3, 1, 0, 0);
                                    }
                                });
                            if index == 0 && !repeated {
                                command.end_cmd();
                                command = graph.begin_cmd().bind_pipeline(&pipeline);
                            }
                        }
                        command.end_cmd();
                        let mut outputs = Vec::new();
                        for aspect in [vk::ImageAspectFlags::DEPTH, vk::ImageAspectFlags::STENCIL] {
                            if !aspects.contains(aspect) {
                                continue;
                            }
                            let output = Arc::new(Buffer::create(
                                &device,
                                // The copy helper sizes combined formats by the full texel size.
                                BufferInfo::host_mem(32, vk::BufferUsageFlags::TRANSFER_DST),
                            )?);
                            let output_node = graph.bind_resource(&output);
                            graph
                                .begin_cmd()
                                .copy_image_to_buffer(
                                    target,
                                    output_node,
                                    [vk::BufferImageCopy::default()
                                        .buffer_row_length(2)
                                        .buffer_image_height(2)
                                        .image_subresource(
                                            vk::ImageSubresourceLayers::default()
                                                .aspect_mask(aspect)
                                                .layer_count(1),
                                        )
                                        .image_extent(vk::Extent3D {
                                            width: 2,
                                            height: 2,
                                            depth: 1,
                                        })],
                                )
                                .end_cmd();
                            graph
                                .begin_cmd()
                                .resource_access(output_node, AccessType::HostRead)
                                .record_cmd(|_| {});
                            outputs.push((aspect, output));
                        }
                        graph.finalize().queue_submit(&mut pool, 0, 0)?.wait()?;
                        for (aspect, output) in outputs {
                            let bytes = Buffer::mapped_slice(&output);
                            for pixel in 0..4 {
                                let drawn = !clear_only && pixel % 2 == 0;
                                if aspect == vk::ImageAspectFlags::DEPTH {
                                    assert_eq!(
                                        f32::from_ne_bytes(
                                            bytes[pixel * 4..pixel * 4 + 4].try_into().unwrap()
                                        ),
                                        if drawn { 0.25 } else { 0.75 },
                                        "format={format:?} repeated={repeated} clear_only={clear_only} pixel={pixel}"
                                    );
                                } else {
                                    assert_eq!(
                                        bytes[pixel],
                                        if drawn { 173 } else { 37 },
                                        "format={format:?} repeated={repeated} clear_only={clear_only} pixel={pixel}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        drop(device);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and validation layers"]
    fn vulkan_subpass_buffer_reads_retain_upload_for_next_graph() -> Result<(), DriverError> {
        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        {
            let mut pool = HashPool::new(&device);
            let fixture = SubpassFixture::new(&device, &mut pool, 1)?;
            let compute = ComputePipeline::create(
                &device,
                ComputePipelineInfo::default(),
                glsl!(kind: comp, r#"
                #version 450
                layout(local_size_x = 1) in;
                layout(set = 0, binding = 0, std430) readonly buffer Source {
                    uvec4 value;
                } source;
                layout(set = 0, binding = 1, std430) writeonly buffer Output {
                    uvec4 value;
                } output_data;
                void main() { output_data.value = source.value; }
            "#)
                .as_slice(),
            )?;
            for intermediate_compute in [false, true] {
                let buffer = Arc::new(Buffer::create(
                    &device,
                    BufferInfo::device_mem(
                        16,
                        vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::UNIFORM_BUFFER
                            | vk::BufferUsageFlags::STORAGE_BUFFER,
                    ),
                )?);
                let computed = Arc::new(Buffer::create(
                    &device,
                    BufferInfo::host_mem(16, vk::BufferUsageFlags::STORAGE_BUFFER),
                )?);
                let copied = Arc::new(Buffer::create(
                    &device,
                    BufferInfo::host_mem(16, vk::BufferUsageFlags::TRANSFER_DST),
                )?);
                for index in [1, 258, 519] {
                    let values = SubpassFixture::uniform_color(index);
                    let bytes: &[u8] = bytemuck::cast_slice(&values);
                    let (target, pixels) = fixture.output(&device)?;
                    let mut graph = Graph::new();
                    let staging = graph.bind_resource(Buffer::create_from_slice(
                        &device,
                        vk::BufferUsageFlags::TRANSFER_SRC,
                        bytes,
                    )?);
                    let buffer_node = graph.bind_resource(&buffer);
                    graph.copy_buffer(staging, buffer_node);
                    if intermediate_compute {
                        let computed = graph.bind_resource(&computed);
                        graph
                            .begin_cmd()
                            .bind_pipeline(&compute)
                            .shader_resource_access(
                                0,
                                buffer_node,
                                AccessType::ComputeShaderReadOther,
                            )
                            .shader_resource_access(1, computed, AccessType::ComputeShaderWrite)
                            .record_cmd(|cmd| {
                                cmd.dispatch(1, 1, 1);
                            });
                    }
                    let target_node = graph.bind_resource(&target);
                    let texture = graph.bind_resource(&fixture.textures[0]);
                    fixture.draws(
                        &mut graph,
                        target_node.into(),
                        &[(buffer_node.into(), texture.into())],
                        false,
                        None,
                        false,
                    );
                    let mut drawn = graph.finalize().queue_submit(&mut pool, 0, 0)?;
                    let sync = buffer.sync_info();

                    // No wait or semaphore between graphs: the next queue submission must recover
                    // the upload writer from buffer history, not just the last shader reader.
                    let mut readback = Graph::new();
                    let buffer_node = readback.bind_resource(&buffer);
                    let copied_node = readback.bind_resource(&copied);
                    readback.copy_buffer(buffer_node, copied_node);
                    readback
                        .begin_cmd()
                        .resource_access(copied_node, AccessType::HostRead)
                        .record_cmd(|_| {});
                    if intermediate_compute {
                        let computed = readback.bind_resource(&computed);
                        readback
                            .begin_cmd()
                            .resource_access(computed, AccessType::HostRead)
                            .record_cmd(|_| {});
                    }
                    let target = readback.bind_resource(&target);
                    SubpassFixture::readback(&mut readback, target.into(), &pixels);
                    readback.finalize().queue_submit(&mut pool, 0, 0)?.wait()?;
                    drawn.wait()?;

                    assert_eq!(
                        Buffer::mapped_slice(&copied),
                        bytes,
                        "buffer copy: intermediate_compute={intermediate_compute} index={index}"
                    );
                    if intermediate_compute {
                        assert_eq!(
                            Buffer::mapped_slice(&computed),
                            bytes,
                            "compute read: index={index}"
                        );
                    }
                    assert_eq!(
                        Buffer::mapped_slice(&pixels),
                        SubpassFixture::expected(index, 0, 17).repeat(4),
                        "graphics read: intermediate_compute={intermediate_compute} index={index}"
                    );
                    assert!(
                        sync.ranges.iter().any(|range| range.access_mask.intersects(
                            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::MEMORY_WRITE
                        )),
                        "graphics read lost upload writer: intermediate_compute={intermediate_compute} index={index}"
                    );
                }
            }
        }
        drop(device);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and validation layers"]
    fn vulkan_subpass_color_depth_resolves_keep_boundary() -> Result<(), DriverError> {
        use crate::driver::render_pass::ResolveMode;

        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        let color_info = ImageInfo::image_2d(
            2,
            2,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        );
        let depth_info = ImageInfo::image_2d(
            2,
            2,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        );
        let mut supported = device
            .physical
            .depth_stencil_resolve_properties
            .supported_depth_resolve_modes
            .contains(vk::ResolveModeFlags::SAMPLE_ZERO);
        if !supported {
            eprintln!("skipping resolve regression: SAMPLE_ZERO depth resolve unsupported");
        }
        for info in [color_info, depth_info] {
            let properties = device.physical.image_format_properties(
                info.format,
                info.image_type,
                info.tiling,
                info.usage,
                info.flags,
            )?;
            if !properties.is_some_and(|properties| {
                properties
                    .sample_counts
                    .contains(vk::SampleCountFlags::TYPE_1 | vk::SampleCountFlags::TYPE_4)
            }) {
                eprintln!(
                    "skipping resolve regression: {:?} lacks 1x/4x image support",
                    info.format
                );
                supported = false;
            }
        }
        if supported {
            let mut pool = HashPool::new(&device);
            let pipeline = GraphicsPipeline::create(
                &device,
                GraphicsPipelineInfo::builder()
                    .cull_mode(vk::CullModeFlags::NONE)
                    .samples(SampleCount::Type4),
                [
                    glsl!(kind: vert, r#"
                    #version 450
                    layout(push_constant) uniform Draw { vec4 color; float depth; } params;
                    void main() {
                        vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                        gl_Position = vec4(p * 2.0 - 1.0, params.depth, 1.0);
                    }
                "#)
                    .as_slice(),
                    glsl!(kind: frag, r#"
                    #version 450
                    layout(push_constant) uniform Draw { vec4 color; float depth; } params;
                    layout(location = 0) out vec4 color;
                    void main() { color = params.color; }
                "#)
                    .as_slice(),
                ],
            )?;
            let mut graph = Graph::new();
            let color = graph.bind_resource(Image::create(
                &device,
                color_info.into_builder().sample_count(SampleCount::Type4),
            )?);
            let depth = graph.bind_resource(Image::create(
                &device,
                depth_info.into_builder().sample_count(SampleCount::Type4),
            )?);
            let color_resolve = graph.bind_resource(Image::create(&device, color_info)?);
            let depth_resolve = graph.bind_resource(Image::create(&device, depth_info)?);
            let output = Arc::new(Buffer::create(
                &device,
                BufferInfo::host_mem(2 * 2 * 4, vk::BufferUsageFlags::TRANSFER_DST),
            )?);
            for (index, constants) in [[1.0_f32, 0.0, 0.0, 1.0, 0.75], [0.0, 1.0, 0.0, 1.0, 0.25]]
                .into_iter()
                .enumerate()
            {
                graph
                    .begin_cmd()
                    .bind_pipeline(&pipeline)
                    .depth_stencil(DepthStencilInfo::DEPTH_WRITE_LESS)
                    .color_attachment_image(
                        0,
                        color,
                        if index == 0 {
                            LoadOp::CLEAR_BLACK_ALPHA_ZERO
                        } else {
                            LoadOp::Load
                        },
                        StoreOp::Store,
                    )
                    .color_attachment_resolve_image(0, 1, color_resolve)
                    .depth_stencil_attachment_image(
                        depth,
                        if index == 0 {
                            LoadOp::CLEAR_ONE_STENCIL_ZERO
                        } else {
                            LoadOp::Load
                        },
                        StoreOp::Store,
                    )
                    // Two color slots put depth at 2; the API places its resolve at index + 1.
                    .depth_stencil_attachment_resolve_image(
                        2,
                        depth_resolve,
                        Some(ResolveMode::SampleZero),
                        None,
                    )
                    .record_cmd(move |cmd| {
                        cmd.push_constants(0, bytemuck::cast_slice(&constants))
                            .draw(3, 1, 0, 0);
                    })
                    .end_cmd();
                // Retain the resolve for readback without binding its 1x image as a 4x color output.
                graph.cmds.last_mut().unwrap().execs[0]
                    .attachments
                    .color_attachment_mut(1)
                    .unwrap()
                    .store = StoreOp::Store;
            }
            SubpassFixture::readback(&mut graph, color_resolve.into(), &output);
            let mut submission = graph.finalize();
            submission.prepare_command_stream(&mut pool)?;
            assert_eq!(SubpassFixture::subpass_counts(&submission), [2]);
            let recording = &submission.recorded_commands[0];
            assert_eq!(&*recording.exec_subpasses, &[0, 1]);
            let info = &recording.render_pass.as_ref().unwrap().info;
            assert_eq!(info.attachments[1].store_op, vk::AttachmentStoreOp::STORE);
            for subpass in &info.subpasses {
                assert_eq!(subpass.color_attachments[0].attachment, 0);
                assert_eq!(
                    subpass.color_attachments[1].attachment,
                    vk::ATTACHMENT_UNUSED
                );
                assert_eq!(subpass.color_resolve_attachments[0].attachment, 1);
                assert_eq!(subpass.depth_stencil_attachment.unwrap().attachment, 2);
                let (resolve, mode, stencil_mode) =
                    subpass.depth_stencil_resolve_attachment.unwrap();
                assert_eq!(resolve.attachment, 3);
                assert_eq!(mode, Some(ResolveMode::SampleZero));
                assert_eq!(stencil_mode, None);
            }
            let dependency = info
                .dependencies
                .iter()
                .find(|dependency| dependency.src_subpass == 0 && dependency.dst_subpass == 1)
                .expect("missing resolve boundary dependency");
            // The first clear/draw has no color load: its COLOR_ATTACHMENT_READ scope comes
            // from fixed-function depth resolves, not fragment depth testing or a color load.
            for (stages, accesses) in [
                (dependency.src_stage_mask, dependency.src_access_mask),
                (dependency.dst_stage_mask, dependency.dst_access_mask),
            ] {
                assert!(stages.contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT));
                assert!(accesses.contains(
                    vk::AccessFlags::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                ));
            }
            submission.recorded_commands.clear();
            submission.queue_submit(&mut pool, 0, 0)?.wait()?;
            for pixel in Buffer::mapped_slice(&output).chunks_exact(4) {
                assert_eq!(
                    pixel,
                    [0, 255, 0, 255],
                    "color resolve must contain the second draw"
                );
            }
        }
        drop(device);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and validation layers"]
    fn vulkan_subpass_depth_clear_and_ordering() -> Result<(), DriverError> {
        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        if !device
            .physical
            .format_properties(vk::Format::D32_SFLOAT)
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            eprintln!("skipping depth regression: D32_SFLOAT depth attachment unsupported");
        } else {
            let mut pool = HashPool::new(&device);
            let fixture = SubpassFixture::new(&device, &mut pool, 2)?;
            for layers in [1u32, 2] {
                if layers == 2
                    && (!device.physical.features_v1_1.multiview
                        || device.physical.properties_v1_1.max_multiview_view_count < 2)
                {
                    eprintln!(
                        "skipping two-view depth variant: multiview with two views unsupported"
                    );
                    continue;
                }
                let view_mask = if layers == 1 { 0 } else { 0b11 };
                let mut graph = Graph::new();
                let target = graph.bind_resource(Image::create(
                    &device,
                    fixture.target_info.into_builder().array_layer_count(layers),
                )?);
                let depth = graph.bind_resource(Image::create(
                    &device,
                    ImageInfo::image_2d(
                        4,
                        2,
                        vk::Format::D32_SFLOAT,
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    )
                    .into_builder()
                    .array_layer_count(layers),
                )?);
                let output = Arc::new(Buffer::create(
                    &device,
                    BufferInfo::host_mem(
                        u64::from(4 * 2 * 4 * layers),
                        vk::BufferUsageFlags::TRANSFER_DST,
                    ),
                )?);
                // Clear only, far background, near left tile, then a middle-depth full-screen draw.
                // Vertex z is zero, so viewport min places each triangle at its chosen depth.
                for (index, (z, uniform, texture, salt, width)) in [
                    (0.0_f32, 0, 0, 0u32, 4u32),
                    (0.75, 0, 0, 17, 4),
                    (0.25, 1, 1, 18, 2),
                    (0.5, 0, 1, 19, 4),
                ]
                .into_iter()
                .enumerate()
                {
                    let uniform = graph.bind_resource(&fixture.uniforms[uniform]);
                    let texture = graph.bind_resource(&fixture.textures[texture]);
                    graph
                        .begin_cmd()
                        .bind_pipeline(&fixture.pipelines[0])
                        .multiview(view_mask, view_mask)
                        .depth_stencil(
                            DepthStencilInfo::DEPTH_WRITE_LESS
                                .into_builder()
                                .min(z)
                                .build(),
                        )
                        .color_attachment_image(
                            0,
                            target,
                            if index == 0 {
                                LoadOp::CLEAR_BLACK_ALPHA_ZERO
                            } else {
                                LoadOp::Load
                            },
                            StoreOp::Store,
                        )
                        .depth_stencil_attachment_image(
                            depth,
                            if index == 0 {
                                LoadOp::CLEAR_ONE_STENCIL_ZERO
                            } else {
                                LoadOp::Load
                            },
                            StoreOp::Store,
                        )
                        .shader_resource_access(
                            (0, 0),
                            uniform,
                            AccessType::FragmentShaderReadUniformBuffer,
                        )
                        .shader_resource_access(
                            (1, 0),
                            texture,
                            AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                        )
                        .record_cmd(move |cmd| {
                            if index != 0 {
                                cmd.set_scissor(
                                    0,
                                    &[vk::Rect2D {
                                        offset: vk::Offset2D { x: 0, y: 0 },
                                        extent: vk::Extent2D { width, height: 2 },
                                    }],
                                )
                                .push_constants(0, &salt.to_ne_bytes())
                                .draw(3, 1, 0, 0);
                            }
                        });
                }
                // The full-image convenience copy currently copies only layer zero.
                let output_node = graph.bind_resource(&output);
                graph
                    .begin_cmd()
                    .copy_image_to_buffer(
                        target,
                        output_node,
                        [vk::BufferImageCopy::default()
                            .buffer_row_length(4)
                            .buffer_image_height(2)
                            .image_subresource(
                                vk::ImageSubresourceLayers::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .layer_count(layers),
                            )
                            .image_extent(vk::Extent3D {
                                width: 4,
                                height: 2,
                                depth: 1,
                            })],
                    )
                    .end_cmd();
                graph
                    .begin_cmd()
                    .resource_access(output_node, AccessType::HostRead)
                    .record_cmd(|_| {});

                let mut submission = graph.finalize();
                submission.prepare_command_stream(&mut pool)?;
                assert_eq!(
                    SubpassFixture::subpass_counts(&submission),
                    [1],
                    "layers={layers}"
                );
                let recording = &submission.recorded_commands[0];
                assert_eq!(&*recording.exec_subpasses, &[0, 0, 0, 0]);
                let subpass = &recording.render_pass.as_ref().unwrap().info.subpasses[0];
                assert_eq!(subpass.view_mask, view_mask);
                assert_eq!(subpass.correlated_view_mask, view_mask);
                assert!(subpass.depth_stencil_attachment.is_some());
                submission.recorded_commands.clear();
                submission.queue_submit(&mut pool, 0, 0)?.wait()?;
                let expected = [
                    SubpassFixture::expected(1, 1, 18),
                    SubpassFixture::expected(0, 1, 19),
                ];
                for (pixel, actual) in Buffer::mapped_slice(&output).chunks_exact(4).enumerate() {
                    assert_eq!(
                        actual,
                        expected[pixel % 4 / 2],
                        "layers={layers} layer={} pixel={}",
                        pixel / 8,
                        pixel % 8
                    );
                }
            }
        }
        drop(device);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and validation layers"]
    fn vulkan_subpass_descriptors_and_state() -> Result<(), DriverError> {
        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        {
            let mut pool = HashPool::new(&device);
            let fixture = SubpassFixture::new(&device, &mut pool, 64)?;
            for supplied in [false, true] {
                let (target, output) = fixture.output(&device)?;
                let mut graph = Graph::new();
                let target = graph.bind_resource(&target);
                let inputs = (0..64)
                    .map(|tile| {
                        (
                            graph.bind_resource(&fixture.uniforms[tile]).into(),
                            graph
                                .bind_resource(&fixture.textures[tile * 13 % 64])
                                .into(),
                        )
                    })
                    .collect::<Vec<_>>();
                let (tracking, timestamps) =
                    fixture.draws(&mut graph, target.into(), &inputs, supplied, None, true);
                SubpassFixture::readback(&mut graph, target.into(), &output);
                for execution in &tracking {
                    assert_eq!(execution.has_submitted(), Ok(false));
                    assert_eq!(execution.has_executed(), Ok(false));
                }
                // Inspect the normal preparation resources, then return the leases before recording.
                let mut submission = graph.finalize();
                submission.prepare_command_stream(&mut pool)?;
                assert_eq!(SubpassFixture::subpass_counts(&submission), [1]);
                submission.recorded_commands.clear();
                let mut fence = submission.queue_submit(&mut pool, 0, 0)?;
                fence.wait()?;
                fixture.check(&output, 0, 17);
                for execution in tracking {
                    assert_eq!(execution.has_submitted(), Ok(true));
                    assert_eq!(execution.has_executed(), Ok(true));
                }
                if device.physical.queue_families[0].timestamp_valid_bits != 0 {
                    for query in timestamps {
                        assert!(
                            fence.timestamps.duration(query).is_some(),
                            "missing per-execution timestamp"
                        );
                    }
                }
                eprintln!(
                    "subpass correctness: N=64 supplied={supplied} all tiles/tracking/timestamps passed"
                );
            }
        }
        drop(device);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and validation layers"]
    fn vulkan_subpass_grouped_writers_feed_input_attachment() -> Result<(), DriverError> {
        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        {
            let mut pool = HashPool::new(&device);
            let fixture = SubpassFixture::new(&device, &mut pool, 2)?;
            let consumer = GraphicsPipeline::create(
                &device,
                GraphicsPipelineInfo::builder()
                    .cull_mode(vk::CullModeFlags::NONE)
                    .bindless_descriptor_count(1),
                [
                    glsl!(kind: vert, r#"
                    #version 450
                    void main() {
                        vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
                    }
                "#)
                    .as_slice(),
                    glsl!(kind: frag, r#"
                    #version 450
                    layout(input_attachment_index = 0, set = 0, binding = 0)
                        uniform subpassInput source;
                    layout(set = 1, binding = 0, std140) uniform Color { uvec4 value; } ubo;
                    layout(set = 2, binding = 0) uniform sampler2D tex;
                    layout(location = 1) out vec4 color;
                    void main() {
                        uvec3 rgb = uvec3(round(subpassLoad(source).rgb * 255.0));
                        rgb += ubo.value.rgb
                            + uvec3(round(texelFetch(tex, ivec2(0), 0).rgb * 255.0));
                        color = vec4(vec3(rgb & uvec3(255)) / 255.0, 1.0);
                    }
                "#)
                    .as_slice(),
                ],
            )?;
            let (target, output) = fixture.output(&device)?;
            let mut graph = Graph::new();
            let input = graph.bind_resource(Image::create(
                &device,
                fixture
                    .target_info
                    .into_builder()
                    .usage(fixture.target_info.usage | vk::ImageUsageFlags::INPUT_ATTACHMENT),
            )?);
            let target = graph.bind_resource(&target);
            let inputs = (0..2)
                .map(|tile| {
                    (
                        graph.bind_resource(&fixture.uniforms[tile]).into(),
                        graph.bind_resource(&fixture.textures[tile]).into(),
                    )
                })
                .collect::<Vec<_>>();
            fixture.draws(&mut graph, input.into(), &inputs, false, None, false);
            graph
                .begin_cmd()
                .bind_pipeline(&consumer)
                .color_attachment_image(0, input, LoadOp::Load, StoreOp::DontCare)
                .color_attachment_image(1, target, LoadOp::DontCare, StoreOp::Store)
                .shader_resource_access(
                    (1, 0),
                    inputs[0].0,
                    AccessType::FragmentShaderReadUniformBuffer,
                )
                .shader_resource_access(
                    (2, 0),
                    inputs[0].1,
                    AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                )
                .record_cmd(|cmd| {
                    cmd.draw(3, 1, 0, 0);
                })
                .end_cmd();
            // The builder marks inputs as color outputs and declares a color read. Match both
            // metadata and explicit access to the shader's input-only use, without feedback sync.
            let input_only = |exec: &mut crate::Execution| {
                let state = exec.attachments.color_attachment_mut(0).unwrap();
                state.is_attachment = false;
                let accesses = exec.accesses.get_mut(&state.attachment.target).unwrap();
                assert_eq!(accesses.len(), 1);
                assert_eq!(accesses[0].access, AccessType::ColorAttachmentRead);
                accesses[0].access = AccessType::FragmentShaderReadColorInputAttachment;
            };
            input_only(&mut graph.cmds.last_mut().unwrap().execs[0]);
            SubpassFixture::readback(&mut graph, target.into(), &output);

            let mut submission = graph.finalize();
            submission.prepare_command_stream(&mut pool)?;
            assert_eq!(SubpassFixture::subpass_counts(&submission), [2]);
            let recording = &submission.recorded_commands[0];
            assert_eq!(&*recording.exec_subpasses, &[0, 0, 1]);
            assert!(matches!(
                recording.descriptor_sets[2].as_slice(),
                [
                    super::RecordingDescriptorSet::Automatic(_),
                    super::RecordingDescriptorSet::Automatic(_),
                    super::RecordingDescriptorSet::Automatic(_)
                ]
            ));
            let subpass = &recording.render_pass.as_ref().unwrap().info.subpasses[1];
            assert_eq!(subpass.input_attachments[0].attachment, 0);
            assert_eq!(
                subpass.color_attachments[0].attachment,
                vk::ATTACHMENT_UNUSED
            );
            assert_eq!(subpass.color_attachments[1].attachment, 1);
            let check = |output: &Buffer, shift: usize, salt: u32| {
                let uniform = SubpassFixture::uniform_color(shift % 2);
                let texture = SubpassFixture::texture_color(shift % 2);
                for (pixel, actual) in Buffer::mapped_slice(output).chunks_exact(4).enumerate() {
                    let tile = pixel % 4 / 2;
                    let mut expected = SubpassFixture::expected(
                        (tile + shift) % 2,
                        (tile + shift) % 2,
                        salt + tile as u32,
                    );
                    for channel in 0..3 {
                        expected[channel] = expected[channel]
                            .wrapping_add(uniform[channel] as u8)
                            .wrapping_add(texture[channel]);
                    }
                    assert_eq!(actual, expected, "pixel={pixel} shift={shift} salt={salt}");
                }
            };
            submission.recorded_commands.clear();
            submission.queue_submit(&mut pool, 0, 0)?.wait()?;
            check(&output, 0, 17);

            let input_info = fixture
                .target_info
                .into_builder()
                .usage(fixture.target_info.usage | vk::ImageUsageFlags::INPUT_ATTACHMENT)
                .build();
            let stream = CommandStream::finalize(|stream| {
                let input = stream.arg(input_info);
                let target = stream.arg(fixture.target_info);
                let args = (0..2)
                    .map(|index| {
                        (
                            stream.arg(fixture.uniforms[index].info),
                            stream.arg(fixture.textures[index].info),
                        )
                    })
                    .collect::<Vec<_>>();
                let salt = stream.add_value_arg::<u32>();
                let inputs = args
                    .iter()
                    .map(|&(ubo, tex)| (ubo.into(), tex.into()))
                    .collect::<Vec<_>>();
                fixture.draws(
                    &mut stream.graph,
                    input.into(),
                    &inputs,
                    false,
                    Some(salt),
                    false,
                );
                stream
                    .graph
                    .begin_cmd()
                    .bind_pipeline(&consumer)
                    .color_attachment_image(0, input, LoadOp::Load, StoreOp::DontCare)
                    .color_attachment_image(1, target, LoadOp::DontCare, StoreOp::Store)
                    .shader_resource_access(
                        (1, 0),
                        args[0].0,
                        AccessType::FragmentShaderReadUniformBuffer,
                    )
                    .shader_resource_access(
                        (2, 0),
                        args[0].1,
                        AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                    )
                    .record_stream_mut(|cmd| {
                        cmd.draw(3, 1, 0, 0);
                    });
                input_only(&mut stream.graph.cmds.last_mut().unwrap().execs[0]);
                (input, target, args, salt)
            })
            .prepare(&mut pool)?;
            {
                let submission = stream.inner.submission.lock().unwrap();
                assert_eq!(SubpassFixture::subpass_counts(&submission), [2]);
                let recording = &submission.recorded_commands[0];
                assert_eq!(&*recording.exec_subpasses, &[0, 0, 1]);
                let info = &recording.render_pass.as_ref().unwrap().info;
                let dependency = info
                    .dependencies
                    .iter()
                    .find(|edge| edge.src_subpass == 0 && edge.dst_subpass == 1)
                    .expect("missing grouped-writer/input dependency");
                // Shared UBO and stable sampled-image reads must not globalize this edge.
                assert_eq!(dependency.dependency_flags, vk::DependencyFlags::BY_REGION);
                assert!(
                    dependency
                        .src_access_mask
                        .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                );
                assert!(
                    dependency
                        .dst_access_mask
                        .contains(vk::AccessFlags::INPUT_ATTACHMENT_READ)
                );
                assert!(
                    !dependency
                        .src_access_mask
                        .intersects(vk::AccessFlags::UNIFORM_READ | vk::AccessFlags::SHADER_READ)
                );
                assert!(
                    !dependency
                        .dst_access_mask
                        .intersects(vk::AccessFlags::UNIFORM_READ | vk::AccessFlags::SHADER_READ)
                );
            }
            for round in 0..2 {
                let mut pending = Vec::new();
                let mut outputs = Vec::new();
                for frame in 0..2 {
                    let mut graph = Graph::new();
                    for invocation in 0..2 {
                        let shift = round * 4 + frame * 2 + invocation;
                        let salt = 31 + shift as u32;
                        let input = graph.bind_resource(Image::create(&device, input_info)?);
                        let (target, output) = fixture.output(&device)?;
                        let target = graph.bind_resource(&target);
                        let inputs = (0..2)
                            .map(|tile| {
                                (
                                    graph.bind_resource(&fixture.uniforms[(tile + shift) % 2]),
                                    graph.bind_resource(&fixture.textures[(tile + shift) % 2]),
                                )
                            })
                            .collect::<Vec<_>>();
                        let mut run = graph
                            .insert_cmd_stream(&stream)
                            .with_arg(stream.args.0, input)
                            .with_arg(stream.args.1, target)
                            .with_value(stream.args.3, salt);
                        for (&(ubo, tex), (uniform, texture)) in stream.args.2.iter().zip(inputs) {
                            run = run.with_arg(ubo, uniform).with_arg(tex, texture);
                        }
                        run.finish();
                        SubpassFixture::readback(&mut graph, target.into(), &output);
                        outputs.push((output, shift, salt));
                    }
                    pending.push(graph.finalize().queue_submit(&mut pool, 0, 0)?);
                }
                // Four invocation leases stay live together; dropping pending frees them for reuse.
                for fence in &mut pending {
                    fence.wait()?;
                }
                for (output, shift, salt) in outputs {
                    check(&output, shift, salt);
                }
            }
            eprintln!(
                "grouped subpass stream: mapping=[0,0,1], BY_REGION with shared reads, 8 rebound invocations passed"
            );
        }
        drop(device);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and validation layers; records barriers without submitting"]
    fn vulkan_subpass_incoming_buffer_deduplicates_repeated_consumers() -> Result<(), DriverError> {
        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        {
            let mut pool = HashPool::new(&device);
            let mut buffers = Vec::new();
            let cmd_buf = pool.resource(CommandBufferInfo::new(0))?;
            cmd_buf.begin(&vk::CommandBufferBeginInfo::default())?;
            for range_count in [2, 64, 1024u64] {
                let size = range_count * 16;
                let buffer = Arc::new(Buffer::create(
                    &device,
                    BufferInfo::device_mem(
                        size,
                        vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                    ),
                )?);
                for part in 0..range_count {
                    Buffer::swap_access(
                        &buffer,
                        if part % 2 == 0 {
                            AccessType::TransferWrite
                        } else {
                            AccessType::ComputeShaderWrite
                        },
                        part * 16..(part + 1) * 16,
                    )
                    .for_each(drop);
                }
                let mut graph = Graph::new();
                let node = graph.bind_resource(&buffer);
                // Descending prefixes used to emit N * (N + 1) / 2 producer barriers.
                let consumers = (1..=range_count)
                    .rev()
                    .map(|end| (AccessType::FragmentShaderReadOther, 0..end * 16))
                    .chain((0..1024).map(|index| {
                        (
                            AccessType::FragmentShaderReadOther,
                            0..if index % 2 == 0 { size } else { vk::WHOLE_SIZE },
                        )
                    }))
                    .chain([
                        (AccessType::VertexShaderReadUniformBuffer, 0..size),
                        (AccessType::FragmentShaderWrite, 0..16),
                        (AccessType::FragmentShaderReadOther, 0..16),
                        (AccessType::FragmentShaderWrite, 0..16),
                    ]);
                let execs = consumers
                    .map(|(access, range)| {
                        let mut exec = Execution::default();
                        exec.accesses.push(
                            node.index(),
                            SubresourceAccess {
                                access,
                                subresource: SubresourceRange::Buffer(range.into()),
                            },
                        );
                        exec.accesses.freeze();
                        exec
                    })
                    .collect();
                INCOMING_BUFFER_BARRIERS.with_borrow_mut(|barriers| *barriers = Some(Vec::new()));
                Submission::record_image_layout_transitions(
                    &cmd_buf,
                    &mut graph.resources,
                    &mut SubpassFixture::subpass_command(execs),
                    &mut None,
                    &mut None,
                );
                let barriers =
                    INCOMING_BUFFER_BARRIERS.with_borrow_mut(|barriers| barriers.take().unwrap());
                assert_eq!(barriers.len(), (range_count * 2 + 1) as usize);
                for (index, &(src, dst, barrier)) in barriers.iter().enumerate() {
                    let part = index as u64 % range_count;
                    let expected_offset = if index < (range_count * 2) as usize {
                        part * 16
                    } else {
                        0
                    };
                    assert_eq!(barrier.buffer, buffer.handle);
                    assert_eq!((barrier.offset, barrier.size), (expected_offset, 16));
                    assert!(src.contains(vk::PipelineStageFlags::TRANSFER));
                    assert_eq!(
                        barrier.src_access_mask,
                        if expected_offset / 16 % 2 == 0 {
                            vk::AccessFlags::TRANSFER_WRITE
                        } else if device.physical.queue_families[0]
                            .queue_flags
                            .contains(vk::QueueFlags::COMPUTE)
                        {
                            vk::AccessFlags::SHADER_WRITE
                        } else {
                            vk::AccessFlags::MEMORY_WRITE
                        }
                    );
                    assert!(dst.contains(
                        vk::PipelineStageFlags::VERTEX_SHADER
                            | vk::PipelineStageFlags::FRAGMENT_SHADER
                    ));
                    assert_eq!(
                        barrier.dst_access_mask,
                        if index < range_count as usize {
                            vk::AccessFlags::SHADER_READ
                        } else if index < (range_count * 2) as usize {
                            vk::AccessFlags::UNIFORM_READ
                        } else {
                            vk::AccessFlags::SHADER_WRITE
                        }
                    );
                }
                // The duplicate final writer must still replace the intervening reader's state.
                let outgoing = Buffer::swap_access(&buffer, AccessType::TransferWrite, 0..size)
                    .collect::<Vec<_>>();
                assert_eq!(
                    outgoing,
                    [
                        (AccessType::FragmentShaderWrite, (0..16).into()),
                        (AccessType::General, (16..size).into()),
                    ]
                );
                buffers.push(buffer);
            }
            cmd_buf.end()?;
        }
        drop(device);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and validation layers"]
    fn vulkan_subpass_incoming_buffer_previous_graph() -> Result<(), DriverError> {
        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        {
            let mut pool = HashPool::new(&device);
            let fixture = SubpassFixture::new(&device, &mut pool, 4)?;
            let graphics = GraphicsPipeline::create(
                &device,
                GraphicsPipelineInfo::builder().cull_mode(vk::CullModeFlags::NONE),
                [
                    glsl!(kind: vert, r#"
                    #version 450
                    layout(set = 0, binding = 1, std140) uniform Position {
                        uvec4 value;
                    } position;
                    void main() {
                        vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                        gl_Position = vec4(p * 2.0 - 1.0 + float(position.value.w), 0.0, 1.0);
                    }
                "#)
                    .as_slice(),
                    glsl!(kind: frag, r#"
                    #version 450
                    layout(set = 0, binding = 0, std140) uniform Color { uvec4 value; } ubo;
                    layout(set = 1, binding = 0) uniform sampler2D tex;
                    layout(push_constant) uniform State { uint salt; } state;
                    layout(location = 0) out vec4 color;
                    void main() {
                        uvec3 sampled = uvec3(round(texelFetch(tex, ivec2(0), 0).rgb * 255.0));
                        uvec3 rgb = (ubo.value.rgb + sampled
                            + uvec3(state.salt, state.salt * 3, state.salt * 5)) & uvec3(255);
                        color = vec4(vec3(rgb) / 255.0, 1.0);
                    }
                "#)
                    .as_slice(),
                ],
            )?;
            let compute = ComputePipeline::create(
                &device,
                ComputePipelineInfo::default(),
                glsl!(kind: comp, r#"
                #version 450
                layout(local_size_x = 1) in;
                layout(set = 0, binding = 0, std430) writeonly buffer Output {
                    uvec4 value;
                } output_data;
                layout(push_constant) uniform State { uvec4 value; } state;
                void main() { output_data.value = state.value; }
            "#)
                .as_slice(),
            )?;
            let limits = &device.physical.properties_v1_0.limits;
            let stride = limits
                .min_uniform_buffer_offset_alignment
                .max(limits.min_storage_buffer_offset_alignment)
                .max(16);
            for compute_upload in [false, true] {
                let buffer = Arc::new(Buffer::create(
                    &device,
                    BufferInfo::device_mem(
                        stride * 2 + 16,
                        vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::UNIFORM_BUFFER
                            | vk::BufferUsageFlags::INDIRECT_BUFFER
                            | vk::BufferUsageFlags::STORAGE_BUFFER,
                    ),
                )?);
                for index in [258, 519] {
                    let mut upload = Graph::new();
                    let node = upload.bind_resource(&buffer);
                    for (part, offset) in [0, stride, stride * 2].into_iter().enumerate() {
                        let values = if part == 2 {
                            [3, 0, 0, 0]
                        } else {
                            SubpassFixture::uniform_color(index + part)
                        };
                        if compute_upload {
                            upload
                                .begin_cmd()
                                .bind_pipeline(&compute)
                                .shader_subresource_access(
                                    0,
                                    node,
                                    offset..offset + 16,
                                    AccessType::ComputeShaderWrite,
                                )
                                .record_cmd(move |cmd| {
                                    cmd.push_constants(0, bytemuck::cast_slice(&values))
                                        .dispatch(1, 1, 1);
                                });
                        } else {
                            let staging = upload.bind_resource(Buffer::create_from_slice(
                                &device,
                                vk::BufferUsageFlags::TRANSFER_SRC,
                                bytemuck::cast_slice(&values),
                            )?);
                            upload
                                .begin_cmd()
                                .copy_buffer(
                                    staging,
                                    node,
                                    [vk::BufferCopy::default().dst_offset(offset).size(16)],
                                )
                                .end_cmd();
                        }
                    }
                    let mut uploaded = upload.finalize().queue_submit(&mut pool, 0, 0)?;

                    // No wait or semaphore: graphics must synchronize the persisted producer on
                    // the same queue, including a buffer absent from the first draw.
                    let (target, output) = fixture.output(&device)?;
                    let mut graph = Graph::new();
                    let target = graph.bind_resource(&target);
                    let node = graph.bind_resource(&buffer);
                    let texture = graph.bind_resource(&fixture.textures[0]);
                    let uniform = graph.bind_resource(&fixture.uniforms[0]);
                    fixture.draws(
                        &mut graph,
                        target.into(),
                        &[(uniform.into(), texture.into())],
                        false,
                        None,
                        false,
                    );
                    for tile in 1..4 {
                        let offset = if tile == 2 { stride } else { 0 };
                        let mut draw = graph
                            .begin_cmd()
                            .bind_pipeline(if tile == 1 {
                                &fixture.pipelines[0]
                            } else {
                                &graphics
                            })
                            .color_attachment_image(0, target, LoadOp::Load, StoreOp::Store)
                            .shader_subresource_access(
                                (0, 0),
                                node,
                                offset..offset + 16,
                                AccessType::FragmentShaderReadUniformBuffer,
                            )
                            .shader_resource_access(
                                (1, 0),
                                texture,
                                AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                            );
                        if tile > 1 {
                            let vertex_offset = if tile == 2 { 0 } else { stride };
                            draw = draw.shader_subresource_access(
                                (0, 1),
                                node,
                                vertex_offset..vertex_offset + 16,
                                AccessType::VertexShaderReadUniformBuffer,
                            );
                        }
                        if tile == 3 {
                            draw = draw.subresource_access(
                                node,
                                stride * 2..stride * 2 + 16,
                                AccessType::IndirectBuffer,
                            );
                        }
                        draw.record_cmd(move |cmd| {
                            if tile == 3 {
                                // Zero instances preserves the pixel oracle while exercising a
                                // late non-descriptor consumer in synchronization validation.
                                cmd.draw_indirect(node, stride * 2, 1, 16);
                            }
                            SubpassFixture::draw(cmd, tile, 4, 17 + tile as u32);
                        });
                    }
                    SubpassFixture::readback(&mut graph, target.into(), &output);
                    let mut submission = graph.finalize();
                    submission.prepare_command_stream(&mut pool)?;
                    assert_eq!(SubpassFixture::subpass_counts(&submission), [1]);
                    assert_eq!(
                        &*submission.recorded_commands[0].exec_subpasses,
                        &[0, 0, 0, 0]
                    );
                    submission.recorded_commands.clear();
                    INCOMING_BUFFER_BARRIERS
                        .with_borrow_mut(|barriers| *barriers = Some(Vec::new()));
                    submission.queue_submit(&mut pool, 0, 0)?.wait()?;
                    let barriers = INCOMING_BUFFER_BARRIERS
                        .with_borrow_mut(|barriers| barriers.take().unwrap());
                    for (part, offset) in [0, stride, stride * 2].into_iter().enumerate() {
                        let expected_access = if part == 2 {
                            vk::AccessFlags::INDIRECT_COMMAND_READ
                        } else {
                            vk::AccessFlags::UNIFORM_READ
                        };
                        assert!(
                            barriers.iter().any(|(_, _, barrier)| {
                                barrier.buffer == buffer.handle
                                    && barrier.offset == offset
                                    && barrier.size == 16
                                    && barrier.dst_access_mask == expected_access
                            }),
                            "missing incoming barrier for part={part}"
                        );
                    }
                    for &(src, dst, barrier) in &barriers {
                        if barrier.buffer != buffer.handle {
                            continue;
                        }
                        assert!(src.contains(if compute_upload {
                            vk::PipelineStageFlags::COMPUTE_SHADER
                        } else {
                            vk::PipelineStageFlags::TRANSFER
                        }));
                        assert!(dst.contains(
                            vk::PipelineStageFlags::VERTEX_SHADER
                                | vk::PipelineStageFlags::FRAGMENT_SHADER
                                | vk::PipelineStageFlags::DRAW_INDIRECT
                        ));
                        assert_eq!(
                            barrier.src_access_mask,
                            if compute_upload {
                                vk::AccessFlags::SHADER_WRITE
                            } else {
                                vk::AccessFlags::TRANSFER_WRITE
                            },
                            "pre-pass source must not include current-pass readers/General"
                        );
                        assert_eq!(barrier.src_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
                        assert_eq!(barrier.dst_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
                    }
                    uploaded.wait()?;
                    for (pixel, actual) in Buffer::mapped_slice(&output).chunks_exact(4).enumerate()
                    {
                        let tile = pixel % 8 / 2;
                        let uniform = match tile {
                            0 => 0,
                            2 => index + 1,
                            _ => index,
                        };
                        assert_eq!(
                            actual,
                            SubpassFixture::expected(uniform, 0, 17 + tile as u32),
                            "compute_upload={compute_upload} index={index} tile={tile}",
                        );
                    }
                }
            }
        }
        drop(device);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and validation layers"]
    fn vulkan_subpass_incoming_buffer_pure_read_chain_before_overwrite() -> Result<(), DriverError>
    {
        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        {
            let mut pool = HashPool::new(&device);
            let fixture = SubpassFixture::new(&device, &mut pool, 1)?;
            let compute = ComputePipeline::create(
            &device,
            ComputePipelineInfo::default(),
            glsl!(kind: comp, r#"
                #version 450
                layout(local_size_x = 1) in;
                layout(set = 0, binding = 0, std430) readonly buffer Source { uvec4 value; } source;
                layout(set = 0, binding = 1, std430) writeonly buffer Output { uvec4 value; } output_data;
                void main() { output_data.value = source.value; }
            "#)
            .as_slice(),
        )?;
            let graphics = GraphicsPipeline::create(
                &device,
                GraphicsPipelineInfo::builder().cull_mode(vk::CullModeFlags::NONE),
                [
                    glsl!(kind: vert, r#"
                    #version 450
                    layout(set = 0, binding = 0, std140) uniform Color { uvec4 value; } ubo;
                    void main() {
                        vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                        gl_Position = vec4(p * 2.0 - 1.0 + float(ubo.value.w), 0.0, 1.0);
                    }
                "#)
                    .as_slice(),
                    glsl!(kind: frag, r#"
                    #version 450
                    layout(set = 0, binding = 0, std140) uniform Color { uvec4 value; } ubo;
                    layout(set = 1, binding = 0) uniform sampler2D tex;
                    layout(push_constant) uniform State { uint salt; } state;
                    layout(location = 0) out vec4 color;
                    void main() {
                        uvec3 sampled = uvec3(round(texelFetch(tex, ivec2(0), 0).rgb * 255.0));
                        uvec3 rgb = (ubo.value.rgb + sampled
                            + uvec3(state.salt, state.salt * 3, state.salt * 5)) & uvec3(255);
                        color = vec4(vec3(rgb) / 255.0, 1.0);
                    }
                "#)
                    .as_slice(),
                ],
            )?;
            let values = SubpassFixture::uniform_color(258);
            let replacement = SubpassFixture::uniform_color(519);
            let buffer = Arc::new(Buffer::create_from_slice(
                &device,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER,
                bytemuck::cast_slice(&values),
            )?);
            assert!(
                buffer
                    .sync_info()
                    .ranges
                    .iter()
                    .all(|range| range.access_mask.is_empty())
            );
            let computed = Arc::new(Buffer::create(
                &device,
                BufferInfo::host_mem(16, vk::BufferUsageFlags::STORAGE_BUFFER),
            )?);
            let (target, pixels) = fixture.output(&device)?;

            let mut graph_a = Graph::new();
            let source = graph_a.bind_resource(&buffer);
            let output = graph_a.bind_resource(&computed);
            graph_a
                .begin_cmd()
                .bind_pipeline(&compute)
                .shader_resource_access(0, source, AccessType::ComputeShaderReadOther)
                .shader_resource_access(1, output, AccessType::ComputeShaderWrite)
                .record_cmd(|cmd| {
                    cmd.dispatch(1, 1, 1);
                });
            let mut read = graph_a.finalize().queue_submit(&mut pool, 0, 0)?;
            let history = buffer.sync_info();
            assert_eq!(history.ranges.len(), 1);
            assert_eq!(
                history.ranges[0].stage_mask,
                vk::PipelineStageFlags::COMPUTE_SHADER
            );
            assert_eq!(history.ranges[0].access_mask, vk::AccessFlags::SHADER_READ);

            // No GPU writer has touched source, so General cannot accidentally hide lost readers.
            let mut graph_b = Graph::new();
            let source = graph_b.bind_resource(&buffer);
            let texture = graph_b.bind_resource(&fixture.textures[0]);
            let target = graph_b.bind_resource(&target);
            graph_b
                .begin_cmd()
                .bind_pipeline(&graphics)
                .color_attachment_image(0, target, LoadOp::CLEAR_BLACK_ALPHA_ZERO, StoreOp::Store)
                .shader_resource_access((0, 0), source, AccessType::VertexShaderReadUniformBuffer)
                .resource_access(source, AccessType::FragmentShaderReadUniformBuffer)
                .shader_resource_access(
                    (1, 0),
                    texture,
                    AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                )
                .record_cmd(|cmd| {
                    SubpassFixture::draw(cmd, 0, 1, 17);
                });
            SubpassFixture::readback(&mut graph_b, target.into(), &pixels);
            INCOMING_BUFFER_BARRIERS.with_borrow_mut(|barriers| *barriers = Some(Vec::new()));
            let mut drawn = graph_b.finalize().queue_submit(&mut pool, 0, 0)?;
            let barriers =
                INCOMING_BUFFER_BARRIERS.with_borrow_mut(|barriers| barriers.take().unwrap());
            let incoming = barriers
                .iter()
                .filter(|(_, _, barrier)| barrier.buffer == buffer.handle)
                .collect::<Vec<_>>();
            assert!(
                !incoming.is_empty(),
                "missing compute-read -> graphics-read execution edge"
            );
            for &&(src, dst, barrier) in &incoming {
                assert!(src.contains(vk::PipelineStageFlags::COMPUTE_SHADER));
                assert!(dst.contains(
                    vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER
                ));
                assert!(barrier.src_access_mask.is_empty());
                assert!(barrier.dst_access_mask.is_empty());
            }
            let history = buffer.sync_info();
            assert_eq!(history.ranges.len(), 1);
            assert_eq!(
                history.ranges[0].stage_mask,
                vk::PipelineStageFlags::ALL_COMMANDS
            );
            assert_eq!(history.ranges[0].access_mask, vk::AccessFlags::SHADER_READ);

            // The ordinary non-graphics recorder consumes this broad reader state, producing
            // ALL_COMMANDS -> TRANSFER with empty access masks (also checked by the range oracle).
            let mut graph_c = Graph::new();
            let source = graph_c.bind_resource(Buffer::create_from_slice(
                &device,
                vk::BufferUsageFlags::TRANSFER_SRC,
                bytemuck::cast_slice(&replacement),
            )?);
            let destination = graph_c.bind_resource(&buffer);
            graph_c.copy_buffer(source, destination);
            let computed_node = graph_c.bind_resource(&computed);
            graph_c
                .begin_cmd()
                .resource_access(computed_node, AccessType::HostRead)
                .resource_access(destination, AccessType::HostRead)
                .record_cmd(|_| {});
            // Wait only after all three graphs have been submitted on the same queue.
            graph_c.finalize().queue_submit(&mut pool, 0, 0)?.wait()?;
            read.wait()?;
            drawn.wait()?;
            assert_eq!(
                Buffer::mapped_slice(&computed),
                bytemuck::cast_slice::<_, u8>(&values)
            );
            assert_eq!(
                Buffer::mapped_slice(&pixels),
                SubpassFixture::expected(258, 0, 17).repeat(4)
            );
            assert_eq!(
                Buffer::mapped_slice(&buffer),
                bytemuck::cast_slice::<_, u8>(&replacement)
            );
        }
        drop(device);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and validation layers; records barriers without submitting"]
    fn vulkan_subpass_incoming_buffer_range_oracle_and_reader_ordering() -> Result<(), DriverError>
    {
        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        {
            let mut pool = HashPool::new(&device);
            let buffer = Arc::new(Buffer::create(
                &device,
                BufferInfo::device_mem(
                    64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                ),
            )?);
            let mut graph = Graph::new();
            let node = graph.bind_resource(&buffer);
            let cmd_buf = pool.resource(CommandBufferInfo::new(0))?;
            cmd_buf.begin(&vk::CommandBufferBeginInfo::default())?;
            for initial in [
                [AccessType::Nothing; 4],
                [AccessType::FragmentShaderReadUniformBuffer; 4],
                [
                    AccessType::TransferWrite,
                    AccessType::FragmentShaderReadUniformBuffer,
                    AccessType::ComputeShaderWrite,
                    AccessType::Nothing,
                ],
            ] {
                for writes in [false, true] {
                    for (part, access) in initial.into_iter().enumerate() {
                        Buffer::swap_access(
                            &buffer,
                            access,
                            part as u64 * 16..(part as u64 + 1) * 16,
                        )
                        .for_each(drop);
                    }
                    let mut expected = std::collections::BTreeMap::new();
                    let mut consumers = std::collections::HashSet::new();
                    let mut execs = Vec::new();
                    for index in 0..100u64 {
                        // Reverse-order slices, then bridges across both cached intervals and gaps.
                        let range = if index < 8 {
                            (7 - index) * 8..(7 - index) * 8 + 4
                        } else if index >= 98 {
                            0..vk::WHOLE_SIZE
                        } else {
                            let start = (index * 17) % 64;
                            start..(start + 1 + index % 23).min(64)
                        };
                        // Finish with a vertex reader after a fragment reader. Tracking only
                        // that last stage would not cover the earlier fragment work on overwrite.
                        let access = if index == 98 {
                            AccessType::FragmentShaderReadUniformBuffer
                        } else if index == 99 {
                            AccessType::VertexShaderReadUniformBuffer
                        } else if writes && index % 3 == 0 {
                            AccessType::FragmentShaderWrite
                        } else if index % 5 == 0 {
                            AccessType::FragmentShaderReadOther
                        } else if index % 2 == 0 {
                            AccessType::VertexShaderReadUniformBuffer
                        } else {
                            AccessType::FragmentShaderReadUniformBuffer
                        };
                        let mut exec = Execution::default();
                        exec.accesses.push(
                            node.index(),
                            SubresourceAccess {
                                access,
                                subresource: SubresourceRange::Buffer(range.clone().into()),
                            },
                        );
                        exec.accesses.freeze();
                        execs.push(exec);
                        for byte in range.start..range.end.min(64) {
                            if !consumers.insert((byte, std::mem::discriminant(&access))) {
                                continue;
                            }
                            let producer = initial[byte as usize / 16];
                            let writer = crate::driver::is_write_access(producer);
                            if producer == AccessType::Nothing
                                && !crate::driver::is_write_access(access)
                            {
                                continue;
                            }
                            let src = if producer == AccessType::ComputeShaderWrite
                                && !device.physical.queue_families[0]
                                    .queue_flags
                                    .contains(vk::QueueFlags::COMPUTE)
                            {
                                vk::AccessFlags::MEMORY_WRITE
                            } else if writer {
                                crate::driver::pipeline_stage_access_flags(producer).1
                            } else {
                                vk::AccessFlags::empty()
                            };
                            let dst = if writer {
                                crate::driver::pipeline_stage_access_flags(access).1
                            } else {
                                vk::AccessFlags::empty()
                            };
                            *expected
                                .entry((byte, src.as_raw(), dst.as_raw()))
                                .or_insert(0usize) += 1;
                        }
                    }
                    INCOMING_BUFFER_BARRIERS
                        .with_borrow_mut(|barriers| *barriers = Some(Vec::new()));
                    Submission::record_image_layout_transitions(
                        &cmd_buf,
                        &mut graph.resources,
                        &mut SubpassFixture::subpass_command(execs),
                        &mut None,
                        &mut None,
                    );
                    let barriers = INCOMING_BUFFER_BARRIERS
                        .with_borrow_mut(|barriers| barriers.take().unwrap());
                    let mut actual = std::collections::BTreeMap::new();
                    for (_, dst, barrier) in barriers {
                        if initial.iter().any(|&access| access != AccessType::Nothing) {
                            assert!(dst.contains(
                                vk::PipelineStageFlags::VERTEX_SHADER
                                    | vk::PipelineStageFlags::FRAGMENT_SHADER
                            ));
                        }
                        for byte in barrier.offset..barrier.offset + barrier.size {
                            *actual
                                .entry((
                                    byte,
                                    barrier.src_access_mask.as_raw(),
                                    barrier.dst_access_mask.as_raw(),
                                ))
                                .or_insert(0usize) += 1;
                        }
                    }
                    assert_eq!(actual, expected, "initial={initial:?}, writes={writes}");

                    if !writes
                        && initial
                            .iter()
                            .all(|&access| !crate::driver::is_write_access(access))
                    {
                        // All bytes were read at both vertex and fragment stages. Consume the
                        // actual outgoing tracker with the same swap and mapper used by graph C.
                        let outgoing = Buffer::swap_accesses(
                            &buffer,
                            [(AccessType::TransferWrite, (0..64).into())],
                        )
                        .collect::<Vec<_>>();
                        assert_eq!(outgoing.len(), 1);
                        let (next, previous, range) = outgoing[0];
                        assert_eq!(previous, AccessType::AnyShaderReadOther);
                        let (src, dst, barrier) = super::Submission::buffer_memory_barrier(
                            &BufferBarrier {
                                previous_accesses: std::slice::from_ref(&previous),
                                next_accesses: std::slice::from_ref(&next),
                                buffer: buffer.handle,
                                offset: range.start as _,
                                size: (range.end - range.start) as _,
                                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            },
                            device.physical.queue_families[0].queue_flags,
                        );
                        assert_eq!(src, vk::PipelineStageFlags::ALL_COMMANDS);
                        assert_eq!(dst, vk::PipelineStageFlags::TRANSFER);
                        assert!(barrier.src_access_mask.is_empty());
                        assert!(barrier.dst_access_mask.is_empty());
                    }
                }
            }
            // Acquires need visibility, while non-transferred readers retain execution-only edges.
            if device.physical.queue_families.len() > 1 {
                for access in [
                    AccessType::Nothing,
                    AccessType::FragmentShaderReadUniformBuffer,
                ] {
                    Buffer::swap_access(&buffer, access, 0..64).for_each(drop);
                    let mut pending = super::PendingTransferNodes::new(graph.resources.len());
                    pending.push_transfer(
                        node.index(),
                        buffer.handle,
                        BufferQueueOwnershipTransfer {
                            range: (16..32).into(),
                            src_queue_family_index: 1,
                            dst_queue_family_index: 0,
                        },
                    );
                    let mut pending = Some(pending);
                    let execs = [8..24, 0..64, 0..32, 0..vk::WHOLE_SIZE]
                        .into_iter()
                        .map(|range| {
                            let mut exec = Execution::default();
                            exec.accesses.push(
                                node.index(),
                                SubresourceAccess {
                                    access: AccessType::VertexShaderReadUniformBuffer,
                                    subresource: SubresourceRange::Buffer(range.into()),
                                },
                            );
                            exec.accesses.freeze();
                            exec
                        })
                        .collect();
                    INCOMING_BUFFER_BARRIERS
                        .with_borrow_mut(|barriers| *barriers = Some(Vec::new()));
                    Submission::record_image_layout_transitions(
                        &cmd_buf,
                        &mut graph.resources,
                        &mut SubpassFixture::subpass_command(execs),
                        &mut pending,
                        &mut None,
                    );
                    let barriers = INCOMING_BUFFER_BARRIERS
                        .with_borrow_mut(|barriers| barriers.take().unwrap());
                    assert!(pending.is_none());
                    assert_eq!(
                        barriers.len(),
                        if access == AccessType::Nothing { 2 } else { 5 }
                    );
                    let acquired = barriers
                        .iter()
                        .filter(|(_, _, barrier)| barrier.src_queue_family_index == 1)
                        .flat_map(|(_, _, barrier)| barrier.offset..barrier.offset + barrier.size)
                        .collect::<Vec<_>>();
                    // Both halves of the transfer must be acquired exactly once, then consumed.
                    assert_eq!(acquired, (16..32).collect::<Vec<_>>());
                    for (src, dst, barrier) in barriers {
                        if barrier.src_queue_family_index == vk::QUEUE_FAMILY_IGNORED {
                            assert!(src.contains(vk::PipelineStageFlags::FRAGMENT_SHADER));
                            assert!(dst.contains(vk::PipelineStageFlags::VERTEX_SHADER));
                            assert!(barrier.src_access_mask.is_empty());
                            assert!(barrier.dst_access_mask.is_empty());
                            continue;
                        }
                        assert_eq!(
                            (
                                barrier.src_queue_family_index,
                                barrier.dst_queue_family_index
                            ),
                            (1, 0)
                        );
                        assert!(barrier.src_access_mask.is_empty());
                        assert_eq!(barrier.dst_access_mask, vk::AccessFlags::UNIFORM_READ);
                    }
                }
            }
            cmd_buf.end()?;
        }
        drop(device);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and synchronization validation layers"]
    fn vulkan_subpass_storage_image_reader_scope_survives_next_graph_overwrite()
    -> Result<(), DriverError> {
        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        {
            let mut pool = HashPool::new(&device);
            let fixture = SubpassFixture::new(&device, &mut pool, 4)?;
            let vertex = glsl!(kind: vert, r#"
                #version 450
                void main() {
                    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
                }
            "#);
            let fragment_reader = glsl!(kind: frag, r#"
                #version 450
                layout(set = 0, binding = 0, rgba8) uniform readonly image2D source;
                layout(push_constant) uniform State { uint salt; } state;
                layout(location = 0) out vec4 color;
                void main() { color = imageLoad(source, ivec2(state.salt)); }
            "#);
            let vertex_reader = glsl!(kind: vert, r#"
                #version 450
                layout(set = 0, binding = 0, rgba8) uniform readonly image2D source;
                layout(push_constant) uniform State { uint salt; } state;
                layout(location = 0) flat out vec4 sampled;
                void main() {
                    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
                    sampled = imageLoad(source, ivec2(state.salt));
                }
            "#);
            let fragment = glsl!(kind: frag, r#"
                #version 450
                layout(location = 0) flat in vec4 sampled;
                layout(location = 0) out vec4 color;
                void main() { color = sampled; }
            "#);
            let pipelines = [
                GraphicsPipeline::create(
                    &device,
                    GraphicsPipelineInfo::builder().cull_mode(vk::CullModeFlags::NONE),
                    [vertex.as_slice(), fragment_reader.as_slice()],
                )?,
                GraphicsPipeline::create(
                    &device,
                    GraphicsPipelineInfo::builder().cull_mode(vk::CullModeFlags::NONE),
                    [vertex_reader.as_slice(), fragment.as_slice()],
                )?,
            ];
            let overwrite = ComputePipeline::create(
                &device,
                ComputePipelineInfo::default(),
                glsl!(kind: comp, r#"
                    #version 450
                    layout(local_size_x = 1) in;
                    layout(set = 0, binding = 0, rgba8) uniform writeonly image2D target;
                    void main() {
                        imageStore(target, ivec2(0), vec4(211.0, 31.0, 7.0, 255.0) / 255.0);
                    }
                "#)
                .as_slice(),
            )?;
            let storage = Arc::new(Image::create(
                &device,
                ImageInfo::image_2d(
                    1,
                    1,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                ),
            )?);
            let original = [17u8, 61, 129, 255];
            let mut upload = Graph::new();
            let staging = upload.bind_resource(Buffer::create_from_slice(
                &device,
                vk::BufferUsageFlags::TRANSFER_SRC,
                &original,
            )?);
            let storage_node = upload.bind_resource(&storage);
            upload.copy_buffer_to_image(staging, storage_node);
            upload.finalize().queue_submit(&mut pool, 0, 0)?.wait()?;

            let (target, pixels) = fixture.output(&device)?;
            let overwritten = Arc::new(Buffer::create(
                &device,
                BufferInfo::host_mem(4, vk::BufferUsageFlags::TRANSFER_DST),
            )?);
            let mut graph = Graph::new();
            let target_node = graph.bind_resource(&target);
            let storage_node = graph.bind_resource(&storage);
            for (tile, access) in [
                AccessType::AnyShaderReadOther,
                AccessType::AnyShaderReadOther,
                AccessType::VertexShaderReadOther,
                AccessType::VertexShaderReadOther,
            ]
            .into_iter()
            .enumerate()
            {
                // The broad declarations really read in the fragment shader; later executions
                // narrow tracking to vertex, but cannot discard fragment ordering.
                graph
                    .begin_cmd()
                    .bind_pipeline(&pipelines[tile / 2])
                    .color_attachment_image(
                        0,
                        target_node,
                        if tile == 0 {
                            LoadOp::CLEAR_BLACK_ALPHA_ZERO
                        } else {
                            LoadOp::Load
                        },
                        StoreOp::Store,
                    )
                    .shader_resource_access(0, storage_node, access)
                    .record_cmd(move |cmd| SubpassFixture::draw(cmd, tile, 4, 0));
            }
            let mut submission = graph.finalize();
            submission.prepare_command_stream(&mut pool)?;
            assert_eq!(SubpassFixture::subpass_counts(&submission), [4]);
            let recording = &submission.recorded_commands[0];
            assert_eq!(&*recording.exec_subpasses, &[0, 1, 2, 3]);
            let dependencies = &recording.render_pass.as_ref().unwrap().info.dependencies;
            // Retiring the first reader leaves only the independent attachment-local edge.
            let retired = dependencies
                .iter()
                .find(|edge| edge.src_subpass == 0 && edge.dst_subpass == 2)
                .unwrap();
            assert_eq!(retired.dependency_flags, vk::DependencyFlags::BY_REGION);
            assert!(
                !retired
                    .src_access_mask
                    .intersects(vk::AccessFlags::SHADER_READ)
            );
            assert!(
                !retired
                    .dst_access_mask
                    .intersects(vk::AccessFlags::SHADER_READ)
            );
            for src in 0..3 {
                let edge = dependencies
                    .iter()
                    .find(|edge| edge.src_subpass == src && edge.dst_subpass == src + 1)
                    .expect("missing adjacent storage image reader dependency");
                assert!(edge.dependency_flags.is_empty());
            }
            let dependency = dependencies
                .iter()
                .find(|edge| edge.src_subpass == 1 && edge.dst_subpass == 2)
                .expect("missing storage image reader execution chain");
            assert!(dependency.dependency_flags.is_empty());
            assert!(
                dependency
                    .src_stage_mask
                    .contains(vk::PipelineStageFlags::ALL_GRAPHICS)
            );
            assert!(
                dependency
                    .dst_stage_mask
                    .contains(vk::PipelineStageFlags::VERTEX_SHADER)
            );
            assert!(
                dependency
                    .src_access_mask
                    .contains(vk::AccessFlags::SHADER_READ)
            );
            assert!(
                dependency
                    .dst_access_mask
                    .contains(vk::AccessFlags::SHADER_READ)
            );
            submission.recorded_commands.clear();
            let mut drawn = submission.queue_submit(&mut pool, 0, 0)?;
            let sync = storage.sync_info();
            assert_eq!(sync.subresources.len(), 1);
            assert_eq!(sync.subresources[0].layout, Some(vk::ImageLayout::GENERAL));
            assert_eq!(
                sync.subresources[0].stage_mask,
                vk::PipelineStageFlags::VERTEX_SHADER
            );
            assert_eq!(
                sync.subresources[0].access_mask,
                vk::AccessFlags::SHADER_READ
            );

            // No wait, semaphore, or target readback before the overwrite: its incoming
            // vertex scope relies on the retained fragment-to-vertex subpass dependency.
            let mut next = Graph::new();
            let storage_node = next.bind_resource(&storage);
            next.begin_cmd()
                .bind_pipeline(&overwrite)
                .shader_resource_access(0, storage_node, AccessType::ComputeShaderWrite)
                .record_cmd(|cmd| {
                    cmd.dispatch(1, 1, 1);
                });
            SubpassFixture::readback(&mut next, storage_node.into(), &overwritten);
            next.finalize().queue_submit(&mut pool, 0, 0)?.wait()?;
            drawn.wait()?;
            let mut readback = Graph::new();
            let target_node = readback.bind_resource(&target);
            SubpassFixture::readback(&mut readback, target_node.into(), &pixels);
            readback.finalize().queue_submit(&mut pool, 0, 0)?.wait()?;
            assert_eq!(
                Buffer::mapped_slice(&pixels),
                original.repeat(16),
                "both fragment and vertex readers must see the original storage image"
            );
            assert_eq!(
                Buffer::mapped_slice(&overwritten),
                &[211, 31, 7, 255],
                "the subsequent compute overwrite must be visible to readback"
            );
        }
        drop(device);
        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device and validation layers"]
    fn vulkan_subpass_stream_slots_rebind() -> Result<(), DriverError> {
        TestDevice::init_validation_test_logging();
        let device = TestDevice::new_debug()?;
        {
            let mut pool = HashPool::new(&device);
            let fixture = SubpassFixture::new(&device, &mut pool, 8)?;
            for prepared in [false, true] {
                let draft = fixture.draft();
                let stream = if prepared {
                    draft.prepare(&mut pool)?
                } else {
                    draft.into_stream()
                };
                if prepared {
                    let submission = stream.inner.submission.lock().unwrap();
                    assert_eq!(SubpassFixture::subpass_counts(&submission), [1]);
                }
                for round in 0..2 {
                    let mut pending = Vec::new();
                    let mut outputs = Vec::new();
                    for frame in 0..2 {
                        let mut graph = Graph::new();
                        for invocation in 0..2 {
                            let shift = round * 4 + frame * 2 + invocation;
                            let salt = shift as u32 + 31;
                            let (target, output) = fixture.output(&device)?;
                            fixture.invoke(&mut graph, &stream, &target, shift, salt);
                            let target = graph.bind_resource(&target);
                            SubpassFixture::readback(&mut graph, target.into(), &output);
                            outputs.push((output, shift, salt));
                        }
                        pending.push(graph.finalize().queue_submit(&mut pool, 0, 0)?);
                    }
                    // Two live submissions, two invocations each; the next round reuses their slots.
                    for fence in &mut pending {
                        fence.wait()?;
                    }
                    for (output, shift, salt) in outputs {
                        fixture.check(&output, shift, salt);
                    }
                }
                eprintln!("subpass streams: prepared={prepared} 8 invocations, all tiles passed");
            }
        }
        drop(device);
        Ok(())
    }

    #[test]
    fn whole_resource_canonical_accesses_preserves_mixed_slice_accesses() {
        let accesses = [
            SubresourceAccess {
                access: AccessType::AccelerationStructureBuildRead,
                subresource: SubresourceRange::AccelerationStructure,
            },
            SubresourceAccess {
                access: AccessType::RayTracingShaderReadAccelerationStructure,
                subresource: SubresourceRange::AccelerationStructure,
            },
        ];

        let mut scratch = Vec::new();
        assert_eq!(
            Submission::whole_resource_canonical_accesses(&accesses, &mut scratch),
            &[
                AccessType::AccelerationStructureBuildRead,
                AccessType::RayTracingShaderReadAccelerationStructure,
            ],
            "mixed acceleration-structure slices should preserve all accesses for next-state tracking"
        );
    }

    mod test_device_lifecycle {
        use {
            super::{DriverError, TestDevice},
            std::{
                cell::Cell,
                panic::{AssertUnwindSafe, catch_unwind},
                sync::{
                    Mutex, TryLockError,
                    atomic::{AtomicUsize, Ordering},
                    mpsc,
                },
                thread,
            },
        };

        struct OnDrop<F: FnMut()>(F);

        impl<F: FnMut()> Drop for OnDrop<F> {
            fn drop(&mut self) {
                (self.0)();
            }
        }

        #[test]
        fn contending_session_does_not_inherit_validation_errors() {
            let lock = Mutex::new(());
            let errors = AtomicUsize::new(0);
            let first = TestDevice::create(
                lock.lock().unwrap(),
                Some(|| errors.load(Ordering::Relaxed)),
                || {
                    Ok(OnDrop(|| {
                        errors.fetch_add(1, Ordering::Relaxed);
                    }))
                },
            )
            .unwrap();
            let (ready_tx, ready_rx) = mpsc::channel();
            thread::scope(|scope| {
                let second = scope.spawn(|| {
                    assert!(matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
                    ready_tx.send(()).unwrap();
                    let device = TestDevice::create(
                        lock.lock().unwrap(),
                        Some(|| errors.load(Ordering::Relaxed)),
                        || {
                            assert_eq!(errors.load(Ordering::Relaxed), 1);
                            Ok(())
                        },
                    )
                    .unwrap();
                    drop(device);
                });
                ready_rx.recv().unwrap();
                assert!(catch_unwind(AssertUnwindSafe(|| drop(first))).is_err());
                second.join().unwrap();
            });
            assert!(lock.try_lock().is_ok());
        }

        #[test]
        fn creation_failure_and_unwind_release_and_drop_once() {
            let lock = Mutex::new(());
            let errors = Cell::new(0);
            let result = TestDevice::create(
                lock.lock().unwrap(),
                Some(|| {
                    assert!(matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
                    errors.get()
                }),
                || {
                    assert!(matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
                    errors.set(1);
                    Err::<(), _>(DriverError::Unsupported)
                },
            );
            assert!(matches!(result, Err(DriverError::Unsupported)));
            assert!(lock.try_lock().is_ok());
            drop(
                TestDevice::create(lock.lock().unwrap(), Some(|| errors.get()), || Ok(())).unwrap(),
            );

            for panic_in_drop in [false, true] {
                let lock = Mutex::new(());
                let errors = Cell::new(0);
                let drops = Cell::new(0);
                let result = catch_unwind(AssertUnwindSafe(|| {
                    let device =
                        TestDevice::create(lock.lock().unwrap(), Some(|| errors.get()), || {
                            Ok(OnDrop(|| {
                                drops.set(drops.get() + 1);
                                errors.set(1);
                                if panic_in_drop {
                                    panic!("device teardown panic");
                                }
                            }))
                        })
                        .unwrap();
                    if !panic_in_drop {
                        panic!("test body panic");
                    }
                    drop(device);
                }));
                assert_eq!(drops.get(), 1);
                assert_eq!(
                    *result.unwrap_err().downcast::<&str>().unwrap(),
                    if panic_in_drop {
                        "device teardown panic"
                    } else {
                        "test body panic"
                    }
                );
                // Existing panics still poison the mutex, but must not retain its guard.
                assert!(matches!(lock.try_lock(), Err(TryLockError::Poisoned(_))));
            }
        }

        #[test]
        fn validation_failures_are_scoped_through_teardown() {
            // No error, creation error, execution error, and teardown error.
            for phase in 0..4 {
                let lock = Mutex::new(());
                let errors = Cell::new(7);
                let snapshots = Cell::new(0);
                let drops = Cell::new(0);
                let result = catch_unwind(AssertUnwindSafe(|| {
                    let device = TestDevice::create(
                        lock.lock().unwrap(),
                        Some(|| {
                            assert!(matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
                            snapshots.set(snapshots.get() + 1);
                            errors.get()
                        }),
                        || {
                            assert_eq!(snapshots.get(), 1, "snapshot must precede creation");
                            assert!(matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
                            errors.set(errors.get() + usize::from(phase == 1));
                            Ok(OnDrop(|| {
                                assert!(matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
                                drops.set(drops.get() + 1);
                                errors.set(errors.get() + usize::from(phase == 3));
                            }))
                        },
                    )
                    .unwrap();
                    errors.set(errors.get() + usize::from(phase == 2));
                    if phase == 0 {
                        return; // Successful early returns must also finish the session.
                    }
                    drop(device);
                }));
                assert_eq!(drops.get(), 1);
                assert_eq!(snapshots.get(), 3);
                assert_eq!(result.is_err(), phase != 0);
                if let Err(error) = result {
                    let message = error.downcast::<String>().unwrap();
                    assert_eq!(message.contains("during teardown"), phase == 3);
                }
                assert!(
                    !lock.is_poisoned(),
                    "validation assertions must release the guard first"
                );
                // An earlier session's error is not a failure in this clean session.
                drop(
                    TestDevice::create(lock.lock().unwrap(), Some(|| errors.get()), || Ok(()))
                        .unwrap(),
                );
            }
        }
    }
}
