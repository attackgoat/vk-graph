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
        Node, NodeIndex, TimestampQueryData, TimestampQueryPlacement,
        cmd::{SubresourceAccess, SubresourceRange},
    },
    crate::{
        StoreOp, TimestampQuery,
        cmd::CommandRef,
        driver::{
            AttachmentInfo, AttachmentRef, Descriptor, DescriptorInfo, DescriptorSet, DriverError,
            FramebufferAttachmentImageInfo, FramebufferInfo, SharingMode, SubpassDependency,
            SubpassInfo,
            accel_struct::AccelerationStructure,
            buffer::{Buffer, BufferSubresourceRange},
            cmd_buf::{CommandBuffer, CommandBufferInfo},
            descriptor_set::{DescriptorPool, DescriptorPoolInfo},
            device::Device,
            fence::{Fence, FenceDroppable},
            format_aspect_mask,
            graphics::{DepthStencilInfo, GraphicsPipeline},
            image::{
                DenseMap, Image, ImageInfo, image_subresource_range_contains,
                image_subresource_range_intersection,
            },
            initial_image_layout_access, is_read_access,
            physical_device::Vulkan10Limits,
            pipeline_stage_access_flags,
            query_pool::{QueryPool, QueryPoolInfo},
            render_pass::{RenderPass, RenderPassInfo},
        },
        lazy_str,
        node::AnyNode,
        pool::{Lease, Pool, SubmissionPool},
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
        collections::{BTreeMap, BTreeSet, HashMap},
        iter::repeat_n,
        mem::take,
        ops::Range,
        slice,
        sync::{Arc, Mutex},
        time::Duration,
    },
    vk_sync::{
        AccessType, BufferBarrier, GlobalBarrier, ImageBarrier, ImageLayout,
        get_buffer_memory_barrier, get_image_memory_barrier, get_memory_barrier,
    },
};

#[cfg(feature = "checked")]
use super::GraphId;

#[cfg(not(feature = "checked"))]
use std::hint::unreachable_unchecked;

thread_local! {
    static SUBMIT: RefCell<SubmitScratch> = Default::default();
}

fn aspect_mask_for_span(base_aspect: u32, start: u32, end: u32) -> vk::ImageAspectFlags {
    let mut mask = vk::ImageAspectFlags::empty();

    for ordinal in start..end {
        mask |= vk::ImageAspectFlags::from_raw(1 << (base_aspect + ordinal));
    }

    mask
}

fn buffer_barriers_from_transfers<'a>(
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
                    src_queue_family_index: transfer.map_or(vk::QUEUE_FAMILY_IGNORED, |transfer| {
                        transfer.src_queue_family_index
                    }),
                    dst_queue_family_index: transfer.map_or(vk::QUEUE_FAMILY_IGNORED, |transfer| {
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

fn buffer_subresource_range_intersects(
    lhs: BufferSubresourceRange,
    rhs: BufferSubresourceRange,
) -> bool {
    lhs.start < rhs.end && lhs.end > rhs.start
}

fn check_queue_submit_args(
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

fn check_queue_submit2_args(
    device: &Device,
    waits: &[SemaphoreSubmit2Info],
    signals: &[SemaphoreSubmit2Info],
) -> Result<(), DriverError> {
    if !device.physical.vk_khr_synchronization2 {
        return Err(DriverError::Unsupported);
    }

    if (waits.iter().any(|wait| wait.value != 0) || signals.iter().any(|signal| signal.value != 0))
        && !supports_timeline_semaphores(device)
    {
        return Err(DriverError::Unsupported);
    }

    Ok(())
}

fn consume_pending_buffer_transfers(
    transfers: &mut Vec<BufferQueueOwnershipTransfer>,
    range: BufferSubresourceRange,
) -> bool {
    transfers.retain(|transfer| !buffer_subresource_range_intersects(transfer.range, range));
    transfers.is_empty()
}

fn consume_pending_image_transfers(
    transfers: &mut Vec<ImageQueueOwnershipTransfer>,
    range: vk::ImageSubresourceRange,
) -> bool {
    transfers
        .retain(|transfer| image_subresource_range_intersection(transfer.range, range).is_none());
    transfers.is_empty()
}

fn exclusive_transfer_source(sharing: SharingMode, queue_family_index: u32) -> Option<(u32, u32)> {
    let SharingMode::Exclusive(Some((src_queue_family_index, src_queue_index))) = sharing else {
        return None;
    };

    (src_queue_family_index != queue_family_index)
        .then_some((src_queue_family_index, src_queue_index))
}

const fn image_access_layout(access: AccessType) -> ImageLayout {
    if matches!(access, AccessType::Present | AccessType::ComputeShaderWrite) {
        ImageLayout::General
    } else {
        ImageLayout::Optimal
    }
}

fn image_barriers_from_transfers<'a>(
    image: vk::Image,
    prev_access: &'a AccessType,
    next_access: &'a AccessType,
    range: vk::ImageSubresourceRange,
    transfers: &'a [ImageQueueOwnershipTransfer],
    discard_contents: bool,
) -> impl Iterator<Item = ImageBarrier<'a>> + 'a {
    image_barrier_transfer_ranges(transfers, range).map(move |(range, transfer)| {
        trace!(
            "    image {:?} {:?} {:?}->{:?}",
            image,
            ImageSubresourceRangeDebug(range),
            prev_access,
            next_access,
        );

        ImageBarrier {
            next_accesses: slice::from_ref(next_access),
            next_layout: image_access_layout(*next_access),
            previous_accesses: slice::from_ref(prev_access),
            previous_layout: image_access_layout(*prev_access),
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
    })
}

fn image_barrier_transfer_ranges<'a>(
    transfers: &'a [ImageQueueOwnershipTransfer],
    range: vk::ImageSubresourceRange,
) -> impl Iterator<
    Item = (
        vk::ImageSubresourceRange,
        Option<&'a ImageQueueOwnershipTransfer>,
    ),
> + 'a {
    thread_local! {
        static IMAGE_TRANSFER: RefCell<ImageTransferScratch> = Default::default();
    }

    #[derive(Default)]
    struct ImageTransferScratch {
        overlaps: Vec<(usize, vk::ImageSubresourceRange)>,
        aspect_cuts: Vec<u32>,
        layer_cuts: Vec<u32>,
        mip_cuts: Vec<u32>,
    }

    struct ImageBarrierTransferIter<'a> {
        transfers: &'a [ImageQueueOwnershipTransfer],
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

    impl<'a> Iterator for ImageBarrierTransferIter<'a> {
        type Item = (
            vk::ImageSubresourceRange,
            Option<&'a ImageQueueOwnershipTransfer>,
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

                let aspect_mask = aspect_mask_for_span(self.base_aspect, aspect_start, aspect_end);

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

fn image_execution_discard_contents(prev_access: AccessType) -> bool {
    prev_access == AccessType::Nothing
}

fn image_layout_transition_discard_contents(
    prev_access: AccessType,
    next_access: AccessType,
) -> bool {
    // Read/modify/write accesses must preserve the existing image contents
    // Check for "not-read" here because some accesses both read and write
    // Color Attachment Read/Write (blending) will prevent discarding contents
    prev_access == AccessType::Nothing || !is_read_access(next_access)
}

fn image_subresource_range_eq(
    lhs: vk::ImageSubresourceRange,
    rhs: vk::ImageSubresourceRange,
) -> bool {
    lhs.aspect_mask == rhs.aspect_mask
        && lhs.base_array_layer == rhs.base_array_layer
        && lhs.layer_count == rhs.layer_count
        && lhs.base_mip_level == rhs.base_mip_level
        && lhs.level_count == rhs.level_count
}

// Added because vk-sync requires allocation to record barriers, see that impl for reference
fn pipeline_barrier_from_iters<'a>(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    global_barrier: Option<GlobalBarrier<'a>>,
    buffer_barriers: impl IntoIterator<Item = BufferBarrier<'a>>,
    image_barriers: impl IntoIterator<Item = ImageBarrier<'a>>,
) {
    #[derive(Default)]
    struct BarrierScratch {
        memory_barriers: Vec<vk::MemoryBarrier<'static>>,
        buffer_barriers: Vec<vk::BufferMemoryBarrier<'static>>,
        image_barriers: Vec<vk::ImageMemoryBarrier<'static>>,
    }

    thread_local! {
        static BARRIER: RefCell<BarrierScratch> = Default::default();
    }

    BARRIER.with_borrow_mut(|tls| {
        tls.memory_barriers.clear();
        tls.buffer_barriers.clear();
        tls.image_barriers.clear();

        let mut src_stage_mask = vk::PipelineStageFlags::TOP_OF_PIPE;
        let mut dst_stage_mask = vk::PipelineStageFlags::BOTTOM_OF_PIPE;

        if let Some(ref barrier) = global_barrier {
            let (src_mask, dst_mask, barrier) = get_memory_barrier(barrier);
            src_stage_mask |= src_mask;
            dst_stage_mask |= dst_mask;
            tls.memory_barriers.push(vk::MemoryBarrier {
                src_access_mask: barrier.src_access_mask,
                dst_access_mask: barrier.dst_access_mask,
                ..Default::default()
            });
        }

        for buffer_barrier in buffer_barriers {
            let (src_mask, dst_mask, barrier) = get_buffer_memory_barrier(&buffer_barrier);
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

        for image_barrier in image_barriers {
            let (src_mask, dst_mask, barrier) = get_image_memory_barrier(&image_barrier);
            src_stage_mask |= src_mask;
            dst_stage_mask |= dst_mask;
            tls.image_barriers.push(vk::ImageMemoryBarrier {
                src_access_mask: barrier.src_access_mask,
                dst_access_mask: barrier.dst_access_mask,
                old_layout: barrier.old_layout,
                new_layout: barrier.new_layout,
                src_queue_family_index: barrier.src_queue_family_index,
                dst_queue_family_index: barrier.dst_queue_family_index,
                image: barrier.image,
                subresource_range: barrier.subresource_range,
                ..Default::default()
            });
        }

        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                src_stage_mask,
                dst_stage_mask,
                vk::DependencyFlags::empty(),
                tls.memory_barriers.as_slice(),
                tls.buffer_barriers.as_slice(),
                tls.image_barriers.as_slice(),
            );
        }
    });
}

fn schedule_dependency_cmds_before_target_access(
    target_node_idx: usize,
    first_target_cmd_idx: usize,
    schedule: &mut Schedule,
) {
    let required_prefixes = schedule
        .access_index
        .read_nodes_for_cmd(first_target_cmd_idx)
        .filter(|&node_idx| node_idx != target_node_idx)
        .map(|node_idx| (node_idx, first_target_cmd_idx))
        .collect::<SmallVec<[_; 8]>>();

    schedule.schedule_required_node_prefixes(required_prefixes);
}

fn submit_stage_mask_legacy(stage_mask: vk::PipelineStageFlags2) -> vk::PipelineStageFlags {
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

fn supports_timeline_semaphores(device: &Device) -> bool {
    device.physical.features_v1_2.timeline_semaphore
}

/// Builds and submits a release barrier command buffer for each release group, calling
/// `submit_release` to perform the final queue submission.
fn submit_queue_ownership_releases<P>(
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
                    let _ = CommandBufferDebugLabel::begin(&release_cmd, "queue ownership barrier");

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

                    tls.release_image_barriers.extend(group.images.iter().map(
                        |&(handle, current_layout, subresource_range)| {
                            vk::ImageMemoryBarrier::default()
                                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                                .dst_access_mask(vk::AccessFlags::empty())
                                .old_layout(current_layout)
                                .new_layout(current_layout)
                                .src_queue_family_index(group.src_queue_family_index)
                                .dst_queue_family_index(target_queue_family_index)
                                .image(handle)
                                .subresource_range(subresource_range)
                        },
                    ));

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

#[derive(Clone, Copy, Debug)]
struct BufferQueueOwnershipTransfer {
    range: BufferSubresourceRange,
    dst_queue_family_index: u32,
    src_queue_family_index: u32,
}

#[derive(Clone, Default)]
struct CommandAccessIndex {
    cmds_by_node: Vec<Vec<usize>>,
    accessed_nodes_by_cmd: Vec<Vec<usize>>,
}

impl CommandAccessIndex {
    #[profiling::function]
    fn read_nodes_for_cmd(&self, cmd_idx: usize) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.accessed_nodes_by_cmd[cmd_idx].iter().copied()
    }

    fn update(&mut self, graph: &Graph, end_cmd_idx: usize) {
        let binding_count = graph.resources.len();
        let cmds = &graph.cmds[0..end_cmd_idx];
        self.update_from_cmds(cmds, binding_count);
    }

    fn update_from_cmds(&mut self, cmds: &[CommandData], binding_count: usize) {
        self.cmds_by_node.clear();
        self.cmds_by_node.resize_with(binding_count, Vec::new);

        self.accessed_nodes_by_cmd.clear();
        self.accessed_nodes_by_cmd.resize_with(cmds.len(), Vec::new);

        thread_local! {
            static SEEN_NODES: RefCell<(FixedBitSet, FixedBitSet)> = Default::default();
        }

        SEEN_NODES.with_borrow_mut(|(seen_nodes, seen_accesses)| {
            seen_nodes.clear();
            seen_nodes.grow(binding_count);

            seen_accesses.clear();
            seen_accesses.grow(binding_count);

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

                seen_nodes.clear();
                seen_nodes.grow(binding_count);
                seen_accesses.clear();
                seen_accesses.grow(binding_count);
            }
        });
    }
}

struct CommandBufferDebugLabel<'a> {
    cmd_buf: &'a CommandBuffer,
}

impl<'a> CommandBufferDebugLabel<'a> {
    fn begin(cmd_buf: &'a CommandBuffer, name: impl AsRef<str>) -> Option<Self> {
        Device::begin_debug_utils_label(&cmd_buf.device, cmd_buf.handle, name)
            .ok()
            .map(|_| Self { cmd_buf })
    }
}

impl Drop for CommandBufferDebugLabel<'_> {
    fn drop(&mut self) {
        let _ = Device::end_debug_utils_label(&self.cmd_buf.device, self.cmd_buf.handle);
    }
}

#[derive(Default)]
struct ExternalRenderPassAccessHistory {
    accesses_by_node: Vec<Vec<PipelineStageAccessFlags>>,
}

impl ExternalRenderPassAccessHistory {
    fn new(node_count: usize) -> Self {
        let mut accesses_by_node = Vec::with_capacity(node_count);
        accesses_by_node.resize_with(node_count, Vec::new);

        Self { accesses_by_node }
    }

    fn accesses(&self, node_idx: usize) -> &[PipelineStageAccessFlags] {
        &self.accesses_by_node[node_idx]
    }

    fn record_cmd(&mut self, cmd: &CommandData) {
        for exec in &cmd.execs {
            for (node_idx, accesses) in exec.accesses.iter() {
                self.accesses_by_node[node_idx].extend(
                    accesses
                        .iter()
                        .map(|access| PipelineStageAccessFlags::new(access.access)),
                );
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct QueueOwnershipReleaseWait {
    semaphore: vk::Semaphore,
    stage_mask: vk::PipelineStageFlags2,
    value: u64,
    device_index: u32,
}

#[derive(Debug, Default)]
struct CommandRecordingResources {
    descriptor_pool: Option<Lease<DescriptorPool>>,
    descriptor_sets: Vec<Vec<DescriptorSet>>,
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

#[derive(Clone, Copy, Debug)]
struct ImageQueueOwnershipTransfer {
    dst_queue_family_index: u32,
    layout: vk::ImageLayout,
    range: vk::ImageSubresourceRange,
    src_queue_family_index: u32,
    src_queue_index: u32,
}

impl PartialEq for ImageQueueOwnershipTransfer {
    fn eq(&self, other: &Self) -> bool {
        self.dst_queue_family_index == other.dst_queue_family_index
            && self.layout == other.layout
            && self.src_queue_family_index == other.src_queue_family_index
            && self.src_queue_index == other.src_queue_index
            && image_subresource_range_eq(self.range, other.range)
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

#[derive(Clone, Copy)]
struct PipelineStageAccessFlags {
    access_flags: vk::AccessFlags,
    stage_flags: vk::PipelineStageFlags,
}

impl PipelineStageAccessFlags {
    fn new(access: AccessType) -> Self {
        let (mut stage_flags, access_flags) = pipeline_stage_access_flags(access);
        if stage_flags.contains(vk::PipelineStageFlags::ALL_COMMANDS) {
            stage_flags |= vk::PipelineStageFlags::ALL_GRAPHICS;
            stage_flags &= !vk::PipelineStageFlags::ALL_COMMANDS;
        }

        Self {
            access_flags,
            stage_flags,
        }
    }
}

#[derive(Debug)]
struct QueueOwnershipRelease {
    _cmd_buf: Lease<CommandBuffer>,
    _fence: Fence,
    semaphore: vk::Semaphore,
}

#[derive(Debug)]
struct QueueOwnershipReleaseGroup {
    buffers: Vec<(vk::Buffer, BufferSubresourceRange)>,
    images: Vec<(vk::Image, vk::ImageLayout, vk::ImageSubresourceRange)>,
    src_queue_family_index: u32,
    src_queue_index: u32,
}

fn queue_ownership_release_group(
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
record_selection_from_node!(crate::node::AccelerationStructureNode);
record_selection_from_node!(crate::node::AccelerationStructureLeaseNode);
record_selection_from_node!(crate::node::BufferNode);
record_selection_from_node!(crate::node::BufferLeaseNode);
record_selection_from_node!(crate::node::ImageNode);
record_selection_from_node!(crate::node::ImageLeaseNode);
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
                check_queue_submit_args(waits, signals)?;

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
                            waits
                                .iter()
                                .map(|wait| submit_stage_mask_legacy(wait.stage_mask)),
                        );
                        tls.wait_semaphores
                            .extend(extra_waits.iter().map(|wait| wait.semaphore));
                        tls.wait_stage_masks.extend(
                            extra_waits
                                .iter()
                                .map(|wait| submit_stage_mask_legacy(wait.stage_mask)),
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
                check_queue_submit2_args(device, waits, signals)?;

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
    Cb: AsRef<CommandBuffer>,
{
    /// Returns `true` when this submission contains no more commands to record.
    pub fn is_empty(&self) -> bool {
        self.submission.is_empty()
    }

    /// Returns a borrow of the original Vulkan resource (buffer, image or acceleration structure)
    /// which the given node represents.
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: Node,
    {
        self.submission.resource(resource_node)
    }

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
        let releases = submit_queue_ownership_releases(
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

#[derive(Debug, Default)]
struct RecordingOwnership {
    // These ranges are effectively owned by this recording, but global ownership is not updated
    // until its command buffer is submitted successfully.
    buffers: HashMap<usize, Vec<BufferSubresourceRange>>,
    images: HashMap<usize, DenseMap<bool>>,
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
        self.images
            .entry(node_idx)
            .or_insert_with(|| DenseMap::new(info, false))
            .swap(true, range)
            .filter_map(|(claimed, range)| (!claimed).then_some(range))
            .collect()
    }
}

#[derive(Default)]
struct NodeScheduleScratch {
    covered_node_prefixes: Vec<usize>,
    pending_cmds: Vec<usize>,
    selected_cmds: FixedBitSet,
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
    node_schedule: NodeScheduleScratch,
}

impl Schedule {
    fn schedule_required_node_prefixes(
        &mut self,
        required_prefixes: impl IntoIterator<Item = (usize, usize)>,
    ) {
        fn schedule_node_prefix(
            access_index: &CommandAccessIndex,
            schedule: &mut Vec<usize>,
            scratch: &mut NodeScheduleScratch,
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

        self.cmds.clear();
        self.node_schedule.covered_node_prefixes.clear();
        self.node_schedule
            .covered_node_prefixes
            .resize(self.access_index.cmds_by_node.len(), 0);
        self.node_schedule.pending_cmds.clear();
        self.node_schedule.selected_cmds.clear();
        self.node_schedule
            .selected_cmds
            .grow(self.access_index.accessed_nodes_by_cmd.len());

        for (node_idx, end_cmd_idx) in required_prefixes {
            schedule_node_prefix(
                &self.access_index,
                &mut self.cmds,
                &mut self.node_schedule,
                node_idx,
                end_cmd_idx,
            );
        }

        while let Some(cmd_idx) = self.node_schedule.pending_cmds.pop() {
            for node_idx in self.access_index.read_nodes_for_cmd(cmd_idx) {
                schedule_node_prefix(
                    &self.access_index,
                    &mut self.cmds,
                    &mut self.node_schedule,
                    node_idx,
                    cmd_idx + 1,
                );
            }
        }

        self.cmds.sort_unstable();
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

        // Consecutive selected users of each resource form a dependency chain. This preserves the
        // original relative order for every shared-resource pair while still allowing unrelated
        // command chains to be grouped for locality.
        for resource_cmds in &self.access_index.cmds_by_node {
            let mut previous = Option::<usize>::None;
            for &cmd_idx in resource_cmds {
                let Some(&local_idx) = self.local_of_global.get(cmd_idx) else {
                    continue;
                };

                if local_idx == usize::MAX {
                    continue;
                }

                if let Some(previous_idx) = previous {
                    self.successors[previous_idx].push(local_idx);
                    self.predecessor_counts[local_idx] += 1;
                }

                previous = Some(local_idx);
            }
        }

        self.remaining_predecessors
            .clone_from(&self.predecessor_counts);
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
    fn is_supported_legacy_submit(&self) -> bool {
        self.value == 0
            && matches!(
                self.stage_mask,
                vk::PipelineStageFlags2::ALL_COMMANDS | vk::PipelineStageFlags2::NONE
            )
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
    pending_image_transfer_nodes:
        Option<PendingTransferNodes<vk::Image, ImageQueueOwnershipTransfer>>,
    queue_ownership_release_groups: Vec<QueueOwnershipReleaseGroup>,
    query_pool_results: Option<SubmittedTimestampQueries>,
    query_pool_reset: bool,
    recorded_commands: Vec<CommandRecordingResources>,
    submit_retained: Vec<SubmittedCommand>,
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
            submit_retained: Vec::new(),
        }
    }

    pub(crate) fn graph(&self) -> &Graph {
        &self.graph
    }

    fn signal_executed(&self) {
        for command in &self.submit_retained {
            command.signal_executed();
        }
    }

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

    pub(crate) fn record_prepared_command_stream(
        &mut self,
        cmd_buf: &CommandBuffer,
        resources: crate::ResourceMap,
    ) -> Result<(), DriverError> {
        let original_resources = std::mem::replace(&mut self.graph.resources, resources);

        let result = self.record_prepared_command_stream_inner(cmd_buf);

        self.graph.resources = original_resources;

        result
    }

    fn record_prepared_command_stream_inner(
        &mut self,
        cmd_buf: &CommandBuffer,
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
            self.track_pending_transfers(schedule, cmd_buf.info.queue_family_index, &mut ownership);
        });

        self.record_cmd_indices(cmd_buf, 0..self.graph.cmds.len())?;

        Ok(())
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

    fn is_framebuffer_space(stages: vk::PipelineStageFlags) -> bool {
        stages.intersects(
            vk::PipelineStageFlags::FRAGMENT_SHADER
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        )
    }

    fn subpass_dependency_stage_masks(
        previous: vk::PipelineStageFlags,
        current: vk::PipelineStageFlags,
    ) -> Option<(vk::PipelineStageFlags, vk::PipelineStageFlags)> {
        let all_graphics = vk::PipelineStageFlags::ALL_GRAPHICS;
        let previous_all_graphics = previous.contains(all_graphics);
        let current_all_graphics = current.contains(all_graphics);

        let overlaps = if previous_all_graphics && current_all_graphics {
            true
        } else if previous_all_graphics {
            current.intersects(Self::GRAPHICS_STAGES)
        } else if current_all_graphics {
            previous.intersects(Self::GRAPHICS_STAGES)
        } else {
            previous.intersects(current)
        };

        if !overlaps {
            return None;
        }

        if previous_all_graphics || current_all_graphics {
            Some((previous, current))
        } else {
            let stages = previous & current;

            Some((stages, stages))
        }
    }

    fn record_subpass_dependency(
        dependencies: &mut BTreeMap<(usize, usize), SubpassDependency>,
        src_subpass: usize,
        dst_subpass: usize,
        previous: PipelineStageAccessFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        current: &mut PipelineStageAccessFlags,
    ) -> bool {
        let Some((src_stage_mask, matched_dst_stages)) =
            Self::subpass_dependency_stage_masks(previous.stage_flags, current.stage_flags)
        else {
            return false;
        };

        let dep = dependencies
            .entry((src_subpass, dst_subpass))
            .or_insert_with(|| SubpassDependency::new(src_subpass as _, dst_subpass as _));

        dep.src_stage_mask |= src_stage_mask;
        dep.src_access_mask |= previous.access_flags;
        dep.dst_stage_mask |= dst_stage_mask;
        dep.dst_access_mask |= current.access_flags;

        if Self::is_framebuffer_space(previous.stage_flags | current.stage_flags) {
            dep.dependency_flags |= vk::DependencyFlags::BY_REGION;
        }

        current.stage_flags &= !matched_dst_stages;

        current.stage_flags.is_empty()
    }

    #[profiling::function]
    fn allow_merge_passes(lhs: &CommandData, rhs: &CommandData) -> bool {
        fn first_graphic_pipeline(pass: &CommandData) -> Option<&GraphicsPipeline> {
            pass.execs
                .first()
                .and_then(|exec| exec.pipeline.as_ref().map(ExecutionPipeline::as_graphics))
                .flatten()
        }

        fn is_multiview(view_mask: u32) -> bool {
            view_mask != 0
        }

        let lhs_pipeline = first_graphic_pipeline(lhs);
        if lhs_pipeline.is_none() {
            trace!("  {} is not graphics", lhs.name());

            return false;
        }

        let rhs_pipeline = first_graphic_pipeline(rhs);
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
            if is_multiview(lhs.view_mask) != is_multiview(rhs.view_mask) {
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

    fn subpass_stage_mask(stages: vk::PipelineStageFlags) -> vk::PipelineStageFlags {
        if stages.is_empty() {
            return stages;
        }

        if stages.contains(vk::PipelineStageFlags::ALL_GRAPHICS) {
            return vk::PipelineStageFlags::ALL_GRAPHICS;
        }

        let graphics_stages = stages & Self::GRAPHICS_STAGES;
        if graphics_stages.is_empty() {
            vk::PipelineStageFlags::ALL_GRAPHICS
        } else {
            graphics_stages
        }
    }

    fn attachment_write_access(aspect_mask: vk::ImageAspectFlags) -> vk::AccessFlags {
        match aspect_mask {
            mask if mask.contains(vk::ImageAspectFlags::COLOR) => {
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            }
            mask if mask
                .intersects(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL) =>
            {
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
            }
            _ => vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        }
    }

    fn accel_struct_canonical_accesses<'a>(
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

    fn color_attachment_is_read(load: LoadOp<[f32; 4]>) -> bool {
        matches!(load, LoadOp::Load)
    }

    fn color_attachment_is_write(
        load: LoadOp<[f32; 4]>,
        store: StoreOp,
        has_resolve: bool,
    ) -> bool {
        matches!(load, LoadOp::Clear(_)) || store == StoreOp::Store || has_resolve
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

    #[profiling::function]
    fn begin_render_pass(
        cmd_buf: &CommandBuffer,
        bindings: &[AnyResource],
        pass: &CommandData,
        recorded_command: &mut CommandRecordingResources,
        render_area: vk::Rect2D,
    ) -> Result<(), DriverError> {
        trace!("  begin render pass");

        let render_pass = recorded_command.expect_render_pass_mut();
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

            for exec in &pass.execs {
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
                    let attachment_idx = attachments.len() - 1 - state.resolve.is_some() as usize;
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
                    let attachment_idx = attachments.len() - 1;
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
                        .map(|descriptor_set| **descriptor_set),
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
            ExecutionPipeline::Graphics(pipeline) => RenderPass::pipeline_handle(
                recorded_command.expect_render_pass_mut(),
                pipeline,
                depth_stencil,
                exec_idx as _,
            )?,
            ExecutionPipeline::RayTracing(pipeline) => pipeline.handle(),
        };

        unsafe {
            cmd_buf
                .device
                .cmd_bind_pipeline(cmd_buf.handle, bind_point, pipeline);
        }

        Ok(())
    }

    /// Returns `true` when this submission contains no more commands to record.
    pub fn is_empty(&self) -> bool {
        self.graph.cmds.is_empty()
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
        let max_set_idx = pass
            .execs
            .iter()
            .flat_map(|exec| exec.bindings.keys())
            .map(|descriptor| descriptor.set())
            .max()
            .unwrap_or_default();
        let max_sets = pass.execs.len() as u32 * (max_set_idx + 1);
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

        // It's possible to execute a command-only pipeline
        if info.is_empty() {
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
        external_access_history: &ExternalRenderPassAccessHistory,
    ) -> Result<Lease<RenderPass>, DriverError>
    where
        P: SubmissionPool,
    {
        let pass = &self.graph.cmds[pass_idx];
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

                if !depth_stencil_resolve_set
                    && let Some(state) = exec
                        .attachments
                        .depth_stencil_attachment()
                        .and_then(|state| state.resolve)
                {
                    let attachment = attachments
                        .last_mut()
                        .expect("missing depth stencil resolve attachment");
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
        for (exec_idx, exec) in pass.execs.iter().enumerate() {
            let pipeline = exec
                .pipeline
                .as_ref()
                .expect("missing graphics pipeline")
                .expect_graphics();
            let mut subpass_info = SubpassInfo::with_capacity(attachment_count);

            // Add input attachments
            for attachment_idx in pipeline.inner.input_attachments.iter() {
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

                // Preserve the attachment in previous subpasses as needed. Input render passes are
                // expected to resolve to real prior subpasses here.
                for prev_exec_idx in (0..exec_idx).rev() {
                    let prev_exec = &pass.execs[prev_exec_idx];
                    if prev_exec
                        .attachments
                        .color_attachment(*attachment_idx)
                        .is_some_and(|state| state.store == StoreOp::Store)
                    {
                        break;
                    }

                    let prev_subpass = &mut subpasses[prev_exec_idx];
                    prev_subpass.preserve_attachments.push(*attachment_idx);
                }
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
                subpass_info.depth_stencil_resolve_attachment = Some((
                    AttachmentRef {
                        attachment: state.dst_attachment_idx + 1,
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

        let dependencies = Self::build_subpass_dependencies(pass, external_access_history);

        // let info = RenderPassInfo {
        //     attachments,
        //     dependencies,
        //     subpasses,
        // };

        // trace!("{:#?}", info);

        pool.render_pass(RenderPassInfo {
            attachments,
            dependencies,
            subpasses,
        })
    }

    fn build_subpass_dependencies(
        pass: &CommandData,
        external_access_history: &ExternalRenderPassAccessHistory,
    ) -> Vec<SubpassDependency> {
        let mut dependencies = BTreeMap::new();
        let mut pass_access_history =
            HashMap::<NodeIndex, Vec<(usize, PipelineStageAccessFlags)>>::new();

        for (exec_idx, exec) in pass.execs.iter().enumerate() {
            'exec_accesses: for (node_idx, accesses) in exec.accesses.iter() {
                for access in accesses {
                    let mut current = PipelineStageAccessFlags::new(access.access);
                    current.stage_flags = Self::subpass_stage_mask(current.stage_flags);

                    if let Some(prev_accesses) = pass_access_history.get(&node_idx) {
                        for &(prev_exec_idx, previous) in prev_accesses.iter().rev() {
                            if Self::record_subpass_dependency(
                                &mut dependencies,
                                prev_exec_idx,
                                exec_idx,
                                previous,
                                current.stage_flags,
                                &mut current,
                            ) {
                                continue 'exec_accesses;
                            }
                        }
                    }

                    for &previous in external_access_history.accesses(node_idx).iter().rev() {
                        if Self::record_subpass_dependency(
                            &mut dependencies,
                            vk::SUBPASS_EXTERNAL as usize,
                            exec_idx,
                            previous,
                            current.stage_flags,
                            &mut current,
                        ) {
                            continue 'exec_accesses;
                        }
                    }

                    if !current.stage_flags.is_empty() {
                        let dep = dependencies
                            .entry((vk::SUBPASS_EXTERNAL as usize, exec_idx))
                            .or_insert_with(|| {
                                SubpassDependency::new(vk::SUBPASS_EXTERNAL, exec_idx as _)
                            });

                        dep.src_stage_mask |= vk::PipelineStageFlags::ALL_COMMANDS;
                        dep.src_access_mask |=
                            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
                        dep.dst_stage_mask |= current.stage_flags;
                        dep.dst_access_mask |= current.access_flags;
                    }
                }
            }

            for (node_idx, accesses) in exec.accesses.iter() {
                let prev_accesses = pass_access_history.entry(node_idx).or_default();
                prev_accesses.extend(accesses.iter().map(|access| {
                    let mut access_info = PipelineStageAccessFlags::new(access.access);
                    access_info.stage_flags = Self::subpass_stage_mask(access_info.stage_flags);

                    (exec_idx, access_info)
                }));
            }

            // Look for attachments of this exec being read or written in other execs of the
            // same pass
            for (other_idx, other) in pass.execs[0..exec_idx].iter().enumerate() {
                // Look for color attachments we're reading
                for (attachment_idx, state) in
                    exec.attachments.color_attachments().filter(|(_, state)| {
                        state.is_input || Self::color_attachment_is_read(state.load)
                    })
                {
                    // Look for writes in the other exec
                    if let Some(other_state) = other.attachments.color_attachment(attachment_idx)
                        && Self::color_attachment_is_write(
                            other_state.load,
                            other_state.store,
                            other_state.resolve.is_some(),
                        )
                    {
                        let dep = dependencies
                            .entry((other_idx, exec_idx))
                            .or_insert_with(|| {
                                SubpassDependency::new(other_idx as _, exec_idx as _)
                            });

                        dep.src_stage_mask |= vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
                        dep.src_access_mask |= vk::AccessFlags::COLOR_ATTACHMENT_WRITE;

                        if state.is_input {
                            dep.dst_stage_mask |= vk::PipelineStageFlags::FRAGMENT_SHADER;
                            dep.dst_access_mask |= vk::AccessFlags::INPUT_ATTACHMENT_READ;
                        } else {
                            dep.dst_stage_mask |=
                                Self::attachment_read_stage(state.attachment.aspect_mask);
                            dep.dst_access_mask |= vk::AccessFlags::COLOR_ATTACHMENT_READ;
                        }
                    }

                    if let Some(other_state) = other.attachments.color_attachment(attachment_idx)
                        && (other_state.is_input
                            || Self::color_attachment_is_read(other_state.load))
                    {
                        let dep = dependencies
                            .entry((other_idx, exec_idx))
                            .or_insert_with(|| {
                                SubpassDependency::new(other_idx as _, exec_idx as _)
                            });

                        if other_state.is_input {
                            dep.src_stage_mask |= vk::PipelineStageFlags::FRAGMENT_SHADER;
                            dep.src_access_mask |= vk::AccessFlags::INPUT_ATTACHMENT_READ;
                        } else {
                            dep.src_stage_mask |=
                                Self::attachment_read_stage(state.attachment.aspect_mask);
                            dep.src_access_mask |= vk::AccessFlags::COLOR_ATTACHMENT_READ;
                        }

                        if state.is_input {
                            dep.dst_stage_mask |= vk::PipelineStageFlags::FRAGMENT_SHADER;
                            dep.dst_access_mask |= vk::AccessFlags::INPUT_ATTACHMENT_READ;
                        } else {
                            dep.dst_stage_mask |=
                                Self::attachment_read_stage(state.attachment.aspect_mask);
                            dep.dst_access_mask |= vk::AccessFlags::COLOR_ATTACHMENT_READ;
                        }
                    }
                }

                if let Some(state) = exec.attachments.depth_stencil_attachment().filter(|state| {
                    state.is_attachment && Self::depth_stencil_attachment_is_read(state.load)
                }) {
                    let aspect_mask = state.attachment.aspect_mask;

                    if other
                        .attachments
                        .depth_stencil_attachment()
                        .is_some_and(|state| {
                            Self::depth_stencil_attachment_is_write(
                                state.load,
                                state.store,
                                state.resolve.is_some(),
                            )
                        })
                    {
                        let dep = dependencies
                            .entry((other_idx, exec_idx))
                            .or_insert_with(|| {
                                SubpassDependency::new(other_idx as _, exec_idx as _)
                            });

                        dep.src_stage_mask |= vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
                        dep.src_access_mask |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
                        dep.dst_stage_mask |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS;
                        dep.dst_access_mask |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
                    }

                    if other
                        .attachments
                        .depth_stencil_attachment()
                        .is_some_and(|state| Self::depth_stencil_attachment_is_read(state.load))
                    {
                        let dep = dependencies
                            .entry((other_idx, exec_idx))
                            .or_insert_with(|| {
                                SubpassDependency::new(other_idx as _, exec_idx as _)
                            });

                        dep.src_stage_mask |= vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
                        dep.src_access_mask |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
                        dep.dst_stage_mask |= Self::attachment_read_stage(aspect_mask);
                        dep.dst_access_mask |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
                    }
                }

                for (attachment_idx, state) in
                    exec.attachments.color_attachments().filter(|(_, state)| {
                        Self::color_attachment_is_write(
                            state.load,
                            state.store,
                            state.resolve.is_some(),
                        )
                    })
                {
                    let aspect_mask = state.attachment.aspect_mask;
                    let stage = Self::attachment_stage(aspect_mask);

                    if other
                        .attachments
                        .color_attachment(attachment_idx)
                        .is_some_and(|state| {
                            Self::color_attachment_is_write(
                                state.load,
                                state.store,
                                state.resolve.is_some(),
                            )
                        })
                    {
                        let access = Self::attachment_write_access(aspect_mask);
                        let dep = dependencies
                            .entry((other_idx, exec_idx))
                            .or_insert_with(|| {
                                SubpassDependency::new(other_idx as _, exec_idx as _)
                            });

                        dep.src_stage_mask |= stage;
                        dep.src_access_mask |= access;
                        dep.dst_stage_mask |= stage;
                        dep.dst_access_mask |= access;
                    }

                    if let Some(other_state) = other.attachments.color_attachment(attachment_idx)
                        && (other_state.is_input
                            || Self::color_attachment_is_read(other_state.load))
                    {
                        let (src_access, dst_access) =
                            Self::attachment_read_write_access(aspect_mask);
                        let dep = dependencies
                            .entry((other_idx, exec_idx))
                            .or_insert_with(|| {
                                SubpassDependency::new(other_idx as _, exec_idx as _)
                            });

                        if other_state.is_input {
                            dep.src_stage_mask |= vk::PipelineStageFlags::FRAGMENT_SHADER;
                            dep.src_access_mask |= vk::AccessFlags::INPUT_ATTACHMENT_READ;
                        } else {
                            dep.src_stage_mask |= Self::attachment_read_stage(aspect_mask);
                            dep.src_access_mask |= src_access;
                        }
                        dep.dst_stage_mask |= stage;
                        dep.dst_access_mask |= dst_access;
                    }
                }

                if let Some(state) = exec.attachments.depth_stencil_attachment().filter(|state| {
                    Self::depth_stencil_attachment_is_write(
                        state.load,
                        state.store,
                        state.resolve.is_some(),
                    )
                }) {
                    let aspect_mask = state.attachment.aspect_mask;
                    let stage = Self::attachment_stage(aspect_mask);

                    if other
                        .attachments
                        .depth_stencil_attachment()
                        .is_some_and(|state| {
                            Self::depth_stencil_attachment_is_write(
                                state.load,
                                state.store,
                                state.resolve.is_some(),
                            )
                        })
                    {
                        let access = Self::attachment_write_access(aspect_mask);
                        let dep = dependencies
                            .entry((other_idx, exec_idx))
                            .or_insert_with(|| {
                                SubpassDependency::new(other_idx as _, exec_idx as _)
                            });

                        dep.src_stage_mask |= stage;
                        dep.src_access_mask |= access;
                        dep.dst_stage_mask |= stage;
                        dep.dst_access_mask |= access;
                    }

                    if other
                        .attachments
                        .depth_stencil_attachment()
                        .is_some_and(|state| Self::depth_stencil_attachment_is_read(state.load))
                    {
                        let (src_access, dst_access) =
                            Self::attachment_read_write_access(aspect_mask);
                        let dep = dependencies
                            .entry((other_idx, exec_idx))
                            .or_insert_with(|| {
                                SubpassDependency::new(other_idx as _, exec_idx as _)
                            });

                        dep.src_stage_mask |= Self::attachment_read_stage(aspect_mask);
                        dep.src_access_mask |= src_access;
                        dep.dst_stage_mask |= stage;
                        dep.dst_access_mask |= dst_access;
                    }
                }
            }
        }

        dependencies.into_values().collect()
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
        let mut render_pass_access_history =
            ExternalRenderPassAccessHistory::new(self.graph.resources.len());

        for pass_idx in schedule.iter().copied() {
            // At the time this function runs the pass will already have been optimized into a
            // larger pass made out of anything that might have been merged into it - so we
            // only care about one pass at a time here
            let pass = &self.graph.cmds[pass_idx];

            trace!("requesting [{pass_idx}: {}]", pass.name());

            let descriptor_pool = Self::lease_descriptor_pool(pool, pass)?;
            let mut descriptor_sets = Vec::with_capacity(pass.execs.len());
            descriptor_sets.resize_with(pass.execs.len(), Vec::new);
            if let Some(descriptor_pool) = descriptor_pool.as_ref() {
                for (exec_idx, exec) in pass.execs.iter().enumerate() {
                    let Some(pipeline) = exec.pipeline.as_ref() else {
                        continue;
                    };

                    let layouts = pipeline.descriptor_info().layouts.values();
                    descriptor_sets[exec_idx] = layouts
                        .into_iter()
                        .map(|descriptor_set_layout| {
                            DescriptorPool::allocate_descriptor_set(
                                descriptor_pool,
                                descriptor_set_layout,
                            )
                        })
                        .collect::<Result<_, _>>()?;
                }
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
                        .values()
                        .filter_map(|pool| pool.get(&vk::DescriptorType::INPUT_ATTACHMENT))
                        .next()
                        .is_none()
            );

            // Also, the render pass may be None if the pass contained no graphics operations.
            let render_pass = if pass
                .expect_first_exec()
                .pipeline
                .as_ref()
                .map(|pipeline| pipeline.is_graphics())
                .unwrap_or_default()
            {
                Some(self.lease_render_pass(pool, pass_idx, &render_pass_access_history)?)
            } else {
                None
            };

            render_pass_access_history.record_cmd(pass);

            self.recorded_commands.push(CommandRecordingResources {
                descriptor_pool,
                descriptor_sets,
                render_pass,
            });
        }

        Ok(())
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
        }
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

    #[profiling::function]
    fn record_execution_barriers<'a>(
        cmd_buf: &CommandBuffer,
        resources: &mut [AnyResource],
        accesses: &'a ExecutionAccess,
        pending_buffer_transfer_nodes: &mut Option<
            PendingTransferNodes<vk::Buffer, BufferQueueOwnershipTransfer>,
        >,
        pending_image_transfer_nodes: &mut Option<
            PendingTransferNodes<vk::Image, ImageQueueOwnershipTransfer>,
        >,
    ) {
        // We store a Barriers in TLS to save an alloc; contents are POD
        thread_local! {
            static BARRIER: RefCell<BarrierScratch> = Default::default();
        }

        struct AccessBarrier<T> {
            next_access: AccessType,
            prev_access: AccessType,
            resource: T,
        }

        struct BufferBarrierTarget {
            buffer: vk::Buffer,
            range: BufferSubresourceRange,
        }

        struct ImageBarrierTarget {
            image: vk::Image,
            range: vk::ImageSubresourceRange,
        }

        #[derive(Default)]
        struct BarrierScratch {
            accel_struct_accesses: Vec<AccessType>,
            buffers: Vec<AccessBarrier<BufferBarrierTarget>>,
            images: Vec<AccessBarrier<ImageBarrierTarget>>,
            next_accesses: Vec<AccessType>,
            pending_buffers: NodeIndexedScratch<AccessBarrier<BufferBarrierTarget>>,
            pending_images: NodeIndexedScratch<AccessBarrier<ImageBarrierTarget>>,
            prev_accesses: Vec<AccessType>,
        }

        BARRIER.with_borrow_mut(|tls| {
            // Initialize TLS from a previous call
            tls.accel_struct_accesses.clear();
            tls.buffers.clear();
            tls.images.clear();
            tls.next_accesses.clear();
            tls.pending_buffers.clear();
            tls.pending_images.clear();
            tls.prev_accesses.clear();

            // Map remaining accesses into vk_sync barriers (some accesses may have been removed by
            // the render pass request function)

            for (node_idx, node_accesses) in accesses.iter() {
                enum ResourceRef<'a> {
                    AccelerationStructure(&'a AccelerationStructure),
                    Buffer(&'a Buffer),
                    Image(&'a Image),
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
                    AnyResource::SwapchainImage(resource) => ResourceRef::Image(resource),
                };

                match resource {
                    ResourceRef::AccelerationStructure(accel_struct) => {
                        let canonical_accesses = Self::accel_struct_canonical_accesses(
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
                        for (next_access, prev_access, range) in Image::swap_accesses(
                            image,
                            node_accesses.iter().map(
                                |&SubresourceAccess {
                                     access,
                                     subresource,
                                 }| {
                                    let SubresourceRange::Image(range) = subresource else {
                                        unreachable!()
                                    };

                                    (access, range)
                                },
                            ),
                        ) {
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
                        buffer_barriers.extend(buffer_barriers_from_transfers(
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

                image_barriers.push(ImageBarrier {
                    next_accesses: slice::from_ref(next_access),
                    previous_accesses: slice::from_ref(prev_access),
                    next_layout: image_access_layout(*next_access),
                    previous_layout: image_access_layout(*prev_access),
                    discard_contents: image_execution_discard_contents(*prev_access),
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    range,
                });
            }

            if let Some(pending_image_transfer_nodes) = pending_image_transfer_nodes.as_ref() {
                for (node_idx, _image, transfers) in pending_image_transfer_nodes.iter() {
                    for AccessBarrier {
                        next_access,
                        prev_access,
                        resource,
                    } in tls.pending_images.get(node_idx)
                    {
                        image_barriers.extend(image_barriers_from_transfers(
                            resource.image,
                            prev_access,
                            next_access,
                            resource.range,
                            transfers,
                            image_execution_discard_contents(*prev_access),
                        ));
                    }
                }
            }

            pipeline_barrier_from_iters(
                &cmd_buf.device,
                cmd_buf.handle,
                global_barrier,
                buffer_barriers.into_iter(),
                image_barriers.into_iter(),
            );

            if let Some(pending) = pending_buffer_transfer_nodes.as_mut() {
                pending.remove_where(|node_idx, _buffer, transfers| {
                    for AccessBarrier { resource, .. } in tls.pending_buffers.get(node_idx) {
                        let range = resource.range;

                        if consume_pending_buffer_transfers(transfers, range) {
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

                        if consume_pending_image_transfers(transfers, range) {
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
            PendingTransferNodes<vk::Image, ImageQueueOwnershipTransfer>,
        >,
    ) {
        struct ImageResourceBarrier {
            image: vk::Image,
            node_idx: NodeIndex,
            next_access: AccessType,
            prev_access: AccessType,
            range: vk::ImageSubresourceRange,
        }

        struct BufferResourceBarrier {
            buffer: vk::Buffer,
            next_access: AccessType,
            prev_access: AccessType,
            range: BufferSubresourceRange,
        }

        #[derive(Default)]
        struct LayoutTransitionScratch {
            buffers: Vec<BufferResourceBarrier>,
            images: Vec<ImageResourceBarrier>,
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
            tls.first_layout_uses.clear();
            tls.pending_buffers.clear();
            tls.pending_images.clear();

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

                            for (prev_access, range) in
                                Buffer::swap_access(buffer, AccessType::Nothing, access_range)
                            {
                                if !pending_buffer_transfer_nodes
                                    .as_ref()
                                    .is_some_and(|pending| pending.contains(node_idx))
                                {
                                    continue;
                                }

                                tls.pending_buffers.push(
                                    node_idx,
                                    BufferResourceBarrier {
                                        buffer: buffer.handle,
                                        next_access: access,
                                        prev_access,
                                        range,
                                    },
                                );
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
                                for (prev_access, range) in
                                    Image::swap_access(image, access, layout_range)
                                {
                                    if is_initial_layout {
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
                                }
                            }
                        }
                    }
                }
            }

            let mut buffer_barriers = Vec::new();
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
                        for transfer in transfers.iter().copied() {
                            let Some(range) = range.intersection(transfer.range) else {
                                continue;
                            };

                            trace!(
                                "    buffer {:?} {:?} {:?}->{:?}",
                                buffer,
                                range.start..range.end,
                                prev_access,
                                next_access,
                            );

                            buffer_barriers.push(BufferBarrier {
                                next_accesses: slice::from_ref(next_access),
                                previous_accesses: slice::from_ref(prev_access),
                                src_queue_family_index: transfer.src_queue_family_index,
                                dst_queue_family_index: transfer.dst_queue_family_index,
                                buffer: *buffer,
                                offset: range.start as _,
                                size: (range.end - range.start) as _,
                            });
                        }
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

                image_barriers.extend(image_barriers_from_transfers(
                    *image,
                    prev_access,
                    next_access,
                    *range,
                    &[],
                    image_layout_transition_discard_contents(*prev_access, *next_access),
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
                        image_barriers.extend(image_barriers_from_transfers(
                            *image,
                            prev_access,
                            next_access,
                            *range,
                            transfers,
                            image_layout_transition_discard_contents(*prev_access, *next_access),
                        ));
                    }
                }
            }

            pipeline_barrier_from_iters(
                &cmd_buf.device,
                cmd_buf.handle,
                None,
                buffer_barriers.into_iter(),
                image_barriers.into_iter(),
            );

            if let Some(pending) = pending_buffer_transfer_nodes.as_mut() {
                pending.remove_where(|node_idx, _buffer, transfers| {
                    for BufferResourceBarrier { range, .. } in tls.pending_buffers.get(node_idx) {
                        if consume_pending_buffer_transfers(transfers, *range) {
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
                        if consume_pending_image_transfers(transfers, *range) {
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

    fn track_pending_transfers(
        &mut self,
        schedule: &Schedule,
        queue_family_index: u32,
        ownership: &mut RecordingOwnership,
    ) {
        let resource_count = self.graph.resources.len();

        for cmd_idx in schedule.cmds.iter().copied() {
            let cmd = &self.graph.cmds[cmd_idx];

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
                                    exclusive_transfer_source(sharing, queue_family_index)
                                else {
                                    continue;
                                };
                                let transfer = BufferQueueOwnershipTransfer {
                                    src_queue_family_index,
                                    dst_queue_family_index: queue_family_index,
                                    range,
                                };

                                queue_ownership_release_group(
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
                                exclusive_transfer_source(sharing, queue_family_index)
                            else {
                                continue;
                            };
                            let layout = subresource.layout.unwrap_or(vk::ImageLayout::UNDEFINED);
                            let transfer = ImageQueueOwnershipTransfer {
                                src_queue_family_index,
                                src_queue_index,
                                dst_queue_family_index: queue_family_index,
                                layout,
                                range,
                            };

                            queue_ownership_release_group(
                                &mut self.queue_ownership_release_groups,
                                src_queue_family_index,
                                src_queue_index,
                            )
                            .images
                            .push((image.handle, layout, range));
                            self.pending_image_transfer_nodes
                                .get_or_insert_with(|| PendingTransferNodes::new(resource_count))
                                .push_transfer(node_idx, image.handle, transfer);
                        }
                    }
                }
            }
        }
    }

    fn record_cmd_indices(
        &mut self,
        cmd_buf: &CommandBuffer,
        cmd_indices: impl IntoIterator<Item = usize>,
    ) -> Result<(), DriverError> {
        #[cfg(feature = "checked")]
        let graph_id = self.graph.graph_id();
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

            trace!("recording cmd [{}: {}]", cmd_idx, cmd.name());

            if !recorded_command.descriptor_sets.is_empty() {
                Self::write_descriptor_sets(cmd_buf, &self.graph.resources, cmd, recorded_command)?;
            }

            let (render_area, render_pass_label) = if is_graphics {
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

            for exec_idx in 0..cmd.execs.len() {
                let render_area = if is_graphics {
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
                    if is_graphics {
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

                if let Some(pipeline) = exec.pipeline.as_mut() {
                    Self::bind_pipeline(
                        cmd_buf,
                        recorded_command,
                        exec_idx,
                        pipeline,
                        exec.depth_stencil,
                    )?;

                    if is_graphics {
                        let render_area = render_area.expect("missing render area");

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
                        exec,
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
        self.track_pending_transfers(schedule, cmd_buf.info.queue_family_index, ownership);

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

        self.record_cmd_indices(cmd_buf, schedule.cmds.iter().copied())?;

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

    /// Returns a borrow of the original Vulkan resource (buffer, image or acceleration structure)
    /// which the given node represents.
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: Node,
    {
        self.graph.resource(resource_node)
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
                schedule_dependency_cmds_before_target_access(node_idx, end_pass_idx, tls);
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
    fn write_descriptor_sets(
        cmd_buf: &CommandBuffer,
        bindings: &[AnyResource],
        pass: &CommandData,
        recorded_command: &CommandRecordingResources,
    ) -> Result<(), DriverError> {
        #[derive(Clone, Copy)]
        struct IndexedWrite<'a> {
            info_idx: usize,
            write: vk::WriteDescriptorSet<'a>,
        }

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
                            if image_view_info.aspect_mask.contains(
                                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                            ) {
                                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                            } else if image_view_info
                                .aspect_mask
                                .contains(vk::ImageAspectFlags::DEPTH)
                            {
                                vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL
                            } else if image_view_info
                                .aspect_mask
                                .contains(vk::ImageAspectFlags::STENCIL)
                            {
                                vk::ImageLayout::STENCIL_READ_ONLY_OPTIMAL
                            } else {
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                            }
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
                                dst_set: *descriptor_sets[descriptor_set_idx as usize],
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
                                dst_set: *descriptor_sets[descriptor_set_idx as usize],
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
                                .dst_set(*descriptor_sets[descriptor_set_idx as usize])
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
                if exec_idx > 0 {
                    for (
                        &Descriptor {
                            set: descriptor_set_idx,
                            binding: dst_binding,
                        },
                        (descriptor_info, _),
                    ) in &pipeline.inner.descriptor_bindings
                    {
                        if let DescriptorInfo::InputAttachment(_, attachment_idx) = *descriptor_info
                        {
                            let current_attachment = exec
                                .attachments
                                .color_attachment(attachment_idx)
                                .map(|state| state.attachment)
                                .expect("missing input attachment target");
                            let attachment = pass.execs[0..exec_idx]
                                .iter()
                                .rev()
                                .find_map(|exec| {
                                    exec.attachments
                                        .color_attachment(attachment_idx)
                                        .map(|state| state.attachment)
                                        .filter(|attachment| {
                                            Attachment::are_compatible(
                                                Some(current_attachment),
                                                Some(*attachment),
                                            )
                                        })
                                })
                                .expect("input attachment not written");
                            let image_binding = &bindings[attachment.target];
                            let image = image_binding.expect_image();
                            let image_view =
                                Image::view(image, attachment.image_view_info(image.info))?;

                            tls.image_writes.push(IndexedWrite {
                                info_idx: tls.image_infos.len(),
                                write: vk::WriteDescriptorSet {
                                    dst_set: *descriptor_sets[descriptor_set_idx as usize],
                                    dst_binding,
                                    descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                                    descriptor_count: 1,
                                    ..Default::default()
                                },
                            });

                            tls.image_infos.push(vk::DescriptorImageInfo {
                                image_layout: Self::attachment_layout(
                                    attachment.aspect_mask,
                                    exec.attachments
                                        .color_attachment(attachment_idx)
                                        .map(|state| {
                                            state.store == StoreOp::Store || state.resolve.is_some()
                                        })
                                        .unwrap_or_default(),
                                    true,
                                ),
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

#[derive(Clone, Copy, Debug)]
struct TimestampQueryResultInfo {
    timestamp_query: u32,
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

    fn set_result_info(&mut self, query: TimestampQuery, result_info: TimestampQueryResultInfo) {
        let index = query.index() as usize;
        if index >= self.result_infos.len() {
            self.result_infos.resize(index + 1, None);
        }

        self.result_infos[index] = Some(result_info);
    }

    fn query_pool(&self) -> vk::QueryPool {
        self.query_pool.handle
    }

    fn reset(&self, cmd_buf: &CommandBuffer) {
        self.query_pool.reset(cmd_buf, 0, self.query_count);
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

    pub(crate) fn set(&self, timestamps: Box<[Option<Duration>]>) {
        let mut inner = self.inner.lock().expect("timestamp query pool poisoned");
        inner.timestamps = Some(timestamps);
        inner.got_results = true;
    }

    pub(crate) fn complete_without_results(&self) {
        self.inner
            .lock()
            .expect("timestamp query pool poisoned")
            .got_results = true;
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
}

#[derive(Debug)]
struct TimestampQueryPoolInner {
    got_results: bool,
    #[cfg(feature = "checked")]
    graph_id: Option<GraphId>,
    timestamps: Option<Box<[Option<Duration>]>>,
}

#[doc(hidden)]
pub mod bench {
    use {
        super::{CommandAccessIndex, Schedule},
        crate::Graph,
    };

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

                let seed = splitmix64(node_idx as u64 ^ ((spec.cmd_count as u64) << 32));
                let stride = odd_stride(seed, spec.cmd_count);
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

                while cmds.len() < uses {
                    let next_cmd = (start + cmds.len() * stride + cmds.len()) % spec.cmd_count;
                    if cmds.binary_search(&next_cmd).is_err() {
                        cmds.push(next_cmd);
                    }
                }

                cmds.sort_unstable();

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
            assert!(base_cmd_count > 0, "graph must contain commands");
            assert!(base_resource_count > 0, "graph must contain resources");

            let mut base = CommandAccessIndex::default();
            base.update_from_cmds(&graph.cmds, base_resource_count);

            let cmd_count = base_cmd_count * repeat_count;
            let resource_count = base_resource_count * repeat_count;
            let mut cmds_by_node = Vec::with_capacity(resource_count);
            let mut accessed_nodes_by_cmd = Vec::with_capacity(cmd_count);

            for copy_idx in 0..repeat_count {
                let cmd_offset = copy_idx * base_cmd_count;
                let resource_offset = copy_idx * base_resource_count;

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
            }

            let cmds = (0..cmd_count).collect::<Vec<_>>();
            Self {
                schedule: Schedule {
                    access_index: CommandAccessIndex {
                        cmds_by_node,
                        accessed_nodes_by_cmd,
                    },
                    cmds: cmds.clone(),
                    ..Default::default()
                },
                original_cmds: cmds,
                end_cmd_idx: cmd_count,
            }
        }

        /// Returns the number of commands reordered by each benchmark iteration.
        pub fn cmd_count(&self) -> usize {
            self.end_cmd_idx
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
    }

    fn odd_stride(seed: u64, cmd_count: usize) -> usize {
        let stride = ((seed >> 32) as usize % cmd_count.max(2)) | 1;

        stride.min(cmd_count.max(1) - 1).max(1)
    }

    fn splitmix64(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
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

    pub fn check_schedule_reordering(cmd_count: usize, resource_accesses: &[Vec<ResourceAccess>]) {
        let cmd_count = cmd_count.min(256);
        if cmd_count == 0 {
            return;
        }

        let (access_index, normalized_accesses) = build_access_index(cmd_count, resource_accesses);

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

        let expected = reference_reorder(access_index, cmd_count);
        assert_eq!(
            reordered, expected,
            "reordering diverged from reference implementation"
        );

        assert_hazard_order_preserved(&reordered, &normalized_accesses);
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
            },
            normalized_accesses,
        )
    }

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

#[cfg(test)]
mod test {
    use super::{
        BufferQueueOwnershipTransfer, CommandAccessIndex, CommandData,
        ExternalRenderPassAccessHistory, ImageQueueOwnershipTransfer, NodeIndex,
        PipelineStageAccessFlags, QueueSubmitInfo, RecordSelection, RecordedSubmission,
        RecordedSubmissionState, RecordingOwnership, Schedule, SemaphoreSubmitInfo, Submission,
        SubresourceAccess, SubresourceRange, check_queue_submit_args, fuzz,
    };
    use crate::{
        AnyResource, Attachment, DepthStencilAttachment, Execution, Graph, LoadOp, Node, StoreOp,
        TimestampQuery,
        driver::{
            DriverError, SharingMode,
            accel_struct::{AccelerationStructure, AccelerationStructureInfo},
            ash::vk,
            buffer::{Buffer, BufferInfo, BufferSubresourceRange},
            cmd_buf::{CommandBuffer, CommandBufferInfo},
            device::{Device, DeviceInfo},
            fence::Fence,
            graphics::{GraphicsPipeline, GraphicsPipelineInfo},
            image::{Image, ImageInfo, SampleCount},
            render_pass::SubpassDependency,
        },
        node::{AnyNode, BufferNode},
        pool::{Pool, hash::HashPool},
    };
    use {
        ash::vk::Handle,
        std::{
            env::set_var,
            mem::ManuallyDrop,
            ops::Deref,
            sync::{Arc, Mutex, MutexGuard, OnceLock},
            time::Duration,
        },
        vk_shader_macros::glsl,
        vk_sync::AccessType,
    };

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
    fn sort_pending_image_transfers(transfers: &mut [ImageQueueOwnershipTransfer]) {
        transfers.sort_unstable_by_key(|transfer| {
            (
                transfer.src_queue_family_index,
                transfer.src_queue_index,
                transfer.dst_queue_family_index,
                transfer.layout.as_raw(),
                transfer.range.aspect_mask.as_raw(),
                transfer.range.base_array_layer,
                transfer.range.layer_count,
                transfer.range.base_mip_level,
                transfer.range.level_count,
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

    fn pending_buffer_transfer_for_range(
        transfers: &[BufferQueueOwnershipTransfer],
        range: BufferSubresourceRange,
    ) -> Option<&BufferQueueOwnershipTransfer> {
        transfers.iter().find(|transfer| transfer.range == range)
    }

    fn pending_transfer_for_node<H: Copy, T>(
        pending: &super::PendingTransferNodes<H, T>,
        node_idx: NodeIndex,
    ) -> Option<(H, &[T])> {
        pending
            .iter()
            .find_map(|(idx, handle, transfers)| (idx == node_idx).then_some((handle, transfers)))
    }

    fn simulate_partial_transfer_discovery(
        submission: &mut Submission,
        schedule: &Schedule,
        queue_family_index: u32,
        ownership: &mut RecordingOwnership,
    ) {
        submission.track_pending_transfers(schedule, queue_family_index, ownership);
        submission.pending_buffer_transfer_nodes = None;
        submission.pending_image_transfer_nodes = None;
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
    fn pending_transfer_nodes_remove_where_drops_stale_indices() {
        let mut pending = super::PendingTransferNodes::new(3);

        pending.push_transfer(1, 11, 21);
        pending.entries[1] = None;
        pending.remove_where(|_, _, _| false);

        assert!(pending.indices.is_empty());
        assert_eq!(pending.iter().count(), 0);
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

        assert!(!super::consume_pending_buffer_transfers(
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
            ImageQueueOwnershipTransfer {
                dst_queue_family_index: 0,
                layout: vk::ImageLayout::GENERAL,
                range: consumed,
                src_queue_family_index: 1,
                src_queue_index: 0,
            },
            ImageQueueOwnershipTransfer {
                dst_queue_family_index: 0,
                layout: vk::ImageLayout::GENERAL,
                range: kept,
                src_queue_family_index: 1,
                src_queue_index: 0,
            },
        ];

        assert!(!super::consume_pending_image_transfers(
            &mut pending,
            consumed
        ));

        assert_eq!(pending.len(), 1);
        assert!(super::image_subresource_range_eq(pending[0].range, kept));
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
        assert!(super::image_subresource_range_eq(claimed[0], first));

        let claimed = ownership.claim_image(0, info, overlap);
        assert_eq!(claimed.len(), 1);
        assert!(super::image_subresource_range_eq(claimed[0], remaining));
        assert!(ownership.claim_image(0, info, overlap).is_empty());
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
        };
        let mut schedule = Schedule {
            access_index,
            ..Default::default()
        };

        super::schedule_dependency_cmds_before_target_access(1, 1, &mut schedule);

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
        };
        let mut schedule = Schedule {
            access_index,
            ..Default::default()
        };

        super::schedule_dependency_cmds_before_target_access(3, 3, &mut schedule);

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

    #[cfg(test)]
    fn sort_queue_ownership_release_groups(groups: &mut [super::QueueOwnershipReleaseGroup]) {
        for group in groups.iter_mut() {
            group
                .buffers
                .sort_unstable_by_key(|(buffer, range)| (buffer.as_raw(), range.start, range.end));

            group.images.sort_unstable_by_key(|(image, layout, range)| {
                (
                    image.as_raw(),
                    layout.as_raw(),
                    range.aspect_mask.as_raw(),
                    range.base_array_layer,
                    range.layer_count,
                    range.base_mip_level,
                    range.level_count,
                )
            });
        }

        groups.sort_unstable_by_key(|group| (group.src_queue_family_index, group.src_queue_index));
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

    #[derive(Debug)]
    struct TestDevice {
        _guard: MutexGuard<'static, ()>,
        device: ManuallyDrop<Device>,
    }

    impl Drop for TestDevice {
        fn drop(&mut self) {
            // Drop the Vulkan device while the global test-device lock is still held
            // `_guard` is a normal field, so it is released after this `Drop` returns
            unsafe {
                ManuallyDrop::drop(&mut self.device);
            }
        }
    }

    impl Deref for TestDevice {
        type Target = Device;

        fn deref(&self) -> &Self::Target {
            &self.device
        }
    }

    fn test_device_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();

        LOCK.get_or_init(|| Mutex::new(()))
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

    fn subpass_dependencies_for_accesses(
        previous: AccessType,
        current: AccessType,
    ) -> Vec<SubpassDependency> {
        let pass = CommandData {
            execs: vec![
                exec_with_buffer_access(previous),
                exec_with_buffer_access(current),
            ],

            #[cfg(debug_assertions)]
            name: None,

            stream_scope_id: None,
            tracking: Default::default(),
        };

        Submission::build_subpass_dependencies(&pass, &ExternalRenderPassAccessHistory::new(1))
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

    fn depth_attachment_dependencies(
        previous_load: LoadOp<vk::ClearDepthStencilValue>,
        previous_store: StoreOp,
        current_load: LoadOp<vk::ClearDepthStencilValue>,
        current_store: StoreOp,
    ) -> Vec<SubpassDependency> {
        let pass = CommandData {
            execs: vec![
                depth_attachment_exec(previous_load, previous_store),
                depth_attachment_exec(current_load, current_store),
            ],

            #[cfg(debug_assertions)]
            name: None,

            stream_scope_id: None,
            tracking: Default::default(),
        };

        Submission::build_subpass_dependencies(&pass, &ExternalRenderPassAccessHistory::new(1))
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
            },
            cmds: cmds.to_vec(),
            ..Default::default()
        }
    }

    #[test]
    fn image_execution_discard_only_when_previous_access_is_nothing() {
        assert!(super::image_execution_discard_contents(AccessType::Nothing));
        assert!(!super::image_execution_discard_contents(
            AccessType::TransferRead
        ));
        assert!(!super::image_execution_discard_contents(
            AccessType::TransferWrite
        ));
        assert!(!super::image_execution_discard_contents(
            AccessType::ColorAttachmentReadWrite
        ));
    }

    #[test]
    fn image_layout_transition_discard_keeps_attachment_write_policy() {
        assert!(super::image_layout_transition_discard_contents(
            AccessType::Nothing,
            AccessType::TransferWrite,
        ));
        assert!(super::image_layout_transition_discard_contents(
            AccessType::TransferRead,
            AccessType::TransferWrite,
        ));
        assert!(!super::image_layout_transition_discard_contents(
            AccessType::TransferWrite,
            AccessType::ColorAttachmentReadWrite,
        ));
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

    #[test]
    fn command_access_index_includes_read_and_write_accesses() {
        let cmds = vec![
            command_with_accesses(&[(0, AccessType::TransferRead)]),
            command_with_accesses(&[(1, AccessType::TransferWrite)]),
            command_with_accesses(&[(1, AccessType::TransferRead)]),
            command_with_accesses(&[(1, AccessType::TransferWrite)]),
        ];
        let mut access_index = CommandAccessIndex::default();

        access_index.update_from_cmds(&cmds, 2);

        assert_eq!(access_index.cmds_by_node[0], vec![0]);
        assert_eq!(access_index.cmds_by_node[1], vec![1, 2, 3]);
        assert_eq!(access_index.accessed_nodes_by_cmd[0], vec![0]);
        assert_eq!(access_index.accessed_nodes_by_cmd[1], vec![1]);
        assert_eq!(access_index.accessed_nodes_by_cmd[2], vec![1]);
        assert_eq!(access_index.accessed_nodes_by_cmd[3], vec![1]);
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

        access_index.update_from_cmds(&cmds, 2);

        assert_eq!(access_index.cmds_by_node[0], vec![0, 1]);
        assert_eq!(access_index.cmds_by_node[1], vec![0, 1]);
        assert_eq!(access_index.accessed_nodes_by_cmd[0], vec![0, 1]);
        assert_eq!(access_index.accessed_nodes_by_cmd[1], vec![0, 1]);
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
        };
        let mut schedule = Schedule {
            access_index,
            ..Default::default()
        };

        super::schedule_dependency_cmds_before_target_access(1, 1, &mut schedule);

        assert_eq!(schedule.cmds, vec![0]);
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
        access_index.update_from_cmds(&cmds, 2);
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
    fn queue_ownership_release_groups_group_by_source_queue() {
        use super::{image_subresource_range_eq, queue_ownership_release_group};

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

        queue_ownership_release_group(&mut submission.queue_ownership_release_groups, 1, 2)
            .images
            .push((image, vk::ImageLayout::GENERAL, range_a));
        queue_ownership_release_group(&mut submission.queue_ownership_release_groups, 1, 2)
            .images
            .push((image, vk::ImageLayout::GENERAL, range_b));
        queue_ownership_release_group(&mut submission.queue_ownership_release_groups, 4, 5)
            .images
            .push((image, vk::ImageLayout::GENERAL, range_a));

        let mut groups = submission.queue_ownership_release_groups;
        sort_queue_ownership_release_groups(&mut groups);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].images.len(), 2);
        assert_eq!(groups[1].images.len(), 1);
        assert_eq!(groups[0].images[0].0, image);
        assert!(image_subresource_range_eq(groups[0].images[0].2, range_a));
    }

    #[test]
    fn barrier_transfer_ranges_only_marks_overlapping_ranges() {
        use super::{image_barrier_transfer_ranges, image_subresource_range_eq};

        let range_a = color_subresource_range(0..1, 0..1);
        let range_b = color_subresource_range(1..2, 0..1);
        let transfers = [ImageQueueOwnershipTransfer {
            src_queue_family_index: 1,
            src_queue_index: 2,
            dst_queue_family_index: 3,
            layout: vk::ImageLayout::GENERAL,
            range: range_a,
        }];

        let ranges = image_barrier_transfer_ranges(&transfers, color_subresource_range(0..2, 0..1))
            .collect::<Vec<_>>();

        assert_eq!(ranges.len(), 2);
        assert!(image_subresource_range_eq(ranges[0].0, range_a));
        assert_eq!(
            ranges[0].1.map(|transfer| (
                transfer.src_queue_family_index,
                transfer.src_queue_index,
                transfer.dst_queue_family_index,
            )),
            Some((1, 2, 3))
        );
        assert!(image_subresource_range_eq(ranges[1].0, range_b));
        assert!(ranges[1].1.is_none());
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn track_pending_transfers_only_collects_touched_subresources() -> Result<(), DriverError> {
        let device = test_device()?;
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
        assert!(super::image_subresource_range_eq(
            transfers[0].range,
            range_a
        ));
        let ranges = &submission.exclusive_image_ranges[&image.index()];
        let mut ranges = ranges.clone();
        sort_image_subresource_ranges(&mut ranges);
        assert_eq!(ranges.len(), 1);
        assert!(super::image_subresource_range_eq(ranges[0], range_a));

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn track_pending_transfers_only_collects_touched_buffer_ranges() -> Result<(), DriverError> {
        let device = test_device()?;
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
    fn repeated_partial_recording_does_not_duplicate_buffer_ownership_transfer()
    -> Result<(), DriverError> {
        let device = test_device()?;
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
    fn partial_recording_transfers_only_unclaimed_buffer_overlap() -> Result<(), DriverError> {
        let device = test_device()?;
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
    #[ignore = "requires Vulkan device"]
    fn repeated_partial_recording_does_not_duplicate_image_ownership_transfer()
    -> Result<(), DriverError> {
        let device = test_device()?;
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
            .map(|(_, _, range)| *range)
            .collect::<Vec<_>>();
        simulate_partial_transfer_discovery(&mut submission, &schedule, 3, &mut ownership);

        let released_ranges = submission
            .queue_ownership_release_groups
            .iter()
            .flat_map(|group| group.images.iter())
            .map(|(_, _, range)| *range)
            .collect::<Vec<_>>();
        assert_eq!(released_ranges.len(), first_released_ranges.len());
        assert!(
            released_ranges
                .iter()
                .zip(&first_released_ranges)
                .all(|(&lhs, &rhs)| super::image_subresource_range_eq(lhs, rhs)),
            "released ranges changed: {first_released_ranges:?} -> {released_ranges:?}"
        );
        assert_eq!(submission.exclusive_image_ranges[&image.index()].len(), 1);
        assert!(super::image_subresource_range_eq(
            submission.exclusive_image_ranges[&image.index()][0],
            range
        ));

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn track_pending_transfers_keeps_exclusive_owner_without_known_layout()
    -> Result<(), DriverError> {
        let device = test_device()?;
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
        assert!(super::image_subresource_range_eq(
            transfers[0].range,
            range_a
        ));
        assert_eq!(transfers[0].layout, vk::ImageLayout::UNDEFINED);

        let ranges = &submission.exclusive_image_ranges[&image.index()];
        let mut ranges = ranges.clone();
        sort_image_subresource_ranges(&mut ranges);
        assert_eq!(ranges.len(), 1);
        assert!(super::image_subresource_range_eq(ranges[0], range_a));

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn recorded_submission_attach_updates_only_touched_subresources() -> Result<(), DriverError> {
        let device = test_device()?;
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
    #[ignore = "requires Vulkan device"]
    fn recorded_submission_attach_updates_only_touched_buffer_ranges() -> Result<(), DriverError> {
        let device = test_device()?;
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
    fn reorder_scheduled_cmds_preserves_both_branches_before_join() {
        let mut schedule =
            schedule_with_access_index(&[0, 1, 2], &[&[0, 2], &[1, 2]], &[&[0], &[1], &[0, 1]]);

        schedule.reorder_cmds(3);

        assert_eq!(schedule.cmds, vec![0, 1, 2]);
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

        assert!(check_queue_submit_args(&waits, &signals).is_ok());
    }

    #[test]
    fn legacy_submit_rejects_precise_wait_stage_masks() {
        let waits = [SemaphoreSubmitInfo {
            semaphore: vk::Semaphore::null(),
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            value: 0,
        }];

        assert!(matches!(
            check_queue_submit_args(&waits, &[]),
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
            check_queue_submit_args(&waits, &[]),
            Err(DriverError::Unsupported)
        ));
    }

    fn test_device() -> Result<TestDevice, DriverError> {
        let guard = test_device_lock()
            .lock()
            .expect("poisoned test device lock");
        let device = Device::create(DeviceInfo::default())?;

        Ok(TestDevice {
            _guard: guard,
            device: ManuallyDrop::new(device),
        })
    }

    fn test_debug_device() -> Result<TestDevice, DriverError> {
        let guard = test_device_lock()
            .lock()
            .expect("poisoned test device lock");
        let device = Device::create(DeviceInfo::builder().debug(true).build())?;

        Ok(TestDevice {
            _guard: guard,
            device: ManuallyDrop::new(device),
        })
    }

    fn init_validation_test_logging() {
        static INIT: OnceLock<()> = OnceLock::new();

        INIT.get_or_init(|| {
            unsafe {
                set_var("RUST_LOG", "trace");
                set_var("VK_GRAPH_SKIP_VALIDATION_PARK", "1");
            }

            let _ = pretty_env_logger::try_init();
        });
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

    #[test]
    #[ignore = "requires Vulkan device"]
    fn submission_record_all_consumes_single_pass_graph() -> Result<(), DriverError> {
        let device = test_device()?;
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
    fn submission_record_nodes_consumes_requested_outputs() -> Result<(), DriverError> {
        let device = test_device()?;
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
    #[ignore = "requires Vulkan device"]
    fn submission_record_can_be_reused() -> Result<(), DriverError> {
        let device = test_device()?;
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
    fn accel_struct_mixed_accesses_preserve_all_stage_bits() -> Result<(), DriverError> {
        let device = test_device()?;
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
    #[ignore = "requires Vulkan validation layers; inspect validation output"]
    fn submission_external_subpass_dependency_validation_repro() -> Result<(), DriverError> {
        init_validation_test_logging();

        let device = test_debug_device()?;
        let mut pool = HashPool::new(&device);
        let pipeline = test_triangle_pipeline(&device)?;
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
    fn external_subpass_dependency_targets_first_subpass_consumer() -> Result<(), DriverError> {
        let device = test_device()?;
        let pipeline = test_triangle_pipeline(&device)?;
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
            ExternalRenderPassAccessHistory::new(submission.graph.resources.len());
        external_access_history.record_cmd(&submission.graph.cmds[0]);

        let dependencies = Submission::build_subpass_dependencies(
            &submission.graph.cmds[1],
            &external_access_history,
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
    #[ignore = "requires Vulkan device"]
    fn color_input_attachment_dependencies_use_fragment_shader_input_reads()
    -> Result<(), DriverError> {
        let device = test_device()?;
        let (pipeline_a, pipeline_b) = test_input_attachment_pipelines(&device)?;
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
            &ExternalRenderPassAccessHistory::new(submission.graph.resources.len()),
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
    #[ignore = "requires Vulkan device"]
    fn color_attachment_load_dependencies_avoid_invalid_stage_access_pairs()
    -> Result<(), DriverError> {
        let device = test_device()?;
        let pipeline = test_triangle_pipeline(&device)?;
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
            &ExternalRenderPassAccessHistory::new(submission.graph.resources.len()),
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
        assert_no_invalid_attachment_stage_access_pairs(dep);
        assert_attachment_read_stage_mappings(dep);

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn color_attachment_read_dependencies_avoid_invalid_stage_access_pairs()
    -> Result<(), DriverError> {
        let device = test_device()?;
        let pipeline = test_triangle_pipeline(&device)?;
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
            &ExternalRenderPassAccessHistory::new(submission.graph.resources.len()),
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
        assert_no_invalid_attachment_stage_access_pairs(dep);
        assert_attachment_read_stage_mappings(dep);

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn color_attachment_read_to_write_dependencies_avoid_invalid_stage_access_pairs()
    -> Result<(), DriverError> {
        let device = test_device()?;
        let pipeline = test_triangle_pipeline(&device)?;
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
            &ExternalRenderPassAccessHistory::new(submission.graph.resources.len()),
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
        assert_no_invalid_attachment_stage_access_pairs(dep);
        assert_attachment_read_stage_mappings(dep);

        Ok(())
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn depth_attachment_load_dependencies_avoid_invalid_stage_access_pairs()
    -> Result<(), DriverError> {
        let device = test_device()?;
        let pipeline = test_triangle_pipeline(&device)?;
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
            &ExternalRenderPassAccessHistory::new(submission.graph.resources.len()),
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
        assert_no_invalid_attachment_stage_access_pairs(dep);
        assert_attachment_read_stage_mappings(dep);

        Ok(())
    }

    #[test]
    fn depth_attachment_read_to_write_dependency_includes_late_read_stage() {
        let dependencies = depth_attachment_dependencies(
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
        let dependencies = depth_attachment_dependencies(
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
    fn subpass_dependency_matches_all_graphics_source_stage() {
        let dependencies = subpass_dependencies_for_accesses(
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
    fn subpass_dependency_matches_all_graphics_destination_stage() {
        let dependencies = subpass_dependencies_for_accesses(
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
    fn record_subpass_dependency_preserves_dst_access_for_unmatched_stages() {
        let mut dependencies = std::collections::BTreeMap::new();
        let mut current = PipelineStageAccessFlags {
            stage_flags: vk::PipelineStageFlags::VERTEX_SHADER
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_flags: vk::AccessFlags::SHADER_READ,
        };

        assert!(!Submission::record_subpass_dependency(
            &mut dependencies,
            0,
            2,
            PipelineStageAccessFlags {
                stage_flags: vk::PipelineStageFlags::VERTEX_SHADER,
                access_flags: vk::AccessFlags::SHADER_READ,
            },
            current.stage_flags,
            &mut current,
        ));
        assert!(Submission::record_subpass_dependency(
            &mut dependencies,
            1,
            2,
            PipelineStageAccessFlags {
                stage_flags: vk::PipelineStageFlags::FRAGMENT_SHADER,
                access_flags: vk::AccessFlags::SHADER_READ,
            },
            current.stage_flags,
            &mut current,
        ));

        let dep = dependencies
            .get(&(1, 2))
            .expect("missing dependency for later matched stage");
        assert!(
            dep.dst_access_mask.contains(vk::AccessFlags::SHADER_READ),
            "later matched stage should retain destination access mask"
        );
    }

    #[test]
    fn record_subpass_dependency_ignores_non_overlapping_stage() {
        let mut dependencies = std::collections::BTreeMap::new();
        let mut current = PipelineStageAccessFlags {
            stage_flags: vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_flags: vk::AccessFlags::SHADER_READ,
        };

        assert!(!Submission::record_subpass_dependency(
            &mut dependencies,
            0,
            1,
            PipelineStageAccessFlags {
                stage_flags: vk::PipelineStageFlags::VERTEX_SHADER,
                access_flags: vk::AccessFlags::SHADER_WRITE,
            },
            current.stage_flags,
            &mut current,
        ));

        assert!(dependencies.is_empty());
        assert_eq!(current.stage_flags, vk::PipelineStageFlags::FRAGMENT_SHADER);
        assert_eq!(current.access_flags, vk::AccessFlags::SHADER_READ);
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
        let dependencies =
            Submission::build_subpass_dependencies(&pass, &ExternalRenderPassAccessHistory::new(1));
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
    fn accel_struct_canonical_accesses_preserves_mixed_slice_accesses() {
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
            Submission::accel_struct_canonical_accesses(&accesses, &mut scratch),
            &[
                AccessType::AccelerationStructureBuildRead,
                AccessType::RayTracingShaderReadAccelerationStructure,
            ],
            "mixed acceleration-structure slices should preserve all accesses for next-state tracking"
        );
    }
}
