//! Submission and recording types.
//!
//! This module contains the execution-facing types produced by [`Graph::finalize`].
//!
//! Typical usage starts with a [`Submission`], which represents a finalized graph that has not yet
//! been bound to a command buffer:
//!
//! - Use [`Submission::queue_submit`] for the one-shot path that allocates, records, and submits a
//!   command buffer internally.
//! - Use [`Submission::record`], [`Submission::record_resource`], or
//!   [`Submission::record_resource_dependencies`] to bind the submission to an existing command
//!   buffer and obtain a [`RecordedSubmission`].
//!
//! A [`RecordedSubmission`] keeps the remaining graph work paired with the command buffer it was
//! recorded into. This typestate prevents recording with one command buffer and accidentally
//! submitting with another.
//!
//! [`Graph::finalize`]: crate::Graph::finalize

use {
    super::{
        AnyResource, Attachment, CommandData, ExecutionPipeline, Graph, Node, NodeIndex,
        cmd::{SubresourceAccess, SubresourceRange},
    },
    crate::{
        driver::{
            AttachmentInfo, AttachmentRef, Descriptor, DescriptorInfo, DescriptorSet, DriverError,
            FramebufferAttachmentImageInfo, FramebufferInfo, SubpassDependency, SubpassInfo,
            accel_struct::AccelerationStructure,
            buffer::Buffer,
            cmd_buf::{CommandBuffer, CommandBufferInfo},
            descriptor_set::{DescriptorPool, DescriptorPoolInfo},
            device::Device,
            format_aspect_mask,
            graphic::{DepthStencilInfo, GraphicsPipeline},
            image::{DenseAccess, Image},
            initial_image_layout_access, is_read_access, is_write_access, pack_queue,
            pipeline_stage_access_flags,
            render_pass::{RenderPass, RenderPassInfo},
            unpack_queue,
        },
        pool::{Lease, Pool},
    },
    ash::vk,
    fixedbitset::FixedBitSet,
    log::{
        Level::{Debug, Trace},
        debug, log_enabled, trace, warn,
    },
    std::{
        cell::RefCell,
        collections::{BTreeMap, HashMap, VecDeque},
        iter::repeat_n,
        ops::Range,
        slice,
        sync::atomic::Ordering,
    },
    vk_sync::{
        AccessType, BufferBarrier, GlobalBarrier, ImageBarrier, ImageLayout, cmd::pipeline_barrier,
    },
};

#[cfg(not(feature = "checked"))]
use std::hint::unreachable_unchecked;

const fn image_access_layout(access: AccessType) -> ImageLayout {
    if matches!(access, AccessType::Present | AccessType::ComputeShaderWrite) {
        ImageLayout::General
    } else {
        ImageLayout::Optimal
    }
}

/// Maps the current access type to the concrete Vulkan image layout,
/// replicating the vk_sync layout-selection logic used during barrier construction.
fn access_type_to_layout(access: AccessType) -> vk::ImageLayout {
    match access {
        // ImageLayout::Optimal → use the layout from AccessInfo
        AccessType::ColorAttachmentRead
        | AccessType::ColorAttachmentReadWrite
        | AccessType::ColorAttachmentWrite => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        AccessType::DepthStencilAttachmentRead => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        AccessType::DepthStencilAttachmentReadWrite | AccessType::DepthStencilAttachmentWrite => {
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        }
        AccessType::DepthAttachmentWriteStencilReadOnly => {
            vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL
        }
        AccessType::StencilAttachmentWriteDepthReadOnly => {
            vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL
        }
        AccessType::TransferRead => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        AccessType::TransferWrite => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        // Shader reads of sampled images / input attachments
        AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::FragmentShaderReadColorInputAttachment
        | AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::TessellationControlShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::TessellationEvaluationShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::GeometryShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::MeshShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::TaskShaderReadSampledImageOrUniformTexelBuffer => {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        }
        AccessType::FragmentShaderReadDepthStencilInputAttachment => {
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
        }
        // ImageLayout::General → GENERAL (or PRESENT_SRC_KHR for Present)
        AccessType::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        // Everything else → GENERAL (safe fallback, covers ComputeShaderWrite,
        // AnyShaderWrite, HostRead/Write, etc.)
        _ => vk::ImageLayout::GENERAL,
    }
}

struct ReleaseGroup {
    old_fam: u32,
    old_idx: u32,
    images: Vec<(vk::Image, AccessType, vk::ImageSubresourceRange)>,
}

#[derive(Debug)]
struct ReleaseBundle {
    #[allow(dead_code)]
    cmd_buf: Lease<CommandBuffer>,

    semaphore: vk::Semaphore,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingTransfer {
    src_access: AccessType,
    src_queue_family_index: u32,
    src_queue_index: u32,
    dst_queue_family_index: u32,
}

#[derive(Default)]
struct AccessIndex {
    passes_by_node: Vec<Vec<usize>>,
    read_nodes_by_pass: Vec<Vec<usize>>,
}

impl AccessIndex {
    #[profiling::function]
    fn dependent_nodes(&self, pass_idx: usize) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.read_nodes_by_pass[pass_idx].iter().copied()
    }

    #[profiling::function]
    fn dependent_passes(
        &self,
        node_idx: usize,
        end_pass_idx: usize,
    ) -> impl Iterator<Item = usize> + '_ {
        let passes = &self.passes_by_node[node_idx];
        let end_idx = passes.partition_point(|&pass_idx| pass_idx < end_pass_idx);

        passes[..end_idx].iter().rev().copied()
    }

    #[profiling::function]
    fn interdependent_passes(
        &self,
        pass_idx: usize,
        end_pass_idx: usize,
    ) -> impl Iterator<Item = usize> + '_ {
        self.dependent_nodes(pass_idx)
            .flat_map(move |node_idx| self.dependent_passes(node_idx, end_pass_idx))
    }

    fn update(&mut self, graph: &Graph, end_pass_idx: usize) {
        let binding_count = graph.resources.len();

        self.passes_by_node.clear();
        self.passes_by_node.resize_with(binding_count, Vec::new);

        self.read_nodes_by_pass.clear();
        self.read_nodes_by_pass.resize_with(end_pass_idx, Vec::new);

        thread_local! {
            static SEEN_NODES: RefCell<(Vec<bool>, Vec<bool>)> = Default::default();
        }

        SEEN_NODES.with_borrow_mut(|(seen_nodes, seen_reads)| {
            seen_nodes.truncate(binding_count);
            seen_nodes.fill(false);
            seen_nodes.resize(binding_count, false);

            seen_reads.truncate(binding_count);
            seen_reads.fill(false);
            seen_reads.resize(binding_count, false);

            for (pass_idx, pass) in graph.cmds[0..end_pass_idx].iter().enumerate() {
                let read_nodes = &mut self.read_nodes_by_pass[pass_idx];

                for (node_idx, accesses) in pass.execs.iter().flat_map(|exec| exec.accesses.iter())
                {
                    if !seen_nodes[node_idx] {
                        self.passes_by_node[node_idx].push(pass_idx);
                        seen_nodes[node_idx] = true;
                    }

                    if !seen_reads[node_idx]
                        && is_read_access(accesses.first().expect("missing resource access").access)
                    {
                        read_nodes.push(node_idx);
                        seen_reads[node_idx] = true;
                    }
                }

                seen_nodes.fill(false);
                seen_reads.fill(false);
            }
        });
    }
}

#[derive(Clone, Copy)]
struct AccessInfo {
    access: vk::AccessFlags,
    stages: vk::PipelineStageFlags,
}

impl AccessInfo {
    fn new(access: AccessType) -> Self {
        let (mut stages, access) = pipeline_stage_access_flags(access);
        if stages.contains(vk::PipelineStageFlags::ALL_COMMANDS) {
            stages |= vk::PipelineStageFlags::ALL_GRAPHICS;
            stages &= !vk::PipelineStageFlags::ALL_COMMANDS;
        }

        Self { access, stages }
    }
}

#[derive(Default)]
struct RenderPassAccessHistory {
    accesses_by_node: Vec<Vec<AccessInfo>>,
}

impl RenderPassAccessHistory {
    fn new(node_count: usize) -> Self {
        let mut accesses_by_node = Vec::with_capacity(node_count);
        accesses_by_node.resize_with(node_count, Vec::new);

        Self { accesses_by_node }
    }

    fn accesses(&self, node_idx: usize) -> &[AccessInfo] {
        &self.accesses_by_node[node_idx]
    }

    fn record_pass(&mut self, pass: &CommandData) {
        for exec in &pass.execs {
            for (node_idx, accesses) in exec.accesses.iter() {
                self.accesses_by_node[node_idx]
                    .extend(accesses.iter().map(|access| AccessInfo::new(access.access)));
            }
        }
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
struct PhysicalPass {
    descriptor_pool: Option<Lease<DescriptorPool>>,
    exec_descriptor_sets: HashMap<usize, Vec<DescriptorSet>>,
    render_pass: Option<Lease<RenderPass>>,
}

impl PhysicalPass {
    /// # Panics
    ///
    /// Panics if the physical pass has no render pass.
    fn expect_render_pass_mut(&mut self) -> &mut Lease<RenderPass> {
        self.render_pass.as_mut().expect("missing render pass")
    }
}

impl Drop for PhysicalPass {
    fn drop(&mut self) {
        self.exec_descriptor_sets.clear();
        self.descriptor_pool = None;
    }
}

/// A finalized graph execution plan.
///
/// `Submission` owns the remaining commands of a [`Graph`] after [`Graph::finalize`] has ended the
/// graph-building phase. It supports two execution styles:
///
/// - [`Submission::queue_submit`] for a one-shot submission path.
/// - [`Submission::record`], [`Submission::record_resource`], or
///   [`Submission::record_resource_dependencies`] for explicit command-buffer recording, returning
///   a [`RecordedSubmission`].
#[derive(Debug)]
pub struct Submission {
    exclusive_image_indices: FixedBitSet,
    graph: Graph,
    physical_passes: Vec<PhysicalPass>,
    pending_transfers: HashMap<vk::Image, PendingTransfer>,
}

/// A [`Submission`] bound to a specific command buffer for explicit recording and submission.
#[derive(Debug)]
#[read_only::cast]
pub struct RecordedSubmission<'a> {
    /// The command buffer bound to this recorded submission.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub cmd_buf: &'a mut CommandBuffer,

    submission: Submission,
}

impl Submission {
    pub(super) fn new(graph: Graph) -> Self {
        let physical_passes = Vec::with_capacity(graph.cmds.len());

        Self {
            exclusive_image_indices: FixedBitSet::with_capacity(graph.resources.len()),
            graph,
            physical_passes,
            pending_transfers: HashMap::new(),
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

    fn record_subpass_dependency(
        dependencies: &mut BTreeMap<(usize, usize), SubpassDependency>,
        src_subpass: usize,
        dst_subpass: usize,
        prev: AccessInfo,
        dst_stage_mask: vk::PipelineStageFlags,
        curr: &mut AccessInfo,
    ) -> bool {
        let common_stages = curr.stages & prev.stages;
        if common_stages.is_empty() {
            return false;
        }

        let dep = dependencies
            .entry((src_subpass, dst_subpass))
            .or_insert_with(|| SubpassDependency::new(src_subpass as _, dst_subpass as _));

        dep.src_stage_mask |= common_stages;
        dep.src_access_mask |= prev.access;
        dep.dst_stage_mask |= dst_stage_mask;
        dep.dst_access_mask |= curr.access;

        if Self::is_framebuffer_space(prev.stages | curr.stages) {
            dep.dependency_flags |= vk::DependencyFlags::BY_REGION;
        }

        curr.stages &= !common_stages;
        curr.access &= !prev.access;

        curr.stages.is_empty()
    }

    #[profiling::function]
    fn allow_merge_passes(lhs: &CommandData, rhs: &CommandData) -> bool {
        fn first_graphic_pipeline(pass: &CommandData) -> Option<&GraphicsPipeline> {
            pass.execs
                .first()
                .and_then(|exec| exec.pipeline.as_ref().map(ExecutionPipeline::as_graphic))
                .flatten()
        }

        fn is_multiview(view_mask: u32) -> bool {
            view_mask != 0
        }

        let lhs_pipeline = first_graphic_pipeline(lhs);
        if lhs_pipeline.is_none() {
            trace!("  {} is not graphic", lhs.name());

            return false;
        }

        let rhs_pipeline = first_graphic_pipeline(rhs);
        if rhs_pipeline.is_none() {
            trace!("  {} is not graphic", rhs.name());

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

        // PassRef makes sure this never happens
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
            for (attachment_idx, lhs_attachment) in lhs
                .color_attachments
                .iter()
                .chain(lhs.color_loads.iter())
                .chain(lhs.color_stores.iter())
                .chain(
                    lhs.color_clears
                        .iter()
                        .map(|(attachment_idx, (attachment, _))| (attachment_idx, attachment)),
                )
                .chain(
                    lhs.color_resolves
                        .iter()
                        .map(|(attachment_idx, (attachment, _))| (attachment_idx, attachment)),
                )
            {
                let rhs_attachment = rhs
                    .color_attachments
                    .get(attachment_idx)
                    .or_else(|| rhs.color_loads.get(attachment_idx))
                    .or_else(|| rhs.color_stores.get(attachment_idx))
                    .or_else(|| {
                        rhs.color_clears
                            .get(attachment_idx)
                            .map(|(attachment, _)| attachment)
                    })
                    .or_else(|| {
                        rhs.color_resolves
                            .get(attachment_idx)
                            .map(|(attachment, _)| attachment)
                    });

                if !Attachment::are_compatible(Some(*lhs_attachment), rhs_attachment.copied()) {
                    trace!("  incompatible color attachments");

                    return false;
                }

                common_color_attachment = true;
            }

            // Compare depth/stencil attachments for compatibility
            let lhs_depth_stencil = lhs
                .depth_stencil_attachment
                .or(lhs.depth_stencil_load)
                .or(lhs.depth_stencil_store)
                .or_else(|| lhs.depth_stencil_resolve.map(|(attachment, ..)| attachment))
                .or_else(|| lhs.depth_stencil_clear.map(|(attachment, _)| attachment));

            let rhs_depth_stencil = rhs
                .depth_stencil_attachment
                .or(rhs.depth_stencil_load)
                .or(rhs.depth_stencil_store)
                .or_else(|| rhs.depth_stencil_resolve.map(|(attachment, ..)| attachment))
                .or_else(|| rhs.depth_stencil_clear.map(|(attachment, _)| attachment));

            if !Attachment::are_compatible(lhs_depth_stencil, rhs_depth_stencil) {
                trace!("  incompatible depth/stencil attachments");

                return false;
            }

            common_depth_attachment |= lhs_depth_stencil.is_some() && rhs_depth_stencil.is_some();
        }

        // Keep color and depth on tile.
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

        // No reason to merge, so don't.
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
        physical_pass: &mut PhysicalPass,
        render_area: vk::Rect2D,
    ) -> Result<(), DriverError> {
        trace!("  begin render pass");

        let render_pass = physical_pass.expect_render_pass_mut();
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
                for (attachment_idx, (attachment, clear_value)) in &exec.color_clears {
                    let attachment_image = &mut attachments[*attachment_idx as usize];
                    if let Err(idx) = attachment_image
                        .view_formats
                        .binary_search(&attachment.format)
                    {
                        clear_values[*attachment_idx as usize] = vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: *clear_value,
                            },
                        };

                        let image = Self::expect_attachment_image(bindings, attachment);

                        attachment_image.flags = image.info.flags;
                        attachment_image.usage = image.info.usage;
                        attachment_image.width = image.info.width >> attachment.base_mip_level;
                        attachment_image.height = image.info.height >> attachment.base_mip_level;
                        attachment_image.layer_count = attachment.array_layer_count;
                        attachment_image.view_formats.insert(idx, attachment.format);

                        image_views[*attachment_idx as usize] =
                            Image::view(image, attachment.image_view_info(image.info))?;
                    }
                }

                for (attachment_idx, attachment) in exec
                    .color_attachments
                    .iter()
                    .chain(&exec.color_loads)
                    .chain(&exec.color_stores)
                    .chain(exec.color_resolves.iter().map(
                        |(dst_attachment_idx, (attachment, _))| (dst_attachment_idx, attachment),
                    ))
                {
                    let attachment_image = &mut attachments[*attachment_idx as usize];
                    if let Err(idx) = attachment_image
                        .view_formats
                        .binary_search(&attachment.format)
                    {
                        let image = Self::expect_attachment_image(bindings, attachment);

                        attachment_image.flags = image.info.flags;
                        attachment_image.usage = image.info.usage;
                        attachment_image.width = image.info.width >> attachment.base_mip_level;
                        attachment_image.height = image.info.height >> attachment.base_mip_level;
                        attachment_image.layer_count = attachment.array_layer_count;
                        attachment_image.view_formats.insert(idx, attachment.format);

                        image_views[*attachment_idx as usize] =
                            Image::view(image, attachment.image_view_info(image.info))?;
                    }
                }

                if let Some((attachment, clear_value)) = &exec.depth_stencil_clear {
                    let attachment_idx =
                        attachments.len() - 1 - exec.depth_stencil_resolve.is_some() as usize;
                    let attachment_image = &mut attachments[attachment_idx];
                    if let Err(idx) = attachment_image
                        .view_formats
                        .binary_search(&attachment.format)
                    {
                        clear_values[attachment_idx] = vk::ClearValue {
                            depth_stencil: *clear_value,
                        };

                        let image = Self::expect_attachment_image(bindings, attachment);

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

                if let Some(attachment) = exec
                    .depth_stencil_attachment
                    .or(exec.depth_stencil_load)
                    .or(exec.depth_stencil_store)
                {
                    let attachment_idx =
                        attachments.len() - 1 - exec.depth_stencil_resolve.is_some() as usize;
                    let attachment_image = &mut attachments[attachment_idx];
                    if let Err(idx) = attachment_image
                        .view_formats
                        .binary_search(&attachment.format)
                    {
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

                if let Some(attachment) = exec
                    .depth_stencil_resolve
                    .map(|(attachment, ..)| attachment)
                {
                    let attachment_idx = attachments.len() - 1;
                    let attachment_image = &mut attachments[attachment_idx];
                    if let Err(idx) = attachment_image
                        .view_formats
                        .binary_search(&attachment.format)
                    {
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
        physical_pass: &PhysicalPass,
        exec_idx: usize,
    ) {
        if let Some(exec_descriptor_sets) = physical_pass.exec_descriptor_sets.get(&exec_idx) {
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
        cmd_buf: &mut CommandBuffer,
        physical_pass: &mut PhysicalPass,
        exec_idx: usize,
        pipeline: &mut ExecutionPipeline,
        depth_stencil: Option<DepthStencilInfo>,
    ) -> Result<(), DriverError> {
        if log_enabled!(Trace) {
            let (ty, name, vk_pipeline) = match pipeline {
                ExecutionPipeline::Compute(pipeline) => {
                    ("compute", pipeline.debug_name(), pipeline.handle())
                }
                ExecutionPipeline::Graphic(pipeline) => {
                    ("graphic", pipeline.debug_name(), vk::Pipeline::null())
                }
                ExecutionPipeline::RayTrace(pipeline) => {
                    ("ray tracing", pipeline.debug_name(), pipeline.handle())
                }
            };
            if let Some(name) = name {
                trace!("    bind {} pipeline {} ({:?})", ty, name, vk_pipeline);
            } else {
                trace!("    bind {} pipeline {:?}", ty, vk_pipeline);
            }
        }

        // We store a shared reference to this pipeline inside the command buffer!
        let pipeline_bind_point = pipeline.bind_point();
        let pipeline = match pipeline {
            ExecutionPipeline::Compute(pipeline) => pipeline.handle(),
            ExecutionPipeline::Graphic(pipeline) => RenderPass::pipeline_handle(
                physical_pass.expect_render_pass_mut(),
                pipeline,
                depth_stencil,
                exec_idx as _,
            )?,
            ExecutionPipeline::RayTrace(pipeline) => pipeline.handle(),
        };

        unsafe {
            cmd_buf
                .device
                .cmd_bind_pipeline(cmd_buf.handle, pipeline_bind_point, pipeline);
        }

        Ok(())
    }

    fn end_render_pass(&mut self, cmd: &CommandBuffer) {
        trace!("  end render pass");

        unsafe {
            cmd.device.cmd_end_render_pass(cmd.handle);
        }
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
        P: Pool<DescriptorPoolInfo, DescriptorPool>,
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
                            "unsupported descriptor type {:?} for pass {}",
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

        // Notice how all sets are big enough for any other set; TODO: efficiently dont

        // debug!("{:#?}", info);

        Ok(Some(pool.resource(info)?))
    }

    #[profiling::function]
    fn lease_render_pass<P>(
        &self,
        pool: &mut P,
        pass_idx: usize,
        external_access_history: &RenderPassAccessHistory,
    ) -> Result<Lease<RenderPass>, DriverError>
    where
        P: Pool<RenderPassInfo, RenderPass>,
    {
        let pass = &self.graph.cmds[pass_idx];
        let (mut color_attachment_count, mut depth_stencil_attachment_count) = (0, 0);
        for exec in &pass.execs {
            color_attachment_count = color_attachment_count
                .max(
                    exec.color_attachments
                        .keys()
                        .max()
                        .map(|attachment_idx| attachment_idx + 1)
                        .unwrap_or_default() as usize,
                )
                .max(
                    exec.color_clears
                        .keys()
                        .max()
                        .map(|attachment_idx| attachment_idx + 1)
                        .unwrap_or_default() as usize,
                )
                .max(
                    exec.color_loads
                        .keys()
                        .max()
                        .map(|attachment_idx| attachment_idx + 1)
                        .unwrap_or_default() as usize,
                )
                .max(
                    exec.color_resolves
                        .keys()
                        .max()
                        .map(|attachment_idx| attachment_idx + 1)
                        .unwrap_or_default() as usize,
                )
                .max(
                    exec.color_stores
                        .keys()
                        .max()
                        .map(|attachment_idx| attachment_idx + 1)
                        .unwrap_or_default() as usize,
                );
            let has_depth_stencil_attachment = exec.depth_stencil_attachment.is_some()
                || exec.depth_stencil_clear.is_some()
                || exec.depth_stencil_load.is_some()
                || exec.depth_stencil_store.is_some();
            let has_depth_stencil_resolve = exec.depth_stencil_resolve.is_some();

            depth_stencil_attachment_count = depth_stencil_attachment_count
                .max(has_depth_stencil_attachment as usize + has_depth_stencil_resolve as usize);
        }

        let attachment_count = color_attachment_count + depth_stencil_attachment_count;
        let mut attachments = Vec::with_capacity(attachment_count);
        attachments.resize_with(attachment_count, AttachmentInfo::default);

        let mut subpasses = Vec::<SubpassInfo>::with_capacity(pass.execs.len());

        {
            let mut color_set = vec![false; attachment_count];
            let mut depth_stencil_set = false;

            // Add load op attachments using the first executions
            for exec in &pass.execs {
                // Cleared color attachments
                for (attachment_idx, (cleared_attachment, _)) in &exec.color_clears {
                    let color_set = &mut color_set[*attachment_idx as usize];
                    if *color_set {
                        continue;
                    }

                    let attachment = &mut attachments[*attachment_idx as usize];
                    attachment.fmt = cleared_attachment.format;
                    attachment.sample_count = cleared_attachment.sample_count;
                    attachment.load_op = vk::AttachmentLoadOp::CLEAR;
                    attachment.initial_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                    *color_set = true;
                }

                // Loaded color attachments
                for (attachment_idx, loaded_attachment) in &exec.color_loads {
                    let color_set = &mut color_set[*attachment_idx as usize];
                    if *color_set {
                        continue;
                    }

                    let attachment = &mut attachments[*attachment_idx as usize];
                    attachment.fmt = loaded_attachment.format;
                    attachment.sample_count = loaded_attachment.sample_count;
                    attachment.load_op = vk::AttachmentLoadOp::LOAD;
                    attachment.initial_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                    *color_set = true;
                }

                // Cleared depth/stencil attachment
                if !depth_stencil_set {
                    if let Some((cleared_attachment, _)) = exec.depth_stencil_clear {
                        let attachment = &mut attachments[color_attachment_count];
                        attachment.fmt = cleared_attachment.format;
                        attachment.sample_count = cleared_attachment.sample_count;
                        attachment.initial_layout = if cleared_attachment
                            .aspect_mask
                            .contains(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                        {
                            attachment.load_op = vk::AttachmentLoadOp::CLEAR;
                            attachment.stencil_load_op = vk::AttachmentLoadOp::CLEAR;

                            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                        } else if cleared_attachment
                            .aspect_mask
                            .contains(vk::ImageAspectFlags::DEPTH)
                        {
                            attachment.load_op = vk::AttachmentLoadOp::CLEAR;

                            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
                        } else {
                            attachment.stencil_load_op = vk::AttachmentLoadOp::CLEAR;

                            vk::ImageLayout::STENCIL_ATTACHMENT_OPTIMAL
                        };
                        depth_stencil_set = true;
                    } else if let Some(loaded_attachment) = exec.depth_stencil_load {
                        // Loaded depth/stencil attachment
                        let attachment = &mut attachments[color_attachment_count];
                        attachment.fmt = loaded_attachment.format;
                        attachment.sample_count = loaded_attachment.sample_count;
                        attachment.initial_layout = if loaded_attachment
                            .aspect_mask
                            .contains(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                        {
                            attachment.load_op = vk::AttachmentLoadOp::LOAD;
                            attachment.stencil_load_op = vk::AttachmentLoadOp::LOAD;

                            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                        } else if loaded_attachment
                            .aspect_mask
                            .contains(vk::ImageAspectFlags::DEPTH)
                        {
                            attachment.load_op = vk::AttachmentLoadOp::LOAD;

                            vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL
                        } else {
                            attachment.stencil_load_op = vk::AttachmentLoadOp::LOAD;

                            vk::ImageLayout::STENCIL_READ_ONLY_OPTIMAL
                        };
                        depth_stencil_set = true;
                    } else if exec.depth_stencil_clear.is_some()
                        || exec.depth_stencil_store.is_some()
                    {
                        depth_stencil_set = true;
                    }
                }
            }
        }

        {
            let mut color_set = vec![false; attachment_count];
            let mut depth_stencil_set = false;
            let mut depth_stencil_resolve_set = false;

            // Add store op attachments using the last executions
            for exec in pass.execs.iter().rev() {
                // Resolved color attachments
                for (attachment_idx, (resolved_attachment, _)) in &exec.color_resolves {
                    let color_set = &mut color_set[*attachment_idx as usize];
                    if *color_set {
                        continue;
                    }

                    let attachment = &mut attachments[*attachment_idx as usize];
                    attachment.fmt = resolved_attachment.format;
                    attachment.sample_count = resolved_attachment.sample_count;
                    attachment.final_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                    *color_set = true;
                }

                // Stored color attachments
                for (attachment_idx, stored_attachment) in &exec.color_stores {
                    let color_set = &mut color_set[*attachment_idx as usize];
                    if *color_set {
                        continue;
                    }

                    let attachment = &mut attachments[*attachment_idx as usize];
                    attachment.fmt = stored_attachment.format;
                    attachment.sample_count = stored_attachment.sample_count;
                    attachment.store_op = vk::AttachmentStoreOp::STORE;
                    attachment.final_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                    *color_set = true;
                }

                // Stored depth/stencil attachment
                if !depth_stencil_set && let Some(stored_attachment) = exec.depth_stencil_store {
                    let attachment = &mut attachments[color_attachment_count];
                    attachment.fmt = stored_attachment.format;
                    attachment.sample_count = stored_attachment.sample_count;
                    attachment.final_layout = if stored_attachment
                        .aspect_mask
                        .contains(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                    {
                        attachment.store_op = vk::AttachmentStoreOp::STORE;
                        attachment.stencil_store_op = vk::AttachmentStoreOp::STORE;

                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                    } else if stored_attachment
                        .aspect_mask
                        .contains(vk::ImageAspectFlags::DEPTH)
                    {
                        attachment.store_op = vk::AttachmentStoreOp::STORE;

                        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
                    } else {
                        attachment.stencil_store_op = vk::AttachmentStoreOp::STORE;

                        vk::ImageLayout::STENCIL_ATTACHMENT_OPTIMAL
                    };
                    depth_stencil_set = true;
                }

                // Resolved depth/stencil attachment
                if !depth_stencil_resolve_set
                    && let Some((resolved_attachment, ..)) = exec.depth_stencil_resolve
                {
                    let attachment = attachments
                        .last_mut()
                        .expect("missing depth stencil resolve attachment");
                    attachment.fmt = resolved_attachment.format;
                    attachment.sample_count = resolved_attachment.sample_count;
                    attachment.final_layout = if resolved_attachment
                        .aspect_mask
                        .contains(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                    {
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                    } else if resolved_attachment
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
                attachment.initial_layout = attachment.final_layout;
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
                .expect_graphic();
            let mut subpass_info = SubpassInfo::with_capacity(attachment_count);

            // Add input attachments
            for attachment_idx in pipeline.inner.input_attachments.iter() {
                debug_assert!(
                    !exec.color_clears.contains_key(attachment_idx),
                    "cannot clear color attachment {attachment_idx} because it uses subpass input",
                );

                let exec_attachment = exec
                    .color_attachments
                    .get(attachment_idx)
                    .or_else(|| exec.color_loads.get(attachment_idx))
                    .or_else(|| exec.color_stores.get(attachment_idx))
                    .expect("missing input attachment");
                let is_random_access = exec.color_stores.contains_key(attachment_idx);
                subpass_info.input_attachments.push(AttachmentRef {
                    attachment: *attachment_idx,
                    aspect_mask: exec_attachment.aspect_mask,
                    layout: Self::attachment_layout(
                        exec_attachment.aspect_mask,
                        is_random_access,
                        true,
                    ),
                });

                // We should preserve the attachment in the previous subpasses as needed
                // (We're asserting that any input renderpasses are actually real subpasses
                // here with prior passes..)
                for prev_exec_idx in (0..exec_idx - 1).rev() {
                    let prev_exec = &pass.execs[prev_exec_idx];
                    if prev_exec.color_stores.contains_key(attachment_idx) {
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

            for attachment_idx in exec
                .color_attachments
                .keys()
                .chain(exec.color_clears.keys())
                .chain(exec.color_loads.keys())
                .chain(exec.color_stores.keys())
            {
                subpass_info.color_attachments[*attachment_idx as usize].attachment =
                    *attachment_idx;
            }

            // Set depth/stencil attachment
            if let Some(depth_stencil) = exec
                .depth_stencil_attachment
                .or(exec.depth_stencil_load)
                .or(exec.depth_stencil_store)
                .or_else(|| exec.depth_stencil_clear.map(|(attachment, _)| attachment))
            {
                let is_random_access = exec.depth_stencil_clear.is_some()
                    || exec.depth_stencil_load.is_some()
                    || exec.depth_stencil_store.is_some();
                subpass_info.depth_stencil_attachment = Some(AttachmentRef {
                    attachment: color_attachment_count as u32,
                    aspect_mask: depth_stencil.aspect_mask,
                    layout: Self::attachment_layout(
                        depth_stencil.aspect_mask,
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
            for (dst_attachment_idx, (resolved_attachment, src_attachment_idx)) in
                &exec.color_resolves
            {
                let is_input = subpass_info
                    .input_attachments
                    .iter()
                    .any(|input| input.attachment == *dst_attachment_idx);
                subpass_info.color_resolve_attachments[*src_attachment_idx as usize] =
                    AttachmentRef {
                        attachment: *dst_attachment_idx,
                        aspect_mask: resolved_attachment.aspect_mask,
                        layout: Self::attachment_layout(
                            resolved_attachment.aspect_mask,
                            true,
                            is_input,
                        ),
                    };
            }

            if let Some((
                resolved_attachment,
                dst_attachment_idx,
                depth_resolve_mode,
                stencil_resolve_mode,
            )) = exec.depth_stencil_resolve
            {
                subpass_info.depth_stencil_resolve_attachment = Some((
                    AttachmentRef {
                        attachment: dst_attachment_idx + 1,
                        aspect_mask: resolved_attachment.aspect_mask,
                        layout: Self::attachment_layout(
                            resolved_attachment.aspect_mask,
                            true,
                            false,
                        ),
                    },
                    depth_resolve_mode,
                    stencil_resolve_mode,
                ))
            }

            subpass_info.view_mask = exec.view_mask;
            subpass_info.correlated_view_mask = exec.correlated_view_mask;

            subpasses.push(subpass_info);
        }

        // Add dependencies
        let dependencies =
            {
                let mut dependencies = BTreeMap::new();
                let mut pass_access_history = HashMap::<NodeIndex, Vec<(usize, AccessInfo)>>::new();

                for (exec_idx, exec) in pass.execs.iter().enumerate() {
                    'accesses: for (node_idx, accesses) in exec.accesses.iter() {
                        let mut curr = AccessInfo::new(
                            accesses.first().expect("missing resource access").access,
                        );

                        if let Some(prev_accesses) = pass_access_history.get(&node_idx) {
                            for &(prev_exec_idx, prev) in prev_accesses.iter().rev() {
                                if Self::record_subpass_dependency(
                                    &mut dependencies,
                                    prev_exec_idx,
                                    exec_idx,
                                    prev,
                                    curr.stages,
                                    &mut curr,
                                ) {
                                    continue 'accesses;
                                }
                            }
                        }

                        for &prev in external_access_history.accesses(node_idx).iter().rev() {
                            if Self::record_subpass_dependency(
                                &mut dependencies,
                                vk::SUBPASS_EXTERNAL as usize,
                                exec_idx,
                                prev,
                                curr.stages.min(vk::PipelineStageFlags::ALL_GRAPHICS),
                                &mut curr,
                            ) {
                                continue 'accesses;
                            }
                        }

                        if !curr.stages.is_empty() {
                            let dep = dependencies
                                .entry((vk::SUBPASS_EXTERNAL as usize, exec_idx))
                                .or_insert_with(|| {
                                    SubpassDependency::new(vk::SUBPASS_EXTERNAL, exec_idx as _)
                                });

                            dep.src_stage_mask |= curr.stages;
                            dep.src_access_mask |= curr.access;
                            dep.dst_stage_mask |= vk::PipelineStageFlags::TOP_OF_PIPE;
                            dep.dst_access_mask =
                                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
                        }
                    }

                    for (node_idx, accesses) in exec.accesses.iter() {
                        let prev_accesses = pass_access_history.entry(node_idx).or_default();
                        prev_accesses.extend(
                            accesses
                                .iter()
                                .map(|access| (exec_idx, AccessInfo::new(access.access))),
                        );
                    }

                    // Look for attachments of this exec being read or written in other execs of the
                    // same pass
                    for (other_idx, other) in pass.execs[0..exec_idx].iter().enumerate() {
                        // Look for color attachments we're reading
                        for attachment_idx in exec.color_loads.keys() {
                            // Look for writes in the other exec
                            if other.color_clears.contains_key(attachment_idx)
                                || other.color_stores.contains_key(attachment_idx)
                                || other.color_resolves.contains_key(attachment_idx)
                            {
                                let dep = dependencies.entry((other_idx, exec_idx)).or_insert_with(
                                    || SubpassDependency::new(other_idx as _, exec_idx as _),
                                );

                                // Wait for ...
                                dep.src_stage_mask |=
                                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
                                dep.src_access_mask |= vk::AccessFlags::COLOR_ATTACHMENT_WRITE;

                                // ... before we:
                                dep.dst_stage_mask |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS;
                                dep.dst_access_mask |= vk::AccessFlags::COLOR_ATTACHMENT_READ;
                            }

                            // look for reads in the other exec
                            if other.color_loads.contains_key(attachment_idx) {
                                let dep = dependencies.entry((other_idx, exec_idx)).or_insert_with(
                                    || SubpassDependency::new(other_idx as _, exec_idx as _),
                                );

                                // Wait for ...
                                dep.src_stage_mask |= vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
                                dep.src_access_mask |= vk::AccessFlags::COLOR_ATTACHMENT_READ;

                                // ... before we:
                                dep.dst_stage_mask |= vk::PipelineStageFlags::FRAGMENT_SHADER;
                                dep.dst_access_mask |= vk::AccessFlags::COLOR_ATTACHMENT_READ;
                            }
                        }

                        // Look for a depth/stencil attachment read
                        if exec.depth_stencil_load.is_some() {
                            // Look for writes in the other exec
                            if other.depth_stencil_clear.is_some()
                                || other.depth_stencil_store.is_some()
                                || other.depth_stencil_resolve.is_some()
                            {
                                let dep = dependencies.entry((other_idx, exec_idx)).or_insert_with(
                                    || SubpassDependency::new(other_idx as _, exec_idx as _),
                                );

                                // Wait for ...
                                dep.src_stage_mask |= vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
                                dep.src_access_mask |=
                                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;

                                // ... before we:
                                dep.dst_stage_mask |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS;
                                dep.dst_access_mask |=
                                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
                            }

                            // TODO: Do we need to depend on a READ..READ between subpasses?
                            // look for reads in the other exec
                            if other.depth_stencil_load.is_some() {
                                let dep = dependencies.entry((other_idx, exec_idx)).or_insert_with(
                                    || SubpassDependency::new(other_idx as _, exec_idx as _),
                                );

                                // Wait for ...
                                dep.src_stage_mask |= vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
                                dep.src_access_mask |=
                                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;

                                // ... before we:
                                dep.dst_stage_mask |= vk::PipelineStageFlags::FRAGMENT_SHADER;
                                dep.dst_access_mask |=
                                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
                            }
                        }

                        // Look for color attachments we're writing
                        for (attachment_idx, aspect_mask) in
                            exec.color_clears
                                .iter()
                                .map(|(attachment_idx, (attachment, _))| {
                                    (*attachment_idx, attachment.aspect_mask)
                                })
                                .chain(exec.color_resolves.iter().map(
                                    |(dst_attachment_idx, (resolved_attachment, _))| {
                                        (*dst_attachment_idx, resolved_attachment.aspect_mask)
                                    },
                                ))
                                .chain(exec.color_stores.iter().map(
                                    |(attachment_idx, attachment)| {
                                        (*attachment_idx, attachment.aspect_mask)
                                    },
                                ))
                        {
                            let stage = match aspect_mask {
                                mask if mask.contains(vk::ImageAspectFlags::COLOR) => {
                                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                                }
                                mask if mask.intersects(
                                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                                ) =>
                                {
                                    vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                                }
                                _ => vk::PipelineStageFlags::ALL_GRAPHICS,
                            };

                            // Look for writes in the other exec
                            if other.color_clears.contains_key(&attachment_idx)
                                || other.color_stores.contains_key(&attachment_idx)
                                || other.color_resolves.contains_key(&attachment_idx)
                            {
                                let access = match aspect_mask {
                                    mask if mask.contains(vk::ImageAspectFlags::COLOR) => {
                                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                                    }
                                    mask if mask.intersects(
                                        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                                    ) =>
                                    {
                                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                                    }
                                    _ => {
                                        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
                                    }
                                };

                                let dep = dependencies.entry((other_idx, exec_idx)).or_insert_with(
                                    || SubpassDependency::new(other_idx as _, exec_idx as _),
                                );

                                // Wait for ...
                                dep.src_stage_mask |= stage;
                                dep.src_access_mask |= access;

                                // ... before we:
                                dep.dst_stage_mask |= stage;
                                dep.dst_access_mask |= access;
                            }

                            // look for reads in the other exec
                            if other.color_loads.contains_key(&attachment_idx) {
                                let (src_access, dst_access) = match aspect_mask {
                                    mask if mask.contains(vk::ImageAspectFlags::COLOR) => (
                                        vk::AccessFlags::COLOR_ATTACHMENT_READ,
                                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                                    ),
                                    mask if mask.intersects(
                                        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                                    ) =>
                                    {
                                        (
                                            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                                            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                                        )
                                    }
                                    _ => (
                                        vk::AccessFlags::MEMORY_READ
                                            | vk::AccessFlags::MEMORY_WRITE,
                                        vk::AccessFlags::MEMORY_READ
                                            | vk::AccessFlags::MEMORY_WRITE,
                                    ),
                                };

                                let dep = dependencies.entry((other_idx, exec_idx)).or_insert_with(
                                    || SubpassDependency::new(other_idx as _, exec_idx as _),
                                );

                                // Wait for ...
                                dep.src_stage_mask |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS;
                                dep.src_access_mask |= src_access;

                                // ... before we:
                                dep.dst_stage_mask |= stage;
                                dep.dst_access_mask |= dst_access;
                            }
                        }

                        // Look for a depth/stencil attachment write
                        if let Some(aspect_mask) = exec
                            .depth_stencil_clear
                            .map(|(attachment, _)| attachment.aspect_mask)
                            .or_else(|| {
                                exec.depth_stencil_store
                                    .map(|attachment| attachment.aspect_mask)
                            })
                            .or_else(|| {
                                exec.depth_stencil_resolve
                                    .map(|(attachment, ..)| attachment.aspect_mask)
                            })
                        {
                            let stage = match aspect_mask {
                                mask if mask.contains(vk::ImageAspectFlags::COLOR) => {
                                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                                }
                                mask if mask.intersects(
                                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                                ) =>
                                {
                                    vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                                }
                                _ => vk::PipelineStageFlags::ALL_GRAPHICS,
                            };

                            // Look for writes in the other exec
                            if other.depth_stencil_clear.is_some()
                                || other.depth_stencil_store.is_some()
                                || other.depth_stencil_resolve.is_some()
                            {
                                let access = match aspect_mask {
                                    mask if mask.contains(vk::ImageAspectFlags::COLOR) => {
                                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                                    }
                                    mask if mask.intersects(
                                        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                                    ) =>
                                    {
                                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                                    }
                                    _ => {
                                        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
                                    }
                                };

                                let dep = dependencies.entry((other_idx, exec_idx)).or_insert_with(
                                    || SubpassDependency::new(other_idx as _, exec_idx as _),
                                );

                                // Wait for ...
                                dep.src_stage_mask |= stage;
                                dep.src_access_mask |= access;

                                // ... before we:
                                dep.dst_stage_mask |= stage;
                                dep.dst_access_mask |= access;
                            }

                            // look for reads in the other exec
                            if other.depth_stencil_load.is_some() {
                                let (src_access, dst_access) = match aspect_mask {
                                    mask if mask.contains(vk::ImageAspectFlags::COLOR) => (
                                        vk::AccessFlags::COLOR_ATTACHMENT_READ,
                                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                                    ),
                                    mask if mask.intersects(
                                        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                                    ) =>
                                    {
                                        (
                                            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                                            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                                        )
                                    }
                                    _ => (
                                        vk::AccessFlags::MEMORY_READ
                                            | vk::AccessFlags::MEMORY_WRITE,
                                        vk::AccessFlags::MEMORY_READ
                                            | vk::AccessFlags::MEMORY_WRITE,
                                    ),
                                };

                                let dep = dependencies.entry((other_idx, exec_idx)).or_insert_with(
                                    || SubpassDependency::new(other_idx as _, exec_idx as _),
                                );

                                // Wait for ...
                                dep.src_stage_mask |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS;
                                dep.src_access_mask |= src_access;

                                // ... before we:
                                dep.dst_stage_mask |= stage;
                                dep.dst_access_mask |= dst_access;
                            }
                        }
                    }
                }

                dependencies.into_values().collect::<Vec<_>>()
            };

        // let info = RenderPassInfo {
        //     attachments,
        //     dependencies,
        //     subpasses,
        // };

        // trace!("{:#?}", info);

        pool.resource(RenderPassInfo {
            attachments,
            dependencies,
            subpasses,
        })
    }

    #[profiling::function]
    fn lease_scheduled_resources<P>(
        &mut self,
        pool: &mut P,
        schedule: &[usize],
    ) -> Result<(), DriverError>
    where
        P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
    {
        let mut render_pass_access_history =
            RenderPassAccessHistory::new(self.graph.resources.len());

        for pass_idx in schedule.iter().copied() {
            // At the time this function runs the pass will already have been optimized into a
            // larger pass made out of anything that might have been merged into it - so we
            // only care about one pass at a time here
            let pass = &self.graph.cmds[pass_idx];

            trace!("requesting [{pass_idx}: {}]", pass.name());

            let descriptor_pool = Self::lease_descriptor_pool(pool, pass)?;
            let mut exec_descriptor_sets = HashMap::with_capacity(
                descriptor_pool
                    .as_ref()
                    .map(|descriptor_pool| descriptor_pool.info.max_sets as usize)
                    .unwrap_or_default(),
            );
            if let Some(descriptor_pool) = descriptor_pool.as_ref() {
                for (exec_idx, pipeline) in
                    pass.execs
                        .iter()
                        .enumerate()
                        .filter_map(|(exec_idx, exec)| {
                            exec.pipeline.as_ref().map(|pipeline| (exec_idx, pipeline))
                        })
                {
                    let layouts = pipeline.descriptor_info().layouts.values();
                    let mut descriptor_sets = Vec::with_capacity(layouts.len());
                    for descriptor_set_layout in layouts {
                        descriptor_sets.push(DescriptorPool::allocate_descriptor_set(
                            descriptor_pool,
                            descriptor_set_layout,
                        )?);
                    }
                    exec_descriptor_sets.insert(exec_idx, descriptor_sets);
                }
            }

            // Note that as a side effect of merging compatible passes all input passes should
            // be globbed onto their preceeding passes by now. This allows subpasses to use
            // input attachments without really doing anything, so we are provided a pass that
            // starts with input we just blow up b/c we can't provide it, or at least shouldn't.
            debug_assert!(!pass.execs.is_empty());
            debug_assert!(
                pass.expect_first_exec().pipeline.is_none()
                    || !pass
                        .expect_first_exec()
                        .pipeline
                        .as_ref()
                        .is_some_and(|pipeline| pipeline.is_graphic())
                    || pass
                        .expect_first_exec()
                        .pipeline
                        .as_ref()
                        .expect("missing graphics pipeline")
                        .expect_graphic()
                        .inner
                        .descriptor_info
                        .pool_sizes
                        .values()
                        .filter_map(|pool| pool.get(&vk::DescriptorType::INPUT_ATTACHMENT))
                        .next()
                        .is_none()
            );

            // Also the renderpass may just be None if the pass contained no graphic ops.
            let render_pass = if pass
                .expect_first_exec()
                .pipeline
                .as_ref()
                .map(|pipeline| pipeline.is_graphic())
                .unwrap_or_default()
            {
                Some(self.lease_render_pass(pool, pass_idx, &render_pass_access_history)?)
            } else {
                None
            };

            render_pass_access_history.record_pass(pass);

            self.physical_passes.push(PhysicalPass {
                descriptor_pool,
                exec_descriptor_sets,
                render_pass,
            });
        }

        Ok(())
    }

    // Merges passes which are graphic with common-ish attachments - note that scheduled pass order
    // is final during this function and so we must merge contiguous groups of passes
    #[profiling::function]
    fn merge_scheduled_passes(&mut self, schedule: &mut Vec<usize>) {
        thread_local! {
            static PASSES: RefCell<Vec<Option<CommandData>>> = Default::default();
        }

        PASSES.with_borrow_mut(|passes| {
            debug_assert!(passes.is_empty());

            passes.extend(self.graph.cmds.drain(..).map(Some));

            let mut idx = 0;

            // debug!("attempting to merge {} passes", schedule.len(),);

            while idx < schedule.len() {
                let mut pass = passes[schedule[idx]]
                    .take()
                    .expect("missing scheduled pass");

                // Find candidates
                let start = idx + 1;
                let mut end = start;
                while end < schedule.len() {
                    let other = passes[schedule[end]]
                        .as_ref()
                        .expect("missing scheduled pass");

                    debug!(
                        "attempting to merge [{idx}: {}] with [{end}: {}]",
                        pass.name(),
                        other.name()
                    );

                    if Self::allow_merge_passes(&pass, other) {
                        end += 1;
                    } else {
                        break;
                    }
                }

                if log_enabled!(Trace) && start != end {
                    trace!(
                        "merging {} passes into [{idx}: {}]",
                        end - start,
                        pass.name()
                    );
                }

                let mut name = pass.name().to_owned();

                // Grow the merged pass once, not per merge
                {
                    let mut name_additional = 0;
                    let mut execs_additional = 0;
                    for idx in start..end {
                        let other = passes[schedule[idx]]
                            .as_ref()
                            .expect("missing scheduled pass");
                        name_additional += other.name().len() + 3;
                        execs_additional += other.execs.len();
                    }

                    name.reserve(name_additional);
                    pass.execs.reserve(execs_additional);
                }

                for idx in start..end {
                    let mut other = passes[schedule[idx]]
                        .take()
                        .expect("missing scheduled pass");
                    name.push_str(" + ");
                    name.push_str(other.name());
                    pass.execs.append(&mut other.execs);
                }

                #[cfg(debug_assertions)]
                {
                    pass.name = Some(name);
                }

                self.graph.cmds.push(pass);
                idx += 1 + end - start;
            }

            // Reschedule passes
            schedule.truncate(self.graph.cmds.len());

            for (idx, pass_idx) in schedule.iter_mut().enumerate() {
                *pass_idx = idx;
            }

            // Add the remaining passes back into the graph for later
            for pass in passes.drain(..).flatten() {
                self.graph.cmds.push(pass);
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

    /// Returns the stages that process the given node.
    ///
    /// Note that this value must be retrieved before resolving a node as there will be no
    /// data left to inspect afterwards!
    #[profiling::function]
    pub fn resource_stages(&self, resource_node: impl Node) -> vk::PipelineStageFlags {
        self.graph.assert_node_owner(&resource_node);

        let node_idx = resource_node.index();
        let mut res = Default::default();

        'pass: for pass in self.graph.cmds.iter() {
            for exec in pass.execs.iter() {
                if exec.accesses.contains(node_idx) {
                    res |= pass
                        .execs
                        .iter()
                        .filter_map(|exec| exec.pipeline.as_ref())
                        .map(|pipeline| pipeline.stage())
                        .reduce(|j, k| j | k)
                        .unwrap_or(vk::PipelineStageFlags::TRANSFER);

                    // The execution pipelines of a pass are always the same type
                    continue 'pass;
                }
            }
        }

        debug_assert_ne!(
            res,
            Default::default(),
            "The given node was not accessed in this graph"
        );

        res
    }

    #[profiling::function]
    fn record_execution_barriers<'a>(
        cmd_buf: &CommandBuffer,
        resources: &mut [AnyResource],
        accesses: impl Iterator<Item = (NodeIndex, &'a [SubresourceAccess])>,
        pending_transfers: &HashMap<vk::Image, PendingTransfer>,
    ) {
        // We store a Barriers in TLS to save an alloc; contents are POD
        thread_local! {
            static TLS: RefCell<Tls> = Default::default();
        }

        struct Barrier<T> {
            next_access: AccessType,
            prev_access: AccessType,
            resource: T,
        }

        struct BufferResource {
            buffer: vk::Buffer,
            offset: usize,
            size: usize,
        }

        struct ImageResource {
            image: vk::Image,
            range: vk::ImageSubresourceRange,
        }

        #[derive(Default)]
        struct Tls {
            buffers: Vec<Barrier<BufferResource>>,
            images: Vec<Barrier<ImageResource>>,
            next_accesses: Vec<AccessType>,
            prev_accesses: Vec<AccessType>,
        }

        TLS.with_borrow_mut(|tls| {
            // Initialize TLS from a previous call
            tls.buffers.clear();
            tls.images.clear();
            tls.next_accesses.clear();
            tls.prev_accesses.clear();

            // Map remaining accesses into vk_sync barriers (some accesses may have been removed by
            // the render pass request function)

            for (node_idx, accesses) in accesses {
                let resource = &resources[node_idx];

                match resource {
                    AnyResource::AccelerationStructure(..)
                    | AnyResource::AccelerationStructureLease(..) => {
                        let Some(accel_struct) = resource.as_accel_struct() else {
                            #[cfg(feature = "checked")]
                            unreachable!();

                            #[cfg(not(feature = "checked"))]
                            unsafe {
                                unreachable_unchecked()
                            }
                        };

                        let prev_access = AccelerationStructure::access(
                            accel_struct,
                            accesses.last().expect("missing resource access").access,
                        );

                        tls.next_accesses.extend(
                            accesses
                                .iter()
                                .map(|&SubresourceAccess { access, .. }| access),
                        );
                        tls.prev_accesses.push(prev_access);
                    }
                    AnyResource::Buffer(..) | AnyResource::BufferLease(..) => {
                        let Some(buffer) = resource.as_buffer() else {
                            #[cfg(feature = "checked")]
                            unreachable!();

                            #[cfg(not(feature = "checked"))]
                            unsafe {
                                unreachable_unchecked()
                            }
                        };

                        for &SubresourceAccess {
                            access,
                            subresource,
                        } in accesses
                        {
                            let SubresourceRange::Buffer(range) = subresource else {
                                unreachable!()
                            };

                            for (prev_access, range) in Buffer::access(buffer, access, range) {
                                tls.buffers.push(Barrier {
                                    next_access: access,
                                    prev_access,
                                    resource: BufferResource {
                                        buffer: buffer.handle,
                                        offset: range.start as _,
                                        size: (range.end - range.start) as _,
                                    },
                                });
                            }
                        }
                    }
                    AnyResource::Image(..)
                    | AnyResource::ImageLease(..)
                    | AnyResource::SwapchainImage(..) => {
                        let Some(image) = resource.as_image() else {
                            #[cfg(feature = "checked")]
                            unreachable!();

                            #[cfg(not(feature = "checked"))]
                            unsafe {
                                unreachable_unchecked()
                            }
                        };

                        for &SubresourceAccess {
                            access,
                            subresource,
                        } in accesses
                        {
                            let SubresourceRange::Image(range) = subresource else {
                                unreachable!()
                            };

                            for (prev_access, range) in Image::access(image, access, range) {
                                tls.images.push(Barrier {
                                    next_access: access,
                                    prev_access,
                                    resource: ImageResource {
                                        image: image.handle,
                                        range,
                                    },
                                })
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
            let buffer_barriers = tls.buffers.iter().map(
                |Barrier {
                     next_access,
                     prev_access,
                     resource,
                 }| {
                    let BufferResource {
                        buffer,
                        offset,
                        size,
                    } = *resource;

                    trace!(
                        "    buffer {:?} {:?} {:?}->{:?}",
                        buffer,
                        offset..offset + size,
                        prev_access,
                        next_access,
                    );

                    BufferBarrier {
                        next_accesses: slice::from_ref(next_access),
                        previous_accesses: slice::from_ref(prev_access),
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        buffer,
                        offset,
                        size,
                    }
                },
            );
            let image_barriers = tls.images.iter().map(
                |Barrier {
                     next_access,
                     prev_access,
                     resource,
                 }| {
                    let ImageResource { image, range } = *resource;

                    struct ImageSubresourceRangeDebug(vk::ImageSubresourceRange);

                    impl std::fmt::Debug for ImageSubresourceRangeDebug {
                        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                            self.0.aspect_mask.fmt(f)?;

                            f.write_str(" array: ")?;

                            let array_layers = self.0.base_array_layer
                                ..self.0.base_array_layer + self.0.layer_count;
                            array_layers.fmt(f)?;

                            f.write_str(" mip: ")?;

                            let mip_levels =
                                self.0.base_mip_level..self.0.base_mip_level + self.0.level_count;
                            mip_levels.fmt(f)
                        }
                    }

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
                        discard_contents: *prev_access == AccessType::Nothing
                            || is_write_access(*next_access),
                        src_queue_family_index: pending_transfers
                            .get(&image)
                            .map_or(vk::QUEUE_FAMILY_IGNORED, |transfer| {
                                transfer.src_queue_family_index
                            }),
                        dst_queue_family_index: pending_transfers
                            .get(&image)
                            .map_or(vk::QUEUE_FAMILY_IGNORED, |transfer| {
                                transfer.dst_queue_family_index
                            }),
                        image,
                        range,
                    }
                },
            );

            pipeline_barrier(
                &cmd_buf.device,
                cmd_buf.handle,
                global_barrier,
                &buffer_barriers.collect::<Box<_>>(),
                &image_barriers.collect::<Box<_>>(),
            );
        });
    }

    #[profiling::function]
    fn record_image_layout_transitions(
        cmd_buf: &CommandBuffer,
        resources: &mut [AnyResource],
        pass: &mut CommandData,
        pending_transfers: &HashMap<vk::Image, PendingTransfer>,
    ) {
        // We store a Barriers in TLS to save an alloc; contents are POD
        thread_local! {
            static TLS: RefCell<Tls> = Default::default();
        }

        struct ImageResourceBarrier {
            image: vk::Image,
            next_access: AccessType,
            prev_access: AccessType,
            range: vk::ImageSubresourceRange,
        }

        #[derive(Default)]
        struct Tls {
            images: Vec<ImageResourceBarrier>,
            initial_layouts: HashMap<usize, DenseAccess<bool>>,
        }

        TLS.with_borrow_mut(|tls| {
            tls.images.clear();
            tls.initial_layouts.clear();

            for (node_idx, accesses) in pass.execs.iter_mut().flat_map(|exec| exec.accesses.iter())
            {
                debug_assert!(resources.get(node_idx).is_some());

                let resource = unsafe {
                    // CommandRef enforces this during push_resource_access
                    resources.get_unchecked(node_idx)
                };

                match resource {
                    AnyResource::AccelerationStructure(..)
                    | AnyResource::AccelerationStructureLease(..) => {
                        let Some(accel_struct) = resource.as_accel_struct() else {
                            #[cfg(feature = "checked")]
                            unreachable!();

                            #[cfg(not(feature = "checked"))]
                            unsafe {
                                unreachable_unchecked()
                            }
                        };

                        AccelerationStructure::access(accel_struct, AccessType::Nothing);
                    }
                    AnyResource::Buffer(..) | AnyResource::BufferLease(..) => {
                        let Some(buffer) = resource.as_buffer() else {
                            #[cfg(feature = "checked")]
                            unreachable!();

                            #[cfg(not(feature = "checked"))]
                            unsafe {
                                unreachable_unchecked()
                            }
                        };

                        for subresource_access in accesses {
                            let &SubresourceAccess {
                                subresource: SubresourceRange::Buffer(access_range),
                                ..
                            } = subresource_access
                            else {
                                #[cfg(feature = "checked")]
                                unreachable!();

                                #[cfg(not(feature = "checked"))]
                                unsafe {
                                    // This cannot be reached because PassRef enforces the subrange
                                    // is of type N::Subresource
                                    // where N is the image node type
                                    unreachable_unchecked()
                                }
                            };

                            for _ in Buffer::access(buffer, AccessType::Nothing, access_range) {}
                        }
                    }
                    AnyResource::Image(..)
                    | AnyResource::ImageLease(..)
                    | AnyResource::SwapchainImage(..) => {
                        let Some(image) = resource.as_image() else {
                            #[cfg(feature = "checked")]
                            unreachable!();

                            #[cfg(not(feature = "checked"))]
                            unsafe {
                                unreachable_unchecked()
                            }
                        };

                        // TODO: Optimize this path for single-aspect single-layer single-mip images
                        let initial_layout = tls
                            .initial_layouts
                            .entry(node_idx)
                            .or_insert_with(|| DenseAccess::new(image.info, true));

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
                                    // This cannot be reached because PassRef enforces the subrange
                                    // is of type N::Subresource
                                    // where N is the image node type
                                    unreachable_unchecked()
                                }
                            };

                            for (initial_layout, layout_range) in
                                initial_layout.swap(false, access_range)
                            {
                                for (prev_access, range) in
                                    Image::access(image, access, layout_range)
                                {
                                    if initial_layout {
                                        tls.images.push(ImageResourceBarrier {
                                            image: image.handle,
                                            next_access: initial_image_layout_access(access),
                                            prev_access,
                                            range,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let image_barriers = tls.images.iter().map(
                |ImageResourceBarrier {
                     image,
                     next_access,
                     prev_access,
                     range,
                 }| {
                    trace!(
                        "    image {:?} {:?} {:?}->{:?}",
                        image,
                        ImageSubresourceRangeDebug(*range),
                        prev_access,
                        next_access,
                    );

                    // Color Attachment Read/Write (blending) will prevent discarding contents.
                    // Note that we must check "not-read" because some reads write!
                    let discard_contents =
                        *prev_access == AccessType::Nothing || !is_read_access(*next_access);

                    ImageBarrier {
                        next_accesses: slice::from_ref(next_access),
                        next_layout: image_access_layout(*next_access),
                        previous_accesses: slice::from_ref(prev_access),
                        previous_layout: image_access_layout(*prev_access),
                        discard_contents,
                        src_queue_family_index: pending_transfers
                            .get(image)
                            .map_or(vk::QUEUE_FAMILY_IGNORED, |transfer| {
                                transfer.src_queue_family_index
                            }),
                        dst_queue_family_index: pending_transfers
                            .get(image)
                            .map_or(vk::QUEUE_FAMILY_IGNORED, |transfer| {
                                transfer.dst_queue_family_index
                            }),
                        image: *image,
                        range: *range,
                    }
                },
            );

            pipeline_barrier(
                &cmd_buf.device,
                cmd_buf.handle,
                None,
                &[],
                &image_barriers.collect::<Box<_>>(),
            );
        });
    }

    #[profiling::function]
    fn record_node_passes<P>(
        &mut self,
        pool: &mut P,
        cmd_buf: &mut CommandBuffer,
        node_idx: usize,
        end_pass_idx: usize,
    ) -> Result<(), DriverError>
    where
        P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
    {
        thread_local! {
            static SCHEDULE: RefCell<Schedule> = Default::default();
        }

        SCHEDULE.with_borrow_mut(|schedule| {
            schedule.access_index.update(&self.graph, end_pass_idx);
            schedule.passes.clear();

            self.schedule_node_passes(node_idx, end_pass_idx, schedule);
            self.record_scheduled_passes(pool, cmd_buf, schedule, end_pass_idx)
        })
    }

    fn track_pending_transfers(&mut self, schedule: &Schedule, queue_family_index: u32) {
        for pass_idx in schedule.passes.iter().copied() {
            let pass = &self.graph.cmds[pass_idx];

            for (node_idx, accesses) in pass.execs.iter().flat_map(|exec| exec.accesses.iter()) {
                if !accesses
                    .iter()
                    .any(|access| matches!(access.subresource, SubresourceRange::Image(..)))
                {
                    continue;
                }

                let Some(image) = self.graph.resources[node_idx].as_image() else {
                    continue;
                };

                if image.info.sharing_mode == vk::SharingMode::CONCURRENT {
                    continue;
                }

                self.exclusive_image_indices.insert(node_idx);

                if image.current_access() == AccessType::Nothing {
                    continue;
                }

                let (src_queue_family_index, src_queue_index) =
                    unpack_queue(image.queue_packed.load(Ordering::Acquire));

                if src_queue_family_index == queue_family_index {
                    continue;
                }

                let transfer = PendingTransfer {
                    src_access: image.current_access(),
                    src_queue_family_index,
                    src_queue_index,
                    dst_queue_family_index: queue_family_index,
                };

                self.pending_transfers
                    .entry(image.handle)
                    .and_modify(|existing| {
                        debug_assert_eq!(
                            *existing, transfer,
                            "conflicting queue transfer recorded for image"
                        );
                    })
                    .or_insert(transfer);
            }
        }
    }

    #[profiling::function]
    fn record_scheduled_passes<P>(
        &mut self,
        pool: &mut P,
        cmd_buf: &mut CommandBuffer,
        schedule: &mut Schedule,
        end_pass_idx: usize,
    ) -> Result<(), DriverError>
    where
        P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
    {
        if schedule.passes.is_empty() {
            return Ok(());
        }

        // // Print some handy details or hit a breakpoint if you set the flag
        // if log_enabled!(Debug) && self.graph.debug {
        //     debug!("resolving the following graph:\n\n{:#?}\n\n", self.graph);
        // }

        debug_assert!(
            schedule.passes.windows(2).all(|w| w[0] <= w[1]),
            "Unsorted schedule"
        );

        // Optimize the schedule; requesting the required resources it needs
        Self::reorder_scheduled_passes(schedule, end_pass_idx);
        self.merge_scheduled_passes(&mut schedule.passes);
        self.lease_scheduled_resources(pool, &schedule.passes)?;
        self.track_pending_transfers(schedule, cmd_buf.info.queue_family_index);

        #[cfg(feature = "checked")]
        let graph_id = self.graph.graph_id();

        for pass_idx in schedule.passes.iter().copied() {
            let pass = &mut self.graph.cmds[pass_idx];

            profiling::scope!("Pass", pass.name());

            let physical_pass = &mut self.physical_passes[pass_idx];
            let is_graphic = physical_pass.render_pass.is_some();

            trace!("recording pass [{}: {}]", pass_idx, pass.name());

            if !physical_pass.exec_descriptor_sets.is_empty() {
                Self::write_descriptor_sets(cmd_buf, &self.graph.resources, pass, physical_pass)?;
            }

            let render_area = if is_graphic {
                Self::record_image_layout_transitions(
                    cmd_buf,
                    &mut self.graph.resources,
                    pass,
                    &self.pending_transfers,
                );

                let render_area = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: Self::render_extent(&self.graph.resources, pass),
                };

                Self::begin_render_pass(
                    cmd_buf,
                    &self.graph.resources,
                    pass,
                    physical_pass,
                    render_area,
                )?;

                Some(render_area)
            } else {
                None
            };

            for exec_idx in 0..pass.execs.len() {
                let render_area = is_graphic.then(|| {
                    pass.execs[exec_idx]
                        .render_area
                        .unwrap_or(render_area.expect("missing render area"))
                });

                let exec = &mut pass.execs[exec_idx];

                if is_graphic && exec_idx > 0 {
                    Self::next_subpass(cmd_buf);
                }

                if let Some(pipeline) = exec.pipeline.as_mut() {
                    Self::bind_pipeline(
                        cmd_buf,
                        physical_pass,
                        exec_idx,
                        pipeline,
                        exec.depth_stencil,
                    )?;

                    if is_graphic {
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

                    Self::bind_descriptor_sets(cmd_buf, pipeline, physical_pass, exec_idx);
                }

                if !is_graphic {
                    Self::record_execution_barriers(
                        cmd_buf,
                        &mut self.graph.resources,
                        exec.accesses.iter(),
                        &self.pending_transfers,
                    );
                }

                trace!("    > exec[{exec_idx}]");

                {
                    profiling::scope!("Execute callback");

                    let exec_func = exec.func.take().expect("missing command function").0;
                    exec_func(crate::cmd::CommandRef::new(
                        cmd_buf,
                        &self.graph.resources,
                        #[cfg(feature = "checked")]
                        exec,
                        #[cfg(feature = "checked")]
                        graph_id,
                    ));
                }
            }

            if is_graphic {
                self.end_render_pass(cmd_buf);
            }
        }

        thread_local! {
            static PASSES: RefCell<Vec<CommandData>> = Default::default();
        }

        PASSES.with_borrow_mut(|passes| {
            debug_assert!(passes.is_empty());

            // We have to keep the bindings and pipelines alive until the gpu is done
            schedule.passes.sort_unstable();
            while let Some(schedule_idx) = schedule.passes.pop() {
                debug_assert!(!self.graph.cmds.is_empty());

                while let Some(pass) = self.graph.cmds.pop() {
                    let pass_idx = self.graph.cmds.len();

                    if pass_idx == schedule_idx {
                        // This was a scheduled pass - store it!

                        cmd_buf.drop_after_executed((
                            pass,
                            self.physical_passes.pop().expect("missing physical pass"),
                        ));
                        break;
                    } else {
                        debug_assert!(pass_idx > schedule_idx);

                        passes.push(pass);
                    }
                }
            }

            debug_assert!(self.physical_passes.is_empty());

            // Put the other passes back for future resolves
            self.graph.cmds.extend(passes.drain(..).rev());
        });

        log::trace!("Recorded passes");

        Ok(())
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
            .color_clears
            .values()
            .copied()
            .map(|(attachment, _)| attachment)
            .chain(first_exec.color_loads.values().copied())
            .chain(first_exec.color_stores.values().copied())
            .chain(
                first_exec
                    .depth_stencil_clear
                    .map(|(attachment, _)| attachment),
            )
            .chain(first_exec.depth_stencil_load)
            .chain(first_exec.depth_stencil_store)
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

    #[profiling::function]
    fn reorder_scheduled_passes(schedule: &mut Schedule, end_pass_idx: usize) {
        // It must be a party
        if schedule.passes.len() < 3 {
            return;
        }

        let mut scheduled = 0;

        thread_local! {
            static UNSCHEDULED: RefCell<Vec<bool>> = Default::default();
        }

        UNSCHEDULED.with_borrow_mut(|unscheduled| {
            unscheduled.truncate(end_pass_idx);
            unscheduled.fill(true);
            unscheduled.resize(end_pass_idx, true);

            // Re-order passes by maximizing the distance between dependent nodes
            while scheduled < schedule.passes.len() {
                let mut best_idx = scheduled;
                let pass_idx = schedule.passes[best_idx];
                let mut best_overlap_factor = schedule
                    .access_index
                    .interdependent_passes(pass_idx, end_pass_idx)
                    .count();

                for (idx, pass_idx) in schedule.passes[best_idx + 1..schedule.passes.len()]
                    .iter()
                    .enumerate()
                {
                    let mut overlap_factor = 0;

                    for other_pass_idx in schedule
                        .access_index
                        .interdependent_passes(*pass_idx, end_pass_idx)
                    {
                        if unscheduled[other_pass_idx] {
                            // This pass can't be the candidate because it depends on unfinished
                            // work
                            break;
                        }

                        overlap_factor += 1;
                    }

                    if overlap_factor > best_overlap_factor {
                        best_idx += idx + 1;
                        best_overlap_factor = overlap_factor;
                    }
                }

                unscheduled[schedule.passes[best_idx]] = false;
                schedule.passes.swap(scheduled, best_idx);
                scheduled += 1;
            }
        });
    }

    /// Returns a borrow of the original Vulkan resource (buffer, image or acceleration structure)
    /// which the given node represents.
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: Node,
    {
        self.graph.resource(resource_node)
    }

    /// Returns a vec of pass indexes that are required to be executed, in order, for the given
    /// node.
    #[profiling::function]
    fn schedule_node_passes(&self, node_idx: usize, end_pass_idx: usize, schedule: &mut Schedule) {
        type UnscheduledUnresolvedUnchecked = (Vec<bool>, Vec<bool>, VecDeque<(usize, usize)>);

        thread_local! {
            static UNSCHEDULED_UNRESOLVED_UNCHECKED: RefCell<
                UnscheduledUnresolvedUnchecked,
            > = Default::default();
        }

        UNSCHEDULED_UNRESOLVED_UNCHECKED.with_borrow_mut(|(unscheduled, unresolved, unchecked)| {
            unscheduled.truncate(end_pass_idx);
            unscheduled.fill(true);
            unscheduled.resize(end_pass_idx, true);

            unresolved.truncate(self.graph.resources.len());
            unresolved.fill(true);
            unresolved.resize(self.graph.resources.len(), true);

            debug_assert!(unchecked.is_empty());

            trace!("scheduling node {node_idx}");

            unresolved[node_idx] = false;

            // Schedule the first set of passes for the node we're trying to resolve
            for pass_idx in schedule
                .access_index
                .dependent_passes(node_idx, end_pass_idx)
            {
                trace!(
                    "  pass [{pass_idx}: {}] is dependent",
                    self.graph.cmds[pass_idx].name()
                );

                debug_assert!(unscheduled[pass_idx]);

                unscheduled[pass_idx] = false;
                schedule.passes.push(pass_idx);

                for node_idx in schedule.access_index.dependent_nodes(pass_idx) {
                    trace!("    node {node_idx} is dependent");

                    let unresolved = &mut unresolved[node_idx];
                    if *unresolved {
                        *unresolved = false;
                        unchecked.push_back((node_idx, pass_idx));
                    }
                }
            }

            trace!("secondary passes below");

            // Now schedule all nodes that are required, going through the tree to find them
            while let Some((node_idx, pass_idx)) = unchecked.pop_front() {
                trace!("  node {node_idx} is dependent");

                for pass_idx in schedule
                    .access_index
                    .dependent_passes(node_idx, pass_idx + 1)
                {
                    let unscheduled = &mut unscheduled[pass_idx];
                    if *unscheduled {
                        *unscheduled = false;
                        schedule.passes.push(pass_idx);

                        trace!(
                            "  pass [{pass_idx}: {}] is dependent",
                            self.graph.cmds[pass_idx].name()
                        );

                        for node_idx in schedule.access_index.dependent_nodes(pass_idx) {
                            trace!("    node {node_idx} is dependent");

                            let unresolved = &mut unresolved[node_idx];
                            if *unresolved {
                                *unresolved = false;
                                unchecked.push_back((node_idx, pass_idx));
                            }
                        }
                    }
                }
            }

            schedule.passes.sort_unstable();

            if log_enabled!(Debug) {
                if !schedule.passes.is_empty() {
                    // These are the indexes of the passes this thread is about to resolve
                    debug!(
                        "schedule: {}",
                        schedule
                            .passes
                            .iter()
                            .copied()
                            .map(|idx| format!("[{}: {}]", idx, self.graph.cmds[idx].name()))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }

                if log_enabled!(Trace) {
                    let unscheduled = (0..end_pass_idx)
                        .filter(|&pass_idx| unscheduled[pass_idx])
                        .collect::<Box<_>>();

                    if !unscheduled.is_empty() {
                        // These passes are within the range of passes we thought we had to do
                        // right now, but it turns out that nothing in "schedule" relies on them
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

                    if end_pass_idx < self.graph.cmds.len() {
                        // These passes existing on the graph but are not being considered right
                        // now because we've been told to stop work at the "end_pass_idx" point
                        trace!(
                            "ignoring: {}",
                            self.graph.cmds[end_pass_idx..]
                                .iter()
                                .enumerate()
                                .map(|(idx, pass)| format!(
                                    "[{}: {}]",
                                    idx + end_pass_idx,
                                    pass.name()
                                ))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                }
            }
        });
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

    /// Records and submits the remaining commands stored in this instance.
    ///
    /// This is the one-shot execution path for a finalized graph. It:
    ///
    /// 1. Leases a command buffer from `pool` for `queue_family_index`.
    /// 2. Waits for that command buffer's prior submission to complete.
    /// 3. Begins recording, records all remaining graph commands, ends recording, and submits.
    /// 4. Returns the leased command buffer so the caller can observe completion or wait on it.
    ///
    /// The returned command buffer owns the submission fence and keeps the graph resources alive
    /// until execution completes. Callers should treat it as in-flight until
    /// [`CommandBuffer::has_executed`] or [`CommandBuffer::wait_until_executed`] indicates the GPU
    /// has finished using it.
    pub fn queue_submit<P>(
        mut self,
        pool: &mut P,
        queue_family_index: u32,
        queue_index: u32,
    ) -> Result<Lease<CommandBuffer>, DriverError>
    where
        P: Pool<CommandBufferInfo, CommandBuffer>
            + Pool<DescriptorPoolInfo, DescriptorPool>
            + Pool<RenderPassInfo, RenderPass>,
    {
        trace!("submit");

        // Phase 1: Get the main command buffer and record commands. This also discovers any
        // ownership transfers required by the scheduled work.
        let mut cmd_buf = pool.resource(CommandBufferInfo::new(queue_family_index as _))?;
        cmd_buf.wait_until_executed()?;
        cmd_buf.begin(
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        self.record_impl(pool, &mut cmd_buf)?;
        cmd_buf.end()?;
        self.queue_submit_recorded(pool, &mut cmd_buf, queue_index, &[], &[], &[])?;

        Ok(cmd_buf)
    }

    /// Submits commands already recorded into `cmd_buf`, along with any ownership-transfer release
    /// work required by this submission.
    ///
    /// Does not call begin or end on the command buffer.
    fn queue_submit_recorded<P>(
        self,
        pool: &mut P,
        cmd_buf: &mut CommandBuffer,
        queue_index: u32,
        wait_semaphores: &[vk::Semaphore],
        wait_stage_mask: &[vk::PipelineStageFlags],
        signal_semaphores: &[vk::Semaphore],
    ) -> Result<(), DriverError>
    where
        P: Pool<CommandBufferInfo, CommandBuffer>,
    {
        let queue_family_index = cmd_buf.info.queue_family_index;

        // Phase 2: Build RELEASE submissions for any transfers discovered while recording.
        let mut release_groups: Vec<ReleaseGroup> = Vec::new();
        for resource in self.graph.resources.iter() {
            let Some(image) = resource.as_image() else {
                continue;
            };
            let Some(&transfer) = self.pending_transfers.get(&image.handle) else {
                continue;
            };
            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: format_aspect_mask(image.info.fmt),
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            };
            if let Some(group) = release_groups.iter_mut().find(|g| {
                g.old_fam == transfer.src_queue_family_index
                    && g.old_idx == transfer.src_queue_index
            }) {
                group
                    .images
                    .push((image.handle, transfer.src_access, subresource_range));
            } else {
                release_groups.push(ReleaseGroup {
                    old_fam: transfer.src_queue_family_index,
                    old_idx: transfer.src_queue_index,
                    images: vec![(image.handle, transfer.src_access, subresource_range)],
                });
            }
        }

        // Phase 3: For each unique old queue, submit a RELEASE barrier.
        let mut release_bundles: Vec<ReleaseBundle> = Vec::new();
        if !release_groups.is_empty() {
            for group in &release_groups {
                let mut release_cmd = pool.resource(CommandBufferInfo::new(group.old_fam as _))?;
                release_cmd.wait_until_executed()?;
                release_cmd.reset_fence()?;
                let semaphore = release_cmd.release_semaphore()?;

                Device::begin_command_buffer(
                    &release_cmd.device,
                    release_cmd.handle,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;

                let barriers: Box<[_]> = group
                    .images
                    .iter()
                    .map(|&(handle, current_access, subresource_range)| {
                        let layout = access_type_to_layout(current_access);
                        vk::ImageMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                            .dst_access_mask(vk::AccessFlags::empty())
                            .old_layout(layout)
                            .new_layout(layout)
                            .src_queue_family_index(group.old_fam)
                            .dst_queue_family_index(queue_family_index)
                            .image(handle)
                            .subresource_range(subresource_range)
                    })
                    .collect();

                unsafe {
                    release_cmd.device.cmd_pipeline_barrier(
                        release_cmd.handle,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &barriers,
                    );
                }

                Device::with_queue(&release_cmd.device, group.old_fam, group.old_idx, |queue| {
                    Device::end_command_buffer(&release_cmd.device, release_cmd.handle)?;
                    Device::queue_submit(
                        &release_cmd.device,
                        queue,
                        slice::from_ref(
                            &vk::SubmitInfo::default()
                                .command_buffers(slice::from_ref(&release_cmd.handle))
                                .signal_semaphores(slice::from_ref(&semaphore)),
                        ),
                        release_cmd.fence,
                    )?;

                    Ok::<_, DriverError>(())
                })?;

                release_bundles.push(ReleaseBundle {
                    cmd_buf: release_cmd,
                    semaphore,
                });
            }
        }

        // Phase 4: Submit the main command buffer, waiting on release semaphores if any.
        Device::with_queue(&cmd_buf.device, queue_family_index, queue_index, |queue| {
            Device::reset_fences(&cmd_buf.device, slice::from_ref(&cmd_buf.fence))?;

            let release_wait_semaphores = release_bundles
                .iter()
                .map(|b| b.semaphore)
                .collect::<Box<[_]>>();
            let release_wait_stages =
                repeat_n(vk::PipelineStageFlags::ALL_COMMANDS, release_bundles.len())
                    .collect::<Box<[_]>>();

            let merged_wait_semaphores = wait_semaphores
                .iter()
                .copied()
                .chain(release_wait_semaphores.iter().copied())
                .collect::<Box<[_]>>();
            let merged_wait_stages = wait_stage_mask
                .iter()
                .copied()
                .chain(release_wait_stages.iter().copied())
                .collect::<Box<[_]>>();

            let mut submit_info = vk::SubmitInfo::default()
                .command_buffers(slice::from_ref(&cmd_buf.handle))
                .signal_semaphores(signal_semaphores);

            if !merged_wait_semaphores.is_empty() {
                submit_info = submit_info
                    .wait_semaphores(&merged_wait_semaphores)
                    .wait_dst_stage_mask(&merged_wait_stages);
            };

            cmd_buf.queue_submit(queue, slice::from_ref(&submit_info))?;

            Ok::<_, DriverError>(())
        })?;

        // Phase 5: Update queue ownership for all exclusive images touched by this submission.
        for node_idx in self.exclusive_image_indices.ones() {
            if let Some(resource) = self.graph.resources[node_idx].as_image() {
                resource.queue_packed.store(
                    pack_queue(queue_family_index, queue_index),
                    Ordering::Release,
                );
            }
        }

        // Keep release bundles alive until the main submission completes.
        for bundle in release_bundles {
            cmd_buf.drop_after_executed(bundle);
        }

        // This graph contains references to buffers, images, and other resources which must be kept
        // alive until this graph execution completes on the GPU. Once those references are dropped
        // they will return to the pool for other things to use. The drop will happen the next time
        // someone tries to lease a command buffer and we notice this one has returned and the fence
        // has been signalled.
        cmd_buf.drop_after_executed(self);

        Ok(())
    }

    /// Records any remaining graph commands into `cmd_buf` and returns a [`RecordedSubmission`].
    #[profiling::function]
    pub fn record<'a, P>(
        mut self,
        pool: &mut P,
        cmd_buf: &'a mut CommandBuffer,
    ) -> Result<RecordedSubmission<'a>, DriverError>
    where
        P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
    {
        self.record_impl(pool, cmd_buf)?;

        Ok(RecordedSubmission {
            cmd_buf,
            submission: self,
        })
    }

    #[profiling::function]
    fn record_impl<P>(
        &mut self,
        pool: &mut P,
        cmd_buf: &mut CommandBuffer,
    ) -> Result<(), DriverError>
    where
        P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
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
            schedule.passes.clear();
            schedule.passes.extend(0..self.graph.cmds.len());

            self.record_scheduled_passes(pool, cmd_buf, schedule, self.graph.cmds.len())
        })
    }

    /// Records any remaining graph commands that the given node requires into `cmd_buf` and
    /// returns a [`RecordedSubmission`].
    ///
    /// This is a mutating execution step, not a pure query. It records work into the provided
    /// command buffer and updates this submission's scheduling state so those commands are not
    /// recorded again later.
    #[profiling::function]
    pub fn record_resource<'a, P>(
        mut self,
        pool: &mut P,
        cmd_buf: &'a mut CommandBuffer,
        resource_node: impl Node,
    ) -> Result<RecordedSubmission<'a>, DriverError>
    where
        P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
    {
        self.record_resource_impl(pool, cmd_buf, resource_node)?;

        Ok(RecordedSubmission {
            cmd_buf,
            submission: self,
        })
    }

    #[profiling::function]
    fn record_resource_impl<P>(
        &mut self,
        pool: &mut P,
        cmd_buf: &mut CommandBuffer,
        resource_node: impl Node,
    ) -> Result<(), DriverError>
    where
        P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
    {
        self.graph.assert_node_owner(&resource_node);

        let node_idx = resource_node.index();

        debug_assert!(self.graph.resources.get(node_idx).is_some());

        if self.graph.cmds.is_empty() {
            return Ok(());
        }

        let end_pass_idx = self.graph.cmds.len();
        self.record_node_passes(pool, cmd_buf, node_idx, end_pass_idx)
    }

    /// Records any pending graph commands required by the given node into `cmd`, but does not
    /// record any command that actually accesses the given node.
    ///
    /// This is a mutating execution step, not a pure query. It records work into the provided
    /// command buffer, updates this submission's scheduling state, and may reorder later recorded
    /// work.
    ///
    /// The call order matters when extracting multiple outputs from the same submission. This
    /// method optimizes the schedule for the requested node, and later calls can only optimize on
    /// top of that existing state. If you are pulling multiple outputs and care about their final
    /// ordering, record the most important output first.
    #[profiling::function]
    pub fn record_resource_dependencies<'a, P>(
        mut self,
        pool: &mut P,
        cmd_buf: &'a mut CommandBuffer,
        resource_node: impl Node,
    ) -> Result<RecordedSubmission<'a>, DriverError>
    where
        P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
    {
        self.record_resource_dependencies_impl(pool, cmd_buf, resource_node)?;

        Ok(RecordedSubmission {
            cmd_buf,
            submission: self,
        })
    }

    #[profiling::function]
    fn record_resource_dependencies_impl<P>(
        &mut self,
        pool: &mut P,
        cmd_buf: &mut CommandBuffer,
        resource_node: impl Node,
    ) -> Result<(), DriverError>
    where
        P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
    {
        self.graph.assert_node_owner(&resource_node);

        let node_idx = resource_node.index();

        debug_assert!(self.graph.resources.get(node_idx).is_some());

        // We record up to but not including the first pass which accesses the target node
        if let Some(end_pass_idx) = self.graph.first_node_access_pass_index(resource_node) {
            self.record_node_passes(pool, cmd_buf, node_idx, end_pass_idx)?;
        }

        Ok(())
    }

    #[profiling::function]
    fn write_descriptor_sets(
        cmd_buf: &CommandBuffer,
        bindings: &[AnyResource],
        pass: &CommandData,
        physical_pass: &PhysicalPass,
    ) -> Result<(), DriverError> {
        struct IndexWrite<'a> {
            idx: usize,
            write: vk::WriteDescriptorSet<'a>,
        }

        #[derive(Default)]
        struct Tls<'a> {
            accel_struct_infos: Vec<vk::WriteDescriptorSetAccelerationStructureKHR<'a>>,
            accel_struct_writes: Vec<IndexWrite<'a>>,
            buffer_infos: Vec<vk::DescriptorBufferInfo>,
            buffer_writes: Vec<IndexWrite<'a>>,
            descriptors: Vec<vk::WriteDescriptorSet<'a>>,
            image_infos: Vec<vk::DescriptorImageInfo>,
            image_writes: Vec<IndexWrite<'a>>,
        }

        let mut tls = Tls::default();

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
            let descriptor_sets = &physical_pass.exec_descriptor_sets[&exec_idx];

            // Write the manually bound things (access, read, and write functions)
            for (descriptor, (node_idx, view_info)) in exec.bindings.iter() {
                let (descriptor_set_idx, dst_binding, binding_offset) = descriptor.into_tuple();
                let Some((descriptor_info, _)) = pipeline.descriptor_bindings().get(&Descriptor {
                    set: descriptor_set_idx,
                    binding: dst_binding,
                }) else {
                    warn!(
                        "binding {}.{}[{}] not found in shader reflection for pass \"{}\"",
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

                    // Handle default views which did not specify a particaular aspect
                    if image_view_info.aspect_mask.is_empty() {
                        image_view_info.aspect_mask = format_aspect_mask(image.info.fmt);
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
                                "invalid image descriptor type at binding {}.{}[{}] in pass \"{}\"",
                                descriptor_set_idx,
                                dst_binding,
                                binding_offset,
                                pass.name()
                            );

                            return Err(DriverError::InvalidData);
                        }
                    };

                    if binding_offset == 0 {
                        tls.image_writes.push(IndexWrite {
                            idx: tls.image_infos.len(),
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
                        tls.buffer_writes.push(IndexWrite {
                            idx: tls.buffer_infos.len(),
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
                        tls.accel_struct_writes.push(IndexWrite {
                            idx: tls.accel_struct_infos.len(),
                            write: vk::WriteDescriptorSet::default()
                                .dst_set(*descriptor_sets[descriptor_set_idx as usize])
                                .dst_binding(dst_binding)
                                .descriptor_type(descriptor_type)
                                .descriptor_count(1),
                        });
                    } else {
                        tls.accel_struct_writes
                            .last_mut()
                            .expect("missing acceleration structure descriptor write")
                            .write
                            .descriptor_count += 1;
                    }

                    tls.accel_struct_infos.push(
                        vk::WriteDescriptorSetAccelerationStructureKHR::default()
                            .acceleration_structures(std::slice::from_ref(&accel_struct.handle)),
                    );
                } else {
                    warn!(
                        "invalid bound resource kind at descriptor {}.{}[{}] in pass \"{}\"",
                        descriptor_set_idx,
                        dst_binding,
                        binding_offset,
                        pass.name()
                    );

                    return Err(DriverError::InvalidData);
                }
            }

            if let ExecutionPipeline::Graphic(pipeline) = pipeline {
                // Write graphic render pass input attachments (they're automatic)
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
                            let is_random_access = exec.color_stores.contains_key(&attachment_idx)
                                || exec.color_resolves.contains_key(&attachment_idx);
                            let current_attachment = exec
                                .color_attachments
                                .get(&attachment_idx)
                                .copied()
                                .or_else(|| {
                                    exec.color_clears
                                        .get(&attachment_idx)
                                        .map(|(attachment, _)| *attachment)
                                })
                                .or_else(|| exec.color_loads.get(&attachment_idx).copied())
                                .or_else(|| exec.color_stores.get(&attachment_idx).copied())
                                .or_else(|| {
                                    exec.color_resolves
                                        .get(&attachment_idx)
                                        .map(|(attachment, _)| *attachment)
                                })
                                .expect("missing input attachment target");
                            let (attachment, write_exec) = pass.execs[0..exec_idx]
                                .iter()
                                .rev()
                                .find_map(|exec| {
                                    exec.color_attachments
                                        .get(&attachment_idx)
                                        .copied()
                                        .or_else(|| {
                                            exec.color_clears
                                                .get(&attachment_idx)
                                                .map(|(attachment, _)| *attachment)
                                        })
                                        .or_else(|| exec.color_loads.get(&attachment_idx).copied())
                                        .or_else(|| exec.color_stores.get(&attachment_idx).copied())
                                        .or_else(|| {
                                            exec.color_resolves.get(&attachment_idx).map(
                                                |(resolved_attachment, _)| *resolved_attachment,
                                            )
                                        })
                                        .filter(|attachment| {
                                            Attachment::are_compatible(
                                                Some(current_attachment),
                                                Some(*attachment),
                                            )
                                        })
                                        .map(|attachment| (attachment, exec))
                                })
                                .expect("input attachment not written");
                            let late = write_exec
                                .accesses
                                .get(attachment.target)
                                .expect("missing input attachment access")
                                .last()
                                .expect("missing input attachment access");
                            let image_range = late.subresource.expect_image();
                            let image_binding = &bindings[attachment.target];
                            let image = image_binding.expect_image();
                            let image_view_info = attachment
                                .image_view_info(image.info)
                                .into_builder()
                                .array_layer_count(image_range.layer_count)
                                .base_array_layer(image_range.base_array_layer)
                                .base_mip_level(image_range.base_mip_level)
                                .mip_level_count(image_range.level_count)
                                .build();
                            let image_view = Image::view(image, image_view_info)?;

                            tls.image_writes.push(IndexWrite {
                                idx: tls.image_infos.len(),
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
                                    is_random_access,
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

        tls.descriptors
            .extend(tls.accel_struct_writes.drain(..).map(
                |IndexWrite { idx, mut write }| unsafe {
                    write.p_next = tls.accel_struct_infos.as_ptr().add(idx) as *const _;
                    write
                },
            ));
        tls.descriptors.extend(tls.buffer_writes.drain(..).map(
            |IndexWrite { idx, mut write }| unsafe {
                write.p_buffer_info = tls.buffer_infos.as_ptr().add(idx);
                write
            },
        ));
        tls.descriptors.extend(tls.image_writes.drain(..).map(
            |IndexWrite { idx, mut write }| unsafe {
                write.p_image_info = tls.image_infos.as_ptr().add(idx);
                write
            },
        ));

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
    }
}

impl<'a> RecordedSubmission<'a> {
    /// Returns `true` when this submission contains no more commands to record.
    pub fn is_empty(&self) -> bool {
        self.submission.is_empty()
    }

    /// Returns the stages that process the given resource.
    #[profiling::function]
    pub fn resource_stages(&self, resource_node: impl Node) -> vk::PipelineStageFlags {
        self.submission.resource_stages(resource_node)
    }

    /// Returns a borrow of the original Vulkan resource (buffer, image or acceleration structure)
    /// which the given node represents.
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: Node,
    {
        self.submission.resource(resource_node)
    }

    /// Records any remaining graph commands into this submission's command buffer.
    #[profiling::function]
    pub fn record<P>(&mut self, pool: &mut P) -> Result<(), DriverError>
    where
        P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
    {
        self.submission.record_impl(pool, self.cmd_buf)
    }

    /// Records any remaining graph commands required by the given resource into this submission's
    /// command buffer.
    #[profiling::function]
    pub fn record_resource<P>(
        &mut self,
        pool: &mut P,
        resource_node: impl Node,
    ) -> Result<(), DriverError>
    where
        P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
    {
        self.submission
            .record_resource_impl(pool, self.cmd_buf, resource_node)
    }

    /// Records any remaining prerequisite commands for the given resource into this submission's
    /// command buffer, excluding passes that directly access the resource.
    #[profiling::function]
    pub fn record_resource_dependencies<P>(
        &mut self,
        pool: &mut P,
        resource_node: impl Node,
    ) -> Result<(), DriverError>
    where
        P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
    {
        self.submission
            .record_resource_dependencies_impl(pool, self.cmd_buf, resource_node)
    }

    /// Submits this recorded submission's command buffer.
    ///
    /// Use this after binding a [`Submission`] to an existing command buffer with
    /// [`Submission::record`], [`Submission::record_resource`], or
    /// [`Submission::record_resource_dependencies`].
    ///
    /// Callers are responsible for beginning and ending the bound command buffer themselves.
    /// This method only submits the already-recorded command buffer to `queue_index`, waiting on
    /// `wait_semaphores` at `wait_stage_mask` and signaling `signal_semaphores` when complete.
    ///
    /// This consumes the `RecordedSubmission`, ensuring the recorded graph work stays paired with
    /// the command buffer it was recorded into. After submission, the caller still owns that same
    /// command buffer and should treat it as in-flight until execution completes.
    pub fn queue_submit<P>(
        self,
        pool: &mut P,
        queue_index: u32,
        wait_semaphores: &[vk::Semaphore],
        wait_stage_mask: &[vk::PipelineStageFlags],
        signal_semaphores: &[vk::Semaphore],
    ) -> Result<(), DriverError>
    where
        P: Pool<CommandBufferInfo, CommandBuffer>,
    {
        let Self {
            cmd_buf,
            submission,
        } = self;

        submission.queue_submit_recorded(
            pool,
            cmd_buf,
            queue_index,
            wait_semaphores,
            wait_stage_mask,
            signal_semaphores,
        )
    }
}

impl From<Graph> for Submission {
    fn from(val: Graph) -> Self {
        val.finalize()
    }
}

#[derive(Default)]
struct Schedule {
    access_index: AccessIndex,
    passes: Vec<usize>,
}
