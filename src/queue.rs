use {
    super::{
        Area, Attachment, Command, ExecutionPipeline, Graph, Node, NodeIndex, Resource,
        cmd_ref::{Resources, SubresourceAccess, SubresourceRange},
    },
    crate::{
        Bound,
        driver::{
            AttachmentInfo, AttachmentRef, Descriptor, DescriptorInfo, DescriptorSet, DriverError,
            FramebufferAttachmentImageInfo, FramebufferInfo, SubpassDependency, SubpassInfo,
            accel_struct::AccelerationStructure,
            buffer::Buffer,
            cmd_buf::{CommandBuffer, CommandBufferInfo},
            descriptor_set::{DescriptorPool, DescriptorPoolInfo},
            device::Device,
            format_aspect_mask,
            graphic::{DepthStencilMode, GraphicPipeline},
            image::{Image, ImageAccess},
            initial_image_layout_access, is_read_access, is_write_access,
            pipeline_stage_access_flags,
            render_pass::{RenderPass, RenderPassInfo},
        },
        pool::{Lease, Pool},
    },
    ash::vk,
    log::{
        Level::{Debug, Trace},
        debug, log_enabled, trace,
    },
    std::{
        cell::RefCell,
        collections::{BTreeMap, HashMap, VecDeque},
        iter::repeat_n,
        ops::Range,
        slice,
    },
    vk_sync::{
        AccessType, BufferBarrier, GlobalBarrier, ImageBarrier, ImageLayout, cmd::pipeline_barrier,
    },
};

#[cfg(not(debug_assertions))]
use std::hint::unreachable_unchecked;

const fn image_access_layout(access: AccessType) -> ImageLayout {
    if matches!(access, AccessType::Present | AccessType::ComputeShaderWrite) {
        ImageLayout::General
    } else {
        ImageLayout::Optimal
    }
}

#[derive(Default)]
struct AccessCache {
    accesses: Vec<bool>,
    binding_count: usize,
    read_count: Vec<usize>,
    reads: Vec<usize>,
}

impl AccessCache {
    /// Finds the unique indexes of the node bindings which a given pass reads. Results are
    /// returned in the opposite order the dependencies must be resolved in.
    ///
    /// Dependent upon means that the node is read from the pass.
    #[profiling::function]
    fn dependent_nodes(&self, pass_idx: usize) -> impl ExactSizeIterator<Item = usize> + '_ {
        let pass_start = pass_idx * self.binding_count;
        let pass_end = pass_start + self.read_count[pass_idx];
        self.reads[pass_start..pass_end].iter().copied()
    }

    /// Finds the unique indexes of the passes which write to a given node; with the restriction
    /// to not inspect later passes. Results are returned in the opposite order the dependencies
    /// must be resolved in.
    ///
    /// Dependent upon means that the pass writes to the node.
    #[profiling::function]
    fn dependent_passes(
        &self,
        node_idx: usize,
        end_pass_idx: usize,
    ) -> impl Iterator<Item = usize> + '_ {
        self.accesses[node_idx..end_pass_idx * self.binding_count]
            .iter()
            .step_by(self.binding_count)
            .enumerate()
            .rev()
            .filter_map(|(pass_idx, write)| write.then_some(pass_idx))
    }

    /// Returns the unique indexes of the passes which are dependent on the given pass.
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
        self.binding_count = graph.resources.len();

        let cache_len = self.binding_count * end_pass_idx;

        self.accesses.truncate(cache_len);
        self.accesses.fill(false);
        self.accesses.resize(cache_len, false);

        self.read_count.clear();

        self.reads.truncate(cache_len);
        self.reads.fill(usize::MAX);
        self.reads.resize(cache_len, usize::MAX);

        thread_local! {
            static NODES: RefCell<Vec<bool>> = Default::default();
        }

        NODES.with_borrow_mut(|nodes| {
            nodes.truncate(self.binding_count);
            nodes.fill(true);
            nodes.resize(self.binding_count, true);

            for (pass_idx, pass) in graph.cmds[0..end_pass_idx].iter().enumerate() {
                let pass_start = pass_idx * self.binding_count;
                let mut read_count = 0;

                for (&node_idx, accesses) in pass.execs.iter().flat_map(|exec| exec.accesses.iter())
                {
                    self.accesses[pass_start + node_idx] = true;

                    if nodes[node_idx] && is_read_access(accesses.first().unwrap().access) {
                        self.reads[pass_start + read_count] = node_idx;
                        nodes[node_idx] = false;
                        read_count += 1;
                    }
                }

                if pass_idx + 1 < end_pass_idx {
                    nodes.fill(true);
                }

                self.read_count.push(read_count);
            }
        });
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

impl Drop for PhysicalPass {
    fn drop(&mut self) {
        self.exec_descriptor_sets.clear();
        self.descriptor_pool = None;
    }
}

/// A structure which can optimize and submit graphs. This pattern was derived from:
///
/// <http://themaister.net/blog/2017/08/15/render-graphs-and-vulkan-a-deep-dive/>
/// <https://github.com/EmbarkStudios/kajiya>
#[derive(Debug)]
pub struct Queue {
    graph: Graph,
    physical_passes: Vec<PhysicalPass>,
}

impl Queue {
    pub(super) fn new(graph: Graph) -> Self {
        let physical_passes = Vec::with_capacity(graph.cmds.len());

        Self {
            graph,
            physical_passes,
        }
    }

    #[profiling::function]
    fn allow_merge_passes(lhs: &Command, rhs: &Command) -> bool {
        fn first_graphic_pipeline(pass: &Command) -> Option<&GraphicPipeline> {
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

    // See https://vulkan.lunarg.com/doc/view/1.3.204.1/linux/1.3-extensions/vkspec.html#attachment-type-imagelayout
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

    #[profiling::function]
    fn begin_render_pass(
        cmd_buf: &CommandBuffer,
        bindings: &[Resource],
        pass: &Command,
        physical_pass: &mut PhysicalPass,
        render_area: Area,
    ) -> Result<(), DriverError> {
        trace!("  begin render pass");

        let render_pass = physical_pass.render_pass.as_mut().unwrap();
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
            static CLEARS_VIEWS: RefCell<(Vec<vk::ClearValue>, Vec<vk::ImageView>)> = Default::default();
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
                                float32: clear_value.0,
                            },
                        };

                        let image = bindings[attachment.target].as_driver_image().unwrap();

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
                        let image = bindings[attachment.target].as_driver_image().unwrap();

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

                        let image = bindings[attachment.target].as_driver_image().unwrap();

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
                        let image = bindings[attachment.target].as_driver_image().unwrap();

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
                        let image = bindings[attachment.target].as_driver_image().unwrap();

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
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D {
                                x: render_area.x,
                                y: render_area.y,
                            },
                            extent: vk::Extent2D {
                                width: render_area.width,
                                height: render_area.height,
                            },
                        })
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
        depth_stencil: Option<DepthStencilMode>,
    ) -> Result<(), DriverError> {
        if log_enabled!(Trace) {
            let (ty, name, vk_pipeline) = match pipeline {
                ExecutionPipeline::Compute(pipeline) => {
                    ("compute", pipeline.name(), pipeline.handle())
                }
                ExecutionPipeline::Graphic(pipeline) => {
                    ("graphic", pipeline.name(), vk::Pipeline::null())
                }
                ExecutionPipeline::RayTrace(pipeline) => {
                    ("ray trace", pipeline.name(), pipeline.handle())
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
                physical_pass.render_pass.as_mut().unwrap(),
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

    fn end_render_pass(&mut self, cmd_buf: &CommandBuffer) {
        trace!("  end render pass");

        unsafe {
            cmd_buf.device.cmd_end_render_pass(cmd_buf.handle);
        }
    }

    /// Returns `true` when all recorded passes have been submitted to a driver command buffer.
    ///
    /// A fully-resolved graph contains no additional work and may be discarded, although doing so
    /// will stall the GPU while the fences are waited on. It is preferrable to wait a few frame so
    /// that the fences will have already been signalled.
    pub fn is_submitted(&self) -> bool {
        self.graph.cmds.is_empty()
    }

    #[allow(clippy::type_complexity)]
    #[profiling::function]
    fn lease_descriptor_pool<P>(
        pool: &mut P,
        pass: &Command,
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
        for pool_sizes in pass.descriptor_pools_sizes() {
            for pool_size in pool_sizes.values() {
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
                        _ => unimplemented!("{descriptor_ty:?}"),
                    };
                }
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

        Ok(Some(pool.lease(info)?))
    }

    #[profiling::function]
    fn lease_render_pass<P>(
        &self,
        pool: &mut P,
        pass_idx: usize,
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
                    let attachment = attachments.last_mut().unwrap();
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
                .map(|pipeline| pipeline.unwrap_graphic())
                .unwrap();
            let mut subpass_info = SubpassInfo::with_capacity(attachment_count);

            // Add input attachments
            for attachment_idx in pipeline.inner.input_attachments.iter() {
                debug_assert!(
                    !exec.color_clears.contains_key(attachment_idx),
                    "cannot clear color attachment index {attachment_idx} because it uses subpass input",
                );

                let exec_attachment = exec
                    .color_attachments
                    .get(attachment_idx)
                    .or_else(|| exec.color_loads.get(attachment_idx))
                    .or_else(|| exec.color_stores.get(attachment_idx))
                    .expect("subpass input attachment index not attached, loaded, or stored");
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
                for (exec_idx, exec) in pass.execs.iter().enumerate() {
                    // Check accesses
                    'accesses: for (node_idx, accesses) in exec.accesses.iter() {
                        let (mut curr_stages, mut curr_access) =
                            pipeline_stage_access_flags(accesses.first().unwrap().access);
                        if curr_stages.contains(vk::PipelineStageFlags::ALL_COMMANDS) {
                            curr_stages |= vk::PipelineStageFlags::ALL_GRAPHICS;
                            curr_stages &= !vk::PipelineStageFlags::ALL_COMMANDS;
                        }

                        // First look for through earlier executions of this pass (in reverse order)
                        for (prev_exec_idx, prev_exec) in
                            pass.execs[0..exec_idx].iter().enumerate().rev()
                        {
                            if let Some(accesses) = prev_exec.accesses.get(node_idx) {
                                for &SubresourceAccess { access, .. } in accesses {
                                    // Is this previous execution access dependent on anything the current
                                    // execution access is dependent upon?
                                    let (mut prev_stages, prev_access) =
                                        pipeline_stage_access_flags(access);
                                    if prev_stages.contains(vk::PipelineStageFlags::ALL_COMMANDS) {
                                        prev_stages |= vk::PipelineStageFlags::ALL_GRAPHICS;
                                        prev_stages &= !vk::PipelineStageFlags::ALL_COMMANDS;
                                    }

                                    let common_stages = curr_stages & prev_stages;
                                    if common_stages.is_empty() {
                                        // No common dependencies
                                        continue;
                                    }

                                    let dep = dependencies
                                        .entry((prev_exec_idx, exec_idx))
                                        .or_insert_with(|| {
                                            SubpassDependency::new(
                                                prev_exec_idx as _,
                                                exec_idx as _,
                                            )
                                        });

                                    // Wait for ...
                                    dep.src_stage_mask |= common_stages;
                                    dep.src_access_mask |= prev_access;

                                    // ... before we:
                                    dep.dst_stage_mask |= curr_stages;
                                    dep.dst_access_mask |= curr_access;

                                    // Do the source and destination stage masks both include
                                    // framebuffer-space stages?
                                    if (prev_stages | curr_stages).intersects(
                                        vk::PipelineStageFlags::FRAGMENT_SHADER
                                            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                                            | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                                    ) {
                                        dep.dependency_flags |= vk::DependencyFlags::BY_REGION;
                                    }

                                    curr_stages &= !common_stages;
                                    curr_access &= !prev_access;

                                    // Have we found all dependencies for this stage? If so no need to
                                    // check external passes
                                    if curr_stages.is_empty() {
                                        continue 'accesses;
                                    }
                                }
                            }
                        }

                        // Second look in previous passes of the entire render graph
                        for prev_subpass in self.graph.cmds[0..pass_idx]
                            .iter()
                            .rev()
                            .flat_map(|pass| pass.execs.iter().rev())
                        {
                            if let Some(accesses) = prev_subpass.accesses.get(node_idx) {
                                for &SubresourceAccess { access, .. } in accesses {
                                    // Is this previous subpass access dependent on anything the current
                                    // subpass access is dependent upon?
                                    let (prev_stages, prev_access) =
                                        pipeline_stage_access_flags(access);
                                    let common_stages = curr_stages & prev_stages;
                                    if common_stages.is_empty() {
                                        // No common dependencies
                                        continue;
                                    }

                                    let dep = dependencies
                                        .entry((vk::SUBPASS_EXTERNAL as _, exec_idx))
                                        .or_insert_with(|| {
                                            SubpassDependency::new(
                                                vk::SUBPASS_EXTERNAL as _,
                                                exec_idx as _,
                                            )
                                        });

                                    // Wait for ...
                                    dep.src_stage_mask |= common_stages;
                                    dep.src_access_mask |= prev_access;

                                    // ... before we:
                                    dep.dst_stage_mask |=
                                        curr_stages.min(vk::PipelineStageFlags::ALL_GRAPHICS);
                                    dep.dst_access_mask |= curr_access;

                                    // If the source and destination stage masks both include
                                    // framebuffer-space stages then we need the BY_REGION flag
                                    if (prev_stages | curr_stages).intersects(
                                        vk::PipelineStageFlags::FRAGMENT_SHADER
                                            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                                            | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                                    ) {
                                        dep.dependency_flags |= vk::DependencyFlags::BY_REGION;
                                    }

                                    curr_stages &= !common_stages;
                                    curr_access &= !prev_access;

                                    // If we found all dependencies for this stage there is no need to check
                                    // external passes
                                    if curr_stages.is_empty() {
                                        continue 'accesses;
                                    }
                                }
                            }
                        }

                        // Fall back to external dependencies
                        if !curr_stages.is_empty() {
                            let dep = dependencies
                                .entry((vk::SUBPASS_EXTERNAL as _, exec_idx))
                                .or_insert_with(|| {
                                    SubpassDependency::new(vk::SUBPASS_EXTERNAL as _, exec_idx as _)
                                });

                            // Wait for ...
                            dep.src_stage_mask |= curr_stages;
                            dep.src_access_mask |= curr_access;

                            // ... before we:
                            dep.dst_stage_mask |= vk::PipelineStageFlags::TOP_OF_PIPE;
                            dep.dst_access_mask =
                                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
                        }
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

        pool.lease(RenderPassInfo {
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
        for pass_idx in schedule.iter().copied() {
            // At the time this function runs the pass will already have been optimized into a
            // larger pass made out of anything that might have been merged into it - so we
            // only care about one pass at a time here
            let pass = &mut self.graph.cmds[pass_idx];

            trace!("leasing [{pass_idx}: {}]", pass.name());

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
                pass.execs[0].pipeline.is_none()
                    || !pass.execs[0].pipeline.as_ref().unwrap().is_graphic()
                    || pass.execs[0]
                        .pipeline
                        .as_ref()
                        .unwrap()
                        .unwrap_graphic()
                        .inner
                        .descriptor_info
                        .pool_sizes
                        .values()
                        .filter_map(|pool| pool.get(&vk::DescriptorType::INPUT_ATTACHMENT))
                        .next()
                        .is_none()
            );

            // Also the renderpass may just be None if the pass contained no graphic ops.
            let render_pass = if pass.execs[0]
                .pipeline
                .as_ref()
                .map(|pipeline| pipeline.is_graphic())
                .unwrap_or_default()
            {
                Some(self.lease_render_pass(pool, pass_idx)?)
            } else {
                None
            };

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
            static PASSES: RefCell<Vec<Option<Command>>> = Default::default();
        }

        PASSES.with_borrow_mut(|passes| {
            debug_assert!(passes.is_empty());

            passes.extend(self.graph.cmds.drain(..).map(Some));

            let mut idx = 0;

            // debug!("attempting to merge {} passes", schedule.len(),);

            while idx < schedule.len() {
                let mut pass = passes[schedule[idx]].take().unwrap();

                // Find candidates
                let start = idx + 1;
                let mut end = start;
                while end < schedule.len() {
                    let other = passes[schedule[end]].as_ref().unwrap();

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

                let mut name = pass.name.take().unwrap_or_default();

                // Grow the merged pass once, not per merge
                {
                    let mut name_additional = 0;
                    let mut execs_additional = 0;
                    for idx in start..end {
                        let other = passes[schedule[idx]].as_ref().unwrap();
                        name_additional += other.name().len() + 3;
                        execs_additional += other.execs.len();
                    }

                    name.reserve(name_additional);
                    pass.execs.reserve(execs_additional);
                }

                for idx in start..end {
                    let mut other = passes[schedule[idx]].take().unwrap();
                    name.push_str(" + ");
                    name.push_str(other.name());
                    pass.execs.append(&mut other.execs);
                }

                pass.name = Some(name);

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

    fn next_subpass(cmd_buf: &CommandBuffer) {
        trace!("next_subpass");

        unsafe {
            cmd_buf
                .device
                .cmd_next_subpass(cmd_buf.handle, vk::SubpassContents::INLINE);
        }
    }

    /// Returns the stages that process the given node.
    ///
    /// Note that this value must be retrieved before resolving a node as there will be no
    /// data left to inspect afterwards!
    #[profiling::function]
    pub fn node_stages(&self, node: impl Node) -> vk::PipelineStageFlags {
        let node_idx = node.index();
        let mut res = Default::default();

        'pass: for pass in self.graph.cmds.iter() {
            for exec in pass.execs.iter() {
                if exec.accesses.contains_key(&node_idx) {
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
        bindings: &mut [Resource],
        accesses: impl Iterator<Item = (&'a NodeIndex, &'a Vec<SubresourceAccess>)>,
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

            // Map remaining accesses into vk_sync barriers (some accesses may have been removed by the
            // render pass leasing function)

            for (node_idx, accesses) in accesses {
                let binding = &bindings[*node_idx];

                match binding {
                    Resource::AccelerationStructure(..)
                    | Resource::AccelerationStructureLease(..) => {
                        let Some(accel_struct) = binding.as_driver_accel_struct() else {
                            #[cfg(debug_assertions)]
                            unreachable!();

                            #[cfg(not(debug_assertions))]
                            unsafe {
                                unreachable_unchecked()
                            }
                        };

                        let prev_access = AccelerationStructure::access(
                            accel_struct,
                            accesses.last().unwrap().access,
                        );

                        tls.next_accesses.extend(
                            accesses
                                .iter()
                                .map(|&SubresourceAccess { access, .. }| access),
                        );
                        tls.prev_accesses.push(prev_access);
                    }
                    Resource::Buffer(..) | Resource::BufferLease(..) => {
                        let Some(buffer) = binding.as_driver_buffer() else {
                            #[cfg(debug_assertions)]
                            unreachable!();

                            #[cfg(not(debug_assertions))]
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
                    Resource::Image(..)
                    | Resource::ImageLease(..)
                    | Resource::SwapchainImage(..) => {
                        let Some(image) = binding.as_driver_image() else {
                            #[cfg(debug_assertions)]
                            unreachable!();

                            #[cfg(not(debug_assertions))]
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
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        image,
                        range,
                    }
                },
            );

            pipeline_barrier(
                &cmd_buf.device,
                cmd_buf.handle,
                global_barrier,
                &buffer_barriers.collect::<Box<[_]>>(),
                &image_barriers.collect::<Box<[_]>>(),
            );
        });
    }

    #[profiling::function]
    fn record_image_layout_transitions(
        cmd_buf: &CommandBuffer,
        bindings: &mut [Resource],
        pass: &mut Command,
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
            initial_layouts: HashMap<usize, ImageAccess<bool>>,
        }

        TLS.with_borrow_mut(|tls| {
            tls.images.clear();
            tls.initial_layouts.clear();

            for (node_idx, accesses) in pass
                .execs
                .iter_mut()
                .flat_map(|exec| exec.accesses.iter())
                .map(|(node_idx, accesses)| (*node_idx, accesses))
            {
                debug_assert!(bindings.get(node_idx).is_some());

                let binding = unsafe {
                    // CommandRef enforces this during push_node_access
                    bindings.get_unchecked(node_idx)
                };

                match binding {
                    Resource::AccelerationStructure(..)
                    | Resource::AccelerationStructureLease(..) => {
                        let Some(accel_struct) = binding.as_driver_accel_struct() else {
                            #[cfg(debug_assertions)]
                            unreachable!();

                            #[cfg(not(debug_assertions))]
                            unsafe {
                                unreachable_unchecked()
                            }
                        };

                        AccelerationStructure::access(accel_struct, AccessType::Nothing);
                    }
                    Resource::Buffer(..) | Resource::BufferLease(..) => {
                        let Some(buffer) = binding.as_driver_buffer() else {
                            #[cfg(debug_assertions)]
                            unreachable!();

                            #[cfg(not(debug_assertions))]
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
                                #[cfg(debug_assertions)]
                                unreachable!();

                                #[cfg(not(debug_assertions))]
                                unsafe {
                                    // This cannot be reached because PassRef enforces the subrange is
                                    // of type N::Subresource where N is the image node type
                                    unreachable_unchecked()
                                }
                            };

                            for _ in Buffer::access(buffer, AccessType::Nothing, access_range) {}
                        }
                    }
                    Resource::Image(..)
                    | Resource::ImageLease(..)
                    | Resource::SwapchainImage(..) => {
                        let Some(image) = binding.as_driver_image() else {
                            #[cfg(debug_assertions)]
                            unreachable!();

                            #[cfg(not(debug_assertions))]
                            unsafe {
                                unreachable_unchecked()
                            }
                        };

                        let initial_layout = tls
                            .initial_layouts
                            .entry(node_idx)
                            .or_insert_with(|| ImageAccess::new(image.info, true));

                        for subresource_access in accesses {
                            let &SubresourceAccess {
                                access,
                                subresource: SubresourceRange::Image(access_range),
                            } = subresource_access
                            else {
                                #[cfg(debug_assertions)]
                                unreachable!();

                                #[cfg(not(debug_assertions))]
                                unsafe {
                                    // This cannot be reached because PassRef enforces the subrange is
                                    // of type N::Subresource where N is the image node type
                                    unreachable_unchecked()
                                }
                            };

                            for (initial_layout, layout_range) in
                                initial_layout.access(false, access_range)
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
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
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
            schedule.access_cache.update(&self.graph, end_pass_idx);
            schedule.passes.clear();

            self.schedule_node_passes(node_idx, end_pass_idx, schedule);
            self.record_scheduled_passes(pool, cmd_buf, schedule, end_pass_idx)
        })
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
        // #[cfg(debug_assertions)]
        // if log_enabled!(Debug) && self.graph.debug {
        //     debug!("resolving the following graph:\n\n{:#?}\n\n", self.graph);
        // }

        debug_assert!(
            schedule.passes.windows(2).all(|w| w[0] <= w[1]),
            "Unsorted schedule"
        );

        // Optimize the schedule; leasing the required stuff it needs
        Self::reorder_scheduled_passes(schedule, end_pass_idx);
        self.merge_scheduled_passes(&mut schedule.passes);
        self.lease_scheduled_resources(pool, &schedule.passes)?;

        for pass_idx in schedule.passes.iter().copied() {
            let pass = &mut self.graph.cmds[pass_idx];

            profiling::scope!("Pass", &pass.name);

            let physical_pass = &mut self.physical_passes[pass_idx];
            let is_graphic = physical_pass.render_pass.is_some();

            trace!("recording pass [{}: {}]", pass_idx, pass.name());

            if !physical_pass.exec_descriptor_sets.is_empty() {
                Self::write_descriptor_sets(cmd_buf, &self.graph.resources, pass, physical_pass)?;
            }

            let render_area = if is_graphic {
                Self::record_image_layout_transitions(cmd_buf, &mut self.graph.resources, pass);

                let render_area = Self::render_area(&self.graph.resources, pass);

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
                        .unwrap_or(render_area.unwrap())
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
                        let render_area = render_area.unwrap();

                        // In this case we set the viewport and scissor for the user
                        Self::set_viewport(
                            cmd_buf,
                            render_area.x as _,
                            render_area.y as _,
                            render_area.width as _,
                            render_area.height as _,
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
                            render_area.x,
                            render_area.y,
                            render_area.width,
                            render_area.height,
                        );
                    }

                    Self::bind_descriptor_sets(cmd_buf, pipeline, physical_pass, exec_idx);
                }

                if !is_graphic {
                    Self::record_execution_barriers(
                        cmd_buf,
                        &mut self.graph.resources,
                        exec.accesses.iter(),
                    );
                }

                trace!("    > exec[{exec_idx}]");

                {
                    profiling::scope!("Execute callback");

                    let exec_func = exec.func.take().unwrap().0;
                    exec_func(
                        cmd_buf,
                        Resources::new(
                            &self.graph.resources,
                            #[cfg(debug_assertions)]
                            exec,
                        ),
                    );
                }
            }

            if is_graphic {
                self.end_render_pass(cmd_buf);
            }
        }

        thread_local! {
            static PASSES: RefCell<Vec<Command>> = Default::default();
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

                        cmd_buf.drop_after_executed((pass, self.physical_passes.pop().unwrap()));
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
    fn render_area(bindings: &[Resource], pass: &Command) -> Area {
        // set_render_area was not specified so we're going to guess using the minimum common
        // attachment extents
        let first_exec = pass.execs.first().unwrap();

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
                let info = bindings[attachment.target].as_driver_image().unwrap().info;

                (
                    info.width >> attachment.base_mip_level,
                    info.height >> attachment.base_mip_level,
                )
            })
        {
            width = width.min(attachment_width);
            height = height.min(attachment_height);
        }

        Area {
            height,
            width,
            x: 0,
            y: 0,
        }
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
                    .access_cache
                    .interdependent_passes(pass_idx, end_pass_idx)
                    .count();

                for (idx, pass_idx) in schedule.passes[best_idx + 1..schedule.passes.len()]
                    .iter()
                    .enumerate()
                {
                    let mut overlap_factor = 0;

                    for other_pass_idx in schedule
                        .access_cache
                        .interdependent_passes(*pass_idx, end_pass_idx)
                    {
                        if unscheduled[other_pass_idx] {
                            // This pass can't be the candidate because it depends on unfinished work
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
    pub fn resource<N>(&self, node: N) -> &N::Resource
    where
        N: Bound,
    {
        self.graph.resource(node)
    }

    /// Returns a vec of pass indexes that are required to be executed, in order, for the given
    /// node.
    #[profiling::function]
    fn schedule_node_passes(&self, node_idx: usize, end_pass_idx: usize, schedule: &mut Schedule) {
        type UnscheduledUnresolvedUnchecked = (Vec<bool>, Vec<bool>, VecDeque<(usize, usize)>);

        thread_local! {
            static UNSCHEDULED_UNRESOLVED_UNCHECKED: RefCell<UnscheduledUnresolvedUnchecked> = Default::default();
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
                .access_cache
                .dependent_passes(node_idx, end_pass_idx)
            {
                trace!(
                    "  pass [{pass_idx}: {}] is dependent",
                    self.graph.cmds[pass_idx].name()
                );

                debug_assert!(unscheduled[pass_idx]);

                unscheduled[pass_idx] = false;
                schedule.passes.push(pass_idx);

                for node_idx in schedule.access_cache.dependent_nodes(pass_idx) {
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
                    .access_cache
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

                        for node_idx in schedule.access_cache.dependent_nodes(pass_idx) {
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

    /// Submits the remaining commands stored in this instance.
    #[profiling::function]
    pub fn submit<P>(
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

        let mut cmd_buf = pool.lease(CommandBufferInfo::new(queue_family_index as _))?;

        cmd_buf.wait_until_executed()?;

        unsafe {
            cmd_buf
                .device
                .begin_command_buffer(
                    cmd_buf.handle,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|_| DriverError::OutOfMemory)?;
        }

        self.submit_cmd_buf(pool, &mut cmd_buf)?;

        let queue = Device::queue(&cmd_buf.device, queue_family_index, queue_index);

        unsafe {
            cmd_buf
                .device
                .end_command_buffer(cmd_buf.handle)
                .map_err(|_| DriverError::OutOfMemory)?;
            cmd_buf
                .device
                .reset_fences(slice::from_ref(&cmd_buf.fence))
                .map_err(|_| DriverError::OutOfMemory)?;
            cmd_buf
                .device
                .queue_submit(
                    queue,
                    slice::from_ref(
                        &vk::SubmitInfo::default()
                            .command_buffers(slice::from_ref(&cmd_buf.handle)),
                    ),
                    cmd_buf.fence,
                )
                .map_err(|_| DriverError::OutOfMemory)?;
        }

        // This graph contains references to buffers, images, and other resources which must be kept
        // alive until this graph execution completes on the GPU. Once those references are dropped
        // they will return to the pool for other things to use. The drop will happen the next time
        // someone tries to lease a command buffer and we notice this one has returned and the fence
        // has been signalled.
        cmd_buf.drop_after_executed(self);

        Ok(cmd_buf)
    }

    /// Records any pending render graph passes that have not been previously scheduled.
    #[profiling::function]
    pub fn submit_cmd_buf<P>(
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
                .access_cache
                .update(&self.graph, self.graph.cmds.len());
            schedule.passes.clear();
            schedule.passes.extend(0..self.graph.cmds.len());

            self.record_scheduled_passes(pool, cmd_buf, schedule, self.graph.cmds.len())
        })
    }

    /// Records any pending render graph passes that the given node requires.
    #[profiling::function]
    pub fn submit_resource<P>(
        &mut self,
        node: impl Node,
        pool: &mut P,
        cmd_buf: &mut CommandBuffer,
    ) -> Result<(), DriverError>
    where
        P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
    {
        let node_idx = node.index();

        debug_assert!(self.graph.resources.get(node_idx).is_some());

        if self.graph.cmds.is_empty() {
            return Ok(());
        }

        let end_pass_idx = self.graph.cmds.len();
        self.record_node_passes(pool, cmd_buf, node_idx, end_pass_idx)
    }

    /// Records any pending render graph passes that are required by the given node, but does not
    /// record any passes that actually contain the given node.
    ///
    /// As a side effect, the graph is optimized for the given node. Future calls may further optimize
    /// the graph, but only on top of the existing optimizations. This only matters if you are pulling
    /// multiple images out and you care - in that case pull the "most important" image first.
    #[profiling::function]
    pub fn submit_resource_dependencies<P>(
        &mut self,
        node: impl Node,
        pool: &mut P,
        cmd_buf: &mut CommandBuffer,
    ) -> Result<(), DriverError>
    where
        P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
    {
        let node_idx = node.index();

        debug_assert!(self.graph.resources.get(node_idx).is_some());

        // We record up to but not including the first pass which accesses the target node
        if let Some(end_pass_idx) = self.graph.first_node_access_pass_index(node) {
            self.record_node_passes(pool, cmd_buf, node_idx, end_pass_idx)?;
        }

        Ok(())
    }

    #[profiling::function]
    fn write_descriptor_sets(
        cmd_buf: &CommandBuffer,
        bindings: &[Resource],
        pass: &Command,
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
                let (descriptor_info, _) = pipeline
                        .descriptor_bindings()
                        .get(&Descriptor { set: descriptor_set_idx, binding: dst_binding })
                        .unwrap_or_else(|| panic!("descriptor {descriptor_set_idx}.{dst_binding}[{binding_offset}] specified in recorded execution of pass \"{}\" was not discovered through shader reflection", pass.name()));
                let descriptor_type = descriptor_info.descriptor_type();
                let bound_node = &bindings[*node_idx];
                if let Some(image) = bound_node.as_driver_image() {
                    let mut image_view_info = *view_info.as_image().unwrap();

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
                        _ => unimplemented!("{descriptor_type:?}"),
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
                        tls.image_writes.last_mut().unwrap().write.descriptor_count += 1;
                    }

                    tls.image_infos.push(
                        vk::DescriptorImageInfo::default()
                            .image_layout(image_layout)
                            .image_view(image_view),
                    );
                } else if let Some(buffer) = bound_node.as_driver_buffer() {
                    let buffer_view_info = view_info.as_buffer().unwrap();

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
                        tls.buffer_writes.last_mut().unwrap().write.descriptor_count += 1;
                    }

                    tls.buffer_infos.push(
                        vk::DescriptorBufferInfo::default()
                            .buffer(buffer.handle)
                            .offset(buffer_view_info.start)
                            .range(buffer_view_info.end - buffer_view_info.start),
                    );
                } else if let Some(accel_struct) = bound_node.as_driver_accel_struct() {
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
                            .unwrap()
                            .write
                            .descriptor_count += 1;
                    }

                    tls.accel_struct_infos.push(
                        vk::WriteDescriptorSetAccelerationStructureKHR::default()
                            .acceleration_structures(std::slice::from_ref(&accel_struct.handle)),
                    );
                } else {
                    unimplemented!();
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
                            let (attachment, write_exec) = pass.execs[0..exec_idx]
                                .iter()
                                .rev()
                                .find_map(|exec| {
                                    exec.color_stores
                                        .get(&attachment_idx)
                                        .copied()
                                        .map(|attachment| (attachment, exec))
                                        .or_else(|| {
                                            exec.color_resolves.get(&attachment_idx).map(
                                                |(resolved_attachment, _)| {
                                                    (*resolved_attachment, exec)
                                                },
                                            )
                                        })
                                })
                                .expect("input attachment not written");
                            let late = &write_exec.accesses[&attachment.target].last().unwrap();
                            let image_range = late.subresource.as_image().unwrap();
                            let image_binding = &bindings[attachment.target];
                            let image = image_binding.as_driver_image().unwrap();
                            let image_view_info = attachment
                                .image_view_info(image.info)
                                .to_builder()
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

#[derive(Default)]
struct Schedule {
    access_cache: AccessCache,
    passes: Vec<usize>,
}

#[allow(private_bounds)]
#[allow(unused)]
mod derecated {
    use crate::{
        Queue,
        driver::{
            DriverError,
            cmd_buf::CommandBuffer,
            descriptor_set::{DescriptorPool, DescriptorPoolInfo},
            render_pass::{RenderPass, RenderPassInfo},
        },
        node::Node,
        pool::Pool,
    };

    impl Queue {
        #[deprecated = "use is_submitted function"]
        #[doc(hidden)]
        pub fn is_resolved(&self) -> bool {
            self.is_submitted()
        }

        #[deprecated = "use submit_resource function"]
        #[doc(hidden)]
        pub fn record_node<P>(
            &mut self,
            pool: &mut P,
            cmd_buf: &mut CommandBuffer,
            node: impl Node,
        ) -> Result<(), DriverError>
        where
            P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
        {
            self.submit_resource(node, pool, cmd_buf)
        }

        #[deprecated = "use submit_resource function"]
        #[doc(hidden)]
        pub fn record_node_dependencies<P>(
            &mut self,
            pool: &mut P,
            cmd_buf: &mut CommandBuffer,
            node: impl Node,
        ) -> Result<(), DriverError>
        where
            P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
        {
            self.submit_resource_dependencies(node, pool, cmd_buf)
        }
    }
}
