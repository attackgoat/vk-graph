//! Render pass related types.

use {
    super::{
        DriverError,
        device::Device,
        graphics::{DepthStencilInfo, GraphicsPipeline},
        image::SampleCount,
    },
    ash::vk::{self, Handle},
    log::{trace, warn},
    std::{
        collections::{HashMap, hash_map::Entry},
        slice,
        thread::panicking,
    },
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct AttachmentInfo {
    pub flags: vk::AttachmentDescriptionFlags,
    pub format: vk::Format,
    pub sample_count: SampleCount,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub stencil_load_op: vk::AttachmentLoadOp,
    pub stencil_store_op: vk::AttachmentStoreOp,
    pub initial_layout: vk::ImageLayout,
    pub final_layout: vk::ImageLayout,
}

impl From<AttachmentInfo> for vk::AttachmentDescription2<'_> {
    fn from(value: AttachmentInfo) -> Self {
        vk::AttachmentDescription2::default()
            .flags(value.flags)
            .format(value.format)
            .samples(value.sample_count.into())
            .load_op(value.load_op)
            .store_op(value.store_op)
            .stencil_load_op(value.stencil_load_op)
            .stencil_store_op(value.stencil_store_op)
            .initial_layout(value.initial_layout)
            .final_layout(value.final_layout)
    }
}

impl Default for AttachmentInfo {
    fn default() -> Self {
        AttachmentInfo {
            flags: vk::AttachmentDescriptionFlags::MAY_ALIAS,
            format: vk::Format::UNDEFINED,
            sample_count: SampleCount::Type1,
            initial_layout: vk::ImageLayout::UNDEFINED,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            final_layout: vk::ImageLayout::UNDEFINED,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct AttachmentRef {
    pub attachment: u32,
    pub aspect_mask: vk::ImageAspectFlags,
    pub layout: vk::ImageLayout,
}

impl From<AttachmentRef> for vk::AttachmentReference2<'_> {
    fn from(attachment_ref: AttachmentRef) -> Self {
        vk::AttachmentReference2::default()
            .attachment(attachment_ref.attachment)
            .aspect_mask(attachment_ref.aspect_mask)
            .layout(attachment_ref.layout)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct FramebufferAttachmentImageInfo {
    pub flags: vk::ImageCreateFlags,
    pub usage: vk::ImageUsageFlags,
    pub width: u32,
    pub height: u32,
    pub layer_count: u32,
    pub view_formats: Vec<vk::Format>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct FramebufferInfo {
    pub attachments: Vec<FramebufferAttachmentImageInfo>,
}

#[derive(Debug, Eq, Hash, PartialEq)]
struct GraphicPipelineKey {
    depth_stencil: Option<DepthStencilInfo>,
    layout: vk::PipelineLayout,
    subpass_idx: u32,
}

/// Vulkan render pass state and cached framebuffer/pipeline objects for compatible attachments.
///
/// See [`VkRenderPass`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderPass.html).
#[derive(Debug)]
#[read_only::cast]
pub(crate) struct RenderPass {
    /// The device which owns this render pass resource.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub device: Device,

    framebuffers: HashMap<FramebufferInfo, vk::Framebuffer>,
    graphic_pipelines: HashMap<GraphicPipelineKey, vk::Pipeline>,

    /// The native Vulkan resource handle of this render pass.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub handle: vk::RenderPass,

    /// Information used to create this render pass resource.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub(crate) info: RenderPassInfo,
}

impl RenderPass {
    #[profiling::function]
    pub(crate) fn create(device: &Device, info: RenderPassInfo) -> Result<Self, DriverError> {
        trace!("create");

        let device = device.clone();
        let attachments = info
            .attachments
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Box<_>>();
        let correlated_view_masks = if info.subpasses.iter().any(|subpass| subpass.view_mask != 0) {
            {
                info.subpasses
                    .iter()
                    .map(|subpass| subpass.correlated_view_mask)
                    .collect::<Box<_>>()
            }
        } else {
            Default::default()
        };
        let dependencies = info
            .dependencies
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Box<_>>();

        let subpass_attachments = info
            .subpasses
            .iter()
            .flat_map(|subpass| {
                subpass
                    .color_attachments
                    .iter()
                    .chain(subpass.input_attachments.iter())
                    .chain(subpass.color_resolve_attachments.iter())
                    .chain(subpass.depth_stencil_attachment.iter())
                    .chain(
                        subpass
                            .depth_stencil_resolve_attachment
                            .as_ref()
                            .map(|(resolve_attachment, _, _)| resolve_attachment)
                            .into_iter(),
                    )
                    .copied()
                    .map(AttachmentRef::into)
            })
            .collect::<Box<[vk::AttachmentReference2]>>();
        let mut subpass_depth_stencil_resolves = info
            .subpasses
            .iter()
            .map(|subpass| {
                subpass.depth_stencil_resolve_attachment.map(
                    |(_, depth_resolve_mode, stencil_resolve_mode)| {
                        vk::SubpassDescriptionDepthStencilResolve::default()
                            .depth_resolve_mode(
                                depth_resolve_mode.map(Into::into).unwrap_or_default(),
                            )
                            .stencil_resolve_mode(
                                stencil_resolve_mode.map(Into::into).unwrap_or_default(),
                            )
                    },
                )
            })
            .collect::<Box<_>>();
        let mut subpasses = Vec::with_capacity(info.subpasses.len());

        let mut base_idx = 0;
        for (subpass, depth_stencil_resolve) in info
            .subpasses
            .iter()
            .zip(subpass_depth_stencil_resolves.iter_mut())
        {
            let mut desc = vk::SubpassDescription2::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

            debug_assert_eq!(
                subpass.color_attachments.len(),
                subpass.color_resolve_attachments.len()
            );

            let color_idx = base_idx;
            let input_idx = color_idx + subpass.color_attachments.len();
            let color_resolve_idx = input_idx + subpass.input_attachments.len();
            let depth_stencil_idx = color_resolve_idx + subpass.color_resolve_attachments.len();
            let depth_stencil_resolve_idx =
                depth_stencil_idx + subpass.depth_stencil_attachment.is_some() as usize;
            base_idx = depth_stencil_resolve_idx
                + subpass.depth_stencil_resolve_attachment.is_some() as usize;

            if subpass.depth_stencil_attachment.is_some() {
                desc = desc.depth_stencil_attachment(&subpass_attachments[depth_stencil_idx]);
            }

            if let Some(depth_stencil_resolve) = depth_stencil_resolve {
                *depth_stencil_resolve = depth_stencil_resolve.depth_stencil_resolve_attachment(
                    &subpass_attachments[depth_stencil_resolve_idx],
                );
                desc = desc.push_next(depth_stencil_resolve);
            }

            subpasses.push(
                desc.color_attachments(&subpass_attachments[color_idx..input_idx])
                    .input_attachments(&subpass_attachments[input_idx..color_resolve_idx])
                    .resolve_attachments(&subpass_attachments[color_resolve_idx..depth_stencil_idx])
                    .preserve_attachments(&subpass.preserve_attachments)
                    .view_mask(subpass.view_mask),
            );
        }

        let handle = unsafe {
            device.create_render_pass2(
                &vk::RenderPassCreateInfo2::default()
                    .attachments(&attachments)
                    .correlated_view_masks(&correlated_view_masks)
                    .dependencies(&dependencies)
                    .subpasses(&subpasses),
                None,
            )
        }
        .map_err(|err| {
            warn!("unable to create render pass: {err}");

            DriverError::Unsupported
        })?;

        Ok(Self {
            device,
            framebuffers: Default::default(),
            graphic_pipelines: Default::default(),
            handle,
            info,
        })
    }

    #[profiling::function]
    pub(crate) fn framebuffer(
        &mut self,
        info: FramebufferInfo,
    ) -> Result<vk::Framebuffer, DriverError> {
        debug_assert!(!info.attachments.is_empty());

        let entry = self.framebuffers.entry(info);
        if let Entry::Occupied(entry) = entry {
            return Ok(*entry.get());
        }

        let entry = match entry {
            Entry::Vacant(entry) => entry,
            _ => unreachable!(),
        };

        let key = entry.key();
        let layers = key
            .attachments
            .iter()
            .map(|attachment| attachment.layer_count)
            .max()
            .unwrap_or(1);
        let attachments = key
            .attachments
            .iter()
            .map(|attachment| {
                vk::FramebufferAttachmentImageInfo::default()
                    .flags(attachment.flags)
                    .width(attachment.width)
                    .height(attachment.height)
                    .layer_count(attachment.layer_count)
                    .usage(attachment.usage)
                    .view_formats(&attachment.view_formats)
            })
            .collect::<Box<_>>();
        let mut imageless_info =
            vk::FramebufferAttachmentsCreateInfoKHR::default().attachment_image_infos(&attachments);
        let mut create_info = vk::FramebufferCreateInfo::default()
            .flags(vk::FramebufferCreateFlags::IMAGELESS)
            .render_pass(self.handle)
            .width(attachments[0].width)
            .height(attachments[0].height)
            .layers(layers)
            .push_next(&mut imageless_info);
        create_info.attachment_count = self.info.attachments.len() as _;

        let framebuffer =
            unsafe { self.device.create_framebuffer(&create_info, None) }.map_err(|err| {
                warn!("unable to create framebuffer: {err}");

                DriverError::Unsupported
            })?;

        entry.insert(framebuffer);

        Ok(framebuffer)
    }

    #[profiling::function]
    pub(crate) fn pipeline_handle(
        &mut self,
        pipeline: &GraphicsPipeline,
        depth_stencil: Option<DepthStencilInfo>,
        subpass_idx: u32,
    ) -> Result<vk::Pipeline, DriverError> {
        let entry = self.graphic_pipelines.entry(GraphicPipelineKey {
            depth_stencil,
            layout: pipeline.inner.layout,
            subpass_idx,
        });
        if let Entry::Occupied(entry) = entry {
            let pipeline_handle = *entry.get();

            if let Some(name) = Device::private_data_object_name(
                &self.device,
                vk::ObjectType::PIPELINE_LAYOUT,
                pipeline.inner.layout,
            ) && match Device::private_data_object_name(
                &self.device,
                vk::ObjectType::PIPELINE,
                pipeline_handle,
            ) {
                None => true,
                Some(previous) => previous != name,
            } {
                pipeline.set_variant_debug_name(pipeline_handle, self.handle, subpass_idx, &name);
            }

            return Ok(pipeline_handle);
        }

        let entry = match entry {
            Entry::Vacant(entry) => entry,
            _ => unreachable!(),
        };

        let color_blend_attachment_states = self.info.subpasses[subpass_idx as usize]
            .color_attachments
            .iter()
            .map(|_| pipeline.inner.info.blend.into())
            .collect::<Box<_>>();
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&color_blend_attachment_states);
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .alpha_to_coverage_enable(pipeline.inner.multisample.alpha_to_coverage_enable)
            .alpha_to_one_enable(pipeline.inner.multisample.alpha_to_one_enable)
            .flags(pipeline.inner.multisample.flags)
            .min_sample_shading(pipeline.inner.multisample.min_sample_shading)
            .rasterization_samples(pipeline.inner.multisample.rasterization_samples.into())
            .sample_shading_enable(pipeline.inner.multisample.sample_shading_enable)
            .sample_mask(&pipeline.inner.multisample.sample_mask);
        let specializations = pipeline
            .inner
            .shader_stages
            .iter()
            .map(|stage| stage.specialization.as_ref().map(Into::into))
            .collect::<Box<_>>();
        let stages = pipeline
            .inner
            .shader_stages
            .iter()
            .zip(specializations.iter())
            .map(|(stage, specialization)| {
                let mut info = vk::PipelineShaderStageCreateInfo::default()
                    .module(stage.module)
                    .name(&stage.name)
                    .stage(stage.flags);

                if let Some(specialization) = specialization {
                    info = info.specialization_info(specialization);
                }

                info
            })
            .collect::<Box<_>>();
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(
                &pipeline.inner.vertex_input.vertex_attribute_descriptions,
            )
            .vertex_binding_descriptions(&pipeline.inner.vertex_input.vertex_binding_descriptions);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: pipeline.inner.info.topology,
            ..Default::default()
        };
        let depth_stencil = depth_stencil.map(Into::into).unwrap_or_default();
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            front_face: pipeline.inner.info.front_face,
            line_width: 1.0,
            polygon_mode: pipeline.inner.info.polygon_mode,
            cull_mode: pipeline.inner.info.cull_mode,
            ..Default::default()
        };
        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .color_blend_state(&color_blend_state)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_state)
            .input_assembly_state(&input_assembly_state)
            .layout(pipeline.inner.layout)
            .multisample_state(&multisample_state)
            .rasterization_state(&rasterization_state)
            .render_pass(self.handle)
            .stages(&stages)
            .subpass(subpass_idx)
            .vertex_input_state(&vertex_input_state)
            .viewport_state(&viewport_state);

        let pipeline_handle = unsafe {
            self.device.create_graphics_pipelines(
                Device::pipeline_cache(&self.device),
                slice::from_ref(&create_info),
                None,
            )
        }
        .map_err(|(_, err)| {
            warn!("create_graphics_pipelines: {err}\n{:#?}", create_info);

            DriverError::Unsupported
        })?
        .into_iter()
        .find(|handle| !handle.is_null())
        .ok_or_else(|| {
            warn!("missing pipeline handle");

            DriverError::Unsupported
        })?;

        if let Some(name) = Device::private_data_object_name(
            &self.device,
            vk::ObjectType::PIPELINE_LAYOUT,
            pipeline.inner.layout,
        ) && match Device::private_data_object_name(
            &self.device,
            vk::ObjectType::PIPELINE,
            pipeline_handle,
        ) {
            None => true,
            Some(previous) => previous != name,
        } {
            pipeline.set_variant_debug_name(pipeline_handle, self.handle, subpass_idx, &name);
        }

        entry.insert(pipeline_handle);

        Ok(pipeline_handle)
    }
}

impl Drop for RenderPass {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        for (_, framebuffer) in self.framebuffers.drain() {
            unsafe {
                self.device.destroy_framebuffer(framebuffer, None);
            }
        }

        for (_, pipeline) in self.graphic_pipelines.drain() {
            Device::clear_private_data_object_name(
                &self.device,
                vk::ObjectType::PIPELINE,
                pipeline,
            )
            .unwrap_or_else(|err| warn!("unable to clear private data object name: {err}"));

            unsafe {
                self.device.destroy_pipeline(pipeline, None);
            }
        }

        unsafe {
            self.device.destroy_render_pass(self.handle, None);
        }
    }
}

/// Attachment, subpass, and dependency information used to create a [`RenderPass`].
///
/// See [`VkRenderPassCreateInfo2`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderPassCreateInfo2.html).
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct RenderPassInfo {
    pub(crate) attachments: Vec<AttachmentInfo>,
    pub(crate) subpasses: Vec<SubpassInfo>,
    pub(crate) dependencies: Vec<SubpassDependency>,
}

/// Specifies depth and stencil resolve modes.
///
/// See [`VkResolveModeFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkResolveModeFlagBits.html).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResolveMode {
    /// The result of the resolve operation is the average of the sample values.
    Average,

    /// The result of the resolve operation is the maximum of the sample values.
    Maximum,

    /// The result of the resolve operation is the minimum of the sample values.
    Minimum,

    /// The result of the resolve operation is equal to the value of sample `0`.
    SampleZero,
}

impl From<ResolveMode> for vk::ResolveModeFlags {
    fn from(mode: ResolveMode) -> Self {
        match mode {
            ResolveMode::Average => vk::ResolveModeFlags::AVERAGE,
            ResolveMode::Maximum => vk::ResolveModeFlags::MAX,
            ResolveMode::Minimum => vk::ResolveModeFlags::MIN,
            ResolveMode::SampleZero => vk::ResolveModeFlags::SAMPLE_ZERO,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct SubpassDependency {
    pub src_subpass: u32,
    pub dst_subpass: u32,
    pub src_stage_mask: vk::PipelineStageFlags,
    pub dst_stage_mask: vk::PipelineStageFlags,
    pub src_access_mask: vk::AccessFlags,
    pub dst_access_mask: vk::AccessFlags,
    pub dependency_flags: vk::DependencyFlags,
}

impl SubpassDependency {
    pub fn new(src_subpass: u32, dst_subpass: u32) -> Self {
        Self {
            src_subpass,
            dst_subpass,
            src_stage_mask: vk::PipelineStageFlags::empty(),
            dst_stage_mask: vk::PipelineStageFlags::empty(),
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::empty(),
            dependency_flags: vk::DependencyFlags::empty(),
        }
    }
}

impl From<SubpassDependency> for vk::SubpassDependency2<'_> {
    fn from(value: SubpassDependency) -> Self {
        vk::SubpassDependency2::default()
            .src_subpass(value.src_subpass)
            .dst_subpass(value.dst_subpass)
            .src_stage_mask(value.src_stage_mask)
            .dst_stage_mask(value.dst_stage_mask)
            .src_access_mask(value.src_access_mask)
            .dst_access_mask(value.dst_access_mask)
            .dependency_flags(value.dependency_flags)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct SubpassInfo {
    pub color_attachments: Vec<AttachmentRef>,
    pub color_resolve_attachments: Vec<AttachmentRef>,
    pub correlated_view_mask: u32,
    pub depth_stencil_attachment: Option<AttachmentRef>,
    pub depth_stencil_resolve_attachment:
        Option<(AttachmentRef, Option<ResolveMode>, Option<ResolveMode>)>,
    pub input_attachments: Vec<AttachmentRef>,
    pub preserve_attachments: Vec<u32>,
    pub view_mask: u32,
}

impl SubpassInfo {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            color_attachments: Vec::with_capacity(capacity),
            color_resolve_attachments: Vec::with_capacity(capacity),
            correlated_view_mask: 0,
            depth_stencil_attachment: None,
            depth_stencil_resolve_attachment: None,
            input_attachments: Vec::with_capacity(capacity),
            preserve_attachments: Vec::with_capacity(capacity),
            view_mask: 0,
        }
    }
}
