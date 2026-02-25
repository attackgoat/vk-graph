use {
    bytemuck::cast_slice,
    glam::{vec3, Mat4},
    vk_graph_prelude::*,
    vk_shader_macros::include_glsl,
};

/// TODO
pub struct ComputePresenter([ComputePipeline; 2]);

impl ComputePresenter {
    /// TODO
    pub fn new(device: &Device) -> Result<Self, DriverError> {
        let pipeline1 = ComputePipeline::create(
            device,
            ComputePipelineInfo::default(),
            Shader::new_compute(include_glsl!("res/shader/compute/present1.comp").as_slice()),
        )?;
        let pipeline2 = ComputePipeline::create(
            device,
            ComputePipelineInfo::default(),
            Shader::new_compute(include_glsl!("res/shader/compute/present2.comp").as_slice()),
        )?;

        Ok(Self([pipeline1, pipeline2]))
    }

    /// TODO
    pub fn present_image(
        &self,
        graph: &mut Graph,
        image: impl Into<AnyImageNode>,
        swapchain: SwapchainImageNode,
    ) {
        let image = image.into();
        // let image_info = graph.node_info(image);
        let swapchain_info = graph.resource(swapchain).info;

        // TODO: Notice non-sRGB images and run a different pipeline

        graph
            .begin_cmd()
            .debug_name("present (from compute)")
            .bind_pipeline(&self.0[0])
            .shader_resource_access(0, image, AccessType::ComputeShaderReadOther)
            .shader_resource_access(1, swapchain, AccessType::ComputeShaderWrite)
            .record_cmd_buf(move |cmd_buf, _| {
                cmd_buf.dispatch(swapchain_info.width, swapchain_info.height, 1);
            });
    }

    /// TODO
    pub fn present_images(
        &self,
        graph: &mut Graph,
        top_image: impl Into<AnyImageNode>,
        bottom_image: impl Into<AnyImageNode>,
        swapchain: SwapchainImageNode,
    ) {
        let top_image = top_image.into();
        let bottom_image = bottom_image.into();
        // let top_image_info = graph.node_info(top_image);
        // let bottom_image_info = graph.node_info(bottom_image);
        let swapchain_info = graph.resource(swapchain).info;

        // TODO: Notice non-sRGB images and run a different pipeline

        graph
            .begin_cmd()
            .debug_name("present (from compute)")
            .bind_pipeline(&self.0[1])
            .shader_resource_access((0, [0]), top_image, AccessType::ComputeShaderReadOther)
            .shader_resource_access((0, [1]), bottom_image, AccessType::ComputeShaderReadOther)
            .shader_resource_access(1, swapchain, AccessType::ComputeShaderWrite)
            .record_cmd_buf(move |cmd_buf, _| {
                cmd_buf.dispatch(swapchain_info.width, swapchain_info.height, 1);
            });
    }
}

/// TODO
pub struct GraphicPresenter {
    pipeline: GraphicPipeline,
}

impl GraphicPresenter {
    /// TODO
    pub fn new(device: &Device) -> Result<Self, DriverError> {
        let pipeline = GraphicPipeline::create(
            device,
            GraphicPipelineInfo::default(),
            [
                Shader::new_vertex(include_glsl!("res/shader/graphic/present.vert").as_slice()),
                Shader::new_fragment(include_glsl!("res/shader/graphic/present.frag").as_slice()),
            ],
        )?;

        Ok(Self { pipeline })
    }

    /// TODO
    pub fn present_image(
        &self,
        graph: &mut Graph,
        image: impl Into<AnyImageNode>,
        swapchain: SwapchainImageNode,
    ) {
        let image = image.into();
        let image_info = graph.resource(image).info;
        let swapchain_info = graph.resource(swapchain).info;

        let (image_width, image_height) = (image_info.width as f32, image_info.height as f32);
        let (swapchain_width, swapchain_height) =
            (swapchain_info.width as f32, swapchain_info.height as f32);

        let scale = (swapchain_width / image_width).max(swapchain_height / image_height);
        let transform = Mat4::from_scale(vec3(
            scale * image_width / swapchain_width,
            scale * image_height / swapchain_height,
            1.0,
        ));

        graph
            .begin_cmd()
            .debug_name("present (from graphic)")
            .bind_pipeline(&self.pipeline)
            .shader_resource_access(
                0,
                image,
                AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
            )
            .store_color(0, swapchain)
            .record_cmd_buf(move |cmd_buf, _| {
                // Draw a quad with implicit vertices (no buffer)
                cmd_buf
                    .push_constants(0, cast_slice(&transform.to_cols_array()))
                    .draw(6, 1, 0, 0);
            });
    }
}
