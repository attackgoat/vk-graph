use {
    bytemuck::cast_slice,
    glam::{vec3, Mat4},
    std::sync::Arc,
    vk_graph::prelude::*,
    vk_shader_macros::include_glsl,
};

pub struct ComputePresenter([Arc<ComputePipeline>; 2]);

impl ComputePresenter {
    pub fn new(device: &Arc<Device>) -> Result<Self, DriverError> {
        let pipeline1 = Arc::new(ComputePipeline::create(
            device,
            ComputePipelineInfo::default(),
            Shader::new_compute(include_glsl!("res/shader/compute/present1.comp").as_slice()),
        )?);
        let pipeline2 = Arc::new(ComputePipeline::create(
            device,
            ComputePipelineInfo::default(),
            Shader::new_compute(include_glsl!("res/shader/compute/present2.comp").as_slice()),
        )?);

        Ok(Self([pipeline1, pipeline2]))
    }

    pub fn present_image(
        &self,
        graph: &mut RenderGraph,
        image: impl Into<AnyImageNode>,
        swapchain: SwapchainImageNode,
    ) {
        let image = image.into();
        // let image_info = graph.node_info(image);
        let swapchain_info = graph.node_info(swapchain);

        // TODO: Notice non-sRGB images and run a different pipeline

        graph
            .begin_pass("present (from compute)")
            .bind_pipeline(&self.0[0])
            .read_descriptor(0, image)
            .write_descriptor(1, swapchain)
            .record_compute(move |compute, _| {
                compute.dispatch(swapchain_info.width, swapchain_info.height, 1);
            });
    }

    pub fn present_images(
        &self,
        graph: &mut RenderGraph,
        top_image: impl Into<AnyImageNode>,
        bottom_image: impl Into<AnyImageNode>,
        swapchain: SwapchainImageNode,
    ) {
        let top_image = top_image.into();
        let bottom_image = bottom_image.into();
        // let top_image_info = graph.node_info(top_image);
        // let bottom_image_info = graph.node_info(bottom_image);
        let swapchain_info = graph.node_info(swapchain);

        // TODO: Notice non-sRGB images and run a different pipeline

        graph
            .begin_pass("present (from compute)")
            .bind_pipeline(&self.0[1])
            .read_descriptor((0, [0]), top_image)
            .read_descriptor((0, [1]), bottom_image)
            .write_descriptor(1, swapchain)
            .record_compute(move |compute, _| {
                compute.dispatch(swapchain_info.width, swapchain_info.height, 1);
            });
    }
}

pub struct GraphicPresenter {
    pipeline: Arc<GraphicPipeline>,
}

impl GraphicPresenter {
    pub fn new(device: &Arc<Device>) -> Result<Self, DriverError> {
        Ok(Self {
            pipeline: Arc::new(GraphicPipeline::create(
                device,
                GraphicPipelineInfo::default(),
                [
                    Shader::new_vertex(include_glsl!("res/shader/graphic/present.vert").as_slice()),
                    Shader::new_fragment(
                        include_glsl!("res/shader/graphic/present.frag").as_slice(),
                    ),
                ],
            )?),
        })
    }

    pub fn present_image(
        &self,
        graph: &mut RenderGraph,
        image: impl Into<AnyImageNode>,
        swapchain: SwapchainImageNode,
    ) {
        let image = image.into();
        let image_info = graph.node_info(image);
        let swapchain_info = graph.node_info(swapchain);

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
            .begin_pass("present (from graphic)")
            .bind_pipeline(&self.pipeline)
            .read_descriptor(0, image)
            .store_color(0, swapchain)
            .record_subpass(move |subpass, _| {
                // Draw a quad with implicit vertices (no buffer)
                subpass.push_constants(cast_slice(&transform.to_cols_array()));
                subpass.draw(6, 1, 0, 0);
            });
    }
}
