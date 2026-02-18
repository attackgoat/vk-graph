mod profile_with_puffin;

use {
    bytemuck::{Pod, Zeroable, cast_slice},
    clap::Parser,
    std::sync::Arc,
    vk_graph_prelude::*,
    vk_graph_window::{WindowBuilder, WindowError},
    vk_shader_macros::glsl,
    winit::dpi::LogicalSize,
};

fn main() -> Result<(), WindowError> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    let args = Args::parse();
    let window = WindowBuilder::default()
        .debug(args.debug)
        .window(|window| window.with_inner_size(LogicalSize::new(512, 512)))
        .build()?;
    let images = create_images(&window.device)?;
    let pipeline = create_graphic_pipeline(&window.device)?;
    let draw_buf = create_indirect_buffer(&window.device)?;

    window.run(|frame| {
        let draw_buf_node = frame.render_graph.bind_node(&draw_buf);

        let mut pass = frame
            .render_graph
            .begin_cmd_buf()
            .with_name("Test")
            .bind_pipeline(&pipeline)
            .access_node(draw_buf_node, AccessType::IndirectBuffer);

        for (idx, image) in images.iter().enumerate() {
            let image_node = pass.bind_node(image);
            pass = pass.read_descriptor((0, [idx as u32]), image_node);
        }

        pass.clear_color(0, frame.swapchain_image)
            .store_color(0, frame.swapchain_image)
            .record_subpass(move |subpass, _| {
                subpass.draw_indirect(draw_buf_node, 0, 64, 16);
            });
    })
}

fn create_images(device: &Arc<Device>) -> Result<Vec<Arc<Image>>, DriverError> {
    let mut textures = Vec::with_capacity(64);

    let (b, a) = (0.0, 1.0);
    let mut graph = RenderGraph::default();
    for y in 0..8 {
        for x in 0..8 {
            let texture = Arc::new(Image::create(
                device,
                ImageInfo::image_2d(
                    100,
                    100,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                ),
            )?);
            let texture_node = graph.bind_node(&texture);
            let r = y as f32 / 7.0;
            let g = x as f32 / 7.0;
            graph.clear_color_image(texture_node, [r, g, b, a]);
            textures.push(texture);
        }
    }

    let mut pool = LazyPool::new(device);

    graph.resolve().submit(&mut pool, 0, 0)?;

    Ok(textures)
}

fn create_indirect_buffer(device: &Arc<Device>) -> Result<Arc<Buffer>, DriverError> {
    let mut draw_cmds = Vec::with_capacity(64);
    for first_instance in 0..64 {
        draw_cmds.push(DrawIndirectCommand {
            vertex_count: 6,
            instance_count: 1,
            first_vertex: 0,
            first_instance,
        });
    }
    let draw_buf = Arc::new(Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::INDIRECT_BUFFER,
        cast_slice(&draw_cmds),
    )?);
    Ok(draw_buf)
}

fn create_graphic_pipeline(device: &Arc<Device>) -> Result<Arc<GraphicPipeline>, DriverError> {
    Ok(Arc::new(GraphicPipeline::create(
        device,
        GraphicPipelineInfo::default(),
        [
            Shader::new_vertex(
                glsl!(
                    r#"
                    #version 460 core
                    #pragma shader_stage(vertex)

                    const vec2 QUAD[] = {
                        vec2(0, 0),
                        vec2(0, 1),
                        vec2(1, 1),
                        vec2(0, 0),
                        vec2(1, 1),
                        vec2(1, 0),
                    };

                    layout(location = 0) out uint instance_index_out;

                    void main() {
                        uint x = gl_InstanceIndex % 8;
                        uint y = gl_InstanceIndex / 8;

                        vec2 scale = vec2(1.0 / 8.0);
                        vec2 offset = vec2((float(x) - 4.0) * scale.x, (float(y) - 4.0) * scale.y);

                        gl_Position = vec4(QUAD[gl_VertexIndex] * scale + offset, 0, 1);
                        instance_index_out = gl_InstanceIndex;
                    }
                    "#
                )
                .as_slice(),
            ),
            Shader::new_fragment(
                glsl!(
                    r#"
                    #version 460 core
                    #extension GL_EXT_nonuniform_qualifier : require
                    #pragma shader_stage(fragment)

                    layout(set = 0, binding = 0) uniform sampler2D sampler_nnr[];

                    layout(location = 0) in flat uint instance_index;

                    layout(location = 0) out vec4 color_out;

                    void main() {
                        color_out = texture(sampler_nnr[nonuniformEXT(instance_index)], vec2(0.5, 0.5));
                    }
                    "#
                )
                .as_slice(),
            ),
        ],
    )?))
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DrawIndirectCommand {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}
