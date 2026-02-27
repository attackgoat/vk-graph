mod profile_with_puffin;

use {
    bytemuck::cast_slice,
    clap::Parser,
    std::sync::Arc,
    vk_graph::cmd_ref::LoadOp,
    vk_graph_prelude::*,
    vk_graph_window::{WindowBuilder, WindowError},
    vk_shader_macros::glsl,
};

// A Vulkan triangle using a graphic pipeline, vertex/fragment shaders, and index/vertex buffers.
fn main() -> Result<(), WindowError> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    let args = Args::parse();
    let window = WindowBuilder::default().debug(args.debug).build()?;
    let triangle_pipeline = GraphicPipeline::create(
        &window.device,
        GraphicPipelineInfo::default(),
        [
            Shader::new_vertex(
                glsl!(
                    r#"
                    #version 460 core
                    #pragma shader_stage(vertex)

                    layout(location = 0) in vec3 position;
                    layout(location = 1) in vec3 color;

                    layout(location = 0) out vec3 vk_Color;

                    void main() {
                        gl_Position = vec4(position, 1);
                        vk_Color = color;
                    }
                    "#
                )
                .as_slice(),
            ),
            Shader::new_fragment(
                glsl!(
                    r#"
                    #version 460 core
                    #pragma shader_stage(fragment)

                    layout(location = 0) in vec3 color;

                    layout(location = 0) out vec4 vk_Color;
                    
                    void main() {
                        vk_Color = vec4(color, 1);
                    }
                    "#
                )
                .as_slice(),
            ),
        ],
    )?;

    let index_buf = Arc::new(Buffer::create_from_slice(
        &window.device,
        vk::BufferUsageFlags::INDEX_BUFFER,
        cast_slice(&[0u16, 1, 2]),
    )?);

    let vertex_buf = Arc::new(Buffer::create_from_slice(
        &window.device,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        cast_slice(&[
            1.0f32, 1.0, 0.0, // v1
            1.0, 0.0, 0.0, // red
            0.0, -1.0, 0.0, // v2
            0.0, 1.0, 0.0, // green
            -1.0, 1.0, 0.0, // v3
            0.0, 0.0, 1.0, // blue
        ]),
    )?);

    window.run(|frame| {
        let index_node = frame.graph.bind_resource(&index_buf);
        let vertex_node = frame.graph.bind_resource(&vertex_buf);

        frame
            .graph
            .begin_cmd()
            .debug_name("Triangle Example")
            .bind_pipeline(&triangle_pipeline)
            .resource_access(index_node, AccessType::IndexBuffer)
            .resource_access(vertex_node, AccessType::VertexBuffer)
            .color_attachment_image(
                0,
                frame.swapchain_image,
                LoadOp::CLEAR_BLACK_ALPHA_ZERO,
                StoreOp::Store,
            )
            .record_cmd_buf(move |cmd_buf| {
                cmd_buf
                    .bind_index_buffer(index_node, 0, vk::IndexType::UINT16)
                    .bind_vertex_buffer(0, vertex_node, 0)
                    .draw_indexed(3, 1, 0, 0, 0);
            });
    })
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
