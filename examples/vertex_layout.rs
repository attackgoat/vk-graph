mod profile_with_puffin;

use {
    ash::vk,
    bytemuck::{Pod, Zeroable, cast_slice},
    clap::Parser,
    half::f16,
    std::{mem::size_of, sync::Arc},
    vk_graph::{
        cmd::{ClearColorValue, LoadOp, StoreOp},
        driver::{
            DriverError,
            buffer::Buffer,
            device::Device,
            graphics::{GraphicsPipeline, GraphicsPipelineInfo},
            shader::{Shader, ShaderBuilder},
        },
    },
    vk_graph_window::{FrameContext, Window},
    vk_shader_macros::glsl,
    vk_sync::AccessType,
};

/// This example draws two triangles using two different vertex formats.
///
/// All hardware should support 32 bit position values, so those are used without checking.
///
/// Most hardware will support 64 bit values, so we first check for support and if that fails
/// we fall back to 16 bit values.
fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    // NOTE: This example uses the 64-bit rules defined in the Vulkan spec, they're not obvious:
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#fxvertex-attrib

    let args = Args::parse();
    let window = Window::builder().debug(args.debug).build()?;

    let f16_pipeline = create_f16_pipeline(&window.device).ok();
    let f16_vertex_buf = {
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Vertex([f16; 2], [f32; 3]);

        let vec2 = |x, y| [f16::from_f32(x), f16::from_f32(y)];

        Arc::new(Buffer::create_from_slice(
            &window.device,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            cast_slice(&[
                Vertex(vec2(-1.0, -1.0), [1.0, 0.0, 0.0]),
                Vertex(vec2(1.0, 1.0), [0.0, 0.0, 1.0]),
                Vertex(vec2(1.0, -1.0), [0.0, 1.0, 0.0]),
            ]),
        )?)
    };

    let f32_pipeline = create_f32_pipeline(&window.device)?;
    let f32_vertex_buf = {
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Vertex([f32; 2], [f32; 3]);

        Arc::new(Buffer::create_from_slice(
            &window.device,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            cast_slice(&[
                Vertex([-1f32, -1.0], [1.0, 0.0, 0.0]),
                Vertex([-1.0, 1.0], [0.0, 1.0, 0.0]),
                Vertex([1.0, 1.0], [0.0, 0.0, 1.0]),
            ]),
        )?)
    };

    let f64_pipeline = create_f64_pipeline(&window.device).ok();
    let f64_vertex_buf = {
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Vertex([f64; 2], [f32; 3], u32);

        Arc::new(Buffer::create_from_slice(
            &window.device,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            cast_slice(&[
                Vertex([-1.0, -1.0], [1.0, 0.0, 0.0], 0),
                Vertex([1.0, 1.0], [0.0, 0.0, 1.0], 0),
                Vertex([1.0, -1.0], [0.0, 1.0, 0.0], 0),
            ]),
        )?)
    };

    window.run(|mut frame| {
        draw_triangle(
            &mut frame,
            &f32_pipeline,
            &f32_vertex_buf,
            LoadOp::CLEAR_BLACK_ALPHA_ZERO,
        );

        if let Some(f64_pipeline) = &f64_pipeline {
            draw_triangle(&mut frame, f64_pipeline, &f64_vertex_buf, LoadOp::Load);
        } else if let Some(f16_pipeline) = &f16_pipeline {
            draw_triangle(&mut frame, f16_pipeline, &f16_vertex_buf, LoadOp::Load);
        }
    })?;

    Ok(())
}

fn draw_triangle(
    frame: &mut FrameContext,
    pipeline: &GraphicsPipeline,
    vertex_buf: &Arc<Buffer>,
    load: LoadOp<ClearColorValue>,
) {
    let vertex_buf = frame.graph.bind_resource(vertex_buf);

    frame
        .graph
        .begin_cmd()
        .debug_name("Triangle")
        .bind_pipeline(pipeline)
        .color_attachment_image(0, frame.swapchain_image, load, StoreOp::Store)
        .resource_access(vertex_buf, AccessType::VertexBuffer)
        .record_cmd(move |cmd| {
            cmd.bind_vertex_buffer(0, vertex_buf, 0).draw(3, 1, 0, 0);
        });
}

fn create_f16_pipeline(device: &Device) -> Result<GraphicsPipeline, DriverError> {
    if !supports_vertex_buffer(device, vk::Format::R16G16_SFLOAT) {
        return Err(DriverError::Unsupported);
    }

    const POSITION_SIZE: u32 = 2 * size_of::<f16>() as u32;
    const COLOR_SIZE: u32 = 3 * size_of::<f32>() as u32;

    let vertex = create_vertex_shader(false).vertex_input(
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: POSITION_SIZE + COLOR_SIZE,
            input_rate: vk::VertexInputRate::VERTEX,
        }],
        [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R16G16_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: POSITION_SIZE,
            },
        ],
    );

    create_pipeline(device, vertex)
}

fn create_f32_pipeline(device: &Device) -> Result<GraphicsPipeline, DriverError> {
    // Uses automatic vertex input layout
    let vertex = create_vertex_shader(false);

    create_pipeline(device, vertex)
}

fn create_f64_pipeline(device: &Device) -> Result<GraphicsPipeline, DriverError> {
    if !supports_vertex_buffer(device, vk::Format::R64G64_SFLOAT) {
        return Err(DriverError::Unsupported);
    }

    const POSITION_SIZE: u32 = 2 * size_of::<f64>() as u32;
    const COLOR_SIZE: u32 = 3 * size_of::<f32>() as u32;
    const PAD_SIZE: u32 = size_of::<u32>() as u32;

    let vertex = create_vertex_shader(true).vertex_input(
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: POSITION_SIZE + COLOR_SIZE + PAD_SIZE,
            input_rate: vk::VertexInputRate::VERTEX,
        }],
        [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R64G64_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: POSITION_SIZE,
            },
        ],
    );

    create_pipeline(device, vertex)
}

fn create_vertex_shader(is_double: bool) -> ShaderBuilder {
    // From the specs: Input attributes which have three- or four-component 64-bit formats will
    // consume two consecutive locations
    //
    // To support a vec3 64-bit case this means color_in needs to be on location 2

    /*
    This shader is compiled with a macro because we want to be able to switch the vec2 type to a
    dvec2 when using 64-bit positions; and for the purposes of this example we don't want to
    duplicate this shader code. You probably don't want to do this, or you may have different
    facilities for generating SPIR-V code - either way ignore the macro unless you're interested in
    the include_glsl! wizardry it contains which is unrelated to this example.
    */
    macro_rules! compile_vert {
        ($vec2_ty:literal) => {
            glsl!(
                define: VEC2_TY $vec2_ty,
                r#"
                #version 460 core
                #pragma shader_stage(vertex)

                layout(location = 0) in VEC2_TY position_in;
                layout(location = 1) in vec3 color_in;

                layout(location = 0) out vec3 color_out;

                void main() {
                    gl_Position = vec4(position_in, 0, 1);
                    color_out = color_in;
                }
                "#
            )
        };
    }

    let spirv = if is_double {
        compile_vert!("dvec2").as_slice()
    } else {
        compile_vert!("vec2").as_slice()
    };

    Shader::new_vertex(spirv)
}

fn create_pipeline(
    device: &Device,
    vertex: ShaderBuilder,
) -> Result<GraphicsPipeline, DriverError> {
    GraphicsPipeline::create(
        device,
        GraphicsPipelineInfo::default(),
        [
            vertex,
            Shader::from_spirv(
                glsl!(
                    r#"
                    #version 460 core
                    #pragma shader_stage(fragment)

                    layout(location = 0) in vec3 color_in;

                    layout(location = 0) out vec4 color_out;

                    void main() {
                        color_out = vec4(color_in, 1.0);
                    }
                    "#
                )
                .as_slice(),
            ),
        ],
    )
}

fn supports_vertex_buffer(device: &Device, format: vk::Format) -> bool {
    device
        .physical_device
        .format_properties(format)
        .buffer_features
        .contains(vk::FormatFeatureFlags::VERTEX_BUFFER)
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
