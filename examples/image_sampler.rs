mod profile_with_puffin;

use {
    clap::Parser,
    hassle_rs::compile_hlsl,
    inline_spirv::inline_spirv,
    std::{
        path::{Path, PathBuf},
        sync::Arc,
    },
    vk_graph::prelude::*,
    vk_graph_window::WindowBuilder,
};

/// Displays a sequence of image samplers.
///
/// Note that manually specifying image samplers is completely optional, valid defaults will be used
/// if they are not specified when creating the shader which uses them. Additionally, you could
/// instead use use name suffixes such as _llr or _nne for linear/linear repeat or nearest/nearest
/// clamp-to-edge.
///
/// You may run this example program with either --hlsl or --separate arguments as follows:
///
/// cargo run --example image_sampler -- --hlsl --separate
///
/// Run with --help for more information.
///
/// See min_max.rs for more advanced image sampler usage.
fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    let args = Args::parse();
    let window = WindowBuilder::default().debug(args.debug).build()?;
    let gulf_image = read_image(&window.device, "examples/res/image/gulf.jpg")?;

    // Sampler info contains the full definition of Vulkan sampler settings using a builder struct
    let edge_edge = SamplerInfoBuilder::default()
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    let border_edge_black = SamplerInfoBuilder::default()
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK);
    let edge_border_white = SamplerInfoBuilder::default()
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE);

    // Image samplers are part of the shader pipeline and so we will create three pipelines total
    let pipelines = [edge_edge, border_edge_black, edge_border_white]
        .into_iter()
        .map(|sampler_info| create_pipeline(&window.device, sampler_info))
        .collect::<Result<Box<_>, _>>()?;
    let mut pipeline_index = 0;
    let mut pipeline_time = 0.0;

    window.run(|frame| {
        // Periodically change the active pipeline index
        pipeline_time += 0.016;
        if pipeline_time > 2.0 {
            pipeline_time = 0.0;
            pipeline_index += 1;
            pipeline_index %= pipelines.len();
        }

        // Draw gulf.jpg using the active pipeline
        let gulf_image = frame.render_graph.bind_node(&gulf_image);
        frame
            .render_graph
            .begin_pass("Draw gulf image to swapchain")
            .bind_pipeline(&pipelines[pipeline_index])
            .read_descriptor(0, gulf_image)
            .store_color(0, frame.swapchain_image)
            .record_subpass(|subpass, _| {
                subpass.draw(3, 1, 0, 0);
            });
    })?;

    Ok(())
}

fn create_pipeline(
    device: &Arc<Device>,
    sampler_info: impl Into<SamplerInfo>,
) -> anyhow::Result<Arc<GraphicPipeline>> {
    let args = Args::parse();

    let mut frag_shader = match (args.hlsl, args.separate) {
        (true, true) => {
            // HLSL separate image sampler
            Shader::new_fragment(
                inline_spirv!(
                    r#"
                struct FullscreenVertexOutput
                {
                    float4 position : SV_Position;
                    [[vk::location(0)]] float2 uv : TEXCOORD0;
                };

                [[vk::binding(0, 0)]] Texture2D screenTexture : register(t0);
                [[vk::binding(1, 0)]] SamplerState textureSampler : register(s0);

                float4 main(FullscreenVertexOutput input)
                    : SV_Target
                {
                    return screenTexture.Sample(textureSampler, input.uv);
                }
                "#,
                    frag,
                    hlsl
                )
                .as_slice(),
            )
        }
        (true, false) => {
            // HLSL combined image sampler: inline_spirv uses shaderc which does not support this, so
            // we are using hassle_rs which uses dxc. You must follow the instructions listed here to
            // use hassle_rs:
            // See: https://github.com/Traverse-Research/hassle-rs
            // See: https://github.com/microsoft/DirectXShaderCompiler/wiki/Vulkan-combined-image-sampler-type
            // See: https://github.com/google/shaderc/issues/1310
            Shader::new_fragment(
            compile_hlsl(
                "fragment.hlsl",
                r#"
                struct FullscreenVertexOutput
                {
                    float4 position : SV_Position;
                    [[vk::location(0)]] float2 uv : TEXCOORD0;
                };

                [[vk::combinedImageSampler]][[vk::binding(0, 0)]]  Texture2D<float4> screenTexture : register(t0);
                [[vk::combinedImageSampler]][[vk::binding(0, 0)]]  SamplerState textureSampler : register(s0);

                float4 main(FullscreenVertexOutput input)
                    : SV_Target
                {
                    return screenTexture.Sample(textureSampler, input.uv);
                }
                "#,
                "main", "ps_5_0", &["-spirv"], &[],
            )?
            .as_slice(),
        )
        }
        (false, true) => {
            // GLSL separate image sampler
            Shader::new_fragment(
                inline_spirv!(
                    r#"
                #version 460 core

                layout(binding = 0) uniform texture2D image;
                layout(binding = 1) uniform sampler image_sampler;
                layout(location = 0) in vec2 vk_TexCoord;
                layout(location = 0) out vec4 vk_Color;

                void main() {
                    vk_Color = texture(sampler2D(image, image_sampler), vk_TexCoord);
                }
                "#,
                    frag
                )
                .as_slice(),
            )
        }
        (false, false) => {
            // GLSL combined image sampler
            Shader::new_fragment(
                inline_spirv!(
                    r#"
                #version 460 core

                    layout(binding = 0) uniform sampler2D image;
                    layout(location = 0) in vec2 vk_TexCoord;
                    layout(location = 0) out vec4 vk_Color;

                    void main() {
                        vk_Color = texture(image, vk_TexCoord);
                    }
                "#,
                    frag
                )
                .as_slice(),
            )
        }
    };

    // Use the builder pattern to specify an image sampler at the combined binding index (0) or
    // separate binding index (1).
    let sampler_binding = args.separate as u32;
    frag_shader = frag_shader.image_sampler(sampler_binding, sampler_info);

    Ok(Arc::new(GraphicPipeline::create(
        device,
        GraphicPipelineInfo::default(),
        [
            Shader::new_vertex(
                inline_spirv!(
                    r#"
                    #version 460 core

                    const vec2[3] VERTICES = {
                        vec2(-1, -1),
                        vec2(-1,  3),
                        vec2( 3, -1),
                    };

                    layout(location = 0) out vec2 vk_TexCoord;

                    void main() {
                        gl_Position = vec4(VERTICES[gl_VertexIndex], 0, 1);
                        vk_TexCoord = 0.75 * gl_Position.xy + vec2(0.5);
                    }
                    "#,
                    vert
                )
                .as_slice(),
            ),
            frag_shader,
        ],
    )?))
}

fn read_image(device: &Arc<Device>, path: impl AsRef<Path>) -> anyhow::Result<Arc<Image>> {
    // For another way to loading images, see vk_graph_fx::ImageLoader
    let gulf_jpg = image::open(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path))?;
    let image = Arc::new(Image::create(
        device,
        ImageInfo::image_2d(
            gulf_jpg.width(),
            gulf_jpg.height(),
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        ),
    )?);

    {
        let mut render_graph = RenderGraph::new();
        let image = render_graph.bind_node(&image);
        let image_buf = render_graph.bind_node(Buffer::create_from_slice(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            gulf_jpg.into_rgba8().into_vec(),
        )?);
        render_graph.copy_buffer_to_image(image_buf, image);
        render_graph
            .resolve()
            .submit(&mut HashPool::new(device), 0, 0)?;

        // Note: There is no need to call wait_until_executed() here
    }

    Ok(image)
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,

    /// Use HLSL fragment shaders instead of the default (GLSL)
    #[arg(long)]
    hlsl: bool,

    /// Use separate image sampler objects instead of the default (combined image sampler objects)
    #[arg(long)]
    separate: bool,
}
