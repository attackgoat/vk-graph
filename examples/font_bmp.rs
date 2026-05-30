mod profile_with_puffin;

use {
    ash::vk,
    bmfont::{BMFont, OrdinateOrientation},
    clap::Parser,
    image::ImageReader,
    std::{io::Cursor, time::Instant},
    vk_graph::{
        driver::{
            compute::{ComputePipeline, ComputePipelineInfo},
            image::ImageInfo,
            physical_device::Vulkan11Properties,
            shader::{Shader, SpecializationMap},
        },
        pool::{Pool as _, hash::HashPool},
    },
    vk_graph_fx::*,
    vk_graph_window::Window,
    vk_shader_macros::glsl,
    vk_sync::AccessType,
};

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    // Standard vk-graph stuff
    let args = Args::parse();
    let window = Window::builder().debug(args.debug).build()?;
    let display = GraphicPresenter::new(&window.device)?;
    let mut image_loader = ImageLoader::new(&window.device)?;
    let mut pool = HashPool::new(&window.device);

    // Load a bitmapped font
    let small_10px_font = BMFont::new(
        Cursor::new(include_bytes!("res/font/small/small_10px.fnt")),
        OrdinateOrientation::TopToBottom,
    )?;
    let mut small_10px_font = image_loader.load_bitmap_font(
        0,
        0,
        small_10px_font,
        [(
            ImageReader::new(Cursor::new(
                include_bytes!("res/font/small/small_10px_0.png").as_slice(),
            ))
            .with_guessed_format()?
            .decode()?
            .into_rgb8()
            .to_vec()
            .as_slice(),
            64,
            64,
        )],
    )?;

    // A neato smoke effect just for fun
    let Vulkan11Properties { subgroup_size, .. } = window.device.physical_device.properties_v1_1;
    let start_time = Instant::now();
    let smoke_pipeline = ComputePipeline::create(&window.device,
        ComputePipelineInfo::default(),
        Shader::new_compute(
        glsl!(
            r#"
            // Derived from https://www.shadertoy.com/view/Xl2XWz
            #version 460 core
            #pragma shader_stage(compute)

            layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

            layout(push_constant) uniform PushConstants {
                layout(offset = 0) float time;
            } push_const;

            layout(set = 0, binding = 0, rgba32f) restrict writeonly uniform image2D image;

            float smoothNoise(vec2 p) {
                vec2 i = floor(p);
                p -= i;
                p *= p * (3 - p - p);

                return dot(
                    mat2(fract(sin(vec4(0, 1, 27, 28) + i.x + i.y * 27) * 1e5)) * vec2(1 - p.y, p.y),
                    vec2(1 - p.x, p.x)
                );
            }

            float fractalNoise(vec2 p) {
                return smoothNoise(p) * 0.57 + smoothNoise(p * 2.45) * 0.28 + smoothNoise(p * 6) * 0.15;
            }

            float warpedNoise(vec2 p) {
                vec2 m = vec2(sin(push_const.time * 0.5), cos(push_const.time * 0.5));
                float x = fractalNoise(p + m);
                float y = fractalNoise(p + m.yx + x);
                float z = fractalNoise(p - m - x + y);
                vec3 w = vec3(x, y, z);

                return fractalNoise(p + w.xy + w.yz + w.zx + length(w) * 0.25);
            }

            void main() {
                vec2 uv = vec2(gl_GlobalInvocationID.xy) / imageSize(image).y;
                float n1 = warpedNoise(uv * 5);
                float n2 = warpedNoise(uv * 5 + 0.04);
                float bump1 = max(n2 - n1, 0.0) / 0.02 * 0.7071;
                float bump2 = max(n1 - n2, 0.1) / 0.04 * 0.7071;
                bump1 = bump1 * bump1 * 0.5 + pow(bump1, 4) * 0.5;
                bump2 = bump2 * bump2 * 0.5 + pow(bump2, 4) * 0.5;
                vec3 col = vec3(n1 * n1 * 0.7, n1, n1 * n1 * 0.4)
                        * n1 * n1
                        * (vec3(0.25, 0.5, 1)
                        * bump1 * 0.2
                        + vec3(1) * bump2 * 0.2 + 0.75);
                vec4 fragColor = vec4(sqrt(max(col, 0.)), 1);

                imageStore(image, ivec2(gl_GlobalInvocationID.xy), fragColor);
            }
            "#,
        )
        .as_slice()).specialization(SpecializationMap::new(subgroup_size.to_ne_bytes()).constant(0,0,4)),
    )?;

    window.run(|frame| {
        let image_node = frame.graph.bind_resource(
            pool.lease_resource(ImageInfo::image_2d(
                320,
                200,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_DST,
            ))
            .unwrap(),
        );

        // Fill the image with a smoke effect
        let elapsed_time = Instant::now() - start_time;
        frame
            .graph
            .begin_cmd()
            .debug_name("smoke")
            .bind_pipeline(&smoke_pipeline)
            .shader_resource_access(0, image_node, AccessType::ComputeShaderWrite)
            .record_cmd(move |cmd| {
                cmd.push_constants(0, &elapsed_time.as_secs_f32().to_ne_bytes())
                    .dispatch(frame.width.div_ceil(subgroup_size), frame.height, 1);
            });

        // Print some text onto the image
        let text = "vk-graph";
        let (_offset, [width, height]) = small_10px_font.measure(text);
        let scale = 4.0;
        let x = 320f32 * 0.5 / scale - width as f32 * 0.5;
        let y = 200f32 * 0.5 / scale - height as f32 * 0.5;
        let color = [196, 172, 230u8];
        small_10px_font.print_scale(frame.graph, image_node, x, y, color, text, scale);

        display.present_image(frame.graph, image_node, frame.swapchain_image);
    })?;

    Ok(())
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
