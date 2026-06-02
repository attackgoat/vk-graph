pub mod data {
    use super::PakBuf;

    // The ".pak" file is a data transport type with compression and other useful features
    // It is used to hold the images used by this example, because they *could* be really
    // big - anyways we generated some bindings to make accessing those less error-prone:
    include!(concat!(env!("OUT_DIR"), "/pak_bindings.rs"));

    // This happens if you want the .pak bytes inside the executable itself
    #[cfg(feature = "include-pak")]
    pub fn open() -> anyhow::Result<PakBuf> {
        Ok(include_bytes!(concat!(env!("OUT_DIR"), "/data.pak"))
            .as_slice()
            .into())
    }

    // This happens if you want the .pak as a file next to the executable
    #[cfg(not(feature = "include-pak"))]
    pub fn open() -> anyhow::Result<PakBuf> {
        use std::env::current_exe;

        let mut pak_path = current_exe()?;
        pak_path.set_file_name("data.pak");

        Ok(PakBuf::open(pak_path)?)
    }
}

mod res {
    pub mod shader {
        include!(concat!(env!("OUT_DIR"), "/shader_bindings.rs"));
    }
}

use {
    anyhow::Context,
    bytemuck::{Pod, Zeroable, bytes_of},
    clap::Parser,
    pak::{Pak, PakBuf},
    std::time::Instant,
    vk_graph::{
        Graph,
        cmd::{LoadOp, StoreOp},
        driver::{
            ash::vk,
            graphic::{GraphicsPipeline, GraphicsPipelineInfo},
            image::ImageInfo,
            shader::Shader,
            sync::AccessType,
        },
        pool::{Pool as _, lazy::LazyPool},
    },
    vk_graph_fx::*,
    vk_graph_window::Window,
    winit::dpi::PhysicalSize,
};

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();

    let args = Args::parse();
    let window = Window::builder()
        .debug(args.debug)
        .min_image_count(3)
        .window(|builder| builder.with_inner_size(PhysicalSize::new(1280.0f64, 720.0f64)))
        .build()?;
    let display = GraphicPresenter::new(&window.device).context("Presenter")?;
    let mut cache = LazyPool::new(&window.device);
    let mut image_loader = ImageLoader::new(&window.device).context("Loader")?;

    // Load source images: PakBuf -> BitmapBuf -> ImageBinding (here) -> ImageNode (during loop)
    let mut data = data::open().context("Pak")?;
    let mut flowers_image_binding = Some({
        let data = data
            .read_bitmap(data::IMAGE_FLOWERS_JPG)
            .context("Unable to read flowers bitmap")?;
        image_loader
            .decode_linear(
                0,
                0,
                data.pixels(),
                ImageFormat::R8G8B8,
                data.width(),
                data.height(),
            )
            .context("Unable to decode flowers bitmap")?
    });
    let mut noise_image_binding = Some({
        let data = data
            .read_bitmap(data::IMAGE_RGBA_NOISE_PNG)
            .context("Unable to read noise bitmap")?;
        image_loader
            .decode_linear(
                0,
                0,
                data.pixels(),
                ImageFormat::R8G8B8A8,
                data.width(),
                data.height(),
            )
            .context("Unable to decode noise bitmap")?
    });

    // The shader toy example used two graphics pipelines with defaults:
    // no depth/stencil
    // 1x sample count
    // one-sided
    let buffer_pipeline = GraphicsPipeline::create(
        &window.device,
        GraphicsPipelineInfo::default(),
        [
            Shader::new_vertex(res::shader::QUAD_VERT),
            Shader::new_fragment(res::shader::FLOCKAROO_BUF_FRAG),
        ],
    )
    .context("FLOCKAROO_BUF_FRAG")?;
    let image_pipeline = GraphicsPipeline::create(
        &window.device,
        GraphicsPipelineInfo::default(),
        [
            Shader::new_vertex(res::shader::QUAD_VERT),
            Shader::new_fragment(res::shader::FLOCKAROO_IMG_FRAG),
        ],
    )
    .context("FLOCKAROO_IMG_FRAG")?;

    let mut graph = Graph::default();
    let blank_image = graph.bind_resource(
        cache
            .resource(ImageInfo::image_2d(
                8,
                8,
                vk::Format::R8G8B8A8_SRGB,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            ))
            .context("Blank image")?,
    );

    let (width, height) = (1280, 720);
    let framebuffer_image = graph.bind_resource(
        cache
            .resource(ImageInfo::image_2d(
                width,
                height,
                vk::Format::R8G8B8A8_SRGB,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            ))
            .context("Framebuffer image")?,
    );
    let temp_image = graph.bind_resource(
        cache
            .resource(ImageInfo::image_2d(
                width,
                height,
                vk::Format::R8G8B8A8_SRGB,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            ))
            .context("Temp image")?,
    );

    graph
        .clear_color_image(framebuffer_image, [1.0, 1.0, 0.0, 1.0])
        .clear_color_image(blank_image, [0.0, 0.0, 0.0, 1.0])
        .clear_color_image(temp_image, [0.0, 1.0, 0.0, 1.0]);

    let mut framebuffer_image_binding = Some(graph.resource(framebuffer_image).clone());
    let mut blank_image_binding = Some(graph.resource(blank_image).clone());
    let mut temp_image_binding = Some(graph.resource(temp_image).clone());

    graph.finalize().queue_submit(&mut cache, 0, 0)?;

    let started_at = Instant::now();
    let mut prev_frame_at = started_at;
    let mut count = 0i32;
    let framebuffer_info = framebuffer_image_binding.as_ref().unwrap().info;
    let flowers_image_info = flowers_image_binding.as_ref().unwrap().info;
    let noise_image_info = noise_image_binding.as_ref().unwrap().info;
    let blank_image_info = blank_image_binding.as_ref().unwrap().info;

    window
        .run(|frame| {
            let now = Instant::now();

            // Update the stuff any shader toy shader would want to know each frame
            let dt = now - prev_frame_at;
            prev_frame_at = now;

            let elapsed = now - started_at;

            count += 1;

            // Bind things to this graph (the graph will own our things until we unbind them)
            let flowers_image = frame
                .graph
                .bind_resource(flowers_image_binding.take().unwrap());
            let noise_image = frame
                .graph
                .bind_resource(noise_image_binding.take().unwrap());
            let framebuffer_image = frame
                .graph
                .bind_resource(framebuffer_image_binding.take().unwrap());
            let blank_image = frame
                .graph
                .bind_resource(blank_image_binding.take().unwrap());
            let temp_image = frame
                .graph
                .bind_resource(temp_image_binding.take().unwrap());

            // We need to push a shader-toy defined set of constants to each pipeline - any copy
            // type will do but we are getting fancy here by defining a struct to be super precise
            // about what we're doing - but you may want to just send a bunch of f32's
            #[repr(C)]
            #[derive(Clone, Copy, Pod, Zeroable)]
            struct PushConstants {
                resolution: [f32; 3],
                _pad_1: u32,
                date: [f32; 4],
                mouse: [f32; 4],
                time: f32,
                time_delta: f32,
                frame: i32,
                sample_rate: f32,
                channel_time: [f32; 4],
                channel_resolution: [f32; 16],
            }

            // Each pipeline gets the same constant data
            let push_consts = PushConstants {
                resolution: [frame.width as f32, frame.height as _, 1.0],
                _pad_1: Default::default(),
                date: [1970.0, 1.0, 1.0, elapsed.as_secs_f32()],
                mouse: [0.0, 0.0, 0.0, 0.0],
                time: elapsed.as_secs_f32(),
                time_delta: dt.as_secs_f32(),
                frame: count,
                sample_rate: 44100.0,
                channel_time: [
                    elapsed.as_secs_f32(),
                    elapsed.as_secs_f32(),
                    elapsed.as_secs_f32(),
                    elapsed.as_secs_f32(),
                ],
                channel_resolution: [
                    framebuffer_info.width as f32,
                    framebuffer_info.height as _,
                    framebuffer_info.depth as _,
                    1.0,
                    noise_image_info.width as _,
                    noise_image_info.height as _,
                    noise_image_info.depth as _,
                    1.0,
                    flowers_image_info.width as _,
                    flowers_image_info.height as _,
                    flowers_image_info.depth as _,
                    1.0,
                    blank_image_info.width as _,
                    blank_image_info.height as _,
                    blank_image_info.depth as _,
                    1.0,
                ],
            };

            let (input, output) = if count % 2 == 0 {
                (framebuffer_image, temp_image)
            } else {
                (temp_image, framebuffer_image)
            };

            // Fill a buffer using a single-pass CFD pipeline where previous output feeds next input
            frame
                .graph
                .begin_cmd()
                .debug_name("Buffer A")
                .bind_pipeline(&buffer_pipeline)
                .shader_resource_access(
                    0,
                    input,
                    AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
                )
                .shader_resource_access(
                    1,
                    noise_image,
                    AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
                )
                .shader_resource_access(
                    2,
                    flowers_image,
                    AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
                )
                .shader_resource_access(
                    3,
                    blank_image,
                    AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
                )
                .color_attachment_image(0, output, LoadOp::DontCare, StoreOp::Store)
                .record_cmd(move |cmd| {
                    cmd.push_constants(0, bytes_of(&push_consts))
                        .draw(6, 1, 0, 0);
                });

            // Make the CFD look more like paint with a second pass
            frame
                .graph
                .begin_cmd()
                .debug_name("Image")
                .bind_pipeline(&image_pipeline)
                .shader_resource_access(
                    0,
                    output,
                    AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
                )
                .color_attachment_image(0, input, LoadOp::DontCare, StoreOp::Store)
                .record_cmd(move |cmd| {
                    cmd.push_constants(0, bytes_of(&push_consts))
                        .draw(6, 1, 0, 0);
                });

            // Done!
            display.present_image(frame.graph, input, frame.swapchain_image);

            // Unbind things from this graph (we want them back for the next frame!)
            flowers_image_binding = Some(frame.graph.resource(flowers_image).clone());
            noise_image_binding = Some(frame.graph.resource(noise_image).clone());
            framebuffer_image_binding = Some(frame.graph.resource(framebuffer_image).clone());
            blank_image_binding = Some(frame.graph.resource(blank_image).clone());
            temp_image_binding = Some(frame.graph.resource(temp_image).clone());
        })
        .context("Unable to run event loop")?;

    Ok(())
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
