mod profile_with_puffin;

use {
    bytemuck::{Pod, Zeroable, bytes_of},
    clap::Parser,
    core::f32,
    glam::{Vec4, vec3},
    std::sync::Arc,
    vk_graph_prelude::*,
    vk_graph_window::{WindowBuilder, WindowError},
    vk_shader_macros::glsl,
};

// TODO: Add texelFetch option

fn main() -> Result<(), WindowError> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    let args = Args::parse();
    let window = WindowBuilder::default().debug(args.debug).build()?;

    let size = 237u32;
    let mip_level_count = size.ilog2();

    assert_ne!(mip_level_count, 0, "size must be greater than one");

    let image_info = ImageInfo::image_2d(
        size,
        size,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
    )
    .to_builder()
    .mip_level_count(mip_level_count)
    .build();
    let image = Arc::new(Image::create(&window.device, image_info)?);

    fill_mip_levels(&window.device, &image)?;

    let splat = splat(&window.device)?;

    window.run(|frame| {
        // It is 100% certain that the swapchain supports color attachment usage, so this is shown
        // for completeness only
        // https://vulkan.gpuinfo.org/listsurfaceusageflags.php
        assert!(
            frame
                .render_graph
                .node_info(frame.swapchain_image)
                .usage
                .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        );

        let image = frame.render_graph.bind_node(&image);
        let swapchain_info = frame.render_graph.node_info(frame.swapchain_image);
        let stripe_width = swapchain_info.width / mip_level_count;

        let mut pass = frame
            .render_graph
            .begin_cmd_buf()
            .with_name("splat mips")
            .bind_pipeline(&splat);

        for mip_level in 0..mip_level_count {
            let stripe_x = mip_level * stripe_width;
            pass = pass
                .read_descriptor_as(
                    0,
                    image,
                    image_info
                        .default_view_info()
                        .to_builder()
                        .base_mip_level(mip_level)
                        .mip_level_count(1),
                )
                .load_color(0, frame.swapchain_image)
                .store_color(0, frame.swapchain_image)
                .set_render_area(stripe_x as _, 0, stripe_width, swapchain_info.height)
                .record_subpass(|subpass, _| {
                    subpass.draw(6, 1, 0, 0);
                });
        }
    })
}

fn fill_mip_levels(device: &Arc<Device>, image: &Arc<Image>) -> Result<(), DriverError> {
    #[derive(Clone, Copy, Pod, Zeroable)]
    #[repr(C)]
    struct PushConstants {
        a: Vec4,
        b: Vec4,
    }

    let vertical_gradient = Arc::new(GraphicPipeline::create(
        device,
        GraphicPipelineInfo::default(),
        [
            Shader::new_vertex(
                glsl!(
                    r#"
                    #version 460 core
                    #pragma shader_stage(vertex)

                    const vec2 POSITION[] = {
                        vec2(-1, -1),
                        vec2(-1,  1),
                        vec2( 1,  1),
                        vec2(-1, -1),
                        vec2( 1,  1),
                        vec2( 1, -1),
                    };

                    layout(location = 0) out float ab;

                    void main() {
                        vec2 position = POSITION[gl_VertexIndex];
                        ab = max(position.y, 0);
                        gl_Position = vec4(position, 0, 1);
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

                    layout(push_constant) uniform PushConstants {
                        layout(offset = 0) vec3 a;
                        layout(offset = 16) vec3 b;
                    };

                    layout(location = 0) in float ab;
                    layout(location = 0) out vec4 color;

                    void main() {
                        color = vec4(mix(a, b, ab), 1);
                    }
                    "#
                )
                .as_slice(),
            ),
        ],
    )?);

    let mut render_graph = RenderGraph::default();
    let image_info = image.info;
    let image = render_graph.bind_node(image);

    // NOTE: Each pass writes to a different mip level, and so although it's the same image they are
    // unable to be used as a single pass so we must call begin_pass for each. Without starting a
    // new pass for each level the Vulkan framebuffer would be set to the size of the first image.
    for mip_level in 0..image_info.mip_level_count {
        render_graph
            .begin_cmd_buf()
            .with_name("fill mip levels")
            .bind_pipeline(&vertical_gradient)
            .store_color_as(
                0,
                image,
                image_info
                    .default_view_info()
                    .to_builder()
                    .base_mip_level(mip_level)
                    .mip_level_count(1),
            )
            .record_subpass(|subpass, _| {
                subpass
                    .push_constants(bytes_of(&PushConstants {
                        a: vec3(0.0, 1.0, 1.0).extend(f32::NAN),
                        b: vec3(1.0, 0.0, 1.0).extend(f32::NAN),
                    }))
                    .draw(6, 1, 0, 0);
            });
    }

    // This is the overly-complicated way of picking queue family 0
    let queue_family_index = device
        .physical_device
        .queue_families
        .iter()
        .enumerate()
        .find_map(|(idx, family)| {
            family
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS)
                .then_some(idx)
        })
        .ok_or(DriverError::Unsupported)?;

    // Submits to the GPU but does not wait for anything to be finished
    render_graph
        .resolve()
        .submit(&mut LazyPool::new(device), queue_family_index, 0)
        .map(|_| ())
}

fn splat(device: &Arc<Device>) -> Result<Arc<GraphicPipeline>, DriverError> {
    Ok(Arc::new(GraphicPipeline::create(
        device,
        GraphicPipelineInfo::default(),
        [
            Shader::new_vertex(
                glsl!(
                    r#"
                    #version 460 core
                    #pragma shader_stage(vertex)

                    const vec2 POSITION[] = {
                        vec2(-1, -1),
                        vec2(-1,  1),
                        vec2( 1,  1),
                        vec2(-1, -1),
                        vec2( 1,  1),
                        vec2( 1, -1),
                    };
                    const vec2 TEXCOORD[] = {
                        vec2(0, 0),
                        vec2(0, 1),
                        vec2(1, 1),
                        vec2(0, 0),
                        vec2(1, 1),
                        vec2(1, 0),
                    };

                    layout(location = 0) out vec2 texcoord;

                    void main() {
                        texcoord = TEXCOORD[gl_VertexIndex];
                        gl_Position = vec4(POSITION[gl_VertexIndex], 0, 1);
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

                    layout(binding = 0) uniform sampler2D image;

                    layout(location = 0) in vec2 texcoord;
                    layout(location = 0) out vec4 color;

                    void main() {
                        color = texture(image, texcoord);
                    }
                    "#
                )
                .as_slice(),
            )
            .image_sampler(
                0,
                SamplerInfoBuilder::default().mipmap_mode(vk::SamplerMipmapMode::LINEAR),
            ),
        ],
    )?))
}

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
