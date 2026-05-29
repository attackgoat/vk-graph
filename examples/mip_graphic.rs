mod profile_with_puffin;

use {
    ash::vk,
    bytemuck::{Pod, Zeroable, bytes_of},
    clap::Parser,
    core::f32,
    glam::{Vec4, vec3},
    std::sync::Arc,
    vk_graph::{
        Graph,
        cmd::{LoadOp, StoreOp},
        driver::{
            DriverError,
            device::Device,
            graphic::{GraphicPipeline, GraphicPipelineInfo},
            image::{Image, ImageInfo},
            shader::{SamplerInfoBuilder, Shader},
        },
        pool::lazy::LazyPool,
    },
    vk_graph_window::{Window, WindowError},
    vk_shader_macros::glsl,
    vk_sync::AccessType,
};

// TODO: Add texelFetch option

fn main() -> Result<(), WindowError> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    let args = Args::parse();
    let window = Window::builder().debug(args.debug).build()?;

    let size = 237u32;
    let mip_level_count = size.ilog2();

    assert_ne!(mip_level_count, 0, "size must be greater than one");

    let image_info = ImageInfo::image_2d(
        size,
        size,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
    )
    .into_builder()
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
                .graph
                .resource(frame.swapchain_image)
                .info
                .usage
                .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        );

        let image = frame.graph.bind_resource(&image);
        let swapchain_info = frame.graph.resource(frame.swapchain_image).info;
        let stripe_width = swapchain_info.width / mip_level_count;

        let mut cmd = frame
            .graph
            .begin_cmd()
            .debug_name("splat mips")
            .bind_pipeline(&splat);

        for mip_level in 0..mip_level_count {
            let stripe_x = mip_level * stripe_width;
            let load_op = if mip_level == 0 {
                LoadOp::CLEAR_BLACK_ALPHA_ZERO
            } else {
                LoadOp::Load
            };
            cmd = cmd
                .shader_subresource_access(
                    0,
                    image,
                    image_info
                        .into_image_view()
                        .into_builder()
                        .base_mip_level(mip_level)
                        .mip_level_count(1),
                    AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
                )
                .color_attachment_image(0, frame.swapchain_image, load_op, StoreOp::Store)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D {
                        x: stripe_x as _,
                        y: 0,
                    },
                    extent: vk::Extent2D {
                        width: stripe_width,
                        height: swapchain_info.height,
                    },
                })
                .record_cmd_buf(|cmd_buf| {
                    cmd_buf.draw(6, 1, 0, 0);
                });
        }
    })
}

fn fill_mip_levels(device: &Device, image: &Arc<Image>) -> Result<(), DriverError> {
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct PushConstants {
        a: Vec4,
        b: Vec4,
    }

    let vertical_gradient = GraphicPipeline::create(
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
    )?;

    let mut graph = Graph::default();
    let image_info = image.info;
    let image = graph.bind_resource(image);

    // NOTE: Each pass writes to a different mip level, and so although it's the same image they are
    // unable to be used as a single pass so we must call begin_pass for each. Without starting a
    // new pass for each level the Vulkan framebuffer would be set to the size of the first image.
    for mip_level in 0..image_info.mip_level_count {
        graph
            .begin_cmd()
            .debug_name("fill mip levels")
            .bind_pipeline(&vertical_gradient)
            .color_attachment_image_view(
                0,
                image,
                image_info
                    .into_image_view()
                    .into_builder()
                    .base_mip_level(mip_level)
                    .mip_level_count(1),
                LoadOp::DontCare,
                StoreOp::Store,
            )
            .record_cmd_buf(|cmd_buf| {
                cmd_buf
                    .push_constants(
                        0,
                        bytes_of(&PushConstants {
                            a: vec3(0.0, 1.0, 1.0).extend(f32::NAN),
                            b: vec3(1.0, 0.0, 1.0).extend(f32::NAN),
                        }),
                    )
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
                .then_some(idx as u32)
        })
        .ok_or(DriverError::Unsupported)?;

    // Submits to the GPU but does not wait for anything to be finished
    graph
        .into_queue()
        .submit(&mut LazyPool::new(device), queue_family_index, 0)
        .map(|_| ())
}

fn splat(device: &Device) -> Result<GraphicPipeline, DriverError> {
    GraphicPipeline::create(
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
    )
}

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
