mod profile_with_puffin;

use {bytemuck::cast_slice, clap::Parser, vk_graph_prelude::*, vk_shader_macros::glsl};

/// This program demonstrates a single render pass which uses multiple executions to record a chain
/// of image copies which reduce an input image from 4x4 into 2x2 and finally 1x1. This is useful
/// for GPU-based mesh instance culling, where the depth buffer is summarized into a mip chain that
/// can be queried to find the maximum depth for a given render area.
///
/// This technique is also known as a depth pyramid.
fn main() -> Result<(), DriverError> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    let args = Args::parse();
    let device_info = DeviceInfoBuilder::default().debug(args.debug);
    let device = Device::new(device_info)?;

    let mut graph = Graph::default();

    let depth_pyramid = graph.bind_resource(Image::create(
        &device,
        ImageInfo::image_2d(
            4,
            4,
            vk::Format::R32_SFLOAT,
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
        .to_builder()
        .mip_level_count(3),
    )?);
    let depth_info = graph.resource(depth_pyramid).info;

    // You would normally create this buffer by copying the depth attachment image
    #[allow(clippy::inconsistent_digit_grouping)]
    let depth_buf = graph.bind_resource(Buffer::create_from_slice(
        &device,
        vk::BufferUsageFlags::TRANSFER_SRC,
        cast_slice(&[
            [1.0f32, 2.0_, 3.0_, 4.0_],
            [5.0___, 6.0_, 7.0_, 8.0_],
            [9.0___, 10.0, 11.0, 12.0],
            [13.0__, 14.0, 15.0, 16.0],
        ]),
    )?);
    graph.copy_buffer_to_image(depth_buf, depth_pyramid);

    let mut pass = graph
        .begin_cmd()
        .debug_name("update depth pyramid")
        .bind_pipeline(ComputePipeline::create(
            &device,
            ComputePipelineInfo::default(),
            Shader::new_compute(
                glsl!(
                    r#"
                    #version 460 core
                    #pragma shader_stage(compute)

                    layout(binding = 0) uniform sampler2D src_mip;
                    layout(binding = 1, r32f) writeonly uniform image2D dst_mip;

                    void main() {
                        vec4 depth = texture(src_mip, vec2(gl_GlobalInvocationID.xy << 1) + 1.0);
                        imageStore(dst_mip, ivec2(gl_GlobalInvocationID.xy), depth);
                    }
                    "#
                )
                .as_slice(),
            )
            .image_sampler(
                0,
                SamplerInfoBuilder::default()
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .reduction_mode(vk::SamplerReductionMode::MAX)
                    .unnormalized_coordinates(true),
            ),
        )?);

    for mip_level in 1..depth_info.mip_level_count {
        pass = pass
            .shader_subresource_access(
                0,
                depth_pyramid,
                depth_info
                    .into_image_view()
                    .to_builder()
                    .base_mip_level(mip_level - 1)
                    .mip_level_count(1),
                AccessType::ComputeShaderReadOther,
            )
            .shader_subresource_access(
                1,
                depth_pyramid,
                depth_info
                    .into_image_view()
                    .to_builder()
                    .base_mip_level(mip_level)
                    .mip_level_count(1),
                AccessType::ComputeShaderWrite,
            )
            .record_cmd_buf(move |cmd_buf, _| {
                cmd_buf.dispatch(
                    depth_info.width >> mip_level,
                    depth_info.height >> mip_level,
                    1,
                );
            });
    }

    let depth_pixel = graph.bind_resource(Buffer::create(
        &device,
        BufferInfo::host_mem(size_of::<f32>() as _, vk::BufferUsageFlags::TRANSFER_DST),
    )?);
    graph.copy_image_to_buffer_region(
        depth_pyramid,
        depth_pixel,
        vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 1,
            buffer_image_height: 1,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: depth_info.mip_level_count - 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
        },
    );

    let depth_pixel = graph.resource(depth_pixel).clone();

    graph
        .resolve()
        .submit(&mut HashPool::new(&device), 0, 0)?
        .wait_until_executed()?;

    let depth_pixel = f32::from_ne_bytes(Buffer::mapped_slice(&depth_pixel).try_into().unwrap());

    println!("Final mip pixel value: {depth_pixel}",);

    assert_eq!(depth_pixel, 16.0);

    Ok(())
}

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
