use {
    bytemuck::cast_slice,
    clap::Parser,
    log::warn,
    std::{mem::size_of, sync::Arc},
    vk_graph_prelude::*,
    vk_shader_macros::glsl,
};

// Min/max sampler reduction is commonly used to create depth buffer mip-maps for use with gpu-based
// visibility determination.
//
// Support for min/max sampling is core to Vulkan 1.2 however different graphics cards may have
// varying supported properties which are detailed by the physical device property structures. This
// example checks for that support.
//
// Note that this example only reduces the sample "depth image" once, and it does not fully occupy
// the compute units of the GPU by using larger local group sizes.
fn main() -> Result<(), DriverError> {
    pretty_env_logger::init();

    let mut graph = Graph::default();
    let args = Args::parse();
    let device_info = DeviceInfoBuilder::default().debug(args.debug);
    let device = Device::new(device_info)?;
    let size = 4;

    // The 4x4 depth image will have pixels that look like this:
    //   0.0   1.0   2.0   3.0
    //   4.0   5.0   6.0   7.0
    //   8.0   9.0  10.0  11.0
    //  12.0  13.0  14.0  15.0
    let depth_image = fill_depth_image(&device, &mut graph, size)?;

    // These 2x2 reduced images have undefined data until we wait on the results later
    let min_reduced_image = reduce_depth_image(
        &device,
        &mut graph,
        depth_image,
        vk::SamplerReductionMode::MIN,
    )?;
    let max_reduced_image = reduce_depth_image(
        &device,
        &mut graph,
        depth_image,
        vk::SamplerReductionMode::MAX,
    )?;

    // Create result buffers so we can read back the results
    let min_result_buf = copy_image_to_buffer(&device, &mut graph, min_reduced_image)?;
    let max_result_buf = copy_image_to_buffer(&device, &mut graph, max_reduced_image)?;

    graph
        .queue()
        .submit(&mut HashPool::new(&device), 0, 0)?
        .wait_until_executed()?;

    // For each image we have reduced each 2x2 pixel group into the min/max values of each group
    let min_result_data: &[f32] = cast_slice(Buffer::mapped_slice(&min_result_buf));
    let max_result_data: &[f32] = cast_slice(Buffer::mapped_slice(&max_result_buf));

    // The minimum result data should look like this:
    //   0.0   2.0
    //   8.0  10.0
    println!("{min_result_data:?}");

    // The maximum result data should look like this:
    //   5.0   7.0
    //  13.0  15.0
    println!("{max_result_data:?}");

    assert_eq!(min_result_data.len(), 4);
    assert_eq!(max_result_data.len(), 4);

    assert_eq!(min_result_data[0], 0.0);
    assert_eq!(min_result_data[1], 2.0);
    assert_eq!(min_result_data[2], 8.0);
    assert_eq!(min_result_data[3], 10.0);

    assert_eq!(max_result_data[0], 5.0);
    assert_eq!(max_result_data[1], 7.0);
    assert_eq!(max_result_data[2], 13.0);
    assert_eq!(max_result_data[3], 15.0);

    Ok(())
}

fn fill_depth_image(
    device: &Device,
    graph: &mut Graph,
    size: u32,
) -> Result<ImageNode, DriverError> {
    let info = ImageInfo::image_2d(
        size,
        size,
        vk::Format::D32_SFLOAT,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
    );
    let ImageInfo {
        fmt,
        ty,
        tiling,
        usage,
        flags,
        ..
    } = info;

    // Sometimes required because support is not 100% common: Check min/max reduction support
    // https://vulkan.gpuinfo.org/listdevicescoverage.php?extension=VK_EXT_sampler_filter_minmax&platform=all
    let fmt_props = device.physical_device.format_properties(fmt);
    if !fmt_props.optimal_tiling_features.contains(
        vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
            | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_MINMAX,
    ) {
        // In this case you might just fall back to a compute shader algorthm
        warn!("Requested image does not support min/max reduction");

        return Err(DriverError::Unsupported);
    }

    // If this is not supported you would need a fallback algorithm (this duplicates the check
    // we already performed above, it's just a different way to go about finding the answer)
    assert!(
        device
            .physical_device
            .sampler_filter_minmax_properties
            .single_component_formats
    );

    // Not required, but good practice: Check image format support
    let image_fmt_props = device
        .physical_device
        .image_format_properties(fmt, ty, tiling, usage, flags)?
        .ok_or(DriverError::Unsupported)?;
    if size > image_fmt_props.max_extent.width || size > image_fmt_props.max_extent.height {
        // In this case you might use a smaller image
        warn!("Requested image is too big");

        return Err(DriverError::Unsupported);
    }

    // You could check this if you needed to reduce multiple channel images:
    // device.physical_device.sampler_filter_minmax_properties.image_component_mapping

    let depth_data = (0..size.pow(2)).map(|x| x as f32).collect::<Box<_>>();
    let depth_data = graph.bind_resource(Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::TRANSFER_SRC,
        cast_slice(&depth_data),
    )?);
    let depth_image = graph.bind_resource(Image::create(device, info)?);
    graph.copy_buffer_to_image(depth_data, depth_image);

    Ok(depth_image)
}

fn reduce_depth_image(
    device: &Device,
    graph: &mut Graph,
    depth_image: ImageNode,
    reduction_mode: vk::SamplerReductionMode,
) -> Result<ImageNode, DriverError> {
    let depth_info = graph.resource(depth_image).info;

    assert_eq!(depth_info.width, depth_info.height);

    // (We use R32_SFLOAT because D32_SFLOAT has very low support for the STORAGE usage and most
    // implementations would be reading the image elsewhere instead of using it as a depth image)
    let reduced_info = ImageInfo::image_2d(
        depth_info.width >> 1,
        depth_info.height >> 1,
        vk::Format::R32_SFLOAT,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
    );
    let reduced_image = graph.bind_resource(Image::create(device, reduced_info)?);

    graph
        .begin_cmd()
        .debug_name("Reduce depth image")
        .bind_pipeline(ComputePipeline::create(
            device,
            ComputePipelineInfo::default(),
            Shader::new_compute(
                glsl!(
                    r#"
                    #version 460 core
                    #pragma shader_stage(compute)
                
                    layout(binding = 0) uniform sampler2D depth_image;
                    layout(binding = 1) writeonly uniform image2D reduced_image;

                    void main() {
                        ivec2 reduced_size = imageSize(reduced_image);
                        vec2 sample_xy = vec2(gl_GlobalInvocationID.xy) + 0.5;
                        vec4 sample_val = texture(depth_image, sample_xy / vec2(reduced_size));

                        ivec2 store_xy = ivec2(gl_GlobalInvocationID.xy);
                        imageStore(reduced_image, store_xy, sample_val);
                    }
                    "#
                )
                .as_slice(),
            )
            .image_sampler(0, SamplerInfo::LINEAR.reduction_mode(reduction_mode)),
        )?)
        .shader_resource_access(0, depth_image, AccessType::ComputeShaderReadOther)
        .shader_resource_access(1, reduced_image, AccessType::ComputeShaderWrite)
        .record_cmd_buf(move |cmd_buf, _| {
            cmd_buf.dispatch(reduced_info.width, reduced_info.height, 1);
        });

    Ok(reduced_image)
}

fn copy_image_to_buffer(
    device: &Device,
    graph: &mut Graph,
    reduced_image: ImageNode,
) -> Result<Arc<Buffer>, DriverError> {
    let reduced_info = graph.resource(reduced_image).info;
    let result_len = (reduced_info.width * reduced_info.height) as vk::DeviceSize
        * size_of::<f32>() as vk::DeviceSize;
    let result_buf = graph.bind_resource(Buffer::create(
        device,
        BufferInfo::host_mem(result_len, vk::BufferUsageFlags::TRANSFER_DST),
    )?);

    graph.copy_image_to_buffer(reduced_image, result_buf);

    Ok(graph.resource(result_buf).clone())
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
