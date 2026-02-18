use {
    bytemuck::cast_slice,
    clap::Parser,
    std::{mem::size_of, sync::Arc, time::Instant},
    vk_graph_prelude::*,
    vk_shader_macros::glsl,
};

/// Advanced example demonstrating subgroup operations (arithmetic and ballot).
///
/// This code demonstrates "exclusive sum", which is a GPU-implementation of "parallel prefix scan".
///
/// Given an input buffer of unsigned integers, the result is an output buffer where each index is
/// the sum of all items prior to that index. Example:
///
/// Input:  [1, 1, 1, 1]
/// Output: [0, 1, 2, 3]
///
/// Input:  [1, 4, 0, 2]
/// Output: [0, 1, 5, 5]
///
/// This algorthim is useful in *many* places, one of which being GPU-driven rendering. In that case
/// you may have a list of meshes where each index contains the number of those meshes in the scene.
/// Exclusive sum might be used to generate an offset where you may safely record instance drawing
/// commands for a given mesh.
///
/// https://www.khronos.org/blog/vulkan-subgroup-tutorial
/// https://www.khronos.org/assets/uploads/developers/library/2018-vulkan-devday/06-subgroups.pdf
fn main() -> Result<(), DriverError> {
    pretty_env_logger::init();

    let args = Args::parse();
    let device_info = DeviceInfoBuilder::default().debug(args.debug);
    let device = Arc::new(Device::new(device_info)?);
    let Vulkan11Properties {
        subgroup_size,
        subgroup_supported_operations,
        ..
    } = device.physical_device.properties_v1_1;

    assert!(subgroup_supported_operations.contains(vk::SubgroupFeatureFlags::ARITHMETIC));
    assert!(subgroup_supported_operations.contains(vk::SubgroupFeatureFlags::BALLOT));

    let reduce_pipeline = create_reduce_pipeline(&device)?;
    let excl_sum_pipeline = create_exclusive_sum_pipeline(&device)?;

    for data_len in [32, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 32768, 65536] {
        // Provided data must always be at least a full workgroup
        if data_len < subgroup_size {
            continue;
        }

        let input_data = generate_input_data(data_len);
        let output_data =
            exclusive_sum(&device, &reduce_pipeline, &excl_sum_pipeline, &input_data)?;

        assert_output_data(&input_data, &output_data);
    }

    Ok(())
}

fn exclusive_sum(
    device: &Arc<Device>,
    reduce_pipeline: &Arc<ComputePipeline>,
    scan_pipeline: &Arc<ComputePipeline>,
    input_data: &[u32],
) -> Result<Vec<u32>, DriverError> {
    let mut render_graph = RenderGraph::default();

    let input_buf = render_graph.bind_node(Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        cast_slice(input_data),
    )?);

    let output_buf = render_graph.bind_node(Arc::new(Buffer::create(
        device,
        BufferInfo::host_mem(
            input_data.len() as vk::DeviceSize * size_of::<u32>() as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        ),
    )?));

    let workgroup_count =
        input_data.len() as u32 / device.physical_device.properties_v1_1.subgroup_size;
    let reduce_count = workgroup_count - 1;
    let workgroup_buf = render_graph.bind_node(Buffer::create(
        device,
        BufferInfo::device_mem(
            reduce_count.max(1) as vk::DeviceSize * size_of::<u32>() as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        ),
    )?);

    if reduce_count > 0 {
        render_graph
            .begin_cmd_buf()
            .with_name("exclusive sum reduce")
            .bind_pipeline(reduce_pipeline)
            .read_descriptor(0, input_buf)
            .write_descriptor(1, workgroup_buf)
            .record_compute(move |compute, _| {
                compute.dispatch(reduce_count, 1, 1);
            });
    }

    render_graph
        .begin_cmd_buf()
        .with_name("exclusive sum scan")
        .bind_pipeline(scan_pipeline)
        .read_descriptor(0, workgroup_buf)
        .read_descriptor(1, input_buf)
        .write_descriptor(2, output_buf)
        .record_compute(move |compute, _| {
            compute.dispatch(workgroup_count, 1, 1);
        });

    let output_buf = render_graph.unbind_node(output_buf);
    let mut cmd_buf = render_graph
        .resolve()
        .submit(&mut HashPool::new(device), 0, 0)?;

    let started = Instant::now();
    cmd_buf.wait_until_executed()?;

    println!(
        "Waited {}μs (len={})",
        (Instant::now() - started).as_micros(),
        input_data.len()
    );

    Ok(cast_slice(Buffer::mapped_slice(&output_buf)).to_vec())
}

fn generate_input_data(data_len: u32) -> Vec<u32> {
    let mut data = Vec::with_capacity(data_len as _);
    for i in 0..data_len {
        data.push(i);
    }

    data
}

fn assert_output_data(input_data: &[u32], output_data: &[u32]) {
    let mut sum = 0;

    for idx in 0..input_data.len() {
        assert_eq!(sum, output_data[idx], "exclusive sum at {idx} not equal");

        sum += input_data[idx];
    }
}

fn create_reduce_pipeline(device: &Arc<Device>) -> Result<Arc<ComputePipeline>, DriverError> {
    Ok(Arc::new(ComputePipeline::create(
        device,
        ComputePipelineInfo::default(),
        Shader::new_compute(
            glsl!(
                target: vulkan1_2,
                r#"
                #version 460 core
                #extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
                #extension GL_KHR_shader_subgroup_arithmetic : require
                #pragma shader_stage(compute)

                layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

                layout(binding = 0) restrict readonly buffer InputBuffer {
                    uint32_t input_buf[];
                };

                layout(binding = 1) restrict writeonly buffer WorkgroupBuffer {
                    uint32_t workgroup_buf[];
                };

                void main() {
                    uint32_t sum = subgroupAdd(input_buf[gl_GlobalInvocationID.x]);

                    if (subgroupElect()) {
                        workgroup_buf[gl_WorkGroupID.x] = sum;
                    }
                }
                "#
            )
            .as_slice(),
        )
        .specialization_info(SpecializationInfo {
            data: device
                .physical_device
                .properties_v1_1
                .subgroup_size
                .to_ne_bytes()
                .to_vec(),
            map_entries: vec![vk::SpecializationMapEntry {
                constant_id: 0,
                offset: 0,
                size: size_of::<u32>(),
            }],
        }),
    )?))
}

fn create_exclusive_sum_pipeline(
    device: &Arc<Device>,
) -> Result<Arc<ComputePipeline>, DriverError> {
    Ok( Arc::new(
        ComputePipeline::create(
            device,
            ComputePipelineInfo::default(),
            Shader::new_compute(glsl!(
                target: vulkan1_2,
                r#"
                #version 460 core
                #extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
                #extension GL_KHR_shader_subgroup_arithmetic : require
                #pragma shader_stage(compute)

                layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

                layout(binding = 0) restrict readonly buffer WorkgroupBuffer {
                    uint32_t workgroup_buf[];
                };

                layout(binding = 1) restrict readonly buffer InputBuffer {
                    uint32_t input_buf[];
                };

                layout(binding = 2) restrict writeonly buffer OutputBuffer {
                    uint32_t output_buf[];
                };

                void main() {
                    uint32_t subgroup_sum = subgroupExclusiveAdd(input_buf[gl_GlobalInvocationID.x]);
                    uint32_t workgroup_sum = 0;

                    uint workgroups_per_subgroup_invocation = (gl_NumWorkGroups.x + gl_SubgroupSize - 1) / gl_SubgroupSize;
                    uint start = gl_SubgroupInvocationID * workgroups_per_subgroup_invocation;
                    uint end = min(start + workgroups_per_subgroup_invocation, gl_WorkGroupID.x);
                    for (uint workgroup_id = start; workgroup_id < end; workgroup_id++) {
                        workgroup_sum += workgroup_buf[workgroup_id];
                    }

                    workgroup_sum = subgroupAdd(workgroup_sum);

                    output_buf[gl_GlobalInvocationID.x] = subgroup_sum + workgroup_sum;
                }
                "#
        ).as_slice())
            .specialization_info(SpecializationInfo {
                data: device.physical_device.properties_v1_1.subgroup_size.to_ne_bytes().to_vec(),
                map_entries: vec![vk::SpecializationMapEntry {
                    constant_id: 0,
                    offset: 0,
                    size: size_of::<u32>(),
                }],
            }),
        )?
    ))
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
