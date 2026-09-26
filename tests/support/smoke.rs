//! Child-process validation of GPU-generated indirect dispatch arguments.

use {
    super::TestDevice,
    ash::vk,
    bytemuck::cast_slice,
    vk_graph::{
        Graph,
        driver::{
            DriverError,
            buffer::{Buffer, BufferInfo},
            compute::{ComputePipeline, ComputePipelineInfo},
            device::Device,
            shader::Shader,
        },
        pool::hash::HashPool,
    },
    vk_shader_macros::glsl,
    vk_sync::AccessType,
};

pub(super) fn run() -> Result<(), DriverError> {
    let device = TestDevice::new()?;
    indirect_dispatch(&device)?;
    device.finish();

    Ok(())
}

fn indirect_dispatch(device: &Device) -> Result<(), DriverError> {
    let queue_family_index = device
        .physical
        .queue_families
        .iter()
        .position(|family| {
            family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
        })
        .ok_or(DriverError::Unsupported)? as u32;
    let producer = ComputePipeline::create(
        device,
        ComputePipelineInfo::default(),
        Shader::new_compute(
            glsl!(
                r#"
                #version 450
                #pragma shader_stage(compute)
                layout(local_size_x = 1) in;
                layout(binding = 0, std430) writeonly buffer Arguments { uvec4 groups[2]; } args;
                void main() {
                    args.groups[0] = uvec4(0, 1, 1, 0);
                    args.groups[1] = uvec4(3, 1, 1, 0);
                }
                "#
            )
            .as_slice(),
        ),
    )?;
    let consumer = ComputePipeline::create(
        device,
        ComputePipelineInfo::default(),
        Shader::new_compute(
            glsl!(
                r#"
                #version 450
                #pragma shader_stage(compute)
                layout(local_size_x = 4) in;
                layout(binding = 0, std430) writeonly buffer Results { uint values[]; } result;
                layout(push_constant) uniform Constants { uint base; } constants;
                void main() {
                    uint index = gl_GlobalInvocationID.x;
                    result.values[constants.base + index] = index * index + 17;
                }
                "#
            )
            .as_slice(),
        ),
    )?;

    let mut graph = Graph::default();
    let arguments = graph.bind_resource(Buffer::create(
        device,
        BufferInfo::device_mem(
            32,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
        ),
    )?);
    const SENTINEL: u32 = 0xdead_beef;
    let results = graph.bind_resource(Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        cast_slice(&[SENTINEL; 16]),
    )?);

    graph
        .begin_cmd()
        .debug_name("generate indirect dispatch arguments")
        .bind_pipeline(&producer)
        .shader_resource_access(0, arguments, AccessType::ComputeShaderWrite)
        .record_cmd(|cmd| {
            cmd.dispatch(1, 1, 1);
        });
    graph
        .begin_cmd()
        .debug_name("consume zero-work indirect dispatch")
        .bind_pipeline(&consumer)
        .resource_access(arguments, AccessType::IndirectBuffer)
        .shader_resource_access(0, results, AccessType::ComputeShaderWrite)
        .record_cmd(move |cmd| {
            // The first record must leave the sentinel prefix untouched.
            cmd.push_constants(0, &0_u32.to_ne_bytes())
                .dispatch_indirect(arguments, 0);
        });
    graph
        .begin_cmd()
        .debug_name("consume nonzero-offset indirect dispatch")
        .bind_pipeline(&consumer)
        .resource_access(arguments, AccessType::IndirectBuffer)
        .shader_resource_access(0, results, AccessType::ComputeShaderWrite)
        .record_cmd(move |cmd| {
            // The second record launches three groups of four invocations after the prefix.
            cmd.push_constants(0, &4_u32.to_ne_bytes())
                .dispatch_indirect(arguments, 16);
        });
    graph
        .begin_cmd()
        .resource_access(results, AccessType::HostRead)
        .record_cmd(|_| {});

    let results = graph.resource(results).clone();
    let mut pool = HashPool::new(device);
    let mut fence = graph
        .finalize()
        .queue_submit(&mut pool, queue_family_index, 0)?;
    fence.wait()?;
    assert!(fence.status()?, "indirect dispatch fence did not signal");

    let actual: &[u32] = cast_slice(Buffer::mapped_slice(&results));
    let mut expected = [SENTINEL; 16];
    for (index, value) in expected[4..].iter_mut().enumerate() {
        *value = (index as u32).pow(2) + 17;
    }
    assert_eq!(actual, expected, "GPU-generated indirect dispatch results");
    println!("validated GPU-generated indirect dispatch: zero-work and nonzero-offset records");

    Ok(())
}
