use {
    super::TestDevice,
    ash::vk,
    std::sync::Arc,
    vk_graph::{
        Graph, Node as _,
        driver::{
            DriverError,
            buffer::{Buffer, BufferInfo},
            compute::{ComputePipeline, ComputePipelineInfo},
            image::{Image, ImageInfo},
            shader::Descriptor,
        },
        pool::hash::HashPool,
    },
    vk_shader_macros::glsl,
    vk_sync::AccessType,
};

#[test]
#[ignore = "requires Vulkan device"]
fn optimized_shader_bindings_skip_removed_resources_and_keep_accesses() -> Result<(), DriverError> {
    let device = TestDevice::new()?;
    let spirv = glsl!(kind: comp, optimize: performance, r#"
        #version 460
        layout(local_size_x = 1) in;
        layout(set = 0, binding = 0, std430) buffer Output { uint value; } outputs[2];
        layout(set = 0, binding = 1, std430) readonly buffer Unused { uint value; } unused_buf[2];
        layout(set = 0, binding = 2, std430) buffer Tail { uint value; } tail;
        layout(set = 0, binding = 3) uniform sampler2D unused_texture;
        layout(set = 3, binding = 0, std430) readonly buffer UnusedSet { uint value; } unused_set;
        layout(push_constant) uniform Parameters {
            uint input_value;
            layout(offset = 64) vec4 unused_tail;
        } parameters;
        void main() {
            outputs[0].value = parameters.input_value;
            outputs[1].value = parameters.input_value + 1;
            tail.value = parameters.input_value + 2;
        }
    "#);
    let pipeline =
        ComputePipeline::create(&device, ComputePipelineInfo::default(), spirv.as_slice())?;
    for (set, binding) in [(0, 1), (0, 3), (3, 0)] {
        assert!(
            !pipeline
                .inner
                .descriptor_bindings
                .contains_key(&Descriptor { set, binding })
        );
    }
    assert_eq!(pipeline.inner.descriptor_info.layouts.len(), 1);

    let mut graph = Graph::new();
    let mut outputs = Vec::new();
    for _ in 0..3 {
        outputs.push(Arc::new(Buffer::create(
            &device,
            BufferInfo::host_mem(4, vk::BufferUsageFlags::STORAGE_BUFFER),
        )?));
    }
    let output_nodes = outputs
        .iter()
        .map(|buffer| graph.bind_resource(buffer))
        .collect::<Vec<_>>();
    let unused = Arc::new(Buffer::create(
        &device,
        BufferInfo::host_mem(
            4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        ),
    )?);
    let weak_unused = Arc::downgrade(&unused);
    let unused_node = graph.bind_resource(&unused);
    let image = graph.bind_resource(Image::create(
        &device,
        ImageInfo::image_2d(
            1,
            1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
        ),
    )?);
    graph.fill_buffer(unused_node, 0..4, 23);
    graph
        .begin_cmd()
        .bind_pipeline(&pipeline)
        .shader_resource_access((0, [0]), output_nodes[0], AccessType::ComputeShaderWrite)
        .shader_resource_access((0, [1]), output_nodes[1], AccessType::ComputeShaderWrite)
        .shader_resource_access((1, [0]), unused_node, AccessType::ComputeShaderReadOther)
        .shader_resource_access((1, [1]), unused_node, AccessType::ComputeShaderReadOther)
        .shader_resource_access(2, output_nodes[2], AccessType::ComputeShaderWrite)
        .shader_resource_access(
            3,
            image,
            AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
        )
        .shader_resource_access((3, 0), unused_node, AccessType::ComputeShaderReadOther)
        .record_cmd(|cmd| {
            cmd.push_constants(0, &41u32.to_ne_bytes())
                .dispatch(1, 1, 1);
        });
    let exec = graph
        .cmds
        .last()
        .unwrap()
        .execs
        .iter()
        .find(|exec| exec.func.is_some())
        .unwrap();
    assert!(exec.accesses.iter().any(|(index, accesses)| {
        index == unused_node.index()
            && accesses
                .iter()
                .any(|access| access.access == AccessType::ComputeShaderReadOther)
    }));
    assert!(
        exec.accesses
            .iter()
            .any(|(index, _)| index == image.index())
    );

    let submission = graph.finalize();
    drop(unused);
    let unused = weak_unused
        .upgrade()
        .expect("declared resource must remain alive");
    let mut fence = submission.queue_submit(&mut HashPool::new(&device), 0, 0)?;
    fence.wait()?;
    for (index, output) in outputs.iter().enumerate() {
        assert_eq!(
            Buffer::mapped_slice(output),
            (41 + index as u32).to_ne_bytes()
        );
    }
    assert_eq!(Buffer::mapped_slice(&unused), 23u32.to_ne_bytes());
    let sync_info = unused.sync_info();
    assert_eq!(sync_info.ranges.len(), 1);
    // A read after a write preserves the writer through the conservative General state.
    assert_eq!(
        (
            sync_info.ranges[0].stage_mask,
            sync_info.ranges[0].access_mask
        ),
        vk_graph::driver::pipeline_stage_access_flags(AccessType::General),
    );

    Ok(())
}

#[test]
#[ignore = "requires Vulkan device"]
fn optimized_shader_bindings_allow_empty_reflection() -> Result<(), DriverError> {
    let device = TestDevice::new()?;
    let spirv = glsl!(kind: comp, optimize: performance, r#"
        #version 460
        layout(local_size_x = 1) in;
        layout(set = 4, binding = 0, std430) readonly buffer Unused { uint value; } unused_buf;
        void main() {}
    "#);
    let pipeline =
        ComputePipeline::create(&device, ComputePipelineInfo::default(), spirv.as_slice())?;
    assert!(pipeline.inner.descriptor_bindings.is_empty());
    let mut graph = Graph::new();
    let unused = graph.bind_resource(Buffer::create(
        &device,
        BufferInfo::device_mem(4, vk::BufferUsageFlags::STORAGE_BUFFER),
    )?);
    graph
        .begin_cmd()
        .bind_pipeline(&pipeline)
        .shader_resource_access((4, 0), unused, AccessType::ComputeShaderReadOther)
        .record_cmd(move |cmd| {
            assert_eq!(cmd.resource(unused).info.size, 4);
            cmd.dispatch(1, 1, 1);
        });
    graph
        .finalize()
        .queue_submit(&mut HashPool::new(&device), 0, 0)?
        .wait()?;

    Ok(())
}

#[test]
#[ignore = "requires Vulkan device"]
fn optimized_shader_bindings_reject_invalid_reflected_resource() -> Result<(), DriverError> {
    let device = TestDevice::new()?;
    let spirv = glsl!(kind: comp, optimize: performance, r#"
        #version 460
        layout(local_size_x = 1) in;
        layout(binding = 0, std430) buffer Output { uint value; } output_buf;
        void main() { output_buf.value = 1; }
    "#);
    let pipeline =
        ComputePipeline::create(&device, ComputePipelineInfo::default(), spirv.as_slice())?;
    let mut graph = Graph::new();
    let image = graph.bind_resource(Image::create(
        &device,
        ImageInfo::image_2d(
            1,
            1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
        ),
    )?);
    graph
        .begin_cmd()
        .bind_pipeline(&pipeline)
        .shader_resource_access(0, image, AccessType::ComputeShaderWrite)
        .record_cmd(|cmd| {
            cmd.dispatch(1, 1, 1);
        });
    assert!(matches!(
        graph
            .finalize()
            .queue_submit(&mut HashPool::new(&device), 0, 0),
        Err(DriverError::InvalidData)
    ));

    Ok(())
}
