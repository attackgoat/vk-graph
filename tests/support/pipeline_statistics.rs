use {
    super::TestDevice,
    ash::vk,
    std::sync::{Arc, Mutex},
    vk_graph::{
        Graph,
        cmd::{LoadOp, StoreOp},
        driver::{
            buffer::{Buffer, BufferInfo},
            device::Device,
            graphics::{GraphicsPipeline, GraphicsPipelineInfo},
            image::{Image, ImageInfo},
        },
        node::AnyBufferNode,
        pool::hash::HashPool,
        stream::CommandStream,
    },
    vk_shader_macros::glsl,
    vk_sync::AccessType,
};

struct QueryOwner {
    device: Device,
    pool: vk::QueryPool,
    result: Arc<Mutex<Option<[[u64; 7]; 2]>>>,
}

impl Drop for QueryOwner {
    fn drop(&mut self) {
        let mut result = [[0u64; 7]; 2];
        unsafe {
            if self
                .device
                .get_query_pool_results(
                    self.pool,
                    0,
                    &mut result,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
                )
                .is_ok()
            {
                *self.result.lock().unwrap() = Some(result);
            }
            self.device.destroy_query_pool(self.pool, None);
        }
    }
}

#[test]
#[ignore = "requires Vulkan validation and pipeline statistics support"]
fn caller_stream_owns_queries_through_completion() -> anyhow::Result<()> {
    let device = TestDevice::new()?;
    let features = unsafe {
        device
            .physical
            .instance
            .get_physical_device_features(device.physical.handle)
    };
    if features.pipeline_statistics_query == vk::FALSE
        || !device.physical.features_v1_2.host_query_reset
    {
        eprintln!("SKIP: pipeline statistics/host query reset unsupported; 0 queries executed");
        return Ok(());
    }
    let Some(queue_family) = device.physical.queue_families.iter().position(|queue| {
        queue.queue_count > 0 && queue.queue_flags.contains(vk::QueueFlags::GRAPHICS)
    }) else {
        eprintln!("SKIP: graphics queue unavailable; 0 queries executed");
        return Ok(());
    };
    let flags = vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_VERTICES
        | vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_PRIMITIVES
        | vk::QueryPipelineStatisticFlags::VERTEX_SHADER_INVOCATIONS
        | vk::QueryPipelineStatisticFlags::CLIPPING_INVOCATIONS
        | vk::QueryPipelineStatisticFlags::CLIPPING_PRIMITIVES
        | vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS;
    let pool = unsafe {
        device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::PIPELINE_STATISTICS)
                .query_count(2)
                .pipeline_statistics(flags),
            None,
        )?
    };
    unsafe {
        device.reset_query_pool(pool, 0, 2);
    }
    let result = Arc::new(Mutex::new(None));
    let owner = Arc::new(QueryOwner {
        device: device.clone(),
        pool,
        result: Arc::clone(&result),
    });
    let weak = Arc::downgrade(&owner);
    let pipeline = GraphicsPipeline::create(
        &device,
        GraphicsPipelineInfo::default()
            .into_builder()
            .cull_mode(vk::CullModeFlags::NONE),
        [
            glsl!(
                r#"#version 460
            #pragma shader_stage(vertex)
            layout(set=0, binding=0) uniform Globals { float scale; } globals;
            void main() {
                vec2 p[3] = vec2[](vec2(-.5,-.5),vec2(.5,-.5),vec2(0,.5));
                gl_Position = vec4(p[gl_VertexIndex] * globals.scale,0,1);
            }"#
            )
            .as_slice(),
            glsl!(
                r#"#version 460
            #pragma shader_stage(fragment)
            layout(location=0) out vec4 color;
            void main() { color=vec4(1); }"#
            )
            .as_slice(),
        ],
    )?;
    let mut graph = Graph::new();
    let image_info = ImageInfo::image_2d(
        16,
        16,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT,
    );
    let image = graph.bind_resource(Image::create(&device, image_info)?);
    let globals_info = BufferInfo::device_mem(
        16,
        vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
    );
    let globals = graph.bind_resource(Buffer::create(&device, globals_info)?);
    graph.fill_buffer(globals, 0..16, 1.0f32.to_bits());
    let captured_owner = Arc::clone(&owner);
    let stream = CommandStream::finalize(|stream| {
        let image = stream.arg(image_info);
        let globals = stream.arg(globals_info);
        stream
            .begin_cmd()
            .bind_pipeline(&pipeline)
            .color_attachment_image(0, image, LoadOp::DontCare, StoreOp::Store)
            .shader_resource_access(
                0,
                AnyBufferNode::from(globals),
                AccessType::VertexShaderReadUniformBuffer,
            )
            .record_cmd(move |cmd| {
                let owner = &captured_owner;
                unsafe {
                    cmd.device.cmd_begin_query(
                        cmd.handle,
                        owner.pool,
                        0,
                        vk::QueryControlFlags::empty(),
                    );
                }
                cmd.draw(3, 1, 0, 0);
                unsafe {
                    cmd.device.cmd_end_query(cmd.handle, owner.pool, 0);
                    cmd.device.cmd_begin_query(
                        cmd.handle,
                        owner.pool,
                        1,
                        vk::QueryControlFlags::empty(),
                    );
                    cmd.device.cmd_end_query(cmd.handle, owner.pool, 1);
                }
            });
        (image, globals)
    })
    .into_stream();
    graph
        .insert_cmd_stream(&stream)
        .with_arg(stream.args.0, image)
        .with_arg(stream.args.1, globals)
        .finish();
    drop(stream);
    drop(owner);
    assert!(weak.upgrade().is_some());
    let mut pool = HashPool::new(&device);
    let mut fence = graph
        .finalize()
        .queue_submit(&mut pool, queue_family as u32, 0)?;
    assert!(
        weak.upgrade().is_some(),
        "query owner dropped after recording but before completion"
    );
    fence.wait()?;
    assert!(
        weak.upgrade().is_none(),
        "query owner not released after completion"
    );
    let result = result
        .lock()
        .unwrap()
        .expect("query results unavailable after completion");
    assert_eq!(&result[0][..5], &[3, 1, 3, 1, 1]);
    assert!(result[0][5] > 0);
    assert_eq!(result[0][6], 1);
    assert_eq!(result[1], [0, 0, 0, 0, 0, 0, 1]);
    eprintln!(
        "pipeline statistics: drawn={:?}, empty={:?}; 2 queries executed",
        result[0], result[1]
    );
    let captured = Arc::new(());
    let abandoned = Arc::downgrade(&captured);
    let mut graph = Graph::new();
    let stream = CommandStream::finalize(|stream| {
        stream
            .begin_cmd()
            .bind_pipeline(&pipeline)
            .record_cmd(move |_| {
                let _ = &captured;
            });
    })
    .into_stream();
    graph.insert_cmd_stream(&stream).finish();
    drop(stream);
    assert!(abandoned.upgrade().is_some());
    drop(graph);
    assert!(abandoned.upgrade().is_none());
    drop(fence);
    drop(pool);
    drop(pipeline);
    device.finish();
    Ok(())
}
