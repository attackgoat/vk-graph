//! Run with Vulkan validation installed:
//! ```sh
//! RUST_LOG=debug VK_GRAPH_SKIP_VALIDATION_PARK=1 \
//! cargo test --offline --lib test_support::buffer_ownership -- --ignored --nocapture
//! ```

use {
    super::{
        TestDevice,
        ownership::{OwnershipSubmissions, supports_in_flight},
    },
    ash::vk,
    std::sync::Arc,
    vk_graph::{
        Graph,
        cmd::{LoadOp, StoreOp},
        driver::{
            buffer::{Buffer, BufferInfo},
            graphics::{GraphicsPipeline, GraphicsPipelineInfo},
            image::{Image, ImageInfo},
        },
        pool::hash::HashPool,
    },
    vk_shader_macros::glsl,
    vk_sync::AccessType,
};

#[test]
#[ignore = "requires Vulkan validation, graphics and a dedicated transfer queue"]
fn graphics_buffer_ownership_concurrent_sharing_and_local_host_write() -> anyhow::Result<()> {
    run_graphics_buffer_ownership(None)
}

#[test]
#[ignore = "requires Vulkan validation, graphics, dedicated transfer queue, timeline semaphores and synchronization2"]
fn graphics_buffer_ownership_in_flight() -> anyhow::Result<()> {
    for submit2 in [false, true] {
        run_graphics_buffer_ownership(Some(submit2))?;
    }
    Ok(())
}

fn run_graphics_buffer_ownership(in_flight: Option<bool>) -> anyhow::Result<()> {
    let device = TestDevice::new()?;
    let result = (|| -> anyhow::Result<()> {
        if in_flight.is_some() && !supports_in_flight(&device) {
            return Ok(());
        }
        let families = &device.physical.queue_families;
        let graphics = families
            .iter()
            .position(|q| q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::GRAPHICS));
        let transfer = families.iter().position(|q| {
            q.queue_count > 0
                && q.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && !q
                    .queue_flags
                    .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        });
        let (Some(graphics), Some(transfer)) = (graphics, transfer) else {
            eprintln!("SKIP: graphics/dedicated transfer queue unavailable; 0 scenarios executed");
            return Ok(());
        };
        if !device.physical.features_v1_0.fragment_stores_and_atomics {
            eprintln!("SKIP: fragment SSBO writes unsupported; 0 scenarios executed");
            return Ok(());
        }
        let (graphics, transfer) = (graphics as u32, transfer as u32);
        eprintln!(
            "ownership queues: graphics={graphics}:0, transfer={transfer}:0, in_flight={in_flight:?}"
        );
        let mut pool = HashPool::new(&device);
        let vertex = glsl!(kind: vert, r#"
            #version 450
            void main() {
                vec2 p[3] = vec2[](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
                gl_Position = vec4(p[gl_VertexIndex], 0, 1);
            }
        "#);
        let writer = GraphicsPipeline::create(
            &device,
            GraphicsPipelineInfo::builder().cull_mode(vk::CullModeFlags::NONE),
            [
                vertex.as_slice(),
                glsl!(kind: frag, r#"
                #version 450
                layout(set=0, binding=0, std430) writeonly buffer Output { uint value; } result;
                layout(location=0) out vec4 color;
                void main() { result.value = 0x12345678u; color = vec4(1); }
            "#)
                .as_slice(),
            ],
        )?;
        let attachment = Arc::new(Image::create(
            &device,
            ImageInfo::image_2d(
                1,
                1,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            ),
        )?);
        for sharing_mode in [vk::SharingMode::EXCLUSIVE, vk::SharingMode::CONCURRENT] {
            let mut submissions = OwnershipSubmissions::new(&device, in_flight)?;
            let exclusive = sharing_mode == vk::SharingMode::EXCLUSIVE;
            let output = Arc::new(Buffer::create(
                &device,
                BufferInfo::device_mem(
                    4,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                )
                .into_builder()
                .sharing_mode(sharing_mode),
            )?);
            let readback = Arc::new(Buffer::create(
                &device,
                BufferInfo::host_mem(4, vk::BufferUsageFlags::TRANSFER_DST),
            )?);
            let mut graph = Graph::new();
            let out = graph.bind_resource(&output);
            let image = graph.bind_resource(&attachment);
            graph
                .begin_cmd()
                .debug_name("fragment SSBO writer")
                .bind_pipeline(&writer)
                .color_attachment_image(0, image, LoadOp::CLEAR_BLACK_ALPHA_ZERO, StoreOp::Store)
                .shader_resource_access(0, out, AccessType::FragmentShaderWrite)
                .record_cmd(|cmd| {
                    cmd.draw(3, 1, 0, 0);
                });
            submissions.producer(graph, &mut pool, graphics, !exclusive)?;
            let state = output.sync_info();
            assert_eq!(
                state.ranges[0].stage_mask,
                vk::PipelineStageFlags::FRAGMENT_SHADER
            );
            assert_eq!(state.ranges[0].access_mask, vk::AccessFlags::SHADER_WRITE);
            assert_eq!(
                state.ranges[0].queue_family_index,
                exclusive.then_some(graphics)
            );

            // Exclusive sharing exercises release/acquire submissions. Concurrent sharing skips
            // ownership transfer, but its ordinary barrier must still be legal on the transfer queue.
            // In-flight concurrent sharing uses an explicit completion semaphore. Exclusive
            // sharing relies entirely on the library's release/acquire dependency.
            let mut graph = Graph::new();
            let source = graph.bind_resource(&output);
            let destination = graph.bind_resource(&readback);
            graph.copy_buffer(source, destination);
            graph
                .begin_cmd()
                .resource_access(destination, AccessType::HostRead)
                .record_cmd(|_| {});
            submissions.consumer(graph, &mut pool, transfer, !exclusive)?;
            assert_eq!(readback.mapped_slice(), &0x12345678u32.to_ne_bytes());
            assert_eq!(
                output.sync_info().ranges[0].queue_family_index,
                exclusive.then_some(transfer)
            );
            drop(readback);
            drop(output);
            device.assert_valid();
        }

        let reader = GraphicsPipeline::create(
            &device,
            GraphicsPipelineInfo::builder().cull_mode(vk::CullModeFlags::NONE),
            [
                vertex.as_slice(),
                glsl!(kind: frag, r#"
                #version 450
                layout(set=0, binding=0, std140) uniform Local { uvec4 value; } local_data;
                layout(set=0, binding=1, std140) uniform Remote { uvec4 value; } remote_data;
                layout(location=0) out vec4 color;
                void main() {
                    uint sum = local_data.value.x + remote_data.value.x;
                    color = vec4(float(sum) / 255.0, 0, 0, 1);
                }
            "#)
                .as_slice(),
            ],
        )?;
        let local = Arc::new(Buffer::create_from_slice(
            &device,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            bytemuck::cast_slice(&[17u32, 0, 0, 0]),
        )?);
        let remote = Arc::new(Buffer::create(
            &device,
            BufferInfo::device_mem(
                16,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            ),
        )?);
        // Preserve actual CPU initialization as local graphics-family HostWrite history.
        let mut graph = Graph::new();
        let node = graph.bind_resource(&local);
        graph
            .begin_cmd()
            .resource_access(node, AccessType::HostWrite)
            .record_cmd(|_| {});
        graph
            .finalize()
            .queue_submit(&mut pool, graphics, 0)?
            .wait()?;
        let mut submissions = OwnershipSubmissions::new(&device, in_flight)?;
        let mut graph = Graph::new();
        let staging = graph.bind_resource(Buffer::create_from_slice(
            &device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            bytemuck::cast_slice(&[25u32, 0, 0, 0]),
        )?);
        let destination = graph.bind_resource(&remote);
        graph.copy_buffer(staging, destination);
        submissions.producer(graph, &mut pool, transfer, false)?;
        assert_eq!(
            local.sync_info().ranges[0].stage_mask,
            vk::PipelineStageFlags::HOST
        );
        assert_eq!(
            local.sync_info().ranges[0].queue_family_index,
            Some(graphics)
        );
        assert_eq!(
            remote.sync_info().ranges[0].queue_family_index,
            Some(transfer)
        );

        let pixels = Arc::new(Buffer::create(
            &device,
            BufferInfo::host_mem(4, vk::BufferUsageFlags::TRANSFER_DST),
        )?);
        let mut graph = Graph::new();
        let local_node = graph.bind_resource(&local);
        let remote_node = graph.bind_resource(&remote);
        let image = graph.bind_resource(&attachment);
        let destination = graph.bind_resource(&pixels);
        graph
            .begin_cmd()
            .debug_name("local HOST plus remote buffer acquire")
            .bind_pipeline(&reader)
            .color_attachment_image(0, image, LoadOp::CLEAR_BLACK_ALPHA_ZERO, StoreOp::Store)
            .shader_resource_access(
                (0, 0),
                local_node,
                AccessType::FragmentShaderReadUniformBuffer,
            )
            .shader_resource_access(
                (0, 1),
                remote_node,
                AccessType::FragmentShaderReadUniformBuffer,
            )
            .record_cmd(|cmd| {
                cmd.draw(3, 1, 0, 0);
            });
        graph.copy_image_to_buffer(image, destination);
        graph
            .begin_cmd()
            .resource_access(destination, AccessType::HostRead)
            .record_cmd(|_| {});
        submissions.consumer(graph, &mut pool, graphics, false)?;
        assert_eq!(pixels.mapped_slice(), &[42, 0, 0, 255]);
        eprintln!("ownership: 3 scenarios executed, in_flight={in_flight:?}");
        Ok(())
    })();
    // Resources have been destroyed; finish also checks device and instance teardown.
    device.finish();
    result
}
