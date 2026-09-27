//! Verified pending-producer fixtures. Run under an external timeout to detect CPU waits.
//!
//! ```sh
//! timeout 90s env VK_GRAPH_SKIP_VALIDATION_PARK=1 \
//! cargo test --offline --lib ownership -- --ignored --nocapture --test-threads=1
//! ```

use {
    super::validation::ValidationSettings,
    ash::vk,
    vk_graph::{
        Graph,
        driver::{
            buffer::{Buffer, BufferInfo},
            cmd_buf::{CommandBuffer, CommandBufferInfo},
            compute::{ComputePipeline, ComputePipelineInfo},
            device::Device,
            fence::Fence,
        },
        pool::{Pool, hash::HashPool},
        submission::{QueueSubmitInfo, RecordSelection, SemaphoreSubmit2Info, SemaphoreSubmitInfo},
    },
    vk_shader_macros::glsl,
    vk_sync::AccessType,
};

pub(super) fn supports_in_flight(device: &Device) -> bool {
    let settings = ValidationSettings::from_env();
    eprintln!(
        "ownership validation: device={}, sync={}, shader_accesses={}, timeline={}, synchronization2={}",
        device.physical.properties_v1_0.device_name,
        settings.synchronization_enabled(),
        settings.shader_accesses_enabled(),
        device.physical.features_v1_2.timeline_semaphore,
        device.physical.vk_khr_synchronization2
    );
    if !settings.synchronization_enabled()
        || !settings.shader_accesses_enabled()
        || !device.physical.features_v1_2.timeline_semaphore
        || !device.physical.vk_khr_synchronization2
        || !device
            .physical
            .queue_families
            .iter()
            .any(|q| q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::COMPUTE))
    {
        eprintln!(
            "SKIP: in-flight ownership requires sync validation, shader accesses, compute, timeline semaphores and synchronization2; 0 scenarios executed"
        );
        return false;
    }
    true
}

/// Owns pending submissions and their semaphores through completion, including assertion unwind.
pub(super) struct OwnershipSubmissions {
    device: Device,
    gate: Option<vk::Semaphore>,
    completion: Option<vk::Semaphore>,
    consumer_submit2: bool,
    fences: Vec<Fence>,
    producer_fence: usize,
}

impl OwnershipSubmissions {
    pub(super) fn new(device: &Device, in_flight: Option<bool>) -> anyhow::Result<Self> {
        let mut result = Self {
            device: device.clone(),
            gate: None,
            completion: None,
            consumer_submit2: in_flight.unwrap_or(false),
            fences: Vec::new(),
            producer_fence: 0,
        };
        if in_flight.is_some() {
            let mut timeline =
                vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
            result.gate = Some(unsafe {
                device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut timeline),
                    None,
                )
            }?);
            result.completion = Some(Device::create_semaphore(device)?);
        }
        Ok(result)
    }

    fn submit_gate(&mut self, pool: &mut HashPool) -> anyhow::Result<()> {
        // Binary semaphore waits require their transitive signals to have been submitted.
        // A host-signaled timeline (or host-set event) cannot legally hold this dependency
        // chain open. Instead submit bounded GPU work and VERIFY the producer fence remains
        // unsignaled after submitting the consumer. Early completion fails, never counts as coverage.
        let family = self
            .device
            .physical
            .queue_families
            .iter()
            .position(|q| q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .expect("ownership tests require compute") as u32;
        let pipeline = ComputePipeline::create(
            &self.device,
            ComputePipelineInfo::default(),
            glsl!(kind: comp, r#"
            #version 450
            layout(local_size_x = 1) in;
            layout(set=0, binding=0, std430) writeonly buffer Result { uint value; } result;
            void main() {
                uint value = 7;
                for (uint i = 0; i < 16000000; ++i) {
                    value ^= value << 13;
                    value ^= value >> 17;
                    value ^= value << 5;
                }
                result.value = value;
            }
        "#)
            .as_slice(),
        )?;
        let mut graph = Graph::new();
        let output = graph.bind_resource(Buffer::create(
            &self.device,
            BufferInfo::device_mem(4, vk::BufferUsageFlags::STORAGE_BUFFER),
        )?);
        graph
            .begin_cmd()
            .debug_name("bounded ownership producer delay")
            .bind_pipeline(&pipeline)
            .shader_resource_access(0, output, AccessType::ComputeShaderWrite)
            .record_cmd(|cmd| {
                cmd.dispatch(1, 1, 1);
            });
        let signals = [SemaphoreSubmit2Info {
            semaphore: self.gate.unwrap(),
            value: 1,
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            device_index: 0,
        }];
        self.submit(
            graph,
            pool,
            family,
            QueueSubmitInfo::queue_submit2(&[], &signals),
        )
    }

    fn submit(
        &mut self,
        graph: Graph,
        pool: &mut HashPool,
        family: u32,
        info: QueueSubmitInfo<'_>,
    ) -> anyhow::Result<()> {
        let cmd: vk_graph::pool::Lease<CommandBuffer> =
            pool.resource(CommandBufferInfo::new(family))?;
        cmd.begin(
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        let recording = graph.finalize().record(pool, cmd, RecordSelection::All)?;
        recording.cmd_buf.end()?;
        let mut recorded = recording.finish()?;
        let mut fence = Fence::create(&self.device, false)?;
        recorded.queue_submit(&mut fence, 0, info)?;
        fence.drop_when_signaled(recorded);
        self.fences.push(fence);
        Ok(())
    }

    pub(super) fn producer(
        &mut self,
        graph: Graph,
        pool: &mut HashPool,
        family: u32,
        concurrent: bool,
    ) -> anyhow::Result<()> {
        if let Some(gate) = self.gate {
            self.submit_gate(pool)?;
            self.producer_fence = self.fences.len();
            let waits = [SemaphoreSubmit2Info {
                semaphore: gate,
                value: 1,
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                device_index: 0,
            }];
            let signals = [SemaphoreSubmit2Info {
                semaphore: self.completion.unwrap(),
                value: 0,
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                device_index: 0,
            }];
            self.submit(
                graph,
                pool,
                family,
                QueueSubmitInfo::queue_submit2(&waits, if concurrent { &signals } else { &[] }),
            )?;
            assert!(!self.fences[self.producer_fence].status()?);
        } else {
            self.submit(graph, pool, family, QueueSubmitInfo::QUEUE_SUBMIT)?;
            self.fences[0].wait()?;
        }
        Ok(())
    }

    pub(super) fn consumer(
        &mut self,
        graph: Graph,
        pool: &mut HashPool,
        family: u32,
        concurrent: bool,
    ) -> anyhow::Result<()> {
        let completion = if concurrent { self.completion } else { None };
        if self.gate.is_some() {
            assert!(
                !self.fences[self.producer_fence].status()?,
                "producer must remain pending before consumer submission"
            );
        }
        if self.consumer_submit2 {
            let waits = completion.map(|semaphore| SemaphoreSubmit2Info {
                semaphore,
                value: 0,
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                device_index: 0,
            });
            self.submit(
                graph,
                pool,
                family,
                QueueSubmitInfo::queue_submit2(waits.as_slice(), &[]),
            )?;
        } else {
            let waits = completion.map(|semaphore| SemaphoreSubmitInfo {
                semaphore,
                value: 0,
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            });
            self.submit(
                graph,
                pool,
                family,
                QueueSubmitInfo::queue_submit(waits.as_slice(), &[]),
            )?;
        }
        if self.gate.is_some() {
            assert!(
                !self.fences[self.producer_fence].status()?,
                "producer must remain pending after consumer submission"
            );
        }
        for fence in &mut self.fences {
            fence.wait()?;
        }
        Ok(())
    }
}

impl Drop for OwnershipSubmissions {
    fn drop(&mut self) {
        for fence in &mut self.fences {
            if let Err(error) = fence.wait() {
                eprintln!("ownership fence cleanup failed: {error}");
            }
        }
        self.fences.clear();
        unsafe {
            if let Some(semaphore) = self.completion {
                self.device.destroy_semaphore(semaphore, None);
            }
            if let Some(semaphore) = self.gate {
                self.device.destroy_semaphore(semaphore, None);
            }
        }
    }
}

#[test]
#[ignore = "requires Vulkan validation, distinct graphics/transfer queues, compute, timeline semaphores and synchronization2"]
fn partial_ownership_in_flight() -> anyhow::Result<()> {
    use {
        super::TestDevice,
        std::sync::Arc,
        vk_graph::driver::image::{Image, ImageInfo},
    };
    let device = TestDevice::new()?;
    let result = (|| -> anyhow::Result<()> {
        if !supports_in_flight(&device) {
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
            eprintln!(
                "SKIP: distinct graphics/dedicated transfer queues unavailable; 0 scenarios executed"
            );
            return Ok(());
        };
        let mut pool = HashPool::new(&device);
        let mut executed = 0;
        for submit2 in [false, true] {
            for (source_family, destination_family) in [
                (graphics as u32, transfer as u32),
                (transfer as u32, graphics as u32),
            ] {
                eprintln!(
                    "partial ownership: source={source_family}:0 destination={destination_family}:0 submit2={submit2}"
                );
                {
                    let source = Arc::new(Buffer::create(
                        &device,
                        BufferInfo::device_mem(
                            16,
                            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                        ),
                    )?);
                    let readback = Arc::new(Buffer::create(
                        &device,
                        BufferInfo::host_mem(8, vk::BufferUsageFlags::TRANSFER_DST),
                    )?);
                    let mut submissions = OwnershipSubmissions::new(&device, Some(submit2))?;
                    let mut producer = Graph::new();
                    let node = producer.bind_resource(&source);
                    producer.fill_buffer(node, 0..16, 0x12345678);
                    submissions.producer(producer, &mut pool, source_family, false)?;
                    let mut consumer = Graph::new();
                    let src = consumer.bind_resource(&source);
                    let dst = consumer.bind_resource(&readback);
                    consumer
                        .begin_cmd()
                        .copy_buffer(
                            src,
                            dst,
                            [vk::BufferCopy {
                                src_offset: 4,
                                dst_offset: 0,
                                size: 8,
                            }],
                        )
                        .end_cmd();
                    consumer
                        .begin_cmd()
                        .resource_access(dst, AccessType::HostRead)
                        .record_cmd(|_| {});
                    submissions.consumer(consumer, &mut pool, destination_family, false)?;
                    assert_eq!(
                        readback.mapped_slice(),
                        &0x12345678u32.to_ne_bytes().repeat(2)
                    );
                    let mut covered = [false; 16];
                    for snapshot in source.sync_info().ranges.iter() {
                        for byte in snapshot.range.start..snapshot.range.end {
                            assert!(!std::mem::replace(&mut covered[byte as usize], true));
                            assert_eq!(
                                snapshot.queue_family_index,
                                Some(if (4..12).contains(&byte) {
                                    destination_family
                                } else {
                                    source_family
                                })
                            );
                        }
                    }
                    assert!(covered.into_iter().all(|cell| cell));
                    executed += 1;
                }
                {
                    let source = Arc::new(Image::create(
                        &device,
                        ImageInfo::image_2d(
                            2,
                            2,
                            vk::Format::R8G8B8A8_UNORM,
                            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                        )
                        .into_builder()
                        .array_layer_count(2)
                        .mip_level_count(2),
                    )?);
                    let readback = Arc::new(Buffer::create(
                        &device,
                        BufferInfo::host_mem(4, vk::BufferUsageFlags::TRANSFER_DST),
                    )?);
                    let mut submissions = OwnershipSubmissions::new(&device, Some(submit2))?;
                    let mut producer = Graph::new();
                    let node = producer.bind_resource(&source);
                    let staging = producer.bind_resource(Buffer::create_from_slice(
                        &device,
                        vk::BufferUsageFlags::TRANSFER_SRC,
                        &[255u8, 0, 0, 255].repeat(10),
                    )?);
                    let regions = (0..2)
                        .flat_map(|layer| {
                            (0..2).map(move |mip| {
                                let width = 2 >> mip;
                                vk::BufferImageCopy::default()
                                    .buffer_offset(
                                        layer as u64 * 20 + if mip == 0 { 0 } else { 16 },
                                    )
                                    .buffer_row_length(width)
                                    .buffer_image_height(width)
                                    .image_subresource(
                                        vk::ImageSubresourceLayers::default()
                                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                                            .base_array_layer(layer)
                                            .layer_count(1)
                                            .mip_level(mip),
                                    )
                                    .image_extent(vk::Extent3D {
                                        width,
                                        height: width,
                                        depth: 1,
                                    })
                            })
                        })
                        .collect::<Vec<_>>();
                    producer
                        .begin_cmd()
                        .copy_buffer_to_image(staging, node, regions)
                        .end_cmd();
                    submissions.producer(producer, &mut pool, source_family, false)?;
                    let mut consumer = Graph::new();
                    let src = consumer.bind_resource(&source);
                    let dst = consumer.bind_resource(&readback);
                    consumer
                        .begin_cmd()
                        .copy_image_to_buffer(
                            src,
                            dst,
                            [vk::BufferImageCopy::default()
                                .buffer_row_length(1)
                                .buffer_image_height(1)
                                .image_subresource(
                                    vk::ImageSubresourceLayers::default()
                                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                                        .base_array_layer(1)
                                        .layer_count(1)
                                        .mip_level(1),
                                )
                                .image_extent(vk::Extent3D {
                                    width: 1,
                                    height: 1,
                                    depth: 1,
                                })],
                        )
                        .end_cmd();
                    consumer
                        .begin_cmd()
                        .resource_access(dst, AccessType::HostRead)
                        .record_cmd(|_| {});
                    submissions.consumer(consumer, &mut pool, destination_family, false)?;
                    assert_eq!(readback.mapped_slice(), &[255, 0, 0, 255]);
                    let mut covered = [[false; 2]; 2];
                    for snapshot in source.sync_info().subresources.iter() {
                        assert_eq!(snapshot.range.aspect_mask, vk::ImageAspectFlags::COLOR);
                        for layer in snapshot.range.base_array_layer
                            ..snapshot.range.base_array_layer + snapshot.range.layer_count
                        {
                            for mip in snapshot.range.base_mip_level
                                ..snapshot.range.base_mip_level + snapshot.range.level_count
                            {
                                assert!(!std::mem::replace(
                                    &mut covered[layer as usize][mip as usize],
                                    true
                                ));
                                let touched = layer == 1 && mip == 1;
                                assert_eq!(
                                    snapshot.queue_family_index,
                                    Some(if touched {
                                        destination_family
                                    } else {
                                        source_family
                                    })
                                );
                                assert_eq!(
                                    snapshot.layout,
                                    Some(if touched {
                                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                                    } else {
                                        vk::ImageLayout::TRANSFER_DST_OPTIMAL
                                    })
                                );
                            }
                        }
                    }
                    assert!(covered.into_iter().flatten().all(|cell| cell));
                    executed += 1;
                }
                device.assert_valid();
            }
        }
        eprintln!("partial ownership: {executed} scenarios executed");
        Ok(())
    })();
    device.finish();
    result
}
