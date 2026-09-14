use {
    super::{DeviceChecks, TestDevice},
    ash::{ext::debug_utils, vk},
    std::{env, ffi::CStr, process::Command},
    vk_graph::driver::{
        DriverError,
        instance::{Instance, InstanceInfo},
    },
};

#[test]
#[ignore = "requires Vulkan device"]
fn real_program() {
    const CHILD: &str = "VK_GRAPH_SMOKE_CHILD";
    if env::var_os(CHILD).is_some() {
        super::smoke::run().expect("Vulkan smoke test failed");
        return;
    }

    // Re-execute just this test in a fresh process with the same profile and feature set.
    // A normal example would link the production library, which has no disposal hooks.
    let module = module_path!().split_once("::").unwrap().1;
    let status = Command::new(env::current_exe().unwrap())
        .args(["--ignored", "--exact", "--nocapture"])
        .arg(format!("{module}::real_program"))
        .env(CHILD, "1")
        .status()
        .unwrap_or_else(|err| panic!("unable to run validation smoke process: {err}"));

    assert!(
        status.success(),
        "validation smoke process exited with {status}"
    );
}

#[test]
#[ignore = "requires Vulkan device and validation layers; records hazards without submitting"]
fn vulkan_validation_settings_control_hazard_detection() -> Result<(), DriverError> {
    use {
        super::validation::ValidationSettings,
        vk_graph::{
            Graph,
            driver::{
                buffer::{Buffer, BufferInfo},
                cmd_buf::{CommandBuffer, CommandBufferInfo},
                compute::{ComputePipeline, ComputePipelineInfo},
                shader::Shader,
            },
            pool::hash::HashPool,
            submission::RecordSelection,
        },
        vk_shader_macros::glsl,
        vk_sync::AccessType,
    };

    let settings = ValidationSettings::from_env();
    let mut device = TestDevice::new()?;
    let expected = usize::from(settings.synchronization_enabled())
        + usize::from(settings.shader_accesses_enabled());
    if expected > 0 {
        device.expect_validation_error("SYNC-HAZARD-WRITE-AFTER-WRITE", expected);
    }
    {
        let family = device
            .physical
            .queue_families
            .iter()
            .position(|family| {
                family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            })
            .ok_or(DriverError::Unsupported)? as u32;
        let mut command = CommandBuffer::create(&device, CommandBufferInfo::new(family))?;
        command.begin(&vk::CommandBufferBeginInfo::default())?;
        let buffer = Buffer::create(
            &device,
            BufferInfo::device_mem(4, vk::BufferUsageFlags::TRANSFER_DST),
        )?;
        // A transfer WAW needs synchronization validation, but no shader-access heuristic.
        unsafe {
            device.cmd_fill_buffer(command.handle, buffer.handle, 0, 4, 1);
            device.cmd_fill_buffer(command.handle, buffer.handle, 0, 4, 2);
        }

        let pipeline = ComputePipeline::create(
            &device,
            ComputePipelineInfo::default(),
            Shader::new_compute(
                glsl!(
                    r#"
                #version 450
                #pragma shader_stage(compute)
                layout(local_size_x = 1) in;
                layout(binding = 0, std430) writeonly buffer Values { uint value; } result;
                void main() { result.value = 1; }
            "#
                )
                .as_slice(),
            ),
        )?;
        let mut pool = HashPool::new(&device);
        let mut graph = Graph::new();
        let output = graph.bind_resource(Buffer::create(
            &device,
            BufferInfo::device_mem(4, vk::BufferUsageFlags::STORAGE_BUFFER),
        )?);
        graph
            .begin_cmd()
            .bind_pipeline(&pipeline)
            .shader_resource_access(0, output, AccessType::ComputeShaderWrite)
            .record_cmd(|cmd| {
                // Within one execution there is no graph-generated barrier between dispatches.
                // Detecting this WAW additionally requires the shader-access heuristic.
                cmd.dispatch(1, 1, 1);
                cmd.dispatch(1, 1, 1);
            });
        let recording = graph
            .finalize()
            .record(&mut pool, &mut command, RecordSelection::All)?;
        recording.cmd_buf.end()?;
        // These intentionally hazardous commands are never submitted to the GPU.
    }
    device.finish();
    Ok(())
}

// Submit synthetic diagnostics through Vulkan itself to exercise the registered callback and
// pUserData without issuing invalid GPU commands or relying on driver-specific VUIDs.
fn submit_validation_error(instance: &Instance, message: &CStr) {
    let debug = debug_utils::Instance::new(Instance::entry(instance), instance);
    let data = vk::DebugUtilsMessengerCallbackDataEXT::default()
        .message_id_name(c"VK_GRAPH_TEST_INSTANCE_SCOPED")
        .message(message);

    unsafe {
        debug.submit_debug_utils_message(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            &data,
        );
    }
}

#[test]
#[ignore = "requires Vulkan device and validation layers"]
fn vulkan_validation_reports_are_instance_scoped() -> Result<(), DriverError> {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::Barrier,
        thread,
    };

    let mut device = TestDevice::new()?;
    let report = Instance::validation_report(&device.physical.instance).unwrap();
    let other = Instance::create(InstanceInfo::builder().debug(true))?;
    let other_report = Instance::validation_report(&other).unwrap();

    assert!(report.errors().is_empty());
    assert!(other_report.errors().is_empty());

    thread::scope(|scope| {
        scope.spawn(|| {
            // Even the Vulkan logger's target must not affect either instance's report.
            log::error!(target: "vk_graph::driver::instance", "intentional unrelated log error");
            submit_validation_error(&other, c"other instance before expectation");
        });
    });

    device.assert_valid();

    assert_eq!(other_report.errors().len(), 1);

    device.expect_validation_error("VK_GRAPH_TEST_INSTANCE_SCOPED", 1);

    // The same message ID on another instance cannot satisfy this expectation.
    assert!(catch_unwind(AssertUnwindSafe(|| device.assert_valid())).is_err());

    let barrier = Barrier::new(2);
    let instance = &device.physical.instance;
    thread::scope(|scope| {
        scope.spawn(|| {
            barrier.wait();
            submit_validation_error(instance, c"own expected error");
        });
        scope.spawn(|| {
            barrier.wait();
            submit_validation_error(&other, c"other concurrent error");
        });
    });

    device.assert_valid();

    assert_eq!(report.errors().len(), 1);
    assert_eq!(report.errors()[0].message, "own expected error");
    assert_eq!(other_report.errors().len(), 2);

    drop(other);
    device.finish();

    // Report handles retain only diagnostics, not the Vulkan instances.
    assert_eq!(report.errors().len(), 1);
    assert_eq!(other_report.errors().len(), 2);

    Ok(())
}

#[test]
#[ignore = "requires Vulkan device and validation layers"]
fn vulkan_unexpected_validation_error_fails_only_its_session() -> Result<(), DriverError> {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let device = TestDevice::new()?;
    submit_validation_error(&device.physical.instance, c"deliberately unaccounted error");
    assert!(catch_unwind(AssertUnwindSafe(|| drop(device))).is_err());

    let next = TestDevice::new()?;
    next.assert_valid();
    Ok(())
}

#[test]
#[ignore = "requires Vulkan device and validation layers"]
fn vulkan_session_checks_are_independent() -> Result<(), DriverError> {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use vk_graph::driver::device::Device;

    for validation in [false, true] {
        for disposal in [false, true] {
            let checks = DeviceChecks {
                validation,
                disposal,
            };
            let mut device = TestDevice::with_checks(checks)?;
            assert_eq!(device.enabled_checks(), checks);
            assert_eq!(
                device.library_assertions_enabled(),
                cfg!(feature = "checked")
            );
            assert_eq!(device.physical.instance.info.debug, validation);
            assert_eq!(
                Instance::validation_report(&device.physical.instance).is_some(),
                validation
            );

            if !validation {
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        device.expect_validation_error("disabled", 1);
                    }))
                    .is_err()
                );
            }

            let device_report = Device::disposal_report(&device);
            let instance_report = Instance::disposal_report(&device.physical.instance).unwrap();
            device.assert_valid();
            assert!(!device_report.is_disposed());
            assert!(!instance_report.is_disposed());
            device.finish();
            assert!(device_report.is_disposed());
            assert!(instance_report.is_disposed());
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires Vulkan device"]
fn vulkan_disposal_checks_reject_escaped_owners() -> Result<(), DriverError> {
    use {
        std::{
            any::Any,
            panic::{AssertUnwindSafe, catch_unwind},
        },
        vk_graph::driver::{
            buffer::{Buffer, BufferInfo},
            device::Device,
        },
    };

    for disposal in [false, true] {
        for explicit in [false, true] {
            for owner in ["device", "instance", "buffer"] {
                let device = TestDevice::with_checks(DeviceChecks {
                    validation: false,
                    disposal,
                })?;
                let device_report = Device::disposal_report(&device);
                let instance_report = Instance::disposal_report(&device.physical.instance).unwrap();
                let escaped: Box<dyn Any> = match owner {
                    "device" => Box::new(device.clone()),
                    "instance" => Box::new(device.physical.instance.clone()),
                    _ => Box::new(Buffer::create(
                        &device,
                        BufferInfo::device_mem(4, vk::BufferUsageFlags::TRANSFER_SRC),
                    )?),
                };
                let result = catch_unwind(AssertUnwindSafe(|| {
                    if explicit {
                        device.finish();
                    } else {
                        drop(device);
                    }
                }));
                assert_eq!(result.is_err(), disposal, "{owner}, explicit={explicit}");
                assert_eq!(device_report.is_disposed(), owner == "instance");
                assert!(!instance_report.is_disposed());
                if let Err(error) = result {
                    let message = error.downcast_ref::<String>().unwrap();
                    assert!(message.contains("Vulkan disposal incomplete"), "{message}");
                }
                // Drop outside the caught panic so real Vulkan teardown still takes place.
                drop(escaped);
                assert!(device_report.is_disposed());
                assert!(instance_report.is_disposed());
            }
        }
    }
    // Failed completion releases the session lock and permits a clean session.
    TestDevice::with_checks(DeviceChecks {
        validation: false,
        disposal: true,
    })?
    .finish();
    Ok(())
}

#[test]
#[ignore = "requires Vulkan acceleration structure support"]
fn vulkan_acceleration_structure_commands() -> Result<(), DriverError> {
    use {
        ash::vk,
        vk_graph::{
            Graph,
            cmd::AccelerationStructureBuildGeometryInfo,
            driver::{
                accel_struct::{
                    AccelerationStructure, AccelerationStructureGeometry,
                    AccelerationStructureGeometryData, AccelerationStructureInfo,
                },
                buffer::{Buffer, BufferInfo},
                compute::{ComputePipeline, ComputePipelineInfo},
                shader::Shader,
            },
            pool::hash::HashPool,
        },
        vk_sync::AccessType,
    };

    let device = TestDevice::new()?;
    let Some(support) = device
        .physical
        .vk_khr_acceleration_structure
        .as_ref()
        .filter(|support| support.features.acceleration_structure)
    else {
        eprintln!(
            "SKIP: acceleration structure extension/feature unavailable; 0 AS commands executed"
        );

        return Ok(());
    };

    let Some(queue_family) = device.physical.queue_families.iter().position(|queue| {
        queue.queue_count > 0 && queue.queue_flags.contains(vk::QueueFlags::COMPUTE)
    }) else {
        eprintln!("SKIP: no compute-capable queue; 0 AS commands executed");

        return Ok(());
    };

    let alignment = support.properties.min_accel_struct_scratch_offset_alignment as u64;
    let mut graph = Graph::default();
    const VERTICES: [[f32; 3]; 3] = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
    let data = bytemuck::cast_slice(&VERTICES);
    let mut input = Buffer::create(
        &device,
        BufferInfo::host_mem(
            data.len() as _,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        ),
    )?;
    input.copy_from_slice(0, data);
    let input = graph.bind_resource(input);
    let geometries = [AccelerationStructureGeometry::opaque(
        AccelerationStructureGeometryData::triangles(
            0,
            vk::IndexType::NONE_KHR,
            2,
            0,
            graph.resource(input).device_address(),
            vk::Format::R32G32B32_SFLOAT,
            12,
        ),
    )];
    let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1)];
    let flags = vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE;
    let ty = vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL;

    // SAFETY: One non-indexed triangle, no transform or micromap; identical metadata
    // and primitive count are used for every build and update below.
    let sizes = unsafe {
        AccelerationStructure::build_sizes(
            &device,
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            ty,
            flags,
            &geometries,
            &[1],
        )
    };

    let mut outputs = Vec::new();

    for _ in 0..3 {
        outputs.push(graph.bind_resource(AccelerationStructure::create(
            &device,
            AccelerationStructureInfo::blas(sizes.acceleration_structure_size),
        )?));
    }

    let (a, b, c) = (outputs[0], outputs[1], outputs[2]);
    let mut scratch = Vec::new();

    for _ in 0..2 {
        let node = graph.bind_resource(Buffer::create(
            &device,
            BufferInfo::device_mem(
                sizes
                    .build_scratch_size
                    .max(sizes.update_scratch_size)
                    .max(1),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .into_builder()
            .alignment(alignment),
        )?);
        let address = graph.resource(node).device_address();

        assert_ne!(address, 0);
        assert_eq!(address % alignment, 0);

        scratch.push((node, address));
    }

    let [(scratch_a, address_a), (scratch_b, address_b)] = scratch[..] else {
        unreachable!();
    };

    assert_ne!(address_a, address_b);

    graph
        .begin_cmd()
        .debug_name("AS initial build")
        .resource_access(input, AccessType::AccelerationStructureBuildInputRead)
        .resource_access(a, AccessType::AccelerationStructureBuildWrite)
        .resource_access(
            scratch_a,
            AccessType::AccelerationStructureBuildScratchReadWrite,
        )
        .record_cmd(move |cmd| {
            // SAFETY: Graph-owned input, output and aligned scratch live through execution; the
            // range fits the static vertices and allocations match the DEVICE query.
            unsafe {
                cmd.build_acceleration_structures(
                    &[AccelerationStructureBuildGeometryInfo::build(
                        ty,
                        flags,
                        a,
                        &geometries,
                        address_a,
                    )],
                    &[&ranges],
                );
            }
        });
    graph
        .begin_cmd()
        .debug_name("AS mixed in-place update and build")
        .resource_access(input, AccessType::AccelerationStructureBuildInputRead)
        .resource_access(a, AccessType::AccelerationStructureBuildRead)
        .resource_access(a, AccessType::AccelerationStructureBuildWrite)
        .resource_access(b, AccessType::AccelerationStructureBuildWrite)
        .resource_access(
            scratch_a,
            AccessType::AccelerationStructureBuildScratchReadWrite,
        )
        .resource_access(
            scratch_b,
            AccessType::AccelerationStructureBuildScratchReadWrite,
        )
        .record_cmd(move |cmd| {
            // SAFETY: A's prior ALLOW_UPDATE build is synchronized. A and B have
            // separate storage and scratch; unchanged geometry and flags match the query.
            unsafe {
                cmd.build_acceleration_structures(
                    &[
                        AccelerationStructureBuildGeometryInfo::update(
                            ty,
                            flags,
                            a,
                            a,
                            &geometries,
                            address_a,
                        ),
                        AccelerationStructureBuildGeometryInfo::build(
                            ty,
                            flags,
                            b,
                            &geometries,
                            address_b,
                        ),
                    ],
                    &[&ranges, &ranges],
                );
            }
        });
    graph
        .begin_cmd()
        .debug_name("AS out-of-place update")
        .resource_access(input, AccessType::AccelerationStructureBuildInputRead)
        .resource_access(b, AccessType::AccelerationStructureBuildRead)
        .resource_access(c, AccessType::AccelerationStructureBuildWrite)
        .resource_access(
            scratch_a,
            AccessType::AccelerationStructureBuildScratchReadWrite,
        )
        .record_cmd(move |cmd| {
            // SAFETY: B's build and scratch reuse are synchronized. C is a separate
            // allocation sized by the same query; all update metadata is unchanged.
            unsafe {
                cmd.build_acceleration_structures(
                    &[AccelerationStructureBuildGeometryInfo::update(
                        ty,
                        flags,
                        b,
                        c,
                        &geometries,
                        address_a,
                    )],
                    &[&ranges],
                );
            }
        });

    let indirect_supported = support.features.acceleration_structure_indirect_build;

    if indirect_supported {
        let stride = size_of::<vk::AccelerationStructureBuildRangeInfoKHR>() as u32;

        assert_eq!(stride, 16);

        let indirect = Buffer::create(
            &device,
            BufferInfo::device_mem(
                stride as _,
                vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .into_builder()
            .alignment(4),
        )?;
        let indirect = graph.bind_resource(indirect);
        let indirect_address = graph.resource(indirect).device_address();

        assert_ne!(indirect_address, 0);
        assert_eq!(indirect_address % 4, 0);

        graph
            .begin_cmd()
            .debug_name("compute write AS indirect ranges")
            .bind_pipeline(ComputePipeline::create(
                &device,
                ComputePipelineInfo::default(),
                Shader::new_compute(
                    vk_shader_macros::glsl!(
                        r#"
                        #version 460 core
                        #pragma shader_stage(compute)
                        layout(local_size_x = 1) in;
                        layout(binding = 0, std430) writeonly buffer Ranges { uvec4 range; };
                        void main() { range = uvec4(1, 0, 0, 0); }
                        "#
                    )
                    .as_slice(),
                ),
            )?)
            .shader_resource_access(0, indirect, AccessType::ComputeShaderWrite)
            .record_cmd(|cmd| {
                cmd.dispatch(1, 1, 1);
            });

        graph
            .begin_cmd()
            .debug_name("AS indirect rebuild")
            .resource_access(input, AccessType::AccelerationStructureBuildInputRead)
            .resource_access(indirect, AccessType::AccelerationStructureBuildIndirectRead)
            .resource_access(c, AccessType::AccelerationStructureBuildWrite)
            .resource_access(
                scratch_a,
                AccessType::AccelerationStructureBuildScratchReadWrite,
            )
            .record_cmd(move |cmd| {
                // SAFETY: The feature is supported; graph-owned, compute-written range bytes
                // describe one triangle within the queried maximum. C and scratch reuse
                // are synchronized after the direct update; BUILD has no source.
                unsafe {
                    cmd.build_acceleration_structures_indirect(
                        &[AccelerationStructureBuildGeometryInfo::build(
                            ty,
                            flags,
                            c,
                            &geometries,
                            address_a,
                        )],
                        &[indirect_address],
                        &[stride],
                        &[&[1]],
                    );
                }
            });
    } else {
        eprintln!(
            "SKIP indirect AS build: acceleration_structure_indirect_build unsupported; 0 indirect commands executed"
        );
    }

    let mut pool = HashPool::new(&device);
    let mut fence = graph
        .finalize()
        .queue_submit(&mut pool, queue_family as _, 0)?;
    fence.wait()?;

    assert!(fence.status()?);
    eprintln!(
        "Executed 3 direct AS commands: 2 BUILD + 2 UPDATE (1 in-place, 1 out-of-place), including 1 mixed batch; fence signaled"
    );

    if indirect_supported {
        eprintln!(
            "Executed compute write -> range buffer -> indirect AS BUILD with AccelerationStructureBuildIndirectRead; total 4 AS commands, 3 BUILD + 2 UPDATE entries; fence signaled"
        );
    }

    drop(fence);
    drop(pool);
    device.assert_valid();
    Ok(())
}
