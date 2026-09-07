use std::{env, ffi::OsString, process::Command, thread, time::Duration};

const EXAMPLE_TIMEOUT: Duration = Duration::from_secs(30);

fn run_example(target_name: &str, extra_args: &[&str]) {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let mut command = Command::new(cargo);
    command.args([
        "run",
        "--quiet",
        "--example",
        target_name,
        "--no-default-features",
    ]);

    for (feature, enabled) in [
        ("checked", cfg!(feature = "checked")),
        ("loaded", cfg!(feature = "loaded")),
        ("linked", cfg!(feature = "linked")),
        ("ash-molten", cfg!(feature = "ash-molten")),
        ("parking_lot", cfg!(feature = "parking_lot")),
    ] {
        if enabled {
            command.args(["--features", feature]);
        }
    }

    let mut child = command
        .arg("--")
        .args(extra_args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .spawn()
        .unwrap_or_else(|err| panic!("unable to run {target_name} example: {err}"));

    let started = std::time::Instant::now();

    loop {
        if let Some(status) = child
            .try_wait()
            .unwrap_or_else(|err| panic!("unable to poll {target_name} example: {err}"))
        {
            assert!(
                status.success(),
                "run error: example `{target_name}` exited with status {:?}",
                status.code(),
            );

            return;
        }

        if started.elapsed() >= EXAMPLE_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();

            panic!("example `{target_name}` exceeded {EXAMPLE_TIMEOUT:?}");
        }

        thread::sleep(Duration::from_millis(100));
    }
}

#[test]
#[ignore = "requires Vulkan acceleration structure support"]
fn vulkan_acceleration_structure_commands() -> Result<(), vk_graph::driver::DriverError> {
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
                device::{Device, DeviceInfo},
                shader::Shader,
            },
            pool::hash::HashPool,
        },
        vk_sync::AccessType,
    };

    let _ = pretty_env_logger::try_init();
    let device = Device::create(DeviceInfo::default())?;
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

    Ok(())
}

#[test]
#[ignore = "requires Vulkan device"]
fn vulkan_cpu_readback() {
    run_example("cpu_readback", &[]);
}

#[test]
#[ignore = "requires Vulkan opacity micromap support"]
fn vulkan_opacity_micromap() {
    run_example("opacity_micromap", &[]);
}
