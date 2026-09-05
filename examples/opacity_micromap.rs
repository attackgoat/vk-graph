//! Headless opacity micromap and BLAS build example.
//!
//! This validates resource setup, graph synchronization, and the two device build commands. It
//! intentionally does not create a ray tracing pipeline or render an image.

use {
    ash::vk,
    std::{mem::size_of, slice},
    vk_graph::{
        Graph,
        cmd::{BuildAccelerationStructureInfo, BuildMicromapInfo},
        driver::{
            DriverError,
            accel_struct::{
                AccelerationStructure, AccelerationStructureGeometry,
                AccelerationStructureGeometryData, AccelerationStructureGeometryInfo,
                AccelerationStructureInfo, AccelerationStructureOpacityMicromap,
            },
            buffer::{Buffer, BufferInfo},
            device::{Device, DeviceInfo},
            micromap::{Micromap, MicromapInfo, OpacityMicromapBuildInfo, OpacityMicromapUsage},
        },
        pool::hash::HashPool,
    },
    vk_sync::AccessType,
};

fn main() -> Result<(), DriverError> {
    pretty_env_logger::init();

    let device = Device::create(DeviceInfo::default())?;
    let Some(extension) = &device.physical.vk_ext_opacity_micromap else {
        println!("VK_EXT_opacity_micromap is unsupported; skipping example");
        return Ok(());
    };
    if !extension.features.micromap || extension.properties.max_opacity2_state_subdivision_level < 1
    {
        println!(
            "two-state subdivision-level-1 opacity micromaps are unsupported; skipping example"
        );
        return Ok(());
    }
    let queue_family_index = device
        .physical
        .queue_families
        .iter()
        .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .map(|index| index as u32)
        .ok_or(DriverError::Unsupported)?;

    let micromap_usage =
        OpacityMicromapUsage::new(1, 1, vk::OpacityMicromapFormatEXT::TYPE_2_STATE);

    // Level 1 has four microtriangles. Two-state data uses one bit per microtriangle; false is
    // transparent and true is opaque.
    let opacity_data = encode_2_state([false, true, false, true]);
    let opacity_buffer = upload_micromap_input(&device, &opacity_data)?;

    let micromap_triangle = vk::MicromapTriangleEXT {
        data_offset: 0,
        subdivision_level: 1,
        format: vk::OpacityMicromapFormatEXT::TYPE_2_STATE.as_raw() as u16,
    };
    let triangle_buffer = upload_micromap_input(&device, bytes_of(&micromap_triangle))?;

    let micromap_build = OpacityMicromapBuildInfo::new([micromap_usage])
        .data(opacity_buffer.device_address())
        .triangle_array(
            triangle_buffer.device_address(),
            size_of::<vk::MicromapTriangleEXT>() as _,
        );
    let micromap_size = Micromap::size_of(&device, &micromap_build);
    let micromap = Micromap::create(&device, MicromapInfo::device_mem(micromap_size.create_size))?;

    let scratch_alignment = device
        .physical
        .vk_khr_acceleration_structure
        .as_ref()
        .expect("VK_EXT_opacity_micromap requires VK_KHR_acceleration_structure")
        .properties
        .min_accel_struct_scratch_offset_alignment as vk::DeviceSize;
    let micromap_scratch = Buffer::create(
        &device,
        BufferInfo::device_mem(
            micromap_size.build_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .into_builder()
        .alignment(scratch_alignment),
    )?;
    let micromap_build = micromap_build.scratch_data(micromap_scratch.device_address());

    let indices = [0_u32, 1, 2];
    let index_buffer = upload_acceleration_structure_input(&device, bytes_of_slice(&indices))?;
    let vertices = [[-1.0_f32, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, -1.0, 0.0]];
    let vertex_buffer = upload_acceleration_structure_input(&device, bytes_of_slice(&vertices))?;

    let geometry = AccelerationStructureGeometry::new(
        1,
        AccelerationStructureGeometryData::triangles(
            index_buffer.device_address(),
            vk::IndexType::UINT32,
            2,
            None,
            vertex_buffer.device_address(),
            vk::Format::R32G32B32_SFLOAT,
            size_of::<[f32; 3]>() as _,
        )
        .opacity_micromap(AccelerationStructureOpacityMicromap::new(
            micromap.handle,
            [micromap_usage],
        )),
    );
    let blas_build = AccelerationStructureGeometryInfo::blas([(
        geometry,
        vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1),
    )]);
    let blas_size = AccelerationStructure::size_of(&device, &blas_build);
    let blas = AccelerationStructure::create(
        &device,
        AccelerationStructureInfo::blas(blas_size.create_size),
    )?;
    let blas_scratch = Buffer::create(
        &device,
        BufferInfo::device_mem(
            blas_size.build_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .into_builder()
        .alignment(scratch_alignment),
    )?;

    let mut graph = Graph::default();
    let opacity_node = graph.bind_resource(opacity_buffer);
    let triangle_node = graph.bind_resource(triangle_buffer);
    let micromap_scratch_node = graph.bind_resource(micromap_scratch);
    let micromap_node = graph.bind_resource(micromap);
    let index_node = graph.bind_resource(index_buffer);
    let vertex_node = graph.bind_resource(vertex_buffer);
    let blas_scratch_node = graph.bind_resource(blas_scratch);
    let blas_node = graph.bind_resource(blas);

    graph
        .begin_cmd()
        .debug_name("build opacity micromap and BLAS")
        .resource_access(opacity_node, AccessType::MicromapBuildInputRead)
        .resource_access(triangle_node, AccessType::MicromapBuildInputRead)
        .resource_access(
            micromap_scratch_node,
            AccessType::MicromapBuildScratchReadWrite,
        )
        .resource_access(micromap_node, AccessType::MicromapBuildWrite)
        .record_cmd(move |cmd| {
            // Safety: The graph retains the separate, correctly sized and aligned buffers until
            // completion. Inputs match the usage record, remain unchanged, and all accesses are
            // declared above; the destination and scratch sizes come from the same build info.
            unsafe {
                cmd.build_micromaps(&[BuildMicromapInfo::new(micromap_node, micromap_build)]);
            }
        })
        .resource_access(index_node, AccessType::AccelerationStructureBuildInputRead)
        .resource_access(vertex_node, AccessType::AccelerationStructureBuildInputRead)
        .resource_access(
            micromap_node,
            AccessType::AccelerationStructureBuildMicromapRead,
        )
        .resource_access(
            blas_scratch_node,
            AccessType::AccelerationStructureBuildScratchReadWrite,
        )
        .resource_access(blas_node, AccessType::AccelerationStructureBuildWrite)
        .record_cmd(move |cmd| {
            cmd.build_accel_struct(&[BuildAccelerationStructureInfo::new(
                blas_node,
                cmd.resource(blas_scratch_node).device_address(),
                blas_build,
            )]);
        });

    let mut fence =
        graph
            .finalize()
            .queue_submit(&mut HashPool::new(&device), queue_family_index, 0)?;
    fence.wait()?;

    println!("built a two-state opacity micromap and attached BLAS");

    Ok(())
}

fn upload_micromap_input(device: &Device, data: &[u8]) -> Result<Buffer, DriverError> {
    let mut buffer = Buffer::create(
        device,
        BufferInfo::host_mem(
            data.len() as _,
            vk::BufferUsageFlags::MICROMAP_BUILD_INPUT_READ_ONLY_EXT
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )
        .into_builder()
        .alignment(256),
    )?;
    buffer.copy_from_slice(0, data);
    Ok(buffer)
}

fn encode_2_state(states: [bool; 4]) -> [u8; 1] {
    let mut packed = 0_u8;
    for (index, opaque) in states.into_iter().enumerate() {
        packed |= u8::from(opaque) << index;
    }
    [packed]
}

fn upload_acceleration_structure_input(
    device: &Device,
    data: &[u8],
) -> Result<Buffer, DriverError> {
    let mut buffer = Buffer::create(
        device,
        BufferInfo::host_mem(
            data.len() as _,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        ),
    )?;
    buffer.copy_from_slice(0, data);
    Ok(buffer)
}

fn bytes_of<T>(value: &T) -> &[u8] {
    unsafe { slice::from_raw_parts((value as *const T).cast(), size_of::<T>()) }
}

fn bytes_of_slice<T>(values: &[T]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}
