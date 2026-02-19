mod profile_with_puffin;

use {
    bytemuck::{NoUninit, cast_slice},
    clap::Parser,
    std::sync::Arc,
    vk_graph_prelude::*,
    vk_graph_window::WindowBuilder,
    vk_shader_macros::glsl,
};

static SHADER_RAY_GEN: &[u32] = glsl!(
    target: vulkan1_2,
    r#"
    #version 460
    #extension GL_EXT_ray_tracing : enable
    #pragma shader_stage(raygen)
    
    layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
    layout(binding = 1, set = 0, rgba32f) uniform image2D image;
    
    layout(location = 0) rayPayloadEXT vec3 hitValue;
    
    void main() {
        const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
        const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
        vec2 d = inUV * 2.0 - 1.0;
    
        vec4 origin = vec4(d.x, d.y, -1,1);
        vec4 target = vec4(d.x, d.y, 1, 1) ;
        vec4 direction = vec4(normalize(target.xyz), 0) ;
    
        float tmin = 0.001;
        float tmax = 10000.0;
    
        traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, origin.xyz, tmin, direction.xyz, tmax, 0);
    
        imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(hitValue, 0.0));
    }
    "#
)
.as_slice();

static SHADER_CLOSEST_HIT: &[u32] = glsl!(
    target: vulkan1_2,
    r#"
    #version 460
    #extension GL_EXT_ray_tracing : enable
    #extension GL_EXT_nonuniform_qualifier : enable
    #pragma shader_stage(closest)
    
    layout(location = 0) rayPayloadInEXT vec3 resultColor;
    hitAttributeEXT vec2 attribs;
    
    void main() {
      const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
      resultColor = barycentricCoords;
    }
    "#
)
.as_slice();

static SHADER_MISS: &[u32] = glsl!(
    target: vulkan1_2,
    r#"
    #version 460
    #extension GL_EXT_ray_tracing : enable
    #pragma shader_stage(miss)
    
    layout(location = 0) rayPayloadInEXT vec3 hitValue;
    
    void main() {
        hitValue = vec3(0.0, 0.0, 0.2);
    }
    "#
)
.as_slice();

fn create_ray_trace_pipeline(device: &Arc<Device>) -> Result<Arc<RayTracePipeline>, DriverError> {
    Ok(Arc::new(RayTracePipeline::create(
        device,
        RayTracePipelineInfoBuilder::default().max_ray_recursion_depth(1),
        [
            Shader::new_ray_gen(SHADER_RAY_GEN),
            Shader::new_closest_hit(SHADER_CLOSEST_HIT),
            Shader::new_miss(SHADER_MISS),
        ],
        [
            RayTraceShaderGroup::new_general(0),
            RayTraceShaderGroup::new_triangles(1, None),
            RayTraceShaderGroup::new_general(2),
        ],
    )?))
}

/// Adapted from https://iorange.github.io/p01/HappyTriangle.html
fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    let args = Args::parse();
    let window = WindowBuilder::default().debug(args.debug).build()?;
    let mut pool = HashPool::new(&window.device);

    // ------------------------------------------------------------------------------------------ //
    // Setup the ray tracing pipeline
    // ------------------------------------------------------------------------------------------ //

    let &RayTraceProperties {
        shader_group_base_alignment,
        shader_group_handle_size,
        ..
    } = window
        .device
        .physical_device
        .ray_trace_properties
        .as_ref()
        .unwrap();
    let ray_trace_pipeline = create_ray_trace_pipeline(&window.device)?;

    // ------------------------------------------------------------------------------------------ //
    // Setup a shader binding table
    // ------------------------------------------------------------------------------------------ //

    let sbt_rgen_size = shader_group_handle_size;
    let sbt_hit_start = sbt_rgen_size.next_multiple_of(shader_group_base_alignment);
    let sbt_hit_size = shader_group_handle_size;
    let sbt_miss_start =
        (sbt_hit_start + sbt_hit_size).next_multiple_of(shader_group_base_alignment);
    let sbt_miss_size = shader_group_handle_size;
    let sbt_buf = Arc::new({
        let mut buf = Buffer::create(
            &window.device,
            BufferInfo::host_mem(
                (sbt_miss_start + sbt_miss_size) as _,
                vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .to_builder()
            .alignment(shader_group_base_alignment as _),
        )
        .unwrap();

        let data = Buffer::mapped_slice_mut(&mut buf);

        let rgen_handle = RayTracePipeline::group_handle(&ray_trace_pipeline, 0);
        data[0..rgen_handle.len()].copy_from_slice(rgen_handle);

        let hit_handle = RayTracePipeline::group_handle(&ray_trace_pipeline, 1);
        data[sbt_hit_start as usize..sbt_hit_start as usize + hit_handle.len()]
            .copy_from_slice(hit_handle);

        let miss_handle = RayTracePipeline::group_handle(&ray_trace_pipeline, 2);
        data[sbt_miss_start as usize..sbt_miss_start as usize + miss_handle.len()]
            .copy_from_slice(miss_handle);

        buf
    });
    let sbt_address = Buffer::device_address(&sbt_buf);
    let sbt_rgen = vk::StridedDeviceAddressRegionKHR {
        device_address: sbt_address,
        stride: shader_group_handle_size as _,
        size: sbt_rgen_size as _,
    };
    let sbt_hit = vk::StridedDeviceAddressRegionKHR {
        device_address: sbt_address + sbt_hit_start as vk::DeviceAddress,
        stride: shader_group_handle_size as _,
        size: sbt_hit_size as _,
    };
    let sbt_miss = vk::StridedDeviceAddressRegionKHR {
        device_address: sbt_address + sbt_miss_start as vk::DeviceAddress,
        stride: shader_group_handle_size as _,
        size: sbt_miss_size as _,
    };
    let sbt_callable = vk::StridedDeviceAddressRegionKHR::default();

    // ------------------------------------------------------------------------------------------ //
    // Generate the geometry and load it into buffers
    // ------------------------------------------------------------------------------------------ //

    let triangle_count = 1;
    let vertex_count = triangle_count * 3;

    #[repr(C)]
    #[derive(Debug, Clone, Copy, NoUninit)]
    #[allow(dead_code)]
    struct Vertex {
        pos: [f32; 3],
    }

    const VERTICES: [Vertex; 3] = [
        Vertex {
            pos: [-1.0, 1.0, 0.0],
        },
        Vertex {
            pos: [1.0, 1.0, 0.0],
        },
        Vertex {
            pos: [0.0, -1.0, 0.0],
        },
    ];

    const INDICES: [u32; 3] = [0, 1, 2];

    let index_buf = {
        let data = cast_slice(&INDICES);
        let mut buf = Buffer::create(
            &window.device,
            BufferInfo::host_mem(
                data.len() as _,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            ),
        )?;
        Buffer::copy_from_slice(&mut buf, 0, data);
        Arc::new(buf)
    };

    let vertex_buf = {
        let data = cast_slice(&VERTICES);
        let mut buf = Buffer::create(
            &window.device,
            BufferInfo::host_mem(
                data.len() as _,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            ),
        )?;
        Buffer::copy_from_slice(&mut buf, 0, data);
        Arc::new(buf)
    };

    // ------------------------------------------------------------------------------------------ //
    // Create the bottom level acceleration structure
    // ------------------------------------------------------------------------------------------ //

    let blas_geometry_info = AccelerationStructureGeometryInfo::blas([(
        AccelerationStructureGeometry::opaque(
            triangle_count,
            AccelerationStructureGeometryData::triangles(
                Buffer::device_address(&index_buf),
                vk::IndexType::UINT32,
                vertex_count,
                None,
                Buffer::device_address(&vertex_buf),
                vk::Format::R32G32B32_SFLOAT,
                12,
            ),
        ),
        vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(triangle_count),
    )]);
    let blas_size = AccelerationStructure::size_of(&window.device, &blas_geometry_info);
    let blas = Arc::new(AccelerationStructure::create(
        &window.device,
        AccelerationStructureInfo::blas(blas_size.create_size),
    )?);
    let blas_device_address = AccelerationStructure::device_address(&blas);

    // ------------------------------------------------------------------------------------------ //
    // Create an instance buffer, which is just one instance for the single BLAS
    // ------------------------------------------------------------------------------------------ //

    let instances = [vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR {
            matrix: [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
            ],
        },
        instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
            0,
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as _,
        ),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: blas_device_address,
        },
    }];
    let instance_data = AccelerationStructure::instance_slice(&instances);
    let instance_buf = Arc::new({
        let mut buffer = Buffer::create(
            &window.device,
            BufferInfo::host_mem(
                instance_data.len() as _,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            ),
        )?;
        Buffer::copy_from_slice(&mut buffer, 0, instance_data);

        buffer
    });

    // ------------------------------------------------------------------------------------------ //
    // Create the top level acceleration structure
    // ------------------------------------------------------------------------------------------ //

    let tlas_geometry_info = AccelerationStructureGeometryInfo::tlas([(
        AccelerationStructureGeometry::opaque(
            1,
            AccelerationStructureGeometryData::instances(Buffer::device_address(&instance_buf)),
        ),
        vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1),
    )]);
    let tlas_size = AccelerationStructure::size_of(&window.device, &tlas_geometry_info);
    let tlas = Arc::new(AccelerationStructure::create(
        &window.device,
        AccelerationStructureInfo::tlas(tlas_size.create_size),
    )?);

    // ------------------------------------------------------------------------------------------ //
    // Build the BLAS and TLAS; note that we don't drop the cache and so there is no CPU stall
    // ------------------------------------------------------------------------------------------ //

    {
        let accel_struct_scratch_offset_alignment = window
            .device
            .physical_device
            .accel_struct_properties
            .as_ref()
            .unwrap()
            .min_accel_struct_scratch_offset_alignment
            as vk::DeviceSize;
        let mut render_graph = Graph::default();
        let index_node = render_graph.bind_node(&index_buf);
        let vertex_node = render_graph.bind_node(&vertex_buf);
        let blas_node = render_graph.bind_node(&blas);

        {
            let scratch_buf = render_graph.bind_node(Buffer::create(
                &window.device,
                BufferInfo::device_mem(
                    blas_size.build_size,
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                )
                .to_builder()
                .alignment(accel_struct_scratch_offset_alignment),
            )?);
            let scratch_data = render_graph.node_device_address(scratch_buf);

            render_graph
                .begin_cmd()
                .with_name("Build BLAS")
                .access_node(index_node, AccessType::AccelerationStructureBuildRead)
                .access_node(vertex_node, AccessType::AccelerationStructureBuildRead)
                .access_node(scratch_buf, AccessType::AccelerationStructureBufferWrite)
                .access_node(blas_node, AccessType::AccelerationStructureBuildWrite)
                .record_acceleration(move |accel, _| {
                    accel.build_structure(&blas_geometry_info, blas_node, scratch_data);
                });
        }

        {
            let instance_node = render_graph.bind_node(instance_buf);
            let scratch_buf = render_graph.bind_node(Buffer::create(
                &window.device,
                BufferInfo::device_mem(
                    tlas_size.build_size,
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                )
                .to_builder()
                .alignment(accel_struct_scratch_offset_alignment),
            )?);
            let scratch_data = render_graph.node_device_address(scratch_buf);
            let tlas_node = render_graph.bind_node(&tlas);

            render_graph
                .begin_cmd()
                .with_name("Build TLAS")
                .access_node(blas_node, AccessType::AccelerationStructureBuildRead)
                .access_node(instance_node, AccessType::AccelerationStructureBuildRead)
                .access_node(scratch_buf, AccessType::AccelerationStructureBufferWrite)
                .access_node(tlas_node, AccessType::AccelerationStructureBuildWrite)
                .record_acceleration(move |accel, _| {
                    accel.build_structure(&tlas_geometry_info, tlas_node, scratch_data);
                });
        }

        render_graph.resolve().submit(&mut pool, 0, 0)?;
    }

    // ------------------------------------------------------------------------------------------ //
    // Setup some state variables to hold between frames
    // ------------------------------------------------------------------------------------------ //

    // The event loop consists of:
    // - Trace the image
    // - Copy image to the swapchain
    window.run(|frame| {
        let blas_node = frame.render_graph.bind_node(&blas);
        let tlas_node = frame.render_graph.bind_node(&tlas);
        let sbt_node = frame.render_graph.bind_node(&sbt_buf);

        frame
            .render_graph
            .begin_cmd()
            .with_name("ray-traced triangle")
            .bind_pipeline(&ray_trace_pipeline)
            .access_node(
                blas_node,
                AccessType::RayTracingShaderReadAccelerationStructure,
            )
            .access_node(sbt_node, AccessType::RayTracingShaderReadOther)
            .access_descriptor(
                0,
                tlas_node,
                AccessType::RayTracingShaderReadAccelerationStructure,
            )
            .write_descriptor(1, frame.swapchain_image)
            .record_pipeline(move |pipeline, _| {
                pipeline.trace_rays(
                    &sbt_rgen,
                    &sbt_miss,
                    &sbt_hit,
                    &sbt_callable,
                    frame.width,
                    frame.height,
                    1,
                );
            })
            .end_cmd();
    })?;

    Ok(())
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
