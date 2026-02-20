mod profile_with_puffin;

use {
    bytemuck::cast_slice,
    clap::Parser,
    log::warn,
    std::{io::BufReader, mem::size_of, sync::Arc},
    tobj::{GPU_LOAD_OPTIONS, load_mtl_buf, load_obj_buf},
    vk_graph_prelude::*,
    vk_graph_window::WindowBuilder,
    vk_shader_macros::glsl,
    winit::{event::Event, keyboard::KeyCode},
    winit_input_helper::WinitInputHelper,
};

static SHADER_RAY_GEN: &[u32] = glsl!(
    target: vulkan1_2,
    r#"
    #version 460
    #extension GL_EXT_ray_tracing : require
    #pragma shader_stage(raygen)

    #define M_PI 3.1415926535897932384626433832795

    layout(location = 0) rayPayloadEXT Payload {
        vec3 rayOrigin;
        vec3 rayDirection;
        vec3 previousNormal;

        vec3 directColor;
        vec3 indirectColor;
        int rayDepth;

        int rayActive;
    } payload;

    layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
    layout(binding = 1, set = 0) uniform Camera {
        vec4 position;
        vec4 right;
        vec4 up;
        vec4 forward;

        uint frameCount;
    } camera;

    layout(binding = 4, set = 0, rgba32f) uniform image2D image;

    float random(vec2 uv, float seed) {
        return fract(sin(mod(dot(uv, vec2(12.9898, 78.233)) + 1113.1 * seed, M_PI)) *
            43758.5453);
    }

    void main() {
        vec2 uv = gl_LaunchIDEXT.xy
                + vec2(random(gl_LaunchIDEXT.xy, 0), random(gl_LaunchIDEXT.xy, 1));
        uv /= vec2(gl_LaunchSizeEXT.xy);
        uv = (uv * 2.0f - 1.0f) * vec2(1.0f, -1.0f);

        payload.rayOrigin = camera.position.xyz;
        payload.rayDirection =
            normalize(uv.x * camera.right + uv.y * camera.up + camera.forward).xyz;
        payload.previousNormal = vec3(0.0, 0.0, 0.0);

        payload.directColor = vec3(0.0, 0.0, 0.0);
        payload.indirectColor = vec3(0.0, 0.0, 0.0);
        payload.rayDepth = 0;

        payload.rayActive = 1;

        for (int x = 0; x < 16; x++) {
            traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0,
                payload.rayOrigin, 0.001, payload.rayDirection, 10000.0, 0);
        }

        vec4 color = vec4(payload.directColor + payload.indirectColor, 1.0);

        if (camera.frameCount > 0) {
            vec4 previousColor = imageLoad(image, ivec2(gl_LaunchIDEXT.xy));
            previousColor *= camera.frameCount;

            color += previousColor;
            color /= (camera.frameCount + 1);
        }

        imageStore(image, ivec2(gl_LaunchIDEXT.xy), color);
    }
    "#
)
.as_slice();

static SHADER_CLOSEST_HIT: &[u32] = glsl!(
    target: vulkan1_2,
    r#"
    #version 460
    #extension GL_EXT_ray_tracing : require
    #extension GL_EXT_nonuniform_qualifier : enable
    #pragma shader_stage(closest)

    #define M_PI 3.1415926535897932384626433832795

    struct Material {
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
        vec3 emission;
    };

    hitAttributeEXT vec2 hitCoordinate;

    layout(location = 0) rayPayloadInEXT Payload {
        vec3 rayOrigin;
        vec3 rayDirection;
        vec3 previousNormal;

        vec3 directColor;
        vec3 indirectColor;
        int rayDepth;

        int rayActive;
    } payload;

    layout(location = 1) rayPayloadEXT bool isShadow;

    layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
    layout(binding = 1, set = 0) uniform Camera {
        vec4 position;
        vec4 right;
        vec4 up;
        vec4 forward;

        uint frameCount;
    } camera;

    layout(binding = 2, set = 0) buffer IndexBuffer {
        uint data[];
    } indexBuffer;
    layout(binding = 3, set = 0) buffer VertexBuffer {
        float data[];
    } vertexBuffer;

    layout(binding = 5, set = 0) buffer MaterialIndexBuffer {
        uint data[];
    } materialIndexBuffer;
    layout(binding = 6, set = 0) buffer MaterialBuffer {
        Material data[];
    } materialBuffer;

    float random(vec2 uv, float seed) {
        return fract(sin(mod(dot(uv, vec2(12.9898, 78.233)) + 1113.1 * seed, M_PI)) *
            43758.5453);
    }

    vec3 uniformSampleHemisphere(vec2 uv) {
        float z = uv.x;
        float r = sqrt(max(0, 1.0 - z * z));
        float phi = 2.0 * M_PI * uv.y;

        return vec3(r * cos(phi), z, r * sin(phi));
    }

    vec3 alignHemisphereWithCoordinateSystem(vec3 hemisphere, vec3 up) {
        vec3 right = normalize(cross(up, vec3(0.0072f, 1.0f, 0.0034f)));
        vec3 forward = cross(right, up);

        return hemisphere.x * right + hemisphere.y * up + hemisphere.z * forward;
    }

    void main() {
        if (payload.rayActive == 0) {
            return;
        }

        ivec3 indices = ivec3(indexBuffer.data[3 * gl_PrimitiveID + 0],
                              indexBuffer.data[3 * gl_PrimitiveID + 1],
                              indexBuffer.data[3 * gl_PrimitiveID + 2]);

        vec3 barycentric = vec3(1.0 - hitCoordinate.x - hitCoordinate.y,
                                hitCoordinate.x,
                                hitCoordinate.y);

        vec3 vertexA = vec3(vertexBuffer.data[3 * indices.x + 0],
                            vertexBuffer.data[3 * indices.x + 1],
                            vertexBuffer.data[3 * indices.x + 2]);
        vec3 vertexB = vec3(vertexBuffer.data[3 * indices.y + 0],
                            vertexBuffer.data[3 * indices.y + 1],
                            vertexBuffer.data[3 * indices.y + 2]);
        vec3 vertexC = vec3(vertexBuffer.data[3 * indices.z + 0],
                            vertexBuffer.data[3 * indices.z + 1],
                            vertexBuffer.data[3 * indices.z + 2]);

        vec3 position = vertexA * barycentric.x
                      + vertexB * barycentric.y
                      + vertexC * barycentric.z;
        vec3 geometricNormal = normalize(cross(vertexB - vertexA, vertexC - vertexA));

        vec3 surfaceColor =
            materialBuffer.data[materialIndexBuffer.data[gl_PrimitiveID]].diffuse;

        if (gl_PrimitiveID == 40 || gl_PrimitiveID == 41) {
            if (payload.rayDepth == 0) {
                payload.directColor =
                    materialBuffer.data[materialIndexBuffer.data[gl_PrimitiveID]].emission;
            } else {
                payload.indirectColor += (1.0 / payload.rayDepth)
                    * materialBuffer.data[materialIndexBuffer.data[gl_PrimitiveID]].emission
                    * dot(payload.previousNormal, payload.rayDirection);
            }
        } else {
            int randomIndex =
                int(random(gl_LaunchIDEXT.xy, camera.frameCount) * 2 + 40);
            vec3 lightColor = vec3(0.6, 0.6, 0.6);

            ivec3 lightIndices = ivec3(indexBuffer.data[3 * randomIndex + 0],
                                       indexBuffer.data[3 * randomIndex + 1],
                                       indexBuffer.data[3 * randomIndex + 2]);

            vec3 lightVertexA = vec3(vertexBuffer.data[3 * lightIndices.x + 0],
                                     vertexBuffer.data[3 * lightIndices.x + 1],
                                     vertexBuffer.data[3 * lightIndices.x + 2]);
            vec3 lightVertexB = vec3(vertexBuffer.data[3 * lightIndices.y + 0],
                                     vertexBuffer.data[3 * lightIndices.y + 1],
                                     vertexBuffer.data[3 * lightIndices.y + 2]);
            vec3 lightVertexC = vec3(vertexBuffer.data[3 * lightIndices.z + 0],
                                     vertexBuffer.data[3 * lightIndices.z + 1],
                                     vertexBuffer.data[3 * lightIndices.z + 2]);

            vec2 uv = vec2(random(gl_LaunchIDEXT.xy, camera.frameCount),
                           random(gl_LaunchIDEXT.xy, camera.frameCount + 1));
            if (uv.x + uv.y > 1.0f) {
                uv.x = 1.0f - uv.x;
                uv.y = 1.0f - uv.y;
            }

            vec3 lightBarycentric = vec3(1.0 - uv.x - uv.y, uv.x, uv.y);
            vec3 lightPosition = lightVertexA * lightBarycentric.x
                               + lightVertexB * lightBarycentric.y
                               + lightVertexC * lightBarycentric.z;

            vec3 positionToLightDirection = normalize(lightPosition - position);

            vec3 shadowRayOrigin = position;
            vec3 shadowRayDirection = positionToLightDirection;
            float shadowRayDistance = length(lightPosition - position) - 0.001f;

            uint shadowRayFlags = gl_RayFlagsTerminateOnFirstHitEXT
                                | gl_RayFlagsOpaqueEXT
                                | gl_RayFlagsSkipClosestHitShaderEXT;

            isShadow = true;
            traceRayEXT(topLevelAS, shadowRayFlags, 0xFF, 0, 0, 1, shadowRayOrigin,
                        0.001, shadowRayDirection, shadowRayDistance, 1);

            if (!isShadow) {
                if (payload.rayDepth == 0) {
                    payload.directColor = surfaceColor * lightColor
                                        * dot(geometricNormal, positionToLightDirection);
                } else {
                    payload.indirectColor +=
                        (1.0 / payload.rayDepth) * surfaceColor * lightColor *
                        dot(payload.previousNormal, payload.rayDirection) *
                        dot(geometricNormal, positionToLightDirection);
                }
            } else {
                if (payload.rayDepth == 0) {
                    payload.directColor = vec3(0.0, 0.0, 0.0);
                } else {
                    payload.rayActive = 0;
                }
            }
        }

        vec3 hemisphere = uniformSampleHemisphere(vec2(
            random(gl_LaunchIDEXT.xy, camera.frameCount),
            random(gl_LaunchIDEXT.xy, camera.frameCount + 1)
        ));
        vec3 alignedHemisphere =
            alignHemisphereWithCoordinateSystem(hemisphere, geometricNormal);

        payload.rayOrigin = position;
        payload.rayDirection = alignedHemisphere;
        payload.previousNormal = geometricNormal;

        payload.rayDepth += 1;
    }
    "#
)
.as_slice();

static SHADER_MISS: &[u32] = glsl!(
    target: vulkan1_2,
    r#"
    #version 460
    #extension GL_EXT_ray_tracing : require
    #pragma shader_stage(miss)

    layout(location = 0) rayPayloadInEXT Payload {
        vec3 rayOrigin;
        vec3 rayDirection;
        vec3 previousNormal;

        vec3 directColor;
        vec3 indirectColor;
        int rayDepth;

        int rayActive;
    } payload;

    void main() {
        payload.rayActive = 0;
    }
    "#
)
.as_slice();

static SHADER_SHADOW_MISS: &[u32] = glsl!(
    target: vulkan1_2,
    r#"
    #version 460
    #extension GL_EXT_ray_tracing : require
    #pragma shader_stage(miss)

    layout(location = 1) rayPayloadInEXT bool isShadow;

    void main() {
        isShadow = false;
    }
    "#
)
.as_slice();

fn create_ray_trace_pipeline(device: &Device) -> Result<Arc<RayTracePipeline>, DriverError> {
    Ok(Arc::new(RayTracePipeline::create(
        device,
        RayTracePipelineInfoBuilder::default().max_ray_recursion_depth(1),
        [
            Shader::new_ray_gen(SHADER_RAY_GEN),
            Shader::new_closest_hit(SHADER_CLOSEST_HIT),
            Shader::new_miss(SHADER_MISS),
            Shader::new_miss(SHADER_SHADOW_MISS),
        ],
        [
            RayTraceShaderGroup::new_general(0),
            RayTraceShaderGroup::new_triangles(1, None),
            RayTraceShaderGroup::new_general(2),
            RayTraceShaderGroup::new_general(3),
        ],
    )?))
}

#[allow(clippy::type_complexity)]
fn load_scene_buffers(
    device: &Device,
) -> Result<(Arc<Buffer>, Arc<Buffer>, u32, u32, Arc<Buffer>, Arc<Buffer>), DriverError> {
    use std::slice::from_raw_parts;

    let (models, materials, ..) = load_obj_buf(
        &mut BufReader::new(include_bytes!("res/cube_scene.obj").as_slice()),
        &GPU_LOAD_OPTIONS,
        |_| {
            load_mtl_buf(&mut BufReader::new(
                include_bytes!("res/cube_scene.mtl").as_slice(),
            ))
        },
    )
    .map_err(|err| {
        warn!("{err}");

        DriverError::InvalidData
    })?;
    let materials = materials.map_err(|err| {
        warn!("{err}");

        DriverError::InvalidData
    })?;

    let mut indices = vec![];
    let mut positions = vec![];
    for model in &models {
        let base_index = positions.len() as u32 / 3;
        for index in &model.mesh.indices {
            indices.push(*index + base_index);
        }

        for position in &model.mesh.positions {
            positions.push(*position);
        }
    }

    let index_buf = {
        let data = cast_slice(&indices);
        let mut buf = Buffer::create(
            device,
            BufferInfo::host_mem(
                data.len() as _,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            ),
        )?;
        Buffer::copy_from_slice(&mut buf, 0, data);
        buf
    };

    let vertex_buf = {
        let data = cast_slice(&positions);
        let mut buf = Buffer::create(
            device,
            BufferInfo::host_mem(
                data.len() as _,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            ),
        )?;
        Buffer::copy_from_slice(&mut buf, 0, data);
        buf
    };

    let material_id_buf = {
        let mut material_ids = vec![];
        for model in &models {
            for _ in 0..model.mesh.indices.len() / 3 {
                material_ids.push(model.mesh.material_id.unwrap() as u32);
            }
        }
        let data = cast_slice(&material_ids);
        let mut buf = Buffer::create(
            device,
            BufferInfo::host_mem(data.len() as _, vk::BufferUsageFlags::STORAGE_BUFFER),
        )?;
        Buffer::copy_from_slice(&mut buf, 0, data);
        buf
    };

    let material_buf = {
        let materials = materials
            .iter()
            .map(|material| {
                let ambient = material.ambient.unwrap_or_default();
                let diffuse = material.diffuse.unwrap_or([1.0, 0.0, 1.0]);
                let specular = material.specular.unwrap_or_default();

                [
                    ambient[0],
                    ambient[1],
                    ambient[2],
                    0.0,
                    diffuse[0],
                    diffuse[1],
                    diffuse[2],
                    0.0,
                    specular[0],
                    specular[1],
                    specular[2],
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                ]
            })
            .collect::<Box<[_]>>();
        let buf_len = materials.len() * 64;
        let mut buf = Buffer::create(
            device,
            BufferInfo::host_mem(buf_len as _, vk::BufferUsageFlags::STORAGE_BUFFER),
        )?;
        Buffer::copy_from_slice(&mut buf, 0, unsafe {
            from_raw_parts(materials.as_ptr() as *const _, buf_len)
        });
        buf
    };

    Ok((
        Arc::new(index_buf),
        Arc::new(vertex_buf),
        indices.len() as u32 / 3,
        positions.len() as u32 / 3,
        Arc::new(material_id_buf),
        Arc::new(material_buf),
    ))
}

/// Adapted from http://williamlewww.com/showcase_website/vk_khr_ray_tracing_tutorial/index.html
fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    let args = Args::parse();
    let window = WindowBuilder::default().debug(args.debug).build()?;
    let mut cache = HashPool::new(&window.device);

    // ------------------------------------------------------------------------------------------ //
    // Setup the ray tracing pipeline
    // ------------------------------------------------------------------------------------------ //

    let &RayTraceProperties {
        shader_group_base_alignment,
        shader_group_handle_alignment,
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
    let sbt_miss_size =
        2 * shader_group_handle_size.next_multiple_of(shader_group_handle_alignment);
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
        let miss_shadow_handle = RayTracePipeline::group_handle(&ray_trace_pipeline, 3);
        let sbt_miss_shadow_start = sbt_miss_start + shader_group_handle_alignment;
        data[sbt_miss_shadow_start as usize
            ..sbt_miss_shadow_start as usize + miss_shadow_handle.len()]
            .copy_from_slice(miss_shadow_handle);

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
    // Load the .obj cube scene
    // ------------------------------------------------------------------------------------------ //

    let (index_buf, vertex_buf, triangle_count, vertex_count, material_id_buf, material_buf) =
        load_scene_buffers(&window.device)?;

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
            let instance_node = render_graph.bind_node(&instance_buf);
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

        render_graph.resolve().submit(&mut cache, 0, 0)?;
    }

    // ------------------------------------------------------------------------------------------ //
    // Setup some state variables to hold between frames
    // ------------------------------------------------------------------------------------------ //

    let mut frame_count = 0;
    let mut image = None;
    let mut input = WinitInputHelper::default();
    let mut position = [1.391_760_3, 3.519_997_4, 5.598_739_6, 1f32];
    let right = [0.999_987_5_f32, 0.00000000, -0.004_999_064_4, 1.00000000];
    let up = [0f32, 1.0, 0.0, 1.0];
    let forward = [-0.004_999_064_4_f32, 0.00000000, -0.999_987_5, 1.00000000];

    // The event loop consists of:
    // - Lazy-init the storage image used to accumulate light
    // - Handle input
    // - Update the camera uniform buffer
    // - Trace the image
    // - Copy image to the swapchain
    window.run(|frame| {
        if image.is_none() {
            image = Some(Arc::new(
                cache
                    .lease(ImageInfo::image_2d(
                        frame.width,
                        frame.height,
                        frame.render_graph.node_info(frame.swapchain_image).fmt,
                        vk::ImageUsageFlags::STORAGE
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::TRANSFER_SRC,
                    ))
                    .unwrap(),
            ));
        }

        let image_node = frame.render_graph.bind_node(image.as_ref().unwrap());

        {
            input.step_with_window_events(
                &frame
                    .events
                    .iter()
                    .filter_map(|event| {
                        if let Event::WindowEvent { event, .. } = event {
                            Some(event.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Box<_>>(),
            );

            const SPEED: f32 = 0.1f32;

            if input.key_pressed(KeyCode::ArrowLeft) {
                frame_count = 0;
                position[0] -= SPEED;
            } else if input.key_pressed(KeyCode::ArrowRight) {
                frame_count = 0;
                position[0] += SPEED;
            } else if input.key_pressed(KeyCode::ArrowUp) {
                frame_count = 0;
                position[2] -= SPEED;
            } else if input.key_pressed(KeyCode::ArrowDown) {
                frame_count = 0;
                position[2] += SPEED;
            } else if input.key_pressed(KeyCode::Space) {
                frame_count = 0;
                position[1] -= SPEED;
            } else if input.key_pressed(KeyCode::AltLeft) {
                frame_count = 0;
                position[1] += SPEED;
            }

            if input.key_pressed(KeyCode::Escape) {
                frame_count = 0;
                frame.render_graph.clear_color_image(image_node, [0f32; 4]);
            } else {
                frame_count += 1;
            }
        }

        let camera_buf = frame.render_graph.bind_node({
            #[repr(C)]
            struct Camera {
                position: [f32; 4],
                right: [f32; 4],
                up: [f32; 4],
                forward: [f32; 4],
                frame_count: u32,
            }

            let mut buf = cache
                .lease(BufferInfo::host_mem(
                    size_of::<Camera>() as _,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                ))
                .unwrap();
            Buffer::copy_from_slice(&mut buf, 0, unsafe {
                std::slice::from_raw_parts(
                    &Camera {
                        position,
                        right,
                        up,
                        forward,
                        frame_count,
                    } as *const _ as *const _,
                    size_of::<Camera>(),
                )
            });

            buf
        });
        let blas_node = frame.render_graph.bind_node(&blas);
        let tlas_node = frame.render_graph.bind_node(&tlas);
        let index_buf_node = frame.render_graph.bind_node(&index_buf);
        let vertex_buf_node = frame.render_graph.bind_node(&vertex_buf);
        let material_id_buf_node = frame.render_graph.bind_node(&material_id_buf);
        let material_buf_node = frame.render_graph.bind_node(&material_buf);
        let sbt_node = frame.render_graph.bind_node(&sbt_buf);

        frame
            .render_graph
            .begin_cmd()
            .with_name("basic ray tracer")
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
            .access_descriptor(1, camera_buf, AccessType::RayTracingShaderReadOther)
            .access_descriptor(2, index_buf_node, AccessType::RayTracingShaderReadOther)
            .access_descriptor(3, vertex_buf_node, AccessType::RayTracingShaderReadOther)
            .write_descriptor(4, image_node)
            .access_descriptor(
                5,
                material_id_buf_node,
                AccessType::RayTracingShaderReadOther,
            )
            .access_descriptor(6, material_buf_node, AccessType::RayTracingShaderReadOther)
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
            .end_cmd()
            .copy_image(image_node, frame.swapchain_image);
    })?;

    Ok(())
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
