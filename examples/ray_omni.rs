mod profile_with_puffin;

use {
    ash::vk,
    bytemuck::{Pod, Zeroable, bytes_of, cast_slice},
    clap::Parser,
    glam::{Mat4, Vec3, Vec4, vec3, vec4},
    log::info,
    meshopt::remap::{generate_vertex_remap, remap_index_buffer, remap_vertex_buffer},
    std::{
        env::current_exe,
        fs::{metadata, write},
        mem::size_of,
        path::{Path, PathBuf},
        sync::Arc,
        time::Instant,
    },
    tobj::{GPU_LOAD_OPTIONS, load_obj},
    vk_graph::{
        Graph,
        cmd::{BuildAccelerationStructureInfo, LoadOp, StoreOp},
        driver::{
            DriverError,
            accel_struct::{
                AccelerationStructure, AccelerationStructureGeometry,
                AccelerationStructureGeometryData, AccelerationStructureGeometryInfo,
                AccelerationStructureInfo,
            },
            buffer::{Buffer, BufferInfo},
            device::Device,
            graphic::{DepthStencilInfo, GraphicPipeline, GraphicPipelineInfo},
            image::ImageInfo,
            shader::Shader,
        },
        node::AccelerationStructureLeaseNode,
        pool::{Pool as _, lazy::LazyPool},
    },
    vk_graph_window::Window,
    vk_shader_macros::glsl,
    vk_sync::AccessType,
};

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    let args = Args::parse();
    let window = Window::builder().debug(args.debug).build()?;
    let mut pool = LazyPool::new(&window.device);

    let depth_fmt = best_2d_optimal_format(
        &window.device,
        &[vk::Format::D32_SFLOAT, vk::Format::D16_UNORM],
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::ImageCreateFlags::empty(),
    );

    let ground_mesh = load_ground_mesh(&window.device)?;
    let model_path = download_model_from_github("happy.obj")?;
    let model_mesh = load_model_mesh(&window.device, model_path)?;
    let scene_blas = create_blas(&window.device, &[&ground_mesh, &model_mesh])?;
    let gfx_pipeline = create_pipeline(&window.device)?;

    let mut angle = 0f32;
    let mut prev_frame_at = Instant::now();

    window.run(|frame| {
        let now = Instant::now();
        let dt = now - prev_frame_at;
        prev_frame_at = now;

        angle += dt.as_secs_f32();

        let scene_tlas = create_tlas(frame.device, &mut pool, frame.graph, &scene_blas).unwrap();

        let ground_mesh_index_buf = frame.graph.bind_resource(&ground_mesh.index_buf);
        let ground_mesh_vertex_buf = frame.graph.bind_resource(&ground_mesh.vertex_buf);
        let model_mesh_index_buf = frame.graph.bind_resource(&model_mesh.index_buf);
        let model_mesh_vertex_buf = frame.graph.bind_resource(&model_mesh.vertex_buf);

        let depth_image = frame.graph.bind_resource(
            pool.lease_resource(ImageInfo::image_2d(
                frame.width,
                frame.height,
                depth_fmt,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ))
            .unwrap(),
        );
        let camera_buf = frame.graph.bind_resource({
            let mut buf = pool
                .lease_resource(BufferInfo::host_mem(
                    size_of::<Camera>() as _,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                ))
                .unwrap();
            buf.copy_from_slice(
                0,
                bytes_of(&Camera {
                    projection: Mat4::perspective_rh(
                        45f32.to_radians(),
                        frame.render_aspect_ratio(),
                        0.1,
                        100.0,
                    ),
                    view: Mat4::look_at_rh(vec3(0.0, 1.2, 1.0), vec3(0.0, 0.6, 0.0), -Vec3::Y),
                    model: Mat4::IDENTITY,
                    light_position: vec4(angle.cos() * 3.0, 2.0, angle.sin() * 3.0, 0.0),
                }),
            );

            buf
        });

        frame
            .graph
            .begin_cmd()
            .debug_name("Mesh with ray-query shadows")
            .bind_pipeline(&gfx_pipeline)
            .resource_access(ground_mesh_index_buf, AccessType::IndexBuffer)
            .resource_access(ground_mesh_vertex_buf, AccessType::VertexBuffer)
            .resource_access(model_mesh_index_buf, AccessType::IndexBuffer)
            .resource_access(model_mesh_vertex_buf, AccessType::VertexBuffer)
            .shader_resource_access(0, camera_buf, AccessType::AnyShaderReadUniformBuffer)
            .shader_resource_access(
                1,
                scene_tlas,
                AccessType::RayTracingShaderReadAccelerationStructure,
            )
            .depth_stencil(DepthStencilInfo::DEPTH_WRITE_LESS_IGNORE_STENCIL)
            .depth_stencil_attachment_image(
                depth_image,
                LoadOp::CLEAR_ZERO_STENCIL_ZERO,
                StoreOp::DontCare,
            )
            .color_attachment_image(
                0,
                frame.swapchain_image,
                LoadOp::CLEAR_WHITE_ALPHA_ONE,
                StoreOp::Store,
            )
            .record_cmd(move |cmd_buf| {
                cmd_buf
                    .bind_index_buffer(model_mesh_index_buf, 0, vk::IndexType::UINT32)
                    .bind_vertex_buffer(0, model_mesh_vertex_buf, 0)
                    .draw_indexed(model_mesh.index_count, 1, 0, 0, 0);

                cmd_buf
                    .bind_index_buffer(ground_mesh_index_buf, 0, vk::IndexType::UINT32)
                    .bind_vertex_buffer(0, ground_mesh_vertex_buf, 0)
                    .draw_indexed(ground_mesh.index_count, 1, 0, 0, 0);
            });
    })?;

    Ok(())
}

fn best_2d_optimal_format(
    device: &Device,
    formats: &[vk::Format],
    usage: vk::ImageUsageFlags,
    flags: vk::ImageCreateFlags,
) -> vk::Format {
    for format in formats {
        let format_props = device.physical_device.image_format_properties(
            *format,
            vk::ImageType::TYPE_2D,
            vk::ImageTiling::OPTIMAL,
            usage,
            flags,
        );

        if matches!(format_props, Ok(Some(_))) {
            return *format;
        }
    }

    panic!("Unsupported format");
}

fn create_blas(
    device: &Device,
    models: &[&Model],
) -> Result<Arc<AccelerationStructure>, DriverError> {
    let info = AccelerationStructureGeometryInfo::blas(
        models
            .iter()
            .map(|model| {
                (
                    AccelerationStructureGeometry {
                        max_primitive_count: model.index_count / 3,
                        flags: vk::GeometryFlagsKHR::OPAQUE,
                        geometry: AccelerationStructureGeometryData::triangles(
                            model.index_buf.device_address(),
                            vk::IndexType::UINT32,
                            model.vertex_count,
                            None,
                            model.vertex_buf.device_address(),
                            vk::Format::R32G32B32_SFLOAT,
                            24,
                        ),
                    },
                    vk::AccelerationStructureBuildRangeInfoKHR::default()
                        .primitive_count(model.index_count / 3),
                )
            })
            .collect::<Box<_>>(),
    )
    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE);
    let size = AccelerationStructure::size_of(device, &info);

    let mut graph = Graph::default();
    let blas = graph.bind_resource(AccelerationStructure::create(
        device,
        AccelerationStructureInfo::blas(size.create_size),
    )?);

    let accel_struct_scratch_offset_alignment = device
        .physical_device
        .accel_struct_properties
        .as_ref()
        .unwrap()
        .min_accel_struct_scratch_offset_alignment
        as vk::DeviceSize;
    let scratch_buf = graph.bind_resource(Buffer::create(
        device,
        BufferInfo::device_mem(
            size.build_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .into_builder()
        .alignment(accel_struct_scratch_offset_alignment),
    )?);
    let scratch_addr = graph.resource(scratch_buf).device_address();

    let mut pass = graph.begin_cmd().debug_name("Build BLAS");

    for model in models.iter().copied() {
        let index_buf = pass.bind_resource(&model.index_buf);
        let vertex_buf = pass.bind_resource(&model.vertex_buf);

        pass.set_resource_access(index_buf, AccessType::AccelerationStructureBuildRead);
        pass.set_resource_access(vertex_buf, AccessType::AccelerationStructureBuildRead);
    }

    pass.resource_access(blas, AccessType::AccelerationStructureBuildWrite)
        .resource_access(scratch_buf, AccessType::AccelerationStructureBufferWrite)
        .record_cmd(move |cmd_buf| {
            cmd_buf.build_accel_struct(&[BuildAccelerationStructureInfo::new(
                blas,
                scratch_addr,
                info,
            )]);
        });

    let blas = graph.resource(blas).clone();

    graph
        .into_submission()
        .queue_submit(&mut LazyPool::new(device), 0, 0)?;

    Ok(blas)
}

fn create_pipeline(device: &Device) -> Result<GraphicPipeline, DriverError> {
    let vert = glsl!(
        r#"
        #version 460 core
        #pragma shader_stage(vertex)

        layout (location = 0) in vec3 inPos;
        layout (location = 1) in vec3 inNormal;
        
        layout (binding = 0) uniform UBO 
        {
            mat4 projection;
            mat4 view;
            mat4 model;
            vec3 lightPos;
        } ubo;
        
        layout (location = 0) out vec3 outNormal;
        layout (location = 1) out vec3 outViewVec;
        layout (location = 2) out vec3 outLightVec;
        layout (location = 3) out vec3 outWorldPos;
        
        void main() 
        {
            outNormal = inNormal;
            gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPos.xyz, 1.0);
            vec4 pos = ubo.model * vec4(inPos, 1.0);
            outWorldPos = vec3(ubo.model * vec4(inPos, 1.0));
            outNormal = mat3(ubo.model) * inNormal;
            outLightVec = normalize(ubo.lightPos - inPos);
            outViewVec = -pos.xyz;
        }
        "#
    );
    let frag = glsl!(
        target: vulkan1_2,
        r#"
        #version 460 core
        #extension GL_EXT_ray_tracing : enable
        #extension GL_EXT_ray_query : enable
        #pragma shader_stage(fragment)

        layout (binding = 1) uniform accelerationStructureEXT topLevelAS;

        layout (location = 0) in vec3 inNormal;
        layout (location = 1) in vec3 inViewVec;
        layout (location = 2) in vec3 inLightVec;
        layout (location = 3) in vec3 inWorldPos;

        layout (location = 0) out vec4 outFragColor;

        #define ambient 0.1

        void main() 
        {	
            vec3 N = normalize(inNormal);
            vec3 L = normalize(inLightVec);
            vec3 V = normalize(inViewVec);
            vec3 R = normalize(-reflect(L, N));
            vec3 diffuse = vec3(max(dot(N, L), ambient));

            outFragColor = vec4(diffuse, 1.0);

            rayQueryEXT rayQuery;
            rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, inWorldPos, 0.01, L, 1000.0);

            // Traverse the acceleration structure and store information about the first intersection (if any)
            rayQueryProceedEXT(rayQuery);

            // If the intersection has hit a triangle, the fragment is shadowed
            if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT ) {
                outFragColor *= 0.1;
            }
        }
        "#
    );

    GraphicPipeline::create(
        device,
        GraphicPipelineInfo::default(),
        [
            Shader::new_vertex(vert.as_slice()),
            Shader::new_fragment(frag.as_slice()),
        ],
    )
}

fn create_tlas(
    device: &Device,
    pool: &mut LazyPool,
    graph: &mut Graph,
    blas: &Arc<AccelerationStructure>,
) -> Result<AccelerationStructureLeaseNode, DriverError> {
    let instances = [vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR {
            matrix: [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
            ],
        },
        instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
            0,
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as _,
        ),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: AccelerationStructure::device_address(blas),
        },
    }];
    let instance_data = AccelerationStructure::instance_slice(&instances);
    let instance_buf = Arc::new({
        let mut buffer = Buffer::create(
            device,
            BufferInfo::host_mem(
                instance_data.len() as _,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            ),
        )?;
        buffer.copy_from_slice(0, instance_data);

        buffer
    });

    let info = AccelerationStructureGeometryInfo::tlas([(
        AccelerationStructureGeometry::opaque(
            2,
            AccelerationStructureGeometryData::instances(instance_buf.device_address()),
        ),
        vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1),
    )])
    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE);
    let size = AccelerationStructure::size_of(device, &info);
    let tlas = graph
        .bind_resource(pool.lease_resource(AccelerationStructureInfo::tlas(size.create_size))?);

    let accel_struct_scratch_offset_alignment = device
        .physical_device
        .accel_struct_properties
        .as_ref()
        .unwrap()
        .min_accel_struct_scratch_offset_alignment
        as vk::DeviceSize;
    let scratch_buf = graph.bind_resource(
        pool.lease_resource(
            BufferInfo::device_mem(
                size.build_size,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .into_builder()
            .alignment(accel_struct_scratch_offset_alignment),
        )?,
    );
    let scratch_addr = graph.resource(scratch_buf).device_address();
    let blas = graph.bind_resource(blas);
    let instance_buf = graph.bind_resource(instance_buf);

    graph
        .begin_cmd()
        .debug_name("Build TLAS")
        .resource_access(blas, AccessType::AccelerationStructureBuildRead)
        .resource_access(instance_buf, AccessType::AccelerationStructureBuildRead)
        .resource_access(scratch_buf, AccessType::AccelerationStructureBufferWrite)
        .resource_access(tlas, AccessType::AccelerationStructureBuildWrite)
        .record_cmd(move |cmd_buf| {
            cmd_buf.build_accel_struct(&[BuildAccelerationStructureInfo::new(
                tlas,
                scratch_addr,
                info,
            )]);
        });

    Ok(tlas)
}

fn download_model_from_github(model_name: &str) -> anyhow::Result<PathBuf> {
    const REPO_URL: &str =
        "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/";

    let model_path = current_exe()?.parent().unwrap().join(model_name);
    let model_metadata = metadata(&model_path);

    if model_metadata.is_err() {
        info!("Downloading model from github");

        let data = reqwest::blocking::get(REPO_URL.to_owned() + model_name)?.bytes()?;
        write(&model_path, data)?;

        info!("Download complete");
    }

    Ok(model_path)
}

fn load_ground_mesh(device: &Device) -> Result<Model, DriverError> {
    let extent = 100f32;
    let v0 = [-extent, 0.0, -extent];
    let v1 = [extent, 0.0, -extent];
    let v2 = [-extent, 0.0, extent];
    let v3 = [extent, 0.0, extent];
    let up = [0f32, 1.0, 0.0];

    let index_buf = Arc::new(Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        cast_slice(&[0u32, 1, 2, 1, 3, 2]),
    )?);
    let vertex_buf = Arc::new(Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        cast_slice(&[v0, up, v1, up, v2, up, v3, up]),
    )?);

    Ok(Model {
        index_buf,
        index_count: 6,
        vertex_buf,
        vertex_count: 4,
    })
}

fn load_model<T>(
    device: &Device,
    path: impl AsRef<Path>,
    face_fn: fn(a: Vec3, b: Vec3, c: Vec3) -> [T; 3],
) -> anyhow::Result<Model>
where
    T: Default + Pod,
{
    let (models, _) = load_obj(path.as_ref(), &GPU_LOAD_OPTIONS)?;
    let mut vertices =
        Vec::with_capacity(models.iter().map(|model| model.mesh.indices.len()).sum());

    // Calculate AABB
    let mut min = Vec3::ZERO;
    let mut max = Vec3::ZERO;
    for model in &models {
        for n in 0..model.mesh.positions.len() / 3 {
            let idx = 3 * n;
            let position = Vec3::from_slice(&model.mesh.positions[idx..idx + 3]);

            min = min.min(position);
            max = max.max(position);
        }
    }

    // Calculate a uniform scale which fits the model to a unit cube
    let scale = Vec3::splat(1.0 / (max - min).max_element());

    // Load the triangles using the face_fn closure to form vertices
    for model in models {
        for n in 0..model.mesh.indices.len() / 3 {
            let idx = 3 * n;
            let a_idx = 3 * model.mesh.indices[idx] as usize;
            let b_idx = 3 * model.mesh.indices[idx + 1] as usize;
            let c_idx = 3 * model.mesh.indices[idx + 2] as usize;
            let a = Vec3::from_slice(&model.mesh.positions[a_idx..a_idx + 3]) * scale;
            let b = Vec3::from_slice(&model.mesh.positions[b_idx..b_idx + 3]) * scale;
            let c = Vec3::from_slice(&model.mesh.positions[c_idx..c_idx + 3]) * scale;
            let face = face_fn(a, b, c);

            vertices.push(face[0]);
            vertices.push(face[1]);
            vertices.push(face[2]);
        }
    }

    // Re-index and de-dupe the model vertices using meshopt
    let indices = (0u32..vertices.len() as u32).collect::<Vec<_>>();
    let (vertex_count, remap) = generate_vertex_remap(&vertices, Some(&indices));
    let indices = remap_index_buffer(Some(&indices), vertex_count, &remap);
    let vertices = remap_vertex_buffer(&vertices, vertex_count, &remap);

    let index_buf = Arc::new(Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        cast_slice(&indices),
    )?);
    let vertex_buf = Arc::new(Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        cast_slice(&vertices),
    )?);

    Ok(Model {
        index_buf,
        index_count: indices.len() as _,
        vertex_buf,
        vertex_count: vertices.len() as _,
    })
}

/// Loads an .obj model as indexed position and normal vertices
fn load_model_mesh(device: &Device, path: impl AsRef<Path>) -> anyhow::Result<Model> {
    #[repr(C)]
    #[derive(Clone, Copy, Default, Pod, Zeroable)]
    struct Vertex {
        position: Vec3,
        normal: Vec3,
    }

    load_model(device, path, |a, b, c| {
        let u = b - a;
        let v = c - a;
        let normal = vec3(
            u.y * v.z - u.z * v.y,
            u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x,
        )
        .normalize();

        // Make faces CCW
        [
            Vertex {
                position: a,
                normal,
            },
            Vertex {
                position: c,
                normal,
            },
            Vertex {
                position: b,
                normal,
            },
        ]
    })
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Camera {
    projection: Mat4,
    view: Mat4,
    model: Mat4,
    light_position: Vec4,
}

struct Model {
    index_buf: Arc<Buffer>,
    index_count: u32,
    vertex_buf: Arc<Buffer>,
    vertex_count: u32,
}
