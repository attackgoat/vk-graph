mod profile_with_puffin;

use {
    bytemuck::{bytes_of, cast_slice},
    clap::Parser,
    glam::{Mat4, Vec3, Vec4, vec3},
    std::sync::Arc,
    vk_graph_prelude::*,
    vk_graph_window::WindowBuilder,
    vk_shader_macros::glsl,
};

#[derive(Clone, Copy)]
struct Camera {
    position: Vec3,
    projection: Mat4,
    view: Mat4,
}

#[derive(Clone, Copy)]
struct Material {
    color: Vec3,
    metallic: f32,
    roughness: f32,
}

struct Shape {
    index_buf: Arc<Buffer>,
    index_count: u32,
    vertex_buf: Arc<Buffer>,
}

const GOLD: Material = Material {
    color: vec3(1.0, 0.76, 0.33),
    metallic: 1.0,
    roughness: 0.3,
};

/// The example demonstrates leasing resources (images and buffers) and composing rendering
/// operations with just a few lines of Graph builder-pattern code.
///
/// Also shown:
/// - Basic PBR rendering (from Sascha Willems)
/// - Depth/stencil buffer usage
/// - Multiple rendering passes with a transient image
fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    let args = Args::parse();
    let window = WindowBuilder::default().debug(args.debug).build()?;
    let depth_stencil_format = best_depth_stencil_format(&window.device);
    let mut pool = LazyPool::new(&window.device);
    let fill_background = create_fill_background_pipeline(&window.device);
    let prepass = create_prepass_pipeline(&window.device);
    let pbr = create_pbr_pipeline(&window.device);
    let funky_shape = create_funky_shape(&window.device, &mut pool)?;

    let mut t = 0.0;
    window.run(|frame| {
        t += 0.016;

        let index_buf = frame.render_graph.bind_node(&funky_shape.index_buf);
        let vertex_buf = frame.render_graph.bind_node(&funky_shape.vertex_buf);

        let depth_stencil = frame.render_graph.bind_node(
            pool.lease(ImageInfo::image_2d(
                frame.width,
                frame.height,
                depth_stencil_format,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            ))
            .unwrap(),
        );

        let camera = camera(frame.width, frame.height);
        let model = Mat4::from_rotation_y(t * 0.4);
        let obj_pos = Vec3::ZERO;
        let material = GOLD;

        let camera_buf = bind_camera_buf(frame.render_graph, &mut pool, camera, model);
        let light_buf = bind_light_buf(frame.render_graph, &mut pool);
        let push_const_data = write_push_consts(obj_pos, material);

        let mut write = DepthStencilMode::DEPTH_WRITE;

        // Depth Prepass
        frame
            .render_graph
            .begin_cmd()
            .with_name("Depth Prepass")
            .bind_pipeline(&prepass)
            .set_depth_stencil(write)
            .read_descriptor(0, camera_buf)
            .access_node(index_buf, AccessType::IndexBuffer)
            .access_node(vertex_buf, AccessType::VertexBuffer)
            .clear_depth_stencil(depth_stencil)
            .store_depth_stencil(depth_stencil)
            .record_pipeline(move |pipeline, _| {
                pipeline
                    .bind_index_buffer(index_buf, 0, vk::IndexType::UINT16)
                    .bind_vertex_buffer(0, vertex_buf, 0)
                    .push_constants(0, bytes_of(&obj_pos))
                    .draw_indexed(funky_shape.index_count, 1, 0, 0, 0);
            });

        write.stencil_test = true;
        write.depth_test = false;
        write.front.compare_op = vk::CompareOp::GREATER_OR_EQUAL;
        write.front.compare_mask = 0xff;
        write.front.write_mask = 0xff;
        write.front.reference = 0x01;
        write.front.pass_op = vk::StencilOp::REPLACE;
        write.front.fail_op = vk::StencilOp::REPLACE;
        write.front.depth_fail_op = vk::StencilOp::REPLACE;
        write.back = write.front;

        // Renders a golden orb on an un-cleared swapchain image
        frame
            .render_graph
            .begin_cmd()
            .with_name("funky shape PBR")
            .bind_pipeline(&pbr)
            .set_depth_stencil(write)
            .read_descriptor(0, camera_buf)
            .read_descriptor(1, light_buf)
            .access_node(index_buf, AccessType::IndexBuffer)
            .access_node(vertex_buf, AccessType::VertexBuffer)
            .load_depth_stencil(depth_stencil)
            .store_depth_stencil(depth_stencil)
            .store_color(0, frame.swapchain_image)
            .record_pipeline(move |pipeline, _| {
                pipeline
                    .bind_index_buffer(index_buf, 0, vk::IndexType::UINT16)
                    .bind_vertex_buffer(0, vertex_buf, 0)
                    .push_constants(0, &push_const_data)
                    .draw_indexed(funky_shape.index_count, 1, 0, 0, 0);
            });

        let mut read = write;
        read.stencil_test = true;
        read.front.compare_op = vk::CompareOp::NOT_EQUAL;
        read.front.pass_op = vk::StencilOp::REPLACE;
        read.front.fail_op = vk::StencilOp::KEEP;
        read.front.depth_fail_op = vk::StencilOp::KEEP;

        // Renders a solid color wherever the golden orb did not draw
        frame
            .render_graph
            .begin_cmd()
            .with_name("fill background")
            .bind_pipeline(&fill_background)
            .set_depth_stencil(read)
            .load_depth_stencil(depth_stencil)
            .load_color(0, frame.swapchain_image)
            .store_color(0, frame.swapchain_image)
            .record_pipeline(move |pipeline, _| {
                pipeline.draw(6, 1, 0, 0);
            });
    })?;

    Ok(())
}

fn best_depth_stencil_format(device: &Device) -> vk::Format {
    for format in [
        vk::Format::D24_UNORM_S8_UINT,
        vk::Format::D16_UNORM_S8_UINT,
        vk::Format::D32_SFLOAT_S8_UINT,
    ] {
        let format_props = device.physical_device.image_format_properties(
            format,
            vk::ImageType::TYPE_2D,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            vk::ImageCreateFlags::empty(),
        );

        if matches!(format_props, Ok(Some(_))) {
            return format;
        }
    }

    panic!("Unsupported depth/stencil format");
}

fn bind_camera_buf(
    render_graph: &mut Graph,
    pool: &mut LazyPool,
    camera: Camera,
    model: Mat4,
) -> BufferLeaseNode {
    let mut buf = pool
        .lease(BufferInfo::host_mem(
            204,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        ))
        .unwrap();
    write_camera_buf(&mut buf, camera, model);

    render_graph.bind_node(buf)
}

fn bind_light_buf(render_graph: &mut Graph, pool: &mut LazyPool) -> BufferLeaseNode {
    let mut buf = pool
        .lease(BufferInfo::host_mem(
            64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        ))
        .unwrap();
    write_light_buf(&mut buf);

    render_graph.bind_node(buf)
}

fn write_push_consts(obj_pos: Vec3, material: Material) -> [u8; 32] {
    let mut data = [0u8; 32];

    write_vec3_to_slice(obj_pos, &mut data[0..]);
    write_f32_to_slice(material.roughness, &mut data[12..]);
    write_f32_to_slice(material.metallic, &mut data[16..]);
    write_vec3_to_slice(material.color, &mut data[20..]);

    data
}

fn camera(width: u32, height: u32) -> Camera {
    let aspect_ratio = width as f32 / height as f32;
    let fov_y_degrees = 45f32;
    let z_near = 0.1f32;
    let z_far = 100f32;
    let projection = Mat4::perspective_rh(fov_y_degrees.to_radians(), aspect_ratio, z_near, z_far);

    let position = vec3(0.0, 0.0, -5.0);
    let view = Mat4::look_at_rh(position, Vec3::ZERO, Vec3::Y);

    Camera {
        position,
        projection,
        view,
    }
}

/// Returns ready-to-use index and vertex buffers. Index count is also returned. The shape data uses
/// temporary staging buffers which are not required but are fun.
fn create_funky_shape(device: &Device, pool: &mut LazyPool) -> Result<Shape, DriverError> {
    // Static index/vertex data courtesy of the polyhedron-ops library
    let (indices, vertices) = funky_shape_data();
    let index_count = indices.len() as u32;

    // Create host-accessible buffers
    let index_buf_host = Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::TRANSFER_SRC,
        cast_slice(&indices),
    )?;
    let vertex_buf_host = Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::TRANSFER_SRC,
        cast_slice(&vertices),
    )?;

    // Create GPU-only buffers
    let index_buf = Arc::new(Buffer::create(
        device,
        BufferInfo::device_mem(
            index_buf_host.info.size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        ),
    )?);
    let vertex_buf = Arc::new(Buffer::create(
        device,
        BufferInfo::device_mem(
            vertex_buf_host.info.size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        ),
    )?);

    // We will use a temporary render graph to copy host data to the GPU
    let mut graph = Graph::default();

    // Bind things to the graph
    let index_buf_host = graph.bind_node(index_buf_host);
    let vertex_buf_host = graph.bind_node(vertex_buf_host);
    let index_buf_gpu = graph.bind_node(&index_buf);
    let vertex_buf_gpu = graph.bind_node(&vertex_buf);

    // Add operations to the graph which copy host-accessible data to GPU
    graph
        .copy_buffer(index_buf_host, index_buf_gpu)
        .copy_buffer(vertex_buf_host, vertex_buf_gpu);

    // Submit the graph, which runs the operations on the GPU
    graph.resolve().submit(pool, 0, 0)?;

    // (We drop the graph here; it's okay the cache keeps things alive until they're done)

    Ok(Shape {
        index_buf,
        index_count,
        vertex_buf,
    })
}

fn create_fill_background_pipeline(device: &Device) -> Arc<GraphicPipeline> {
    let vertex_shader = Shader::new_vertex(
        glsl!(
            r#"
            #version 450 core
            #pragma shader_stage(vertex)

            const float X[6] = {-1, -1, 1, 1, 1, -1};
            const float Y[6] = {-1, 1, -1, 1, -1, 1};

            vec2 vertex_pos() {
                float x = X[gl_VertexIndex];
                float y = Y[gl_VertexIndex];

                return vec2(x, y);
            }

            void main() {
                gl_Position = vec4(vertex_pos(), 0, 1);
            }
            "#
        )
        .as_slice(),
    );

    let fragment_shader = Shader::new_fragment(
        glsl!(
            r#"
            #version 450
            #pragma shader_stage(fragment)

            layout(location = 0) out vec4 color;

            void main() {
                color = vec4(vec3(0.75), 1.0);
            }
            "#
        )
        .as_slice(),
    );

    Arc::new(
        GraphicPipeline::create(
            device,
            GraphicPipelineInfo::default(),
            [vertex_shader, fragment_shader],
        )
        .unwrap(),
    )
}

fn create_prepass_pipeline(device: &Device) -> Arc<GraphicPipeline> {
    let vertex_shader = Shader::new_vertex(
        glsl!(
            r#"
            #version 450
            #pragma shader_stage(vertex)

            layout (location = 0) in vec3 inPos;
            layout (location = 1) in vec3 inNormal;

            layout (binding = 0) uniform UBO
            {
                mat4 projection;
                mat4 model;
                mat4 view;
                vec3 camPos;
            } ubo;

            layout (location = 0) out vec3 outWorldPos;
            layout (location = 1) out vec3 outNormal;

            layout(push_constant) uniform PushConsts {
                vec3 objPos;
            } pushConsts;

            out gl_PerVertex {
                vec4 gl_Position;
            };

            void main() {
                vec3 locPos = vec3(ubo.model * vec4(inPos, 1.0));
                outWorldPos = locPos + pushConsts.objPos;
                outNormal = mat3(ubo.model) * inNormal;
                gl_Position =  ubo.projection * ubo.view * vec4(outWorldPos, 1.0);
            }
            "#
        )
        .as_slice(),
    );

    let fragment_shader = Shader::new_fragment(
        glsl!(
            r#"
            #version 450
            #pragma shader_stage(fragment)

            layout (location = 0) in vec3 inWorldPos;
            layout (location = 1) in vec3 inNormal;

            layout (binding = 0) uniform UBO {
                mat4 projection;
                mat4 model;
                mat4 view;
                vec3 camPos;
            } ubo;

            void main() {
            }
            "#
        )
        .as_slice(),
    );

    Arc::new(
        GraphicPipeline::create(
            device,
            GraphicPipelineInfo::default(),
            [vertex_shader, fragment_shader],
        )
        .unwrap(),
    )
}

fn create_pbr_pipeline(device: &Device) -> Arc<GraphicPipeline> {
    // See: https://github.com/SaschaWillems/Vulkan/blob/master/data/shaders/glsl/pbrbasic/pbr.vert
    let vertex_shader = Shader::new_vertex(
        glsl!(
            r#"
            #version 450
            #pragma shader_stage(vertex)

            layout (location = 0) in vec3 inPos;
            layout (location = 1) in vec3 inNormal;

            layout (binding = 0) uniform UBO
            {
                mat4 projection;
                mat4 model;
                mat4 view;
                vec3 camPos;
            } ubo;

            layout (location = 0) out vec3 outWorldPos;
            layout (location = 1) out vec3 outNormal;

            layout(push_constant) uniform PushConsts {
                vec3 objPos;
            } pushConsts;

            out gl_PerVertex {
                vec4 gl_Position;
            };

            void main() {
                vec3 locPos = vec3(ubo.model * vec4(inPos, 1.0));
                outWorldPos = locPos + pushConsts.objPos;
                outNormal = mat3(ubo.model) * inNormal;
                gl_Position =  ubo.projection * ubo.view * vec4(outWorldPos, 1.0);
            }
            "#
        )
        .as_slice(),
    );

    // See: https://github.com/SaschaWillems/Vulkan/blob/master/data/shaders/glsl/pbrbasic/pbr.frag
    let fragment_shader = Shader::new_fragment(
        glsl!(
            r#"
            #version 450
            #pragma shader_stage(fragment)

            layout (location = 0) in vec3 inWorldPos;
            layout (location = 1) in vec3 inNormal;

            layout (binding = 0) uniform UBO 
            {
                mat4 projection;
                mat4 model;
                mat4 view;
                vec3 camPos;
            } ubo;

            layout (binding = 1) uniform UBOShared {
                vec4 lights[4];
            } uboParams;

            layout (location = 0) out vec4 outColor;

            layout(push_constant) uniform PushConsts {
                layout(offset = 12) float roughness;
                layout(offset = 16) float metallic;
                layout(offset = 20) float r;
                layout(offset = 24) float g;
                layout(offset = 28) float b;
            } material;

            const float PI = 3.14159265359;

            //#define ROUGHNESS_PATTERN 1

            vec3 materialcolor()
            {
                return vec3(material.r, material.g, material.b);
            }

            // Normal Distribution function --------------------------------------
            float D_GGX(float dotNH, float roughness)
            {
                float alpha = roughness * roughness;
                float alpha2 = alpha * alpha;
                float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
                return (alpha2)/(PI * denom*denom); 
            }

            // Geometric Shadowing function --------------------------------------
            float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
            {
                float r = (roughness + 1.0);
                float k = (r*r) / 8.0;
                float GL = dotNL / (dotNL * (1.0 - k) + k);
                float GV = dotNV / (dotNV * (1.0 - k) + k);
                return GL * GV;
            }

            // Fresnel function ----------------------------------------------------
            vec3 F_Schlick(float cosTheta, float metallic)
            {
                vec3 F0 = mix(vec3(0.04), materialcolor(), metallic); // * material.specular
                vec3 F = F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0); 
                return F;
            }

            // Specular BRDF composition --------------------------------------------

            vec3 BRDF(vec3 L, vec3 V, vec3 N, float metallic, float roughness)
            {
                // Precalculate vectors and dot products	
                vec3 H = normalize (V + L);
                float dotNV = clamp(dot(N, V), 0.0, 1.0);
                float dotNL = clamp(dot(N, L), 0.0, 1.0);
                float dotLH = clamp(dot(L, H), 0.0, 1.0);
                float dotNH = clamp(dot(N, H), 0.0, 1.0);

                // Light color fixed
                vec3 lightColor = vec3(1.0);

                vec3 color = vec3(0.0);

                if (dotNL > 0.0)
                {
                    float rroughness = max(0.05, roughness);
                    // D = Normal distribution (Distribution of the microfacets)
                    float D = D_GGX(dotNH, roughness); 
                    // G = Geometric shadowing term (Microfacets shadowing)
                    float G = G_SchlicksmithGGX(dotNL, dotNV, rroughness);
                    // F = Fresnel factor (Reflectance depending on angle of incidence)
                    vec3 F = F_Schlick(dotNV, metallic);

                    vec3 spec = D * F * G / (4.0 * dotNL * dotNV);

                    color += spec * dotNL * lightColor;
                }

                return color;
            }

            // ----------------------------------------------------------------------------
            void main()
            {
                vec3 N = normalize(inNormal);
                vec3 V = normalize(ubo.camPos - inWorldPos);

                float roughness = material.roughness;

                // Add striped pattern to roughness based on vertex position
            #ifdef ROUGHNESS_PATTERN
                roughness = max(roughness, step(fract(inWorldPos.y * 2.02), 0.5));
            #endif

                // Specular contribution
                vec3 Lo = vec3(0.0);
                for (int i = 0; i < uboParams.lights.length(); i++) {
                    vec3 L = normalize(uboParams.lights[i].xyz - inWorldPos);
                    Lo += BRDF(L, V, N, material.metallic, roughness);
                };

                // Combine with ambient
                vec3 color = materialcolor() * 0.02;
                color += Lo;

                // Gamma correct
                color = pow(color, vec3(0.4545));

                outColor = vec4(color, 1.0);
            }
            "#
        )
        .as_slice(),
    );

    Arc::new(
        GraphicPipeline::create(
            device,
            GraphicPipelineInfo::default(),
            [vertex_shader, fragment_shader],
        )
        .unwrap(),
    )
}

/// Returns index and position/normal data (polyhedron_ops you are 🥇🏆🥂💯)
fn funky_shape_data() -> (Vec<u16>, Vec<[f32; 6]>) {
    let (indices, positions, normals) = polyhedron_ops::Polyhedron::dodecahedron()
        .chamfer(None, false)
        .bevel(None, None, None, None, false)
        .catmull_clark_subdivide(false)
        .bevel(None, None, None, None, false)
        .finalize()
        .to_triangle_mesh_buffers();
    let indices = indices.into_iter().map(|idx| idx as u16).collect();
    let vertices = positions
        .into_iter()
        .zip(normals)
        .map(|(position, normal)| {
            [
                position.x, position.y, position.z, normal.x, normal.y, normal.z,
            ]
        })
        .collect();

    (indices, vertices)
}

fn write_cols_to_slice(data: Mat4, slice: &mut [u8]) -> usize {
    let mut start = 0;
    for data in data.to_cols_array() {
        let data = data.to_ne_bytes();
        let end = start + data.len();
        slice[start..end].clone_from_slice(&data);
        start = end;
    }

    start
}

fn write_f32_to_slice(data: f32, slice: &mut [u8]) -> usize {
    slice[0..4].clone_from_slice(&data.to_ne_bytes());

    4
}

fn write_vec3_to_slice(data: Vec3, slice: &mut [u8]) -> usize {
    let mut start = 0;
    for data in data.to_array() {
        let data = data.to_ne_bytes();
        let end = start + data.len();
        slice[start..end].clone_from_slice(&data);
        start = end;
    }

    start
}

fn write_vec4_to_slice(data: Vec4, slice: &mut [u8]) -> usize {
    let mut start = 0;
    for data in data.to_array() {
        let data = data.to_ne_bytes();
        let end = start + data.len();
        slice[start..end].clone_from_slice(&data);
        start = end;
    }

    start
}

fn write_camera_buf(buf: &mut Lease<Buffer>, camera: Camera, model: Mat4) {
    let data = Buffer::mapped_slice_mut(buf);

    write_cols_to_slice(camera.projection, &mut data[0..]);
    write_cols_to_slice(model, &mut data[64..]);
    write_cols_to_slice(camera.view, &mut data[128..]);

    write_vec3_to_slice(camera.position, &mut data[192..]);
}

fn write_light_buf(buf: &mut Lease<Buffer>) {
    let data = Buffer::mapped_slice_mut(buf);

    let p = 4.0;
    write_vec4_to_slice(vec3(0.0, -p, -p).extend(1.0), &mut data[0..]);
    write_vec4_to_slice(vec3(p * 0.5, p, -p).extend(1.0), &mut data[16..]);
    write_vec4_to_slice(vec3(-p, -p * 0.5, -p).extend(1.0), &mut data[32..]);
    write_vec4_to_slice(vec3(p, -p * 0.5, -p).extend(1.0), &mut data[48..]);
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
