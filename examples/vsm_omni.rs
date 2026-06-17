mod profile_with_puffin;

use {
    ash::vk,
    bytemuck::{NoUninit, Pod, Zeroable, bytes_of, cast_slice},
    clap::Parser,
    glam::{Mat4, Quat, Vec3, vec3},
    log::info,
    meshopt::remap::{generate_vertex_remap, remap_index_buffer, remap_vertex_buffer},
    std::{
        env::current_exe,
        fs::{metadata, write},
        path::{Path, PathBuf},
        sync::Arc,
        time::Instant,
    },
    tobj::{GPU_LOAD_OPTIONS, load_obj},
    vk_graph::{
        cmd::{LoadOp, StoreOp},
        driver::{
            DriverError,
            buffer::{Buffer, BufferInfo},
            compute::{ComputePipeline, ComputePipelineInfo},
            device::Device,
            graphics::{DepthStencilInfo, GraphicsPipeline, GraphicsPipelineInfo},
            image::ImageInfo,
            physical_device::Vulkan10Features,
            shader::{Shader, SpecializationMap},
        },
        pool::{Lease, Pool, fifo::FifoPool},
    },
    vk_graph_window::Window,
    vk_shader_macros::glsl,
    vk_sync::AccessType,
    winit::{dpi::LogicalSize, event::Event, keyboard::KeyCode, window::Fullscreen},
    winit_input_helper::WinitInputHelper,
};

const BLUR_PASSES: usize = 2;
const BLUR_RADIUS: u32 = 2;
const CUBEMAP_SIZE: u32 = 512;
const SHADOW_BIAS: f32 = 0.3;

/// Adapted from https://github.com/sydneyzh/variance_shadow_mapping_vk
///
/// This example does an HTTPS GET to acquire model data!
/// Model data courtesy: https://github.com/alecjacobson/common-3d-test-models/
///
/// Also, see similar techniques here:
/// https://github.com/SaschaWillems/Vulkan/blob/ae3c1325f8c7a55941dc5b325db58bb482dce04c/examples/shadowmappingomni/shadowmappingomni.cpp
/// https://github.com/SaschaWillems/Vulkan/pull/783
fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    let model_path = download_model_from_github("nefertiti.obj")?;
    let model_transform = Mat4::from_scale_rotation_translation(
        Vec3::splat(15.0),
        Quat::from_rotation_z(180f32.to_radians()) * Quat::from_rotation_x(90f32.to_radians()),
        vec3(0.0, -2.0, 0.0),
    )
    .to_cols_array();
    let cube_transform = Mat4::from_scale(Vec3::splat(10.0)).to_cols_array();

    let mut input = WinitInputHelper::default();
    let args = Args::parse();
    let window = Window::builder()
        .debug(args.debug)
        .window(|window| window.with_inner_size(LogicalSize::new(800, 600)))
        .build()?;

    // We may use a geometry shader on supported devices
    let use_geometry_shader = {
        let Vulkan10Features {
            geometry_shader, ..
        } = window.device.physical_device.features_v1_0;

        args.geometry_shader && geometry_shader
    };

    // Load all the immutable graphics data we will need
    let cubemap_format = best_2d_optimal_format(
        &window.device,
        &[
            vk::Format::R32G32_SFLOAT,
            vk::Format::R16G16_SFLOAT,
            vk::Format::R32G32B32_SFLOAT,
            vk::Format::R16G16B16_SFLOAT,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::Format::R16G16B16A16_SFLOAT,
        ],
        vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE,
        vk::ImageCreateFlags::CUBE_COMPATIBLE,
    );
    let depth_format = best_2d_optimal_format(
        &window.device,
        &[vk::Format::D32_SFLOAT, vk::Format::D16_UNORM],
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::ImageCreateFlags::empty(),
    );
    let model_mesh = load_model_mesh(&window.device, &model_path)?;
    let model_shadow = load_model_shadow(&window.device, &model_path)?;
    let cube_mesh = load_cube_mesh(&window.device)?;
    let cube_shadow = load_cube_shadow(&window.device)?;
    let debug_pipeline = create_debug_pipeline(&window.device)?;
    let blur_x_pipeline = create_blur_x_pipeline(&window.device)?;
    let blur_y_pipeline = create_blur_y_pipeline(&window.device)?;
    let mesh_pipeline = create_mesh_pipeline(&window.device)?;
    let shadow_pipeline = if use_geometry_shader {
        create_shadow_pipeline_with_geometry_shader(&window.device)
    } else {
        create_shadow_pipeline(&window.device)
    }?;

    // A pool will be used for per-frame resources
    let mut pool = FifoPool::new(&window.device);

    let mut elapsed = 0.0;
    let mut prev_frame_at = Instant::now();
    window.run(|frame| {
        let now = Instant::now();
        let dt = now - prev_frame_at;
        prev_frame_at = now;

        input.step();
        for event in frame.events {
            match event {
                Event::WindowEvent { event, .. } => {
                    let _ = input.process_window_event(event);
                }
                Event::DeviceEvent { event, .. } => {
                    input.process_device_event(event);
                }
                _ => {}
            }
        }
        input.end_step();

        // Hold spacebar to stop the light
        if !input.key_held(KeyCode::Space) {
            elapsed += dt.as_secs_f32();
        }

        // Hit F11 to enable borderless fullscreen
        if input.key_pressed(KeyCode::F11) {
            frame
                .window
                .set_fullscreen(Some(Fullscreen::Borderless(None)));
        }

        // Hit F12 to enable exclusive fullscreen
        if input.key_pressed(KeyCode::F12)
            && let Some(monitor) = frame.window.current_monitor()
            && let Some(video_mode) = monitor.video_modes().next()
        {
            frame
                .window
                .set_fullscreen(Some(Fullscreen::Exclusive(video_mode)));
        }

        // Hit Escape to cancel fullscreen or exit
        if input.key_pressed(KeyCode::Escape) {
            if frame.window.fullscreen().is_some() {
                frame.window.set_fullscreen(None);
            } else {
                *frame.will_exit = true;
            }
        }

        // Calculate values for and fill some plain-old-data structs we will bind as UBO's
        let camera = {
            let aspect_ratio = frame.width as f32 / frame.height as f32;
            let fov_y = 45f32.to_radians();
            let projection = Mat4::perspective_rh(fov_y, aspect_ratio, 0.1, 100.0);

            let eye = vec3(0.0, 0.0, -25.0);
            let view = Mat4::look_at_rh(eye, eye + Vec3::Z, -Vec3::Y);

            Camera { view, projection }
        };
        let light = {
            let fov_y = 90f32.to_radians();
            let radius = 7f32;
            let t = elapsed / 2.0 + 3.5;
            let position = vec3(radius * t.cos(), 0.0, radius * t.sin());

            Light {
                position,
                range: 1000.0,
                view: Mat4::look_at_rh(position, position + Vec3::X, Vec3::Y),
                projection: Mat4::perspective_rh(fov_y, 1.0, 0.1, 100.0),
            }
        };

        // Bind resources to the render graph of the current frame
        let cube_mesh_index_buf = frame.graph.bind_resource(&cube_mesh.index_buf);
        let cube_mesh_vertex_buf = frame.graph.bind_resource(&cube_mesh.vertex_buf);
        let cube_shadow_index_buf = frame.graph.bind_resource(&cube_shadow.index_buf);
        let cube_shadow_vertex_buf = frame.graph.bind_resource(&cube_shadow.vertex_buf);
        let model_mesh_index_buf = frame.graph.bind_resource(&model_mesh.index_buf);
        let model_mesh_vertex_buf = frame.graph.bind_resource(&model_mesh.vertex_buf);
        let model_shadow_index_buf = frame.graph.bind_resource(&model_shadow.index_buf);
        let model_shadow_vertex_buf = frame.graph.bind_resource(&model_shadow.vertex_buf);
        let camera_uniform_buf = frame
            .graph
            .bind_resource(lease_uniform_buffer(&mut pool, &camera).unwrap());
        let light_uniform_buf = frame
            .graph
            .bind_resource(lease_uniform_buffer(&mut pool, &light).unwrap());

        // Lease and bind a cube-compatible shadow 2D image array to the graph of the current frame
        let shadow_faces_image = pool
            .resource(
                ImageInfo::image_2d_array(
                    CUBEMAP_SIZE,
                    CUBEMAP_SIZE,
                    6,
                    cubemap_format,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::STORAGE,
                )
                .into_builder()
                .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE),
            )
            .unwrap();
        let shadow_faces_info = shadow_faces_image.info;
        let shadow_faces_node = frame.graph.bind_resource(shadow_faces_image);

        // Lease and bind a temporary image we'll use during blur passes
        let temp_image = frame
            .graph
            .bind_resource(pool.resource(shadow_faces_info).unwrap());

        // Lastly we lease and bind depth images needed for rendering
        let shadow_depth_image = frame.graph.bind_resource(
            pool.resource(ImageInfo::image_2d_array(
                frame.width,
                frame.height,
                if use_geometry_shader { 6 } else { 1 },
                depth_format,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ))
            .unwrap(),
        );
        let depth_image = frame.graph.bind_resource(
            pool.resource(ImageInfo::image_2d(
                frame.width,
                frame.height,
                depth_format,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ))
            .unwrap(),
        );

        // Hold tab to view a debug mode
        if input.key_held(KeyCode::Tab) {
            frame
                .graph
                .begin_cmd()
                .debug_name("DEBUG")
                .bind_pipeline(&debug_pipeline)
                .depth_stencil(DepthStencilInfo::DEPTH_WRITE_LESS)
                .shader_resource_access(
                    0,
                    camera_uniform_buf,
                    AccessType::AnyShaderReadUniformBuffer,
                )
                .shader_resource_access(
                    1,
                    light_uniform_buf,
                    AccessType::AnyShaderReadUniformBuffer,
                )
                .resource_access(model_mesh_index_buf, AccessType::IndexBuffer)
                .resource_access(model_mesh_vertex_buf, AccessType::VertexBuffer)
                .resource_access(cube_mesh_index_buf, AccessType::IndexBuffer)
                .resource_access(cube_mesh_vertex_buf, AccessType::VertexBuffer)
                .depth_stencil_attachment_image(
                    depth_image,
                    LoadOp::CLEAR_ONE_STENCIL_ZERO,
                    StoreOp::Store,
                )
                .color_attachment_image(
                    0,
                    frame.swapchain_image,
                    LoadOp::CLEAR_BLACK_ALPHA_ZERO,
                    StoreOp::Store,
                )
                .record_cmd(move |cmd| {
                    cmd.bind_index_buffer(model_mesh_index_buf, 0, vk::IndexType::UINT32)
                        .bind_vertex_buffer(0, model_mesh_vertex_buf, 0)
                        .push_constants(0, cast_slice(&model_transform))
                        .draw_indexed(model_mesh.index_count, 1, 0, 0, 0)
                        .bind_index_buffer(cube_mesh_index_buf, 0, vk::IndexType::UINT32)
                        .bind_vertex_buffer(0, cube_mesh_vertex_buf, 0)
                        .push_constants(0, cast_slice(&cube_transform))
                        .draw_indexed(cube_mesh.index_count, 1, 0, 0, 0);
                });
        } else {
            // Render the omni light point of view into the six-layer image we leased above
            if use_geometry_shader {
                frame
                    .graph
                    .begin_cmd()
                    .debug_name("Shadow (Using geometry shader)")
                    .bind_pipeline(&shadow_pipeline)
                    .depth_stencil(DepthStencilInfo::DEPTH_WRITE_LESS)
                    .shader_resource_access(
                        0,
                        light_uniform_buf,
                        AccessType::AnyShaderReadUniformBuffer,
                    )
                    .resource_access(model_shadow_index_buf, AccessType::IndexBuffer)
                    .resource_access(model_shadow_vertex_buf, AccessType::VertexBuffer)
                    .resource_access(cube_shadow_index_buf, AccessType::IndexBuffer)
                    .resource_access(cube_shadow_vertex_buf, AccessType::VertexBuffer)
                    .depth_stencil_attachment_image(
                        shadow_depth_image,
                        LoadOp::CLEAR_ONE_STENCIL_ZERO,
                        StoreOp::DontCare,
                    )
                    .color_attachment_image(
                        0,
                        shadow_faces_node,
                        LoadOp::clear_rgba(light.range, light.range, 0.0, 0.0),
                        StoreOp::Store,
                    )
                    .record_cmd(move |cmd| {
                        cmd.bind_index_buffer(model_shadow_index_buf, 0, vk::IndexType::UINT32)
                            .bind_vertex_buffer(0, model_shadow_vertex_buf, 0)
                            .push_constants(0, cast_slice(&model_transform))
                            .draw_indexed(model_shadow.index_count, 1, 0, 0, 0)
                            .bind_index_buffer(cube_shadow_index_buf, 0, vk::IndexType::UINT32)
                            .bind_vertex_buffer(0, cube_shadow_vertex_buf, 0)
                            .push_constants(0, cast_slice(&cube_transform))
                            .draw_indexed(cube_shadow.index_count, 1, 0, 0, 0);
                    });
            } else {
                for array_layer in 0..6 {
                    let mut shadow_faces_view_info = shadow_faces_info.into_image_view();
                    shadow_faces_view_info.array_layer_count = 1;
                    shadow_faces_view_info.base_array_layer = array_layer;

                    let mut light = light;
                    light.view = match array_layer {
                        0 => Mat4::look_at_rh(light.position, light.position + Vec3::X, -Vec3::Y),
                        1 => Mat4::look_at_rh(light.position, light.position - Vec3::X, -Vec3::Y),
                        2 => Mat4::look_at_rh(light.position, light.position + Vec3::Y, Vec3::Z),
                        3 => Mat4::look_at_rh(light.position, light.position - Vec3::Y, -Vec3::Z),
                        4 => Mat4::look_at_rh(light.position, light.position + Vec3::Z, -Vec3::Y),
                        _ => Mat4::look_at_rh(light.position, light.position - Vec3::Z, -Vec3::Y),
                    };

                    let light_uniform_buf = frame
                        .graph
                        .bind_resource(lease_uniform_buffer(&mut pool, &light).unwrap());

                    frame
                        .graph
                        .begin_cmd()
                        .debug_name("Shadow")
                        .bind_pipeline(&shadow_pipeline)
                        .depth_stencil(DepthStencilInfo::DEPTH_WRITE_LESS)
                        .shader_resource_access(
                            0,
                            light_uniform_buf,
                            AccessType::AnyShaderReadUniformBuffer,
                        )
                        .resource_access(model_shadow_index_buf, AccessType::IndexBuffer)
                        .resource_access(model_shadow_vertex_buf, AccessType::VertexBuffer)
                        .resource_access(cube_shadow_index_buf, AccessType::IndexBuffer)
                        .resource_access(cube_shadow_vertex_buf, AccessType::VertexBuffer)
                        .depth_stencil_attachment_image(
                            shadow_depth_image,
                            LoadOp::CLEAR_ONE_STENCIL_ZERO,
                            StoreOp::DontCare,
                        )
                        .color_attachment_image_view(
                            0,
                            shadow_faces_node,
                            shadow_faces_view_info,
                            LoadOp::clear_rgba(light.range, light.range, 0.0, 0.0),
                            StoreOp::Store,
                        )
                        .record_cmd(move |cmd| {
                            cmd.bind_index_buffer(model_shadow_index_buf, 0, vk::IndexType::UINT32)
                                .bind_vertex_buffer(0, model_shadow_vertex_buf, 0)
                                .push_constants(0, cast_slice(&model_transform))
                                .draw_indexed(model_shadow.index_count, 1, 0, 0, 0)
                                .bind_index_buffer(cube_shadow_index_buf, 0, vk::IndexType::UINT32)
                                .bind_vertex_buffer(0, cube_shadow_vertex_buf, 0)
                                .push_constants(0, cast_slice(&cube_transform))
                                .draw_indexed(cube_shadow.index_count, 1, 0, 0, 0);
                        });
                }
            }

            if BLUR_RADIUS > 0 {
                for _ in 0..BLUR_PASSES {
                    // Flip-flop between the shadow image and a temporary image using a
                    // separable box blur filter which approximates a gaussian blur
                    frame
                        .graph
                        .begin_cmd()
                        .debug_name("Blur X")
                        .bind_pipeline(&blur_x_pipeline)
                        .shader_resource_access(
                            0,
                            shadow_faces_node,
                            AccessType::ComputeShaderReadOther,
                        )
                        .shader_resource_access(1, temp_image, AccessType::ComputeShaderWrite)
                        .record_cmd(move |cmd| {
                            cmd.dispatch(1, CUBEMAP_SIZE, 6);
                        })
                        .end_cmd()
                        .begin_cmd()
                        .debug_name("Blur Y")
                        .bind_pipeline(&blur_y_pipeline)
                        .shader_resource_access(0, temp_image, AccessType::ComputeShaderReadOther)
                        .shader_resource_access(
                            1,
                            shadow_faces_node,
                            AccessType::ComputeShaderWrite,
                        )
                        .record_cmd(move |cmd| {
                            cmd.dispatch(CUBEMAP_SIZE, 1, 6);
                        });
                }
            }

            // Render the scene directly to the swapchain using the shadow map from the above pass
            frame
                .graph
                .begin_cmd()
                .debug_name("Mesh objects")
                .bind_pipeline(&mesh_pipeline)
                .depth_stencil(DepthStencilInfo::DEPTH_WRITE_LESS)
                .shader_resource_access(
                    0,
                    camera_uniform_buf,
                    AccessType::AnyShaderReadUniformBuffer,
                )
                .shader_resource_access(
                    1,
                    light_uniform_buf,
                    AccessType::AnyShaderReadUniformBuffer,
                )
                .shader_subresource_access(
                    2,
                    shadow_faces_node,
                    shadow_faces_info
                        .into_image_view()
                        .into_builder()
                        .ty(vk::ImageViewType::CUBE),
                    AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
                )
                .resource_access(model_mesh_index_buf, AccessType::IndexBuffer)
                .resource_access(model_mesh_vertex_buf, AccessType::VertexBuffer)
                .resource_access(cube_mesh_index_buf, AccessType::IndexBuffer)
                .resource_access(cube_mesh_vertex_buf, AccessType::VertexBuffer)
                .depth_stencil_attachment_image(
                    depth_image,
                    LoadOp::CLEAR_ONE_STENCIL_ZERO,
                    StoreOp::DontCare,
                )
                .color_attachment_image(
                    0,
                    frame.swapchain_image,
                    LoadOp::CLEAR_BLACK_ALPHA_ZERO,
                    StoreOp::Store,
                )
                .record_cmd(move |cmd| {
                    cmd.bind_index_buffer(model_mesh_index_buf, 0, vk::IndexType::UINT32)
                        .bind_vertex_buffer(0, model_mesh_vertex_buf, 0)
                        .push_constants(0, cast_slice(&model_transform))
                        .draw_indexed(model_mesh.index_count, 1, 0, 0, 0);

                    cmd.bind_index_buffer(cube_mesh_index_buf, 0, vk::IndexType::UINT32)
                        .bind_vertex_buffer(0, cube_mesh_vertex_buf, 0)
                        .push_constants(0, cast_slice(&cube_transform))
                        .draw_indexed(cube_mesh.index_count, 1, 0, 0, 0);
                });
        }
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

fn create_blur_x_pipeline(device: &Device) -> Result<ComputePipeline, DriverError> {
    let comp = glsl!(
        r#"
        #version 450 core
        #pragma shader_stage(compute)

        #define POS_X 0
        #define NEG_X 1
        #define POS_Y 2
        #define NEG_Y 3
        #define POS_Z 4
        #define NEG_Z 5

        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        layout(constant_id = 0) const uint IMAGE_SIZE = 512;
        layout(constant_id = 1) const uint RADIUS = 4;

        layout(binding = 0, rg32f) restrict readonly uniform image2DArray image;
        layout(binding = 1, rg32f) restrict writeonly uniform image2DArray image_out;

        ivec3 leading_face(uint x) {
            uint face = gl_GlobalInvocationID.z;
            uint y = gl_GlobalInvocationID.y;

            switch (face) {
                case POS_X:
                    return ivec3(x, y, POS_Z);
                case NEG_X:
                    return ivec3(x, y, NEG_Z);
                case POS_Y:
                    return ivec3(y, IMAGE_SIZE - (x + 1), NEG_X);
                case NEG_Y:
                    return ivec3(IMAGE_SIZE - (y + 1), x, NEG_X);
                case POS_Z:
                    return ivec3(x, y, NEG_X);
                default:
                    return ivec3(x, y, POS_X);
            }
        }

        ivec3 trailing_face(uint x) {
            uint face = gl_GlobalInvocationID.z;
            uint y = gl_GlobalInvocationID.y;

            switch (face) {
                case POS_X:
                    return ivec3(x, y, NEG_Z);
                case NEG_X:
                    return ivec3(x, y, POS_Z);
                case POS_Y:
                    return ivec3(IMAGE_SIZE - (y + 1), x, POS_X);
                case NEG_Y:
                    return ivec3(y, IMAGE_SIZE - (x + 1), POS_X);
                case POS_Z:
                    return ivec3(x, y, POS_X);
                default:
                    return ivec3(x, y, NEG_X);
            }
        }

        void main() {
            uint face = gl_GlobalInvocationID.z;
            uint y = gl_GlobalInvocationID.y;

            vec2 accumulator = vec2(0.0);
            float per_texel = 1.0 / float((RADIUS << 1) + 1);

            for (uint x = IMAGE_SIZE - RADIUS; x < IMAGE_SIZE; x++) {
                accumulator += imageLoad(image, leading_face(x)).rg;
            }

            for (uint x = 0; x < RADIUS; x++) {
                accumulator += imageLoad(image, ivec3(x, y, face)).rg;
            }

            for (uint x = 0; x < RADIUS; x++) {
                accumulator += imageLoad(image, ivec3(x + RADIUS, y, face)).rg;
                imageStore(image_out, ivec3(x, y, face), vec4(accumulator * per_texel, 0.0, 0.0));
                accumulator -= imageLoad(image, leading_face((IMAGE_SIZE - RADIUS) + x)).rg;
            }

            for (uint x = RADIUS; x < IMAGE_SIZE - RADIUS; x++) {
                accumulator += imageLoad(image, ivec3(x + RADIUS, y, face)).rg;
                imageStore(image_out, ivec3(x, y, face), vec4(accumulator * per_texel, 0.0, 0.0));
                accumulator -= imageLoad(image, ivec3(x - RADIUS, y, face)).rg;
            }

            for (uint x = IMAGE_SIZE - RADIUS; x < IMAGE_SIZE; x++) {
                accumulator += imageLoad(image, trailing_face((x + RADIUS) - IMAGE_SIZE)).rg;
                imageStore(image_out, ivec3(x, y, face), vec4(accumulator * per_texel, 0.0, 0.0));
                accumulator -= imageLoad(image, ivec3(x - RADIUS, y, face)).rg;
            }
        }
        "#
    );

    let shader = Shader::new_compute(comp.as_slice()).specialization(
        SpecializationMap::new(bytes_of(&Blur {
            image_size: CUBEMAP_SIZE,
            radius: BLUR_RADIUS,
        }))
        .constant(0, 0, 4)
        .constant(1, 4, 4),
    );

    ComputePipeline::create(device, ComputePipelineInfo::default(), shader)
}

fn create_blur_y_pipeline(device: &Device) -> Result<ComputePipeline, DriverError> {
    let comp = glsl!(
        r#"
        #version 450 core
        #pragma shader_stage(compute)

        #define POS_X 0
        #define NEG_X 1
        #define POS_Y 2
        #define NEG_Y 3
        #define POS_Z 4
        #define NEG_Z 5

        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        layout(constant_id = 0) const uint IMAGE_SIZE = 512;
        layout(constant_id = 1) const uint RADIUS = 4;

        layout(binding = 0, rg32f) restrict readonly uniform image2DArray image;
        layout(binding = 1, rg32f) restrict writeonly uniform image2DArray image_out;

        ivec3 leading_face(uint y) {
            uint face = gl_GlobalInvocationID.z;
            uint x = gl_GlobalInvocationID.x;

            switch (face) {
                case POS_X:
                    return ivec3(y, IMAGE_SIZE - (x + 1), POS_Y);
                case NEG_X:
                    return ivec3(IMAGE_SIZE - (y + 1), x, POS_Y);
                case POS_Y:
                    return ivec3(IMAGE_SIZE - (x + 1), IMAGE_SIZE - (y + 1), NEG_Z);
                case NEG_Y:
                    return ivec3(x, y, POS_Z);
                case POS_Z:
                    return ivec3(x, y, POS_Y);
                default:
                    return ivec3(IMAGE_SIZE - (x + 1), IMAGE_SIZE - (y + 1), POS_Y);
            }
        }

        ivec3 trailing_face(uint y) {
            uint face = gl_GlobalInvocationID.z;
            uint x = gl_GlobalInvocationID.x;

            switch (face) {
                case POS_X:
                    return ivec3(IMAGE_SIZE - (y + 1), x, NEG_Y);
                case NEG_X:
                    return ivec3(y, IMAGE_SIZE - (x + 1), NEG_Y);
                case POS_Y:
                    return ivec3(x, y, POS_Z);
                case NEG_Y:
                    return ivec3(IMAGE_SIZE - (x + 1), IMAGE_SIZE - (y + 1), NEG_Z);
                case POS_Z:
                    return ivec3(x, y, NEG_Y);
                default:
                    return ivec3(IMAGE_SIZE - (x + 1), IMAGE_SIZE - (y + 1), NEG_Y);
            }
        }

        void main() {
            uint face = gl_GlobalInvocationID.z;
            uint x = gl_GlobalInvocationID.x;

            vec2 accumulator = vec2(0.0);
            float per_texel = 1.0 / float((RADIUS << 1) + 1);

            for (uint y = IMAGE_SIZE - RADIUS; y < IMAGE_SIZE; y++) {
                accumulator += imageLoad(image, leading_face(y)).rg;
            }

            for (uint y = 0; y < RADIUS; y++) {
                accumulator += imageLoad(image, ivec3(x, y, face)).rg;
            }

            for (uint y = 0; y < RADIUS; y++) {
                accumulator += imageLoad(image, ivec3(x, y + RADIUS, face)).rg;
                imageStore(image_out, ivec3(x, y, face), vec4(accumulator * per_texel, 0.0, 0.0));
                accumulator -= imageLoad(image, leading_face((IMAGE_SIZE - RADIUS) + y)).rg;
            }

            for (uint y = RADIUS; y < IMAGE_SIZE - RADIUS; y++) {
                accumulator += imageLoad(image, ivec3(x, y + RADIUS, face)).rg;
                imageStore(image_out, ivec3(x, y, face), vec4(accumulator * per_texel, 0.0, 0.0));
                accumulator -= imageLoad(image, ivec3(x, y - RADIUS, face)).rg;
            }

            for (uint y = IMAGE_SIZE - RADIUS; y < IMAGE_SIZE; y++) {
                accumulator += imageLoad(image, trailing_face((y + RADIUS) - IMAGE_SIZE)).rg;
                imageStore(image_out, ivec3(x, y, face), vec4(accumulator * per_texel, 0.0, 0.0));
                accumulator -= imageLoad(image, ivec3(x, y - RADIUS, face)).rg;
            }
        }
        "#
    );

    let shader = Shader::new_compute(comp.as_slice()).specialization(
        SpecializationMap::new(bytes_of(&Blur {
            image_size: CUBEMAP_SIZE,
            radius: BLUR_RADIUS,
        }))
        .constant(0, 0, 4)
        .constant(1, 4, 4),
    );

    ComputePipeline::create(device, ComputePipelineInfo::default(), shader)
}

fn create_debug_pipeline(device: &Device) -> Result<GraphicsPipeline, DriverError> {
    let vert = glsl!(
        r#"
        #version 450 core
        #pragma shader_stage(vertex)

        layout(push_constant) uniform PushConstants {
            layout(offset = 0) mat4 model;
        } push_const;

        layout(binding = 0) uniform Camera {
            mat4 view;
            mat4 projection;
        } camera;

        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;

        layout(location = 0) out vec3 world_position_out;
        layout(location = 1) out vec3 world_normal_out;

        void main() {
            world_position_out = (push_const.model * vec4(position, 1)).xyz;
            world_normal_out = normalize((push_const.model * vec4(normal, 1)).xyz);
            gl_Position = camera.projection * camera.view * vec4(world_position_out, 1);
        }
        "#
    );
    let frag = glsl!(
        r#"
        #version 450 core
        #pragma shader_stage(fragment)

        layout(binding = 1) uniform Light {
            vec3 position;
            float range;
            mat4 view;
            mat4 projection;
        } light;

        layout(location = 0) in vec3 world_position;
        layout(location = 1) in vec3 world_normal;

        layout(location = 0) out vec4 color_out;

        void main() {
            color_out = vec4(vec3(0.0), 1.0);

            vec3 light_dir = light.position - world_position.xyz;
            float light_dist = length(light_dir);

            if (light_dist < light.range) {
                light_dir = normalize(light_dir);

                float lambertian = max(0.0, dot(world_normal, light_dir));
                float attenuation = max(0.0, min(1.0, light_dist / light.range));
                attenuation = 1.0 - attenuation * attenuation;

                color_out.rgb = vec3(lambertian * attenuation);
            }
        }
        "#
    );

    GraphicsPipeline::create(
        device,
        GraphicsPipelineInfo::default(),
        [
            Shader::new_vertex(vert.as_slice()),
            Shader::new_fragment(frag.as_slice()),
        ],
    )
}

fn create_mesh_pipeline(device: &Device) -> Result<GraphicsPipeline, DriverError> {
    let vert = glsl!(
        r#"
        #version 450 core
        #pragma shader_stage(vertex)

        layout(push_constant) uniform PushConstants {
            layout(offset = 0) mat4 model;
        } push_const;

        layout(binding = 0) uniform Camera {
            mat4 view;
            mat4 projection;
        } camera;

        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;

        out gl_PerVertex {
            vec4 gl_Position;
        };

        layout(location = 0) out vec4 world_position_out;
        layout(location = 1) out vec3 world_normal_out;

        void main() {
            world_normal_out = normalize((push_const.model * vec4(normal, 1.0)).xyz);
            world_position_out = push_const.model * vec4(position, 1.0);
            gl_Position = camera.projection * camera.view * world_position_out;
        }
        "#
    );
    let frag = glsl!(
        r#"
        #version 450 core
        #pragma shader_stage(fragment)

        #define EPSILON 0.0001
        #define MIN_SHADOW 0.1

        layout(constant_id = 0) const float BIAS = 0.15;

        layout(binding = 1) uniform Light {
            vec3 position;
            float range;
            mat4 view;
            mat4 projection;
        } light;

        layout(binding = 2) uniform samplerCube shadow_map;

        layout(location = 0) in vec4 world_position;
        layout(location = 1) in vec3 world_normal;

        layout(location = 0) out vec4 color_out;

        float upper_bound_shadow(vec2 moments, float scene_depth) {
            float p = step(scene_depth, moments.x + BIAS); // eliminates cubemap boundary thin line
            // 0 if moments.x < scene_depth; 1 if otherwise

            float variance = max(moments.y - moments.x * moments.x, EPSILON);
            // depth^2 - mean^2
            // ensure it as a denominator is not zero

            float dist = scene_depth - moments.x;
            float p_max = variance / (variance + dist * dist);

            return max(p, p_max);
        }

        float sample_shadow(samplerCube shadow_map, vec3 light, float scene_depth) {
            vec2 moments = texture(shadow_map, light).rg;
            // moments.r is mean, moments.g is depth^2

            return upper_bound_shadow(moments, scene_depth);
        }

        void main() {
            color_out = vec4(0.0, 0.0, 0.0, 1.0);

            vec3 light_dir = light.position - world_position.xyz;
            float light_dist = length(light_dir);

            if (light_dist < light.range) {
                light_dir = normalize(light_dir);

                float shadow = sample_shadow(shadow_map, -light_dir, light_dist);
                float lambertian = max(0.0, dot(world_normal, light_dir));
                float attenuation = max(0.0, min(1.0, light_dist / light.range));
                attenuation = 1.0 - attenuation * attenuation;

                // Make shadows not fully dark
                shadow = max(MIN_SHADOW, shadow);

                color_out.rgb = vec3(attenuation * lambertian * shadow);
            }
        }
        "#
    );

    GraphicsPipeline::create(
        device,
        GraphicsPipelineInfo::default(),
        [
            Shader::new_vertex(vert.as_slice()),
            Shader::new_fragment(frag.as_slice())
                .specialization(SpecializationMap::new(bytes_of(&SHADOW_BIAS)).constant(0, 0, 4)),
        ],
    )
}

fn create_shadow_pipeline(device: &Device) -> Result<GraphicsPipeline, DriverError> {
    let vert = glsl!(
        r#"
        #version 450 core
        #pragma shader_stage(vertex)

        layout(push_constant) uniform PushConstants {
            layout(offset = 0) mat4 model;
        } push_const;

        layout(binding = 0) uniform Light {
            vec3 position;
            float range;
            mat4 view;
            mat4 projection;
        } light;

        layout(location = 0) in vec3 position;

        layout(location = 0) out vec3 world_position_out;

        out gl_PerVertex {
            vec4 gl_Position;
        };

        void main(void) {
            vec4 world_position = push_const.model * vec4(position, 1.0);

            gl_Position = light.projection * light.view * world_position;
            world_position_out = world_position.xyz;
        }
        "#
    );
    let frag = glsl!(
        r#"
        #version 450 core
        #pragma shader_stage(fragment)

        layout(binding = 0) uniform Light {
            vec3 position;
            float range;
            mat4 view;
            mat4 projection;
        } light;

        layout(location = 0) in vec3 world_position;

        layout(location = 0) out vec4 color_out;

        void main() {
            float dist = distance(world_position.xyz, light.position);
            color_out.x = dist;
            color_out.y = dist * dist;
        }
        "#
    );

    GraphicsPipeline::create(
        device,
        GraphicsPipelineInfo::default(),
        [
            Shader::new_vertex(vert.as_slice()),
            Shader::new_fragment(frag.as_slice()),
        ],
    )
}

fn create_shadow_pipeline_with_geometry_shader(
    device: &Device,
) -> Result<GraphicsPipeline, DriverError> {
    let vert = glsl!(
        r#"
        #version 450 core
        #pragma shader_stage(vertex)

        #define TAN_HALF_FOVY 1.0 // tan(45deg)
        #define ASPECT_RATIO 1.0
        #define LIGHT_FRUSTUM_NEAR 0.1

        struct ViewPositions {
            vec4 positions[6];
        };

        layout(push_constant) uniform PushConstants {
            layout(offset = 0) mat4 model;
        } push_const;

        layout(binding = 0) uniform Light {
            vec3 position;
            float range;
            mat4 view;
            mat4 projection;
        } light;

        layout(location = 0) in vec3 position;

        layout(location = 0) out vec3 world_position_out;
        layout(location = 1) out uint layer_mask_out;
        layout(location = 2) out ViewPositions view_positions_out;

        uint get_layer_flag(vec4 view_pos, uint flag) {
            // if view_pos is in frustum, return flag
            // otherwise return 0

            uint res = 1;
            res *= uint(step(LIGHT_FRUSTUM_NEAR, -view_pos.z));
            res *= uint(step(-light.range, view_pos.z));

            float ymax = -TAN_HALF_FOVY * view_pos.z;
            float xmax = ymax * ASPECT_RATIO;
            res *= uint(step(-xmax, view_pos.x));
            res *= uint(step(view_pos.x, xmax));
            res *- uint(step(-ymax, view_pos.y));
            res *- uint(step(view_pos.y, ymax));

            return res * flag;
        }

        void main(void) {
            vec4 world_position = push_const.model * vec4(position, 1.0);
            world_position_out = world_position.xyz;

            // posx
            vec4 view0_pos = light.view * world_position;
            view0_pos.x *= -1.0;

            // negx
            vec4 view1_pos = vec4(-view0_pos.x, view0_pos.y, -view0_pos.z, 1.0);

            // posy
            vec4 view2_pos = vec4(-view0_pos.z, view0_pos.x, -view0_pos.y, 1.0);

            // negy
            vec4 view3_pos = vec4(-view0_pos.z, -view0_pos.x, view0_pos.y, 1.0);

            // posz
            vec4 view4_pos = vec4(-view0_pos.z, view0_pos.y, view0_pos.x, 1.0);

            // negz
            vec4 view5_pos = vec4(view0_pos.z, view0_pos.y, -view0_pos.x, 1.0);

            layer_mask_out = get_layer_flag(view0_pos, 0x01)
                           | get_layer_flag(view1_pos, 0x02)
                           | get_layer_flag(view2_pos, 0x04)
                           | get_layer_flag(view3_pos, 0x08)
                           | get_layer_flag(view4_pos, 0x16)
                           | get_layer_flag(view5_pos, 0x64);

            view_positions_out.positions[0] = view0_pos;
            view_positions_out.positions[1] = view1_pos;
            view_positions_out.positions[2] = view2_pos;
            view_positions_out.positions[3] = view3_pos;
            view_positions_out.positions[4] = view4_pos;
            view_positions_out.positions[5] = view5_pos;
        }
        "#
    );
    let geom = glsl!(
        r#"
        #version 450 core
        #pragma shader_stage(geometry)

        const mat4 CLIP = mat4(
            1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0
        );

        struct ViewPositions {
            vec4 positions[6];
        };

        layout(binding = 0) uniform Light {
            vec3 position;
            float range;
            mat4 view;
            mat4 projection;
        } light;

        layout(triangles) in;
        layout(location = 0) in vec3 world_positions[];
        layout(location = 1) in uint layer_mask[];
        layout(location = 2) in ViewPositions view_positions[];

        out gl_PerVertex {
            vec4 gl_Position;
        };

        layout(triangle_strip, max_vertices = 18) out;
        layout(location = 0) out vec3 world_position_out;

        void emit(uint flag, int view_idx) {
            uint layer_flag = (layer_mask[0] | layer_mask[1] | layer_mask[2]) & flag;

            if (layer_flag > 0) {
                gl_Position = CLIP * light.projection * view_positions[0].positions[view_idx];
                world_position_out = world_positions[0];
                EmitVertex();

                gl_Position = CLIP * light.projection * view_positions[1].positions[view_idx];
                world_position_out = world_positions[1];
                EmitVertex();

                gl_Position = CLIP * light.projection * view_positions[2].positions[view_idx];
                world_position_out = world_positions[2];
                EmitVertex();

                EndPrimitive();
            }
        }

        void main() {
            gl_Layer = 0;
            emit(0x01, 0);

            gl_Layer = 1;
            emit(0x02, 1);

            gl_Layer = 2;
            emit(0x04, 2);

            gl_Layer = 3;
            emit(0x08, 3);

            gl_Layer = 4;
            emit(0x16, 4);

            gl_Layer = 5;
            emit(0x64, 5);
        }
        "#
    );
    let frag = glsl!(
        r#"
        #version 450 core
        #pragma shader_stage(fragment)

        layout(binding = 0) uniform Light {
            vec3 position;
            float range;
            mat4 view;
            mat4 projection;
        } light;

        layout(location = 0) in vec3 world_position;

        layout(location = 0) out vec4 color_out;

        void main() {
            float dist = distance(world_position.xyz, light.position);
            color_out.x = dist;
            color_out.y = dist * dist;
        }
        "#
    );

    let info = GraphicsPipelineInfo::default();

    GraphicsPipeline::create(
        device,
        info,
        [
            Shader::new_vertex(vert.as_slice()),
            Shader::new_geometry(geom.as_slice()),
            Shader::new_fragment(frag.as_slice()),
        ],
    )
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

fn lease_uniform_buffer(
    pool: &mut impl Pool<BufferInfo, Buffer>,
    data: &impl NoUninit,
) -> Result<Lease<Buffer>, DriverError> {
    let data = bytes_of(data);
    let mut buf = pool.resource(BufferInfo::host_mem(
        data.len() as _,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
    ))?;
    buf.copy_from_slice(0, data);

    Ok(buf)
}

/// Returns vertices of a cube where the faces face inside
fn load_cube_data() -> [[f32; 6]; 36] {
    const N: f32 = -1f32;
    const P: f32 = 1f32;
    const Z: f32 = 0f32;

    const LEFT_BOTTOM_BACK: [f32; 3] = [N, N, P];
    const LEFT_BOTTOM_FRONT: [f32; 3] = [N, N, N];
    const LEFT_TOP_BACK: [f32; 3] = [N, P, P];
    const LEFT_TOP_FRONT: [f32; 3] = [N, P, N];
    const RIGHT_BOTTOM_BACK: [f32; 3] = [P, N, P];
    const RIGHT_BOTTOM_FRONT: [f32; 3] = [P, N, N];
    const RIGHT_TOP_BACK: [f32; 3] = [P, P, P];
    const RIGHT_TOP_FRONT: [f32; 3] = [P, P, N];

    const FORWARD: [f32; 3] = [Z, Z, P];
    const BACKWARD: [f32; 3] = [Z, Z, N];
    const LEFTWARD: [f32; 3] = [N, Z, Z];
    const RIGHTWARD: [f32; 3] = [P, Z, Z];
    const UPWARD: [f32; 3] = [Z, P, Z];
    const DOWNWARD: [f32; 3] = [Z, N, Z];

    const fn vertex(position: [f32; 3], normal: [f32; 3]) -> [f32; 6] {
        [
            position[0],
            position[1],
            position[2],
            normal[0],
            normal[1],
            normal[2],
        ]
    }

    [
        // Triangle 0
        vertex(LEFT_TOP_BACK, BACKWARD),
        vertex(LEFT_BOTTOM_BACK, BACKWARD),
        vertex(RIGHT_TOP_BACK, BACKWARD),
        // Triangle 1
        vertex(RIGHT_TOP_BACK, BACKWARD),
        vertex(LEFT_BOTTOM_BACK, BACKWARD),
        vertex(RIGHT_BOTTOM_BACK, BACKWARD),
        // // Triangle 2
        vertex(RIGHT_TOP_FRONT, FORWARD),
        vertex(RIGHT_BOTTOM_FRONT, FORWARD),
        vertex(LEFT_TOP_FRONT, FORWARD),
        // Triangle 3
        vertex(LEFT_TOP_FRONT, FORWARD),
        vertex(RIGHT_BOTTOM_FRONT, FORWARD),
        vertex(LEFT_BOTTOM_FRONT, FORWARD),
        // Triangle 4
        vertex(LEFT_TOP_FRONT, RIGHTWARD),
        vertex(LEFT_BOTTOM_FRONT, RIGHTWARD),
        vertex(LEFT_TOP_BACK, RIGHTWARD),
        // Triangle 5
        vertex(LEFT_TOP_BACK, RIGHTWARD),
        vertex(LEFT_BOTTOM_FRONT, RIGHTWARD),
        vertex(LEFT_BOTTOM_BACK, RIGHTWARD),
        // Triangle 6
        vertex(RIGHT_TOP_BACK, LEFTWARD),
        vertex(RIGHT_BOTTOM_BACK, LEFTWARD),
        vertex(RIGHT_TOP_FRONT, LEFTWARD),
        // Triangle 7
        vertex(RIGHT_TOP_FRONT, LEFTWARD),
        vertex(RIGHT_BOTTOM_BACK, LEFTWARD),
        vertex(RIGHT_BOTTOM_FRONT, LEFTWARD),
        // Triangle 8
        vertex(LEFT_BOTTOM_BACK, UPWARD),
        vertex(LEFT_BOTTOM_FRONT, UPWARD),
        vertex(RIGHT_BOTTOM_BACK, UPWARD),
        // Triangle 9
        vertex(RIGHT_BOTTOM_BACK, UPWARD),
        vertex(LEFT_BOTTOM_FRONT, UPWARD),
        vertex(RIGHT_BOTTOM_FRONT, UPWARD),
        // Triangle 10
        vertex(LEFT_TOP_FRONT, DOWNWARD),
        vertex(LEFT_TOP_BACK, DOWNWARD),
        vertex(RIGHT_TOP_FRONT, DOWNWARD),
        // Triangle 11
        vertex(RIGHT_TOP_FRONT, DOWNWARD),
        vertex(LEFT_TOP_BACK, DOWNWARD),
        vertex(RIGHT_TOP_BACK, DOWNWARD),
    ]
}

/// Loads a cube as indexed position and normal vertices
fn load_cube_mesh(device: &Device) -> Result<Model, DriverError> {
    let vertices = load_cube_data();
    let indices = (0u32..vertices.len() as u32).collect::<Vec<_>>();

    let index_buf = Arc::new(Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::INDEX_BUFFER,
        cast_slice(indices.as_slice()),
    )?);
    let vertex_buf = Arc::new(Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        cast_slice(vertices.as_slice()),
    )?);

    Ok(Model {
        index_buf,
        index_count: indices.len() as _,
        vertex_buf,
    })
}

/// Loads a cube as indexed position vertices
fn load_cube_shadow(device: &Device) -> Result<Model, DriverError> {
    let vertices = load_cube_data()
        .iter()
        .map(|vertex| [vertex[0], vertex[1], vertex[2]])
        .collect::<Vec<_>>();
    let indices = (0u32..vertices.len() as u32).collect::<Vec<_>>();

    let index_buf = Arc::new(Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::INDEX_BUFFER,
        cast_slice(indices.as_slice()),
    )?);
    let vertex_buf = Arc::new(Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        cast_slice(vertices.as_slice()),
    )?);

    Ok(Model {
        index_buf,
        index_count: indices.len() as _,
        vertex_buf,
    })
}

fn load_model<T>(
    device: &Device,
    path: impl AsRef<Path>,
    face_fn: fn(a: Vec3, b: Vec3, c: Vec3) -> [T; 3],
) -> anyhow::Result<Model>
where
    T: Default + NoUninit,
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
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
        cast_slice(&indices),
    )?);
    let vertex_buf = Arc::new(Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        cast_slice(&vertices),
    )?);

    Ok(Model {
        index_buf,
        index_count: indices.len() as _,
        vertex_buf,
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

/// Loads an .obj model as indexed position vertices
fn load_model_shadow(device: &Device, path: impl AsRef<Path>) -> anyhow::Result<Model> {
    #[repr(C)]
    #[derive(Clone, Copy, Default, Pod, Zeroable)]
    struct Vertex {
        position: Vec3,
    }

    load_model(device, path, |a, b, c| {
        // Make faces CCW
        [
            Vertex { position: a },
            Vertex { position: c },
            Vertex { position: b },
        ]
    })
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,

    /// Use geometry shader for shadow rendering (if supported)
    #[arg(long)]
    geometry_shader: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Blur {
    image_size: u32,
    radius: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Camera {
    view: Mat4,
    projection: Mat4,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Light {
    position: Vec3,
    range: f32,
    view: Mat4,
    projection: Mat4,
}

struct Model {
    index_buf: Arc<Buffer>,
    index_count: u32,
    vertex_buf: Arc<Buffer>,
}
