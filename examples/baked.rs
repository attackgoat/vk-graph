//! Demonstrates a reusable [`CommandStream`](vk_graph::stream::CommandStream) pattern for a game-like
//! frame.
//!
//! The scene renders a spinning tetrahedron every frame, then invokes a prepared command stream for
//! static UI chrome. Press Enter to regenerate the stream with different pseudo-text geometry.
//!
//! This pattern is useful when part of the frame is expensive to build but cheap to run: static HUD
//! panels, editor chrome, minimap backgrounds, large text blocks, or cached vector/UI tessellation.
//! The parent graph stays dynamic while the prepared stream keeps its internal graph optimization
//! and leased recording resources.

use {
    bytemuck::{Pod, Zeroable, cast_slice},
    clap::Parser,
    glam::{Mat4, Vec3, vec3},
    rand::{Rng, SeedableRng, rngs::SmallRng},
    std::{mem::size_of, sync::Arc, time::Instant},
    vk_graph::{
        cmd::{ClearColorValue, LoadOp, StoreOp},
        driver::{
            DriverError,
            ash::vk,
            buffer::{Buffer, BufferInfo},
            graphics::{
                BlendInfo, DepthStencilInfo, GraphicsPipeline, GraphicsPipelineInfoBuilder,
            },
            image::ImageInfo,
            shader::Shader,
            sync::AccessType,
        },
        pool::{Pool as _, hash::HashPool},
        stream::{CommandStream, ImageArg},
    },
    vk_graph_window::{Window, WindowError, winit},
    vk_shader_macros::glsl,
    winit::{
        event::{ElementState, Event, WindowEvent},
        keyboard::{KeyCode, PhysicalKey},
    },
};

const UI_DESIGN_WIDTH: f32 = 1280.0;
const UI_DESIGN_HEIGHT: f32 = 720.0;

fn main() -> Result<(), WindowError> {
    pretty_env_logger::init();

    let args = Args::parse();
    let window = Window::builder()
        .debug(args.debug)
        .min_image_count(3)
        .build()?;

    let pipelines = Pipelines::create(&window.device)?;
    let tetrahedron = Tetrahedron::create(&window.device)?;
    let mut pool = HashPool::new(&window.device);
    let mut ui = UiLayer::new();
    let started_at = Instant::now();

    window.run(move |frame| {
        let swapchain_info = frame.graph.resource(frame.swapchain_image).info;
        let ui_dirty = frame.events.iter().any(enter_pressed)
            || ui.width != frame.width
            || ui.height != frame.height;

        if ui_dirty {
            ui.regenerate(
                &mut pool,
                &pipelines.ui,
                frame.width,
                frame.height,
                swapchain_info,
            )
            .unwrap();
        }

        let elapsed = started_at.elapsed().as_secs_f32();

        // Bind the current immutable UI generation into this frame's transient graph
        let tetrahedron_vtx_buf = frame.graph.bind_resource(&tetrahedron.vertex_buffer);
        let tetrahedron_idx_buf = frame.graph.bind_resource(&tetrahedron.index_buffer);
        let depth_image = frame.graph.bind_resource(
            pool.resource(ImageInfo::image_2d(
                frame.width,
                frame.height,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ))
            .unwrap(),
        );

        let model = Mat4::from_rotation_y(elapsed * 0.9) * Mat4::from_rotation_x(elapsed * 0.6);
        let view = Mat4::look_at_rh(vec3(0.0, 0.0, 3.2), Vec3::ZERO, Vec3::Y);

        // Square projection plus X correction keeps the model from stretching with the window
        let projection = Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 32.0);
        let aspect_correction = Mat4::from_scale(vec3(
            frame.height as f32 / frame.width.max(1) as f32,
            1.0,
            1.0,
        ));
        let transform = aspect_correction * projection * view * model;

        frame
            .graph
            .begin_cmd()
            .debug_name("spinning tetrahedron")
            .bind_pipeline(&pipelines.tetrahedron)
            .depth_stencil(DepthStencilInfo::DEPTH_WRITE_LESS)
            // Declaring access lets the graph schedule barriers around these buffer reads
            .resource_access(tetrahedron_vtx_buf, AccessType::VertexBuffer)
            .resource_access(tetrahedron_idx_buf, AccessType::IndexBuffer)
            .color_attachment_image(
                0,
                frame.swapchain_image,
                LoadOp::Clear(ClearColorValue::rgba(0.012, 0.012, 0.014, 1.0)),
                StoreOp::Store,
            )
            .depth_stencil_attachment_image(
                depth_image,
                LoadOp::CLEAR_ONE_STENCIL_ZERO,
                StoreOp::DontCare,
            )
            .record_cmd(move |cmd| {
                cmd.bind_vertex_buffer(0, tetrahedron_vtx_buf, 0)
                    .bind_index_buffer(tetrahedron_idx_buf, 0, vk::IndexType::UINT16)
                    .push_constants(0, cast_slice(&transform.to_cols_array()))
                    .draw_indexed(12, 1, 0, 0, 0);
            });

        if let Some(ui) = ui.stream.as_ref() {
            frame
                .graph
                .insert_cmd_stream(&ui.stream)
                .with_arg(ui.stream.args.output, frame.swapchain_image)
                .finish();
        }
    })
}

fn enter_pressed(event: &Event<()>) -> bool {
    let Event::WindowEvent {
        event: WindowEvent::KeyboardInput { event, .. },
        ..
    } = event
    else {
        return false;
    };

    event.state == ElementState::Pressed
        && matches!(event.physical_key, PhysicalKey::Code(KeyCode::Enter))
}

struct Pipelines {
    tetrahedron: GraphicsPipeline,
    ui: GraphicsPipeline,
}

impl Pipelines {
    fn create(device: &vk_graph::driver::device::Device) -> Result<Self, DriverError> {
        let tetrahedron = GraphicsPipeline::create(
            device,
            GraphicsPipelineInfoBuilder::default(),
            [
                Shader::new_vertex(
                    glsl!(
                        r#"
                        #version 460 core
                        #pragma shader_stage(vertex)

                        layout(push_constant) uniform PushConstants {
                            mat4 transform;
                        } pc;

                        layout(location = 0) in vec3 position;
                        layout(location = 1) in vec4 color;
                        layout(location = 0) out vec4 frag_color;

                        void main() {
                            gl_Position = pc.transform * vec4(position, 1.0);
                            frag_color = color;
                        }
                        "#
                    )
                    .as_slice(),
                ),
                Shader::new_fragment(
                    glsl!(
                        r#"
                        #version 460 core
                        #pragma shader_stage(fragment)

                        layout(location = 0) in vec4 frag_color;
                        layout(location = 0) out vec4 out_color;

                        void main() {
                            out_color = frag_color;
                        }
                        "#
                    )
                    .as_slice(),
                ),
            ],
        )?;

        let ui = GraphicsPipeline::create(
            device,
            GraphicsPipelineInfoBuilder::default()
                .blend(BlendInfo::ALPHA)
                .cull_mode(vk::CullModeFlags::NONE),
            [
                Shader::new_vertex(
                    glsl!(
                        r#"
                        #version 460 core
                        #pragma shader_stage(vertex)

                        layout(location = 0) in vec2 position;
                        layout(location = 1) in vec4 color;
                        layout(location = 0) out vec4 frag_color;

                        void main() {
                            gl_Position = vec4(position, 0.0, 1.0);
                            frag_color = color;
                        }
                        "#
                    )
                    .as_slice(),
                ),
                Shader::new_fragment(
                    glsl!(
                        r#"
                        #version 460 core
                        #pragma shader_stage(fragment)

                        layout(location = 0) in vec4 frag_color;
                        layout(location = 0) out vec4 out_color;

                        void main() {
                            out_color = frag_color;
                        }
                        "#
                    )
                    .as_slice(),
                ),
            ],
        )?;

        Ok(Self { tetrahedron, ui })
    }
}

struct UiLayer {
    stream: Option<UiStream>,
    width: u32,
    height: u32,
    seed: u64,
}

impl UiLayer {
    fn new() -> Self {
        Self {
            stream: None,
            width: 0,
            height: 0,
            seed: rand::rng().random(),
        }
    }

    fn regenerate(
        &mut self,
        pool: &mut HashPool,
        pipeline: &GraphicsPipeline,
        width: u32,
        height: u32,
        output_info: ImageInfo,
    ) -> Result<(), DriverError> {
        self.seed = self.seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        self.stream = Some(UiStream::prepare(
            pipeline,
            pool,
            self.seed,
            width,
            height,
            output_info,
        )?);
        self.width = width;
        self.height = height;

        Ok(())
    }
}

#[derive(Clone, Copy)]
struct UiArgs {
    output: ImageArg,
}

struct UiStream {
    stream: CommandStream<UiArgs>,
}

impl UiStream {
    fn prepare(
        pipeline: &GraphicsPipeline,
        pool: &mut HashPool,
        seed: u64,
        width: u32,
        height: u32,
        output_info: ImageInfo,
    ) -> Result<Self, DriverError> {
        let vertices = build_ui_vertices(seed, width, height);
        let vertex_count = vertices.len() as u32;
        let vertex_data = cast_slice(vertices.as_slice());
        let vertex_buf_info =
            BufferInfo::host_mem(vertex_data.len() as _, vk::BufferUsageFlags::VERTEX_BUFFER);
        let mut vertex_buf = pool.resource(vertex_buf_info)?;
        vertex_buf.copy_from_slice(0, vertex_data);

        let stream = CommandStream::prepare(pool, |stream| {
            let output = stream.arg(output_info);
            let vertex_node = stream.bind_resource(vertex_buf);

            stream
                .begin_cmd()
                .debug_name("cached UI stream")
                .bind_pipeline(pipeline)
                .resource_access(vertex_node, AccessType::VertexBuffer)
                .color_attachment_image(0, output, LoadOp::Load, StoreOp::Store)
                .record_cmd(move |cmd| {
                    cmd.bind_vertex_buffer(0, vertex_node, 0)
                        .draw(vertex_count, 1, 0, 0);
                });

            UiArgs { output }
        })?;

        Ok(Self { stream })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UiVertex {
    position: [f32; 2],
    color: [f32; 4],
}

fn build_ui_vertices(seed: u64, target_width: u32, target_height: u32) -> Vec<UiVertex> {
    let mut rng = SmallRng::seed_from_u64(seed);

    // Use one scale factor for both axes; extra window space remains empty instead of stretching
    let scale =
        (target_width as f32 / UI_DESIGN_WIDTH).min(target_height as f32 / UI_DESIGN_HEIGHT);
    let mut canvas = UiCanvas {
        vertices: Vec::new(),
        width: target_width,
        height: target_height,
        scale,
    };

    canvas.rect(Rect::new(0.0, 0.0, UI_DESIGN_WIDTH, 30.0), gray(0.22, 0.92));
    canvas.rect(Rect::new(0.0, 30.0, UI_DESIGN_WIDTH, 3.0), gray(0.55, 0.95));
    canvas.rect(Rect::new(0.0, 33.0, UI_DESIGN_WIDTH, 4.0), gray(0.06, 0.9));

    let mut x = 16.0;
    for width in [52.0, 38.0, 64.0, 45.0, 58.0] {
        canvas.rect(Rect::new(x, 8.0, width, 12.0), gray(0.72, 0.95));
        x += width + 18.0;
    }

    canvas.outline(Rect::new(1018.0, 6.0, 20.0, 18.0), 2.0, gray(0.78, 0.95));
    canvas.outline(Rect::new(1048.0, 6.0, 20.0, 18.0), 2.0, gray(0.78, 0.95));
    canvas.outline(Rect::new(1078.0, 6.0, 20.0, 18.0), 2.0, gray(0.78, 0.95));

    canvas.rect(Rect::new(16.0, 56.0, 378.0, 164.0), gray(0.13, 0.84));
    canvas.outline(Rect::new(16.0, 56.0, 378.0, 164.0), 4.0, gray(0.62, 0.9));
    canvas.rect(Rect::new(22.0, 62.0, 366.0, 20.0), gray(0.3, 0.95));

    for i in 0..4 {
        let y = 99.0 + i as f32 * 28.0;
        let mut x = 36.0;
        let target = rng.random_range(245.0..330.0);

        while x < target {
            let width = rng.random_range(7.0..18.0);
            canvas.outline(Rect::new(x, y, width, 15.0), 2.0, [0.05, 0.42, 1.0, 0.94]);
            x += width + rng.random_range(4.0..8.0);

            if rng.random_bool(0.18) {
                x += rng.random_range(8.0..15.0);
            }
        }
    }

    canvas.rect(Rect::new(420.0, 54.0, 132.0, 88.0), gray(0.18, 0.75));
    canvas.outline(Rect::new(420.0, 54.0, 132.0, 88.0), 3.0, gray(0.64, 0.88));
    canvas.rect(Rect::new(438.0, 74.0, 96.0, 10.0), gray(0.5, 0.88));
    canvas.rect(Rect::new(438.0, 94.0, 74.0, 10.0), gray(0.38, 0.88));
    canvas.rect(Rect::new(438.0, 114.0, 106.0, 10.0), gray(0.46, 0.88));

    canvas.vertices
}

fn gray(value: f32, alpha: f32) -> [f32; 4] {
    [value, value, value, alpha]
}

#[derive(Clone, Copy)]
struct Rect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl Rect {
    const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }
}

struct UiCanvas {
    vertices: Vec<UiVertex>,
    width: u32,
    height: u32,
    scale: f32,
}

impl UiCanvas {
    fn rect(&mut self, rect: Rect, color: [f32; 4]) {
        let rect = Rect::new(
            rect.x * self.scale,
            rect.y * self.scale,
            rect.w * self.scale,
            rect.h * self.scale,
        );
        let l = rect.x / self.width as f32 * 2.0 - 1.0;
        let r = (rect.x + rect.w) / self.width as f32 * 2.0 - 1.0;
        let t = rect.y / self.height as f32 * 2.0 - 1.0;
        let b = (rect.y + rect.h) / self.height as f32 * 2.0 - 1.0;

        self.vertices.extend_from_slice(&[
            UiVertex {
                position: [l, t],
                color,
            },
            UiVertex {
                position: [l, b],
                color,
            },
            UiVertex {
                position: [r, b],
                color,
            },
            UiVertex {
                position: [l, t],
                color,
            },
            UiVertex {
                position: [r, b],
                color,
            },
            UiVertex {
                position: [r, t],
                color,
            },
        ]);
    }

    fn outline(&mut self, rect: Rect, thickness: f32, color: [f32; 4]) {
        self.rect(Rect::new(rect.x, rect.y, rect.w, thickness), color);
        self.rect(
            Rect::new(rect.x, rect.y + rect.h - thickness, rect.w, thickness),
            color,
        );
        self.rect(Rect::new(rect.x, rect.y, thickness, rect.h), color);
        self.rect(
            Rect::new(rect.x + rect.w - thickness, rect.y, thickness, rect.h),
            color,
        );
    }
}

struct Tetrahedron {
    index_buffer: Arc<Buffer>,
    vertex_buffer: Arc<Buffer>,
}

impl Tetrahedron {
    fn create(device: &vk_graph::driver::device::Device) -> Result<Self, DriverError> {
        let vertices = [
            TetrahedronVertex::new([1.0, 1.0, 1.0], [0.0, 0.9, 0.9, 1.0]),
            TetrahedronVertex::new([-1.0, -1.0, 1.0], [0.9, 0.0, 0.78, 1.0]),
            TetrahedronVertex::new([-1.0, 1.0, -1.0], [0.82, 0.58, 0.12, 1.0]),
            TetrahedronVertex::new([1.0, -1.0, -1.0], [0.34, 0.2, 0.11, 1.0]),
        ];
        let indices = [0_u16, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2];

        Ok(Self {
            index_buffer: Arc::new(Buffer::create_from_slice(
                device,
                vk::BufferUsageFlags::INDEX_BUFFER,
                cast_slice(&indices),
            )?),
            vertex_buffer: Arc::new(Buffer::create_from_slice(
                device,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                cast_slice(&vertices),
            )?),
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TetrahedronVertex {
    position: [f32; 3],
    color: [f32; 4],
}

impl TetrahedronVertex {
    const fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        Self { position, color }
    }
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers.
    #[arg(long)]
    debug: bool,
}

const _: () = {
    assert!(size_of::<UiVertex>() == 24);
    assert!(size_of::<TetrahedronVertex>() == 28);
};
