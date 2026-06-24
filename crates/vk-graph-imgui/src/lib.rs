//! Dear ImGui renderer integration for `vk-graph`.

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

/// Common imports for applications using the ImGui integration.
pub mod prelude {
    pub use super::{Condition, Frame, ImGui, Image, ImageSource, Ui, imgui};
}

pub use imgui::{self, Condition, Ui};

type DrawCmdInfo = (usize, [f32; 4], usize, usize, TextureId);

use {
    bytemuck::cast_slice,
    imgui::{Context, DrawCmd, DrawCmdParams, TextureId},
    imgui_winit_support::{
        winit::{event::Event, window::Window},
        {HiDpiMode, WinitPlatform},
    },
    log::warn,
    std::{collections::HashMap, marker::PhantomData, sync::Arc, time::Duration},
    vk_graph::{
        Graph,
        cmd::{LoadOp, StoreOp},
        driver::{
            ash::vk,
            buffer::{Buffer, BufferInfo},
            device::Device,
            graphics::{BlendInfo, GraphicsPipeline, GraphicsPipelineInfo},
            image::{Image as DriverImage, ImageInfo},
            shader::Shader,
            sync::AccessType,
        },
        node::{AnyImageNode, ImageLeaseNode, ImageNode},
        pool::{Lease, Pool},
    },
    vk_shader_macros::include_glsl,
};

/// Dear ImGui renderer state backed by `vk-graph` resources.
#[derive(Debug)]
pub struct ImGui {
    context: Context,
    font_atlas_image: Option<Arc<Lease<DriverImage>>>,
    next_texture_id: usize,
    pipeline: GraphicsPipeline,
    platform: WinitPlatform,
    user_images: HashMap<TextureId, AnyImageNode>,
}

/// Frame-scoped helper for registering user images with ImGui draw commands.
pub struct Frame<'a> {
    next_texture_id: &'a mut usize,
    user_images: &'a mut HashMap<TextureId, AnyImageNode>,
}

/// A frame-scoped ImGui image registration.
///
/// Dropping this value releases the typed handle. The underlying texture binding remains valid
/// until the current ImGui frame is rendered, because ImGui consumes draw commands after UI code
/// returns.
pub struct Image<'a> {
    id: TextureId,
    _frame: PhantomData<&'a Frame<'a>>,
}

/// Image source accepted by [`Frame::image`].
#[derive(Clone, Copy, Debug)]
pub struct ImageSource {
    image: AnyImageNode,
}

fn supported_draw_cmd(draw_cmd: DrawCmd) -> Option<DrawCmdInfo> {
    match draw_cmd {
        DrawCmd::Elements {
            count,
            cmd_params:
                DrawCmdParams {
                    clip_rect,
                    idx_offset,
                    vtx_offset,
                    texture_id,
                    ..
                },
        } => Some((count, clip_rect, idx_offset, vtx_offset, texture_id)),
        DrawCmd::ResetRenderState => {
            warn!("unsupported imgui draw command: reset render state");
            None
        }
        DrawCmd::RawCallback { .. } => {
            warn!("unsupported imgui draw command: raw callback");
            None
        }
    }
}

impl<'a> Frame<'a> {
    /// Registers an image for use by ImGui widgets during this frame.
    pub fn image(&mut self, image: impl Into<ImageSource>) -> Image<'_> {
        let id = TextureId::new(*self.next_texture_id);
        *self.next_texture_id += 1;
        self.user_images.insert(id, image.into().image);

        Image {
            id,
            _frame: PhantomData,
        }
    }
}

impl Image<'_> {
    /// Returns the ImGui texture id for this image.
    pub const fn id(&self) -> TextureId {
        self.id
    }
}

impl Drop for Image<'_> {
    fn drop(&mut self) {}
}

impl From<AnyImageNode> for ImageSource {
    fn from(image: AnyImageNode) -> Self {
        Self { image }
    }
}

impl From<ImageLeaseNode> for ImageSource {
    fn from(image: ImageLeaseNode) -> Self {
        Self {
            image: image.into(),
        }
    }
}

impl From<ImageNode> for ImageSource {
    fn from(image: ImageNode) -> Self {
        Self {
            image: image.into(),
        }
    }
}

impl ImGui {
    /// Creates a new ImGui renderer for the given device.
    pub fn new(device: &Device) -> Self {
        let mut context = Context::create();
        let platform = WinitPlatform::new(&mut context);
        let pipeline = GraphicsPipeline::create(
            device,
            GraphicsPipelineInfo::builder()
                .blend(BlendInfo::PRE_MULTIPLIED_ALPHA)
                .cull_mode(vk::CullModeFlags::NONE),
            [
                Shader::new_vertex(include_glsl!("res/shader/imgui.vert").as_slice()),
                Shader::new_fragment(include_glsl!("res/shader/imgui.frag").as_slice()),
            ],
        )
        .expect("invalid imgui pipeline");

        Self {
            context,
            font_atlas_image: None,
            next_texture_id: 1,
            pipeline,
            platform,
            user_images: Default::default(),
        }
    }

    /*
    TODO: This produces an image which is RGBA8 UNORM and has STORAGE set. *We* don't need storage
    here and should instead ask the user what settings to give the output image.
    */
    /// Builds a frame, records the necessary draw commands, and returns the rendered image.
    pub fn draw<P>(
        &mut self,
        dt: f32,
        events: &[Event<()>],
        window: &Window,
        pool: &mut P,
        graph: &mut Graph,
        ui_func: impl FnOnce(&mut Frame<'_>, &mut Ui, &mut P, &mut Graph),
    ) -> ImageLeaseNode
    where
        P: Pool<BufferInfo, Buffer> + Pool<ImageInfo, DriverImage>,
    {
        let hidpi = self.platform.hidpi_factor();

        self.platform
            .attach_window(self.context.io_mut(), window, HiDpiMode::Default);

        if self.font_atlas_image.is_none() || self.platform.hidpi_factor() != hidpi {
            self.lease_font_atlas_image(pool, graph);
        }

        let io = self.context.io_mut();
        io.update_delta_time(Duration::from_secs_f32(dt));

        for event in events {
            self.platform.handle_event(io, window, event);
        }

        self.platform
            .prepare_frame(io, window)
            .expect("invalid imgui frame");

        // Let the caller draw the GUI and register graph images used by image widgets.
        let ui = self.context.frame();
        let mut frame = Frame {
            next_texture_id: &mut self.next_texture_id,
            user_images: &mut self.user_images,
        };

        ui_func(&mut frame, ui, pool, graph);

        self.platform.prepare_render(ui, window);
        let draw_data = self.context.render();

        let image = graph.bind_resource(
            pool.resource(ImageInfo::image_2d(
                window.inner_size().width,
                window.inner_size().height,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC, /* TODO: Make TRANSFER_SRC an
                                                          * "extra flags" */
            ))
            .expect("missing imgui output image")
            .with_debug_name("ImGui Output"),
        );
        let font_atlas_image = graph.bind_resource(
            self.font_atlas_image
                .as_ref()
                .expect("missing imgui font atlas image"),
        );
        let display_pos = draw_data.display_pos;
        let framebuffer_scale = draw_data.framebuffer_scale;

        graph.clear_color_image(image, [0f32; 4]);

        if draw_data.draw_lists_count() == 0 {
            return image;
        }

        for draw_list in draw_data.draw_lists() {
            let indices = cast_slice(draw_list.idx_buffer());
            let mut index_buf = pool
                .resource(BufferInfo::host_mem(
                    indices.len() as _,
                    vk::BufferUsageFlags::INDEX_BUFFER,
                ))
                .expect("missing imgui index buffer");

            {
                Buffer::mapped_slice_mut(&mut index_buf)[0..indices.len()].copy_from_slice(indices);
            }

            let index_buf = graph.bind_resource(index_buf);

            let vertices = draw_list.vtx_buffer();
            let vertex_buf_len = vertices.len() * 20;
            let mut vertex_buf = pool
                .resource(BufferInfo::host_mem(
                    vertex_buf_len as _,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                ))
                .expect("missing imgui vertex buffer");

            {
                let vertex_buf = Buffer::mapped_slice_mut(&mut vertex_buf);
                for (idx, vertex) in vertices.iter().enumerate() {
                    let offset = idx * 20;
                    vertex_buf[offset..offset + 8].copy_from_slice(cast_slice(&vertex.pos));
                    vertex_buf[offset + 8..offset + 16].copy_from_slice(cast_slice(&vertex.uv));
                    vertex_buf[offset + 16..offset + 20].copy_from_slice(&vertex.col);
                }
            }

            let vertex_buf = graph.bind_resource(vertex_buf);

            let draw_cmds = draw_list
                .commands()
                .filter_map(supported_draw_cmd)
                .collect::<Vec<_>>();

            let window_width =
                self.platform.hidpi_factor() as f32 / window.inner_size().width as f32;
            let window_height =
                self.platform.hidpi_factor() as f32 / window.inner_size().height as f32;

            for (index_count, clip_rect, first_index, vertex_offset, texture_id) in draw_cmds {
                let texture = self
                    .user_images
                    .get(&texture_id)
                    .copied()
                    .unwrap_or_else(|| font_atlas_image.into());
                let clip_rect = [
                    (clip_rect[0] - display_pos[0]) * framebuffer_scale[0],
                    (clip_rect[1] - display_pos[1]) * framebuffer_scale[1],
                    (clip_rect[2] - display_pos[0]) * framebuffer_scale[0],
                    (clip_rect[3] - display_pos[1]) * framebuffer_scale[1],
                ];
                let x = clip_rect[0].floor() as i32;
                let y = clip_rect[1].floor() as i32;
                let width = (clip_rect[2] - clip_rect[0]).ceil() as u32;
                let height = (clip_rect[3] - clip_rect[1]).ceil() as u32;

                graph
                    .begin_cmd()
                    .debug_name("imgui")
                    .bind_pipeline(&self.pipeline)
                    .resource_access(index_buf, AccessType::IndexBuffer)
                    .resource_access(vertex_buf, AccessType::VertexBuffer)
                    .shader_resource_access(
                        0,
                        texture,
                        AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                    )
                    .color_attachment_image(0, image, LoadOp::Load, StoreOp::Store)
                    .record_cmd(move |cmd| {
                        cmd.push_constants(0, &window_width.to_ne_bytes())
                            .push_constants(4, &window_height.to_ne_bytes())
                            .bind_index_buffer(index_buf, 0, vk::IndexType::UINT16)
                            .bind_vertex_buffer(0, vertex_buf, 0)
                            .set_scissor(
                                0,
                                &[vk::Rect2D {
                                    offset: vk::Offset2D { x, y },
                                    extent: vk::Extent2D { width, height },
                                }],
                            )
                            .draw_indexed(
                                index_count as _,
                                1,
                                first_index as _,
                                vertex_offset as _,
                                0,
                            );
                    });
            }
        }

        self.user_images.clear();
        self.next_texture_id = 1;

        image
    }

    fn lease_font_atlas_image<P>(&mut self, pool: &mut P, graph: &mut Graph)
    where
        P: Pool<BufferInfo, Buffer> + Pool<ImageInfo, DriverImage>,
    {
        use imgui::{FontConfig, FontGlyphRanges, FontSource};

        let hidpi_factor = self.platform.hidpi_factor();
        self.context.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        let font_size = (14.0 * hidpi_factor) as f32;
        let fonts = self.context.fonts();
        fonts.clear_fonts();
        fonts.add_font(&[
            FontSource::TtfData {
                data: include_bytes!("../res/font/roboto/roboto-regular.ttf"),
                size_pixels: font_size,
                config: Some(FontConfig {
                    rasterizer_multiply: 2.0,
                    glyph_ranges: FontGlyphRanges::japanese(),
                    ..FontConfig::default()
                }),
            },
            FontSource::TtfData {
                data: include_bytes!("../res/font/mplus-1p/mplus-1p-regular.ttf"),
                size_pixels: font_size,
                config: Some(FontConfig {
                    oversample_h: 2,
                    oversample_v: 2,
                    // Range of glyphs to rasterize
                    glyph_ranges: FontGlyphRanges::japanese(),
                    ..FontConfig::default()
                }),
            },
        ]);

        let texture = fonts.build_rgba32_texture(); // TODO: Fix fb channel writes and use alpha8!
        let temp_buf_len = texture.data.len();
        let mut temp_buf = pool
            .resource(BufferInfo::host_mem(
                temp_buf_len as _,
                vk::BufferUsageFlags::TRANSFER_SRC,
            ))
            .expect("missing imgui font atlas buffer");

        {
            let temp_buf = Buffer::mapped_slice_mut(&mut temp_buf);
            temp_buf[0..temp_buf_len].copy_from_slice(texture.data);
        }

        let temp_buf = graph.bind_resource(temp_buf);
        let image = pool
            .resource(ImageInfo::image_2d(
                texture.width,
                texture.height,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_DST,
            ))
            .expect("missing imgui font atlas image")
            .with_debug_name("ImGui Font Atlas");

        let image = graph.bind_resource(image);

        graph.copy_buffer_to_image(temp_buf, image);

        self.font_atlas_image = Some(graph.resource(image).clone());
    }
}

#[cfg(test)]
mod test {
    use super::supported_draw_cmd;
    use imgui::{DrawCmd, DrawCmdParams, TextureId};

    #[test]
    fn supported_draw_cmd_extracts_element_draws() {
        let draw_cmd = DrawCmd::Elements {
            count: 42,
            cmd_params: DrawCmdParams {
                clip_rect: [1.0, 2.0, 3.0, 4.0],
                texture_id: TextureId::new(7),
                vtx_offset: 5,
                idx_offset: 6,
            },
        };

        assert_eq!(
            supported_draw_cmd(draw_cmd),
            Some((42, [1.0, 2.0, 3.0, 4.0], 6, 5, TextureId::new(7)))
        );
    }

    #[test]
    fn supported_draw_cmd_skips_reset_render_state() {
        assert_eq!(supported_draw_cmd(DrawCmd::ResetRenderState), None);
    }
}
