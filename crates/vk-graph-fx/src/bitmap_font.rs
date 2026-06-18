use {
    anyhow::Context,
    bmfont::BMFont,
    bytemuck::{cast, cast_slice},
    glam::{Mat4, vec3},
    std::sync::Arc,
    vk_graph::{
        Graph,
        cmd::{LoadOp, StoreOp},
        driver::{
            ash::vk,
            buffer::{Buffer, BufferInfo},
            device::Device,
            graphics::{BlendInfo, GraphicsPipeline, GraphicsPipelineInfo},
            image::Image,
            shader::{Shader, SpecializationMap},
            sync::AccessType,
        },
        node::{AnyImageNode, ImageNode},
        pool::{Pool as _, lazy::LazyPool},
    },
    vk_shader_macros::include_glsl,
};

type Color = [u8; 4];

fn color_to_unorm(color: Color) -> [u8; 16] {
    cast([
        (color[0] as f32 / u8::MAX as f32).to_ne_bytes(),
        (color[1] as f32 / u8::MAX as f32).to_ne_bytes(),
        (color[2] as f32 / u8::MAX as f32).to_ne_bytes(),
        (color[3] as f32 / u8::MAX as f32).to_ne_bytes(),
    ])
}

/// Holds a decoded bitmap Font.
#[derive(Debug)]
pub struct BitmapFont {
    font: BMFont,
    pages: Vec<Arc<Image>>,
    pipeline: GraphicsPipeline,
    pool: LazyPool,
}

impl BitmapFont {
    /// Creates a bitmap font renderer from decoded font metadata and page images.
    pub fn new(
        device: &Device,
        font: BMFont,
        pages: impl Into<Vec<Arc<Image>>>,
    ) -> anyhow::Result<Self> {
        let pool = LazyPool::new(device);
        let pages = pages.into();
        let num_pages = pages.len() as u32;
        let pipeline = GraphicsPipeline::create(
            device,
            GraphicsPipelineInfo::builder().blend(BlendInfo::ALPHA),
            [
                Shader::new_vertex(include_glsl!("res/shader/graphics/font.vert").as_slice()),
                Shader::new_fragment(include_glsl!("res/shader/graphics/font.frag").as_slice())
                    .specialization(
                        SpecializationMap::new(num_pages.to_ne_bytes()).constant(0, 0, 4),
                    ),
            ],
        )
        .context("Unable to create bitmap font pipeline")?;

        Ok(Self {
            font,
            pages,
            pipeline,
            pool,
        })
    }

    // TODO: Add description and example showing layout area, top/bottom explanation, etc
    /// Returns the position and area, in pixels, required to render the given text.
    ///
    /// **_NOTE:_** The 'start' of the render area is at the zero coordinate, however it may extend
    /// into the negative x direction due to ligatures.
    pub fn measure(&self, text: &str) -> ([i32; 2], [u32; 2]) {
        let parse = self.font.parse(text);

        // TODO: Use if we enable parsing errors on bmfont library
        // if parse.is_err() {
        //     return (IVec2::ZERO, UVec2::ZERO);
        // }
        // let parse = parse.unwrap();

        let mut min_x = 0;
        let mut max_x = 0;
        let mut max_y = 0;
        for char in parse {
            if char.screen_rect.x < min_x {
                min_x = char.screen_rect.x;
            }

            let screen_x = char.screen_rect.max_x();
            if screen_x > max_x {
                max_x = screen_x;
            }

            let screen_y = char.screen_rect.max_y();
            if screen_y > max_y {
                max_y = screen_y;
            }
        }

        let position = [min_x, 0];
        let size = [(max_x - min_x) as _, max_y as _];

        (position, size)
    }

    /// Prints text at the given position using the default scale factor of `1.0`.
    pub fn print(
        &mut self,
        graph: &mut Graph,
        image: impl Into<AnyImageNode>,
        x: f32,
        y: f32,
        color: impl Into<BitmapGlyphColor>,
        text: impl AsRef<str>,
    ) {
        self.print_scale(graph, image, x, y, color, text, 1.0);
    }

    // TODO: Better API, but not sure what, probably builder-something
    /// Prints text at the given position using a caller-specified scale factor.
    #[allow(clippy::too_many_arguments)]
    pub fn print_scale(
        &mut self,
        graph: &mut Graph,
        image: impl Into<AnyImageNode>,
        x: f32,
        y: f32,
        color: impl Into<BitmapGlyphColor>,
        text: impl AsRef<str>,
        scale: f32,
    ) {
        self.print_scale_scissor(graph, image, x, y, color, text, scale, None);
    }

    // TODO: Better API, but not sure what, probably builder-something
    /// Prints text with an optional scissor rectangle and caller-specified scale factor.
    #[allow(clippy::too_many_arguments)]
    pub fn print_scale_scissor(
        &mut self,
        graph: &mut Graph,
        image: impl Into<AnyImageNode>,
        x: f32,
        y: f32,
        color: impl Into<BitmapGlyphColor>,
        text: impl AsRef<str>,
        scale: f32,
        scissor: Option<(i32, i32, u32, u32)>,
    ) {
        let color = color.into();
        let image = image.into();
        let text = text.as_ref();
        let image_info = graph.resource(image).info;
        let transform = Mat4::from_translation(vec3(-1.0, -1.0, 0.0))
            * Mat4::from_scale(vec3(2.0 * scale, 2.0 * scale, 1.0))
            * Mat4::from_translation(vec3(
                x / image_info.width as f32,
                y / image_info.height as f32,
                0.0,
            ));

        let vertex_buf_len = 120 * text.chars().count() as vk::DeviceSize;
        let mut vertex_buf = self
            .pool
            .resource(BufferInfo::host_mem(
                vertex_buf_len,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            ))
            .expect("missing bitmap font vertex buffer");

        let mut vertex_count = 0;

        {
            let vertex_buf =
                &mut Buffer::mapped_slice_mut(&mut vertex_buf)[0..vertex_buf_len as usize];

            let mut offset = 0;
            for (data, char) in self.font.parse(text).map(|char| (char.tessellate(), char)) {
                vertex_buf[offset..offset + 16].copy_from_slice(&data[0]);
                vertex_buf[offset + 20..offset + 36].copy_from_slice(&data[1]);
                vertex_buf[offset + 40..offset + 56].copy_from_slice(&data[2]);
                vertex_buf[offset + 60..offset + 76].copy_from_slice(&data[3]);
                vertex_buf[offset + 80..offset + 96].copy_from_slice(&data[4]);
                vertex_buf[offset + 100..offset + 116].copy_from_slice(&data[5]);

                let page_idx = char.page_index as i32;
                let page_idx = page_idx.to_ne_bytes();
                vertex_buf[offset + 16..offset + 20].copy_from_slice(&page_idx);
                vertex_buf[offset + 36..offset + 40].copy_from_slice(&page_idx);
                vertex_buf[offset + 56..offset + 60].copy_from_slice(&page_idx);
                vertex_buf[offset + 76..offset + 80].copy_from_slice(&page_idx);
                vertex_buf[offset + 96..offset + 100].copy_from_slice(&page_idx);
                vertex_buf[offset + 116..offset + 120].copy_from_slice(&page_idx);

                vertex_count += 6;
                offset += 120;
            }
        }

        let vertex_buf = graph.bind_resource(vertex_buf);

        let mut page_nodes: Vec<ImageNode> = Vec::with_capacity(self.pages.len());
        for page in self.pages.iter() {
            page_nodes.push(graph.bind_resource(page));
        }

        let mut cmd = graph
            .begin_cmd()
            .debug_name("text")
            .bind_pipeline(&self.pipeline)
            .resource_access(vertex_buf, AccessType::VertexBuffer)
            .color_attachment_image(0, image, LoadOp::Load, StoreOp::Store);

        for (idx, page_node) in page_nodes.iter().copied().enumerate() {
            let descriptor = (0, [idx as _]);
            cmd.set_shader_resource_access(
                descriptor,
                page_node,
                AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
            );
        }

        cmd.record_cmd(move |cmd| {
            if let Some((x, y, width, height)) = scissor {
                cmd.set_scissor(
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D { x, y },
                        extent: vk::Extent2D { width, height },
                    }],
                );
            }

            cmd.push_constants(0, cast_slice(&transform.to_cols_array()))
                .push_constants(64, &(1.0 / image_info.width as f32).to_ne_bytes())
                .push_constants(68, &(1.0 / image_info.height as f32).to_ne_bytes())
                .push_constants(80, &color_to_unorm(color.solid()))
                .push_constants(96, &color_to_unorm(color.outline()))
                .bind_vertex_buffer(0, vertex_buf, 0)
                .draw(vertex_count, 1, 0, 0);
        });
    }
}

/// Color selection modes for bitmap glyph rendering.
#[derive(Debug)]
pub enum BitmapGlyphColor {
    /// Render only the glyph outline color.
    Outline(Color),

    /// Render only the glyph fill color.
    Solid(Color),

    /// Render both glyph fill and outline colors.
    SolidOutline(Color, Color),
}

impl BitmapGlyphColor {
    const TRANSPARENT: Color = [0, 0, 0, u8::MAX];

    fn outline(&self) -> Color {
        match self {
            Self::Outline(color) => *color,
            _ => Self::TRANSPARENT,
        }
    }

    fn solid(&self) -> Color {
        match self {
            Self::Outline(_) => Self::TRANSPARENT,
            Self::Solid(color) => *color,
            Self::SolidOutline(color, _) => *color,
        }
    }
}

impl From<[f32; 3]> for BitmapGlyphColor {
    fn from(color: [f32; 3]) -> Self {
        Self::Solid([
            (color[0].clamp(0.0, 1.0) * u8::MAX as f32) as _,
            (color[1].clamp(0.0, 1.0) * u8::MAX as f32) as _,
            (color[2].clamp(0.0, 1.0) * u8::MAX as f32) as _,
            u8::MAX,
        ])
    }
}

impl From<[f32; 4]> for BitmapGlyphColor {
    fn from(color: [f32; 4]) -> Self {
        Self::Solid([
            (color[0].clamp(0.0, 1.0) * u8::MAX as f32) as _,
            (color[1].clamp(0.0, 1.0) * u8::MAX as f32) as _,
            (color[2].clamp(0.0, 1.0) * u8::MAX as f32) as _,
            (color[3].clamp(0.0, 1.0) * u8::MAX as f32) as _,
        ])
    }
}

impl From<[u8; 3]> for BitmapGlyphColor {
    fn from(color: [u8; 3]) -> Self {
        Self::Solid([color[0], color[1], color[2], u8::MAX])
    }
}

impl From<[u8; 4]> for BitmapGlyphColor {
    fn from(color: [u8; 4]) -> Self {
        Self::Solid(color)
    }
}

pub use bmfont::CharPosition as BitmapGlyph;

/// Common accessors for glyph geometry and atlas placement.
pub trait Glyph {
    fn page_height(&self) -> u32;
    fn page_width(&self) -> u32;
    fn page_x(&self) -> u32;
    fn page_y(&self) -> u32;
    fn screen_height(&self) -> f32;
    fn screen_width(&self) -> f32;
    fn screen_x(&self) -> f32;
    fn screen_y(&self) -> f32;

    fn tessellate(&self) -> [[u8; 16]; 6] {
        let x1 = self.screen_x();
        let y1 = self.screen_y();
        let x2 = self.screen_x() + self.screen_width();
        let y2 = self.screen_y() + self.screen_height();

        let u1 = self.page_x() as f32;
        let u2 = (self.page_x() + self.page_width()) as f32;
        let v1 = self.page_y() as f32;
        let v2 = (self.page_y() + self.page_height()) as f32;

        let x1 = x1.to_ne_bytes();
        let x2 = x2.to_ne_bytes();
        let y1 = y1.to_ne_bytes();
        let y2 = y2.to_ne_bytes();
        let u1 = u1.to_ne_bytes();
        let u2 = u2.to_ne_bytes();
        let v1 = v1.to_ne_bytes();
        let v2 = v2.to_ne_bytes();

        let mut top_left = [0u8; 16];
        top_left[0..4].copy_from_slice(&x1);
        top_left[4..8].copy_from_slice(&y1);
        top_left[8..12].copy_from_slice(&u1);
        top_left[12..16].copy_from_slice(&v1);

        let mut bottom_right = [0u8; 16];
        bottom_right[0..4].copy_from_slice(&x2);
        bottom_right[4..8].copy_from_slice(&y2);
        bottom_right[8..12].copy_from_slice(&u2);
        bottom_right[12..16].copy_from_slice(&v2);

        let mut top_right = [0u8; 16];
        top_right[0..4].copy_from_slice(&x2);
        top_right[4..8].copy_from_slice(&y1);
        top_right[8..12].copy_from_slice(&u2);
        top_right[12..16].copy_from_slice(&v1);

        let mut bottom_left = [0u8; 16];
        bottom_left[0..4].copy_from_slice(&x1);
        bottom_left[4..8].copy_from_slice(&y2);
        bottom_left[8..12].copy_from_slice(&u1);
        bottom_left[12..16].copy_from_slice(&v2);

        [
            // First triangle
            top_left,
            bottom_right,
            top_right,
            // Second triangle
            top_left,
            bottom_left,
            bottom_right,
        ]
    }
}

impl Glyph for BitmapGlyph {
    #[inline(always)]
    fn page_height(&self) -> u32 {
        self.page_rect.height
    }

    #[inline(always)]
    fn page_width(&self) -> u32 {
        self.page_rect.width
    }

    #[inline(always)]
    fn page_x(&self) -> u32 {
        debug_assert!(self.page_rect.x >= 0);

        self.page_rect.x as _
    }

    #[inline(always)]
    fn page_y(&self) -> u32 {
        debug_assert!(self.page_rect.y >= 0);

        self.page_rect.y as _
    }

    #[inline(always)]
    fn screen_height(&self) -> f32 {
        self.screen_rect.height as _
    }

    #[inline(always)]
    fn screen_width(&self) -> f32 {
        self.screen_rect.width as _
    }

    #[inline(always)]
    fn screen_x(&self) -> f32 {
        self.screen_rect.x as _
    }

    #[inline(always)]
    fn screen_y(&self) -> f32 {
        self.screen_rect.y as _
    }
}

#[cfg(test)]
mod test {
    use super::BitmapGlyphColor;

    #[test]
    fn glyph_color_from_f32_rgb_defaults_alpha_to_opaque() {
        let color = BitmapGlyphColor::from([0.5, 0.25, 1.0]);

        match color {
            BitmapGlyphColor::Solid([127, 63, 255, 255]) => {}
            other => panic!("unexpected glyph color: {other:?}"),
        }
    }

    #[test]
    fn glyph_color_from_f32_rgba_clamps_values() {
        let color = BitmapGlyphColor::from([2.0, -1.0, 0.5, 0.25]);

        match color {
            BitmapGlyphColor::Solid([255, 0, 127, 63]) => {}
            other => panic!("unexpected glyph color: {other:?}"),
        }
    }

    #[test]
    fn glyph_color_from_u8_rgb_defaults_alpha_to_opaque() {
        let color = BitmapGlyphColor::from([1, 2, 3]);

        match color {
            BitmapGlyphColor::Solid([1, 2, 3, 255]) => {}
            other => panic!("unexpected glyph color: {other:?}"),
        }
    }

    #[test]
    fn glyph_color_from_u8_rgba_preserves_channels() {
        let color = BitmapGlyphColor::from([1, 2, 3, 4]);

        match color {
            BitmapGlyphColor::Solid([1, 2, 3, 4]) => {}
            other => panic!("unexpected glyph color: {other:?}"),
        }
    }
}
