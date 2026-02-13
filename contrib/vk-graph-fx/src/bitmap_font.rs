use {
    anyhow::Context,
    bmfont::BMFont,
    bytemuck::{cast, cast_slice},
    glam::{vec3, Mat4},
    std::sync::Arc,
    vk_graph::prelude::*,
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
    cache: HashPool,
    font: BMFont,
    pages: Vec<Arc<Image>>,
    pipeline: Arc<GraphicPipeline>,
}

impl BitmapFont {
    pub fn new(
        device: &Arc<Device>,
        font: BMFont,
        pages: impl Into<Vec<Arc<Image>>>,
    ) -> anyhow::Result<Self> {
        let cache = HashPool::new(device);
        let pages = pages.into();
        let num_pages = pages.len() as u32;
        let pipeline = Arc::new(
            GraphicPipeline::create(
                device,
                GraphicPipelineInfoBuilder::default().blend(BlendMode::ALPHA),
                [
                    Shader::new_vertex(include_glsl!("res/shader/graphic/font.vert").as_slice()),
                    Shader::new_fragment(include_glsl!("res/shader/graphic/font.frag").as_slice())
                        .specialization_info(SpecializationInfo::new(
                            [vk::SpecializationMapEntry {
                                constant_id: 0,
                                offset: 0,
                                size: 4,
                            }],
                            num_pages.to_ne_bytes(),
                        )),
                ],
            )
            .context("Unable to create bitmap font pipeline")?,
        );

        Ok(Self {
            cache,
            font,
            pages,
            pipeline,
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

    pub fn print(
        &mut self,
        graph: &mut RenderGraph,
        image: impl Into<AnyImageNode>,
        x: f32,
        y: f32,
        color: impl Into<BitmapGlyphColor>,
        text: impl AsRef<str>,
    ) {
        self.print_scale(graph, image, x, y, color, text, 1.0);
    }

    // TODO: Better API, but not sure what, probably builder-something
    #[allow(clippy::too_many_arguments)]
    pub fn print_scale(
        &mut self,
        graph: &mut RenderGraph,
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
    #[allow(clippy::too_many_arguments)]
    pub fn print_scale_scissor(
        &mut self,
        graph: &mut RenderGraph,
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
        let image_info = graph.node_info(image);
        let transform = Mat4::from_translation(vec3(-1.0, -1.0, 0.0))
            * Mat4::from_scale(vec3(2.0 * scale, 2.0 * scale, 1.0))
            * Mat4::from_translation(vec3(
                x / image_info.width as f32,
                y / image_info.height as f32,
                0.0,
            ));

        let vertex_buf_len = 120 * text.chars().count() as vk::DeviceSize;
        let mut vertex_buf = self
            .cache
            .lease(BufferInfo::host_mem(
                vertex_buf_len,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            ))
            .unwrap();

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

        let vertex_buf = graph.bind_node(vertex_buf);

        let mut page_nodes: Vec<ImageNode> = Vec::with_capacity(self.pages.len());
        for page in self.pages.iter() {
            page_nodes.push(graph.bind_node(page));
        }

        let mut pass = graph
            .begin_pass("text")
            .bind_pipeline(&self.pipeline)
            .access_node(vertex_buf, AccessType::IndexBuffer)
            .load_color(0, image)
            .store_color(0, image);

        for (idx, page_node) in page_nodes.iter().enumerate() {
            pass = pass.read_descriptor((0, [idx as _]), *page_node);
        }

        pass.record_subpass(move |subpass, _| {
            if let Some((x, y, width, height)) = scissor {
                subpass.set_scissor(x, y, width, height);
            }

            subpass
                .push_constants(cast_slice(&transform.to_cols_array()))
                .push_constants_offset(64, &(1.0 / image_info.width as f32).to_ne_bytes())
                .push_constants_offset(68, &(1.0 / image_info.height as f32).to_ne_bytes())
                .push_constants_offset(80, &color_to_unorm(color.solid()))
                .push_constants_offset(96, &color_to_unorm(color.outline()))
                .bind_vertex_buffer(vertex_buf)
                .draw(vertex_count, 1, 0, 0);
        });
    }
}

pub enum BitmapGlyphColor {
    Outline(Color),
    Solid(Color),
    SolidOutline(Color, Color),
}

impl BitmapGlyphColor {
    const TRANSARENT: Color = [0, 0, 0, u8::MAX];

    fn outline(&self) -> Color {
        match self {
            Self::Outline(color) => *color,
            _ => Self::TRANSARENT,
        }
    }

    fn solid(&self) -> Color {
        match self {
            Self::Outline(_) => Self::TRANSARENT,
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
