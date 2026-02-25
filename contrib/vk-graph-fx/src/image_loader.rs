use {
    super::BitmapFont, anyhow::Context, bmfont::BMFont, log::info, std::sync::Arc,
    vk_graph_prelude::*, vk_shader_macros::include_glsl,
};

#[cfg(debug_assertions)]
use log::warn;

/// Describes the channels and pixel stride of an image format
#[derive(Clone, Copy, Debug)]
pub enum ImageFormat {
    /// TODO
    R8,

    /// TODO
    R8G8,

    /// TODO
    R8G8B8,

    /// TODO
    R8G8B8A8,
}

impl ImageFormat {
    fn stride(self) -> usize {
        match self {
            Self::R8 => 1,
            Self::R8G8 => 2,
            Self::R8G8B8 => 3,
            Self::R8G8B8A8 => 4,
        }
    }
}

/// TODO
#[derive(Debug)]
pub struct ImageLoader {
    pool: HashPool,
    _decode_r_rg: ComputePipeline,
    decode_rgb_rgba: ComputePipeline,

    /// TODO
    pub device: Device,
}

impl ImageLoader {
    /// TODO
    pub fn new(device: &Device) -> Result<Self, DriverError> {
        Ok(Self {
            pool: HashPool::new(device),
            _decode_r_rg: ComputePipeline::create(
                device,
                ComputePipelineInfo::default(),
                Shader::new_compute(
                    include_glsl!("res/shader/compute/decode_bitmap_r_rg.comp").as_slice(),
                ),
            )?,
            decode_rgb_rgba: ComputePipeline::create(
                device,
                ComputePipelineInfo::default(),
                Shader::new_compute(
                    include_glsl!("res/shader/compute/decode_bitmap_rgb_rgba.comp").as_slice(),
                ),
            )?,
            device: device.clone(),
        })
    }

    fn create_image(
        &self,
        format: ImageFormat,
        width: u32,
        height: u32,
        is_srgb: bool,
        is_temporary: bool,
    ) -> anyhow::Result<Arc<Image>> {
        let format = match format {
            ImageFormat::R8 | ImageFormat::R8G8 => {
                if is_temporary {
                    vk::Format::R8G8_UINT
                } else if is_srgb {
                    panic!("Unsupported format: R8G8_SRGB");
                } else {
                    vk::Format::R8G8_UNORM
                }
            }
            ImageFormat::R8G8B8 | ImageFormat::R8G8B8A8 => {
                if is_temporary {
                    vk::Format::R8G8B8A8_UINT
                } else if is_srgb {
                    vk::Format::R8G8B8A8_SRGB
                } else {
                    vk::Format::R8G8B8A8_UNORM
                }
            }
        };
        let usage = if is_temporary {
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
        } else {
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
        };

        Ok(Arc::new(
            Image::create(
                &self.device,
                ImageInfo::image_2d(width, height, format, usage),
            )
            .context("Unable to create new image")?,
        ))
    }

    /// TODO
    #[allow(clippy::too_many_arguments)]
    pub fn decode_bitmap(
        &mut self,
        queue_family_index: u32,
        queue_index: u32,
        pixels: &[u8],
        format: ImageFormat,
        width: u32,
        height: u32,
        is_srgb: bool,
    ) -> anyhow::Result<Arc<Image>> {
        info!(
            "decoding {}x{} {:?} bitmap ({} K)",
            width,
            height,
            format,
            pixels.len() / 1024
        );

        debug_assert!(
            pixels.len() >= format.stride() * (width * height) as usize,
            "insufficient data"
        );

        #[cfg(debug_assertions)]
        if pixels.len() > (format.stride() as u32 * width * height).next_multiple_of(4) as usize {
            warn!("unused data");
        }

        let mut graph = Graph::default();
        let image = graph.bind_resource(self.create_image(format, width, height, is_srgb, false)?);

        // Fill the image from the temporary buffer
        match format {
            ImageFormat::R8 => {
                // This format requires a conversion
                info!("Converting R to RG");
                todo!()
            }
            ImageFormat::R8G8B8 => {
                // This format requires a conversion
                //info!("Converting RGB to RGBA");

                let stride = width * format.stride() as u32;

                //trace!("{bitmap_width}x{bitmap_height} Stride={bitmap_stride}");

                let pixel_buf_stride = stride.next_multiple_of(12);
                let pixel_buf_len = (pixel_buf_stride * height) as vk::DeviceSize;

                //trace!("pixel_buf_len={pixel_buf_len} pixel_buf_stride={pixel_buf_stride}");

                // Lease a temporary buffer from the cache pool
                let mut pixel_buf = self.pool.lease(BufferInfo::host_mem(
                    pixel_buf_len,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ))?;

                {
                    let pixel_buf =
                        &mut Buffer::mapped_slice_mut(&mut pixel_buf)[0..pixel_buf_len as usize];

                    // Fill the temporary buffer with the bitmap pixels - it has a different stride
                    for y in 0..height {
                        let src_offset = y * stride;
                        let src = &pixels[src_offset as usize..(src_offset + stride) as usize];

                        let dst_offset = y * pixel_buf_stride;
                        let dst =
                            &mut pixel_buf[dst_offset as usize..(dst_offset + stride) as usize];

                        dst.copy_from_slice(src);
                    }
                }

                let pixel_buf = graph.bind_resource(pixel_buf);

                // We create a temporary storage image because SRGB support isn't wide enough to
                // have SRGB storage images directly
                let temp_image =
                    graph.bind_resource(self.create_image(format, width, height, false, true)?);

                // Copy host-local data in the buffer to the temporary buffer on the GPU and then
                // use a compute shader to decode it before copying it over the output image

                let dispatch_x = (width + 3) >> 2;
                let dispatch_y = height;
                graph
                    .begin_cmd()
                    .debug_name("Decode RGB image")
                    .bind_pipeline(&self.decode_rgb_rgba)
                    .shader_resource_access(0, pixel_buf, AccessType::ComputeShaderReadOther)
                    .shader_resource_access(1, temp_image, AccessType::ComputeShaderWrite)
                    .record_pipeline(move |pipeline, _| {
                        pipeline
                            .push_constants(0, &(pixel_buf_stride >> 2).to_ne_bytes())
                            .dispatch(dispatch_x, dispatch_y, 1);
                    })
                    .end_cmd()
                    .copy_image(temp_image, image);
            }
            ImageFormat::R8G8 | ImageFormat::R8G8B8A8 => {
                // Lease a temporary buffer from the pool
                let mut pixel_buf = self.pool.lease(BufferInfo::host_mem(
                    pixels.len() as _,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                ))?;

                {
                    // Fill the temporary buffer with the bitmap pixels
                    let pixel_buf = &mut Buffer::mapped_slice_mut(&mut pixel_buf)[0..pixels.len()];
                    pixel_buf.copy_from_slice(pixels);
                }

                let pixel_buf = graph.bind_resource(pixel_buf);
                graph.copy_buffer_to_image(pixel_buf, image);
            }
        }

        let image = graph.resource(image).clone();

        graph
            .resolve()
            .submit(&mut self.pool, queue_family_index, queue_index)?;

        Ok(image)
    }

    /// TODO
    pub fn decode_linear(
        &mut self,
        queue_family_index: u32,
        queue_index: u32,
        pixels: &[u8],
        format: ImageFormat,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Arc<Image>> {
        self.decode_bitmap(
            queue_family_index,
            queue_index,
            pixels,
            format,
            width,
            height,
            false,
        )
    }

    /// TODO
    pub fn decode_srgb(
        &mut self,
        queue_family_index: u32,
        queue_index: u32,
        pixels: &[u8],
        format: ImageFormat,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Arc<Image>> {
        self.decode_bitmap(
            queue_family_index,
            queue_index,
            pixels,
            format,
            width,
            height,
            true,
        )
    }

    /// TODO
    pub fn load_bitmap_font<'a>(
        &mut self,
        queue_family_index: u32,
        queue_index: u32,
        font: BMFont,
        pages: impl IntoIterator<Item = (&'a [u8], u32, u32)>,
    ) -> anyhow::Result<BitmapFont> {
        let pages = pages
            .into_iter()
            .map(|(pixels, width, height)| {
                self.decode_linear(
                    queue_family_index,
                    queue_index,
                    pixels,
                    ImageFormat::R8G8B8,
                    width,
                    height,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        BitmapFont::new(&self.device, font, pages)
    }
}
