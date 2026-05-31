use {
    super::{AttachmentIndex, cmd_ref::CommandRef, pipeline::PipelineCommand},
    crate::{
        driver::{
            graphic::{DepthStencilInfo, GraphicPipeline},
            image::ImageViewInfo,
            render_pass::ResolveMode,
        },
        node::{AnyBufferNode, AnyImageNode},
    },
    ash::vk,
    std::{cell::RefCell, ops::Deref, slice},
};

impl PipelineCommand<'_, GraphicPipeline> {
    /// Sets the `color_attachment` attachment index of the following render pass to the given
    /// `image`.
    ///
    /// Note: To use multi-sampled (MSAA) rendering, use an image created with a sample count
    /// greater than one.
    ///
    /// Note: The default view (the whole image) is used for `image`.
    ///
    /// See [_Render Pass_](https://docs.vulkan.org/spec/latest/chapters/renderpass.html)
    pub fn color_attachment_image(
        mut self,
        color_attachment: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        load: LoadOp<ClearColorValue>,
        store: StoreOp,
    ) -> Self {
        self.set_color_attachment_image(color_attachment, image, load, store);
        self
    }

    /// Sets the `color_attachment` attachment index of the following render pass to the given
    /// `image`, as interpreted by `image_view_info`.
    ///
    /// See [_Render Pass_](https://docs.vulkan.org/spec/latest/chapters/renderpass.html)
    pub fn color_attachment_image_view(
        mut self,
        color_attachment: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
        load: LoadOp<ClearColorValue>,
        store: StoreOp,
    ) -> Self {
        self.set_color_attachment_image_view(color_attachment, image, image_view_info, load, store);
        self
    }

    /// Resolves a multi-sampled (MSAA) color image attachment into a single-sampled attachment
    /// using the given `image`.
    ///
    /// Note: The default view (the whole image) is used for `image`.
    ///
    /// See [_Render Pass_](https://docs.vulkan.org/spec/latest/chapters/renderpass.html)
    pub fn color_attachment_resolve_image(
        mut self,
        msaa_attachment: AttachmentIndex,
        color_attachment: AttachmentIndex,
        image: impl Into<AnyImageNode>,
    ) -> Self {
        self.set_color_attachment_resolve_image(msaa_attachment, color_attachment, image);
        self
    }

    /// Resolves a multi-sampled (MSAA) color image attachment into a single-sampled attachment
    /// using the given `image`, as interpreted by `image_view_info`.
    ///
    /// See [_Render Pass_](https://docs.vulkan.org/spec/latest/chapters/renderpass.html)
    pub fn color_attachment_resolve_image_view(
        mut self,
        msaa_attachment: AttachmentIndex,
        color_attachment: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
    ) -> Self {
        self.set_color_attachment_resolve_image_view(
            msaa_attachment,
            color_attachment,
            image,
            image_view_info,
        );
        self
    }

    /// Sets the combined depth and stencil state used by any subsequent command buffer recordings
    /// of the current graph command.
    pub fn depth_stencil(mut self, depth_stencil: impl Into<DepthStencilInfo>) -> Self {
        self.set_depth_stencil(depth_stencil);
        self
    }

    /// Sets the combined depth and stencil attachment of the following render pass to the given
    /// `image`.
    ///
    /// Note: To use multi-sampled (MSAA) rendering, use an image created with a sample count
    /// greater than one.
    ///
    /// Note: The default view (the whole image) is used for `image`.
    ///
    /// See [_Render Pass_](https://docs.vulkan.org/spec/latest/chapters/renderpass.html)
    pub fn depth_stencil_attachment_image(
        mut self,
        image: impl Into<AnyImageNode>,
        load: LoadOp<vk::ClearDepthStencilValue>,
        store: StoreOp,
    ) -> Self {
        self.set_depth_stencil_attachment_image(image, load, store);
        self
    }

    /// Sets the combined depth and stencil attachment of the following render pass to the given
    /// `image`, as interpreted by `image_view_info`.
    ///
    /// Note: To use multi-sampled (MSAA) rendering, use an image created with a sample count
    /// greater than one.
    ///
    /// Note: The default view (the whole image) is used for `image`.
    ///
    /// See [_Render Pass_](https://docs.vulkan.org/spec/latest/chapters/renderpass.html)
    pub fn depth_stencil_attachment_image_view(
        mut self,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
        load: LoadOp<vk::ClearDepthStencilValue>,
        store: StoreOp,
    ) -> Self {
        self.set_depth_stencil_attachment_image_view(image, image_view_info, load, store);
        self
    }

    /// Resolves a multi-sampled (MSAA) combined depth and stencil image attachment into a
    /// single-sampled attachment using the given `image`.
    ///
    /// Note: The default view (the whole image) is used for `image`.
    ///
    /// See [_Render Pass_](https://docs.vulkan.org/spec/latest/chapters/renderpass.html)
    pub fn depth_stencil_attachment_resolve_image(
        mut self,
        depth_stencil_attachment: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        depth_mode: Option<ResolveMode>,
        stencil_mode: Option<ResolveMode>,
    ) -> Self {
        self.set_depth_stencil_attachment_resolve_image(
            depth_stencil_attachment,
            image,
            depth_mode,
            stencil_mode,
        );
        self
    }

    /// Resolves a multi-sampled (MSAA) combined depth and stencil image attachment into a
    /// single-sampled attachment using the given `image`, as interpreted by `image_view_info`.
    ///
    /// Note: The default view (the whole image) is used for `image`.
    ///
    /// See [_Render Pass_](https://docs.vulkan.org/spec/latest/chapters/renderpass.html)
    pub fn depth_stencil_attachment_resolve_image_view(
        mut self,
        depth_stencil_attachment: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
        depth_mode: Option<ResolveMode>,
        stencil_mode: Option<ResolveMode>,
    ) -> Self {
        self.set_depth_stencil_attachment_resolve_image_view(
            depth_stencil_attachment,
            image,
            image_view_info,
            depth_mode,
            stencil_mode,
        );
        self
    }

    /// Sets multiview view and correlation masks used by any subsequent command buffer recordings
    /// of the current graph command.
    ///
    /// See [`VkRenderPassMultiviewCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderPassMultiviewCreateInfo.html).
    pub fn multiview(mut self, view_mask: u32, correlated_view_mask: u32) -> Self {
        self.set_multiview(view_mask, correlated_view_mask);
        self
    }

    /// Begin recording a graphics pipeline command buffer.
    pub fn record_cmd(mut self, func: impl FnOnce(GraphicCommandRef<'_>) + Send + 'static) -> Self {
        self.record_cmd_mut(func);
        self
    }

    /// Begin recording a graphics pipeline command buffer.
    pub fn record_cmd_mut(&mut self, func: impl FnOnce(GraphicCommandRef<'_>) + Send + 'static) {
        let pipeline = self
            .cmd
            .cmd()
            .expect_last_pipeline()
            .expect_graphic()
            .clone();

        self.cmd.push_exec(move |cmd| {
            func(GraphicCommandRef { cmd, pipeline });
        });
    }

    /// See [`VkRenderPassBeginInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderPassBeginInfo.html).
    /// field when beginning a render pass used by any subsequent command buffer recordings
    /// of the current graph command.
    ///
    /// _NOTE:_ Setting this value will cause the viewport and scissor to be unset, which is not the
    /// default behavior. When this value is set you should call `set_viewport` and `set_scissor` on
    /// the command buffer.
    ///
    /// If not set, this value defaults to the first loaded, resolved, or stored attachment
    /// dimensions and sets the viewport and scissor to the same values, with a `0..1` depth if not
    /// specified by `depth_stencil`.
    pub fn render_area(mut self, area: vk::Rect2D) -> Self {
        self.set_render_area(area);
        self
    }

    /// See [`Self::color_attachment_image`]
    pub fn set_color_attachment_image(
        &mut self,
        color_attachment: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        load: LoadOp<ClearColorValue>,
        store: StoreOp,
    ) -> &mut Self {
        let image = image.into();
        let image_view = self.resource(image).info;

        self.set_color_attachment_image_view(color_attachment, image, image_view, load, store);

        self
    }

    /// See [`Self::color_attachment_image_view`]
    pub fn set_color_attachment_image_view(
        &mut self,
        color_attachment: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
        load: LoadOp<ClearColorValue>,
        store: StoreOp,
    ) -> &mut Self {
        let image = image.into();
        let image_view_info = image_view_info.into();

        #[allow(deprecated)]
        {
            match load {
                LoadOp::Clear(color) => {
                    self.set_clear_color_value_as(color_attachment, image, color, image_view_info)
                }
                LoadOp::DontCare => {
                    self.set_attach_color_as(color_attachment, image, image_view_info)
                }
                LoadOp::Load => self.set_load_color_as(color_attachment, image, image_view_info),
            };

            if let StoreOp::Store = store {
                self.set_store_color_as(color_attachment, image, image_view_info);
            }
        }

        self
    }

    /// See [`Self::color_attachment_resolve_image`]
    pub fn set_color_attachment_resolve_image(
        &mut self,
        msaa_attachment: AttachmentIndex,
        color_attachment: AttachmentIndex,
        image: impl Into<AnyImageNode>,
    ) -> &mut Self {
        let image = image.into();
        let image_view = self.resource(image).info;

        self.set_color_attachment_resolve_image_view(
            msaa_attachment,
            color_attachment,
            image,
            image_view,
        );

        self
    }

    /// See [`Self::color_attachment_resolve_image_view`]
    pub fn set_color_attachment_resolve_image_view(
        &mut self,
        msaa_attachment: AttachmentIndex,
        color_attachment: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
    ) -> &mut Self {
        let image = image.into();
        let image_view_info = image_view_info.into();

        #[allow(deprecated)]
        self.set_resolve_color_as(msaa_attachment, color_attachment, image, image_view_info);

        self
    }

    /// See [`Self::depth_stencil`]
    pub fn set_depth_stencil(&mut self, depth_stencil: impl Into<DepthStencilInfo>) -> &mut Self {
        let depth_stencil = depth_stencil.into();
        let cmd = self.cmd.cmd_mut();
        let exec = cmd.expect_last_exec_mut();

        assert!(exec.depth_stencil.is_none());

        exec.depth_stencil = Some(depth_stencil);

        self
    }

    /// See [`Self::depth_stencil_attachment_image`]
    pub fn set_depth_stencil_attachment_image(
        &mut self,
        image: impl Into<AnyImageNode>,
        load: LoadOp<vk::ClearDepthStencilValue>,
        store: StoreOp,
    ) -> &mut Self {
        let image = image.into();
        let image_view_info = self.resource(image).info;

        self.set_depth_stencil_attachment_image_view(image, image_view_info, load, store);

        self
    }

    /// See [`Self::depth_stencil_attachment_image_view`]
    pub fn set_depth_stencil_attachment_image_view(
        &mut self,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
        load: LoadOp<vk::ClearDepthStencilValue>,
        store: StoreOp,
    ) -> &mut Self {
        let image = image.into();
        let image_view_info = image_view_info.into();

        #[allow(deprecated)]
        {
            match load {
                LoadOp::Clear(color) => self.set_clear_depth_stencil_value_as(
                    image,
                    color.depth,
                    color.stencil,
                    image_view_info,
                ),
                LoadOp::DontCare => self.set_attach_depth_stencil_as(image, image_view_info),
                LoadOp::Load => self.set_load_depth_stencil_as(image, image_view_info),
            };

            if let StoreOp::Store = store {
                self.set_store_depth_stencil_as(image, image_view_info);
            }
        }

        self
    }

    /// See [`Self::depth_stencil_attachment_resolve_image`]
    pub fn set_depth_stencil_attachment_resolve_image(
        &mut self,
        depth_stencil_attachment: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        depth_mode: Option<ResolveMode>,
        stencil_mode: Option<ResolveMode>,
    ) -> &mut Self {
        let image = image.into();
        let image_view = self.resource(image).info;

        self.set_depth_stencil_attachment_resolve_image_view(
            depth_stencil_attachment,
            image,
            image_view,
            depth_mode,
            stencil_mode,
        );

        self
    }

    /// See [`Self::depth_stencil_attachment_resolve_image_view`]
    pub fn set_depth_stencil_attachment_resolve_image_view(
        &mut self,
        depth_stencil_attachment: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
        depth_mode: Option<ResolveMode>,
        stencil_mode: Option<ResolveMode>,
    ) -> &mut Self {
        let image = image.into();
        let image_view_info = image_view_info.into();

        #[allow(deprecated)]
        self.set_resolve_depth_stencil_as(
            depth_stencil_attachment,
            image,
            image_view_info,
            depth_mode,
            stencil_mode,
        );

        self
    }

    /// See [`Self::multiview`]
    pub fn set_multiview(&mut self, view_mask: u32, correlated_view_mask: u32) -> &mut Self {
        let cmd = self.cmd.cmd_mut();
        let exec = cmd.expect_last_exec_mut();

        exec.correlated_view_mask = correlated_view_mask;
        exec.view_mask = view_mask;

        self
    }

    /// See [`Self::render_area`]
    pub fn set_render_area(&mut self, area: vk::Rect2D) -> &mut Self {
        self.cmd.cmd_mut().expect_last_exec_mut().render_area = Some(area);
        self
    }
}

/// Structure specifying a clear color value.
#[derive(Clone, Copy, Debug)]
pub enum ClearColorValue {
    /// Value as [f32].
    ///
    /// Use this member for color clear values when the format of the image or attachment is one of
    /// the numeric formats with a numeric type that is floating-point. Floating-point values are
    /// automatically converted to the format of the image, with the clear value being treated as
    /// linear if the image is sRGB.
    Float32([f32; 4]),

    /// Value as [i32].
    ///
    /// Use this member for color clear values when the format of the image or attachment has a
    /// numeric type that is signed integer. Signed integer values are converted to the format of
    /// the image by casting to the smaller type (with negative 32-bit values mapping to negative
    /// values in the smaller type). If the integer clear value is not representable in the target
    /// type (e.g. would overflow in conversion to that type), the clear value is undefined.
    Int32([i32; 4]),

    /// Value as [u32].
    ///
    /// Use this member for color clear values when the format of the image or attachment has a
    /// numeric type that is unsigned integer. Unsigned integer values are converted to the format
    /// of the image by casting to the integer type with fewer bits.
    Uint32([u32; 4]),
}

impl ClearColorValue {
    /// RGB zeros and alpha ones.
    pub const BLACK_ALPHA_ONE: Self = Self::Float32([0.0, 0.0, 0.0, 1.0]);

    /// All zeros.
    pub const BLACK_ALPHA_ZERO: Self = Self::Float32([0.0, 0.0, 0.0, 0.0]);

    /// RGB zeros and alpha ones.
    pub const WHITE_ALPHA_ONE: Self = Self::Float32([1.0, 1.0, 1.0, 1.0]);

    /// RGB ones and alpha zeros.
    pub const WHITE_ALPHA_ZERO: Self = Self::Float32([1.0, 1.0, 1.0, 0.0]);

    /// Convenience constructor for clear color values.
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self::Float32([r, g, b, a])
    }

    /// Convert RGB+A values into a ClearColorValue.
    pub const fn from_f32(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self::rgba(r, g, b, a)
    }

    /// Convert RGB+A values into a ClearColorValue.
    pub const fn from_u8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self::from_f32(
            r as f32 / u8::MAX as f32,
            g as f32 / u8::MAX as f32,
            b as f32 / u8::MAX as f32,
            a as f32 / u8::MAX as f32,
        )
    }
}

impl Default for ClearColorValue {
    fn default() -> Self {
        Self::from_f32(0.0, 0.0, 0.0, 0.0)
    }
}

impl From<[f32; 4]> for ClearColorValue {
    fn from(float32: [f32; 4]) -> Self {
        Self::Float32(float32)
    }
}

impl From<[i32; 4]> for ClearColorValue {
    fn from(int32: [i32; 4]) -> Self {
        Self::Int32(int32)
    }
}

impl From<[u8; 4]> for ClearColorValue {
    fn from(uint8: [u8; 4]) -> Self {
        Self::from_u8(uint8[0], uint8[1], uint8[2], uint8[3])
    }
}

impl From<[u32; 4]> for ClearColorValue {
    fn from(uint32: [u32; 4]) -> Self {
        Self::Uint32(uint32)
    }
}

impl From<ClearColorValue> for vk::ClearColorValue {
    fn from(value: ClearColorValue) -> Self {
        match value {
            ClearColorValue::Float32(float32) => Self { float32 },
            ClearColorValue::Int32(int32) => Self { int32 },
            ClearColorValue::Uint32(uint32) => Self { uint32 },
        }
    }
}

/// Recording interface for drawing commands.
///
/// This structure provides a strongly-typed set of methods which allow raster graphics shader code
/// to be executed. An instance is provided to the closure argument of
/// [`PipelineCommand::record_cmd`] which may be accessed by binding a [`GraphicPipeline`] to a
/// command.
///
/// # Examples
///
/// Basic usage:
///
/// ```no_run
/// # use ash::vk;
/// # use vk_graph::cmd::{LoadOp, StoreOp};
/// # use vk_graph::driver::DriverError;
/// # use vk_graph::driver::device::{Device, DeviceInfo};
/// # use vk_graph::driver::graphic::{GraphicPipeline, GraphicPipelineInfo};
/// # use vk_graph::driver::image::{Image, ImageInfo};
/// # use vk_graph::Graph;
/// # use vk_graph::driver::shader::Shader;
/// # fn main() -> Result<(), DriverError> {
/// # let device = Device::new(DeviceInfo::default())?;
/// # let my_frag_code = [0u8; 1];
/// # let my_vert_code = [0u8; 1];
/// # let vert = Shader::new_vertex(my_vert_code.as_slice());
/// # let frag = Shader::new_fragment(my_frag_code.as_slice());
/// # let info = GraphicPipelineInfo::default();
/// # let my_graphic_pipeline = GraphicPipeline::create(&device, info, [vert, frag])?;
/// # let mut my_graph = Graph::default();
/// # let info = ImageInfo::image_2d(
/// #     32,
/// #     32,
/// #     vk::Format::R8G8B8A8_UNORM,
/// #     vk::ImageUsageFlags::SAMPLED,
/// # );
/// # let swapchain_image = my_graph.bind_resource(Image::create(&device, info)?);
/// my_graph
///     .begin_cmd()
///     .debug_name("my draw command")
///     .bind_pipeline(&my_graphic_pipeline)
///     .color_attachment_image(0, swapchain_image, LoadOp::DontCare, StoreOp::Store)
///     .record_cmd(move |cmd| {
///         // During this closure we have access to the drawing functions!
///         cmd.draw(3, 1, 0, 0);
///     });
/// # Ok(()) }
/// ```
pub struct GraphicCommandRef<'a> {
    cmd: CommandRef<'a>,
    pipeline: GraphicPipeline,
}

impl GraphicCommandRef<'_> {
    /// Bind an index buffer to the current command.
    ///
    /// `offset` is the starting offset in bytes within `buffer` used in index buffer address
    /// calculations.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use ash::vk;
    /// # use vk_graph::cmd::{LoadOp, StoreOp};
    /// # use vk_graph::driver::{AccessType, DriverError};
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::graphic::{GraphicPipeline, GraphicPipelineInfo};
    /// # use vk_graph::driver::image::{Image, ImageInfo};
    /// # use vk_graph::driver::shader::Shader;
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::new(DeviceInfo::default())?;
    /// # let my_frag_code = [0u8; 1];
    /// # let my_vert_code = [0u8; 1];
    /// # let vert = Shader::new_vertex(my_vert_code.as_slice());
    /// # let frag = Shader::new_fragment(my_frag_code.as_slice());
    /// # let info = GraphicPipelineInfo::default();
    /// # let my_graphic_pipeline = GraphicPipeline::create(&device, info, [vert, frag])?;
    /// # let mut my_graph = Graph::default();
    /// # let info = ImageInfo::image_2d(
    /// #     32,
    /// #     32,
    /// #     vk::Format::R8G8B8A8_UNORM,
    /// #     vk::ImageUsageFlags::SAMPLED,
    /// # );
    /// # let swapchain_image = my_graph.bind_resource(Image::create(&device, info)?);
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::INDEX_BUFFER);
    /// # let my_idx_buf = Buffer::create(&device, buf_info)?;
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::VERTEX_BUFFER);
    /// # let my_vtx_buf = Buffer::create(&device, buf_info)?;
    /// # let my_idx_buf = my_graph.bind_resource(my_idx_buf);
    /// # let my_vtx_buf = my_graph.bind_resource(my_vtx_buf);
    /// my_graph
    ///     .begin_cmd()
    ///     .debug_name("my indexed geometry draw pass")
    ///     .bind_pipeline(&my_graphic_pipeline)
    ///     .color_attachment_image(0, swapchain_image, LoadOp::DontCare, StoreOp::Store)
    ///     .resource_access(my_idx_buf, AccessType::IndexBuffer)
    ///     .resource_access(my_vtx_buf, AccessType::VertexBuffer)
    ///     .record_cmd(move |cmd| {
    ///         cmd
    ///             .bind_index_buffer(my_idx_buf, 0, vk::IndexType::UINT16)
    ///             .bind_vertex_buffer(0, my_vtx_buf, 0)
    ///             .draw_indexed(42, 1, 0, 0, 0);
    ///     });
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn bind_index_buffer(
        &self,
        buffer: impl Into<AnyBufferNode>,
        offset: vk::DeviceSize,
        index_ty: vk::IndexType,
    ) -> &Self {
        let buffer = buffer.into();
        let buffer = self.resource(buffer);

        unsafe {
            self.cmd
                .device
                .cmd_bind_index_buffer(self.cmd.handle, buffer.handle, offset, index_ty);
        }

        self
    }

    /// Bind a vertex buffer to the current pass.
    ///
    /// The vertex input binding is updated to start at `offset` from the start of `buffer`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use ash::vk;
    /// # use vk_graph::cmd::{LoadOp, StoreOp};
    /// # use vk_graph::driver::{sync::AccessType, DriverError};
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::graphic::{GraphicPipeline, GraphicPipelineInfo};
    /// # use vk_graph::driver::image::{Image, ImageInfo};
    /// # use vk_graph::driver::shader::Shader;
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::new(DeviceInfo::default())?;
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::VERTEX_BUFFER);
    /// # let my_vtx_buf = Buffer::create(&device, buf_info)?;
    /// # let my_frag_code = [0u8; 1];
    /// # let my_vert_code = [0u8; 1];
    /// # let vert = Shader::new_vertex(my_vert_code.as_slice());
    /// # let frag = Shader::new_fragment(my_frag_code.as_slice());
    /// # let info = GraphicPipelineInfo::default();
    /// # let my_graphic_pipeline = GraphicPipeline::create(&device, info, [vert, frag])?;
    /// # let mut my_graph = Graph::default();
    /// # let info = ImageInfo::image_2d(
    /// #     32,
    /// #     32,
    /// #     vk::Format::R8G8B8A8_UNORM,
    /// #     vk::ImageUsageFlags::SAMPLED,
    /// # );
    /// # let swapchain_image = my_graph.bind_resource(Image::create(&device, info)?);
    /// # let my_vtx_buf = my_graph.bind_resource(my_vtx_buf);
    /// my_graph
    ///     .begin_cmd()
    ///     .debug_name("my unindexed geometry draw pass")
    ///     .bind_pipeline(&my_graphic_pipeline)
    ///     .color_attachment_image(0, swapchain_image, LoadOp::DontCare, StoreOp::Store)
    ///     .resource_access(my_vtx_buf, AccessType::VertexBuffer)
    ///     .record_cmd(move |cmd| {
    ///         cmd
    ///             .bind_vertex_buffer(0, my_vtx_buf, 0)
    ///             .draw(42, 1, 0, 0);
    ///     });
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn bind_vertex_buffer(
        &self,
        binding: u32,
        buffer: impl Into<AnyBufferNode>,
        offset: vk::DeviceSize,
    ) -> &Self {
        let buffer = buffer.into();
        let buffer = self.resource(buffer);

        unsafe {
            self.cmd.device.cmd_bind_vertex_buffers(
                self.cmd.handle,
                binding,
                slice::from_ref(&buffer.handle),
                slice::from_ref(&offset),
            );
        }

        self
    }

    /// Binds multiple vertex buffers to the current pass, starting at the given `first_binding`.
    ///
    /// Each vertex input binding in `buffers` specifies an offset from the start of the
    /// corresponding buffer.
    ///
    /// The vertex input attributes that use each of these bindings will use these updated addresses
    /// in their address calculations for subsequent drawing commands.
    #[profiling::function]
    pub fn bind_vertex_buffers<N>(
        &self,
        first_binding: u32,
        buffer_offsets: impl IntoIterator<Item = (N, vk::DeviceSize)>,
    ) -> &Self
    where
        N: Into<AnyBufferNode>,
    {
        #[derive(Default)]
        struct Tls {
            buffers: Vec<vk::Buffer>,
            offsets: Vec<vk::DeviceSize>,
        }

        thread_local! {
            static TLS: RefCell<Tls> = Default::default();
        }

        TLS.with_borrow_mut(|tls| {
            tls.buffers.clear();
            tls.offsets.clear();

            for (buffer, offset) in buffer_offsets {
                let buffer = buffer.into();
                let buffer = self.resource(buffer);

                tls.buffers.push(buffer.handle);
                tls.offsets.push(offset);
            }

            unsafe {
                self.cmd.device.cmd_bind_vertex_buffers(
                    self.cmd.handle,
                    first_binding,
                    tls.buffers.as_slice(),
                    tls.offsets.as_slice(),
                );
            }
        });

        self
    }

    /// Draw unindexed primitives.
    ///
    /// When the command is executed, primitives are assembled using the current primitive topology
    /// and `vertex_count` consecutive vertex indices with the first `vertex_index` value equal to
    /// `first_vertex`. The primitives are drawn `instance_count` times with `instance_index`
    /// starting with `first_instance` and increasing sequentially for each instance.
    #[profiling::function]
    pub fn draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> &Self {
        unsafe {
            self.cmd.device.cmd_draw(
                self.cmd.handle,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }

        self
    }

    /// Draw indexed primitives.
    ///
    /// When the command is executed, primitives are assembled using the current primitive topology
    /// and `index_count` vertices whose indices are retrieved from the index buffer. The index
    /// buffer is treated as an array of tightly packed unsigned integers of size defined by the
    /// `index_ty` parameter with which the buffer was bound.
    #[profiling::function]
    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> &Self {
        unsafe {
            self.cmd.device.cmd_draw_indexed(
                self.cmd.handle,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }

        self
    }

    /// Draw primitives with indirect parameters and indexed vertices.
    ///
    /// `draw_indexed_indirect` behaves similarly to `draw_indexed` except that the parameters are
    /// read by the device from `buffer` during execution. `draw_count` draws are executed by the
    /// command, with parameters taken from `buffer` starting at `offset` and increasing by `stride`
    /// bytes for each successive draw. The parameters of each draw are encoded in an array of
    /// [`vk::DrawIndexedIndirectCommand`] structures.
    ///
    /// If `draw_count` is less than or equal to one, `stride` is ignored.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use std::mem::size_of;
    /// # use ash::vk;
    /// # use vk_graph::cmd::{LoadOp, StoreOp};
    /// # use vk_graph::driver::{AccessType, DriverError};
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::graphic::{GraphicPipeline, GraphicPipelineInfo};
    /// # use vk_graph::driver::image::{Image, ImageInfo};
    /// # use vk_graph::driver::shader::Shader;
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::new(DeviceInfo::default())?;
    /// # let my_frag_code = [0u8; 1];
    /// # let my_vert_code = [0u8; 1];
    /// # let vert = Shader::new_vertex(my_vert_code.as_slice());
    /// # let frag = Shader::new_fragment(my_frag_code.as_slice());
    /// # let info = GraphicPipelineInfo::default();
    /// # let my_graphic_pipeline = GraphicPipeline::create(&device, info, [vert, frag])?;
    /// # let mut my_graph = Graph::default();
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::INDEX_BUFFER);
    /// # let my_idx_buf = Buffer::create(&device, buf_info)?;
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::VERTEX_BUFFER);
    /// # let my_vtx_buf = Buffer::create(&device, buf_info)?;
    /// # let my_idx_buf = my_graph.bind_resource(my_idx_buf);
    /// # let my_vtx_buf = my_graph.bind_resource(my_vtx_buf);
    /// # let info = ImageInfo::image_2d(
    /// #     32,
    /// #     32,
    /// #     vk::Format::R8G8B8A8_UNORM,
    /// #     vk::ImageUsageFlags::SAMPLED,
    /// # );
    /// # let swapchain_image = my_graph.bind_resource(Image::create(&device, info)?);
    /// const CMD_SIZE: usize = size_of::<vk::DrawIndexedIndirectCommand>();
    ///
    /// let cmd = vk::DrawIndexedIndirectCommand {
    ///     index_count: 3,
    ///     instance_count: 1,
    ///     first_index: 0,
    ///     vertex_offset: 0,
    ///     first_instance: 0,
    /// };
    /// let cmd_data = unsafe {
    ///     std::slice::from_raw_parts(&cmd as *const _ as *const _, CMD_SIZE)
    /// };
    ///
    /// let buf_flags = vk::BufferUsageFlags::STORAGE_BUFFER;
    /// let buf = Buffer::create_from_slice(&device, buf_flags, cmd_data)?;
    /// let buf_node = my_graph.bind_resource(buf);
    ///
    /// my_graph
    ///     .begin_cmd()
    ///     .debug_name("draw a single triangle")
    ///     .bind_pipeline(&my_graphic_pipeline)
    ///     .color_attachment_image(0, swapchain_image, LoadOp::DontCare, StoreOp::Store)
    ///     .resource_access(my_idx_buf, AccessType::IndexBuffer)
    ///     .resource_access(my_vtx_buf, AccessType::VertexBuffer)
    ///     .resource_access(buf_node, AccessType::IndirectBuffer)
    ///     .record_cmd(move |cmd| {
    ///         cmd
    ///             .bind_index_buffer(my_idx_buf, 0, vk::IndexType::UINT16)
    ///             .bind_vertex_buffer(0, my_vtx_buf, 0)
    ///             .draw_indexed_indirect(buf_node, 0, 1, 0);
    ///     });
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn draw_indexed_indirect(
        &self,
        buffer: impl Into<AnyBufferNode>,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> &Self {
        let buffer = buffer.into();
        let buffer = self.resource(buffer);

        unsafe {
            self.cmd.device.cmd_draw_indexed_indirect(
                self.cmd.handle,
                buffer.handle,
                offset,
                draw_count,
                stride,
            );
        }

        self
    }

    /// Draw primitives with indirect parameters, indexed vertices, and draw count.
    ///
    /// `draw_indexed_indirect_count` behaves similarly to `draw_indexed_indirect` except that the
    /// draw count is read by the device from `buffer` during execution. The command will read an
    /// unsigned 32-bit integer from `count_buf` located at `count_buf_offset` and use this as the
    /// draw count.
    ///
    /// `max_draw_count` specifies the maximum number of draws that will be executed. The actual
    /// number of executed draw calls is the minimum of the count specified in `count_buf` and
    /// `max_draw_count`.
    ///
    /// `stride` is the byte stride between successive sets of draw parameters.
    #[profiling::function]
    pub fn draw_indexed_indirect_count(
        &self,
        buffer: impl Into<AnyBufferNode>,
        offset: vk::DeviceSize,
        count_buf: impl Into<AnyBufferNode>,
        count_buf_offset: vk::DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> &Self {
        let buffer = buffer.into();
        let buffer = self.resource(buffer);

        let count_buf = count_buf.into();
        let count_buf = self.resource(count_buf);

        unsafe {
            self.cmd.device.cmd_draw_indexed_indirect_count(
                self.cmd.handle,
                buffer.handle,
                offset,
                count_buf.handle,
                count_buf_offset,
                max_draw_count,
                stride,
            );
        }

        self
    }

    /// Draw primitives with indirect parameters and unindexed vertices.
    ///
    /// Behaves otherwise similar to [`Self::draw_indexed_indirect`].
    #[profiling::function]
    pub fn draw_indirect(
        &self,
        buffer: impl Into<AnyBufferNode>,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> &Self {
        let buffer = buffer.into();
        let buffer = self.resource(buffer);

        unsafe {
            self.cmd.device.cmd_draw_indirect(
                self.cmd.handle,
                buffer.handle,
                offset,
                draw_count,
                stride,
            );
        }

        self
    }

    /// Draw primitives with indirect parameters, unindexed vertices, and draw count.
    ///
    /// Behaves otherwise similar to [`Self::draw_indexed_indirect_count`].
    #[profiling::function]
    pub fn draw_indirect_count(
        &self,
        buffer: impl Into<AnyBufferNode>,
        offset: vk::DeviceSize,
        count_buf: impl Into<AnyBufferNode>,
        count_buf_offset: vk::DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> &Self {
        let buffer = buffer.into();
        let buffer = self.resource(buffer);

        let count_buf = count_buf.into();
        let count_buf = self.resource(count_buf);

        unsafe {
            self.cmd.device.cmd_draw_indirect_count(
                self.cmd.handle,
                buffer.handle,
                offset,
                count_buf.handle,
                count_buf_offset,
                max_draw_count,
                stride,
            );
        }

        self
    }

    /// Updates push constants.
    ///
    /// Push constants represent a high speed path to modify constant data in pipelines that is
    /// expected to outperform memory-backed resource updates.
    ///
    /// Push constant values can be updated incrementally, causing shader stages to read the new
    /// data for push constants modified by this command, while still reading the previous data for
    /// push constants not modified by this command.
    ///
    /// # Device limitations
    ///
    /// See
    /// [`device.physical_device.props.limits.max_push_constants_size`](vk::PhysicalDeviceLimits)
    /// for the limits of the current device. You may also check [gpuinfo.org] for a listing of
    /// reported limits on other devices.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # vk_shader_macros::glsl!(r#"
    /// #version 450
    /// #pragma shader_stage(compute)
    ///
    /// layout(push_constant) uniform PushConstants {
    ///     layout(offset = 0) uint the_answer;
    /// } push_constants;
    ///
    /// void main() {
    ///     // TODO: Add code!
    /// }
    /// # "#);
    /// ```
    ///
    /// ```no_run
    /// # use ash::vk;
    /// # use vk_graph::cmd::{LoadOp, StoreOp};
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::graphic::{GraphicPipeline, GraphicPipelineInfo};
    /// # use vk_graph::driver::image::{Image, ImageInfo};
    /// # use vk_graph::Graph;
    /// # use vk_graph::driver::shader::Shader;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::new(DeviceInfo::default())?;
    /// # let my_frag_code = [0u8; 1];
    /// # let my_vert_code = [0u8; 1];
    /// # let vert = Shader::new_vertex(my_vert_code.as_slice());
    /// # let frag = Shader::new_fragment(my_frag_code.as_slice());
    /// # let info = GraphicPipelineInfo::default();
    /// # let my_graphic_pipeline = GraphicPipeline::create(&device, info, [vert, frag])?;
    /// # let info = ImageInfo::image_2d(
    /// #     32,
    /// #     32,
    /// #     vk::Format::R8G8B8A8_UNORM,
    /// #     vk::ImageUsageFlags::SAMPLED,
    /// # );
    /// # let swapchain_image = Image::create(&device, info)?;
    /// # let mut my_graph = Graph::default();
    /// # let swapchain_image = my_graph.bind_resource(swapchain_image);
    /// my_graph
    ///     .begin_cmd()
    ///     .debug_name("draw a quad")
    ///     .bind_pipeline(&my_graphic_pipeline)
    ///     .color_attachment_image(0, swapchain_image, LoadOp::DontCare, StoreOp::Store)
    ///     .record_cmd(move |cmd| {
    ///         cmd
    ///             .push_constants(0, &[42])
    ///             .draw(6, 1, 0, 0);
    ///     });
    /// # Ok(()) }
    /// ```
    ///
    /// See [`vkCmdPushConstants`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdPushConstants.html).
    #[profiling::function]
    pub fn push_constants(&self, offset: u32, data: &[u8]) -> &Self {
        self.cmd_push_constants(
            self.pipeline.inner.layout,
            &self.pipeline.inner.push_constants,
            offset,
            data,
        );

        self
    }

    /// Set scissor rectangle dynamically for the current command.
    ///
    /// The default scissor state is no-clip.
    #[profiling::function]
    pub fn set_scissor(&self, first_scissor: u32, scissors: &[vk::Rect2D]) -> &Self {
        unsafe {
            self.cmd
                .device
                .cmd_set_scissor(self.cmd.handle, first_scissor, scissors);
        }

        self
    }

    /// Set the viewport dynamically for the current command.
    ///
    /// The default viewport state is the entire render target as defined by all combined image
    /// attachments.
    #[profiling::function]
    pub fn set_viewport(&self, first_viewport: u32, viewports: &[vk::Viewport]) -> &Self {
        unsafe {
            self.cmd
                .device
                .cmd_set_viewport(self.cmd.handle, first_viewport, viewports);
        }

        self
    }
}

impl<'a> Deref for GraphicCommandRef<'a> {
    type Target = CommandRef<'a>;

    fn deref(&self) -> &Self::Target {
        &self.cmd
    }
}

/// Specifies the state of a color or combined depth and stencil attachment image during graphic
/// render pass framebuffer load operations.
///
/// Use this to specify the desired contents of any image before use in a pipeline command buffer.
#[derive(Clone, Copy, Debug)]
pub enum LoadOp<T> {
    /// Clears the attachment.
    ///
    /// `T` will be [ClearColorValue] for color images or [vk::ClearDepthStencilValue] for
    /// combined depth and stencil images.
    Clear(T),

    /// The attachment will become undefined and reads will produce garbage data.
    DontCare,

    /// The attachment will be preserved in memory.
    Load,
}

impl LoadOp<ClearColorValue> {
    /// A load operation which results in a color attachment filled with rgb zeros and alpha ones.
    pub const CLEAR_BLACK_ALPHA_ONE: Self = Self::Clear(ClearColorValue::BLACK_ALPHA_ONE);

    /// A load operation which results in a color attachment filled with zeros.
    pub const CLEAR_BLACK_ALPHA_ZERO: Self = Self::Clear(ClearColorValue::BLACK_ALPHA_ZERO);

    /// A load operation which results in a color attachment filled with rgb zeros and alpha ones.
    pub const CLEAR_WHITE_ALPHA_ONE: Self = Self::Clear(ClearColorValue::WHITE_ALPHA_ONE);

    /// A load operation which results in a color attachment filled with rgb ones and alpha zeros.
    pub const CLEAR_WHITE_ALPHA_ZERO: Self = Self::Clear(ClearColorValue::WHITE_ALPHA_ZERO);

    /// Convenience constructor for clear color values.
    pub fn clear_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self::Clear(ClearColorValue::rgba(r, g, b, a))
    }
}

impl LoadOp<vk::ClearDepthStencilValue> {
    /// A load operation which results in a depth attachment filled with ones and stencil filled
    /// with zeros.
    pub const CLEAR_ONE_STENCIL_ZERO: Self = Self::Clear(vk::ClearDepthStencilValue {
        depth: 1.0,
        stencil: 0,
    });

    /// A load operation which results in a depth and stencil attachment filled with zeros.
    pub const CLEAR_ZERO_STENCIL_ZERO: Self = Self::Clear(vk::ClearDepthStencilValue {
        depth: 0.0,
        stencil: 0,
    });

    /// Convenience constructor for clear depth and stencil values.
    pub fn clear_depth_stencil(depth: f32, stencil: u32) -> Self {
        Self::Clear(vk::ClearDepthStencilValue { depth, stencil })
    }
}

/// Specifies the state of a color or combined depth and stencil attachment image after graphic
/// render pass framebuffer store operations.
///
/// Use this to specify the desired contents of any image after use in a pipeline command buffer.
#[derive(Clone, Copy, Debug)]
pub enum StoreOp {
    /// The attachment will become undefined and reads will produce garbage data.
    DontCare,

    /// The attachment will be preserved in memory.
    Store,
}

#[allow(unused)]
mod deprecated {
    use {
        crate::{
            Attachment, Node, SubresourceAccess,
            cmd::{
                AttachmentIndex, Binding, ClearColorValue, PipelineCommand, Subresource,
                SubresourceRange, ViewInfo, graphic::GraphicCommandRef,
            },
            driver::{
                graphic::GraphicPipeline,
                image::{
                    ImageInfo, ImageViewInfo, image_subresource_range_contains,
                    image_subresource_range_intersects,
                },
                render_pass::ResolveMode,
            },
            node::AnyImageNode,
        },
        ash::vk,
        vk_sync::AccessType,
    };

    impl GraphicCommandRef<'_> {
        #[deprecated = "use push_constants function"]
        #[doc(hidden)]
        pub fn push_constants_offset(&self, offset: u32, data: &[u8]) -> &Self {
            self.push_constants(offset, data)
        }
    }

    // Attachment functions from previous version
    impl PipelineCommand<'_, GraphicPipeline> {
        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn attach_color(
            self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
        ) -> Self {
            let image = image.into();
            let image_info = self.resource(image).info;
            let image_view_info: ImageViewInfo = image_info.into();

            #[allow(deprecated)]
            self.attach_color_as(attachment_idx, image, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn attach_color_as(
            mut self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> Self {
            #[allow(deprecated)]
            self.set_attach_color_as(attachment_idx, image, image_view_info);

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn attach_depth_stencil(self, image: impl Into<AnyImageNode>) -> Self {
            let image = image.into();
            let image_view_info = self.resource(image).info;

            #[allow(deprecated)]
            self.attach_depth_stencil_as(image, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn attach_depth_stencil_as(
            mut self,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> Self {
            #[allow(deprecated)]
            self.set_attach_depth_stencil_as(image, image_view_info);

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn clear_color(
            self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
        ) -> Self {
            #[allow(deprecated)]
            self.clear_color_value(attachment_idx, image, [0.0, 0.0, 0.0, 0.0])
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn clear_color_value(
            self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            color: impl Into<ClearColorValue>,
        ) -> Self {
            let image = image.into();
            let image_info = self.resource(image).info;
            let image_view_info: ImageViewInfo = image_info.into();

            #[allow(deprecated)]
            self.clear_color_value_as(attachment_idx, image, color, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn clear_color_value_as(
            mut self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            color: impl Into<ClearColorValue>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> Self {
            #[allow(deprecated)]
            self.set_clear_color_value_as(attachment_idx, image, color, image_view_info);

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn clear_depth_stencil(self, image: impl Into<AnyImageNode>) -> Self {
            #[allow(deprecated)]
            self.clear_depth_stencil_value(image, 1.0, 0)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn clear_depth_stencil_value(
            self,
            image: impl Into<AnyImageNode>,
            depth: f32,
            stencil: u32,
        ) -> Self {
            let image = image.into();
            let image_info = self.resource(image).info;
            let image_view_info: ImageViewInfo = image_info.into();

            #[allow(deprecated)]
            self.clear_depth_stencil_value_as(image, depth, stencil, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn clear_depth_stencil_value_as(
            mut self,
            image: impl Into<AnyImageNode>,
            depth: f32,
            stencil: u32,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> Self {
            #[allow(deprecated)]
            self.set_clear_depth_stencil_value_as(image, depth, stencil, image_view_info);

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn load_color(
            self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
        ) -> Self {
            let image = image.into();
            let image_info = self.resource(image).info;

            // Use the plain node information as the whole view of the node
            let image_view_info = image_info;

            #[allow(deprecated)]
            self.load_color_as(attachment_idx, image, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn load_color_as(
            mut self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> Self {
            #[allow(deprecated)]
            self.set_load_color_as(attachment_idx, image, image_view_info);

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn load_depth_stencil(self, image: impl Into<AnyImageNode>) -> Self {
            let image = image.into();
            let image_view_info = self.resource(image).info;

            #[allow(deprecated)]
            self.load_depth_stencil_as(image, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn load_depth_stencil_as(
            mut self,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> Self {
            #[allow(deprecated)]
            self.set_load_depth_stencil_as(image, image_view_info);

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn store_color(
            self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
        ) -> Self {
            let image = image.into();
            let image_view_info = self.resource(image).info;

            #[allow(deprecated)]
            self.store_color_as(attachment_idx, image, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn store_color_as(
            mut self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> Self {
            #[allow(deprecated)]
            self.set_store_color_as(attachment_idx, image, image_view_info);

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn store_depth_stencil(self, image: impl Into<AnyImageNode>) -> Self {
            let image = image.into();
            let image_view_info = self.resource(image).info;

            #[allow(deprecated)]
            self.store_depth_stencil_as(image, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn store_depth_stencil_as(
            mut self,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> Self {
            #[allow(deprecated)]
            self.set_store_depth_stencil_as(image, image_view_info);

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn resolve_color(
            self,
            src_attachment_idx: AttachmentIndex,
            dst_attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
        ) -> Self {
            let image = image.into();
            let image_view_info = self.resource(image).info;

            #[allow(deprecated)]
            self.resolve_color_as(
                src_attachment_idx,
                dst_attachment_idx,
                image,
                image_view_info,
            )
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn resolve_color_as(
            mut self,
            src_attachment_idx: AttachmentIndex,
            dst_attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> Self {
            #[allow(deprecated)]
            self.set_resolve_color_as(
                src_attachment_idx,
                dst_attachment_idx,
                image,
                image_view_info,
            );

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn resolve_depth_stencil(
            self,
            dst_attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            depth_mode: Option<ResolveMode>,
            stencil_mode: Option<ResolveMode>,
        ) -> Self {
            let image = image.into();
            let image_view_info = self.resource(image).info;

            #[allow(deprecated)]
            self.resolve_depth_stencil_as(
                dst_attachment_idx,
                image,
                image_view_info,
                depth_mode,
                stencil_mode,
            )
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn resolve_depth_stencil_as(
            mut self,
            dst_attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
            depth_mode: Option<ResolveMode>,
            stencil_mode: Option<ResolveMode>,
        ) -> Self {
            #[allow(deprecated)]
            self.set_resolve_depth_stencil_as(
                dst_attachment_idx,
                image,
                image_view_info,
                depth_mode,
                stencil_mode,
            );

            self
        }
    }

    // Attachment functions as setters
    impl PipelineCommand<'_, GraphicPipeline> {
        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_attach_color(
            &mut self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
        ) -> &mut Self {
            let image = image.into();
            let image_info = self.resource(image).info;
            let image_view_info: ImageViewInfo = image_info.into();

            #[allow(deprecated)]
            self.set_attach_color_as(attachment_idx, image, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_attach_color_as(
            &mut self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> &mut Self {
            let image = image.into();
            let image_view_info = image_view_info.into();
            let node_idx = image.index();
            let ImageInfo { sample_count, .. } = self.resource(image).info;

            debug_assert!(
                !self
                    .cmd
                    .cmd()
                    .expect_last_exec()
                    .color_clears
                    .contains_key(&attachment_idx),
                "color attachment {attachment_idx} already attached via clear"
            );
            debug_assert!(
                !self
                    .cmd
                    .cmd()
                    .expect_last_exec()
                    .color_loads
                    .contains_key(&attachment_idx),
                "color attachment {attachment_idx} already attached via load"
            );

            self.cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .color_attachments
                .insert(
                    attachment_idx,
                    Attachment::new(image_view_info, sample_count, node_idx),
                );

            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_resolves
                        .get(&attachment_idx)
                        .map(|(attachment, _)| *attachment),
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_attachments
                        .get(&attachment_idx)
                        .copied()
                ),
                "color attachment {attachment_idx} incompatible with existing resolve"
            );
            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_stores
                        .get(&attachment_idx)
                        .copied(),
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_attachments
                        .get(&attachment_idx)
                        .copied()
                ),
                "color attachment {attachment_idx} incompatible with existing store"
            );

            self.cmd.push_subresource_access(
                image,
                SubresourceRange::Image(image_view_info.into()),
                AccessType::ColorAttachmentWrite,
            );

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_attach_depth_stencil(&mut self, image: impl Into<AnyImageNode>) -> &mut Self {
            let image = image.into();
            let image_view_info = self.resource(image).info;

            #[allow(deprecated)]
            self.set_attach_depth_stencil_as(image, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_attach_depth_stencil_as(
            &mut self,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> &mut Self {
            let image = image.into();
            let image_view_info = image_view_info.into();
            let node_idx = image.index();
            let ImageInfo { sample_count, .. } = self.resource(image).info;

            debug_assert!(
                self.cmd
                    .cmd()
                    .expect_last_exec()
                    .depth_stencil_clear
                    .is_none(),
                "depth/stencil attachment already attached via clear"
            );
            debug_assert!(
                self.cmd
                    .cmd()
                    .expect_last_exec()
                    .depth_stencil_load
                    .is_none(),
                "depth/stencil attachment already attached via load"
            );

            self.cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .depth_stencil_attachment =
                Some(Attachment::new(image_view_info, sample_count, node_idx));

            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .depth_stencil_resolve
                        .map(|(attachment, ..)| attachment),
                    self.cmd.cmd().expect_last_exec().depth_stencil_attachment
                ),
                "depth/stencil attachment incompatible with existing resolve"
            );
            debug_assert!(
                Attachment::are_compatible(
                    self.cmd.cmd().expect_last_exec().depth_stencil_store,
                    self.cmd.cmd().expect_last_exec().depth_stencil_attachment
                ),
                "depth/stencil attachment incompatible with existing store"
            );

            self.cmd.push_subresource_access(
                image,
                SubresourceRange::Image(image_view_info.into()),
                if image_view_info
                    .aspect_mask
                    .contains(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                {
                    AccessType::DepthStencilAttachmentWrite
                } else if image_view_info
                    .aspect_mask
                    .contains(vk::ImageAspectFlags::DEPTH)
                {
                    AccessType::DepthAttachmentWriteStencilReadOnly
                } else {
                    AccessType::StencilAttachmentWriteDepthReadOnly
                },
            );

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_clear_color(
            &mut self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
        ) -> &mut Self {
            #[allow(deprecated)]
            self.set_clear_color_value(attachment_idx, image, [0.0, 0.0, 0.0, 0.0])
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_clear_color_value(
            &mut self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            color: impl Into<ClearColorValue>,
        ) -> &mut Self {
            let image = image.into();
            let image_info = self.resource(image).info;
            let image_view_info: ImageViewInfo = image_info.into();

            #[allow(deprecated)]
            self.set_clear_color_value_as(attachment_idx, image, color, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_clear_color_value_as(
            &mut self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            color: impl Into<ClearColorValue>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> &mut Self {
            let image = image.into();
            let image_view_info = image_view_info.into();
            let node_idx = image.index();
            let ImageInfo { sample_count, .. } = self.resource(image).info;

            let color = color.into();
            let color: vk::ClearColorValue = color.into();
            let color = unsafe { color.float32 };

            debug_assert!(
                !self
                    .cmd
                    .cmd()
                    .expect_last_exec()
                    .color_attachments
                    .contains_key(&attachment_idx),
                "color attachment {attachment_idx} already attached"
            );
            debug_assert!(
                !self
                    .cmd
                    .cmd()
                    .expect_last_exec()
                    .color_loads
                    .contains_key(&attachment_idx),
                "color attachment {attachment_idx} already attached via load"
            );

            self.cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .color_clears
                .insert(
                    attachment_idx,
                    (
                        Attachment::new(image_view_info, sample_count, node_idx),
                        color,
                    ),
                );

            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_resolves
                        .get(&attachment_idx)
                        .map(|(attachment, _)| *attachment),
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_clears
                        .get(&attachment_idx)
                        .map(|(attachment, _)| *attachment)
                ),
                "color attachment {attachment_idx} clear incompatible with existing resolve"
            );
            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_stores
                        .get(&attachment_idx)
                        .copied(),
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_clears
                        .get(&attachment_idx)
                        .map(|(attachment, _)| *attachment)
                ),
                "color attachment {attachment_idx} clear incompatible with existing store"
            );

            let mut image_access = AccessType::ColorAttachmentWrite;
            let image_range = image_view_info.into();

            // Upgrade existing read access to read-write
            if let Some(accesses) = self
                .cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .accesses
                .get_mut(&node_idx)
            {
                for SubresourceAccess {
                    access,
                    subresource,
                } in accesses
                {
                    let access_image_range = *subresource.expect_image();
                    if !image_subresource_range_intersects(access_image_range, image_range) {
                        continue;
                    }

                    image_access = match *access {
                        AccessType::ColorAttachmentRead | AccessType::ColorAttachmentReadWrite => {
                            AccessType::ColorAttachmentReadWrite
                        }
                        AccessType::ColorAttachmentWrite => AccessType::ColorAttachmentWrite,
                        _ => continue,
                    };

                    *access = image_access;

                    // If the clear access is a subset of the existing access range there is no need
                    // to push a new access
                    if image_subresource_range_contains(access_image_range, image_range) {
                        return self;
                    }
                }
            }

            self.cmd.push_subresource_access(
                image,
                SubresourceRange::Image(image_range),
                image_access,
            );

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_clear_depth_stencil(&mut self, image: impl Into<AnyImageNode>) -> &mut Self {
            #[allow(deprecated)]
            self.set_clear_depth_stencil_value(image, 1.0, 0)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_clear_depth_stencil_value(
            &mut self,
            image: impl Into<AnyImageNode>,
            depth: f32,
            stencil: u32,
        ) -> &mut Self {
            let image = image.into();
            let image_info = self.resource(image).info;
            let image_view_info: ImageViewInfo = image_info.into();

            #[allow(deprecated)]
            self.set_clear_depth_stencil_value_as(image, depth, stencil, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_clear_depth_stencil_value_as(
            &mut self,
            image: impl Into<AnyImageNode>,
            depth: f32,
            stencil: u32,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> &mut Self {
            let image = image.into();
            let image_view_info = image_view_info.into();
            let node_idx = image.index();
            let ImageInfo { sample_count, .. } = self.resource(image).info;

            debug_assert!(
                self.cmd
                    .cmd()
                    .expect_last_exec()
                    .depth_stencil_attachment
                    .is_none(),
                "depth/stencil attachment already attached"
            );
            debug_assert!(
                self.cmd
                    .cmd()
                    .expect_last_exec()
                    .depth_stencil_load
                    .is_none(),
                "depth/stencil attachment already attached via load"
            );

            self.cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .depth_stencil_clear = Some((
                Attachment::new(image_view_info, sample_count, node_idx),
                vk::ClearDepthStencilValue { depth, stencil },
            ));

            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .depth_stencil_resolve
                        .map(|(attachment, ..)| attachment),
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .depth_stencil_clear
                        .map(|(attachment, _)| attachment)
                ),
                "depth/stencil attachment clear incompatible with existing resolve"
            );
            debug_assert!(
                Attachment::are_compatible(
                    self.cmd.cmd().expect_last_exec().depth_stencil_store,
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .depth_stencil_clear
                        .map(|(attachment, _)| attachment)
                ),
                "depth/stencil attachment clear incompatible with existing store"
            );

            let mut image_access = if image_view_info
                .aspect_mask
                .contains(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
            {
                AccessType::DepthStencilAttachmentWrite
            } else if image_view_info
                .aspect_mask
                .contains(vk::ImageAspectFlags::DEPTH)
            {
                AccessType::DepthAttachmentWriteStencilReadOnly
            } else {
                debug_assert!(
                    image_view_info
                        .aspect_mask
                        .contains(vk::ImageAspectFlags::STENCIL)
                );

                AccessType::StencilAttachmentWriteDepthReadOnly
            };
            let image_range = image_view_info.into();

            // Upgrade existing read access to read-write
            if let Some(accesses) = self
                .cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .accesses
                .get_mut(&node_idx)
            {
                for SubresourceAccess {
                    access,
                    subresource,
                } in accesses
                {
                    let access_image_range = *subresource.expect_image();
                    if !image_subresource_range_intersects(access_image_range, image_range) {
                        continue;
                    }

                    image_access = match *access {
                        AccessType::DepthAttachmentWriteStencilReadOnly => {
                            if image_view_info
                                .aspect_mask
                                .contains(vk::ImageAspectFlags::STENCIL)
                            {
                                AccessType::DepthStencilAttachmentReadWrite
                            } else {
                                AccessType::DepthAttachmentWriteStencilReadOnly
                            }
                        }
                        AccessType::DepthStencilAttachmentRead => {
                            if !image_view_info
                                .aspect_mask
                                .contains(vk::ImageAspectFlags::DEPTH)
                            {
                                AccessType::StencilAttachmentWriteDepthReadOnly
                            } else {
                                AccessType::DepthAttachmentWriteStencilReadOnly
                            }
                        }
                        AccessType::DepthStencilAttachmentWrite => {
                            AccessType::DepthStencilAttachmentWrite
                        }
                        AccessType::StencilAttachmentWriteDepthReadOnly => {
                            if image_view_info
                                .aspect_mask
                                .contains(vk::ImageAspectFlags::DEPTH)
                            {
                                AccessType::DepthStencilAttachmentReadWrite
                            } else {
                                AccessType::StencilAttachmentWriteDepthReadOnly
                            }
                        }
                        _ => continue,
                    };

                    *access = image_access;

                    // If the clear access is a subset of the existing access range there is no need
                    // to push a new access
                    if image_subresource_range_contains(access_image_range, image_range) {
                        return self;
                    }
                }
            }

            self.cmd.push_subresource_access(
                image,
                SubresourceRange::Image(image_range),
                image_access,
            );

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_load_color(
            &mut self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
        ) -> &mut Self {
            let image = image.into();
            let image_info = self.resource(image).info;

            // Use the plain node information as the whole view of the node
            let image_view_info = image_info;

            #[allow(deprecated)]
            self.set_load_color_as(attachment_idx, image, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_load_color_as(
            &mut self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> &mut Self {
            let image = image.into();
            let image_view_info = image_view_info.into();
            let node_idx = image.index();
            let ImageInfo { sample_count, .. } = self.resource(image).info;

            debug_assert!(
                !self
                    .cmd
                    .cmd()
                    .expect_last_exec()
                    .color_attachments
                    .contains_key(&attachment_idx),
                "color attachment {attachment_idx} already attached"
            );
            debug_assert!(
                !self
                    .cmd
                    .cmd()
                    .expect_last_exec()
                    .color_clears
                    .contains_key(&attachment_idx),
                "color attachment {attachment_idx} already attached via clear"
            );

            self.cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .color_loads
                .insert(
                    attachment_idx,
                    Attachment::new(image_view_info, sample_count, node_idx),
                );

            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_resolves
                        .get(&attachment_idx)
                        .map(|(attachment, _)| *attachment),
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_loads
                        .get(&attachment_idx)
                        .copied()
                ),
                "color attachment {attachment_idx} load incompatible with existing resolve"
            );
            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_stores
                        .get(&attachment_idx)
                        .copied(),
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_loads
                        .get(&attachment_idx)
                        .copied()
                ),
                "color attachment {attachment_idx} load incompatible with existing store"
            );

            let mut image_access = AccessType::ColorAttachmentRead;
            let image_range = image_view_info.into();

            // Upgrade existing write access to read-write
            if let Some(accesses) = self
                .cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .accesses
                .get_mut(&node_idx)
            {
                for SubresourceAccess {
                    access,
                    subresource,
                } in accesses
                {
                    let access_image_range = *subresource.expect_image();
                    if !image_subresource_range_intersects(access_image_range, image_range) {
                        continue;
                    }

                    image_access = match *access {
                        AccessType::ColorAttachmentRead => AccessType::ColorAttachmentRead,
                        AccessType::ColorAttachmentReadWrite | AccessType::ColorAttachmentWrite => {
                            AccessType::ColorAttachmentReadWrite
                        }
                        _ => continue,
                    };

                    *access = image_access;

                    // If the load access is a subset of the existing access range there is no need
                    // to push a new access
                    if image_subresource_range_contains(access_image_range, image_range) {
                        return self;
                    }
                }
            }

            self.cmd.push_subresource_access(
                image,
                SubresourceRange::Image(image_range),
                image_access,
            );

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_load_depth_stencil(&mut self, image: impl Into<AnyImageNode>) -> &mut Self {
            let image = image.into();
            let image_view_info = self.resource(image).info;

            #[allow(deprecated)]
            self.set_load_depth_stencil_as(image, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_load_depth_stencil_as(
            &mut self,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> &mut Self {
            let image = image.into();
            let image_view_info = image_view_info.into();
            let node_idx = image.index();
            let ImageInfo { sample_count, .. } = self.resource(image).info;

            debug_assert!(
                self.cmd
                    .cmd()
                    .expect_last_exec()
                    .depth_stencil_attachment
                    .is_none(),
                "depth/stencil attachment already attached"
            );
            debug_assert!(
                self.cmd
                    .cmd()
                    .expect_last_exec()
                    .depth_stencil_clear
                    .is_none(),
                "depth/stencil attachment already attached via clear"
            );

            self.cmd.cmd_mut().expect_last_exec_mut().depth_stencil_load =
                Some(Attachment::new(image_view_info, sample_count, node_idx));

            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .depth_stencil_resolve
                        .map(|(attachment, ..)| attachment),
                    self.cmd.cmd().expect_last_exec().depth_stencil_load
                ),
                "depth/stencil attachment load incompatible with existing resolve"
            );
            debug_assert!(
                Attachment::are_compatible(
                    self.cmd.cmd().expect_last_exec().depth_stencil_store,
                    self.cmd.cmd().expect_last_exec().depth_stencil_load
                ),
                "depth/stencil attachment load incompatible with existing store"
            );

            let mut image_access = AccessType::DepthStencilAttachmentRead;
            let image_range = image_view_info.into();

            // Upgrade existing write access to read-write
            if let Some(accesses) = self
                .cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .accesses
                .get_mut(&node_idx)
            {
                for SubresourceAccess {
                    access,
                    subresource,
                } in accesses
                {
                    let access_image_range = *subresource.expect_image();
                    if !image_subresource_range_intersects(access_image_range, image_range) {
                        continue;
                    }

                    image_access = match *access {
                        AccessType::DepthAttachmentWriteStencilReadOnly => {
                            AccessType::DepthAttachmentWriteStencilReadOnly
                        }
                        AccessType::DepthStencilAttachmentRead => {
                            AccessType::DepthStencilAttachmentRead
                        }
                        AccessType::DepthStencilAttachmentWrite => {
                            AccessType::DepthStencilAttachmentReadWrite
                        }
                        AccessType::StencilAttachmentWriteDepthReadOnly => {
                            AccessType::StencilAttachmentWriteDepthReadOnly
                        }
                        _ => continue,
                    };

                    *access = image_access;

                    // If the load access is a subset of the existing access range there is no need
                    // to push a new access
                    if image_subresource_range_contains(access_image_range, image_range) {
                        return self;
                    }
                }
            }

            self.cmd.push_subresource_access(
                image,
                SubresourceRange::Image(image_range),
                image_access,
            );

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_store_color(
            &mut self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
        ) -> &mut Self {
            let image = image.into();
            let image_view_info = self.resource(image).info;

            #[allow(deprecated)]
            self.set_store_color_as(attachment_idx, image, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_store_color_as(
            &mut self,
            attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> &mut Self {
            let image = image.into();
            let image_view_info = image_view_info.into();
            let node_idx = image.index();
            let ImageInfo { sample_count, .. } = self.resource(image).info;

            self.cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .color_stores
                .insert(
                    attachment_idx,
                    Attachment::new(image_view_info, sample_count, node_idx),
                );

            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_attachments
                        .get(&attachment_idx)
                        .copied(),
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_stores
                        .get(&attachment_idx)
                        .copied()
                ),
                "color attachment {attachment_idx} store incompatible with existing attachment"
            );
            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_clears
                        .get(&attachment_idx)
                        .map(|(attachment, _)| *attachment),
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_stores
                        .get(&attachment_idx)
                        .copied()
                ),
                "color attachment {attachment_idx} store incompatible with existing clear"
            );
            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_loads
                        .get(&attachment_idx)
                        .copied(),
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_stores
                        .get(&attachment_idx)
                        .copied()
                ),
                "color attachment {attachment_idx} store incompatible with existing load"
            );

            let mut image_access = AccessType::ColorAttachmentWrite;
            let image_range = image_view_info.into();

            // Upgrade existing read access to read-write
            if let Some(accesses) = self
                .cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .accesses
                .get_mut(&node_idx)
            {
                for SubresourceAccess {
                    access,
                    subresource,
                } in accesses
                {
                    let access_image_range = *subresource.expect_image();
                    if !image_subresource_range_intersects(access_image_range, image_range) {
                        continue;
                    }

                    image_access = match *access {
                        AccessType::ColorAttachmentRead | AccessType::ColorAttachmentReadWrite => {
                            AccessType::ColorAttachmentReadWrite
                        }
                        AccessType::ColorAttachmentWrite => AccessType::ColorAttachmentWrite,
                        _ => continue,
                    };

                    *access = image_access;

                    // If the store access is a subset of the existing access range there is no need
                    // to push a new access
                    if image_subresource_range_contains(access_image_range, image_range) {
                        return self;
                    }
                }
            }

            self.cmd.push_subresource_access(
                image,
                SubresourceRange::Image(image_range),
                image_access,
            );

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_store_depth_stencil(&mut self, image: impl Into<AnyImageNode>) -> &mut Self {
            let image = image.into();
            let image_view_info = self.resource(image).info;

            #[allow(deprecated)]
            self.set_store_depth_stencil_as(image, image_view_info)
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_store_depth_stencil_as(
            &mut self,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> &mut Self {
            let image = image.into();
            let image_view_info = image_view_info.into();
            let node_idx = image.index();
            let ImageInfo { sample_count, .. } = self.resource(image).info;

            self.cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .depth_stencil_store =
                Some(Attachment::new(image_view_info, sample_count, node_idx));

            debug_assert!(
                Attachment::are_compatible(
                    self.cmd.cmd().expect_last_exec().depth_stencil_attachment,
                    self.cmd.cmd().expect_last_exec().depth_stencil_store
                ),
                "depth/stencil attachment store incompatible with existing attachment"
            );
            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .depth_stencil_clear
                        .map(|(attachment, _)| attachment),
                    self.cmd.cmd().expect_last_exec().depth_stencil_store
                ),
                "depth/stencil attachment store incompatible with existing clear"
            );
            debug_assert!(
                Attachment::are_compatible(
                    self.cmd.cmd().expect_last_exec().depth_stencil_load,
                    self.cmd.cmd().expect_last_exec().depth_stencil_store
                ),
                "depth/stencil attachment store incompatible with existing load"
            );

            let mut image_access = if image_view_info
                .aspect_mask
                .contains(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
            {
                AccessType::DepthStencilAttachmentWrite
            } else if image_view_info
                .aspect_mask
                .contains(vk::ImageAspectFlags::DEPTH)
            {
                AccessType::DepthAttachmentWriteStencilReadOnly
            } else {
                debug_assert!(
                    image_view_info
                        .aspect_mask
                        .contains(vk::ImageAspectFlags::STENCIL)
                );

                AccessType::StencilAttachmentWriteDepthReadOnly
            };
            let image_range = image_view_info.into();

            // Upgrade existing read access to read-write
            if let Some(accesses) = self
                .cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .accesses
                .get_mut(&node_idx)
            {
                for SubresourceAccess {
                    access,
                    subresource,
                } in accesses
                {
                    let access_image_range = *subresource.expect_image();
                    if !image_subresource_range_intersects(access_image_range, image_range) {
                        continue;
                    }

                    image_access = match *access {
                        AccessType::DepthAttachmentWriteStencilReadOnly => {
                            if image_view_info
                                .aspect_mask
                                .contains(vk::ImageAspectFlags::STENCIL)
                            {
                                AccessType::DepthStencilAttachmentReadWrite
                            } else {
                                AccessType::DepthAttachmentWriteStencilReadOnly
                            }
                        }
                        AccessType::DepthStencilAttachmentRead => {
                            if !image_view_info
                                .aspect_mask
                                .contains(vk::ImageAspectFlags::DEPTH)
                            {
                                AccessType::StencilAttachmentWriteDepthReadOnly
                            } else {
                                AccessType::DepthStencilAttachmentReadWrite
                            }
                        }
                        AccessType::DepthStencilAttachmentWrite => {
                            AccessType::DepthStencilAttachmentWrite
                        }
                        AccessType::StencilAttachmentWriteDepthReadOnly => {
                            if image_view_info
                                .aspect_mask
                                .contains(vk::ImageAspectFlags::DEPTH)
                            {
                                AccessType::DepthStencilAttachmentReadWrite
                            } else {
                                AccessType::StencilAttachmentWriteDepthReadOnly
                            }
                        }
                        _ => continue,
                    };

                    *access = image_access;

                    // If the store access is a subset of the existing access range there is no need
                    // to push a new access
                    if image_subresource_range_contains(access_image_range, image_range) {
                        return self;
                    }
                }
            }

            self.cmd.push_subresource_access(
                image,
                SubresourceRange::Image(image_range),
                image_access,
            );

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_resolve_color(
            &mut self,
            src_attachment_idx: AttachmentIndex,
            dst_attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
        ) -> &mut Self {
            let image = image.into();
            let image_view_info = self.resource(image).info;

            #[allow(deprecated)]
            self.set_resolve_color_as(
                src_attachment_idx,
                dst_attachment_idx,
                image,
                image_view_info,
            )
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_resolve_color_as(
            &mut self,
            src_attachment_idx: AttachmentIndex,
            dst_attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
        ) -> &mut Self {
            let image = image.into();
            let image_view_info = image_view_info.into();
            let node_idx = image.index();
            let ImageInfo { sample_count, .. } = self.resource(image).info;

            self.cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .color_resolves
                .insert(
                    dst_attachment_idx,
                    (
                        Attachment::new(image_view_info, sample_count, node_idx),
                        src_attachment_idx,
                    ),
                );

            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_attachments
                        .get(&dst_attachment_idx)
                        .copied(),
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_resolves
                        .get(&dst_attachment_idx)
                        .map(|(attachment, _)| *attachment)
                ),
                "color attachment {dst_attachment_idx} resolve conflict",
            );
            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_clears
                        .get(&dst_attachment_idx)
                        .map(|(attachment, _)| *attachment),
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_resolves
                        .get(&dst_attachment_idx)
                        .map(|(attachment, _)| *attachment)
                ),
                "color attachment {dst_attachment_idx} resolve incompatible with existing clear"
            );
            debug_assert!(
                Attachment::are_compatible(
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_loads
                        .get(&dst_attachment_idx)
                        .copied(),
                    self.cmd
                        .cmd()
                        .expect_last_exec()
                        .color_resolves
                        .get(&dst_attachment_idx)
                        .map(|(attachment, _)| *attachment)
                ),
                "color attachment {dst_attachment_idx} resolve incompatible with existing load"
            );

            let mut image_access = AccessType::ColorAttachmentWrite;
            let image_range = image_view_info.into();

            // Upgrade existing read access to read-write
            if let Some(accesses) = self
                .cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .accesses
                .get_mut(&node_idx)
            {
                for SubresourceAccess {
                    access,
                    subresource,
                } in accesses
                {
                    let access_image_range = *subresource.expect_image();
                    if !image_subresource_range_intersects(access_image_range, image_range) {
                        continue;
                    }

                    image_access = match *access {
                        AccessType::ColorAttachmentRead | AccessType::ColorAttachmentReadWrite => {
                            AccessType::ColorAttachmentReadWrite
                        }
                        AccessType::ColorAttachmentWrite => AccessType::ColorAttachmentWrite,
                        _ => continue,
                    };

                    *access = image_access;

                    // If the resolve access is a subset of the existing access range there is no
                    // need to push a new access
                    if image_subresource_range_contains(access_image_range, image_range) {
                        return self;
                    }
                }
            }

            self.cmd.push_subresource_access(
                image,
                SubresourceRange::Image(image_range),
                image_access,
            );

            self
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_resolve_depth_stencil(
            &mut self,
            dst_attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            depth_mode: Option<ResolveMode>,
            stencil_mode: Option<ResolveMode>,
        ) -> &mut Self {
            let image = image.into();
            let image_view_info = self.resource(image).info;

            #[allow(deprecated)]
            self.set_resolve_depth_stencil_as(
                dst_attachment_idx,
                image,
                image_view_info,
                depth_mode,
                stencil_mode,
            )
        }

        #[deprecated = "upgrade guide: https://github.com/attackgoat/vk-graph/pull/107"]
        #[doc(hidden)]
        pub fn set_resolve_depth_stencil_as(
            &mut self,
            dst_attachment_idx: AttachmentIndex,
            image: impl Into<AnyImageNode>,
            image_view_info: impl Into<ImageViewInfo>,
            depth_mode: Option<ResolveMode>,
            stencil_mode: Option<ResolveMode>,
        ) -> &mut Self {
            let image = image.into();
            let image_view_info = image_view_info.into();
            let node_idx = image.index();
            let ImageInfo { sample_count, .. } = self.resource(image).info;

            self.cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .depth_stencil_resolve = Some((
                Attachment::new(image_view_info, sample_count, node_idx),
                dst_attachment_idx,
                depth_mode,
                stencil_mode,
            ));

            let mut image_access = if image_view_info
                .aspect_mask
                .contains(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
            {
                AccessType::DepthStencilAttachmentWrite
            } else if image_view_info
                .aspect_mask
                .contains(vk::ImageAspectFlags::DEPTH)
            {
                AccessType::DepthAttachmentWriteStencilReadOnly
            } else {
                debug_assert!(
                    image_view_info
                        .aspect_mask
                        .contains(vk::ImageAspectFlags::STENCIL)
                );

                AccessType::StencilAttachmentWriteDepthReadOnly
            };
            let image_range = image_view_info.into();

            // Upgrade existing read access to read-write
            if let Some(accesses) = self
                .cmd
                .cmd_mut()
                .expect_last_exec_mut()
                .accesses
                .get_mut(&node_idx)
            {
                for SubresourceAccess {
                    access,
                    subresource,
                } in accesses
                {
                    let access_image_range = *subresource.expect_image();
                    if !image_subresource_range_intersects(access_image_range, image_range) {
                        continue;
                    }

                    image_access = match *access {
                        AccessType::DepthAttachmentWriteStencilReadOnly => {
                            if image_view_info
                                .aspect_mask
                                .contains(vk::ImageAspectFlags::STENCIL)
                            {
                                AccessType::DepthStencilAttachmentReadWrite
                            } else {
                                AccessType::DepthAttachmentWriteStencilReadOnly
                            }
                        }
                        AccessType::DepthStencilAttachmentRead => {
                            if !image_view_info
                                .aspect_mask
                                .contains(vk::ImageAspectFlags::DEPTH)
                            {
                                AccessType::StencilAttachmentWriteDepthReadOnly
                            } else {
                                AccessType::DepthStencilAttachmentReadWrite
                            }
                        }
                        AccessType::DepthStencilAttachmentWrite => {
                            AccessType::DepthStencilAttachmentWrite
                        }
                        AccessType::StencilAttachmentWriteDepthReadOnly => {
                            if image_view_info
                                .aspect_mask
                                .contains(vk::ImageAspectFlags::DEPTH)
                            {
                                AccessType::DepthStencilAttachmentReadWrite
                            } else {
                                AccessType::StencilAttachmentWriteDepthReadOnly
                            }
                        }
                        _ => continue,
                    };

                    *access = image_access;

                    // If the resolve access is a subset of the existing access range there is no
                    // need to push a new access
                    if image_subresource_range_contains(access_image_range, image_range) {
                        return self;
                    }
                }
            }

            self.cmd.push_subresource_access(
                image,
                SubresourceRange::Image(image_range),
                image_access,
            );

            self
        }
    }

    // Resource functions
    impl PipelineCommand<'_, GraphicPipeline> {
        #[deprecated = "use shader_resource_access"]
        #[doc(hidden)]
        pub fn read_descriptor<N>(self, binding: impl Into<Binding>, node: N) -> Self
        where
            N: Node + Subresource,
            N::Info: Copy,
            SubresourceRange: From<N::Info>,
            ViewInfo: From<N::Info>,
        {
            self.shader_resource_access(
                binding,
                node,
                AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
            )
        }

        #[deprecated = "use shader_subresource_access"]
        #[doc(hidden)]
        pub fn read_descriptor_as<N>(
            self,
            descriptor: impl Into<Binding>,
            node: N,
            node_view: impl Into<N::Info>,
        ) -> Self
        where
            N: Node + Subresource,
            N::Info: Copy,
            SubresourceRange: From<N::Info>,
            ViewInfo: From<N::Info>,
        {
            self.shader_subresource_access(
                descriptor,
                node,
                node_view,
                AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
            )
        }

        #[deprecated = "use record_cmd function"]
        #[doc(hidden)]
        pub fn record_subpass(
            self,
            func: impl FnOnce(GraphicCommandRef<'_>, ()) + Send + 'static,
        ) -> Self {
            self.record_cmd(|cmd| {
                func(cmd, ());
            })
        }

        #[deprecated = "use shader_resource_access function with AccessType::AnyShaderWrite"]
        #[doc(hidden)]
        pub fn write_descriptor<N>(self, descriptor: impl Into<Binding>, node: N) -> Self
        where
            N: Node + Subresource,
            N::Info: Copy,
            SubresourceRange: From<N::Info>,
            ViewInfo: From<N::Info>,
        {
            self.shader_resource_access(descriptor, node, AccessType::AnyShaderWrite)
        }

        #[deprecated = "use shader_subresource_access function with AccessType::AnyShaderWrite"]
        #[doc(hidden)]
        pub fn write_descriptor_as<N>(
            self,
            descriptor: impl Into<Binding>,
            node: N,
            node_view: impl Into<N::Info>,
        ) -> Self
        where
            N: Node + Subresource,
            N::Info: Copy,
            SubresourceRange: From<N::Info>,
            ViewInfo: From<N::Info>,
        {
            self.shader_subresource_access(descriptor, node, node_view, AccessType::AnyShaderWrite)
        }
    }
}
