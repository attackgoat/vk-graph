use {
    super::{AttachmentIndex, Bindings, Info, PipelineCommandRef, Subresource, SubresourceAccess},
    crate::{
        AnyBufferNode, AnyImageNode, Area, Attachment, ClearColorValue, Node, NodeIndex,
        SampleCount,
        driver::{
            device::Device,
            graphic::{DepthStencilMode, GraphicPipeline},
            image::{
                ImageViewInfo, image_subresource_range_contains, image_subresource_range_intersects,
            },
            render_pass::ResolveMode,
        },
    },
    ash::vk,
    log::trace,
    std::{cell::RefCell, slice, sync::Arc},
    vk_sync::AccessType,
};

/// Recording interface for drawing commands.
///
/// This structure provides a strongly-typed set of methods which allow rasterization shader code to
/// be executed. An instance of `Draw` is provided to the closure parameter of
/// [`PipelineCommandRef::record_pipeline`] which may be accessed by binding a [`GraphicPipeline`] to a
/// render pass.
///
/// # Examples
///
/// Basic usage:
///
/// ```no_run
/// # use std::sync::Arc;
/// # use ash::vk;
/// # use vk_graph::driver::DriverError;
/// # use vk_graph::driver::device::{Device, DeviceInfo};
/// # use vk_graph::driver::graphic::{GraphicPipeline, GraphicPipelineInfo};
/// # use vk_graph::driver::image::{Image, ImageInfo};
/// # use vk_graph::Graph;
/// # use vk_graph::driver::shader::Shader;
/// # fn main() -> Result<(), DriverError> {
/// # let device = Arc::new(Device::new(DeviceInfo::default())?);
/// # let my_frag_code = [0u8; 1];
/// # let my_vert_code = [0u8; 1];
/// # let vert = Shader::new_vertex(my_vert_code.as_slice());
/// # let frag = Shader::new_fragment(my_frag_code.as_slice());
/// # let info = GraphicPipelineInfo::default();
/// # let my_graphic_pipeline = Arc::new(GraphicPipeline::create(&device, info, [vert, frag])?);
/// # let mut my_graph = Graph::default();
/// # let info = ImageInfo::image_2d(32, 32, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::SAMPLED);
/// # let swapchain_image = my_graph.bind_node(Image::create(&device, info)?);
/// my_graph.begin_cmd().with_name("my draw pass")
///         .bind_pipeline(&my_graphic_pipeline)
///         .store_color(0, swapchain_image)
///         .record_pipeline(move |graphic, bindings| {
///             // During this closure we have access to the draw methods!
///         });
/// # Ok(()) }
/// ```
pub struct Graphic<'a> {
    pub(super) bindings: Bindings<'a>,
    pub(super) cmd_buf: vk::CommandBuffer,
    pub(super) device: &'a Device,
    pub(super) pipeline: GraphicPipeline,
}

impl Graphic<'_> {
    /// Bind an index buffer to the current pass.
    ///
    /// `offset` is the starting offset in bytes within `buffer` used in index buffer address
    /// calculations.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::graphic::{GraphicPipeline, GraphicPipelineInfo};
    /// # use vk_graph::driver::image::{Image, ImageInfo};
    /// # use vk_graph::driver::shader::Shader;
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Arc::new(Device::new(DeviceInfo::default())?);
    /// # let my_frag_code = [0u8; 1];
    /// # let my_vert_code = [0u8; 1];
    /// # let vert = Shader::new_vertex(my_vert_code.as_slice());
    /// # let frag = Shader::new_fragment(my_frag_code.as_slice());
    /// # let info = GraphicPipelineInfo::default();
    /// # let my_graphic_pipeline = Arc::new(GraphicPipeline::create(&device, info, [vert, frag])?);
    /// # let mut my_graph = Graph::default();
    /// # let info = ImageInfo::image_2d(32, 32, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::SAMPLED);
    /// # let swapchain_image = my_graph.bind_node(Image::create(&device, info)?);
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::INDEX_BUFFER);
    /// # let my_idx_buf = Buffer::create(&device, buf_info)?;
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::VERTEX_BUFFER);
    /// # let my_vtx_buf = Buffer::create(&device, buf_info)?;
    /// # let my_idx_buf = my_graph.bind_node(my_idx_buf);
    /// # let my_vtx_buf = my_graph.bind_node(my_vtx_buf);
    /// my_graph.begin_cmd().with_name("my indexed geometry draw pass")
    ///         .bind_pipeline(&my_graphic_pipeline)
    ///         .store_color(0, swapchain_image)
    ///         .read_node(my_idx_buf)
    ///         .read_node(my_vtx_buf)
    ///         .record_pipeline(move |pipeline, bindings| {
    ///             pipeline.bind_index_buffer(my_idx_buf, 0, vk::IndexType::UINT16)
    ///                     .bind_vertex_buffer(0, my_vtx_buf, 0)
    ///                     .draw_indexed(42, 1, 0, 0, 0);
    ///         });
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

        unsafe {
            self.device.cmd_bind_index_buffer(
                self.cmd_buf,
                self.bindings[buffer].handle,
                offset,
                index_ty,
            );
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
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::graphic::{GraphicPipeline, GraphicPipelineInfo};
    /// # use vk_graph::driver::image::{Image, ImageInfo};
    /// # use vk_graph::driver::shader::Shader;
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Arc::new(Device::new(DeviceInfo::default())?);
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::VERTEX_BUFFER);
    /// # let my_vtx_buf = Buffer::create(&device, buf_info)?;
    /// # let my_frag_code = [0u8; 1];
    /// # let my_vert_code = [0u8; 1];
    /// # let vert = Shader::new_vertex(my_vert_code.as_slice());
    /// # let frag = Shader::new_fragment(my_frag_code.as_slice());
    /// # let info = GraphicPipelineInfo::default();
    /// # let my_graphic_pipeline = Arc::new(GraphicPipeline::create(&device, info, [vert, frag])?);
    /// # let mut my_graph = Graph::default();
    /// # let info = ImageInfo::image_2d(32, 32, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::SAMPLED);
    /// # let swapchain_image = my_graph.bind_node(Image::create(&device, info)?);
    /// # let my_vtx_buf = my_graph.bind_node(my_vtx_buf);
    /// my_graph.begin_cmd().with_name("my unindexed geometry draw pass")
    ///         .bind_pipeline(&my_graphic_pipeline)
    ///         .store_color(0, swapchain_image)
    ///         .read_node(my_vtx_buf)
    ///         .record_pipeline(move |pipeline, bindings| {
    ///             pipeline.bind_vertex_buffer(0, my_vtx_buf, 0)
    ///                     .draw(42, 1, 0, 0);
    ///         });
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

        unsafe {
            self.device.cmd_bind_vertex_buffers(
                self.cmd_buf,
                binding,
                slice::from_ref(&self.bindings[buffer].handle),
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
    pub fn bind_vertex_buffers<B>(
        &self,
        first_binding: u32,
        buffer_offsets: impl IntoIterator<Item = (B, vk::DeviceSize)>,
    ) -> &Self
    where
        B: Into<AnyBufferNode>,
    {
        thread_local! {
            static BUFFERS_OFFSETS: RefCell<(Vec<vk::Buffer>, Vec<vk::DeviceSize>)> = Default::default();
        }

        BUFFERS_OFFSETS.with_borrow_mut(|(buffers, offsets)| {
            buffers.clear();
            offsets.clear();

            for (buffer, offset) in buffer_offsets {
                let buffer = buffer.into();

                buffers.push(self.bindings[buffer].handle);
                offsets.push(offset);
            }

            unsafe {
                self.device.cmd_bind_vertex_buffers(
                    self.cmd_buf,
                    first_binding,
                    buffers.as_slice(),
                    offsets.as_slice(),
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
            self.device.cmd_draw(
                self.cmd_buf,
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
            self.device.cmd_draw_indexed(
                self.cmd_buf,
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
    /// # use std::sync::Arc;
    /// # use std::mem::size_of;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::graphic::{GraphicPipeline, GraphicPipelineInfo};
    /// # use vk_graph::driver::image::{Image, ImageInfo};
    /// # use vk_graph::driver::shader::Shader;
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Arc::new(Device::new(DeviceInfo::default())?);
    /// # let my_frag_code = [0u8; 1];
    /// # let my_vert_code = [0u8; 1];
    /// # let vert = Shader::new_vertex(my_vert_code.as_slice());
    /// # let frag = Shader::new_fragment(my_frag_code.as_slice());
    /// # let info = GraphicPipelineInfo::default();
    /// # let my_graphic_pipeline = Arc::new(GraphicPipeline::create(&device, info, [vert, frag])?);
    /// # let mut my_graph = Graph::default();
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::INDEX_BUFFER);
    /// # let my_idx_buf = Buffer::create(&device, buf_info)?;
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::VERTEX_BUFFER);
    /// # let my_vtx_buf = Buffer::create(&device, buf_info)?;
    /// # let my_idx_buf = my_graph.bind_node(my_idx_buf);
    /// # let my_vtx_buf = my_graph.bind_node(my_vtx_buf);
    /// # let info = ImageInfo::image_2d(32, 32, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::SAMPLED);
    /// # let swapchain_image = my_graph.bind_node(Image::create(&device, info)?);
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
    /// let buf_node = my_graph.bind_node(buf);
    ///
    /// my_graph.begin_cmd().with_name("draw a single triangle")
    ///         .bind_pipeline(&my_graphic_pipeline)
    ///         .store_color(0, swapchain_image)
    ///         .read_node(my_idx_buf)
    ///         .read_node(my_vtx_buf)
    ///         .read_node(buf_node)
    ///         .record_pipeline(move |pipeline, bindings| {
    ///             pipeline.bind_index_buffer(my_idx_buf, 0, vk::IndexType::UINT16)
    ///                     .bind_vertex_buffer(0, my_vtx_buf, 0)
    ///                     .draw_indexed_indirect(buf_node, 0, 1, 0);
    ///         });
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

        unsafe {
            self.device.cmd_draw_indexed_indirect(
                self.cmd_buf,
                self.bindings[buffer].handle,
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
        let count_buf = count_buf.into();

        unsafe {
            self.device.cmd_draw_indexed_indirect_count(
                self.cmd_buf,
                self.bindings[buffer].handle,
                offset,
                self.bindings[count_buf].handle,
                count_buf_offset,
                max_draw_count,
                stride,
            );
        }

        self
    }

    /// Draw primitives with indirect parameters and unindexed vertices.
    ///
    /// Behaves otherwise similar to [`Draw::draw_indexed_indirect`].
    #[profiling::function]
    pub fn draw_indirect(
        &self,
        buffer: impl Into<AnyBufferNode>,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> &Self {
        let buffer = buffer.into();

        unsafe {
            self.device.cmd_draw_indirect(
                self.cmd_buf,
                self.bindings[buffer].handle,
                offset,
                draw_count,
                stride,
            );
        }

        self
    }

    /// Draw primitives with indirect parameters, unindexed vertices, and draw count.
    ///
    /// Behaves otherwise similar to [`Draw::draw_indexed_indirect_count`].
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
        let count_buf = count_buf.into();

        unsafe {
            self.device.cmd_draw_indirect_count(
                self.cmd_buf,
                self.bindings[buffer].handle,
                offset,
                self.bindings[count_buf].handle,
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
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::graphic::{GraphicPipeline, GraphicPipelineInfo};
    /// # use vk_graph::driver::image::{Image, ImageInfo};
    /// # use vk_graph::Graph;
    /// # use vk_graph::driver::shader::Shader;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Arc::new(Device::new(DeviceInfo::default())?);
    /// # let my_frag_code = [0u8; 1];
    /// # let my_vert_code = [0u8; 1];
    /// # let vert = Shader::new_vertex(my_vert_code.as_slice());
    /// # let frag = Shader::new_fragment(my_frag_code.as_slice());
    /// # let info = GraphicPipelineInfo::default();
    /// # let my_graphic_pipeline = Arc::new(GraphicPipeline::create(&device, info, [vert, frag])?);
    /// # let info = ImageInfo::image_2d(32, 32, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::SAMPLED);
    /// # let swapchain_image = Image::create(&device, info)?;
    /// # let mut my_graph = Graph::default();
    /// # let swapchain_image = my_graph.bind_node(swapchain_image);
    /// my_graph.begin_cmd().with_name("draw a quad")
    ///         .bind_pipeline(&my_graphic_pipeline)
    ///         .store_color(0, swapchain_image)
    ///         .record_pipeline(move |pipeline, bindings| {
    ///             pipeline.push_constants(0, &[42])
    ///                     .draw(6, 1, 0, 0);
    ///         });
    /// # Ok(()) }
    /// ```
    ///
    /// [gpuinfo.org]: https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxPushConstantsSize&platform=all
    #[profiling::function]
    pub fn push_constants(&self, offset: u32, data: &[u8]) -> &Self {
        for push_const in self.pipeline.push_constants.iter() {
            // Determine the range of the overall pipline push constants which overlap with `data`
            let push_const_end = push_const.offset + push_const.size;
            let data_end = offset + data.len() as u32;
            let end = data_end.min(push_const_end);
            let start = offset.max(push_const.offset);

            if end > start {
                trace!(
                    "      push constants {:?} {}..{}",
                    push_const.stage_flags, start, end
                );

                unsafe {
                    self.device.cmd_push_constants(
                        self.cmd_buf,
                        self.pipeline.layout,
                        push_const.stage_flags,
                        start,
                        &data[(start - offset) as usize..(end - offset) as usize],
                    );
                }
            }
        }

        self
    }

    /// Set scissor rectangle dynamically for a pass.
    #[profiling::function]
    pub fn set_scissor(&self, scissor: &vk::Rect2D) -> &Self {
        unsafe {
            self.device
                .cmd_set_scissor(self.cmd_buf, 0, slice::from_ref(scissor));
        }

        self
    }

    /// Set scissor rectangles dynamically for a pass.
    #[profiling::function]
    pub fn set_scissors<S>(
        &self,
        first_scissor: u32,
        scissors: impl IntoIterator<Item = S>,
    ) -> &Self
    where
        S: Into<vk::Rect2D>,
    {
        thread_local! {
            static TLS: RefCell<Vec<vk::Rect2D>> = Default::default();
        }

        TLS.with_borrow_mut(|tls| {
            tls.clear();
            tls.extend(scissors.into_iter().map(Into::into));

            unsafe {
                self.device
                    .cmd_set_scissor(self.cmd_buf, first_scissor, tls);
            }
        });

        self
    }

    /// Set the viewport dynamically for a pass.
    #[profiling::function]
    pub fn set_viewport(&self, viewport: &vk::Viewport) -> &Self {
        unsafe {
            self.device
                .cmd_set_viewport(self.cmd_buf, 0, slice::from_ref(viewport));
        }

        self
    }

    /// Set the viewports dynamically for a pass.
    #[profiling::function]
    pub fn set_viewports<V>(
        &self,
        first_viewport: u32,
        viewports: impl IntoIterator<Item = V>,
    ) -> &Self
    where
        V: Into<vk::Viewport>,
    {
        thread_local! {
            static TLS: RefCell<Vec<vk::Viewport>> = Default::default();
        }

        TLS.with_borrow_mut(|tls| {
            tls.clear();
            tls.extend(viewports.into_iter().map(Into::into));

            unsafe {
                self.device
                    .cmd_set_viewport(self.cmd_buf, first_viewport, tls);
            }
        });

        self
    }
}

// NOTE: local implementation of type from super module
impl PipelineCommandRef<'_, GraphicPipeline> {
    /// Specifies `VK_ATTACHMENT_LOAD_OP_DONT_CARE` for the render pass attachment, and loads an
    /// image into the framebuffer.
    pub fn attach_color(
        self,
        attachment_idx: AttachmentIndex,
        image: impl Into<AnyImageNode>,
    ) -> Self {
        let image: AnyImageNode = image.into();
        let image_info = image.info(&self.cmd.graph.bindings);
        let image_view_info: ImageViewInfo = image_info.into();

        self.attach_color_as(attachment_idx, image, image_view_info)
    }

    /// Specifies `VK_ATTACHMENT_LOAD_OP_DONT_CARE` for the render pass attachment, and loads an
    /// image into the framebuffer.
    pub fn attach_color_as(
        mut self,
        attachment_idx: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
    ) -> Self {
        let image = image.into();
        let image_view_info = image_view_info.into();
        let node_idx = image.index();
        let (_, sample_count) = self.image_info(node_idx);

        debug_assert!(
            !self
                .cmd
                .as_ref()
                .execs
                .last()
                .unwrap()
                .color_clears
                .contains_key(&attachment_idx),
            "color attachment {attachment_idx} already attached via clear"
        );
        debug_assert!(
            !self
                .cmd
                .as_ref()
                .execs
                .last()
                .unwrap()
                .color_loads
                .contains_key(&attachment_idx),
            "color attachment {attachment_idx} already attached via load"
        );

        self.cmd
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
            .color_attachments
            .insert(
                attachment_idx,
                Attachment::new(image_view_info, sample_count, node_idx),
            );

        debug_assert!(
            Attachment::are_compatible(
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_resolves
                    .get(&attachment_idx)
                    .map(|(attachment, _)| *attachment),
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_attachments
                    .get(&attachment_idx)
                    .copied()
            ),
            "color attachment {attachment_idx} incompatible with existing resolve"
        );
        debug_assert!(
            Attachment::are_compatible(
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_stores
                    .get(&attachment_idx)
                    .copied(),
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_attachments
                    .get(&attachment_idx)
                    .copied()
            ),
            "color attachment {attachment_idx} incompatible with existing store"
        );

        self.cmd.push_node_access(
            image,
            AccessType::ColorAttachmentWrite,
            Subresource::Image(image_view_info.into()),
        );

        self
    }

    /// Specifies `VK_ATTACHMENT_LOAD_OP_DONT_CARE` for the render pass attachment, and loads an
    /// image into the framebuffer.
    pub fn attach_depth_stencil(self, image: impl Into<AnyImageNode>) -> Self {
        let image: AnyImageNode = image.into();
        let image_info = image.info(&self.cmd.graph.bindings);
        let image_view_info: ImageViewInfo = image_info.into();

        self.attach_depth_stencil_as(image, image_view_info)
    }

    /// Specifies `VK_ATTACHMENT_LOAD_OP_DONT_CARE` for the render pass attachment, and loads an
    /// image into the framebuffer.
    pub fn attach_depth_stencil_as(
        mut self,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
    ) -> Self {
        let image = image.into();
        let image_view_info = image_view_info.into();
        let node_idx = image.index();
        let (_, sample_count) = self.image_info(node_idx);

        debug_assert!(
            self.cmd
                .as_ref()
                .execs
                .last()
                .unwrap()
                .depth_stencil_clear
                .is_none(),
            "depth/stencil attachment already attached via clear"
        );
        debug_assert!(
            self.cmd
                .as_ref()
                .execs
                .last()
                .unwrap()
                .depth_stencil_load
                .is_none(),
            "depth/stencil attachment already attached via load"
        );

        self.cmd
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
            .depth_stencil_attachment =
            Some(Attachment::new(image_view_info, sample_count, node_idx));

        debug_assert!(
            Attachment::are_compatible(
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .depth_stencil_resolve
                    .map(|(attachment, ..)| attachment),
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .depth_stencil_attachment
            ),
            "depth/stencil attachment incompatible with existing resolve"
        );
        debug_assert!(
            Attachment::are_compatible(
                self.cmd.as_ref().execs.last().unwrap().depth_stencil_store,
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .depth_stencil_attachment
            ),
            "depth/stencil attachment incompatible with existing store"
        );

        self.cmd.push_node_access(
            image,
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
            Subresource::Image(image_view_info.into()),
        );

        self
    }

    /// Clears the render pass attachment of any existing data.
    pub fn clear_color(
        self,
        attachment_idx: AttachmentIndex,
        image: impl Into<AnyImageNode>,
    ) -> Self {
        self.clear_color_value(attachment_idx, image, [0.0, 0.0, 0.0, 0.0])
    }

    /// Clears the render pass attachment of any existing data.
    pub fn clear_color_value(
        self,
        attachment_idx: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        color: impl Into<ClearColorValue>,
    ) -> Self {
        let image: AnyImageNode = image.into();
        let image_info = image.info(&self.cmd.graph.bindings);
        let image_view_info: ImageViewInfo = image_info.into();

        self.clear_color_value_as(attachment_idx, image, color, image_view_info)
    }

    /// Clears the render pass attachment of any existing data.
    pub fn clear_color_value_as(
        mut self,
        attachment_idx: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        color: impl Into<ClearColorValue>,
        image_view_info: impl Into<ImageViewInfo>,
    ) -> Self {
        let image = image.into();
        let image_view_info = image_view_info.into();
        let node_idx = image.index();
        let (_, sample_count) = self.image_info(node_idx);

        let color = color.into();

        debug_assert!(
            !self
                .cmd
                .as_ref()
                .execs
                .last()
                .unwrap()
                .color_attachments
                .contains_key(&attachment_idx),
            "color attachment {attachment_idx} already attached"
        );
        debug_assert!(
            !self
                .cmd
                .as_ref()
                .execs
                .last()
                .unwrap()
                .color_loads
                .contains_key(&attachment_idx),
            "color attachment {attachment_idx} already attached via load"
        );

        self.cmd
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
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
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_resolves
                    .get(&attachment_idx)
                    .map(|(attachment, _)| *attachment),
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_clears
                    .get(&attachment_idx)
                    .map(|(attachment, _)| *attachment)
            ),
            "color attachment {attachment_idx} clear incompatible with existing resolve"
        );
        debug_assert!(
            Attachment::are_compatible(
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_stores
                    .get(&attachment_idx)
                    .copied(),
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
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
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
            .accesses
            .get_mut(&node_idx)
        {
            for SubresourceAccess {
                access,
                subresource,
            } in accesses
            {
                let access_image_range = *subresource.as_image().unwrap();
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

        self.cmd
            .push_node_access(image, image_access, Subresource::Image(image_range));

        self
    }

    /// Clears the render pass attachment of any existing data.
    pub fn clear_depth_stencil(self, image: impl Into<AnyImageNode>) -> Self {
        self.clear_depth_stencil_value(image, 1.0, 0)
    }

    /// Clears the render pass attachment of any existing data.
    pub fn clear_depth_stencil_value(
        self,
        image: impl Into<AnyImageNode>,
        depth: f32,
        stencil: u32,
    ) -> Self {
        let image: AnyImageNode = image.into();
        let image_info = image.info(&self.cmd.graph.bindings);
        let image_view_info: ImageViewInfo = image_info.into();

        self.clear_depth_stencil_value_as(image, depth, stencil, image_view_info)
    }

    /// Clears the render pass attachment of any existing data.
    pub fn clear_depth_stencil_value_as(
        mut self,
        image: impl Into<AnyImageNode>,
        depth: f32,
        stencil: u32,
        image_view_info: impl Into<ImageViewInfo>,
    ) -> Self {
        let image = image.into();
        let image_view_info = image_view_info.into();
        let node_idx = image.index();
        let (_, sample_count) = self.image_info(node_idx);

        debug_assert!(
            self.cmd
                .as_ref()
                .execs
                .last()
                .unwrap()
                .depth_stencil_attachment
                .is_none(),
            "depth/stencil attachment already attached"
        );
        debug_assert!(
            self.cmd
                .as_ref()
                .execs
                .last()
                .unwrap()
                .depth_stencil_load
                .is_none(),
            "depth/stencil attachment already attached via load"
        );

        self.cmd
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
            .depth_stencil_clear = Some((
            Attachment::new(image_view_info, sample_count, node_idx),
            vk::ClearDepthStencilValue { depth, stencil },
        ));

        debug_assert!(
            Attachment::are_compatible(
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .depth_stencil_resolve
                    .map(|(attachment, ..)| attachment),
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .depth_stencil_clear
                    .map(|(attachment, _)| attachment)
            ),
            "depth/stencil attachment clear incompatible with existing resolve"
        );
        debug_assert!(
            Attachment::are_compatible(
                self.cmd.as_ref().execs.last().unwrap().depth_stencil_store,
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
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
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
            .accesses
            .get_mut(&node_idx)
        {
            for SubresourceAccess {
                access,
                subresource,
            } in accesses
            {
                let access_image_range = *subresource.as_image().unwrap();
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

        self.cmd
            .push_node_access(image, image_access, Subresource::Image(image_range));

        self
    }

    fn image_info(&self, node_idx: NodeIndex) -> (vk::Format, SampleCount) {
        let image_info = self.cmd.graph.bindings[node_idx]
            .as_driver_image()
            .unwrap()
            .info;

        (image_info.fmt, image_info.sample_count)
    }

    /// Specifies `VK_ATTACHMENT_LOAD_OP_LOAD` for the render pass attachment, and loads an image
    /// into the framebuffer.
    pub fn load_color(
        self,
        attachment_idx: AttachmentIndex,
        image: impl Into<AnyImageNode>,
    ) -> Self {
        let image: AnyImageNode = image.into();
        let image_info = self.node_info(image);

        // Use the plain node information as the whole view of the node
        let image_view_info = image_info;

        self.load_color_as(attachment_idx, image, image_view_info)
    }

    /// Specifies `VK_ATTACHMENT_LOAD_OP_LOAD` for the render pass attachment, and loads an image
    /// into the framebuffer.
    pub fn load_color_as(
        mut self,
        attachment_idx: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
    ) -> Self {
        let image = image.into();
        let image_view_info = image_view_info.into();
        let node_idx = image.index();
        let (_, sample_count) = self.image_info(node_idx);

        debug_assert!(
            !self
                .cmd
                .as_ref()
                .execs
                .last()
                .unwrap()
                .color_attachments
                .contains_key(&attachment_idx),
            "color attachment {attachment_idx} already attached"
        );
        debug_assert!(
            !self
                .cmd
                .as_ref()
                .execs
                .last()
                .unwrap()
                .color_clears
                .contains_key(&attachment_idx),
            "color attachment {attachment_idx} already attached via clear"
        );

        self.cmd
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
            .color_loads
            .insert(
                attachment_idx,
                Attachment::new(image_view_info, sample_count, node_idx),
            );

        debug_assert!(
            Attachment::are_compatible(
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_resolves
                    .get(&attachment_idx)
                    .map(|(attachment, _)| *attachment),
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_loads
                    .get(&attachment_idx)
                    .copied()
            ),
            "color attachment {attachment_idx} load incompatible with existing resolve"
        );
        debug_assert!(
            Attachment::are_compatible(
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_stores
                    .get(&attachment_idx)
                    .copied(),
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
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
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
            .accesses
            .get_mut(&node_idx)
        {
            for SubresourceAccess {
                access,
                subresource,
            } in accesses
            {
                let access_image_range = *subresource.as_image().unwrap();
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

        self.cmd
            .push_node_access(image, image_access, Subresource::Image(image_range));

        self
    }

    /// Specifies `VK_ATTACHMENT_LOAD_OP_LOAD` for the render pass attachment, and loads an image
    /// into the framebuffer.
    pub fn load_depth_stencil(self, image: impl Into<AnyImageNode>) -> Self {
        let image: AnyImageNode = image.into();
        let image_info = self.node_info(image);
        let image_view_info: ImageViewInfo = image_info.into();

        self.load_depth_stencil_as(image, image_view_info)
    }

    /// Specifies `VK_ATTACHMENT_LOAD_OP_LOAD` for the render pass attachment, and loads an image
    /// into the framebuffer.
    pub fn load_depth_stencil_as(
        mut self,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
    ) -> Self {
        let image = image.into();
        let image_view_info = image_view_info.into();
        let node_idx = image.index();
        let (_, sample_count) = self.image_info(node_idx);

        debug_assert!(
            self.cmd
                .as_ref()
                .execs
                .last()
                .unwrap()
                .depth_stencil_attachment
                .is_none(),
            "depth/stencil attachment already attached"
        );
        debug_assert!(
            self.cmd
                .as_ref()
                .execs
                .last()
                .unwrap()
                .depth_stencil_clear
                .is_none(),
            "depth/stencil attachment already attached via clear"
        );

        self.cmd
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
            .depth_stencil_load = Some(Attachment::new(image_view_info, sample_count, node_idx));

        debug_assert!(
            Attachment::are_compatible(
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .depth_stencil_resolve
                    .map(|(attachment, ..)| attachment),
                self.cmd.as_ref().execs.last().unwrap().depth_stencil_load
            ),
            "depth/stencil attachment load incompatible with existing resolve"
        );
        debug_assert!(
            Attachment::are_compatible(
                self.cmd.as_ref().execs.last().unwrap().depth_stencil_store,
                self.cmd.as_ref().execs.last().unwrap().depth_stencil_load
            ),
            "depth/stencil attachment load incompatible with existing store"
        );

        let mut image_access = AccessType::DepthStencilAttachmentRead;
        let image_range = image_view_info.into();

        // Upgrade existing write access to read-write
        if let Some(accesses) = self
            .cmd
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
            .accesses
            .get_mut(&node_idx)
        {
            for SubresourceAccess {
                access,
                subresource,
            } in accesses
            {
                let access_image_range = *subresource.as_image().unwrap();
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

        self.cmd
            .push_node_access(image, image_access, Subresource::Image(image_range));

        self
    }

    /// Begin recording a graphics command buffer.
    pub fn record_pipeline(
        mut self,
        func: impl FnOnce(Graphic<'_>, Bindings<'_>) + Send + 'static,
    ) -> Self {
        let pipeline = 
            self.cmd
                .as_ref()
                .execs
                .last()
                .unwrap()
                .pipeline
                .as_ref()
                .unwrap()
                .unwrap_graphic().clone()
        ;

        self.cmd.push_execute(move |device, cmd_buf, bindings| {
            func(
                Graphic {
                    bindings,
                    cmd_buf,
                    device,
                    pipeline,
                },
                bindings,
            );
        });

        self
    }

    /// Resolves a multisample framebuffer to a non-multisample image for the render pass
    /// attachment.
    pub fn resolve_color(
        self,
        src_attachment_idx: AttachmentIndex,
        dst_attachment_idx: AttachmentIndex,
        image: impl Into<AnyImageNode>,
    ) -> Self {
        let image: AnyImageNode = image.into();
        let image_info = self.node_info(image);

        // Use the plain node information as the whole view of the node
        let image_view_info = image_info;

        self.resolve_color_as(
            src_attachment_idx,
            dst_attachment_idx,
            image,
            image_view_info,
        )
    }

    /// Resolves a multisample framebuffer to a non-multisample image for the render pass
    /// attachment.
    pub fn resolve_color_as(
        mut self,
        src_attachment_idx: AttachmentIndex,
        dst_attachment_idx: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
    ) -> Self {
        let image = image.into();
        let image_view_info = image_view_info.into();
        let node_idx = image.index();
        let (_, sample_count) = self.image_info(node_idx);

        self.cmd
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
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
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_attachments
                    .get(&dst_attachment_idx)
                    .copied(),
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_resolves
                    .get(&dst_attachment_idx)
                    .map(|(attachment, _)| *attachment)
            ),
            "color attachment {dst_attachment_idx} resolve incompatible with existing attachment"
        );
        debug_assert!(
            Attachment::are_compatible(
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_clears
                    .get(&dst_attachment_idx)
                    .map(|(attachment, _)| *attachment),
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_resolves
                    .get(&dst_attachment_idx)
                    .map(|(attachment, _)| *attachment)
            ),
            "color attachment {dst_attachment_idx} resolve incompatible with existing clear"
        );
        debug_assert!(
            Attachment::are_compatible(
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_loads
                    .get(&dst_attachment_idx)
                    .copied(),
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
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
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
            .accesses
            .get_mut(&node_idx)
        {
            for SubresourceAccess {
                access,
                subresource,
            } in accesses
            {
                let access_image_range = *subresource.as_image().unwrap();
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

                // If the resolve access is a subset of the existing access range there is no need
                // to push a new access
                if image_subresource_range_contains(access_image_range, image_range) {
                    return self;
                }
            }
        }

        self.cmd
            .push_node_access(image, image_access, Subresource::Image(image_range));

        self
    }

    /// Resolves a multisample framebuffer to a non-multisample image for the render pass
    /// attachment.
    pub fn resolve_depth_stencil(
        self,
        dst_attachment_idx: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        depth_mode: Option<ResolveMode>,
        stencil_mode: Option<ResolveMode>,
    ) -> Self {
        let image: AnyImageNode = image.into();
        let image_info = self.node_info(image);

        // Use the plain node information as the whole view of the node
        let image_view_info = image_info;

        self.resolve_depth_stencil_as(
            dst_attachment_idx,
            image,
            image_view_info,
            depth_mode,
            stencil_mode,
        )
    }

    /// Resolves a multisample framebuffer to a non-multisample image for the render pass
    /// attachment.
    pub fn resolve_depth_stencil_as(
        mut self,
        dst_attachment_idx: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
        depth_mode: Option<ResolveMode>,
        stencil_mode: Option<ResolveMode>,
    ) -> Self {
        let image = image.into();
        let image_view_info = image_view_info.into();
        let node_idx = image.index();
        let (_, sample_count) = self.image_info(node_idx);

        self.cmd
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
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
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
            .accesses
            .get_mut(&node_idx)
        {
            for SubresourceAccess {
                access,
                subresource,
            } in accesses
            {
                let access_image_range = *subresource.as_image().unwrap();
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

                // If the resolve access is a subset of the existing access range there is no need
                // to push a new access
                if image_subresource_range_contains(access_image_range, image_range) {
                    return self;
                }
            }
        }

        self.cmd
            .push_node_access(image, image_access, Subresource::Image(image_range));

        self
    }

    /// Sets a particular depth/stencil mode.
    pub fn set_depth_stencil(mut self, depth_stencil: DepthStencilMode) -> Self {
        let pass = self.cmd.as_mut();
        let exec = pass.execs.last_mut().unwrap();

        assert!(exec.depth_stencil.is_none());

        exec.depth_stencil = Some(depth_stencil);

        self
    }

    /// Sets multiview view and correlation masks.
    ///
    /// See [`VkRenderPassMultiviewCreateInfo`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkRenderPassMultiviewCreateInfo.html#_description).
    pub fn set_multiview(mut self, view_mask: u32, correlated_view_mask: u32) -> Self {
        let pass = self.cmd.as_mut();
        let exec = pass.execs.last_mut().unwrap();

        exec.correlated_view_mask = correlated_view_mask;
        exec.view_mask = view_mask;

        self
    }

    /// Sets the [`renderArea`](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkRenderPassBeginInfo.html#_c_specification)
    /// field when beginning a render pass.
    ///
    /// NOTE: Setting this value will cause the viewport and scissor to be unset, which is not the default
    /// behavior. When this value is set you should call `set_viewport` and `set_scissor` on the subpass.
    ///
    /// If not set, this value defaults to the first loaded, resolved, or stored attachment dimensions and
    /// sets the viewport and scissor to the same values, with a `0..1` depth if not specified by
    /// `set_depth_stencil`.
    pub fn set_render_area(mut self, x: i32, y: i32, width: u32, height: u32) -> Self {
        self.cmd.as_mut().execs.last_mut().unwrap().render_area = Some(Area {
            height,
            width,
            x,
            y,
        });

        self
    }

    /// Specifies `VK_ATTACHMENT_STORE_OP_STORE` for the render pass attachment, and stores the
    /// rendered pixels into an image.
    pub fn store_color(
        self,
        attachment_idx: AttachmentIndex,
        image: impl Into<AnyImageNode>,
    ) -> Self {
        let image: AnyImageNode = image.into();
        let image_info = self.node_info(image);

        // Use the plain node information as the whole view of the node
        let image_view_info = image_info;

        self.store_color_as(attachment_idx, image, image_view_info)
    }

    /// Specifies `VK_ATTACHMENT_STORE_OP_STORE` for the render pass attachment, and stores the
    /// rendered pixels into an image.
    pub fn store_color_as(
        mut self,
        attachment_idx: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
    ) -> Self {
        let image = image.into();
        let image_view_info = image_view_info.into();
        let node_idx = image.index();
        let (_, sample_count) = self.image_info(node_idx);

        self.cmd
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
            .color_stores
            .insert(
                attachment_idx,
                Attachment::new(image_view_info, sample_count, node_idx),
            );

        debug_assert!(
            Attachment::are_compatible(
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_attachments
                    .get(&attachment_idx)
                    .copied(),
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_stores
                    .get(&attachment_idx)
                    .copied()
            ),
            "color attachment {attachment_idx} store incompatible with existing attachment"
        );
        debug_assert!(
            Attachment::are_compatible(
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_clears
                    .get(&attachment_idx)
                    .map(|(attachment, _)| *attachment),
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_stores
                    .get(&attachment_idx)
                    .copied()
            ),
            "color attachment {attachment_idx} store incompatible with existing clear"
        );
        debug_assert!(
            Attachment::are_compatible(
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .color_loads
                    .get(&attachment_idx)
                    .copied(),
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
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
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
            .accesses
            .get_mut(&node_idx)
        {
            for SubresourceAccess {
                access,
                subresource,
            } in accesses
            {
                let access_image_range = *subresource.as_image().unwrap();
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

        self.cmd
            .push_node_access(image, image_access, Subresource::Image(image_range));

        self
    }

    /// Specifies `VK_ATTACHMENT_STORE_OP_STORE` for the render pass attachment, and stores the
    /// rendered pixels into an image.
    pub fn store_depth_stencil(self, image: impl Into<AnyImageNode>) -> Self {
        let image: AnyImageNode = image.into();
        let image_info = self.node_info(image);

        // Use the plain node information as the whole view of the node
        let image_view_info = image_info;

        self.store_depth_stencil_as(image, image_view_info)
    }

    /// Specifies `VK_ATTACHMENT_STORE_OP_STORE` for the render pass attachment, and stores the
    /// rendered pixels into an image.
    ///
    /// _NOTE:_ Order matters, call store after clear or load.
    pub fn store_depth_stencil_as(
        mut self,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
    ) -> Self {
        let image = image.into();
        let image_view_info = image_view_info.into();
        let node_idx = image.index();
        let (_, sample_count) = self.image_info(node_idx);

        self.cmd
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
            .depth_stencil_store = Some(Attachment::new(image_view_info, sample_count, node_idx));

        debug_assert!(
            Attachment::are_compatible(
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .depth_stencil_attachment,
                self.cmd.as_ref().execs.last().unwrap().depth_stencil_store
            ),
            "depth/stencil attachment store incompatible with existing attachment"
        );
        debug_assert!(
            Attachment::are_compatible(
                self.cmd
                    .as_ref()
                    .execs
                    .last()
                    .unwrap()
                    .depth_stencil_clear
                    .map(|(attachment, _)| attachment),
                self.cmd.as_ref().execs.last().unwrap().depth_stencil_store
            ),
            "depth/stencil attachment store incompatible with existing clear"
        );
        debug_assert!(
            Attachment::are_compatible(
                self.cmd.as_ref().execs.last().unwrap().depth_stencil_load,
                self.cmd.as_ref().execs.last().unwrap().depth_stencil_store
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
            .as_mut()
            .execs
            .last_mut()
            .unwrap()
            .accesses
            .get_mut(&node_idx)
        {
            for SubresourceAccess {
                access,
                subresource,
            } in accesses
            {
                let access_image_range = *subresource.as_image().unwrap();
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

        self.cmd
            .push_node_access(image, image_access, Subresource::Image(image_range));

        self
    }
}
