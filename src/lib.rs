/*!

This crate provides a high-performance [Vulkan](https://www.vulkan.org/) driver featuring automated
resource management and execution.

For an overview, including installation and typical usage, see the
[Guide Book](https://attackgoat.github.io/vk-graph).




*/

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod cmd;
pub mod driver;
pub mod node;
pub mod pool;

mod queue;

use std::sync::Arc;

use crate::{
    cmd::{ClearColorValue, CommandBuffer},
    driver::{
        accel_struct::AccelerationStructure, buffer::Buffer, image::Image,
        swapchain::SwapchainImage,
    },
    pool::Lease,
};

pub use self::queue::Queue;

#[allow(deprecated)]
pub use self::deprecated::{Display, DisplayInfo, DisplayInfoBuilder};

use {
    self::{
        cmd::{AttachmentIndex, Command, Descriptor, SubresourceAccess, ViewInfo},
        node::{
            AccelerationStructureLeaseNode, AccelerationStructureNode,
            AnyAccelerationStructureNode, AnyBufferNode, AnyImageNode, BufferLeaseNode, BufferNode,
            ImageLeaseNode, ImageNode, SwapchainImageNode,
        },
    },
    crate::driver::{
        DescriptorBindingMap,
        compute::ComputePipeline,
        format_aspect_mask, format_texel_block_extent, format_texel_block_size,
        graphic::{DepthStencilInfo, GraphicPipeline},
        image::{ImageInfo, ImageViewInfo, SampleCount},
        image_subresource_range_from_layers,
        ray_trace::RayTracePipeline,
        render_pass::ResolveMode,
        shader::PipelineDescriptorInfo,
    },
    ash::vk,
    std::{
        cmp::Ord,
        collections::{BTreeMap, HashMap},
        fmt::{Debug, Formatter},
        ops::Range,
    },
    vk_sync::AccessType,
};

type ExecFn = Box<dyn FnOnce(CommandBuffer) + Send>;
type NodeIndex = usize;

#[derive(Debug)]
#[doc(hidden)]
pub enum AnyResource {
    AccelerationStructure(Arc<AccelerationStructure>),
    AccelerationStructureLease(Arc<Lease<AccelerationStructure>>),
    Buffer(Arc<Buffer>),
    BufferLease(Arc<Lease<Buffer>>),
    Image(Arc<Image>),
    ImageLease(Arc<Lease<Image>>),
    SwapchainImage(Box<SwapchainImage>),
}

impl AnyResource {
    fn as_accel_struct(&self) -> Option<&AccelerationStructure> {
        Some(match self {
            Self::AccelerationStructure(resource) => resource,
            Self::AccelerationStructureLease(resource) => resource,
            _ => return None,
        })
    }

    fn as_buffer(&self) -> Option<&Buffer> {
        Some(match self {
            Self::Buffer(resource) => resource,
            Self::BufferLease(resource) => resource,
            _ => return None,
        })
    }

    fn as_image(&self) -> Option<&Image> {
        Some(match self {
            Self::Image(resource) => resource,
            Self::ImageLease(resource) => resource,
            Self::SwapchainImage(resource) => resource,
            _ => return None,
        })
    }

    fn as_swapchain_image(&self) -> Option<&SwapchainImage> {
        Some(match self {
            Self::SwapchainImage(resource) => resource,
            _ => return None,
        })
    }

    fn expect_accel_struct(&self) -> &AccelerationStructure {
        self.as_accel_struct()
            .expect("missing acceleration structure resource")
    }

    fn expect_buffer(&self) -> &Buffer {
        self.as_buffer().expect("missing buffer resource")
    }

    fn expect_image(&self) -> &Image {
        self.as_image().expect("missing image resource")
    }
}

#[derive(Clone, Copy, Debug)]
struct Attachment {
    array_layer_count: u32,
    aspect_mask: vk::ImageAspectFlags,
    base_array_layer: u32,
    base_mip_level: u32,
    format: vk::Format,
    mip_level_count: u32,
    sample_count: SampleCount,
    target: NodeIndex,
}

impl Attachment {
    fn new(image_view_info: ImageViewInfo, sample_count: SampleCount, target: NodeIndex) -> Self {
        Self {
            array_layer_count: image_view_info.array_layer_count,
            aspect_mask: image_view_info.aspect_mask,
            base_array_layer: image_view_info.base_array_layer,
            base_mip_level: image_view_info.base_mip_level,
            format: image_view_info.fmt,
            mip_level_count: image_view_info.mip_level_count,
            sample_count,
            target,
        }
    }

    fn are_compatible(lhs: Option<Self>, rhs: Option<Self>) -> bool {
        // Two attachment references are compatible if they have matching format and sample
        // count, or are both VK_ATTACHMENT_UNUSED or the pointer that would contain the
        // reference is NULL.
        let (Some(lhs), Some(rhs)) = (lhs, rhs) else {
            return true;
        };

        Self::are_identical(lhs, rhs)
    }

    fn are_identical(lhs: Self, rhs: Self) -> bool {
        lhs.array_layer_count == rhs.array_layer_count
            && lhs.base_array_layer == rhs.base_array_layer
            && lhs.base_mip_level == rhs.base_mip_level
            && lhs.format == rhs.format
            && lhs.mip_level_count == rhs.mip_level_count
            && lhs.sample_count == rhs.sample_count
            && lhs.target == rhs.target
    }

    fn image_view_info(self, image_info: ImageInfo) -> ImageViewInfo {
        image_info
            .into_builder()
            .array_layer_count(self.array_layer_count)
            .mip_level_count(self.mip_level_count)
            .fmt(self.format)
            .into_image_view()
            .aspect_mask(self.aspect_mask)
            .base_array_layer(self.base_array_layer)
            .base_mip_level(self.base_mip_level)
            .build()
    }
}

#[derive(Default)]
struct Execution {
    accesses: HashMap<NodeIndex, Vec<SubresourceAccess>>,
    bindings: BTreeMap<Descriptor, (NodeIndex, ViewInfo)>,

    correlated_view_mask: u32,
    depth_stencil: Option<DepthStencilInfo>,
    render_area: Option<vk::Rect2D>,
    view_mask: u32,

    color_attachments: HashMap<AttachmentIndex, Attachment>,
    color_clears: HashMap<AttachmentIndex, (Attachment, [f32; 4])>,
    color_loads: HashMap<AttachmentIndex, Attachment>,
    color_resolves: HashMap<AttachmentIndex, (Attachment, AttachmentIndex)>,
    color_stores: HashMap<AttachmentIndex, Attachment>,
    depth_stencil_attachment: Option<Attachment>,
    depth_stencil_clear: Option<(Attachment, vk::ClearDepthStencilValue)>,
    depth_stencil_load: Option<Attachment>,
    depth_stencil_resolve: Option<(
        Attachment,
        AttachmentIndex,
        Option<ResolveMode>,
        Option<ResolveMode>,
    )>,
    depth_stencil_store: Option<Attachment>,

    func: Option<ExecutionFunction>,
    pipeline: Option<ExecutionPipeline>,
}

impl Debug for Execution {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // The only field missing is func which cannot easily be implemented because it is a
        // FnOnce.
        f.debug_struct("Execution")
            .field("accesses", &self.accesses)
            .field("bindings", &self.bindings)
            .field("depth_stencil", &self.depth_stencil)
            .field("color_attachments", &self.color_attachments)
            .field("color_clears", &self.color_clears)
            .field("color_loads", &self.color_loads)
            .field("color_resolves", &self.color_resolves)
            .field("color_stores", &self.color_stores)
            .field("depth_stencil_attachment", &self.depth_stencil_attachment)
            .field("depth_stencil_clear", &self.depth_stencil_clear)
            .field("depth_stencil_load", &self.depth_stencil_load)
            .field("depth_stencil_resolve", &self.depth_stencil_resolve)
            .field("depth_stencil_store", &self.depth_stencil_store)
            .field("pipeline", &self.pipeline)
            .finish()
    }
}

struct ExecutionFunction(ExecFn);

#[derive(Clone, Debug)]
enum ExecutionPipeline {
    Compute(ComputePipeline),
    Graphic(GraphicPipeline),
    RayTrace(RayTracePipeline),
}

impl ExecutionPipeline {
    fn as_graphic(&self) -> Option<&GraphicPipeline> {
        if let Self::Graphic(pipeline) = self {
            Some(pipeline)
        } else {
            None
        }
    }

    fn bind_point(&self) -> vk::PipelineBindPoint {
        match self {
            ExecutionPipeline::Compute(_) => vk::PipelineBindPoint::COMPUTE,
            ExecutionPipeline::Graphic(_) => vk::PipelineBindPoint::GRAPHICS,
            ExecutionPipeline::RayTrace(_) => vk::PipelineBindPoint::RAY_TRACING_KHR,
        }
    }

    fn descriptor_bindings(&self) -> &DescriptorBindingMap {
        match self {
            ExecutionPipeline::Compute(pipeline) => &pipeline.inner.descriptor_bindings,
            ExecutionPipeline::Graphic(pipeline) => &pipeline.inner.descriptor_bindings,
            ExecutionPipeline::RayTrace(pipeline) => &pipeline.inner.descriptor_bindings,
        }
    }

    fn descriptor_info(&self) -> &PipelineDescriptorInfo {
        match self {
            ExecutionPipeline::Compute(pipeline) => &pipeline.inner.descriptor_info,
            ExecutionPipeline::Graphic(pipeline) => &pipeline.inner.descriptor_info,
            ExecutionPipeline::RayTrace(pipeline) => &pipeline.inner.descriptor_info,
        }
    }

    fn expect_compute(&self) -> &ComputePipeline {
        if let Self::Compute(pipeline) = self {
            pipeline
        } else {
            panic!("missing compute pipeline")
        }
    }

    fn expect_graphic(&self) -> &GraphicPipeline {
        self.as_graphic().expect("missing graphic pipeline")
    }

    fn expect_ray_trace(&self) -> &RayTracePipeline {
        if let Self::RayTrace(pipeline) = self {
            pipeline
        } else {
            panic!("missing ray trace pipeline")
        }
    }

    fn layout(&self) -> vk::PipelineLayout {
        match self {
            ExecutionPipeline::Compute(pipeline) => pipeline.inner.layout,
            ExecutionPipeline::Graphic(pipeline) => pipeline.inner.layout,
            ExecutionPipeline::RayTrace(pipeline) => pipeline.inner.layout,
        }
    }

    fn stage(&self) -> vk::PipelineStageFlags {
        match self {
            ExecutionPipeline::Compute(_) => vk::PipelineStageFlags::COMPUTE_SHADER,
            ExecutionPipeline::Graphic(_) => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ExecutionPipeline::RayTrace(_) => vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        }
    }
}

#[derive(Debug)]
struct CommandData {
    execs: Vec<Execution>,
    name: Option<String>,
}

impl CommandData {
    fn descriptor_pools_sizes(
        &self,
    ) -> impl Iterator<Item = impl Iterator<Item = (&vk::DescriptorType, &u32)>> {
        self.execs
            .iter()
            .flat_map(|exec| &exec.pipeline)
            .map(|pipeline| {
                pipeline
                    .descriptor_info()
                    .pool_sizes
                    .values()
                    .flat_map(HashMap::iter)
            })
    }

    fn expect_first_exec(&self) -> &Execution {
        self.execs.first().expect("missing command execution")
    }

    /// # Panics
    ///
    /// Panics if the execution list is empty (a command always has at least one execution).
    fn expect_last_exec(&self) -> &Execution {
        self.execs.last().expect("missing command execution")
    }

    /// # Panics
    ///
    /// Panics if the execution list is empty (a command always has at least one execution).
    fn expect_last_exec_mut(&mut self) -> &mut Execution {
        self.execs.last_mut().expect("missing command execution")
    }

    fn expect_last_pipeline(&self) -> &ExecutionPipeline {
        self.expect_last_exec()
            .pipeline
            .as_ref()
            .expect("missing command pipeline")
    }

    fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("command")
    }
}

/// A composable graph of Vulkan command buffer operations.
///
/// `Graph` instances are are intended for one-time use.
///
/// The design of this code originated with a combination of
/// [`PassBuilder`](https://github.com/EmbarkStudios/kajiya/blob/main/crates/lib/kajiya-rg/src/pass_builder.rs)
/// and
/// [`graph.cpp`](https://github.com/Themaister/Granite/blob/master/renderer/graph.cpp).
#[derive(Debug, Default)]
pub struct Graph {
    cmds: Vec<CommandData>,
    resources: Vec<AnyResource>,
}

impl Graph {
    /// Constructs a default `Graph`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocates and begins writing a new command.
    pub fn begin_cmd(&mut self) -> Command<'_> {
        Command::new(self)
    }

    /// Binds a Vulkan buffer, image, or acceleration structure resource to this graph.
    ///
    /// Bound resource nodes may be used in commands for shader pipeline operations and other
    /// general functions.
    pub fn bind_resource<R>(&mut self, resource: R) -> R::Node
    where
        R: Resource,
    {
        resource.bind_graph(self)
    }

    /// Copy an image, potentially performing format conversion.
    pub fn blit_image(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyImageNode>,
        filter: vk::Filter,
    ) -> &mut Self {
        let src = src.into();
        let src_info = self.resource(src).info;

        let dst = dst.into();
        let dst_info = self.resource(dst).info;

        self.blit_image_region(
            src,
            dst,
            filter,
            [vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(src_info.fmt),
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: src_info.width as _,
                        y: src_info.height as _,
                        z: src_info.depth as _,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(dst_info.fmt),
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: dst_info.width as _,
                        y: dst_info.height as _,
                        z: dst_info.depth as _,
                    },
                ],
            }],
        )
    }

    /// Copy regions of an image, potentially performing format conversion.
    #[profiling::function]
    pub fn blit_image_region(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyImageNode>,
        filter: vk::Filter,
        regions: impl AsRef<[vk::ImageBlit]> + 'static + Send,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();

        let mut cmd = self.begin_cmd().debug_name("blit image");

        for region in regions.as_ref() {
            let src_region = image_subresource_range_from_layers(region.src_subresource);
            cmd.set_subresource_access(src, src_region, AccessType::TransferRead);

            let dst_region = image_subresource_range_from_layers(region.dst_subresource);
            cmd.set_subresource_access(dst, dst_region, AccessType::TransferWrite);
        }

        cmd.record_cmd_buf(move |cmd_buf| {
            let src_image = cmd_buf.resource(src).handle;
            let dst_image = cmd_buf.resource(dst).handle;

            unsafe {
                cmd_buf.device.cmd_blit_image(
                    cmd_buf.handle,
                    src_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions.as_ref(),
                    filter,
                );
            }
        })
        .end_cmd()
    }

    /// Clear a color image.
    #[profiling::function]
    pub fn clear_color_image(
        &mut self,
        image: impl Into<AnyImageNode>,
        color: impl Into<ClearColorValue>,
    ) -> &mut Self {
        let color = color.into().into();
        let image = image.into();
        let image_view = self.resource(image).info.into();

        self.begin_cmd()
            .debug_name("clear color")
            .subresource_access(image, image_view, AccessType::TransferWrite)
            .record_cmd_buf(move |cmd_buf| {
                let image = cmd_buf.resource(image);

                unsafe {
                    cmd_buf.device.cmd_clear_color_image(
                        cmd_buf.handle,
                        image.handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &color,
                        &[image_view],
                    );
                }
            })
            .end_cmd()
    }

    /// Clears a depth/stencil image.
    #[profiling::function]
    pub fn clear_depth_stencil_image(
        &mut self,
        image: impl Into<AnyImageNode>,
        depth: f32,
        stencil: u32,
    ) -> &mut Self {
        let image = image.into();
        let image_view = self.resource(image).info.into();

        self.begin_cmd()
            .debug_name("clear depth/stencil")
            .subresource_access(image, image_view, AccessType::TransferWrite)
            .record_cmd_buf(move |cmd_buf| {
                let image = cmd_buf.resource(image);

                unsafe {
                    cmd_buf.device.cmd_clear_depth_stencil_image(
                        cmd_buf.handle,
                        image.handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &vk::ClearDepthStencilValue { depth, stencil },
                        &[image_view],
                    );
                }
            })
            .end_cmd()
    }

    /// Copy data between buffers
    pub fn copy_buffer(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyBufferNode>,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();
        let src_info = self.resource(src).info;
        let dst_info = self.resource(dst).info;

        self.copy_buffer_region(
            src,
            dst,
            [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: src_info.size.min(dst_info.size),
            }],
        )
    }

    /// Copy data between buffer regions.
    #[profiling::function]
    pub fn copy_buffer_region(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyBufferNode>,
        regions: impl AsRef<[vk::BufferCopy]> + 'static + Send,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();

        #[cfg(debug_assertions)]
        let src_size = self.resource(src).info.size;

        #[cfg(debug_assertions)]
        let dst_size = self.resource(dst).info.size;

        let mut cmd = self.begin_cmd().debug_name("copy buffer");

        for region in regions.as_ref() {
            #[cfg(debug_assertions)]
            {
                assert!(
                    region.src_offset + region.size <= src_size,
                    "source range end ({}) exceeds source size ({src_size})",
                    region.src_offset + region.size
                );
                assert!(
                    region.dst_offset + region.size <= dst_size,
                    "destination range end ({}) exceeds destination size ({dst_size})",
                    region.dst_offset + region.size
                );
            };

            cmd.set_subresource_access(
                src,
                region.src_offset..region.src_offset + region.size,
                AccessType::TransferRead,
            );
            cmd.set_subresource_access(
                dst,
                region.dst_offset..region.dst_offset + region.size,
                AccessType::TransferWrite,
            );
        }

        cmd.record_cmd_buf(move |cmd_buf| {
            let src = cmd_buf.resource(src);
            let dst = cmd_buf.resource(dst);

            unsafe {
                cmd_buf.device.cmd_copy_buffer(
                    cmd_buf.handle,
                    src.handle,
                    dst.handle,
                    regions.as_ref(),
                );
            }
        })
        .end_cmd()
    }

    /// Copy data from a buffer into an image.
    pub fn copy_buffer_to_image(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyImageNode>,
    ) -> &mut Self {
        let dst = dst.into();
        let dst_info = self.resource(dst).info;

        self.copy_buffer_to_image_region(
            src,
            dst,
            [vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: dst_info.width,
                buffer_image_height: dst_info.height,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(dst_info.fmt),
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: Default::default(),
                image_extent: vk::Extent3D {
                    depth: dst_info.depth,
                    height: dst_info.height,
                    width: dst_info.width,
                },
            }],
        )
    }

    /// Copy data from a buffer into an image.
    #[profiling::function]
    pub fn copy_buffer_to_image_region(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyImageNode>,
        regions: impl AsRef<[vk::BufferImageCopy]> + 'static + Send,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();
        let dst_info = self.resource(dst).info;

        let mut cmd = self.begin_cmd().debug_name("copy buffer to image");

        for region in regions.as_ref() {
            let block_bytes_size = format_texel_block_size(dst_info.fmt);
            let (block_height, block_width) = format_texel_block_extent(dst_info.fmt);
            let data_size = block_bytes_size
                * (region.buffer_row_length / block_width)
                * (region.buffer_image_height / block_height);

            cmd.set_subresource_access(
                src,
                region.buffer_offset..region.buffer_offset + data_size as vk::DeviceSize,
                AccessType::TransferRead,
            );
            cmd.set_subresource_access(
                dst,
                image_subresource_range_from_layers(region.image_subresource),
                AccessType::TransferWrite,
            );
        }

        cmd.record_cmd_buf(move |cmd_buf| {
            let src = cmd_buf.resource(src);
            let dst = cmd_buf.resource(dst);

            unsafe {
                cmd_buf.device.cmd_copy_buffer_to_image(
                    cmd_buf.handle,
                    src.handle,
                    dst.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions.as_ref(),
                );
            }
        })
        .end_cmd()
    }

    /// Copy all layers of a source image to a destination image.
    pub fn copy_image(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyImageNode>,
    ) -> &mut Self {
        let src = src.into();
        let src_info = self.resource(src).info;

        let dst = dst.into();
        let dst_info = self.resource(dst).info;

        self.copy_image_region(
            src,
            dst,
            [vk::ImageCopy {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(src_info.fmt),
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: src_info.array_layer_count,
                },
                src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(dst_info.fmt),
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: src_info.array_layer_count,
                },
                dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                extent: vk::Extent3D {
                    depth: src_info.depth.clamp(1, dst_info.depth),
                    height: src_info.height.clamp(1, dst_info.height),
                    width: src_info.width.min(dst_info.width),
                },
            }],
        )
    }

    /// Copy data between images.
    #[profiling::function]
    pub fn copy_image_region(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyImageNode>,
        regions: impl AsRef<[vk::ImageCopy]> + 'static + Send,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();

        let mut cmd = self.begin_cmd().debug_name("copy image");

        for region in regions.as_ref() {
            cmd.set_subresource_access(
                src,
                image_subresource_range_from_layers(region.src_subresource),
                AccessType::TransferRead,
            );
            cmd.set_subresource_access(
                dst,
                image_subresource_range_from_layers(region.dst_subresource),
                AccessType::TransferWrite,
            );
        }

        cmd.record_cmd_buf(move |cmd_buf| {
            let src = cmd_buf.resource(src);
            let dst = cmd_buf.resource(dst);

            unsafe {
                cmd_buf.device.cmd_copy_image(
                    cmd_buf.handle,
                    src.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions.as_ref(),
                );
            }
        })
        .end_cmd()
    }

    /// Copy image data into a buffer.
    pub fn copy_image_to_buffer(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyBufferNode>,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();

        let src_info = self.resource(src).info;

        self.copy_image_to_buffer_region(
            src,
            dst,
            [vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: src_info.width,
                buffer_image_height: src_info.height,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(src_info.fmt),
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: Default::default(),
                image_extent: vk::Extent3D {
                    depth: src_info.depth,
                    height: src_info.height,
                    width: src_info.width,
                },
            }],
        )
    }

    /// Copy image data into a buffer.
    #[profiling::function]
    pub fn copy_image_to_buffer_region(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyBufferNode>,
        regions: impl AsRef<[vk::BufferImageCopy]> + 'static + Send,
    ) -> &mut Self {
        let src = src.into();
        let src_info = self.resource(src).info;
        let dst = dst.into();

        let mut cmd = self.begin_cmd().debug_name("copy image to buffer");

        for region in regions.as_ref() {
            let block_bytes_size = format_texel_block_size(src_info.fmt);
            let (block_height, block_width) = format_texel_block_extent(src_info.fmt);
            let data_size = block_bytes_size
                * (region.buffer_row_length / block_width)
                * (region.buffer_image_height / block_height);

            cmd.set_subresource_access(
                src,
                image_subresource_range_from_layers(region.image_subresource),
                AccessType::TransferRead,
            );
            cmd.set_subresource_access(
                dst,
                region.buffer_offset..region.buffer_offset + data_size as vk::DeviceSize,
                AccessType::TransferWrite,
            );
        }

        cmd.record_cmd_buf(move |cmd_buf| {
            let src = cmd_buf.resource(src);
            let dst = cmd_buf.resource(dst);

            unsafe {
                cmd_buf.device.cmd_copy_image_to_buffer(
                    cmd_buf.handle,
                    src.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst.handle,
                    regions.as_ref(),
                );
            }
        })
        .end_cmd()
    }

    /// Fill a region of a buffer with a fixed value.
    pub fn fill_buffer(
        &mut self,
        buffer: impl Into<AnyBufferNode>,
        region: Range<vk::DeviceSize>,
        data: u32,
    ) -> &mut Self {
        let buffer = buffer.into();

        self.begin_cmd()
            .debug_name("fill buffer")
            .subresource_access(buffer, region.clone(), AccessType::TransferWrite)
            .record_cmd_buf(move |cmd_buf| {
                let buffer = cmd_buf.resource(buffer);

                unsafe {
                    cmd_buf.device.cmd_fill_buffer(
                        cmd_buf.handle,
                        buffer.handle,
                        region.start,
                        region.end - region.start,
                        data,
                    );
                }
            })
            .end_cmd()
    }

    /// Returns the index of the first pass which accesses a given node
    #[profiling::function]
    fn first_node_access_pass_index(&self, resource_node: impl Node) -> Option<usize> {
        let node_idx = resource_node.index();

        for (pass_idx, pass) in self.cmds.iter().enumerate() {
            for exec in pass.execs.iter() {
                if exec.accesses.contains_key(&node_idx) {
                    return Some(pass_idx);
                }
            }
        }

        None
    }

    /// Finalizes the graph and provides an object with functions for submitting the resulting
    /// commands.
    #[profiling::function]
    pub fn into_queue(mut self) -> Queue {
        // The final execution of each pass has no function
        for cmd in &mut self.cmds {
            debug_assert!(cmd.expect_last_exec().func.is_none());

            cmd.execs.pop();
        }

        Queue::new(self)
    }

    /// Returns a borrow of the original Vulkan resource (buffer, image or acceleration structure)
    /// which the given bound resource node represents.
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: Node,
    {
        resource_node.borrow(&self.resources)
    }

    /// Note: `data` must not exceed 65536 bytes.
    #[profiling::function]
    pub fn update_buffer(
        &mut self,
        buffer: impl Into<AnyBufferNode>,
        offset: vk::DeviceSize,
        data: impl AsRef<[u8]> + 'static + Send,
    ) -> &mut Self {
        let buffer = buffer.into();
        let data_end = offset + data.as_ref().len() as vk::DeviceSize;

        #[cfg(debug_assertions)]
        {
            let buffer_info = self.resource(buffer).info;

            assert!(
                data_end <= buffer_info.size,
                "data range end ({data_end}) exceeds buffer size ({})",
                buffer_info.size
            );
        }

        self.begin_cmd()
            .debug_name("update buffer")
            .subresource_access(buffer, offset..data_end, AccessType::TransferWrite)
            .record_cmd_buf(move |cmd_buf| {
                let buffer = cmd_buf.resource(buffer);

                unsafe {
                    cmd_buf.device.cmd_update_buffer(
                        cmd_buf.handle,
                        buffer.handle,
                        offset,
                        data.as_ref(),
                    );
                }
            })
            .end_cmd()
    }
}

/// A Vulkan resource which has been bound to a [`Graph`].
///
/// See [`Graph::bind_resource`].
pub trait Node {
    /// The Vulkan buffer, image, or acceleration struction type.
    type Resource;

    #[doc(hidden)]
    fn borrow(self, resources: &[AnyResource]) -> &Self::Resource;

    #[doc(hidden)]
    fn index(&self) -> NodeIndex;
}

/// A Vulkan resource which may be bound to a [`Graph`].
///
/// See [`Graph::bind_resource`] and
/// [`Command::bind_resource`](crate::cmd::Command::bind_resource).
pub trait Resource {
    /// The resource handle type.
    type Node;

    #[doc(hidden)]
    fn bind_graph(self, _: &mut Graph) -> Self::Node;

    #[deprecated = "use bind_graph function"]
    #[doc(hidden)]
    fn bind(self, graph: &mut Graph) -> Self::Node
    where
        Self: Sized,
    {
        self.bind_graph(graph)
    }
}

impl Resource for SwapchainImage {
    type Node = SwapchainImageNode;

    fn bind_graph(self, graph: &mut Graph) -> Self::Node {
        // We will return a new node
        let node = Self::Node::new(graph.resources.len());

        //trace!("Node {}: {:?}", res.idx, &self);

        let resource = AnyResource::SwapchainImage(Box::new(self));
        graph.resources.push(resource);

        node
    }
}

macro_rules! resource {
    ($name:ident) => {
        paste::paste! {
            impl Resource for $name {
                type Node = [<$name Node>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource a new item (Image or Buffer or etc)
                    // We will return a new node
                    let node = Self::Node::new(graph.resources.len());

                    let resource = AnyResource::$name(Arc::new(self));
                    graph.resources.push(resource);

                    node
                }
            }

            impl Resource for Arc<$name> {
                type Node = [<$name Node>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource an existing resource (Arc<Image> or
                    // Arc<Buffer> or etc)
                    // We will return an existing node, if possible
                    // TODO: Could store a sorted list of these shared pointers to avoid the O(N)
                    for (idx, existing_resource) in graph.resources.iter_mut().enumerate() {
                        if let AnyResource::$name(existing_resource) = existing_resource
                            && Arc::ptr_eq(existing_resource, &self) {
                                return Self::Node::new(idx);
                        }
                    }

                    // Return a new node
                    let node = Self::Node::new(graph.resources.len());
                    let resource = AnyResource::$name(self);
                    graph.resources.push(resource);

                    node
                }
            }

            impl<'a> Resource for &'a Arc<$name> {
                type Node = [<$name Node>];

                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource a borrowed resource (&Arc<Image> or
                    // &Arc<Buffer> or etc)

                    Arc::clone(self).bind_graph(graph)
                }
            }

            impl Resource for Lease<$name> {
                type Node = [<$name LeaseNode>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource a new lease (Lease<Image> or Lease<Buffer> or
                    // etc)

                    // We will return a new node
                    let node = Self::Node::new(graph.resources.len());
                    let resource = AnyResource::[<$name Lease>](Arc::new(self));
                    graph.resources.push(resource);

                    node
                }
            }

            impl Resource  for Arc<Lease<$name>> {
                type Node = [<$name LeaseNode>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource an existing lease resource
                    // (Arc<Lease<Image>> or Arc<Lease<Buffer>> or etc)

                    // We will return an existing node, if possible
                    // TODO: Could store a sorted list of these shared pointers to avoid the O(N)
                    for (idx, existing_resource) in graph.resources.iter().enumerate() {
                        if let AnyResource::[<$name Lease>](existing_resource) = existing_resource
                            && Arc::ptr_eq(existing_resource, &self) {
                                return Self::Node::new(idx);
                        }
                    }

                    // We will return a new node
                    let node = Self::Node::new(graph.resources.len());
                    let resource = AnyResource::[<$name Lease>](self);
                    graph.resources.push(resource);

                    node
                }
            }

            impl<'a> Resource for &'a Arc<Lease<$name>> {
                type Node = [<$name LeaseNode>];

                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource a borrowed resource (&Arc<Lease<Image>> or
                    // &Arc<Lease<Buffer>> or etc)

                    Arc::clone(self).bind_graph(graph)
                }
            }
        }
    };
}

resource!(AccelerationStructure);
resource!(Image);
resource!(Buffer);

#[deprecated]
#[doc(hidden)]
pub mod graph {
    #[deprecated = "use vk_graph::node module"]
    pub mod node {
        #[deprecated = "use vk_graph::node::AccelerationStructureLeaseNode"]
        pub type AccelerationStructureLeaseNode = crate::node::AccelerationStructureLeaseNode;

        #[deprecated = "use vk_graph::node::AccelerationStructureNode"]
        pub type AccelerationStructureNode = crate::node::AccelerationStructureNode;

        #[deprecated = "use vk_graph::node::AnyAccelerationStructureNode"]
        pub type AnyAccelerationStructureNode = crate::node::AnyAccelerationStructureNode;

        #[deprecated = "use vk_graph::node::AnyBufferNode"]
        pub type AnyBufferNode = crate::node::AnyBufferNode;

        #[deprecated = "use vk_graph::node::AnyImageNode"]
        pub type AnyImageNode = crate::node::AnyImageNode;

        #[deprecated = "use vk_graph::node::BufferLeaseNode"]
        pub type BufferLeaseNode = crate::node::BufferLeaseNode;

        #[deprecated = "use vk_graph::node::BufferNode"]
        pub type BufferNode = crate::node::BufferNode;

        #[deprecated = "use vk_graph::node::ImageLeaseNode"]
        pub type ImageLeaseNode = crate::node::ImageLeaseNode;

        #[deprecated = "use vk_graph::node::ImageNode"]
        pub type ImageNode = crate::node::ImageNode;

        #[deprecated = "use vk_graph::node::Node"]
        pub type Node = dyn crate::Node<Resource = ()>;

        #[deprecated = "use vk_graph::node::SwapchainImageNode"]
        pub type SwapchainImageNode = crate::node::SwapchainImageNode;
    }

    #[deprecated]
    #[doc(hidden)]
    pub mod pass_ref {
        #[deprecated = "use vk_graph::cmd::CommandBufferRef"]
        pub type Acceleration<'a> = crate::cmd::CommandBuffer<'a>;

        #[deprecated = "use vk_graph::cmd::CommandBufferRef"]
        pub type AccelerationStructureBuildInfo = crate::cmd::BuildAccelerationStructureInfo;

        #[deprecated = "use vk_graph::cmd::CommandBufferRef"]
        pub type AccelerationStructureIndirectBuildInfo =
            crate::cmd::BuildAccelerationStructureIndirectInfo;

        #[deprecated = "use vk_graph::cmd::CommandBufferRef"]
        pub type AccelerationStructureIndirectUpdateInfo =
            crate::cmd::UpdateAccelerationStructureIndirectInfo;

        #[deprecated = "use vk_graph::cmd::CommandBufferRef"]
        pub type AccelerationStructureUpdateInfo = crate::cmd::UpdateAccelerationStructureInfo;

        #[deprecated = "use vk_graph::cmd::Descriptor"]
        pub type Descriptor = crate::cmd::Descriptor;

        #[deprecated = "use vk_graph::cmd::GraphicCommandBufferRef"]
        pub type Draw<'a> = crate::cmd::GraphicCommandBuffer<'a>;

        #[deprecated = "use vk_graph::cmd::CommandRef"]
        pub type PassRef<'a> = crate::cmd::Command<'a>;

        #[deprecated = "use vk_graph::cmd::PipelineCommand"]
        pub type PipelinePassRef<'a, T> = crate::cmd::PipelineCommand<'a, T>;

        #[deprecated = "use vk_graph::cmd::RayTraceCommandBufferRef"]
        pub type RayTrace<'a> = crate::cmd::RayTraceCommandBuffer<'a>;

        #[deprecated = "use vk_graph::ViewInfo"]
        pub type ViewType = crate::cmd::ViewInfo;

        #[deprecated = "remove"]
        pub trait View {
            type Information;
        }
    }

    #[deprecated = "use vk_graph::Graph"]
    pub type RenderGraph = crate::Graph;

    #[deprecated = "use vk_graph::Queue"]
    pub type Resolver = crate::Queue;
}

#[allow(deprecated)]
#[allow(unused)]
#[doc(hidden)]
pub(crate) mod deprecated {
    use {
        crate::{
            AnyResource, Graph, Node, Resource,
            driver::{
                DriverError,
                accel_struct::{AccelerationStructure, AccelerationStructureInfo},
                buffer::{Buffer, BufferInfo},
                cmd_buf::{CommandBuffer, CommandBufferInfo},
                descriptor_set::{DescriptorPool, DescriptorPoolInfo},
                device::Device,
                image::{Image, ImageInfo},
                render_pass::{RenderPass, RenderPassInfo},
                swapchain::{Swapchain, SwapchainImage, SwapchainInfo},
            },
            node::{
                AccelerationStructureLeaseNode, AccelerationStructureNode,
                AnyAccelerationStructureNode, AnyBufferNode, AnyImageNode, BufferLeaseNode,
                BufferNode, ImageLeaseNode, ImageNode, SwapchainImageNode,
            },
            pool::{Lease, Pool},
        },
        ash::vk,
        std::{error, fmt, ops::Range, sync::Arc},
    };

    /// Specifies a color attachment clear value which can be used to initliaze an image.
    #[derive(Clone, Copy, Debug)]
    pub struct ClearColorValue(pub [f32; 4]);

    impl From<[f32; 3]> for ClearColorValue {
        fn from(color: [f32; 3]) -> Self {
            [color[0], color[1], color[2], 1.0].into()
        }
    }

    impl From<[f32; 4]> for ClearColorValue {
        fn from(color: [f32; 4]) -> Self {
            Self(color)
        }
    }

    impl From<[u8; 3]> for ClearColorValue {
        fn from(color: [u8; 3]) -> Self {
            [color[0], color[1], color[2], u8::MAX].into()
        }
    }

    impl From<[u8; 4]> for ClearColorValue {
        fn from(color: [u8; 4]) -> Self {
            [
                color[0] as f32 / u8::MAX as f32,
                color[1] as f32 / u8::MAX as f32,
                color[2] as f32 / u8::MAX as f32,
                color[3] as f32 / u8::MAX as f32,
            ]
            .into()
        }
    }

    #[deprecated = "use Swapchain from vk_graph_window crate"]
    #[derive(Debug)]
    #[doc(hidden)]
    pub struct Display {
        swapchain_info: SwapchainInfo,
    }

    impl Display {
        pub fn new(
            device: &Arc<Device>,
            swapchain: Swapchain,
            info: impl Into<DisplayInfo>,
        ) -> Result<Self, DriverError> {
            let _ = device;
            let _ = swapchain;
            let _ = info.into();

            Err(DriverError::Unsupported)
        }

        pub fn acquire_next_image(&mut self) -> Result<Option<SwapchainImage>, DisplayError> {
            Err(DisplayError)
        }

        pub fn present_image(
            &mut self,
            pool: &mut impl ResolverPool,
            render_graph: crate::graph::RenderGraph,
            swapchain_image: SwapchainImageNode,
            queue_index: u32,
        ) -> Result<(), DisplayError> {
            let _ = pool;
            let _ = render_graph;
            let _ = swapchain_image;
            let _ = queue_index;

            Err(DisplayError)
        }

        pub fn set_swapchain_info(&mut self, info: impl Into<SwapchainInfo>) {
            self.swapchain_info = info.into();
        }

        pub fn swapchain_info(&self) -> SwapchainInfo {
            self.swapchain_info
        }
    }

    #[deprecated = "use vk_graph_window::SwapchainError"]
    #[derive(Clone, Copy, Debug, Default)]
    #[doc(hidden)]
    pub struct DisplayError;

    impl fmt::Display for DisplayError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("deprecated Display API is unsupported; use vk_graph_window swapchain APIs")
        }
    }

    impl error::Error for DisplayError {}

    #[deprecated = "use vk_graph_window::SwapchainInfo"]
    #[derive(Clone, Copy, Debug, Default)]
    #[doc(hidden)]
    pub struct DisplayInfo;

    #[deprecated = "use vk_graph_window::SwapchainInfoBuilder"]
    #[derive(Clone, Copy, Debug, Default)]
    #[doc(hidden)]
    pub struct DisplayInfoBuilder;

    // General stuff
    impl Graph {
        #[deprecated = "use begin_cmd function"]
        #[doc(hidden)]
        pub fn begin_pass(&mut self, name: impl AsRef<str>) -> crate::graph::pass_ref::PassRef<'_> {
            self.begin_cmd().debug_name(name.as_ref().to_owned())
        }

        #[deprecated = "use bind_resource function"]
        #[doc(hidden)]
        pub fn bind_node<R>(&mut self, resource: R) -> R::Node
        where
            R: Resource,
        {
            self.bind_resource(resource)
        }

        #[deprecated = "use blit_image_region function"]
        #[doc(hidden)]
        pub fn blit_image_regions(
            &mut self,
            src_node: impl Into<AnyImageNode>,
            dst_node: impl Into<AnyImageNode>,
            filter: vk::Filter,
            regions: impl AsRef<[vk::ImageBlit]> + 'static + Send,
        ) -> &mut Self {
            self.blit_image_region(src_node, dst_node, filter, regions)
        }

        #[deprecated = "use clear_color_image function"]
        #[doc(hidden)]
        pub fn clear_color_image_value(
            &mut self,
            image_node: impl Into<AnyImageNode>,
            color_value: impl Into<ClearColorValue>,
        ) -> &mut Self {
            self.clear_color_image(image_node, color_value.into().0)
        }

        #[deprecated = "use clear_depth_stencil_image function"]
        #[doc(hidden)]
        pub fn clear_depth_stencil_image_value(
            &mut self,
            image_node: impl Into<AnyImageNode>,
            depth: f32,
            stencil: u32,
        ) -> &mut Self {
            self.clear_depth_stencil_image(image_node, depth, stencil)
        }

        #[deprecated = "use copy_buffer_region function"]
        #[doc(hidden)]
        pub fn copy_buffer_regions(
            &mut self,
            src_node: impl Into<AnyBufferNode>,
            dst_node: impl Into<AnyBufferNode>,
            regions: impl AsRef<[vk::BufferCopy]> + 'static + Send,
        ) -> &mut Self {
            self.copy_buffer_region(src_node, dst_node, regions)
        }

        #[deprecated = "use copy_buffer_to_image_region function"]
        #[doc(hidden)]
        pub fn copy_buffer_to_image_regions(
            &mut self,
            src_node: impl Into<AnyBufferNode>,
            dst_node: impl Into<AnyImageNode>,
            regions: impl AsRef<[vk::BufferImageCopy]> + 'static + Send,
        ) -> &mut Self {
            self.copy_buffer_to_image_region(src_node, dst_node, regions)
        }

        #[deprecated = "use copy_image_region function"]
        #[doc(hidden)]
        pub fn copy_image_regions(
            &mut self,
            src_node: impl Into<AnyImageNode>,
            dst_node: impl Into<AnyImageNode>,
            regions: impl AsRef<[vk::ImageCopy]> + 'static + Send,
        ) -> &mut Self {
            self.copy_image_region(src_node, dst_node, regions)
        }

        #[deprecated = "use copy_image_to_buffer_region function"]
        #[doc(hidden)]
        pub fn copy_image_to_buffer_regions(
            &mut self,
            src_node: impl Into<AnyImageNode>,
            dst_node: impl Into<AnyBufferNode>,
            regions: impl AsRef<[vk::BufferImageCopy]> + 'static + Send,
        ) -> &mut Self {
            self.copy_image_to_buffer_region(src_node, dst_node, regions)
        }

        #[deprecated = "use fill_buffer function"]
        #[doc(hidden)]
        pub fn fill_buffer_region(
            &mut self,
            buffer_node: impl Into<AnyBufferNode>,
            data: u32,
            region: Range<vk::DeviceSize>,
        ) -> &mut Self {
            self.fill_buffer(buffer_node, region, data)
        }

        #[deprecated = "use device_address function of resource function result"]
        #[doc(hidden)]
        pub fn node_device_address(&self, node: impl Node) -> vk::DeviceAddress {
            let idx = node.index();

            self.resources[idx]
                .as_buffer()
                .expect("missing buffer resource")
                .device_address()
        }

        #[deprecated = "dereference info field of resource function result"]
        #[doc(hidden)]
        pub fn node_info<N>(&self, node: N) -> N::Type
        where
            N: Node + Info,
        {
            node.info(&self.resources)
        }

        #[deprecated = "use into_queue function"]
        #[doc(hidden)]
        pub fn resolve(self) -> crate::graph::Resolver {
            self.into_queue()
        }

        #[deprecated = "use resource and clone functions"]
        #[doc(hidden)]
        pub fn unbind_node<N>(&mut self, node: N) -> N::Result
        where
            N: Unbind,
        {
            node.unbind(&self.resources)
        }

        #[deprecated = "use update_buffer function"]
        #[doc(hidden)]
        pub fn update_buffer_offset(
            &mut self,
            buffer_node: impl Into<AnyBufferNode>,
            offset: vk::DeviceSize,
            data: impl AsRef<[u8]> + 'static + Send,
        ) -> &mut Self {
            self.update_buffer(buffer_node, offset, data)
        }
    }

    pub trait Info {
        type Type;

        fn info(&self, _: &[AnyResource]) -> Self::Type
        where
            Self: Node;
    }

    impl Info for SwapchainImageNode {
        type Type = ImageInfo;

        fn info(&self, resources: &[AnyResource]) -> Self::Type
        where
            Self: Node,
        {
            resources[self.index()]
                .as_swapchain_image()
                .expect("missing swapchain image")
                .info
        }
    }

    macro_rules! info {
        ($name:ident) => {
            paste::paste! {
                impl Info for [<$name Node>] {
                    type Type = [<$name Info>];

                    fn info(&self, resources: &[AnyResource]) -> Self::Type
                    where
                        Self: Node,
                    {
                        let AnyResource::$name(resource) = &resources[self.index()] else {
                            panic!("invalid node");
                        };

                        resource.info
                    }
                }

                impl Info for [<Any $name Node>] {
                    type Type = [<$name Info>];

                    fn info(&self, resources: &[AnyResource]) -> Self::Type
                    where
                        Self: Node,
                    {
                        let AnyResource::$name(resource) = &resources[self.index()] else {
                            panic!("invalid node");
                        };

                        resource.info
                    }
                }

                impl Info for [<$name LeaseNode>] {
                    type Type = [<$name Info>];

                    fn info(&self, resources: &[AnyResource]) -> Self::Type
                    where
                        Self: Node,
                    {
                        let AnyResource::[<$name Lease>](resource) = &resources[self.index()] else {
                            panic!("invalid node");
                        };

                        resource.info
                    }
                }

                impl Unbind for [<$name Node>] {
                    type Result = Arc<$name>;

                    fn unbind(&self, resources: &[AnyResource]) -> Self::Result {
                        let AnyResource::$name(resource) = &resources[self.index()] else {
                            panic!("invalid node");
                        };

                        resource.clone()
                    }
                }

                impl Unbind for [<$name LeaseNode>] {
                    type Result = Arc<Lease<$name>>;

                    fn unbind(&self, resources: &[AnyResource]) -> Self::Result {
                        let AnyResource::[<$name Lease>](resource) = &resources[self.index()] else {
                            panic!("invalid node");
                        };

                        resource.clone()
                    }
                }
            }
        };
    }

    info!(AccelerationStructure);
    info!(Buffer);
    info!(Image);

    #[deprecated = "remove"]
    pub trait ResolverPool:
        Pool<DescriptorPoolInfo, DescriptorPool>
        + Pool<RenderPassInfo, RenderPass>
        + Pool<CommandBufferInfo, CommandBuffer>
        + Send
    {
    }

    pub trait Unbind: Node {
        type Result;

        fn unbind(&self, _: &[AnyResource]) -> Self::Result;
    }
}
