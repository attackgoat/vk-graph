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

mod submission;

use std::sync::Arc;

use crate::{
    cmd::{ClearColorValue, CommandRef},
    driver::{
        accel_struct::AccelerationStructure, buffer::Buffer, image::Image,
        swapchain::SwapchainImage,
    },
    pool::Lease,
};

pub use self::submission::Submission;

use {
    self::{
        cmd::{AttachmentIndex, Binding, Command, SubresourceAccess, ViewInfo},
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
        graphic::{DepthStencilInfo, GraphicsPipeline},
        image::{ImageInfo, ImageViewInfo, SampleCount},
        image_subresource_range_from_layers,
        ray_trace::RayTracingPipeline,
        render_pass::ResolveMode,
        shader::PipelineDescriptorInfo,
    },
    ash::vk,
    std::{
        cmp::Ord,
        collections::{BTreeMap, HashMap},
        fmt::{Debug, Formatter},
        ops::Range,
        ops::{Deref, DerefMut},
    },
    vk_sync::AccessType,
};

type ExecFn = Box<dyn FnOnce(CommandRef) + Send>;
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

macro_rules! any_resource_from_arc {
    ($name:ident) => {
        paste::paste! {
            impl From<Arc<$name>> for AnyResource {
                fn from(resource: Arc<$name>) -> Self {
                    Self::$name(resource)
                }
            }

            impl From<Arc<Lease<$name>>> for AnyResource {
                fn from(resource: Arc<Lease<$name>>) -> Self {
                    Self::[<$name Lease>](resource)
                }
            }
        }
    };
}

any_resource_from_arc!(AccelerationStructure);
any_resource_from_arc!(Buffer);
any_resource_from_arc!(Image);

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
            Self::SwapchainImage(resource) => &resource.image,
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
    bindings: BTreeMap<Binding, (NodeIndex, ViewInfo)>,

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
    Graphic(GraphicsPipeline),
    RayTrace(RayTracingPipeline),
}

impl ExecutionPipeline {
    fn as_graphic(&self) -> Option<&GraphicsPipeline> {
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

    fn expect_graphic(&self) -> &GraphicsPipeline {
        self.as_graphic().expect("missing graphics pipeline")
    }

    fn expect_ray_trace(&self) -> &RayTracingPipeline {
        if let Self::RayTrace(pipeline) = self {
            pipeline
        } else {
            panic!("missing ray tracing pipeline")
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
    resources: ResourceMap,
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

        cmd.record_cmd(move |cmd| {
            let src_image = cmd.resource(src).handle;
            let dst_image = cmd.resource(dst).handle;

            unsafe {
                cmd.device.cmd_blit_image(
                    cmd.handle,
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
            .record_cmd(move |cmd| {
                let image = cmd.resource(image);

                unsafe {
                    cmd.device.cmd_clear_color_image(
                        cmd.handle,
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
            .record_cmd(move |cmd| {
                let image = cmd.resource(image);

                unsafe {
                    cmd.device.cmd_clear_depth_stencil_image(
                        cmd.handle,
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

        cmd.record_cmd(move |cmd| {
            let src = cmd.resource(src);
            let dst = cmd.resource(dst);

            unsafe {
                cmd.device
                    .cmd_copy_buffer(cmd.handle, src.handle, dst.handle, regions.as_ref());
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

        cmd.record_cmd(move |cmd| {
            let src = cmd.resource(src);
            let dst = cmd.resource(dst);

            unsafe {
                cmd.device.cmd_copy_buffer_to_image(
                    cmd.handle,
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

        cmd.record_cmd(move |cmd| {
            let src = cmd.resource(src);
            let dst = cmd.resource(dst);

            unsafe {
                cmd.device.cmd_copy_image(
                    cmd.handle,
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

        cmd.record_cmd(move |cmd| {
            let src = cmd.resource(src);
            let dst = cmd.resource(dst);

            unsafe {
                cmd.device.cmd_copy_image_to_buffer(
                    cmd.handle,
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
            .record_cmd(move |cmd| {
                let buffer = cmd.resource(buffer);

                unsafe {
                    cmd.device.cmd_fill_buffer(
                        cmd.handle,
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
    pub fn into_submission(mut self) -> Submission {
        // The final execution of each pass has no function
        for cmd in &mut self.cmds {
            debug_assert!(cmd.expect_last_exec().func.is_none());

            cmd.execs.pop();
        }

        Submission::new(self)
    }

    /// Returns a borrow of the original Vulkan resource (buffer, image or acceleration structure)
    /// which the given bound resource node represents.
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: Node,
    {
        resource_node.borrow(&self.resources)
    }

    /// Records a [`vkCmdUpdateBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdUpdateBuffer.html)
    /// command.
    ///
    /// Vulkan requires `data` to be at most `65536` bytes.
    ///
    /// In debug builds, this method asserts that `data.len()` does not exceed that Vulkan limit
    /// and that `offset + data.len()` does not exceed the bound buffer size. In release builds,
    /// those conditions are not checked here.
    #[profiling::function]
    pub fn update_buffer(
        &mut self,
        buffer: impl Into<AnyBufferNode>,
        offset: vk::DeviceSize,
        data: impl AsRef<[u8]> + 'static + Send,
    ) -> &mut Self {
        debug_assert!(data.as_ref().len() <= 64 * 1024);

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
            .record_cmd(move |cmd| {
                let buffer = cmd.resource(buffer);

                unsafe {
                    cmd.device
                        .cmd_update_buffer(cmd.handle, buffer.handle, offset, data.as_ref());
                }
            })
            .end_cmd()
    }
}

/// A Vulkan resource which has been bound to a [`Graph`].
///
/// See [`Graph::bind_resource`].
///
/// This trait is sealed and cannot be implemented outside of `vk-graph`.
pub trait Node: private::Sealed {
    /// The Vulkan buffer, image, or acceleration struction type.
    type Resource;

    #[doc(hidden)]
    fn borrow(self, resources: &[AnyResource]) -> &Self::Resource;

    #[doc(hidden)]
    fn index(&self) -> NodeIndex;
}

mod private {
    /// Prevents external implementations of [`Node`] and [`Resource`].
    pub trait Sealed {}
}

/// A Vulkan resource which may be bound to a [`Graph`].
///
/// See [`Graph::bind_resource`] and
/// [`Command::bind_resource`](crate::cmd::Command::bind_resource).
///
/// This trait is sealed and cannot be implemented outside of `vk-graph`.
pub trait Resource: private::Sealed {
    /// The resource handle type.
    type Node;

    #[doc(hidden)]
    fn bind_graph(self, _: &mut Graph) -> Self::Node;
}

impl private::Sealed for SwapchainImage {}

impl Resource for SwapchainImage {
    type Node = SwapchainImageNode;

    fn bind_graph(self, graph: &mut Graph) -> Self::Node {
        let node = Self::Node::new(graph.resources.len());

        //trace!("Node {}: {:?}", res.idx, &self);

        let resource = AnyResource::SwapchainImage(Box::new(self));
        graph.resources.bind(resource);

        node
    }
}

macro_rules! resource {
    ($name:ident) => {
        paste::paste! {
            impl private::Sealed for $name {}
            impl private::Sealed for Arc<$name> {}
            impl<'a> private::Sealed for &'a Arc<$name> {}
            impl private::Sealed for Lease<$name> {}
            impl private::Sealed for Arc<Lease<$name>> {}
            impl<'a> private::Sealed for &'a Arc<Lease<$name>> {}

            impl Resource for $name {
                type Node = [<$name Node>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource a new item (Image or Buffer or etc)

                    // We will return a new node
                    Arc::new(self).bind_graph(graph)
                }
            }

            impl Resource for Arc<$name> {
                type Node = [<$name Node>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource an existing resource (Arc<Image> or
                    // Arc<Buffer> or etc)

                    // We will return an existing node, if possible
                    Self::Node::new(graph.resources.bind_shared(self))
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
                    // In this function we are wrapping a newly pooled resource (Lease<Image> or
                    // Lease<Buffer> or etc)

                    // We will return a new node
                    Arc::new(self).bind_graph(graph)
                }
            }

            impl Resource  for Arc<Lease<$name>> {
                type Node = [<$name LeaseNode>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are wrapping an existing pooled resource
                    // (Arc<Lease<Image>> or Arc<Lease<Buffer>> or etc)

                    // We will return an existing node, if possible
                    Self::Node::new(graph.resources.bind_shared(self))
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

#[derive(Debug, Default)]
struct ResourceMap {
    addr_index: HashMap<usize, NodeIndex>,
    resources: Vec<AnyResource>,
}

impl ResourceMap {
    fn bind(&mut self, resource: AnyResource) -> NodeIndex {
        let node_idx = self.resources.len();
        self.resources.push(resource);

        node_idx
    }

    fn bind_shared<T>(&mut self, resource: Arc<T>) -> NodeIndex
    where
        Arc<T>: Into<AnyResource>,
    {
        let addr = Arc::as_ptr(&resource) as usize;

        *self.addr_index.entry(addr).or_insert_with(|| {
            let node_idx = self.resources.len();
            self.resources.push(resource.into());

            node_idx
        })
    }
}

impl Deref for ResourceMap {
    type Target = [AnyResource];

    fn deref(&self) -> &Self::Target {
        &self.resources
    }
}

impl DerefMut for ResourceMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.resources
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use ash::vk;

    use super::{AnyResource, Graph, Node, ResourceMap};
    use crate::driver::{
        DriverError,
        accel_struct::{AccelerationStructure, AccelerationStructureInfo},
        buffer::{Buffer, BufferInfo},
        device::{Device, DeviceInfo},
        image::{Image, ImageInfo},
        swapchain::SwapchainImage,
    };
    use crate::pool::{Pool, hash::HashPool};

    mod integration {
        use super::*;

        fn test_device() -> Result<Device, DriverError> {
            Device::create(DeviceInfo::default())
        }

        mod resource_map {
            use super::*;

            #[test]
            #[ignore = "requires Vulkan device"]
            fn bind_assigns_a_new_node_index_every_time() -> Result<(), DriverError> {
                let device = test_device()?;
                let buffer = Arc::new(Buffer::create(
                    &device,
                    BufferInfo::device_mem(4, vk::BufferUsageFlags::STORAGE_BUFFER),
                )?);
                let image = Arc::new(Image::create(
                    &device,
                    ImageInfo::image_2d(
                        1,
                        1,
                        vk::Format::R8G8B8A8_UNORM,
                        vk::ImageUsageFlags::SAMPLED,
                    ),
                )?);
                let mut resources = ResourceMap::default();

                assert_eq!(resources.bind(AnyResource::from(buffer)), 0);
                assert_eq!(resources.bind(AnyResource::from(image)), 1);
                assert_eq!(resources.len(), 2);

                Ok(())
            }

            #[test]
            #[ignore = "requires Vulkan device"]
            fn bind_shared_reuses_the_existing_node_index_for_the_same_address()
            -> Result<(), DriverError> {
                let device = test_device()?;
                let buffer = Arc::new(Buffer::create(
                    &device,
                    BufferInfo::device_mem(4, vk::BufferUsageFlags::STORAGE_BUFFER),
                )?);
                let mut resources = ResourceMap::default();

                assert_eq!(resources.bind_shared(Arc::clone(&buffer)), 0);
                assert_eq!(resources.bind_shared(buffer), 0);
                assert_eq!(resources.len(), 1);

                Ok(())
            }

            #[test]
            #[ignore = "requires Vulkan device"]
            fn bind_shared_creates_distinct_node_indices_for_different_addresses()
            -> Result<(), DriverError> {
                let device = test_device()?;
                let buffer = Arc::new(Buffer::create(
                    &device,
                    BufferInfo::device_mem(4, vk::BufferUsageFlags::STORAGE_BUFFER),
                )?);
                let image = Arc::new(Image::create(
                    &device,
                    ImageInfo::image_2d(
                        1,
                        1,
                        vk::Format::R8G8B8A8_UNORM,
                        vk::ImageUsageFlags::SAMPLED,
                    ),
                )?);
                let mut resources = ResourceMap::default();

                assert_eq!(resources.bind_shared(buffer), 0);
                assert_eq!(resources.bind_shared(image), 1);
                assert_eq!(resources.len(), 2);

                Ok(())
            }

            #[test]
            #[ignore = "requires Vulkan device"]
            fn graph_bind_fuzzes_all_resource_paths() -> Result<(), DriverError> {
                #[derive(Clone, Copy)]
                enum ResourceKind {
                    OwnedBuffer,
                    SharedBuffer,
                    OwnedBufferLease,
                    SharedBufferLease,
                    OwnedImage,
                    SharedImage,
                    OwnedImageLease,
                    SharedImageLease,
                    SwapchainImage,
                    OwnedAccelerationStructure,
                    SharedAccelerationStructure,
                    OwnedAccelerationStructureLease,
                    SharedAccelerationStructureLease,
                }

                struct SharedNodes<T> {
                    values: Vec<(Arc<T>, usize)>,
                }

                impl<T> Default for SharedNodes<T> {
                    fn default() -> Self {
                        Self { values: Vec::new() }
                    }
                }

                impl<T> SharedNodes<T> {
                    fn get(&self, idx: usize) -> Option<(Arc<T>, usize)> {
                        self.values
                            .get(idx)
                            .map(|(resource, node_idx)| (Arc::clone(resource), *node_idx))
                    }

                    fn push(&mut self, resource: Arc<T>, node_idx: usize) {
                        self.values.push((resource, node_idx));
                    }

                    fn len(&self) -> usize {
                        self.values.len()
                    }
                }

                fn next_rand(state: &mut u64) -> u64 {
                    *state ^= *state << 13;
                    *state ^= *state >> 7;
                    *state ^= *state << 17;
                    *state
                }

                let device = test_device()?;
                let mut pool = HashPool::new(&device);
                let mut graph = Graph::new();

                let mut rand_state = 0x5eed_u64;
                let mut shared_buffers = SharedNodes::<Buffer>::default();
                let mut shared_buffer_leases = SharedNodes::<crate::pool::Lease<Buffer>>::default();
                let mut shared_images = SharedNodes::<Image>::default();
                let mut shared_image_leases = SharedNodes::<crate::pool::Lease<Image>>::default();
                let mut shared_accels = SharedNodes::<AccelerationStructure>::default();
                let mut shared_accel_leases =
                    SharedNodes::<crate::pool::Lease<AccelerationStructure>>::default();
                let accel_supported = device.physical_device.accel_struct_properties.is_some();

                let mut resource_kinds = vec![
                    ResourceKind::OwnedBuffer,
                    ResourceKind::SharedBuffer,
                    ResourceKind::OwnedBufferLease,
                    ResourceKind::SharedBufferLease,
                    ResourceKind::OwnedImage,
                    ResourceKind::SharedImage,
                    ResourceKind::OwnedImageLease,
                    ResourceKind::SharedImageLease,
                    ResourceKind::SwapchainImage,
                ];

                if accel_supported {
                    resource_kinds.push(ResourceKind::OwnedAccelerationStructure);
                    resource_kinds.push(ResourceKind::SharedAccelerationStructure);
                    resource_kinds.push(ResourceKind::OwnedAccelerationStructureLease);
                    resource_kinds.push(ResourceKind::SharedAccelerationStructureLease);
                }

                for step in 0..64 {
                    let kind = resource_kinds
                        [(next_rand(&mut rand_state) as usize) % resource_kinds.len()];
                    let expect_new = match kind {
                        ResourceKind::OwnedBuffer
                        | ResourceKind::OwnedBufferLease
                        | ResourceKind::OwnedImage
                        | ResourceKind::OwnedImageLease
                        | ResourceKind::SwapchainImage
                        | ResourceKind::OwnedAccelerationStructure
                        | ResourceKind::OwnedAccelerationStructureLease => true,
                        ResourceKind::SharedBuffer => {
                            shared_buffers.len() == 0 || next_rand(&mut rand_state) & 1 == 0
                        }
                        ResourceKind::SharedBufferLease => {
                            shared_buffer_leases.len() == 0 || next_rand(&mut rand_state) & 1 == 0
                        }
                        ResourceKind::SharedImage => {
                            shared_images.len() == 0 || next_rand(&mut rand_state) & 1 == 0
                        }
                        ResourceKind::SharedImageLease => {
                            shared_image_leases.len() == 0 || next_rand(&mut rand_state) & 1 == 0
                        }
                        ResourceKind::SharedAccelerationStructure => {
                            shared_accels.len() == 0 || next_rand(&mut rand_state) & 1 == 0
                        }
                        ResourceKind::SharedAccelerationStructureLease => {
                            shared_accel_leases.len() == 0 || next_rand(&mut rand_state) & 1 == 0
                        }
                    };

                    let expected_node_idx = graph.resources.len();

                    let node_idx = match kind {
                        ResourceKind::OwnedBuffer => graph
                            .bind_resource(Buffer::create(
                                &device,
                                BufferInfo::device_mem(
                                    16 + step,
                                    vk::BufferUsageFlags::STORAGE_BUFFER,
                                ),
                            )?)
                            .index(),
                        ResourceKind::SharedBuffer if expect_new => {
                            let resource = Arc::new(Buffer::create(
                                &device,
                                BufferInfo::device_mem(
                                    16 + step,
                                    vk::BufferUsageFlags::STORAGE_BUFFER,
                                ),
                            )?);
                            let node_idx = graph.bind_resource(Arc::clone(&resource)).index();
                            shared_buffers.push(resource, node_idx);
                            node_idx
                        }
                        ResourceKind::SharedBuffer => {
                            let reuse_idx =
                                (next_rand(&mut rand_state) as usize) % shared_buffers.len();
                            let (resource, node_idx) = shared_buffers.get(reuse_idx).unwrap();
                            assert_eq!(graph.bind_resource(resource).index(), node_idx);
                            node_idx
                        }
                        ResourceKind::OwnedBufferLease => graph
                            .bind_resource(pool.resource(BufferInfo::device_mem(
                                32 + step,
                                vk::BufferUsageFlags::STORAGE_BUFFER,
                            ))?)
                            .index(),
                        ResourceKind::SharedBufferLease if expect_new => {
                            let resource = Arc::new(pool.resource(BufferInfo::device_mem(
                                32 + step,
                                vk::BufferUsageFlags::STORAGE_BUFFER,
                            ))?);
                            let node_idx = graph.bind_resource(Arc::clone(&resource)).index();
                            shared_buffer_leases.push(resource, node_idx);
                            node_idx
                        }
                        ResourceKind::SharedBufferLease => {
                            let reuse_idx =
                                (next_rand(&mut rand_state) as usize) % shared_buffer_leases.len();
                            let (resource, node_idx) = shared_buffer_leases.get(reuse_idx).unwrap();
                            assert_eq!(graph.bind_resource(resource).index(), node_idx);
                            node_idx
                        }
                        ResourceKind::OwnedImage => graph
                            .bind_resource(Image::create(
                                &device,
                                ImageInfo::image_2d(
                                    1,
                                    1,
                                    vk::Format::R8G8B8A8_UNORM,
                                    vk::ImageUsageFlags::SAMPLED,
                                ),
                            )?)
                            .index(),
                        ResourceKind::SharedImage if expect_new => {
                            let resource = Arc::new(Image::create(
                                &device,
                                ImageInfo::image_2d(
                                    1,
                                    1,
                                    vk::Format::R8G8B8A8_UNORM,
                                    vk::ImageUsageFlags::SAMPLED,
                                ),
                            )?);
                            let node_idx = graph.bind_resource(Arc::clone(&resource)).index();
                            shared_images.push(resource, node_idx);
                            node_idx
                        }
                        ResourceKind::SharedImage => {
                            let reuse_idx =
                                (next_rand(&mut rand_state) as usize) % shared_images.len();
                            let (resource, node_idx) = shared_images.get(reuse_idx).unwrap();
                            assert_eq!(graph.bind_resource(resource).index(), node_idx);
                            node_idx
                        }
                        ResourceKind::OwnedImageLease => graph
                            .bind_resource(pool.resource(ImageInfo::image_2d(
                                1,
                                1,
                                vk::Format::R8G8B8A8_UNORM,
                                vk::ImageUsageFlags::SAMPLED,
                            ))?)
                            .index(),
                        ResourceKind::SharedImageLease if expect_new => {
                            let resource = Arc::new(pool.resource(ImageInfo::image_2d(
                                1,
                                1,
                                vk::Format::R8G8B8A8_UNORM,
                                vk::ImageUsageFlags::SAMPLED,
                            ))?);
                            let node_idx = graph.bind_resource(Arc::clone(&resource)).index();
                            shared_image_leases.push(resource, node_idx);
                            node_idx
                        }
                        ResourceKind::SharedImageLease => {
                            let reuse_idx =
                                (next_rand(&mut rand_state) as usize) % shared_image_leases.len();
                            let (resource, node_idx) = shared_image_leases.get(reuse_idx).unwrap();
                            assert_eq!(graph.bind_resource(resource).index(), node_idx);
                            node_idx
                        }
                        ResourceKind::SwapchainImage => graph
                            .bind_resource(SwapchainImage::from_raw(
                                &device,
                                vk::Image::null(),
                                ImageInfo::image_2d(
                                    1,
                                    1,
                                    vk::Format::R8G8B8A8_UNORM,
                                    vk::ImageUsageFlags::COLOR_ATTACHMENT,
                                ),
                                step as u32,
                            ))
                            .index(),
                        ResourceKind::OwnedAccelerationStructure => graph
                            .bind_resource(AccelerationStructure::create(
                                &device,
                                AccelerationStructureInfo::blas(256 + step),
                            )?)
                            .index(),
                        ResourceKind::SharedAccelerationStructure if expect_new => {
                            let resource = Arc::new(AccelerationStructure::create(
                                &device,
                                AccelerationStructureInfo::blas(256 + step),
                            )?);
                            let node_idx = graph.bind_resource(Arc::clone(&resource)).index();
                            shared_accels.push(resource, node_idx);
                            node_idx
                        }
                        ResourceKind::SharedAccelerationStructure => {
                            let reuse_idx =
                                (next_rand(&mut rand_state) as usize) % shared_accels.len();
                            let (resource, node_idx) = shared_accels.get(reuse_idx).unwrap();
                            assert_eq!(graph.bind_resource(resource).index(), node_idx);
                            node_idx
                        }
                        ResourceKind::OwnedAccelerationStructureLease => graph
                            .bind_resource(
                                pool.resource(AccelerationStructureInfo::blas(512 + step))?,
                            )
                            .index(),
                        ResourceKind::SharedAccelerationStructureLease if expect_new => {
                            let resource = Arc::new(
                                pool.resource(AccelerationStructureInfo::blas(512 + step))?,
                            );
                            let node_idx = graph.bind_resource(Arc::clone(&resource)).index();
                            shared_accel_leases.push(resource, node_idx);
                            node_idx
                        }
                        ResourceKind::SharedAccelerationStructureLease => {
                            let reuse_idx =
                                (next_rand(&mut rand_state) as usize) % shared_accel_leases.len();
                            let (resource, node_idx) = shared_accel_leases.get(reuse_idx).unwrap();
                            assert_eq!(graph.bind_resource(resource).index(), node_idx);
                            node_idx
                        }
                    };

                    if expect_new {
                        assert_eq!(node_idx, expected_node_idx);
                        assert_eq!(graph.resources.len(), expected_node_idx + 1);
                    } else {
                        assert!(node_idx < expected_node_idx);
                        assert_eq!(graph.resources.len(), expected_node_idx);
                    }
                }

                Ok(())
            }
        }
    }
}
