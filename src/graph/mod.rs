//! Rendering operations and command submission.
//!
//!

pub mod node;
pub mod pass_ref;

mod binding;
mod edge;
mod info;
mod resolver;
mod swapchain;

pub use self::{
    binding::{Bind, Unbind},
    resolver::Resolver,
};

use {
    self::{
        binding::Binding,
        edge::Edge,
        info::Information,
        node::Node,
        node::{
            AccelerationStructureLeaseNode, AccelerationStructureNode,
            AnyAccelerationStructureNode, AnyBufferNode, AnyImageNode, BufferLeaseNode, BufferNode,
            ImageLeaseNode, ImageNode, SwapchainImageNode,
        },
        pass_ref::{AttachmentIndex, Bindings, Descriptor, PassRef, SubresourceAccess, ViewType},
    },
    crate::driver::{
        DescriptorBindingMap,
        buffer::Buffer,
        compute::ComputePipeline,
        device::Device,
        format_aspect_mask, format_texel_block_extent, format_texel_block_size,
        graphic::{DepthStencilMode, GraphicPipeline},
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
        sync::Arc,
    },
    vk_sync::AccessType,
};

type ExecFn = Box<dyn FnOnce(&Device, vk::CommandBuffer, Bindings<'_>) + Send>;
type NodeIndex = usize;

#[derive(Clone, Copy, Debug)]
struct Area {
    height: u32,
    width: u32,
    x: i32,
    y: i32,
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
        if lhs.is_none() || rhs.is_none() {
            return true;
        }

        Self::are_identical(lhs.unwrap(), rhs.unwrap())
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
            .to_builder()
            .array_layer_count(self.array_layer_count)
            .mip_level_count(self.mip_level_count)
            .fmt(self.format)
            .build()
            .default_view_info()
            .to_builder()
            .aspect_mask(self.aspect_mask)
            .base_array_layer(self.base_array_layer)
            .base_mip_level(self.base_mip_level)
            .build()
    }
}

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

#[derive(Default)]
struct Execution {
    accesses: HashMap<NodeIndex, Vec<SubresourceAccess>>,
    bindings: BTreeMap<Descriptor, (NodeIndex, Option<ViewType>)>,

    correlated_view_mask: u32,
    depth_stencil: Option<DepthStencilMode>,
    render_area: Option<Area>,
    view_mask: u32,

    color_attachments: HashMap<AttachmentIndex, Attachment>,
    color_clears: HashMap<AttachmentIndex, (Attachment, ClearColorValue)>,
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

#[derive(Debug)]
enum ExecutionPipeline {
    Compute(Arc<ComputePipeline>),
    Graphic(Arc<GraphicPipeline>),
    RayTrace(Arc<RayTracePipeline>),
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
            ExecutionPipeline::Compute(pipeline) => &pipeline.descriptor_bindings,
            ExecutionPipeline::Graphic(pipeline) => &pipeline.descriptor_bindings,
            ExecutionPipeline::RayTrace(pipeline) => &pipeline.descriptor_bindings,
        }
    }

    fn descriptor_info(&self) -> &PipelineDescriptorInfo {
        match self {
            ExecutionPipeline::Compute(pipeline) => &pipeline.descriptor_info,
            ExecutionPipeline::Graphic(pipeline) => &pipeline.descriptor_info,
            ExecutionPipeline::RayTrace(pipeline) => &pipeline.descriptor_info,
        }
    }

    fn layout(&self) -> vk::PipelineLayout {
        match self {
            ExecutionPipeline::Compute(pipeline) => pipeline.layout,
            ExecutionPipeline::Graphic(pipeline) => pipeline.layout,
            ExecutionPipeline::RayTrace(pipeline) => pipeline.layout,
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

impl Clone for ExecutionPipeline {
    fn clone(&self) -> Self {
        match self {
            Self::Compute(pipeline) => Self::Compute(Arc::clone(pipeline)),
            Self::Graphic(pipeline) => Self::Graphic(Arc::clone(pipeline)),
            Self::RayTrace(pipeline) => Self::RayTrace(Arc::clone(pipeline)),
        }
    }
}

#[derive(Debug)]
struct Pass {
    execs: Vec<Execution>,
    name: String,
}

impl Pass {
    fn descriptor_pools_sizes(
        &self,
    ) -> impl Iterator<Item = &HashMap<u32, HashMap<vk::DescriptorType, u32>>> {
        self.execs
            .iter()
            .flat_map(|exec| exec.pipeline.as_ref())
            .map(|pipeline| &pipeline.descriptor_info().pool_sizes)
    }
}

/// A composable graph of render pass operations.
///
/// `RenderGraph` instances are are intended for one-time use.
///
/// The design of this code originated with a combination of
/// [`PassBuilder`](https://github.com/EmbarkStudios/kajiya/blob/main/crates/lib/kajiya-rg/src/pass_builder.rs)
/// and
/// [`render_graph.cpp`](https://github.com/Themaister/Granite/blob/master/renderer/render_graph.cpp).
#[derive(Debug)]
pub struct RenderGraph {
    bindings: Vec<Binding>,
    passes: Vec<Pass>,

    /// Set to true (when in debug mode) in order to get a breakpoint hit where you want.
    #[cfg(debug_assertions)]
    pub debug: bool,
}

impl RenderGraph {
    /// Constructs a new `RenderGraph`.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let bindings = vec![];
        let passes = vec![];

        #[cfg(debug_assertions)]
        let debug = false;

        Self {
            bindings,
            passes,
            #[cfg(debug_assertions)]
            debug,
        }
    }

    /// Begins a new pass.
    pub fn begin_pass(&mut self, name: impl AsRef<str>) -> PassRef<'_> {
        PassRef::new(self, name.as_ref().to_string())
    }

    /// Binds a Vulkan acceleration structure, buffer, or image to this graph.
    ///
    /// Bound nodes may be used in passes for pipeline and shader operations.
    pub fn bind_node<'a, B>(&'a mut self, binding: B) -> <B as Edge<Self>>::Result
    where
        B: Edge<Self>,
        B: Bind<&'a mut Self, <B as Edge<Self>>::Result>,
    {
        binding.bind(self)
    }

    /// Copy an image, potentially performing format conversion.
    pub fn blit_image(
        &mut self,
        src_node: impl Into<AnyImageNode>,
        dst_node: impl Into<AnyImageNode>,
        filter: vk::Filter,
    ) -> &mut Self {
        let src_node = src_node.into();
        let dst_node = dst_node.into();

        let src_info = self.node_info(src_node);
        let dst_info = self.node_info(dst_node);

        self.blit_image_region(
            src_node,
            dst_node,
            filter,
            vk::ImageBlit {
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
            },
        )
    }

    /// Copy a region of an image, potentially performing format conversion.
    pub fn blit_image_region(
        &mut self,
        src_node: impl Into<AnyImageNode>,
        dst_node: impl Into<AnyImageNode>,
        filter: vk::Filter,
        region: vk::ImageBlit,
    ) -> &mut Self {
        self.blit_image_regions(src_node, dst_node, filter, [region])
    }

    /// Copy regions of an image, potentially performing format conversion.
    #[profiling::function]
    pub fn blit_image_regions(
        &mut self,
        src_node: impl Into<AnyImageNode>,
        dst_node: impl Into<AnyImageNode>,
        filter: vk::Filter,
        regions: impl AsRef<[vk::ImageBlit]> + 'static + Send,
    ) -> &mut Self {
        let src_node = src_node.into();
        let dst_node = dst_node.into();

        let mut pass = self.begin_pass("blit image");

        for region in regions.as_ref() {
            pass = pass
                .access_node_subrange(
                    src_node,
                    AccessType::TransferRead,
                    image_subresource_range_from_layers(region.src_subresource),
                )
                .access_node_subrange(
                    dst_node,
                    AccessType::TransferWrite,
                    image_subresource_range_from_layers(region.dst_subresource),
                );
        }

        pass.record_cmd_buf(move |device, cmd_buf, bindings| {
            let src_image = bindings[src_node].handle;
            let dst_image = bindings[dst_node].handle;

            unsafe {
                device.cmd_blit_image(
                    cmd_buf,
                    src_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions.as_ref(),
                    filter,
                );
            }
        })
        .submit_pass()
    }

    /// Clear a color image.
    pub fn clear_color_image(&mut self, image_node: impl Into<AnyImageNode>) -> &mut Self {
        self.clear_color_image_value(image_node, [0, 0, 0, 0])
    }

    /// Clear a color image.
    #[profiling::function]
    pub fn clear_color_image_value(
        &mut self,
        image_node: impl Into<AnyImageNode>,
        color_value: impl Into<ClearColorValue>,
    ) -> &mut Self {
        let color_value = color_value.into();
        let image_node = image_node.into();
        let image_info = self.node_info(image_node);
        let image_view_info = image_info.default_view_info();

        self.begin_pass("clear color")
            .access_node_subrange(image_node, AccessType::TransferWrite, image_view_info)
            .record_cmd_buf(move |device, cmd_buf, bindings| unsafe {
                device.cmd_clear_color_image(
                    cmd_buf,
                    bindings[image_node].handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue {
                        float32: color_value.0,
                    },
                    &[image_view_info.into()],
                );
            })
            .submit_pass()
    }

    /// Clears a depth/stencil image.
    pub fn clear_depth_stencil_image(&mut self, image_node: impl Into<AnyImageNode>) -> &mut Self {
        self.clear_depth_stencil_image_value(image_node, 1.0, 0)
    }

    /// Clears a depth/stencil image.
    #[profiling::function]
    pub fn clear_depth_stencil_image_value(
        &mut self,
        image_node: impl Into<AnyImageNode>,
        depth: f32,
        stencil: u32,
    ) -> &mut Self {
        let image_node = image_node.into();
        let image_info = self.node_info(image_node);
        let image_view_info = image_info.default_view_info();

        self.begin_pass("clear depth/stencil")
            .access_node_subrange(image_node, AccessType::TransferWrite, image_view_info)
            .record_cmd_buf(move |device, cmd_buf, bindings| unsafe {
                device.cmd_clear_depth_stencil_image(
                    cmd_buf,
                    bindings[image_node].handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearDepthStencilValue { depth, stencil },
                    &[image_view_info.into()],
                );
            })
            .submit_pass()
    }

    /// Copy data between buffers
    pub fn copy_buffer(
        &mut self,
        src_node: impl Into<AnyBufferNode>,
        dst_node: impl Into<AnyBufferNode>,
    ) -> &mut Self {
        let src_node = src_node.into();
        let dst_node = dst_node.into();
        let src_info = self.node_info(src_node);
        let dst_info = self.node_info(dst_node);

        self.copy_buffer_region(
            src_node,
            dst_node,
            vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: src_info.size.min(dst_info.size),
            },
        )
    }

    /// Copy data between buffer regions.
    pub fn copy_buffer_region(
        &mut self,
        src_node: impl Into<AnyBufferNode>,
        dst_node: impl Into<AnyBufferNode>,
        region: vk::BufferCopy,
    ) -> &mut Self {
        self.copy_buffer_regions(src_node, dst_node, [region])
    }

    /// Copy data between buffer regions.
    #[profiling::function]
    pub fn copy_buffer_regions(
        &mut self,
        src_node: impl Into<AnyBufferNode>,
        dst_node: impl Into<AnyBufferNode>,
        regions: impl AsRef<[vk::BufferCopy]> + 'static + Send,
    ) -> &mut Self {
        let src_node = src_node.into();
        let dst_node = dst_node.into();

        #[cfg(debug_assertions)]
        let (src_size, dst_size) = (self.node_info(src_node).size, self.node_info(dst_node).size);

        let mut pass = self.begin_pass("copy buffer");

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

            pass = pass
                .access_node_subrange(
                    src_node,
                    AccessType::TransferRead,
                    region.src_offset..region.src_offset + region.size,
                )
                .access_node_subrange(
                    dst_node,
                    AccessType::TransferWrite,
                    region.dst_offset..region.dst_offset + region.size,
                );
        }

        pass.record_cmd_buf(move |device, cmd_buf, bindings| {
            let src_buf = bindings[src_node].handle;
            let dst_buf = bindings[dst_node].handle;

            unsafe {
                device.cmd_copy_buffer(cmd_buf, src_buf, dst_buf, regions.as_ref());
            }
        })
        .submit_pass()
    }

    /// Copy data from a buffer into an image.
    pub fn copy_buffer_to_image(
        &mut self,
        src_node: impl Into<AnyBufferNode>,
        dst_node: impl Into<AnyImageNode>,
    ) -> &mut Self {
        let dst_node = dst_node.into();
        let dst_info = self.node_info(dst_node);

        self.copy_buffer_to_image_region(
            src_node,
            dst_node,
            vk::BufferImageCopy {
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
            },
        )
    }

    /// Copy data from a buffer into an image.
    pub fn copy_buffer_to_image_region(
        &mut self,
        src_node: impl Into<AnyBufferNode>,
        dst_node: impl Into<AnyImageNode>,
        region: vk::BufferImageCopy,
    ) -> &mut Self {
        self.copy_buffer_to_image_regions(src_node, dst_node, [region])
    }

    /// Copy data from a buffer into an image.
    #[profiling::function]
    pub fn copy_buffer_to_image_regions(
        &mut self,
        src_node: impl Into<AnyBufferNode>,
        dst_node: impl Into<AnyImageNode>,
        regions: impl AsRef<[vk::BufferImageCopy]> + 'static + Send,
    ) -> &mut Self {
        let src_node = src_node.into();
        let dst_node = dst_node.into();
        let dst_info = self.node_info(dst_node);

        let mut pass = self.begin_pass("copy buffer to image");

        for region in regions.as_ref() {
            let block_bytes_size = format_texel_block_size(dst_info.fmt);
            let (block_height, block_width) = format_texel_block_extent(dst_info.fmt);
            let data_size = block_bytes_size
                * (region.buffer_row_length / block_width)
                * (region.buffer_image_height / block_height);

            pass = pass
                .access_node_subrange(
                    src_node,
                    AccessType::TransferRead,
                    region.buffer_offset..region.buffer_offset + data_size as vk::DeviceSize,
                )
                .access_node_subrange(
                    dst_node,
                    AccessType::TransferWrite,
                    image_subresource_range_from_layers(region.image_subresource),
                );
        }

        pass.record_cmd_buf(move |device, cmd_buf, bindings| {
            let src_buf = bindings[src_node].handle;
            let dst_image = bindings[dst_node].handle;

            unsafe {
                device.cmd_copy_buffer_to_image(
                    cmd_buf,
                    src_buf,
                    dst_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions.as_ref(),
                );
            }
        })
        .submit_pass()
    }

    /// Copy all layers of a source image to a destination image.
    pub fn copy_image(
        &mut self,
        src_node: impl Into<AnyImageNode>,
        dst_node: impl Into<AnyImageNode>,
    ) -> &mut Self {
        let src_node = src_node.into();
        let src_info = self.node_info(src_node);

        let dst_node = dst_node.into();
        let dst_info = self.node_info(dst_node);

        self.copy_image_region(
            src_node,
            dst_node,
            vk::ImageCopy {
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
            },
        )
    }

    /// Copy data between images.
    pub fn copy_image_region(
        &mut self,
        src_node: impl Into<AnyImageNode>,
        dst_node: impl Into<AnyImageNode>,
        region: vk::ImageCopy,
    ) -> &mut Self {
        self.copy_image_regions(src_node, dst_node, [region])
    }

    /// Copy data between images.
    #[profiling::function]
    pub fn copy_image_regions(
        &mut self,
        src_node: impl Into<AnyImageNode>,
        dst_node: impl Into<AnyImageNode>,
        regions: impl AsRef<[vk::ImageCopy]> + 'static + Send,
    ) -> &mut Self {
        let src_node = src_node.into();
        let dst_node = dst_node.into();

        let mut pass = self.begin_pass("copy image");

        for region in regions.as_ref() {
            pass = pass
                .access_node_subrange(
                    src_node,
                    AccessType::TransferRead,
                    image_subresource_range_from_layers(region.src_subresource),
                )
                .access_node_subrange(
                    dst_node,
                    AccessType::TransferWrite,
                    image_subresource_range_from_layers(region.dst_subresource),
                );
        }

        pass.record_cmd_buf(move |device, cmd_buf, bindings| {
            let src_image = bindings[src_node].handle;
            let dst_image = bindings[dst_node].handle;

            unsafe {
                device.cmd_copy_image(
                    cmd_buf,
                    src_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions.as_ref(),
                );
            }
        })
        .submit_pass()
    }

    /// Copy image data into a buffer.
    pub fn copy_image_to_buffer(
        &mut self,
        src_node: impl Into<AnyImageNode>,
        dst_node: impl Into<AnyBufferNode>,
    ) -> &mut Self {
        let src_node = src_node.into();
        let dst_node = dst_node.into();

        let src_info = self.node_info(src_node);

        self.copy_image_to_buffer_region(
            src_node,
            dst_node,
            vk::BufferImageCopy {
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
            },
        )
    }

    /// Copy image data into a buffer.
    pub fn copy_image_to_buffer_region(
        &mut self,
        src_node: impl Into<AnyImageNode>,
        dst_node: impl Into<AnyBufferNode>,
        region: vk::BufferImageCopy,
    ) -> &mut Self {
        self.copy_image_to_buffer_regions(src_node, dst_node, [region])
    }

    /// Copy image data into a buffer.
    #[profiling::function]
    pub fn copy_image_to_buffer_regions(
        &mut self,
        src_node: impl Into<AnyImageNode>,
        dst_node: impl Into<AnyBufferNode>,
        regions: impl AsRef<[vk::BufferImageCopy]> + 'static + Send,
    ) -> &mut Self {
        let src_node = src_node.into();
        let src_info = self.node_info(src_node);
        let dst_node = dst_node.into();

        let mut pass = self.begin_pass("copy image to buffer");

        for region in regions.as_ref() {
            let block_bytes_size = format_texel_block_size(src_info.fmt);
            let (block_height, block_width) = format_texel_block_extent(src_info.fmt);
            let data_size = block_bytes_size
                * (region.buffer_row_length / block_width)
                * (region.buffer_image_height / block_height);

            pass = pass
                .access_node_subrange(
                    src_node,
                    AccessType::TransferRead,
                    image_subresource_range_from_layers(region.image_subresource),
                )
                .access_node_subrange(
                    dst_node,
                    AccessType::TransferWrite,
                    region.buffer_offset..region.buffer_offset + data_size as vk::DeviceSize,
                );
        }

        pass.record_cmd_buf(move |device, cmd_buf, bindings| {
            let src_image = bindings[src_node].handle;
            let dst_buf = bindings[dst_node].handle;

            unsafe {
                device.cmd_copy_image_to_buffer(
                    cmd_buf,
                    src_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_buf,
                    regions.as_ref(),
                );
            }
        })
        .submit_pass()
    }

    /// Fill a region of a buffer with a fixed value.
    pub fn fill_buffer(&mut self, buffer_node: impl Into<AnyBufferNode>, data: u32) -> &mut Self {
        let buffer_node = buffer_node.into();

        let buffer_info = self.node_info(buffer_node);

        self.fill_buffer_region(buffer_node, data, 0..buffer_info.size)
    }

    /// Fill a region of a buffer with a fixed value.
    #[profiling::function]
    pub fn fill_buffer_region(
        &mut self,
        buffer_node: impl Into<AnyBufferNode>,
        data: u32,
        region: Range<vk::DeviceSize>,
    ) -> &mut Self {
        let buffer_node = buffer_node.into();

        self.begin_pass("fill buffer")
            .access_node_subrange(buffer_node, AccessType::TransferWrite, region.clone())
            .record_cmd_buf(move |device, cmd_buf, bindings| {
                let buffer = bindings[buffer_node].handle;

                unsafe {
                    device.cmd_fill_buffer(
                        cmd_buf,
                        buffer,
                        region.start,
                        region.end - region.start,
                        data,
                    );
                }
            })
            .submit_pass()
    }

    /// Returns the index of the first pass which accesses a given node
    #[profiling::function]
    fn first_node_access_pass_index(&self, node: impl Node) -> Option<usize> {
        let node_idx = node.index();

        for (pass_idx, pass) in self.passes.iter().enumerate() {
            for exec in pass.execs.iter() {
                if exec.accesses.contains_key(&node_idx) {
                    return Some(pass_idx);
                }
            }
        }

        None
    }

    /// Returns the device address of a buffer node.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not currently bound or was not created with the
    /// `SHADER_DEVICE_ADDRESS` usage flag.
    pub fn node_device_address(&self, node: impl Into<AnyBufferNode>) -> vk::DeviceAddress {
        let node: AnyBufferNode = node.into();
        let buffer = self.bindings[node.index()].as_driver_buffer().unwrap();

        Buffer::device_address(buffer)
    }

    /// Returns information used to crate a node.
    pub fn node_info<N>(&self, node: N) -> <N as Information>::Info
    where
        N: Information,
    {
        node.get(self)
    }

    /// Finalizes the graph and provides an object with functions for submitting the resulting
    /// commands.
    #[profiling::function]
    pub fn resolve(mut self) -> Resolver {
        // The final execution of each pass has no function
        for pass in &mut self.passes {
            pass.execs.pop();
        }

        Resolver::new(self)
    }

    /// Removes a node from this graph.
    ///
    /// Future access to `node` on this graph will return invalid results.
    pub fn unbind_node<N>(&mut self, node: N) -> <N as Edge<Self>>::Result
    where
        N: Edge<Self>,
        N: Unbind<Self, <N as Edge<Self>>::Result>,
    {
        node.unbind(self)
    }

    /// Note: `data` must not exceed 65536 bytes.
    pub fn update_buffer(
        &mut self,
        buffer_node: impl Into<AnyBufferNode>,
        data: impl AsRef<[u8]> + 'static + Send,
    ) -> &mut Self {
        self.update_buffer_offset(buffer_node, 0, data)
    }

    /// Note: `data` must not exceed 65536 bytes.
    #[profiling::function]
    pub fn update_buffer_offset(
        &mut self,
        buffer_node: impl Into<AnyBufferNode>,
        offset: vk::DeviceSize,
        data: impl AsRef<[u8]> + 'static + Send,
    ) -> &mut Self {
        let buffer_node = buffer_node.into();
        let data_end = offset + data.as_ref().len() as vk::DeviceSize;

        #[cfg(debug_assertions)]
        {
            let buffer_info = self.node_info(buffer_node);

            assert!(
                data_end <= buffer_info.size,
                "data range end ({data_end}) exceeds buffer size ({})",
                buffer_info.size
            );
        }

        self.begin_pass("update buffer")
            .access_node_subrange(buffer_node, AccessType::TransferWrite, offset..data_end)
            .record_cmd_buf(move |device, cmd_buf, bindings| {
                let buffer = bindings[buffer_node].handle;

                unsafe {
                    device.cmd_update_buffer(cmd_buf, buffer, offset, data.as_ref());
                }
            })
            .submit_pass()
    }
}
