/*!

A high-performance [Vulkan](https://www.vulkan.org/) driver with automated resource management and
execution. Start with [`Graph`] — bind resources, record commands, and submit for execution.

- **Beginner**: [`Graph`], [`cmd`], [`node`], [`pool`] — the high-level graph API
- **Intermediate**: [`driver`] — smart-pointer Vulkan wrappers (resources, pipelines, device)
- **Expert**: [`driver`] re-exports `ash` and `vk_sync` — raw Vulkan bindings

For installation, guides, and examples see the [Guide Book](https://attackgoat.github.io/vk-graph).

*/

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod cmd;
pub mod driver;
pub mod node;
pub mod pool;
pub mod stream;
pub mod submission;

mod lazy_str;

pub use self::lazy_str::LazyStr;

use {
    self::{
        cmd::{AttachmentIndex, Binding, Command, SubresourceAccess, ViewInfo},
        node::{
            AccelerationStructureLeaseNode, AccelerationStructureNode,
            AnyAccelerationStructureNode, AnyBufferNode, AnyImageNode, BufferLeaseNode, BufferNode,
            ImageLeaseNode, ImageNode, SwapchainImageNode,
        },
    },
    crate::{
        cmd::{ClearColorValue, CommandRef},
        driver::{
            DescriptorBindingMap,
            accel_struct::AccelerationStructureInfo,
            buffer::BufferInfo,
            compute::ComputePipeline,
            format_aspect_mask, format_texel_block_extent, format_texel_block_size,
            graphics::{DepthStencilInfo, GraphicsPipeline},
            image::{ImageInfo, ImageViewInfo, SampleCount},
            image_subresource_range_from_layers,
            ray_tracing::RayTracingPipeline,
            render_pass::ResolveMode,
            shader::PipelineDescriptorInfo,
        },
        driver::{
            accel_struct::AccelerationStructure, buffer::Buffer, image::Image,
            swapchain::SwapchainImage,
        },
        pool::Lease,
        submission::Submission,
    },
    ash::vk,
    smallvec::SmallVec,
    std::{
        cmp::Ord,
        collections::{BTreeMap, HashMap},
        fmt::{Debug, Formatter},
        mem,
        ops::Range,
        ops::{Deref, DerefMut},
        slice::Iter,
        sync::Arc,
    },
    vk_sync::AccessType,
};

#[cfg(feature = "checked")]
use std::sync::atomic::{AtomicU64, Ordering};

type CommandFn = Arc<dyn for<'a> Fn(CommandRef<'a>) + Send + Sync>;
type CommandFnOnce = Box<dyn FnOnce(CommandRef) + Send>;
type NodeIndex = usize;

#[cfg(feature = "checked")]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GraphId(u64);

#[cfg(feature = "checked")]
impl GraphId {
    fn next() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);

        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug)]
enum AnyResource {
    AccelerationStructure(Arc<AccelerationStructure>),
    AccelerationStructureArg(AccelerationStructureInfo),
    AccelerationStructureLease(Arc<Lease<AccelerationStructure>>),
    Buffer(Arc<Buffer>),
    BufferArg(BufferInfo),
    BufferLease(Arc<Lease<Buffer>>),
    Image(Arc<Image>),
    ImageArg(ImageInfo),
    ImageLease(Arc<Lease<Image>>),
    SwapchainImage(Box<SwapchainImage>),
}

impl Clone for AnyResource {
    fn clone(&self) -> Self {
        match self {
            Self::AccelerationStructure(resource) => {
                Self::AccelerationStructure(Arc::clone(resource))
            }
            Self::AccelerationStructureArg(info) => Self::AccelerationStructureArg(*info),
            Self::AccelerationStructureLease(resource) => {
                Self::AccelerationStructureLease(Arc::clone(resource))
            }
            Self::Buffer(resource) => Self::Buffer(Arc::clone(resource)),
            Self::BufferArg(info) => Self::BufferArg(*info),
            Self::BufferLease(resource) => Self::BufferLease(Arc::clone(resource)),
            Self::Image(resource) => Self::Image(Arc::clone(resource)),
            Self::ImageArg(info) => Self::ImageArg(*info),
            Self::ImageLease(resource) => Self::ImageLease(Arc::clone(resource)),
            Self::SwapchainImage(resource) => {
                Self::SwapchainImage(Box::new(unsafe { resource.to_detached() }))
            }
        }
    }
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
            Self::SwapchainImage(resource) => resource,
            _ => return None,
        })
    }

    fn expect_accel_struct(&self) -> &AccelerationStructure {
        self.as_accel_struct()
            .expect("missing acceleration structure resource")
    }

    pub(crate) fn expect_accel_struct_info(
        &self,
    ) -> crate::driver::accel_struct::AccelerationStructureInfo {
        match self {
            Self::AccelerationStructure(resource) => resource.info,
            Self::AccelerationStructureArg(info) => *info,
            Self::AccelerationStructureLease(resource) => resource.info,
            _ => panic!("missing acceleration structure resource"),
        }
    }

    fn expect_buffer(&self) -> &Buffer {
        self.as_buffer().expect("missing buffer resource")
    }

    pub(crate) fn expect_buffer_info(&self) -> crate::driver::buffer::BufferInfo {
        match self {
            Self::Buffer(resource) => resource.info,
            Self::BufferArg(info) => *info,
            Self::BufferLease(resource) => resource.info,
            _ => panic!("missing buffer resource"),
        }
    }

    fn expect_image(&self) -> &Image {
        self.as_image().expect("missing image resource")
    }

    pub(crate) fn expect_image_info(&self) -> ImageInfo {
        match self {
            Self::Image(resource) => resource.info,
            Self::ImageArg(info) => *info,
            Self::ImageLease(resource) => resource.info,
            Self::SwapchainImage(resource) => resource.info,
            _ => panic!("missing image resource"),
        }
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
            format: image_view_info.format,
            mip_level_count: image_view_info.mip_level_count,
            sample_count,
            target,
        }
    }

    fn are_compatible(lhs: Option<Self>, rhs: Option<Self>) -> bool {
        // Two attachment references are compatible if they have matching format and sample
        // count, or are both VK_ATTACHMENT_UNUSED or the pointer that would contain the
        // reference is NULL
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
            .format(self.format)
            .into_image_view()
            .aspect_mask(self.aspect_mask)
            .base_array_layer(self.base_array_layer)
            .base_mip_level(self.base_mip_level)
            .build()
    }

    fn remap_nodes(&mut self, node_map: &[NodeIndex]) {
        self.target = node_map[self.target];
    }
}

#[derive(Clone, Copy, Debug)]
struct ColorAttachment {
    attachment: Attachment,
    load: LoadOp<[f32; 4]>,
    store: StoreOp,
    resolve: Option<ColorResolve>,
    is_input: bool,
    is_attachment: bool,
}

#[derive(Clone, Debug, Default)]
struct ExecutionAttachmentMap {
    color: Vec<Option<ColorAttachment>>,
    depth_stencil: Option<DepthStencilAttachment>,
}

impl ExecutionAttachmentMap {
    fn color_attachment(&self, attachment_idx: AttachmentIndex) -> Option<&ColorAttachment> {
        self.color
            .get(attachment_idx as usize)
            .and_then(|slot| slot.as_ref())
    }

    fn color_attachment_mut(
        &mut self,
        attachment_idx: AttachmentIndex,
    ) -> Option<&mut ColorAttachment> {
        self.color
            .get_mut(attachment_idx as usize)
            .and_then(|slot| slot.as_mut())
    }

    fn color_attachments(&self) -> impl Iterator<Item = (AttachmentIndex, &ColorAttachment)> + '_ {
        self.color
            .iter()
            .enumerate()
            .filter_map(|(attachment_idx, slot)| {
                Some((attachment_idx as AttachmentIndex, slot.as_ref()?))
            })
    }

    fn depth_stencil_attachment(&self) -> Option<&DepthStencilAttachment> {
        self.depth_stencil.as_ref()
    }

    fn depth_stencil_attachment_mut(&mut self) -> Option<&mut DepthStencilAttachment> {
        self.depth_stencil.as_mut()
    }

    fn set_color_attachment(
        &mut self,
        attachment_idx: AttachmentIndex,
        attachment: ColorAttachment,
    ) {
        let attachment_idx = attachment_idx as usize;

        if self.color.len() <= attachment_idx {
            self.color.resize(attachment_idx + 1, None);
        }

        #[cfg(feature = "checked")]
        {
            let existing_attachment = self.color[attachment_idx]
                .as_ref()
                .map(|&color| color.attachment);

            assert!(
                Attachment::are_compatible(existing_attachment, Some(attachment.attachment)),
                "incompatible with existing attachment"
            );
        }

        self.color[attachment_idx] = Some(attachment);
    }

    fn set_depth_stencil_attachment(&mut self, attachment: DepthStencilAttachment) {
        #[cfg(feature = "checked")]
        {
            let existing_attachment = self
                .depth_stencil
                .as_ref()
                .map(|&depth_stencil| depth_stencil.attachment);

            assert!(
                Attachment::are_compatible(existing_attachment, Some(attachment.attachment)),
                "incompatible with existing attachment"
            );
        }

        self.depth_stencil = Some(attachment);
    }

    fn remap_nodes(&mut self, node_map: &[NodeIndex]) {
        for attachment in self.color.iter_mut().flatten() {
            attachment.attachment.remap_nodes(node_map);

            if let Some(resolve) = &mut attachment.resolve {
                resolve.attachment.remap_nodes(node_map);
            }
        }

        if let Some(attachment) = &mut self.depth_stencil {
            attachment.attachment.remap_nodes(node_map);

            if let Some(resolve) = &mut attachment.resolve {
                resolve.attachment.remap_nodes(node_map);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ColorResolve {
    attachment: Attachment,
    src_attachment_idx: AttachmentIndex,
}

#[derive(Clone, Debug)]
struct CommandData {
    execs: Vec<Execution>,

    #[cfg(debug_assertions)]
    name: Option<String>,

    stream_scope_id: Option<u64>,
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
        const DEFAULT: &str = "command";

        #[cfg(debug_assertions)]
        {
            self.name.as_deref().unwrap_or(DEFAULT)
        }

        #[cfg(not(debug_assertions))]
        {
            DEFAULT
        }
    }

    fn remap_nodes(&mut self, node_map: &[NodeIndex]) {
        for exec in &mut self.execs {
            exec.remap_nodes(node_map);
        }
    }
}

enum CommandFunction {
    Once(CommandFnOnce),
    Reusable(CommandFn),
}

impl CommandFunction {
    fn is_reusable(&self) -> bool {
        matches!(self, Self::Reusable(_))
    }

    fn record(self, cmd: CommandRef<'_>) -> Option<Self> {
        match self {
            Self::Once(func) => {
                func(cmd);
                None
            }
            Self::Reusable(func) => {
                func(cmd);
                Some(Self::Reusable(func))
            }
        }
    }
}

impl Clone for CommandFunction {
    fn clone(&self) -> Self {
        match self {
            Self::Once(_) => panic!("one-shot command callback cannot be cloned"),
            Self::Reusable(func) => Self::Reusable(Arc::clone(func)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct DepthStencilAttachment {
    attachment: Attachment,
    load: LoadOp<vk::ClearDepthStencilValue>,
    store: StoreOp,
    resolve: Option<DepthStencilResolve>,
    is_attachment: bool,
}

#[derive(Clone, Copy, Debug)]
struct DepthStencilResolve {
    attachment: Attachment,
    dst_attachment_idx: AttachmentIndex,
    depth_mode: Option<ResolveMode>,
    stencil_mode: Option<ResolveMode>,
}

#[derive(Clone)]
enum ExecutionAccess {
    Building(ExecutionAccessBuilder),
    Frozen(FrozenExecutionAccess),
}

impl ExecutionAccess {
    fn contains(&self, node_idx: NodeIndex) -> bool {
        match self {
            Self::Building(builder) => builder.lookup.contains_key(&node_idx),
            Self::Frozen(frozen) => frozen.lookup.contains_key(&node_idx),
        }
    }

    fn freeze(&mut self) {
        let Self::Building(builder) = mem::take(self) else {
            return;
        };

        let ExecutionAccessBuilder { entries, lookup } = builder;
        let entries = entries
            .into_iter()
            .map(|entry| NodeAccess {
                node_idx: entry.node_idx,
                accesses: entry.accesses.into_vec().into_boxed_slice(),
            })
            .collect();

        *self = Self::Frozen(FrozenExecutionAccess { entries, lookup });
    }

    fn get_mut(&mut self, node_idx: &NodeIndex) -> Option<&mut [SubresourceAccess]> {
        let Self::Building(builder) = self else {
            panic!("execution accesses are frozen")
        };

        builder
            .lookup
            .get(node_idx)
            .copied()
            .map(|entry_idx| builder.entries[entry_idx].accesses.as_mut_slice())
    }

    fn iter(&self) -> ExecutionAccessIter<'_> {
        match self {
            Self::Building(builder) => ExecutionAccessIter::Building(builder.entries.iter()),
            Self::Frozen(frozen) => ExecutionAccessIter::Frozen(frozen.entries.iter()),
        }
    }

    fn push(&mut self, node_idx: NodeIndex, access: SubresourceAccess) {
        let Self::Building(builder) = self else {
            panic!("execution accesses are frozen")
        };

        let idx = *builder.lookup.entry(node_idx).or_insert_with(|| {
            let idx = builder.entries.len();
            builder.entries.push(NodeAccessBuilder {
                node_idx,
                accesses: Default::default(),
            });

            idx
        });
        builder.entries[idx].accesses.push(access);
    }

    fn remap_nodes(&mut self, node_map: &[NodeIndex]) {
        match self {
            Self::Building(builder) => {
                for entry in &mut builder.entries {
                    entry.node_idx = node_map[entry.node_idx];
                }

                builder.lookup = builder
                    .entries
                    .iter()
                    .enumerate()
                    .map(|(idx, entry)| (entry.node_idx, idx))
                    .collect();
            }
            Self::Frozen(frozen) => {
                for entry in frozen.entries.iter_mut() {
                    entry.node_idx = node_map[entry.node_idx];
                }

                frozen.lookup = frozen
                    .entries
                    .iter()
                    .enumerate()
                    .map(|(idx, entry)| (entry.node_idx, idx))
                    .collect();
            }
        }
    }
}

impl Debug for ExecutionAccess {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Building(builder) => builder.entries.fmt(f),
            Self::Frozen(frozen) => frozen.entries.fmt(f),
        }
    }
}

impl Default for ExecutionAccess {
    fn default() -> Self {
        Self::Building(Default::default())
    }
}

#[derive(Clone, Debug, Default)]
struct ExecutionAccessBuilder {
    entries: Vec<NodeAccessBuilder>,
    lookup: HashMap<NodeIndex, usize>,
}

enum ExecutionAccessIter<'a> {
    Building(Iter<'a, NodeAccessBuilder>),
    Frozen(Iter<'a, NodeAccess>),
}

impl<'a> Iterator for ExecutionAccessIter<'a> {
    type Item = (NodeIndex, &'a [SubresourceAccess]);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ExecutionAccessIter::Building(iter) => iter
                .next()
                .map(|entry| (entry.node_idx, entry.accesses.as_slice())),
            ExecutionAccessIter::Frozen(iter) => iter
                .next()
                .map(|entry| (entry.node_idx, entry.accesses.as_ref())),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();

        (len, Some(len))
    }
}

impl ExactSizeIterator for ExecutionAccessIter<'_> {
    fn len(&self) -> usize {
        match self {
            ExecutionAccessIter::Building(iter) => iter.len(),
            ExecutionAccessIter::Frozen(iter) => iter.len(),
        }
    }
}

#[derive(Clone, Default)]
struct Execution {
    accesses: ExecutionAccess,
    attachments: ExecutionAttachmentMap,
    bindings: BTreeMap<Binding, (NodeIndex, ViewInfo)>,

    correlated_view_mask: u32,
    depth_stencil: Option<DepthStencilInfo>,
    render_area: Option<vk::Rect2D>,
    view_mask: u32,

    func: Option<CommandFunction>,
    node_map: Option<Arc<[NodeIndex]>>,
    pipeline: Option<ExecutionPipeline>,

    #[cfg(feature = "checked")]
    stream_graph_id: Option<GraphId>,
}

impl Execution {
    fn remap_nodes(&mut self, node_map: &[NodeIndex]) {
        let original_node_map = Arc::<[NodeIndex]>::from(node_map.to_vec());
        self.accesses.remap_nodes(node_map);
        self.attachments.remap_nodes(node_map);

        self.bindings = mem::take(&mut self.bindings)
            .into_iter()
            .map(|(binding, (node_idx, view))| (binding, (node_map[node_idx], view)))
            .collect();
        self.node_map = Some(original_node_map);
    }
}

impl Debug for Execution {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // The only field missing is func which cannot easily be implemented because it is a
        // FnOnce
        f.debug_struct("Execution")
            .field("accesses", &self.accesses)
            .field("attachments", &self.attachments)
            .field("bindings", &self.bindings)
            .field("correlated_view_mask", &self.correlated_view_mask)
            .field("depth_stencil", &self.depth_stencil)
            .field("render_area", &self.render_area)
            .field("view_mask", &self.view_mask)
            .field("pipeline", &self.pipeline)
            .finish()
    }
}

#[derive(Clone, Debug)]
enum ExecutionPipeline {
    Compute(ComputePipeline),
    Graphics(GraphicsPipeline),
    RayTracing(RayTracingPipeline),
}

impl ExecutionPipeline {
    fn as_graphics(&self) -> Option<&GraphicsPipeline> {
        if let Self::Graphics(pipeline) = self {
            Some(pipeline)
        } else {
            None
        }
    }

    fn bind_point(&self) -> vk::PipelineBindPoint {
        match self {
            ExecutionPipeline::Compute(_) => vk::PipelineBindPoint::COMPUTE,
            ExecutionPipeline::Graphics(_) => vk::PipelineBindPoint::GRAPHICS,
            ExecutionPipeline::RayTracing(_) => vk::PipelineBindPoint::RAY_TRACING_KHR,
        }
    }

    fn descriptor_bindings(&self) -> &DescriptorBindingMap {
        match self {
            ExecutionPipeline::Compute(pipeline) => &pipeline.inner.descriptor_bindings,
            ExecutionPipeline::Graphics(pipeline) => &pipeline.inner.descriptor_bindings,
            ExecutionPipeline::RayTracing(pipeline) => &pipeline.inner.descriptor_bindings,
        }
    }

    fn descriptor_info(&self) -> &PipelineDescriptorInfo {
        match self {
            ExecutionPipeline::Compute(pipeline) => &pipeline.inner.descriptor_info,
            ExecutionPipeline::Graphics(pipeline) => &pipeline.inner.descriptor_info,
            ExecutionPipeline::RayTracing(pipeline) => &pipeline.inner.descriptor_info,
        }
    }

    fn expect_compute(&self) -> &ComputePipeline {
        if let Self::Compute(pipeline) = self {
            pipeline
        } else {
            panic!("missing compute pipeline")
        }
    }

    fn expect_graphics(&self) -> &GraphicsPipeline {
        self.as_graphics().expect("missing graphics pipeline")
    }

    fn expect_ray_tracing(&self) -> &RayTracingPipeline {
        if let Self::RayTracing(pipeline) = self {
            pipeline
        } else {
            panic!("missing ray tracing pipeline")
        }
    }

    fn layout(&self) -> vk::PipelineLayout {
        match self {
            ExecutionPipeline::Compute(pipeline) => pipeline.inner.layout,
            ExecutionPipeline::Graphics(pipeline) => pipeline.inner.layout,
            ExecutionPipeline::RayTracing(pipeline) => pipeline.inner.layout,
        }
    }
}

#[derive(Clone, Debug)]
struct FrozenExecutionAccess {
    entries: Box<[NodeAccess]>,
    lookup: HashMap<NodeIndex, usize>,
}

/// A composable graph of Vulkan command buffer operations.
///
/// `Graph` instances are intended for one-time use.
///
/// The design of this code originated with a combination of
/// [`PassBuilder`](https://github.com/EmbarkStudios/kajiya/blob/main/crates/lib/kajiya-rg/src/pass_builder.rs)
/// and
/// [`graph.cpp`](https://github.com/Themaister/Granite/blob/master/renderer/graph.cpp).
#[derive(Debug)]
pub struct Graph {
    cmds: Vec<CommandData>,
    resources: ResourceMap,

    #[cfg(feature = "checked")]
    graph_id: GraphId,
}

impl Default for Graph {
    fn default() -> Self {
        Self {
            cmds: Default::default(),
            resources: Default::default(),

            #[cfg(feature = "checked")]
            graph_id: GraphId::next(),
        }
    }
}

impl Graph {
    /// Constructs a default `Graph`.
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn assert_node_owner<N>(&self, _resource_node: &N)
    where
        N: Node,
    {
        #[cfg(feature = "checked")]
        _resource_node.assert_owner(self.graph_id);
    }

    #[cfg(feature = "checked")]
    pub(crate) fn graph_id(&self) -> GraphId {
        self.graph_id
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

    pub(crate) fn bind_stream_arg_resource(&mut self, resource: AnyResource) -> NodeIndex {
        self.resources.bind(resource)
    }

    /// Copies an image, potentially performing format conversion.
    ///
    /// Records a [`vkCmdBlitImage`] operation covering the full extent of the source and
    /// destination images.
    ///
    /// [`vkCmdBlitImage`]: https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBlitImage.html
    pub fn blit_image(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyImageNode>,
        filter: vk::Filter,
    ) -> &mut Self {
        let src = src.into();
        let src_info = self.resources[src.index()].expect_image_info();

        let dst = dst.into();
        let dst_info = self.resources[dst.index()].expect_image_info();

        self.blit_image_region(
            src,
            dst,
            filter,
            [vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(src_info.format),
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
                    aspect_mask: format_aspect_mask(dst_info.format),
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

    /// Copies regions of an image, potentially performing format conversion.
    ///
    /// Records a [`vkCmdBlitImage`] operation. The caller supplies the Vulkan blit regions and
    /// filter exactly as they will be passed to Vulkan.
    ///
    /// [`vkCmdBlitImage`]: https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdBlitImage.html
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
        let regions = Arc::<[vk::ImageBlit]>::from(regions.as_ref());

        let mut cmd = self.begin_cmd().debug_name("blit image");

        for region in regions.as_ref() {
            let src_region = image_subresource_range_from_layers(region.src_subresource);
            cmd.set_subresource_access(src, src_region, AccessType::TransferRead);

            let dst_region = image_subresource_range_from_layers(region.dst_subresource);
            cmd.set_subresource_access(dst, dst_region, AccessType::TransferWrite);
        }

        cmd.record_stream(move |cmd| {
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

    /// Clears a color image.
    ///
    /// Records a [`vkCmdClearColorImage`] operation for the full image subresource range described
    /// by the image's [`ImageInfo`].
    ///
    /// [`vkCmdClearColorImage`]: https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdClearColorImage.html
    #[profiling::function]
    pub fn clear_color_image(
        &mut self,
        image: impl Into<AnyImageNode>,
        color: impl Into<ClearColorValue>,
    ) -> &mut Self {
        let color = color.into().into();
        let image = image.into();
        let image_view = self.resources[image.index()].expect_image_info().into();

        self.begin_cmd()
            .debug_name("clear color")
            .subresource_access(image, image_view, AccessType::TransferWrite)
            .record_stream(move |cmd| {
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
    ///
    /// Records a [`vkCmdClearDepthStencilImage`] operation for the full image subresource range
    /// described by the image's [`ImageInfo`].
    ///
    /// [`vkCmdClearDepthStencilImage`]: https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdClearDepthStencilImage.html
    #[profiling::function]
    pub fn clear_depth_stencil_image(
        &mut self,
        image: impl Into<AnyImageNode>,
        depth: f32,
        stencil: u32,
    ) -> &mut Self {
        let image = image.into();
        let image_view = self.resources[image.index()].expect_image_info().into();

        self.begin_cmd()
            .debug_name("clear depth/stencil")
            .subresource_access(image, image_view, AccessType::TransferWrite)
            .record_stream(move |cmd| {
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

    /// Copies data between buffers.
    ///
    /// Records a [`vkCmdCopyBuffer`] operation covering the common size of the source and
    /// destination buffers.
    ///
    /// [`vkCmdCopyBuffer`]: https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyBuffer.html
    pub fn copy_buffer(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyBufferNode>,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();
        let src_info = self.resources[src.index()].expect_buffer_info();
        let dst_info = self.resources[dst.index()].expect_buffer_info();

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

    /// Copies data between buffer regions.
    ///
    /// Records a [`vkCmdCopyBuffer`] operation. The caller supplies the Vulkan copy regions exactly
    /// as they will be passed to Vulkan.
    ///
    /// [`vkCmdCopyBuffer`]: https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyBuffer.html
    #[profiling::function]
    pub fn copy_buffer_region(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyBufferNode>,
        regions: impl AsRef<[vk::BufferCopy]> + 'static + Send,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();
        let regions = Arc::<[vk::BufferCopy]>::from(regions.as_ref());

        #[cfg(feature = "checked")]
        let src_size = self.resources[src.index()].expect_buffer_info().size;

        #[cfg(feature = "checked")]
        let dst_size = self.resources[dst.index()].expect_buffer_info().size;

        let mut cmd = self.begin_cmd().debug_name("copy buffer");

        for region in regions.iter() {
            #[cfg(feature = "checked")]
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

        cmd.record_stream(move |cmd| {
            let src = cmd.resource(src);
            let dst = cmd.resource(dst);

            unsafe {
                cmd.device
                    .cmd_copy_buffer(cmd.handle, src.handle, dst.handle, &regions);
            }
        })
        .end_cmd()
    }

    /// Copies data from a buffer into an image.
    ///
    /// Records a [`vkCmdCopyBufferToImage`] operation covering the full destination image.
    ///
    /// [`vkCmdCopyBufferToImage`]: https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyBufferToImage.html
    pub fn copy_buffer_to_image(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyImageNode>,
    ) -> &mut Self {
        let dst = dst.into();
        let dst_info = self.resources[dst.index()].expect_image_info();

        self.copy_buffer_to_image_region(
            src,
            dst,
            [vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: dst_info.width,
                buffer_image_height: dst_info.height,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(dst_info.format),
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

    /// Copies data from a buffer into image regions.
    ///
    /// Records a [`vkCmdCopyBufferToImage`] operation. The caller supplies the Vulkan copy regions
    /// exactly as they will be passed to Vulkan.
    ///
    /// [`vkCmdCopyBufferToImage`]: https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyBufferToImage.html
    #[profiling::function]
    pub fn copy_buffer_to_image_region(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyImageNode>,
        regions: impl AsRef<[vk::BufferImageCopy]> + 'static + Send,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();
        let dst_info = self.resources[dst.index()].expect_image_info();
        let regions = Arc::<[vk::BufferImageCopy]>::from(regions.as_ref());

        let mut cmd = self.begin_cmd().debug_name("copy buffer to image");

        for region in regions.iter() {
            let block_bytes_size = format_texel_block_size(dst_info.format);
            let (block_height, block_width) = format_texel_block_extent(dst_info.format);
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

        cmd.record_stream(move |cmd| {
            let src = cmd.resource(src);
            let dst = cmd.resource(dst);

            unsafe {
                cmd.device.cmd_copy_buffer_to_image(
                    cmd.handle,
                    src.handle,
                    dst.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                );
            }
        })
        .end_cmd()
    }

    /// Copies all layers of a source image to a destination image.
    ///
    /// Records a [`vkCmdCopyImage`] operation covering the common extent of the source and
    /// destination images.
    ///
    /// [`vkCmdCopyImage`]: https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyImage.html
    pub fn copy_image(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyImageNode>,
    ) -> &mut Self {
        let src = src.into();
        let src_info = self.resources[src.index()].expect_image_info();

        let dst = dst.into();
        let dst_info = self.resources[dst.index()].expect_image_info();

        self.copy_image_region(
            src,
            dst,
            [vk::ImageCopy {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(src_info.format),
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: src_info.array_layer_count,
                },
                src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(dst_info.format),
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

    /// Copies data between image regions.
    ///
    /// Records a [`vkCmdCopyImage`] operation. The caller supplies the Vulkan copy regions exactly
    /// as they will be passed to Vulkan.
    ///
    /// [`vkCmdCopyImage`]: https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyImage.html
    #[profiling::function]
    pub fn copy_image_region(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyImageNode>,
        regions: impl AsRef<[vk::ImageCopy]> + 'static + Send,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();
        let regions = Arc::<[vk::ImageCopy]>::from(regions.as_ref());

        let mut cmd = self.begin_cmd().debug_name("copy image");

        for region in regions.iter() {
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

        cmd.record_stream(move |cmd| {
            let src = cmd.resource(src);
            let dst = cmd.resource(dst);

            unsafe {
                cmd.device.cmd_copy_image(
                    cmd.handle,
                    src.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                );
            }
        })
        .end_cmd()
    }

    /// Copies image data into a buffer.
    ///
    /// Records a [`vkCmdCopyImageToBuffer`] operation covering the full source image.
    ///
    /// [`vkCmdCopyImageToBuffer`]: https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyImageToBuffer.html
    pub fn copy_image_to_buffer(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyBufferNode>,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();

        let src_info = self.resources[src.index()].expect_image_info();

        self.copy_image_to_buffer_region(
            src,
            dst,
            [vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: src_info.width,
                buffer_image_height: src_info.height,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(src_info.format),
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

    /// Copies image region data into a buffer.
    ///
    /// Records a [`vkCmdCopyImageToBuffer`] operation. The caller supplies the Vulkan copy regions
    /// exactly as they will be passed to Vulkan.
    ///
    /// [`vkCmdCopyImageToBuffer`]: https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdCopyImageToBuffer.html
    #[profiling::function]
    pub fn copy_image_to_buffer_region(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyBufferNode>,
        regions: impl AsRef<[vk::BufferImageCopy]> + 'static + Send,
    ) -> &mut Self {
        let src = src.into();
        let src_info = self.resources[src.index()].expect_image_info();
        let dst = dst.into();
        let regions = Arc::<[vk::BufferImageCopy]>::from(regions.as_ref());

        let mut cmd = self.begin_cmd().debug_name("copy image to buffer");

        for region in regions.iter() {
            let block_bytes_size = format_texel_block_size(src_info.format);
            let (block_height, block_width) = format_texel_block_extent(src_info.format);
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

        cmd.record_stream(move |cmd| {
            let src = cmd.resource(src);
            let dst = cmd.resource(dst);

            unsafe {
                cmd.device.cmd_copy_image_to_buffer(
                    cmd.handle,
                    src.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst.handle,
                    &regions,
                );
            }
        })
        .end_cmd()
    }

    /// Fills a region of a buffer with a fixed value.
    ///
    /// Records a [`vkCmdFillBuffer`] operation.
    ///
    /// [`vkCmdFillBuffer`]: https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdFillBuffer.html
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
            .record_stream(move |cmd| {
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

    /// Returns the index of the first command which accesses a given node.
    #[profiling::function]
    fn first_node_access_pass_index(&self, resource_node: impl Node) -> Option<usize> {
        self.assert_node_owner(&resource_node);

        let node_idx = resource_node.index();

        for (pass_idx, pass) in self.cmds.iter().enumerate() {
            for exec in pass.execs.iter() {
                if exec.accesses.contains(node_idx) {
                    return Some(pass_idx);
                }
            }
        }

        None
    }

    /// Finalizes the graph and provides an object with functions for submitting the resulting
    /// commands.
    #[profiling::function]
    pub fn finalize(mut self) -> Submission {
        // The final execution of each command has no function.
        for cmd in &mut self.cmds {
            debug_assert!(cmd.expect_last_exec().func.is_none());

            cmd.execs.pop();

            for exec in &mut cmd.execs {
                exec.accesses.freeze();
            }
        }

        Submission::new(self)
    }

    /// Returns a borrow of the Vulkan resource represented by `resource_node`.
    ///
    /// The exact return type depends on the node type:
    ///
    /// - Concrete nodes such as [`BufferNode`] and [`ImageNode`] return the exact stored handle
    ///   type, such as
    ///   `&Arc<Buffer>` or `&Arc<Image>`.
    /// - Erased nodes such as [`AnyBufferNode`] and [`AnyImageNode`] return a borrow of the
    ///   underlying resource,
    ///   such as `&Buffer` or `&Image`.
    ///
    /// This distinction lets erased node enums unify owned, leased, and swapchain-backed resources
    /// behind a single resource view.
    ///
    /// Node ownership is validated here when the `checked` feature is enabled. With `checked`
    /// disabled, callers must ensure `resource_node` came from this graph.
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: Node,
    {
        self.assert_node_owner(&resource_node);
        resource_node.borrow(&self.resources)
    }

    /// Records a [`vkCmdUpdateBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdUpdateBuffer.html)
    /// command.
    ///
    /// Vulkan requires `data` to be at most `65536` bytes.
    ///
    /// These constraints are validated by the Vulkan Validation Layer (VVL) when it is active.
    /// When the `checked` feature is enabled, `vk-graph` also validates the data size and bounds
    /// before recording the command.
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

        #[cfg(feature = "checked")]
        {
            assert!(
                data.as_ref().len() <= 64 * 1024,
                "data length ({}) exceeds vkCmdUpdateBuffer limit (65536)",
                data.as_ref().len()
            );

            let buffer_info = self.resources[buffer.index()].expect_buffer_info();

            assert!(
                data_end <= buffer_info.size,
                "data range end ({data_end}) exceeds buffer size ({})",
                buffer_info.size
            );
        }

        let data = Arc::<[u8]>::from(data.as_ref());

        self.begin_cmd()
            .debug_name("update buffer")
            .subresource_access(buffer, offset..data_end, AccessType::TransferWrite)
            .record_stream(move |cmd| {
                let buffer = cmd.resource(buffer);

                unsafe {
                    cmd.device
                        .cmd_update_buffer(cmd.handle, buffer.handle, offset, &data);
                }
            })
            .end_cmd()
    }
}

/// Specifies the state of a color or combined depth and stencil attachment image during graphics
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

/// A Vulkan resource which has been bound to a [`Graph`].
///
/// See [`Graph::bind_resource`].
///
/// This trait is sealed and cannot be implemented outside of `vk-graph`.
#[allow(private_bounds)]
pub trait Node: private::NodeSealed {
    /// The Vulkan buffer, image, or acceleration structure type.
    type Resource;

    /// Synchronization state snapshot returned for this node's resource type.
    type SyncInfo;

    #[doc(hidden)]
    fn index(&self) -> usize;
}

#[derive(Clone, Debug)]
struct NodeAccess {
    node_idx: NodeIndex,
    accesses: Box<[SubresourceAccess]>,
}

#[derive(Clone, Debug)]
struct NodeAccessBuilder {
    node_idx: NodeIndex,
    accesses: SmallVec<[SubresourceAccess; 2]>,
}

mod private {
    use super::{AnyResource, Node};

    #[cfg(feature = "checked")]
    use super::GraphId;

    /// Prevents external implementations of [`Node`] and provides crate-private node internals.
    pub(crate) trait NodeSealed: Sized {
        fn borrow(self, resources: &[AnyResource]) -> &<Self as Node>::Resource
        where
            Self: Node;

        fn borrow_at(self, resources: &[AnyResource], index: usize) -> &<Self as Node>::Resource
        where
            Self: Node,
        {
            debug_assert_eq!(self.index(), index);
            self.borrow(resources)
        }

        #[cfg(feature = "checked")]
        fn assert_owner(&self, _graph_id: GraphId) {}
    }

    /// Prevents external implementations of [`Resource`](super::Resource).
    pub(crate) trait ResourceSealed {}
}

/// A Vulkan resource which may be bound to a [`Graph`].
///
/// See [`Graph::bind_resource`] and
/// [`Command::bind_resource`](crate::cmd::Command::bind_resource).
///
/// This trait is sealed and cannot be implemented outside of `vk-graph`.
#[allow(private_bounds)]
pub trait Resource: private::ResourceSealed {
    /// The resource handle type.
    type Node;

    #[doc(hidden)]
    fn bind_graph(self, _: &mut Graph) -> Self::Node;
}

impl private::ResourceSealed for SwapchainImage {}

impl Resource for SwapchainImage {
    type Node = SwapchainImageNode;

    fn bind_graph(self, graph: &mut Graph) -> Self::Node {
        let node = Self::Node::new(
            graph.resources.len(),
            #[cfg(feature = "checked")]
            graph.graph_id,
        );

        //trace!("Node {}: {:?}", res.idx, &self);

        let resource = AnyResource::SwapchainImage(Box::new(self));
        graph.resources.bind(resource);

        node
    }
}

macro_rules! resource {
    ($name:ident) => {
        paste::paste! {
            impl private::ResourceSealed for $name {}
            impl private::ResourceSealed for Arc<$name> {}
            impl<'a> private::ResourceSealed for &'a Arc<$name> {}
            impl private::ResourceSealed for Lease<$name> {}
            impl private::ResourceSealed for Arc<Lease<$name>> {}
            impl<'a> private::ResourceSealed for &'a Arc<Lease<$name>> {}

            impl Resource for $name {
                type Node = [<$name Node>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // Bind a new owned resource, such as Image or Buffer.

                    // We will return a new node
                    Arc::new(self).bind_graph(graph)
                }
            }

            impl Resource for Arc<$name> {
                type Node = [<$name Node>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // Bind an existing shared resource, such as Arc<Image> or Arc<Buffer>.

                    // We will return an existing node, if possible
                    Self::Node::new(
                        graph.resources.bind_shared(self),
                        #[cfg(feature = "checked")]
                        graph.graph_id,
                    )
                }
            }

            impl<'a> Resource for &'a Arc<$name> {
                type Node = [<$name Node>];

                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // Bind a borrowed shared resource, such as &Arc<Image> or &Arc<Buffer>.

                    Arc::clone(self).bind_graph(graph)
                }
            }

            impl Resource for Lease<$name> {
                type Node = [<$name LeaseNode>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // Bind a new pooled resource, such as Lease<Image> or Lease<Buffer>.

                    // We will return a new node
                    Arc::new(self).bind_graph(graph)
                }
            }

            impl Resource  for Arc<Lease<$name>> {
                type Node = [<$name LeaseNode>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // Bind an existing shared pooled resource, such as Arc<Lease<Image>> or
                    // Arc<Lease<Buffer>>.

                    // We will return an existing node, if possible
                    Self::Node::new(
                        graph.resources.bind_shared(self),
                        #[cfg(feature = "checked")]
                        graph.graph_id,
                    )
                }
            }

            impl<'a> Resource for &'a Arc<Lease<$name>> {
                type Node = [<$name LeaseNode>];

                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // Bind a borrowed shared pooled resource, such as &Arc<Lease<Image>> or
                    // &Arc<Lease<Buffer>>.

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
    pub(crate) fn from_resources(resources: Vec<AnyResource>) -> Self {
        Self {
            addr_index: HashMap::new(),
            resources,
        }
    }

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

/// Specifies the state of a color or combined depth and stencil attachment image after graphics
/// render pass framebuffer store operations.
///
/// Use this to specify the desired contents of any image after use in a pipeline command buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StoreOp {
    /// The attachment will become undefined and reads will produce garbage data.
    DontCare,

    /// The attachment will be preserved in memory.
    Store,
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
                let accel_supported = device
                    .physical_device
                    .vk_khr_acceleration_structure
                    .is_some();

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
