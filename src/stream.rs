//! Reusable command streams.
//!
//! A [`CommandStream`] is a prepared graph-like command sequence that can be inserted into a
//! per-frame [`Graph`] with typed arguments.
//!
//! Streams are useful when part of a frame is structurally the same across many frames but still
//! needs per-frame resources such as the current swapchain image. Declare those resources as stream
//! arguments, record reusable commands once, and bind concrete graph nodes when inserting the stream.
//!
//! ```no_run
//! # use ash::vk;
//! # use vk_graph::{Graph, node::{BufferNode, ImageNode}, pool::hash::HashPool};
//! # use vk_graph::cmd::{LoadOp, StoreOp};
//! # use vk_graph::driver::buffer::BufferInfo;
//! # use vk_graph::driver::graphics::GraphicsPipeline;
//! # use vk_graph::driver::image::ImageInfo;
//! # use vk_graph::stream::CommandStream;
//! # use vk_sync::AccessType;
//! # let mut pool: HashPool = todo!();
//! # let pipeline: GraphicsPipeline = todo!();
//! # let swapchain_image: ImageNode = todo!();
//! # let vertex_buffer: BufferNode = todo!();
//! let stream = CommandStream::prepare(&mut pool, |stream| {
//!     let output = stream.arg(ImageInfo::image_2d(
//!         1280,
//!         720,
//!         vk::Format::R8G8B8A8_UNORM,
//!         vk::ImageUsageFlags::COLOR_ATTACHMENT,
//!     ));
//!     let vertices = stream.arg(BufferInfo::device_mem(
//!         4096,
//!         vk::BufferUsageFlags::VERTEX_BUFFER,
//!     ));
//!
//!     stream
//!         .begin_cmd()
//!         .debug_name("reusable overlay")
//!         .bind_pipeline(&pipeline)
//!         .color_attachment_image(0, output, LoadOp::Load, StoreOp::Store)
//!         .resource_access(vertices, AccessType::VertexBuffer)
//!         .record_cmd(move |cmd| {
//!             cmd.bind_vertex_buffer(0, vertices, 0).draw(3, 1, 0, 0);
//!         });
//!
//!     (output, vertices)
//! })?;
//!
//! let mut graph = Graph::new();
//! graph
//!     .insert_cmd_stream(&stream)
//!     .with_arg(stream.args.0, swapchain_image)
//!     .with_arg(stream.args.1, vertex_buffer)
//!     .finish();
//! # Ok::<(), vk_graph::driver::DriverError>(())
//! ```

use crate::private::NodeSealed;
use {
    crate::{
        AnyResource, Graph, Node, Resource, ResourceMap,
        cmd::{
            AttachmentIndex, ClearColorValue, Command, CommandRef, ComputeCommandRef,
            GraphicsCommandRef, LoadOp, PipelineCommand, RayTracingCommandRef, StoreOp,
            Subresource, SubresourceRange,
        },
        driver::{
            DriverError,
            accel_struct::{
                AccelerationStructure, AccelerationStructureInfo, AccelerationStructureInfoBuilder,
            },
            buffer::{Buffer, BufferInfo, BufferInfoBuilder},
            compute::ComputePipeline,
            graphics::{DepthStencilInfo, GraphicsPipeline},
            image::{Image, ImageInfo, ImageInfoBuilder, ImageViewInfo},
            ray_tracing::RayTracingPipeline,
        },
        node::{
            AccelerationStructureLeaseNode, AccelerationStructureNode,
            AnyAccelerationStructureNode, AnyBufferNode, AnyImageNode, BufferLeaseNode, BufferNode,
            ImageLeaseNode, ImageNode, SwapchainImageNode,
        },
        pool::SubmissionPool,
        submission::Submission,
    },
    ash::vk,
    std::{
        collections::HashMap,
        marker::PhantomData,
        ops::Range,
        sync::{Arc, Mutex},
    },
};

#[cfg(feature = "checked")]
use crate::GraphId;

use std::sync::atomic::{AtomicU64, Ordering};

fn next_stream_scope_id() -> u64 {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);

    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(feature = "checked")]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct CommandStreamId(u64);

#[cfg(feature = "checked")]
impl CommandStreamId {
    fn next() -> Self {
        Self(next_stream_scope_id())
    }
}

/// A typed external argument for a [`CommandStream`].
///
/// `StreamArg` values are created with [`CommandStreamMut::arg`] while building a stream and are
/// later bound to parent-graph nodes with [`CommandStreamRun::with_arg`].
///
/// ```no_run
/// # use ash::vk;
/// # use vk_graph::{Graph, driver::image::ImageInfo, node::ImageNode, stream::CommandStream};
/// # let swapchain_image: ImageNode = todo!();
/// let stream = CommandStream::finalize(|stream| {
///     stream.arg(ImageInfo::image_2d(
///         640,
///         480,
///         vk::Format::R8G8B8A8_UNORM,
///         vk::ImageUsageFlags::COLOR_ATTACHMENT,
///     ))
/// })
/// .into_stream();
///
/// let mut graph = Graph::new();
/// graph
///     .insert_cmd_stream(&stream)
///     .with_arg(stream.args, swapchain_image)
///     .finish();
/// ```
#[derive(Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StreamArg<T> {
    pub(crate) arg_index: usize,
    pub(crate) index: usize,

    #[cfg(feature = "checked")]
    pub(crate) stream_id: CommandStreamId,

    #[cfg(feature = "checked")]
    pub(crate) graph_id: GraphId,

    __: PhantomData<fn() -> T>,
}

impl<T> Clone for StreamArg<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for StreamArg<T> {}

impl<T> StreamArg<T> {
    pub(crate) fn new(
        arg_index: usize,
        index: usize,
        #[cfg(feature = "checked")] stream_id: CommandStreamId,
        #[cfg(feature = "checked")] graph_id: GraphId,
    ) -> Self {
        Self {
            arg_index,
            index,
            #[cfg(feature = "checked")]
            stream_id,
            #[cfg(feature = "checked")]
            graph_id,
            __: PhantomData,
        }
    }
}

impl NodeSealed for StreamArg<AccelerationStructure> {
    fn borrow(self, resources: &[AnyResource]) -> &<Self as Node>::Resource {
        resources[self.index].expect_accel_struct()
    }

    fn borrow_at(self, resources: &[AnyResource], index: usize) -> &<Self as Node>::Resource {
        resources[index].expect_accel_struct()
    }

    #[cfg(feature = "checked")]
    fn assert_owner(&self, _graph_id: GraphId) {
        #[cfg(feature = "checked")]
        assert!(
            self.graph_id == _graph_id,
            "node belongs to a different graph"
        );
    }
}

impl Node for StreamArg<AccelerationStructure> {
    type Resource = AccelerationStructure;
    type SyncInfo = crate::driver::accel_struct::AccelerationStructureSyncInfo;

    fn index(&self) -> usize {
        self.index
    }
}

impl NodeSealed for StreamArg<Buffer> {
    fn borrow(self, resources: &[AnyResource]) -> &<Self as Node>::Resource {
        resources[self.index].expect_buffer()
    }

    fn borrow_at(self, resources: &[AnyResource], index: usize) -> &<Self as Node>::Resource {
        resources[index].expect_buffer()
    }

    #[cfg(feature = "checked")]
    fn assert_owner(&self, _graph_id: GraphId) {
        #[cfg(feature = "checked")]
        assert!(
            self.graph_id == _graph_id,
            "node belongs to a different graph"
        );
    }
}

impl Node for StreamArg<Buffer> {
    type Resource = Buffer;
    type SyncInfo = crate::driver::buffer::BufferSyncInfo;

    fn index(&self) -> usize {
        self.index
    }
}

impl NodeSealed for StreamArg<Image> {
    fn borrow(self, resources: &[AnyResource]) -> &<Self as Node>::Resource {
        resources[self.index].expect_image()
    }

    fn borrow_at(self, resources: &[AnyResource], index: usize) -> &<Self as Node>::Resource {
        resources[index].expect_image()
    }

    #[cfg(feature = "checked")]
    fn assert_owner(&self, _graph_id: GraphId) {
        #[cfg(feature = "checked")]
        assert!(
            self.graph_id == _graph_id,
            "node belongs to a different graph"
        );
    }
}

impl Node for StreamArg<Image> {
    type Resource = Image;
    type SyncInfo = crate::driver::image::ImageSyncInfo;

    fn index(&self) -> usize {
        self.index
    }
}

/// A stream argument for an acceleration structure.
///
/// ```no_run
/// # use vk_graph::driver::accel_struct::AccelerationStructureInfo;
/// # use vk_graph::stream::{AccelerationStructureArg, CommandStream};
/// # let info: AccelerationStructureInfo = todo!();
/// let stream = CommandStream::finalize(|stream| -> AccelerationStructureArg {
///     stream.arg(info)
/// })
/// .into_stream();
/// ```
pub type AccelerationStructureArg = StreamArg<AccelerationStructure>;

/// A stream argument for a buffer.
///
/// ```no_run
/// # use ash::vk;
/// # use vk_graph::driver::buffer::BufferInfo;
/// # use vk_graph::stream::{BufferArg, CommandStream};
/// let stream = CommandStream::finalize(|stream| -> BufferArg {
///     stream.arg(BufferInfo::device_mem(
///         4096,
///         vk::BufferUsageFlags::STORAGE_BUFFER,
///     ))
/// })
/// .into_stream();
/// ```
pub type BufferArg = StreamArg<Buffer>;

/// A stream argument for an image.
///
/// ```no_run
/// # use ash::vk;
/// # use vk_graph::driver::image::ImageInfo;
/// # use vk_graph::stream::{CommandStream, ImageArg};
/// let stream = CommandStream::finalize(|stream| -> ImageArg {
///     stream.arg(ImageInfo::image_2d(
///         128,
///         128,
///         vk::Format::R8G8B8A8_UNORM,
///         vk::ImageUsageFlags::SAMPLED,
///     ))
/// })
/// .into_stream();
/// ```
pub type ImageArg = StreamArg<Image>;

#[derive(Clone, Copy, Debug)]
pub(crate) enum StreamArgData {
    AccelerationStructure(AccelerationStructureInfo),
    Buffer(BufferInfo),
    Image(ImageInfo),
}

#[derive(Debug)]
pub(crate) struct CommandStreamInner {
    pub(crate) arg_nodes: Box<[usize]>,
    pub(crate) args: Box<[StreamArgData]>,
    pub(crate) prepared: bool,
    pub(crate) submission: Mutex<Submission>,

    #[cfg(feature = "checked")]
    pub(crate) stream_id: CommandStreamId,

    #[cfg(feature = "checked")]
    pub(crate) graph_id: GraphId,
}

/// A reusable command stream.
///
/// Prepared streams reduce repeated CPU-side graph construction and preparation work by caching an
/// optimized schedule and static recording resources. Unprepared streams keep finalization cheaper
/// up front, but each insertion still has to reconcile arguments, dependencies, scheduling, and
/// recording with the parent graph.
///
/// Inserting or concatenating many tiny streams is not free. Profile release builds before designing
/// around heavy stream composition.
///
/// ```no_run
/// # use vk_graph::{Graph, pool::hash::HashPool, stream::CommandStream};
/// # let mut pool: HashPool = todo!();
/// let stream = CommandStream::prepare(&mut pool, |stream| {
///     stream.begin_cmd().debug_name("cached commands").record_cmd(|_| {});
/// })?;
///
/// let mut graph = Graph::new();
/// graph.insert_cmd_stream(&stream).finish();
/// # Ok::<(), vk_graph::driver::DriverError>(())
/// ```
#[derive(Clone, Debug)]
pub struct CommandStream<A = ()> {
    /// Typed handles returned by the preparation callback.
    pub args: A,
    pub(crate) inner: Arc<CommandStreamInner>,
}

/// A finalized command stream definition that can be prepared later.
///
/// Drafts are useful when construction should happen separately from preparation. Convert a draft
/// with [`CommandStreamDraft::into_stream`] for unprepared insertion or
/// [`CommandStreamDraft::prepare`] to cache preparation work.
///
/// ```no_run
/// # use vk_graph::{Graph, pool::hash::HashPool, stream::CommandStream};
/// # let mut pool: HashPool = todo!();
/// let draft = CommandStream::finalize(|stream| {
///     stream.begin_cmd().record_cmd(|_| {});
/// });
///
/// let prepared = draft.prepare(&mut pool)?;
/// let mut graph = Graph::new();
/// graph.insert_cmd_stream(&prepared).finish();
/// # Ok::<(), vk_graph::driver::DriverError>(())
/// ```
#[derive(Debug)]
pub struct CommandStreamDraft<A = ()> {
    /// Typed handles returned by the finalization callback.
    pub args: A,
    inner: CommandStreamInner,
}

/// A mutable graph-like command stream being prepared.
///
/// `CommandStreamMut` is passed to [`CommandStream::finalize`] and [`CommandStream::prepare`]
/// callbacks. It provides graph-like methods plus [`CommandStreamMut::arg`] for typed stream
/// inputs.
///
/// ```no_run
/// # use ash::vk;
/// # use vk_graph::{driver::buffer::BufferInfo, stream::CommandStream};
/// let stream = CommandStream::finalize(|stream| {
///     let staging = stream.arg(BufferInfo::host_mem(
///         1024,
///         vk::BufferUsageFlags::TRANSFER_SRC,
///     ));
///     stream.begin_cmd().resource_access(staging, vk_sync::AccessType::TransferRead);
///     staging
/// })
/// .into_stream();
/// ```
pub struct CommandStreamMut {
    pub(crate) arg_nodes: Vec<usize>,
    pub(crate) args: Vec<StreamArgData>,
    pub(crate) graph: Graph,
    #[cfg(feature = "checked")]
    pub(crate) stream_id: CommandStreamId,
}

/// A command being recorded into a [`CommandStreamMut`].
///
/// ```no_run
/// # use vk_graph::stream::CommandStream;
/// let stream = CommandStream::finalize(|stream| {
///     stream
///         .begin_cmd()
///         .debug_name("stream command")
///         .record_cmd(|cmd| {
///             let _ = cmd;
///         });
/// })
/// .into_stream();
/// ```
pub struct StreamCommand<'a> {
    inner: Command<'a>,
}

/// A stream command with a bound pipeline.
///
/// ```no_run
/// # use vk_graph::stream::CommandStream;
/// # use vk_graph::driver::compute::ComputePipeline;
/// # let pipeline: ComputePipeline = todo!();
/// let stream = CommandStream::finalize(|stream| {
///     stream
///         .begin_cmd()
///         .bind_pipeline(&pipeline)
///         .record_cmd(|cmd| {
///             cmd.dispatch(1, 1, 1);
///         });
/// })
/// .into_stream();
/// ```
pub struct StreamPipelineCommand<'a, T> {
    inner: PipelineCommand<'a, T>,
}

/// A pipeline that can be bound to a stream command.
#[doc(hidden)]
pub trait StreamPipeline<'a>: stream_private::StreamPipelineSealed {
    /// The stream command type returned after binding.
    type Command;

    /// Stream equivalent of [`Pipeline::bind_cmd`].
    fn bind_stream_cmd(self, cmd: StreamCommand<'a>) -> Self::Command;
}

macro_rules! stream_pipeline {
    ($pipeline:ty) => {
        impl<'a> StreamPipeline<'a> for $pipeline {
            type Command = StreamPipelineCommand<'a, $pipeline>;

            fn bind_stream_cmd(self, cmd: StreamCommand<'a>) -> Self::Command {
                StreamPipelineCommand {
                    inner: cmd.inner.bind_pipeline(self),
                }
            }
        }

        impl stream_private::StreamPipelineSealed for $pipeline {}

        impl<'a> StreamPipeline<'a> for &'a $pipeline {
            type Command = StreamPipelineCommand<'a, $pipeline>;

            fn bind_stream_cmd(self, cmd: StreamCommand<'a>) -> Self::Command {
                StreamPipelineCommand {
                    inner: cmd.inner.bind_pipeline(self),
                }
            }
        }

        impl<'a> stream_private::StreamPipelineSealed for &'a $pipeline {}
    };
}

stream_pipeline!(ComputePipeline);
stream_pipeline!(GraphicsPipeline);
stream_pipeline!(RayTracingPipeline);

#[allow(private_bounds)]
impl<'a> StreamCommand<'a> {
    /// Stream equivalent of [`Command::bind_resource`].
    pub fn bind_resource<R>(&mut self, resource: R) -> R::Node
    where
        R: Resource,
    {
        self.inner.bind_resource(resource)
    }

    /// Stream equivalent of [`Command::bind_pipeline`].
    pub fn bind_pipeline<P>(self, pipeline: P) -> P::Command
    where
        P: StreamPipeline<'a>,
    {
        pipeline.bind_stream_cmd(self)
    }

    /// Stream equivalent of [`Command::debug_name`].
    pub fn debug_name(mut self, name: impl Into<String>) -> Self {
        self.inner.set_debug_name(name);
        self
    }

    /// Stream equivalent of [`Command::record_cmd`].
    ///
    /// Unlike graph commands, stream callbacks must be reusable and therefore implement
    /// `Fn + Send + Sync + 'static`.
    pub fn record_cmd(
        mut self,
        func: impl for<'r> Fn(CommandRef<'r>) + Send + Sync + 'static,
    ) -> Self {
        self.record_cmd_mut(func);
        self
    }

    /// Mutable-borrow stream equivalent of [`Command::record_cmd`].
    ///
    /// Unlike graph commands, stream callbacks must be reusable and therefore implement
    /// `Fn + Send + Sync + 'static`.
    pub fn record_cmd_mut(
        &mut self,
        func: impl for<'r> Fn(CommandRef<'r>) + Send + Sync + 'static,
    ) {
        self.inner.record_stream_mut(func);
    }

    /// Stream equivalent of [`Command::resource_access`].
    pub fn resource_access<N>(mut self, resource_node: N, access: vk_sync::AccessType) -> Self
    where
        N: Node + Subresource,
        SubresourceRange: From<N::Range>,
    {
        self.inner.set_resource_access(resource_node, access);
        self
    }

    /// Mutable-borrow stream equivalent of [`Command::resource_access`].
    pub fn set_resource_access<N>(&mut self, resource_node: N, access: vk_sync::AccessType)
    where
        N: Node + Subresource,
        SubresourceRange: From<N::Range>,
    {
        self.inner.set_resource_access(resource_node, access);
    }
}

#[allow(private_bounds)]
impl<'a, T> StreamPipelineCommand<'a, T> {
    /// Stream equivalent of [`PipelineCommand::bind_resource`].
    pub fn bind_resource<R>(&mut self, resource: R) -> R::Node
    where
        R: Resource,
    {
        self.inner.bind_resource(resource)
    }

    /// Stream equivalent of [`PipelineCommand::resource_access`].
    pub fn resource_access<N>(mut self, resource_node: N, access: vk_sync::AccessType) -> Self
    where
        N: Node + Subresource,
        SubresourceRange: From<N::Range>,
    {
        self.inner.set_resource_access(resource_node, access);
        self
    }

    /// Mutable-borrow stream equivalent of [`PipelineCommand::resource_access`].
    pub fn set_resource_access<N>(
        &mut self,
        resource_node: N,
        access: vk_sync::AccessType,
    ) -> &mut Self
    where
        N: Node + Subresource,
        SubresourceRange: From<N::Range>,
    {
        self.inner.set_resource_access(resource_node, access);
        self
    }
}

impl StreamPipelineCommand<'_, ComputePipeline> {
    /// Stream equivalent of [`PipelineCommand::<ComputePipeline>::record_cmd`].
    ///
    /// Unlike graph commands, stream callbacks must be reusable and therefore implement
    /// `Fn + Send + Sync + 'static`.
    pub fn record_cmd(
        mut self,
        func: impl for<'r> Fn(ComputeCommandRef<'r>) + Send + Sync + 'static,
    ) -> Self {
        self.record_cmd_mut(func);
        self
    }

    /// Mutable-borrow stream equivalent of [`PipelineCommand::<ComputePipeline>::record_cmd`].
    ///
    /// Unlike graph commands, stream callbacks must be reusable and therefore implement
    /// `Fn + Send + Sync + 'static`.
    pub fn record_cmd_mut(
        &mut self,
        func: impl for<'r> Fn(ComputeCommandRef<'r>) + Send + Sync + 'static,
    ) {
        self.inner.record_stream_mut(func);
    }
}

impl StreamPipelineCommand<'_, GraphicsPipeline> {
    /// Stream equivalent of [`PipelineCommand::<GraphicsPipeline>::depth_stencil`].
    pub fn depth_stencil(mut self, depth_stencil: impl Into<DepthStencilInfo>) -> Self {
        self.inner.set_depth_stencil(depth_stencil);
        self
    }

    /// Stream equivalent of [`PipelineCommand::<GraphicsPipeline>::color_attachment_image`].
    pub fn color_attachment_image(
        mut self,
        color_attachment: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        load: LoadOp<ClearColorValue>,
        store: StoreOp,
    ) -> Self {
        self.inner
            .set_color_attachment_image(color_attachment, image, load, store);
        self
    }

    /// Stream equivalent of [`PipelineCommand::<GraphicsPipeline>::color_attachment_image_view`].
    pub fn color_attachment_image_view(
        mut self,
        color_attachment: AttachmentIndex,
        image: impl Into<AnyImageNode>,
        image_view_info: impl Into<ImageViewInfo>,
        load: LoadOp<ClearColorValue>,
        store: StoreOp,
    ) -> Self {
        self.inner.set_color_attachment_image_view(
            color_attachment,
            image,
            image_view_info,
            load,
            store,
        );
        self
    }

    /// Stream equivalent of [`PipelineCommand::<GraphicsPipeline>::depth_stencil_attachment_image`].
    pub fn depth_stencil_attachment_image(
        mut self,
        image: impl Into<AnyImageNode>,
        load: LoadOp<vk::ClearDepthStencilValue>,
        store: StoreOp,
    ) -> Self {
        self.inner
            .set_depth_stencil_attachment_image(image, load, store);
        self
    }

    /// Stream equivalent of [`PipelineCommand::<GraphicsPipeline>::record_cmd`].
    ///
    /// Unlike graph commands, stream callbacks must be reusable and therefore implement
    /// `Fn + Send + Sync + 'static`.
    pub fn record_cmd(
        mut self,
        func: impl for<'r> Fn(GraphicsCommandRef<'r>) + Send + Sync + 'static,
    ) -> Self {
        self.record_cmd_mut(func);
        self
    }

    /// Mutable-borrow stream equivalent of [`PipelineCommand::<GraphicsPipeline>::record_cmd`].
    ///
    /// Unlike graph commands, stream callbacks must be reusable and therefore implement
    /// `Fn + Send + Sync + 'static`.
    pub fn record_cmd_mut(
        &mut self,
        func: impl for<'r> Fn(GraphicsCommandRef<'r>) + Send + Sync + 'static,
    ) {
        self.inner.record_stream_mut(func);
    }
}

impl StreamPipelineCommand<'_, RayTracingPipeline> {
    /// Stream equivalent of [`PipelineCommand::<RayTracingPipeline>::record_cmd`].
    ///
    /// Unlike graph commands, stream callbacks must be reusable and therefore implement
    /// `Fn + Send + Sync + 'static`.
    pub fn record_cmd(
        mut self,
        func: impl for<'r> Fn(RayTracingCommandRef<'r>) + Send + Sync + 'static,
    ) -> Self {
        self.record_cmd_mut(func);
        self
    }

    /// Mutable-borrow stream equivalent of [`PipelineCommand::<RayTracingPipeline>::record_cmd`].
    ///
    /// Unlike graph commands, stream callbacks must be reusable and therefore implement
    /// `Fn + Send + Sync + 'static`.
    pub fn record_cmd_mut(
        &mut self,
        func: impl for<'r> Fn(RayTracingCommandRef<'r>) + Send + Sync + 'static,
    ) {
        self.inner.record_stream_mut(func);
    }
}

impl CommandStreamMut {
    /// Declares a typed argument required by this command stream.
    pub fn arg<I>(&mut self, info: I) -> I::Arg
    where
        I: StreamArgInfo,
    {
        info.bind_stream_arg(self)
    }

    /// Stream equivalent of [`Graph::begin_cmd`].
    pub fn begin_cmd(&mut self) -> StreamCommand<'_> {
        StreamCommand {
            inner: self.graph.begin_cmd(),
        }
    }

    /// Stream equivalent of [`Graph::bind_resource`].
    pub fn bind_resource<R>(&mut self, resource: R) -> R::Node
    where
        R: Resource,
    {
        self.graph.bind_resource(resource)
    }

    /// Stream equivalent of [`Graph::resource`].
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: StreamResourceNode,
    {
        self.graph.resource(resource_node)
    }

    /// Stream equivalent of [`Graph::blit_image`].
    pub fn blit_image(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyImageNode>,
        filter: vk::Filter,
    ) -> &mut Self {
        self.graph.blit_image(src, dst, filter);
        self
    }

    /// Deprecated stream equivalent of explicit-region blitting.
    #[doc(hidden)]
    #[deprecated(note = "use Command::blit_image for explicit regions")]
    pub fn blit_image_region(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyImageNode>,
        filter: vk::Filter,
        regions: impl AsRef<[vk::ImageBlit]> + 'static + Send,
    ) -> &mut Self {
        self.graph
            .begin_cmd()
            .debug_name("blit image")
            .blit_image(src, dst, filter, regions)
            .end_cmd();
        self
    }

    /// Stream equivalent of [`Graph::clear_color_image`].
    pub fn clear_color_image(
        &mut self,
        image: impl Into<AnyImageNode>,
        color: impl Into<ClearColorValue>,
    ) -> &mut Self {
        self.graph.clear_color_image(image, color);
        self
    }

    /// Stream equivalent of [`Graph::clear_depth_stencil_image`].
    pub fn clear_depth_stencil_image(
        &mut self,
        image: impl Into<AnyImageNode>,
        depth: f32,
        stencil: u32,
    ) -> &mut Self {
        self.graph.clear_depth_stencil_image(image, depth, stencil);
        self
    }

    /// Stream equivalent of [`Graph::copy_buffer`].
    pub fn copy_buffer(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyBufferNode>,
    ) -> &mut Self {
        self.graph.copy_buffer(src, dst);
        self
    }

    /// Deprecated stream equivalent of explicit-region buffer copies.
    #[doc(hidden)]
    #[deprecated(note = "use Command::copy_buffer for explicit regions")]
    pub fn copy_buffer_region(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyBufferNode>,
        regions: impl AsRef<[vk::BufferCopy]> + 'static + Send,
    ) -> &mut Self {
        self.graph
            .begin_cmd()
            .debug_name("copy buffer")
            .copy_buffer(src, dst, regions)
            .end_cmd();
        self
    }

    /// Stream equivalent of [`Graph::copy_buffer_to_image`].
    pub fn copy_buffer_to_image(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyImageNode>,
    ) -> &mut Self {
        self.graph.copy_buffer_to_image(src, dst);
        self
    }

    /// Deprecated stream equivalent of explicit-region buffer-to-image copies.
    #[doc(hidden)]
    #[deprecated(note = "use Command::copy_buffer_to_image for explicit regions")]
    pub fn copy_buffer_to_image_region(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyImageNode>,
        regions: impl AsRef<[vk::BufferImageCopy]> + 'static + Send,
    ) -> &mut Self {
        self.graph
            .begin_cmd()
            .debug_name("copy buffer to image")
            .copy_buffer_to_image(src, dst, regions)
            .end_cmd();
        self
    }

    /// Stream equivalent of [`Graph::copy_image`].
    pub fn copy_image(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyImageNode>,
    ) -> &mut Self {
        self.graph.copy_image(src, dst);
        self
    }

    /// Deprecated stream equivalent of explicit-region image copies.
    #[doc(hidden)]
    #[deprecated(note = "use Command::copy_image for explicit regions")]
    pub fn copy_image_region(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyImageNode>,
        regions: impl AsRef<[vk::ImageCopy]> + 'static + Send,
    ) -> &mut Self {
        self.graph
            .begin_cmd()
            .debug_name("copy image")
            .copy_image(src, dst, regions)
            .end_cmd();
        self
    }

    /// Stream equivalent of [`Graph::copy_image_to_buffer`].
    pub fn copy_image_to_buffer(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyBufferNode>,
    ) -> &mut Self {
        self.graph.copy_image_to_buffer(src, dst);
        self
    }

    /// Deprecated stream equivalent of explicit-region image-to-buffer copies.
    #[doc(hidden)]
    #[deprecated(note = "use Command::copy_image_to_buffer for explicit regions")]
    pub fn copy_image_to_buffer_region(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyBufferNode>,
        regions: impl AsRef<[vk::BufferImageCopy]> + 'static + Send,
    ) -> &mut Self {
        self.graph
            .begin_cmd()
            .debug_name("copy image to buffer")
            .copy_image_to_buffer(src, dst, regions)
            .end_cmd();
        self
    }

    /// Stream equivalent of [`Graph::fill_buffer`].
    pub fn fill_buffer(
        &mut self,
        buffer: impl Into<AnyBufferNode>,
        region: Range<vk::DeviceSize>,
        data: u32,
    ) -> &mut Self {
        self.graph.fill_buffer(buffer, region, data);
        self
    }

    /// Stream equivalent of [`Graph::update_buffer`].
    pub fn update_buffer(
        &mut self,
        buffer: impl Into<AnyBufferNode>,
        offset: vk::DeviceSize,
        data: impl AsRef<[u8]> + 'static + Send,
    ) -> &mut Self {
        self.graph.update_buffer(buffer, offset, data);
        self
    }

    fn push_arg(&mut self, data: StreamArgData) -> usize {
        let index = self.args.len();
        self.args.push(data);
        index
    }

    fn bind_arg_resource(&mut self, data: StreamArgData) -> usize {
        let resource = match data {
            StreamArgData::AccelerationStructure(info) => {
                AnyResource::AccelerationStructureArg(info)
            }
            StreamArgData::Buffer(info) => AnyResource::BufferArg(info),
            StreamArgData::Image(info) => AnyResource::ImageArg(info),
        };

        self.graph.bind_stream_arg_resource(resource)
    }
}

impl CommandStream<()> {
    /// Finalizes a reusable command stream without preparing optimizations.
    ///
    /// The returned draft can be inserted as an unprepared stream with [`CommandStreamDraft::into_stream`]
    /// or prepared later with [`CommandStreamDraft::prepare`].
    pub fn finalize<A>(build: impl FnOnce(&mut CommandStreamMut) -> A) -> CommandStreamDraft<A> {
        let mut stream = CommandStreamMut {
            arg_nodes: Vec::new(),
            args: Vec::new(),
            graph: Graph::new(),
            #[cfg(feature = "checked")]
            stream_id: CommandStreamId::next(),
        };
        let args = build(&mut stream);

        #[cfg(feature = "checked")]
        let graph_id = stream.graph.graph_id();

        let submission = stream.graph.finalize();
        submission.assert_reusable_commands();

        CommandStreamDraft {
            args,
            inner: CommandStreamInner {
                arg_nodes: stream.arg_nodes.into_boxed_slice(),
                args: stream.args.into_boxed_slice(),
                prepared: false,
                submission: Mutex::new(submission),

                #[cfg(feature = "checked")]
                stream_id: stream.stream_id,

                #[cfg(feature = "checked")]
                graph_id,
            },
        }
    }

    /// Finalizes and prepares a reusable command stream.
    ///
    /// Prepared streams do more work up front so repeated insertions can reuse prepared scheduling
    /// and static recording resources.
    pub fn prepare<P, A>(
        pool: &mut P,
        build: impl FnOnce(&mut CommandStreamMut) -> A,
    ) -> Result<CommandStream<A>, DriverError>
    where
        P: SubmissionPool,
    {
        Self::finalize(build).prepare(pool)
    }
}

impl<A> CommandStreamDraft<A> {
    /// Converts this draft into a command stream without preparing optimizations.
    ///
    /// Unprepared streams avoid preparation cost until insertion, but they do not cache the prepared
    /// schedule or static recording resources.
    pub fn into_stream(self) -> CommandStream<A> {
        CommandStream {
            args: self.args,
            inner: Arc::new(self.inner),
        }
    }

    /// Prepares this stream by optimizing its finalized graph and leasing static recording
    /// resources for the prepared schedule.
    ///
    /// This is most useful when the same stream is inserted many times with different arguments.
    pub fn prepare<P>(mut self, pool: &mut P) -> Result<CommandStream<A>, DriverError>
    where
        P: SubmissionPool,
    {
        self.inner
            .submission
            .get_mut()
            .expect("poisoned command stream submission")
            .prepare_command_stream(pool)?;
        self.inner.prepared = true;

        Ok(self.into_stream())
    }
}

/// An in-progress invocation of a [`CommandStream`] into a [`Graph`].
///
/// Bind every declared stream argument before calling [`CommandStreamRun::finish`].
/// Distinct stream arguments must bind to distinct parent-graph nodes.
///
/// ```no_run
/// # use ash::vk;
/// # use vk_graph::{Graph, driver::image::ImageInfo, node::ImageNode, stream::CommandStream};
/// # let image: ImageNode = todo!();
/// let stream = CommandStream::finalize(|stream| {
///     stream.arg(ImageInfo::image_2d(
///         32,
///         32,
///         vk::Format::R8G8B8A8_UNORM,
///         vk::ImageUsageFlags::TRANSFER_DST,
///     ))
/// })
/// .into_stream();
///
/// let mut graph = Graph::new();
/// graph
///     .insert_cmd_stream(&stream)
///     .with_arg(stream.args, image)
///     .finish();
/// ```
pub struct CommandStreamRun<'a, A> {
    pub(crate) bindings: Vec<Option<usize>>,
    pub(crate) graph: &'a mut Graph,
    pub(crate) stream: &'a CommandStream<A>,
}

impl<'a, A> CommandStreamRun<'a, A> {
    /// Sets a stream argument to a graph node for this invocation.
    ///
    /// The same argument may be rebound, but distinct arguments cannot bind to the same parent
    /// node. Stream scheduling and resource ownership tracking use node identity.
    ///
    /// # Panics
    ///
    /// Panics if another argument is already bound to `node`.
    pub fn with_arg<T, N>(mut self, arg: StreamArg<T>, node: N) -> Self
    where
        N: StreamArgBindable<T>,
    {
        #[cfg(feature = "checked")]
        assert!(
            arg.stream_id == self.stream.inner.stream_id,
            "argument belongs to a different command stream"
        );
        node.assert_parent_node();
        self.graph.assert_node_owner(&node);
        let node_idx = node.index();
        assert!(
            self.bindings
                .iter()
                .enumerate()
                .all(|(arg_idx, binding)| arg_idx == arg.arg_index || *binding != Some(node_idx)),
            "distinct command stream arguments cannot bind to the same parent graph node"
        );
        self.bindings[arg.arg_index] = Some(node_idx);
        self
    }

    /// Finishes this stream invocation and returns to the parent graph.
    pub fn finish(self) -> &'a mut Graph {
        #[cfg(feature = "checked")]
        assert!(
            self.bindings.iter().all(Option::is_some),
            "missing command stream argument"
        );

        self.graph
            .append_command_stream(self.stream, &self.bindings);
        self.graph
    }
}

impl Graph {
    /// Inserts a command stream into this graph.
    ///
    /// Prepared streams reduce repeated preparation work, but insertion still has argument binding,
    /// dependency reconciliation, scheduling, and recording costs.
    pub fn insert_cmd_stream<'a, A>(
        &'a mut self,
        stream: &'a CommandStream<A>,
    ) -> CommandStreamRun<'a, A> {
        CommandStreamRun {
            bindings: vec![None; stream.inner.args.len()],
            graph: self,
            stream,
        }
    }
}

impl Graph {
    pub(crate) fn append_command_stream<A>(
        &mut self,
        stream: &CommandStream<A>,
        bindings: &[Option<usize>],
    ) {
        if stream.inner.prepared {
            self.append_prepared_command_stream(stream, bindings);
        } else {
            self.append_unprepared_command_stream(stream, bindings);
        }
    }

    fn append_unprepared_command_stream<A>(
        &mut self,
        stream: &CommandStream<A>,
        bindings: &[Option<usize>],
    ) {
        let submission = stream
            .inner
            .submission
            .lock()
            .expect("poisoned command stream submission");
        let stream_graph = submission.graph();
        let mut arg_by_node = HashMap::new();

        for (arg_idx, &node_idx) in stream.inner.arg_nodes.iter().enumerate() {
            arg_by_node.insert(node_idx, arg_idx);
        }

        let mut node_map = Vec::with_capacity(stream_graph.resources.len());
        for (node_idx, resource) in stream_graph.resources.iter().enumerate() {
            if let Some(&arg_idx) = arg_by_node.get(&node_idx) {
                node_map.push(bindings[arg_idx].expect("missing command stream argument"));
            } else {
                node_map.push(self.resources.bind(resource.clone()));
            }
        }

        for cmd in &stream_graph.cmds {
            let mut cmd = cmd.clone();
            cmd.remap_nodes(&node_map);

            #[cfg(feature = "checked")]
            for exec in &mut cmd.execs {
                exec.stream_graph_id = Some(stream.inner.graph_id);
            }

            self.cmds.push(cmd);
        }
    }

    fn append_prepared_command_stream<A>(
        &mut self,
        stream: &CommandStream<A>,
        bindings: &[Option<usize>],
    ) {
        let stream_scope_id = next_stream_scope_id();
        let submission = stream
            .inner
            .submission
            .lock()
            .expect("poisoned command stream submission");
        let stream_graph = submission.graph();
        let mut arg_by_node = HashMap::new();

        for (arg_idx, &node_idx) in stream.inner.arg_nodes.iter().enumerate() {
            arg_by_node.insert(node_idx, arg_idx);
        }

        let mut cmd = self.begin_cmd().debug_name("command stream");
        cmd.set_stream_scope_id(stream_scope_id);

        for stream_cmd in &stream_graph.cmds {
            for (node_idx, accesses) in stream_cmd
                .execs
                .iter()
                .flat_map(|exec| exec.accesses.iter())
            {
                let Some(&arg_idx) = arg_by_node.get(&node_idx) else {
                    continue;
                };
                let parent_node_idx = bindings[arg_idx].expect("missing command stream argument");

                for access in accesses {
                    cmd.push_subresource_access_index(
                        parent_node_idx,
                        access.subresource,
                        access.access,
                    );
                }
            }
        }

        drop(submission);

        let stream = Arc::clone(&stream.inner);
        let bindings = bindings.to_vec();
        cmd.record_stream(move |cmd| {
            let submission = stream
                .submission
                .lock()
                .expect("poisoned command stream submission");
            let stream_graph = submission.graph();
            let mut arg_by_node = HashMap::new();

            for (arg_idx, &node_idx) in stream.arg_nodes.iter().enumerate() {
                arg_by_node.insert(node_idx, arg_idx);
            }

            let resources = stream_graph
                .resources
                .iter()
                .enumerate()
                .map(|(node_idx, resource)| {
                    if let Some(&arg_idx) = arg_by_node.get(&node_idx) {
                        cmd.clone_resource_at(
                            bindings[arg_idx].expect("missing command stream argument"),
                        )
                    } else {
                        resource.clone()
                    }
                })
                .collect();
            drop(submission);

            stream
                .submission
                .lock()
                .expect("poisoned command stream submission")
                .record_prepared_command_stream(&cmd, ResourceMap::from_resources(resources))
                .expect("unable to record command stream");
        });
    }
}

/// Information that can declare a typed [`CommandStream`] argument.
#[allow(private_bounds)]
#[doc(hidden)]
pub trait StreamArgInfo: stream_private::StreamArgInfoSealed {
    /// The typed argument handle returned for this info.
    type Arg;

    #[doc(hidden)]
    fn bind_stream_arg(self, stream: &mut CommandStreamMut) -> Self::Arg;
}

/// A graph node that can be supplied for a [`StreamArg`].
#[allow(private_bounds)]
#[doc(hidden)]
pub trait StreamArgBindable<T>: stream_private::StreamArgBindableSealed<T> + Node {
    #[doc(hidden)]
    fn assert_parent_node(&self);
}

/// A graph node that can be borrowed while building a [`CommandStream`].
#[allow(private_bounds)]
#[doc(hidden)]
pub trait StreamResourceNode: stream_private::StreamResourceNodeSealed + Node {}

mod stream_private {
    pub trait StreamArgInfoSealed {}

    pub trait StreamArgBindableSealed<T> {}

    pub trait StreamResourceNodeSealed {}

    pub trait StreamPipelineSealed {}
}

macro_rules! stream_arg_info {
    ($info:ty, $builder:ty, $variant:ident, $arg:ty) => {
        impl stream_private::StreamArgInfoSealed for $info {}

        impl StreamArgInfo for $info {
            type Arg = $arg;

            fn bind_stream_arg(self, stream: &mut CommandStreamMut) -> Self::Arg {
                let data = StreamArgData::$variant(self);
                let arg_index = stream.push_arg(data);
                let node_index = stream.bind_arg_resource(data);
                stream.arg_nodes.push(node_index);
                StreamArg::new(
                    arg_index,
                    node_index,
                    #[cfg(feature = "checked")]
                    stream.stream_id,
                    #[cfg(feature = "checked")]
                    stream.graph.graph_id(),
                )
            }
        }

        impl stream_private::StreamArgInfoSealed for $builder {}

        impl StreamArgInfo for $builder {
            type Arg = $arg;

            fn bind_stream_arg(self, stream: &mut CommandStreamMut) -> Self::Arg {
                self.build().bind_stream_arg(stream)
            }
        }
    };
}

stream_arg_info!(
    AccelerationStructureInfo,
    AccelerationStructureInfoBuilder,
    AccelerationStructure,
    AccelerationStructureArg
);
stream_arg_info!(BufferInfo, BufferInfoBuilder, Buffer, BufferArg);
stream_arg_info!(ImageInfo, ImageInfoBuilder, Image, ImageArg);

macro_rules! stream_arg_bindable {
    ($resource:ty => $($node:ty),+ $(,)?) => {
        $(
            impl stream_private::StreamArgBindableSealed<$resource> for $node {}

            impl StreamArgBindable<$resource> for $node {
                fn assert_parent_node(&self) {}
            }
        )+
    };
}

stream_arg_bindable!(
    AccelerationStructure => AccelerationStructureNode,
    AccelerationStructureLeaseNode,
);
stream_arg_bindable!(Buffer => BufferNode, BufferLeaseNode);
stream_arg_bindable!(Image => ImageNode, ImageLeaseNode, SwapchainImageNode);

impl stream_private::StreamArgBindableSealed<AccelerationStructure>
    for AnyAccelerationStructureNode
{
}

impl StreamArgBindable<AccelerationStructure> for AnyAccelerationStructureNode {
    fn assert_parent_node(&self) {
        assert!(
            !matches!(self, Self::Arg(_)),
            "stream argument cannot be supplied as a parent graph node"
        );
    }
}

impl stream_private::StreamArgBindableSealed<Buffer> for AnyBufferNode {}

impl StreamArgBindable<Buffer> for AnyBufferNode {
    fn assert_parent_node(&self) {
        assert!(
            !matches!(self, Self::Arg(_)),
            "stream argument cannot be supplied as a parent graph node"
        );
    }
}

impl stream_private::StreamArgBindableSealed<Image> for AnyImageNode {}

impl StreamArgBindable<Image> for AnyImageNode {
    fn assert_parent_node(&self) {
        assert!(
            !matches!(self, Self::Arg(_)),
            "stream argument cannot be supplied as a parent graph node"
        );
    }
}

macro_rules! stream_resource_node {
    ($($node:ty),+ $(,)?) => {
        $(
            impl stream_private::StreamResourceNodeSealed for $node {}
            impl StreamResourceNode for $node {}
        )+
    };
}

stream_resource_node!(
    AccelerationStructureNode,
    AccelerationStructureLeaseNode,
    BufferNode,
    BufferLeaseNode,
    ImageNode,
    ImageLeaseNode,
    SwapchainImageNode,
);

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        driver::{
            buffer::BufferInfo,
            descriptor_set::{DescriptorPool, DescriptorPoolInfo},
            render_pass::{RenderPass, RenderPassInfo},
        },
        pool::Pool,
    };

    struct NoopPool;

    impl Pool<DescriptorPoolInfo, DescriptorPool> for NoopPool {
        fn resource(
            &mut self,
            _: DescriptorPoolInfo,
        ) -> Result<crate::pool::Lease<DescriptorPool>, DriverError> {
            unreachable!()
        }
    }

    impl Pool<RenderPassInfo, RenderPass> for NoopPool {
        fn resource(
            &mut self,
            _: RenderPassInfo,
        ) -> Result<crate::pool::Lease<RenderPass>, DriverError> {
            unreachable!()
        }
    }

    fn bind_test_buffer(graph: &mut Graph) -> BufferNode {
        let index = graph.bind_stream_arg_resource(AnyResource::BufferArg(BufferInfo::device_mem(
            4,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )));
        BufferNode::new(
            index,
            #[cfg(feature = "checked")]
            graph.graph_id(),
        )
    }

    fn two_buffer_arg_stream() -> CommandStreamDraft<(BufferArg, BufferArg)> {
        CommandStream::finalize(|stream| {
            let first = stream.arg(BufferInfo::device_mem(
                4,
                vk::BufferUsageFlags::TRANSFER_SRC,
            ));
            let second = stream.arg(BufferInfo::device_mem(
                4,
                vk::BufferUsageFlags::TRANSFER_DST,
            ));

            stream
                .begin_cmd()
                .resource_access(first, vk_sync::AccessType::TransferRead)
                .record_cmd(|_| {});
            stream
                .begin_cmd()
                .resource_access(second, vk_sync::AccessType::TransferWrite)
                .record_cmd(|_| {});
            stream
                .begin_cmd()
                .resource_access(first, vk_sync::AccessType::TransferRead)
                .record_cmd(|_| {});

            (first, second)
        })
    }

    #[test]
    fn empty_stream_can_be_inserted() {
        let stream = CommandStream::finalize(|_| {}).into_stream();
        let mut graph = Graph::new();

        graph.insert_cmd_stream(&stream).finish();
    }

    #[test]
    fn reusable_callback_can_prepare_stream() {
        let stream = CommandStream::finalize(|stream| {
            stream.begin_cmd().record_cmd(|_| {});
        })
        .into_stream();
        let mut graph = Graph::new();

        graph.insert_cmd_stream(&stream).finish();
        assert_eq!(graph.cmds.len(), 1);
    }

    #[test]
    fn graph_copy_wrapper_can_prepare_stream() {
        let _stream = CommandStream::finalize(|stream| {
            let src = stream.arg(BufferInfo::device_mem(
                4,
                vk::BufferUsageFlags::TRANSFER_SRC,
            ));
            let dst = stream.arg(BufferInfo::device_mem(
                4,
                vk::BufferUsageFlags::TRANSFER_DST,
            ));

            stream.graph.copy_buffer(src, dst);
        })
        .into_stream();
    }

    #[test]
    fn reusable_callback_can_prepare_optimized_stream() {
        let mut pool = NoopPool;
        let stream = CommandStream::prepare(&mut pool, |stream| {
            stream.begin_cmd().record_cmd(|_| {});
        })
        .expect("prepare stream");
        let mut graph = Graph::new();

        graph.insert_cmd_stream(&stream).finish();
        assert_eq!(graph.cmds.len(), 1);
    }

    #[test]
    fn unprepared_stream_expands_commands() {
        let stream = CommandStream::finalize(|stream| {
            stream.begin_cmd().record_cmd(|_| {});
            stream.begin_cmd().record_cmd(|_| {});
        })
        .into_stream();
        let mut graph = Graph::new();

        graph.insert_cmd_stream(&stream).finish();
        assert_eq!(graph.cmds.len(), 2);
    }

    #[test]
    fn prepared_stream_is_opaque_by_default() {
        let mut pool = NoopPool;
        let stream = CommandStream::prepare(&mut pool, |stream| {
            stream.begin_cmd().record_cmd(|_| {});
            stream.begin_cmd().record_cmd(|_| {});
        })
        .expect("prepare stream");
        let mut graph = Graph::new();

        graph.insert_cmd_stream(&stream).finish();
        assert_eq!(graph.cmds.len(), 1);
    }

    #[test]
    #[should_panic(
        expected = "distinct command stream arguments cannot bind to the same parent graph node"
    )]
    fn unprepared_stream_rejects_aliased_arguments() {
        let stream = two_buffer_arg_stream().into_stream();
        let mut graph = Graph::new();
        let buffer = bind_test_buffer(&mut graph);

        graph
            .insert_cmd_stream(&stream)
            .with_arg(stream.args.0, buffer)
            .with_arg(stream.args.1, buffer)
            .finish();
    }

    #[test]
    #[should_panic(
        expected = "distinct command stream arguments cannot bind to the same parent graph node"
    )]
    fn prepared_stream_rejects_aliased_arguments() {
        let mut pool = NoopPool;
        let stream = two_buffer_arg_stream()
            .prepare(&mut pool)
            .expect("prepare stream");
        let mut graph = Graph::new();
        let buffer = bind_test_buffer(&mut graph);

        graph
            .insert_cmd_stream(&stream)
            .with_arg(stream.args.0, buffer)
            .with_arg(stream.args.1, buffer)
            .finish();
    }

    #[test]
    fn same_argument_can_be_rebound() {
        let stream = two_buffer_arg_stream().into_stream();
        let mut graph = Graph::new();
        let first = bind_test_buffer(&mut graph);
        let second = bind_test_buffer(&mut graph);

        graph
            .insert_cmd_stream(&stream)
            .with_arg(stream.args.0, first)
            .with_arg(stream.args.0, second)
            .with_arg(stream.args.1, first)
            .finish();
    }

    #[test]
    fn image_arg_can_use_info_based_helpers() {
        let stream = CommandStream::finalize(|stream| {
            let output = stream.arg(ImageInfo::image_2d(
                1,
                1,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::TRANSFER_DST,
            ));

            stream.clear_color_image(output, [0.0, 0.0, 0.0, 0.0]);

            output
        })
        .into_stream();

        assert_eq!(stream.inner.args.len(), 1);
    }

    #[test]
    #[should_panic(expected = "missing command stream argument")]
    fn missing_arg_panics_at_finish() {
        let stream = CommandStream::finalize(|stream| {
            stream.arg(ImageInfo::image_2d(
                1,
                1,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED,
            ));
        })
        .into_stream();
        let mut graph = Graph::new();

        graph.insert_cmd_stream(&stream).finish();
    }

    #[test]
    #[cfg(feature = "checked")]
    #[should_panic(expected = "argument belongs to a different command stream")]
    fn wrong_stream_arg_panics_at_with_arg() {
        let stream_a = CommandStream::finalize(|stream| {
            stream.arg(ImageInfo::image_2d(
                1,
                1,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED,
            ))
        })
        .into_stream();
        let stream_b = CommandStream::finalize(|stream| {
            stream.arg(ImageInfo::image_2d(
                1,
                1,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED,
            ))
        })
        .into_stream();
        let mut graph = Graph::new();

        graph
            .insert_cmd_stream(&stream_a)
            .with_arg(stream_b.args, AnyImageNode::from(stream_b.args))
            .finish();
    }

    #[test]
    #[should_panic(expected = "stream argument cannot be supplied as a parent graph node")]
    fn stream_arg_cannot_bind_as_parent_graph_node() {
        let stream = CommandStream::finalize(|stream| {
            stream.arg(ImageInfo::image_2d(
                1,
                1,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED,
            ))
        })
        .into_stream();
        let mut graph = Graph::new();

        graph
            .insert_cmd_stream(&stream)
            .with_arg(stream.args, AnyImageNode::from(stream.args))
            .finish();
    }

    #[test]
    #[should_panic(expected = "command stream contains a one-shot callback")]
    fn one_shot_callback_cannot_prepare_stream() {
        let _ = CommandStream::finalize(|stream| {
            stream.graph.begin_cmd().record_cmd(|_| {});
        });
    }
}
