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

use {
    crate::{
        AnyResource, Graph, Node, Resource, ResourceMap, ResourceNode,
        cmd::{
            AttachmentIndex, Binding, ClearColorValue, Command, CommandRef, ComputeCommandRef,
            GraphicsCommandRef, LoadOp, PipelineCommand, RayTracingCommandRef, ResourceAccess,
            StoreOp, Subresource, SubresourceRange, ViewInfo,
        },
        driver::{
            DriverError,
            accel_struct::{
                AccelerationStructure, AccelerationStructureInfo, AccelerationStructureInfoBuilder,
            },
            buffer::{Buffer, BufferInfo, BufferInfoBuilder},
            compute::ComputePipeline,
            descriptor_set::DescriptorSet,
            graphics::{DepthStencilInfo, GraphicsPipeline},
            image::{Image, ImageInfo, ImageInfoBuilder, ImageViewInfo},
            micromap::{Micromap, MicromapInfo, MicromapInfoBuilder},
            ray_tracing::RayTracingPipeline,
        },
        node::{
            AccelerationStructureLeaseNode, AccelerationStructureNode,
            AccelerationStructureSetNode, AnyAccelerationStructureNode, AnyBufferNode,
            AnyImageNode, AnyMicromapNode, BufferLeaseNode, BufferNode, ImageLeaseNode, ImageNode,
            ImageSetNode, MicromapLeaseNode, MicromapNode, SwapchainImageNode,
        },
        pool::SubmissionPool,
        private::NodeSealed,
        submission::{PreparedStreamRecording, Submission},
    },
    ash::vk,
    std::{
        any::Any,
        collections::HashMap,
        marker::PhantomData,
        ops::Range,
        sync::{
            Arc, Mutex,
            atomic::{AtomicU64, Ordering},
        },
    },
    vk_sync::AccessType,
};

#[cfg(feature = "checked")]
use {crate::GraphId, std::cell::RefCell};

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

/// A reusable command stream.
///
/// Prepared streams cache an optimized schedule and static recording resources to reduce repeated
/// CPU-side graph construction and preparation. Unprepared streams cost less to finalize, but each
/// insertion still reconciles arguments, dependencies, scheduling, and recording with the parent graph.
///
/// Inserting or concatenating many small streams has overhead. Profile release builds before relying
/// on extensive stream composition.
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

impl CommandStream<()> {
    /// Finalizes a reusable command stream without preparing optimizations.
    ///
    /// The returned draft can be inserted as an unprepared stream with [`CommandStreamDraft::into_stream`]
    /// or prepared later with [`CommandStreamDraft::prepare`].
    /// Definitions must contain only reusable callbacks and must not capture prepared stream
    /// invocations, even without `checked`. Capturing a prepared invocation in another stream is
    /// unsupported: outer replays would share descriptor recording slots and incorrectly map parent nodes.
    ///
    /// # Panics
    /// With `checked`, panics for one-shot callbacks or captured prepared stream invocations.
    pub fn finalize<A>(build: impl FnOnce(&mut CommandStreamMut) -> A) -> CommandStreamDraft<A> {
        let mut stream = CommandStreamMut {
            arg_nodes: Vec::new(),
            args: Vec::new(),
            graph: Graph::new(),
            value_count: 0,
            #[cfg(feature = "checked")]
            value_scope: next_stream_scope_id(),
            #[cfg(feature = "checked")]
            stream_id: CommandStreamId::next(),
        };
        let args = build(&mut stream);

        // Prepared callbacks capture invocation-owned slots and parent node indices. Replaying
        // them through an outer stream would share descriptors and require another node remap.
        #[cfg(feature = "checked")]
        assert!(
            stream
                .graph
                .cmds
                .iter()
                .all(|cmd| cmd.stream_scope_id.is_none()),
            "prepared command stream invocations cannot be captured in another command stream"
        );

        #[cfg(feature = "checked")]
        let graph_id = stream.graph.graph_id();

        let submission = stream.graph.finalize();
        #[cfg(feature = "checked")]
        submission.assert_reusable_commands();
        let mut node_args = vec![None; submission.graph().resources.len()];
        for (arg_idx, &node_idx) in stream.arg_nodes.iter().enumerate() {
            node_args[node_idx] = Some(arg_idx);
        }

        CommandStreamDraft {
            args,
            inner: CommandStreamInner {
                arg_nodes: stream.arg_nodes.into_boxed_slice(),
                args: stream.args.into_boxed_slice(),
                node_args: node_args.into_boxed_slice(),
                prepared: false,
                submission: Mutex::new(submission),
                recordings: Mutex::new(Vec::new()),
                value_count: stream.value_count,
                #[cfg(feature = "checked")]
                value_scope: stream.value_scope,

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
        let submission = self
            .inner
            .submission
            .get_mut()
            .expect("poisoned command stream submission");
        submission.prepare_command_stream(pool)?;
        self.inner.prepared = true;

        Ok(self.into_stream())
    }
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

#[derive(Debug)]
pub(crate) struct CommandStreamInner {
    pub(crate) arg_nodes: Box<[usize]>,
    pub(crate) args: Box<[StreamArgData]>,
    node_args: Box<[Option<usize>]>,
    pub(crate) prepared: bool,
    pub(crate) submission: Mutex<Submission>,
    recordings: Mutex<Vec<Arc<PreparedStreamRecording>>>,
    value_count: usize,
    #[cfg(feature = "checked")]
    value_scope: u64,

    #[cfg(feature = "checked")]
    pub(crate) stream_id: CommandStreamId,

    #[cfg(feature = "checked")]
    pub(crate) graph_id: GraphId,
}

/// A mutable command stream builder with graph-like methods.
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
    value_count: usize,
    #[cfg(feature = "checked")]
    value_scope: u64,
    #[cfg(feature = "checked")]
    pub(crate) stream_id: CommandStreamId,
}

impl CommandStreamMut {
    /// Declares a required constant copied into each invocation by [`CommandStreamRun::with_value`].
    /// Reusable callbacks read it with [`CommandRef::value`].
    pub fn add_value_arg<T: Copy + Send + Sync + 'static>(&mut self) -> StreamValueArg<T> {
        let arg = StreamValueArg {
            index: self.value_count,
            #[cfg(feature = "checked")]
            scope: self.value_scope,
            __: PhantomData,
        };
        self.value_count += 1;

        arg
    }

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

    fn bind_arg_resource(&mut self, data: StreamArgData) -> usize {
        let resource = match data {
            StreamArgData::AccelerationStructure(info) => {
                AnyResource::AccelerationStructureArg(info)
            }
            StreamArgData::Buffer(info) => AnyResource::BufferArg(info),
            StreamArgData::Image(info) => AnyResource::ImageArg(info),
            StreamArgData::Micromap(info) => AnyResource::MicromapArg(info),
        };

        self.graph.bind_stream_arg_resource(resource)
    }

    /// Stream equivalent of [`Graph::bind_resource`].
    pub fn bind_resource<R>(&mut self, resource: R) -> R::Node
    where
        R: Resource,
        R::Node: StreamResourceNode,
    {
        self.graph.bind_resource(resource)
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

    fn push_arg(&mut self, data: StreamArgData) -> usize {
        let index = self.args.len();
        self.args.push(data);

        index
    }

    /// Stream equivalent of [`Graph::resource`].
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: StreamResourceNode,
    {
        self.graph.resource(resource_node)
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
}

/// An in-progress invocation of a [`CommandStream`] into a [`Graph`].
///
/// Bind every declared stream argument before calling [`CommandStreamRun::finish`].
/// Distinct stream arguments must bind to distinct parent-graph nodes.
/// Arguments must belong to this stream, and resource nodes must belong to the parent graph
/// (not be stream argument nodes). These preconditions apply even without `checked`.
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
    values: Vec<Option<Arc<dyn Any + Send + Sync>>>,
}

impl<'a, A> CommandStreamRun<'a, A> {
    /// Finishes this stream invocation and returns to the parent graph.
    ///
    /// # Panics
    ///
    /// With `checked`, panics if any resource argument is missing. Without `checked`, missing values
    /// and resources needed during expansion or recording still cause panics when converted or
    /// accessed through safe APIs. All declared arguments must be supplied in either build.
    pub fn finish(self) -> &'a mut Graph {
        let values = Arc::new(StreamValues {
            #[cfg(feature = "checked")]
            scope: self.stream.inner.value_scope,
            values: self
                .values
                .into_iter()
                .map(|value| value.expect("missing command stream value argument"))
                .collect(),
        });

        #[cfg(feature = "checked")]
        assert!(
            self.bindings.iter().all(Option::is_some),
            "missing command stream argument"
        );

        self.graph
            .append_command_stream(self.stream, &self.bindings, values);
        self.graph
    }

    /// Sets a stream argument to a graph node for this invocation.
    ///
    /// The same argument may be rebound, but distinct arguments cannot bind to the same parent
    /// node. Stream scheduling and resource ownership tracking use node identity.
    ///
    /// # Panics
    ///
    /// With `checked`, panics if another argument is already bound to `node`.
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
        #[cfg(feature = "checked")]
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

    /// Binds a batch of resource arguments. With `checked`, validates uniqueness once per batch.
    ///
    /// Use separate batches for different resource types. The last binding for a repeated argument
    /// wins, including bindings from earlier calls. Distinct arguments must bind to distinct parent
    /// nodes after each batch, even without `checked`; a batch may swap previously bound nodes.
    /// Checked validation is O(n log n) in the total number of bound arguments and reuses thread-local
    /// scratch storage. Without `checked`, no collection, sorting, or alias scanning is performed.
    /// [`Self::with_arg`] retains its immediate per-binding validation in checked builds.
    ///
    /// # Panics
    /// With `checked`, panics if distinct arguments bind the same parent node after this batch,
    /// or if an argument belongs to another stream or a node belongs to another graph, as `with_arg` does.
    pub fn with_args<T, N>(mut self, bindings: impl IntoIterator<Item = (StreamArg<T>, N)>) -> Self
    where
        N: StreamArgBindable<T>,
    {
        for (arg, node) in bindings {
            #[cfg(feature = "checked")]
            assert!(
                arg.stream_id == self.stream.inner.stream_id,
                "argument belongs to a different command stream"
            );
            node.assert_parent_node();
            self.graph.assert_node_owner(&node);
            self.bindings[arg.arg_index] = Some(node.index());
        }

        #[cfg(feature = "checked")]
        {
            thread_local! {
                static BOUND_NODES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
            }

            BOUND_NODES.with_borrow_mut(|nodes| {
                nodes.clear();
                nodes.extend(self.bindings.iter().flatten().copied());
                nodes.sort_unstable();

                assert!(
                    nodes.windows(2).all(|pair| pair[0] != pair[1]),
                    "distinct command stream arguments cannot bind to the same parent graph node"
                );
            });
        }

        self
    }

    /// Copies a constant into this invocation. Rebinding the same argument replaces its value.
    /// The argument must belong to this stream, even without `checked`.
    ///
    /// # Panics
    /// With `checked`, panics for a handle from another stream.
    pub fn with_value<T: Copy + Send + Sync + 'static>(
        mut self,
        arg: StreamValueArg<T>,
        value: T,
    ) -> Self {
        #[cfg(feature = "checked")]
        assert_eq!(
            arg.scope, self.stream.inner.value_scope,
            "value argument belongs to a different command stream"
        );
        self.values[arg.index] = Some(Arc::new(value));
        self
    }
}

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

/// A stream argument for a micromap.
pub type MicromapArg = StreamArg<Micromap>;

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

    #[inline]
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

    #[inline]
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

impl NodeSealed for StreamArg<Micromap> {
    fn borrow(self, resources: &[AnyResource]) -> &<Self as Node>::Resource {
        resources[self.index].expect_micromap()
    }

    fn borrow_at(self, resources: &[AnyResource], index: usize) -> &<Self as Node>::Resource {
        resources[index].expect_micromap()
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

impl Node for StreamArg<Micromap> {
    type Resource = Micromap;
    type SyncInfo = crate::driver::micromap::MicromapSyncInfo;

    fn index(&self) -> usize {
        self.index
    }
}

impl<T> Clone for StreamArg<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for StreamArg<T> {}

/// A graph node that can be supplied for a [`StreamArg`].
#[allow(private_bounds)]
#[doc(hidden)]
pub trait StreamArgBindable<T>: stream_private::StreamArgBindableSealed<T> + Node {
    #[doc(hidden)]
    fn assert_parent_node(&self);
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum StreamArgData {
    AccelerationStructure(AccelerationStructureInfo),
    Buffer(BufferInfo),
    Image(ImageInfo),
    Micromap(MicromapInfo),
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

#[allow(private_bounds)]
impl<'a> StreamCommand<'a> {
    /// Stream equivalent of [`Command::bind_pipeline`].
    pub fn bind_pipeline<P>(self, pipeline: P) -> P::Command
    where
        P: StreamPipeline<'a>,
    {
        pipeline.bind_stream_cmd(self)
    }

    /// Stream equivalent of [`Command::bind_resource`].
    pub fn bind_resource<R>(&mut self, resource: R) -> R::Node
    where
        R: Resource,
        R::Node: StreamResourceNode,
    {
        self.inner.bind_resource(resource)
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
    pub fn resource_access<N>(mut self, resource_node: N, access: N::Access) -> Self
    where
        N: ResourceAccess,
    {
        self.inner.set_resource_access(resource_node, access);
        self
    }

    /// Mutable-borrow stream equivalent of [`Command::resource_access`].
    pub fn set_resource_access<N>(&mut self, resource_node: N, access: N::Access)
    where
        N: ResourceAccess,
    {
        self.inner.set_resource_access(resource_node, access);
    }
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

#[allow(private_bounds)]
impl<'a, T> StreamPipelineCommand<'a, T> {
    /// Captures and retains an explicitly populated descriptor set for every invocation.
    ///
    /// This does not declare resource accesses; declare them separately with `resource_access`.
    /// The captured set must not be updated while any invocation may use it.
    /// `index` must equal the descriptor set's index, even without `checked`.
    ///
    /// # Panics
    /// With `checked`, panics if `index` differs from the set's index. Pipeline layout
    /// compatibility is validated by the underlying descriptor binding API.
    pub fn bind_descriptor_set(mut self, index: u32, descriptor_set: &DescriptorSet) -> Self {
        #[cfg(feature = "checked")]
        assert_eq!(
            index,
            descriptor_set.info().set,
            "descriptor set index mismatch"
        );
        #[cfg(not(feature = "checked"))]
        let _ = index;
        self.inner.set_descriptor_set(descriptor_set);
        self
    }

    /// Stream equivalent of [`PipelineCommand::bind_resource`].
    pub fn bind_resource<R>(&mut self, resource: R) -> R::Node
    where
        R: Resource,
        R::Node: StreamResourceNode,
    {
        self.inner.bind_resource(resource)
    }

    /// Stream equivalent of [`PipelineCommand::resource_access`].
    pub fn resource_access<N>(mut self, resource_node: N, access: N::Access) -> Self
    where
        N: ResourceAccess,
    {
        self.inner.set_resource_access(resource_node, access);
        self
    }

    /// Mutable-borrow stream equivalent of [`PipelineCommand::resource_access`].
    pub fn set_resource_access<N>(&mut self, resource_node: N, access: N::Access) -> &mut Self
    where
        N: ResourceAccess,
    {
        self.inner.set_resource_access(resource_node, access);
        self
    }

    /// Stream equivalent of [`PipelineCommand::shader_resource_access`].
    ///
    /// Buffer and image arguments can be converted to their corresponding `Any*Node` type.
    /// Prepared invocations retain separate descriptor sets until their parent graph is dropped.
    pub fn shader_resource_access<N>(
        mut self,
        binding: impl Into<Binding>,
        resource_node: N,
        access: AccessType,
    ) -> Self
    where
        N: Node + Subresource,
        N::Info: Copy,
        SubresourceRange: From<N::Info>,
        ViewInfo: From<N::Info>,
    {
        self.inner
            .set_shader_resource_access(binding, resource_node, access);
        self
    }

    /// Stream equivalent of [`PipelineCommand::shader_subresource_access`], including image views.
    pub fn shader_subresource_access<N>(
        mut self,
        binding: impl Into<Binding>,
        resource_node: N,
        subresource: impl Into<N::Info>,
        access: AccessType,
    ) -> Self
    where
        N: Node + Subresource,
        N::Info: Copy,
        SubresourceRange: From<N::Info>,
        ViewInfo: From<N::Info>,
    {
        self.inner
            .set_shader_subresource_access(binding, resource_node, subresource, access);
        self
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
            values: vec![None; stream.inner.value_count],
        }
    }
}

impl Graph {
    pub(crate) fn append_command_stream<A>(
        &mut self,
        stream: &CommandStream<A>,
        bindings: &[Option<usize>],
        values: Arc<StreamValues>,
    ) {
        if stream.inner.prepared {
            self.append_prepared_command_stream(stream, bindings, values);
        } else {
            self.append_unprepared_command_stream(stream, bindings, values);
        }
    }

    fn append_prepared_command_stream<A>(
        &mut self,
        stream: &CommandStream<A>,
        bindings: &[Option<usize>],
        values: Arc<StreamValues>,
    ) {
        let stream_scope_id = next_stream_scope_id();
        let submission = stream
            .inner
            .submission
            .lock()
            .expect("poisoned command stream submission");
        let stream_graph = submission.graph();

        #[cfg(feature = "checked")]
        {
            self.prepared_stream_acceleration_structure_accesses.extend(
                stream_graph
                    .prepared_stream_acceleration_structure_accesses
                    .iter()
                    .copied(),
            );
            self.prepared_stream_image_accesses
                .extend(stream_graph.prepared_stream_image_accesses.iter().copied());
            self.prepared_stream_micromap_accesses.extend(
                stream_graph
                    .prepared_stream_micromap_accesses
                    .iter()
                    .copied(),
            );
        }

        let resource_set_map = stream_graph
            .resource_sets
            .iter()
            .map(|resource_set| self.resource_sets.bind(resource_set))
            .collect::<Vec<_>>();

        #[cfg(feature = "checked")]
        for (node_idx, _) in stream_graph
            .cmds
            .iter()
            .flat_map(|cmd| &cmd.execs)
            .flat_map(|exec| exec.accesses.iter())
        {
            if let Some(acceleration_structure) = stream_graph.resources[node_idx].as_accel_struct()
            {
                self.prepared_stream_acceleration_structure_accesses.insert(
                    crate::resource::PhysicalAccelerationStructureId::of(acceleration_structure),
                );
            }

            if let Some(image) = stream_graph.resources[node_idx].as_image() {
                self.prepared_stream_image_accesses
                    .insert(crate::resource::PhysicalImageId::of(image));
            }

            if let Some(micromap) = stream_graph.resources[node_idx].as_micromap() {
                self.prepared_stream_micromap_accesses
                    .insert(crate::resource::PhysicalMicromapId::of(micromap));
            }
        }

        let mut cmd = self.begin_cmd().debug_name("command stream");
        cmd.set_stream_scope_id(stream_scope_id);

        for stream_cmd in &stream_graph.cmds {
            for (node_idx, accesses) in stream_cmd
                .execs
                .iter()
                .flat_map(|exec| exec.accesses.iter())
            {
                let Some(arg_idx) = stream.inner.node_args[node_idx] else {
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

        for access in stream_graph
            .cmds
            .iter()
            .flat_map(|cmd| &cmd.execs)
            .flat_map(|exec| &exec.resource_set_accesses)
        {
            cmd.push_resource_set_access(
                resource_set_map[access.resource_set_idx.as_usize()],
                access.access_type,
            );
        }

        drop(submission);

        // A later invocation must not update descriptors referenced by earlier GPU work.
        // The callback keeps its slot leased for the lifetime of the parent submission.
        let recording = {
            let mut recordings = stream
                .inner
                .recordings
                .lock()
                .expect("poisoned stream recordings");
            if let Some(recording) = recordings.iter().find(|slot| Arc::strong_count(slot) == 1) {
                Arc::clone(recording)
            } else {
                let recording = Arc::new(
                    stream
                        .inner
                        .submission
                        .lock()
                        .expect("poisoned command stream submission")
                        .take_prepared_stream_recording(),
                );
                recordings.push(Arc::clone(&recording));

                recording
            }
        };
        let stream = Arc::clone(&stream.inner);
        let bindings = bindings.to_vec();

        cmd.record_stream(move |cmd| {
            let submission = stream
                .submission
                .lock()
                .expect("poisoned command stream submission");
            let stream_graph = submission.graph();
            let resources = stream_graph
                .resources
                .iter()
                .enumerate()
                .map(|(node_idx, resource)| {
                    if let Some(arg_idx) = stream.node_args[node_idx] {
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
                .record_prepared_command_stream(
                    &cmd,
                    ResourceMap::from_resources(resources),
                    &recording,
                    &values,
                )
                .expect("unable to record command stream");
        });
    }

    fn append_unprepared_command_stream<A>(
        &mut self,
        stream: &CommandStream<A>,
        bindings: &[Option<usize>],
        values: Arc<StreamValues>,
    ) {
        let submission = stream
            .inner
            .submission
            .lock()
            .expect("poisoned command stream submission");
        let stream_graph = submission.graph();
        let mut arg_by_node = HashMap::new();

        #[cfg(feature = "checked")]
        {
            self.prepared_stream_acceleration_structure_accesses.extend(
                stream_graph
                    .prepared_stream_acceleration_structure_accesses
                    .iter()
                    .copied(),
            );
            self.prepared_stream_image_accesses
                .extend(stream_graph.prepared_stream_image_accesses.iter().copied());
            self.prepared_stream_micromap_accesses.extend(
                stream_graph
                    .prepared_stream_micromap_accesses
                    .iter()
                    .copied(),
            );
        }

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

        let resource_set_map = stream_graph
            .resource_sets
            .iter()
            .map(|resource_set| self.resource_sets.bind(resource_set))
            .collect::<Vec<_>>();

        for cmd in &stream_graph.cmds {
            let mut cmd = cmd.clone();
            cmd.remap_nodes(&node_map);
            cmd.remap_resource_sets(&resource_set_map);

            for exec in &mut cmd.execs {
                exec.stream_values
                    .get_or_insert_with(|| Arc::clone(&values));
            }

            #[cfg(feature = "checked")]
            for exec in &mut cmd.execs {
                exec.stream_graph_id.get_or_insert(stream.inner.graph_id);
            }

            cmd.execs.push(Default::default());
            self.cmds.push(cmd);
        }
    }
}

/// A graph node that can be borrowed while building a [`CommandStream`].
#[allow(private_bounds)]
#[doc(hidden)]
pub trait StreamResourceNode: stream_private::StreamResourceNodeSealed + ResourceNode {}

/// A typed, invocation-owned constant declared with [`CommandStreamMut::add_value_arg`].
#[derive(Clone, Copy, Debug)]
pub struct StreamValueArg<T: Copy + Send + Sync + 'static> {
    index: usize,
    #[cfg(feature = "checked")]
    scope: u64,
    __: PhantomData<fn() -> T>,
}

pub(crate) struct StreamValues {
    #[cfg(feature = "checked")]
    scope: u64,
    values: Box<[Arc<dyn Any + Send + Sync>]>,
}

impl StreamValues {
    pub(crate) fn value<T: Copy + Send + Sync + 'static>(&self, arg: StreamValueArg<T>) -> T {
        #[cfg(feature = "checked")]
        assert_eq!(
            self.scope, arg.scope,
            "value argument belongs to a different command stream"
        );

        *self.values[arg.index]
            .downcast_ref::<T>()
            .expect("invalid command stream value type")
    }
}

mod stream_private {
    pub trait StreamArgBindableSealed<T> {}

    pub trait StreamArgInfoSealed {}

    pub trait StreamPipelineSealed {}

    pub trait StreamResourceNodeSealed {}
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
stream_arg_info!(MicromapInfo, MicromapInfoBuilder, Micromap, MicromapArg);

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
stream_arg_bindable!(Micromap => MicromapNode, MicromapLeaseNode);

impl stream_private::StreamArgBindableSealed<AccelerationStructure>
    for AnyAccelerationStructureNode
{
}

impl StreamArgBindable<AccelerationStructure> for AnyAccelerationStructureNode {
    fn assert_parent_node(&self) {
        #[cfg(feature = "checked")]
        assert!(
            !matches!(self, Self::Arg(_)),
            "stream argument cannot be supplied as a parent graph node"
        );
    }
}

impl stream_private::StreamArgBindableSealed<Buffer> for AnyBufferNode {}

impl StreamArgBindable<Buffer> for AnyBufferNode {
    fn assert_parent_node(&self) {
        #[cfg(feature = "checked")]
        assert!(
            !matches!(self, Self::Arg(_)),
            "stream argument cannot be supplied as a parent graph node"
        );
    }
}

impl stream_private::StreamArgBindableSealed<Image> for AnyImageNode {}

impl StreamArgBindable<Image> for AnyImageNode {
    fn assert_parent_node(&self) {
        #[cfg(feature = "checked")]
        assert!(
            !matches!(self, Self::Arg(_)),
            "stream argument cannot be supplied as a parent graph node"
        );
    }
}

impl stream_private::StreamArgBindableSealed<Micromap> for AnyMicromapNode {}

impl StreamArgBindable<Micromap> for AnyMicromapNode {
    fn assert_parent_node(&self) {
        #[cfg(feature = "checked")]
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
    AccelerationStructureLeaseNode,
    AccelerationStructureNode,
    AccelerationStructureSetNode,
    BufferLeaseNode,
    BufferNode,
    ImageLeaseNode,
    ImageNode,
    ImageSetNode,
    MicromapLeaseNode,
    MicromapNode,
    SwapchainImageNode,
);

fn next_stream_scope_id() -> u64 {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);

    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

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
        resource::{
            AccelerationStructureAccessType, AccelerationStructureSet,
            AccelerationStructureSetMember, ImageAccessType, ImageSet, ImageSetMember,
            ResourceSetAccessType, ResourceSetIndex,
        },
    };

    #[cfg(feature = "checked")]
    #[test]
    #[should_panic(expected = "value argument belongs to a different command stream")]
    fn cross_stream_value_argument_panics() {
        let first = CommandStream::finalize(|stream| stream.add_value_arg::<u32>()).into_stream();
        let second = CommandStream::finalize(|stream| stream.add_value_arg::<u32>()).into_stream();

        Graph::new()
            .insert_cmd_stream(&second)
            .with_value(first.args, 1);
    }

    #[cfg(feature = "checked")]
    #[test]
    #[should_panic(expected = "missing command stream value argument")]
    fn missing_value_argument_panics() {
        let stream = CommandStream::finalize(|stream| stream.add_value_arg::<u32>()).into_stream();
        Graph::new().insert_cmd_stream(&stream).finish();
    }

    #[test]
    fn value_arguments_snapshot_and_rebind() {
        let stream = CommandStream::finalize(|stream| {
            let value = stream.add_value_arg::<u32>();
            stream.begin_cmd().record_cmd(|_| {});

            value
        })
        .into_stream();
        let mut graph = Graph::new();
        graph
            .insert_cmd_stream(&stream)
            .with_value(stream.args, 1)
            .with_value(stream.args, 2)
            .finish();
        graph
            .insert_cmd_stream(&stream)
            .with_value(stream.args, 3)
            .finish();

        assert_eq!(
            graph.cmds[0].execs[0]
                .stream_values
                .as_ref()
                .unwrap()
                .value(stream.args),
            2
        );
        assert_eq!(
            graph.cmds[1].execs[0]
                .stream_values
                .as_ref()
                .unwrap()
                .value(stream.args),
            3
        );
    }

    #[test]
    #[ignore = "requires Vulkan device"]
    fn vulkan_shader_arguments_keep_descriptors_per_invocation() {
        use crate::{
            driver::{
                compute::{ComputePipeline, ComputePipelineInfo},
                descriptor_set::{DescriptorSetInfo, DescriptorSetUpdateInfo},
                device::{Device, DeviceInfo},
                shader::Shader,
            },
            pool::hash::HashPool,
        };

        let device = Device::create(DeviceInfo::default()).unwrap();
        let mut pool = HashPool::new(&device);
        let pipeline = ComputePipeline::create(
            &device,
            ComputePipelineInfo::default(),
            Shader::new_compute(
                vk_shader_macros::glsl!(
                    r#"
                #version 450
                #pragma shader_stage(compute)
                layout(local_size_x = 1) in;
                layout(binding = 0) readonly buffer Input { uint value; } src;
                layout(binding = 1) writeonly buffer Output { uint value; } dst;
                layout(set = 1, binding = 0) readonly buffer Bias { uint value; } bias;
                layout(push_constant) uniform Constants { uint value; } constants;
                void main() { dst.value = src.value + bias.value + constants.value; }
                "#
                )
                .as_slice(),
            ),
        )
        .unwrap();

        for prepared in [false, true] {
            let bias = Arc::new(
                Buffer::create_from_slice(
                    &device,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    &7u32.to_ne_bytes(),
                )
                .unwrap(),
            );
            let descriptor_set = DescriptorSet::alloc_and_update(
                &pipeline,
                DescriptorSetInfo { set: 1 },
                [DescriptorSetUpdateInfo::buffer(0, &bias)],
            )
            .unwrap();
            let draft = CommandStream::finalize(|stream| {
                let constant = stream.add_value_arg::<u32>();
                let bias_node = stream.bind_resource(&bias);
                let input = stream.arg(BufferInfo::device_mem(
                    4,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ));
                let output = stream.arg(BufferInfo::device_mem(
                    4,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ));
                stream
                    .begin_cmd()
                    .bind_pipeline(&pipeline)
                    .bind_descriptor_set(1, &descriptor_set)
                    .resource_access(bias_node, AccessType::ComputeShaderReadOther)
                    .shader_resource_access(
                        0,
                        AnyBufferNode::from(input),
                        AccessType::ComputeShaderReadOther,
                    )
                    .shader_resource_access(
                        1,
                        AnyBufferNode::from(output),
                        AccessType::ComputeShaderWrite,
                    )
                    .record_cmd(move |cmd| {
                        cmd.push_constants(0, &cmd.value(constant).to_ne_bytes())
                            .dispatch(1, 1, 1);
                    });

                (input, output, constant)
            });
            drop(descriptor_set);
            drop(bias);
            let stream = if prepared {
                draft.prepare(&mut pool).unwrap()
            } else {
                draft.into_stream()
            };
            // Keep two submissions alive, then reuse their recording slots with new buffers.
            for round in 0..2 {
                let mut pending = Vec::new();
                let mut outputs = Vec::new();
                for frame in 0..2 {
                    let mut graph = Graph::new();
                    for invocation in 0..2 {
                        let value = round * 100 + frame * 10 + invocation;
                        let input = Arc::new(
                            Buffer::create_from_slice(
                                &device,
                                vk::BufferUsageFlags::STORAGE_BUFFER,
                                &u32::to_ne_bytes(value),
                            )
                            .unwrap(),
                        );
                        let output = Arc::new(
                            Buffer::create_from_slice(
                                &device,
                                vk::BufferUsageFlags::STORAGE_BUFFER,
                                &[0; 4],
                            )
                            .unwrap(),
                        );
                        let input_node = graph.bind_resource(input);
                        let output_node = graph.bind_resource(&output);
                        let run = graph.insert_cmd_stream(&stream);
                        let run = if round == 0 {
                            run.with_arg(stream.args.0, input_node)
                                .with_arg(stream.args.1, output_node)
                        } else {
                            run.with_args([
                                (stream.args.0, input_node),
                                (stream.args.1, output_node),
                            ])
                        };
                        run.with_value(stream.args.2, value * 3).finish();
                        outputs.push((output, value * 4 + 7));
                    }

                    pending.push(graph.finalize().queue_submit(&mut pool, 0, 0).unwrap());
                }

                for mut submission in pending {
                    submission.wait().unwrap();
                }

                for (output, expected) in outputs {
                    assert_eq!(Buffer::mapped_slice(&output), expected.to_ne_bytes());
                }
            }
        }
    }

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

    /// Metadata only: real typed argument/node identities, no Vulkan allocation or recording.
    #[test]
    #[ignore = "release-mode binding microbenchmark"]
    fn binding_metadata_benchmark() {
        use std::{hint::black_box, time::Instant};

        for (buffers, acceleration_structures) in [(4, 0), (5, 0), (30, 54)] {
            for prepared in [false, true] {
                let draft = CommandStream::finalize(|stream| {
                    let buffers = (0..buffers)
                        .map(|_| {
                            stream.arg(BufferInfo::device_mem(
                                4096,
                                vk::BufferUsageFlags::STORAGE_BUFFER,
                            ))
                        })
                        .collect::<Vec<_>>();
                    let acceleration_structures = (0..acceleration_structures)
                        .map(|_| stream.arg(AccelerationStructureInfo::blas(4096)))
                        .collect::<Vec<_>>();
                    stream.begin_cmd().record_cmd(|_| {});

                    (buffers, acceleration_structures)
                });
                let stream = if prepared {
                    draft.prepare(&mut NoopPool).unwrap()
                } else {
                    draft.into_stream()
                };
                let mut graph = Graph::new();
                let buffers = (0..buffers)
                    .map(|_| bind_test_buffer(&mut graph))
                    .collect::<Vec<_>>();
                let acceleration_structures = (0..acceleration_structures)
                    .map(|_| {
                        let index =
                            graph.bind_stream_arg_resource(AnyResource::AccelerationStructureArg(
                                AccelerationStructureInfo::blas(4096),
                            ));

                        AccelerationStructureNode::new(
                            index,
                            #[cfg(feature = "checked")]
                            graph.graph_id(),
                        )
                    })
                    .collect::<Vec<_>>();
                // Deliberately nonmonotonic bindings, as in an already populated frame graph.
                let buffers = buffers.into_iter().rev().collect::<Vec<_>>();
                let acceleration_structures = acceleration_structures
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>();
                for (batched, finish) in
                    [(false, false), (true, false), (false, true), (true, true)]
                {
                    let mut samples = Vec::new();
                    for _ in 0..7 {
                        let start = Instant::now();
                        for _ in 0..20_000 {
                            let mut run = graph.insert_cmd_stream(black_box(&stream));
                            if batched {
                                run = run.with_args(
                                    stream
                                        .args
                                        .0
                                        .iter()
                                        .copied()
                                        .zip(buffers.iter().copied())
                                        .map(black_box),
                                );
                                if !acceleration_structures.is_empty() {
                                    run = run.with_args(
                                        stream
                                            .args
                                            .1
                                            .iter()
                                            .copied()
                                            .zip(acceleration_structures.iter().copied())
                                            .map(black_box),
                                    );
                                }
                            } else {
                                for (&arg, &node) in stream.args.0.iter().zip(&buffers) {
                                    run = run.with_arg(black_box(arg), black_box(node));
                                }

                                for (&arg, &node) in
                                    stream.args.1.iter().zip(&acceleration_structures)
                                {
                                    run = run.with_arg(black_box(arg), black_box(node));
                                }
                            }
                            if finish {
                                black_box(run.finish()).cmds.clear();
                            } else {
                                black_box(run);
                            }
                        }

                        samples.push(start.elapsed().as_nanos() as f64 / 20_000.0);
                    }

                    samples.sort_by(f64::total_cmp);

                    eprintln!(
                        "buffers={} AS={} prepared={prepared} batched={batched} finish={finish}: {:.1} ns median",
                        buffers.len(),
                        acceleration_structures.len(),
                        samples[3]
                    );
                }
            }
        }
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
        let submission = graph.finalize();

        assert_eq!(submission.graph().cmds.len(), 2);
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

    #[cfg(feature = "checked")]
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

    #[cfg(feature = "checked")]
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
    fn batched_arguments_validate_and_replace() {
        for prepared in [false, true] {
            let draft = two_buffer_arg_stream();
            let stream = if prepared {
                draft.prepare(&mut NoopPool).unwrap()
            } else {
                draft.into_stream()
            };
            let mut graph = Graph::new();
            let first = bind_test_buffer(&mut graph);
            let second = bind_test_buffer(&mut graph);
            let run = graph
                .insert_cmd_stream(&stream)
                .with_arg(stream.args.0, first)
                .with_args([
                    (stream.args.0, second),
                    (stream.args.0, first),
                    (stream.args.1, second),
                ])
                .with_args([(stream.args.0, second), (stream.args.1, first)])
                .with_args(std::iter::empty::<(BufferArg, BufferNode)>());

            assert_eq!(run.bindings, [Some(second.index()), Some(first.index())]);
            run.finish();

            #[cfg(feature = "checked")]
            for scenario in 0..4 {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let run = graph.insert_cmd_stream(&stream);
                    match scenario {
                        0 => {
                            run.with_args([(stream.args.0, first), (stream.args.1, first)]);
                        }
                        1 => {
                            run.with_args([(stream.args.0, first)])
                                .with_args([(stream.args.1, first)]);
                        }
                        2 => {
                            run.with_args([(stream.args.0, first)])
                                .with_arg(stream.args.1, first);
                        }
                        3 => {
                            run.with_args([(stream.args.0, first)]).finish();
                        }
                        _ => unreachable!(),
                    }
                }));

                assert!(result.is_err(), "prepared={prepared} scenario={scenario}");
            }
        }
    }

    #[cfg(feature = "checked")]
    #[test]
    fn batched_arguments_reject_wrong_owners() {
        for prepared in [false, true] {
            let other = two_buffer_arg_stream().into_stream();
            let draft = two_buffer_arg_stream();
            let stream = if prepared {
                draft.prepare(&mut NoopPool).unwrap()
            } else {
                draft.into_stream()
            };
            let mut graph = Graph::new();
            let node = bind_test_buffer(&mut graph);

            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    graph
                        .insert_cmd_stream(&stream)
                        .with_args([(other.args.0, node)]);
                }))
                .is_err()
            );
            let mut other_graph = Graph::new();

            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    other_graph
                        .insert_cmd_stream(&stream)
                        .with_args([(stream.args.0, node)]);
                }))
                .is_err()
            );
        }
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
    fn micromap_arg_can_declare_resource_access() {
        let stream = CommandStream::finalize(|stream| {
            let micromap = stream.arg(MicromapInfo::device_mem(64));

            stream
                .begin_cmd()
                .resource_access(micromap, vk_sync::AccessType::MicromapBuildRead)
                .record_cmd(|_| {});

            micromap
        })
        .into_stream();

        assert_eq!(stream.inner.args.len(), 1);
        assert!(matches!(
            AnyMicromapNode::from(stream.args),
            AnyMicromapNode::Arg(_)
        ));
    }

    #[cfg(feature = "checked")]
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

    #[cfg(feature = "checked")]
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

    #[cfg(feature = "checked")]
    #[test]
    #[should_panic(expected = "command stream contains a one-shot callback")]
    fn one_shot_callback_cannot_prepare_stream() {
        let _ = CommandStream::finalize(|stream| {
            stream.graph.begin_cmd().record_cmd(|_| {});
        });
    }

    #[test]
    fn image_sets_can_bind_to_unprepared_command_streams() {
        let parent_resource_set = ImageSet::new(std::iter::empty::<ImageSetMember>()).unwrap();
        let stream_resource_set = ImageSet::new(std::iter::empty::<ImageSetMember>()).unwrap();

        let stream = CommandStream::finalize(|stream| {
            let resource_set = stream.bind_resource(&stream_resource_set);
            stream
                .begin_cmd()
                .resource_access(resource_set, ImageAccessType::SampledRead)
                .record_cmd(|_| {});
        })
        .into_stream();
        let mut graph = Graph::new();
        graph.bind_resource(&parent_resource_set);
        graph.insert_cmd_stream(&stream).finish();

        let submission = graph.finalize();
        let graph = submission.graph();
        let accesses = &graph.cmds[0].execs[0].resource_set_accesses;

        assert_eq!(graph.resource_sets.len(), 2);
        assert_eq!(accesses.len(), 1);
        assert_eq!(accesses[0].resource_set_idx, ResourceSetIndex::new(1));
        assert_eq!(
            accesses[0].access_type,
            ResourceSetAccessType::Image(ImageAccessType::SampledRead)
        );
    }

    #[test]
    fn resource_sets_can_prepare_command_streams() {
        let mut pool = NoopPool;
        let parent_resource_set = ImageSet::new(std::iter::empty::<ImageSetMember>()).unwrap();
        let image_set = ImageSet::new(std::iter::empty::<ImageSetMember>()).unwrap();
        let acceleration_structure_set =
            AccelerationStructureSet::new(std::iter::empty::<AccelerationStructureSetMember>())
                .unwrap();

        let stream = CommandStream::prepare(&mut pool, |stream| {
            let image_set = stream.bind_resource(&image_set);
            let acceleration_structure_set = stream.bind_resource(&acceleration_structure_set);
            stream
                .begin_cmd()
                .resource_access(
                    acceleration_structure_set,
                    AccelerationStructureAccessType::BuildRead,
                )
                .resource_access(image_set, ImageAccessType::SampledRead)
                .record_cmd(|_| {});
            stream
                .begin_cmd()
                .resource_access(
                    acceleration_structure_set,
                    AccelerationStructureAccessType::RayTracingRead,
                )
                .resource_access(image_set, ImageAccessType::SampledRead)
                .record_cmd(|_| {});
        })
        .expect("prepare stream");
        let mut graph = Graph::new();
        let parent_resource_set = graph.bind_resource(&parent_resource_set);
        graph
            .begin_cmd()
            .resource_access(parent_resource_set, ImageAccessType::SampledRead)
            .record_cmd(|_| {});
        graph.insert_cmd_stream(&stream).finish();

        let submission = graph.finalize();
        let graph = submission.graph();
        let opaque_cmd = graph
            .cmds
            .iter()
            .find(|cmd| cmd.stream_scope_id.is_some())
            .expect("missing prepared command stream");
        let accesses = &opaque_cmd.execs[0].resource_set_accesses;

        assert_eq!(graph.resource_sets.len(), 3);
        assert_eq!(accesses.len(), 3);
        assert!(accesses.contains(&crate::ResourceSetAccess {
            resource_set_idx: ResourceSetIndex::new(1),
            access_type: ResourceSetAccessType::Image(ImageAccessType::SampledRead),
        }));
        assert!(accesses.contains(&crate::ResourceSetAccess {
            resource_set_idx: ResourceSetIndex::new(2),
            access_type: ResourceSetAccessType::AccelerationStructure(
                AccelerationStructureAccessType::BuildRead,
            ),
        }));
        assert!(accesses.contains(&crate::ResourceSetAccess {
            resource_set_idx: ResourceSetIndex::new(2),
            access_type: ResourceSetAccessType::AccelerationStructure(
                AccelerationStructureAccessType::RayTracingRead,
            ),
        }));
    }

    #[test]
    fn prepared_command_stream_resource_sets_reuse_parent_identity() {
        let mut pool = NoopPool;
        let resource_set = ImageSet::new(std::iter::empty::<ImageSetMember>()).unwrap();
        let stream = CommandStream::prepare(&mut pool, |stream| {
            let resource_set = stream.bind_resource(&resource_set);
            stream
                .begin_cmd()
                .resource_access(resource_set, ImageAccessType::SampledRead)
                .record_cmd(|_| {});
        })
        .expect("prepare stream");
        let mut graph = Graph::new();
        graph.bind_resource(&resource_set);
        graph.insert_cmd_stream(&stream).finish();

        let submission = graph.finalize();
        let graph = submission.graph();
        let accesses = &graph.cmds[0].execs[0].resource_set_accesses;

        assert_eq!(graph.resource_sets.len(), 1);
        assert_eq!(accesses[0].resource_set_idx, ResourceSetIndex::new(0));
    }

    #[cfg(feature = "checked")]
    #[test]
    #[should_panic(expected = "prepared command stream invocations cannot be captured")]
    fn prepared_outer_stream_rejects_prepared_capture() {
        let mut pool = NoopPool;
        let inner = CommandStream::prepare(&mut pool, |stream| {
            stream.begin_cmd().record_cmd(|_| {});
        })
        .expect("prepare inner stream");
        CommandStream::prepare(&mut pool, |stream| {
            stream.graph.insert_cmd_stream(&inner).finish();
        })
        .expect("prepare outer stream");
    }

    #[cfg(feature = "checked")]
    #[test]
    #[should_panic(expected = "prepared command stream invocations cannot be captured")]
    fn unprepared_outer_stream_rejects_prepared_capture() {
        let mut pool = NoopPool;
        let inner = CommandStream::prepare(&mut pool, |stream| {
            stream.begin_cmd().record_cmd(|_| {});
        })
        .expect("prepare inner stream");
        CommandStream::finalize(|stream| {
            stream.graph.insert_cmd_stream(&inner).finish();
        })
        .into_stream();
    }

    #[cfg(feature = "checked")]
    #[test]
    fn nested_unprepared_stream_preserves_callback_owner_and_composes_set_map() {
        let inner_set = ImageSet::new(std::iter::empty::<ImageSetMember>()).unwrap();
        let inner = CommandStream::finalize(|stream| {
            let set = stream.bind_resource(&inner_set);
            stream
                .begin_cmd()
                .resource_access(set, ImageAccessType::SampledRead)
                .record_cmd(move |_| {
                    let _ = set;
                });
        })
        .into_stream();
        let inner_graph_id = inner.inner.graph_id;
        let middle_set = ImageSet::new(std::iter::empty::<ImageSetMember>()).unwrap();
        let middle = CommandStream::finalize(|stream| {
            stream.bind_resource(&middle_set);
            stream.graph.insert_cmd_stream(&inner).finish();
        })
        .into_stream();
        let parent_set = ImageSet::new(std::iter::empty::<ImageSetMember>()).unwrap();
        let mut graph = Graph::new();
        graph.bind_resource(&parent_set);

        graph.insert_cmd_stream(&middle).finish();

        let exec = &graph.cmds[0].execs[0];

        assert_eq!(exec.stream_graph_id, Some(inner_graph_id));
        assert_eq!(
            exec.resource_set_map.as_deref(),
            Some([ResourceSetIndex::new(2)].as_slice())
        );
        assert_eq!(
            exec.resource_set_accesses[0].resource_set_idx,
            ResourceSetIndex::new(2)
        );
    }
}
