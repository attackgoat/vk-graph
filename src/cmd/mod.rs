//! Strongly-typed [`Graph`] commands.
//!
//! ## Lifecycle
//!
//! Commands follow a builder-style chain:
//!
//! 1. [`Graph::begin_cmd`] opens a [`Command`].
//! 2. Declare resource accesses with [`Command::resource_access`] or bind a shader pipeline
//!    with [`Command::bind_pipeline`], returning a [`PipelineCommand`].
//! 3. With a pipeline, declare shader bindings with [`PipelineCommand::shader_resource_access`].
//! 4. Record work with [`record_cmd`](Command::record_cmd) — available on both
//!    [`Command`] and [`PipelineCommand`].
//! 5. The command auto-closes when dropped or when [`Graph::finalize`] is called.
//!
//! A single command can call `record_cmd` multiple times — each call creates a separate
//! "execution" within that command. Executions within a command stay in the specified
//! order, but the graph system may re-order entire commands or merge them during
//! submission for optimal scheduling.

mod cmd_ref;
mod compute;
mod graphics;
mod pipeline;
mod ray_tracing;

pub use {
    self::{
        cmd_ref::{
            BuildAccelerationStructureIndirectInfo, BuildAccelerationStructureInfo, CommandRef,
            UpdateAccelerationStructureIndirectInfo, UpdateAccelerationStructureInfo,
        },
        compute::ComputeCommandRef,
        graphics::{ClearColorValue, GraphicsCommandRef},
        pipeline::{Pipeline, PipelineCommand},
        ray_tracing::RayTracingCommandRef,
    },
    super::{LoadOp, StoreOp},
};

use {
    super::{
        AccelerationStructureLeaseNode, AccelerationStructureNode, AnyAccelerationStructureNode,
        AnyBufferNode, AnyImageNode, AnyResource, BufferLeaseNode, BufferNode, CommandData,
        CommandExecution, CommandFunction, Execution, Graph, ImageLeaseNode, ImageNode, Node,
        Resource, SwapchainImageNode,
    },
    crate::{
        NodeIndex,
        driver::{buffer::BufferSubresourceRange, image::ImageViewInfo},
        stream::{AccelerationStructureArg, BufferArg, ImageArg},
    },
    ash::vk,
    std::{ops::Range, sync::Arc},
    vk_sync::AccessType,
};

/// Alias for the index of a framebuffer attachment.
pub(crate) type AttachmentIndex = u32;

/// Alias for the binding index of a shader descriptor.
pub(crate) type BindingIndex = u32;

/// Alias for the binding offset of a shader descriptor array element.
pub(crate) type BindingOffset = u32;

/// Alias for the descriptor set index of a shader descriptor.
pub(crate) type DescriptorSetIndex = u32;

/// A general-purpose Vulkan command which may contain acceleration structure operations, transfers,
/// or shader pipelines.
///
/// There are four main uses of a [`Command`]:
///
/// 1. Bind resources ([`Self::bind_resource`])
/// 1. Declare resource accesses ([`Self::resource_access`])
/// 1. Record general-purpose command buffers or acceleration structure operations
///    ([`Self::record_cmd`])
/// 1. Bind shader pipelines ([`Self::bind_pipeline`])
///
/// When bound, a shader pipeline consumes the `Command` and returns a [`PipelineCommand`] which
/// provides command recording functions specific to each pipeline type.
pub struct Command<'a> {
    pub(super) cmd_idx: usize,
    pub(super) exec_idx: usize,
    pub(super) graph: &'a mut Graph,
}

#[allow(private_bounds)]
impl<'a> Command<'a> {
    pub(super) fn new(graph: &'a mut Graph) -> Self {
        let cmd_idx = graph.cmds.len();
        graph.cmds.push(CommandData {
            execs: vec![Default::default()], // We start off with a default execution!
            #[cfg(debug_assertions)]
            name: None,
            stream_scope_id: None,
            tracking: Default::default(),
        });

        Self {
            cmd_idx,
            exec_idx: 0,
            graph,
        }
    }

    /// Returns a handle that tracks whether this graph command has completed device execution.
    ///
    /// This may be called multiple times. Each returned handle independently observes the same
    /// command execution.
    pub fn track_execution(&mut self) -> CommandExecution {
        self.cmd_mut().tracking.track()
    }

    fn cmd(&self) -> &CommandData {
        &self.graph.cmds[self.cmd_idx]
    }

    fn cmd_mut(&mut self) -> &mut CommandData {
        &mut self.graph.cmds[self.cmd_idx]
    }

    /// Binds a Vulkan buffer, image, or acceleration structure resource to the graph associated
    /// with this command.
    ///
    /// Bound nodes may be used in commands for pipeline and shader operations.
    pub fn bind_resource<R>(&mut self, resource: R) -> R::Node
    where
        R: Resource,
    {
        self.graph.bind_resource(resource)
    }

    /// Binds a shader pipeline to the current command, allowing for strongly typed access to the
    /// related functions.
    ///
    /// | `P` | `P::Command` |
    /// | --- | --- |
    /// | [`ComputePipeline`](crate::driver::compute::ComputePipeline) | [`PipelineCommand<'_, ComputePipeline>`] |
    /// | [`GraphicsPipeline`](crate::driver::graphics::GraphicsPipeline) | [`PipelineCommand<'_, GraphicsPipeline>`] |
    /// | [`RayTracingPipeline`](crate::driver::ray_tracing::RayTracingPipeline) | [`PipelineCommand<'_, RayTracingPipeline>`] |
    pub fn bind_pipeline<P>(self, pipeline: P) -> P::Command
    where
        P: Pipeline<'a>,
    {
        pipeline.bind_cmd(self)
    }

    /// Sets a debugging name, but only in debug builds.
    pub fn debug_name(mut self, name: impl Into<String>) -> Self {
        self.set_debug_name(name);
        self
    }

    /// Finalize the recording of this command and return to the `Graph` where you may record
    /// additional commands.
    pub fn end_cmd(self) -> &'a mut Graph {
        // If nothing was done in this command we can just ignore it.
        if self.exec_idx == 0 {
            self.graph.cmds.pop();
        }

        self.graph
    }

    fn push_exec(&mut self, func: impl FnOnce(CommandRef) + Send + 'static) {
        let cmd = self.cmd_mut();
        let exec = {
            let last_exec = cmd.expect_last_exec_mut();
            last_exec.func = Some(CommandFunction::Once(Box::new(func)));

            Execution {
                pipeline: last_exec.pipeline.clone(),
                ..Default::default()
            }
        };

        cmd.execs.push(exec);
        self.exec_idx += 1;
    }

    pub(crate) fn push_reusable_exec(
        &mut self,
        func: impl for<'r> Fn(CommandRef<'r>) + Send + Sync + 'static,
    ) {
        let cmd = self.cmd_mut();
        let exec = {
            let last_exec = cmd.expect_last_exec_mut();
            last_exec.func = Some(CommandFunction::Reusable(Arc::new(func)));

            Execution {
                pipeline: last_exec.pipeline.clone(),
                ..Default::default()
            }
        };

        cmd.execs.push(exec);
        self.exec_idx += 1;
    }

    fn push_subresource_access(
        &mut self,
        resource_node: impl Node,
        subresource: SubresourceRange,
        access: AccessType,
    ) {
        self.graph.assert_node_owner(&resource_node);

        let node_idx = resource_node.index();

        self.push_subresource_access_index(node_idx, subresource, access);
    }

    pub(crate) fn push_subresource_access_index(
        &mut self,
        node_idx: NodeIndex,
        subresource: SubresourceRange,
        access: AccessType,
    ) {
        debug_assert!(self.graph.resources.get(node_idx).is_some());

        self.cmd_mut().expect_last_exec_mut().accesses.push(
            node_idx,
            SubresourceAccess {
                access,
                subresource,
            },
        );
    }

    /// Begin recording general-purpose work for this graph command.
    ///
    /// This is the entry point for building and updating an
    /// [`AccelerationStructure`](crate::driver::accel_struct::AccelerationStructure) instance.
    ///
    /// The provided closure allows you to run any Vulkan code, or interoperate with other Vulkan
    /// code and interfaces.
    pub fn record_cmd(mut self, func: impl FnOnce(CommandRef<'_>) + Send + 'static) -> Self {
        self.record_cmd_mut(func);
        self
    }

    /// Mutable-borrow form of [`Self::record_cmd`].
    pub fn record_cmd_mut(&mut self, func: impl FnOnce(CommandRef<'_>) + Send + 'static) {
        self.push_exec(move |cmd| {
            func(cmd);
        });
    }

    pub(crate) fn record_stream(
        mut self,
        func: impl for<'r> Fn(CommandRef<'r>) + Send + Sync + 'static,
    ) -> Self {
        self.record_stream_mut(func);
        self
    }

    pub(crate) fn record_stream_mut(
        &mut self,
        func: impl for<'r> Fn(CommandRef<'r>) + Send + Sync + 'static,
    ) {
        self.push_reusable_exec(func);
    }

    /// Returns a borrow of the original Vulkan resource (buffer, image or acceleration structure)
    /// which the given bound resource node represents.
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: Node,
    {
        self.graph.resource(resource_node)
    }

    /// Informs the command that recorded work will read or write `resource_node`
    /// using `access`.
    ///
    /// An access function must be called for `resource_node` before it is used within a recording
    /// function.
    pub fn resource_access<N>(mut self, resource_node: N, access: AccessType) -> Self
    where
        N: Node + Subresource,
        SubresourceRange: From<N::Range>,
    {
        self.set_resource_access(resource_node, access);
        self
    }

    /// Mutable-borrow form of [`Self::debug_name`].
    pub fn set_debug_name(&mut self, name: impl Into<String>) -> &mut Self {
        #[cfg(debug_assertions)]
        {
            self.cmd_mut().name = Some(name.into());
        }

        #[cfg(not(debug_assertions))]
        {
            let _ = name;
        }

        self
    }

    /// Mutable-borrow form of [`Self::resource_access`].
    pub fn set_resource_access<N>(&mut self, resource_node: N, access: AccessType)
    where
        N: Node + Subresource,
        SubresourceRange: From<N::Range>,
    {
        let whole_resource = resource_node.range(&self.graph.resources);
        let subresource = SubresourceRange::from(whole_resource);

        self.push_subresource_access(resource_node, subresource, access);
    }

    pub(crate) fn set_stream_scope_id(&mut self, stream_scope_id: u64) {
        self.cmd_mut().stream_scope_id = Some(stream_scope_id);
    }

    /// Mutable-borrow form of [`Self::subresource_access`].
    pub fn set_subresource_access<N>(
        &mut self,
        resource_node: N,
        subresource: impl Into<N::Range>,
        access: AccessType,
    ) where
        N: Node + Subresource,
        SubresourceRange: From<N::Range>,
    {
        let subresource = subresource.into();
        let subresource = SubresourceRange::from(subresource);

        self.push_subresource_access(resource_node, subresource, access);
    }

    /// Informs the command that recorded work will read or write the `subresource` of
    /// `resource_node` using `access`.
    ///
    /// An access function must be called for `resource_node` before it is used within a recording
    /// function.
    pub fn subresource_access<N>(
        mut self,
        resource_node: N,
        subresource: impl Into<N::Range>,
        access: AccessType,
    ) -> Self
    where
        N: Node + Subresource,
        SubresourceRange: From<N::Range>,
    {
        self.set_subresource_access(resource_node, subresource, access);
        self
    }
}

/// Describes the SPIR-V binding index, and optionally a specific descriptor set
/// and array index.
///
/// Generally you might pass a function a descriptor using a simple integer:
///
/// ```rust
/// # fn my_func(_: usize, _: ()) {}
/// # let image = ();
/// let descriptor = 42;
/// my_func(descriptor, image);
/// ```
///
/// But also:
///
/// - `(0, 42)` for descriptor set `0` and binding index `42`
/// - `(42, [8])` for the same binding, but the 8th element
/// - `(0, 42, [8])` same as the previous example
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Binding {
    /// The value of the descriptor binding decoration applied to the variable.
    pub binding: u32,

    /// An array-element offset applied to this descriptor.
    pub offset: u32,

    /// An optional descriptor set index value.
    pub set: u32,
}

impl Binding {
    pub(super) fn into_tuple(self) -> (DescriptorSetIndex, BindingIndex, BindingOffset) {
        (self.set, self.binding, self.offset)
    }

    pub(super) fn set(self) -> DescriptorSetIndex {
        let (res, _, _) = self.into_tuple();
        res
    }
}

impl From<BindingIndex> for Binding {
    fn from(binding: BindingIndex) -> Self {
        Self {
            binding,
            offset: 0,
            set: 0,
        }
    }
}

impl From<(DescriptorSetIndex, BindingIndex)> for Binding {
    fn from((set, binding): (DescriptorSetIndex, BindingIndex)) -> Self {
        Self {
            binding,
            offset: 0,
            set,
        }
    }
}

impl From<(BindingIndex, [BindingOffset; 1])> for Binding {
    fn from((binding, [offset]): (BindingIndex, [BindingOffset; 1])) -> Self {
        Self {
            binding,
            offset,
            set: 0,
        }
    }
}

impl From<(DescriptorSetIndex, BindingIndex, [BindingOffset; 1])> for Binding {
    fn from(
        (set, binding, [offset]): (DescriptorSetIndex, BindingIndex, [BindingOffset; 1]),
    ) -> Self {
        Self {
            binding,
            offset,
            set,
        }
    }
}

/// Allows for a resource to be reinterpreted as differently formatted data.
#[allow(private_bounds)]
pub trait Subresource: private::SubresourceSealed {
    /// The information about the subresource when bound directly to shader descriptors.
    type Info;

    /// The information about the subresource when used indirectly by any part of a graph.
    type Range;
}

macro_rules! view_accel_struct {
    ($name:ty) => {
        impl Subresource for $name {
            type Info = Self::Range;
            type Range = ();
        }

        impl private::SubresourceSealed for $name {
            fn info(&self, _: &[AnyResource]) -> <Self as Subresource>::Info
            where
                Self: Node + Subresource,
            {
            }

            fn range(&self, resources: &[AnyResource]) -> <Self as Subresource>::Range
            where
                Self: Node + Subresource,
            {
                resources[self.index()].expect_accel_struct_info();
            }
        }
    };
}

view_accel_struct!(AnyAccelerationStructureNode);
view_accel_struct!(AccelerationStructureArg);
view_accel_struct!(AccelerationStructureLeaseNode);
view_accel_struct!(AccelerationStructureNode);

macro_rules! view_buffer {
    ($name:ty) => {
        impl Subresource for $name {
            type Info = Self::Range;
            type Range = BufferSubresourceRange;
        }

        impl private::SubresourceSealed for $name {
            fn info(&self, resources: &[AnyResource]) -> <Self as Subresource>::Info
            where
                Self: Node + Subresource,
            {
                self.range(resources)
            }

            fn range(&self, resources: &[AnyResource]) -> <Self as Subresource>::Range
            where
                Self: Node + Subresource,
            {
                let idx = self.index();

                resources[idx].expect_buffer_info().into()
            }
        }
    };
}

view_buffer!(AnyBufferNode);
view_buffer!(BufferArg);
view_buffer!(BufferLeaseNode);
view_buffer!(BufferNode);

macro_rules! view_image {
    ($name:ty) => {
        impl Subresource for $name {
            type Info = ImageViewInfo;
            type Range = vk::ImageSubresourceRange;
        }

        impl private::SubresourceSealed for $name {
            fn info(&self, resources: &[AnyResource]) -> <Self as Subresource>::Info
            where
                Self: Node + Subresource,
            {
                let idx = self.index();

                resources[idx].expect_image_info().into()
            }

            fn range(&self, resources: &[AnyResource]) -> <Self as Subresource>::Range
            where
                Self: Node + Subresource,
            {
                self.info(resources).into()
            }
        }
    };
}

view_image!(AnyImageNode);
view_image!(ImageArg);
view_image!(ImageLeaseNode);
view_image!(ImageNode);
view_image!(SwapchainImageNode);

#[derive(Clone, Copy, Debug)]
pub(crate) enum SubresourceRange {
    /// Acceleration structures are bound whole.
    AccelerationStructure,

    /// Images may be partially bound.
    Image(vk::ImageSubresourceRange),

    /// Buffers may be partially bound.
    Buffer(BufferSubresourceRange),
}

impl SubresourceRange {
    pub(super) fn as_image(&self) -> Option<&vk::ImageSubresourceRange> {
        if let Self::Image(subresource) = self {
            Some(subresource)
        } else {
            None
        }
    }

    pub(super) fn expect_image(&self) -> &vk::ImageSubresourceRange {
        self.as_image().expect("missing image subresource")
    }
}

impl From<BufferSubresourceRange> for SubresourceRange {
    fn from(subresource: BufferSubresourceRange) -> Self {
        Self::Buffer(subresource)
    }
}

impl From<()> for SubresourceRange {
    fn from(_: ()) -> Self {
        Self::AccelerationStructure
    }
}

impl From<ImageViewInfo> for SubresourceRange {
    fn from(subresource: ImageViewInfo) -> Self {
        Self::Image(subresource.into())
    }
}

impl From<vk::ImageSubresourceRange> for SubresourceRange {
    fn from(subresource: vk::ImageSubresourceRange) -> Self {
        Self::Image(subresource)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SubresourceAccess {
    pub access: AccessType,
    pub subresource: SubresourceRange,
}

/// Describes the interpretation of a resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ViewInfo {
    /// Acceleration structures are always whole resources.
    AccelerationStructure,

    /// Images may be interpreted as differently formatted images.
    Image(ImageViewInfo),

    /// Buffers may be interpreted as subregions of the same buffer.
    Buffer(BufferSubresourceRange),
}

impl ViewInfo {
    pub(crate) fn as_buffer(&self) -> Option<&BufferSubresourceRange> {
        match self {
            Self::Buffer(info) => Some(info),
            _ => None,
        }
    }

    pub(crate) fn as_image(&self) -> Option<&ImageViewInfo> {
        match self {
            Self::Image(info) => Some(info),
            _ => None,
        }
    }

    pub(crate) fn expect_buffer(&self) -> &BufferSubresourceRange {
        self.as_buffer().expect("missing buffer view info")
    }

    pub(crate) fn expect_image(&self) -> &ImageViewInfo {
        self.as_image().expect("missing image view info")
    }
}

impl From<()> for ViewInfo {
    fn from(_: ()) -> Self {
        Self::AccelerationStructure
    }
}

impl From<BufferSubresourceRange> for ViewInfo {
    fn from(info: BufferSubresourceRange) -> Self {
        Self::Buffer(info)
    }
}

impl From<ImageViewInfo> for ViewInfo {
    fn from(info: ImageViewInfo) -> Self {
        Self::Image(info)
    }
}

impl From<Range<vk::DeviceSize>> for ViewInfo {
    fn from(range: Range<vk::DeviceSize>) -> Self {
        Self::Buffer(BufferSubresourceRange {
            start: range.start,
            end: range.end,
        })
    }
}

mod private {
    use crate::{AnyResource, Node};

    pub(crate) trait SubresourceSealed: Sized {
        fn info(&self, resources: &[AnyResource]) -> <Self as super::Subresource>::Info
        where
            Self: Node + super::Subresource;

        fn range(&self, resources: &[AnyResource]) -> <Self as super::Subresource>::Range
        where
            Self: Node + super::Subresource;
    }
}
