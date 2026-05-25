//! Strongly-typed [`Graph`] commands.

mod cmd_buf;
mod compute;
mod graphic;
mod pipeline;
mod ray_trace;

pub use self::{
    cmd_buf::{
        BuildAccelerationStructureIndirectInfo, BuildAccelerationStructureInfo, CommandBuffer,
        UpdateAccelerationStructureIndirectInfo, UpdateAccelerationStructureInfo,
    },
    compute::ComputeCommandBuffer,
    graphic::{ClearColorValue, GraphicCommandBuffer, LoadOp, StoreOp},
    pipeline::{Pipeline, PipelineCommand},
    ray_trace::RayTraceCommandBuffer,
};

use {
    super::{
        AccelerationStructureLeaseNode, AccelerationStructureNode, AnyAccelerationStructureNode,
        AnyBufferNode, AnyImageNode, AnyResource, BufferLeaseNode, BufferNode, CommandData,
        Execution, ExecutionFunction, Graph, ImageLeaseNode, ImageNode, Node, Resource,
        SwapchainImageNode,
    },
    crate::driver::{buffer::BufferSubresourceRange, image::ImageViewInfo},
    ash::vk,
    std::ops::Range,
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
///    ([`Self::record_cmd_buf`])
/// 1. Bind shader pipelines ([`Self::bind_pipeline`])
///
/// When bound, a shader pipeline consumes the `Command` and returns a [`PipelineCommand`] which
/// provides command recording functions specific to each pipeline type.
pub struct Command<'a> {
    pub(super) cmd_idx: usize,
    pub(super) exec_idx: usize,
    pub(super) graph: &'a mut Graph,
}

impl<'a> Command<'a> {
    pub(super) fn new(graph: &'a mut Graph) -> Self {
        let cmd_idx = graph.cmds.len();
        graph.cmds.push(CommandData {
            execs: vec![Default::default()], // We start off with a default execution!
            name: None,
        });

        Self {
            cmd_idx,
            exec_idx: 0,
            graph,
        }
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
    /// `P`|`P::Command`
    /// -|-
    /// [`ComputePipeline`](crate::driver::compute::ComputePipeline)|[`PipelineCommand<'_, ComputePipeline>`]
    /// [`GraphicPipeline`](crate::driver::graphic::GraphicPipeline)|[`PipelineCommand<'_, GraphicPipeline>`]
    /// [`RayTracePipeline`](crate::driver::ray_trace::RayTracePipeline)|[`PipelineCommand<'_, RayTracePipeline>`]
    pub fn bind_pipeline<P>(self, pipeline: P) -> P::Command
    where
        P: Pipeline<'a>,
    {
        pipeline.bind_cmd(self)
    }

    /// Sets a debugging name, but only in debug builds
    pub fn debug_name(mut self, name: impl Into<String>) -> Self {
        self.set_debug_name(name);
        self
    }

    /// Finalize the recording of this command and return to the `Graph` where you may record
    /// additional commands.
    pub fn end_cmd(self) -> &'a mut Graph {
        // If nothing was done in this pass we can just ignore it
        if self.exec_idx == 0 {
            self.graph.cmds.pop();
        }

        self.graph
    }

    fn push_exec(&mut self, func: impl FnOnce(CommandBuffer) + Send + 'static) {
        let cmd = self.cmd_mut();
        let exec = {
            let last_exec = cmd.expect_last_exec_mut();
            last_exec.func = Some(ExecutionFunction(Box::new(func)));

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
        let node_idx = resource_node.index();

        debug_assert!(self.graph.resources.get(node_idx).is_some());

        let access = SubresourceAccess {
            access,
            subresource,
        };
        self.cmd_mut()
            .expect_last_exec_mut()
            .accesses
            .entry(node_idx)
            .and_modify(|accesses| accesses.push(access))
            .or_insert(vec![access]);
    }

    /// Begin recording a general-purpose command buffer.
    ///
    /// This is the entry point for building and updating an
    /// [`AccelerationStructure`](crate::driver::accel_struct::AccelerationStructure) instance.
    ///
    /// The provided closure allows you to run any Vulkan code, or interoperate with other Vulkan
    /// code and interfaces.
    pub fn record_cmd_buf(mut self, func: impl FnOnce(CommandBuffer<'_>) + Send + 'static) -> Self {
        self.record_cmd_buf_mut(func);
        self
    }

    /// Begin recording a general-purpose command buffer.
    ///
    /// This is the entry point for building and updating an
    /// [`AccelerationStructure`](crate::driver::accel_struct::AccelerationStructure) instance.
    ///
    /// The provided closure allows you to run any Vulkan code, or interoperate with other Vulkan
    /// code and interfaces.
    pub fn record_cmd_buf_mut(&mut self, func: impl FnOnce(CommandBuffer<'_>) + Send + 'static) {
        self.push_exec(move |cmd_buf| {
            func(cmd_buf);
        });
    }

    /// Returns a borrow of the original Vulkan resource (buffer, image or acceleration structure)
    /// which the given bound resource node represents.
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: Node,
    {
        self.graph.resource(resource_node)
    }

    /// Informs the command that the next recorded command buffer will read or write `resource_node`
    /// using `access`.
    ///
    /// An access function must be called for `resource_node` before it is used within a
    /// `record_`-function.
    pub fn resource_access<N>(mut self, resource_node: N, access: AccessType) -> Self
    where
        N: Node + Subresource,
        SubresourceRange: From<N::Range>,
    {
        self.set_resource_access(resource_node, access);
        self
    }

    /// Sets a debugging name, but only in debug builds.
    pub fn set_debug_name(&mut self, name: impl Into<String>) -> &mut Self {
        #[cfg(debug_assertions)]
        {
            self.cmd_mut().name = Some(name.into());
        }

        self
    }

    /// Informs the command that the next recorded command buffer will read or write `resource_node`
    /// using `access`.
    ///
    /// An access function must be called for `resource_node` before it is used within a
    /// `record_`-function.
    pub fn set_resource_access<N>(&mut self, resource_node: N, access: AccessType)
    where
        N: Node + Subresource,
        SubresourceRange: From<N::Range>,
    {
        let whole_resource = resource_node.range(&self.graph.resources);
        let subresource = SubresourceRange::from(whole_resource);

        self.push_subresource_access(resource_node, subresource, access);
    }

    /// Informs the command that the next recorded command buffer will read or write the
    /// `subresource` of `resource_node` using `access`.
    ///
    /// An access function must be called for `resource_node` before it is used within a
    /// `record_`-function.
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

    /// Informs the command that the next recorded command buffer will read or write the
    /// `subresource` of `resource` using `access`.
    ///
    /// An access function must be called for `resource` before it is used within a `record_`-function.
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
pub struct Descriptor {
    /// The value of the descriptor binding decoration applied to the variable.
    pub binding: u32,

    /// An array-element offset applied to this descriptor.
    pub offset: u32,

    /// An optional descriptor set index value.
    pub set: u32,
}

impl Descriptor {
    pub(super) fn into_tuple(self) -> (DescriptorSetIndex, BindingIndex, BindingOffset) {
        (self.set, self.binding, self.offset)
    }

    pub(super) fn set(self) -> DescriptorSetIndex {
        let (res, _, _) = self.into_tuple();
        res
    }
}

impl From<BindingIndex> for Descriptor {
    fn from(binding: BindingIndex) -> Self {
        Self {
            binding,
            offset: 0,
            set: 0,
        }
    }
}

impl From<(DescriptorSetIndex, BindingIndex)> for Descriptor {
    fn from((set, binding): (DescriptorSetIndex, BindingIndex)) -> Self {
        Self {
            binding,
            offset: 0,
            set,
        }
    }
}

impl From<(BindingIndex, [BindingOffset; 1])> for Descriptor {
    fn from((binding, [offset]): (BindingIndex, [BindingOffset; 1])) -> Self {
        Self {
            binding,
            offset,
            set: 0,
        }
    }
}

impl From<(DescriptorSetIndex, BindingIndex, [BindingOffset; 1])> for Descriptor {
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
pub trait Subresource {
    /// The information about the subresource when bound directly to shader descriptors.
    type Info;

    /// The information about the subresource when used indirectly by any part of a graph.
    type Range;

    #[doc(hidden)]
    fn info(&self, _: &[AnyResource]) -> Self::Info
    where
        Self: Node;

    #[doc(hidden)]
    fn range(&self, _: &[AnyResource]) -> Self::Range
    where
        Self: Node;
}

macro_rules! view_accel_struct {
    ($name:ident) => {
        impl Subresource for $name {
            type Info = Self::Range;
            type Range = ();

            fn info(&self, _: &[AnyResource]) -> Self::Info
            where
                Self: Node,
            {
            }

            fn range(&self, _: &[AnyResource]) -> Self::Range
            where
                Self: Node,
            {
            }
        }
    };
}

view_accel_struct!(AnyAccelerationStructureNode);
view_accel_struct!(AccelerationStructureLeaseNode);
view_accel_struct!(AccelerationStructureNode);

macro_rules! view_buffer {
    ($name:ident) => {
        impl Subresource for $name {
            type Info = Self::Range;
            type Range = BufferSubresourceRange;

            fn info(&self, resources: &[AnyResource]) -> Self::Info
            where
                Self: Node,
            {
                self.range(resources)
            }

            fn range(&self, resources: &[AnyResource]) -> Self::Range
            where
                Self: Node,
            {
                let idx = self.index();

                resources[idx].expect_buffer().info.into()
            }
        }
    };
}

view_buffer!(AnyBufferNode);
view_buffer!(BufferLeaseNode);
view_buffer!(BufferNode);

macro_rules! view_image {
    ($name:ident) => {
        impl Subresource for $name {
            type Info = ImageViewInfo;
            type Range = vk::ImageSubresourceRange;

            fn info(&self, resources: &[AnyResource]) -> Self::Info
            where
                Self: Node,
            {
                let idx = self.index();

                resources[idx].expect_image().info.into()
            }

            fn range(&self, resources: &[AnyResource]) -> Self::Range
            where
                Self: Node,
            {
                self.info(resources).into()
            }
        }
    };
}

view_image!(AnyImageNode);
view_image!(ImageLeaseNode);
view_image!(ImageNode);
view_image!(SwapchainImageNode);

#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub enum SubresourceRange {
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
#[derive(Debug)]
#[doc(hidden)]
pub enum ViewInfo {
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

#[allow(deprecated)]
#[allow(unused)]
mod deprecated {
    use {
        crate::{
            Graph, Node, Resource,
            cmd::{Command, CommandBuffer, Subresource, SubresourceRange},
            deprecated::Info,
        },
        ash::vk,
        vk_sync::AccessType,
    };

    impl<'a> Command<'a> {
        #[deprecated = "use resource_access function"]
        #[doc(hidden)]
        pub fn access_node<N>(mut self, node: N, access: AccessType) -> Self
        where
            N: Node + Subresource,
            SubresourceRange: From<N::Range>,
        {
            self.resource_access(node, access)
        }

        #[deprecated = "use set_resource_access function"]
        #[doc(hidden)]
        pub fn access_node_mut<N>(&mut self, node: N, access: AccessType) -> &mut Self
        where
            N: Node + Subresource,
            SubresourceRange: From<N::Range>,
        {
            self.set_resource_access(node, access);
            self
        }

        #[deprecated = "use subresource_access function"]
        #[doc(hidden)]
        pub fn access_node_subrange<N>(
            mut self,
            node: N,
            access: AccessType,
            subresource: impl Into<N::Range>,
        ) -> Self
        where
            N: Node + Subresource,
            SubresourceRange: From<N::Range>,
        {
            self.access_node_subrange_mut(node, access, subresource);
            self
        }

        #[deprecated = "use set_subresource_access function"]
        #[doc(hidden)]
        pub fn access_node_subrange_mut<N>(
            &mut self,
            node: N,
            access: AccessType,
            subresource: impl Into<N::Range>,
        ) -> &mut Self
        where
            N: Node + Subresource,
            SubresourceRange: From<N::Range>,
        {
            self.set_subresource_access(node, subresource, access);
            self
        }

        #[deprecated = "use resource_access function"]
        #[doc(hidden)]
        pub fn access_resource<N>(mut self, node: N, access: AccessType) -> Self
        where
            N: Node + Subresource,
            SubresourceRange: From<N::Range>,
        {
            self.resource_access(node, access)
        }

        #[deprecated = "use subresource_access function"]
        #[doc(hidden)]
        pub fn access_subresource<N>(
            mut self,
            node: N,
            subresource: impl Into<N::Range>,
            access: AccessType,
        ) -> Self
        where
            N: Node + Subresource,
            SubresourceRange: From<N::Range>,
        {
            self.subresource_access(node, subresource, access)
        }

        #[deprecated = "use bind_resource function"]
        #[doc(hidden)]
        pub fn bind_node<R>(&mut self, resource: R) -> R::Node
        where
            R: Resource,
        {
            self.bind_resource(resource)
        }

        #[deprecated = "use device_address function of resource function result"]
        #[doc(hidden)]
        pub fn node_device_address(&self, node: impl Node) -> vk::DeviceAddress {
            let idx = node.index();

            self.graph.resources[idx].expect_buffer().device_address()
        }

        #[deprecated = "dereference info field of resource function result"]
        #[doc(hidden)]
        pub fn node_info<N>(&self, node: N) -> N::Type
        where
            N: Node + Info,
        {
            node.info(&self.graph.resources)
        }

        #[deprecated = "use record_cmd_buf function"]
        #[doc(hidden)]
        pub fn record_acceleration(
            mut self,
            func: impl FnOnce(CommandBuffer<'_>, ()) + Send + 'static,
        ) -> Self {
            self.push_exec(|cmd_buf| {
                func(cmd_buf, ());
            });

            self
        }

        #[deprecated = "use end_cmd function"]
        #[doc(hidden)]
        pub fn submit_pass(self) -> &'a mut Graph {
            self.end_cmd()
        }
    }
}
