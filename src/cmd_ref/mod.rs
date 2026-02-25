//! Strongly-typed rendering commands.

mod bind;
mod cmd_buf;
mod compute;
mod graphic;
mod pipeline;
mod ray_trace;

pub use self::{
    cmd_buf::{
        BuildAccelerationStructureIndirectInfo, BuildAccelerationStructureInfo, CommandBufferRef,
        UpdateAccelerationStructureIndirectInfo, UpdateAccelerationStructureInfo,
    },
    pipeline::PipelineRef,
};

use {
    self::bind::BindCommand,
    super::{
        AccelerationStructureLeaseNode, AccelerationStructureNode, AnyAccelerationStructureNode,
        AnyBufferNode, AnyImageNode, BindGraph, Bound, BufferLeaseNode, BufferNode, Command,
        Execution, ExecutionFunction, Graph, ImageLeaseNode, ImageNode, Node, Resource,
        SwapchainImageNode,
    },
    crate::driver::{
        CommandBuffer,
        accel_struct::{AccelerationStructure, AccelerationStructureRange},
        buffer::{Buffer, BufferSubresourceRange},
        image::{Image, ImageViewInfo},
    },
    ash::vk,
    std::ops::{Index, Range},
    vk_sync::AccessType,
};

/// Alias for the index of a framebuffer attachment.
pub type AttachmentIndex = u32;

/// Alias for the binding index of a shader descriptor.
pub type BindingIndex = u32;

/// Alias for the binding offset of a shader descriptor array element.
pub type BindingOffset = u32;

/// Alias for the descriptor set index of a shader descriptor.
pub type DescriptorSetIndex = u32;

/// A general render pass which may contain acceleration structure commands, general commands, or
/// have pipeline bound to then record commands specific to those pipeline types.
pub struct CommandRef<'a> {
    pub(super) cmd_idx: usize,
    pub(super) exec_idx: usize,
    pub(super) graph: &'a mut Graph,
}

impl<'a> CommandRef<'a> {
    pub(super) fn new(graph: &'a mut Graph) -> Self {
        let cmd_idx = graph.cmds.len();
        graph.cmds.push(Command {
            execs: vec![Default::default()], // We start off with a default execution!
            name: None,
        });

        Self {
            cmd_idx,
            exec_idx: 0,
            graph,
        }
    }

    fn cmd(&self) -> &Command {
        &self.graph.cmds[self.cmd_idx]
    }

    fn cmd_mut(&mut self) -> &mut Command {
        &mut self.graph.cmds[self.cmd_idx]
    }

    /// Binds a Vulkan buffer, image, or acceleration structure resource to the graph associated
    /// with this command.
    ///
    /// Bound nodes may be used in passes for pipeline and shader operations.
    pub fn bind_resource<R>(&mut self, resource: R) -> R::Node
    where
        R: BindGraph,
    {
        self.graph.bind_resource(resource)
    }

    /// Binds a [`ComputePipeline`], [`GraphicPipeline`], or [`RayTracePipeline`] to the current
    /// pass, allowing for strongly typed access to the related functions.
    pub fn bind_pipeline<P>(self, pipeline: P) -> P::Ref
    where
        P: BindCommand<'a>,
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

    fn push_execute(&mut self, func: impl FnOnce(&CommandBuffer, Resources<'_>) + Send + 'static) {
        let cmd = self.cmd_mut();
        let exec = {
            let last_exec = cmd.execs.last_mut().unwrap();
            last_exec.func = Some(ExecutionFunction(Box::new(func)));

            Execution {
                pipeline: last_exec.pipeline.clone(),
                ..Default::default()
            }
        };

        cmd.execs.push(exec);
        self.exec_idx += 1;
    }

    fn push_node_access(
        &mut self,
        node: impl Node,
        access: AccessType,
        subresource: SubresourceRange,
    ) {
        let node_idx = node.index();

        assert!(self.graph.resources.get(node_idx).is_some());

        let access = SubresourceAccess {
            access,
            subresource,
        };
        self.cmd_mut()
            .execs
            .last_mut()
            .unwrap()
            .accesses
            .entry(node_idx)
            .and_modify(|accesses| accesses.push(access))
            .or_insert(vec![access]);
    }

    /// Begin recording an acceleration structure command buffer.
    ///
    /// This is the entry point for building and updating an [`AccelerationStructure`] instance.
    ///
    /// The provided closure allows you to run any Vulkan code, or interoperate with other Vulkan
    /// code and interfaces.
    pub fn record_cmd_buf(
        mut self,
        func: impl FnOnce(CommandBufferRef<'_>, Resources<'_>) + Send + 'static,
    ) -> Self {
        self.push_execute(move |cmd_buf, resources| {
            func(CommandBufferRef { cmd_buf, resources }, resources);
        });

        self
    }

    /// Returns a borrow of the original Vulkan resource (buffer, image or acceleration structure)
    /// which the given node represents.
    pub fn resource<N>(&self, node: N) -> &N::Resource
    where
        N: Bound,
    {
        self.graph.resource(node)
    }

    /// Informs the command that the next recorded command buffer will read or write `node` using
    /// `access`.
    ///
    /// This function must be called for `node` before it is used within a `record_`-function.
    pub fn resource_access<N>(mut self, node: N, access: AccessType) -> Self
    where
        N: Node + View,
        SubresourceRange: From<N::Range>,
    {
        self.set_resource_access(node, access);
        self
    }

    /// Sets a debugging name, but only in debug builds
    pub fn set_debug_name(&mut self, name: impl Into<String>) -> &mut Self {
        #[cfg(debug_assertions)]
        {
            self.cmd_mut().name = Some(name.into());
        }

        self
    }

    /// Informs the command that the next recorded command buffer will read or write `node` using
    /// `access`.
    ///
    /// This function must be called for `node` before it is used within a `record_`-function.
    pub fn set_resource_access<N>(&mut self, node: N, access: AccessType)
    where
        N: Node + View,
        SubresourceRange: From<N::Range>,
    {
        let subresource = node.default_range(&self.graph.resources).into();
        self.push_node_access(node, access, subresource);
    }

    /// Informs the command that the next recorded command buffer will read or write the
    /// `subresource` of `node` using `access`.
    ///
    /// This function must be called for `node` before it is used within a `record_`-function.
    pub fn set_subresource_access<N>(
        &mut self,
        node: N,
        subresource: impl Into<N::Range>,
        access: AccessType,
    ) where
        N: Node + View,
        SubresourceRange: From<N::Range>,
    {
        let subresource = subresource.into().into();
        self.push_node_access(node, access, subresource);
    }

    /// Informs the command that the next recorded command buffer will read or write the
    /// `subresource` of `node` using `access`.
    ///
    /// This function must be called for `node` before it is used within a `record_`-function.
    pub fn subresource_access<N>(
        mut self,
        node: N,
        subresource: impl Into<N::Range>,
        access: AccessType,
    ) -> Self
    where
        N: Node + View,
        SubresourceRange: From<N::Range>,
    {
        self.set_subresource_access(node, subresource, access);
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
pub enum Descriptor {
    /// An array binding which includes an `offset` argument for the bound element.
    ArrayBinding(DescriptorSetIndex, BindingIndex, BindingOffset),

    /// A single binding.
    Binding(DescriptorSetIndex, BindingIndex),
}

impl Descriptor {
    pub(super) fn into_tuple(self) -> (DescriptorSetIndex, BindingIndex, BindingOffset) {
        match self {
            Self::ArrayBinding(descriptor_set_idx, binding_idx, binding_offset) => {
                (descriptor_set_idx, binding_idx, binding_offset)
            }
            Self::Binding(descriptor_set_idx, binding_idx) => (descriptor_set_idx, binding_idx, 0),
        }
    }

    pub(super) fn set(self) -> DescriptorSetIndex {
        let (res, _, _) = self.into_tuple();
        res
    }
}

impl From<BindingIndex> for Descriptor {
    fn from(val: BindingIndex) -> Self {
        Self::Binding(0, val)
    }
}

impl From<(DescriptorSetIndex, BindingIndex)> for Descriptor {
    fn from(tuple: (DescriptorSetIndex, BindingIndex)) -> Self {
        Self::Binding(tuple.0, tuple.1)
    }
}

impl From<(BindingIndex, [BindingOffset; 1])> for Descriptor {
    fn from(tuple: (BindingIndex, [BindingOffset; 1])) -> Self {
        Self::ArrayBinding(0, tuple.0, tuple.1[0])
    }
}

impl From<(DescriptorSetIndex, BindingIndex, [BindingOffset; 1])> for Descriptor {
    fn from(tuple: (DescriptorSetIndex, BindingIndex, [BindingOffset; 1])) -> Self {
        Self::ArrayBinding(tuple.0, tuple.1, tuple.2[0])
    }
}

/// An indexable structure will provides access to Vulkan resources inside a command closure.
///
/// This type is available while recording commands in the following closures:
///
/// - [`PassRef::record_accel_struct`] for building and updating acceleration structures
/// - [`PassRef::record_cmd_buf`] for general command streams
/// - [`PipelineRef::record_pipeline`] for dispatched compute operations
/// - [`PipelineRef::record_pipeline`] for raster drawing operations, such as triangle streams
/// - [`PipelineRef::record_pipeline`] for ray-traced operations
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
/// # use vk_graph::driver::image::{Image, ImageInfo};
/// # use vk_graph::Graph;
/// # use vk_graph::node::ImageNode;
/// # fn main() -> Result<(), DriverError> {
/// # let device = Device::new(DeviceInfo::default())?;
/// # let info = ImageInfo::image_2d(32, 32, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::SAMPLED);
/// # let image = Image::create(&device, info)?;
/// # let mut my_graph = Graph::default();
/// # let my_image_node = my_graph.bind_resource(image);
/// my_graph
///     .begin_cmd()
///     .debug_name("custom vulkan commands")
///     .record_cmd_buf(move |cmd_buf, resources| {
///         let my_image: &Image = &resources[my_image_node];
///
///         assert_ne!(my_image.handle, vk::Image::null());
///         assert_eq!(my_image.info.width, 32);
///     });
/// # Ok(()) }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Resources<'a> {
    #[cfg(debug_assertions)]
    exec: &'a Execution,

    resources: &'a [Resource],
}

impl<'a> Resources<'a> {
    pub(super) fn new(
        resources: &'a [Resource],
        #[cfg(debug_assertions)] exec: &'a Execution,
    ) -> Self {
        Self {
            #[cfg(debug_assertions)]
            exec,
            resources,
        }
    }

    // TODO...
    fn resource(&self, node_idx: usize) -> &Resource {
        // You must have called read or write for this node on this execution before indexing
        // into the bindings data!
        //
        // Why: Code that attempts to access this function is attempting to get access to the Vulkan
        // resource (buffer, image, or acceleration structure). In order to access any resources the
        // access type must first be specified so the correct barriers may be added.
        debug_assert!(
            self.exec.accesses.contains_key(&node_idx),
            "unexpected node access: call access, read, or write first"
        );

        &self.resources[node_idx]
    }
}

macro_rules! index {
    ($name:ident, $handle:ident) => {
        paste::paste! {
            impl<'a> Index<[<$name Node>]> for Resources<'a>
            {
                type Output = $handle;

                fn index(&self, node: [<$name Node>]) -> &Self::Output {
                    &*self.resource(node.idx).[<as_ $name:snake>]().unwrap()
                }
            }
        }
    };
}

// Allow indexing the Nodes data during command execution:
// (This gets you access to the driver images or other resources)
index!(AccelerationStructure, AccelerationStructure);
index!(AccelerationStructureLease, AccelerationStructure);
index!(Buffer, Buffer);
index!(BufferLease, Buffer);
index!(Image, Image);
index!(ImageLease, Image);
index!(SwapchainImage, Image);

impl Index<AnyAccelerationStructureNode> for Resources<'_> {
    type Output = AccelerationStructure;

    fn index(&self, node: AnyAccelerationStructureNode) -> &Self::Output {
        let node_idx = node.index();
        let resource = self.resource(node_idx);

        resource.as_driver_accel_struct().unwrap()
    }
}

impl Index<AnyBufferNode> for Resources<'_> {
    type Output = Buffer;

    fn index(&self, node: AnyBufferNode) -> &Self::Output {
        let node_idx = node.index();
        let resource = self.resource(node_idx);

        resource.as_driver_buffer().unwrap()
    }
}

impl Index<AnyImageNode> for Resources<'_> {
    type Output = Image;

    fn index(&self, node: AnyImageNode) -> &Self::Output {
        let node_idx = node.index();
        let resource = self.resource(node_idx);

        resource.as_driver_image().unwrap()
    }
}

#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub enum SubresourceRange {
    /// Acceleration structures are bound whole.
    AccelerationStructure(AccelerationStructureRange),

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
}

impl From<AccelerationStructureRange> for SubresourceRange {
    fn from(subresource: AccelerationStructureRange) -> Self {
        Self::AccelerationStructure(subresource)
    }
}

impl From<BufferSubresourceRange> for SubresourceRange {
    fn from(subresource: BufferSubresourceRange) -> Self {
        Self::Buffer(subresource)
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

/// Allows for a resource to be reinterpreted as differently formatted data.
#[doc(hidden)]
pub trait View {
    /// The information about the resource when bound directly to shader descriptors.
    type Info;

    /// The information about the resource when used indirectly by any part of a graph.
    type Range;

    fn default_range(&self, _: &[Resource]) -> Self::Range
    where
        Self: Node;

    fn default_info(&self, _: &[Resource]) -> Self::Info
    where
        Self: Node;
}

macro_rules! view_accel_struct {
    ($name:ident) => {
        impl View for $name {
            type Info = Self::Range;
            type Range = AccelerationStructureRange;

            fn default_range(&self, _: &[Resource]) -> Self::Range
            where
                Self: Node,
            {
                Self::Range::default()
            }

            fn default_info(&self, resources: &[Resource]) -> Self::Info
            where
                Self: Node,
            {
                self.default_range(resources)
            }
        }
    };
}

view_accel_struct!(AnyAccelerationStructureNode);
view_accel_struct!(AccelerationStructureLeaseNode);
view_accel_struct!(AccelerationStructureNode);

macro_rules! view_buffer {
    ($name:ident) => {
        impl View for $name {
            type Info = Self::Range;
            type Range = BufferSubresourceRange;

            fn default_range(&self, resources: &[Resource]) -> Self::Range
            where
                Self: Node,
            {
                let idx = self.index();

                resources[idx].as_buffer().unwrap().info.into()
            }

            fn default_info(&self, resources: &[Resource]) -> Self::Info
            where
                Self: Node,
            {
                self.default_range(resources)
            }
        }
    };
}

view_buffer!(AnyBufferNode);
view_buffer!(BufferLeaseNode);
view_buffer!(BufferNode);

macro_rules! view_image {
    ($name:ident) => {
        impl View for $name {
            type Info = ImageViewInfo;
            type Range = vk::ImageSubresourceRange;

            fn default_range(&self, resources: &[Resource]) -> Self::Range
            where
                Self: Node,
            {
                self.default_info(resources).into()
            }

            fn default_info(&self, resources: &[Resource]) -> Self::Info
            where
                Self: Node,
            {
                let idx = self.index();

                resources[idx].as_image().unwrap().info.into()
            }
        }
    };
}

view_image!(AnyImageNode);
view_image!(ImageLeaseNode);
view_image!(ImageNode);
view_image!(SwapchainImageNode);

/// Describes the interpretation of a resource.
#[derive(Debug)]
#[doc(hidden)]
pub enum ViewInfo {
    /// Acceleration structures are always whole resources.
    AccelerationStructure(AccelerationStructureRange),

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
}

impl From<AccelerationStructureRange> for ViewInfo {
    fn from(info: AccelerationStructureRange) -> Self {
        Self::AccelerationStructure(info)
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

#[allow(unused)]
mod deprecated {
    use {
        crate::{
            cmd_ref::{CommandBufferRef, CommandRef, Resources, SubresourceRange, View},
            deprecated::Info,
            node::Node,
        },
        ash::vk,
        vk_sync::AccessType,
    };

    impl<'a> CommandRef<'a> {
        #[deprecated = "use set_resource_access function"]
        #[doc(hidden)]
        pub fn access_node_mut<N>(&mut self, node: N, access: AccessType) -> &mut Self
        where
            N: Node + View,
            SubresourceRange: From<N::Range>,
        {
            self.set_resource_access(node, access);
            self
        }

        #[deprecated = "use resource_access function"]
        #[doc(hidden)]
        pub fn access_resource<N>(mut self, node: N, access: AccessType) -> Self
        where
            N: Node + View,
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
            N: Node + View,
            SubresourceRange: From<N::Range>,
        {
            self.subresource_access(node, subresource, access)
        }

        #[deprecated = "use device_address function of resource function result"]
        #[doc(hidden)]
        pub fn node_device_address(&self, node: impl Node) -> vk::DeviceAddress {
            let idx = node.index();

            self.graph.resources[idx]
                .as_driver_buffer()
                .unwrap()
                .device_address()
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
        pub fn record_accel_struct(
            mut self,
            func: impl FnOnce(CommandBufferRef<'_>, Resources<'_>) + Send + 'static,
        ) -> Self {
            self.push_execute(move |cmd_buf, resources| {
                func(CommandBufferRef { cmd_buf, resources }, resources);
            });

            self
        }
    }
}
