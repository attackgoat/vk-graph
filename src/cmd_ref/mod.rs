//! Strongly-typed rendering commands.

mod accel_struct;
mod compute;
mod graphic;
mod pipeline;
mod ray_trace;
mod view;

pub use self::{
    accel_struct::{
        AccelerationStructureRef, BuildAccelerationStructureIndirectInfo,
        BuildAccelerationStructureInfo, UpdateAccelerationStructureIndirectInfo,
        UpdateAccelerationStructureInfo,
    },
    pipeline::PipelineRef,
    view::{View, ViewType},
};

use {
    super::{
        AccelerationStructureLeaseNode, AccelerationStructureNode, AnyAccelerationStructureNode,
        AnyBufferNode, AnyImageNode, Bind, Binding, BufferLeaseNode, BufferNode, Command, Edge,
        Execution, ExecutionFunction, ExecutionPipeline, Graph, ImageLeaseNode, ImageNode, Info,
        Node, SwapchainImageNode,
    },
    crate::driver::{
        accel_struct::AccelerationStructure,
        buffer::{Buffer, BufferSubresourceRange},
        compute::ComputePipeline,
        device::Device,
        graphic::GraphicPipeline,
        image::{Image, ImageViewInfo},
        ray_trace::RayTracePipeline,
    },
    ash::vk,
    std::{marker::PhantomData, ops::Index},
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

/// Associated type trait which enables default values for read and write methods.
pub trait Access {
    /// The default `AccessType` for read operations, if not specified explicitly.
    const DEFAULT_READ: AccessType;

    /// The default `AccessType` for write operations, if not specified explicitly.
    const DEFAULT_WRITE: AccessType;
}

impl Access for ComputePipeline {
    const DEFAULT_READ: AccessType = AccessType::ComputeShaderReadOther;
    const DEFAULT_WRITE: AccessType = AccessType::ComputeShaderWrite;
}

impl Access for GraphicPipeline {
    const DEFAULT_READ: AccessType = AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer;
    const DEFAULT_WRITE: AccessType = AccessType::AnyShaderWrite;
}

impl Access for RayTracePipeline {
    const DEFAULT_READ: AccessType =
        AccessType::RayTracingShaderReadSampledImageOrUniformTexelBuffer;
    const DEFAULT_WRITE: AccessType = AccessType::AnyShaderWrite;
}

macro_rules! bind {
    ($name:ident) => {
        paste::paste! {
            impl<'a> Bind<CommandRef<'a>, PipelineRef<'a, [<$name Pipeline>]>> for &'a [<$name Pipeline>] {
                // TODO: Allow binding as explicit secondary command buffers? like with compute/raytrace stuff
                fn bind(self, mut cmd: CommandRef<'a>) -> PipelineRef<'a, [<$name Pipeline>]> {
                    let cmd_ref = cmd.as_mut();
                    if cmd_ref.execs.last().unwrap().pipeline.is_some() {
                        // Binding from PipelinePass -> PipelinePass (changing shaders)
                        cmd_ref.execs.push(Default::default());
                    }

                    cmd_ref.execs.last_mut().unwrap().pipeline = Some(ExecutionPipeline::$name(self.clone()));

                    PipelineRef {
                        __: PhantomData,
                        cmd,
                    }
                }
            }

            impl<'a> Bind<CommandRef<'a>, PipelineRef<'a, [<$name Pipeline>]>> for [<$name Pipeline>] {
                // TODO: Allow binding as explicit secondary command buffers? like with compute/raytrace stuff
                fn bind(self, mut cmd: CommandRef<'a>) -> PipelineRef<'a, [<$name Pipeline>]> {
                    let cmd_ref = cmd.as_mut();
                    if cmd_ref.execs.last().unwrap().pipeline.is_some() {
                        // Binding from PipelinePass -> PipelinePass (changing shaders)
                        cmd_ref.execs.push(Default::default());
                    }

                    cmd_ref.execs.last_mut().unwrap().pipeline = Some(ExecutionPipeline::$name(self));

                    PipelineRef {
                        __: PhantomData,
                        cmd,
                    }
                }
            }

            impl ExecutionPipeline {
                #[allow(unused)]
                pub(super) fn [<is_ $name:snake>](&self) -> bool {
                    matches!(self, Self::$name(_))
                }

                #[allow(unused)]
                pub(super) fn [<unwrap_ $name:snake>](&self) -> &[<$name Pipeline>] {
                    if let Self::$name(binding) = self {
                        &binding
                    } else {
                        panic!();
                    }
                }
            }
        }
    };
}

// Pipelines you can bind to a pass
bind!(Compute);
bind!(Graphic);
bind!(RayTrace);

/// An indexable structure will provides access to Vulkan smart-pointer resources inside a record
/// closure.
///
/// This type is available while recording commands in the following closures:
///
/// - [`PassRef::record_accel_struct`] for building and updating acceleration structures
/// - [`PassRef::record_cmd_buf`] for general command streams
/// - [`PipelineCommandRef::record_pipeline`] for dispatched compute operations
/// - [`PipelineCommandRef::record_pipeline`] for raster drawing operations, such as triangles streams
/// - [`PipelineCommandRef::record_ray_trace`] for ray-traced operations
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
/// # let my_image_node = my_graph.bind_node(image);
/// my_graph.begin_cmd().with_name("custom vulkan commands")
///         .record_cmd_buf(move |device, cmd_buf, bindings| {
///             let my_image = &bindings[my_image_node];
///
///             assert_ne!(my_image.handle, vk::Image::null());
///             assert_eq!(my_image.info.width, 32);
///         });
/// # Ok(()) }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Bindings<'a> {
    bindings: &'a [Binding],
    exec: &'a Execution,
}

impl<'a> Bindings<'a> {
    pub(super) fn new(bindings: &'a [Binding], exec: &'a Execution) -> Self {
        Self { bindings, exec }
    }

    fn binding_ref(&self, node_idx: usize) -> &Binding {
        // You must have called read or write for this node on this execution before indexing
        // into the bindings data!
        debug_assert!(
            self.exec.accesses.contains_key(&node_idx),
            "unexpected node access: call access, read, or write first"
        );

        &self.bindings[node_idx]
    }
}

macro_rules! index {
    ($name:ident, $handle:ident) => {
        paste::paste! {
            impl<'a> Index<[<$name Node>]> for Bindings<'a>
            {
                type Output = $handle;

                fn index(&self, node: [<$name Node>]) -> &Self::Output {
                    &*self.binding_ref(node.idx).[<as_ $name:snake>]().unwrap()
                }
            }
        }
    };
}

// Allow indexing the Bindings data during command execution:
// (This gets you access to the driver images or other resources)
index!(AccelerationStructure, AccelerationStructure);
index!(AccelerationStructureLease, AccelerationStructure);
index!(Buffer, Buffer);
index!(BufferLease, Buffer);
index!(Image, Image);
index!(ImageLease, Image);
index!(SwapchainImage, Image);

impl Index<AnyAccelerationStructureNode> for Bindings<'_> {
    type Output = AccelerationStructure;

    fn index(&self, node: AnyAccelerationStructureNode) -> &Self::Output {
        let node_idx = match node {
            AnyAccelerationStructureNode::AccelerationStructure(node) => node.idx,
            AnyAccelerationStructureNode::AccelerationStructureLease(node) => node.idx,
        };
        let binding = self.binding_ref(node_idx);

        match node {
            AnyAccelerationStructureNode::AccelerationStructure(_) => {
                binding.as_acceleration_structure().unwrap()
            }
            AnyAccelerationStructureNode::AccelerationStructureLease(_) => {
                binding.as_acceleration_structure_lease().unwrap()
            }
        }
    }
}

impl Index<AnyBufferNode> for Bindings<'_> {
    type Output = Buffer;

    fn index(&self, node: AnyBufferNode) -> &Self::Output {
        let node_idx = match node {
            AnyBufferNode::Buffer(node) => node.idx,
            AnyBufferNode::BufferLease(node) => node.idx,
        };
        let binding = self.binding_ref(node_idx);

        match node {
            AnyBufferNode::Buffer(_) => binding.as_buffer().unwrap(),
            AnyBufferNode::BufferLease(_) => binding.as_buffer_lease().unwrap(),
        }
    }
}

impl Index<AnyImageNode> for Bindings<'_> {
    type Output = Image;

    fn index(&self, node: AnyImageNode) -> &Self::Output {
        let node_idx = match node {
            AnyImageNode::Image(node) => node.idx,
            AnyImageNode::ImageLease(node) => node.idx,
            AnyImageNode::SwapchainImage(node) => node.idx,
        };
        let binding = self.binding_ref(node_idx);

        match node {
            AnyImageNode::Image(_) => binding.as_image().unwrap(),
            AnyImageNode::ImageLease(_) => binding.as_image_lease().unwrap(),
            AnyImageNode::SwapchainImage(_) => binding.as_swapchain_image().unwrap(),
        }
    }
}

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

    /// Informs the pass that the next recorded command buffer will read or write the given `node`
    /// using `access`.
    ///
    /// This function must be called for `node` before it is read or written within a `record`
    /// function. For general purpose access, see [`PassRef::read_node`] or [`PassRef::write_node`].
    pub fn access_node(mut self, node: impl Node + Info, access: AccessType) -> Self {
        self.access_node_mut(node, access);

        self
    }

    /// Informs the pass that the next recorded command buffer will read or write the given `node`
    /// using `access`.
    ///
    /// This function must be called for `node` before it is read or written within a `record`
    /// function. For general purpose access, see [`PassRef::read_node_mut`] or
    /// [`PassRef::write_node_mut`].
    pub fn access_node_mut(&mut self, node: impl Node + Info, access: AccessType) {
        self.assert_bound_graph_node(node);

        let idx = node.index();
        let binding = &self.graph.bindings[idx];

        let node_access_range = if let Some(buf) = binding.as_driver_buffer() {
            Subresource::Buffer((0..buf.info.size).into())
        } else if let Some(image) = binding.as_driver_image() {
            Subresource::Image(image.info.default_view_info().into())
        } else {
            Subresource::AccelerationStructure
        };

        self.push_node_access(node, access, node_access_range);
    }

    /// Informs the pass that the next recorded command buffer will read or write the `subresource`
    /// of `node` using `access`.
    ///
    /// This function must be called for `node` before it is read or written within a `record`
    /// function. For general purpose access, see [`PassRef::read_node`] or [`PassRef::write_node`].
    pub fn access_node_subrange<N>(
        mut self,
        node: N,
        access: AccessType,
        subresource: impl Into<N::Subresource>,
    ) -> Self
    where
        N: View,
    {
        self.access_node_subrange_mut(node, access, subresource);

        self
    }

    /// Informs the pass that the next recorded command buffer will read or write the `subresource`
    /// of `node` using `access`.
    ///
    /// This function must be called for `node` before it is read or written within a `record`
    /// function. For general purpose access, see [`PassRef::read_node`] or [`PassRef::write_node`].
    pub fn access_node_subrange_mut<N>(
        &mut self,
        node: N,
        access: AccessType,
        subresource: impl Into<N::Subresource>,
    ) where
        N: View,
    {
        self.push_node_access(node, access, subresource.into().into());
    }

    fn as_mut(&mut self) -> &mut Command {
        &mut self.graph.cmds[self.cmd_idx]
    }

    fn as_ref(&self) -> &Command {
        &self.graph.cmds[self.cmd_idx]
    }

    fn assert_bound_graph_node(&self, node: impl Node) {
        let idx = node.index();

        assert!(self.graph.bindings[idx].is_bound());
    }

    /// Binds a Vulkan acceleration structure, buffer, or image to the graph associated with this
    /// pass.
    ///
    /// Bound nodes may be used in passes for pipeline and shader operations.
    pub fn bind_node<'b, B>(&'b mut self, binding: B) -> <B as Edge<Graph>>::Result
    where
        B: Edge<Graph>,
        B: Bind<&'b mut Graph, <B as Edge<Graph>>::Result>,
    {
        self.graph.bind_node(binding)
    }

    /// Binds a [`ComputePipeline`], [`GraphicPipeline`], or [`RayTracePipeline`] to the current
    /// pass, allowing for strongly typed access to the related functions.
    pub fn bind_pipeline<B>(self, binding: B) -> <B as Edge<Self>>::Result
    where
        B: Edge<Self>,
        B: Bind<Self, <B as Edge<Self>>::Result>,
    {
        binding.bind(self)
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

    /// Returns Info used to crate a node.
    pub fn node_info<N>(&self, node: N) -> <N as Info>::Info
    where
        N: Info,
    {
        node.info(&self.graph.bindings)
    }

    fn push_execute(
        &mut self,
        func: impl FnOnce(&Device, vk::CommandBuffer, Bindings<'_>) + Send + 'static,
    ) {
        let pass = self.as_mut();
        let exec = {
            let last_exec = pass.execs.last_mut().unwrap();
            last_exec.func = Some(ExecutionFunction(Box::new(func)));

            Execution {
                pipeline: last_exec.pipeline.clone(),
                ..Default::default()
            }
        };

        pass.execs.push(exec);
        self.exec_idx += 1;
    }

    fn push_node_access(&mut self, node: impl Node, access: AccessType, subresource: Subresource) {
        let node_idx = node.index();
        self.assert_bound_graph_node(node);

        let access = SubresourceAccess {
            access,
            subresource,
        };
        self.as_mut()
            .execs
            .last_mut()
            .unwrap()
            .accesses
            .entry(node_idx)
            .and_modify(|accesses| accesses.push(access))
            .or_insert(vec![access]);
    }

    /// Informs the pass that the next recorded command buffer will read the given `node` using
    /// [`AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer`].
    ///
    /// This function must be called for `node` before it is read within a `record` function. For
    /// more specific access, see [`PassRef::access_node`].
    pub fn read_node(mut self, node: impl Node + Info) -> Self {
        self.read_node_mut(node);

        self
    }

    /// Informs the pass that the next recorded command buffer will read the given `node` using
    /// [`AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer`].
    ///
    /// This function must be called for `node` before it is read within a `record` function. For
    /// more specific access, see [`PassRef::access_node`].
    pub fn read_node_mut(&mut self, node: impl Node + Info) {
        self.access_node_mut(
            node,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );
    }

    /// Begin recording an acceleration structure command buffer.
    ///
    /// This is the entry point for building and updating an [`AccelerationStructure`] instance.
    pub fn record_accel_struct(
        mut self,
        func: impl FnOnce(AccelerationStructureRef<'_>, Bindings<'_>) + Send + 'static,
    ) -> Self {
        self.push_execute(move |device, cmd_buf, bindings| {
            func(
                AccelerationStructureRef {
                    bindings,
                    cmd_buf,
                    device,
                },
                bindings,
            );
        });

        self
    }

    /// Begin recording a general command buffer.
    ///
    /// The provided closure allows you to run any Vulkan code, or interoperate with other Vulkan
    /// code and interfaces.
    pub fn record_cmd_buf(
        mut self,
        func: impl FnOnce(&Device, vk::CommandBuffer, Bindings<'_>) + Send + 'static,
    ) -> Self {
        self.push_execute(func);

        self
    }

    /// Sets a debugging name, but only in debug builds
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        #[cfg(debug_assertions)]
        {
            self.as_mut().name = Some(name.into());
        }

        self
    }

    /// Informs the pass that the next recorded command buffer will write the given `node` using
    /// [`AccessType::AnyShaderWrite`].
    ///
    /// This function must be called for `node` before it is written within a `record` function. For
    /// more specific access, see [`PassRef::access_node`].
    pub fn write_node(mut self, node: impl Node + Info) -> Self {
        self.write_node_mut(node);

        self
    }

    /// Informs the pass that the next recorded command buffer will write the given `node` using
    /// [`AccessType::AnyShaderWrite`].
    ///
    /// This function must be called for `node` before it is written within a `record` function. For
    /// more specific access, see [`PassRef::access_node`].
    pub fn write_node_mut(&mut self, node: impl Node + Info) {
        self.access_node_mut(node, AccessType::AnyShaderWrite);
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

/// Describes a portion of a resource which is bound.
#[derive(Clone, Copy, Debug)]
pub enum Subresource {
    /// Acceleration structures are bound whole.
    AccelerationStructure,

    /// Images may be partially bound.
    Image(vk::ImageSubresourceRange),

    /// Buffers may be partially bound.
    Buffer(BufferSubresourceRange),
}

impl Subresource {
    pub(super) fn as_image(&self) -> Option<&vk::ImageSubresourceRange> {
        if let Self::Image(subresource) = self {
            Some(subresource)
        } else {
            None
        }
    }
}

impl From<()> for Subresource {
    fn from(_: ()) -> Self {
        Self::AccelerationStructure
    }
}

impl From<vk::ImageSubresourceRange> for Subresource {
    fn from(subresource: vk::ImageSubresourceRange) -> Self {
        Self::Image(subresource)
    }
}

impl From<BufferSubresourceRange> for Subresource {
    fn from(subresource: BufferSubresourceRange) -> Self {
        Self::Buffer(subresource)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SubresourceAccess {
    pub access: AccessType,
    pub subresource: Subresource,
}
