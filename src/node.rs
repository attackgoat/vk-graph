//! Bindings for Vulkan smart-pointer resources.

use {
    super::{Information, NodeIndex, RenderGraph, Unbind},
    crate::{
        driver::{
            accel_struct::{AccelerationStructure, AccelerationStructureInfo},
            buffer::{Buffer, BufferInfo},
            image::{Image, ImageInfo},
        },
        pool::Lease,
    },
    std::sync::Arc,
};

/// Specifies either an owned acceleration structure or an acceleration structure leased from a
/// pool.
#[derive(Debug)]
pub enum AnyAccelerationStructureNode {
    /// An owned acceleration structure.
    AccelerationStructure(AccelerationStructureNode),

    /// An acceleration structure leased from a pool.
    AccelerationStructureLease(AccelerationStructureLeaseNode),
}

impl Clone for AnyAccelerationStructureNode {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for AnyAccelerationStructureNode {}

impl Information for AnyAccelerationStructureNode {
    type Info = AccelerationStructureInfo;

    fn get(self, graph: &RenderGraph) -> Self::Info {
        match self {
            Self::AccelerationStructure(node) => node.get(graph),
            Self::AccelerationStructureLease(node) => node.get(graph),
        }
    }
}

impl From<AccelerationStructureNode> for AnyAccelerationStructureNode {
    fn from(node: AccelerationStructureNode) -> Self {
        Self::AccelerationStructure(node)
    }
}

impl From<AccelerationStructureLeaseNode> for AnyAccelerationStructureNode {
    fn from(node: AccelerationStructureLeaseNode) -> Self {
        Self::AccelerationStructureLease(node)
    }
}

impl Node for AnyAccelerationStructureNode {
    fn index(self) -> NodeIndex {
        match self {
            Self::AccelerationStructure(node) => node.index(),
            Self::AccelerationStructureLease(node) => node.index(),
        }
    }
}

/// Specifies either an owned buffer or a buffer leased from a pool.
#[derive(Debug)]
pub enum AnyBufferNode {
    /// An owned buffer.
    Buffer(BufferNode),

    /// A buffer leased from a pool.
    BufferLease(BufferLeaseNode),
}

impl Clone for AnyBufferNode {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for AnyBufferNode {}

impl Information for AnyBufferNode {
    type Info = BufferInfo;

    fn get(self, graph: &RenderGraph) -> Self::Info {
        match self {
            Self::Buffer(node) => node.get(graph),
            Self::BufferLease(node) => node.get(graph),
        }
    }
}

impl From<BufferNode> for AnyBufferNode {
    fn from(node: BufferNode) -> Self {
        Self::Buffer(node)
    }
}

impl From<BufferLeaseNode> for AnyBufferNode {
    fn from(node: BufferLeaseNode) -> Self {
        Self::BufferLease(node)
    }
}

impl Node for AnyBufferNode {
    fn index(self) -> NodeIndex {
        match self {
            Self::Buffer(node) => node.index(),
            Self::BufferLease(node) => node.index(),
        }
    }
}

/// Specifies either an owned image or an image leased from a pool.
///
/// The image may also be a special swapchain type of image.
#[derive(Debug)]
pub enum AnyImageNode {
    /// An owned image.
    Image(ImageNode),

    /// An image leased from a pool.
    ImageLease(ImageLeaseNode),

    /// A special swapchain image.
    SwapchainImage(SwapchainImageNode),
}

impl Clone for AnyImageNode {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for AnyImageNode {}

impl Information for AnyImageNode {
    type Info = ImageInfo;

    fn get(self, graph: &RenderGraph) -> Self::Info {
        match self {
            Self::Image(node) => node.get(graph),
            Self::ImageLease(node) => node.get(graph),
            Self::SwapchainImage(node) => node.get(graph),
        }
    }
}

impl From<ImageNode> for AnyImageNode {
    fn from(node: ImageNode) -> Self {
        Self::Image(node)
    }
}

impl From<ImageLeaseNode> for AnyImageNode {
    fn from(node: ImageLeaseNode) -> Self {
        Self::ImageLease(node)
    }
}

impl From<SwapchainImageNode> for AnyImageNode {
    fn from(node: SwapchainImageNode) -> Self {
        Self::SwapchainImage(node)
    }
}

impl Node for AnyImageNode {
    fn index(self) -> NodeIndex {
        match self {
            Self::Image(node) => node.index(),
            Self::ImageLease(node) => node.index(),
            Self::SwapchainImage(node) => node.index(),
        }
    }
}

/// A Vulkan resource which has been bound to a [`RenderGraph`] using [`RenderGraph::bind_node`].
pub trait Node: Copy {
    /// The internal node index of this bound resource.
    fn index(self) -> NodeIndex;
}

macro_rules! node {
    ($name:ident) => {
        paste::paste! {
            /// Resource node.
            #[derive(Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
            pub struct [<$name Node>] {
                pub(super) idx: NodeIndex,
            }

            impl [<$name Node>] {
                pub(super) fn new(idx: NodeIndex) -> Self {
                    Self {
                        idx,
                    }
                }
            }

            impl Clone for [<$name Node>] {
                fn clone(&self) -> Self {
                    *self
                }
            }

            impl Copy for [<$name Node>] {}

            impl Node for [<$name Node>] {
                fn index(self) -> NodeIndex {
                    self.idx
                }
            }
        }
    };
}

node!(AccelerationStructure);
node!(AccelerationStructureLease);
node!(Buffer);
node!(BufferLease);
node!(Image);
node!(ImageLease);
node!(SwapchainImage);

macro_rules! node_unbind {
    ($name:ident) => {
        paste::paste! {
            impl Unbind<RenderGraph, Arc<$name>> for [<$name Node>] {
                fn unbind(self, graph: &mut RenderGraph) -> Arc<$name> {
                    let binding = &mut graph.bindings[self.idx];
                    let res = Arc::clone(
                        binding
                            .[<as_ $name:snake>]()
                            .unwrap()
                    );
                    binding.unbind();

                    res
                }
            }
        }
    };
}

node_unbind!(AccelerationStructure);
node_unbind!(Buffer);
node_unbind!(Image);

macro_rules! node_unbind_lease {
    ($name:ident) => {
        paste::paste! {
            impl Unbind<RenderGraph, Arc<Lease<$name>>> for [<$name LeaseNode>] {
                fn unbind(self, graph: &mut RenderGraph) -> Arc<Lease<$name>> {
                    let binding = &mut graph.bindings[self.idx];
                    let res = Arc::clone(
                        binding
                            .[<as_ $name:snake _lease>]()
                            .unwrap()
                    );
                    binding.unbind();

                    res
                }
            }
        }
    };
}

node_unbind_lease!(AccelerationStructure);
node_unbind_lease!(Buffer);
node_unbind_lease!(Image);
