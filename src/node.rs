//! Bindings for Vulkan smart-pointer resources.

use {
    super::{Binding, Bound, Graph, Info, NodeIndex},
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

impl Info for AnyAccelerationStructureNode {
    type Info = AccelerationStructureInfo;

    fn info(self, bindings: &[Binding]) -> Self::Info {
        match self {
            Self::AccelerationStructure(node) => node.info(bindings),
            Self::AccelerationStructureLease(node) => node.info(bindings),
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

impl Info for AnyBufferNode {
    type Info = BufferInfo;

    fn info(self, bindings: &[Binding]) -> Self::Info {
        match self {
            Self::Buffer(node) => node.info(bindings),
            Self::BufferLease(node) => node.info(bindings),
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

impl Info for AnyImageNode {
    type Info = ImageInfo;

    fn info(self, bindings: &[Binding]) -> Self::Info {
        match self {
            Self::Image(node) => node.info(bindings),
            Self::ImageLease(node) => node.info(bindings),
            Self::SwapchainImage(node) => node.info(bindings),
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

/// A Vulkan resource which has been bound to a [`Graph`] using [`Graph::bind_node`].
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

macro_rules! node_bound {
    ($name:ident) => {
        paste::paste! {
            impl Bound<Graph, Arc<$name>> for [<$name Node>] {
                fn borrow(self, graph: &Graph) -> &Arc<$name> {
                    graph
                        .bindings[self.idx]
                        .[<as_ $name:snake>]()
                        .unwrap()
                }
            }
        }
    };
}

node_bound!(AccelerationStructure);
node_bound!(Buffer);
node_bound!(Image);

macro_rules! node_lease_bound {
    ($name:ident) => {
        paste::paste! {
            impl Bound<Graph, Arc<Lease<$name>>> for [<$name LeaseNode>] {
                fn borrow(self, graph: &Graph) -> &Arc<Lease<$name>> {
                    graph
                        .bindings[self.idx]
                        .[<as_ $name:snake _lease>]()
                        .unwrap()
                }
            }
        }
    };
}

node_lease_bound!(AccelerationStructure);
node_lease_bound!(Buffer);
node_lease_bound!(Image);
