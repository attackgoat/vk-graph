//! Bindings for Vulkan smart-pointer resources.

use crate::NodeIndex;

/// Specifies either an owned acceleration structure or an acceleration structure leased from a
/// pool.
#[derive(Clone, Copy, Debug)]
pub enum AnyAccelerationStructureNode {
    /// An owned acceleration structure.
    AccelerationStructure(AccelerationStructureNode),

    /// An acceleration structure leased from a pool.
    AccelerationStructureLease(AccelerationStructureLeaseNode),
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
    fn index(&self) -> NodeIndex {
        match self {
            Self::AccelerationStructure(node) => node.index(),
            Self::AccelerationStructureLease(node) => node.index(),
        }
    }
}

/// Specifies either an owned buffer or a buffer leased from a pool.
#[derive(Clone, Copy, Debug)]
pub enum AnyBufferNode {
    /// An owned buffer.
    Buffer(BufferNode),

    /// A buffer leased from a pool.
    BufferLease(BufferLeaseNode),
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
    fn index(&self) -> NodeIndex {
        match self {
            Self::Buffer(node) => node.index(),
            Self::BufferLease(node) => node.index(),
        }
    }
}

/// Specifies either an owned image or an image leased from a pool.
///
/// The image may also be a special swapchain type of image.
#[derive(Clone, Copy, Debug)]
pub enum AnyImageNode {
    /// An owned image.
    Image(ImageNode),

    /// An image leased from a pool.
    ImageLease(ImageLeaseNode),

    /// A special swapchain image.
    SwapchainImage(SwapchainImageNode),
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
    fn index(&self) -> NodeIndex {
        match self {
            Self::Image(node) => node.index(),
            Self::ImageLease(node) => node.index(),
            Self::SwapchainImage(node) => node.index(),
        }
    }
}

/// A Vulkan resource which has been bound to a [`Graph`] using [`Graph::bind_node`].
pub trait Node {
    #[doc(hidden)]
    fn index(&self) -> NodeIndex;
}

macro_rules! node {
    ($name:ident) => {
        paste::paste! {
            /// Resource node.
            #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
            pub struct [<$name Node>] {
                pub(super) idx: NodeIndex,
            }

            impl [<$name Node>] {
                pub(super) fn new(idx: usize) -> Self {
                    Self {
                        idx,
                    }
                }
            }

            impl Node for [<$name Node>] {
                fn index(&self) -> NodeIndex {
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
