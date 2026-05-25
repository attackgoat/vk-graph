//! Handles for Vulkan smart-pointer resources.

use std::sync::Arc;

use crate::{
    Node,
    driver::{
        accel_struct::AccelerationStructure, buffer::Buffer, image::Image,
        swapchain::SwapchainImage,
    },
    pool::Lease,
};

use super::{AnyResource, NodeIndex};

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
    type Resource = AccelerationStructure;

    fn borrow(self, resources: &[AnyResource]) -> &Self::Resource {
        resources[self.index()].expect_accel_struct()
    }

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
    type Resource = Buffer;

    fn borrow(self, resources: &[AnyResource]) -> &Self::Resource {
        resources[self.index()].expect_buffer()
    }

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
    type Resource = Image;

    fn borrow(self, resources: &[AnyResource]) -> &Self::Resource {
        resources[self.index()].expect_image()
    }

    fn index(&self) -> NodeIndex {
        match self {
            Self::Image(node) => node.index(),
            Self::ImageLease(node) => node.index(),
            Self::SwapchainImage(node) => node.index(),
        }
    }
}

macro_rules! node {
    ($name:ident, $resource:ty, $fn_name:ident) => {
        paste::paste! {
            /// Resource node.
            #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
            pub struct [<$name Node>] {
                index: NodeIndex,
            }

            impl [<$name Node>] {
                pub(crate) fn new(index: usize) -> Self {
                    Self {
                        index,
                    }
                }
            }

            impl Node for [<$name Node>] {
                type Resource = $resource;

                fn borrow(self, resources: &[AnyResource]) -> &Self::Resource {
                    let AnyResource::$name(res) = &resources[self.index] else {
                        panic!("invalid resource node handle");
                    };

                    res
                }

                fn index(&self) -> NodeIndex {
                    self.index
                }
            }
        }
    };
}

node!(
    AccelerationStructure,
    Arc<AccelerationStructure>,
    as_accel_struct
);
node!(
    AccelerationStructureLease,
    Arc<Lease<AccelerationStructure>>,
    as_accel_struct
);
node!(Buffer, Arc<Buffer>, as_buffer);
node!(BufferLease, Arc<Lease<Buffer>>, as_buffer);
node!(Image, Arc<Image>, as_image);
node!(ImageLease, Arc<Lease<Image>>, as_image);
node!(SwapchainImage, SwapchainImage, as_swapchain_image);
