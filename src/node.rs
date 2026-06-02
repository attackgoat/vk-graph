//! Handles for Vulkan smart-pointer resources.
//!
//! When you bind a resource to a [`Graph`](crate::Graph), you get back a node handle:
//!
//! ```ignore
//! let buf_node: BufferNode = graph.bind_resource(my_buffer);
//! let img_node: ImageNode   = graph.bind_resource(my_image);
//! ```
//!
//! These handles are then passed to command-building methods like
//! [`resource_access`](crate::cmd::PipelineCommand::resource_access) or
//! [`shader_resource_access`](crate::cmd::PipelineCommand::shader_resource_access).
//!
//! ## Node kinds
//!
//! | Handle | Resource type | Use case |
//! |---|---|:--|
//! | [`BufferNode`] | Owned [`Buffer`] | Most common |
//! | [`ImageNode`] | Owned [`Image`] | Most common |
//! | [`AccelerationStructureNode`] | Owned [`AccelerationStructure`] | Ray tracing |
//! | [`SwapchainImageNode`] | [`SwapchainImage`] | Swapchain presentation |
//! | [`BufferLeaseNode`], [`ImageLeaseNode`], [`AccelerationStructureLeaseNode`] | Pool-leased resource | Pool-based allocation |
//! | [`AnyBufferNode`], [`AnyImageNode`], [`AnyAccelerationStructureNode`] | Any of the above | Heterogeneous collections |
//!
//! For most users, [`BufferNode`] and [`ImageNode`] are all you need. The `Lease` and
//! `Any*` variants exist for advanced pooling and dynamic dispatch scenarios.
//!
//! When borrowing resources back out of a graph with [`Graph::resource`](crate::Graph::resource),
//! concrete node types return the exact stored handle type, while `Any*` node types return a
//! borrow of the underlying resource. For example, `BufferNode` yields `&Arc<Buffer>`, but
//! `AnyBufferNode` yields `&Buffer`.

use std::sync::Arc;

use crate::{
    GraphId, Node,
    driver::{
        accel_struct::AccelerationStructure, buffer::Buffer, image::Image,
        swapchain::SwapchainImage,
    },
    pool::Lease,
    private,
};

use super::{AnyResource, NodeIndex};

/// Specifies either an owned acceleration structure or one obtained from a pool.
#[derive(Clone, Copy, Debug)]
pub enum AnyAccelerationStructureNode {
    /// An owned acceleration structure.
    AccelerationStructure(AccelerationStructureNode),

    /// An acceleration structure obtained from a pool.
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

impl private::Sealed for AnyAccelerationStructureNode {}

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

    fn assert_owner(&self, graph_id: GraphId) {
        match self {
            Self::AccelerationStructure(node) => node.assert_owner(graph_id),
            Self::AccelerationStructureLease(node) => node.assert_owner(graph_id),
        }
    }
}

/// Specifies either an owned buffer or one obtained from a pool.
#[derive(Clone, Copy, Debug)]
pub enum AnyBufferNode {
    /// An owned buffer.
    Buffer(BufferNode),

    /// A buffer obtained from a pool.
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

impl private::Sealed for AnyBufferNode {}

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

    fn assert_owner(&self, graph_id: GraphId) {
        match self {
            Self::Buffer(node) => node.assert_owner(graph_id),
            Self::BufferLease(node) => node.assert_owner(graph_id),
        }
    }
}

/// Specifies either an owned image or one obtained from a pool.
///
/// The image may also be a special swapchain type of image.
#[derive(Clone, Copy, Debug)]
pub enum AnyImageNode {
    /// An owned image.
    Image(ImageNode),

    /// An image obtained from a pool.
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

impl private::Sealed for AnyImageNode {}

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

    fn assert_owner(&self, graph_id: GraphId) {
        match self {
            Self::Image(node) => node.assert_owner(graph_id),
            Self::ImageLease(node) => node.assert_owner(graph_id),
            Self::SwapchainImage(node) => node.assert_owner(graph_id),
        }
    }
}

macro_rules! node {
    ($name:ident, $resource:ty, $fn_name:ident) => {
        paste::paste! {
            /// A graph-local handle for a bound resource.
            ///
            /// Node handles are only valid with the graph that produced them.
            ///
            /// When the `checked` feature is enabled, using a node with a different graph will
            /// panic immediately.
            ///
            /// When `checked` is disabled, this ownership check is skipped for zero-overhead
            /// builds, so cross-graph node misuse is invalid and may resolve to the wrong resource.
            #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
            pub struct [<$name Node>] {
                index: NodeIndex,

                #[cfg(feature = "checked")]
                graph_id: GraphId,
            }

            impl [<$name Node>] {
                pub(crate) fn new(
                    index: usize,
                    #[cfg(feature = "checked")] graph_id: GraphId,
                ) -> Self {
                    Self {
                        index,

                        #[cfg(feature = "checked")]
                        graph_id,
                    }
                }
            }

            impl private::Sealed for [<$name Node>] {}

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

                fn assert_owner(&self, _graph_id: GraphId) {
                    #[cfg(feature = "checked")]
                    assert!(self.graph_id == _graph_id, "node belongs to a different graph");
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
