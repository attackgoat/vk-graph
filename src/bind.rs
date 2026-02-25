use {
    super::{
        AccelerationStructureLeaseNode, AccelerationStructureNode, AnyAccelerationStructureNode,
        AnyBufferNode, AnyImageNode, BufferLeaseNode, BufferNode, Graph, ImageLeaseNode, ImageNode,
        SwapchainImageNode,
    },
    crate::{
        driver::{
            accel_struct::AccelerationStructure, buffer::Buffer, image::Image,
            swapchain::SwapchainImage,
        },
        node::Node,
        pool::Lease,
    },
    std::{fmt::Debug, sync::Arc},
};

/// A trait for resources which may be bound to a `Graph`.
///
/// See [`Graph::bind_resource`] and
/// [`CommandRef::bind_resource`](super::cmd_ref::CommandRef::bind_resource) for details.
pub trait BindGraph {
    /// The resource handle type.
    type Node;

    /// Binds the resource to a graph.
    ///
    /// Returns a resource node handle.
    fn bind_graph(self, graph: &mut Graph) -> Self::Node;

    #[deprecated = "use bind_graph function"]
    #[doc(hidden)]
    fn bind(self, graph: &mut Graph) -> Self::Node;
}

/// TODO
#[allow(missing_docs)]
#[derive(Debug)]
pub enum Resource {
    AccelerationStructure(Arc<AccelerationStructure>),
    AccelerationStructureLease(Arc<Lease<AccelerationStructure>>),
    Buffer(Arc<Buffer>),
    BufferLease(Arc<Lease<Buffer>>),
    Image(Arc<Image>),
    ImageLease(Arc<Lease<Image>>),
    SwapchainImage(Box<SwapchainImage>),
}

impl Resource {
    pub(super) fn as_driver_accel_struct(&self) -> Option<&AccelerationStructure> {
        Some(match self {
            Self::AccelerationStructure(resource) => resource,
            Self::AccelerationStructureLease(resource) => resource,
            _ => return None,
        })
    }

    pub(super) fn as_driver_buffer(&self) -> Option<&Buffer> {
        Some(match self {
            Self::Buffer(resource) => resource,
            Self::BufferLease(resource) => resource,
            _ => return None,
        })
    }

    pub(super) fn as_driver_image(&self) -> Option<&Image> {
        Some(match self {
            Self::Image(resource) => resource,
            Self::ImageLease(resource) => resource,
            Self::SwapchainImage(resource) => resource,
            _ => return None,
        })
    }

    pub(super) fn as_swapchain_image(&self) -> Option<&SwapchainImage> {
        Some(match self {
            Self::SwapchainImage(resource) => resource,
            _ => return None,
        })
    }
}

impl BindGraph for SwapchainImage {
    type Node = SwapchainImageNode;

    fn bind_graph(self, graph: &mut Graph) -> Self::Node {
        // We will return a new node
        let res = Self::Node::new(graph.resources.len());

        //trace!("Node {}: {:?}", res.idx, &self);

        graph
            .resources
            .push(Resource::SwapchainImage(Box::new(self)));

        res
    }

    fn bind(self, graph: &mut Graph) -> Self::Node {
        self.bind_graph(graph)
    }
}

macro_rules! bind_graph_resource {
    ($name:ident) => {
        paste::paste! {
            impl BindGraph for $name {
                type Node = [<$name Node>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource a new item (Image or Buffer or etc)

                    // We will return a new node
                    let res = Self::Node::new(graph.resources.len());
                    let resource = Resource::$name(Arc::new(self));
                    graph.resources.push(resource);

                    res
                }

                fn bind(self, graph: &mut Graph) -> Self::Node {
                    self.bind_graph(graph)
                }
            }

            impl BindGraph for Arc<$name> {
                type Node = [<$name Node>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource an existing resource (Arc<Image> or
                    // Arc<Buffer> or etc)

                    // We will return an existing node, if possible
                    // TODO: Could store a sorted list of these shared pointers to avoid the O(N)
                    for (idx, existing_resource) in graph.resources.iter_mut().enumerate() {
                        if let Some(existing_resource) = existing_resource.[<as_ $name:snake _mut>]() {
                            if Arc::ptr_eq(existing_resource, &self) {
                                return Self::Node::new(idx);
                            }
                        }
                    }

                    // Return a new node
                    let res = Self::Node::new(graph.resources.len());
                    let resource = Resource::$name(self);
                    graph.resources.push(resource);

                    res
                }

                fn bind(self, graph: &mut Graph) -> Self::Node {
                    self.bind_graph(graph)
                }
            }

            impl<'a> BindGraph for &'a Arc<$name> {
                type Node = [<$name Node>];

                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource a borrowed resource (&Arc<Image> or
                    // &Arc<Buffer> or etc)

                    Arc::clone(self).bind_graph(graph)
                }

                fn bind(self, graph: &mut Graph) -> Self::Node {
                    self.bind_graph(graph)
                }
            }

            impl BindGraph for Lease<$name> {
                type Node = [<$name LeaseNode>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource a new lease (Lease<Image> or Lease<Buffer> or
                    // etc)

                    // We will return a new node
                    let res = Self::Node::new(graph.resources.len());
                    let resource = Resource::[<$name Lease>](Arc::new(self));
                    graph.resources.push(resource);

                    res
                }

                fn bind(self, graph: &mut Graph) -> Self::Node {
                    self.bind_graph(graph)
                }
            }

            impl BindGraph for Arc<Lease<$name>> {
                type Node = [<$name LeaseNode>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource an existing lease resource
                    // (Arc<Lease<Image>> or Arc<Lease<Buffer>> or etc)

                    // We will return an existing node, if possible
                    // TODO: Could store a sorted list of these shared pointers to avoid the O(N)
                    for (idx, existing_resource) in graph.resources.iter_mut().enumerate() {
                        if let Some(existing_resource) = existing_resource.[<as_ $name:snake _lease_mut>]() {
                            if Arc::ptr_eq(existing_resource, &self) {
                                return Self::Node::new(idx);
                            }
                        }
                    }

                    // We will return a new node
                    let res = Self::Node::new(graph.resources.len());
                    let resource = Resource::[<$name Lease>](self);
                    graph.resources.push(resource);

                    res
                }

                fn bind(self, graph: &mut Graph) -> Self::Node {
                    self.bind_graph(graph)
                }
            }

            impl<'a> BindGraph for &'a Arc<Lease<$name>> {
                type Node = [<$name LeaseNode>];

                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource a borrowed resource (&Arc<Lease<Image>> or
                    // &Arc<Lease<Buffer>> or etc)

                    Arc::clone(self).bind_graph(graph)
                }

                fn bind(self, graph: &mut Graph) -> Self::Node {
                    self.bind_graph(graph)
                }
            }

            impl Resource {
                pub(super) fn [<as_ $name:snake>](&self) -> Option<&Arc<$name>> {
                    let Self::$name(resource) = self else {
                        return None;
                    };

                    Some(resource)
                }

                pub(super) fn [<as_ $name:snake _mut>](&mut self) -> Option<&mut Arc<$name>> {
                    let Self::$name(resource) = self else {
                        return None;
                    };

                    Some(resource)
                }

                pub(super) fn [<as_ $name:snake _lease>](&self) -> Option<&Arc<Lease<$name>>> {
                    let Self::[<$name Lease>](resource) = self else {
                        return None
                    };

                    Some(resource)
                }

                pub(super) fn [<as_ $name:snake _lease_mut>](&mut self) -> Option<&Arc<Lease<$name>>> {
                    let Self::[<$name Lease>](resource) = self else {
                        return None;
                    };

                    Some(resource)
                }
            }
        }
    };
}

bind_graph_resource!(AccelerationStructure);
bind_graph_resource!(Image);
bind_graph_resource!(Buffer);

/// A trait for resources which may be borrowed from a `Graph`.
///
/// See [`Graph::node`] for details.
pub trait Bound<G = Graph> {
    /// The Vulkan buffer, image, or acceleration struction type.
    type Resource;

    /// Borrows the resource from a graph.
    fn borrow(self, graph: &Graph) -> &Self::Resource;
}

impl Bound for AnyAccelerationStructureNode {
    type Resource = AccelerationStructure;

    fn borrow(self, graph: &Graph) -> &Self::Resource {
        graph.resources[self.index()]
            .as_driver_accel_struct()
            .unwrap()
    }
}

impl Bound for AnyBufferNode {
    type Resource = Buffer;

    fn borrow(self, graph: &Graph) -> &Self::Resource {
        graph.resources[self.index()].as_driver_buffer().unwrap()
    }
}

impl Bound for AnyImageNode {
    type Resource = Image;

    fn borrow(self, graph: &Graph) -> &Self::Resource {
        graph.resources[self.index()].as_driver_image().unwrap()
    }
}

impl Bound for SwapchainImageNode {
    type Resource = SwapchainImage;

    fn borrow(self, graph: &Graph) -> &Self::Resource {
        graph.resources[self.idx].as_swapchain_image().unwrap()
    }
}

macro_rules! bound {
    ($name:ident) => {
        paste::paste! {
            impl Bound for [<$name Node>] {
                type Resource = Arc<$name>;

                fn borrow(self, graph: &Graph) -> &Self::Resource {
                    graph
                        .resources[self.idx]
                        .[<as_ $name:snake>]()
                        .unwrap()
                }
            }

            impl Bound for [<$name LeaseNode>] {
                type Resource = Arc<Lease<$name>>;

                fn borrow(self, graph: &Graph) -> &Self::Resource {
                    graph
                        .resources[self.idx]
                        .[<as_ $name:snake _lease>]()
                        .unwrap()
                }
            }
        }
    };
}

bound!(AccelerationStructure);
bound!(Buffer);
bound!(Image);
