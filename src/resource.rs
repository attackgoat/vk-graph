use {
    super::{
        AccelerationStructureLeaseNode, AccelerationStructureNode, BufferLeaseNode, BufferNode,
        Graph, ImageLeaseNode, ImageNode, SwapchainImageNode,
    },
    crate::{
        driver::{
            accel_struct::AccelerationStructure, buffer::Buffer, image::Image,
            swapchain::SwapchainImage,
        },
        pool::Lease,
    },
    std::{fmt::Debug, sync::Arc},
};

/// A trait for resources which may be bound to a `Graph`.
///
/// See [`Graph::bind_resource`] and
/// [`CommandRef::bind_resource`](super::cmd::CommandRef::bind_resource) for details.
pub trait GraphResource {
    /// The resource handle type.
    type Node;

    #[doc(hidden)]
    fn bind_graph(self, _: &mut Graph) -> Self::Node;

    #[deprecated = "use bind_graph function"]
    #[doc(hidden)]
    fn bind(self, graph: &mut Graph) -> Self::Node
    where
        Self: Sized,
    {
        self.bind_graph(graph)
    }
}

impl GraphResource for SwapchainImage {
    type Node = SwapchainImageNode;

    fn bind_graph(self, graph: &mut Graph) -> Self::Node {
        // We will return a new node
        let res = Self::Node::new(graph.resources.len());

        //trace!("Node {}: {:?}", res.idx, &self);

        graph.resources.push(Resource {
            inner: ResourceInner::SwapchainImage(Box::new(self)),
        });

        res
    }
}

macro_rules! graph_resource {
    ($name:ident) => {
        paste::paste! {
            impl GraphResource for $name {
                type Node = [<$name Node>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource a new item (Image or Buffer or etc)

                    // We will return a new node
                    let res = Self::Node::new(graph.resources.len());
                    let resource = Resource {
                        inner: ResourceInner::$name(Arc::new(self)),
                    };
                    graph.resources.push(resource);

                    res
                }
            }

            impl GraphResource for Arc<$name> {
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
                    let resource = Resource {
                        inner: ResourceInner::$name(self),
                    };
                    graph.resources.push(resource);

                    res
                }
            }

            impl<'a> GraphResource for &'a Arc<$name> {
                type Node = [<$name Node>];

                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource a borrowed resource (&Arc<Image> or
                    // &Arc<Buffer> or etc)

                    Arc::clone(self).bind_graph(graph)
                }
            }

            impl GraphResource for Lease<$name> {
                type Node = [<$name LeaseNode>];

                #[profiling::function]
                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource a new lease (Lease<Image> or Lease<Buffer> or
                    // etc)

                    // We will return a new node
                    let res = Self::Node::new(graph.resources.len());
                    let resource = Resource {
                        inner: ResourceInner::[<$name Lease>](Arc::new(self)),
                    };
                    graph.resources.push(resource);

                    res
                }
            }

            impl GraphResource  for Arc<Lease<$name>> {
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
                    let resource = Resource {
                        inner: ResourceInner::[<$name Lease>](self),
                    };
                    graph.resources.push(resource);

                    res
                }
            }

            impl<'a> GraphResource for &'a Arc<Lease<$name>> {
                type Node = [<$name LeaseNode>];

                fn bind_graph(self, graph: &mut Graph) -> Self::Node {
                    // In this function we are resource a borrowed resource (&Arc<Lease<Image>> or
                    // &Arc<Lease<Buffer>> or etc)

                    Arc::clone(self).bind_graph(graph)
                }
            }

            impl Resource {
                pub(super) fn [<as_ $name:snake>](&self) -> Option<&Arc<$name>> {
                    let ResourceInner::$name(resource) = &self.inner else {
                        return None;
                    };

                    Some(resource)
                }

                pub(super) fn [<as_ $name:snake _mut>](&mut self) -> Option<&mut Arc<$name>> {
                    let ResourceInner::$name(resource) = &mut self.inner else {
                        return None;
                    };

                    Some(resource)
                }

                pub(super) fn [<as_ $name:snake _lease>](&self) -> Option<&Arc<Lease<$name>>> {
                    let ResourceInner::[<$name Lease>](resource) = &self.inner else {
                        return None
                    };

                    Some(resource)
                }

                pub(super) fn [<as_ $name:snake _lease_mut>](&mut self) -> Option<&mut Arc<Lease<$name>>> {
                    let ResourceInner::[<$name Lease>](resource) = &mut self.inner else {
                        return None;
                    };

                    Some(resource)
                }
            }
        }
    };
}

graph_resource!(AccelerationStructure);
graph_resource!(Image);
graph_resource!(Buffer);

#[derive(Debug)]
#[doc(hidden)]
pub struct Resource {
    pub(crate) inner: ResourceInner,
}

impl Resource {
    pub(super) fn as_driver_accel_struct(&self) -> Option<&AccelerationStructure> {
        Some(match &self.inner {
            ResourceInner::AccelerationStructure(resource) => resource,
            ResourceInner::AccelerationStructureLease(resource) => resource,
            _ => return None,
        })
    }

    pub(super) fn as_driver_buffer(&self) -> Option<&Buffer> {
        Some(match &self.inner {
            ResourceInner::Buffer(resource) => resource,
            ResourceInner::BufferLease(resource) => resource,
            _ => return None,
        })
    }

    pub(super) fn as_driver_image(&self) -> Option<&Image> {
        Some(match &self.inner {
            ResourceInner::Image(resource) => resource,
            ResourceInner::ImageLease(resource) => resource,
            ResourceInner::SwapchainImage(resource) => resource,
            _ => return None,
        })
    }

    pub(super) fn as_swapchain_image(&self) -> Option<&SwapchainImage> {
        Some(match &self.inner {
            ResourceInner::SwapchainImage(resource) => resource,
            _ => return None,
        })
    }
}

#[derive(Debug)]
pub(crate) enum ResourceInner {
    AccelerationStructure(Arc<AccelerationStructure>),
    AccelerationStructureLease(Arc<Lease<AccelerationStructure>>),
    Buffer(Arc<Buffer>),
    BufferLease(Arc<Lease<Buffer>>),
    Image(Arc<Image>),
    ImageLease(Arc<Lease<Image>>),
    SwapchainImage(Box<SwapchainImage>),
}
