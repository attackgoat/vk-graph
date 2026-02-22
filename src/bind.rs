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
/// See [`Graph::bind_node`] and
/// [`CommandRef::bind_pipeline`](super::cmd_ref::CommandRef::bind_pipeline) for details.
pub trait Bind<Graph, Node> {
    /// Binds the resource to a graph.
    ///
    /// Returns a node handle.
    fn bind(self, graph: Graph) -> Node;
}

#[derive(Debug)]
pub enum Binding {
    AccelerationStructure(Arc<AccelerationStructure>),
    AccelerationStructureLease(Arc<Lease<AccelerationStructure>>),
    Buffer(Arc<Buffer>),
    BufferLease(Arc<Lease<Buffer>>),
    Image(Arc<Image>),
    ImageLease(Arc<Lease<Image>>),
    SwapchainImage(Box<SwapchainImage>),
}

impl Binding {
    pub(super) fn as_driver_acceleration_structure(&self) -> Option<&AccelerationStructure> {
        Some(match self {
            Self::AccelerationStructure(binding) => binding,
            Self::AccelerationStructureLease(binding) => binding,
            _ => return None,
        })
    }

    pub(super) fn as_driver_buffer(&self) -> Option<&Buffer> {
        Some(match self {
            Self::Buffer(binding) => binding,
            Self::BufferLease(binding) => binding,
            _ => return None,
        })
    }

    pub(super) fn as_driver_image(&self) -> Option<&Image> {
        Some(match self {
            Self::Image(binding) => binding,
            Self::ImageLease(binding) => binding,
            Self::SwapchainImage(binding) => binding,
            _ => return None,
        })
    }

    pub(super) fn as_swapchain_image(&self) -> Option<&SwapchainImage> {
        let Self::SwapchainImage(binding) = self else {
            // The private code in this module should prevent this branch
            unreachable!();
        };

        Some(binding)
    }
}

impl Bind<&mut Graph, SwapchainImageNode> for SwapchainImage {
    fn bind(self, graph: &mut Graph) -> SwapchainImageNode {
        // We will return a new node
        let res = SwapchainImageNode::new(graph.bindings.len());

        //trace!("Node {}: {:?}", res.idx, &self);

        graph.bindings.push(Binding::SwapchainImage(Box::new(self)));

        res
    }
}

macro_rules! bind {
    ($name:ident) => {
        paste::paste! {
            impl Bind<&mut Graph, [<$name Node>]> for $name {
                #[profiling::function]
                fn bind(self, graph: &mut Graph) -> [<$name Node>] {
                    // In this function we are binding a new item (Image or Buffer or etc)

                    // We will return a new node
                    let res = [<$name Node>]::new(graph.bindings.len());
                    let binding = Binding::$name(Arc::new(self));
                    graph.bindings.push(binding);

                    res
                }
            }

            impl<'a> Bind<&mut Graph, [<$name Node>]> for &'a Arc<$name> {
                fn bind(self, graph: &mut Graph) -> [<$name Node>] {
                    // In this function we are binding a borrowed binding (&Arc<Image> or
                    // &Arc<Buffer> or etc)

                    Arc::clone(self).bind(graph)
                }
            }

            impl Bind<&mut Graph, [<$name Node>]> for Arc<$name> {
                #[profiling::function]
                fn bind(self, graph: &mut Graph) -> [<$name Node>] {
                    // In this function we are binding an existing binding (Arc<Image> or
                    // Arc<Buffer> or etc)

                    // We will return an existing node, if possible
                    // TODO: Could store a sorted list of these shared pointers to avoid the O(N)
                    for (idx, existing_binding) in graph.bindings.iter_mut().enumerate() {
                        if let Some(existing_binding) = existing_binding.[<as_ $name:snake _mut>]() {
                            if Arc::ptr_eq(existing_binding, &self) {
                                return [<$name Node>]::new(idx);
                            }
                        }
                    }

                    // Return a new node
                    let res = [<$name Node>]::new(graph.bindings.len());
                    let binding = Binding::$name(self);
                    graph.bindings.push(binding);

                    res
                }
            }

            impl Binding {
                pub(super) fn [<as_ $name:snake>](&self) -> Option<&Arc<$name>> {
                    if let Self::$name(binding) = self {
                        Some(&binding)
                    } else {
                        None
                    }
                }

                pub(super) fn [<as_ $name:snake _mut>](&mut self) -> Option<&mut Arc<$name>> {
                    if let Self::$name(binding) = self {
                        Some(binding)
                    } else {
                        None
                    }
                }
            }
        }
    };
}

bind!(AccelerationStructure);
bind!(Image);
bind!(Buffer);

macro_rules! bind_lease {
    ($name:ident) => {
        paste::paste! {
            impl Bind<&mut Graph, [<$name LeaseNode>]> for Lease<$name> {
                #[profiling::function]
                fn bind(self, graph: &mut Graph) -> [<$name LeaseNode>] {
                    // In this function we are binding a new lease (Lease<Image> or Lease<Buffer> or
                    // etc)

                    // We will return a new node
                    let res = [<$name LeaseNode>]::new(graph.bindings.len());
                    let binding = Binding::[<$name Lease>](Arc::new(self));
                    graph.bindings.push(binding);

                    res
                }
            }

            impl<'a> Bind<&mut Graph, [<$name LeaseNode>]> for &'a Arc<Lease<$name>> {
                fn bind(self, graph: &mut Graph) -> [<$name LeaseNode>] {
                    // In this function we are binding a borrowed binding (&Arc<Lease<Image>> or
                    // &Arc<Lease<Buffer>> or etc)

                    Arc::clone(self).bind(graph)
                }
            }

            impl Bind<&mut Graph, [<$name LeaseNode>]> for Arc<Lease<$name>> {
                #[profiling::function]
                fn bind(self, graph: &mut Graph) -> [<$name LeaseNode>] {
                    // In this function we are binding an existing lease binding
                    // (Arc<Lease<Image>> or Arc<Lease<Buffer>> or etc)

                    // We will return an existing node, if possible
                    // TODO: Could store a sorted list of these shared pointers to avoid the O(N)
                    for (idx, existing_binding) in graph.bindings.iter_mut().enumerate() {
                        if let Some(existing_binding) = existing_binding.[<as_ $name:snake _lease_mut>]() {
                            if Arc::ptr_eq(existing_binding, &self) {
                                return [<$name LeaseNode>]::new(idx);
                            }
                        }
                    }

                    // We will return a new node
                    let res = [<$name LeaseNode>]::new(graph.bindings.len());
                    let binding = Binding::[<$name Lease>](self);
                    graph.bindings.push(binding);

                    res
                }
            }

            impl Binding {
                pub(super) fn [<as_ $name:snake _lease>](&self) -> Option<&Arc<Lease<$name>>> {
                    if let Self::[<$name Lease>](binding) = self {
                        Some(binding)
                    } else {
                        None
                    }
                }

                pub(super) fn [<as_ $name:snake _lease_mut>](&mut self) -> Option<&Arc<Lease<$name>>> {
                    if let Self::[<$name Lease>](binding) = self {
                        Some(binding)
                    } else {
                        None
                    }
                }
            }
        }
    }
}

bind_lease!(AccelerationStructure);
bind_lease!(Image);
bind_lease!(Buffer);

/// A trait for resources which may be borrowed from a `Graph`.
///
/// See [`Graph::node`] for details.
pub trait Bound<Graph, Binding> {
    /// Borrows the resource from a graph.
    fn borrow(self, graph: &Graph) -> &Binding;
}
