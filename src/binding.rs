use {
    super::{
        AccelerationStructureLeaseNode, AccelerationStructureNode, BufferLeaseNode, BufferNode,
        ImageLeaseNode, ImageNode, RenderGraph,
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

// #[derive(Debug)]
// pub enum AnyBufferBinding<'a> {
//     Buffer(&'a Arc<Buffer>),
//     BufferLease(&'a Lease<Buffer>),
// }

// impl<'a> From<&'a Arc<Buffer>> for AnyBufferBinding<'a> {
//     fn from(buffer: &'a Arc<Buffer>) -> Self {
//         Self::Buffer(buffer)
//     }
// }

// impl<'a> From<&'a Lease<Buffer>> for AnyBufferBinding<'a> {
//     fn from(buffer: &'a Lease<Buffer>) -> Self {
//         Self::BufferLease(buffer)
//     }
// }

// #[derive(Debug)]
// pub enum AnyImageBinding<'a> {
//     Image(&'a Arc<Image>),
//     ImageLease(&'a Lease<Image>),
//     SwapchainImage(&'a SwapchainImage),
// }

// impl<'a> From<&'a Arc<Image>> for AnyImageBinding<'a> {
//     fn from(image: &'a Arc<Image>) -> Self {
//         Self::Image(image)
//     }
// }

// impl<'a> From<&'a Lease<Image>> for AnyImageBinding<'a> {
//     fn from(image: &'a Lease<Image>) -> Self {
//         Self::ImageLease(image)
//     }
// }

// impl<'a> From<&'a SwapchainImage> for AnyImageBinding<'a> {
//     fn from(image: &'a SwapchainImage) -> Self {
//         Self::SwapchainImage(image)
//     }
// }

/// A trait for resources which may be bound to a `RenderGraph`.
///
/// See [`RenderGraph::bind_node`] and
/// [`PassRef::bind_pipeline`](super::pass_ref::PassRef::bind_pipeline) for details.
pub trait Bind<Graph, Node> {
    /// Binds the resource to a graph-like object.
    ///
    /// Returns a reference Node object.
    fn bind(self, graph: Graph) -> Node;
}

#[derive(Debug)]
pub enum Binding {
    AccelerationStructure(Arc<AccelerationStructure>, bool),
    AccelerationStructureLease(Arc<Lease<AccelerationStructure>>, bool),
    Buffer(Arc<Buffer>, bool),
    BufferLease(Arc<Lease<Buffer>>, bool),
    Image(Arc<Image>, bool),
    ImageLease(Arc<Lease<Image>>, bool),
    SwapchainImage(Box<SwapchainImage>, bool),
}

impl Binding {
    pub(super) fn as_driver_acceleration_structure(&self) -> Option<&AccelerationStructure> {
        Some(match self {
            Self::AccelerationStructure(binding, _) => binding,
            Self::AccelerationStructureLease(binding, _) => binding,
            _ => return None,
        })
    }

    pub(super) fn as_driver_buffer(&self) -> Option<&Buffer> {
        Some(match self {
            Self::Buffer(binding, _) => binding,
            Self::BufferLease(binding, _) => binding,
            _ => return None,
        })
    }

    pub(super) fn as_driver_image(&self) -> Option<&Image> {
        Some(match self {
            Self::Image(binding, _) => binding,
            Self::ImageLease(binding, _) => binding,
            Self::SwapchainImage(binding, _) => binding,
            _ => return None,
        })
    }

    pub(super) fn is_bound(&self) -> bool {
        match self {
            Self::AccelerationStructure(_, is_bound) => *is_bound,
            Self::AccelerationStructureLease(_, is_bound) => *is_bound,
            Self::Buffer(_, is_bound) => *is_bound,
            Self::BufferLease(_, is_bound) => *is_bound,
            Self::Image(_, is_bound) => *is_bound,
            Self::ImageLease(_, is_bound) => *is_bound,
            Self::SwapchainImage(_, is_bound) => *is_bound,
        }
    }

    pub(super) fn unbind(&mut self) {
        *match self {
            Self::AccelerationStructure(_, is_bound) => is_bound,
            Self::AccelerationStructureLease(_, is_bound) => is_bound,
            Self::Buffer(_, is_bound) => is_bound,
            Self::BufferLease(_, is_bound) => is_bound,
            Self::Image(_, is_bound) => is_bound,
            Self::ImageLease(_, is_bound) => is_bound,
            Self::SwapchainImage(_, is_bound) => is_bound,
        } = false;
    }
}

macro_rules! bind {
    ($name:ident) => {
        paste::paste! {
            impl Bind<&mut RenderGraph, [<$name Node>]> for $name {
                #[profiling::function]
                fn bind(self, graph: &mut RenderGraph) -> [<$name Node>] {
                    // In this function we are binding a new item (Image or Buffer or etc)

                    // We will return a new node
                    let res = [<$name Node>]::new(graph.bindings.len());
                    let binding = Binding::$name(Arc::new(self), true);
                    graph.bindings.push(binding);

                    res
                }
            }

            impl<'a> Bind<&mut RenderGraph, [<$name Node>]> for &'a Arc<$name> {
                fn bind(self, graph: &mut RenderGraph) -> [<$name Node>] {
                    // In this function we are binding a borrowed binding (&Arc<Image> or
                    // &Arc<Buffer> or etc)

                    Arc::clone(self).bind(graph)
                }
            }

            impl Bind<&mut RenderGraph, [<$name Node>]> for Arc<$name> {
                #[profiling::function]
                fn bind(self, graph: &mut RenderGraph) -> [<$name Node>] {
                    // In this function we are binding an existing binding (Arc<Image> or
                    // Arc<Buffer> or etc)

                    // We will return an existing node, if possible
                    // TODO: Could store a sorted list of these shared pointers to avoid the O(N)
                    for (idx, existing_binding) in graph.bindings.iter_mut().enumerate() {
                        if let Some((existing_binding, is_bound)) = existing_binding.[<as_ $name:snake _mut>]() {
                            if Arc::ptr_eq(existing_binding, &self) {
                                *is_bound = true;

                                return [<$name Node>]::new(idx);
                            }
                        }
                    }

                    // Return a new node
                    let res = [<$name Node>]::new(graph.bindings.len());
                    let binding = Binding::$name(self, true);
                    graph.bindings.push(binding);

                    res
                }
            }

            impl Binding {
                pub(super) fn [<as_ $name:snake>](&self) -> Option<&Arc<$name>> {
                    if let Self::$name(binding, _) = self {
                        Some(&binding)
                    } else {
                        None
                    }
                }

                pub(super) fn [<as_ $name:snake _mut>](&mut self) -> Option<(&mut Arc<$name>, &mut bool)> {
                    if let Self::$name(binding, is_bound) = self {
                        Some((binding, is_bound))
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
            impl Bind<&mut RenderGraph, [<$name LeaseNode>]> for Lease<$name> {
                #[profiling::function]
                fn bind(self, graph: &mut RenderGraph) -> [<$name LeaseNode>] {
                    // In this function we are binding a new lease (Lease<Image> or Lease<Buffer> or
                    // etc)

                    // We will return a new node
                    let res = [<$name LeaseNode>]::new(graph.bindings.len());
                    let binding = Binding::[<$name Lease>](Arc::new(self), true);
                    graph.bindings.push(binding);

                    res
                }
            }

            impl<'a> Bind<&mut RenderGraph, [<$name LeaseNode>]> for &'a Arc<Lease<$name>> {
                fn bind(self, graph: &mut RenderGraph) -> [<$name LeaseNode>] {
                    // In this function we are binding a borrowed binding (&Arc<Lease<Image>> or
                    // &Arc<Lease<Buffer>> or etc)

                    Arc::clone(self).bind(graph)
                }
            }

            impl Bind<&mut RenderGraph, [<$name LeaseNode>]> for Arc<Lease<$name>> {
                #[profiling::function]
                fn bind(self, graph: &mut RenderGraph) -> [<$name LeaseNode>] {
                    // In this function we are binding an existing lease binding
                    // (Arc<Lease<Image>> or Arc<Lease<Buffer>> or etc)

                    // We will return an existing node, if possible
                    // TODO: Could store a sorted list of these shared pointers to avoid the O(N)
                    for (idx, existing_binding) in graph.bindings.iter_mut().enumerate() {
                        if let Some((existing_binding, is_bound)) = existing_binding.[<as_ $name:snake _lease_mut>]() {
                            if Arc::ptr_eq(existing_binding, &self) {
                                *is_bound = true;

                                return [<$name LeaseNode>]::new(idx);
                            }
                        }
                    }

                    // We will return a new node
                    let res = [<$name LeaseNode>]::new(graph.bindings.len());
                    let binding = Binding::[<$name Lease>](self, true);
                    graph.bindings.push(binding);

                    res
                }
            }

            impl Binding {
                pub(super) fn [<as_ $name:snake _lease>](&self) -> Option<&Arc<Lease<$name>>> {
                    if let Self::[<$name Lease>](binding, _) = self {
                        Some(binding)
                    } else {
                        None
                    }
                }

                pub(super) fn [<as_ $name:snake _lease_mut>](&mut self) -> Option<(&Arc<Lease<$name>>, &mut bool)> {
                    if let Self::[<$name Lease>](binding, is_bound) = self {
                        Some((binding, is_bound))
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

/// A trait for resources which may be unbound from a `RenderGraph`.
///
/// See [`RenderGraph::unbind_node`] for details.
pub trait Unbind<Graph, Binding> {
    /// Unbinds the resource from a graph-like object.
    ///
    /// Returns the original Binding object.
    fn unbind(self, graph: &mut Graph) -> Binding;
}
