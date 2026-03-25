use {
    super::{
        AccessType, Command, Descriptor, Graph, Node, Resource, SubresourceRange, View, ViewInfo,
    },
    crate::{
        ExecutionPipeline,
        driver::{compute::ComputePipeline, graphic::GraphicPipeline, ray_trace::RayTracePipeline},
    },
    std::marker::PhantomData,
};

/// A trait for pipelines which may be bound to a `Command`.
///
/// See [`Command::bind_pipeline`](crate::cmd::Command::bind_pipeline) for details.
pub trait Pipeline<'a> {
    /// The resource reference type.
    type Command;

    /// Binds the resource to a command.
    ///
    /// Returns a reference type.
    fn bind_cmd(self, _: Command<'a>) -> Self::Command;
}

macro_rules! pipeline {
    ($name:ident) => {
        paste::paste! {
            impl<'a> Pipeline<'a> for [<$name Pipeline>] {
                type Command = PipelineCommand<'a, [<$name Pipeline>]>;

                fn bind_cmd(self, mut cmd: Command<'a>) -> Self::Command {
                    {
                        let cmd = cmd.cmd_mut();
                        if cmd.execs.last().unwrap().pipeline.is_some() {
                            cmd.execs.push(Default::default());
                        }

                        cmd.execs.last_mut().unwrap().pipeline
                            = Some(ExecutionPipeline::$name(self));
                    }

                    Self::Command {
                        __: PhantomData,
                        cmd,
                    }
                }
            }

            impl<'a> Pipeline<'a> for &'a [<$name Pipeline>] {
                type Command = PipelineCommand<'a, [<$name Pipeline>]>;

                fn bind_cmd(self, mut cmd: Command<'a>) -> Self::Command {
                    {
                        let cmd = cmd.cmd_mut();
                        if cmd.execs.last().unwrap().pipeline.is_some() {
                            cmd.execs.push(Default::default());
                        }

                        cmd.execs.last_mut().unwrap().pipeline
                            = Some(ExecutionPipeline::$name(self.clone()));
                    }

                    Self::Command {
                        __: PhantomData,
                        cmd,
                    }
                }

            }

            impl ExecutionPipeline {
                #[allow(unused)]
                pub(crate) fn [<is_ $name:snake>](&self) -> bool {
                    matches!(self, Self::$name(_))
                }

                #[allow(unused)]
                pub(crate) fn [<unwrap_ $name:snake>](&self) -> &[<$name Pipeline>] {
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

// Pipelines you can bind to a command ref
pipeline!(Compute);
pipeline!(Graphic);
pipeline!(RayTrace);

/// A [`Command`] which has been bound to a particular compute, graphic, or ray-trace pipeline.
pub struct PipelineCommand<'c, T> {
    pub(super) __: PhantomData<T>,
    pub(super) cmd: Command<'c>,
}

// NOTE: There are specific implementations of T in the compute, graphic, and ray trace modules
impl<'c, T> PipelineCommand<'c, T> {
    /// Binds a shader pipeline to the current command, allowing for strongly typed access to the
    /// related functions.
    ///
    /// `P`|`P::Command`
    /// -|-
    /// [`ComputePipeline`](crate::driver::compute::ComputePipeline)|[`PipelineCommand<'_, ComputePipeline>`]
    /// [`GraphicPipeline`](crate::driver::graphic::GraphicPipeline)|[`PipelineCommand<'_, GraphicPipeline>`]
    /// [`RayTracePipeline`](crate::driver::ray_trace::RayTracePipeline)|[`PipelineCommand<'_, RayTracePipeline>`]
    pub fn bind_pipeline<P>(self, pipeline: P) -> P::Command
    where
        P: Pipeline<'c>,
    {
        pipeline.bind_cmd(self.cmd)
    }

    /// Binds a Vulkan buffer, image, or acceleration structure resource to the graph associated
    /// with this command.
    ///
    /// Bound nodes may be used in passes for pipeline and shader operations.
    pub fn bind_resource<R>(&mut self, resource: R) -> R::Node
    where
        R: Resource,
    {
        self.cmd.bind_resource(resource)
    }

    /// Finalizes a command and returns the graph so that additional commands may be added.
    pub fn end_cmd(self) -> &'c mut Graph {
        self.cmd.end_cmd()
    }

    /// Returns a borrow of the original Vulkan resource (buffer, image or acceleration structure)
    /// which the given bound resource node represents.
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: Node,
    {
        self.cmd.resource(resource_node)
    }

    /// Informs the command that the next recorded command buffer will read or write `resource_node`
    /// using `access`.
    ///
    /// An access function must be called for `resource_node` before it is used within a
    /// `record_`-function.
    pub fn resource_access<N>(mut self, resource_node: N, access: AccessType) -> Self
    where
        N: Node + View,
        SubresourceRange: From<N::Range>,
    {
        self.cmd.set_resource_access(resource_node, access);
        self
    }

    /// Informs the command that the next recorded command buffer will read or write `resource_node`
    /// using `access`.
    ///
    /// An access function must be called for `resource_node` before it is used within a
    /// `record_`-function.
    pub fn set_resource_access<N>(&mut self, resource_node: N, access: AccessType) -> &mut Self
    where
        N: Node + View,
        SubresourceRange: From<N::Range>,
    {
        self.cmd.set_resource_access(resource_node, access);
        self
    }

    /// Informs the command that the next recorded command buffer will read or write the `resource_node`
    /// at the specified shader `descriptor` using `access`.
    ///
    /// An access function must be called for `resource_node` before it is used within a
    /// `record_`-function.
    pub fn set_shader_resource_access<N>(
        &mut self,
        descriptor: impl Into<Descriptor>,
        resource_node: N,
        access: AccessType,
    ) -> &mut Self
    where
        N: Node + View,
        N::Info: Copy,
        SubresourceRange: From<N::Info>,
        ViewInfo: From<N::Info>,
    {
        let subresource = resource_node.info(&self.cmd.graph.resources);

        self.set_shader_subresource_access(descriptor, resource_node, subresource, access)
    }

    /// Informs the command that the next recorded command buffer will read or write the
    /// `resource_node` at the specified shader `descriptor` using `access`. The resource will be
    /// interpreted using `view_info`.
    ///
    /// An access function must be called for `resource_node` before it is used within a
    /// `record_`-function.
    pub fn set_shader_subresource_access<N>(
        &mut self,
        descriptor: impl Into<Descriptor>,
        resource_node: N,
        subresource: impl Into<N::Info>,
        access: AccessType,
    ) -> &mut Self
    where
        N: Node + View,
        N::Info: Copy,
        SubresourceRange: From<N::Info>,
        ViewInfo: From<N::Info>,
    {
        let descriptor = descriptor.into();
        let subresource = subresource.into();
        let node_idx = resource_node.index();

        self.cmd.push_subresource_access(
            resource_node,
            SubresourceRange::from(subresource),
            access,
        );

        assert!(
            self.cmd
                .cmd_mut()
                .execs
                .last_mut()
                .unwrap()
                .bindings
                .insert(descriptor, (node_idx, subresource.into()))
                .is_none(),
            "descriptor {descriptor:?} has already been bound"
        );

        self
    }

    /// Informs the command that the next recorded command buffer will read or write the
    /// `subresource` of `resource_node` using `access`.
    ///
    /// An access function must be called for `resource_node` before it is used within a
    /// `record_`-function.
    pub fn set_subresource_access<N>(
        &mut self,
        resource_node: N,
        subresource: impl Into<N::Range>,
        access: AccessType,
    ) -> &mut Self
    where
        N: Node + View,
        SubresourceRange: From<N::Range>,
    {
        self.cmd
            .set_subresource_access(resource_node, subresource, access);
        self
    }

    /// Informs the command that the next recorded command buffer will read or write the
    /// `resource_node` at the specified shader `descriptor` using `access`.
    ///
    /// An access function must be called for `resource_node` before it is used within a
    /// `record_`-function.
    pub fn shader_resource_access<N>(
        mut self,
        descriptor: impl Into<Descriptor>,
        resource_node: N,
        access: AccessType,
    ) -> Self
    where
        N: Node + View,
        N::Info: Copy,
        SubresourceRange: From<N::Info>,
        ViewInfo: From<N::Info>,
    {
        self.set_shader_resource_access(descriptor, resource_node, access);
        self
    }

    /// Informs the command that the next recorded command buffer will read or write the
    /// `resource_node` at the specified shader `descriptor` using `access`. The resource will be
    /// interpreted using `view_info`.
    ///
    /// An access function must be called for `resource_node` before it is used within a
    /// `record_`-function.
    pub fn shader_subresource_access<N>(
        mut self,
        descriptor: impl Into<Descriptor>,
        resource_node: N,
        subresource: impl Into<N::Info>,
        access: AccessType,
    ) -> Self
    where
        N: Node + View,
        N::Info: Copy,
        SubresourceRange: From<N::Info>,
        ViewInfo: From<N::Info>,
    {
        self.set_shader_subresource_access(descriptor, resource_node, subresource, access);
        self
    }

    /// Informs the command that the next recorded command buffer will read or write the
    /// `subresource` of `resource_node` using `access`.
    ///
    /// An access function must be called for `resource_node` before it is used within a
    /// `record_`-function.
    pub fn subresource_access<N>(
        mut self,
        resource_node: N,
        subresource: impl Into<N::Range>,
        access: AccessType,
    ) -> Self
    where
        N: Node + View,
        SubresourceRange: From<N::Range>,
    {
        self.cmd
            .set_subresource_access(resource_node, subresource, access);
        self
    }
}

#[allow(deprecated)]
#[allow(unused)]
mod deprecated {
    use {
        crate::{
            Graph, Node, Resource,
            cmd::{Descriptor, PipelineCommand, SubresourceRange, View, ViewInfo},
            deprecated::Info,
            graph::pass_ref::ViewType,
        },
        ash::vk,
        vk_sync::AccessType,
    };

    impl<'a, T> PipelineCommand<'a, T> {
        #[deprecated = "use shader_resource_access function"]
        #[doc(hidden)]
        pub fn access_descriptor<N>(
            self,
            descriptor: impl Into<Descriptor>,
            node: N,
            access: AccessType,
        ) -> Self
        where
            N: Node + Info + View,
            ViewType: From<<N as View>::Info>,
            <N as View>::Info: Copy + From<<N as Info>::Type>,
            <N as View>::Range: From<<N as View>::Info>,
        {
            let view_info = View::info(&node, &self.cmd.graph.resources);

            self.access_descriptor_as(descriptor, node, access, view_info)
        }

        #[deprecated = "use shader_subresource_access function"]
        #[doc(hidden)]
        pub fn access_descriptor_as<N>(
            self,
            descriptor: impl Into<Descriptor>,
            node: N,
            access: AccessType,
            view_info: impl Into<N::Info>,
        ) -> Self
        where
            N: View,
            <N as View>::Info: Copy + Into<ViewType>,
            <N as View>::Range: From<<N as View>::Info>,
        {
            let view_info = view_info.into();
            let subresource = <N as View>::Range::from(view_info);

            self.access_descriptor_subrange(descriptor, node, access, view_info, subresource)
        }

        #[deprecated = "use shader_subresource_access function"]
        #[doc(hidden)]
        pub fn access_descriptor_subrange<N>(
            self,
            descriptor: impl Into<Descriptor>,
            node: N,
            access: AccessType,
            view_info: impl Into<N::Info>,
            subresource: impl Into<N::Range>,
        ) -> Self
        where
            N: View,
            <N as View>::Info: Into<ViewType>,
        {
            unimplemented!()
        }

        #[deprecated = "use resource_access function"]
        #[doc(hidden)]
        pub fn access_node<N>(mut self, node: N, access: AccessType) -> Self
        where
            N: Node + View,
            SubresourceRange: From<N::Range>,
        {
            self.resource_access(node, access)
        }

        #[deprecated = "use subresource_access function"]
        #[doc(hidden)]
        pub fn access_node_subrange<N>(
            mut self,
            node: N,
            access: AccessType,
            subresource: impl Into<N::Range>,
        ) -> Self
        where
            N: Node + View,
            SubresourceRange: From<N::Range>,
        {
            self.access_node_subrange_mut(node, access, subresource);
            self
        }

        #[deprecated = "use set_subresource_access function"]
        #[doc(hidden)]
        pub fn access_node_subrange_mut<N>(
            &mut self,
            node: N,
            access: AccessType,
            subresource: impl Into<N::Range>,
        ) -> &mut Self
        where
            N: Node + View,
            SubresourceRange: From<N::Range>,
        {
            self.set_subresource_access(node, subresource, access)
        }

        #[deprecated = "use bind_resource function"]
        #[doc(hidden)]
        pub fn bind_node<R>(&mut self, resource: R) -> R::Node
        where
            R: Resource,
        {
            self.bind_resource(resource)
        }

        #[deprecated = "use device_address function of resource function result"]
        #[doc(hidden)]
        pub fn node_device_address(&self, node: impl Node) -> vk::DeviceAddress {
            let idx = node.index();

            self.cmd.graph.resources[idx]
                .as_buffer()
                .unwrap()
                .device_address()
        }

        #[deprecated = "dereference info field of resource function result"]
        #[doc(hidden)]
        pub fn node_info<N>(&self, node: N) -> N::Type
        where
            N: Node + Info,
        {
            node.info(&self.cmd.graph.resources)
        }

        #[deprecated = "use end_cmd function"]
        #[doc(hidden)]
        pub fn submit_pass(self) -> &'a mut Graph {
            self.end_cmd()
        }
    }
}
