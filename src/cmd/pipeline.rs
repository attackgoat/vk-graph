use {
    super::{
        AccessType, Binding, Command, Graph, Node, Resource, Subresource, SubresourceRange,
        ViewInfo,
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
                        if cmd.expect_last_exec().pipeline.is_some() {
                            cmd.execs.push(Default::default());
                        }

                        cmd.expect_last_exec_mut().pipeline
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
                        if cmd.expect_last_exec().pipeline.is_some() {
                            cmd.execs.push(Default::default());
                        }

                        cmd.expect_last_exec_mut().pipeline
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
    /// [`ComputePipeline`](crate::driver::compute::ComputePipeline)|[`PipelineCommand<'_,
    /// ComputePipeline>`]
    /// [`GraphicPipeline`](crate::driver::graphic::GraphicPipeline)|[`PipelineCommand<'_,
    /// GraphicPipeline>`]
    /// [`RayTracePipeline`](crate::driver::ray_trace::RayTracePipeline)|[`PipelineCommand<'_,
    /// RayTracePipeline>`]
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
        N: Node + Subresource,
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
        N: Node + Subresource,
        SubresourceRange: From<N::Range>,
    {
        self.cmd.set_resource_access(resource_node, access);
        self
    }

    /// Informs the command that the next recorded command buffer will read or write the
    /// `resource_node` at the specified shader `binding` using `access`.
    ///
    /// An access function must be called for `resource_node` before it is used within a
    /// `record_`-function.
    pub fn set_shader_resource_access<N>(
        &mut self,
        binding: impl Into<Binding>,
        resource_node: N,
        access: AccessType,
    ) -> &mut Self
    where
        N: Node + Subresource,
        N::Info: Copy,
        SubresourceRange: From<N::Info>,
        ViewInfo: From<N::Info>,
    {
        let subresource = resource_node.info(&self.cmd.graph.resources);

        self.set_shader_subresource_access(binding, resource_node, subresource, access)
    }

    /// Informs the command that the next recorded command buffer will read or write the
    /// `resource_node` at the specified shader `binding` using `access`. The resource will be
    /// interpreted using `view_info`.
    ///
    /// An access function must be called for `resource_node` before it is used within a
    /// `record_`-function.
    pub fn set_shader_subresource_access<N>(
        &mut self,
        binding: impl Into<Binding>,
        resource_node: N,
        subresource: impl Into<N::Info>,
        access: AccessType,
    ) -> &mut Self
    where
        N: Node + Subresource,
        N::Info: Copy,
        SubresourceRange: From<N::Info>,
        ViewInfo: From<N::Info>,
    {
        let binding = binding.into();
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
                .expect_last_exec_mut()
                .bindings
                .insert(binding, (node_idx, subresource.into()))
                .is_none(),
            "binding {binding:?} has already been bound"
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
        N: Node + Subresource,
        SubresourceRange: From<N::Range>,
    {
        self.cmd
            .set_subresource_access(resource_node, subresource, access);
        self
    }

    /// Informs the command that the next recorded command buffer will read or write the
    /// `resource_node` at the specified shader `binding` using `access`.
    ///
    /// An access function must be called for `resource_node` before it is used within a
    /// `record_`-function.
    pub fn shader_resource_access<N>(
        mut self,
        binding: impl Into<Binding>,
        resource_node: N,
        access: AccessType,
    ) -> Self
    where
        N: Node + Subresource,
        N::Info: Copy,
        SubresourceRange: From<N::Info>,
        ViewInfo: From<N::Info>,
    {
        self.set_shader_resource_access(binding, resource_node, access);
        self
    }

    /// Informs the command that the next recorded command buffer will read or write the
    /// `resource_node` at the specified shader `binding` using `access`. The resource will be
    /// interpreted using `view_info`.
    ///
    /// An access function must be called for `resource_node` before it is used within a
    /// `record_`-function.
    pub fn shader_subresource_access<N>(
        mut self,
        binding: impl Into<Binding>,
        resource_node: N,
        subresource: impl Into<N::Info>,
        access: AccessType,
    ) -> Self
    where
        N: Node + Subresource,
        N::Info: Copy,
        SubresourceRange: From<N::Info>,
        ViewInfo: From<N::Info>,
    {
        self.set_shader_subresource_access(binding, resource_node, subresource, access);
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
        N: Node + Subresource,
        SubresourceRange: From<N::Range>,
    {
        self.cmd
            .set_subresource_access(resource_node, subresource, access);
        self
    }
}
