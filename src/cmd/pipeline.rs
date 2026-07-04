use {
    super::{
        AccessType, Binding, Command, Graph, Node, Resource, Subresource, SubresourceRange,
        ViewInfo,
    },
    crate::{
        ExecutionPipeline, TimestampQuery,
        driver::{
            compute::ComputePipeline, graphics::GraphicsPipeline, ray_tracing::RayTracingPipeline,
        },
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
    ($variant:ident, $pipeline:ident, $is_fn:ident, $unwrap_fn:ident) => {
        paste::paste! {
            impl<'a> Pipeline<'a> for $pipeline {
                type Command = PipelineCommand<'a, $pipeline>;

                fn bind_cmd(self, mut cmd: Command<'a>) -> Self::Command {
                    {
                        let cmd = cmd.cmd_mut();
                        if cmd.expect_last_exec().pipeline.is_some() {
                            cmd.execs.push(Default::default());
                        }

                        cmd.expect_last_exec_mut().pipeline = Some(ExecutionPipeline::$variant(self));
                    }

                    Self::Command {
                        __: PhantomData,
                        cmd,
                    }
                }
            }

            impl<'a> Pipeline<'a> for &'a $pipeline {
                type Command = PipelineCommand<'a, $pipeline>;

                fn bind_cmd(self, mut cmd: Command<'a>) -> Self::Command {
                    {
                        let cmd = cmd.cmd_mut();
                        if cmd.expect_last_exec().pipeline.is_some() {
                            cmd.execs.push(Default::default());
                        }

                        cmd.expect_last_exec_mut().pipeline
                            = Some(ExecutionPipeline::$variant(self.clone()));
                    }

                    Self::Command {
                        __: PhantomData,
                        cmd,
                    }
                }

            }

            impl ExecutionPipeline {
                #[allow(unused)]
                pub(crate) fn $is_fn(&self) -> bool {
                    matches!(self, Self::$variant(_))
                }

                #[allow(unused)]
                pub(crate) fn $unwrap_fn(&self) -> &$pipeline {
                    if let Self::$variant(binding) = self {
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
pipeline!(Compute, ComputePipeline, is_compute, unwrap_compute);
pipeline!(Graphics, GraphicsPipeline, is_graphics, unwrap_graphics);
pipeline!(
    RayTracing,
    RayTracingPipeline,
    is_ray_tracing,
    unwrap_ray_tracing
);

/// A [`Command`] which has been bound to a particular compute, graphics, or ray tracing pipeline.
pub struct PipelineCommand<'c, T> {
    pub(super) __: PhantomData<T>,
    pub(super) cmd: Command<'c>,
}

// NOTE: There are specific implementations of T in the compute, graphics, and ray tracing modules
#[allow(private_bounds)]
impl<'c, T> PipelineCommand<'c, T> {
    /// Equivalent to [`Command::bind_pipeline`] for a command that already has a bound pipeline.
    pub fn bind_pipeline<P>(self, pipeline: P) -> P::Command
    where
        P: Pipeline<'c>,
    {
        pipeline.bind_cmd(self.cmd)
    }

    /// Equivalent to [`Command::bind_resource`] for a command that already has a bound pipeline.
    pub fn bind_resource<R>(&mut self, resource: R) -> R::Node
    where
        R: Resource,
    {
        self.cmd.bind_resource(resource)
    }

    /// Equivalent to [`Command::write_timestamp`] for a command that already has a bound pipeline.
    pub fn write_timestamp(&mut self) -> TimestampQuery {
        self.cmd.write_timestamp()
    }

    /// Equivalent to [`Command::end_cmd`] for a command that already has a bound pipeline.
    pub fn end_cmd(self) -> &'c mut Graph {
        self.cmd.end_cmd()
    }

    /// Equivalent to [`Command::resource`] for a command that already has a bound pipeline.
    pub fn resource<N>(&self, resource_node: N) -> &N::Resource
    where
        N: Node,
    {
        self.cmd.resource(resource_node)
    }

    /// Informs the command that recorded work will read or write `resource_node` using `access`.
    ///
    /// An access function must be called for `resource_node` before it is used within a recording
    /// function.
    pub fn resource_access<N>(mut self, resource_node: N, access: AccessType) -> Self
    where
        N: Node + Subresource,
        SubresourceRange: From<N::Range>,
    {
        self.cmd.set_resource_access(resource_node, access);
        self
    }

    /// Mutable-borrow form of [`Self::resource_access`].
    pub fn set_resource_access<N>(&mut self, resource_node: N, access: AccessType) -> &mut Self
    where
        N: Node + Subresource,
        SubresourceRange: From<N::Range>,
    {
        self.cmd.set_resource_access(resource_node, access);
        self
    }

    /// Mutable-borrow form of [`Self::shader_resource_access`].
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

    /// Mutable-borrow form of [`Self::shader_subresource_access`].
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
        let view_info = subresource.into();

        self.cmd.push_subresource_access(
            resource_node,
            SubresourceRange::from(subresource),
            access,
        );

        #[cfg(feature = "checked")]
        {
            if let Some(prev) = self.cmd.cmd().expect_last_exec().bindings.get(&binding) {
                assert!(
                    *prev == (node_idx, view_info),
                    "shader binding {binding:?} already bound to a different resource or view"
                );
            }
        }

        self.cmd
            .cmd_mut()
            .expect_last_exec_mut()
            .bindings
            .insert(binding, (node_idx, view_info));

        self
    }

    /// Mutable-borrow form of [`Self::subresource_access`].
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

    /// Informs the command that recorded work will read or write the `resource_node` at the
    /// specified shader `binding` using `access`.
    ///
    /// If the same `binding` slot is used more than once, the last call wins and the previous
    /// binding is silently overwritten.
    ///
    /// An access function must be called for `resource_node` before it is used within a recording
    /// function.
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

    /// Informs the command that recorded work will read or write the `resource_node` at the
    /// specified shader `binding` using `access`. The resource will be interpreted using
    /// `view_info`.
    ///
    /// If the same `binding` slot is used more than once, the last call wins and the previous
    /// binding is silently overwritten.
    ///
    /// An access function must be called for `resource_node` before it is used within a recording
    /// function.
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

    /// Informs the command that recorded work will read or write the `subresource` of
    /// `resource_node` using `access`.
    ///
    /// An access function must be called for `resource_node` before it is used within a recording
    /// function.
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
