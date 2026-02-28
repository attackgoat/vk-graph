use {
    super::{
        AccessType, BindGraph, Bound, CommandRef, Descriptor, Graph, Node, SubresourceRange, View,
        ViewInfo,
    },
    std::marker::PhantomData,
};

/// A render pass which has been bound to a particular compute, graphic, or ray-trace pipeline.
pub struct PipelineCommandRef<'a, T> {
    pub(super) __: PhantomData<T>,
    pub(super) cmd: CommandRef<'a>,
}

// NOTE: There are specific implementations of T in the compute, graphic, and ray trace modules
impl<'a, T> PipelineCommandRef<'a, T> {
    /// Binds a Vulkan buffer, image, or acceleration structure resource to the graph associated
    /// with this command.
    ///
    /// Bound nodes may be used in passes for pipeline and shader operations.
    pub fn bind_resource<R>(&mut self, resource: R) -> R::Node
    where
        R: BindGraph,
    {
        self.cmd.bind_resource(resource)
    }

    /// Finalizes a command and returns the graph so that additional commands may be added.
    pub fn end_cmd(self) -> &'a mut Graph {
        self.cmd.end_cmd()
    }

    /// Returns a borrow of the original Vulkan resource (buffer, image or acceleration structure)
    /// which the given node represents.
    pub fn resource<N>(&self, resource: N) -> &N::Resource
    where
        N: Bound,
    {
        self.cmd.resource(resource)
    }

    /// Informs the command that the next recorded command buffer will read or write `node` using
    /// `access`.
    ///
    /// This function must be called for `node` before it is used within a `record_`-function.
    pub fn resource_access<N>(mut self, resource: N, access: AccessType) -> Self
    where
        N: Node + View,
        SubresourceRange: From<N::Range>,
    {
        self.cmd.set_resource_access(resource, access);
        self
    }

    /// Informs the command that the next recorded command buffer will read or write `node` using
    /// `access`.
    ///
    /// This function must be called for `node` before it is used within a `record_`-function.
    pub fn set_resource_access<N>(&mut self, resource: N, access: AccessType) -> &mut Self
    where
        N: Node + View,
        SubresourceRange: From<N::Range>,
    {
        self.cmd.set_resource_access(resource, access);
        self
    }

    /// Informs the command that the next recorded command buffer will read or write the `resource`
    /// at the specified shader `descriptor` using `access`.
    ///
    /// This function must be called for `resource` before it is used within a `record_`-function.
    pub fn set_shader_resource_access<N>(
        &mut self,
        descriptor: impl Into<Descriptor>,
        resource: N,
        access: AccessType,
    ) -> &mut Self
    where
        N: Node + View,
        N::Info: Copy,
        SubresourceRange: From<N::Info>,
        ViewInfo: From<N::Info>,
    {
        let subresource = resource.info(&self.cmd.graph.resources);

        self.set_shader_subresource_access(descriptor, resource, subresource, access)
    }

    /// Informs the command that the next recorded command buffer will read or write the `node` at
    /// the specified shader `descriptor` using `access`. The node will be interpreted
    /// using `view_info`.
    ///
    /// This function must be called for `node` before it is used within a `record_`-function.
    pub fn set_shader_subresource_access<N>(
        &mut self,
        descriptor: impl Into<Descriptor>,
        resource: N,
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
        let node_idx = resource.index();

        self.cmd
            .push_subresource_access(resource, SubresourceRange::from(subresource), access);

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
    /// `subresource` of `node` using `access`.
    ///
    /// This function must be called for `node` before it is used within a `record_`-function.
    pub fn set_subresource_access<N>(
        &mut self,
        resource: N,
        subresource: impl Into<N::Range>,
        access: AccessType,
    ) -> &mut Self
    where
        N: Node + View,
        SubresourceRange: From<N::Range>,
    {
        self.cmd
            .set_subresource_access(resource, subresource, access);
        self
    }

    /// Informs the command that the next recorded command buffer will read or write the `node` at
    /// the specified shader `descriptor` using `access`.
    ///
    /// This function must be called for `node` before it is used within a `record_`-function.
    pub fn shader_resource_access<N>(
        mut self,
        descriptor: impl Into<Descriptor>,
        resource: N,
        access: AccessType,
    ) -> Self
    where
        N: Node + View,
        N::Info: Copy,
        SubresourceRange: From<N::Info>,
        ViewInfo: From<N::Info>,
    {
        self.set_shader_resource_access(descriptor, resource, access);
        self
    }

    /// Informs the command that the next recorded command buffer will read or write the `node` at
    /// the specified shader `descriptor` using `access`. The node will be interpreted
    /// using `view_info`.
    ///
    /// This function must be called for `node` before it is used within a `record_`-function.
    pub fn shader_subresource_access<N>(
        mut self,
        descriptor: impl Into<Descriptor>,
        resource: N,
        subresource: impl Into<N::Info>,
        access: AccessType,
    ) -> Self
    where
        N: Node + View,
        N::Info: Copy,
        SubresourceRange: From<N::Info>,
        ViewInfo: From<N::Info>,
    {
        self.set_shader_subresource_access(descriptor, resource, subresource, access);
        self
    }

    /// Informs the command that the next recorded command buffer will read or write the
    /// `subresource` of `node` using `access`.
    ///
    /// This function must be called for `node` before it is used within a `record_`-function.
    pub fn subresource_access<N>(
        mut self,
        resource: N,
        subresource: impl Into<N::Range>,
        access: AccessType,
    ) -> Self
    where
        N: Node + View,
        SubresourceRange: From<N::Range>,
    {
        self.cmd
            .set_subresource_access(resource, subresource, access);
        self
    }
}

#[allow(unused)]
mod deprecated {
    use {
        crate::{
            cmd_ref::{PipelineCommandRef, SubresourceRange, View},
            deprecated::Info,
            node::Node,
        },
        ash::vk,
        vk_sync::AccessType,
    };

    impl<'a, T> PipelineCommandRef<'a, T> {
        #[deprecated = "use resource_access function"]
        #[doc(hidden)]
        pub fn access_resource<N>(mut self, node: N, access: AccessType) -> Self
        where
            N: Node + View,
            SubresourceRange: From<N::Range>,
        {
            self.resource_access(node, access)
        }

        #[deprecated = "use subresource_access function"]
        #[doc(hidden)]
        pub fn access_subresource<N>(
            mut self,
            node: N,
            subresource: impl Into<N::Range>,
            access: AccessType,
        ) -> Self
        where
            N: Node + View,
            SubresourceRange: From<N::Range>,
        {
            self.subresource_access(node, subresource, access)
        }

        #[deprecated = "use device_address function of resource function result"]
        #[doc(hidden)]
        pub fn node_device_address(&self, node: impl Node) -> vk::DeviceAddress {
            let idx = node.index();

            self.cmd.graph.resources[idx]
                .as_driver_buffer()
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
    }
}
