use {
    super::{
        Access, AccessType, Bind, CommandRef, Descriptor, Edge, Graph, Info, Node, Subresource,
        View, ViewType,
    },
    std::marker::PhantomData,
};

/// A render pass which has been bound to a particular compute, graphic, or ray-trace pipeline.
pub struct PipelineCommandRef<'a, T> {
    pub(super) __: PhantomData<T>,
    pub(super) cmd: CommandRef<'a>,
}

// NOTE: There are specific implementations of T in the compute, graphic, and ray trace modules
impl<'a, T> PipelineCommandRef<'a, T>
where
    T: Access,
{
    /// Informs the pass that the next recorded command buffer will read or write the given `node`
    /// at the specified shader descriptor using `access`.
    ///
    /// This function must be called for `node` before it is read or written within a `record`
    /// function. For general purpose access, see [`PipelineCommandRef::read_descriptor`] or
    /// [`PipelineCommandRef::write_descriptor`].
    pub fn access_descriptor<N>(
        self,
        descriptor: impl Into<Descriptor>,
        node: N,
        access: AccessType,
    ) -> Self
    where
        N: Info,
        N: View,
        ViewType: From<<N as View>::Info>,
        <N as View>::Info: From<<N as Info>::Info>,
        <N as View>::Subresource: From<<N as View>::Info>,
    {
        let node_info = self.node_info(node);

        // Use the plain node information as the whole view of the node
        let view_info = node_info;

        self.access_descriptor_as(descriptor, node, access, view_info)
    }

    /// Informs the pass that the next recorded command buffer will read or write the given `node`
    /// at the specified shader descriptor using `access`. The node will be interpreted using
    /// `view_info`.
    ///
    /// This function must be called for `node` before it is read or written within a `record`
    /// function. For general purpose access, see [`PipelineCommandRef::read_descriptor_as`] or
    /// [`PipelineCommandRef::write_descriptor_as`].
    pub fn access_descriptor_as<N>(
        self,
        descriptor: impl Into<Descriptor>,
        node: N,
        access: AccessType,
        view_info: impl Into<N::Info>,
    ) -> Self
    where
        N: View,
        <N as View>::Info: Into<ViewType>,
        <N as View>::Subresource: From<<N as View>::Info>,
    {
        let view_info = view_info.into();
        let subresource = <N as View>::Subresource::from(view_info);

        self.access_descriptor_subrange(descriptor, node, access, view_info, subresource)
    }

    /// Informs the pass that the next recorded command buffer will read or write the `subresource`
    /// of `node` at the specified shader descriptor using `access`. The node will be interpreted
    /// using `view_info`.
    ///
    /// This function must be called for `node` before it is read or written within a `record`
    /// function. For general purpose access, see [`PipelineCommandRef::read_descriptor_subrange`] or
    /// [`PipelineCommandRef::write_descriptor_subrange`].
    pub fn access_descriptor_subrange<N>(
        mut self,
        descriptor: impl Into<Descriptor>,
        node: N,
        access: AccessType,
        view_info: impl Into<N::Info>,
        subresource: impl Into<N::Subresource>,
    ) -> Self
    where
        N: View,
        <N as View>::Info: Into<ViewType>,
    {
        self.cmd
            .push_node_access(node, access, subresource.into().into());
        self.push_node_view_bind(node, view_info.into(), descriptor.into());

        self
    }

    /// Informs the pass that the next recorded command buffer will read or write the given `node`
    /// using `access`.
    ///
    /// This function must be called for `node` before it is read or written within a `record`
    /// function. For general purpose access, see [`PipelineCommandRef::read_node`] or
    /// [`PipelineCommandRef::write_node`].
    pub fn access_node(mut self, node: impl Node + Info, access: AccessType) -> Self {
        self.access_node_mut(node, access);

        self
    }

    /// Informs the pass that the next recorded command buffer will read or write the given `node`
    /// using `access`.
    ///
    /// This function must be called for `node` before it is read or written within a `record`
    /// function. For general purpose access, see [`PipelineCommandRef::read_node_mut`] or
    /// [`PipelineCommandRef::write_node_mut`].
    pub fn access_node_mut(&mut self, node: impl Node + Info, access: AccessType) {
        self.cmd.assert_bound_graph_node(node);

        let idx = node.index();
        let binding = &self.cmd.graph.bindings[idx];

        let node_access_range = if let Some(buf) = binding.as_driver_buffer() {
            Subresource::Buffer((0..buf.info.size).into())
        } else if let Some(image) = binding.as_driver_image() {
            Subresource::Image(image.info.default_view_info().into())
        } else {
            Subresource::AccelerationStructure
        };

        self.cmd.push_node_access(node, access, node_access_range);
    }

    /// Informs the pass that the next recorded command buffer will read or write the `subresource`
    /// of `node` using `access`.
    ///
    /// This function must be called for `node` before it is read or written within a `record`
    /// function. For general purpose access, see [`PipelineCommandRef::read_node_subrange`] or
    /// [`PipelineCommandRef::write_node_subrange`].
    pub fn access_node_subrange<N>(
        mut self,
        node: N,
        access: AccessType,
        subresource: impl Into<N::Subresource>,
    ) -> Self
    where
        N: View,
    {
        self.access_node_subrange_mut(node, access, subresource);

        self
    }

    /// Informs the pass that the next recorded command buffer will read or write the `subresource`
    /// of `node` using `access`.
    ///
    /// This function must be called for `node` before it is read or written within a `record`
    /// function. For general purpose access, see [`PipelineCommandRef::read_node_subrange_mut`] or
    /// [`PipelineCommandRef::write_node_subrange_mut`].
    pub fn access_node_subrange_mut<N>(
        &mut self,
        node: N,
        access: AccessType,
        subresource: impl Into<N::Subresource>,
    ) where
        N: View,
    {
        self.cmd
            .push_node_access(node, access, subresource.into().into());
    }

    /// Binds a Vulkan acceleration structure, buffer, or image to the graph associated with this
    /// pass.
    ///
    /// Bound nodes may be used in passes for pipeline and shader operations.
    pub fn bind_node<'b, B>(&'b mut self, binding: B) -> <B as Edge<Graph>>::Result
    where
        B: Edge<Graph>,
        B: Bind<&'b mut Graph, <B as Edge<Graph>>::Result>,
    {
        self.cmd.graph.bind_node(binding)
    }

    /// Finalizes a command and returns the render graph so that additional commands may be added.
    pub fn end_cmd(self) -> &'a mut Graph {
        self.cmd.end_cmd()
    }

    /// Returns Info used to crate a node.
    pub fn node_info<N>(&self, node: N) -> <N as Info>::Info
    where
        N: Info,
    {
        node.info(&self.cmd.graph.bindings)
    }

    fn push_node_view_bind(
        &mut self,
        node: impl Node,
        view_info: impl Into<ViewType>,
        binding: Descriptor,
    ) {
        let node_idx = node.index();
        self.cmd.assert_bound_graph_node(node);

        assert!(
            self.cmd
                .as_mut()
                .execs
                .last_mut()
                .unwrap()
                .bindings
                .insert(binding, (node_idx, Some(view_info.into())))
                .is_none(),
            "descriptor {binding:?} has already been bound"
        );
    }

    /// Informs the pass that the next recorded command buffer will read the given `node` at the
    /// specified shader descriptor.
    ///
    /// The [`AccessType`] is inferred by the currently bound pipeline. See [`Access`] for details.
    ///
    /// This function must be called for `node` before it is read within a `record` function. For
    /// more specific access, see [`PipelineCommandRef::access_descriptor`].
    pub fn read_descriptor<N>(self, descriptor: impl Into<Descriptor>, node: N) -> Self
    where
        N: Info,
        N: View,
        ViewType: From<<N as View>::Info>,
        <N as View>::Info: From<<N as Info>::Info>,
        <N as View>::Subresource: From<<N as View>::Info>,
    {
        let node_info = self.node_info(node);

        // Use the plain node information as the whole view of the node
        let view_info = node_info;

        self.read_descriptor_as(descriptor, node, view_info)
    }

    /// Informs the pass that the next recorded command buffer will read the given `node` at the
    /// specified shader descriptor. The node will be interpreted using `view_info`.
    ///
    /// The [`AccessType`] is inferred by the currently bound pipeline. See [`Access`] for details.
    ///
    /// This function must be called for `node` before it is read within a `record` function. For
    /// more specific access, see [`PipelineCommandRef::access_descriptor_as`].
    pub fn read_descriptor_as<N>(
        self,
        descriptor: impl Into<Descriptor>,
        node: N,
        view_info: impl Into<N::Info>,
    ) -> Self
    where
        N: View,
        <N as View>::Info: Into<ViewType>,
        <N as View>::Subresource: From<<N as View>::Info>,
    {
        let view_info = view_info.into();
        let subresource = <N as View>::Subresource::from(view_info);

        self.read_descriptor_subrange(descriptor, node, view_info, subresource)
    }

    /// Informs the pass that the next recorded command buffer will read the `subresource` of `node`
    /// at the specified shader descriptor. The node will be interpreted using `view_info`.
    ///
    /// The [`AccessType`] is inferred by the currently bound pipeline. See [`Access`] for details.
    ///
    /// This function must be called for `node` before it is read within a `record` function. For
    /// more specific access, see [`PipelineCommandRef::access_descriptor_subrange`].
    pub fn read_descriptor_subrange<N>(
        self,
        descriptor: impl Into<Descriptor>,
        node: N,
        view_info: impl Into<N::Info>,
        subresource: impl Into<N::Subresource>,
    ) -> Self
    where
        N: View,
        <N as View>::Info: Into<ViewType>,
    {
        let access = <T as Access>::DEFAULT_READ;
        self.access_descriptor_subrange(descriptor, node, access, view_info, subresource)
    }

    /// Informs the pass that the next recorded command buffer will read the given `node`.
    ///
    /// The [`AccessType`] is inferred by the currently bound pipeline. See [`Access`] for details.
    ///
    /// This function must be called for `node` before it is read within a `record` function. For
    /// more specific access, see [`PipelineCommandRef::access_node`].
    pub fn read_node(mut self, node: impl Node + Info) -> Self {
        self.read_node_mut(node);

        self
    }

    /// Informs the pass that the next recorded command buffer will read the given `node`.
    ///
    /// The [`AccessType`] is inferred by the currently bound pipeline. See [`Access`] for details.
    ///
    /// This function must be called for `node` before it is read within a `record` function. For
    /// more specific access, see [`PipelineCommandRef::access_node_mut`].
    pub fn read_node_mut(&mut self, node: impl Node + Info) {
        let access = <T as Access>::DEFAULT_READ;
        self.access_node_mut(node, access);
    }

    /// Informs the pass that the next recorded command buffer will read the `subresource` of
    /// `node`.
    ///
    /// The [`AccessType`] is inferred by the currently bound pipeline. See [`Access`] for details.
    ///
    /// This function must be called for `node` before it is read within a `record` function. For
    /// more specific access, see [`PipelineCommandRef::access_node_subrange`].
    pub fn read_node_subrange<N>(mut self, node: N, subresource: impl Into<N::Subresource>) -> Self
    where
        N: View,
    {
        self.read_node_subrange_mut(node, subresource);

        self
    }

    /// Informs the pass that the next recorded command buffer will read the `subresource` of
    /// `node`.
    ///
    /// The [`AccessType`] is inferred by the currently bound pipeline. See [`Access`] for details.
    ///
    /// This function must be called for `node` before it is read within a `record` function. For
    /// more specific access, see [`PipelineCommandRef::access_node_subrange_mut`].
    pub fn read_node_subrange_mut<N>(&mut self, node: N, subresource: impl Into<N::Subresource>)
    where
        N: View,
    {
        let access = <T as Access>::DEFAULT_READ;
        self.access_node_subrange_mut(node, access, subresource);
    }

    /// Informs the pass that the next recorded command buffer will write the given `node` at the
    /// specified shader descriptor.
    ///
    /// The [`AccessType`] is inferred by the currently bound pipeline. See [`Access`] for details.
    ///
    /// This function must be called for `node` before it is written within a `record` function. For
    /// more specific access, see [`PipelineCommandRef::access_descriptor`].
    pub fn write_descriptor<N>(self, descriptor: impl Into<Descriptor>, node: N) -> Self
    where
        N: Info,
        N: View,
        <N as View>::Info: Into<ViewType>,
        <N as View>::Info: From<<N as Info>::Info>,
        <N as View>::Subresource: From<<N as View>::Info>,
    {
        let node_info = self.node_info(node);

        // Use the plain node information as the whole view of the node
        let view_info = node_info;

        self.write_descriptor_as(descriptor, node, view_info)
    }

    /// Informs the pass that the next recorded command buffer will write the given `node` at the
    /// specified shader descriptor. The node will be interpreted using `view_info`.
    ///
    /// The [`AccessType`] is inferred by the currently bound pipeline. See [`Access`] for details.
    ///
    /// This function must be called for `node` before it is written within a `record` function. For
    /// more specific access, see [`PipelineCommandRef::access_descriptor_as`].
    pub fn write_descriptor_as<N>(
        self,
        descriptor: impl Into<Descriptor>,
        node: N,
        view_info: impl Into<N::Info>,
    ) -> Self
    where
        N: View,
        <N as View>::Info: Into<ViewType>,
        <N as View>::Subresource: From<<N as View>::Info>,
    {
        let view_info = view_info.into();
        let subresource = <N as View>::Subresource::from(view_info);

        self.write_descriptor_subrange(descriptor, node, view_info, subresource)
    }

    /// Informs the pass that the next recorded command buffer will write the `subresource` of
    /// `node` at the specified shader descriptor. The node will be interpreted using `view_info`.
    ///
    /// The [`AccessType`] is inferred by the currently bound pipeline. See [`Access`] for details.
    ///
    /// This function must be called for `node` before it is written within a `record` function. For
    /// more specific access, see [`PipelineCommandRef::access_descriptor_subrange`].
    pub fn write_descriptor_subrange<N>(
        self,
        descriptor: impl Into<Descriptor>,
        node: N,
        view_info: impl Into<N::Info>,
        subresource: impl Into<N::Subresource>,
    ) -> Self
    where
        N: View,
        <N as View>::Info: Into<ViewType>,
    {
        let access = <T as Access>::DEFAULT_WRITE;
        self.access_descriptor_subrange(descriptor, node, access, view_info, subresource)
    }

    /// Informs the pass that the next recorded command buffer will write the given `node`.
    ///
    /// The [`AccessType`] is inferred by the currently bound pipeline. See [`Access`] for details.
    ///
    /// This function must be called for `node` before it is written within a `record` function. For
    /// more specific access, see [`PipelineCommandRef::access_node`].
    pub fn write_node(mut self, node: impl Node + Info) -> Self {
        self.write_node_mut(node);

        self
    }

    /// Informs the pass that the next recorded command buffer will write the given `node`.
    ///
    /// The [`AccessType`] is inferred by the currently bound pipeline. See [`Access`] for details.
    ///
    /// This function must be called for `node` before it is written within a `record` function. For
    /// more specific access, see [`PipelineCommandRef::access_node_mut`].
    pub fn write_node_mut(&mut self, node: impl Node + Info) {
        let access = <T as Access>::DEFAULT_WRITE;
        self.access_node_mut(node, access);
    }

    /// Informs the pass that the next recorded command buffer will write the `subresource` of
    /// `node`.
    ///
    /// The [`AccessType`] is inferred by the currently bound pipeline. See [`Access`] for details.
    ///
    /// This function must be called for `node` before it is written within a `record` function. For
    /// more specific access, see [`PipelineCommandRef::access_node_subrange`].
    pub fn write_node_subrange<N>(mut self, node: N, subresource: impl Into<N::Subresource>) -> Self
    where
        N: View,
    {
        self.write_node_subrange_mut(node, subresource);

        self
    }

    /// Informs the pass that the next recorded command buffer will write the `subresource` of
    /// `node`.
    ///
    /// The [`AccessType`] is inferred by the currently bound pipeline. See [`Access`] for details.
    ///
    /// This function must be called for `node` before it is written within a `record` function. For
    /// more specific access, see [`PipelineCommandRef::access_node_subrange_mut`].
    pub fn write_node_subrange_mut<N>(&mut self, node: N, subresource: impl Into<N::Subresource>)
    where
        N: View,
    {
        let access = <T as Access>::DEFAULT_WRITE;
        self.access_node_subrange_mut(node, access, subresource);
    }
}
