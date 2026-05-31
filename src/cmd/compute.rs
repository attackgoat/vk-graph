use {
    super::{cmd_ref::CommandRef, pipeline::PipelineCommand},
    crate::{driver::compute::ComputePipeline, node::AnyBufferNode},
    ash::vk,
    std::ops::Deref,
};

impl PipelineCommand<'_, ComputePipeline> {
    /// Begin recording a compute pipeline command buffer.
    pub fn record_cmd(mut self, func: impl FnOnce(ComputeCommandRef<'_>) + Send + 'static) -> Self {
        self.record_cmd_mut(func);
        self
    }

    /// Begin recording a compute pipeline command buffer.
    pub fn record_cmd_mut(&mut self, func: impl FnOnce(ComputeCommandRef<'_>) + Send + 'static) {
        let pipeline = self
            .cmd
            .cmd()
            .expect_last_pipeline()
            .expect_compute()
            .clone();

        self.cmd.push_exec(move |cmd| {
            func(ComputeCommandRef { cmd, pipeline });
        });
    }
}

/// Recording interface for computing commands.
///
/// This structure provides a strongly-typed set of methods which allow compute shader code to be
/// executed. An instance is provided to the closure argument of
/// [`PipelineCommand::record_cmd`] which may be accessed by binding a [`ComputePipeline`] to a
/// command.
///
/// # Examples
///
/// Basic usage:
///
/// ```no_run
/// # use ash::vk;
/// # use vk_graph::driver::DriverError;
/// # use vk_graph::driver::device::{Device, DeviceInfo};
/// # use vk_graph::driver::compute::{ComputePipeline, ComputePipelineInfo};
/// # use vk_graph::driver::shader::{Shader};
/// # use vk_graph::Graph;
/// # fn main() -> Result<(), DriverError> {
/// # let device = Device::new(DeviceInfo::default())?;
/// # let info = ComputePipelineInfo::default();
/// # let shader = Shader::new_compute([0u8; 1].as_slice());
/// # let my_compute_pipeline = ComputePipeline::create(&device, info, shader)?;
/// # let mut my_graph = Graph::default();
/// my_graph
///     .begin_cmd()
///     .bind_pipeline(&my_compute_pipeline)
///     .record_cmd(move |cmd| {
///         // During this closure we have access to the compute functions!
///         cmd.dispatch(64, 1, 1);
///     });
/// # Ok(()) }
/// ```
pub struct ComputeCommandRef<'a> {
    cmd: CommandRef<'a>,
    pipeline: ComputePipeline,
}

impl ComputeCommandRef<'_> {
    /// [Dispatch] compute work items.
    ///
    /// When the command is executed, a global workgroup consisting of
    /// `group_count_x × group_count_y × group_count_z` local workgroups is assembled.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # vk_shader_macros::glsl!(r#"
    /// #version 450
    /// #pragma shader_stage(compute)
    ///
    /// layout(set = 0, binding = 0, std430) restrict writeonly buffer MyBufer {
    ///     uint my_buf[];
    /// };
    ///
    /// void main() {
    ///     // TODO
    /// }
    /// # "#);
    /// ```
    ///
    /// ```no_run
    /// # use ash::vk;
    /// # use vk_graph::driver::{AccessType, DriverError};
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::compute::{ComputePipeline, ComputePipelineInfo};
    /// # use vk_graph::driver::shader::{Shader};
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::new(DeviceInfo::default())?;
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::STORAGE_BUFFER);
    /// # let my_buf = Buffer::create(&device, buf_info)?;
    /// # let info = ComputePipelineInfo::default();
    /// # let shader = Shader::new_compute([0u8; 1].as_slice());
    /// # let my_compute_pipeline = ComputePipeline::create(&device, info, shader)?;
    /// # let mut my_graph = Graph::default();
    /// # let my_buf_node = my_graph.bind_resource(my_buf);
    /// my_graph
    ///     .begin_cmd()
    ///     .debug_name("fill my_buf_node with data")
    ///     .bind_pipeline(&my_compute_pipeline)
    ///     .shader_resource_access(0, my_buf_node, AccessType::ComputeShaderWrite)
    ///     .record_cmd(move |cmd| {
    ///         cmd.dispatch(128, 64, 32);
    ///     });
    /// # Ok(()) }
    /// ```
    ///
    /// See [`vkCmdDispatch`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDispatch.html).
    #[profiling::function]
    pub fn dispatch(&self, group_count_x: u32, group_count_y: u32, group_count_z: u32) -> &Self {
        unsafe {
            self.cmd.device.cmd_dispatch(
                self.cmd.handle,
                group_count_x,
                group_count_y,
                group_count_z,
            );
        }

        self
    }

    /// [Dispatch] compute work items with non-zero base values for the workgroup IDs.
    ///
    /// When the command is executed, a global workgroup consisting of
    /// `group_count_x × group_count_y × group_count_z` local workgroups is assembled, with
    /// WorkgroupId values ranging from `[base_group*, base_group* + group_count*)` in each
    /// component.
    ///
    /// [`Self::dispatch`] is equivalent to
    /// `dispatch_base(0, 0, 0, group_count_x, group_count_y, group_count_z)`.
    ///
    /// See [`vkCmdDispatchBase`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDispatchBase.html).
    #[profiling::function]
    pub fn dispatch_base(
        &self,
        base_group_x: u32,
        base_group_y: u32,
        base_group_z: u32,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) -> &Self {
        unsafe {
            self.cmd.device.cmd_dispatch_base(
                self.cmd.handle,
                base_group_x,
                base_group_y,
                base_group_z,
                group_count_x,
                group_count_y,
                group_count_z,
            );
        }

        self
    }

    /// Dispatch compute work items with indirect parameters.
    ///
    /// `dispatch_indirect` behaves similarly to [`Self::dispatch`] except that the parameters
    /// are read by the device from `args_buf` during execution. The parameters of the dispatch are
    /// encoded in a [`vk::DispatchIndirectCommand`] structure taken from `args_buf` starting at
    /// `args_offset`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use ash::vk;
    /// # use bytemuck::{bytes_of, Pod, Zeroable};
    /// # use vk_graph::driver::{AccessType, DriverError};
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::compute::{ComputePipeline, ComputePipelineInfo};
    /// # use vk_graph::driver::shader::{Shader};
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::new(DeviceInfo::default())?;
    /// # let buf_info = BufferInfo::device_mem(8, vk::BufferUsageFlags::STORAGE_BUFFER);
    /// # let my_buf = Buffer::create(&device, buf_info)?;
    /// # let info = ComputePipelineInfo::default();
    /// # let shader = Shader::new_compute([0u8; 1].as_slice());
    /// # let my_compute_pipeline = ComputePipeline::create(&device, info, shader)?;
    /// # let mut my_graph = Graph::default();
    /// # let my_buf_node = my_graph.bind_resource(my_buf);
    /// # #[repr(C)]
    /// # #[derive(Clone, Copy, Pod, Zeroable)]
    /// # struct DispatchIndirectCommand { x: u32, y: u32, z: u32, }
    /// let args = DispatchIndirectCommand {
    ///     x: 1,
    ///     y: 2,
    ///     z: 3,
    /// };
    /// let data = bytes_of(&args);
    /// let usage = vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER;
    /// let args_buf = Buffer::create_from_slice(&device, usage, data)?;
    /// let args_buf = my_graph.bind_resource(args_buf);
    ///
    /// my_graph
    ///     .begin_cmd()
    ///     .debug_name("fill my_buf_node with data")
    ///     .bind_pipeline(&my_compute_pipeline)
    ///     .resource_access(args_buf, AccessType::IndirectBuffer)
    ///     .shader_resource_access(0, my_buf_node, AccessType::ComputeShaderWrite)
    ///     .record_cmd(move |cmd| {
    ///         cmd.dispatch_indirect(args_buf, 0);
    ///     });
    /// # Ok(()) }
    /// ```
    ///
    /// See [`vkCmdDispatchIndirect`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDispatchIndirect.html).
    #[profiling::function]
    pub fn dispatch_indirect(
        &self,
        args_buf: impl Into<AnyBufferNode>,
        args_offset: vk::DeviceSize,
    ) -> &Self {
        let args_buf = args_buf.into();
        let args_buf = self.resource(args_buf);

        unsafe {
            self.cmd
                .device
                .cmd_dispatch_indirect(self.cmd.handle, args_buf.handle, args_offset);
        }

        self
    }

    /// Updates push constants.
    ///
    /// Push constants represent a high speed path to modify constant data in pipelines that is
    /// expected to outperform memory-backed resource updates.
    ///
    /// Push constant values can be updated incrementally, causing shader stages to read the new
    /// data for push constants modified by this command, while still reading the previous data for
    /// push constants not modified by this command.
    ///
    /// # Device limitations
    ///
    /// See
    /// [`device.physical_device.props.limits.max_push_constants_size`](vk::PhysicalDeviceLimits)
    /// for the limits of the current device. You may also check [gpuinfo.org] for a listing of
    /// reported limits on other devices.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # vk_shader_macros::glsl!(r#"
    /// #version 450
    /// #pragma shader_stage(compute)
    ///
    /// layout(push_constant) uniform PushConstants {
    ///     layout(offset = 0) uint the_answer;
    /// } push_constants;
    ///
    /// void main()
    /// {
    ///     // TODO: Add bindings to read/write things!
    /// }
    /// # "#);
    /// ```
    ///
    /// ```no_run
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::compute::{ComputePipeline, ComputePipelineInfo};
    /// # use vk_graph::driver::shader::{Shader};
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::new(DeviceInfo::default())?;
    /// # let info = ComputePipelineInfo::default();
    /// # let shader = Shader::new_compute([0u8; 1].as_slice());
    /// # let my_compute_pipeline = ComputePipeline::create(&device, info, shader)?;
    /// # let mut my_graph = Graph::default();
    /// my_graph
    ///     .begin_cmd()
    ///     .debug_name("compute the ultimate question")
    ///     .bind_pipeline(&my_compute_pipeline)
    ///     .record_cmd(move |cmd| {
    ///         cmd
    ///             .push_constants(0, &[42])
    ///             .dispatch(1, 1, 1);
    ///     });
    /// # Ok(()) }
    /// ```
    ///
    /// See [`vkCmdPushConstants`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdPushConstants.html).
    #[profiling::function]
    pub fn push_constants(&self, offset: u32, data: &[u8]) -> &Self {
        self.cmd_push_constants(
            self.pipeline.inner.layout,
            self.pipeline.inner.push_constants.as_slice(),
            offset,
            data,
        );

        self
    }
}

impl<'a> Deref for ComputeCommandRef<'a> {
    type Target = CommandRef<'a>;

    fn deref(&self) -> &Self::Target {
        &self.cmd
    }
}

#[allow(unused)]
mod deprecated {
    use {
        crate::{
            Node,
            cmd::{
                Binding, PipelineCommand, Subresource, SubresourceRange, ViewInfo,
                compute::ComputeCommandRef,
            },
            driver::compute::ComputePipeline,
        },
        std::any::Any,
        vk_sync::AccessType,
    };

    impl ComputeCommandRef<'_> {
        #[deprecated = "use push_constants function"]
        #[doc(hidden)]
        pub fn push_constants_offset(&self, offset: u32, data: &[u8]) -> &Self {
            self.push_constants(offset, data)
        }
    }

    impl PipelineCommand<'_, ComputePipeline> {
        #[deprecated = "use shader_resource_access with ComputeShaderReadOther"]
        #[doc(hidden)]
        pub fn read_descriptor<N>(self, descriptor: impl Into<Binding>, node: N) -> Self
        where
            N: Node + Subresource,
            N::Info: Copy,
            SubresourceRange: From<N::Info>,
            ViewInfo: From<N::Info>,
        {
            self.shader_resource_access(descriptor, node, AccessType::ComputeShaderReadOther)
        }

        #[deprecated = "use shader_subresource_access with ComputeShaderReadOther"]
        #[doc(hidden)]
        pub fn read_descriptor_as<N>(
            self,
            descriptor: impl Into<Binding>,
            node: N,
            node_view: impl Into<N::Info>,
        ) -> Self
        where
            N: Node + Subresource,
            N::Info: Copy,
            SubresourceRange: From<N::Info>,
            ViewInfo: From<N::Info>,
        {
            self.shader_subresource_access(
                descriptor,
                node,
                node_view,
                AccessType::ComputeShaderReadOther,
            )
        }

        #[deprecated = "use record_cmd function"]
        #[doc(hidden)]
        pub fn record_compute(
            self,
            func: impl FnOnce(ComputeCommandRef<'_>, ()) + Send + 'static,
        ) -> Self {
            self.record_cmd(|cmd| func(cmd, ()))
        }

        #[deprecated = "use shader_resource_access function with AccessType::ComputeShaderWrite"]
        #[doc(hidden)]
        pub fn write_descriptor<N>(self, descriptor: impl Into<Binding>, node: N) -> Self
        where
            N: Node + Subresource,
            N::Info: Copy,
            SubresourceRange: From<N::Info>,
            ViewInfo: From<N::Info>,
        {
            self.shader_resource_access(descriptor, node, AccessType::ComputeShaderWrite)
        }

        #[deprecated = "use shader_subresource_access function with AccessType::ComputeShaderWrite"]
        #[doc(hidden)]
        pub fn write_descriptor_as<N>(
            self,
            descriptor: impl Into<Binding>,
            node: N,
            node_view: impl Into<N::Info>,
        ) -> Self
        where
            N: Node + Subresource,
            N::Info: Copy,
            SubresourceRange: From<N::Info>,
            ViewInfo: From<N::Info>,
        {
            self.shader_subresource_access(
                descriptor,
                node,
                node_view,
                AccessType::ComputeShaderWrite,
            )
        }
    }
}
