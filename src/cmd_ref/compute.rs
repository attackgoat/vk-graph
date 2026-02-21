use {
    super::{Bindings, pipeline::PipelineRef},
    crate::{
        AnyBufferNode,
        driver::{compute::ComputePipeline, device::Device},
    },
    ash::vk,
    log::trace,
};

/// Recording interface for computing commands.
///
/// This structure provides a strongly-typed set of methods which allow compute shader code to be
/// executed. An instance of `Compute` is provided to the closure parameter of
/// [`PipelineCommandRef::record_pipeline`] which may be accessed by binding a [`ComputePipeline`] to a
/// render pass.
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
/// my_graph.begin_cmd().with_name("my compute pass")
///         .bind_pipeline(&my_compute_pipeline)
///         .record_pipeline(move |compute, bindings| {
///             // During this closure we have access to the compute methods!
///         });
/// # Ok(()) }
/// ```
pub struct ComputePipelineRef<'a> {
    pub(super) bindings: Bindings<'a>,
    pub(super) cmd_buf: vk::CommandBuffer,
    pub(super) device: &'a Device,
    pub(super) pipeline: ComputePipeline,
}

impl ComputePipelineRef<'_> {
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
    /// # use vk_graph::driver::DriverError;
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
    /// # let my_buf_node = my_graph.bind_node(my_buf);
    /// my_graph.begin_cmd().with_name("fill my_buf_node with data")
    ///         .bind_pipeline(&my_compute_pipeline)
    ///         .write_descriptor(0, my_buf_node)
    ///         .record_pipeline(move |compute, bindings| {
    ///             compute.dispatch(128, 64, 32);
    ///         });
    /// # Ok(()) }
    /// ```
    ///
    /// [Dispatch]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdDispatch.html
    #[profiling::function]
    pub fn dispatch(&self, group_count_x: u32, group_count_y: u32, group_count_z: u32) -> &Self {
        unsafe {
            self.device
                .cmd_dispatch(self.cmd_buf, group_count_x, group_count_y, group_count_z);
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
    /// [`Compute::dispatch`] is equivalent to
    /// `dispatch_base(0, 0, 0, group_count_x, group_count_y, group_count_z)`.
    ///
    /// [Dispatch]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdDispatchBase.html
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
            self.device.cmd_dispatch_base(
                self.cmd_buf,
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
    /// `dispatch_indirect` behaves similarly to [`Compute::dispatch`] except that the parameters
    /// are read by the device from `args_buf` during execution. The parameters of the dispatch are
    /// encoded in a [`vk::DispatchIndirectCommand`] structure taken from `args_buf` starting at
    /// `args_offset`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use std::mem::size_of;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
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
    /// # let my_buf_node = my_graph.bind_node(my_buf);
    /// const CMD_SIZE: usize = size_of::<vk::DispatchIndirectCommand>();
    ///
    /// let cmd = vk::DispatchIndirectCommand {
    ///     x: 1,
    ///     y: 2,
    ///     z: 3,
    /// };
    /// let cmd_data = unsafe {
    ///     std::slice::from_raw_parts(&cmd as *const _ as *const _, CMD_SIZE)
    /// };
    ///
    /// let args_buf_flags = vk::BufferUsageFlags::STORAGE_BUFFER;
    /// let args_buf = Buffer::create_from_slice(&device, args_buf_flags, cmd_data)?;
    /// let args_buf_node = my_graph.bind_node(args_buf);
    ///
    /// my_graph.begin_cmd().with_name("fill my_buf_node with data")
    ///         .bind_pipeline(&my_compute_pipeline)
    ///         .read_node(args_buf_node)
    ///         .write_descriptor(0, my_buf_node)
    ///         .record_pipeline(move |compute, bindings| {
    ///             compute.dispatch_indirect(args_buf_node, 0);
    ///         });
    /// # Ok(()) }
    /// ```
    ///
    /// [Dispatch]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdDispatchIndirect.html
    /// [VkDispatchIndirectCommand]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDispatchIndirectCommand.html
    #[profiling::function]
    pub fn dispatch_indirect(
        &self,
        args_buf: impl Into<AnyBufferNode>,
        args_offset: vk::DeviceSize,
    ) -> &Self {
        let args_buf = args_buf.into();

        unsafe {
            self.device.cmd_dispatch_indirect(
                self.cmd_buf,
                self.bindings[args_buf].handle,
                args_offset,
            );
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
    /// my_graph.begin_cmd().with_name("compute the ultimate question")
    ///         .bind_pipeline(&my_compute_pipeline)
    ///         .record_pipeline(move |compute, bindings| {
    ///             compute.push_constants(0, &[42])
    ///                    .dispatch(1, 1, 1);
    ///         });
    /// # Ok(()) }
    /// ```
    ///
    /// [gpuinfo.org]: https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxPushConstantsSize&platform=all
    #[profiling::function]
    pub fn push_constants(&self, offset: u32, data: &[u8]) -> &Self {
        if let Some(push_const) = self.pipeline.inner.push_constants {
            // Determine the range of the overall pipline push constants which overlap with `data`
            let push_const_end = push_const.offset + push_const.size;
            let data_end = offset + data.len() as u32;
            let end = data_end.min(push_const_end);
            let start = offset.max(push_const.offset);

            if end > start {
                trace!(
                    "      push constants {:?} {}..{}",
                    push_const.stage_flags, start, end
                );

                unsafe {
                    self.device.cmd_push_constants(
                        self.cmd_buf,
                        self.pipeline.inner.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        push_const.offset,
                        &data[(start - offset) as usize..(end - offset) as usize],
                    );
                }
            }
        }

        self
    }
}

// NOTE: local implementation of type from super module
impl PipelineRef<'_, ComputePipeline> {
    /// Begin recording a compute pipeline command buffer.
    pub fn record_pipeline(
        mut self,
        func: impl FnOnce(ComputePipelineRef<'_>, Bindings<'_>) + Send + 'static,
    ) -> Self {
        let pipeline = self
            .cmd
            .as_ref()
            .execs
            .last()
            .unwrap()
            .pipeline
            .as_ref()
            .unwrap()
            .unwrap_compute()
            .clone();

        self.cmd.push_execute(move |device, cmd_buf, bindings| {
            func(
                ComputePipelineRef {
                    bindings,
                    cmd_buf,
                    device,
                    pipeline,
                },
                bindings,
            );
        });

        self
    }
}
