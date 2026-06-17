use {
    super::{PipelineCommand, cmd_ref::CommandRef},
    crate::driver::{device::Device, ray_tracing::RayTracingPipeline},
    ash::vk,
    std::ops::Deref,
};

impl PipelineCommand<'_, RayTracingPipeline> {
    /// Begin recording ray tracing pipeline work for this graph command.
    pub fn record_cmd(
        mut self,
        func: impl FnOnce(RayTracingCommandRef<'_>) + Send + 'static,
    ) -> Self {
        self.record_cmd_mut(func);
        self
    }

    /// Mutable-borrow form of [`Self::record_cmd`].
    pub fn record_cmd_mut(&mut self, func: impl FnOnce(RayTracingCommandRef<'_>) + Send + 'static) {
        let pipeline = self
            .cmd
            .cmd()
            .expect_last_pipeline()
            .expect_ray_tracing()
            .clone();

        self.cmd.push_exec(move |cmd| {
            func(RayTracingCommandRef { cmd, pipeline });
        });
    }

    pub(crate) fn record_stream_mut(
        &mut self,
        func: impl for<'r> Fn(RayTracingCommandRef<'r>) + Send + Sync + 'static,
    ) {
        let pipeline = self
            .cmd
            .cmd()
            .expect_last_pipeline()
            .expect_ray_tracing()
            .clone();

        self.cmd.push_reusable_exec(move |cmd| {
            func(RayTracingCommandRef {
                cmd,
                pipeline: pipeline.clone(),
            });
        });
    }
}

/// Recording interface for ray tracing commands.
///
/// This structure provides a strongly-typed set of methods which allow ray tracing shader code to
/// be executed. An instance is provided to the closure argument of [`PipelineCommand::record_cmd`]
/// which may be accessed by binding a [`RayTracingPipeline`] to a command.
///
/// # Examples
///
/// Basic usage:
///
/// ```no_run
/// # use ash::vk;
/// # use vk_graph::driver::DriverError;
/// # use vk_graph::driver::device::{Device, DeviceInfo};
/// # use vk_graph::driver::ray_tracing::{
/// #     RayTracingPipeline,
/// #     RayTracingPipelineInfo,
/// #     RayTracingShaderGroup,
/// # };
/// # use vk_graph::driver::shader::Shader;
/// # use vk_graph::Graph;
/// # fn main() -> Result<(), DriverError> {
/// # let device = Device::create(DeviceInfo::default())?;
/// # let info = RayTracingPipelineInfo::default();
/// # let my_miss_code = [0u8; 1];
/// # let my_ray_tracing_pipeline = RayTracingPipeline::create(&device, info,
/// #     [Shader::new_miss(my_miss_code.as_slice())],
/// #     [RayTracingShaderGroup::new_general(0)],
/// # )?;
/// # let mut my_graph = Graph::default();
/// my_graph.begin_cmd()
///         .debug_name("my ray tracing command")
///         .bind_pipeline(&my_ray_tracing_pipeline)
///         .record_cmd(move |cmd| {
///             // During this closure we have access to the ray tracing functions!
///         });
/// # Ok(()) }
/// ```
pub struct RayTracingCommandRef<'a> {
    cmd: CommandRef<'a>,
    pipeline: RayTracingPipeline,
}

impl RayTracingCommandRef<'_> {
    /// Updates push constants.
    ///
    /// Push constants represent a high-speed path to modify constant data in pipelines that is
    /// expected to outperform memory-backed resource updates.
    ///
    /// Push constant values can be updated incrementally, causing shader stages to read the new
    /// data for push constants modified by this command, while still reading the previous data for
    /// push constants not modified by this command.
    ///
    /// # Device limitations
    ///
    /// See [`VkPhysicalDeviceLimits::maxPushConstantsSize`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceLimits.html)
    /// for the limit of the current device. You may also check [gpuinfo.org] for a listing of
    /// reported limits on other devices.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # vk_shader_macros::glsl!(target: vulkan1_2, r#"
    /// #version 460
    /// #pragma shader_stage(closest)
    ///
    /// layout(push_constant) uniform PushConstants {
    ///     layout(offset = 0) uint some_val;
    /// } push_constants;
    ///
    /// void main() {
    ///     uint value = push_constants.some_val;
    /// }
    /// # "#);
    /// ```
    ///
    /// ```no_run
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::ray_tracing::{
    /// #     RayTracingPipeline,
    /// #     RayTracingPipelineInfo,
    /// #     RayTracingShaderGroup,
    /// # };
    /// # use vk_graph::driver::shader::Shader;
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// # let shader = [0u8; 1];
    /// # let info = RayTracingPipelineInfo::default();
    /// # let my_miss_code = [0u8; 1];
    /// # let my_ray_tracing_pipeline = RayTracingPipeline::create(&device, info,
    /// #     [Shader::new_miss(my_miss_code.as_slice())],
    /// #     [RayTracingShaderGroup::new_general(0)],
    /// # )?;
    /// # let rgen_sbt = vk::StridedDeviceAddressRegionKHR {
    /// #     device_address: 0,
    /// #     stride: 0,
    /// #     size: 0,
    /// # };
    /// # let hit_sbt = vk::StridedDeviceAddressRegionKHR {
    /// #     device_address: 0,
    /// #     stride: 0,
    /// #     size: 0,
    /// # };
    /// # let miss_sbt = vk::StridedDeviceAddressRegionKHR {
    /// #     device_address: 0,
    /// #     stride: 0,
    /// #     size: 0,
    /// # };
    /// # let call_sbt = vk::StridedDeviceAddressRegionKHR {
    /// #     device_address: 0,
    /// #     stride: 0,
    /// #     size: 0,
    /// # };
    /// # let mut my_graph = Graph::default();
    /// my_graph.begin_cmd()
    ///         .debug_name("draw a cornell box")
    ///         .bind_pipeline(&my_ray_tracing_pipeline)
    ///         .record_cmd(move |cmd| {
    ///             cmd.push_constants(0, &[0xcb])
    ///                    .trace_rays(&rgen_sbt, &hit_sbt, &miss_sbt, &call_sbt, 320, 200, 1);
    ///         });
    /// # Ok(()) }
    /// ```
    ///
    /// See [`vkCmdPushConstants`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdPushConstants.html).
    #[profiling::function]
    pub fn push_constants(&self, offset: u32, data: &[u8]) -> &Self {
        self.cmd_push_constants(
            self.pipeline.inner.layout,
            &self.pipeline.inner.push_constants,
            offset,
            data,
        );

        self
    }

    /// Sets the stack size dynamically for a ray tracing pipeline.
    ///
    /// The pipeline must have been created with
    /// [`RayTracingPipelineInfo::dynamic_stack_size`](crate::driver::ray_tracing::RayTracingPipelineInfo::dynamic_stack_size)
    /// enabled.
    ///
    /// See [`vkCmdSetRayTracingPipelineStackSizeKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetRayTracingPipelineStackSizeKHR.html).
    #[profiling::function]
    pub fn set_stack_size(&self, pipeline_stack_size: u32) -> &Self {
        let khr_ray_tracing_pipeline = Device::expect_vk_khr_ray_tracing_pipeline(&self.cmd.device);

        #[cfg(feature = "checked")]
        assert!(
            self.pipeline.inner.info.dynamic_stack_size,
            "ray tracing pipeline was not created with dynamic_stack_size enabled"
        );

        unsafe {
            /*
            Checked mode catches missing dynamic_stack_size enablement early. Other Vulkan
            validation remains the responsibility of the validation layer.
            */
            khr_ray_tracing_pipeline
                .cmd_set_ray_tracing_pipeline_stack_size(self.cmd.handle, pipeline_stack_size);
        }

        self
    }

    /*
    TODO: If the rayTraversalPrimitiveCulling or rayQuery features are enabled, the SkipTrianglesKHR
    and SkipAABBsKHR ray flags can be specified when tracing a ray. SkipTrianglesKHR and
    SkipAABBsKHR are mutually exclusive.
    */

    /// Ray traces using the currently-bound [`RayTracingPipeline`] and the given shader binding
    /// tables.
    ///
    /// Shader binding tables must be constructed according to this [example].
    ///
    /// See [`vkCmdTraceRaysKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysKHR.html).
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::ray_tracing::{
    /// #     RayTracingPipeline,
    /// #     RayTracingPipelineInfo,
    /// #     RayTracingShaderGroup,
    /// # };
    /// # use vk_graph::driver::shader::Shader;
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// # let shader = [0u8; 1];
    /// # let info = RayTracingPipelineInfo::default();
    /// # let my_miss_code = [0u8; 1];
    /// # let my_ray_tracing_pipeline = RayTracingPipeline::create(&device, info,
    /// #     [Shader::new_miss(my_miss_code.as_slice())],
    /// #     [RayTracingShaderGroup::new_general(0)],
    /// # )?;
    /// # let rgen_sbt = vk::StridedDeviceAddressRegionKHR {
    /// #     device_address: 0,
    /// #     stride: 0,
    /// #     size: 0,
    /// # };
    /// # let hit_sbt = vk::StridedDeviceAddressRegionKHR {
    /// #     device_address: 0,
    /// #     stride: 0,
    /// #     size: 0,
    /// # };
    /// # let miss_sbt = vk::StridedDeviceAddressRegionKHR {
    /// #     device_address: 0,
    /// #     stride: 0,
    /// #     size: 0,
    /// # };
    /// # let call_sbt = vk::StridedDeviceAddressRegionKHR {
    /// #     device_address: 0,
    /// #     stride: 0,
    /// #     size: 0,
    /// # };
    /// # let mut my_graph = Graph::default();
    /// my_graph.begin_cmd()
    ///         .debug_name("draw a cornell box")
    ///         .bind_pipeline(&my_ray_tracing_pipeline)
    ///         .record_cmd(move |cmd| {
    ///             cmd.trace_rays(&rgen_sbt, &hit_sbt, &miss_sbt, &call_sbt, 320, 200, 1);
    ///         });
    /// # Ok(()) }
    /// ```
    ///
    /// [example]: https://github.com/attackgoat/vk-graph/blob/master/examples/ray_tracing.rs
    #[allow(clippy::too_many_arguments)]
    #[profiling::function]
    pub fn trace_rays(
        &self,
        raygen_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        miss_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        hit_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        callable_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        width: u32,
        height: u32,
        depth: u32,
    ) -> &Self {
        let khr_ray_tracing_pipeline = Device::expect_vk_khr_ray_tracing_pipeline(&self.cmd.device);

        unsafe {
            khr_ray_tracing_pipeline.cmd_trace_rays(
                self.cmd.handle,
                raygen_shader_binding_table,
                miss_shader_binding_table,
                hit_shader_binding_table,
                callable_shader_binding_table,
                width,
                height,
                depth,
            );
        }

        self
    }

    /// Ray traces using the currently-bound [`RayTracingPipeline`] and the given shader binding
    /// tables.
    ///
    /// `indirect_device_address` is a [buffer device address] which is a pointer to a
    /// [`vk::TraceRaysIndirectCommandKHR`] structure containing the trace ray parameters.
    ///
    /// See [`vkCmdTraceRaysIndirectKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysIndirectKHR.html).
    ///
    /// [buffer device address]: crate::driver::buffer::Buffer::device_address
    #[profiling::function]
    pub fn trace_rays_indirect(
        &self,
        raygen_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        miss_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        hit_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        callable_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        indirect_device_address: vk::DeviceAddress,
    ) -> &Self {
        let khr_ray_tracing_pipeline = Device::expect_vk_khr_ray_tracing_pipeline(&self.cmd.device);

        unsafe {
            khr_ray_tracing_pipeline.cmd_trace_rays_indirect(
                self.cmd.handle,
                raygen_shader_binding_table,
                miss_shader_binding_table,
                hit_shader_binding_table,
                callable_shader_binding_table,
                indirect_device_address,
            )
        }

        self
    }
}

impl<'a> Deref for RayTracingCommandRef<'a> {
    type Target = CommandRef<'a>;

    fn deref(&self) -> &Self::Target {
        &self.cmd
    }
}
