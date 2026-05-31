use {
    super::{PipelineCommand, cmd_ref::CommandRef},
    crate::driver::{device::Device, ray_trace::RayTracePipeline},
    ash::vk,
    std::ops::Deref,
};

impl PipelineCommand<'_, RayTracePipeline> {
    /// Begin recording a ray trace pipeline command buffer.
    pub fn record_cmd(
        mut self,
        func: impl FnOnce(RayTraceCommandRef<'_>) + Send + 'static,
    ) -> Self {
        self.record_cmd_mut(func);
        self
    }

    /// Begin recording a ray trace pipeline command buffer.
    pub fn record_cmd_mut(&mut self, func: impl FnOnce(RayTraceCommandRef<'_>) + Send + 'static) {
        let pipeline = self
            .cmd
            .cmd()
            .expect_last_pipeline()
            .expect_ray_trace()
            .clone();

        #[cfg(debug_assertions)]
        let dynamic_stack_size = pipeline.inner.info.dynamic_stack_size;

        self.cmd.push_exec(move |cmd| {
            func(RayTraceCommandRef {
                cmd,

                #[cfg(debug_assertions)]
                dynamic_stack_size,

                pipeline,
            });
        });
    }
}

/// Recording interface for ray tracing commands.
///
/// This structure provides a strongly-typed set of methods which allow ray trace shader code to be
/// executed. An instance is provided to the closure argument of
/// [`PipelineCommand::record_cmd`] which may be accessed by binding a [`RayTracePipeline`] to
/// a command.
///
/// # Examples
///
/// Basic usage:
///
/// ```no_run
/// # use ash::vk;
/// # use vk_graph::driver::DriverError;
/// # use vk_graph::driver::device::{Device, DeviceInfo};
/// # use vk_graph::driver::ray_trace::{
/// #     RayTracePipeline,
/// #     RayTracePipelineInfo,
/// #     RayTraceShaderGroup,
/// # };
/// # use vk_graph::driver::shader::Shader;
/// # use vk_graph::Graph;
/// # fn main() -> Result<(), DriverError> {
/// # let device = Device::create(DeviceInfo::default())?;
/// # let info = RayTracePipelineInfo::default();
/// # let my_miss_code = [0u8; 1];
/// # let my_ray_trace_pipeline = RayTracePipeline::create(&device, info,
/// #     [Shader::new_miss(my_miss_code.as_slice())],
/// #     [RayTraceShaderGroup::new_general(0)],
/// # )?;
/// # let mut my_graph = Graph::default();
/// my_graph.begin_cmd()
///         .debug_name("my ray trace command")
///         .bind_pipeline(&my_ray_trace_pipeline)
///         .record_cmd(move |cmd| {
///             // During this closure we have access to the ray trace functions!
///         });
/// # Ok(()) }
/// ```
pub struct RayTraceCommandRef<'a> {
    cmd: CommandRef<'a>,

    #[cfg(debug_assertions)]
    dynamic_stack_size: bool,

    pipeline: RayTracePipeline,
}

impl RayTraceCommandRef<'_> {
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
    /// # vk_shader_macros::glsl!(target: vulkan1_2, r#"
    /// #version 460
    /// #pragma shader_stage(closest)
    ///
    /// layout(push_constant) uniform PushConstants {
    ///     layout(offset = 0) uint some_val;
    /// } push_constants;
    ///
    /// void main() {
    ///     // TODO: Add bindings to write things!
    /// }
    /// # "#);
    /// ```
    ///
    /// ```no_run
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::ray_trace::{
    /// #     RayTracePipeline,
    /// #     RayTracePipelineInfo,
    /// #     RayTraceShaderGroup,
    /// # };
    /// # use vk_graph::driver::shader::Shader;
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// # let shader = [0u8; 1];
    /// # let info = RayTracePipelineInfo::default();
    /// # let my_miss_code = [0u8; 1];
    /// # let my_ray_trace_pipeline = RayTracePipeline::create(&device, info,
    /// #     [Shader::new_miss(my_miss_code.as_slice())],
    /// #     [RayTraceShaderGroup::new_general(0)],
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
    ///         .bind_pipeline(&my_ray_trace_pipeline)
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

    /// Set the stack size dynamically for a ray trace pipeline.
    ///
    /// See
    /// `RayTracePipelineInfo::dynamic_stack_size` and see the Vulkan spec.
    #[profiling::function]
    pub fn set_stack_size(&self, pipeline_stack_size: u32) -> &Self {
        #[cfg(debug_assertions)]
        assert!(self.dynamic_stack_size);

        let ray_trace_ext = Device::expect_ray_trace_ext(&self.cmd.device);

        unsafe {
            ray_trace_ext
                .cmd_set_ray_tracing_pipeline_stack_size(self.cmd.handle, pipeline_stack_size);
        }

        self
    }

    // TODO: If the rayTraversalPrimitiveCulling or rayQuery features are enabled, the
    // SkipTrianglesKHR and SkipAABBsKHR ray flags can be specified when tracing a ray.
    // SkipTrianglesKHR and SkipAABBsKHR are mutually exclusive.

    /// Ray traces using the currently-bound [`RayTracePipeline`] and the given shader binding
    /// tables.
    ///
    /// Shader binding tables must be constructed according to this [example].
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
    /// # use vk_graph::driver::ray_trace::{
    /// #     RayTracePipeline,
    /// #     RayTracePipelineInfo,
    /// #     RayTraceShaderGroup,
    /// # };
    /// # use vk_graph::driver::shader::Shader;
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// # let shader = [0u8; 1];
    /// # let info = RayTracePipelineInfo::default();
    /// # let my_miss_code = [0u8; 1];
    /// # let my_ray_trace_pipeline = RayTracePipeline::create(&device, info,
    /// #     [Shader::new_miss(my_miss_code.as_slice())],
    /// #     [RayTraceShaderGroup::new_general(0)],
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
    ///         .bind_pipeline(&my_ray_trace_pipeline)
    ///         .record_cmd(move |cmd| {
    ///             cmd.trace_rays(&rgen_sbt, &hit_sbt, &miss_sbt, &call_sbt, 320, 200, 1);
    ///         });
    /// # Ok(()) }
    /// ```
    ///
    /// [example]: https://github.com/attackgoat/vk-graph/blob/master/examples/ray_trace.rs
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
        let ray_trace_ext = Device::expect_ray_trace_ext(&self.cmd.device);

        unsafe {
            ray_trace_ext.cmd_trace_rays(
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

    /// Ray traces using the currently-bound [`RayTracePipeline`] and the given shader binding
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
        let ray_trace_ext = Device::expect_ray_trace_ext(&self.cmd.device);

        unsafe {
            ray_trace_ext.cmd_trace_rays_indirect(
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

impl<'a> Deref for RayTraceCommandRef<'a> {
    type Target = CommandRef<'a>;

    fn deref(&self) -> &Self::Target {
        &self.cmd
    }
}
