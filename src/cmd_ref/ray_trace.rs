use {
    super::{Bindings, PipelineCommandRef},
    crate::driver::{device::Device, ray_trace::RayTracePipeline},
    ash::vk,
    log::trace,
    std::sync::Arc,
};

// NOTE: local implementation of type from super module
impl PipelineCommandRef<'_, RayTracePipeline> {
    /// Begin recording a ray tracing command buffer.
    pub fn record_pipeline(
        mut self,
        func: impl FnOnce(RayTrace<'_>, Bindings<'_>) + Send + 'static,
    ) -> Self {
        let pipeline = Arc::clone(
            self.cmd
                .as_ref()
                .execs
                .last()
                .unwrap()
                .pipeline
                .as_ref()
                .unwrap()
                .unwrap_ray_trace(),
        );

        #[cfg(debug_assertions)]
        let dynamic_stack_size = pipeline.info.dynamic_stack_size;

        self.cmd.push_execute(move |device, cmd_buf, bindings| {
            func(
                RayTrace {
                    cmd_buf,
                    device,

                    #[cfg(debug_assertions)]
                    dynamic_stack_size,

                    pipeline,
                },
                bindings,
            );
        });

        self
    }
}

/// Recording interface for ray tracing commands.
///
/// This structure provides a strongly-typed set of methods which allow ray trace shader code to be
/// executed. An instance of `RayTrace` is provided to the closure parameter of
/// [`PipelineCommandRef::record_pipeline`] which may be accessed by binding a [`RayTracePipeline`] to
/// a render pass.
///
/// # Examples
///
/// Basic usage:
///
/// ```no_run
/// # use std::sync::Arc;
/// # use ash::vk;
/// # use vk_graph::driver::DriverError;
/// # use vk_graph::driver::device::{Device, DeviceInfo};
/// # use vk_graph::driver::ray_trace::{RayTracePipeline, RayTracePipelineInfo, RayTraceShaderGroup};
/// # use vk_graph::driver::shader::Shader;
/// # use vk_graph::Graph;
/// # fn main() -> Result<(), DriverError> {
/// # let device = Arc::new(Device::new(DeviceInfo::default())?);
/// # let info = RayTracePipelineInfo::default();
/// # let my_miss_code = [0u8; 1];
/// # let my_ray_trace_pipeline = Arc::new(RayTracePipeline::create(&device, info,
///     [Shader::new_miss(my_miss_code.as_slice())],
///     [RayTraceShaderGroup::new_general(0)],
/// )?);
/// # let mut my_graph = Graph::default();
/// my_graph.begin_cmd().with_name("my ray trace pass")
///         .bind_pipeline(&my_ray_trace_pipeline)
///         .record_pipeline(move |pipeline, bindings| {
///             // During this closure we have access to the ray trace methods!
///         });
/// # Ok(()) }
/// ```
pub struct RayTrace<'a> {
    cmd_buf: vk::CommandBuffer,
    device: &'a Device,

    #[cfg(debug_assertions)]
    dynamic_stack_size: bool,

    pipeline: Arc<RayTracePipeline>,
}

impl RayTrace<'_> {
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
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::ray_trace::{RayTracePipeline, RayTracePipelineInfo, RayTraceShaderGroup};
    /// # use vk_graph::driver::shader::Shader;
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Arc::new(Device::new(DeviceInfo::default())?);
    /// # let shader = [0u8; 1];
    /// # let info = RayTracePipelineInfo::default();
    /// # let my_miss_code = [0u8; 1];
    /// # let my_ray_trace_pipeline = Arc::new(RayTracePipeline::create(&device, info,
    /// #     [Shader::new_miss(my_miss_code.as_slice())],
    /// #     [RayTraceShaderGroup::new_general(0)],
    /// # )?);
    /// # let rgen_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let hit_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let miss_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let call_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let mut my_graph = Graph::default();
    /// my_graph.begin_cmd().with_name("draw a cornell box")
    ///         .bind_pipeline(&my_ray_trace_pipeline)
    ///         .record_pipeline(move |pipeline, bindings| {
    ///             pipeline.push_constants(&[0xcb])
    ///                     .trace_rays(&rgen_sbt, &hit_sbt, &miss_sbt, &call_sbt, 320, 200, 1);
    ///         });
    /// # Ok(()) }
    /// ```
    ///
    /// [gpuinfo.org]: https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxPushConstantsSize&platform=all
    pub fn push_constants(&self, data: &[u8]) -> &Self {
        self.push_constants_offset(0, data)
    }

    /// Updates push constants starting at the given `offset`.
    ///
    /// Behaves similary to [`RayTrace::push_constants`] except that `offset` describes the position
    /// at which `data` updates the push constants of the currently bound pipeline. This may be used
    /// to update a subset or single field of previously set push constant data.
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
    ///     layout(offset = 0) uint some_val1;
    ///     layout(offset = 4) uint some_val2;
    /// } push_constants;
    ///
    /// void main()
    /// {
    ///     // TODO: Add bindings to write things!
    /// }
    /// # "#);
    /// ```
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::ray_trace::{RayTracePipeline, RayTracePipelineInfo, RayTraceShaderGroup};
    /// # use vk_graph::driver::shader::Shader;
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Arc::new(Device::new(DeviceInfo::default())?);
    /// # let shader = [0u8; 1];
    /// # let info = RayTracePipelineInfo::default();
    /// # let my_miss_code = [0u8; 1];
    /// # let my_ray_trace_pipeline = Arc::new(RayTracePipeline::create(&device, info,
    /// #     [Shader::new_miss(my_miss_code.as_slice())],
    /// #     [RayTraceShaderGroup::new_general(0)],
    /// # )?);
    /// # let rgen_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let hit_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let miss_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let call_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let mut my_graph = Graph::default();
    /// my_graph.begin_cmd().with_name("draw a cornell box")
    ///         .bind_pipeline(&my_ray_trace_pipeline)
    ///         .record_pipeline(move |pipeline, bindings| {
    ///             pipeline.push_constants(&[0xcb, 0xff])
    ///                     .trace_rays(&rgen_sbt, &hit_sbt, &miss_sbt, &call_sbt, 320, 200, 1)
    ///                     .push_constants_offset(4, &[0xae])
    ///                     .trace_rays(&rgen_sbt, &hit_sbt, &miss_sbt, &call_sbt, 320, 200, 1);
    ///         });
    /// # Ok(()) }
    /// ```
    ///
    /// [gpuinfo.org]: https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxPushConstantsSize&platform=all
    #[profiling::function]
    pub fn push_constants_offset(&self, offset: u32, data: &[u8]) -> &Self {
        for push_const in self.pipeline.push_constants.iter() {
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
                        self.pipeline.layout,
                        push_const.stage_flags,
                        start,
                        &data[(start - offset) as usize..(end - offset) as usize],
                    );
                }
            }
        }
        self
    }

    /// Set the stack size dynamically for a ray trace pipeline.
    ///
    /// See
    /// [`RayTracePipelineInfo::dynamic_stack_size`](crate::driver::ray_trace::RayTracePipelineInfo::dynamic_stack_size)
    /// and
    /// [`vkCmdSetRayTracingPipelineStackSizeKHR`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdSetRayTracingPipelineStackSizeKHR.html).
    #[profiling::function]
    pub fn set_stack_size(&self, pipeline_stack_size: u32) -> &Self {
        #[cfg(debug_assertions)]
        assert!(self.dynamic_stack_size);

        unsafe {
            Device::expect_ray_trace_ext(self.device)
                .cmd_set_ray_tracing_pipeline_stack_size(self.cmd_buf, pipeline_stack_size);
        }

        self
    }

    // TODO: If the rayTraversalPrimitiveCulling or rayQuery features are enabled, the SkipTrianglesKHR and SkipAABBsKHR ray flags can be specified when tracing a ray. SkipTrianglesKHR and SkipAABBsKHR are mutually exclusive.

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
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::ray_trace::{RayTracePipeline, RayTracePipelineInfo, RayTraceShaderGroup};
    /// # use vk_graph::driver::shader::Shader;
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Arc::new(Device::new(DeviceInfo::default())?);
    /// # let shader = [0u8; 1];
    /// # let info = RayTracePipelineInfo::default();
    /// # let my_miss_code = [0u8; 1];
    /// # let my_ray_trace_pipeline = Arc::new(RayTracePipeline::create(&device, info,
    /// #     [Shader::new_miss(my_miss_code.as_slice())],
    /// #     [RayTraceShaderGroup::new_general(0)],
    /// # )?);
    /// # let rgen_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let hit_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let miss_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let call_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let mut my_graph = Graph::default();
    /// my_graph.begin_cmd().with_name("draw a cornell box")
    ///         .bind_pipeline(&my_ray_trace_pipeline)
    ///         .record_pipeline(move |pipeline, bindings| {
    ///             pipeline.trace_rays(&rgen_sbt, &hit_sbt, &miss_sbt, &call_sbt, 320, 200, 1);
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
        unsafe {
            Device::expect_ray_trace_ext(self.device).cmd_trace_rays(
                self.cmd_buf,
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
    /// See [`vkCmdTraceRaysIndirectKHR`](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCmdTraceRaysIndirectKHR.html).
    ///
    /// [buffer device address]: Buffer::device_address
    #[profiling::function]
    pub fn trace_rays_indirect(
        &self,
        raygen_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        miss_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        hit_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        callable_shader_binding_table: &vk::StridedDeviceAddressRegionKHR,
        indirect_device_address: vk::DeviceAddress,
    ) -> &Self {
        unsafe {
            Device::expect_ray_trace_ext(self.device).cmd_trace_rays_indirect(
                self.cmd_buf,
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
