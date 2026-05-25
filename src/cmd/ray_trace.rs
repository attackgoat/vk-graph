use {
    super::{PipelineCommand, cmd_buf::CommandBuffer},
    crate::driver::{device::Device, ray_trace::RayTracePipeline},
    ash::vk,
    std::ops::Deref,
};

impl PipelineCommand<'_, RayTracePipeline> {
    /// Begin recording a ray trace pipeline command buffer.
    pub fn record_cmd_buf(
        mut self,
        func: impl FnOnce(RayTraceCommandBuffer<'_>) + Send + 'static,
    ) -> Self {
        self.record_cmd_buf_mut(func);
        self
    }

    /// Begin recording a ray trace pipeline command buffer.
    pub fn record_cmd_buf_mut(
        &mut self,
        func: impl FnOnce(RayTraceCommandBuffer<'_>) + Send + 'static,
    ) {
        let pipeline = self
            .cmd
            .cmd()
            .expect_last_pipeline()
            .expect_ray_trace()
            .clone();

        #[cfg(debug_assertions)]
        let dynamic_stack_size = pipeline.inner.info.dynamic_stack_size;

        self.cmd.push_exec(move |cmd_buf| {
            func(RayTraceCommandBuffer {
                cmd_buf,

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
/// [`PipelineCommand::record_cmd_buf`] which may be accessed by binding a [`RayTracePipeline`] to
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
/// # use vk_graph::driver::ray_trace::{RayTracePipeline, RayTracePipelineInfo, RayTraceShaderGroup};
/// # use vk_graph::driver::shader::Shader;
/// # use vk_graph::Graph;
/// # fn main() -> Result<(), DriverError> {
/// # let device = Device::new(DeviceInfo::default())?;
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
///         .record_cmd_buf(move |cmd_buf| {
///             // During this closure we have access to the ray trace functions!
///         });
/// # Ok(()) }
/// ```
pub struct RayTraceCommandBuffer<'a> {
    cmd_buf: CommandBuffer<'a>,

    #[cfg(debug_assertions)]
    dynamic_stack_size: bool,

    pipeline: RayTracePipeline,
}

impl RayTraceCommandBuffer<'_> {
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
    /// # use vk_graph::driver::ray_trace::{RayTracePipeline, RayTracePipelineInfo, RayTraceShaderGroup};
    /// # use vk_graph::driver::shader::Shader;
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::new(DeviceInfo::default())?;
    /// # let shader = [0u8; 1];
    /// # let info = RayTracePipelineInfo::default();
    /// # let my_miss_code = [0u8; 1];
    /// # let my_ray_trace_pipeline = RayTracePipeline::create(&device, info,
    /// #     [Shader::new_miss(my_miss_code.as_slice())],
    /// #     [RayTraceShaderGroup::new_general(0)],
    /// # )?;
    /// # let rgen_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let hit_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let miss_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let call_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let mut my_graph = Graph::default();
    /// my_graph.begin_cmd()
    ///         .debug_name("draw a cornell box")
    ///         .bind_pipeline(&my_ray_trace_pipeline)
    ///         .record_cmd_buf(move |cmd_buf| {
    ///             cmd_buf.push_constants(0, &[0xcb])
    ///                    .trace_rays(&rgen_sbt, &hit_sbt, &miss_sbt, &call_sbt, 320, 200, 1);
    ///         });
    /// # Ok(()) }
    /// ```
    ///
    /// [gpuinfo.org]: https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxPushConstantsSize&platform=all
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
    /// [`RayTracePipelineInfo::dynamic_stack_size`](crate::driver::ray_trace::RayTracePipelineInfo::dynamic_stack_size)
    /// and
    /// [`vkCmdSetRayTracingPipelineStackSizeKHR`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdSetRayTracingPipelineStackSizeKHR.html).
    #[profiling::function]
    pub fn set_stack_size(&self, pipeline_stack_size: u32) -> &Self {
        #[cfg(debug_assertions)]
        assert!(self.dynamic_stack_size);

        let ray_trace_ext = Device::expect_ray_trace_ext(&self.cmd_buf.device);

        unsafe {
            ray_trace_ext
                .cmd_set_ray_tracing_pipeline_stack_size(self.cmd_buf.handle, pipeline_stack_size);
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
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # use vk_graph::driver::ray_trace::{RayTracePipeline, RayTracePipelineInfo, RayTraceShaderGroup};
    /// # use vk_graph::driver::shader::Shader;
    /// # use vk_graph::Graph;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::new(DeviceInfo::default())?;
    /// # let shader = [0u8; 1];
    /// # let info = RayTracePipelineInfo::default();
    /// # let my_miss_code = [0u8; 1];
    /// # let my_ray_trace_pipeline = RayTracePipeline::create(&device, info,
    /// #     [Shader::new_miss(my_miss_code.as_slice())],
    /// #     [RayTraceShaderGroup::new_general(0)],
    /// # )?;
    /// # let rgen_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let hit_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let miss_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let call_sbt = vk::StridedDeviceAddressRegionKHR { device_address: 0, stride: 0, size: 0 };
    /// # let mut my_graph = Graph::default();
    /// my_graph.begin_cmd()
    ///         .debug_name("draw a cornell box")
    ///         .bind_pipeline(&my_ray_trace_pipeline)
    ///         .record_cmd_buf(move |cmd_buf| {
    ///             cmd_buf.trace_rays(&rgen_sbt, &hit_sbt, &miss_sbt, &call_sbt, 320, 200, 1);
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
        let ray_trace_ext = Device::expect_ray_trace_ext(&self.cmd_buf.device);

        unsafe {
            ray_trace_ext.cmd_trace_rays(
                self.cmd_buf.handle,
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
        let ray_trace_ext = Device::expect_ray_trace_ext(&self.cmd_buf.device);

        unsafe {
            ray_trace_ext.cmd_trace_rays_indirect(
                self.cmd_buf.handle,
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

impl<'a> Deref for RayTraceCommandBuffer<'a> {
    type Target = CommandBuffer<'a>;

    fn deref(&self) -> &Self::Target {
        &self.cmd_buf
    }
}

#[allow(unused)]
mod deprecated {
    use {
        crate::{
            Node,
            cmd::{
                Descriptor, PipelineCommand, Subresource, SubresourceRange, ViewInfo,
                ray_trace::RayTraceCommandBuffer,
            },
            driver::ray_trace::RayTracePipeline,
        },
        vk_sync::AccessType,
    };

    impl RayTraceCommandBuffer<'_> {
        #[deprecated = "use push_constants function"]
        #[doc(hidden)]
        pub fn push_constants_offset(&self, offset: u32, data: &[u8]) -> &Self {
            self.push_constants(offset, data)
        }
    }

    impl PipelineCommand<'_, RayTracePipeline> {
        #[deprecated = "use shader_resource_access function with AccessType::RayTracingShaderReadSampledImageOrUniformTexelBuffer"]
        #[doc(hidden)]
        pub fn read_descriptor<N>(self, descriptor: impl Into<Descriptor>, node: N) -> Self
        where
            N: Node + Subresource,
            N::Info: Copy,
            SubresourceRange: From<N::Info>,
            ViewInfo: From<N::Info>,
        {
            self.shader_resource_access(
                descriptor,
                node,
                AccessType::RayTracingShaderReadSampledImageOrUniformTexelBuffer,
            )
        }

        #[deprecated = "use shader_subresource_access function with AccessType::RayTracingShaderReadSampledImageOrUniformTexelBuffer"]
        #[doc(hidden)]
        pub fn read_descriptor_as<N>(
            self,
            descriptor: impl Into<Descriptor>,
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
                AccessType::RayTracingShaderReadSampledImageOrUniformTexelBuffer,
            )
        }

        #[deprecated = "use record_cmd_buf function"]
        #[doc(hidden)]
        pub fn record_ray_trace(
            self,
            func: impl FnOnce(RayTraceCommandBuffer<'_>, ()) + Send + 'static,
        ) -> Self {
            self.record_cmd_buf(|cmd_buf| {
                func(cmd_buf, ());
            })
        }

        #[deprecated = "use shader_resource_access function with AccessType::AnyShaderWrite"]
        #[doc(hidden)]
        pub fn write_descriptor<N>(self, descriptor: impl Into<Descriptor>, node: N) -> Self
        where
            N: Node + Subresource,
            N::Info: Copy,
            SubresourceRange: From<N::Info>,
            ViewInfo: From<N::Info>,
        {
            self.shader_resource_access(descriptor, node, AccessType::AnyShaderWrite)
        }

        #[deprecated = "use shader_subresource_access function with AccessType::AnyShaderWrite"]
        #[doc(hidden)]
        pub fn write_descriptor_as<N>(
            self,
            descriptor: impl Into<Descriptor>,
            node: N,
            node_view: impl Into<N::Info>,
        ) -> Self
        where
            N: Node + Subresource,
            N::Info: Copy,
            SubresourceRange: From<N::Info>,
            ViewInfo: From<N::Info>,
        {
            self.shader_subresource_access(descriptor, node, node_view, AccessType::AnyShaderWrite)
        }
    }
}
