//! Ray tracing pipeline types

use {
    super::{
        DriverError,
        device::Device,
        merge_push_constant_ranges,
        physical_device::RayTraceProperties,
        shader::{DescriptorBindingMap, PipelineDescriptorInfo, Shader},
    },
    ash::vk,
    derive_builder::{Builder, UninitializedFieldError},
    log::warn,
    std::{ffi::CString, thread::panicking},
};

/// Smart pointer handle to a [pipeline] object.
///
/// Also contains information about the object.
///
/// ## `Deref` behavior
///
/// `RayTracePipeline` automatically dereferences to [`vk::Pipeline`] (via the [`Deref`]
/// trait), so you can call `vk::Pipeline`'s methods on a value of type `RayTracePipeline`. To avoid
/// name clashes with `vk::Pipeline`'s methods, the methods of `RayTracePipeline` itself are
/// associated functions, called using [fully qualified syntax]:
///
/// [pipeline]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipeline.html
/// [deref]: core::ops::Deref
/// [fully qualified syntax]: https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#fully-qualified-syntax-for-disambiguation-calling-methods-with-the-same-name
#[derive(Debug)]
#[readonly::make]
pub struct RayTracePipeline {
    pub(crate) descriptor_bindings: DescriptorBindingMap,
    pub(crate) descriptor_info: PipelineDescriptorInfo,

    /// The device which owns this buffer resource.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub device: Device,

    /// The native Vulkan resource handle of this pipeline.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub handle: vk::Pipeline,

    /// Information used to create this object.
    #[readonly]
    pub info: RayTracePipelineInfo,

    pub(crate) layout: vk::PipelineLayout,

    /// A descriptive name used in debugging messages.
    pub name: Option<String>,

    pub(crate) push_constants: Vec<vk::PushConstantRange>,
    shader_modules: Vec<vk::ShaderModule>,
    shader_group_handles: Vec<u8>,
}

impl RayTracePipeline {
    /// Creates a new ray trace pipeline on the given device.
    ///
    /// The correct pipeline stages will be enabled based on the provided shaders. See [Shader] for
    /// details on all available stages.
    ///
    /// The number and composition of the `shader_groups` parameter must match the actual shaders
    /// provided.
    ///
    /// # Panics
    ///
    /// If shader code is not a multiple of four bytes.
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
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Arc::new(Device::new(DeviceInfo::default())?);
    /// # let my_rgen_code = [0u8; 1];
    /// # let my_chit_code = [0u8; 1];
    /// # let my_miss_code = [0u8; 1];
    /// # let my_shadow_code = [0u8; 1];
    /// // shader code is raw SPIR-V code as bytes
    /// let info = RayTracePipelineInfo::default().to_builder().max_ray_recursion_depth(1);
    /// let pipeline = RayTracePipeline::create(
    ///     &device,
    ///     info,
    ///     [
    ///         Shader::new_ray_gen(my_rgen_code.as_slice()),
    ///         Shader::new_closest_hit(my_chit_code.as_slice()),
    ///         Shader::new_miss(my_miss_code.as_slice()),
    ///         Shader::new_miss(my_shadow_code.as_slice()),
    ///     ],
    ///     [
    ///         RayTraceShaderGroup::new_general(0),
    ///         RayTraceShaderGroup::new_triangles(1, None),
    ///         RayTraceShaderGroup::new_general(2),
    ///         RayTraceShaderGroup::new_general(3),
    ///     ],
    /// )?;
    ///
    /// assert_ne!(pipeline.handle, vk::Pipeline::null());
    /// assert_eq!(pipeline.info.max_ray_recursion_depth, 1);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn create<S>(
        device: &Device,
        info: impl Into<RayTracePipelineInfo>,
        shaders: impl IntoIterator<Item = S>,
        shader_groups: impl IntoIterator<Item = RayTraceShaderGroup>,
    ) -> Result<Self, DriverError>
    where
        S: Into<Shader>,
    {
        if device.physical_device.ray_trace_properties.is_none() {
            return Err(DriverError::Unsupported);
        }

        let info = info.into();
        let shader_groups = shader_groups
            .into_iter()
            .map(|shader_group| shader_group.into())
            .collect::<Vec<_>>();
        let group_count = shader_groups.len();

        let shaders = shaders
            .into_iter()
            .map(|shader| shader.into())
            .collect::<Vec<Shader>>();
        let push_constants = shaders
            .iter()
            .map(|shader| shader.push_constant_range())
            .filter_map(|mut push_const| push_const.take())
            .collect::<Vec<_>>();

        // Use SPIR-V reflection to get the types and counts of all descriptors
        let mut descriptor_bindings = Shader::merge_descriptor_bindings(
            shaders.iter().map(|shader| shader.descriptor_bindings()),
        );
        for (descriptor_info, _) in descriptor_bindings.values_mut() {
            if descriptor_info.binding_count() == 0 {
                descriptor_info.set_binding_count(info.bindless_descriptor_count);
            }
        }

        let descriptor_info = PipelineDescriptorInfo::create(device, &descriptor_bindings)?;
        let layouts = descriptor_info
            .layouts
            .values()
            .map(|layout| layout.handle)
            .collect::<Box<[_]>>();

        unsafe {
            let layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&layouts)
                        .push_constant_ranges(&push_constants),
                    None,
                )
                .map_err(|err| {
                    warn!("{err}");

                    DriverError::Unsupported
                })?;
            let entry_points: Box<[CString]> = shaders
                .iter()
                .map(|shader| CString::new(shader.entry_name.as_str()))
                .collect::<Result<_, _>>()
                .map_err(|err| {
                    warn!("{err}");

                    DriverError::InvalidData
                })?;
            let specialization_infos: Box<[Option<vk::SpecializationInfo>]> = shaders
                .iter()
                .map(|shader| {
                    shader.specialization_info.as_ref().map(|info| {
                        vk::SpecializationInfo::default()
                            .data(&info.data)
                            .map_entries(&info.map_entries)
                    })
                })
                .collect();
            let mut shader_stages: Vec<vk::PipelineShaderStageCreateInfo> =
                Vec::with_capacity(shaders.len());
            let mut shader_modules = Vec::with_capacity(shaders.len());
            for (idx, shader) in shaders.iter().enumerate() {
                let module = device
                    .create_shader_module(
                        &vk::ShaderModuleCreateInfo::default().code(shader.spirv.words()),
                        None,
                    )
                    .map_err(|err| {
                        warn!("{err}");

                        device.destroy_pipeline_layout(layout, None);

                        for module in shader_modules.drain(..) {
                            device.destroy_shader_module(module, None);
                        }

                        DriverError::Unsupported
                    })?;

                shader_modules.push(module);

                let mut stage = vk::PipelineShaderStageCreateInfo::default()
                    .module(module)
                    .name(entry_points[idx].as_ref())
                    .stage(shader.stage);

                if let Some(specialization_info) = &specialization_infos[idx] {
                    stage = stage.specialization_info(specialization_info);
                }

                shader_stages.push(stage);
            }

            let mut dynamic_states = Vec::with_capacity(1);

            if info.dynamic_stack_size {
                dynamic_states.push(vk::DynamicState::RAY_TRACING_PIPELINE_STACK_SIZE_KHR);
            }

            let ray_trace_ext = Device::expect_ray_trace_ext(device);
            let handle = ray_trace_ext
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    Device::pipeline_cache(device),
                    &[vk::RayTracingPipelineCreateInfoKHR::default()
                        .stages(&shader_stages)
                        .groups(&shader_groups)
                        .max_pipeline_ray_recursion_depth(
                            info.max_ray_recursion_depth.min(
                                device
                                    .physical_device
                                    .ray_trace_properties
                                    .as_ref()
                                    .unwrap()
                                    .max_ray_recursion_depth,
                            ),
                        )
                        .layout(layout)
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default()
                                .dynamic_states(&dynamic_states),
                        )],
                    None,
                )
                .map_err(|(pipelines, err)| {
                    warn!("{err}");

                    for pipeline in pipelines {
                        device.destroy_pipeline(pipeline, None);
                    }

                    device.destroy_pipeline_layout(layout, None);

                    for shader_module in shader_modules.iter().copied() {
                        device.destroy_shader_module(shader_module, None);
                    }

                    DriverError::Unsupported
                })?[0];
            let device = device.clone();
            let &RayTraceProperties {
                shader_group_handle_size,
                ..
            } = device
                .physical_device
                .ray_trace_properties
                .as_ref()
                .unwrap();

            let push_constants = merge_push_constant_ranges(&push_constants);

            // SAFETY:
            // According to [vulkan spec](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkGetRayTracingShaderGroupHandlesKHR.html)
            // Valid usage of this function requires:
            // 1. pipeline must be raytracing pipeline.
            // 2. first_group must be less than the number of shader groups in the pipeline.
            // 3. the sum of first group and group_count must be less or equal to the number of shader
            //    modules in the pipeline.
            // 4. data_size must be at least shader_group_handle_size * group_count.
            // 5. pipeline must not have been created with VK_PIPELINE_CREATE_LIBRARY_BIT_KHR.
            //
            let shader_group_handles = {
                ray_trace_ext.get_ray_tracing_shader_group_handles(
                    handle,
                    0,
                    group_count as u32,
                    group_count * shader_group_handle_size as usize,
                )
            }
            .map_err(|_| DriverError::InvalidData)?;

            Ok(Self {
                descriptor_bindings,
                descriptor_info,
                device,
                handle,
                info,
                layout,
                name: None,
                push_constants,
                shader_modules,
                shader_group_handles,
            })
        }
    }

    /// Function returning a handle to a shader group of this pipeline.
    /// This can be used to construct a sbt.
    ///
    /// # Examples
    ///
    /// See
    /// [ray_trace.rs](https://github.com/attackgoat/vk-graph/blob/master/examples/ray_trace.rs)
    /// for a detail example which constructs a shader binding table buffer using this function.
    pub fn group_handle(&self, idx: usize) -> &[u8] {
        let &RayTraceProperties {
            shader_group_handle_size,
            ..
        } = self
            .device
            .physical_device
            .ray_trace_properties
            .as_ref()
            .unwrap();
        let start = idx * shader_group_handle_size as usize;
        let end = start + shader_group_handle_size as usize;

        &self.shader_group_handles[start..end]
    }

    /// Query ray trace pipeline shader group shader stack size.
    ///
    /// The return value is the ray tracing pipeline stack size in bytes for the specified shader as
    /// called from the specified shader group.
    #[profiling::function]
    pub fn group_stack_size(
        &self,
        group: u32,
        group_shader: vk::ShaderGroupShaderKHR,
    ) -> vk::DeviceSize {
        unsafe {
            // Safely use unchecked because ray_trace_ext is checked during pipeline creation
            Device::expect_ray_trace_ext(&self.device).get_ray_tracing_shader_group_stack_size(
                self.handle,
                group,
                group_shader,
            )
        }
    }

    /// Sets the debugging name assigned to this pipeline.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl Drop for RayTracePipeline {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        unsafe {
            self.device.destroy_pipeline(self.handle, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }

        for shader_module in self.shader_modules.drain(..) {
            unsafe {
                self.device.destroy_shader_module(shader_module, None);
            }
        }
    }
}

/// Information used to create a [`RayTracePipeline`] instance.
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build", error = "UninitializedFieldError"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
#[non_exhaustive]
pub struct RayTracePipelineInfo {
    /// The number of descriptors to allocate for a given binding when using bindless (unbounded)
    /// syntax.
    ///
    /// The default is `8192`.
    ///
    /// # Examples
    ///
    /// Basic usage (GLSL):
    ///
    /// ```
    /// # vk_shader_macros::glsl!(target: vulkan1_2, r#"
    /// #version 460 core
    /// #extension GL_EXT_nonuniform_qualifier : require
    /// #pragma shader_stage(closest)
    ///
    /// layout(set = 0, binding = 0, rgba8) readonly uniform image2D my_binding[];
    ///
    /// void main() {
    ///     // my_binding will have space for 8,192 images by default
    /// }
    /// # "#);
    /// ```
    #[builder(default = "8192")]
    pub bindless_descriptor_count: u32,

    /// Allow [setting the stack size dynamically] for a ray trace pipeline.
    ///
    /// When set, you must manually set the stack size during ray trace passes using
    /// [`RayTrace::set_stack_size`](crate::graph::pass_ref::RayTrace::set_stack_size).
    ///
    /// [setting the stack size dynamically]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdSetRayTracingPipelineStackSizeKHR.html
    #[builder(default)]
    pub dynamic_stack_size: bool,

    /// The [maximum recursion depth] of shaders executed by this pipeline.
    ///
    /// The default is `16`.
    ///
    /// [maximum recursion depth]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#ray-tracing-recursion-depth
    #[builder(default = "16")]
    pub max_ray_recursion_depth: u32,
}

impl RayTracePipelineInfo {
    /// Creates a default `RayTracePipelineInfoBuilder`.
    pub fn builder() -> RayTracePipelineInfoBuilder {
        Default::default()
    }

    /// Converts a `RayTracePipelineInfo` into a `RayTracePipelineInfoBuilder`.
    #[inline(always)]
    pub fn to_builder(self) -> RayTracePipelineInfoBuilder {
        RayTracePipelineInfoBuilder {
            bindless_descriptor_count: Some(self.bindless_descriptor_count),
            dynamic_stack_size: Some(self.dynamic_stack_size),
            max_ray_recursion_depth: Some(self.max_ray_recursion_depth),
        }
    }
}

impl Default for RayTracePipelineInfo {
    fn default() -> Self {
        Self {
            bindless_descriptor_count: 8192,
            dynamic_stack_size: false,
            max_ray_recursion_depth: 16,
        }
    }
}

impl From<RayTracePipelineInfoBuilder> for RayTracePipelineInfo {
    fn from(info: RayTracePipelineInfoBuilder) -> Self {
        info.build()
    }
}

impl RayTracePipelineInfoBuilder {
    /// Builds a new `RayTracePipelineInfo`.
    #[inline(always)]
    pub fn build(self) -> RayTracePipelineInfo {
        let res = self.fallible_build();

        #[cfg(test)]
        let res = res.unwrap();

        #[cfg(not(test))]
        let res = unsafe { res.unwrap_unchecked() };

        res
    }
}

/// Describes the set of the shader stages to be included in each shader group in the ray trace
/// pipeline.
///
/// See
/// [VkRayTracingShaderGroupCreateInfoKHR](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VkRayTracingShaderGroupCreateInfoKHR).
#[derive(Clone, Copy, Debug)]
pub struct RayTraceShaderGroup {
    /// The optional index of the any-hit shader in the group if the shader group has type of
    /// [RayTraceShaderGroupType::TrianglesHitGroup] or
    /// [RayTraceShaderGroupType::ProceduralHitGroup].
    pub any_hit_shader: Option<u32>,

    /// The optional index of the closest hit shader in the group if the shader group has type of
    /// [RayTraceShaderGroupType::TrianglesHitGroup] or
    /// [RayTraceShaderGroupType::ProceduralHitGroup].
    pub closest_hit_shader: Option<u32>,

    /// The index of the ray generation, miss, or callable shader in the group if the shader group
    /// has type of [RayTraceShaderGroupType::General].
    pub general_shader: Option<u32>,

    /// The index of the intersection shader in the group if the shader group has type of
    /// [RayTraceShaderGroupType::ProceduralHitGroup].
    pub intersection_shader: Option<u32>,

    /// The type of hit group specified in this structure.
    pub ty: RayTraceShaderGroupType,
}

impl RayTraceShaderGroup {
    fn new(
        ty: RayTraceShaderGroupType,
        general_shader: impl Into<Option<u32>>,
        intersection_shader: impl Into<Option<u32>>,
        closest_hit_shader: impl Into<Option<u32>>,
        any_hit_shader: impl Into<Option<u32>>,
    ) -> Self {
        let any_hit_shader = any_hit_shader.into();
        let closest_hit_shader = closest_hit_shader.into();
        let general_shader = general_shader.into();
        let intersection_shader = intersection_shader.into();

        Self {
            any_hit_shader,
            closest_hit_shader,
            general_shader,
            intersection_shader,
            ty,
        }
    }

    /// Creates a new general-type shader group with the given general shader.
    pub fn new_general(general_shader: impl Into<Option<u32>>) -> Self {
        Self::new(
            RayTraceShaderGroupType::General,
            general_shader,
            None,
            None,
            None,
        )
    }

    /// Creates a new procedural-type shader group with the given intersection shader, and optional
    /// closest-hit and any-hit shaders.
    pub fn new_procedural(
        intersection_shader: u32,
        closest_hit_shader: impl Into<Option<u32>>,
        any_hit_shader: impl Into<Option<u32>>,
    ) -> Self {
        Self::new(
            RayTraceShaderGroupType::ProceduralHitGroup,
            None,
            intersection_shader,
            closest_hit_shader,
            any_hit_shader,
        )
    }

    /// Creates a new triangles-type shader group with the given closest-hit shader and optional any-hit
    /// shader.
    pub fn new_triangles(closest_hit_shader: u32, any_hit_shader: impl Into<Option<u32>>) -> Self {
        Self::new(
            RayTraceShaderGroupType::TrianglesHitGroup,
            None,
            None,
            closest_hit_shader,
            any_hit_shader,
        )
    }
}

impl From<RayTraceShaderGroup> for vk::RayTracingShaderGroupCreateInfoKHR<'static> {
    fn from(shader_group: RayTraceShaderGroup) -> Self {
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(shader_group.ty.into())
            .any_hit_shader(shader_group.any_hit_shader.unwrap_or(vk::SHADER_UNUSED_KHR))
            .closest_hit_shader(
                shader_group
                    .closest_hit_shader
                    .unwrap_or(vk::SHADER_UNUSED_KHR),
            )
            .general_shader(shader_group.general_shader.unwrap_or(vk::SHADER_UNUSED_KHR))
            .intersection_shader(
                shader_group
                    .intersection_shader
                    .unwrap_or(vk::SHADER_UNUSED_KHR),
            )
    }
}

/// Describes a type of ray tracing shader group, which is a collection of shaders which run in the
/// specified mode.
#[derive(Clone, Copy, Debug)]
pub enum RayTraceShaderGroupType {
    /// A shader group with a general shader.
    General,

    /// A shader group with an intersection shader, and optional closest-hit and any-hit shaders.
    ProceduralHitGroup,

    /// A shader group with a closest-hit shader and optional any-hit shader.
    TrianglesHitGroup,
}

impl From<RayTraceShaderGroupType> for vk::RayTracingShaderGroupTypeKHR {
    fn from(ty: RayTraceShaderGroupType) -> Self {
        match ty {
            RayTraceShaderGroupType::General => vk::RayTracingShaderGroupTypeKHR::GENERAL,
            RayTraceShaderGroupType::ProceduralHitGroup => {
                vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP
            }
            RayTraceShaderGroupType::TrianglesHitGroup => {
                vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Info = RayTracePipelineInfo;
    type Builder = RayTracePipelineInfoBuilder;

    #[test]
    pub fn ray_trace_pipeline_info() {
        let info = Info::default();
        let builder = info.to_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn ray_trace_pipeline_info_builder() {
        let info = Info::default();
        let builder = Builder::default().build();

        assert_eq!(info, builder);
    }
}
