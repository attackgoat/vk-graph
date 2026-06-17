//! Ray tracing pipeline types

use {
    super::{
        DriverError,
        device::Device,
        merge_push_constant_ranges,
        physical_device::RayTracingPipelineProperties,
        shader::{DescriptorBindingMap, PipelineDescriptorInfo, Shader},
    },
    crate::lazy_str,
    ash::vk::{self, Handle},
    derive_builder::Builder,
    log::warn,
    std::{
        ffi::CString,
        fmt::{Debug, Formatter},
        hash::{Hash, Hasher},
        sync::Arc,
        thread::panicking,
    },
};

/// Smart pointer handle of a ray tracing pipeline object.
///
/// Also contains information about the object.
///
/// See [`VkPipeline`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipeline.html).
#[derive(Clone)]
#[read_only::cast]
pub struct RayTracingPipeline {
    pub(crate) inner: Arc<RayTracingPipelineInner>,
}

impl RayTracingPipeline {
    /// Creates a new ray tracing pipeline on the given device.
    ///
    /// The correct pipeline stages will be enabled based on the provided shaders. See [`Shader`]
    /// for details on all available stages.
    ///
    /// See [`VkRayTracingPipelineCreateInfoKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkRayTracingPipelineCreateInfoKHR.html).
    ///
    /// The number and composition of the `shader_groups` parameter must match the actual shaders
    /// provided.
    ///
    /// `shaders` may contain pre-built [`Shader`] values or any inputs that can be converted into
    /// them. Invalid shader data is returned as [`DriverError::InvalidData`] through the `Result`
    /// instead of panicking.
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
    /// # use vk_graph::driver::ray_tracing::{
    /// #     RayTracingPipeline,
    /// #     RayTracingPipelineInfo,
    /// #     RayTracingShaderGroup,
    /// # };
    /// # use vk_graph::driver::shader::Shader;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// # let my_rgen_code = [0u8; 1];
    /// # let my_chit_code = [0u8; 1];
    /// # let my_miss_code = [0u8; 1];
    /// # let my_shadow_code = [0u8; 1];
    /// // shader code is raw SPIR-V code as bytes
    /// let info = RayTracingPipelineInfo::default().into_builder().max_ray_recursion_depth(1);
    /// let pipeline = RayTracingPipeline::create(
    ///     &device,
    ///     info,
    ///     [
    ///         Shader::new_ray_gen(my_rgen_code.as_slice()),
    ///         Shader::new_closest_hit(my_chit_code.as_slice()),
    ///         Shader::new_miss(my_miss_code.as_slice()),
    ///         Shader::new_miss(my_shadow_code.as_slice()),
    ///     ],
    ///     [
    ///         RayTracingShaderGroup::new_general(0),
    ///         RayTracingShaderGroup::new_triangles(1, None),
    ///         RayTracingShaderGroup::new_general(2),
    ///         RayTracingShaderGroup::new_general(3),
    ///     ],
    /// )?;
    ///
    /// assert_ne!(pipeline.handle(), vk::Pipeline::null());
    /// assert_eq!(pipeline.info().max_ray_recursion_depth, 1);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn create<S>(
        device: &Device,
        info: impl Into<RayTracingPipelineInfo>,
        shaders: impl IntoIterator<Item = S>,
        shader_groups: impl IntoIterator<Item = RayTracingShaderGroup>,
    ) -> Result<Self, DriverError>
    where
        S: TryInto<Shader>,
        S::Error: Into<DriverError>,
    {
        if device
            .physical_device
            .ray_tracing_pipeline_properties
            .is_none()
        {
            warn!("unsupported ray tracing pipeline creation: missing ray tracing properties");

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
            .map(|shader| shader.try_into().map_err(Into::into))
            .collect::<Result<Vec<_>, _>>()?;
        let push_constants = shaders
            .iter()
            .map(|shader| shader.push_constant_range())
            .filter_map(|mut push_const| push_const.take())
            .collect::<Vec<_>>();

        // Use SPIR-V reflection to get the types and counts of all descriptors
        let mut descriptor_bindings = Shader::merge_descriptor_bindings(
            shaders.iter().map(|shader| shader.descriptor_bindings()),
        )?;
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
            .collect::<Box<_>>();

        unsafe {
            let layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&layouts)
                        .push_constant_ranges(&push_constants),
                    None,
                )
                .map_err(|err| {
                    warn!("unable to create ray tracing pipeline layout: {err}");

                    DriverError::Unsupported
                })?;
            let entry_points: Box<[CString]> = shaders
                .iter()
                .map(|shader| CString::new(shader.entry_name.as_str()))
                .collect::<Result<_, _>>()
                .map_err(|err| {
                    warn!("invalid ray tracing shader entry name: {err}");

                    DriverError::InvalidData
                })?;
            let specialization_infos: Box<[Option<vk::SpecializationInfo>]> = shaders
                .iter()
                .map(|shader| shader.specialization.as_ref().map(Into::into))
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
                        warn!("unable to create ray tracing shader module: {err}");

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

            let khr_ray_tracing_pipeline = Device::expect_vk_khr_ray_tracing_pipeline(device);

            let handle = khr_ray_tracing_pipeline.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                Device::pipeline_cache(device),
                &[vk::RayTracingPipelineCreateInfoKHR::default()
                    .stages(&shader_stages)
                    .groups(&shader_groups)
                    .max_pipeline_ray_recursion_depth(
                        info.max_ray_recursion_depth.min(
                            device
                                .physical_device
                                .expect_ray_tracing_pipeline_properties()
                                .max_ray_recursion_depth,
                        ),
                    )
                    .layout(layout)
                    .dynamic_state(
                        &vk::PipelineDynamicStateCreateInfo::default()
                            .dynamic_states(&dynamic_states),
                    )],
                None,
            );

            for shader_module in shader_modules.iter().copied() {
                device.destroy_shader_module(shader_module, None);
            }

            let handle = handle
                .map_err(|(pipelines, err)| {
                    warn!("unable to create ray tracing pipeline: {err}");

                    for pipeline in pipelines {
                        device.destroy_pipeline(pipeline, None);
                    }

                    device.destroy_pipeline_layout(layout, None);

                    DriverError::Unsupported
                })?
                .into_iter()
                .find(|handle| !handle.is_null())
                .ok_or_else(|| {
                    warn!("missing pipeline handle");

                    DriverError::Unsupported
                })?;
            let &RayTracingPipelineProperties {
                shader_group_handle_size,
                ..
            } = device
                .physical_device
                .expect_ray_tracing_pipeline_properties();

            let push_constants = merge_push_constant_ranges(&push_constants).into_boxed_slice();

            /*
            SAFETY:
            See [`vkGetRayTracingShaderGroupHandlesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetRayTracingShaderGroupHandlesKHR.html)
            Valid usage of this function requires:
            1. pipeline must be a ray tracing pipeline
            2. first_group must be less than the number of shader groups in the pipeline
            3. the sum of first_group and group_count must be less than or equal to the number of
               shader groups in the pipeline
            4. data_size must be at least shader_group_handle_size * group_count
            5. pipeline must not have been created with VK_PIPELINE_CREATE_LIBRARY_BIT_KHR
            */
            let shader_group_handles = {
                khr_ray_tracing_pipeline.get_ray_tracing_shader_group_handles(
                    handle,
                    0,
                    group_count as u32,
                    group_count * shader_group_handle_size as usize,
                )
            }
            .map_err(|_| DriverError::InvalidData)?
            .into_boxed_slice();

            Ok(Self {
                inner: Arc::new(RayTracingPipelineInner {
                    descriptor_bindings,
                    descriptor_info,
                    device: device.clone(),
                    handle,
                    info,
                    layout,
                    push_constants,
                    shader_group_handles,
                }),
            })
        }
    }

    /// The device which owns this ray tracing pipeline.
    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    /// Returns a handle to a shader group of this pipeline.
    ///
    /// This can be used to construct a shader binding table.
    ///
    /// # Examples
    ///
    /// See
    /// [`ray_tracing.rs`](https://github.com/attackgoat/vk-graph/blob/master/examples/ray_tracing.rs)
    /// for a detailed example that constructs a shader binding table buffer using this function.
    ///
    /// See [`vkGetRayTracingShaderGroupHandlesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetRayTracingShaderGroupHandlesKHR.html).
    pub fn group_handle(&self, idx: usize) -> &[u8] {
        let &RayTracingPipelineProperties {
            shader_group_handle_size,
            ..
        } = self
            .inner
            .device
            .physical_device
            .expect_ray_tracing_pipeline_properties();
        let start = idx * shader_group_handle_size as usize;
        let end = start + shader_group_handle_size as usize;

        &self.inner.shader_group_handles[start..end]
    }

    /// Queries ray tracing pipeline shader group shader stack size.
    ///
    /// The return value is the ray tracing pipeline stack size in bytes for the specified shader as
    /// called from the specified shader group.
    ///
    /// See [`vkGetRayTracingShaderGroupStackSizeKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetRayTracingShaderGroupStackSizeKHR.html).
    #[profiling::function]
    pub fn group_stack_size(
        &self,
        group: u32,
        group_shader: vk::ShaderGroupShaderKHR,
    ) -> vk::DeviceSize {
        /*
        Safely use unchecked because the ray tracing extension is checked during pipeline creation
        */
        let khr_ray_tracing_pipeline =
            Device::expect_vk_khr_ray_tracing_pipeline(&self.inner.device);

        unsafe {
            khr_ray_tracing_pipeline.get_ray_tracing_shader_group_stack_size(
                self.handle(),
                group,
                group_shader,
            )
        }
    }

    /// The native Vulkan pipeline handle of this ray tracing pipeline.
    pub fn handle(&self) -> vk::Pipeline {
        self.inner.handle
    }

    /// Gets the information used to create this object.
    pub fn info(&self) -> RayTracingPipelineInfo {
        self.inner.info
    }

    /// Sets the debugging name assigned to this pipeline.
    pub fn set_debug_name(&self, name: impl AsRef<str>) {
        Device::try_set_debug_utils_object_name(&self.inner.device, self.inner.handle, &name);
        Device::try_set_private_data_object_name(
            &self.inner.device,
            vk::ObjectType::PIPELINE,
            self.inner.handle,
            &name,
        );
        Device::try_set_debug_utils_object_name(
            &self.inner.device,
            self.inner.layout,
            lazy_str!("{} (layout)", name.as_ref()),
        );

        for (set_idx, layout) in &self.inner.descriptor_info.layouts {
            layout.set_debug_name(lazy_str!("{} (DS{set_idx})", name.as_ref()));
        }
    }

    /// Sets the debugging name assigned to this pipeline.
    pub fn with_debug_name(self, name: impl AsRef<str>) -> Self {
        self.set_debug_name(name);

        self
    }
}

impl Debug for RayTracingPipeline {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut res = f.debug_struct(stringify!(RayTracingPipeline));

        if let Some(debug_name) = &Device::private_data_object_name(
            &self.inner.device,
            vk::ObjectType::PIPELINE,
            self.inner.handle,
        ) {
            res.field("debug_name", debug_name);
        }

        res.field("handle", &self.inner.handle)
            .finish_non_exhaustive()
    }
}

impl Eq for RayTracingPipeline {}

impl Hash for RayTracingPipeline {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}

impl PartialEq for RayTracingPipeline {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

/// Information used to create a [`RayTracingPipeline`] instance.
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct RayTracingPipelineInfo {
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

    /// Allow [setting the stack size dynamically] for a ray tracing pipeline.
    ///
    /// When set, you must manually set the stack size during ray tracing commands using
    /// [`RayTracingCommandRef::set_stack_size`](crate::cmd::RayTracingCommandRef::set_stack_size).
    ///
    /// See [`vkCmdSetRayTracingPipelineStackSizeKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdSetRayTracingPipelineStackSizeKHR.html).
    #[builder(default)]
    pub dynamic_stack_size: bool,

    /// The [maximum recursion depth] of shaders executed by this pipeline.
    ///
    /// The default is `16`.
    ///
    /// See [`VkRayTracingPipelineCreateInfoKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkRayTracingPipelineCreateInfoKHR.html).
    #[builder(default = "16")]
    pub max_ray_recursion_depth: u32,
}

impl RayTracingPipelineInfo {
    /// Creates a default `RayTracingPipelineInfoBuilder`.
    pub fn builder() -> RayTracingPipelineInfoBuilder {
        Default::default()
    }

    /// Converts a `RayTracingPipelineInfo` into a `RayTracingPipelineInfoBuilder`.
    pub fn into_builder(self) -> RayTracingPipelineInfoBuilder {
        RayTracingPipelineInfoBuilder {
            bindless_descriptor_count: Some(self.bindless_descriptor_count),
            dynamic_stack_size: Some(self.dynamic_stack_size),
            max_ray_recursion_depth: Some(self.max_ray_recursion_depth),
        }
    }
}

impl Default for RayTracingPipelineInfo {
    fn default() -> Self {
        Self {
            bindless_descriptor_count: 8192,
            dynamic_stack_size: false,
            max_ray_recursion_depth: 16,
        }
    }
}

impl From<RayTracingPipelineInfoBuilder> for RayTracingPipelineInfo {
    fn from(info: RayTracingPipelineInfoBuilder) -> Self {
        info.build()
    }
}

impl RayTracingPipelineInfoBuilder {
    /// Builds a new `RayTracingPipelineInfo`.
    #[inline(always)]
    pub fn build(self) -> RayTracingPipelineInfo {
        self.fallible_build()
            .expect("invalid ray tracing pipeline info")
    }
}

#[derive(Debug)]
pub(crate) struct RayTracingPipelineInner {
    pub descriptor_bindings: DescriptorBindingMap,
    pub descriptor_info: PipelineDescriptorInfo,
    pub device: Device,
    pub handle: vk::Pipeline,
    pub info: RayTracingPipelineInfo,
    pub layout: vk::PipelineLayout,
    pub push_constants: Box<[vk::PushConstantRange]>,
    pub shader_group_handles: Box<[u8]>,
}

impl Drop for RayTracingPipelineInner {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        Device::clear_private_data_object_name(&self.device, vk::ObjectType::PIPELINE, self.handle)
            .unwrap_or_else(|err| warn!("unable to clear private data object name: {err}"));

        unsafe {
            self.device.destroy_pipeline(self.handle, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// Describes the set of shader stages to be included in each shader group in the ray tracing
/// pipeline.
///
/// See [`VkRayTracingShaderGroupCreateInfoKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkRayTracingShaderGroupCreateInfoKHR.html).
#[derive(Clone, Copy, Debug)]
pub struct RayTracingShaderGroup {
    /// The optional index of the any-hit shader in the group if the shader group has type of
    /// [RayTracingShaderGroupType::TrianglesHitGroup] or
    /// [RayTracingShaderGroupType::ProceduralHitGroup].
    pub any_hit_shader: Option<u32>,

    /// The optional index of the closest hit shader in the group if the shader group has type of
    /// [RayTracingShaderGroupType::TrianglesHitGroup] or
    /// [RayTracingShaderGroupType::ProceduralHitGroup].
    pub closest_hit_shader: Option<u32>,

    /// The index of the ray generation, miss, or callable shader in the group if the shader group
    /// has type of [RayTracingShaderGroupType::General].
    pub general_shader: Option<u32>,

    /// The index of the intersection shader in the group if the shader group has type of
    /// [RayTracingShaderGroupType::ProceduralHitGroup].
    pub intersection_shader: Option<u32>,

    /// The type of hit group specified in this structure.
    pub ty: RayTracingShaderGroupType,
}

impl RayTracingShaderGroup {
    fn new(
        ty: RayTracingShaderGroupType,
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
            RayTracingShaderGroupType::General,
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
            RayTracingShaderGroupType::ProceduralHitGroup,
            None,
            intersection_shader,
            closest_hit_shader,
            any_hit_shader,
        )
    }

    /// Creates a new triangles-type shader group with the given closest-hit shader and optional
    /// any-hit shader.
    pub fn new_triangles(closest_hit_shader: u32, any_hit_shader: impl Into<Option<u32>>) -> Self {
        Self::new(
            RayTracingShaderGroupType::TrianglesHitGroup,
            None,
            None,
            closest_hit_shader,
            any_hit_shader,
        )
    }
}

impl From<RayTracingShaderGroup> for vk::RayTracingShaderGroupCreateInfoKHR<'static> {
    fn from(shader_group: RayTracingShaderGroup) -> Self {
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
pub enum RayTracingShaderGroupType {
    /// A shader group with a general shader.
    General,

    /// A shader group with an intersection shader, and optional closest-hit and any-hit shaders.
    ProceduralHitGroup,

    /// A shader group with a closest-hit shader and optional any-hit shader.
    TrianglesHitGroup,
}

impl From<RayTracingShaderGroupType> for vk::RayTracingShaderGroupTypeKHR {
    fn from(ty: RayTracingShaderGroupType) -> Self {
        match ty {
            RayTracingShaderGroupType::General => vk::RayTracingShaderGroupTypeKHR::GENERAL,
            RayTracingShaderGroupType::ProceduralHitGroup => {
                vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP
            }
            RayTracingShaderGroupType::TrianglesHitGroup => {
                vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    type Info = RayTracingPipelineInfo;
    type Builder = RayTracingPipelineInfoBuilder;

    #[test]
    pub fn ray_tracing_pipeline_info() {
        let info = Info::default();
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn ray_tracing_pipeline_info_builder() {
        let info = Info::default();
        let builder = Builder::default().build();

        assert_eq!(info, builder);
    }
}
