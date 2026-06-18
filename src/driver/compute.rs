//! Compute pipeline types.

use {
    super::{
        DriverError,
        device::Device,
        shader::{DescriptorBindingMap, PipelineDescriptorInfo, Shader},
    },
    crate::lazy_str,
    ash::vk::{self, Handle as _},
    derive_builder::Builder,
    log::{trace, warn},
    std::{
        ffi::CString,
        fmt::{Debug, Formatter},
        hash::{Hash, Hasher},
        slice,
        sync::Arc,
        thread::panicking,
    },
};

/// Smart pointer handle of a compute pipeline object.
///
/// Also contains information about the object.
///
/// See [`VkPipeline`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipeline.html).
#[derive(Clone)]
pub struct ComputePipeline {
    pub(crate) inner: Arc<ComputePipelineInner>,
}

impl ComputePipeline {
    /// Creates a new compute pipeline on the given device.
    ///
    /// `shader` may be a pre-built [`Shader`] or any input that can be converted into one.
    /// Invalid shader data is returned as [`DriverError::InvalidData`] through the `Result`
    /// instead of panicking.
    ///
    /// See [`VkComputePipelineCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkComputePipelineCreateInfo.html).
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
    /// # use vk_graph::driver::compute::{ComputePipeline, ComputePipelineInfo};
    /// # use vk_graph::driver::shader::{Shader};
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// # let my_shader_code = [0u8; 1];
    /// // my_shader_code is raw SPIR-V code as bytes
    /// let shader = Shader::new_compute(my_shader_code.as_slice());
    /// let pipeline = ComputePipeline::create(&device, ComputePipelineInfo::default(), shader)?;
    ///
    /// assert_ne!(pipeline.handle(), vk::Pipeline::null());
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn create<S>(
        device: &Device,
        info: impl Into<ComputePipelineInfo>,
        shader: S,
    ) -> Result<Self, DriverError>
    where
        S: TryInto<Shader>,
        S::Error: Into<DriverError>,
    {
        trace!("create");

        let info = info.into();
        let shader = shader.try_into().map_err(Into::into)?;

        // Use SPIR-V reflection to get the types and counts of all descriptors
        let mut descriptor_bindings = shader.descriptor_bindings();
        for (descriptor_info, _) in descriptor_bindings.values_mut() {
            if descriptor_info.binding_count() == 0 {
                descriptor_info.set_binding_count(info.bindless_descriptor_count);
            }
        }

        let descriptor_info = PipelineDescriptorInfo::create(device, &descriptor_bindings)?;
        let descriptor_set_layouts = descriptor_info
            .layouts
            .values()
            .map(|descriptor_set_layout| descriptor_set_layout.handle)
            .collect::<Box<_>>();

        unsafe {
            let shader_module = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(shader.spirv.words()),
                    None,
                )
                .map_err(|err| {
                    warn!("unable to create compute shader module: {err}");

                    DriverError::Unsupported
                })?;
            let entry_name = CString::new(shader.entry_name.as_bytes()).map_err(|err| {
                warn!("invalid compute shader entry name: {err}");

                DriverError::InvalidData
            })?;
            let mut stage_create_info = vk::PipelineShaderStageCreateInfo::default()
                .module(shader_module)
                .stage(shader.stage)
                .name(&entry_name);
            let specialization_info = shader.specialization.as_ref().map(Into::into);

            if let Some(specialization_info) = &specialization_info {
                stage_create_info = stage_create_info.specialization_info(specialization_info);
            }

            let mut layout_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);

            let push_constants = shader.push_constant_range();
            if let Some(push_constants) = &push_constants {
                layout_info = layout_info.push_constant_ranges(slice::from_ref(push_constants));
            }

            let layout = device
                .create_pipeline_layout(&layout_info, None)
                .map_err(|err| {
                    warn!("unable to create compute pipeline layout: {err}");

                    device.destroy_shader_module(shader_module, None);

                    DriverError::Unsupported
                })?;
            let create_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage_create_info)
                .layout(layout);
            let handle = device
                .create_compute_pipelines(
                    Device::pipeline_cache(device),
                    slice::from_ref(&create_info),
                    None,
                )
                .map_err(|(_, err)| {
                    warn!("unable to create compute pipeline: {err}");

                    device.destroy_shader_module(shader_module, None);

                    DriverError::Unsupported
                })?
                .into_iter()
                .find(|handle| !handle.is_null())
                .ok_or_else(|| {
                    warn!("missing pipeline handle");

                    DriverError::Unsupported
                })?;

            device.destroy_shader_module(shader_module, None);

            Ok(ComputePipeline {
                inner: Arc::new(ComputePipelineInner {
                    descriptor_bindings,
                    descriptor_info,
                    device: device.clone(),
                    handle,
                    info,
                    layout,
                    push_constants,
                }),
            })
        }
    }

    /// The device which owns this compute pipeline.
    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    /// The native Vulkan pipeline handle of this compute pipeline.
    pub fn handle(&self) -> vk::Pipeline {
        self.inner.handle
    }

    /// Gets the information used to create this object.
    pub fn info(&self) -> ComputePipelineInfo {
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

impl Debug for ComputePipeline {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut res = f.debug_struct(stringify!(ComputePipeline));

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

impl Eq for ComputePipeline {}

impl Hash for ComputePipeline {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}

impl PartialEq for ComputePipeline {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

/// Information used to create a [`ComputePipeline`] instance.
///
/// See [`VkComputePipelineCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkComputePipelineCreateInfo.html).
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct ComputePipelineInfo {
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
    /// # vk_shader_macros::glsl!(r#"
    /// #version 460 core
    /// #extension GL_EXT_nonuniform_qualifier : require
    /// #pragma shader_stage(compute)
    ///
    /// layout(set = 0, binding = 0, rgba8) writeonly uniform image2D my_binding[];
    ///
    /// void main()
    /// {
    ///     // my_binding will have space for 8,192 images by default
    /// }
    /// # "#);
    /// ```
    #[builder(default = "8192")]
    pub bindless_descriptor_count: u32,
}

impl ComputePipelineInfo {
    /// Creates a default `ComputePipelineInfoBuilder`.
    pub fn builder() -> ComputePipelineInfoBuilder {
        Default::default()
    }

    /// Converts a `ComputePipelineInfo` into a `ComputePipelineInfoBuilder`.
    pub fn into_builder(self) -> ComputePipelineInfoBuilder {
        ComputePipelineInfoBuilder {
            bindless_descriptor_count: Some(self.bindless_descriptor_count),
        }
    }
}

impl Default for ComputePipelineInfo {
    fn default() -> Self {
        Self {
            bindless_descriptor_count: 8192,
        }
    }
}

impl From<ComputePipelineInfoBuilder> for ComputePipelineInfo {
    fn from(info: ComputePipelineInfoBuilder) -> Self {
        info.build()
    }
}

impl ComputePipelineInfoBuilder {
    /// Builds a new `ComputePipelineInfo`.
    #[inline(always)]
    pub fn build(self) -> ComputePipelineInfo {
        self.fallible_build()
            .expect("invalid compute pipeline info")
    }
}

#[derive(Debug)]
pub(crate) struct ComputePipelineInner {
    pub descriptor_bindings: DescriptorBindingMap,
    pub descriptor_info: PipelineDescriptorInfo,
    pub device: Device,
    pub handle: vk::Pipeline,
    pub info: ComputePipelineInfo,
    pub layout: vk::PipelineLayout,
    pub push_constants: Option<vk::PushConstantRange>,
}

impl Drop for ComputePipelineInner {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        Device::try_clear_private_data_object_name(
            &self.device,
            vk::ObjectType::PIPELINE,
            self.handle,
        );

        unsafe {
            self.device.destroy_pipeline(self.handle, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    type Info = ComputePipelineInfo;
    type Builder = ComputePipelineInfoBuilder;

    #[test]
    pub fn compute_pipeline_info() {
        let info = Info::default();
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn compute_pipeline_info_builder() {
        let info = Info::default();
        let builder = Builder::default().build();

        assert_eq!(info, builder);
    }
}
