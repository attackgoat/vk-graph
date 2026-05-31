//! Computing pipeline types

use {
    super::{
        DriverError,
        device::Device,
        shader::{DescriptorBindingMap, PipelineDescriptorInfo, Shader},
    },
    ash::vk,
    derive_builder::{Builder, UninitializedFieldError},
    log::{trace, warn},
    std::{
        ffi::CString,
        hash::{Hash, Hasher},
        slice,
        sync::{Arc, OnceLock},
        thread::panicking,
    },
};

/// Smart pointer handle of a pipeline object.
///
/// Also contains information about the object.
#[derive(Clone, Debug)]
pub struct ComputePipeline {
    pub(crate) inner: Arc<ComputePipelineInner>,
}

impl ComputePipeline {
    /// Creates a new compute pipeline on the given device.
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
    pub fn create(
        device: &Device,
        info: impl Into<ComputePipelineInfo>,
        shader: impl Into<Shader>,
    ) -> Result<Self, DriverError> {
        trace!("create");

        let info = info.into();
        let shader = shader.into();

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
            let entry_name =
                CString::new(shader.entry_name.as_bytes()).expect("invalid entry name");
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
                })?[0];

            device.destroy_shader_module(shader_module, None);

            Ok(ComputePipeline {
                inner: Arc::new(ComputePipelineInner {
                    descriptor_bindings,
                    descriptor_info,
                    device: device.clone(),
                    handle,
                    info,
                    layout,
                    name: Default::default(),
                    push_constants,
                }),
            })
        }
    }

    /// Gets the debugging name assigned to this pipeline, if one has been set.
    pub fn debug_name(&self) -> Option<&str> {
        self.inner.name.get().map(String::as_str)
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
    ///
    /// _Note:_ The pipeline name may only be assigned once. Subsequent calls will not update the
    /// previously set name value.
    pub fn set_debug_name(&mut self, name: impl Into<String>) {
        if !self.inner.device.physical_device.instance.info.debug {
            return;
        }

        // Both Ok and Err are valid conditions
        let _ = self.inner.name.set(name.into());
    }

    /// Sets the debugging name assigned to this pipeline.
    ///
    /// _Note:_ The pipeline name may only be assigned once. Subsequent calls will not update the
    /// previously set name value.
    pub fn with_debug_name(mut self, name: impl Into<String>) -> Self {
        self.set_debug_name(name);

        self
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
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build", error = "UninitializedFieldError"),
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
    pub name: OnceLock<String>,
    pub push_constants: Option<vk::PushConstantRange>,
}

impl Drop for ComputePipelineInner {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

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
