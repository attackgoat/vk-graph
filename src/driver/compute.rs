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
    std::{ffi::CString, ops::Deref, sync::Arc, thread::panicking},
};

/// Smart pointer handle to a [pipeline] object.
///
/// Also contains information about the object.
///
/// ## `Deref` behavior
///
/// `ComputePipeline` automatically dereferences to [`vk::Pipeline`] (via the [`Deref`]
/// trait), so you can call `vk::Pipeline`'s methods on a value of type `ComputePipeline`.
///
/// [pipeline]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipeline.html
/// [deref]: core::ops::Deref
#[derive(Debug)]
#[repr(C)]
pub struct ComputePipeline {
    pub(crate) descriptor_bindings: DescriptorBindingMap,
    pub(crate) descriptor_info: PipelineDescriptorInfo,

    /// The device which owns this buffer resource.
    ///
    /// _Note:_ This field is read-only.
    #[cfg(doc)]
    pub device: Arc<Device>,

    #[cfg(not(doc))]
    device: Arc<Device>,

    pub(crate) layout: vk::PipelineLayout,

    /// The native Vulkan resource handle of this pipeline.
    ///
    /// _Note:_ This field is read-only.
    #[cfg(doc)]
    pub handle: vk::Pipeline,

    #[cfg(not(doc))]
    handle: vk::Pipeline,

    /// Information used to create this object.
    #[cfg(doc)]
    pub info: ComputePipelineInfo,

    #[cfg(not(doc))]
    info: ComputePipelineInfo,

    /// A descriptive name used in debugging messages.
    pub name: Option<String>,

    pub(crate) push_constants: Option<vk::PushConstantRange>,
}

#[doc(hidden)]
#[repr(C)]
pub struct ComputePipelineRef {
    pub(crate) descriptor_bindings: DescriptorBindingMap,
    pub(crate) descriptor_info: PipelineDescriptorInfo,
    pub device: Arc<Device>,
    pub(crate) layout: vk::PipelineLayout,
    pub handle: vk::Pipeline,
    pub info: ComputePipelineInfo,
    pub name: Option<String>,
    pub(crate) push_constants: Option<vk::PushConstantRange>,
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
    /// # let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
    /// # let my_shader_code = [0u8; 1];
    /// // my_shader_code is raw SPIR-V code as bytes
    /// let shader = Shader::new_compute(my_shader_code.as_slice());
    /// let pipeline = ComputePipeline::create(&device, ComputePipelineInfo::default(), shader)?;
    ///
    /// assert_ne!(pipeline.handle, vk::Pipeline::null());
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn create(
        device: &Arc<Device>,
        info: impl Into<ComputePipelineInfo>,
        shader: impl Into<Shader>,
    ) -> Result<Self, DriverError> {
        use std::slice::from_ref;

        trace!("create");

        let device = Arc::clone(device);
        let info: ComputePipelineInfo = info.into();
        let shader = shader.into();

        // Use SPIR-V reflection to get the types and counts of all descriptors
        let mut descriptor_bindings = shader.descriptor_bindings();
        for (descriptor_info, _) in descriptor_bindings.values_mut() {
            if descriptor_info.binding_count() == 0 {
                descriptor_info.set_binding_count(info.bindless_descriptor_count);
            }
        }

        let descriptor_info = PipelineDescriptorInfo::create(&device, &descriptor_bindings)?;
        let descriptor_set_layouts = descriptor_info
            .layouts
            .values()
            .map(|descriptor_set_layout| **descriptor_set_layout)
            .collect::<Box<[_]>>();

        unsafe {
            let shader_module = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(shader.spirv.words()),
                    None,
                )
                .map_err(|err| {
                    warn!("{err}");

                    DriverError::Unsupported
                })?;
            let entry_name = CString::new(shader.entry_name.as_bytes()).unwrap();
            let mut stage_create_info = vk::PipelineShaderStageCreateInfo::default()
                .module(shader_module)
                .stage(shader.stage)
                .name(&entry_name);
            let specialization_info = shader.specialization_info.as_ref().map(|info| {
                vk::SpecializationInfo::default()
                    .map_entries(&info.map_entries)
                    .data(&info.data)
            });

            if let Some(specialization_info) = &specialization_info {
                stage_create_info = stage_create_info.specialization_info(specialization_info);
            }

            let mut layout_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);

            let push_constants = shader.push_constant_range();
            if let Some(push_constants) = &push_constants {
                layout_info = layout_info.push_constant_ranges(from_ref(push_constants));
            }

            let layout = device
                .create_pipeline_layout(&layout_info, None)
                .map_err(|err| {
                    warn!("{err}");

                    device.destroy_shader_module(shader_module, None);

                    DriverError::Unsupported
                })?;
            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage_create_info)
                .layout(layout);
            let handle = device
                .create_compute_pipelines(
                    Device::pipeline_cache(&device),
                    from_ref(&pipeline_info),
                    None,
                )
                .map_err(|(_, err)| {
                    warn!("{err}");

                    device.destroy_shader_module(shader_module, None);

                    DriverError::Unsupported
                })?[0];

            device.destroy_shader_module(shader_module, None);

            Ok(ComputePipeline {
                descriptor_bindings,
                descriptor_info,
                device,
                handle,
                info,
                layout,
                name: None,
                push_constants,
            })
        }
    }

    /// Sets the debugging name assigned to this pipeline.
    pub fn with_name(mut this: Self, name: impl Into<String>) -> Self {
        this.name = Some(name.into());
        this
    }
}

#[doc(hidden)]
impl Deref for ComputePipeline {
    type Target = ComputePipelineRef;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self as *const Self::Target) }
    }
}

impl Drop for ComputePipeline {
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

/// Information used to create a [`ComputePipeline`] instance.
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(
        private,
        name = "fallible_build",
        error = "ComputePipelineInfoBuilderError"
    ),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
#[non_exhaustive]
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
    /// Converts a `ComputePipelineInfo` into a `ComputePipelineInfoBuilder`.
    #[inline(always)]
    pub fn to_builder(self) -> ComputePipelineInfoBuilder {
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
        let res = self.fallible_build();

        #[cfg(test)]
        let res = res.unwrap();

        #[cfg(not(test))]
        let res = unsafe { res.unwrap_unchecked() };

        res
    }
}

#[derive(Debug)]
struct ComputePipelineInfoBuilderError;

impl From<UninitializedFieldError> for ComputePipelineInfoBuilderError {
    fn from(_: UninitializedFieldError) -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Info = ComputePipelineInfo;
    type Builder = ComputePipelineInfoBuilder;

    #[test]
    pub fn compute_pipeline_info() {
        let info = Info::default();
        let builder = info.to_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn compute_pipeline_info_builder() {
        let info = Info::default();
        let builder = Builder::default().build();

        assert_eq!(info, builder);
    }
}
