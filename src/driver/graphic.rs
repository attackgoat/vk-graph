//! Graphics pipeline types

use {
    super::{
        DriverError,
        device::Device,
        image::SampleCount,
        merge_push_constant_ranges,
        shader::{DescriptorBindingMap, PipelineDescriptorInfo, Shader, SpecializationMap},
    },
    ash::vk,
    derive_builder::Builder,
    log::{Level::Trace, log_enabled, trace, warn},
    ordered_float::OrderedFloat,
    std::{
        collections::HashSet,
        ffi::CString,
        hash::{Hash, Hasher},
        sync::{Arc, OnceLock},
        thread::panicking,
    },
};

const RGBA_COLOR_COMPONENTS: vk::ColorComponentFlags = vk::ColorComponentFlags::from_raw(
    vk::ColorComponentFlags::R.as_raw()
        | vk::ColorComponentFlags::G.as_raw()
        | vk::ColorComponentFlags::B.as_raw()
        | vk::ColorComponentFlags::A.as_raw(),
);

/// Specifies color blend state used when rasterization is enabled for any color attachments
/// accessed during rendering.
///
/// See [`VkPipelineColorBlendAttachmentState`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineColorBlendAttachmentState.html).
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct BlendInfo {
    /// Controls whether blending is enabled for the corresponding color attachment.
    ///
    /// If blending is not enabled, the source fragment’s color for that attachment is passed
    /// through unmodified.
    #[builder(default = "false")]
    pub blend_enable: bool,

    /// Selects which blend factor is used to determine the source factors.
    #[builder(default = "vk::BlendFactor::SRC_COLOR")]
    pub src_color_blend_factor: vk::BlendFactor,

    /// Selects which blend factor is used to determine the destination factors.
    #[builder(default = "vk::BlendFactor::ONE_MINUS_DST_COLOR")]
    pub dst_color_blend_factor: vk::BlendFactor,

    /// Selects which blend operation is used to calculate the RGB values to write to the color
    /// attachment.
    #[builder(default = "vk::BlendOp::ADD")]
    pub color_blend_op: vk::BlendOp,

    /// Selects which blend factor is used to determine the source factor.
    #[builder(default = "vk::BlendFactor::ZERO")]
    pub src_alpha_blend_factor: vk::BlendFactor,

    /// Selects which blend factor is used to determine the destination factor.
    #[builder(default = "vk::BlendFactor::ZERO")]
    pub dst_alpha_blend_factor: vk::BlendFactor,

    /// Selects which blend operation is used to calculate the alpha values to write to the color
    /// attachment.
    #[builder(default = "vk::BlendOp::ADD")]
    pub alpha_blend_op: vk::BlendOp,

    /// A bitmask of specifying which of the R, G, B, and/or A components are enabled for writing,
    /// as described for [`VkPipelineColorBlendAttachmentState`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineColorBlendAttachmentState.html).
    #[builder(default = "RGBA_COLOR_COMPONENTS")]
    pub color_write_mask: vk::ColorComponentFlags,
}

impl BlendInfo {
    /// A commonly used blend mode for replacing color attachment values with new ones.
    pub const REPLACE: Self = Self {
        blend_enable: false,
        src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ZERO,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: RGBA_COLOR_COMPONENTS,
    };

    /// A commonly used blend mode for blending color attachment values based on the alpha channel.
    pub const ALPHA: Self = Self {
        blend_enable: true,
        src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::SRC_ALPHA,
        dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: RGBA_COLOR_COMPONENTS,
    };

    /// A commonly used blend mode for blending color attachment values based on the alpha channel,
    /// where the color components have been pre-multiplied with the alpha component value.
    pub const PRE_MULTIPLIED_ALPHA: Self = Self {
        blend_enable: true,
        src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ONE,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: RGBA_COLOR_COMPONENTS,
    };

    /// Specifies a default blend mode which is not enabled.
    pub fn builder() -> BlendInfoBuilder {
        BlendInfoBuilder::default()
    }

    /// Converts a `BlendInfo` into a `BlendInfoBuilder`.
    pub fn into_builder(self) -> BlendInfoBuilder {
        BlendInfoBuilder {
            blend_enable: Some(self.blend_enable),
            src_color_blend_factor: Some(self.src_color_blend_factor),
            dst_color_blend_factor: Some(self.dst_color_blend_factor),
            color_blend_op: Some(self.color_blend_op),
            src_alpha_blend_factor: Some(self.src_alpha_blend_factor),
            dst_alpha_blend_factor: Some(self.dst_alpha_blend_factor),
            alpha_blend_op: Some(self.alpha_blend_op),
            color_write_mask: Some(self.color_write_mask),
        }
    }
}

// the Builder derive Macro wants Default to be implemented for BlendMode
impl Default for BlendInfo {
    fn default() -> Self {
        Self::REPLACE
    }
}

impl From<BlendInfo> for vk::PipelineColorBlendAttachmentState {
    fn from(mode: BlendInfo) -> Self {
        Self {
            blend_enable: mode.blend_enable as _,
            src_color_blend_factor: mode.src_color_blend_factor,
            dst_color_blend_factor: mode.dst_color_blend_factor,
            color_blend_op: mode.color_blend_op,
            src_alpha_blend_factor: mode.src_alpha_blend_factor,
            dst_alpha_blend_factor: mode.dst_alpha_blend_factor,
            alpha_blend_op: mode.alpha_blend_op,
            color_write_mask: mode.color_write_mask,
        }
    }
}

impl BlendInfoBuilder {
    /// Builds a new `BlendMode`.
    pub fn build(self) -> BlendInfo {
        self.fallible_build().expect("invalid blend info")
    }
}

// TODO: This could be simplified (bounds_test controsl min/max etc)
/// Specifies the [depth bounds tests], [stencil test], and [depth test] pipeline state.
///
/// See [`VkPipelineDepthStencilStateCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineDepthStencilStateCreateInfo.html).
#[derive(Builder, Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct DepthStencilInfo {
    /// Control parameters of the stencil test.
    #[builder(default)]
    pub back: StencilMode,

    /// Controls whether [depth bounds testing] is enabled.
    ///
    /// See [`VkPipelineMultisampleStateCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineMultisampleStateCreateInfo.html).
    #[builder(default)]
    pub bounds_test: bool,

    /// A value specifying the comparison operator to use in the [depth comparison] step of the
    /// [depth test].
    ///
    /// See [`VkPipelineDepthStencilStateCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineDepthStencilStateCreateInfo.html).
    #[builder(default)]
    pub compare_op: vk::CompareOp,

    /// Controls whether [depth testing] is enabled.
    ///
    /// See [`VkPipelineDepthStencilStateCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineDepthStencilStateCreateInfo.html).
    #[builder(default)]
    pub depth_test: bool,

    /// Controls whether [depth writes] are enabled when `depth_test` is `true`.
    ///
    /// Depth writes are always disabled when `depth_test` is `false`.
    ///
    /// See [`VkPipelineDepthStencilStateCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineDepthStencilStateCreateInfo.html).
    #[builder(default)]
    pub depth_write: bool,

    /// Control parameters of the stencil test.
    #[builder(default)]
    pub front: StencilMode,

    // Note: Using setter(into) so caller does not need our version of OrderedFloat
    /// Minimum depth bound used in the [depth bounds test].
    ///
    /// See [`VkPipelineDepthStencilStateCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineDepthStencilStateCreateInfo.html).
    #[builder(default, setter(into))]
    pub min: OrderedFloat<f32>,

    // Note: Using setter(into) so caller does not need our version of OrderedFloat
    /// Maximum depth bound used in the [depth bounds test].
    ///
    /// See [`VkPipelineDepthStencilStateCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineDepthStencilStateCreateInfo.html).
    #[builder(default, setter(into))]
    pub max: OrderedFloat<f32>,

    /// Controls whether [stencil testing] is enabled.
    ///
    /// See [`VkPipelineDepthStencilStateCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineDepthStencilStateCreateInfo.html).
    #[builder(default)]
    pub stencil_test: bool,
}

impl DepthStencilInfo {
    /// A commonly used depth/stencil mode
    pub const DEPTH_WRITE_LESS_IGNORE_STENCIL: Self = Self {
        back: StencilMode::IGNORE,
        bounds_test: true,
        compare_op: vk::CompareOp::LESS,
        depth_test: true,
        depth_write: true,
        front: StencilMode::IGNORE,
        min: OrderedFloat(0.0),
        max: OrderedFloat(1.0),
        stencil_test: false,
    };

    /// Specifies a no-depth/no-stencil mode.
    ///
    /// This is the default state.
    pub const IGNORE: Self = Self {
        back: StencilMode::IGNORE,
        bounds_test: false,
        compare_op: vk::CompareOp::NEVER,
        depth_test: false,
        depth_write: false,
        front: StencilMode::IGNORE,
        min: OrderedFloat(0.0),
        max: OrderedFloat(0.0),
        stencil_test: false,
    };

    /// Creates a default `DepthStencilInfoBuilder`.
    pub fn builder() -> DepthStencilInfoBuilder {
        Default::default()
    }

    /// Converts a `DepthStencilInfo` into a `DepthStencilInfoBuilder`.
    pub fn into_builder(self) -> DepthStencilInfoBuilder {
        DepthStencilInfoBuilder {
            back: Some(self.back),
            bounds_test: Some(self.bounds_test),
            compare_op: Some(self.compare_op),
            depth_test: Some(self.depth_test),
            depth_write: Some(self.depth_write),
            front: Some(self.front),
            max: Some(self.max),
            min: Some(self.min),
            stencil_test: Some(self.stencil_test),
        }
    }
}

impl From<DepthStencilInfo> for vk::PipelineDepthStencilStateCreateInfo<'_> {
    fn from(info: DepthStencilInfo) -> Self {
        Self::default()
            .back(info.back.into())
            .depth_bounds_test_enable(info.bounds_test as _)
            .depth_compare_op(info.compare_op)
            .depth_test_enable(info.depth_test as _)
            .depth_write_enable(info.depth_write as _)
            .front(info.front.into())
            .max_depth_bounds(info.max.into_inner())
            .min_depth_bounds(info.min.into_inner())
            .stencil_test_enable(info.stencil_test as _)
    }
}

impl DepthStencilInfoBuilder {
    /// Builds a new `DepthStencilInfo`.
    pub fn build(self) -> DepthStencilInfo {
        self.fallible_build().expect("invalid depth stencil info")
    }
}

/// Opaque representation of a pipeline object.
///
/// Also contains information about the object.
///
/// [pipeline]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipeline.html
#[derive(Clone, Debug)]
#[read_only::cast]
pub struct GraphicsPipeline {
    pub(crate) inner: Arc<GraphicsPipelineInner>,
}

impl GraphicsPipeline {
    /// Creates a new graphics pipeline on the given device.
    ///
    /// The correct pipeline stages will be enabled based on the provided shaders. See [Shader] for
    /// details on all available stages.
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
    /// # use vk_graph::driver::graphic::{GraphicsPipeline, GraphicsPipelineInfo};
    /// # use vk_graph::driver::shader::Shader;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// # let my_frag_code = [0u8; 1];
    /// # let my_vert_code = [0u8; 1];
    /// // shader code is raw SPIR-V code as bytes
    /// let vert = Shader::new_vertex(my_vert_code.as_slice());
    /// let frag = Shader::new_fragment(my_frag_code.as_slice());
    /// let info = GraphicsPipelineInfo::default();
    /// let pipeline = GraphicsPipeline::create(&device, info, [vert, frag])?;
    ///
    /// assert_eq!(pipeline.info().front_face, vk::FrontFace::COUNTER_CLOCKWISE);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn create<S>(
        device: &Device,
        info: impl Into<GraphicsPipelineInfo>,
        shaders: impl IntoIterator<Item = S>,
    ) -> Result<Self, DriverError>
    where
        S: TryInto<Shader>,
        S::Error: Into<DriverError>,
    {
        trace!("create");

        let device = device.clone();
        let info = info.into();
        let shaders = shaders
            .into_iter()
            .map(|shader| shader.try_into().map_err(Into::into))
            .collect::<Result<Vec<_>, _>>()?;

        let vertex_input = shaders
            .iter()
            .find(|shader| shader.stage == vk::ShaderStageFlags::VERTEX)
            .ok_or(DriverError::InvalidData)?
            .try_vertex_input()?;

        // Check for proper stages because vulkan may not complain but this is bad
        let has_fragment_stage = shaders
            .iter()
            .any(|shader| shader.stage.contains(vk::ShaderStageFlags::FRAGMENT));
        let has_tesselation_stage = shaders.iter().any(|shader| {
            shader
                .stage
                .contains(vk::ShaderStageFlags::TESSELLATION_CONTROL)
        }) && shaders.iter().any(|shader| {
            shader
                .stage
                .contains(vk::ShaderStageFlags::TESSELLATION_EVALUATION)
        });
        let has_geometry_stage = shaders
            .iter()
            .any(|shader| shader.stage.contains(vk::ShaderStageFlags::GEOMETRY));

        debug_assert!(
            has_fragment_stage || has_tesselation_stage || has_geometry_stage,
            "invalid shader stage combination"
        );

        let mut descriptor_bindings = Shader::merge_descriptor_bindings(
            shaders.iter().map(|shader| shader.descriptor_bindings()),
        )?;
        for (descriptor_info, _) in descriptor_bindings.values_mut() {
            if descriptor_info.binding_count() == 0 {
                descriptor_info.set_binding_count(info.bindless_descriptor_count);
            }
        }

        let descriptor_info = PipelineDescriptorInfo::create(&device, &descriptor_bindings)?;
        let descriptor_sets_layouts = descriptor_info
            .layouts
            .values()
            .map(|descriptor_set_layout| descriptor_set_layout.handle)
            .collect::<Box<_>>();

        let push_constants = shaders
            .iter()
            .map(|shader| shader.push_constant_range())
            .filter_map(|mut push_const| push_const.take())
            .collect::<Vec<_>>();

        let input_attachments = shaders
            .iter()
            .find(|shader| shader.stage == vk::ShaderStageFlags::FRAGMENT)
            .map(|shader| {
                let (input, write) = shader.attachments();
                let (input, write) = (
                    input
                        .collect::<HashSet<_>>()
                        .into_iter()
                        .collect::<Box<_>>(),
                    write.collect::<HashSet<_>>(),
                );

                if log_enabled!(Trace) {
                    for input in input.iter() {
                        trace!("detected input attachment {input}");
                    }

                    for write in &write {
                        trace!("detected write attachment {write}");
                    }
                }

                input
            })
            .unwrap_or_default();

        unsafe {
            let layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&descriptor_sets_layouts)
                        .push_constant_ranges(&push_constants),
                    None,
                )
                .map_err(|err| {
                    warn!("unable to create graphics pipeline layout: {err}");

                    DriverError::Unsupported
                })?;
            let shader_stages = shaders
                .into_iter()
                .map(|shader| {
                    let shader_module = device
                        .create_shader_module(
                            &vk::ShaderModuleCreateInfo::default().code(shader.spirv.words()),
                            None,
                        )
                        .map_err(|err| {
                            warn!("unable to create graphic shader module: {err}");

                            DriverError::Unsupported
                        })?;
                    let shader_stage = ShaderStage {
                        flags: shader.stage,
                        module: shader_module,
                        name: CString::new(shader.entry_name.as_str()).map_err(|err| {
                            warn!("invalid graphics shader entry name: {err}");

                            DriverError::InvalidData
                        })?,
                        specialization: shader.specialization,
                    };

                    Result::<_, DriverError>::Ok(shader_stage)
                })
                .collect::<Result<Box<_>, _>>()?;

            let mut multisample = MultisampleState {
                alpha_to_coverage_enable: info.alpha_to_coverage,
                alpha_to_one_enable: info.alpha_to_one,
                rasterization_samples: info.samples,
                ..Default::default()
            };

            if let Some(OrderedFloat(min_sample_shading)) = info.min_sample_shading {
                #[cfg(debug_assertions)]
                if info.samples.is_single() {
                    // This combination of a single-sampled pipeline and minimum sample shading
                    // does not make sense and should not be requested. In the future maybe this is
                    // part of the MSAA value so it can't be specified.
                    warn!("unsupported sample rate shading of single-sample pipeline");
                }

                // Callers should check this before attempting to use the feature
                debug_assert!(
                    device.physical_device.features_v1_0.sample_rate_shading,
                    "unsupported sample rate shading feature"
                );

                multisample.sample_shading_enable = true;
                multisample.min_sample_shading = min_sample_shading;
            }

            let push_constants = merge_push_constant_ranges(&push_constants).into_boxed_slice();

            Ok(Self {
                inner: Arc::new(GraphicsPipelineInner {
                    descriptor_bindings,
                    descriptor_info,
                    device,
                    info,
                    input_attachments,
                    layout,
                    multisample,
                    name: Default::default(),
                    push_constants,
                    shader_stages,
                    vertex_input,
                }),
            })
        }
    }

    /// Gets the debugging name assigned to this pipeline, if one has been set.
    pub fn debug_name(&self) -> Option<&str> {
        self.inner.name.get().map(String::as_str)
    }

    /// The device which owns this graphics pipeline.
    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    /// Gets the information used to create this object.
    pub fn info(&self) -> GraphicsPipelineInfo {
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

impl Eq for GraphicsPipeline {}

impl Hash for GraphicsPipeline {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}

impl PartialEq for GraphicsPipeline {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

/// Information used to create a [`GraphicsPipeline`] instance.
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct GraphicsPipelineInfo {
    /// Controls whether a temporary coverage value is generated based on the alpha component of
    /// the fragment’s first color output.
    #[builder(default)]
    pub alpha_to_coverage: bool,

    /// Controls whether the alpha component of the fragment’s first color output is replaced with
    /// one.
    #[builder(default)]
    pub alpha_to_one: bool,

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
    /// #pragma shader_stage(fragment)
    ///
    /// layout(set = 0, binding = 0) uniform sampler2D my_binding[];
    ///
    /// void main() {
    ///     // my_binding will have space for 8,192 images by default
    /// }
    /// # "#);
    /// ```
    #[builder(default = "8192")]
    pub bindless_descriptor_count: u32,

    /// Specifies color blend state used when rasterization is enabled for any color attachments
    /// accessed during rendering.
    ///
    /// The default value is [`BlendInfo::REPLACE`].
    #[builder(default)]
    pub blend: BlendInfo,

    /// Bitmask controlling triangle culling.
    ///
    /// The default value is `vk::CullModeFlags::BACK`.
    #[builder(default = "vk::CullModeFlags::BACK")]
    pub cull_mode: vk::CullModeFlags,

    /// Interpret polygon front-facing orientation.
    ///
    /// The default value is `vk::FrontFace::COUNTER_CLOCKWISE`.
    #[builder(default = "vk::FrontFace::COUNTER_CLOCKWISE")]
    pub front_face: vk::FrontFace,

    /// Specify a fraction of the minimum number of unique samples to process for each fragment.
    #[builder(default, setter(into, strip_option))]
    pub min_sample_shading: Option<OrderedFloat<f32>>,

    /// Control polygon rasterization mode.
    ///
    /// The default value is `vk::PolygonMode::FILL`.
    #[builder(default = "vk::PolygonMode::FILL")]
    pub polygon_mode: vk::PolygonMode,

    /// Input primitive topology.
    ///
    /// The default value is `vk::PrimitiveTopology::TRIANGLE_LIST`.
    #[builder(default = "vk::PrimitiveTopology::TRIANGLE_LIST")]
    pub topology: vk::PrimitiveTopology,

    /// Multisampling antialias mode.
    ///
    /// The default value is `SampleCount::Type1`.
    ///
    /// See [`VkPipelineMultisampleStateCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineMultisampleStateCreateInfo.html).
    #[builder(default = "SampleCount::Type1")]
    pub samples: SampleCount,
}

impl GraphicsPipelineInfo {
    /// Creates a default `GraphicsPipelineInfoBuilder`.
    pub fn builder() -> GraphicsPipelineInfoBuilder {
        Default::default()
    }

    /// Converts a `GraphicsPipelineInfo` into a `GraphicsPipelineInfoBuilder`.
    pub fn into_builder(self) -> GraphicsPipelineInfoBuilder {
        GraphicsPipelineInfoBuilder {
            alpha_to_coverage: Some(self.alpha_to_coverage),
            alpha_to_one: Some(self.alpha_to_one),
            bindless_descriptor_count: Some(self.bindless_descriptor_count),
            blend: Some(self.blend),
            cull_mode: Some(self.cull_mode),
            front_face: Some(self.front_face),
            min_sample_shading: Some(self.min_sample_shading),
            polygon_mode: Some(self.polygon_mode),
            topology: Some(self.topology),
            samples: Some(self.samples),
        }
    }
}

impl Default for GraphicsPipelineInfo {
    fn default() -> Self {
        Self {
            alpha_to_coverage: false,
            alpha_to_one: false,
            bindless_descriptor_count: 8192,
            blend: BlendInfo::REPLACE,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            min_sample_shading: None,
            polygon_mode: vk::PolygonMode::FILL,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            samples: SampleCount::Type1,
        }
    }
}

impl From<GraphicsPipelineInfoBuilder> for GraphicsPipelineInfo {
    fn from(info: GraphicsPipelineInfoBuilder) -> Self {
        info.build()
    }
}

impl GraphicsPipelineInfoBuilder {
    /// Builds a new `GraphicsPipelineInfo`.
    #[inline(always)]
    pub fn build(self) -> GraphicsPipelineInfo {
        self.fallible_build()
            .expect("invalid graphics pipeline info")
    }
}

#[derive(Debug)]
pub(crate) struct GraphicsPipelineInner {
    pub descriptor_bindings: DescriptorBindingMap,
    pub descriptor_info: PipelineDescriptorInfo,
    pub device: Device,
    pub info: GraphicsPipelineInfo,
    pub input_attachments: Box<[u32]>,
    pub layout: vk::PipelineLayout,
    pub multisample: MultisampleState,
    pub name: OnceLock<String>,
    pub push_constants: Box<[vk::PushConstantRange]>,
    pub shader_stages: Box<[ShaderStage]>,
    pub vertex_input: VertexInputState,
}

impl Drop for GraphicsPipelineInner {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        unsafe {
            self.device.destroy_pipeline_layout(self.layout, None);
        }

        for shader_stage in &mut self.shader_stages {
            unsafe {
                self.device.destroy_shader_module(shader_stage.module, None);
            }
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct MultisampleState {
    pub alpha_to_coverage_enable: bool,
    pub alpha_to_one_enable: bool,
    pub flags: vk::PipelineMultisampleStateCreateFlags,
    pub min_sample_shading: f32,
    pub rasterization_samples: SampleCount,
    pub sample_mask: Vec<u32>,
    pub sample_shading_enable: bool,
}

#[derive(Debug)]
pub(crate) struct ShaderStage {
    pub flags: vk::ShaderStageFlags,
    pub module: vk::ShaderModule,
    pub name: CString, // TODO
    pub specialization: Option<SpecializationMap>,
}

/// Specifies stencil mode during rasterization.
///
/// See [`VkStencilOpState`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkStencilOpState.html).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StencilMode {
    /// The action performed on samples that fail the stencil test.
    pub fail_op: vk::StencilOp,

    /// The action performed on samples that pass both the depth and stencil tests.
    pub pass_op: vk::StencilOp,

    /// The action performed on samples that pass the stencil test and fail the depth test.
    pub depth_fail_op: vk::StencilOp,

    /// The comparison operator used in the stencil test.
    pub compare_op: vk::CompareOp,

    /// The bits of the unsigned integer stencil values participating in the stencil test.
    pub compare_mask: u32,

    /// The bits of the unsigned integer stencil values updated by the stencil test in the stencil
    /// framebuffer attachment.
    pub write_mask: u32,

    /// An unsigned integer stencil reference value that is used in the unsigned stencil
    /// comparison.
    pub reference: u32,
}

impl StencilMode {
    /// Specifes a stencil mode which is has no effect.
    pub const IGNORE: Self = Self {
        fail_op: vk::StencilOp::KEEP,
        pass_op: vk::StencilOp::KEEP,
        depth_fail_op: vk::StencilOp::KEEP,
        compare_op: vk::CompareOp::NEVER,
        compare_mask: 0,
        write_mask: 0,
        reference: 0,
    };
}

impl Default for StencilMode {
    fn default() -> Self {
        Self::IGNORE
    }
}

impl From<StencilMode> for vk::StencilOpState {
    fn from(mode: StencilMode) -> Self {
        Self {
            fail_op: mode.fail_op,
            pass_op: mode.pass_op,
            depth_fail_op: mode.depth_fail_op,
            compare_op: mode.compare_op,
            compare_mask: mode.compare_mask,
            write_mask: mode.write_mask,
            reference: mode.reference,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct VertexInputState {
    pub vertex_binding_descriptions: Vec<vk::VertexInputBindingDescription>,
    pub vertex_attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn blend_info() {
        let info = BlendInfo::default();
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn blend_info_builder() {
        let info = BlendInfo::default();
        let builder = BlendInfoBuilder::default().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn depth_stencil_info() {
        let info = DepthStencilInfo::default();
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn depth_stencil_info_builder() {
        let info = DepthStencilInfo::default();
        let builder = DepthStencilInfoBuilder::default().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn graphics_pipeline_info() {
        let info = GraphicsPipelineInfo::default();
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn graphics_pipeline_info_builder() {
        let info = GraphicsPipelineInfo::default();
        let builder = GraphicsPipelineInfoBuilder::default().build();

        assert_eq!(info, builder);
    }
}
