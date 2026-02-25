//! Shader resource types

use {
    super::{DescriptorSetLayout, DriverError, VertexInputState, device::Device},
    ash::vk,
    derive_builder::{Builder, UninitializedFieldError},
    log::{debug, error, trace, warn},
    ordered_float::OrderedFloat,
    spirq::{
        ReflectConfig,
        entry_point::EntryPoint,
        parse::SpirvBinary,
        ty::{DescriptorType, ScalarType, Type, VectorType},
        var::Variable,
    },
    std::{
        collections::{BTreeMap, HashMap},
        fmt::{Debug, Formatter},
        iter::repeat_n,
        ops::Deref,
        thread::panicking,
    },
};

pub(crate) type DescriptorBindingMap = HashMap<Descriptor, (DescriptorInfo, vk::ShaderStageFlags)>;

#[profiling::function]
fn guess_immutable_sampler(binding_name: &str) -> SamplerInfo {
    const INVALID_ERR: &str = "Invalid sampler specification";

    let (texel_filter, mipmap_mode, address_modes) = if binding_name.contains("_sampler_") {
        let spec = &binding_name[binding_name.len() - 3..];
        let texel_filter = match &spec[0..1] {
            "n" => vk::Filter::NEAREST,
            "l" => vk::Filter::LINEAR,
            _ => panic!("{INVALID_ERR}: {}", &spec[0..1]),
        };

        let mipmap_mode = match &spec[1..2] {
            "n" => vk::SamplerMipmapMode::NEAREST,
            "l" => vk::SamplerMipmapMode::LINEAR,
            _ => panic!("{INVALID_ERR}: {}", &spec[1..2]),
        };

        let address_modes = match &spec[2..3] {
            "b" => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            "e" => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            "m" => vk::SamplerAddressMode::MIRRORED_REPEAT,
            "r" => vk::SamplerAddressMode::REPEAT,
            _ => panic!("{INVALID_ERR}: {}", &spec[2..3]),
        };

        (texel_filter, mipmap_mode, address_modes)
    } else {
        debug!("image binding {binding_name} using default sampler");

        (
            vk::Filter::LINEAR,
            vk::SamplerMipmapMode::LINEAR,
            vk::SamplerAddressMode::REPEAT,
        )
    };
    let anisotropy_enable = texel_filter == vk::Filter::LINEAR;
    let mut info = SamplerInfoBuilder::default()
        .mag_filter(texel_filter)
        .min_filter(texel_filter)
        .mipmap_mode(mipmap_mode)
        .address_mode_u(address_modes)
        .address_mode_v(address_modes)
        .address_mode_w(address_modes)
        .max_lod(vk::LOD_CLAMP_NONE)
        .anisotropy_enable(anisotropy_enable);

    if anisotropy_enable {
        info = info.max_anisotropy(16.0);
    }

    info.build()
}

/// Tuple of descriptor set index and binding index.
///
/// This is a generic representation of the descriptor binding point within the shader and not a
/// bound descriptor reference.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Descriptor {
    /// Descriptor set index
    pub set: u32,

    /// Descriptor binding index
    pub binding: u32,
}

impl From<u32> for Descriptor {
    fn from(binding: u32) -> Self {
        Self { set: 0, binding }
    }
}

impl From<(u32, u32)> for Descriptor {
    fn from((set, binding): (u32, u32)) -> Self {
        Self { set, binding }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DescriptorInfo {
    AccelerationStructure(u32),
    CombinedImageSampler(u32, SamplerInfo, bool), //count, sampler, is-manually-defined?
    InputAttachment(u32, u32),                    //count, input index,
    SampledImage(u32),
    Sampler(u32, SamplerInfo, bool), //count, sampler, is-manually-defined?
    StorageBuffer(u32),
    StorageImage(u32),
    StorageTexelBuffer(u32),
    UniformBuffer(u32),
    UniformTexelBuffer(u32),
}

impl DescriptorInfo {
    pub fn binding_count(self) -> u32 {
        match self {
            Self::AccelerationStructure(binding_count) => binding_count,
            Self::CombinedImageSampler(binding_count, ..) => binding_count,
            Self::InputAttachment(binding_count, _) => binding_count,
            Self::SampledImage(binding_count) => binding_count,
            Self::Sampler(binding_count, ..) => binding_count,
            Self::StorageBuffer(binding_count) => binding_count,
            Self::StorageImage(binding_count) => binding_count,
            Self::StorageTexelBuffer(binding_count) => binding_count,
            Self::UniformBuffer(binding_count) => binding_count,
            Self::UniformTexelBuffer(binding_count) => binding_count,
        }
    }

    pub fn descriptor_type(self) -> vk::DescriptorType {
        match self {
            Self::AccelerationStructure(_) => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            Self::CombinedImageSampler(..) => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            Self::InputAttachment(..) => vk::DescriptorType::INPUT_ATTACHMENT,
            Self::SampledImage(_) => vk::DescriptorType::SAMPLED_IMAGE,
            Self::Sampler(..) => vk::DescriptorType::SAMPLER,
            Self::StorageBuffer(_) => vk::DescriptorType::STORAGE_BUFFER,
            Self::StorageImage(_) => vk::DescriptorType::STORAGE_IMAGE,
            Self::StorageTexelBuffer(_) => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
            Self::UniformBuffer(_) => vk::DescriptorType::UNIFORM_BUFFER,
            Self::UniformTexelBuffer(_) => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
        }
    }

    fn sampler_info(self) -> Option<SamplerInfo> {
        match self {
            Self::CombinedImageSampler(_, sampler_info, _) | Self::Sampler(_, sampler_info, _) => {
                Some(sampler_info)
            }
            _ => None,
        }
    }

    pub fn set_binding_count(&mut self, binding_count: u32) {
        *match self {
            Self::AccelerationStructure(binding_count) => binding_count,
            Self::CombinedImageSampler(binding_count, ..) => binding_count,
            Self::InputAttachment(binding_count, _) => binding_count,
            Self::SampledImage(binding_count) => binding_count,
            Self::Sampler(binding_count, ..) => binding_count,
            Self::StorageBuffer(binding_count) => binding_count,
            Self::StorageImage(binding_count) => binding_count,
            Self::StorageTexelBuffer(binding_count) => binding_count,
            Self::UniformBuffer(binding_count) => binding_count,
            Self::UniformTexelBuffer(binding_count) => binding_count,
        } = binding_count;
    }
}

#[derive(Debug)]
pub(crate) struct PipelineDescriptorInfo {
    pub layouts: BTreeMap<u32, DescriptorSetLayout>,
    pub pool_sizes: HashMap<u32, HashMap<vk::DescriptorType, u32>>,

    #[allow(dead_code)]
    samplers: Box<[Sampler]>,
}

impl PipelineDescriptorInfo {
    #[profiling::function]
    pub fn create(
        device: &Device,
        descriptor_bindings: &DescriptorBindingMap,
    ) -> Result<Self, DriverError> {
        let descriptor_set_count = descriptor_bindings
            .keys()
            .map(|descriptor| descriptor.set)
            .max()
            .map(|set| set + 1)
            .unwrap_or_default();
        let mut layouts = BTreeMap::new();
        let mut pool_sizes = HashMap::new();

        //trace!("descriptor_bindings: {:#?}", &descriptor_bindings);

        let mut sampler_info_binding_count = HashMap::<_, u32>::with_capacity(
            descriptor_bindings
                .values()
                .filter(|(descriptor_info, _)| descriptor_info.sampler_info().is_some())
                .count(),
        );

        for (sampler_info, binding_count) in
            descriptor_bindings
                .values()
                .filter_map(|(descriptor_info, _)| {
                    descriptor_info
                        .sampler_info()
                        .map(|sampler_info| (sampler_info, descriptor_info.binding_count()))
                })
        {
            sampler_info_binding_count
                .entry(sampler_info)
                .and_modify(|sampler_info_binding_count| {
                    *sampler_info_binding_count = binding_count.max(*sampler_info_binding_count);
                })
                .or_insert(binding_count);
        }

        let mut samplers = sampler_info_binding_count
            .keys()
            .copied()
            .map(|sampler_info| {
                Sampler::create(device, sampler_info).map(|sampler| (sampler_info, sampler))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let immutable_samplers = sampler_info_binding_count
            .iter()
            .map(|(sampler_info, &binding_count)| {
                (
                    *sampler_info,
                    repeat_n(*samplers[sampler_info], binding_count as _).collect::<Box<_>>(),
                )
            })
            .collect::<HashMap<_, _>>();

        for descriptor_set_idx in 0..descriptor_set_count {
            let mut binding_counts = HashMap::<vk::DescriptorType, u32>::new();
            let mut bindings = vec![];

            for (descriptor, (descriptor_info, stage_flags)) in descriptor_bindings
                .iter()
                .filter(|(descriptor, _)| descriptor.set == descriptor_set_idx)
            {
                let descriptor_ty = descriptor_info.descriptor_type();
                *binding_counts.entry(descriptor_ty).or_default() +=
                    descriptor_info.binding_count();
                let mut binding = vk::DescriptorSetLayoutBinding::default()
                    .binding(descriptor.binding)
                    .descriptor_count(descriptor_info.binding_count())
                    .descriptor_type(descriptor_ty)
                    .stage_flags(*stage_flags);

                if let Some(immutable_samplers) =
                    descriptor_info.sampler_info().map(|sampler_info| {
                        &immutable_samplers[&sampler_info]
                            [0..descriptor_info.binding_count() as usize]
                    })
                {
                    binding = binding.immutable_samplers(immutable_samplers);
                }

                bindings.push(binding);
            }

            let pool_size = pool_sizes
                .entry(descriptor_set_idx)
                .or_insert_with(HashMap::new);

            for (descriptor_ty, binding_count) in binding_counts.into_iter() {
                *pool_size.entry(descriptor_ty).or_default() += binding_count;
            }

            //trace!("bindings: {:#?}", &bindings);

            let mut create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

            // The bindless flags have to be created for every descriptor set layout binding.
            // [vulkan spec](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkDescriptorSetLayoutBindingFlagsCreateInfo.html)
            // Maybe using one vector and updating it would be more efficient.
            let bindless_flags = vec![vk::DescriptorBindingFlags::PARTIALLY_BOUND; bindings.len()];
            let mut bindless_flags = if device
                .physical_device
                .features_v1_2
                .descriptor_binding_partially_bound
            {
                let bindless_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                    .binding_flags(&bindless_flags);
                Some(bindless_flags)
            } else {
                None
            };

            if let Some(bindless_flags) = bindless_flags.as_mut() {
                create_info = create_info.push_next(bindless_flags);
            }

            layouts.insert(
                descriptor_set_idx,
                DescriptorSetLayout::create(device, &create_info)?,
            );
        }

        let samplers = samplers
            .drain()
            .map(|(_, sampler)| sampler)
            .collect::<Box<_>>();

        //trace!("layouts {:#?}", &layouts);
        // trace!("pool_sizes {:#?}", &pool_sizes);

        Ok(Self {
            layouts,
            pool_sizes,
            samplers,
        })
    }
}

pub(crate) struct Sampler {
    device: Device,
    sampler: vk::Sampler,
}

impl Sampler {
    #[profiling::function]
    pub fn create(device: &Device, info: impl Into<SamplerInfo>) -> Result<Self, DriverError> {
        let device = device.clone();
        let info = info.into();

        let sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .flags(info.flags)
                        .mag_filter(info.mag_filter)
                        .min_filter(info.min_filter)
                        .mipmap_mode(info.mipmap_mode)
                        .address_mode_u(info.address_mode_u)
                        .address_mode_v(info.address_mode_v)
                        .address_mode_w(info.address_mode_w)
                        .mip_lod_bias(info.mip_lod_bias.0)
                        .anisotropy_enable(info.anisotropy_enable)
                        .max_anisotropy(info.max_anisotropy.0)
                        .compare_enable(info.compare_enable)
                        .compare_op(info.compare_op)
                        .min_lod(info.min_lod.0)
                        .max_lod(info.max_lod.0)
                        .border_color(info.border_color)
                        .unnormalized_coordinates(info.unnormalized_coordinates)
                        .push_next(
                            &mut vk::SamplerReductionModeCreateInfo::default()
                                .reduction_mode(info.reduction_mode),
                        ),
                    None,
                )
                .map_err(|err| {
                    warn!("{err}");

                    match err {
                        vk::Result::ERROR_OUT_OF_HOST_MEMORY
                        | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => DriverError::OutOfMemory,
                        _ => DriverError::Unsupported,
                    }
                })?
        };

        Ok(Self { device, sampler })
    }
}

impl Debug for Sampler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.sampler)
    }
}

impl Deref for Sampler {
    type Target = vk::Sampler;

    fn deref(&self) -> &Self::Target {
        &self.sampler
    }
}

impl Drop for Sampler {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        unsafe {
            self.device.destroy_sampler(self.sampler, None);
        }
    }
}

/// Information used to create a [`vk::Sampler`] instance.
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build", error = "SamplerInfoBuilderError"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
#[non_exhaustive]
pub struct SamplerInfo {
    /// Bitmask specifying additional parameters of a sampler.
    #[builder(default)]
    pub flags: vk::SamplerCreateFlags,

    /// Specify the magnification filter to apply to texture lookups.
    ///
    /// The default value is [`vk::Filter::NEAREST`]
    #[builder(default)]
    pub mag_filter: vk::Filter,

    /// Specify the minification filter to apply to texture lookups.
    ///
    /// The default value is [`vk::Filter::NEAREST`]
    #[builder(default)]
    pub min_filter: vk::Filter,

    /// A value specifying the mipmap filter to apply to lookups.
    ///
    /// The default value is [`vk::SamplerMipmapMode::NEAREST`]
    #[builder(default)]
    pub mipmap_mode: vk::SamplerMipmapMode,

    /// A value specifying the addressing mode for U coordinates outside `[0, 1)`.
    ///
    /// The default value is [`vk::SamplerAddressMode::REPEAT`]
    #[builder(default)]
    pub address_mode_u: vk::SamplerAddressMode,

    /// A value specifying the addressing mode for V coordinates outside `[0, 1)`.
    ///
    /// The default value is [`vk::SamplerAddressMode::REPEAT`]
    #[builder(default)]
    pub address_mode_v: vk::SamplerAddressMode,

    /// A value specifying the addressing mode for W coordinates outside `[0, 1)`.
    ///
    /// The default value is [`vk::SamplerAddressMode::REPEAT`]
    #[builder(default)]
    pub address_mode_w: vk::SamplerAddressMode,

    /// The bias to be added to mipmap LOD calculation and bias provided by image sampling functions
    /// in SPIR-V, as described in the
    /// [LOD Operation](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#textures-level-of-detail-operation)
    /// section.
    #[builder(default, setter(into))]
    pub mip_lod_bias: OrderedFloat<f32>,

    /// Enables anisotropic filtering, as described in the
    /// [Texel Anisotropic Filtering](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#textures-texel-anisotropic-filtering)
    /// section
    #[builder(default)]
    pub anisotropy_enable: bool,

    /// The anisotropy value clamp used by the sampler when `anisotropy_enable` is `true`.
    ///
    /// If `anisotropy_enable` is `false`, max_anisotropy is ignored.
    #[builder(default, setter(into))]
    pub max_anisotropy: OrderedFloat<f32>,

    /// Enables comparison against a reference value during lookups.
    #[builder(default)]
    pub compare_enable: bool,

    /// Specifies the comparison operator to apply to fetched data before filtering as described in
    /// the
    /// [Depth Compare Operation](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#textures-depth-compare-operation)
    /// section.
    #[builder(default)]
    pub compare_op: vk::CompareOp,

    /// Used to clamp the
    /// [minimum of the computed LOD value](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#textures-level-of-detail-operation).
    #[builder(default, setter(into))]
    pub min_lod: OrderedFloat<f32>,

    /// Used to clamp the
    /// [maximum of the computed LOD value](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#textures-level-of-detail-operation).
    ///
    /// To avoid clamping the maximum value, set maxLod to the constant `vk::LOD_CLAMP_NONE`.
    #[builder(default, setter(into))]
    pub max_lod: OrderedFloat<f32>,

    /// Secifies the predefined border color to use.
    ///
    /// The default value is [`vk::BorderColor::FLOAT_TRANSPARENT_BLACK`]
    #[builder(default)]
    pub border_color: vk::BorderColor,

    /// Controls whether to use unnormalized or normalized texel coordinates to address texels of
    /// the image.
    ///
    /// When set to `true`, the range of the image coordinates used to lookup the texel is in the
    /// range of zero to the image size in each dimension.
    ///
    /// When set to `false` the range of image coordinates is zero to one.
    ///
    /// See
    /// [requirements](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSamplerCreateInfo.html).
    #[builder(default)]
    pub unnormalized_coordinates: bool,

    /// Specifies sampler reduction mode.
    ///
    /// Setting magnification filter ([`mag_filter`](Self::mag_filter)) to [`vk::Filter::NEAREST`]
    /// disables sampler reduction mode.
    ///
    /// The default value is [`vk::SamplerReductionMode::WEIGHTED_AVERAGE`]
    ///
    /// See
    /// [requirements](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSamplerCreateInfo.html).
    #[builder(default)]
    pub reduction_mode: vk::SamplerReductionMode,
}

impl SamplerInfo {
    /// Default sampler information with `mag_filter`, `min_filter` and `mipmap_mode` set to linear.
    pub const LINEAR: SamplerInfoBuilder = SamplerInfoBuilder {
        flags: None,
        mag_filter: Some(vk::Filter::LINEAR),
        min_filter: Some(vk::Filter::LINEAR),
        mipmap_mode: Some(vk::SamplerMipmapMode::LINEAR),
        address_mode_u: None,
        address_mode_v: None,
        address_mode_w: None,
        mip_lod_bias: None,
        anisotropy_enable: None,
        max_anisotropy: None,
        compare_enable: None,
        compare_op: None,
        min_lod: None,
        max_lod: None,
        border_color: None,
        unnormalized_coordinates: None,
        reduction_mode: None,
    };

    /// Default sampler information with `mag_filter`, `min_filter` and `mipmap_mode` set to
    /// nearest.
    pub const NEAREST: SamplerInfoBuilder = SamplerInfoBuilder {
        flags: None,
        mag_filter: Some(vk::Filter::NEAREST),
        min_filter: Some(vk::Filter::NEAREST),
        mipmap_mode: Some(vk::SamplerMipmapMode::NEAREST),
        address_mode_u: None,
        address_mode_v: None,
        address_mode_w: None,
        mip_lod_bias: None,
        anisotropy_enable: None,
        max_anisotropy: None,
        compare_enable: None,
        compare_op: None,
        min_lod: None,
        max_lod: None,
        border_color: None,
        unnormalized_coordinates: None,
        reduction_mode: None,
    };

    /// Creates a default `SamplerInfoBuilder`.
    #[allow(clippy::new_ret_no_self)]
    #[deprecated = "Use SamplerInfo::default()"]
    #[doc(hidden)]
    pub fn new() -> SamplerInfoBuilder {
        Self::default().to_builder()
    }

    /// Converts a `SamplerInfo` into a `SamplerInfoBuilder`.
    #[inline(always)]
    pub fn to_builder(self) -> SamplerInfoBuilder {
        SamplerInfoBuilder {
            flags: Some(self.flags),
            mag_filter: Some(self.mag_filter),
            min_filter: Some(self.min_filter),
            mipmap_mode: Some(self.mipmap_mode),
            address_mode_u: Some(self.address_mode_u),
            address_mode_v: Some(self.address_mode_v),
            address_mode_w: Some(self.address_mode_w),
            mip_lod_bias: Some(self.mip_lod_bias),
            anisotropy_enable: Some(self.anisotropy_enable),
            max_anisotropy: Some(self.max_anisotropy),
            compare_enable: Some(self.compare_enable),
            compare_op: Some(self.compare_op),
            min_lod: Some(self.min_lod),
            max_lod: Some(self.max_lod),
            border_color: Some(self.border_color),
            unnormalized_coordinates: Some(self.unnormalized_coordinates),
            reduction_mode: Some(self.reduction_mode),
        }
    }
}

impl Default for SamplerInfo {
    fn default() -> Self {
        Self {
            flags: vk::SamplerCreateFlags::empty(),
            mag_filter: vk::Filter::NEAREST,
            min_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            mip_lod_bias: OrderedFloat(0.0),
            anisotropy_enable: false,
            max_anisotropy: OrderedFloat(0.0),
            compare_enable: false,
            compare_op: vk::CompareOp::NEVER,
            min_lod: OrderedFloat(0.0),
            max_lod: OrderedFloat(0.0),
            border_color: vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            unnormalized_coordinates: false,
            reduction_mode: vk::SamplerReductionMode::WEIGHTED_AVERAGE,
        }
    }
}

impl SamplerInfoBuilder {
    /// Builds a new `SamplerInfo`.
    #[inline(always)]
    pub fn build(self) -> SamplerInfo {
        let res = self.fallible_build();

        #[cfg(test)]
        let res = res.unwrap();

        #[cfg(not(test))]
        let res = unsafe { res.unwrap_unchecked() };

        res
    }
}

impl From<SamplerInfoBuilder> for SamplerInfo {
    fn from(info: SamplerInfoBuilder) -> Self {
        info.build()
    }
}

#[derive(Debug)]
struct SamplerInfoBuilderError;

impl From<UninitializedFieldError> for SamplerInfoBuilderError {
    fn from(_: UninitializedFieldError) -> Self {
        Self
    }
}

/// Describes a shader program which runs on some pipeline stage.
#[allow(missing_docs)]
#[derive(Builder, Clone)]
#[builder(
    build_fn(private, name = "fallible_build", error = "ShaderBuilderError"),
    derive(Clone, Debug),
    pattern = "owned"
)]
pub struct Shader {
    /// The name of the entry point which will be executed by this shader.
    ///
    /// The default value is `main`.
    #[builder(default = "\"main\".to_owned()")]
    pub entry_name: String,

    /// Data about Vulkan specialization constants.
    ///
    /// # Examples
    ///
    /// Basic usage (GLSL):
    ///
    /// ```
    /// # vk_shader_macros::glsl!(r#"
    /// #version 460 core
    /// #pragma shader_stage(compute)
    ///
    /// // Defaults to 6 if not set using Shader specialization_info!
    /// layout(constant_id = 0) const uint MY_COUNT = 6;
    ///
    /// layout(set = 0, binding = 0) uniform sampler2D my_samplers[MY_COUNT];
    ///
    /// void main()
    /// {
    ///     // Code uses MY_COUNT number of my_samplers here
    /// }
    /// # "#);
    /// ```
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::shader::{Shader, SpecializationMap};
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::new(DeviceInfo::default())?;
    /// # let my_shader_code = [0u8; 1];
    /// // We instead specify 42 for MY_COUNT:
    /// let shader = Shader::new_fragment(my_shader_code.as_slice())
    ///     .specialization_info(SpecializationMap::new(
    ///         [vk::SpecializationMapEntry {
    ///             constant_id: 0,
    ///             offset: 0,
    ///             size: 4,
    ///         }],
    ///         42u32.to_ne_bytes()
    ///     ));
    /// # Ok(()) }
    /// ```
    #[builder(default, setter(strip_option))]
    pub specialization: Option<SpecializationMap>,

    /// Shader code.
    ///
    /// Although SPIR-V code is specified as `u32` values, this field uses `u8` in order to make
    /// loading from file simpler. You should always have a SPIR-V code length which is a multiple
    /// of four bytes, or an error will be returned during pipeline creation.
    #[builder(setter(into))]
    pub spirv: SpirvBinary,

    /// The shader stage this structure applies to.
    pub stage: vk::ShaderStageFlags,

    #[builder(private)]
    entry_point: EntryPoint,

    #[builder(default, private)]
    image_samplers: HashMap<Descriptor, SamplerInfo>,

    #[builder(default, private, setter(strip_option))]
    vertex_input_state: Option<VertexInputState>,
}

impl Shader {
    /// Specifies a shader with the given `stage` and shader code values.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(stage: vk::ShaderStageFlags, spirv: impl Into<SpirvBinary>) -> ShaderBuilder {
        ShaderBuilder::default().spirv(spirv).stage(stage)
    }

    /// Creates a new ray trace shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid or not a multiple of four bytes in length.
    pub fn new_any_hit(spirv: impl Into<SpirvBinary>) -> ShaderBuilder {
        Self::new(vk::ShaderStageFlags::ANY_HIT_KHR, spirv)
    }

    /// Creates a new ray trace shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid or not a multiple of four bytes in length.
    pub fn new_callable(spirv: impl Into<SpirvBinary>) -> ShaderBuilder {
        Self::new(vk::ShaderStageFlags::CALLABLE_KHR, spirv)
    }

    /// Creates a new ray trace shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid or not a multiple of four bytes in length.
    pub fn new_closest_hit(spirv: impl Into<SpirvBinary>) -> ShaderBuilder {
        Self::new(vk::ShaderStageFlags::CLOSEST_HIT_KHR, spirv)
    }

    /// Creates a new compute shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid or not a multiple of four bytes in length.
    pub fn new_compute(spirv: impl Into<SpirvBinary>) -> ShaderBuilder {
        Self::new(vk::ShaderStageFlags::COMPUTE, spirv)
    }

    /// Creates a new fragment shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid or not a multiple of four bytes in length.
    pub fn new_fragment(spirv: impl Into<SpirvBinary>) -> ShaderBuilder {
        Self::new(vk::ShaderStageFlags::FRAGMENT, spirv)
    }

    /// Creates a new geometry shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid or not a multiple of four bytes in length.
    pub fn new_geometry(spirv: impl Into<SpirvBinary>) -> ShaderBuilder {
        Self::new(vk::ShaderStageFlags::GEOMETRY, spirv)
    }

    /// Creates a new ray trace shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid or not a multiple of four bytes in length.
    pub fn new_intersection(spirv: impl Into<SpirvBinary>) -> ShaderBuilder {
        Self::new(vk::ShaderStageFlags::INTERSECTION_KHR, spirv)
    }

    /// Creates a new mesh shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_mesh(spirv: impl Into<SpirvBinary>) -> ShaderBuilder {
        Self::new(vk::ShaderStageFlags::MESH_EXT, spirv)
    }

    /// Creates a new ray trace shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid or not a multiple of four bytes in length.
    pub fn new_miss(spirv: impl Into<SpirvBinary>) -> ShaderBuilder {
        Self::new(vk::ShaderStageFlags::MISS_KHR, spirv)
    }

    /// Creates a new ray trace shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid or not a multiple of four bytes in length.
    pub fn new_ray_gen(spirv: impl Into<SpirvBinary>) -> ShaderBuilder {
        Self::new(vk::ShaderStageFlags::RAYGEN_KHR, spirv)
    }

    /// Creates a new mesh task shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_task(spirv: impl Into<SpirvBinary>) -> ShaderBuilder {
        Self::new(vk::ShaderStageFlags::TASK_EXT, spirv)
    }

    /// Creates a new tesselation control shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid or not a multiple of four bytes in length.
    pub fn new_tesselation_ctrl(spirv: impl Into<SpirvBinary>) -> ShaderBuilder {
        Self::new(vk::ShaderStageFlags::TESSELLATION_CONTROL, spirv)
    }

    /// Creates a new tesselation evaluation shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid or not a multiple of four bytes in length.
    pub fn new_tesselation_eval(spirv: impl Into<SpirvBinary>) -> ShaderBuilder {
        Self::new(vk::ShaderStageFlags::TESSELLATION_EVALUATION, spirv)
    }

    /// Creates a new vertex shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid or not a multiple of four bytes in length.
    pub fn new_vertex(spirv: impl Into<SpirvBinary>) -> ShaderBuilder {
        Self::new(vk::ShaderStageFlags::VERTEX, spirv)
    }

    /// Returns the input and write attachments of a shader.
    #[profiling::function]
    pub(super) fn attachments(
        &self,
    ) -> (
        impl Iterator<Item = u32> + '_,
        impl Iterator<Item = u32> + '_,
    ) {
        (
            self.entry_point.vars.iter().filter_map(|var| match var {
                Variable::Descriptor {
                    desc_ty: DescriptorType::InputAttachment(attachment),
                    ..
                } => Some(*attachment),
                _ => None,
            }),
            self.entry_point.vars.iter().filter_map(|var| match var {
                Variable::Output { location, .. } => Some(location.loc()),
                _ => None,
            }),
        )
    }

    #[profiling::function]
    pub(super) fn descriptor_bindings(&self) -> DescriptorBindingMap {
        let mut res = DescriptorBindingMap::default();

        for (name, descriptor, desc_ty, binding_count) in
            self.entry_point.vars.iter().filter_map(|var| match var {
                Variable::Descriptor {
                    name,
                    desc_bind,
                    desc_ty,
                    nbind,
                    ..
                } => Some((
                    name,
                    Descriptor {
                        set: desc_bind.set(),
                        binding: desc_bind.bind(),
                    },
                    desc_ty,
                    *nbind,
                )),
                _ => None,
            })
        {
            trace!(
                "descriptor {}: {}.{} = {:?}[{}]",
                name.as_deref().unwrap_or_default(),
                descriptor.set,
                descriptor.binding,
                *desc_ty,
                binding_count
            );

            let descriptor_info = match desc_ty {
                DescriptorType::AccelStruct() => {
                    DescriptorInfo::AccelerationStructure(binding_count)
                }
                DescriptorType::CombinedImageSampler() => {
                    let (sampler_info, is_manually_defined) =
                        self.image_sampler(descriptor, name.as_deref().unwrap_or_default());

                    DescriptorInfo::CombinedImageSampler(
                        binding_count,
                        sampler_info,
                        is_manually_defined,
                    )
                }
                DescriptorType::InputAttachment(attachment) => {
                    DescriptorInfo::InputAttachment(binding_count, *attachment)
                }
                DescriptorType::SampledImage() => DescriptorInfo::SampledImage(binding_count),
                DescriptorType::Sampler() => {
                    let (sampler_info, is_manually_defined) =
                        self.image_sampler(descriptor, name.as_deref().unwrap_or_default());

                    DescriptorInfo::Sampler(binding_count, sampler_info, is_manually_defined)
                }
                DescriptorType::StorageBuffer(_access_ty) => {
                    DescriptorInfo::StorageBuffer(binding_count)
                }
                DescriptorType::StorageImage(_access_ty) => {
                    DescriptorInfo::StorageImage(binding_count)
                }
                DescriptorType::StorageTexelBuffer(_access_ty) => {
                    DescriptorInfo::StorageTexelBuffer(binding_count)
                }
                DescriptorType::UniformBuffer() => DescriptorInfo::UniformBuffer(binding_count),
                DescriptorType::UniformTexelBuffer() => {
                    DescriptorInfo::UniformTexelBuffer(binding_count)
                }
            };
            res.insert(descriptor, (descriptor_info, self.stage));
        }

        res
    }

    fn image_sampler(&self, descriptor: Descriptor, name: &str) -> (SamplerInfo, bool) {
        self.image_samplers
            .get(&descriptor)
            .copied()
            .map(|sampler_info| (sampler_info, true))
            .unwrap_or_else(|| (guess_immutable_sampler(name), false))
    }

    #[profiling::function]
    pub(super) fn merge_descriptor_bindings(
        descriptor_bindings: impl IntoIterator<Item = DescriptorBindingMap>,
    ) -> DescriptorBindingMap {
        fn merge_info(lhs: &mut DescriptorInfo, rhs: DescriptorInfo) -> bool {
            let (lhs_count, rhs_count) = match lhs {
                DescriptorInfo::AccelerationStructure(lhs) => {
                    if let DescriptorInfo::AccelerationStructure(rhs) = rhs {
                        (lhs, rhs)
                    } else {
                        return false;
                    }
                }
                DescriptorInfo::CombinedImageSampler(lhs, lhs_sampler, lhs_is_manually_defined) => {
                    if let DescriptorInfo::CombinedImageSampler(
                        rhs,
                        rhs_sampler,
                        rhs_is_manually_defined,
                    ) = rhs
                    {
                        // Allow one of the samplers to be manually defined (only one!)
                        if *lhs_is_manually_defined && rhs_is_manually_defined {
                            return false;
                        } else if rhs_is_manually_defined {
                            *lhs_sampler = rhs_sampler;
                        }

                        (lhs, rhs)
                    } else {
                        return false;
                    }
                }
                DescriptorInfo::InputAttachment(lhs, lhs_idx) => {
                    if let DescriptorInfo::InputAttachment(rhs, rhs_idx) = rhs {
                        if *lhs_idx != rhs_idx {
                            return false;
                        }

                        (lhs, rhs)
                    } else {
                        return false;
                    }
                }
                DescriptorInfo::SampledImage(lhs) => {
                    if let DescriptorInfo::SampledImage(rhs) = rhs {
                        (lhs, rhs)
                    } else {
                        return false;
                    }
                }
                DescriptorInfo::Sampler(lhs, lhs_sampler, lhs_is_manually_defined) => {
                    if let DescriptorInfo::Sampler(rhs, rhs_sampler, rhs_is_manually_defined) = rhs
                    {
                        // Allow one of the samplers to be manually defined (only one!)
                        if *lhs_is_manually_defined && rhs_is_manually_defined {
                            return false;
                        } else if rhs_is_manually_defined {
                            *lhs_sampler = rhs_sampler;
                        }

                        (lhs, rhs)
                    } else {
                        return false;
                    }
                }
                DescriptorInfo::StorageBuffer(lhs) => {
                    if let DescriptorInfo::StorageBuffer(rhs) = rhs {
                        (lhs, rhs)
                    } else {
                        return false;
                    }
                }
                DescriptorInfo::StorageImage(lhs) => {
                    if let DescriptorInfo::StorageImage(rhs) = rhs {
                        (lhs, rhs)
                    } else {
                        return false;
                    }
                }
                DescriptorInfo::StorageTexelBuffer(lhs) => {
                    if let DescriptorInfo::StorageTexelBuffer(rhs) = rhs {
                        (lhs, rhs)
                    } else {
                        return false;
                    }
                }
                DescriptorInfo::UniformBuffer(lhs) => {
                    if let DescriptorInfo::UniformBuffer(rhs) = rhs {
                        (lhs, rhs)
                    } else {
                        return false;
                    }
                }
                DescriptorInfo::UniformTexelBuffer(lhs) => {
                    if let DescriptorInfo::UniformTexelBuffer(rhs) = rhs {
                        (lhs, rhs)
                    } else {
                        return false;
                    }
                }
            };

            *lhs_count = rhs_count.max(*lhs_count);

            true
        }

        #[profiling::function]
        fn merge_pair(src: DescriptorBindingMap, dst: &mut DescriptorBindingMap) {
            for (descriptor_binding, (descriptor_info, descriptor_flags)) in src.into_iter() {
                if let Some((existing_info, existing_flags)) = dst.get_mut(&descriptor_binding) {
                    if !merge_info(existing_info, descriptor_info) {
                        panic!("Inconsistent shader descriptors ({descriptor_binding:?})");
                    }

                    *existing_flags |= descriptor_flags;
                } else {
                    dst.insert(descriptor_binding, (descriptor_info, descriptor_flags));
                }
            }
        }

        let mut descriptor_bindings = descriptor_bindings.into_iter();
        let mut res = descriptor_bindings.next().unwrap_or_default();
        for descriptor_binding in descriptor_bindings {
            merge_pair(descriptor_binding, &mut res);
        }

        res
    }

    #[profiling::function]
    pub(super) fn push_constant_range(&self) -> Option<vk::PushConstantRange> {
        self.entry_point
            .vars
            .iter()
            .filter_map(|var| match var {
                Variable::PushConstant {
                    ty: Type::Struct(ty),
                    ..
                } => Some(ty.members.clone()),
                _ => None,
            })
            .flatten()
            .map(|push_const| {
                let offset = push_const.offset.unwrap_or_default();
                let size = push_const
                    .ty
                    .nbyte()
                    .unwrap_or_default()
                    .next_multiple_of(4);
                offset..offset + size
            })
            .reduce(|a, b| a.start.min(b.start)..a.end.max(b.end))
            .map(|push_const| vk::PushConstantRange {
                stage_flags: self.stage,
                size: (push_const.end - push_const.start) as _,
                offset: push_const.start as _,
            })
    }

    #[profiling::function]
    fn reflect_entry_point(
        entry_name: &str,
        spirv: impl Into<SpirvBinary>,
        specialization: Option<&SpecializationMap>,
    ) -> Result<EntryPoint, DriverError> {
        let mut config = ReflectConfig::new();
        config.ref_all_rscs(true).spv(spirv);

        if let Some(specialization) = specialization {
            for &vk::SpecializationMapEntry {
                constant_id,
                offset,
                size,
            } in &specialization.entries
            {
                config.specialize(
                    constant_id,
                    specialization.data[offset as usize..offset as usize + size].into(),
                );
            }
        }

        let entry_points = config.reflect().map_err(|err| {
            error!("Unable to reflect spirv: {err}");

            DriverError::InvalidData
        })?;
        let entry_point = entry_points
            .into_iter()
            .find(|entry_point| entry_point.name == entry_name)
            .ok_or_else(|| {
                error!("Entry point not found");

                DriverError::InvalidData
            })?;

        Ok(entry_point)
    }

    #[profiling::function]
    pub(super) fn vertex_input(&self) -> VertexInputState {
        // Check for manually-specified vertex layout descriptions
        if let Some(vertex_input) = &self.vertex_input_state {
            return vertex_input.clone();
        }

        fn scalar_format(ty: &ScalarType) -> vk::Format {
            match *ty {
                ScalarType::Float { bits } => match bits {
                    u8::BITS => vk::Format::R8_SNORM,
                    u16::BITS => vk::Format::R16_SFLOAT,
                    u32::BITS => vk::Format::R32_SFLOAT,
                    u64::BITS => vk::Format::R64_SFLOAT,
                    _ => unimplemented!("{bits}-bit float"),
                },
                ScalarType::Integer {
                    bits,
                    is_signed: false,
                } => match bits {
                    u8::BITS => vk::Format::R8_UINT,
                    u16::BITS => vk::Format::R16_UINT,
                    u32::BITS => vk::Format::R32_UINT,
                    u64::BITS => vk::Format::R64_UINT,
                    _ => unimplemented!("{bits}-bit unsigned integer"),
                },
                ScalarType::Integer {
                    bits,
                    is_signed: true,
                } => match bits {
                    u8::BITS => vk::Format::R8_SINT,
                    u16::BITS => vk::Format::R16_SINT,
                    u32::BITS => vk::Format::R32_SINT,
                    u64::BITS => vk::Format::R64_SINT,
                    _ => unimplemented!("{bits}-bit signed integer"),
                },
                _ => unimplemented!("{ty:?}"),
            }
        }

        fn vector_format(ty: &VectorType) -> vk::Format {
            match *ty {
                VectorType {
                    scalar_ty: ScalarType::Float { bits },
                    nscalar,
                } => match (bits, nscalar) {
                    (u8::BITS, 2) => vk::Format::R8G8_SNORM,
                    (u8::BITS, 3) => vk::Format::R8G8B8_SNORM,
                    (u8::BITS, 4) => vk::Format::R8G8B8A8_SNORM,
                    (u16::BITS, 2) => vk::Format::R16G16_SFLOAT,
                    (u16::BITS, 3) => vk::Format::R16G16B16_SFLOAT,
                    (u16::BITS, 4) => vk::Format::R16G16B16A16_SFLOAT,
                    (u32::BITS, 2) => vk::Format::R32G32_SFLOAT,
                    (u32::BITS, 3) => vk::Format::R32G32B32_SFLOAT,
                    (u32::BITS, 4) => vk::Format::R32G32B32A32_SFLOAT,
                    (u64::BITS, 2) => vk::Format::R64G64_SFLOAT,
                    (u64::BITS, 3) => vk::Format::R64G64B64_SFLOAT,
                    (u64::BITS, 4) => vk::Format::R64G64B64A64_SFLOAT,
                    _ => unimplemented!("{bits}-bit vec{nscalar} float"),
                },
                VectorType {
                    scalar_ty:
                        ScalarType::Integer {
                            bits,
                            is_signed: false,
                        },
                    nscalar,
                } => match (bits, nscalar) {
                    (u8::BITS, 2) => vk::Format::R8G8_UINT,
                    (u8::BITS, 3) => vk::Format::R8G8B8_UINT,
                    (u8::BITS, 4) => vk::Format::R8G8B8A8_UINT,
                    (u16::BITS, 2) => vk::Format::R16G16_UINT,
                    (u16::BITS, 3) => vk::Format::R16G16B16_UINT,
                    (u16::BITS, 4) => vk::Format::R16G16B16A16_UINT,
                    (u32::BITS, 2) => vk::Format::R32G32_UINT,
                    (u32::BITS, 3) => vk::Format::R32G32B32_UINT,
                    (u32::BITS, 4) => vk::Format::R32G32B32A32_UINT,
                    (u64::BITS, 2) => vk::Format::R64G64_UINT,
                    (u64::BITS, 3) => vk::Format::R64G64B64_UINT,
                    (u64::BITS, 4) => vk::Format::R64G64B64A64_UINT,
                    _ => unimplemented!("{bits}-bit vec{nscalar} unsigned integer"),
                },
                VectorType {
                    scalar_ty:
                        ScalarType::Integer {
                            bits,
                            is_signed: true,
                        },
                    nscalar,
                } => match (bits, nscalar) {
                    (u8::BITS, 2) => vk::Format::R8G8_SINT,
                    (u8::BITS, 3) => vk::Format::R8G8B8_SINT,
                    (u8::BITS, 4) => vk::Format::R8G8B8A8_SINT,
                    (u16::BITS, 2) => vk::Format::R16G16_SINT,
                    (u16::BITS, 3) => vk::Format::R16G16B16_SINT,
                    (u16::BITS, 4) => vk::Format::R16G16B16A16_SINT,
                    (u32::BITS, 2) => vk::Format::R32G32_SINT,
                    (u32::BITS, 3) => vk::Format::R32G32B32_SINT,
                    (u32::BITS, 4) => vk::Format::R32G32B32A32_SINT,
                    (u64::BITS, 2) => vk::Format::R64G64_SINT,
                    (u64::BITS, 3) => vk::Format::R64G64B64_SINT,
                    (u64::BITS, 4) => vk::Format::R64G64B64A64_SINT,
                    _ => unimplemented!("{bits}-bit vec{nscalar} signed integer"),
                },
                _ => unimplemented!("{ty:?}"),
            }
        }

        let mut input_rates_strides = HashMap::new();
        let mut vertex_attribute_descriptions = vec![];

        for (name, location, ty) in self.entry_point.vars.iter().filter_map(|var| match var {
            Variable::Input { name, location, ty } => Some((name, location, ty)),
            _ => None,
        }) {
            let (binding, guessed_rate) = name
                .as_ref()
                .filter(|name| name.contains("_ibind") || name.contains("_vbind"))
                .map(|name| {
                    let binding = name[name.rfind("bind").unwrap()..]
                        .parse()
                        .unwrap_or_default();
                    let rate = if name.contains("_ibind") {
                        vk::VertexInputRate::INSTANCE
                    } else {
                        vk::VertexInputRate::VERTEX
                    };

                    (binding, rate)
                })
                .unwrap_or_default();
            let (location, _) = location.into_inner();
            if let Some((input_rate, _)) = input_rates_strides.get(&binding) {
                assert_eq!(*input_rate, guessed_rate);
            }

            let byte_stride = ty.nbyte().unwrap_or_default() as u32;
            let (input_rate, stride) = input_rates_strides.entry(binding).or_default();
            *input_rate = guessed_rate;
            *stride += byte_stride;

            //trace!("{location} {:?} is {byte_stride} bytes", name);

            vertex_attribute_descriptions.push(vk::VertexInputAttributeDescription {
                location,
                binding,
                format: match ty {
                    Type::Scalar(ty) => scalar_format(ty),
                    Type::Vector(ty) => vector_format(ty),
                    _ => unimplemented!("{:?}", ty),
                },
                offset: byte_stride, // Figured out below - this data is iter'd in an unknown order
            });
        }

        vertex_attribute_descriptions.sort_unstable_by(|lhs, rhs| {
            let binding = lhs.binding.cmp(&rhs.binding);
            if binding.is_lt() {
                return binding;
            }

            lhs.location.cmp(&rhs.location)
        });

        let mut offset = 0;
        let mut offset_binding = 0;

        for vertex_attribute_description in &mut vertex_attribute_descriptions {
            if vertex_attribute_description.binding != offset_binding {
                offset_binding = vertex_attribute_description.binding;
                offset = 0;
            }

            let stride = vertex_attribute_description.offset;
            vertex_attribute_description.offset = offset;
            offset += stride;

            debug!(
                "vertex attribute {}.{}: {:?} (offset={})",
                vertex_attribute_description.binding,
                vertex_attribute_description.location,
                vertex_attribute_description.format,
                vertex_attribute_description.offset,
            );
        }

        let mut vertex_binding_descriptions = vec![];
        for (binding, (input_rate, stride)) in input_rates_strides.into_iter() {
            vertex_binding_descriptions.push(vk::VertexInputBindingDescription {
                binding,
                input_rate,
                stride,
            });
        }

        VertexInputState {
            vertex_attribute_descriptions,
            vertex_binding_descriptions,
        }
    }
}

impl Debug for Shader {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // We don't want the default formatter bc vec u8
        // TODO: Better output message
        f.write_str("Shader")
    }
}

impl From<ShaderBuilder> for Shader {
    fn from(shader: ShaderBuilder) -> Self {
        shader.build()
    }
}

// HACK: https://github.com/colin-kiegel/rust-derive-builder/issues/56
impl ShaderBuilder {
    /// Specifies a shader with the given `stage` and shader code values.
    pub fn new(stage: vk::ShaderStageFlags, spirv: Vec<u8>) -> Self {
        Self::default().stage(stage).spirv(spirv)
    }

    /// Builds a new `Shader`.
    pub fn build(mut self) -> Shader {
        let entry_name = self.entry_name.as_deref().unwrap_or("main");
        self.entry_point = Some(
            Shader::reflect_entry_point(
                entry_name,
                self.spirv
                    .as_ref()
                    .map(|spirv| spirv.words())
                    .expect("spirv code must be set at initialization"),
                self.specialization
                    .as_ref()
                    .map(|opt| opt.as_ref())
                    .unwrap_or_default(),
            )
            .unwrap_or_else(|_| panic!("invalid shader code for entry name \'{entry_name}\'")),
        );

        self.fallible_build()
            .expect("All required fields set at initialization")
    }

    /// Specifies a manually-defined image sampler.
    ///
    /// Sampled images, by default, use reflection to automatically assign image samplers. Each
    /// sampled image may use a suffix such as `_llr` or `_nne` for common linear/linear repeat or
    /// nearest/nearest clamp-to-edge samplers, respectively.
    ///
    /// See the [main documentation] for more information about automatic image samplers.
    ///
    /// Descriptor bindings may be specified as `(1, 2)` for descriptor set index `1` and binding
    /// index `2`, or if the descriptor set index is `0` simply specify `2` for the same case.
    ///
    /// _NOTE:_ When defining image samplers which are used in multiple stages of a single pipeline
    /// you must only call this function on one of the shader stages, it does not matter which one.
    ///
    /// # Panics
    ///
    /// Panics if two shader stages of the same pipeline define individual calls to `image_sampler`.
    ///
    /// [main documentation]: crate
    #[profiling::function]
    pub fn image_sampler(
        mut self,
        descriptor: impl Into<Descriptor>,
        info: impl Into<SamplerInfo>,
    ) -> Self {
        let descriptor = descriptor.into();
        let info = info.into();

        if self.image_samplers.is_none() {
            self.image_samplers = Some(Default::default());
        }

        self.image_samplers
            .as_mut()
            .unwrap()
            .insert(descriptor, info);

        self
    }

    /// Specifies a manually-defined vertex input layout.
    ///
    /// The vertex input layout, by default, uses reflection to automatically define vertex binding
    /// and attribute descriptions. Each vertex location is inferred to have 32-bit channels and be
    /// tightly packed in the vertex buffer. In this mode, a location with `_ibind_0` or `_vbind3`
    /// suffixes is inferred to use instance-rate on vertex buffer binding `0` or vertex rate on
    /// binding `3`, respectively.
    ///
    /// See the [main documentation] for more information about automatic vertex input layout.
    ///
    /// [main documentation]: crate
    #[profiling::function]
    pub fn vertex_input(
        mut self,
        bindings: impl Into<Vec<vk::VertexInputBindingDescription>>,
        attributes: impl Into<Vec<vk::VertexInputAttributeDescription>>,
    ) -> Self {
        self.vertex_input_state = Some(Some(VertexInputState {
            vertex_binding_descriptions: bindings.into(),
            vertex_attribute_descriptions: attributes.into(),
        }));
        self
    }
}

#[derive(Debug)]
struct ShaderBuilderError;

impl From<UninitializedFieldError> for ShaderBuilderError {
    fn from(_: UninitializedFieldError) -> Self {
        Self
    }
}

/// Describes specialized constant values.
#[derive(Clone, Debug, Default)]
pub struct SpecializationMap {
    /// A buffer of data which holds the constant values.
    pub data: Vec<u8>,

    /// Mapping of locations within the constant value data which describe each individual constant.
    pub entries: Vec<vk::SpecializationMapEntry>,
}

impl SpecializationMap {
    /// Constructs a new `SpecializationMap`.
    pub fn new(data: impl Into<Vec<u8>>) -> Self {
        Self {
            data: data.into(),
            entries: Default::default(),
        }
    }

    /// Adds a single constant offset and size to the map and returns the map for further building.
    pub fn constant(mut self, constant_id: u32, offset: u32, size: usize) -> Self {
        self.set_constant(constant_id, offset, size);
        self
    }

    /// Adds a single constant offset and size to the map and returns the map for further building.
    pub fn set_constant(&mut self, constant_id: u32, offset: u32, size: usize) {
        self.entries.push(vk::SpecializationMapEntry {
            constant_id,
            offset,
            size,
        });
    }
}

impl<'a> From<&'a SpecializationMap> for vk::SpecializationInfo<'a> {
    fn from(value: &'a SpecializationMap) -> Self {
        vk::SpecializationInfo::default()
            .map_entries(&value.entries)
            .data(&value.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Info = SamplerInfo;
    type Builder = SamplerInfoBuilder;

    #[test]
    pub fn sampler_info() {
        let info = Info::default();
        let builder = info.to_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn sampler_info_builder() {
        let info = Info::default();
        let builder = Builder::default().build();

        assert_eq!(info, builder);
    }
}
