//! TODO

pub use shaderc::{OptimizationLevel, SourceLanguage, SpirvVersion};

use {
    super::{compile_shader, guess_shader_source_language},
    derive_builder::{Builder, UninitializedFieldError},
    log::{debug, error},
    notify::{RecommendedWatcher, RecursiveMode, Watcher},
    shaderc::{CompileOptions, EnvVersion, ShaderKind, TargetEnv},
    std::path::{Path, PathBuf},
    vk_graph::driver::{ash::vk, shader::SpecializationInfo, DriverError},
};

/// Describes a shader program which runs on some pipeline stage.
///
/// _NOTE:_ When compiled on Apple platforms the macro `MOLTEN_VK` will be defined automatically.
/// This may be used to handle any differences introduced by SPIRV-Cross translation to Metal
/// Shading Language (MSL) at runtime.
#[allow(missing_docs)]
#[derive(Builder, Clone, Debug)]
#[builder(
    build_fn(private, name = "fallible_build", error = "HotShaderBuilderError"),
    derive(Clone, Debug),
    pattern = "owned"
)]
pub struct HotShader {
    /// The name of the entry point which will be executed by this shader.
    ///
    /// The default value is `main`.
    #[builder(default = "\"main\".to_owned()")]
    pub entry_name: String,

    /// Macro definitions.
    #[builder(default, setter(strip_option))]
    pub macro_definitions: Option<Vec<(String, Option<String>)>>,

    /// Sets the optimization level.
    #[builder(default, setter(strip_option))]
    pub optimization_level: Option<OptimizationLevel>,

    /// Shader source code path.
    pub path: PathBuf,

    /// Sets the source language.
    #[builder(default, setter(strip_option))]
    pub source_language: Option<SourceLanguage>,

    /// Data about Vulkan specialization constants.
    ///
    /// # Examples
    ///
    /// Basic usage (GLSL):
    ///
    /// ```
    /// # vk_shader_macros::glsl!(kind: comp, r#"
    /// #version 460 core
    ///
    /// // Defaults to 6 if not set using HotShader specialization_info!
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
    /// # use vk_graph::driver::shader::{SpecializationInfo};
    /// # use vk_graph_hot::shader::HotShader;
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
    /// # let my_shader_code = [0u8; 1];
    /// // We instead specify 42 for MY_COUNT:
    /// let shader = HotShader::new_fragment(my_shader_code.as_slice())
    ///     .specialization_info(SpecializationInfo::new(
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
    pub specialization_info: Option<SpecializationInfo>,

    /// The shader stage this structure applies to.
    pub stage: vk::ShaderStageFlags,

    /// Sets the target SPIR-V version.
    #[builder(default, setter(strip_option))]
    pub target_spirv: Option<SpirvVersion>,

    /// Sets the compiler mode to treat all warnings as errors.
    #[builder(default)]
    pub warnings_as_errors: bool,
}

impl HotShader {
    /// Specifies a shader with the given `stage` and shader code values.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(stage: vk::ShaderStageFlags, path: impl AsRef<Path>) -> HotShaderBuilder {
        HotShaderBuilder::new(stage, path)
    }

    /// Creates a new ray trace shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_any_hit(path: impl AsRef<Path>) -> HotShaderBuilder {
        Self::new(vk::ShaderStageFlags::ANY_HIT_KHR, path)
    }

    /// Creates a new ray trace shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_callable(path: impl AsRef<Path>) -> HotShaderBuilder {
        Self::new(vk::ShaderStageFlags::CALLABLE_KHR, path)
    }

    /// Creates a new ray trace shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_closest_hit(path: impl AsRef<Path>) -> HotShaderBuilder {
        Self::new(vk::ShaderStageFlags::CLOSEST_HIT_KHR, path)
    }

    /// Creates a new compute shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_compute(path: impl AsRef<Path>) -> HotShaderBuilder {
        Self::new(vk::ShaderStageFlags::COMPUTE, path)
    }

    /// Creates a new fragment shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_fragment(path: impl AsRef<Path>) -> HotShaderBuilder {
        Self::new(vk::ShaderStageFlags::FRAGMENT, path)
    }

    /// Creates a new geometry shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_geometry(path: impl AsRef<Path>) -> HotShaderBuilder {
        Self::new(vk::ShaderStageFlags::GEOMETRY, path)
    }

    /// Creates a new ray trace shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_intersection(path: impl AsRef<Path>) -> HotShaderBuilder {
        Self::new(vk::ShaderStageFlags::INTERSECTION_KHR, path)
    }

    /// Creates a new mesh shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_mesh(path: impl AsRef<Path>) -> HotShaderBuilder {
        Self::new(vk::ShaderStageFlags::MESH_EXT, path)
    }

    /// Creates a new ray trace shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_miss(path: impl AsRef<Path>) -> HotShaderBuilder {
        Self::new(vk::ShaderStageFlags::MISS_KHR, path)
    }

    /// Creates a new ray trace shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_ray_gen(path: impl AsRef<Path>) -> HotShaderBuilder {
        Self::new(vk::ShaderStageFlags::RAYGEN_KHR, path)
    }

    /// Creates a new mesh task shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_task(path: impl AsRef<Path>) -> HotShaderBuilder {
        Self::new(vk::ShaderStageFlags::TASK_EXT, path)
    }

    /// Creates a new tesselation control shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_tesselation_ctrl(path: impl AsRef<Path>) -> HotShaderBuilder {
        Self::new(vk::ShaderStageFlags::TESSELLATION_CONTROL, path)
    }

    /// Creates a new tesselation evaluation shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_tesselation_eval(spirv: impl AsRef<Path>) -> HotShaderBuilder {
        Self::new(vk::ShaderStageFlags::TESSELLATION_EVALUATION, spirv)
    }

    /// Creates a new vertex shader.
    ///
    /// # Panics
    ///
    /// If the shader code is invalid.
    pub fn new_vertex(path: impl AsRef<Path>) -> HotShaderBuilder {
        Self::new(vk::ShaderStageFlags::VERTEX, path)
    }

    pub(super) fn compile_and_watch(
        &self,
        watcher: &mut RecommendedWatcher,
    ) -> Result<Vec<u8>, DriverError> {
        let shader_kind = match self.stage {
            vk::ShaderStageFlags::ANY_HIT_KHR => ShaderKind::AnyHit,
            vk::ShaderStageFlags::CALLABLE_KHR => ShaderKind::Callable,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR => ShaderKind::ClosestHit,
            vk::ShaderStageFlags::COMPUTE => ShaderKind::Compute,
            vk::ShaderStageFlags::FRAGMENT => ShaderKind::Fragment,
            vk::ShaderStageFlags::GEOMETRY => ShaderKind::Geometry,
            vk::ShaderStageFlags::INTERSECTION_KHR => ShaderKind::Intersection,
            vk::ShaderStageFlags::MISS_KHR => ShaderKind::Miss,
            vk::ShaderStageFlags::RAYGEN_KHR => ShaderKind::RayGeneration,
            vk::ShaderStageFlags::TASK_EXT => ShaderKind::Task,
            vk::ShaderStageFlags::TESSELLATION_CONTROL => ShaderKind::TessControl,
            vk::ShaderStageFlags::TESSELLATION_EVALUATION => ShaderKind::TessEvaluation,
            vk::ShaderStageFlags::VERTEX => ShaderKind::Vertex,
            _ => unimplemented!("{:?}", self.stage),
        };

        let mut additional_opts = CompileOptions::new().map_err(|err| {
            error!("Unable to initialize compiler options: {err:?}");

            DriverError::Unsupported
        })?;

        if let Some(macro_definitions) = &self.macro_definitions {
            for (name, value) in macro_definitions {
                additional_opts.add_macro_definition(name, value.as_deref());
            }
        }

        additional_opts.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_2 as _);

        if let Some(language) = self.source_language.or_else(|| {
            let language = guess_shader_source_language(&self.path);

            if let Some(language) = language {
                debug!("Guessed source language: {:?}", language);
            }

            language
        }) {
            additional_opts.set_source_language(language);
        }

        additional_opts.set_target_spirv(self.target_spirv.unwrap_or(SpirvVersion::V1_5));

        if let Some(level) = self.optimization_level {
            additional_opts.set_optimization_level(level);
        }

        if self.warnings_as_errors {
            additional_opts.set_warnings_as_errors();
        }

        let res = compile_shader(
            &self.path,
            &self.entry_name,
            Some(shader_kind),
            Some(&additional_opts),
        )
        .map_err(|err| {
            error!("Unable to compile shader {}: {err}", self.path.display());

            DriverError::InvalidData
        })?;

        for path in res.files_included {
            watcher
                .watch(&path, RecursiveMode::NonRecursive)
                .map_err(|err| {
                    error!("Unable to watch file: {err}");

                    DriverError::Unsupported
                })?;
        }

        Ok(res.spirv_code)
    }
}

impl From<HotShaderBuilder> for HotShader {
    fn from(builder: HotShaderBuilder) -> HotShader {
        builder.build()
    }
}

// HACK: https://github.com/colin-kiegel/rust-derive-builder/issues/56
impl HotShaderBuilder {
    /// Specifies a shader with the given `stage` and shader path values.
    pub fn new(stage: vk::ShaderStageFlags, path: impl AsRef<Path>) -> Self {
        Self::default()
            .stage(stage)
            .path(path.as_ref().to_path_buf())
    }

    /// Builds a new `HotShader`.
    pub fn build(self) -> HotShader {
        let this = self;

        #[cfg(target_os = "macos")]
        let this = this.macro_definition("MOLTEN_VK", Some("1".to_string()));

        this.fallible_build()
            .expect("All required fields set at initialization")
    }

    /// Defines a single macro.
    pub fn macro_definition(
        mut self,
        key: impl Into<String>,
        value: impl Into<Option<String>>,
    ) -> Self {
        if self.macro_definitions.is_none() || self.macro_definitions.as_ref().unwrap().is_none() {
            self.macro_definitions = Some(Some(vec![]));
        }

        self.macro_definitions
            .as_mut()
            .unwrap()
            .as_mut()
            .unwrap()
            .push((key.into(), value.into()));

        self
    }
}

#[derive(Debug)]
struct HotShaderBuilderError;

impl From<UninitializedFieldError> for HotShaderBuilderError {
    fn from(_: UninitializedFieldError) -> Self {
        Self
    }
}
