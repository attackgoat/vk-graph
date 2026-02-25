//! TODO

use {
    super::{compile_shader_and_watch, create_watcher, shader::HotShader},
    log::info,
    notify::RecommendedWatcher,
    std::{
        ops::Deref,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
    },
    vk_graph::driver::{
        device::Device,
        ray_trace::{RayTracePipeline, RayTracePipelineInfo, RayTraceShaderGroup},
        DriverError,
    },
};

/// TODO
#[derive(Debug)]
pub struct HotRayTracePipeline {
    has_changes: Arc<AtomicBool>,
    pipeline: RayTracePipeline,
    shader_groups: Box<[RayTraceShaderGroup]>,
    shaders: Box<[HotShader]>,
    watcher: RecommendedWatcher,
}

impl HotRayTracePipeline {
    /// TODO
    pub fn create<S>(
        device: &Device,
        info: impl Into<RayTracePipelineInfo>,
        shaders: impl IntoIterator<Item = S>,
        shader_groups: impl IntoIterator<Item = RayTraceShaderGroup>,
    ) -> Result<Self, DriverError>
    where
        S: Into<HotShader>,
    {
        let shader_groups = shader_groups.into_iter().collect::<Box<_>>();
        let shaders = shaders
            .into_iter()
            .map(|shader| shader.into())
            .collect::<Box<_>>();

        let (mut watcher, has_changes) = create_watcher();
        let compiled_shaders = shaders
            .iter()
            .map(|shader| compile_shader_and_watch(shader, &mut watcher))
            .collect::<Result<Vec<_>, _>>()?;

        let pipeline = RayTracePipeline::create(
            device,
            info,
            compiled_shaders,
            shader_groups.iter().copied(),
        )?;

        Ok(Self {
            has_changes,
            pipeline,
            shader_groups,
            shaders,
            watcher,
        })
    }

    /// Returns the most recent compilation without checking for changes or re-compiling the shader
    /// source code.
    #[deprecated = "use Deref instead"]
    #[doc(hidden)]
    pub fn cold(&self) -> &RayTracePipeline {
        self
    }

    /// Returns the most recent compilation after checking for changes, and if needed re-compiling
    /// the shader source code.
    pub fn hot(&mut self) -> &RayTracePipeline {
        let has_changes = self.has_changes.swap(false, Ordering::Relaxed);

        if has_changes {
            info!("Shader change detected");

            let (mut watcher, has_changes) = create_watcher();
            if let Ok(compiled_shaders) = self
                .shaders
                .iter()
                .map(|shader| compile_shader_and_watch(shader, &mut watcher))
                .collect::<Result<Vec<_>, DriverError>>()
            {
                if let Ok(pipeline) = RayTracePipeline::create(
                    self.pipeline.device(),
                    self.pipeline.info(),
                    compiled_shaders,
                    self.shader_groups.iter().copied(),
                ) {
                    self.pipeline = pipeline;
                    self.has_changes = has_changes;
                    self.watcher = watcher;
                }
            }
        }

        self
    }
}

impl AsRef<RayTracePipeline> for HotRayTracePipeline {
    fn as_ref(&self) -> &RayTracePipeline {
        self
    }
}

impl Deref for HotRayTracePipeline {
    type Target = RayTracePipeline;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

#[allow(unused)]
mod deprecated {
    use {crate::ray_trace::HotRayTracePipeline, vk_graph::driver::ray_trace::RayTracePipeline};

    impl HotRayTracePipeline {
        #[deprecated = "use Deref instead"]
        fn as_ref(&self) -> &RayTracePipeline {
            self
        }
    }
}
