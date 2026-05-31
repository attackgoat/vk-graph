//! Hot-reload ray-trace pipeline support.

use {
    super::{
        HotPipeline, compile_shaders_and_watch, create_watcher, pipeline, pipeline_handle,
        shader::HotShader,
    },
    log::info,
    std::sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
    },
    vk_graph::{
        cmd::{Command, Pipeline},
        driver::{
            DriverError,
            device::Device,
            ray_trace::{RayTracePipeline, RayTracePipelineInfo, RayTraceShaderGroup},
        },
    },
};

/// A ray-trace pipeline wrapper that recompiles its shaders when source files change.
#[derive(Debug)]
pub struct HotRayTracePipeline {
    cache: RwLock<HotPipeline<RayTracePipeline>>,
    device: Device,
    has_changes: Arc<AtomicBool>,
    shader_groups: Box<[RayTraceShaderGroup]>,
    shaders: Box<[HotShader]>,
}

impl HotRayTracePipeline {
    /// Creates a hot-reload ray-trace pipeline from shader files and shader groups.
    pub fn create<S>(
        device: &Device,
        info: impl Into<RayTracePipelineInfo>,
        shaders: impl IntoIterator<Item = S>,
        shader_groups: impl IntoIterator<Item = RayTraceShaderGroup>,
    ) -> Result<Self, DriverError>
    where
        S: Into<HotShader>,
    {
        let shaders = shaders.into_iter().map(Into::into).collect::<Box<_>>();
        let shader_groups = shader_groups.into_iter().collect::<Box<_>>();

        let has_changes = Default::default();
        let mut watcher = create_watcher(&has_changes);

        let pipeline = RayTracePipeline::create(
            device,
            info,
            compile_shaders_and_watch(&shaders, &mut watcher)?,
            shader_groups.iter().copied(),
        )?;

        Ok(Self {
            cache: RwLock::new(HotPipeline { pipeline, watcher }),
            device: device.clone(),
            has_changes,
            shader_groups,
            shaders,
        })
    }

    fn compile_shader_and_bind_cmd<'a>(
        &self,
        cmd: Command<'a>,
    ) -> <RayTracePipeline as Pipeline<'a>>::Command {
        if self.has_changes.swap(false, Ordering::Relaxed) {
            info!("Shader change detected");

            let mut cache = self.cache_mut();

            if let Ok(shaders) = compile_shaders_and_watch(&self.shaders, &mut cache.watcher)
                && let Ok(pipeline) = RayTracePipeline::create(
                    &self.device,
                    cache.pipeline.info(),
                    shaders,
                    self.shader_groups.iter().copied(),
                )
            {
                cache.pipeline = pipeline;
            }
        }

        self.cache().pipeline.clone().bind_cmd(cmd)
    }
}

pipeline!(RayTrace);
pipeline_handle!(RayTrace);
