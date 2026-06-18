//! Hot-reload graphics pipeline support.

use {
    super::{HotPipeline, compile_shaders_and_watch, create_watcher, pipeline, shader::HotShader},
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
            graphics::{GraphicsPipeline, GraphicsPipelineInfo},
        },
    },
};

/// A graphics pipeline wrapper that recompiles its shaders when source files change.
#[derive(Debug)]
pub struct HotGraphicsPipeline {
    cache: RwLock<HotPipeline<GraphicsPipeline>>,
    device: Device,
    has_changes: Arc<AtomicBool>,
    shaders: Box<[HotShader]>,
}

impl HotGraphicsPipeline {
    /// Creates a hot-reload graphics pipeline from one or more shader files.
    pub fn create<S>(
        device: &Device,
        info: impl Into<GraphicsPipelineInfo>,
        shaders: impl IntoIterator<Item = S>,
    ) -> Result<Self, DriverError>
    where
        S: Into<HotShader>,
    {
        let shaders = shaders.into_iter().map(Into::into).collect::<Box<_>>();

        let has_changes = Default::default();
        let mut watcher = create_watcher(&has_changes);

        let pipeline = {
            GraphicsPipeline::create(
                device,
                info,
                compile_shaders_and_watch(&shaders, &mut watcher)?,
            )
        }?;

        Ok(Self {
            cache: RwLock::new(HotPipeline { pipeline, watcher }),
            device: device.clone(),
            has_changes,
            shaders,
        })
    }

    fn compile_shader_and_bind_cmd<'a>(
        &self,
        cmd: Command<'a>,
    ) -> <GraphicsPipeline as Pipeline<'a>>::Command {
        if self.has_changes.swap(false, Ordering::Relaxed) {
            info!("Shader change detected");

            let mut cache = self.cache_mut();

            if let Ok(shaders) = compile_shaders_and_watch(&self.shaders, &mut cache.watcher)
                && let Ok(pipeline) =
                    GraphicsPipeline::create(&self.device, cache.pipeline.info(), shaders)
            {
                cache.pipeline = pipeline;
            }
        }

        self.cache().pipeline.clone().bind_cmd(cmd)
    }
}

pipeline!(GraphicsPipeline);
