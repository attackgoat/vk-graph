//! TODO

use {
    super::{
        HotPipeline, compile_shader_and_watch, create_watcher, pipeline, pipeline_handle,
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
            compute::{ComputePipeline, ComputePipelineInfo},
            device::Device,
        },
    },
};

/// TODO
#[derive(Debug)]
pub struct HotComputePipeline {
    cache: RwLock<HotPipeline<ComputePipeline>>,
    device: Device,
    has_changes: Arc<AtomicBool>,
    shader: HotShader,
}

impl HotComputePipeline {
    /// TODO
    pub fn create(
        device: &Device,
        info: impl Into<ComputePipelineInfo>,
        shader: impl Into<HotShader>,
    ) -> Result<Self, DriverError> {
        let shader = shader.into();

        let has_changes = Default::default();
        let mut watcher = create_watcher(&has_changes);

        let compiled_shader = compile_shader_and_watch(&shader, &mut watcher)?;

        let pipeline = ComputePipeline::create(device, info, compiled_shader)?;

        Ok(Self {
            cache: RwLock::new(HotPipeline { pipeline, watcher }),
            device: device.clone(),
            has_changes,
            shader,
        })
    }

    fn compile_shader_and_bind_cmd<'a>(
        &self,
        cmd: Command<'a>,
    ) -> <ComputePipeline as Pipeline<'a>>::Command {
        if self.has_changes.swap(false, Ordering::Relaxed) {
            info!("Shader change detected");

            let mut cache = self.cache_mut();

            if let Ok(shader) = compile_shader_and_watch(&self.shader, &mut cache.watcher)
                && let Ok(pipeline) =
                    ComputePipeline::create(&self.device, cache.pipeline.info(), shader)
            {
                cache.pipeline = pipeline;
            }
        }

        self.cache().pipeline.clone().bind_cmd(cmd)
    }
}

pipeline!(Compute);
pipeline_handle!(Compute);
