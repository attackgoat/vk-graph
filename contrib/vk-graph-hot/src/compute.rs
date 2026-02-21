//! TODO

use {
    super::{compile_shader_and_watch, create_watcher, shader::HotShader},
    log::info,
    notify::RecommendedWatcher,
    std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    vk_graph::driver::{
        compute::{ComputePipeline, ComputePipelineInfo},
        device::Device,
        DriverError,
    },
};

/// TODO
#[derive(Debug)]
pub struct HotComputePipeline {
    has_changes: Arc<AtomicBool>,
    pipeline: ComputePipeline,
    shader: HotShader,
    watcher: RecommendedWatcher,
}

impl HotComputePipeline {
    /// TODO
    pub fn create(
        device: &Device,
        info: impl Into<ComputePipelineInfo>,
        shader: impl Into<HotShader>,
    ) -> Result<Self, DriverError> {
        let shader = shader.into();

        let (mut watcher, has_changes) = create_watcher();
        let compiled_shader = compile_shader_and_watch(&shader, &mut watcher)?;

        let pipeline = ComputePipeline::create(device, info, compiled_shader)?;

        Ok(Self {
            has_changes,
            pipeline,
            shader,
            watcher,
        })
    }

    /// Returns the most recent compilation without checking for changes or re-compiling the shader
    /// source code.
    pub fn cold(&self) -> &ComputePipeline {
        &self.pipeline
    }

    /// Returns the most recent compilation after checking for changes, and if needed re-compiling
    /// the shader source code.
    pub fn hot(&mut self) -> &ComputePipeline {
        let has_changes = self.has_changes.swap(false, Ordering::Relaxed);

        if has_changes {
            info!("Shader change detected");

            let (mut watcher, has_changes) = create_watcher();
            if let Ok(compiled_shader) = compile_shader_and_watch(&self.shader, &mut watcher) {
                if let Ok(pipeline) = ComputePipeline::create(
                    self.pipeline.device(),
                    self.pipeline.info(),
                    compiled_shader,
                ) {
                    self.pipeline = pipeline;
                    self.has_changes = has_changes;
                    self.watcher = watcher;
                }
            }
        }

        self.cold()
    }
}

impl AsRef<ComputePipeline> for HotComputePipeline {
    fn as_ref(&self) -> &ComputePipeline {
        &self.pipeline
    }
}
