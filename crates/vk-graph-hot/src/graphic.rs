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
        graphic::{GraphicPipeline, GraphicPipelineInfo},
        DriverError,
    },
};

/// TODO
#[derive(Debug)]
pub struct HotGraphicPipeline {
    has_changes: Arc<AtomicBool>,
    pipeline: GraphicPipeline,
    shaders: Box<[HotShader]>,
    watcher: RecommendedWatcher,
}

impl HotGraphicPipeline {
    /// TODO
    pub fn create<S>(
        device: &Device,
        info: impl Into<GraphicPipelineInfo>,
        shaders: impl IntoIterator<Item = S>,
    ) -> Result<Self, DriverError>
    where
        S: Into<HotShader>,
    {
        let shaders = shaders
            .into_iter()
            .map(|shader| shader.into())
            .collect::<Box<_>>();

        let (mut watcher, has_changes) = create_watcher();
        let compiled_shaders = shaders
            .iter()
            .map(|shader| compile_shader_and_watch(shader, &mut watcher))
            .collect::<Result<Vec<_>, _>>()?;

        let pipeline = GraphicPipeline::create(device, info, compiled_shaders)?;

        Ok(Self {
            has_changes,
            pipeline,
            shaders,
            watcher,
        })
    }

    /// Returns the most recent compilation without checking for changes or re-compiling the shader
    /// source code.
    #[deprecated = "use Deref instead"]
    #[doc(hidden)]
    pub fn cold(&self) -> &GraphicPipeline {
        self
    }

    /// Returns the most recent compilation after checking for changes, and if needed re-compiling
    /// the shader source code.
    pub fn hot(&mut self) -> &GraphicPipeline {
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
                if let Ok(pipeline) = GraphicPipeline::create(
                    self.pipeline.device(),
                    self.pipeline.info(),
                    compiled_shaders,
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

impl AsRef<GraphicPipeline> for HotGraphicPipeline {
    fn as_ref(&self) -> &GraphicPipeline {
        self
    }
}

impl Deref for HotGraphicPipeline {
    type Target = GraphicPipeline;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

#[allow(unused)]
mod deprecated {
    use {crate::graphic::HotGraphicPipeline, vk_graph::driver::graphic::GraphicPipeline};

    impl HotGraphicPipeline {
        #[deprecated = "use Deref instead"]
        fn as_ref(&self) -> &GraphicPipeline {
            self
        }
    }
}
