pub mod compute;
pub mod graphic;
pub mod ray_trace;
pub mod shader;

pub mod prelude {
    pub use super::{
        compute::HotComputePipeline,
        graphic::HotGraphicPipeline,
        ray_trace::HotRayTracePipeline,
        shader::{HotShader, HotShaderBuilder, OptimizationLevel, SourceLanguage, SpirvVersion},
    };
}

use {
    self::shader::HotShader,
    log::{error, info},
    notify::{recommended_watcher, Event, EventKind, RecommendedWatcher},
    shader_prepper::{
        process_file, BoxedIncludeProviderError, IncludeProvider, ResolvedInclude,
        ResolvedIncludePath,
    },
    shaderc::{CompileOptions, Compiler, ShaderKind, SourceLanguage},
    std::{
        collections::HashSet,
        fs::read_to_string,
        io::{Error, ErrorKind},
        path::{Path, PathBuf},
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc, OnceLock,
        },
    },
    vk_graph::prelude::*,
};

struct CompiledShader {
    files_included: HashSet<PathBuf>,
    spirv_code: Vec<u8>,
}

fn compile_shader(
    path: impl AsRef<Path>,
    entry_name: &str,
    shader_kind: Option<ShaderKind>,
    additional_opts: Option<&CompileOptions<'_>>,
) -> anyhow::Result<CompiledShader> {
    info!("Compiling: {}", path.as_ref().display());

    let path = path.as_ref().to_path_buf();
    let shader_kind = shader_kind.unwrap_or_else(|| guess_shader_kind(&path));

    #[derive(Default)]
    struct FileIncludeProvider(HashSet<PathBuf>);

    impl IncludeProvider for FileIncludeProvider {
        type IncludeContext = PathBuf;

        fn get_include(
            &mut self,
            path: &ResolvedIncludePath,
        ) -> Result<String, BoxedIncludeProviderError> {
            self.0.insert(PathBuf::from(&path.0));

            Ok(read_to_string(&path.0)?)
        }

        fn resolve_path(
            &self,
            path: &str,
            context: &Self::IncludeContext,
        ) -> Result<ResolvedInclude<Self::IncludeContext>, BoxedIncludeProviderError> {
            let path = context.join(path);

            Ok(ResolvedInclude {
                resolved_path: ResolvedIncludePath(path.to_str().unwrap_or_default().to_string()),
                context: path
                    .parent()
                    .map(|path| path.to_path_buf())
                    .unwrap_or_default(),
            })
        }
    }

    let mut file_include_provider = FileIncludeProvider::default();
    let source_code = process_file(
        path.to_string_lossy().as_ref(),
        &mut file_include_provider,
        PathBuf::new(),
    )
    .map_err(|err| {
        error!("Unable to process shader file: {err}");

        Error::new(ErrorKind::InvalidData, err)
    })?
    .iter()
    .map(|chunk| chunk.source.as_str())
    .collect::<String>();
    let files_included = file_include_provider.0;

    static COMPILER: OnceLock<Compiler> = OnceLock::new();
    let spirv_code = COMPILER
        .get_or_init(|| Compiler::new().expect("Unable to initialize shaderc"))
        .compile_into_spirv(
            &source_code,
            shader_kind,
            &path.to_string_lossy(),
            entry_name,
            additional_opts,
        )
        .inspect_err(|_| {
            eprintln!("Shader: {}", path.display());

            for (line_index, line) in source_code.split('\n').enumerate() {
                let line_number = line_index + 1;
                eprintln!("{line_number}: {line}");
            }
        })?
        .as_binary_u8()
        .to_vec();

    Ok(CompiledShader {
        files_included,
        spirv_code,
    })
}

fn compile_shader_and_watch(
    shader: &HotShader,
    watcher: &mut RecommendedWatcher,
) -> Result<ShaderBuilder, DriverError> {
    let mut base_shader = Shader::new(shader.stage, shader.compile_and_watch(watcher)?.as_slice());

    base_shader = base_shader.entry_name(shader.entry_name.clone());

    if let Some(specialization_info) = &shader.specialization_info {
        base_shader = base_shader.specialization_info(specialization_info.clone());
    }

    Ok(base_shader)
}

fn create_watcher() -> (RecommendedWatcher, Arc<AtomicBool>) {
    let has_changes = Arc::new(AtomicBool::new(false));
    let has_changes_clone = Arc::clone(&has_changes);
    let watcher = recommended_watcher(move |event: notify::Result<Event>| {
        let event = event.unwrap_or_else(|_| Event::new(EventKind::Any));
        if matches!(
            event.kind,
            EventKind::Any | EventKind::Modify(_) | EventKind::Other
        ) {
            has_changes_clone.store(true, Ordering::Relaxed);
        }
    })
    .unwrap();

    (watcher, has_changes)
}

fn guess_shader_kind(path: impl AsRef<Path>) -> ShaderKind {
    match path
        .as_ref()
        .extension()
        .map(|ext| ext.to_string_lossy().to_string())
        .unwrap_or_default()
        .as_str()
    {
        "comp" => ShaderKind::Compute,
        "task" => ShaderKind::Task,
        "mesh" => ShaderKind::Mesh,
        "vert" => ShaderKind::Vertex,
        "geom" => ShaderKind::Geometry,
        "tesc" => ShaderKind::TessControl,
        "tese" => ShaderKind::TessEvaluation,
        "frag" => ShaderKind::Fragment,
        "rgen" => ShaderKind::RayGeneration,
        "rahit" => ShaderKind::AnyHit,
        "rchit" => ShaderKind::ClosestHit,
        "rint" => ShaderKind::Intersection,
        "rcall" => ShaderKind::Callable,
        "rmiss" => ShaderKind::Miss,
        _ => ShaderKind::InferFromSource,
    }
}

fn guess_shader_source_language(path: impl AsRef<Path>) -> Option<SourceLanguage> {
    match path
        .as_ref()
        .extension()
        .map(|ext| ext.to_string_lossy().to_string())
        .unwrap_or_default()
        .as_str()
    {
        "comp" | "task" | "mesh" | "vert" | "geom" | "tesc" | "tese" | "frag" | "rgen"
        | "rahit" | "rchit" | "rint" | "rcall" | "rmiss" | "glsl" => Some(SourceLanguage::GLSL),
        "hlsl" => Some(SourceLanguage::HLSL),
        _ => None,
    }
}
