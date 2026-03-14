use {
    anyhow::Context,
    pak::PakBuf,
    shaderc::{Compiler, ShaderKind},
    simplelog::{CombinedLogger, ConfigBuilder, LevelFilter, WriteLogger},
    std::{
        env::var,
        fs::{File, read_to_string, write},
        path::{Path, PathBuf},
    },
};

fn main() -> anyhow::Result<()> {
    let cargo_manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(var("OUT_DIR").unwrap());
    let target_dir = out_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let log_path = cargo_manifest_dir.join("build.log");
    let mut log_config = ConfigBuilder::new();
    log_config.set_time_offset_to_local().unwrap();
    CombinedLogger::init(vec![WriteLogger::new(
        LevelFilter::Trace,
        log_config.build(),
        File::create(log_path).context("Creating log file")?,
    )])?;

    compile_shader(
        cargo_manifest_dir.join("res/shader/animated_mesh.vert"),
        cargo_manifest_dir.join("res/shader/animated_mesh_vert.spirv"),
    )?;
    compile_shader(
        cargo_manifest_dir.join("res/shader/mesh.frag"),
        cargo_manifest_dir.join("res/shader/mesh_frag.spirv"),
    )?;

    let toml_path = cargo_manifest_dir.join("res/pak.toml");
    let pak_path = target_dir.join("res.pak");
    PakBuf::bake(toml_path, pak_path)?;

    Ok(())
}

fn compile_shader(
    source_path: impl AsRef<Path>,
    dest_path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let source_path = source_path.as_ref();
    let source_text = read_to_string(source_path).context("Reading source path")?;

    let compiler = Compiler::new().unwrap();
    let spirv_code = compiler
        .compile_into_spirv(
            &source_text,
            match source_path
                .extension()
                .unwrap_or_default()
                .to_string_lossy()
                .as_ref()
            {
                "frag" => ShaderKind::Fragment,
                "vert" => ShaderKind::Vertex,
                _ => unimplemented!(),
            },
            &source_path.to_string_lossy(),
            "main",
            None,
        )
        .inspect_err(|_| {
            eprintln!("Shader: {}", source_path.display());

            for (idx, line) in source_text.split('\n').enumerate() {
                eprintln!("{}: {line}", idx + 1);
            }

            eprintln!();
        })
        .context("Compiling")?
        .as_binary_u8()
        .to_vec();

    write(dest_path, spirv_code).context("Writing SPIR-V")?;

    Ok(())
}
