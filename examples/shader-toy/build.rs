use {
    anyhow::Context,
    pak::PakBuf,
    shaderc::{Compiler, ShaderKind},
    std::{
        env::var,
        fs::write,
        path::{Path, PathBuf},
    },
};

// lazy_static is being deprecated so this stand-in provides similar functionality
macro_rules! lazy_static {
    ($name: ident: $ty: ty = $expr: expr) => {
        ::paste::paste! {
            struct [<__ $name:camel>];

            #[allow(unused)]
            static [<$name:upper>]: [<__ $name:camel>] = [<__ $name:camel>];

            impl ::std::ops::Deref for [<__ $name:camel>] {
                type Target = $ty;
                fn deref(&self) -> &Self::Target {
                    static S: ::std::sync::OnceLock<$ty> = ::std::sync::OnceLock::new();
                    S.get_or_init(|| $expr)
                }
            }
        }
    };

    {$(static ref $name: ident: $ty: ty = $expr: expr;)+} => {
        $(lazy_static!($name: $ty = $expr);)+
    };
}

lazy_static! {
    static ref CARGO_MANIFEST_DIR: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    static ref OUT_DIR: PathBuf = PathBuf::from(var("OUT_DIR").unwrap());
}

fn main() -> anyhow::Result<()> {
    let pak_output_dir = OUT_DIR.to_path_buf();

    // In this mode we put the pak in the target/debug (eg) directory
    #[cfg(not(feature = "include-pak"))]
    let pak_output_dir = pak_output_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    // Pack images and heavy data into a .pak file
    let pak_output_path = pak_output_dir.join("data.pak");
    PakBuf::bake(CARGO_MANIFEST_DIR.join("res/pak.toml"), &pak_output_path)?;
    create_pak_bindings(&pak_output_path, OUT_DIR.join("pak_bindings.rs"))
        .context("Unable to write pak bindings")?;

    // Compile the shaders and include their bytes in the rust code
    let shader_dir = CARGO_MANIFEST_DIR.join("res/shader");
    let shaders = vec![
        compile_glsl(shader_dir.join("quad.vert"), ShaderKind::Vertex)
            .context("Unable to compile quad shader")?,
        compile_glsl(shader_dir.join("flockaroo_buf.frag"), ShaderKind::Fragment)
            .context("Unable to compile buf shader")?,
        compile_glsl(shader_dir.join("flockaroo_img.frag"), ShaderKind::Fragment)
            .context("Unable to compile img shader")?,
    ];
    create_shader_bindings(shader_dir, shaders, OUT_DIR.join("shader_bindings.rs"))
        .context("Unable to write shader bindings")?;

    Ok(())
}

fn join_strings(strings: impl IntoIterator<Item = String>, separator: &str) -> String {
    strings.into_iter().collect::<Vec<_>>().join(separator)
}

fn compile_glsl(path: impl AsRef<Path>, kind: ShaderKind) -> anyhow::Result<(PathBuf, PathBuf)> {
    let source_path = path.as_ref().to_path_buf();
    let source = read_shader_source(&source_path);
    let compiler = Compiler::new().unwrap();
    let spirv = compiler
        .compile_into_spirv(
            &source,
            kind,
            &path.as_ref().to_string_lossy(),
            "main",
            None,
        )?
        .as_binary_u8()
        .to_vec();

    let mut spirv_path = OUT_DIR.join(source_path.file_name().unwrap());
    spirv_path.set_extension(format!(
        "{}.spirv",
        source_path.extension().unwrap().to_string_lossy()
    ));
    write(&spirv_path, spirv).context("Unable to write SPIR-V binary")?;

    Ok((source_path, spirv_path))
}

fn create_pak_bindings(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> anyhow::Result<()> {
    let mut bindings = String::new();
    for key in PakBuf::open(src)?.keys() {
        bindings.push_str("pub const ");
        bindings.push_str(
            key.to_ascii_uppercase()
                .replace(['\\', '/', '-', '.', '!'], "_")
                .as_str(),
        );
        bindings.push_str(": &str = r#\"");
        bindings.push_str(key);
        bindings.push_str("\"#;\n");
    }

    write(dst, bindings).context("Unable to write bindings text")?;

    Ok(())
}

fn create_shader_bindings(
    shader_path: impl AsRef<Path>,
    shaders: Vec<(PathBuf, PathBuf)>,
    dst: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let mut bindings = String::new();
    for shader in &shaders {
        bindings.push_str("pub const ");
        bindings.push_str(
            join_strings(remove_common_path(shader_path.as_ref(), &shader.0), "_")
                .to_ascii_uppercase()
                .replace(['\\', '/', '-', '.', '!'], "_")
                .as_str(),
        );
        bindings.push_str(": &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/");
        bindings.push_str(
            join_strings(remove_common_path(&*OUT_DIR, &shader.1), "/")
                .replace('\\', "/")
                .as_str(),
        );
        bindings.push_str("\"));\n");
    }

    write(dst, bindings).context("Unable to write bindings text")?;

    Ok(())
}

// Given two paths, returns the strings of the unique parts of the given path only:
// "c:\foo\bar" and "c:\foo\bar\baz\bop.txt" will return "baz\bop.txt"
fn remove_common_path(
    common_path: impl AsRef<Path>,
    path: impl AsRef<Path>,
) -> impl IntoIterator<Item = String> {
    let common_path = common_path.as_ref().to_path_buf();
    let mut path = path.as_ref().to_path_buf();
    let mut res = vec![];
    while path != common_path {
        res.push(path.file_name().unwrap().to_string_lossy().to_string());
        path = path.parent().unwrap().to_path_buf();
    }

    res.into_iter().rev()
}

fn read_shader_source(path: impl AsRef<Path>) -> String {
    use {
        shader_prepper::{
            BoxedIncludeProviderError, IncludeProvider, ResolvedInclude, ResolvedIncludePath,
            process_file,
        },
        std::fs::read_to_string,
    };

    struct FileIncludeProvider;

    impl IncludeProvider for FileIncludeProvider {
        type IncludeContext = PathBuf;

        fn get_include(
            &mut self,
            path: &ResolvedIncludePath,
        ) -> Result<String, BoxedIncludeProviderError> {
            println!("cargo:rerun-if-changed={}", &path.0);

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

    process_file(
        path.as_ref().to_string_lossy().as_ref(),
        &mut FileIncludeProvider,
        PathBuf::new(),
    )
    .unwrap()
    .iter()
    .map(|chunk| chunk.source.as_str())
    .collect()
}
