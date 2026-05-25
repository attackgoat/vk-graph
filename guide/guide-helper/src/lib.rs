//! Preprocessor for the vk-graph guide.

use {
    cargo_toml::Manifest,
    mdbook_preprocessor::{
        Preprocessor, PreprocessorContext,
        book::Book,
        errors::{Error, Result},
    },
    semver::{Version, VersionReq},
    std::{io, path::PathBuf, sync::LazyLock},
};

const LATEST_KNOWN_VULKAN_SDK_VERSION: &str = VULKAN_SDK_VERSION_1_3_281;
const VULKAN_SDK_VERSION_1_3_281: &str = "1.3.281";

static WORKSPACE_ROOT: LazyLock<PathBuf> = LazyLock::new(|| {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("missing guide dir")
        .parent()
        .expect("missing workspace root")
        .to_path_buf()
});

/// Preprocessing entry point.
pub fn handle_preprocessing() -> Result<()> {
    let pre = GuideHelper;
    let (ctx, book) = mdbook_preprocessor::parse_input(io::stdin())?;

    let book_version = Version::parse(&ctx.mdbook_version)?;
    let version_req = VersionReq::parse(mdbook_preprocessor::MDBOOK_VERSION)?;

    if !version_req.matches(&book_version) {
        eprintln!(
            "warning: The {} plugin was built against version {} of mdbook, \
             but we're being called from version {}",
            pre.name(),
            mdbook_preprocessor::MDBOOK_VERSION,
            ctx.mdbook_version
        );
    }

    let processed_book = pre.run(&ctx, book)?;
    serde_json::to_writer(io::stdout(), &processed_book)?;

    Ok(())
}

struct GuideHelper;

impl Preprocessor for GuideHelper {
    fn name(&self) -> &str {
        "guide-helper"
    }

    fn run(&self, _ctx: &PreprocessorContext, mut book: Book) -> Result<Book> {
        insert_crate_version(&mut book);
        insert_dependency_req(&mut book, "log");
        insert_dependency_req(&mut book, "profiling");
        insert_member_crate_version(&mut book, "vk-graph-hot", "crates/vk-graph-hot/Cargo.toml");
        insert_member_crate_version(
            &mut book,
            "vk-graph-window",
            "crates/vk-graph-window/Cargo.toml",
        );
        insert_vulkan_sdk_version(&mut book);

        let mut ok = true;
        book.for_each_chapter_mut(|ch| {
            ok &= !ch.content.contains("{{");
            ok &= !ch.content.contains("}}");
        });

        if ok {
            Ok(book)
        } else {
            Err(Error::msg("unredacted formatting marks"))
        }
    }
}

fn manifest() -> Manifest {
    manifest_at("Cargo.toml")
}

fn manifest_at(manifest_path: &str) -> Manifest {
    let path = WORKSPACE_ROOT.join(manifest_path);
    let res = Manifest::from_path(&path).expect("invalid manifest");

    if manifest_path == "Cargo.toml" {
        assert_eq!(
            res.package.as_ref().expect("missing package").name,
            "vk-graph"
        );
    }

    res
}

fn insert_crate_version(book: &mut Book) {
    let Version { major, minor, .. } =
        Version::parse(manifest().package.expect("missing package").version())
            .expect("invalid version");
    let version = format!("{major}.{minor}");

    const MARKER: &str = "{{ crate.version }}";

    book.for_each_chapter_mut(|ch| {
        if ch.content.contains(MARKER) {
            ch.content = ch.content.replace(MARKER, &version);
        }
    });
}

fn insert_dependency_req(book: &mut Book, dep: &str) {
    let req = manifest()
        .dependencies
        .get(dep)
        .expect("missing dependency")
        .req()
        .to_owned();
    let marker = format!("{{{{ {dep}.version }}}}");

    book.for_each_chapter_mut(|ch| {
        if ch.content.contains(&marker) {
            ch.content = ch.content.replace(&marker, &req);
        }
    });
}

fn insert_member_crate_version(book: &mut Book, dep: &str, manifest_path: &str) {
    let Version { major, minor, .. } = Version::parse(
        manifest_at(manifest_path)
            .package
            .as_ref()
            .expect("missing package")
            .version(),
    )
    .expect("invalid version");
    let version = format!("{major}.{minor}");
    let marker = format!("{{{{ {dep}.version }}}}");

    book.for_each_chapter_mut(|ch| {
        if ch.content.contains(&marker) {
            ch.content = ch.content.replace(&marker, &version);
        }
    });
}

fn insert_vulkan_sdk_version(book: &mut Book) {
    // Technically this is a VersionReq but we're not using it that way!
    let Version { major, minor, .. } = Version::parse(
        manifest()
            .dependencies
            .get("ash")
            .expect("missing dependency")
            .req(),
    )
    .expect("invalid version");

    let vulkan_sdk_version = vulkan_sdk_version_for_ash(major, minor);

    const MARKER: &str = "{{ vulkan_sdk.version }}";

    book.for_each_chapter_mut(|ch| {
        if ch.content.contains(MARKER) {
            ch.content = ch.content.replace(MARKER, vulkan_sdk_version);
        }
    });
}

fn vulkan_sdk_version_for_ash(major: u64, minor: u64) -> &'static str {
    match (major, minor) {
        (0, 38) => LATEST_KNOWN_VULKAN_SDK_VERSION,
        _ => panic!("unknown ash version; update Vulkan SDK guide mapping"),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn current_ash_version_maps_to_expected_sdk() {
        assert_eq!(
            vulkan_sdk_version_for_ash(0, 38),
            VULKAN_SDK_VERSION_1_3_281
        );
    }

    #[test]
    #[should_panic(expected = "unknown ash version; update Vulkan SDK guide mapping")]
    fn unknown_ash_version_panics() {
        let _ = vulkan_sdk_version_for_ash(0, 99);
    }
}
