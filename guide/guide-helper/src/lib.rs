//! Preprocessor for the vk-graph guide.

use {
    cargo_toml::Manifest,
    mdbook_preprocessor::{Preprocessor, PreprocessorContext, book::Book, errors::Result},
    semver::{Version, VersionReq},
    std::{env::current_dir, io},
};

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
        insert_vulkan_sdk_version(&mut book);

        Ok(book)
    }
}

fn manifest() -> Manifest {
    let path = current_dir().unwrap().parent().unwrap().join("Cargo.toml");

    Manifest::from_path(path).unwrap()
}

fn insert_crate_version(book: &mut Book) {
    let Version { major, minor, .. } =
        Version::parse(manifest().package.unwrap().version()).unwrap();
    let version = format!("{major}.{minor}");

    const MARKER: &str = "{{ crate.version }}";

    book.for_each_chapter_mut(|ch| {
        if ch.content.contains(MARKER) {
            ch.content = ch.content.replace(MARKER, &version);
        }
    });
}

fn insert_vulkan_sdk_version(book: &mut Book) {
    // Technically this is a VersionReq but we're not using it that way and want the build metadata
    let Version { build, .. } =
        Version::parse(manifest().dependencies.get("ash").unwrap().req()).unwrap();

    const MARKER: &str = "{{ vulkan_sdk.version }}";

    book.for_each_chapter_mut(|ch| {
        if ch.content.contains(MARKER) {
            ch.content = ch.content.replace(MARKER, build.as_str());
        }
    });
}
