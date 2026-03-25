//! Preprocessor for the vk-graph guide.

use {
    cargo_toml::Manifest,
    mdbook_preprocessor::{
        Preprocessor, PreprocessorContext,
        book::Book,
        errors::{Error, Result},
    },
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
        insert_dependency_req(&mut book, "log");
        insert_dependency_req(&mut book, "profiling");
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
    let path = current_dir().unwrap().parent().unwrap().join("Cargo.toml");
    let res = Manifest::from_path(path).unwrap();

    assert_eq!(res.package.as_ref().unwrap().name, "vk-graph");

    res
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

fn insert_dependency_req(book: &mut Book, dep: &str) {
    let req = manifest().dependencies.get(dep).unwrap().req().to_owned();
    let marker = format!("{{{{ {dep}.version }}}}");

    book.for_each_chapter_mut(|ch| {
        if ch.content.contains(&marker) {
            ch.content = ch.content.replace(&marker, &req);
        }
    });
}

fn insert_vulkan_sdk_version(book: &mut Book) {
    // Technically this is a VersionReq but we're not using it that way!
    let Version { major, minor, .. } =
        Version::parse(manifest().dependencies.get("ash").unwrap().req()).unwrap();

    // HACK: Instead of parsing the current lock file, just hardcode new versions into this
    let vulkan_sdk_version = match (major, minor) {
        (0, 38) => "1.3.281",
        _ => todo!("add new version details"),
    };

    const MARKER: &str = "{{ vulkan_sdk.version }}";

    book.for_each_chapter_mut(|ch| {
        if ch.content.contains(MARKER) {
            ch.content = ch.content.replace(MARKER, vulkan_sdk_version);
        }
    });
}
