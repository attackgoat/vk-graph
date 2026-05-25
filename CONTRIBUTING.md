# Contributing to vk-graph

Thank you for taking the time to look over this document. You are encouraged to open issues, submit
PRs, suggest changes, or anything else you feel might move this project forward.

If you have any questions or would like private coorespondence with the main author, John Wells,
please use john@attackgoat.com.

## Ground Rules

This section lists the absolute minimum requirements you must understand and practice in order to be
involved with this project. If you fail to uphold the spirit of these rules you will not have any
access to make changes within this project and you may be banned entirely.

## Licensing Requirements

All contributions, ideas, issues, or other efforts you expend on this project must be provided
using the existing MIT or Apache 2.0 license agreements applied to this project. This means that
anything you do for this project will be provided to the public without any strings or conditions
attached. You must also have the right to provide any code or ideas under these licenses as you will
retain no ownership or control after contribution.

## Technical Requirements

All code must:

- Be modern Rust code (currently 2024 edition, latest stable compiler) _or_ GLSL.
- Pass `cargo fmt` and `cargo clippy` (debug and release) with no warnings
- Support required platforms: Linux, Mac, Windows
- Use only `crates.io`-published crates

### Recommendations

All code should:

- Follow the [guidelines](https://rust-lang.github.io/api-guidelines/)
- Provide useful documentation and comments, including private code
- Support other platforms: Android, iOS

## How do I get started

Download the source code and play around with the examples, see README.md for more.

Any and all contributions are acceptable, including major changes to design or capabilities. If your
change is big please ask or open an issue to make sure the rest of the community is on board with
your new ideas.

### What if my idea is too radical

If we cannot fit your change into the existing design without breaking API for other users or
creating other headaches then we will add your code into a `contrib` directory or separate branch as
needed.

## Release checklist

- Run `cargo update` to ensure you have the latest dependencies
- Double-check `cargo clippy --release` and `cargo fmt` _(use `bin/check`)_
- Double-check all examples compile and run as intended _(use `bin/run-all-examples`)_
- Double-check the above on all supported platforms
- Change log: Add a section for the new version
- Change log: Transfer unreleased details to the new version
- Change log: Add diff link (they are at the bottom)
- Run `cargo publish`
- Github: Add a tag for the version
