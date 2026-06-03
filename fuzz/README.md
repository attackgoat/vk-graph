# Fuzzing

This directory contains standalone libFuzzer targets for `vk-graph`.

Targets here should focus on small, high-value invariants in subsystems that benefit from broad,
generated input coverage.

## Targets

- `submission_schedule`
  - Exercises `vk_graph::submission::fuzz::check_schedule_reordering(...)`.
  - Generates synthetic pass/resource usage graphs.
  - Verifies that `Schedule::reorder_passes(...)` matches the reference implementation.
  - Also checks permutation and determinism invariants.

Additional fuzz targets can be added under `fuzz_targets/` as other subsystems need coverage.

## Layout

- `Cargo.toml`: standalone fuzz crate manifest.
- `fuzz_targets/`: libFuzzer entrypoints and input generators.

Fuzz-specific entrypoints live here, while invariant checkers may live in the main crate when that
lets fuzzing exercise real production logic without duplicating it in the harness.

## Build

Compile the target directly:

```sh
cargo build --manifest-path fuzz/Cargo.toml --bin submission_schedule
```

Available fuzz targets are defined by filenames under `fuzz/fuzz_targets/`.

## Run With cargo-fuzz

If `cargo-fuzz` is installed, run:

```sh
cargo fuzz run submission_schedule
```

Install it with:

```sh
cargo install cargo-fuzz
```

## Notes

- The fuzz crate is kept separate from the main crate to avoid pulling `libfuzzer-sys` into normal
  library builds.
- Inputs should usually be intentionally bounded so fuzzing spends time exploring behavior instead
  of growing huge synthetic structures.
