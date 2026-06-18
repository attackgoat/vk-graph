
Open API cleanup candidates:

- Audit hidden-but-public items and decide which are intended advanced API, deprecated shims, or
  accidental public surface.
- Consider replacing custom range-like public structs with standard `Range` only if that improves
  ergonomics without losing Vulkan-specific meaning.
- Simplify complex public generic bounds where possible.
- Expand `DriverError` coverage for Vulkan error codes that downstream applications may need to
  handle explicitly.
- Consider a public memory-location control if users need more allocation placement control than the
  current buffer/image info exposes.

## Current Implementation Follow-Ups

Small remaining source follow-ups:

- The image barrier-emission paths in submission still contain similar logic and may benefit from a
  shared helper if the code is touched again.
- Dense image ownership promotion allocates backing storage on the first partial update. This is
  correct but should remain visible in API docs or performance notes if partial ownership updates are
  common in user code.

## Current Issue Notes

Short latest-issue list:

- Extra compaction in image sync.
- Silly device field in `ImageView`.
- `QueueSignals` needs a new name.
- `PartialQueueFamily` seems overkill.
- Swapchain functions have an odor.

Static-analysis cleanup areas:

- Pool implementations in `src/pool/fifo.rs`, `src/pool/lazy.rs`, and `src/pool/hash.rs` repeat
  entry, lease, and `get_unchecked` structure. A trait or macro could unify them.
- `src/node.rs` has multiple match-dispatch patterns where every enum variant delegates to the same
  method.
- `src/lib.rs` has roughly a dozen `AnyResource` dispatch matches with identical bodies per arm.
- Dead or unused code includes `src/driver/shader.rs:183` `samplers`,
  `src/driver/device.rs:212` `expect_vk_khr_present_wait()`,
  `src/driver/device.rs:1156` `vk_khr_present_wait`, and generated `is_*` / `unwrap_*` methods in
  `src/cmd/pipeline.rs:74,79`.
- Unsafe concerns include `unreachable_unchecked()` in `src/submission.rs`, un-commented
  `get_unchecked()` calls in `src/pool/fifo.rs` and `src/pool/lazy.rs`.
- Large files/functions include `src/submission.rs` with functions such as `lease_render_pass`,
  `write_descriptor_sets`, `record_image_layout_transitions`, `record_execution_barriers`, and
  `build_subpass_dependencies`, plus oversized `src/driver/image.rs`.
- Several public functions returning `Result`, `Option`, or `bool` in `src/lib.rs` may need
  `#[must_use]`.
- Redundant clone candidates include `ranges.clone()` patterns in `src/submission.rs`, `cmd.clone()`
  in `src/stream.rs:925`, and `.entry(info.clone())` in pool files.
- Heavy `as usize` / `as u32` usage appears in `submission.rs`, `shader.rs`, `graphics.rs`,
  `render_pass.rs`, and `image.rs`; `TryFrom` or `From` may be safer, especially `bool as usize` in
  `render_pass.rs:217-219`.
- Range loops in `src/submission.rs` could use iterators or `enumerate()`.
- Lint suppressions worth revisiting include `#[allow(private_bounds)]`,
  `#[allow(clippy::type_complexity)]`, `#[allow(missing_docs)]`, and
  `#[allow(unused_must_use)]` in `egui/src/lib.rs:324`.
- Miscellaneous cleanup includes replacing pointer casts with `Arc::ptr_eq`, checking whether
  `Box::new(physical_device)` is necessary, and reviewing `#[allow(private_interfaces)]` in
  `src/driver/instance.rs:287`.


## Naming Cleanup Checklist

Image naming suggestions:

| Current | Suggested | Reason |
|---|---|---|
| `Image::view` | `get_or_create_view` | Creates or caches a view, not just returns a field. |
