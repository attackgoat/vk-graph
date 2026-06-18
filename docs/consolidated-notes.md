
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


## RenderDoc Debug Labeling Feature

The debug-labeling feature adds Vulkan debug-utils object naming and command label regions so
RenderDoc and similar tools can show useful labels.

Core behavior:

- Debug naming is enabled when `InstanceInfo::debug` or `DeviceInfo::debug` is `true`.
- Debug construction fails unless both `VK_EXT_debug_utils` and `VK_EXT_private_data` are supported.
- `VK_EXT_debug_utils` provides Vulkan object naming and command labels.
- `VK_EXT_private_data` provides Rust-side name lookup and internal naming propagation.
- Vulkan validation layers are not required for object naming, but are useful for validating labels
  and command regions.
- `instance.info.debug` remains the runtime source of truth for debug behavior.

Device foundation changes:

- Add `vk_ext_debug_utils: Option<ext::debug_utils::Device>` to `DeviceInner`.
- Load the extension in `try_from_ash`.
- Add helpers for object naming and command label regions.
- Fail instance/device creation in debug mode when required extensions are unavailable.

Named resource types:

- `Buffer` in `src/driver/buffer.rs`.
- `Image` in `src/driver/image.rs`.
- `AccelerationStructure` in `src/driver/accel_struct.rs`.
- `Lease<Buffer>`, `Lease<Image>`, and `Lease<AccelerationStructure>` in `src/pool/mod.rs`.
- `CommandBuffer`.
- Swapchain images, named as `swapchain{index}` at creation.

Naming API convention:

- Primary setter: `set_debug_name(&self, name: impl AsRef<str>)`.
- Convenience builder: `with_debug_name(self, name: impl AsRef<str>) -> Self`.
- Setters may be called multiple times and should replace the previous name.
- Public setters and builders accept `impl AsRef<str>`; getter APIs are internal-only.

Pipeline naming:

- `ComputePipeline`, `GraphicsPipeline`, and `RayTracingPipeline` automatically name internal
  objects when a debug name is set.
- Compute and ray tracing `VkPipeline` handles are named directly.
- `VkPipelineLayout` is named `"{pipeline_name} (layout)"`.
- `VkDescriptorSetLayout` is named `"{pipeline_name} (DS{set_index})"`.
- Cached graphics `VkPipeline` variants derive names from the logical pipeline name and
  render-pass/subpass context.
- Graphics variants recover the logical pipeline name from private-data metadata when a
  render-pass-specific variant is created or revisited.

Command labeling:

- `CommandData` should keep its internal name field only for debug-build diagnostics.
- Submission-time recording should wrap command regions in `vkCmdBeginDebugUtilsLabelEXT` and
  `vkCmdEndDebugUtilsLabelEXT`.
- Nested execution labels should be prefixed with the originating command name, for example
  `"{command_name} / exec {n}"`.
- `CommandBuffer` gets a debug-name helper that forwards directly to Vulkan naming.
- Prepared command-stream handoff is labeled with one `command stream boundary` scope.
- Internal labels include command buffer submission scope, graph command scopes, execution scopes,
  render pass scopes, command stream boundary scopes, and swapchain image scopes.

Example usage:

```rust
let buffer = Buffer::create(&device, info)?.with_debug_name("my_buffer");

let buffer = Buffer::create(&device, info)?;
buffer.set_debug_name("my_buffer");

let pipeline = ComputePipeline::create(&device, info)?.with_debug_name("my_compute_pipeline");

let pipeline = ComputePipeline::create(&device, info)?;
pipeline.set_debug_name("my_compute_pipeline");

let mut cmd_buf = CommandBuffer::create(...);
cmd_buf.set_debug_name("my_command_buffer");

graph.begin_cmd().debug_name("My Render Pass").bind_pipeline(&pipeline);
```

## Queue Ownership Transfer Sequence

For an exclusive Vulkan resource moving from one queue family to another, the acquire barrier must be
submitted after a matching release barrier has executed on a queue from the source queue family.

Required event order:

1. Previous work establishes the resource owner as source family `S`.
2. `Submission::record` selects destination family `D`.
3. Destination recording detects access while exclusive owner is family `S`.
4. Destination command buffer records acquire barriers with
   `srcQueueFamilyIndex = S` and `dstQueueFamilyIndex = D`.
5. `Recording::finish()` allocates release command buffers from the source command pool.
6. Release command buffer records matching release barriers with the same resource, range, and image
   layout as the acquire barriers.
7. Release command buffer is submitted on a source-family queue and signals a semaphore.
8. Release command buffer, fence, and semaphore are retained for the `RecordedSubmission` lifetime.
9. `RecordedSubmission::queue_submit` submits the destination command buffer.
10. Destination submit waits on the release semaphore.
11. Acquire barriers execute before destination work reads or writes the resource.
12. Submission fence keeps recorded state alive until GPU completion.
13. `RecordedSubmission::attach` records owner as destination family and queue.

Call and lifetime flow:

- `Graph::finalize` creates `Submission`.
- `Submission::record` selects commands.
- `track_pending_transfers` reads current exclusive sharing state.
- `queue_ownership_release_groups` snapshots source family and queue.
- `record_cmd_indices` records the destination command buffer.
- `record_execution_barriers` and `record_image_layout_transitions` record acquire barriers and
  consume pending acquire state.
- `Recording::finish` submits matching source-side release barriers.
- `submit_queue_ownership_releases` creates short one-time source-family command buffers, records
  release barriers, submits them on remembered source queues, and returns semaphores for destination
  submit waits.
- `RecordedSubmission` retains release objects through `_releases`.
- `Fence::drop_when_signaled` keeps recorded submission state alive while the GPU may reference it.

Vulkan requirements:

- Release barriers are recorded into command buffers allocated for the source queue family.
- Acquire barriers are recorded into command buffers allocated for the destination queue family.
- Release and acquire barriers use the same resource, queue family indices, and buffer range or image
  subresource range.
- Image ownership transfers also require matching image layout parameters for the transferred range.
- Release submit must happen-before the acquire operation; vk-graph uses a semaphore signaled by the
  release submit and waited by destination submit.
- Release command buffer, fence, and semaphore must stay alive until destination submission no longer
  needs them.
- Tracked exclusive owner should not update to the destination queue until the destination submission
  is attached to a fence that keeps recorded submission state alive until completion.

## Verification Baseline

For source changes related to these notes, run narrow tests first and then broaden as needed:

1. `cargo fmt --package vk-graph`.
2. `cargo check -p vk-graph`.
3. `cargo test -p vk-graph driver::buffer`.
4. `cargo test -p vk-graph driver::image`.
5. `cargo test -p vk-graph submission::tests`.
6. `cargo test -p vk-graph --doc`.

Review verification already recorded:

- `cargo test --workspace --no-run` passed.
- `cargo test -p vk-graph --lib` passed.
- `cargo test -p guide --doc` passed after stale allocation-field snippets were updated.
