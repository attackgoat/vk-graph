# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.14.5] - 2026-07-13

### Added

- `Fixture` import and export support for serializing graph resource and command relationships.
- Real-workload fixtures and benchmarks for submission scheduling.

### Changed

- Submission command reordering now uses dependency-safe topological scheduling to reduce
  scheduling overhead while preserving per-resource command order.
- Distinct command-stream arguments can no longer bind to the same parent graph node.

### Fixed

- Partial `RecordSelection` scheduling now includes complete predecessor command prefixes when a
  resource is rediscovered at a later command boundary.
- Repeated partial recordings no longer duplicate queue-family ownership transfers for overlapping
  buffer and image subresource ranges.

## [0.14.4] - 2026-07-03

### Added

- `Graph::write_timestamp`, `Command::write_timestamp`, and `PipelineCommand::write_timestamp` for
  device performance queries.
- `Fence::timestamps` and `TimestampQueryPool` for retrieving submitted timestamp results.
- Checked builds validate that `TimestampQueryPool::duration` receives a timestamp query from the
  same graph.

### Changed

- `vk-graph-window` now targets `vk-graph` `0.14.4` and is versioned as `0.1.3`.

## [0.14.3] - 2026-06-25

### Added

- Command builder one-shot commands for recording graph work.
- `vk-graph-imgui` support for frame-scoped user images in ImGui widgets.

### Changed

- Workspace integration crates affected by this release now target `vk-graph` `0.14.3` and are
  versioned as `0.1.2`.

### Deprecated

- `Fence::is_signaled`; use `Fence::status` instead.
- `Fence::wait_signaled`; use `Fence::wait` instead.

## [0.14.2] - 2026-06-18

### Added

- `Graph` replaces the old `RenderGraph` workflow with command-first graph construction and
  `Submission` finalization.
- `stream` module with reusable `CommandStream` values and typed stream arguments.
- `submission` module for prepared submission data, dependency tracking, and graph execution state.
- `node` module with resource node types moved out of the old `graph::node` namespace.
- `driver::graphics` and `cmd::graphics` modules replace the old singular `graphic` modules.
- `driver::ray_tracing` and `cmd::ray_tracing` modules replace the old `ray_trace` modules.
- `vk-graph-window` `Graphchain` API replaces the old display/swapchain presentation wrapper.
- `Fence` driver type and additional swapchain presentation result/error types.
- `checked` feature for runtime graph and stream validation.

### Changed

- This release intentionally ignores semver-checks compatibility failures against the published
  `0.14.0+alpha` crate. It includes removed modules, renamed types, renamed fields, changed method
  signatures, and trait changes while remaining on the `0.14` release line.
- The old pass-oriented `graph` API was replaced by command builder APIs under `cmd` and root-level
  `Graph` methods.
- `Display`/`DisplayInfo` presentation APIs moved out of the main crate; use `vk-graph-window` and
  `Graphchain` for windowed presentation.
- `GraphicPipeline`, `GraphicPipelineInfo`, `BlendMode`, and `DepthStencilMode` were renamed to
  `GraphicsPipeline`, `GraphicsPipelineInfo`, `BlendInfo`, and `DepthStencilInfo`.
- `RayTracePipeline`, `RayTracePipelineInfo`, and `RayTraceShaderGroup` were renamed to
  `RayTracingPipeline`, `RayTracingPipelineInfo`, and `RayTracingShaderGroup`.
- `Device::physical_device` was renamed to `Device::physical`.
- Physical-device extension support now uses actual Vulkan extension names such as
  `vk_khr_swapchain`, `vk_khr_synchronization2`, and grouped `Option<khr::...>` support structs for
  KHR extensions with feature/property data.
- Ambiguous public `ty` fields were renamed to domain-specific names such as `image_type`,
  `view_type`, `acceleration_structure_type`, and `shader_group_type`.
- `SpecializationInfo` was renamed to `SpecializationMap`.
- Builder conversion methods now consistently use `into_builder` instead of `to_builder`.
- Device, resource, pipeline, swapchain, and surface APIs were tightened around explicit creation,
  ownership, and unsafe raw-handle construction.
- Workspace integration crates now target `vk-graph` `0.14.2` and are versioned as `0.1.1`.

### Removed

- Removed the old `graph`, `graph::pass_ref`, and `display` public APIs.
- Removed the old `prelude` re-export surface from the main crate.
- Removed `pool::alias` and `AliasPool`.
- Removed old `Device::create_headless`, `Device::create_display`, `Device::load`,
  `Device::format_properties`, `Device::image_format_properties`, and `Device::instance` helpers.
- Removed old `Buffer::access`, `Image::access`, and `AccelerationStructure::access` helpers.
- Removed `Pool::lease`; implement `Pool::resource` for custom pools instead.

### Fixed

- Documentation examples now compile as `no_run` doctests instead of being ignored.
- Rustdoc tables and module docs were cleaned up across command, driver, shader, stream, and example
  documentation.

## Planned

- Support https://vulkan.gpuinfo.org/displayextensiondetail.php?extension=VK_EXT_full_screen_exclusive, possibly through extension crate
- Fix a few scheduling and merging issues found with specific workloads
- Add optional new-style Vulkan render passes when not on mobile, or when requested, to see if that improves performance by reducing some of the inter-renderpass optimization steps - possibly a build feature
- Clean up graph pass API - the names of the methods are not all consistent or in line with the Vulkan-spec terminology
- Make all Info structs have the same properties: Copy, exact parameter values sent to Vulkan are not magically interpreted and changed by vk-graph
- New crate name to reflect the stability and functionality of this crate, retire "QBasic" references, sad face

## [0.13.0] - 2025-12-14

## Added

- Image format texel block size/extent support functions (`format_texel_block_size()`/`format_texel_block_extent()`)
- `Surface::capabilities()` and `Surface::present_modes()` support functions

## Changed

- `Swapchain` allows configurable present modes
- Custom `ImageType` enum removed and replaced with existing `vk::ImageType`

## Fixed

- Issue where RenderDoc clears images between renderpasses ("undefined img") due to incorrect usage
  of image layout barriers
- Out-of-memory errors during image and buffer creation leaked Vulkan resource handles
- Build error seen on Android and Raspberry Pi (_See [#105](https://github.com/attackgoat/screen-13/pull/105)_)

## [0.12.6] - 2025-05-10

## Added

- Support for more automatic vertex-layout type detection in vertex shaders

## Changed

- Switch local fork of `vk-sync-fork` to upstream crate (at v0.5)

## Fixed

- `ImageInfo::cube(..)` now returns valid information (layers set to 6; auto-set CUBE flags)

## [0.12.5] - 2025-04-07

### Fixed

- Segmentation fault crash and flickering on macOS when resizing the swapchain (_See [#99](https://github.com/attackgoat/screen-13/pull/99)_)

## [0.12.4] - 2025-03-30

### Added

- Support for MSAA [sample rate shading](https://vulkan.gpuinfo.org/listdevicescoverage.php?feature=sampleRateShading&platform=all) (_[`GraphicPipelineInfo::min_sample_shading`](https://docs.rs/screen-13/latest/screen_13/driver/graphic/struct.GraphicPipelineInfo.html#structfield.min_sample_shading)_)
- Initial support for MSAA [coverage](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#fragops-covg) (_[`GraphicPipelineInfo::alpha_to_coverage`](https://docs.rs/screen-13/latest/screen_13/driver/graphic/struct.GraphicPipelineInfo.html#structfield.alpha_to_coverage)/[`GraphicPipelineInfo::alpha_to_one`](https://docs.rs/screen-13/latest/screen_13/driver/graphic/struct.GraphicPipelineInfo.html#structfield.alpha_to_one)_)

## [0.12.3] - 2025-03-27

### Changed

- Symbolic link to `vk-sync` now uses a path attribute (easier to build from source on Windows)

### Fixed

- Fix compilation issue on macOS
- Remove incorrect debug assertion for swapchain desired image count

## [0.12.2] - 2025-03-24

### Fixed

- `RenderGraph::blit_image()` internal offset logic was backwards; caused validation error

## [0.12.1] - 2025-03-23

### Fixed

- Swapchain image displays nothing after Window presentation on Linux Mesa drivers

## [0.12.0] - 2025-03-13

### Added

- Automatic `vk::PipelineCache` for compute, graphic and ray-trace pipelines
- `Acceleration::build_structures()` (and update/indirect variant commands)
- `SampleCount::is_single()` and `is_multiple()` helper methods
- `mip_compute.rs` and `mip_graphic.rs` examples of typical mip-map usage

### Changed

- Use Rust 2024 edition
- Updated `ash` to v0.38
- Updated `winit` to v0.30 (_and moved related functionality to new `screen-13-window` crate_)
- Updated `ordered-float` to v5.0
- `ResolverPool` trait now requires `Send`
- `screen_13_egui` allows creation using `raw_window_traits` instead of requiring an `EventLoop`
- `vk-sync` brought into source tree until an updated fork is created or existing PRs/fixes merged

### Fixed

- Handling of `u8` and `u16` values used as push constants
- Framebuffer attachments generated from certain valid configurations
- Resolving MSAA depth attachments
- Device feature detection issue which caused a validation layer error
- Performance during resize and overall correctness of swapchain handling
- Multiple issues related to subresource handling, extraneous pipeline barriers and render pass
  merge logic

### Removed

- `log` and `winit` are no longer exported by `use screen_13::prelude::*`

## [0.11.4] - 2024-07-16

### Fixed

- Validation error: `Attempted write update to an immutable sampler descriptor`

## [0.11.3] - 2024-05-29

### Added

- Support for separate image samplers (`SamplerState` in HLSL, `sampler` in GLSL)

### Changed

- Updated `egui` to v0.26
- Updated `gpu-allocator` to v0.26
- Updated `spirq` to v1.2
  * If you see errors such as `expected spirq_core::ty::Type, found spirq::prelude::Type` you will
    need to run `cargo update` or remove your `Cargo.lock` file

### Removed

- `lazy_static` dependency

## [0.11.2] - 2024-02-26

### Changed

- `parking_lot::Mutex` is now an optional feature 
- `pool` types now use more efficient internal caching

### Fixed

- `Resolver::submit` would sometimes drop and create command pools too often

## [0.11.1] - 2024-02-20

### Added

- `puffin` profiling to most example code - see [getting started guide](examples/getting-started.md)
  for more information

### Changed

- `ResolveMode` moved from `driver` to `driver::render_pass`

### Fixed

- `Swapchain::present_image` change introduced in `v0.11` needlessly spammed the `WARN` log while
  waiting for presentation images to be ready 
- Vulkan validation error introduced in `v0.11`: "_All queue submission commands that refer to fence
  must have completed execution_"

### Removed

- Unnecessary explicit reset of Vulkan command pools
- Unnecessary `Mutex` guarding `Framebuffer` and `GraphicPipeline` access

## [0.11.0] - 2024-02-18

### Added

- Min/max image sampler reduction mode - see [`examples/min_max.rs`](examples/min_max.rs)
  - `PhysicalDevice::sampler_filter_minmax_properties` added to report properties
  - `SamplerInfo::reduction_mode` added to set mode
- `SamplerInfo::LINEAR` and `NEAREST` to make sampler creation easier
- `ComputePipeline::with_name`, `GraphicPipeline::with_name` and `RayTracePipeline::with_name`
  debug helper functions
- `AccelerationStructureInfo::generic` function

### Changed

- `Device::image_format_properties` returns an `Option` so that unsupported formats may return
  `None` instead of relying on user code to detect `DriverError::Unsupported`
- Information struct trait implementations, field and function naming normalized:
  - Constructors no longer return builders
    - Use `to_builder` to convert an info struct into a builder
    - Use `build` to convert a builder into an info struct
  - `AccelerationStructureInfo`
    - Function `new_blas` renamed to `blas`
    - Function `new_tlas` renamed to `tlas`
  - `BufferInfo`
    - Function `new` renamed to `device_mem`
    - Function `new_mappable` renamed to `host_mem`
  - `ComputePipelineInfo` now implements `Copy`, `Eq` and `Hash`
  - `DeviceInfo` now implements `Default`
  - `GraphicPipelineInfo` now implements `Copy`
  - `ImageInfo`
    - Constructor parameters reordered: `fmt` now after image size
    - Function `new_2d` renamed to `image_2d` (_in addition to `cube`, `image_1d`, etc._)
    - Field `linear_tiling` renamed to `tiling` (_type changed from `bool` to `vk::ImageTiling`_)
  - `ImageViewInfo::new` function now `const`
  - `RayTracePipelineInfo` now implements `Copy`
  - `SwapchainInfo`
    - Function `new` now returns `SwapchainInfo` (_previously returned `SwapchainInfoBuilder`_)
    - Field `format` renamed to `surface`
    - Default values for `width` and `height` fields removed
- `ComputePipelineInfo::name`, `GraphicPipelineInfo::name` and `RayTracePipelineInfo::name` have
  each been moved to their respective pipeline struct
- `SampleCount` enum members renamed from `X1` to `Type1` (_etc._) to match Vulkan spec
- `EventLoop` now produces linear surfaces by default - use `desired_surface_format` and
  `Surface::srgb` to select sRGB
- `Swapchain::present_image` now uses event-based waiting for rendering operations instead of
  polling, greatly reducing CPU usage
- Updated `ash-molten` (_Mac OS support_) to v0.17

### Fixed

- `ComputePipelineInfo::default` now properly sets a default value for `bindless_descriptor_count`

### Removed

- `GraphicPipelineInfo::new` function: Use `Default` implementation instead
- `RayTracePipelineInfo::new` function: Use `Default` implementation instead
- `SamplerInfo::new` function: Use `Default` implementation instead

## [0.10.0] - 2024-02-09

### Added

- Ray tracing support for `vkCmdTraceRaysIndirectKHR` and dynamic stack size
- Resource aliasing re-introduced - see [`examples/aliasing.rs`](examples/aliasing.rs)
- Expanded the number of functions and scopes profiled by the `profiling` crate

### Changed

- Information structs are now `#[non_exhaustive]` in order to make future additions minor changes -
  update strategies:
  - Use `..Default::default()` syntax during struct creation
  - Use associated constructor functions such as `ImageInfo::new_2d(..)`
- `BufferInfo::can_map` renamed to `BufferInfo::mappable`
- Increase `PoolInfo::DEFAULT_RESOURCE_CAPACITY` from 4 to 16 in order to prevent excess resource
  creation

### Fixed

- `EventLoop`: Resize swapchain in response to events instead of each frame (_save 50 μs/frame_)

### Removed

- `input`: This module did not support functionality unique to _Screen 13_ and did not have higher
  quality than existing solutions such as
  [`winit_input_helper`](https://crates.io/crates/winit_input_helper)
- `EventLoopBuilder::linear_surface_format`/`srgb_surface_format`: Use `Surface::linear`/`srgb`
  instead

## [0.9.4] - 2024-02-07

### Changed

- Improved performance during render graph resolution: `vsm_omni` example now records frames 10%
  faster (~100 μs) and complex render graphs may be significantly more performant

## [0.9.3] - 2024-01-30

### Added

- `FifoPool` resource pool implementation
- Memory management functions and configurable bucket sizes for `Pool` implementations

### Fixed

- Compilation bug for `rustc` v1.75.0 on Mac OS

## [0.9.2] - 2024-01-23

### Changed

- Deprecated `EventLoop` surface format functions
- Updated `derive_builder` to v0.13
- Updated `gpu-allocator` to v0.25
- Updated `winit` to v0.29
- Updated `egui` to v0.25
- Updated `imgui-rs` to latest
  [`main`](https://github.com/imgui-rs/imgui-rs/tree/ca05418cb449dadaabf014487c5c965908dfcbdd)

## [0.9.1] - 2023-12-29

### Added

- Ability to select from available swapchain surface formats when creating an `EventLoop`
- Driver `surface` and `swapchain` modules (_and their types_) are now public API

### Changed

- Changed `KeyBuf` implementation functions to take values instead of borrows
- Updated `gpu-allocator` to v0.24
- Updated `spirq` to v1.0.2

## [0.9.0] - 2023-09-07

### Fixed

- Incorrect handling of images with multiple array layers during render passes
- Validation error related to `VK_KHR_surface` when using headless devices
- Shader modules of graphic pipelines cached by a render pass were not considered during lookup

### Added

- Support for performance profiling crates
- Queue family index is now a part of the API and allows for submission of render graph work using 
  secondary queue families
- Expose all Vulkan 1.0 properties via `PhysicalDevice::features_v1_0`
- `Device::format_properties` and `Device::image_format_properties` so user code may avoid
  calling unsafe `ash` functions
- `RenderGraph::node_device_address` function
- `contrib/screen-13-hot`: Shader compilation macro definition support
- Virtual reality example using OpenXR - see [`examples/vr`](examples/vr/README.md)
- Support for `VK_EXT_index_type_uint8`; use `device.physical_device.index_type_uint8_features.index_type_uint8` to check for support
- Manually configurable image samplers - see [`examples/image_sampler.rs`](examples/image_sampler.rs)

### Changed

- Device creation (and `EventLoop::build()`) no longer take a ray-tracing parameter; instead the
  device will be created and you should use
  `device.physical_device.ray_trace_features.ray_tracing_pipeline` to check for support
- Logical device (`Device`) structure has been moved to `screen_13::driver::device`
- Physical device feature and property structures have been moved to
  `screen_13::driver::physical_device`
- Re-ordered parameters of `RenderGraph` functions: `blit_image_region`, `blit_image_regions`, and
  `update_buffer_offset`
- Updated parameters of `RenderGraph` functions to be more efficient (`Into<Box<[_]>>` is now
  `AsRef<[_]>` and take values of `Copy`-types instead of borrows)
- `ResolverPool` trait has been moved from the `screen_13::graph` module to `screen_13`

### Removed

- `Driver` structure; use `Device::create_headless` directly
- `PhysicalDeviceDescriptorIndexingFeatures` and `FeatureFlags` as they are no longer required

## [0.8.1] - 2023-02-18

### Fixed

- Pipelines which use multiple descriptor sets (different `set =` values) sometimes trigger
  validation errors
- `contrib/screen-13-hot`: build error on Windows platform

### Added

- Custom vertex layout support - see [`examples/vertex_layout.rs`](examples/vertex_layout.rs)
- Enabled full set of Vulkan 1.1 and Vulkan 1.2 core features during device creation
- Ray query support with `ray_omni.rs` example
- Exposed existing command buffer implementation so that programs may wait for render graph GPU
  submissions to finish executing before reading the results with the CPU - see
  [`examples/cpu_readback.rs`](examples/cpu_readback.rs)
- `KeyBuf::is_down` helper function

### Changed

- `contrib/screen-13-egui`: Updated `egui` to v0.20

## [0.8.0] - 2022-12-28

### Added

- Shader hot-reload feature for compute, graphic and ray-trace pipelines (see examples)
- `Buffer` objects may be created with an alignment specified in `BufferInfo` (useful for shader
  binding tables)

### Changed

- `ComputePipeline::create` now takes three arguments: the device, info, and shader
- `ComputePipelineInfo` no longer contains shader information; use `Shader::new_compute` for that
  instead

## [0.7.1] - 2022-12-17

### Fixed

- Soundness issue in `AccelerationStructure::instance_slice` helper function

### Added

- Skeletal mesh animation demonstration in [`examples/skeletal-anim`](examples/skeletal-anim/README.md)

## [0.7.0] - 2022-12-05

### Fixed

- Validation error caused by image blit operations
- `multipass.rs` and other examples use unsupported image formats without checking for fallbacks

### Added

- `EventLoop` may be constructed with multiple hardware queues, see `desired_queue_count` and the
  new `multithread.rs` example

### Changed

- `Resolver::submit()` now takes a queue index instead of an instance; `Device::queue_count`
  provides the total number of queues available

## [0.6.5] - 2022-11-11

### Fixed

- Incorrectly skipped pipeline barriers on resources used in secondary render passes
- Semaphore in-use validation error when dropping swapchain
- Validation error caused by back-to-back image reads in auto-merged fragment shader passes
- Validation error caused by node access for the ALL_COMMANDS stage before graphic passes
- Multiple validation errors in the `example/` and `contrib/` code

### Added

- `bindless.rs` example using an unbounded image sampler array and draw indirect call

### Changed

- Leased resources now reference their pool using `Weak` reference counting to improve drop ordering

## [0.6.4] - 2022-10-31

### Fixed

- Framebuffer resolve functionality was implemented incorrectly, did not work
- Synchronization error when using compute written-resources in fragment shaders
- Validation error in `multipass.rs` example
- Unnecessary depth buffer store operations in `vsm_omni.rs` example

### Added

- Multisampled anti-aliasing example (MSAA)
- `attach_color` and `attach_depth_stencil` functions on `PipelinePassRef` when bound to a `GraphicPipeline` for attachments which would otherwise use `VK_ATTACHMENT_LOAD_OP_DONT_CARE`
- `node_info` function on `PassRef` and `PipelinePassRef` which may be accessed while recording passes

## [0.6.3] - 2022-10-25

### Fixed

- Panic when setting exclusive fullscreen if the monitor is set to less than maximum resolution
- Panic when overlapping push constant ranges in graphic and ray trace pipelines

### Added

- `bind_node` function on `PassRef` and `PipelinePassRef` which may be accessed while recording passes

### Changed

- Improved fullscreen experience: no extra decoration or briefly displayed small window
- Cursor re-displayed, if hidden, when event loop window loses focus

## [0.6.2] - 2022-10-20

### Fixed

- Crash/device lost while resizing the window
- Inconsistent frame timing on certain drivers
- Incorrect window size on certain drivers

### Added

- Fullscreen demonstration in `vsm_omni` example using F11 and F12 keys
- Configurable frames-in-flight setting

## [0.6.1] - 2022-10-16

### Fixed

- Depth/stencil images are now cleared properly
- Multi-layer framebuffers work as intended
- Render graph resolver orders renderpasses correctly

### Added

- `default_view_info()` helper on `ImageInfo` to assist in defining new views
- Variance shadow mapping example using a filtered cubemap

## [0.6.0] - 2022-10-06

### Changed

- `clear_color` and `clear_depth_stencil` functions now take the image being cleared: it is now possible to clear and attach, but not store or resolve, an image
- `record_`-* methods now also provide a `Bindings` parameter to the recording closure
- `RayTracePipeline::group_handle` is now an associated function where previously it was a method
- Many types have been moved between modules in order to document things clearly

### Removed

- `attach_color` and `attach_depth_stencil` functions: replace with the `load_` and `store_` functions for color or depth/stencil attachments
- `device_api.rs` helper functions: create resources directly
- `run` stand-alone function: Use `EventLoop` directly.
- Various internal-only fields and other types within the `driver` module

### Fixed

- Depth/stencil attachment clear requests are properly handled in cases where the image used is transient

## [0.5.0] - 2022-09-17

### Added

- `LazyPool` resource pool which tries to find acceptable resources before creating new ones
- `instance_slice` function for acceleration structures
- `new_blas` and `new_tlas` helper functions for acceleration structure info
- Node-`_mut` functions for `PassRef`: enables clearer code patterns when building passes
- `rt_triangle.rs` example; similar to `triangle.rs` but uses a ray trace pipeline

### Changed

- `build_structure` and `update_structure` now take geometry info as a borrow instead of by value

### Deprecated

- `input::Typing` struct; it has use valid cases but is not within the scope of this crate

### Removed

- `prelude_arc`: Use `prelude` module instead
- `driver::BlendMode::Replace` and other camel-case constants; use screaming-snake versions, i.e. `driver::BlendMode::REPLACE`

### Fixed

- Windows platform: `EventLoop` no longer panics if the window is minimized

## [0.4.2] - 2022-06-28

### Added

- `create_from_slice` function for buffers

## [0.4.1] - 2022-06-24

### Added

- `update_structure` function for acceleration structures
- `group_handle` function for ray trace pipelines

## [0.4.0] - 2022-06-06

_See [#25](https://github.com/attackgoat/screen-13/pull/25) for migration details_

### Added

- Resources may now be bound using `Arc<T>` of `driver` smart pointers: _`Buffer`, `Image`, etc_

### Changed

- Resource state is now held in the `driver` smart pointers instead of the current graph

### Removed

- "Binding" types, such as `ImageBinding` and `ImageLeaseBinding`: _use `Arc<Image>` instead_
- Dependency on the `archery` crate; _see [rationale](https://github.com/attackgoat/screen-13/pull/24)_

## [0.3.2] - 2022-06-01

### Added

- Additional memory mapping functions to `Buffer` structure

### Changed

- `BlendMode` graphic pipeline enumeration is now a structure with full options

## [0.3.1] - 2022-05-27

### Added

- Bindless descriptor support (unsized arrays in shader code) and example

### Fixed

- Improve swapchain image flag handling

## [0.3.0] - 2022-05-20

### Added

- Ray tracing support
- Subpass API, additional examples

### Removed

- Pak file functionality moved to `pak` [crate](https://crates.io/crates/pak)

## 0.2.1 - _Unreleased_

### Added

- Dear ImGui library and example
- Bitmapped text rendering

### Changed

- Pak file baking is now multi-threaded; assets still only get packed exactly once

### Removed

- `CommandChain` structure functionality is now found on the `RenderGraph` structure

## [0.2.0] - 2022-02-08

### Added

- Render Graph module, bindings, nodes, and executions: with render pass merging/re-ordering/etc
- `CommandChain` structure

### Changed

- Driver now directly based on Vulkan, having removed support for the deprecated Gfx-Hal library
- Lease/pool functionality simplified: leases are now obtained through a common interface using info
- `Engine`/`Program` structures have been merged into a simpler EventLoop structure

### Removed

- _Screen 13_ file-based configuration: use DriverConfig now
- `Gpu` and `Render` structures: use `RenderGraph` and `ImageNode` now
- Existing bitmap/draw/text/write/etc operations: functionality replaced and in some cases TODO

## 0.1.9 - _Unreleased_

### Added

- Implementation of `draw` functionality
- Implementation of `text` functionality

## 0.1.8 - _Unreleased_

### Added

- Features: `auto-cull`, `debug-names`, `deferred-3d`, and `forward-3d`
- Selectable `Arc` or `Rc` shared types

### Changed

- Allow `write` function to specify multiple writes per call
- Use builder pattern for all rendering commands
- Switched asset schema from JSON to TOML

## [0.1.0] - 2020-07-05

### Added

- Easy-to-use API designed to allow developers to create graphics programs which run on many
  platforms and require no bare-metal graphics API knowledge
- "Hello, world!" example using a bitmapped font

[Unreleased]: https://github.com/attackgoat/vk-graph/compare/v0.14.5...HEAD
[0.14.5]: https://crates.io/crates/vk-graph/0.14.5
[0.14.4]: https://crates.io/crates/vk-graph/0.14.4
[0.14.3]: https://crates.io/crates/vk-graph/0.14.3
[0.14.2]: https://crates.io/crates/vk-graph/0.14.2
[0.1.0]: https://crates.io/crates/screen-13/0.1.0
[0.2.0]: https://crates.io/crates/screen-13/0.2.0
[0.3.0]: https://crates.io/crates/screen-13/0.3.0
[0.3.1]: https://crates.io/crates/screen-13/0.3.1
[0.3.2]: https://crates.io/crates/screen-13/0.3.2
[0.4.0]: https://crates.io/crates/screen-13/0.4.0
[0.4.1]: https://crates.io/crates/screen-13/0.4.1
[0.4.2]: https://crates.io/crates/screen-13/0.4.2
[0.5.0]: https://crates.io/crates/screen-13/0.5.0
[0.6.0]: https://crates.io/crates/screen-13/0.6.0
[0.6.1]: https://crates.io/crates/screen-13/0.6.1
[0.6.2]: https://crates.io/crates/screen-13/0.6.2
[0.6.3]: https://crates.io/crates/screen-13/0.6.3
[0.6.4]: https://crates.io/crates/screen-13/0.6.4
[0.6.5]: https://crates.io/crates/screen-13/0.6.5
[0.7.0]: https://crates.io/crates/screen-13/0.7.0
[0.7.1]: https://crates.io/crates/screen-13/0.7.1
[0.8.0]: https://crates.io/crates/screen-13/0.8.0
[0.8.1]: https://crates.io/crates/screen-13/0.8.1
[0.9.0]: https://crates.io/crates/screen-13/0.9.0
[0.9.1]: https://crates.io/crates/screen-13/0.9.1
[0.9.2]: https://crates.io/crates/screen-13/0.9.2
[0.9.3]: https://crates.io/crates/screen-13/0.9.3
[0.9.4]: https://crates.io/crates/screen-13/0.9.4
[0.10.0]: https://crates.io/crates/screen-13/0.10.0
[0.11.0]: https://crates.io/crates/screen-13/0.11.0
[0.11.1]: https://crates.io/crates/screen-13/0.11.1
[0.11.2]: https://crates.io/crates/screen-13/0.11.2
[0.11.3]: https://crates.io/crates/screen-13/0.11.3
[0.11.4]: https://crates.io/crates/screen-13/0.11.4
[0.12.0]: https://crates.io/crates/screen-13/0.12.0
[0.12.1]: https://crates.io/crates/screen-13/0.12.1
[0.12.2]: https://crates.io/crates/screen-13/0.12.2
[0.12.3]: https://crates.io/crates/screen-13/0.12.3
[0.12.4]: https://crates.io/crates/screen-13/0.12.4
[0.12.5]: https://crates.io/crates/screen-13/0.12.5
[0.12.6]: https://crates.io/crates/screen-13/0.12.6
[0.13.0]: https://crates.io/crates/screen-13/0.13.0
