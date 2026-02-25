# vk-graph

[![Crates.io](https://img.shields.io/crates/v/vk-graph.svg)](https://crates.io/crates/vk-graph)
[![Docs.rs](https://docs.rs/vk-graph/badge.svg)](https://docs.rs/vk-graph)
[![LoC](https://tokei.rs/b1/github/attackgoat/vk-graph?category=code)](https://github.com/attackgoat/vk-graph)

_vk-graph_ is a high-performance Vulkan graphics driver with automatic resource management and
execution.

```toml
[dependencies]
vk-graph = "0.14"
```

## Overview

_vk-graph_ provides a high performance [Vulkan](https://www.vulkan.org/) driver using smart
pointers. The driver may be created manually for headless rendering or automatically using the
built-in window abstraction:

```rust
use vk_graph_window::{Window, WindowError};

fn main() -> Result<(), WindowError> {
    Window::new()?.run(|frame| {
        // It's time to do some graphics! 😲
    })
}
```

## Usage

_vk-graph_ provides a fully-generic render graph structure for simple and statically
typed access to all the resources used while rendering. The `Graph` structure allows Vulkan
smart pointer resources to be bound as "nodes" which may be used anywhere in a graph. The graph
itself is not tied to swapchain access and may be used to execute general command streams.

Features of the render graph:

 - Compute, graphic, and ray-trace pipelines
 - Automatic Vulkan management (render passes, subpasses, descriptors, pools, _etc._)
 - Automatic render pass scheduling, re-ordering, merging, with resource aliasing
 - Interoperable with existing Vulkan code
 - Optional [shader hot-reload](contrib/vk-graph-hot/README.md) from disk

```rust
graph
    .begin_cmd()
    .debug_name("Fancy new algorithm for shading a moving character who is actively on fire")
    .bind_pipeline(&gfx_pipeline)
    .shader_resource_access(0, prev_image, AccessType::FragmentShaderReadColorInputAttachment)
    .shader_resource_access(1, some_image, AccessType::FragmentShaderReadOther)
    .shader_resource_access(3, fire_buffer, Access::FragmentShaderReadUniformBuffer)
    .clear_color(0, swapchain_image)
    .store_color(0, swapchain_image)
    .record_pipeline(move |pipeline, _| {
        pipeline
            .push_constants(some_u8_slice)
            .draw(6, 1, 0, 0);
    });
```
### Debug Logging

This crate uses [`log`](https://crates.io/crates/log) for low-overhead logging.

To enable logging, set the `RUST_LOG` environment variable to `trace`, `debug`, `info`, `warn` or
`error` and initialize the logging provider of your choice. Examples use
[`pretty_env_logger`](https://docs.rs/pretty_env_logger/latest/pretty_env_logger/).

_You may also filter messages, for example:_

```bash
RUST_LOG=vk_graph::driver=trace,vk_graph=warn cargo run --example ray_trace
```

```
TRACE vk_graph::driver::instance > created a Vulkan instance
DEBUG vk_graph::driver::physical_device > physical device: NVIDIA GeForce RTX 3090
DEBUG vk_graph::driver::physical_device > extension "VK_KHR_16bit_storage" v1
DEBUG vk_graph::driver::physical_device > extension "VK_KHR_8bit_storage" v1
DEBUG vk_graph::driver::physical_device > extension "VK_KHR_acceleration_structure" v13
...
```

### Performance Profiling

This crates uses [`profiling`](https://crates.io/crates/profiling) and supports multiple profiling
providers. When not in use profiling has zero cost.

To enable profiling, compile with one of the `profile-with-*` features enabled and initialize the
profiling provider of your choice.

_Example code uses [puffin](https://crates.io/crates/puffin):_

```bash
cargo run --features profile-with-puffin --release --example vsm_omni
```

<img src=".github/img/profile.png" alt="Flamegraph of performance data" width=30%>

## Quick Start

Included are some examples you might find helpful:

- [`hello_world.rs`](contrib/vk-graph-window/examples/hello_world.rs) — Displays a window on the screen. Please start here.
- [`triangle.rs`](examples/triangle.rs) — Shaders and full setup of index/vertex buffers; < 100 LOC.
- [`shader-toy/`](examples/shader-toy) — Recreation of a two-pass shader toy using the original
  shader code.

See the [example code](examples/README.md), 
[documentation](https://docs.rs/vk-graph/latest/vk_graph/), or helpful
[getting started guide](examples/getting-started.md) for more information.

**_NOTE:_** Required development packages and libraries are listed in the _getting started guide_.
All new users should read and understand the guide.

## History

As a child I was given access to a computer that had _GW-Basic_; and later one with _QBasic_. All of
my favorite programs started with:

```basic
CLS
SCREEN 13
```

These commands cleared the screen of text and setup a 320x200 256-color paletized video mode. There
were other video modes available, but none of them had the 'magic' of 256 colors.

Additional commands _QBasic_ offered, such as `DRAW`, allowed you to build simple games quickly
because you didn't have to grok the entirety of compiling and linking. I think we should have
options like this today, and so I started this project to allow future developers to have the
ability to get things done quickly while using modern tools.

### Inspirations

_vk-graph_ was built from the learnings and lessons shared by others throughout our community. In
particular, here are some of the repositories I found useful:

 - [Bevy](https://bevyengine.org/): A refreshingly simple data-driven game engine built in Rust
 - [Granite](https://github.com/Themaister/Granite) - Open-source Vulkan renderer
 - [Kajiya](https://github.com/EmbarkStudios/kajiya) - Experimental real-time global illumination
   renderer made with Rust and Vulkan
