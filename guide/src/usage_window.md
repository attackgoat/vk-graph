# Window Handling

`vk-graph` does not directly provide any window implementation. Instead an accessory crate,
`vk-graph-window` is provided, based on `winit`.

> [!TIP]
> [`vk-graph-window`](https://github.com/attackgoat/vk-graph/tree/main/crates/vk-graph-window)
> provides additional documentation and examples.

## Swapchain

The bifurcation of `vk-graph` along the window abstraction results in two `Swapchain` types, one in
each crate.

Type | Usage
-- | --
`vk_graph::driver::swapchain::Swapchain` | Vulkan swapchain smart pointer, contains "raw" functions
`vk_graph_window::swapchain::Swapchain` | High-level display interface for building window handlers

### OpenXR

Virtual reality support via OpenXR is provided as
[an example](https://github.com/attackgoat/vk-graph/tree/main/examples/vr) which also implements a
swapchain.
