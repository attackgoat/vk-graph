# Window Handling

`vk-graph` does not directly provide any window implementation. Instead an accessory crate,
`vk-graph-window` is provided, based on `winit`.

> [!TIP]
> [_`vk-graph-window`_ ](https://github.com/attackgoat/vk-graph/tree/main/crates/vk-graph-window)
> <i class="fa-solid fa-arrow-up-right-from-square"></i> provides additional documentation and
> examples.

## Swapchain

The bifurcation of `vk-graph` along the window abstraction results in two `Swapchain` types, one in
each crate.

Type | Usage
-- | --
`vk_graph::driver::swapchain::Swapchain` | Vulkan swapchain smart pointer, contains "raw" functions
`vk_graph_window::swapchain::Swapchain` | High-level display interface for building window handlers

### OpenXR

Virtual reality support via OpenXR is provided as
[_an example_](https://github.com/attackgoat/vk-graph/tree/main/examples/vr)
<i class="fa-solid fa-arrow-up-right-from-square"></i> which also implements a
swapchain.
