# Window Handling

`vk-graph` does not directly provide any window implementation. Instead an accessory crate,
`vk-graph-window` is provided, based on `winit`.

> [!TIP]
> [_`vk-graph-window`_ ](https://github.com/attackgoat/vk-graph/tree/main/crates/vk-graph-window)
> <i class="fa-solid fa-arrow-up-right-from-square"></i> provides additional documentation and
> examples.

## Presentation

The core crate and window crate intentionally separate low-level Vulkan presentation from the
frame-oriented helper used by most windowed applications.

Type | Usage
-- | --
`vk_graph::driver::swapchain::Swapchain` | Vulkan swapchain smart pointer, contains "raw" functions
`vk_graph_window::graphchain::Graphchain` | High-level frame-presentation helper for window handlers

`Graphchain` acquires the next swapchain image, exposes it through [`FrameContext`], submits the
frame graph, and presents it. Applications normally use [`Window::run`] and write to
`frame.swapchain_image` rather than calling low-level acquire/present functions directly.

`GraphchainInfo` controls presentation policy such as `frame_capacity`, `min_image_count`,
`present_mode`, `acquire_timeout`, and `composite_alpha`. The effective runtime values are exposed
through `EffectiveGraphchainInfo` because surface capabilities may force different values at runtime.

Resize and surface transitions may cause a frame to be skipped internally. If the surface is lost,
the window layer tears down the current surface/graphchain and recreates them on a later draw
request.

### OpenXR

Virtual reality support via OpenXR is provided as
[_an example_](https://github.com/attackgoat/vk-graph/tree/main/examples/vr)
<i class="fa-solid fa-arrow-up-right-from-square"></i> which also implements a
swapchain.

[`FrameContext`]: https://docs.rs/vk-graph-window/latest/vk_graph_window/struct.FrameContext.html
[`Window::run`]: https://docs.rs/vk-graph-window/latest/vk_graph_window/struct.Window.html#method.run
