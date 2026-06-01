# Workspace Crates

This directory contains the publishable helper crates that extend `vk-graph`.
Each helper crate is versioned independently and has its own changelog.

## Included Crates

- [`vk-graph-window`](vk-graph-window/README.md): `winit` integration for window creation and frame
  handling.
- [`vk-graph-egui`](vk-graph-egui/README.md): renderer integration for
  [egui](https://github.com/emilk/egui).
- [`vk-graph-fx`](vk-graph-fx/README.md): reusable effects and utility helpers built on top of
  `vk-graph`.
- [`vk-graph-hot`](vk-graph-hot/README.md): shader hot-reload support for compute, graphics, and
  ray tracing pipelines.
- [`vk-graph-imgui`](vk-graph-imgui/README.md): renderer integration for
  [Dear ImGui](https://github.com/imgui-rs/imgui-rs).
