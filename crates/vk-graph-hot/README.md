# vk-graph Hot

Hot-reloading shader pipelines for _vk-graph_. Supports compute, graphic, and ray-trace shader
pipelines.
This crate is versioned independently from `vk-graph`.

Based on shaderc. Feel free to submit PRs for other compilers.

## Quick Start

See the [example code](examples/README.md) for complete GLSL and HLSL hot-reload examples.

## Basic usage

See the [GLSL](examples/glsl.rs) and [HLSL](examples/hlsl.rs) examples for usage - the hot pipelines
are drop-in replacements for the regular shader pipelines offered by _vk-graph_.

Use `HotShader` with a file path and it will automatically update the created pipeline whenever the
files included in the source code change.

## Advanced usage

There are a few options available when creating a `HotShader` instance, which is a wrapper around
regular `Shader` instances. These options allow you to set compilation settings such as optimization
level and warnings-as-errors, among other things.

## More information

Run `cargo doc --open` to view detailed API documentation and find available compilation options.
