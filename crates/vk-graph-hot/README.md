# vk-graph Hot

Hot-reloading shader pipelines for _vk-graph_. Supports compute, graphic, and ray-trace shader
pipelines.

Based on shaderc. Feel free to submit PRs for other compilers.

## Quick Start

See the [example code](examples/README.md), 

## Basic usage

See the [GLSL](examples/glsl.rs) and [HLSL](examples/hlsl.rs) examples for usage - the hot pipelines
are drop-in replacements for the regular shader pipelines offered by _vk-graph_.

After creating a pipeline two functions are available, `hot` or `cold`. The result of each may be
bound to a render graph for any sort of regular use.

- `hot()`: Returns the pipeline instance which includes any changes found on disk.
- `cold()`: Returns the most recent successful compilation without watching for changes.

## Advanced usage

There are a few options available when creating a `HotShader` instance, which is a wrapper around
regular `Shader` instances. These options allow you to set compilation settings such as optimization
level and warnings-as-errors, among other things.

## More infomation

Run `cargo doc --open` to view detailed API documentation and find available compilation options.
