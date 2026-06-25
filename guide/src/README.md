# Introduction

`vk-graph` is a high-performance Vulkan driver for the Rust programming language featuring automated
resource management and execution. It is _blazingly_-fast, built for real-world use, and supports
modern Vulkan commands[^modern].

This guide book will walk you through the mental model of this crate and help explain how it maps to
Vulkan API usage.

Source code and issues live on the [GitHub repository](https://github.com/attackgoat/vk-graph).
If you find a problem in this guide, please [open an issue](https://github.com/attackgoat/vk-graph/issues).

> [!IMPORTANT]
> Users should be familiar with the Vulkan
[_specification_](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html)
<i class="fa-solid fa-arrow-up-right-from-square"></i>.

## Design

This guide provides a tour of the main public types:

Driver
  : _Buffer, Image, Shader, etc.._

Graph
  : _Builder-pattern for Vulkan commands_

Command
  : _Explicit command recording and region-level transfer helpers_

Submission
  : _Automated graph execution_

A `Graph` is built dynamically by your program each frame. Once complete, it is optimized into a
`Submission` that can be queued for execution on the Vulkan device.

Building and submitting a graph typically takes only a few hundred microseconds.

## Philosophy

Vulkan is hard. Synchronization is _extremely_ hard. `vk-graph` makes Vulkan *less painful* to write
and *a joy* to maintain.

The driver is based off the popular `ash` crate and `vk-sync`; reasoned as follows:
- _Everything_ is constructed from "`Info`" structs; all info is `Copy`
- Match the naming described in the specification
- Support all modern Vulkan usage[^modern] except video[^video]
- Don't use macro-magic or anything that needs to be learned
- Don't rely on "helper" functions unless absolutely required

## History

- 2018 --- Project started privately as a game engine using
[_`Corange`_](https://github.com/orangeduck/Corange)
<i class="fa-solid fa-arrow-up-right-from-square"></i>
- 2020 --- Project migrated to Github and named `screen-13`
- 2022 --- v0.2 released with an earlier graph API based on
[_`Kajiya`_](https://github.com/EmbarkStudios/kajiya)
<i class="fa-solid fa-arrow-up-right-from-square"></i>
- 2026 --- Project renamed `vk-graph` (v0.14) and the graph API was redesigned (v0.14.2)

[^modern]: Modern Vulkan usage means no pixel queries. Anything else unsupported is due to there
being better options, no current need, or no interest. Please open an issue.
[^video]: Video encode/decode is interesting but unsupported. As an alternative consider `ffmpeg`,
`libavcodec`, or one of the experimental Rust bindings to the Vulkan video API.
