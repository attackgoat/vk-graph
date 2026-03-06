# Introduction

`vk-graph` is a high-performance Vulkan driver for the Rust programming language featuring automated resource management and execution. It is _blazingly_-fast, built for real-world use, and executes all modern Vulkan commands[^modern].

This guide book will walk you through the mental model of this crate and help explain how it maps to Vulkan API usage.

> [!IMPORTANT]
> Users should be familiar with the [Vulkan specification](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html).

## Design

This guide provides a tour of the main public types:

Resources
  : [Buffer](), [Image](), [Shader](), _etc.._

[Graph]()
  : _Builder-pattern for Vulkan commands_

[Queue]()
  : _Automated graph submission_

A `Graph` is data built dynamically by your program every frame. Once complete, the graph is optimized into a `Queue` which may be used to submit commands to the Vulkan implementation.

The wall-time overhead of this crate is intended to be in the 250 μs/frame range.

## Philosophy

Vulkan is hard. Synchronization is _extremely_ hard. `vk-graph` makes Vulkan *less painful* to write and *a joy* to maintain.

The driver is based off the popular `ash` crate and `vk-sync`; reasoned as follows:
- _Everything_ is constructed from "`Info`" structs; all info is `Copy`
- Match the naming described in the specification
- Support all modern Vulkan usage[^modern] except video[^video]
- Don't use macro-magic or anything that needs to be learned
- Don't rely on "helper" functions unless absolutely required

[^modern]: Modern Vulkan usage means no pixel queries. Anything else unsupported is due to there being better options, no current need, or no interest. Please open an issue.
[^video]: Video encode/decode is interesting but unsupported. As an alternative consider `ffmpeg`, `libavcodec`, or one of the experimental Rust bindings to the Vulkan video API.

## History




- 2026 --- v0.15 released and renamed `vk-graph`
- 2022 --- v0.2 released with `RenderGraph` type based on
[`Kajiya`](https://github.com/EmbarkStudios/kajiya)
- 2020 --- Project migrated to Github and named `Screen-13`
- 2018 --- Project started privately as a game engine using
[`Corange`](https://github.com/orangeduck/Corange)
