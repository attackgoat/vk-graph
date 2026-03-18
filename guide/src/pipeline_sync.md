# Synchronization

`vk-graph` provides a high-performance abstraction over Vulkan synchronization which retains the
low driver overhead of correctly synchronized command buffers.

## Pipeline Barriers

Vulkan specifies that resources and pipelines will have synchronized access when barriers are
inserted into the command stream. Unsynchronized access results in undefined behavior.

> [!TIP]
> Unsynchronized access *may* be detected through debug assertions or Vulkan SDK debugging layers.

## Access Type Abstraction

`vk-graph` uses an enumeration of possible states to define all supported pipeline barriers in an
easy-to-use way.

Sample access types:

Type|Usage
-|-
`AccessType::General`|Covers any access - useful for debug, generally avoid for performance reasons
`AccessType::ColorAttachmentWrite`|Written as a color attachment during rendering
`AccessType::ComputeShaderReadUniformBuffer`|Read as a uniform buffer in a compute shader

## Resource Access

The required access varies depending on the function being called and what the Vulkan specification
requires for a given command.

Generally, access must be specified before each command uses a resource. It appears as an "access"
function call:

```rust
graph
    .begin_cmd()
    .resource_access(some_buffer, AccessType::TransferRead)
    .resource_access(some_image, AccessType::TransferWrite)
    .record_cmd_buf(|cmd_buf| {
        // we are synchronized!
        // You may:
        //  - Read some_buffer
        //  - Write some_image
    });
```

Resource access is specified for and consumed by the following command buffer recording. For
multiple accesses, use multiple "access" and "record" function calls:

```rust
graph
    .begin_cmd()
    .resource_access(buffer, AccessType::TransferRead)
    .resource_access(image, AccessType::TransferWrite)
    .record_cmd_buf(|cmd_buf| {
        // Safe to copy buffer to image
    })
    .resource_access(image, AccessType::TransferRead)
    .resource_access(buffer, AccessType::TransferWrite)
    .record_cmd_buf(|cmd_buf| {
        // Safe to copy image to buffer
    });
```

## Shader Resource Access

When a resource (buffer, image, or acceleration structure) is accessed from a shader the
`shader_resource_access` function is used:

```glsl
// clear_image.glsl
#version 460 core
#pragma shader_stage(compute)

layout(binding = 42, rgba8) writeonly uniform image2D dstImage;

void main() {
    imageStore(
        dstImage,
        ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y),
        vec4(0)
    );
}
```

```rust
let mut graph = Graph::default();

let fmt = vk::Format::R8G8B8A8_UNORM;
let usage = vk::ImageUsageFlags::STORAGE;
let info = ImageInfo::image_2d(32, 32, fmt, usage);
let image = graph.bind_resource(Image::create(
    device,
    info,
)?);

graph
    .begin_cmd()
    .bind_pipeline(ComputePipeline::create(
        device,
        include_glsl!("clear_image.glsl").as_slice(),
    ))
    .shader_resource_access(42, image, AccessType::ComputeShaderWrite)
    .record_cmd_buf(|cmd_buf| {
        cmd_buf.dispatch(32, 32, 1);
    });
```

## Subresource Access

Buffer ranges and image views are referred to as subresource ranges and accessed using "subresource"
function variants:

```rust
let mut graph = Graph::default();

let fmt = vk::Format::R8G8B8A8_UNORM;
let usage = vk::ImageUsageFlags::STORAGE;
let info = ImageInfo::image_2d(32, 32, fmt, usage);
let image = graph.bind_resource(Image::create(
    device,
    info,
)?);

graph
    .begin_cmd()
    .bind_pipeline(ComputePipeline::create(
        device,
        include_glsl!("clear_image.glsl").as_slice(),
    ))
    .shader_subresource_access(42, image, info, AccessType::ComputeShaderWrite)
    .record_cmd_buf(|cmd_buf| {
        cmd_buf.dispatch(32, 32, 1);
    });
```

## Built-In Commands

The commands directly attached to a `Graph`, such as `Graph::copy_buffer_to_image`, do not require
any access function calls.

The source code for these built-in commands uses public graph functions and provides good examples
of typical usage.
