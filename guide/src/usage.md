# Usage

`vk-graph` acts as a safe builder-pattern for the Vulkan API.

Typical usage contains:

```rust
// A borrow of Device is an argument of many vk-graph functions
let device: &Device = &self.device;
```

## Resources

Resources, such as buffers and images, may be created from "`Info`" structs:

```rust
let usage = vk::BufferUsageFlags::TRANSFER_SRC;
let buffer_info = BufferInfo::device_mem(320 * 200 * 4, usage);
let buffer = Buffer::create(device, buffer_info)?;

let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
let image_info = ImageInfo::image_2d(320, 200, vk::Format::R8G8B8A8_UNORM, usage)?;
let image = Image::create(device, image_info)?;
```

### Memory Allocation

`vk-graph` uses a single memory allocator instance (currently `gpu-allocator`) for all allocations.

The memory allocation strategy provides a large section of memory which is then sub-allocated for
the resources which use it. This may lead to fragmentation and memory exhaustion in some scenarios.

Individual buffers or images may use dedicated memory allocations by setting their `dedicated`
field:

```rust
// The info fields may be used or set directly
let uber_mesh_buf = Buffer::create(
    device,
    BufferInfo {
        dedicated: true,
        ..buffer_info
    }
)?;

// Builder functions are also availble
// (builder and the info types are interchangable)
let dedicated_info = image_info.into_builder().dedicated(true);
let important_image = Image::create(device, dedicated_info)?;
```

Resources may be bound to a graph as `usize` handles referred to as _"nodes"_:

```rust
let mut graph = Graph::default();
let buffer: BufferNode = graph.bind_resource(buffer);
let image: ImageNode = graph.bind_resource(image);
```

Bound resources may be borrowed from graphs, commands, pipeline commands, or active command buffers using their node handle:

```rust
let shared_image: &Arc<Image> = graph.resource(image);

assert_eq!(shared_image.info.width, 320);
```

## Commands

Nodes may be used with built-in graph commands:

```rust
graph.clear_color_image(image, ClearColorValue::BLACK_ALPHA_ZERO);
```

Graphs may contain many commands:

```
graph
    .fill_buffer(buffer, 0..320 * 200, 0)
    .copy_buffer_to_image(buffer, image);
```

Custom commands enable advanced Vulkan behavior:

```rust
graph
    .begin_cmd()
    .access_resource(image, AccessType::TransferRead)
    .access_resource(buffer, AccessType::TransferWrite)
    .record_cmd_buf(move |cmd_buf| {
        // Borrow resources from nodes we move into the closure
        let buffer = cmd_buf.resource(buffer);
        let image = cmd_buf.resource(image);

        // Run *any* Vulkan code using ash::Device
        unsafe {
            // Note: for example only, use safe versions!
            cmd_buf.device.cmd_copy_image_to_buffer2(
                cmd_buf.handle,
                vk::CopyImageToBufferInfo2::default()
                    .src_image(image.handle)
                    .dst_buffer(buffer.handle),
            );
        }
    })
    .end_cmd();
```

## Pipelines

Pipelines allow shader code to execute as a graph command. A borrow of a pipeline may be bound to
record shader-stage specific commands:

```glsl
// compute.glsl
#version 460 core
#pragma shader_stage(compute)

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0, rgba8) writeonly uniform image2D dstImage;

void main() {
    imageStore(
        dstImage,
        ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y),
        vec4(0.0)
    );
}
```

```bash
# See: "Shader Compilation"
glslc compute.glsl -o compute.spv
```

```rust
let pipeline = ComputePipeline::create(
    device,
    ComputePipelineInfo::default(),
    include!("compute.spv"),
)?;

graph
    .begin_cmd()
    .bind_pipeline(&pipeline)
    .shader_resource_access(0, image, AccessType::ComputeShaderWrite)
    .record_cmd_buf(|cmd_buf| {
        cmd_buf.dispatch(320, 200, 1);
    });
```

## Queue Submission

Completed graphs are submitted to a Vulkan implementation queue for execution.

> [!NOTE]
> While executing, resources used in a graph may be bound and used by other graphs. Graph commands
> access resources in the logical state defined by all prior commands and previously submitted
> graphs.

Typical programs rely on a single `Graph` per frame and let their window implementation submit the
graph, but they may do so manually:

```rust
// NOTE: This blocks, but may run on backgound threads or checked periodically
// NOTE: Dropping the submitted queue blocks, so we must choose wait-or-async
graph
    .into_queue()
    .submit(LazyPool::new(device), 0, 0)?
    .wait_until_executed()?;
```

### Device Usage

Buffers, images, and acceleration structures resources are created and used by a single `Device`.
All commands which use a resource must execute on the same `Device` which created the resource.
