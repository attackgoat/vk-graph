# Usage

`vk-graph` acts as a safe builder-pattern for the Vulkan API.

API docs: [`Graph`](https://docs.rs/vk-graph/latest/vk_graph/struct.Graph.html),
[`Device`](https://docs.rs/vk-graph/latest/vk_graph/driver/device/struct.Device.html),
[`Graph::begin_cmd`](https://docs.rs/vk-graph/latest/vk_graph/struct.Graph.html#method.begin_cmd),
[`Graph::bind_resource`](https://docs.rs/vk-graph/latest/vk_graph/struct.Graph.html#method.bind_resource),
[`Graph::resource`](https://docs.rs/vk-graph/latest/vk_graph/struct.Graph.html#method.resource),
[`Graph::finalize`](https://docs.rs/vk-graph/latest/vk_graph/struct.Graph.html#method.finalize).

Typical usage contains:

```rust
# use vk_graph::driver::DriverError;
# struct Foo { device: vk_graph::driver::device::Device }
# impl Foo {
# fn test(
#     &self,
# ) {
use vk_graph::driver::ash::vk;
use vk_graph::driver::device::Device;

// A borrow of Device is an argument of many vk-graph functions
let device: &Device = &self.device;
# } }
```

## Resources

Resources, such as buffers and images, may be created from "`Info`" structs:

```rust
# use vk_graph::Graph;
# use vk_graph::driver::{DriverError, ash::vk, device::Device, sync::AccessType};
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# fn test(
#     device: &Device,
# ) -> Result<(), DriverError> {
let usage = vk::BufferUsageFlags::TRANSFER_SRC;
let buffer_info = BufferInfo::device_mem(320 * 200 * 4, usage);
let buffer = Buffer::create(device, buffer_info)?;

let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
let image_info = ImageInfo::image_2d(320, 200, vk::Format::R8G8B8A8_UNORM, usage);
let image = Image::create(device, image_info)?;
# Ok(()) }
```

### Memory Allocation

`vk-graph` uses an external memory allocator (currently `gpu-allocator`) for resource memory
allocations.

The allocation strategy provides a large section of memory which is then sub-allocated for any
resources which use it. This may lead to fragmentation and memory exhaustion in some scenarios.

Individual buffers or images may use dedicated memory allocations by setting their `alloc_dedicated`
field:

```rust
# use vk_graph::Graph;
# use vk_graph::driver::{DriverError, ash::vk, device::Device, sync::AccessType};
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# fn test(
#     device: &Device,
# ) -> Result<(), DriverError> {
# let buffer_info = BufferInfo::device_mem(1, vk::BufferUsageFlags::empty());
# let image_info = ImageInfo::image_2d(32, 32, vk::Format::R16_UNORM, vk::ImageUsageFlags::empty());
// The info fields may be used or set directly
let uber_mesh_buf = Buffer::create(
    device,
    BufferInfo {
        alloc_dedicated: true,
        ..buffer_info
    }
)?;

// Builder functions are also available
// (builder and info types are interchangeable)
let dedicated_info = image_info.into_builder().alloc_dedicated(true);
let important_image = Image::create(device, dedicated_info)?;
# Ok(()) }
```

Resources may be bound to a graph as typed node handles referred to as _"nodes"_:

```rust
# use vk_graph::Graph;
# use vk_graph::driver::{DriverError, ash::vk, device::Device, sync::AccessType};
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# use vk_graph::node::{BufferNode, ImageNode};
# fn test(
#     device: &Device,
#     buffer: Buffer,
#     image: Image,
# ) -> Result<(), DriverError> {
let mut graph = Graph::default();
let buffer: BufferNode = graph.bind_resource(buffer);
let image: ImageNode = graph.bind_resource(image);
# Ok(()) }
```

Bound resources may be borrowed from graphs, commands, pipeline commands, or command buffers using
their node handle:

```rust
# use vk_graph::Graph;
# use vk_graph::driver::image::Image;
# use vk_graph::node::ImageNode;
# use std::sync::Arc;
# fn test(
#     graph: &mut Graph,
#     image: ImageNode,
# ) {
let shared_image: &Arc<Image> = graph.resource(image);

assert_eq!(shared_image.info.width, 320);
# }
```

Concrete node types return the exact stored handle type. For example, `ImageNode` returns
`&Arc<Image>`. Erased node types such as `AnyImageNode` instead return `&Image` so they can unify
owned, leased, and swapchain-backed resources behind one view.

## Commands

Nodes may be used with built-in graph commands:

```rust
# use vk_graph::Graph;
# use vk_graph::cmd::ClearColorValue;
# use vk_graph::node::ImageNode;
# use std::sync::Arc;
# fn test(
#     graph: &mut Graph,
#     image: ImageNode,
# ) {
graph.clear_color_image(image, ClearColorValue::BLACK_ALPHA_ZERO);
# }
```

Graphs may contain many commands:

```rust
# use vk_graph::Graph;
# use vk_graph::cmd::ClearColorValue;
# use vk_graph::node::{BufferNode, ImageNode};
# use std::sync::Arc;
# fn test(
#     graph: &mut Graph,
#     buffer: BufferNode,
#     image: ImageNode,
# ) {
graph
    .fill_buffer(buffer, 0..320 * 200, 0)
    .copy_buffer_to_image(buffer, image);
# }
```

Custom commands enable advanced Vulkan behavior:

```rust
# use vk_graph::Graph;
# use vk_graph::cmd::ClearColorValue;
# use vk_graph::driver::{ash::vk, sync::AccessType};
# use vk_graph::node::{BufferNode, ImageNode};
# use std::sync::Arc;
# fn test(
#     graph: &mut Graph,
#     buffer: BufferNode,
#     image: ImageNode,
# ) {
graph
    .begin_cmd()
    .resource_access(image, AccessType::TransferRead)
    .resource_access(buffer, AccessType::TransferWrite)
    .record_cmd(move |cmd| {
        // Borrow resources from nodes we move into the closure
        let buffer = cmd.resource(buffer);
        let image = cmd.resource(image);

        // Run *any* Vulkan code using ash::Device
        unsafe {
            // Note: for example only, use safe versions!
            cmd.device.cmd_copy_image_to_buffer2(
                cmd.handle,
                &vk::CopyImageToBufferInfo2::default()
                    .src_image(image.handle)
                    .dst_buffer(buffer.handle),
            );
        }
    })
    .end_cmd();
# }
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
# macro_rules! include_bytes { ($path:expr) => { [0u8] }; }
# use vk_graph::Graph;
# use vk_graph::driver::{DriverError, device::Device, sync::AccessType};
# use vk_graph::driver::compute::{ComputePipeline, ComputePipelineInfo};
# use vk_graph::node::ImageNode;
# fn test(
#     graph: &mut Graph,
#     device: &Device,
#     image: ImageNode,
# ) -> Result<(), DriverError> {
let pipeline = ComputePipeline::create(
    device,
    ComputePipelineInfo::default(),
    include_bytes!("compute.spv").as_slice(),
)?;

graph
    .begin_cmd()
    .bind_pipeline(&pipeline)
    .shader_resource_access(0, image, AccessType::ComputeShaderWrite)
    .record_cmd(|cmd| {
        cmd.dispatch(320, 200, 1);
    });
# Ok(()) }
```

## Queue Submission

Completed graphs are queued for execution by a Vulkan implementation.

> [!NOTE]
> While executing, resources used in a graph may be bound and used by other graphs. Graph commands
> access resources in the logical state defined by all prior commands and previously submitted
> graphs.

Typical programs rely on a single `Graph` per frame and let their window implementation submit the
graph, but they may do so manually:

```rust
# use vk_graph::Graph;
# use vk_graph::driver::{DriverError, device::Device};
# use vk_graph::pool::lazy::LazyPool;
# fn test(
#     graph: Graph,
#     device: &Device,
# ) -> Result<(), DriverError> {
// NOTE: This will stall! Use Fence::is_signaled to check periodically instead.
let mut fence = graph
    .finalize()
    .queue_submit(&mut LazyPool::new(device), 0, 0)?;
fence.wait_signaled()?;
# Ok(()) }
```

### Device Usage

Buffers, images, and acceleration structure resources are created and used by a single `Device`.
All commands which use a resource must execute on the same `Device` which created the resource.
