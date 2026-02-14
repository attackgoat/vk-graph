/*!

Provides a high performance [Vulkan 1.2](https://www.vulkan.org/) driver using smart pointers.
Supports windowed and headless rendering with a fully-featured render graph and resource pooling
types.

This crate allows graphics programmers to focus on acceleration structure, buffer, and image
resources and the shader pipelines they are accessed from. There are no restrictions or opinions
placed on the types of graphics algorithms you might create. Some implementations of common graphics
patterns are provided in the `contrib` directory.

# Getting Sarted

Typical usage involves creating an operating system window event loop, and rendering frames
using the provided [`FrameContext`] closure. The [`EventLoop`] builder handles creating an instance
of the [`Device`] driver, however you may construct one manually for headless rendering.

```no_run
use vk_graph_window::{Window, WindowError};

fn main() -> Result<(), WindowError> {
    let window = Window::new()?;

    // Use the device to create resources and pipelines before running
    let device = &window.device;

    window.run(|frame| {
        // You may also create resources and pipelines while running
        let device = &frame.device;
    })
}
```

# Resources and Pipelines

All resources and pipelines, as well as the driver itself, use shared reference tracking to keep
pointers alive. _vk-graph_ uses `std::sync::Arc` to track references.

## Information

All [`driver`] types have associated information structures which describe their properties.
Each object provides a `create` function which uses the information to return an instance.

| Resource                      | Create Using                                        |
|-------------------------------|-----------------------------------------------------|
| [`AccelerationStructureInfo`] | [`AccelerationStructure::create`]                   |
| [`BufferInfo`]                | [`Buffer::create`] or [`Buffer::create_from_slice`] |
| [`ImageInfo`]                 | [`Image::create`]                                   |

For example, a typical host-mappable buffer:

```no_run
# use std::sync::Arc;
# use ash::vk;
# use vk_graph::driver::DriverError;
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# fn main() -> Result<(), DriverError> {
# let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
let info = BufferInfo::host_mem(1024, vk::BufferUsageFlags::STORAGE_BUFFER);
let my_buf = Buffer::create(&device, info)?;
# Ok(()) }
```

| Pipeline                      | Create Using                                        |
|-------------------------------|-----------------------------------------------------|
| [`ComputePipelineInfo`]       | [`ComputePipeline::create`]                         |
| [`GraphicPipelineInfo`]       | [`GraphicPipeline::create`]                         |
| [`RayTracePipelineInfo`]      | [`RayTracePipeline::create`]                        |

For example, a graphics pipeline:

```no_run
# use std::sync::Arc;
# use ash::vk;
# use vk_graph::driver::DriverError;
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::graphic::{GraphicPipeline, GraphicPipelineInfo};
# use vk_graph::driver::shader::Shader;
# fn main() -> Result<(), DriverError> {
# let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
# let my_frag_code = [0u8; 1];
# let my_vert_code = [0u8; 1];
// shader code is raw SPIR-V code as bytes
let vert = Shader::new_vertex(my_vert_code.as_slice());
let frag = Shader::new_fragment(my_frag_code.as_slice());
let info = GraphicPipelineInfo::default();
let my_pipeline = GraphicPipeline::create(&device, info, [vert, frag])?;
# Ok(()) }
```

_Note:_ dtolnay's read-only public field deref pattern
(_[Link](https://github.com/dtolnay/case-studies/blob/master/readonly-fields/README.md)_) is used to
make the information of each resource easily available and immutable.

## Pooling

Multiple [`pool`] types are available to reduce the impact of frequently creating and dropping
resources. Leased resources behave identically to owned resources and can be used in a render graph.

Resource aliasing is also availble as an optional way to reduce the number of concurrent resources
that may be required.

For example, leasing an image:

```no_run
# use std::sync::Arc;
# use ash::vk;
# use vk_graph::driver::DriverError;
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::image::{ImageInfo};
# use vk_graph::pool::{Pool};
# use vk_graph::pool::lazy::{LazyPool};
# fn main() -> Result<(), DriverError> {
# let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
let mut pool = LazyPool::new(&device);

let info = ImageInfo::image_2d(8, 8, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::STORAGE);
let my_image = pool.lease(info)?;
# Ok(()) }
```

# Render Graph Operations

All rendering in _vk-graph_ is performed using a [`RenderGraph`] composed of user-specified passes,
which may include pipelines and read/write access to resources. Recorded passes are automatically
optimized before submission to the graphics hardware.

Some notes about the awesome render pass optimization which was _totally stolen_ from [Granite]:

- Scheduling: passes are submitted to the Vulkan API using batches designed for low-latency
- Re-ordering: passes are shuffled using a heuristic which gives the GPU more time to complete work
- Merging: compatible passes are merged into dynamic subpasses when it is more efficient (_on-tile
  rendering_)
- Aliasing: resources and pipelines are optimized to emit minimal barriers per unit of work (_max
  one, typically zero_)

## Nodes

Resources may be directly bound to a render graph. During the time a resource is bound we refer to
it as a node. Bound nodes may only be used with the graphs they were bound to. Nodes implement
`Copy` to make using them easier.

```no_run
# use std::sync::Arc;
# use ash::vk;
# use vk_graph::driver::DriverError;
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# use vk_graph::graph::RenderGraph;
# use vk_graph::pool::{Pool};
# use vk_graph::pool::lazy::{LazyPool};
# fn main() -> Result<(), DriverError> {
# let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
# let info = BufferInfo::host_mem(1024, vk::BufferUsageFlags::STORAGE_BUFFER);
# let buffer = Buffer::create(&device, info)?;
# let info = ImageInfo::image_2d(8, 8, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::STORAGE);
# let image = Image::create(&device, info)?;
# let mut graph = RenderGraph::new();
println!("{:?}", buffer); // Buffer
println!("{:?}", image); // Image

// Bind our resources into opaque "usize" nodes
let buffer = graph.bind_node(buffer);
let image = graph.bind_node(image);

// The results have unique types!
println!("{:?}", buffer); // BufferNode
println!("{:?}", image); // ImageNode

// Unbind nodes back into resources (Optional!)
let buffer = graph.unbind_node(buffer);
let image = graph.unbind_node(image);

// Magically, they return to the correct types! (the graph wrapped them in Arc for us)
println!("{:?}", buffer); // Arc<Buffer>
println!("{:?}", image); // Arc<Image>
# Ok(()) }
```

_Note:_ See [this code](https://github.com/attackgoat/vk-graph/blob/master/src/graph/edge.rs#L34)
for all the things that can be bound or unbound from a graph.

_Note:_ Once unbound, the node is invalid and should be dropped.

## Access and synchronization

Render graphs and their passes contain a set of functions used to handle Vulkan synchronization with
prefixes of `access`, `read`, or `write`. For each resource used in a computing, graphics subpass,
ray tracing, or general command buffer you must call an access function. Generally choose a `read`
or `write` function unless you want to be most efficient.

Example:

```no_run
# use std::sync::Arc;
# use ash::vk;
# use vk_graph::driver::DriverError;
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# use vk_graph::graph::RenderGraph;
# use vk_graph::pool::{Pool};
# use vk_graph::pool::lazy::{LazyPool};
# fn main() -> Result<(), DriverError> {
# let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
# let info = BufferInfo::host_mem(1024, vk::BufferUsageFlags::STORAGE_BUFFER);
# let buffer = Buffer::create(&device, info)?;
# let info = ImageInfo::image_2d(8, 8, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::STORAGE);
# let image = Image::create(&device, info)?;
let mut graph = RenderGraph::new();
let buffer_node = graph.bind_node(buffer);
let image_node = graph.bind_node(image);
graph
    .begin_pass("Do some raw Vulkan or interop with another Vulkan library")
    .record_cmd_buf(move |device, cmd_buf, bindings| unsafe {
        // I always run first!
    })
    .read_node(buffer_node) // <-- These two functions, read_node/write_node, completely
    .write_node(image_node) //     handle vulkan synchronization.
    .record_cmd_buf(move |device, cmd_buf, bindings| unsafe {
        // device is &ash::Device
        // cmd_buf is vk::CommandBuffer
        // bindings is a magical object you can retrieve the Vulkan resource from
        let vk_buffer: vk::Buffer = bindings[buffer_node].handle;
        let vk_image: vk::Image = bindings[image_node].handle;

        // You are free to READ vk_buffer and WRITE vk_image!
    });
# Ok(()) }
```

## Shader pipelines

Pipeline instances may be bound to a [`PassRef`] in order to execute the associated shader code:

```no_run
# use std::sync::Arc;
# use ash::vk;
# use vk_graph::driver::DriverError;
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::compute::{ComputePipeline, ComputePipelineInfo};
# use vk_graph::driver::shader::{Shader};
# use vk_graph::graph::RenderGraph;
# fn main() -> Result<(), DriverError> {
# let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
# let my_shader_code = [0u8; 1];
# let info = ComputePipelineInfo::default();
# let shader = Shader::new_compute(my_shader_code.as_slice());
# let my_compute_pipeline = Arc::new(ComputePipeline::create(&device, info, shader)?);
# let mut graph = RenderGraph::new();
graph
    .begin_pass("My compute pass")
    .bind_pipeline(&my_compute_pipeline)
    .record_compute(|compute, _| {
        compute.push_constants(&42u32.to_ne_bytes())
               .dispatch(128, 1, 1);
    });
# Ok(()) }
```

## Image samplers

By default, _vk-graph_ will use "linear repeat-mode" samplers unless a special suffix appears as
part of the name within GLSL or HLSL shader code. The `_sampler_123` suffix should be used where
`1`, `2`, and `3` are replaced with:

1. `l` for `LINEAR` texel filtering (default) or `n` for `NEAREST`
1. `l` (default) or `n`, as above, but for mipmap filtering
1. Addressing mode where:
    - `b` is `CLAMP_TO_BORDER`
    - `e` is `CLAMP_TO_EDGE`
    - `m` is `MIRRORED_REPEAT`
    - `r` is `REPEAT`

For example, the following sampler named `pages_sampler_nnr` specifies nearest texel/mipmap modes and repeat addressing:

```glsl
layout(set = 0, binding = 0) uniform sampler2D pages_sampler_nnr[NUM_PAGES];
```

For more complex image sampling, use [`ShaderBuilder::image_sampler`] to specify the exact image
sampling mode.

## Vertex input

Optional name suffixes are used in the same way with vertex input as with image samplers. The
additional attribution of your shader code is optional but may help in a few scenarios:

- Per-instance vertex rate data
- Multiple vertex buffer binding indexes

The data for vertex input is assumed to be per-vertex and bound to vertex buffer binding index zero.
Add `_ibindX` for per-instance data, or the matching `_vbindX` for per-vertex data where `X` is
replaced with the vertex buffer binding index in each case.

For more complex vertex layouts, use the [`ShaderBuilder::vertex_input`] to specify the exact
layout.

[`AccelerationStructureInfo`]: driver::accel_struct::AccelerationStructureInfo
[`AccelerationStructure::create`]: driver::accel_struct::AccelerationStructure::create
[`Buffer::create`]: driver::buffer::Buffer::create
[`Buffer::create_from_slice`]: driver::buffer::Buffer::create_from_slice
[`BufferInfo`]: driver::buffer::BufferInfo
[`ComputePipeline::create`]: driver::compute::ComputePipeline::create
[`ComputePipelineInfo`]: driver::compute::ComputePipelineInfo
[`Device`]: driver::device::Device
[`EventLoop`]: EventLoop
[`FrameContext`]: FrameContext
[Granite]: https://github.com/Themaister/Granite
[`GraphicPipeline::create`]: driver::graphic::GraphicPipeline::create
[`GraphicPipelineInfo`]: driver::graphic::GraphicPipelineInfo
[`Image::create`]: driver::image::Image::create
[`ImageInfo`]: driver::image::ImageInfo
[`PassRef`]: graph::pass_ref::PassRef
[`RayTracePipeline::create`]: driver::ray_trace::RayTracePipeline::create
[`RayTracePipelineInfo`]: driver::ray_trace::RayTracePipelineInfo
[`RenderGraph`]: graph::RenderGraph
[`ShaderBuilder::image_sampler`]: driver::shader::ShaderBuilder::image_sampler
[`ShaderBuilder::vertex_input`]: driver::shader::ShaderBuilder::vertex_input

*/

#![warn(missing_docs)]

pub mod driver;
pub mod graph;
pub mod pool;

mod display;

/// Things which are used in almost every single _vk-graph_ program.
pub mod prelude {
    pub use super::{
        display::{Display, DisplayError, DisplayInfo, DisplayInfoBuilder, ResolverPool},
        driver::{
            AccessType, CommandBuffer, DriverError, Instance,
            accel_struct::{
                AccelerationStructure, AccelerationStructureGeometry,
                AccelerationStructureGeometryData, AccelerationStructureGeometryInfo,
                AccelerationStructureInfo, AccelerationStructureInfoBuilder,
                AccelerationStructureSize, DeviceOrHostAddress,
            },
            ash::vk,
            buffer::{Buffer, BufferInfo, BufferInfoBuilder, BufferSubresourceRange},
            compute::{ComputePipeline, ComputePipelineInfo, ComputePipelineInfoBuilder},
            device::{Device, DeviceInfo, DeviceInfoBuilder},
            graphic::{
                BlendMode, BlendModeBuilder, DepthStencilMode, DepthStencilModeBuilder,
                GraphicPipeline, GraphicPipelineInfo, GraphicPipelineInfoBuilder, StencilMode,
            },
            image::{
                Image, ImageInfo, ImageInfoBuilder, ImageViewInfo, ImageViewInfoBuilder,
                SampleCount,
            },
            physical_device::{
                AccelerationStructureProperties, PhysicalDevice, RayQueryFeatures,
                RayTraceFeatures, RayTraceProperties, Vulkan10Features, Vulkan10Limits,
                Vulkan10Properties, Vulkan11Features, Vulkan11Properties, Vulkan12Features,
                Vulkan12Properties,
            },
            ray_trace::{
                RayTracePipeline, RayTracePipelineInfo, RayTracePipelineInfoBuilder,
                RayTraceShaderGroup, RayTraceShaderGroupType,
            },
            render_pass::ResolveMode,
            shader::{
                SamplerInfo, SamplerInfoBuilder, Shader, ShaderBuilder, ShaderCode,
                SpecializationInfo,
            },
            surface::Surface,
            swapchain::{
                Swapchain, SwapchainError, SwapchainImage, SwapchainInfo, SwapchainInfoBuilder,
            },
        },
        graph::{
            Bind, ClearColorValue, RenderGraph, Unbind,
            node::{
                AccelerationStructureLeaseNode, AccelerationStructureNode,
                AnyAccelerationStructureNode, AnyBufferNode, AnyImageNode, BufferLeaseNode,
                BufferNode, ImageLeaseNode, ImageNode, SwapchainImageNode,
            },
            pass_ref::{PassRef, PipelinePassRef},
        },
        pool::{
            Lease, Pool, PoolInfo, PoolInfoBuilder,
            alias::{Alias, AliasPool},
            fifo::FifoPool,
            hash::HashPool,
            lazy::LazyPool,
        },
    };
}

pub use self::display::{Display, DisplayError, DisplayInfo, DisplayInfoBuilder, ResolverPool};
