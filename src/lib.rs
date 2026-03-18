/*!

This crate provides a high-performance [Vulkan](https://www.vulkan.org/) driver featuring automated
resource management and execution.

For a general overview, including installation and typical usage, see the
[Guide Book](https://attackgoat.github.io/vk-graph).

# Getting Sarted

Typical usage begins by displaying a winit [`Window`()]. The provided example code displays a window
and creates Vulkan [`Device`] driver automatically:

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

## _Optional_: Headless Rendering

```no_run
use vk_graph::driver::{device::{Device, DeviceInfo}, DriverError};

fn main() -> Result<(), DriverError> {
    let device = Device::new(DeviceInfo::default())?;

    // Do stuff...
    # Ok(())
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
# let device = Device::new(DeviceInfo::default())?;
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
# let device = Device::new(DeviceInfo::default())?;
# let my_frag_code = [0u8; 1];
# let my_vert_code = [0u8; 1];
// shader code is SPIR-V in u32 format
let vert = Shader::new_vertex(my_vert_code.as_slice());
let frag = Shader::new_fragment(my_frag_code.as_slice());
let info = GraphicPipelineInfo::default();
let my_pipeline = GraphicPipeline::create(&device, info, [vert, frag])?;
# Ok(()) }
```

_Note:_ dtolnay's read-only public field deref pattern
(_[link](https://github.com/dtolnay/case-studies/blob/master/readonly-fields/README.md)_) is used to
make the information of each resource easily available and immutable.

```no_run
# use std::sync::Arc;
# use ash::vk;
# use vk_graph::driver::DriverError;
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# fn main() -> Result<(), DriverError> {
# let device = Device::new(DeviceInfo::default())?;
let info = ImageInfo::image_2d(8, 8, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::empty());
let my_image = Image::create(&device, info)?;

// Note: info is a field provided through the Deref trait and is immutable!
assert_eq!(8, my_image.info.width);
# Ok(()) }
```

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
# let device = Device::new(DeviceInfo::default())?;
let mut pool = LazyPool::new(&device);

let info = ImageInfo::image_2d(8, 8, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::STORAGE);
let my_image = pool.lease_resource(info)?;
# Ok(()) }
```

# Render Graph Operations

All rendering in _vk-graph_ is performed using a [`Graph`] composed of user-specified passes,
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
# use vk_graph::Graph;
# use vk_graph::pool::{Pool};
# use vk_graph::pool::lazy::{LazyPool};
# fn main() -> Result<(), DriverError> {
# let device = Device::new(DeviceInfo::default())?;
# let info = BufferInfo::host_mem(1024, vk::BufferUsageFlags::STORAGE_BUFFER);
# let buffer = Buffer::create(&device, info)?;
# let info = ImageInfo::image_2d(8, 8, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::STORAGE);
# let image = Image::create(&device, info)?;
# let mut graph = Graph::default();
println!("{:?}", buffer); // Buffer
println!("{:?}", image); // Image

// Bind our resources into opaque "usize" nodes
let buffer = graph.bind_resource(buffer);
let image = graph.bind_resource(image);

// The results have unique types!
println!("{:?}", buffer); // BufferNode
println!("{:?}", image); // ImageNode

// Borrow resources using nodes (Optional!)
println!("{:?}", graph.resource(buffer)); // &Arc<Buffer>
println!("{:?}", graph.resource(image)); // &Arc<Image>
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
# use vk_graph::driver::sync::AccessType;
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# use vk_graph::Graph;
# use vk_graph::node::{BufferNode, ImageNode};
# use vk_graph::pool::{Pool};
# use vk_graph::pool::lazy::{LazyPool};
# fn main() -> Result<(), DriverError> {
# let device = Device::new(DeviceInfo::default())?;
# let info = BufferInfo::host_mem(1024, vk::BufferUsageFlags::STORAGE_BUFFER);
# let buffer = Buffer::create(&device, info)?;
# let info = ImageInfo::image_2d(8, 8, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::STORAGE);
# let image = Image::create(&device, info)?;
let mut graph = Graph::default();
let buffer: BufferNode = graph.bind_resource(buffer);
let image: ImageNode = graph.bind_resource(image);
graph
    .begin_cmd()
    .debug_name("Do some raw Vulkan or interop with another Vulkan library")
    .record_cmd_buf(|cmd_buf| {
        // I always run first!
    })
    .resource_access(buffer, AccessType::HostRead)
    .resource_access(image, AccessType::HostWrite)
    .record_cmd_buf(move |cmd_buf| {
        // cmd_buf allows you to borrow the Vulkan resources
        let buffer: vk::Buffer = cmd_buf.resource(buffer).handle;
        let image: vk::Image = cmd_buf.resource(image).handle;

        // Raw ash types are also available
        let device: &ash::Device = &cmd_buf.device;
        let cmd_buf: vk::CommandBuffer = cmd_buf.handle;

        // You are free to READ buffer and WRITE image!
    });
# Ok(()) }
```

## Shader pipelines

Pipeline instances may be bound to a [`PassRef`] in order to execute the associated shader code:

```no_run
# use ash::vk;
# use vk_graph::driver::DriverError;
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::compute::{ComputePipeline, ComputePipelineInfo};
# use vk_graph::driver::shader::{Shader};
# use vk_graph::Graph;
# fn main() -> Result<(), DriverError> {
# let device = Device::new(DeviceInfo::default())?;
# let my_shader_code = [0u8; 1];
# let info = ComputePipelineInfo::default();
# let shader = Shader::new_compute(my_shader_code.as_slice());
# let my_compute_pipeline = ComputePipeline::create(&device, info, shader)?;
# let mut graph = Graph::default();
graph
    .begin_cmd()
    .debug_name("My compute pass")
    .bind_pipeline(&my_compute_pipeline)
    .record_cmd_buf(|cmd_buf| {
        cmd_buf.push_constants(0, &42u32.to_ne_bytes())
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
[`Graph`]: graph::Graph
[`ShaderBuilder::image_sampler`]: driver::shader::ShaderBuilder::image_sampler
[`ShaderBuilder::vertex_input`]: driver::shader::ShaderBuilder::vertex_input

*/

#![warn(missing_docs)]

pub mod cmd;
pub mod driver;
pub mod node;
pub mod pool;

mod queue;
mod resource;

use crate::cmd::CommandBufferRef;

pub use self::{
    queue::Queue,
    resource::{GraphResource, Resource},
};

#[allow(deprecated)]
pub use self::deprecated::{Display, DisplayInfo, DisplayInfoBuilder};

use {
    self::{
        cmd::{AttachmentIndex, CommandRef, Descriptor, SubresourceAccess, ViewInfo},
        node::Node,
        node::{
            AccelerationStructureLeaseNode, AccelerationStructureNode,
            AnyAccelerationStructureNode, AnyBufferNode, AnyImageNode, BufferLeaseNode, BufferNode,
            ImageLeaseNode, ImageNode, SwapchainImageNode,
        },
    },
    crate::driver::{
        DescriptorBindingMap,
        compute::ComputePipeline,
        format_aspect_mask, format_texel_block_extent, format_texel_block_size,
        graphic::{DepthStencilInfo, GraphicPipeline},
        image::{ImageInfo, ImageViewInfo, SampleCount},
        image_subresource_range_from_layers,
        ray_trace::RayTracePipeline,
        render_pass::ResolveMode,
        shader::PipelineDescriptorInfo,
    },
    ash::vk,
    std::{
        cmp::Ord,
        collections::{BTreeMap, HashMap},
        fmt::{Debug, Formatter},
        ops::Range,
    },
    vk_sync::AccessType,
};

type ExecFn = Box<dyn FnOnce(CommandBufferRef) + Send>;
type NodeIndex = usize;

#[derive(Clone, Copy, Debug)]
struct Attachment {
    array_layer_count: u32,
    aspect_mask: vk::ImageAspectFlags,
    base_array_layer: u32,
    base_mip_level: u32,
    format: vk::Format,
    mip_level_count: u32,
    sample_count: SampleCount,
    target: NodeIndex,
}

impl Attachment {
    fn new(image_view_info: ImageViewInfo, sample_count: SampleCount, target: NodeIndex) -> Self {
        Self {
            array_layer_count: image_view_info.array_layer_count,
            aspect_mask: image_view_info.aspect_mask,
            base_array_layer: image_view_info.base_array_layer,
            base_mip_level: image_view_info.base_mip_level,
            format: image_view_info.fmt,
            mip_level_count: image_view_info.mip_level_count,
            sample_count,
            target,
        }
    }

    fn are_compatible(lhs: Option<Self>, rhs: Option<Self>) -> bool {
        // Two attachment references are compatible if they have matching format and sample
        // count, or are both VK_ATTACHMENT_UNUSED or the pointer that would contain the
        // reference is NULL.
        if lhs.is_none() || rhs.is_none() {
            return true;
        }

        Self::are_identical(lhs.unwrap(), rhs.unwrap())
    }

    fn are_identical(lhs: Self, rhs: Self) -> bool {
        lhs.array_layer_count == rhs.array_layer_count
            && lhs.base_array_layer == rhs.base_array_layer
            && lhs.base_mip_level == rhs.base_mip_level
            && lhs.format == rhs.format
            && lhs.mip_level_count == rhs.mip_level_count
            && lhs.sample_count == rhs.sample_count
            && lhs.target == rhs.target
    }

    fn image_view_info(self, image_info: ImageInfo) -> ImageViewInfo {
        image_info
            .into_builder()
            .array_layer_count(self.array_layer_count)
            .mip_level_count(self.mip_level_count)
            .fmt(self.format)
            .into_image_view()
            .aspect_mask(self.aspect_mask)
            .base_array_layer(self.base_array_layer)
            .base_mip_level(self.base_mip_level)
            .build()
    }
}

/// TODO
#[derive(Clone, Copy, Debug)]
pub enum ClearColorValue {
    /// Value as [f32].
    Float32([f32; 4]),

    /// Value as [i32].
    Int32([i32; 4]),

    /// Value as [u32].
    Uint32([u32; 4]),
}

impl ClearColorValue {
    /// rgb zeros and alpha ones.
    pub const BLACK_ALPHA_ONE: Self = Self::Float32([0.0, 0.0, 0.0, 1.0]);

    /// zeros.
    pub const BLACK_ALPHA_ZERO: Self = Self::Float32([0.0, 0.0, 0.0, 0.0]);

    /// rgb zeros and alpha ones.
    pub const WHITE_ALPHA_ONE: Self = Self::Float32([1.0, 1.0, 1.0, 1.0]);

    /// rgb ones and alpha zeros.
    pub const WHITE_ALPHA_ZERO: Self = Self::Float32([1.0, 1.0, 1.0, 0.0]);

    /// Convenience constructor for clear color values.
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self::Float32([r, g, b, a])
    }

    /// Convert RGB+A values into a ClearColorValue.
    pub const fn from_f32(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self::rgba(r, g, b, a)
    }

    /// Convert RGB+A values into a ClearColorValue.
    pub const fn from_u8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self::from_f32(
            r as f32 / u8::MAX as f32,
            g as f32 / u8::MAX as f32,
            b as f32 / u8::MAX as f32,
            a as f32 / u8::MAX as f32,
        )
    }
}

impl Default for ClearColorValue {
    fn default() -> Self {
        Self::from_f32(0.0, 0.0, 0.0, 0.0)
    }
}

impl From<[f32; 4]> for ClearColorValue {
    fn from(float32: [f32; 4]) -> Self {
        Self::Float32(float32)
    }
}

impl From<[i32; 4]> for ClearColorValue {
    fn from(int32: [i32; 4]) -> Self {
        Self::Int32(int32)
    }
}

impl From<[u8; 4]> for ClearColorValue {
    fn from(uint8: [u8; 4]) -> Self {
        Self::from_u8(uint8[0], uint8[1], uint8[2], uint8[3])
    }
}

impl From<[u32; 4]> for ClearColorValue {
    fn from(uint32: [u32; 4]) -> Self {
        Self::Uint32(uint32)
    }
}

impl From<ClearColorValue> for vk::ClearColorValue {
    fn from(value: ClearColorValue) -> Self {
        match value {
            ClearColorValue::Float32(float32) => Self { float32 },
            ClearColorValue::Int32(int32) => Self { int32 },
            ClearColorValue::Uint32(uint32) => Self { uint32 },
        }
    }
}

#[derive(Default)]
struct Execution {
    accesses: HashMap<NodeIndex, Vec<SubresourceAccess>>,
    bindings: BTreeMap<Descriptor, (NodeIndex, ViewInfo)>,

    correlated_view_mask: u32,
    depth_stencil: Option<DepthStencilInfo>,
    render_area: Option<vk::Rect2D>,
    view_mask: u32,

    color_attachments: HashMap<AttachmentIndex, Attachment>,
    color_clears: HashMap<AttachmentIndex, (Attachment, [f32; 4])>,
    color_loads: HashMap<AttachmentIndex, Attachment>,
    color_resolves: HashMap<AttachmentIndex, (Attachment, AttachmentIndex)>,
    color_stores: HashMap<AttachmentIndex, Attachment>,
    depth_stencil_attachment: Option<Attachment>,
    depth_stencil_clear: Option<(Attachment, vk::ClearDepthStencilValue)>,
    depth_stencil_load: Option<Attachment>,
    depth_stencil_resolve: Option<(
        Attachment,
        AttachmentIndex,
        Option<ResolveMode>,
        Option<ResolveMode>,
    )>,
    depth_stencil_store: Option<Attachment>,

    func: Option<ExecutionFunction>,
    pipeline: Option<ExecutionPipeline>,
}

impl Debug for Execution {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // The only field missing is func which cannot easily be implemented because it is a
        // FnOnce.
        f.debug_struct("Execution")
            .field("accesses", &self.accesses)
            .field("bindings", &self.bindings)
            .field("depth_stencil", &self.depth_stencil)
            .field("color_attachments", &self.color_attachments)
            .field("color_clears", &self.color_clears)
            .field("color_loads", &self.color_loads)
            .field("color_resolves", &self.color_resolves)
            .field("color_stores", &self.color_stores)
            .field("depth_stencil_attachment", &self.depth_stencil_attachment)
            .field("depth_stencil_clear", &self.depth_stencil_clear)
            .field("depth_stencil_load", &self.depth_stencil_load)
            .field("depth_stencil_resolve", &self.depth_stencil_resolve)
            .field("depth_stencil_store", &self.depth_stencil_store)
            .field("pipeline", &self.pipeline)
            .finish()
    }
}

struct ExecutionFunction(ExecFn);

#[derive(Clone, Debug)]
enum ExecutionPipeline {
    Compute(ComputePipeline),
    Graphic(GraphicPipeline),
    RayTrace(RayTracePipeline),
}

impl ExecutionPipeline {
    fn as_graphic(&self) -> Option<&GraphicPipeline> {
        if let Self::Graphic(pipeline) = self {
            Some(pipeline)
        } else {
            None
        }
    }

    fn bind_point(&self) -> vk::PipelineBindPoint {
        match self {
            ExecutionPipeline::Compute(_) => vk::PipelineBindPoint::COMPUTE,
            ExecutionPipeline::Graphic(_) => vk::PipelineBindPoint::GRAPHICS,
            ExecutionPipeline::RayTrace(_) => vk::PipelineBindPoint::RAY_TRACING_KHR,
        }
    }

    fn descriptor_bindings(&self) -> &DescriptorBindingMap {
        match self {
            ExecutionPipeline::Compute(pipeline) => &pipeline.inner.descriptor_bindings,
            ExecutionPipeline::Graphic(pipeline) => &pipeline.inner.descriptor_bindings,
            ExecutionPipeline::RayTrace(pipeline) => &pipeline.inner.descriptor_bindings,
        }
    }

    fn descriptor_info(&self) -> &PipelineDescriptorInfo {
        match self {
            ExecutionPipeline::Compute(pipeline) => &pipeline.inner.descriptor_info,
            ExecutionPipeline::Graphic(pipeline) => &pipeline.inner.descriptor_info,
            ExecutionPipeline::RayTrace(pipeline) => &pipeline.inner.descriptor_info,
        }
    }

    fn layout(&self) -> vk::PipelineLayout {
        match self {
            ExecutionPipeline::Compute(pipeline) => pipeline.inner.layout,
            ExecutionPipeline::Graphic(pipeline) => pipeline.inner.layout,
            ExecutionPipeline::RayTrace(pipeline) => pipeline.inner.layout,
        }
    }

    fn stage(&self) -> vk::PipelineStageFlags {
        match self {
            ExecutionPipeline::Compute(_) => vk::PipelineStageFlags::COMPUTE_SHADER,
            ExecutionPipeline::Graphic(_) => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ExecutionPipeline::RayTrace(_) => vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        }
    }
}

#[derive(Debug)]
struct Command {
    execs: Vec<Execution>,
    name: Option<String>,
}

impl Command {
    fn descriptor_pools_sizes(
        &self,
    ) -> impl Iterator<Item = &HashMap<u32, HashMap<vk::DescriptorType, u32>>> {
        self.execs
            .iter()
            .flat_map(|exec| exec.pipeline.as_ref())
            .map(|pipeline| &pipeline.descriptor_info().pool_sizes)
    }

    fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("command")
    }
}

/// A composable graph of render pass operations.
///
/// `Graph` instances are are intended for one-time use.
///
/// The design of this code originated with a combination of
/// [`PassBuilder`](https://github.com/EmbarkStudios/kajiya/blob/main/crates/lib/kajiya-rg/src/pass_builder.rs)
/// and
/// [`graph.cpp`](https://github.com/Themaister/Granite/blob/master/renderer/graph.cpp).
#[derive(Debug, Default)]
pub struct Graph {
    cmds: Vec<Command>,
    resources: Vec<Resource>,
}

impl Graph {
    /// Constructs a default `Graph`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocates and begins writing a new command.
    pub fn begin_cmd(&mut self) -> CommandRef<'_> {
        CommandRef::new(self)
    }

    /// Binds a Vulkan buffer, image, or acceleration structure resource to this graph.
    ///
    /// Bound resource nodes may be used in commands for shader pipeline operations and other
    /// general functions.
    pub fn bind_resource<R>(&mut self, resource: R) -> R::Node
    where
        R: GraphResource,
    {
        resource.bind_graph(self)
    }

    /// Copy an image, potentially performing format conversion.
    pub fn blit_image(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyImageNode>,
        filter: vk::Filter,
    ) -> &mut Self {
        let src = src.into();
        let src_info = self.resource(src).info;

        let dst = dst.into();
        let dst_info = self.resource(dst).info;

        self.blit_image_region(
            src,
            dst,
            filter,
            [vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(src_info.fmt),
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: src_info.width as _,
                        y: src_info.height as _,
                        z: src_info.depth as _,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(dst_info.fmt),
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: dst_info.width as _,
                        y: dst_info.height as _,
                        z: dst_info.depth as _,
                    },
                ],
            }],
        )
    }

    /// Copy regions of an image, potentially performing format conversion.
    #[profiling::function]
    pub fn blit_image_region(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyImageNode>,
        filter: vk::Filter,
        regions: impl AsRef<[vk::ImageBlit]> + 'static + Send,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();

        let mut cmd = self.begin_cmd().debug_name("blit image");

        for region in regions.as_ref() {
            let src_region = image_subresource_range_from_layers(region.src_subresource);
            cmd.set_subresource_access(src, src_region, AccessType::TransferRead);

            let dst_region = image_subresource_range_from_layers(region.dst_subresource);
            cmd.set_subresource_access(dst, dst_region, AccessType::TransferWrite);
        }

        cmd.record_cmd_buf(move |cmd_buf| {
            let src_image = cmd_buf.resource(src).handle;
            let dst_image = cmd_buf.resource(dst).handle;

            unsafe {
                cmd_buf.device.cmd_blit_image(
                    cmd_buf.handle,
                    src_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions.as_ref(),
                    filter,
                );
            }
        })
        .end_cmd()
    }

    /// Clear a color image.
    #[profiling::function]
    pub fn clear_color_image(
        &mut self,
        image: impl Into<AnyImageNode>,
        color: impl Into<ClearColorValue>,
    ) -> &mut Self {
        let color = color.into().into();
        let image = image.into();
        let image_view = self.resource(image).info.into();

        self.begin_cmd()
            .debug_name("clear color")
            .subresource_access(image, image_view, AccessType::TransferWrite)
            .record_cmd_buf(move |cmd_buf| {
                let image = cmd_buf.resource(image);

                unsafe {
                    cmd_buf.device.cmd_clear_color_image(
                        cmd_buf.handle,
                        image.handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &color,
                        &[image_view],
                    );
                }
            })
            .end_cmd()
    }

    /// Clears a depth/stencil image.
    #[profiling::function]
    pub fn clear_depth_stencil_image(
        &mut self,
        image: impl Into<AnyImageNode>,
        depth: f32,
        stencil: u32,
    ) -> &mut Self {
        let image = image.into();
        let image_view = self.resource(image).info.into();

        self.begin_cmd()
            .debug_name("clear depth/stencil")
            .subresource_access(image, image_view, AccessType::TransferWrite)
            .record_cmd_buf(move |cmd_buf| {
                let image = cmd_buf.resource(image);

                unsafe {
                    cmd_buf.device.cmd_clear_depth_stencil_image(
                        cmd_buf.handle,
                        image.handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &vk::ClearDepthStencilValue { depth, stencil },
                        &[image_view],
                    );
                }
            })
            .end_cmd()
    }

    /// Copy data between buffers
    pub fn copy_buffer(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyBufferNode>,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();
        let src_info = self.resource(src).info;
        let dst_info = self.resource(dst).info;

        self.copy_buffer_region(
            src,
            dst,
            [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: src_info.size.min(dst_info.size),
            }],
        )
    }

    /// Copy data between buffer regions.
    #[profiling::function]
    pub fn copy_buffer_region(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyBufferNode>,
        regions: impl AsRef<[vk::BufferCopy]> + 'static + Send,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();

        #[cfg(debug_assertions)]
        let src_size = self.resource(src).info.size;

        #[cfg(debug_assertions)]
        let dst_size = self.resource(dst).info.size;

        let mut cmd = self.begin_cmd().debug_name("copy buffer");

        for region in regions.as_ref() {
            #[cfg(debug_assertions)]
            {
                assert!(
                    region.src_offset + region.size <= src_size,
                    "source range end ({}) exceeds source size ({src_size})",
                    region.src_offset + region.size
                );
                assert!(
                    region.dst_offset + region.size <= dst_size,
                    "destination range end ({}) exceeds destination size ({dst_size})",
                    region.dst_offset + region.size
                );
            };

            cmd.set_subresource_access(
                src,
                region.src_offset..region.src_offset + region.size,
                AccessType::TransferRead,
            );
            cmd.set_subresource_access(
                dst,
                region.dst_offset..region.dst_offset + region.size,
                AccessType::TransferWrite,
            );
        }

        cmd.record_cmd_buf(move |cmd_buf| {
            let src = cmd_buf.resource(src);
            let dst = cmd_buf.resource(dst);

            unsafe {
                cmd_buf.device.cmd_copy_buffer(
                    cmd_buf.handle,
                    src.handle,
                    dst.handle,
                    regions.as_ref(),
                );
            }
        })
        .end_cmd()
    }

    /// Copy data from a buffer into an image.
    pub fn copy_buffer_to_image(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyImageNode>,
    ) -> &mut Self {
        let dst = dst.into();
        let dst_info = self.resource(dst).info;

        self.copy_buffer_to_image_region(
            src,
            dst,
            [vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: dst_info.width,
                buffer_image_height: dst_info.height,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(dst_info.fmt),
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: Default::default(),
                image_extent: vk::Extent3D {
                    depth: dst_info.depth,
                    height: dst_info.height,
                    width: dst_info.width,
                },
            }],
        )
    }

    /// Copy data from a buffer into an image.
    #[profiling::function]
    pub fn copy_buffer_to_image_region(
        &mut self,
        src: impl Into<AnyBufferNode>,
        dst: impl Into<AnyImageNode>,
        regions: impl AsRef<[vk::BufferImageCopy]> + 'static + Send,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();
        let dst_info = self.resource(dst).info;

        let mut cmd = self.begin_cmd().debug_name("copy buffer to image");

        for region in regions.as_ref() {
            let block_bytes_size = format_texel_block_size(dst_info.fmt);
            let (block_height, block_width) = format_texel_block_extent(dst_info.fmt);
            let data_size = block_bytes_size
                * (region.buffer_row_length / block_width)
                * (region.buffer_image_height / block_height);

            cmd.set_subresource_access(
                src,
                region.buffer_offset..region.buffer_offset + data_size as vk::DeviceSize,
                AccessType::TransferRead,
            );
            cmd.set_subresource_access(
                dst,
                image_subresource_range_from_layers(region.image_subresource),
                AccessType::TransferWrite,
            );
        }

        cmd.record_cmd_buf(move |cmd_buf| {
            let src = cmd_buf.resource(src);
            let dst = cmd_buf.resource(dst);

            unsafe {
                cmd_buf.device.cmd_copy_buffer_to_image(
                    cmd_buf.handle,
                    src.handle,
                    dst.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions.as_ref(),
                );
            }
        })
        .end_cmd()
    }

    /// Copy all layers of a source image to a destination image.
    pub fn copy_image(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyImageNode>,
    ) -> &mut Self {
        let src = src.into();
        let src_info = self.resource(src).info;

        let dst = dst.into();
        let dst_info = self.resource(dst).info;

        self.copy_image_region(
            src,
            dst,
            [vk::ImageCopy {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(src_info.fmt),
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: src_info.array_layer_count,
                },
                src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(dst_info.fmt),
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: src_info.array_layer_count,
                },
                dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                extent: vk::Extent3D {
                    depth: src_info.depth.clamp(1, dst_info.depth),
                    height: src_info.height.clamp(1, dst_info.height),
                    width: src_info.width.min(dst_info.width),
                },
            }],
        )
    }

    /// Copy data between images.
    #[profiling::function]
    pub fn copy_image_region(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyImageNode>,
        regions: impl AsRef<[vk::ImageCopy]> + 'static + Send,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();

        let mut cmd = self.begin_cmd().debug_name("copy image");

        for region in regions.as_ref() {
            cmd.set_subresource_access(
                src,
                image_subresource_range_from_layers(region.src_subresource),
                AccessType::TransferRead,
            );
            cmd.set_subresource_access(
                dst,
                image_subresource_range_from_layers(region.dst_subresource),
                AccessType::TransferWrite,
            );
        }

        cmd.record_cmd_buf(move |cmd_buf| {
            let src = cmd_buf.resource(src);
            let dst = cmd_buf.resource(dst);

            unsafe {
                cmd_buf.device.cmd_copy_image(
                    cmd_buf.handle,
                    src.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions.as_ref(),
                );
            }
        })
        .end_cmd()
    }

    /// Copy image data into a buffer.
    pub fn copy_image_to_buffer(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyBufferNode>,
    ) -> &mut Self {
        let src = src.into();
        let dst = dst.into();

        let src_info = self.resource(src).info;

        self.copy_image_to_buffer_region(
            src,
            dst,
            [vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: src_info.width,
                buffer_image_height: src_info.height,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: format_aspect_mask(src_info.fmt),
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: Default::default(),
                image_extent: vk::Extent3D {
                    depth: src_info.depth,
                    height: src_info.height,
                    width: src_info.width,
                },
            }],
        )
    }

    /// Copy image data into a buffer.
    #[profiling::function]
    pub fn copy_image_to_buffer_region(
        &mut self,
        src: impl Into<AnyImageNode>,
        dst: impl Into<AnyBufferNode>,
        regions: impl AsRef<[vk::BufferImageCopy]> + 'static + Send,
    ) -> &mut Self {
        let src = src.into();
        let src_info = self.resource(src).info;
        let dst = dst.into();

        let mut cmd = self.begin_cmd().debug_name("copy image to buffer");

        for region in regions.as_ref() {
            let block_bytes_size = format_texel_block_size(src_info.fmt);
            let (block_height, block_width) = format_texel_block_extent(src_info.fmt);
            let data_size = block_bytes_size
                * (region.buffer_row_length / block_width)
                * (region.buffer_image_height / block_height);

            cmd.set_subresource_access(
                src,
                image_subresource_range_from_layers(region.image_subresource),
                AccessType::TransferRead,
            );
            cmd.set_subresource_access(
                dst,
                region.buffer_offset..region.buffer_offset + data_size as vk::DeviceSize,
                AccessType::TransferWrite,
            );
        }

        cmd.record_cmd_buf(move |cmd_buf| {
            let src = cmd_buf.resource(src);
            let dst = cmd_buf.resource(dst);

            unsafe {
                cmd_buf.device.cmd_copy_image_to_buffer(
                    cmd_buf.handle,
                    src.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst.handle,
                    regions.as_ref(),
                );
            }
        })
        .end_cmd()
    }

    /// Fill a region of a buffer with a fixed value.
    pub fn fill_buffer(
        &mut self,
        buffer: impl Into<AnyBufferNode>,
        region: Range<vk::DeviceSize>,
        data: u32,
    ) -> &mut Self {
        let buffer = buffer.into();

        self.begin_cmd()
            .debug_name("fill buffer")
            .subresource_access(buffer, region.clone(), AccessType::TransferWrite)
            .record_cmd_buf(move |cmd_buf| {
                let buffer = cmd_buf.resource(buffer);

                unsafe {
                    cmd_buf.device.cmd_fill_buffer(
                        cmd_buf.handle,
                        buffer.handle,
                        region.start,
                        region.end - region.start,
                        data,
                    );
                }
            })
            .end_cmd()
    }

    /// Returns the index of the first pass which accesses a given node
    #[profiling::function]
    fn first_node_access_pass_index(&self, node: impl Node) -> Option<usize> {
        let node_idx = node.index();

        for (pass_idx, pass) in self.cmds.iter().enumerate() {
            for exec in pass.execs.iter() {
                if exec.accesses.contains_key(&node_idx) {
                    return Some(pass_idx);
                }
            }
        }

        None
    }

    /// Finalizes the graph and provides an object with functions for submitting the resulting
    /// commands.
    #[profiling::function]
    pub fn into_queue(mut self) -> Queue {
        // The final execution of each pass has no function
        for cmd in &mut self.cmds {
            debug_assert!(!cmd.execs.is_empty());
            debug_assert!(cmd.execs.last().unwrap().func.is_none());

            cmd.execs.pop();
        }

        Queue::new(self)
    }

    /// Returns a borrow of the original Vulkan resource (buffer, image or acceleration structure)
    /// which the given node represents.
    pub fn resource<N>(&self, node: N) -> &N::Resource
    where
        N: Node,
    {
        node.borrow(&self.resources)
    }

    /// Note: `data` must not exceed 65536 bytes.
    #[profiling::function]
    pub fn update_buffer(
        &mut self,
        buffer: impl Into<AnyBufferNode>,
        offset: vk::DeviceSize,
        data: impl AsRef<[u8]> + 'static + Send,
    ) -> &mut Self {
        let buffer = buffer.into();
        let data_end = offset + data.as_ref().len() as vk::DeviceSize;

        #[cfg(debug_assertions)]
        {
            let buffer_info = self.resource(buffer).info;

            assert!(
                data_end <= buffer_info.size,
                "data range end ({data_end}) exceeds buffer size ({})",
                buffer_info.size
            );
        }

        self.begin_cmd()
            .debug_name("update buffer")
            .subresource_access(buffer, offset..data_end, AccessType::TransferWrite)
            .record_cmd_buf(move |cmd_buf| {
                let buffer = cmd_buf.resource(buffer);

                unsafe {
                    cmd_buf.device.cmd_update_buffer(
                        cmd_buf.handle,
                        buffer.handle,
                        offset,
                        data.as_ref(),
                    );
                }
            })
            .end_cmd()
    }
}

#[deprecated]
#[doc(hidden)]
pub mod graph {
    #[deprecated = "use vk_graph::node module"]
    pub mod node {
        #[deprecated = "use vk_graph::node::AccelerationStructureLeaseNode"]
        pub type AccelerationStructureLeaseNode = crate::node::AccelerationStructureLeaseNode;

        #[deprecated = "use vk_graph::node::AccelerationStructureNode"]
        pub type AccelerationStructureNode = crate::node::AccelerationStructureNode;

        #[deprecated = "use vk_graph::node::AnyAccelerationStructureNode"]
        pub type AnyAccelerationStructureNode = crate::node::AnyAccelerationStructureNode;

        #[deprecated = "use vk_graph::node::AnyBufferNode"]
        pub type AnyBufferNode = crate::node::AnyBufferNode;

        #[deprecated = "use vk_graph::node::AnyImageNode"]
        pub type AnyImageNode = crate::node::AnyImageNode;

        #[deprecated = "use vk_graph::node::BufferLeaseNode"]
        pub type BufferLeaseNode = crate::node::BufferLeaseNode;

        #[deprecated = "use vk_graph::node::BufferNode"]
        pub type BufferNode = crate::node::BufferNode;

        #[deprecated = "use vk_graph::node::ImageLeaseNode"]
        pub type ImageLeaseNode = crate::node::ImageLeaseNode;

        #[deprecated = "use vk_graph::node::ImageNode"]
        pub type ImageNode = crate::node::ImageNode;

        #[deprecated = "use vk_graph::node::Node"]
        pub type Node = dyn crate::node::Node<Resource = ()>;

        #[deprecated = "use vk_graph::node::SwapchainImageNode"]
        pub type SwapchainImageNode = crate::node::SwapchainImageNode;
    }

    #[deprecated]
    #[doc(hidden)]
    pub mod pass_ref {
        #[deprecated = "use vk_graph::cmd::CommandBufferRef"]
        pub type Acceleration<'a> = crate::cmd::CommandBufferRef<'a>;

        #[deprecated = "use vk_graph::cmd::CommandBufferRef"]
        pub type AccelerationStructureBuildInfo = crate::cmd::BuildAccelerationStructureInfo;

        #[deprecated = "use vk_graph::cmd::CommandBufferRef"]
        pub type AccelerationStructureIndirectBuildInfo =
            crate::cmd::BuildAccelerationStructureIndirectInfo;

        #[deprecated = "use vk_graph::cmd::CommandBufferRef"]
        pub type AccelerationStructureIndirectUpdateInfo =
            crate::cmd::UpdateAccelerationStructureIndirectInfo;

        #[deprecated = "use vk_graph::cmd::CommandBufferRef"]
        pub type AccelerationStructureUpdateInfo = crate::cmd::UpdateAccelerationStructureInfo;

        #[deprecated = "use vk_graph::cmd::Descriptor"]
        pub type Descriptor = crate::cmd::Descriptor;

        #[deprecated = "use vk_graph::cmd::GraphicCommandBufferRef"]
        pub type Draw<'a> = crate::cmd::GraphicCommandBufferRef<'a>;

        #[deprecated = "use vk_graph::cmd::CommandRef"]
        pub type PassRef<'a> = crate::cmd::CommandRef<'a>;

        #[deprecated = "use vk_graph::cmd::PipelineCommandRef"]
        pub type PipelinePassRef<'a, T> = crate::cmd::PipelineCommandRef<'a, T>;

        #[deprecated = "use vk_graph::cmd::RayTraceCommandBufferRef"]
        pub type RayTrace<'a> = crate::cmd::RayTraceCommandBufferRef<'a>;

        #[deprecated = "use vk_graph::ViewInfo"]
        pub type ViewType = crate::cmd::ViewInfo;

        #[deprecated = "remove"]
        pub trait View {
            type Information;
        }
    }

    #[deprecated = "use vk_graph::Graph"]
    pub type RenderGraph = crate::Graph;

    #[deprecated = "use vk_graph::Queue"]
    pub type Resolver = crate::Queue;
}

#[allow(deprecated)]
#[allow(unused)]
#[doc(hidden)]
pub(crate) mod deprecated {
    use {
        crate::{
            Graph, GraphResource,
            driver::{
                DriverError,
                accel_struct::{AccelerationStructure, AccelerationStructureInfo},
                buffer::{Buffer, BufferInfo},
                cmd_buf::{CommandBuffer, CommandBufferInfo},
                descriptor_set::{DescriptorPool, DescriptorPoolInfo},
                device::Device,
                image::{Image, ImageInfo},
                render_pass::{RenderPass, RenderPassInfo},
                swapchain::{Swapchain, SwapchainImage, SwapchainInfo},
            },
            node::{
                AccelerationStructureLeaseNode, AccelerationStructureNode,
                AnyAccelerationStructureNode, AnyBufferNode, AnyImageNode, BufferLeaseNode,
                BufferNode, ImageLeaseNode, ImageNode, Node, SwapchainImageNode,
            },
            pool::{Lease, Pool},
            resource::Resource,
        },
        ash::vk,
        std::{ops::Range, sync::Arc},
    };

    /// Specifies a color attachment clear value which can be used to initliaze an image.
    #[derive(Clone, Copy, Debug)]
    pub struct ClearColorValue(pub [f32; 4]);

    impl From<[f32; 3]> for ClearColorValue {
        fn from(color: [f32; 3]) -> Self {
            [color[0], color[1], color[2], 1.0].into()
        }
    }

    impl From<[f32; 4]> for ClearColorValue {
        fn from(color: [f32; 4]) -> Self {
            Self(color)
        }
    }

    impl From<[u8; 3]> for ClearColorValue {
        fn from(color: [u8; 3]) -> Self {
            [color[0], color[1], color[2], u8::MAX].into()
        }
    }

    impl From<[u8; 4]> for ClearColorValue {
        fn from(color: [u8; 4]) -> Self {
            [
                color[0] as f32 / u8::MAX as f32,
                color[1] as f32 / u8::MAX as f32,
                color[2] as f32 / u8::MAX as f32,
                color[3] as f32 / u8::MAX as f32,
            ]
            .into()
        }
    }

    #[deprecated = "use Swapchain from vk_graph_window crate"]
    #[derive(Debug)]
    #[doc(hidden)]
    pub struct Display;

    impl Display {
        pub fn new(
            device: &Arc<Device>,
            swapchain: Swapchain,
            info: impl Into<DisplayInfo>,
        ) -> Result<Self, DriverError> {
            todo!()
        }

        pub fn acquire_next_image(&mut self) -> Result<Option<SwapchainImage>, DisplayError> {
            todo!()
        }

        pub fn present_image(
            &mut self,
            pool: &mut impl ResolverPool,
            render_graph: crate::graph::RenderGraph,
            swapchain_image: SwapchainImageNode,
            queue_index: u32,
        ) -> Result<(), DisplayError> {
            todo!()
        }

        pub fn set_swapchain_info(&mut self, info: impl Into<SwapchainInfo>) {
            todo!()
        }

        pub fn swapchain_info(&self) -> SwapchainInfo {
            todo!()
        }
    }

    #[deprecated = "use vk_graph_window::SwapchainError"]
    #[derive(Clone, Copy, Debug, Default)]
    #[doc(hidden)]
    pub struct DisplayError;

    #[deprecated = "use vk_graph_window::SwapchainInfo"]
    #[derive(Clone, Copy, Debug, Default)]
    #[doc(hidden)]
    pub struct DisplayInfo;

    #[deprecated = "use vk_graph_window::SwapchainInfoBuilder"]
    #[derive(Clone, Copy, Debug, Default)]
    #[doc(hidden)]
    pub struct DisplayInfoBuilder;

    // General stuff
    impl Graph {
        #[deprecated = "use begin_cmd function"]
        #[doc(hidden)]
        pub fn begin_pass(&mut self, name: impl AsRef<str>) -> crate::graph::pass_ref::PassRef<'_> {
            self.begin_cmd().debug_name(name.as_ref().to_owned())
        }

        #[deprecated = "use bind_resource function"]
        #[doc(hidden)]
        pub fn bind_node<R>(&mut self, resource: R) -> R::Node
        where
            R: GraphResource,
        {
            self.bind_resource(resource)
        }

        #[deprecated = "use blit_image_region function"]
        #[doc(hidden)]
        pub fn blit_image_regions(
            &mut self,
            src_node: impl Into<AnyImageNode>,
            dst_node: impl Into<AnyImageNode>,
            filter: vk::Filter,
            regions: impl AsRef<[vk::ImageBlit]> + 'static + Send,
        ) -> &mut Self {
            self.blit_image_region(src_node, dst_node, filter, regions)
        }

        #[deprecated = "use clear_color_image function"]
        #[doc(hidden)]
        pub fn clear_color_image_value(
            &mut self,
            image_node: impl Into<AnyImageNode>,
            color_value: impl Into<ClearColorValue>,
        ) -> &mut Self {
            self.clear_color_image(image_node, color_value.into().0)
        }

        #[deprecated = "use clear_depth_stencil_image function"]
        #[doc(hidden)]
        pub fn clear_depth_stencil_image_value(
            &mut self,
            image_node: impl Into<AnyImageNode>,
            depth: f32,
            stencil: u32,
        ) -> &mut Self {
            self.clear_depth_stencil_image(image_node, depth, stencil)
        }

        #[deprecated = "use copy_buffer_region function"]
        #[doc(hidden)]
        pub fn copy_buffer_regions(
            &mut self,
            src_node: impl Into<AnyBufferNode>,
            dst_node: impl Into<AnyBufferNode>,
            regions: impl AsRef<[vk::BufferCopy]> + 'static + Send,
        ) -> &mut Self {
            self.copy_buffer_region(src_node, dst_node, regions)
        }

        #[deprecated = "use copy_buffer_to_image_region function"]
        #[doc(hidden)]
        pub fn copy_buffer_to_image_regions(
            &mut self,
            src_node: impl Into<AnyBufferNode>,
            dst_node: impl Into<AnyImageNode>,
            regions: impl AsRef<[vk::BufferImageCopy]> + 'static + Send,
        ) -> &mut Self {
            self.copy_buffer_to_image_region(src_node, dst_node, regions)
        }

        #[deprecated = "use copy_image_region function"]
        #[doc(hidden)]
        pub fn copy_image_regions(
            &mut self,
            src_node: impl Into<AnyImageNode>,
            dst_node: impl Into<AnyImageNode>,
            regions: impl AsRef<[vk::ImageCopy]> + 'static + Send,
        ) -> &mut Self {
            self.copy_image_region(src_node, dst_node, regions)
        }

        #[deprecated = "use copy_image_to_buffer_region function"]
        #[doc(hidden)]
        pub fn copy_image_to_buffer_regions(
            &mut self,
            src_node: impl Into<AnyImageNode>,
            dst_node: impl Into<AnyBufferNode>,
            regions: impl AsRef<[vk::BufferImageCopy]> + 'static + Send,
        ) -> &mut Self {
            self.copy_image_to_buffer_region(src_node, dst_node, regions)
        }

        #[deprecated = "use fill_buffer function"]
        #[doc(hidden)]
        pub fn fill_buffer_region(
            &mut self,
            buffer_node: impl Into<AnyBufferNode>,
            data: u32,
            region: Range<vk::DeviceSize>,
        ) -> &mut Self {
            self.fill_buffer(buffer_node, region, data)
        }

        #[deprecated = "use device_address function of resource function result"]
        #[doc(hidden)]
        pub fn node_device_address(&self, node: impl Node) -> vk::DeviceAddress {
            let idx = node.index();

            self.resources[idx]
                .as_driver_buffer()
                .unwrap()
                .device_address()
        }

        #[deprecated = "dereference info field of resource function result"]
        #[doc(hidden)]
        pub fn node_info<N>(&self, node: N) -> N::Type
        where
            N: Node + Info,
        {
            node.info(&self.resources)
        }

        #[deprecated = "use into_queue function"]
        #[doc(hidden)]
        pub fn resolve(self) -> crate::graph::Resolver {
            self.into_queue()
        }

        #[deprecated = "use resource and clone functions"]
        #[doc(hidden)]
        pub fn unbind_node<N>(&mut self, node: N) -> N::Result
        where
            N: Unbind,
        {
            node.unbind(&self.resources)
        }

        #[deprecated = "use update_buffer function"]
        #[doc(hidden)]
        pub fn update_buffer_offset(
            &mut self,
            buffer_node: impl Into<AnyBufferNode>,
            offset: vk::DeviceSize,
            data: impl AsRef<[u8]> + 'static + Send,
        ) -> &mut Self {
            self.update_buffer(buffer_node, offset, data)
        }
    }

    pub trait Info {
        type Type;

        fn info(&self, _: &[Resource]) -> Self::Type
        where
            Self: Node;
    }

    impl Info for SwapchainImageNode {
        type Type = ImageInfo;

        fn info(&self, resources: &[Resource]) -> Self::Type
        where
            Self: Node,
        {
            resources[self.idx].as_swapchain_image().unwrap().info
        }
    }

    macro_rules! info {
        ($name:ident) => {
            paste::paste! {
                impl Info for [<$name Node>] {
                    type Type = [<$name Info>];

                    fn info(&self, resources: &[Resource]) -> Self::Type
                    where
                        Self: Node,
                    {
                        resources[self.idx].[<as_ $name:snake>]().unwrap().info
                    }
                }

                impl Info for [<Any $name Node>] {
                    type Type = [<$name Info>];

                    fn info(&self, resources: &[Resource]) -> Self::Type
                    where
                        Self: Node,
                    {
                        resources[self.index()].[<as_ $name:snake>]().unwrap().info
                    }
                }

                impl Info for [<$name LeaseNode>] {
                    type Type = [<$name Info>];

                    fn info(&self, resources: &[Resource]) -> Self::Type
                    where
                        Self: Node,
                    {
                        resources[self.idx].[<as_ $name:snake _lease>]().unwrap().info
                    }
                }

                impl Unbind for [<$name Node>] {
                    type Result = Arc<$name>;

                    fn unbind(&self, resources: &[Resource]) -> Self::Result {
                        resources[self.index()].[<as_ $name:snake>]().unwrap().clone()
                    }
                }

                impl Unbind for [<$name LeaseNode>] {
                    type Result = Arc<Lease<$name>>;

                    fn unbind(&self, resources: &[Resource]) -> Self::Result {
                        resources[self.index()].[<as_ $name:snake _lease>]().unwrap().clone()
                    }
                }
            }
        };
    }

    info!(AccelerationStructure);
    info!(Buffer);
    info!(Image);

    #[deprecated = "remove"]
    pub trait ResolverPool:
        Pool<DescriptorPoolInfo, DescriptorPool>
        + Pool<RenderPassInfo, RenderPass>
        + Pool<CommandBufferInfo, CommandBuffer>
        + Send
    {
    }

    pub trait Unbind: Node {
        type Result;

        fn unbind(&self, _: &[Resource]) -> Self::Result;
    }
}
