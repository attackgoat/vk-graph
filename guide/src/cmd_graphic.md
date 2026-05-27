# Graphics

Graphic commands are recorded after binding a `GraphicPipeline` and declaring attachments such as
`color_attachment_image` or `depth_stencil_attachment_image`.

## Available Commands

Command | Typical use
-|-
`bind_index_buffer` | Provide indices for indexed drawing
`bind_vertex_buffers` | Bind one or more vertex streams
`draw` | Draw non-indexed geometry
`draw_indexed` | Draw indexed geometry
`draw_indexed_indirect` | Read indexed draw parameters from a buffer
`draw_indexed_indirect_count` | GPU-driven indexed draws with a count buffer
`draw_indirect` | Read non-indexed draw parameters from a buffer
`draw_indirect_count` | GPU-driven non-indexed draws with a count buffer
`set_scissor` | Restrict drawing to one or more rectangles
`set_viewport` | Override the default viewport dynamically

## Direct Draws

The most common pattern is to bind vertex and index buffers, then issue `draw` or `draw_indexed`.

```no_run
# use vk_graph::Graph;
# use vk_graph::cmd::{LoadOp, StoreOp};
# use vk_graph::driver::ash::vk;
# use vk_graph::cmd::ClearColorValue;
# use vk_graph::driver::{DriverError, sync::AccessType};
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::graphic::{GraphicPipeline, GraphicPipelineInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# use vk_graph::driver::shader::Shader;
# fn main() -> Result<(), DriverError> {
# let device = Device::new(DeviceInfo::default())?;
let mut graph = Graph::default();

let color = graph.bind_resource(Image::create(
    &device,
    ImageInfo::image_2d(
        1280,
        720,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    ),
)?);
let vertices = graph.bind_resource(Buffer::create(
    &device,
    BufferInfo::device_mem(4096, vk::BufferUsageFlags::VERTEX_BUFFER),
)?);
let indices = graph.bind_resource(Buffer::create(
    &device,
    BufferInfo::device_mem(1024, vk::BufferUsageFlags::INDEX_BUFFER),
)?);

let pipeline = GraphicPipeline::create(
    &device,
    GraphicPipelineInfo::default(),
    [
        Shader::new_vertex([0u8; 4].as_slice()),
        Shader::new_fragment([0u8; 4].as_slice()),
    ],
)?;

graph
    .begin_cmd()
    .debug_name("main geometry pass")
    .bind_pipeline(&pipeline)
    .color_attachment_image(
        0,
        color,
        LoadOp::Clear(ClearColorValue::Float32([0.0, 0.0, 0.0, 1.0])),
        StoreOp::Store,
    )
    .resource_access(vertices, AccessType::VertexBuffer)
    .resource_access(indices, AccessType::IndexBuffer)
    .record_cmd_buf(move |cmd_buf| {
        cmd_buf
            .bind_vertex_buffers(0, [(vertices, 0)])
            .bind_index_buffer(indices, 0, vk::IndexType::UINT32)
            .draw_indexed(36, 1, 0, 0, 0);
    });
# Ok(()) }
```

## Dynamic Viewports And Scissors

The default viewport covers the full attachment extent and the default scissor does not clip.
Override them when a pass renders only part of the target.

```no_run
# use vk_graph::Graph;
# use vk_graph::cmd::{LoadOp, StoreOp};
# use vk_graph::driver::ash::vk;
# use vk_graph::driver::{DriverError, sync::AccessType};
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::graphic::{GraphicPipeline, GraphicPipelineInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# use vk_graph::driver::shader::Shader;
# fn main() -> Result<(), DriverError> {
# let device = Device::new(DeviceInfo::default())?;
let mut graph = Graph::default();
let color = graph.bind_resource(Image::create(
    &device,
    ImageInfo::image_2d(
        1280,
        720,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT,
    ),
)?);
let vertices = graph.bind_resource(Buffer::create(
    &device,
    BufferInfo::device_mem(4096, vk::BufferUsageFlags::VERTEX_BUFFER),
)?);
let pipeline = GraphicPipeline::create(
    &device,
    GraphicPipelineInfo::default(),
    [
        Shader::new_vertex([0u8; 4].as_slice()),
        Shader::new_fragment([0u8; 4].as_slice()),
    ],
)?;

graph
    .begin_cmd()
    .bind_pipeline(&pipeline)
    .color_attachment_image(0, color, LoadOp::DontCare, StoreOp::Store)
    .resource_access(vertices, AccessType::VertexBuffer)
    .record_cmd_buf(move |cmd_buf| {
        cmd_buf
            .set_viewport(
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: 640.0,
                    height: 360.0,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            )
            .set_scissor(
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: 640, height: 360 },
                }],
            )
            .bind_vertex_buffers(0, [(vertices, 0)])
            .draw(3, 1, 0, 0);
    });
# Ok(()) }
```

## Indirect Draws

Indirect drawing is the usual next step once culling, LOD selection, or instance generation moves
onto the GPU.

```no_run
# use std::mem::size_of;
# use vk_graph::Graph;
# use vk_graph::cmd::{LoadOp, StoreOp};
# use vk_graph::driver::ash::vk;
# use vk_graph::driver::{DriverError, sync::AccessType};
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::graphic::{GraphicPipeline, GraphicPipelineInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# use vk_graph::driver::shader::Shader;
# fn main() -> Result<(), DriverError> {
# let device = Device::new(DeviceInfo::default())?;
let mut graph = Graph::default();
let color = graph.bind_resource(Image::create(
    &device,
    ImageInfo::image_2d(
        1280,
        720,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT,
    ),
)?);
let vertices = graph.bind_resource(Buffer::create(
    &device,
    BufferInfo::device_mem(4096, vk::BufferUsageFlags::VERTEX_BUFFER),
)?);
let indices = graph.bind_resource(Buffer::create(
    &device,
    BufferInfo::device_mem(1024, vk::BufferUsageFlags::INDEX_BUFFER),
)?);
let draw_command = vk::DrawIndexedIndirectCommand {
    index_count: 36,
    instance_count: 1,
    first_index: 0,
    vertex_offset: 0,
    first_instance: 0,
};
let draw_args = graph.bind_resource(Buffer::create_from_slice(
    &device,
    vk::BufferUsageFlags::INDIRECT_BUFFER,
    bytemuck::cast_slice(&[
        draw_command.index_count as i32,
        draw_command.instance_count as i32,
        draw_command.first_index as i32,
        draw_command.vertex_offset,
        draw_command.first_instance as i32,
    ]),
)?);
let draw_count = graph.bind_resource(Buffer::create_from_slice(
    &device,
    vk::BufferUsageFlags::INDIRECT_BUFFER,
    &1u32.to_ne_bytes(),
)?);
let pipeline = GraphicPipeline::create(
    &device,
    GraphicPipelineInfo::default(),
    [
        Shader::new_vertex([0u8; 4].as_slice()),
        Shader::new_fragment([0u8; 4].as_slice()),
    ],
)?;

graph
    .begin_cmd()
    .bind_pipeline(&pipeline)
    .color_attachment_image(0, color, LoadOp::DontCare, StoreOp::Store)
    .resource_access(vertices, AccessType::VertexBuffer)
    .resource_access(indices, AccessType::IndexBuffer)
    .resource_access(draw_args, AccessType::IndirectBuffer)
    .resource_access(draw_count, AccessType::IndirectBuffer)
    .record_cmd_buf(move |cmd_buf| {
        cmd_buf
            .bind_vertex_buffers(0, [(vertices, 0)])
            .bind_index_buffer(indices, 0, vk::IndexType::UINT32)
            .draw_indexed_indirect_count(
                draw_args,
                0,
                draw_count,
                0,
                16,
                size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
    });
# Ok(()) }
```

## Notes

- `draw` and `draw_indexed` are the best fit for CPU-driven rendering.
- `draw_indirect` and `draw_indexed_indirect` move only the parameters onto the GPU.
- `draw_*_indirect_count` is the usual choice for fully GPU-driven visibility results.
