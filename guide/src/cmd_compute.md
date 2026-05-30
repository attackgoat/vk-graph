# Computing

Compute commands are recorded after binding a `ComputePipeline`. They are typically paired with
`shader_resource_access` for storage buffers or images, and `resource_access` for indirect argument
buffers.

## Available Commands

Command | Typical use
-|-
`dispatch` | Launch workgroups directly from CPU-provided dimensions
`dispatch_base` | Launch workgroups with a non-zero base workgroup ID
`dispatch_indirect` | Read dispatch dimensions from a buffer on the device

## Direct Dispatch

`dispatch` is the default option. Use it when the CPU already knows the workgroup count.

```no_run
# use vk_graph::Graph;
# use vk_graph::driver::ash::vk;
# use vk_graph::driver::{DriverError, sync::AccessType};
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::compute::{ComputePipeline, ComputePipelineInfo};
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::shader::Shader;
# fn main() -> Result<(), DriverError> {
# let device = Device::new(DeviceInfo::default())?;
let mut graph = Graph::default();

let output = graph.bind_resource(Buffer::create(
    &device,
    BufferInfo::device_mem(
        4096,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    ),
)?);

let pipeline = ComputePipeline::create(
    &device,
    ComputePipelineInfo::default(),
    Shader::new_compute([0u8; 4].as_slice()),
)?;

graph
    .begin_cmd()
    .debug_name("prefix sum")
    .bind_pipeline(&pipeline)
    .shader_resource_access(0, output, AccessType::ComputeShaderWrite)
    .record_cmd(move |cmd_buf| {
        cmd_buf.dispatch(64, 1, 1);
    });
# Ok(()) }
```

## Offset Dispatches

`dispatch_base` is useful when a pipeline processes a tiled domain and each invocation needs a
non-zero `gl_WorkGroupID` origin.

```no_run
# use vk_graph::Graph;
# use vk_graph::driver::ash::vk;
# use vk_graph::driver::{DriverError, sync::AccessType};
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::compute::{ComputePipeline, ComputePipelineInfo};
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::shader::Shader;
# fn main() -> Result<(), DriverError> {
# let device = Device::new(DeviceInfo::default())?;
let mut graph = Graph::default();

let output = graph.bind_resource(Buffer::create(
    &device,
    BufferInfo::device_mem(4096, vk::BufferUsageFlags::STORAGE_BUFFER),
)?);
let pipeline = ComputePipeline::create(
    &device,
    ComputePipelineInfo::default(),
    Shader::new_compute([0u8; 4].as_slice()),
)?;

graph
    .begin_cmd()
    .bind_pipeline(&pipeline)
    .shader_resource_access(0, output, AccessType::ComputeShaderWrite)
    .record_cmd(move |cmd_buf| {
        cmd_buf.dispatch_base(4, 2, 0, 16, 8, 1);
    });
# Ok(()) }
```

## GPU-Driven Dispatch

`dispatch_indirect` lets an earlier pass write the group counts into a buffer. The compute pass
then consumes those parameters without CPU intervention.

```no_run
# use vk_graph::Graph;
# use vk_graph::driver::ash::vk;
# use vk_graph::driver::{DriverError, sync::AccessType};
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::compute::{ComputePipeline, ComputePipelineInfo};
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::shader::Shader;
# fn main() -> Result<(), DriverError> {
# let device = Device::new(DeviceInfo::default())?;
let mut graph = Graph::default();
let output = graph.bind_resource(Buffer::create(
    &device,
    BufferInfo::device_mem(4096, vk::BufferUsageFlags::STORAGE_BUFFER),
)?);
let args = vk::DispatchIndirectCommand { x: 32, y: 8, z: 1 };
let args_buffer = graph.bind_resource(Buffer::create_from_slice(
    &device,
    vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
    bytemuck::cast_slice::<u32, u8>(&[args.x, args.y, args.z]),
)?);
let pipeline = ComputePipeline::create(
    &device,
    ComputePipelineInfo::default(),
    Shader::new_compute([0u8; 4].as_slice()),
)?;

graph
    .begin_cmd()
    .bind_pipeline(&pipeline)
    .resource_access(args_buffer, AccessType::IndirectBuffer)
    .shader_resource_access(0, output, AccessType::ComputeShaderWrite)
    .record_cmd(move |cmd_buf| {
        cmd_buf.dispatch_indirect(args_buffer, 0);
    });
# Ok(()) }
```

## Notes

- `dispatch` and `dispatch_base` are the simplest and cheapest commands to drive from CPU code.
- `dispatch_indirect` is the usual choice for GPU-generated work queues or culling results.
- The bound pipeline and declared resource access determine the synchronization requirements around
  the dispatch.
