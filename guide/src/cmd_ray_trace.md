# Ray Tracing

Ray tracing work in `vk-graph` usually has two phases:

- Build or update acceleration structures with a general command buffer
- Bind a `RayTracingPipeline` and issue `trace_rays` or `trace_rays_indirect`

API docs: [`RayTracingCommandRef::build_accel_struct`](https://docs.rs/vk-graph/latest/vk_graph/cmd/ray_trace/struct.RayTracingCommandRef.html#method.build_accel_struct),
[`RayTracingCommandRef::trace_rays`](https://docs.rs/vk-graph/latest/vk_graph/cmd/ray_trace/struct.RayTracingCommandRef.html#method.trace_rays),
[`RayTracingCommandRef::trace_rays_indirect`](https://docs.rs/vk-graph/latest/vk_graph/cmd/ray_trace/struct.RayTracingCommandRef.html#method.trace_rays_indirect),
[`RayTracingCommandRef::push_constants`](https://docs.rs/vk-graph/latest/vk_graph/cmd/ray_trace/struct.RayTracingCommandRef.html#method.push_constants).

## Available Commands

Command | Typical use
-|-
`build_accel_struct` | Build BLAS or TLAS from CPU-provided build ranges
`build_accel_struct_indirect` | Build acceleration structures using device-provided ranges
`set_stack_size` | Override stack size when the pipeline enables dynamic stack sizing
`trace_rays` | Launch rays with CPU-provided dimensions
`trace_rays_indirect` | Launch rays with dimensions read from device memory
`update_accel_struct` | Refit or rebuild an existing structure in-place
`update_accel_struct_indirect` | Device-driven in-place update path
`push_constants` | Update small pipeline constants without a buffer upload

## Building Acceleration Structures

Acceleration-structure builds are recorded on a plain `CommandBuffer`, not a pipeline-specific
command buffer.

```no_run
# use vk_graph::Graph;
# use vk_graph::cmd::BuildAccelerationStructureInfo;
# use vk_graph::driver::ash::vk;
# use vk_graph::driver::{DriverError, sync::AccessType};
# use vk_graph::driver::accel_struct::{AccelerationStructure, AccelerationStructureGeometry, AccelerationStructureGeometryInfo, AccelerationStructureInfo};
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::device::{Device, DeviceInfo};
# fn main() -> Result<(), DriverError> {
# let device = Device::create(DeviceInfo::default())?;
let mut graph = Graph::default();

let blas = graph.bind_resource(AccelerationStructure::create(
    &device,
    AccelerationStructureInfo::blas(1),
)?);
let scratch = graph.bind_resource(Buffer::create(
    &device,
    BufferInfo::device_mem(
        4096,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    ),
)?);

graph
    .begin_cmd()
    .resource_access(scratch, AccessType::AccelerationStructureBufferWrite)
    .resource_access(blas, AccessType::AccelerationStructureBuildWrite)
    .record_cmd(move |cmd| {
        let scratch_addr = cmd.resource(scratch).device_address();
        let build_info: AccelerationStructureGeometryInfo<(
            AccelerationStructureGeometry,
            vk::AccelerationStructureBuildRangeInfoKHR,
        )> = todo!("geometry setup");

        cmd.build_accel_struct(&[
            BuildAccelerationStructureInfo::new(blas, scratch_addr, build_info),
        ]);
    });
# Ok(()) }
```

The indirect form is the same idea, but the range data lives on the device. That is useful when a
previous GPU pass writes primitive counts or build ranges.

## Tracing Rays

Once the acceleration structures and shader binding table are ready, bind a `RayTracingPipeline` and
issue `trace_rays`.

```no_run
# use vk_graph::Graph;
# use vk_graph::driver::ash::vk;
# use vk_graph::driver::{DriverError, sync::AccessType};
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# use vk_graph::driver::ray_trace::{RayTracingPipeline, RayTracingPipelineInfo, RayTracingShaderGroup};
# use vk_graph::driver::shader::Shader;
# fn main() -> Result<(), DriverError> {
# let device = Device::create(DeviceInfo::default())?;
let mut graph = Graph::default();
let output = graph.bind_resource(Image::create(
    &device,
    ImageInfo::image_2d(
        1280,
        720,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
    ),
)?);

let pipeline = RayTracingPipeline::create(
    &device,
    RayTracingPipelineInfo::default(),
    [
        Shader::new_ray_gen([0u8; 4].as_slice()),
        Shader::new_miss([0u8; 4].as_slice()),
    ],
    [
        RayTracingShaderGroup::new_general(0),
        RayTracingShaderGroup::new_general(1),
    ],
)?;

let raygen_sbt: vk::StridedDeviceAddressRegionKHR = todo!("raygen shader binding table");
let miss_sbt: vk::StridedDeviceAddressRegionKHR = todo!("miss shader binding table");
let hit_sbt = vk::StridedDeviceAddressRegionKHR::default();
let callable_sbt = vk::StridedDeviceAddressRegionKHR::default();

graph
    .begin_cmd()
    .bind_pipeline(&pipeline)
    .shader_resource_access(0, output, AccessType::General)
    .record_cmd(move |cmd| {
        cmd.trace_rays(&raygen_sbt, &miss_sbt, &hit_sbt, &callable_sbt, 1280, 720, 1);
    });
# Ok(()) }
```

## Push Constants

Use [`RayTracingCommandRef::push_constants`](https://docs.rs/vk-graph/latest/vk_graph/cmd/ray_trace/struct.RayTracingCommandRef.html#method.push_constants)
for small ray tracing state such as frame counters or camera parameters.

```no_run
# use vk_graph::driver::{ash::vk, DriverError};
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# use vk_graph::driver::ray_trace::{RayTracingPipeline, RayTracingPipelineInfo, RayTracingShaderGroup};
# use vk_graph::driver::shader::Shader;
# use vk_graph::Graph;
# fn main() -> Result<(), DriverError> {
# let device = Device::create(DeviceInfo::default())?;
# let pipeline = RayTracingPipeline::create(
#     &device,
#     RayTracingPipelineInfo::default(),
#     [Shader::new_ray_gen([0u8; 4].as_slice())],
#     [RayTracingShaderGroup::new_general(0)],
# )?;
# let output = Image::create(
#     &device,
#     ImageInfo::image_2d(
#         1280,
#         720,
#         vk::Format::R16G16B16A16_SFLOAT,
#         vk::ImageUsageFlags::STORAGE,
#     ),
# )?;
# let mut graph = Graph::default();
# let output = graph.bind_resource(output);
graph
    .begin_cmd()
    .bind_pipeline(&pipeline)
    .record_cmd(move |cmd| {
        cmd.push_constants(0, &[42])
            .trace_rays(
                &vk::StridedDeviceAddressRegionKHR::default(),
                &vk::StridedDeviceAddressRegionKHR::default(),
                &vk::StridedDeviceAddressRegionKHR::default(),
                &vk::StridedDeviceAddressRegionKHR::default(),
                1280,
                720,
                1,
            );
    });
# Ok(()) }
```

## Dynamic Stack Size And Indirect Trace

Use `set_stack_size` only when the pipeline was created with `dynamic_stack_size(true)`. Combine it
with `trace_rays_indirect` when another pass writes the trace dimensions into a device-addressable
buffer.

```no_run
# use vk_graph::Graph;
# use vk_graph::driver::ash::vk;
# use vk_graph::driver::{DriverError, sync::AccessType};
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# use vk_graph::driver::ray_trace::{RayTracingPipeline, RayTracingPipelineInfo, RayTracingShaderGroup};
# use vk_graph::driver::shader::Shader;
# fn main() -> Result<(), DriverError> {
# let device = Device::create(DeviceInfo::default())?;
let mut graph = Graph::default();
let output = graph.bind_resource(Image::create(
    &device,
    ImageInfo::image_2d(
        1280,
        720,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ImageUsageFlags::STORAGE,
    ),
)?);
let args = graph.bind_resource(Buffer::create(
    &device,
    BufferInfo::device_mem(
        std::mem::size_of::<vk::TraceRaysIndirectCommandKHR>() as u64,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    ),
)?);
let pipeline = RayTracingPipeline::create(
    &device,
    RayTracingPipelineInfo::builder().dynamic_stack_size(true),
    [
        Shader::new_ray_gen([0u8; 4].as_slice()),
        Shader::new_miss([0u8; 4].as_slice()),
    ],
    [
        RayTracingShaderGroup::new_general(0),
        RayTracingShaderGroup::new_general(1),
    ],
)?;

let raygen_sbt: vk::StridedDeviceAddressRegionKHR = todo!("raygen shader binding table");
let miss_sbt: vk::StridedDeviceAddressRegionKHR = todo!("miss shader binding table");
let hit_sbt = vk::StridedDeviceAddressRegionKHR::default();
let callable_sbt = vk::StridedDeviceAddressRegionKHR::default();

graph
    .begin_cmd()
    .bind_pipeline(&pipeline)
    .resource_access(args, AccessType::IndirectBuffer)
    .shader_resource_access(0, output, AccessType::General)
    .record_cmd(move |cmd| {
        cmd
            .set_stack_size(4096)
            .trace_rays_indirect(
                &raygen_sbt,
                &miss_sbt,
                &hit_sbt,
                &callable_sbt,
                cmd.resource(args).device_address(),
            );
    });
# Ok(()) }
```

## Notes

- Build/update commands and trace commands are separate because they have different setup needs.
- `trace_rays` is the easiest path when the CPU already knows the launch dimensions.
- `trace_rays_indirect` is the better fit when a GPU pass writes the ray count or image extent.
- `update_accel_struct` and `update_accel_struct_indirect` are for refit-style workloads where the
  topology is stable but transforms or vertex positions change.
