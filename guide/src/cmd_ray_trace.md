# Ray Tracing

Ray tracing work in `vk-graph` usually has two phases:

1. Build or update acceleration structures with a general command buffer, building any opacity
   micromaps before the BLAS builds that consume them.
2. Bind a `RayTracingPipeline` and issue `trace_rays` or `trace_rays_indirect`.

API docs: [`CommandRef::build_acceleration_structures`](https://docs.rs/vk-graph/latest/vk_graph/cmd/struct.CommandRef.html#method.build_acceleration_structures),
[`CommandRef::build_micromaps`](https://docs.rs/vk-graph/latest/vk_graph/cmd/struct.CommandRef.html#method.build_micromaps),
[`RayTracingCommandRef::trace_rays`](https://docs.rs/vk-graph/latest/vk_graph/cmd/struct.RayTracingCommandRef.html#method.trace_rays),
[`RayTracingCommandRef::trace_rays_indirect`](https://docs.rs/vk-graph/latest/vk_graph/cmd/struct.RayTracingCommandRef.html#method.trace_rays_indirect),
[`RayTracingCommandRef::push_constants`](https://docs.rs/vk-graph/latest/vk_graph/cmd/struct.RayTracingCommandRef.html#method.push_constants).

## Available Commands

Command | Typical use
-|-
`build_acceleration_structures` | Build or update BLAS or TLAS from CPU-provided Vulkan build ranges
`build_acceleration_structures_indirect` | Build or update using device-provided Vulkan ranges
`build_micromaps` | Build opacity micromaps from encoded opacity and triangle metadata
`copy_micromap` | Clone or compact a micromap
`serialize_micromap` / `deserialize_micromap` | Move a compatible serialized representation to or from a device address
`write_micromaps_properties` | Write compacted or serialization sizes to a caller-owned query pool
`set_stack_size` | Override stack size when the pipeline enables dynamic stack sizing
`trace_rays` | Launch rays with CPU-provided dimensions
`trace_rays_indirect` | Launch rays with dimensions read from device memory
`push_constants` | Update small pipeline constants without a buffer upload

## Building Acceleration Structures

Acceleration-structure builds are recorded on a plain `CommandBuffer`, not a pipeline-specific
command buffer.

```no_run
# use vk_graph::Graph;
# use vk_graph::cmd::AccelerationStructureBuildGeometryInfo;
# use vk_graph::driver::ash::vk;
# use vk_graph::driver::sync::AccessType;
# use vk_graph::driver::accel_struct::{AccelerationStructure, AccelerationStructureGeometry, AccelerationStructureGeometryData};
# use vk_graph::driver::buffer::Buffer;
# unsafe fn record(graph: &mut Graph, blas: AccelerationStructure, scratch: Buffer, vertices: Buffer) {
// Prepared non-indexed triangle input; allocations cover a matching DEVICE or HOST_OR_DEVICE query.
let blas = graph.bind_resource(blas);
let scratch = graph.bind_resource(scratch);
let vertices = graph.bind_resource(vertices);

graph
    .begin_cmd()
    .resource_access(vertices, AccessType::AccelerationStructureBuildInputRead)
    .resource_access(scratch, AccessType::AccelerationStructureBuildScratchReadWrite)
    .resource_access(blas, AccessType::AccelerationStructureBuildWrite)
    .record_cmd(move |cmd| {
        let scratch_addr = cmd.resource(scratch).device_address();
        let geometries = [AccelerationStructureGeometry::opaque(
            AccelerationStructureGeometryData::triangles(
                0, vk::IndexType::NONE_KHR, 2, 0,
                cmd.resource(vertices).device_address(), vk::Format::R32G32B32_SFLOAT, 12,
            ),
        )];
        let infos = [AccelerationStructureBuildGeometryInfo::build(
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
            blas, &geometries, scratch_addr,
        )];
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count: 1,
            ..Default::default()
        }];

        // Caller guarantees valid input contents, allocation sizes/alignment, and lifetimes.
        unsafe {
            cmd.build_acceleration_structures(&infos, &[&ranges]);
        }
    });
# }
```

`AccelerationStructureBuildGeometryInfo<'a>` borrows the geometries and specifies the AS type,
flags, `mode`, optional `src_acceleration_structure`, `dst_acceleration_structure`, and a device-only
`scratch_data` address. Use `build(...)` for BUILD or `update(...)` for UPDATE. UPDATE requires a
successfully built `ALLOW_UPDATE` source and Vulkan-compatible flags, geometry, primitive counts,
and micromap state. Source and destination may be the same for an in-place update; otherwise their
storage must not overlap. Use the matching query's `update_scratch_size` for UPDATE.

The direct method takes `infos` and `&[&[vk::AccelerationStructureBuildRangeInfoKHR]]`, one range
slice per build and one range per geometry. The unsafe indirect method takes `infos`,
`&[vk::DeviceAddress]`, `&[u32]` strides, and `&[&[u32]]` maximum primitive counts. Each address
points to strided Vulkan build ranges, one per geometry; outer arrays have one entry per build.
Counts must not exceed the limits used in the size query. Indirect builds require the enabled
`acceleration_structure_indirect_build` feature, four-byte-aligned addresses and strides, and
`INDIRECT_BUFFER | SHADER_DEVICE_ADDRESS` buffers declared as `AccessType::General`.
Do not use `AccessType::IndirectBuffer` for indirect AS ranges: it maps to `DRAW_INDIRECT`, while
Vulkan reads these ranges at `ACCELERATION_STRUCTURE_BUILD_KHR` with `INDIRECT_COMMAND_READ`.
`vk-sync` has no exact access type for this combination. `General` uses `ALL_COMMANDS` with
memory read/write access, which is safe but may synchronize more work than needed.

Both commands are unsafe. A batch may mix BUILD and UPDATE entries, but entries are not synchronized
with each other.
Keep all input, referenced AS/micromap, destination, and scratch resources alive through execution,
with valid contents, usage flags, alignment, capacity, and queue ownership. Inputs must not change
while read; scratch must not overlap inputs or AS storage, and batch destinations and scratch must
not overlap or depend on another entry's writes. Declare input buffers as
`AccelerationStructureBuildInputRead`, source/referenced structures as `AccelerationStructureBuildRead`,
destinations as `AccelerationStructureBuildWrite`, and scratch as `AccelerationStructureBuildScratchReadWrite`.
An in-place update needs both source-read and destination-write declarations. Device addresses do
not identify or retain graph resources; access declarations alone do not satisfy Vulkan's requirements.

For opacity micromaps, record the unsafe `CommandRef::build_micromaps` call before the consuming
BLAS build in separate graph executions. See [Opacity Micromaps](resource_micromap.md#device-operations)
for access declarations, lifetimes, and safety contracts, and the headless
[`opacity_micromap.rs`](https://github.com/attackgoat/vk-graph/blob/main/examples/opacity_micromap.rs)
example for the complete build flow.

## Tracing Rays

Once the acceleration structures and shader binding table are ready, bind a `RayTracingPipeline` and
issue `trace_rays`.

```no_run
# use vk_graph::Graph;
# use vk_graph::driver::ash::vk;
# use vk_graph::driver::{DriverError, sync::AccessType};
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# use vk_graph::driver::ray_tracing::{RayTracingPipeline, RayTracingPipelineInfo, RayTracingShaderGroup};
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

Use [`RayTracingCommandRef::push_constants`](https://docs.rs/vk-graph/latest/vk_graph/cmd/struct.RayTracingCommandRef.html#method.push_constants)
for small ray tracing state such as frame counters or camera parameters.

```no_run
# use vk_graph::driver::{ash::vk, DriverError};
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# use vk_graph::driver::ray_tracing::{RayTracingPipeline, RayTracingPipelineInfo, RayTracingShaderGroup};
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
# use vk_graph::driver::ray_tracing::{RayTracingPipeline, RayTracingPipelineInfo, RayTracingShaderGroup};
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
- Use UPDATE mode for refit-style workloads where topology is stable but transforms or vertex
  positions change, subject to Vulkan's update restrictions.
- For opacity micromap traversal, follow the [pipeline opt-in and opacity-flag requirements](resource_micromap.md#blas-attachment).
