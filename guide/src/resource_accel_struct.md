# Acceleration Structures

Geometry descriptions borrow metadata: `AccelerationStructureGeometry<'a>` contains
`AccelerationStructureGeometryData<'a>`, whose `Triangles` variant holds
`AccelerationStructureTriangles<'a>` with an optional opacity attachment. Addresses are
`vk::DeviceAddress`; use zero when there is no transform. `max_vertex` is the highest addressable
vertex index, not a vertex count. Primitive-count limits and build ranges are supplied separately.

```no_run
# use vk_graph::driver::{DriverError, ash::vk, device::Device};
# use vk_graph::driver::accel_struct::{
#   AccelerationStructure, AccelerationStructureGeometry, AccelerationStructureGeometryData,
#   AccelerationStructureInfo, AccelerationStructureInfoBuilder
# };
# fn test(
#     device: &Device,
# ) -> Result<(), DriverError> {
// Size queries do not dereference input addresses.
let geometries = [AccelerationStructureGeometry::opaque(
    AccelerationStructureGeometryData::triangles(
        0,
        vk::IndexType::UINT16,
        99,
        0,
        0,
        vk::Format::R32G32B32_SFLOAT,
        12,
    ),
)];
let ty = vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL;

// No micromap handles are attached here. The caller ensures geometry/count metadata,
// flags, and device limits satisfy Vulkan's build-size query constraints.
let sizes = unsafe {
    AccelerationStructure::build_sizes(
        device,
        vk::AccelerationStructureBuildTypeKHR::DEVICE,
        ty,
        vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
        &geometries,
        &[120], // One maximum primitive count per geometry, in the same order.
    )
};

// Create acceleration structure info multiple ways:
let info = AccelerationStructureInfo {
    acceleration_structure_type: ty,
    size: sizes.acceleration_structure_size,
};
let other_info = AccelerationStructureInfo::blas(sizes.acceleration_structure_size);

assert_eq!(info, other_info);

// Builder pattern
let same_info = AccelerationStructureInfoBuilder::default()
    .acceleration_structure_type(ty)
    .size(sizes.acceleration_structure_size);

// Create directly from info
let blas = AccelerationStructure::create(device, info)?;

// The provided fields are helpful:
assert_eq!(blas.buffer.device, *device);
assert_eq!(blas.info, info);
assert_ne!(blas.buffer.handle, vk::Buffer::null());
assert_ne!(blas.handle, vk::AccelerationStructureKHR::null());

// Acceleration structures have no "subresources" and are bound whole
# Ok(()) }
```

`AccelerationStructureBuildSizes::acceleration_structure_size` is the AS storage size, **not** the scratch size.
Allocate separate device-addressable `STORAGE_BUFFER` scratch using `build_scratch_size` for BUILD
or `update_scratch_size` for UPDATE, with its address aligned to
`min_accel_struct_scratch_offset_alignment`. Query with `DEVICE` or `HOST_OR_DEVICE` for device
builds, and `HOST` or `HOST_OR_DEVICE` for host builds. Match AS type, flags, geometry metadata,
and primitive-count limits to the intended build.

`AccelerationStructure::build_sizes` is unsafe: attached micromap handles must remain valid and
belong to the same device for the entire query. Geometry, primitive counts, and flags must meet
Vulkan's query requirements and device limits. Geometry input addresses are not
dereferenced, but transform presence (a nonzero address) and opacity usage metadata must match the
intended build. Borrowed metadata must remain alive through the call.
See [Ray Tracing](cmd_ray_trace.md#building-acceleration-structures) for command inputs and safety.

## Opacity Micromap Attachments

An opacity micromap is built before the BLAS that consumes it. Attach it only to triangle geometry
and declare the micromap as `AccessType::AccelerationStructureBuildMicromapRead` for the BLAS build.
Any optional micromap-index buffer is a separate acceleration-structure build input and must also
be graph-bound. See [Opacity Micromaps](resource_micromap.md) for usage counts, lifetime requirements,
and opacity flags, or the headless
[`opacity_micromap.rs`](https://github.com/attackgoat/vk-graph/blob/main/examples/opacity_micromap.rs)
example for the complete build flow.
