# Acceleration Structures

```rust
# use vk_graph::Graph;
# use vk_graph::driver::{DriverError, ash::vk, device::Device};
# use vk_graph::driver::accel_struct::{
#   AccelerationStructure, AccelerationStructureGeometry, AccelerationStructureGeometryData,
#   AccelerationStructureGeometryInfo, AccelerationStructureInfo, AccelerationStructureInfoBuilder,
#   AccelerationStructureSize, DeviceOrHostAddress
# };
# use vk_graph::driver::buffer::Buffer;
# fn test(
#     device: &Device,
# ) -> Result<(), DriverError> {
// Some buffer holding geometry data
let buffer: Buffer = todo!();

// Some sample geometry to put into a BLAS:
let geometry = AccelerationStructureGeometryData::triangles(
    DeviceOrHostAddress::DeviceAddress(buffer.device_address()),
    vk::IndexType::UINT16,
    100,
    None,
    DeviceOrHostAddress::DeviceAddress(buffer.device_address() + 2_048),
    vk::Format::R32G32B32_SFLOAT,
    12,
);
let geom = AccelerationStructureGeometry {
    max_primitive_count: 120,
    flags: vk::GeometryFlagsKHR::OPAQUE,
    geometry,
};
let build_range = vk::AccelerationStructureBuildRangeInfoKHR {
    primitive_count: 120,
    primitive_offset: 0,
    first_vertex: 0,
    transform_offset: 0,
};
let ty = vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL;
let geom_info = AccelerationStructureGeometryInfo {
    acceleration_structure_type: ty,
    flags: vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
    geometries: vec![
        (geom, build_range),
    ].into_boxed_slice(),
};

// Use helper function to find size
let AccelerationStructureSize {
    build_size,
    ..
} = AccelerationStructure::size_of(device, &geom_info);

// Create acceleration structure info multiple ways:
let info = AccelerationStructureInfo {
    acceleration_structure_type: ty,
    size: build_size,
};
let other_info = AccelerationStructureInfo::blas(build_size);

assert_eq!(info, other_info);

// Builder pattern
let same_info = AccelerationStructureInfoBuilder::default()
    .acceleration_structure_type(ty)
    .size(build_size);

// Create directly from info
let blas = AccelerationStructure::create(device, info)?;

// Info built from other info
// Note: Never calculate size/always get from function
let more_info = blas
    .info
    .into_builder()
    .size(build_size * 2)
    .build();

// The provided fields are helpful:
assert_eq!(blas.buffer.device, *device);
assert_eq!(blas.info, info);
assert_ne!(blas.buffer.handle, vk::Buffer::null());
assert_ne!(blas.handle, vk::AccelerationStructureKHR::null());

// Acceleration structures have no "subresources" and are bound whole
# Ok(()) }
```

## Opacity Micromap Attachments

An opacity micromap is built before the BLAS that consumes it. Attach it only to triangle geometry
and declare the micromap as `AccessType::AccelerationStructureBuildMicromapRead` for the BLAS build.
Any optional micromap-index buffer is a separate acceleration-structure build input and must also
be graph-bound. See [Opacity Micromaps](resource_micromap.md) for usage counts, lifetime requirements,
and opacity flags, or the headless
[`opacity_micromap.rs`](https://github.com/attackgoat/vk-graph/blob/main/examples/opacity_micromap.rs)
example for the complete build flow.
