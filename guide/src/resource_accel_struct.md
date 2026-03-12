# Acceleration Structures

```rust
// Some buffer holding geometry data
let buffer: Buffer = todo!();

// Some sample geometry to put into a BLAS:
let geometry = AccelerationStructureGeometryData::Triangles {
    index_addr: buffer.device_address(),
    index_type: vk::IndexType::UINT16,
    max_vertex: 100,
    transform_addr: None,
    vertex_addr: buffer.device_address() + 2_048,
    vertex_format: vk::Format::R32G32B32_SFLOAT,
    vertex_stride: 12,
};
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
    ty,
    flags: vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
    geometries: vec![
        (geom, build_range),
    ].into_boxed_slice(),
};

// Use helper function to find size
let size = AccelerationStructure::size_of(device, &geom_info);

// Create acceleration structure info multiple ways:
let info = AccelerationStructureInfo {
    ty,
    size,
};
let other_info = AccelerationStructureInfo::blas(size);

assert_eq!(info, other_info);

// Builder pattern
let same_info = AccelerationStructureInfoBuilder::default()
    .ty(ty)
    .size(size);

// Create directly from info
let blas = AccelerationStructure::create(device, info)?;

// Info built from other info
// Note: Never calculate size/always get from function
let more_info = blas
    .info
    .into_builder()
    .size(size * 2)
    .build();

// The provided fields are helpful:
assert_eq!(blas.device, *device);
assert_eq!(blas.info, info);
assert_ne!(blas.buffer, vk::Buffer::null());
assert_ne!(blas.handle, vk::AccelerationStrucuture::null());

// Acceleration structures have no "subresources" and are bound whole
let my_subresource = AccelerationStructureSubresourceRange;
```
