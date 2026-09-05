# Opacity Micromaps

`VK_EXT_opacity_micromap` stores per-microtriangle opacity states separately from triangle geometry.
`vk-graph` exposes physical-device support through
`device.physical.vk_ext_opacity_micromap`, owns native objects with `Micromap`, and supports device
and host operations. See the compile-checked, headless
[`opacity_micromap.rs`](https://github.com/attackgoat/vk-graph/blob/main/examples/opacity_micromap.rs)
example for a complete device build and BLAS attachment.

## Support

Check support before creating or recording micromap work:

```rust
# use vk_graph::driver::device::Device;
# fn check(device: &Device) {
let Some(extension) = &device.physical.vk_ext_opacity_micromap else {
    println!("VK_EXT_opacity_micromap is unavailable");
    return;
};

if extension.features.micromap {
    println!(
        "maximum two-state subdivision level: {}",
        extension.properties.max_opacity2_state_subdivision_level,
    );
}
# }
```

`micromap_host_commands` is a separate feature. It is required by the synchronous host methods but
not by `CommandRef` device operations. Capture replay is reported but is not exposed because
`BufferInfo` cannot request a replayable backing-buffer address.

## Data And Creation

`OpacityMicromapUsage` pairs a count, subdivision level, and two-state or four-state format. Build
usage counts describe all micromap triangles constructed in the micromap. BLAS geometry usage counts
instead describe the micromap triangles referenced by that geometry, so they can be a subset of the
build counts when geometry selects entries through micromap indices. They are identical only for a
simple one-to-one attachment such as the example. For each micromap-build input triangle, upload one
`vk::MicromapTriangleEXT` describing the encoded-data offset, subdivision level, and format. Encoded
opacity states are tightly bit-packed according to the Vulkan specification.

Call `Micromap::size_of` with an `OpacityMicromapBuildInfo`, then allocate `MicromapInfo` with
`create_size` and scratch storage with `build_size`. Device input addresses and serialized device
addresses require 256-byte alignment; build scratch uses
`min_accel_struct_scratch_offset_alignment`. Micromaps have no update mode: every build is
`vk::BuildMicromapModeEXT::BUILD`.

## Device Operations

`CommandRef` provides:

Operation | Access declarations
-|-
`build_micromaps` | destination `MicromapBuildWrite`; encoded data and triangle array `MicromapBuildInputRead`; scratch `MicromapBuildScratchReadWrite`
`copy_micromap` | source `MicromapBuildRead`; destination `MicromapBuildWrite`
`serialize_micromap` | source `MicromapBuildRead`; destination buffer `MicromapBuildBufferWrite`
`deserialize_micromap` | source buffer `MicromapBuildBufferRead`; destination `MicromapBuildWrite`
`write_micromaps_properties` | every micromap `MicromapBuildRead`; the raw query pool is managed by the caller

Device addresses do not bind or identify graph resources. Bind every input, output, and scratch
resource and declare it on the execution that uses the address. Micromap-build input and scratch
buffers must remain alive until the micromap build completes; they may then be released independently
of `MicromapSize::discardable`. That flag instead controls the lifetime of the micromap object:
when true, it may be released after the acceleration-structure build or update completes; when false,
the acceleration structure may retain references to its storage, so keep the micromap alive until
ray traversal has concluded. Before a BLAS build consumes a completed micromap, declare
`AccelerationStructureBuildMicromapRead` on the micromap. Also bind and declare the ordinary index,
vertex, optional micromap-index, BLAS scratch, and BLAS resources.

Build, copy, serialization, deserialization, and property-query command methods are `unsafe`.
The graph cannot verify input contents, raw device-address ranges, successful source construction,
or required build flags. Follow each method's Safety contract; access declarations alone do not
satisfy these preconditions.

## Host Operations

When `micromap_host_commands` is true, a micromap created with `MicromapInfo::host_mem` supports
synchronous `build_host`,
`clone_from_host`, `compact_from_host`, `serialize_host`, `deserialize_host`, `property`,
`serialization_size`, and `compacted_size`. Host operations reject device-local micromaps, including
either side of a copy. `compatibility` checks serialized version data before deserialization. These
methods are `unsafe` where Rust cannot prove that a source was successfully constructed, that source
memory is complete, or that a source was built with `ALLOW_COMPACTION`. Observe each method's Safety
contract, the documented 16-byte host serialization alignment, and pointer lifetimes.

Build with `vk::BuildMicromapFlagsEXT::ALLOW_COMPACTION` before requesting a compacted-size property
or issuing either a host or device compact copy. This Vulkan precondition is not encoded in the
micromap type and is therefore part of those unsafe APIs' contracts.

Host operations update the resource's tracked access state, but they are not graph executions. The
next submitted graph access synchronizes against that state.

## BLAS Attachment

Call `AccelerationStructureGeometryData::opacity_micromap` on triangle geometry to attach an
`AccelerationStructureOpacityMicromap`. It returns `AccelerationStructureGeometryDataExt`; convert
ordinary geometry data with `.into()` when mixing both in one build. The original geometry variants,
traits, and Vulkan conversions remain available.

The attachment owns usage counts and holds the native micromap handle and optional index data.
`base_triangle` is added to non-negative micromap indices, not special opacity indices.

For ray tracing pipelines that traverse opacity micromaps, opt in with
`RayTracingPipelineInfo::builder().opacity_micromap(true)`; this is disabled by default.

Do not use `AccelerationStructureGeometry::opaque` or set `vk::GeometryFlagsKHR::OPAQUE` on geometry
using an opacity micromap. During tracing, do not use force-opaque instance flags or ray flags such as
`gl_RayFlagsOpaqueEXT` /
`VK_RAY_FLAG_FORCE_OPAQUE_BIT_KHR`; those settings force opaque treatment and prevent the micromap
from controlling opacity.

The example is intentionally build-only. Rendering the resulting cutout additionally requires a
ray tracing pipeline, shader binding table, TLAS, output image, and shaders, which are independent of
micromap construction and graph synchronization.
