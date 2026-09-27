# Opacity Micromaps

`VK_EXT_opacity_micromap` stores per-microtriangle opacity states separately from triangle geometry.
Check device support through `device.physical.vk_ext_opacity_micromap`. `Micromap` owns the Vulkan
object and supports device and host operations. See the headless
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

`OpacityMicromapUsage` specifies a count, subdivision level, and two-state or four-state format. Build
usage counts describe all micromap triangles constructed in the micromap. BLAS geometry usage counts
instead describe the micromap triangles referenced by that geometry, so they can be a subset of the
build counts when geometry selects entries through micromap indices. They are identical only for a
simple one-to-one attachment such as the example. For each micromap-build input triangle, upload one
`vk::MicromapTriangleEXT` describing the encoded-data offset, subdivision level, and format. Encoded
opacity states are tightly bit-packed according to the Vulkan specification.

Query storage and scratch sizes before allocating buffers:

```no_run
# use vk_graph::driver::{ash::vk, device::Device, DriverError};
# use vk_graph::driver::micromap::{Micromap, MicromapInfo, OpacityMicromapUsage};
# fn create(device: &Device) -> Result<(), DriverError> {
let usage_counts = [OpacityMicromapUsage::new(1, 0, vk::OpacityMicromapFormatEXT::TYPE_2_STATE)];
let flags = vk::BuildMicromapFlagsEXT::empty();
let sizes = Micromap::build_sizes(
    device, vk::AccelerationStructureBuildTypeKHR::DEVICE, flags, &usage_counts,
);
let micromap = Micromap::create(device, MicromapInfo::device_mem(sizes.micromap_size))?;
// Allocate separate scratch storage for sizes.build_scratch_size.
# Ok(()) }
```

`MicromapBuildSizes` contains `micromap_size`, `build_scratch_size`, and `discardable`.
`Micromap::build_sizes` is safe. Match flags and usage counts to the intended build; query
with `DEVICE` or `HOST_OR_DEVICE` for device builds, and `HOST` or `HOST_OR_DEVICE` for host builds.
Device input addresses and serialized device addresses require 256-byte alignment; build scratch uses
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

`build_micromaps` accepts `&[MicromapBuildInfo<'_>]`. Each descriptor contains `dst_micromap`,
`flags`, borrowed `usage_counts: &[OpacityMicromapUsage]`, and device-only `data`, `triangle_array`,
and `scratch_data` addresses plus `triangle_array_stride`. Usage metadata is borrowed during
recording and converted to temporary Vulkan arrays; it does not keep buffers alive.
Batch entries are not ordered or synchronized with each other and must not depend on each other's writes.

Device addresses do not bind or identify graph resources. Bind every input, output, and scratch
resource and declare it on the execution that uses the address. Micromap-build input and scratch
buffers must remain alive until the micromap build completes; they may then be released independently
of `MicromapBuildSizes::discardable`. That flag instead controls the lifetime of the micromap object:
when true, it may be released after the acceleration-structure build or update completes; when false,
the acceleration structure may retain references to its storage, so keep the micromap alive until
ray traversal has concluded. Before a BLAS build consumes a completed micromap, declare
`AccelerationStructureBuildMicromapRead` on the micromap. Also bind and declare the ordinary index,
vertex, optional micromap-index, BLAS scratch, and BLAS resources.

Build, copy, serialization, deserialization, and property-query command methods are `unsafe`.
The graph cannot verify input contents, device-address ranges, whether a source was built successfully,
or required build flags. Follow each method's Safety contract; access declarations alone do not
satisfy these preconditions.

## Host Operations

When `micromap_host_commands` is true, a micromap created with `MicromapInfo::host_mem` supports
synchronous `build_host`, `clone_from_host`, `compact_from_host`, `serialize_host`, `deserialize_host`, `property`,
`serialization_size`, and `compacted_size`. Host operations reject device-local micromaps, including
either side of a copy. `compatibility` checks serialized version data before deserialization. These
methods are `unsafe` where Rust cannot verify that a source was built successfully, serialized data
is complete, or a source was built with `ALLOW_COMPACTION`. Follow each method's Safety
contract, the documented 16-byte host serialization alignment, and pointer lifetimes.

`build_host(&HostMicromapBuildInfo<'_>)` uses a separate descriptor with `flags`, borrowed
`usage_counts: &[OpacityMicromapUsage]`, `data` and `triangle_array` as `*const c_void`,
`triangle_array_stride`, and writable `scratch_data: *mut c_void`. Raw pointers do not keep memory
alive: inputs must be initialized and readable, scratch writable, and all ranges
aligned, sufficiently sized, non-overlapping (except read-only inputs), and alive for the entire
synchronous call. Null scratch is allowed only when the matching query requires zero bytes.
The destination must not be in GPU use; synchronize conflicting host/device accesses explicitly.

Build with `vk::BuildMicromapFlagsEXT::ALLOW_COMPACTION` before requesting a compacted-size property
or compacting a micromap on the host or device. The micromap type does not enforce this Vulkan
requirement; the caller must check it before using these unsafe methods.

Host operations update the resource's tracked access state, but they are not graph executions. The
next submitted graph access synchronizes against that state.

## BLAS Attachment

Call `AccelerationStructureTriangles::opacity_micromap` to attach an
`AccelerationStructureOpacityMicromap<'a>`, then use `.into()` to convert the triangles to
`AccelerationStructureGeometryData<'a>`. Its `Triangles(AccelerationStructureTriangles<'a>)`
variant stores the optional attachment; there is no enum-wide attachment method.

The attachment borrows `&[OpacityMicromapUsage]` and holds the native micromap handle and optional
index-buffer device address. Keep the usage slice alive through the query or recording call; the
handle and address do not retain resources. The unsafe AS size query requires live, valid attachment
handles from the querying device; it does not ignore them. `base_triangle` is added to non-negative
micromap indices, not special opacity indices. With `index_type == NONE_KHR`, triangle `i` uses
`base_triangle + i`.

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
