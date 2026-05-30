# Commands

`vk-graph` exposes two styles of commands:

- Built-in graph commands such as `copy_buffer`, `clear_color_image`, and `update_buffer`
- Explicit command-buffer recording through `begin_cmd().record_cmd(...)`

The built-in commands are the easiest place to start. They automatically describe the required
transfer access and insert the synchronization they need.

## Built-In Commands

These helpers cover common transfer-style work:

Command | Typical use
-|-
`blit_image` | Scale or format-convert one image into another
`clear_color_image` | Clear a color render target, staging image, or scratch image
`clear_depth_stencil_image` | Initialize or reset a depth/stencil image
`copy_buffer` | Copy data between buffers
`copy_buffer_to_image` | Upload staging-buffer contents into an image
`copy_image` | Copy texels between images without filtering
`copy_image_to_buffer` | Read back image data into a buffer
`fill_buffer` | Fill a buffer region with a repeated `u32` value
`update_buffer` | Upload up to 64 KiB of inline data directly into a buffer

## Typical Flow

The most common pattern is to stage data in a buffer, upload it into an image, and then clear or
copy other resources as part of the same graph:

```no_run
# use vk_graph::Graph;
# use vk_graph::driver::DriverError;
# use vk_graph::driver::ash::vk;
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::device::{Device, DeviceInfo};
# use vk_graph::driver::image::{Image, ImageInfo};
# fn main() -> Result<(), DriverError> {
# let device = Device::new(DeviceInfo::default())?;
let mut graph = Graph::default();

let staging = Buffer::create(
    &device,
    BufferInfo::host_mem(
        256 * 256 * 4,
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
    ),
)?;
let upload_image = Image::create(
    &device,
    ImageInfo::image_2d(
        256,
        256,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::SAMPLED,
    ),
)?;
let mip_preview = Image::create(
    &device,
    ImageInfo::image_2d(
        128,
        128,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
    ),
)?;
let readback = Buffer::create(
    &device,
    BufferInfo::host_mem(
        128 * 128 * 4,
        vk::BufferUsageFlags::TRANSFER_DST,
    ),
)?;

let staging = graph.bind_resource(staging);
let upload_image = graph.bind_resource(upload_image);
let mip_preview = graph.bind_resource(mip_preview);
let readback = graph.bind_resource(readback);

graph
    .update_buffer(staging, 0, [0xff; 64])
    .copy_buffer_to_image(staging, upload_image)
    .blit_image(upload_image, mip_preview, vk::Filter::LINEAR)
    .clear_color_image(mip_preview, [0.1, 0.2, 0.3, 1.0])
    .copy_image_to_buffer(mip_preview, readback)
    .fill_buffer(readback, 0..64, 0);
# Ok(()) }
```

## Choosing The Right Command

- Use `update_buffer` for small inline uploads that fit in Vulkan's `cmd_update_buffer` limits.
- Use `fill_buffer` when you need a repeated `u32` pattern, often for resets or counters.
- Use `copy_buffer_to_image` and `copy_image_to_buffer` for upload and readback paths.
- Use `copy_image` when source and destination texel footprints already match.
- Use `blit_image` when you need scaling or filtering.
- Use the `*_region` variants when you need precise offsets, layers, mip levels, or partial copies.

## Region Variants

Each built-in helper also has a more explicit form such as `copy_buffer_region` or
`copy_buffer_to_image_region`. Use those variants when the whole-resource convenience behavior is
too broad.

```no_run
# use vk_graph::Graph;
# use vk_graph::driver::DriverError;
# use vk_graph::driver::ash::vk;
# use vk_graph::driver::buffer::{Buffer, BufferInfo};
# use vk_graph::driver::device::{Device, DeviceInfo};
# fn main() -> Result<(), DriverError> {
# let device = Device::new(DeviceInfo::default())?;
let mut graph = Graph::default();

let src = graph.bind_resource(Buffer::create(
    &device,
    BufferInfo::host_mem(4096, vk::BufferUsageFlags::TRANSFER_SRC),
)?);
let dst = graph.bind_resource(Buffer::create(
    &device,
    BufferInfo::device_mem(4096, vk::BufferUsageFlags::TRANSFER_DST),
)?);

graph.copy_buffer_region(
    src,
    dst,
    [vk::BufferCopy {
        src_offset: 512,
        dst_offset: 1024,
        size: 256,
    }],
);
# Ok(()) }
```
