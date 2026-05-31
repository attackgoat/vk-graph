# Images

```rust
# use vk_graph::Graph;
# use vk_graph::driver::{DriverError, ash::vk, device::Device};
# use vk_graph::driver::image::{Image, ImageInfo, ImageInfoBuilder, SampleCount};
# use vk_graph::driver::image::{ImageViewInfo, ImageViewInfoBuilder};
# fn test(
#     device: &Device,
# ) -> Result<(), DriverError> {
let (width, height) = (320, 200);
let usage = vk::ImageUsageFlags::SAMPLED;
let fmt = vk::Format::R8G8B8A8_UNORM;

// Create image info multiple ways
let info = ImageInfo {
    array_layer_count: 1,
    dedicated: false,
    depth: 1,
    flags: vk::ImageCreateFlags::empty(),
    fmt,
    height,
    mip_level_count: 1,
    sample_count: SampleCount::Type1,
    tiling: vk::ImageTiling::OPTIMAL,
    ty: vk::ImageType::TYPE_2D,
    usage,
    width,
};
let other_info = ImageInfo::image_2d(width, height, fmt, usage);
let cube_info = ImageInfo::cube(width, fmt, usage);

assert_eq!(info, other_info);
assert_ne!(info, cube_info);

// Builder pattern
let same_info = ImageInfoBuilder::default()
    .width(width)
    .height(height)
    .depth(1)
    .fmt(fmt)
    .usage(usage)
    .ty(vk::ImageType::TYPE_2D);

// Info built from other info
let array_info = cube_info
    .into_builder()
    .flags(vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE)
    .build();

// Images are created simply
let image = Image::create(device, info)?;

// For interop this may be handy:
let image = Image::from_raw(device, vk::Image::null(), info);

// The provided fields are helpful:
assert_eq!(image.device, *device);
assert_eq!(image.info, info);
assert_ne!(image.handle, vk::Image::null());

// Image "subresources" are the native type:
let my_subresource = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

// Image views are also subresources:
let image_view = ImageViewInfo {
    array_layer_count: 1,
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_array_layer: 0,
    base_mip_level: 0,
    fmt,
    mip_level_count: 1,
    ty: vk::ImageViewType::TYPE_2D,
};

// Image views have the same builder functionality:
let other_view = ImageViewInfoBuilder::default();

// Image views can be inferred from the whole image info:
let addl_view = info.into_image_view();

assert_eq!(image_view, addl_view);
# Ok(()) }
```
