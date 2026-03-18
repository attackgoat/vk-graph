# Buffers

```rust
let size = 1_024;
let usage = vk::BufferUsageFlags::STORAGE;

// Create buffer info multiple ways:
let info = BufferInfo {
    alignment: 1,
    dedicated: false,
    host_read: false,
    host_write: false,
    size,
    usage,
};
let device_mem = BufferInfo::device_mem(size, usage);
let host_mem = BufferInfo::host_mem(size, usage);

assert_eq!(info, device_mem);
assert_ne!(info, host_mem);

// Builder pattern
let same_info = BufferInfoBuilder::default()
    .size(size)
    .usage(usage);

// Info built from other info
let more_info = host_mem
    .info
    .into_builder()
    .usage(usage | vk::BufferUsageFlags::INDIRECT_BUFFER)
    .build();

// There is a helper function for creating buffers from a slice
let data = [1u8, 2, 3, 4];
let buffer = Buffer::create_from_slice(device, usage, &data)?;

// This is equivalent to:
let mut buffer = Buffer::create(device, host_mem)?;
buffer.copy_from_slice(&data);

// Or use the std copy_from_slice (it panics if size != range)
let mut buffer = Buffer::create(device, host_mem)?;
buffer.mapped_slice_mut().copy_from_slice(&data);

// The provided fields are helpful:
assert_eq!(buffer.device, *device);
assert_eq!(buffer.info, host_mem);
assert_ne!(buffer.handle, vk::Buffer::null());

// Buffer "subresources" are just ranges of that buffer
let my_subresource = 0..size;
```
