# Device Creation

Most Vulkan operations occur within the context of a logical device, provided by
`Device` (_a smart pointer for `ash::Device`_).

> [!WARNING]
> Vulkan has no global state and does not share resources between devices by default.
>
> Do not combine resources from multiple devices! The steps required to share resources across
> devices are not currently documented.

## Headless Operation

For any sort of server-based rendering or similar Vulkan usage without a display, the following is
production-ready code used to create a device:

```rust
let info = DeviceInfo::default();
let device = Device::new(info)?;

assert_eq!(device.info, DeviceInfo {
    debug: false,
    physical_device_index: 0,
});
```

## Windowed Operation

Prototype and demo code might use the built-in window handler, which creates a `Device` during
window creation:

```toml
# Cargo.toml

[dependencies]
vk-graph-window = "{{ crate.version }}"
```

```rust
use vk_graph_window::WindowBuilder;

let window = WindowBuilder::default().build()?;

// Before run
let _: &Device = &window.device;

window.run(|frame| {
    // During any frame
    let _: &Device = frame.device;
})?;
```

## Advanced

There are several scenarios that require advanced `Device` creation techniques:

- Allowing user-selection of device
- Custom Window(s) handling
- FFI with OpenXR (_or similar_)
- Unsupported drivers/platforms

### Device Selection

The entrypoint is an `Instance` from which the available hardware is enumerated and inspected:

```rust
let instance = Instance::new(InstanceInfo::default())?;
let physical_devices = Instance::physical_devices(&instance)?;

for physical_device in physical_devices {
    // We are looking for a device with support for these features
    if !physical_device.swapchain_ext
    || !physical_device.ray_trace_features.ray_tracing_pipeline {
        continue;
    }

    let _: Device = physical_device.try_into_device()?;
}
```

### Native Device Usage

Some scenarios require the Vulkan instance and/or device be created by other code and accepted for
use by `vk-graph`:

```rust
// Native ash types from somewhere else
let entry: ash::Entry = todo!();
let instance: vk::Instance = todo!();
let physical_device: vk::PhysicalDevice = todo!();

// vk-graph types
let instance = Instance::from_entry(entry, instance)?;
let physical_device = Instance::phsyical_device(&instance, physical_device)?;

// Use our PhysicalDevice to create a native ash::Device (OpenXR requires this pattern)
let device: ash::Device = physical_device
    .create_ash_device(|create_info| {
        // Somewhere else also provides the logical device!
        let device: vk::Device = todo!();

        Ok(ash::Device::load(instance.fp_v1_0(), device))
    })?;

// Create a Device from their native stuff
let device = Device::try_from_ash_device(device, physical_device)?;
```

> [!TIP]
> See [_`examples/vr`_](https://github.com/attackgoat/vk-graph/tree/main/examples/vr)
> <i class="fa-solid fa-arrow-up-right-from-square"></i> for an in-depth example of native device
> usage.
