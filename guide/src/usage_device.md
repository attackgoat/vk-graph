# Device Creation

Most Vulkan operations occur within the context of a logical device, provided by
`Device` (_a smart pointer for `ash::Device`_).

API docs: [`Device::create`](https://docs.rs/vk-graph/latest/vk_graph/driver/device/struct.Device.html#method.create),
[`Device::try_from_ash`](https://docs.rs/vk-graph/latest/vk_graph/driver/device/struct.Device.html#method.try_from_ash),
[`Device::try_from_display`](https://docs.rs/vk-graph/latest/vk_graph/driver/device/struct.Device.html#method.try_from_display).

> [!WARNING]
> Vulkan has no global state and does not share resources between devices by default.
>
> Do not combine resources from multiple devices! The steps required to share resources across
> devices are not currently documented.

## Headless Operation

For any sort of server-based rendering or similar Vulkan usage without a display, the following is
production-ready code used to create a device:

```rust
# use vk_graph::driver::DriverError;
# use vk_graph::driver::device::{Device, DeviceInfo};
# fn test() -> Result<(), DriverError> {
let info = DeviceInfo::default();
let device = Device::create(info)?;

assert_eq!(device.physical.instance.info.debug, false);
# Ok(()) }
```

## Background Fence Cleanup

Fence payloads are normally dropped on the thread that observes completion. A device can instead
queue that destruction, including returning command buffers to their pools, on a background worker:

```rust
# use vk_graph::driver::DriverError;
# use vk_graph::driver::device::{Device, DeviceInfo};
# fn test() -> Result<(), DriverError> {
let device = Device::create(DeviceInfo::default())?;
let fence_cleanup = Device::enable_background_fence_cleanup(&device)?;

// Payloads are queued for background cleanup after a fence operation observes completion.

fence_cleanup.wait_for_pending_cleanup()?;
# Ok(()) }
```

`BackgroundFenceCleanupGuard::wait_for_pending_cleanup` waits for fence payloads queued to the
guard's current worker before the call. It does not poll or wait for fences: `Fence::status`,
`Fence::wait`, `Fence::reset`, or fence destruction must first observe completion and queue the
payload. Fence waits and payload completion hooks remain synchronous. The method returns
`DriverError::InvalidData` if the worker reports a payload-drop failure since its previous wait or
cannot complete the wait. Failure reporting is shared and best-effort, so concurrent waiters are not
guaranteed to observe the same failure.

Guards active at the same time for a device share a worker. Dropping the final guard signals that
worker to drain and detaches it without blocking, after which newly completed fence payloads are
dropped synchronously. Re-enabling cleanup can start a new worker while a detached worker finishes
draining; waits apply only to the current worker. The worker queue is bounded; when it is full,
payloads fall back to being dropped synchronously rather than accumulating without limit. Panics
during synchronous fallback are caught and logged rather than reported by a later cleanup wait.

## Windowed Operation

Prototype and demo code might use the built-in window handler, which creates a `Device` during
window creation:

```toml
# Cargo.toml

[dependencies]
vk-graph-window = "{{ vk-graph-window.version }}"
```

```rust
# use vk_graph::driver::device::Device;
# use vk_graph_window::WindowError;
# fn test() -> Result<(), WindowError> {
use vk_graph_window::WindowBuilder;

let window = WindowBuilder::default().build()?;

// Before run
let _: &Device = &window.device;

window.run(|frame| {
    // During any frame
    let _: &Device = frame.device;
})?;
# Ok(()) }
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
# use vk_graph::driver::DriverError;
# use vk_graph::driver::device::Device;
# use vk_graph::driver::instance::{Instance, InstanceInfo};
# fn test() -> Result<(), DriverError> {
let instance = Instance::create(InstanceInfo::default())?;
let physical_devices = Instance::physical_devices(&instance)?;

for physical_device in physical_devices {
    // We are looking for a device with support for these features
    if !physical_device.vk_khr_swapchain
        || !physical_device
            .vk_khr_ray_tracing_pipeline
            .as_ref()
            .is_some_and(|ext| ext.features.ray_tracing_pipeline)
    {
        continue;
    }

    let _: Device = physical_device.try_into_device()?;
}
# Ok(()) }
```

### Native Device Usage

Some scenarios require the Vulkan instance and/or device be created by other code and accepted for
use by `vk-graph`:

```rust
# use vk_graph::Graph;
# use vk_graph::driver::DriverError;
# use vk_graph::driver::ash::{self, vk};
# use vk_graph::driver::device::Device;
# use vk_graph::driver::instance::Instance;
# use vk_graph::driver::physical_device::PhysicalDevice;
# fn test() -> Result<(), DriverError> {
// Native ash types from somewhere else
let entry: ash::Entry = todo!();
let instance: vk::Instance = todo!();
let physical_device: vk::PhysicalDevice = todo!();

// vk-graph types
let instance = Instance::try_from_entry(entry, instance)?;
let physical_device = unsafe { PhysicalDevice::try_from_ash(&instance, physical_device) }?;

// Use our PhysicalDevice to create a native ash::Device (OpenXR requires this)
let device: ash::Device = unsafe {
    physical_device
        .create_ash_device(|create_info| {
            // Somewhere else also provides the logical device!
            let device: vk::Device = todo!();

            let device: ash::Device = unsafe {
                ash::Device::load(instance.fp_v1_0(), device)
            };

            Ok(device)
        })
}.unwrap();

// Create a Device from their native stuff
let device = unsafe { Device::try_from_ash(device, physical_device) }?;
# Ok(()) }
```

> [!TIP]
> See [_`examples/vr`_](https://github.com/attackgoat/vk-graph/tree/main/examples/vr)
> <i class="fa-solid fa-arrow-up-right-from-square"></i> for an in-depth example of native device
> usage.
