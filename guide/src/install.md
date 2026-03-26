# Installation

To get started with `vk-graph`, add it as a project dependency to your `Cargo.toml`:

```toml
# Cargo.toml

[dependencies]
vk-graph = "{{ crate.version }}"
```

## Features

_vk-graph_ puts a lot of functionality behind optional features in order to optimize
compile time for the most common use cases. The following features are
available.

- **`loaded`** *(enabled by default)* — Support searching for the Vulkan loader manually at runtime.
- **`linked`** — Link the Vulkan loader at compile time.
- **`profile_with_`** — Use the specified profiling backend
    - ...**`puffin`**
    - ...**`optick`**
    - ...**`superluminal`**
    - ...**`tracy`**

## Vulkan SDK

Debug mode (setting the `debug` field of `DeviceInfo` or `InstanceInfo` to `true`) is supported only
when a compatible [_Vulkan SDK_](https://vulkan.lunarg.com/sdk/home)
<i class="fa-solid fa-arrow-up-right-from-square"></i> is installed.

> [!IMPORTANT]
> The installed Vulkan SDK version must be at least v{{ vulkan_sdk.version }}.

## Required Packages

_Linux (Debian-like)_:
- `sudo apt install cmake uuid-dev libfontconfig-dev libssl-dev`

_Mac OS (10.15 or later)_:
- Xcode 12
- Python 2.7
- `brew install cmake ossp-uuid`

_Windows_:
- TODO
