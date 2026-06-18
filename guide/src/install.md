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

- **`checked`** *(enabled by default)* — Enable runtime validation for common misuse patterns that
  Vulkan validation layers cannot always catch.
- **`loaded`** *(enabled by default)* — Support searching for the Vulkan loader manually at runtime.
- **`linked`** — Link the Vulkan loader at compile time.
- **`parking_lot`** *(enabled by default)* — Use `parking_lot` synchronization primitives internally.
- **`ash-molten`** — Enable MoltenVK loading support on macOS.
- **`profile-with-*`** — Use the specified profiling backend:
  `profile-with-puffin`, `profile-with-optick`, `profile-with-superluminal`, or
  `profile-with-tracy`

## Required Development Packages

_Linux (Debian-like)_:
- `sudo apt install cmake uuid-dev libfontconfig-dev libssl-dev`

_Mac OS (10.15 or later)_:
- Xcode 12
- Python 2.7
- `brew install cmake ossp-uuid`

_Windows_:
- Install the Vulkan SDK and the current Visual Studio C++ build tools.

## Vulkan SDK

Debug mode (setting the `debug` field of `DeviceInfo` or `InstanceInfo` to `true`) is only supported
when certain validation layers are installed. The [_Vulkan SDK_](https://vulkan.lunarg.com/sdk/home)
<i class="fa-solid fa-arrow-up-right-from-square"></i> provides these layers and a number of helpful
tools.

> [!IMPORTANT]
> The installed Vulkan SDK version must be at least v{{ vulkan_sdk.version }}.

### Optional Distribution-Provided Validation Layers

_Linux (Debian-like)_:
- `sudo apt install vulkan-validationlayers`
