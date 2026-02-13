# Getting Started with _vk-graph_

This guide is intended for developers who are new to _vk-graph_ and want a step-by-step introduction. For further details on these
topics refer to the online [documentation](https://docs.rs/vk-graph/latest/vk_graph/).

## Required Packages

_Linux (Debian-like)_:
- `sudo apt install cmake uuid-dev libfontconfig-dev libssl-dev`

_Mac OS (10.15 or later)_:
- Xcode 12
- Python 2.7
- `brew install cmake ossp-uuid`

_Windows_:
- TODO (works but I haven't gathered the requirements)

## Documentation

Read the generated [documentation](https://docs.rs/vk-graph/latest/vk_graph/) online, or run the
following command locally:

```
cargo doc --open
```

## Changes

Stay informed of recent changes to _vk-graph_ using the
[change log](https://github.com/attackgoat/vk-graph/blob/master/CHANGELOG.md) file.

## Performance Profiling

Most of the example code (_the ones which use an `EventLoop`_) support profiling using `puffin`. To
use it, first install and run `puffin_viewer` and then run the example with the
`--features profile-with-puffin` and `--release` flags.

You may need to disable CPU thermal throttling in order to get consistent results on some platforms.
The inconsistent results are certainly valid, but they do not help in accurately measuring potential
changes. This may be done on Intel Linux machines by modifying the Intel P-State driver:

```bash
echo 100 | sudo tee /sys/devices/system/cpu/intel_pstate/min_perf_pct
```

(_[Source](https://www.kernel.org/doc/Documentation/cpu-freq/intel-pstate.txt)_)

## Helpful tools

- [VulkanSDK](https://vulkan.lunarg.com/sdk/home) _(Required when calling `EventLoop::debug(true)`)_
- NVIDIA: [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface)
- AMD: [RadeonTop](https://github.com/clbr/radeontop)
- [RenderDoc](https://renderdoc.org/)
