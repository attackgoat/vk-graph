# Debugging

`vk-graph` uses [`log`](https://crates.io/crates/log) for low-overhead logging.

To enable logging, set the `RUST_LOG` environment variable to `trace`, `debug`, `info`, `warn` or
`error` and initialize the logging provider of your choice. Examples use
[`pretty_env_logger`](https://docs.rs/pretty_env_logger/latest/pretty_env_logger/).

_You may also filter messages, for example:_

```bash
RUST_LOG=vk_graph::driver=trace,vk_graph=warn cargo run --example ray_trace
```

```
TRACE vk_graph::driver::instance > created a Vulkan instance
DEBUG vk_graph::driver::physical_device > physical device: NVIDIA GeForce RTX 3090
DEBUG vk_graph::driver::physical_device > extension "VK_KHR_16bit_storage" v1
DEBUG vk_graph::driver::physical_device > extension "VK_KHR_8bit_storage" v1
DEBUG vk_graph::driver::physical_device > extension "VK_KHR_acceleration_structure" v13
...
```

## Performance Profiling

`vk-graph` uses [`profiling`](https://crates.io/crates/profiling) and supports multiple profiling
providers. When not in use profiling has zero cost.

To enable profiling, compile with one of the `profile-with-*` features enabled and initialize the
profiling provider of your choice.

_Example code uses [puffin](https://crates.io/crates/puffin):_

```bash
cargo run --features profile-with-puffin --release --example vsm_omni
```

<img src="profile.png" alt="Flamegraph of performance data" width=30%>


### Comparing Results

Always profile code using a release-mode build.

You may need to disable CPU thermal throttling in order to get consistent results on some platforms.
The inconsistent results are certainly valid, but they do not help in accurately measuring potential
changes. This may be done on Intel Linux machines by modifying the Intel P-State driver:

```bash
echo 100 | sudo tee /sys/devices/system/cpu/intel_pstate/min_perf_pct
```

(_[Source](https://www.kernel.org/doc/Documentation/cpu-freq/intel-pstate.txt)_)

## Helpful tools

- [VulkanSDK](https://vulkan.lunarg.com/sdk/home) _(Required when setting `debug` to `true`)_
- NVIDIA: [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface)
- AMD: [RadeonTop](https://github.com/clbr/radeontop)
- [RenderDoc](https://renderdoc.org/)
