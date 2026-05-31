# Hot Reload

An accessory crate is provided to support automatic reloading of changed shader pipelines.

`vk-graph-hot` uses a file watcher and `Shaderc`. It may be used directly or may be swapped out
using a build feature:

```toml
# Cargo.toml

[features]
default = []
hot = ["dep:vk-graph-hot"]

[dependencies]
vk-graph = "{{ crate.version }}"
vk-graph-hot = { version = "{{ vk-graph-hot.version }}", optional = true }
```

```rust
# macro_rules! include_bytes { ($path:expr) => { [0u8] }; }
use vk_graph::driver::{DriverError, compute::ComputePipelineInfo, device::Device};

#[cfg(feature = "hot")]
use vk_graph_hot::{
    HotComputePipeline as ComputePipeline,
    HotShader,
};

#[cfg(not(feature = "hot"))]
use vk_graph::driver::{
    compute::ComputePipeline,
    shader::Shader,
};

pub fn create_fire_pipeline(
    device: &Device,
) -> Result<ComputePipeline, DriverError> {
    let info = ComputePipelineInfo::default();

    #[cfg(feature = "hot")]
    let shader = HotShader::from_path("fire.glsl");

    #[cfg(not(feature = "hot"))]
    let shader = Shader::from_spirv(include_bytes!("fire.spv").as_slice());

    ComputePipeline::create(device, info, shader)
}
```

> [!NOTE]
> The hot versions of each type support all features, options, and usage provided by the normal
> types. This include public fields, available information, and graph binding features.
