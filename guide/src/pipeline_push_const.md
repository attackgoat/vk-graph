# Push Constants

Command buffers may update a very small data cache which shaders may read during execution using
push constants.

It is recommended to target 128 bytes as the maximum push constant data size.

```glsl
// render_mesh.glsl
#version 460 core

layout(push_constant) uniform PushConstants {
    layout(offset = 0) uint mesh_index;
};

...
```

```rust
# macro_rules! include_bytes { ($path:expr) => { [0u8] }; }
# use vk_graph::Graph;
# use vk_graph::driver::{DriverError, device::Device};
# use vk_graph::driver::compute::{ComputePipeline, ComputePipelineInfo};
# use vk_graph::driver::shader::Shader;
# fn test(device: &Device) -> Result<(), DriverError> {
let info = ComputePipelineInfo::default();
let code = include_bytes!("render_mesh.spv");
let shader = Shader::new_compute(code.as_slice());
let pipeline = ComputePipeline::create(device, info, shader)?;

let mut graph = Graph::default();
let data = 42u32.to_ne_bytes();

graph
    .begin_cmd()
    .bind_pipeline(&pipeline)
    .record_cmd(move |cmd| {
        cmd
            .push_constants(0, &data)
            .dispatch(1, 1, 1);
    });
# Ok(()) }
```

> [!TIP]
> A crate such as `bytemuck` is helpful for converting Rust structures to bytes suitable for push
> constant usage. See the example code for more.
