# Push Constants

Command buffers may update a very small data cache which shaders may read during execution using
push constants. It is recommended to target 128 bytes as the maximum push constant data size.

```glsl
// render_mesh.glsl
#version 460 core

layout(push_constant) uniform PushConstants {
    layout(offset = 0) uint mesh_index;
};

...
```

```rust
let info = ComputePipelineInfo::default();
let code = include_bytes!("render_mesh.spv");
let shader = Shader::new_compute(code.as_slice());
let pipeline = ComputePipeline::create(device, info, shader)?;

let mut graph = Graph::default();
graph
    .begin_cmd()
    .bind_pipeline(&pipeline)
    .record_cmd_buf(|cmd_buf| {
        cmd_buf
            .push_constants(0, 42u32.to_ne_bytes())
            .dispatch(1, 1, 1);
    });
```

> [!TIP]
> A crate such as `bytemuck` is helpful for converting Rust structures to bytes suitable for push
> constant usage. See the example code for more.
