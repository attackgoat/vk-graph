# Shader Compilation

`vk-graph` does not provide any shader compiler or require any specific shading language. Users must
provide SPIR-V binary-format shaders.

> [!TIP]
> See [Hot Reload](./pipeline_hot_reload.md) for details on a shader compiler provided as a separate
> crate.

Examples using multiple shading languages and compilers are provided in the
[_`examples/`_](https://github.com/attackgoat/vk-graph/tree/main/examples)
<i class="fa-solid fa-arrow-up-right-from-square"></i> directory.

## Shader-stage `#pragma`

This applies to GLSL and Shaderc generally but you might find similar functionality with other
languages and compilers.

```glsl
// shader.glsl
#version 460 core
#pragma shader_stage(compute)

void main() {
    // Some code here
}
```

```rust
let spirv = include_glsl!("shader.glsl");

// #pragma allows for from_spirv syntax:
let vertex_shader = Shader::from_spirv(
    spirv.as_slice(),
);

// Without this #pragma we must specify stage:
let vertex_shader = Shader::new_compute(
    spirv.as_slice(),
);
```
