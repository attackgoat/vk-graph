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

```bash
glslc shader.glsl -o shader.spv
```

```rust
# macro_rules! include_bytes { ($path:expr) => { [0u8] }; }
# use vk_graph::driver::shader::Shader;
let compile_time_spirv = include_bytes!("shader.spv");

// #pragma allows for from_spirv syntax:
let shader = Shader::from_spirv(
    compile_time_spirv.as_slice(),
);

// Without this #pragma we must specify stage:
let shader = Shader::new_compute(
    compile_time_spirv.as_slice(),
);

// For dynamically loaded SPIR-V (e.g. from a file at runtime), use try_new_*
// to handle invalid shader code gracefully:
# fn load_spirv_from_disk() -> Vec<u8> { vec![] }
let runtime_spirv = load_spirv_from_disk();
match Shader::try_new_compute(runtime_spirv.as_slice()) {
    Ok(_shader) => { /* use shader */ }
    Err(e) => { eprintln!("Shader compilation failed: {e}"); }
}
```
