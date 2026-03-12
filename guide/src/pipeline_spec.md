# Specialization

Pipeline specialization allows pre-compiled SPIR-V binary shaders to be specialized with constant
values specified at run-time.

The Vulkan implementation may use these constant values to generate optimized shader code.

`vk-graph` provides `SpecializationMap` as an easy-to-use way of storing the data and lookup entries
required to use this feature.

```glsl
// kaboom.glsl
#version 460 core

layout(constant_id = 0) const float INFERNO_EPSILON = 0.999;
layout(constant_id = 1) const float COEFF_OF_BOOM = 1.4;

...
```

```rust
// Use this shader for the glsl-specified values:
let shader = Shader::new_compute(include_bytes!("kaboom.spv").as_slice());

let specialization = SpecializationMap::new(bytemuck::bytes_of([
        0.99999f32,
        1.0,
    ]))
    .constant(0, 0, 4)
    .constant(1, 4, 8);

// Use this shader for the updated run-time values:
let spec_shader = shader.specialization(specialization);
....
```
