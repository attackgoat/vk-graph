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
```

```rust
# macro_rules! include_bytes { ($path:expr) => { [0u8] }; }
# use vk_graph::driver::{DriverError, device::Device};
# use vk_graph::driver::shader::{Shader, SpecializationMap};
# fn test(device: &Device) -> Result<(), DriverError> {
use bytemuck::bytes_of;

let kaboom = include_bytes!("kaboom.spv");

// Use this shader for the glsl-specified values:
let shader = Shader::new_compute(kaboom.as_slice());

let better_consts = [
    0.99999f32,
    1.0,
];
let better_consts = bytes_of(&better_consts);
let spec = SpecializationMap::new(better_consts)
    .constant(0, 0, 4)
    .constant(1, 4, 8);

// Use this shader for the updated run-time values:
let spec_shader = shader.specialization(spec);
# Ok(()) }
```
