# Pipelines

> [!CAUTION]
> All pipelines and resources (_buffers, images, and acceleration structures_) used in a `Graph`
> must have been created using the same `Device`.

Pipelines are created from `Device` references. They may be bound to graph commands.

```rust
let info = ComputePipelineInfo::default();
let shader = include_bytes!("shader.spv");
let pipeline = ComputePipeline::create(device, info, shader)?;

let mut graph = Graph::default()
    .begin_cmd()
    .bind_pipeline(&pipeline)
    .record_cmd_buf(|cmd_buf| {
        // Record vulkan commands here
    })
    .end_cmd();
```

Pipelines are cheap to `Clone` and should be cached in between use. The recommendation is to bind a
borrow of a pipeline to when beginning a command.

## Commands

A graph command is the smallest unit which the `Queue` type will schedule for execution.

Calls to `Graph::begin_cmd` (and, optionally `Graph::end_cmd`) define a single graph command which
will execute in physical order as recorded. During graph command recording you may change pipelines,
modify shader descriptor bindings, or otherwise modify the state of the command buffer.

Example:

```rust
let fire_pipeline = ComputePipeline::create(device, include_bytes!("fire.spv"))?;
let water_pipeline = ComputePipeline::create(device, include_bytes!("water.spv"))?;

let mut graph = Graph::default();
graph
    .begin_cmd()
    .bind_pipeline(&fire_pipeline)
    .record_cmd_buf(|cmd_buf| todo!("1st"))
    .bind_pipeline(&water_pipeline)
    .record_cmd_buf(|cmd_buf| todo!("2nd"))
    .bind_pipeline(&fire_pipeline)
    .record_cmd_buf(|cmd_buf| todo!("3rd"));
```

A call to `Graph::end_cmd` is never requried and the command is automatically ended. The call may be
useful for builder-pattern code which is building a very large series of commands.

## Shaders

Compute, graphic, and ray trace pipelines require one or more shaders:

Pipeline Type|Shaders
--|--
`ComputePipeline`|Single: must be compute stage
`GraphicPipeline`|Multiple: must be a raster stage
`RayTracePipeline`|Multiple: must be a ray tracing stage

> [!CAUTION]
> All `Shader` constructors panic when provided with invalid SPIR-V shader code.

The `Shader` type uses a builder pattern:

```rust
// Pipelines may be created using "code", "shader", or "custom":
let code = include_bytes!("raygen.spv");
let shader = Shader::from_spirv(code.as_slice());
let custom = shader
                .entry_name("main_but_faster")
                .image_sampler(0, SamplerInfo::default())
                .image_sampler(1, SamplerInfo::LINEAR);
```
