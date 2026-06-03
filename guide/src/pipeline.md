# Pipelines

> [!CAUTION]
> All pipelines and resources (_buffers, images, and acceleration structures_) used in a `Graph`
> must have been created using the same `Device`.

Pipelines are created from `Device` references. They may be bound to graph commands.

```rust
# macro_rules! include_bytes { ($path:expr) => { [0u8] }; }
# use vk_graph::Graph;
# use vk_graph::driver::{DriverError, device::Device};
# use vk_graph::driver::compute::{ComputePipeline, ComputePipelineInfo};
# fn test(device: &Device) -> Result<(), DriverError> {
let info = ComputePipelineInfo::default();
let shader = include_bytes!("shader.spv");
let pipeline = ComputePipeline::create(device, info, shader.as_slice())?;

let mut graph = Graph::default()
    .begin_cmd()
    .bind_pipeline(&pipeline)
    .record_cmd(|cmd| {
        // Record vulkan commands here
    })
    .end_cmd();
# Ok(()) }
```

Pipelines are cheap to `Clone` and should be cached in between use. The recommendation is to bind a
borrow of a pipeline to when beginning a command.

## Commands

A graph command is the smallest unit which the `Submission` type will schedule for execution.

Calls to `Graph::begin_cmd` (and, optionally `Graph::end_cmd`) define a single graph command which
will execute in physical order as recorded. During graph command recording you may change pipelines,
modify shader descriptor bindings, or otherwise modify the state of the command buffer.

Example:

```rust
# macro_rules! include_bytes { ($path:expr) => { [0u8] }; }
# use vk_graph::Graph;
# use vk_graph::driver::{DriverError, device::Device};
# use vk_graph::driver::compute::{ComputePipeline, ComputePipelineInfo};
# fn test(device: &Device) -> Result<(), DriverError> {
let info = ComputePipelineInfo::default();

let fire = include_bytes!("fire.spv");
let fire = ComputePipeline::create(device, info, fire.as_slice())?;

let water = include_bytes!("water.spv");
let water = ComputePipeline::create(device, info, water.as_slice())?;

let mut graph = Graph::default();
graph
    .begin_cmd()
    .bind_pipeline(&fire)
    .record_cmd(|cmd| {
        println!("1st");
    })
    .bind_pipeline(&water)
    .record_cmd(|cmd| {
        println!("2nd");
    })
    .bind_pipeline(&fire)
    .record_cmd(|cmd| {
        println!("3rd");
    })
    .end_cmd()
    .begin_cmd()
    .bind_pipeline(&water)
    .record_cmd(|cmd| {
        println!("4th");
    });
# Ok(()) }
```

A call to `Graph::end_cmd` is not required. The _end-command_ method exists to support builder-style
function-chaining. In the above example two commands are built and added to the graph.

## Shaders

Compute, graphics, and ray tracing pipelines require one or more shaders:

Pipeline Type|Shaders
--|--
`ComputePipeline`|Single: must be compute stage
`GraphicsPipeline`|Multiple: must be a raster stage
`RayTracingPipeline`|Multiple: must be a ray tracing stage

> [!CAUTION]
> All `Shader` constructors panic when provided with invalid SPIR-V shader code.

The `Shader` type uses a builder pattern:

```rust
# macro_rules! include_bytes { ($path:expr) => { [0u8] }; }
# use vk_graph::Graph;
# use vk_graph::driver::{DriverError, device::Device};
# use vk_graph::driver::compute::{ComputePipeline, ComputePipelineInfo};
# use vk_graph::driver::shader::{SamplerInfo, Shader};
# fn test(device: &Device) -> Result<(), DriverError> {
// Pipelines may be created using "shader" or "custom":
let code = include_bytes!("raygen.spv");
let shader = Shader::from_spirv(code.as_slice());
let custom = shader
                .entry_name("main_but_faster")
                .image_sampler(0, SamplerInfo::default())
                .image_sampler(1, SamplerInfo::LINEAR);
# Ok(()) }
```
