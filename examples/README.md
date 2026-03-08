# _vk-graph_ Example Code

## Getting Started

A helpful [guide](https://attackgoat.github.io/vk-graph) is available which describes _vk-graph_
types and functions.

See the [README](../README.md) for more information.

## Example Code

Example | Instructions | Preview
 --- | --- | :---:
[aliasing.rs](aliasing.rs) | <pre>cargo run --example aliasing</pre> | _See console output_
[cpu_readback.rs](cpu_readback.rs) | <pre>cargo run --example cpu_readback</pre> | _See console output_
[debugger.rs](debugger.rs) | <pre>cargo run --example debugger</pre> | _See console output_
[min_max.rs](min_max.rs) | <pre>cargo run --example min_max</pre> | _See console output_
[mip_compute.rs](mip_compute.rs) | <pre>cargo run --example mip_compute</pre> | _See console output_
[subgroup_ops.rs](subgroup_ops.rs) | <pre>cargo run --example subgroup_ops</pre> | _See console output_
[hello_world.rs](../crates/vk-graph-window/examples/hello_world.rs) | _See [vk-graph-window](../crates/vk-graph-window/README.md)_ | <image alt="hello_world.rs" src="../.github/img/hello_world.png" width="176" height="150">
[app.rs](app.rs) | <pre>cargo run --example app</pre> | <image alt="app.rs" src="../.github/img/app.png" width="176" height="150">
[triangle.rs](triangle.rs) | <pre>cargo run --example triangle</pre> | <image alt="triangle.rs" src="../.github/img/triangle.png" width="176" height="150">
[vertex_layout.rs](vertex_layout.rs) | <pre>cargo run --example vertex_layout</pre> | <image alt="vertex_layout.rs" src="../.github/img/vertex_layout.png" width="176" height="150">
[bindless.rs](bindless.rs) | <pre>cargo run --example bindless</pre> | <image alt="bindless.rs" src="../.github/img/bindless.png" width="176" height="188">
[image_sampler.rs](image_sampler.rs) | <pre>cargo run --example image_sampler</pre> | <image alt="image_sampler.rs" src="../.github/img/image_sampler.png" width="176" height="150">
[egui.rs](egui.rs) | <pre>cargo run --example egui</pre> | <image alt="egui.rs" src="../.github/img/egui.png" width="176" height="150">
[imgui.rs](imgui.rs) | <pre>cargo run --example imgui</pre> | <image alt="imgui.rs" src="../.github/img/imgui.png" width="176" height="150">
[font_bmp.rs](font_bmp.rs) | <pre>cargo run --example font_bmp</pre> | <image alt="font_bmp.rs" src="../.github/img/font_bmp.png" width="176" height="150">
[mip_graphic.rs](mip_graphic.rs) | <pre>cargo run --example mip_graphic</pre> | <image alt="mip_graphic.rs" src="../.github/img/mip_graphic.png" width="176" height="150">
[multipass.rs](multipass.rs) | <pre>cargo run --example multipass</pre> | <image alt="multipass.rs" src="../.github/img/multipass.png" width="176" height="150">
[multithread.rs](multithread.rs) | <pre>cargo run --example multithread --release</pre> | <image alt="multithread.rs" src="../.github/img/multithread.png" width="176" height="150">
[msaa.rs](msaa.rs) | <pre>cargo run --example msaa</pre> Multisample anti-aliasing | <image alt="msaa.rs" src="../.github/img/msaa.png" width="176" height="150">
[rt_triangle.rs](rt_triangle.rs) | <pre>cargo run --example rt_triangle</pre> | <image alt="rt_triangle.rs" src="../.github/img/rt_triangle.png" width="176" height="150">
[ray_trace.rs](ray_trace.rs) | <pre>cargo run --example ray_trace</pre> | <image alt="ray_trace.rs" src="../.github/img/ray_trace.png" width="176" height="150">
[vsm_omni.rs](vsm_omni.rs) | <pre>cargo run --example vsm_omni</pre> Variance shadow mapping for omni/point lights | <image alt="vsm_omni.rs" src="../.github/img/vsm_omni.png" width="176" height="150">
[ray_omni.rs](ray_omni.rs) | <pre>cargo run --example ray_omni</pre> Ray query for omni/point lights | <image alt="ray_omni.rs" src="../.github/img/ray_omni.png" width="176" height="150">
[transitions.rs](transitions.rs) | <pre>cargo run --example transitions</pre> | <image alt="transitions.rs" src="../.github/img/transitions.png" width="176" height="150">
[skeletal-anim/](skeletal-anim/src/main.rs) | <pre>cargo run --manifest-path examples/skeletal-anim/Cargo.toml</pre> Skeletal mesh animation using GLTF | <image alt="skeletal-anim" src="../.github/img/skeletal-anim.png" width="176" height="150">
[shader-toy/](shader-toy/src/main.rs) | <pre>cargo run --manifest-path examples/shader-toy/Cargo.toml</pre> | <image alt="shader-toy" src="../.github/img/shader-toy.png" width="176" height="105">
[vr/](vr/src/main.rs) | <pre>cargo run --manifest-path examples/vr/Cargo.toml</pre> | <image alt="vr" src="../.github/img/vr.png" width="176" height="146">

## Additional Examples

The following packages offer examples for specific cases not listed here:

- [crates/vk-graph-hot](../crates/vk-graph-hot/examples/README.md): Shader pipeline hot-reload
- [attackgoat/mood](https://github.com/attackgoat/mood): FPS game prototype with level loading and
  multiple rendering backends
- [attackgoat/jw-basic](https://github.com/attackgoat/jw-basic): BASIC interpreter with graphics
  commands powered by _vk-graph_
