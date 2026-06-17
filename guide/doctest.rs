// HACK: Test tooling for mdBook is lacking so this fix is applied:
// https://github.com/BurntSushi/jiff/blob/985b4156c4dbaf2ed69150a4ec4e6d5352f1d47e/book/doctest.rs

#[doc = include_str!("src/cmd.md")]
pub mod cmd {}

#[doc = include_str!("src/cmd_compute.md")]
pub mod cmd_compute {}

#[doc = include_str!("src/pipeline.md")]
pub mod pipeline {}

#[doc = include_str!("src/pipeline_hot_reload.md")]
pub mod pipeline_hot_reload {}

#[doc = include_str!("src/pipeline_push_const.md")]
pub mod pipeline_push_const {}

#[doc = include_str!("src/pipeline_sync.md")]
pub mod pipeline_sync {}

#[doc = include_str!("src/pipeline_spec.md")]
pub mod pipeline_spec {}

#[doc = include_str!("src/resource_accel_struct.md")]
pub mod resource_accel_struct {}

#[doc = include_str!("src/resource_buffer.md")]
pub mod resource_buffer {}

#[doc = include_str!("src/resource_image.md")]
pub mod resource_image {}

#[doc = include_str!("src/usage.md")]
pub mod usage {}

#[doc = include_str!("src/usage_device.md")]
pub mod usage_device {}

#[doc = include_str!("src/usage_shader.md")]
pub mod usage_shader {}

#[doc = include_str!("src/usage_window.md")]
pub mod usage_window {}
