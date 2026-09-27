//! Library-test-only harness and Vulkan test cases.
//!
//! Compiled through `src/lib.rs` under `cfg(test)` so disposal hooks stay crate-private and
//! production builds contain neither the harness nor disposal observation state.

mod buffer_ownership;
mod device;
pub(crate) mod disposal;
mod ownership;
mod pipeline_statistics;
mod smoke;
pub(crate) mod validation;
mod vulkan;

pub(crate) use device::{DeviceChecks, TestDevice};
