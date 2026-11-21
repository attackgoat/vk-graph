# Changes

## 0.6.0 (Unreleased)

* Add acceleration structure build input and scratch access types.
* Correct write synchronization for acceleration structure builds and serialization.

## 0.5.0 (2025-01-25)

* Update to `ash` `0.38.0`.

## 0.4.0 (2022-04-06)

* Update to `ash` `0.37.0`.

## 0.3.0 (2021-12-28)

* Update to `ash` `0.35.0`.

## 0.2.2  (2021-12-10)

* Adds destination access marks to image barriers in all cases to avoid syncronisation hazards.

## 0.2.1 (2021-10-21)

* Changes the crate description to make it clearer in the crates browser that this crate is a fork.

## 0.2.0 (2021-10-21)

* Forked off of the original crate.
* Updated to ash 0.33.
* Added several `AccessType`s for the `VK_KHR_ray_tracing_pipeline` and `VK_KHR_acceleration_structure` extensions.
## 0.1.6 (2019-07-14)

* Removed inefficient Vec<AccessType> on barrier structs in favor of slice references.

## 0.1.5

* Updated to ash 0.29.

## 0.1.4

* Minor optimizations.

## 0.1.3

* Rust 2018 Edition.

## 0.1.2 (2018-11-17)

* Updated to ash 0.26
* Use default struct init from ash
* Made function pointer structs borrowed for performance
* Some minor cleanup

## 0.1.1 (2018-11-15)

* Updated to ash 0.25 (Vulkan 1.1)
* Added support for NVX generated commands
* Added support for read-only depth/stencil + writeable depth/stencil
* Added Copy and Default traits to AccessType and ImageLayout
* Added Debug, Default, and Clone traits to GlobalBarrier, BufferBarrier, and ImageBarrier

## 0.1.0 (2018-08-26)

* First release
