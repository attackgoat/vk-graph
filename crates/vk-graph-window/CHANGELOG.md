# Changelog

All notable changes to this crate will be documented in this file.

This crate is versioned independently from `vk-graph`.

## [Unreleased]

## [0.2.0]

- Requires `vk-graph` `0.15.1` or later in the `0.15` series and Rust 1.95 or newer. The public
  API uses `vk-graph` types and is not compatible with `vk-graph` `0.14`.
- Fix a window-resize crash on NVIDIA 595.x Linux drivers by waiting for the presentation queue
  before recreating the swapchain ([#120](https://github.com/attackgoat/vk-graph/pull/120)).
- Use per-image semaphores in `MAILBOX` mode instead of waiting on the previous presentation
  for each frame slot. Reevaluate the semaphore strategy when the swapchain is recreated.
- Expose the selected queue family in `EffectiveGraphchainInfo` and add
  `Graphchain::wait_for_in_flight_limit` to cap overlapping submissions.

## [0.1.4] - 2026-07-14

- Fixed built-in profiling feature

## [0.1.3] - 2026-07-03

- Supports `vk-graph` `0.14.4`.
- `Graphchain::present_image` now returns `SubmissionQueryPool` with timestamp results for the
  submitted graph work.

## [0.1.2] - 2026-06-25

- Supports `vk-graph` `0.14.3`.
- Uses the renamed `Fence::wait` API.

## [0.1.1] - 2026-06-18

- Supports `vk-graph` `0.14.2`.

## [0.1.0] - 2026-05-30

- Initial release.
- Supports `vk-graph` `0.14`.
