# Threading Behavior

`vk-graph` is intended to provide scalable performance when used on multiple host threads.
Resources are externally synchronized, and mutable graph-building APIs such as `Graph::begin_cmd`
require exclusive access to the `Graph` itself.

API docs: [`Submission`](https://docs.rs/vk-graph/latest/vk_graph/struct.Submission.html),
[`RecordedSubmission`](https://docs.rs/vk-graph/latest/vk_graph/struct.RecordedSubmission.html),
[`Submission::queue_submit`](https://docs.rs/vk-graph/latest/vk_graph/struct.Submission.html#method.queue_submit),
[`Submission::record_resource`](https://docs.rs/vk-graph/latest/vk_graph/struct.Submission.html#method.record_resource),
[`Submission::record_resource_dependencies`](https://docs.rs/vk-graph/latest/vk_graph/struct.Submission.html#method.record_resource_dependencies),
[`RecordedSubmission::queue_submit`](https://docs.rs/vk-graph/latest/vk_graph/struct.RecordedSubmission.html#method.queue_submit),
[`CommandBuffer::has_executed`](https://docs.rs/vk-graph/latest/vk_graph/driver/cmd_buf/struct.CommandBuffer.html#method.has_executed).

More precisely, `vk-graph` stores the most recent access type of each subresource of a resource. As
commands are submitted to the Vulkan implementation queue, the internal state of these resources is
updated.

Resource state is updated during the following function calls:

- `Submission::queue_submit`
- `Submission::record_resource`
- `Submission::record_resource_dependencies`
- `RecordedSubmission::queue_submit`

> [!CAUTION]
> Do not call any `Submission` recording or queue function that accesses buffers, images, or acceleration
> structures currently being submitted on other threads.

## Execution

The provided `Submission` recording and queue functions are designed to support a typical
swapchain-based
workflow:
1. Queue all commands the swapchain depends on
1. Acquire swapchain
1. Queue swapchain commands
1. Present swapchain
1. Submit any final unrelated commands

## Safe Patterns

Resources (buffers, images, or acceleration structures) are the only mutable types which require any
thread safety notes. All other types provided by `vk-graph` are immutable data structures or Vulkan
handle smart pointers.

For example, there is no race condition or thread contention caused by using the same pipeline on
two threads.[^threads] In fact, there is no runtime overhead at all from this.

Additionally, it is safe to build `Graph` instances, bind resources, record command buffers, and
call `Graph::finalize` at *any* time on *any* thread, as long as each `Graph` instance is not
mutably shared across threads at the same time.

These patterns are safe:
- Build `Graph` and `Send` to another thread for submission
- Build `Graph` and `Drop` it without submission
- `Send` resources to other threads _or_ share as `Arc<T>`
- `Clone` devices or pipelines and `Send` them to other threads

## Risky Patterns

Host-mappable buffers require extra understanding to use properly.

The contents of a buffer are undefined from the time of submission until that `Submission` has been
fully executed, as indicated by `CommandBuffer::has_executed`. This means that you should not call
`Buffer::mapped_slice` during any submission or execution accessing that memory.

See:
[_`examples/cpu_readback.rs`_](https://github.com/attackgoat/vk-graph/blob/main/examples/cpu_readback.rs)
<i class="fa-solid fa-arrow-up-right-from-square"></i>


[^threads]: The internal implementation of `GraphicsPipeline` does do a bit of caching in order to
improve performance, however this behavior should not generate issues with any reasonable workload.
