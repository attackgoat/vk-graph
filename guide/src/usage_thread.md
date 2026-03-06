# Threading Behavior

`vk-graph` is intended to provide scalable performance when used on multiple host threads. All
commands support being called concurrently from multiple threads, but resources are defined to be
externally synchronized. This means that the caller must guarantee that no more than one thread is
submitting a resource at a given time.

More precisely, `vk-graph` stores the most recent access of each subresource of a resource. As
commands are submitted to the Vulkan implementation queue, the internal state of these resources is
updated.

Resource state is updated during the following function calls:

- `Queue::submit`
- `Queue::submit_resource`
- `Queue::submit_resource_dependencies`

> [!CAUTION]
> Do not call any `Queue` submission function accessing buffers, images, or acceleration structures
> currently being submitted on other threads.

## Safe Patterns

Resources (buffers, images, or acceleration structures) are the only mutable types which require any
thread safety notes. All other types provided by `vk-graph` are immutable data structures or Vulkan
handle smart pointers.

For example, there is no race condition or thread contention caused by using the same pipeline on
two threads.[^threads] In fact, there is no runtime overhead at all from this.

Additionally, it is safe to build `Graph` instances, bind resources, record command buffers, and
call `Graph::into_queue` at *any* time on *any* thread.

These patterns are safe:
- Build `Graph` and `Send` to another thread for submission
- Build `Graph` and `Drop` it without submission
- `Send` resources to other threads _or_ share as `Arc<T>`
- `Clone` device or pipelines and `Send` to other threads

## Risky Patterns

Host-mappable buffers require extra understanding to use properly.

The contents of a buffer are undefined from the time of submission until that `Queue` has been
fully submitted, as indicated by `Queue::is_submitted`. This means that you should not call
`Buffer::mapped_slice` during any execution accessing that memory.

_See:
[`examples/cpu_readback.rs`](https://github.com/attackgoat/vk-graph/blob/main/examples/cpu_readback.rs)_


[^threads]: The internal implementation of `GraphicPipeline` does do a bit of caching in order to
improve performance, however this behavior should never generate issues with any fathomable
workload.
