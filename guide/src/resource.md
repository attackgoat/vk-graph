# Resources

> [!CAUTION]
> All pipelines and resources (_buffers, images, and acceleration structures_) "bound" to any
> `Graph` must have been created by the same `Device`.

Owned resources are created from `Device` references. They may be bound directly to graphs.

A borrow of `Arc<T>` of any resource may be bound to a graph if the resource needs to be referenced
in future graphs.

Binding resources to a graph produces a "Node" handle which may be used in commands and shader
pipelines. `Graph::bind_resource<T, N>(resource: T) -> N`:

`T`|`N`
-|-
`Buffer`|`BufferNode`
`Arc<Buffer>`|`BufferNode`
`Lease<Buffer>`|`BufferLeaseNode`
`Arc<Lease<Buffer>>`|`BufferLeaseNode`

_(etc...)_

Resources may be borrowed from a graph. `Graph::resource<N, T>(node: N) -> &T`:

`N`|`T`
-|-
`BufferNode`|`Arc<Buffer>`
`BufferLeaseNode`|`Arc<Lease<Buffer>>`

## Bound Resource Nodes

The concept of binding resources to graphs as node handles exists to support the callback-style
command buffer recording provided by `vk-graph`.

Commands are recorded in logical order, but the execution is re-ordered for performance and so a
closure argument is provided to call Vulkan command buffer functions. The use of a small and `Copy`
node handle allows resource handles to be moved into command buffer closures without `Arc::clone`.

Additionally, node handles support internal optimizations by providing direct indexed access to
graph data structures.

## Pooling Resources

Pooled resources are leased from `Pool` implementations. Dropped leases return to the pool.

The `Lease<T>` type otherwise acts identically to an owned resource.

## Aliased Resources

Resource aliasing is available using the `AliasWrapper` and any `Pool`.

Aliased resources allow extremely optimized programs to ensure minimal resources during complex
graphs.
