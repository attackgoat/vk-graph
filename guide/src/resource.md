# Resources

API docs: [`Graph::bind_resource`](https://docs.rs/vk-graph/latest/vk_graph/struct.Graph.html#method.bind_resource),
[`Graph::resource`](https://docs.rs/vk-graph/latest/vk_graph/struct.Graph.html#method.resource),
[`Node`](https://docs.rs/vk-graph/latest/vk_graph/node/trait.Node.html),
[`Pool`](https://docs.rs/vk-graph/latest/vk_graph/pool/trait.Pool.html),
[`Cache`](https://docs.rs/vk-graph/latest/vk_graph/pool/cache/struct.Cache.html).

> [!CAUTION]
> All pipelines and resources (_buffers, images, and acceleration structures_) used in a `Graph`
> must have been created using the same `Device`.

Owned resources are created from `Device` references. They may be bound directly to graphs.

An `Arc<T>` or `&Arc<T>` of any resource may be bound to a graph if the resource needs to be
referenced in future graphs.

## Binding

Binding resources to a graph produces a "Node" handle which may be used in commands and shader
pipelines.

Example for buffers using `Graph::bind_resource<R, N>(&mut self, resource: R) -> N`:

`R`|`N`
-|-
`Buffer`|`BufferNode`
`Arc<Buffer>`|`BufferNode`
`Lease<Buffer>`|`BufferLeaseNode`
`Arc<Lease<Buffer>>`|`BufferLeaseNode`

## Borrowing

Resources may be borrowed from a graph.

Example for buffers using `Graph::resource<N, R>(&self, node: N) -> &R`:

`N`|`R`
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

Resource aliasing is available using [`Cache`](https://docs.rs/vk-graph/latest/vk_graph/pool/cache/struct.Cache.html)
and any `Pool`.

Aliased resources allow extremely optimized programs to ensure minimal resources during complex
graphs.
