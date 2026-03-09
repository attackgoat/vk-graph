# Resources

Buffers, images, and acceleration structures are the user-definable 

## Memory Allocation

`vk-graph` uses a single memory allocator (currently `gpu-allocator`).

The memory allocation strategy provides a large section of memory which is then sub-allocated for
the resources which use it. This may lead to fragmentation and memory exhaustion in some scenarios.

Individual buffers or images may use dedicated memory allocations 
