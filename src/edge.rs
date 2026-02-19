use {
    super::{
        AccelerationStructureLeaseNode, AccelerationStructureNode, BufferLeaseNode, BufferNode,
        Graph, ImageLeaseNode, ImageNode, Resolver, SwapchainImageNode,
        cmd_ref::{CommandRef, PipelineCommandRef},
    },
    crate::{
        driver::{
            accel_struct::AccelerationStructure, buffer::Buffer, compute::ComputePipeline,
            graphic::GraphicPipeline, image::Image, ray_trace::RayTracePipeline,
            swapchain::SwapchainImage,
        },
        pool::Lease,
    },
    std::sync::Arc,
};

/// A marker trait that says some graph object can transition into a different
/// graph object; it is a one-way transition unless the other direction has
/// been implemented too.
pub trait Edge<Graph> {
    type Result;
}

macro_rules! node {
    ($src:ty => $dst:ty) => {
        impl Edge<Graph> for $src {
            type Result = $dst;
        }
    };
}

// Edges that can be bound as nodes to the render graph:
// Ex: Graph::bind_node(&mut self, binding: X) -> Y
node!(AccelerationStructure => AccelerationStructureNode);
node!(Arc<AccelerationStructure> => AccelerationStructureNode);
node!(Lease<AccelerationStructure> => AccelerationStructureLeaseNode);
node!(Arc<Lease<AccelerationStructure>> => AccelerationStructureLeaseNode);
node!(Buffer => BufferNode);
node!(Arc<Buffer> => BufferNode);
node!(Lease<Buffer> => BufferLeaseNode);
node!(Arc<Lease<Buffer>> => BufferLeaseNode);
node!(Image => ImageNode);
node!(Arc<Image> => ImageNode);
node!(Lease<Image> => ImageLeaseNode);
node!(Arc<Lease<Image>> => ImageLeaseNode);
node!(SwapchainImage => SwapchainImageNode);

// Edges that can be unbound from the render graph:
// Ex: Graph::unbind_node(&mut self, node: X) -> Y
node!(AccelerationStructureNode => Arc<AccelerationStructure>);
node!(AccelerationStructureLeaseNode => Arc<Lease<AccelerationStructure>>);
node!(BufferNode => Arc<Buffer>);
node!(BufferLeaseNode => Arc<Lease<Buffer>>);
node!(ImageNode => Arc<Image>);
node!(ImageLeaseNode => Arc<Lease<Image>>);
node!(SwapchainImageNode => SwapchainImage);

macro_rules! node_ref {
    ($src:ty => $dst:ty) => {
        impl<'a> Edge<Graph> for &'a $src {
            type Result = $dst;
        }
    };
}

node_ref!(Arc<AccelerationStructure> => AccelerationStructureNode);
node_ref!(Arc<Lease<AccelerationStructure>> => AccelerationStructureLeaseNode);
node_ref!(Arc<Buffer> => BufferNode);
node_ref!(Arc<Lease<Buffer>> => BufferLeaseNode);
node_ref!(Arc<Image> => ImageNode);
node_ref!(Arc<Lease<Image>> => ImageLeaseNode);

// Specialized edges for pipelines added to a pass:
// Ex: PassRef::bind_pipeline(&mut self, pipeline: X) -> PipelineCommandRef
macro_rules! pipeline {
    ($name:ident) => {
        paste::paste! {
            impl<'a> Edge<CommandRef<'a>> for &'a Arc<[<$name Pipeline>]> {
                type Result = PipelineCommandRef<'a, [<$name Pipeline>]>;
            }

            impl<'a> Edge<CommandRef<'a>> for Arc<[<$name Pipeline>]> {
                type Result = PipelineCommandRef<'a, [<$name Pipeline>]>;
            }

            impl<'a> Edge<CommandRef<'a>> for [<$name Pipeline>] {
                type Result = PipelineCommandRef<'a, [<$name Pipeline>]>;
            }
        }
    };
}

pipeline!(Compute);
pipeline!(Graphic);
pipeline!(RayTrace);

macro_rules! resolve {
    ($src:ident -> $dst:ident) => {
        impl Edge<Resolver> for $src {
            type Result = $dst;
        }
    };
}

// Edges that can be unbound from a resolved graph:
// (You get the full real actual swapchain image woo hoo!)
resolve!(SwapchainImageNode -> SwapchainImage);
