use {
    super::{
        AccelerationStructureLeaseNode, AccelerationStructureNode, BufferLeaseNode, BufferNode,
        ImageLeaseNode, ImageNode, RenderGraph, Resolver, SwapchainImageNode,
        pass_ref::{PassRef, PipelinePassRef},
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

macro_rules! graph_edge {
    ($src:ty => $dst:ty) => {
        impl Edge<RenderGraph> for $src {
            type Result = $dst;
        }
    };
}

// Edges that can be bound as nodes to the render graph:
// Ex: RenderGraph::bind_node(&mut self, binding: X) -> Y
graph_edge!(AccelerationStructure => AccelerationStructureNode);
graph_edge!(Arc<AccelerationStructure> => AccelerationStructureNode);
graph_edge!(Lease<AccelerationStructure> => AccelerationStructureLeaseNode);
graph_edge!(Arc<Lease<AccelerationStructure>> => AccelerationStructureLeaseNode);
graph_edge!(Buffer => BufferNode);
graph_edge!(Arc<Buffer> => BufferNode);
graph_edge!(Lease<Buffer> => BufferLeaseNode);
graph_edge!(Arc<Lease<Buffer>> => BufferLeaseNode);
graph_edge!(Image => ImageNode);
graph_edge!(Arc<Image> => ImageNode);
graph_edge!(Lease<Image> => ImageLeaseNode);
graph_edge!(Arc<Lease<Image>> => ImageLeaseNode);
graph_edge!(SwapchainImage => SwapchainImageNode);

// Edges that can be unbound from the render graph:
// Ex: RenderGraph::unbind_node(&mut self, node: X) -> Y
graph_edge!(AccelerationStructureNode => Arc<AccelerationStructure>);
graph_edge!(AccelerationStructureLeaseNode => Arc<Lease<AccelerationStructure>>);
graph_edge!(BufferNode => Arc<Buffer>);
graph_edge!(BufferLeaseNode => Arc<Lease<Buffer>>);
graph_edge!(ImageNode => Arc<Image>);
graph_edge!(ImageLeaseNode => Arc<Lease<Image>>);
graph_edge!(SwapchainImageNode => SwapchainImage);

macro_rules! graph_edge_borrow {
    ($src:ty => $dst:ty) => {
        impl<'a> Edge<RenderGraph> for &'a $src {
            type Result = $dst;
        }
    };
}

graph_edge_borrow!(Arc<AccelerationStructure> => AccelerationStructureNode);
graph_edge_borrow!(Arc<Lease<AccelerationStructure>> => AccelerationStructureLeaseNode);
graph_edge_borrow!(Arc<Buffer> => BufferNode);
graph_edge_borrow!(Arc<Lease<Buffer>> => BufferLeaseNode);
graph_edge_borrow!(Arc<Image> => ImageNode);
graph_edge_borrow!(Arc<Lease<Image>> => ImageLeaseNode);

// Specialized edges for pipelines added to a pass:
// Ex: PassRef::bind_pipeline(&mut self, pipeline: X) -> PipelinePassRef
macro_rules! pipeline_edge {
    ($name:ident) => {
        paste::paste! {
            impl<'a> Edge<PassRef<'a>> for &'a Arc<[<$name Pipeline>]> {
                type Result = PipelinePassRef<'a, [<$name Pipeline>]>;
            }

            impl<'a> Edge<PassRef<'a>> for Arc<[<$name Pipeline>]> {
                type Result = PipelinePassRef<'a, [<$name Pipeline>]>;
            }

            impl<'a> Edge<PassRef<'a>> for [<$name Pipeline>] {
                type Result = PipelinePassRef<'a, [<$name Pipeline>]>;
            }
        }
    };
}

pipeline_edge!(Compute);
pipeline_edge!(Graphic);
pipeline_edge!(RayTrace);

macro_rules! resolver_edge {
    ($src:ident -> $dst:ident) => {
        impl Edge<Resolver> for $src {
            type Result = $dst;
        }
    };
}

// Edges that can be unbound from a resolved render graph:
// (You get the full real actual swapchain image woo hoo!)
resolver_edge!(SwapchainImageNode -> SwapchainImage);
