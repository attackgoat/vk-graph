use {
    super::{
        AccelerationStructureLeaseNode, AccelerationStructureNode, BufferLeaseNode, BufferNode,
        ImageLeaseNode, ImageNode, RenderGraph, SwapchainImageNode,
    },
    crate::driver::{
        accel_struct::AccelerationStructureInfo, buffer::BufferInfo, image::ImageInfo,
    },
};

pub trait Information {
    type Info;

    fn get(self, graph: &RenderGraph) -> Self::Info;
}

macro_rules! information {
    ($name:ident: $src:ident -> $dst:ident) => {
        paste::paste! {
            impl Information for $src {
                type Info = $dst;

                fn get(self, graph: &RenderGraph) -> $dst {
                    graph.bindings[self.idx].[<as_ $name>]().unwrap().info
                }
            }
        }
    };
}

information!(acceleration_structure: AccelerationStructureNode -> AccelerationStructureInfo);
information!(acceleration_structure_lease: AccelerationStructureLeaseNode -> AccelerationStructureInfo);
information!(buffer: BufferNode -> BufferInfo);
information!(buffer_lease: BufferLeaseNode -> BufferInfo);
information!(image: ImageNode -> ImageInfo);
information!(image_lease: ImageLeaseNode -> ImageInfo);
information!(swapchain_image: SwapchainImageNode -> ImageInfo);
