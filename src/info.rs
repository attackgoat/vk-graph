use {
    super::{
        AccelerationStructureLeaseNode, AccelerationStructureNode, Binding, BufferLeaseNode,
        BufferNode, ImageLeaseNode, ImageNode, SwapchainImageNode,
    },
    crate::driver::{
        accel_struct::AccelerationStructureInfo, buffer::BufferInfo, image::ImageInfo,
    },
};

pub trait Info {
    type Info;

    fn info(self, bindings: &[Binding]) -> Self::Info;
}

macro_rules! info {
    ($name:ident: $src:ident -> $dst:ident) => {
        paste::paste! {
            impl Info for $src {
                type Info = $dst;

                fn info(self, bindings: &[Binding]) -> $dst {
                    bindings[self.idx].[<as_ $name>]().unwrap().info
                }
            }
        }
    };
}

info!(acceleration_structure: AccelerationStructureNode -> AccelerationStructureInfo);
info!(acceleration_structure_lease: AccelerationStructureLeaseNode -> AccelerationStructureInfo);
info!(buffer: BufferNode -> BufferInfo);
info!(buffer_lease: BufferLeaseNode -> BufferInfo);
info!(image: ImageNode -> ImageInfo);
info!(image_lease: ImageLeaseNode -> ImageInfo);
info!(swapchain_image: SwapchainImageNode -> ImageInfo);
