use {
    super::{
        AccelerationStructureLeaseNode, AccelerationStructureNode, AnyAccelerationStructureNode,
        AnyBufferNode, AnyImageNode, BufferLeaseNode, BufferNode, BufferSubresourceRange,
        ImageLeaseNode, ImageNode, ImageViewInfo, Node, Subresource, SwapchainImageNode, vk,
    },
    std::ops::Range,
};

/// Allows for a resource to be reinterpreted as differently formatted data.
pub trait View: Node
where
    Self::Info: Copy,
    Self::Subresource: Into<Subresource>,
{
    /// The Info about the resource interpretation.
    type Info;

    /// The portion of the resource which is bound.
    type Subresource;
}

impl View for AccelerationStructureNode {
    type Info = ();
    type Subresource = ();
}

impl View for AccelerationStructureLeaseNode {
    type Info = ();
    type Subresource = ();
}

impl View for AnyAccelerationStructureNode {
    type Info = ();
    type Subresource = ();
}

impl View for AnyBufferNode {
    type Info = BufferSubresourceRange;
    type Subresource = BufferSubresourceRange;
}

impl View for AnyImageNode {
    type Info = ImageViewInfo;
    type Subresource = vk::ImageSubresourceRange;
}

impl View for BufferLeaseNode {
    type Info = BufferSubresourceRange;
    type Subresource = BufferSubresourceRange;
}

impl View for BufferNode {
    type Info = BufferSubresourceRange;
    type Subresource = BufferSubresourceRange;
}

impl View for ImageLeaseNode {
    type Info = ImageViewInfo;
    type Subresource = vk::ImageSubresourceRange;
}

impl View for ImageNode {
    type Info = ImageViewInfo;
    type Subresource = vk::ImageSubresourceRange;
}

impl View for SwapchainImageNode {
    type Info = ImageViewInfo;
    type Subresource = vk::ImageSubresourceRange;
}

/// Describes the interpretation of a resource.
#[derive(Debug)]
pub enum ViewType {
    /// Acceleration structures are not reinterpreted.
    AccelerationStructure,

    /// Images may be interpreted as differently formatted images.
    Image(ImageViewInfo),

    /// Buffers may be interpreted as subregions of the same buffer.
    Buffer(Range<vk::DeviceSize>),
}

impl ViewType {
    pub(crate) fn as_buffer(&self) -> Option<&Range<vk::DeviceSize>> {
        match self {
            Self::Buffer(view_info) => Some(view_info),
            _ => None,
        }
    }

    pub(crate) fn as_image(&self) -> Option<&ImageViewInfo> {
        match self {
            Self::Image(view_info) => Some(view_info),
            _ => None,
        }
    }
}

impl From<()> for ViewType {
    fn from(_: ()) -> Self {
        Self::AccelerationStructure
    }
}

impl From<BufferSubresourceRange> for ViewType {
    fn from(subresource: BufferSubresourceRange) -> Self {
        Self::Buffer(subresource.start..subresource.end)
    }
}

impl From<ImageViewInfo> for ViewType {
    fn from(info: ImageViewInfo) -> Self {
        Self::Image(info)
    }
}

impl From<Range<vk::DeviceSize>> for ViewType {
    fn from(range: Range<vk::DeviceSize>) -> Self {
        Self::Buffer(range)
    }
}
