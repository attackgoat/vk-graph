use {
    super::XrInstance,
    openxr as xr,
    std::{
        ops::{Deref, DerefMut},
        sync::Arc,
    },
    vk_graph::driver::{
        ash::vk::{self, Handle as _},
        image::{Image, ImageInfo},
    },
};

pub struct Swapchain {
    images: Vec<Arc<Image>>,
    resolution: vk::Extent2D,
    swapchain: xr::Swapchain<xr::Vulkan>,
}

impl Swapchain {
    pub fn new(instance: &XrInstance, session: &xr::Session<xr::Vulkan>) -> Self {
        let device = XrInstance::device(instance);

        let views = XrInstance::enumerate_view_configuration_views(
            instance,
            xr::ViewConfigurationType::PRIMARY_STEREO,
        )
        .unwrap();
        assert_eq!(views.len(), 2);
        assert_eq!(views[0], views[1]);

        let resolution = vk::Extent2D {
            width: views[0].recommended_image_rect_width,
            height: views[0].recommended_image_rect_height,
        };
        let swapchain = session
            .create_swapchain(&xr::SwapchainCreateInfo {
                create_flags: xr::SwapchainCreateFlags::EMPTY,
                usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                    | xr::SwapchainUsageFlags::SAMPLED,
                format: vk::Format::R8G8B8A8_SRGB.as_raw() as _,
                sample_count: 1,
                width: resolution.width,
                height: resolution.height,
                face_count: 1,
                array_size: 2,
                mip_count: 1,
            })
            .unwrap();

        let images = swapchain.enumerate_images().unwrap();

        Self {
            images: images
                .into_iter()
                .map(|image| {
                    let image = vk::Image::from_raw(image);
                    let info = ImageInfo::image_2d_array(
                        resolution.width,
                        resolution.height,
                        2,
                        vk::Format::R8G8B8A8_SRGB,
                        vk::ImageUsageFlags::SAMPLED,
                    );

                    Arc::new(Image::from_raw(device, image, info))
                })
                .collect(),
            resolution,
            swapchain,
        }
    }

    pub fn image(this: &Self, index: usize) -> &Arc<Image> {
        &this.images[index]
    }

    pub fn images(this: &Self) -> &[Arc<Image>] {
        &this.images
    }

    pub fn resolution(this: &Self) -> vk::Extent2D {
        this.resolution
    }
}

impl Deref for Swapchain {
    type Target = xr::Swapchain<xr::Vulkan>;

    fn deref(&self) -> &Self::Target {
        &self.swapchain
    }
}

impl DerefMut for Swapchain {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.swapchain
    }
}
