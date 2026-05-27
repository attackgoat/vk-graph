//! Native window presentation types.

use {
    super::{
        DriverError, Surface,
        device::Device,
        image::{Image, ImageInfo},
    },
    ash::vk,
    derive_builder::{Builder, UninitializedFieldError},
    log::{debug, info, trace, warn},
    std::{mem::replace, ops::Deref, slice, thread::panicking},
};

// TODO: This needs to track completed command buffers and not constantly create semaphores

/// Provides the ability to present rendering results to a [`Surface`].
#[derive(Debug)]
#[read_only::cast]
pub struct Swapchain {
    /// The native Vulkan resource handle of this swapchain.
    ///
    /// _Note:_ This field is read-only.
    pub handle: vk::SwapchainKHR,

    handle_prev: vk::SwapchainKHR,
    images: Box<[SwapchainImage]>,

    /// Information used to create this resource.
    ///
    /// _Note:_ This field is read-only.
    pub info: SwapchainInfo,

    suboptimal: bool,

    /// The surface which supports this swapchain.
    ///
    /// _Note:_ This field is read-only.
    pub surface: Surface,
}

impl Swapchain {
    /// Prepares a [`vk::SwapchainKHR`] object which is lazily created after calling
    /// [`acquire_next_image`][Self::acquire_next_image].
    #[profiling::function]
    pub fn new(surface: Surface, info: impl Into<SwapchainInfo>) -> Result<Self, DriverError> {
        let info = info.into();

        Ok(Swapchain {
            images: Default::default(),
            handle: vk::SwapchainKHR::null(),
            handle_prev: vk::SwapchainKHR::null(),
            info,
            suboptimal: true,
            surface,
        })
    }

    /// Acquires the next available swapchain image for rendering.
    ///
    /// The returned [`SwapchainImage`] should be rendered to and then presented using
    /// [`present_image`][Self::present_image]. Each call returns a unique image from the
    /// swapchain's image ring buffer; the caller must not render to the same image concurrently.
    ///
    /// # Parameters
    ///
    /// * `timeout` — Maximum time to wait in nanoseconds before the operation times out.
    ///   Pass [`u64::MAX`] to wait indefinitely. A short timeout may return
    ///   [`SwapchainError::Suboptimal`] without acquiring an image.
    /// * `acquired` — A semaphore that will be signaled when the acquired image is ready
    ///   for use. The caller must wait on this semaphore before submitting commands that
    ///   write to the returned image.
    ///
    /// # Errors
    ///
    /// Returns [`SwapchainError::Suboptimal`] if the swapchain no longer matches the
    /// surface properties (e.g. after a window resize). The caller should typically
    /// recreate any framebuffer-sized resources and try again next frame.
    ///
    /// Returns [`SwapchainError::SurfaceLost`] if the underlying surface has been
    /// destroyed. The swapchain and surface must be recreated.
    ///
    /// Returns [`SwapchainError::DeviceLost`] if the Vulkan device has been lost.
    /// The application must destroy and recreate the device.
    ///
    /// # Panics
    ///
    /// Panics if the acquired image index is out of bounds of the internal image array
    /// (this indicates a driver or swapchain consistency bug).
    ///
    /// # Retry behavior
    ///
    /// Internally this method retries once on transient errors
    /// (`TIMEOUT`, `NOT_READY`, `OUT_OF_DATE_KHR`, `FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT`).
    /// Between retries the swapchain is recreated if the current state is suboptimal.
    #[profiling::function]
    pub fn acquire_next_image(
        &mut self,
        timeout: u64,
        acquired: vk::Semaphore,
    ) -> Result<SwapchainImage, SwapchainError> {
        for _ in 0..2 {
            if self.suboptimal {
                self.recreate().map_err(|err| {
                    if matches!(err, DriverError::Unsupported) {
                        SwapchainError::Suboptimal
                    } else {
                        SwapchainError::SurfaceLost
                    }
                })?;
            }

            let swapchain_ext = Device::expect_swapchain_ext(&self.surface.device);

            let image_idx = unsafe {
                swapchain_ext.acquire_next_image(self.handle, timeout, acquired, vk::Fence::null())
            }
            .map(|(idx, suboptimal)| {
                if suboptimal {
                    debug!("acquired image is suboptimal");
                }

                self.suboptimal = suboptimal;

                idx
            });

            match image_idx {
                Ok(image_idx) => {
                    let image_idx = image_idx as usize;

                    assert!(image_idx < self.images.len());

                    let image = unsafe { self.images.get_unchecked(image_idx) }.clone();

                    return Ok(replace(
                        unsafe { self.images.get_unchecked_mut(image_idx) },
                        image,
                    ));
                }
                Err(err)
                    if err == vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT
                        || err == vk::Result::ERROR_OUT_OF_DATE_KHR
                        || err == vk::Result::NOT_READY
                        || err == vk::Result::TIMEOUT =>
                {
                    warn!("unable to acquire image: {err}");

                    self.suboptimal = true;

                    // Try again to see if we can acquire an image this frame
                    // (Makes redraw during resize look slightly better)
                    continue;
                }
                Err(err) if err == vk::Result::ERROR_DEVICE_LOST => {
                    warn!("unable to acquire image: {err}");

                    self.suboptimal = true;

                    return Err(SwapchainError::DeviceLost);
                }
                Err(err) if err == vk::Result::ERROR_SURFACE_LOST_KHR => {
                    warn!("unable to acquire image: {err}");

                    self.suboptimal = true;

                    return Err(SwapchainError::SurfaceLost);
                }
                Err(err) => {
                    // Probably:
                    // VK_ERROR_OUT_OF_HOST_MEMORY
                    // VK_ERROR_OUT_OF_DEVICE_MEMORY

                    // TODO: Maybe handle timeout in here

                    warn!("unable to acquire image: {err}");

                    return Err(SwapchainError::SurfaceLost);
                }
            }
        }

        Err(SwapchainError::Suboptimal)
    }

    fn clamp_min_image_count(min_image_count: u32, surface: vk::SurfaceCapabilitiesKHR) -> u32 {
        let min_image_count = min_image_count.max(surface.min_image_count);

        if surface.max_image_count == 0 {
            return min_image_count;
        }

        min_image_count.min(surface.max_image_count)
    }

    #[profiling::function]
    fn destroy(device: &Device, swapchain: &mut vk::SwapchainKHR) {
        if *swapchain == vk::SwapchainKHR::null() {
            return;
        }

        // wait for device to be finished with swapchain before destroying it.
        // This avoid crashes when resizing windows
        #[cfg(target_os = "macos")]
        if let Err(err) = unsafe { device.device_wait_idle() } {
            warn!("device_wait_idle() failed: {err}");
        }

        let swapchain_ext = Device::expect_swapchain_ext(device);

        unsafe {
            swapchain_ext.destroy_swapchain(*swapchain, None);
        }

        *swapchain = vk::SwapchainKHR::null();
    }

    /// Presents an image which has been previously acquired using
    /// [`acquire_next_image`][Self::acquire_next_image].
    #[profiling::function]
    pub fn present_image(
        &mut self,
        image: SwapchainImage,
        wait_semaphores: &[vk::Semaphore],
        queue_family_index: u32,
        queue_index: u32,
    ) {
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(slice::from_ref(&self.handle))
            .image_indices(slice::from_ref(&image.index));

        let swapchain_ext = Device::expect_swapchain_ext(&self.surface.device);

        unsafe {
            match swapchain_ext.queue_present(
                Device::queue(&self.surface.device, queue_family_index, queue_index),
                &present_info,
            ) {
                Ok(_) => {
                    Self::destroy(&self.surface.device, &mut self.handle_prev);
                }
                Err(err)
                    if err == vk::Result::ERROR_DEVICE_LOST
                        || err == vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT
                        || err == vk::Result::ERROR_OUT_OF_DATE_KHR
                        || err == vk::Result::ERROR_SURFACE_LOST_KHR
                        || err == vk::Result::SUBOPTIMAL_KHR =>
                {
                    // Handled in the next frame
                    self.suboptimal = true;
                }
                Err(err) => {
                    // Probably:
                    // VK_ERROR_OUT_OF_HOST_MEMORY
                    // VK_ERROR_OUT_OF_DEVICE_MEMORY
                    warn!("unable to destroy retired swapchain resources cleanly: {err}");
                }
            }
        }

        let image_idx = image.index as usize;
        self.images[image_idx] = image;
    }

    #[profiling::function]
    fn recreate(&mut self) -> Result<(), DriverError> {
        Self::destroy(&self.surface.device, &mut self.handle_prev);

        let surface_caps = Surface::capabilities(&self.surface)?;

        let min_image_count = Self::clamp_min_image_count(self.info.min_image_count, surface_caps);

        let image_usage = self.supported_surface_usage(surface_caps.supported_usage_flags)?;

        let (surface_width, surface_height) = match surface_caps.current_extent.width {
            std::u32::MAX => (
                // TODO: Maybe handle this case with aspect-correct clamping?
                self.info.width.clamp(
                    surface_caps.min_image_extent.width,
                    surface_caps.max_image_extent.width,
                ),
                self.info.height.clamp(
                    surface_caps.min_image_extent.height,
                    surface_caps.max_image_extent.height,
                ),
            ),
            _ => (
                surface_caps.current_extent.width,
                surface_caps.current_extent.height,
            ),
        };

        if surface_width * surface_height == 0 {
            warn!(
                "invalid surface extent: computed {}x{}",
                surface_width, surface_height
            );

            return Err(DriverError::Unsupported);
        }

        let pre_transform = if surface_caps
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_caps.current_transform
        };

        let swapchain_ext = Device::expect_swapchain_ext(&self.surface.device);
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface.handle)
            .min_image_count(min_image_count)
            .image_color_space(self.info.surface.color_space)
            .image_format(self.info.surface.format)
            .image_extent(vk::Extent2D {
                width: surface_width,
                height: surface_height,
            })
            .image_usage(image_usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.info.present_mode)
            .clipped(true)
            .old_swapchain(self.handle)
            .image_array_layers(1);
        let swapchain = unsafe { swapchain_ext.create_swapchain(&swapchain_create_info, None) }
            .map_err(|err| {
                warn!("unable to create swapchain: {err}");

                DriverError::Unsupported
            })?;

        let images =
            unsafe { swapchain_ext.get_swapchain_images(swapchain) }.map_err(|err| match err {
                vk::Result::INCOMPLETE => {
                    warn!("invalid swapchain image enumeration: incomplete");

                    DriverError::InvalidData
                }
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                    warn!("unable to get swapchain images: {err}");

                    DriverError::OutOfMemory
                }
                _ => {
                    warn!("unable to get swapchain images: {err}");

                    DriverError::Unsupported
                }
            })?;
        let images = images
            .into_iter()
            .enumerate()
            .map(|(image_idx, image)| {
                let mut image = Image::from_raw(
                    &self.surface.device,
                    image,
                    ImageInfo::image_2d(
                        surface_width,
                        surface_height,
                        self.info.surface.format,
                        image_usage,
                    ),
                );

                let image_idx = image_idx as u32;
                image.name = Some(format!("swapchain{image_idx}"));

                Ok(SwapchainImage {
                    read_only: ReadOnlySwapchainImage {
                        image,
                        index: image_idx,
                    },
                })
            })
            .collect::<Result<Box<_>, _>>()?;

        self.info.height = surface_height;
        self.info.width = surface_width;
        self.images = images;
        self.handle_prev = self.handle;
        self.handle = swapchain;
        self.suboptimal = false;

        info!(
            "swapchain {}x{} {:?}x{} {:?} {image_usage:#?}",
            self.info.width,
            self.info.height,
            self.info.present_mode,
            self.images.len(),
            self.info.surface.format,
        );

        Ok(())
    }

    /// Updates the information which controls this swapchain.
    ///
    /// Previously acquired swapchain images should be discarded after calling this function.
    pub fn set_info(&mut self, info: impl Into<SwapchainInfo>) {
        let info: SwapchainInfo = info.into();

        if self.info != info {
            // attempt to reducing flickering when resizing windows on mac
            #[cfg(target_os = "macos")]
            if let Err(err) = unsafe { self.surface.device.device_wait_idle() } {
                warn!("device_wait_idle() failed: {err}");
            }

            self.info = info;

            trace!("info: {:?}", self.info);

            self.suboptimal = true;
        }
    }

    fn supported_surface_usage(
        &mut self,
        surface_capabilities: vk::ImageUsageFlags,
    ) -> Result<vk::ImageUsageFlags, DriverError> {
        let mut res = vk::ImageUsageFlags::empty();

        for bit in 0..u32::BITS {
            let usage = vk::ImageUsageFlags::from_raw((1 << bit) & surface_capabilities.as_raw());
            if usage.is_empty() {
                continue;
            }

            if self
                .surface
                .device
                .physical_device
                .image_format_properties(
                    self.info.surface.format,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    usage,
                    vk::ImageCreateFlags::empty(),
                )
                .inspect_err(|err| {
                    warn!(
                        "unable to get image format properties: {:?} {:?} {err}",
                        self.info.surface.format, usage
                    )
                })?
                .is_none()
            {
                continue;
            }

            res |= usage;
        }

        // On mesa the device will return this usage flag as supported even when the extension
        // that is needed for an image to have this flag isn't enabled
        res &= !vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;

        Ok(res)
    }
}

impl Drop for Swapchain {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        Self::destroy(&self.surface.device, &mut self.handle_prev);
        Self::destroy(&self.surface.device, &mut self.handle);
    }
}

impl Eq for Swapchain {}

impl PartialEq for Swapchain {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

/// Describes the condition of a swapchain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SwapchainError {
    /// This frame is lost but more may be acquired later.
    DeviceLost,

    /// This frame is not lost but there may be a delay while the next frame is recreated.
    Suboptimal,

    /// The surface was lost and must be recreated, which includes any operating system window.
    SurfaceLost,
}

/// An opaque type representing a swapchain image.
#[derive(Debug)]
#[read_only::embed]
pub struct SwapchainImage {
    /// The underlying image resource.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    image: Image,

    /// The swapchain image index.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub index: u32,
}

impl Clone for SwapchainImage {
    fn clone(&self) -> Self {
        Self {
            read_only: ReadOnlySwapchainImage {
                image: self.image.clone_swapchain(),
                index: self.index,
            },
        }
    }
}

impl Deref for ReadOnlySwapchainImage {
    type Target = Image;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

/// Information used to create a [`Swapchain`] instance.
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build", error = "SwapchainInfoBuilderError"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct SwapchainInfo {
    /// The initial height of the surface.
    pub height: u32,

    /// The minimum number of presentable images that the application needs. The implementation will
    /// either create the swapchain with at least that many images, or it will fail to create the
    /// swapchain.
    ///
    /// More images introduce more display lag, but smoother animation.
    #[builder(default = "2")]
    pub min_image_count: u32,

    /// `vk::PresentModeKHR` determines timing and queueing with which frames are actually displayed
    /// to the user.
    ///
    /// `vk::PresentModeKHR::FIFO` - Presentation frames are kept in a First-In-First-Out queue approximately 3 frames
    /// long. Every vertical blanking period, the presentation engine will pop a frame off the queue to display. If
    /// there is no frame to display, it will present the same frame again until the next vblank.
    ///
    /// When a present command is executed on the GPU, the presented image is added on the queue.
    ///
    /// * **Tearing:** No tearing will be observed.
    /// * **Also known as**: "Vsync On"
    ///
    /// `vk::PresentModeKHR::FIFO_RELAXED` - Presentation frames are kept in a First-In-First-Out queue approximately 3
    /// frames long. Every vertical blanking period, the presentation engine will pop a frame off the queue to display.
    /// If there is no frame to display, it will present the same frame until there is a frame in the queue. The moment
    /// there is a frame in the queue, it will immediately pop the frame off the queue.
    ///
    /// When a present command is executed on the GPU, the presented image is added on the queue.
    ///
    /// * **Tearing**:
    ///   Tearing will be observed if frames last more than one vblank as the front buffer.
    /// * **Also known as**: "Adaptive Vsync"
    ///
    /// `vk::PresentModeKHR::IMMEDIATE` - Presentation frames are not queued at all. The moment a present command is
    /// executed on the GPU, the presented image is swapped onto the front buffer immediately.
    ///
    /// * **Tearing**: Tearing can be observed.
    /// * **Also known as**: "Vsync Off"
    ///
    /// `vk::PresentModeKHR::MAILBOX` - Presentation frames are kept in a single-frame queue. Every vertical blanking
    /// period, the presentation engine will pop a frame from the queue. If there is no frame to display, it will
    /// present the same frame again until the next vblank.
    ///
    /// When a present command is executed on the GPU, the frame will be put into the queue.
    /// If there was already a frame in the queue, the new frame will _replace_ the old frame.
    /// on the queue.
    ///
    /// * **Tearing**: No tearing will be observed.
    /// * **Also known as**: "Fast Vsync"
    #[builder(default = vk::PresentModeKHR::IMMEDIATE)]
    pub present_mode: vk::PresentModeKHR,

    /// The format and color space of the surface.
    pub surface: vk::SurfaceFormatKHR,

    /// The initial width of the surface.
    pub width: u32,

    /// NOTE: This field does not do anything, use the new one.
    #[builder(default)]
    #[builder_field_attr(deprecated = "use min_image_count field")]
    #[builder_setter_attr(deprecated = "use min_image_count field")]
    #[deprecated = "use min_image_count field"]
    pub desired_image_count: u32,
}

impl SwapchainInfo {
    /// Specifies a default swapchain with the given `width`, `height` and `format` values.
    #[inline(always)]
    pub fn new(width: u32, height: u32, surface: vk::SurfaceFormatKHR) -> SwapchainInfo {
        Self {
            #[allow(deprecated)]
            desired_image_count: 0,
            height,
            min_image_count: 2,
            present_mode: vk::PresentModeKHR::IMMEDIATE,
            surface,
            width,
        }
    }

    /// Creates a default `SwapchainInfoBuilder`.
    pub fn builder() -> SwapchainInfoBuilder {
        Default::default()
    }

    /// Converts a `SwapchainInfo` into a `SwapchainInfoBuilder`.
    pub fn into_builder(self) -> SwapchainInfoBuilder {
        SwapchainInfoBuilder {
            #[allow(deprecated)]
            desired_image_count: Some(self.desired_image_count),
            height: Some(self.height),
            min_image_count: Some(self.min_image_count),
            present_mode: Some(self.present_mode),
            surface: Some(self.surface),
            width: Some(self.width),
        }
    }

    #[deprecated = "use into_builder function"]
    #[doc(hidden)]
    pub fn to_builder(self) -> SwapchainInfoBuilder {
        self.into_builder()
    }
}

impl From<SwapchainInfoBuilder> for SwapchainInfo {
    fn from(info: SwapchainInfoBuilder) -> Self {
        info.build()
    }
}

impl SwapchainInfoBuilder {
    /// Builds a new `SwapchainInfo`.
    ///
    /// # Panics
    ///
    /// If any of the following values have not been set this function will panic.
    ///
    /// * `width`
    /// * `height`
    /// * `surface`
    #[inline(always)]
    pub fn build(self) -> SwapchainInfo {
        match self.fallible_build() {
            Err(SwapchainInfoBuilderError(err)) => panic!("{err}"),
            Ok(info) => info,
        }
    }
}

#[derive(Debug)]
struct SwapchainInfoBuilderError(UninitializedFieldError);

impl From<UninitializedFieldError> for SwapchainInfoBuilderError {
    fn from(err: UninitializedFieldError) -> Self {
        Self(err)
    }
}

mod deprecated {
    use ash::vk;

    use crate::driver::swapchain::SwapchainInfoBuilder;

    impl SwapchainInfoBuilder {
        #[deprecated = "use present_mode function"]
        #[doc(hidden)]
        pub fn present_modes(self, modes: impl Into<Vec<vk::PresentModeKHR>>) -> Self {
            self.present_mode(modes.into()[0])
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    type Info = SwapchainInfo;
    type Builder = SwapchainInfoBuilder;

    #[test]
    pub fn swapchain_info() {
        let info = Info::new(20, 24, vk::SurfaceFormatKHR::default());
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn swapchain_info_builder() {
        let info = Info::new(23, 64, vk::SurfaceFormatKHR::default());
        let builder = Builder::default()
            .width(23)
            .height(64)
            .surface(vk::SurfaceFormatKHR::default())
            .build();

        assert_eq!(info, builder);
    }

    #[test]
    #[should_panic(expected = "Field not initialized: height")]
    pub fn swapchain_info_builder_uninit_height() {
        Builder::default().build();
    }

    #[test]
    #[should_panic(expected = "Field not initialized: surface")]
    pub fn swapchain_info_builder_uninit_surface() {
        Builder::default().height(42).build();
    }

    #[test]
    #[should_panic(expected = "Field not initialized: width")]
    pub fn swapchain_info_builder_uninit_width() {
        Builder::default()
            .height(42)
            .surface(vk::SurfaceFormatKHR::default())
            .build();
    }
}
