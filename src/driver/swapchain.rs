//! Native window presentation types.

use {
    super::{
        DriverError, Surface,
        device::Device,
        image::{Image, ImageInfo},
    },
    ash::vk::{self, Handle},
    derive_builder::Builder,
    log::{debug, info, trace, warn},
    std::{mem::replace, ops::Deref, slice, thread::panicking},
};

#[derive(Debug)]
struct QueueFamilySignal {
    cmd_pool: vk::CommandPool,
    queue_signals: Box<[QueueSignal]>,
}

#[derive(Debug)]
struct QueueSignal {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    queued: bool,
}

#[derive(Debug, Default)]
struct QueueSwapchain {
    handle: vk::SwapchainKHR,
    queue_family_index: u32,
    queue_index: u32,
}

/// Provides the ability to present rendering results to a [`Surface`].
#[derive(Debug)]
#[read_only::cast]
pub struct Swapchain {
    /// The native Vulkan resource handle of this swapchain.
    ///
    /// _Note:_ This field is read-only.
    pub handle: vk::SwapchainKHR,

    images: Box<[SwapchainImage]>,

    /// Information used to create this resource.
    ///
    /// _Note:_ This field is read-only.
    pub info: SwapchainInfo,

    prev_queue: (u32, u32),
    prev_swapchain: QueueSwapchain,
    queue_family_signals: Box<[QueueFamilySignal]>,
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

        let device = &surface.device;
        let queue_families = surface.device.physical_device.queue_families.iter();
        let mut queue_family_signals = Vec::with_capacity(queue_families.len());

        for (idx, queue_family) in queue_families.enumerate() {
            let cmd_pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(idx as _);
            if !surface.physical_device_support(cmd_pool_info.queue_family_index)? {
                continue;
            }

            let cmd_pool = unsafe {
                device
                    .create_command_pool(&cmd_pool_info, None)
                    .map_err(|err| {
                        warn!("unable to create command pool: {err}");

                        match err {
                            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                            | vk::Result::ERROR_OUT_OF_HOST_MEMORY => DriverError::OutOfMemory,
                            _ => DriverError::Unsupported,
                        }
                    })?
            };
            let cmd_bufs = unsafe {
                device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_buffer_count(queue_family.queue_count)
                            .command_pool(cmd_pool)
                            .level(vk::CommandBufferLevel::PRIMARY),
                    )
                    .map_err(|err| {
                        warn!("unable to allocate command buffer: {err}");

                        match err {
                            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                            | vk::Result::ERROR_OUT_OF_HOST_MEMORY => DriverError::OutOfMemory,
                            _ => DriverError::Unsupported,
                        }
                    })?
            };

            let mut queue_signals = Vec::with_capacity(queue_family.queue_count as _);

            for cmd in cmd_bufs {
                Device::begin_command_buffer(
                    device,
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE),
                )?;
                Device::end_command_buffer(device, cmd)?;

                let queued = false;
                let fence = Device::create_fence(device, queued)?;
                queue_signals.push(QueueSignal { cmd, fence, queued });
            }

            queue_family_signals.push(QueueFamilySignal {
                cmd_pool,
                queue_signals: queue_signals.into_boxed_slice(),
            });
        }

        Ok(Swapchain {
            handle: Default::default(),
            images: Default::default(),
            info,
            prev_queue: Default::default(),
            prev_swapchain: Default::default(),
            queue_family_signals: queue_family_signals.into_boxed_slice(),
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
    /// * `timeout` — Maximum time to wait in nanoseconds before the operation times out. Pass
    ///   [`u64::MAX`] to wait indefinitely. A short timeout may return
    ///   [`SwapchainError::Suboptimal`] without acquiring an image.
    /// * `acquired` — A semaphore that will be signaled when the acquired image is ready for use.
    ///   The caller must wait on this semaphore before submitting commands that write to the
    ///   returned image.
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

            let ext = Device::expect_vk_khr_swapchain(&self.surface.device);

            let image_idx = unsafe {
                ext.acquire_next_image(self.handle, timeout, acquired, vk::Fence::null())
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
    fn destroy(device: &Device, handle: &mut vk::SwapchainKHR) {
        if *handle == vk::SwapchainKHR::null() {
            return;
        }

        // wait for device to be finished with swapchain before destroying it.
        // This avoid crashes when resizing windows
        #[cfg(target_os = "macos")]
        if let Err(err) = unsafe { device.device_wait_idle() } {
            warn!("device_wait_idle() failed: {err}");
        }

        let ext = Device::expect_vk_khr_swapchain(device);

        unsafe {
            ext.destroy_swapchain(*handle, None);
        }

        *handle = vk::SwapchainKHR::null();
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
    ) -> Result<(), DriverError> {
        let device = &self.surface.device;
        let queue_signal = &mut self.queue_family_signals[queue_family_index as usize]
            .queue_signals[queue_index as usize];

        let ext = Device::expect_vk_khr_swapchain(device);

        Device::with_queue(device, queue_family_index, queue_index, |queue| {
            unsafe {
                match ext.queue_present(
                    queue,
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(wait_semaphores)
                        .swapchains(slice::from_ref(&self.handle))
                        .image_indices(slice::from_ref(&image.index)),
                ) {
                    Ok(suboptimal) => self.suboptimal = suboptimal,
                    Err(err)
                        if err == vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT
                            || err == vk::Result::ERROR_OUT_OF_DATE_KHR
                            || err == vk::Result::SUBOPTIMAL_KHR =>
                    {
                        self.suboptimal = true
                    }
                    Err(err)
                        if err == vk::Result::ERROR_DEVICE_LOST
                            || err == vk::Result::ERROR_SURFACE_LOST_KHR =>
                    {
                        info!("skipping present: {err}");

                        self.suboptimal = true;
                    }
                    Err(err) => {
                        // Probably:
                        // VK_ERROR_OUT_OF_HOST_MEMORY
                        // VK_ERROR_OUT_OF_DEVICE_MEMORY
                        warn!("failed to present: {err}");

                        self.suboptimal = true;
                    }
                }
            }

            if queue_signal.queued {
                Device::wait_for_fence(device, &queue_signal.fence)?;
                Device::reset_fences(device, slice::from_ref(&queue_signal.fence))?;
            }

            Device::queue_submit(
                device,
                queue,
                &[vk::SubmitInfo::default().command_buffers(slice::from_ref(&queue_signal.cmd))],
                queue_signal.fence,
            )?;
            queue_signal.queued = true;

            Ok::<_, DriverError>(())
        })?;

        if !self.prev_swapchain.handle.is_null() {
            let QueueSignal { fence, queued, .. } = &mut self.queue_family_signals
                [self.prev_swapchain.queue_family_index as usize]
                .queue_signals[self.prev_swapchain.queue_index as usize];
            Device::wait_for_fence(device, &*fence)?;
            Device::reset_fences(device, slice::from_ref(&*fence))?;
            *queued = false;

            Self::destroy(device, &mut self.prev_swapchain.handle);
        }

        let image_idx = image.index as usize;
        self.images[image_idx] = image;

        self.prev_queue = (queue_family_index, queue_index);

        Ok(())
    }

    #[profiling::function]
    fn recreate(&mut self) -> Result<(), DriverError> {
        self.wait_for_all_signals()?;
        Self::destroy(&self.surface.device, &mut self.prev_swapchain.handle);

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

        let ext = Device::expect_vk_khr_swapchain(&self.surface.device);
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
        let swapchain =
            unsafe { ext.create_swapchain(&swapchain_create_info, None) }.map_err(|err| {
                warn!("unable to create swapchain: {err}");

                DriverError::Unsupported
            })?;

        let images = unsafe { ext.get_swapchain_images(swapchain) }.map_err(|err| match err {
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

                SwapchainImage {
                    read_only: ReadOnlySwapchainImage {
                        image,
                        index: image_idx,
                    },
                }
            })
            .collect::<Box<_>>();

        self.info.height = surface_height;
        self.info.width = surface_width;
        self.images = images;
        self.prev_swapchain.handle = self.handle;
        self.prev_swapchain.queue_family_index = self.prev_queue.0;
        self.prev_swapchain.queue_index = self.prev_queue.1;
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

    /// Waits for all tracked per-queue fences to become signaled and resets their state.
    fn wait_for_all_signals(&mut self) -> Result<(), DriverError> {
        let fences = self
            .queue_family_signals
            .iter()
            .flat_map(|queue_family| {
                queue_family
                    .queue_signals
                    .iter()
                    .filter_map(|queue| queue.queued.then_some(queue.fence))
            })
            .collect::<Box<_>>();
        if fences.is_empty() {
            return Ok(());
        }

        let device = &self.surface.device;

        Device::wait_for_fences(device, &fences)?;
        Device::reset_fences(device, &fences)?;

        for queue_family in &mut self.queue_family_signals {
            for queue in &mut queue_family.queue_signals {
                queue.queued = false;
            }
        }

        Ok(())
    }
}

impl Drop for Swapchain {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        if let Err(err) = self.wait_for_all_signals() {
            warn!("unable to wait for swapchain signals: {err}");
        }

        let device = &self.surface.device;

        Self::destroy(device, &mut self.handle);
        Self::destroy(device, &mut self.prev_swapchain.handle);

        for queue_family in &self.queue_family_signals {
            let mut cmd_bufs = Vec::with_capacity(queue_family.queue_signals.len());
            for queue in &queue_family.queue_signals {
                cmd_bufs.push(queue.cmd);

                unsafe {
                    device.destroy_fence(queue.fence, None);
                }
            }

            unsafe {
                device.free_command_buffers(queue_family.cmd_pool, &cmd_bufs);
                device.destroy_command_pool(queue_family.cmd_pool, None);
            }
        }
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

impl SwapchainImage {
    #[cfg(test)]
    pub(crate) fn from_raw(
        device: &Device,
        handle: vk::Image,
        info: impl Into<ImageInfo>,
        index: u32,
    ) -> Self {
        Self {
            read_only: ReadOnlySwapchainImage {
                image: Image::from_raw(device, handle, info),
                index,
            },
        }
    }
}

impl Clone for SwapchainImage {
    fn clone(&self) -> Self {
        Self {
            read_only: ReadOnlySwapchainImage {
                image: unsafe { self.image.to_detached() },
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
    build_fn(private, name = "fallible_build"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct SwapchainInfo {
    /// The initial height of the surface.
    #[builder(default)]
    pub height: u32,

    /// The minimum number of presentable images that the application needs. The implementation
    /// will either create the swapchain with at least that many images, or it will fail to
    /// create the swapchain.
    ///
    /// More images introduce more display lag, but smoother animation.
    #[builder(default = "2")]
    pub min_image_count: u32,

    /// `vk::PresentModeKHR` determines timing and queueing with which frames are actually
    /// displayed to the user.
    ///
    /// `vk::PresentModeKHR::FIFO` - Presentation frames are kept in a First-In-First-Out queue
    /// approximately 3 frames long. Every vertical blanking period, the presentation engine
    /// will pop a frame off the queue to display. If there is no frame to display, it will
    /// present the same frame again until the next vblank.
    ///
    /// When a present command is executed on the GPU, the presented image is added on the queue.
    ///
    /// * **Tearing:** No tearing will be observed.
    /// * **Also known as**: "Vsync On"
    ///
    /// `vk::PresentModeKHR::FIFO_RELAXED` - Presentation frames are kept in a First-In-First-Out
    /// queue approximately 3 frames long. Every vertical blanking period, the presentation
    /// engine will pop a frame off the queue to display. If there is no frame to display, it
    /// will present the same frame until there is a frame in the queue. The moment there is a
    /// frame in the queue, it will immediately pop the frame off the queue.
    ///
    /// When a present command is executed on the GPU, the presented image is added on the queue.
    ///
    /// * **Tearing**: Tearing will be observed if frames last more than one vblank as the front
    ///   buffer.
    /// * **Also known as**: "Adaptive Vsync"
    ///
    /// `vk::PresentModeKHR::IMMEDIATE` - Presentation frames are not queued at all. The moment a
    /// present command is executed on the GPU, the presented image is swapped onto the front
    /// buffer immediately.
    ///
    /// * **Tearing**: Tearing can be observed.
    /// * **Also known as**: "Vsync Off"
    ///
    /// `vk::PresentModeKHR::MAILBOX` - Presentation frames are kept in a single-frame queue. Every
    /// vertical blanking period, the presentation engine will pop a frame from the queue. If
    /// there is no frame to display, it will present the same frame again until the next
    /// vblank.
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
    #[builder(default)]
    pub surface: vk::SurfaceFormatKHR,

    /// The initial width of the surface.
    #[builder(default)]
    pub width: u32,
}

impl SwapchainInfo {
    /// Specifies a default swapchain with the given `width`, `height` and `format` values.
    #[inline(always)]
    pub fn new(width: u32, height: u32, surface: vk::SurfaceFormatKHR) -> SwapchainInfo {
        Self {
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
            height: Some(self.height),
            min_image_count: Some(self.min_image_count),
            present_mode: Some(self.present_mode),
            surface: Some(self.surface),
            width: Some(self.width),
        }
    }
}

impl From<SwapchainInfoBuilder> for SwapchainInfo {
    fn from(info: SwapchainInfoBuilder) -> Self {
        info.build()
    }
}

impl SwapchainInfoBuilder {
    /// Builds a new `SwapchainInfo`.
    #[inline(always)]
    pub fn build(self) -> SwapchainInfo {
        self.fallible_build().expect("all fields have defaults")
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
    pub fn swapchain_info_builder_defaults() {
        assert_eq!(
            Builder::default().build(),
            Info::new(0, 0, vk::SurfaceFormatKHR::default())
        );
    }
}
