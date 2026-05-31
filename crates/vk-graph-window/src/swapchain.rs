//! Window swapchain creation, acquisition, and presentation helpers.

use {
    derive_builder::{Builder, UninitializedFieldError},
    log::{trace, warn},
    std::{
        error::Error,
        fmt::{Debug, Formatter},
        ops::Deref,
        slice,
        thread::panicking,
        time::Instant,
    },
    vk_graph::{
        Graph,
        driver::{
            DriverError,
            ash::{self, vk},
            cmd_buf::{CommandBuffer, CommandBufferInfo},
            descriptor_set::{DescriptorPool, DescriptorPoolInfo},
            device::Device,
            image::Image,
            render_pass::{RenderPass, RenderPassInfo},
            surface::Surface,
            swapchain::{self, SwapchainImage},
            sync::{AccessType, ImageBarrier, ImageLayout, cmd::pipeline_barrier},
        },
        node::SwapchainImageNode,
        pool::Pool,
    },
};

fn create_semaphore(device: &ash::Device) -> Result<vk::Semaphore, DriverError> {
    let create_info = vk::SemaphoreCreateInfo::default();
    let allocation_callbacks = None;

    unsafe { device.create_semaphore(&create_info, allocation_callbacks) }.map_err(|err| {
        warn!("unable to create semaphore: {err}");

        DriverError::OutOfMemory
    })
}

const fn image_access_layout(access: AccessType) -> ImageLayout {
    if matches!(access, AccessType::Present | AccessType::ComputeShaderWrite) {
        ImageLayout::General
    } else {
        ImageLayout::Optimal
    }
}

/// A physical display interface.
#[read_only::embed]
pub struct Swapchain {
    exec_idx: usize,
    execs: Box<[Execution]>,
    image_execs: Vec<usize>,

    /// Information used to create this resource.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub info: SwapchainInfo,

    /// The swapchain which supports this display.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub swapchain: swapchain::Swapchain,
}

impl Deref for ReadOnlySwapchain {
    type Target = swapchain::Swapchain;

    fn deref(&self) -> &Self::Target {
        &self.swapchain
    }
}

impl Swapchain {
    /// Constructs a new `Swapchain` object.
    pub fn new(surface: Surface, info: impl Into<SwapchainInfo>) -> Result<Self, DriverError> {
        let info = info.into();

        assert_ne!(info.command_buffer_count, 0);

        let swapchain_info: swapchain::SwapchainInfo = info.into();
        let swapchain = swapchain::Swapchain::new(surface, swapchain_info)?;

        let mut execs = Vec::with_capacity(info.command_buffer_count as _);
        for _ in 0..info.command_buffer_count {
            let cmd = CommandBuffer::create(
                &swapchain.surface.device,
                CommandBufferInfo::new(info.queue_family_index),
            )?;
            let swapchain_acquired = create_semaphore(&swapchain.surface.device)?;
            let swapchain_rendered = create_semaphore(&swapchain.surface.device)?;

            execs.push(Execution {
                cmd,
                swapchain_acquired,
                swapchain_rendered,
            });
        }
        let execs = execs.into_boxed_slice();

        Ok(Self {
            exec_idx: info.command_buffer_count,
            execs,
            image_execs: Default::default(),
            read_only: ReadOnlySwapchain { info, swapchain },
        })
    }

    /// Acquires the next available swapchain image for rendering.
    ///
    /// This is a high-level wrapper around [`Swapchain::acquire_next_image`].
    /// It manages the internal execution slot, fence, and semaphore lifecycle automatically.
    ///
    /// The returned [`SwapchainImage`] (if `Some`) has already had its acquire semaphore
    /// submitted to the associated queue for this frame. The caller should record and
    /// submit rendering commands that depend on this image in the current frame's graph.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(image))` — An image was acquired and may be rendered to.
    /// * `Ok(None)` — The swapchain is suboptimal (e.g. the window was resized). Render nothing for
    ///   this frame and call [`acquire_next_image`][Self::acquire_next_image] again next frame.
    /// * `Err(SwapchainError::DeviceLost)` — The Vulkan device has been lost.
    /// * `Err(SwapchainError::Driver(_))` — The surface was lost or another unrecoverable driver
    ///   error occurred.
    ///
    /// # Internal behavior
    ///
    /// Each call advances to the next internal execution slot (round-robin across
    /// a fixed number of submissions in flight). Before acquiring, the method waits
    /// on the fence for the previous submission in that slot, ensuring at most one
    /// in-flight submission per slot. It then calls the core
    /// [`acquire_next_image`](vk_graph::driver::swapchain::Swapchain::acquire_next_image)
    /// with an infinite timeout ([`u64::MAX`]) and the slot's acquire semaphore.
    ///
    /// After successful acquisition the slot index is recorded against the returned
    /// image index so that future waits are correctly associated with the right
    /// submission.
    ///
    /// # Errors
    ///
    /// Returns `Ok(None)` instead of surfacing a suboptimal-state error; the window-level
    /// API collapses suboptimal into a non-error signal so the draw loop can skip
    /// the frame gracefully.
    pub fn acquire_next_image(&mut self) -> Result<Option<SwapchainImage>, SwapchainError> {
        self.exec_idx += 1;
        self.exec_idx %= self.execs.len();
        let exec = &mut self.execs[self.exec_idx];

        exec.cmd.wait_until_executed().inspect_err(|err| {
            warn!("unable to wait for swapchain fence: {err}");
        })?;

        Device::reset_fences(&exec.cmd.device, slice::from_ref(&exec.cmd.fence))?;

        let acquire_next_image = self
            .read_only
            .swapchain
            .acquire_next_image(u64::MAX, exec.swapchain_acquired);

        if let Err(err) = acquire_next_image {
            warn!("unable to acquire next swapchain image: {err:?}");
        }

        let swapchain_image = match acquire_next_image {
            Err(swapchain::SwapchainError::DeviceLost) => Err(SwapchainError::DeviceLost),
            Err(swapchain::SwapchainError::Suboptimal) => return Ok(None),
            Err(swapchain::SwapchainError::SurfaceLost) => {
                warn!("invalid swapchain surface state: surface lost");
                Err(SwapchainError::Driver(DriverError::InvalidData))
            }
            Ok(swapchain_image) => Ok(swapchain_image),
        }?;

        while swapchain_image.index >= self.image_execs.len() as u32 {
            self.image_execs.push(0);
        }

        self.image_execs[swapchain_image.index as usize] = self.exec_idx;

        Ok(Some(swapchain_image))
    }

    /// Displays the given swapchain image using passes specified in `graph`, if possible.
    #[profiling::function]
    pub fn present_image<P>(
        &mut self,
        pool: &mut P,
        graph: Graph,
        swapchain_image: SwapchainImageNode,
        queue_index: u32,
    ) -> Result<(), SwapchainError>
    where
        P: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
    {
        trace!("present_image");

        let mut submission = graph.into_submission();
        let wait_dst_stage_mask = submission.node_stages(swapchain_image);

        // The swapchain should have been written to, otherwise it would be noise and that's a panic
        assert!(
            !wait_dst_stage_mask.is_empty(),
            "uninitialized swapchain image: write something each frame!",
        );

        let image_idx = submission.resource(swapchain_image).index;
        let exec_idx = self.image_execs[image_idx as usize];
        let exec = &mut self.execs[exec_idx];

        debug_assert!(!exec.cmd.has_executed().unwrap());

        let started = Instant::now();

        Device::begin_command_buffer(
            &exec.cmd.device,
            exec.cmd.handle,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        // submission.record_node_dependencies(&mut *self.pool, cmd, swapchain_image)?;
        submission.queue_resource(swapchain_image, pool, &mut exec.cmd)?;

        {
            let swapchain_image = submission.resource(swapchain_image);
            for (access, range) in Image::access(
                swapchain_image,
                AccessType::Present,
                vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_array_layer: 0,
                    base_mip_level: 0,
                    layer_count: 1,
                    level_count: 1,
                },
            ) {
                trace!(
                    "image {:?} {:?}->{:?}",
                    swapchain_image,
                    access,
                    AccessType::Present,
                );

                // Force a presentation layout transition
                pipeline_barrier(
                    &exec.cmd.device,
                    exec.cmd.handle,
                    None,
                    &[],
                    slice::from_ref(&ImageBarrier {
                        previous_accesses: slice::from_ref(&access),
                        previous_layout: image_access_layout(access),
                        next_accesses: slice::from_ref(&AccessType::Present),
                        next_layout: image_access_layout(AccessType::Present),
                        discard_contents: false,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        image: swapchain_image.handle,
                        range,
                    }),
                );
            }
        }

        // We may have unresolved nodes; things like copies that happen after present or operations
        // before present which use nodes that are unused in the remainder of the graph.
        // These operations are still important, but they don't need to wait for any of the above
        // things so we do them last
        submission.submit_cmd_buf(pool, &mut exec.cmd)?;

        Device::with_queue(
            &exec.cmd.device,
            self.read_only.info.queue_family_index,
            queue_index,
            |queue| {
                Device::end_command_buffer(&exec.cmd.device, exec.cmd.handle)?;
                Device::queue_submit(
                    &exec.cmd.device,
                    queue,
                    slice::from_ref(
                        &vk::SubmitInfo::default()
                            .command_buffers(slice::from_ref(&exec.cmd.handle))
                            .wait_semaphores(slice::from_ref(&exec.swapchain_acquired))
                            .wait_dst_stage_mask(slice::from_ref(&wait_dst_stage_mask))
                            .signal_semaphores(slice::from_ref(&exec.swapchain_rendered)),
                    ),
                    exec.cmd.fence,
                )
            },
        )?;

        let elapsed = Instant::now() - started;
        trace!("🔜🔜🔜 vkQueueSubmit took {} μs", elapsed.as_micros(),);

        let swapchain_image = submission.resource(swapchain_image).clone();

        self.read_only.swapchain.present_image(
            swapchain_image,
            slice::from_ref(&exec.swapchain_rendered),
            self.read_only.info.queue_family_index,
            queue_index,
        )?;

        // Store the resolved graph because it contains bindings, pooled resources, and other shared
        // resources that need to be kept alive until the fence is waited upon.
        exec.cmd.drop_after_executed(submission);

        Ok(())
    }

    /// Updates the information which controls the swapchain.
    ///
    /// Previously acquired swapchain images should be discarded after calling this function.
    pub fn set_info(&mut self, info: impl Into<swapchain::SwapchainInfo>) {
        let info = info.into();

        self.read_only.swapchain.set_info(info);
        self.read_only.info.height = info.height;
        self.read_only.info.min_image_count = info.min_image_count;
        self.read_only.info.present_mode = info.present_mode;
        self.read_only.info.surface = info.surface;
        self.read_only.info.width = info.width;
    }
}

impl Debug for Swapchain {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Swapchain")
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        let idle = unsafe { self.execs[0].cmd.device.device_wait_idle() };
        if idle.is_err() {
            warn!("unable to wait for device");

            return;
        }

        for batch in &mut self.execs {
            // Wait for presentation to stop
            let present = batch.cmd.wait_until_executed();
            if present.is_err() {
                warn!("unable to wait for queue");

                continue;
            }

            unsafe {
                batch
                    .cmd
                    .device
                    .destroy_semaphore(batch.swapchain_acquired, None);
                batch
                    .cmd
                    .device
                    .destroy_semaphore(batch.swapchain_rendered, None);
            }
        }
    }
}

/// Describes error conditions relating to physical displays.
#[derive(Clone, Copy, Debug)]
pub enum SwapchainError {
    /// Unrecoverable device error; must destroy this device and display and start a new one.
    DeviceLost,

    /// Recoverable driver error.
    Driver(DriverError),
}

impl Error for SwapchainError {}

impl From<()> for SwapchainError {
    fn from(_: ()) -> Self {
        Self::DeviceLost
    }
}

impl From<DriverError> for SwapchainError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl std::fmt::Display for SwapchainError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
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
    /// The number of command buffers to use for image submissions.
    ///
    /// Generally one more than the swapchain image count is best.
    #[builder(default = "4")]
    pub command_buffer_count: usize,

    /// The initial height of the surface.
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
    #[builder(default = vk::PresentModeKHR::MAILBOX)]
    pub present_mode: vk::PresentModeKHR,

    /// The device queue family which will be used to submit and present images.
    #[builder(default = "0")]
    pub queue_family_index: u32,

    /// The format and color space of the surface.
    pub surface: vk::SurfaceFormatKHR,

    /// The initial width of the surface.
    pub width: u32,
}

impl SwapchainInfo {
    /// Specifies a default swapchain with the given `width`, `height` and `format` values.
    #[inline(always)]
    pub fn new(width: u32, height: u32, surface: vk::SurfaceFormatKHR) -> SwapchainInfo {
        Self {
            command_buffer_count: 4,
            height,
            min_image_count: 2,
            present_mode: vk::PresentModeKHR::MAILBOX,
            queue_family_index: 0,
            surface,
            width,
        }
    }

    /// Converts a `SwapchainInfo` into a `SwapchainInfoBuilder`.
    pub fn into_builder(self) -> SwapchainInfoBuilder {
        SwapchainInfoBuilder {
            command_buffer_count: Some(self.command_buffer_count),
            height: Some(self.height),
            min_image_count: Some(self.min_image_count),
            present_mode: Some(self.present_mode),
            queue_family_index: Some(self.queue_family_index),
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

impl From<SwapchainInfo> for swapchain::SwapchainInfo {
    fn from(val: SwapchainInfo) -> Self {
        swapchain::SwapchainInfoBuilder::default()
            .height(val.height)
            .min_image_count(val.min_image_count)
            .present_mode(val.present_mode)
            .surface(val.surface)
            .width(val.width)
            .build()
    }
}

impl SwapchainInfoBuilder {
    /// Builds a new `SwapchainInfo`.
    ///
    /// # Panics
    ///
    /// If any of the following values have not been set this function will panic.
    ///
    /// * `command_buffer_count`
    /// * `width`
    /// * `height`
    /// * `surface`
    #[inline(always)]
    pub fn build(self) -> SwapchainInfo {
        let info = match self.fallible_build() {
            Err(SwapchainInfoBuilderError(err)) => panic!("{err}"),
            Ok(info) => info,
        };

        assert_ne!(
            info.command_buffer_count, 0,
            "Field value invalid: command_buffer_count"
        );

        info
    }
}

#[derive(Debug)]
struct SwapchainInfoBuilderError(UninitializedFieldError);

impl From<UninitializedFieldError> for SwapchainInfoBuilderError {
    fn from(err: UninitializedFieldError) -> Self {
        Self(err)
    }
}

struct Execution {
    cmd: CommandBuffer,
    swapchain_acquired: vk::Semaphore,
    swapchain_rendered: vk::Semaphore,
}

#[cfg(test)]
mod test {
    use super::*;

    type Info = SwapchainInfo;
    type Builder = SwapchainInfoBuilder;

    #[test]
    pub fn swapchain_info() {
        let info = Info {
            command_buffer_count: 42,
            height: 123,
            min_image_count: 99,
            present_mode: vk::PresentModeKHR::IMMEDIATE,
            queue_family_index: 16,
            surface: vk::SurfaceFormatKHR::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .color_space(vk::ColorSpaceKHR::PASS_THROUGH_EXT),
            width: 88,
        };
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn swapchain_info_builder() {
        let info = Info {
            command_buffer_count: 42,
            height: 123,
            min_image_count: 99,
            present_mode: vk::PresentModeKHR::IMMEDIATE,
            queue_family_index: 16,
            surface: vk::SurfaceFormatKHR::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .color_space(vk::ColorSpaceKHR::PASS_THROUGH_EXT),
            width: 88,
        };
        let builder = Builder::default()
            .command_buffer_count(42)
            .height(123)
            .min_image_count(99)
            .present_mode(vk::PresentModeKHR::IMMEDIATE)
            .queue_family_index(16)
            .surface(
                vk::SurfaceFormatKHR::default()
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .color_space(vk::ColorSpaceKHR::PASS_THROUGH_EXT),
            )
            .width(88)
            .build();

        assert_eq!(info, builder);
    }

    #[test]
    #[should_panic(expected = "Field value invalid: command_buffer_count")]
    pub fn swapchain_info_builder_uninit_command_buffer_count() {
        Builder::default()
            .height(1)
            .surface(vk::SurfaceFormatKHR::default())
            .width(1)
            .command_buffer_count(0)
            .build();
    }

    #[test]
    #[should_panic(expected = "Field not initialized: height")]
    pub fn swapchain_info_builder_uninit_height() {
        Builder::default().build();
    }

    #[test]
    #[should_panic(expected = "Field not initialized: surface")]
    pub fn swapchain_info_builder_uninit_surface() {
        Builder::default().height(1).build();
    }

    #[test]
    #[should_panic(expected = "Field not initialized: width")]
    pub fn swapchain_info_builder_uninit_width() {
        Builder::default()
            .height(1)
            .surface(vk::SurfaceFormatKHR::default())
            .build();
    }
}
