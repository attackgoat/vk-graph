//! Window graphchain creation, acquisition, and presentation helpers.

use {
    derive_builder::Builder,
    log::{Level, log_enabled, trace, warn},
    std::{
        error::Error,
        fmt::{Debug, Formatter},
        mem::take,
        ops::Deref,
        slice,
        thread::panicking,
    },
    vk_graph::{
        Graph,
        driver::{
            DriverError,
            ash::vk,
            cmd_buf::{CommandBuffer, CommandBufferInfo},
            device::Device,
            fence::Fence,
            surface::Surface,
            swapchain::{
                PresentError, PresentFailure, PresentInfo, PresentResult, Swapchain,
                SwapchainError, SwapchainImage, SwapchainInfo, SwapchainInfoBuilder,
            },
        },
        node::SwapchainImageNode,
        pool::{Pool, SubmissionPool},
        submission::{QueueSubmitInfo, RecordSelection, SemaphoreSubmitInfo},
    },
};

/// A physical display interface.
#[read_only::embed]
pub struct Graphchain {
    frame_idx: usize,
    frames: Box<[FrameSlot]>,
    image_frames: Vec<usize>,
    last_present_queue_index: u32,
    queue_family_index: u32,
    recreate_pending: bool,
    requested_info: GraphchainInfo,
    strategy: PresentRetirementStrategy,

    /// Effective runtime information for the live graphchain.
    ///
    /// Calls to [`set_info`](Self::set_info) update the requested configuration. This field is
    /// updated lazily after the next successful acquire recreates the swapchain.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub info: EffectiveGraphchainInfo,

    /// The driver swapchain backing this graphchain.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub swapchain: Swapchain,
}

impl Deref for ReadOnlyGraphchain {
    type Target = Swapchain;

    fn deref(&self) -> &Self::Target {
        &self.swapchain
    }
}

impl Graphchain {
    /// Constructs a new `Graphchain` object.
    pub fn new(surface: Surface, info: impl Into<GraphchainInfo>) -> Result<Self, DriverError> {
        let mut info = info.into();

        // Frame capacity must always be able to cover at least the requested minimum image count.
        info.frame_capacity = info
            .frame_capacity
            .max(info.min_image_count as usize)
            .max(1);

        let swapchain_info: SwapchainInfo = info.into();
        let swapchain = Swapchain::create(surface, swapchain_info)?;
        let physical_device = &swapchain.surface.device.physical;
        let queue_family_index = physical_device
            .queue_families
            .iter()
            .enumerate()
            .find_map(|(idx, _)| {
                swapchain
                    .surface
                    .physical_device_support(idx as u32)
                    .ok()
                    .and_then(|supported| supported.then_some(idx as u32))
            })
            .ok_or(DriverError::Unsupported)?;

        // Use the modern strategy only when both present-id and present-wait are supported.
        let strategy = if swapchain
            .surface
            .device
            .physical
            .vk_khr_present_id
            .is_some()
            && swapchain
                .surface
                .device
                .physical
                .vk_khr_present_wait
                .is_some()
        {
            PresentRetirementStrategy::PresentWait(PresentWait::default())
        } else {
            PresentRetirementStrategy::PerImageSemaphore(PerImageSemaphore::default())
        };

        let mut frames = Vec::with_capacity(info.frame_capacity);
        for _ in 0..info.frame_capacity {
            let cmd_buf = CommandBuffer::create(
                &swapchain.surface.device,
                CommandBufferInfo::new(queue_family_index),
            )?;
            let fence = Fence::create(&swapchain.surface.device, false)?;
            let swapchain_acquired = Device::create_semaphore(&swapchain.surface.device)?;
            Device::try_set_debug_utils_object_name(
                &swapchain.surface.device,
                swapchain_acquired,
                "graphchain acquired semaphore",
            );

            frames.push(FrameSlot {
                cmd_buf,
                fence,
                swapchain_acquired,
            });
        }

        let effective_info = {
            let GraphchainInfo {
                acquire_timeout,
                clipped,
                composite_alpha,
                frame_capacity,
                height,
                min_image_count,
                present_mode,
                surface,
                width,
            } = info;

            EffectiveGraphchainInfo {
                acquire_timeout,
                clipped,
                composite_alpha,
                frame_capacity,
                frame_count: frame_capacity,
                height,
                image_count: swapchain.info.image_count as usize,
                min_image_count,
                present_mode,
                surface,
                width,
            }
        };

        Ok(Self {
            frame_idx: info.frame_capacity - 1,
            frames: frames.into_boxed_slice(),
            image_frames: Vec::new(),
            last_present_queue_index: 0,
            queue_family_index,
            recreate_pending: false,
            requested_info: info,
            strategy,
            read_only: ReadOnlyGraphchain {
                info: effective_info,
                swapchain,
            },
        })
    }

    /// Acquires the next available swapchain image for rendering.
    pub fn acquire_next_image(&mut self) -> Result<Option<SwapchainImage>, GraphchainError> {
        if self.recreate_pending {
            for frame in &mut self.frames {
                if frame.fence.is_queued() {
                    frame.fence.wait_signaled()?.reset()?;
                }
            }

            self.strategy
                .retire_pending(&mut self.read_only.swapchain)?;

            if self.read_only.swapchain.recreate_pending {
                self.read_only.swapchain.recreate()?;
            }

            self.strategy
                .reset(&self.read_only.swapchain.surface.device);
            self.image_frames.clear();
            self.recreate_pending = false;
        }

        self.frame_idx += 1;
        self.frame_idx %= self.frames.len();

        let frame = &mut self.frames[self.frame_idx];

        if frame.fence.is_queued() {
            frame.fence.wait_signaled()?.reset()?;
        }

        self.strategy.prepare_frame(
            &frame.cmd_buf.device,
            &mut self.read_only.swapchain,
            self.frame_idx,
        )?;

        let swapchain_image = match self
            .read_only
            .swapchain
            .acquire_next_image(frame.swapchain_acquired)
        {
            Ok(acquired) => acquired,
            Err(SwapchainError::DeviceLost) => return Err(GraphchainError::DeviceLost),
            Err(SwapchainError::Driver(err)) => return Err(GraphchainError::Driver(err)),
            Err(
                SwapchainError::FullScreenExclusiveModeLost
                | SwapchainError::NotReady
                | SwapchainError::Timeout,
            ) => return Ok(None),
            Err(SwapchainError::SurfaceLost) => {
                return Err(GraphchainError::SurfaceLost);
            }
        };

        if swapchain_image.suboptimal {
            self.recreate_pending = true;
        }

        let swapchain_info = self.read_only.swapchain.info;
        self.read_only.info.acquire_timeout = swapchain_info.acquire_timeout;
        self.read_only.info.clipped = swapchain_info.clipped;
        self.read_only.info.composite_alpha = swapchain_info.composite_alpha;
        self.read_only.info.frame_capacity = self.requested_info.frame_capacity;
        self.read_only.info.height = swapchain_info.height;
        self.read_only.info.image_count = swapchain_info.image_count as usize;
        self.read_only.info.present_mode = swapchain_info.present_mode;
        self.read_only.info.surface = swapchain_info.surface;
        self.read_only.info.width = swapchain_info.width;

        // Live frame count cannot be smaller than the live swapchain image count.
        self.read_only.info.frame_count = self
            .read_only
            .info
            .frame_capacity
            .max(self.read_only.info.image_count);

        self.ensure_frame_count()?;

        while swapchain_image.index >= self.image_frames.len() as u32 {
            self.image_frames.push(0);
        }

        self.image_frames[swapchain_image.index as usize] = self.frame_idx;
        self.strategy.acquire_image(
            &self.read_only.swapchain.surface.device,
            self.frame_idx,
            swapchain_image.index,
        )?;

        Ok(Some(swapchain_image))
    }

    /// Displays the given swapchain image using passes specified in `graph`, if possible.
    #[profiling::function]
    pub fn present_image<P>(
        &mut self,
        pool: &mut P,
        graph: Graph,
        image_node: SwapchainImageNode,
        queue_index: u32,
    ) -> Result<(), GraphchainError>
    where
        P: Pool<CommandBufferInfo, CommandBuffer> + SubmissionPool,
    {
        trace!("present_image");

        let submission = graph.finalize();
        let (image_idx, image_sync_info) = {
            let image = submission.resource(image_node);

            (image.index as usize, image.sync_info())
        };
        let frame_idx = self.image_frames[image_idx];
        let frame = &mut self.frames[frame_idx];
        let rendered =
            self.strategy
                .rendered_semaphore(&frame.cmd_buf.device, frame_idx, image_idx as u32)?;

        frame.cmd_buf.begin(
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let wait_signal = image_sync_info.subresources.first().map(|_| {
            let wait = SemaphoreSubmitInfo {
                semaphore: frame.swapchain_acquired,
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                value: 0,
            };
            let signal = SemaphoreSubmitInfo {
                semaphore: rendered,
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                value: 0,
            };

            (wait, signal)
        });

        let (mut recording, wait, signal, record_remaining) =
            if let Some((wait, signal)) = wait_signal {
                (
                    submission.record(pool, &frame.cmd_buf, image_node)?,
                    wait,
                    signal,
                    true,
                )
            } else {
                warn!("uninitialized swapchain image");

                let wait = SemaphoreSubmitInfo {
                    semaphore: frame.swapchain_acquired,
                    stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                    value: 0,
                };
                let signal = SemaphoreSubmitInfo {
                    semaphore: rendered,
                    stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                    value: 0,
                };

                (
                    submission.record(pool, &frame.cmd_buf, RecordSelection::All)?,
                    wait,
                    signal,
                    false,
                )
            };

        let supports_synchronization2 = recording.cmd_buf.device.physical.vk_khr_synchronization2;

        let image = recording.resource(image_node);
        let image_sync_info = image.sync_info();
        let image_handle = image.handle;

        if log_enabled!(Level::Trace) {
            for sync_info in &image_sync_info.subresources {
                trace!(
                    "image {:?} {:?}->{:?}",
                    image,
                    sync_info.layout,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                );
            }
        }

        if supports_synchronization2 {
            for sync_info in &image_sync_info.subresources {
                Device::cmd_pipeline_barrier2(
                    &recording.cmd_buf.device,
                    recording.cmd_buf.handle,
                    &vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(
                        &vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::from_raw(
                                sync_info.stage_mask.as_raw() as u64,
                            ))
                            .src_access_mask(vk::AccessFlags2::from_raw(
                                sync_info.access_mask.as_raw() as u64,
                            ))
                            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                            .dst_access_mask(vk::AccessFlags2::empty())
                            .old_layout(sync_info.layout.unwrap_or(vk::ImageLayout::UNDEFINED))
                            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                            .src_queue_family_index(
                                sync_info
                                    .queue_family_index
                                    .unwrap_or(vk::QUEUE_FAMILY_IGNORED),
                            )
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(image_handle)
                            .subresource_range(sync_info.range),
                    )),
                );
            }
        } else {
            for sync_info in &image_sync_info.subresources {
                unsafe {
                    recording.cmd_buf.device.cmd_pipeline_barrier(
                        recording.cmd_buf.handle,
                        sync_info.stage_mask,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        slice::from_ref(
                            &vk::ImageMemoryBarrier::default()
                                .src_access_mask(sync_info.access_mask)
                                .dst_access_mask(vk::AccessFlags::empty())
                                .old_layout(sync_info.layout.unwrap_or(vk::ImageLayout::UNDEFINED))
                                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                                .src_queue_family_index(
                                    sync_info
                                        .queue_family_index
                                        .unwrap_or(vk::QUEUE_FAMILY_IGNORED),
                                )
                                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .image(image_handle)
                                .subresource_range(sync_info.range),
                        ),
                    )
                };
            }
        }

        if record_remaining {
            recording.record(RecordSelection::All)?;
        }

        recording.cmd_buf.end()?;

        let image = unsafe { recording.resource(image_node).to_detached() };

        let mut recorded = recording.finish()?;
        let waits = slice::from_ref(&wait);
        let signals = slice::from_ref(&signal);
        let submit_info = QueueSubmitInfo::queue_submit(waits, signals);
        recorded.queue_submit(&mut frame.fence, queue_index, submit_info)?;

        // Only mark presentation as pending when the driver actually queued the present.
        let present_info = PresentInfo {
            image,
            swapchain: &mut self.read_only.swapchain,
            wait_semaphores: slice::from_ref(&rendered),
        };

        let present =
            Swapchain::queue_present(self.queue_family_index, queue_index, [present_info]);
        let present_result = match &present {
            Ok(results) => Some(results.first().copied().expect("missing present result")),
            Err(err) => err.results.first().copied().flatten(),
        };

        if present_result.is_some_and(PresentResult::suboptimal) {
            self.recreate_pending = true;
        }

        if let Some(PresentResult::NotQueued(err)) = present_result {
            match err {
                PresentError::DeviceLost | PresentError::SurfaceLost => {}
                PresentError::FullScreenExclusiveModeLost | PresentError::OutOfDate => {
                    self.recreate_pending = true;
                }
            }
        }

        if present_result.is_some_and(PresentResult::queued) {
            self.last_present_queue_index = queue_index;
            self.strategy.present_image(frame_idx, image_idx as u32);
        }

        if let Err(err) = present {
            match err.source {
                PresentFailure::DeviceLost => return Err(GraphchainError::DeviceLost),
                PresentFailure::SurfaceLost => return Err(GraphchainError::SurfaceLost),
                PresentFailure::Driver(err) => return Err(GraphchainError::Driver(err)),
                PresentFailure::FullScreenExclusiveModeLost | PresentFailure::OutOfDate => {
                    self.recreate_pending = true;
                }
            }
        }

        Ok(())
    }

    /// Updates the requested information which controls the graphchain.
    ///
    /// Previously acquired swapchain images should be discarded after calling this function.
    /// The live effective [`info`](Self::info) field updates [`acquire_timeout`](GraphchainInfo::acquire_timeout)
    /// immediately. Other fields are updated lazily after the next successful acquire recreates the
    /// swapchain and reflects the runtime values that were actually selected.
    pub fn set_info(&mut self, info: impl Into<GraphchainInfo>) {
        let info = info.into();
        let swapchain_info = info.into_swapchain_info();
        let recreate = self.requested_info.height != info.height
            || self.requested_info.clipped != info.clipped
            || self.requested_info.min_image_count != info.min_image_count
            || self.requested_info.present_mode != info.present_mode
            || self.requested_info.composite_alpha != info.composite_alpha
            || self.requested_info.surface != info.surface
            || self.requested_info.width != info.width;

        self.read_only.swapchain.set_info(swapchain_info);
        self.requested_info = info;
        self.read_only.info.acquire_timeout = info.acquire_timeout;
        self.read_only.info.clipped = info.clipped;
        self.read_only.info.min_image_count = info.min_image_count;

        if recreate {
            self.recreate_pending = true;
        }
    }

    fn ensure_frame_count(&mut self) -> Result<(), GraphchainError> {
        if self.frames.len() >= self.read_only.info.frame_count {
            return Ok(());
        }

        let device = self.read_only.swapchain.surface.device.clone();
        let mut frames = take(&mut self.frames).into_vec();

        for _ in frames.len()..self.read_only.info.frame_count {
            let cmd_buf =
                CommandBuffer::create(&device, CommandBufferInfo::new(self.queue_family_index))?;
            let fence = Fence::create(&device, false)?;
            let swapchain_acquired = Device::create_semaphore(&device)?;
            Device::try_set_debug_utils_object_name(
                &device,
                swapchain_acquired,
                "graphchain acquired semaphore",
            );

            frames.push(FrameSlot {
                cmd_buf,
                fence,
                swapchain_acquired,
            });
        }

        self.frames = frames.into_boxed_slice();

        Ok(())
    }
}

impl Debug for Graphchain {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Graphchain")
    }
}

impl Drop for Graphchain {
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        let Some(frame) = self.frames.first() else {
            return;
        };
        let device = frame.cmd_buf.device.clone();

        for frame in &mut self.frames {
            if frame.fence.is_queued() && frame.fence.wait_signaled().is_err() {
                return;
            }
        }

        if self
            .strategy
            .retire_pending(&mut self.read_only.swapchain)
            .is_err()
        {
            return;
        }

        if Device::with_queue(
            &device,
            self.queue_family_index,
            self.last_present_queue_index,
            |queue| Device::queue_wait_idle(&device, queue),
        )
        .is_err()
        {
            return;
        }

        for frame in &mut self.frames {
            unsafe {
                frame
                    .cmd_buf
                    .device
                    .destroy_semaphore(frame.swapchain_acquired, None);
            }
        }

        self.strategy.destroy(&device);
    }
}

/// Describes error conditions relating to physical displays.
#[derive(Clone, Copy, Debug)]
pub enum GraphchainError {
    /// Unrecoverable device error; must destroy this device and display and start a new one.
    DeviceLost,

    /// The surface was lost and must be recreated.
    SurfaceLost,

    /// Recoverable driver error.
    Driver(DriverError),
}

impl Error for GraphchainError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Driver(err) => Some(err),
            Self::DeviceLost | Self::SurfaceLost => None,
        }
    }
}

impl From<()> for GraphchainError {
    fn from(_: ()) -> Self {
        Self::DeviceLost
    }
}

impl From<DriverError> for GraphchainError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<SwapchainError> for GraphchainError {
    fn from(err: SwapchainError) -> Self {
        match err {
            SwapchainError::DeviceLost => Self::DeviceLost,
            SwapchainError::Driver(err) => Self::Driver(err),
            SwapchainError::FullScreenExclusiveModeLost
            | SwapchainError::NotReady
            | SwapchainError::Timeout => Self::Driver(DriverError::Unsupported),
            SwapchainError::SurfaceLost => Self::SurfaceLost,
        }
    }
}

impl std::fmt::Display for GraphchainError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceLost => f.write_str("device lost"),
            Self::SurfaceLost => f.write_str("surface lost"),
            Self::Driver(err) => std::fmt::Display::fmt(err, f),
        }
    }
}

/// Information used to create a [`Graphchain`] instance.
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct GraphchainInfo {
    /// Timeout in nanoseconds used when acquiring the next image.
    ///
    /// Do not use `u64::MAX` on surfaces where acquire forward progress cannot be guaranteed.
    #[builder(default = "u64::MAX")]
    pub acquire_timeout: u64,

    /// Alpha compositing mode used by the presentation engine.
    #[builder(default = vk::CompositeAlphaFlagsKHR::OPAQUE)]
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,

    /// Whether to clip pixels obscured by other windows on the native surface.
    #[builder(default = "true")]
    pub clipped: bool,

    /// The requested frame capacity used for in-flight graphchain work.
    ///
    /// The effective live capacity may be raised at runtime to match the swapchain image count.
    /// That resolved value is reflected through [`EffectiveGraphchainInfo`].
    #[builder(default = "4")]
    pub frame_capacity: usize,

    /// The initial height of the surface.
    #[builder(default)]
    pub height: u32,

    /// The minimum number of presentable images that the application needs.
    #[builder(default = "2")]
    pub min_image_count: u32,

    /// `vk::PresentModeKHR` determines timing and queueing with which frames are displayed.
    #[builder(default = vk::PresentModeKHR::MAILBOX)]
    pub present_mode: vk::PresentModeKHR,

    /// The format and color space of the surface.
    #[builder(default)]
    pub surface: vk::SurfaceFormatKHR,

    /// The initial width of the surface.
    #[builder(default)]
    pub width: u32,
}

/// Effective runtime information for a live [`Graphchain`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EffectiveGraphchainInfo {
    /// Timeout in nanoseconds used when acquiring the next image.
    pub acquire_timeout: u64,

    /// Active alpha compositing mode used by the presentation engine.
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,

    /// Whether pixels obscured by other windows on the native surface are clipped.
    pub clipped: bool,

    frame_capacity: usize,

    /// The effective number of tracked frame slots.
    pub frame_count: usize,

    /// The live height of the surface.
    pub height: u32,

    /// The live swapchain image count.
    pub image_count: usize,

    min_image_count: u32,

    /// `vk::PresentModeKHR` determines timing and queueing with which frames are displayed.
    pub present_mode: vk::PresentModeKHR,

    /// The format and color space of the surface.
    pub surface: vk::SurfaceFormatKHR,

    /// The live width of the surface.
    pub width: u32,
}

impl GraphchainInfo {
    /// Specifies a default graphchain with the given `width`, `height` and `format` values.
    #[inline(always)]
    pub fn new(width: u32, height: u32, surface: vk::SurfaceFormatKHR) -> GraphchainInfo {
        Self {
            acquire_timeout: u64::MAX,
            clipped: true,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            frame_capacity: 4,
            height,
            min_image_count: 2,
            present_mode: vk::PresentModeKHR::MAILBOX,
            surface,
            width,
        }
    }

    /// Creates a default `GraphchainInfoBuilder`.
    pub fn builder() -> GraphchainInfoBuilder {
        Default::default()
    }

    /// Converts a `GraphchainInfo` into a `GraphchainInfoBuilder`.
    pub fn into_builder(self) -> GraphchainInfoBuilder {
        GraphchainInfoBuilder {
            acquire_timeout: Some(self.acquire_timeout),
            clipped: Some(self.clipped),
            composite_alpha: Some(self.composite_alpha),
            frame_capacity: Some(self.frame_capacity),
            height: Some(self.height),
            min_image_count: Some(self.min_image_count),
            present_mode: Some(self.present_mode),
            surface: Some(self.surface),
            width: Some(self.width),
        }
    }

    /// Converts this graphchain info into low-level swapchain info.
    pub fn into_swapchain_info(self) -> SwapchainInfo {
        self.into()
    }
}

impl EffectiveGraphchainInfo {
    /// Creates a default `GraphchainInfoBuilder`.
    pub fn builder() -> GraphchainInfoBuilder {
        Default::default()
    }

    /// Converts an `EffectiveGraphchainInfo` into a `GraphchainInfoBuilder`.
    pub fn into_builder(self) -> GraphchainInfoBuilder {
        self.into_requested_info().into_builder()
    }

    /// Converts this effective runtime info back into requested graphchain info.
    pub fn into_requested_info(self) -> GraphchainInfo {
        let Self {
            acquire_timeout,
            clipped,
            composite_alpha,
            frame_capacity,
            frame_count: _,
            height,
            image_count: _,
            min_image_count,
            present_mode,
            surface,
            width,
        } = self;

        GraphchainInfo {
            acquire_timeout,
            clipped,
            composite_alpha,
            frame_capacity,
            height,
            min_image_count,
            present_mode,
            surface,
            width,
        }
    }
}

impl From<EffectiveGraphchainInfo> for GraphchainInfo {
    fn from(info: EffectiveGraphchainInfo) -> Self {
        info.into_requested_info()
    }
}

impl From<GraphchainInfoBuilder> for GraphchainInfo {
    fn from(info: GraphchainInfoBuilder) -> Self {
        info.build()
    }
}

impl From<GraphchainInfo> for SwapchainInfo {
    fn from(val: GraphchainInfo) -> Self {
        SwapchainInfoBuilder::default()
            .acquire_timeout(val.acquire_timeout)
            .clipped(val.clipped)
            .composite_alpha(val.composite_alpha)
            .height(val.height)
            .min_image_count(val.min_image_count)
            .present_mode(val.present_mode)
            .surface(val.surface)
            .width(val.width)
            .build()
    }
}

impl From<GraphchainInfoBuilder> for SwapchainInfo {
    fn from(info: GraphchainInfoBuilder) -> Self {
        info.build().into_swapchain_info()
    }
}

impl GraphchainInfoBuilder {
    /// Builds a new `GraphchainInfo`.
    #[inline(always)]
    pub fn build(self) -> GraphchainInfo {
        self.fallible_build().expect("all fields have defaults")
    }
}

enum PresentRetirementStrategy {
    PresentWait(PresentWait),
    PerImageSemaphore(PerImageSemaphore),
}

impl PresentRetirementStrategy {
    fn prepare_frame(
        &mut self,
        device: &Device,
        swapchain: &mut Swapchain,
        frame_idx: usize,
    ) -> Result<(), GraphchainError> {
        match self {
            Self::PresentWait(strategy) => strategy.prepare_frame(device, swapchain, frame_idx),
            Self::PerImageSemaphore(strategy) => {
                strategy.prepare_frame(device, swapchain, frame_idx)
            }
        }
    }

    fn acquire_image(
        &mut self,
        device: &Device,
        frame_idx: usize,
        image_idx: u32,
    ) -> Result<(), DriverError> {
        match self {
            Self::PresentWait(strategy) => strategy.acquire_image(device, frame_idx, image_idx),
            Self::PerImageSemaphore(strategy) => {
                strategy.acquire_image(device, frame_idx, image_idx)
            }
        }
    }

    fn rendered_semaphore(
        &mut self,
        device: &Device,
        frame_idx: usize,
        image_idx: u32,
    ) -> Result<vk::Semaphore, DriverError> {
        match self {
            Self::PresentWait(strategy) => {
                strategy.rendered_semaphore(device, frame_idx, image_idx)
            }
            Self::PerImageSemaphore(strategy) => {
                strategy.rendered_semaphore(device, frame_idx, image_idx)
            }
        }
    }

    fn present_image(&mut self, frame_idx: usize, image_idx: u32) {
        match self {
            Self::PresentWait(strategy) => strategy.present_image(frame_idx, image_idx),
            Self::PerImageSemaphore(strategy) => strategy.present_image(frame_idx, image_idx),
        }
    }

    fn retire_pending(&mut self, swapchain: &mut Swapchain) -> Result<(), GraphchainError> {
        match self {
            Self::PresentWait(strategy) => strategy.retire_pending(swapchain),
            Self::PerImageSemaphore(strategy) => strategy.retire_pending(swapchain),
        }
    }

    fn reset(&mut self, device: &Device) {
        match self {
            Self::PresentWait(strategy) => strategy.reset(device),
            Self::PerImageSemaphore(strategy) => strategy.reset(device),
        }
    }

    fn destroy(&mut self, device: &Device) {
        match self {
            Self::PresentWait(strategy) => strategy.destroy(device),
            Self::PerImageSemaphore(strategy) => strategy.destroy(device),
        }
    }
}

#[derive(Default)]
struct PresentWait {
    retired: Vec<vk::Semaphore>,
    rendered: Vec<Option<vk::Semaphore>>,
    pending_images: Vec<Option<u32>>,
}

impl PresentWait {
    fn prepare_frame(
        &mut self,
        _device: &Device,
        swapchain: &mut Swapchain,
        frame_idx: usize,
    ) -> Result<(), GraphchainError> {
        while self.pending_images.len() <= frame_idx {
            self.pending_images.push(None);
        }

        if let Some(image_idx) = self.pending_images[frame_idx].take() {
            unsafe {
                match swapchain.wait_for_present_image(image_idx, u64::MAX) {
                    Ok(()) => {}
                    Err(SwapchainError::DeviceLost) => {
                        return Err(GraphchainError::DeviceLost);
                    }
                    Err(SwapchainError::Driver(err)) => {
                        return Err(GraphchainError::Driver(err));
                    }
                    Err(SwapchainError::FullScreenExclusiveModeLost) => {}
                    Err(SwapchainError::NotReady | SwapchainError::Timeout) => {
                        return Err(GraphchainError::Driver(DriverError::Unsupported));
                    }
                    Err(SwapchainError::SurfaceLost) => {
                        return Err(GraphchainError::SurfaceLost);
                    }
                }
            }
        }

        Ok(())
    }

    fn acquire_image(
        &mut self,
        device: &Device,
        _frame_idx: usize,
        image_idx: u32,
    ) -> Result<(), DriverError> {
        self.ensure_rendered(device, image_idx).map(|_| ())
    }

    fn rendered_semaphore(
        &mut self,
        device: &Device,
        _frame_idx: usize,
        image_idx: u32,
    ) -> Result<vk::Semaphore, DriverError> {
        self.ensure_rendered(device, image_idx)
    }

    fn ensure_rendered(
        &mut self,
        device: &Device,
        image_idx: u32,
    ) -> Result<vk::Semaphore, DriverError> {
        while self.rendered.len() <= image_idx as usize {
            self.rendered.push(None);
        }

        if self.rendered[image_idx as usize].is_none() {
            let semaphore = Device::create_semaphore(device)?;
            Device::try_set_debug_utils_object_name(
                device,
                semaphore,
                "graphchain present-wait per-image rendered semaphore",
            );

            self.rendered[image_idx as usize] = Some(semaphore);
        }

        Ok(self.rendered[image_idx as usize].expect("missing rendered semaphore"))
    }

    fn present_image(&mut self, frame_idx: usize, image_idx: u32) {
        while self.pending_images.len() <= frame_idx {
            self.pending_images.push(None);
        }

        self.pending_images[frame_idx] = Some(image_idx);
    }

    fn retire_pending(&mut self, swapchain: &mut Swapchain) -> Result<(), GraphchainError> {
        for image_idx in self.pending_images.drain(..).flatten() {
            unsafe {
                match swapchain.wait_for_present_image(image_idx, u64::MAX) {
                    Ok(()) | Err(SwapchainError::FullScreenExclusiveModeLost) => {}
                    Err(SwapchainError::DeviceLost) => return Err(GraphchainError::DeviceLost),
                    Err(SwapchainError::Driver(err)) => return Err(GraphchainError::Driver(err)),
                    Err(SwapchainError::NotReady | SwapchainError::Timeout) => {
                        return Err(GraphchainError::Driver(DriverError::Unsupported));
                    }
                    Err(SwapchainError::SurfaceLost) => return Err(GraphchainError::SurfaceLost),
                }
            }
        }

        Ok(())
    }

    fn reset(&mut self, device: &Device) {
        self.destroy_retired(device);
        self.pending_images.clear();
        self.retired.extend(self.rendered.drain(..).flatten());
    }

    fn destroy(&mut self, device: &Device) {
        self.destroy_retired(device);

        for semaphore in self
            .rendered
            .drain(..)
            .flatten()
            .chain(self.retired.drain(..))
        {
            unsafe {
                device.destroy_semaphore(semaphore, None);
            }
        }
    }

    fn destroy_retired(&mut self, device: &Device) {
        for semaphore in self.retired.drain(..) {
            unsafe {
                device.destroy_semaphore(semaphore, None);
            }
        }
    }
}

#[derive(Default)]
struct PerImageSemaphore {
    retired: Vec<vk::Semaphore>,
    rendered: Vec<Option<vk::Semaphore>>,
}

impl PerImageSemaphore {
    fn prepare_frame(
        &mut self,
        _device: &Device,
        _swapchain: &mut Swapchain,
        _frame_idx: usize,
    ) -> Result<(), GraphchainError> {
        Ok(())
    }

    fn acquire_image(
        &mut self,
        device: &Device,
        _frame_idx: usize,
        image_idx: u32,
    ) -> Result<(), DriverError> {
        self.ensure_rendered(device, image_idx).map(|_| ())
    }

    fn rendered_semaphore(
        &mut self,
        device: &Device,
        _frame_idx: usize,
        image_idx: u32,
    ) -> Result<vk::Semaphore, DriverError> {
        self.ensure_rendered(device, image_idx)
    }

    fn ensure_rendered(
        &mut self,
        device: &Device,
        image_idx: u32,
    ) -> Result<vk::Semaphore, DriverError> {
        while self.rendered.len() <= image_idx as usize {
            self.rendered.push(None);
        }

        if self.rendered[image_idx as usize].is_none() {
            let semaphore = Device::create_semaphore(device)?;
            Device::try_set_debug_utils_object_name(
                device,
                semaphore,
                "graphchain per-image rendered semaphore",
            );

            self.rendered[image_idx as usize] = Some(semaphore);
        }

        Ok(self.rendered[image_idx as usize].expect("missing rendered semaphore"))
    }

    fn present_image(&mut self, _frame_idx: usize, _image_idx: u32) {}

    fn retire_pending(&mut self, _swapchain: &mut Swapchain) -> Result<(), GraphchainError> {
        Ok(())
    }

    fn reset(&mut self, device: &Device) {
        self.destroy_retired(device);
        self.retired.extend(self.rendered.drain(..).flatten());
    }

    fn destroy(&mut self, device: &Device) {
        self.destroy_retired(device);

        for semaphore in self
            .rendered
            .drain(..)
            .flatten()
            .chain(self.retired.drain(..))
        {
            unsafe {
                device.destroy_semaphore(semaphore, None);
            }
        }
    }

    fn destroy_retired(&mut self, device: &Device) {
        for semaphore in self.retired.drain(..) {
            unsafe {
                device.destroy_semaphore(semaphore, None);
            }
        }
    }
}

struct FrameSlot {
    cmd_buf: CommandBuffer,
    fence: Fence,
    swapchain_acquired: vk::Semaphore,
}

#[cfg(test)]
mod test {
    use super::*;

    type Info = GraphchainInfo;
    type Builder = GraphchainInfoBuilder;
    type Effective = EffectiveGraphchainInfo;

    #[test]
    fn graphchain_info_round_trips_through_builder() {
        let info = Info {
            acquire_timeout: 42,
            clipped: false,
            composite_alpha: vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            frame_capacity: 6,
            height: 123,
            min_image_count: 3,
            present_mode: vk::PresentModeKHR::FIFO,
            surface: vk::SurfaceFormatKHR::default()
                .format(vk::Format::B8G8R8A8_UNORM)
                .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR),
            width: 456,
        };

        assert_eq!(info, info.into_builder().build());
    }

    #[test]
    fn graphchain_info_builder_defaults() {
        assert_eq!(
            Builder::default().build(),
            Info {
                acquire_timeout: u64::MAX,
                clipped: true,
                composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
                frame_capacity: 4,
                height: 0,
                min_image_count: 2,
                present_mode: vk::PresentModeKHR::MAILBOX,
                surface: vk::SurfaceFormatKHR::default(),
                width: 0,
            }
        );
    }

    #[test]
    fn effective_graphchain_info_into_requested_info() {
        let info = Effective {
            acquire_timeout: 42,
            clipped: false,
            composite_alpha: vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            frame_capacity: 6,
            frame_count: 8,
            height: 123,
            image_count: 7,
            min_image_count: 3,
            present_mode: vk::PresentModeKHR::FIFO,
            surface: vk::SurfaceFormatKHR::default()
                .format(vk::Format::B8G8R8A8_UNORM)
                .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR),
            width: 456,
        };

        assert_eq!(
            info.into_requested_info(),
            Info {
                acquire_timeout: 42,
                clipped: false,
                composite_alpha: vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
                frame_capacity: 6,
                height: 123,
                min_image_count: 3,
                present_mode: vk::PresentModeKHR::FIFO,
                surface: vk::SurfaceFormatKHR::default()
                    .format(vk::Format::B8G8R8A8_UNORM)
                    .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR),
                width: 456,
            }
        );
    }

    // #[test]
    // fn effective_graphchain_info_from_requested_info() {
    //     let info = Info {
    //         acquire_timeout: 42,
    //         composite_alpha: vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
    //         frame_capacity: 6,
    //         height: 123,
    //         min_image_count: 3,
    //         present_mode: vk::PresentModeKHR::FIFO,
    //         surface: vk::SurfaceFormatKHR::default()
    //             .format(vk::Format::B8G8R8A8_UNORM)
    //             .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR),
    //         width: 456,
    //     };

    //     assert_eq!(
    //         Effective::from(info),
    //         Effective {
    //             acquire_timeout: 42,
    //             composite_alpha: vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
    //             frame_capacity: 6,
    //             frame_count: 6,
    //             height: 123,
    //             image_count: 0,
    //             present_mode: vk::PresentModeKHR::FIFO,
    //             surface: vk::SurfaceFormatKHR::default()
    //                 .format(vk::Format::B8G8R8A8_UNORM)
    //                 .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR),
    //             width: 456,
    //         }
    //     );
    // }

    // #[test]
    // fn effective_graphchain_info_has_only_runtime_fields() {
    //     let info = Effective {
    //         acquire_timeout: 42,
    //         composite_alpha: vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
    //         frame_capacity: 6,
    //         frame_count: 8,
    //         height: 123,
    //         image_count: 7,
    //         present_mode: vk::PresentModeKHR::FIFO,
    //         surface: vk::SurfaceFormatKHR::default()
    //             .format(vk::Format::B8G8R8A8_UNORM)
    //             .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR),
    //         width: 456,
    //     };

    //     assert_eq!(info.frame_count, 8);
    //     assert_eq!(info.image_count, 7);
    // }

    // #[test]
    // fn effective_graphchain_info_round_trips_through_builder() {
    //     let info = Effective {
    //         acquire_timeout: 42,
    //         composite_alpha: vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
    //         frame_capacity: 6,
    //         frame_count: 8,
    //         height: 123,
    //         image_count: 7,
    //         present_mode: vk::PresentModeKHR::FIFO,
    //         surface: vk::SurfaceFormatKHR::default()
    //             .format(vk::Format::B8G8R8A8_UNORM)
    //             .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR),
    //         width: 456,
    //     };

    //     assert_eq!(info, info.into_builder().build());
    // }

    // #[test]
    // fn effective_graphchain_info_builder_defaults() {
    //     assert_eq!(
    //         EffectiveBuilder::default().build(),
    //         Effective {
    //             acquire_timeout: u64::MAX,
    //             composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
    //             frame_capacity: 4,
    //             frame_count: 4,
    //             height: 0,
    //             image_count: 0,
    //             present_mode: vk::PresentModeKHR::MAILBOX,
    //             surface: vk::SurfaceFormatKHR::default(),
    //             width: 0,
    //         }
    //     );
    // }

    // #[test]
    // fn effective_graphchain_info_is_not_requested_info() {
    //     let info = Info {
    //         acquire_timeout: 42,
    //         composite_alpha: vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
    //         frame_capacity: 6,
    //         height: 123,
    //         min_image_count: 3,
    //         present_mode: vk::PresentModeKHR::FIFO,
    //         surface: vk::SurfaceFormatKHR::default()
    //             .format(vk::Format::B8G8R8A8_UNORM)
    //             .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR),
    //         width: 456,
    //     };

    //     assert_eq!(Effective::from(info).image_count, 0);
    // }
}
