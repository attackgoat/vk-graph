//! Native window presentation types.

use {
    super::{
        DriverError, Surface,
        device::Device,
        fence::Fence,
        image::{Image, ImageInfo},
    },
    ash::vk::{self, Handle as _},
    derive_builder::Builder,
    log::{Level, debug, error, info, log, trace, warn},
    std::{
        cell::RefCell,
        error::Error,
        fmt::{Display, Formatter},
        iter::FusedIterator,
        marker::PhantomData,
        mem::replace,
        ops::{Deref, Index},
        slice,
        thread::panicking,
    },
};

#[cfg(feature = "checked")]
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug)]
enum AcquireImageError {
    OutOfDate,
    Swapchain(SwapchainError),
}

impl From<SwapchainError> for AcquireImageError {
    fn from(err: SwapchainError) -> Self {
        Self::Swapchain(err)
    }
}

/// Effective runtime information for a live [`Swapchain`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EffectiveSwapchainInfo {
    /// Timeout in nanoseconds used when acquiring the next image.
    pub acquire_timeout: u64,

    /// Active alpha compositing mode used by the presentation engine.
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,

    /// Whether pixels obscured by other windows on the native surface are clipped.
    pub clipped: bool,

    /// The live height of the surface.
    pub height: u32,

    /// The active number of array layers for each swapchain image.
    pub image_array_layers: u32,

    /// The number of images currently owned by the live swapchain.
    pub image_count: u32,

    /// The active sharing mode used for swapchain images.
    pub image_sharing_mode: vk::SharingMode,

    min_image_count: u32,

    /// The active transform applied to presented images.
    pub pre_transform: vk::SurfaceTransformFlagsKHR,

    /// The active presentation mode.
    pub present_mode: vk::PresentModeKHR,

    /// Timeout in nanoseconds used when presenting an image.
    pub present_timeout: u64,

    /// The active format and color space of the surface.
    pub surface: vk::SurfaceFormatKHR,

    /// The live width of the surface.
    pub width: u32,
}

impl EffectiveSwapchainInfo {
    /// Creates a default `SwapchainInfoBuilder`.
    pub fn builder() -> SwapchainInfoBuilder {
        Default::default()
    }

    /// Converts an `EffectiveSwapchainInfo` into an `EffectiveSwapchainInfoBuilder`.
    pub fn into_builder(self) -> SwapchainInfoBuilder {
        let Self {
            acquire_timeout,
            clipped,
            composite_alpha,
            height,
            image_array_layers,
            image_count: _,
            image_sharing_mode,
            min_image_count,
            pre_transform,
            present_timeout,
            present_mode,
            surface,
            width,
        } = self;

        SwapchainInfo {
            acquire_timeout,
            clipped,
            composite_alpha,
            height,
            image_array_layers,
            image_sharing_mode,
            min_image_count,
            pre_transform,
            present_mode,
            present_timeout,
            surface,
            width,
        }
        .into_builder()
    }

    /// Converts this effective runtime info back into requested swapchain info.
    pub fn into_requested_info(self) -> SwapchainInfo {
        SwapchainInfo::from(self)
    }
}

#[derive(Debug)]
struct Live;

/// Describes one swapchain image to present.
#[derive(Debug)]
pub struct PresentInfo<'a> {
    /// The acquired image to present.
    pub image: SwapchainImage,

    /// The swapchain that acquired `image`.
    pub swapchain: &'a mut Swapchain,

    /// Semaphores to wait on before presentation.
    ///
    /// Vulkan applies the flattened semaphore list to the whole present batch, not just this image.
    pub wait_semaphores: &'a [vk::Semaphore],
}

/// Result of one image in a queue presentation batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresentResult {
    /// The image was queued and the swapchain remained optimal.
    QueuedOptimal,

    /// The image was queued, but the swapchain should be recreated soon.
    QueuedSuboptimal,

    /// The image was not queued for presentation.
    NotQueued(PresentError),
}

impl PresentResult {
    /// Returns `true` if this image was queued for presentation.
    pub fn queued(self) -> bool {
        matches!(self, Self::QueuedOptimal | Self::QueuedSuboptimal)
    }

    /// Returns `true` if this present reported suboptimal state.
    pub fn suboptimal(self) -> bool {
        matches!(self, Self::QueuedSuboptimal)
    }
}

/// Ordered results for a queue presentation batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PresentBatch {
    results: Box<[PresentResult]>,
}

impl PresentBatch {
    fn new(results: Box<[PresentResult]>) -> Self {
        Self { results }
    }

    /// Returns the per-input results in submission order.
    pub fn as_slice(&self) -> &[PresentResult] {
        &self.results
    }

    /// Returns `true` if there are no results.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Iterates over per-input results in submission order.
    pub fn iter(&self) -> slice::Iter<'_, PresentResult> {
        self.results.iter()
    }

    /// Returns the number of results.
    pub fn len(&self) -> usize {
        self.results.len()
    }
}

impl AsRef<[PresentResult]> for PresentBatch {
    fn as_ref(&self) -> &[PresentResult] {
        self.as_slice()
    }
}

impl Deref for PresentBatch {
    type Target = [PresentResult];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl Index<usize> for PresentBatch {
    type Output = PresentResult;

    fn index(&self, index: usize) -> &Self::Output {
        &self.results[index]
    }
}

impl IntoIterator for PresentBatch {
    type IntoIter = PresentBatchIntoIter;
    type Item = PresentResult;

    fn into_iter(self) -> Self::IntoIter {
        PresentBatchIntoIter {
            inner: self.results.into_vec().into_iter(),
        }
    }
}

impl<'a> IntoIterator for &'a PresentBatch {
    type IntoIter = slice::Iter<'a, PresentResult>;
    type Item = &'a PresentResult;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Error returned by a presentation batch.
#[derive(Clone, Debug)]
pub struct PresentBatchError {
    /// Per-input results in the same order as the submitted [`PresentInfo`] values.
    ///
    /// Entries are `None` when presentation did not run far enough to determine that input's
    /// result.
    pub results: Box<[Option<PresentResult>]>,

    /// The batch-level failure.
    pub source: PresentFailure,
}

impl PresentBatchError {
    fn new(source: PresentFailure, result_count: usize) -> Self {
        Self {
            results: vec![None; result_count].into_boxed_slice(),
            source,
        }
    }

    fn with_results(source: PresentFailure, results: &[PresentResult]) -> Self {
        Self {
            results: results.iter().copied().map(Some).collect(),
            source,
        }
    }
}

impl Display for PresentBatchError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "unable to present batch: {}", self.source)
    }
}

impl Error for PresentBatchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.source {
            PresentFailure::Driver(err) => Some(err),
            PresentFailure::DeviceLost
            | PresentFailure::FullScreenExclusiveModeLost
            | PresentFailure::OutOfDate
            | PresentFailure::SurfaceLost => None,
        }
    }
}

/// Owning iterator over [`PresentBatch`] results.
#[derive(Clone, Debug)]
pub struct PresentBatchIntoIter {
    inner: std::vec::IntoIter<PresentResult>,
}

impl DoubleEndedIterator for PresentBatchIntoIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl ExactSizeIterator for PresentBatchIntoIter {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl FusedIterator for PresentBatchIntoIter {}

impl Iterator for PresentBatchIntoIter {
    type Item = PresentResult;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// Reason an image was not queued for presentation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresentError {
    /// The device was lost.
    DeviceLost,

    /// Exclusive full-screen ownership was lost.
    FullScreenExclusiveModeLost,

    /// The swapchain is out of date and should be recreated.
    OutOfDate,

    /// The surface was lost and must be recreated.
    SurfaceLost,
}

/// Batch-level presentation failure.
#[derive(Clone, Copy, Debug)]
pub enum PresentFailure {
    /// The device was lost.
    DeviceLost,

    /// The driver rejected the request before item-level present results were available.
    Driver(DriverError),

    /// Exclusive full-screen ownership was lost.
    FullScreenExclusiveModeLost,

    /// The swapchain is out of date and should be recreated.
    OutOfDate,

    /// The surface was lost and must be recreated.
    SurfaceLost,
}

impl Display for PresentFailure {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceLost => f.write_str("device lost"),
            Self::Driver(err) => write!(f, "driver error: {err}"),
            Self::FullScreenExclusiveModeLost => f.write_str("full-screen exclusive mode lost"),
            Self::OutOfDate => f.write_str("out of date"),
            Self::SurfaceLost => f.write_str("surface lost"),
        }
    }
}

impl Error for PresentFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Driver(err) => Some(err),
            Self::DeviceLost
            | Self::FullScreenExclusiveModeLost
            | Self::OutOfDate
            | Self::SurfaceLost => None,
        }
    }
}

#[derive(Debug)]
struct PresentRetirement {
    next_present_id: u64,
    image_present_ids: Box<[Option<u64>]>,
    present_pending: bool,
    queue: Option<QueueAddress>,
    use_present_wait: bool,
}

impl PresentRetirement {
    fn new(image_count: usize, use_present_wait: bool) -> Self {
        Self {
            // Present ID `0` means "no associated present ID".
            next_present_id: 1,
            image_present_ids: vec![None; image_count].into_boxed_slice(),
            present_pending: false,
            queue: None,
            use_present_wait,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QueueAddress {
    family_index: u32,
    queue_index: u32,
}

#[derive(Debug, Default)]
struct QueueFamily {
    cmd_pool: vk::CommandPool,
    queue_signals: Box<[QueueSignal]>,
}

#[derive(Debug)]
struct QueueSignals {
    device: Device,
    families: Box<[QueueFamily]>,
}

impl QueueSignals {
    fn create(surface: &Surface) -> Result<Self, DriverError> {
        struct PartialQueueSignals<'a> {
            device: &'a Device,
            families: Vec<QueueFamily>,
        }

        impl Drop for PartialQueueSignals<'_> {
            fn drop(&mut self) {
                let mut queue_signals = QueueSignals {
                    device: self.device.clone(),
                    families: std::mem::take(&mut self.families).into_boxed_slice(),
                };

                queue_signals.destroy();
            }
        }

        struct PartialQueueFamily<'a> {
            device: &'a Device,
            cmd_pool: vk::CommandPool,
            queue_signals: Vec<QueueSignal>,
        }

        impl PartialQueueFamily<'_> {
            fn finish(mut self) -> QueueFamily {
                let family = QueueFamily {
                    cmd_pool: self.cmd_pool,
                    queue_signals: std::mem::take(&mut self.queue_signals).into_boxed_slice(),
                };
                self.cmd_pool = vk::CommandPool::null();

                family
            }
        }

        impl Drop for PartialQueueFamily<'_> {
            fn drop(&mut self) {
                if self.cmd_pool.is_null() {
                    return;
                }

                thread_local! {
                    static CMD_BUFS: RefCell<Vec<vk::CommandBuffer>> = Default::default();
                }

                CMD_BUFS.with_borrow_mut(|tls| {
                    tls.clear();
                    tls.extend(self.queue_signals.iter().map(|queue| queue.cmd_buf));

                    unsafe {
                        self.device.free_command_buffers(self.cmd_pool, tls);
                        self.device.destroy_command_pool(self.cmd_pool, None);
                    }
                });
            }
        }

        let device = &surface.device;
        let all_queue_families = &device.physical.queue_families;
        let mut partial = PartialQueueSignals {
            device,
            families: Vec::with_capacity(all_queue_families.len()),
        };

        for (idx, queue_family) in all_queue_families.iter().enumerate() {
            let cmd_pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(idx as _);
            if !surface.physical_device_support(cmd_pool_info.queue_family_index)? {
                partial.families.push(Default::default());

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
            let mut partial_family = PartialQueueFamily {
                device,
                cmd_pool,
                queue_signals: Vec::with_capacity(queue_family.queue_count as _),
            };

            for cmd_buf in cmd_bufs {
                Device::begin_command_buffer(
                    device,
                    cmd_buf,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE),
                )?;
                Device::end_command_buffer(device, cmd_buf)?;

                let fence = Fence::create(device, false)?;
                partial_family
                    .queue_signals
                    .push(QueueSignal { cmd_buf, fence });
            }

            partial.families.push(partial_family.finish());
        }

        let families = std::mem::take(&mut partial.families);
        std::mem::forget(partial);

        Ok(Self {
            device: device.clone(),
            families: families.into_boxed_slice(),
        })
    }

    fn destroy(&mut self) {
        // Intentionally idempotent: callers may destroy early, then `Drop` will call this again.
        let families = std::mem::take(&mut self.families);

        for queue_family in &families {
            if queue_family.cmd_pool.is_null() {
                continue;
            }

            thread_local! {
                static CMD_BUFS: RefCell<Vec<vk::CommandBuffer>> = Default::default();
            }

            CMD_BUFS.with_borrow_mut(|tls| {
                tls.clear();
                tls.extend(queue_family.queue_signals.iter().map(|queue| queue.cmd_buf));
                unsafe {
                    self.device.free_command_buffers(queue_family.cmd_pool, tls);
                }
            });

            unsafe {
                self.device
                    .destroy_command_pool(queue_family.cmd_pool, None);
            }
        }
    }

    fn reset_all(&mut self) -> Result<(), DriverError> {
        for queue_family in &mut self.families {
            for queue in &mut queue_family.queue_signals {
                queue.fence.reset()?;
            }
        }

        Ok(())
    }

    fn submit(
        &mut self,
        device: &Device,
        queue: vk::Queue,
        queue_family_index: u32,
        queue_index: u32,
    ) -> Result<(), SwapchainError> {
        let queue_signal =
            &mut self.families[queue_family_index as usize].queue_signals[queue_index as usize];

        if queue_signal.fence.is_queued() {
            queue_signal
                .fence
                .wait_signaled()
                .map_err(|_| SwapchainError::DeviceLost)?
                .reset()
                .map_err(|_| SwapchainError::DeviceLost)?;
        }

        Device::queue_submit(
            device,
            queue,
            slice::from_ref(
                &vk::SubmitInfo::default().command_buffers(slice::from_ref(&queue_signal.cmd_buf)),
            ),
            queue_signal.fence.handle,
        )
        .map_err(|_| SwapchainError::DeviceLost)?;

        queue_signal.fence.mark_queued();

        Ok(())
    }

    fn wait_for_and_reset_all(&mut self, device: &Device) -> Result<(), DriverError> {
        let mut fences = self.families.iter().flat_map(|queue_family| {
            queue_family
                .queue_signals
                .iter()
                .filter_map(|queue| queue.fence.is_queued().then_some(queue.fence.handle))
        });

        let Some(first) = fences.next() else {
            return Ok(());
        };

        let Some(second) = fences.next() else {
            Device::wait_for_fences(device, slice::from_ref(&first))?;
            self.reset_all()?;

            return Ok(());
        };

        thread_local! {
            static FENCES: RefCell<Vec<vk::Fence>> = Default::default();
        }

        FENCES.with_borrow_mut(|tls| {
            tls.clear();
            tls.extend([first, second]);
            tls.extend(fences);

            Device::wait_for_fences(device, tls)
        })?;

        self.reset_all()?;

        Ok(())
    }
}

impl Drop for QueueSignals {
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        self.destroy();
    }
}

#[derive(Debug)]
struct QueueSignal {
    cmd_buf: vk::CommandBuffer,
    fence: Fence,
}

#[derive(Debug)]
struct Retired;

/// Provides the ability to present rendering results to a [`Surface`].
///
/// See [`VkSwapchainKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSwapchainKHR.html).
#[derive(Debug)]
#[read_only::cast]
pub struct Swapchain {
    /// The native Vulkan resource handle of this swapchain.
    ///
    /// _Note:_ This field is read-only.
    pub handle: vk::SwapchainKHR,

    /// Effective runtime information for the live swapchain.
    ///
    /// Calls to [`set_info`](Self::set_info) update the requested configuration. This field is
    /// updated lazily after the next successful acquire recreates the swapchain. The timeout fields
    /// are updated immediately.
    ///
    /// _Note:_ This field is read-only.
    pub info: EffectiveSwapchainInfo,

    live: SwapchainState<Live>,
    queue_signals: QueueSignals,

    /// Whether swapchain recreation has been requested but not completed yet.
    pub recreate_pending: bool,

    requested_info: SwapchainInfo,
    retired: Vec<SwapchainState<Retired>>,

    /// The surface which supports this swapchain.
    ///
    /// _Note:_ This field is read-only.
    pub surface: Surface,

    #[cfg(feature = "checked")]
    swapchain_id: SwapchainId,
}

impl Swapchain {
    /// Creates a [`vk::SwapchainKHR`] object.
    #[profiling::function]
    pub fn create(surface: Surface, info: impl Into<SwapchainInfo>) -> Result<Self, DriverError> {
        let mut requested_info = info.into();
        let queue_signals = QueueSignals::create(&surface)?;
        let (live, info) = SwapchainState::create(&surface, &mut requested_info, None)?;
        let handle = live.handle();

        Ok(Self {
            handle,
            info,
            queue_signals,
            recreate_pending: false,
            requested_info,
            retired: Vec::new(),
            surface,

            #[cfg(feature = "checked")]
            swapchain_id: live.swapchain_id,

            live,
        })
    }

    /// Acquires the next available swapchain image for rendering.
    #[profiling::function]
    pub fn acquire_next_image(
        &mut self,
        acquired: vk::Semaphore,
    ) -> Result<SwapchainImage, SwapchainError> {
        if self.recreate_pending {
            self.recreate()?;
        }

        let image = match self.live.acquire_next_image(&self.surface.device, acquired) {
            Ok(acquired) => acquired,
            Err(AcquireImageError::OutOfDate) => {
                self.recreate_pending = true;

                return Err(SwapchainError::NotReady);
            }
            Err(AcquireImageError::Swapchain(
                err @ SwapchainError::FullScreenExclusiveModeLost,
            )) => {
                self.recreate_pending = true;

                return Err(err);
            }
            Err(AcquireImageError::Swapchain(err)) => return Err(err),
        };

        self.recreate_pending = image.suboptimal;

        Ok(image)
    }

    fn clamp_min_image_count(min_image_count: u32, surface: vk::SurfaceCapabilitiesKHR) -> u32 {
        let min_image_count = min_image_count.max(surface.min_image_count);

        if surface.max_image_count == 0 {
            return min_image_count;
        }

        min_image_count.min(surface.max_image_count)
    }

    fn destroy_retired(&mut self) -> Result<(), SwapchainError> {
        let idx = 0;
        let mut res = None;

        while idx < self.retired.len() {
            if let Err(err) = self
                .retired
                .swap_remove(idx)
                .destroy_when_idle(&self.surface.device)
                && !matches!(res, Some(SwapchainError::Driver(DriverError::Unsupported)))
                && res
                    .replace(err)
                    .map(|old| old.different(err))
                    .unwrap_or_default()
            {
                warn!("unsupported multi-error: {err:?}");

                res = Some(SwapchainError::Driver(DriverError::Unsupported))
            }
        }

        res.map(Result::Err).unwrap_or(Ok(()))
    }

    fn destroy_retired_on_drop(&mut self) {
        for retired in &mut self.retired {
            if let Err(err) =
                retired.wait_until_retired(&self.surface.device, retired.present_timeout)
            {
                // Drop is best-effort shutdown. Preserve normal retryable semantics outside Drop,
                // but do not silently lose retired Vulkan handles when the public owner is going away.
                warn!("unable to wait for retired swapchain presents: {err:?}");
            }

            retired.handle.destroy();
        }

        self.retired.clear();
    }

    fn map_present_error(err: vk::Result) -> PresentError {
        match err {
            vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT => {
                debug!("unable to present: {}", err);

                PresentError::FullScreenExclusiveModeLost
            }
            vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => {
                debug!("unable to present: {}", err);

                PresentError::OutOfDate
            }
            vk::Result::ERROR_DEVICE_LOST => {
                info!("device lost");

                PresentError::DeviceLost
            }
            vk::Result::ERROR_SURFACE_LOST_KHR => {
                info!("surface lost");

                PresentError::SurfaceLost
            }
            err => {
                warn!("unable to present: {err}");

                PresentError::DeviceLost
            }
        }
    }

    fn present_failure_from_present_error(err: PresentError) -> PresentFailure {
        match err {
            PresentError::DeviceLost => PresentFailure::DeviceLost,
            PresentError::FullScreenExclusiveModeLost => {
                PresentFailure::FullScreenExclusiveModeLost
            }
            PresentError::OutOfDate => PresentFailure::OutOfDate,
            PresentError::SurfaceLost => PresentFailure::SurfaceLost,
        }
    }

    fn present_failure_from_swapchain_error(err: SwapchainError) -> PresentFailure {
        match err {
            SwapchainError::Driver(err) => PresentFailure::Driver(err),
            SwapchainError::DeviceLost => PresentFailure::DeviceLost,
            SwapchainError::FullScreenExclusiveModeLost => {
                PresentFailure::FullScreenExclusiveModeLost
            }
            SwapchainError::NotReady | SwapchainError::Timeout => {
                // TODO: Better errors
                PresentFailure::Driver(DriverError::Unsupported)
            }
            SwapchainError::SurfaceLost => PresentFailure::SurfaceLost,
        }
    }

    fn present_failure_requires_recreate(err: PresentFailure) -> bool {
        matches!(
            err,
            PresentFailure::FullScreenExclusiveModeLost | PresentFailure::OutOfDate,
        )
    }

    fn present_result(
        result: vk::Result,
        aggregate_suboptimal: bool,
        fallback_error: Option<PresentError>,
    ) -> PresentResult {
        match result {
            vk::Result::SUCCESS if aggregate_suboptimal => PresentResult::QueuedSuboptimal,
            vk::Result::SUCCESS => PresentResult::QueuedOptimal,
            vk::Result::SUBOPTIMAL_KHR => PresentResult::QueuedSuboptimal,
            vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT => {
                PresentResult::NotQueued(PresentError::FullScreenExclusiveModeLost)
            }
            vk::Result::ERROR_OUT_OF_DATE_KHR => PresentResult::NotQueued(PresentError::OutOfDate),
            vk::Result::ERROR_SURFACE_LOST_KHR => {
                PresentResult::NotQueued(PresentError::SurfaceLost)
            }
            vk::Result::ERROR_DEVICE_LOST => PresentResult::NotQueued(PresentError::DeviceLost),
            _ => PresentResult::NotQueued(fallback_error.unwrap_or(PresentError::DeviceLost)),
        }
    }

    fn present_result_requires_recreate(result: PresentResult) -> bool {
        matches!(
            result,
            PresentResult::QueuedSuboptimal
                | PresentResult::NotQueued(
                    PresentError::FullScreenExclusiveModeLost | PresentError::OutOfDate,
                )
        )
    }

    /// Presents images previously acquired from their [`Swapchain`] instances.
    ///
    /// Panics with the same safety as [`Device::with_queue`].
    #[profiling::function]
    pub fn queue_present<'a>(
        queue_family_index: u32,
        queue_index: u32,
        present_info: impl IntoIterator<Item = PresentInfo<'a>>,
    ) -> Result<PresentBatch, PresentBatchError> {
        let mut present_info = present_info.into_iter().collect::<Vec<_>>();
        let Some(first) = present_info.first() else {
            return Ok(PresentBatch::new(Box::new([])));
        };
        let present_count = present_info.len();
        let device = first.swapchain.surface.device.clone();
        let queue = QueueAddress {
            family_index: queue_family_index,
            queue_index,
        };

        for info in &mut present_info {
            if !Device::is_same(&device, &info.swapchain.surface.device) {
                warn!("unable to present swapchains from different devices in one batch");

                return Err(PresentBatchError::new(
                    PresentFailure::Driver(DriverError::InvalidData),
                    present_count,
                ));
            }

            info.swapchain
                .validate_present_info(&info.image, queue_family_index, queue_index);
            info.swapchain
                .live
                .wait_for_present_idx(
                    &device,
                    info.image.index,
                    info.swapchain.info.present_timeout,
                )
                .map_err(|err| {
                    PresentBatchError::new(
                        Self::present_failure_from_swapchain_error(err),
                        present_count,
                    )
                })?;
        }

        let mut image_indices = Vec::with_capacity(present_info.len());
        let mut present_ids = Vec::with_capacity(present_info.len());
        let mut results = vec![vk::Result::ERROR_UNKNOWN; present_info.len()];
        let mut swapchains = Vec::with_capacity(present_info.len());
        let mut wait_semaphores = Vec::new();

        for info in &mut present_info {
            let image_idx = info.image.index as usize;

            image_indices.push(info.image.index);
            swapchains.push(info.swapchain.live.handle());
            wait_semaphores.extend_from_slice(info.wait_semaphores);
            present_ids.push(
                info.swapchain
                    .live
                    .next_present_id(image_idx)
                    .unwrap_or_default(),
            );
        }

        let mut vk_present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .results(&mut results);

        let mut present_id_info;
        if present_ids.iter().any(|present_id| *present_id != 0) {
            present_id_info = vk::PresentIdKHR::default().present_ids(&present_ids);
            vk_present_info = vk_present_info.push_next(&mut present_id_info);
        }

        let present =
            Device::with_queue(&device, queue_family_index, queue_index, |queue| unsafe {
                Device::expect_vk_khr_swapchain(&device)
                    .queue_present(queue, &vk_present_info)
                    .map_err(Self::map_present_error)
            });
        let present_error = present.err();
        let aggregate_suboptimal = matches!(present, Ok(true));
        let present_results = results
            .iter()
            .map(|result| Self::present_result(*result, aggregate_suboptimal, present_error))
            .collect::<Box<_>>();

        let present_failure = present_error.map(Self::present_failure_from_present_error);
        let recreate_batch = present_failure.is_some_and(Self::present_failure_requires_recreate);

        for (idx, info) in present_info.iter_mut().enumerate() {
            if recreate_batch || Self::present_result_requires_recreate(present_results[idx]) {
                info.swapchain.recreate_pending = true;
            }

            if !present_results[idx].queued() {
                info.swapchain
                    .live
                    .clear_present_id(info.image.index as usize);
            }
        }

        for (idx, info) in present_info.iter_mut().enumerate() {
            if !present_results[idx].queued() {
                continue;
            }

            let image = unsafe { info.image.to_detached() };
            info.swapchain.live.mark_present_queued(image, queue);
        }

        let signal = Device::with_queue(&device, queue_family_index, queue_index, |queue| {
            for (idx, info) in present_info.iter_mut().enumerate() {
                if !present_results[idx].queued() {
                    continue;
                }

                info.swapchain.queue_signals.submit(
                    &device,
                    queue,
                    queue_family_index,
                    queue_index,
                )?;
            }

            Ok::<_, SwapchainError>(())
        });

        if let Err(err) = signal {
            return Err(PresentBatchError::with_results(
                Self::present_failure_from_swapchain_error(err),
                &present_results,
            ));
        }

        if let Some(err) = present_error {
            return Err(PresentBatchError::with_results(
                Self::present_failure_from_present_error(err),
                &present_results,
            ));
        }

        Ok(PresentBatch::new(present_results))
    }

    /// Recreates the swapchain using the currently requested configuration.
    #[profiling::function]
    pub fn recreate(&mut self) -> Result<(), SwapchainError> {
        self.queue_signals
            .wait_for_and_reset_all(&self.surface.device)?;
        self.destroy_retired()?;

        let (new_live, info) = SwapchainState::create(
            &self.surface,
            &mut self.requested_info,
            Some(self.live.handle()),
        )?;
        let old_live = replace(&mut self.live, new_live);

        self.retired.push(old_live.into_retired());
        self.handle = self.live.handle();
        self.info = info;
        self.recreate_pending = false;

        #[cfg(feature = "checked")]
        {
            self.swapchain_id = self.live.swapchain_id;
        }

        Ok(())
    }

    /// Updates the requested information which controls this swapchain.
    ///
    /// The live effective [`info`](Self::info) field updates [`acquire_timeout`](SwapchainInfo::acquire_timeout)
    /// immediately. Other fields are updated lazily after the next successful acquire recreates the
    /// swapchain and reflects the runtime values that were actually selected.
    pub fn set_info(&mut self, info: impl Into<SwapchainInfo>) {
        let info: SwapchainInfo = info.into();

        let recreate = self.requested_info.height != info.height
            || self.requested_info.clipped != info.clipped
            || self.requested_info.image_array_layers != info.image_array_layers
            || self.requested_info.image_sharing_mode != info.image_sharing_mode
            || self.requested_info.min_image_count != info.min_image_count
            || self.requested_info.pre_transform != info.pre_transform
            || self.requested_info.present_mode != info.present_mode
            || self.requested_info.composite_alpha != info.composite_alpha
            || self.requested_info.surface != info.surface
            || self.requested_info.width != info.width;

        self.requested_info = info;
        self.info.acquire_timeout = info.acquire_timeout;
        self.info.present_timeout = info.present_timeout;
        self.live.acquire_timeout = info.acquire_timeout;
        self.live.present_timeout = info.present_timeout;

        trace!("requested info: {:?}", self.requested_info);

        if recreate {
            self.recreate_pending = true;
        }
    }

    fn supported_surface_usage(
        device: &Device,
        surface_format: vk::Format,
        surface_capabilities: vk::ImageUsageFlags,
    ) -> Result<vk::ImageUsageFlags, DriverError> {
        let image_usage = surface_capabilities & !vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;

        if image_usage.is_empty() {
            warn!("surface reports no supported swapchain image usage");

            return Err(DriverError::Unsupported);
        }

        if device
            .physical
            .image_format_properties(
                surface_format,
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::OPTIMAL,
                image_usage,
                vk::ImageCreateFlags::empty(),
            )
            .inspect_err(|err| {
                warn!(
                    "unable to get image format properties: {:?} {:?} {err}",
                    surface_format, image_usage
                )
            })?
            .is_none()
        {
            warn!(
                "unsupported swapchain image usage combination: {:?} {:?}",
                surface_format, image_usage
            );

            return Err(DriverError::Unsupported);
        }

        Ok(image_usage)
    }

    fn validate_present_info(
        &self,
        image: &SwapchainImage,
        queue_family_index: u32,
        queue_index: u32,
    ) {
        #[cfg(feature = "checked")]
        {
            assert!(
                image.index < self.live.images.len() as u32,
                "swapchain image index out of bounds"
            );
            assert!(
                image.swapchain_id == self.swapchain_id,
                "swapchain image belongs to a different swapchain"
            );
        }

        #[cfg(feature = "checked")]
        assert!(
            self.queue_signals
                .families
                .get(queue_family_index as usize)
                .and_then(|queue_family| queue_family.queue_signals.get(queue_index as usize))
                .is_some(),
            "unsupported queue index"
        );
    }

    /// Waits for a previously queued present on `image_idx` from the live swapchain to complete.
    ///
    /// This does not search retired swapchain handles after recreation. Callers that may span
    /// recreation should retire their own pending image references before acquiring again.
    ///
    /// # Safety
    ///
    /// Callers must ensure that `image_idx` refers to an image acquired from this swapchain and
    /// that waiting for the associated present is valid for the current lifetime of the swapchain.
    pub unsafe fn wait_for_present_image(
        &mut self,
        image_idx: u32,
        timeout: u64,
    ) -> Result<(), SwapchainError> {
        self.live
            .wait_for_present_idx(&self.surface.device, image_idx, timeout)
    }
}

impl Drop for Swapchain {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        if let Err(err) = self
            .queue_signals
            .wait_for_and_reset_all(&self.surface.device)
        {
            warn!("unable to wait for swapchain signals: {err}");
        }

        self.destroy_retired_on_drop();

        if let Err(err) = self
            .live
            .wait_for_all_presents(&self.surface.device, self.info.present_timeout)
        {
            // Drop is best-effort shutdown. At this point there is no recoverable path for the
            // public owner, so log the failed wait and still release the Vulkan handle.
            warn!("unable to wait for live swapchain presents: {err:?}");
        }

        // Destroy handles explicitly while queue signals are still alive; their field drop order is
        // otherwise independent of the present-retirement fallback queues recorded by the states.
        self.live.handle.destroy();
        self.handle = vk::SwapchainKHR::null();
    }
}

impl Eq for Swapchain {}

impl PartialEq for Swapchain {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

/// Describes the condition of a swapchain.
#[derive(Clone, Copy, Debug)]
pub enum SwapchainError {
    /// This frame is lost but more may be acquired later.
    DeviceLost,

    /// Recoverable driver error.
    Driver(DriverError),

    /// Exclusive full-screen ownership was lost and the swapchain should be recreated.
    FullScreenExclusiveModeLost,

    /// No image is currently available, but a future acquire may succeed.
    NotReady,

    /// The surface was lost and must be recreated, which includes any operating system window.
    SurfaceLost,

    /// No image was acquired before the configured acquire timeout elapsed.
    Timeout,
}

impl SwapchainError {
    fn different(self, other: Self) -> bool {
        !self.matches(other)
    }

    fn matches(self, other: Self) -> bool {
        use SwapchainError::*;

        matches!(
            (self, other),
            (DeviceLost, DeviceLost)
                | (
                    Driver(DriverError::InvalidData),
                    Driver(DriverError::InvalidData)
                )
                | (
                    Driver(DriverError::OutOfMemory),
                    Driver(DriverError::OutOfMemory)
                )
                | (
                    Driver(DriverError::Unsupported),
                    Driver(DriverError::Unsupported)
                )
                | (FullScreenExclusiveModeLost, FullScreenExclusiveModeLost)
                | (NotReady, NotReady)
                | (SurfaceLost, SurfaceLost)
                | (Timeout, Timeout)
        )
    }
}

impl From<DriverError> for SwapchainError {
    fn from(err: DriverError) -> Self {
        match err {
            DriverError::InvalidData => Self::DeviceLost,
            err => Self::Driver(err),
        }
    }
}

#[derive(Debug)]
struct SwapchainHandle {
    device: Device,
    handle: vk::SwapchainKHR,
}

impl SwapchainHandle {
    fn new(device: &Device, handle: vk::SwapchainKHR) -> Self {
        Self {
            device: device.clone(),
            handle,
        }
    }

    fn destroy(&mut self) {
        Self::destroy_raw(&self.device, &mut self.handle);
    }

    fn destroy_raw(device: &Device, handle: &mut vk::SwapchainKHR) {
        if handle.is_null() {
            return;
        }

        unsafe {
            Device::expect_vk_khr_swapchain(device).destroy_swapchain(*handle, None);
        }

        *handle = vk::SwapchainKHR::null();
    }

    fn raw(&self) -> vk::SwapchainKHR {
        self.handle
    }
}

impl Drop for SwapchainHandle {
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        self.destroy();
    }
}

/// Opaque swapchain ownership identifier used by checked builds.
#[cfg(feature = "checked")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SwapchainId(u64);

#[cfg(feature = "checked")]
impl SwapchainId {
    fn next() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);

        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// An opaque type representing a swapchain image.
///
/// In checked builds, using a swapchain image with a different swapchain than the one that acquired
/// it will panic.
#[derive(Debug)]
#[read_only::embed]
pub struct SwapchainImage {
    #[readonly]
    pub(self) image: Image,

    /// The swapchain image index.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub index: u32,

    /// Whether the swapchain was suboptimal when this image was acquired.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub suboptimal: bool,

    #[cfg(feature = "checked")]
    #[readonly]
    pub(self) swapchain_id: SwapchainId,
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
                image: unsafe { Image::from_raw(device, handle, info) },
                index,
                suboptimal: false,

                #[cfg(feature = "checked")]
                swapchain_id: SwapchainId::next(),
            },
        }
    }

    /// Creates an acquired wrapper for the same native swapchain image.
    ///
    /// # Safety
    ///
    /// The caller must ensure the Vulkan image handle remains valid for the lifetime of the
    /// returned image. This should only be called for images owned by a live swapchain, and each
    /// detached wrapper must be used according to the swapchain image acquire/present lifecycle.
    pub unsafe fn to_detached(&self) -> Self {
        Self {
            read_only: ReadOnlySwapchainImage {
                image: unsafe { self.image.to_detached() },
                index: self.index,
                suboptimal: false,

                #[cfg(feature = "checked")]
                swapchain_id: self.swapchain_id,
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

    /// The initial height of the surface.
    #[builder(default)]
    pub height: u32,

    /// The requested number of array layers for each swapchain image.
    #[builder(default = "1")]
    pub image_array_layers: u32,

    /// Sharing mode used for swapchain images.
    #[builder(default = vk::SharingMode::EXCLUSIVE)]
    pub image_sharing_mode: vk::SharingMode,

    /// The minimum number of presentable images that the application needs.
    #[builder(default = "2")]
    pub min_image_count: u32,

    /// Transform applied to presented images.
    #[builder(default = vk::SurfaceTransformFlagsKHR::IDENTITY)]
    pub pre_transform: vk::SurfaceTransformFlagsKHR,

    /// `vk::PresentModeKHR` determines timing and queueing with which frames are displayed.
    #[builder(default = vk::PresentModeKHR::IMMEDIATE)]
    pub present_mode: vk::PresentModeKHR,

    /// Timeout in nanoseconds used when presenting an image.
    #[builder(default = "u64::MAX")]
    pub present_timeout: u64,

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
            acquire_timeout: u64::MAX,
            clipped: true,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            height,
            image_array_layers: 1,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            min_image_count: 2,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            present_mode: vk::PresentModeKHR::IMMEDIATE,
            present_timeout: u64::MAX,
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
        let Self {
            acquire_timeout,
            clipped,
            composite_alpha,
            height,
            image_array_layers,
            image_sharing_mode,
            min_image_count,
            pre_transform,
            present_mode,
            present_timeout,
            surface,
            width,
        } = self;

        let acquire_timeout = Some(acquire_timeout);
        let clipped = Some(clipped);
        let composite_alpha = Some(composite_alpha);
        let height = Some(height);
        let image_array_layers = Some(image_array_layers);
        let image_sharing_mode = Some(image_sharing_mode);
        let min_image_count = Some(min_image_count);
        let pre_transform = Some(pre_transform);
        let present_mode = Some(present_mode);
        let present_timeout = Some(present_timeout);
        let surface = Some(surface);
        let width = Some(width);

        SwapchainInfoBuilder {
            acquire_timeout,
            clipped,
            composite_alpha,
            height,
            image_array_layers,
            image_sharing_mode,
            min_image_count,
            pre_transform,
            present_mode,
            present_timeout,
            surface,
            width,
        }
    }
}

impl From<SwapchainInfoBuilder> for SwapchainInfo {
    fn from(info: SwapchainInfoBuilder) -> Self {
        info.build()
    }
}

impl From<EffectiveSwapchainInfo> for SwapchainInfo {
    fn from(info: EffectiveSwapchainInfo) -> Self {
        let EffectiveSwapchainInfo {
            acquire_timeout,
            clipped,
            composite_alpha,
            height,
            image_array_layers,
            image_count: _,
            image_sharing_mode,
            min_image_count,
            pre_transform,
            present_mode,
            present_timeout,
            surface,
            width,
        } = info;

        Self {
            acquire_timeout,
            clipped,
            composite_alpha,
            height,
            image_array_layers,
            image_sharing_mode,
            min_image_count,
            pre_transform,
            present_mode,
            present_timeout,
            surface,
            width,
        }
    }
}

impl SwapchainInfoBuilder {
    /// Builds a new `SwapchainInfo`.
    #[inline(always)]
    pub fn build(self) -> SwapchainInfo {
        self.fallible_build().expect("all fields have defaults")
    }
}

#[derive(Debug)]
struct SwapchainState<S> {
    __: PhantomData<S>,
    acquire_timeout: u64,
    handle: SwapchainHandle,
    images: Box<[Option<Image>]>,
    present_timeout: u64,
    retirement: PresentRetirement,

    #[cfg(feature = "checked")]
    swapchain_id: SwapchainId,
}

impl<S> SwapchainState<S> {
    fn handle(&self) -> vk::SwapchainKHR {
        self.handle.raw()
    }

    fn wait_for_all_presents(
        &mut self,
        device: &Device,
        timeout: u64,
    ) -> Result<(), SwapchainError> {
        if !self.retirement.present_pending {
            return Ok(());
        }

        if !self.retirement.use_present_wait {
            let queue = self.retirement.queue.expect("missing present queue");
            let queue_wait_idle =
                Device::with_queue(device, queue.family_index, queue.queue_index, |queue| {
                    Device::queue_wait_idle(device, queue)
                });

            self.retirement.present_pending = false;
            queue_wait_idle?;

            return Ok(());
        }

        let swapchain = self.handle();
        let mut res = None;

        for present_id in self
            .retirement
            .image_present_ids
            .iter_mut()
            .filter_map(|present_id| present_id.map(|_| present_id))
        {
            let Some(id) = *present_id else {
                unreachable!();
            };

            let wait_for_present =
                unsafe { Self::wait_for_present(device, swapchain, id, timeout) };

            *present_id = None;

            if let Err(err) = wait_for_present
                && !matches!(res, Some(SwapchainError::Driver(DriverError::Unsupported)))
                && res
                    .replace(err)
                    .map(|old| old.different(err))
                    .unwrap_or_default()
            {
                warn!("unsupported multi-error: {err:?}");

                res = Some(SwapchainError::Driver(DriverError::Unsupported))
            }
        }

        self.retirement.present_pending = false;

        res.map(Result::Err).unwrap_or(Ok(()))
    }

    unsafe fn wait_for_present(
        device: &Device,
        swapchain: vk::SwapchainKHR,
        present_id: u64,
        timeout: u64,
    ) -> Result<(), SwapchainError> {
        unsafe {
            Device::expect_vk_khr_present_wait(device)
                .wait_for_present(swapchain, present_id, timeout)
                .map_err(|err| {
                    warn!("unable to wait for present: {err}");

                    match err {
                        vk::Result::ERROR_DEVICE_LOST => SwapchainError::DeviceLost,
                        vk::Result::ERROR_SURFACE_LOST_KHR => SwapchainError::SurfaceLost,
                        vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT => {
                            SwapchainError::FullScreenExclusiveModeLost
                        }
                        vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => {
                            SwapchainError::NotReady
                        }
                        _ => SwapchainError::SurfaceLost,
                    }
                })
        }
    }
}

impl SwapchainState<Live> {
    fn create(
        surface: &Surface,
        requested_info: &mut SwapchainInfo,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> Result<(Self, EffectiveSwapchainInfo), DriverError> {
        let device = &surface.device;
        let surface_caps = Surface::capabilities(surface)?;

        {
            let composite_alpha = Surface::composite_alpha_or_default(
                surface_caps.supported_composite_alpha,
                requested_info.composite_alpha,
            );
            if requested_info.composite_alpha != composite_alpha {
                warn!(
                    "requested composite alpha unsupported: {:?}; falling back to {:?}",
                    requested_info.composite_alpha, composite_alpha
                );

                requested_info.composite_alpha = composite_alpha;
            }
        }

        {
            let image_array_layers = requested_info
                .image_array_layers
                .clamp(1, surface_caps.max_image_array_layers);
            if requested_info.image_array_layers != image_array_layers {
                warn!(
                    "requested image array layers unsupported: {}; falling back to {}",
                    requested_info.image_array_layers, image_array_layers
                );

                requested_info.image_array_layers = image_array_layers;
            }
        }

        {
            if requested_info.image_sharing_mode == vk::SharingMode::CONCURRENT
                && device.physical.queue_families.len() < 2
            {
                warn!(
                    "requested concurrent image sharing with one queue family; falling back to exclusive"
                );

                requested_info.image_sharing_mode = vk::SharingMode::EXCLUSIVE;
            }
        }

        {
            requested_info.min_image_count =
                Swapchain::clamp_min_image_count(requested_info.min_image_count, surface_caps);
        }

        {
            if !surface_caps
                .supported_transforms
                .contains(requested_info.pre_transform)
            {
                warn!(
                    "requested pre-transform unsupported: {:?}; falling back to {:?}",
                    requested_info.pre_transform, surface_caps.current_transform
                );

                requested_info.pre_transform = surface_caps.current_transform;
            }
        }

        {
            let present_modes = Surface::present_modes(surface)?;
            if !present_modes.contains(&requested_info.present_mode) {
                warn!(
                    "requested present mode unsupported: {:?}; falling back to FIFO",
                    requested_info.present_mode
                );

                requested_info.present_mode = vk::PresentModeKHR::FIFO;
            }
        }

        {
            // TODO: There is special handling for the MAX case we're not doing (read specs)
            if surface_caps.current_extent.width == u32::MAX
                && surface_caps.current_extent.height == u32::MAX
            {
                requested_info.width = requested_info.width.clamp(
                    surface_caps.min_image_extent.width,
                    surface_caps.max_image_extent.width,
                );
                requested_info.height = requested_info.height.clamp(
                    surface_caps.min_image_extent.height,
                    surface_caps.max_image_extent.height,
                );
            } else {
                requested_info.width = surface_caps.current_extent.width;
                requested_info.height = surface_caps.current_extent.height;
            }

            if requested_info.width == 0 || requested_info.height == 0 {
                warn!(
                    "invalid surface extent: computed {}x{}",
                    requested_info.width, requested_info.height
                );

                return Err(DriverError::Unsupported);
            }
        }

        let image_extent = vk::Extent2D {
            width: requested_info.width,
            height: requested_info.height,
        };
        let image_usage = Swapchain::supported_surface_usage(
            device,
            requested_info.surface.format,
            surface_caps.supported_usage_flags,
        )?;

        let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .clipped(requested_info.clipped)
            .composite_alpha(requested_info.composite_alpha)
            .image_color_space(requested_info.surface.color_space)
            .image_extent(image_extent)
            .image_format(requested_info.surface.format)
            .image_sharing_mode(requested_info.image_sharing_mode)
            .image_usage(image_usage)
            .min_image_count(requested_info.min_image_count)
            .old_swapchain(old_swapchain.unwrap_or_default())
            .pre_transform(requested_info.pre_transform)
            .present_mode(requested_info.present_mode)
            .image_array_layers(requested_info.image_array_layers)
            .surface(surface.handle);

        let queue_family_indices = matches!(
            requested_info.image_sharing_mode,
            vk::SharingMode::CONCURRENT
        )
        .then(|| (0..device.physical.queue_families.len() as u32).collect::<Box<_>>());

        if let Some(queue_family_indices) = &queue_family_indices {
            swapchain_create_info =
                swapchain_create_info.queue_family_indices(queue_family_indices);
        }

        let vk_khr_swapchain = Device::expect_vk_khr_swapchain(device);
        let mut handle = unsafe {
            vk_khr_swapchain
                .create_swapchain(&swapchain_create_info, None)
                .map_err(|err| {
                    warn!("unable to create swapchain: {err}");

                    // TODO: Improve error handling

                    DriverError::Unsupported
                })?
        };
        let images = unsafe {
            vk_khr_swapchain
                .get_swapchain_images(handle)
                .map_err(|err| {
                    SwapchainHandle::destroy_raw(device, &mut handle);

                    match err {
                        vk::Result::INCOMPLETE => {
                            warn!("invalid swapchain image enumeration: incomplete");

                            DriverError::InvalidData
                        }
                        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                        | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                            warn!("unable to get swapchain images: {err}");

                            DriverError::OutOfMemory
                        }
                        _ => {
                            error!("unable to get swapchain images: {err}");

                            DriverError::Unsupported
                        }
                    }
                })?
        };

        #[cfg(feature = "checked")]
        let swapchain_id = SwapchainId::next();

        let vk::Extent2D { height, width } = image_extent;
        let vk::SwapchainCreateInfoKHR {
            image_format,
            image_array_layers,
            ..
        } = swapchain_create_info;

        let images = {
            let image_info = ImageInfo::image_2d_array(
                width,
                height,
                image_array_layers,
                image_format,
                image_usage,
            );

            images
                .into_iter()
                .enumerate()
                .map(|(idx, image)| {
                    let index = idx as u32;
                    let image = unsafe { Image::from_raw(device, image, image_info) };

                    image.set_debug_name(format!("swapchain{index}"));

                    Some(image)
                })
                .collect::<Box<_>>()
        };

        let image_count = images.len() as _;

        let info = {
            let SwapchainInfo {
                acquire_timeout,
                clipped,
                image_array_layers,
                present_timeout,
                surface,
                ..
            } = *requested_info;
            let vk::SwapchainCreateInfoKHR {
                composite_alpha,
                image_sharing_mode,
                min_image_count,
                pre_transform,
                present_mode,
                ..
            } = swapchain_create_info;

            EffectiveSwapchainInfo {
                acquire_timeout,
                clipped,
                composite_alpha,
                height,
                image_array_layers,
                image_count,
                image_sharing_mode,
                min_image_count,
                pre_transform,
                present_mode,
                present_timeout,
                surface,
                width,
            }
        };

        info!(
            "swapchain {}x{} {:?}x{image_count} {:?} {image_usage:#?}",
            info.width, info.height, info.present_mode, info.surface.format,
        );

        let use_present_wait = {
            surface.device.physical.vk_khr_present_id.is_some()
                && surface.device.physical.vk_khr_present_wait.is_some()
        };

        Ok((
            Self {
                acquire_timeout: info.acquire_timeout,
                handle: SwapchainHandle::new(device, handle),
                images,
                present_timeout: info.present_timeout,
                retirement: PresentRetirement::new(image_count as _, use_present_wait),

                #[cfg(feature = "checked")]
                swapchain_id,

                __: PhantomData,
            },
            info,
        ))
    }

    fn acquire_next_image(
        &mut self,
        device: &Device,
        acquired: vk::Semaphore,
    ) -> Result<SwapchainImage, AcquireImageError> {
        let res = unsafe {
            Device::expect_vk_khr_swapchain(device)
                .acquire_next_image(
                    self.handle(),
                    self.acquire_timeout,
                    acquired,
                    vk::Fence::null(),
                )
                .map_err(|err| match err {
                    vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT => {
                        SwapchainError::FullScreenExclusiveModeLost.into()
                    }
                    vk::Result::ERROR_OUT_OF_DATE_KHR => AcquireImageError::OutOfDate,
                    vk::Result::NOT_READY => SwapchainError::NotReady.into(),
                    vk::Result::TIMEOUT => SwapchainError::Timeout.into(),
                    vk::Result::ERROR_DEVICE_LOST => SwapchainError::DeviceLost.into(),
                    vk::Result::ERROR_SURFACE_LOST_KHR => SwapchainError::SurfaceLost.into(),
                    _ => AcquireImageError::Swapchain(DriverError::Unsupported.into()),
                })
        };

        if let Err(err) = res {
            use {AcquireImageError::*, Level::*, SwapchainError::*};

            let level = match err {
                OutOfDate | Swapchain(NotReady | Timeout) => Debug,
                Swapchain(DeviceLost | SurfaceLost) => Warn,
                _ => Error,
            };

            log!(level, "unable to acquire image: {err:?}");
        }

        let (index, suboptimal) = res?;

        if suboptimal {
            debug!("acquired image is suboptimal");
        }

        let image = self
            .images
            .get_mut(index as usize)
            .map(|image| image.take().expect("expected available image"))
            .expect("expected image");

        let swapchain_image = SwapchainImage {
            read_only: ReadOnlySwapchainImage {
                image,
                index,
                suboptimal,
                swapchain_id: self.swapchain_id,
            },
        };

        Ok(swapchain_image)
    }

    fn clear_present_id(&mut self, image_idx: usize) {
        let present_id = &mut self.retirement.image_present_ids[image_idx];

        debug_assert!(present_id.is_some());

        *present_id = None;
    }

    fn into_retired(self) -> SwapchainState<Retired> {
        SwapchainState {
            acquire_timeout: self.acquire_timeout,
            handle: self.handle,
            images: self.images,
            present_timeout: self.present_timeout,
            retirement: self.retirement,

            #[cfg(feature = "checked")]
            swapchain_id: self.swapchain_id,

            __: PhantomData,
        }
    }

    fn mark_present_queued(&mut self, image: SwapchainImage, queue: QueueAddress) {
        let ReadOnlySwapchainImage { image, index, .. } = image.read_only;

        self.images[index as usize] = Some(image);
        self.retirement.present_pending = true;
        self.retirement.queue = Some(queue);
    }

    fn next_present_id(&mut self, image_idx: usize) -> Option<u64> {
        if !self.retirement.use_present_wait {
            return None;
        }

        let next_present_id = self.retirement.next_present_id;
        self.retirement.next_present_id += 1;

        let present_id = &mut self.retirement.image_present_ids[image_idx];

        debug_assert!(present_id.is_none());

        *present_id = Some(next_present_id);

        Some(next_present_id)
    }

    fn wait_for_present_idx(
        &mut self,
        device: &Device,
        image_idx: u32,
        timeout: u64,
    ) -> Result<(), SwapchainError> {
        let swapchain = self.handle();
        let Some(image_present_id) = self
            .retirement
            .image_present_ids
            .get_mut(image_idx as usize)
        else {
            return Ok(());
        };
        let Some(present_id) = *image_present_id else {
            return Ok(());
        };

        unsafe {
            Self::wait_for_present(device, swapchain, present_id, timeout)?;
        }

        *image_present_id = None;

        if self
            .retirement
            .image_present_ids
            .iter()
            .all(Option::is_none)
        {
            self.retirement.present_pending = false;
        }

        Ok(())
    }
}

impl SwapchainState<Retired> {
    fn destroy_when_idle(mut self, device: &Device) -> Result<(), SwapchainError> {
        self.wait_until_retired(device, self.present_timeout)?;
        self.handle.destroy();

        Ok(())
    }

    fn wait_until_retired(&mut self, device: &Device, timeout: u64) -> Result<(), SwapchainError> {
        self.wait_for_all_presents(device, timeout)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    type Info = SwapchainInfo;
    type Builder = SwapchainInfoBuilder;
    type EffectiveInfo = EffectiveSwapchainInfo;

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

    #[test]
    pub fn effective_swapchain_info_round_trips_to_requested_info() {
        let info = EffectiveInfo {
            acquire_timeout: 42,
            clipped: false,
            composite_alpha: vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            height: 123,
            image_array_layers: 2,
            image_count: 7,
            image_sharing_mode: vk::SharingMode::CONCURRENT,
            min_image_count: 3,
            pre_transform: vk::SurfaceTransformFlagsKHR::ROTATE_90,
            present_mode: vk::PresentModeKHR::FIFO,
            present_timeout: 999,
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
                height: 123,
                image_array_layers: 2,
                image_sharing_mode: vk::SharingMode::CONCURRENT,
                min_image_count: 3,
                pre_transform: vk::SurfaceTransformFlagsKHR::ROTATE_90,
                present_mode: vk::PresentModeKHR::FIFO,
                present_timeout: 999,
                surface: vk::SurfaceFormatKHR::default()
                    .format(vk::Format::B8G8R8A8_UNORM)
                    .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR),
                width: 456,
            }
        );
    }
}
