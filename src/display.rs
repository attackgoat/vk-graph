//! Resource leasing and pooling types.

use {
    super::{
        Graph,
        driver::{
            CommandBuffer, CommandBufferInfo, DescriptorPool, DescriptorPoolInfo, DriverError,
            RenderPass, RenderPassInfo,
            device::Device,
            image::Image,
            image_access_layout,
            swapchain::{Swapchain, SwapchainError, SwapchainImage, SwapchainInfo},
        },
        node::SwapchainImageNode,
        pool::Pool,
    },
    ash::vk,
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
    vk_sync::{AccessType, ImageBarrier, cmd::pipeline_barrier},
};

/// A physical display interface.
#[repr(C)]
pub struct Display {
    exec_idx: usize,
    execs: Box<[Execution]>,

    /// Information used to create this resource.
    ///
    /// _Note:_ This field is read-only.
    #[cfg(doc)]
    pub info: DisplayInfo,

    #[cfg(not(doc))]
    info: DisplayInfo,

    /// The swapchain which supports this display.
    ///
    /// _Note:_ This field is read-only.
    #[cfg(doc)]
    pub swapchain: Swapchain,

    #[cfg(not(doc))]
    swapchain: Swapchain,
}

#[doc(hidden)]
#[repr(C)]
pub struct DisplayRef {
    exec_idx: usize,
    execs: Box<[Execution]>,
    pub info: DisplayInfo,
    pub swapchain: Swapchain,
}

impl Display {
    /// Constructs a new `Display` object.
    pub fn new(
        device: &Device,
        swapchain: Swapchain,
        info: impl Into<DisplayInfo>,
    ) -> Result<Self, DriverError> {
        let info: DisplayInfo = info.into();

        assert_ne!(info.command_buffer_count, 0);

        let mut execs = Vec::with_capacity(info.command_buffer_count as _);
        for _ in 0..info.command_buffer_count {
            let cmd_buf =
                CommandBuffer::create(device, CommandBufferInfo::new(info.queue_family_index))?;
            let swapchain_acquired = Device::create_semaphore(device)?;
            let swapchain_rendered = Device::create_semaphore(device)?;

            execs.push(Execution {
                cmd_buf,
                queue: None,
                swapchain_acquired,
                swapchain_rendered,
            });
        }
        let execs = execs.into_boxed_slice();

        Ok(Self {
            exec_idx: info.command_buffer_count,
            execs,
            info,
            swapchain,
        })
    }

    /// Gets the next available swapchain image which should be rendered to and then presented using
    /// [`present_image`][Self::present_image].
    pub fn acquire_next_image(&mut self) -> Result<Option<SwapchainImage>, DisplayError> {
        self.exec_idx += 1;
        self.exec_idx %= self.execs.len();
        let exec = &mut self.execs[self.exec_idx];

        if exec.queue.is_some() {
            exec.cmd_buf.wait_until_executed().inspect_err(|err| {
                warn!("unable to wait for display fence: {err}");
            })?;

            exec.queue = None;
        }

        exec.cmd_buf.drop_fenced();

        unsafe {
            exec.cmd_buf
                .device
                .reset_fences(slice::from_ref(&exec.cmd_buf.fence))
                .map_err(|err| {
                    warn!("unable to reset display fence: {err}");

                    DriverError::InvalidData
                })?;
        }

        let acquire_next_image = self.swapchain.acquire_next_image(exec.swapchain_acquired);

        if let Err(err) = acquire_next_image {
            warn!("unable to acquire next swapchain image: {err:?}");
        }

        let mut swapchain_image = match acquire_next_image {
            Err(SwapchainError::DeviceLost) => Err(DisplayError::DeviceLost),
            Err(SwapchainError::Suboptimal) => return Ok(None),
            Err(SwapchainError::SurfaceLost) => Err(DisplayError::Driver(DriverError::InvalidData)),
            Ok(swapchain_image) => Ok(swapchain_image),
        }?;
        swapchain_image.exec_idx = self.exec_idx;

        Ok(Some(swapchain_image))
    }

    /// Displays the given swapchain image using passes specified in `render_graph`, if possible.
    #[profiling::function]
    pub fn present_image(
        &mut self,
        pool: &mut impl ResolverPool,
        render_graph: Graph,
        swapchain_image: SwapchainImageNode,
        queue_index: u32,
    ) -> Result<(), DisplayError> {
        trace!("present_image");

        let mut resolver = render_graph.resolve();
        let wait_dst_stage_mask = resolver.node_pipeline_stages(swapchain_image);

        // The swapchain should have been written to, otherwise it would be noise and that's a panic
        assert!(
            !wait_dst_stage_mask.is_empty(),
            "uninitialized swapchain image: write something each frame!",
        );

        let exec_idx = resolver.swapchain_image(swapchain_image).exec_idx;
        let exec = &mut self.execs[exec_idx];

        debug_assert!(exec.queue.is_none());

        let started = Instant::now();

        unsafe {
            exec.cmd_buf
                .device
                .begin_command_buffer(
                    exec.cmd_buf.handle,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|_| ())?;
        }

        // resolver.record_node_dependencies(&mut *self.pool, cmd_buf, swapchain_image)?;
        resolver.record_node(pool, &mut exec.cmd_buf, swapchain_image)?;

        {
            let swapchain_image = resolver.swapchain_image(swapchain_image);
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
                    **swapchain_image,
                    access,
                    AccessType::Present,
                );

                // Force a presentation layout transition
                pipeline_barrier(
                    &exec.cmd_buf.device,
                    exec.cmd_buf.handle,
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
        resolver.record_unscheduled_passes(pool, &mut exec.cmd_buf)?;

        let queue =
            exec.cmd_buf.device.queues[self.info.queue_family_index as usize][queue_index as usize];

        unsafe {
            exec.cmd_buf
                .device
                .end_command_buffer(exec.cmd_buf.handle)
                .map_err(|err| {
                    warn!("unable to end display command buffer: {err}");

                    DriverError::InvalidData
                })?;
            exec.cmd_buf
                .device
                .queue_submit(
                    queue,
                    slice::from_ref(
                        &vk::SubmitInfo::default()
                            .command_buffers(slice::from_ref(&exec.cmd_buf.handle))
                            .wait_semaphores(slice::from_ref(&exec.swapchain_acquired))
                            .wait_dst_stage_mask(slice::from_ref(&wait_dst_stage_mask))
                            .signal_semaphores(slice::from_ref(&exec.swapchain_rendered)),
                    ),
                    exec.cmd_buf.fence,
                )
                .map_err(|err| {
                    warn!("unable to submit display command buffer: {err}");

                    DriverError::InvalidData
                })?
        }

        exec.cmd_buf.waiting = true;
        exec.queue = Some(queue);

        let elapsed = Instant::now() - started;
        trace!("🔜🔜🔜 vkQueueSubmit took {} μs", elapsed.as_micros(),);

        let swapchain_image = resolver.swapchain_image(swapchain_image).clone();

        self.swapchain.present_image(
            swapchain_image,
            slice::from_ref(&exec.swapchain_rendered),
            self.info.queue_family_index,
            queue_index,
        );

        // Store the resolved graph because it contains bindings, leases, and other shared resources
        // that need to be kept alive until the fence is waited upon.
        exec.cmd_buf.push_fenced_drop(resolver);

        Ok(())
    }

    /// Updates the information which controls the swapchain.
    ///
    /// Previously acquired swapchain images should be discarded after calling this function.
    pub fn update_swapchain(&mut self, info: impl Into<SwapchainInfo>) {
        self.swapchain.update(info);
    }
}

impl Debug for Display {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Display")
    }
}

#[doc(hidden)]
impl Deref for Display {
    type Target = DisplayRef;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self as *const Self::Target) }
    }
}

impl Drop for Display {
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        let idle = unsafe { self.execs[0].cmd_buf.device.device_wait_idle() };
        if idle.is_err() {
            warn!("unable to wait for device");

            return;
        }

        for batch in &mut self.execs {
            if let Some(queue) = batch.queue {
                // Wait for presentation to stop
                let present = unsafe { batch.cmd_buf.device.queue_wait_idle(queue) };
                if present.is_err() {
                    warn!("unable to wait for queue");

                    continue;
                }
            }

            unsafe {
                batch
                    .cmd_buf
                    .device
                    .destroy_semaphore(batch.swapchain_acquired, None);
                batch
                    .cmd_buf
                    .device
                    .destroy_semaphore(batch.swapchain_rendered, None);
            }
        }
    }
}

/// Describes error conditions relating to physical displays.
#[derive(Debug)]
pub enum DisplayError {
    /// Unrecoverable device error; must destroy this device and display and start a new one
    DeviceLost,

    /// Recoverable driver error
    Driver(DriverError),
}

impl Error for DisplayError {}

impl From<()> for DisplayError {
    fn from(_: ()) -> Self {
        Self::DeviceLost
    }
}

impl From<DriverError> for DisplayError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl std::fmt::Display for DisplayError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Information used to create a [`Display`] instance.
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build", error = "DisplayInfoBuilderError"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
#[non_exhaustive]
pub struct DisplayInfo {
    /// The number of command buffers to use for image submissions.
    ///
    /// Generally one more than the swapchain image count is best.
    #[builder(default = "4")]
    command_buffer_count: usize,

    /// The device queue family which will be used to submit and present images.
    #[builder(default = "0")]
    queue_family_index: u32,
}

impl DisplayInfo {
    /// Converts a `DisplayInfo` into a `DisplayInfoBuilder`.
    #[inline(always)]
    pub fn to_builder(self) -> DisplayInfoBuilder {
        DisplayInfoBuilder {
            command_buffer_count: Some(self.command_buffer_count),
            queue_family_index: Some(self.queue_family_index),
        }
    }
}

impl Default for DisplayInfo {
    fn default() -> Self {
        Self {
            command_buffer_count: 4,
            queue_family_index: 0,
        }
    }
}

impl From<DisplayInfoBuilder> for DisplayInfo {
    fn from(info: DisplayInfoBuilder) -> Self {
        info.build()
    }
}

impl DisplayInfoBuilder {
    /// Builds a new `DisplayInfo`.
    ///
    /// # Panics
    ///
    /// If any of the following values have not been set this function will panic:
    ///
    /// * `command_buffer_count`
    #[inline(always)]
    pub fn build(self) -> DisplayInfo {
        let info = match self.fallible_build() {
            Err(DisplayInfoBuilderError(err)) => panic!("{err}"),
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
struct DisplayInfoBuilderError(UninitializedFieldError);

impl From<UninitializedFieldError> for DisplayInfoBuilderError {
    fn from(err: UninitializedFieldError) -> Self {
        Self(err)
    }
}

struct Execution {
    cmd_buf: CommandBuffer,
    queue: Option<vk::Queue>,
    swapchain_acquired: vk::Semaphore,
    swapchain_rendered: vk::Semaphore,
}

/// Combination trait which groups together all [`Pool`] traits required for a [`Resolver`]
/// instance.
///
/// [`Resolver`]: crate::graph::Resolver
#[allow(private_bounds)]
pub trait ResolverPool:
    Pool<DescriptorPoolInfo, DescriptorPool>
    + Pool<RenderPassInfo, RenderPass>
    + Pool<CommandBufferInfo, CommandBuffer>
    + Send
{
}

impl<T> ResolverPool for T where
    T: Pool<DescriptorPoolInfo, DescriptorPool>
        + Pool<RenderPassInfo, RenderPass>
        + Pool<CommandBufferInfo, CommandBuffer>
        + Send
{
}

#[cfg(test)]
mod tests {
    use super::*;

    type Info = DisplayInfo;
    type Builder = DisplayInfoBuilder;

    #[test]
    pub fn display_info() {
        let info = Info {
            command_buffer_count: 42,
            queue_family_index: 16,
        };
        let builder = info.to_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn display_info_builder() {
        let info = Info {
            command_buffer_count: 42,
            queue_family_index: 16,
        };
        let builder = Builder::default()
            .command_buffer_count(42)
            .queue_family_index(16)
            .build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn display_info_default() {
        let info = Info::default();
        let builder = Builder::default().build();

        assert_eq!(info, builder);
    }

    #[test]
    #[should_panic(expected = "Field value invalid: command_buffer_count")]
    pub fn display_info_builder_uninit_command_buffer_count() {
        Builder::default().command_buffer_count(0).build();
    }
}
