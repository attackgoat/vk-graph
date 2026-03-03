//! TODO

use {
    super::{DriverError, device::Device},
    ash::vk,
    derive_builder::{Builder, UninitializedFieldError},
    log::{error, trace, warn},
    std::{fmt::Debug, slice, thread::panicking},
};

// TODO: Expose command functions so the fence, device, waiting flags do not
// need to be public

/// Represents a Vulkan command buffer to which some work has been submitted.
#[derive(Debug)]
#[readonly::make]
pub struct CommandBuffer {
    /// The device which owns this command buffer resource.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub device: Device,

    droppables: Vec<Box<dyn Debug + Send + 'static>>,

    /// The native Vulkan fence handle of this command buffer.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub fence: vk::Fence,

    /// The native Vulkan resource handle of this command buffer.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub handle: vk::CommandBuffer,

    /// Information used to create this object.
    #[readonly]
    pub info: CommandBufferInfo,

    pub(crate) pool: vk::CommandPool,
}

impl CommandBuffer {
    /// TODO
    #[profiling::function]
    pub fn create(device: &Device, info: CommandBufferInfo) -> Result<Self, DriverError> {
        let device = device.clone();

        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(
                        vk::CommandPoolCreateFlags::TRANSIENT
                            | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    )
                    .queue_family_index(info.queue_family_index),
                None,
            )
        }
        .map_err(|err| {
            warn!("unable to create command pool: {err}");

            DriverError::Unsupported
        })?;

        let handle = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(1)
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY),
            )
        }
        .map_err(|err| {
            warn!("unable to allocate command buffer: {err}");

            DriverError::Unsupported
        })?[0];

        let fence = Device::create_fence(&device, false)?;

        Ok(Self {
            device,
            droppables: vec![],
            fence,
            handle,
            info,
            pool,
        })
    }

    /// Drops an item after execution has been completed.
    pub fn drop_after_executed(&mut self, x: impl Debug + Send + 'static) {
        self.droppables.push(Box::new(x));
    }

    /// Signals that execution has completed and it is time to drop anything we collected.
    #[profiling::function]
    fn drop_fenced(&mut self) {
        if !self.droppables.is_empty() {
            trace!("dropping {} shared references", self.droppables.len());
        }

        self.droppables.clear();
    }

    /// Returns `true` after the GPU has executed the previous submission to this command buffer.
    ///
    /// See [`Self::wait_until_executed`] to block while checking.
    #[profiling::function]
    pub fn has_executed(&self) -> Result<bool, DriverError> {
        let res = unsafe { self.device.get_fence_status(self.fence) };

        match res {
            Ok(status) => Ok(status),
            Err(err) if err == vk::Result::ERROR_DEVICE_LOST => {
                error!("Device lost");

                Err(DriverError::InvalidData)
            }
            Err(err) => {
                // VK_SUCCESS and VK_NOT_READY handled by get_fence_status in ash
                // VK_ERROR_DEVICE_LOST already handled above, so no idea what happened
                error!("{}", err);

                Err(DriverError::InvalidData)
            }
        }
    }

    /// Stalls by blocking the current thread until the GPU has executed the previous submission to
    /// this command buffer.
    ///
    /// See [`Self::has_executed`] to check without blocking.
    #[profiling::function]
    pub fn wait_until_executed(&mut self) -> Result<(), DriverError> {
        if self.droppables.is_empty() {
            return Ok(());
        }

        Device::wait_for_fence(&self.device, &self.fence)?;

        self.drop_fenced();

        Ok(())
    }
}

impl Drop for CommandBuffer {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        if self.wait_until_executed().is_err() {
            return;
        }

        unsafe {
            self.device
                .free_command_buffers(self.pool, slice::from_ref(&self.handle));
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_fence(self.fence, None);
        }
    }
}

/// Information used to create a [`CommandBuffer`] instance.
#[derive(Builder, Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build", error = "UninitializedFieldError"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct CommandBufferInfo {
    /// Designates a queue family as described in section
    /// [Queue Family Properties](https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html#devsandqueues-queueprops).
    /// All command buffers allocated from this command pool must be submitted on queues from the
    /// same queue family
    pub queue_family_index: u32,
}

impl CommandBufferInfo {
    /// TODO
    pub fn new(queue_family_index: u32) -> Self {
        Self { queue_family_index }
    }
}

impl CommandBufferInfo {
    /// Creates a default `CommandBufferInfoBuilder`.
    pub fn builder() -> CommandBufferInfoBuilder {
        Default::default()
    }

    /// Converts a `CommandBufferInfo` into a `CommandBufferInfoBuilder`.
    pub fn into_builder(self) -> CommandBufferInfoBuilder {
        CommandBufferInfoBuilder {
            queue_family_index: Some(self.queue_family_index),
        }
    }

    #[deprecated = "use into_builder function"]
    #[doc(hidden)]
    pub fn to_builder(self) -> CommandBufferInfoBuilder {
        self.into_builder()
    }
}

impl From<CommandBufferInfoBuilder> for CommandBufferInfo {
    fn from(info: CommandBufferInfoBuilder) -> Self {
        info.build()
    }
}

impl CommandBufferInfoBuilder {
    /// Builds a new `CommandBufferInfo`.
    #[inline(always)]
    pub fn build(self) -> CommandBufferInfo {
        self.fallible_build().unwrap()
    }
}
