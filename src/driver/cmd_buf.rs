//! Command buffer types

use {
    super::{DriverError, device::Device},
    ash::vk,
    derive_builder::Builder,
    log::{error, trace, warn},
    std::{fmt::Debug, slice, thread::panicking},
};

// TODO: Expose command functions so the fence, device, waiting flags do not
// need to be public

/// Represents a Vulkan command buffer to which some work has been submitted.
#[derive(Debug)]
#[read_only::cast]
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
    /// Begins recording this command buffer.
    ///
    /// This is a thin wrapper around [`ash::Device::begin_command_buffer`] that maps Vulkan errors
    /// to [`DriverError`] variants.
    pub fn begin(&self, info: &vk::CommandBufferBeginInfo) -> Result<(), DriverError> {
        Device::begin_command_buffer(&self.device, self.handle, info)
    }

    /// Creates a command buffer allocation backed by a transient resettable command pool.
    #[profiling::function]
    pub fn create(
        device: &Device,
        info: impl Into<CommandBufferInfo>,
    ) -> Result<Self, DriverError> {
        let info = info.into();
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

            match err {
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                    DriverError::OutOfMemory
                }
                _ => DriverError::Unsupported,
            }
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

            match err {
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                    DriverError::OutOfMemory
                }
                _ => DriverError::Unsupported,
            }
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

    /// Ends recording this command buffer.
    ///
    /// This is a thin wrapper around [`ash::Device::end_command_buffer`] that maps Vulkan errors
    /// to [`DriverError`] variants.
    pub fn end(&self) -> Result<(), DriverError> {
        Device::end_command_buffer(&self.device, self.handle)
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
                error!("invalid device state: lost");

                Err(DriverError::InvalidData)
            }
            Err(err) => {
                // VK_SUCCESS and VK_NOT_READY handled by get_fence_status in ash
                // VK_ERROR_DEVICE_LOST already handled above, so no idea what happened
                error!("unable to get fence status: {err}");

                Err(DriverError::InvalidData)
            }
        }
    }

    /// Resets the embedded fence to the unsignaled state.
    pub fn reset_fence(&self) -> Result<(), DriverError> {
        Device::reset_fences(&self.device, slice::from_ref(&self.fence))
    }

    /// Submits command buffers to a queue.
    pub fn queue_submit(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo],
    ) -> Result<(), DriverError> {
        Device::queue_submit(&self.device, queue, submits, self.fence)
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
        self.reset_fence()?;
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
    build_fn(private, name = "fallible_build"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct CommandBufferInfo {
    /// Designates the queue family used by the command pool that allocates this command buffer.
    ///
    /// See [`VkCommandPoolCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandPoolCreateInfo.html).
    #[builder(default)]
    pub queue_family_index: u32,
}

impl CommandBufferInfo {
    /// Creates command buffer allocation info for the given queue family.
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
        self.fallible_build().expect("invalid command buffer info")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    type Info = CommandBufferInfo;
    type Builder = CommandBufferInfoBuilder;

    #[test]
    pub fn command_buffer_info() {
        let info = Info::new(3);
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn command_buffer_info_builder_default_queue_family_index() {
        assert_eq!(Builder::default().build(), Info::new(0));
    }
}
