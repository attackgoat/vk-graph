//! Command buffer types

use {
    super::{DriverError, device::Device, fence::Fence},
    ash::vk::{self, Handle as _},
    derive_builder::Builder,
    log::warn,
    std::{
        fmt::{Debug, Formatter},
        slice,
        thread::panicking,
    },
};

// TODO: Expose command functions so the fence, device, waiting flags do not
// need to be public

/// Represents a Vulkan command buffer allocation.
///
/// See [`VkCommandBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandBuffer.html).
#[read_only::cast]
pub struct CommandBuffer {
    /// The device which owns this command buffer resource.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub device: Device,

    /// The native Vulkan resource handle of this command buffer.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub handle: vk::CommandBuffer,

    /// Information used to create this object.
    #[readonly]
    pub info: CommandBufferInfo,

    pub(crate) pool: vk::CommandPool,
    release_semaphore: Option<vk::Semaphore>,
}

impl CommandBuffer {
    /// Begins recording this command buffer.
    ///
    /// This is a thin wrapper around [`ash::Device::begin_command_buffer`] that maps Vulkan errors
    /// to [`DriverError`] variants.
    ///
    /// See [`vkBeginCommandBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkBeginCommandBuffer.html).
    pub fn begin(&self, info: &vk::CommandBufferBeginInfo) -> Result<(), DriverError> {
        Device::begin_command_buffer(&self.device, self.handle, info)
    }

    /// Creates a command buffer allocation backed by a transient resettable command pool.
    ///
    /// See [`vkAllocateCommandBuffers`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkAllocateCommandBuffers.html).
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
        })?
        .into_iter()
        .find(|handle| !handle.is_null())
        .ok_or_else(|| {
            warn!("missing command buffer handle");

            DriverError::Unsupported
        })?;

        Ok(Self {
            device,
            handle,
            info,
            pool,
            release_semaphore: None,
        })
    }

    /// Ends recording this command buffer.
    ///
    /// This is a thin wrapper around [`ash::Device::end_command_buffer`] that maps Vulkan errors
    /// to [`DriverError`] variants.
    ///
    /// See [`vkEndCommandBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkEndCommandBuffer.html).
    pub fn end(&self) -> Result<(), DriverError> {
        Device::end_command_buffer(&self.device, self.handle)
    }

    /// Ends recording a render pass.
    pub fn end_render_pass(&self) {
        unsafe {
            self.device.cmd_end_render_pass(self.handle);
        }
    }

    /// Submits command buffers to a queue using `fence`.
    ///
    /// This method does not begin, end, or reset `self` or `fence`. Callers are expected to
    /// submit only executable command buffers and to manage fence waits and resets as needed.
    ///
    /// Typical handling is:
    ///
    /// 1. Begin recording with [`Self::begin`].
    /// 2. Record commands.
    /// 3. End recording with [`Self::end`].
    /// 4. Submit this command buffer with `queue_submit`.
    /// 5. Later, wait for completion with [`Fence::is_signaled`] or [`Fence::wait_signaled`].
    /// 6. Before re-submitting this same command buffer, reset the fence with [`Fence::reset`],
    ///    then begin recording again.
    ///
    /// See [`vkQueueSubmit`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueSubmit.html).
    pub fn queue_submit(
        &self,
        queue: vk::Queue,
        fence: &mut Fence,
        submits: &[vk::SubmitInfo],
    ) -> Result<(), DriverError> {
        Device::queue_submit(&self.device, queue, submits, fence.handle)?;
        fence.mark_queued();

        Ok(())
    }

    /// Submits command buffers to a queue using `vkQueueSubmit2` (Vulkan 1.3 core or
    /// `VK_KHR_synchronization2`).
    ///
    /// See [`vkQueueSubmit2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueSubmit2.html)
    /// and [`VK_KHR_synchronization2`](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_synchronization2.html).
    pub fn queue_submit2(
        &self,
        queue: vk::Queue,
        fence: &mut Fence,
        submits: &[vk::SubmitInfo2],
    ) -> Result<(), DriverError> {
        Device::queue_submit2(&self.device, queue, submits, fence.handle)?;
        fence.mark_queued();

        Ok(())
    }

    /// Returns a cached semaphore used to signal temporary queue-ownership release submissions.
    ///
    /// The semaphore is created lazily on first use and then reused with this command buffer for
    /// subsequent release submissions.
    pub(crate) fn release_semaphore(&mut self) -> Result<vk::Semaphore, DriverError> {
        if let Some(semaphore) = self.release_semaphore {
            return Ok(semaphore);
        }

        let semaphore = Device::create_semaphore(&self.device)?;

        Device::try_set_debug_utils_object_name(&self.device, semaphore, "queue ownership release");

        self.release_semaphore = Some(semaphore);

        Ok(semaphore)
    }

    /// Sets the debugging name assigned to this command buffer.
    pub fn set_debug_name(&self, name: impl AsRef<str>) {
        Device::try_set_debug_utils_object_name(&self.device, self.handle, &name);
        Device::try_set_private_data_object_name(
            &self.device,
            vk::ObjectType::COMMAND_BUFFER,
            self.handle,
            &name,
        );
    }

    /// Sets the debugging name assigned to this command buffer.
    pub fn with_debug_name(self, name: impl AsRef<str>) -> Self {
        self.set_debug_name(name);

        self
    }
}

impl AsRef<CommandBuffer> for CommandBuffer {
    fn as_ref(&self) -> &CommandBuffer {
        self
    }
}

impl Debug for CommandBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut res = f.debug_struct(stringify!(CommandBuffer));

        if let Some(debug_name) = &Device::private_data_object_name(
            &self.device,
            vk::ObjectType::COMMAND_BUFFER,
            self.handle,
        ) {
            res.field("debug_name", debug_name);
        }

        res.field("handle", &self.handle).finish_non_exhaustive()
    }
}

impl Drop for CommandBuffer {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        Device::try_clear_private_data_object_name(
            &self.device,
            vk::ObjectType::COMMAND_BUFFER,
            self.handle,
        );

        unsafe {
            if let Some(semaphore) = self.release_semaphore.take() {
                self.device.destroy_semaphore(semaphore, None);
            }

            self.device
                .free_command_buffers(self.pool, slice::from_ref(&self.handle));
            self.device.destroy_command_pool(self.pool, None);
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
