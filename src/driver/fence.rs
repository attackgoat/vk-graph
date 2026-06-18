//! Fence types.

use {
    super::{DriverError, device::Device},
    ash::vk,
    log::{error, trace},
    std::{fmt::Debug, thread::panicking},
};

/// Represents a Vulkan fence used to track queue submission completion.
///
/// See [`VkFence`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkFence.html).
#[derive(Debug)]
#[read_only::cast]
pub struct Fence {
    /// The device which owns this fence resource.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub device: Device,

    /// The native Vulkan fence handle.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub handle: vk::Fence,

    pub(crate) queued: bool,
    droppables: Vec<Box<dyn Debug + Send + 'static>>,
}

impl Fence {
    /// Creates a Vulkan fence owned by `device`.
    ///
    /// See [`vkCreateFence`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateFence.html).
    pub fn create(device: &Device, signaled: bool) -> Result<Self, DriverError> {
        Ok(Self {
            device: device.clone(),
            handle: Device::create_fence(device, signaled)?,
            queued: signaled,
            droppables: Vec::new(),
        })
    }

    /// Drops an item after this fence signals.
    pub(crate) fn drop_when_signaled(&mut self, x: impl Debug + Send + 'static) {
        self.droppables.push(Box::new(x));
    }

    #[profiling::function]
    fn drop_signaled(&mut self) {
        if !self.droppables.is_empty() {
            trace!("dropping {} shared references", self.droppables.len());
        }

        self.droppables.clear();
    }

    /// Returns `true` if this fence is signaled.
    ///
    /// See [`vkGetFenceStatus`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetFenceStatus.html).
    #[profiling::function]
    pub fn is_signaled(&self) -> Result<bool, DriverError> {
        let res = unsafe { self.device.get_fence_status(self.handle) };

        match res {
            Ok(status) => Ok(status),
            Err(err) if err == vk::Result::ERROR_DEVICE_LOST => {
                error!("invalid device state: lost");

                Err(DriverError::InvalidData)
            }
            Err(err) => {
                error!("unable to get fence status: {err}");

                Err(DriverError::InvalidData)
            }
        }
    }

    /// Returns `true` if work has been queued against this fence.
    pub fn is_queued(&self) -> bool {
        self.queued
    }

    /// Marks this fence as having work queued against it.
    pub(crate) fn mark_queued(&mut self) {
        self.queued = true;
    }

    /// Resets this fence to the unsignaled state.
    ///
    /// See [`vkResetFences`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkResetFences.html).
    pub fn reset(&mut self) -> Result<&mut Self, DriverError> {
        #[cfg(feature = "checked")]
        if !self.queued {
            return Ok(self);
        }

        Device::reset_fences(&self.device, std::slice::from_ref(&self.handle))?;
        self.queued = false;

        Ok(self)
    }

    /// Waits for this fence to signal, then drops any deferred payloads.
    ///
    /// See [`vkWaitForFences`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkWaitForFences.html).
    #[profiling::function]
    pub fn wait_signaled(&mut self) -> Result<&mut Self, DriverError> {
        #[cfg(feature = "checked")]
        if !self.queued {
            return Ok(self);
        }

        Device::wait_for_fence(&self.device, &self.handle)?;
        self.drop_signaled();

        Ok(self)
    }
}

impl Drop for Fence {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        if self.queued && self.wait_signaled().is_err() {
            return;
        }

        unsafe {
            self.device.destroy_fence(self.handle, None);
        }
    }
}
