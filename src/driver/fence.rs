//! Fence types.

use {
    super::{DriverError, device::Device},
    ash::vk,
    log::{error, trace},
    std::{cell::Cell, cell::RefCell, fmt::Debug, thread::panicking},
};

pub(crate) trait FenceDroppable: Debug + Send {
    fn fence_signaled(&mut self) {}
}

#[derive(Debug)]
struct DeferredDrop<T>(T);

impl<T> FenceDroppable for DeferredDrop<T> where T: Debug + Send {}

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

    pub(crate) queued: Cell<bool>,
    droppables: RefCell<Vec<Box<dyn FenceDroppable + 'static>>>,
}

impl Fence {
    /// Creates a Vulkan fence owned by `device`.
    ///
    /// See [`vkCreateFence`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateFence.html).
    pub fn create(device: &Device, signaled: bool) -> Result<Self, DriverError> {
        Ok(Self {
            device: device.clone(),
            handle: Device::create_fence(device, signaled)?,
            queued: Cell::new(signaled),
            droppables: RefCell::new(Vec::new()),
        })
    }

    /// Drops an item after this fence signals.
    pub(crate) fn drop_when_signaled(&self, x: impl Debug + Send + 'static) {
        self.droppables.borrow_mut().push(Box::new(DeferredDrop(x)));
    }

    pub(crate) fn drop_fence_droppable(&self, x: impl FenceDroppable + 'static) {
        self.droppables.borrow_mut().push(Box::new(x));
    }

    #[profiling::function]
    fn drop_signaled(&self) {
        let mut droppables = self.droppables.borrow_mut();

        if !droppables.is_empty() {
            trace!("dropping {} shared references", droppables.len());
        }

        for droppable in droppables.iter_mut() {
            droppable.fence_signaled();
        }

        droppables.clear();
    }

    #[deprecated = "use status"]
    #[doc(hidden)]
    pub fn is_signaled(&self) -> Result<bool, DriverError> {
        self.status()
    }

    /// Returns `true` if this fence is signaled.
    ///
    /// Signaled deferred payloads are released before this returns `Ok(true)`.
    ///
    /// See [`vkGetFenceStatus`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetFenceStatus.html).
    #[profiling::function]
    pub fn status(&self) -> Result<bool, DriverError> {
        let res = unsafe { self.device.get_fence_status(self.handle) };

        match res {
            Ok(status) => {
                if status {
                    self.drop_signaled();
                }

                Ok(status)
            }
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
        self.queued.get()
    }

    /// Marks this fence as having work queued against it.
    pub(crate) fn mark_queued(&mut self) {
        self.queued.set(true);
    }

    /// Resets this fence to the unsignaled state.
    ///
    /// If queued work has already signaled, deferred payloads are released before the fence is
    /// reset.
    ///
    /// See [`vkResetFences`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkResetFences.html).
    pub fn reset(&mut self) -> Result<&mut Self, DriverError> {
        #[cfg(feature = "checked")]
        if !self.queued.get() {
            return Ok(self);
        }

        if self.status()? {
            Device::reset_fences(&self.device, std::slice::from_ref(&self.handle))?;
        }

        self.queued.set(false);

        Ok(self)
    }

    #[deprecated = "use wait"]
    #[doc(hidden)]
    pub fn wait_signaled(&mut self) -> Result<&mut Self, DriverError> {
        self.wait()
    }

    /// Waits for this fence to signal, then releases deferred payloads.
    ///
    /// See [`vkWaitForFences`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkWaitForFences.html).
    #[profiling::function]
    pub fn wait(&mut self) -> Result<&mut Self, DriverError> {
        #[cfg(feature = "checked")]
        if !self.queued.get() {
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

        if self.queued.get() && self.wait().is_err() {
            return;
        }

        unsafe {
            self.device.destroy_fence(self.handle, None);
        }
    }
}
