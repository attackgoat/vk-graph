//! Fence types.

use {
    super::{DriverError, device::Device},
    crate::submission::TimestampQueryPool,
    ash::vk,
    log::{error, trace, warn},
    std::{
        any::Any,
        cell::{Cell, RefCell},
        fmt::Debug,
        panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
        thread::panicking,
    },
};

pub(crate) trait FenceDroppable: Debug + Send {
    fn fence_signaled(&mut self, _fence: &Fence) {}
}

pub(crate) type FencePayloads = Vec<Box<dyn FenceDroppable + 'static>>;

pub(crate) fn drop_fence_payloads(payloads: FencePayloads) -> bool {
    let mut all_succeeded = true;

    for payload in payloads {
        if catch_unwind(AssertUnwindSafe(|| drop(payload))).is_err() {
            all_succeeded = false;
            error!("fence payload panicked while being dropped");
        }
    }

    all_succeeded
}

fn run_all_catching_unwind<T>(
    values: &mut [T],
    mut operation: impl FnMut(&mut T),
) -> Option<Box<dyn Any + Send>> {
    let mut first_panic = None;

    for value in values {
        if let Err(panic) = catch_unwind(AssertUnwindSafe(|| operation(value)))
            && first_panic.is_none()
        {
            first_panic = Some(panic);
        }
    }

    first_panic
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

    payloads: RefCell<FencePayloads>,
    pub(crate) queued: Cell<bool>,

    /// Timestamp query results for queued work once this fence has signaled.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub timestamps: TimestampQueryPool,
}

impl Fence {
    /// Creates a Vulkan fence owned by `device`.
    ///
    /// See [`vkCreateFence`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateFence.html).
    pub fn create(device: &Device, signaled: bool) -> Result<Self, DriverError> {
        Ok(Self {
            device: device.clone(),
            handle: Device::create_fence(device, signaled)?,
            payloads: Default::default(),
            queued: Cell::new(signaled),
            timestamps: TimestampQueryPool::empty(),
        })
    }

    /// Drops an item after this fence signals.
    pub(crate) fn drop_when_signaled(&self, x: impl Debug + Send + 'static) {
        self.payloads.borrow_mut().push(Box::new(DeferredDrop(x)));
    }

    pub(crate) fn drop_fence_droppable(&self, x: impl FenceDroppable + 'static) {
        self.payloads.borrow_mut().push(Box::new(x));
    }

    #[profiling::function]
    fn drop_signaled(&self) {
        if !Device::background_fence_cleanup_enabled(&self.device) {
            let mut payloads = self.payloads.borrow_mut();

            if !payloads.is_empty() {
                trace!("releasing {} fence payloads", payloads.len());
            }

            for payload in payloads.iter_mut() {
                payload.fence_signaled(self);
            }

            payloads.clear();

            return;
        }

        let mut payloads = self.payloads.take();

        if !payloads.is_empty() {
            trace!("releasing {} fence payloads", payloads.len());
        } else {
            return;
        }

        let panic = run_all_catching_unwind(&mut payloads, |payload| {
            payload.fence_signaled(self);
        });

        if let Err(payloads) = Device::try_enqueue_fence_cleanup(&self.device, payloads) {
            drop_fence_payloads(payloads);
        }

        if let Some(panic) = panic {
            resume_unwind(panic);
        }
    }

    #[deprecated = "use status"]
    #[doc(hidden)]
    pub fn is_signaled(&self) -> Result<bool, DriverError> {
        self.status()
    }

    pub(crate) fn set_timestamps(&mut self, timestamps: TimestampQueryPool) {
        self.timestamps = timestamps;
    }

    /// Returns `true` if this fence is signaled.
    ///
    /// Fence-payload completion hooks run before this returns `Ok(true)`. Payload destruction may
    /// continue asynchronously when background fence cleanup is enabled on the device.
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
    /// If queued work has already signaled, fence-payload completion hooks run before the fence is
    /// reset. Payload destruction may continue asynchronously when background fence cleanup is
    /// enabled.
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
        self.timestamps = TimestampQueryPool::empty();

        Ok(self)
    }

    #[deprecated = "use wait"]
    #[doc(hidden)]
    pub fn wait_signaled(&mut self) -> Result<&mut Self, DriverError> {
        self.wait()
    }

    /// Waits for this fence to signal, then runs fence-payload completion hooks.
    ///
    /// Payload destruction may continue asynchronously when background fence cleanup is enabled on
    /// the device.
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

        if self.queued.get()
            && let Err(err) = self.wait()
        {
            warn!("unable to wait for fence during drop: {err}");

            return;
        }

        unsafe {
            self.device.destroy_fence(self.handle, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use {
        super::{FenceDroppable, drop_fence_payloads, run_all_catching_unwind},
        std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    #[derive(Debug)]
    struct CountDrop(Arc<AtomicUsize>);

    #[derive(Debug)]
    struct PanicDrop;

    impl FenceDroppable for CountDrop {}

    impl Drop for CountDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl FenceDroppable for PanicDrop {}

    impl Drop for PanicDrop {
        fn drop(&mut self) {
            panic!("expected test panic");
        }
    }

    #[test]
    fn payload_drop_isolates_each_panic() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let payloads: Vec<Box<dyn FenceDroppable>> = vec![
            Box::new(PanicDrop),
            Box::new(PanicDrop),
            Box::new(CountDrop(Arc::clone(&dropped))),
        ];

        assert!(!drop_fence_payloads(payloads));
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn completion_hooks_continue_after_panic() {
        let calls = AtomicUsize::new(0);
        let mut should_panic = [true, false];

        let panic = run_all_catching_unwind(&mut should_panic, |should_panic| {
            calls.fetch_add(1, Ordering::Relaxed);

            if *should_panic {
                panic!("expected test panic");
            }
        });

        assert!(panic.is_some());
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }
}
