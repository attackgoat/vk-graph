//! Logical device types

use {
    super::{
        DriverError,
        instance::{Instance, InstanceInfoBuilder},
        physical_device::PhysicalDevice,
    },
    ash::{khr, vk},
    derive_builder::Builder,
    gpu_allocator::{
        AllocatorDebugSettings,
        vulkan::{Allocator, AllocatorCreateDesc},
    },
    log::{error, info, trace, warn},
    raw_window_handle::HasDisplayHandle,
    std::{
        fmt::{Debug, Formatter},
        mem::{ManuallyDrop, forget},
        ops::Deref,
        slice,
        sync::Arc,
        thread::panicking,
        time::Instant,
    },
};

#[cfg(feature = "parking_lot")]
use parking_lot::Mutex;

#[cfg(not(feature = "parking_lot"))]
use std::sync::Mutex;

fn select_physical_device(
    instance: &Instance,
    mut index: usize,
) -> Result<PhysicalDevice, DriverError> {
    let mut physical_devices = Instance::physical_devices(instance)?
        .into_iter()
        .collect::<Vec<_>>();
    if physical_devices.is_empty() {
        warn!("unable to find physical devices");

        return Err(DriverError::Unsupported);
    }

    if index >= physical_devices.len() {
        index = 0;
    }

    let physical_device = physical_devices.remove(index);

    Ok(physical_device)
}

/// Opaque handle to a device object.
#[read_only::embed]
#[derive(Clone)]
pub struct Device {
    #[readonly]
    pub(self) inner: Arc<DeviceInner>,

    /// The physical device, which contains useful data about features, properties, and limits.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub physical_device: Box<PhysicalDevice>,
}

impl Device {
    /// Begins recording a command buffer on this device.
    ///
    /// This is a thin wrapper around [`ash::Device::begin_command_buffer`] that maps Vulkan errors
    /// to [`DriverError`] variants.
    pub fn begin_command_buffer(
        this: &Self,
        cmd: vk::CommandBuffer,
        begin_info: &vk::CommandBufferBeginInfo,
    ) -> Result<(), DriverError> {
        unsafe {
            this.begin_command_buffer(cmd, begin_info).map_err(|err| {
                warn!("unable to begin command buffer: {err}");

                match err {
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                    | vk::Result::ERROR_OUT_OF_HOST_MEMORY => DriverError::OutOfMemory,
                    _ => DriverError::Unsupported,
                }
            })
        }
    }

    /// Constructs a new device using the given configuration.
    ///
    /// This constructor is intended for headless or manually managed setups. It does not infer or
    /// enable display platform surface extensions. Use [`Self::try_from_display`] when the
    /// resulting device must be capable of later surface creation.
    #[profiling::function]
    pub fn create(info: impl Into<DeviceInfo>) -> Result<Self, DriverError> {
        let DeviceInfo {
            debug,
            physical_device_index,
        } = info.into();
        let instance_info = InstanceInfoBuilder::default().debug(debug);
        let instance = Instance::create(instance_info)?;
        let physical_device = select_physical_device(&instance, physical_device_index)?;

        Self::try_from_physical_device(physical_device)
    }

    pub(crate) fn create_fence(this: &Self, signaled: bool) -> Result<vk::Fence, DriverError> {
        let mut flags = vk::FenceCreateFlags::empty();

        if signaled {
            flags |= vk::FenceCreateFlags::SIGNALED;
        }

        let create_info = vk::FenceCreateInfo::default().flags(flags);
        let allocation_callbacks = None;

        unsafe { this.create_fence(&create_info, allocation_callbacks) }.map_err(|err| {
            warn!("unable to create fence: {err}");

            DriverError::OutOfMemory
        })
    }

    /// Ends recording a command buffer on this device.
    ///
    /// This is a thin wrapper around [`ash::Device::end_command_buffer`] that maps Vulkan errors
    /// to [`DriverError`] variants.
    pub fn end_command_buffer(this: &Self, cmd: vk::CommandBuffer) -> Result<(), DriverError> {
        unsafe {
            this.end_command_buffer(cmd).map_err(|err| {
                warn!("unable to end command buffer: {err}");

                match err {
                    vk::Result::ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR => DriverError::InvalidData,
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                    | vk::Result::ERROR_OUT_OF_HOST_MEMORY => DriverError::OutOfMemory,
                    _ => DriverError::Unsupported,
                }
            })
        }
    }

    /// Helper for times when you already know that the device supports the acceleration
    /// structure extension.
    ///
    /// # Panics
    ///
    /// Panics if [Self.physical_device.accel_struct_properties] is `None`.
    pub(crate) fn expect_vk_khr_acceleration_structure(
        this: &Self,
    ) -> &khr::acceleration_structure::Device {
        this.inner
            .vk_khr_acceleration_structure
            .as_ref()
            .expect("missing VK_KHR_acceleration_structure")
    }

    /// Helper for times when you already know that the device supports the ray tracing pipeline
    /// extension.
    ///
    /// # Panics
    ///
    /// Panics if [Self.physical_device.ray_tracing_features] is `None`.
    pub(crate) fn expect_vk_khr_ray_tracing_pipeline(
        this: &Self,
    ) -> &khr::ray_tracing_pipeline::Device {
        this.inner
            .vk_khr_ray_tracing_pipeline
            .as_ref()
            .expect("missing VK_KHR_ray_tracing_pipeline")
    }

    /// Helper for times when you already know that the instance supports the surface extension.
    ///
    /// # Panics
    ///
    /// Panics if the device was not created for display window access.
    pub(crate) fn expect_vk_khr_surface(this: &Self) -> &khr::surface::Instance {
        this.inner
            .vk_khr_surface
            .as_ref()
            .expect("missing VK_KHR_surface")
    }

    /// Helper for times when you already know that the device supports the swapchain extension.
    ///
    /// # Panics
    ///
    /// Panics if the device was not created for display window access.
    pub(crate) fn expect_vk_khr_swapchain(this: &Self) -> &khr::swapchain::Device {
        this.inner
            .vk_khr_swapchain
            .as_ref()
            .expect("missing VK_KHR_swapchain")
    }

    pub(crate) fn pipeline_cache(this: &Self) -> vk::PipelineCache {
        this.inner.pipeline_cache
    }

    /// Submits command buffers to a queue, optionally signaling a fence.
    pub fn queue_submit(
        this: &Self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> Result<(), DriverError> {
        unsafe {
            this.queue_submit(queue, submits, fence).map_err(|err| {
                warn!("unable to queue submits: {err}");

                match err {
                    vk::Result::ERROR_DEVICE_LOST => DriverError::InvalidData,
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                    | vk::Result::ERROR_OUT_OF_HOST_MEMORY => DriverError::OutOfMemory,
                    _ => DriverError::Unsupported,
                }
            })
        }
    }

    /// Resets one or more fences to the unsignaled state.
    pub fn reset_fences(this: &Self, fences: &[vk::Fence]) -> Result<(), DriverError> {
        unsafe {
            this.reset_fences(fences).map_err(|err| {
                warn!("unable to reset fences: {err}");

                match err {
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => DriverError::OutOfMemory,
                    _ => DriverError::Unsupported,
                }
            })
        }
    }

    /// Loads and existing `ash` Vulkan device that may have been created by other means.
    #[profiling::function]
    pub fn try_from_ash(
        device: ash::Device,
        physical_device: PhysicalDevice,
    ) -> Result<Self, DriverError> {
        let debug = physical_device.instance.info.debug;
        let mut debug_settings = AllocatorDebugSettings::default();
        debug_settings.log_leaks_on_shutdown = debug;
        debug_settings.log_memory_information = debug;
        debug_settings.log_allocations = debug;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: (*physical_device.instance).clone(),
            device: device.clone(),
            physical_device: physical_device.handle,
            debug_settings,
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })
        .map_err(|err| {
            warn!("unable to create allocator: {err}");

            DriverError::Unsupported
        })?;

        let mut queues = Vec::with_capacity(physical_device.queue_families.len());

        for (queue_family_index, properties) in physical_device.queue_families.iter().enumerate() {
            let mut queue_family = Vec::with_capacity(properties.queue_count as _);

            for queue_index in 0..properties.queue_count {
                queue_family.push(Mutex::new(unsafe {
                    device.get_device_queue(queue_family_index as _, queue_index)
                }));
            }

            queues.push(queue_family.into_boxed_slice());
        }

        let vk_khr_surface = physical_device.khr_swapchain.then(|| {
            let entry = Instance::entry(&physical_device.instance);
            khr::surface::Instance::new(entry, &physical_device.instance)
        });
        let vk_khr_swapchain = physical_device
            .khr_swapchain
            .then(|| khr::swapchain::Device::new(&physical_device.instance, &device));
        let vk_khr_acceleration_structure = physical_device
            .accel_struct_properties
            .is_some()
            .then(|| khr::acceleration_structure::Device::new(&physical_device.instance, &device));
        let vk_khr_ray_tracing_pipeline = physical_device
            .ray_tracing_pipeline_features
            .ray_tracing_pipeline
            .then(|| khr::ray_tracing_pipeline::Device::new(&physical_device.instance, &device));

        let pipeline_cache =
            unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }
                .map_err(|err| {
                    warn!("unable to create pipeline cache: {err}");

                    DriverError::Unsupported
                })?;

        Ok(Self {
            read_only: ReadOnlyDevice {
                inner: Arc::new(DeviceInner {
                    allocator: ManuallyDrop::new(Mutex::new(allocator)),
                    device,
                    pipeline_cache,
                    queues: queues.into_boxed_slice(),
                    vk_khr_acceleration_structure,
                    vk_khr_ray_tracing_pipeline,
                    vk_khr_surface,
                    vk_khr_swapchain,
                }),
                physical_device: Box::new(physical_device),
            },
        })
    }

    /// Constructs a new device using the given configuration.
    #[profiling::function]
    pub fn try_from_display(
        display: impl HasDisplayHandle,
        info: impl Into<DeviceInfo>,
    ) -> Result<Self, DriverError> {
        let DeviceInfo {
            debug,
            physical_device_index,
        } = info.into();
        let instance_info = InstanceInfoBuilder::default().debug(debug);
        let instance = Instance::try_from_display(display, instance_info)?;
        let physical_device = select_physical_device(&instance, physical_device_index)?;

        Self::try_from_physical_device(physical_device)
    }

    /// Constructs a new device using the given physical device.
    #[profiling::function]
    pub fn try_from_physical_device(physical_device: PhysicalDevice) -> Result<Self, DriverError> {
        let device = unsafe {
            physical_device.create_ash_device(|device_create_info| {
                physical_device.instance.create_device(
                    physical_device.handle,
                    &device_create_info,
                    None,
                )
            })
        }
        .map_err(|err| {
            error!("unable to create device: {err}");

            DriverError::Unsupported
        })?;

        info!("created {}", physical_device.properties_v1_0.device_name);

        Self::try_from_ash(device, physical_device)
    }

    #[profiling::function]
    pub(crate) fn wait_for_fence(this: &Self, fence: &vk::Fence) -> Result<(), DriverError> {
        Device::wait_for_fences(this, slice::from_ref(fence))
    }

    #[profiling::function]
    pub(crate) fn wait_for_fences(this: &Self, fences: &[vk::Fence]) -> Result<(), DriverError> {
        unsafe {
            match this.wait_for_fences(fences, true, 100) {
                Ok(_) => return Ok(()),
                Err(err) if err == vk::Result::ERROR_DEVICE_LOST => {
                    error!("invalid device state: lost");

                    return Err(DriverError::InvalidData);
                }
                Err(err) if err == vk::Result::TIMEOUT => {
                    trace!("waiting...");
                }
                Err(err) => {
                    warn!("unable to wait for fences during polling phase: {err}");

                    return Err(DriverError::OutOfMemory);
                }
            }

            let started = Instant::now();

            match this.wait_for_fences(fences, true, u64::MAX) {
                Ok(_) => (),
                Err(err) if err == vk::Result::ERROR_DEVICE_LOST => {
                    error!("invalid device state: lost");

                    return Err(DriverError::InvalidData);
                }
                Err(err) => {
                    warn!("unable to wait for fences to completion: {err}");

                    return Err(DriverError::OutOfMemory);
                }
            }

            let elapsed = Instant::now() - started;
            let elapsed_millis = elapsed.as_millis();

            if elapsed_millis > 0 {
                warn!("slow fence wait: {} ms", elapsed_millis);
            }
        }

        Ok(())
    }

    pub(crate) fn with_allocator<R>(this: &Self, f: impl FnOnce(&mut Allocator) -> R) -> R {
        let allocator = this.inner.allocator.lock();

        #[cfg(not(feature = "parking_lot"))]
        let allocator = allocator.expect("poisoned allocator lock");

        let mut allocator = allocator;

        f(&mut allocator)
    }

    /// Provides locked access to a device queue.
    ///
    /// Acquires the mutex for the queue at the given family and index, calls `f` with the
    /// [`vk::Queue`], and releases the mutex after `f` returns.
    ///
    /// # Panics
    ///
    /// Panics if `queue_family_index` or `queue_index` is out of range for this device.
    pub fn with_queue<R>(
        this: &Self,
        queue_family_index: u32,
        queue_index: u32,
        f: impl FnOnce(vk::Queue) -> R,
    ) -> R {
        let queue_family = this
            .inner
            .queues
            .get(queue_family_index as usize)
            .expect("invalid queue family index");
        let queue = queue_family
            .get(queue_index as usize)
            .expect("invalid queue index");
        #[cfg(not(feature = "parking_lot"))]
        let guard = queue.lock().expect("poisoned queue lock");

        #[cfg(feature = "parking_lot")]
        let guard = queue.lock();

        f(*guard)
    }
}

impl Debug for Device {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Device")
    }
}

#[cfg(doc)]
impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        unreachable!()
    }
}

impl Eq for Device {}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

/// Information used to create a [`Device`] instance.
#[derive(Builder, Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[builder(
    build_fn(private, name = "fallible_build"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct DeviceInfo {
    /// Enables Vulkan validation layers.
    ///
    /// This requires a Vulkan SDK installation and will cause validation errors to introduce
    /// panics as they happen.
    ///
    /// When `stderr` is attached to an interactive terminal, validation errors will park the
    /// callback thread for debugger attach.
    ///
    /// _NOTE:_ Consider turning OFF debug if you discover an unknown issue. Often the validation
    /// layers will throw an error before other layers can provide additional context such as the
    /// API dump info or other messages. You might find the "actual" issue is detailed in those
    /// subsequent details.
    ///
    /// ## Platform-specific
    ///
    /// **macOS:** Has no effect unless the `loaded` feature is enabled.
    #[builder(default)]
    pub debug: bool,

    /// Index of the [`PhysicalDevice`] from the available devices. See
    /// [`Instance::physical_devices`].
    #[builder(default)]
    pub physical_device_index: usize,
}

impl DeviceInfo {
    /// Creates a default `DeviceInfoBuilder`.
    pub fn builder() -> DeviceInfoBuilder {
        Default::default()
    }

    /// Converts a `DeviceInfo` into a `DeviceInfoBuilder`.
    pub fn into_builder(self) -> DeviceInfoBuilder {
        DeviceInfoBuilder {
            debug: Some(self.debug),
            physical_device_index: Some(self.physical_device_index),
        }
    }
}

impl From<DeviceInfoBuilder> for DeviceInfo {
    fn from(info: DeviceInfoBuilder) -> Self {
        info.build()
    }
}

impl DeviceInfoBuilder {
    /// Builds a new `DeviceInfo`.
    #[inline(always)]
    pub fn build(self) -> DeviceInfo {
        self.fallible_build().expect("invalid device info")
    }
}

struct DeviceInner {
    allocator: ManuallyDrop<Mutex<Allocator>>,
    device: ash::Device,
    pipeline_cache: vk::PipelineCache,
    queues: Box<[Box<[Mutex<vk::Queue>]>]>,
    vk_khr_acceleration_structure: Option<khr::acceleration_structure::Device>,
    vk_khr_ray_tracing_pipeline: Option<khr::ray_tracing_pipeline::Device>,
    vk_khr_surface: Option<khr::surface::Instance>,
    vk_khr_swapchain: Option<khr::swapchain::Device>,
}

impl Drop for DeviceInner {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            // When panicking we don't want the GPU allocator to complain about leaks
            unsafe {
                forget(ManuallyDrop::take(&mut self.allocator));
            }

            return;
        }

        // trace!("drop");

        if let Err(err) = unsafe { self.device.device_wait_idle() } {
            warn!("device_wait_idle() failed: {err}");
        }

        unsafe {
            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);

            ManuallyDrop::drop(&mut self.allocator);
        }

        unsafe {
            self.device.destroy_device(None);
        }
    }
}

#[doc(hidden)]
impl Clone for ReadOnlyDevice {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            physical_device: self.physical_device.clone(),
        }
    }
}

#[doc(hidden)]
impl Deref for ReadOnlyDevice {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.inner.device
    }
}

#[cfg(test)]
mod test {
    use super::*;

    type Info = DeviceInfo;
    type Builder = DeviceInfoBuilder;

    #[test]
    pub fn device_info() {
        Info::default().into_builder().build();
    }

    #[test]
    pub fn device_info_builder() {
        Builder::default().build();
    }
}
