//! Logical device resource types

use {
    super::{
        DriverError,
        instance::{Instance, InstanceInfoBuilder},
        physical_device::PhysicalDevice,
    },
    ash::{ext, khr, vk},
    derive_builder::{Builder, UninitializedFieldError},
    gpu_allocator::{
        AllocatorDebugSettings,
        vulkan::{Allocator, AllocatorCreateDesc},
    },
    log::{error, info, trace, warn},
    raw_window_handle::{HasDisplayHandle, RawDisplayHandle},
    std::{
        ffi::CStr,
        fmt::{Debug, Formatter},
        mem::{ManuallyDrop, forget},
        ops::Deref,
        thread::panicking,
        time::Instant,
    },
};

#[cfg(feature = "parking_lot")]
use parking_lot::Mutex;

#[cfg(not(feature = "parking_lot"))]
use std::sync::Mutex;

// Copied from ash-window to change the signature
fn enumerate_required_extensions(
    display_handle: RawDisplayHandle,
) -> Result<&'static [&'static CStr], DriverError> {
    let extensions = match display_handle {
        RawDisplayHandle::Windows(_) => {
            const WINDOWS_EXTS: [&CStr; 2] = [khr::surface::NAME, khr::win32_surface::NAME];
            &WINDOWS_EXTS
        }
        RawDisplayHandle::Wayland(_) => {
            const WAYLAND_EXTS: [&CStr; 2] = [khr::surface::NAME, khr::wayland_surface::NAME];
            &WAYLAND_EXTS
        }
        RawDisplayHandle::Xlib(_) => {
            const XLIB_EXTS: [&CStr; 2] = [khr::surface::NAME, khr::xlib_surface::NAME];
            &XLIB_EXTS
        }
        RawDisplayHandle::Xcb(_) => {
            const XCB_EXTS: [&CStr; 2] = [khr::surface::NAME, khr::xcb_surface::NAME];
            &XCB_EXTS
        }
        RawDisplayHandle::Android(_) => {
            const ANDROID_EXTS: [&CStr; 2] = [khr::surface::NAME, khr::android_surface::NAME];
            &ANDROID_EXTS
        }
        RawDisplayHandle::AppKit(_) | RawDisplayHandle::UiKit(_) => {
            const METAL_EXTS: [&CStr; 2] = [khr::surface::NAME, ext::metal_surface::NAME];
            &METAL_EXTS
        }
        _ => return Err(DriverError::Unsupported),
    };

    Ok(extensions)
}

fn select_physical_device(
    instance: &Instance,
    mut index: usize,
) -> Result<PhysicalDevice, DriverError> {
    let mut physical_devices = Instance::physical_devices(instance)?;
    if physical_devices.is_empty() {
        warn!("no physical devices found");

        return Err(DriverError::Unsupported);
    }

    if index >= physical_devices.len() {
        index = 0;
    }

    let physical_device = physical_devices.remove(index);

    Ok(physical_device)
}

/// Opaque handle to a device object.
#[repr(C)]
pub struct Device {
    accel_struct_ext: Option<khr::acceleration_structure::Device>,

    pub(super) allocator: ManuallyDrop<Mutex<Allocator>>,

    device: ash::Device,
    pipeline_cache: vk::PipelineCache,

    /// The physical device, which contains useful data about features, properties, and limits.
    ///
    /// _Note:_ This field is read-only.
    #[cfg(doc)]
    pub physical_device: PhysicalDevice,

    #[cfg(not(doc))]
    physical_device: PhysicalDevice,

    /// The physical execution queues which all work will be submitted to.
    pub(crate) queues: Vec<Vec<vk::Queue>>,

    ray_trace_ext: Option<khr::ray_tracing_pipeline::Device>,
    surface_ext: Option<khr::surface::Instance>,
    swapchain_ext: Option<khr::swapchain::Device>,
}

#[doc(hidden)]
#[repr(C)]
pub struct DeviceRef {
    accel_struct_ext: Option<khr::acceleration_structure::Device>,
    pub(super) allocator: ManuallyDrop<Mutex<Allocator>>,
    device: ash::Device,
    pipeline_cache: vk::PipelineCache,
    pub physical_device: PhysicalDevice,
    pub(crate) queues: Vec<Vec<vk::Queue>>,
    ray_trace_ext: Option<khr::ray_tracing_pipeline::Device>,
    surface_ext: Option<khr::surface::Instance>,
    swapchain_ext: Option<khr::swapchain::Device>,
}

impl Device {
    /// Constructs a new device using the given physical device.
    #[profiling::function]
    pub fn create(
        physical_device: PhysicalDevice,
        display_window: bool,
    ) -> Result<Self, DriverError> {
        let device = unsafe {
            physical_device.create_ash_device(display_window, |device_create_info| {
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

        Self::load(physical_device, device, display_window)
    }

    /// Constructs a new device using the given configuration.
    #[profiling::function]
    pub fn create_headless(info: impl Into<DeviceInfo>) -> Result<Self, DriverError> {
        let DeviceInfo {
            debug,
            physical_device_index,
        } = info.into();
        let instance_info = InstanceInfoBuilder::default().debug(debug);
        let instance = Instance::load(instance_info)?;
        let physical_device = select_physical_device(&instance, physical_device_index)?;

        Self::create(physical_device, false)
    }

    /// Constructs a new device using the given configuration.
    #[profiling::function]
    pub fn create_display(
        info: impl Into<DeviceInfo>,
        display_handle: impl HasDisplayHandle,
    ) -> Result<Self, DriverError> {
        let DeviceInfo {
            debug,
            physical_device_index,
        } = info.into();
        let display_handle = display_handle.display_handle().map_err(|err| {
            warn!("unable to get display handle: {err}");

            DriverError::Unsupported
        })?;
        let extension_names =
            enumerate_required_extensions(display_handle.as_raw()).map_err(|err| {
                warn!("unable to enumerate window extensions: {err}");

                DriverError::Unsupported
            })?;
        let instance_info = InstanceInfoBuilder::default()
            .debug(debug)
            .extension_names(extension_names);
        let instance = Instance::load(instance_info)?;
        let physical_device = select_physical_device(&instance, physical_device_index)?;

        Self::create(physical_device, true)
    }

    pub(crate) fn create_fence(this: &Self, signaled: bool) -> Result<vk::Fence, DriverError> {
        let mut flags = vk::FenceCreateFlags::empty();

        if signaled {
            flags |= vk::FenceCreateFlags::SIGNALED;
        }

        let create_info = vk::FenceCreateInfo::default().flags(flags);
        let allocation_callbacks = None;

        unsafe { this.create_fence(&create_info, allocation_callbacks) }.map_err(|err| {
            warn!("{err}");

            DriverError::OutOfMemory
        })
    }

    pub(crate) fn create_semaphore(this: &Self) -> Result<vk::Semaphore, DriverError> {
        let create_info = vk::SemaphoreCreateInfo::default();
        let allocation_callbacks = None;

        unsafe { this.create_semaphore(&create_info, allocation_callbacks) }.map_err(|err| {
            warn!("{err}");

            DriverError::OutOfMemory
        })
    }

    /// Helper for times when you already know that the device supports the acceleration
    /// structure extension.
    ///
    /// # Panics
    ///
    /// Panics if [Self.physical_device.accel_struct_properties] is `None`.
    pub(crate) fn expect_accel_struct_ext(this: &Self) -> &khr::acceleration_structure::Device {
        this.accel_struct_ext
            .as_ref()
            .expect("VK_KHR_acceleration_structure")
    }

    /// Helper for times when you already know that the device supports the ray tracing pipeline
    /// extension.
    ///
    /// # Panics
    ///
    /// Panics if [Self.physical_device.ray_trace_properties] is `None`.
    pub(crate) fn expect_ray_trace_ext(this: &Self) -> &khr::ray_tracing_pipeline::Device {
        this.ray_trace_ext
            .as_ref()
            .expect("VK_KHR_ray_tracing_pipeline")
    }

    /// Helper for times when you already know that the instance supports the surface extension.
    ///
    /// # Panics
    ///
    /// Panics if the device was not created for display window access.
    pub(crate) fn expect_surface_ext(this: &Self) -> &khr::surface::Instance {
        this.surface_ext.as_ref().expect("VK_KHR_surface")
    }

    /// Helper for times when you already know that the device supports the swapchain extension.
    ///
    /// # Panics
    ///
    /// Panics if the device was not created for display window access.
    pub(crate) fn expect_swapchain_ext(this: &Self) -> &khr::swapchain::Device {
        this.swapchain_ext.as_ref().expect("VK_KHR_swapchain")
    }

    /// Loads and existing `ash` Vulkan device that may have been created by other means.
    #[profiling::function]
    pub fn load(
        physical_device: PhysicalDevice,
        device: ash::Device,
        display_window: bool,
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
            warn!("{err}");

            DriverError::Unsupported
        })?;

        let mut queues = Vec::with_capacity(physical_device.queue_families.len());

        for (queue_family_index, properties) in physical_device.queue_families.iter().enumerate() {
            let mut queue_family = Vec::with_capacity(properties.queue_count as _);

            for queue_index in 0..properties.queue_count {
                queue_family
                    .push(unsafe { device.get_device_queue(queue_family_index as _, queue_index) });
            }

            queues.push(queue_family);
        }

        let surface_ext = display_window.then(|| {
            khr::surface::Instance::new(&physical_device.instance.entry, &physical_device.instance)
        });
        let swapchain_ext =
            display_window.then(|| khr::swapchain::Device::new(&physical_device.instance, &device));
        let accel_struct_ext = physical_device
            .accel_struct_properties
            .is_some()
            .then(|| khr::acceleration_structure::Device::new(&physical_device.instance, &device));
        let ray_trace_ext = physical_device
            .ray_trace_features
            .ray_tracing_pipeline
            .then(|| khr::ray_tracing_pipeline::Device::new(&physical_device.instance, &device));

        let pipeline_cache =
            unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }
                .map_err(|err| {
                    warn!("{err}");

                    DriverError::Unsupported
                })?;

        Ok(Self {
            accel_struct_ext,
            allocator: ManuallyDrop::new(Mutex::new(allocator)),
            device,
            pipeline_cache,
            physical_device,
            queues,
            ray_trace_ext,
            surface_ext,
            swapchain_ext,
        })
    }

    /// Lists the physical device's image format capabilities.
    ///
    /// A result of `None` indicates the format is not supported.
    #[profiling::function]
    pub fn image_format_properties(
        this: &Self,
        format: vk::Format,
        ty: vk::ImageType,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        flags: vk::ImageCreateFlags,
    ) -> Result<Option<vk::ImageFormatProperties>, DriverError> {
        unsafe {
            match this
                .physical_device
                .instance
                .get_physical_device_image_format_properties(
                    this.physical_device.handle,
                    format,
                    ty,
                    tiling,
                    usage,
                    flags,
                ) {
                Ok(properties) => Ok(Some(properties)),
                Err(err) if err == vk::Result::ERROR_FORMAT_NOT_SUPPORTED => {
                    // We don't log this condition because it is normal for unsupported
                    // formats to be checked - we use the result to inform callers they
                    // cannot use those formats.

                    Ok(None)
                }
                _ => Err(DriverError::OutOfMemory),
            }
        }
    }

    pub(crate) fn pipeline_cache(this: &Self) -> vk::PipelineCache {
        this.pipeline_cache
    }

    #[profiling::function]
    pub(crate) fn wait_for_fence(this: &Self, fence: &vk::Fence) -> Result<(), DriverError> {
        use std::slice::from_ref;

        Device::wait_for_fences(this, from_ref(fence))
    }

    #[profiling::function]
    pub(crate) fn wait_for_fences(this: &Self, fences: &[vk::Fence]) -> Result<(), DriverError> {
        unsafe {
            match this.device.wait_for_fences(fences, true, 100) {
                Ok(_) => return Ok(()),
                Err(err) if err == vk::Result::ERROR_DEVICE_LOST => {
                    error!("Device lost");

                    return Err(DriverError::InvalidData);
                }
                Err(err) if err == vk::Result::TIMEOUT => {
                    trace!("waiting...");
                }
                _ => return Err(DriverError::OutOfMemory),
            }

            let started = Instant::now();

            match this.device.wait_for_fences(fences, true, u64::MAX) {
                Ok(_) => (),
                Err(err) if err == vk::Result::ERROR_DEVICE_LOST => {
                    error!("Device lost");

                    return Err(DriverError::InvalidData);
                }
                _ => return Err(DriverError::OutOfMemory),
            }

            let elapsed = Instant::now() - started;
            let elapsed_millis = elapsed.as_millis();

            if elapsed_millis > 0 {
                warn!("waited for {} ms", elapsed_millis);
            }
        }

        Ok(())
    }
}

impl Debug for Device {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Device")
    }
}

#[doc(hidden)]
impl Deref for Device {
    type Target = DeviceRef;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self as *const Self::Target) }
    }
}

impl Deref for DeviceRef {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Drop for Device {
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

/// Information used to create a [`Device`] instance.
#[derive(Builder, Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[builder(
    build_fn(private, name = "fallible_build", error = "DeviceInfoBuilderError"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
#[non_exhaustive]
pub struct DeviceInfo {
    /// Enables Vulkan validation layers.
    ///
    /// This requires a Vulkan SDK installation and will cause validation errors to introduce
    /// panics as they happen.
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
    /// A helper function which prioritizes selection of lower-power integrated GPU devices.
    #[profiling::function]
    pub fn integrated_gpu_index(physical_devices: &[PhysicalDevice]) -> usize {
        Self::pick_best_gpu(
            physical_devices,
            &[
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                vk::PhysicalDeviceType::DISCRETE_GPU,
                vk::PhysicalDeviceType::VIRTUAL_GPU,
                vk::PhysicalDeviceType::CPU,
                vk::PhysicalDeviceType::OTHER,
            ],
        )
    }

    /// A helper function which prioritizes selection of higher-performance discrete GPU devices.
    #[profiling::function]
    pub fn discrete_gpu_index(physical_devices: &[PhysicalDevice]) -> usize {
        Self::pick_best_gpu(
            physical_devices,
            &[
                vk::PhysicalDeviceType::DISCRETE_GPU,
                vk::PhysicalDeviceType::VIRTUAL_GPU,
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                vk::PhysicalDeviceType::CPU,
                vk::PhysicalDeviceType::OTHER,
            ],
        )
    }

    fn pick_best_gpu(
        physical_devices: &[PhysicalDevice],
        best: &[vk::PhysicalDeviceType],
    ) -> usize {
        for best in best.iter().copied() {
            for (idx, physical_device) in physical_devices.iter().enumerate() {
                if physical_device.properties_v1_0.device_type == best {
                    return idx;
                }
            }
        }

        0
    }

    /// Converts a `DeviceInfo` into a `DeviceInfoBuilder`.
    #[inline(always)]
    pub fn to_builder(self) -> DeviceInfoBuilder {
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
        let res = self.fallible_build();

        #[cfg(test)]
        let res = res.unwrap();

        #[cfg(not(test))]
        let res = unsafe { res.unwrap_unchecked() };

        res
    }
}

#[derive(Debug)]
struct DeviceInfoBuilderError;

impl From<UninitializedFieldError> for DeviceInfoBuilderError {
    fn from(_: UninitializedFieldError) -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Info = DeviceInfo;
    type Builder = DeviceInfoBuilder;

    #[test]
    pub fn device_info() {
        Info::default().to_builder().build();
    }

    #[test]
    pub fn device_info_builder() {
        Builder::default().build();
    }
}
