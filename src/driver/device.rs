//! Logical device resource types

use {
    super::{DriverError, Instance, physical_device::PhysicalDevice},
    ash::{ext, khr, vk},
    ash_window::enumerate_required_extensions,
    derive_builder::{Builder, UninitializedFieldError},
    gpu_allocator::{
        AllocatorDebugSettings,
        vulkan::{Allocator, AllocatorCreateDesc},
    },
    log::{error, info, trace, warn},
    raw_window_handle::HasDisplayHandle,
    std::{
        cmp::Ordering,
        ffi::CStr,
        fmt::{Debug, Formatter},
        iter::{empty, repeat_n},
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

/// Function type for selection of physical devices.
pub type SelectPhysicalDeviceFn = dyn FnOnce(&[PhysicalDevice]) -> usize;

/// Opaque handle to a device object.
pub struct Device {
    accel_struct_ext: Option<khr::acceleration_structure::Device>,

    pub(super) allocator: ManuallyDrop<Mutex<Allocator>>,

    device: ash::Device,

    /// Vulkan instance pointer, which includes useful functions.
    instance: Instance,

    pipeline_cache: vk::PipelineCache,

    /// The physical device, which contains useful data about features, properties, and limits.
    pub physical_device: PhysicalDevice,

    /// The physical execution queues which all work will be submitted to.
    pub(crate) queues: Vec<Vec<vk::Queue>>,

    pub(crate) ray_trace_ext: Option<khr::ray_tracing_pipeline::Device>,

    pub(super) surface_ext: Option<khr::surface::Instance>,
    pub(super) swapchain_ext: Option<khr::swapchain::Device>,
}

impl Device {
    /// Prepares device creation information and calls the provided callback to allow an application
    /// to control the device creation process.
    ///
    /// # Safety
    ///
    /// This is only required for interoperting with other libraries and comes with all the caveats
    /// of using `ash` builder types, which are inherently dangerous. Use with extreme caution.
    #[profiling::function]
    pub unsafe fn create_ash_device<F>(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        display_window: bool,
        create_fn: F,
    ) -> ash::prelude::VkResult<ash::Device>
    where
        F: FnOnce(vk::DeviceCreateInfo) -> ash::prelude::VkResult<ash::Device>,
    {
        let mut enabled_ext_names = Vec::with_capacity(7);

        if display_window {
            enabled_ext_names.push(khr::swapchain::NAME.as_ptr());
        }

        if physical_device.accel_struct_properties.is_some() {
            enabled_ext_names.push(khr::acceleration_structure::NAME.as_ptr());
            enabled_ext_names.push(khr::deferred_host_operations::NAME.as_ptr());
        }

        if physical_device.ray_query_features.ray_query {
            enabled_ext_names.push(khr::ray_query::NAME.as_ptr());
        }

        if physical_device.ray_trace_features.ray_tracing_pipeline {
            enabled_ext_names.push(khr::ray_tracing_pipeline::NAME.as_ptr());
        }

        if physical_device.index_type_uint8_features.index_type_uint8 {
            enabled_ext_names.push(ext::index_type_uint8::NAME.as_ptr());
        }

        // Molten-vk doesn't support the full Vulkan feature set, hence the portability subset extension must be enabled.
        #[cfg(all(target_os = "macos", feature = "loaded"))]
        {
            enabled_ext_names.push(khr::portability_subset::NAME.as_ptr());
        }

        let priorities = repeat_n(
            1.0,
            physical_device
                .queue_families
                .iter()
                .map(|family| family.queue_count)
                .max()
                .unwrap_or_default() as _,
        )
        .collect::<Box<_>>();

        let queue_infos = physical_device
            .queue_families
            .iter()
            .enumerate()
            .map(|(idx, family)| {
                let mut queue_info = vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(idx as _)
                    .queue_priorities(&priorities[0..family.queue_count as usize]);
                queue_info.queue_count = family.queue_count;

                queue_info
            })
            .collect::<Box<_>>();

        let ash::InstanceFnV1_1 {
            get_physical_device_features2,
            ..
        } = instance.fp_v1_1();
        let mut features_v1_1 = vk::PhysicalDeviceVulkan11Features::default();
        let mut features_v1_2 = vk::PhysicalDeviceVulkan12Features::default();
        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut index_type_uint8_features = vk::PhysicalDeviceIndexTypeUint8FeaturesEXT::default();
        let mut ray_query_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
        let mut ray_trace_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut features_v1_1)
            .push_next(&mut features_v1_2);

        if physical_device.accel_struct_properties.is_some() {
            features = features.push_next(&mut acceleration_structure_features);
        }

        if physical_device.ray_query_features.ray_query {
            features = features.push_next(&mut ray_query_features);
        }

        if physical_device.ray_trace_features.ray_tracing_pipeline {
            features = features.push_next(&mut ray_trace_features);
        }

        if physical_device.index_type_uint8_features.index_type_uint8 {
            features = features.push_next(&mut index_type_uint8_features);
        }

        unsafe { get_physical_device_features2(**physical_device, &mut features) };

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&enabled_ext_names)
            .push_next(&mut features);

        create_fn(device_create_info)
    }

    #[profiling::function]
    fn create(
        instance: Instance,
        select_physical_device: Box<SelectPhysicalDeviceFn>,
        display_window: bool,
    ) -> Result<Self, DriverError> {
        let mut physical_devices = Instance::physical_devices(&instance)?;

        if physical_devices.is_empty() {
            error!("no supported devices found");

            return Err(DriverError::Unsupported);
        }

        let mut phyical_device_idx = select_physical_device(&physical_devices);

        if phyical_device_idx >= physical_devices.len() {
            warn!("invalid device selected");

            phyical_device_idx = 0;
        }

        let physical_device = physical_devices.remove(phyical_device_idx);

        let device = unsafe {
            Self::create_ash_device(
                &instance,
                &physical_device,
                display_window,
                |device_create_info| {
                    instance.create_device(*physical_device, &device_create_info, None)
                },
            )
        }
        .map_err(|err| {
            error!("unable to create device: {err}");

            DriverError::Unsupported
        })?;

        info!("created {}", physical_device.properties_v1_0.device_name);

        Self::load(instance, physical_device, device, display_window)
    }

    /// Constructs a new device using the given configuration.
    #[profiling::function]
    pub fn create_headless(info: impl Into<DeviceInfo>) -> Result<Self, DriverError> {
        let DeviceInfo {
            debug,
            select_physical_device,
        } = info.into();
        let instance = Instance::create(debug, empty())?;

        Self::create(instance, select_physical_device, false)
    }

    /// Constructs a new device using the given configuration.
    #[profiling::function]
    pub fn create_display(
        info: impl Into<DeviceInfo>,
        display_handle: &impl HasDisplayHandle,
    ) -> Result<Self, DriverError> {
        let DeviceInfo {
            debug,
            select_physical_device,
        } = info.into();
        let display_handle = display_handle.display_handle().map_err(|err| {
            warn!("{err}");

            DriverError::Unsupported
        })?;
        let required_extensions = enumerate_required_extensions(display_handle.as_raw())
            .map_err(|err| {
                warn!("{err}");

                DriverError::Unsupported
            })?
            .iter()
            .map(|ext| unsafe { CStr::from_ptr(*ext as *const _) });
        let instance = Instance::create(debug, required_extensions)?;

        Self::create(instance, select_physical_device, true)
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
        instance: Instance,
        physical_device: PhysicalDevice,
        device: ash::Device,
        display_window: bool,
    ) -> Result<Self, DriverError> {
        let debug = Instance::is_debug(&instance);
        let mut debug_settings = AllocatorDebugSettings::default();
        debug_settings.log_leaks_on_shutdown = debug;
        debug_settings.log_memory_information = debug;
        debug_settings.log_allocations = debug;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: (*instance).clone(),
            device: device.clone(),
            physical_device: *physical_device,
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

        let surface_ext = display_window
            .then(|| khr::surface::Instance::new(Instance::entry(&instance), &instance));
        let swapchain_ext = display_window.then(|| khr::swapchain::Device::new(&instance, &device));
        let accel_struct_ext = physical_device
            .accel_struct_properties
            .is_some()
            .then(|| khr::acceleration_structure::Device::new(&instance, &device));
        let ray_trace_ext = physical_device
            .ray_trace_features
            .ray_tracing_pipeline
            .then(|| khr::ray_tracing_pipeline::Device::new(&instance, &device));

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
            instance,
            pipeline_cache,
            physical_device,
            queues,
            ray_trace_ext,
            surface_ext,
            swapchain_ext,
        })
    }

    /// Lists the physical device's format capabilities.
    #[profiling::function]
    pub fn format_properties(this: &Self, format: vk::Format) -> vk::FormatProperties {
        unsafe {
            this.instance
                .get_physical_device_format_properties(*this.physical_device, format)
        }
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
            match this.instance.get_physical_device_image_format_properties(
                *this.physical_device,
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

    /// Provides a reference to the Vulkan instance used by this device.
    pub fn instance(this: &Self) -> &Instance {
        &this.instance
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

impl Deref for Device {
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
#[derive(Builder)]
#[builder(
    build_fn(private, name = "fallible_build", error = "DeviceInfoBuilderError"),
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
    /// **macOS:** Has no effect.
    #[builder(default)]
    pub debug: bool,

    /// Callback function used to select a [`PhysicalDevice`] from the available devices. The
    /// callback must return the index of the selected device.
    #[builder(default = "Box::new(DeviceInfo::discrete_gpu)")]
    pub select_physical_device: Box<SelectPhysicalDeviceFn>,
}

impl DeviceInfo {
    /// Specifies default device information.
    #[allow(clippy::new_ret_no_self)]
    #[deprecated = "Use DeviceInfo::default()"]
    #[doc(hidden)]
    pub fn new() -> DeviceInfoBuilder {
        Default::default()
    }

    /// A builtin [`DeviceInfo::select_physical_device`] function which prioritizes selection of
    /// lower-power integrated GPU devices.
    #[profiling::function]
    pub fn integrated_gpu(physical_devices: &[PhysicalDevice]) -> usize {
        assert!(!physical_devices.is_empty());

        let mut physical_devices = physical_devices.iter().enumerate().collect::<Box<_>>();

        if physical_devices.len() == 1 {
            return 0;
        }

        fn device_type(ty: vk::PhysicalDeviceType) -> usize {
            match ty {
                vk::PhysicalDeviceType::INTEGRATED_GPU => 0,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                vk::PhysicalDeviceType::CPU => 2,
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                _ => 4,
            }
        }

        physical_devices.sort_unstable_by(|(_, lhs), (_, rhs)| {
            let lhs_device_ty = device_type(lhs.properties_v1_0.device_type);
            let rhs_device_ty = device_type(rhs.properties_v1_0.device_type);
            let device_ty = lhs_device_ty.cmp(&rhs_device_ty);

            if device_ty != Ordering::Equal {
                return device_ty;
            }

            // TODO: Select the device with the most memory

            Ordering::Equal
        });

        let (idx, _) = physical_devices[0];

        idx
    }

    /// A builtin [`DeviceInfo::select_physical_device`] function which prioritizes selection of
    /// higher-performance discrete GPU devices.
    #[profiling::function]
    pub fn discrete_gpu(physical_devices: &[PhysicalDevice]) -> usize {
        assert!(!physical_devices.is_empty());

        let mut physical_devices = physical_devices.iter().enumerate().collect::<Box<_>>();

        if physical_devices.len() == 1 {
            return 0;
        }

        fn device_type(ty: vk::PhysicalDeviceType) -> usize {
            match ty {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 3,
                _ => 4,
            }
        }

        physical_devices.sort_unstable_by(|(_, lhs), (_, rhs)| {
            let lhs_device_ty = device_type(lhs.properties_v1_0.device_type);
            let rhs_device_ty = device_type(rhs.properties_v1_0.device_type);
            let device_ty = lhs_device_ty.cmp(&rhs_device_ty);

            if device_ty != Ordering::Equal {
                return device_ty;
            }

            // TODO: Select the device with the most memory

            Ordering::Equal
        });

        let (idx, _) = physical_devices[0];

        idx
    }

    /// Converts a `DeviceInfo` into a `DeviceInfoBuilder`.
    #[inline(always)]
    pub fn to_builder(self) -> DeviceInfoBuilder {
        DeviceInfoBuilder {
            debug: Some(self.debug),
            select_physical_device: Some(self.select_physical_device),
        }
    }
}

impl Debug for DeviceInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceInfo")
            .field("debug", &self.debug)
            .field("select_physical_device", &"fn")
            .finish()
    }
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            debug: false,
            select_physical_device: Box::new(DeviceInfo::discrete_gpu),
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
