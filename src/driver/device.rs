//! Logical device types

use {
    super::{
        DriverError,
        instance::{ApiVersion, Instance, InstanceInfoBuilder},
        physical_device::PhysicalDevice,
    },
    ash::{ext, khr, vk},
    derive_builder::Builder,
    gpu_allocator::{
        AllocatorDebugSettings,
        vulkan::{Allocator, AllocatorCreateDesc},
    },
    log::{error, info, trace, warn},
    raw_window_handle::HasDisplayHandle,
    std::{
        collections::HashMap,
        ffi::CString,
        fmt::{Debug, Formatter},
        mem::{ManuallyDrop, forget},
        ops::Deref,
        slice,
        sync::Arc,
        sync::atomic::{AtomicU64, Ordering},
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

    /// Begins a Vulkan debug label region on `command_buffer` when debug labeling is enabled.
    ///
    /// Returns without doing any work if debug mode is `false`.
    pub fn begin_debug_utils_label(
        this: &Self,
        command_buffer: vk::CommandBuffer,
        label_name: impl AsRef<str>,
    ) -> Result<(), DriverError> {
        if !this.physical_device.instance.info.debug {
            return Ok(());
        }

        let Ok(label_name) = CString::new(label_name.as_ref()) else {
            warn!("invalid label name");

            return Err(DriverError::InvalidData);
        };

        let ext = Self::try_vk_ext_debug_utils(this)?;

        unsafe {
            ext.cmd_begin_debug_utils_label(
                command_buffer,
                &vk::DebugUtilsLabelEXT::default().label_name(label_name.as_c_str()),
            );
        }

        Ok(())
    }

    /// Clears Vulkan private-data metadata associated with `object_type` and `object_handle`.
    pub(crate) fn clear_private_data_object_name<T>(
        this: &Self,
        object_type: vk::ObjectType,
        object_handle: T,
    ) -> Result<(), DriverError>
    where
        T: vk::Handle + Copy,
    {
        if this.inner.private_data_slot.is_none() {
            return Ok(());
        }

        if object_handle.is_null() {
            warn!("invalid object handle");

            return Err(DriverError::InvalidData);
        }

        let object_key = (object_type, object_handle.as_raw());
        let previous_metadata_id = Self::with_object_metadata_ids(this, |object_to_metadata_id| {
            object_to_metadata_id.remove(&object_key)
        });

        if previous_metadata_id.is_none() {
            return Ok(());
        }

        let ext = Self::try_vk_ext_private_data(this)?;
        let private_data_slot = this
            .inner
            .private_data_slot
            .expect("missing private data slot");

        if let Err(err) = unsafe { ext.set_private_data(object_handle, private_data_slot, 0) } {
            Self::with_object_metadata_ids(this, |object_metadata_ids| {
                if let Some(metadata_id) = previous_metadata_id {
                    object_metadata_ids.insert(object_key, metadata_id);
                }
            });

            warn!("unable to clear private data object name: {err}");

            return Err(match err {
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                    DriverError::OutOfMemory
                }
                _ => DriverError::Unsupported,
            });
        }

        Self::with_private_data_metadata(this, |metadata| {
            metadata
                .names
                .remove(&previous_metadata_id.expect("metadata id removed"));
        });

        Ok(())
    }

    /// Records a pipeline barrier using the `VK_KHR_synchronization2` extension or the Vulkan 1.3
    /// core `vkCmdPipelineBarrier2` path.
    ///
    /// See [`vkCmdPipelineBarrier2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdPipelineBarrier2.html)
    /// and [`VK_KHR_synchronization2`](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_synchronization2.html).
    pub fn cmd_pipeline_barrier2(
        this: &Self,
        command_buffer: vk::CommandBuffer,
        dependency_info: &vk::DependencyInfo,
    ) {
        #[cfg(feature = "checked")]
        assert!(
            this.physical_device.vk_khr_synchronization2,
            "missing synchronization2 feature"
        );

        unsafe {
            if this.physical_device.instance.info.api_version >= ApiVersion::Vulkan13 {
                this.cmd_pipeline_barrier2(command_buffer, dependency_info);
            } else {
                let khr_synchronization2 = Device::expect_vk_khr_synchronization2(this);

                khr_synchronization2.cmd_pipeline_barrier2(command_buffer, dependency_info);
            }
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

    /// Creates a Vulkan fence on this device.
    ///
    /// Pass `true` for `signaled` when the fence should begin in the signaled state.
    ///
    /// See [`vkCreateFence`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateFence.html).
    pub fn create_fence(this: &Self, signaled: bool) -> Result<vk::Fence, DriverError> {
        let mut flags = vk::FenceCreateFlags::empty();

        if signaled {
            flags |= vk::FenceCreateFlags::SIGNALED;
        }

        let create_info = vk::FenceCreateInfo::default().flags(flags);
        let allocation_callbacks = None;

        unsafe {
            this.create_fence(&create_info, allocation_callbacks)
                .map_err(|err| {
                    warn!("unable to create fence: {err}");

                    DriverError::OutOfMemory
                })
        }
    }

    /// Creates a Vulkan binary semaphore on this device.
    ///
    /// See [`vkCreateSemaphore`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateSemaphore.html).
    pub fn create_semaphore(this: &Self) -> Result<vk::Semaphore, DriverError> {
        let create_info = vk::SemaphoreCreateInfo::default();
        let allocation_callbacks = None;

        unsafe {
            this.create_semaphore(&create_info, allocation_callbacks)
                .map_err(|err| {
                    warn!("unable to create semaphore: {err}");

                    DriverError::OutOfMemory
                })
        }
    }

    /// Ends recording a command buffer on this device.
    ///
    /// This is a thin wrapper around [`ash::Device::end_command_buffer`] that maps Vulkan errors
    /// to [`DriverError`] variants.
    ///
    /// See [`vkEndCommandBuffer`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkEndCommandBuffer.html).
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

    /// Ends a Vulkan debug label region on `command_buffer` when debug labeling is enabled.
    ///
    /// Returns without doing any work if debug mode is `false`.
    pub fn end_debug_utils_label(
        this: &Self,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), DriverError> {
        if !this.physical_device.instance.info.debug {
            return Ok(());
        }

        let ext = Self::try_vk_ext_debug_utils(this)?;

        unsafe {
            ext.cmd_end_debug_utils_label(command_buffer);
        }

        Ok(())
    }

    /// Helper for times when you already know that the device supports the acceleration
    /// structure extension.
    ///
    /// # Panics
    ///
    /// Panics if acceleration structure support was not enabled for this device.
    pub(crate) fn expect_vk_khr_acceleration_structure(
        this: &Self,
    ) -> &khr::acceleration_structure::Device {
        this.inner
            .vk_khr_acceleration_structure
            .as_ref()
            .expect("missing VK_KHR_acceleration_structure")
    }

    /// Helper for times when you already know that present wait support is enabled.
    ///
    /// # Panics
    ///
    /// Panics if `VK_KHR_present_wait` support was not enabled for this device.
    pub(crate) fn expect_vk_khr_present_wait(this: &Self) -> &khr::present_wait::Device {
        this.inner
            .vk_khr_present_wait
            .as_ref()
            .expect("missing VK_KHR_present_wait")
    }

    /// Helper for times when you already know that the device supports the ray tracing pipeline
    /// extension.
    ///
    /// # Panics
    ///
    /// Panics if ray tracing pipeline support was not enabled for this device.
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

    /// Helper for times when you already know that the device supports the synchronization2
    /// extension.
    ///
    /// # Panics
    ///
    /// Panics if `VK_KHR_synchronization2` is not available.
    pub(crate) fn expect_vk_khr_synchronization2(this: &Self) -> &khr::synchronization2::Device {
        this.inner
            .vk_khr_synchronization2
            .as_ref()
            .expect("missing VK_KHR_synchronization2")
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

    /// Removes local Vulkan private-data metadata without touching the Vulkan object.
    pub(crate) fn forget_private_data_object_name<T>(
        this: &Self,
        object_type: vk::ObjectType,
        object_handle: T,
    ) where
        T: vk::Handle + Copy,
    {
        if this.inner.private_data_slot.is_none() || object_handle.is_null() {
            return;
        }

        let object_key = (object_type, object_handle.as_raw());
        let Some(metadata_id) = Self::with_object_metadata_ids(this, |object_metadata_ids| {
            object_metadata_ids.remove(&object_key)
        }) else {
            return;
        };

        Self::with_private_data_metadata(this, |metadata| {
            metadata.names.remove(&metadata_id);
        });
    }

    /// Returns `true` if both handles refer to the same logical device allocation.
    pub(crate) fn is_same(lhs: &Self, rhs: &Self) -> bool {
        Arc::ptr_eq(&lhs.inner, &rhs.inner)
    }

    /// Returns the device-owned pipeline cache handle.
    pub(crate) fn pipeline_cache(this: &Self) -> vk::PipelineCache {
        this.inner.pipeline_cache
    }

    /// Retrieves a Vulkan private-data name associated with `handle`.
    ///
    /// Returns `None` when metadata is not available or when the `VK_EXT_private_data` extension is
    /// not available.
    pub(crate) fn private_data_object_name<T>(
        this: &Self,
        object_type: vk::ObjectType,
        object_handle: T,
    ) -> Option<String>
    where
        T: vk::Handle + Copy,
    {
        this.inner.private_data_slot?;

        if object_handle.is_null() {
            return None;
        }

        let object_key = (object_type, object_handle.as_raw());
        Self::with_private_data_metadata(this, |metadata| {
            let metadata_id = metadata.object_metadata_ids.get(&object_key)?;

            metadata.names.get(metadata_id).cloned()
        })
    }

    /// Submits command buffers to a queue, optionally signaling a fence.
    ///
    /// See [`vkQueueSubmit`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueSubmit.html).
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

    /// Submits command buffers to a queue using the `VK_KHR_synchronization2` extension or the
    /// Vulkan 1.3 core `vkQueueSubmit2` path.
    ///
    /// See [`vkQueueSubmit2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueSubmit2.html)
    /// and [`VK_KHR_synchronization2`](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_synchronization2.html).
    pub fn queue_submit2(
        this: &Self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo2],
        fence: vk::Fence,
    ) -> Result<(), DriverError> {
        #[cfg(feature = "checked")]
        assert!(this.physical_device.vk_khr_synchronization2);

        unsafe {
            if this.physical_device.instance.info.api_version >= ApiVersion::Vulkan13 {
                // Support derived from Vulkan v1.3 implementation
                this.queue_submit2(queue, submits, fence)
            } else {
                let khr_synchronization2 = Device::expect_vk_khr_synchronization2(this);

                // Support derived from Vulkan v1.2 implementation + extension
                khr_synchronization2.queue_submit2(queue, submits, fence)
            }
            .map_err(|err| {
                warn!("unable to queue submit2 submissions: {err}");

                match err {
                    vk::Result::ERROR_DEVICE_LOST => DriverError::InvalidData,
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                    | vk::Result::ERROR_OUT_OF_HOST_MEMORY => DriverError::OutOfMemory,
                    _ => DriverError::Unsupported,
                }
            })
        }
    }

    /// Waits for queue idle.
    pub fn queue_wait_idle(this: &Self, queue: vk::Queue) -> Result<(), DriverError> {
        unsafe {
            this.queue_wait_idle(queue).map_err(|err| {
                warn!("unable to wait for queue idle: {err}");

                match err {
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                    | vk::Result::ERROR_OUT_OF_HOST_MEMORY => DriverError::OutOfMemory,
                    vk::Result::ERROR_DEVICE_LOST | vk::Result::ERROR_VALIDATION_FAILED_EXT => {
                        DriverError::InvalidData
                    }
                    _ => DriverError::Unsupported,
                }
            })
        }
    }

    /// Resets one or more fences to the unsignaled state.
    ///
    /// See [`vkResetFences`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkResetFences.html).
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

    /// Assigns a Vulkan debug-utils name to `handle` when debug labeling is enabled.
    ///
    /// Returns without doing any work if debug mode is `false`.
    pub fn set_debug_utils_object_name<T>(
        this: &Self,
        object_handle: T,
        object_name: impl AsRef<str>,
    ) -> Result<(), DriverError>
    where
        T: vk::Handle + Copy,
    {
        if !this.physical_device.instance.info.debug {
            return Ok(());
        }

        if object_handle.is_null() {
            warn!("invalid object handle");

            return Err(DriverError::InvalidData);
        }

        let Ok(object_name) = CString::new(object_name.as_ref()) else {
            warn!("invalid object name");

            return Err(DriverError::InvalidData);
        };

        let ext = Self::try_vk_ext_debug_utils(this)?;

        unsafe {
            match ext.set_debug_utils_object_name(
                &vk::DebugUtilsObjectNameInfoEXT::default()
                    .object_handle(object_handle)
                    .object_name(object_name.as_c_str()),
            ) {
                Err(
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                ) => Err(DriverError::OutOfMemory),
                Err(vk::Result::ERROR_VALIDATION_FAILED_EXT) => Err(DriverError::InvalidData),
                Err(err) => {
                    warn!("unable to set debug utils object name: {err}");

                    Err(DriverError::Unsupported)
                }
                Ok(_) => Ok(()),
            }
        }
    }

    /// Stores a Vulkan name in private-data metadata.
    ///
    /// Returns without doing any work when `VK_EXT_private_data` is not available.
    pub(crate) fn set_private_data_object_name<T>(
        this: &Self,
        object_type: vk::ObjectType,
        object_handle: T,
        object_name: impl AsRef<str>,
    ) -> Result<(), DriverError>
    where
        T: vk::Handle + Copy,
    {
        if this.inner.private_data_slot.is_none() {
            return Ok(());
        }

        if object_handle.is_null() {
            warn!("invalid object handle");

            return Err(DriverError::InvalidData);
        }

        let object_key = (object_type, object_handle.as_raw());
        let metadata_id = this
            .inner
            .private_data_name_id
            .fetch_add(1, Ordering::Relaxed)
            + 1;

        let (previous_metadata_id, previous_name) =
            Self::with_private_data_metadata(this, |metadata| {
                let previous_metadata_id =
                    metadata.object_metadata_ids.insert(object_key, metadata_id);
                let previous_name = previous_metadata_id.and_then(|id| metadata.names.remove(&id));

                metadata
                    .names
                    .insert(metadata_id, object_name.as_ref().to_owned());

                (previous_metadata_id, previous_name)
            });

        let ext = Self::try_vk_ext_private_data(this)?;
        let private_data_slot = this
            .inner
            .private_data_slot
            .expect("missing private data slot");

        if let Err(err) =
            unsafe { ext.set_private_data(object_handle, private_data_slot, metadata_id) }
        {
            Self::with_private_data_metadata(this, |metadata| {
                let _ = metadata.names.remove(&metadata_id);
                match previous_metadata_id {
                    Some(id) => {
                        metadata.object_metadata_ids.insert(object_key, id);
                        if let Some(name) = previous_name {
                            metadata.names.insert(id, name);
                        }
                    }
                    None => {
                        metadata.object_metadata_ids.remove(&object_key);
                    }
                }
            });

            warn!("unable to set private data object name: {err}");

            return Err(match err {
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                    DriverError::OutOfMemory
                }
                _ => DriverError::Unsupported,
            });
        }

        Ok(())
    }

    /// Loads an existing `ash` Vulkan device that may have been created by other means.
    ///
    /// # Safety
    ///
    /// `device` must have been created from `physical_device` and must have all queues,
    /// extensions, and features expected by `vk-graph`. The device must not be destroyed outside
    /// this wrapper while any cloned [`Device`] or resources created from it remain alive.
    #[profiling::function]
    pub unsafe fn try_from_ash(
        device: ash::Device,
        physical_device: PhysicalDevice,
    ) -> Result<Self, DriverError> {
        let debug = physical_device.instance.info.debug;

        if debug && !Instance::supports_debug_utils(&physical_device.instance) {
            error!("unsupported VK_EXT_debug_utils");

            return Err(DriverError::Unsupported);
        }

        if debug && !physical_device.vk_ext_private_data {
            error!("unsupported VK_EXT_private_data");

            return Err(DriverError::Unsupported);
        }

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

        let vk_ext_debug_utils = Some(ext::debug_utils::Device::new(
            &physical_device.instance,
            &device,
        ));
        let vk_ext_private_data = physical_device
            .vk_ext_private_data
            .then(|| ext::private_data::Device::new(&physical_device.instance, &device));
        let vk_ext_private_data_slot = vk_ext_private_data
            .as_ref()
            .map(|vk_ext_private_data| unsafe {
                vk_ext_private_data
                    .create_private_data_slot(
                        &vk::PrivateDataSlotCreateInfoEXT::default()
                            .flags(vk::PrivateDataSlotCreateFlagsEXT::empty()),
                        None,
                    )
                    .map_err(|err| {
                        warn!("unable to create private data slot: {err}");

                        DriverError::Unsupported
                    })
            })
            .transpose()?;
        let vk_khr_present_wait = physical_device
            .vk_khr_present_wait
            .is_some()
            .then(|| khr::present_wait::Device::new(&physical_device.instance, &device));
        let vk_khr_surface = physical_device.vk_khr_swapchain.then(|| {
            let entry = Instance::entry(&physical_device.instance);
            khr::surface::Instance::new(entry, &physical_device.instance)
        });
        let vk_khr_swapchain = physical_device
            .vk_khr_swapchain
            .then(|| khr::swapchain::Device::new(&physical_device.instance, &device));
        let vk_khr_acceleration_structure = physical_device
            .vk_khr_acceleration_structure
            .is_some()
            .then(|| khr::acceleration_structure::Device::new(&physical_device.instance, &device));
        let vk_khr_ray_tracing_pipeline = physical_device
            .vk_khr_ray_tracing_pipeline
            .as_ref()
            .is_some_and(|ext| ext.features.ray_tracing_pipeline)
            .then(|| khr::ray_tracing_pipeline::Device::new(&physical_device.instance, &device));
        let vk_khr_synchronization2 = physical_device
            .vk_khr_synchronization2
            .then(|| khr::synchronization2::Device::new(&physical_device.instance, &device));

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
                    vk_ext_debug_utils,
                    vk_ext_private_data,
                    vk_khr_acceleration_structure,
                    vk_khr_present_wait,
                    vk_khr_ray_tracing_pipeline,
                    vk_khr_surface,
                    vk_khr_swapchain,
                    vk_khr_synchronization2,
                    private_data_slot: vk_ext_private_data_slot,
                    private_data_name_id: AtomicU64::new(0),
                    private_data_metadata: Mutex::new(Default::default()),
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

        unsafe { Self::try_from_ash(device, physical_device) }
    }

    pub(crate) fn try_clear_private_data_object_name<T>(
        this: &Self,
        object_type: vk::ObjectType,
        object_handle: T,
    ) where
        T: vk::Handle + Copy,
    {
        let _ = Self::clear_private_data_object_name(this, object_type, object_handle);
    }

    /// Assigns a Vulkan debug-utils name to `handle` when debug labeling is enabled.
    ///
    /// Returns without doing any work if debug mode is `false`.
    pub fn try_set_debug_utils_object_name<T>(
        this: &Self,
        object_handle: T,
        object_name: impl AsRef<str>,
    ) where
        T: vk::Handle + Copy,
    {
        let _ = Self::set_debug_utils_object_name(this, object_handle, object_name);
    }

    /// Stores a Vulkan name in private-data metadata.
    ///
    /// Returns without doing any work when `VK_EXT_private_data` is not available.
    pub(crate) fn try_set_private_data_object_name<T>(
        this: &Self,
        object_type: vk::ObjectType,
        object_handle: T,
        object_name: impl AsRef<str>,
    ) where
        T: vk::Handle + Copy,
    {
        let _ = Self::set_private_data_object_name(this, object_type, object_handle, object_name);
    }

    fn try_vk_ext_debug_utils(this: &Self) -> Result<&ext::debug_utils::Device, DriverError> {
        this.inner
            .vk_ext_debug_utils
            .as_ref()
            .ok_or(DriverError::Unsupported)
    }

    fn try_vk_ext_private_data(this: &Self) -> Result<&ext::private_data::Device, DriverError> {
        this.inner
            .vk_ext_private_data
            .as_ref()
            .ok_or(DriverError::Unsupported)
    }

    /// Waits for a single fence to signal.
    #[profiling::function]
    pub(crate) fn wait_for_fence(this: &Self, fence: &vk::Fence) -> Result<(), DriverError> {
        Device::wait_for_fences(this, slice::from_ref(fence))
    }

    /// Waits for all fences in `fences` to signal.
    ///
    /// This first performs a short poll so uncontended waits return quickly, then falls back to an
    /// indefinite wait while logging unusually slow completions.
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

            let started = cfg!(debug_assertions).then(Instant::now);

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

            if let Some(started) = started {
                let elapsed = Instant::now() - started;
                let elapsed_millis = elapsed.as_millis();

                if elapsed_millis > 0 {
                    warn!("slow fence wait: {} ms", elapsed_millis);
                }
            }
        }

        Ok(())
    }

    /// Waits for device idle.
    pub fn wait_idle(this: &Self) -> Result<(), DriverError> {
        unsafe {
            this.device_wait_idle().map_err(|err| {
                warn!("unable to wait for device idle: {err}");

                match err {
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                    | vk::Result::ERROR_OUT_OF_HOST_MEMORY => DriverError::OutOfMemory,
                    vk::Result::ERROR_DEVICE_LOST | vk::Result::ERROR_VALIDATION_FAILED_EXT => {
                        DriverError::InvalidData
                    }
                    _ => DriverError::Unsupported,
                }
            })
        }
    }

    /// Provides mutable access to the device allocator under its internal lock.
    pub(crate) fn with_allocator<R>(this: &Self, f: impl FnOnce(&mut Allocator) -> R) -> R {
        let allocator = this.inner.allocator.lock();

        #[cfg(not(feature = "parking_lot"))]
        let allocator = allocator.expect("poisoned allocator lock");

        let mut allocator = allocator;

        f(&mut allocator)
    }

    fn with_object_metadata_ids<R>(
        this: &Self,
        f: impl FnOnce(&mut HashMap<(vk::ObjectType, u64), u64>) -> R,
    ) -> R {
        Self::with_private_data_metadata(this, |metadata| f(&mut metadata.object_metadata_ids))
    }

    fn with_private_data_metadata<R>(
        this: &Self,
        f: impl FnOnce(&mut PrivateDataMetadata) -> R,
    ) -> R {
        let mut metadata = this.inner.private_data_metadata.lock();

        #[cfg(not(feature = "parking_lot"))]
        let mut metadata = metadata.expect("poisoned private data metadata");

        f(&mut metadata)
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
        f.debug_struct(stringify!(Device))
            .field("handle", &self.inner.device.handle())
            .field("physical_device", &self.physical_device)
            .finish_non_exhaustive()
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
    /// Enables the Vulkan validation layers.
    ///
    /// This requires a Vulkan SDK installation and will panic when validation errors happen. See
    /// the LunarG [Vulkan Validation Layers] documentation for setup and behavior details.
    ///
    /// When `stderr` is attached to an interactive terminal, validation errors will park the
    /// callback thread for debugger attach.
    ///
    /// _NOTE:_ Consider turning OFF debug if you discover an unknown issue. Often the validation
    /// layers will report an error before other layers can provide additional context such as the
    /// API dump info or other messages. You might find the "actual" issue is detailed in those
    /// subsequent details.
    ///
    /// ## Platform-specific
    ///
    /// **macOS:** Has no effect unless the `loaded` feature is enabled.
    ///
    /// [Vulkan Validation Layers]: https://vulkan.lunarg.com/doc/sdk/latest/windows/validation_layers.html
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
    vk_ext_debug_utils: Option<ext::debug_utils::Device>,
    vk_ext_private_data: Option<ext::private_data::Device>,
    vk_khr_acceleration_structure: Option<khr::acceleration_structure::Device>,
    vk_khr_present_wait: Option<khr::present_wait::Device>,
    vk_khr_ray_tracing_pipeline: Option<khr::ray_tracing_pipeline::Device>,
    vk_khr_surface: Option<khr::surface::Instance>,
    vk_khr_swapchain: Option<khr::swapchain::Device>,
    vk_khr_synchronization2: Option<khr::synchronization2::Device>,
    private_data_slot: Option<vk::PrivateDataSlot>,
    private_data_name_id: AtomicU64,
    private_data_metadata: Mutex<PrivateDataMetadata>,
}

#[derive(Default)]
struct PrivateDataMetadata {
    object_metadata_ids: HashMap<(vk::ObjectType, u64), u64>,
    names: HashMap<u64, String>,
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

            if let (Some(vk_ext_private_data), Some(private_data_slot)) = (
                self.vk_ext_private_data.as_ref(),
                self.private_data_slot.take(),
            ) {
                vk_ext_private_data.destroy_private_data_slot(private_data_slot, None);
            }

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
