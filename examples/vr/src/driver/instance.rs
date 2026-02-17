use {
    log::{debug, error},
    openxr as xr,
    std::{
        ffi::c_void,
        fmt::{Debug, Formatter},
        mem::transmute,
        ops::Deref,
        sync::Arc,
    },
    vk_graph::driver::{
        ash::{
            self,
            vk::{self, Handle as _},
        },
        device::Device,
        instance::Instance as VkInstance,
    },
};

pub struct XrInstance {
    device: Arc<Device>,
    event_buf: xr::EventDataBuffer,
    instance: xr::Instance,
    system: xr::SystemId,
}

impl XrInstance {
    const VK_TARGET_VERSION: u32 = vk::make_api_version(
        0,
        Self::XR_TARGET_VERSION.major() as _,
        Self::XR_TARGET_VERSION.minor() as _,
        0,
    );
    const XR_TARGET_VERSION: xr::Version = xr::Version::new(1, 2, 0);

    pub fn new() -> Result<Self, InstanceCreateError> {
        let xr_entry = unsafe { xr::Entry::load().ok() };

        if xr_entry.is_none() {
            debug!("Using statically linked OpenXR")
        }

        let xr_entry = xr_entry.unwrap_or_else(xr::Entry::linked);
        let available_extensions = xr_entry.enumerate_extensions().unwrap_or_default();
        let mut required_extensions = xr::ExtensionSet::default();
        required_extensions.khr_vulkan_enable2 = available_extensions.khr_vulkan_enable2;

        if !required_extensions.khr_vulkan_enable2 {
            return Err(InstanceCreateError::VulkanUnsupported);
        }

        let app_info = xr::ApplicationInfo {
            application_name: "vk-graph-example-vr",
            application_version: 0,
            engine_name: "vk-graph-example-vr",
            engine_version: 0,
        };
        let xr_instance = xr_entry
            .create_instance(&app_info, &required_extensions, &[])
            .map_err(|err| {
                error!("Unable to create OpenXR instance: {err}");

                InstanceCreateError::OpenXRUnsupported
            })?;
        let xr::InstanceProperties {
            runtime_name,
            runtime_version,
        } = xr_instance.properties().map_err(|err| {
            error!("OpenXR instance properties: {err}");

            InstanceCreateError::OpenXRUnsupported
        })?;

        debug!(
            "loaded OpenXR runtime: {} {}",
            runtime_name, runtime_version
        );

        let system = xr_instance
            .system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)
            .map_err(|err| {
                error!("OpenXR system: {err}");

                InstanceCreateError::OpenXRUnsupported
            })?;
        if !xr_instance
            .enumerate_environment_blend_modes(system, xr::ViewConfigurationType::PRIMARY_STEREO)
            .unwrap_or_default()
            .contains(&xr::EnvironmentBlendMode::OPAQUE)
        {
            error!("OpenXR opaque blend mode not supported");

            return Err(InstanceCreateError::OpenXRUnsupported);
        }

        let xr::vulkan::Requirements {
            max_api_version_supported,
            min_api_version_supported,
        } = xr_instance
            .graphics_requirements::<xr::Vulkan>(system)
            .map_err(|err| {
                error!("OpenXR vulkan requirements: {err}");

                InstanceCreateError::OpenXRUnsupported
            })?;

        if min_api_version_supported > Self::XR_TARGET_VERSION
            || max_api_version_supported.major() < Self::XR_TARGET_VERSION.major()
        {
            error!(
                "OpenXR runtime requires Vulkan version > {}, < {}.0.0",
                min_api_version_supported,
                max_api_version_supported.major() + 1
            );

            return Err(InstanceCreateError::VulkanUnsupported);
        }

        let app_info = vk::ApplicationInfo::default().api_version(Self::VK_TARGET_VERSION);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        unsafe {
            let entry = ash::Entry::load().map_err(|err| {
                error!("Vulkan entry point: {err}");

                InstanceCreateError::VulkanUnsupported
            })?;

            let get_instance_proc_addr = {
                type Fn<T> =
                    unsafe extern "system" fn(T, *const i8) -> Option<unsafe extern "system" fn()>;
                type AshFn = Fn<vk::Instance>;
                type OpenXrFn = Fn<*const c_void>;
                transmute::<AshFn, OpenXrFn>(entry.static_fn().get_instance_proc_addr)
            };

            let vk_instance = {
                let vk_instance = xr_instance
                    .create_vulkan_instance(
                        system,
                        get_instance_proc_addr,
                        &create_info as *const _ as *const _,
                    )
                    .map_err(|err| {
                        error!("OpenXR unable to create Vulkan instance: {err}");

                        InstanceCreateError::OpenXRUnsupported
                    })?
                    .map_err(vk::Result::from_raw)
                    .map_err(|err| {
                        error!("Vulkan instance create: {err}");

                        InstanceCreateError::VulkanUnsupported
                    })?;
                let vk_instance = vk::Instance::from_raw(vk_instance as _);

                VkInstance::from_entry(entry, vk_instance).map_err(|err| {
                    error!("Vulkan instance load: {err}");

                    InstanceCreateError::VulkanUnsupported
                })?
            };

            let physical_device = vk::PhysicalDevice::from_raw(
                xr_instance
                    .vulkan_graphics_device(system, vk_instance.handle().as_raw() as _)
                    .map_err(|err| {
                        error!("OpenXR unable to create Vulkan graphics device: {err}");

                        InstanceCreateError::OpenXRUnsupported
                    })? as _,
            );
            let physical_device = VkInstance::physical_device(&vk_instance, physical_device)
                .map_err(|err| {
                    error!("Vulkan physical device: {err}");

                    InstanceCreateError::VulkanUnsupported
                })?;

            let ash_device = physical_device
                .create_ash_device(true, |create_info| {
                    let device = xr_instance
                        .create_vulkan_device(
                            system,
                            get_instance_proc_addr,
                            physical_device.handle.as_raw() as _,
                            &create_info as *const _ as *const _,
                        )
                        .map_err(|err| {
                            error!("OpenXR unable to create Vulkan device: {err}");

                            vk::Result::ERROR_INITIALIZATION_FAILED
                        })?
                        .map_err(vk::Result::from_raw)?;
                    let device = vk::Device::from_raw(device as _);

                    Ok(ash::Device::load(vk_instance.fp_v1_0(), device))
                })
                .map_err(|err| {
                    error!("Vulkan device: {err}");

                    InstanceCreateError::VulkanUnsupported
                })?;
            let device = Arc::new(Device::load(physical_device, ash_device, true).map_err(
                |err| {
                    error!("Vulkan device: {err}");

                    InstanceCreateError::VulkanUnsupported
                },
            )?);
            let event_buf = xr::EventDataBuffer::new();

            Ok(Self {
                device,
                event_buf,
                instance: xr_instance,
                system,
            })
        }
    }

    #[inline]
    pub fn create_session(
        this: &Self,
        queue_family_index: u32,
        queue_index: u32,
    ) -> xr::Result<(
        xr::Session<xr::Vulkan>,
        xr::FrameWaiter,
        xr::FrameStream<xr::Vulkan>,
    )> {
        unsafe {
            this.instance.create_session::<xr::Vulkan>(
                this.system,
                &xr::vulkan::SessionCreateInfo {
                    instance: this.device.physical_device.instance.handle().as_raw() as _,
                    physical_device: this.device.physical_device.handle.as_raw() as _,
                    device: this.device.handle().as_raw() as _,
                    queue_family_index,
                    queue_index,
                },
            )
        }
    }

    pub fn device(this: &Self) -> &Arc<Device> {
        &this.device
    }

    #[inline]
    pub fn enumerate_view_configuration_views(
        this: &Self,
        ty: xr::ViewConfigurationType,
    ) -> xr::Result<Vec<xr::ViewConfigurationView>> {
        this.enumerate_view_configuration_views(this.system, ty)
    }

    /// Get the next event, if available
    ///
    /// Returns immediately regardless of whether an event was available.
    #[inline]
    pub fn poll_event(this: &mut Self) -> xr::Result<Option<xr::Event<'_>>> {
        this.instance.poll_event(&mut this.event_buf)
    }

    #[allow(dead_code)]
    pub fn system(this: &Self) -> xr::SystemId {
        this.system
    }
}

impl Debug for XrInstance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Instance")
    }
}

impl Deref for XrInstance {
    type Target = xr::Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

#[derive(Debug)]
pub enum InstanceCreateError {
    OpenXRUnsupported,
    VulkanUnsupported,
}
