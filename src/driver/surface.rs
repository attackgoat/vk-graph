//! Native platform window surface types.

use {
    super::{DriverError, Instance, device::Device},
    ash::vk,
    ash_window::create_surface,
    log::warn,
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    std::{
        fmt::{Debug, Formatter},
        ops::Deref,
        sync::Arc,
        thread::panicking,
    },
};

/// Smart pointer handle to a [`vk::SurfaceKHR`] object.
#[repr(C)]
pub struct Surface {
    /// The device which owns this buffer resource.
    ///
    /// _Note:_ This field is read-only.
    #[cfg(doc)]
    pub device: Arc<Device>,

    #[cfg(not(doc))]
    device: Arc<Device>,

    /// The native Vulkan resource handle of this surface.
    ///
    /// _Note:_ This field is read-only.
    #[cfg(doc)]
    pub handle: vk::SurfaceKHR,

    #[cfg(not(doc))]
    handle: vk::SurfaceKHR,
}

#[doc(hidden)]
#[repr(C)]
pub struct SurfaceRef {
    pub device: Arc<Device>,
    pub handle: vk::SurfaceKHR,
}

impl Surface {
    /// Query surface capabilities
    pub fn capabilities(&self) -> Result<vk::SurfaceCapabilitiesKHR, DriverError> {
        let surface_ext = Device::expect_surface_ext(&self.device);

        unsafe {
            surface_ext
                .get_physical_device_surface_capabilities(*self.device.physical_device, self.handle)
        }
        .inspect_err(|err| warn!("unable to get surface capabilities: {err}"))
        .or(Err(DriverError::Unsupported))
    }

    /// Create a surface from a raw window display handle.
    ///
    /// `device` must have been created with platform specific surface extensions enabled, acquired
    /// through [`Device::create_display_window`].
    #[profiling::function]
    pub fn create(
        device: &Arc<Device>,
        display: impl HasDisplayHandle,
        window: impl HasWindowHandle,
    ) -> Result<Self, DriverError> {
        let device = Arc::clone(device);
        let instance = Device::instance(&device);
        let display_handle = display.display_handle().map_err(|err| {
            warn!("{err}");

            DriverError::Unsupported
        })?;
        let window_handle = window.window_handle().map_err(|err| {
            warn!("{err}");

            DriverError::Unsupported
        })?;
        let handle = unsafe {
            create_surface(
                Instance::entry(instance),
                instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            )
        }
        .map_err(|err| {
            warn!("Unable to create surface: {err}");

            DriverError::Unsupported
        })?;

        Ok(Self { device, handle })
    }

    /// Lists the supported surface formats.
    #[profiling::function]
    pub fn formats(&self) -> Result<Vec<vk::SurfaceFormatKHR>, DriverError> {
        unsafe {
            self.device
                .surface_ext
                .as_ref()
                .unwrap()
                .get_physical_device_surface_formats(*self.device.physical_device, self.handle)
                .map_err(|err| {
                    warn!("Unable to get surface formats: {err}");

                    DriverError::Unsupported
                })
        }
    }

    /// Helper function to automatically select the best UNORM format, if one is available.
    #[profiling::function]
    pub fn linear(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
        formats
            .iter()
            .find(|&&vk::SurfaceFormatKHR { format, .. }| {
                matches!(
                    format,
                    vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM
                )
            })
            .copied()
    }

    /// Helper function to automatically select the best UNORM format.
    ///
    /// **_NOTE:_** The default surface format is undefined, and although legal the results _may_
    /// not support presentation. You should prefer to use [`Surface::linear`] and fall back to
    /// supported values manually.
    pub fn linear_or_default(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        Self::linear(formats).unwrap_or_else(|| formats.first().copied().unwrap_or_default())
    }

    /// Query supported presentation modes.
    pub fn present_modes(&self) -> Result<Vec<vk::PresentModeKHR>, DriverError> {
        let surface_ext = Device::expect_surface_ext(&self.device);

        unsafe {
            surface_ext.get_physical_device_surface_present_modes(
                *self.device.physical_device,
                self.handle,
            )
        }
        .inspect_err(|err| warn!("unable to get surface present modes: {err}"))
        .or(Err(DriverError::Unsupported))
    }

    /// Helper function to automatically select the best sRGB format, if one is available.
    #[profiling::function]
    pub fn srgb(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
        formats
            .iter()
            .find(
                |&&vk::SurfaceFormatKHR {
                     color_space,
                     format,
                 }| {
                    matches!(color_space, vk::ColorSpaceKHR::SRGB_NONLINEAR)
                        && matches!(
                            format,
                            vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
                        )
                },
            )
            .copied()
    }

    /// Helper function to automatically select the best sRGB format.
    ///
    /// **_NOTE:_** The default surface format is undefined, and although legal the results _may_
    /// not support presentation. You should prefer to use [`Surface::srgb`] and fall back to
    /// supported values manually.
    pub fn srgb_or_default(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        Self::srgb(formats).unwrap_or_else(|| formats.first().copied().unwrap_or_default())
    }
}

impl Debug for Surface {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Surface")
    }
}

#[doc(hidden)]
impl Deref for Surface {
    type Target = SurfaceRef;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self as *const Self::Target) }
    }
}

impl Drop for Surface {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        let surface_ext = Device::expect_surface_ext(&self.device);

        unsafe {
            surface_ext.destroy_surface(self.handle, None);
        }
    }
}
