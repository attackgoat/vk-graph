//! Native platform window surface types.

use {
    super::{DriverError, device::Device, instance::Instance},
    ash::vk,
    ash_window::create_surface,
    log::warn,
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    std::{
        fmt::{Debug, Formatter},
        thread::panicking,
    },
};

/// Smart pointer handle to a [`vk::SurfaceKHR`] object.
#[read_only::cast]
pub struct Surface {
    /// The device which owns this surface resource.
    ///
    /// _Note:_ This field is read-only.
    pub device: Device,

    /// The native Vulkan resource handle of this surface.
    ///
    /// _Note:_ This field is read-only.
    pub handle: vk::SurfaceKHR,
}

impl Surface {
    /// Query surface capabilities
    pub fn capabilities(&self) -> Result<vk::SurfaceCapabilitiesKHR, DriverError> {
        let khr_surface = Device::expect_vk_khr_surface(&self.device);

        unsafe {
            khr_surface.get_physical_device_surface_capabilities(
                self.device.physical_device.handle,
                self.handle,
            )
        }
        .map_err(|err| {
            warn!("unable to get surface capabilities: {err}");

            DriverError::Unsupported
        })
    }

    /// Create a surface from a raw window display handle.
    ///
    /// `device` must have been created with platform specific surface extensions enabled.
    #[profiling::function]
    pub fn create(
        device: &Device,
        display: impl HasDisplayHandle,
        window: impl HasWindowHandle,
    ) -> Result<Self, DriverError> {
        let device = device.clone();

        let display_handle = display.display_handle().map_err(|err| {
            warn!("unable to get display handle: {err}");

            DriverError::Unsupported
        })?;
        let window_handle = window.window_handle().map_err(|err| {
            warn!("unable to get window handle: {err}");

            DriverError::Unsupported
        })?;

        let handle = unsafe {
            create_surface(
                Instance::entry(&device.physical_device.instance),
                &device.physical_device.instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            )
        }
        .map_err(|err| {
            warn!("unable to create surface: {err}");

            DriverError::Unsupported
        })?;

        Ok(Self { device, handle })
    }

    /// Lists the supported surface formats.
    #[profiling::function]
    pub fn formats(&self) -> Result<Vec<vk::SurfaceFormatKHR>, DriverError> {
        let khr_surface = Device::expect_vk_khr_surface(&self.device);

        unsafe {
            khr_surface.get_physical_device_surface_formats(
                self.device.physical_device.handle,
                self.handle,
            )
        }
        .map_err(|err| {
            warn!("unable to get surface formats: {err}");

            DriverError::Unsupported
        })
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

    /// Returns `true` if the given queue family supports presentation on this surface.
    pub fn physical_device_support(&self, queue_family_index: u32) -> Result<bool, DriverError> {
        let khr_surface = Device::expect_vk_khr_surface(&self.device);

        unsafe {
            khr_surface
                .get_physical_device_surface_support(
                    self.device.physical_device.handle,
                    queue_family_index,
                    self.handle,
                )
                .map_err(|err| {
                    warn!("unable to get physical device support: {err}");

                    match err {
                        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                        | vk::Result::ERROR_OUT_OF_HOST_MEMORY => DriverError::OutOfMemory,
                        vk::Result::ERROR_SURFACE_LOST_KHR => DriverError::InvalidData,
                        _ => DriverError::Unsupported,
                    }
                })
        }
    }

    /// Query supported presentation modes.
    pub fn present_modes(&self) -> Result<Vec<vk::PresentModeKHR>, DriverError> {
        let khr_surface = Device::expect_vk_khr_surface(&self.device);

        unsafe {
            khr_surface
                .get_physical_device_surface_present_modes(
                    self.device.physical_device.handle,
                    self.handle,
                )
                .map_err(|err| {
                    warn!("unable to get present modes: {err}");

                    DriverError::Unsupported
                })
        }
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

impl Drop for Surface {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        let khr_surface = Device::expect_vk_khr_surface(&self.device);

        unsafe {
            khr_surface.destroy_surface(self.handle, None);
        }
    }
}

impl Eq for Surface {}

impl PartialEq for Surface {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

#[cfg(test)]
mod test {
    use super::Surface;
    use ash::vk;

    #[test]
    fn linear_prefers_known_unorm_formats() {
        let formats = [
            vk::SurfaceFormatKHR {
                format: vk::Format::R8G8B8A8_SRGB,
                ..Default::default()
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::R8G8B8A8_UNORM,
                ..Default::default()
            },
        ];

        assert_eq!(
            Surface::linear(&formats).unwrap().format,
            vk::Format::R8G8B8A8_UNORM
        );
    }

    #[test]
    fn linear_or_default_falls_back_to_first_format() {
        let formats = [vk::SurfaceFormatKHR {
            format: vk::Format::R16G16B16A16_SFLOAT,
            ..Default::default()
        }];

        assert_eq!(
            Surface::linear_or_default(&formats).format,
            vk::Format::R16G16B16A16_SFLOAT
        );
    }

    #[test]
    fn srgb_prefers_known_srgb_formats() {
        let formats = [
            vk::SurfaceFormatKHR {
                color_space: vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT,
                format: vk::Format::B8G8R8A8_SRGB,
            },
            vk::SurfaceFormatKHR {
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                format: vk::Format::R8G8B8A8_SRGB,
            },
        ];

        assert_eq!(
            Surface::srgb(&formats).unwrap().format,
            vk::Format::R8G8B8A8_SRGB
        );
    }

    #[test]
    fn srgb_or_default_falls_back_to_first_format() {
        let formats = [vk::SurfaceFormatKHR {
            color_space: vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT,
            format: vk::Format::R16G16B16A16_SFLOAT,
        }];

        assert_eq!(
            Surface::srgb_or_default(&formats).format,
            vk::Format::R16G16B16A16_SFLOAT
        );
    }
}
