//! Vulkan initialization types

use {
    super::{DriverError, physical_device::PhysicalDevice},
    ash::{ext, khr, vk, vk::Handle},
    derive_builder::{Builder, UninitializedFieldError},
    log::{debug, error, trace, warn},
    raw_window_handle::{HasDisplayHandle, RawDisplayHandle},
    std::{
        error::Error,
        ffi::CStr,
        fmt::{Debug, Display, Formatter},
        ops::Deref,
        sync::Arc,
        thread::panicking,
    },
};

#[cfg(any(not(target_os = "macos"), feature = "loaded"))]
use {
    log::{Level, Metadata, info, logger},
    std::{
        env::var,
        ffi::c_void,
        process::{abort, id},
        thread::{current, park},
    },
};

#[cfg(any(not(target_os = "macos"), feature = "loaded"))]
const SKIP_VALIDATION_PARK_ENV: &str = "VK_GRAPH_SKIP_VALIDATION_PARK";

#[cfg(target_os = "macos")]
use std::env::set_var;

#[cfg(any(not(target_os = "macos"), feature = "loaded"))]
unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    if panicking() {
        return vk::FALSE;
    }

    assert!(!callback_data.is_null());

    let callback_data = unsafe { &*callback_data };
    let message = if callback_data.p_message.is_null() {
        "<missing Vulkan validation message>"
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message) }
            .to_str()
            .unwrap_or("<invalid Vulkan validation message>")
    };

    if !callback_data.p_message_id_name.is_null() {
        let vuid = unsafe { CStr::from_ptr(callback_data.p_message_id_name) }
            .to_str()
            .unwrap_or("<invalid Vulkan validation message ID name>");
        if vuid != "Loader Message" {
            debug!("{vuid}");
        }
    };

    let is_error = message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR);

    // HACK: This is not production-quality
    // TODO: This was debugged and the issue has not been found, so this may or may not be valid
    // The validation layer reports `UNASSIGNED-Threading-MultipleThreads-Write` when two threads
    // touch different VkQueue handles at the same time. Ignoring until the issue is found.
    if is_error
        && message.contains("THREADING ERROR")
        && message.contains("VkQueue is simultaneously used")
    {
        info!("ignoring: {message}");

        return vk::FALSE;
    }

    if is_error {
        error!("{message}");
    } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        warn!("{message}");
    } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        info!("{message}");
    } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE) {
        debug!("{message}");
    }

    if !is_error {
        return vk::FALSE;
    }

    if !logger().enabled(&Metadata::builder().level(Level::Debug).build())
        || var("RUST_LOG")
            .map(|rust_log| rust_log.is_empty())
            .unwrap_or(true)
    {
        eprintln!(
            "note: run with `RUST_LOG=trace` environment variable to display more information"
        );
        eprintln!("note: see https://github.com/rust-lang/log#in-executables");
        abort()
    }

    if current().name() != Some("main") {
        warn!("invalid validation callback thread: child thread")
    }

    if var(SKIP_VALIDATION_PARK_ENV)
        .map(|value| !matches!(value.as_str(), "" | "0" | "false" | "False" | "FALSE"))
        .unwrap_or(false)
    {
        warn!("validation callback park skipped; execution will continue");
        logger().flush();

        return vk::FALSE;
    }

    debug!(
        "parking validation callback thread `{}` for debugger attach to pid {}",
        current().name().unwrap_or_default(),
        id()
    );

    logger().flush();
    park();

    vk::FALSE
}

#[cfg(any(not(target_os = "macos"), feature = "loaded"))]
fn debug_extension_names() -> &'static [&'static CStr] {
    &[ext::debug_utils::NAME]
}

#[cfg(all(target_os = "macos", not(feature = "loaded")))]
fn debug_extension_names() -> &'static [&'static CStr] {
    &[]
}

#[cfg(any(not(target_os = "macos"), feature = "loaded"))]
fn debug_layer_names() -> &'static [&'static CStr] {
    &[c"VK_LAYER_KHRONOS_validation"]
}

#[cfg(all(target_os = "macos", not(feature = "loaded")))]
fn debug_layer_names() -> &'static [&'static CStr] {
    &[]
}

// Copied from ash_window::enumerate_required_extensions to change the signature.
fn display_extension_names(
    display_handle: RawDisplayHandle,
) -> Result<&'static [&'static CStr], DriverError> {
    let extensions = match display_handle {
        RawDisplayHandle::Windows(_) => &[khr::surface::NAME, khr::win32_surface::NAME],
        RawDisplayHandle::Wayland(_) => &[khr::surface::NAME, khr::wayland_surface::NAME],
        RawDisplayHandle::Xlib(_) => &[khr::surface::NAME, khr::xlib_surface::NAME],
        RawDisplayHandle::Xcb(_) => &[khr::surface::NAME, khr::xcb_surface::NAME],
        RawDisplayHandle::Android(_) => &[khr::surface::NAME, khr::android_surface::NAME],
        RawDisplayHandle::AppKit(_) | RawDisplayHandle::UiKit(_) => {
            &[khr::surface::NAME, ext::metal_surface::NAME]
        }
        _ => {
            warn!("unsupported display handle type: {display_handle:?}");

            return Err(DriverError::Unsupported);
        }
    };

    Ok(extensions)
}

// Estimates surface extension support.
//
// Imported instances do not expose their enabled extension list, so we infer support by
// checking that the VK_KHR_surface entry points resolve for this instance handle.
fn has_surface_ext(entry: &ash::Entry, instance: vk::Instance) -> bool {
    [
        c"vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
        c"vkGetPhysicalDeviceSurfaceFormatsKHR",
        c"vkGetPhysicalDeviceSurfacePresentModesKHR",
        c"vkGetPhysicalDeviceSurfaceSupportKHR",
        c"vkDestroySurfaceKHR",
    ]
    .into_iter()
    .all(|name| unsafe {
        entry
            .get_instance_proc_addr(instance, name.as_ptr())
            .is_some()
    })
}

/// Vulkan API version.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum ApiVersion {
    /// Version `1.2`.
    #[default]
    Vulkan12,

    /// Version `1.3`.
    Vulkan13,
}

impl ApiVersion {
    /// Returns a version parsed from a native Vulkan value.
    pub fn try_parse_vk_api_version(version: u32) -> Result<Self, ParseApiVersionError> {
        Self::try_from(version)
    }

    /// Vulkan API major version number component. Ex: `vX.0.0-0`.
    ///
    /// Always one.
    pub fn major(self) -> u32 {
        1
    }

    /// Vulkan API minor version number component. Ex: `v0.X.0-0`.
    pub fn minor(self) -> u32 {
        match self {
            Self::Vulkan12 => 2,
            Self::Vulkan13 => 3,
        }
    }

    /// Vulkan API minor version number component. Ex: `v0.0.X-0`.
    ///
    /// Always zero.
    pub fn patch(self) -> u32 {
        0
    }

    /// Returns a native Vulkan value.
    pub fn to_vk_api_version(self) -> u32 {
        self.into()
    }

    /// Vulkan API variant version number component. Ex: `v0.0.0-X`.
    ///
    /// Always zero.
    pub fn variant(self) -> u32 {
        0
    }
}

impl From<ApiVersion> for u32 {
    fn from(val: ApiVersion) -> Self {
        vk::make_api_version(val.variant(), val.major(), val.minor(), val.patch())
    }
}

impl TryFrom<u32> for ApiVersion {
    type Error = ParseApiVersionError;

    fn try_from(val: u32) -> Result<Self, Self::Error> {
        let major = vk::api_version_major(val);
        let minor = vk::api_version_minor(val);
        let patch = vk::api_version_patch(val);
        let variant = vk::api_version_variant(val);

        if variant != 0 || major != 1 || minor < 2 {
            return Err(ParseApiVersionError {
                major,
                minor,
                patch,
                variant,
            });
        }

        Ok(match minor {
            2 => ApiVersion::Vulkan12,
            _ => ApiVersion::Vulkan13,
        })
    }
}

/// There is no global state in Vulkan and all per-application state is stored in a VkInstance
/// object.
///
/// Creating an Instance initializes the Vulkan library and allows the application to pass
/// information about itself to the implementation.
#[read_only::embed]
#[allow(private_interfaces)]
pub struct Instance {
    /// Information used to create this resource.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub info: InstanceInfo,

    #[readonly]
    pub(self) inner: Arc<InstanceInner>,

    /// True if `VK_KHR_surface` is enabled on this instance.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub surface_ext: bool,
}

impl Clone for Instance {
    fn clone(&self) -> Self {
        Self {
            read_only: ReadOnlyInstance {
                info: self.info,
                surface_ext: self.surface_ext,
                inner: self.inner.clone(),
            },
        }
    }
}

impl Instance {
    /// The most recent supported version of Vulkan.
    pub const LATEST_API_VERSION: ApiVersion = ApiVersion::Vulkan13;

    #[deprecated = "use create"]
    #[doc(hidden)]
    pub fn new(info: impl Into<InstanceInfo>) -> Result<Self, DriverError> {
        Self::create(info)
    }

    /// Creates a new Vulkan instance.
    ///
    /// This constructor is intended for headless or manually managed setups. It does not infer or
    /// enable display platform surface extensions. Use [`Self::try_from_display`] when the
    /// resulting instance must be capable of later surface creation.
    #[profiling::function]
    pub fn create(info: impl Into<InstanceInfo>) -> Result<Self, DriverError> {
        Self::create_with_extension_names(info.into(), &[])
    }

    fn create_with_extension_names(
        info: InstanceInfo,
        extra_extension_names: &[&CStr],
    ) -> Result<Self, DriverError> {
        // Required to enable non-uniform descriptor indexing (bindless)
        #[cfg(target_os = "macos")]
        unsafe {
            set_var("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "1");
        }

        // Link the Vulkan loader dynamically (default feature).
        #[cfg(feature = "loaded")]
        let entry = unsafe {
            ash::Entry::load().map_err(|err| {
                error!("unable to load Vulkan driver: {err}");

                DriverError::Unsupported
            })?
        };

        // Link the Vulkan loader statically if explicitly requested
        #[cfg(not(feature = "loaded"))]
        let entry = {
            #[cfg(not(target_os = "macos"))]
            let entry = ash::Entry::linked();

            // On MacOS, by default link molten-vk statically using ash-molten.
            #[cfg(target_os = "macos")]
            let entry = ash_molten::load();
        };

        let mut extension_names = info
            .extension_names
            .iter()
            .chain(extra_extension_names)
            .copied()
            .collect::<Vec<_>>();

        if info.debug {
            extension_names.extend(debug_extension_names());
        }

        // If linking dynamically on MacOS, we require a few additional extensions.
        // Based on "Encountered VK_ERROR_INCOMPATIBLE_DRIVER" section in:
        // https://vulkan.lunarg.com/doc/view/latest/mac/getting_started.html
        #[cfg(all(target_os = "macos", feature = "loaded"))]
        {
            extension_names.extend(&[
                ash::khr::get_physical_device_properties2::NAME,
                ash::khr::portability_enumeration::NAME,
            ]);
        }

        let surface_ext = extension_names.contains(&khr::surface::NAME);

        let extension_name_ptrs = extension_names
            .iter()
            .copied()
            .map(CStr::as_ptr)
            .collect::<Box<_>>();

        let mut layer_names = Vec::with_capacity(info.debug as _);

        if info.debug {
            layer_names.extend(debug_layer_names());
        }

        let layer_name_ptrs = layer_names
            .iter()
            .copied()
            .map(CStr::as_ptr)
            .collect::<Box<_>>();

        let app_desc =
            vk::ApplicationInfo::default().api_version(info.api_version.to_vk_api_version());
        let instance_desc = vk::InstanceCreateInfo::default()
            .application_info(&app_desc)
            .enabled_layer_names(&layer_name_ptrs)
            .enabled_extension_names(&extension_name_ptrs);

        // Molten-vk doesn't support the full Vulkan feature set, hence the portability flag needs
        // to be set.
        #[cfg(all(target_os = "macos", feature = "loaded"))]
        let instance_desc = instance_desc.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);

        #[cfg(any(not(target_os = "macos"), feature = "loaded"))]
        let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback));

        #[cfg(any(not(target_os = "macos"), feature = "loaded"))]
        let instance_desc = if info.debug {
            instance_desc.push_next(&mut debug_create_info)
        } else {
            instance_desc
        };

        let instance = unsafe {
            entry.create_instance(&instance_desc, None).map_err(|_| {
                if info.debug {
                    warn!("debug may only be enabled with a valid Vulkan SDK installation");
                }

                error!(
                    "Vulkan driver does not support API v{}",
                    match info.api_version {
                        ApiVersion::Vulkan12 => "1.2",
                        ApiVersion::Vulkan13 => "1.3",
                    }
                );

                for layer_name in &layer_names {
                    debug!("Layer: {:?}", layer_name);
                }

                for extension_name in extension_names {
                    debug!("Extension: {:?}", extension_name);
                }

                DriverError::Unsupported
            })?
        };

        trace!("created a Vulkan instance");

        #[cfg(all(target_os = "macos", not(feature = "loaded")))]
        let debug_utils = None;

        #[cfg(any(not(target_os = "macos"), feature = "loaded"))]
        let debug_utils = if info.debug {
            let debug_utils = ext::debug_utils::Instance::new(&entry, &instance);
            let debug_messenger =
                unsafe { debug_utils.create_debug_utils_messenger(&debug_create_info, None) }
                    .map_err(|err| {
                        unsafe {
                            instance.destroy_instance(None);
                        }

                        error!("unable to create debug utils messenger: {err}");

                        DriverError::Unsupported
                    })?;

            Some((debug_utils, debug_messenger))
        } else {
            None
        };

        Ok(Self {
            read_only: ReadOnlyInstance {
                info,
                inner: Arc::new(InstanceInner {
                    debug_utils,
                    entry,
                    instance,
                    instance_created: true,
                }),
                surface_ext,
            },
        })
    }

    /// The ash entrypoint used to load Vulkan instance functions.
    pub fn entry(this: &Self) -> &ash::Entry {
        &this.inner.entry
    }

    #[deprecated = "use try_from_entry"]
    #[doc(hidden)]
    pub fn from_entry(entry: ash::Entry, instance: vk::Instance) -> Result<Self, DriverError> {
        Self::try_from_entry(entry, instance)
    }

    /// Returns a wrapper structure for a physical device of this instance.
    #[profiling::function]
    pub fn physical_device(
        this: &Self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<PhysicalDevice, DriverError> {
        let physical_device = PhysicalDevice::new(this.clone(), physical_device)?;
        if let Err(err) =
            ApiVersion::try_parse_vk_api_version(physical_device.properties_v1_0.api_version)
        {
            warn!(
                "unsupported physical device `{}`: {err}",
                physical_device.properties_v1_0.device_name
            );

            return Err(DriverError::Unsupported);
        }

        Ok(physical_device)
    }

    /// Returns the available physical devices of this instance.
    #[profiling::function]
    pub fn physical_devices(
        this: &Self,
    ) -> Result<impl IntoIterator<Item = PhysicalDevice>, DriverError> {
        let physical_devices = unsafe { this.enumerate_physical_devices() }.map_err(|err| {
            error!("unable to enumerate physical devices: {err}");

            match err {
                vk::Result::ERROR_INITIALIZATION_FAILED => DriverError::Unsupported,
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                    DriverError::OutOfMemory
                }
                vk::Result::ERROR_VALIDATION_FAILED_EXT => DriverError::InvalidData,
                _ => {
                    warn!("unexpected enumerate_physical_devices error: {err}");

                    DriverError::Unsupported
                }
            }
        })?;

        Ok(physical_devices
            .into_iter()
            .enumerate()
            .filter_map(|(idx, physical_device)| {
                let res = PhysicalDevice::new(this.clone(), physical_device);

                if let Err(err) = &res {
                    warn!("unsupported physical device #{idx}: {err}");
                }

                res.ok().filter(|physical_device| {
                    ApiVersion::try_parse_vk_api_version(
                        physical_device.properties_v1_0.api_version,
                    )
                    .inspect_err(|err| {
                        debug!(
                            "unsupported physical device `{}`: {err}",
                            physical_device.properties_v1_0.device_name
                        );
                    })
                    .is_ok()
                })
            }))
    }

    /// Creates a new Vulkan instance with the platform surface extensions required by the provided
    /// display handle.
    #[profiling::function]
    pub fn try_from_display(
        display: impl HasDisplayHandle,
        info: impl Into<InstanceInfo>,
    ) -> Result<Self, DriverError> {
        let display_handle = display.display_handle().map_err(|err| {
            warn!("unable to get display handle: {err}");

            DriverError::Unsupported
        })?;
        let display_extension_names =
            display_extension_names(display_handle.as_raw()).map_err(|err| {
                warn!("unable to enumerate display extensions: {err}");

                DriverError::Unsupported
            })?;

        Self::create_with_extension_names(info.into(), display_extension_names)
    }

    /// Loads an existing Vulkan instance that may have been created by other means.
    ///
    /// This is useful when you want to use a Vulkan instance created by some other library, such
    /// as OpenXR.
    #[profiling::function]
    pub fn try_from_entry(entry: ash::Entry, instance: vk::Instance) -> Result<Self, DriverError> {
        if instance == vk::Instance::null() {
            warn!("invalid VkInstance handle: null");

            return Err(DriverError::InvalidData);
        }

        let api_version = unsafe { entry.try_enumerate_instance_version() }
            .map_err(|err| match err {
                vk::Result::ERROR_OUT_OF_HOST_MEMORY => DriverError::OutOfMemory,
                vk::Result::ERROR_VALIDATION_FAILED_EXT => DriverError::InvalidData,
                err => {
                    error!("unable to enumerate instance version: {err}");

                    DriverError::Unsupported
                }
            })?
            .unwrap_or_else(|| {
                // The implementation *should* provide a version. If it does not we just send it.
                Self::LATEST_API_VERSION.to_vk_api_version()
            })
            .try_into()
            .map_err(|err| {
                warn!("unsupported instance: {err}");

                DriverError::Unsupported
            })?;
        let surface_ext = has_surface_ext(&entry, instance);

        let instance = unsafe { ash::Instance::load(entry.static_fn(), instance) };

        Ok(Self {
            read_only: ReadOnlyInstance {
                info: InstanceInfo {
                    api_version,
                    ..Default::default()
                },
                inner: Arc::new(InstanceInner {
                    debug_utils: None,
                    entry,
                    instance,
                    instance_created: false,
                }),
                surface_ext,
            },
        })
    }
}

impl Debug for Instance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Instance")
    }
}

/// Information used to create an [`Instance`] instance.
#[derive(Builder, Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build", error = "UninitializedFieldError"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct InstanceInfo {
    /// The Vulkan API version to target.
    #[builder(default = "ApiVersion::Vulkan13")]
    pub api_version: ApiVersion,

    /// Enables Vulkan validation layers.
    ///
    /// This requires a Vulkan SDK installation and will cause validation errors to introduce
    /// panics as they happen.
    ///
    /// Set `VK_GRAPH_SKIP_VALIDATION_PARK=1` to keep logging validation errors without parking the
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

    /// Required Vulkan instance extension names to load.
    #[builder(default)]
    pub extension_names: &'static [&'static CStr],
}

impl InstanceInfo {
    /// Creates a default `InstanceInfoBuilder`.
    pub fn builder() -> InstanceInfoBuilder {
        Default::default()
    }

    /// Converts a `InstanceInfo` into a `InstanceInfoBuilder`.
    pub fn into_builder(self) -> InstanceInfoBuilder {
        InstanceInfoBuilder {
            api_version: Some(self.api_version),
            debug: Some(self.debug),
            extension_names: Some(self.extension_names),
        }
    }

    #[deprecated = "use into_builder function"]
    #[doc(hidden)]
    pub fn to_builder(self) -> InstanceInfoBuilder {
        self.into_builder()
    }
}

impl InstanceInfoBuilder {
    /// Builds a new `InstanceInfo`.
    #[inline(always)]
    pub fn build(self) -> InstanceInfo {
        self.fallible_build().expect("invalid instance info")
    }
}

impl From<InstanceInfoBuilder> for InstanceInfo {
    fn from(info: InstanceInfoBuilder) -> Self {
        info.build()
    }
}

struct InstanceInner {
    debug_utils: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    entry: ash::Entry,
    instance: ash::Instance,
    instance_created: bool,
}

impl Drop for InstanceInner {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        unsafe {
            if let Some((debug_utils, debug_messenger)) = self.debug_utils.take() {
                trace!("destroy debug_utils_messenger {}", debug_messenger.as_raw());
                debug_utils.destroy_debug_utils_messenger(debug_messenger, None);
                trace!(
                    "destroy debug_utils_messenger {} DONE",
                    debug_messenger.as_raw()
                );
            }

            if self.instance_created {
                trace!("destroy instance {}", self.instance.handle().as_raw());
                self.instance.destroy_instance(None);
                self.instance_created = false;
            }
        }
    }
}

/// Data returned when attempting to parse a Vulkan API version number.
#[derive(Clone, Copy, Debug)]
pub struct ParseApiVersionError {
    /// The _major_ version indicates a significant change in the API, which will encompass a
    /// wholly new version of the specification.
    pub major: u32,

    /// The _minor_ version indicates the incorporation of new functionality into the core
    /// specification.
    pub minor: u32,

    /// The _patch_ version indicates bug fixes, clarifications, and language improvements have
    /// been incorporated into the specification.
    pub patch: u32,

    /// The _variant_ indicates the variant of the Vulkan API supported by the implementation. This
    /// is always 0 for the Vulkan API.
    pub variant: u32,
}

impl Display for ParseApiVersionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "v{}.{}.{}-{}",
            self.major, self.minor, self.patch, self.variant
        ))
    }
}

impl Error for ParseApiVersionError {}

#[doc(hidden)]
impl Deref for ReadOnlyInstance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.inner.instance
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn api_versions_match() {
        assert_eq!(
            ApiVersion::Vulkan12.to_vk_api_version(),
            vk::API_VERSION_1_2
        );
        assert_eq!(
            ApiVersion::Vulkan13.to_vk_api_version(),
            vk::API_VERSION_1_3
        );
    }

    #[test]
    pub fn api_versions_from() {
        assert_eq!(
            ApiVersion::try_parse_vk_api_version(vk::API_VERSION_1_2).unwrap(),
            ApiVersion::Vulkan12
        );
        assert_eq!(
            ApiVersion::try_parse_vk_api_version(vk::API_VERSION_1_3).unwrap(),
            ApiVersion::Vulkan13
        );
    }
}
