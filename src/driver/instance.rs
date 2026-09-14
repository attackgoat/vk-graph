//! Vulkan initialization types.

use {
    super::{DriverError, physical_device::PhysicalDevice},
    ash::{
        ext, khr,
        vk::{self, Handle},
    },
    derive_builder::Builder,
    log::{debug, error, trace, warn},
    raw_window_handle::{HasDisplayHandle, RawDisplayHandle},
    std::{
        collections::HashSet,
        error::Error,
        ffi::CStr,
        fmt::{Debug, Display, Formatter},
        ops::Deref,
        sync::{Arc, Mutex},
        thread::panicking,
    },
};

#[cfg(any(not(target_os = "macos"), feature = "loaded"))]
use {
    log::{info, logger},
    std::{
        env::var,
        ffi::c_void,
        io::{IsTerminal, stderr},
        process::id,
        thread::{current, park},
    },
};

#[cfg(target_os = "macos")]
use std::env::set_var;

#[cfg(test)]
use crate::test_support::disposal::{DisposalReport, DisposalTracker};

#[cfg(any(not(target_os = "macos"), feature = "loaded"))]
const SKIP_VALIDATION_PARK_ENV: &str = "VK_GRAPH_SKIP_VALIDATION_PARK";

/// Vulkan API version.
///
/// See [`VkApplicationInfo::apiVersion`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkApplicationInfo.html).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum ApiVersion {
    /// Version `1.2`.
    Vulkan12,

    /// Version `1.3`.
    ///
    /// This is the default value.
    #[default]
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

    /// Vulkan API patch version number component. Ex: `v0.0.X-0`.
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

/// There is no global state in Vulkan and all per-application state is stored in a `VkInstance`
/// object.
///
/// Creating an `Instance` initializes the Vulkan library and allows the application to pass
/// information about itself to the implementation.
///
/// See [`VkInstance`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkInstance.html).
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
    pub khr_surface: bool,
}

impl Instance {
    /// Default Vulkan API version requested when creating an instance.
    pub const DEFAULT_API_VERSION: ApiVersion = ApiVersion::Vulkan13;

    /// Creates a new Vulkan instance.
    ///
    /// This constructor is intended for headless or manually managed setups. It does not infer or
    /// enable display platform surface extensions. Use [`Self::try_from_display`] when the
    /// resulting instance must be capable of later surface creation.
    ///
    /// See [`vkCreateInstance`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateInstance.html).
    #[profiling::function]
    pub fn create(info: impl Into<InstanceInfo>) -> Result<Self, DriverError> {
        Self::create_with_extension_names(info.into(), &[])
    }

    fn create_with_extension_names(
        info: InstanceInfo,
        extra_extension_names: &[&CStr],
    ) -> Result<Self, DriverError> {
        if info.debug && Self::debug_extension_names().is_empty() {
            error!("debug mode requires VK_EXT_debug_utils support");

            return Err(DriverError::Unsupported);
        }

        // Required to enable non-uniform descriptor indexing (bindless)
        #[cfg(target_os = "macos")]
        unsafe {
            set_var("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "1");
        }

        // Link the Vulkan loader dynamically (default feature)
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

            // On macOS, by default link molten-vk statically using ash-molten
            #[cfg(target_os = "macos")]
            let entry = ash_molten::load();
        };

        let mut extension_names = info
            .extension_names
            .iter()
            .chain(extra_extension_names)
            .copied()
            .collect::<HashSet<_>>();

        if info.debug {
            extension_names.extend(Self::debug_extension_names());
            #[cfg(test)]
            extension_names.insert(ext::layer_settings::NAME);
        }

        /*
        If linking dynamically on macOS, we require a few additional extensions. Based on
        "Encountered VK_ERROR_INCOMPATIBLE_DRIVER" section in:
        https://vulkan.lunarg.com/doc/view/latest/mac/getting_started.html
        */
        #[cfg(all(target_os = "macos", feature = "loaded"))]
        {
            extension_names.extend(&[
                ash::khr::get_physical_device_properties2::NAME,
                ash::khr::portability_enumeration::NAME,
            ]);
        }

        let khr_surface = extension_names.contains(&khr::surface::NAME);

        let extension_name_ptrs = extension_names
            .iter()
            .copied()
            .map(CStr::as_ptr)
            .collect::<Box<_>>();

        let mut layer_names = Vec::with_capacity(info.debug as _);

        if info.debug {
            layer_names.extend(Self::debug_layer_names());
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

        /*
        MoltenVK doesn't support the full Vulkan feature set, hence the portability flag needs to be
        set.
        */
        #[cfg(all(target_os = "macos", feature = "loaded"))]
        let instance_desc = instance_desc.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);

        // Creation itself can invoke the chained callback. Both callbacks borrow this stable Arc
        // allocation, which is moved into InstanceInner after successful instance creation.
        let validation_report = info.debug.then(ValidationReport::default);

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
            .pfn_user_callback(Some(Self::debug_callback))
            .user_data(
                validation_report
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |report| {
                        Arc::as_ptr(&report.0).cast_mut().cast()
                    }),
            );

        #[cfg(any(not(target_os = "macos"), feature = "loaded"))]
        let instance_desc = if info.debug {
            instance_desc.push_next(&mut debug_create_info)
        } else {
            instance_desc
        };

        // Test defaults are supplied per instance, without mutating the process environment.
        #[cfg(test)]
        let validation_settings = crate::test_support::validation::ValidationSettings::from_env();
        #[cfg(test)]
        let layer_settings = validation_settings.layer_settings();
        #[cfg(test)]
        let mut layer_settings_info =
            vk::LayerSettingsCreateInfoEXT::default().settings(&layer_settings);
        #[cfg(test)]
        let instance_desc = if info.debug {
            instance_desc.push_next(&mut layer_settings_info)
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

        // Establish ownership before subsequent setup or logging can fail or unwind.
        #[cfg_attr(all(target_os = "macos", not(feature = "loaded")), allow(unused_mut))]
        let mut inner = InstanceInner {
            #[cfg(test)]
            disposal: DisposalTracker::default(),
            debug_utils: None,
            entry,
            instance,
            instance_created: true,
            validation_report,
        };

        trace!("created a Vulkan instance");

        #[cfg(any(not(target_os = "macos"), feature = "loaded"))]
        if info.debug {
            let debug_utils = ext::debug_utils::Instance::new(&inner.entry, &inner.instance);
            let debug_messenger =
                unsafe { debug_utils.create_debug_utils_messenger(&debug_create_info, None) }
                    .map_err(|err| {
                        unsafe {
                            inner.instance.destroy_instance(None);
                        }
                        inner.instance_created = false;

                        error!("unable to create debug utils messenger: {err}");

                        DriverError::Unsupported
                    })?;

            inner.debug_utils = Some((debug_utils, debug_messenger));
        }

        Ok(Self {
            read_only: ReadOnlyInstance {
                info,
                inner: Arc::new(inner),
                khr_surface,
            },
        })
    }

    /// Creates a new Vulkan instance with the platform surface extensions required by the provided
    /// display handle.
    ///
    /// See [`VK_KHR_surface`](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_surface.html).
    #[profiling::function]
    pub fn try_from_display(
        display: impl HasDisplayHandle,
        info: impl Into<InstanceInfo>,
    ) -> Result<Self, DriverError> {
        let display_handle = display.display_handle().map_err(|err| {
            warn!("unable to get display handle: {err}");

            DriverError::Unsupported
        })?;
        let display_extension_names = Self::display_extension_names(display_handle.as_raw())
            .map_err(|err| {
                warn!("unable to enumerate display extensions: {err}");

                DriverError::Unsupported
            })?;

        Self::create_with_extension_names(info.into(), display_extension_names)
    }

    /// Loads an existing Vulkan instance that may have been created by other means.
    ///
    /// This is useful when you want to use a Vulkan instance created by some other library, such
    /// as OpenXR.
    ///
    /// See [`VkInstance`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkInstance.html).
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
                /*
                The implementation *should* provide a version. If it does not, use the default.
                */
                Self::DEFAULT_API_VERSION.to_vk_api_version()
            })
            .try_into()
            .map_err(|err| {
                warn!("unsupported instance: {err}");

                DriverError::Unsupported
            })?;
        let khr_surface = Self::has_vk_khr_surface(&entry, instance);

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
                    validation_report: None,
                    #[cfg(test)]
                    disposal: DisposalTracker::default(),
                }),
                khr_surface,
            },
        })
    }

    #[cfg(any(not(target_os = "macos"), feature = "loaded"))]
    unsafe extern "system" fn debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
        callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        user_data: *mut c_void,
    ) -> vk::Bool32 {
        let Some(callback_data) = (unsafe { callback_data.as_ref() }) else {
            return vk::FALSE;
        };
        let is_error = message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR);

        if is_error && !user_data.is_null() {
            let error = ValidationError {
                message_id_name: if callback_data.p_message_id_name.is_null() {
                    None
                } else {
                    Some(
                        unsafe { CStr::from_ptr(callback_data.p_message_id_name) }
                            .to_string_lossy()
                            .into_owned(),
                    )
                },
                message: if callback_data.p_message.is_null() {
                    "<missing Vulkan validation message>".to_owned()
                } else {
                    unsafe { CStr::from_ptr(callback_data.p_message) }
                        .to_string_lossy()
                        .into_owned()
                },
            };
            // Vulkan borrows this pointee until both messenger and instance destruction finish.
            // Recover poisoned locks rather than panicking across the FFI boundary.
            let errors = unsafe { &*user_data.cast::<Mutex<Vec<ValidationError>>>() };
            errors
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(error);
        }

        // Keep recording during unwind, but suppress logging and debugger parking.
        if panicking() {
            return vk::FALSE;
        }

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

        if is_error {
            error!("{message}");
        } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
            && !var("VK_GRAPH_DEBUG_IGNORE_WARNING")
                .map(Self::var_value_is_set)
                .unwrap_or_default()
        {
            warn!("{message}");
        } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO)
            && !var("VK_GRAPH_DEBUG_IGNORE_INFO")
                .map(Self::var_value_is_set)
                .unwrap_or_default()
        {
            info!("{message}");
        } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE)
            && !var("VK_GRAPH_DEBUG_IGNORE_VERBOSE")
                .map(Self::var_value_is_set)
                .unwrap_or_default()
        {
            debug!("{message}");
        }

        if !is_error {
            return vk::FALSE;
        }

        if current().name() != Some("main") {
            warn!("invalid validation callback thread: child thread")
        }

        if var(SKIP_VALIDATION_PARK_ENV)
            .map(Self::var_value_is_set)
            .unwrap_or_default()
        {
            warn!("validation callback park skipped; execution will continue");

            return vk::FALSE;
        }

        if !stderr().is_terminal() {
            warn!("validation callback park skipped; stderr is not an interactive terminal");

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

    fn debug_extension_names() -> &'static [&'static CStr] {
        #[cfg(any(not(target_os = "macos"), feature = "loaded"))]
        return &[ext::debug_utils::NAME];

        #[cfg(all(target_os = "macos", not(feature = "loaded")))]
        return &[];
    }

    fn debug_layer_names() -> &'static [&'static CStr] {
        #[cfg(any(not(target_os = "macos"), feature = "loaded"))]
        return &[c"VK_LAYER_KHRONOS_validation"];

        #[cfg(all(target_os = "macos", not(feature = "loaded")))]
        return &[];
    }

    // Copied from ash_window::enumerate_required_extensions to change the signature
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

    /// The ash entry point used to load Vulkan instance functions.
    pub fn entry(this: &Self) -> &ash::Entry {
        &this.inner.entry
    }

    /*
    Estimates surface extension support.

    Imported instances do not expose their enabled extension list, so we infer support by checking that
    the VK_KHR_surface entry points resolve for this instance handle.
    */
    fn has_vk_khr_surface(entry: &ash::Entry, instance: vk::Instance) -> bool {
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

    /// Returns the available physical devices of this instance.
    ///
    /// See [`vkEnumeratePhysicalDevices`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkEnumeratePhysicalDevices.html).
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
                let physical_device = unsafe {
                    PhysicalDevice::try_from_ash(this, physical_device)
                        .inspect_err(|err| warn!("unsupported physical device #{idx}: {err}"))
                        .ok()?
                };

                let api_version = ApiVersion::try_parse_vk_api_version(
                    physical_device.properties_v1_0.api_version,
                )
                .inspect_err(|err| {
                    warn!(
                        "unsupported physical device #{idx} {}: {err}",
                        physical_device.properties_v1_0.device_name
                    );
                })
                .ok()?;

                if api_version < this.info.api_version {
                    return None;
                }

                if this.info.debug && !physical_device.vk_ext_private_data {
                    warn!(
                        "unsupported physical device #{idx} {}: missing VK_EXT_private_data",
                        physical_device.properties_v1_0.device_name
                    );

                    return None;
                }

                Some(physical_device)
            }))
    }

    pub(crate) fn supports_debug_utils(this: &Self) -> bool {
        this.inner.debug_utils.is_some()
    }

    /// Returns the shared validation report for an owned, debug-enabled instance.
    ///
    /// Non-debug and imported instances return `None`. Retain a clone to inspect errors after
    /// destruction; taking a snapshot does not clear the report.
    pub fn validation_report(this: &Self) -> Option<ValidationReport> {
        this.inner.validation_report.clone()
    }

    /// Observes completed instance destruction without retaining the instance.
    ///
    /// Imported instances return `None` because their destruction is managed externally.
    /// Available independently of validation layers and the `checked` feature.
    #[cfg(test)]
    pub(crate) fn disposal_report(this: &Self) -> Option<DisposalReport> {
        this.inner
            .instance_created
            .then(|| this.inner.disposal.report())
    }

    fn var_value_is_set(val: String) -> bool {
        !matches!(val.as_str(), "" | "0" | "false" | "False" | "FALSE")
    }
}

impl Clone for Instance {
    fn clone(&self) -> Self {
        Self {
            read_only: ReadOnlyInstance {
                info: self.info,
                inner: self.inner.clone(),
                khr_surface: self.khr_surface,
            },
        }
    }
}

impl Debug for Instance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(stringify!(Instance))
            .field("handle", &self.inner.instance.handle())
            .field("info", &self.info)
            .field("khr_surface", &self.khr_surface)
            .field("debug_utils", &self.inner.debug_utils.is_some())
            .finish_non_exhaustive()
    }
}

/// Information used to create an [`Instance`] instance.
#[derive(Builder, Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct InstanceInfo {
    /// The Vulkan API version to target.
    ///
    /// Defaults to [`Instance::DEFAULT_API_VERSION`].
    #[builder(default = "Instance::DEFAULT_API_VERSION")]
    pub api_version: ApiVersion,

    /// Enables Vulkan validation layers.
    ///
    /// This requires a Vulkan SDK installation. Additionally, the device must support
    /// VK_EXT_private_data. Errors are recorded in [`ValidationReport`] independently of logging.
    ///
    /// When `stderr` is attached to an interactive terminal, validation errors will park the
    /// callback thread for debugger attach.
    ///
    /// Set `VK_GRAPH_SKIP_VALIDATION_PARK=1` to keep logging validation errors without parking.
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
}

impl From<InstanceInfoBuilder> for InstanceInfo {
    fn from(info: InstanceInfoBuilder) -> Self {
        info.build()
    }
}

impl InstanceInfoBuilder {
    /// Builds a new `InstanceInfo`.
    #[inline(always)]
    pub fn build(self) -> InstanceInfo {
        self.fallible_build().expect("invalid instance info")
    }
}

struct InstanceInner {
    #[cfg(test)]
    disposal: DisposalTracker,
    debug_utils: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    entry: ash::Entry,
    instance: ash::Instance,
    instance_created: bool,
    validation_report: Option<ValidationReport>,
}

impl Drop for InstanceInner {
    #[profiling::function]
    fn drop(&mut self) {
        // If teardown is skipped or unwinds, Vulkan still holds pUserData. Retain its owner along
        // with the deliberately leaked Vulkan object instead of leaving a dangling pointer.
        let validation_report = std::mem::ManuallyDrop::new(self.validation_report.take());
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
                #[cfg(test)]
                self.disposal.complete();
            }
        }
        drop(std::mem::ManuallyDrop::into_inner(validation_report));
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

/// An error reported by an instance's Vulkan debug callback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationError {
    /// The validation message ID, usually a VUID, when supplied by Vulkan.
    pub message_id_name: Option<String>,

    /// The complete diagnostic message.
    pub message: String,
}

/// Instance-scoped validation errors, retained independently of the instance's lifetime.
#[derive(Clone, Debug, Default)]
pub struct ValidationReport(Arc<Mutex<Vec<ValidationError>>>);

impl ValidationReport {
    /// Returns all errors recorded so far, including instance creation and teardown errors.
    ///
    /// Wait for pending Vulkan work before inspecting the report. Retain a clone to inspect
    /// destruction errors after dropping every device, resource, and instance handle.
    pub fn errors(&self) -> Vec<ValidationError> {
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[cfg(any(not(target_os = "macos"), feature = "loaded"))]
    mod validation {
        use {
            super::*,
            std::{
                cell::Cell,
                ffi::CString,
                panic::{UnwindSafe, catch_unwind, resume_unwind},
                sync::{Barrier, Weak},
                thread,
            },
        };

        // Exercise the real FFI callback while logging/parking are suppressed. resume_unwind
        // neither calls nor replaces the process-wide panic hook.
        fn during_unwind(f: impl FnOnce() + UnwindSafe) {
            struct OnDrop<F: FnOnce()>(Option<F>);

            impl<F: FnOnce()> Drop for OnDrop<F> {
                fn drop(&mut self) {
                    if let Some(f) = self.0.take() {
                        f();
                    }
                }
            }

            assert!(
                catch_unwind(move || {
                    let _on_drop = OnDrop(Some(f));
                    resume_unwind(Box::new(()));
                })
                .is_err()
            );
        }

        impl ValidationReport {
            fn invoke_callback(
                &self,
                severity: vk::DebugUtilsMessageSeverityFlagsEXT,
                message_id_name: Option<&CStr>,
                message: &CStr,
            ) -> vk::Bool32 {
                let mut data = vk::DebugUtilsMessengerCallbackDataEXT::default().message(message);
                if let Some(name) = message_id_name {
                    data = data.message_id_name(name);
                }
                unsafe {
                    Instance::debug_callback(
                        severity,
                        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                        &data,
                        Arc::as_ptr(&self.0).cast_mut().cast(),
                    )
                }
            }
        }

        #[test]
        fn errors_are_recorded_without_a_logger() {
            const CHILD: &str = "VK_GRAPH_TEST_NO_LOGGER_CHILD";
            if std::env::var_os(CHILD).is_some() {
                assert_eq!(log::max_level(), log::LevelFilter::Off);
                let report = ValidationReport::default();
                assert_eq!(
                    report.invoke_callback(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                        Some(c"VUID-no-logger"),
                        c"recorded without diagnostic output",
                    ),
                    vk::FALSE
                );
                assert_eq!(report.errors().len(), 1);
                assert_eq!(
                    report.errors()[0].message_id_name.as_deref(),
                    Some("VUID-no-logger")
                );
                return;
            }

            // A fresh process has no logger, even if other tests install one. Configure the
            // child's environment without changing it underneath concurrent parent tests.
            for rust_log in [None, Some(""), Some("off"), Some("warn")] {
                let mut command = std::process::Command::new(std::env::current_exe().unwrap());
                command
                    .args([
                        "--exact",
                        "driver::instance::test::validation::errors_are_recorded_without_a_logger",
                    ])
                    .env(CHILD, "1")
                    .env("VK_GRAPH_SKIP_VALIDATION_PARK", "1")
                    .env_remove("RUST_LOG");
                if let Some(rust_log) = rust_log {
                    command.env("RUST_LOG", rust_log);
                }
                let output = command.output().unwrap();
                assert!(
                    output.status.success(),
                    "callback failed with RUST_LOG={rust_log:?}: {}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr),
                );
            }
        }

        #[test]
        fn reports_isolate_and_copy_errors_with_optional_ids() {
            let first = ValidationReport::default();
            let second = ValidationReport::default();
            let observer = first.clone();
            let empty_snapshot = observer.errors();

            {
                let name = CString::new("VUID-owned-name").unwrap();
                let message = CString::new("owned diagnostic").unwrap();
                during_unwind(|| {
                    first.invoke_callback(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                        Some(&name),
                        &message,
                    );
                    second.invoke_callback(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                        None,
                        c"second instance",
                    );
                    for severity in [
                        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                        vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
                    ] {
                        first.invoke_callback(severity, Some(&name), c"not an error");
                    }
                });
            }
            drop(first);

            let expected = vec![ValidationError {
                message_id_name: Some("VUID-owned-name".to_owned()),
                message: "owned diagnostic".to_owned(),
            }];
            assert_eq!(observer.errors(), expected);
            assert_eq!(observer.errors(), expected);
            assert!(empty_snapshot.is_empty());
            let mut snapshot = observer.errors();
            snapshot[0].message.clear();
            assert_eq!(observer.errors(), expected);
            assert_eq!(
                second.errors(),
                vec![ValidationError {
                    message_id_name: None,
                    message: "second instance".to_owned(),
                }]
            );
        }

        #[test]
        fn concurrent_callbacks_record_every_error() {
            const THREADS: usize = 8;
            const ERRORS: usize = 64;
            let first = ValidationReport::default();
            let second = ValidationReport::default();
            let barrier = Barrier::new(THREADS);
            thread::scope(|scope| {
                for index in 0..THREADS {
                    let report = if index % 2 == 0 { &first } else { &second };
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        during_unwind(|| {
                            for error in 0..ERRORS {
                                let message = CString::new(format!("{index}:{error}")).unwrap();
                                report.invoke_callback(
                                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                                    None,
                                    &message,
                                );
                                // Snapshots contend with writers on arbitrary callback threads.
                                let _snapshot = report.errors();
                            }
                        });
                    });
                }
            });

            for (parity, report) in [first, second].into_iter().enumerate() {
                let errors = report.errors();
                assert_eq!(errors.len(), THREADS / 2 * ERRORS);
                let messages = errors
                    .into_iter()
                    .map(|error| error.message)
                    .collect::<HashSet<_>>();
                let expected = (parity..THREADS)
                    .step_by(2)
                    .flat_map(|thread| (0..ERRORS).map(move |error| format!("{thread}:{error}")))
                    .collect::<HashSet<_>>();
                assert_eq!(messages, expected);
            }
        }

        #[test]
        fn poisoned_report_records_during_unwind() {
            let report = ValidationReport::default();
            assert!(
                catch_unwind(|| {
                    let _guard = report.0.lock().unwrap();
                    resume_unwind(Box::new(()));
                })
                .is_err()
            );
            assert!(report.0.is_poisoned());

            let result = Cell::new(vk::TRUE);
            // Cell is only accessed on this thread; AssertUnwindSafe is needed for the test result.
            let result_ref = std::panic::AssertUnwindSafe(&result);
            during_unwind(|| {
                result_ref.set(report.invoke_callback(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                    None,
                    c"unwinding with a poisoned report",
                ));
            });
            assert_eq!(result.get(), vk::FALSE);
            assert_eq!(report.errors().len(), 1);
            assert_eq!(
                report.errors()[0].message,
                "unwinding with a poisoned report"
            );
        }

        #[test]
        fn callback_handles_null_and_non_utf8_data() {
            let report = ValidationReport::default();
            assert_eq!(
                unsafe {
                    Instance::debug_callback(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                        std::ptr::null(),
                        Arc::as_ptr(&report.0).cast_mut().cast(),
                    )
                },
                vk::FALSE
            );
            during_unwind(|| unsafe {
                let data = vk::DebugUtilsMessengerCallbackDataEXT::default();
                Instance::debug_callback(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                    &data,
                    Arc::as_ptr(&report.0).cast_mut().cast(),
                );
                Instance::debug_callback(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                    &data,
                    std::ptr::null_mut(),
                );
                report.invoke_callback(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                    Some(c"id\xff"),
                    c"message\xff",
                );
            });
            assert_eq!(
                report.errors(),
                vec![
                    ValidationError {
                        message_id_name: None,
                        message: "<missing Vulkan validation message>".to_owned(),
                    },
                    ValidationError {
                        message_id_name: Some("id\u{fffd}".to_owned()),
                        message: "message\u{fffd}".to_owned(),
                    },
                ]
            );
        }

        // No loader or driver is involved in these lifetime tests. Only destruction entry points
        // are implemented; the mock instance handle carries a weak reference to callback storage.
        struct MockInstance {
            report: Weak<Mutex<Vec<ValidationError>>>,
            destructions: Mutex<Vec<(&'static str, usize)>>,
        }

        impl MockInstance {
            fn destroyed(&self, what: &'static str) {
                self.destructions
                    .lock()
                    .unwrap()
                    .push((what, self.report.strong_count()));
                if let Some(errors) = self.report.upgrade() {
                    let report = ValidationReport(errors);
                    during_unwind(|| {
                        report.invoke_callback(
                            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                            None,
                            c"destruction diagnostic",
                        );
                    });
                }
            }

            unsafe extern "system" fn destroy_instance(
                instance: vk::Instance,
                _allocator: *const vk::AllocationCallbacks<'_>,
            ) {
                unsafe { &*(instance.as_raw() as *const MockInstance) }.destroyed("instance");
            }

            unsafe extern "system" fn destroy_messenger(
                instance: vk::Instance,
                _messenger: vk::DebugUtilsMessengerEXT,
                _allocator: *const vk::AllocationCallbacks<'_>,
            ) {
                unsafe { &*(instance.as_raw() as *const MockInstance) }.destroyed("messenger");
            }

            unsafe extern "system" fn get_instance_proc_addr(
                _instance: vk::Instance,
                name: *const std::ffi::c_char,
            ) -> vk::PFN_vkVoidFunction {
                match unsafe { CStr::from_ptr(name) }.to_bytes() {
                    b"vkDestroyInstance" => Some(unsafe {
                        std::mem::transmute::<vk::PFN_vkDestroyInstance, unsafe extern "system" fn()>(
                            Self::destroy_instance,
                        )
                    }),
                    b"vkDestroyDebugUtilsMessengerEXT" => Some(unsafe {
                        std::mem::transmute::<
                            vk::PFN_vkDestroyDebugUtilsMessengerEXT,
                            unsafe extern "system" fn(),
                        >(Self::destroy_messenger)
                    }),
                    _ => None,
                }
            }

            fn entry() -> ash::Entry {
                unsafe {
                    ash::Entry::from_static_fn(ash::StaticFn {
                        get_instance_proc_addr: Self::get_instance_proc_addr,
                    })
                }
            }

            fn instance(&self, report: Option<ValidationReport>) -> Instance {
                let entry = Self::entry();
                let instance = unsafe {
                    ash::Instance::load(
                        entry.static_fn(),
                        vk::Instance::from_raw(self as *const MockInstance as u64),
                    )
                };
                let debug = report.is_some();
                let debug_utils = debug.then(|| {
                    (
                        ext::debug_utils::Instance::new(&entry, &instance),
                        vk::DebugUtilsMessengerEXT::from_raw(1),
                    )
                });
                Instance {
                    read_only: ReadOnlyInstance {
                        info: InstanceInfo {
                            debug,
                            ..Default::default()
                        },
                        inner: Arc::new(InstanceInner {
                            disposal: DisposalTracker::default(),
                            debug_utils,
                            entry,
                            instance,
                            instance_created: true,
                            validation_report: report,
                        }),
                        khr_surface: false,
                    },
                }
            }
        }

        #[test]
        fn report_lives_through_messenger_and_instance_destruction() {
            let report = ValidationReport::default();
            let state = MockInstance {
                report: Arc::downgrade(&report.0),
                destructions: Mutex::new(Vec::new()),
            };
            let instance = state.instance(Some(report));
            let observer = Instance::validation_report(&instance).unwrap();
            let disposal = Instance::disposal_report(&instance).unwrap();
            let clone = instance.clone();
            drop(instance);
            assert!(state.destructions.lock().unwrap().is_empty());
            assert!(!disposal.is_disposed());
            drop(clone);
            assert!(disposal.is_disposed());
            assert_eq!(
                *state.destructions.lock().unwrap(),
                [("messenger", 2), ("instance", 2)]
            );
            assert_eq!(observer.errors().len(), 2);
            assert_eq!(state.report.strong_count(), 1);
            drop(observer);
            assert!(state.report.upgrade().is_none());
        }

        #[test]
        fn skipped_teardown_retains_callback_storage() {
            let report = ValidationReport::default();
            let pointer = Arc::as_ptr(&report.0);
            let state = MockInstance {
                report: Arc::downgrade(&report.0),
                destructions: Mutex::new(Vec::new()),
            };
            let instance = state.instance(Some(report));
            let disposal = Instance::disposal_report(&instance).unwrap();
            assert!(
                catch_unwind(move || {
                    let _instance = instance;
                    resume_unwind(Box::new(()));
                })
                .is_err()
            );
            assert!(state.destructions.lock().unwrap().is_empty());
            assert_eq!(state.report.strong_count(), 1);
            assert!(
                !disposal.is_disposed(),
                "skipped destruction must remain incomplete"
            );
            // The only strong reference was deliberately leaked by InstanceInner. Reclaim it in
            // this mock-only test, since no real Vulkan instance can invoke the callback later.
            let retained = ValidationReport(unsafe { Arc::from_raw(pointer) });
            during_unwind(|| {
                retained.invoke_callback(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                    None,
                    c"after skipped teardown",
                );
            });
            assert_eq!(retained.errors()[0].message, "after skipped teardown");
            drop(retained);
            assert!(state.report.upgrade().is_none());
        }

        #[test]
        fn nondebug_and_imported_instances_have_no_report() {
            let state = MockInstance {
                report: Weak::new(),
                destructions: Mutex::new(Vec::new()),
            };
            let instance = state.instance(None);
            assert!(Instance::validation_report(&instance).is_none());
            let disposal = Instance::disposal_report(&instance).unwrap();
            let imported =
                Instance::try_from_entry(MockInstance::entry(), instance.handle()).unwrap();
            assert!(Instance::validation_report(&imported).is_none());
            assert!(Instance::disposal_report(&imported).is_none());
            drop(imported);
            assert!(state.destructions.lock().unwrap().is_empty());
            drop(instance);
            assert_eq!(*state.destructions.lock().unwrap(), [("instance", 0)]);
            assert!(disposal.is_disposed());
        }
    }

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

    #[test]
    pub fn default_api_version_matches_instance_info_default() {
        assert_eq!(
            Instance::DEFAULT_API_VERSION,
            InstanceInfo::default().api_version
        );
    }

    #[test]
    pub fn default_api_version_matches_api_version_default() {
        assert_eq!(Instance::DEFAULT_API_VERSION, ApiVersion::default());
    }

    #[test]
    pub fn invalid_api_versions_are_rejected() {
        assert!(ApiVersion::try_parse_vk_api_version(vk::API_VERSION_1_1).is_err());
        assert!(ApiVersion::try_parse_vk_api_version(vk::make_api_version(0, 2, 0, 0)).is_err());
        assert!(ApiVersion::try_parse_vk_api_version(vk::make_api_version(1, 1, 9, 0)).is_err());
        assert!(ApiVersion::try_parse_vk_api_version(vk::make_api_version(1, 4, 2, 0)).is_err());
        assert!(ApiVersion::try_parse_vk_api_version(vk::make_api_version(1, 2, 0, 1)).is_err());
        assert!(ApiVersion::try_parse_vk_api_version(vk::make_api_version(1, 3, 0, 1)).is_err());
    }
}
