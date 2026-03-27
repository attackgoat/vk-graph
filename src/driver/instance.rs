//! Vulkan initialization types

use {
    super::{DriverError, physical_device::PhysicalDevice},
    ash::{ext, vk},
    derive_builder::{Builder, UninitializedFieldError},
    log::{debug, error, trace, warn},
    std::{
        error::Error,
        ffi::CStr,
        fmt::{Debug, Display, Formatter},
        ops::Deref,
        os::raw::c_char,
        sync::Arc,
        thread::panicking,
    },
};

#[cfg(feature = "loaded")]
use {
    log::{Level, Metadata, info, logger},
    std::{
        env::var,
        ffi::c_void,
        process::{abort, id},
        thread::{current, park},
    },
};

#[cfg(target_os = "macos")]
use std::env::set_var;

#[cfg(feature = "loaded")]
unsafe extern "system" fn vulkan_debug_callback(
    _flags: vk::DebugReportFlagsEXT,
    _obj_type: vk::DebugReportObjectTypeEXT,
    _src_obj: u64,
    _location: usize,
    _msg_code: i32,
    _layer_prefix: *const c_char,
    message: *const c_char,
    _user_data: *mut c_void,
) -> u32 {
    if panicking() {
        return vk::FALSE;
    }

    assert!(!message.is_null());

    let mut found_null = false;
    for i in 0..u16::MAX as _ {
        if unsafe { *message.add(i) } == 0 {
            found_null = true;
            break;
        }
    }

    assert!(found_null);

    let message = unsafe { CStr::from_ptr(message) }.to_str().unwrap();

    if message.starts_with("Validation Warning: [ UNASSIGNED-BestPractices-pipeline-stage-flags ]")
    {
        // vk_sync uses vk::PipelineStageFlags::ALL_COMMANDS with AccessType::NOTHING and others
        warn!("{}", message);
    } else {
        let prefix = "Validation Error: [ ";

        let (vuid, message) = if message.starts_with(prefix) {
            let (vuid, message) = message
                .trim_start_matches(prefix)
                .split_once(" ]")
                .unwrap_or_default();
            let message = message.split(" | ").nth(2).unwrap_or(message);

            (Some(vuid.trim()), message)
        } else {
            (None, message)
        };

        if let Some(vuid) = vuid {
            info!("{vuid}");
        }

        error!("🆘 {message}");

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
            warn!("executing on a child thread!")
        }

        debug!(
            "🛑 PARKING THREAD `{}` -> attach debugger to pid {}!",
            current().name().unwrap_or_default(),
            id()
        );

        logger().flush();

        park();
    }

    vk::FALSE
}

/// Vulkan API version.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum ApiVersion {
    /// Version 1.2
    #[default]
    Vulkan12,

    /// Version 1.3
    Vulkan13,
}

impl ApiVersion {
    /// Returns a version parsed from a native Vulkan value.
    pub fn try_parse_vk_api_version(version: u32) -> Result<Self, ParseApiVersionError> {
        Self::try_from(version)
    }

    /// Vulkan API major version number component. Ex: vX.0.0-0
    ///
    /// Always one.
    pub fn major(self) -> u32 {
        1
    }

    /// Vulkan API minor version number component. Ex: v0.X.0-0
    pub fn minor(self) -> u32 {
        match self {
            Self::Vulkan12 => 2,
            Self::Vulkan13 => 3,
        }
    }

    /// Vulkan API minor version number component. Ex: v0.0.X-0
    ///
    /// Always zero.
    pub fn patch(self) -> u32 {
        0
    }

    /// Returns a native Vulkan value.
    pub fn to_vk_api_version(self) -> u32 {
        self.into()
    }

    /// Vulkan API variant version number component. Ex: v0.0.0-X
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
#[derive(Clone)]
#[repr(C)]
pub struct Instance {
    /// The ash entrypoint.
    ///
    /// _Note:_ This field is read-only.
    #[cfg(doc)]
    pub entry: ash::Entry,

    #[cfg(not(doc))]
    entry: ash::Entry,

    /// Information used to create this resource.
    ///
    /// _Note:_ This field is read-only.
    #[cfg(doc)]
    pub info: InstanceInfo,

    #[cfg(not(doc))]
    info: InstanceInfo,

    inner: Arc<InstanceInner>,
}

impl Instance {
    /// The most recent supported version of Vulkan
    pub const LATEST_API_VERSION: ApiVersion = ApiVersion::Vulkan13;

    /// Creates a new Vulkan instance.
    #[profiling::function]
    pub fn new(info: impl Into<InstanceInfo>) -> Result<Self, DriverError> {
        let info = info.into();

        // Required to enable non-uniform descriptor indexing (bindless)
        #[cfg(target_os = "macos")]
        unsafe {
            set_var("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "1");
        }

        // Link molten-vk dynamically if not on MacOS, or if explicitly requested.
        #[cfg(feature = "loaded")]
        let entry = unsafe {
            ash::Entry::load().map_err(|err| {
                error!("Vulkan driver not found: {err}");

                DriverError::Unsupported
            })?
        };

        // On MacOS, by default link molten-vk statically using ash-molten.
        #[cfg(all(target_os = "macos", not(feature = "loaded")))]
        let entry = ash_molten::load();

        let mut extension_names = Vec::with_capacity(16);
        extension_names.extend(info.extension_names);

        if info.debug {
            extension_names.extend(Self::debug_extension_names());
        }

        // If linking dynamically on MacOS, we require a few additional extensions.
        // Based on "Encountered VK_ERROR_INCOMPATIBLE_DRIVER" section in:
        // https://vulkan.lunarg.com/doc/view/latest/mac/getting_started.html
        #[cfg(all(target_os = "macos", feature = "loaded"))]
        {
            extension_names.push(ash::khr::get_physical_device_properties2::NAME);
            extension_names.push(ash::khr::portability_enumeration::NAME);
        }

        let extension_name_ptrs = extension_names
            .iter()
            .copied()
            .map(CStr::as_ptr)
            .collect::<Box<[_]>>();

        let mut layer_names = Vec::with_capacity(1);

        if info.debug {
            layer_names.extend(Self::debug_layer_names());
        }

        let layer_name_ptrs = layer_names
            .iter()
            .copied()
            .map(CStr::as_ptr)
            .collect::<Box<[_]>>();

        let app_desc = vk::ApplicationInfo::default().api_version(match info.api_version {
            ApiVersion::Vulkan12 => vk::API_VERSION_1_2,
            ApiVersion::Vulkan13 => vk::API_VERSION_1_3,
        });
        let instance_desc = vk::InstanceCreateInfo::default()
            .application_info(&app_desc)
            .enabled_layer_names(&layer_name_ptrs)
            .enabled_extension_names(&extension_name_ptrs);

        // Molten-vk doesn't support the full Vulkan feature set, hence the portability flag needs
        // to be set.
        #[cfg(all(target_os = "macos", feature = "loaded"))]
        let instance_desc = instance_desc.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);

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

                for extension_name in &extension_names {
                    debug!("Extension: {:?}", extension_name);
                }

                DriverError::Unsupported
            })?
        };

        trace!("created a Vulkan instance");

        #[cfg(all(target_os = "macos", not(feature = "loaded")))]
        let (debug_loader, debug_callback, debug_utils) = (None, None, None);

        #[cfg(any(not(target_os = "macos"), feature = "loaded"))]
        let (debug_loader, debug_callback, debug_utils) = if info.debug {
            let debug_info = vk::DebugReportCallbackCreateInfoEXT {
                flags: vk::DebugReportFlagsEXT::ERROR
                    | vk::DebugReportFlagsEXT::WARNING
                    | vk::DebugReportFlagsEXT::PERFORMANCE_WARNING,
                pfn_callback: Some(vulkan_debug_callback),
                ..Default::default()
            };

            #[allow(deprecated)]
            let debug_loader = ext::debug_report::Instance::new(&entry, &instance);

            let debug_callback = unsafe {
                #[allow(deprecated)]
                debug_loader
                    .create_debug_report_callback(&debug_info, None)
                    .unwrap()
            };

            let debug_utils = ext::debug_utils::Instance::new(&entry, &instance);

            (Some(debug_loader), Some(debug_callback), Some(debug_utils))
        } else {
            (None, None, None)
        };

        Ok(Self {
            entry,
            info,
            inner: Arc::new(InstanceInner {
                debug_callback,
                _debug_loader: debug_loader,
                debug_utils,
                instance,
                instance_created: true,
            }),
        })
    }

    /// Loads an existing Vulkan instance that may have been created by other means.
    ///
    /// This is useful when you want to use a Vulkan instance created by some other library, such
    /// as OpenXR.
    #[profiling::function]
    pub fn from_entry(entry: ash::Entry, instance: vk::Instance) -> Result<Self, DriverError> {
        if instance == vk::Instance::null() {
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

        let instance = unsafe { ash::Instance::load(entry.static_fn(), instance) };

        Ok(Self {
            entry,
            info: InstanceInfo {
                api_version,
                ..Default::default()
            },
            inner: Arc::new(InstanceInner {
                debug_callback: None,
                _debug_loader: None,
                debug_utils: None,
                instance,
                instance_created: false,
            }),
        })
    }

    #[cfg(any(not(target_os = "macos"), feature = "loaded"))]
    fn debug_extension_names() -> impl IntoIterator<Item = &'static CStr> {
        vec![
            #[allow(deprecated)]
            ext::debug_report::NAME,
            ext::debug_utils::NAME,
        ]
    }

    #[cfg(all(target_os = "macos", not(feature = "loaded")))]
    fn debug_extension_names() -> impl IntoIterator<Item = &'static CStr> {
        vec![]
    }

    #[cfg(any(not(target_os = "macos"), feature = "loaded"))]
    fn debug_layer_names() -> impl IntoIterator<Item = &'static CStr> {
        vec![c"VK_LAYER_KHRONOS_validation"]
    }

    #[cfg(all(target_os = "macos", not(feature = "loaded")))]
    fn debug_layer_names() -> Vec<&'static CStr> {
        vec![]
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
                _ => DriverError::Unsupported,
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
}

impl Debug for Instance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Instance")
    }
}

#[doc(hidden)]
impl Deref for Instance {
    type Target = ReadOnlyInstance;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self as *const Self::Target) }
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
    /// The Vulkan API version to target
    #[builder(default = "ApiVersion::Vulkan13")]
    pub api_version: ApiVersion,

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

    /// Required Vulkan instance extension names to load
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
        self.fallible_build().unwrap()
    }
}

impl From<InstanceInfoBuilder> for InstanceInfo {
    fn from(info: InstanceInfoBuilder) -> Self {
        info.build()
    }
}

struct InstanceInner {
    debug_callback: Option<vk::DebugReportCallbackEXT>,

    #[allow(deprecated)] // TODO: Remove? Look into this....
    _debug_loader: Option<ext::debug_report::Instance>,

    #[allow(dead_code)]
    debug_utils: Option<ext::debug_utils::Instance>,

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
            #[allow(deprecated)]
            if let Some(debug_loader) = &self._debug_loader {
                let debug_callback = self.debug_callback.unwrap();
                debug_loader.destroy_debug_report_callback(debug_callback, None);
            }

            if self.instance_created {
                self.instance.destroy_instance(None);
            }
        }
    }
}

/// Data returned when attempting to parse a Vulkan API version number.
#[derive(Clone, Copy, Debug)]
pub struct ParseApiVersionError {
    /// The _major_ version indicates a significant change in the API, which will encompass a wholly
    /// new version of the specification.
    pub major: u32,

    /// The _minor_ version indicates the incorporation of new functionality into the core
    /// specification.
    pub minor: u32,

    /// The _patch_ version indicates bug fixes, clarifications, and language improvements have been
    /// incorporated into the specification.
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
#[repr(C)]
pub struct ReadOnlyInstance {
    pub entry: ash::Entry,
    pub info: InstanceInfo,
    inner: Arc<InstanceInner>,
}

#[doc(hidden)]
impl Deref for ReadOnlyInstance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.inner.instance
    }
}

#[cfg(test)]
mod test {
    use {
        super::*,
        std::mem::{offset_of, size_of},
    };

    #[test]
    pub fn instance_repr_c() {
        // HACK: The readonly crate uses a private implementation and so we can't further deref it
        // into the native object type. Because of this the ReadOnly part is manually implemented.
        assert_eq!(size_of::<Instance>(), size_of::<ReadOnlyInstance>());
        assert_eq!(
            offset_of!(Instance, entry),
            offset_of!(ReadOnlyInstance, entry),
        );
        assert_eq!(
            offset_of!(Instance, info),
            offset_of!(ReadOnlyInstance, info),
        );
        assert_eq!(
            offset_of!(Instance, inner),
            offset_of!(ReadOnlyInstance, inner),
        );
    }
}
