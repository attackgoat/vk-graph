//! Vulkan initialization types

use {
    super::{DriverError, physical_device::PhysicalDevice},
    ash::{ext, vk},
    derive_builder::{Builder, UninitializedFieldError},
    log::{debug, error, trace, warn},
    std::{
        ffi::CStr,
        fmt::{Debug, Formatter},
        ops::Deref,
        os::raw::c_char,
        sync::Arc,
        thread::panicking,
    },
};

#[cfg(not(target_os = "macos"))]
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

#[cfg(not(target_os = "macos"))]
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
    Vulkan12,

    /// Version 1.3
    #[default]
    Vulkan13,
}

impl ApiVersion {
    /// The most recent supported version of Vulkan
    pub const MAX: Self = Self::Vulkan13;
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
    /// Creates a new Vulkan instance.
    #[profiling::function]
    pub fn new(info: impl Into<InstanceInfo>) -> Result<Self, DriverError> {
        let info = info.into();

        // Required to enable non-uniform descriptor indexing (bindless)
        #[cfg(target_os = "macos")]
        unsafe {
            set_var("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "1");
        }

        #[cfg(not(target_os = "macos"))]
        let entry = unsafe {
            ash::Entry::load().map_err(|err| {
                error!("Vulkan driver not found: {err}");

                DriverError::Unsupported
            })?
        };

        #[cfg(target_os = "macos")]
        let entry = ash_molten::load();

        let mut extension_names = Self::debug_extension_names(info.debug);
        extension_names.extend(info.extension_names);
        let extension_name_ptrs = extension_names
            .iter()
            .copied()
            .map(CStr::as_ptr)
            .collect::<Box<[_]>>();

        let layer_names = Self::debug_layer_names(info.debug);
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

        #[cfg(target_os = "macos")]
        let (debug_loader, debug_callback, debug_utils) = (None, None, None);

        #[cfg(not(target_os = "macos"))]
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
                    warn!("unable to enumerate instance version: {err}");

                    DriverError::Unsupported
                }
            })?
            .map(|version| {
                match (
                    vk::api_version_major(version),
                    vk::api_version_minor(version),
                ) {
                    (1, x) if x >= 3 => Some(ApiVersion::Vulkan13),
                    (1, 2) => Some(ApiVersion::Vulkan12),
                    (major, minor) => {
                        warn!("unsupported Vulkan version: {major}.{minor}");

                        None
                    }
                }
            })
            .ok_or(DriverError::Unsupported)?
            .unwrap_or(ApiVersion::MAX);

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
            }),
        })
    }

    fn debug_extension_names(
        #[cfg_attr(target_os = "macos", allow(unused_variables))] debug: bool,
    ) -> Vec<&'static CStr> {
        #[cfg_attr(target_os = "macos", allow(unused_mut))]
        let mut res = vec![];

        #[cfg(not(target_os = "macos"))]
        if debug {
            #[allow(deprecated)]
            res.push(ext::debug_report::NAME);
            res.push(ext::debug_utils::NAME);
        }

        res
    }

    fn debug_layer_names(
        #[cfg_attr(target_os = "macos", allow(unused_variables))] debug: bool,
    ) -> Vec<&'static CStr> {
        #[cfg_attr(target_os = "macos", allow(unused_mut))]
        let mut res = vec![];

        #[cfg(not(target_os = "macos"))]
        if debug {
            res.push(c"VK_LAYER_KHRONOS_validation");
        }

        res
    }

    /// Returns a wrapper structure for a physical device of this instance.
    #[profiling::function]
    pub fn physical_device(
        this: &Self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<PhysicalDevice, DriverError> {
        let physical_device = PhysicalDevice::new(this.clone(), physical_device)?;
        let major = vk::api_version_major(physical_device.properties_v1_0.api_version);
        let minor = vk::api_version_minor(physical_device.properties_v1_0.api_version);
        let supports_vulkan_1_2 = major == 1 && minor >= 2;

        if !supports_vulkan_1_2 {
            warn!(
                "physical device `{}` does not support Vulkan v1.2",
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
                    warn!("unable to create physical device at index {idx}: {err}");
                }

                res.ok().filter(|physical_device| {
                    let major = vk::api_version_major(physical_device.properties_v1_0.api_version);
                    let minor = vk::api_version_minor(physical_device.properties_v1_0.api_version);
                    let supports_vulkan_1_2 = major == 1 && minor >= 2;

                    if !supports_vulkan_1_2 {
                        warn!(
                            "physical device `{}` does not support Vulkan v1.2",
                            physical_device.properties_v1_0.device_name
                        );
                    }

                    supports_vulkan_1_2
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
    derive(Clone, Debug),
    pattern = "owned"
)]
#[non_exhaustive]
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
    /// Converts a `InstanceInfo` into a `InstanceInfoBuilder`.
    #[inline(always)]
    pub fn to_builder(self) -> InstanceInfoBuilder {
        InstanceInfoBuilder {
            api_version: Some(self.api_version),
            debug: Some(self.debug),
            extension_names: Some(self.extension_names),
        }
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

            self.instance.destroy_instance(None);
        }
    }
}

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
mod tests {
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
