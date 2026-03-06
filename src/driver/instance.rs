use {
    super::{DriverError, physical_device::PhysicalDevice},
    ash::{Entry, ext, vk},
    log::{debug, error, trace, warn},
    std::{
        ffi::{CStr, CString},
        fmt::{Debug, Formatter},
        ops::Deref,
        os::raw::c_char,
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

#[cfg(target_os = "macos")]
use std::env::set_var;

#[cfg(any(not(target_os = "macos"), feature = "loaded"))]
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

/// There is no global state in Vulkan and all per-application state is stored in a VkInstance
/// object.
///
/// Creating an Instance initializes the Vulkan library and allows the application to pass
/// information about itself to the implementation.
pub struct Instance {
    _debug_callback: Option<vk::DebugReportCallbackEXT>,
    #[allow(deprecated)] // TODO: Remove? Look into this....
    _debug_loader: Option<ext::debug_report::Instance>,
    debug_utils: Option<ext::debug_utils::Instance>,
    entry: Entry,
    instance: ash::Instance,
}

impl Instance {
    /// Creates a new Vulkan instance.
    #[profiling::function]
    pub fn create<'a>(
        debug: bool,
        required_extensions: impl Iterator<Item = &'a CStr>,
    ) -> Result<Self, DriverError> {
        // Required to enable non-uniform descriptor indexing (bindless)
        #[cfg(target_os = "macos")]
        unsafe {
            set_var("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "1");
        }

        // Link molten-vk dynamically if not on MacOS, or if explicitly requested.
        #[cfg(any(not(target_os = "macos"), feature = "loaded"))]
        let entry = unsafe {
            Entry::load().map_err(|err| {
                error!("Vulkan driver not found: {err}");

                DriverError::Unsupported
            })?
        };
        // On MacOS, by default link molten-vk statically using ash-molten.
        #[cfg(all(target_os = "macos", not(feature = "loaded")))]
        let entry = ash_molten::load();

        #[allow(unused_mut)]
        let mut required_extensions = required_extensions.collect::<Vec<_>>();

        // If linking dynamically on MacOS, we require a few additional extensions.
        // Based on "Encountered VK_ERROR_INCOMPATIBLE_DRIVER" section in:
        // https://vulkan.lunarg.com/doc/view/latest/mac/getting_started.html
        #[cfg(all(target_os = "macos", feature = "loaded"))]
        {
            required_extensions.push(ash::khr::get_physical_device_properties2::NAME);
            required_extensions.push(ash::khr::portability_enumeration::NAME);
        }

        let instance_extensions = required_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .chain(unsafe { Self::extension_names(debug).into_iter() })
            .collect::<Box<[_]>>();
        let layer_names = Self::layer_names(debug);
        let layer_names: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();
        let app_desc = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let instance_desc = vk::InstanceCreateInfo::default()
            .application_info(&app_desc)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&instance_extensions);

        // Molten-vk doesn't support the full Vulkan feature set, hence the portability flag needs to be set.
        #[cfg(all(target_os = "macos", feature = "loaded"))]
        let instance_desc = instance_desc.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);

        let instance = unsafe {
            entry
                .create_instance(&instance_desc, None)
                .map_err(|vkerr| {
                    if debug {
                        warn!("debug may only be enabled with a valid Vulkan SDK installation");
                    }
                    error!("Vulkan Error: {}", vkerr);
                    error!("Vulkan driver does not support API v1.2");

                    for layer_name in Self::layer_names(debug) {
                        debug!("Layer: {:?}", layer_name);
                    }

                    for extension_name in required_extensions {
                        debug!("Extension: {:?}", extension_name);
                    }

                    DriverError::Unsupported
                })?
        };

        trace!("created a Vulkan instance");

        #[cfg(all(target_os = "macos", not(feature = "loaded")))]
        let (debug_loader, debug_callback, debug_utils) = (None, None, None);

        #[cfg(any(not(target_os = "macos"), feature = "loaded"))]
        let (debug_loader, debug_callback, debug_utils) = if debug {
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
            _debug_callback: debug_callback,
            _debug_loader: debug_loader,
            debug_utils,
            entry,
            instance,
        })
    }

    /// Loads an existing Vulkan instance that may have been created by other means.
    ///
    /// This is useful when you want to use a Vulkan instance created by some other library, such
    /// as OpenXR.
    #[profiling::function]
    pub fn load(entry: Entry, instance: vk::Instance) -> Result<Self, DriverError> {
        if instance == vk::Instance::null() {
            return Err(DriverError::InvalidData);
        }

        let instance = unsafe { ash::Instance::load(entry.static_fn(), instance) };

        Ok(Self {
            _debug_callback: None,
            _debug_loader: None,
            debug_utils: None,
            entry,
            instance,
        })
    }

    /// Returns the `ash` entrypoint for Vulkan functions.
    pub fn entry(this: &Self) -> &Entry {
        &this.entry
    }

    unsafe fn extension_names(
        #[cfg_attr(target_os = "macos", allow(unused_variables))] debug: bool,
    ) -> Vec<*const c_char> {
        if cfg!(all(
            target_os = "macos",
            not(feature = "loaded")
        )) {
            vec![]
        } else {
            let mut res = vec![];
            if debug {
                #[allow(deprecated)]
                res.push(ext::debug_report::NAME.as_ptr());
                res.push(ext::debug_utils::NAME.as_ptr());
            }
            res
        }
    }

    /// Returns `true` if this instance was created with debug layers enabled.
    pub fn is_debug(this: &Self) -> bool {
        this.debug_utils.is_some()
    }

    fn layer_names(
        #[cfg_attr(target_os = "macos", allow(unused_variables))] debug: bool,
    ) -> Vec<CString> {
        if cfg!(all(
            target_os = "macos",
            not(feature = "loaded")
        )) {
            vec![]
        } else {
            let mut res = vec![];
            if debug {
                res.push(CString::new("VK_LAYER_KHRONOS_validation").unwrap());
            }
            res
        }
    }

    /// Returns the available physical devices of this instance.
    #[profiling::function]
    pub fn physical_devices(this: &Self) -> Result<Vec<PhysicalDevice>, DriverError> {
        let physical_devices = unsafe { this.enumerate_physical_devices() };

        Ok(physical_devices
            .map_err(|err| {
                error!("unable to enumerate physical devices: {err}");

                DriverError::Unsupported
            })?
            .into_iter()
            .enumerate()
            .filter_map(|(idx, physical_device)| {
                let res = PhysicalDevice::new(this, physical_device);

                if let Err(err) = &res {
                    warn!("unable to create physical device at index {idx}: {err}");
                }

                res.ok().filter(|physical_device| {
                    let major = vk::api_version_major(physical_device.properties_v1_0.api_version);
                    let minor = vk::api_version_minor(physical_device.properties_v1_0.api_version);
                    let supports_vulkan_1_2 = major > 1 || (major == 1 && minor >= 2);

                    if !supports_vulkan_1_2 {
                        warn!(
                            "physical device `{}` does not support Vulkan v1.2",
                            physical_device.properties_v1_0.device_name
                        );
                    }

                    supports_vulkan_1_2
                })
            })
            .collect())
    }
}

impl Debug for Instance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Instance")
    }
}

impl Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

impl Drop for Instance {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        unsafe {
            #[allow(deprecated)]
            if let Some(debug_loader) = &self._debug_loader {
                let debug_callback = self._debug_callback.unwrap();
                debug_loader.destroy_debug_report_callback(debug_callback, None);
            }

            self.instance.destroy_instance(None);
        }
    }
}
