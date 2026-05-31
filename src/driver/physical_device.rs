//! Physical device types

use {
    super::{DriverError, instance::Instance},
    crate::driver::device::Device,
    ash::{ext, khr, vk},
    log::{debug, error, warn},
    std::{
        collections::HashSet,
        ffi::{CStr, c_char},
        fmt::{Debug, Formatter},
        iter::repeat_n,
    },
};

// TODO: There is a bunch of unsafe cstr handling here - does not check for null-termination

fn vk_cstr_to_string_lossy(cstr: &[c_char]) -> String {
    unsafe { CStr::from_ptr(cstr.as_ptr()) }
        .to_string_lossy()
        .to_string()
}

/// Properties of the physical device for acceleration structures.
///
/// See [`VkPhysicalDeviceAccelerationStructurePropertiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceAccelerationStructurePropertiesKHR.html).
#[derive(Clone, Copy, Debug)]
pub struct AccelerationStructureProperties {
    /// The maximum number of geometries in a bottom level acceleration structure.
    pub max_geometry_count: u64,

    /// The maximum number of instances in a top level acceleration structure.
    pub max_instance_count: u64,

    /// The maximum number of triangles or AABBs in all geometries in a bottom level acceleration
    /// structure.
    pub max_primitive_count: u64,

    /// The maximum number of acceleration structure bindings that can be accessible to a single
    /// shader stage in a pipeline layout.
    ///
    /// Descriptor bindings with a descriptor type of
    /// `VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR` count against this limit.
    pub max_per_stage_descriptor_accel_structs: u32,

    /// The maximum number of acceleration structure descriptors that can be included in descriptor
    /// bindings in a pipeline layout across all pipeline shader stages and descriptor set numbers.
    ///
    /// Descriptor bindings with a descriptor type of
    /// `VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR` count against this limit.
    pub max_descriptor_set_accel_structs: u32,

    /// The minimum required alignment, in bytes, for scratch data passed in to an acceleration
    /// structure build command.
    pub min_accel_struct_scratch_offset_alignment: u32,
}

impl From<vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'_>>
    for AccelerationStructureProperties
{
    fn from(props: vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'_>) -> Self {
        Self {
            max_geometry_count: props.max_geometry_count,
            max_instance_count: props.max_instance_count,
            max_primitive_count: props.max_primitive_count,
            max_per_stage_descriptor_accel_structs: props
                .max_per_stage_descriptor_acceleration_structures,
            max_descriptor_set_accel_structs: props.max_descriptor_set_acceleration_structures,
            min_accel_struct_scratch_offset_alignment: props
                .min_acceleration_structure_scratch_offset_alignment,
        }
    }
}

/// Structure describing depth/stencil resolve properties that can be supported by an
/// implementation.
///
/// See [`VkPhysicalDeviceDepthStencilResolveProperties`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceDepthStencilResolveProperties.html).
#[derive(Clone, Copy, Debug)]
pub struct DepthStencilResolveProperties {
    /// A bitmask indicating the set of supported depth resolve modes.
    ///
    /// `VK_RESOLVE_MODE_SAMPLE_ZERO_BIT` must be included in the set but implementations may
    /// support additional modes.
    pub supported_depth_resolve_modes: vk::ResolveModeFlags,

    /// A bitmask of indicating the set of supported stencil resolve modes.
    ///
    /// `VK_RESOLVE_MODE_SAMPLE_ZERO_BIT` must be included in the set but implementations may
    /// support additional modes. `VK_RESOLVE_MODE_AVERAGE_BIT` must not be included in the set.
    pub supported_stencil_resolve_modes: vk::ResolveModeFlags,

    /// `true` if the implementation supports setting the depth and stencil resolve modes to
    /// different values when one of those modes is `VK_RESOLVE_MODE_NONE`. Otherwise the
    /// implementation only supports setting both modes to the same value.
    pub independent_resolve_none: bool,

    /// `true` if the implementation supports all combinations of the supported depth and stencil
    /// resolve modes, including setting either depth or stencil resolve mode to
    /// `VK_RESOLVE_MODE_NONE`.
    ///
    /// An implementation that supports `independent_resolve` must also support
    /// `independent_resolve_none`.
    pub independent_resolve: bool,
}

impl From<vk::PhysicalDeviceDepthStencilResolveProperties<'_>> for DepthStencilResolveProperties {
    fn from(props: vk::PhysicalDeviceDepthStencilResolveProperties<'_>) -> Self {
        Self {
            supported_depth_resolve_modes: props.supported_depth_resolve_modes,
            supported_stencil_resolve_modes: props.supported_stencil_resolve_modes,
            independent_resolve_none: props.independent_resolve_none == vk::TRUE,
            independent_resolve: props.independent_resolve == vk::TRUE,
        }
    }
}

/// Features of the physical device for vertex indexing.
///
/// See [`VkPhysicalDeviceIndexTypeUint8FeaturesEXT`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceIndexTypeUint8FeaturesEXT.html).
#[derive(Clone, Copy, Debug, Default)]
pub struct IndexTypeUint8Features {
    /// Indicates that VK_INDEX_TYPE_UINT8_EXT can be used with vkCmdBindIndexBuffer2KHR and
    /// vkCmdBindIndexBuffer.
    pub index_type_uint8: bool,
}

impl From<vk::PhysicalDeviceIndexTypeUint8FeaturesEXT<'_>> for IndexTypeUint8Features {
    fn from(features: vk::PhysicalDeviceIndexTypeUint8FeaturesEXT<'_>) -> Self {
        Self {
            index_type_uint8: features.index_type_uint8 == vk::TRUE,
        }
    }
}

/// Structure which holds data about the physical hardware selected by the current device.
#[derive(Clone)]
#[read_only::cast]
pub struct PhysicalDevice {
    /// Describes the properties of the device which relate to acceleration structures, if
    /// available.
    ///
    /// _Note:_ This field is read-only.
    pub accel_struct_properties: Option<AccelerationStructureProperties>,

    /// Describes the properties of the device which relate to depth/stencil resolve operations.
    ///
    /// _Note:_ This field is read-only.
    pub depth_stencil_resolve_properties: DepthStencilResolveProperties,

    /// Describes the features of the physical device which are part of the Vulkan 1.0 base feature
    /// set.
    ///
    /// _Note:_ This field is read-only.
    pub features_v1_0: Vulkan10Features,

    /// Describes the features of the physical device which are part of the Vulkan 1.1 base feature
    /// set.
    ///
    /// _Note:_ This field is read-only.
    pub features_v1_1: Vulkan11Features,

    /// Describes the features of the physical device which are part of the Vulkan 1.2 base feature
    /// set.
    ///
    /// _Note:_ This field is read-only.
    pub features_v1_2: Vulkan12Features,

    /// The native Vulkan resource handle of this buffer.
    ///
    /// _Note:_ This field is read-only.
    pub handle: vk::PhysicalDevice,

    /// Describes the features of the physical device which relate to vertex indexing.
    ///
    /// _Note:_ This field is read-only.
    pub index_type_uint8_features: IndexTypeUint8Features,

    /// The Vulkan instance which owns this device.
    ///
    /// _Note:_ This field is read-only.
    pub instance: Instance,

    /// Memory properties of the physical device.
    ///
    /// _Note:_ This field is read-only.
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,

    /// Device properties of the physical device which are part of the Vulkan 1.0 base feature set.
    ///
    /// _Note:_ This field is read-only.
    pub properties_v1_0: Vulkan10Properties,

    /// Describes the properties of the physical device which are part of the Vulkan 1.1 base
    /// feature set.
    ///
    /// _Note:_ This field is read-only.
    pub properties_v1_1: Vulkan11Properties,

    /// Describes the properties of the physical device which are part of the Vulkan 1.2 base
    /// feature set.
    ///
    /// _Note:_ This field is read-only.
    pub properties_v1_2: Vulkan12Properties,

    /// Describes the queues offered by this physical device.
    ///
    /// _Note:_ This field is read-only.
    pub queue_families: Box<[vk::QueueFamilyProperties]>,

    pub(crate) queue_family_indices: Box<[u32]>,

    /// Describes the features of the device which relate to ray query, if available.
    ///
    /// _Note:_ This field is read-only.
    pub ray_query_features: RayQueryFeatures,

    /// Describes the features of the device which relate to ray tracing, if available.
    ///
    /// _Note:_ This field is read-only.
    pub ray_trace_features: RayTraceFeatures,

    /// Describes the properties of the device which relate to ray tracing, if available.
    ///
    /// _Note:_ This field is read-only.
    pub ray_trace_properties: Option<RayTraceProperties>,

    /// Describes the properties of the device which relate to min/max sampler filtering.
    ///
    /// _Note:_ This field is read-only.
    pub sampler_filter_minmax_properties: SamplerFilterMinmaxProperties,

    /// True if the device supports swapchain use.
    ///
    /// _Note:_ This field is read-only.
    pub swapchain_ext: bool,
}

impl PhysicalDevice {
    /// Creates a physical device wrapper which reports features and properties.
    #[profiling::function]
    pub(super) fn new(
        instance: Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, DriverError> {
        if physical_device == vk::PhysicalDevice::null() {
            warn!("invalid physical device handle: null");

            return Err(DriverError::InvalidData);
        }

        let (memory_properties, queue_families) = unsafe {
            (
                instance.get_physical_device_memory_properties(physical_device),
                instance.get_physical_device_queue_family_properties(physical_device),
            )
        };

        let mut queue_family_indices = Vec::with_capacity(queue_families.len());
        for idx in 0..queue_families.len() as u32 {
            queue_family_indices.push(idx);
        }

        let queue_families = queue_families.into();
        let queue_family_indices = queue_family_indices.into();

        let ash::InstanceFnV1_1 {
            get_physical_device_features2,
            get_physical_device_properties2,
            ..
        } = instance.fp_v1_1();

        // Gather required features of the physical device
        let mut features_v1_1 = vk::PhysicalDeviceVulkan11Features::default();
        let mut features_v1_2 = vk::PhysicalDeviceVulkan12Features::default();
        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut index_type_u8_features = vk::PhysicalDeviceIndexTypeUint8FeaturesEXT::default();
        let mut ray_query_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
        let mut ray_trace_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut features_v1_1)
            .push_next(&mut features_v1_2)
            .push_next(&mut acceleration_structure_features)
            .push_next(&mut index_type_u8_features)
            .push_next(&mut ray_query_features)
            .push_next(&mut ray_trace_features);
        unsafe {
            get_physical_device_features2(physical_device, &mut features);
        }
        let features_v1_0 = features.features.into();
        let features_v1_1 = features_v1_1.into();
        let features_v1_2 = features_v1_2.into();

        // Gather required properties of the physical device
        let mut properties_v1_1 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut properties_v1_2 = vk::PhysicalDeviceVulkan12Properties::default();
        let mut accel_struct_properties =
            vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut depth_stencil_resolve_properties =
            vk::PhysicalDeviceDepthStencilResolveProperties::default();
        let mut ray_trace_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut sampler_filter_minmax_properties =
            vk::PhysicalDeviceSamplerFilterMinmaxProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut properties_v1_1)
            .push_next(&mut properties_v1_2)
            .push_next(&mut accel_struct_properties)
            .push_next(&mut depth_stencil_resolve_properties)
            .push_next(&mut ray_trace_properties)
            .push_next(&mut sampler_filter_minmax_properties);
        unsafe {
            get_physical_device_properties2(physical_device, &mut properties);
        }
        let properties_v1_0: Vulkan10Properties = properties.properties.into();
        let properties_v1_1 = properties_v1_1.into();
        let properties_v1_2 = properties_v1_2.into();
        let depth_stencil_resolve_properties = depth_stencil_resolve_properties.into();
        let sampler_filter_minmax_properties = sampler_filter_minmax_properties.into();

        let extension_properties = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .map_err(|err| {
                    error!("unable to enumerate device extensions: {err}");

                    DriverError::Unsupported
                })?
        };

        debug!("physical device: {}", &properties_v1_0.device_name);

        for prop in &extension_properties {
            let extension_name = prop.extension_name.as_ptr();

            if extension_name.is_null() {
                warn!("invalid device extension name pointer: null");

                return Err(DriverError::InvalidData);
            }

            let extension_name = unsafe { CStr::from_ptr(extension_name) };

            debug!("extension {:?} v{}", extension_name, prop.spec_version);
        }

        // Check for supported extensions
        let extension_names = extension_properties
            .iter()
            .map(|prop| prop.extension_name.as_ptr())
            .filter(|extension_name| !extension_name.is_null())
            .map(|extension_name| unsafe { CStr::from_ptr(extension_name) })
            .collect::<HashSet<_>>();
        let supports_accel_struct = extension_names.contains(khr::acceleration_structure::NAME)
            && extension_names.contains(khr::deferred_host_operations::NAME);
        let supports_index_type_uint8 = extension_names.contains(ext::index_type_uint8::NAME);
        let supports_ray_query = extension_names.contains(khr::ray_query::NAME);
        let supports_ray_trace = extension_names.contains(khr::ray_tracing_pipeline::NAME);
        let swapchain_ext = extension_names.contains(khr::swapchain::NAME) && instance.surface_ext;

        // Gather optional features and properties of the physical device
        let index_type_uint8_features = if supports_index_type_uint8 {
            index_type_u8_features.into()
        } else {
            Default::default()
        };
        let ray_query_features = if supports_ray_query {
            ray_query_features.into()
        } else {
            Default::default()
        };
        let ray_trace_features = if supports_ray_trace {
            ray_trace_features.into()
        } else {
            Default::default()
        };
        let accel_struct_properties = supports_accel_struct.then(|| accel_struct_properties.into());
        let ray_trace_properties = supports_ray_trace.then(|| ray_trace_properties.into());

        Ok(Self {
            accel_struct_properties,
            depth_stencil_resolve_properties,
            features_v1_0,
            features_v1_1,
            features_v1_2,
            handle: physical_device,
            index_type_uint8_features,
            instance,
            memory_properties,
            properties_v1_0,
            properties_v1_1,
            properties_v1_2,
            queue_families,
            queue_family_indices,
            ray_query_features,
            ray_trace_features,
            ray_trace_properties,
            sampler_filter_minmax_properties,
            swapchain_ext,
        })
    }

    /// Prepares device creation information and calls the provided callback to allow an application
    /// to control the device creation process.
    ///
    /// _Note:_  This is only useful for interoperating with other libraries as device creation is
    /// normally handled by the [`Device::create_display`] and [`Device::new`]
    /// functions.
    ///
    /// # Safety
    ///
    /// This comes with all the caveats of using `ash` builder types, which are inherently
    /// dangerous. Use with extreme caution.
    #[profiling::function]
    pub unsafe fn create_ash_device<F>(&self, create_fn: F) -> ash::prelude::VkResult<ash::Device>
    where
        F: FnOnce(vk::DeviceCreateInfo) -> ash::prelude::VkResult<ash::Device>,
    {
        let mut enabled_ext_names = Vec::with_capacity(6);

        // The swapchain extension is required for presentation support, so we enable it whenever
        // the physical device reports support. Imported instances may already carry the required
        // instance extensions even though vk-graph did not create them.
        if self.swapchain_ext {
            enabled_ext_names.push(khr::swapchain::NAME.as_ptr());
        }

        if self.accel_struct_properties.is_some() {
            enabled_ext_names.push(khr::acceleration_structure::NAME.as_ptr());
            enabled_ext_names.push(khr::deferred_host_operations::NAME.as_ptr());
        }

        if self.ray_query_features.ray_query {
            enabled_ext_names.push(khr::ray_query::NAME.as_ptr());
        }

        if self.ray_trace_features.ray_tracing_pipeline {
            enabled_ext_names.push(khr::ray_tracing_pipeline::NAME.as_ptr());
        }

        if self.index_type_uint8_features.index_type_uint8 {
            enabled_ext_names.push(ext::index_type_uint8::NAME.as_ptr());
        }

        // Molten-vk doesn't support the full Vulkan feature set, hence the portability subset
        // extension must be enabled.
        #[cfg(all(target_os = "macos", feature = "loaded"))]
        enabled_ext_names.push(khr::portability_subset::NAME.as_ptr());

        let priorities = repeat_n(
            1.0,
            self.queue_families
                .iter()
                .map(|family| family.queue_count)
                .max()
                .unwrap_or_default() as _,
        )
        .collect::<Box<_>>();

        let queue_infos = self
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
        } = self.instance.fp_v1_1();
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

        if self.accel_struct_properties.is_some() {
            features = features.push_next(&mut acceleration_structure_features);
        }

        if self.ray_query_features.ray_query {
            features = features.push_next(&mut ray_query_features);
        }

        if self.ray_trace_features.ray_tracing_pipeline {
            features = features.push_next(&mut ray_trace_features);
        }

        if self.index_type_uint8_features.index_type_uint8 {
            features = features.push_next(&mut index_type_uint8_features);
        }

        unsafe { get_physical_device_features2(self.handle, &mut features) };

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&enabled_ext_names)
            .push_next(&mut features);

        create_fn(device_create_info)
    }

    /// Lists the capabilities of a given format.
    #[profiling::function]
    pub fn format_properties(&self, format: vk::Format) -> vk::FormatProperties {
        unsafe {
            self.instance
                .get_physical_device_format_properties(self.handle, format)
        }
    }

    /// Lists the physical device's image format capabilities.
    ///
    /// A result of `None` indicates the format is not supported.
    #[profiling::function]
    pub fn image_format_properties(
        &self,
        format: vk::Format,
        ty: vk::ImageType,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        flags: vk::ImageCreateFlags,
    ) -> Result<Option<vk::ImageFormatProperties>, DriverError> {
        unsafe {
            match self.instance.get_physical_device_image_format_properties(
                self.handle,
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
                Err(err) => {
                    warn!("unable to query image format properties: {err}");

                    Err(DriverError::OutOfMemory)
                }
            }
        }
    }

    /// Creates a logical [`Device`] from this selected physical device.
    pub fn try_into_device(self) -> Result<Device, DriverError> {
        Device::try_from_physical_device(self)
    }
}

impl Debug for PhysicalDevice {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({:?})",
            &self.properties_v1_0.device_name, self.properties_v1_0.device_type
        )
    }
}

/// Features of the physical device for ray query.
///
/// See [`VkPhysicalDeviceRayQueryFeaturesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceRayQueryFeaturesKHR.html).
#[derive(Clone, Copy, Debug, Default)]
pub struct RayQueryFeatures {
    /// Indicates whether the implementation supports ray query (`OpRayQueryProceedKHR`)
    /// functionality.
    pub ray_query: bool,
}

impl From<vk::PhysicalDeviceRayQueryFeaturesKHR<'_>> for RayQueryFeatures {
    fn from(features: vk::PhysicalDeviceRayQueryFeaturesKHR<'_>) -> Self {
        Self {
            ray_query: features.ray_query == vk::TRUE,
        }
    }
}

/// Features of the physical device for ray tracing.
///
/// See [`VkPhysicalDeviceRayTracingPipelineFeaturesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceRayTracingPipelineFeaturesKHR.html).
#[derive(Clone, Copy, Debug, Default)]
pub struct RayTraceFeatures {
    /// Indicates whether the implementation supports the ray tracing pipeline functionality.
    ///
    /// See
    /// See the [ray tracing pipeline chapter](https://docs.vulkan.org/spec/latest/chapters/raytracing.html).
    pub ray_tracing_pipeline: bool,

    /// Indicates whether the implementation supports saving and reusing shader group handles, e.g.
    /// for trace capture and replay.
    pub ray_tracing_pipeline_shader_group_handle_capture_replay: bool,

    /// Indicates whether the implementation supports reuse of shader group handles being
    /// arbitrarily mixed with creation of non-reused shader group handles.
    ///
    /// If this is `false`, all reused shader group handles must be specified before any non-reused
    /// handles may be created.
    pub ray_tracing_pipeline_shader_group_handle_capture_replay_mixed: bool,

    /// Indicates whether the implementation supports indirect ray tracing commands, e.g.
    /// `vkCmdTraceRaysIndirectKHR`.
    pub ray_tracing_pipeline_trace_rays_indirect: bool,

    /// Indicates whether the implementation supports primitive culling during ray traversal.
    pub ray_traversal_primitive_culling: bool,
}

impl From<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR<'_>> for RayTraceFeatures {
    fn from(features: vk::PhysicalDeviceRayTracingPipelineFeaturesKHR<'_>) -> Self {
        Self {
            ray_tracing_pipeline: features.ray_tracing_pipeline == vk::TRUE,
            ray_tracing_pipeline_shader_group_handle_capture_replay: features
                .ray_tracing_pipeline_shader_group_handle_capture_replay
                == vk::TRUE,
            ray_tracing_pipeline_shader_group_handle_capture_replay_mixed: features
                .ray_tracing_pipeline_shader_group_handle_capture_replay_mixed
                == vk::TRUE,
            ray_tracing_pipeline_trace_rays_indirect: features
                .ray_tracing_pipeline_trace_rays_indirect
                == vk::TRUE,
            ray_traversal_primitive_culling: features.ray_traversal_primitive_culling == vk::TRUE,
        }
    }
}

/// Properties of the physical device for ray tracing.
///
/// See [`VkPhysicalDeviceRayTracingPipelinePropertiesKHR`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceRayTracingPipelinePropertiesKHR.html).
#[derive(Clone, Copy, Debug)]
pub struct RayTraceProperties {
    /// The size in bytes of the shader header.
    pub shader_group_handle_size: u32,

    /// The maximum number of levels of ray recursion allowed in a trace command.
    pub max_ray_recursion_depth: u32,

    /// The maximum stride in bytes allowed between shader groups in the shader binding table.
    pub max_shader_group_stride: u32,

    /// The required alignment in bytes for the base of the shader binding table.
    pub shader_group_base_alignment: u32,

    /// The number of bytes for the information required to do capture and replay for shader group
    /// handles.
    pub shader_group_handle_capture_replay_size: u32,

    /// The maximum number of ray generation shader invocations which may be produced by a single
    /// vkCmdTraceRaysIndirectKHR or vkCmdTraceRaysKHR command.
    pub max_ray_dispatch_invocation_count: u32,

    /// The required alignment in bytes for each shader binding table entry.
    ///
    /// The value must be a power of two.
    pub shader_group_handle_alignment: u32,

    /// The maximum size in bytes for a ray attribute structure.
    pub max_ray_hit_attribute_size: u32,
}

impl From<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'_>> for RayTraceProperties {
    fn from(props: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'_>) -> Self {
        Self {
            shader_group_handle_size: props.shader_group_handle_size,
            max_ray_recursion_depth: props.max_ray_recursion_depth,
            max_shader_group_stride: props.max_shader_group_stride,
            shader_group_base_alignment: props.shader_group_base_alignment,
            shader_group_handle_capture_replay_size: props.shader_group_handle_capture_replay_size,
            max_ray_dispatch_invocation_count: props.max_ray_dispatch_invocation_count,
            shader_group_handle_alignment: props.shader_group_handle_alignment,
            max_ray_hit_attribute_size: props.max_ray_hit_attribute_size,
        }
    }
}

/// Properties of the physical device for min/max sampler filtering.
///
/// See [`VkPhysicalDeviceLimits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceLimits.html).
#[derive(Clone, Copy, Debug)]
pub struct SamplerFilterMinmaxProperties {
    /// When `false` the component mapping of the image view used with min/max filtering must have
    /// been created with the r component set to the identity swizzle. Only the r component of the
    /// sampled image value is defined and the other component values are undefined.
    ///
    /// When `true` this restriction does not apply and image component mapping works as normal.
    pub image_component_mapping: bool,

    /// When `true` the following formats support the
    /// `VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT` feature with `VK_IMAGE_TILING_OPTIMAL`,
    /// if they support `VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT`:
    ///
    /// * [`vk::Format::R8_UNORM`]
    /// * [`vk::Format::R8_SNORM`]
    /// * [`vk::Format::R16_UNORM`]
    /// * [`vk::Format::R16_SNORM`]
    /// * [`vk::Format::R16_SFLOAT`]
    /// * [`vk::Format::R32_SFLOAT`]
    /// * [`vk::Format::D16_UNORM`]
    /// * [`vk::Format::X8_D24_UNORM_PACK32`]
    /// * [`vk::Format::D32_SFLOAT`]
    /// * [`vk::Format::D16_UNORM_S8_UINT`]
    /// * [`vk::Format::D24_UNORM_S8_UINT`]
    /// * [`vk::Format::D32_SFLOAT_S8_UINT`]
    ///
    /// If the format is a depth/stencil format, this bit only specifies that the depth aspect (not
    /// the stencil aspect) of an image of this format supports min/max filtering, and that min/max
    /// filtering of the depth aspect is supported when depth compare is disabled in the sampler.
    pub single_component_formats: bool,
}

impl From<vk::PhysicalDeviceSamplerFilterMinmaxProperties<'_>> for SamplerFilterMinmaxProperties {
    fn from(value: vk::PhysicalDeviceSamplerFilterMinmaxProperties<'_>) -> Self {
        Self {
            image_component_mapping: value.filter_minmax_image_component_mapping == vk::TRUE,
            single_component_formats: value.filter_minmax_single_component_formats == vk::TRUE,
        }
    }
}

/// Description of Vulkan features.
///
/// See [`VkPhysicalDeviceProperties`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceProperties.html).
#[derive(Clone, Copy, Debug)]
pub struct Vulkan10Features {
    /// Specifies that accesses to buffers are bounds-checked against the range of the buffer
    /// descriptor.
    pub robust_buffer_access: bool,

    /// Specifies the full 32-bit range of indices is supported for indexed draw calls when using a
    /// `VkIndexType` of `VK_INDEX_TYPE_UINT32`.
    ///
    /// `maxDrawIndexedIndexValue` is the maximum index value that may be used (aside from the
    /// primitive restart index, which is always 2^32 - 1 when the VkIndexType is
    /// `VK_INDEX_TYPE_UINT32`).
    ///
    /// If this feature is supported, `maxDrawIndexedIndexValue` must be 2^32 - 1; otherwise it
    /// must be no smaller than 2^24 - 1. See maxDrawIndexedIndexValue.
    pub full_draw_index_uint32: bool,

    /// Specifies whether image views with a `VkImageViewType` of `VK_IMAGE_VIEW_TYPE_CUBE_ARRAY`
    /// can be created, and that the corresponding `SampledCubeArray` and `ImageCubeArray` SPIR-V
    /// capabilities can be used in shader code.
    pub image_cube_array: bool,

    /// Specifies whether the `VkPipelineColorBlendAttachmentState` settings are controlled
    /// independently per-attachment.
    ///
    /// If this feature is not enabled, the `VkPipelineColorBlendAttachmentState` settings for all
    /// color attachments must be identical. Otherwise, a different
    /// `VkPipelineColorBlendAttachmentState` can be provided for each bound color attachment.
    pub independent_blend: bool,

    /// Specifies whether geometry shaders are supported.
    ///
    /// If this feature is not enabled, the `VK_SHADER_STAGE_GEOMETRY_BIT` and
    /// `VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT` enum values must not be used.
    ///
    /// This also specifies whether shader modules can declare the `Geometry` capability.
    pub geometry_shader: bool,

    /// Specifies whether tessellation control and evaluation shaders are supported.
    ///
    /// If this feature is not enabled, the `VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT`,
    /// `VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT`,
    /// `VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT`,
    /// `VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT`, and
    /// `VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO` enum values must not be used.
    ///
    /// This also specifies whether shader modules can declare the `Tessellation` capability.
    pub tessellation_shader: bool,

    /// Specifies whether Sample Shading and multisample interpolation are supported.
    ///
    /// If this feature is not enabled, the `sampleShadingEnable` member of the
    /// `VkPipelineMultisampleStateCreateInfo` structure must be set to `VK_FALSE` and the
    /// `minSampleShading` member is ignored.
    ///
    /// This also specifies whether shader modules can declare the `SampleRateShading` capability.
    pub sample_rate_shading: bool,

    /// Specifies whether blend operations which take two sources are supported.
    ///
    /// If this feature is not enabled, the `VK_BLEND_FACTOR_SRC1_COLOR`,
    /// `VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR`, `VK_BLEND_FACTOR_SRC1_ALPHA`, and
    /// `VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA` enum values must not be used as source or
    /// destination blending factors.
    ///
    /// See the [features chapter](https://docs.vulkan.org/spec/latest/chapters/features.html).
    pub dual_src_blend: bool,

    /// Specifies whether logic operations are supported.
    ///
    /// If this feature is not enabled, the `logicOpEnable` member of the
    /// `VkPipelineColorBlendStateCreateInfo` structure must be set to `VK_FALSE`, and the
    /// `logicOp` member is ignored.
    pub logic_op: bool,

    /// Specifies whether multiple draw indirect is supported.
    ///
    /// If this feature is not enabled, the `drawCount` parameter to the `vkCmdDrawIndirect` and
    /// `vkCmdDrawIndexedIndirect` commands must be `0` or `1`. The `maxDrawIndirectCount` member
    /// of the `VkPhysicalDeviceLimits` structure must also be `1` if this feature is not
    /// supported.
    ///
    /// See the [features chapter](https://docs.vulkan.org/spec/latest/chapters/features.html).
    pub multi_draw_indirect: bool,

    /// Specifies whether indirect drawing calls support the `firstInstance` parameter.
    ///
    /// If this feature is not enabled, the `firstInstance` member of all `VkDrawIndirectCommand`
    /// and `VkDrawIndexedIndirectCommand` structures that are provided to the `vkCmdDrawIndirect`
    /// and `vkCmdDrawIndexedIndirect` commands must be `0`.
    pub draw_indirect_first_instance: bool,

    /// Specifies whether depth clamping is supported.
    ///
    /// If this feature is not enabled, the `depthClampEnable` member of the
    /// `VkPipelineRasterizationStateCreateInfo` structure must be set to `VK_FALSE`. Otherwise,
    /// setting `depthClampEnable` to `VK_TRUE` will enable depth clamping.
    pub depth_clamp: bool,

    /// Specifies whether depth bias clamping is supported.
    ///
    /// If this feature is not enabled, the `depthBiasClamp` member of the
    /// `VkPipelineRasterizationStateCreateInfo` structure must be set to `0.0` unless the
    /// `VK_DYNAMIC_STATE_DEPTH_BIAS` dynamic state is enabled, and the `depthBiasClamp` parameter
    /// to `vkCmdSetDepthBias` must be set to `0.0`.
    pub depth_bias_clamp: bool,

    /// Specifies whether point and wireframe fill modes are supported.
    ///
    /// If this feature is not enabled, the `VK_POLYGON_MODE_POINT` and `VK_POLYGON_MODE_LINE` enum
    /// values must not be used.
    pub fill_mode_non_solid: bool,

    /// Specifies whether depth bounds tests are supported.
    ///
    /// If this feature is not enabled, the `depthBoundsTestEnable` member of the
    /// `VkPipelineDepthStencilStateCreateInfo` structure must be set to `VK_FALSE`. When
    /// `depthBoundsTestEnable` is set to `VK_FALSE`, the `minDepthBounds` and `maxDepthBounds`
    /// members of the `VkPipelineDepthStencilStateCreateInfo` structure are ignored.
    pub depth_bounds: bool,

    /// Specifies whether lines with width other than `1.0` are supported.
    ///
    /// If this feature is not enabled, the `lineWidth` member of the
    /// `VkPipelineRasterizationStateCreateInfo` structure must be set to `1.0` unless the
    /// `VK_DYNAMIC_STATE_LINE_WIDTH` dynamic state is enabled, and the `lineWidth` parameter to
    /// `vkCmdSetLineWidth` must be set to `1.0`.
    ///
    /// When this feature is supported, the range and granularity of supported line widths are
    /// indicated by the `lineWidthRange` and `lineWidthGranularity` members of the
    /// `VkPhysicalDeviceLimits` structure, respectively.
    pub wide_lines: bool,

    /// Specifies whether points with size greater than `1.0` are supported.
    ///
    /// If this feature is not enabled, only a point size of `1.0` written by a shader is
    /// supported.
    ///
    /// The range and granularity of supported point sizes are indicated by the `pointSizeRange`
    /// and `pointSizeGranularity` members of the `VkPhysicalDeviceLimits` structure,
    /// respectively.
    pub large_points: bool,

    /// Specifies whether the implementation is able to replace the alpha value of the fragment
    /// See the [features chapter](https://docs.vulkan.org/spec/latest/chapters/features.html).
    /// fragment operation.
    ///
    /// If this feature is not enabled, then the `alphaToOneEnable` member of the
    /// `VkPipelineMultisampleStateCreateInfo` structure must be set to `VK_FALSE`. Otherwise
    /// setting `alphaToOneEnable` to `VK_TRUE` will enable alpha-to-one behavior.
    pub alpha_to_one: bool,

    /// Specifies whether more than one viewport is supported.
    ///
    /// If this feature is not enabled:
    ///
    /// - The `viewportCount` and `scissorCount` members of the `VkPipelineViewportStateCreateInfo`
    ///   structure must be set to `1`.
    /// - The `firstViewport` and `viewportCount` parameters to the `vkCmdSetViewport` command must
    ///   be set to `0` and `1`, respectively.
    /// - The `firstScissor` and `scissorCount` parameters to the `vkCmdSetScissor` command must be
    ///   set to `0` and `1`, respectively.
    pub multi_viewport: bool,

    /// Specifies whether anisotropic filtering is supported.
    ///
    /// If this feature is not enabled, the `anisotropyEnable` member of the `VkSamplerCreateInfo`
    /// structure must be `VK_FALSE`.
    pub sampler_anisotropy: bool,

    /// Specifies whether all of the ETC2 and EAC compressed texture formats are supported.
    ///
    /// If this feature is enabled, then the `VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT`,
    /// `VK_FORMAT_FEATURE_BLIT_SRC_BIT` and `VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT`
    /// features must be supported in `optimalTilingFeatures` for the following formats:
    ///
    /// - VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK
    /// - VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK
    /// - VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK
    /// - VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK
    /// - VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK
    /// - VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK
    /// - VK_FORMAT_EAC_R11_UNORM_BLOCK
    /// - VK_FORMAT_EAC_R11_SNORM_BLOCK
    /// - VK_FORMAT_EAC_R11G11_UNORM_BLOCK
    /// - VK_FORMAT_EAC_R11G11_SNORM_BLOCK
    ///
    /// To query for additional properties, or if the feature is not enabled,
    /// `vkGetPhysicalDeviceFormatProperties` and `vkGetPhysicalDeviceImageFormatProperties` can be
    /// used to check for supported properties of individual formats as normal.
    pub texture_compression_etc2: bool,

    /// Specifies whether all of the ASTC LDR compressed texture formats are supported.
    ///
    /// If this feature is enabled, then the `VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT`,
    /// `VK_FORMAT_FEATURE_BLIT_SRC_BIT` and `VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT`
    /// features must be supported in `optimalTilingFeatures` for the following formats:
    ///
    /// - VK_FORMAT_ASTC_4x4_UNORM_BLOCK
    /// - VK_FORMAT_ASTC_4x4_SRGB_BLOCK
    /// - VK_FORMAT_ASTC_5x4_UNORM_BLOCK
    /// - VK_FORMAT_ASTC_5x4_SRGB_BLOCK
    /// - VK_FORMAT_ASTC_5x5_UNORM_BLOCK
    /// - VK_FORMAT_ASTC_5x5_SRGB_BLOCK
    /// - VK_FORMAT_ASTC_6x5_UNORM_BLOCK
    /// - VK_FORMAT_ASTC_6x5_SRGB_BLOCK
    /// - VK_FORMAT_ASTC_6x6_UNORM_BLOCK
    /// - VK_FORMAT_ASTC_6x6_SRGB_BLOCK
    /// - VK_FORMAT_ASTC_8x5_UNORM_BLOCK
    /// - VK_FORMAT_ASTC_8x5_SRGB_BLOCK
    /// - VK_FORMAT_ASTC_8x6_UNORM_BLOCK
    /// - VK_FORMAT_ASTC_8x6_SRGB_BLOCK
    /// - VK_FORMAT_ASTC_8x8_UNORM_BLOCK
    /// - VK_FORMAT_ASTC_8x8_SRGB_BLOCK
    /// - VK_FORMAT_ASTC_10x5_UNORM_BLOCK
    /// - VK_FORMAT_ASTC_10x5_SRGB_BLOCK
    /// - VK_FORMAT_ASTC_10x6_UNORM_BLOCK
    /// - VK_FORMAT_ASTC_10x6_SRGB_BLOCK
    /// - VK_FORMAT_ASTC_10x8_UNORM_BLOCK
    /// - VK_FORMAT_ASTC_10x8_SRGB_BLOCK
    /// - VK_FORMAT_ASTC_10x10_UNORM_BLOCK
    /// - VK_FORMAT_ASTC_10x10_SRGB_BLOCK
    /// - VK_FORMAT_ASTC_12x10_UNORM_BLOCK
    /// - VK_FORMAT_ASTC_12x10_SRGB_BLOCK
    /// - VK_FORMAT_ASTC_12x12_UNORM_BLOCK
    /// - VK_FORMAT_ASTC_12x12_SRGB_BLOCK
    ///
    /// To query for additional properties, or if the feature is not enabled,
    /// `vkGetPhysicalDeviceFormatProperties` and `vkGetPhysicalDeviceImageFormatProperties` can be
    /// used to check for supported properties of individual formats as normal.
    pub texture_compression_astc_ldr: bool,

    /// Specifies whether all of the BC compressed texture formats are supported.
    ///
    /// If this feature is enabled, then the `VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT`,
    /// `VK_FORMAT_FEATURE_BLIT_SRC_BIT` and `VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT`
    /// features must be supported in `optimalTilingFeatures` for the following formats:
    ///
    /// - VK_FORMAT_BC1_RGB_UNORM_BLOCK
    /// - VK_FORMAT_BC1_RGB_SRGB_BLOCK
    /// - VK_FORMAT_BC1_RGBA_UNORM_BLOCK
    /// - VK_FORMAT_BC1_RGBA_SRGB_BLOCK
    /// - VK_FORMAT_BC2_UNORM_BLOCK
    /// - VK_FORMAT_BC2_SRGB_BLOCK
    /// - VK_FORMAT_BC3_UNORM_BLOCK
    /// - VK_FORMAT_BC3_SRGB_BLOCK
    /// - VK_FORMAT_BC4_UNORM_BLOCK
    /// - VK_FORMAT_BC4_SNORM_BLOCK
    /// - VK_FORMAT_BC5_UNORM_BLOCK
    /// - VK_FORMAT_BC5_SNORM_BLOCK
    /// - VK_FORMAT_BC6H_UFLOAT_BLOCK
    /// - VK_FORMAT_BC6H_SFLOAT_BLOCK
    /// - VK_FORMAT_BC7_UNORM_BLOCK
    /// - VK_FORMAT_BC7_SRGB_BLOCK
    ///
    /// To query for additional properties, or if the feature is not enabled,
    /// `vkGetPhysicalDeviceFormatProperties` and `vkGetPhysicalDeviceImageFormatProperties` can be
    /// used to check for supported properties of individual formats as normal.
    pub texture_compression_bc: bool,

    /// Specifies whether storage buffers and images support stores and atomic operations in the
    /// vertex, tessellation, and geometry shader stages.
    ///
    /// If this feature is not enabled, all storage image, storage texel buffer, and storage buffer
    /// variables used by these stages in shader modules must be decorated with the `NonWritable`
    /// decoration (or the `readonly` memory qualifier in GLSL).
    pub vertex_pipeline_stores_and_atomics: bool,

    /// Specifies whether storage buffers and images support stores and atomic operations in the
    /// fragment shader stage.
    ///
    /// If this feature is not enabled, all storage image, storage texel buffer, and storage buffer
    /// variables used by the fragment stage in shader modules must be decorated with the
    /// `NonWritable` decoration (or the `readonly` memory qualifier in GLSL).
    pub fragment_stores_and_atomics: bool,

    /// Specifies whether the `PointSize` built-in decoration is available in the tessellation
    /// control, tessellation evaluation, and geometry shader stages.
    ///
    /// If this feature is not enabled, members decorated with the `PointSize` built-in decoration
    /// must not be read from or written to and all points written from a tessellation or geometry
    /// shader will have a size of `1.0`.
    ///
    /// This also specifies whether shader modules can declare the `TessellationPointSize`
    /// capability for tessellation control and evaluation shaders, or if the shader modules can
    /// declare the `GeometryPointSize` capability for geometry shaders.
    ///
    /// An implementation supporting this feature must also support one or both of the
    /// `tessellationShader` or `geometryShader` features.
    pub shader_tessellation_and_geometry_point_size: bool,

    /// Specifies whether the extended set of image gather instructions are available in shader
    /// code.
    ///
    /// If this feature is not enabled, the `OpImage*Gather` instructions do not support the
    /// `Offset` and `ConstOffsets` operands.
    ///
    /// This also specifies whether shader modules can declare the `ImageGatherExtended`
    /// capability.
    pub shader_image_gather_extended: bool,

    /// Specifies whether all the “storage image extended formats” below are supported.
    ///
    /// If this feature is supported, then the `VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT` must be
    /// supported in `optimalTilingFeatures` for the following formats:
    ///
    /// - VK_FORMAT_R16G16_SFLOAT
    /// - VK_FORMAT_B10G11R11_UFLOAT_PACK32
    /// - VK_FORMAT_R16_SFLOAT
    /// - VK_FORMAT_R16G16B16A16_UNORM
    /// - VK_FORMAT_A2B10G10R10_UNORM_PACK32
    /// - VK_FORMAT_R16G16_UNORM
    /// - VK_FORMAT_R8G8_UNORM
    /// - VK_FORMAT_R16_UNORM
    /// - VK_FORMAT_R8_UNORM
    /// - VK_FORMAT_R16G16B16A16_SNORM
    /// - VK_FORMAT_R16G16_SNORM
    /// - VK_FORMAT_R8G8_SNORM
    /// - VK_FORMAT_R16_SNORM
    /// - VK_FORMAT_R8_SNORM
    /// - VK_FORMAT_R16G16_SINT
    /// - VK_FORMAT_R8G8_SINT
    /// - VK_FORMAT_R16_SINT
    /// - VK_FORMAT_R8_SINT
    /// - VK_FORMAT_A2B10G10R10_UINT_PACK32
    /// - VK_FORMAT_R16G16_UINT
    /// - VK_FORMAT_R8G8_UINT
    /// - VK_FORMAT_R16_UINT
    /// - VK_FORMAT_R8_UINT
    ///
    /// _Note:_ `shaderStorageImageExtendedFormats` feature only adds a guarantee of format
    /// support, which is specified for the whole physical device. Therefore enabling or
    /// disabling the feature via vkCreateDevice has no practical effect.
    ///
    /// To query for additional properties, or if the feature is not supported,
    /// `vkGetPhysicalDeviceFormatProperties` and `vkGetPhysicalDeviceImageFormatProperties` can be
    /// used to check for supported properties of individual formats, as usual rules allow.
    ///
    /// `VK_FORMAT_R32G32_UINT`, `VK_FORMAT_R32G32_SINT`, and `VK_FORMAT_R32G32_SFLOAT` from
    /// `StorageImageExtendedFormats` SPIR-V capability, are already covered by core Vulkan
    /// See the [features chapter](https://docs.vulkan.org/spec/latest/chapters/features.html).
    pub shader_storage_image_extended_formats: bool,

    /// Specifies whether multisampled storage images are supported.
    ///
    /// If this feature is not enabled, images that are created with a usage that includes
    /// `VK_IMAGE_USAGE_STORAGE_BIT` must be created with samples equal to `VK_SAMPLE_COUNT_1_BIT`.
    ///
    /// This also specifies whether shader modules can declare the `StorageImageMultisample` and
    /// `ImageMSArray` capabilities.
    pub shader_storage_image_multisample: bool,

    /// Specifies whether storage images and storage texel buffers require a format qualifier to be
    /// specified when reading.
    ///
    /// `shaderStorageImageReadWithoutFormat` applies only to formats listed in the
    /// See the [features chapter](https://docs.vulkan.org/spec/latest/chapters/features.html).
    /// list.
    pub shader_storage_image_read_without_format: bool,

    /// Specifies whether storage images and storage texel buffers require a format qualifier to be
    /// specified when writing.
    ///
    /// `shaderStorageImageWriteWithoutFormat` applies only to formats listed in the
    /// See the [features chapter](https://docs.vulkan.org/spec/latest/chapters/features.html).
    /// list.
    pub shader_storage_image_write_without_format: bool,

    /// Specifies whether arrays of uniform buffers can be indexed by dynamically uniform integer
    /// expressions in shader code.
    ///
    /// If this feature is not enabled, resources with a descriptor type of
    /// `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` or `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC` must be
    /// indexed only by constant integral expressions when aggregated into arrays in shader code.
    ///
    /// This also specifies whether shader modules can declare the
    /// `UniformBufferArrayDynamicIndexing` capability.
    pub shader_uniform_buffer_array_dynamic_indexing: bool,

    /// Specifies whether arrays of samplers or sampled images can be indexed by dynamically
    /// uniform integer expressions in shader code.
    ///
    /// If this feature is not enabled, resources with a descriptor type of
    /// `VK_DESCRIPTOR_TYPE_SAMPLER`, `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER`, or
    /// `VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE` must be indexed only by constant integral expressions
    /// when aggregated into arrays in shader code.
    ///
    /// This also specifies whether shader modules can declare the
    /// `SampledImageArrayDynamicIndexing` capability.
    pub shader_sampled_image_array_dynamic_indexing: bool,

    /// Specifies whether arrays of storage buffers can be indexed by dynamically uniform integer
    /// expressions in shader code.
    ///
    /// If this feature is not enabled, resources with a descriptor type of
    /// `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` or `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC` must be
    /// indexed only by constant integral expressions when aggregated into arrays in shader code.
    ///
    /// This also specifies whether shader modules can declare the
    /// `StorageBufferArrayDynamicIndexing` capability.
    pub shader_storage_buffer_array_dynamic_indexing: bool,

    /// Specifies whether arrays of storage images can be indexed by dynamically uniform integer
    /// expressions in shader code.
    ///
    /// If this feature is not enabled, resources with a descriptor type of
    /// `VK_DESCRIPTOR_TYPE_STORAGE_IMAGE` must be indexed only by constant integral expressions
    /// when aggregated into arrays in shader code.
    ///
    /// This also specifies whether shader modules can declare the
    /// `StorageImageArrayDynamicIndexing` capability.
    pub shader_storage_image_array_dynamic_indexing: bool,

    /// Specifies whether clip distances are supported in shader code.
    ///
    /// If this feature is not enabled, any members decorated with the `ClipDistance` built-in
    /// decoration must not be read from or written to in shader modules.
    ///
    /// This also specifies whether shader modules can declare the `ClipDistance` capability.
    pub shader_clip_distance: bool,

    /// Specifies whether cull distances are supported in shader code.
    ///
    /// If this feature is not enabled, any members decorated with the `CullDistance` built-in
    /// decoration must not be read from or written to in shader modules.
    ///
    /// This also specifies whether shader modules can declare the `CullDistance` capability.
    pub shader_cull_distance: bool,

    /// Specifies whether 64-bit floats (doubles) are supported in shader code.
    ///
    /// If this feature is not enabled, 64-bit floating-point types must not be used in shader
    /// code.
    ///
    /// This also specifies whether shader modules can declare the `Float64` capability. Declaring
    /// and using 64-bit floats is enabled for all storage classes that SPIR-V allows with the
    /// `Float64` capability.
    pub shader_float64: bool,

    /// Specifies whether 64-bit integers (signed and unsigned) are supported in shader code.
    ///
    /// If this feature is not enabled, 64-bit integer types must not be used in shader code.
    ///
    /// This also specifies whether shader modules can declare the `Int64` capability. Declaring
    /// and using 64-bit integers is enabled for all storage classes that SPIR-V allows with
    /// the `Int64` capability.
    pub shader_int64: bool,

    /// Specifies whether 16-bit integers (signed and unsigned) are supported in shader code.
    ///
    /// If this feature is not enabled, 16-bit integer types must not be used in shader code.
    ///
    /// This also specifies whether shader modules can declare the `Int16` capability. However,
    /// this only enables a subset of the storage classes that SPIR-V allows for the `Int16`
    /// SPIR-V capability: Declaring and using 16-bit integers in the `Private`, `Workgroup`
    /// (for non-Block variables), and `Function` storage classes is enabled, while declaring
    /// them in the interface storage classes (e.g., `UniformConstant`, `Uniform`,
    /// `StorageBuffer`, `Input`, `Output`, and `PushConstant`) is not enabled.
    pub shader_int16: bool,

    /// Specifies whether image operations specifying the minimum resource LOD are supported in
    /// shader code.
    ///
    /// If this feature is not enabled, the `MinLod` image operand must not be used in shader code.
    ///
    /// This also specifies whether shader modules can declare the `MinLod` capability.
    pub shader_resource_min_lod: bool,

    /// Specifies whether all pipelines that will be bound to a command buffer during a subpass
    /// which uses no attachments must have the same value for
    /// `VkPipelineMultisampleStateCreateInfo::rasterizationSamples`.
    ///
    /// If set to `VK_TRUE`, the implementation supports variable multisample rates in a subpass
    /// which uses no attachments.
    ///
    /// If set to `VK_FALSE`, then all pipelines bound in such a subpass must have the same
    /// multisample rate.
    ///
    /// This has no effect in situations where a subpass uses any attachments.
    pub variable_multisample_rate: bool,
    // Unsupported (queries):
    // pub occlusion_query_precise: bool,
    // pub pipeline_statistics_query: bool,
    // pub inherited_queries: bool,

    // Unsupported (sparse residency):
    // pub shader_resource_residency: bool,
    // pub sparse_binding: bool,
    // pub sparse_residency_buffer: bool,
    // pub sparse_residency_image2_d: bool,
    // pub sparse_residency_image3_d: bool,
    // pub sparse_residency2_samples: bool,
    // pub sparse_residency4_samples: bool,
    // pub sparse_residency8_samples: bool,
    // pub sparse_residency16_samples: bool,
    // pub sparse_residency_aliased: bool,
}

impl From<vk::PhysicalDeviceFeatures> for Vulkan10Features {
    fn from(features: vk::PhysicalDeviceFeatures) -> Self {
        Self {
            robust_buffer_access: features.robust_buffer_access == vk::TRUE,
            full_draw_index_uint32: features.full_draw_index_uint32 == vk::TRUE,
            image_cube_array: features.image_cube_array == vk::TRUE,
            independent_blend: features.independent_blend == vk::TRUE,
            geometry_shader: features.geometry_shader == vk::TRUE,
            tessellation_shader: features.tessellation_shader == vk::TRUE,
            sample_rate_shading: features.sample_rate_shading == vk::TRUE,
            dual_src_blend: features.dual_src_blend == vk::TRUE,
            logic_op: features.logic_op == vk::TRUE,
            multi_draw_indirect: features.multi_draw_indirect == vk::TRUE,
            draw_indirect_first_instance: features.draw_indirect_first_instance == vk::TRUE,
            depth_clamp: features.depth_clamp == vk::TRUE,
            depth_bias_clamp: features.depth_bias_clamp == vk::TRUE,
            fill_mode_non_solid: features.fill_mode_non_solid == vk::TRUE,
            depth_bounds: features.depth_bounds == vk::TRUE,
            wide_lines: features.wide_lines == vk::TRUE,
            large_points: features.large_points == vk::TRUE,
            alpha_to_one: features.alpha_to_one == vk::TRUE,
            multi_viewport: features.multi_viewport == vk::TRUE,
            sampler_anisotropy: features.sampler_anisotropy == vk::TRUE,
            texture_compression_etc2: features.texture_compression_etc2 == vk::TRUE,
            texture_compression_astc_ldr: features.texture_compression_astc_ldr == vk::TRUE,
            texture_compression_bc: features.texture_compression_bc == vk::TRUE,
            vertex_pipeline_stores_and_atomics: features.vertex_pipeline_stores_and_atomics
                == vk::TRUE,
            fragment_stores_and_atomics: features.fragment_stores_and_atomics == vk::TRUE,
            shader_tessellation_and_geometry_point_size: features
                .shader_tessellation_and_geometry_point_size
                == vk::TRUE,
            shader_image_gather_extended: features.shader_image_gather_extended == vk::TRUE,
            shader_storage_image_extended_formats: features.shader_storage_image_extended_formats
                == vk::TRUE,
            shader_storage_image_multisample: features.shader_storage_image_multisample == vk::TRUE,
            shader_storage_image_read_without_format: features
                .shader_storage_image_read_without_format
                == vk::TRUE,
            shader_storage_image_write_without_format: features
                .shader_storage_image_write_without_format
                == vk::TRUE,
            shader_uniform_buffer_array_dynamic_indexing: features
                .shader_uniform_buffer_array_dynamic_indexing
                == vk::TRUE,
            shader_sampled_image_array_dynamic_indexing: features
                .shader_sampled_image_array_dynamic_indexing
                == vk::TRUE,
            shader_storage_buffer_array_dynamic_indexing: features
                .shader_storage_buffer_array_dynamic_indexing
                == vk::TRUE,
            shader_storage_image_array_dynamic_indexing: features
                .shader_storage_image_array_dynamic_indexing
                == vk::TRUE,
            shader_clip_distance: features.shader_clip_distance == vk::TRUE,
            shader_cull_distance: features.shader_cull_distance == vk::TRUE,
            shader_float64: features.shader_float64 == vk::TRUE,
            shader_int64: features.shader_int64 == vk::TRUE,
            shader_int16: features.shader_int16 == vk::TRUE,
            shader_resource_min_lod: features.shader_resource_min_lod == vk::TRUE,
            variable_multisample_rate: features.variable_multisample_rate == vk::TRUE,
        }
    }
}

/// Description of Vulkan limitations.
///
/// See [`VkPhysicalDeviceVulkan11Features`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceVulkan11Features.html).
#[derive(Clone, Copy, Debug)]
pub struct Vulkan10Limits {
    /// The largest dimension (width) that is guaranteed to be supported for all images created
    /// with an image type of [`vk::ImageType::TYPE_1D`].
    ///
    /// Some combinations of image parameters (format, usage, etc.) may allow support for larger
    /// dimensions, which can be queried using
    /// [`PhysicalDevice::image_format_properties`].
    pub max_image_dimension1_d: u32,

    /// The largest dimension (width or height) that is guaranteed to be supported for all images
    /// created with an image type of [`vk::ImageType::TYPE_2D`] and without
    /// [`vk::ImageCreateFlags::CUBE_COMPATIBLE`] set in
    /// [`ImageInfo::flags`](super::image::ImageInfo::flags).
    ///
    /// Some combinations of image parameters (format, usage, etc.) may allow support for larger
    /// dimensions, which can be queried using
    /// [`PhysicalDevice::image_format_properties`].
    pub max_image_dimension2_d: u32,

    /// The largest dimension (width, height, or depth) that is guaranteed to be supported for all
    /// images created with an image type of [`vk::ImageType::TYPE_3D`].
    ///
    /// Some combinations of image parameters (format, usage, etc.) may allow support for larger
    /// dimensions, which can be queried using
    /// [`PhysicalDevice::image_format_properties`].
    pub max_image_dimension3_d: u32,

    /// The largest dimension (width or height) that is guaranteed to be supported for all images
    /// created with an image type of [`vk::ImageType::TYPE_2D`] and with
    /// [`vk::ImageCreateFlags::CUBE_COMPATIBLE`] set in
    /// [`ImageInfo::flags`](super::image::ImageInfo::flags).
    ///
    /// Some combinations of image parameters (format, usage, etc.) may allow support for larger
    /// dimensions, which can be queried using
    /// [`PhysicalDevice::image_format_properties`].
    pub max_image_dimension_cube: u32,

    /// The maximum number of layers
    /// ([`ImageInfo::array_layer_count`](super::image::ImageInfo::array_layer_count)) for an
    /// image.
    pub max_image_array_layers: u32,

    /// The maximum number of addressable texels for a buffer view created on a buffer which was
    /// created with the [`vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER`] or
    /// [`vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER`] set in
    /// [`BufferInfo::usage`](super::buffer::BufferInfo::usage).
    pub max_texel_buffer_elements: u32,

    /// The maximum range, in bytes, for a uniform buffer binding.
    pub max_uniform_buffer_range: u32,

    /// The maximum range, in bytes, for a storage buffer binding.
    pub max_storage_buffer_range: u32,

    /// The maximum size, in bytes, of the push constant storage available to a pipeline layout.
    pub max_push_constants_size: u32,

    /// The maximum number of simultaneously active device memory allocations.
    pub max_memory_allocation_count: u32,

    /// The maximum number of sampler objects that may exist at one time.
    pub max_sampler_allocation_count: u32,

    /// The required granularity, in bytes, for aliasing linear and optimal resources in memory.
    pub buffer_image_granularity: vk::DeviceSize,

    /// The maximum size, in bytes, of the address space available for sparse resources.
    pub sparse_address_space_size: vk::DeviceSize,

    /// The maximum number of descriptor sets that can be bound simultaneously.
    pub max_bound_descriptor_sets: u32,

    /// The maximum number of sampler descriptors accessible from a single shader stage.
    pub max_per_stage_descriptor_samplers: u32,

    /// The maximum number of uniform buffer descriptors accessible from a single shader stage.
    pub max_per_stage_descriptor_uniform_buffers: u32,

    /// The maximum number of storage buffer descriptors accessible from a single shader stage.
    pub max_per_stage_descriptor_storage_buffers: u32,

    /// The maximum number of sampled image descriptors accessible from a single shader stage.
    pub max_per_stage_descriptor_sampled_images: u32,

    /// The maximum number of storage image descriptors accessible from a single shader stage.
    pub max_per_stage_descriptor_storage_images: u32,

    /// The maximum number of input attachment descriptors accessible from a single shader stage.
    pub max_per_stage_descriptor_input_attachments: u32,

    /// The maximum total number of resources accessible from a single shader stage.
    pub max_per_stage_resources: u32,

    /// The maximum number of sampler descriptors available across a single descriptor set.
    pub max_descriptor_set_samplers: u32,

    /// The maximum number of uniform buffer descriptors available across a single descriptor set.
    pub max_descriptor_set_uniform_buffers: u32,

    /// The maximum number of dynamic uniform buffer descriptors available across a descriptor set.
    pub max_descriptor_set_uniform_buffers_dynamic: u32,

    /// The maximum number of storage buffer descriptors available across a single descriptor set.
    pub max_descriptor_set_storage_buffers: u32,

    /// The maximum number of dynamic storage buffer descriptors available across a descriptor set.
    pub max_descriptor_set_storage_buffers_dynamic: u32,

    /// The maximum number of sampled image descriptors available across a single descriptor set.
    pub max_descriptor_set_sampled_images: u32,

    /// The maximum number of storage image descriptors available across a single descriptor set.
    pub max_descriptor_set_storage_images: u32,

    /// The maximum number of input attachment descriptors available across a descriptor set.
    pub max_descriptor_set_input_attachments: u32,

    /// The maximum number of vertex input attributes for a graphics pipeline.
    pub max_vertex_input_attributes: u32,

    /// The maximum number of vertex input bindings for a graphics pipeline.
    pub max_vertex_input_bindings: u32,

    /// The maximum offset, in bytes, allowed for a vertex input attribute.
    pub max_vertex_input_attribute_offset: u32,

    /// The maximum stride, in bytes, for a vertex input binding.
    pub max_vertex_input_binding_stride: u32,

    /// The maximum number of components output by the vertex stage.
    pub max_vertex_output_components: u32,

    /// The maximum tessellation generation level.
    pub max_tessellation_generation_level: u32,

    /// The maximum number of control points in a tessellation patch.
    pub max_tessellation_patch_size: u32,

    /// The maximum number of per-vertex input components to tessellation control shaders.
    pub max_tessellation_control_per_vertex_input_components: u32,

    /// The maximum number of per-vertex output components from tessellation control shaders.
    pub max_tessellation_control_per_vertex_output_components: u32,

    /// The maximum number of per-patch output components from tessellation control shaders.
    pub max_tessellation_control_per_patch_output_components: u32,

    /// The maximum total number of output components from tessellation control shaders.
    pub max_tessellation_control_total_output_components: u32,

    /// The maximum number of input components to tessellation evaluation shaders.
    pub max_tessellation_evaluation_input_components: u32,

    /// The maximum number of output components from tessellation evaluation shaders.
    pub max_tessellation_evaluation_output_components: u32,

    /// The maximum number of geometry shader invocations per primitive.
    pub max_geometry_shader_invocations: u32,

    /// The maximum number of input components to geometry shaders.
    pub max_geometry_input_components: u32,

    /// The maximum number of output components from geometry shaders.
    pub max_geometry_output_components: u32,

    /// The maximum number of vertices a geometry shader may emit.
    pub max_geometry_output_vertices: u32,

    /// The maximum total number of output components from a geometry shader invocation.
    pub max_geometry_total_output_components: u32,

    /// The maximum number of input components to fragment shaders.
    pub max_fragment_input_components: u32,

    /// The maximum number of color attachments written by a fragment shader.
    pub max_fragment_output_attachments: u32,

    /// The maximum number of dual-source attachments written by a fragment shader.
    pub max_fragment_dual_src_attachments: u32,

    /// The maximum combined number of fragment shader output resources.
    pub max_fragment_combined_output_resources: u32,

    /// The maximum amount of shared memory, in bytes, available to a compute workgroup.
    pub max_compute_shared_memory_size: u32,

    /// The maximum workgroup counts for compute dispatch in each dimension.
    pub max_compute_work_group_count: [u32; 3],

    /// The maximum total number of invocations in a single compute workgroup.
    pub max_compute_work_group_invocations: u32,

    /// The maximum workgroup size for compute dispatch in each dimension.
    pub max_compute_work_group_size: [u32; 3],

    /// The number of bits of sub-pixel precision in framebuffer coordinates.
    pub sub_pixel_precision_bits: u32,

    /// The number of bits of sub-texel precision for image sampling.
    pub sub_texel_precision_bits: u32,

    /// The number of bits of precision available for mipmap level selection.
    pub mipmap_precision_bits: u32,

    /// The maximum value of an index used with indexed drawing.
    pub max_draw_indexed_index_value: u32,

    /// The maximum number of draws executed by an indirect draw-count command.
    pub max_draw_indirect_count: u32,

    /// The maximum absolute value of sampler LOD bias.
    pub max_sampler_lod_bias: f32,

    /// The maximum sampler anisotropy level.
    pub max_sampler_anisotropy: f32,

    /// The maximum number of simultaneously active viewports.
    pub max_viewports: u32,

    /// The maximum viewport width and height.
    pub max_viewport_dimensions: [u32; 2],

    /// The minimum and maximum supported viewport bounds.
    pub viewport_bounds_range: [f32; 2],

    /// The number of bits of precision for viewport sub-pixel coordinates.
    pub viewport_sub_pixel_bits: u32,

    /// The minimum alignment, in bytes, of host pointer values passed to mapping functions.
    pub min_memory_map_alignment: usize,

    /// The minimum alignment, in bytes, for texel buffer offsets.
    pub min_texel_buffer_offset_alignment: vk::DeviceSize,

    /// The minimum alignment, in bytes, for uniform buffer offsets.
    pub min_uniform_buffer_offset_alignment: vk::DeviceSize,

    /// The minimum alignment, in bytes, for storage buffer offsets.
    pub min_storage_buffer_offset_alignment: vk::DeviceSize,

    /// The minimum offset allowed for texel fetch operations.
    pub min_texel_offset: i32,

    /// The maximum offset allowed for texel fetch operations.
    pub max_texel_offset: u32,

    /// The minimum offset allowed for texel gather operations.
    pub min_texel_gather_offset: i32,

    /// The maximum offset allowed for texel gather operations.
    pub max_texel_gather_offset: u32,

    /// The minimum interpolation offset supported by the implementation.
    pub min_interpolation_offset: f32,

    /// The maximum interpolation offset supported by the implementation.
    pub max_interpolation_offset: f32,

    /// The number of bits of sub-pixel precision for interpolation offsets.
    pub sub_pixel_interpolation_offset_bits: u32,

    /// The maximum framebuffer width.
    pub max_framebuffer_width: u32,

    /// The maximum framebuffer height.
    pub max_framebuffer_height: u32,

    /// The maximum number of framebuffer layers.
    pub max_framebuffer_layers: u32,

    /// The sample counts supported for color framebuffer attachments.
    pub framebuffer_color_sample_counts: vk::SampleCountFlags,

    /// The sample counts supported for depth framebuffer attachments.
    pub framebuffer_depth_sample_counts: vk::SampleCountFlags,

    /// The sample counts supported for stencil framebuffer attachments.
    pub framebuffer_stencil_sample_counts: vk::SampleCountFlags,

    /// The sample counts supported for framebuffers with no attachments.
    pub framebuffer_no_attachments_sample_counts: vk::SampleCountFlags,

    /// The maximum number of color attachments in a framebuffer.
    pub max_color_attachments: u32,

    /// The sample counts supported for sampled color images.
    pub sampled_image_color_sample_counts: vk::SampleCountFlags,

    /// The sample counts supported for sampled integer images.
    pub sampled_image_integer_sample_counts: vk::SampleCountFlags,

    /// The sample counts supported for sampled depth images.
    pub sampled_image_depth_sample_counts: vk::SampleCountFlags,

    /// The sample counts supported for sampled stencil images.
    pub sampled_image_stencil_sample_counts: vk::SampleCountFlags,

    /// The sample counts supported for storage images.
    pub storage_image_sample_counts: vk::SampleCountFlags,

    /// The maximum number of words in a sample mask.
    pub max_sample_mask_words: u32,

    /// True if timestamps are supported in both graphics and compute queues.
    pub timestamp_compute_and_graphics: bool,

    /// The number of nanoseconds per timestamp increment.
    pub timestamp_period: f32,

    /// The maximum number of clip distances.
    pub max_clip_distances: u32,

    /// The maximum number of cull distances.
    pub max_cull_distances: u32,

    /// The maximum combined number of clip and cull distances.
    pub max_combined_clip_and_cull_distances: u32,

    /// The number of discrete queue priorities supported by the implementation.
    pub discrete_queue_priorities: u32,

    /// The supported range of point sizes.
    pub point_size_range: [f32; 2],

    /// The supported range of line widths.
    pub line_width_range: [f32; 2],

    /// The granularity of point-size values.
    pub point_size_granularity: f32,

    /// The granularity of line-width values.
    pub line_width_granularity: f32,

    /// True if line rasterization follows strict diamond-exit rules.
    pub strict_lines: bool,

    /// True if standard sample locations are supported.
    pub standard_sample_locations: bool,

    /// The optimal alignment, in bytes, for buffer copy source and destination offsets.
    pub optimal_buffer_copy_offset_alignment: vk::DeviceSize,

    /// The optimal alignment, in bytes, for buffer copy row pitch values.
    pub optimal_buffer_copy_row_pitch_alignment: vk::DeviceSize,

    /// The non-coherent atom size, in bytes, used for host cache management.
    pub non_coherent_atom_size: vk::DeviceSize,
}

impl From<vk::PhysicalDeviceLimits> for Vulkan10Limits {
    fn from(limits: vk::PhysicalDeviceLimits) -> Self {
        Self {
            max_image_dimension1_d: limits.max_image_dimension1_d,
            max_image_dimension2_d: limits.max_image_dimension2_d,
            max_image_dimension3_d: limits.max_image_dimension3_d,
            max_image_dimension_cube: limits.max_image_dimension_cube,
            max_image_array_layers: limits.max_image_array_layers,
            max_texel_buffer_elements: limits.max_texel_buffer_elements,
            max_uniform_buffer_range: limits.max_uniform_buffer_range,
            max_storage_buffer_range: limits.max_storage_buffer_range,
            max_push_constants_size: limits.max_push_constants_size,
            max_memory_allocation_count: limits.max_memory_allocation_count,
            max_sampler_allocation_count: limits.max_sampler_allocation_count,
            buffer_image_granularity: limits.buffer_image_granularity,
            sparse_address_space_size: limits.sparse_address_space_size,
            max_bound_descriptor_sets: limits.max_bound_descriptor_sets,
            max_per_stage_descriptor_samplers: limits.max_per_stage_descriptor_samplers,
            max_per_stage_descriptor_uniform_buffers: limits
                .max_per_stage_descriptor_uniform_buffers,
            max_per_stage_descriptor_storage_buffers: limits
                .max_per_stage_descriptor_storage_buffers,
            max_per_stage_descriptor_sampled_images: limits.max_per_stage_descriptor_sampled_images,
            max_per_stage_descriptor_storage_images: limits.max_per_stage_descriptor_storage_images,
            max_per_stage_descriptor_input_attachments: limits
                .max_per_stage_descriptor_input_attachments,
            max_per_stage_resources: limits.max_per_stage_resources,
            max_descriptor_set_samplers: limits.max_descriptor_set_samplers,
            max_descriptor_set_uniform_buffers: limits.max_descriptor_set_uniform_buffers,
            max_descriptor_set_uniform_buffers_dynamic: limits
                .max_descriptor_set_uniform_buffers_dynamic,
            max_descriptor_set_storage_buffers: limits.max_descriptor_set_storage_buffers,
            max_descriptor_set_storage_buffers_dynamic: limits
                .max_descriptor_set_storage_buffers_dynamic,
            max_descriptor_set_sampled_images: limits.max_descriptor_set_sampled_images,
            max_descriptor_set_storage_images: limits.max_descriptor_set_storage_images,
            max_descriptor_set_input_attachments: limits.max_descriptor_set_input_attachments,
            max_vertex_input_attributes: limits.max_vertex_input_attributes,
            max_vertex_input_bindings: limits.max_vertex_input_bindings,
            max_vertex_input_attribute_offset: limits.max_vertex_input_attribute_offset,
            max_vertex_input_binding_stride: limits.max_vertex_input_binding_stride,
            max_vertex_output_components: limits.max_vertex_output_components,
            max_tessellation_generation_level: limits.max_tessellation_generation_level,
            max_tessellation_patch_size: limits.max_tessellation_patch_size,
            max_tessellation_control_per_vertex_input_components: limits
                .max_tessellation_control_per_vertex_input_components,
            max_tessellation_control_per_vertex_output_components: limits
                .max_tessellation_control_per_vertex_output_components,
            max_tessellation_control_per_patch_output_components: limits
                .max_tessellation_control_per_patch_output_components,
            max_tessellation_control_total_output_components: limits
                .max_tessellation_control_total_output_components,
            max_tessellation_evaluation_input_components: limits
                .max_tessellation_evaluation_input_components,
            max_tessellation_evaluation_output_components: limits
                .max_tessellation_evaluation_output_components,
            max_geometry_shader_invocations: limits.max_geometry_shader_invocations,
            max_geometry_input_components: limits.max_geometry_input_components,
            max_geometry_output_components: limits.max_geometry_output_components,
            max_geometry_output_vertices: limits.max_geometry_output_vertices,
            max_geometry_total_output_components: limits.max_geometry_total_output_components,
            max_fragment_input_components: limits.max_fragment_input_components,
            max_fragment_output_attachments: limits.max_fragment_output_attachments,
            max_fragment_dual_src_attachments: limits.max_fragment_dual_src_attachments,
            max_fragment_combined_output_resources: limits.max_fragment_combined_output_resources,
            max_compute_shared_memory_size: limits.max_compute_shared_memory_size,
            max_compute_work_group_count: limits.max_compute_work_group_count,
            max_compute_work_group_invocations: limits.max_compute_work_group_invocations,
            max_compute_work_group_size: limits.max_compute_work_group_size,
            sub_pixel_precision_bits: limits.sub_pixel_precision_bits,
            sub_texel_precision_bits: limits.sub_texel_precision_bits,
            mipmap_precision_bits: limits.mipmap_precision_bits,
            max_draw_indexed_index_value: limits.max_draw_indexed_index_value,
            max_draw_indirect_count: limits.max_draw_indirect_count,
            max_sampler_lod_bias: limits.max_sampler_lod_bias,
            max_sampler_anisotropy: limits.max_sampler_anisotropy,
            max_viewports: limits.max_viewports,
            max_viewport_dimensions: limits.max_viewport_dimensions,
            viewport_bounds_range: limits.viewport_bounds_range,
            viewport_sub_pixel_bits: limits.viewport_sub_pixel_bits,
            min_memory_map_alignment: limits.min_memory_map_alignment,
            min_texel_buffer_offset_alignment: limits.min_texel_buffer_offset_alignment,
            min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment,
            min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
            min_texel_offset: limits.min_texel_offset,
            max_texel_offset: limits.max_texel_offset,
            min_texel_gather_offset: limits.min_texel_gather_offset,
            max_texel_gather_offset: limits.max_texel_gather_offset,
            min_interpolation_offset: limits.min_interpolation_offset,
            max_interpolation_offset: limits.max_interpolation_offset,
            sub_pixel_interpolation_offset_bits: limits.sub_pixel_interpolation_offset_bits,
            max_framebuffer_width: limits.max_framebuffer_width,
            max_framebuffer_height: limits.max_framebuffer_height,
            max_framebuffer_layers: limits.max_framebuffer_layers,
            framebuffer_color_sample_counts: limits.framebuffer_color_sample_counts,
            framebuffer_depth_sample_counts: limits.framebuffer_depth_sample_counts,
            framebuffer_stencil_sample_counts: limits.framebuffer_stencil_sample_counts,
            framebuffer_no_attachments_sample_counts: limits
                .framebuffer_no_attachments_sample_counts,
            max_color_attachments: limits.max_color_attachments,
            sampled_image_color_sample_counts: limits.sampled_image_color_sample_counts,
            sampled_image_integer_sample_counts: limits.sampled_image_integer_sample_counts,
            sampled_image_depth_sample_counts: limits.sampled_image_depth_sample_counts,
            sampled_image_stencil_sample_counts: limits.sampled_image_stencil_sample_counts,
            storage_image_sample_counts: limits.storage_image_sample_counts,
            max_sample_mask_words: limits.max_sample_mask_words,
            timestamp_compute_and_graphics: limits.timestamp_compute_and_graphics == vk::TRUE,
            timestamp_period: limits.timestamp_period,
            max_clip_distances: limits.max_clip_distances,
            max_cull_distances: limits.max_cull_distances,
            max_combined_clip_and_cull_distances: limits.max_combined_clip_and_cull_distances,
            discrete_queue_priorities: limits.discrete_queue_priorities,
            point_size_range: limits.point_size_range,
            line_width_range: limits.line_width_range,
            point_size_granularity: limits.point_size_granularity,
            line_width_granularity: limits.line_width_granularity,
            strict_lines: limits.strict_lines == vk::TRUE,
            standard_sample_locations: limits.standard_sample_locations == vk::TRUE,
            optimal_buffer_copy_offset_alignment: limits.optimal_buffer_copy_offset_alignment,
            optimal_buffer_copy_row_pitch_alignment: limits.optimal_buffer_copy_row_pitch_alignment,
            non_coherent_atom_size: limits.non_coherent_atom_size,
        }
    }
}

/// Description of Vulkan 1.0 properties.
///
/// See [`VkPhysicalDeviceVulkan11Properties`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceVulkan11Properties.html).
#[derive(Clone, Debug)]
pub struct Vulkan10Properties {
    /// The version of Vulkan supported by the device, encoded as described
    /// See the [limits chapter](https://docs.vulkan.org/spec/latest/chapters/limits.html).
    pub api_version: u32,

    /// The vendor-specified version of the driver.
    pub driver_version: u32,

    /// A unique identifier for the vendor (see
    /// See the [limits chapter](https://docs.vulkan.org/spec/latest/chapters/limits.html).
    /// of the physical device.
    pub vendor_id: u32,

    /// A unique identifier for the physical device among devices available from the vendor.
    pub device_id: u32,

    /// a
    /// See the [limits chapter](https://docs.vulkan.org/spec/latest/chapters/limits.html).
    /// specifying the type of device.
    pub device_type: vk::PhysicalDeviceType,

    /// A UTF-8 string which is the name of the device.
    pub device_name: String,

    /// An array of VK_UUID_SIZE `u8` values representing a universally unique identifier for the
    /// device.
    pub pipeline_cache_uuid: [u8; vk::UUID_SIZE],

    /// The [`Vulkan10Limits`] structure specifying device-specific limits of the physical device.
    /// See
    /// [Limits](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#limits)
    /// for details.
    pub limits: Vulkan10Limits,
    // Unsupported (sparse residency):
    // pub sparse_properties: vk::PhysicalDeviceSparseProperties,
}

impl From<vk::PhysicalDeviceProperties> for Vulkan10Properties {
    fn from(properties: vk::PhysicalDeviceProperties) -> Self {
        Self {
            api_version: properties.api_version,
            driver_version: properties.driver_version,
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            device_type: properties.device_type,
            device_name: vk_cstr_to_string_lossy(&properties.device_name),
            pipeline_cache_uuid: properties.pipeline_cache_uuid,
            limits: properties.limits.into(),
        }
    }
}

/// Description of Vulkan 1.1 features.
///
/// See [`VkPhysicalDeviceVulkan12Features`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceVulkan12Features.html).
#[derive(Clone, Copy, Debug)]
pub struct Vulkan11Features {
    /// Specifies whether objects in the StorageBuffer, ShaderRecordBufferKHR, or
    /// PhysicalStorageBuffer storage class with the Block decoration can have 16-bit integer and
    /// 16-bit floating-point members.
    ///
    /// If this feature is not enabled, 16-bit integer or 16-bit floating-point members must not be
    /// used in such objects. This also specifies whether shader modules can declare the
    /// StorageBuffer16BitAccess capability.
    pub storage_buffer16_bit_access: bool,

    /// Specifies whether objects in the Uniform storage class with the Block decoration can have
    /// 16-bit integer and 16-bit floating-point members.
    ///
    /// If this feature is not enabled, 16-bit integer or 16-bit floating-point members must not be
    /// used in such objects. This also specifies whether shader modules can declare the
    /// UniformAndStorageBuffer16BitAccess capability.
    pub uniform_and_storage_buffer16_bit_access: bool,

    /// Specifies whether objects in the PushConstant storage class can have 16-bit integer and
    /// 16-bit floating-point members.
    ///
    /// If this feature is not enabled, 16-bit integer or floating-point members must not be used
    /// in such objects. This also specifies whether shader modules can declare the
    /// StoragePushConstant16 capability.
    pub storage_push_constant16: bool,

    /// Specifies whether objects in the Input and Output storage classes can have 16-bit integer
    /// and 16-bit floating-point members.
    ///
    /// If this feature is not enabled, 16-bit integer or 16-bit floating-point members must not be
    /// used in such objects. This also specifies whether shader modules can declare the
    /// StorageInputOutput16 capability.
    pub storage_input_output16: bool,

    /// Specifies whether the implementation supports multiview rendering within a render pass.
    ///
    /// If this feature is not enabled, the view mask of each subpass must always be zero.
    pub multiview: bool,

    /// Specifies whether the implementation supports multiview rendering within a render pass,
    /// with geometry shaders.
    ///
    /// If this feature is not enabled, then a pipeline compiled against a subpass with a non-zero
    /// view mask must not include a geometry shader.
    pub multiview_geometry_shader: bool,

    /// Specifies whether the implementation supports multiview rendering within a render pass,
    /// with tessellation shaders.
    ///
    /// If this feature is not enabled, then a pipeline compiled against a subpass with a non-zero
    /// view mask must not include any tessellation shaders.
    pub multiview_tessellation_shader: bool,

    /// Specifies whether the implementation supports the SPIR-V VariablePointersStorageBuffer
    /// capability.
    ///
    /// When this feature is not enabled, shader modules must not declare the
    /// SPV_KHR_variable_pointers extension or the VariablePointersStorageBuffer capability.
    pub variable_pointers_storage_buffer: bool,

    /// Specifies whether the implementation supports the SPIR-V VariablePointers capability.
    ///
    /// When this feature is not enabled, shader modules must not declare the VariablePointers
    /// capability.
    pub variable_pointers: bool,

    /// Specifies whether protected memory is supported.
    pub protected_memory: bool,

    /// Specifies whether the implementation supports sampler Y′CBCR conversion.
    ///
    /// If `sampler_ycbcr_conversion` is `false`, sampler Y′CBCR conversion is not supported, and
    /// samplers using sampler Y′CBCR conversion must not be used.
    pub sampler_ycbcr_conversion: bool,

    /// Specifies whether the implementation supports the SPIR-V DrawParameters capability.
    ///
    /// When this feature is not enabled, shader modules must not declare the
    /// SPV_KHR_shader_draw_parameters extension or the DrawParameters capability.
    pub shader_draw_parameters: bool,
}

impl From<vk::PhysicalDeviceVulkan11Features<'_>> for Vulkan11Features {
    fn from(features: vk::PhysicalDeviceVulkan11Features<'_>) -> Self {
        Self {
            storage_buffer16_bit_access: features.storage_buffer16_bit_access == vk::TRUE,
            uniform_and_storage_buffer16_bit_access: features
                .uniform_and_storage_buffer16_bit_access
                == vk::TRUE,
            storage_push_constant16: features.storage_push_constant16 == vk::TRUE,
            storage_input_output16: features.storage_input_output16 == vk::TRUE,
            multiview: features.multiview == vk::TRUE,
            multiview_geometry_shader: features.multiview_geometry_shader == vk::TRUE,
            multiview_tessellation_shader: features.multiview_tessellation_shader == vk::TRUE,
            variable_pointers_storage_buffer: features.variable_pointers_storage_buffer == vk::TRUE,
            variable_pointers: features.variable_pointers == vk::TRUE,
            protected_memory: features.protected_memory == vk::TRUE,
            sampler_ycbcr_conversion: features.sampler_ycbcr_conversion == vk::TRUE,
            shader_draw_parameters: features.shader_draw_parameters == vk::TRUE,
        }
    }
}

/// Description of Vulkan 1.1 properties.
///
/// See [`VkPhysicalDeviceVulkan12Properties`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceVulkan12Properties.html).
#[derive(Clone, Copy, Debug)]
pub struct Vulkan11Properties {
    /// An array of `VK_UUID_SIZE` `u8` values representing a universally unique identifier for
    /// the device
    pub device_uuid: [u8; vk::UUID_SIZE],

    /// An array of `VK_UUID_SIZE` `u8` values representing a universally unique identifier for the
    /// driver build in use by the device.
    pub driver_uuid: [u8; vk::UUID_SIZE],

    /// An array of `VK_LUID_SIZE` `u8` values representing a locally unique identifier for the
    /// device
    pub device_luid: [u8; vk::LUID_SIZE],

    /// A `u32` bitfield identifying the node within a linked device adapter corresponding to the
    /// device.
    pub device_node_mask: u32,

    /// A `bool` value that will be `true` if `device_luid` contains a valid LUID and
    /// `device_node_mask` contains a valid node mask, and `false` if they do not.
    pub device_luid_valid: bool,

    /// The default number of invocations in each subgroup. `subgroup_size` is at least `1` if any
    /// of the physical device’s queues support `VK_QUEUE_GRAPHICS_BIT` or `VK_QUEUE_COMPUTE_BIT`.
    /// `subgroup_size` is a power-of-two.
    pub subgroup_size: u32,

    /// A bitfield of `vk::ShaderStageFlagBits` describing the shader stages that group operations
    /// with subgroup scope are supported in. `subgroup_supported_stages` will have the
    /// `VK_SHADER_STAGE_COMPUTE_BIT` bit set if any of the physical device’s queues support
    /// `VK_QUEUE_COMPUTE_BIT`.
    pub subgroup_supported_stages: vk::ShaderStageFlags,

    /// A bitmask of `vk::SubgroupFeatureFlagBits` specifying the sets of group operations with
    /// subgroup scope supported on this device. `subgroup_supported_operations` will have the
    /// `VK_SUBGROUP_FEATURE_BASIC_BIT` bit set if any of the physical device’s queues support
    /// `VK_QUEUE_GRAPHICS_BIT` or `VK_QUEUE_COMPUTE_BIT`.
    pub subgroup_supported_operations: vk::SubgroupFeatureFlags,

    /// A `bool` specifying whether quad group operations are available in all stages, or are
    /// restricted to fragment and compute stages.
    pub subgroup_quad_operations_in_all_stages: bool,

    /// A `vk::PointClippingBehavior` value specifying the point clipping behavior supported by the
    /// implementation.
    pub point_clipping_behavior: vk::PointClippingBehavior,

    /// `max_multiview_view_count` is one greater than the maximum view index that can be used in a
    /// subpass.
    pub max_multiview_view_count: u32,

    /// The maximum valid value of instance index allowed to be generated by a drawing command
    /// recorded within a subpass of a multiview render pass instance.
    pub max_multiview_instance_index: u32,

    /// Specifies how an implementation behaves when an application attempts to write to
    /// unprotected memory in a protected queue operation, read from protected memory in an
    /// unprotected queue operation, or perform a query in a protected queue operation.
    ///
    /// If this limit is `true`, such writes will be discarded or have undefined values written,
    /// reads and queries will return undefined values.
    ///
    /// If this limit is `false`, applications must not perform these operations.
    ///
    /// See the [features chapter](https://docs.vulkan.org/spec/latest/chapters/features.html).
    /// for more information.
    pub protected_no_fault: bool,

    /// A maximum number of descriptors (summed over all descriptor types) in a single descriptor
    /// set that is guaranteed to satisfy any implementation-dependent constraints on the size of a
    /// descriptor set itself.
    ///
    /// Applications can query whether a descriptor set that goes beyond this limit is supported
    /// using `vkGetDescriptorSetLayoutSupport`.
    pub max_per_set_descriptors: u32,

    /// The maximum size of a memory allocation that can be created, even if there is more space
    /// available in the heap.
    pub max_memory_allocation_size: vk::DeviceSize,
}

impl From<vk::PhysicalDeviceVulkan11Properties<'_>> for Vulkan11Properties {
    fn from(props: vk::PhysicalDeviceVulkan11Properties<'_>) -> Self {
        Self {
            device_uuid: props.device_uuid,
            driver_uuid: props.driver_uuid,
            device_luid: props.device_luid,
            device_node_mask: props.device_node_mask,
            device_luid_valid: props.device_luid_valid == vk::TRUE,
            subgroup_size: props.subgroup_size,
            subgroup_supported_stages: props.subgroup_supported_stages,
            subgroup_supported_operations: props.subgroup_supported_operations,
            subgroup_quad_operations_in_all_stages: props.subgroup_quad_operations_in_all_stages
                == vk::TRUE,
            point_clipping_behavior: props.point_clipping_behavior,
            max_multiview_view_count: props.max_multiview_view_count,
            max_multiview_instance_index: props.max_multiview_instance_index,
            protected_no_fault: props.protected_no_fault == vk::TRUE,
            max_per_set_descriptors: props.max_per_set_descriptors,
            max_memory_allocation_size: props.max_memory_allocation_size,
        }
    }
}

/// Description of Vulkan 1.2 features.
///
/// See [`VkPhysicalDeviceVulkan12Features`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceVulkan12Features.html).
#[derive(Clone, Copy, Debug)]
pub struct Vulkan12Features {
    /// Indicates whether the implementation supports the
    /// `VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE` sampler address mode.
    ///
    /// If this feature is not enabled, the `VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE` sampler
    /// address mode must not be used.
    pub sampler_mirror_clamp_to_edge: bool,

    /// Indicates whether the implementation supports the vkCmdDrawIndirectCount and
    /// vkCmdDrawIndexedIndirectCount functions.
    ///
    /// If this feature is not enabled, these functions must not be used.
    pub draw_indirect_count: bool,

    /// Indicates whether objects in the StorageBuffer, ShaderRecordBufferKHR, or
    /// PhysicalStorageBuffer storage class with the Block decoration can have 8-bit integer
    /// members.
    ///
    /// If this feature is not enabled, 8-bit integer members must not be used in such objects.
    /// This also indicates whether shader modules can declare the StorageBuffer8BitAccess
    /// capability.
    pub storage_buffer8_bit_access: bool,

    /// Indicates whether objects in the Uniform storage class with the Block decoration can have
    /// 8-bit integer members.
    ///
    /// If this feature is not enabled, 8-bit integer members must not be used in such objects.
    /// This also indicates whether shader modules can declare the
    /// UniformAndStorageBuffer8BitAccess capability.
    pub uniform_and_storage_buffer8_bit_access: bool,

    /// Indicates whether objects in the PushConstant storage class can have 8-bit integer members.
    ///
    /// If this feature is not enabled, 8-bit integer members must not be used in such objects.
    /// This also indicates whether shader modules can declare the StoragePushConstant8
    /// capability.
    pub storage_push_constant8: bool,

    /// Indicates whether shaders can perform 64-bit unsigned and signed integer atomic operations
    /// on buffers.
    pub shader_buffer_int64_atomics: bool,

    /// Indicates whether shaders can perform 64-bit unsigned and signed integer atomic operations
    /// on shared and payload memory.
    pub shader_shared_int64_atomics: bool,

    /// Indicates whether 16-bit floats (halfs) are supported in shader code.
    ///
    /// This also indicates whether shader modules can declare the Float16 capability. However,
    /// this only enables a subset of the storage classes that SPIR-V allows for the Float16
    /// SPIR-V capability: Declaring and using 16-bit floats in the Private, Workgroup (for
    /// non-Block variables), and Function storage classes is enabled, while declaring them in
    /// the interface storage classes (e.g., UniformConstant, Uniform, StorageBuffer, Input,
    /// Output, and PushConstant) is not enabled.
    pub shader_float16: bool,

    /// Indicates whether 8-bit integers (signed and unsigned) are supported in shader code.
    ///
    /// This also indicates whether shader modules can declare the Int8 capability. However, this
    /// only enables a subset of the storage classes that SPIR-V allows for the Int8 SPIR-V
    /// capability: Declaring and using 8-bit integers in the Private, Workgroup (for non-Block
    /// variables), and Function storage classes is enabled, while declaring them in the interface
    /// storage classes (e.g., UniformConstant, Uniform, StorageBuffer, Input, Output, and
    /// PushConstant) is not enabled.
    pub shader_int8: bool,

    /// Indicates whether the implementation supports the minimum set of descriptor indexing
    /// features as described in the [Feature Requirements] section. Enabling the
    /// descriptorIndexing member when vkCreateDevice is called does not imply the other
    /// minimum descriptor indexing features are also enabled. Those other descriptor indexing
    /// features must be enabled individually as needed by the application.
    ///
    /// See the [features chapter](https://docs.vulkan.org/spec/latest/chapters/features.html).
    pub descriptor_indexing: bool,

    /// Indicates whether arrays of input attachments can be indexed by dynamically uniform integer
    /// expressions in shader code.
    ///
    /// If this feature is not enabled, resources with a descriptor type of
    /// VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT must be indexed only by constant integral expressions
    /// when aggregated into arrays in shader code. This also indicates whether shader modules can
    /// declare the InputAttachmentArrayDynamicIndexing capability.
    pub shader_input_attachment_array_dynamic_indexing: bool,

    /// Indicates whether arrays of uniform texel buffers can be indexed by dynamically uniform
    /// integer expressions in shader code.
    ///
    /// If this feature is not enabled, resources with a descriptor type of
    /// VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER must be indexed only by constant integral
    /// expressions when aggregated into arrays in shader code. This also indicates whether shader
    /// modules can declare the UniformTexelBufferArrayDynamicIndexing capability.
    pub shader_uniform_texel_buffer_array_dynamic_indexing: bool,

    /// Indicates whether arrays of storage texel buffers can be indexed by dynamically uniform
    /// integer expressions in shader code.
    ///
    /// If this feature is not enabled, resources with a descriptor type of
    /// VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER must be indexed only by constant integral
    /// expressions when aggregated into arrays in shader code. This also indicates whether shader
    /// modules can declare the StorageTexelBufferArrayDynamicIndexing capability.
    pub shader_storage_texel_buffer_array_dynamic_indexing: bool,

    /// Indicates whether arrays of uniform buffers can be indexed by non-uniform integer
    /// expressions in shader code.
    ///
    /// If this feature is not enabled, resources with a descriptor type of
    /// VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER or VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC must not be
    /// indexed by non-uniform integer expressions when aggregated into arrays in shader code. This
    /// also indicates whether shader modules can declare the UniformBufferArrayNonUniformIndexing
    /// capability.
    pub shader_uniform_buffer_array_non_uniform_indexing: bool,

    /// Indicates whether arrays of samplers or sampled images can be indexed by non-uniform
    /// integer expressions in shader code.
    ///
    /// If this feature is not enabled, resources with a descriptor type of
    /// VK_DESCRIPTOR_TYPE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, or
    /// VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE must not be indexed by non-uniform integer expressions
    /// when aggregated into arrays in shader code. This also indicates whether shader modules
    /// can declare the SampledImageArrayNonUniformIndexing capability.
    pub shader_sampled_image_array_non_uniform_indexing: bool,

    /// Indicates whether arrays of storage buffers can be indexed by non-uniform integer
    /// expressions in shader code.
    ///
    /// If this feature is not enabled, resources with a descriptor type of
    /// VK_DESCRIPTOR_TYPE_STORAGE_BUFFER or VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC must not be
    /// indexed by non-uniform integer expressions when aggregated into arrays in shader code. This
    /// also indicates whether shader modules can declare the StorageBufferArrayNonUniformIndexing
    /// capability.
    pub shader_storage_buffer_array_non_uniform_indexing: bool,

    /// Indicates whether arrays of storage images can be indexed by non-uniform integer
    /// expressions in shader code.
    ///
    /// If this feature is not enabled, resources with a descriptor type of
    /// VK_DESCRIPTOR_TYPE_STORAGE_IMAGE must not be indexed by non-uniform integer expressions
    /// when aggregated into arrays in shader code. This also indicates whether shader modules
    /// can declare the StorageImageArrayNonUniformIndexing capability.
    pub shader_storage_image_array_non_uniform_indexing: bool,

    /// Indicates whether arrays of input attachments can be indexed by non-uniform integer
    /// expressions in shader code.
    ///
    /// If this feature is not enabled, resources with a descriptor type of
    /// VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT must not be indexed by non-uniform integer expressions
    /// when aggregated into arrays in shader code. This also indicates whether shader modules can
    /// declare the InputAttachmentArrayNonUniformIndexing capability.
    pub shader_input_attachment_array_non_uniform_indexing: bool,

    /// Indicates whether arrays of uniform texel buffers can be indexed by non-uniform integer
    /// expressions in shader code.
    ///
    /// If this feature is not enabled, resources with a descriptor type of
    /// VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER must not be indexed by non-uniform integer
    /// expressions when aggregated into arrays in shader code. This also indicates whether shader
    /// modules can declare the UniformTexelBufferArrayNonUniformIndexing capability.
    pub shader_uniform_texel_buffer_array_non_uniform_indexing: bool,

    /// Indicates whether arrays of storage texel buffers can be indexed by non-uniform integer
    /// expressions in shader code.
    ///
    /// If this feature is not enabled, resources with a descriptor type of
    /// VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER must not be indexed by non-uniform integer
    /// expressions when aggregated into arrays in shader code. This also indicates whether shader
    /// modules can declare the StorageTexelBufferArrayNonUniformIndexing capability.
    pub shader_storage_texel_buffer_array_non_uniform_indexing: bool,

    /// Indicates whether the implementation supports updating uniform buffer descriptors after a
    /// set is bound.
    ///
    /// If this feature is not enabled, VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT must not be
    /// used with VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER.
    pub descriptor_binding_uniform_buffer_update_after_bind: bool,

    /// Indicates whether the implementation supports updating sampled image descriptors after a
    /// set is bound.
    ///
    /// If this feature is not enabled, VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT must not be
    /// used with VK_DESCRIPTOR_TYPE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, or
    /// VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE.
    pub descriptor_binding_sampled_image_update_after_bind: bool,

    /// Indicates whether the implementation supports updating storage image descriptors after a
    /// set is bound.
    ///
    /// If this feature is not enabled, VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT must not be
    /// used with VK_DESCRIPTOR_TYPE_STORAGE_IMAGE.
    pub descriptor_binding_storage_image_update_after_bind: bool,

    /// Indicates whether the implementation supports updating storage buffer descriptors after a
    /// set is bound.
    ///
    /// If this feature is not enabled, VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT must not be
    /// used with VK_DESCRIPTOR_TYPE_STORAGE_BUFFER.
    pub descriptor_binding_storage_buffer_update_after_bind: bool,

    /// Indicates whether the implementation supports updating uniform texel buffer descriptors
    /// after a set is bound.
    ///
    /// If this feature is not enabled, VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT must not be
    /// used with VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER.
    pub descriptor_binding_uniform_texel_buffer_update_after_bind: bool,

    /// Indicates whether the implementation supports updating storage texel buffer descriptors
    /// after a set is bound.
    ///
    /// If this feature is not enabled, VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT must not be
    /// used with VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER.
    pub descriptor_binding_storage_texel_buffer_update_after_bind: bool,

    /// Indicates whether the implementation supports updating descriptors while the set is in use.
    ///
    /// If this feature is not enabled, VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT must
    /// not be used.
    pub descriptor_binding_update_unused_while_pending: bool,

    /// Indicates whether the implementation supports statically using a descriptor set binding in
    /// which some descriptors are not valid. If this feature is not enabled,
    /// VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT must not be used.
    pub descriptor_binding_partially_bound: bool,

    /// Indicates whether the implementation supports descriptor sets with a variable-sized last
    /// binding. If this feature is not enabled,
    /// VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT must not be used.
    pub descriptor_binding_variable_descriptor_count: bool,

    /// Indicates whether the implementation supports the SPIR-V RuntimeDescriptorArray capability.
    ///
    /// If this feature is not enabled, descriptors must not be declared in runtime arrays.
    pub runtime_descriptor_array: bool,

    /// Indicates whether the implementation supports a minimum set of required formats supporting
    /// min/max filtering as defined by the filterMinmaxSingleComponentFormats property minimum
    /// requirements.
    ///
    /// If this feature is not enabled, then VkSamplerReductionModeCreateInfo must only use
    /// VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE.
    pub sampler_filter_minmax: bool,

    /// Indicates that the implementation supports the layout of resource blocks in shaders using
    /// scalar alignment.
    pub scalar_block_layout: bool,

    /// Indicates that the implementation supports specifying the image view for attachments at
    /// render pass begin time via VkRenderPassAttachmentBeginInfo.
    pub imageless_framebuffer: bool,

    /// Indicates that the implementation supports the same layouts for uniform buffers as for
    /// storage and other kinds of buffers.
    ///
    /// See [Standard Buffer Layout].
    ///
    /// See the [features chapter](https://docs.vulkan.org/spec/latest/chapters/features.html).
    pub uniform_buffer_standard_layout: bool,

    /// A boolean specifying whether subgroup operations can use 8-bit integer, 16-bit integer,
    /// 64-bit integer, 16-bit floating-point, and vectors of these types in group operations with
    /// subgroup scope, if the implementation supports the types.
    pub shader_subgroup_extended_types: bool,

    /// Indicates whether the implementation supports a VkImageMemoryBarrier for a depth/stencil
    /// image with only one of VK_IMAGE_ASPECT_DEPTH_BIT or VK_IMAGE_ASPECT_STENCIL_BIT set, and
    /// whether VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
    /// VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL, or VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL
    /// can be used.
    pub separate_depth_stencil_layouts: bool,

    /// Indicates that the implementation supports resetting queries from the host with
    /// vkResetQueryPool.
    pub host_query_reset: bool,

    /// Indicates whether semaphores created with a VkSemaphoreType of VK_SEMAPHORE_TYPE_TIMELINE
    /// are supported.
    pub timeline_semaphore: bool,

    /// Indicates that the implementation supports accessing buffer memory in shaders as storage
    /// buffers via an address queried from vkGetBufferDeviceAddress.
    pub buffer_device_address: bool,

    /// Indicates that the implementation supports saving and reusing buffer and device addresses,
    /// e.g. for trace capture and replay.
    pub buffer_device_address_capture_replay: bool,

    /// Indicates that the implementation supports the bufferDeviceAddress, rayTracingPipeline and
    /// rayQuery features for logical devices created with multiple physical devices.
    ///
    /// If this feature is not supported, buffer and acceleration structure addresses must not be
    /// queried on a logical device created with more than one physical device.
    pub buffer_device_address_multi_device: bool,

    /// Indicates whether the [Vulkan Memory Model] is supported.
    ///
    /// This also indicates whether shader modules can declare the VulkanMemoryModel capability.
    ///
    /// See the [features chapter](https://docs.vulkan.org/spec/latest/chapters/features.html).
    pub vulkan_memory_model: bool,

    /// Indicates whether the [Vulkan Memory Model] can use Device scope synchronization.
    ///
    /// This also indicates whether shader modules can declare the VulkanMemoryModelDeviceScope
    /// capability.
    ///
    /// See the [features chapter](https://docs.vulkan.org/spec/latest/chapters/features.html).
    pub vulkan_memory_model_device_scope: bool,

    /// Indicates whether the [Vulkan Memory Model] can use availability and visibility chains with
    /// more than one element.
    ///
    /// See the [features chapter](https://docs.vulkan.org/spec/latest/chapters/features.html).
    pub vulkan_memory_model_availability_visibility_chains: bool,

    /// Indicates whether the implementation supports the ShaderViewportIndex SPIR-V capability
    /// enabling variables decorated with the ViewportIndex built-in to be exported from mesh,
    /// vertex or tessellation evaluation shaders.
    ///
    /// If this feature is not enabled, the ViewportIndex built-in decoration must not be used on
    /// outputs in mesh, vertex or tessellation evaluation shaders.
    pub shader_output_viewport_index: bool,

    /// Indicates whether the implementation supports the ShaderLayer SPIR-V capability enabling
    /// variables decorated with the Layer built-in to be exported from mesh, vertex or
    /// tessellation evaluation shaders.
    ///
    /// If this feature is not enabled, the Layer built-in decoration must not be used on outputs
    /// in mesh, vertex or tessellation evaluation shaders.
    pub shader_output_layer: bool,

    /// If `true`, the “Id” operand of OpGroupNonUniformBroadcast can be dynamically uniform within
    /// a subgroup, and the “Index” operand of OpGroupNonUniformQuadBroadcast can be dynamically
    /// uniform within the derivative group.
    ///
    /// If `false`, these operands must be constants.
    pub subgroup_broadcast_dynamic_id: bool,
}

impl From<vk::PhysicalDeviceVulkan12Features<'_>> for Vulkan12Features {
    fn from(features: vk::PhysicalDeviceVulkan12Features<'_>) -> Self {
        Self {
            sampler_mirror_clamp_to_edge: features.sampler_mirror_clamp_to_edge == vk::TRUE,
            draw_indirect_count: features.draw_indirect_count == vk::TRUE,
            storage_buffer8_bit_access: features.storage_buffer8_bit_access == vk::TRUE,
            uniform_and_storage_buffer8_bit_access: features.uniform_and_storage_buffer8_bit_access
                == vk::TRUE,
            storage_push_constant8: features.storage_push_constant8 == vk::TRUE,
            shader_buffer_int64_atomics: features.shader_buffer_int64_atomics == vk::TRUE,
            shader_shared_int64_atomics: features.shader_shared_int64_atomics == vk::TRUE,
            shader_float16: features.shader_float16 == vk::TRUE,
            shader_int8: features.shader_int8 == vk::TRUE,
            descriptor_indexing: features.descriptor_indexing == vk::TRUE,
            shader_input_attachment_array_dynamic_indexing: features
                .shader_input_attachment_array_dynamic_indexing
                == vk::TRUE,
            shader_uniform_texel_buffer_array_dynamic_indexing: features
                .shader_uniform_texel_buffer_array_dynamic_indexing
                == vk::TRUE,
            shader_storage_texel_buffer_array_dynamic_indexing: features
                .shader_storage_texel_buffer_array_dynamic_indexing
                == vk::TRUE,
            shader_uniform_buffer_array_non_uniform_indexing: features
                .shader_uniform_buffer_array_non_uniform_indexing
                == vk::TRUE,
            shader_sampled_image_array_non_uniform_indexing: features
                .shader_sampled_image_array_non_uniform_indexing
                == vk::TRUE,
            shader_storage_buffer_array_non_uniform_indexing: features
                .shader_storage_buffer_array_non_uniform_indexing
                == vk::TRUE,
            shader_storage_image_array_non_uniform_indexing: features
                .shader_storage_image_array_non_uniform_indexing
                == vk::TRUE,
            shader_input_attachment_array_non_uniform_indexing: features
                .shader_input_attachment_array_non_uniform_indexing
                == vk::TRUE,
            shader_uniform_texel_buffer_array_non_uniform_indexing: features
                .shader_uniform_texel_buffer_array_non_uniform_indexing
                == vk::TRUE,
            shader_storage_texel_buffer_array_non_uniform_indexing: features
                .shader_storage_texel_buffer_array_non_uniform_indexing
                == vk::TRUE,
            descriptor_binding_uniform_buffer_update_after_bind: features
                .descriptor_binding_uniform_buffer_update_after_bind
                == vk::TRUE,
            descriptor_binding_sampled_image_update_after_bind: features
                .descriptor_binding_sampled_image_update_after_bind
                == vk::TRUE,
            descriptor_binding_storage_image_update_after_bind: features
                .descriptor_binding_storage_image_update_after_bind
                == vk::TRUE,
            descriptor_binding_storage_buffer_update_after_bind: features
                .descriptor_binding_storage_buffer_update_after_bind
                == vk::TRUE,
            descriptor_binding_uniform_texel_buffer_update_after_bind: features
                .descriptor_binding_uniform_texel_buffer_update_after_bind
                == vk::TRUE,
            descriptor_binding_storage_texel_buffer_update_after_bind: features
                .descriptor_binding_storage_texel_buffer_update_after_bind
                == vk::TRUE,
            descriptor_binding_update_unused_while_pending: features
                .descriptor_binding_update_unused_while_pending
                == vk::TRUE,
            descriptor_binding_partially_bound: features.descriptor_binding_partially_bound
                == vk::TRUE,
            descriptor_binding_variable_descriptor_count: features
                .descriptor_binding_variable_descriptor_count
                == vk::TRUE,
            runtime_descriptor_array: features.runtime_descriptor_array == vk::TRUE,
            sampler_filter_minmax: features.sampler_filter_minmax == vk::TRUE,
            scalar_block_layout: features.scalar_block_layout == vk::TRUE,
            imageless_framebuffer: features.imageless_framebuffer == vk::TRUE,
            uniform_buffer_standard_layout: features.uniform_buffer_standard_layout == vk::TRUE,
            shader_subgroup_extended_types: features.shader_subgroup_extended_types == vk::TRUE,
            separate_depth_stencil_layouts: features.separate_depth_stencil_layouts == vk::TRUE,
            host_query_reset: features.host_query_reset == vk::TRUE,
            timeline_semaphore: features.timeline_semaphore == vk::TRUE,
            buffer_device_address: features.buffer_device_address == vk::TRUE,
            buffer_device_address_capture_replay: features.buffer_device_address_capture_replay
                == vk::TRUE,
            buffer_device_address_multi_device: features.buffer_device_address_multi_device
                == vk::TRUE,
            vulkan_memory_model: features.vulkan_memory_model == vk::TRUE,
            vulkan_memory_model_device_scope: features.vulkan_memory_model_device_scope == vk::TRUE,
            vulkan_memory_model_availability_visibility_chains: features
                .vulkan_memory_model_availability_visibility_chains
                == vk::TRUE,
            shader_output_viewport_index: features.shader_output_viewport_index == vk::TRUE,
            shader_output_layer: features.shader_output_layer == vk::TRUE,
            subgroup_broadcast_dynamic_id: features.subgroup_broadcast_dynamic_id == vk::TRUE,
        }
    }
}

/// Description of Vulkan 1.2 properties.
///
/// See [`VkPhysicalDeviceVulkan12Properties`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceVulkan12Properties.html).
#[derive(Clone, Debug)]
pub struct Vulkan12Properties {
    /// A unique identifier for the driver of the physical device.
    pub driver_id: vk::DriverId,

    /// An array of `VK_MAX_DRIVER_NAME_SIZE` char containing a null-terminated UTF-8 string which
    /// is the name of the driver.
    pub driver_name: String,

    /// An array of `VK_MAX_DRIVER_INFO_SIZE` char containing a null-terminated UTF-8 string with
    /// additional information about the driver.
    pub driver_info: String,

    /// The version of the Vulkan conformance test this driver is conformant against (see
    /// See the [limits chapter](https://docs.vulkan.org/spec/latest/chapters/limits.html).
    pub conformance_version: vk::ConformanceVersion,

    /// A `vk::ShaderFloatControlsIndependence` value indicating whether, and how, denorm behavior
    /// can be set independently for different bit widths.
    pub denorm_behavior_independence: vk::ShaderFloatControlsIndependence,

    /// A `vk::ShaderFloatControlsIndependence` value indicating whether, and how, rounding modes
    /// can be set independently for different bit widths.
    pub rounding_mode_independence: vk::ShaderFloatControlsIndependence,

    /// A `bool` value indicating whether sign of a zero, Nans and ±∞ can be preserved in 16-bit
    /// floating-point computations.
    ///
    /// It also indicates whether the SignedZeroInfNanPreserve execution mode can be used for
    /// 16-bit floating-point types.
    pub shader_signed_zero_inf_nan_preserve_float16: bool,

    /// A `bool` value indicating whether sign of a zero, Nans and ±∞ can be preserved in 32-bit
    /// floating-point computations.
    ///
    /// It also indicates whether the SignedZeroInfNanPreserve execution mode can be used for
    /// 32-bit floating-point types.
    pub shader_signed_zero_inf_nan_preserve_float32: bool,

    /// A `bool` value indicating whether sign of a zero, Nans and ±∞ can be preserved in 64-bit
    /// floating-point computations.
    ///
    /// It also indicates whether the SignedZeroInfNanPreserve execution mode can be used for
    /// 64-bit floating-point types.
    pub shader_signed_zero_inf_nan_preserve_float64: bool,

    /// A `bool` value indicating whether denormals can be preserved in 16-bit floating-point
    /// computations.
    ///
    /// It also indicates whether the DenormPreserve execution mode can be used for 16-bit
    /// floating-point types.
    pub shader_denorm_preserve_float16: bool,

    /// A `bool` value indicating whether denormals can be preserved in 32-bit floating-point
    /// computations.
    ///
    /// It also indicates whether the DenormPreserve execution mode can be used for 32-bit
    /// floating-point types.
    pub shader_denorm_preserve_float32: bool,

    /// A `bool` value indicating whether denormals can be preserved in 64-bit floating-point
    /// computations.
    ///
    /// It also indicates whether the DenormPreserve execution mode can be used for 64-bit
    /// floating-point types.
    pub shader_denorm_preserve_float64: bool,

    /// A `bool` value indicating whether denormals can be flushed to zero in 16-bit floating-point
    /// computations.
    ///
    /// It also indicates whether the DenormFlushToZero execution mode can be used for 16-bit
    /// floating-point types.
    pub shader_denorm_flush_to_zero_float16: bool,

    /// A `bool` value indicating whether denormals can be flushed to zero in 32-bit floating-point
    /// computations.
    ///
    /// It also indicates whether the DenormFlushToZero execution mode can be used for 32-bit
    /// floating-point types.
    pub shader_denorm_flush_to_zero_float32: bool,

    /// A `bool` value indicating whether denormals can be flushed to zero in 64-bit floating-point
    /// computations.
    ///
    /// It also indicates whether the DenormFlushToZero execution mode can be used for 64-bit
    /// floating-point types.
    pub shader_denorm_flush_to_zero_float64: bool,

    /// A `bool` value indicating whether an implementation supports the round-to-nearest-even
    /// rounding mode for 16-bit floating-point arithmetic and conversion instructions.
    ///
    /// It also indicates whether the RoundingModeRTE execution mode can be used for 16-bit
    /// floating-point types.
    pub shader_rounding_mode_rte_float16: bool,

    /// A `bool` value indicating whether an implementation supports the round-to-nearest-even
    /// rounding mode for 32-bit floating-point arithmetic and conversion instructions.
    ///
    /// It also indicates whether the RoundingModeRTE execution mode can be used for 32-bit
    /// floating-point types.
    pub shader_rounding_mode_rte_float32: bool,

    /// A `bool` value indicating whether an implementation supports the round-to-nearest-even
    /// rounding mode for 64-bit floating-point arithmetic and conversion instructions.
    ///
    /// It also indicates whether the RoundingModeRTE execution mode can be used for 64-bit
    /// floating-point types.
    pub shader_rounding_mode_rte_float64: bool,

    /// A `bool` value indicating whether an implementation supports the round-towards-zero
    /// rounding mode for 16-bit floating-point arithmetic and conversion instructions.
    ///
    /// It also indicates whether the RoundingModeRTZ execution mode can be used for 16-bit
    /// floating-point types.
    pub shader_rounding_mode_rtz_float16: bool,

    /// A `bool` value indicating whether an implementation supports the round-towards-zero
    /// rounding mode for 32-bit floating-point arithmetic and conversion instructions.
    ///
    /// It also indicates whether the RoundingModeRTZ execution mode can be used for 32-bit
    /// floating-point types.
    pub shader_rounding_mode_rtz_float32: bool,

    /// A `bool` value indicating whether an implementation supports the round-towards-zero
    /// rounding mode for 64-bit floating-point arithmetic and conversion instructions.
    ///
    /// It also indicates whether the RoundingModeRTZ execution mode can be used for 64-bit
    /// floating-point types.
    pub shader_rounding_mode_rtz_float64: bool,

    /// The maximum number of descriptors (summed over all descriptor types) that can be created
    /// across all pools that are created with the VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT
    /// bit set.
    ///
    /// Pool creation may fail when this limit is exceeded, or when the space this limit represents
    /// is unable to satisfy a pool creation due to fragmentation.
    pub max_update_after_bind_descriptors_in_all_pools: u32,

    /// A `bool` value indicating whether uniform buffer descriptors natively support nonuniform
    /// indexing.
    ///
    /// If this is `false`, then a single dynamic instance of an instruction that nonuniformly
    /// indexes an array of uniform buffers may execute multiple times in order to access all the
    /// descriptors.
    pub shader_uniform_buffer_array_non_uniform_indexing_native: bool,

    /// A `bool` value indicating whether sampler and image descriptors natively support nonuniform
    /// indexing.
    ///
    /// If this is `false`, then a single dynamic instance of an instruction that nonuniformly
    /// indexes an array of samplers or images may execute multiple times in order to access all
    /// the descriptors.
    pub shader_sampled_image_array_non_uniform_indexing_native: bool,

    /// A `bool` value indicating whether storage buffer descriptors natively support nonuniform
    /// indexing.
    ///
    /// If this is `false`, then a single dynamic instance of an instruction that nonuniformly
    /// indexes an array of storage buffers may execute multiple times in order to access all the
    /// descriptors.
    pub shader_storage_buffer_array_non_uniform_indexing_native: bool,

    /// A `bool` value indicating whether storage image descriptors natively support nonuniform
    /// indexing.
    ///
    /// If this is `false`, then a single dynamic instance of an instruction that nonuniformly
    /// indexes an array of storage images may execute multiple times in order to access all the
    /// descriptors.
    pub shader_storage_image_array_non_uniform_indexing_native: bool,

    /// A `bool` value indicating whether input attachment descriptors natively support nonuniform
    /// indexing.
    ///
    /// If this is `false`, then a single dynamic instance of an instruction that nonuniformly
    /// indexes an array of input attachments may execute multiple times in order to access all the
    /// descriptors.
    pub shader_input_attachment_array_non_uniform_indexing_native: bool,

    /// A `bool` value indicating whether `robustBufferAccess` can be enabled on a device
    /// simultaneously with `descriptorBindingUniformBufferUpdateAfterBind`,
    /// `descriptorBindingStorageBufferUpdateAfterBind`,
    /// `descriptorBindingUniformTexelBufferUpdateAfterBind`, and/or
    /// `descriptorBindingStorageTexelBufferUpdateAfterBind`.
    ///
    /// If this is `false`, then either `robustBufferAccess` must be disabled or all of these
    /// update-after-bind features must be disabled.
    pub robust_buffer_access_update_after_bind: bool,

    /// A `bool` value indicating whether implicit level of detail calculations for image
    /// operations have well-defined results when the image and/or sampler objects used for the
    /// instruction are not uniform within a quad.
    ///
    /// See the [features chapter](https://docs.vulkan.org/spec/latest/chapters/features.html).
    pub quad_divergent_implicit_lod: bool,

    /// Similar to `maxPerStageDescriptorSamplers` but counts descriptors from descriptor sets
    /// created with or without the `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT`
    /// bit set.
    pub max_per_stage_descriptor_update_after_bind_samplers: u32,

    /// Similar to `maxPerStageDescriptorUniformBuffers` but counts descriptors from descriptor
    /// sets created with or without the
    /// `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT` bit set.
    pub max_per_stage_descriptor_update_after_bind_uniform_buffers: u32,

    /// Similar to `maxPerStageDescriptorStorageBuffers` but counts descriptors from descriptor
    /// sets created with or without the
    /// `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT` bit set.
    pub max_per_stage_descriptor_update_after_bind_storage_buffers: u32,

    /// Similar to `maxPerStageDescriptorSampledImages` but counts descriptors from descriptor sets
    /// created with or without the `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT`
    /// bit set.
    pub max_per_stage_descriptor_update_after_bind_sampled_images: u32,

    /// Similar to `maxPerStageDescriptorStorageImages` but counts descriptors from descriptor sets
    /// created with or without the `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT`
    /// bit set.
    pub max_per_stage_descriptor_update_after_bind_storage_images: u32,

    /// Similar to `maxPerStageDescriptorInputAttachments` but counts descriptors from descriptor
    /// sets created with or without the
    /// `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT` bit set.
    pub max_per_stage_descriptor_update_after_bind_input_attachments: u32,

    /// Similar to `maxPerStageResources` but counts descriptors from descriptor sets created with
    /// or without the `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT` bit set.
    pub max_per_stage_update_after_bind_resources: u32,

    /// Similar to `maxDescriptorSetSamplers` but counts descriptors from descriptor sets created
    /// with or without the `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT` bit set.
    pub max_descriptor_set_update_after_bind_samplers: u32,

    /// Similar to `maxDescriptorSetUniformBuffers` but counts descriptors from descriptor sets
    /// created with or without the `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT`
    /// bit set.
    pub max_descriptor_set_update_after_bind_uniform_buffers: u32,

    /// Similar to `maxDescriptorSetUniformBuffersDynamic` but counts descriptors from descriptor
    /// sets created with or without the
    /// `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT` bit set.
    ///
    /// While an application can allocate dynamic uniform buffer descriptors from a pool created
    /// with the `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT`, bindings for these
    /// descriptors must not be present in any descriptor set layout that includes bindings created
    /// with `VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT`.
    pub max_descriptor_set_update_after_bind_uniform_buffers_dynamic: u32,

    /// Similar to `maxDescriptorSetStorageBuffers` but counts descriptors from descriptor sets
    /// created with or without the `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT`
    /// bit set.
    pub max_descriptor_set_update_after_bind_storage_buffers: u32,

    /// Similar to `maxDescriptorSetStorageBuffersDynamic` but counts descriptors from descriptor
    /// sets created with or without the
    /// `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT` bit set.
    ///
    /// While an application can allocate dynamic storage buffer descriptors from a pool created
    /// with the `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT`, bindings for these
    /// descriptors must not be present in any descriptor set layout that includes bindings created
    /// with `VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT`.
    pub max_descriptor_set_update_after_bind_storage_buffers_dynamic: u32,

    /// Similar to `maxDescriptorSetSampledImages` but counts descriptors from descriptor sets
    /// created with or without the `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT`
    /// bit set.
    pub max_descriptor_set_update_after_bind_sampled_images: u32,

    /// Similar to `maxDescriptorSetStorageImages` but counts descriptors from descriptor sets
    /// created with or without the `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT`
    /// bit set.
    pub max_descriptor_set_update_after_bind_storage_images: u32,

    /// Similar to `maxDescriptorSetInputAttachments` but counts descriptors from descriptor sets
    /// created with or without the `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT`
    /// bit set.
    pub max_descriptor_set_update_after_bind_input_attachments: u32,

    /// A bitmask of `vk::ResolveModeFlagBits` indicating the set of supported depth resolve modes.
    ///
    /// `VK_RESOLVE_MODE_SAMPLE_ZERO_BIT` must be included in the set but implementations may
    /// support additional modes.
    pub supported_depth_resolve_modes: vk::ResolveModeFlags,

    /// A bitmask of `vk::ResolveModeFlagBits` indicating the set of supported stencil resolve
    /// modes.
    ///
    /// `VK_RESOLVE_MODE_SAMPLE_ZERO_BIT` must be included in the set but implementations may
    /// support additional modes. `VK_RESOLVE_MODE_AVERAGE_BIT` must not be included in the set.
    pub supported_stencil_resolve_modes: vk::ResolveModeFlags,

    /// `true` if the implementation supports setting the depth and stencil resolve modes to
    /// different values when one of those modes is `VK_RESOLVE_MODE_NONE`.
    ///
    /// Otherwise the implementation only supports setting both modes to the same value.
    pub independent_resolve_none: bool,

    /// `true` if the implementation supports all combinations of the supported depth and stencil
    /// resolve modes, including setting either depth or stencil resolve mode to
    /// `VK_RESOLVE_MODE_NONE`.
    ///
    /// An implementation that supports `independent_resolve` must also support
    /// `independent_resolve_none`.
    pub independent_resolve: bool,

    /// A `bool` value indicating whether a minimum set of required formats support min/max
    /// filtering.
    pub filter_minmax_single_component_formats: bool,

    /// A `bool` value indicating whether the implementation supports non-identity component
    /// mapping of the image when doing min/max filtering.
    pub filter_minmax_image_component_mapping: bool,

    /// Indicates the maximum difference allowed by the implementation between the current value of
    /// a timeline semaphore and any pending signal or wait operations.
    pub max_timeline_semaphore_value_difference: u64,

    /// A bitmask of `vk::SampleCountFlagBits` indicating the color sample counts that are
    /// supported for all framebuffer color attachments with integer formats.
    pub framebuffer_integer_color_sample_counts: vk::SampleCountFlags,
}

impl From<vk::PhysicalDeviceVulkan12Properties<'_>> for Vulkan12Properties {
    fn from(properties: vk::PhysicalDeviceVulkan12Properties<'_>) -> Self {
        Self {
            driver_id: properties.driver_id,
            driver_name: vk_cstr_to_string_lossy(&properties.driver_name),
            driver_info: vk_cstr_to_string_lossy(&properties.driver_info),
            conformance_version: properties.conformance_version,
            denorm_behavior_independence: properties.denorm_behavior_independence,
            rounding_mode_independence: properties.rounding_mode_independence,
            shader_signed_zero_inf_nan_preserve_float16: properties
                .shader_signed_zero_inf_nan_preserve_float16
                == vk::TRUE,
            shader_signed_zero_inf_nan_preserve_float32: properties
                .shader_signed_zero_inf_nan_preserve_float32
                == vk::TRUE,
            shader_signed_zero_inf_nan_preserve_float64: properties
                .shader_signed_zero_inf_nan_preserve_float64
                == vk::TRUE,
            shader_denorm_preserve_float16: properties.shader_denorm_preserve_float16 == vk::TRUE,
            shader_denorm_preserve_float32: properties.shader_denorm_preserve_float32 == vk::TRUE,
            shader_denorm_preserve_float64: properties.shader_denorm_preserve_float64 == vk::TRUE,
            shader_denorm_flush_to_zero_float16: properties.shader_denorm_flush_to_zero_float16
                == vk::TRUE,
            shader_denorm_flush_to_zero_float32: properties.shader_denorm_flush_to_zero_float32
                == vk::TRUE,
            shader_denorm_flush_to_zero_float64: properties.shader_denorm_flush_to_zero_float64
                == vk::TRUE,
            shader_rounding_mode_rte_float16: properties.shader_rounding_mode_rte_float16
                == vk::TRUE,
            shader_rounding_mode_rte_float32: properties.shader_rounding_mode_rte_float32
                == vk::TRUE,
            shader_rounding_mode_rte_float64: properties.shader_rounding_mode_rte_float64
                == vk::TRUE,
            shader_rounding_mode_rtz_float16: properties.shader_rounding_mode_rtz_float16
                == vk::TRUE,
            shader_rounding_mode_rtz_float32: properties.shader_rounding_mode_rtz_float32
                == vk::TRUE,
            shader_rounding_mode_rtz_float64: properties.shader_rounding_mode_rtz_float64
                == vk::TRUE,
            max_update_after_bind_descriptors_in_all_pools: properties
                .max_update_after_bind_descriptors_in_all_pools,
            shader_uniform_buffer_array_non_uniform_indexing_native: properties
                .shader_uniform_buffer_array_non_uniform_indexing_native
                == vk::TRUE,
            shader_sampled_image_array_non_uniform_indexing_native: properties
                .shader_sampled_image_array_non_uniform_indexing_native
                == vk::TRUE,
            shader_storage_buffer_array_non_uniform_indexing_native: properties
                .shader_storage_buffer_array_non_uniform_indexing_native
                == vk::TRUE,
            shader_storage_image_array_non_uniform_indexing_native: properties
                .shader_storage_image_array_non_uniform_indexing_native
                == vk::TRUE,
            shader_input_attachment_array_non_uniform_indexing_native: properties
                .shader_input_attachment_array_non_uniform_indexing_native
                == vk::TRUE,
            robust_buffer_access_update_after_bind: properties
                .robust_buffer_access_update_after_bind
                == vk::TRUE,
            quad_divergent_implicit_lod: properties.quad_divergent_implicit_lod == vk::TRUE,
            max_per_stage_descriptor_update_after_bind_samplers: properties
                .max_per_stage_descriptor_update_after_bind_samplers,
            max_per_stage_descriptor_update_after_bind_uniform_buffers: properties
                .max_per_stage_descriptor_update_after_bind_uniform_buffers,
            max_per_stage_descriptor_update_after_bind_storage_buffers: properties
                .max_per_stage_descriptor_update_after_bind_storage_buffers,
            max_per_stage_descriptor_update_after_bind_sampled_images: properties
                .max_per_stage_descriptor_update_after_bind_sampled_images,
            max_per_stage_descriptor_update_after_bind_storage_images: properties
                .max_per_stage_descriptor_update_after_bind_storage_images,
            max_per_stage_descriptor_update_after_bind_input_attachments: properties
                .max_per_stage_descriptor_update_after_bind_input_attachments,
            max_per_stage_update_after_bind_resources: properties
                .max_per_stage_update_after_bind_resources,
            max_descriptor_set_update_after_bind_samplers: properties
                .max_descriptor_set_update_after_bind_samplers,
            max_descriptor_set_update_after_bind_uniform_buffers: properties
                .max_descriptor_set_update_after_bind_uniform_buffers,
            max_descriptor_set_update_after_bind_uniform_buffers_dynamic: properties
                .max_descriptor_set_update_after_bind_uniform_buffers_dynamic,
            max_descriptor_set_update_after_bind_storage_buffers: properties
                .max_descriptor_set_update_after_bind_storage_buffers,
            max_descriptor_set_update_after_bind_storage_buffers_dynamic: properties
                .max_descriptor_set_update_after_bind_storage_buffers_dynamic,
            max_descriptor_set_update_after_bind_sampled_images: properties
                .max_descriptor_set_update_after_bind_sampled_images,
            max_descriptor_set_update_after_bind_storage_images: properties
                .max_descriptor_set_update_after_bind_storage_images,
            max_descriptor_set_update_after_bind_input_attachments: properties
                .max_descriptor_set_update_after_bind_input_attachments,
            supported_depth_resolve_modes: properties.supported_depth_resolve_modes,
            supported_stencil_resolve_modes: properties.supported_stencil_resolve_modes,
            independent_resolve_none: properties.independent_resolve_none == vk::TRUE,
            independent_resolve: properties.independent_resolve == vk::TRUE,
            filter_minmax_single_component_formats: properties
                .filter_minmax_single_component_formats
                == vk::TRUE,
            filter_minmax_image_component_mapping: properties.filter_minmax_image_component_mapping
                == vk::TRUE,
            max_timeline_semaphore_value_difference: properties
                .max_timeline_semaphore_value_difference,
            framebuffer_integer_color_sample_counts: properties
                .framebuffer_integer_color_sample_counts,
        }
    }
}
