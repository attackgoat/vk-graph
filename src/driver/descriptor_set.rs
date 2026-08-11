//! Explicit descriptor set allocation and update support.
//!
//! [`DescriptorSet::alloc_and_update`] creates an immutable descriptor set from a reflected
//! pipeline layout. The result owns its Vulkan descriptor pool and keeps every resource referenced
//! by a write or copy alive. Clone it cheaply to reuse the same allocation.
//!
//! Descriptor contents do not declare graph synchronization. Bind each referenced resource to the
//! graph and declare its [`AccessType`](crate::driver::sync::AccessType) separately, then bind the
//! set with [`PipelineCommand::bind_descriptor_set`](crate::cmd::PipelineCommand::bind_descriptor_set).
//!
//! Input attachments, texel buffers, and dynamic buffer descriptors are currently unsupported by
//! this immutable API and prevent allocating a first-class set for their set index.

use {
    super::{
        DescriptorSetLayout, DriverError,
        accel_struct::AccelerationStructure,
        buffer::{Buffer, BufferSubresourceRange},
        compute::ComputePipeline,
        device::Device,
        format_aspect_mask,
        graphics::GraphicsPipeline,
        image::{Image, ImageViewInfo},
        ray_tracing::RayTracingPipeline,
        shader::PipelineDescriptorInfo,
    },
    ash::vk,
    log::warn,
    std::{
        fmt::{Debug, Formatter},
        iter,
        ops::Deref,
        slice,
        sync::Arc,
        thread::panicking,
    },
};

/// Descriptor pool resource used to allocate descriptor sets for pipeline execution.
///
/// See [`VkDescriptorPool`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorPool.html).
#[derive(Debug)]
#[read_only::cast]
pub(crate) struct DescriptorPool {
    /// The device which owns this descriptor pool resource.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub device: Device,

    /// The native Vulkan resource handle of this descriptor pool.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub handle: vk::DescriptorPool,

    /// Information used to create this descriptor pool resource.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub(crate) info: DescriptorPoolInfo,
}

impl DescriptorPool {
    #[profiling::function]
    pub(crate) fn create(
        device: &Device,
        info: impl Into<DescriptorPoolInfo>,
    ) -> Result<Self, DriverError> {
        let device = device.clone();
        let info = info.into();

        let mut pool_sizes = [vk::DescriptorPoolSize {
            ty: Default::default(),
            descriptor_count: 0,
        }; 12];
        let mut pool_size_count = 0;

        if info.acceleration_structure_count > 0 {
            pool_sizes[pool_size_count] = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: info.acceleration_structure_count,
            };
            pool_size_count += 1;
        }

        if info.combined_image_sampler_count > 0 {
            pool_sizes[pool_size_count] = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: info.combined_image_sampler_count,
            };
            pool_size_count += 1;
        }

        if info.input_attachment_count > 0 {
            pool_sizes[pool_size_count] = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::INPUT_ATTACHMENT,
                descriptor_count: info.input_attachment_count,
            };
            pool_size_count += 1;
        }

        if info.sampled_image_count > 0 {
            pool_sizes[pool_size_count] = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: info.sampled_image_count,
            };
            pool_size_count += 1;
        }

        if info.sampler_count > 0 {
            pool_sizes[pool_size_count] = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: info.sampler_count,
            };
            pool_size_count += 1;
        }

        if info.storage_buffer_count > 0 {
            pool_sizes[pool_size_count] = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: info.storage_buffer_count,
            };
            pool_size_count += 1;
        }

        if info.storage_buffer_dynamic_count > 0 {
            pool_sizes[pool_size_count] = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                descriptor_count: info.storage_buffer_dynamic_count,
            };
            pool_size_count += 1;
        }

        if info.storage_image_count > 0 {
            pool_sizes[pool_size_count] = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: info.storage_image_count,
            };
            pool_size_count += 1;
        }

        if info.storage_texel_buffer_count > 0 {
            pool_sizes[pool_size_count] = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                descriptor_count: info.storage_texel_buffer_count,
            };
            pool_size_count += 1;
        }

        if info.uniform_buffer_count > 0 {
            pool_sizes[pool_size_count] = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: info.uniform_buffer_count,
            };
            pool_size_count += 1;
        }

        if info.uniform_buffer_dynamic_count > 0 {
            pool_sizes[pool_size_count] = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                descriptor_count: info.uniform_buffer_dynamic_count,
            };
            pool_size_count += 1;
        }

        if info.uniform_texel_buffer_count > 0 {
            pool_sizes[pool_size_count] = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                descriptor_count: info.uniform_texel_buffer_count,
            };
            pool_size_count += 1;
        }

        let handle = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(info.max_sets)
                    .pool_sizes(&pool_sizes[0..pool_size_count]),
                None,
            )
        }
        .map_err(|err| {
            warn!("unable to create descriptor pool: {err}");

            match err {
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                    DriverError::OutOfMemory
                }
                _ => DriverError::Unsupported,
            }
        })?;

        Ok(Self {
            device,
            handle,
            info,
        })
    }

    pub(crate) fn allocate_descriptor_set(
        this: &Self,
        layout: &DescriptorSetLayout,
    ) -> Result<RawDescriptorSet, DriverError> {
        Ok(Self::allocate_descriptor_sets(this, layout, 1)?
            .next()
            .expect("missing descriptor set"))
    }

    #[profiling::function]
    pub(crate) fn allocate_descriptor_sets<'a>(
        &'a self,
        layout: &DescriptorSetLayout,
        count: u32,
    ) -> Result<impl Iterator<Item = RawDescriptorSet> + 'a, DriverError> {
        let layout_handles = vec![layout.handle(); count as usize];
        let create_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.handle)
            .set_layouts(&layout_handles);

        Ok(unsafe {
            self.device
                .allocate_descriptor_sets(&create_info)
                .map_err(|err| {
                    use {DriverError::*, vk::Result as vk};

                    warn!("unable to allocate descriptor sets: {err}");

                    match err {
                        e if e == vk::ERROR_FRAGMENTED_POOL => InvalidData,
                        e if e == vk::ERROR_OUT_OF_DEVICE_MEMORY => OutOfMemory,
                        e if e == vk::ERROR_OUT_OF_HOST_MEMORY => OutOfMemory,
                        e if e == vk::ERROR_OUT_OF_POOL_MEMORY => OutOfMemory,
                        _ => Unsupported,
                    }
                })?
                .into_iter()
                .map(move |descriptor_set| RawDescriptorSet {
                    descriptor_pool: self.handle,
                    descriptor_set,
                    device: self.device.clone(),
                })
        })
    }
}

impl Drop for DescriptorPool {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        unsafe {
            self.device.destroy_descriptor_pool(self.handle, None);
        }
    }
}

/// Descriptor counts and limits used to create a [`DescriptorPool`].
///
/// See [`VkDescriptorPoolCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorPoolCreateInfo.html)
/// and [`VkDescriptorPoolSize`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorPoolSize.html).
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct DescriptorPoolInfo {
    pub(crate) acceleration_structure_count: u32,
    pub(crate) combined_image_sampler_count: u32,
    pub(crate) input_attachment_count: u32,
    pub(crate) max_sets: u32,
    pub(crate) sampled_image_count: u32,
    pub(crate) sampler_count: u32,
    pub(crate) storage_buffer_count: u32,
    pub(crate) storage_buffer_dynamic_count: u32,
    pub(crate) storage_image_count: u32,
    pub(crate) storage_texel_buffer_count: u32,
    pub(crate) uniform_buffer_count: u32,
    pub(crate) uniform_buffer_dynamic_count: u32,
    pub(crate) uniform_texel_buffer_count: u32,
}

impl DescriptorPoolInfo {
    fn for_layout(layout: &DescriptorSetLayout) -> Result<Self, DriverError> {
        let mut info = Self {
            max_sets: 1,
            ..Default::default()
        };

        for binding in &layout.info().bindings {
            let count = binding.descriptor_count;
            let destination = match binding.descriptor_type {
                vk::DescriptorType::ACCELERATION_STRUCTURE_KHR => {
                    &mut info.acceleration_structure_count
                }
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                    &mut info.combined_image_sampler_count
                }
                vk::DescriptorType::INPUT_ATTACHMENT => &mut info.input_attachment_count,
                vk::DescriptorType::SAMPLED_IMAGE => &mut info.sampled_image_count,
                vk::DescriptorType::SAMPLER => &mut info.sampler_count,
                vk::DescriptorType::STORAGE_BUFFER => &mut info.storage_buffer_count,
                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                    &mut info.storage_buffer_dynamic_count
                }
                vk::DescriptorType::STORAGE_IMAGE => &mut info.storage_image_count,
                vk::DescriptorType::STORAGE_TEXEL_BUFFER => &mut info.storage_texel_buffer_count,
                vk::DescriptorType::UNIFORM_BUFFER => &mut info.uniform_buffer_count,
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                    &mut info.uniform_buffer_dynamic_count
                }
                vk::DescriptorType::UNIFORM_TEXEL_BUFFER => &mut info.uniform_texel_buffer_count,
                _ => {
                    warn!(
                        "unsupported descriptor type {:?} in descriptor set layout",
                        binding.descriptor_type
                    );
                    return Err(DriverError::Unsupported);
                }
            };
            *destination = destination
                .checked_add(count)
                .ok_or(DriverError::InvalidData)?;
        }

        Ok(info)
    }
}

/// An immutable, explicitly populated Vulkan descriptor set.
///
/// The set owns its descriptor pool and every resource referenced by its writes. Cloning this value
/// shares the same allocation. To change descriptor contents, allocate a new set with
/// [`Self::alloc_and_update`].
#[derive(Clone)]
pub struct DescriptorSet {
    inner: Arc<DescriptorSetInner>,
}

impl DescriptorSet {
    /// Allocates one descriptor set and applies all supplied writes and copies.
    ///
    /// `pipeline` supplies the reflected layout selected by [`DescriptorSetInfo::set`]. A single
    /// [`DescriptorSetUpdateInfo`] or any owned collection of updates may be passed.
    /// Writes are applied before copies, matching `vkUpdateDescriptorSets` ordering.
    ///
    /// Input attachments, texel buffers, and dynamic buffer descriptors are rejected because they
    /// require render-pass-specific layouts, owned buffer views, or dynamic bind offsets.
    #[profiling::function]
    pub fn alloc_and_update<P, I>(
        pipeline: &P,
        info: impl Into<DescriptorSetInfo>,
        updates: I,
    ) -> Result<Self, DriverError>
    where
        P: DescriptorSetPipeline + ?Sized,
        I: IntoIterator<Item = DescriptorSetUpdateInfo>,
    {
        let info = info.into();
        let pipeline_info = descriptor_set_private::Sealed::descriptor_info(pipeline);
        let device = descriptor_set_private::Sealed::device(pipeline);
        let layout = pipeline_info
            .layouts
            .get(&info.set)
            .ok_or_else(|| {
                warn!("pipeline descriptor set {} does not exist", info.set);
                DriverError::InvalidData
            })?
            .clone();
        Self::validate_layout(&layout)?;
        let updates = updates.into_iter().collect::<Vec<_>>();
        let mut writes = Vec::with_capacity(updates.len());
        let mut copies = Vec::with_capacity(updates.len());

        for update in updates {
            let destination_info = Self::validate_binding(&layout, update.destination, 1)?;
            let descriptor_type = destination_info.descriptor_type;

            match update.update {
                DescriptorSetUpdate::AccelerationStructure(resource) => {
                    if descriptor_type != vk::DescriptorType::ACCELERATION_STRUCTURE_KHR
                        || !Device::is_same(device, &resource.buffer.device)
                        || !matches!(
                            resource.info.acceleration_structure_type,
                            vk::AccelerationStructureTypeKHR::GENERIC
                                | vk::AccelerationStructureTypeKHR::TOP_LEVEL
                        )
                    {
                        warn!("invalid acceleration structure descriptor write");
                        return Err(DriverError::InvalidData);
                    }

                    writes.push(PreparedWrite::AccelerationStructure {
                        destination: update.destination,
                        descriptor_type,
                        resource,
                    });
                }
                DescriptorSetUpdate::Buffer { buffer, range } => {
                    let (required_usage, required_alignment, max_range) = match descriptor_type {
                        vk::DescriptorType::STORAGE_BUFFER => (
                            vk::BufferUsageFlags::STORAGE_BUFFER,
                            device
                                .physical
                                .properties_v1_0
                                .limits
                                .min_storage_buffer_offset_alignment,
                            vk::DeviceSize::from(
                                device
                                    .physical
                                    .properties_v1_0
                                    .limits
                                    .max_storage_buffer_range,
                            ),
                        ),
                        vk::DescriptorType::UNIFORM_BUFFER => (
                            vk::BufferUsageFlags::UNIFORM_BUFFER,
                            device
                                .physical
                                .properties_v1_0
                                .limits
                                .min_uniform_buffer_offset_alignment,
                            vk::DeviceSize::from(
                                device
                                    .physical
                                    .properties_v1_0
                                    .limits
                                    .max_uniform_buffer_range,
                            ),
                        ),
                        _ => {
                            warn!("invalid buffer descriptor type {descriptor_type:?}");
                            return Err(DriverError::InvalidData);
                        }
                    };
                    let range_size = if range.end == vk::WHOLE_SIZE {
                        buffer.info.size.saturating_sub(range.start)
                    } else {
                        range.end.saturating_sub(range.start)
                    };

                    if !Device::is_same(device, &buffer.device)
                        || !buffer.info.usage.contains(required_usage)
                        || range.start >= buffer.info.size
                        || range.end != vk::WHOLE_SIZE
                            && (range.end <= range.start || range.end > buffer.info.size)
                        || required_alignment > 1 && range.start % required_alignment != 0
                        || range_size == 0
                        || range_size > max_range
                    {
                        warn!("invalid buffer descriptor write");
                        return Err(DriverError::InvalidData);
                    }

                    writes.push(PreparedWrite::Buffer {
                        destination: update.destination,
                        descriptor_type,
                        resource: buffer,
                        range,
                    });
                }
                DescriptorSetUpdate::Copy {
                    descriptor_count,
                    source,
                    source_binding,
                } => {
                    let destination_info =
                        Self::validate_binding(&layout, update.destination, descriptor_count)?;
                    let source_info = Self::validate_binding(
                        &source.inner.layout,
                        source_binding,
                        descriptor_count,
                    )?;

                    if !Device::is_same(device, source.device())
                        || source_info.descriptor_type != destination_info.descriptor_type
                        || destination_info.descriptor_type == vk::DescriptorType::SAMPLER
                            && destination_info.immutable_sampler.is_some()
                    {
                        warn!("invalid descriptor copy");
                        return Err(DriverError::InvalidData);
                    }

                    copies.push(PreparedCopy {
                        descriptor_count,
                        destination: update.destination,
                        source,
                        source_binding,
                    });
                }
                DescriptorSetUpdate::Image { image, view } => {
                    let image_layout = Self::image_layout(descriptor_type).ok_or_else(|| {
                        warn!("invalid image descriptor write");
                        DriverError::InvalidData
                    })?;
                    let required_usage = match descriptor_type {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                        | vk::DescriptorType::SAMPLED_IMAGE => vk::ImageUsageFlags::SAMPLED,
                        vk::DescriptorType::STORAGE_IMAGE => vk::ImageUsageFlags::STORAGE,
                        _ => {
                            warn!("invalid image descriptor type {descriptor_type:?}");
                            return Err(DriverError::InvalidData);
                        }
                    };
                    let image_aspects = format_aspect_mask(image.info.format);
                    let mip_level_count = if view.mip_level_count == vk::REMAINING_MIP_LEVELS {
                        image
                            .info
                            .mip_level_count
                            .saturating_sub(view.base_mip_level)
                    } else {
                        view.mip_level_count
                    };
                    let array_layer_count = if view.array_layer_count == vk::REMAINING_ARRAY_LAYERS
                    {
                        image
                            .info
                            .array_layer_count
                            .saturating_sub(view.base_array_layer)
                    } else {
                        view.array_layer_count
                    };

                    if !Device::is_same(device, &image.device)
                        || !image.info.usage.contains(required_usage)
                        || view.format == vk::Format::UNDEFINED
                        || view.aspect_mask.as_raw().count_ones() != 1
                        || !image_aspects.contains(view.aspect_mask)
                        || view.base_mip_level >= image.info.mip_level_count
                        || mip_level_count == 0
                        || mip_level_count > image.info.mip_level_count - view.base_mip_level
                        || view.base_array_layer >= image.info.array_layer_count
                        || array_layer_count == 0
                        || array_layer_count > image.info.array_layer_count - view.base_array_layer
                        || view.format != image.info.format
                            && !image
                                .info
                                .flags
                                .contains(vk::ImageCreateFlags::MUTABLE_FORMAT)
                    {
                        warn!("invalid image descriptor write");
                        return Err(DriverError::InvalidData);
                    }

                    let image_view = image.view(view)?;
                    writes.push(PreparedWrite::Image {
                        destination: update.destination,
                        descriptor_type,
                        image_layout,
                        image_view,
                        resource: image,
                    });
                }
            }
        }

        let descriptor_pool =
            DescriptorPool::create(device, DescriptorPoolInfo::for_layout(&layout)?)?;
        let descriptor_set = DescriptorPool::allocate_descriptor_set(&descriptor_pool, &layout)?;
        let handle = *descriptor_set;

        for write in &writes {
            unsafe {
                match write {
                    PreparedWrite::AccelerationStructure {
                        destination,
                        descriptor_type,
                        resource,
                    } => {
                        let acceleration_structures = [resource.handle];
                        let mut acceleration_structure_info =
                            vk::WriteDescriptorSetAccelerationStructureKHR::default()
                                .acceleration_structures(&acceleration_structures);
                        let write = vk::WriteDescriptorSet::default()
                            .dst_set(handle)
                            .dst_binding(destination.binding)
                            .dst_array_element(destination.array_element)
                            .descriptor_type(*descriptor_type)
                            .descriptor_count(1)
                            .push_next(&mut acceleration_structure_info);
                        device.update_descriptor_sets(slice::from_ref(&write), &[]);
                    }
                    PreparedWrite::Buffer {
                        destination,
                        descriptor_type,
                        resource,
                        range,
                    } => {
                        let range_size = if range.end == vk::WHOLE_SIZE {
                            vk::WHOLE_SIZE
                        } else {
                            range.end - range.start
                        };
                        let buffer_info = vk::DescriptorBufferInfo::default()
                            .buffer(resource.handle)
                            .offset(range.start)
                            .range(range_size);
                        let write = vk::WriteDescriptorSet::default()
                            .dst_set(handle)
                            .dst_binding(destination.binding)
                            .dst_array_element(destination.array_element)
                            .descriptor_type(*descriptor_type)
                            .buffer_info(slice::from_ref(&buffer_info));
                        device.update_descriptor_sets(slice::from_ref(&write), &[]);
                    }
                    PreparedWrite::Image {
                        destination,
                        descriptor_type,
                        image_layout,
                        image_view,
                        ..
                    } => {
                        let image_info = vk::DescriptorImageInfo::default()
                            .image_layout(*image_layout)
                            .image_view(*image_view);
                        let write = vk::WriteDescriptorSet::default()
                            .dst_set(handle)
                            .dst_binding(destination.binding)
                            .dst_array_element(destination.array_element)
                            .descriptor_type(*descriptor_type)
                            .image_info(slice::from_ref(&image_info));
                        device.update_descriptor_sets(slice::from_ref(&write), &[]);
                    }
                }
            }
        }

        if !copies.is_empty() {
            let copy_infos = copies
                .iter()
                .map(|copy| {
                    vk::CopyDescriptorSet::default()
                        .src_set(copy.source.handle())
                        .src_binding(copy.source_binding.binding)
                        .src_array_element(copy.source_binding.array_element)
                        .dst_set(handle)
                        .dst_binding(copy.destination.binding)
                        .dst_array_element(copy.destination.array_element)
                        .descriptor_count(copy.descriptor_count)
                })
                .collect::<Vec<_>>();

            unsafe {
                device.update_descriptor_sets(&[], &copy_infos);
            }
        }

        let resources = writes
            .into_iter()
            .map(|write| match write {
                PreparedWrite::AccelerationStructure { resource, .. } => {
                    DescriptorSetResource::AccelerationStructure(resource)
                }
                PreparedWrite::Buffer { resource, .. } => DescriptorSetResource::Buffer(resource),
                PreparedWrite::Image { resource, .. } => DescriptorSetResource::Image(resource),
            })
            .chain(
                copies
                    .into_iter()
                    .map(|copy| DescriptorSetResource::DescriptorSet(copy.source)),
            )
            .collect();

        Ok(Self {
            inner: Arc::new(DescriptorSetInner {
                descriptor_set,
                _descriptor_pool: descriptor_pool,
                info,
                layout,
                _resources: resources,
            }),
        })
    }

    /// The device which owns this descriptor set.
    pub fn device(&self) -> &Device {
        self.inner.layout.device()
    }

    /// The native Vulkan descriptor set handle.
    pub fn handle(&self) -> vk::DescriptorSet {
        *self.inner.descriptor_set
    }

    /// The information used to allocate this descriptor set.
    pub fn info(&self) -> DescriptorSetInfo {
        self.inner.info
    }

    /// Sets the debugging name assigned to this descriptor set.
    pub fn set_debug_name(&self, name: impl AsRef<str>) {
        Device::try_set_debug_utils_object_name(self.device(), self.handle(), &name);
        Device::try_set_private_data_object_name(
            self.device(),
            vk::ObjectType::DESCRIPTOR_SET,
            self.handle(),
            &name,
        );
    }

    pub(crate) fn is_compatible(&self, set: u32, layout: &DescriptorSetLayout) -> bool {
        self.inner.info.set == set && self.inner.layout.is_same(layout)
    }

    fn validate_layout(layout: &DescriptorSetLayout) -> Result<(), DriverError> {
        for binding in &layout.info().bindings {
            if matches!(
                binding.descriptor_type,
                vk::DescriptorType::INPUT_ATTACHMENT
                    | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                    | vk::DescriptorType::STORAGE_TEXEL_BUFFER
                    | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                    | vk::DescriptorType::UNIFORM_TEXEL_BUFFER
            ) {
                warn!(
                    "descriptor set layout contains unsupported descriptor type {:?}",
                    binding.descriptor_type
                );
                return Err(DriverError::Unsupported);
            }
        }

        Ok(())
    }

    fn image_layout(descriptor_type: vk::DescriptorType) -> Option<vk::ImageLayout> {
        match descriptor_type {
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER | vk::DescriptorType::SAMPLED_IMAGE => {
                Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            }
            vk::DescriptorType::STORAGE_IMAGE => Some(vk::ImageLayout::GENERAL),
            _ => None,
        }
    }

    fn validate_binding(
        layout: &DescriptorSetLayout,
        binding: DescriptorSetBinding,
        descriptor_count: u32,
    ) -> Result<&super::descriptor_set_layout::DescriptorSetLayoutBindingInfo, DriverError> {
        let binding_info = layout.info().binding(binding.binding).ok_or_else(|| {
            warn!("descriptor binding {} does not exist", binding.binding);
            DriverError::InvalidData
        })?;
        let end = binding
            .array_element
            .checked_add(descriptor_count)
            .ok_or(DriverError::InvalidData)?;

        if descriptor_count == 0 || end > binding_info.descriptor_count {
            warn!(
                "descriptor binding {} array range {}..{} is out of bounds",
                binding.binding, binding.array_element, end
            );
            return Err(DriverError::InvalidData);
        }

        Ok(binding_info)
    }
}

impl Debug for DescriptorSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut result = f.debug_struct(stringify!(DescriptorSet));

        if let Some(debug_name) = &Device::private_data_object_name(
            self.device(),
            vk::ObjectType::DESCRIPTOR_SET,
            self.handle(),
        ) {
            result.field("debug_name", debug_name);
        }

        result
            .field("handle", &self.handle())
            .field("info", &self.info())
            .finish()
    }
}

/// Identifies one binding and array element within a descriptor set.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DescriptorSetBinding {
    /// The descriptor binding index.
    pub binding: u32,

    /// The array element within `binding`.
    pub array_element: u32,
}

impl From<u32> for DescriptorSetBinding {
    fn from(binding: u32) -> Self {
        Self {
            binding,
            array_element: 0,
        }
    }
}

impl From<(u32, u32)> for DescriptorSetBinding {
    fn from((binding, array_element): (u32, u32)) -> Self {
        Self {
            binding,
            array_element,
        }
    }
}

impl From<(u32, [u32; 1])> for DescriptorSetBinding {
    fn from((binding, [array_element]): (u32, [u32; 1])) -> Self {
        Self {
            binding,
            array_element,
        }
    }
}

/// Information used to allocate a [`DescriptorSet`].
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct DescriptorSetInfo {
    /// The pipeline descriptor set index whose layout will be used.
    pub set: u32,
}

impl DescriptorSetInfo {
    /// Creates a default descriptor set information builder.
    pub const fn builder() -> DescriptorSetInfoBuilder {
        DescriptorSetInfoBuilder { set: 0 }
    }

    /// Converts this information into a builder.
    pub const fn into_builder(self) -> DescriptorSetInfoBuilder {
        DescriptorSetInfoBuilder { set: self.set }
    }
}

impl From<DescriptorSetInfoBuilder> for DescriptorSetInfo {
    fn from(info: DescriptorSetInfoBuilder) -> Self {
        info.build()
    }
}

/// Builder for [`DescriptorSetInfo`].
#[derive(Clone, Copy, Debug, Default)]
pub struct DescriptorSetInfoBuilder {
    set: u32,
}

impl DescriptorSetInfoBuilder {
    /// Selects the pipeline descriptor set index whose layout will be used.
    pub const fn set(mut self, set: u32) -> Self {
        self.set = set;
        self
    }

    /// Builds descriptor set information.
    pub const fn build(self) -> DescriptorSetInfo {
        DescriptorSetInfo { set: self.set }
    }
}

struct DescriptorSetInner {
    descriptor_set: RawDescriptorSet,
    _descriptor_pool: DescriptorPool,
    info: DescriptorSetInfo,
    layout: DescriptorSetLayout,
    _resources: Box<[DescriptorSetResource]>,
}

impl Drop for DescriptorSetInner {
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        Device::try_clear_private_data_object_name(
            self.layout.device(),
            vk::ObjectType::DESCRIPTOR_SET,
            *self.descriptor_set,
        );
    }
}

#[allow(dead_code)]
enum DescriptorSetResource {
    AccelerationStructure(Arc<AccelerationStructure>),
    Buffer(Arc<Buffer>),
    DescriptorSet(DescriptorSet),
    Image(Arc<Image>),
}

#[derive(Clone, Debug)]
enum DescriptorSetUpdate {
    AccelerationStructure(Arc<AccelerationStructure>),
    Buffer {
        buffer: Arc<Buffer>,
        range: BufferSubresourceRange,
    },
    Copy {
        descriptor_count: u32,
        source: DescriptorSet,
        source_binding: DescriptorSetBinding,
    },
    Image {
        image: Arc<Image>,
        view: ImageViewInfo,
    },
}

/// One write or copy performed while allocating a [`DescriptorSet`].
#[derive(Clone, Debug)]
pub struct DescriptorSetUpdateInfo {
    destination: DescriptorSetBinding,
    update: DescriptorSetUpdate,
}

impl DescriptorSetUpdateInfo {
    /// Writes an acceleration structure descriptor.
    pub fn acceleration_structure(
        destination: impl Into<DescriptorSetBinding>,
        acceleration_structure: &Arc<AccelerationStructure>,
    ) -> Self {
        Self {
            destination: destination.into(),
            update: DescriptorSetUpdate::AccelerationStructure(acceleration_structure.clone()),
        }
    }

    /// Writes a whole-buffer descriptor.
    pub fn buffer(destination: impl Into<DescriptorSetBinding>, buffer: &Arc<Buffer>) -> Self {
        Self::buffer_range(destination, buffer, buffer.info)
    }

    /// Writes a descriptor for a range of a buffer.
    pub fn buffer_range(
        destination: impl Into<DescriptorSetBinding>,
        buffer: &Arc<Buffer>,
        range: impl Into<BufferSubresourceRange>,
    ) -> Self {
        Self {
            destination: destination.into(),
            update: DescriptorSetUpdate::Buffer {
                buffer: buffer.clone(),
                range: range.into(),
            },
        }
    }

    /// Copies one descriptor from an existing descriptor set.
    pub fn copy(
        source: &DescriptorSet,
        source_binding: impl Into<DescriptorSetBinding>,
        destination: impl Into<DescriptorSetBinding>,
    ) -> Self {
        Self::copy_many(source, source_binding, destination, 1)
    }

    /// Copies consecutive descriptors from an existing descriptor set.
    ///
    /// The source and destination ranges must each remain within one reflected binding.
    pub fn copy_many(
        source: &DescriptorSet,
        source_binding: impl Into<DescriptorSetBinding>,
        destination: impl Into<DescriptorSetBinding>,
        descriptor_count: u32,
    ) -> Self {
        Self {
            destination: destination.into(),
            update: DescriptorSetUpdate::Copy {
                descriptor_count,
                source: source.clone(),
                source_binding: source_binding.into(),
            },
        }
    }

    /// Writes a descriptor using the image's default view.
    ///
    /// Depth/stencil formats require [`Self::image_view`] with exactly one selected aspect.
    pub fn image(destination: impl Into<DescriptorSetBinding>, image: &Arc<Image>) -> Self {
        Self::image_view(destination, image, image.info)
    }

    /// Writes a descriptor using a specific image view.
    pub fn image_view(
        destination: impl Into<DescriptorSetBinding>,
        image: &Arc<Image>,
        view: impl Into<ImageViewInfo>,
    ) -> Self {
        Self {
            destination: destination.into(),
            update: DescriptorSetUpdate::Image {
                image: image.clone(),
                view: view.into(),
            },
        }
    }
}

impl IntoIterator for DescriptorSetUpdateInfo {
    type Item = Self;
    type IntoIter = iter::Once<Self>;

    fn into_iter(self) -> Self::IntoIter {
        iter::once(self)
    }
}

impl IntoIterator for &DescriptorSetUpdateInfo {
    type Item = DescriptorSetUpdateInfo;
    type IntoIter = iter::Once<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        iter::once(self.clone())
    }
}

/// A compute, graphics, or ray tracing pipeline that can provide a descriptor set layout.
#[doc(hidden)]
pub trait DescriptorSetPipeline: descriptor_set_private::Sealed {}

macro_rules! descriptor_set_pipeline {
    ($pipeline:ty) => {
        #[allow(private_interfaces)]
        impl descriptor_set_private::Sealed for $pipeline {
            fn descriptor_info(&self) -> &PipelineDescriptorInfo {
                &self.inner.descriptor_info
            }

            fn device(&self) -> &Device {
                self.device()
            }
        }

        impl DescriptorSetPipeline for $pipeline {}
    };
}

descriptor_set_pipeline!(ComputePipeline);
descriptor_set_pipeline!(GraphicsPipeline);
descriptor_set_pipeline!(RayTracingPipeline);

#[allow(private_interfaces)]
mod descriptor_set_private {
    use super::{Device, PipelineDescriptorInfo};

    pub trait Sealed {
        fn descriptor_info(&self) -> &PipelineDescriptorInfo;

        fn device(&self) -> &Device;
    }
}

enum PreparedWrite {
    AccelerationStructure {
        destination: DescriptorSetBinding,
        descriptor_type: vk::DescriptorType,
        resource: Arc<AccelerationStructure>,
    },
    Buffer {
        destination: DescriptorSetBinding,
        descriptor_type: vk::DescriptorType,
        resource: Arc<Buffer>,
        range: BufferSubresourceRange,
    },
    Image {
        destination: DescriptorSetBinding,
        descriptor_type: vk::DescriptorType,
        image_layout: vk::ImageLayout,
        image_view: vk::ImageView,
        resource: Arc<Image>,
    },
}

struct PreparedCopy {
    descriptor_count: u32,
    destination: DescriptorSetBinding,
    source: DescriptorSet,
    source_binding: DescriptorSetBinding,
}

#[derive(Debug)]
pub(crate) struct RawDescriptorSet {
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    device: Device,
}

impl Deref for RawDescriptorSet {
    type Target = vk::DescriptorSet;

    fn deref(&self) -> &Self::Target {
        &self.descriptor_set
    }
}

impl Drop for RawDescriptorSet {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        if let Err(err) = unsafe {
            self.device
                .free_descriptor_sets(self.descriptor_pool, slice::from_ref(&self.descriptor_set))
        } {
            warn!("unable to free descriptor set: {err}");
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn descriptor_set_binding_conversions() {
        assert_eq!(
            DescriptorSetBinding::from(3),
            DescriptorSetBinding {
                binding: 3,
                array_element: 0,
            }
        );
        assert_eq!(
            DescriptorSetBinding::from((3, [7])),
            DescriptorSetBinding {
                binding: 3,
                array_element: 7,
            }
        );
    }

    #[test]
    fn descriptor_set_info_builder() {
        let info = DescriptorSetInfo::builder().set(2).build();

        assert_eq!(info.set, 2);
        assert_eq!(info, info.into_builder().build());
        assert_eq!(
            std::mem::size_of::<DescriptorSetInfo>(),
            std::mem::size_of::<u32>()
        );
    }

    #[test]
    fn empty_descriptor_set_updates_are_inferred() {
        fn update_count(updates: impl IntoIterator<Item = DescriptorSetUpdateInfo>) -> usize {
            updates.into_iter().count()
        }

        assert_eq!(update_count([]), 0);
        assert_eq!(update_count(None), 0);
    }
}
