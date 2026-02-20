use {
    super::{DescriptorSetLayout, DriverError, device::Device},
    ash::vk,
    log::warn,
    std::{ops::Deref, slice, thread::panicking},
};

#[derive(Debug)]
#[readonly::make]
pub struct DescriptorPool {
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
    pub info: DescriptorPoolInfo,
}

impl DescriptorPool {
    #[profiling::function]
    pub fn create(
        device: &Device,
        info: impl Into<DescriptorPoolInfo>,
    ) -> Result<Self, DriverError> {
        let device = device.clone();
        let info = info.into();

        let mut pool_sizes = [vk::DescriptorPoolSize {
            ty: Default::default(),
            descriptor_count: 0,
        }; 11];
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
            warn!("{err}");

            DriverError::Unsupported
        })?;

        Ok(Self {
            device,
            handle,
            info,
        })
    }

    pub fn allocate_descriptor_set(
        this: &Self,
        layout: &DescriptorSetLayout,
    ) -> Result<DescriptorSet, DriverError> {
        Ok(Self::allocate_descriptor_sets(this, layout, 1)?
            .next()
            .unwrap())
    }

    #[profiling::function]
    pub fn allocate_descriptor_sets<'a>(
        &'a self,
        layout: &DescriptorSetLayout,
        count: u32,
    ) -> Result<impl Iterator<Item = DescriptorSet> + 'a, DriverError> {
        let mut create_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.handle)
            .set_layouts(slice::from_ref(&layout.handle));
        create_info.descriptor_set_count = count;

        Ok(unsafe {
            self.device
                .allocate_descriptor_sets(&create_info)
                .map_err(|err| {
                    use {DriverError::*, vk::Result as vk};

                    warn!("{err}");

                    match err {
                        e if e == vk::ERROR_FRAGMENTED_POOL => InvalidData,
                        e if e == vk::ERROR_OUT_OF_DEVICE_MEMORY => OutOfMemory,
                        e if e == vk::ERROR_OUT_OF_HOST_MEMORY => OutOfMemory,
                        e if e == vk::ERROR_OUT_OF_POOL_MEMORY => OutOfMemory,
                        _ => Unsupported,
                    }
                })?
                .into_iter()
                .map(move |descriptor_set| DescriptorSet {
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

#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct DescriptorPoolInfo {
    pub acceleration_structure_count: u32,
    pub combined_image_sampler_count: u32,
    pub input_attachment_count: u32,
    pub max_sets: u32,
    pub sampled_image_count: u32,
    pub sampler_count: u32,
    pub storage_buffer_count: u32,
    pub storage_buffer_dynamic_count: u32,
    pub storage_image_count: u32,
    pub storage_texel_buffer_count: u32,
    pub uniform_buffer_count: u32,
    pub uniform_buffer_dynamic_count: u32,
    pub uniform_texel_buffer_count: u32,
}

impl DescriptorPoolInfo {
    pub fn is_empty(&self) -> bool {
        self.acceleration_structure_count
            + self.combined_image_sampler_count
            + self.input_attachment_count
            + self.sampled_image_count
            + self.sampler_count
            + self.storage_buffer_count
            + self.storage_buffer_dynamic_count
            + self.storage_image_count
            + self.storage_texel_buffer_count
            + self.uniform_buffer_count
            + self.uniform_buffer_dynamic_count
            + self.uniform_texel_buffer_count
            == 0
    }
}

#[derive(Debug)]
pub struct DescriptorSet {
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    device: Device,
}

impl Deref for DescriptorSet {
    type Target = vk::DescriptorSet;

    fn deref(&self) -> &Self::Target {
        &self.descriptor_set
    }
}

impl Drop for DescriptorSet {
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
