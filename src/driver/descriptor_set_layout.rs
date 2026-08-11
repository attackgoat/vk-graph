use {
    super::{
        DriverError,
        device::Device,
        shader::{Sampler, SamplerInfo},
    },
    ash::vk,
    log::warn,
    std::{
        collections::{HashMap, hash_map::Entry},
        fmt::{Debug, Formatter},
        ops::Deref,
        sync::{Arc, Mutex, OnceLock, Weak},
        thread::panicking,
    },
};

#[derive(Clone)]
pub(crate) struct DescriptorSetLayout {
    inner: Arc<DescriptorSetLayoutInner>,
}

impl DescriptorSetLayout {
    #[profiling::function]
    pub(crate) fn get_or_create(
        device: &Device,
        info: DescriptorSetLayoutInfo,
    ) -> Result<Self, DriverError> {
        type Cache = HashMap<(usize, DescriptorSetLayoutInfo), Weak<DescriptorSetLayoutInner>>;

        static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();

        let cache = CACHE.get_or_init(Default::default);
        let mut cache = cache.lock().expect("poisoned descriptor set layout cache");
        cache.retain(|_, layout| layout.strong_count() != 0);
        let key = (Device::identity(device), info.clone());
        if let Some(inner) = cache.get(&key).and_then(Weak::upgrade) {
            return Ok(Self { inner });
        }

        let mut samplers = HashMap::<SamplerInfo, Sampler>::new();
        for binding in &info.bindings {
            let Some(sampler_info) = binding.immutable_sampler else {
                continue;
            };

            if let Entry::Vacant(entry) = samplers.entry(sampler_info) {
                entry.insert(Sampler::create(device, sampler_info)?);
            }
        }

        let immutable_samplers = info
            .bindings
            .iter()
            .map(|binding| {
                binding.immutable_sampler.map(|sampler_info| {
                    let sampler = *samplers
                        .get(&sampler_info)
                        .expect("missing immutable sampler")
                        .deref();
                    vec![sampler; binding.descriptor_count as usize].into_boxed_slice()
                })
            })
            .collect::<Box<_>>();
        let bindings = info
            .bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| {
                let binding_info = vk::DescriptorSetLayoutBinding::default()
                    .binding(binding.binding)
                    .descriptor_count(binding.descriptor_count)
                    .descriptor_type(binding.descriptor_type)
                    .stage_flags(binding.stage_flags);

                if let Some(immutable_samplers) = immutable_samplers[index].as_deref() {
                    binding_info.immutable_samplers(immutable_samplers)
                } else {
                    binding_info
                }
            })
            .collect::<Box<_>>();
        let binding_flags = info
            .bindings
            .iter()
            .map(|binding| binding.binding_flags)
            .collect::<Box<_>>();
        let mut binding_flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
        let mut create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        if binding_flags.iter().any(|flags| !flags.is_empty()) {
            create_info = create_info.push_next(&mut binding_flags_info);
        }

        let handle = unsafe {
            device
                .create_descriptor_set_layout(&create_info, None)
                .map_err(|err| {
                    warn!("unable to create descriptor set layout: {err}");
                    match err {
                        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                        | vk::Result::ERROR_OUT_OF_HOST_MEMORY => DriverError::OutOfMemory,
                        _ => DriverError::Unsupported,
                    }
                })?
        };
        let inner = Arc::new(DescriptorSetLayoutInner {
            device: device.clone(),
            handle,
            info,
            samplers: samplers.into_values().collect(),
        });

        cache.insert(key, Arc::downgrade(&inner));

        Ok(Self { inner })
    }

    pub(crate) fn device(&self) -> &Device {
        &self.inner.device
    }

    pub(crate) fn handle(&self) -> vk::DescriptorSetLayout {
        self.inner.handle
    }

    pub(crate) fn info(&self) -> &DescriptorSetLayoutInfo {
        &self.inner.info
    }

    pub(crate) fn is_same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub(crate) fn set_debug_name(&self, name: impl AsRef<str>) {
        Device::try_set_debug_utils_object_name(&self.inner.device, self.inner.handle, &name);
        Device::try_set_private_data_object_name(
            &self.inner.device,
            vk::ObjectType::DESCRIPTOR_SET_LAYOUT,
            self.inner.handle,
            &name,
        );
    }
}

impl Debug for DescriptorSetLayout {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut result = f.debug_struct(stringify!(DescriptorSetLayout));

        if let Some(debug_name) = &Device::private_data_object_name(
            &self.inner.device,
            vk::ObjectType::DESCRIPTOR_SET_LAYOUT,
            self.inner.handle,
        ) {
            result.field("debug_name", debug_name);
        }

        result.field("handle", &self.inner.handle).finish()
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DescriptorSetLayoutBindingInfo {
    pub binding: u32,
    pub binding_flags: vk::DescriptorBindingFlags,
    pub descriptor_count: u32,
    pub descriptor_type: vk::DescriptorType,
    pub immutable_sampler: Option<SamplerInfo>,
    pub stage_flags: vk::ShaderStageFlags,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DescriptorSetLayoutInfo {
    pub bindings: Box<[DescriptorSetLayoutBindingInfo]>,
}

impl DescriptorSetLayoutInfo {
    pub fn binding(&self, binding: u32) -> Option<&DescriptorSetLayoutBindingInfo> {
        self.bindings
            .binary_search_by_key(&binding, |info| info.binding)
            .ok()
            .map(|index| &self.bindings[index])
    }
}

struct DescriptorSetLayoutInner {
    device: Device,
    handle: vk::DescriptorSetLayout,
    info: DescriptorSetLayoutInfo,
    samplers: Box<[Sampler]>,
}

impl Debug for DescriptorSetLayoutInner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(stringify!(DescriptorSetLayoutInner))
            .field("handle", &self.handle)
            .field("info", &self.info)
            .field("samplers", &self.samplers)
            .finish_non_exhaustive()
    }
}

impl Drop for DescriptorSetLayoutInner {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        Device::try_clear_private_data_object_name(
            &self.device,
            vk::ObjectType::DESCRIPTOR_SET_LAYOUT,
            self.handle,
        );

        unsafe {
            self.device.destroy_descriptor_set_layout(self.handle, None);
        }
    }
}
