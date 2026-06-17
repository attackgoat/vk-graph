use {
    super::{DriverError, device::Device},
    ash::vk,
    log::warn,
    std::{
        fmt::{Debug, Formatter},
        thread::panicking,
    },
};

#[read_only::cast]
pub struct DescriptorSetLayout {
    pub device: Device,
    pub handle: vk::DescriptorSetLayout,
}

impl DescriptorSetLayout {
    #[profiling::function]
    pub fn create(
        device: &Device,
        info: &vk::DescriptorSetLayoutCreateInfo,
    ) -> Result<Self, DriverError> {
        let device = device.clone();
        let handle = unsafe {
            device
                .create_descriptor_set_layout(info, None)
                .map_err(|err| {
                    warn!("unable to create descriptor set layout: {err}");

                    DriverError::Unsupported
                })
        }?;

        Ok(Self { device, handle })
    }

    /// Sets the debugging name assigned to this descriptor set layout.
    pub fn set_debug_name(&self, name: impl AsRef<str>) {
        Device::try_set_debug_utils_object_name(&self.device, self.handle, &name);
        Device::try_set_private_data_object_name(
            &self.device,
            vk::ObjectType::DESCRIPTOR_SET_LAYOUT,
            self.handle,
            &name,
        );
    }
}

impl Debug for DescriptorSetLayout {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut res = f.debug_struct(stringify!(DescriptorSetLayout));

        if let Some(debug_name) = &Device::private_data_object_name(
            &self.device,
            vk::ObjectType::DESCRIPTOR_SET_LAYOUT,
            self.handle,
        ) {
            res.field("debug_name", debug_name);
        }

        res.field("handle", &self.handle).finish()
    }
}

impl Drop for DescriptorSetLayout {
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
