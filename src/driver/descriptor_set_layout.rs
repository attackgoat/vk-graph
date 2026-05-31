use {
    super::{DriverError, device::Device},
    ash::vk,
    log::warn,
    std::thread::panicking,
};

#[derive(Debug)]
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
}

impl Drop for DescriptorSetLayout {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        unsafe {
            self.device.destroy_descriptor_set_layout(self.handle, None);
        }
    }
}
