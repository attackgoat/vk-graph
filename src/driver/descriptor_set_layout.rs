use {
    super::{DriverError, device::Device},
    ash::vk,
    log::warn,
    std::{sync::Arc, thread::panicking},
};

#[derive(Debug)]
#[readonly::make]
pub struct DescriptorSetLayout {
    pub device: Arc<Device>,
    pub handle: vk::DescriptorSetLayout,
}

impl DescriptorSetLayout {
    #[profiling::function]
    pub fn create(
        device: &Arc<Device>,
        info: &vk::DescriptorSetLayoutCreateInfo,
    ) -> Result<Self, DriverError> {
        let device = Arc::clone(device);
        let handle = unsafe {
            device
                .create_descriptor_set_layout(info, None)
                .map_err(|err| {
                    warn!("{err}");

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
