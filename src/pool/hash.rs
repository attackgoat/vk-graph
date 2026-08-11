//! Pool which requests by exactly matching the information before creating new resources.

use {
    super::{
        Cache, Lease, Pool, PoolConfig,
        garbage_collector::{CollectResources, ResourceRequests},
    },
    crate::driver::{
        DriverError,
        accel_struct::{AccelerationStructure, AccelerationStructureInfo},
        buffer::{Buffer, BufferInfo},
        cmd_buf::{CommandBuffer, CommandBufferInfo},
        descriptor_set::{DescriptorPool, DescriptorPoolInfo},
        device::Device,
        image::{Image, ImageInfo},
        render_pass::{RenderPass, RenderPassInfo},
    },
    log::debug,
    paste::paste,
    std::{collections::HashMap, sync::Arc},
};

#[cfg(feature = "parking_lot")]
use parking_lot::Mutex;

#[cfg(not(feature = "parking_lot"))]
use std::sync::Mutex;

/// A high-performance resource allocator.
///
/// # Bucket Strategy
///
/// The information for each resource request is the key for a `HashMap` of buckets. If no bucket
/// exists with the exact information provided a new bucket is created.
///
/// In practice this means that for a [`PoolConfig::image_capacity`] of `4`, requests for a
/// 1024x1024 image with certain attributes will store a maximum of `4` such images. Requests for
/// any image having a different size or attributes will store an additional maximum of `4` images.
///
/// # Memory Management
///
/// If requests for varying resources are common [`HashPool::clear_images_by_info`] and other
/// memory management functions are necessary in order to avoid using all available device memory.
#[derive(Debug)]
#[read_only::cast]
pub struct HashPool {
    acceleration_structure_cache: HashMap<AccelerationStructureInfo, Cache<AccelerationStructure>>,
    buffer_cache: HashMap<BufferInfo, Cache<Buffer>>,
    command_buffer_cache: HashMap<u32, Cache<CommandBuffer>>,
    descriptor_pool_cache: HashMap<DescriptorPoolInfo, Cache<DescriptorPool>>,

    /// The device which owns this pool.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub device: Device,

    image_cache: HashMap<ImageInfo, Cache<Image>>,

    /// Information used to create this pool.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub info: PoolConfig,

    render_pass_cache: HashMap<RenderPassInfo, Cache<RenderPass>>,
}

impl HashPool {
    /// Constructs a new `HashPool`.
    pub fn new(device: &Device) -> Self {
        Self::with_capacity(device, PoolConfig::default())
    }

    /// Constructs a new `HashPool` with the given capacity information.
    pub fn with_capacity(device: &Device, info: impl Into<PoolConfig>) -> Self {
        let info: PoolConfig = info.into();
        let device = device.clone();

        Self {
            acceleration_structure_cache: Default::default(),
            buffer_cache: Default::default(),
            command_buffer_cache: Default::default(),
            descriptor_pool_cache: Default::default(),
            device,
            image_cache: Default::default(),
            info,
            render_pass_cache: Default::default(),
        }
    }

    /// Clears the pool, removing all resources.
    pub fn clear(&mut self) {
        self.clear_accel_structs();
        self.clear_buffers();
        self.clear_images();
    }
}

impl CollectResources for HashPool {
    fn collect_resources(&mut self, requests: &ResourceRequests) {
        self.acceleration_structure_cache
            .retain(|info, _| requests.accel_structs.contains(info));
        self.buffer_cache
            .retain(|info, _| requests.buffers.contains(info));
        self.image_cache
            .retain(|info, _| requests.images.contains(info));
    }
}

macro_rules! resource_mgmt_fns {
    ($fn_plural:literal, $doc_singular:literal, $ty:ty, $field:ident) => {
        paste! {
            impl HashPool {
                #[doc = "Clears the pool of " $doc_singular " resources."]
                pub fn [<clear_ $fn_plural>](&mut self) {
                    self.$field.clear();
                }

                #[doc = "Clears the pool of all " $doc_singular " resources matching the given
information."]
                pub fn [<clear_ $fn_plural _by_info>](
                    &mut self,
                    info: impl Into<$ty>,
                ) {
                    self.$field.remove(&info.into());
                }

                #[doc = "Retains only the " $doc_singular " resources specified by the predicate.\n
\nIn other words, remove all " $doc_singular " resources for which `f(" $ty ")` returns `false`.\n
\n"]
                /// The elements are visited in unsorted (and unspecified) order.
                ///
                /// # Performance
                ///
                /// Provides the same performance guarantees as
                /// [`HashMap::retain`](HashMap::retain).
                pub fn [<retain_ $fn_plural>]<F>(&mut self, mut f: F)
                where
                    F: FnMut($ty) -> bool,
                {
                    self.$field.retain(|&info, _| f(info))
                }
            }
        }
    };
}

resource_mgmt_fns!(
    "accel_structs",
    "acceleration structure",
    AccelerationStructureInfo,
    acceleration_structure_cache
);
resource_mgmt_fns!("buffers", "buffer", BufferInfo, buffer_cache);
resource_mgmt_fns!("images", "image", ImageInfo, image_cache);

impl Pool<CommandBufferInfo, CommandBuffer> for HashPool {
    #[profiling::function]
    fn resource(&mut self, info: CommandBufferInfo) -> Result<Lease<CommandBuffer>, DriverError> {
        let cache_ref = self
            .command_buffer_cache
            .entry(info.queue_family_index)
            .or_insert_with(PoolConfig::default_cache);
        let item = {
            #[cfg_attr(not(feature = "parking_lot"), allow(unused_mut))]
            let mut cache = cache_ref.lock();

            #[cfg(not(feature = "parking_lot"))]
            let mut cache = cache.expect("poisoned cache lock");

            cache.pop()
        }
        .map(Ok)
        .unwrap_or_else(|| {
            debug!("Creating new {}", stringify!(CommandBuffer));

            CommandBuffer::create(&self.device, info)
        })?;

        // Drop anything we were holding from the last submission
        //item.wait_until_executed()?;

        Ok(Lease::new(Arc::downgrade(cache_ref), item))
    }
}

impl Pool<DescriptorPoolInfo, DescriptorPool> for HashPool {
    #[profiling::function]
    fn resource(&mut self, info: DescriptorPoolInfo) -> Result<Lease<DescriptorPool>, DriverError> {
        let cache_ref = self
            .descriptor_pool_cache
            .entry(info.clone())
            .or_insert_with(PoolConfig::default_cache);
        let item = {
            #[cfg_attr(not(feature = "parking_lot"), allow(unused_mut))]
            let mut cache = cache_ref.lock();

            #[cfg(not(feature = "parking_lot"))]
            let mut cache = cache.expect("poisoned cache lock");

            cache.pop()
        }
        .map(Ok)
        .unwrap_or_else(|| {
            debug!("Creating new {}", stringify!(DescriptorPool));

            DescriptorPool::create(&self.device, info)
        })?;

        Ok(Lease::new(Arc::downgrade(cache_ref), item))
    }
}

impl Pool<RenderPassInfo, RenderPass> for HashPool {
    #[profiling::function]
    fn resource(&mut self, info: RenderPassInfo) -> Result<Lease<RenderPass>, DriverError> {
        let cache_ref = if let Some(cache) = self.render_pass_cache.get(&info) {
            cache
        } else {
            // We tried to get the cache first in order to avoid this clone
            self.render_pass_cache
                .entry(info.clone())
                .or_insert_with(PoolConfig::default_cache)
        };
        let item = {
            #[cfg_attr(not(feature = "parking_lot"), allow(unused_mut))]
            let mut cache = cache_ref.lock();

            #[cfg(not(feature = "parking_lot"))]
            let mut cache = cache.expect("poisoned cache lock");

            cache.pop()
        }
        .map(Ok)
        .unwrap_or_else(|| {
            debug!("Creating new {}", stringify!(RenderPass));

            RenderPass::create(&self.device, info)
        })?;

        Ok(Lease::new(Arc::downgrade(cache_ref), item))
    }
}

// Enable requesting items using their basic info
macro_rules! lease {
    ($info:ident => $item:ident, $capacity:ident) => {
        paste::paste! {
            impl Pool<$info, $item> for HashPool {
                #[profiling::function]
                fn resource(&mut self, info: $info) -> Result<Lease<$item>, DriverError> {
                    let cache_ref = self.[<$item:snake _cache>].entry(info)
                        .or_insert_with(|| {
                            Cache::new(Mutex::new(Vec::with_capacity(self.info.$capacity)))
                        });
                    let item = {
                        #[cfg_attr(not(feature = "parking_lot"), allow(unused_mut))]
                        let mut cache = cache_ref.lock();

                        #[cfg(not(feature = "parking_lot"))]
                        let mut cache = cache.expect("poisoned cache lock");

                        cache.pop()
                    }
                    .map(Ok)
                    .unwrap_or_else(|| {
                        debug!("Creating new {}", stringify!($item));

                        $item::create(&self.device, info)
                    })?;

                    Ok(Lease::new(Arc::downgrade(cache_ref), item))
                }
            }
        }
    };
}

lease!(AccelerationStructureInfo => AccelerationStructure, accel_struct_capacity);
lease!(BufferInfo => Buffer, buffer_capacity);
lease!(ImageInfo => Image, image_capacity);

#[cfg(test)]
mod test {
    use {
        super::*,
        crate::{
            driver::device::{Device, DeviceInfo},
            pool::garbage_collector::GarbageCollector,
        },
        ash::vk,
    };

    #[test]
    #[ignore = "requires Vulkan device"]
    fn vulkan_garbage_collector_retains_requested_hash_buckets() -> Result<(), DriverError> {
        let device = Device::create(DeviceInfo::default())?;
        let mut collector = GarbageCollector::new(HashPool::with_capacity(&device, 4));
        let retained_info = BufferInfo::device_mem(64, vk::BufferUsageFlags::TRANSFER_SRC);
        let removed_info = BufferInfo::device_mem(64, vk::BufferUsageFlags::STORAGE_BUFFER);

        drop(collector.resource(retained_info)?);
        drop(collector.resource(removed_info)?);
        collector.collect_resources();
        assert_eq!(collector.buffer_cache.len(), 2);

        drop(collector.resource(retained_info)?);
        collector.collect_resources();
        assert_eq!(collector.buffer_cache.len(), 1);
        assert!(collector.buffer_cache.contains_key(&retained_info));

        collector.collect_resources();
        assert!(collector.buffer_cache.is_empty());

        Ok(())
    }
}
