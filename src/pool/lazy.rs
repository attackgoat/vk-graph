//! Pool which requests by looking for compatible information before creating new resources.

use {
    super::{
        BufferHostMappingCompatibility, Cache, Lease, Pool, PoolConfig, compatible_buffer_info,
        garbage_collector::{CollectResources, ResourceRequests},
        with_cache,
    },
    crate::driver::{
        DriverError,
        accel_struct::{AccelerationStructure, AccelerationStructureInfo},
        buffer::{Buffer, BufferInfo},
        cmd_buf::{CommandBuffer, CommandBufferInfo},
        descriptor_set::{DescriptorPool, DescriptorPoolInfo},
        device::Device,
        image::{Image, ImageInfo, SampleCount},
        render_pass::{RenderPass, RenderPassInfo},
    },
    ash::vk,
    log::debug,
    std::{collections::HashMap, sync::Arc},
};

type BufferKey = (bool, vk::DeviceSize, vk::SharingMode);

fn buffer_key(info: &BufferInfo) -> BufferKey {
    (
        info.host_readable | info.host_writable,
        info.alignment,
        info.sharing_mode,
    )
}

fn compatible_accel_struct_info(
    item_info: &AccelerationStructureInfo,
    requested_info: &AccelerationStructureInfo,
) -> bool {
    item_info.acceleration_structure_type == requested_info.acceleration_structure_type
        && item_info.size >= requested_info.size
}

fn compatible_lazy_buffer_info(item_info: &BufferInfo, requested_info: &BufferInfo) -> bool {
    buffer_key(item_info) == buffer_key(requested_info)
        && compatible_buffer_info(
            item_info,
            requested_info,
            BufferHostMappingCompatibility::Superset,
        )
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct ImageKey {
    array_layer_count: u32,
    depth: u32,
    fmt: vk::Format,
    height: u32,
    mip_level_count: u32,
    sample_count: SampleCount,
    sharing_mode: vk::SharingMode,
    tiling: vk::ImageTiling,
    image_type: vk::ImageType,
    width: u32,
}

impl From<ImageInfo> for ImageKey {
    fn from(info: ImageInfo) -> Self {
        Self {
            array_layer_count: info.array_layer_count,
            depth: info.depth,
            fmt: info.format,
            height: info.height,
            mip_level_count: info.mip_level_count,
            sample_count: info.sample_count,
            sharing_mode: info.sharing_mode,
            tiling: info.tiling,
            image_type: info.image_type,
            width: info.width,
        }
    }
}

fn compatible_lazy_image_info(item_info: &ImageInfo, requested_info: &ImageInfo) -> bool {
    ImageKey::from(*item_info) == ImageKey::from(*requested_info)
        && item_info.flags.contains(requested_info.flags)
        && item_info.usage.contains(requested_info.usage)
}

/// A balanced resource allocator.
///
/// The information for each resource request is compared against the stored resources for
/// compatibility. If no acceptable resources are stored for the information provided a new resource
/// is created and returned.
///
/// # Details
///
/// * Acceleration structures may be larger than requested
/// * Buffers may be larger than requested or have additional usage flags
/// * Images may have additional usage flags
///
/// # Bucket Strategy
///
/// The information for each resource request is the key for a `HashMap` of buckets. If no bucket
/// exists with compatible information a new bucket is created.
///
/// In practice this means that for a [`PoolConfig::image_capacity`] of `4`, requests for a
/// 1024x1024 image with certain attributes will store a maximum of `4` such images. Requests for
/// any image having a different size or incompatible attributes will store an additional maximum of
/// `4` images.
///
/// # Memory Management
///
/// If requests for varying resources are common [`LazyPool::clear_images_by_info`] and other
/// memory management functions are necessary in order to avoid using all available device memory.
#[derive(Debug)]
#[read_only::cast]
pub struct LazyPool {
    accel_struct_cache: HashMap<vk::AccelerationStructureTypeKHR, Cache<AccelerationStructure>>,
    buffer_cache: HashMap<BufferKey, Cache<Buffer>>,
    command_buffer_cache: HashMap<u32, Cache<CommandBuffer>>,
    descriptor_pool_cache: Cache<DescriptorPool>,

    /// The device which owns this pool.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub device: Device,

    image_cache: HashMap<ImageKey, Cache<Image>>,

    /// Information used to create this pool.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub info: PoolConfig,

    render_pass_cache: HashMap<RenderPassInfo, Cache<RenderPass>>,
}

impl LazyPool {
    /// Constructs a new `LazyPool`.
    pub fn new(device: &Device) -> Self {
        Self::with_capacity(device, PoolConfig::default())
    }

    /// Constructs a new `LazyPool` with the given capacity information.
    pub fn with_capacity(device: &Device, info: impl Into<PoolConfig>) -> Self {
        let info: PoolConfig = info.into();
        let device = device.clone();

        Self {
            accel_struct_cache: Default::default(),
            buffer_cache: Default::default(),
            command_buffer_cache: Default::default(),
            descriptor_pool_cache: PoolConfig::default_cache(),
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

    /// Clears the pool of acceleration structure resources.
    pub fn clear_accel_structs(&mut self) {
        self.accel_struct_cache.clear();
    }

    /// Clears the pool of all acceleration structure resources matching the given type.
    pub fn clear_accel_structs_by_type(
        &mut self,
        accel_struct_ty: vk::AccelerationStructureTypeKHR,
    ) {
        self.accel_struct_cache.remove(&accel_struct_ty);
    }

    /// Clears the pool of buffer resources.
    pub fn clear_buffers(&mut self) {
        self.buffer_cache.clear();
    }

    /// Clears the pool of image resources.
    pub fn clear_images(&mut self) {
        self.image_cache.clear();
    }

    /// Clears the pool of image resources matching the given information.
    pub fn clear_images_by_info(&mut self, info: impl Into<ImageInfo>) {
        self.image_cache.remove(&info.into().into());
    }

    /// Retains only the acceleration structure resources specified by the predicate.
    ///
    /// In other words, remove all resources for which `f(vk::AccelerationStructureTypeKHR)` returns
    /// `false`.
    ///
    /// The elements are visited in unsorted (and unspecified) order.
    ///
    /// # Performance
    ///
    /// Provides the same performance guarantees as
    /// [`HashMap::retain`](HashMap::retain).
    pub fn retain_accel_structs<F>(&mut self, mut f: F)
    where
        F: FnMut(vk::AccelerationStructureTypeKHR) -> bool,
    {
        self.accel_struct_cache
            .retain(|&accel_struct_ty, _| f(accel_struct_ty))
    }
}

impl CollectResources for LazyPool {
    fn collect_resources(&mut self, requests: &ResourceRequests) {
        self.accel_struct_cache.retain(|accel_struct_ty, cache| {
            let retain_bucket = requests
                .accel_structs
                .iter()
                .any(|info| info.acceleration_structure_type == *accel_struct_ty);

            if retain_bucket {
                with_cache(cache, |cache| {
                    cache.retain(|item| {
                        requests
                            .accel_structs
                            .iter()
                            .any(|info| compatible_accel_struct_info(&item.info, info))
                    });
                });
            }

            retain_bucket
        });

        self.buffer_cache.retain(|key, cache| {
            let retain_bucket = requests.buffers.iter().any(|info| buffer_key(info) == *key);

            if retain_bucket {
                with_cache(cache, |cache| {
                    cache.retain(|item| {
                        requests
                            .buffers
                            .iter()
                            .any(|info| compatible_lazy_buffer_info(&item.info, info))
                    });
                });
            }

            retain_bucket
        });

        self.image_cache.retain(|key, cache| {
            let retain_bucket = requests
                .images
                .iter()
                .any(|info| ImageKey::from(*info) == *key);

            if retain_bucket {
                with_cache(cache, |cache| {
                    cache.retain(|item| {
                        requests
                            .images
                            .iter()
                            .any(|info| compatible_lazy_image_info(&item.info, info))
                    });
                });
            }

            retain_bucket
        });
    }
}

impl Pool<AccelerationStructureInfo, AccelerationStructure> for LazyPool {
    #[profiling::function]
    fn resource(
        &mut self,
        info: AccelerationStructureInfo,
    ) -> Result<Lease<AccelerationStructure>, DriverError> {
        let cache = self
            .accel_struct_cache
            .entry(info.acceleration_structure_type)
            .or_insert_with(|| PoolConfig::explicit_cache(self.info.accel_struct_capacity));
        let cache_ref = Arc::downgrade(cache);

        {
            profiling::scope!("check cache");

            #[cfg_attr(not(feature = "parking_lot"), allow(unused_mut))]
            let mut cache = cache.lock();

            #[cfg(not(feature = "parking_lot"))]
            let mut cache = cache.expect("poisoned cache lock");

            // Look for a compatible acceleration structure (big enough)
            for idx in 0..cache.len() {
                let item = unsafe { cache.get_unchecked(idx) };
                if compatible_accel_struct_info(&item.info, &info) {
                    let item = cache.swap_remove(idx);

                    return Ok(Lease::new(cache_ref, item));
                }
            }
        }

        debug!("Creating new {}", stringify!(AccelerationStructure));

        let item = AccelerationStructure::create(&self.device, info)?;

        Ok(Lease::new(cache_ref, item))
    }
}

impl Pool<BufferInfo, Buffer> for LazyPool {
    #[profiling::function]
    fn resource(&mut self, info: BufferInfo) -> Result<Lease<Buffer>, DriverError> {
        let cache = self
            .buffer_cache
            .entry(buffer_key(&info))
            .or_insert_with(|| PoolConfig::explicit_cache(self.info.buffer_capacity));
        let cache_ref = Arc::downgrade(cache);

        {
            profiling::scope!("check cache");

            #[cfg_attr(not(feature = "parking_lot"), allow(unused_mut))]
            let mut cache = cache.lock();

            #[cfg(not(feature = "parking_lot"))]
            let mut cache = cache.expect("poisoned cache lock");

            // Look for a compatible buffer (big enough and superset of usage flags)
            for idx in 0..cache.len() {
                let item = unsafe { cache.get_unchecked(idx) };
                if compatible_lazy_buffer_info(&item.info, &info) {
                    let item = cache.swap_remove(idx);

                    return Ok(Lease::new(cache_ref, item));
                }
            }
        }

        debug!("Creating new {}", stringify!(Buffer));

        let item = Buffer::create(&self.device, info)?;

        Ok(Lease::new(cache_ref, item))
    }
}

impl Pool<CommandBufferInfo, CommandBuffer> for LazyPool {
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

impl Pool<DescriptorPoolInfo, DescriptorPool> for LazyPool {
    #[profiling::function]
    fn resource(&mut self, info: DescriptorPoolInfo) -> Result<Lease<DescriptorPool>, DriverError> {
        let cache_ref = Arc::downgrade(&self.descriptor_pool_cache);

        {
            profiling::scope!("check cache");

            #[cfg_attr(not(feature = "parking_lot"), allow(unused_mut))]
            let mut cache = self.descriptor_pool_cache.lock();

            #[cfg(not(feature = "parking_lot"))]
            let mut cache = cache.expect("poisoned cache lock");

            // Look for a compatible descriptor pool (has enough sets and descriptors)
            for idx in 0..cache.len() {
                let item = unsafe { cache.get_unchecked(idx) };
                if item.info.max_sets >= info.max_sets
                    && item.info.acceleration_structure_count >= info.acceleration_structure_count
                    && item.info.combined_image_sampler_count >= info.combined_image_sampler_count
                    && item.info.input_attachment_count >= info.input_attachment_count
                    && item.info.sampled_image_count >= info.sampled_image_count
                    && item.info.sampler_count >= info.sampled_image_count
                    && item.info.storage_buffer_count >= info.storage_buffer_count
                    && item.info.storage_buffer_dynamic_count >= info.storage_buffer_dynamic_count
                    && item.info.storage_image_count >= info.storage_image_count
                    && item.info.storage_texel_buffer_count >= info.storage_texel_buffer_count
                    && item.info.uniform_buffer_count >= info.uniform_buffer_count
                    && item.info.uniform_buffer_dynamic_count >= info.uniform_buffer_dynamic_count
                    && item.info.uniform_texel_buffer_count >= info.uniform_texel_buffer_count
                {
                    let item = cache.swap_remove(idx);

                    return Ok(Lease::new(cache_ref, item));
                }
            }
        }

        debug!("Creating new {}", stringify!(DescriptorPool));

        let item = DescriptorPool::create(&self.device, info)?;

        Ok(Lease::new(cache_ref, item))
    }
}

impl Pool<ImageInfo, Image> for LazyPool {
    #[profiling::function]
    fn resource(&mut self, info: ImageInfo) -> Result<Lease<Image>, DriverError> {
        let cache = self
            .image_cache
            .entry(info.into())
            .or_insert_with(|| PoolConfig::explicit_cache(self.info.image_capacity));
        let cache_ref = Arc::downgrade(cache);

        {
            profiling::scope!("check cache");

            #[cfg_attr(not(feature = "parking_lot"), allow(unused_mut))]
            let mut cache = cache.lock();

            #[cfg(not(feature = "parking_lot"))]
            let mut cache = cache.expect("poisoned cache lock");

            // Look for a compatible image (superset of creation flags and usage flags)
            for idx in 0..cache.len() {
                let item = unsafe { cache.get_unchecked(idx) };
                if compatible_lazy_image_info(&item.info, &info) {
                    let item = cache.swap_remove(idx);

                    return Ok(Lease::new(cache_ref, item));
                }
            }
        }

        debug!("Creating new {}", stringify!(Image));

        let item = Image::create(&self.device, info)?;

        Ok(Lease::new(cache_ref, item))
    }
}

impl Pool<RenderPassInfo, RenderPass> for LazyPool {
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

#[cfg(test)]
mod test {
    use {
        super::*,
        crate::{
            driver::device::{Device, DeviceInfo},
            pool::garbage_collector::GarbageCollector,
        },
    };

    #[test]
    #[ignore = "requires Vulkan device"]
    fn vulkan_garbage_collector_retains_supported_lazy_resources() -> Result<(), DriverError> {
        let device = Device::create(DeviceInfo::default())?;
        let mut collector = GarbageCollector::new(LazyPool::with_capacity(&device, 4));
        let retained_info = BufferInfo::device_mem(64, vk::BufferUsageFlags::TRANSFER_SRC);
        let removed_info = BufferInfo::device_mem(64, vk::BufferUsageFlags::STORAGE_BUFFER);
        let key = buffer_key(&retained_info);

        drop(collector.resource(retained_info)?);
        drop(collector.resource(removed_info)?);
        collector.collect_resources();
        assert_eq!(
            with_cache(&collector.buffer_cache[&key], |cache| cache.len()),
            2
        );

        drop(collector.resource(retained_info)?);
        collector.collect_resources();
        assert_eq!(
            with_cache(&collector.buffer_cache[&key], |cache| cache.len()),
            1
        );

        collector.collect_resources();
        assert!(collector.buffer_cache.is_empty());

        Ok(())
    }
}
