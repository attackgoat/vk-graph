//! Pool which requests by looking for compatibile information before creating new resources.

use {
    super::{Cache, Lease, Pool, PoolConfig, lease_command_buffer},
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
    ty: vk::ImageType,
    width: u32,
}

impl From<ImageInfo> for ImageKey {
    fn from(info: ImageInfo) -> Self {
        Self {
            array_layer_count: info.array_layer_count,
            depth: info.depth,
            fmt: info.fmt,
            height: info.height,
            mip_level_count: info.mip_level_count,
            sample_count: info.sample_count,
            sharing_mode: info.sharing_mode,
            tiling: info.tiling,
            ty: info.ty,
            width: info.width,
        }
    }
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
/// If requests for varying resources is common [`LazyPool::clear_images_by_info`] and other memory
/// management functions are nessecery in order to avoid using all available device memory.
#[derive(Debug)]
#[read_only::cast]
pub struct LazyPool {
    accel_struct_cache: HashMap<vk::AccelerationStructureTypeKHR, Cache<AccelerationStructure>>,
    buffer_cache: HashMap<(bool, vk::DeviceSize), Cache<Buffer>>,
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
    pub fn clear_accel_structs_by_ty(&mut self, ty: vk::AccelerationStructureTypeKHR) {
        self.accel_struct_cache.remove(&ty);
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
        self.accel_struct_cache.retain(|&ty, _| f(ty))
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
            .entry(info.ty)
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
                if item.info.size >= info.size {
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
            .entry((info.host_read | info.host_write, info.alignment))
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
                if (item.info.dedicated & info.dedicated) == info.dedicated
                    && (item.info.host_read & info.host_read) == info.host_read
                    && (item.info.host_write & info.host_write) == info.host_write
                    && item.info.alignment >= info.alignment
                    && item.info.size >= info.size
                    && item.info.usage.contains(info.usage)
                {
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

            lease_command_buffer(&mut cache)
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
                if item.info.flags.contains(info.flags) && item.info.usage.contains(info.usage) {
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
