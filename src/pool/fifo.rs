//! Pool which leases from a single bucket per resource type.

use {
    super::{Cache, Lease, Pool, PoolInfo, lease_command_buffer, with_cache},
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
    std::{collections::HashMap, sync::Arc},
};

/// A memory-efficient resource allocator.
///
/// The information for each lease request is compared against the stored resources for
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
/// All resources are stored in a single bucket per resource type, regardless of their individual
/// attributes.
///
/// In practice this means that for a [`PoolInfo::image_capacity`] of `4`, a maximum of `4` images
/// will be stored. Requests to lease an image or other resource will first look for a compatible
/// resource in the bucket and create a new resource as needed.
///
/// # Memory Management
///
/// The single-bucket strategy means that there will always be a reasonable and predictable number
/// of stored resources, however you may call [`FifoPool::clear`] or the other memory management
/// functions at any time to discard stored resources.
#[derive(Debug)]
#[read_only::cast]
pub struct FifoPool {
    accel_struct_cache: Cache<AccelerationStructure>,
    buffer_cache: Cache<Buffer>,
    command_buffer_cache: HashMap<u32, Cache<CommandBuffer>>,
    descriptor_pool_cache: Cache<DescriptorPool>,

    /// The device which owns this pool.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub device: Device,

    image_cache: Cache<Image>,

    /// Information used to create this pool.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub info: PoolInfo,

    render_pass_cache: HashMap<RenderPassInfo, Cache<RenderPass>>,
}

impl FifoPool {
    /// Constructs a new `FifoPool`.
    pub fn new(device: &Device) -> Self {
        Self::with_capacity(device, PoolInfo::default())
    }

    /// Constructs a new `FifoPool` with the given capacity information.
    pub fn with_capacity(device: &Device, info: impl Into<PoolInfo>) -> Self {
        let info: PoolInfo = info.into();
        let device = device.clone();

        Self {
            accel_struct_cache: PoolInfo::explicit_cache(info.accel_struct_capacity),
            buffer_cache: PoolInfo::explicit_cache(info.buffer_capacity),
            command_buffer_cache: Default::default(),
            descriptor_pool_cache: PoolInfo::default_cache(),
            device,
            image_cache: PoolInfo::explicit_cache(info.image_capacity),
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
        self.accel_struct_cache = PoolInfo::explicit_cache(self.info.accel_struct_capacity);
    }

    /// Clears the pool of buffer resources.
    pub fn clear_buffers(&mut self) {
        self.buffer_cache = PoolInfo::explicit_cache(self.info.buffer_capacity);
    }

    /// Clears the pool of image resources.
    pub fn clear_images(&mut self) {
        self.image_cache = PoolInfo::explicit_cache(self.info.image_capacity);
    }
}

impl Pool<AccelerationStructureInfo, AccelerationStructure> for FifoPool {
    #[profiling::function]
    fn resource(
        &mut self,
        info: AccelerationStructureInfo,
    ) -> Result<Lease<AccelerationStructure>, DriverError> {
        let cache_ref = Arc::downgrade(&self.accel_struct_cache);

        {
            profiling::scope!("check cache");

            if let Some(item) = with_cache(&self.accel_struct_cache, |cache| {
                // Look for a compatible acceleration structure (big enough and same type)
                for idx in 0..cache.len() {
                    let item = unsafe { cache.get_unchecked(idx) };
                    if item.info.size >= info.size && item.info.ty == info.ty {
                        let item = cache.swap_remove(idx);

                        return Some(Lease::new(cache_ref.clone(), item));
                    }
                }

                None
            }) {
                return Ok(item);
            }
        }

        debug!("Creating new {}", stringify!(AccelerationStructure));

        let item = AccelerationStructure::create(&self.device, info)?;

        Ok(Lease::new(cache_ref, item))
    }
}

impl Pool<BufferInfo, Buffer> for FifoPool {
    #[profiling::function]
    fn resource(&mut self, info: BufferInfo) -> Result<Lease<Buffer>, DriverError> {
        let cache_ref = Arc::downgrade(&self.buffer_cache);

        {
            profiling::scope!("check cache");

            if let Some(item) = with_cache(&self.buffer_cache, |cache| {
                // Look for a compatible buffer (compatible alignment, same mapping mode, big enough and
                // superset of usage flags)
                for idx in 0..cache.len() {
                    let item = unsafe { cache.get_unchecked(idx) };
                    if (item.info.dedicated & info.dedicated) == info.dedicated
                        && item.info.host_read == info.host_read
                        && item.info.host_write == info.host_write
                        && item.info.alignment >= info.alignment
                        && item.info.size >= info.size
                        && item.info.usage.contains(info.usage)
                    {
                        let item = cache.swap_remove(idx);

                        return Some(Lease::new(cache_ref.clone(), item));
                    }
                }

                None
            }) {
                return Ok(item);
            }
        }

        debug!("Creating new {}", stringify!(Buffer));

        let item = Buffer::create(&self.device, info)?;

        Ok(Lease::new(cache_ref, item))
    }
}

impl Pool<CommandBufferInfo, CommandBuffer> for FifoPool {
    #[profiling::function]
    fn resource(&mut self, info: CommandBufferInfo) -> Result<Lease<CommandBuffer>, DriverError> {
        let cache_ref = self
            .command_buffer_cache
            .entry(info.queue_family_index)
            .or_insert_with(PoolInfo::default_cache);

        let item = with_cache(cache_ref, lease_command_buffer)
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

impl Pool<DescriptorPoolInfo, DescriptorPool> for FifoPool {
    #[profiling::function]
    fn resource(&mut self, info: DescriptorPoolInfo) -> Result<Lease<DescriptorPool>, DriverError> {
        let cache_ref = Arc::downgrade(&self.descriptor_pool_cache);

        {
            profiling::scope!("check cache");

            if let Some(item) = with_cache(&self.descriptor_pool_cache, |cache| {
                // Look for a compatible descriptor pool (has enough sets and descriptors)
                for idx in 0..cache.len() {
                    let item = unsafe { cache.get_unchecked(idx) };
                    if item.info.max_sets >= info.max_sets
                        && item.info.acceleration_structure_count
                            >= info.acceleration_structure_count
                        && item.info.combined_image_sampler_count
                            >= info.combined_image_sampler_count
                        && item.info.input_attachment_count >= info.input_attachment_count
                        && item.info.sampled_image_count >= info.sampled_image_count
                        && item.info.sampler_count >= info.sampler_count
                        && item.info.storage_buffer_count >= info.storage_buffer_count
                        && item.info.storage_buffer_dynamic_count
                            >= info.storage_buffer_dynamic_count
                        && item.info.storage_image_count >= info.storage_image_count
                        && item.info.storage_texel_buffer_count >= info.storage_texel_buffer_count
                        && item.info.uniform_buffer_count >= info.uniform_buffer_count
                        && item.info.uniform_buffer_dynamic_count
                            >= info.uniform_buffer_dynamic_count
                        && item.info.uniform_texel_buffer_count >= info.uniform_texel_buffer_count
                    {
                        let item = cache.swap_remove(idx);

                        return Some(Lease::new(cache_ref.clone(), item));
                    }
                }

                None
            }) {
                return Ok(item);
            }
        }

        debug!("Creating new {}", stringify!(DescriptorPool));

        let item = DescriptorPool::create(&self.device, info)?;

        Ok(Lease::new(cache_ref, item))
    }
}

impl Pool<ImageInfo, Image> for FifoPool {
    #[profiling::function]
    fn resource(&mut self, info: ImageInfo) -> Result<Lease<Image>, DriverError> {
        let cache_ref = Arc::downgrade(&self.image_cache);

        {
            profiling::scope!("check cache");

            if let Some(item) = with_cache(&self.image_cache, |cache| {
                // Look for a compatible image (same properties, superset of creation flags and usage
                // flags)
                for idx in 0..cache.len() {
                    let item = unsafe { cache.get_unchecked(idx) };
                    if item.info.array_layer_count == info.array_layer_count
                        && item.info.dedicated == info.dedicated
                        && item.info.depth == info.depth
                        && item.info.fmt == info.fmt
                        && item.info.height == info.height
                        && item.info.mip_level_count == info.mip_level_count
                        && item.info.sample_count == info.sample_count
                        && item.info.tiling == info.tiling
                        && item.info.ty == info.ty
                        && item.info.width == info.width
                        && item.info.flags.contains(info.flags)
                        && item.info.usage.contains(info.usage)
                    {
                        let item = cache.swap_remove(idx);

                        return Some(Lease::new(cache_ref.clone(), item));
                    }
                }

                None
            }) {
                return Ok(item);
            }
        }

        debug!("Creating new {}", stringify!(Image));

        let item = Image::create(&self.device, info)?;

        Ok(Lease::new(cache_ref, item))
    }
}

impl Pool<RenderPassInfo, RenderPass> for FifoPool {
    #[profiling::function]
    fn resource(&mut self, info: RenderPassInfo) -> Result<Lease<RenderPass>, DriverError> {
        let cache_ref = if let Some(cache) = self.render_pass_cache.get(&info) {
            cache
        } else {
            // We tried to get the cache first in order to avoid this clone
            self.render_pass_cache
                .entry(info.clone())
                .or_insert_with(PoolInfo::default_cache)
        };
        let item = with_cache(cache_ref, |cache| cache.pop())
            .map(Ok)
            .unwrap_or_else(|| {
                debug!("Creating new {}", stringify!(RenderPass));

                RenderPass::create(&self.device, info)
            })?;

        Ok(Lease::new(Arc::downgrade(cache_ref), item))
    }
}
