//! Resource pooling, requesting, and caching types.
//!
//! Resource pools provide caching for buffer, image, and acceleration structure resources. Pooled
//! resources may be requested from a pool using their corresponding information structure.
//!
//! Leased resources may be bound directly to a [`Graph`](crate::Graph) and used in the same manner
//! as regular resources. After execution has completed pooled resources are automatically returned
//! to their pool for reuse.
//!
//! # Buckets
//!
//! The provided [`Pool`] implementations store resources in buckets, with each implementation
//! offering a different strategy which balances performance (_more buckets_) with memory efficiency
//! (_fewer buckets_).
//!
//! _vk-graph_'s pools can be grouped into two major categories:
//!
//! * Single-bucket: [`FifoPool`](self::fifo::FifoPool)
//! * Multi-bucket: [`LazyPool`](self::lazy::LazyPool), [`HashPool`](self::hash::HashPool)
//!
//! # Examples
//!
//! Leasing an image:
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use ash::vk;
//! # use vk_graph::driver::DriverError;
//! # use vk_graph::driver::device::{Device, DeviceInfo};
//! # use vk_graph::driver::image::{ImageInfo};
//! # use vk_graph::pool::{Pool};
//! # use vk_graph::pool::lazy::{LazyPool};
//! # fn main() -> Result<(), DriverError> {
//! # let device = Device::create(DeviceInfo::default())?;
//! let mut pool = LazyPool::new(&device);
//!
//! let info = ImageInfo::image_2d(8, 8, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::STORAGE);
//! let my_image = pool.resource(info)?;
//!
//! assert!(my_image.info.usage.contains(vk::ImageUsageFlags::STORAGE));
//! # Ok(()) }
//! ```
//!
//! # When Should You Use Which Pool?
//!
//! These are fairly high-level break-downs of when each pool should be considered. You may need
//! to investigate each type of pool individually to provide the absolute best fit for your purpose.
//!
//! ### Use a [`FifoPool`](self::fifo::FifoPool) when:
//! * Low memory usage is most important
//! * Automatic bucket management is desired
//!
//! ### Use a [`LazyPool`](self::lazy::LazyPool) when:
//! * Resources have different attributes each frame
//!
//! ### Use a [`HashPool`](self::hash::HashPool) when:
//! * High performance is most important
//! * Resources have consistent attributes each frame
//!
//! # When Should You Use Resource Caching?
//!
//! Wrapping any pool using [`cache::Cache::new`] enables resource caching, which prevents excess
//! resources from being created even when different parts of your code request compatible
//! resources.
//!
//! **_NOTE:_** Graph submission will automatically attempt to re-order submitted commands to
//! reduce contention between individual resources.
//!
//! **_NOTE:_** In cases where multiple cached resources using identical request information are
//! used in the same graph command, ensure they come from different cache tags or different pool
//! wrappers. Otherwise, two requests may resolve to the same underlying resource and trigger
//! Vulkan validation warnings when reading from and writing to the same images.
//!
//! ### Pros:
//!
//! * Fewer resources are created overall
//! * Wrapped pools behave like and retain all functionality of unwrapped pools
//! * Easy to experiment with and benchmark in your existing code
//!
//! ### Cons:
//!
//! * Non-zero cost: atomic load and compatibility check per active cached resource
//! * May cause GPU stalling if there is not enough work being submitted
//! * Cached resources are typed `Arc<Lease<T>>` and are not guaranteed to be mutable or unique

pub mod cache;
pub mod fifo;
pub mod hash;
pub mod lazy;

use {
    crate::driver::{
        DriverError,
        accel_struct::{
            AccelerationStructure, AccelerationStructureInfo, AccelerationStructureInfoBuilder,
        },
        buffer::{Buffer, BufferInfo, BufferInfoBuilder},
        descriptor_set::{DescriptorPool, DescriptorPoolInfo},
        image::{Image, ImageInfo, ImageInfoBuilder},
        render_pass::{RenderPass, RenderPassInfo},
    },
    derive_builder::{Builder, UninitializedFieldError},
    std::{
        fmt::Debug,
        mem::ManuallyDrop,
        ops::{Deref, DerefMut},
        sync::{Arc, Weak},
        thread::panicking,
    },
};

#[derive(Clone, Copy)]
enum BufferHostMappingCompatibility {
    Exact,
    Superset,
}

fn compatible_buffer_info(
    item_info: &BufferInfo,
    requested_info: &BufferInfo,
    host_mapping: BufferHostMappingCompatibility,
) -> bool {
    (item_info.alloc_dedicated & requested_info.alloc_dedicated) == requested_info.alloc_dedicated
        && compatible_buffer_host_mapping(item_info, requested_info, host_mapping)
        && item_info.alignment >= requested_info.alignment
        && item_info.sharing_mode == requested_info.sharing_mode
        && item_info.size >= requested_info.size
        && item_info.usage.contains(requested_info.usage)
}

fn compatible_buffer_host_mapping(
    item_info: &BufferInfo,
    requested_info: &BufferInfo,
    compatibility: BufferHostMappingCompatibility,
) -> bool {
    match compatibility {
        BufferHostMappingCompatibility::Exact => {
            item_info.host_readable == requested_info.host_readable
                && item_info.host_writable == requested_info.host_writable
        }
        BufferHostMappingCompatibility::Superset => {
            (item_info.host_readable & requested_info.host_readable) == requested_info.host_readable
                && (item_info.host_writable & requested_info.host_writable)
                    == requested_info.host_writable
        }
    }
}

fn compatible_image_info(item_info: &ImageInfo, requested_info: &ImageInfo) -> bool {
    item_info.array_layer_count == requested_info.array_layer_count
        && item_info.alloc_dedicated == requested_info.alloc_dedicated
        && item_info.depth == requested_info.depth
        && item_info.format == requested_info.format
        && item_info.height == requested_info.height
        && item_info.host_readable == requested_info.host_readable
        && item_info.host_writable == requested_info.host_writable
        && item_info.mip_level_count == requested_info.mip_level_count
        && item_info.sample_count == requested_info.sample_count
        && item_info.sharing_mode == requested_info.sharing_mode
        && item_info.tiling == requested_info.tiling
        && item_info.ty == requested_info.ty
        && item_info.width == requested_info.width
        && item_info.flags.contains(requested_info.flags)
        && item_info.usage.contains(requested_info.usage)
}

#[cfg(feature = "parking_lot")]
use parking_lot::Mutex;

#[cfg(not(feature = "parking_lot"))]
use std::sync::Mutex;

type Cache<T> = Arc<Mutex<Vec<T>>>;
type CacheRef<T> = Weak<Mutex<Vec<T>>>;

fn with_cache<T, R>(cache: &Cache<T>, f: impl FnOnce(&mut Vec<T>) -> R) -> R {
    let cache = cache.lock();

    #[cfg(not(feature = "parking_lot"))]
    let cache = cache.expect("poisoned cache lock");

    let mut cache = cache;

    f(&mut cache)
}

/// Holds a pooled resource and implements `Drop` in order to return the resource.
///
/// This simple wrapper type implements only the `AsRef`, `AsMut`, `Deref` and `DerefMut` traits
/// and provides no other functionality. A freshly obtained resource is guaranteed to have no other
/// owners and may be mutably accessed.
#[derive(Debug)]
pub struct Lease<T> {
    cache_ref: CacheRef<T>,
    item: ManuallyDrop<T>,
}

/*
The following debug_name functions take a self of Lease<T> and return Self.
This allows pooled resources to have the same `.debug_name("bugs")` chaining.
*/

impl Lease<AccelerationStructure> {
    /// Sets the debugging name assigned to this acceleration structure.
    pub fn with_debug_name(self, name: impl AsRef<str>) -> Self {
        self.set_debug_name(name);

        self
    }
}

impl Lease<Buffer> {
    /// Sets the debugging name assigned to this buffer.
    pub fn with_debug_name(self, name: impl AsRef<str>) -> Self {
        self.set_debug_name(name);

        self
    }
}

impl Lease<Image> {
    /// Sets the debugging name assigned to this image.
    pub fn with_debug_name(self, name: impl AsRef<str>) -> Self {
        self.set_debug_name(name);

        self
    }
}

impl<T> Lease<T> {
    fn new(cache_ref: CacheRef<T>, item: T) -> Self {
        Self {
            cache_ref,
            item: ManuallyDrop::new(item),
        }
    }
}

impl<T> AsRef<T> for Lease<T> {
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T> Deref for Lease<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.item
    }
}

impl<T> DerefMut for Lease<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.item
    }
}

impl<T> Drop for Lease<T> {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        // If the pool cache has been dropped we must manually drop the item, otherwise it goes back
        // into the pool
        if let Some(cache) = self.cache_ref.upgrade() {
            with_cache(&cache, |cache| {
                if cache.len() >= cache.capacity() {
                    cache.pop();
                }

                cache.push(unsafe { ManuallyDrop::take(&mut self.item) });
            });
        } else {
            unsafe {
                ManuallyDrop::drop(&mut self.item);
            }
        }
    }
}

/// Allows requesting resources using driver information structures.
pub trait Pool<I, T> {
    /// Request a resource.
    fn resource(&mut self, info: I) -> Result<Lease<T>, DriverError>;
}

/// Pool capability required by graph submission scheduling.
///
/// This sealed trait is implemented by the built-in pools. It covers internal descriptor-pool and
/// render-pass leases without exposing their cache-key types in public API bounds.
#[allow(private_bounds)]
pub trait SubmissionPool: submission_pool_private::SubmissionPoolSealed {}

impl<T> SubmissionPool for T where T: submission_pool_private::SubmissionPoolSealed {}

pub(crate) mod submission_pool_private {
    use super::*;

    pub(crate) trait SubmissionPoolSealed {
        fn descriptor_pool(
            &mut self,
            info: DescriptorPoolInfo,
        ) -> Result<Lease<DescriptorPool>, DriverError>;

        fn render_pass(&mut self, info: RenderPassInfo) -> Result<Lease<RenderPass>, DriverError>;
    }

    impl<T> SubmissionPoolSealed for T
    where
        T: Pool<DescriptorPoolInfo, DescriptorPool> + Pool<RenderPassInfo, RenderPass>,
    {
        fn descriptor_pool(
            &mut self,
            info: DescriptorPoolInfo,
        ) -> Result<Lease<DescriptorPool>, DriverError> {
            self.resource(info)
        }

        fn render_pass(&mut self, info: RenderPassInfo) -> Result<Lease<RenderPass>, DriverError> {
            self.resource(info)
        }
    }
}

// Enable requesting items using their info builder type for convenience
macro_rules! lease_builder {
    ($info:ident => $item:ident) => {
        paste::paste! {
            impl<T> Pool<[<$info Builder>], $item> for T where T: Pool<$info, $item> {
                fn resource(
                    &mut self,
                    builder: [<$info Builder>],
                ) -> Result<Lease<$item>, DriverError> {
                    let info = builder.build();

                    self.resource(info)
                }
            }
        }
    };
}

lease_builder!(AccelerationStructureInfo => AccelerationStructure);
lease_builder!(BufferInfo => Buffer);
lease_builder!(ImageInfo => Image);

/// Information used to create a [`FifoPool`](self::fifo::FifoPool),
/// [`HashPool`](self::hash::HashPool) or [`LazyPool`](self::lazy::LazyPool) instance.
#[derive(Builder, Clone, Copy, Debug, Eq, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build", error = "UninitializedFieldError"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct PoolConfig {
    /// The maximum size of a single bucket of acceleration structure resource instances. The
    /// default value is [`PoolConfig::DEFAULT_RESOURCE_CAPACITY`].
    ///
    /// # Note
    ///
    /// Individual [`Pool`] implementations store varying numbers of buckets. Read the
    /// documentation of each implementation to understand how this affects total number of
    /// stored acceleration structure instances.
    #[builder(
        default = "PoolConfig::DEFAULT_RESOURCE_CAPACITY",
        setter(strip_option)
    )]
    pub accel_struct_capacity: usize,

    /// The maximum size of a single bucket of buffer resource instances. The default value is
    /// [`PoolConfig::DEFAULT_RESOURCE_CAPACITY`].
    ///
    /// # Note
    ///
    /// Individual [`Pool`] implementations store varying numbers of buckets. Read the
    /// documentation of each implementation to understand how this affects total number of
    /// stored buffer instances.
    #[builder(
        default = "PoolConfig::DEFAULT_RESOURCE_CAPACITY",
        setter(strip_option)
    )]
    pub buffer_capacity: usize,

    /// The maximum size of a single bucket of image resource instances. The default value is
    /// [`PoolConfig::DEFAULT_RESOURCE_CAPACITY`].
    ///
    /// # Note
    ///
    /// Individual [`Pool`] implementations store varying numbers of buckets. Read the
    /// documentation of each implementation to understand how this affects total number of
    /// stored image instances.
    #[builder(
        default = "PoolConfig::DEFAULT_RESOURCE_CAPACITY",
        setter(strip_option)
    )]
    pub image_capacity: usize,
}

impl PoolConfig {
    /// The maximum size of a single bucket of resource instances.
    pub const DEFAULT_RESOURCE_CAPACITY: usize = 16;

    /// Creates a default `PoolConfigBuilder`.
    pub fn builder() -> PoolConfigBuilder {
        Default::default()
    }

    fn default_cache<T>() -> Cache<T> {
        Cache::new(Mutex::new(Vec::with_capacity(
            Self::DEFAULT_RESOURCE_CAPACITY,
        )))
    }

    fn explicit_cache<T>(capacity: usize) -> Cache<T> {
        Cache::new(Mutex::new(Vec::with_capacity(capacity)))
    }

    /// Converts a `PoolConfig` into a `PoolConfigBuilder`.
    pub fn into_builder(self) -> PoolConfigBuilder {
        PoolConfigBuilder {
            accel_struct_capacity: Some(self.accel_struct_capacity),
            buffer_capacity: Some(self.buffer_capacity),
            image_capacity: Some(self.image_capacity),
        }
    }

    /// Constructs a new `PoolConfig` with the given acceleration structure, buffer and image
    /// resource capacity for any single bucket.
    pub const fn with_capacity(resource_capacity: usize) -> Self {
        Self {
            accel_struct_capacity: resource_capacity,
            buffer_capacity: resource_capacity,
            image_capacity: resource_capacity,
        }
    }
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfigBuilder::default().into()
    }
}

impl From<PoolConfigBuilder> for PoolConfig {
    fn from(info: PoolConfigBuilder) -> Self {
        info.build()
    }
}

impl From<usize> for PoolConfig {
    fn from(value: usize) -> Self {
        Self {
            accel_struct_capacity: value,
            buffer_capacity: value,
            image_capacity: value,
        }
    }
}

// HACK: https://github.com/colin-kiegel/rust-derive-builder/issues/56
impl PoolConfigBuilder {
    /// Builds a new `PoolConfig`.
    pub fn build(self) -> PoolConfig {
        self.fallible_build().expect("invalid pool config")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::driver::ash::vk;

    type Info = PoolConfig;
    type Builder = PoolConfigBuilder;

    #[test]
    pub fn pool_info() {
        let info = Info::default();
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn pool_info_builder() {
        let info = Info {
            accel_struct_capacity: 1,
            buffer_capacity: 2,
            image_capacity: 3,
        };
        let builder = Builder::default()
            .accel_struct_capacity(1)
            .buffer_capacity(2)
            .image_capacity(3)
            .build();

        assert_eq!(info, builder);
    }

    #[test]
    fn buffer_info_compatibility_rejects_different_sharing_mode() {
        let exclusive = BufferInfo::device_mem(64, vk::BufferUsageFlags::STORAGE_BUFFER);
        let concurrent = BufferInfo {
            sharing_mode: vk::SharingMode::CONCURRENT,
            ..exclusive
        };

        assert!(!compatible_buffer_info(
            &exclusive,
            &concurrent,
            BufferHostMappingCompatibility::Exact,
        ));
        assert!(!compatible_buffer_info(
            &exclusive,
            &concurrent,
            BufferHostMappingCompatibility::Superset,
        ));
    }

    #[test]
    fn image_info_compatibility_rejects_different_sharing_mode() {
        let exclusive = ImageInfo::image_2d(
            16,
            16,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
        );
        let concurrent = ImageInfo {
            sharing_mode: vk::SharingMode::CONCURRENT,
            ..exclusive
        };

        assert!(!compatible_image_info(&exclusive, &concurrent));
    }
}
