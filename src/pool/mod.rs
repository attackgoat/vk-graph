//! Resource leasing and pooling types.
//!
//! _vk-graph_ provides caching for acceleration structure, buffer and image resources which may be
//! leased from configurable pools using their corresponding information structure. Most programs
//! will do fine with a single [`FifoPool`](self::fifo::FifoPool).
//!
//! Leased resources may be bound directly to a render graph and used in the same manner as regular
//! resources. After rendering has finished, the leased resources will return to the pool for reuse.
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
//! # let device = Device::new(DeviceInfo::default())?;
//! let mut pool = LazyPool::new(&device);
//!
//! let info = ImageInfo::image_2d(8, 8, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::STORAGE);
//! let my_image = pool.lease(info)?;
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
//! # When Should You Use Resource Aliasing?
//!
//! Wrapping any pool using [`AliasPool::new`](self::alias::AliasPool::new) enables resource
//! aliasing, which prevents excess resources from being created even when different parts of your
//! code request new resources.
//!
//! **_NOTE:_** Render graph submission will automatically attempt to re-order submitted passes to
//! reduce contention between individual resources.
//!
//! **_NOTE:_** In cases where multiple aliased resources using identical request information are
//! used in the same render graph pass you must ensure the resources are aliased from different
//! pools. There is currently no tagging or filter which would prevent "ping-pong" rendering of such
//! resources from being the same actual resources; this causes Vulkan validation warnings when
//! reading from and writing to the same images, or whatever your operations may be.
//!
//! ### Pros:
//!
//! * Fewer resources are created overall
//! * Wrapped pools behave like and retain all functionality of unwrapped pools
//! * Easy to experiment with and benchmark in your existing code
//!
//! ### Cons:
//!
//! * Non-zero cost: Atomic load and compatibility check per active alias
//! * May cause GPU stalling if there is not enough work being submitted
//! * Aliased resources are typed `Arc<Lease<T>>` and are not guaranteed to be mutable or unique

pub mod alias;
pub mod fifo;
pub mod hash;
pub mod lazy;

use {
    crate::driver::{
        CommandBuffer, DriverError,
        accel_struct::{
            AccelerationStructure, AccelerationStructureInfo, AccelerationStructureInfoBuilder,
        },
        buffer::{Buffer, BufferInfo, BufferInfoBuilder},
        image::{Image, ImageInfo, ImageInfoBuilder},
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

#[cfg(feature = "parking_lot")]
use parking_lot::Mutex;

#[cfg(not(feature = "parking_lot"))]
use std::sync::Mutex;

type Cache<T> = Arc<Mutex<Vec<T>>>;
type CacheRef<T> = Weak<Mutex<Vec<T>>>;

fn lease_command_buffer(cache: &mut Vec<CommandBuffer>) -> Option<CommandBuffer> {
    for idx in 0..cache.len() {
        if unsafe {
            let cmd_buf = cache.get_unchecked(idx);

            // Don't lease this command buffer if it is unsignalled; we'll create a new one
            // and wait for this, and those behind it, to signal.
            cmd_buf
                .device
                .get_fence_status(cmd_buf.fence)
                .unwrap_or_default()
        } {
            return Some(cache.swap_remove(idx));
        }
    }

    None
}

/// Holds a leased resource and implements `Drop` in order to return the resource.
///
/// This simple wrapper type implements only the `AsRef`, `AsMut`, `Deref` and `DerefMut` traits
/// and provides no other functionality. A freshly leased resource is guaranteed to have no other
/// owners and may be mutably accessed.
#[derive(Debug)]
pub struct Lease<T> {
    cache_ref: CacheRef<T>,
    item: ManuallyDrop<T>,
}

impl<T> Lease<T> {
    #[inline(always)]
    fn new(cache_ref: CacheRef<T>, item: T) -> Self {
        Self {
            cache_ref,
            item: ManuallyDrop::new(item),
        }
    }
}

impl<T> AsRef<T> for Lease<T> {
    fn as_ref(&self) -> &T {
        &self.item
    }
}

impl<T> AsMut<T> for Lease<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.item
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
        // into the pool.
        if let Some(cache) = self.cache_ref.upgrade() {
            #[cfg_attr(not(feature = "parking_lot"), allow(unused_mut))]
            let mut cache = cache.lock();

            #[cfg(not(feature = "parking_lot"))]
            let mut cache = cache.unwrap();

            if cache.len() == cache.capacity() {
                cache.pop();
            }

            cache.push(unsafe { ManuallyDrop::take(&mut self.item) });
        } else {
            unsafe {
                ManuallyDrop::drop(&mut self.item);
            }
        }
    }
}

/// Allows leasing of resources using driver information structures.
pub trait Pool<I, T> {
    /// Lease a resource.
    fn lease(&mut self, info: I) -> Result<Lease<T>, DriverError>;
}

// Enable leasing items using their info builder type for convenience
macro_rules! lease_builder {
    ($info:ident => $item:ident) => {
        paste::paste! {
            impl<T> Pool<[<$info Builder>], $item> for T where T: Pool<$info, $item> {
                fn lease(&mut self, builder: [<$info Builder>]) -> Result<Lease<$item>, DriverError> {
                    let info = builder.build();

                    self.lease(info)
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
#[derive(Builder, Clone, Copy, Debug)]
#[builder(
    build_fn(private, name = "fallible_build", error = "PoolInfoBuilderError"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
#[non_exhaustive]
pub struct PoolInfo {
    /// The maximum size of a single bucket of acceleration structure resource instances. The
    /// default value is [`PoolInfo::DEFAULT_RESOURCE_CAPACITY`].
    ///
    /// # Note
    ///
    /// Individual [`Pool`] implementations store varying numbers of buckets. Read the documentation
    /// of each implementation to understand how this affects total number of stored acceleration
    /// structure instances.
    #[builder(default = "PoolInfo::DEFAULT_RESOURCE_CAPACITY", setter(strip_option))]
    pub accel_struct_capacity: usize,

    /// The maximum size of a single bucket of buffer resource instances. The default value is
    /// [`PoolInfo::DEFAULT_RESOURCE_CAPACITY`].
    ///
    /// # Note
    ///
    /// Individual [`Pool`] implementations store varying numbers of buckets. Read the documentation
    /// of each implementation to understand how this affects total number of stored buffer
    /// instances.
    #[builder(default = "PoolInfo::DEFAULT_RESOURCE_CAPACITY", setter(strip_option))]
    pub buffer_capacity: usize,

    /// The maximum size of a single bucket of image resource instances. The default value is
    /// [`PoolInfo::DEFAULT_RESOURCE_CAPACITY`].
    ///
    /// # Note
    ///
    /// Individual [`Pool`] implementations store varying numbers of buckets. Read the documentation
    /// of each implementation to understand how this affects total number of stored image
    /// instances.
    #[builder(default = "PoolInfo::DEFAULT_RESOURCE_CAPACITY", setter(strip_option))]
    pub image_capacity: usize,
}

impl PoolInfo {
    /// The maximum size of a single bucket of resource instances.
    pub const DEFAULT_RESOURCE_CAPACITY: usize = 16;

    /// Constructs a new `PoolInfo` with the given acceleration structure, buffer and image resource
    /// capacity for any single bucket.
    pub const fn with_capacity(resource_capacity: usize) -> Self {
        Self {
            accel_struct_capacity: resource_capacity,
            buffer_capacity: resource_capacity,
            image_capacity: resource_capacity,
        }
    }

    fn default_cache<T>() -> Cache<T> {
        Cache::new(Mutex::new(Vec::with_capacity(
            Self::DEFAULT_RESOURCE_CAPACITY,
        )))
    }

    fn explicit_cache<T>(capacity: usize) -> Cache<T> {
        Cache::new(Mutex::new(Vec::with_capacity(capacity)))
    }
}

impl Default for PoolInfo {
    fn default() -> Self {
        PoolInfoBuilder::default().into()
    }
}

impl From<PoolInfoBuilder> for PoolInfo {
    fn from(info: PoolInfoBuilder) -> Self {
        info.build()
    }
}

impl From<usize> for PoolInfo {
    fn from(value: usize) -> Self {
        Self {
            accel_struct_capacity: value,
            buffer_capacity: value,
            image_capacity: value,
        }
    }
}

// HACK: https://github.com/colin-kiegel/rust-derive-builder/issues/56
impl PoolInfoBuilder {
    /// Builds a new `PoolInfo`.
    pub fn build(self) -> PoolInfo {
        self.fallible_build()
            .expect("All required fields set at initialization")
    }
}

#[derive(Debug)]
struct PoolInfoBuilderError;

impl From<UninitializedFieldError> for PoolInfoBuilderError {
    fn from(_: UninitializedFieldError) -> Self {
        Self
    }
}
