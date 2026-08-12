//! Pool wrapper which removes cached resources that no longer support observed requests.

use {
    super::{Lease, Pool},
    crate::driver::{
        DriverError,
        accel_struct::{AccelerationStructure, AccelerationStructureInfo},
        buffer::{Buffer, BufferInfo},
        cmd_buf::{CommandBuffer, CommandBufferInfo},
        descriptor_set::{DescriptorPool, DescriptorPoolInfo},
        image::{Image, ImageInfo},
        render_pass::{RenderPass, RenderPassInfo},
    },
    std::{
        collections::HashSet,
        ops::{Deref, DerefMut},
    },
};

#[derive(Default)]
pub(super) struct ResourceRequests {
    pub(super) accel_structs: HashSet<AccelerationStructureInfo>,
    pub(super) buffers: HashSet<BufferInfo>,
    pub(super) images: HashSet<ImageInfo>,
}

impl ResourceRequests {
    fn clear(&mut self) {
        self.accel_structs.clear();
        self.buffers.clear();
        self.images.clear();
    }
}

pub(super) trait CollectResources {
    fn collect_resources(&mut self, requests: &ResourceRequests);
}

/// A request-aware garbage collector for built-in [`Pool`] types.
///
/// Successful acceleration-structure, buffer, and image requests are recorded until
/// [`GarbageCollector::collect_resources`] is called. Collection retains only cached resources that
/// the wrapped pool could use to satisfy those requests, then begins a new observation interval.
/// Calling `collect_resources` without making any requests clears all managed resources.
///
/// Checked-out resources remain valid during collection. Resources returning to a retained bucket
/// are evaluated by the next collection, while resources returning to a removed bucket are dropped.
/// Command buffers, descriptor pools, and render passes are forwarded without being collected.
///
/// # Examples
///
/// ```no_run
/// # use ash::vk;
/// # use vk_graph::driver::DriverError;
/// # use vk_graph::driver::buffer::BufferInfo;
/// # use vk_graph::driver::device::{Device, DeviceInfo};
/// # use vk_graph::pool::Pool;
/// # use vk_graph::pool::garbage_collector::GarbageCollector;
/// # use vk_graph::pool::lazy::LazyPool;
/// # fn main() -> Result<(), DriverError> {
/// # let device = Device::create(DeviceInfo::default())?;
/// let mut pool = GarbageCollector::new(LazyPool::new(&device));
///
/// let buffer = pool.resource(BufferInfo::device_mem(
///     1024,
///     vk::BufferUsageFlags::STORAGE_BUFFER,
/// ))?;
/// drop(buffer);
///
/// // Retain cached resources supporting requests made since the previous collection.
/// pool.collect_resources();
/// # Ok(()) }
/// ```
pub struct GarbageCollector<T> {
    pool: T,
    requests: ResourceRequests,
}

impl<T> GarbageCollector<T> {
    /// Creates a garbage collector wrapper over the given pool.
    pub fn new(pool: T) -> Self {
        Self {
            pool,
            requests: Default::default(),
        }
    }
}

#[allow(private_bounds)]
impl<T> GarbageCollector<T>
where
    T: CollectResources,
{
    /// Collects cached resources and begins a new request observation interval.
    ///
    /// Only acceleration structures, buffers, and images supporting successful requests made since
    /// the previous call are retained. If there were no such requests, all managed resources are
    /// removed.
    pub fn collect_resources(&mut self) {
        self.pool.collect_resources(&self.requests);
        self.requests.clear();
    }
}

macro_rules! tracked_pool {
    ($info:ty => $item:ty, $requests:ident) => {
        impl<T> Pool<$info, $item> for GarbageCollector<T>
        where
            T: Pool<$info, $item>,
        {
            fn resource(&mut self, info: $info) -> Result<Lease<$item>, DriverError> {
                let item = self.pool.resource(info)?;
                self.requests.$requests.insert(info);

                Ok(item)
            }
        }
    };
}

tracked_pool!(AccelerationStructureInfo => AccelerationStructure, accel_structs);
tracked_pool!(BufferInfo => Buffer, buffers);
tracked_pool!(ImageInfo => Image, images);

macro_rules! forwarded_pool {
    ($info:ty => $item:ty) => {
        impl<T> Pool<$info, $item> for GarbageCollector<T>
        where
            T: Pool<$info, $item>,
        {
            fn resource(&mut self, info: $info) -> Result<Lease<$item>, DriverError> {
                self.pool.resource(info)
            }
        }
    };
}

forwarded_pool!(CommandBufferInfo => CommandBuffer);
forwarded_pool!(DescriptorPoolInfo => DescriptorPool);
forwarded_pool!(RenderPassInfo => RenderPass);

impl<T> Deref for GarbageCollector<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.pool
    }
}

impl<T> DerefMut for GarbageCollector<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.pool
    }
}

#[cfg(test)]
mod test {
    use {
        super::*,
        crate::{
            driver::{
                accel_struct::AccelerationStructureInfoBuilder, buffer::BufferInfoBuilder,
                image::ImageInfoBuilder,
            },
            pool::{SubmissionPool, fifo::FifoPool, hash::HashPool, lazy::LazyPool},
        },
        ash::vk,
    };

    #[derive(Default)]
    struct CollectSpy {
        calls: Vec<(usize, usize, usize)>,
    }

    impl CollectResources for CollectSpy {
        fn collect_resources(&mut self, requests: &ResourceRequests) {
            self.calls.push((
                requests.accel_structs.len(),
                requests.buffers.len(),
                requests.images.len(),
            ));
        }
    }

    struct FailingPool;

    impl Pool<BufferInfo, Buffer> for FailingPool {
        fn resource(&mut self, _: BufferInfo) -> Result<Lease<Buffer>, DriverError> {
            Err(DriverError::Unsupported)
        }
    }

    fn assert_pool_capabilities<T>()
    where
        T: Pool<AccelerationStructureInfo, AccelerationStructure>
            + Pool<AccelerationStructureInfoBuilder, AccelerationStructure>
            + Pool<BufferInfo, Buffer>
            + Pool<BufferInfoBuilder, Buffer>
            + Pool<ImageInfo, Image>
            + Pool<ImageInfoBuilder, Image>
            + Pool<CommandBufferInfo, CommandBuffer>
            + SubmissionPool,
    {
    }

    fn assert_collect_resources<T: CollectResources>() {}

    #[test]
    fn collect_resources_sweeps_and_resets_observed_requests() {
        let mut collector = GarbageCollector::new(CollectSpy::default());
        let info = BufferInfo::device_mem(64, vk::BufferUsageFlags::STORAGE_BUFFER);

        collector.requests.buffers.insert(info);
        collector.requests.buffers.insert(info);
        collector.collect_resources();
        collector.collect_resources();

        assert_eq!(collector.pool.calls, [(0, 1, 0), (0, 0, 0)]);
    }

    #[test]
    fn failed_requests_are_not_observed() {
        let mut collector = GarbageCollector::new(FailingPool);
        let info = BufferInfo::device_mem(64, vk::BufferUsageFlags::STORAGE_BUFFER);

        assert!(matches!(
            Pool::<BufferInfo, Buffer>::resource(&mut collector, info),
            Err(DriverError::Unsupported)
        ));
        assert!(collector.requests.buffers.is_empty());
    }

    #[test]
    fn built_in_pools_support_collection_and_pool_capabilities() {
        assert_collect_resources::<FifoPool>();
        assert_collect_resources::<HashPool>();
        assert_collect_resources::<LazyPool>();
        assert_pool_capabilities::<GarbageCollector<FifoPool>>();
        assert_pool_capabilities::<GarbageCollector<HashPool>>();
        assert_pool_capabilities::<GarbageCollector<LazyPool>>();
    }
}
