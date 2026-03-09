//! Pool wrapper which enables memory-efficient resource aliasing.

use {
    super::{Lease, Pool},
    crate::driver::{
        DriverError,
        accel_struct::{
            AccelerationStructure, AccelerationStructureInfo, AccelerationStructureInfoBuilder,
        },
        buffer::{Buffer, BufferInfo, BufferInfoBuilder},
        image::{Image, ImageInfo, ImageInfoBuilder},
    },
    log::debug,
    std::{
        ops::{Deref, DerefMut},
        sync::{Arc, Weak},
    },
};

#[deprecated = "use AliasWrapper type"]
#[doc(hidden)]
pub type AliasPool<T> = self::AliasWrapper<T>;

/// Allows aliasing of resources using driver information structures.
///
/// Aliasing is just like leasing except you don't get an instance and somebody else might have
/// already recorded commands against the resource which have not yet completed.
pub trait Alias<I, T> {
    #[deprecated = "use alias_resource function"]
    #[doc(hidden)]
    fn alias(&mut self, info: I) -> Result<Arc<Lease<T>>, DriverError> {
        self.alias_resource(info)
    }

    /// Returns an alias of a resource.
    fn alias_resource(&mut self, info: I) -> Result<Arc<Lease<T>>, DriverError>;
}

// Enable aliasing items using their info builder type for convenience
macro_rules! alias_builder {
    ($info:ident => $item:ident) => {
        paste::paste! {
            impl<T> Alias<[<$info Builder>], $item> for T where T: Alias<$info, $item> {
                fn alias_resource(&mut self, builder: [<$info Builder>]) -> Result<Arc<Lease<$item>>, DriverError> {
                    let info = builder.build();

                    self.alias_resource(info)
                }
            }
        }
    };
}

alias_builder!(AccelerationStructureInfo => AccelerationStructure);
alias_builder!(BufferInfo => Buffer);
alias_builder!(ImageInfo => Image);

/// A memory-efficient resource wrapper for any [`Pool`] type.
///
/// The information for each alias request is compared against the actively aliased resources for
/// compatibility. If no acceptable resources are aliased for the information provided a new
/// resource is leased and returned.
///
/// All regular leasing and other functionality of the wrapped pool is available through `Deref` and
/// `DerefMut`.
///
/// **_NOTE:_** You must call `alias_resource(..)` to use resource aliasing as regular
/// `lease_resource(..)` calls will not inspect or return aliased resources.
///
/// # Details
///
/// * Acceleration structures may be larger than requested
/// * Buffers may be larger than requested or have additional usage flags
/// * Images may have additional usage flags
///
/// # Examples
///
/// See [`aliasing.rs`](https://github.com/attackgoat/vk-graph/blob/master/examples/aliasing.rs)
pub struct AliasWrapper<T> {
    accel_structs: Vec<(
        AccelerationStructureInfo,
        Weak<Lease<AccelerationStructure>>,
    )>,
    buffers: Vec<(BufferInfo, Weak<Lease<Buffer>>)>,
    images: Vec<(ImageInfo, Weak<Lease<Image>>)>,
    pool: T,
}

impl<T> AliasWrapper<T> {
    /// Creates a new aliasable wrapper over the given pool.
    pub fn new(pool: T) -> Self {
        Self {
            accel_structs: Default::default(),
            buffers: Default::default(),
            images: Default::default(),
            pool,
        }
    }
}

// Enable aliasing items using their info builder type for convenience
macro_rules! lease_pass_through {
    ($info:ident => $item:ident) => {
        paste::paste! {
            impl<T> Pool<$info, $item> for AliasWrapper<T> where T: Pool<$info, $item> {
                fn lease_resource(&mut self, info: $info) -> Result<Lease<$item>, DriverError> {
                    self.pool.lease_resource(info)
                }
            }
        }
    };
}

lease_pass_through!(AccelerationStructureInfo => AccelerationStructure);
lease_pass_through!(BufferInfo => Buffer);
lease_pass_through!(ImageInfo => Image);

impl<T> Alias<AccelerationStructureInfo, AccelerationStructure> for AliasWrapper<T>
where
    T: Pool<AccelerationStructureInfo, AccelerationStructure>,
{
    fn alias_resource(
        &mut self,
        info: AccelerationStructureInfo,
    ) -> Result<Arc<Lease<AccelerationStructure>>, DriverError> {
        self.accel_structs
            .retain(|(_, item)| item.strong_count() > 0);

        {
            profiling::scope!("check aliases");

            for (item_info, item) in &self.accel_structs {
                if item_info.ty == info.ty && item_info.size >= info.size {
                    if let Some(item) = item.upgrade() {
                        return Ok(item);
                    } else {
                        break;
                    }
                }
            }
        }

        debug!("Leasing new {}", stringify!(AccelerationStructure));

        let item = Arc::new(self.pool.lease_resource(info)?);
        self.accel_structs.push((info, Arc::downgrade(&item)));

        Ok(item)
    }
}

impl<T> Alias<BufferInfo, Buffer> for AliasWrapper<T>
where
    T: Pool<BufferInfo, Buffer>,
{
    fn alias_resource(&mut self, info: BufferInfo) -> Result<Arc<Lease<Buffer>>, DriverError> {
        self.buffers.retain(|(_, item)| item.strong_count() > 0);

        {
            profiling::scope!("check aliases");

            for (item_info, item) in &self.buffers {
                if (item_info.dedicated & info.dedicated) == info.dedicated
                    && item_info.host_read == info.host_read
                    && item_info.host_write == info.host_write
                    && item_info.alignment >= info.alignment
                    && item_info.size >= info.size
                    && item_info.usage.contains(info.usage)
                {
                    if let Some(item) = item.upgrade() {
                        return Ok(item);
                    } else {
                        break;
                    }
                }
            }
        }

        debug!("Leasing new {}", stringify!(Buffer));

        let item = Arc::new(self.pool.lease_resource(info)?);
        self.buffers.push((info, Arc::downgrade(&item)));

        Ok(item)
    }
}

impl<T> Alias<ImageInfo, Image> for AliasWrapper<T>
where
    T: Pool<ImageInfo, Image>,
{
    fn alias_resource(&mut self, info: ImageInfo) -> Result<Arc<Lease<Image>>, DriverError> {
        self.images.retain(|(_, item)| item.strong_count() > 0);

        {
            profiling::scope!("check aliases");

            for (item_info, item) in &self.images {
                if item_info.array_layer_count == info.array_layer_count
                    && item_info.dedicated == info.dedicated
                    && item_info.depth == info.depth
                    && item_info.fmt == info.fmt
                    && item_info.height == info.height
                    && item_info.mip_level_count == info.mip_level_count
                    && item_info.sample_count == info.sample_count
                    && item_info.tiling == info.tiling
                    && item_info.ty == info.ty
                    && item_info.width == info.width
                    && item_info.flags.contains(info.flags)
                    && item_info.usage.contains(info.usage)
                {
                    if let Some(item) = item.upgrade() {
                        return Ok(item);
                    } else {
                        break;
                    }
                }
            }
        }

        debug!("Leasing new {}", stringify!(Image));

        let item = Arc::new(self.pool.lease_resource(info)?);
        self.images.push((info, Arc::downgrade(&item)));

        Ok(item)
    }
}

impl<T> Deref for AliasWrapper<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.pool
    }
}

impl<T> DerefMut for AliasWrapper<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.pool
    }
}
