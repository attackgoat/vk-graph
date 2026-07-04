//! Query pool types.

use {
    super::{DriverError, cmd_buf::CommandBuffer, device::Device},
    ash::vk,
    log::warn,
    std::{fmt::Debug, thread::panicking},
};

/// Represents a Vulkan query pool.
///
/// See [`VkQueryPool`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryPool.html).
#[derive(Debug)]
#[read_only::cast]
pub struct QueryPool {
    /// The device which owns this query pool.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub device: Device,

    /// The native Vulkan query pool handle.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub handle: vk::QueryPool,

    /// Information used to create this query pool.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub info: QueryPoolInfo,
}

impl QueryPool {
    /// Creates a Vulkan query pool owned by `device`.
    ///
    /// See [`vkCreateQueryPool`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateQueryPool.html).
    pub fn create(device: &Device, info: impl Into<QueryPoolInfo>) -> Result<Self, DriverError> {
        let info = info.into();
        let handle = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(info.query_type)
                    .query_count(info.query_count)
                    .pipeline_statistics(info.pipeline_statistics),
                None,
            )
        }
        .map_err(|err| {
            warn!("unable to create query pool: {err}");

            match err {
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                    DriverError::OutOfMemory
                }
                _ => DriverError::Unsupported,
            }
        })?;

        Ok(Self {
            device: device.clone(),
            handle,
            info,
        })
    }

    /// Resets query results in this pool using a command buffer.
    ///
    /// See [`vkCmdResetQueryPool`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdResetQueryPool.html).
    pub fn reset(&self, cmd_buf: &CommandBuffer, first_query: u32, query_count: u32) {
        unsafe {
            cmd_buf.device.cmd_reset_query_pool(
                cmd_buf.handle,
                self.handle,
                first_query,
                query_count,
            );
        }
    }

    /// Reads 64-bit query pool results.
    ///
    /// `TYPE_64` is always included in the flags passed to Vulkan.
    ///
    /// See [`vkGetQueryPoolResults`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetQueryPoolResults.html).
    pub fn results_u64(
        &self,
        first_query: u32,
        query_count: u32,
        flags: vk::QueryResultFlags,
    ) -> Result<Vec<u64>, DriverError> {
        let mut results = vec![0_u64; query_count as usize];

        unsafe {
            self.device
                .get_query_pool_results(
                    self.handle,
                    first_query,
                    results.as_mut_slice(),
                    flags | vk::QueryResultFlags::TYPE_64,
                )
                .map_err(|err| {
                    warn!("unable to get query pool results: {err}");

                    match err {
                        vk::Result::ERROR_DEVICE_LOST => DriverError::InvalidData,
                        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                        | vk::Result::ERROR_OUT_OF_HOST_MEMORY => DriverError::OutOfMemory,
                        _ => DriverError::Unsupported,
                    }
                })?;
        }

        Ok(results)
    }
}

impl Drop for QueryPool {
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        unsafe {
            self.device.destroy_query_pool(self.handle, None);
        }
    }
}

/// Information used to create a [`QueryPool`].
#[derive(Clone, Copy, Debug)]
pub struct QueryPoolInfo {
    /// Type of queries managed by the pool.
    pub query_type: vk::QueryType,

    /// Number of queries managed by the pool.
    pub query_count: u32,

    /// Pipeline statistics to query when `query_type` is [`vk::QueryType::PIPELINE_STATISTICS`].
    pub pipeline_statistics: vk::QueryPipelineStatisticFlags,
}

impl QueryPoolInfo {
    /// Creates timestamp query pool information.
    pub fn timestamp(query_count: u32) -> Self {
        Self {
            query_type: vk::QueryType::TIMESTAMP,
            query_count,
            pipeline_statistics: vk::QueryPipelineStatisticFlags::empty(),
        }
    }
}

impl Default for QueryPoolInfo {
    fn default() -> Self {
        Self::timestamp(1)
    }
}
