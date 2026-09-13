//! Buffer resource types

use {
    super::{
        DriverError, SharingMode, device::Device, is_write_access, pipeline_stage_access_flags,
    },
    ash::vk,
    derive_builder::Builder,
    gpu_allocator::{
        MemoryLocation,
        vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
    },
    log::trace,
    log::warn,
    smallvec::{SmallVec, smallvec},
    std::{
        fmt::{Debug, Formatter},
        iter::once,
        mem::{ManuallyDrop, take},
        ops::{DerefMut, Range},
        sync::atomic::{AtomicU64, Ordering},
        thread::panicking,
    },
    vk_sync::AccessType,
};

#[cfg(feature = "parking_lot")]
use parking_lot::{Mutex, MutexGuard};

#[cfg(not(feature = "parking_lot"))]
use std::sync::{Mutex, MutexGuard};

type AccessRuns = RunMap<AccessType>;

const fn tracked_access_after(previous: AccessType, next: AccessType) -> AccessType {
    if is_write_access(previous) && !is_write_access(next) {
        AccessType::General
    } else {
        next
    }
}

/// Smart pointer handle to a [buffer] object.
///
/// Also contains information about the object.
///
/// ```no_run
/// # use ash::vk;
/// # use vk_graph::driver::DriverError;
/// # use vk_graph::driver::device::{Device, DeviceInfo};
/// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
/// # fn main() -> Result<(), DriverError> {
/// # let device = Device::create(DeviceInfo::default())?;
/// let info = BufferInfo::device_mem(1_024, vk::BufferUsageFlags::STORAGE_BUFFER);
/// let my_buf = Buffer::create(&device, info)?;
///
/// assert_eq!(my_buf.info, info);
/// assert_ne!(my_buf.handle, vk::Buffer::null());
/// # Ok(()) }
/// ```
///
/// [buffer]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkBuffer.html
#[read_only::cast]
pub struct Buffer {
    access_runs: Mutex<AccessRuns>,
    allocation: ManuallyDrop<Allocation>,

    /// The device which owns this buffer resource.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub device: Device,

    /// The native Vulkan resource handle of this buffer.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub handle: vk::Buffer,

    /// Information used to create this resource.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub info: BufferInfo,

    sharing: Sharing,
}

impl Buffer {
    /// Creates a new buffer on the given device.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// const SIZE: vk::DeviceSize = 1024;
    /// let info = BufferInfo::host_mem(SIZE, vk::BufferUsageFlags::UNIFORM_BUFFER);
    /// let buf = Buffer::create(&device, info)?;
    ///
    /// assert_ne!(buf.handle, vk::Buffer::null());
    /// assert_eq!(buf.info.size, SIZE);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn create(device: &Device, info: impl Into<BufferInfo>) -> Result<Self, DriverError> {
        let info = info.into();

        trace!("create: {:?}", info);

        debug_assert_ne!(info.size, 0, "Size must be non-zero");

        let device = device.clone();
        let buffer_info = vk::BufferCreateInfo::default()
            .size(info.size)
            .usage(info.usage)
            .sharing_mode(info.sharing_mode);

        let buffer_info = if info.sharing_mode == vk::SharingMode::CONCURRENT {
            buffer_info.queue_family_indices(&device.physical.queue_family_indices)
        } else {
            buffer_info
        };
        let handle = unsafe {
            device.create_buffer(&buffer_info, None).map_err(|err| {
                warn!("unable to create buffer: {err}");

                DriverError::Unsupported
            })?
        };
        let mut requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        requirements.alignment = requirements.alignment.max(info.alignment);

        let allocation_scheme = if info.alloc_dedicated {
            AllocationScheme::DedicatedBuffer(handle)
        } else {
            AllocationScheme::GpuAllocatorManaged
        };

        // Read/write buffers need the cached host-memory preference too. CpuToGpu
        // prefers uncached device-local mappings, where CPU memcpy can be very slow.
        let location = if info.host_readable {
            MemoryLocation::GpuToCpu
        } else if info.host_writable {
            MemoryLocation::CpuToGpu
        } else {
            MemoryLocation::GpuOnly
        };
        let allocation = {
            profiling::scope!("allocate");

            Device::with_allocator(&device, |allocator| {
                allocator
                    .allocate(&AllocationCreateDesc {
                        name: "buffer",
                        requirements,
                        location,
                        linear: true, // Buffers are always linear
                        allocation_scheme,
                    })
                    .map_err(|err| {
                        warn!("unable to allocate buffer memory: {err}");

                        unsafe {
                            device.destroy_buffer(handle, None);
                        }

                        DriverError::from_alloc_err(err)
                    })
                    .and_then(|allocation| {
                        if let Err(err) = unsafe {
                            device.bind_buffer_memory(
                                handle,
                                allocation.memory(),
                                allocation.offset(),
                            )
                        } {
                            warn!("unable to bind buffer memory: {err}");

                            if let Err(err) = allocator.free(allocation) {
                                warn!("unable to free buffer allocation: {err}")
                            }

                            unsafe {
                                device.destroy_buffer(handle, None);
                            }

                            Err(DriverError::OutOfMemory)
                        } else {
                            Ok(allocation)
                        }
                    })
            })
        }?;

        debug_assert_ne!(handle, vk::Buffer::null());

        Ok(Self {
            access_runs: Mutex::new(AccessRuns::new(info.size, AccessType::Nothing)),
            allocation: ManuallyDrop::new(allocation),
            device,
            handle,
            info,
            sharing: Sharing::new(info.size, info.sharing_mode),
        })
    }

    /// Creates a new mappable buffer on the given device and fills it with the data in `slice`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// const DATA: [u8; 4] = [0xfe, 0xed, 0xbe, 0xef];
    /// let buf = Buffer::create_from_slice(&device, vk::BufferUsageFlags::UNIFORM_BUFFER, &DATA)?;
    ///
    /// assert_ne!(buf.handle, vk::Buffer::null());
    /// assert_eq!(buf.info.size, 4);
    /// assert_eq!(Buffer::mapped_slice(&buf), &DATA);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn create_from_slice(
        device: &Device,
        usage: vk::BufferUsageFlags,
        data: &[u8],
    ) -> Result<Self, DriverError> {
        let info = BufferInfo::host_mem(data.len() as _, usage);
        let mut buffer = Self::create(device, info)?;

        Self::copy_from_slice(&mut buffer, 0, data);

        Ok(buffer)
    }

    /// Updates a mappable buffer starting at `offset` with the data in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if the buffer was not created with host-writable memory.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// # let info = BufferInfo::host_mem(4, vk::BufferUsageFlags::empty());
    /// # let mut my_buf = Buffer::create(&device, info)?;
    /// const DATA: [u8; 4] = [0xde, 0xad, 0xc0, 0xde];
    /// Buffer::copy_from_slice(&mut my_buf, 0, &DATA);
    ///
    /// assert_eq!(Buffer::mapped_slice(&my_buf), &DATA);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn copy_from_slice(&mut self, offset: vk::DeviceSize, data: &[u8]) {
        profiling::scope!(
            "Mapped Upload",
            format!(
                "bytes={} buffer={} allocation={} flags={:?}",
                data.len(),
                self.info.size,
                self.allocation.size(),
                self.allocation.memory_properties()
            )
            .as_str()
        );
        let range = offset as _..offset as usize + data.len();
        let mapped_data = self.mapped_slice_mut();

        mapped_data[range].copy_from_slice(data);
    }

    /// Returns the device address of this object.
    ///
    /// # Panics
    ///
    /// Panics if the buffer was not created with the `SHADER_DEVICE_ADDRESS` usage flag.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// # let info = BufferInfo::host_mem(4, vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
    /// # let my_buf = Buffer::create(&device, info)?;
    /// let addr = my_buf.device_address();
    ///
    /// assert_ne!(addr, 0);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn device_address(&self) -> vk::DeviceAddress {
        #[cfg(feature = "checked")]
        assert!(
            self.info
                .usage
                .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        );

        unsafe {
            self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(self.handle),
            )
        }
    }

    fn lock_access_runs(&self) -> MutexGuard<'_, AccessRuns> {
        let access_runs = self.access_runs.lock();

        #[cfg(not(feature = "parking_lot"))]
        let access_runs = access_runs.expect("poisoned buffer access lock");

        access_runs
    }

    /// Returns a mapped slice.
    ///
    /// # Panics
    ///
    /// Panics if the buffer was not created with host-readable memory.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// # const DATA: [u8; 4] = [0; 4];
    /// # let my_buf = Buffer::create_from_slice(&device, vk::BufferUsageFlags::empty(), &DATA)?;
    /// // my_buf is mappable and filled with four zeroes
    /// let data = Buffer::mapped_slice(&my_buf);
    ///
    /// assert_eq!(data.len(), 4);
    /// assert_eq!(data[0], 0x00);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn mapped_slice(&self) -> &[u8] {
        #[cfg(feature = "checked")]
        assert!(
            self.info.host_readable,
            "Buffer is not readable - create using host_readable flag"
        );

        &self
            .allocation
            .mapped_slice()
            .expect("missing mapped buffer memory")[0..self.info.size as usize]
    }

    /// Returns a mapped mutable slice.
    ///
    /// # Panics
    ///
    /// Panics if the buffer was not created with host-writable memory.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use glam::Mat4;
    /// # use vk_graph::driver::DriverError;
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// # const DATA: [u8; 4] = [0; 4];
    /// # let mut my_buf = Buffer::create_from_slice(
    /// #     &device,
    /// #     vk::BufferUsageFlags::empty(),
    /// #     &DATA,
    /// # )?;
    /// let mut data = Buffer::mapped_slice_mut(&mut my_buf);
    /// data.copy_from_slice(&42f32.to_be_bytes());
    ///
    /// assert_eq!(data.len(), 4);
    /// assert_eq!(data[0], 0x42);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn mapped_slice_mut(&mut self) -> &mut [u8] {
        #[cfg(feature = "checked")]
        assert!(
            self.info.host_writable,
            "Buffer is not writable - create using host_writable flag"
        );

        &mut self
            .allocation
            .mapped_slice_mut()
            .expect("missing mapped buffer memory")[0..self.info.size as usize]
    }

    /// Sets the debugging name assigned to this buffer.
    pub fn set_debug_name(&self, name: impl AsRef<str>) {
        Device::try_set_debug_utils_object_name(&self.device, self.handle, &name);
        Device::try_set_private_data_object_name(
            &self.device,
            vk::ObjectType::BUFFER,
            self.handle,
            &name,
        );
    }

    pub(crate) fn set_sharing_ranges(
        &self,
        sharing: SharingMode,
        sharing_ranges: &[BufferSubresourceRange],
    ) {
        if sharing_ranges
            .iter()
            .all(|range| range.end != vk::WHOLE_SIZE)
        {
            self.sharing
                .set_ranges(self.info.size, sharing, sharing_ranges.iter().copied());

            return;
        }

        self.sharing.set_ranges(
            self.info.size,
            sharing,
            sharing_ranges
                .iter()
                .copied()
                .map(|range| range.resolve_whole(self.info.size)),
        );
    }

    /// Keeps track of some `next_access` which affects this object.
    ///
    /// Returns the previous access for which a pipeline barrier should be used to prevent data
    /// corruption.
    #[profiling::function]
    pub(crate) fn swap_access(
        &self,
        next_access: AccessType,
        access_range: impl Into<BufferSubresourceRange>,
    ) -> impl Iterator<Item = (AccessType, BufferSubresourceRange)> + '_ {
        let mut access_range: BufferSubresourceRange = access_range.into();

        if access_range.end == vk::WHOLE_SIZE {
            access_range.end = self.info.size;
        }

        RunMapIter::new(self.lock_access_runs(), next_access, access_range)
    }

    pub(crate) fn swap_accesses<'a, I>(
        &'a self,
        accesses: I,
    ) -> impl Iterator<Item = (AccessType, AccessType, BufferSubresourceRange)> + 'a
    where
        I: IntoIterator<Item = (AccessType, BufferSubresourceRange)>,
        I::IntoIter: 'a,
    {
        struct Iter<'a, I>
        where
            I: Iterator<Item = (AccessType, BufferSubresourceRange)>,
        {
            access_runs: MutexGuard<'a, AccessRuns>,
            accesses: I,
            current: Option<(AccessType, RunMapCursor)>,
            size: vk::DeviceSize,
        }

        impl<'a, I> Iter<'a, I>
        where
            I: Iterator<Item = (AccessType, BufferSubresourceRange)>,
        {
            fn new(
                access_runs: MutexGuard<'a, AccessRuns>,
                accesses: I,
                size: vk::DeviceSize,
            ) -> Self {
                Self {
                    access_runs,
                    accesses,
                    current: None,
                    size,
                }
            }
        }

        impl<I> Iterator for Iter<'_, I>
        where
            I: Iterator<Item = (AccessType, BufferSubresourceRange)>,
        {
            type Item = (AccessType, AccessType, BufferSubresourceRange);

            fn next(&mut self) -> Option<Self::Item> {
                loop {
                    if let Some((next_access, cursor)) = &mut self.current {
                        if let Some((prev_access, range)) = cursor
                            .next_with(&mut self.access_runs, |prev_access| {
                                tracked_access_after(prev_access, *next_access)
                            })
                        {
                            return Some((*next_access, prev_access, range));
                        }

                        self.current = None;
                    }

                    let (next_access, mut access_range) = self.accesses.next()?;
                    if access_range.end == vk::WHOLE_SIZE {
                        access_range.end = self.size;
                    }

                    self.current = Some((
                        next_access,
                        RunMapCursor::new(&self.access_runs, access_range),
                    ));
                }
            }
        }

        impl<I> Drop for Iter<'_, I>
        where
            I: Iterator<Item = (AccessType, BufferSubresourceRange)>,
        {
            fn drop(&mut self) {
                while self.next().is_some() {}
            }
        }

        let accesses = accesses.into_iter();
        let (min_accesses, _) = accesses.size_hint();
        let mut access_runs = self.lock_access_runs();
        access_runs.runs.reserve(min_accesses.saturating_mul(2));

        Iter::new(access_runs, accesses, self.info.size)
    }

    /// Returns compact synchronization information for the buffer's current access ranges.
    pub fn sync_info(&self) -> BufferSyncInfo {
        let ranges = self
            .sync_info_with_sharing()
            .map(|(range, sharing)| range.into_public(sharing))
            .collect();

        BufferSyncInfo { ranges }
    }

    pub(crate) fn sync_info_with_sharing(
        &self,
    ) -> impl Iterator<Item = (BufferSubresourceSyncInfo, SharingMode)> + '_ {
        self.sync_info_with_sharing_range(BufferSubresourceRange {
            start: 0,
            end: self.info.size,
        })
    }

    pub(crate) fn sync_info_with_sharing_range(
        &self,
        query_range: BufferSubresourceRange,
    ) -> impl Iterator<Item = (BufferSubresourceSyncInfo, SharingMode)> + '_ {
        struct SyncInfoIter<'a> {
            access_runs: MutexGuard<'a, AccessRuns>,
            access_run_idx: usize,
            query_range: BufferSubresourceRange,
            sharing_run: Option<(SharingMode, BufferSubresourceRange)>,
            sharing_runs: SharingRunIter<'a>,
        }

        impl Iterator for SyncInfoIter<'_> {
            type Item = (BufferSubresourceSyncInfo, SharingMode);

            fn next(&mut self) -> Option<Self::Item> {
                while self.access_run_idx < self.access_runs.runs.len() {
                    let (access, start) = self.access_runs.runs[self.access_run_idx];
                    let end = self
                        .access_runs
                        .runs
                        .get(self.access_run_idx + 1)
                        .map(|(_, next_start)| *next_start)
                        .unwrap_or(self.access_runs.size);
                    let access_range = BufferSubresourceRange { start, end };

                    let Some(access_range) = access_range.intersection(self.query_range) else {
                        if end <= self.query_range.start {
                            self.access_run_idx += 1;

                            continue;
                        }

                        return None;
                    };

                    let (sharing, sharing_run_range) = self.sharing_run?;

                    let Some(range) = access_range.intersection(sharing_run_range) else {
                        if sharing_run_range.end <= access_range.start {
                            self.sharing_run = self.sharing_runs.next();
                        } else {
                            self.access_run_idx += 1;
                        }

                        continue;
                    };

                    if sharing_run_range.end <= access_range.end {
                        self.sharing_run = self.sharing_runs.next();
                    }

                    if access_range.end <= sharing_run_range.end {
                        self.access_run_idx += 1;
                    }

                    return Some((
                        BufferSubresourceSyncInfo::from_access(access, range),
                        sharing,
                    ));
                }

                None
            }
        }

        let access_runs = self.access_runs.lock();

        #[cfg(not(feature = "parking_lot"))]
        let access_runs = access_runs.expect("poisoned buffer access lock");

        let query_range = query_range.resolve_whole(self.info.size);
        let access_run_idx = access_runs.run_index_at(query_range.start);
        let mut sharing_runs = self.sharing.ranges_in(query_range);
        let sharing_run = sharing_runs.next();

        SyncInfoIter {
            access_runs,
            access_run_idx,
            query_range,
            sharing_run,
            sharing_runs,
        }
    }

    /// Sets the debugging name assigned to this buffer.
    pub fn with_debug_name(self, name: impl AsRef<str>) -> Self {
        self.set_debug_name(name);

        self
    }
}

impl Debug for Buffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut res = f.debug_struct(stringify!(Buffer));

        if let Some(debug_name) =
            &Device::private_data_object_name(&self.device, vk::ObjectType::BUFFER, self.handle)
        {
            res.field("debug_name", debug_name);
        }

        res.field("handle", &self.handle).finish_non_exhaustive()
    }
}

impl Drop for Buffer {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        {
            profiling::scope!("deallocate");

            Device::with_allocator(&self.device, |allocator| {
                allocator.free(unsafe { ManuallyDrop::take(&mut self.allocation) })
            })
        }
        .unwrap_or_else(|err| warn!("unable to free buffer allocation: {err}"));

        Device::try_clear_private_data_object_name(
            &self.device,
            vk::ObjectType::BUFFER,
            self.handle,
        );

        unsafe {
            self.device.destroy_buffer(self.handle, None);
        }
    }
}

impl Eq for Buffer {}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

/// Information used to create a [`Buffer`] instance.
///
/// See [`VkBufferCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferCreateInfo.html).
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct BufferInfo {
    /// Byte alignment of the base device address of the buffer.
    ///
    /// Must be a power of two.
    #[builder(default = "1")]
    pub alignment: vk::DeviceSize,

    /// Specifies a dedicated memory allocation managed by the Vulkan driver and not by the
    /// internal memory allocation pool transient resources share.
    ///
    /// The driver may optimize access to dedicated buffers.
    #[builder(default)]
    pub alloc_dedicated: bool,

    /// Specifies a buffer whose memory is host-visible and may be mapped for reads.
    ///
    /// Memory optimal for CPU readback of data may be used.
    ///
    #[builder(default)]
    pub host_readable: bool,

    /// Specifies a buffer whose memory is host-visible and may be mapped for writes.
    ///
    /// Memory optimal for uploading data to the GPU and potentially for constant buffers may be
    /// used.
    ///
    #[builder(default)]
    pub host_writable: bool,

    /// Controls whether the buffer is accessible from a single queue family (`EXCLUSIVE`) or
    /// from all queues (`CONCURRENT`).
    #[builder(default = "vk::SharingMode::EXCLUSIVE")]
    pub sharing_mode: vk::SharingMode,

    /// Size in bytes of the buffer to be created.
    #[builder(default)]
    pub size: vk::DeviceSize,

    /// A bitmask specifying the allowed usages of the buffer.
    ///
    /// See [`VkBufferUsageFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferUsageFlagBits.html).
    #[builder(default)]
    pub usage: vk::BufferUsageFlags,
}

impl BufferInfo {
    /// Creates a default `BufferInfoBuilder`.
    pub fn builder() -> BufferInfoBuilder {
        Default::default()
    }

    /// Specifies a non-mappable buffer with the given `size` and `usage` values.
    ///
    /// Device-local memory (located on the GPU) is used.
    #[inline(always)]
    pub const fn device_mem(size: vk::DeviceSize, usage: vk::BufferUsageFlags) -> BufferInfo {
        BufferInfo {
            alignment: 1,
            alloc_dedicated: false,
            host_readable: false,
            host_writable: false,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            size,
            usage,
        }
    }

    /// Specifies a mappable buffer with the given `size` and `usage` values.
    ///
    /// Cached, coherent host-visible memory is preferred. Coherent host-visible memory
    /// is used when cached memory is unavailable. For upload-only device-local memory
    /// preference, use the builder with `host_writable(true)` and `host_readable(false)`.
    ///
    /// # Note
    ///
    /// For convenience the given usage value will be bitwise OR'd with
    /// `TRANSFER_DST | TRANSFER_SRC`.
    #[inline(always)]
    pub const fn host_mem(size: vk::DeviceSize, usage: vk::BufferUsageFlags) -> BufferInfo {
        let usage = vk::BufferUsageFlags::from_raw(
            usage.as_raw()
                | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
                | vk::BufferUsageFlags::TRANSFER_SRC.as_raw(),
        );

        BufferInfo {
            alignment: 1,
            alloc_dedicated: false,
            host_readable: true,
            host_writable: true,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            size,
            usage,
        }
    }

    /// Converts a `BufferInfo` into a `BufferInfoBuilder`.
    pub fn into_builder(self) -> BufferInfoBuilder {
        BufferInfoBuilder {
            alignment: Some(self.alignment),
            alloc_dedicated: Some(self.alloc_dedicated),
            host_readable: Some(self.host_readable),
            host_writable: Some(self.host_writable),
            sharing_mode: Some(self.sharing_mode),
            size: Some(self.size),
            usage: Some(self.usage),
        }
    }

    /// Returns `true` if this information specifies host-visible memory.
    pub fn is_host_visible(&self) -> bool {
        self.host_readable | self.host_writable
    }
}

impl From<BufferInfoBuilder> for BufferInfo {
    fn from(info: BufferInfoBuilder) -> Self {
        info.build()
    }
}

impl BufferInfoBuilder {
    /// Builds a new `BufferInfo`.
    ///
    /// If `alignment` is not a power of two and the `checked` feature is active this function will
    /// panic.
    #[inline(always)]
    pub fn build(self) -> BufferInfo {
        let res = self.fallible_build().expect("all fields have defaults");

        #[cfg(feature = "checked")]
        assert!(
            res.alignment.is_power_of_two(),
            "Alignment must be a power of two"
        );

        res
    }
}

/// Specifies a range of buffer data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferSubresourceRange {
    /// The start of range.
    pub start: vk::DeviceSize,

    /// The exclusive end of the range.
    pub end: vk::DeviceSize,
}

impl BufferSubresourceRange {
    pub(crate) fn contains(self, other: Self) -> bool {
        self.start <= other.start && self.end >= other.end
    }

    pub(crate) fn intersection(self, other: Self) -> Option<Self> {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);

        (start < end).then_some(Self { start, end })
    }

    #[cfg(test)]
    pub(crate) fn intersects(self, other: Self) -> bool {
        self.start < other.end && self.end > other.start
    }

    pub(crate) fn resolve_whole(mut self, size: vk::DeviceSize) -> Self {
        if self.end == vk::WHOLE_SIZE {
            self.end = size;
        }

        self
    }
}

impl From<BufferInfo> for BufferSubresourceRange {
    fn from(info: BufferInfo) -> Self {
        Self {
            start: 0,
            end: info.size,
        }
    }
}

impl From<Range<vk::DeviceSize>> for BufferSubresourceRange {
    fn from(range: Range<vk::DeviceSize>) -> Self {
        Self {
            start: range.start,
            end: range.end,
        }
    }
}

impl From<BufferSubresourceRange> for Range<vk::DeviceSize> {
    fn from(range: BufferSubresourceRange) -> Self {
        range.start..range.end
    }
}

/// Synchronization information for one accessed buffer range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferSubresourceSyncInfo {
    /// Access types performed by `stage_mask`.
    pub access_mask: vk::AccessFlags,

    /// Queue-family ownership for this range, when exclusive ownership is known.
    pub queue_family_index: Option<u32>,

    /// The tracked buffer range.
    pub range: BufferSubresourceRange,

    /// Pipeline stages that access `range`.
    pub stage_mask: vk::PipelineStageFlags,
}

impl BufferSubresourceSyncInfo {
    fn can_merge(self, other: Self) -> bool {
        self.stage_mask == other.stage_mask
            && self.access_mask == other.access_mask
            && self.queue_family_index == other.queue_family_index
            && self.range.end == other.range.start
    }

    fn from_access(access: AccessType, range: BufferSubresourceRange) -> Self {
        let (stage_mask, access_mask) = pipeline_stage_access_flags(access);

        Self {
            access_mask,
            queue_family_index: None,
            range,
            stage_mask,
        }
    }

    fn into_public(self, sharing: SharingMode) -> Self {
        Self {
            queue_family_index: match sharing {
                SharingMode::Concurrent | SharingMode::Exclusive(None) => None,
                SharingMode::Exclusive(Some((queue_family_index, _))) => Some(queue_family_index),
            },
            ..self
        }
    }

    fn merge(&mut self, other: Self) {
        self.range.end = other.range.end;
    }
}

/// Synchronization information for a buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferSyncInfo {
    /// Access state for the tracked buffer ranges.
    pub ranges: Box<[BufferSubresourceSyncInfo]>,
}

impl BufferSyncInfo {
    /// Compacts adjacent ranges with identical synchronization requirements.
    ///
    /// Runs in linear time over `ranges`. The implementation reuses the existing range storage by
    /// converting the boxed slice into a vector, compacting entries in place, and converting it back
    /// into a boxed slice.
    pub fn compact(&mut self) {
        let ranges = take(&mut self.ranges);
        let mut ranges = ranges.into_vec();
        let mut compacted_len = 0;

        for idx in 0..ranges.len() {
            let sync_info = ranges[idx];

            if compacted_len > 0 && ranges[compacted_len - 1].can_merge(sync_info) {
                ranges[compacted_len - 1].merge(sync_info);
            } else {
                ranges[compacted_len] = sync_info;
                compacted_len += 1;
            }
        }

        ranges.truncate(compacted_len);
        self.ranges = ranges.into_boxed_slice();
    }

    /// Returns a compacted copy of this synchronization snapshot.
    ///
    /// This has the same linear-time and in-place storage characteristics as [`Self::compact`], but
    /// consumes and returns the snapshot for use in iterator chains or expression-oriented code.
    pub fn into_compacted(mut self) -> Self {
        self.compact();
        self
    }
}

#[derive(Debug)]
struct ExclusiveSharing {
    sharing_runs: Mutex<RunMap<SharingMode>>,
    uniform: AtomicU64,
}

impl ExclusiveSharing {
    // Distinct from Concurrent/Unknown; its family is QUEUE_FAMILY_IGNORED,
    // which cannot identify a real queue. Once published, this marker is permanent.
    const DENSE: u64 = u64::MAX - 2;

    fn new(size: vk::DeviceSize) -> Self {
        let sharing = SharingMode::Exclusive(None);

        Self {
            sharing_runs: Mutex::new(RunMap::new(size, sharing)),
            uniform: AtomicU64::new(sharing.encode()),
        }
    }

    fn promote_and_set_ranges<I>(
        &self,
        size: vk::DeviceSize,
        sharing: SharingMode,
        sharing_ranges: I,
    ) where
        I: Iterator<Item = BufferSubresourceRange>,
    {
        let sharing_runs = self.sharing_runs.lock();

        #[cfg(not(feature = "parking_lot"))]
        let mut sharing_runs = sharing_runs.expect("poisoned buffer sharing lock");

        #[cfg(feature = "parking_lot")]
        let mut sharing_runs = sharing_runs;

        let (min_ranges, _) = sharing_ranges.size_hint();

        // A uniform writer either precedes this swap and is captured here, or its
        // CAS fails and it takes the locked path. Readers also lock on the marker.
        let current = self.uniform.swap(Self::DENSE, Ordering::AcqRel);
        if current != Self::DENSE {
            *sharing_runs = RunMap::new(size, SharingMode::decode(current));
        }

        if min_ranges > 1 {
            sharing_runs.runs.reserve(min_ranges.saturating_mul(2));
        }

        for sharing_range in sharing_ranges {
            sharing_runs.set_range(sharing, sharing_range);
        }
    }

    fn ranges_in(&self, query_range: BufferSubresourceRange) -> SharingRunIter<'_> {
        let current = self.uniform.load(Ordering::Acquire);
        if current != Self::DENSE {
            let sharing = SharingMode::decode(current);

            return SharingRunIter::Constant(Some((sharing, query_range)));
        }

        let sharing_runs = self.sharing_runs.lock();

        #[cfg(not(feature = "parking_lot"))]
        let sharing_runs = sharing_runs.expect("poisoned buffer sharing lock");

        let run_idx = sharing_runs.run_index_at(query_range.start);

        SharingRunIter::Dense {
            query_range,
            run_idx,
            sharing_runs,
        }
    }

    fn set_range(
        &self,
        size: vk::DeviceSize,
        sharing: SharingMode,
        sharing_range: BufferSubresourceRange,
    ) {
        if sharing_range.start == 0 && sharing_range.end == size {
            self.set_uniform_or_dense(sharing, sharing_range);
            return;
        }

        self.promote_and_set_ranges(size, sharing, once(sharing_range));
    }

    fn set_ranges<I>(&self, size: vk::DeviceSize, sharing: SharingMode, sharing_ranges: I)
    where
        I: IntoIterator<Item = BufferSubresourceRange>,
    {
        let mut sharing_ranges = sharing_ranges.into_iter();
        let Some(first) = sharing_ranges.next() else {
            return;
        };

        let Some(second) = sharing_ranges.next() else {
            self.set_range(size, sharing, first);

            return;
        };

        self.promote_and_set_ranges(
            size,
            sharing,
            once(first).chain(once(second)).chain(sharing_ranges),
        );
    }

    fn set_uniform_or_dense(&self, sharing: SharingMode, sharing_range: BufferSubresourceRange) {
        let encoded_sharing = sharing.encode();

        debug_assert_ne!(encoded_sharing, Self::DENSE);

        let mut current = self.uniform.load(Ordering::Acquire);

        loop {
            if current == Self::DENSE {
                let sharing_runs = self.sharing_runs.lock();

                #[cfg(not(feature = "parking_lot"))]
                let mut sharing_runs = sharing_runs.expect("poisoned buffer sharing lock");

                #[cfg(feature = "parking_lot")]
                let mut sharing_runs = sharing_runs;

                sharing_runs.set_range(sharing, sharing_range);

                return;
            }

            #[cfg(test)]
            if let Some(before_cas) = test::BEFORE_SHARING_CAS.with(|hook| hook.take()) {
                before_cas();
            }

            match self.uniform.compare_exchange(
                current,
                encoded_sharing,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }
}

#[derive(Debug)]
struct RunMap<V> {
    runs: SmallVec<[(V, vk::DeviceSize); 4]>,
    size: vk::DeviceSize,
}

impl<V> RunMap<V> {
    fn new(size: vk::DeviceSize, value: V) -> Self {
        Self {
            runs: smallvec![(value, 0)],
            size,
        }
    }

    fn run_index_at(&self, offset: vk::DeviceSize) -> usize {
        let needle = (offset << 1) | 1;
        let run_idx = self
            .runs
            .binary_search_by(|(_, probe)| (probe << 1).cmp(&needle));

        debug_assert!(run_idx.is_err());

        let run_idx = {
            #[cfg(feature = "checked")]
            {
                run_idx.unwrap_err()
            }

            #[cfg(not(feature = "checked"))]
            unsafe {
                run_idx.unwrap_err_unchecked()
            }
        };

        run_idx.saturating_sub(1)
    }

    /// Assigns without visiting old values, retaining only the boundary runs.
    fn set_range(&mut self, value: V, range: BufferSubresourceRange)
    where
        V: Copy + PartialEq,
    {
        debug_assert!(range.start < range.end && range.end <= self.size);

        if range.start == 0 && range.end == self.size {
            self.runs.truncate(1);
            self.runs[0] = (value, 0);
            return;
        }

        let start_idx = self.run_index_at(range.start);
        if self.runs[start_idx].0 == value
            && self
                .runs
                .get(start_idx + 1)
                .is_none_or(|(_, start)| range.end <= *start)
        {
            return;
        }

        let end_idx = self.run_index_at(range.end);
        let right = (range.end < self.size).then(|| self.runs[end_idx].0);
        let mut insert_idx = start_idx + usize::from(self.runs[start_idx].1 < range.start);
        let removed = end_idx + 1 - insert_idx;

        #[cfg(test)]
        if removed != 0 {
            test::SET_RANGE_SHIFTED_RUNS.with(|count| {
                count.set(count.get() + self.runs.len() - (end_idx + 1));
            });
        }

        self.runs.copy_within(end_idx + 1.., insert_idx);
        self.runs.truncate(self.runs.len() - removed);

        if insert_idx == 0 || self.runs[insert_idx - 1].0 != value {
            self.runs.insert(insert_idx, (value, range.start));
            insert_idx += 1;
        }

        if let Some(right) = right.filter(|right| *right != value) {
            self.runs.insert(insert_idx, (right, range.end));
        }
    }
}

struct RunMapCursor {
    run_idx: usize,
    write_idx: usize,
    remaining_range: BufferSubresourceRange,
}

impl RunMapCursor {
    fn new<V>(map: &RunMap<V>, remaining_range: BufferSubresourceRange) -> Self
    where
        V: Copy + PartialEq + Debug,
    {
        debug_assert!(remaining_range.start < remaining_range.end);
        debug_assert!(remaining_range.end <= map.size);

        #[cfg(feature = "checked")]
        {
            let run_start = |(_, start): &(V, vk::DeviceSize)| *start;

            assert_eq!(map.runs.first().map(run_start), Some(0));
            assert!(map.runs.last().map(run_start).unwrap() < map.size);

            // Custom is-sorted-by key to additionally check that all run starts are unique
            let (mut prev_value, mut prev_start) = map.runs.first().copied().unwrap();
            for (next_value, next_start) in map.runs.iter().skip(1).copied() {
                debug_assert_ne!(prev_value, next_value);
                debug_assert!(prev_start < next_start);

                prev_value = next_value;
                prev_start = next_start;
            }
        };

        // The needle will always be odd, and the probe always even, the result will always be err
        let needle = (remaining_range.start << 1) | 1;
        let run_idx = map
            .runs
            .binary_search_by(|(_, probe)| (probe << 1).cmp(&needle));

        debug_assert!(run_idx.is_err());

        let mut run_idx = {
            #[cfg(feature = "checked")]
            {
                run_idx.unwrap_err()
            }

            #[cfg(not(feature = "checked"))]
            unsafe {
                run_idx.unwrap_err_unchecked()
            }
        };

        // The first access will always be at start == 0, which is even, so run_idx cannot be 0
        debug_assert_ne!(run_idx, 0);

        run_idx -= 1;

        Self {
            remaining_range,
            run_idx,
            write_idx: run_idx,
        }
    }

    fn next_with<V>(
        &mut self,
        map: &mut RunMap<V>,
        new_value: impl FnOnce(V) -> V,
    ) -> Option<(V, BufferSubresourceRange)>
    where
        V: Copy + PartialEq + Debug,
    {
        debug_assert!(self.remaining_range.start <= self.remaining_range.end);
        debug_assert!(self.remaining_range.end <= map.size);

        if self.remaining_range.start == self.remaining_range.end {
            return None;
        }

        debug_assert!(map.runs.get(self.run_idx).is_some());

        let (old_value, old_start) = unsafe { *map.runs.get_unchecked(self.run_idx) };

        let new_value = new_value(old_value);
        let old_end = map
            .runs
            .get(self.run_idx + 1)
            .map(|(_, start)| *start)
            .unwrap_or(map.size);
        let mut remaining_range = self.remaining_range;

        remaining_range.end = remaining_range.end.min(old_end);
        self.remaining_range.start = remaining_range.end;

        let mut new_start = remaining_range.start;
        if old_start < new_start {
            if old_value == new_value {
                new_start = old_start;
            } else {
                // Only the first run can need a left-boundary split.
                self.run_idx += 1;
                self.write_idx += 1;
                map.runs.insert(self.run_idx, (old_value, new_start));
            }
        }

        // Compact behind the read index without shifting unvisited old values.
        // The gap is private to this cursor until its final next (also on Drop).
        if self.write_idx == 0 || map.runs[self.write_idx - 1].0 != new_value {
            map.runs[self.write_idx] = (new_value, new_start);
            self.write_idx += 1;
        }

        self.run_idx += 1;

        if self.remaining_range.start == self.remaining_range.end {
            if old_end > remaining_range.end && old_value != new_value {
                if self.write_idx < self.run_idx {
                    map.runs[self.write_idx] = (old_value, remaining_range.end);
                    self.write_idx += 1;
                } else {
                    map.runs
                        .insert(self.run_idx, (old_value, remaining_range.end));
                }
            }

            if map
                .runs
                .get(self.run_idx)
                .is_some_and(|(value, _)| *value == map.runs[self.write_idx - 1].0)
            {
                self.run_idx += 1;
            }

            let removed = self.run_idx - self.write_idx;
            map.runs.copy_within(self.run_idx.., self.write_idx);
            map.runs.truncate(map.runs.len() - removed);
        }

        Some((old_value, remaining_range))
    }
}

#[allow(
    dead_code,
    reason = "The single-range access path is currently only used in tests"
)]
struct RunMapIter<M, V>
where
    M: DerefMut<Target = RunMap<V>>,
    V: Copy + PartialEq + Debug,
{
    cursor: RunMapCursor,
    map: M,
    new_value: V,
}

impl<M, V> RunMapIter<M, V>
where
    M: DerefMut<Target = RunMap<V>>,
    V: Copy + PartialEq + Debug,
{
    #[allow(
        dead_code,
        reason = "The single-range access path is currently only used in tests"
    )]
    fn new(map: M, new_value: V, remaining_range: BufferSubresourceRange) -> Self {
        let cursor = RunMapCursor::new(&map, remaining_range);

        Self {
            cursor,
            map,
            new_value,
        }
    }
}

impl<M, V> Iterator for RunMapIter<M, V>
where
    M: DerefMut<Target = RunMap<V>>,
    V: Copy + PartialEq + Debug,
{
    type Item = (V, BufferSubresourceRange);

    fn next(&mut self) -> Option<Self::Item> {
        self.cursor.next_with(&mut self.map, |_| self.new_value)
    }
}

impl<M, V> Drop for RunMapIter<M, V>
where
    M: DerefMut<Target = RunMap<V>>,
    V: Copy + PartialEq + Debug,
{
    fn drop(&mut self) {
        while self.next().is_some() {}
    }
}

#[derive(Debug)]
enum Sharing {
    Concurrent,
    Exclusive(ExclusiveSharing),
}

impl Sharing {
    fn new(size: vk::DeviceSize, sharing_mode: vk::SharingMode) -> Self {
        if sharing_mode == vk::SharingMode::CONCURRENT {
            Self::Concurrent
        } else {
            Self::Exclusive(ExclusiveSharing::new(size))
        }
    }

    fn ranges_in(&self, range: BufferSubresourceRange) -> SharingRunIter<'_> {
        match self {
            Self::Concurrent => SharingRunIter::Constant(Some((SharingMode::Concurrent, range))),
            Self::Exclusive(sharing) => sharing.ranges_in(range),
        }
    }

    fn set_ranges<I>(&self, size: vk::DeviceSize, sharing: SharingMode, sharing_ranges: I)
    where
        I: IntoIterator<Item = BufferSubresourceRange>,
    {
        if let Self::Exclusive(exclusive) = self {
            exclusive.set_ranges(size, sharing, sharing_ranges);
        }
    }
}

enum SharingRunIter<'a> {
    Constant(Option<(SharingMode, BufferSubresourceRange)>),
    Dense {
        query_range: BufferSubresourceRange,
        run_idx: usize,
        sharing_runs: MutexGuard<'a, RunMap<SharingMode>>,
    },
}

impl Iterator for SharingRunIter<'_> {
    type Item = (SharingMode, BufferSubresourceRange);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Constant(range) => range.take(),
            Self::Dense {
                query_range,
                run_idx,
                sharing_runs,
            } => {
                let &(sharing, start) = sharing_runs.runs.get(*run_idx)?;
                if start >= query_range.end {
                    return None;
                }

                let end = sharing_runs
                    .runs
                    .get(*run_idx + 1)
                    .map(|(_, next_start)| *next_start)
                    .unwrap_or(sharing_runs.size);

                *run_idx += 1;

                let range = BufferSubresourceRange { start, end }.intersection(*query_range)?;

                Some((sharing, range))
            }
        }
    }
}

#[cfg(test)]
mod test {
    use {
        super::*,
        rand::{Rng, SeedableRng, rngs::SmallRng},
    };

    type Info = BufferInfo;
    type Builder = BufferInfoBuilder;

    const FUZZ_COUNT: usize = 100_000;

    std::thread_local! {
        // Count entries shifted by set_range's tail removal, not comparisons or elapsed time.
        pub(super) static SET_RANGE_SHIFTED_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        pub(super) static BEFORE_SHARING_CAS: std::cell::Cell<Option<Box<dyn FnOnce()>>> = const { std::cell::Cell::new(None) };
    }

    #[test]
    fn buffer_sharing_whole_update_retries_after_promotion() {
        for bulk in [false, true] {
            let sharing = std::sync::Arc::new(ExclusiveSharing::new(16));
            let whole_owner = SharingMode::Exclusive(Some((1, 0)));
            let partial_owner = SharingMode::Exclusive(Some((2, 1)));
            let promoter = sharing.clone();
            BEFORE_SHARING_CAS.with(|hook| {
                hook.set(Some(Box::new(move || {
                    // Complete promotion after the whole-buffer writer has loaded its
                    // expected owner, but before it attempts the CAS.
                    if bulk {
                        promoter.set_ranges(16, partial_owner, [(4..8).into(), (10..12).into()]);
                    } else {
                        promoter.set_range(16, partial_owner, (4..8).into());
                    }
                })));
            });
            sharing.set_range(16, whole_owner, (0..16).into());
            assert_eq!(
                sharing.ranges_in((0..16).into()).collect::<Vec<_>>(),
                vec![(whole_owner, (0..16).into())],
                "bulk={bulk}"
            );
        }
    }

    #[test]
    fn buffer_sharing_uniform_updates_survive_promotion() {
        for bulk in [false, true] {
            for whole_owner in [
                SharingMode::Exclusive(None),
                SharingMode::Concurrent,
                SharingMode::Exclusive(Some((1, 0))),
            ] {
                let sharing = ExclusiveSharing::new(16);
                let partial_owner = SharingMode::Exclusive(Some((2, 1)));
                sharing.set_range(16, whole_owner, (0..16).into());
                assert_eq!(
                    sharing.uniform.load(Ordering::Acquire),
                    whole_owner.encode()
                );
                assert_eq!(
                    sharing.ranges_in((0..16).into()).collect::<Vec<_>>(),
                    vec![(whole_owner, (0..16).into())]
                );

                if bulk {
                    sharing.set_ranges(16, partial_owner, [(4..6).into(), (6..8).into()]);
                } else {
                    sharing.set_range(16, partial_owner, (4..8).into());
                }
                assert_eq!(
                    sharing.ranges_in((0..16).into()).collect::<Vec<_>>(),
                    vec![
                        (whole_owner, (0..4).into()),
                        (partial_owner, (4..8).into()),
                        (whole_owner, (8..16).into()),
                    ],
                    "bulk={bulk}, whole_owner={whole_owner:?}"
                );
            }
        }
    }

    #[test]
    fn run_map_set_range_noops_do_not_shift_tail() {
        let mut map = RunMap {
            runs: (0..1024).map(|idx| ((idx % 2) as u8, idx * 4)).collect(),
            size: 4096,
        };
        let expected = map.runs.clone();
        SET_RANGE_SHIFTED_RUNS.with(|count| count.set(0));
        for _ in 0..128 {
            for range in [8..9, 9..11, 8..12, 9..12] {
                map.set_range(0, range.into());
            }
        }
        assert_eq!(map.runs, expected);
        SET_RANGE_SHIFTED_RUNS.with(|count| assert_eq!(count.get(), 0));
    }

    #[test]
    fn run_map_set_range_matches_exhaustive_oracle() {
        const SIZE: usize = 8;
        let compact = |data: &[u8]| -> SmallVec<[(u8, vk::DeviceSize); 4]> {
            data.iter()
                .copied()
                .enumerate()
                .filter(|&(idx, value)| idx == 0 || data[idx - 1] != value)
                .map(|(idx, value)| (value, idx as vk::DeviceSize))
                .collect()
        };
        for bits in 0..1 << SIZE {
            let data: [u8; SIZE] = std::array::from_fn(|idx| ((bits >> idx) & 1) as u8);
            for start in 0..SIZE {
                for end in start + 1..=SIZE {
                    for value in 0..3 {
                        let mut map = RunMap {
                            runs: compact(&data),
                            size: SIZE as vk::DeviceSize,
                        };
                        let mut expected = data;
                        expected[start..end].fill(value);
                        let expected = compact(&expected);
                        for _ in 0..2 {
                            map.set_range(
                                value,
                                (start as vk::DeviceSize..end as vk::DeviceSize).into(),
                            );
                            assert_eq!(
                                map.runs, expected,
                                "bits={bits} range={start}..{end} value={value}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fragmented_cursor_defers_tail_shifts() {
        let mut map = RunMap {
            runs: (0..1024).map(|idx| ((idx % 2) as u8, idx)).collect(),
            size: 1024,
        };
        let mut iter = RunMapIter::new(&mut map, 2, (0..1024).into());
        assert_eq!(iter.map.runs[0], (0, 0));
        for idx in 0..128 {
            assert_eq!(iter.next(), Some(((idx % 2) as u8, (idx..idx + 1).into())));
        }
        // Unvisited entries stay in place until the cursor finishes, rather than
        // shifting the entire tail once for every merged run.
        assert_eq!(iter.map.runs.len(), 1024);
        assert_eq!(iter.map.runs[128], (0, 128));
        drop(iter);
        assert_eq!(map.runs.as_slice(), &[(2, 0)]);
    }

    #[test]
    fn fragmented_cursor_partial_drop_preserves_boundaries() {
        for consumed in [0, 1, 17, 63] {
            let mut map = RunMap {
                runs: (0..64).map(|idx| ((idx % 2) as u8, idx * 4)).collect(),
                size: 256,
            };
            let mut iter = RunMapIter::new(&mut map, 2, (2..254).into());
            for idx in 0..consumed {
                assert_eq!(
                    iter.next(),
                    Some((
                        (idx % 2) as u8,
                        ((idx * 4).max(2)..(idx * 4 + 4).min(254)).into()
                    ))
                );
            }
            drop(iter);
            assert_eq!(map.runs.as_slice(), &[(0, 0), (2, 2), (1, 254)]);
        }
    }

    #[test]
    fn fragmented_cursor_transforms_each_old_value_lazily() {
        let mut map = AccessRuns {
            runs: (0..64)
                .map(|idx| {
                    (
                        if idx % 2 == 0 {
                            AccessType::TransferWrite
                        } else {
                            AccessType::TransferRead
                        },
                        idx * 4,
                    )
                })
                .collect(),
            size: 256,
        };
        let mut cursor = RunMapCursor::new(&map, (2..254).into());
        let mut calls = 0;
        for idx in 0..64 {
            let expected = if idx % 2 == 0 {
                AccessType::TransferWrite
            } else {
                AccessType::TransferRead
            };
            assert_eq!(
                cursor.next_with(&mut map, |old| {
                    calls += 1;
                    assert_eq!(old, expected);
                    tracked_access_after(old, AccessType::HostRead)
                }),
                Some((expected, ((idx * 4).max(2)..(idx * 4 + 4).min(254)).into()))
            );
            assert_eq!(calls, idx + 1);
        }
        assert!(
            cursor
                .next_with(&mut map, |_| panic!("exhausted cursor transformed a value"))
                .is_none()
        );
        let mut expected = vec![(AccessType::TransferWrite, 0)];
        expected.extend((0..64).map(|idx| {
            (
                if idx % 2 == 0 {
                    AccessType::General
                } else {
                    AccessType::HostRead
                },
                (idx * 4).max(2),
            )
        }));
        expected.push((AccessType::TransferRead, 254));
        assert_eq!(map.runs.as_slice(), expected);
    }

    #[test]
    fn fragmented_ownership_bulk_assignment_preserves_gaps_and_resets() {
        let sharing = ExclusiveSharing::new(256);
        let owner_a = SharingMode::Exclusive(Some((1, 0)));
        let owner_b = SharingMode::Exclusive(Some((2, 1)));
        sharing.set_ranges(
            256,
            owner_a,
            (0..128).map(|idx| (idx * 2..idx * 2 + 1).into()),
        );
        sharing.set_ranges(
            256,
            owner_b,
            [(3..101).into(), (99..201).into(), (220..240).into()],
        );
        for offset in 0..256 {
            let expected = if (3..201).contains(&offset) || (220..240).contains(&offset) {
                owner_b
            } else if offset % 2 == 0 {
                owner_a
            } else {
                SharingMode::Exclusive(None)
            };
            assert_eq!(
                sharing
                    .ranges_in((offset..offset + 1).into())
                    .collect::<Vec<_>>(),
                vec![(expected, (offset..offset + 1).into())]
            );
        }
        for ranges in [
            vec![(0..256).into()],
            vec![(0..128).into(), (128..256).into()],
        ] {
            sharing.set_ranges(256, owner_a, ranges);
            assert_eq!(
                sharing.ranges_in((0..256).into()).collect::<Vec<_>>(),
                vec![(owner_a, (0..256).into())]
            );
            assert_eq!(
                sharing.uniform.load(Ordering::Acquire),
                ExclusiveSharing::DENSE
            );
            sharing.set_range(256, owner_b, (80..160).into());
        }
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn mapped_upload_payloads() {
        let device = Device::create(crate::driver::device::DeviceInfo::default()).unwrap();
        let info = BufferInfo::host_mem(65536, vk::BufferUsageFlags::STORAGE_BUFFER);
        let readback = Buffer::create(&device, info.into_builder().host_writable(false)).unwrap();
        let mut buf = Buffer::create(&device, info).unwrap();

        assert_eq!(
            buf.allocation.memory_properties(),
            readback.allocation.memory_properties()
        );
        assert!(
            buf.allocation
                .memory_properties()
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        );

        for (size, offset) in [(0, 65536), (384, 0), (33792, 0), (34560, 16)] {
            let data = (0..size).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();
            buf.mapped_slice_mut().fill(0xa5);
            buf.copy_from_slice(offset as u64, &data);
            let mapped = buf.mapped_slice();

            assert_eq!(&mapped[offset..offset + size], data);
            assert!(mapped[..offset].iter().all(|byte| *byte == 0xa5));
            assert!(mapped[offset + size..].iter().all(|byte| *byte == 0xa5));
        }
    }

    fn buffer_sync_info(range: Range<vk::DeviceSize>) -> BufferSubresourceSyncInfo {
        BufferSubresourceSyncInfo {
            access_mask: vk::AccessFlags::SHADER_READ,
            queue_family_index: None,
            range: buffer_subresource_range(range),
            stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
        }
    }

    fn assert_access_runs_eq(access_runs: &AccessRuns, expected: &[(AccessType, vk::DeviceSize)]) {
        assert_eq!(access_runs.runs.as_slice(), expected);
    }

    #[test]
    fn tracked_access_preserves_write_and_later_read_dependencies() {
        let write = AccessType::TransferWrite;

        assert_eq!(
            tracked_access_after(write, AccessType::ComputeShaderReadOther),
            AccessType::General
        );
        assert_eq!(
            tracked_access_after(AccessType::General, AccessType::TransferRead),
            AccessType::General
        );
        assert_eq!(
            tracked_access_after(write, AccessType::ComputeShaderWrite),
            AccessType::ComputeShaderWrite
        );
        assert_eq!(
            tracked_access_after(AccessType::ComputeShaderReadOther, AccessType::TransferRead),
            AccessType::TransferRead
        );
    }

    #[test]
    pub fn buffer_access() {
        let mut access_runs = AccessRuns::new(100, AccessType::Nothing);

        {
            let mut accesses = RunMapIter::new(
                &mut access_runs,
                AccessType::TransferWrite,
                buffer_subresource_range(0..10),
            );

            assert_access_runs_eq(accesses.map, &[(AccessType::Nothing, 0)]);
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::Nothing, buffer_subresource_range(0..10))
            );
            assert_access_runs_eq(
                accesses.map,
                &[(AccessType::TransferWrite, 0), (AccessType::Nothing, 10)],
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = RunMapIter::new(
                &mut access_runs,
                AccessType::TransferRead,
                buffer_subresource_range(5..15),
            );

            assert_access_runs_eq(
                accesses.map,
                &[(AccessType::TransferWrite, 0), (AccessType::Nothing, 10)],
            );
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::TransferWrite, buffer_subresource_range(5..10))
            );
            assert_access_runs_eq(
                accesses.map,
                &[
                    (AccessType::TransferWrite, 0),
                    (AccessType::TransferRead, 5),
                    (AccessType::Nothing, 10),
                ],
            );
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::Nothing, buffer_subresource_range(10..15))
            );
            assert_access_runs_eq(
                accesses.map,
                &[
                    (AccessType::TransferWrite, 0),
                    (AccessType::TransferRead, 5),
                    (AccessType::Nothing, 15),
                ],
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = RunMapIter::new(
                &mut access_runs,
                AccessType::HostRead,
                buffer_subresource_range(0..100),
            );

            assert_access_runs_eq(
                accesses.map,
                &[
                    (AccessType::TransferWrite, 0),
                    (AccessType::TransferRead, 5),
                    (AccessType::Nothing, 15),
                ],
            );
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::TransferWrite, buffer_subresource_range(0..5))
            );
            assert_access_runs_eq(
                accesses.map,
                &[
                    (AccessType::HostRead, 0),
                    (AccessType::TransferRead, 5),
                    (AccessType::Nothing, 15),
                ],
            );
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::TransferRead, buffer_subresource_range(5..15))
            );
            assert_access_runs_eq(
                accesses.map,
                &[
                    (AccessType::HostRead, 0),
                    // Compaction leaves a gap until the last old run is visited.
                    (AccessType::TransferRead, 5),
                    (AccessType::Nothing, 15),
                ],
            );
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::Nothing, buffer_subresource_range(15..100))
            );
            assert_access_runs_eq(accesses.map, &[(AccessType::HostRead, 0)]);
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = RunMapIter::new(
                &mut access_runs,
                AccessType::HostWrite,
                buffer_subresource_range(0..100),
            );

            assert_access_runs_eq(accesses.map, &[(AccessType::HostRead, 0)]);
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::HostRead, buffer_subresource_range(0..100))
            );
            assert_access_runs_eq(accesses.map, &[(AccessType::HostWrite, 0)]);
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = RunMapIter::new(
                &mut access_runs,
                AccessType::HostWrite,
                buffer_subresource_range(0..100),
            );

            assert_access_runs_eq(accesses.map, &[(AccessType::HostWrite, 0)]);
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::HostWrite, buffer_subresource_range(0..100))
            );
            assert_access_runs_eq(accesses.map, &[(AccessType::HostWrite, 0)]);
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = RunMapIter::new(
                &mut access_runs,
                AccessType::HostWrite,
                buffer_subresource_range(1..99),
            );

            assert_access_runs_eq(accesses.map, &[(AccessType::HostWrite, 0)]);
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::HostWrite, buffer_subresource_range(1..99))
            );
            assert_access_runs_eq(accesses.map, &[(AccessType::HostWrite, 0)]);
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = RunMapIter::new(
                &mut access_runs,
                AccessType::HostRead,
                buffer_subresource_range(1..99),
            );

            assert_access_runs_eq(accesses.map, &[(AccessType::HostWrite, 0)]);
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::HostWrite, buffer_subresource_range(1..99))
            );
            assert_access_runs_eq(
                accesses.map,
                &[
                    (AccessType::HostWrite, 0),
                    (AccessType::HostRead, 1),
                    (AccessType::HostWrite, 99),
                ],
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = RunMapIter::new(
                &mut access_runs,
                AccessType::Nothing,
                buffer_subresource_range(0..100),
            );

            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::HostWrite, buffer_subresource_range(0..1))
            );
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::HostRead, buffer_subresource_range(1..99))
            );
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::HostWrite, buffer_subresource_range(99..100))
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = RunMapIter::new(
                &mut access_runs,
                AccessType::AnyShaderWrite,
                buffer_subresource_range(0..100),
            );

            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::Nothing, buffer_subresource_range(0..100))
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = RunMapIter::new(
                &mut access_runs,
                AccessType::AnyShaderReadOther,
                buffer_subresource_range(1..2),
            );

            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::AnyShaderWrite, buffer_subresource_range(1..2))
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = RunMapIter::new(
                &mut access_runs,
                AccessType::AnyShaderReadOther,
                buffer_subresource_range(3..4),
            );

            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::AnyShaderWrite, buffer_subresource_range(3..4))
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = RunMapIter::new(
                &mut access_runs,
                AccessType::Nothing,
                buffer_subresource_range(0..5),
            );

            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::AnyShaderWrite, buffer_subresource_range(0..1))
            );
            assert_eq!(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderReadOther,
                    buffer_subresource_range(1..2)
                )
            );
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::AnyShaderWrite, buffer_subresource_range(2..3))
            );
            assert_eq!(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderReadOther,
                    buffer_subresource_range(3..4)
                )
            );
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::AnyShaderWrite, buffer_subresource_range(4..5))
            );
            assert!(accesses.next().is_none());
        }
    }

    #[test]
    pub fn buffer_access_basic() {
        let mut access_runs = AccessRuns::new(5, AccessType::Nothing);

        access_runs.runs = smallvec![
            (AccessType::ColorAttachmentRead, 0),
            (AccessType::AnyShaderWrite, 4),
        ];

        {
            let mut accesses = RunMapIter::new(
                &mut access_runs,
                AccessType::AnyShaderWrite,
                buffer_subresource_range(0..2),
            );

            assert_eq!(
                accesses.next().unwrap(),
                (
                    AccessType::ColorAttachmentRead,
                    buffer_subresource_range(0..2)
                )
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = RunMapIter::new(
                &mut access_runs,
                AccessType::HostWrite,
                buffer_subresource_range(0..5),
            );

            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::AnyShaderWrite, buffer_subresource_range(0..2))
            );
            assert_eq!(
                accesses.next().unwrap(),
                (
                    AccessType::ColorAttachmentRead,
                    buffer_subresource_range(2..4)
                )
            );
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::AnyShaderWrite, buffer_subresource_range(4..5))
            );

            assert!(accesses.next().is_none());
        }
    }

    #[test]
    pub fn buffer_sharing_ranges_in_clips_dense_runs_to_query_range() {
        let sharing = ExclusiveSharing::new(16);
        let owner_a = SharingMode::Exclusive(Some((1, 0)));
        let owner_b = SharingMode::Exclusive(Some((2, 0)));
        let range_a = buffer_subresource_range(0..8);
        let range_b = buffer_subresource_range(8..16);
        let query_range = buffer_subresource_range(4..12);

        sharing.set_ranges(16, owner_a, [range_a]);
        sharing.set_ranges(16, owner_b, [range_b]);

        let ranges = sharing.ranges_in(query_range).collect::<Vec<_>>();

        assert_eq!(
            ranges,
            vec![
                (owner_a, buffer_subresource_range(4..8)),
                (owner_b, buffer_subresource_range(8..12)),
            ]
        );
    }

    fn buffer_access_fuzz(buffer_size: vk::DeviceSize) {
        static ACCESS_TYPES: &[AccessType] = &[
            AccessType::AnyShaderReadOther,
            AccessType::AnyShaderWrite,
            AccessType::ColorAttachmentRead,
            AccessType::ColorAttachmentWrite,
            AccessType::HostRead,
            AccessType::HostWrite,
            AccessType::Nothing,
        ];

        let mut rng = SmallRng::seed_from_u64(42);
        let mut access_runs = AccessRuns::new(buffer_size, AccessType::Nothing);
        let mut data = vec![AccessType::Nothing; buffer_size as usize];

        for _ in 0..FUZZ_COUNT {
            let access = ACCESS_TYPES[rng.random_range(..ACCESS_TYPES.len())];
            let access_start = rng.random_range(..buffer_size);
            let access_end = rng.random_range(access_start + 1..=buffer_size);

            // println!("{access:?} {access_start}..{access_end}");

            let accesses = RunMapIter::new(
                &mut access_runs,
                access,
                buffer_subresource_range(access_start..access_end),
            );

            for (access, access_range) in accesses {
                // println!("\t{access:?} {}..{}", access_range.start, access_range.end);
                assert!(
                    data[access_range.start as usize..access_range.end as usize]
                        .iter()
                        .all(|data| *data == access),
                    "{:?}",
                    &data[access_range.start as usize..access_range.end as usize]
                );
            }

            for data in &mut data[access_start as usize..access_end as usize] {
                *data = access;
            }
        }
    }

    #[test]
    pub fn buffer_access_fuzz_small() {
        buffer_access_fuzz(5);
    }

    #[test]
    pub fn buffer_access_fuzz_medium() {
        buffer_access_fuzz(101);
    }

    #[test]
    pub fn buffer_access_fuzz_large() {
        buffer_access_fuzz(10_000);
    }

    #[test]
    pub fn buffer_sync_info_compact_merges_adjacent_equal_ranges() {
        let mut sync_info = BufferSyncInfo {
            ranges: vec![
                buffer_sync_info(0..4),
                buffer_sync_info(4..8),
                BufferSubresourceSyncInfo {
                    access_mask: vk::AccessFlags::SHADER_WRITE,
                    queue_family_index: None,
                    range: buffer_subresource_range(8..12),
                    stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
                },
            ]
            .into_boxed_slice(),
        };

        sync_info.compact();

        assert_eq!(sync_info.ranges.len(), 2);
        assert_eq!(sync_info.ranges[0], buffer_sync_info(0..8));
        assert_eq!(
            sync_info.ranges[1],
            BufferSubresourceSyncInfo {
                access_mask: vk::AccessFlags::SHADER_WRITE,
                queue_family_index: None,
                range: buffer_subresource_range(8..12),
                stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
            }
        );
    }

    #[test]
    pub fn buffer_sync_info_into_compacted_preserves_non_adjacent_ranges() {
        let sync_info = BufferSyncInfo {
            ranges: vec![
                BufferSubresourceSyncInfo {
                    queue_family_index: Some(3),
                    ..buffer_sync_info(0..4)
                },
                BufferSubresourceSyncInfo {
                    queue_family_index: Some(3),
                    ..buffer_sync_info(5..9)
                },
            ]
            .into_boxed_slice(),
        };

        let sync_info = sync_info.into_compacted();

        assert_eq!(sync_info.ranges.len(), 2);
        assert_eq!(sync_info.ranges[0].queue_family_index, Some(3));
        assert_eq!(sync_info.ranges[1].queue_family_index, Some(3));
        assert_eq!(sync_info.ranges[0].range, buffer_subresource_range(0..4));
        assert_eq!(sync_info.ranges[1].range, buffer_subresource_range(5..9));
    }

    #[test]
    pub fn buffer_info() {
        let info = Info::device_mem(0, vk::BufferUsageFlags::empty());
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn buffer_info_alignment() {
        let info = Info::device_mem(0, vk::BufferUsageFlags::empty());

        assert_eq!(info.alignment, 1);
    }

    #[test]
    pub fn buffer_info_builder() {
        let info = Info::device_mem(0, vk::BufferUsageFlags::empty());
        let builder = Builder::default().size(0).build();

        assert_eq!(info, builder);
    }

    #[test]
    #[cfg_attr(
        feature = "checked",
        should_panic(expected = "Alignment must be a power of two")
    )]
    pub fn buffer_info_builder_alignment_0() {
        Builder::default().size(0).alignment(0).build();
    }

    #[test]
    pub fn buffer_info_builder_alignment_256() {
        let mut info = Info::device_mem(42, vk::BufferUsageFlags::empty());
        info.alignment = 256;

        let builder = Builder::default().size(42).alignment(256).build();

        assert_eq!(info, builder);
    }

    #[test]
    #[cfg_attr(
        feature = "checked",
        should_panic(expected = "Alignment must be a power of two")
    )]
    pub fn buffer_info_builder_alignment_42() {
        Builder::default().size(0).alignment(42).build();
    }

    #[test]
    pub fn buffer_info_builder_default_size() {
        assert_eq!(
            Builder::default().build(),
            Info::device_mem(0, vk::BufferUsageFlags::empty())
        );
    }

    fn buffer_subresource_range(
        Range { start, end }: Range<vk::DeviceSize>,
    ) -> BufferSubresourceRange {
        BufferSubresourceRange { start, end }
    }

    #[test]
    pub fn buffer_subresource_range_intersects() {
        use BufferSubresourceRange as B;

        assert!(!B { start: 10, end: 20 }.intersects(B { start: 0, end: 5 }));
        assert!(!B { start: 10, end: 20 }.intersects(B { start: 5, end: 10 }));
        assert!(B { start: 10, end: 20 }.intersects(B { start: 10, end: 15 }));
        assert!(B { start: 10, end: 20 }.intersects(B { start: 15, end: 20 }));
        assert!(!B { start: 10, end: 20 }.intersects(B { start: 20, end: 25 }));
        assert!(!B { start: 10, end: 20 }.intersects(B { start: 25, end: 30 }));

        assert!(!B { start: 5, end: 10 }.intersects(B { start: 10, end: 20 }));
        assert!(B { start: 5, end: 25 }.intersects(B { start: 10, end: 20 }));
        assert!(B { start: 5, end: 15 }.intersects(B { start: 10, end: 20 }));
        assert!(B { start: 10, end: 20 }.intersects(B { start: 10, end: 20 }));
        assert!(B { start: 11, end: 19 }.intersects(B { start: 10, end: 20 }));
        assert!(B { start: 15, end: 25 }.intersects(B { start: 10, end: 20 }));
        assert!(!B { start: 20, end: 25 }.intersects(B { start: 10, end: 20 }));
    }
}
