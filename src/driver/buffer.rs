//! Buffer resource types

use {
    super::{DriverError, SharingMode, device::Device, pipeline_stage_access_flags},
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
        sync::atomic::{AtomicU8, AtomicU64, Ordering},
        thread::panicking,
    },
    vk_sync::AccessType,
};

#[cfg(feature = "parking_lot")]
use parking_lot::{Mutex, MutexGuard};

#[cfg(not(feature = "parking_lot"))]
use std::sync::{Mutex, MutexGuard};

type AccessRuns = RunMap<AccessType>;

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
        let location = if info.host_writable {
            MemoryLocation::CpuToGpu
        } else if info.host_readable {
            MemoryLocation::GpuToCpu
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
                        if let Some((prev_access, range)) =
                            cursor.next(&mut self.access_runs, *next_access)
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
    /// Host-local memory (located in CPU-accessible RAM) is used.
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
    sharing_runs_state: AtomicU8,
    uniform: AtomicU64,
}

impl ExclusiveSharing {
    fn new(size: vk::DeviceSize) -> Self {
        let sharing = SharingMode::Exclusive(None);

        Self {
            sharing_runs: Mutex::new(RunMap::new(size, sharing)),
            sharing_runs_state: AtomicU8::new(RunTrackingState::Uniform as _),
            uniform: AtomicU64::new(sharing.encode()),
        }
    }

    fn is_sharing_runs_active(&self) -> bool {
        self.sharing_runs_state() == RunTrackingState::Dense
    }

    fn is_promoting(&self) -> bool {
        self.sharing_runs_state() == RunTrackingState::Promoting
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
        sharing_runs.runs.reserve(min_ranges.saturating_mul(2));

        if self.is_sharing_runs_active() {
            for sharing_range in sharing_ranges {
                RunMapIter::new(&mut *sharing_runs, sharing, sharing_range).finish();
            }

            return;
        }

        self.set_promoting();
        let current = SharingMode::decode(self.uniform.load(Ordering::Acquire));
        *sharing_runs = RunMap::new(size, current);
        sharing_runs.runs.reserve(min_ranges.saturating_mul(2));

        for sharing_range in sharing_ranges {
            RunMapIter::new(&mut *sharing_runs, sharing, sharing_range).finish();
        }

        self.set_dense();
    }

    fn ranges_in(&self, query_range: BufferSubresourceRange) -> SharingRunIter<'_> {
        if !self.uses_sharing_runs() {
            let sharing = SharingMode::decode(self.uniform.load(Ordering::Acquire));

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

    fn set_promoting(&self) {
        self.sharing_runs_state
            .store(RunTrackingState::Promoting as _, Ordering::Release);
    }

    fn set_dense(&self) {
        self.sharing_runs_state
            .store(RunTrackingState::Dense as _, Ordering::Release);
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

        let sharing_runs = self.sharing_runs.lock();

        #[cfg(not(feature = "parking_lot"))]
        let mut sharing_runs = sharing_runs.expect("poisoned buffer sharing lock");

        #[cfg(feature = "parking_lot")]
        let mut sharing_runs = sharing_runs;

        if self.is_sharing_runs_active() {
            RunMapIter::new(sharing_runs, sharing, sharing_range).finish();

            return;
        }

        self.set_promoting();
        let current = SharingMode::decode(self.uniform.load(Ordering::Acquire));
        *sharing_runs = RunMap::new(size, current);
        RunMapIter::new(sharing_runs, sharing, sharing_range).finish();
        self.set_dense();
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

        loop {
            if self.uses_sharing_runs() {
                let sharing_runs = self.sharing_runs.lock();

                #[cfg(not(feature = "parking_lot"))]
                let mut sharing_runs = sharing_runs.expect("poisoned buffer sharing lock");

                #[cfg(feature = "parking_lot")]
                let sharing_runs = sharing_runs;

                RunMapIter::new(sharing_runs, sharing, sharing_range).finish();

                return;
            }

            let current = self.uniform.load(Ordering::Acquire);
            if self
                .uniform
                .compare_exchange(
                    current,
                    encoded_sharing,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                if self.is_promoting() {
                    let sharing_runs = self.sharing_runs.lock();

                    #[cfg(not(feature = "parking_lot"))]
                    let mut sharing_runs = sharing_runs.expect("poisoned buffer sharing lock");

                    #[cfg(feature = "parking_lot")]
                    let sharing_runs = sharing_runs;

                    RunMapIter::new(sharing_runs, sharing, sharing_range).finish();
                }

                return;
            }
        }
    }

    fn sharing_runs_state(&self) -> RunTrackingState {
        match self.sharing_runs_state.load(Ordering::Acquire) {
            0 => RunTrackingState::Uniform,
            1 => RunTrackingState::Promoting,
            2 => RunTrackingState::Dense,
            _ => unreachable!("invalid buffer sharing_runs_state"),
        }
    }

    fn uses_sharing_runs(&self) -> bool {
        self.sharing_runs_state() != RunTrackingState::Uniform
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
}

struct RunMapCursor {
    run_idx: usize,
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
        }
    }

    fn next<V>(&mut self, map: &mut RunMap<V>, new_value: V) -> Option<(V, BufferSubresourceRange)>
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
        let old_end = map
            .runs
            .get(self.run_idx + 1)
            .map(|(_, start)| *start)
            .unwrap_or(map.size);
        let mut remaining_range = self.remaining_range;

        remaining_range.end = remaining_range.end.min(old_end);
        self.remaining_range.start = remaining_range.end;

        if old_value == new_value {
            self.run_idx += 1;
        } else if old_start < remaining_range.start {
            if let Some((_, start)) = map
                .runs
                .get_mut(self.run_idx + 1)
                .filter(|(value, _)| *value == new_value && old_end == remaining_range.end)
            {
                *start = remaining_range.start;
                self.run_idx += 1;
            } else {
                self.run_idx += 1;
                map.runs
                    .insert(self.run_idx, (new_value, remaining_range.start));

                if old_end > remaining_range.end {
                    map.runs
                        .insert(self.run_idx + 1, (old_value, remaining_range.end));
                }

                self.run_idx += 1;
            }
        } else if self.run_idx > 0 {
            if map
                .runs
                .get(self.run_idx - 1)
                .filter(|(value, _)| *value == new_value)
                .is_some()
            {
                if old_end == remaining_range.end {
                    map.runs.remove(self.run_idx);

                    if map
                        .runs
                        .get(self.run_idx)
                        .filter(|(value, _)| *value == new_value)
                        .is_some()
                    {
                        map.runs.remove(self.run_idx);
                        self.run_idx -= 1;
                    }
                } else {
                    debug_assert!(map.runs.get(self.run_idx).is_some());

                    let (_, start) = unsafe { map.runs.get_unchecked_mut(self.run_idx) };
                    *start = remaining_range.end;
                }
            } else if old_end == remaining_range.end {
                debug_assert!(map.runs.get(self.run_idx).is_some());

                let (value, _) = unsafe { map.runs.get_unchecked_mut(self.run_idx) };
                *value = new_value;

                if map
                    .runs
                    .get(self.run_idx + 1)
                    .filter(|(value, _)| *value == new_value)
                    .is_some()
                {
                    map.runs.remove(self.run_idx + 1);
                } else {
                    self.run_idx += 1;
                }
            } else {
                if let Some((_, start)) = map.runs.get_mut(self.run_idx) {
                    *start = remaining_range.end;
                }

                map.runs
                    .insert(self.run_idx, (new_value, remaining_range.start));
                self.run_idx += 2;
            }
        } else if let Some((_, start)) = map
            .runs
            .get_mut(1)
            .filter(|(value, _)| *value == new_value && old_end == remaining_range.end)
        {
            *start = 0;
            map.runs.remove(0);
        } else if old_end > remaining_range.end {
            map.runs.insert(0, (new_value, 0));

            debug_assert!(map.runs.get(1).is_some());

            let (_, start) = unsafe { map.runs.get_unchecked_mut(1) };
            *start = remaining_range.end;
        } else {
            debug_assert!(!map.runs.is_empty());

            let (value, _) = unsafe { map.runs.get_unchecked_mut(0) };
            *value = new_value;

            if map
                .runs
                .get(1)
                .filter(|(value, _)| *value == new_value)
                .is_some()
            {
                map.runs.remove(1);
            } else {
                self.run_idx += 1;
            }
        }

        Some((old_value, remaining_range))
    }
}

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
    fn new(map: M, new_value: V, remaining_range: BufferSubresourceRange) -> Self {
        let cursor = RunMapCursor::new(&map, remaining_range);

        Self {
            cursor,
            map,
            new_value,
        }
    }

    fn finish(self) {}
}

impl<M, V> Iterator for RunMapIter<M, V>
where
    M: DerefMut<Target = RunMap<V>>,
    V: Copy + PartialEq + Debug,
{
    type Item = (V, BufferSubresourceRange);

    fn next(&mut self) -> Option<Self::Item> {
        self.cursor.next(&mut self.map, self.new_value)
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

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RunTrackingState {
    Uniform = 0,
    Promoting = 1,
    Dense = 2,
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
                &[(AccessType::HostRead, 0), (AccessType::Nothing, 15)],
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
    #[should_panic(expected = "Alignment must be a power of two")]
    pub fn buffer_info_builder_alignment_0() {
        Builder::default().size(0).alignment(0).build();
    }

    #[test]
    #[should_panic(expected = "Alignment must be a power of two")]
    pub fn buffer_info_builder_alignment_42() {
        Builder::default().size(0).alignment(42).build();
    }

    #[test]
    pub fn buffer_info_builder_alignment_256() {
        let mut info = Info::device_mem(42, vk::BufferUsageFlags::empty());
        info.alignment = 256;

        let builder = Builder::default().size(42).alignment(256).build();

        assert_eq!(info, builder);
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
