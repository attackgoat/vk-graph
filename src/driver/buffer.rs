//! Buffer resource types

use {
    super::{DriverError, device::Device},
    ash::vk,
    derive_builder::{Builder, UninitializedFieldError},
    gpu_allocator::{
        MemoryLocation,
        vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
    },
    log::trace,
    log::warn,
    std::{
        fmt::{Debug, Formatter},
        mem::ManuallyDrop,
        ops::{Deref, DerefMut, Range},
        sync::Arc,
        thread::panicking,
    },
    vk_sync::AccessType,
};

#[cfg(feature = "parking_lot")]
use parking_lot::Mutex;

#[cfg(not(feature = "parking_lot"))]
use std::sync::Mutex;

/// Smart pointer handle to a [buffer] object.
///
/// Also contains information about the object.
///
/// ## `Deref` behavior
///
/// `Buffer` automatically dereferences to [`vk::Buffer`] (via the [`Deref`] trait), so you
/// can call `vk::Buffer`'s methods on a value of type `Buffer`. To avoid name clashes with
/// `vk::Buffer`'s methods, the methods of `Buffer` itself are associated functions, called using
/// [fully qualified syntax]:
///
/// ```no_run
/// # use std::sync::Arc;
/// # use ash::vk;
/// # use vk_graph::driver::{AccessType, DriverError};
/// # use vk_graph::driver::device::{Device, DeviceInfo};
/// # use vk_graph::driver::buffer::{Buffer, BufferInfo};
/// # fn main() -> Result<(), DriverError> {
/// # let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
/// # let info = BufferInfo::device_mem(8, vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
/// # let my_buf = Buffer::create(&device, info)?;
/// let addr = Buffer::device_address(&my_buf);
/// # Ok(()) }
/// ```
///
/// [buffer]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkBuffer.html
/// [deref]: core::ops::Deref
/// [fully qualified syntax]: https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#fully-qualified-syntax-for-disambiguation-calling-methods-with-the-same-name
pub struct Buffer {
    accesses: Mutex<BufferAccess>,
    allocation: ManuallyDrop<Allocation>,
    buffer: vk::Buffer,
    device: Arc<Device>,

    /// Information used to create this object.
    pub info: BufferInfo,

    /// A name for debugging purposes.
    pub name: Option<String>,
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
    /// # let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
    /// const SIZE: vk::DeviceSize = 1024;
    /// let info = BufferInfo::host_mem(SIZE, vk::BufferUsageFlags::UNIFORM_BUFFER);
    /// let buf = Buffer::create(&device, info)?;
    ///
    /// assert_ne!(*buf, vk::Buffer::null());
    /// assert_eq!(buf.info.size, SIZE);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn create(device: &Arc<Device>, info: impl Into<BufferInfo>) -> Result<Self, DriverError> {
        let info = info.into();

        trace!("create: {:?}", info);

        debug_assert_ne!(info.size, 0, "Size must be non-zero");

        let device = Arc::clone(device);
        let buffer_info = vk::BufferCreateInfo::default()
            .size(info.size)
            .usage(info.usage)
            .sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(&device.physical_device.queue_family_indices);
        let buffer = unsafe {
            device.create_buffer(&buffer_info, None).map_err(|err| {
                warn!("unable to create buffer: {err}");

                DriverError::Unsupported
            })?
        };
        let mut requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        requirements.alignment = requirements.alignment.max(info.alignment);

        let memory_location = if info.mappable {
            MemoryLocation::CpuToGpu
        } else {
            MemoryLocation::GpuOnly
        };
        let allocation = {
            profiling::scope!("allocate");

            #[cfg_attr(not(feature = "parking_lot"), allow(unused_mut))]
            let mut allocator = device.allocator.lock();

            #[cfg(not(feature = "parking_lot"))]
            let mut allocator = allocator.unwrap();

            allocator
                .allocate(&AllocationCreateDesc {
                    name: "buffer",
                    requirements,
                    location: memory_location,
                    linear: true, // Buffers are always linear
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|err| {
                    warn!("unable to allocate buffer memory: {err}");

                    unsafe {
                        device.destroy_buffer(buffer, None);
                    }

                    DriverError::from_alloc_err(err)
                })
                .and_then(|allocation| {
                    if let Err(err) = unsafe {
                        device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                    } {
                        warn!("unable to bind buffer memory: {err}");

                        if let Err(err) = allocator.free(allocation) {
                            warn!("unable to free buffer allocation: {err}")
                        }

                        unsafe {
                            device.destroy_buffer(buffer, None);
                        }

                        Err(DriverError::OutOfMemory)
                    } else {
                        Ok(allocation)
                    }
                })
        }?;

        debug_assert_ne!(buffer, vk::Buffer::null());

        Ok(Self {
            accesses: Mutex::new(BufferAccess::new(info.size)),
            allocation: ManuallyDrop::new(allocation),
            buffer,
            device,
            info,
            name: None,
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
    /// # let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
    /// const DATA: [u8; 4] = [0xfe, 0xed, 0xbe, 0xef];
    /// let buf = Buffer::create_from_slice(&device, vk::BufferUsageFlags::UNIFORM_BUFFER, &DATA)?;
    ///
    /// assert_ne!(*buf, vk::Buffer::null());
    /// assert_eq!(buf.info.size, 4);
    /// assert_eq!(Buffer::mapped_slice(&buf), &DATA);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn create_from_slice(
        device: &Arc<Device>,
        usage: vk::BufferUsageFlags,
        slice: impl AsRef<[u8]>,
    ) -> Result<Self, DriverError> {
        let slice = slice.as_ref();
        let info = BufferInfo::host_mem(slice.len() as _, usage);
        let mut buffer = Self::create(device, info)?;

        Self::copy_from_slice(&mut buffer, 0, slice);

        Ok(buffer)
    }

    /// Keeps track of some `next_access` which affects this object.
    ///
    /// Returns the previous access for which a pipeline barrier should be used to prevent data
    /// corruption.
    ///
    /// # Note
    ///
    /// Used to maintain object state when passing a _vk-graph_-created `vk::Buffer` handle to
    /// external code such as [_Ash_] or [_Erupt_] bindings.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ash::vk;
    /// # use vk_graph::driver::{AccessType, DriverError};
    /// # use vk_graph::driver::device::{Device, DeviceInfo};
    /// # use vk_graph::driver::buffer::{Buffer, BufferInfo, BufferSubresourceRange};
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
    /// # const SIZE: vk::DeviceSize = 1024;
    /// # let info = BufferInfo::device_mem(SIZE, vk::BufferUsageFlags::STORAGE_BUFFER);
    /// # let my_buf = Buffer::create(&device, info)?;
    /// // Initially we want to "write"
    /// let access = AccessType::ComputeShaderWrite;
    /// let access_range = BufferSubresourceRange { start: 0, end: SIZE };
    /// let mut accesses = Buffer::access(&my_buf, access, access_range);
    ///
    /// assert_eq!(accesses.next(), Some((AccessType::Nothing, access_range)));
    /// assert!(accesses.next().is_none());
    ///
    /// // External code may now "write"; no barrier required in this case
    ///
    /// // Subsequently we want to "read"
    /// let access = AccessType::ComputeShaderReadOther;
    /// let mut accesses = Buffer::access(&my_buf, access, access_range);
    ///
    /// assert_eq!(accesses.next(), Some((AccessType::ComputeShaderWrite, access_range)));
    /// assert!(accesses.next().is_none());
    ///
    /// // A barrier on "write" before "read" is required! A render graph will do this
    /// // automatically when resovled, but manual access like this requires manual barriers
    /// # Ok(()) }
    /// ```
    ///
    /// [_Ash_]: https://crates.io/crates/ash
    /// [_Erupt_]: https://crates.io/crates/erupt
    #[profiling::function]
    pub fn access(
        this: &Self,
        access: AccessType,
        access_range: impl Into<BufferSubresourceRange>,
    ) -> impl Iterator<Item = (AccessType, BufferSubresourceRange)> + '_ {
        let mut access_range: BufferSubresourceRange = access_range.into();

        if access_range.end == vk::WHOLE_SIZE {
            access_range.end = this.info.size;
        }

        let accesses = this.accesses.lock();

        #[cfg(not(feature = "parking_lot"))]
        let accesses = accesses.unwrap();

        BufferAccessIter::new(accesses, access, access_range)
    }

    /// Updates a mappable buffer starting at `offset` with the data in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if the buffer was not created with the `mappable` flag set to `true`.
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
    /// # let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
    /// # let info = BufferInfo::host_mem(4, vk::BufferUsageFlags::empty());
    /// # let mut my_buf = Buffer::create(&device, info)?;
    /// const DATA: [u8; 4] = [0xde, 0xad, 0xc0, 0xde];
    /// Buffer::copy_from_slice(&mut my_buf, 0, &DATA);
    ///
    /// assert_eq!(Buffer::mapped_slice(&my_buf), &DATA);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn copy_from_slice(this: &mut Self, offset: vk::DeviceSize, slice: impl AsRef<[u8]>) {
        let slice = slice.as_ref();
        Self::mapped_slice_mut(this)[offset as _..offset as usize + slice.len()]
            .copy_from_slice(slice);
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
    /// # let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
    /// # let info = BufferInfo::host_mem(4, vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
    /// # let my_buf = Buffer::create(&device, info)?;
    /// let addr = Buffer::device_address(&my_buf);
    ///
    /// assert_ne!(addr, 0);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn device_address(this: &Self) -> vk::DeviceAddress {
        debug_assert!(
            this.info
                .usage
                .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        );

        unsafe {
            this.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(this.buffer),
            )
        }
    }

    /// Returns a mapped slice.
    ///
    /// # Panics
    ///
    /// Panics if the buffer was not created with the `mappable` flag set to `true`.
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
    /// # let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
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
    pub fn mapped_slice(this: &Self) -> &[u8] {
        debug_assert!(
            this.info.mappable,
            "Buffer is not mappable - create using mappable flag"
        );

        &this.allocation.mapped_slice().unwrap()[0..this.info.size as usize]
    }

    /// Returns a mapped mutable slice.
    ///
    /// # Panics
    ///
    /// Panics if the buffer was not created with the `mappable` flag set to `true`.
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
    /// # let device = Arc::new(Device::create_headless(DeviceInfo::default())?);
    /// # const DATA: [u8; 4] = [0; 4];
    /// # let mut my_buf = Buffer::create_from_slice(&device, vk::BufferUsageFlags::empty(), &DATA)?;
    /// let mut data = Buffer::mapped_slice_mut(&mut my_buf);
    /// data.copy_from_slice(&42f32.to_be_bytes());
    ///
    /// assert_eq!(data.len(), 4);
    /// assert_eq!(data[0], 0x42);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn mapped_slice_mut(this: &mut Self) -> &mut [u8] {
        debug_assert!(
            this.info.mappable,
            "Buffer is not mappable - create using mappable flag"
        );

        &mut this.allocation.mapped_slice_mut().unwrap()[0..this.info.size as usize]
    }
}

impl Debug for Buffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "{} ({:?})", name, self.buffer)
        } else {
            write!(f, "{:?}", self.buffer)
        }
    }
}

impl Deref for Buffer {
    type Target = vk::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
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

            #[cfg_attr(not(feature = "parking_lot"), allow(unused_mut))]
            let mut allocator = self.device.allocator.lock();

            #[cfg(not(feature = "parking_lot"))]
            let mut allocator = allocator.unwrap();

            allocator.free(unsafe { ManuallyDrop::take(&mut self.allocation) })
        }
        .unwrap_or_else(|err| warn!("unable to free buffer allocation: {err}"));

        unsafe {
            self.device.destroy_buffer(self.buffer, None);
        }
    }
}

#[derive(Debug)]
struct BufferAccess {
    accesses: Vec<(AccessType, vk::DeviceSize)>,
    size: vk::DeviceSize,
}

impl BufferAccess {
    fn new(size: vk::DeviceSize) -> Self {
        Self {
            accesses: vec![(AccessType::Nothing, 0)],
            size,
        }
    }
}

struct BufferAccessIter<T> {
    access: AccessType,
    access_range: BufferSubresourceRange,
    buffer: T,
    idx: usize,
}

impl<T> BufferAccessIter<T>
where
    T: DerefMut<Target = BufferAccess>,
{
    fn new(buffer: T, access: AccessType, access_range: BufferSubresourceRange) -> Self {
        debug_assert!(access_range.start < access_range.end);
        debug_assert!(access_range.end <= buffer.size);

        #[cfg(debug_assertions)]
        {
            let access_start = |(_, access_start): &(AccessType, vk::DeviceSize)| *access_start;

            assert_eq!(buffer.accesses.first().map(access_start), Some(0));
            assert!(buffer.accesses.last().map(access_start).unwrap() < buffer.size);

            // Custom is-sorted-by key to additionally check that all access starts are unique
            let (mut prev_access, mut prev_start) = buffer.accesses.first().copied().unwrap();
            for (next_access, next_start) in buffer.accesses.iter().skip(1).copied() {
                debug_assert_ne!(prev_access, next_access);
                debug_assert!(prev_start < next_start);

                prev_access = next_access;
                prev_start = next_start;
            }
        };

        // The needle will always be odd, and the probe always even, the result will always be err
        let needle = (access_range.start << 1) | 1;
        let idx = buffer
            .accesses
            .binary_search_by(|(_, probe)| (probe << 1).cmp(&needle));

        debug_assert!(idx.is_err());

        let mut idx = unsafe { idx.unwrap_err_unchecked() };

        // The first access will always be at start == 0, which is even, so idx cannot be 0
        debug_assert_ne!(idx, 0);

        idx -= 1;

        Self {
            access,
            access_range,
            buffer,
            idx,
        }
    }
}

impl<T> Iterator for BufferAccessIter<T>
where
    T: DerefMut<Target = BufferAccess>,
{
    type Item = (AccessType, BufferSubresourceRange);

    fn next(&mut self) -> Option<Self::Item> {
        debug_assert!(self.access_range.start <= self.access_range.end);
        debug_assert!(self.access_range.end <= self.buffer.size);

        if self.access_range.start == self.access_range.end {
            return None;
        }

        debug_assert!(self.buffer.accesses.get(self.idx).is_some());

        let (access, access_start) = unsafe { *self.buffer.accesses.get_unchecked(self.idx) };
        let access_end = self
            .buffer
            .accesses
            .get(self.idx + 1)
            .map(|(_, access_start)| *access_start)
            .unwrap_or(self.buffer.size);
        let mut access_range = self.access_range;

        access_range.end = access_range.end.min(access_end);
        self.access_range.start = access_range.end;

        if access == self.access {
            self.idx += 1;
        } else if access_start < access_range.start {
            if let Some((_, access_start)) = self
                .buffer
                .accesses
                .get_mut(self.idx + 1)
                .filter(|(access, _)| *access == self.access && access_end == access_range.end)
            {
                *access_start = access_range.start;
                self.idx += 1;
            } else {
                self.idx += 1;
                self.buffer
                    .accesses
                    .insert(self.idx, (self.access, access_range.start));

                if access_end > access_range.end {
                    self.buffer
                        .accesses
                        .insert(self.idx + 1, (access, access_range.end));
                }

                self.idx += 1;
            }
        } else if self.idx > 0 {
            if self
                .buffer
                .accesses
                .get(self.idx - 1)
                .filter(|(access, _)| *access == self.access)
                .is_some()
            {
                if access_end == access_range.end {
                    self.buffer.accesses.remove(self.idx);

                    if self
                        .buffer
                        .accesses
                        .get(self.idx)
                        .filter(|(access, _)| *access == self.access)
                        .is_some()
                    {
                        self.buffer.accesses.remove(self.idx);
                        self.idx -= 1;
                    }
                } else {
                    debug_assert!(self.buffer.accesses.get(self.idx).is_some());

                    let (_, access_start) =
                        unsafe { self.buffer.accesses.get_unchecked_mut(self.idx) };
                    *access_start = access_range.end;
                }
            } else if access_end == access_range.end {
                debug_assert!(self.buffer.accesses.get(self.idx).is_some());

                let (access, _) = unsafe { self.buffer.accesses.get_unchecked_mut(self.idx) };
                *access = self.access;

                if self
                    .buffer
                    .accesses
                    .get(self.idx + 1)
                    .filter(|(access, _)| *access == self.access)
                    .is_some()
                {
                    self.buffer.accesses.remove(self.idx + 1);
                } else {
                    self.idx += 1;
                }
            } else {
                if let Some((_, access_start)) = self.buffer.accesses.get_mut(self.idx) {
                    *access_start = access_range.end;
                }

                self.buffer
                    .accesses
                    .insert(self.idx, (self.access, access_range.start));
                self.idx += 2;
            }
        } else if let Some((_, access_start)) = self
            .buffer
            .accesses
            .get_mut(1)
            .filter(|(access, _)| *access == self.access && access_end == access_range.end)
        {
            *access_start = 0;
            self.buffer.accesses.remove(0);
        } else if access_end > access_range.end {
            self.buffer.accesses.insert(0, (self.access, 0));

            debug_assert!(self.buffer.accesses.get(1).is_some());

            let (_, access_start) = unsafe { self.buffer.accesses.get_unchecked_mut(1) };
            *access_start = access_range.end;
        } else {
            debug_assert!(!self.buffer.accesses.is_empty());

            let (access, _) = unsafe { self.buffer.accesses.get_unchecked_mut(0) };
            *access = self.access;

            if self
                .buffer
                .accesses
                .get(1)
                .filter(|(access, _)| *access == self.access)
                .is_some()
            {
                self.buffer.accesses.remove(1);
            } else {
                self.idx += 1;
            }
        }

        Some((access, access_range))
    }
}

/// Information used to create a [`Buffer`] instance.
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build", error = "BufferInfoBuilderError"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
#[non_exhaustive]
pub struct BufferInfo {
    /// Byte alignment of the base device address of the buffer.
    ///
    /// Must be a power of two.
    #[builder(default = "1")]
    pub alignment: vk::DeviceSize,

    /// Specifies a buffer whose memory is host visible and may be mapped.
    #[builder(default)]
    pub mappable: bool,

    /// Size in bytes of the buffer to be created.
    pub size: vk::DeviceSize,

    /// A bitmask of specifying allowed usages of the buffer.
    #[builder(default)]
    pub usage: vk::BufferUsageFlags,
}

impl BufferInfo {
    /// Specifies a non-mappable buffer with the given `size` and `usage` values.
    ///
    /// Device-local memory (located on the GPU) is used.
    #[inline(always)]
    pub const fn device_mem(size: vk::DeviceSize, usage: vk::BufferUsageFlags) -> BufferInfo {
        BufferInfo {
            alignment: 1,
            mappable: false,
            size,
            usage,
        }
    }

    /// Specifies a mappable buffer with the given `size` and `usage` values.
    ///
    /// Host-local memory (located in CPU-accesible RAM) is used.
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
            mappable: true,
            size,
            usage,
        }
    }

    /// Specifies a non-mappable buffer with the given `size` and `usage` values.
    #[allow(clippy::new_ret_no_self)]
    #[deprecated = "Use BufferInfo::device_mem()"]
    #[doc(hidden)]
    pub fn new(size: vk::DeviceSize, usage: vk::BufferUsageFlags) -> BufferInfoBuilder {
        Self::device_mem(size, usage).to_builder()
    }

    /// Specifies a mappable buffer with the given `size` and `usage` values.
    ///
    /// # Note
    ///
    /// For convenience the given usage value will be bitwise OR'd with
    /// `TRANSFER_DST | TRANSFER_SRC`.
    #[deprecated = "Use BufferInfo::host_mem()"]
    #[doc(hidden)]
    pub fn new_mappable(size: vk::DeviceSize, usage: vk::BufferUsageFlags) -> BufferInfoBuilder {
        Self::host_mem(size, usage).to_builder()
    }

    /// Converts a `BufferInfo` into a `BufferInfoBuilder`.
    #[inline(always)]
    pub fn to_builder(self) -> BufferInfoBuilder {
        BufferInfoBuilder {
            alignment: Some(self.alignment),
            mappable: Some(self.mappable),
            size: Some(self.size),
            usage: Some(self.usage),
        }
    }
}

impl BufferInfoBuilder {
    /// Builds a new `BufferInfo`.
    ///    
    /// # Panics
    ///
    /// If any of the following values have not been set this function will panic:
    ///
    /// * `size`
    ///
    /// If `alignment` is not a power to two this function will panic.
    #[inline(always)]
    pub fn build(self) -> BufferInfo {
        let res = match self.fallible_build() {
            Err(BufferInfoBuilderError(err)) => panic!("{err}"),
            Ok(info) => info,
        };

        assert_eq!(
            res.alignment.count_ones(),
            1,
            "Alignment must be a power of two"
        );

        res
    }
}

impl From<BufferInfoBuilder> for BufferInfo {
    fn from(info: BufferInfoBuilder) -> Self {
        info.build()
    }
}

#[derive(Debug)]
struct BufferInfoBuilderError(UninitializedFieldError);

impl From<UninitializedFieldError> for BufferInfoBuilderError {
    fn from(err: UninitializedFieldError) -> Self {
        Self(err)
    }
}

/// Specifies a range of buffer data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BufferSubresourceRange {
    /// The start of range.
    pub start: vk::DeviceSize,

    /// The non-inclusive end of the range.
    pub end: vk::DeviceSize,
}

impl BufferSubresourceRange {
    #[cfg(test)]
    pub(crate) fn intersects(self, other: Self) -> bool {
        self.start < other.end && self.end > other.start
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

impl From<Option<Range<vk::DeviceSize>>> for BufferSubresourceRange {
    fn from(range: Option<Range<vk::DeviceSize>>) -> Self {
        range.unwrap_or(0..vk::WHOLE_SIZE).into()
    }
}

impl From<BufferSubresourceRange> for Range<vk::DeviceSize> {
    fn from(subresource: BufferSubresourceRange) -> Self {
        subresource.start..subresource.end
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        rand::{Rng, SeedableRng, rngs::SmallRng},
    };

    type Info = BufferInfo;
    type Builder = BufferInfoBuilder;

    const FUZZ_COUNT: usize = 100_000;

    #[test]
    pub fn buffer_access() {
        let mut buffer = BufferAccess::new(100);

        {
            let mut accesses = BufferAccessIter::new(
                &mut buffer,
                AccessType::TransferWrite,
                buffer_subresource_range(0..10),
            );

            assert_eq!(accesses.buffer.accesses, vec![(AccessType::Nothing, 0)]);
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::Nothing, buffer_subresource_range(0..10))
            );
            assert_eq!(
                accesses.buffer.accesses,
                vec![(AccessType::TransferWrite, 0), (AccessType::Nothing, 10)]
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = BufferAccessIter::new(
                &mut buffer,
                AccessType::TransferRead,
                buffer_subresource_range(5..15),
            );

            assert_eq!(
                accesses.buffer.accesses,
                vec![(AccessType::TransferWrite, 0), (AccessType::Nothing, 10)]
            );
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::TransferWrite, buffer_subresource_range(5..10))
            );
            assert_eq!(
                accesses.buffer.accesses,
                vec![
                    (AccessType::TransferWrite, 0),
                    (AccessType::TransferRead, 5),
                    (AccessType::Nothing, 10)
                ]
            );
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::Nothing, buffer_subresource_range(10..15))
            );
            assert_eq!(
                accesses.buffer.accesses,
                vec![
                    (AccessType::TransferWrite, 0),
                    (AccessType::TransferRead, 5),
                    (AccessType::Nothing, 15)
                ]
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = BufferAccessIter::new(
                &mut buffer,
                AccessType::HostRead,
                buffer_subresource_range(0..100),
            );

            assert_eq!(
                accesses.buffer.accesses,
                vec![
                    (AccessType::TransferWrite, 0),
                    (AccessType::TransferRead, 5),
                    (AccessType::Nothing, 15)
                ]
            );
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::TransferWrite, buffer_subresource_range(0..5))
            );
            assert_eq!(
                accesses.buffer.accesses,
                vec![
                    (AccessType::HostRead, 0),
                    (AccessType::TransferRead, 5),
                    (AccessType::Nothing, 15)
                ]
            );
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::TransferRead, buffer_subresource_range(5..15))
            );
            assert_eq!(
                accesses.buffer.accesses,
                vec![(AccessType::HostRead, 0), (AccessType::Nothing, 15)]
            );
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::Nothing, buffer_subresource_range(15..100))
            );
            assert_eq!(accesses.buffer.accesses, vec![(AccessType::HostRead, 0),]);
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = BufferAccessIter::new(
                &mut buffer,
                AccessType::HostWrite,
                buffer_subresource_range(0..100),
            );

            assert_eq!(accesses.buffer.accesses, vec![(AccessType::HostRead, 0)]);
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::HostRead, buffer_subresource_range(0..100))
            );
            assert_eq!(accesses.buffer.accesses, vec![(AccessType::HostWrite, 0)]);
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = BufferAccessIter::new(
                &mut buffer,
                AccessType::HostWrite,
                buffer_subresource_range(0..100),
            );

            assert_eq!(accesses.buffer.accesses, vec![(AccessType::HostWrite, 0)]);
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::HostWrite, buffer_subresource_range(0..100))
            );
            assert_eq!(accesses.buffer.accesses, vec![(AccessType::HostWrite, 0)]);
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = BufferAccessIter::new(
                &mut buffer,
                AccessType::HostWrite,
                buffer_subresource_range(1..99),
            );

            assert_eq!(accesses.buffer.accesses, vec![(AccessType::HostWrite, 0)]);
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::HostWrite, buffer_subresource_range(1..99))
            );
            assert_eq!(accesses.buffer.accesses, vec![(AccessType::HostWrite, 0)]);
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = BufferAccessIter::new(
                &mut buffer,
                AccessType::HostRead,
                buffer_subresource_range(1..99),
            );

            assert_eq!(accesses.buffer.accesses, vec![(AccessType::HostWrite, 0)]);
            assert_eq!(
                accesses.next().unwrap(),
                (AccessType::HostWrite, buffer_subresource_range(1..99))
            );
            assert_eq!(
                accesses.buffer.accesses,
                vec![
                    (AccessType::HostWrite, 0),
                    (AccessType::HostRead, 1),
                    (AccessType::HostWrite, 99)
                ]
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = BufferAccessIter::new(
                &mut buffer,
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
            let mut accesses = BufferAccessIter::new(
                &mut buffer,
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
            let mut accesses = BufferAccessIter::new(
                &mut buffer,
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
            let mut accesses = BufferAccessIter::new(
                &mut buffer,
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
            let mut accesses = BufferAccessIter::new(
                &mut buffer,
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
        let mut buffer = BufferAccess::new(5);

        buffer.accesses = vec![
            (AccessType::ColorAttachmentRead, 0),
            (AccessType::AnyShaderWrite, 4),
        ];

        {
            let mut accesses = BufferAccessIter::new(
                &mut buffer,
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
            let mut accesses = BufferAccessIter::new(
                &mut buffer,
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
        let mut buffer = BufferAccess::new(buffer_size);
        let mut data = vec![AccessType::Nothing; buffer_size as usize];

        for _ in 0..FUZZ_COUNT {
            let access = ACCESS_TYPES[rng.random_range(..ACCESS_TYPES.len())];
            let access_start = rng.random_range(..buffer_size);
            let access_end = rng.random_range(access_start + 1..=buffer_size);

            // println!("{access:?} {access_start}..{access_end}");

            let accesses = BufferAccessIter::new(
                &mut buffer,
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
    pub fn buffer_info() {
        let info = Info::device_mem(0, vk::BufferUsageFlags::empty());
        let builder = info.to_builder().build();

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
    #[should_panic(expected = "Field not initialized: size")]
    pub fn buffer_info_builder_uninit_size() {
        Builder::default().build();
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
