//! Image resource types

use {
    super::{
        DriverError, SharingMode, access_type_from_u8, access_type_into_u8, device::Device,
        format_aspect_mask, pipeline_stage_access_flags,
    },
    ash::vk::{self, ImageCreateInfo},
    derive_builder::Builder,
    gpu_allocator::{
        MemoryLocation,
        vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
    },
    log::{trace, warn},
    std::{
        collections::{HashMap, hash_map::Entry},
        fmt::{Debug, Formatter},
        marker::PhantomData,
        mem::{replace, take},
        ops::{Deref, DerefMut},
        sync::atomic::{AtomicU8, AtomicU16, AtomicU64, Ordering},
        thread::panicking,
    },
    vk_sync::AccessType,
};

#[cfg(feature = "parking_lot")]
use parking_lot::{Mutex, MutexGuard};

#[cfg(not(feature = "parking_lot"))]
use std::sync::{Mutex, MutexGuard};

const fn access_type_to_layout(access: AccessType) -> Option<vk::ImageLayout> {
    match access {
        AccessType::Nothing => None,
        AccessType::ColorAttachmentRead
        | AccessType::ColorAttachmentReadWrite
        | AccessType::ColorAttachmentWrite => Some(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        AccessType::DepthStencilAttachmentRead => {
            Some(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        }
        AccessType::DepthStencilAttachmentReadWrite | AccessType::DepthStencilAttachmentWrite => {
            Some(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        }
        AccessType::DepthAttachmentWriteStencilReadOnly => {
            Some(vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL)
        }
        AccessType::StencilAttachmentWriteDepthReadOnly => {
            Some(vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL)
        }
        AccessType::TransferRead => Some(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
        AccessType::TransferWrite => Some(vk::ImageLayout::TRANSFER_DST_OPTIMAL),
        AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::FragmentShaderReadColorInputAttachment
        | AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::TessellationControlShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::TessellationEvaluationShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::GeometryShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::MeshShaderReadSampledImageOrUniformTexelBuffer
        | AccessType::TaskShaderReadSampledImageOrUniformTexelBuffer => {
            Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        }
        AccessType::FragmentShaderReadDepthStencilInputAttachment => {
            Some(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        }
        AccessType::Present => Some(vk::ImageLayout::PRESENT_SRC_KHR),
        _ => Some(vk::ImageLayout::GENERAL),
    }
}

const fn aspect_mask_at_ordinal(
    aspect_mask: vk::ImageAspectFlags,
    ordinal: u32,
) -> vk::ImageAspectFlags {
    // Common cases:
    // - COLOR with ordinal 0 -> COLOR
    // - DEPTH | STENCIL with ordinal 0 -> DEPTH
    // - DEPTH | STENCIL with ordinal 1 -> STENCIL
    let mut bits = aspect_mask.as_raw();
    let mut idx = 0;

    while bits != 0 {
        let bit = bits.trailing_zeros();
        if idx == ordinal {
            return vk::ImageAspectFlags::from_raw(1 << bit);
        }

        bits &= !(1 << bit);
        idx += 1;
    }

    vk::ImageAspectFlags::empty()
}

const fn aspect_ordinal(aspect_mask: vk::ImageAspectFlags, aspect: vk::ImageAspectFlags) -> u8 {
    let mut bits = aspect_mask.as_raw();
    let target = aspect.as_raw();
    let mut idx = 0;

    while bits != 0 {
        let bit = bits.trailing_zeros();
        if target == (1 << bit) {
            return idx;
        }

        bits &= !(1 << bit);
        idx += 1;
    }

    0
}

#[cfg(feature = "checked")]
fn assert_aspect_mask_supported(aspect_mask: vk::ImageAspectFlags) {
    use vk::ImageAspectFlags as A;

    const COLOR: A = A::COLOR;
    const DEPTH: A = A::DEPTH;
    const DEPTH_STENCIL: A = A::from_raw(A::DEPTH.as_raw() | A::STENCIL.as_raw());
    const STENCIL: A = A::STENCIL;

    assert!(matches!(
        aspect_mask,
        COLOR | DEPTH | DEPTH_STENCIL | STENCIL
    ));
}

pub(crate) fn image_subresource_range_contains(
    lhs: vk::ImageSubresourceRange,
    rhs: vk::ImageSubresourceRange,
) -> bool {
    lhs.aspect_mask.contains(rhs.aspect_mask)
        && lhs.base_array_layer <= rhs.base_array_layer
        && lhs.base_array_layer + lhs.layer_count >= rhs.base_array_layer + rhs.layer_count
        && lhs.base_mip_level <= rhs.base_mip_level
        && lhs.base_mip_level + lhs.level_count >= rhs.base_mip_level + rhs.level_count
}

pub(crate) fn image_subresource_range_intersection(
    lhs: vk::ImageSubresourceRange,
    rhs: vk::ImageSubresourceRange,
) -> Option<vk::ImageSubresourceRange> {
    if !image_subresource_range_intersects(lhs, rhs) {
        return None;
    }

    let aspect_mask = lhs.aspect_mask & rhs.aspect_mask;
    let base_array_layer = lhs.base_array_layer.max(rhs.base_array_layer);
    let end_array_layer =
        (lhs.base_array_layer + lhs.layer_count).min(rhs.base_array_layer + rhs.layer_count);
    let base_mip_level = lhs.base_mip_level.max(rhs.base_mip_level);
    let end_mip_level =
        (lhs.base_mip_level + lhs.level_count).min(rhs.base_mip_level + rhs.level_count);

    Some(vk::ImageSubresourceRange {
        aspect_mask,
        base_array_layer,
        layer_count: end_array_layer - base_array_layer,
        base_mip_level,
        level_count: end_mip_level - base_mip_level,
    })
}

pub(crate) fn image_subresource_range_intersects(
    lhs: vk::ImageSubresourceRange,
    rhs: vk::ImageSubresourceRange,
) -> bool {
    lhs.aspect_mask.intersects(rhs.aspect_mask)
        && lhs.base_array_layer < rhs.base_array_layer + rhs.layer_count
        && lhs.base_array_layer + lhs.layer_count > rhs.base_array_layer
        && lhs.base_mip_level < rhs.base_mip_level + rhs.level_count
        && lhs.base_mip_level + lhs.level_count > rhs.base_mip_level
}

#[derive(Debug)]
enum Access {
    Dense(DenseAccess),
    DualAspect(DualAspectAccess),
    Uniform(UniformAccess),
}

impl Access {
    fn new(info: ImageInfo, access: AccessType) -> Self {
        let aspect_count = format_aspect_mask(info.format).as_raw().count_ones() as u8;

        if aspect_count == 1 && info.array_layer_count == 1 && info.mip_level_count == 1 {
            Self::Uniform(UniformAccess::new(access))
        } else if aspect_count == 2 && info.array_layer_count == 1 && info.mip_level_count == 1 {
            Self::DualAspect(DualAspectAccess::new(access))
        } else {
            Self::Dense(DenseAccess::new(access))
        }
    }

    fn swap<'a>(
        &'a self,
        dense: &'a Mutex<Option<DenseMap<AccessType>>>,
        info: ImageInfo,
        next_access: AccessType,
        access_range: vk::ImageSubresourceRange,
    ) -> AccessIter<'a> {
        match self {
            Self::Uniform(uniform) => {
                AccessIter::Uniform(Some(uniform.swap(next_access, access_range)))
            }
            Self::DualAspect(dual) => AccessIter::DualAspect(DualAspectAccessIter::new(
                dual,
                info,
                next_access,
                access_range,
            )),
            Self::Dense(access) => {
                if !access.uses_dense() && info.is_full_subresource_range(access_range) {
                    return AccessIter::Uniform(Some(access.swap_range(next_access, access_range)));
                }

                let mut dense = dense.lock();

                #[cfg(not(feature = "parking_lot"))]
                let mut dense = dense.expect("poisoned image dense lock");

                access.ensure_dense(&mut dense, info);

                AccessIter::DenseMap(DenseMapIter::new(
                    DenseAccessMapGuard { access, dense },
                    next_access,
                    access_range,
                ))
            }
        }
    }
}

enum AccessIter<'a> {
    DenseMap(DenseMapIter<'a, DenseAccessMapGuard<'a>, AccessType>),
    DualAspect(DualAspectAccessIter<'a>),
    Uniform(Option<(AccessType, vk::ImageSubresourceRange)>),
}

impl Drop for AccessIter<'_> {
    fn drop(&mut self) {
        while self.next().is_some() {}
    }
}

impl Iterator for AccessIter<'_> {
    type Item = (AccessType, vk::ImageSubresourceRange);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::DenseMap(iter) => iter.next(),
            Self::DualAspect(iter) => iter.next(),
            Self::Uniform(item) => item.take(),
        }
    }
}

#[derive(Debug)]
struct DenseAccess(AtomicU16);

impl DenseAccess {
    const ACCESS_MASK: u16 = 0x00_FF;
    const STATE_MASK: u16 = 0xFF_00;
    const STATE_SHIFT: u16 = 8;

    fn new(access: AccessType) -> Self {
        Self(AtomicU16::new(
            (DenseAccessState::Uniform as u16) << Self::STATE_SHIFT
                | access_type_into_u8(access) as u16,
        ))
    }

    fn ensure_dense(&self, dense: &mut Option<DenseMap<AccessType>>, info: ImageInfo) {
        if self.is_dense_active() {
            debug_assert!(dense.is_some());
            return;
        }

        self.set_promoting();
        let current = self.load();
        *dense = Some(DenseMap::new(info, current));
        self.set_dense();
    }

    fn is_dense_active(&self) -> bool {
        self.state() == DenseAccessState::Dense
    }

    fn load(&self) -> AccessType {
        access_type_from_u8((self.0.load(Ordering::Acquire) & Self::ACCESS_MASK) as u8)
    }

    fn set_dense(&self) {
        let current = self.0.load(Ordering::Acquire);
        self.0.store(
            (current & !Self::STATE_MASK) | (DenseAccessState::Dense as u16) << Self::STATE_SHIFT,
            Ordering::Release,
        );
    }

    fn set_promoting(&self) {
        let current = self.0.load(Ordering::Acquire);
        self.0.store(
            (current & !Self::STATE_MASK)
                | (DenseAccessState::Promoting as u16) << Self::STATE_SHIFT,
            Ordering::Release,
        );
    }

    fn set_uniform(&self, next_access: AccessType) {
        self.0.store(
            (DenseAccessState::Uniform as u16) << Self::STATE_SHIFT
                | access_type_into_u8(next_access) as u16,
            Ordering::Release,
        );
    }

    fn state(&self) -> DenseAccessState {
        match (self.0.load(Ordering::Acquire) >> Self::STATE_SHIFT) as u8 {
            0 => DenseAccessState::Uniform,
            1 => DenseAccessState::Promoting,
            2 => DenseAccessState::Dense,
            _ => unreachable!("invalid image dense access state"),
        }
    }

    fn swap_range(
        &self,
        next_access: AccessType,
        access_range: vk::ImageSubresourceRange,
    ) -> (AccessType, vk::ImageSubresourceRange) {
        let packed = (DenseAccessState::Uniform as u16) << Self::STATE_SHIFT
            | access_type_into_u8(next_access) as u16;
        let prev = self.0.swap(packed, Ordering::AcqRel);

        (access_type_from_u8(prev as u8), access_range)
    }

    fn uses_dense(&self) -> bool {
        self.state() != DenseAccessState::Uniform
    }
}

struct DenseAccessMapGuard<'a> {
    access: &'a DenseAccess,
    dense: MutexGuard<'a, Option<DenseMap<AccessType>>>,
}

impl DenseAccessMapGuard<'_> {
    fn try_demote_to_uniform(&mut self) {
        let DenseAccessState::Dense = self.access.state() else {
            return;
        };

        let dense_map = self.dense.as_ref().expect("missing dense access state");
        let Some(access) = dense_map.uniform_value() else {
            return;
        };

        *self.dense = None;
        self.access.set_uniform(access);
    }
}

impl Deref for DenseAccessMapGuard<'_> {
    type Target = DenseMap<AccessType>;

    fn deref(&self) -> &Self::Target {
        self.dense.as_ref().expect("missing dense access state")
    }
}

impl DerefMut for DenseAccessMapGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.dense.as_mut().expect("missing dense access state")
    }
}

impl Drop for DenseAccessMapGuard<'_> {
    fn drop(&mut self) {
        self.try_demote_to_uniform();
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DenseAccessState {
    Uniform = 0,
    Promoting = 1,
    Dense = 2,
}

#[derive(Debug)]
pub(crate) struct DenseMap<V> {
    #[cfg(feature = "checked")]
    array_layer_count: u32,

    aspect_count: u8,
    mip_level_count: u32,
    values: Box<[V]>,
}

impl<V> DenseMap<V> {
    fn base_aspect_ordinal(&self, base_aspect_bit: u8) -> u8 {
        let stencil_bit = vk::ImageAspectFlags::STENCIL.as_raw().trailing_zeros() as u8;

        // DenseMap stores depth/stencil aspects as compact ordinals: depth = 0, stencil = 1
        (self.aspect_count == 2 && base_aspect_bit == stencil_bit) as u8
    }

    fn idx(&self, aspect: u8, array_layer: u32, mip_level: u32) -> usize {
        let idx = (array_layer * self.aspect_count as u32 * self.mip_level_count
            + mip_level * self.aspect_count as u32
            + aspect as u32) as _;

        #[cfg(feature = "checked")]
        assert!(
            idx < self.values.len(),
            "idx={idx}, aspect={aspect}, layer={array_layer}, mip={mip_level}, aspect_count={}, mip_level_count={}, array_layer_count={}, len={}",
            self.aspect_count,
            self.mip_level_count,
            self.array_layer_count,
            self.values.len(),
        );

        idx
    }
}

impl<V: Copy> DenseMap<V> {
    pub(crate) fn new(info: ImageInfo, value: V) -> Self {
        let aspect_mask = format_aspect_mask(info.format);

        #[cfg(feature = "checked")]
        assert_aspect_mask_supported(aspect_mask);

        let aspect_count = aspect_mask.as_raw().count_ones() as u8;
        let array_layer_count = info.array_layer_count;
        let mip_level_count = info.mip_level_count;

        Self {
            aspect_count,
            mip_level_count,
            values: vec![value; (aspect_count as u32 * array_layer_count * mip_level_count) as _]
                .into_boxed_slice(),

            #[cfg(feature = "checked")]
            array_layer_count,
        }
    }

    fn subresource(&self, aspect: u8, array_layer: u32, mip_level: u32) -> V {
        self.values[self.idx(aspect, array_layer, mip_level)]
    }
}

impl<V: Copy + PartialEq> DenseMap<V> {
    pub(crate) fn swap(
        &mut self,
        value: V,
        range: vk::ImageSubresourceRange,
    ) -> DenseMapIter<'_, &mut Self, V> {
        DenseMapIter::new(self, value, range)
    }

    fn uniform_value(&self) -> Option<V> {
        let mut iter = self.values.iter().copied();
        let first = iter.next()?;

        iter.all(|value| value == first).then_some(first)
    }
}

struct DenseMapCursor {
    range: DenseMapRange,
    array_layer: u32,
    aspect: u8,
    mip_level: u32,
}

impl DenseMapCursor {
    fn new<V>(map: &DenseMap<V>, range: vk::ImageSubresourceRange) -> Self {
        #[cfg(feature = "checked")]
        assert_aspect_mask_supported(range.aspect_mask);

        #[cfg(feature = "checked")]
        assert!(range.base_array_layer < map.array_layer_count);

        debug_assert!(range.base_mip_level < map.mip_level_count);
        debug_assert_ne!(range.layer_count, 0);
        debug_assert_ne!(range.level_count, 0);

        let aspect_count = range.aspect_mask.as_raw().count_ones() as _;

        debug_assert!(aspect_count <= map.aspect_count);

        let base_aspect_bit = range.aspect_mask.as_raw().trailing_zeros() as _;

        Self {
            array_layer: 0,
            aspect: 0,
            mip_level: 0,
            range: DenseMapRange {
                aspect_count,
                base_array_layer: range.base_array_layer,
                base_aspect_bit,
                base_mip_level: range.base_mip_level,
                layer_count: range.layer_count,
                level_count: range.level_count,
            },
        }
    }

    fn next<V>(&mut self, map: &mut DenseMap<V>, value: V) -> Option<(V, vk::ImageSubresourceRange)>
    where
        V: Copy + PartialEq,
    {
        if self.aspect == self.range.aspect_count {
            return None;
        }

        let mut range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::from_raw(
                (1 << (self.range.base_aspect_bit + self.aspect)) as _,
            ),
            base_array_layer: self.range.base_array_layer + self.array_layer,
            base_mip_level: self.range.base_mip_level + self.mip_level,
            layer_count: 1,
            level_count: 1,
        };

        let base_aspect_ordinal = map.base_aspect_ordinal(self.range.base_aspect_bit);
        let prev_value = replace(
            {
                let idx = map.idx(
                    base_aspect_ordinal + self.aspect,
                    range.base_array_layer,
                    range.base_mip_level,
                );

                unsafe { map.values.get_unchecked_mut(idx) }
            },
            value,
        );

        loop {
            self.mip_level += 1;
            self.mip_level %= self.range.level_count;
            if self.mip_level == 0 {
                break;
            }

            let idx = map.idx(
                base_aspect_ordinal + self.aspect,
                self.range.base_array_layer + self.array_layer,
                self.range.base_mip_level + self.mip_level,
            );
            let next_value = unsafe { map.values.get_unchecked_mut(idx) };
            if *next_value != prev_value {
                return Some((prev_value, range));
            }

            *next_value = value;
            range.level_count += 1;
        }

        loop {
            self.array_layer += 1;
            self.array_layer %= self.range.layer_count;
            if self.array_layer == 0 {
                break;
            }

            if range.base_mip_level != self.range.base_mip_level {
                return Some((prev_value, range));
            }

            let array_layer = self.range.base_array_layer + self.array_layer;
            let end_mip_level = self.range.base_mip_level + self.range.level_count;

            for mip_level in self.range.base_mip_level..end_mip_level {
                let idx = map.idx(base_aspect_ordinal + self.aspect, array_layer, mip_level);
                let next_value = unsafe { *map.values.get_unchecked(idx) };
                if next_value != prev_value {
                    return Some((prev_value, range));
                }
            }

            for mip_level in self.range.base_mip_level..end_mip_level {
                let idx = map.idx(base_aspect_ordinal + self.aspect, array_layer, mip_level);
                let next_value = unsafe { map.values.get_unchecked_mut(idx) };
                *next_value = value;
            }

            range.layer_count += 1;
        }

        loop {
            self.aspect += 1;
            if self.aspect == self.range.aspect_count {
                return Some((prev_value, range));
            }

            let end_array_layer = self.range.base_array_layer + self.range.layer_count;
            let end_mip_level = self.range.base_mip_level + self.range.level_count;

            for array_layer in self.range.base_array_layer..end_array_layer {
                for mip_level in self.range.base_mip_level..end_mip_level {
                    let idx = map.idx(base_aspect_ordinal + self.aspect, array_layer, mip_level);
                    let next_value = unsafe { *map.values.get_unchecked(idx) };
                    if next_value != prev_value {
                        return Some((prev_value, range));
                    }
                }
            }

            for array_layer in self.range.base_array_layer..end_array_layer {
                for mip_level in self.range.base_mip_level..end_mip_level {
                    let idx = map.idx(base_aspect_ordinal + self.aspect, array_layer, mip_level);
                    let next_value = unsafe { map.values.get_unchecked_mut(idx) };
                    *next_value = value;
                }
            }

            range.aspect_mask = vk::ImageAspectFlags::from_raw(
                range.aspect_mask.as_raw() | (1 << (self.range.base_aspect_bit + self.aspect)),
            );
        }
    }
}

pub(crate) struct DenseMapIter<'a, M, V>
where
    M: DerefMut<Target = DenseMap<V>>,
    V: Copy + PartialEq,
{
    __: PhantomData<&'a mut DenseMap<V>>,
    cursor: DenseMapCursor,
    map: M,
    value: V,
}

impl<M, V> Drop for DenseMapIter<'_, M, V>
where
    M: DerefMut<Target = DenseMap<V>>,
    V: Copy + PartialEq,
{
    fn drop(&mut self) {
        while self.next().is_some() {}
    }
}

impl<'a, M, V: Copy + PartialEq> DenseMapIter<'a, M, V>
where
    M: DerefMut<Target = DenseMap<V>>,
{
    fn new(map: M, value: V, range: vk::ImageSubresourceRange) -> Self {
        let cursor = DenseMapCursor::new(&map, range);

        Self {
            __: PhantomData,
            cursor,
            map,
            value,
        }
    }
}

impl<'a, M, V: Copy + PartialEq> Iterator for DenseMapIter<'a, M, V>
where
    M: DerefMut<Target = DenseMap<V>>,
{
    type Item = (V, vk::ImageSubresourceRange);

    fn next(&mut self) -> Option<Self::Item> {
        self.cursor.next(&mut self.map, self.value)
    }
}

#[derive(Copy, Clone)]
struct DenseMapRange {
    aspect_count: u8,
    base_array_layer: u32,
    base_aspect_bit: u8,
    base_mip_level: u32,
    layer_count: u32,
    level_count: u32,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DenseSharingState {
    Idle = 0,
    Promoting = 1,
    Dense = 2,
}

#[derive(Debug)]
struct DualAspectAccess([AtomicU8; 2]);

impl DualAspectAccess {
    fn new(access: AccessType) -> Self {
        let access = access_type_into_u8(access);

        Self([AtomicU8::new(access), AtomicU8::new(access)])
    }

    fn load(&self, aspect_idx: usize) -> AccessType {
        access_type_from_u8(self.0[aspect_idx].load(Ordering::Acquire))
    }
}

struct DualAspectAccessIter<'a> {
    dual: &'a DualAspectAccess,
    format_aspect_mask: vk::ImageAspectFlags,
    next_access: AccessType,
    ranges: ImageSubresourceRangeIter,
}

impl<'a> DualAspectAccessIter<'a> {
    fn new(
        dual: &'a DualAspectAccess,
        info: ImageInfo,
        next_access: AccessType,
        access_range: vk::ImageSubresourceRange,
    ) -> Self {
        debug_assert_eq!(access_range.base_array_layer, 0);
        debug_assert_eq!(access_range.base_mip_level, 0);
        debug_assert_eq!(access_range.layer_count, 1);
        debug_assert_eq!(access_range.level_count, 1);

        Self {
            dual,
            format_aspect_mask: format_aspect_mask(info.format),
            next_access,
            ranges: ImageSubresourceRangeIter::new(access_range),
        }
    }
}

impl ExactSizeIterator for DualAspectAccessIter<'_> {
    fn len(&self) -> usize {
        self.ranges.len()
    }
}

impl Iterator for DualAspectAccessIter<'_> {
    type Item = (AccessType, vk::ImageSubresourceRange);

    fn next(&mut self) -> Option<Self::Item> {
        let range = self.ranges.next()?;
        let aspect_idx = aspect_ordinal(self.format_aspect_mask, range.aspect_mask) as usize;
        let prev_access = access_type_from_u8(
            self.dual.0[aspect_idx].swap(access_type_into_u8(self.next_access), Ordering::AcqRel),
        );

        Some((prev_access, range))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ranges.size_hint()
    }
}

#[derive(Debug)]
struct ExclusiveSharing {
    // `promoting` keeps whole-image updates on the dense path while a partial update is
    // converting uniform tracking into subresource tracking
    dense_sharing_state: AtomicU8,
    uniform: AtomicU64,
}

impl ExclusiveSharing {
    fn new(_info: ImageInfo) -> Self {
        let sharing = SharingMode::Exclusive(None);

        Self {
            uniform: AtomicU64::new(sharing.encode()),
            dense_sharing_state: AtomicU8::new(0),
        }
    }

    fn dense_sharing_state(&self) -> DenseSharingState {
        match self.dense_sharing_state.load(Ordering::Acquire) {
            0 => DenseSharingState::Idle,
            1 => DenseSharingState::Promoting,
            2 => DenseSharingState::Dense,
            _ => unreachable!("invalid image dense sharing state"),
        }
    }

    fn is_dense_sharing_active(&self) -> bool {
        self.dense_sharing_state() == DenseSharingState::Dense
    }

    fn is_promoting_dense_sharing(&self) -> bool {
        self.dense_sharing_state() == DenseSharingState::Promoting
    }

    fn uses_dense_sharing(&self) -> bool {
        self.dense_sharing_state() != DenseSharingState::Idle
    }

    fn set_promoting_dense_sharing(&self) {
        self.dense_sharing_state
            .store(DenseSharingState::Promoting as _, Ordering::Release);
    }

    fn set_dense_sharing_active(&self) {
        self.dense_sharing_state
            .store(DenseSharingState::Dense as _, Ordering::Release);
    }

    fn set_ranges(
        &self,
        dense: &Mutex<Option<DenseMap<SharingMode>>>,
        info: ImageInfo,
        sharing: SharingMode,
        sharing_ranges: &[vk::ImageSubresourceRange],
    ) {
        if sharing_ranges.is_empty() {
            return;
        }

        if sharing_ranges.len() == 1 && info.is_full_subresource_range(sharing_ranges[0]) {
            self.set_uniform_or_dense_sharing(dense, info, sharing, sharing_ranges[0]);

            return;
        }

        self.promote_dense_sharing_and_set_ranges(dense, info, sharing, sharing_ranges);
    }

    fn set_uniform_or_dense_sharing(
        &self,
        dense: &Mutex<Option<DenseMap<SharingMode>>>,
        _info: ImageInfo,
        sharing: SharingMode,
        sharing_range: vk::ImageSubresourceRange,
    ) {
        let encoded_sharing = sharing.encode();

        loop {
            if self.uses_dense_sharing() {
                let mut dense = dense.lock();

                #[cfg(not(feature = "parking_lot"))]
                let mut dense = dense.expect("poisoned image dense lock");

                dense
                    .as_mut()
                    .expect("missing dense sharing state")
                    .swap(sharing, sharing_range);

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
                if self.is_promoting_dense_sharing() {
                    let mut dense = dense.lock();

                    #[cfg(not(feature = "parking_lot"))]
                    let mut dense = dense.expect("poisoned image dense lock");

                    dense
                        .as_mut()
                        .expect("missing dense sharing state")
                        .swap(sharing, sharing_range);
                }

                return;
            }
        }
    }

    fn promote_dense_sharing_and_set_ranges(
        &self,
        dense: &Mutex<Option<DenseMap<SharingMode>>>,
        info: ImageInfo,
        sharing: SharingMode,
        sharing_ranges: &[vk::ImageSubresourceRange],
    ) {
        let mut dense = dense.lock();

        #[cfg(not(feature = "parking_lot"))]
        let mut dense = dense.expect("poisoned image dense lock");

        if self.is_dense_sharing_active() {
            let dense_sharing = dense.as_mut().expect("missing dense sharing state");
            for &sharing_range in sharing_ranges {
                dense_sharing.swap(sharing, info.resolve_subresource_counts(sharing_range));
            }

            return;
        }

        self.set_promoting_dense_sharing();

        let current = SharingMode::decode(self.uniform.load(Ordering::Acquire));

        *dense = Some(DenseMap::new(info, current));
        let sharing_state = dense.as_mut().expect("missing dense sharing state");
        for &sharing_range in sharing_ranges {
            sharing_state.swap(sharing, info.resolve_subresource_counts(sharing_range));
        }

        self.set_dense_sharing_active();
    }
}

/// Smart pointer handle to an [image] object.
///
/// Also contains information about the object.
///
/// ```no_run
/// # use ash::vk;
/// # use vk_sync::AccessType;
/// # use vk_graph::driver::DriverError;
/// # use vk_graph::driver::device::{Device, DeviceInfo};
/// # use vk_graph::driver::image::{Image, ImageInfo};
/// # fn main() -> Result<(), DriverError> {
/// # let device = Device::create(DeviceInfo::default())?;
/// let fmt = vk::Format::R8G8B8A8_UNORM;
/// let usage = vk::ImageUsageFlags::SAMPLED;
/// let info = ImageInfo::image_2d(320, 200, fmt, usage);
/// let my_img = Image::create(&device, info)?;
///
/// assert_eq!(my_img.info, info);
/// assert_ne!(my_img.handle, vk::Image::null());
/// # Ok(()) }
/// ```
///
/// [image]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkImage.html
#[read_only::cast]
pub struct Image {
    access: Access,
    allocation: Option<Allocation>, // None when we don't own the image (Swapchain images)
    dense_access: Mutex<Option<DenseMap<AccessType>>>,
    dense_sharing: Mutex<Option<DenseMap<SharingMode>>>,

    /// The device which owns this image resource.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub device: Device,

    /// The native Vulkan resource handle of this image.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub handle: vk::Image,

    #[allow(clippy::type_complexity)]
    image_view_cache: Mutex<HashMap<ImageViewInfo, ImageView>>,

    /// Information used to create this resource.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub info: ImageInfo,

    sharing: Sharing,
}

impl Image {
    /// Creates a new image on the given device.
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
    /// # use vk_graph::driver::image::{Image, ImageInfo};
    /// # fn main() -> Result<(), DriverError> {
    /// # let device = Device::create(DeviceInfo::default())?;
    /// let info = ImageInfo::image_2d(
    ///     32,
    ///     32,
    ///     vk::Format::R8G8B8A8_UNORM,
    ///     vk::ImageUsageFlags::SAMPLED,
    /// );
    /// let image = Image::create(&device, info)?;
    ///
    /// assert_ne!(image.handle, vk::Image::null());
    /// assert_eq!(image.info.width, 32);
    /// assert_eq!(image.info.height, 32);
    /// # Ok(()) }
    /// ```
    #[profiling::function]
    pub fn create(device: &Device, info: impl Into<ImageInfo>) -> Result<Self, DriverError> {
        let info = info.into();

        //trace!("create: {:?}", &info);
        trace!("create");

        if info.usage.is_empty() {
            return Err(DriverError::InvalidData);
        }

        let access = Access::new(info, AccessType::Nothing);

        let device = device.clone();
        let create_info: ImageCreateInfo = info.into();
        let create_info = if info.sharing_mode == vk::SharingMode::CONCURRENT {
            create_info.queue_family_indices(&device.physical_device.queue_family_indices)
        } else {
            create_info
        };
        let handle = unsafe {
            device.create_image(&create_info, None).map_err(|err| {
                warn!("unable to create image: {err}");

                DriverError::Unsupported
            })?
        };
        let requirements = unsafe { device.get_image_memory_requirements(handle) };
        let allocation_scheme = if info.alloc_dedicated {
            AllocationScheme::DedicatedImage(handle)
        } else {
            AllocationScheme::GpuAllocatorManaged
        };
        let allocation = {
            profiling::scope!("allocate");

            Device::with_allocator(&device, |allocator| {
                allocator
                    .allocate(&AllocationCreateDesc {
                        name: "image",
                        requirements,
                        location: info.memory_location(),
                        linear: false,
                        allocation_scheme,
                    })
                    .map_err(|err| {
                        warn!("unable to allocate image memory: {err}");

                        unsafe {
                            device.destroy_image(handle, None);
                        }

                        DriverError::from_alloc_err(err)
                    })
                    .and_then(|allocation| {
                        if let Err(err) = unsafe {
                            device.bind_image_memory(
                                handle,
                                allocation.memory(),
                                allocation.offset(),
                            )
                        } {
                            warn!("unable to bind image memory: {err}");

                            if let Err(err) = allocator.free(allocation) {
                                warn!("unable to free image allocation: {err}")
                            }

                            unsafe {
                                device.destroy_image(handle, None);
                            }

                            Err(DriverError::OutOfMemory)
                        } else {
                            Ok(allocation)
                        }
                    })
            })
        }?;

        debug_assert_ne!(handle, vk::Image::null());

        Ok(Self {
            access,
            allocation: Some(allocation),
            dense_access: Mutex::new(None),
            dense_sharing: Mutex::new(None),
            device,
            handle,
            image_view_cache: Mutex::new(Default::default()),
            info,
            sharing: Sharing::new(info, info.sharing_mode),
        })
    }

    /// Drops the given allocation, all views, and the handle.
    #[profiling::function]
    fn drop_allocation(&self, allocation: Allocation) {
        {
            profiling::scope!("views");

            self.with_image_view_cache(|cache| cache.clear());
        }

        unsafe {
            self.device.destroy_image(self.handle, None);
        }

        {
            profiling::scope!("deallocate");

            Device::with_allocator(&self.device, |allocator| allocator.free(allocation))
        }
        .unwrap_or_else(|err| warn!("unable to free image allocation: {err}"));
    }

    /// Consumes a Vulkan image created by some other library.
    ///
    /// The image is not destroyed automatically on drop, unlike images created through the
    /// [`Image::create`] function.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid [`vk::Image`] created from `device`, and `info` must accurately
    /// describe the image's format, extent, usage, sharing mode, and subresource counts. The caller
    /// remains responsible for keeping the handle and its memory backing valid until all wrappers
    /// created from this function are no longer used.
    #[profiling::function]
    pub unsafe fn from_raw(device: &Device, handle: vk::Image, info: impl Into<ImageInfo>) -> Self {
        let device = device.clone();
        let info = info.into();

        let access = Access::new(info, AccessType::Nothing);

        Self {
            access,
            allocation: None,
            dense_access: Mutex::new(None),
            dense_sharing: Mutex::new(None),
            device,
            handle,
            image_view_cache: Mutex::new(Default::default()),
            info,
            sharing: Sharing::new(info, info.sharing_mode),
        }
    }

    /// Sets the debugging name assigned to this image.
    pub fn set_debug_name(&self, name: impl AsRef<str>) {
        Device::try_set_debug_utils_object_name(&self.device, self.handle, &name);
        Device::try_set_private_data_object_name(
            &self.device,
            vk::ObjectType::IMAGE,
            self.handle,
            &name,
        );
    }

    pub(crate) fn set_sharing_ranges(
        &self,
        sharing: SharingMode,
        sharing_ranges: &[vk::ImageSubresourceRange],
    ) {
        self.sharing
            .set_ranges(&self.dense_sharing, self.info, sharing, sharing_ranges);
    }

    /// Keeps track of some next `access` which affects a `range` of this image.
    ///
    /// Returns the previous access for which a pipeline barrier should be used to prevent data
    /// corruption.
    #[profiling::function]
    pub(crate) fn swap_access(
        &self,
        next_access: AccessType,
        mut access_range: vk::ImageSubresourceRange,
    ) -> impl Iterator<Item = (AccessType, vk::ImageSubresourceRange)> + '_ {
        #[cfg(feature = "checked")]
        {
            assert_aspect_mask_supported(access_range.aspect_mask);

            assert!(format_aspect_mask(self.info.format).contains(access_range.aspect_mask));
        }

        if access_range.layer_count == vk::REMAINING_ARRAY_LAYERS {
            debug_assert!(access_range.base_array_layer < self.info.array_layer_count);

            access_range.layer_count = self.info.array_layer_count - access_range.base_array_layer
        }

        debug_assert!(
            access_range.base_array_layer + access_range.layer_count <= self.info.array_layer_count
        );

        if access_range.level_count == vk::REMAINING_MIP_LEVELS {
            debug_assert!(access_range.base_mip_level < self.info.mip_level_count);

            access_range.level_count = self.info.mip_level_count - access_range.base_mip_level
        }

        debug_assert!(
            access_range.base_mip_level + access_range.level_count <= self.info.mip_level_count
        );

        self.access
            .swap(&self.dense_access, self.info, next_access, access_range)
    }

    pub(crate) fn swap_accesses<'a, I>(
        &'a self,
        accesses: I,
    ) -> impl Iterator<Item = (AccessType, AccessType, vk::ImageSubresourceRange)> + 'a
    where
        I: IntoIterator<Item = (AccessType, vk::ImageSubresourceRange)>,
        I::IntoIter: 'a,
    {
        let info = self.info;
        let format_aspect_mask = format_aspect_mask(info.format);
        let accesses = accesses
            .into_iter()
            .map(move |(next_access, access_range)| {
                #[cfg(feature = "checked")]
                {
                    assert_aspect_mask_supported(access_range.aspect_mask);

                    assert!(format_aspect_mask.contains(access_range.aspect_mask));
                }

                (next_access, info.resolve_subresource_counts(access_range))
            });

        struct Iter<'a, I>
        where
            I: Iterator<Item = (AccessType, vk::ImageSubresourceRange)>,
        {
            access: &'a Access,
            accesses: I,
            dense_access: &'a Mutex<Option<DenseMap<AccessType>>>,
            info: ImageInfo,
            current: Option<(AccessType, AccessIter<'a>)>,
        }

        impl<I> Iterator for Iter<'_, I>
        where
            I: Iterator<Item = (AccessType, vk::ImageSubresourceRange)>,
        {
            type Item = (AccessType, AccessType, vk::ImageSubresourceRange);

            fn next(&mut self) -> Option<Self::Item> {
                loop {
                    if let Some((next_access, iter)) = self.current.as_mut() {
                        if let Some((prev_access, range)) = iter.next() {
                            return Some((*next_access, prev_access, range));
                        }

                        self.current = None;
                    }

                    let (next_access, access_range) = self.accesses.next()?;
                    self.current = Some((
                        next_access,
                        self.access
                            .swap(self.dense_access, self.info, next_access, access_range),
                    ));
                }
            }
        }

        impl<I> Drop for Iter<'_, I>
        where
            I: Iterator<Item = (AccessType, vk::ImageSubresourceRange)>,
        {
            fn drop(&mut self) {
                while self.next().is_some() {}
            }
        }

        Iter {
            access: &self.access,
            accesses,
            dense_access: &self.dense_access,
            info,
            current: None,
        }
    }

    /// TODO
    pub fn sync_info(&self) -> ImageSyncInfo {
        ImageSyncInfo {
            subresources: ImageSyncInfo::compact_subresources(
                self.sync_info_with_sharing()
                    .map(|(subresource, sharing)| subresource.into_public(sharing)),
            ),
        }
    }

    pub(crate) fn sync_info_with_sharing(
        &self,
    ) -> impl Iterator<Item = (ImageSubresourceSyncInfo, SharingMode)> {
        self.sync_info_with_sharing_range(vk::ImageSubresourceRange {
            aspect_mask: format_aspect_mask(self.info.format),
            base_mip_level: 0,
            level_count: self.info.mip_level_count,
            base_array_layer: 0,
            layer_count: self.info.array_layer_count,
        })
    }

    pub(crate) fn sync_info_with_sharing_range(
        &self,
        query_range: vk::ImageSubresourceRange,
    ) -> impl Iterator<Item = (ImageSubresourceSyncInfo, SharingMode)> {
        #[derive(Clone, Copy)]
        enum SharingSource {
            Concurrent,
            Uniform(SharingMode),
            Dense,
        }

        let query_range = self.info.resolve_subresource_counts(query_range);
        let subresource_ranges = ImageSubresourceRangeIter::new(query_range);
        let format_aspect_mask = format_aspect_mask(self.info.format);
        #[derive(Clone, Copy)]
        enum AccessSource<'a> {
            Uniform(AccessType),
            DualAspect(&'a DualAspectAccess),
            Dense,
        }

        let access_source = match &self.access {
            Access::Uniform(uniform) => AccessSource::Uniform(uniform.load()),
            Access::DualAspect(dual) => AccessSource::DualAspect(dual),
            Access::Dense(access) if access.uses_dense() => AccessSource::Dense,
            Access::Dense(access) => AccessSource::Uniform(access.load()),
        };
        let sharing_source = match &self.sharing {
            Sharing::Concurrent => SharingSource::Concurrent,
            Sharing::Exclusive(exclusive) if exclusive.uses_dense_sharing() => SharingSource::Dense,
            Sharing::Exclusive(exclusive) => SharingSource::Uniform(SharingMode::decode(
                exclusive.uniform.load(Ordering::Acquire),
            )),
        };

        struct UniformSyncInfoIter {
            access: AccessType,
            sharing: SharingMode,
            subresource_ranges: ImageSubresourceRangeIter,
        }

        impl Iterator for UniformSyncInfoIter {
            type Item = (ImageSubresourceSyncInfo, SharingMode);

            fn next(&mut self) -> Option<Self::Item> {
                self.subresource_ranges.next().map(|range| {
                    (
                        ImageSubresourceSyncInfo::from_access(self.access, range),
                        self.sharing,
                    )
                })
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.subresource_ranges.size_hint()
            }
        }

        impl ExactSizeIterator for UniformSyncInfoIter {
            fn len(&self) -> usize {
                self.subresource_ranges.len()
            }
        }

        struct DenseSyncInfoIter<'a> {
            access_source: AccessSource<'a>,
            format_aspect_mask: vk::ImageAspectFlags,
            access_dense: Option<MutexGuard<'a, Option<DenseMap<AccessType>>>>,
            sharing_dense: Option<MutexGuard<'a, Option<DenseMap<SharingMode>>>>,
            sharing_source: SharingSource,
            subresource_ranges: ImageSubresourceRangeIter,
        }

        impl Iterator for DenseSyncInfoIter<'_> {
            type Item = (ImageSubresourceSyncInfo, SharingMode);

            fn next(&mut self) -> Option<Self::Item> {
                let range = self.subresource_ranges.next()?;
                let aspect = aspect_ordinal(self.format_aspect_mask, range.aspect_mask);
                let access = match self.access_source {
                    AccessSource::Uniform(access) => access,
                    AccessSource::DualAspect(dual) => dual.load(aspect as usize),
                    AccessSource::Dense => self
                        .access_dense
                        .as_ref()
                        .expect("missing dense access state")
                        .as_ref()
                        .expect("missing dense access map")
                        .subresource(aspect, range.base_array_layer, range.base_mip_level),
                };
                let sharing = match self.sharing_source {
                    SharingSource::Concurrent => SharingMode::Concurrent,
                    SharingSource::Uniform(sharing) => sharing,
                    SharingSource::Dense => self
                        .sharing_dense
                        .as_ref()
                        .expect("missing dense sharing state")
                        .as_ref()
                        .expect("missing dense sharing map")
                        .subresource(aspect, range.base_array_layer, range.base_mip_level),
                };

                Some((
                    ImageSubresourceSyncInfo::from_access(access, range),
                    sharing,
                ))
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.subresource_ranges.size_hint()
            }
        }

        impl ExactSizeIterator for DenseSyncInfoIter<'_> {
            fn len(&self) -> usize {
                self.subresource_ranges.len()
            }
        }

        enum SyncInfoIter<'a> {
            Uniform(UniformSyncInfoIter),
            Dense(DenseSyncInfoIter<'a>),
        }

        impl Iterator for SyncInfoIter<'_> {
            type Item = (ImageSubresourceSyncInfo, SharingMode);

            fn next(&mut self) -> Option<Self::Item> {
                match self {
                    Self::Uniform(iter) => iter.next(),
                    Self::Dense(iter) => iter.next(),
                }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.len();

                (len, Some(len))
            }
        }

        impl ExactSizeIterator for SyncInfoIter<'_> {
            fn len(&self) -> usize {
                match self {
                    Self::Uniform(iter) => iter.len(),
                    Self::Dense(iter) => iter.len(),
                }
            }
        }

        let uniform_sharing = match sharing_source {
            SharingSource::Concurrent => Some(SharingMode::Concurrent),
            SharingSource::Uniform(sharing) => Some(sharing),
            SharingSource::Dense => None,
        };

        let sync_infos = if let (AccessSource::Uniform(access), Some(sharing)) =
            (access_source, uniform_sharing)
        {
            SyncInfoIter::Uniform(UniformSyncInfoIter {
                access,
                sharing,
                subresource_ranges,
            })
        } else {
            let access_dense = if matches!(access_source, AccessSource::Dense) {
                let dense = self.dense_access.lock();

                #[cfg(not(feature = "parking_lot"))]
                let dense = dense.expect("poisoned image dense access lock");

                Some(dense)
            } else {
                None
            };
            let sharing_dense = if matches!(sharing_source, SharingSource::Dense) {
                let dense = self.dense_sharing.lock();

                #[cfg(not(feature = "parking_lot"))]
                let dense = dense.expect("poisoned image dense sharing lock");

                Some(dense)
            } else {
                None
            };

            SyncInfoIter::Dense(DenseSyncInfoIter {
                access_source,
                format_aspect_mask,
                access_dense,
                sharing_dense,
                sharing_source,
                subresource_ranges,
            })
        };

        struct CompactIter<I, P, M> {
            iter: I,
            pending: Option<(ImageSubresourceSyncInfo, SharingMode)>,
            can_merge: P,
            merge: M,
        }

        /*
        Lazily compacts adjacent iterator entries. Each pass is linear in the number of source
        entries and keeps only one pending entry, so it uses `O(1)` extra memory. The image sync
        iterator applies this twice: first to merge mip levels, then to merge array layers.
        */
        impl<I, P, M> CompactIter<I, P, M>
        where
            I: Iterator<Item = (ImageSubresourceSyncInfo, SharingMode)>,
            P: Fn(
                (ImageSubresourceSyncInfo, SharingMode),
                (ImageSubresourceSyncInfo, SharingMode),
            ) -> bool,
            M: Fn(
                &mut (ImageSubresourceSyncInfo, SharingMode),
                (ImageSubresourceSyncInfo, SharingMode),
            ),
        {
            fn new(iter: I, can_merge: P, merge: M) -> Self {
                Self {
                    iter,
                    pending: None,
                    can_merge,
                    merge,
                }
            }
        }

        impl<I, P, M> Iterator for CompactIter<I, P, M>
        where
            I: Iterator<Item = (ImageSubresourceSyncInfo, SharingMode)>,
            P: Fn(
                (ImageSubresourceSyncInfo, SharingMode),
                (ImageSubresourceSyncInfo, SharingMode),
            ) -> bool,
            M: Fn(
                &mut (ImageSubresourceSyncInfo, SharingMode),
                (ImageSubresourceSyncInfo, SharingMode),
            ),
        {
            type Item = (ImageSubresourceSyncInfo, SharingMode);

            fn next(&mut self) -> Option<Self::Item> {
                let mut pending = self.pending.take().or_else(|| self.iter.next())?;

                for next in self.iter.by_ref() {
                    if (self.can_merge)(pending, next) {
                        (self.merge)(&mut pending, next);
                    } else {
                        self.pending = Some(next);
                        return Some(pending);
                    }
                }

                Some(pending)
            }
        }

        let same_sync_and_sharing =
            |lhs: (ImageSubresourceSyncInfo, SharingMode),
             rhs: (ImageSubresourceSyncInfo, SharingMode)| {
                lhs.0.same_sync(rhs.0) && lhs.1 == rhs.1
            };
        let merge_array_layers =
            |lhs: &mut (ImageSubresourceSyncInfo, SharingMode),
             rhs: (ImageSubresourceSyncInfo, SharingMode)| {
                lhs.0.merge_array_layers(rhs.0);
            };
        let merge_mip_levels =
            |lhs: &mut (ImageSubresourceSyncInfo, SharingMode),
             rhs: (ImageSubresourceSyncInfo, SharingMode)| {
                lhs.0.merge_mip_levels(rhs.0);
            };

        let mip_levels = CompactIter::new(sync_infos, same_sync_and_sharing, merge_mip_levels);

        CompactIter::new(mip_levels, same_sync_and_sharing, merge_array_layers)
    }

    /// Produces a new `Image` sharing the same Vulkan handle with independent access tracking.
    ///
    /// The returned image retains the handle, device, and debug name of `self` but starts with
    /// no prior access history (`AccessType::Nothing`) and does not claim ownership of the image's
    /// memory backing. Internal caches are moved out of `self` so they are not duplicated.
    ///
    /// This is used to create separate tracking instances for swapchain images that may be
    /// used concurrently across different graph executions.
    ///
    /// # Safety
    ///
    /// The caller must ensure the Vulkan image handle remains valid for the lifetime of the
    /// returned `Image`. This function should only be called on swapchain images or other
    /// platform or extension images.
    #[profiling::function]
    pub unsafe fn to_detached(&self) -> Self {
        debug_assert!(self.allocation.is_none());

        let image_view_cache = self.with_image_view_cache(take);

        let Self { handle, info, .. } = *self;

        Self {
            access: Access::new(info, AccessType::Nothing),
            allocation: None,
            dense_access: Mutex::new(None),
            dense_sharing: Mutex::new(None),
            device: self.device.clone(),
            handle,
            image_view_cache: Mutex::new(image_view_cache),
            info,
            sharing: Sharing::new(info, info.sharing_mode),
        }
    }

    #[profiling::function]
    pub(crate) fn view(&self, info: ImageViewInfo) -> Result<vk::ImageView, DriverError> {
        self.with_image_view_cache(|cache| {
            Ok(match cache.entry(info) {
                Entry::Occupied(entry) => entry.get().image_view,
                Entry::Vacant(entry) => {
                    entry
                        .insert(ImageView::create(&self.device, info, self.handle)?)
                        .image_view
                }
            })
        })
    }

    /// Sets the debugging name assigned to this image.
    pub fn with_debug_name(self, name: impl AsRef<str>) -> Self {
        self.set_debug_name(name);

        self
    }

    fn with_image_view_cache<R>(
        &self,
        f: impl FnOnce(&mut HashMap<ImageViewInfo, ImageView>) -> R,
    ) -> R {
        let cache = self.image_view_cache.lock();

        #[cfg(not(feature = "parking_lot"))]
        let cache = cache.expect("poisoned image view lock");

        let mut cache = cache;

        f(&mut cache)
    }
}

impl Debug for Image {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut res = f.debug_struct(stringify!(Image));

        if let Some(debug_name) =
            &Device::private_data_object_name(&self.device, vk::ObjectType::IMAGE, self.handle)
        {
            res.field("debug_name", debug_name);
        }

        res.field("handle", &self.handle).finish_non_exhaustive()
    }
}

impl Drop for Image {
    // This function is not profiled because dropping the allocation may run during shutdown.
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        /*
        When allocation is Some, we allocated the image ourselves; otherwise somebody else owns this
        image and we should not destroy it. Usually it's the swapchain.
        */
        if let Some(allocation) = self.allocation.take() {
            Device::try_clear_private_data_object_name(
                &self.device,
                vk::ObjectType::IMAGE,
                self.handle,
            );
            Self::drop_allocation(self, allocation);
        } else {
            // Non-owned handles may already be invalid when their owner, such as a swapchain, has
            // been destroyed. Remove local metadata without issuing vkSetPrivateDataEXT.
            Device::forget_private_data_object_name(
                &self.device,
                vk::ObjectType::IMAGE,
                self.handle,
            );
        }
    }
}

impl Eq for Image {}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

/// Information used to create an [`Image`] instance.
///
/// See [`VkImageCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageCreateInfo.html).
#[derive(Builder, Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[builder(
    build_fn(private, name = "fallible_build"),
    derive(Copy, Clone, Debug),
    pattern = "owned"
)]
pub struct ImageInfo {
    /// Specifies a dedicated memory allocation managed by the Vulkan driver and not by the
    /// internal memory allocation pool transient resources share.
    ///
    /// The driver may optimize access to dedicated images.
    #[builder(default)]
    pub alloc_dedicated: bool,

    /// The number of layers in the image.
    #[builder(default = "1")]
    pub array_layer_count: u32,

    /// Image extent of the Z axis, when describing a three dimensional image.
    #[builder(default)]
    pub depth: u32,

    /// A bitmask describing additional parameters of the image.
    ///
    /// See [`VkImageCreateFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageCreateFlagBits.html).
    #[builder(default)]
    pub flags: vk::ImageCreateFlags,

    /// The format and type of the texel blocks that will be contained in the image.
    #[builder(default = "vk::Format::UNDEFINED")]
    pub format: vk::Format,

    /// Image extent of the Y axis, when describing a two or three dimensional image.
    #[builder(default)]
    pub height: u32,

    /// Specifies an image whose memory is host-visible and may be mapped for reads.
    ///
    /// Memory optimal for CPU readback of data may be used.
    ///
    #[builder(default)]
    pub host_readable: bool,

    /// Specifies an image whose memory is host-visible and may be mapped for writes.
    ///
    /// Memory optimal for uploading data to the GPU may be used.
    ///
    #[builder(default)]
    pub host_writable: bool,

    /// The number of levels of detail available for minified sampling of the image.
    #[builder(default = "1")]
    pub mip_level_count: u32,

    /// Specifies the number of [samples per texel].
    ///
    /// See [`VkImageCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageCreateInfo.html).
    #[builder(default = "SampleCount::Type1")]
    pub sample_count: SampleCount,

    /// Controls whether the image is accessible from a single queue family (`EXCLUSIVE`) or from
    /// multiple queue families concurrently (`CONCURRENT`).
    ///
    /// `EXCLUSIVE` (the default) restricts the image to a single queue family. This may enable
    /// driver optimizations but requires ownership transfers to use the image on a different queue
    /// family.
    ///
    /// Set to `CONCURRENT` when the image will be accessed from multiple queue families (e.g.
    /// graphics and compute on separate queues).
    ///
    /// See
    /// [`VkSharingMode`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSharingMode.html)
    /// in the Vulkan specification.
    #[builder(default = "vk::SharingMode::EXCLUSIVE")]
    pub sharing_mode: vk::SharingMode,

    /// Specifies the tiling arrangement of the texel blocks in memory.
    ///
    /// The default value is [`vk::ImageTiling::OPTIMAL`].
    #[builder(default = "vk::ImageTiling::OPTIMAL")]
    pub tiling: vk::ImageTiling,

    /// The basic dimensionality of the image.
    ///
    /// Layers in array textures do not count as a dimension for the purposes of the image type.
    #[builder(default = "vk::ImageType::TYPE_2D")]
    pub ty: vk::ImageType,

    /// A bitmask describing the intended usage of the image.
    ///
    /// See [`VkImageUsageFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageUsageFlagBits.html).
    #[builder(default)]
    pub usage: vk::ImageUsageFlags,

    /// Image extent of the X axis.
    #[builder(default)]
    pub width: u32,
}

impl ImageInfo {
    /// Specifies a cube image.
    #[inline(always)]
    pub const fn cube(size: u32, format: vk::Format, usage: vk::ImageUsageFlags) -> ImageInfo {
        let mut res = Self::new(vk::ImageType::TYPE_2D, size, size, 1, 6, format, usage);
        res.flags = vk::ImageCreateFlags::from_raw(
            vk::ImageCreateFlags::CUBE_COMPATIBLE.as_raw() | res.flags.as_raw(),
        );

        res
    }

    /// Specifies a one-dimensional image.
    #[inline(always)]
    pub const fn image_1d(size: u32, format: vk::Format, usage: vk::ImageUsageFlags) -> ImageInfo {
        Self::new(vk::ImageType::TYPE_1D, size, 1, 1, 1, format, usage)
    }

    /// Specifies a two-dimensional image.
    #[inline(always)]
    pub const fn image_2d(
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> ImageInfo {
        Self::new(vk::ImageType::TYPE_2D, width, height, 1, 1, format, usage)
    }

    /// Specifies a two-dimensional image array.
    #[inline(always)]
    pub const fn image_2d_array(
        width: u32,
        height: u32,
        array_layer_count: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> ImageInfo {
        Self::new(
            vk::ImageType::TYPE_2D,
            width,
            height,
            1,
            array_layer_count,
            format,
            usage,
        )
    }

    /// Specifies a three-dimensional image.
    #[inline(always)]
    pub const fn image_3d(
        width: u32,
        height: u32,
        depth: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> ImageInfo {
        Self::new(
            vk::ImageType::TYPE_3D,
            width,
            height,
            depth,
            1,
            format,
            usage,
        )
    }

    #[inline(always)]
    const fn new(
        ty: vk::ImageType,
        width: u32,
        height: u32,
        depth: u32,
        array_layer_count: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Self {
        Self {
            alloc_dedicated: false,
            ty,
            width,
            height,
            depth,
            array_layer_count,
            format,
            usage,
            flags: vk::ImageCreateFlags::empty(),
            host_readable: false,
            host_writable: false,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            tiling: vk::ImageTiling::OPTIMAL,
            mip_level_count: 1,
            sample_count: SampleCount::Type1,
        }
    }

    /// Creates a default `ImageInfoBuilder`.
    pub fn builder() -> ImageInfoBuilder {
        Default::default()
    }

    /// Provides an `ImageViewInfo` for this format, type, aspect, array elements, and mip levels.
    pub fn into_image_view(self) -> ImageViewInfo {
        self.into()
    }

    pub(crate) fn resolve_subresource_counts(
        self,
        mut range: vk::ImageSubresourceRange,
    ) -> vk::ImageSubresourceRange {
        if range.layer_count == vk::REMAINING_ARRAY_LAYERS {
            range.layer_count = self.array_layer_count - range.base_array_layer;
        }

        if range.level_count == vk::REMAINING_MIP_LEVELS {
            range.level_count = self.mip_level_count - range.base_mip_level;
        }

        range
    }

    fn is_full_subresource_range(self, range: vk::ImageSubresourceRange) -> bool {
        range.aspect_mask == format_aspect_mask(self.format)
            && range.base_array_layer == 0
            && range.layer_count == self.array_layer_count
            && range.base_mip_level == 0
            && range.level_count == self.mip_level_count
    }

    /// Returns `true` if this image is an array.
    pub fn is_array(self) -> bool {
        self.array_layer_count > 1
    }

    /// Returns `true` if this image is a cube or cube array.
    pub fn is_cube(self) -> bool {
        self.ty == vk::ImageType::TYPE_2D
            && self.width == self.height
            && self.depth == 1
            && self.array_layer_count >= 6
            && self.flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE)
    }

    /// Returns `true` if this image is a cube array.
    pub fn is_cube_array(self) -> bool {
        self.is_cube() && self.array_layer_count > 6
    }

    /// Returns `true` if this information specifies host-visible memory.
    pub fn is_host_visible(self) -> bool {
        self.host_readable | self.host_writable
    }

    const fn memory_location(self) -> MemoryLocation {
        if self.host_writable {
            MemoryLocation::CpuToGpu
        } else if self.host_readable {
            MemoryLocation::GpuToCpu
        } else {
            MemoryLocation::GpuOnly
        }
    }

    /// Converts an `ImageInfo` into an `ImageInfoBuilder`.
    pub fn into_builder(self) -> ImageInfoBuilder {
        ImageInfoBuilder {
            array_layer_count: Some(self.array_layer_count),
            alloc_dedicated: Some(self.alloc_dedicated),
            depth: Some(self.depth),
            flags: Some(self.flags),
            format: Some(self.format),
            height: Some(self.height),
            host_readable: Some(self.host_readable),
            host_writable: Some(self.host_writable),
            mip_level_count: Some(self.mip_level_count),
            sample_count: Some(self.sample_count),
            sharing_mode: Some(self.sharing_mode),
            tiling: Some(self.tiling),
            ty: Some(self.ty),
            usage: Some(self.usage),
            width: Some(self.width),
        }
    }
}

impl From<ImageInfo> for vk::ImageCreateInfo<'_> {
    fn from(value: ImageInfo) -> Self {
        Self::default()
            .flags(value.flags)
            .image_type(value.ty)
            .format(value.format)
            .extent(vk::Extent3D {
                width: value.width,
                height: value.height,
                depth: value.depth,
            })
            .mip_levels(value.mip_level_count)
            .array_layers(value.array_layer_count)
            .samples(value.sample_count.into())
            .tiling(value.tiling)
            .usage(value.usage)
            .sharing_mode(value.sharing_mode)
            .initial_layout(vk::ImageLayout::UNDEFINED)
    }
}

impl From<ImageInfoBuilder> for ImageInfo {
    fn from(info: ImageInfoBuilder) -> Self {
        info.build()
    }
}

impl From<ImageInfo> for vk::ImageSubresourceRange {
    fn from(info: ImageInfo) -> Self {
        let image_view_info: ImageViewInfo = info.into();

        image_view_info.into()
    }
}

impl ImageInfoBuilder {
    /// Builds a new `ImageInfo`.
    #[inline(always)]
    pub fn build(self) -> ImageInfo {
        self.fallible_build().expect("all fields have defaults")
    }

    /// Provides an `ImageViewInfo` for this format, type, aspect, array elements, and mip levels.
    pub fn into_image_view(self) -> ImageViewInfoBuilder {
        self.build().into_image_view().into_builder()
    }
}

struct ImageSubresourceRangeIter {
    aspect_mask: vk::ImageAspectFlags,
    aspect: u8,
    aspect_count: u8,
    array_layer: u32,
    end_array_layer: u32,
    base_array_layer: u32,
    base_mip_level: u32,
    mip_level: u32,
    end_mip_level: u32,
    remaining: usize,
}

impl ImageSubresourceRangeIter {
    fn new(range: vk::ImageSubresourceRange) -> Self {
        let aspect_mask = range.aspect_mask;
        let aspect_count = aspect_mask.as_raw().count_ones() as u8;

        Self {
            aspect_mask,
            aspect: 0,
            aspect_count,
            array_layer: range.base_array_layer,
            end_array_layer: range.base_array_layer + range.layer_count,
            base_array_layer: range.base_array_layer,
            base_mip_level: range.base_mip_level,
            mip_level: range.base_mip_level,
            end_mip_level: range.base_mip_level + range.level_count,
            remaining: aspect_count as usize
                * range.layer_count as usize
                * range.level_count as usize,
        }
    }
}

impl ExactSizeIterator for ImageSubresourceRangeIter {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl Iterator for ImageSubresourceRangeIter {
    type Item = vk::ImageSubresourceRange;

    fn next(&mut self) -> Option<Self::Item> {
        if self.aspect >= self.aspect_count {
            return None;
        }

        let range = vk::ImageSubresourceRange {
            aspect_mask: aspect_mask_at_ordinal(self.aspect_mask, self.aspect as u32),
            base_array_layer: self.array_layer,
            layer_count: 1,
            base_mip_level: self.mip_level,
            level_count: 1,
        };

        self.mip_level += 1;
        if self.mip_level >= self.end_mip_level {
            self.mip_level = self.base_mip_level;
            self.array_layer += 1;
            if self.array_layer >= self.end_array_layer {
                self.array_layer = self.base_array_layer;
                self.aspect += 1;
            }
        }

        self.remaining -= 1;

        Some(range)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();

        (len, Some(len))
    }
}

/// Synchronization information for one accessed image subresource range.
#[derive(Clone, Copy, Debug)]
pub struct ImageSubresourceSyncInfo {
    /// Access types performed by `stage_mask`.
    pub access_mask: vk::AccessFlags,

    /// Required image layout for the next external use, when one is defined.
    pub layout: Option<vk::ImageLayout>,

    /// Queue-family ownership for this subresource, when exclusive ownership is known.
    pub queue_family_index: Option<u32>,

    /// The tracked image subresource range.
    pub range: vk::ImageSubresourceRange,

    /// Pipeline stages that access this `range`.
    pub stage_mask: vk::PipelineStageFlags,
}

impl ImageSubresourceSyncInfo {
    fn can_merge_array_layers(self, other: Self) -> bool {
        self.same_sync(other)
            && self.range.aspect_mask == other.range.aspect_mask
            && self.range.base_mip_level == other.range.base_mip_level
            && self.range.level_count == other.range.level_count
            && self.range.base_array_layer + self.range.layer_count == other.range.base_array_layer
    }

    fn can_merge_mip_levels(self, other: Self) -> bool {
        self.same_sync(other)
            && self.range.aspect_mask == other.range.aspect_mask
            && self.range.base_array_layer == other.range.base_array_layer
            && self.range.layer_count == other.range.layer_count
            && self.range.base_mip_level + self.range.level_count == other.range.base_mip_level
    }

    fn from_access(access: AccessType, range: vk::ImageSubresourceRange) -> Self {
        let (stage_mask, access_mask) = pipeline_stage_access_flags(access);

        Self {
            access_mask,
            layout: access_type_to_layout(access),
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

    fn merge_array_layers(&mut self, other: Self) {
        self.range.layer_count += other.range.layer_count;
    }

    fn merge_mip_levels(&mut self, other: Self) {
        self.range.level_count += other.range.level_count;
    }

    fn same_sync(self, other: Self) -> bool {
        self.access_mask == other.access_mask
            && self.layout == other.layout
            && self.queue_family_index == other.queue_family_index
            && self.stage_mask == other.stage_mask
    }
}

/// Synchronization information for an image.
#[derive(Clone, Debug)]
pub struct ImageSyncInfo {
    /// Access state for the tracked image subresource ranges.
    pub subresources: Box<[ImageSubresourceSyncInfo]>,
}

impl ImageSyncInfo {
    fn compact_subresources(
        subresources: impl IntoIterator<Item = ImageSubresourceSyncInfo>,
    ) -> Box<[ImageSubresourceSyncInfo]> {
        let mut mip_levels = Vec::new();

        for sync_info in subresources {
            if let Some(prev) = mip_levels.last_mut()
                && ImageSubresourceSyncInfo::can_merge_mip_levels(*prev, sync_info)
            {
                prev.merge_mip_levels(sync_info);
            } else {
                mip_levels.push(sync_info);
            }
        }

        let mut array_layers = Vec::with_capacity(mip_levels.len());

        for sync_info in mip_levels {
            if let Some(prev) = array_layers.last_mut()
                && ImageSubresourceSyncInfo::can_merge_array_layers(*prev, sync_info)
            {
                prev.merge_array_layers(sync_info);
            } else {
                array_layers.push(sync_info);
            }
        }

        array_layers.into_boxed_slice()
    }

    /// Compacts adjacent subresource entries with identical synchronization requirements.
    ///
    /// This is opt-in because some callers may prefer the exact per-range snapshot produced by the
    /// internal access tracker.
    ///
    /// Runs in linear time over `subresources`. Image compaction performs two passes, first merging
    /// adjacent mip levels and then adjacent array layers, and uses temporary vector storage for
    /// each pass.
    pub fn compact(&mut self) {
        let subresources = take(&mut self.subresources);
        self.subresources = Self::compact_subresources(subresources);
    }

    /// Returns a compacted copy of this synchronization snapshot.
    ///
    /// This has the same linear-time and temporary-storage characteristics as [`Self::compact`],
    /// but consumes and returns the snapshot for use in iterator chains or expression-oriented code.
    pub fn into_compacted(mut self) -> Self {
        self.compact();
        self
    }
}

struct ImageView {
    device: Device,
    image_view: vk::ImageView,
}

impl ImageView {
    #[profiling::function]
    fn create(
        device: &Device,
        info: impl Into<ImageViewInfo>,
        image: vk::Image,
    ) -> Result<Self, DriverError> {
        let info = info.into();
        let device = device.clone();
        let create_info = vk::ImageViewCreateInfo::default()
            .view_type(info.ty)
            .format(info.format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            })
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: info.aspect_mask,
                base_array_layer: info.base_array_layer,
                base_mip_level: info.base_mip_level,
                level_count: info.mip_level_count,
                layer_count: info.array_layer_count,
            });

        let image_view =
            unsafe { device.create_image_view(&create_info, None) }.map_err(|err| {
                warn!("unable to create image view: {err}");

                DriverError::Unsupported
            })?;

        Ok(Self { device, image_view })
    }
}

impl Drop for ImageView {
    #[profiling::function]
    fn drop(&mut self) {
        if panicking() {
            return;
        }

        unsafe {
            self.device.destroy_image_view(self.image_view, None);
        }
    }
}

/// Information used to reinterpret an existing [`Image`] instance.
///
/// See [`VkImageViewCreateInfo`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageViewCreateInfo.html).
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[builder(
    build_fn(private, name = "fallible_build"),
    derive(Clone, Copy, Debug),
    pattern = "owned"
)]
pub struct ImageViewInfo {
    /// The number of layers that will be contained in the view.
    ///
    /// The default value is `vk::REMAINING_ARRAY_LAYERS`.
    #[builder(default = "vk::REMAINING_ARRAY_LAYERS")]
    pub array_layer_count: u32,

    /// The portion of the image that will be contained in the view.
    #[builder(default = "vk::ImageAspectFlags::COLOR")]
    pub aspect_mask: vk::ImageAspectFlags,

    /// The first array layer that will be contained in the view.
    #[builder(default)]
    pub base_array_layer: u32,

    /// The first mip level that will be contained in the view.
    #[builder(default)]
    pub base_mip_level: u32,

    /// The format and type of the texel blocks that will be contained in the view.
    #[builder(default = "vk::Format::UNDEFINED")]
    pub format: vk::Format,

    /// The number of mip levels that will be contained in the view.
    ///
    /// The default value is `vk::REMAINING_MIP_LEVELS`.
    #[builder(default = "vk::REMAINING_MIP_LEVELS")]
    pub mip_level_count: u32,

    /// The basic dimensionality of the view.
    #[builder(default = "vk::ImageViewType::TYPE_2D")]
    pub ty: vk::ImageViewType,
}

impl ImageViewInfo {
    /// Specifies a default view with the given `fmt` and `ty` values.
    ///
    /// # Note
    ///
    /// Automatically sets [`aspect_mask`](Self::aspect_mask) to a suggested value.
    #[inline(always)]
    pub const fn new(format: vk::Format, ty: vk::ImageViewType) -> ImageViewInfo {
        Self {
            array_layer_count: vk::REMAINING_ARRAY_LAYERS,
            aspect_mask: format_aspect_mask(format),
            base_array_layer: 0,
            base_mip_level: 0,
            format,
            mip_level_count: vk::REMAINING_MIP_LEVELS,
            ty,
        }
    }

    /// Converts an `ImageViewInfo` into an `ImageViewInfoBuilder`.
    pub fn into_builder(self) -> ImageViewInfoBuilder {
        ImageViewInfoBuilder {
            array_layer_count: Some(self.array_layer_count),
            aspect_mask: Some(self.aspect_mask),
            base_array_layer: Some(self.base_array_layer),
            base_mip_level: Some(self.base_mip_level),
            format: Some(self.format),
            mip_level_count: Some(self.mip_level_count),
            ty: Some(self.ty),
        }
    }
}

impl From<ImageInfo> for ImageViewInfo {
    fn from(info: ImageInfo) -> Self {
        Self::from_image_info(info).expect("unsupported image type for image view info")
    }
}

impl ImageViewInfo {
    /// Creates an image view description from image creation info.
    pub fn from_image_info(info: ImageInfo) -> Result<Self, DriverError> {
        Ok(Self {
            array_layer_count: info.array_layer_count,
            aspect_mask: format_aspect_mask(info.format),
            base_array_layer: 0,
            base_mip_level: 0,
            format: info.format,
            mip_level_count: info.mip_level_count,
            ty: match (info.ty, info.array_layer_count) {
                (vk::ImageType::TYPE_1D, 1) => vk::ImageViewType::TYPE_1D,
                (vk::ImageType::TYPE_1D, _) => vk::ImageViewType::TYPE_1D_ARRAY,
                (vk::ImageType::TYPE_2D, 1) => vk::ImageViewType::TYPE_2D,
                (vk::ImageType::TYPE_2D, 6)
                    if info.flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE) =>
                {
                    vk::ImageViewType::CUBE
                }
                (vk::ImageType::TYPE_2D, _)
                    if info.flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                        && info.array_layer_count > 6 =>
                {
                    vk::ImageViewType::CUBE_ARRAY
                }
                (vk::ImageType::TYPE_2D, _) => vk::ImageViewType::TYPE_2D_ARRAY,
                (vk::ImageType::TYPE_3D, _) => vk::ImageViewType::TYPE_3D,
                _ => {
                    warn!(
                        "invalid image view source info: image type {:?} with {} array layers",
                        info.ty, info.array_layer_count
                    );

                    return Err(DriverError::InvalidData);
                }
            },
        })
    }
}

impl From<ImageViewInfoBuilder> for ImageViewInfo {
    fn from(info: ImageViewInfoBuilder) -> Self {
        info.build()
    }
}

impl From<ImageViewInfo> for vk::ImageSubresourceRange {
    fn from(info: ImageViewInfo) -> Self {
        Self {
            aspect_mask: info.aspect_mask,
            base_mip_level: info.base_mip_level,
            base_array_layer: info.base_array_layer,
            layer_count: info.array_layer_count,
            level_count: info.mip_level_count,
        }
    }
}

impl ImageViewInfoBuilder {
    /// Builds a new `ImageViewInfo`.
    #[inline(always)]
    pub fn build(self) -> ImageViewInfo {
        self.fallible_build().expect("all fields have defaults")
    }
}

/// Specifies sample counts supported for an image used for storage operation.
///
/// Values must not exceed the device limits specified by the physical device properties.
///
/// See [`VkSampleCountFlagBits`](https://registry.khronos.org/vulkan/specs/latest/man/html/VkSampleCountFlagBits.html).
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum SampleCount {
    /// Single image sample. This is the usual mode.
    ///
    /// This is the default value.
    #[default]
    Type1,

    /// Multiple image samples.
    Type2,

    /// Multiple image samples.
    Type4,

    /// Multiple image samples.
    Type8,

    /// Multiple image samples.
    Type16,

    /// Multiple image samples.
    Type32,

    /// Multiple image samples.
    Type64,
}

impl SampleCount {
    /// Returns `true` when the value represents a single sample mode.
    pub fn is_single(self) -> bool {
        matches!(self, Self::Type1)
    }

    /// Returns `true` when the value represents a multiple sample mode.
    pub fn is_multiple(self) -> bool {
        matches!(
            self,
            Self::Type2 | Self::Type4 | Self::Type8 | Self::Type16 | Self::Type32 | Self::Type64
        )
    }
}

impl From<SampleCount> for vk::SampleCountFlags {
    fn from(sample_count: SampleCount) -> Self {
        match sample_count {
            SampleCount::Type1 => Self::TYPE_1,
            SampleCount::Type2 => Self::TYPE_2,
            SampleCount::Type4 => Self::TYPE_4,
            SampleCount::Type8 => Self::TYPE_8,
            SampleCount::Type16 => Self::TYPE_16,
            SampleCount::Type32 => Self::TYPE_32,
            SampleCount::Type64 => Self::TYPE_64,
        }
    }
}

#[derive(Debug)]
enum Sharing {
    Concurrent,
    Exclusive(ExclusiveSharing),
}

impl Sharing {
    fn new(info: ImageInfo, sharing_mode: vk::SharingMode) -> Self {
        if sharing_mode == vk::SharingMode::CONCURRENT {
            Self::Concurrent
        } else {
            Self::Exclusive(ExclusiveSharing::new(info))
        }
    }

    fn set_ranges(
        &self,
        dense: &Mutex<Option<DenseMap<SharingMode>>>,
        info: ImageInfo,
        sharing: SharingMode,
        sharing_ranges: &[vk::ImageSubresourceRange],
    ) {
        let Self::Exclusive(exclusive) = self else {
            return;
        };

        exclusive.set_ranges(dense, info, sharing, sharing_ranges);
    }
}

#[derive(Debug)]
struct UniformAccess(AtomicU8);

impl UniformAccess {
    fn new(access: AccessType) -> Self {
        Self(AtomicU8::new(access_type_into_u8(access)))
    }

    fn load(&self) -> AccessType {
        access_type_from_u8(self.0.load(Ordering::Acquire))
    }

    fn swap(
        &self,
        next_access: AccessType,
        access_range: vk::ImageSubresourceRange,
    ) -> (AccessType, vk::ImageSubresourceRange) {
        debug_assert_eq!(access_range.base_array_layer, 0);
        debug_assert_eq!(access_range.base_mip_level, 0);
        debug_assert_eq!(access_range.layer_count, 1);
        debug_assert_eq!(access_range.level_count, 1);
        debug_assert_eq!(access_range.aspect_mask.as_raw().count_ones(), 1);

        self.swap_range(next_access, access_range)
    }

    fn swap_range(
        &self,
        next_access: AccessType,
        access_range: vk::ImageSubresourceRange,
    ) -> (AccessType, vk::ImageSubresourceRange) {
        let prev_access = access_type_from_u8(
            self.0
                .swap(access_type_into_u8(next_access), Ordering::AcqRel),
        );

        (prev_access, access_range)
    }
}

#[doc(hidden)]
pub mod bench {
    use super::*;

    pub struct SwapAccessBenchHarness {
        access: Access,
        dense_access: Mutex<Option<DenseMap<AccessType>>>,
        info: ImageInfo,
    }

    impl SwapAccessBenchHarness {
        pub fn new(layers: u32, mips: u32, format: vk::Format) -> Self {
            let info = ImageInfo::image_2d(1, 1, format, vk::ImageUsageFlags::empty())
                .into_builder()
                .array_layer_count(layers)
                .mip_level_count(mips)
                .build();
            Self {
                access: Access::new(info, AccessType::Nothing),
                dense_access: Mutex::new(None),
                info,
            }
        }

        pub fn swap_access(
            &self,
            next_access: AccessType,
            mut access_range: vk::ImageSubresourceRange,
        ) -> Vec<(AccessType, vk::ImageSubresourceRange)> {
            #[cfg(feature = "checked")]
            {
                assert_aspect_mask_supported(access_range.aspect_mask);
                assert!(format_aspect_mask(self.info.format).contains(access_range.aspect_mask));
            }

            if access_range.layer_count == vk::REMAINING_ARRAY_LAYERS {
                debug_assert!(access_range.base_array_layer < self.info.array_layer_count);
                access_range.layer_count =
                    self.info.array_layer_count - access_range.base_array_layer;
            }

            debug_assert!(
                access_range.base_array_layer + access_range.layer_count
                    <= self.info.array_layer_count
            );

            if access_range.level_count == vk::REMAINING_MIP_LEVELS {
                debug_assert!(access_range.base_mip_level < self.info.mip_level_count);
                access_range.level_count = self.info.mip_level_count - access_range.base_mip_level;
            }

            debug_assert!(
                access_range.base_mip_level + access_range.level_count <= self.info.mip_level_count
            );

            self.access
                .swap(&self.dense_access, self.info, next_access, access_range)
                .collect()
        }
    }
}

#[cfg(test)]
mod test {
    use {
        super::*,
        rand::{Rng, SeedableRng, rngs::SmallRng},
        std::ops::Range,
    };

    // ImageSubresourceRange does not implement PartialEq
    fn assert_access_ranges_eq(
        lhs: (AccessType, vk::ImageSubresourceRange),
        rhs: (AccessType, vk::ImageSubresourceRange),
    ) {
        assert_eq!(
            (
                lhs.0,
                lhs.1.aspect_mask,
                lhs.1.base_array_layer,
                lhs.1.layer_count,
                lhs.1.base_mip_level,
                lhs.1.level_count
            ),
            (
                rhs.0,
                rhs.1.aspect_mask,
                rhs.1.base_array_layer,
                rhs.1.layer_count,
                rhs.1.base_mip_level,
                rhs.1.level_count
            )
        );
    }

    fn image_sync_subresource(
        aspect_mask: vk::ImageAspectFlags,
        array_layers: Range<u32>,
        mip_levels: Range<u32>,
    ) -> ImageSubresourceSyncInfo {
        ImageSubresourceSyncInfo {
            access_mask: vk::AccessFlags::SHADER_READ,
            layout: Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            queue_family_index: None,
            range: image_subresource_range(aspect_mask, array_layers, mip_levels),
            stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
        }
    }

    #[test]
    pub fn image_access_basic() {
        use vk::ImageAspectFlags as A;

        let mut image = DenseMap::new(
            image_subresource(vk::Format::R8G8B8A8_UNORM, 1, 1),
            AccessType::Nothing,
        );

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderWrite,
                image_subresource_range(A::COLOR, 0..1, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::COLOR, 0..1, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderReadOther,
                image_subresource_range(A::COLOR, 0..1, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::COLOR, 0..1, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }
    }

    #[test]
    pub fn image_access_uniform() {
        use vk::ImageAspectFlags as A;

        let info = image_subresource(vk::Format::R8G8B8A8_UNORM, 1, 1);
        let image = Access::new(info, AccessType::Nothing);
        let dense = Mutex::new(None);

        let mut accesses = image.swap(
            &dense,
            info,
            AccessType::AnyShaderWrite,
            image_subresource_range(A::COLOR, 0..1, 0..1),
        );

        assert_access_ranges_eq(
            accesses.next().unwrap(),
            (
                AccessType::Nothing,
                image_subresource_range(A::COLOR, 0..1, 0..1),
            ),
        );
        assert!(accesses.next().is_none());
    }

    #[test]
    pub fn image_access_dual_aspect_tracks_aspects_independently() {
        use vk::ImageAspectFlags as A;

        let info = image_subresource(vk::Format::D32_SFLOAT_S8_UINT, 1, 1);
        let image = Access::new(info, AccessType::Nothing);
        let dense = Mutex::new(None);

        let mut accesses = image.swap(
            &dense,
            info,
            AccessType::DepthStencilAttachmentWrite,
            image_subresource_range(A::DEPTH, 0..1, 0..1),
        );

        assert_access_ranges_eq(
            accesses.next().unwrap(),
            (
                AccessType::Nothing,
                image_subresource_range(A::DEPTH, 0..1, 0..1),
            ),
        );
        assert!(accesses.next().is_none());

        let mut accesses = image.swap(
            &dense,
            info,
            AccessType::DepthStencilAttachmentRead,
            image_subresource_range(A::STENCIL, 0..1, 0..1),
        );

        assert_access_ranges_eq(
            accesses.next().unwrap(),
            (
                AccessType::Nothing,
                image_subresource_range(A::STENCIL, 0..1, 0..1),
            ),
        );
        assert!(accesses.next().is_none());

        let mut accesses = image.swap(
            &dense,
            info,
            AccessType::AnyShaderReadOther,
            image_subresource_range(A::DEPTH | A::STENCIL, 0..1, 0..1),
        );

        assert_access_ranges_eq(
            accesses.next().unwrap(),
            (
                AccessType::DepthStencilAttachmentWrite,
                image_subresource_range(A::DEPTH, 0..1, 0..1),
            ),
        );
        assert_access_ranges_eq(
            accesses.next().unwrap(),
            (
                AccessType::DepthStencilAttachmentRead,
                image_subresource_range(A::STENCIL, 0..1, 0..1),
            ),
        );
        assert!(accesses.next().is_none());
    }

    #[test]
    pub fn image_access_dense_promotes_only_on_partial_update() {
        use vk::ImageAspectFlags as A;

        let info = image_subresource(vk::Format::R8_UINT, 2, 2);
        let image = Access::new(info, AccessType::Nothing);
        let dense = Mutex::new(None);

        let Access::Dense(access) = &image else {
            panic!("expected dense-capable access tracking");
        };

        let mut accesses = image.swap(
            &dense,
            info,
            AccessType::AnyShaderReadOther,
            image_subresource_range(A::COLOR, 0..2, 0..2),
        );

        assert_access_ranges_eq(
            accesses.next().unwrap(),
            (
                AccessType::Nothing,
                image_subresource_range(A::COLOR, 0..2, 0..2),
            ),
        );
        assert!(accesses.next().is_none());
        assert!(!access.is_dense_active());

        let mut accesses = image.swap(
            &dense,
            info,
            AccessType::AnyShaderWrite,
            image_subresource_range(A::COLOR, 0..1, 0..1),
        );

        assert_access_ranges_eq(
            accesses.next().unwrap(),
            (
                AccessType::AnyShaderReadOther,
                image_subresource_range(A::COLOR, 0..1, 0..1),
            ),
        );
        assert!(accesses.next().is_none());
        assert!(access.is_dense_active());
    }

    #[test]
    pub fn image_access_dense_collapses_to_uniform_after_equalizing_updates() {
        use vk::ImageAspectFlags as A;

        let info = image_subresource(vk::Format::R8_UINT, 2, 2);
        let image = Access::new(info, AccessType::Nothing);
        let dense = Mutex::new(None);

        let Access::Dense(access) = &image else {
            panic!("expected dense-capable access tracking");
        };

        {
            let mut accesses = image.swap(
                &dense,
                info,
                AccessType::AnyShaderReadOther,
                image_subresource_range(A::COLOR, 0..1, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::COLOR, 0..1, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }

        assert!(access.is_dense_active());

        {
            let mut accesses = image.swap(
                &dense,
                info,
                AccessType::AnyShaderReadOther,
                image_subresource_range(A::COLOR, 0..2, 0..2),
            );

            assert!(accesses.next().is_some());
            while accesses.next().is_some() {}
        }

        assert!(!access.is_dense_active());
        assert_eq!(access.load(), AccessType::AnyShaderReadOther);

        let dense = dense.lock();
        #[cfg(not(feature = "parking_lot"))]
        let dense = dense.expect("poisoned image dense lock");

        assert!(dense.is_none());
    }

    #[test]
    pub fn image_access_dense_stays_active_for_mixed_updates() {
        use vk::ImageAspectFlags as A;

        let info = image_subresource(vk::Format::R8_UINT, 2, 2);
        let image = Access::new(info, AccessType::Nothing);
        let dense = Mutex::new(None);

        let Access::Dense(access) = &image else {
            panic!("expected dense-capable access tracking");
        };

        {
            let mut accesses = image.swap(
                &dense,
                info,
                AccessType::AnyShaderReadOther,
                image_subresource_range(A::COLOR, 0..1, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::COLOR, 0..1, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = image.swap(
                &dense,
                info,
                AccessType::AnyShaderWrite,
                image_subresource_range(A::COLOR, 1..2, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::COLOR, 1..2, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }

        assert!(access.is_dense_active());

        let dense = dense.lock();
        #[cfg(not(feature = "parking_lot"))]
        let dense = dense.expect("poisoned image dense lock");

        let dense_map = dense.as_ref().expect("missing dense access map");
        assert_eq!(
            dense_map.subresource(0, 0, 0),
            AccessType::AnyShaderReadOther
        );
        assert_eq!(dense_map.subresource(0, 1, 0), AccessType::AnyShaderWrite);
        assert_eq!(dense_map.subresource(0, 0, 1), AccessType::Nothing);
        assert_eq!(dense_map.subresource(0, 1, 1), AccessType::Nothing);
    }

    #[test]
    pub fn image_access_dense_iter_drains_on_drop() {
        use vk::ImageAspectFlags as A;

        let info = image_subresource(vk::Format::R8_UINT, 2, 2);
        let image = Access::new(info, AccessType::Nothing);
        let dense = Mutex::new(None);

        let Access::Dense(access) = &image else {
            panic!("expected dense-capable access tracking");
        };

        {
            let mut accesses = image.swap(
                &dense,
                info,
                AccessType::AnyShaderReadOther,
                image_subresource_range(A::COLOR, 0..1, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::COLOR, 0..1, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = image.swap(
                &dense,
                info,
                AccessType::AnyShaderWrite,
                image_subresource_range(A::COLOR, 1..2, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::COLOR, 1..2, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }

        let mut accesses = image.swap(
            &dense,
            info,
            AccessType::HostRead,
            image_subresource_range(A::COLOR, 0..2, 0..2),
        );

        assert!(accesses.next().is_some());
        drop(accesses);

        assert!(!access.is_dense_active());
        assert_eq!(access.load(), AccessType::HostRead);

        let dense = dense.lock();
        #[cfg(not(feature = "parking_lot"))]
        let dense = dense.expect("poisoned image dense lock");

        assert!(dense.is_none());
    }

    #[test]
    pub fn image_access_color() {
        use vk::ImageAspectFlags as A;

        let mut image = DenseMap::new(
            image_subresource(vk::Format::R8G8B8A8_UNORM, 3, 3),
            AccessType::Nothing,
        );

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderWrite,
                image_subresource_range(A::COLOR, 0..3, 0..3),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::COLOR, 0..3, 0..3),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderReadOther,
                image_subresource_range(A::COLOR, 0..1, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::COLOR, 0..1, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::ComputeShaderWrite,
                image_subresource_range(A::COLOR, 0..3, 0..3),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderReadOther,
                    image_subresource_range(A::COLOR, 0..1, 0..1),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::COLOR, 0..1, 1..3),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::COLOR, 1..3, 0..3),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::HostRead,
                image_subresource_range(A::COLOR, 0..3, 0..3),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::ComputeShaderWrite,
                    image_subresource_range(A::COLOR, 0..3, 0..3),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::HostWrite,
                image_subresource_range(A::COLOR, 1..2, 1..2),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::HostRead,
                    image_subresource_range(A::COLOR, 1..2, 1..2),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::GeometryShaderReadOther,
                image_subresource_range(A::COLOR, 0..3, 0..3),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::HostRead,
                    image_subresource_range(A::COLOR, 0..1, 0..3),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::HostRead,
                    image_subresource_range(A::COLOR, 1..2, 0..1),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::HostWrite,
                    image_subresource_range(A::COLOR, 1..2, 1..2),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::HostRead,
                    image_subresource_range(A::COLOR, 1..2, 2..3),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::HostRead,
                    image_subresource_range(A::COLOR, 2..3, 0..3),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::VertexBuffer,
                image_subresource_range(A::COLOR, 0..3, 1..2),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::GeometryShaderReadOther,
                    image_subresource_range(A::COLOR, 0..3, 1..2),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::ColorAttachmentRead,
                image_subresource_range(A::COLOR, 0..3, 0..3),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::GeometryShaderReadOther,
                    image_subresource_range(A::COLOR, 0..1, 0..1),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::VertexBuffer,
                    image_subresource_range(A::COLOR, 0..1, 1..2),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::GeometryShaderReadOther,
                    image_subresource_range(A::COLOR, 0..1, 2..3),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::GeometryShaderReadOther,
                    image_subresource_range(A::COLOR, 1..2, 0..1),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::VertexBuffer,
                    image_subresource_range(A::COLOR, 1..2, 1..2),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::GeometryShaderReadOther,
                    image_subresource_range(A::COLOR, 1..2, 2..3),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::GeometryShaderReadOther,
                    image_subresource_range(A::COLOR, 2..3, 0..1),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::VertexBuffer,
                    image_subresource_range(A::COLOR, 2..3, 1..2),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::GeometryShaderReadOther,
                    image_subresource_range(A::COLOR, 2..3, 2..3),
                ),
            );
            assert!(accesses.next().is_none());
        }
    }

    #[test]
    pub fn image_access_layers() {
        use vk::ImageAspectFlags as A;

        let mut image = DenseMap::new(
            image_subresource(vk::Format::R8G8B8A8_UNORM, 3, 1),
            AccessType::Nothing,
        );

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderWrite,
                image_subresource_range(A::COLOR, 0..3, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::COLOR, 0..3, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderReadOther,
                image_subresource_range(A::COLOR, 2..3, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::COLOR, 2..3, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::HostRead,
                image_subresource_range(A::COLOR, 0..2, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::COLOR, 0..2, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderReadOther,
                image_subresource_range(A::COLOR, 0..1, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::HostRead,
                    image_subresource_range(A::COLOR, 0..1, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderReadOther,
                image_subresource_range(A::COLOR, 1..2, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::HostRead,
                    image_subresource_range(A::COLOR, 1..2, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::HostWrite,
                image_subresource_range(A::COLOR, 0..3, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderReadOther,
                    image_subresource_range(A::COLOR, 0..3, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }
    }

    #[test]
    pub fn image_access_levels() {
        use vk::ImageAspectFlags as A;

        let mut image = DenseMap::new(
            image_subresource(vk::Format::R8G8B8A8_UNORM, 1, 3),
            AccessType::Nothing,
        );

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderWrite,
                image_subresource_range(A::COLOR, 0..1, 0..3),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::COLOR, 0..1, 0..3),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderReadOther,
                image_subresource_range(A::COLOR, 0..1, 2..3),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::COLOR, 0..1, 2..3),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::HostRead,
                image_subresource_range(A::COLOR, 0..1, 0..2),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::COLOR, 0..1, 0..2),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderReadOther,
                image_subresource_range(A::COLOR, 0..1, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::HostRead,
                    image_subresource_range(A::COLOR, 0..1, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderReadOther,
                image_subresource_range(A::COLOR, 0..1, 1..2),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::HostRead,
                    image_subresource_range(A::COLOR, 0..1, 1..2),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::HostWrite,
                image_subresource_range(A::COLOR, 0..1, 0..3),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderReadOther,
                    image_subresource_range(A::COLOR, 0..1, 0..3),
                ),
            );
            assert!(accesses.next().is_none());
        }
    }

    #[test]
    pub fn image_access_depth_stencil() {
        use vk::ImageAspectFlags as A;

        let mut image = DenseMap::new(
            image_subresource(vk::Format::D24_UNORM_S8_UINT, 4, 3),
            AccessType::Nothing,
        );

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderWrite,
                image_subresource_range(A::DEPTH, 0..4, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::DEPTH, 0..4, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderWrite,
                image_subresource_range(A::STENCIL, 0..4, 1..2),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::STENCIL, 0..4, 1..2),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderReadOther,
                image_subresource_range(A::DEPTH | A::STENCIL, 0..4, 0..2),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::DEPTH, 0..1, 0..1),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::DEPTH, 0..1, 1..2),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::DEPTH, 1..2, 0..1),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::DEPTH, 1..2, 1..2),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::DEPTH, 2..3, 0..1),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::DEPTH, 2..3, 1..2),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::DEPTH, 3..4, 0..1),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::DEPTH, 3..4, 1..2),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::STENCIL, 0..1, 0..1),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::STENCIL, 0..1, 1..2),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::STENCIL, 1..2, 0..1),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::STENCIL, 1..2, 1..2),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::STENCIL, 2..3, 0..1),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::STENCIL, 2..3, 1..2),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::STENCIL, 3..4, 0..1),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::STENCIL, 3..4, 1..2),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AccelerationStructureBuildWrite,
                image_subresource_range(A::DEPTH | A::STENCIL, 0..4, 0..2),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderReadOther,
                    image_subresource_range(A::DEPTH | A::STENCIL, 0..4, 0..2),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AccelerationStructureBuildRead,
                image_subresource_range(A::DEPTH, 1..3, 0..2),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AccelerationStructureBuildWrite,
                    image_subresource_range(A::DEPTH, 1..3, 0..2),
                ),
            );
            assert!(accesses.next().is_none());
        }
    }

    #[test]
    pub fn image_access_stencil() {
        use vk::ImageAspectFlags as A;

        let mut image = DenseMap::new(
            image_subresource(vk::Format::S8_UINT, 2, 2),
            AccessType::Nothing,
        );

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderWrite,
                image_subresource_range(A::STENCIL, 0..2, 0..1),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::STENCIL, 0..2, 0..1),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::AnyShaderReadOther,
                image_subresource_range(A::STENCIL, 0..2, 1..2),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::Nothing,
                    image_subresource_range(A::STENCIL, 0..2, 1..2),
                ),
            );
            assert!(accesses.next().is_none());
        }

        {
            let mut accesses = DenseMapIter::new(
                &mut image,
                AccessType::HostRead,
                image_subresource_range(A::STENCIL, 0..2, 0..2),
            );

            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::STENCIL, 0..1, 0..1),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderReadOther,
                    image_subresource_range(A::STENCIL, 0..1, 1..2),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderWrite,
                    image_subresource_range(A::STENCIL, 1..2, 0..1),
                ),
            );
            assert_access_ranges_eq(
                accesses.next().unwrap(),
                (
                    AccessType::AnyShaderReadOther,
                    image_subresource_range(A::STENCIL, 1..2, 1..2),
                ),
            );
            assert!(accesses.next().is_none());
        }
    }

    #[test]
    pub fn image_info_cube() {
        let info = ImageInfo::cube(42, vk::Format::R32_SFLOAT, vk::ImageUsageFlags::empty());
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn image_info_cube_builder() {
        let info = ImageInfo::cube(42, vk::Format::R32_SFLOAT, vk::ImageUsageFlags::empty());
        let builder = ImageInfoBuilder::default()
            .ty(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32_SFLOAT)
            .width(42)
            .height(42)
            .depth(1)
            .array_layer_count(6)
            .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
            .build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn image_info_image_1d() {
        let info = ImageInfo::image_1d(42, vk::Format::R32_SFLOAT, vk::ImageUsageFlags::empty());
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn image_info_image_1d_builder() {
        let info = ImageInfo::image_1d(42, vk::Format::R32_SFLOAT, vk::ImageUsageFlags::empty());
        let builder = ImageInfoBuilder::default()
            .ty(vk::ImageType::TYPE_1D)
            .format(vk::Format::R32_SFLOAT)
            .width(42)
            .height(1)
            .depth(1)
            .build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn image_info_image_2d() {
        let info =
            ImageInfo::image_2d(42, 84, vk::Format::R32_SFLOAT, vk::ImageUsageFlags::empty());
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn image_info_image_2d_builder() {
        let info =
            ImageInfo::image_2d(42, 84, vk::Format::R32_SFLOAT, vk::ImageUsageFlags::empty());
        let builder = ImageInfoBuilder::default()
            .ty(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32_SFLOAT)
            .width(42)
            .height(84)
            .depth(1)
            .build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn image_info_image_2d_array() {
        let info = ImageInfo::image_2d_array(
            42,
            84,
            100,
            vk::Format::default(),
            vk::ImageUsageFlags::empty(),
        );
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn image_info_image_2d_array_builder() {
        let info = ImageInfo::image_2d_array(
            42,
            84,
            100,
            vk::Format::R32_SFLOAT,
            vk::ImageUsageFlags::empty(),
        );
        let builder = ImageInfoBuilder::default()
            .ty(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32_SFLOAT)
            .width(42)
            .height(84)
            .depth(1)
            .array_layer_count(100)
            .build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn image_info_image_3d() {
        let info = ImageInfo::image_3d(
            42,
            84,
            100,
            vk::Format::R32_SFLOAT,
            vk::ImageUsageFlags::empty(),
        );
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn image_info_image_3d_builder() {
        let info = ImageInfo::image_3d(
            42,
            84,
            100,
            vk::Format::R32_SFLOAT,
            vk::ImageUsageFlags::empty(),
        );
        let builder = ImageInfoBuilder::default()
            .ty(vk::ImageType::TYPE_3D)
            .format(vk::Format::R32_SFLOAT)
            .width(42)
            .height(84)
            .depth(100)
            .build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn image_info_builder_defaults() {
        let info = ImageInfo {
            array_layer_count: 1,
            alloc_dedicated: false,
            depth: 0,
            flags: vk::ImageCreateFlags::empty(),
            format: vk::Format::UNDEFINED,
            height: 0,
            host_readable: false,
            host_writable: false,
            mip_level_count: 1,
            sample_count: SampleCount::Type1,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            tiling: vk::ImageTiling::OPTIMAL,
            ty: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::empty(),
            width: 0,
        };

        assert_eq!(ImageInfoBuilder::default().build(), info);
    }

    fn image_access_fuzz(aspect_count: u8, array_layer_count: u32, mip_level_count: u32) {
        const FUZZ_COUNT: usize = 100_000;
        static ACCESS_TYPES: &[AccessType] = &[
            AccessType::AnyShaderReadOther,
            AccessType::AnyShaderWrite,
            AccessType::ColorAttachmentRead,
            AccessType::ColorAttachmentWrite,
            AccessType::HostRead,
            AccessType::HostWrite,
            AccessType::Nothing,
        ];

        let fmt = match aspect_count {
            1 => vk::Format::R8G8B8A8_UNORM,
            2 => vk::Format::D24_UNORM_S8_UINT,
            _ => unreachable!(),
        };

        let mut rng = SmallRng::seed_from_u64(42);
        let total = (aspect_count as u32 * array_layer_count * mip_level_count) as usize;
        let mut access_map = DenseMap::new(
            image_subresource(fmt, array_layer_count, mip_level_count),
            AccessType::Nothing,
        );
        let mut data = vec![AccessType::Nothing; total];

        let aspect_bits = format_aspect_mask(fmt);

        for _ in 0..FUZZ_COUNT {
            let new_access = ACCESS_TYPES[rng.random_range(..ACCESS_TYPES.len())];

            // Pick a valid aspect mask from the format's supported aspects
            let aspect_mask = if aspect_count == 2 && rng.random_bool(0.5) {
                aspect_bits
            } else {
                let bit_index =
                    rng.random_range(..aspect_count) + aspect_bits.as_raw().trailing_zeros() as u8;
                vk::ImageAspectFlags::from_raw(1 << bit_index)
            };

            let layer_start = rng.random_range(..array_layer_count);
            let layer_end = rng.random_range(layer_start + 1..=array_layer_count);
            let mip_start = rng.random_range(..mip_level_count);
            let mip_end = rng.random_range(mip_start + 1..=mip_level_count);

            let range =
                image_subresource_range(aspect_mask, layer_start..layer_end, mip_start..mip_end);

            for (prev, range) in access_map.swap(new_access, range) {
                let range_mask = range.aspect_mask.as_raw();
                for ai in 0..range_mask.count_ones() as u8 {
                    let bit = range_mask.trailing_zeros() + ai as u32;
                    let a = (aspect_bits.as_raw() & ((1 << bit) - 1)).count_ones() as u8;
                    for l in range.base_array_layer..range.base_array_layer + range.layer_count {
                        for m in range.base_mip_level..range.base_mip_level + range.level_count {
                            let idx = (l * aspect_count as u32 * mip_level_count
                                + m * aspect_count as u32
                                + a as u32) as usize;
                            assert_eq!(
                                data[idx], prev,
                                "prev mismatch at aspect={a} layer={l} mip={m} idx={idx}: expected {prev:?}, got {:?}",
                                data[idx],
                            );
                        }
                    }
                }
            }

            for a in 0..aspect_count {
                let bit = aspect_bits.as_raw().trailing_zeros() as u8 + a;
                if aspect_mask.as_raw() & (1 << bit) == 0 {
                    continue;
                }
                for l in layer_start..layer_end {
                    for m in mip_start..mip_end {
                        let idx = access_map.idx(a, l, m);
                        data[idx] = new_access;
                    }
                }
            }
        }
    }

    #[test]
    pub fn image_access_fuzz_small() {
        image_access_fuzz(1, 3, 3);
    }

    #[test]
    pub fn image_access_fuzz_medium() {
        image_access_fuzz(2, 4, 3);
    }

    #[test]
    pub fn image_access_fuzz_large() {
        image_access_fuzz(1, 10, 10);
    }

    fn image_access_fuzz_through_access(
        aspect_count: u8,
        array_layer_count: u32,
        mip_level_count: u32,
    ) {
        const FUZZ_COUNT: usize = 10_000;
        static ACCESS_TYPES: &[AccessType] = &[
            AccessType::AnyShaderReadOther,
            AccessType::AnyShaderWrite,
            AccessType::ColorAttachmentRead,
            AccessType::ColorAttachmentWrite,
            AccessType::HostRead,
            AccessType::HostWrite,
            AccessType::Nothing,
        ];

        let fmt = match aspect_count {
            1 => vk::Format::R8G8B8A8_UNORM,
            2 => vk::Format::D24_UNORM_S8_UINT,
            _ => unreachable!(),
        };

        let mut rng = SmallRng::seed_from_u64(42);
        let info = image_subresource(fmt, array_layer_count, mip_level_count);
        let total = (aspect_count as u32 * array_layer_count * mip_level_count) as usize;
        let access = Access::new(info, AccessType::Nothing);
        let dense = Mutex::new(None);
        let mut data = vec![AccessType::Nothing; total];

        let aspect_bits = format_aspect_mask(fmt);

        for _ in 0..FUZZ_COUNT {
            let new_access = ACCESS_TYPES[rng.random_range(..ACCESS_TYPES.len())];

            let aspect_mask = if aspect_count == 2 && rng.random_bool(0.5) {
                aspect_bits
            } else {
                let bit_index =
                    rng.random_range(..aspect_count) + aspect_bits.as_raw().trailing_zeros() as u8;
                vk::ImageAspectFlags::from_raw(1 << bit_index)
            };

            let layer_start = rng.random_range(..array_layer_count);
            let layer_end = rng.random_range(layer_start + 1..=array_layer_count);
            let mip_start = rng.random_range(..mip_level_count);
            let mip_end = rng.random_range(mip_start + 1..=mip_level_count);

            let range =
                image_subresource_range(aspect_mask, layer_start..layer_end, mip_start..mip_end);
            let resolved = info.resolve_subresource_counts(range);

            for (prev, returned_range) in access.swap(&dense, info, new_access, resolved) {
                let range_mask = returned_range.aspect_mask.as_raw();
                for ai in 0..range_mask.count_ones() as u8 {
                    let bit = range_mask.trailing_zeros() + ai as u32;
                    let a = (aspect_bits.as_raw() & ((1 << bit) - 1)).count_ones() as u8;
                    for l in returned_range.base_array_layer
                        ..returned_range.base_array_layer + returned_range.layer_count
                    {
                        for m in returned_range.base_mip_level
                            ..returned_range.base_mip_level + returned_range.level_count
                        {
                            let idx = (l * aspect_count as u32 * mip_level_count
                                + m * aspect_count as u32
                                + a as u32) as usize;
                            assert_eq!(
                                data[idx], prev,
                                "prev mismatch at aspect={a} layer={l} mip={m} idx={idx}: expected {prev:?}, got {:?}",
                                data[idx],
                            );
                        }
                    }
                }
            }

            for a in 0..aspect_count {
                let bit = aspect_bits.as_raw().trailing_zeros() as u8 + a;
                if aspect_mask.as_raw() & (1 << bit) == 0 {
                    continue;
                }
                for l in layer_start..layer_end {
                    for m in mip_start..mip_end {
                        let idx = (l * aspect_count as u32 * mip_level_count
                            + m * aspect_count as u32
                            + a as u32) as usize;
                        data[idx] = new_access;
                    }
                }
            }
        }
    }

    #[test]
    pub fn image_access_fuzz_access_uniform() {
        image_access_fuzz_through_access(1, 1, 1);
    }

    #[test]
    pub fn image_access_fuzz_access_dual_aspect() {
        image_access_fuzz_through_access(2, 1, 1);
    }

    #[test]
    pub fn image_access_fuzz_access_dense_small() {
        image_access_fuzz_through_access(1, 4, 4);
    }

    #[test]
    pub fn image_access_fuzz_access_dense_large() {
        image_access_fuzz_through_access(1, 8, 8);
    }

    #[test]
    pub fn image_access_fuzz_access_dense_dual_aspect() {
        image_access_fuzz_through_access(2, 3, 3);
    }

    #[test]
    pub fn image_sync_info_compact_merges_mips_then_layers() {
        use vk::ImageAspectFlags as A;

        let mut sync_info = ImageSyncInfo {
            subresources: vec![
                image_sync_subresource(A::COLOR, 0..1, 0..1),
                image_sync_subresource(A::COLOR, 0..1, 1..2),
                image_sync_subresource(A::COLOR, 1..2, 0..1),
                image_sync_subresource(A::COLOR, 1..2, 1..2),
            ]
            .into_boxed_slice(),
        };

        sync_info.compact();

        assert_eq!(sync_info.subresources.len(), 1);
        let subresource = &sync_info.subresources[0];
        let range = image_subresource_range(A::COLOR, 0..2, 0..2);
        assert_eq!(subresource.access_mask, vk::AccessFlags::SHADER_READ);
        assert_eq!(
            subresource.layout,
            Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        );
        assert_eq!(subresource.range.aspect_mask, range.aspect_mask);
        assert_eq!(subresource.range.base_array_layer, range.base_array_layer);
        assert_eq!(subresource.range.layer_count, range.layer_count);
        assert_eq!(subresource.range.base_mip_level, range.base_mip_level);
        assert_eq!(subresource.range.level_count, range.level_count);
        assert_eq!(
            subresource.stage_mask,
            vk::PipelineStageFlags::COMPUTE_SHADER
        );
    }

    #[test]
    pub fn image_sync_info_compact_keeps_different_sync_separate() {
        use vk::ImageAspectFlags as A;

        let sync_info = ImageSyncInfo {
            subresources: vec![
                image_sync_subresource(A::COLOR, 0..1, 0..1),
                ImageSubresourceSyncInfo {
                    access_mask: vk::AccessFlags::SHADER_WRITE,
                    layout: Some(vk::ImageLayout::GENERAL),
                    queue_family_index: None,
                    range: image_subresource_range(A::COLOR, 0..1, 1..2),
                    stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
                },
            ]
            .into_boxed_slice(),
        };

        let sync_info = sync_info.into_compacted();

        assert_eq!(sync_info.subresources.len(), 2);
        let subresource = &sync_info.subresources[0];
        let range = image_subresource_range(A::COLOR, 0..1, 0..1);
        assert_eq!(subresource.access_mask, vk::AccessFlags::SHADER_READ);
        assert_eq!(
            subresource.layout,
            Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        );
        assert_eq!(subresource.range.aspect_mask, range.aspect_mask);
        assert_eq!(subresource.range.base_array_layer, range.base_array_layer);
        assert_eq!(subresource.range.layer_count, range.layer_count);
        assert_eq!(subresource.range.base_mip_level, range.base_mip_level);
        assert_eq!(subresource.range.level_count, range.level_count);
        assert_eq!(
            subresource.stage_mask,
            vk::PipelineStageFlags::COMPUTE_SHADER
        );
        assert_eq!(
            sync_info.subresources[1].access_mask,
            vk::AccessFlags::SHADER_WRITE
        );
        assert_eq!(
            sync_info.subresources[1].layout,
            Some(vk::ImageLayout::GENERAL)
        );
    }

    #[test]
    pub fn image_sync_info_compact_keeps_different_queue_families_separate() {
        use vk::ImageAspectFlags as A;

        let sync_info = ImageSyncInfo {
            subresources: vec![
                ImageSubresourceSyncInfo {
                    access_mask: vk::AccessFlags::SHADER_READ,
                    layout: Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    queue_family_index: Some(1),
                    range: image_subresource_range(A::COLOR, 0..1, 0..1),
                    stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
                },
                ImageSubresourceSyncInfo {
                    access_mask: vk::AccessFlags::SHADER_READ,
                    layout: Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    queue_family_index: Some(2),
                    range: image_subresource_range(A::COLOR, 0..1, 1..2),
                    stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
                },
            ]
            .into_boxed_slice(),
        };

        let sync_info = sync_info.into_compacted();

        assert_eq!(sync_info.subresources.len(), 2);
        assert_eq!(sync_info.subresources[0].queue_family_index, Some(1));
        assert_eq!(sync_info.subresources[1].queue_family_index, Some(2));
    }

    #[test]
    pub fn image_ownership_set_promotes_dense_on_partial_update() {
        use vk::ImageAspectFlags as A;

        let info = image_subresource(vk::Format::R8_UINT, 2, 2);
        let sharing = Sharing::new(info, vk::SharingMode::EXCLUSIVE);
        let dense = Mutex::new(None);

        sharing.set_ranges(
            &dense,
            info,
            SharingMode::Exclusive(Some((7, 3))),
            &[image_subresource_range(A::COLOR, 0..1, 0..1)],
        );

        match &sharing {
            Sharing::Exclusive(exclusive) => {
                assert!(exclusive.is_dense_sharing_active());
            }
            Sharing::Concurrent => panic!("expected exclusive ownership"),
        }

        let dense = dense.lock();

        #[cfg(not(feature = "parking_lot"))]
        let dense = dense.expect("poisoned image dense lock");

        let dense = dense.as_ref().expect("missing dense sharing state");
        assert_eq!(
            dense.subresource(0, 0, 0),
            SharingMode::Exclusive(Some((7, 3)))
        );
        assert_eq!(dense.subresource(0, 1, 0), SharingMode::Exclusive(None));
        assert_eq!(dense.subresource(0, 0, 1), SharingMode::Exclusive(None));
        assert_eq!(dense.subresource(0, 1, 1), SharingMode::Exclusive(None));
    }

    #[test]
    pub fn image_ownership_set_whole_image_stays_uniform() {
        use vk::ImageAspectFlags as A;

        let info = image_subresource(vk::Format::R8_UINT, 2, 2);
        let sharing = Sharing::new(info, vk::SharingMode::EXCLUSIVE);
        let dense = Mutex::new(None);

        sharing.set_ranges(
            &dense,
            info,
            SharingMode::Exclusive(Some((1, 2))),
            &[image_subresource_range(A::COLOR, 0..2, 0..2)],
        );

        match &sharing {
            Sharing::Exclusive(exclusive) => {
                assert!(!exclusive.is_dense_sharing_active());
                assert_eq!(
                    SharingMode::decode(exclusive.uniform.load(Ordering::Acquire)),
                    SharingMode::Exclusive(Some((1, 2)))
                );
            }
            Sharing::Concurrent => panic!("expected exclusive ownership"),
        }
    }

    fn image_subresource(
        format: vk::Format,
        array_layer_count: u32,
        mip_level_count: u32,
    ) -> ImageInfo {
        ImageInfo::image_2d(1, 1, format, vk::ImageUsageFlags::empty())
            .into_builder()
            .array_layer_count(array_layer_count)
            .mip_level_count(mip_level_count)
            .build()
    }

    fn image_subresource_range(
        aspect_mask: vk::ImageAspectFlags,
        array_layers: Range<u32>,
        mip_levels: Range<u32>,
    ) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask,
            base_array_layer: array_layers.start,
            base_mip_level: mip_levels.start,
            layer_count: array_layers.len() as _,
            level_count: mip_levels.len() as _,
        }
    }

    #[test]
    pub fn image_subresource_range_contains() {
        use {
            super::image_subresource_range_contains as f, image_subresource_range as i,
            vk::ImageAspectFlags as A,
        };

        assert!(f(i(A::COLOR, 0..1, 0..1), i(A::COLOR, 0..1, 0..1)));
        assert!(f(i(A::COLOR, 0..2, 0..1), i(A::COLOR, 0..1, 0..1)));
        assert!(f(i(A::COLOR, 0..1, 0..2), i(A::COLOR, 0..1, 0..1)));
        assert!(f(i(A::COLOR, 0..2, 0..2), i(A::COLOR, 0..1, 0..1)));
        assert!(!f(i(A::COLOR, 0..1, 1..3), i(A::COLOR, 0..1, 0..1)));
        assert!(!f(i(A::COLOR, 1..3, 0..1), i(A::COLOR, 0..1, 0..1)));
        assert!(!f(i(A::COLOR, 0..1, 1..3), i(A::COLOR, 0..1, 0..2)));
        assert!(!f(i(A::COLOR, 1..3, 0..1), i(A::COLOR, 0..2, 0..1)));
    }

    #[test]
    pub fn image_subresource_range_intersects() {
        use {
            super::image_subresource_range_intersects as f, image_subresource_range as i,
            vk::ImageAspectFlags as A,
        };

        assert!(f(i(A::COLOR, 0..1, 0..1), i(A::COLOR, 0..1, 0..1)));
        assert!(!f(i(A::COLOR, 0..1, 0..1), i(A::DEPTH, 0..1, 0..1)));

        assert!(!f(i(A::COLOR, 0..1, 0..1), i(A::COLOR, 1..2, 0..1)));
        assert!(!f(i(A::COLOR, 0..1, 0..1), i(A::COLOR, 0..1, 1..2)));
        assert!(!f(i(A::COLOR, 0..1, 0..1), i(A::DEPTH, 1..2, 0..1)));
        assert!(!f(i(A::COLOR, 0..1, 0..1), i(A::DEPTH, 0..1, 1..2)));
        assert!(!f(i(A::COLOR, 1..2, 1..2), i(A::COLOR, 0..1, 0..1)));

        assert!(f(
            i(A::DEPTH | A::STENCIL, 2..3, 3..5),
            i(A::DEPTH, 2..3, 2..4)
        ));
        assert!(f(
            i(A::DEPTH | A::STENCIL, 2..3, 3..5),
            i(A::DEPTH, 2..3, 4..6)
        ));
        assert!(!f(
            i(A::DEPTH | A::STENCIL, 2..3, 3..5),
            i(A::DEPTH, 2..3, 2..3)
        ));
        assert!(!f(
            i(A::DEPTH | A::STENCIL, 2..3, 3..5),
            i(A::DEPTH, 2..3, 5..6)
        ));
    }

    #[test]
    pub fn image_subresource_range_normalize_remaining_counts() {
        let info = image_subresource(vk::Format::R8_UINT, 4, 6);
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_array_layer: 1,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
            base_mip_level: 2,
            level_count: vk::REMAINING_MIP_LEVELS,
        };

        let range = info.resolve_subresource_counts(range);

        assert_eq!(range.base_array_layer, 1);
        assert_eq!(range.layer_count, 3);
        assert_eq!(range.base_mip_level, 2);
        assert_eq!(range.level_count, 4);
    }

    #[test]
    pub fn image_view_info() {
        let info = ImageViewInfo::new(vk::Format::default(), vk::ImageViewType::TYPE_1D);
        let builder = info.into_builder().build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn image_view_info_builder() {
        let info = ImageViewInfo::new(vk::Format::default(), vk::ImageViewType::TYPE_1D);
        let builder = ImageViewInfoBuilder::default()
            .format(vk::Format::default())
            .ty(vk::ImageViewType::TYPE_1D)
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .build();

        assert_eq!(info, builder);
    }

    #[test]
    pub fn image_view_info_builder_defaults() {
        assert_eq!(
            ImageViewInfoBuilder::default().build(),
            ImageViewInfo::new(vk::Format::UNDEFINED, vk::ImageViewType::TYPE_2D)
        );
    }
}
