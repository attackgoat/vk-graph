use {
    bytemuck::cast_slice,
    imgui::{DrawData, DrawIdx, DrawList, DrawVert, internal::RawWrapper as _},
    std::mem::size_of,
    vk_graph::{
        driver::{
            DriverError,
            ash::vk,
            buffer::{Buffer, BufferInfo},
        },
        pool::{Lease, Pool},
    },
};

const VERTEX_STRIDE: usize = 20;

pub(super) struct GeometryBuffers {
    pub(super) indices: Lease<Buffer>,
    pub(super) vertices: Lease<Buffer>,
}

#[derive(Debug, Default)]
pub(super) struct GeometryCapacity {
    index_bytes: vk::DeviceSize,
    vertex_bytes: vk::DeviceSize,
}

impl GeometryCapacity {
    fn grow(&mut self, required: GeometryOffsets) -> Result<(), DriverError> {
        let grow =
            |current: vk::DeviceSize, required: usize| -> Result<vk::DeviceSize, DriverError> {
                let capacity = if required as vk::DeviceSize > current {
                    (required as vk::DeviceSize)
                        .checked_next_power_of_two()
                        .ok_or(DriverError::OutOfMemory)?
                } else {
                    current
                };
                usize::try_from(capacity).map_err(|_| DriverError::OutOfMemory)?;
                Ok(capacity)
            };
        let index_bytes = grow(self.index_bytes, required.indices)?;
        let vertex_bytes = grow(self.vertex_bytes, required.vertices)?;
        self.index_bytes = index_bytes;
        self.vertex_bytes = vertex_bytes;
        Ok(())
    }

    pub(super) fn lease<P: Pool<BufferInfo, Buffer>>(
        &mut self,
        pool: &mut P,
        data: &DrawData,
    ) -> Result<Option<GeometryBuffers>, DriverError> {
        // imgui 0.12 constructs a slice from a null list pointer on empty frames.
        if data.draw_lists_count() == 0 {
            return Ok(None);
        }

        let mut required = GeometryOffsets::default();
        for list in data.draw_lists() {
            let Some((indices, vertices)) = draw_list_geometry(list) else {
                continue;
            };
            required.push(indices.len(), vertices.len())?;
        }
        if required.indices == 0 || required.vertices == 0 {
            return Ok(None);
        }
        self.grow(required)?;
        let mut indices = pool.resource(BufferInfo::host_mem(
            self.index_bytes,
            vk::BufferUsageFlags::INDEX_BUFFER,
        ))?;
        let mut vertices = pool.resource(BufferInfo::host_mem(
            self.vertex_bytes,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        ))?;
        {
            let indices = Buffer::mapped_slice_mut(&mut indices);
            let vertices = Buffer::mapped_slice_mut(&mut vertices);
            let mut end = GeometryOffsets::default();
            for list in data.draw_lists() {
                let Some((list_indices, list_vertices)) = draw_list_geometry(list) else {
                    continue;
                };
                let start = end.push(list_indices.len(), list_vertices.len())?;
                indices[start.indices..end.indices].copy_from_slice(cast_slice(list_indices));
                for (dst, vertex) in vertices[start.vertices..end.vertices]
                    .chunks_exact_mut(VERTEX_STRIDE)
                    .zip(list_vertices)
                {
                    dst[..8].copy_from_slice(cast_slice(&vertex.pos));
                    dst[8..16].copy_from_slice(cast_slice(&vertex.uv));
                    dst[16..20].copy_from_slice(&vertex.col);
                }
            }
        }
        Ok(Some(GeometryBuffers { indices, vertices }))
    }
}

/// CPU byte offsets into the frame's packed index and vertex buffers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct GeometryOffsets {
    pub(super) indices: usize,
    pub(super) vertices: usize,
}

impl GeometryOffsets {
    pub(super) fn push(&mut self, indices: usize, vertices: usize) -> Result<Self, DriverError> {
        let start = *self;
        let indices = indices
            .checked_mul(size_of::<DrawIdx>())
            .and_then(|bytes| self.indices.checked_add(bytes))
            .ok_or(DriverError::OutOfMemory)?;
        let vertices = vertices
            .checked_mul(VERTEX_STRIDE)
            .and_then(|bytes| self.vertices.checked_add(bytes))
            .ok_or(DriverError::OutOfMemory)?;
        *self = Self { indices, vertices };
        Ok(start)
    }
}

pub(super) fn draw_list_geometry(list: &DrawList) -> Option<(&[DrawIdx], &[DrawVert])> {
    // SAFETY: Only read the borrowed list's buffer sizes, without accessing its raw buffers.
    let raw = unsafe { list.raw() };
    // imgui 0.12 creates slices from null buffer pointers even when their length is zero.
    if raw.IdxBuffer.Size == 0 || raw.VtxBuffer.Size == 0 {
        return None;
    }

    Some((list.idx_buffer(), list.vtx_buffer()))
}

#[cfg(test)]
mod test;
