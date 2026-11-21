//! Tests are based on the common synchronization examples on the Vulkan-Docs wiki: https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples.

use crate::driver::sync as vk_sync;
use ash::vk;

#[test]
fn compute_write_storage_compute_read_storage() {
    // Compute write to storage buffer/image, Compute read from storage buffer/image
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[vk_sync::AccessType::ComputeShaderReadOther],
    };

    let (src_mask, dst_mask, barrier) = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(src_mask, vk::PipelineStageFlags::COMPUTE_SHADER);
    assert_eq!(dst_mask, vk::PipelineStageFlags::COMPUTE_SHADER);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags::SHADER_WRITE);
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags::SHADER_READ);
}

#[test]
fn compute_read_storage_compute_write_storage() {
    // Compute read from storage buffer, Compute write from storage buffer
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[vk_sync::AccessType::ComputeShaderReadOther],
    };

    let (src_mask, dst_mask, barrier) = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(src_mask, vk::PipelineStageFlags::COMPUTE_SHADER);
    assert_eq!(dst_mask, vk::PipelineStageFlags::COMPUTE_SHADER);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags::SHADER_WRITE);
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags::SHADER_READ);
}

#[test]
fn compute_write_storage_graphics_read_index() {
    // Compute write to storage buffer, Graphics read as index buffer
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[vk_sync::AccessType::IndexBuffer],
    };

    let (src_mask, dst_mask, barrier) = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(src_mask, vk::PipelineStageFlags::COMPUTE_SHADER);
    assert_eq!(dst_mask, vk::PipelineStageFlags::VERTEX_INPUT);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags::SHADER_WRITE);
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags::INDEX_READ);
}

#[test]
fn compute_write_storage_graphics_read_indirect() {
    // Compute write to storage buffer, Graphics read as indirect buffer
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[vk_sync::AccessType::IndirectBuffer],
    };

    let (src_mask, dst_mask, barrier) = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(src_mask, vk::PipelineStageFlags::COMPUTE_SHADER);
    assert_eq!(dst_mask, vk::PipelineStageFlags::DRAW_INDIRECT);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags::SHADER_WRITE);
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags::INDIRECT_COMMAND_READ
    );
}

#[test]
fn nothing_transfer_read() {
    // None, Transfer read from buffer
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::Nothing],
        next_accesses: &[vk_sync::AccessType::TransferRead],
    };

    let (src_mask, dst_mask, barrier) = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(src_mask, vk::PipelineStageFlags::TOP_OF_PIPE);
    assert_eq!(dst_mask, vk::PipelineStageFlags::TRANSFER);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags::empty());
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags::empty());
}

#[test]
fn transfer_write_graphics_read_vertex() {
    // Transfer write to buffer, Graphics read from vertex buffer
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::TransferWrite],
        next_accesses: &[vk_sync::AccessType::VertexBuffer],
    };

    let (src_mask, dst_mask, barrier) = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(src_mask, vk::PipelineStageFlags::TRANSFER);
    assert_eq!(dst_mask, vk::PipelineStageFlags::VERTEX_INPUT);
    assert_eq!(barrier.src_access_mask, vk::AccessFlags::TRANSFER_WRITE);
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags::VERTEX_ATTRIBUTE_READ
    );
}

#[test]
fn full_pipeline_barrier() {
    // Full pipeline barrier
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::General],
        next_accesses: &[vk_sync::AccessType::General],
    };

    let (src_mask, dst_mask, barrier) = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(src_mask, vk::PipelineStageFlags::ALL_COMMANDS);
    assert_eq!(dst_mask, vk::PipelineStageFlags::ALL_COMMANDS);
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
    );
}

#[test]
fn compute_write_storage_graphics_read_index_compute_read_uniform() {
    // Compute write to storage buffer, Graphics read as index buffer & Compute read as uniform buffer
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[
            vk_sync::AccessType::IndexBuffer,
            vk_sync::AccessType::ComputeShaderReadUniformBuffer,
        ],
    };

    let (src_mask, dst_mask, barrier) = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(src_mask, vk::PipelineStageFlags::COMPUTE_SHADER);
    assert_eq!(
        dst_mask,
        vk::PipelineStageFlags::VERTEX_INPUT | vk::PipelineStageFlags::COMPUTE_SHADER
    );
    assert_eq!(barrier.src_access_mask, vk::AccessFlags::SHADER_WRITE);
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags::INDEX_READ | vk::AccessFlags::UNIFORM_READ
    );
}

#[test]
fn compute_write_texel_graphics_read_indirect_fragment_read_uniform() {
    // Compute write to storage texel buffer, Graphics read as indirect buffer & fragment read as uniform buffer
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[
            vk_sync::AccessType::IndirectBuffer,
            vk_sync::AccessType::FragmentShaderReadUniformBuffer,
        ],
    };

    let (src_mask, dst_mask, barrier) = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(src_mask, vk::PipelineStageFlags::COMPUTE_SHADER);
    assert_eq!(
        dst_mask,
        vk::PipelineStageFlags::DRAW_INDIRECT | vk::PipelineStageFlags::FRAGMENT_SHADER
    );
    assert_eq!(barrier.src_access_mask, vk::AccessFlags::SHADER_WRITE);
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags::INDIRECT_COMMAND_READ | vk::AccessFlags::UNIFORM_READ
    );
}

#[test]
fn compute_write_acceleration_structure_build_input_read() {
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &[vk_sync::AccessType::AccelerationStructureBuildInputRead],
    };

    let (src_mask, dst_mask, barrier) = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(src_mask, vk::PipelineStageFlags::COMPUTE_SHADER);
    assert_eq!(
        dst_mask,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
    );
    assert_eq!(barrier.src_access_mask, vk::AccessFlags::SHADER_WRITE);
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags::SHADER_READ);
}

#[test]
fn compute_write_acceleration_structure_build_indirect_read() {
    let indirect = vk_sync::AccessType::AccelerationStructureBuildIndirectRead;
    let indirect_accesses = [indirect];
    let (src, dst, barrier) = vk_sync::get_memory_barrier(&vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
        next_accesses: &indirect_accesses,
    });
    assert_eq!(src, vk::PipelineStageFlags::COMPUTE_SHADER);
    assert_eq!(
        dst,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
    );
    assert_eq!(barrier.src_access_mask, vk::AccessFlags::SHADER_WRITE);
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags::INDIRECT_COMMAND_READ
    );

    let info = vk_sync::get_access_info2(indirect);
    assert_eq!(
        info.stage_mask,
        vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
    );
    assert_eq!(info.access_mask, vk::AccessFlags2::INDIRECT_COMMAND_READ);
    assert_eq!(info.image_layout, vk::ImageLayout::UNDEFINED);
    let draw = vk_sync::get_access_info2(vk_sync::AccessType::IndirectBuffer);
    assert_eq!(draw.stage_mask, vk::PipelineStageFlags2::DRAW_INDIRECT);
    assert_eq!(draw.access_mask, vk::AccessFlags2::INDIRECT_COMMAND_READ);
    let general = vk_sync::get_access_info2(vk_sync::AccessType::General);
    assert_eq!(general.stage_mask, vk::PipelineStageFlags2::ALL_COMMANDS);
    assert_eq!(
        general.access_mask,
        vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE
    );

    let (_, _, read_then_write) = vk_sync::get_memory_barrier(&vk_sync::GlobalBarrier {
        previous_accesses: &indirect_accesses,
        next_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
    });
    assert!(read_then_write.src_access_mask.is_empty());
}

#[test]
fn acceleration_structure_serialization_buffer_write() {
    let accesses = [vk_sync::AccessType::AccelerationStructureBufferWrite];
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &accesses,
        next_accesses: &accesses,
    };

    let (src_mask, dst_mask, barrier) = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(
        src_mask,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
    );
    assert_eq!(
        dst_mask,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
    );
    assert_eq!(barrier.src_access_mask, vk::AccessFlags::TRANSFER_WRITE);
    assert_eq!(barrier.dst_access_mask, vk::AccessFlags::TRANSFER_WRITE);
}

#[test]
fn acceleration_structure_scratch_read_write() {
    let accesses = [vk_sync::AccessType::AccelerationStructureBuildScratchReadWrite];
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &accesses,
        next_accesses: &accesses,
    };

    let (src_mask, dst_mask, barrier) = vk_sync::get_memory_barrier(&global_barrier);
    let scratch_access = vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
        | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR;

    assert_eq!(
        src_mask,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
    );
    assert_eq!(
        dst_mask,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
    );
    assert_eq!(barrier.src_access_mask, scratch_access);
    assert_eq!(barrier.dst_access_mask, scratch_access);
}

#[test]
fn acceleration_structure_build_write_ray_tracing_read() {
    let global_barrier = vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::AccelerationStructureBuildWrite],
        next_accesses: &[vk_sync::AccessType::RayTracingShaderReadAccelerationStructure],
    };

    let (src_mask, dst_mask, barrier) = vk_sync::get_memory_barrier(&global_barrier);

    assert_eq!(
        src_mask,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
    );
    assert_eq!(dst_mask, vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR);
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
    );
}

#[test]
fn acceleration_structure_build_write_compute_query_read() {
    let read = vk_sync::AccessType::ComputeShaderReadAccelerationStructure;
    let reads = [read];
    let (src_mask, dst_mask, barrier) = vk_sync::get_memory_barrier(&vk_sync::GlobalBarrier {
        previous_accesses: &[vk_sync::AccessType::AccelerationStructureBuildWrite],
        next_accesses: &reads,
    });

    assert_eq!(
        src_mask,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
    );
    assert_eq!(dst_mask, vk::PipelineStageFlags::COMPUTE_SHADER);
    assert_eq!(
        barrier.src_access_mask,
        vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
    );
    assert_eq!(
        barrier.dst_access_mask,
        vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
    );

    let info = vk_sync::get_access_info2(read);
    assert_eq!(info.stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
    assert_eq!(
        info.access_mask,
        vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
    );
    assert_eq!(info.image_layout, vk::ImageLayout::UNDEFINED);
}
