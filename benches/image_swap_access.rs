use {
    ash::vk,
    criterion::{Criterion, black_box, criterion_group, criterion_main},
    vk_graph::driver::image::bench::SwapAccessBenchHarness,
    vk_sync::AccessType,
};

fn full_range(layers: u32, mips: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_array_layer: 0,
        layer_count: layers,
        base_mip_level: 0,
        level_count: mips,
    }
}

fn full_dual_range(layers: u32, mips: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        base_array_layer: 0,
        layer_count: layers,
        base_mip_level: 0,
        level_count: mips,
    }
}

fn single_subresource(layer: u32, mip: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_array_layer: layer,
        layer_count: 1,
        base_mip_level: mip,
        level_count: 1,
    }
}

fn dual_single_subresource(
    aspect: vk::ImageAspectFlags,
    layer: u32,
    mip: u32,
) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_array_layer: layer,
        layer_count: 1,
        base_mip_level: mip,
        level_count: 1,
    }
}

fn subresource_block(
    aspect: vk::ImageAspectFlags,
    start_layer: u32,
    start_mip: u32,
    count_layers: u32,
    count_mips: u32,
) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_array_layer: start_layer,
        layer_count: count_layers,
        base_mip_level: start_mip,
        level_count: count_mips,
    }
}

fn remaining_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_array_layer: 0,
        layer_count: vk::REMAINING_ARRAY_LAYERS,
        base_mip_level: 0,
        level_count: vk::REMAINING_MIP_LEVELS,
    }
}

fn image_swap_access_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_swap_access");

    // 1. Uniform, 1x1, full range
    let h = SwapAccessBenchHarness::new(1, 1, vk::Format::R8G8B8A8_UNORM);
    group.bench_function("uniform_1x1_full", |b| {
        b.iter(|| black_box(h.swap_access(AccessType::AnyShaderWrite, full_range(1, 1))));
    });

    // 2. Uniform, 1x1, remaining ranges
    let h = SwapAccessBenchHarness::new(1, 1, vk::Format::R8G8B8A8_UNORM);
    group.bench_function("uniform_1x1_remaining", |b| {
        b.iter(|| black_box(h.swap_access(AccessType::AnyShaderWrite, remaining_range())));
    });

    // 3. Dual-aspect, full range
    let h = SwapAccessBenchHarness::new(1, 1, vk::Format::D32_SFLOAT_S8_UINT);
    group.bench_function("dual_aspect_full", |b| {
        b.iter(|| {
            black_box(h.swap_access(
                AccessType::DepthStencilAttachmentWrite,
                full_dual_range(1, 1),
            ))
        });
    });

    // 4. Dual-aspect, single aspect
    let h = SwapAccessBenchHarness::new(1, 1, vk::Format::D32_SFLOAT_S8_UINT);
    group.bench_function("dual_aspect_single_aspect", |b| {
        b.iter(|| {
            black_box(h.swap_access(
                AccessType::DepthStencilAttachmentWrite,
                dual_single_subresource(vk::ImageAspectFlags::DEPTH, 0, 0),
            ))
        });
    });

    // 5. Dense, 1x2 mips, full range (promotes + uniform shortcut)
    let h = SwapAccessBenchHarness::new(1, 2, vk::Format::R8G8B8A8_UNORM);
    group.bench_function("dense_2mip_full", |b| {
        b.iter(|| black_box(h.swap_access(AccessType::AnyShaderWrite, full_range(1, 2))));
    });

    // 6. Dense, 4x4, single subresource (promotes + dense iteration)
    let h = SwapAccessBenchHarness::new(4, 4, vk::Format::R8G8B8A8_UNORM);
    group.bench_function("dense_4x4_partial", |b| {
        b.iter(|| black_box(h.swap_access(AccessType::AnyShaderWrite, single_subresource(0, 0))));
    });

    // 7. Dense, 4x4, many subresources (promotes + dense iteration over 2x2)
    let h = SwapAccessBenchHarness::new(4, 4, vk::Format::R8G8B8A8_UNORM);
    group.bench_function("dense_4x4_many", |b| {
        b.iter(|| {
            black_box(h.swap_access(
                AccessType::AnyShaderWrite,
                subresource_block(vk::ImageAspectFlags::COLOR, 0, 0, 2, 2),
            ))
        });
    });

    // 8. Dense, 4x4, steady-state partial
    let h = SwapAccessBenchHarness::new(4, 4, vk::Format::R8G8B8A8_UNORM);
    h.swap_access(AccessType::AnyShaderWrite, single_subresource(0, 0));
    group.bench_function("dense_4x4_partial_steady", |b| {
        b.iter(|| black_box(h.swap_access(AccessType::AnyShaderWrite, single_subresource(0, 0))));
    });

    // 9. Dense, 4x4, steady-state many subresources
    let h = SwapAccessBenchHarness::new(4, 4, vk::Format::R8G8B8A8_UNORM);
    h.swap_access(
        AccessType::AnyShaderWrite,
        subresource_block(vk::ImageAspectFlags::COLOR, 0, 0, 2, 2),
    );
    group.bench_function("dense_4x4_many_steady", |b| {
        b.iter(|| {
            black_box(h.swap_access(
                AccessType::AnyShaderWrite,
                subresource_block(vk::ImageAspectFlags::COLOR, 0, 0, 2, 2),
            ))
        });
    });

    // 10. Dense, 8x8, single subresource (large promote)
    let h = SwapAccessBenchHarness::new(8, 8, vk::Format::R8G8B8A8_UNORM);
    group.bench_function("dense_8x8_partial", |b| {
        b.iter(|| black_box(h.swap_access(AccessType::AnyShaderWrite, single_subresource(0, 0))));
    });

    // 11. Dense, 8x8, many subresources (large promote + 4x4 iteration)
    let h = SwapAccessBenchHarness::new(8, 8, vk::Format::R8G8B8A8_UNORM);
    group.bench_function("dense_8x8_many", |b| {
        b.iter(|| {
            black_box(h.swap_access(
                AccessType::AnyShaderWrite,
                subresource_block(vk::ImageAspectFlags::COLOR, 0, 0, 4, 4),
            ))
        });
    });

    // 12. Dense, 8x8, steady-state partial
    let h = SwapAccessBenchHarness::new(8, 8, vk::Format::R8G8B8A8_UNORM);
    h.swap_access(AccessType::AnyShaderWrite, single_subresource(0, 0));
    group.bench_function("dense_8x8_partial_steady", |b| {
        b.iter(|| black_box(h.swap_access(AccessType::AnyShaderWrite, single_subresource(0, 0))));
    });

    // 13. Dense, 8x8, steady-state many subresources
    let h = SwapAccessBenchHarness::new(8, 8, vk::Format::R8G8B8A8_UNORM);
    h.swap_access(
        AccessType::AnyShaderWrite,
        subresource_block(vk::ImageAspectFlags::COLOR, 0, 0, 4, 4),
    );
    group.bench_function("dense_8x8_many_steady", |b| {
        b.iter(|| {
            black_box(h.swap_access(
                AccessType::AnyShaderWrite,
                subresource_block(vk::ImageAspectFlags::COLOR, 0, 0, 4, 4),
            ))
        });
    });

    // 14. Dense, 1x8, single subresource (single layer, many mips)
    let h = SwapAccessBenchHarness::new(1, 8, vk::Format::R8G8B8A8_UNORM);
    group.bench_function("dense_1x8_partial", |b| {
        b.iter(|| black_box(h.swap_access(AccessType::AnyShaderWrite, single_subresource(0, 0))));
    });

    // 15. Dense, 8x1, single subresource (many layers, single mip)
    let h = SwapAccessBenchHarness::new(8, 1, vk::Format::R8G8B8A8_UNORM);
    group.bench_function("dense_8x1_partial", |b| {
        b.iter(|| black_box(h.swap_access(AccessType::AnyShaderWrite, single_subresource(0, 0))));
    });

    // 16. Dual-aspect dense, 4x4, partial
    let h = SwapAccessBenchHarness::new(4, 4, vk::Format::D32_SFLOAT_S8_UINT);
    group.bench_function("dual_aspect_dense_4x4_partial", |b| {
        b.iter(|| {
            black_box(h.swap_access(
                AccessType::DepthStencilAttachmentWrite,
                subresource_block(
                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                    0,
                    0,
                    1,
                    1,
                ),
            ))
        });
    });

    // 17. Uniform steady-state (pre-warmed)
    let h = SwapAccessBenchHarness::new(1, 1, vk::Format::R8G8B8A8_UNORM);
    h.swap_access(AccessType::AnyShaderWrite, full_range(1, 1));
    group.bench_function("uniform_steady", |b| {
        b.iter(|| black_box(h.swap_access(AccessType::AnyShaderWrite, full_range(1, 1))));
    });

    // 18. Batch 4 swaps on dense 4x4 — mix of single and multi-subresource ranges
    let h = SwapAccessBenchHarness::new(4, 4, vk::Format::R8G8B8A8_UNORM);
    group.bench_function("batch_4_swaps", |b| {
        b.iter(|| {
            black_box(h.swap_access(AccessType::AnyShaderWrite, single_subresource(0, 0)));
            black_box(h.swap_access(
                AccessType::AnyShaderReadOther,
                subresource_block(vk::ImageAspectFlags::COLOR, 1, 0, 2, 1),
            ));
            black_box(h.swap_access(AccessType::TransferWrite, single_subresource(0, 1)));
            black_box(h.swap_access(
                AccessType::TransferRead,
                subresource_block(vk::ImageAspectFlags::COLOR, 0, 2, 1, 2),
            ));
        });
    });

    // 19. Batch 16 swaps on dense 4x4 — mix of single and multi-subresource ranges
    let h = SwapAccessBenchHarness::new(4, 4, vk::Format::R8G8B8A8_UNORM);
    let access_types = [
        AccessType::AnyShaderWrite,
        AccessType::AnyShaderReadOther,
        AccessType::TransferWrite,
        AccessType::TransferRead,
    ];
    let ranges = [
        subresource_block(vk::ImageAspectFlags::COLOR, 0, 0, 2, 2),
        single_subresource(1, 2),
        single_subresource(0, 3),
        subresource_block(vk::ImageAspectFlags::COLOR, 2, 0, 2, 1),
        single_subresource(2, 2),
        single_subresource(3, 2),
        single_subresource(2, 3),
        subresource_block(vk::ImageAspectFlags::COLOR, 0, 1, 1, 2),
        single_subresource(3, 0),
        single_subresource(0, 2),
        subresource_block(vk::ImageAspectFlags::COLOR, 1, 3, 2, 1),
        single_subresource(1, 1),
        single_subresource(2, 0),
        single_subresource(3, 3),
        subresource_block(vk::ImageAspectFlags::COLOR, 0, 0, 1, 1),
        single_subresource(3, 1),
    ];
    group.bench_function("batch_16_swaps", |b| {
        b.iter(|| {
            for i in 0..16 {
                black_box(h.swap_access(access_types[i & 3], ranges[i]));
            }
        });
    });

    group.finish();
}

criterion_group!(benches, image_swap_access_bench);
criterion_main!(benches);
