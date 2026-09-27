// CPU-only microbenchmarks for production access trackers. Each cycle splits a
// uniform map, then restores it; setup and tracker construction are untimed.
use {
    ash::vk,
    criterion::{Criterion, black_box, criterion_group, criterion_main},
    std::time::Duration,
    vk_graph::driver::{
        buffer::bench::{AccessBenchHarness as Buffer, RunLookupBenchHarness as RunLookup},
        image::bench::SwapAccessBenchHarness as Image,
    },
    vk_sync::AccessType,
};

fn range(layers: u32, mips: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(layers)
        .level_count(mips)
}

fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("access_tracking");
    let buffer = Buffer::new(512);
    buffer.swap(AccessType::TransferRead, 0, 512);
    group.bench_function("buffer_uniform_partial_same", |b| {
        b.iter(|| black_box(buffer.swap(black_box(AccessType::TransferRead), 511, 512)))
    });

    let buffer = Buffer::with_runs(6);
    group.bench_function("buffer_mixed_partial_same", |b| {
        b.iter(|| black_box(buffer.swap(black_box(AccessType::TransferWrite), 64 + 31, 64 + 32)))
    });

    let buffer = Buffer::new(512);
    buffer.swap(AccessType::TransferRead, 0, 512);
    group.bench_function("buffer_uniform_whole_change", |b| {
        let mut write = false;
        b.iter(|| {
            write = !write;
            black_box(buffer.swap(
                black_box(if write {
                    AccessType::TransferWrite
                } else {
                    AccessType::TransferRead
                }),
                0,
                512,
            ))
        })
    });

    for (layers, mips) in [(4, 4), (8, 8)] {
        let image = Image::new(layers, mips, vk::Format::R8G8B8A8_UNORM);
        let whole = range(layers, mips);
        let partial = range(1, 1);
        image.swap_checksum(AccessType::TransferRead, whole);
        // Establish the split/equalize cycle before timing repeated re-promotions.
        image.swap_checksum(AccessType::TransferWrite, partial);
        image.swap_checksum(AccessType::TransferRead, whole);
        group.bench_function(format!("image_{layers}x{mips}_split_equalize"), |b| {
            b.iter(|| {
                black_box(image.swap_checksum(AccessType::TransferWrite, black_box(partial)));
                black_box(image.swap_checksum(AccessType::TransferRead, black_box(whole)));
            })
        });
    }

    let image = Image::new(8, 8, vk::Format::R8G8B8A8_UNORM);
    image.swap_checksum(AccessType::TransferRead, range(8, 8));
    group.bench_function("image_uniform_partial_same", |b| {
        b.iter(|| black_box(image.swap_checksum(black_box(AccessType::TransferRead), range(1, 1))))
    });

    let image = Image::new(8, 8, vk::Format::R8G8B8A8_UNORM);
    image.swap_checksum(AccessType::TransferRead, range(8, 8));
    group.bench_function("image_uniform_full_same", |b| {
        b.iter(|| black_box(image.swap_checksum(black_box(AccessType::TransferRead), range(8, 8))))
    });

    group.finish();

    let mut group = c.benchmark_group("buffer_fragmented_access");
    for count in [4, 6, 7, 8] {
        let buffer = Buffer::with_runs(count);
        for (position, index) in [("first", 0), ("middle", count / 2), ("last", count - 1)] {
            let start = index as u64 * 64 + 31;
            let access = if index % 2 == 0 {
                AccessType::TransferRead
            } else {
                AccessType::TransferWrite
            };
            group.bench_function(format!("{count}/{position}/selected"), |b| {
                b.iter(|| black_box(buffer.swap(black_box(access), black_box(start), start + 1)))
            });
            group.bench_function(format!("{count}/{position}/binary"), |b| {
                b.iter(|| {
                    black_box(buffer.swap_binary(black_box(access), black_box(start), start + 1))
                })
            });
        }
    }
    group.finish();

    let mut group = c.benchmark_group("buffer_run_lookup");
    for count in [2, 4, 6, 7, 8, 16] {
        let runs = RunLookup::new(count);
        for (position, offset) in [
            ("first", 31),
            ("middle", (count / 2) as u64 * 64 + 31),
            ("last", (count - 1) as u64 * 64 + 31),
        ] {
            group.bench_function(format!("{count}/{position}/binary"), |b| {
                b.iter(|| black_box(runs.binary(black_box(offset))))
            });
            group.bench_function(format!("{count}/{position}/linear"), |b| {
                b.iter(|| black_box(runs.linear(black_box(offset))))
            });
        }
    }
    group.finish();
}

criterion_group! { name = access; config = Criterion::default()
.warm_up_time(Duration::from_millis(300)).measurement_time(Duration::from_secs(1))
.sample_size(30); targets = benches }
criterion_main!(access);
