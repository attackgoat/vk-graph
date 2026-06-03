use {
    criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main},
    vk_graph::submission::bench::{ReorderBenchHarness, ReorderBenchSpec},
};

fn submission_reorder_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("submission_reorder");

    for (shape, spec) in [
        (
            "sparse",
            ReorderBenchSpec {
                pass_count: 128,
                resource_count: 64,
                short_lived_uses: 3,
                long_lived_resource_count: 0,
                long_lived_uses: 0,
            },
        ),
        (
            "mixed",
            ReorderBenchSpec {
                pass_count: 512,
                resource_count: 128,
                short_lived_uses: 3,
                long_lived_resource_count: 4,
                long_lived_uses: 96,
            },
        ),
        (
            "mixed",
            ReorderBenchSpec {
                pass_count: 1024,
                resource_count: 192,
                short_lived_uses: 3,
                long_lived_resource_count: 8,
                long_lived_uses: 160,
            },
        ),
        (
            "mixed",
            ReorderBenchSpec {
                pass_count: 2000,
                resource_count: 256,
                short_lived_uses: 2,
                long_lived_resource_count: 12,
                long_lived_uses: 220,
            },
        ),
    ] {
        let mut harness = ReorderBenchHarness::new(spec);
        group.throughput(Throughput::Elements(spec.pass_count as u64));
        group.bench_with_input(
            BenchmarkId::new(
                shape,
                format!("{}p_{}r", spec.pass_count, spec.resource_count),
            ),
            &spec,
            |b, _| {
                b.iter(|| black_box(harness.reorder_once()));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, submission_reorder_bench);
criterion_main!(benches);
