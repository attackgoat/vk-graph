use {
    criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main},
    std::path::Path,
    vk_graph::{
        Graph,
        submission::bench::{ReorderBenchHarness, ReorderBenchSpec},
    },
};

fn submission_reorder_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("submission_reorder");

    for (shape, spec) in [
        (
            "sparse",
            ReorderBenchSpec {
                cmd_count: 128,
                resource_count: 64,
                short_lived_uses: 3,
                long_lived_resource_count: 0,
                long_lived_uses: 0,
            },
        ),
        (
            "mixed",
            ReorderBenchSpec {
                cmd_count: 512,
                resource_count: 128,
                short_lived_uses: 3,
                long_lived_resource_count: 4,
                long_lived_uses: 96,
            },
        ),
        (
            "mixed",
            ReorderBenchSpec {
                cmd_count: 1024,
                resource_count: 192,
                short_lived_uses: 3,
                long_lived_resource_count: 8,
                long_lived_uses: 160,
            },
        ),
        (
            "mixed",
            ReorderBenchSpec {
                cmd_count: 2000,
                resource_count: 256,
                short_lived_uses: 2,
                long_lived_resource_count: 12,
                long_lived_uses: 220,
            },
        ),
    ] {
        let mut harness = ReorderBenchHarness::new(spec);
        group.throughput(Throughput::Elements(spec.cmd_count as u64));
        group.bench_with_input(
            BenchmarkId::new(
                shape,
                format!("{}c_{}r", spec.cmd_count, spec.resource_count),
            ),
            &spec,
            |b, _| {
                b.iter(|| black_box(harness.reorder_once()));
            },
        );
    }

    for (fixture_name, file_name) in [
        ("real_game_49", "graph-1783212230368.bin"),
        ("real_game_114", "graph-1783212245365.bin"),
    ] {
        let graph = Graph::import_fixture(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("res/graph-fixture")
                .join(file_name),
        )
        .unwrap_or_else(|err| panic!("unable to import {file_name}: {err}"));

        for repeat_count in [1, 8, 32] {
            let mut harness = ReorderBenchHarness::from_graph(&graph, repeat_count);
            let cmd_count = harness.cmd_count();
            group.throughput(Throughput::Elements(cmd_count as u64));
            group.bench_with_input(
                BenchmarkId::new(fixture_name, format!("{cmd_count}c_{repeat_count}x")),
                &repeat_count,
                |b, _| b.iter(|| black_box(harness.reorder_once())),
            );
        }
    }

    group.finish();
}

criterion_group!(benches, submission_reorder_bench);
criterion_main!(benches);
