use {
    criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main},
    std::path::Path,
    vk_graph::{
        Graph,
        resource::{
            AccelerationStructureAccessType, AccelerationStructureSet,
            AccelerationStructureSetMember, ImageAccessType, ImageSet, ImageSetMember,
        },
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
        (
            "material_array",
            ReorderBenchSpec {
                cmd_count: 10,
                resource_count: 2_048,
                short_lived_uses: 1,
                long_lived_resource_count: 2_048,
                long_lived_uses: 10,
            },
        ),
        (
            "material_array_rr",
            ReorderBenchSpec {
                cmd_count: 11,
                resource_count: 2_048,
                short_lived_uses: 1,
                long_lived_resource_count: 2_048,
                long_lived_uses: 11,
            },
        ),
        (
            "material_array_max_bounces",
            ReorderBenchSpec {
                cmd_count: 25,
                resource_count: 2_048,
                short_lived_uses: 1,
                long_lived_resource_count: 2_048,
                long_lived_uses: 25,
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

    // Alternate the two read profiles while retaining one aggregate scheduling token.
    for cmd_count in [10, 11, 25] {
        let resource_set =
            AccelerationStructureSet::new(std::iter::empty::<AccelerationStructureSetMember>())
                .expect("empty acceleration structure set");
        let mut graph = Graph::new();
        let resource_set_node = graph.bind_resource(&resource_set);
        for cmd_idx in 0..cmd_count {
            let access = if cmd_idx % 2 == 0 {
                AccelerationStructureAccessType::BuildRead
            } else {
                AccelerationStructureAccessType::RayTracingRead
            };
            graph
                .begin_cmd()
                .resource_access(resource_set_node, access)
                .record_cmd(|_| {})
                .end_cmd();
        }
        let submission = graph.finalize();
        let mut harness = ReorderBenchHarness::from_submission(&submission, 1);

        group.throughput(Throughput::Elements(cmd_count));
        group.bench_with_input(
            BenchmarkId::new("acceleration_structure_set", format!("{cmd_count}c_empty")),
            &cmd_count,
            |b, _| b.iter(|| black_box(harness.reorder_once())),
        );
    }

    // Reordering depends on the set token count, so an empty set isolates aggregate scheduling.
    for cmd_count in [10, 11, 25] {
        let resource_set =
            ImageSet::new(std::iter::empty::<ImageSetMember>()).expect("empty image set");
        let mut graph = Graph::new();
        let resource_set_node = graph.bind_resource(&resource_set);
        for _ in 0..cmd_count {
            graph
                .begin_cmd()
                .resource_access(resource_set_node, ImageAccessType::SampledRead)
                .record_cmd(|_| {})
                .end_cmd();
        }
        let submission = graph.finalize();
        let mut harness = ReorderBenchHarness::from_submission(&submission, 1);

        group.throughput(Throughput::Elements(cmd_count));
        group.bench_with_input(
            BenchmarkId::new("material_image_set", format!("{cmd_count}c_empty")),
            &cmd_count,
            |b, _| b.iter(|| black_box(harness.reorder_once())),
        );
    }

    group.finish();
}

criterion_group!(benches, submission_reorder_bench);
criterion_main!(benches);
