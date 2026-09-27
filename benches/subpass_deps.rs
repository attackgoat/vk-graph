use {
    criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main},
    std::hint::black_box,
    vk_graph::submission::bench::{SubpassDepsHarness, SubpassDepsSpec},
};

fn subpass_deps_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("subpass_deps");
    for spec in SubpassDepsSpec::matrix() {
        let harness = SubpassDepsHarness::new(spec);
        harness.validate();
        group.throughput(Throughput::Elements(spec.subpass_count as u64));
        group.bench_function(
            BenchmarkId::new(spec.case.name(), spec.subpass_count),
            |b| b.iter(|| black_box(black_box(&harness).build_deps_once())),
        );
    }
    group.finish();
}

criterion_group!(benches, subpass_deps_bench);
criterion_main!(benches);
