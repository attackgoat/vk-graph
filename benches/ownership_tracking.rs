// CPU-only ownership microbenchmarks. One iteration is one operation except the explicitly
// named cycles. Contention measures aggregate wall time for four persistent-in-sample workers;
// thread creation/join is outside the reported duration.
use {
    criterion::{BatchSize, Criterion, criterion_group, criterion_main},
    std::{
        hint::black_box,
        sync::Barrier,
        time::{Duration, Instant},
    },
    vk_graph::driver::{
        buffer::bench::OwnershipBenchHarness as Buffer,
        image::bench::OwnershipBenchHarness as Image,
    },
};

const CELLS: u32 = 512;

trait Tracker: Sync {
    fn new() -> Self;
    fn set(&self, owner: u32, start: u32, count: u32);
    fn read(&self) -> (u64, usize);
}

impl Tracker for Buffer {
    fn new() -> Self {
        Self::new(CELLS)
    }
    fn set(&self, owner: u32, start: u32, count: u32) {
        self.set(owner, start, count);
    }
    fn read(&self) -> (u64, usize) {
        self.read()
    }
}

impl Tracker for Image {
    fn new() -> Self {
        Self::new(64, 8)
    }
    fn set(&self, owner: u32, start: u32, count: u32) {
        self.set(owner, start, count);
    }
    fn read(&self) -> (u64, usize) {
        self.read()
    }
}

fn uniform<T: Tracker>() -> T {
    let tracker = T::new();
    tracker.set(1, 0, CELLS);
    tracker
}

fn recovered<T: Tracker>() -> T {
    let tracker = uniform::<T>();
    tracker.set(2, CELLS - 1, 1);
    tracker.set(1, 0, CELLS);
    tracker
}

fn sequential<T: Tracker>(c: &mut Criterion, kind: &str) {
    let mut group = c.benchmark_group(format!("ownership/{kind}"));
    let tracker = uniform::<T>();
    group.bench_function("uniform_read", |b| {
        b.iter(|| black_box(black_box(&tracker).read()))
    });
    let tracker = uniform::<T>();
    group.bench_function("uniform_whole_change", |b| {
        let mut owner = 1;
        b.iter(|| {
            owner ^= 3;
            black_box(&tracker).set(black_box(owner), 0, CELLS);
        });
    });
    let tracker = uniform::<T>();
    group.bench_function("same_owner_partial", |b| {
        b.iter(|| black_box(&tracker).set(1, CELLS - 1, 1))
    });
    group.bench_function("first_promotion", |b| {
        b.iter_batched_ref(
            uniform::<T>,
            |tracker| black_box(tracker).set(2, CELLS - 1, 1),
            BatchSize::SmallInput,
        )
    });
    let tracker = recovered::<T>();
    group.bench_function("recovered_read", |b| {
        b.iter(|| black_box(black_box(&tracker).read()))
    });
    let tracker = recovered::<T>();
    group.bench_function("recovered_whole_change", |b| {
        let mut owner = 1;
        b.iter(|| {
            owner ^= 3;
            black_box(&tracker).set(black_box(owner), 0, CELLS);
        });
    });
    let tracker = recovered::<T>();
    group.bench_function("recovered_same_owner_partial", |b| {
        b.iter(|| black_box(&tracker).set(1, CELLS - 1, 1))
    });
    let tracker = uniform::<T>();
    tracker.set(2, CELLS - 1, 1);
    group.bench_function("mixed_partial_change", |b| {
        let mut owner = 2;
        b.iter(|| {
            owner ^= 1;
            black_box(&tracker).set(black_box(owner), CELLS - 1, 1);
        });
    });
    let tracker = uniform::<T>();
    tracker.set(2, CELLS - 1, 1);
    group.bench_function("mixed_read", |b| {
        b.iter(|| black_box(black_box(&tracker).read()))
    });
    let tracker = uniform::<T>();
    group.bench_function("split_equalize_cycle", |b| {
        b.iter(|| {
            black_box(&tracker).set(2, CELLS - 1, 1);
            black_box(&tracker).set(1, 0, CELLS);
        })
    });
    let tracker = uniform::<T>();
    group.bench_function("fragment_equalize_cycle", |b| {
        b.iter(|| {
            for cell in (0..128).step_by(2) {
                black_box(&tracker).set(2, cell, 1);
            }
            black_box(&tracker).set(1, 0, CELLS);
        })
    });
    group.finish();
}

fn contended<T: Tracker>(c: &mut Criterion, kind: &str) {
    let mut group = c.benchmark_group(format!("ownership_contended/{kind}"));
    for (name, read) in [
        ("recovered_read_4t", true),
        ("recovered_same_owner_partial_4t", false),
    ] {
        let tracker = recovered::<T>();
        group.bench_function(name, |b| {
            b.iter_custom(|iterations| {
                let start = Barrier::new(5);
                let finish = Barrier::new(5);
                std::thread::scope(|scope| {
                    let start = &start;
                    let finish = &finish;
                    let tracker = &tracker;
                    // Barriers live until all joins; spawning itself is excluded from elapsed time.
                    let workers = (0..4)
                        .map(|_| {
                            scope.spawn(move || {
                                start.wait();
                                for _ in 0..iterations {
                                    if read {
                                        black_box(black_box(tracker).read());
                                    } else {
                                        black_box(tracker).set(1, CELLS - 1, 1);
                                    }
                                }
                                finish.wait();
                            })
                        })
                        .collect::<Vec<_>>();
                    let now = Instant::now();
                    start.wait();
                    finish.wait();
                    let elapsed = now.elapsed();
                    for worker in workers {
                        worker.join().unwrap();
                    }
                    elapsed / 4
                })
            })
        });
    }
    group.finish();
}

fn benches(c: &mut Criterion) {
    sequential::<Buffer>(c, "buffer_512");
    sequential::<Image>(c, "image_64x8");
    let mut tracker = Image::new(64, 8);
    tracker.set(1, 0, CELLS);
    tracker.register_queue(1);
    c.bench_function(
        "ownership/image_64x8/same_owner_partial_registered_set",
        |b| b.iter(|| black_box(&tracker).set(1, CELLS - 1, 1)),
    );
    contended::<Buffer>(c, "buffer_512");
    contended::<Image>(c, "image_64x8");
}

criterion_group! { name = ownership; config = Criterion::default()
.warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(2))
.sample_size(50); targets = benches }
criterion_main!(ownership);
