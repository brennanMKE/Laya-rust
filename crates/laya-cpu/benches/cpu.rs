use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use criterion::{Bencher, Criterion, Throughput, criterion_group, criterion_main};
use laya_core::{Batch, Engine, checkpoint};
use laya_cpu::{CpuEngine, GemmKernels, Kernels, Options, PortableKernels};

fn batch(sequences: usize, length: usize) -> Batch {
    let mut batch = Batch::default();
    let mut rng = 1u32;
    for i in 0..sequences {
        let mut tokens: Vec<_> = (0..length)
            .map(|_| {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                (rng >> 8) % 50_000
            })
            .collect();
        let markers = [length / 4, length / 4 + 5];
        tokens[0] = 50281;
        tokens[length - 1] = 50282;
        for marker in markers {
            tokens[marker] = 50284;
        }
        batch.push(&tokens, markers.map(|n| n as u32).to_vec(), (i % 3) as u8);
    }
    batch
}

fn measured(b: &mut Bencher<'_>, times: &mut Vec<Duration>, mut run: impl FnMut()) {
    b.iter_custom(|iterations| {
        let mut elapsed = Duration::ZERO;
        for _ in 0..iterations {
            let start = Instant::now();
            run();
            let duration = start.elapsed();
            times.push(duration);
            elapsed += duration;
        }
        elapsed
    });
}

fn report(name: &str, times: &mut [Duration], flops: Option<u64>) {
    if times.is_empty() {
        return;
    }
    times.sort_unstable();
    let median = times[times.len() / 2].as_secs_f64();
    let p99 = times[(times.len() * 99).div_ceil(100) - 1].as_secs_f64();
    let throughput = flops.map(|flops| format!(", {:.2} GFLOP/s at median", flops as f64 / median / 1e9)).unwrap_or_default();
    eprintln!("{name}: median {:.2} ms, p99 {:.2} ms{throughput} ({} timings)", median * 1e3, p99 * 1e3, times.len());
}

fn cpu(c: &mut Criterion) {
    let dir = std::env::var_os("LAYA_CHECKPOINT").map(PathBuf::from).or_else(|| checkpoint::find(None, "", "", "").ok());
    if let Some(dir) = dir {
        let options = Options::default();
        let mut times = Vec::new();
        c.bench_function("load", |b| measured(b, &mut times, || drop(black_box(CpuEngine::load(&dir, options).unwrap()))));
        report("load", &mut times, None);
        let engine = CpuEngine::load(&dir, options).unwrap();
        let mut group = c.benchmark_group("forward");
        for (n, length) in [(1, 56), (6, 50), (14, 56), (1, 512)] {
            let name = format!("{n}x{length}");
            let batch = batch(n, length);
            let mut times = Vec::new();
            group.bench_function(&name, |b| measured(b, &mut times, || drop(black_box(engine.forward(&batch).unwrap()))));
            report(&format!("forward/{name}"), &mut times, None);
        }
        group.finish();
    } else {
        eprintln!("Skipping load and forward without the checkpoint");
    }

    let mut group = c.benchmark_group("matmul_t");
    let kernels: [&dyn Kernels; 2] = [&GemmKernels::default(), &PortableKernels];
    let values = |len| (0..len).map(|i| ((i % 251) as f32 - 125.0) / 125.0).collect::<Vec<_>>();
    for kernel in kernels {
        for m in [56, 784] {
            for (n, k) in [(3072, 1024), (1024, 1024), (5248, 1024), (1024, 2624)] {
                let a = values(m * k);
                let w = kernel.new_weight(n, k, values(n * k));
                let mut output = vec![0.0; m * n];
                let flops = (2 * m * n * k) as u64;
                let name = format!("{}/m={m}/{n}x{k}", kernel.name());
                let mut times = Vec::new();
                group.throughput(Throughput::Elements(flops));
                group.bench_function(&name, |b| {
                    measured(b, &mut times, || {
                        kernel.matmul_t(black_box(&mut output), black_box(&a), black_box(&w), m, None);
                    })
                });
                report(&format!("matmul_t/{name}"), &mut times, Some(flops));
            }
        }
    }
    group.finish();
    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: getrusage initializes the supplied structure when it succeeds.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0 {
            let rss = unsafe { usage.assume_init() }.ru_maxrss as f64;
            let units = if cfg!(target_os = "macos") { 1024.0 * 1024.0 } else { 1024.0 };
            eprintln!("process peak RSS: {:.1} MiB", rss / units);
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(2));
    targets = cpu
}
criterion_main!(benches);
