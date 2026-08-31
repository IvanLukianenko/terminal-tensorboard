//! `ttb gen-demo`: synthetic TensorBoard logs for trying out the TUI.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use crate::tfevents::{encode_scalar_event, frame_record};

/// Tiny deterministic RNG (xorshift) so demo runs are reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximate standard normal (sum of uniforms).
    fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let s: f64 = (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0;
        mu + sigma * s
    }
}

struct DemoRun {
    name: &'static str,
    lr: f64,
    seed: u64,
    tensor_format: bool,
}

const RUNS: &[DemoRun] = &[
    DemoRun { name: "baseline", lr: 1.0, seed: 1, tensor_format: false },
    DemoRun { name: "high_lr", lr: 2.2, seed: 2, tensor_format: false },
    DemoRun { name: "low_lr/warmup", lr: 0.4, seed: 3, tensor_format: true },
];

fn append_steps(
    logdir: &Path,
    run: &DemoRun,
    start: i64,
    count: i64,
    total_hint: i64,
) -> std::io::Result<()> {
    let dir = logdir.join(run.name);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("events.out.tfevents.demo");
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    let mut rng = Rng(run.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(start as u64 + 1));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let t0 = now - (total_hint - start) as f64 * 0.05;
    let mut out = Vec::with_capacity(count as usize * 160);
    for i in start..start + count {
        let wall = t0 + (i - start) as f64 * 0.05;
        let progress = i as f64 / total_hint.max(1) as f64;
        let loss = 2.5 * (-3.0 * run.lr * i as f64 / 100.0).exp() + 0.08 + rng.gauss(0.0, 0.05);
        let acc = (1.0 - 0.9 * (-2.5 * run.lr * i as f64 / 100.0).exp() + rng.gauss(0.0, 0.01))
            .clamp(0.0, 1.0);
        let lr_now = run.lr * 0.5 * (1.0 + (std::f64::consts::PI * progress.min(1.0)).cos());
        let grad = rng.gauss(1.0, 0.4).abs() * (-(i as f64) / 4000.0).exp();
        for (tag, value) in [
            ("train/loss", loss),
            ("train/accuracy", acc),
            ("train/lr", lr_now),
            ("train/grad_norm", grad),
        ] {
            out.extend(frame_record(&encode_scalar_event(tag, i, wall, value as f32, run.tensor_format)));
        }
        if i % 25 == 0 {
            let v = (loss + 0.15 + rng.gauss(0.0, 0.03)) as f32;
            out.extend(frame_record(&encode_scalar_event("val/loss", i, wall, v, false)));
        }
    }
    f.write_all(&out)
}

pub fn run(logdir: &Path, steps: i64, live: bool) -> std::io::Result<()> {
    for run in RUNS {
        append_steps(logdir, run, 0, steps, steps)?;
    }
    println!("wrote {} runs x {} steps to {}", RUNS.len(), steps, logdir.display());
    if live {
        println!("appending 20 steps/second per run — Ctrl-C to stop");
        let mut step = steps;
        loop {
            for run in RUNS {
                append_steps(logdir, run, step, 20, step + 20)?;
            }
            step += 20;
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }
    Ok(())
}
