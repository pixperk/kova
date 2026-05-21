//! Criterion benchmarks for the three [`Distance`] metrics across realistic
//! embedding dimensions.
//!
//! - 128  : small vision encoders / quantized models
//! - 768  : `BERT-base`, sentence-transformers
//! - 1536 : `OpenAI` `text-embedding-3-small`
//!
//! Inputs are deterministic so runs are comparable across machines and commits.
//! Future SIMD work (Phase 2 Week 3) should slot in behind a `cfg` flag and be
//! compared against the numbers these benches record.

#![allow(missing_docs, clippy::cast_precision_loss)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use kova_core::{Cosine, Distance, InnerProduct, L2, Vector};

const DIMS: &[usize] = &[128, 768, 1536];

fn make_pair(dim: usize) -> (Vector, Vector) {
    let denom = dim as f32;
    let a: Vec<f32> = (0..dim).map(|i| i as f32 / denom).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32 + 0.5) / denom).collect();
    (
        Vector::try_new(a).expect("bench vector valid"),
        Vector::try_new(b).expect("bench vector valid"),
    )
}

fn bench_metric<D: Distance>(c: &mut Criterion, metric: &D) {
    for &dim in DIMS {
        let (a, b) = make_pair(dim);
        let name = format!("{}_{}", metric.name(), dim);
        c.bench_function(&name, |bench| {
            bench.iter(|| metric.distance(black_box(&a), black_box(&b)));
        });
    }
}

fn bench_l2(c: &mut Criterion) {
    bench_metric(c, &L2);
}

fn bench_cosine(c: &mut Criterion) {
    bench_metric(c, &Cosine);
}

fn bench_inner_product(c: &mut Criterion) {
    bench_metric(c, &InnerProduct);
}

criterion_group!(benches, bench_l2, bench_cosine, bench_inner_product);
criterion_main!(benches);
