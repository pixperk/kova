//! Criterion benchmarks comparing HNSW and Flat search latency, and
//! measuring HNSW insert latency on a pre-populated index.
//!
//! Deterministic input via a seeded RNG so runs are comparable across
//! commits and machines. Both indexes are built once per benched size and
//! reused across iterations.

#![allow(missing_docs, clippy::cast_precision_loss)]

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use kova_core::{L2, Vector, VectorId};
use kova_index::{FlatIndex, HnswIndex, HnswParams, Index};
use rand::{RngExt, SeedableRng, rngs::StdRng};

const DIM: usize = 32;
const K: usize = 10;

fn random_vector(rng: &mut StdRng, dim: usize) -> Vector {
    let data: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
    Vector::try_new(data).expect("bench vector valid")
}

fn build_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vector> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| random_vector(&mut rng, dim)).collect()
}

fn build_hnsw(vectors: &[Vector]) -> HnswIndex<L2> {
    let mut idx = HnswIndex::seeded(L2, HnswParams::default(), 13);
    for (i, v) in vectors.iter().enumerate() {
        idx.insert(VectorId::new(i as u64), v.clone()).unwrap();
    }
    idx
}

fn build_flat(vectors: &[Vector]) -> FlatIndex<L2> {
    let mut idx = FlatIndex::new(L2);
    for (i, v) in vectors.iter().enumerate() {
        idx.insert(VectorId::new(i as u64), v.clone()).unwrap();
    }
    idx
}

fn bench_search(c: &mut Criterion) {
    let queries = build_vectors(100, DIM, 99);

    for &n in &[1_000_usize, 10_000] {
        let vectors = build_vectors(n, DIM, 7);
        let hnsw = build_hnsw(&vectors);
        let flat = build_flat(&vectors);

        // Cycle through 100 queries so we don't measure cache-friendliness of
        // a single fixed query.
        c.bench_function(&format!("hnsw_search_at_{n}"), |b| {
            let mut q_iter = queries.iter().cycle();
            b.iter(|| {
                let q = q_iter.next().unwrap();
                hnsw.search(black_box(q), K).unwrap()
            });
        });

        c.bench_function(&format!("flat_search_at_{n}"), |b| {
            let mut q_iter = queries.iter().cycle();
            b.iter(|| {
                let q = q_iter.next().unwrap();
                flat.search(black_box(q), K).unwrap()
            });
        });
    }
}

fn bench_search_at_100k(c: &mut Criterion) {
    // Build once : the 100k HNSW build takes ~minutes. Smaller sample size
    // keeps the run time reasonable while still giving stable mean estimates.
    let vectors = build_vectors(100_000, DIM, 7);
    let queries = build_vectors(100, DIM, 99);
    let hnsw = build_hnsw(&vectors);
    let flat = build_flat(&vectors);

    let mut group = c.benchmark_group("at_100k");
    group.sample_size(20);

    group.bench_function("hnsw_search", |b| {
        let mut q_iter = queries.iter().cycle();
        b.iter(|| {
            let q = q_iter.next().unwrap();
            hnsw.search(black_box(q), K).unwrap()
        });
    });

    group.bench_function("flat_search", |b| {
        let mut q_iter = queries.iter().cycle();
        b.iter(|| {
            let q = q_iter.next().unwrap();
            flat.search(black_box(q), K).unwrap()
        });
    });

    group.finish();
}

fn bench_insert(c: &mut Criterion) {
    let vectors = build_vectors(2_000, DIM, 7);

    c.bench_function("hnsw_insert_into_1k", |b| {
        b.iter_batched(
            || {
                let mut idx = HnswIndex::seeded(L2, HnswParams::default(), 13);
                for (i, v) in vectors[..1_000].iter().enumerate() {
                    idx.insert(VectorId::new(i as u64), v.clone()).unwrap();
                }
                (idx, vectors[1_000].clone())
            },
            |(mut idx, v)| {
                idx.insert(VectorId::new(1_001), v).unwrap();
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_search, bench_insert);
// Gate the 100k benches behind a separate group so users can opt out:
//   cargo bench -p kova-index --bench hnsw -- 'at_100k|hnsw_search|...'
criterion_group!(big_benches, bench_search_at_100k);
criterion_main!(benches, big_benches);
