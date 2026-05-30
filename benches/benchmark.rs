use assembly_engine::engine::{AssemblyEngine, AssemblyEngineConfig, Part, Placement, Query};
use criterion::{Criterion, criterion_group, criterion_main};
use glam::{Quat, Vec3};
use rand::seq::IndexedRandom;

fn assemble_model(mut engine: AssemblyEngine) {
    let mut rng = rand::rng();
    for _ in 0..32 {
        let candidates = engine.query(&Query::default());
        let candidate = candidates.choose(&mut rng).unwrap();
        engine.add_placement(*candidate);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let config = AssemblyEngineConfig::new(4, 1.0);
    let part_ids = [3003, 3004, 3005, 3009, 3010, 3020];
    let mut parts = Vec::new();
    for part_id in part_ids {
        let part_path = format!("../part_converter/output/{part_id}.json");
        let mut file = std::fs::File::open(part_path).unwrap();
        let part: Part = serde_json::from_reader(&mut file).unwrap();
        parts.push(part);
    }
    c.bench_function("assemble model", |b| {
        b.iter(|| {
            let mut engine = AssemblyEngine::new(&parts, &config);
            let start = Placement {
                part_index: 0,
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
            };
            engine.add_placement(start);
            assemble_model(engine)
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark
}
criterion_main!(benches);
