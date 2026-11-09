//! criterion benchmarks on the REAL left-arm model (benches/data/arm_l.json from benches/export_model.py) and a
//! Kaggle v7 head frame (optional binary files, skipped when absent).
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use team048_accel::col::{Collision, Env};
use team048_accel::kin::{Chain, Rt, N, Q};
use team048_accel::perc;

fn load() -> (Collision, Vec<Q>, Vec<Rt>, Vec<Q>, Q) {
    let d: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/benches/data/arm_l.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let f = |v: &serde_json::Value| -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    let origins: Vec<[f64; 16]> = d["origins"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| f(o).try_into().unwrap())
        .collect();
    let axes: Vec<[f64; 3]> = d["axes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| f(o).try_into().unwrap())
        .collect();
    let movable: Vec<bool> = d["movable"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_bool().unwrap())
        .collect();
    let tcp: [f64; 16] = f(&d["tcp"]).try_into().unwrap();
    let chain = Chain::new(
        &origins,
        &axes,
        &movable,
        &tcp,
        &f(&d["lower"]),
        &f(&d["upper"]),
    )
    .unwrap();
    let frame: Vec<usize> = d["frame"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let centers: Vec<[f64; 3]> = d["centers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| f(o).try_into().unwrap())
        .collect();
    let pairs: Vec<(u16, u16)> = d["pairs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (p[0].as_u64().unwrap() as u16, p[1].as_u64().unwrap() as u16))
        .collect();
    let levers: Vec<[f64; N]> = d["levers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| f(o).try_into().unwrap())
        .collect();
    let mut col = Collision::new(
        chain,
        frame,
        &centers,
        f(&d["radii"]),
        pairs,
        levers,
        d["margin"].as_f64().unwrap(),
    )
    .unwrap();
    let ec: Vec<[f64; 3]> = d["env_centers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| f(o).try_into().unwrap())
        .collect();
    col.env = std::sync::Arc::new(Env::new(&ec, &f(&d["env_radii"])));
    let qs = |k: &str| -> Vec<Q> {
        d[k].as_array()
            .unwrap()
            .iter()
            .map(|o| Q::from_column_slice(&f(o)))
            .collect()
    };
    let targets = d["targets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| Rt::from_rows(&f(o)))
        .collect();
    let home = Q::from_column_slice(&f(&d["home"]));
    (col, qs("q"), targets, qs("free"), home)
}

fn bench(c: &mut Criterion) {
    let (col, q, targets, free, home) = load();
    let ch = col.chain.clone();
    let infl = vec![0.; col.n_spheres()];
    let mut i = 0usize;
    c.bench_function("fk", |b| {
        b.iter(|| {
            i += 1;
            black_box(ch.fk(&q[i % q.len()]))
        })
    });
    c.bench_function("jacobian", |b| {
        b.iter(|| {
            i += 1;
            black_box(ch.jacobian(&q[i % q.len()]))
        })
    });
    c.bench_function("ik_null_space_from_home", |b| {
        b.iter(|| {
            i += 1;
            black_box(ch.ik(
                &targets[i % targets.len()],
                &home,
                150,
                0.002,
                0.015,
                0.25,
                0.15,
                Some(&home),
                0.1,
            ))
        })
    });
    c.bench_function("collision_valid", |b| {
        b.iter(|| {
            i += 1;
            black_box(col.valid(&q[i % q.len()], &infl))
        })
    });
    c.bench_function("certify_free_sweep_from_home", |b| {
        b.iter(|| {
            i += 1;
            black_box(col.certify(&home, &free[i % free.len()], 0.003, false))
        })
    });
    c.bench_function("certify_free_sweep_from_home_rayon", |b| {
        b.iter(|| {
            i += 1;
            black_box(col.certify(&home, &free[i % free.len()], 0.003, true))
        })
    });
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/data/");
    if let (Ok(pb), Ok(fb), Ok(mb), Ok(meta)) = (
        std::fs::read(format!("{dir}plane_points.f64")),
        std::fs::read(format!("{dir}frame_points.f64")),
        std::fs::read(format!("{dir}frame_mask.u8")),
        std::fs::read_to_string(format!("{dir}frame.json")),
    ) {
        let pts: Vec<f64> = pb
            .chunks_exact(8)
            .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let planes: Vec<[f64; 4]> = (0..60)
            .map(|k| {
                let t = k as f64 * 0.001;
                [-0.07 + t, 0., 0.9975, 0.178]
            })
            .collect();
        c.bench_function("ransac_counts_60_candidates", |b| {
            b.iter(|| black_box(perc::plane_inlier_counts(&pts, &planes, 0.006, false)))
        });
        let p: Vec<f64> = fb
            .chunks_exact(8)
            .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let m: Vec<bool> = mb.iter().map(|v| *v != 0).collect();
        let meta: serde_json::Value = serde_json::from_str(&meta).unwrap();
        let (h, w) = (
            meta["h"].as_u64().unwrap() as usize,
            meta["w"].as_u64().unwrap() as usize,
        );
        c.bench_function("flood_fill_frame", |b| {
            b.iter(|| black_box(perc::flood_fill(&m, &p, h, w, 0.025)))
        });
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
