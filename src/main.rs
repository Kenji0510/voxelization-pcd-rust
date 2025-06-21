use rayon::prelude::*;
use std::{collections::HashMap, time::Instant};

use nohash_hasher::NoHashHasher;
use voxelization_myself::load_pcd::{self, Point};

#[derive(Default, Debug)]
struct VoxelStat {
    sum: [f32; 3],
    count: u32,
}

type FastMap<V> = HashMap<u64, V, std::hash::BuildHasherDefault<NoHashHasher<u64>>>;

#[inline]
fn morton3d(ix: u32, iy: u32, iz: u32) -> u64 {
    fn part1by2(n: u32) -> u64 {
        // 21bit → 63bit へ 0bit 埋め込み（マジックビット法）
        let mut x = n as u64 & 0x1fffff; // 21 bit
        x = (x | x << 32) & 0x1f00000000ffff;
        x = (x | x << 16) & 0x1f0000ff0000ff;
        x = (x | x << 8) & 0x100f00f00f00f00f;
        x = (x | x << 4) & 0x10c30c30c30c30c3;
        x = (x | x << 2) & 0x1249249249249249;
        x
    }
    part1by2(ix) | (part1by2(iy) << 1) | (part1by2(iz) << 2)
}

fn voxel_downsample(points: &[Point], voxel_size: f32) -> Vec<Point> {
    let (mut min_x, mut min_y, mut min_z) = (f32::INFINITY, f32::INFINITY, f32::INFINITY);
    for p in points {
        min_x = min_x.min(p.x);
        min_y = min_x.min(p.y);
        min_z = min_x.min(p.z);
    }

    let inv = 1.0 / voxel_size;

    let thread_maps: Vec<FastMap<VoxelStat>> = points
        .par_chunks(10_000)
        .map(|chunk| {
            let mut map: FastMap<VoxelStat> = FastMap::default();
            for p in chunk {
                let ix = ((p.x - min_x) * inv).floor() as u32;
                let iy = ((p.y - min_y) * inv).floor() as u32;
                let iz = ((p.z - min_z) * inv).floor() as u32;
                let key = morton3d(ix, iy, iz);
                let stat = map.entry(key).or_default();
                stat.sum[0] += p.x;
                stat.sum[1] += p.y;
                stat.sum[2] += p.z;
                stat.count += 1;
            }
            map
        })
        .collect();

    let mut global: FastMap<VoxelStat> = FastMap::default();
    for local in thread_maps {
        for (k, v) in local {
            let g = global.entry(k).or_default();
            g.sum[0] += v.sum[0];
            g.sum[1] += v.sum[1];
            g.sum[2] += v.sum[2];
            g.count += v.count;
        }
    }

    global
        .into_values()
        .map(|s| Point {
            x: s.sum[0] / s.count as f32,
            y: s.sum[1] / s.count as f32,
            z: s.sum[2] / s.count as f32,
        })
        .collect()

    // let mut grid: HashMap<(i32, i32, i32), VoxelStat> = HashMap::with_capacity(points.len());

    // for p in points {
    //     let ix = ((p.x - min_x) / voxel_size).floor() as i32;
    //     let iy = ((p.y - min_y) / voxel_size).floor() as i32;
    //     let iz = ((p.z - min_z) / voxel_size).floor() as i32;

    //     let entry = grid.entry((ix, iy, iz)).or_default();
    //     entry.sum[0] += p.x;
    //     entry.sum[1] += p.y;
    //     entry.sum[2] += p.z;
    //     entry.count += 1;
    // }

    // grid.values()
    //     .map(|stat| Point {
    //         x: stat.sum[0] / stat.count as f32,
    //         y: stat.sum[1] / stat.count as f32,
    //         z: stat.sum[2] / stat.count as f32,
    //     })
    //     .collect()
}

fn main() {
    println!("Hello, world!");

    let pcd_points = match load_pcd::load_pcd(
        // "/Users/kenji/workspace/Rust/rerun-sample/data/Laser_map/Laser_map_35.pcd",
        "/Users/kenji/Downloads/combined_120.pcd",
    ) {
        Ok(points) => points,
        Err(e) => {
            eprintln!("Error loading PCD file: {}", e);
            return;
        }
    };

    // let mut points = Vec::new();
    // for i in 0..10_000 {
    //     points.push(Point {
    //         x: (i as f32).sin() * 3.0,
    //         y: (i as f32).cos() * 5.0,
    //         z: (i as f32) * 0.001,
    //     });
    // }

    let start = Instant::now();

    let voxel_size = 0.1;
    let down = voxel_downsample(&pcd_points, voxel_size);

    let elpased = start.elapsed();

    println!("Processed time: {:?}", elpased);
    println!("Original points: {}", pcd_points.len());
    println!("After voxelization points: {}", down.len());

    let s = match load_pcd::save_pcd("aa", down) {
        Ok(()) => println!("Successfully to save!"),
        Err(e) => {
            eprintln!("Failed to save voxelized pcd!");
            return;
        }
    };
}
