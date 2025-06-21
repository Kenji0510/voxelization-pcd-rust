use std::{collections::HashMap, time::Instant};

use voxelization_myself::load_pcd::{self, Point};

#[derive(Default, Debug)]
struct VoxelStat {
    sum: [f32; 3],
    count: u32,
}

fn voxel_downsample(points: &[Point], voxel_size: f32) -> Vec<Point> {
    let (mut min_x, mut min_y, mut min_z) = (f32::INFINITY, f32::INFINITY, f32::INFINITY);
    for p in points {
        min_x = min_x.min(p.x);
        min_y = min_x.min(p.y);
        min_z = min_x.min(p.z);
    }

    let mut grid: HashMap<(i32, i32, i32), VoxelStat> = HashMap::with_capacity(points.len());

    for p in points {
        let ix = ((p.x - min_x) / voxel_size).floor() as i32;
        let iy = ((p.y - min_y) / voxel_size).floor() as i32;
        let iz = ((p.z - min_z) / voxel_size).floor() as i32;

        let entry = grid.entry((ix, iy, iz)).or_default();
        entry.sum[0] += p.x;
        entry.sum[1] += p.y;
        entry.sum[2] += p.z;
        entry.count += 1;
    }

    grid.values()
        .map(|stat| Point {
            x: stat.sum[0] / stat.count as f32,
            y: stat.sum[1] / stat.count as f32,
            z: stat.sum[2] / stat.count as f32,
        })
        .collect()
}

fn main() {
    println!("Hello, world!");

    let pcd_points = match load_pcd::load_pcd(
        "/Users/kenji/workspace/Rust/rerun-sample/data/Laser_map/Laser_map_35.pcd",
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
