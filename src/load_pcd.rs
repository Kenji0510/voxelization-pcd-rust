// use std::fmt::Result;
use anyhow::Result;

use pcd_rs::{PcdDeserialize, PcdSerialize, Reader, WriterInit};

#[derive(Clone, Copy, Debug, PcdDeserialize, PcdSerialize)]
pub struct Point {
    // #[pcd(rename = "new_x")]
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

// #[derive(Debug, PcdSerialize)]
// pub struct ExportPoint {
//     pub x: f32,
//     pub y: f32,
//     pub z: f32,
// }

pub fn load_pcd(file_path: &str) -> Result<Vec<Point>, Box<dyn std::error::Error>> {
    let reader = match Reader::open(file_path) {
        Ok(reader) => reader,
        Err(e) => {
            eprintln!("Error opening PCD file: {}", e);
            return Err(e.into());
        }
    };

    let points: Vec<Point> = match reader.collect() {
        Ok(points) => points,
        Err(e) => {
            eprintln!("Error reading PCD file: {}", e);
            return Err(e.into());
        }
    };

    Ok(points)
}

pub fn save_pcd(file_path: &str, points: Vec<Point>) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = WriterInit {
        width: 1,
        height: points.len() as u64,
        viewpoint: Default::default(),
        data_kind: pcd_rs::DataKind::Ascii,
        schema: None,
    }
    .create("/Users/kenji/workspace/Rust/voxelization-myself/data/export-back-subtracted.pcd")?;

    for point in points {
        writer.push(&point)?;
    }

    writer.finish()?;
    Ok(())
}

pub fn convert_to_point(data: &[f32], max_num: usize) -> Vec<Point> {
    let mut points = Vec::new();
    for i in (0..max_num).step_by(3) {
        if i + 2 < max_num {
            points.push(Point {
                x: data[i],
                y: data[i + 1],
                z: data[i + 2],
            });
        }
    }

    points
}
