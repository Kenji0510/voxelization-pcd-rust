use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use std::{collections::HashMap, num::NonZeroU64, time::Instant};
use wgpu::util::DeviceExt;

use nohash_hasher::NoHashHasher;
use voxelization_myself::load_pcd::{self, Point};

#[derive(Default, Debug)]
struct VoxelStat {
    sum: [f32; 3],
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    origin: [f32; 3],
    inv_vox: f32,
    scale: f32,
    inv_scale: f32,
    hash_mask: u32,
    _pad: f32,
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

fn convert_to_f32(pcd_points: &Vec<Point>) -> Vec<f32> {
    let mut points = Vec::new();
    for point in pcd_points {
        points.push(point.x);
        points.push(point.y);
        points.push(point.z);
    }

    points
}

fn get_min_value(points: &[Point]) -> (f32, f32, f32) {
    let (mut min_x, mut min_y, mut min_z) = (f32::INFINITY, f32::INFINITY, f32::INFINITY);
    for p in points {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        min_z = min_z.min(p.z);
    }
    (min_x, min_y, min_z)
}

fn voxel_downsample(points: &[Point], voxel_size: f32) -> Vec<Point> {
    let (mut min_x, mut min_y, mut min_z) = (f32::INFINITY, f32::INFINITY, f32::INFINITY);
    for p in points {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        min_z = min_z.min(p.z);
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
}

fn main() {
    println!("Hello, world!");

    let pcd_data = match load_pcd::load_pcd(
        "/Users/kenji/workspace/Rust/rerun-sample/data/Laser_map/Laser_map_35.pcd",
        // "/Users/kenji/Downloads/combined_120.pcd",
    ) {
        Ok(points) => points,
        Err(e) => {
            eprintln!("Error loading PCD file: {}", e);
            return;
        }
    };

    let pcd_points = convert_to_f32(&pcd_data);
    if pcd_points.is_empty() {
        println!("No arguments provided. Please provide a list of floats.");
        return;
    }

    let voxel_size = 0.05;
    let (min_x, min_y, min_z) = get_min_value(&pcd_data);
    // let down = voxel_downsample(&pcd_points, voxel_size);

    // let elpased = start.elapsed();

    // println!("Processed time: {:?}", elpased);
    println!("Original points: {}", pcd_points.len() / 3);
    // println!("After voxelization points: {}", down.len());

    // let s = match load_pcd::save_pcd("aa", down) {
    //     Ok(()) => println!("Successfully to save!"),
    //     Err(e) => {
    //         eprintln!("Failed to save voxelized pcd!");
    //         return;
    //     }
    // };

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptionsBase::default()))
            .expect("Failed to create adapter!");

    println!("Running on adapter: {:#?}", adapter.get_info());

    let downlevel_capabilities = adapter.get_downlevel_capabilities();
    if !downlevel_capabilities
        .flags
        .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
    {
        panic!("Adapter does not support compute shaders!");
    }

    let mut limits = wgpu::Limits::default();

    // println!(
    //     "max_storage_buffer_binding_size: {:?}",
    //     limits.max_storage_buffer_binding_size
    // );

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    }))
    .expect("Failed to create device!");

    let module = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    let capacity = (pcd_points.len() * 2).next_power_of_two();
    println!("Capasity: {:?}", &capacity);
    let buf_bytes = capacity * 4;
    let zero_key = vec![0u32; capacity];
    let uni = Uniforms {
        origin: [min_x, min_y, min_z],
        inv_vox: 1.0 / voxel_size,
        scale: 1000.0,
        inv_scale: 1.0 / 1000.0,
        hash_mask: (capacity - 1) as u32,
        _pad: 0.0,
    };
    let zero_i32 = vec![0i32; capacity as usize];

    let elem_bytes = std::mem::size_of::<i32>() as u64; // =4
    let buf_size = capacity as u64 * elem_bytes; // バイト長

    let table_keys = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("table_keys"),
        contents: bytemuck::cast_slice(&zero_key),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let sum_x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sum_x"),
        contents: bytemuck::cast_slice(&zero_i32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let sum_y_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sum_y"),
        contents: bytemuck::cast_slice(&zero_i32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let sum_z_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sum_z"),
        contents: bytemuck::cast_slice(&zero_i32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    // let table_sum = device.create_buffer(&wgpu::BufferDescriptor {
    //     label: Some("table_sum"),
    //     size: (capacity * 12) as u64,
    //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    //     mapped_at_creation: false,
    // });

    let table_cnt = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("table_cnt"),
        size: buf_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let fail_counter = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fail_counter"),
        contents: bytemuck::bytes_of(&0u32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniforms"),
        contents: bytemuck::bytes_of(&uni),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let input_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&pcd_points),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    //     label: None,
    //     size: input_data_buffer.size(),
    //     usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    //     mapped_at_creation: false,
    // });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    min_binding_size: None,
                    has_dynamic_offset: false,
                },
                count: None,
            },
            // 1: table_keys  (read-write)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 2: table_sum
            // wgpu::BindGroupLayoutEntry {
            //     binding: 2,
            //     visibility: wgpu::ShaderStages::COMPUTE,
            //     ty: wgpu::BindingType::Buffer {
            //         ty: wgpu::BufferBindingType::Storage { read_only: false },
            //         has_dynamic_offset: false,
            //         min_binding_size: None,
            //     },
            //     count: None,
            // },
            // 2: sum_x
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 3: sum_y
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 4: sum_z
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 3: table_cnt
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 4: fail_counter
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 5: Uniforms
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(std::num::NonZeroU64::new(32).unwrap()),
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel_bind"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_data_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: table_keys.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: sum_x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: sum_y_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: sum_z_buf.as_entire_binding(),
            },
            // wgpu::BindGroupEntry {
            //     binding: 2,
            //     resource: table_sum.as_entire_binding(),
            // },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: table_cnt.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: fail_counter.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: uniform_buf.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // let mut compilation_op = wgpu::PipelineCompilationOptions::default();
    // compilation_op.zero_initialize_workgroup_memory = false;

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        // compilation_options: compilation_op,
        cache: None,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
    //     label: None,
    //     timestamp_writes: None,
    // });

    //------------------------------------------------------------------
    // ① ワークグループ数を計算
    //------------------------------------------------------------------
    const WG_SIZE: u32 = 256;
    let workgroups = ((pcd_points.len() as u32) + WG_SIZE - 1) / WG_SIZE; // 切り上げ

    //------------------------------------------------------------------
    // ② Compute Pass: パイプライン + BindGroup + dispatch
    //------------------------------------------------------------------
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel_compute"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    } // ← drop で自動 end_pass

    //------------------------------------------------------------------
    // ③ fail_counter を CPU にコピーするステージングバッファ
    //------------------------------------------------------------------
    let staging_fail = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_fail"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // コピー命令をエンコード
    encoder.copy_buffer_to_buffer(
        &fail_counter, // src
        0,
        &staging_fail, // dst
        0,
        std::mem::size_of::<u32>() as u64,
    );

    let byte_len_i32 = (capacity * std::mem::size_of::<i32>()) as u64;
    let byte_len_u32 = (capacity * std::mem::size_of::<u32>()) as u64;

    // 1. ステージングバッファ作成
    fn staging(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }
    let st_sumx = staging(&device, "st_sumx", byte_len_i32);
    let st_sumy = staging(&device, "st_sumy", byte_len_i32);
    let st_sumz = staging(&device, "st_sumz", byte_len_i32);
    let st_cnt = staging(&device, "st_cnt", byte_len_u32);
    let st_keys = staging(&device, "st_keys", byte_len_u32);

    // 2. コピー命令をエンコード（encoder は新しく作り直す）
    encoder.copy_buffer_to_buffer(&sum_x_buf, 0, &st_sumx, 0, byte_len_i32);
    encoder.copy_buffer_to_buffer(&sum_y_buf, 0, &st_sumy, 0, byte_len_i32);
    encoder.copy_buffer_to_buffer(&sum_z_buf, 0, &st_sumz, 0, byte_len_i32);
    encoder.copy_buffer_to_buffer(&table_cnt, 0, &st_cnt, 0, byte_len_u32);
    encoder.copy_buffer_to_buffer(&table_keys, 0, &st_keys, 0, byte_len_u32);

    device.poll(wgpu::PollType::Wait).unwrap();

    let start = Instant::now();

    //------------------------------------------------------------------
    // ④ GPU キューへ submit
    //------------------------------------------------------------------
    queue.submit(Some(encoder.finish()));

    // GPU 完了を保証（ポーリング）
    device.poll(wgpu::PollType::Wait).unwrap();

    // let elpased = start.elapsed();

    //------------------------------------------------------------------
    // ⑤ fail_counter をマップしてチェック
    //------------------------------------------------------------------
    let slice = staging_fail.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    device.poll(wgpu::PollType::Wait).unwrap(); // ブロッキング待機

    let data = slice.get_mapped_range();
    let fail_val = *bytemuck::from_bytes::<u32>(&data);
    drop(data);
    staging_fail.unmap();

    assert!(
        fail_val == 0,
        "Hash table overflowed (fail_cnt = {}), consider larger capacity!",
        fail_val
    );

    println!("Compute shader finished — fail_cnt = {}", fail_val);

    // 3. マップ & スライス
    macro_rules! map_slice {
        ($buf:ident) => {{
            let slice = $buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| ());
            device.poll(wgpu::PollType::Wait).unwrap();
            slice.get_mapped_range()
        }};
    }
    let m_sumx = map_slice!(st_sumx);
    let m_sumy = map_slice!(st_sumy);
    let m_sumz = map_slice!(st_sumz);
    let m_cnt = map_slice!(st_cnt);
    let m_keys = map_slice!(st_keys);

    // 4. 重心計算
    let mut centroids = Vec::new();
    let sumx_i32: &[i32] = bytemuck::cast_slice(&m_sumx);
    let sumy_i32: &[i32] = bytemuck::cast_slice(&m_sumy);
    let sumz_i32: &[i32] = bytemuck::cast_slice(&m_sumz);

    let cnt_u32: &[u32] = bytemuck::cast_slice(&m_cnt);
    // println!("cnt_u32: {:?}", cnt_u32.len());
    let keys_u32: &[u32] = bytemuck::cast_slice(&m_keys);
    // println!("keys_u32: {:?}", keys_u32);

    // ❷ capacity 要素あるか確認
    assert_eq!(sumx_i32.len(), capacity as usize);

    // ❸ ループは “i番目の要素” を直接参照
    for i in 0..capacity as usize {
        let key = keys_u32[i];
        if key == 0 {
            continue;
        }

        let cnt = cnt_u32[i] as f32;
        let sumx = sumx_i32[i] as f32 * uni.inv_scale;
        let sumy = sumy_i32[i] as f32 * uni.inv_scale;
        let sumz = sumz_i32[i] as f32 * uni.inv_scale;

        centroids.push(Point {
            x: sumx / cnt,
            y: sumy / cnt,
            z: sumz / cnt,
        });
    }

    drop(m_sumx);
    drop(m_sumy);
    drop(m_sumz);
    drop(m_cnt);
    drop(m_keys);

    // アンマップ
    st_sumx.unmap();
    st_sumy.unmap();
    st_sumz.unmap();
    st_cnt.unmap();
    st_keys.unmap();

    let elpased = start.elapsed();

    println!("Processed time: {:?}", elpased);
    println!("Voxelized points: {}", centroids.len());

    // let s = match load_pcd::save_pcd("aa", centroids) {
    //     Ok(()) => println!("Successfully to save!"),
    //     Err(e) => {
    //         eprintln!("Failed to save voxelized pcd!");
    //         return;
    //     }
    // };
}
