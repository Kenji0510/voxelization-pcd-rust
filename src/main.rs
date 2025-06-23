use bytemuck::{Pod, Zeroable};
use std::time::Instant;
use voxelization_myself::load_pcd::{self, Point};
use wgpu::util::DeviceExt;

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

fn main() {
    println!("Hello, world!");

    let pcd_data = match load_pcd::load_pcd(
        // "/Users/kenji/workspace/Rust/rerun-sample/data/Laser_map/Laser_map_35.pcd",
        "/Users/kenji/Downloads/combined_120.pcd",
        // "/home/kenji/workspace/Rust/voxelization-pcd-rust/data/combined_120.pcd",
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
    println!("Original points: {}", pcd_points.len() / 3);

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
    limits.max_storage_buffers_per_shader_stage = 9;

    println!(
        "max_storage_buffer_binding_size: {:?}",
        limits.max_storage_buffers_per_shader_stage
    );

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

    let centroids_num = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("centroids_num"),
        contents: bytemuck::bytes_of(&0u32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let points_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&pcd_points),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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
            // 5: table_cnt
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
            // 6: fail_counter
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
            // 7: Uniforms
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
            // 8. centroids_num
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
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
                resource: points_data_buffer.as_entire_binding(),
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
            wgpu::BindGroupEntry {
                binding: 8,
                resource: centroids_num.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let pipeline2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("centroid_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    //------------------------------------------------------------------
    // ① ワークグループ数を計算
    //------------------------------------------------------------------
    const WG_SIZE: u32 = 256;
    let workgroups = ((pcd_points.len() as u32) + WG_SIZE - 1) / WG_SIZE; // 切り上げ

    let wg_cnt = (uni.hash_mask + 1 + WG_SIZE - 1) / WG_SIZE;

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

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("centroid_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline2);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(wg_cnt, 1, 1);
    }

    //------------------------------------------------------------------
    // ③ fail_counter を CPU にコピーするステージングバッファ
    //------------------------------------------------------------------
    let staging_fail = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_fail"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let staging_centroids_num = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_centroids_num"),
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

    encoder.copy_buffer_to_buffer(
        &centroids_num, // src
        0,
        &staging_centroids_num, // dst
        0,
        std::mem::size_of::<u32>() as u64,
    );

    // 1. ステージングバッファ作成
    fn staging(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }
    let d_buffer = staging(&device, "d_buffer", points_data_buffer.size());

    // 2. コピー命令をエンコード（encoder は新しく作り直す）
    encoder.copy_buffer_to_buffer(
        &points_data_buffer,
        0,
        &d_buffer,
        0,
        points_data_buffer.size(),
    );

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

    let cent_slice = staging_centroids_num.slice(..);
    cent_slice.map_async(wgpu::MapMode::Read, |_| ());
    device.poll(wgpu::PollType::Wait).unwrap(); // ブロッキング待機

    let cent_data = cent_slice.get_mapped_range();
    let cent_num_value = *bytemuck::from_bytes::<u32>(&cent_data);
    drop(cent_data);
    staging_centroids_num.unmap();

    assert!(
        fail_val == 0,
        "Hash table overflowed (fail_cnt = {}), consider larger capacity!",
        fail_val
    );

    println!("Compute shader finished — fail_cnt = {}", fail_val);
    println!("Number of Centroids: {}", &cent_num_value);

    // 3. マップ & スライス
    macro_rules! map_slice {
        ($buf:ident) => {{
            let slice = $buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| ());
            device.poll(wgpu::PollType::Wait).unwrap();
            slice.get_mapped_range()
        }};
    }

    let m_d_buffer = map_slice!(d_buffer);

    let centroids_ref: &[f32] = bytemuck::cast_slice(&m_d_buffer);

    let mut centroids = Vec::new();

    for i in 0..cent_num_value as usize {
        centroids.push(Point {
            x: centroids_ref[i * 3],
            y: centroids_ref[i * 3 + 1],
            z: centroids_ref[i * 3 + 2],
        })
    }

    let elpased = start.elapsed();
    // 4. 重心計算
    drop(m_d_buffer);

    // // アンマップ
    d_buffer.unmap();

    println!("Processed time: {:?}", elpased);
    println!("Voxelized points: {}", centroids.len());

    match load_pcd::save_pcd("aa", centroids) {
        Ok(()) => println!("Successfully to save!"),
        Err(_) => {
            eprintln!("Failed to save voxelized pcd!");
            return;
        }
    };
}
