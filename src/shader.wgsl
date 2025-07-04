struct Point { x: f32, y: f32, z: f32 };
struct Uniforms {
    origin   : vec3<f32>,
    inv_vox  : f32,
    scale    : f32,
    inv_scale: f32,
    hash_mask: u32, // capasity
};

@group(0) @binding(0) var<storage, read>  in_back_pts : array<Point>;
@group(0) @binding(1) var<storage, read>  in_target_pts : array<Point>;
@group(0) @binding(2) var<storage, read_write>  out_pts : array<Point>;
@group(0) @binding(3) var<storage, read_write> bg_table_keys : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> table_cnt  : array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> fail_cnt   : atomic<u32>;
@group(0) @binding(6) var<uniform>             uni        : Uniforms;
@group(0) @binding(7) var<storage, read_write> out_pts_num : atomic<u32>;

fn part1by2(n: u32) -> u32 {
    // “n” のビットを 0b00n00n00n… の形に広げる関数
    var x = n & 0x000003ffu;
    x = (x | (x << 16)) & 0xFF0000FFu;
    x = (x | (x << 8))  & 0x0300F00Fu;
    x = (x | (x << 4))  & 0x030C30C3u;
    x = (x | (x << 2))  & 0x09249249u;
    return x;
}
fn morton3d(ix:u32,iy:u32,iz:u32)->u32{
  return part1by2(ix) | (part1by2(iy)<<1u) | (part1by2(iz)<<2u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {

  let idx = gid.x;
  if (idx >= arrayLength(&in_back_pts)) { return; }

  // --- KeyGen --------------------------------
  let p = in_back_pts[idx];
  let p_vec = vec3<f32>(p.x, p.y, p.z);
  let v = (p_vec - uni.origin) * uni.inv_vox;
  let key = morton3d(u32(floor(v.x)),u32(floor(v.y)),u32(floor(v.z)));

  atomicStore(&bg_table_keys[key], 1u);
}

@compute @workgroup_size(256)
fn bg_subtraction_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    // if (i >= uni.hash_mask) {
    //     return;
    // }

    if (i >= arrayLength(&in_back_pts)) { return; }

    let p = in_target_pts[i];
    let p_vec = vec3<f32>(p.x, p.y, p.z);
    let v = (p_vec - uni.origin) * uni.inv_vox;
    let key = morton3d(u32(floor(v.x)),u32(floor(v.y)),u32(floor(v.z)));

    if (atomicLoad(&bg_table_keys[key]) == 0u) {
        let dst = atomicAdd(&out_pts_num, 1u);
        out_pts[dst] = p;
    }

    return;

    // let k = atomicLoad(&bg_table_keys[i]);
    // if (k == 0u) {
    //     return;
    // }
    // let dst = atomicAdd(&out_cnt, 1u);
    // let c  = f32(atomicLoad(&table_cnt[i]));
    // let sx = f32(atomicLoad(&sum_x[i])) * uni.inv_scale;
    // let sy = f32(atomicLoad(&sum_y[i])) * uni.inv_scale;
    // let sz = f32(atomicLoad(&sum_z[i])) * uni.inv_scale;

    // in_pts[dst] = Point(sx / c, sy / c, sz / c);
}
