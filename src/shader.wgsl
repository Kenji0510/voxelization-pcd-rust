struct Point { x: f32, y: f32, z: f32 };
// struct Uniforms { origin: vec3<f32>, inv_vox: f32 };
struct Uniforms {
    origin   : vec3<f32>,
    inv_vox  : f32,
    scale    : f32,
    inv_scale: f32,
    hash_mask: u32, // capasity
};

@group(0) @binding(0) var<storage, read_write>  points : array<Point>;
@group(0) @binding(1) var<storage, read_write> table_keys : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> sum_x : array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> sum_y : array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> sum_z : array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> table_cnt  : array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> fail_cnt   : atomic<u32>;
@group(0) @binding(7) var<uniform>             uni        : Uniforms;
@group(0) @binding(8) var<storage, read_write> centroids_num : atomic<u32>;

// const HASH_MASK : u32 = 1023u; /* capacity - 1 を Rust で `#define` 的に埋め込む */
// override HASH_MASK: u32;

var<workgroup> w_key  : array<u32, 256>;
var<workgroup> w_sum  : array<vec3<f32>, 256>;
var<workgroup> w_cnt  : array<u32, 256>;

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
  if (idx >= arrayLength(&points)) { return; }

  // --- KeyGen --------------------------------
  let p = points[idx];
  let p_vec = vec3<f32>(p.x, p.y, p.z);
  let v = (p_vec - uni.origin) * uni.inv_vox;
  let key = morton3d(u32(floor(v.x)),u32(floor(v.y)),u32(floor(v.z)));

  w_key[lid.x] = key;
  w_sum[lid.x] = vec3<f32>(p.x, p.y, p.z);
  w_cnt[lid.x] = 1u;
  workgroupBarrier();

  // --- WG 内 Reduction (隣り合う同キーをまとめる) ----
  var step:u32 = 128u;
  loop {
      if (lid.x < step && w_key[lid.x] == w_key[lid.x + step]) {
          w_sum[lid.x] += w_sum[lid.x + step];
          w_cnt[lid.x] += w_cnt[lid.x + step];
          w_key[lid.x + step] = 0xffffffffu;   // 無効化
      }
      workgroupBarrier();
      if (step == 1u) { break; }
      step = step >> 1u;
  }

  // --- Global Hash Insert ---------------------
  if (w_key[lid.x] != 0xffffffffu) {
        var slot = w_key[lid.x] & uni.hash_mask;
        for (var tries:u32=0u; tries<64u; tries=tries+1u) {
            let prev = atomicCompareExchangeWeak(&table_keys[slot], 0u, w_key[lid.x]);
            if (prev.exchanged || prev.old_value == w_key[lid.x]) {
                // 座標を整数スケールへ → i32
                let sx = i32(round(w_sum[lid.x].x * uni.scale));
                let sy = i32(round(w_sum[lid.x].y * uni.scale));
                let sz = i32(round(w_sum[lid.x].z * uni.scale));

                atomicAdd(&sum_x[slot], sx);
                atomicAdd(&sum_y[slot], sy);
                atomicAdd(&sum_z[slot], sz);
                atomicAdd(&table_cnt[slot], w_cnt[lid.x]);  // u32
                return;
            }
            slot = (slot + 1u) & uni.hash_mask;
        }
        // 挿入失敗
        atomicAdd(&fail_cnt, 1u);
  }
}

@compute @workgroup_size(256)
fn centroid_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uni.hash_mask) {
        return;
    }

    let k = atomicLoad(&table_keys[i]);
    if (k == 0u) {
        return;
    }
    let dst = atomicAdd(&centroids_num, 1u);
    let c  = f32(atomicLoad(&table_cnt[i]));
    let sx = f32(atomicLoad(&sum_x[i])) * uni.inv_scale;
    let sy = f32(atomicLoad(&sum_y[i])) * uni.inv_scale;
    let sz = f32(atomicLoad(&sum_z[i])) * uni.inv_scale;

    points[dst] = Point(sx / c, sy / c, sz / c);
}
