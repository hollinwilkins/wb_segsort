
enable subgroups;

override WG: u32 = 64u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 32u;
const M: u32 = 32u;
const WPT: u32 = 1u;

var<workgroup> smem_keys: array<u32, WG>;
var<workgroup> smem_vals: array<u32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_cute_sg64_n32_m32_striped(
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 5u;

    let bin_base = select(bin_offsets[BIN - 1u], 0u, BIN == 0u);
    let bin_count = bin_offsets[BIN] - bin_base;

    // WG == subgroup width (R), so one subgroup covers the whole workgroup and
    // packs R/M segments (each M consecutive lanes = one segment). This keeps a
    // full subgroup busy even for tiny segments, instead of one per workgroup.
    let local_tid = sid & (M - 1u);
    let seg_lane_base = sid - local_tid;            // my segment's base lane in the subgroup
    let wg_index = wg_id.x + wg_id.y * wg_dim.x;
    let global_seg = (wg_index * WG + sid) / M;

    let is_active = global_seg < bin_count;
    let slot = bin_base + select(0u, global_seg, is_active);   // clamp so the read is in-range
    let seg_id = bin_indices[slot];
    let seg_start = select(segments[seg_id - 1u], 0u, seg_id == 0u);
    let seg_end = segments[seg_id];
    let seg_size = select(0u, seg_end - seg_start, is_active);

    var key: u32;
    var value: u32;
    if is_active && local_tid < seg_size {
        key = global_keys[seg_start + local_tid];
        value = seg_start + local_tid;
    } else {
        key = 0xffffffffu;                          // sentinels sort to the top, dropped at store
        value = 0xffffffffu;
    }

    // M <= 32: 1 segment(s) per 32-bit word; scalar multisplit sorts them
    // together, bin_mask picks out this lane's segment within its word.
    let word = seg_lane_base >> 5u;
    let bin_mask = 0xffffffffu << (seg_lane_base & 31u);
    var ge_mask = (1u << (sid & 31u)) - 1u;
    for (var bit = 0u; bit < 32u; bit = bit + 1u) {
        let is_zero = (key & (1u << bit)) == 0u;
        let ballot0 = subgroupBallot(is_zero)[word];
        ge_mask = select(ge_mask | ballot0, ge_mask & ballot0, is_zero);
    }
    let rank = countOneBits(ge_mask & bin_mask);    // my sorted position within the segment

    smem_keys[seg_lane_base + rank] = key;
    smem_vals[seg_lane_base + rank] = value;
    workgroupBarrier();

    var keys: array<u32, 1>;
    var values: array<u32, 1>;
    keys[0] = smem_keys[sid];
    values[0] = smem_vals[sid];

    // striped (coalesced) store (WPT==1, no transpose)
    if is_active && local_tid < seg_size {
        global_keys[seg_start + local_tid] = keys[0];
        global_value_indices[seg_start + local_tid] = values[0];
    }
}
