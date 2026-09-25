
enable subgroups;

override WG: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;
@group(0) @binding(5) var<storage, read> seg_ballots: array<u32>;
@group(0) @binding(6) var<storage, read> group_first_seg: array<u32>;

const N: u32 = 128u;
const M: u32 = 128u;
const WPT: u32 = 1u;

var<workgroup> smem_keys: array<u32, WG>;
var<workgroup> smem_vals: array<u32, WG>;

fn lane_mask_lt(sid: u32) -> vec4<u32> {
    var m = vec4<u32>(0u, 0u, 0u, 0u);
    if (sid >= 32u) { m.x = 0xffffffffu; } else { m.x = (1u << sid) - 1u; }
    if (sid >= 64u) { m.y = 0xffffffffu; } else if (sid > 32u) { m.y = (1u << (sid - 32u)) - 1u; }
    if (sid >= 96u) { m.z = 0xffffffffu; } else if (sid > 64u) { m.z = (1u << (sid - 64u)) - 1u; }
    if (sid >= 128u) { m.w = 0xffffffffu; } else if (sid > 96u) { m.w = (1u << (sid - 96u)) - 1u; }
    return m;
}

fn ballot_popc(v: vec4<u32>) -> u32 {
    let c = countOneBits(v);
    return c.x + c.y + c.z + c.w;
}

// lowest set bit index across the 128-bit ballot (128 if none)
fn ctz128(v: vec4<u32>) -> u32 {
    if (v.x != 0u) { return       countTrailingZeros(v.x); }
    if (v.y != 0u) { return 32u + countTrailingZeros(v.y); }
    if (v.z != 0u) { return 64u + countTrailingZeros(v.z); }
    return 96u + countTrailingZeros(v.w);
}

// (highest set bit index + 1) across the ballot; 0 if all zero
fn hi_bit_plus1(v: vec4<u32>) -> u32 {
    if (v.w != 0u) { return 96u + (32u - countLeadingZeros(v.w)); }
    if (v.z != 0u) { return 64u + (32u - countLeadingZeros(v.z)); }
    if (v.y != 0u) { return 32u + (32u - countLeadingZeros(v.y)); }
    if (v.x != 0u) { return       (32u - countLeadingZeros(v.x)); }
    return 0u;
}

@compute @workgroup_size(WG, 1, 1)
fn segsort_cuteseg_sg128_n128_m128_striped(
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 7u;

    // Groups are indexed from 0 by wb_nf_bin; bin_offsets[BIN] holds the group count.
    let group_count = bin_offsets[BIN];

    // R/M packed groups per subgroup; each group is M consecutive lanes and holds
    // several variable-length segments concatenated, bounded by its ballot. The
    // segments are size-sorted (not contiguous), so each lane resolves its own
    // global key position through bin_indices + segments[] (see addr_block).
    let local_tid = sid & (M - 1u);
    let m_group_base = sid - local_tid;              // my group's base lane in the subgroup
    let wg_index = wg_id.x + wg_id.y * wg_dim.x;
    let global_seg = (wg_index * WG + sid) / M;       // absolute packed-group index

    let is_active = global_seg < group_count;
    let slot = select(0u, global_seg, is_active);     // group index (clamped so reads are in-range)
    var bal = vec4<u32>(0u, 0u, 0u, 0u);
    bal.x = seg_ballots[slot * 4u + 0u];
    bal.y = seg_ballots[slot * 4u + 1u];
    bal.z = seg_ballots[slot * 4u + 2u];
    bal.w = seg_ballots[slot * 4u + 3u];
    let group_size = select(0u, hi_bit_plus1(bal), is_active);                    // valid elements in the group (top set bit + 1)

    // which segment within the group this lane belongs to, from the end-bit ballot
    let below = bal & lane_mask_lt(local_tid);
    let above = bal & ~lane_mask_lt(local_tid);
    let seg_ordinal = ballot_popc(below);                        // 0-based segment index in group
    let seg_start_rel = hi_bit_plus1(below);                     // start (prev end + 1; 0 if none)
    let seg_end_rel = min(M - 1u, ctz128(above));               // end (terminal bit guarantees a hit)

    // resolve the lane's global key position through bin_indices + segments[]
    let in_group = is_active && local_tid < group_size;
    var gpos = 0u;
    if in_group {
        let seg_id = bin_indices[group_first_seg[slot] + seg_ordinal];
        let seg_gstart = select(0u, segments[seg_id - 1u], seg_id > 0u);
        gpos = seg_gstart + (local_tid - seg_start_rel);
    }

    var key: u32 = 0xffffffffu;                        // sentinels sort to the top, dropped at store
    var value: u32 = 0xffffffffu;
    if in_group {
        key = global_keys[gpos];
        value = gpos;                                  // original position (value_indices are identity)
    }

    // confine the multisplit popcount to this lane's segment
    let seg_lane_base = m_group_base + seg_start_rel;
    let bin_mask = lane_mask_lt(m_group_base + seg_end_rel + 1u) & ~lane_mask_lt(seg_lane_base);
    var ge_mask = lane_mask_lt(sid);
    for (var bit = 0u; bit < 32u; bit = bit + 1u) {
        let is_zero = (key & (1u << bit)) == 0u;
        let ballot0 = subgroupBallot(is_zero);
        ge_mask = select(ge_mask | ballot0, ge_mask & ballot0, is_zero);
    }
    let rank = ballot_popc(ge_mask & bin_mask);     // sorted position within the segment

    let dst = m_group_base + seg_start_rel + rank;     // dense sorted slot within the subgroup
    smem_keys[dst] = key;
    smem_vals[dst] = value;
    workgroupBarrier();

    // Lane sid picks up the element sorted into group-slot local_tid and writes it
    // back to that slot's own global position (same gpos as the load: within a
    // segment, group-slot order maps 1:1 to the segment's global range).
    let out_key = smem_keys[sid];
    let out_val = smem_vals[sid];
    if in_group {
        global_keys[gpos] = out_key;
        global_value_indices[gpos] = out_val;
    }
}
