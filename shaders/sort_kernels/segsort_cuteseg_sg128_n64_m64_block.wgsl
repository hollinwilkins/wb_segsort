
enable subgroups;

override WG: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;
@group(0) @binding(5) var<storage, read> seg_ballots: array<u32>;

const N: u32 = 64u;
const M: u32 = 64u;
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
fn segsort_cuteseg_sg128_n64_m64_block(
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 6u;

    let bin_base = select(bin_offsets[BIN - 1u], 0u, BIN == 0u);
    let group_count = bin_offsets[BIN] - bin_base;   // packed groups in this bin

    // R/M packed groups per subgroup; each group is M consecutive lanes and holds
    // several variable-length segments concatenated, bounded by its ballot.
    let local_tid = sid & (M - 1u);
    let m_group_base = sid - local_tid;              // my group's base lane in the subgroup
    let wg_index = wg_id.x + wg_id.y * wg_dim.x;
    let global_seg = (wg_index * WG + sid) / M;       // absolute packed-group index

    let is_active = global_seg < group_count;
    let slot = bin_base + select(0u, global_seg, is_active);   // clamp so reads are in-range
    let group_base = bin_indices[slot];               // global offset of this group's first element
    var bal = vec4<u32>(0u, 0u, 0u, 0u);
    bal.x = seg_ballots[slot * 2u + 0u];
    bal.y = seg_ballots[slot * 2u + 1u];
    let group_size = select(0u, hi_bit_plus1(bal), is_active);                    // valid elements in the group (top set bit + 1)

    var key: u32 = 0xffffffffu;                        // sentinels sort to the top, dropped at store
    var value: u32 = 0xffffffffu;
    if is_active && local_tid < group_size {
        key = global_keys[group_base + local_tid];
        value = group_base + local_tid;
    }

    // per-lane segment bounds within the group, from the end-bit ballot
    let below = bal & lane_mask_lt(local_tid);
    let above = bal & ~lane_mask_lt(local_tid);
    let seg_start_rel = hi_bit_plus1(below);                     // start (prev end + 1; 0 if none)
    let seg_end_rel = min(M - 1u, ctz128(above));               // end (terminal bit guarantees a hit)

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

    var keys: array<u32, 1>;
    var values: array<u32, 1>;
    keys[0] = smem_keys[sid];
    values[0] = smem_vals[sid];

    // block store
    for (var r = 0u; r < WPT; r = r + 1u) {
        let pos = local_tid * WPT + r;
        if is_active && pos < group_size {
            global_keys[group_base + pos] = keys[r];
            global_value_indices[group_base + pos] = values[r];
        }
    }
}
