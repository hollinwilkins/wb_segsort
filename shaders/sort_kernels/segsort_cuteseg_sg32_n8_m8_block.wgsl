
enable subgroups;

override WG: u32 = 32u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;
@group(0) @binding(5) var<storage, read> seg_ballots: array<u32>;
@group(0) @binding(6) var<storage, read> group_first_seg: array<u32>;

const N: u32 = 8u;
const M: u32 = 8u;
const WPT: u32 = 1u;

var<workgroup> smem_keys: array<u32, WG>;
var<workgroup> smem_vals: array<u32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_cuteseg_sg32_n8_m8_block(
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 3u;

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
    let ballot = seg_ballots[slot];
    let group_size = select(0u, 32u - countLeadingZeros(ballot), is_active);                    // valid elements in the group (top set bit + 1)

    // which segment within the group this lane belongs to, from the end-bit ballot
    let below = ballot & ((1u << local_tid) - 1u);
    let above = ballot & ~((1u << local_tid) - 1u);
    let seg_ordinal = countOneBits(below);                       // 0-based segment index in group
    let seg_start_rel = 32u - countLeadingZeros(below);          // start (prev end + 1; 0 if none)
    let seg_end_rel = min(M - 1u, countTrailingZeros(above));    // end (terminal bit guarantees a hit)

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

    // confine the multisplit popcount to this lane's segment inside its word
    let g_off = m_group_base & 31u;
    let gstart = g_off + seg_start_rel;
    let gend = g_off + seg_end_rel;
    let bin_mask = (0xffffffffu << gstart) & (0xffffffffu >> (31u - gend));

    let word = sid >> 5u;
    var ge_mask = (1u << (sid & 31u)) - 1u;
    for (var bit = 0u; bit < 32u; bit = bit + 1u) {
        let is_zero = (key & (1u << bit)) == 0u;
        let ballot0 = subgroupBallot(is_zero)[word];
        ge_mask = select(ge_mask | ballot0, ge_mask & ballot0, is_zero);
    }
    let rank = countOneBits(ge_mask & bin_mask);    // sorted position within the segment

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
