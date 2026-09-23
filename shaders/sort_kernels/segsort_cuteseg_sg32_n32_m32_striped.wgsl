
enable subgroups;

override WG: u32 = 32u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;
@group(0) @binding(5) var<storage, read> seg_ballots: array<u32>;

const N: u32 = 32u;
const M: u32 = 32u;
const WPT: u32 = 1u;

var<workgroup> smem_keys: array<u32, WG>;
var<workgroup> smem_vals: array<u32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_cuteseg_sg32_n32_m32_striped(
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 5u;

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
    let ballot = seg_ballots[slot];
    let group_size = select(0u, 32u - countLeadingZeros(ballot), is_active);                    // valid elements in the group (top set bit + 1)

    var key: u32 = 0xffffffffu;                        // sentinels sort to the top, dropped at store
    var value: u32 = 0xffffffffu;
    if is_active && local_tid < group_size {
        key = global_keys[group_base + local_tid];
        value = group_base + local_tid;
    }

    // per-lane segment bounds within the group, from the end-bit ballot
    let below = ballot & ((1u << local_tid) - 1u);
    let above = ballot & ~((1u << local_tid) - 1u);
    let seg_start_rel = 32u - countLeadingZeros(below);          // start (prev end + 1; 0 if none)
    let seg_end_rel = min(M - 1u, countTrailingZeros(above));    // end (terminal bit guarantees a hit)

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

    var keys: array<u32, 1>;
    var values: array<u32, 1>;
    keys[0] = smem_keys[sid];
    values[0] = smem_vals[sid];

    // striped (coalesced) store (WPT==1, no transpose)
    if is_active && local_tid < group_size {
        global_keys[group_base + local_tid] = keys[0];
        global_value_indices[group_base + local_tid] = values[0];
    }
}
