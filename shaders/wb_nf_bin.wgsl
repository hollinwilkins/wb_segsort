// Next-fit-bin packing pass for the cuteseg sort family.
//
// Runs AFTER wb_bin. Consumes wb_bin's size-sorted `bin_indices` (segment ids)
// for the cuteseg-target bin range [FIRST_BIN, LAST_BIN] (all segment lengths
// 2 < len <= M) and greedily packs consecutive segments into groups of capacity
// M. It does NOT move any key data: the cuteseg kernel gathers/scatters through
// `bin_indices` + `segments[]`, so the only new per-group outputs are:
//   group_first_seg[g] : index into bin_indices of the group's FIRST segment
//   seg_ballots[g*W+w]  : segment-end ballot (bit set == inclusive last element
//                         of a segment, native LSB=lane order; W = max(1,M/32))
//
// Parallel scheme: thread t packs K WHOLE segments (segments never straddle a
// thread boundary), running independent next-fit -> local groups. A thread's
// trailing group may be under-full; there is no cross-thread merge. Group slots
// are reserved with an atomic counter (group order is irrelevant -- each group
// sorts into its own segments' global positions).

override WG_SIZE: u32 = 256u;   // threads per workgroup for the pack pass
override M: u32 = 32u;          // group capacity == cuteseg N
override K: u32 = 8u;           // whole segments packed per thread
override FIRST_BIN: u32 = 2u;   // first cuteseg-target bin (len > 2)
override LAST_BIN: u32 = 5u;    // last cuteseg-target bin (== log2(M))
override BIN: u32 = 5u;         // cuteseg's bin_offsets group-count slot (log2(M))
override CUTESEG_WG: u32 = 32u; // cuteseg sort workgroup size (subgroup width R)

struct Config {
    segments_len: u32,
    max_bin: u32,
}

struct DispatchSize {
    x: u32,
    y: u32,
    z: u32,
}

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> segments: array<u32>;
@group(0) @binding(2) var<storage, read_write> bin_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> bin_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> group_first_seg: array<u32>;
@group(0) @binding(5) var<storage, read_write> seg_ballots: array<u32>;
@group(0) @binding(6) var<storage, read_write> group_counter: atomic<u32>;
@group(0) @binding(7) var<storage, read_write> dispatch: array<DispatchSize>;

fn words() -> u32 {
    return max(1u, M / 32u);
}

fn seg_len(seg_id: u32) -> u32 {
    let seg_end = segments[seg_id];
    let seg_start = select(0u, segments[seg_id - 1u], seg_id > 0u);
    return seg_end - seg_start;
}

// Reserve a group slot and write its outputs.
fn emit_group(first_slot: u32, ballot: ptr<function, array<u32, 4>>) {
    let g = atomicAdd(&group_counter, 1u);
    group_first_seg[g] = first_slot;
    let W = words();
    for (var w = 0u; w < W; w = w + 1u) {
        seg_ballots[g * W + w] = (*ballot)[w];
    }
}

@compute @workgroup_size(1, 1, 1)
fn nf_clear() {
    atomicStore(&group_counter, 0u);
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn nf_pack(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // Slice of bin_indices this pass owns (all cuteseg-target segments, size-sorted).
    let slice_base = select(bin_offsets[FIRST_BIN - 1u], 0u, FIRST_BIN == 0u);
    let slice_end = bin_offsets[LAST_BIN];
    let slice_len = slice_end - slice_base;

    let my_first = gid.x * K;                 // first target-slice segment for this thread
    if my_first >= slice_len {
        return;
    }
    let my_count = min(K, slice_len - my_first);

    var ballot = array<u32, 4>(0u, 0u, 0u, 0u);
    var cur_fill = 0u;
    var group_start = my_first;               // target-slice offset of current group's first seg

    for (var i = 0u; i < my_count; i = i + 1u) {
        let slot = slice_base + my_first + i;
        let len = seg_len(bin_indices[slot]);

        // Close the current (non-empty) group if this segment would overflow it.
        if cur_fill > 0u && cur_fill + len > M {
            emit_group(slice_base + group_start, &ballot);
            ballot = array<u32, 4>(0u, 0u, 0u, 0u);
            cur_fill = 0u;
            group_start = my_first + i;
        }

        // Mark this segment's inclusive end bit within the group.
        let end_pos = cur_fill + len - 1u;
        ballot[end_pos >> 5u] = ballot[end_pos >> 5u] | (1u << (end_pos & 31u));
        cur_fill = cur_fill + len;
    }

    // Flush the trailing (possibly under-full) group.
    if cur_fill > 0u {
        emit_group(slice_base + group_start, &ballot);
    }
}

@compute @workgroup_size(1, 1, 1)
fn nf_schedule() {
    let group_count = atomicLoad(&group_counter);

    // cuteseg reads its group count from bin_offsets[BIN] with bin_base = 0.
    bin_offsets[BIN] = group_count;

    // Indirect dispatch for the cuteseg sort: groups_per_wg = CUTESEG_WG / M.
    let groups_per_wg = max(CUTESEG_WG / M, 1u);
    let groups = select(
        (group_count + groups_per_wg - 1u) / groups_per_wg,
        0u,
        group_count == 0u,
    );

    var x = groups;
    var y = 1u;
    if x > 65535u {
        y = (groups + 65535u - 1u) / 65535u;
        x = 65535u;
    }
    dispatch[BIN] = DispatchSize(x, y, 1u);
}
