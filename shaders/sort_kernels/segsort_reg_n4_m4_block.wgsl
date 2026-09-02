
enable subgroups;

override WG: u32 = 4u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 4u;
const M: u32 = 4u;
const WPT: u32 = 1u;

@compute @workgroup_size(WG, 1, 1)
fn segsort_reg_n4_m4_block(
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    const BIN: u32 = 3u;

    let bin_base = select(bin_offsets[BIN - 1u], 0u, BIN == 0u);
    let bin_count = bin_offsets[BIN] - bin_base;

    let local_tid = sid & (M - 1u);
    let global_seg = (wg_id.x * WG + lid) / M;

    let is_active = global_seg < bin_count;
    let slot = bin_base + select(0u, global_seg, is_active);   // clamp so the read is in-range
    let seg_id = bin_indices[slot];
    let seg_start = select(segments[seg_id - 1u], 0u, seg_id == 0u);
    let seg_end = segments[seg_id];
    let seg_size = select(0u, seg_end - seg_start, is_active);

    var keys: array<u32, 1>;
    var values: array<u32, 1>;

    for (var r = 0u; r < WPT; r = r + 1u) {
        let pos = local_tid * WPT + r;
        if is_active && pos < seg_size {
            keys[r] = global_keys[seg_start + pos];
            values[r] = seg_start + pos;
        } else {
            keys[r] = 0xffffffffu;
            values[r] = 0xffffffffu;
        }
    }

    // exch_intxn(tmask:1,swbit:0,wpt:1)
    {
    let tmp_0 = subgroupShuffleXor(keys[0], 1u);
    let tmp_1 = subgroupShuffleXor(values[0], 1u);
    let tmp_2 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_3 = keys[0] < tmp_0 || (keys[0] == tmp_0 && values[0] < tmp_1);
    if tmp_2 == tmp_3 { keys[0] = tmp_0; values[0] = tmp_1; }
    }
    // exch_intxn(tmask:3,swbit:1,wpt:1)
    {
    let tmp_4 = subgroupShuffleXor(keys[0], 3u);
    let tmp_5 = subgroupShuffleXor(values[0], 3u);
    let tmp_6 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_7 = keys[0] < tmp_4 || (keys[0] == tmp_4 && values[0] < tmp_5);
    if tmp_6 == tmp_7 { keys[0] = tmp_4; values[0] = tmp_5; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:1) 
    {
    let tmp_8 = subgroupShuffleXor(keys[0], 1u);
    let tmp_9 = subgroupShuffleXor(values[0], 1u);
    let tmp_10 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_11 = keys[0] < tmp_8 || (keys[0] == tmp_8 && values[0] < tmp_9);
    if tmp_10 == tmp_11 { keys[0] = tmp_8; values[0] = tmp_9; }
    }

    // block store
    for (var r = 0u; r < WPT; r = r + 1u) {
        let pos = local_tid * WPT + r;
        if is_active && pos < seg_size {
            global_keys[seg_start + pos] = keys[r];
            global_value_indices[seg_start + pos] = values[r];
        }
    }
}
