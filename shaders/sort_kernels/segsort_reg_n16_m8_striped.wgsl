
enable subgroups;

override WG: u32 = 8u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 16u;
const M: u32 = 8u;
const WPT: u32 = 2u;

@compute @workgroup_size(WG, 1, 1)
fn segsort_reg_n16_m8_striped(
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 5u;

    let bin_base = select(bin_offsets[BIN - 1u], 0u, BIN == 0u);
    let bin_count = bin_offsets[BIN] - bin_base;

    let local_tid = sid & (M - 1u);
    let wg_index = wg_id.x + wg_id.y * wg_dim.x;
    let global_seg = (wg_index * WG + lid) / M;

    let is_active = global_seg < bin_count;
    let slot = bin_base + select(0u, global_seg, is_active);   // clamp so the read is in-range
    let seg_id = bin_indices[slot];
    let seg_start = select(segments[seg_id - 1u], 0u, seg_id == 0u);
    let seg_end = segments[seg_id];
    let seg_size = select(0u, seg_end - seg_start, is_active);

    var keys: array<u32, 2>;
    var values: array<u32, 2>;

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

    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_0 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_0;let tmp_1 = values[0]; values[0] = values[1]; values[1] = tmp_1; }
    }
    // exch_intxn(tmask:1,swbit:0,wpt:2)
    {
    let tmp_2 = subgroupShuffleXor(keys[1], 1u);
    let tmp_3 = subgroupShuffleXor(values[1], 1u);
    let tmp_4 = subgroupShuffleXor(keys[0], 1u);
    let tmp_5 = subgroupShuffleXor(values[0], 1u);
    let tmp_6 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_7 = keys[0] < tmp_2 || (keys[0] == tmp_2 && values[0] < tmp_3);
    if tmp_6 == tmp_7 { keys[0] = tmp_2; values[0] = tmp_3; }
    let tmp_8 = keys[1] < tmp_4 || (keys[1] == tmp_4 && values[1] < tmp_5);
    if tmp_6 == tmp_8 { keys[1] = tmp_4; values[1] = tmp_5; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_9 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_9;let tmp_10 = values[0]; values[0] = values[1]; values[1] = tmp_10; }
    }
    // exch_intxn(tmask:3,swbit:1,wpt:2)
    {
    let tmp_11 = subgroupShuffleXor(keys[1], 3u);
    let tmp_12 = subgroupShuffleXor(values[1], 3u);
    let tmp_13 = subgroupShuffleXor(keys[0], 3u);
    let tmp_14 = subgroupShuffleXor(values[0], 3u);
    let tmp_15 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_16 = keys[0] < tmp_11 || (keys[0] == tmp_11 && values[0] < tmp_12);
    if tmp_15 == tmp_16 { keys[0] = tmp_11; values[0] = tmp_12; }
    let tmp_17 = keys[1] < tmp_13 || (keys[1] == tmp_13 && values[1] < tmp_14);
    if tmp_15 == tmp_17 { keys[1] = tmp_13; values[1] = tmp_14; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_18 = subgroupShuffleXor(keys[0], 1u);
    let tmp_19 = subgroupShuffleXor(values[0], 1u);
    let tmp_20 = subgroupShuffleXor(keys[1], 1u);
    let tmp_21 = subgroupShuffleXor(values[1], 1u);
    let tmp_22 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_23 = keys[0] < tmp_18 || (keys[0] == tmp_18 && values[0] < tmp_19);
    if tmp_22 == tmp_23 { keys[0] = tmp_18; values[0] = tmp_19; }
    let tmp_24 = keys[1] < tmp_20 || (keys[1] == tmp_20 && values[1] < tmp_21);
    if tmp_22 == tmp_24 { keys[1] = tmp_20; values[1] = tmp_21; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_25 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_25;let tmp_26 = values[0]; values[0] = values[1]; values[1] = tmp_26; }
    }
    // exch_intxn(tmask:7,swbit:2,wpt:2)
    {
    let tmp_27 = subgroupShuffleXor(keys[1], 7u);
    let tmp_28 = subgroupShuffleXor(values[1], 7u);
    let tmp_29 = subgroupShuffleXor(keys[0], 7u);
    let tmp_30 = subgroupShuffleXor(values[0], 7u);
    let tmp_31 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_32 = keys[0] < tmp_27 || (keys[0] == tmp_27 && values[0] < tmp_28);
    if tmp_31 == tmp_32 { keys[0] = tmp_27; values[0] = tmp_28; }
    let tmp_33 = keys[1] < tmp_29 || (keys[1] == tmp_29 && values[1] < tmp_30);
    if tmp_31 == tmp_33 { keys[1] = tmp_29; values[1] = tmp_30; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_34 = subgroupShuffleXor(keys[0], 2u);
    let tmp_35 = subgroupShuffleXor(values[0], 2u);
    let tmp_36 = subgroupShuffleXor(keys[1], 2u);
    let tmp_37 = subgroupShuffleXor(values[1], 2u);
    let tmp_38 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_39 = keys[0] < tmp_34 || (keys[0] == tmp_34 && values[0] < tmp_35);
    if tmp_38 == tmp_39 { keys[0] = tmp_34; values[0] = tmp_35; }
    let tmp_40 = keys[1] < tmp_36 || (keys[1] == tmp_36 && values[1] < tmp_37);
    if tmp_38 == tmp_40 { keys[1] = tmp_36; values[1] = tmp_37; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_41 = subgroupShuffleXor(keys[0], 1u);
    let tmp_42 = subgroupShuffleXor(values[0], 1u);
    let tmp_43 = subgroupShuffleXor(keys[1], 1u);
    let tmp_44 = subgroupShuffleXor(values[1], 1u);
    let tmp_45 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_46 = keys[0] < tmp_41 || (keys[0] == tmp_41 && values[0] < tmp_42);
    if tmp_45 == tmp_46 { keys[0] = tmp_41; values[0] = tmp_42; }
    let tmp_47 = keys[1] < tmp_43 || (keys[1] == tmp_43 && values[1] < tmp_44);
    if tmp_45 == tmp_47 { keys[1] = tmp_43; values[1] = tmp_44; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_48 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_48;let tmp_49 = values[0]; values[0] = values[1]; values[1] = tmp_49; }
    }

    // striped (coalesced) store via subgroup shuffle transpose
    {
        let grp_base = sid - local_tid;   // first lane of this segment
        let want = local_tid & (WPT - 1u);
        var out_keys: array<u32, WPT>;
        var out_values: array<u32, WPT>;
        { let src = grp_base + ((0u * M + local_tid) >> 1u);
          { let k = subgroupShuffle(keys[0], src); let v = subgroupShuffle(values[0], src); if want == 0u { out_keys[0] = k; out_values[0] = v; } }
          { let k = subgroupShuffle(keys[1], src); let v = subgroupShuffle(values[1], src); if want == 1u { out_keys[0] = k; out_values[0] = v; } }
        }
        { let src = grp_base + ((1u * M + local_tid) >> 1u);
          { let k = subgroupShuffle(keys[0], src); let v = subgroupShuffle(values[0], src); if want == 0u { out_keys[1] = k; out_values[1] = v; } }
          { let k = subgroupShuffle(keys[1], src); let v = subgroupShuffle(values[1], src); if want == 1u { out_keys[1] = k; out_values[1] = v; } }
        }
        for (var r = 0u; r < WPT; r = r + 1u) {
            let j = r * M + local_tid;
            if is_active && j < seg_size {
                global_keys[seg_start + j] = out_keys[r];
                global_value_indices[seg_start + j] = out_values[r];
            }
        }
    }
}
