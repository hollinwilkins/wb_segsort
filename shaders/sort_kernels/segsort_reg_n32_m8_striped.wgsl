
enable subgroups;

override WG: u32 = 8u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 32u;
const M: u32 = 8u;
const WPT: u32 = 4u;

@compute @workgroup_size(WG, 1, 1)
fn segsort_reg_n32_m8_striped(
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

    var keys: array<u32, 4>;
    var values: array<u32, 4>;

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

    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_0 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_0;let tmp_1 = values[0]; values[0] = values[1]; values[1] = tmp_1; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_2 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_2;let tmp_3 = values[2]; values[2] = values[3]; values[3] = tmp_3; }
    }
    // exch_local(3,4) 
    // cmp_swap(0,3)
    if keys[0] > keys[3] || (keys[0] == keys[3] && values[0] > values[3]) {
    // swap(0,3) 
    { let tmp_4 = keys[0]; keys[0] = keys[3]; keys[3] = tmp_4;let tmp_5 = values[0]; values[0] = values[3]; values[3] = tmp_5; }
    }
    // cmp_swap(1,2)
    if keys[1] > keys[2] || (keys[1] == keys[2] && values[1] > values[2]) {
    // swap(1,2) 
    { let tmp_6 = keys[1]; keys[1] = keys[2]; keys[2] = tmp_6;let tmp_7 = values[1]; values[1] = values[2]; values[2] = tmp_7; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_8 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_8;let tmp_9 = values[0]; values[0] = values[1]; values[1] = tmp_9; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_10 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_10;let tmp_11 = values[2]; values[2] = values[3]; values[3] = tmp_11; }
    }
    // exch_intxn(tmask:1,swbit:0,wpt:4)
    {
    let tmp_12 = subgroupShuffleXor(keys[3], 1u);
    let tmp_13 = subgroupShuffleXor(values[3], 1u);
    let tmp_14 = subgroupShuffleXor(keys[2], 1u);
    let tmp_15 = subgroupShuffleXor(values[2], 1u);
    let tmp_16 = subgroupShuffleXor(keys[1], 1u);
    let tmp_17 = subgroupShuffleXor(values[1], 1u);
    let tmp_18 = subgroupShuffleXor(keys[0], 1u);
    let tmp_19 = subgroupShuffleXor(values[0], 1u);
    let tmp_20 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_21 = keys[0] < tmp_12 || (keys[0] == tmp_12 && values[0] < tmp_13);
    if tmp_20 == tmp_21 { keys[0] = tmp_12; values[0] = tmp_13; }
    let tmp_22 = keys[1] < tmp_14 || (keys[1] == tmp_14 && values[1] < tmp_15);
    if tmp_20 == tmp_22 { keys[1] = tmp_14; values[1] = tmp_15; }
    let tmp_23 = keys[2] < tmp_16 || (keys[2] == tmp_16 && values[2] < tmp_17);
    if tmp_20 == tmp_23 { keys[2] = tmp_16; values[2] = tmp_17; }
    let tmp_24 = keys[3] < tmp_18 || (keys[3] == tmp_18 && values[3] < tmp_19);
    if tmp_20 == tmp_24 { keys[3] = tmp_18; values[3] = tmp_19; }
    }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_25 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_25;let tmp_26 = values[0]; values[0] = values[2]; values[2] = tmp_26; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_27 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_27;let tmp_28 = values[1]; values[1] = values[3]; values[3] = tmp_28; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_29 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_29;let tmp_30 = values[0]; values[0] = values[1]; values[1] = tmp_30; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_31 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_31;let tmp_32 = values[2]; values[2] = values[3]; values[3] = tmp_32; }
    }
    // exch_intxn(tmask:3,swbit:1,wpt:4)
    {
    let tmp_33 = subgroupShuffleXor(keys[3], 3u);
    let tmp_34 = subgroupShuffleXor(values[3], 3u);
    let tmp_35 = subgroupShuffleXor(keys[2], 3u);
    let tmp_36 = subgroupShuffleXor(values[2], 3u);
    let tmp_37 = subgroupShuffleXor(keys[1], 3u);
    let tmp_38 = subgroupShuffleXor(values[1], 3u);
    let tmp_39 = subgroupShuffleXor(keys[0], 3u);
    let tmp_40 = subgroupShuffleXor(values[0], 3u);
    let tmp_41 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_42 = keys[0] < tmp_33 || (keys[0] == tmp_33 && values[0] < tmp_34);
    if tmp_41 == tmp_42 { keys[0] = tmp_33; values[0] = tmp_34; }
    let tmp_43 = keys[1] < tmp_35 || (keys[1] == tmp_35 && values[1] < tmp_36);
    if tmp_41 == tmp_43 { keys[1] = tmp_35; values[1] = tmp_36; }
    let tmp_44 = keys[2] < tmp_37 || (keys[2] == tmp_37 && values[2] < tmp_38);
    if tmp_41 == tmp_44 { keys[2] = tmp_37; values[2] = tmp_38; }
    let tmp_45 = keys[3] < tmp_39 || (keys[3] == tmp_39 && values[3] < tmp_40);
    if tmp_41 == tmp_45 { keys[3] = tmp_39; values[3] = tmp_40; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:4) 
    {
    let tmp_46 = subgroupShuffleXor(keys[0], 1u);
    let tmp_47 = subgroupShuffleXor(values[0], 1u);
    let tmp_48 = subgroupShuffleXor(keys[1], 1u);
    let tmp_49 = subgroupShuffleXor(values[1], 1u);
    let tmp_50 = subgroupShuffleXor(keys[2], 1u);
    let tmp_51 = subgroupShuffleXor(values[2], 1u);
    let tmp_52 = subgroupShuffleXor(keys[3], 1u);
    let tmp_53 = subgroupShuffleXor(values[3], 1u);
    let tmp_54 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_55 = keys[0] < tmp_46 || (keys[0] == tmp_46 && values[0] < tmp_47);
    if tmp_54 == tmp_55 { keys[0] = tmp_46; values[0] = tmp_47; }
    let tmp_56 = keys[1] < tmp_48 || (keys[1] == tmp_48 && values[1] < tmp_49);
    if tmp_54 == tmp_56 { keys[1] = tmp_48; values[1] = tmp_49; }
    let tmp_57 = keys[2] < tmp_50 || (keys[2] == tmp_50 && values[2] < tmp_51);
    if tmp_54 == tmp_57 { keys[2] = tmp_50; values[2] = tmp_51; }
    let tmp_58 = keys[3] < tmp_52 || (keys[3] == tmp_52 && values[3] < tmp_53);
    if tmp_54 == tmp_58 { keys[3] = tmp_52; values[3] = tmp_53; }
    }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_59 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_59;let tmp_60 = values[0]; values[0] = values[2]; values[2] = tmp_60; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_61 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_61;let tmp_62 = values[1]; values[1] = values[3]; values[3] = tmp_62; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_63 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_63;let tmp_64 = values[0]; values[0] = values[1]; values[1] = tmp_64; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_65 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_65;let tmp_66 = values[2]; values[2] = values[3]; values[3] = tmp_66; }
    }
    // exch_intxn(tmask:7,swbit:2,wpt:4)
    {
    let tmp_67 = subgroupShuffleXor(keys[3], 7u);
    let tmp_68 = subgroupShuffleXor(values[3], 7u);
    let tmp_69 = subgroupShuffleXor(keys[2], 7u);
    let tmp_70 = subgroupShuffleXor(values[2], 7u);
    let tmp_71 = subgroupShuffleXor(keys[1], 7u);
    let tmp_72 = subgroupShuffleXor(values[1], 7u);
    let tmp_73 = subgroupShuffleXor(keys[0], 7u);
    let tmp_74 = subgroupShuffleXor(values[0], 7u);
    let tmp_75 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_76 = keys[0] < tmp_67 || (keys[0] == tmp_67 && values[0] < tmp_68);
    if tmp_75 == tmp_76 { keys[0] = tmp_67; values[0] = tmp_68; }
    let tmp_77 = keys[1] < tmp_69 || (keys[1] == tmp_69 && values[1] < tmp_70);
    if tmp_75 == tmp_77 { keys[1] = tmp_69; values[1] = tmp_70; }
    let tmp_78 = keys[2] < tmp_71 || (keys[2] == tmp_71 && values[2] < tmp_72);
    if tmp_75 == tmp_78 { keys[2] = tmp_71; values[2] = tmp_72; }
    let tmp_79 = keys[3] < tmp_73 || (keys[3] == tmp_73 && values[3] < tmp_74);
    if tmp_75 == tmp_79 { keys[3] = tmp_73; values[3] = tmp_74; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:4) 
    {
    let tmp_80 = subgroupShuffleXor(keys[0], 2u);
    let tmp_81 = subgroupShuffleXor(values[0], 2u);
    let tmp_82 = subgroupShuffleXor(keys[1], 2u);
    let tmp_83 = subgroupShuffleXor(values[1], 2u);
    let tmp_84 = subgroupShuffleXor(keys[2], 2u);
    let tmp_85 = subgroupShuffleXor(values[2], 2u);
    let tmp_86 = subgroupShuffleXor(keys[3], 2u);
    let tmp_87 = subgroupShuffleXor(values[3], 2u);
    let tmp_88 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_89 = keys[0] < tmp_80 || (keys[0] == tmp_80 && values[0] < tmp_81);
    if tmp_88 == tmp_89 { keys[0] = tmp_80; values[0] = tmp_81; }
    let tmp_90 = keys[1] < tmp_82 || (keys[1] == tmp_82 && values[1] < tmp_83);
    if tmp_88 == tmp_90 { keys[1] = tmp_82; values[1] = tmp_83; }
    let tmp_91 = keys[2] < tmp_84 || (keys[2] == tmp_84 && values[2] < tmp_85);
    if tmp_88 == tmp_91 { keys[2] = tmp_84; values[2] = tmp_85; }
    let tmp_92 = keys[3] < tmp_86 || (keys[3] == tmp_86 && values[3] < tmp_87);
    if tmp_88 == tmp_92 { keys[3] = tmp_86; values[3] = tmp_87; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:4) 
    {
    let tmp_93 = subgroupShuffleXor(keys[0], 1u);
    let tmp_94 = subgroupShuffleXor(values[0], 1u);
    let tmp_95 = subgroupShuffleXor(keys[1], 1u);
    let tmp_96 = subgroupShuffleXor(values[1], 1u);
    let tmp_97 = subgroupShuffleXor(keys[2], 1u);
    let tmp_98 = subgroupShuffleXor(values[2], 1u);
    let tmp_99 = subgroupShuffleXor(keys[3], 1u);
    let tmp_100 = subgroupShuffleXor(values[3], 1u);
    let tmp_101 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_102 = keys[0] < tmp_93 || (keys[0] == tmp_93 && values[0] < tmp_94);
    if tmp_101 == tmp_102 { keys[0] = tmp_93; values[0] = tmp_94; }
    let tmp_103 = keys[1] < tmp_95 || (keys[1] == tmp_95 && values[1] < tmp_96);
    if tmp_101 == tmp_103 { keys[1] = tmp_95; values[1] = tmp_96; }
    let tmp_104 = keys[2] < tmp_97 || (keys[2] == tmp_97 && values[2] < tmp_98);
    if tmp_101 == tmp_104 { keys[2] = tmp_97; values[2] = tmp_98; }
    let tmp_105 = keys[3] < tmp_99 || (keys[3] == tmp_99 && values[3] < tmp_100);
    if tmp_101 == tmp_105 { keys[3] = tmp_99; values[3] = tmp_100; }
    }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_106 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_106;let tmp_107 = values[0]; values[0] = values[2]; values[2] = tmp_107; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_108 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_108;let tmp_109 = values[1]; values[1] = values[3]; values[3] = tmp_109; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_110 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_110;let tmp_111 = values[0]; values[0] = values[1]; values[1] = tmp_111; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_112 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_112;let tmp_113 = values[2]; values[2] = values[3]; values[3] = tmp_113; }
    }

    // striped (coalesced) store via subgroup shuffle transpose
    {
        let grp_base = sid - local_tid;   // first lane of this segment
        let want = local_tid & (WPT - 1u);
        var out_keys: array<u32, WPT>;
        var out_values: array<u32, WPT>;
        { let src = grp_base + ((0u * M + local_tid) >> 2u);
          { let k = subgroupShuffle(keys[0], src); let v = subgroupShuffle(values[0], src); if want == 0u { out_keys[0] = k; out_values[0] = v; } }
          { let k = subgroupShuffle(keys[1], src); let v = subgroupShuffle(values[1], src); if want == 1u { out_keys[0] = k; out_values[0] = v; } }
          { let k = subgroupShuffle(keys[2], src); let v = subgroupShuffle(values[2], src); if want == 2u { out_keys[0] = k; out_values[0] = v; } }
          { let k = subgroupShuffle(keys[3], src); let v = subgroupShuffle(values[3], src); if want == 3u { out_keys[0] = k; out_values[0] = v; } }
        }
        { let src = grp_base + ((1u * M + local_tid) >> 2u);
          { let k = subgroupShuffle(keys[0], src); let v = subgroupShuffle(values[0], src); if want == 0u { out_keys[1] = k; out_values[1] = v; } }
          { let k = subgroupShuffle(keys[1], src); let v = subgroupShuffle(values[1], src); if want == 1u { out_keys[1] = k; out_values[1] = v; } }
          { let k = subgroupShuffle(keys[2], src); let v = subgroupShuffle(values[2], src); if want == 2u { out_keys[1] = k; out_values[1] = v; } }
          { let k = subgroupShuffle(keys[3], src); let v = subgroupShuffle(values[3], src); if want == 3u { out_keys[1] = k; out_values[1] = v; } }
        }
        { let src = grp_base + ((2u * M + local_tid) >> 2u);
          { let k = subgroupShuffle(keys[0], src); let v = subgroupShuffle(values[0], src); if want == 0u { out_keys[2] = k; out_values[2] = v; } }
          { let k = subgroupShuffle(keys[1], src); let v = subgroupShuffle(values[1], src); if want == 1u { out_keys[2] = k; out_values[2] = v; } }
          { let k = subgroupShuffle(keys[2], src); let v = subgroupShuffle(values[2], src); if want == 2u { out_keys[2] = k; out_values[2] = v; } }
          { let k = subgroupShuffle(keys[3], src); let v = subgroupShuffle(values[3], src); if want == 3u { out_keys[2] = k; out_values[2] = v; } }
        }
        { let src = grp_base + ((3u * M + local_tid) >> 2u);
          { let k = subgroupShuffle(keys[0], src); let v = subgroupShuffle(values[0], src); if want == 0u { out_keys[3] = k; out_values[3] = v; } }
          { let k = subgroupShuffle(keys[1], src); let v = subgroupShuffle(values[1], src); if want == 1u { out_keys[3] = k; out_values[3] = v; } }
          { let k = subgroupShuffle(keys[2], src); let v = subgroupShuffle(values[2], src); if want == 2u { out_keys[3] = k; out_values[3] = v; } }
          { let k = subgroupShuffle(keys[3], src); let v = subgroupShuffle(values[3], src); if want == 3u { out_keys[3] = k; out_values[3] = v; } }
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
