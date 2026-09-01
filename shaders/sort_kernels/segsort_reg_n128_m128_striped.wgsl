
enable subgroups;

override WG: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 128u;
const M: u32 = 128u;
const WPT: u32 = 1u;

@compute @workgroup_size(WG, 1, 1)
fn segsort_reg_n128_m128_striped(
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    const BIN: u32 = 7u;

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
    // exch_intxn(tmask:7,swbit:2,wpt:1)
    {
    let tmp_12 = subgroupShuffleXor(keys[0], 7u);
    let tmp_13 = subgroupShuffleXor(values[0], 7u);
    let tmp_14 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_15 = keys[0] < tmp_12 || (keys[0] == tmp_12 && values[0] < tmp_13);
    if tmp_14 == tmp_15 { keys[0] = tmp_12; values[0] = tmp_13; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:1) 
    {
    let tmp_16 = subgroupShuffleXor(keys[0], 2u);
    let tmp_17 = subgroupShuffleXor(values[0], 2u);
    let tmp_18 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_19 = keys[0] < tmp_16 || (keys[0] == tmp_16 && values[0] < tmp_17);
    if tmp_18 == tmp_19 { keys[0] = tmp_16; values[0] = tmp_17; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:1) 
    {
    let tmp_20 = subgroupShuffleXor(keys[0], 1u);
    let tmp_21 = subgroupShuffleXor(values[0], 1u);
    let tmp_22 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_23 = keys[0] < tmp_20 || (keys[0] == tmp_20 && values[0] < tmp_21);
    if tmp_22 == tmp_23 { keys[0] = tmp_20; values[0] = tmp_21; }
    }
    // exch_intxn(tmask:15,swbit:3,wpt:1)
    {
    let tmp_24 = subgroupShuffleXor(keys[0], 15u);
    let tmp_25 = subgroupShuffleXor(values[0], 15u);
    let tmp_26 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_27 = keys[0] < tmp_24 || (keys[0] == tmp_24 && values[0] < tmp_25);
    if tmp_26 == tmp_27 { keys[0] = tmp_24; values[0] = tmp_25; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:1) 
    {
    let tmp_28 = subgroupShuffleXor(keys[0], 4u);
    let tmp_29 = subgroupShuffleXor(values[0], 4u);
    let tmp_30 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_31 = keys[0] < tmp_28 || (keys[0] == tmp_28 && values[0] < tmp_29);
    if tmp_30 == tmp_31 { keys[0] = tmp_28; values[0] = tmp_29; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:1) 
    {
    let tmp_32 = subgroupShuffleXor(keys[0], 2u);
    let tmp_33 = subgroupShuffleXor(values[0], 2u);
    let tmp_34 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_35 = keys[0] < tmp_32 || (keys[0] == tmp_32 && values[0] < tmp_33);
    if tmp_34 == tmp_35 { keys[0] = tmp_32; values[0] = tmp_33; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:1) 
    {
    let tmp_36 = subgroupShuffleXor(keys[0], 1u);
    let tmp_37 = subgroupShuffleXor(values[0], 1u);
    let tmp_38 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_39 = keys[0] < tmp_36 || (keys[0] == tmp_36 && values[0] < tmp_37);
    if tmp_38 == tmp_39 { keys[0] = tmp_36; values[0] = tmp_37; }
    }
    // exch_intxn(tmask:31,swbit:4,wpt:1)
    {
    let tmp_40 = subgroupShuffleXor(keys[0], 31u);
    let tmp_41 = subgroupShuffleXor(values[0], 31u);
    let tmp_42 = extractBits(local_tid, 4u, 1u) != 0u;
    let tmp_43 = keys[0] < tmp_40 || (keys[0] == tmp_40 && values[0] < tmp_41);
    if tmp_42 == tmp_43 { keys[0] = tmp_40; values[0] = tmp_41; }
    }
    // exch_paral(tmask:8,swbit:3,wpt:1) 
    {
    let tmp_44 = subgroupShuffleXor(keys[0], 8u);
    let tmp_45 = subgroupShuffleXor(values[0], 8u);
    let tmp_46 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_47 = keys[0] < tmp_44 || (keys[0] == tmp_44 && values[0] < tmp_45);
    if tmp_46 == tmp_47 { keys[0] = tmp_44; values[0] = tmp_45; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:1) 
    {
    let tmp_48 = subgroupShuffleXor(keys[0], 4u);
    let tmp_49 = subgroupShuffleXor(values[0], 4u);
    let tmp_50 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_51 = keys[0] < tmp_48 || (keys[0] == tmp_48 && values[0] < tmp_49);
    if tmp_50 == tmp_51 { keys[0] = tmp_48; values[0] = tmp_49; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:1) 
    {
    let tmp_52 = subgroupShuffleXor(keys[0], 2u);
    let tmp_53 = subgroupShuffleXor(values[0], 2u);
    let tmp_54 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_55 = keys[0] < tmp_52 || (keys[0] == tmp_52 && values[0] < tmp_53);
    if tmp_54 == tmp_55 { keys[0] = tmp_52; values[0] = tmp_53; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:1) 
    {
    let tmp_56 = subgroupShuffleXor(keys[0], 1u);
    let tmp_57 = subgroupShuffleXor(values[0], 1u);
    let tmp_58 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_59 = keys[0] < tmp_56 || (keys[0] == tmp_56 && values[0] < tmp_57);
    if tmp_58 == tmp_59 { keys[0] = tmp_56; values[0] = tmp_57; }
    }
    // exch_intxn(tmask:63,swbit:5,wpt:1)
    {
    let tmp_60 = subgroupShuffleXor(keys[0], 63u);
    let tmp_61 = subgroupShuffleXor(values[0], 63u);
    let tmp_62 = extractBits(local_tid, 5u, 1u) != 0u;
    let tmp_63 = keys[0] < tmp_60 || (keys[0] == tmp_60 && values[0] < tmp_61);
    if tmp_62 == tmp_63 { keys[0] = tmp_60; values[0] = tmp_61; }
    }
    // exch_paral(tmask:16,swbit:4,wpt:1) 
    {
    let tmp_64 = subgroupShuffleXor(keys[0], 16u);
    let tmp_65 = subgroupShuffleXor(values[0], 16u);
    let tmp_66 = extractBits(local_tid, 4u, 1u) != 0u;
    let tmp_67 = keys[0] < tmp_64 || (keys[0] == tmp_64 && values[0] < tmp_65);
    if tmp_66 == tmp_67 { keys[0] = tmp_64; values[0] = tmp_65; }
    }
    // exch_paral(tmask:8,swbit:3,wpt:1) 
    {
    let tmp_68 = subgroupShuffleXor(keys[0], 8u);
    let tmp_69 = subgroupShuffleXor(values[0], 8u);
    let tmp_70 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_71 = keys[0] < tmp_68 || (keys[0] == tmp_68 && values[0] < tmp_69);
    if tmp_70 == tmp_71 { keys[0] = tmp_68; values[0] = tmp_69; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:1) 
    {
    let tmp_72 = subgroupShuffleXor(keys[0], 4u);
    let tmp_73 = subgroupShuffleXor(values[0], 4u);
    let tmp_74 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_75 = keys[0] < tmp_72 || (keys[0] == tmp_72 && values[0] < tmp_73);
    if tmp_74 == tmp_75 { keys[0] = tmp_72; values[0] = tmp_73; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:1) 
    {
    let tmp_76 = subgroupShuffleXor(keys[0], 2u);
    let tmp_77 = subgroupShuffleXor(values[0], 2u);
    let tmp_78 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_79 = keys[0] < tmp_76 || (keys[0] == tmp_76 && values[0] < tmp_77);
    if tmp_78 == tmp_79 { keys[0] = tmp_76; values[0] = tmp_77; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:1) 
    {
    let tmp_80 = subgroupShuffleXor(keys[0], 1u);
    let tmp_81 = subgroupShuffleXor(values[0], 1u);
    let tmp_82 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_83 = keys[0] < tmp_80 || (keys[0] == tmp_80 && values[0] < tmp_81);
    if tmp_82 == tmp_83 { keys[0] = tmp_80; values[0] = tmp_81; }
    }
    // exch_intxn(tmask:127,swbit:6,wpt:1)
    {
    let tmp_84 = subgroupShuffleXor(keys[0], 127u);
    let tmp_85 = subgroupShuffleXor(values[0], 127u);
    let tmp_86 = extractBits(local_tid, 6u, 1u) != 0u;
    let tmp_87 = keys[0] < tmp_84 || (keys[0] == tmp_84 && values[0] < tmp_85);
    if tmp_86 == tmp_87 { keys[0] = tmp_84; values[0] = tmp_85; }
    }
    // exch_paral(tmask:32,swbit:5,wpt:1) 
    {
    let tmp_88 = subgroupShuffleXor(keys[0], 32u);
    let tmp_89 = subgroupShuffleXor(values[0], 32u);
    let tmp_90 = extractBits(local_tid, 5u, 1u) != 0u;
    let tmp_91 = keys[0] < tmp_88 || (keys[0] == tmp_88 && values[0] < tmp_89);
    if tmp_90 == tmp_91 { keys[0] = tmp_88; values[0] = tmp_89; }
    }
    // exch_paral(tmask:16,swbit:4,wpt:1) 
    {
    let tmp_92 = subgroupShuffleXor(keys[0], 16u);
    let tmp_93 = subgroupShuffleXor(values[0], 16u);
    let tmp_94 = extractBits(local_tid, 4u, 1u) != 0u;
    let tmp_95 = keys[0] < tmp_92 || (keys[0] == tmp_92 && values[0] < tmp_93);
    if tmp_94 == tmp_95 { keys[0] = tmp_92; values[0] = tmp_93; }
    }
    // exch_paral(tmask:8,swbit:3,wpt:1) 
    {
    let tmp_96 = subgroupShuffleXor(keys[0], 8u);
    let tmp_97 = subgroupShuffleXor(values[0], 8u);
    let tmp_98 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_99 = keys[0] < tmp_96 || (keys[0] == tmp_96 && values[0] < tmp_97);
    if tmp_98 == tmp_99 { keys[0] = tmp_96; values[0] = tmp_97; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:1) 
    {
    let tmp_100 = subgroupShuffleXor(keys[0], 4u);
    let tmp_101 = subgroupShuffleXor(values[0], 4u);
    let tmp_102 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_103 = keys[0] < tmp_100 || (keys[0] == tmp_100 && values[0] < tmp_101);
    if tmp_102 == tmp_103 { keys[0] = tmp_100; values[0] = tmp_101; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:1) 
    {
    let tmp_104 = subgroupShuffleXor(keys[0], 2u);
    let tmp_105 = subgroupShuffleXor(values[0], 2u);
    let tmp_106 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_107 = keys[0] < tmp_104 || (keys[0] == tmp_104 && values[0] < tmp_105);
    if tmp_106 == tmp_107 { keys[0] = tmp_104; values[0] = tmp_105; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:1) 
    {
    let tmp_108 = subgroupShuffleXor(keys[0], 1u);
    let tmp_109 = subgroupShuffleXor(values[0], 1u);
    let tmp_110 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_111 = keys[0] < tmp_108 || (keys[0] == tmp_108 && values[0] < tmp_109);
    if tmp_110 == tmp_111 { keys[0] = tmp_108; values[0] = tmp_109; }
    }

    // striped (coalesced) store via subgroup shuffle transpose
    {
        let grp_base = sid - local_tid;   // first lane of this segment
        let want = local_tid & (WPT - 1u);
        var out_keys: array<u32, WPT>;
        var out_values: array<u32, WPT>;
        { let src = grp_base + ((0u * M + local_tid) >> 0u);
          { let k = subgroupShuffle(keys[0], src); let v = subgroupShuffle(values[0], src); if want == 0u { out_keys[0] = k; out_values[0] = v; } }
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
