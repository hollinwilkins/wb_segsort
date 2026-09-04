
enable subgroups;

override WG: u32 = 32u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 256u;
const M: u32 = 32u;
const WPT: u32 = 8u;

@compute @workgroup_size(WG, 1, 1)
fn segsort_reg_n256_m32_striped(
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 8u;

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

    var keys: array<u32, 8>;
    var values: array<u32, 8>;

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

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_0 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_0;
                let tmp_1 = values[0]; values[0] = values[1]; values[1] = tmp_1;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_2 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_2;
                let tmp_3 = values[2]; values[2] = values[3]; values[3] = tmp_3;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_4 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_4;
                let tmp_5 = values[4]; values[4] = values[5]; values[5] = tmp_5;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_6 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_6;
                let tmp_7 = values[6]; values[6] = values[7]; values[7] = tmp_7;
            }
        }
    }

    // exch_local(3,8)
    {
        // cmp_swap(0,3)
        if keys[0] > keys[3] || (keys[0] == keys[3] && values[0] > values[3]) {
            // swap(0,3)
            {
                let tmp_8 = keys[0]; keys[0] = keys[3]; keys[3] = tmp_8;
                let tmp_9 = values[0]; values[0] = values[3]; values[3] = tmp_9;
            }
        }
        // cmp_swap(1,2)
        if keys[1] > keys[2] || (keys[1] == keys[2] && values[1] > values[2]) {
            // swap(1,2)
            {
                let tmp_10 = keys[1]; keys[1] = keys[2]; keys[2] = tmp_10;
                let tmp_11 = values[1]; values[1] = values[2]; values[2] = tmp_11;
            }
        }
        // cmp_swap(4,7)
        if keys[4] > keys[7] || (keys[4] == keys[7] && values[4] > values[7]) {
            // swap(4,7)
            {
                let tmp_12 = keys[4]; keys[4] = keys[7]; keys[7] = tmp_12;
                let tmp_13 = values[4]; values[4] = values[7]; values[7] = tmp_13;
            }
        }
        // cmp_swap(5,6)
        if keys[5] > keys[6] || (keys[5] == keys[6] && values[5] > values[6]) {
            // swap(5,6)
            {
                let tmp_14 = keys[5]; keys[5] = keys[6]; keys[6] = tmp_14;
                let tmp_15 = values[5]; values[5] = values[6]; values[6] = tmp_15;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_16 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_16;
                let tmp_17 = values[0]; values[0] = values[1]; values[1] = tmp_17;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_18 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_18;
                let tmp_19 = values[2]; values[2] = values[3]; values[3] = tmp_19;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_20 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_20;
                let tmp_21 = values[4]; values[4] = values[5]; values[5] = tmp_21;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_22 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_22;
                let tmp_23 = values[6]; values[6] = values[7]; values[7] = tmp_23;
            }
        }
    }

    // exch_local(7,8)
    {
        // cmp_swap(0,7)
        if keys[0] > keys[7] || (keys[0] == keys[7] && values[0] > values[7]) {
            // swap(0,7)
            {
                let tmp_24 = keys[0]; keys[0] = keys[7]; keys[7] = tmp_24;
                let tmp_25 = values[0]; values[0] = values[7]; values[7] = tmp_25;
            }
        }
        // cmp_swap(1,6)
        if keys[1] > keys[6] || (keys[1] == keys[6] && values[1] > values[6]) {
            // swap(1,6)
            {
                let tmp_26 = keys[1]; keys[1] = keys[6]; keys[6] = tmp_26;
                let tmp_27 = values[1]; values[1] = values[6]; values[6] = tmp_27;
            }
        }
        // cmp_swap(2,5)
        if keys[2] > keys[5] || (keys[2] == keys[5] && values[2] > values[5]) {
            // swap(2,5)
            {
                let tmp_28 = keys[2]; keys[2] = keys[5]; keys[5] = tmp_28;
                let tmp_29 = values[2]; values[2] = values[5]; values[5] = tmp_29;
            }
        }
        // cmp_swap(3,4)
        if keys[3] > keys[4] || (keys[3] == keys[4] && values[3] > values[4]) {
            // swap(3,4)
            {
                let tmp_30 = keys[3]; keys[3] = keys[4]; keys[4] = tmp_30;
                let tmp_31 = values[3]; values[3] = values[4]; values[4] = tmp_31;
            }
        }
    }

    // exch_local(2,8)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_32 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_32;
                let tmp_33 = values[0]; values[0] = values[2]; values[2] = tmp_33;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_34 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_34;
                let tmp_35 = values[1]; values[1] = values[3]; values[3] = tmp_35;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_36 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_36;
                let tmp_37 = values[4]; values[4] = values[6]; values[6] = tmp_37;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_38 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_38;
                let tmp_39 = values[5]; values[5] = values[7]; values[7] = tmp_39;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_40 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_40;
                let tmp_41 = values[0]; values[0] = values[1]; values[1] = tmp_41;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_42 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_42;
                let tmp_43 = values[2]; values[2] = values[3]; values[3] = tmp_43;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_44 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_44;
                let tmp_45 = values[4]; values[4] = values[5]; values[5] = tmp_45;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_46 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_46;
                let tmp_47 = values[6]; values[6] = values[7]; values[7] = tmp_47;
            }
        }
    }

    // exch_intxn(tmask:1,swbit:0,wpt:8)
    {
        // _exch_subgroup([(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)],1,0)
        {
            let tmp_48 = subgroupShuffleXor(keys[7], 1u);
            let tmp_49 = subgroupShuffleXor(values[7], 1u);
            let tmp_50 = subgroupShuffleXor(keys[6], 1u);
            let tmp_51 = subgroupShuffleXor(values[6], 1u);
            let tmp_52 = subgroupShuffleXor(keys[5], 1u);
            let tmp_53 = subgroupShuffleXor(values[5], 1u);
            let tmp_54 = subgroupShuffleXor(keys[4], 1u);
            let tmp_55 = subgroupShuffleXor(values[4], 1u);
            let tmp_56 = subgroupShuffleXor(keys[3], 1u);
            let tmp_57 = subgroupShuffleXor(values[3], 1u);
            let tmp_58 = subgroupShuffleXor(keys[2], 1u);
            let tmp_59 = subgroupShuffleXor(values[2], 1u);
            let tmp_60 = subgroupShuffleXor(keys[1], 1u);
            let tmp_61 = subgroupShuffleXor(values[1], 1u);
            let tmp_62 = subgroupShuffleXor(keys[0], 1u);
            let tmp_63 = subgroupShuffleXor(values[0], 1u);
            let tmp_64 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_65 = keys[0] < tmp_48 || (keys[0] == tmp_48 && values[0] < tmp_49);
            if tmp_64 == tmp_65 { keys[0] = tmp_48; values[0] = tmp_49; }
            let tmp_66 = keys[1] < tmp_50 || (keys[1] == tmp_50 && values[1] < tmp_51);
            if tmp_64 == tmp_66 { keys[1] = tmp_50; values[1] = tmp_51; }
            let tmp_67 = keys[2] < tmp_52 || (keys[2] == tmp_52 && values[2] < tmp_53);
            if tmp_64 == tmp_67 { keys[2] = tmp_52; values[2] = tmp_53; }
            let tmp_68 = keys[3] < tmp_54 || (keys[3] == tmp_54 && values[3] < tmp_55);
            if tmp_64 == tmp_68 { keys[3] = tmp_54; values[3] = tmp_55; }
            let tmp_69 = keys[4] < tmp_56 || (keys[4] == tmp_56 && values[4] < tmp_57);
            if tmp_64 == tmp_69 { keys[4] = tmp_56; values[4] = tmp_57; }
            let tmp_70 = keys[5] < tmp_58 || (keys[5] == tmp_58 && values[5] < tmp_59);
            if tmp_64 == tmp_70 { keys[5] = tmp_58; values[5] = tmp_59; }
            let tmp_71 = keys[6] < tmp_60 || (keys[6] == tmp_60 && values[6] < tmp_61);
            if tmp_64 == tmp_71 { keys[6] = tmp_60; values[6] = tmp_61; }
            let tmp_72 = keys[7] < tmp_62 || (keys[7] == tmp_62 && values[7] < tmp_63);
            if tmp_64 == tmp_72 { keys[7] = tmp_62; values[7] = tmp_63; }
        }
    }

    // exch_local(4,8)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_73 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_73;
                let tmp_74 = values[0]; values[0] = values[4]; values[4] = tmp_74;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_75 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_75;
                let tmp_76 = values[1]; values[1] = values[5]; values[5] = tmp_76;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_77 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_77;
                let tmp_78 = values[2]; values[2] = values[6]; values[6] = tmp_78;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_79 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_79;
                let tmp_80 = values[3]; values[3] = values[7]; values[7] = tmp_80;
            }
        }
    }

    // exch_local(2,8)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_81 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_81;
                let tmp_82 = values[0]; values[0] = values[2]; values[2] = tmp_82;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_83 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_83;
                let tmp_84 = values[1]; values[1] = values[3]; values[3] = tmp_84;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_85 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_85;
                let tmp_86 = values[4]; values[4] = values[6]; values[6] = tmp_86;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_87 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_87;
                let tmp_88 = values[5]; values[5] = values[7]; values[7] = tmp_88;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_89 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_89;
                let tmp_90 = values[0]; values[0] = values[1]; values[1] = tmp_90;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_91 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_91;
                let tmp_92 = values[2]; values[2] = values[3]; values[3] = tmp_92;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_93 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_93;
                let tmp_94 = values[4]; values[4] = values[5]; values[5] = tmp_94;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_95 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_95;
                let tmp_96 = values[6]; values[6] = values[7]; values[7] = tmp_96;
            }
        }
    }

    // exch_intxn(tmask:3,swbit:1,wpt:8)
    {
        // _exch_subgroup([(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)],3,1)
        {
            let tmp_97 = subgroupShuffleXor(keys[7], 3u);
            let tmp_98 = subgroupShuffleXor(values[7], 3u);
            let tmp_99 = subgroupShuffleXor(keys[6], 3u);
            let tmp_100 = subgroupShuffleXor(values[6], 3u);
            let tmp_101 = subgroupShuffleXor(keys[5], 3u);
            let tmp_102 = subgroupShuffleXor(values[5], 3u);
            let tmp_103 = subgroupShuffleXor(keys[4], 3u);
            let tmp_104 = subgroupShuffleXor(values[4], 3u);
            let tmp_105 = subgroupShuffleXor(keys[3], 3u);
            let tmp_106 = subgroupShuffleXor(values[3], 3u);
            let tmp_107 = subgroupShuffleXor(keys[2], 3u);
            let tmp_108 = subgroupShuffleXor(values[2], 3u);
            let tmp_109 = subgroupShuffleXor(keys[1], 3u);
            let tmp_110 = subgroupShuffleXor(values[1], 3u);
            let tmp_111 = subgroupShuffleXor(keys[0], 3u);
            let tmp_112 = subgroupShuffleXor(values[0], 3u);
            let tmp_113 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_114 = keys[0] < tmp_97 || (keys[0] == tmp_97 && values[0] < tmp_98);
            if tmp_113 == tmp_114 { keys[0] = tmp_97; values[0] = tmp_98; }
            let tmp_115 = keys[1] < tmp_99 || (keys[1] == tmp_99 && values[1] < tmp_100);
            if tmp_113 == tmp_115 { keys[1] = tmp_99; values[1] = tmp_100; }
            let tmp_116 = keys[2] < tmp_101 || (keys[2] == tmp_101 && values[2] < tmp_102);
            if tmp_113 == tmp_116 { keys[2] = tmp_101; values[2] = tmp_102; }
            let tmp_117 = keys[3] < tmp_103 || (keys[3] == tmp_103 && values[3] < tmp_104);
            if tmp_113 == tmp_117 { keys[3] = tmp_103; values[3] = tmp_104; }
            let tmp_118 = keys[4] < tmp_105 || (keys[4] == tmp_105 && values[4] < tmp_106);
            if tmp_113 == tmp_118 { keys[4] = tmp_105; values[4] = tmp_106; }
            let tmp_119 = keys[5] < tmp_107 || (keys[5] == tmp_107 && values[5] < tmp_108);
            if tmp_113 == tmp_119 { keys[5] = tmp_107; values[5] = tmp_108; }
            let tmp_120 = keys[6] < tmp_109 || (keys[6] == tmp_109 && values[6] < tmp_110);
            if tmp_113 == tmp_120 { keys[6] = tmp_109; values[6] = tmp_110; }
            let tmp_121 = keys[7] < tmp_111 || (keys[7] == tmp_111 && values[7] < tmp_112);
            if tmp_113 == tmp_121 { keys[7] = tmp_111; values[7] = tmp_112; }
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],1,0)
        {
            let tmp_122 = subgroupShuffleXor(keys[0], 1u);
            let tmp_123 = subgroupShuffleXor(values[0], 1u);
            let tmp_124 = subgroupShuffleXor(keys[1], 1u);
            let tmp_125 = subgroupShuffleXor(values[1], 1u);
            let tmp_126 = subgroupShuffleXor(keys[2], 1u);
            let tmp_127 = subgroupShuffleXor(values[2], 1u);
            let tmp_128 = subgroupShuffleXor(keys[3], 1u);
            let tmp_129 = subgroupShuffleXor(values[3], 1u);
            let tmp_130 = subgroupShuffleXor(keys[4], 1u);
            let tmp_131 = subgroupShuffleXor(values[4], 1u);
            let tmp_132 = subgroupShuffleXor(keys[5], 1u);
            let tmp_133 = subgroupShuffleXor(values[5], 1u);
            let tmp_134 = subgroupShuffleXor(keys[6], 1u);
            let tmp_135 = subgroupShuffleXor(values[6], 1u);
            let tmp_136 = subgroupShuffleXor(keys[7], 1u);
            let tmp_137 = subgroupShuffleXor(values[7], 1u);
            let tmp_138 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_139 = keys[0] < tmp_122 || (keys[0] == tmp_122 && values[0] < tmp_123);
            if tmp_138 == tmp_139 { keys[0] = tmp_122; values[0] = tmp_123; }
            let tmp_140 = keys[1] < tmp_124 || (keys[1] == tmp_124 && values[1] < tmp_125);
            if tmp_138 == tmp_140 { keys[1] = tmp_124; values[1] = tmp_125; }
            let tmp_141 = keys[2] < tmp_126 || (keys[2] == tmp_126 && values[2] < tmp_127);
            if tmp_138 == tmp_141 { keys[2] = tmp_126; values[2] = tmp_127; }
            let tmp_142 = keys[3] < tmp_128 || (keys[3] == tmp_128 && values[3] < tmp_129);
            if tmp_138 == tmp_142 { keys[3] = tmp_128; values[3] = tmp_129; }
            let tmp_143 = keys[4] < tmp_130 || (keys[4] == tmp_130 && values[4] < tmp_131);
            if tmp_138 == tmp_143 { keys[4] = tmp_130; values[4] = tmp_131; }
            let tmp_144 = keys[5] < tmp_132 || (keys[5] == tmp_132 && values[5] < tmp_133);
            if tmp_138 == tmp_144 { keys[5] = tmp_132; values[5] = tmp_133; }
            let tmp_145 = keys[6] < tmp_134 || (keys[6] == tmp_134 && values[6] < tmp_135);
            if tmp_138 == tmp_145 { keys[6] = tmp_134; values[6] = tmp_135; }
            let tmp_146 = keys[7] < tmp_136 || (keys[7] == tmp_136 && values[7] < tmp_137);
            if tmp_138 == tmp_146 { keys[7] = tmp_136; values[7] = tmp_137; }
        }
    }

    // exch_local(4,8)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_147 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_147;
                let tmp_148 = values[0]; values[0] = values[4]; values[4] = tmp_148;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_149 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_149;
                let tmp_150 = values[1]; values[1] = values[5]; values[5] = tmp_150;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_151 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_151;
                let tmp_152 = values[2]; values[2] = values[6]; values[6] = tmp_152;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_153 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_153;
                let tmp_154 = values[3]; values[3] = values[7]; values[7] = tmp_154;
            }
        }
    }

    // exch_local(2,8)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_155 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_155;
                let tmp_156 = values[0]; values[0] = values[2]; values[2] = tmp_156;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_157 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_157;
                let tmp_158 = values[1]; values[1] = values[3]; values[3] = tmp_158;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_159 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_159;
                let tmp_160 = values[4]; values[4] = values[6]; values[6] = tmp_160;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_161 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_161;
                let tmp_162 = values[5]; values[5] = values[7]; values[7] = tmp_162;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_163 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_163;
                let tmp_164 = values[0]; values[0] = values[1]; values[1] = tmp_164;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_165 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_165;
                let tmp_166 = values[2]; values[2] = values[3]; values[3] = tmp_166;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_167 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_167;
                let tmp_168 = values[4]; values[4] = values[5]; values[5] = tmp_168;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_169 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_169;
                let tmp_170 = values[6]; values[6] = values[7]; values[7] = tmp_170;
            }
        }
    }

    // exch_intxn(tmask:7,swbit:2,wpt:8)
    {
        // _exch_subgroup([(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)],7,2)
        {
            let tmp_171 = subgroupShuffleXor(keys[7], 7u);
            let tmp_172 = subgroupShuffleXor(values[7], 7u);
            let tmp_173 = subgroupShuffleXor(keys[6], 7u);
            let tmp_174 = subgroupShuffleXor(values[6], 7u);
            let tmp_175 = subgroupShuffleXor(keys[5], 7u);
            let tmp_176 = subgroupShuffleXor(values[5], 7u);
            let tmp_177 = subgroupShuffleXor(keys[4], 7u);
            let tmp_178 = subgroupShuffleXor(values[4], 7u);
            let tmp_179 = subgroupShuffleXor(keys[3], 7u);
            let tmp_180 = subgroupShuffleXor(values[3], 7u);
            let tmp_181 = subgroupShuffleXor(keys[2], 7u);
            let tmp_182 = subgroupShuffleXor(values[2], 7u);
            let tmp_183 = subgroupShuffleXor(keys[1], 7u);
            let tmp_184 = subgroupShuffleXor(values[1], 7u);
            let tmp_185 = subgroupShuffleXor(keys[0], 7u);
            let tmp_186 = subgroupShuffleXor(values[0], 7u);
            let tmp_187 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_188 = keys[0] < tmp_171 || (keys[0] == tmp_171 && values[0] < tmp_172);
            if tmp_187 == tmp_188 { keys[0] = tmp_171; values[0] = tmp_172; }
            let tmp_189 = keys[1] < tmp_173 || (keys[1] == tmp_173 && values[1] < tmp_174);
            if tmp_187 == tmp_189 { keys[1] = tmp_173; values[1] = tmp_174; }
            let tmp_190 = keys[2] < tmp_175 || (keys[2] == tmp_175 && values[2] < tmp_176);
            if tmp_187 == tmp_190 { keys[2] = tmp_175; values[2] = tmp_176; }
            let tmp_191 = keys[3] < tmp_177 || (keys[3] == tmp_177 && values[3] < tmp_178);
            if tmp_187 == tmp_191 { keys[3] = tmp_177; values[3] = tmp_178; }
            let tmp_192 = keys[4] < tmp_179 || (keys[4] == tmp_179 && values[4] < tmp_180);
            if tmp_187 == tmp_192 { keys[4] = tmp_179; values[4] = tmp_180; }
            let tmp_193 = keys[5] < tmp_181 || (keys[5] == tmp_181 && values[5] < tmp_182);
            if tmp_187 == tmp_193 { keys[5] = tmp_181; values[5] = tmp_182; }
            let tmp_194 = keys[6] < tmp_183 || (keys[6] == tmp_183 && values[6] < tmp_184);
            if tmp_187 == tmp_194 { keys[6] = tmp_183; values[6] = tmp_184; }
            let tmp_195 = keys[7] < tmp_185 || (keys[7] == tmp_185 && values[7] < tmp_186);
            if tmp_187 == tmp_195 { keys[7] = tmp_185; values[7] = tmp_186; }
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],2,1)
        {
            let tmp_196 = subgroupShuffleXor(keys[0], 2u);
            let tmp_197 = subgroupShuffleXor(values[0], 2u);
            let tmp_198 = subgroupShuffleXor(keys[1], 2u);
            let tmp_199 = subgroupShuffleXor(values[1], 2u);
            let tmp_200 = subgroupShuffleXor(keys[2], 2u);
            let tmp_201 = subgroupShuffleXor(values[2], 2u);
            let tmp_202 = subgroupShuffleXor(keys[3], 2u);
            let tmp_203 = subgroupShuffleXor(values[3], 2u);
            let tmp_204 = subgroupShuffleXor(keys[4], 2u);
            let tmp_205 = subgroupShuffleXor(values[4], 2u);
            let tmp_206 = subgroupShuffleXor(keys[5], 2u);
            let tmp_207 = subgroupShuffleXor(values[5], 2u);
            let tmp_208 = subgroupShuffleXor(keys[6], 2u);
            let tmp_209 = subgroupShuffleXor(values[6], 2u);
            let tmp_210 = subgroupShuffleXor(keys[7], 2u);
            let tmp_211 = subgroupShuffleXor(values[7], 2u);
            let tmp_212 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_213 = keys[0] < tmp_196 || (keys[0] == tmp_196 && values[0] < tmp_197);
            if tmp_212 == tmp_213 { keys[0] = tmp_196; values[0] = tmp_197; }
            let tmp_214 = keys[1] < tmp_198 || (keys[1] == tmp_198 && values[1] < tmp_199);
            if tmp_212 == tmp_214 { keys[1] = tmp_198; values[1] = tmp_199; }
            let tmp_215 = keys[2] < tmp_200 || (keys[2] == tmp_200 && values[2] < tmp_201);
            if tmp_212 == tmp_215 { keys[2] = tmp_200; values[2] = tmp_201; }
            let tmp_216 = keys[3] < tmp_202 || (keys[3] == tmp_202 && values[3] < tmp_203);
            if tmp_212 == tmp_216 { keys[3] = tmp_202; values[3] = tmp_203; }
            let tmp_217 = keys[4] < tmp_204 || (keys[4] == tmp_204 && values[4] < tmp_205);
            if tmp_212 == tmp_217 { keys[4] = tmp_204; values[4] = tmp_205; }
            let tmp_218 = keys[5] < tmp_206 || (keys[5] == tmp_206 && values[5] < tmp_207);
            if tmp_212 == tmp_218 { keys[5] = tmp_206; values[5] = tmp_207; }
            let tmp_219 = keys[6] < tmp_208 || (keys[6] == tmp_208 && values[6] < tmp_209);
            if tmp_212 == tmp_219 { keys[6] = tmp_208; values[6] = tmp_209; }
            let tmp_220 = keys[7] < tmp_210 || (keys[7] == tmp_210 && values[7] < tmp_211);
            if tmp_212 == tmp_220 { keys[7] = tmp_210; values[7] = tmp_211; }
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],1,0)
        {
            let tmp_221 = subgroupShuffleXor(keys[0], 1u);
            let tmp_222 = subgroupShuffleXor(values[0], 1u);
            let tmp_223 = subgroupShuffleXor(keys[1], 1u);
            let tmp_224 = subgroupShuffleXor(values[1], 1u);
            let tmp_225 = subgroupShuffleXor(keys[2], 1u);
            let tmp_226 = subgroupShuffleXor(values[2], 1u);
            let tmp_227 = subgroupShuffleXor(keys[3], 1u);
            let tmp_228 = subgroupShuffleXor(values[3], 1u);
            let tmp_229 = subgroupShuffleXor(keys[4], 1u);
            let tmp_230 = subgroupShuffleXor(values[4], 1u);
            let tmp_231 = subgroupShuffleXor(keys[5], 1u);
            let tmp_232 = subgroupShuffleXor(values[5], 1u);
            let tmp_233 = subgroupShuffleXor(keys[6], 1u);
            let tmp_234 = subgroupShuffleXor(values[6], 1u);
            let tmp_235 = subgroupShuffleXor(keys[7], 1u);
            let tmp_236 = subgroupShuffleXor(values[7], 1u);
            let tmp_237 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_238 = keys[0] < tmp_221 || (keys[0] == tmp_221 && values[0] < tmp_222);
            if tmp_237 == tmp_238 { keys[0] = tmp_221; values[0] = tmp_222; }
            let tmp_239 = keys[1] < tmp_223 || (keys[1] == tmp_223 && values[1] < tmp_224);
            if tmp_237 == tmp_239 { keys[1] = tmp_223; values[1] = tmp_224; }
            let tmp_240 = keys[2] < tmp_225 || (keys[2] == tmp_225 && values[2] < tmp_226);
            if tmp_237 == tmp_240 { keys[2] = tmp_225; values[2] = tmp_226; }
            let tmp_241 = keys[3] < tmp_227 || (keys[3] == tmp_227 && values[3] < tmp_228);
            if tmp_237 == tmp_241 { keys[3] = tmp_227; values[3] = tmp_228; }
            let tmp_242 = keys[4] < tmp_229 || (keys[4] == tmp_229 && values[4] < tmp_230);
            if tmp_237 == tmp_242 { keys[4] = tmp_229; values[4] = tmp_230; }
            let tmp_243 = keys[5] < tmp_231 || (keys[5] == tmp_231 && values[5] < tmp_232);
            if tmp_237 == tmp_243 { keys[5] = tmp_231; values[5] = tmp_232; }
            let tmp_244 = keys[6] < tmp_233 || (keys[6] == tmp_233 && values[6] < tmp_234);
            if tmp_237 == tmp_244 { keys[6] = tmp_233; values[6] = tmp_234; }
            let tmp_245 = keys[7] < tmp_235 || (keys[7] == tmp_235 && values[7] < tmp_236);
            if tmp_237 == tmp_245 { keys[7] = tmp_235; values[7] = tmp_236; }
        }
    }

    // exch_local(4,8)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_246 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_246;
                let tmp_247 = values[0]; values[0] = values[4]; values[4] = tmp_247;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_248 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_248;
                let tmp_249 = values[1]; values[1] = values[5]; values[5] = tmp_249;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_250 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_250;
                let tmp_251 = values[2]; values[2] = values[6]; values[6] = tmp_251;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_252 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_252;
                let tmp_253 = values[3]; values[3] = values[7]; values[7] = tmp_253;
            }
        }
    }

    // exch_local(2,8)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_254 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_254;
                let tmp_255 = values[0]; values[0] = values[2]; values[2] = tmp_255;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_256 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_256;
                let tmp_257 = values[1]; values[1] = values[3]; values[3] = tmp_257;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_258 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_258;
                let tmp_259 = values[4]; values[4] = values[6]; values[6] = tmp_259;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_260 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_260;
                let tmp_261 = values[5]; values[5] = values[7]; values[7] = tmp_261;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_262 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_262;
                let tmp_263 = values[0]; values[0] = values[1]; values[1] = tmp_263;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_264 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_264;
                let tmp_265 = values[2]; values[2] = values[3]; values[3] = tmp_265;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_266 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_266;
                let tmp_267 = values[4]; values[4] = values[5]; values[5] = tmp_267;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_268 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_268;
                let tmp_269 = values[6]; values[6] = values[7]; values[7] = tmp_269;
            }
        }
    }

    // exch_intxn(tmask:15,swbit:3,wpt:8)
    {
        // _exch_subgroup([(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)],15,3)
        {
            let tmp_270 = subgroupShuffleXor(keys[7], 15u);
            let tmp_271 = subgroupShuffleXor(values[7], 15u);
            let tmp_272 = subgroupShuffleXor(keys[6], 15u);
            let tmp_273 = subgroupShuffleXor(values[6], 15u);
            let tmp_274 = subgroupShuffleXor(keys[5], 15u);
            let tmp_275 = subgroupShuffleXor(values[5], 15u);
            let tmp_276 = subgroupShuffleXor(keys[4], 15u);
            let tmp_277 = subgroupShuffleXor(values[4], 15u);
            let tmp_278 = subgroupShuffleXor(keys[3], 15u);
            let tmp_279 = subgroupShuffleXor(values[3], 15u);
            let tmp_280 = subgroupShuffleXor(keys[2], 15u);
            let tmp_281 = subgroupShuffleXor(values[2], 15u);
            let tmp_282 = subgroupShuffleXor(keys[1], 15u);
            let tmp_283 = subgroupShuffleXor(values[1], 15u);
            let tmp_284 = subgroupShuffleXor(keys[0], 15u);
            let tmp_285 = subgroupShuffleXor(values[0], 15u);
            let tmp_286 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_287 = keys[0] < tmp_270 || (keys[0] == tmp_270 && values[0] < tmp_271);
            if tmp_286 == tmp_287 { keys[0] = tmp_270; values[0] = tmp_271; }
            let tmp_288 = keys[1] < tmp_272 || (keys[1] == tmp_272 && values[1] < tmp_273);
            if tmp_286 == tmp_288 { keys[1] = tmp_272; values[1] = tmp_273; }
            let tmp_289 = keys[2] < tmp_274 || (keys[2] == tmp_274 && values[2] < tmp_275);
            if tmp_286 == tmp_289 { keys[2] = tmp_274; values[2] = tmp_275; }
            let tmp_290 = keys[3] < tmp_276 || (keys[3] == tmp_276 && values[3] < tmp_277);
            if tmp_286 == tmp_290 { keys[3] = tmp_276; values[3] = tmp_277; }
            let tmp_291 = keys[4] < tmp_278 || (keys[4] == tmp_278 && values[4] < tmp_279);
            if tmp_286 == tmp_291 { keys[4] = tmp_278; values[4] = tmp_279; }
            let tmp_292 = keys[5] < tmp_280 || (keys[5] == tmp_280 && values[5] < tmp_281);
            if tmp_286 == tmp_292 { keys[5] = tmp_280; values[5] = tmp_281; }
            let tmp_293 = keys[6] < tmp_282 || (keys[6] == tmp_282 && values[6] < tmp_283);
            if tmp_286 == tmp_293 { keys[6] = tmp_282; values[6] = tmp_283; }
            let tmp_294 = keys[7] < tmp_284 || (keys[7] == tmp_284 && values[7] < tmp_285);
            if tmp_286 == tmp_294 { keys[7] = tmp_284; values[7] = tmp_285; }
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],4,2)
        {
            let tmp_295 = subgroupShuffleXor(keys[0], 4u);
            let tmp_296 = subgroupShuffleXor(values[0], 4u);
            let tmp_297 = subgroupShuffleXor(keys[1], 4u);
            let tmp_298 = subgroupShuffleXor(values[1], 4u);
            let tmp_299 = subgroupShuffleXor(keys[2], 4u);
            let tmp_300 = subgroupShuffleXor(values[2], 4u);
            let tmp_301 = subgroupShuffleXor(keys[3], 4u);
            let tmp_302 = subgroupShuffleXor(values[3], 4u);
            let tmp_303 = subgroupShuffleXor(keys[4], 4u);
            let tmp_304 = subgroupShuffleXor(values[4], 4u);
            let tmp_305 = subgroupShuffleXor(keys[5], 4u);
            let tmp_306 = subgroupShuffleXor(values[5], 4u);
            let tmp_307 = subgroupShuffleXor(keys[6], 4u);
            let tmp_308 = subgroupShuffleXor(values[6], 4u);
            let tmp_309 = subgroupShuffleXor(keys[7], 4u);
            let tmp_310 = subgroupShuffleXor(values[7], 4u);
            let tmp_311 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_312 = keys[0] < tmp_295 || (keys[0] == tmp_295 && values[0] < tmp_296);
            if tmp_311 == tmp_312 { keys[0] = tmp_295; values[0] = tmp_296; }
            let tmp_313 = keys[1] < tmp_297 || (keys[1] == tmp_297 && values[1] < tmp_298);
            if tmp_311 == tmp_313 { keys[1] = tmp_297; values[1] = tmp_298; }
            let tmp_314 = keys[2] < tmp_299 || (keys[2] == tmp_299 && values[2] < tmp_300);
            if tmp_311 == tmp_314 { keys[2] = tmp_299; values[2] = tmp_300; }
            let tmp_315 = keys[3] < tmp_301 || (keys[3] == tmp_301 && values[3] < tmp_302);
            if tmp_311 == tmp_315 { keys[3] = tmp_301; values[3] = tmp_302; }
            let tmp_316 = keys[4] < tmp_303 || (keys[4] == tmp_303 && values[4] < tmp_304);
            if tmp_311 == tmp_316 { keys[4] = tmp_303; values[4] = tmp_304; }
            let tmp_317 = keys[5] < tmp_305 || (keys[5] == tmp_305 && values[5] < tmp_306);
            if tmp_311 == tmp_317 { keys[5] = tmp_305; values[5] = tmp_306; }
            let tmp_318 = keys[6] < tmp_307 || (keys[6] == tmp_307 && values[6] < tmp_308);
            if tmp_311 == tmp_318 { keys[6] = tmp_307; values[6] = tmp_308; }
            let tmp_319 = keys[7] < tmp_309 || (keys[7] == tmp_309 && values[7] < tmp_310);
            if tmp_311 == tmp_319 { keys[7] = tmp_309; values[7] = tmp_310; }
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],2,1)
        {
            let tmp_320 = subgroupShuffleXor(keys[0], 2u);
            let tmp_321 = subgroupShuffleXor(values[0], 2u);
            let tmp_322 = subgroupShuffleXor(keys[1], 2u);
            let tmp_323 = subgroupShuffleXor(values[1], 2u);
            let tmp_324 = subgroupShuffleXor(keys[2], 2u);
            let tmp_325 = subgroupShuffleXor(values[2], 2u);
            let tmp_326 = subgroupShuffleXor(keys[3], 2u);
            let tmp_327 = subgroupShuffleXor(values[3], 2u);
            let tmp_328 = subgroupShuffleXor(keys[4], 2u);
            let tmp_329 = subgroupShuffleXor(values[4], 2u);
            let tmp_330 = subgroupShuffleXor(keys[5], 2u);
            let tmp_331 = subgroupShuffleXor(values[5], 2u);
            let tmp_332 = subgroupShuffleXor(keys[6], 2u);
            let tmp_333 = subgroupShuffleXor(values[6], 2u);
            let tmp_334 = subgroupShuffleXor(keys[7], 2u);
            let tmp_335 = subgroupShuffleXor(values[7], 2u);
            let tmp_336 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_337 = keys[0] < tmp_320 || (keys[0] == tmp_320 && values[0] < tmp_321);
            if tmp_336 == tmp_337 { keys[0] = tmp_320; values[0] = tmp_321; }
            let tmp_338 = keys[1] < tmp_322 || (keys[1] == tmp_322 && values[1] < tmp_323);
            if tmp_336 == tmp_338 { keys[1] = tmp_322; values[1] = tmp_323; }
            let tmp_339 = keys[2] < tmp_324 || (keys[2] == tmp_324 && values[2] < tmp_325);
            if tmp_336 == tmp_339 { keys[2] = tmp_324; values[2] = tmp_325; }
            let tmp_340 = keys[3] < tmp_326 || (keys[3] == tmp_326 && values[3] < tmp_327);
            if tmp_336 == tmp_340 { keys[3] = tmp_326; values[3] = tmp_327; }
            let tmp_341 = keys[4] < tmp_328 || (keys[4] == tmp_328 && values[4] < tmp_329);
            if tmp_336 == tmp_341 { keys[4] = tmp_328; values[4] = tmp_329; }
            let tmp_342 = keys[5] < tmp_330 || (keys[5] == tmp_330 && values[5] < tmp_331);
            if tmp_336 == tmp_342 { keys[5] = tmp_330; values[5] = tmp_331; }
            let tmp_343 = keys[6] < tmp_332 || (keys[6] == tmp_332 && values[6] < tmp_333);
            if tmp_336 == tmp_343 { keys[6] = tmp_332; values[6] = tmp_333; }
            let tmp_344 = keys[7] < tmp_334 || (keys[7] == tmp_334 && values[7] < tmp_335);
            if tmp_336 == tmp_344 { keys[7] = tmp_334; values[7] = tmp_335; }
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],1,0)
        {
            let tmp_345 = subgroupShuffleXor(keys[0], 1u);
            let tmp_346 = subgroupShuffleXor(values[0], 1u);
            let tmp_347 = subgroupShuffleXor(keys[1], 1u);
            let tmp_348 = subgroupShuffleXor(values[1], 1u);
            let tmp_349 = subgroupShuffleXor(keys[2], 1u);
            let tmp_350 = subgroupShuffleXor(values[2], 1u);
            let tmp_351 = subgroupShuffleXor(keys[3], 1u);
            let tmp_352 = subgroupShuffleXor(values[3], 1u);
            let tmp_353 = subgroupShuffleXor(keys[4], 1u);
            let tmp_354 = subgroupShuffleXor(values[4], 1u);
            let tmp_355 = subgroupShuffleXor(keys[5], 1u);
            let tmp_356 = subgroupShuffleXor(values[5], 1u);
            let tmp_357 = subgroupShuffleXor(keys[6], 1u);
            let tmp_358 = subgroupShuffleXor(values[6], 1u);
            let tmp_359 = subgroupShuffleXor(keys[7], 1u);
            let tmp_360 = subgroupShuffleXor(values[7], 1u);
            let tmp_361 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_362 = keys[0] < tmp_345 || (keys[0] == tmp_345 && values[0] < tmp_346);
            if tmp_361 == tmp_362 { keys[0] = tmp_345; values[0] = tmp_346; }
            let tmp_363 = keys[1] < tmp_347 || (keys[1] == tmp_347 && values[1] < tmp_348);
            if tmp_361 == tmp_363 { keys[1] = tmp_347; values[1] = tmp_348; }
            let tmp_364 = keys[2] < tmp_349 || (keys[2] == tmp_349 && values[2] < tmp_350);
            if tmp_361 == tmp_364 { keys[2] = tmp_349; values[2] = tmp_350; }
            let tmp_365 = keys[3] < tmp_351 || (keys[3] == tmp_351 && values[3] < tmp_352);
            if tmp_361 == tmp_365 { keys[3] = tmp_351; values[3] = tmp_352; }
            let tmp_366 = keys[4] < tmp_353 || (keys[4] == tmp_353 && values[4] < tmp_354);
            if tmp_361 == tmp_366 { keys[4] = tmp_353; values[4] = tmp_354; }
            let tmp_367 = keys[5] < tmp_355 || (keys[5] == tmp_355 && values[5] < tmp_356);
            if tmp_361 == tmp_367 { keys[5] = tmp_355; values[5] = tmp_356; }
            let tmp_368 = keys[6] < tmp_357 || (keys[6] == tmp_357 && values[6] < tmp_358);
            if tmp_361 == tmp_368 { keys[6] = tmp_357; values[6] = tmp_358; }
            let tmp_369 = keys[7] < tmp_359 || (keys[7] == tmp_359 && values[7] < tmp_360);
            if tmp_361 == tmp_369 { keys[7] = tmp_359; values[7] = tmp_360; }
        }
    }

    // exch_local(4,8)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_370 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_370;
                let tmp_371 = values[0]; values[0] = values[4]; values[4] = tmp_371;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_372 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_372;
                let tmp_373 = values[1]; values[1] = values[5]; values[5] = tmp_373;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_374 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_374;
                let tmp_375 = values[2]; values[2] = values[6]; values[6] = tmp_375;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_376 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_376;
                let tmp_377 = values[3]; values[3] = values[7]; values[7] = tmp_377;
            }
        }
    }

    // exch_local(2,8)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_378 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_378;
                let tmp_379 = values[0]; values[0] = values[2]; values[2] = tmp_379;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_380 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_380;
                let tmp_381 = values[1]; values[1] = values[3]; values[3] = tmp_381;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_382 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_382;
                let tmp_383 = values[4]; values[4] = values[6]; values[6] = tmp_383;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_384 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_384;
                let tmp_385 = values[5]; values[5] = values[7]; values[7] = tmp_385;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_386 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_386;
                let tmp_387 = values[0]; values[0] = values[1]; values[1] = tmp_387;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_388 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_388;
                let tmp_389 = values[2]; values[2] = values[3]; values[3] = tmp_389;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_390 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_390;
                let tmp_391 = values[4]; values[4] = values[5]; values[5] = tmp_391;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_392 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_392;
                let tmp_393 = values[6]; values[6] = values[7]; values[7] = tmp_393;
            }
        }
    }

    // exch_intxn(tmask:31,swbit:4,wpt:8)
    {
        // _exch_subgroup([(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)],31,4)
        {
            let tmp_394 = subgroupShuffleXor(keys[7], 31u);
            let tmp_395 = subgroupShuffleXor(values[7], 31u);
            let tmp_396 = subgroupShuffleXor(keys[6], 31u);
            let tmp_397 = subgroupShuffleXor(values[6], 31u);
            let tmp_398 = subgroupShuffleXor(keys[5], 31u);
            let tmp_399 = subgroupShuffleXor(values[5], 31u);
            let tmp_400 = subgroupShuffleXor(keys[4], 31u);
            let tmp_401 = subgroupShuffleXor(values[4], 31u);
            let tmp_402 = subgroupShuffleXor(keys[3], 31u);
            let tmp_403 = subgroupShuffleXor(values[3], 31u);
            let tmp_404 = subgroupShuffleXor(keys[2], 31u);
            let tmp_405 = subgroupShuffleXor(values[2], 31u);
            let tmp_406 = subgroupShuffleXor(keys[1], 31u);
            let tmp_407 = subgroupShuffleXor(values[1], 31u);
            let tmp_408 = subgroupShuffleXor(keys[0], 31u);
            let tmp_409 = subgroupShuffleXor(values[0], 31u);
            let tmp_410 = extractBits(local_tid, 4u, 1u) != 0u;
            let tmp_411 = keys[0] < tmp_394 || (keys[0] == tmp_394 && values[0] < tmp_395);
            if tmp_410 == tmp_411 { keys[0] = tmp_394; values[0] = tmp_395; }
            let tmp_412 = keys[1] < tmp_396 || (keys[1] == tmp_396 && values[1] < tmp_397);
            if tmp_410 == tmp_412 { keys[1] = tmp_396; values[1] = tmp_397; }
            let tmp_413 = keys[2] < tmp_398 || (keys[2] == tmp_398 && values[2] < tmp_399);
            if tmp_410 == tmp_413 { keys[2] = tmp_398; values[2] = tmp_399; }
            let tmp_414 = keys[3] < tmp_400 || (keys[3] == tmp_400 && values[3] < tmp_401);
            if tmp_410 == tmp_414 { keys[3] = tmp_400; values[3] = tmp_401; }
            let tmp_415 = keys[4] < tmp_402 || (keys[4] == tmp_402 && values[4] < tmp_403);
            if tmp_410 == tmp_415 { keys[4] = tmp_402; values[4] = tmp_403; }
            let tmp_416 = keys[5] < tmp_404 || (keys[5] == tmp_404 && values[5] < tmp_405);
            if tmp_410 == tmp_416 { keys[5] = tmp_404; values[5] = tmp_405; }
            let tmp_417 = keys[6] < tmp_406 || (keys[6] == tmp_406 && values[6] < tmp_407);
            if tmp_410 == tmp_417 { keys[6] = tmp_406; values[6] = tmp_407; }
            let tmp_418 = keys[7] < tmp_408 || (keys[7] == tmp_408 && values[7] < tmp_409);
            if tmp_410 == tmp_418 { keys[7] = tmp_408; values[7] = tmp_409; }
        }
    }

    // exch_paral(tmask:8,swbit:3,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],8,3)
        {
            let tmp_419 = subgroupShuffleXor(keys[0], 8u);
            let tmp_420 = subgroupShuffleXor(values[0], 8u);
            let tmp_421 = subgroupShuffleXor(keys[1], 8u);
            let tmp_422 = subgroupShuffleXor(values[1], 8u);
            let tmp_423 = subgroupShuffleXor(keys[2], 8u);
            let tmp_424 = subgroupShuffleXor(values[2], 8u);
            let tmp_425 = subgroupShuffleXor(keys[3], 8u);
            let tmp_426 = subgroupShuffleXor(values[3], 8u);
            let tmp_427 = subgroupShuffleXor(keys[4], 8u);
            let tmp_428 = subgroupShuffleXor(values[4], 8u);
            let tmp_429 = subgroupShuffleXor(keys[5], 8u);
            let tmp_430 = subgroupShuffleXor(values[5], 8u);
            let tmp_431 = subgroupShuffleXor(keys[6], 8u);
            let tmp_432 = subgroupShuffleXor(values[6], 8u);
            let tmp_433 = subgroupShuffleXor(keys[7], 8u);
            let tmp_434 = subgroupShuffleXor(values[7], 8u);
            let tmp_435 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_436 = keys[0] < tmp_419 || (keys[0] == tmp_419 && values[0] < tmp_420);
            if tmp_435 == tmp_436 { keys[0] = tmp_419; values[0] = tmp_420; }
            let tmp_437 = keys[1] < tmp_421 || (keys[1] == tmp_421 && values[1] < tmp_422);
            if tmp_435 == tmp_437 { keys[1] = tmp_421; values[1] = tmp_422; }
            let tmp_438 = keys[2] < tmp_423 || (keys[2] == tmp_423 && values[2] < tmp_424);
            if tmp_435 == tmp_438 { keys[2] = tmp_423; values[2] = tmp_424; }
            let tmp_439 = keys[3] < tmp_425 || (keys[3] == tmp_425 && values[3] < tmp_426);
            if tmp_435 == tmp_439 { keys[3] = tmp_425; values[3] = tmp_426; }
            let tmp_440 = keys[4] < tmp_427 || (keys[4] == tmp_427 && values[4] < tmp_428);
            if tmp_435 == tmp_440 { keys[4] = tmp_427; values[4] = tmp_428; }
            let tmp_441 = keys[5] < tmp_429 || (keys[5] == tmp_429 && values[5] < tmp_430);
            if tmp_435 == tmp_441 { keys[5] = tmp_429; values[5] = tmp_430; }
            let tmp_442 = keys[6] < tmp_431 || (keys[6] == tmp_431 && values[6] < tmp_432);
            if tmp_435 == tmp_442 { keys[6] = tmp_431; values[6] = tmp_432; }
            let tmp_443 = keys[7] < tmp_433 || (keys[7] == tmp_433 && values[7] < tmp_434);
            if tmp_435 == tmp_443 { keys[7] = tmp_433; values[7] = tmp_434; }
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],4,2)
        {
            let tmp_444 = subgroupShuffleXor(keys[0], 4u);
            let tmp_445 = subgroupShuffleXor(values[0], 4u);
            let tmp_446 = subgroupShuffleXor(keys[1], 4u);
            let tmp_447 = subgroupShuffleXor(values[1], 4u);
            let tmp_448 = subgroupShuffleXor(keys[2], 4u);
            let tmp_449 = subgroupShuffleXor(values[2], 4u);
            let tmp_450 = subgroupShuffleXor(keys[3], 4u);
            let tmp_451 = subgroupShuffleXor(values[3], 4u);
            let tmp_452 = subgroupShuffleXor(keys[4], 4u);
            let tmp_453 = subgroupShuffleXor(values[4], 4u);
            let tmp_454 = subgroupShuffleXor(keys[5], 4u);
            let tmp_455 = subgroupShuffleXor(values[5], 4u);
            let tmp_456 = subgroupShuffleXor(keys[6], 4u);
            let tmp_457 = subgroupShuffleXor(values[6], 4u);
            let tmp_458 = subgroupShuffleXor(keys[7], 4u);
            let tmp_459 = subgroupShuffleXor(values[7], 4u);
            let tmp_460 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_461 = keys[0] < tmp_444 || (keys[0] == tmp_444 && values[0] < tmp_445);
            if tmp_460 == tmp_461 { keys[0] = tmp_444; values[0] = tmp_445; }
            let tmp_462 = keys[1] < tmp_446 || (keys[1] == tmp_446 && values[1] < tmp_447);
            if tmp_460 == tmp_462 { keys[1] = tmp_446; values[1] = tmp_447; }
            let tmp_463 = keys[2] < tmp_448 || (keys[2] == tmp_448 && values[2] < tmp_449);
            if tmp_460 == tmp_463 { keys[2] = tmp_448; values[2] = tmp_449; }
            let tmp_464 = keys[3] < tmp_450 || (keys[3] == tmp_450 && values[3] < tmp_451);
            if tmp_460 == tmp_464 { keys[3] = tmp_450; values[3] = tmp_451; }
            let tmp_465 = keys[4] < tmp_452 || (keys[4] == tmp_452 && values[4] < tmp_453);
            if tmp_460 == tmp_465 { keys[4] = tmp_452; values[4] = tmp_453; }
            let tmp_466 = keys[5] < tmp_454 || (keys[5] == tmp_454 && values[5] < tmp_455);
            if tmp_460 == tmp_466 { keys[5] = tmp_454; values[5] = tmp_455; }
            let tmp_467 = keys[6] < tmp_456 || (keys[6] == tmp_456 && values[6] < tmp_457);
            if tmp_460 == tmp_467 { keys[6] = tmp_456; values[6] = tmp_457; }
            let tmp_468 = keys[7] < tmp_458 || (keys[7] == tmp_458 && values[7] < tmp_459);
            if tmp_460 == tmp_468 { keys[7] = tmp_458; values[7] = tmp_459; }
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],2,1)
        {
            let tmp_469 = subgroupShuffleXor(keys[0], 2u);
            let tmp_470 = subgroupShuffleXor(values[0], 2u);
            let tmp_471 = subgroupShuffleXor(keys[1], 2u);
            let tmp_472 = subgroupShuffleXor(values[1], 2u);
            let tmp_473 = subgroupShuffleXor(keys[2], 2u);
            let tmp_474 = subgroupShuffleXor(values[2], 2u);
            let tmp_475 = subgroupShuffleXor(keys[3], 2u);
            let tmp_476 = subgroupShuffleXor(values[3], 2u);
            let tmp_477 = subgroupShuffleXor(keys[4], 2u);
            let tmp_478 = subgroupShuffleXor(values[4], 2u);
            let tmp_479 = subgroupShuffleXor(keys[5], 2u);
            let tmp_480 = subgroupShuffleXor(values[5], 2u);
            let tmp_481 = subgroupShuffleXor(keys[6], 2u);
            let tmp_482 = subgroupShuffleXor(values[6], 2u);
            let tmp_483 = subgroupShuffleXor(keys[7], 2u);
            let tmp_484 = subgroupShuffleXor(values[7], 2u);
            let tmp_485 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_486 = keys[0] < tmp_469 || (keys[0] == tmp_469 && values[0] < tmp_470);
            if tmp_485 == tmp_486 { keys[0] = tmp_469; values[0] = tmp_470; }
            let tmp_487 = keys[1] < tmp_471 || (keys[1] == tmp_471 && values[1] < tmp_472);
            if tmp_485 == tmp_487 { keys[1] = tmp_471; values[1] = tmp_472; }
            let tmp_488 = keys[2] < tmp_473 || (keys[2] == tmp_473 && values[2] < tmp_474);
            if tmp_485 == tmp_488 { keys[2] = tmp_473; values[2] = tmp_474; }
            let tmp_489 = keys[3] < tmp_475 || (keys[3] == tmp_475 && values[3] < tmp_476);
            if tmp_485 == tmp_489 { keys[3] = tmp_475; values[3] = tmp_476; }
            let tmp_490 = keys[4] < tmp_477 || (keys[4] == tmp_477 && values[4] < tmp_478);
            if tmp_485 == tmp_490 { keys[4] = tmp_477; values[4] = tmp_478; }
            let tmp_491 = keys[5] < tmp_479 || (keys[5] == tmp_479 && values[5] < tmp_480);
            if tmp_485 == tmp_491 { keys[5] = tmp_479; values[5] = tmp_480; }
            let tmp_492 = keys[6] < tmp_481 || (keys[6] == tmp_481 && values[6] < tmp_482);
            if tmp_485 == tmp_492 { keys[6] = tmp_481; values[6] = tmp_482; }
            let tmp_493 = keys[7] < tmp_483 || (keys[7] == tmp_483 && values[7] < tmp_484);
            if tmp_485 == tmp_493 { keys[7] = tmp_483; values[7] = tmp_484; }
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],1,0)
        {
            let tmp_494 = subgroupShuffleXor(keys[0], 1u);
            let tmp_495 = subgroupShuffleXor(values[0], 1u);
            let tmp_496 = subgroupShuffleXor(keys[1], 1u);
            let tmp_497 = subgroupShuffleXor(values[1], 1u);
            let tmp_498 = subgroupShuffleXor(keys[2], 1u);
            let tmp_499 = subgroupShuffleXor(values[2], 1u);
            let tmp_500 = subgroupShuffleXor(keys[3], 1u);
            let tmp_501 = subgroupShuffleXor(values[3], 1u);
            let tmp_502 = subgroupShuffleXor(keys[4], 1u);
            let tmp_503 = subgroupShuffleXor(values[4], 1u);
            let tmp_504 = subgroupShuffleXor(keys[5], 1u);
            let tmp_505 = subgroupShuffleXor(values[5], 1u);
            let tmp_506 = subgroupShuffleXor(keys[6], 1u);
            let tmp_507 = subgroupShuffleXor(values[6], 1u);
            let tmp_508 = subgroupShuffleXor(keys[7], 1u);
            let tmp_509 = subgroupShuffleXor(values[7], 1u);
            let tmp_510 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_511 = keys[0] < tmp_494 || (keys[0] == tmp_494 && values[0] < tmp_495);
            if tmp_510 == tmp_511 { keys[0] = tmp_494; values[0] = tmp_495; }
            let tmp_512 = keys[1] < tmp_496 || (keys[1] == tmp_496 && values[1] < tmp_497);
            if tmp_510 == tmp_512 { keys[1] = tmp_496; values[1] = tmp_497; }
            let tmp_513 = keys[2] < tmp_498 || (keys[2] == tmp_498 && values[2] < tmp_499);
            if tmp_510 == tmp_513 { keys[2] = tmp_498; values[2] = tmp_499; }
            let tmp_514 = keys[3] < tmp_500 || (keys[3] == tmp_500 && values[3] < tmp_501);
            if tmp_510 == tmp_514 { keys[3] = tmp_500; values[3] = tmp_501; }
            let tmp_515 = keys[4] < tmp_502 || (keys[4] == tmp_502 && values[4] < tmp_503);
            if tmp_510 == tmp_515 { keys[4] = tmp_502; values[4] = tmp_503; }
            let tmp_516 = keys[5] < tmp_504 || (keys[5] == tmp_504 && values[5] < tmp_505);
            if tmp_510 == tmp_516 { keys[5] = tmp_504; values[5] = tmp_505; }
            let tmp_517 = keys[6] < tmp_506 || (keys[6] == tmp_506 && values[6] < tmp_507);
            if tmp_510 == tmp_517 { keys[6] = tmp_506; values[6] = tmp_507; }
            let tmp_518 = keys[7] < tmp_508 || (keys[7] == tmp_508 && values[7] < tmp_509);
            if tmp_510 == tmp_518 { keys[7] = tmp_508; values[7] = tmp_509; }
        }
    }

    // exch_local(4,8)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_519 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_519;
                let tmp_520 = values[0]; values[0] = values[4]; values[4] = tmp_520;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_521 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_521;
                let tmp_522 = values[1]; values[1] = values[5]; values[5] = tmp_522;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_523 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_523;
                let tmp_524 = values[2]; values[2] = values[6]; values[6] = tmp_524;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_525 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_525;
                let tmp_526 = values[3]; values[3] = values[7]; values[7] = tmp_526;
            }
        }
    }

    // exch_local(2,8)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_527 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_527;
                let tmp_528 = values[0]; values[0] = values[2]; values[2] = tmp_528;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_529 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_529;
                let tmp_530 = values[1]; values[1] = values[3]; values[3] = tmp_530;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_531 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_531;
                let tmp_532 = values[4]; values[4] = values[6]; values[6] = tmp_532;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_533 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_533;
                let tmp_534 = values[5]; values[5] = values[7]; values[7] = tmp_534;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_535 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_535;
                let tmp_536 = values[0]; values[0] = values[1]; values[1] = tmp_536;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_537 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_537;
                let tmp_538 = values[2]; values[2] = values[3]; values[3] = tmp_538;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_539 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_539;
                let tmp_540 = values[4]; values[4] = values[5]; values[5] = tmp_540;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_541 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_541;
                let tmp_542 = values[6]; values[6] = values[7]; values[7] = tmp_542;
            }
        }
    }

    // swap(lane_bit:0,register_bit:0,hi_register_index:1)
        {
            let lo_ex_k = subgroupShuffleXor(keys[0], 1u);
            let lo_ex_v = subgroupShuffleXor(values[0], 1u);
            let hi_ex_k = subgroupShuffleXor(keys[1], 1u);
            let hi_ex_v = subgroupShuffleXor(values[1], 1u);
            if ((local_tid & 1u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[1] = lo_ex_k;
                values[1] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[0] = hi_ex_k;
                values[0] = hi_ex_v;
            }
        }
    // swap(lane_bit:0,register_bit:0,hi_register_index:3)
        {
            let lo_ex_k = subgroupShuffleXor(keys[2], 1u);
            let lo_ex_v = subgroupShuffleXor(values[2], 1u);
            let hi_ex_k = subgroupShuffleXor(keys[3], 1u);
            let hi_ex_v = subgroupShuffleXor(values[3], 1u);
            if ((local_tid & 1u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[3] = lo_ex_k;
                values[3] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[2] = hi_ex_k;
                values[2] = hi_ex_v;
            }
        }
    // swap(lane_bit:0,register_bit:0,hi_register_index:5)
        {
            let lo_ex_k = subgroupShuffleXor(keys[4], 1u);
            let lo_ex_v = subgroupShuffleXor(values[4], 1u);
            let hi_ex_k = subgroupShuffleXor(keys[5], 1u);
            let hi_ex_v = subgroupShuffleXor(values[5], 1u);
            if ((local_tid & 1u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[5] = lo_ex_k;
                values[5] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[4] = hi_ex_k;
                values[4] = hi_ex_v;
            }
        }
    // swap(lane_bit:0,register_bit:0,hi_register_index:7)
        {
            let lo_ex_k = subgroupShuffleXor(keys[6], 1u);
            let lo_ex_v = subgroupShuffleXor(values[6], 1u);
            let hi_ex_k = subgroupShuffleXor(keys[7], 1u);
            let hi_ex_v = subgroupShuffleXor(values[7], 1u);
            if ((local_tid & 1u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[7] = lo_ex_k;
                values[7] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[6] = hi_ex_k;
                values[6] = hi_ex_v;
            }
        }
    // swap(lane_bit:1,register_bit:1,hi_register_index:2)
        {
            let lo_ex_k = subgroupShuffleXor(keys[0], 2u);
            let lo_ex_v = subgroupShuffleXor(values[0], 2u);
            let hi_ex_k = subgroupShuffleXor(keys[2], 2u);
            let hi_ex_v = subgroupShuffleXor(values[2], 2u);
            if ((local_tid & 2u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[2] = lo_ex_k;
                values[2] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[0] = hi_ex_k;
                values[0] = hi_ex_v;
            }
        }
    // swap(lane_bit:1,register_bit:1,hi_register_index:3)
        {
            let lo_ex_k = subgroupShuffleXor(keys[1], 2u);
            let lo_ex_v = subgroupShuffleXor(values[1], 2u);
            let hi_ex_k = subgroupShuffleXor(keys[3], 2u);
            let hi_ex_v = subgroupShuffleXor(values[3], 2u);
            if ((local_tid & 2u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[3] = lo_ex_k;
                values[3] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[1] = hi_ex_k;
                values[1] = hi_ex_v;
            }
        }
    // swap(lane_bit:1,register_bit:1,hi_register_index:6)
        {
            let lo_ex_k = subgroupShuffleXor(keys[4], 2u);
            let lo_ex_v = subgroupShuffleXor(values[4], 2u);
            let hi_ex_k = subgroupShuffleXor(keys[6], 2u);
            let hi_ex_v = subgroupShuffleXor(values[6], 2u);
            if ((local_tid & 2u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[6] = lo_ex_k;
                values[6] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[4] = hi_ex_k;
                values[4] = hi_ex_v;
            }
        }
    // swap(lane_bit:1,register_bit:1,hi_register_index:7)
        {
            let lo_ex_k = subgroupShuffleXor(keys[5], 2u);
            let lo_ex_v = subgroupShuffleXor(values[5], 2u);
            let hi_ex_k = subgroupShuffleXor(keys[7], 2u);
            let hi_ex_v = subgroupShuffleXor(values[7], 2u);
            if ((local_tid & 2u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[7] = lo_ex_k;
                values[7] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[5] = hi_ex_k;
                values[5] = hi_ex_v;
            }
        }
    // swap(lane_bit:2,register_bit:2,hi_register_index:4)
        {
            let lo_ex_k = subgroupShuffleXor(keys[0], 4u);
            let lo_ex_v = subgroupShuffleXor(values[0], 4u);
            let hi_ex_k = subgroupShuffleXor(keys[4], 4u);
            let hi_ex_v = subgroupShuffleXor(values[4], 4u);
            if ((local_tid & 4u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[4] = lo_ex_k;
                values[4] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[0] = hi_ex_k;
                values[0] = hi_ex_v;
            }
        }
    // swap(lane_bit:2,register_bit:2,hi_register_index:5)
        {
            let lo_ex_k = subgroupShuffleXor(keys[1], 4u);
            let lo_ex_v = subgroupShuffleXor(values[1], 4u);
            let hi_ex_k = subgroupShuffleXor(keys[5], 4u);
            let hi_ex_v = subgroupShuffleXor(values[5], 4u);
            if ((local_tid & 4u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[5] = lo_ex_k;
                values[5] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[1] = hi_ex_k;
                values[1] = hi_ex_v;
            }
        }
    // swap(lane_bit:2,register_bit:2,hi_register_index:6)
        {
            let lo_ex_k = subgroupShuffleXor(keys[2], 4u);
            let lo_ex_v = subgroupShuffleXor(values[2], 4u);
            let hi_ex_k = subgroupShuffleXor(keys[6], 4u);
            let hi_ex_v = subgroupShuffleXor(values[6], 4u);
            if ((local_tid & 4u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[6] = lo_ex_k;
                values[6] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[2] = hi_ex_k;
                values[2] = hi_ex_v;
            }
        }
    // swap(lane_bit:2,register_bit:2,hi_register_index:7)
        {
            let lo_ex_k = subgroupShuffleXor(keys[3], 4u);
            let lo_ex_v = subgroupShuffleXor(values[3], 4u);
            let hi_ex_k = subgroupShuffleXor(keys[7], 4u);
            let hi_ex_v = subgroupShuffleXor(values[7], 4u);
            if ((local_tid & 4u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[7] = lo_ex_k;
                values[7] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[3] = hi_ex_k;
                values[3] = hi_ex_v;
            }
        }
    // swap(lane_bit:3,register_bit:0,hi_register_index:1)
        {
            let lo_ex_k = subgroupShuffleXor(keys[0], 8u);
            let lo_ex_v = subgroupShuffleXor(values[0], 8u);
            let hi_ex_k = subgroupShuffleXor(keys[1], 8u);
            let hi_ex_v = subgroupShuffleXor(values[1], 8u);
            if ((local_tid & 8u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[1] = lo_ex_k;
                values[1] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[0] = hi_ex_k;
                values[0] = hi_ex_v;
            }
        }
    // swap(lane_bit:3,register_bit:0,hi_register_index:3)
        {
            let lo_ex_k = subgroupShuffleXor(keys[2], 8u);
            let lo_ex_v = subgroupShuffleXor(values[2], 8u);
            let hi_ex_k = subgroupShuffleXor(keys[3], 8u);
            let hi_ex_v = subgroupShuffleXor(values[3], 8u);
            if ((local_tid & 8u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[3] = lo_ex_k;
                values[3] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[2] = hi_ex_k;
                values[2] = hi_ex_v;
            }
        }
    // swap(lane_bit:3,register_bit:0,hi_register_index:5)
        {
            let lo_ex_k = subgroupShuffleXor(keys[4], 8u);
            let lo_ex_v = subgroupShuffleXor(values[4], 8u);
            let hi_ex_k = subgroupShuffleXor(keys[5], 8u);
            let hi_ex_v = subgroupShuffleXor(values[5], 8u);
            if ((local_tid & 8u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[5] = lo_ex_k;
                values[5] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[4] = hi_ex_k;
                values[4] = hi_ex_v;
            }
        }
    // swap(lane_bit:3,register_bit:0,hi_register_index:7)
        {
            let lo_ex_k = subgroupShuffleXor(keys[6], 8u);
            let lo_ex_v = subgroupShuffleXor(values[6], 8u);
            let hi_ex_k = subgroupShuffleXor(keys[7], 8u);
            let hi_ex_v = subgroupShuffleXor(values[7], 8u);
            if ((local_tid & 8u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[7] = lo_ex_k;
                values[7] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[6] = hi_ex_k;
                values[6] = hi_ex_v;
            }
        }
    // swap(lane_bit:4,register_bit:1,hi_register_index:2)
        {
            let lo_ex_k = subgroupShuffleXor(keys[0], 16u);
            let lo_ex_v = subgroupShuffleXor(values[0], 16u);
            let hi_ex_k = subgroupShuffleXor(keys[2], 16u);
            let hi_ex_v = subgroupShuffleXor(values[2], 16u);
            if ((local_tid & 16u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[2] = lo_ex_k;
                values[2] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[0] = hi_ex_k;
                values[0] = hi_ex_v;
            }
        }
    // swap(lane_bit:4,register_bit:1,hi_register_index:3)
        {
            let lo_ex_k = subgroupShuffleXor(keys[1], 16u);
            let lo_ex_v = subgroupShuffleXor(values[1], 16u);
            let hi_ex_k = subgroupShuffleXor(keys[3], 16u);
            let hi_ex_v = subgroupShuffleXor(values[3], 16u);
            if ((local_tid & 16u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[3] = lo_ex_k;
                values[3] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[1] = hi_ex_k;
                values[1] = hi_ex_v;
            }
        }
    // swap(lane_bit:4,register_bit:1,hi_register_index:6)
        {
            let lo_ex_k = subgroupShuffleXor(keys[4], 16u);
            let lo_ex_v = subgroupShuffleXor(values[4], 16u);
            let hi_ex_k = subgroupShuffleXor(keys[6], 16u);
            let hi_ex_v = subgroupShuffleXor(values[6], 16u);
            if ((local_tid & 16u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[6] = lo_ex_k;
                values[6] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[4] = hi_ex_k;
                values[4] = hi_ex_v;
            }
        }
    // swap(lane_bit:4,register_bit:1,hi_register_index:7)
        {
            let lo_ex_k = subgroupShuffleXor(keys[5], 16u);
            let lo_ex_v = subgroupShuffleXor(values[5], 16u);
            let hi_ex_k = subgroupShuffleXor(keys[7], 16u);
            let hi_ex_v = subgroupShuffleXor(values[7], 16u);
            if ((local_tid & 16u) == 0u) {
                // lo lane: swap self hi register with pair lo register
                keys[7] = lo_ex_k;
                values[7] = lo_ex_v;
            } else {
                // hi lane: swap self lo register with pair hi register
                keys[5] = hi_ex_k;
                values[5] = hi_ex_v;
            }
        }
        if is_active {
            { let global_offset = 0 * M + local_tid;
                if global_offset < seg_size {
            global_keys[seg_start + global_offset] = keys[0];
            global_value_indices[seg_start + global_offset] = values[0]; } }
            { let global_offset = 2 * M + local_tid;
                if global_offset < seg_size {
            global_keys[seg_start + global_offset] = keys[1];
            global_value_indices[seg_start + global_offset] = values[1]; } }
            { let global_offset = 4 * M + local_tid;
                if global_offset < seg_size {
            global_keys[seg_start + global_offset] = keys[2];
            global_value_indices[seg_start + global_offset] = values[2]; } }
            { let global_offset = 6 * M + local_tid;
                if global_offset < seg_size {
            global_keys[seg_start + global_offset] = keys[3];
            global_value_indices[seg_start + global_offset] = values[3]; } }
            { let global_offset = 1 * M + local_tid;
                if global_offset < seg_size {
            global_keys[seg_start + global_offset] = keys[4];
            global_value_indices[seg_start + global_offset] = values[4]; } }
            { let global_offset = 3 * M + local_tid;
                if global_offset < seg_size {
            global_keys[seg_start + global_offset] = keys[5];
            global_value_indices[seg_start + global_offset] = values[5]; } }
            { let global_offset = 5 * M + local_tid;
                if global_offset < seg_size {
            global_keys[seg_start + global_offset] = keys[6];
            global_value_indices[seg_start + global_offset] = values[6]; } }
            { let global_offset = 7 * M + local_tid;
                if global_offset < seg_size {
            global_keys[seg_start + global_offset] = keys[7];
            global_value_indices[seg_start + global_offset] = values[7]; } }
        }
}
