
enable subgroups;

override WG: u32 = 256u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 2048u;
const M: u32 = 256u;
const WPT: u32 = 8u;
const R: u32 = 8u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybrid_sg8_n2048_m256_block(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 11u;

    let bin_base = select(bin_offsets[BIN - 1u], 0u, BIN == 0u);
    let bin_count = bin_offsets[BIN] - bin_base;

    let local_tid = tid_g % M;
    let seg_base = tid_g - local_tid;
    let wg_index = wg_id.x + wg_id.y * wg_dim.x;
    let global_seg = (wg_index * WG + tid_g) / M;

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
        // _exch_workgroup([(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)],15,3)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            workgroupBarrier();
            let tmp_270 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_271 = seg_base + (local_tid ^ 15u);
            let tmp_272 = smem_keys[tmp_271 * WPT + 7u];
            let tmp_273 = smem_vals[tmp_271 * WPT + 7u];
            let tmp_274 = keys[0] < tmp_272 || (keys[0] == tmp_272 && values[0] < tmp_273);
            if tmp_270 == tmp_274 { keys[0] = tmp_272; values[0] = tmp_273; }
            let tmp_275 = smem_keys[tmp_271 * WPT + 6u];
            let tmp_276 = smem_vals[tmp_271 * WPT + 6u];
            let tmp_277 = keys[1] < tmp_275 || (keys[1] == tmp_275 && values[1] < tmp_276);
            if tmp_270 == tmp_277 { keys[1] = tmp_275; values[1] = tmp_276; }
            let tmp_278 = smem_keys[tmp_271 * WPT + 5u];
            let tmp_279 = smem_vals[tmp_271 * WPT + 5u];
            let tmp_280 = keys[2] < tmp_278 || (keys[2] == tmp_278 && values[2] < tmp_279);
            if tmp_270 == tmp_280 { keys[2] = tmp_278; values[2] = tmp_279; }
            let tmp_281 = smem_keys[tmp_271 * WPT + 4u];
            let tmp_282 = smem_vals[tmp_271 * WPT + 4u];
            let tmp_283 = keys[3] < tmp_281 || (keys[3] == tmp_281 && values[3] < tmp_282);
            if tmp_270 == tmp_283 { keys[3] = tmp_281; values[3] = tmp_282; }
            let tmp_284 = smem_keys[tmp_271 * WPT + 3u];
            let tmp_285 = smem_vals[tmp_271 * WPT + 3u];
            let tmp_286 = keys[4] < tmp_284 || (keys[4] == tmp_284 && values[4] < tmp_285);
            if tmp_270 == tmp_286 { keys[4] = tmp_284; values[4] = tmp_285; }
            let tmp_287 = smem_keys[tmp_271 * WPT + 2u];
            let tmp_288 = smem_vals[tmp_271 * WPT + 2u];
            let tmp_289 = keys[5] < tmp_287 || (keys[5] == tmp_287 && values[5] < tmp_288);
            if tmp_270 == tmp_289 { keys[5] = tmp_287; values[5] = tmp_288; }
            let tmp_290 = smem_keys[tmp_271 * WPT + 1u];
            let tmp_291 = smem_vals[tmp_271 * WPT + 1u];
            let tmp_292 = keys[6] < tmp_290 || (keys[6] == tmp_290 && values[6] < tmp_291);
            if tmp_270 == tmp_292 { keys[6] = tmp_290; values[6] = tmp_291; }
            let tmp_293 = smem_keys[tmp_271 * WPT + 0u];
            let tmp_294 = smem_vals[tmp_271 * WPT + 0u];
            let tmp_295 = keys[7] < tmp_293 || (keys[7] == tmp_293 && values[7] < tmp_294);
            if tmp_270 == tmp_295 { keys[7] = tmp_293; values[7] = tmp_294; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],4,2)
        {
            let tmp_296 = subgroupShuffleXor(keys[0], 4u);
            let tmp_297 = subgroupShuffleXor(values[0], 4u);
            let tmp_298 = subgroupShuffleXor(keys[1], 4u);
            let tmp_299 = subgroupShuffleXor(values[1], 4u);
            let tmp_300 = subgroupShuffleXor(keys[2], 4u);
            let tmp_301 = subgroupShuffleXor(values[2], 4u);
            let tmp_302 = subgroupShuffleXor(keys[3], 4u);
            let tmp_303 = subgroupShuffleXor(values[3], 4u);
            let tmp_304 = subgroupShuffleXor(keys[4], 4u);
            let tmp_305 = subgroupShuffleXor(values[4], 4u);
            let tmp_306 = subgroupShuffleXor(keys[5], 4u);
            let tmp_307 = subgroupShuffleXor(values[5], 4u);
            let tmp_308 = subgroupShuffleXor(keys[6], 4u);
            let tmp_309 = subgroupShuffleXor(values[6], 4u);
            let tmp_310 = subgroupShuffleXor(keys[7], 4u);
            let tmp_311 = subgroupShuffleXor(values[7], 4u);
            let tmp_312 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_313 = keys[0] < tmp_296 || (keys[0] == tmp_296 && values[0] < tmp_297);
            if tmp_312 == tmp_313 { keys[0] = tmp_296; values[0] = tmp_297; }
            let tmp_314 = keys[1] < tmp_298 || (keys[1] == tmp_298 && values[1] < tmp_299);
            if tmp_312 == tmp_314 { keys[1] = tmp_298; values[1] = tmp_299; }
            let tmp_315 = keys[2] < tmp_300 || (keys[2] == tmp_300 && values[2] < tmp_301);
            if tmp_312 == tmp_315 { keys[2] = tmp_300; values[2] = tmp_301; }
            let tmp_316 = keys[3] < tmp_302 || (keys[3] == tmp_302 && values[3] < tmp_303);
            if tmp_312 == tmp_316 { keys[3] = tmp_302; values[3] = tmp_303; }
            let tmp_317 = keys[4] < tmp_304 || (keys[4] == tmp_304 && values[4] < tmp_305);
            if tmp_312 == tmp_317 { keys[4] = tmp_304; values[4] = tmp_305; }
            let tmp_318 = keys[5] < tmp_306 || (keys[5] == tmp_306 && values[5] < tmp_307);
            if tmp_312 == tmp_318 { keys[5] = tmp_306; values[5] = tmp_307; }
            let tmp_319 = keys[6] < tmp_308 || (keys[6] == tmp_308 && values[6] < tmp_309);
            if tmp_312 == tmp_319 { keys[6] = tmp_308; values[6] = tmp_309; }
            let tmp_320 = keys[7] < tmp_310 || (keys[7] == tmp_310 && values[7] < tmp_311);
            if tmp_312 == tmp_320 { keys[7] = tmp_310; values[7] = tmp_311; }
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],2,1)
        {
            let tmp_321 = subgroupShuffleXor(keys[0], 2u);
            let tmp_322 = subgroupShuffleXor(values[0], 2u);
            let tmp_323 = subgroupShuffleXor(keys[1], 2u);
            let tmp_324 = subgroupShuffleXor(values[1], 2u);
            let tmp_325 = subgroupShuffleXor(keys[2], 2u);
            let tmp_326 = subgroupShuffleXor(values[2], 2u);
            let tmp_327 = subgroupShuffleXor(keys[3], 2u);
            let tmp_328 = subgroupShuffleXor(values[3], 2u);
            let tmp_329 = subgroupShuffleXor(keys[4], 2u);
            let tmp_330 = subgroupShuffleXor(values[4], 2u);
            let tmp_331 = subgroupShuffleXor(keys[5], 2u);
            let tmp_332 = subgroupShuffleXor(values[5], 2u);
            let tmp_333 = subgroupShuffleXor(keys[6], 2u);
            let tmp_334 = subgroupShuffleXor(values[6], 2u);
            let tmp_335 = subgroupShuffleXor(keys[7], 2u);
            let tmp_336 = subgroupShuffleXor(values[7], 2u);
            let tmp_337 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_338 = keys[0] < tmp_321 || (keys[0] == tmp_321 && values[0] < tmp_322);
            if tmp_337 == tmp_338 { keys[0] = tmp_321; values[0] = tmp_322; }
            let tmp_339 = keys[1] < tmp_323 || (keys[1] == tmp_323 && values[1] < tmp_324);
            if tmp_337 == tmp_339 { keys[1] = tmp_323; values[1] = tmp_324; }
            let tmp_340 = keys[2] < tmp_325 || (keys[2] == tmp_325 && values[2] < tmp_326);
            if tmp_337 == tmp_340 { keys[2] = tmp_325; values[2] = tmp_326; }
            let tmp_341 = keys[3] < tmp_327 || (keys[3] == tmp_327 && values[3] < tmp_328);
            if tmp_337 == tmp_341 { keys[3] = tmp_327; values[3] = tmp_328; }
            let tmp_342 = keys[4] < tmp_329 || (keys[4] == tmp_329 && values[4] < tmp_330);
            if tmp_337 == tmp_342 { keys[4] = tmp_329; values[4] = tmp_330; }
            let tmp_343 = keys[5] < tmp_331 || (keys[5] == tmp_331 && values[5] < tmp_332);
            if tmp_337 == tmp_343 { keys[5] = tmp_331; values[5] = tmp_332; }
            let tmp_344 = keys[6] < tmp_333 || (keys[6] == tmp_333 && values[6] < tmp_334);
            if tmp_337 == tmp_344 { keys[6] = tmp_333; values[6] = tmp_334; }
            let tmp_345 = keys[7] < tmp_335 || (keys[7] == tmp_335 && values[7] < tmp_336);
            if tmp_337 == tmp_345 { keys[7] = tmp_335; values[7] = tmp_336; }
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],1,0)
        {
            let tmp_346 = subgroupShuffleXor(keys[0], 1u);
            let tmp_347 = subgroupShuffleXor(values[0], 1u);
            let tmp_348 = subgroupShuffleXor(keys[1], 1u);
            let tmp_349 = subgroupShuffleXor(values[1], 1u);
            let tmp_350 = subgroupShuffleXor(keys[2], 1u);
            let tmp_351 = subgroupShuffleXor(values[2], 1u);
            let tmp_352 = subgroupShuffleXor(keys[3], 1u);
            let tmp_353 = subgroupShuffleXor(values[3], 1u);
            let tmp_354 = subgroupShuffleXor(keys[4], 1u);
            let tmp_355 = subgroupShuffleXor(values[4], 1u);
            let tmp_356 = subgroupShuffleXor(keys[5], 1u);
            let tmp_357 = subgroupShuffleXor(values[5], 1u);
            let tmp_358 = subgroupShuffleXor(keys[6], 1u);
            let tmp_359 = subgroupShuffleXor(values[6], 1u);
            let tmp_360 = subgroupShuffleXor(keys[7], 1u);
            let tmp_361 = subgroupShuffleXor(values[7], 1u);
            let tmp_362 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_363 = keys[0] < tmp_346 || (keys[0] == tmp_346 && values[0] < tmp_347);
            if tmp_362 == tmp_363 { keys[0] = tmp_346; values[0] = tmp_347; }
            let tmp_364 = keys[1] < tmp_348 || (keys[1] == tmp_348 && values[1] < tmp_349);
            if tmp_362 == tmp_364 { keys[1] = tmp_348; values[1] = tmp_349; }
            let tmp_365 = keys[2] < tmp_350 || (keys[2] == tmp_350 && values[2] < tmp_351);
            if tmp_362 == tmp_365 { keys[2] = tmp_350; values[2] = tmp_351; }
            let tmp_366 = keys[3] < tmp_352 || (keys[3] == tmp_352 && values[3] < tmp_353);
            if tmp_362 == tmp_366 { keys[3] = tmp_352; values[3] = tmp_353; }
            let tmp_367 = keys[4] < tmp_354 || (keys[4] == tmp_354 && values[4] < tmp_355);
            if tmp_362 == tmp_367 { keys[4] = tmp_354; values[4] = tmp_355; }
            let tmp_368 = keys[5] < tmp_356 || (keys[5] == tmp_356 && values[5] < tmp_357);
            if tmp_362 == tmp_368 { keys[5] = tmp_356; values[5] = tmp_357; }
            let tmp_369 = keys[6] < tmp_358 || (keys[6] == tmp_358 && values[6] < tmp_359);
            if tmp_362 == tmp_369 { keys[6] = tmp_358; values[6] = tmp_359; }
            let tmp_370 = keys[7] < tmp_360 || (keys[7] == tmp_360 && values[7] < tmp_361);
            if tmp_362 == tmp_370 { keys[7] = tmp_360; values[7] = tmp_361; }
        }
    }

    // exch_local(4,8)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_371 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_371;
                let tmp_372 = values[0]; values[0] = values[4]; values[4] = tmp_372;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_373 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_373;
                let tmp_374 = values[1]; values[1] = values[5]; values[5] = tmp_374;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_375 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_375;
                let tmp_376 = values[2]; values[2] = values[6]; values[6] = tmp_376;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_377 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_377;
                let tmp_378 = values[3]; values[3] = values[7]; values[7] = tmp_378;
            }
        }
    }

    // exch_local(2,8)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_379 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_379;
                let tmp_380 = values[0]; values[0] = values[2]; values[2] = tmp_380;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_381 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_381;
                let tmp_382 = values[1]; values[1] = values[3]; values[3] = tmp_382;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_383 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_383;
                let tmp_384 = values[4]; values[4] = values[6]; values[6] = tmp_384;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_385 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_385;
                let tmp_386 = values[5]; values[5] = values[7]; values[7] = tmp_386;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_387 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_387;
                let tmp_388 = values[0]; values[0] = values[1]; values[1] = tmp_388;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_389 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_389;
                let tmp_390 = values[2]; values[2] = values[3]; values[3] = tmp_390;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_391 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_391;
                let tmp_392 = values[4]; values[4] = values[5]; values[5] = tmp_392;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_393 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_393;
                let tmp_394 = values[6]; values[6] = values[7]; values[7] = tmp_394;
            }
        }
    }

    // exch_intxn(tmask:31,swbit:4,wpt:8)
    {
        // _exch_workgroup([(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)],31,4)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            workgroupBarrier();
            let tmp_395 = extractBits(local_tid, 4u, 1u) != 0u;
            let tmp_396 = seg_base + (local_tid ^ 31u);
            let tmp_397 = smem_keys[tmp_396 * WPT + 7u];
            let tmp_398 = smem_vals[tmp_396 * WPT + 7u];
            let tmp_399 = keys[0] < tmp_397 || (keys[0] == tmp_397 && values[0] < tmp_398);
            if tmp_395 == tmp_399 { keys[0] = tmp_397; values[0] = tmp_398; }
            let tmp_400 = smem_keys[tmp_396 * WPT + 6u];
            let tmp_401 = smem_vals[tmp_396 * WPT + 6u];
            let tmp_402 = keys[1] < tmp_400 || (keys[1] == tmp_400 && values[1] < tmp_401);
            if tmp_395 == tmp_402 { keys[1] = tmp_400; values[1] = tmp_401; }
            let tmp_403 = smem_keys[tmp_396 * WPT + 5u];
            let tmp_404 = smem_vals[tmp_396 * WPT + 5u];
            let tmp_405 = keys[2] < tmp_403 || (keys[2] == tmp_403 && values[2] < tmp_404);
            if tmp_395 == tmp_405 { keys[2] = tmp_403; values[2] = tmp_404; }
            let tmp_406 = smem_keys[tmp_396 * WPT + 4u];
            let tmp_407 = smem_vals[tmp_396 * WPT + 4u];
            let tmp_408 = keys[3] < tmp_406 || (keys[3] == tmp_406 && values[3] < tmp_407);
            if tmp_395 == tmp_408 { keys[3] = tmp_406; values[3] = tmp_407; }
            let tmp_409 = smem_keys[tmp_396 * WPT + 3u];
            let tmp_410 = smem_vals[tmp_396 * WPT + 3u];
            let tmp_411 = keys[4] < tmp_409 || (keys[4] == tmp_409 && values[4] < tmp_410);
            if tmp_395 == tmp_411 { keys[4] = tmp_409; values[4] = tmp_410; }
            let tmp_412 = smem_keys[tmp_396 * WPT + 2u];
            let tmp_413 = smem_vals[tmp_396 * WPT + 2u];
            let tmp_414 = keys[5] < tmp_412 || (keys[5] == tmp_412 && values[5] < tmp_413);
            if tmp_395 == tmp_414 { keys[5] = tmp_412; values[5] = tmp_413; }
            let tmp_415 = smem_keys[tmp_396 * WPT + 1u];
            let tmp_416 = smem_vals[tmp_396 * WPT + 1u];
            let tmp_417 = keys[6] < tmp_415 || (keys[6] == tmp_415 && values[6] < tmp_416);
            if tmp_395 == tmp_417 { keys[6] = tmp_415; values[6] = tmp_416; }
            let tmp_418 = smem_keys[tmp_396 * WPT + 0u];
            let tmp_419 = smem_vals[tmp_396 * WPT + 0u];
            let tmp_420 = keys[7] < tmp_418 || (keys[7] == tmp_418 && values[7] < tmp_419);
            if tmp_395 == tmp_420 { keys[7] = tmp_418; values[7] = tmp_419; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:8,swbit:3,wpt:8)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],8,3)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            workgroupBarrier();
            let tmp_421 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_422 = seg_base + (local_tid ^ 8u);
            let tmp_423 = smem_keys[tmp_422 * WPT + 0u];
            let tmp_424 = smem_vals[tmp_422 * WPT + 0u];
            let tmp_425 = keys[0] < tmp_423 || (keys[0] == tmp_423 && values[0] < tmp_424);
            if tmp_421 == tmp_425 { keys[0] = tmp_423; values[0] = tmp_424; }
            let tmp_426 = smem_keys[tmp_422 * WPT + 1u];
            let tmp_427 = smem_vals[tmp_422 * WPT + 1u];
            let tmp_428 = keys[1] < tmp_426 || (keys[1] == tmp_426 && values[1] < tmp_427);
            if tmp_421 == tmp_428 { keys[1] = tmp_426; values[1] = tmp_427; }
            let tmp_429 = smem_keys[tmp_422 * WPT + 2u];
            let tmp_430 = smem_vals[tmp_422 * WPT + 2u];
            let tmp_431 = keys[2] < tmp_429 || (keys[2] == tmp_429 && values[2] < tmp_430);
            if tmp_421 == tmp_431 { keys[2] = tmp_429; values[2] = tmp_430; }
            let tmp_432 = smem_keys[tmp_422 * WPT + 3u];
            let tmp_433 = smem_vals[tmp_422 * WPT + 3u];
            let tmp_434 = keys[3] < tmp_432 || (keys[3] == tmp_432 && values[3] < tmp_433);
            if tmp_421 == tmp_434 { keys[3] = tmp_432; values[3] = tmp_433; }
            let tmp_435 = smem_keys[tmp_422 * WPT + 4u];
            let tmp_436 = smem_vals[tmp_422 * WPT + 4u];
            let tmp_437 = keys[4] < tmp_435 || (keys[4] == tmp_435 && values[4] < tmp_436);
            if tmp_421 == tmp_437 { keys[4] = tmp_435; values[4] = tmp_436; }
            let tmp_438 = smem_keys[tmp_422 * WPT + 5u];
            let tmp_439 = smem_vals[tmp_422 * WPT + 5u];
            let tmp_440 = keys[5] < tmp_438 || (keys[5] == tmp_438 && values[5] < tmp_439);
            if tmp_421 == tmp_440 { keys[5] = tmp_438; values[5] = tmp_439; }
            let tmp_441 = smem_keys[tmp_422 * WPT + 6u];
            let tmp_442 = smem_vals[tmp_422 * WPT + 6u];
            let tmp_443 = keys[6] < tmp_441 || (keys[6] == tmp_441 && values[6] < tmp_442);
            if tmp_421 == tmp_443 { keys[6] = tmp_441; values[6] = tmp_442; }
            let tmp_444 = smem_keys[tmp_422 * WPT + 7u];
            let tmp_445 = smem_vals[tmp_422 * WPT + 7u];
            let tmp_446 = keys[7] < tmp_444 || (keys[7] == tmp_444 && values[7] < tmp_445);
            if tmp_421 == tmp_446 { keys[7] = tmp_444; values[7] = tmp_445; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],4,2)
        {
            let tmp_447 = subgroupShuffleXor(keys[0], 4u);
            let tmp_448 = subgroupShuffleXor(values[0], 4u);
            let tmp_449 = subgroupShuffleXor(keys[1], 4u);
            let tmp_450 = subgroupShuffleXor(values[1], 4u);
            let tmp_451 = subgroupShuffleXor(keys[2], 4u);
            let tmp_452 = subgroupShuffleXor(values[2], 4u);
            let tmp_453 = subgroupShuffleXor(keys[3], 4u);
            let tmp_454 = subgroupShuffleXor(values[3], 4u);
            let tmp_455 = subgroupShuffleXor(keys[4], 4u);
            let tmp_456 = subgroupShuffleXor(values[4], 4u);
            let tmp_457 = subgroupShuffleXor(keys[5], 4u);
            let tmp_458 = subgroupShuffleXor(values[5], 4u);
            let tmp_459 = subgroupShuffleXor(keys[6], 4u);
            let tmp_460 = subgroupShuffleXor(values[6], 4u);
            let tmp_461 = subgroupShuffleXor(keys[7], 4u);
            let tmp_462 = subgroupShuffleXor(values[7], 4u);
            let tmp_463 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_464 = keys[0] < tmp_447 || (keys[0] == tmp_447 && values[0] < tmp_448);
            if tmp_463 == tmp_464 { keys[0] = tmp_447; values[0] = tmp_448; }
            let tmp_465 = keys[1] < tmp_449 || (keys[1] == tmp_449 && values[1] < tmp_450);
            if tmp_463 == tmp_465 { keys[1] = tmp_449; values[1] = tmp_450; }
            let tmp_466 = keys[2] < tmp_451 || (keys[2] == tmp_451 && values[2] < tmp_452);
            if tmp_463 == tmp_466 { keys[2] = tmp_451; values[2] = tmp_452; }
            let tmp_467 = keys[3] < tmp_453 || (keys[3] == tmp_453 && values[3] < tmp_454);
            if tmp_463 == tmp_467 { keys[3] = tmp_453; values[3] = tmp_454; }
            let tmp_468 = keys[4] < tmp_455 || (keys[4] == tmp_455 && values[4] < tmp_456);
            if tmp_463 == tmp_468 { keys[4] = tmp_455; values[4] = tmp_456; }
            let tmp_469 = keys[5] < tmp_457 || (keys[5] == tmp_457 && values[5] < tmp_458);
            if tmp_463 == tmp_469 { keys[5] = tmp_457; values[5] = tmp_458; }
            let tmp_470 = keys[6] < tmp_459 || (keys[6] == tmp_459 && values[6] < tmp_460);
            if tmp_463 == tmp_470 { keys[6] = tmp_459; values[6] = tmp_460; }
            let tmp_471 = keys[7] < tmp_461 || (keys[7] == tmp_461 && values[7] < tmp_462);
            if tmp_463 == tmp_471 { keys[7] = tmp_461; values[7] = tmp_462; }
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],2,1)
        {
            let tmp_472 = subgroupShuffleXor(keys[0], 2u);
            let tmp_473 = subgroupShuffleXor(values[0], 2u);
            let tmp_474 = subgroupShuffleXor(keys[1], 2u);
            let tmp_475 = subgroupShuffleXor(values[1], 2u);
            let tmp_476 = subgroupShuffleXor(keys[2], 2u);
            let tmp_477 = subgroupShuffleXor(values[2], 2u);
            let tmp_478 = subgroupShuffleXor(keys[3], 2u);
            let tmp_479 = subgroupShuffleXor(values[3], 2u);
            let tmp_480 = subgroupShuffleXor(keys[4], 2u);
            let tmp_481 = subgroupShuffleXor(values[4], 2u);
            let tmp_482 = subgroupShuffleXor(keys[5], 2u);
            let tmp_483 = subgroupShuffleXor(values[5], 2u);
            let tmp_484 = subgroupShuffleXor(keys[6], 2u);
            let tmp_485 = subgroupShuffleXor(values[6], 2u);
            let tmp_486 = subgroupShuffleXor(keys[7], 2u);
            let tmp_487 = subgroupShuffleXor(values[7], 2u);
            let tmp_488 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_489 = keys[0] < tmp_472 || (keys[0] == tmp_472 && values[0] < tmp_473);
            if tmp_488 == tmp_489 { keys[0] = tmp_472; values[0] = tmp_473; }
            let tmp_490 = keys[1] < tmp_474 || (keys[1] == tmp_474 && values[1] < tmp_475);
            if tmp_488 == tmp_490 { keys[1] = tmp_474; values[1] = tmp_475; }
            let tmp_491 = keys[2] < tmp_476 || (keys[2] == tmp_476 && values[2] < tmp_477);
            if tmp_488 == tmp_491 { keys[2] = tmp_476; values[2] = tmp_477; }
            let tmp_492 = keys[3] < tmp_478 || (keys[3] == tmp_478 && values[3] < tmp_479);
            if tmp_488 == tmp_492 { keys[3] = tmp_478; values[3] = tmp_479; }
            let tmp_493 = keys[4] < tmp_480 || (keys[4] == tmp_480 && values[4] < tmp_481);
            if tmp_488 == tmp_493 { keys[4] = tmp_480; values[4] = tmp_481; }
            let tmp_494 = keys[5] < tmp_482 || (keys[5] == tmp_482 && values[5] < tmp_483);
            if tmp_488 == tmp_494 { keys[5] = tmp_482; values[5] = tmp_483; }
            let tmp_495 = keys[6] < tmp_484 || (keys[6] == tmp_484 && values[6] < tmp_485);
            if tmp_488 == tmp_495 { keys[6] = tmp_484; values[6] = tmp_485; }
            let tmp_496 = keys[7] < tmp_486 || (keys[7] == tmp_486 && values[7] < tmp_487);
            if tmp_488 == tmp_496 { keys[7] = tmp_486; values[7] = tmp_487; }
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],1,0)
        {
            let tmp_497 = subgroupShuffleXor(keys[0], 1u);
            let tmp_498 = subgroupShuffleXor(values[0], 1u);
            let tmp_499 = subgroupShuffleXor(keys[1], 1u);
            let tmp_500 = subgroupShuffleXor(values[1], 1u);
            let tmp_501 = subgroupShuffleXor(keys[2], 1u);
            let tmp_502 = subgroupShuffleXor(values[2], 1u);
            let tmp_503 = subgroupShuffleXor(keys[3], 1u);
            let tmp_504 = subgroupShuffleXor(values[3], 1u);
            let tmp_505 = subgroupShuffleXor(keys[4], 1u);
            let tmp_506 = subgroupShuffleXor(values[4], 1u);
            let tmp_507 = subgroupShuffleXor(keys[5], 1u);
            let tmp_508 = subgroupShuffleXor(values[5], 1u);
            let tmp_509 = subgroupShuffleXor(keys[6], 1u);
            let tmp_510 = subgroupShuffleXor(values[6], 1u);
            let tmp_511 = subgroupShuffleXor(keys[7], 1u);
            let tmp_512 = subgroupShuffleXor(values[7], 1u);
            let tmp_513 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_514 = keys[0] < tmp_497 || (keys[0] == tmp_497 && values[0] < tmp_498);
            if tmp_513 == tmp_514 { keys[0] = tmp_497; values[0] = tmp_498; }
            let tmp_515 = keys[1] < tmp_499 || (keys[1] == tmp_499 && values[1] < tmp_500);
            if tmp_513 == tmp_515 { keys[1] = tmp_499; values[1] = tmp_500; }
            let tmp_516 = keys[2] < tmp_501 || (keys[2] == tmp_501 && values[2] < tmp_502);
            if tmp_513 == tmp_516 { keys[2] = tmp_501; values[2] = tmp_502; }
            let tmp_517 = keys[3] < tmp_503 || (keys[3] == tmp_503 && values[3] < tmp_504);
            if tmp_513 == tmp_517 { keys[3] = tmp_503; values[3] = tmp_504; }
            let tmp_518 = keys[4] < tmp_505 || (keys[4] == tmp_505 && values[4] < tmp_506);
            if tmp_513 == tmp_518 { keys[4] = tmp_505; values[4] = tmp_506; }
            let tmp_519 = keys[5] < tmp_507 || (keys[5] == tmp_507 && values[5] < tmp_508);
            if tmp_513 == tmp_519 { keys[5] = tmp_507; values[5] = tmp_508; }
            let tmp_520 = keys[6] < tmp_509 || (keys[6] == tmp_509 && values[6] < tmp_510);
            if tmp_513 == tmp_520 { keys[6] = tmp_509; values[6] = tmp_510; }
            let tmp_521 = keys[7] < tmp_511 || (keys[7] == tmp_511 && values[7] < tmp_512);
            if tmp_513 == tmp_521 { keys[7] = tmp_511; values[7] = tmp_512; }
        }
    }

    // exch_local(4,8)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_522 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_522;
                let tmp_523 = values[0]; values[0] = values[4]; values[4] = tmp_523;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_524 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_524;
                let tmp_525 = values[1]; values[1] = values[5]; values[5] = tmp_525;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_526 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_526;
                let tmp_527 = values[2]; values[2] = values[6]; values[6] = tmp_527;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_528 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_528;
                let tmp_529 = values[3]; values[3] = values[7]; values[7] = tmp_529;
            }
        }
    }

    // exch_local(2,8)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_530 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_530;
                let tmp_531 = values[0]; values[0] = values[2]; values[2] = tmp_531;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_532 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_532;
                let tmp_533 = values[1]; values[1] = values[3]; values[3] = tmp_533;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_534 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_534;
                let tmp_535 = values[4]; values[4] = values[6]; values[6] = tmp_535;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_536 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_536;
                let tmp_537 = values[5]; values[5] = values[7]; values[7] = tmp_537;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_538 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_538;
                let tmp_539 = values[0]; values[0] = values[1]; values[1] = tmp_539;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_540 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_540;
                let tmp_541 = values[2]; values[2] = values[3]; values[3] = tmp_541;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_542 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_542;
                let tmp_543 = values[4]; values[4] = values[5]; values[5] = tmp_543;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_544 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_544;
                let tmp_545 = values[6]; values[6] = values[7]; values[7] = tmp_545;
            }
        }
    }

    // exch_intxn(tmask:63,swbit:5,wpt:8)
    {
        // _exch_workgroup([(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)],63,5)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            workgroupBarrier();
            let tmp_546 = extractBits(local_tid, 5u, 1u) != 0u;
            let tmp_547 = seg_base + (local_tid ^ 63u);
            let tmp_548 = smem_keys[tmp_547 * WPT + 7u];
            let tmp_549 = smem_vals[tmp_547 * WPT + 7u];
            let tmp_550 = keys[0] < tmp_548 || (keys[0] == tmp_548 && values[0] < tmp_549);
            if tmp_546 == tmp_550 { keys[0] = tmp_548; values[0] = tmp_549; }
            let tmp_551 = smem_keys[tmp_547 * WPT + 6u];
            let tmp_552 = smem_vals[tmp_547 * WPT + 6u];
            let tmp_553 = keys[1] < tmp_551 || (keys[1] == tmp_551 && values[1] < tmp_552);
            if tmp_546 == tmp_553 { keys[1] = tmp_551; values[1] = tmp_552; }
            let tmp_554 = smem_keys[tmp_547 * WPT + 5u];
            let tmp_555 = smem_vals[tmp_547 * WPT + 5u];
            let tmp_556 = keys[2] < tmp_554 || (keys[2] == tmp_554 && values[2] < tmp_555);
            if tmp_546 == tmp_556 { keys[2] = tmp_554; values[2] = tmp_555; }
            let tmp_557 = smem_keys[tmp_547 * WPT + 4u];
            let tmp_558 = smem_vals[tmp_547 * WPT + 4u];
            let tmp_559 = keys[3] < tmp_557 || (keys[3] == tmp_557 && values[3] < tmp_558);
            if tmp_546 == tmp_559 { keys[3] = tmp_557; values[3] = tmp_558; }
            let tmp_560 = smem_keys[tmp_547 * WPT + 3u];
            let tmp_561 = smem_vals[tmp_547 * WPT + 3u];
            let tmp_562 = keys[4] < tmp_560 || (keys[4] == tmp_560 && values[4] < tmp_561);
            if tmp_546 == tmp_562 { keys[4] = tmp_560; values[4] = tmp_561; }
            let tmp_563 = smem_keys[tmp_547 * WPT + 2u];
            let tmp_564 = smem_vals[tmp_547 * WPT + 2u];
            let tmp_565 = keys[5] < tmp_563 || (keys[5] == tmp_563 && values[5] < tmp_564);
            if tmp_546 == tmp_565 { keys[5] = tmp_563; values[5] = tmp_564; }
            let tmp_566 = smem_keys[tmp_547 * WPT + 1u];
            let tmp_567 = smem_vals[tmp_547 * WPT + 1u];
            let tmp_568 = keys[6] < tmp_566 || (keys[6] == tmp_566 && values[6] < tmp_567);
            if tmp_546 == tmp_568 { keys[6] = tmp_566; values[6] = tmp_567; }
            let tmp_569 = smem_keys[tmp_547 * WPT + 0u];
            let tmp_570 = smem_vals[tmp_547 * WPT + 0u];
            let tmp_571 = keys[7] < tmp_569 || (keys[7] == tmp_569 && values[7] < tmp_570);
            if tmp_546 == tmp_571 { keys[7] = tmp_569; values[7] = tmp_570; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:16,swbit:4,wpt:8)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],16,4)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            workgroupBarrier();
            let tmp_572 = extractBits(local_tid, 4u, 1u) != 0u;
            let tmp_573 = seg_base + (local_tid ^ 16u);
            let tmp_574 = smem_keys[tmp_573 * WPT + 0u];
            let tmp_575 = smem_vals[tmp_573 * WPT + 0u];
            let tmp_576 = keys[0] < tmp_574 || (keys[0] == tmp_574 && values[0] < tmp_575);
            if tmp_572 == tmp_576 { keys[0] = tmp_574; values[0] = tmp_575; }
            let tmp_577 = smem_keys[tmp_573 * WPT + 1u];
            let tmp_578 = smem_vals[tmp_573 * WPT + 1u];
            let tmp_579 = keys[1] < tmp_577 || (keys[1] == tmp_577 && values[1] < tmp_578);
            if tmp_572 == tmp_579 { keys[1] = tmp_577; values[1] = tmp_578; }
            let tmp_580 = smem_keys[tmp_573 * WPT + 2u];
            let tmp_581 = smem_vals[tmp_573 * WPT + 2u];
            let tmp_582 = keys[2] < tmp_580 || (keys[2] == tmp_580 && values[2] < tmp_581);
            if tmp_572 == tmp_582 { keys[2] = tmp_580; values[2] = tmp_581; }
            let tmp_583 = smem_keys[tmp_573 * WPT + 3u];
            let tmp_584 = smem_vals[tmp_573 * WPT + 3u];
            let tmp_585 = keys[3] < tmp_583 || (keys[3] == tmp_583 && values[3] < tmp_584);
            if tmp_572 == tmp_585 { keys[3] = tmp_583; values[3] = tmp_584; }
            let tmp_586 = smem_keys[tmp_573 * WPT + 4u];
            let tmp_587 = smem_vals[tmp_573 * WPT + 4u];
            let tmp_588 = keys[4] < tmp_586 || (keys[4] == tmp_586 && values[4] < tmp_587);
            if tmp_572 == tmp_588 { keys[4] = tmp_586; values[4] = tmp_587; }
            let tmp_589 = smem_keys[tmp_573 * WPT + 5u];
            let tmp_590 = smem_vals[tmp_573 * WPT + 5u];
            let tmp_591 = keys[5] < tmp_589 || (keys[5] == tmp_589 && values[5] < tmp_590);
            if tmp_572 == tmp_591 { keys[5] = tmp_589; values[5] = tmp_590; }
            let tmp_592 = smem_keys[tmp_573 * WPT + 6u];
            let tmp_593 = smem_vals[tmp_573 * WPT + 6u];
            let tmp_594 = keys[6] < tmp_592 || (keys[6] == tmp_592 && values[6] < tmp_593);
            if tmp_572 == tmp_594 { keys[6] = tmp_592; values[6] = tmp_593; }
            let tmp_595 = smem_keys[tmp_573 * WPT + 7u];
            let tmp_596 = smem_vals[tmp_573 * WPT + 7u];
            let tmp_597 = keys[7] < tmp_595 || (keys[7] == tmp_595 && values[7] < tmp_596);
            if tmp_572 == tmp_597 { keys[7] = tmp_595; values[7] = tmp_596; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:8,swbit:3,wpt:8)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],8,3)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            workgroupBarrier();
            let tmp_598 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_599 = seg_base + (local_tid ^ 8u);
            let tmp_600 = smem_keys[tmp_599 * WPT + 0u];
            let tmp_601 = smem_vals[tmp_599 * WPT + 0u];
            let tmp_602 = keys[0] < tmp_600 || (keys[0] == tmp_600 && values[0] < tmp_601);
            if tmp_598 == tmp_602 { keys[0] = tmp_600; values[0] = tmp_601; }
            let tmp_603 = smem_keys[tmp_599 * WPT + 1u];
            let tmp_604 = smem_vals[tmp_599 * WPT + 1u];
            let tmp_605 = keys[1] < tmp_603 || (keys[1] == tmp_603 && values[1] < tmp_604);
            if tmp_598 == tmp_605 { keys[1] = tmp_603; values[1] = tmp_604; }
            let tmp_606 = smem_keys[tmp_599 * WPT + 2u];
            let tmp_607 = smem_vals[tmp_599 * WPT + 2u];
            let tmp_608 = keys[2] < tmp_606 || (keys[2] == tmp_606 && values[2] < tmp_607);
            if tmp_598 == tmp_608 { keys[2] = tmp_606; values[2] = tmp_607; }
            let tmp_609 = smem_keys[tmp_599 * WPT + 3u];
            let tmp_610 = smem_vals[tmp_599 * WPT + 3u];
            let tmp_611 = keys[3] < tmp_609 || (keys[3] == tmp_609 && values[3] < tmp_610);
            if tmp_598 == tmp_611 { keys[3] = tmp_609; values[3] = tmp_610; }
            let tmp_612 = smem_keys[tmp_599 * WPT + 4u];
            let tmp_613 = smem_vals[tmp_599 * WPT + 4u];
            let tmp_614 = keys[4] < tmp_612 || (keys[4] == tmp_612 && values[4] < tmp_613);
            if tmp_598 == tmp_614 { keys[4] = tmp_612; values[4] = tmp_613; }
            let tmp_615 = smem_keys[tmp_599 * WPT + 5u];
            let tmp_616 = smem_vals[tmp_599 * WPT + 5u];
            let tmp_617 = keys[5] < tmp_615 || (keys[5] == tmp_615 && values[5] < tmp_616);
            if tmp_598 == tmp_617 { keys[5] = tmp_615; values[5] = tmp_616; }
            let tmp_618 = smem_keys[tmp_599 * WPT + 6u];
            let tmp_619 = smem_vals[tmp_599 * WPT + 6u];
            let tmp_620 = keys[6] < tmp_618 || (keys[6] == tmp_618 && values[6] < tmp_619);
            if tmp_598 == tmp_620 { keys[6] = tmp_618; values[6] = tmp_619; }
            let tmp_621 = smem_keys[tmp_599 * WPT + 7u];
            let tmp_622 = smem_vals[tmp_599 * WPT + 7u];
            let tmp_623 = keys[7] < tmp_621 || (keys[7] == tmp_621 && values[7] < tmp_622);
            if tmp_598 == tmp_623 { keys[7] = tmp_621; values[7] = tmp_622; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],4,2)
        {
            let tmp_624 = subgroupShuffleXor(keys[0], 4u);
            let tmp_625 = subgroupShuffleXor(values[0], 4u);
            let tmp_626 = subgroupShuffleXor(keys[1], 4u);
            let tmp_627 = subgroupShuffleXor(values[1], 4u);
            let tmp_628 = subgroupShuffleXor(keys[2], 4u);
            let tmp_629 = subgroupShuffleXor(values[2], 4u);
            let tmp_630 = subgroupShuffleXor(keys[3], 4u);
            let tmp_631 = subgroupShuffleXor(values[3], 4u);
            let tmp_632 = subgroupShuffleXor(keys[4], 4u);
            let tmp_633 = subgroupShuffleXor(values[4], 4u);
            let tmp_634 = subgroupShuffleXor(keys[5], 4u);
            let tmp_635 = subgroupShuffleXor(values[5], 4u);
            let tmp_636 = subgroupShuffleXor(keys[6], 4u);
            let tmp_637 = subgroupShuffleXor(values[6], 4u);
            let tmp_638 = subgroupShuffleXor(keys[7], 4u);
            let tmp_639 = subgroupShuffleXor(values[7], 4u);
            let tmp_640 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_641 = keys[0] < tmp_624 || (keys[0] == tmp_624 && values[0] < tmp_625);
            if tmp_640 == tmp_641 { keys[0] = tmp_624; values[0] = tmp_625; }
            let tmp_642 = keys[1] < tmp_626 || (keys[1] == tmp_626 && values[1] < tmp_627);
            if tmp_640 == tmp_642 { keys[1] = tmp_626; values[1] = tmp_627; }
            let tmp_643 = keys[2] < tmp_628 || (keys[2] == tmp_628 && values[2] < tmp_629);
            if tmp_640 == tmp_643 { keys[2] = tmp_628; values[2] = tmp_629; }
            let tmp_644 = keys[3] < tmp_630 || (keys[3] == tmp_630 && values[3] < tmp_631);
            if tmp_640 == tmp_644 { keys[3] = tmp_630; values[3] = tmp_631; }
            let tmp_645 = keys[4] < tmp_632 || (keys[4] == tmp_632 && values[4] < tmp_633);
            if tmp_640 == tmp_645 { keys[4] = tmp_632; values[4] = tmp_633; }
            let tmp_646 = keys[5] < tmp_634 || (keys[5] == tmp_634 && values[5] < tmp_635);
            if tmp_640 == tmp_646 { keys[5] = tmp_634; values[5] = tmp_635; }
            let tmp_647 = keys[6] < tmp_636 || (keys[6] == tmp_636 && values[6] < tmp_637);
            if tmp_640 == tmp_647 { keys[6] = tmp_636; values[6] = tmp_637; }
            let tmp_648 = keys[7] < tmp_638 || (keys[7] == tmp_638 && values[7] < tmp_639);
            if tmp_640 == tmp_648 { keys[7] = tmp_638; values[7] = tmp_639; }
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],2,1)
        {
            let tmp_649 = subgroupShuffleXor(keys[0], 2u);
            let tmp_650 = subgroupShuffleXor(values[0], 2u);
            let tmp_651 = subgroupShuffleXor(keys[1], 2u);
            let tmp_652 = subgroupShuffleXor(values[1], 2u);
            let tmp_653 = subgroupShuffleXor(keys[2], 2u);
            let tmp_654 = subgroupShuffleXor(values[2], 2u);
            let tmp_655 = subgroupShuffleXor(keys[3], 2u);
            let tmp_656 = subgroupShuffleXor(values[3], 2u);
            let tmp_657 = subgroupShuffleXor(keys[4], 2u);
            let tmp_658 = subgroupShuffleXor(values[4], 2u);
            let tmp_659 = subgroupShuffleXor(keys[5], 2u);
            let tmp_660 = subgroupShuffleXor(values[5], 2u);
            let tmp_661 = subgroupShuffleXor(keys[6], 2u);
            let tmp_662 = subgroupShuffleXor(values[6], 2u);
            let tmp_663 = subgroupShuffleXor(keys[7], 2u);
            let tmp_664 = subgroupShuffleXor(values[7], 2u);
            let tmp_665 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_666 = keys[0] < tmp_649 || (keys[0] == tmp_649 && values[0] < tmp_650);
            if tmp_665 == tmp_666 { keys[0] = tmp_649; values[0] = tmp_650; }
            let tmp_667 = keys[1] < tmp_651 || (keys[1] == tmp_651 && values[1] < tmp_652);
            if tmp_665 == tmp_667 { keys[1] = tmp_651; values[1] = tmp_652; }
            let tmp_668 = keys[2] < tmp_653 || (keys[2] == tmp_653 && values[2] < tmp_654);
            if tmp_665 == tmp_668 { keys[2] = tmp_653; values[2] = tmp_654; }
            let tmp_669 = keys[3] < tmp_655 || (keys[3] == tmp_655 && values[3] < tmp_656);
            if tmp_665 == tmp_669 { keys[3] = tmp_655; values[3] = tmp_656; }
            let tmp_670 = keys[4] < tmp_657 || (keys[4] == tmp_657 && values[4] < tmp_658);
            if tmp_665 == tmp_670 { keys[4] = tmp_657; values[4] = tmp_658; }
            let tmp_671 = keys[5] < tmp_659 || (keys[5] == tmp_659 && values[5] < tmp_660);
            if tmp_665 == tmp_671 { keys[5] = tmp_659; values[5] = tmp_660; }
            let tmp_672 = keys[6] < tmp_661 || (keys[6] == tmp_661 && values[6] < tmp_662);
            if tmp_665 == tmp_672 { keys[6] = tmp_661; values[6] = tmp_662; }
            let tmp_673 = keys[7] < tmp_663 || (keys[7] == tmp_663 && values[7] < tmp_664);
            if tmp_665 == tmp_673 { keys[7] = tmp_663; values[7] = tmp_664; }
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],1,0)
        {
            let tmp_674 = subgroupShuffleXor(keys[0], 1u);
            let tmp_675 = subgroupShuffleXor(values[0], 1u);
            let tmp_676 = subgroupShuffleXor(keys[1], 1u);
            let tmp_677 = subgroupShuffleXor(values[1], 1u);
            let tmp_678 = subgroupShuffleXor(keys[2], 1u);
            let tmp_679 = subgroupShuffleXor(values[2], 1u);
            let tmp_680 = subgroupShuffleXor(keys[3], 1u);
            let tmp_681 = subgroupShuffleXor(values[3], 1u);
            let tmp_682 = subgroupShuffleXor(keys[4], 1u);
            let tmp_683 = subgroupShuffleXor(values[4], 1u);
            let tmp_684 = subgroupShuffleXor(keys[5], 1u);
            let tmp_685 = subgroupShuffleXor(values[5], 1u);
            let tmp_686 = subgroupShuffleXor(keys[6], 1u);
            let tmp_687 = subgroupShuffleXor(values[6], 1u);
            let tmp_688 = subgroupShuffleXor(keys[7], 1u);
            let tmp_689 = subgroupShuffleXor(values[7], 1u);
            let tmp_690 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_691 = keys[0] < tmp_674 || (keys[0] == tmp_674 && values[0] < tmp_675);
            if tmp_690 == tmp_691 { keys[0] = tmp_674; values[0] = tmp_675; }
            let tmp_692 = keys[1] < tmp_676 || (keys[1] == tmp_676 && values[1] < tmp_677);
            if tmp_690 == tmp_692 { keys[1] = tmp_676; values[1] = tmp_677; }
            let tmp_693 = keys[2] < tmp_678 || (keys[2] == tmp_678 && values[2] < tmp_679);
            if tmp_690 == tmp_693 { keys[2] = tmp_678; values[2] = tmp_679; }
            let tmp_694 = keys[3] < tmp_680 || (keys[3] == tmp_680 && values[3] < tmp_681);
            if tmp_690 == tmp_694 { keys[3] = tmp_680; values[3] = tmp_681; }
            let tmp_695 = keys[4] < tmp_682 || (keys[4] == tmp_682 && values[4] < tmp_683);
            if tmp_690 == tmp_695 { keys[4] = tmp_682; values[4] = tmp_683; }
            let tmp_696 = keys[5] < tmp_684 || (keys[5] == tmp_684 && values[5] < tmp_685);
            if tmp_690 == tmp_696 { keys[5] = tmp_684; values[5] = tmp_685; }
            let tmp_697 = keys[6] < tmp_686 || (keys[6] == tmp_686 && values[6] < tmp_687);
            if tmp_690 == tmp_697 { keys[6] = tmp_686; values[6] = tmp_687; }
            let tmp_698 = keys[7] < tmp_688 || (keys[7] == tmp_688 && values[7] < tmp_689);
            if tmp_690 == tmp_698 { keys[7] = tmp_688; values[7] = tmp_689; }
        }
    }

    // exch_local(4,8)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_699 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_699;
                let tmp_700 = values[0]; values[0] = values[4]; values[4] = tmp_700;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_701 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_701;
                let tmp_702 = values[1]; values[1] = values[5]; values[5] = tmp_702;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_703 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_703;
                let tmp_704 = values[2]; values[2] = values[6]; values[6] = tmp_704;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_705 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_705;
                let tmp_706 = values[3]; values[3] = values[7]; values[7] = tmp_706;
            }
        }
    }

    // exch_local(2,8)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_707 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_707;
                let tmp_708 = values[0]; values[0] = values[2]; values[2] = tmp_708;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_709 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_709;
                let tmp_710 = values[1]; values[1] = values[3]; values[3] = tmp_710;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_711 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_711;
                let tmp_712 = values[4]; values[4] = values[6]; values[6] = tmp_712;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_713 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_713;
                let tmp_714 = values[5]; values[5] = values[7]; values[7] = tmp_714;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_715 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_715;
                let tmp_716 = values[0]; values[0] = values[1]; values[1] = tmp_716;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_717 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_717;
                let tmp_718 = values[2]; values[2] = values[3]; values[3] = tmp_718;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_719 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_719;
                let tmp_720 = values[4]; values[4] = values[5]; values[5] = tmp_720;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_721 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_721;
                let tmp_722 = values[6]; values[6] = values[7]; values[7] = tmp_722;
            }
        }
    }

    // exch_intxn(tmask:127,swbit:6,wpt:8)
    {
        // _exch_workgroup([(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)],127,6)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            workgroupBarrier();
            let tmp_723 = extractBits(local_tid, 6u, 1u) != 0u;
            let tmp_724 = seg_base + (local_tid ^ 127u);
            let tmp_725 = smem_keys[tmp_724 * WPT + 7u];
            let tmp_726 = smem_vals[tmp_724 * WPT + 7u];
            let tmp_727 = keys[0] < tmp_725 || (keys[0] == tmp_725 && values[0] < tmp_726);
            if tmp_723 == tmp_727 { keys[0] = tmp_725; values[0] = tmp_726; }
            let tmp_728 = smem_keys[tmp_724 * WPT + 6u];
            let tmp_729 = smem_vals[tmp_724 * WPT + 6u];
            let tmp_730 = keys[1] < tmp_728 || (keys[1] == tmp_728 && values[1] < tmp_729);
            if tmp_723 == tmp_730 { keys[1] = tmp_728; values[1] = tmp_729; }
            let tmp_731 = smem_keys[tmp_724 * WPT + 5u];
            let tmp_732 = smem_vals[tmp_724 * WPT + 5u];
            let tmp_733 = keys[2] < tmp_731 || (keys[2] == tmp_731 && values[2] < tmp_732);
            if tmp_723 == tmp_733 { keys[2] = tmp_731; values[2] = tmp_732; }
            let tmp_734 = smem_keys[tmp_724 * WPT + 4u];
            let tmp_735 = smem_vals[tmp_724 * WPT + 4u];
            let tmp_736 = keys[3] < tmp_734 || (keys[3] == tmp_734 && values[3] < tmp_735);
            if tmp_723 == tmp_736 { keys[3] = tmp_734; values[3] = tmp_735; }
            let tmp_737 = smem_keys[tmp_724 * WPT + 3u];
            let tmp_738 = smem_vals[tmp_724 * WPT + 3u];
            let tmp_739 = keys[4] < tmp_737 || (keys[4] == tmp_737 && values[4] < tmp_738);
            if tmp_723 == tmp_739 { keys[4] = tmp_737; values[4] = tmp_738; }
            let tmp_740 = smem_keys[tmp_724 * WPT + 2u];
            let tmp_741 = smem_vals[tmp_724 * WPT + 2u];
            let tmp_742 = keys[5] < tmp_740 || (keys[5] == tmp_740 && values[5] < tmp_741);
            if tmp_723 == tmp_742 { keys[5] = tmp_740; values[5] = tmp_741; }
            let tmp_743 = smem_keys[tmp_724 * WPT + 1u];
            let tmp_744 = smem_vals[tmp_724 * WPT + 1u];
            let tmp_745 = keys[6] < tmp_743 || (keys[6] == tmp_743 && values[6] < tmp_744);
            if tmp_723 == tmp_745 { keys[6] = tmp_743; values[6] = tmp_744; }
            let tmp_746 = smem_keys[tmp_724 * WPT + 0u];
            let tmp_747 = smem_vals[tmp_724 * WPT + 0u];
            let tmp_748 = keys[7] < tmp_746 || (keys[7] == tmp_746 && values[7] < tmp_747);
            if tmp_723 == tmp_748 { keys[7] = tmp_746; values[7] = tmp_747; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:32,swbit:5,wpt:8)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],32,5)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            workgroupBarrier();
            let tmp_749 = extractBits(local_tid, 5u, 1u) != 0u;
            let tmp_750 = seg_base + (local_tid ^ 32u);
            let tmp_751 = smem_keys[tmp_750 * WPT + 0u];
            let tmp_752 = smem_vals[tmp_750 * WPT + 0u];
            let tmp_753 = keys[0] < tmp_751 || (keys[0] == tmp_751 && values[0] < tmp_752);
            if tmp_749 == tmp_753 { keys[0] = tmp_751; values[0] = tmp_752; }
            let tmp_754 = smem_keys[tmp_750 * WPT + 1u];
            let tmp_755 = smem_vals[tmp_750 * WPT + 1u];
            let tmp_756 = keys[1] < tmp_754 || (keys[1] == tmp_754 && values[1] < tmp_755);
            if tmp_749 == tmp_756 { keys[1] = tmp_754; values[1] = tmp_755; }
            let tmp_757 = smem_keys[tmp_750 * WPT + 2u];
            let tmp_758 = smem_vals[tmp_750 * WPT + 2u];
            let tmp_759 = keys[2] < tmp_757 || (keys[2] == tmp_757 && values[2] < tmp_758);
            if tmp_749 == tmp_759 { keys[2] = tmp_757; values[2] = tmp_758; }
            let tmp_760 = smem_keys[tmp_750 * WPT + 3u];
            let tmp_761 = smem_vals[tmp_750 * WPT + 3u];
            let tmp_762 = keys[3] < tmp_760 || (keys[3] == tmp_760 && values[3] < tmp_761);
            if tmp_749 == tmp_762 { keys[3] = tmp_760; values[3] = tmp_761; }
            let tmp_763 = smem_keys[tmp_750 * WPT + 4u];
            let tmp_764 = smem_vals[tmp_750 * WPT + 4u];
            let tmp_765 = keys[4] < tmp_763 || (keys[4] == tmp_763 && values[4] < tmp_764);
            if tmp_749 == tmp_765 { keys[4] = tmp_763; values[4] = tmp_764; }
            let tmp_766 = smem_keys[tmp_750 * WPT + 5u];
            let tmp_767 = smem_vals[tmp_750 * WPT + 5u];
            let tmp_768 = keys[5] < tmp_766 || (keys[5] == tmp_766 && values[5] < tmp_767);
            if tmp_749 == tmp_768 { keys[5] = tmp_766; values[5] = tmp_767; }
            let tmp_769 = smem_keys[tmp_750 * WPT + 6u];
            let tmp_770 = smem_vals[tmp_750 * WPT + 6u];
            let tmp_771 = keys[6] < tmp_769 || (keys[6] == tmp_769 && values[6] < tmp_770);
            if tmp_749 == tmp_771 { keys[6] = tmp_769; values[6] = tmp_770; }
            let tmp_772 = smem_keys[tmp_750 * WPT + 7u];
            let tmp_773 = smem_vals[tmp_750 * WPT + 7u];
            let tmp_774 = keys[7] < tmp_772 || (keys[7] == tmp_772 && values[7] < tmp_773);
            if tmp_749 == tmp_774 { keys[7] = tmp_772; values[7] = tmp_773; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:16,swbit:4,wpt:8)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],16,4)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            workgroupBarrier();
            let tmp_775 = extractBits(local_tid, 4u, 1u) != 0u;
            let tmp_776 = seg_base + (local_tid ^ 16u);
            let tmp_777 = smem_keys[tmp_776 * WPT + 0u];
            let tmp_778 = smem_vals[tmp_776 * WPT + 0u];
            let tmp_779 = keys[0] < tmp_777 || (keys[0] == tmp_777 && values[0] < tmp_778);
            if tmp_775 == tmp_779 { keys[0] = tmp_777; values[0] = tmp_778; }
            let tmp_780 = smem_keys[tmp_776 * WPT + 1u];
            let tmp_781 = smem_vals[tmp_776 * WPT + 1u];
            let tmp_782 = keys[1] < tmp_780 || (keys[1] == tmp_780 && values[1] < tmp_781);
            if tmp_775 == tmp_782 { keys[1] = tmp_780; values[1] = tmp_781; }
            let tmp_783 = smem_keys[tmp_776 * WPT + 2u];
            let tmp_784 = smem_vals[tmp_776 * WPT + 2u];
            let tmp_785 = keys[2] < tmp_783 || (keys[2] == tmp_783 && values[2] < tmp_784);
            if tmp_775 == tmp_785 { keys[2] = tmp_783; values[2] = tmp_784; }
            let tmp_786 = smem_keys[tmp_776 * WPT + 3u];
            let tmp_787 = smem_vals[tmp_776 * WPT + 3u];
            let tmp_788 = keys[3] < tmp_786 || (keys[3] == tmp_786 && values[3] < tmp_787);
            if tmp_775 == tmp_788 { keys[3] = tmp_786; values[3] = tmp_787; }
            let tmp_789 = smem_keys[tmp_776 * WPT + 4u];
            let tmp_790 = smem_vals[tmp_776 * WPT + 4u];
            let tmp_791 = keys[4] < tmp_789 || (keys[4] == tmp_789 && values[4] < tmp_790);
            if tmp_775 == tmp_791 { keys[4] = tmp_789; values[4] = tmp_790; }
            let tmp_792 = smem_keys[tmp_776 * WPT + 5u];
            let tmp_793 = smem_vals[tmp_776 * WPT + 5u];
            let tmp_794 = keys[5] < tmp_792 || (keys[5] == tmp_792 && values[5] < tmp_793);
            if tmp_775 == tmp_794 { keys[5] = tmp_792; values[5] = tmp_793; }
            let tmp_795 = smem_keys[tmp_776 * WPT + 6u];
            let tmp_796 = smem_vals[tmp_776 * WPT + 6u];
            let tmp_797 = keys[6] < tmp_795 || (keys[6] == tmp_795 && values[6] < tmp_796);
            if tmp_775 == tmp_797 { keys[6] = tmp_795; values[6] = tmp_796; }
            let tmp_798 = smem_keys[tmp_776 * WPT + 7u];
            let tmp_799 = smem_vals[tmp_776 * WPT + 7u];
            let tmp_800 = keys[7] < tmp_798 || (keys[7] == tmp_798 && values[7] < tmp_799);
            if tmp_775 == tmp_800 { keys[7] = tmp_798; values[7] = tmp_799; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:8,swbit:3,wpt:8)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],8,3)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            workgroupBarrier();
            let tmp_801 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_802 = seg_base + (local_tid ^ 8u);
            let tmp_803 = smem_keys[tmp_802 * WPT + 0u];
            let tmp_804 = smem_vals[tmp_802 * WPT + 0u];
            let tmp_805 = keys[0] < tmp_803 || (keys[0] == tmp_803 && values[0] < tmp_804);
            if tmp_801 == tmp_805 { keys[0] = tmp_803; values[0] = tmp_804; }
            let tmp_806 = smem_keys[tmp_802 * WPT + 1u];
            let tmp_807 = smem_vals[tmp_802 * WPT + 1u];
            let tmp_808 = keys[1] < tmp_806 || (keys[1] == tmp_806 && values[1] < tmp_807);
            if tmp_801 == tmp_808 { keys[1] = tmp_806; values[1] = tmp_807; }
            let tmp_809 = smem_keys[tmp_802 * WPT + 2u];
            let tmp_810 = smem_vals[tmp_802 * WPT + 2u];
            let tmp_811 = keys[2] < tmp_809 || (keys[2] == tmp_809 && values[2] < tmp_810);
            if tmp_801 == tmp_811 { keys[2] = tmp_809; values[2] = tmp_810; }
            let tmp_812 = smem_keys[tmp_802 * WPT + 3u];
            let tmp_813 = smem_vals[tmp_802 * WPT + 3u];
            let tmp_814 = keys[3] < tmp_812 || (keys[3] == tmp_812 && values[3] < tmp_813);
            if tmp_801 == tmp_814 { keys[3] = tmp_812; values[3] = tmp_813; }
            let tmp_815 = smem_keys[tmp_802 * WPT + 4u];
            let tmp_816 = smem_vals[tmp_802 * WPT + 4u];
            let tmp_817 = keys[4] < tmp_815 || (keys[4] == tmp_815 && values[4] < tmp_816);
            if tmp_801 == tmp_817 { keys[4] = tmp_815; values[4] = tmp_816; }
            let tmp_818 = smem_keys[tmp_802 * WPT + 5u];
            let tmp_819 = smem_vals[tmp_802 * WPT + 5u];
            let tmp_820 = keys[5] < tmp_818 || (keys[5] == tmp_818 && values[5] < tmp_819);
            if tmp_801 == tmp_820 { keys[5] = tmp_818; values[5] = tmp_819; }
            let tmp_821 = smem_keys[tmp_802 * WPT + 6u];
            let tmp_822 = smem_vals[tmp_802 * WPT + 6u];
            let tmp_823 = keys[6] < tmp_821 || (keys[6] == tmp_821 && values[6] < tmp_822);
            if tmp_801 == tmp_823 { keys[6] = tmp_821; values[6] = tmp_822; }
            let tmp_824 = smem_keys[tmp_802 * WPT + 7u];
            let tmp_825 = smem_vals[tmp_802 * WPT + 7u];
            let tmp_826 = keys[7] < tmp_824 || (keys[7] == tmp_824 && values[7] < tmp_825);
            if tmp_801 == tmp_826 { keys[7] = tmp_824; values[7] = tmp_825; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],4,2)
        {
            let tmp_827 = subgroupShuffleXor(keys[0], 4u);
            let tmp_828 = subgroupShuffleXor(values[0], 4u);
            let tmp_829 = subgroupShuffleXor(keys[1], 4u);
            let tmp_830 = subgroupShuffleXor(values[1], 4u);
            let tmp_831 = subgroupShuffleXor(keys[2], 4u);
            let tmp_832 = subgroupShuffleXor(values[2], 4u);
            let tmp_833 = subgroupShuffleXor(keys[3], 4u);
            let tmp_834 = subgroupShuffleXor(values[3], 4u);
            let tmp_835 = subgroupShuffleXor(keys[4], 4u);
            let tmp_836 = subgroupShuffleXor(values[4], 4u);
            let tmp_837 = subgroupShuffleXor(keys[5], 4u);
            let tmp_838 = subgroupShuffleXor(values[5], 4u);
            let tmp_839 = subgroupShuffleXor(keys[6], 4u);
            let tmp_840 = subgroupShuffleXor(values[6], 4u);
            let tmp_841 = subgroupShuffleXor(keys[7], 4u);
            let tmp_842 = subgroupShuffleXor(values[7], 4u);
            let tmp_843 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_844 = keys[0] < tmp_827 || (keys[0] == tmp_827 && values[0] < tmp_828);
            if tmp_843 == tmp_844 { keys[0] = tmp_827; values[0] = tmp_828; }
            let tmp_845 = keys[1] < tmp_829 || (keys[1] == tmp_829 && values[1] < tmp_830);
            if tmp_843 == tmp_845 { keys[1] = tmp_829; values[1] = tmp_830; }
            let tmp_846 = keys[2] < tmp_831 || (keys[2] == tmp_831 && values[2] < tmp_832);
            if tmp_843 == tmp_846 { keys[2] = tmp_831; values[2] = tmp_832; }
            let tmp_847 = keys[3] < tmp_833 || (keys[3] == tmp_833 && values[3] < tmp_834);
            if tmp_843 == tmp_847 { keys[3] = tmp_833; values[3] = tmp_834; }
            let tmp_848 = keys[4] < tmp_835 || (keys[4] == tmp_835 && values[4] < tmp_836);
            if tmp_843 == tmp_848 { keys[4] = tmp_835; values[4] = tmp_836; }
            let tmp_849 = keys[5] < tmp_837 || (keys[5] == tmp_837 && values[5] < tmp_838);
            if tmp_843 == tmp_849 { keys[5] = tmp_837; values[5] = tmp_838; }
            let tmp_850 = keys[6] < tmp_839 || (keys[6] == tmp_839 && values[6] < tmp_840);
            if tmp_843 == tmp_850 { keys[6] = tmp_839; values[6] = tmp_840; }
            let tmp_851 = keys[7] < tmp_841 || (keys[7] == tmp_841 && values[7] < tmp_842);
            if tmp_843 == tmp_851 { keys[7] = tmp_841; values[7] = tmp_842; }
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],2,1)
        {
            let tmp_852 = subgroupShuffleXor(keys[0], 2u);
            let tmp_853 = subgroupShuffleXor(values[0], 2u);
            let tmp_854 = subgroupShuffleXor(keys[1], 2u);
            let tmp_855 = subgroupShuffleXor(values[1], 2u);
            let tmp_856 = subgroupShuffleXor(keys[2], 2u);
            let tmp_857 = subgroupShuffleXor(values[2], 2u);
            let tmp_858 = subgroupShuffleXor(keys[3], 2u);
            let tmp_859 = subgroupShuffleXor(values[3], 2u);
            let tmp_860 = subgroupShuffleXor(keys[4], 2u);
            let tmp_861 = subgroupShuffleXor(values[4], 2u);
            let tmp_862 = subgroupShuffleXor(keys[5], 2u);
            let tmp_863 = subgroupShuffleXor(values[5], 2u);
            let tmp_864 = subgroupShuffleXor(keys[6], 2u);
            let tmp_865 = subgroupShuffleXor(values[6], 2u);
            let tmp_866 = subgroupShuffleXor(keys[7], 2u);
            let tmp_867 = subgroupShuffleXor(values[7], 2u);
            let tmp_868 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_869 = keys[0] < tmp_852 || (keys[0] == tmp_852 && values[0] < tmp_853);
            if tmp_868 == tmp_869 { keys[0] = tmp_852; values[0] = tmp_853; }
            let tmp_870 = keys[1] < tmp_854 || (keys[1] == tmp_854 && values[1] < tmp_855);
            if tmp_868 == tmp_870 { keys[1] = tmp_854; values[1] = tmp_855; }
            let tmp_871 = keys[2] < tmp_856 || (keys[2] == tmp_856 && values[2] < tmp_857);
            if tmp_868 == tmp_871 { keys[2] = tmp_856; values[2] = tmp_857; }
            let tmp_872 = keys[3] < tmp_858 || (keys[3] == tmp_858 && values[3] < tmp_859);
            if tmp_868 == tmp_872 { keys[3] = tmp_858; values[3] = tmp_859; }
            let tmp_873 = keys[4] < tmp_860 || (keys[4] == tmp_860 && values[4] < tmp_861);
            if tmp_868 == tmp_873 { keys[4] = tmp_860; values[4] = tmp_861; }
            let tmp_874 = keys[5] < tmp_862 || (keys[5] == tmp_862 && values[5] < tmp_863);
            if tmp_868 == tmp_874 { keys[5] = tmp_862; values[5] = tmp_863; }
            let tmp_875 = keys[6] < tmp_864 || (keys[6] == tmp_864 && values[6] < tmp_865);
            if tmp_868 == tmp_875 { keys[6] = tmp_864; values[6] = tmp_865; }
            let tmp_876 = keys[7] < tmp_866 || (keys[7] == tmp_866 && values[7] < tmp_867);
            if tmp_868 == tmp_876 { keys[7] = tmp_866; values[7] = tmp_867; }
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],1,0)
        {
            let tmp_877 = subgroupShuffleXor(keys[0], 1u);
            let tmp_878 = subgroupShuffleXor(values[0], 1u);
            let tmp_879 = subgroupShuffleXor(keys[1], 1u);
            let tmp_880 = subgroupShuffleXor(values[1], 1u);
            let tmp_881 = subgroupShuffleXor(keys[2], 1u);
            let tmp_882 = subgroupShuffleXor(values[2], 1u);
            let tmp_883 = subgroupShuffleXor(keys[3], 1u);
            let tmp_884 = subgroupShuffleXor(values[3], 1u);
            let tmp_885 = subgroupShuffleXor(keys[4], 1u);
            let tmp_886 = subgroupShuffleXor(values[4], 1u);
            let tmp_887 = subgroupShuffleXor(keys[5], 1u);
            let tmp_888 = subgroupShuffleXor(values[5], 1u);
            let tmp_889 = subgroupShuffleXor(keys[6], 1u);
            let tmp_890 = subgroupShuffleXor(values[6], 1u);
            let tmp_891 = subgroupShuffleXor(keys[7], 1u);
            let tmp_892 = subgroupShuffleXor(values[7], 1u);
            let tmp_893 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_894 = keys[0] < tmp_877 || (keys[0] == tmp_877 && values[0] < tmp_878);
            if tmp_893 == tmp_894 { keys[0] = tmp_877; values[0] = tmp_878; }
            let tmp_895 = keys[1] < tmp_879 || (keys[1] == tmp_879 && values[1] < tmp_880);
            if tmp_893 == tmp_895 { keys[1] = tmp_879; values[1] = tmp_880; }
            let tmp_896 = keys[2] < tmp_881 || (keys[2] == tmp_881 && values[2] < tmp_882);
            if tmp_893 == tmp_896 { keys[2] = tmp_881; values[2] = tmp_882; }
            let tmp_897 = keys[3] < tmp_883 || (keys[3] == tmp_883 && values[3] < tmp_884);
            if tmp_893 == tmp_897 { keys[3] = tmp_883; values[3] = tmp_884; }
            let tmp_898 = keys[4] < tmp_885 || (keys[4] == tmp_885 && values[4] < tmp_886);
            if tmp_893 == tmp_898 { keys[4] = tmp_885; values[4] = tmp_886; }
            let tmp_899 = keys[5] < tmp_887 || (keys[5] == tmp_887 && values[5] < tmp_888);
            if tmp_893 == tmp_899 { keys[5] = tmp_887; values[5] = tmp_888; }
            let tmp_900 = keys[6] < tmp_889 || (keys[6] == tmp_889 && values[6] < tmp_890);
            if tmp_893 == tmp_900 { keys[6] = tmp_889; values[6] = tmp_890; }
            let tmp_901 = keys[7] < tmp_891 || (keys[7] == tmp_891 && values[7] < tmp_892);
            if tmp_893 == tmp_901 { keys[7] = tmp_891; values[7] = tmp_892; }
        }
    }

    // exch_local(4,8)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_902 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_902;
                let tmp_903 = values[0]; values[0] = values[4]; values[4] = tmp_903;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_904 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_904;
                let tmp_905 = values[1]; values[1] = values[5]; values[5] = tmp_905;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_906 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_906;
                let tmp_907 = values[2]; values[2] = values[6]; values[6] = tmp_907;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_908 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_908;
                let tmp_909 = values[3]; values[3] = values[7]; values[7] = tmp_909;
            }
        }
    }

    // exch_local(2,8)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_910 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_910;
                let tmp_911 = values[0]; values[0] = values[2]; values[2] = tmp_911;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_912 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_912;
                let tmp_913 = values[1]; values[1] = values[3]; values[3] = tmp_913;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_914 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_914;
                let tmp_915 = values[4]; values[4] = values[6]; values[6] = tmp_915;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_916 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_916;
                let tmp_917 = values[5]; values[5] = values[7]; values[7] = tmp_917;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_918 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_918;
                let tmp_919 = values[0]; values[0] = values[1]; values[1] = tmp_919;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_920 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_920;
                let tmp_921 = values[2]; values[2] = values[3]; values[3] = tmp_921;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_922 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_922;
                let tmp_923 = values[4]; values[4] = values[5]; values[5] = tmp_923;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_924 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_924;
                let tmp_925 = values[6]; values[6] = values[7]; values[7] = tmp_925;
            }
        }
    }

    // exch_intxn(tmask:255,swbit:7,wpt:8)
    {
        // _exch_workgroup([(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)],255,7)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            workgroupBarrier();
            let tmp_926 = extractBits(local_tid, 7u, 1u) != 0u;
            let tmp_927 = seg_base + (local_tid ^ 255u);
            let tmp_928 = smem_keys[tmp_927 * WPT + 7u];
            let tmp_929 = smem_vals[tmp_927 * WPT + 7u];
            let tmp_930 = keys[0] < tmp_928 || (keys[0] == tmp_928 && values[0] < tmp_929);
            if tmp_926 == tmp_930 { keys[0] = tmp_928; values[0] = tmp_929; }
            let tmp_931 = smem_keys[tmp_927 * WPT + 6u];
            let tmp_932 = smem_vals[tmp_927 * WPT + 6u];
            let tmp_933 = keys[1] < tmp_931 || (keys[1] == tmp_931 && values[1] < tmp_932);
            if tmp_926 == tmp_933 { keys[1] = tmp_931; values[1] = tmp_932; }
            let tmp_934 = smem_keys[tmp_927 * WPT + 5u];
            let tmp_935 = smem_vals[tmp_927 * WPT + 5u];
            let tmp_936 = keys[2] < tmp_934 || (keys[2] == tmp_934 && values[2] < tmp_935);
            if tmp_926 == tmp_936 { keys[2] = tmp_934; values[2] = tmp_935; }
            let tmp_937 = smem_keys[tmp_927 * WPT + 4u];
            let tmp_938 = smem_vals[tmp_927 * WPT + 4u];
            let tmp_939 = keys[3] < tmp_937 || (keys[3] == tmp_937 && values[3] < tmp_938);
            if tmp_926 == tmp_939 { keys[3] = tmp_937; values[3] = tmp_938; }
            let tmp_940 = smem_keys[tmp_927 * WPT + 3u];
            let tmp_941 = smem_vals[tmp_927 * WPT + 3u];
            let tmp_942 = keys[4] < tmp_940 || (keys[4] == tmp_940 && values[4] < tmp_941);
            if tmp_926 == tmp_942 { keys[4] = tmp_940; values[4] = tmp_941; }
            let tmp_943 = smem_keys[tmp_927 * WPT + 2u];
            let tmp_944 = smem_vals[tmp_927 * WPT + 2u];
            let tmp_945 = keys[5] < tmp_943 || (keys[5] == tmp_943 && values[5] < tmp_944);
            if tmp_926 == tmp_945 { keys[5] = tmp_943; values[5] = tmp_944; }
            let tmp_946 = smem_keys[tmp_927 * WPT + 1u];
            let tmp_947 = smem_vals[tmp_927 * WPT + 1u];
            let tmp_948 = keys[6] < tmp_946 || (keys[6] == tmp_946 && values[6] < tmp_947);
            if tmp_926 == tmp_948 { keys[6] = tmp_946; values[6] = tmp_947; }
            let tmp_949 = smem_keys[tmp_927 * WPT + 0u];
            let tmp_950 = smem_vals[tmp_927 * WPT + 0u];
            let tmp_951 = keys[7] < tmp_949 || (keys[7] == tmp_949 && values[7] < tmp_950);
            if tmp_926 == tmp_951 { keys[7] = tmp_949; values[7] = tmp_950; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:64,swbit:6,wpt:8)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],64,6)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            workgroupBarrier();
            let tmp_952 = extractBits(local_tid, 6u, 1u) != 0u;
            let tmp_953 = seg_base + (local_tid ^ 64u);
            let tmp_954 = smem_keys[tmp_953 * WPT + 0u];
            let tmp_955 = smem_vals[tmp_953 * WPT + 0u];
            let tmp_956 = keys[0] < tmp_954 || (keys[0] == tmp_954 && values[0] < tmp_955);
            if tmp_952 == tmp_956 { keys[0] = tmp_954; values[0] = tmp_955; }
            let tmp_957 = smem_keys[tmp_953 * WPT + 1u];
            let tmp_958 = smem_vals[tmp_953 * WPT + 1u];
            let tmp_959 = keys[1] < tmp_957 || (keys[1] == tmp_957 && values[1] < tmp_958);
            if tmp_952 == tmp_959 { keys[1] = tmp_957; values[1] = tmp_958; }
            let tmp_960 = smem_keys[tmp_953 * WPT + 2u];
            let tmp_961 = smem_vals[tmp_953 * WPT + 2u];
            let tmp_962 = keys[2] < tmp_960 || (keys[2] == tmp_960 && values[2] < tmp_961);
            if tmp_952 == tmp_962 { keys[2] = tmp_960; values[2] = tmp_961; }
            let tmp_963 = smem_keys[tmp_953 * WPT + 3u];
            let tmp_964 = smem_vals[tmp_953 * WPT + 3u];
            let tmp_965 = keys[3] < tmp_963 || (keys[3] == tmp_963 && values[3] < tmp_964);
            if tmp_952 == tmp_965 { keys[3] = tmp_963; values[3] = tmp_964; }
            let tmp_966 = smem_keys[tmp_953 * WPT + 4u];
            let tmp_967 = smem_vals[tmp_953 * WPT + 4u];
            let tmp_968 = keys[4] < tmp_966 || (keys[4] == tmp_966 && values[4] < tmp_967);
            if tmp_952 == tmp_968 { keys[4] = tmp_966; values[4] = tmp_967; }
            let tmp_969 = smem_keys[tmp_953 * WPT + 5u];
            let tmp_970 = smem_vals[tmp_953 * WPT + 5u];
            let tmp_971 = keys[5] < tmp_969 || (keys[5] == tmp_969 && values[5] < tmp_970);
            if tmp_952 == tmp_971 { keys[5] = tmp_969; values[5] = tmp_970; }
            let tmp_972 = smem_keys[tmp_953 * WPT + 6u];
            let tmp_973 = smem_vals[tmp_953 * WPT + 6u];
            let tmp_974 = keys[6] < tmp_972 || (keys[6] == tmp_972 && values[6] < tmp_973);
            if tmp_952 == tmp_974 { keys[6] = tmp_972; values[6] = tmp_973; }
            let tmp_975 = smem_keys[tmp_953 * WPT + 7u];
            let tmp_976 = smem_vals[tmp_953 * WPT + 7u];
            let tmp_977 = keys[7] < tmp_975 || (keys[7] == tmp_975 && values[7] < tmp_976);
            if tmp_952 == tmp_977 { keys[7] = tmp_975; values[7] = tmp_976; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:32,swbit:5,wpt:8)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],32,5)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            workgroupBarrier();
            let tmp_978 = extractBits(local_tid, 5u, 1u) != 0u;
            let tmp_979 = seg_base + (local_tid ^ 32u);
            let tmp_980 = smem_keys[tmp_979 * WPT + 0u];
            let tmp_981 = smem_vals[tmp_979 * WPT + 0u];
            let tmp_982 = keys[0] < tmp_980 || (keys[0] == tmp_980 && values[0] < tmp_981);
            if tmp_978 == tmp_982 { keys[0] = tmp_980; values[0] = tmp_981; }
            let tmp_983 = smem_keys[tmp_979 * WPT + 1u];
            let tmp_984 = smem_vals[tmp_979 * WPT + 1u];
            let tmp_985 = keys[1] < tmp_983 || (keys[1] == tmp_983 && values[1] < tmp_984);
            if tmp_978 == tmp_985 { keys[1] = tmp_983; values[1] = tmp_984; }
            let tmp_986 = smem_keys[tmp_979 * WPT + 2u];
            let tmp_987 = smem_vals[tmp_979 * WPT + 2u];
            let tmp_988 = keys[2] < tmp_986 || (keys[2] == tmp_986 && values[2] < tmp_987);
            if tmp_978 == tmp_988 { keys[2] = tmp_986; values[2] = tmp_987; }
            let tmp_989 = smem_keys[tmp_979 * WPT + 3u];
            let tmp_990 = smem_vals[tmp_979 * WPT + 3u];
            let tmp_991 = keys[3] < tmp_989 || (keys[3] == tmp_989 && values[3] < tmp_990);
            if tmp_978 == tmp_991 { keys[3] = tmp_989; values[3] = tmp_990; }
            let tmp_992 = smem_keys[tmp_979 * WPT + 4u];
            let tmp_993 = smem_vals[tmp_979 * WPT + 4u];
            let tmp_994 = keys[4] < tmp_992 || (keys[4] == tmp_992 && values[4] < tmp_993);
            if tmp_978 == tmp_994 { keys[4] = tmp_992; values[4] = tmp_993; }
            let tmp_995 = smem_keys[tmp_979 * WPT + 5u];
            let tmp_996 = smem_vals[tmp_979 * WPT + 5u];
            let tmp_997 = keys[5] < tmp_995 || (keys[5] == tmp_995 && values[5] < tmp_996);
            if tmp_978 == tmp_997 { keys[5] = tmp_995; values[5] = tmp_996; }
            let tmp_998 = smem_keys[tmp_979 * WPT + 6u];
            let tmp_999 = smem_vals[tmp_979 * WPT + 6u];
            let tmp_1000 = keys[6] < tmp_998 || (keys[6] == tmp_998 && values[6] < tmp_999);
            if tmp_978 == tmp_1000 { keys[6] = tmp_998; values[6] = tmp_999; }
            let tmp_1001 = smem_keys[tmp_979 * WPT + 7u];
            let tmp_1002 = smem_vals[tmp_979 * WPT + 7u];
            let tmp_1003 = keys[7] < tmp_1001 || (keys[7] == tmp_1001 && values[7] < tmp_1002);
            if tmp_978 == tmp_1003 { keys[7] = tmp_1001; values[7] = tmp_1002; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:16,swbit:4,wpt:8)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],16,4)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            workgroupBarrier();
            let tmp_1004 = extractBits(local_tid, 4u, 1u) != 0u;
            let tmp_1005 = seg_base + (local_tid ^ 16u);
            let tmp_1006 = smem_keys[tmp_1005 * WPT + 0u];
            let tmp_1007 = smem_vals[tmp_1005 * WPT + 0u];
            let tmp_1008 = keys[0] < tmp_1006 || (keys[0] == tmp_1006 && values[0] < tmp_1007);
            if tmp_1004 == tmp_1008 { keys[0] = tmp_1006; values[0] = tmp_1007; }
            let tmp_1009 = smem_keys[tmp_1005 * WPT + 1u];
            let tmp_1010 = smem_vals[tmp_1005 * WPT + 1u];
            let tmp_1011 = keys[1] < tmp_1009 || (keys[1] == tmp_1009 && values[1] < tmp_1010);
            if tmp_1004 == tmp_1011 { keys[1] = tmp_1009; values[1] = tmp_1010; }
            let tmp_1012 = smem_keys[tmp_1005 * WPT + 2u];
            let tmp_1013 = smem_vals[tmp_1005 * WPT + 2u];
            let tmp_1014 = keys[2] < tmp_1012 || (keys[2] == tmp_1012 && values[2] < tmp_1013);
            if tmp_1004 == tmp_1014 { keys[2] = tmp_1012; values[2] = tmp_1013; }
            let tmp_1015 = smem_keys[tmp_1005 * WPT + 3u];
            let tmp_1016 = smem_vals[tmp_1005 * WPT + 3u];
            let tmp_1017 = keys[3] < tmp_1015 || (keys[3] == tmp_1015 && values[3] < tmp_1016);
            if tmp_1004 == tmp_1017 { keys[3] = tmp_1015; values[3] = tmp_1016; }
            let tmp_1018 = smem_keys[tmp_1005 * WPT + 4u];
            let tmp_1019 = smem_vals[tmp_1005 * WPT + 4u];
            let tmp_1020 = keys[4] < tmp_1018 || (keys[4] == tmp_1018 && values[4] < tmp_1019);
            if tmp_1004 == tmp_1020 { keys[4] = tmp_1018; values[4] = tmp_1019; }
            let tmp_1021 = smem_keys[tmp_1005 * WPT + 5u];
            let tmp_1022 = smem_vals[tmp_1005 * WPT + 5u];
            let tmp_1023 = keys[5] < tmp_1021 || (keys[5] == tmp_1021 && values[5] < tmp_1022);
            if tmp_1004 == tmp_1023 { keys[5] = tmp_1021; values[5] = tmp_1022; }
            let tmp_1024 = smem_keys[tmp_1005 * WPT + 6u];
            let tmp_1025 = smem_vals[tmp_1005 * WPT + 6u];
            let tmp_1026 = keys[6] < tmp_1024 || (keys[6] == tmp_1024 && values[6] < tmp_1025);
            if tmp_1004 == tmp_1026 { keys[6] = tmp_1024; values[6] = tmp_1025; }
            let tmp_1027 = smem_keys[tmp_1005 * WPT + 7u];
            let tmp_1028 = smem_vals[tmp_1005 * WPT + 7u];
            let tmp_1029 = keys[7] < tmp_1027 || (keys[7] == tmp_1027 && values[7] < tmp_1028);
            if tmp_1004 == tmp_1029 { keys[7] = tmp_1027; values[7] = tmp_1028; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:8,swbit:3,wpt:8)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],8,3)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            workgroupBarrier();
            let tmp_1030 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_1031 = seg_base + (local_tid ^ 8u);
            let tmp_1032 = smem_keys[tmp_1031 * WPT + 0u];
            let tmp_1033 = smem_vals[tmp_1031 * WPT + 0u];
            let tmp_1034 = keys[0] < tmp_1032 || (keys[0] == tmp_1032 && values[0] < tmp_1033);
            if tmp_1030 == tmp_1034 { keys[0] = tmp_1032; values[0] = tmp_1033; }
            let tmp_1035 = smem_keys[tmp_1031 * WPT + 1u];
            let tmp_1036 = smem_vals[tmp_1031 * WPT + 1u];
            let tmp_1037 = keys[1] < tmp_1035 || (keys[1] == tmp_1035 && values[1] < tmp_1036);
            if tmp_1030 == tmp_1037 { keys[1] = tmp_1035; values[1] = tmp_1036; }
            let tmp_1038 = smem_keys[tmp_1031 * WPT + 2u];
            let tmp_1039 = smem_vals[tmp_1031 * WPT + 2u];
            let tmp_1040 = keys[2] < tmp_1038 || (keys[2] == tmp_1038 && values[2] < tmp_1039);
            if tmp_1030 == tmp_1040 { keys[2] = tmp_1038; values[2] = tmp_1039; }
            let tmp_1041 = smem_keys[tmp_1031 * WPT + 3u];
            let tmp_1042 = smem_vals[tmp_1031 * WPT + 3u];
            let tmp_1043 = keys[3] < tmp_1041 || (keys[3] == tmp_1041 && values[3] < tmp_1042);
            if tmp_1030 == tmp_1043 { keys[3] = tmp_1041; values[3] = tmp_1042; }
            let tmp_1044 = smem_keys[tmp_1031 * WPT + 4u];
            let tmp_1045 = smem_vals[tmp_1031 * WPT + 4u];
            let tmp_1046 = keys[4] < tmp_1044 || (keys[4] == tmp_1044 && values[4] < tmp_1045);
            if tmp_1030 == tmp_1046 { keys[4] = tmp_1044; values[4] = tmp_1045; }
            let tmp_1047 = smem_keys[tmp_1031 * WPT + 5u];
            let tmp_1048 = smem_vals[tmp_1031 * WPT + 5u];
            let tmp_1049 = keys[5] < tmp_1047 || (keys[5] == tmp_1047 && values[5] < tmp_1048);
            if tmp_1030 == tmp_1049 { keys[5] = tmp_1047; values[5] = tmp_1048; }
            let tmp_1050 = smem_keys[tmp_1031 * WPT + 6u];
            let tmp_1051 = smem_vals[tmp_1031 * WPT + 6u];
            let tmp_1052 = keys[6] < tmp_1050 || (keys[6] == tmp_1050 && values[6] < tmp_1051);
            if tmp_1030 == tmp_1052 { keys[6] = tmp_1050; values[6] = tmp_1051; }
            let tmp_1053 = smem_keys[tmp_1031 * WPT + 7u];
            let tmp_1054 = smem_vals[tmp_1031 * WPT + 7u];
            let tmp_1055 = keys[7] < tmp_1053 || (keys[7] == tmp_1053 && values[7] < tmp_1054);
            if tmp_1030 == tmp_1055 { keys[7] = tmp_1053; values[7] = tmp_1054; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],4,2)
        {
            let tmp_1056 = subgroupShuffleXor(keys[0], 4u);
            let tmp_1057 = subgroupShuffleXor(values[0], 4u);
            let tmp_1058 = subgroupShuffleXor(keys[1], 4u);
            let tmp_1059 = subgroupShuffleXor(values[1], 4u);
            let tmp_1060 = subgroupShuffleXor(keys[2], 4u);
            let tmp_1061 = subgroupShuffleXor(values[2], 4u);
            let tmp_1062 = subgroupShuffleXor(keys[3], 4u);
            let tmp_1063 = subgroupShuffleXor(values[3], 4u);
            let tmp_1064 = subgroupShuffleXor(keys[4], 4u);
            let tmp_1065 = subgroupShuffleXor(values[4], 4u);
            let tmp_1066 = subgroupShuffleXor(keys[5], 4u);
            let tmp_1067 = subgroupShuffleXor(values[5], 4u);
            let tmp_1068 = subgroupShuffleXor(keys[6], 4u);
            let tmp_1069 = subgroupShuffleXor(values[6], 4u);
            let tmp_1070 = subgroupShuffleXor(keys[7], 4u);
            let tmp_1071 = subgroupShuffleXor(values[7], 4u);
            let tmp_1072 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_1073 = keys[0] < tmp_1056 || (keys[0] == tmp_1056 && values[0] < tmp_1057);
            if tmp_1072 == tmp_1073 { keys[0] = tmp_1056; values[0] = tmp_1057; }
            let tmp_1074 = keys[1] < tmp_1058 || (keys[1] == tmp_1058 && values[1] < tmp_1059);
            if tmp_1072 == tmp_1074 { keys[1] = tmp_1058; values[1] = tmp_1059; }
            let tmp_1075 = keys[2] < tmp_1060 || (keys[2] == tmp_1060 && values[2] < tmp_1061);
            if tmp_1072 == tmp_1075 { keys[2] = tmp_1060; values[2] = tmp_1061; }
            let tmp_1076 = keys[3] < tmp_1062 || (keys[3] == tmp_1062 && values[3] < tmp_1063);
            if tmp_1072 == tmp_1076 { keys[3] = tmp_1062; values[3] = tmp_1063; }
            let tmp_1077 = keys[4] < tmp_1064 || (keys[4] == tmp_1064 && values[4] < tmp_1065);
            if tmp_1072 == tmp_1077 { keys[4] = tmp_1064; values[4] = tmp_1065; }
            let tmp_1078 = keys[5] < tmp_1066 || (keys[5] == tmp_1066 && values[5] < tmp_1067);
            if tmp_1072 == tmp_1078 { keys[5] = tmp_1066; values[5] = tmp_1067; }
            let tmp_1079 = keys[6] < tmp_1068 || (keys[6] == tmp_1068 && values[6] < tmp_1069);
            if tmp_1072 == tmp_1079 { keys[6] = tmp_1068; values[6] = tmp_1069; }
            let tmp_1080 = keys[7] < tmp_1070 || (keys[7] == tmp_1070 && values[7] < tmp_1071);
            if tmp_1072 == tmp_1080 { keys[7] = tmp_1070; values[7] = tmp_1071; }
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],2,1)
        {
            let tmp_1081 = subgroupShuffleXor(keys[0], 2u);
            let tmp_1082 = subgroupShuffleXor(values[0], 2u);
            let tmp_1083 = subgroupShuffleXor(keys[1], 2u);
            let tmp_1084 = subgroupShuffleXor(values[1], 2u);
            let tmp_1085 = subgroupShuffleXor(keys[2], 2u);
            let tmp_1086 = subgroupShuffleXor(values[2], 2u);
            let tmp_1087 = subgroupShuffleXor(keys[3], 2u);
            let tmp_1088 = subgroupShuffleXor(values[3], 2u);
            let tmp_1089 = subgroupShuffleXor(keys[4], 2u);
            let tmp_1090 = subgroupShuffleXor(values[4], 2u);
            let tmp_1091 = subgroupShuffleXor(keys[5], 2u);
            let tmp_1092 = subgroupShuffleXor(values[5], 2u);
            let tmp_1093 = subgroupShuffleXor(keys[6], 2u);
            let tmp_1094 = subgroupShuffleXor(values[6], 2u);
            let tmp_1095 = subgroupShuffleXor(keys[7], 2u);
            let tmp_1096 = subgroupShuffleXor(values[7], 2u);
            let tmp_1097 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_1098 = keys[0] < tmp_1081 || (keys[0] == tmp_1081 && values[0] < tmp_1082);
            if tmp_1097 == tmp_1098 { keys[0] = tmp_1081; values[0] = tmp_1082; }
            let tmp_1099 = keys[1] < tmp_1083 || (keys[1] == tmp_1083 && values[1] < tmp_1084);
            if tmp_1097 == tmp_1099 { keys[1] = tmp_1083; values[1] = tmp_1084; }
            let tmp_1100 = keys[2] < tmp_1085 || (keys[2] == tmp_1085 && values[2] < tmp_1086);
            if tmp_1097 == tmp_1100 { keys[2] = tmp_1085; values[2] = tmp_1086; }
            let tmp_1101 = keys[3] < tmp_1087 || (keys[3] == tmp_1087 && values[3] < tmp_1088);
            if tmp_1097 == tmp_1101 { keys[3] = tmp_1087; values[3] = tmp_1088; }
            let tmp_1102 = keys[4] < tmp_1089 || (keys[4] == tmp_1089 && values[4] < tmp_1090);
            if tmp_1097 == tmp_1102 { keys[4] = tmp_1089; values[4] = tmp_1090; }
            let tmp_1103 = keys[5] < tmp_1091 || (keys[5] == tmp_1091 && values[5] < tmp_1092);
            if tmp_1097 == tmp_1103 { keys[5] = tmp_1091; values[5] = tmp_1092; }
            let tmp_1104 = keys[6] < tmp_1093 || (keys[6] == tmp_1093 && values[6] < tmp_1094);
            if tmp_1097 == tmp_1104 { keys[6] = tmp_1093; values[6] = tmp_1094; }
            let tmp_1105 = keys[7] < tmp_1095 || (keys[7] == tmp_1095 && values[7] < tmp_1096);
            if tmp_1097 == tmp_1105 { keys[7] = tmp_1095; values[7] = tmp_1096; }
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:8)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],1,0)
        {
            let tmp_1106 = subgroupShuffleXor(keys[0], 1u);
            let tmp_1107 = subgroupShuffleXor(values[0], 1u);
            let tmp_1108 = subgroupShuffleXor(keys[1], 1u);
            let tmp_1109 = subgroupShuffleXor(values[1], 1u);
            let tmp_1110 = subgroupShuffleXor(keys[2], 1u);
            let tmp_1111 = subgroupShuffleXor(values[2], 1u);
            let tmp_1112 = subgroupShuffleXor(keys[3], 1u);
            let tmp_1113 = subgroupShuffleXor(values[3], 1u);
            let tmp_1114 = subgroupShuffleXor(keys[4], 1u);
            let tmp_1115 = subgroupShuffleXor(values[4], 1u);
            let tmp_1116 = subgroupShuffleXor(keys[5], 1u);
            let tmp_1117 = subgroupShuffleXor(values[5], 1u);
            let tmp_1118 = subgroupShuffleXor(keys[6], 1u);
            let tmp_1119 = subgroupShuffleXor(values[6], 1u);
            let tmp_1120 = subgroupShuffleXor(keys[7], 1u);
            let tmp_1121 = subgroupShuffleXor(values[7], 1u);
            let tmp_1122 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_1123 = keys[0] < tmp_1106 || (keys[0] == tmp_1106 && values[0] < tmp_1107);
            if tmp_1122 == tmp_1123 { keys[0] = tmp_1106; values[0] = tmp_1107; }
            let tmp_1124 = keys[1] < tmp_1108 || (keys[1] == tmp_1108 && values[1] < tmp_1109);
            if tmp_1122 == tmp_1124 { keys[1] = tmp_1108; values[1] = tmp_1109; }
            let tmp_1125 = keys[2] < tmp_1110 || (keys[2] == tmp_1110 && values[2] < tmp_1111);
            if tmp_1122 == tmp_1125 { keys[2] = tmp_1110; values[2] = tmp_1111; }
            let tmp_1126 = keys[3] < tmp_1112 || (keys[3] == tmp_1112 && values[3] < tmp_1113);
            if tmp_1122 == tmp_1126 { keys[3] = tmp_1112; values[3] = tmp_1113; }
            let tmp_1127 = keys[4] < tmp_1114 || (keys[4] == tmp_1114 && values[4] < tmp_1115);
            if tmp_1122 == tmp_1127 { keys[4] = tmp_1114; values[4] = tmp_1115; }
            let tmp_1128 = keys[5] < tmp_1116 || (keys[5] == tmp_1116 && values[5] < tmp_1117);
            if tmp_1122 == tmp_1128 { keys[5] = tmp_1116; values[5] = tmp_1117; }
            let tmp_1129 = keys[6] < tmp_1118 || (keys[6] == tmp_1118 && values[6] < tmp_1119);
            if tmp_1122 == tmp_1129 { keys[6] = tmp_1118; values[6] = tmp_1119; }
            let tmp_1130 = keys[7] < tmp_1120 || (keys[7] == tmp_1120 && values[7] < tmp_1121);
            if tmp_1122 == tmp_1130 { keys[7] = tmp_1120; values[7] = tmp_1121; }
        }
    }

    // exch_local(4,8)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_1131 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_1131;
                let tmp_1132 = values[0]; values[0] = values[4]; values[4] = tmp_1132;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_1133 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_1133;
                let tmp_1134 = values[1]; values[1] = values[5]; values[5] = tmp_1134;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_1135 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_1135;
                let tmp_1136 = values[2]; values[2] = values[6]; values[6] = tmp_1136;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_1137 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_1137;
                let tmp_1138 = values[3]; values[3] = values[7]; values[7] = tmp_1138;
            }
        }
    }

    // exch_local(2,8)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_1139 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_1139;
                let tmp_1140 = values[0]; values[0] = values[2]; values[2] = tmp_1140;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_1141 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_1141;
                let tmp_1142 = values[1]; values[1] = values[3]; values[3] = tmp_1142;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_1143 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_1143;
                let tmp_1144 = values[4]; values[4] = values[6]; values[6] = tmp_1144;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_1145 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_1145;
                let tmp_1146 = values[5]; values[5] = values[7]; values[7] = tmp_1146;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_1147 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_1147;
                let tmp_1148 = values[0]; values[0] = values[1]; values[1] = tmp_1148;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_1149 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_1149;
                let tmp_1150 = values[2]; values[2] = values[3]; values[3] = tmp_1150;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_1151 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_1151;
                let tmp_1152 = values[4]; values[4] = values[5]; values[5] = tmp_1152;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_1153 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_1153;
                let tmp_1154 = values[6]; values[6] = values[7]; values[7] = tmp_1154;
            }
        }
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
