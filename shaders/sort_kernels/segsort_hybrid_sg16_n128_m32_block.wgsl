
enable subgroups;

override WG: u32 = 32u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 128u;
const M: u32 = 32u;
const WPT: u32 = 4u;
const R: u32 = 16u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybrid_sg16_n128_m32_block(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 7u;

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
    }

    // exch_local(3,4)
    {
        // cmp_swap(0,3)
        if keys[0] > keys[3] || (keys[0] == keys[3] && values[0] > values[3]) {
            // swap(0,3)
            {
                let tmp_4 = keys[0]; keys[0] = keys[3]; keys[3] = tmp_4;
                let tmp_5 = values[0]; values[0] = values[3]; values[3] = tmp_5;
            }
        }
        // cmp_swap(1,2)
        if keys[1] > keys[2] || (keys[1] == keys[2] && values[1] > values[2]) {
            // swap(1,2)
            {
                let tmp_6 = keys[1]; keys[1] = keys[2]; keys[2] = tmp_6;
                let tmp_7 = values[1]; values[1] = values[2]; values[2] = tmp_7;
            }
        }
    }

    // exch_local(1,4)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_8 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_8;
                let tmp_9 = values[0]; values[0] = values[1]; values[1] = tmp_9;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_10 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_10;
                let tmp_11 = values[2]; values[2] = values[3]; values[3] = tmp_11;
            }
        }
    }

    // exch_intxn(tmask:1,swbit:0,wpt:4)
    {
        // _exch_subgroup([(0, 3), (1, 2), (2, 1), (3, 0)],1,0)
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
    }

    // exch_local(2,4)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_25 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_25;
                let tmp_26 = values[0]; values[0] = values[2]; values[2] = tmp_26;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_27 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_27;
                let tmp_28 = values[1]; values[1] = values[3]; values[3] = tmp_28;
            }
        }
    }

    // exch_local(1,4)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_29 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_29;
                let tmp_30 = values[0]; values[0] = values[1]; values[1] = tmp_30;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_31 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_31;
                let tmp_32 = values[2]; values[2] = values[3]; values[3] = tmp_32;
            }
        }
    }

    // exch_intxn(tmask:3,swbit:1,wpt:4)
    {
        // _exch_subgroup([(0, 3), (1, 2), (2, 1), (3, 0)],3,1)
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
    }

    // exch_paral(tmask:1,swbit:0,wpt:4)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3)],1,0)
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
    }

    // exch_local(2,4)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_59 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_59;
                let tmp_60 = values[0]; values[0] = values[2]; values[2] = tmp_60;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_61 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_61;
                let tmp_62 = values[1]; values[1] = values[3]; values[3] = tmp_62;
            }
        }
    }

    // exch_local(1,4)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_63 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_63;
                let tmp_64 = values[0]; values[0] = values[1]; values[1] = tmp_64;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_65 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_65;
                let tmp_66 = values[2]; values[2] = values[3]; values[3] = tmp_66;
            }
        }
    }

    // exch_intxn(tmask:7,swbit:2,wpt:4)
    {
        // _exch_subgroup([(0, 3), (1, 2), (2, 1), (3, 0)],7,2)
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
    }

    // exch_paral(tmask:2,swbit:1,wpt:4)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3)],2,1)
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
    }

    // exch_paral(tmask:1,swbit:0,wpt:4)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3)],1,0)
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
    }

    // exch_local(2,4)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_106 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_106;
                let tmp_107 = values[0]; values[0] = values[2]; values[2] = tmp_107;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_108 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_108;
                let tmp_109 = values[1]; values[1] = values[3]; values[3] = tmp_109;
            }
        }
    }

    // exch_local(1,4)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_110 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_110;
                let tmp_111 = values[0]; values[0] = values[1]; values[1] = tmp_111;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_112 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_112;
                let tmp_113 = values[2]; values[2] = values[3]; values[3] = tmp_113;
            }
        }
    }

    // exch_intxn(tmask:15,swbit:3,wpt:4)
    {
        // _exch_subgroup([(0, 3), (1, 2), (2, 1), (3, 0)],15,3)
        {
            let tmp_114 = subgroupShuffleXor(keys[3], 15u);
            let tmp_115 = subgroupShuffleXor(values[3], 15u);
            let tmp_116 = subgroupShuffleXor(keys[2], 15u);
            let tmp_117 = subgroupShuffleXor(values[2], 15u);
            let tmp_118 = subgroupShuffleXor(keys[1], 15u);
            let tmp_119 = subgroupShuffleXor(values[1], 15u);
            let tmp_120 = subgroupShuffleXor(keys[0], 15u);
            let tmp_121 = subgroupShuffleXor(values[0], 15u);
            let tmp_122 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_123 = keys[0] < tmp_114 || (keys[0] == tmp_114 && values[0] < tmp_115);
            if tmp_122 == tmp_123 { keys[0] = tmp_114; values[0] = tmp_115; }
            let tmp_124 = keys[1] < tmp_116 || (keys[1] == tmp_116 && values[1] < tmp_117);
            if tmp_122 == tmp_124 { keys[1] = tmp_116; values[1] = tmp_117; }
            let tmp_125 = keys[2] < tmp_118 || (keys[2] == tmp_118 && values[2] < tmp_119);
            if tmp_122 == tmp_125 { keys[2] = tmp_118; values[2] = tmp_119; }
            let tmp_126 = keys[3] < tmp_120 || (keys[3] == tmp_120 && values[3] < tmp_121);
            if tmp_122 == tmp_126 { keys[3] = tmp_120; values[3] = tmp_121; }
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:4)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3)],4,2)
        {
            let tmp_127 = subgroupShuffleXor(keys[0], 4u);
            let tmp_128 = subgroupShuffleXor(values[0], 4u);
            let tmp_129 = subgroupShuffleXor(keys[1], 4u);
            let tmp_130 = subgroupShuffleXor(values[1], 4u);
            let tmp_131 = subgroupShuffleXor(keys[2], 4u);
            let tmp_132 = subgroupShuffleXor(values[2], 4u);
            let tmp_133 = subgroupShuffleXor(keys[3], 4u);
            let tmp_134 = subgroupShuffleXor(values[3], 4u);
            let tmp_135 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_136 = keys[0] < tmp_127 || (keys[0] == tmp_127 && values[0] < tmp_128);
            if tmp_135 == tmp_136 { keys[0] = tmp_127; values[0] = tmp_128; }
            let tmp_137 = keys[1] < tmp_129 || (keys[1] == tmp_129 && values[1] < tmp_130);
            if tmp_135 == tmp_137 { keys[1] = tmp_129; values[1] = tmp_130; }
            let tmp_138 = keys[2] < tmp_131 || (keys[2] == tmp_131 && values[2] < tmp_132);
            if tmp_135 == tmp_138 { keys[2] = tmp_131; values[2] = tmp_132; }
            let tmp_139 = keys[3] < tmp_133 || (keys[3] == tmp_133 && values[3] < tmp_134);
            if tmp_135 == tmp_139 { keys[3] = tmp_133; values[3] = tmp_134; }
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:4)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3)],2,1)
        {
            let tmp_140 = subgroupShuffleXor(keys[0], 2u);
            let tmp_141 = subgroupShuffleXor(values[0], 2u);
            let tmp_142 = subgroupShuffleXor(keys[1], 2u);
            let tmp_143 = subgroupShuffleXor(values[1], 2u);
            let tmp_144 = subgroupShuffleXor(keys[2], 2u);
            let tmp_145 = subgroupShuffleXor(values[2], 2u);
            let tmp_146 = subgroupShuffleXor(keys[3], 2u);
            let tmp_147 = subgroupShuffleXor(values[3], 2u);
            let tmp_148 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_149 = keys[0] < tmp_140 || (keys[0] == tmp_140 && values[0] < tmp_141);
            if tmp_148 == tmp_149 { keys[0] = tmp_140; values[0] = tmp_141; }
            let tmp_150 = keys[1] < tmp_142 || (keys[1] == tmp_142 && values[1] < tmp_143);
            if tmp_148 == tmp_150 { keys[1] = tmp_142; values[1] = tmp_143; }
            let tmp_151 = keys[2] < tmp_144 || (keys[2] == tmp_144 && values[2] < tmp_145);
            if tmp_148 == tmp_151 { keys[2] = tmp_144; values[2] = tmp_145; }
            let tmp_152 = keys[3] < tmp_146 || (keys[3] == tmp_146 && values[3] < tmp_147);
            if tmp_148 == tmp_152 { keys[3] = tmp_146; values[3] = tmp_147; }
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:4)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3)],1,0)
        {
            let tmp_153 = subgroupShuffleXor(keys[0], 1u);
            let tmp_154 = subgroupShuffleXor(values[0], 1u);
            let tmp_155 = subgroupShuffleXor(keys[1], 1u);
            let tmp_156 = subgroupShuffleXor(values[1], 1u);
            let tmp_157 = subgroupShuffleXor(keys[2], 1u);
            let tmp_158 = subgroupShuffleXor(values[2], 1u);
            let tmp_159 = subgroupShuffleXor(keys[3], 1u);
            let tmp_160 = subgroupShuffleXor(values[3], 1u);
            let tmp_161 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_162 = keys[0] < tmp_153 || (keys[0] == tmp_153 && values[0] < tmp_154);
            if tmp_161 == tmp_162 { keys[0] = tmp_153; values[0] = tmp_154; }
            let tmp_163 = keys[1] < tmp_155 || (keys[1] == tmp_155 && values[1] < tmp_156);
            if tmp_161 == tmp_163 { keys[1] = tmp_155; values[1] = tmp_156; }
            let tmp_164 = keys[2] < tmp_157 || (keys[2] == tmp_157 && values[2] < tmp_158);
            if tmp_161 == tmp_164 { keys[2] = tmp_157; values[2] = tmp_158; }
            let tmp_165 = keys[3] < tmp_159 || (keys[3] == tmp_159 && values[3] < tmp_160);
            if tmp_161 == tmp_165 { keys[3] = tmp_159; values[3] = tmp_160; }
        }
    }

    // exch_local(2,4)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_166 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_166;
                let tmp_167 = values[0]; values[0] = values[2]; values[2] = tmp_167;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_168 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_168;
                let tmp_169 = values[1]; values[1] = values[3]; values[3] = tmp_169;
            }
        }
    }

    // exch_local(1,4)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_170 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_170;
                let tmp_171 = values[0]; values[0] = values[1]; values[1] = tmp_171;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_172 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_172;
                let tmp_173 = values[2]; values[2] = values[3]; values[3] = tmp_173;
            }
        }
    }

    // exch_intxn(tmask:31,swbit:4,wpt:4)
    {
        // _exch_workgroup([(0, 3), (1, 2), (2, 1), (3, 0)],31,4)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_174 = extractBits(local_tid, 4u, 1u) != 0u;
            let tmp_175 = seg_base + (local_tid ^ 31u);
            let tmp_176 = smem_keys[tmp_175 * WPT + 3u];
            let tmp_177 = smem_vals[tmp_175 * WPT + 3u];
            let tmp_178 = keys[0] < tmp_176 || (keys[0] == tmp_176 && values[0] < tmp_177);
            if tmp_174 == tmp_178 { keys[0] = tmp_176; values[0] = tmp_177; }
            let tmp_179 = smem_keys[tmp_175 * WPT + 2u];
            let tmp_180 = smem_vals[tmp_175 * WPT + 2u];
            let tmp_181 = keys[1] < tmp_179 || (keys[1] == tmp_179 && values[1] < tmp_180);
            if tmp_174 == tmp_181 { keys[1] = tmp_179; values[1] = tmp_180; }
            let tmp_182 = smem_keys[tmp_175 * WPT + 1u];
            let tmp_183 = smem_vals[tmp_175 * WPT + 1u];
            let tmp_184 = keys[2] < tmp_182 || (keys[2] == tmp_182 && values[2] < tmp_183);
            if tmp_174 == tmp_184 { keys[2] = tmp_182; values[2] = tmp_183; }
            let tmp_185 = smem_keys[tmp_175 * WPT + 0u];
            let tmp_186 = smem_vals[tmp_175 * WPT + 0u];
            let tmp_187 = keys[3] < tmp_185 || (keys[3] == tmp_185 && values[3] < tmp_186);
            if tmp_174 == tmp_187 { keys[3] = tmp_185; values[3] = tmp_186; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:8,swbit:3,wpt:4)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3)],8,3)
        {
            let tmp_188 = subgroupShuffleXor(keys[0], 8u);
            let tmp_189 = subgroupShuffleXor(values[0], 8u);
            let tmp_190 = subgroupShuffleXor(keys[1], 8u);
            let tmp_191 = subgroupShuffleXor(values[1], 8u);
            let tmp_192 = subgroupShuffleXor(keys[2], 8u);
            let tmp_193 = subgroupShuffleXor(values[2], 8u);
            let tmp_194 = subgroupShuffleXor(keys[3], 8u);
            let tmp_195 = subgroupShuffleXor(values[3], 8u);
            let tmp_196 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_197 = keys[0] < tmp_188 || (keys[0] == tmp_188 && values[0] < tmp_189);
            if tmp_196 == tmp_197 { keys[0] = tmp_188; values[0] = tmp_189; }
            let tmp_198 = keys[1] < tmp_190 || (keys[1] == tmp_190 && values[1] < tmp_191);
            if tmp_196 == tmp_198 { keys[1] = tmp_190; values[1] = tmp_191; }
            let tmp_199 = keys[2] < tmp_192 || (keys[2] == tmp_192 && values[2] < tmp_193);
            if tmp_196 == tmp_199 { keys[2] = tmp_192; values[2] = tmp_193; }
            let tmp_200 = keys[3] < tmp_194 || (keys[3] == tmp_194 && values[3] < tmp_195);
            if tmp_196 == tmp_200 { keys[3] = tmp_194; values[3] = tmp_195; }
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:4)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3)],4,2)
        {
            let tmp_201 = subgroupShuffleXor(keys[0], 4u);
            let tmp_202 = subgroupShuffleXor(values[0], 4u);
            let tmp_203 = subgroupShuffleXor(keys[1], 4u);
            let tmp_204 = subgroupShuffleXor(values[1], 4u);
            let tmp_205 = subgroupShuffleXor(keys[2], 4u);
            let tmp_206 = subgroupShuffleXor(values[2], 4u);
            let tmp_207 = subgroupShuffleXor(keys[3], 4u);
            let tmp_208 = subgroupShuffleXor(values[3], 4u);
            let tmp_209 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_210 = keys[0] < tmp_201 || (keys[0] == tmp_201 && values[0] < tmp_202);
            if tmp_209 == tmp_210 { keys[0] = tmp_201; values[0] = tmp_202; }
            let tmp_211 = keys[1] < tmp_203 || (keys[1] == tmp_203 && values[1] < tmp_204);
            if tmp_209 == tmp_211 { keys[1] = tmp_203; values[1] = tmp_204; }
            let tmp_212 = keys[2] < tmp_205 || (keys[2] == tmp_205 && values[2] < tmp_206);
            if tmp_209 == tmp_212 { keys[2] = tmp_205; values[2] = tmp_206; }
            let tmp_213 = keys[3] < tmp_207 || (keys[3] == tmp_207 && values[3] < tmp_208);
            if tmp_209 == tmp_213 { keys[3] = tmp_207; values[3] = tmp_208; }
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:4)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3)],2,1)
        {
            let tmp_214 = subgroupShuffleXor(keys[0], 2u);
            let tmp_215 = subgroupShuffleXor(values[0], 2u);
            let tmp_216 = subgroupShuffleXor(keys[1], 2u);
            let tmp_217 = subgroupShuffleXor(values[1], 2u);
            let tmp_218 = subgroupShuffleXor(keys[2], 2u);
            let tmp_219 = subgroupShuffleXor(values[2], 2u);
            let tmp_220 = subgroupShuffleXor(keys[3], 2u);
            let tmp_221 = subgroupShuffleXor(values[3], 2u);
            let tmp_222 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_223 = keys[0] < tmp_214 || (keys[0] == tmp_214 && values[0] < tmp_215);
            if tmp_222 == tmp_223 { keys[0] = tmp_214; values[0] = tmp_215; }
            let tmp_224 = keys[1] < tmp_216 || (keys[1] == tmp_216 && values[1] < tmp_217);
            if tmp_222 == tmp_224 { keys[1] = tmp_216; values[1] = tmp_217; }
            let tmp_225 = keys[2] < tmp_218 || (keys[2] == tmp_218 && values[2] < tmp_219);
            if tmp_222 == tmp_225 { keys[2] = tmp_218; values[2] = tmp_219; }
            let tmp_226 = keys[3] < tmp_220 || (keys[3] == tmp_220 && values[3] < tmp_221);
            if tmp_222 == tmp_226 { keys[3] = tmp_220; values[3] = tmp_221; }
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:4)
    {
        // _exch_subgroup([(0, 0), (1, 1), (2, 2), (3, 3)],1,0)
        {
            let tmp_227 = subgroupShuffleXor(keys[0], 1u);
            let tmp_228 = subgroupShuffleXor(values[0], 1u);
            let tmp_229 = subgroupShuffleXor(keys[1], 1u);
            let tmp_230 = subgroupShuffleXor(values[1], 1u);
            let tmp_231 = subgroupShuffleXor(keys[2], 1u);
            let tmp_232 = subgroupShuffleXor(values[2], 1u);
            let tmp_233 = subgroupShuffleXor(keys[3], 1u);
            let tmp_234 = subgroupShuffleXor(values[3], 1u);
            let tmp_235 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_236 = keys[0] < tmp_227 || (keys[0] == tmp_227 && values[0] < tmp_228);
            if tmp_235 == tmp_236 { keys[0] = tmp_227; values[0] = tmp_228; }
            let tmp_237 = keys[1] < tmp_229 || (keys[1] == tmp_229 && values[1] < tmp_230);
            if tmp_235 == tmp_237 { keys[1] = tmp_229; values[1] = tmp_230; }
            let tmp_238 = keys[2] < tmp_231 || (keys[2] == tmp_231 && values[2] < tmp_232);
            if tmp_235 == tmp_238 { keys[2] = tmp_231; values[2] = tmp_232; }
            let tmp_239 = keys[3] < tmp_233 || (keys[3] == tmp_233 && values[3] < tmp_234);
            if tmp_235 == tmp_239 { keys[3] = tmp_233; values[3] = tmp_234; }
        }
    }

    // exch_local(2,4)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_240 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_240;
                let tmp_241 = values[0]; values[0] = values[2]; values[2] = tmp_241;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_242 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_242;
                let tmp_243 = values[1]; values[1] = values[3]; values[3] = tmp_243;
            }
        }
    }

    // exch_local(1,4)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_244 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_244;
                let tmp_245 = values[0]; values[0] = values[1]; values[1] = tmp_245;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_246 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_246;
                let tmp_247 = values[2]; values[2] = values[3]; values[3] = tmp_247;
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
