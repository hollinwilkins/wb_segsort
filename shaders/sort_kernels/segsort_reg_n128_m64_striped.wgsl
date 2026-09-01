
enable subgroups;

override WG: u32 = 64u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 128u;
const M: u32 = 64u;
const WPT: u32 = 2u;

@compute @workgroup_size(WG, 1, 1)
fn segsort_reg_n128_m64_striped(
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
    // exch_intxn(tmask:15,swbit:3,wpt:2)
    {
    let tmp_50 = subgroupShuffleXor(keys[1], 15u);
    let tmp_51 = subgroupShuffleXor(values[1], 15u);
    let tmp_52 = subgroupShuffleXor(keys[0], 15u);
    let tmp_53 = subgroupShuffleXor(values[0], 15u);
    let tmp_54 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_55 = keys[0] < tmp_50 || (keys[0] == tmp_50 && values[0] < tmp_51);
    if tmp_54 == tmp_55 { keys[0] = tmp_50; values[0] = tmp_51; }
    let tmp_56 = keys[1] < tmp_52 || (keys[1] == tmp_52 && values[1] < tmp_53);
    if tmp_54 == tmp_56 { keys[1] = tmp_52; values[1] = tmp_53; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    {
    let tmp_57 = subgroupShuffleXor(keys[0], 4u);
    let tmp_58 = subgroupShuffleXor(values[0], 4u);
    let tmp_59 = subgroupShuffleXor(keys[1], 4u);
    let tmp_60 = subgroupShuffleXor(values[1], 4u);
    let tmp_61 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_62 = keys[0] < tmp_57 || (keys[0] == tmp_57 && values[0] < tmp_58);
    if tmp_61 == tmp_62 { keys[0] = tmp_57; values[0] = tmp_58; }
    let tmp_63 = keys[1] < tmp_59 || (keys[1] == tmp_59 && values[1] < tmp_60);
    if tmp_61 == tmp_63 { keys[1] = tmp_59; values[1] = tmp_60; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_64 = subgroupShuffleXor(keys[0], 2u);
    let tmp_65 = subgroupShuffleXor(values[0], 2u);
    let tmp_66 = subgroupShuffleXor(keys[1], 2u);
    let tmp_67 = subgroupShuffleXor(values[1], 2u);
    let tmp_68 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_69 = keys[0] < tmp_64 || (keys[0] == tmp_64 && values[0] < tmp_65);
    if tmp_68 == tmp_69 { keys[0] = tmp_64; values[0] = tmp_65; }
    let tmp_70 = keys[1] < tmp_66 || (keys[1] == tmp_66 && values[1] < tmp_67);
    if tmp_68 == tmp_70 { keys[1] = tmp_66; values[1] = tmp_67; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_71 = subgroupShuffleXor(keys[0], 1u);
    let tmp_72 = subgroupShuffleXor(values[0], 1u);
    let tmp_73 = subgroupShuffleXor(keys[1], 1u);
    let tmp_74 = subgroupShuffleXor(values[1], 1u);
    let tmp_75 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_76 = keys[0] < tmp_71 || (keys[0] == tmp_71 && values[0] < tmp_72);
    if tmp_75 == tmp_76 { keys[0] = tmp_71; values[0] = tmp_72; }
    let tmp_77 = keys[1] < tmp_73 || (keys[1] == tmp_73 && values[1] < tmp_74);
    if tmp_75 == tmp_77 { keys[1] = tmp_73; values[1] = tmp_74; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_78 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_78;let tmp_79 = values[0]; values[0] = values[1]; values[1] = tmp_79; }
    }
    // exch_intxn(tmask:31,swbit:4,wpt:2)
    {
    let tmp_80 = subgroupShuffleXor(keys[1], 31u);
    let tmp_81 = subgroupShuffleXor(values[1], 31u);
    let tmp_82 = subgroupShuffleXor(keys[0], 31u);
    let tmp_83 = subgroupShuffleXor(values[0], 31u);
    let tmp_84 = extractBits(local_tid, 4u, 1u) != 0u;
    let tmp_85 = keys[0] < tmp_80 || (keys[0] == tmp_80 && values[0] < tmp_81);
    if tmp_84 == tmp_85 { keys[0] = tmp_80; values[0] = tmp_81; }
    let tmp_86 = keys[1] < tmp_82 || (keys[1] == tmp_82 && values[1] < tmp_83);
    if tmp_84 == tmp_86 { keys[1] = tmp_82; values[1] = tmp_83; }
    }
    // exch_paral(tmask:8,swbit:3,wpt:2) 
    {
    let tmp_87 = subgroupShuffleXor(keys[0], 8u);
    let tmp_88 = subgroupShuffleXor(values[0], 8u);
    let tmp_89 = subgroupShuffleXor(keys[1], 8u);
    let tmp_90 = subgroupShuffleXor(values[1], 8u);
    let tmp_91 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_92 = keys[0] < tmp_87 || (keys[0] == tmp_87 && values[0] < tmp_88);
    if tmp_91 == tmp_92 { keys[0] = tmp_87; values[0] = tmp_88; }
    let tmp_93 = keys[1] < tmp_89 || (keys[1] == tmp_89 && values[1] < tmp_90);
    if tmp_91 == tmp_93 { keys[1] = tmp_89; values[1] = tmp_90; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    {
    let tmp_94 = subgroupShuffleXor(keys[0], 4u);
    let tmp_95 = subgroupShuffleXor(values[0], 4u);
    let tmp_96 = subgroupShuffleXor(keys[1], 4u);
    let tmp_97 = subgroupShuffleXor(values[1], 4u);
    let tmp_98 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_99 = keys[0] < tmp_94 || (keys[0] == tmp_94 && values[0] < tmp_95);
    if tmp_98 == tmp_99 { keys[0] = tmp_94; values[0] = tmp_95; }
    let tmp_100 = keys[1] < tmp_96 || (keys[1] == tmp_96 && values[1] < tmp_97);
    if tmp_98 == tmp_100 { keys[1] = tmp_96; values[1] = tmp_97; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_101 = subgroupShuffleXor(keys[0], 2u);
    let tmp_102 = subgroupShuffleXor(values[0], 2u);
    let tmp_103 = subgroupShuffleXor(keys[1], 2u);
    let tmp_104 = subgroupShuffleXor(values[1], 2u);
    let tmp_105 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_106 = keys[0] < tmp_101 || (keys[0] == tmp_101 && values[0] < tmp_102);
    if tmp_105 == tmp_106 { keys[0] = tmp_101; values[0] = tmp_102; }
    let tmp_107 = keys[1] < tmp_103 || (keys[1] == tmp_103 && values[1] < tmp_104);
    if tmp_105 == tmp_107 { keys[1] = tmp_103; values[1] = tmp_104; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_108 = subgroupShuffleXor(keys[0], 1u);
    let tmp_109 = subgroupShuffleXor(values[0], 1u);
    let tmp_110 = subgroupShuffleXor(keys[1], 1u);
    let tmp_111 = subgroupShuffleXor(values[1], 1u);
    let tmp_112 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_113 = keys[0] < tmp_108 || (keys[0] == tmp_108 && values[0] < tmp_109);
    if tmp_112 == tmp_113 { keys[0] = tmp_108; values[0] = tmp_109; }
    let tmp_114 = keys[1] < tmp_110 || (keys[1] == tmp_110 && values[1] < tmp_111);
    if tmp_112 == tmp_114 { keys[1] = tmp_110; values[1] = tmp_111; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_115 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_115;let tmp_116 = values[0]; values[0] = values[1]; values[1] = tmp_116; }
    }
    // exch_intxn(tmask:63,swbit:5,wpt:2)
    {
    let tmp_117 = subgroupShuffleXor(keys[1], 63u);
    let tmp_118 = subgroupShuffleXor(values[1], 63u);
    let tmp_119 = subgroupShuffleXor(keys[0], 63u);
    let tmp_120 = subgroupShuffleXor(values[0], 63u);
    let tmp_121 = extractBits(local_tid, 5u, 1u) != 0u;
    let tmp_122 = keys[0] < tmp_117 || (keys[0] == tmp_117 && values[0] < tmp_118);
    if tmp_121 == tmp_122 { keys[0] = tmp_117; values[0] = tmp_118; }
    let tmp_123 = keys[1] < tmp_119 || (keys[1] == tmp_119 && values[1] < tmp_120);
    if tmp_121 == tmp_123 { keys[1] = tmp_119; values[1] = tmp_120; }
    }
    // exch_paral(tmask:16,swbit:4,wpt:2) 
    {
    let tmp_124 = subgroupShuffleXor(keys[0], 16u);
    let tmp_125 = subgroupShuffleXor(values[0], 16u);
    let tmp_126 = subgroupShuffleXor(keys[1], 16u);
    let tmp_127 = subgroupShuffleXor(values[1], 16u);
    let tmp_128 = extractBits(local_tid, 4u, 1u) != 0u;
    let tmp_129 = keys[0] < tmp_124 || (keys[0] == tmp_124 && values[0] < tmp_125);
    if tmp_128 == tmp_129 { keys[0] = tmp_124; values[0] = tmp_125; }
    let tmp_130 = keys[1] < tmp_126 || (keys[1] == tmp_126 && values[1] < tmp_127);
    if tmp_128 == tmp_130 { keys[1] = tmp_126; values[1] = tmp_127; }
    }
    // exch_paral(tmask:8,swbit:3,wpt:2) 
    {
    let tmp_131 = subgroupShuffleXor(keys[0], 8u);
    let tmp_132 = subgroupShuffleXor(values[0], 8u);
    let tmp_133 = subgroupShuffleXor(keys[1], 8u);
    let tmp_134 = subgroupShuffleXor(values[1], 8u);
    let tmp_135 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_136 = keys[0] < tmp_131 || (keys[0] == tmp_131 && values[0] < tmp_132);
    if tmp_135 == tmp_136 { keys[0] = tmp_131; values[0] = tmp_132; }
    let tmp_137 = keys[1] < tmp_133 || (keys[1] == tmp_133 && values[1] < tmp_134);
    if tmp_135 == tmp_137 { keys[1] = tmp_133; values[1] = tmp_134; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    {
    let tmp_138 = subgroupShuffleXor(keys[0], 4u);
    let tmp_139 = subgroupShuffleXor(values[0], 4u);
    let tmp_140 = subgroupShuffleXor(keys[1], 4u);
    let tmp_141 = subgroupShuffleXor(values[1], 4u);
    let tmp_142 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_143 = keys[0] < tmp_138 || (keys[0] == tmp_138 && values[0] < tmp_139);
    if tmp_142 == tmp_143 { keys[0] = tmp_138; values[0] = tmp_139; }
    let tmp_144 = keys[1] < tmp_140 || (keys[1] == tmp_140 && values[1] < tmp_141);
    if tmp_142 == tmp_144 { keys[1] = tmp_140; values[1] = tmp_141; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_145 = subgroupShuffleXor(keys[0], 2u);
    let tmp_146 = subgroupShuffleXor(values[0], 2u);
    let tmp_147 = subgroupShuffleXor(keys[1], 2u);
    let tmp_148 = subgroupShuffleXor(values[1], 2u);
    let tmp_149 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_150 = keys[0] < tmp_145 || (keys[0] == tmp_145 && values[0] < tmp_146);
    if tmp_149 == tmp_150 { keys[0] = tmp_145; values[0] = tmp_146; }
    let tmp_151 = keys[1] < tmp_147 || (keys[1] == tmp_147 && values[1] < tmp_148);
    if tmp_149 == tmp_151 { keys[1] = tmp_147; values[1] = tmp_148; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_152 = subgroupShuffleXor(keys[0], 1u);
    let tmp_153 = subgroupShuffleXor(values[0], 1u);
    let tmp_154 = subgroupShuffleXor(keys[1], 1u);
    let tmp_155 = subgroupShuffleXor(values[1], 1u);
    let tmp_156 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_157 = keys[0] < tmp_152 || (keys[0] == tmp_152 && values[0] < tmp_153);
    if tmp_156 == tmp_157 { keys[0] = tmp_152; values[0] = tmp_153; }
    let tmp_158 = keys[1] < tmp_154 || (keys[1] == tmp_154 && values[1] < tmp_155);
    if tmp_156 == tmp_158 { keys[1] = tmp_154; values[1] = tmp_155; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_159 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_159;let tmp_160 = values[0]; values[0] = values[1]; values[1] = tmp_160; }
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
