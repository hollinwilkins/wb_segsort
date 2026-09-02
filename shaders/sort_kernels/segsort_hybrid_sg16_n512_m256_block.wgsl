
enable subgroups;

override WG: u32 = 256u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 512u;
const M: u32 = 256u;
const WPT: u32 = 2u;
const R: u32 = 16u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybrid_sg16_n512_m256_block(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 9u;

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
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_80 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_81 = seg_base + (local_tid ^ 31u); let tmp_82 = smem_keys[tmp_81 * WPT + 1u]; let tmp_83 = smem_vals[tmp_81 * WPT + 1u]; let tmp_84 = keys[0] < tmp_82 || (keys[0] == tmp_82 && values[0] < tmp_83); if tmp_80 == tmp_84 { keys[0] = tmp_82; values[0] = tmp_83; } let tmp_85 = smem_keys[tmp_81 * WPT + 0u]; let tmp_86 = smem_vals[tmp_81 * WPT + 0u]; let tmp_87 = keys[1] < tmp_85 || (keys[1] == tmp_85 && values[1] < tmp_86); if tmp_80 == tmp_87 { keys[1] = tmp_85; values[1] = tmp_86; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:2) 
    {
    let tmp_88 = subgroupShuffleXor(keys[0], 8u);
    let tmp_89 = subgroupShuffleXor(values[0], 8u);
    let tmp_90 = subgroupShuffleXor(keys[1], 8u);
    let tmp_91 = subgroupShuffleXor(values[1], 8u);
    let tmp_92 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_93 = keys[0] < tmp_88 || (keys[0] == tmp_88 && values[0] < tmp_89);
    if tmp_92 == tmp_93 { keys[0] = tmp_88; values[0] = tmp_89; }
    let tmp_94 = keys[1] < tmp_90 || (keys[1] == tmp_90 && values[1] < tmp_91);
    if tmp_92 == tmp_94 { keys[1] = tmp_90; values[1] = tmp_91; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    {
    let tmp_95 = subgroupShuffleXor(keys[0], 4u);
    let tmp_96 = subgroupShuffleXor(values[0], 4u);
    let tmp_97 = subgroupShuffleXor(keys[1], 4u);
    let tmp_98 = subgroupShuffleXor(values[1], 4u);
    let tmp_99 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_100 = keys[0] < tmp_95 || (keys[0] == tmp_95 && values[0] < tmp_96);
    if tmp_99 == tmp_100 { keys[0] = tmp_95; values[0] = tmp_96; }
    let tmp_101 = keys[1] < tmp_97 || (keys[1] == tmp_97 && values[1] < tmp_98);
    if tmp_99 == tmp_101 { keys[1] = tmp_97; values[1] = tmp_98; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_102 = subgroupShuffleXor(keys[0], 2u);
    let tmp_103 = subgroupShuffleXor(values[0], 2u);
    let tmp_104 = subgroupShuffleXor(keys[1], 2u);
    let tmp_105 = subgroupShuffleXor(values[1], 2u);
    let tmp_106 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_107 = keys[0] < tmp_102 || (keys[0] == tmp_102 && values[0] < tmp_103);
    if tmp_106 == tmp_107 { keys[0] = tmp_102; values[0] = tmp_103; }
    let tmp_108 = keys[1] < tmp_104 || (keys[1] == tmp_104 && values[1] < tmp_105);
    if tmp_106 == tmp_108 { keys[1] = tmp_104; values[1] = tmp_105; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_109 = subgroupShuffleXor(keys[0], 1u);
    let tmp_110 = subgroupShuffleXor(values[0], 1u);
    let tmp_111 = subgroupShuffleXor(keys[1], 1u);
    let tmp_112 = subgroupShuffleXor(values[1], 1u);
    let tmp_113 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_114 = keys[0] < tmp_109 || (keys[0] == tmp_109 && values[0] < tmp_110);
    if tmp_113 == tmp_114 { keys[0] = tmp_109; values[0] = tmp_110; }
    let tmp_115 = keys[1] < tmp_111 || (keys[1] == tmp_111 && values[1] < tmp_112);
    if tmp_113 == tmp_115 { keys[1] = tmp_111; values[1] = tmp_112; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_116 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_116;let tmp_117 = values[0]; values[0] = values[1]; values[1] = tmp_117; }
    }
    // exch_intxn(tmask:63,swbit:5,wpt:2)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_118 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_119 = seg_base + (local_tid ^ 63u); let tmp_120 = smem_keys[tmp_119 * WPT + 1u]; let tmp_121 = smem_vals[tmp_119 * WPT + 1u]; let tmp_122 = keys[0] < tmp_120 || (keys[0] == tmp_120 && values[0] < tmp_121); if tmp_118 == tmp_122 { keys[0] = tmp_120; values[0] = tmp_121; } let tmp_123 = smem_keys[tmp_119 * WPT + 0u]; let tmp_124 = smem_vals[tmp_119 * WPT + 0u]; let tmp_125 = keys[1] < tmp_123 || (keys[1] == tmp_123 && values[1] < tmp_124); if tmp_118 == tmp_125 { keys[1] = tmp_123; values[1] = tmp_124; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_126 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_127 = seg_base + (local_tid ^ 16u); let tmp_128 = smem_keys[tmp_127 * WPT + 0u]; let tmp_129 = smem_vals[tmp_127 * WPT + 0u]; let tmp_130 = keys[0] < tmp_128 || (keys[0] == tmp_128 && values[0] < tmp_129); if tmp_126 == tmp_130 { keys[0] = tmp_128; values[0] = tmp_129; } let tmp_131 = smem_keys[tmp_127 * WPT + 1u]; let tmp_132 = smem_vals[tmp_127 * WPT + 1u]; let tmp_133 = keys[1] < tmp_131 || (keys[1] == tmp_131 && values[1] < tmp_132); if tmp_126 == tmp_133 { keys[1] = tmp_131; values[1] = tmp_132; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:2) 
    {
    let tmp_134 = subgroupShuffleXor(keys[0], 8u);
    let tmp_135 = subgroupShuffleXor(values[0], 8u);
    let tmp_136 = subgroupShuffleXor(keys[1], 8u);
    let tmp_137 = subgroupShuffleXor(values[1], 8u);
    let tmp_138 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_139 = keys[0] < tmp_134 || (keys[0] == tmp_134 && values[0] < tmp_135);
    if tmp_138 == tmp_139 { keys[0] = tmp_134; values[0] = tmp_135; }
    let tmp_140 = keys[1] < tmp_136 || (keys[1] == tmp_136 && values[1] < tmp_137);
    if tmp_138 == tmp_140 { keys[1] = tmp_136; values[1] = tmp_137; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    {
    let tmp_141 = subgroupShuffleXor(keys[0], 4u);
    let tmp_142 = subgroupShuffleXor(values[0], 4u);
    let tmp_143 = subgroupShuffleXor(keys[1], 4u);
    let tmp_144 = subgroupShuffleXor(values[1], 4u);
    let tmp_145 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_146 = keys[0] < tmp_141 || (keys[0] == tmp_141 && values[0] < tmp_142);
    if tmp_145 == tmp_146 { keys[0] = tmp_141; values[0] = tmp_142; }
    let tmp_147 = keys[1] < tmp_143 || (keys[1] == tmp_143 && values[1] < tmp_144);
    if tmp_145 == tmp_147 { keys[1] = tmp_143; values[1] = tmp_144; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_148 = subgroupShuffleXor(keys[0], 2u);
    let tmp_149 = subgroupShuffleXor(values[0], 2u);
    let tmp_150 = subgroupShuffleXor(keys[1], 2u);
    let tmp_151 = subgroupShuffleXor(values[1], 2u);
    let tmp_152 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_153 = keys[0] < tmp_148 || (keys[0] == tmp_148 && values[0] < tmp_149);
    if tmp_152 == tmp_153 { keys[0] = tmp_148; values[0] = tmp_149; }
    let tmp_154 = keys[1] < tmp_150 || (keys[1] == tmp_150 && values[1] < tmp_151);
    if tmp_152 == tmp_154 { keys[1] = tmp_150; values[1] = tmp_151; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_155 = subgroupShuffleXor(keys[0], 1u);
    let tmp_156 = subgroupShuffleXor(values[0], 1u);
    let tmp_157 = subgroupShuffleXor(keys[1], 1u);
    let tmp_158 = subgroupShuffleXor(values[1], 1u);
    let tmp_159 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_160 = keys[0] < tmp_155 || (keys[0] == tmp_155 && values[0] < tmp_156);
    if tmp_159 == tmp_160 { keys[0] = tmp_155; values[0] = tmp_156; }
    let tmp_161 = keys[1] < tmp_157 || (keys[1] == tmp_157 && values[1] < tmp_158);
    if tmp_159 == tmp_161 { keys[1] = tmp_157; values[1] = tmp_158; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_162 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_162;let tmp_163 = values[0]; values[0] = values[1]; values[1] = tmp_163; }
    }
    // exch_intxn(tmask:127,swbit:6,wpt:2)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_164 = extractBits(local_tid, 6u, 1u) != 0u; let tmp_165 = seg_base + (local_tid ^ 127u); let tmp_166 = smem_keys[tmp_165 * WPT + 1u]; let tmp_167 = smem_vals[tmp_165 * WPT + 1u]; let tmp_168 = keys[0] < tmp_166 || (keys[0] == tmp_166 && values[0] < tmp_167); if tmp_164 == tmp_168 { keys[0] = tmp_166; values[0] = tmp_167; } let tmp_169 = smem_keys[tmp_165 * WPT + 0u]; let tmp_170 = smem_vals[tmp_165 * WPT + 0u]; let tmp_171 = keys[1] < tmp_169 || (keys[1] == tmp_169 && values[1] < tmp_170); if tmp_164 == tmp_171 { keys[1] = tmp_169; values[1] = tmp_170; } workgroupBarrier(); }
    // exch_paral(tmask:32,swbit:5,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_172 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_173 = seg_base + (local_tid ^ 32u); let tmp_174 = smem_keys[tmp_173 * WPT + 0u]; let tmp_175 = smem_vals[tmp_173 * WPT + 0u]; let tmp_176 = keys[0] < tmp_174 || (keys[0] == tmp_174 && values[0] < tmp_175); if tmp_172 == tmp_176 { keys[0] = tmp_174; values[0] = tmp_175; } let tmp_177 = smem_keys[tmp_173 * WPT + 1u]; let tmp_178 = smem_vals[tmp_173 * WPT + 1u]; let tmp_179 = keys[1] < tmp_177 || (keys[1] == tmp_177 && values[1] < tmp_178); if tmp_172 == tmp_179 { keys[1] = tmp_177; values[1] = tmp_178; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_180 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_181 = seg_base + (local_tid ^ 16u); let tmp_182 = smem_keys[tmp_181 * WPT + 0u]; let tmp_183 = smem_vals[tmp_181 * WPT + 0u]; let tmp_184 = keys[0] < tmp_182 || (keys[0] == tmp_182 && values[0] < tmp_183); if tmp_180 == tmp_184 { keys[0] = tmp_182; values[0] = tmp_183; } let tmp_185 = smem_keys[tmp_181 * WPT + 1u]; let tmp_186 = smem_vals[tmp_181 * WPT + 1u]; let tmp_187 = keys[1] < tmp_185 || (keys[1] == tmp_185 && values[1] < tmp_186); if tmp_180 == tmp_187 { keys[1] = tmp_185; values[1] = tmp_186; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:2) 
    {
    let tmp_188 = subgroupShuffleXor(keys[0], 8u);
    let tmp_189 = subgroupShuffleXor(values[0], 8u);
    let tmp_190 = subgroupShuffleXor(keys[1], 8u);
    let tmp_191 = subgroupShuffleXor(values[1], 8u);
    let tmp_192 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_193 = keys[0] < tmp_188 || (keys[0] == tmp_188 && values[0] < tmp_189);
    if tmp_192 == tmp_193 { keys[0] = tmp_188; values[0] = tmp_189; }
    let tmp_194 = keys[1] < tmp_190 || (keys[1] == tmp_190 && values[1] < tmp_191);
    if tmp_192 == tmp_194 { keys[1] = tmp_190; values[1] = tmp_191; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    {
    let tmp_195 = subgroupShuffleXor(keys[0], 4u);
    let tmp_196 = subgroupShuffleXor(values[0], 4u);
    let tmp_197 = subgroupShuffleXor(keys[1], 4u);
    let tmp_198 = subgroupShuffleXor(values[1], 4u);
    let tmp_199 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_200 = keys[0] < tmp_195 || (keys[0] == tmp_195 && values[0] < tmp_196);
    if tmp_199 == tmp_200 { keys[0] = tmp_195; values[0] = tmp_196; }
    let tmp_201 = keys[1] < tmp_197 || (keys[1] == tmp_197 && values[1] < tmp_198);
    if tmp_199 == tmp_201 { keys[1] = tmp_197; values[1] = tmp_198; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_202 = subgroupShuffleXor(keys[0], 2u);
    let tmp_203 = subgroupShuffleXor(values[0], 2u);
    let tmp_204 = subgroupShuffleXor(keys[1], 2u);
    let tmp_205 = subgroupShuffleXor(values[1], 2u);
    let tmp_206 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_207 = keys[0] < tmp_202 || (keys[0] == tmp_202 && values[0] < tmp_203);
    if tmp_206 == tmp_207 { keys[0] = tmp_202; values[0] = tmp_203; }
    let tmp_208 = keys[1] < tmp_204 || (keys[1] == tmp_204 && values[1] < tmp_205);
    if tmp_206 == tmp_208 { keys[1] = tmp_204; values[1] = tmp_205; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_209 = subgroupShuffleXor(keys[0], 1u);
    let tmp_210 = subgroupShuffleXor(values[0], 1u);
    let tmp_211 = subgroupShuffleXor(keys[1], 1u);
    let tmp_212 = subgroupShuffleXor(values[1], 1u);
    let tmp_213 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_214 = keys[0] < tmp_209 || (keys[0] == tmp_209 && values[0] < tmp_210);
    if tmp_213 == tmp_214 { keys[0] = tmp_209; values[0] = tmp_210; }
    let tmp_215 = keys[1] < tmp_211 || (keys[1] == tmp_211 && values[1] < tmp_212);
    if tmp_213 == tmp_215 { keys[1] = tmp_211; values[1] = tmp_212; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_216 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_216;let tmp_217 = values[0]; values[0] = values[1]; values[1] = tmp_217; }
    }
    // exch_intxn(tmask:255,swbit:7,wpt:2)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_218 = extractBits(local_tid, 7u, 1u) != 0u; let tmp_219 = seg_base + (local_tid ^ 255u); let tmp_220 = smem_keys[tmp_219 * WPT + 1u]; let tmp_221 = smem_vals[tmp_219 * WPT + 1u]; let tmp_222 = keys[0] < tmp_220 || (keys[0] == tmp_220 && values[0] < tmp_221); if tmp_218 == tmp_222 { keys[0] = tmp_220; values[0] = tmp_221; } let tmp_223 = smem_keys[tmp_219 * WPT + 0u]; let tmp_224 = smem_vals[tmp_219 * WPT + 0u]; let tmp_225 = keys[1] < tmp_223 || (keys[1] == tmp_223 && values[1] < tmp_224); if tmp_218 == tmp_225 { keys[1] = tmp_223; values[1] = tmp_224; } workgroupBarrier(); }
    // exch_paral(tmask:64,swbit:6,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_226 = extractBits(local_tid, 6u, 1u) != 0u; let tmp_227 = seg_base + (local_tid ^ 64u); let tmp_228 = smem_keys[tmp_227 * WPT + 0u]; let tmp_229 = smem_vals[tmp_227 * WPT + 0u]; let tmp_230 = keys[0] < tmp_228 || (keys[0] == tmp_228 && values[0] < tmp_229); if tmp_226 == tmp_230 { keys[0] = tmp_228; values[0] = tmp_229; } let tmp_231 = smem_keys[tmp_227 * WPT + 1u]; let tmp_232 = smem_vals[tmp_227 * WPT + 1u]; let tmp_233 = keys[1] < tmp_231 || (keys[1] == tmp_231 && values[1] < tmp_232); if tmp_226 == tmp_233 { keys[1] = tmp_231; values[1] = tmp_232; } workgroupBarrier(); }
    // exch_paral(tmask:32,swbit:5,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_234 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_235 = seg_base + (local_tid ^ 32u); let tmp_236 = smem_keys[tmp_235 * WPT + 0u]; let tmp_237 = smem_vals[tmp_235 * WPT + 0u]; let tmp_238 = keys[0] < tmp_236 || (keys[0] == tmp_236 && values[0] < tmp_237); if tmp_234 == tmp_238 { keys[0] = tmp_236; values[0] = tmp_237; } let tmp_239 = smem_keys[tmp_235 * WPT + 1u]; let tmp_240 = smem_vals[tmp_235 * WPT + 1u]; let tmp_241 = keys[1] < tmp_239 || (keys[1] == tmp_239 && values[1] < tmp_240); if tmp_234 == tmp_241 { keys[1] = tmp_239; values[1] = tmp_240; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_242 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_243 = seg_base + (local_tid ^ 16u); let tmp_244 = smem_keys[tmp_243 * WPT + 0u]; let tmp_245 = smem_vals[tmp_243 * WPT + 0u]; let tmp_246 = keys[0] < tmp_244 || (keys[0] == tmp_244 && values[0] < tmp_245); if tmp_242 == tmp_246 { keys[0] = tmp_244; values[0] = tmp_245; } let tmp_247 = smem_keys[tmp_243 * WPT + 1u]; let tmp_248 = smem_vals[tmp_243 * WPT + 1u]; let tmp_249 = keys[1] < tmp_247 || (keys[1] == tmp_247 && values[1] < tmp_248); if tmp_242 == tmp_249 { keys[1] = tmp_247; values[1] = tmp_248; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:2) 
    {
    let tmp_250 = subgroupShuffleXor(keys[0], 8u);
    let tmp_251 = subgroupShuffleXor(values[0], 8u);
    let tmp_252 = subgroupShuffleXor(keys[1], 8u);
    let tmp_253 = subgroupShuffleXor(values[1], 8u);
    let tmp_254 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_255 = keys[0] < tmp_250 || (keys[0] == tmp_250 && values[0] < tmp_251);
    if tmp_254 == tmp_255 { keys[0] = tmp_250; values[0] = tmp_251; }
    let tmp_256 = keys[1] < tmp_252 || (keys[1] == tmp_252 && values[1] < tmp_253);
    if tmp_254 == tmp_256 { keys[1] = tmp_252; values[1] = tmp_253; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    {
    let tmp_257 = subgroupShuffleXor(keys[0], 4u);
    let tmp_258 = subgroupShuffleXor(values[0], 4u);
    let tmp_259 = subgroupShuffleXor(keys[1], 4u);
    let tmp_260 = subgroupShuffleXor(values[1], 4u);
    let tmp_261 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_262 = keys[0] < tmp_257 || (keys[0] == tmp_257 && values[0] < tmp_258);
    if tmp_261 == tmp_262 { keys[0] = tmp_257; values[0] = tmp_258; }
    let tmp_263 = keys[1] < tmp_259 || (keys[1] == tmp_259 && values[1] < tmp_260);
    if tmp_261 == tmp_263 { keys[1] = tmp_259; values[1] = tmp_260; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_264 = subgroupShuffleXor(keys[0], 2u);
    let tmp_265 = subgroupShuffleXor(values[0], 2u);
    let tmp_266 = subgroupShuffleXor(keys[1], 2u);
    let tmp_267 = subgroupShuffleXor(values[1], 2u);
    let tmp_268 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_269 = keys[0] < tmp_264 || (keys[0] == tmp_264 && values[0] < tmp_265);
    if tmp_268 == tmp_269 { keys[0] = tmp_264; values[0] = tmp_265; }
    let tmp_270 = keys[1] < tmp_266 || (keys[1] == tmp_266 && values[1] < tmp_267);
    if tmp_268 == tmp_270 { keys[1] = tmp_266; values[1] = tmp_267; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_271 = subgroupShuffleXor(keys[0], 1u);
    let tmp_272 = subgroupShuffleXor(values[0], 1u);
    let tmp_273 = subgroupShuffleXor(keys[1], 1u);
    let tmp_274 = subgroupShuffleXor(values[1], 1u);
    let tmp_275 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_276 = keys[0] < tmp_271 || (keys[0] == tmp_271 && values[0] < tmp_272);
    if tmp_275 == tmp_276 { keys[0] = tmp_271; values[0] = tmp_272; }
    let tmp_277 = keys[1] < tmp_273 || (keys[1] == tmp_273 && values[1] < tmp_274);
    if tmp_275 == tmp_277 { keys[1] = tmp_273; values[1] = tmp_274; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_278 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_278;let tmp_279 = values[0]; values[0] = values[1]; values[1] = tmp_279; }
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
