
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
const R: u32 = 16u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybrid_sg16_n128_m64_striped(
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

    // striped (coalesced) store via shared memory
    for (var r = 0u; r < WPT; r = r + 1u) {
        smem_keys[tid_g * WPT + r] = keys[r];
        smem_vals[tid_g * WPT + r] = values[r];
    }
    workgroupBarrier();
    for (var c = 0u; c < WPT; c = c + 1u) {
        let j = c * M + local_tid;
        if is_active && j < seg_size {
            global_keys[seg_start + j] = smem_keys[seg_base * WPT + j];
            global_value_indices[seg_start + j] = smem_vals[seg_base * WPT + j];
        }
    }
}
