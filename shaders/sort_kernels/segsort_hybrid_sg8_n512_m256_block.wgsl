
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
const R: u32 = 8u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybrid_sg8_n512_m256_block(
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
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_50 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_51 = seg_base + (local_tid ^ 15u); let tmp_52 = smem_keys[tmp_51 * WPT + 1u]; let tmp_53 = smem_vals[tmp_51 * WPT + 1u]; let tmp_54 = keys[0] < tmp_52 || (keys[0] == tmp_52 && values[0] < tmp_53); if tmp_50 == tmp_54 { keys[0] = tmp_52; values[0] = tmp_53; } let tmp_55 = smem_keys[tmp_51 * WPT + 0u]; let tmp_56 = smem_vals[tmp_51 * WPT + 0u]; let tmp_57 = keys[1] < tmp_55 || (keys[1] == tmp_55 && values[1] < tmp_56); if tmp_50 == tmp_57 { keys[1] = tmp_55; values[1] = tmp_56; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    {
    let tmp_58 = subgroupShuffleXor(keys[0], 4u);
    let tmp_59 = subgroupShuffleXor(values[0], 4u);
    let tmp_60 = subgroupShuffleXor(keys[1], 4u);
    let tmp_61 = subgroupShuffleXor(values[1], 4u);
    let tmp_62 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_63 = keys[0] < tmp_58 || (keys[0] == tmp_58 && values[0] < tmp_59);
    if tmp_62 == tmp_63 { keys[0] = tmp_58; values[0] = tmp_59; }
    let tmp_64 = keys[1] < tmp_60 || (keys[1] == tmp_60 && values[1] < tmp_61);
    if tmp_62 == tmp_64 { keys[1] = tmp_60; values[1] = tmp_61; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_65 = subgroupShuffleXor(keys[0], 2u);
    let tmp_66 = subgroupShuffleXor(values[0], 2u);
    let tmp_67 = subgroupShuffleXor(keys[1], 2u);
    let tmp_68 = subgroupShuffleXor(values[1], 2u);
    let tmp_69 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_70 = keys[0] < tmp_65 || (keys[0] == tmp_65 && values[0] < tmp_66);
    if tmp_69 == tmp_70 { keys[0] = tmp_65; values[0] = tmp_66; }
    let tmp_71 = keys[1] < tmp_67 || (keys[1] == tmp_67 && values[1] < tmp_68);
    if tmp_69 == tmp_71 { keys[1] = tmp_67; values[1] = tmp_68; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_72 = subgroupShuffleXor(keys[0], 1u);
    let tmp_73 = subgroupShuffleXor(values[0], 1u);
    let tmp_74 = subgroupShuffleXor(keys[1], 1u);
    let tmp_75 = subgroupShuffleXor(values[1], 1u);
    let tmp_76 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_77 = keys[0] < tmp_72 || (keys[0] == tmp_72 && values[0] < tmp_73);
    if tmp_76 == tmp_77 { keys[0] = tmp_72; values[0] = tmp_73; }
    let tmp_78 = keys[1] < tmp_74 || (keys[1] == tmp_74 && values[1] < tmp_75);
    if tmp_76 == tmp_78 { keys[1] = tmp_74; values[1] = tmp_75; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_79 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_79;let tmp_80 = values[0]; values[0] = values[1]; values[1] = tmp_80; }
    }
    // exch_intxn(tmask:31,swbit:4,wpt:2)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_81 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_82 = seg_base + (local_tid ^ 31u); let tmp_83 = smem_keys[tmp_82 * WPT + 1u]; let tmp_84 = smem_vals[tmp_82 * WPT + 1u]; let tmp_85 = keys[0] < tmp_83 || (keys[0] == tmp_83 && values[0] < tmp_84); if tmp_81 == tmp_85 { keys[0] = tmp_83; values[0] = tmp_84; } let tmp_86 = smem_keys[tmp_82 * WPT + 0u]; let tmp_87 = smem_vals[tmp_82 * WPT + 0u]; let tmp_88 = keys[1] < tmp_86 || (keys[1] == tmp_86 && values[1] < tmp_87); if tmp_81 == tmp_88 { keys[1] = tmp_86; values[1] = tmp_87; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_89 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_90 = seg_base + (local_tid ^ 8u); let tmp_91 = smem_keys[tmp_90 * WPT + 0u]; let tmp_92 = smem_vals[tmp_90 * WPT + 0u]; let tmp_93 = keys[0] < tmp_91 || (keys[0] == tmp_91 && values[0] < tmp_92); if tmp_89 == tmp_93 { keys[0] = tmp_91; values[0] = tmp_92; } let tmp_94 = smem_keys[tmp_90 * WPT + 1u]; let tmp_95 = smem_vals[tmp_90 * WPT + 1u]; let tmp_96 = keys[1] < tmp_94 || (keys[1] == tmp_94 && values[1] < tmp_95); if tmp_89 == tmp_96 { keys[1] = tmp_94; values[1] = tmp_95; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    {
    let tmp_97 = subgroupShuffleXor(keys[0], 4u);
    let tmp_98 = subgroupShuffleXor(values[0], 4u);
    let tmp_99 = subgroupShuffleXor(keys[1], 4u);
    let tmp_100 = subgroupShuffleXor(values[1], 4u);
    let tmp_101 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_102 = keys[0] < tmp_97 || (keys[0] == tmp_97 && values[0] < tmp_98);
    if tmp_101 == tmp_102 { keys[0] = tmp_97; values[0] = tmp_98; }
    let tmp_103 = keys[1] < tmp_99 || (keys[1] == tmp_99 && values[1] < tmp_100);
    if tmp_101 == tmp_103 { keys[1] = tmp_99; values[1] = tmp_100; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_104 = subgroupShuffleXor(keys[0], 2u);
    let tmp_105 = subgroupShuffleXor(values[0], 2u);
    let tmp_106 = subgroupShuffleXor(keys[1], 2u);
    let tmp_107 = subgroupShuffleXor(values[1], 2u);
    let tmp_108 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_109 = keys[0] < tmp_104 || (keys[0] == tmp_104 && values[0] < tmp_105);
    if tmp_108 == tmp_109 { keys[0] = tmp_104; values[0] = tmp_105; }
    let tmp_110 = keys[1] < tmp_106 || (keys[1] == tmp_106 && values[1] < tmp_107);
    if tmp_108 == tmp_110 { keys[1] = tmp_106; values[1] = tmp_107; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_111 = subgroupShuffleXor(keys[0], 1u);
    let tmp_112 = subgroupShuffleXor(values[0], 1u);
    let tmp_113 = subgroupShuffleXor(keys[1], 1u);
    let tmp_114 = subgroupShuffleXor(values[1], 1u);
    let tmp_115 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_116 = keys[0] < tmp_111 || (keys[0] == tmp_111 && values[0] < tmp_112);
    if tmp_115 == tmp_116 { keys[0] = tmp_111; values[0] = tmp_112; }
    let tmp_117 = keys[1] < tmp_113 || (keys[1] == tmp_113 && values[1] < tmp_114);
    if tmp_115 == tmp_117 { keys[1] = tmp_113; values[1] = tmp_114; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_118 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_118;let tmp_119 = values[0]; values[0] = values[1]; values[1] = tmp_119; }
    }
    // exch_intxn(tmask:63,swbit:5,wpt:2)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_120 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_121 = seg_base + (local_tid ^ 63u); let tmp_122 = smem_keys[tmp_121 * WPT + 1u]; let tmp_123 = smem_vals[tmp_121 * WPT + 1u]; let tmp_124 = keys[0] < tmp_122 || (keys[0] == tmp_122 && values[0] < tmp_123); if tmp_120 == tmp_124 { keys[0] = tmp_122; values[0] = tmp_123; } let tmp_125 = smem_keys[tmp_121 * WPT + 0u]; let tmp_126 = smem_vals[tmp_121 * WPT + 0u]; let tmp_127 = keys[1] < tmp_125 || (keys[1] == tmp_125 && values[1] < tmp_126); if tmp_120 == tmp_127 { keys[1] = tmp_125; values[1] = tmp_126; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_128 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_129 = seg_base + (local_tid ^ 16u); let tmp_130 = smem_keys[tmp_129 * WPT + 0u]; let tmp_131 = smem_vals[tmp_129 * WPT + 0u]; let tmp_132 = keys[0] < tmp_130 || (keys[0] == tmp_130 && values[0] < tmp_131); if tmp_128 == tmp_132 { keys[0] = tmp_130; values[0] = tmp_131; } let tmp_133 = smem_keys[tmp_129 * WPT + 1u]; let tmp_134 = smem_vals[tmp_129 * WPT + 1u]; let tmp_135 = keys[1] < tmp_133 || (keys[1] == tmp_133 && values[1] < tmp_134); if tmp_128 == tmp_135 { keys[1] = tmp_133; values[1] = tmp_134; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_136 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_137 = seg_base + (local_tid ^ 8u); let tmp_138 = smem_keys[tmp_137 * WPT + 0u]; let tmp_139 = smem_vals[tmp_137 * WPT + 0u]; let tmp_140 = keys[0] < tmp_138 || (keys[0] == tmp_138 && values[0] < tmp_139); if tmp_136 == tmp_140 { keys[0] = tmp_138; values[0] = tmp_139; } let tmp_141 = smem_keys[tmp_137 * WPT + 1u]; let tmp_142 = smem_vals[tmp_137 * WPT + 1u]; let tmp_143 = keys[1] < tmp_141 || (keys[1] == tmp_141 && values[1] < tmp_142); if tmp_136 == tmp_143 { keys[1] = tmp_141; values[1] = tmp_142; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    {
    let tmp_144 = subgroupShuffleXor(keys[0], 4u);
    let tmp_145 = subgroupShuffleXor(values[0], 4u);
    let tmp_146 = subgroupShuffleXor(keys[1], 4u);
    let tmp_147 = subgroupShuffleXor(values[1], 4u);
    let tmp_148 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_149 = keys[0] < tmp_144 || (keys[0] == tmp_144 && values[0] < tmp_145);
    if tmp_148 == tmp_149 { keys[0] = tmp_144; values[0] = tmp_145; }
    let tmp_150 = keys[1] < tmp_146 || (keys[1] == tmp_146 && values[1] < tmp_147);
    if tmp_148 == tmp_150 { keys[1] = tmp_146; values[1] = tmp_147; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_151 = subgroupShuffleXor(keys[0], 2u);
    let tmp_152 = subgroupShuffleXor(values[0], 2u);
    let tmp_153 = subgroupShuffleXor(keys[1], 2u);
    let tmp_154 = subgroupShuffleXor(values[1], 2u);
    let tmp_155 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_156 = keys[0] < tmp_151 || (keys[0] == tmp_151 && values[0] < tmp_152);
    if tmp_155 == tmp_156 { keys[0] = tmp_151; values[0] = tmp_152; }
    let tmp_157 = keys[1] < tmp_153 || (keys[1] == tmp_153 && values[1] < tmp_154);
    if tmp_155 == tmp_157 { keys[1] = tmp_153; values[1] = tmp_154; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_158 = subgroupShuffleXor(keys[0], 1u);
    let tmp_159 = subgroupShuffleXor(values[0], 1u);
    let tmp_160 = subgroupShuffleXor(keys[1], 1u);
    let tmp_161 = subgroupShuffleXor(values[1], 1u);
    let tmp_162 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_163 = keys[0] < tmp_158 || (keys[0] == tmp_158 && values[0] < tmp_159);
    if tmp_162 == tmp_163 { keys[0] = tmp_158; values[0] = tmp_159; }
    let tmp_164 = keys[1] < tmp_160 || (keys[1] == tmp_160 && values[1] < tmp_161);
    if tmp_162 == tmp_164 { keys[1] = tmp_160; values[1] = tmp_161; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_165 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_165;let tmp_166 = values[0]; values[0] = values[1]; values[1] = tmp_166; }
    }
    // exch_intxn(tmask:127,swbit:6,wpt:2)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_167 = extractBits(local_tid, 6u, 1u) != 0u; let tmp_168 = seg_base + (local_tid ^ 127u); let tmp_169 = smem_keys[tmp_168 * WPT + 1u]; let tmp_170 = smem_vals[tmp_168 * WPT + 1u]; let tmp_171 = keys[0] < tmp_169 || (keys[0] == tmp_169 && values[0] < tmp_170); if tmp_167 == tmp_171 { keys[0] = tmp_169; values[0] = tmp_170; } let tmp_172 = smem_keys[tmp_168 * WPT + 0u]; let tmp_173 = smem_vals[tmp_168 * WPT + 0u]; let tmp_174 = keys[1] < tmp_172 || (keys[1] == tmp_172 && values[1] < tmp_173); if tmp_167 == tmp_174 { keys[1] = tmp_172; values[1] = tmp_173; } workgroupBarrier(); }
    // exch_paral(tmask:32,swbit:5,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_175 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_176 = seg_base + (local_tid ^ 32u); let tmp_177 = smem_keys[tmp_176 * WPT + 0u]; let tmp_178 = smem_vals[tmp_176 * WPT + 0u]; let tmp_179 = keys[0] < tmp_177 || (keys[0] == tmp_177 && values[0] < tmp_178); if tmp_175 == tmp_179 { keys[0] = tmp_177; values[0] = tmp_178; } let tmp_180 = smem_keys[tmp_176 * WPT + 1u]; let tmp_181 = smem_vals[tmp_176 * WPT + 1u]; let tmp_182 = keys[1] < tmp_180 || (keys[1] == tmp_180 && values[1] < tmp_181); if tmp_175 == tmp_182 { keys[1] = tmp_180; values[1] = tmp_181; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_183 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_184 = seg_base + (local_tid ^ 16u); let tmp_185 = smem_keys[tmp_184 * WPT + 0u]; let tmp_186 = smem_vals[tmp_184 * WPT + 0u]; let tmp_187 = keys[0] < tmp_185 || (keys[0] == tmp_185 && values[0] < tmp_186); if tmp_183 == tmp_187 { keys[0] = tmp_185; values[0] = tmp_186; } let tmp_188 = smem_keys[tmp_184 * WPT + 1u]; let tmp_189 = smem_vals[tmp_184 * WPT + 1u]; let tmp_190 = keys[1] < tmp_188 || (keys[1] == tmp_188 && values[1] < tmp_189); if tmp_183 == tmp_190 { keys[1] = tmp_188; values[1] = tmp_189; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_191 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_192 = seg_base + (local_tid ^ 8u); let tmp_193 = smem_keys[tmp_192 * WPT + 0u]; let tmp_194 = smem_vals[tmp_192 * WPT + 0u]; let tmp_195 = keys[0] < tmp_193 || (keys[0] == tmp_193 && values[0] < tmp_194); if tmp_191 == tmp_195 { keys[0] = tmp_193; values[0] = tmp_194; } let tmp_196 = smem_keys[tmp_192 * WPT + 1u]; let tmp_197 = smem_vals[tmp_192 * WPT + 1u]; let tmp_198 = keys[1] < tmp_196 || (keys[1] == tmp_196 && values[1] < tmp_197); if tmp_191 == tmp_198 { keys[1] = tmp_196; values[1] = tmp_197; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    {
    let tmp_199 = subgroupShuffleXor(keys[0], 4u);
    let tmp_200 = subgroupShuffleXor(values[0], 4u);
    let tmp_201 = subgroupShuffleXor(keys[1], 4u);
    let tmp_202 = subgroupShuffleXor(values[1], 4u);
    let tmp_203 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_204 = keys[0] < tmp_199 || (keys[0] == tmp_199 && values[0] < tmp_200);
    if tmp_203 == tmp_204 { keys[0] = tmp_199; values[0] = tmp_200; }
    let tmp_205 = keys[1] < tmp_201 || (keys[1] == tmp_201 && values[1] < tmp_202);
    if tmp_203 == tmp_205 { keys[1] = tmp_201; values[1] = tmp_202; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_206 = subgroupShuffleXor(keys[0], 2u);
    let tmp_207 = subgroupShuffleXor(values[0], 2u);
    let tmp_208 = subgroupShuffleXor(keys[1], 2u);
    let tmp_209 = subgroupShuffleXor(values[1], 2u);
    let tmp_210 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_211 = keys[0] < tmp_206 || (keys[0] == tmp_206 && values[0] < tmp_207);
    if tmp_210 == tmp_211 { keys[0] = tmp_206; values[0] = tmp_207; }
    let tmp_212 = keys[1] < tmp_208 || (keys[1] == tmp_208 && values[1] < tmp_209);
    if tmp_210 == tmp_212 { keys[1] = tmp_208; values[1] = tmp_209; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_213 = subgroupShuffleXor(keys[0], 1u);
    let tmp_214 = subgroupShuffleXor(values[0], 1u);
    let tmp_215 = subgroupShuffleXor(keys[1], 1u);
    let tmp_216 = subgroupShuffleXor(values[1], 1u);
    let tmp_217 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_218 = keys[0] < tmp_213 || (keys[0] == tmp_213 && values[0] < tmp_214);
    if tmp_217 == tmp_218 { keys[0] = tmp_213; values[0] = tmp_214; }
    let tmp_219 = keys[1] < tmp_215 || (keys[1] == tmp_215 && values[1] < tmp_216);
    if tmp_217 == tmp_219 { keys[1] = tmp_215; values[1] = tmp_216; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_220 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_220;let tmp_221 = values[0]; values[0] = values[1]; values[1] = tmp_221; }
    }
    // exch_intxn(tmask:255,swbit:7,wpt:2)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_222 = extractBits(local_tid, 7u, 1u) != 0u; let tmp_223 = seg_base + (local_tid ^ 255u); let tmp_224 = smem_keys[tmp_223 * WPT + 1u]; let tmp_225 = smem_vals[tmp_223 * WPT + 1u]; let tmp_226 = keys[0] < tmp_224 || (keys[0] == tmp_224 && values[0] < tmp_225); if tmp_222 == tmp_226 { keys[0] = tmp_224; values[0] = tmp_225; } let tmp_227 = smem_keys[tmp_223 * WPT + 0u]; let tmp_228 = smem_vals[tmp_223 * WPT + 0u]; let tmp_229 = keys[1] < tmp_227 || (keys[1] == tmp_227 && values[1] < tmp_228); if tmp_222 == tmp_229 { keys[1] = tmp_227; values[1] = tmp_228; } workgroupBarrier(); }
    // exch_paral(tmask:64,swbit:6,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_230 = extractBits(local_tid, 6u, 1u) != 0u; let tmp_231 = seg_base + (local_tid ^ 64u); let tmp_232 = smem_keys[tmp_231 * WPT + 0u]; let tmp_233 = smem_vals[tmp_231 * WPT + 0u]; let tmp_234 = keys[0] < tmp_232 || (keys[0] == tmp_232 && values[0] < tmp_233); if tmp_230 == tmp_234 { keys[0] = tmp_232; values[0] = tmp_233; } let tmp_235 = smem_keys[tmp_231 * WPT + 1u]; let tmp_236 = smem_vals[tmp_231 * WPT + 1u]; let tmp_237 = keys[1] < tmp_235 || (keys[1] == tmp_235 && values[1] < tmp_236); if tmp_230 == tmp_237 { keys[1] = tmp_235; values[1] = tmp_236; } workgroupBarrier(); }
    // exch_paral(tmask:32,swbit:5,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_238 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_239 = seg_base + (local_tid ^ 32u); let tmp_240 = smem_keys[tmp_239 * WPT + 0u]; let tmp_241 = smem_vals[tmp_239 * WPT + 0u]; let tmp_242 = keys[0] < tmp_240 || (keys[0] == tmp_240 && values[0] < tmp_241); if tmp_238 == tmp_242 { keys[0] = tmp_240; values[0] = tmp_241; } let tmp_243 = smem_keys[tmp_239 * WPT + 1u]; let tmp_244 = smem_vals[tmp_239 * WPT + 1u]; let tmp_245 = keys[1] < tmp_243 || (keys[1] == tmp_243 && values[1] < tmp_244); if tmp_238 == tmp_245 { keys[1] = tmp_243; values[1] = tmp_244; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_246 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_247 = seg_base + (local_tid ^ 16u); let tmp_248 = smem_keys[tmp_247 * WPT + 0u]; let tmp_249 = smem_vals[tmp_247 * WPT + 0u]; let tmp_250 = keys[0] < tmp_248 || (keys[0] == tmp_248 && values[0] < tmp_249); if tmp_246 == tmp_250 { keys[0] = tmp_248; values[0] = tmp_249; } let tmp_251 = smem_keys[tmp_247 * WPT + 1u]; let tmp_252 = smem_vals[tmp_247 * WPT + 1u]; let tmp_253 = keys[1] < tmp_251 || (keys[1] == tmp_251 && values[1] < tmp_252); if tmp_246 == tmp_253 { keys[1] = tmp_251; values[1] = tmp_252; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_254 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_255 = seg_base + (local_tid ^ 8u); let tmp_256 = smem_keys[tmp_255 * WPT + 0u]; let tmp_257 = smem_vals[tmp_255 * WPT + 0u]; let tmp_258 = keys[0] < tmp_256 || (keys[0] == tmp_256 && values[0] < tmp_257); if tmp_254 == tmp_258 { keys[0] = tmp_256; values[0] = tmp_257; } let tmp_259 = smem_keys[tmp_255 * WPT + 1u]; let tmp_260 = smem_vals[tmp_255 * WPT + 1u]; let tmp_261 = keys[1] < tmp_259 || (keys[1] == tmp_259 && values[1] < tmp_260); if tmp_254 == tmp_261 { keys[1] = tmp_259; values[1] = tmp_260; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    {
    let tmp_262 = subgroupShuffleXor(keys[0], 4u);
    let tmp_263 = subgroupShuffleXor(values[0], 4u);
    let tmp_264 = subgroupShuffleXor(keys[1], 4u);
    let tmp_265 = subgroupShuffleXor(values[1], 4u);
    let tmp_266 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_267 = keys[0] < tmp_262 || (keys[0] == tmp_262 && values[0] < tmp_263);
    if tmp_266 == tmp_267 { keys[0] = tmp_262; values[0] = tmp_263; }
    let tmp_268 = keys[1] < tmp_264 || (keys[1] == tmp_264 && values[1] < tmp_265);
    if tmp_266 == tmp_268 { keys[1] = tmp_264; values[1] = tmp_265; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_269 = subgroupShuffleXor(keys[0], 2u);
    let tmp_270 = subgroupShuffleXor(values[0], 2u);
    let tmp_271 = subgroupShuffleXor(keys[1], 2u);
    let tmp_272 = subgroupShuffleXor(values[1], 2u);
    let tmp_273 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_274 = keys[0] < tmp_269 || (keys[0] == tmp_269 && values[0] < tmp_270);
    if tmp_273 == tmp_274 { keys[0] = tmp_269; values[0] = tmp_270; }
    let tmp_275 = keys[1] < tmp_271 || (keys[1] == tmp_271 && values[1] < tmp_272);
    if tmp_273 == tmp_275 { keys[1] = tmp_271; values[1] = tmp_272; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_276 = subgroupShuffleXor(keys[0], 1u);
    let tmp_277 = subgroupShuffleXor(values[0], 1u);
    let tmp_278 = subgroupShuffleXor(keys[1], 1u);
    let tmp_279 = subgroupShuffleXor(values[1], 1u);
    let tmp_280 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_281 = keys[0] < tmp_276 || (keys[0] == tmp_276 && values[0] < tmp_277);
    if tmp_280 == tmp_281 { keys[0] = tmp_276; values[0] = tmp_277; }
    let tmp_282 = keys[1] < tmp_278 || (keys[1] == tmp_278 && values[1] < tmp_279);
    if tmp_280 == tmp_282 { keys[1] = tmp_278; values[1] = tmp_279; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_283 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_283;let tmp_284 = values[0]; values[0] = values[1]; values[1] = tmp_284; }
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
