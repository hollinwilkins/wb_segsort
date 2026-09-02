
enable subgroups;

override WG: u32 = 64u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 256u;
const M: u32 = 64u;
const WPT: u32 = 4u;
const R: u32 = 8u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybrid_sg8_n256_m64_block(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 8u;

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
    // exch_intxn(tmask:15,swbit:3,wpt:4)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_114 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_115 = seg_base + (local_tid ^ 15u); let tmp_116 = smem_keys[tmp_115 * WPT + 3u]; let tmp_117 = smem_vals[tmp_115 * WPT + 3u]; let tmp_118 = keys[0] < tmp_116 || (keys[0] == tmp_116 && values[0] < tmp_117); if tmp_114 == tmp_118 { keys[0] = tmp_116; values[0] = tmp_117; } let tmp_119 = smem_keys[tmp_115 * WPT + 2u]; let tmp_120 = smem_vals[tmp_115 * WPT + 2u]; let tmp_121 = keys[1] < tmp_119 || (keys[1] == tmp_119 && values[1] < tmp_120); if tmp_114 == tmp_121 { keys[1] = tmp_119; values[1] = tmp_120; } let tmp_122 = smem_keys[tmp_115 * WPT + 1u]; let tmp_123 = smem_vals[tmp_115 * WPT + 1u]; let tmp_124 = keys[2] < tmp_122 || (keys[2] == tmp_122 && values[2] < tmp_123); if tmp_114 == tmp_124 { keys[2] = tmp_122; values[2] = tmp_123; } let tmp_125 = smem_keys[tmp_115 * WPT + 0u]; let tmp_126 = smem_vals[tmp_115 * WPT + 0u]; let tmp_127 = keys[3] < tmp_125 || (keys[3] == tmp_125 && values[3] < tmp_126); if tmp_114 == tmp_127 { keys[3] = tmp_125; values[3] = tmp_126; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:4) 
    {
    let tmp_128 = subgroupShuffleXor(keys[0], 4u);
    let tmp_129 = subgroupShuffleXor(values[0], 4u);
    let tmp_130 = subgroupShuffleXor(keys[1], 4u);
    let tmp_131 = subgroupShuffleXor(values[1], 4u);
    let tmp_132 = subgroupShuffleXor(keys[2], 4u);
    let tmp_133 = subgroupShuffleXor(values[2], 4u);
    let tmp_134 = subgroupShuffleXor(keys[3], 4u);
    let tmp_135 = subgroupShuffleXor(values[3], 4u);
    let tmp_136 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_137 = keys[0] < tmp_128 || (keys[0] == tmp_128 && values[0] < tmp_129);
    if tmp_136 == tmp_137 { keys[0] = tmp_128; values[0] = tmp_129; }
    let tmp_138 = keys[1] < tmp_130 || (keys[1] == tmp_130 && values[1] < tmp_131);
    if tmp_136 == tmp_138 { keys[1] = tmp_130; values[1] = tmp_131; }
    let tmp_139 = keys[2] < tmp_132 || (keys[2] == tmp_132 && values[2] < tmp_133);
    if tmp_136 == tmp_139 { keys[2] = tmp_132; values[2] = tmp_133; }
    let tmp_140 = keys[3] < tmp_134 || (keys[3] == tmp_134 && values[3] < tmp_135);
    if tmp_136 == tmp_140 { keys[3] = tmp_134; values[3] = tmp_135; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:4) 
    {
    let tmp_141 = subgroupShuffleXor(keys[0], 2u);
    let tmp_142 = subgroupShuffleXor(values[0], 2u);
    let tmp_143 = subgroupShuffleXor(keys[1], 2u);
    let tmp_144 = subgroupShuffleXor(values[1], 2u);
    let tmp_145 = subgroupShuffleXor(keys[2], 2u);
    let tmp_146 = subgroupShuffleXor(values[2], 2u);
    let tmp_147 = subgroupShuffleXor(keys[3], 2u);
    let tmp_148 = subgroupShuffleXor(values[3], 2u);
    let tmp_149 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_150 = keys[0] < tmp_141 || (keys[0] == tmp_141 && values[0] < tmp_142);
    if tmp_149 == tmp_150 { keys[0] = tmp_141; values[0] = tmp_142; }
    let tmp_151 = keys[1] < tmp_143 || (keys[1] == tmp_143 && values[1] < tmp_144);
    if tmp_149 == tmp_151 { keys[1] = tmp_143; values[1] = tmp_144; }
    let tmp_152 = keys[2] < tmp_145 || (keys[2] == tmp_145 && values[2] < tmp_146);
    if tmp_149 == tmp_152 { keys[2] = tmp_145; values[2] = tmp_146; }
    let tmp_153 = keys[3] < tmp_147 || (keys[3] == tmp_147 && values[3] < tmp_148);
    if tmp_149 == tmp_153 { keys[3] = tmp_147; values[3] = tmp_148; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:4) 
    {
    let tmp_154 = subgroupShuffleXor(keys[0], 1u);
    let tmp_155 = subgroupShuffleXor(values[0], 1u);
    let tmp_156 = subgroupShuffleXor(keys[1], 1u);
    let tmp_157 = subgroupShuffleXor(values[1], 1u);
    let tmp_158 = subgroupShuffleXor(keys[2], 1u);
    let tmp_159 = subgroupShuffleXor(values[2], 1u);
    let tmp_160 = subgroupShuffleXor(keys[3], 1u);
    let tmp_161 = subgroupShuffleXor(values[3], 1u);
    let tmp_162 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_163 = keys[0] < tmp_154 || (keys[0] == tmp_154 && values[0] < tmp_155);
    if tmp_162 == tmp_163 { keys[0] = tmp_154; values[0] = tmp_155; }
    let tmp_164 = keys[1] < tmp_156 || (keys[1] == tmp_156 && values[1] < tmp_157);
    if tmp_162 == tmp_164 { keys[1] = tmp_156; values[1] = tmp_157; }
    let tmp_165 = keys[2] < tmp_158 || (keys[2] == tmp_158 && values[2] < tmp_159);
    if tmp_162 == tmp_165 { keys[2] = tmp_158; values[2] = tmp_159; }
    let tmp_166 = keys[3] < tmp_160 || (keys[3] == tmp_160 && values[3] < tmp_161);
    if tmp_162 == tmp_166 { keys[3] = tmp_160; values[3] = tmp_161; }
    }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_167 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_167;let tmp_168 = values[0]; values[0] = values[2]; values[2] = tmp_168; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_169 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_169;let tmp_170 = values[1]; values[1] = values[3]; values[3] = tmp_170; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_171 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_171;let tmp_172 = values[0]; values[0] = values[1]; values[1] = tmp_172; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_173 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_173;let tmp_174 = values[2]; values[2] = values[3]; values[3] = tmp_174; }
    }
    // exch_intxn(tmask:31,swbit:4,wpt:4)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_175 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_176 = seg_base + (local_tid ^ 31u); let tmp_177 = smem_keys[tmp_176 * WPT + 3u]; let tmp_178 = smem_vals[tmp_176 * WPT + 3u]; let tmp_179 = keys[0] < tmp_177 || (keys[0] == tmp_177 && values[0] < tmp_178); if tmp_175 == tmp_179 { keys[0] = tmp_177; values[0] = tmp_178; } let tmp_180 = smem_keys[tmp_176 * WPT + 2u]; let tmp_181 = smem_vals[tmp_176 * WPT + 2u]; let tmp_182 = keys[1] < tmp_180 || (keys[1] == tmp_180 && values[1] < tmp_181); if tmp_175 == tmp_182 { keys[1] = tmp_180; values[1] = tmp_181; } let tmp_183 = smem_keys[tmp_176 * WPT + 1u]; let tmp_184 = smem_vals[tmp_176 * WPT + 1u]; let tmp_185 = keys[2] < tmp_183 || (keys[2] == tmp_183 && values[2] < tmp_184); if tmp_175 == tmp_185 { keys[2] = tmp_183; values[2] = tmp_184; } let tmp_186 = smem_keys[tmp_176 * WPT + 0u]; let tmp_187 = smem_vals[tmp_176 * WPT + 0u]; let tmp_188 = keys[3] < tmp_186 || (keys[3] == tmp_186 && values[3] < tmp_187); if tmp_175 == tmp_188 { keys[3] = tmp_186; values[3] = tmp_187; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_189 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_190 = seg_base + (local_tid ^ 8u); let tmp_191 = smem_keys[tmp_190 * WPT + 0u]; let tmp_192 = smem_vals[tmp_190 * WPT + 0u]; let tmp_193 = keys[0] < tmp_191 || (keys[0] == tmp_191 && values[0] < tmp_192); if tmp_189 == tmp_193 { keys[0] = tmp_191; values[0] = tmp_192; } let tmp_194 = smem_keys[tmp_190 * WPT + 1u]; let tmp_195 = smem_vals[tmp_190 * WPT + 1u]; let tmp_196 = keys[1] < tmp_194 || (keys[1] == tmp_194 && values[1] < tmp_195); if tmp_189 == tmp_196 { keys[1] = tmp_194; values[1] = tmp_195; } let tmp_197 = smem_keys[tmp_190 * WPT + 2u]; let tmp_198 = smem_vals[tmp_190 * WPT + 2u]; let tmp_199 = keys[2] < tmp_197 || (keys[2] == tmp_197 && values[2] < tmp_198); if tmp_189 == tmp_199 { keys[2] = tmp_197; values[2] = tmp_198; } let tmp_200 = smem_keys[tmp_190 * WPT + 3u]; let tmp_201 = smem_vals[tmp_190 * WPT + 3u]; let tmp_202 = keys[3] < tmp_200 || (keys[3] == tmp_200 && values[3] < tmp_201); if tmp_189 == tmp_202 { keys[3] = tmp_200; values[3] = tmp_201; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:4) 
    {
    let tmp_203 = subgroupShuffleXor(keys[0], 4u);
    let tmp_204 = subgroupShuffleXor(values[0], 4u);
    let tmp_205 = subgroupShuffleXor(keys[1], 4u);
    let tmp_206 = subgroupShuffleXor(values[1], 4u);
    let tmp_207 = subgroupShuffleXor(keys[2], 4u);
    let tmp_208 = subgroupShuffleXor(values[2], 4u);
    let tmp_209 = subgroupShuffleXor(keys[3], 4u);
    let tmp_210 = subgroupShuffleXor(values[3], 4u);
    let tmp_211 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_212 = keys[0] < tmp_203 || (keys[0] == tmp_203 && values[0] < tmp_204);
    if tmp_211 == tmp_212 { keys[0] = tmp_203; values[0] = tmp_204; }
    let tmp_213 = keys[1] < tmp_205 || (keys[1] == tmp_205 && values[1] < tmp_206);
    if tmp_211 == tmp_213 { keys[1] = tmp_205; values[1] = tmp_206; }
    let tmp_214 = keys[2] < tmp_207 || (keys[2] == tmp_207 && values[2] < tmp_208);
    if tmp_211 == tmp_214 { keys[2] = tmp_207; values[2] = tmp_208; }
    let tmp_215 = keys[3] < tmp_209 || (keys[3] == tmp_209 && values[3] < tmp_210);
    if tmp_211 == tmp_215 { keys[3] = tmp_209; values[3] = tmp_210; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:4) 
    {
    let tmp_216 = subgroupShuffleXor(keys[0], 2u);
    let tmp_217 = subgroupShuffleXor(values[0], 2u);
    let tmp_218 = subgroupShuffleXor(keys[1], 2u);
    let tmp_219 = subgroupShuffleXor(values[1], 2u);
    let tmp_220 = subgroupShuffleXor(keys[2], 2u);
    let tmp_221 = subgroupShuffleXor(values[2], 2u);
    let tmp_222 = subgroupShuffleXor(keys[3], 2u);
    let tmp_223 = subgroupShuffleXor(values[3], 2u);
    let tmp_224 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_225 = keys[0] < tmp_216 || (keys[0] == tmp_216 && values[0] < tmp_217);
    if tmp_224 == tmp_225 { keys[0] = tmp_216; values[0] = tmp_217; }
    let tmp_226 = keys[1] < tmp_218 || (keys[1] == tmp_218 && values[1] < tmp_219);
    if tmp_224 == tmp_226 { keys[1] = tmp_218; values[1] = tmp_219; }
    let tmp_227 = keys[2] < tmp_220 || (keys[2] == tmp_220 && values[2] < tmp_221);
    if tmp_224 == tmp_227 { keys[2] = tmp_220; values[2] = tmp_221; }
    let tmp_228 = keys[3] < tmp_222 || (keys[3] == tmp_222 && values[3] < tmp_223);
    if tmp_224 == tmp_228 { keys[3] = tmp_222; values[3] = tmp_223; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:4) 
    {
    let tmp_229 = subgroupShuffleXor(keys[0], 1u);
    let tmp_230 = subgroupShuffleXor(values[0], 1u);
    let tmp_231 = subgroupShuffleXor(keys[1], 1u);
    let tmp_232 = subgroupShuffleXor(values[1], 1u);
    let tmp_233 = subgroupShuffleXor(keys[2], 1u);
    let tmp_234 = subgroupShuffleXor(values[2], 1u);
    let tmp_235 = subgroupShuffleXor(keys[3], 1u);
    let tmp_236 = subgroupShuffleXor(values[3], 1u);
    let tmp_237 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_238 = keys[0] < tmp_229 || (keys[0] == tmp_229 && values[0] < tmp_230);
    if tmp_237 == tmp_238 { keys[0] = tmp_229; values[0] = tmp_230; }
    let tmp_239 = keys[1] < tmp_231 || (keys[1] == tmp_231 && values[1] < tmp_232);
    if tmp_237 == tmp_239 { keys[1] = tmp_231; values[1] = tmp_232; }
    let tmp_240 = keys[2] < tmp_233 || (keys[2] == tmp_233 && values[2] < tmp_234);
    if tmp_237 == tmp_240 { keys[2] = tmp_233; values[2] = tmp_234; }
    let tmp_241 = keys[3] < tmp_235 || (keys[3] == tmp_235 && values[3] < tmp_236);
    if tmp_237 == tmp_241 { keys[3] = tmp_235; values[3] = tmp_236; }
    }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_242 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_242;let tmp_243 = values[0]; values[0] = values[2]; values[2] = tmp_243; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_244 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_244;let tmp_245 = values[1]; values[1] = values[3]; values[3] = tmp_245; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_246 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_246;let tmp_247 = values[0]; values[0] = values[1]; values[1] = tmp_247; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_248 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_248;let tmp_249 = values[2]; values[2] = values[3]; values[3] = tmp_249; }
    }
    // exch_intxn(tmask:63,swbit:5,wpt:4)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_250 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_251 = seg_base + (local_tid ^ 63u); let tmp_252 = smem_keys[tmp_251 * WPT + 3u]; let tmp_253 = smem_vals[tmp_251 * WPT + 3u]; let tmp_254 = keys[0] < tmp_252 || (keys[0] == tmp_252 && values[0] < tmp_253); if tmp_250 == tmp_254 { keys[0] = tmp_252; values[0] = tmp_253; } let tmp_255 = smem_keys[tmp_251 * WPT + 2u]; let tmp_256 = smem_vals[tmp_251 * WPT + 2u]; let tmp_257 = keys[1] < tmp_255 || (keys[1] == tmp_255 && values[1] < tmp_256); if tmp_250 == tmp_257 { keys[1] = tmp_255; values[1] = tmp_256; } let tmp_258 = smem_keys[tmp_251 * WPT + 1u]; let tmp_259 = smem_vals[tmp_251 * WPT + 1u]; let tmp_260 = keys[2] < tmp_258 || (keys[2] == tmp_258 && values[2] < tmp_259); if tmp_250 == tmp_260 { keys[2] = tmp_258; values[2] = tmp_259; } let tmp_261 = smem_keys[tmp_251 * WPT + 0u]; let tmp_262 = smem_vals[tmp_251 * WPT + 0u]; let tmp_263 = keys[3] < tmp_261 || (keys[3] == tmp_261 && values[3] < tmp_262); if tmp_250 == tmp_263 { keys[3] = tmp_261; values[3] = tmp_262; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_264 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_265 = seg_base + (local_tid ^ 16u); let tmp_266 = smem_keys[tmp_265 * WPT + 0u]; let tmp_267 = smem_vals[tmp_265 * WPT + 0u]; let tmp_268 = keys[0] < tmp_266 || (keys[0] == tmp_266 && values[0] < tmp_267); if tmp_264 == tmp_268 { keys[0] = tmp_266; values[0] = tmp_267; } let tmp_269 = smem_keys[tmp_265 * WPT + 1u]; let tmp_270 = smem_vals[tmp_265 * WPT + 1u]; let tmp_271 = keys[1] < tmp_269 || (keys[1] == tmp_269 && values[1] < tmp_270); if tmp_264 == tmp_271 { keys[1] = tmp_269; values[1] = tmp_270; } let tmp_272 = smem_keys[tmp_265 * WPT + 2u]; let tmp_273 = smem_vals[tmp_265 * WPT + 2u]; let tmp_274 = keys[2] < tmp_272 || (keys[2] == tmp_272 && values[2] < tmp_273); if tmp_264 == tmp_274 { keys[2] = tmp_272; values[2] = tmp_273; } let tmp_275 = smem_keys[tmp_265 * WPT + 3u]; let tmp_276 = smem_vals[tmp_265 * WPT + 3u]; let tmp_277 = keys[3] < tmp_275 || (keys[3] == tmp_275 && values[3] < tmp_276); if tmp_264 == tmp_277 { keys[3] = tmp_275; values[3] = tmp_276; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_278 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_279 = seg_base + (local_tid ^ 8u); let tmp_280 = smem_keys[tmp_279 * WPT + 0u]; let tmp_281 = smem_vals[tmp_279 * WPT + 0u]; let tmp_282 = keys[0] < tmp_280 || (keys[0] == tmp_280 && values[0] < tmp_281); if tmp_278 == tmp_282 { keys[0] = tmp_280; values[0] = tmp_281; } let tmp_283 = smem_keys[tmp_279 * WPT + 1u]; let tmp_284 = smem_vals[tmp_279 * WPT + 1u]; let tmp_285 = keys[1] < tmp_283 || (keys[1] == tmp_283 && values[1] < tmp_284); if tmp_278 == tmp_285 { keys[1] = tmp_283; values[1] = tmp_284; } let tmp_286 = smem_keys[tmp_279 * WPT + 2u]; let tmp_287 = smem_vals[tmp_279 * WPT + 2u]; let tmp_288 = keys[2] < tmp_286 || (keys[2] == tmp_286 && values[2] < tmp_287); if tmp_278 == tmp_288 { keys[2] = tmp_286; values[2] = tmp_287; } let tmp_289 = smem_keys[tmp_279 * WPT + 3u]; let tmp_290 = smem_vals[tmp_279 * WPT + 3u]; let tmp_291 = keys[3] < tmp_289 || (keys[3] == tmp_289 && values[3] < tmp_290); if tmp_278 == tmp_291 { keys[3] = tmp_289; values[3] = tmp_290; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:4) 
    {
    let tmp_292 = subgroupShuffleXor(keys[0], 4u);
    let tmp_293 = subgroupShuffleXor(values[0], 4u);
    let tmp_294 = subgroupShuffleXor(keys[1], 4u);
    let tmp_295 = subgroupShuffleXor(values[1], 4u);
    let tmp_296 = subgroupShuffleXor(keys[2], 4u);
    let tmp_297 = subgroupShuffleXor(values[2], 4u);
    let tmp_298 = subgroupShuffleXor(keys[3], 4u);
    let tmp_299 = subgroupShuffleXor(values[3], 4u);
    let tmp_300 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_301 = keys[0] < tmp_292 || (keys[0] == tmp_292 && values[0] < tmp_293);
    if tmp_300 == tmp_301 { keys[0] = tmp_292; values[0] = tmp_293; }
    let tmp_302 = keys[1] < tmp_294 || (keys[1] == tmp_294 && values[1] < tmp_295);
    if tmp_300 == tmp_302 { keys[1] = tmp_294; values[1] = tmp_295; }
    let tmp_303 = keys[2] < tmp_296 || (keys[2] == tmp_296 && values[2] < tmp_297);
    if tmp_300 == tmp_303 { keys[2] = tmp_296; values[2] = tmp_297; }
    let tmp_304 = keys[3] < tmp_298 || (keys[3] == tmp_298 && values[3] < tmp_299);
    if tmp_300 == tmp_304 { keys[3] = tmp_298; values[3] = tmp_299; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:4) 
    {
    let tmp_305 = subgroupShuffleXor(keys[0], 2u);
    let tmp_306 = subgroupShuffleXor(values[0], 2u);
    let tmp_307 = subgroupShuffleXor(keys[1], 2u);
    let tmp_308 = subgroupShuffleXor(values[1], 2u);
    let tmp_309 = subgroupShuffleXor(keys[2], 2u);
    let tmp_310 = subgroupShuffleXor(values[2], 2u);
    let tmp_311 = subgroupShuffleXor(keys[3], 2u);
    let tmp_312 = subgroupShuffleXor(values[3], 2u);
    let tmp_313 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_314 = keys[0] < tmp_305 || (keys[0] == tmp_305 && values[0] < tmp_306);
    if tmp_313 == tmp_314 { keys[0] = tmp_305; values[0] = tmp_306; }
    let tmp_315 = keys[1] < tmp_307 || (keys[1] == tmp_307 && values[1] < tmp_308);
    if tmp_313 == tmp_315 { keys[1] = tmp_307; values[1] = tmp_308; }
    let tmp_316 = keys[2] < tmp_309 || (keys[2] == tmp_309 && values[2] < tmp_310);
    if tmp_313 == tmp_316 { keys[2] = tmp_309; values[2] = tmp_310; }
    let tmp_317 = keys[3] < tmp_311 || (keys[3] == tmp_311 && values[3] < tmp_312);
    if tmp_313 == tmp_317 { keys[3] = tmp_311; values[3] = tmp_312; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:4) 
    {
    let tmp_318 = subgroupShuffleXor(keys[0], 1u);
    let tmp_319 = subgroupShuffleXor(values[0], 1u);
    let tmp_320 = subgroupShuffleXor(keys[1], 1u);
    let tmp_321 = subgroupShuffleXor(values[1], 1u);
    let tmp_322 = subgroupShuffleXor(keys[2], 1u);
    let tmp_323 = subgroupShuffleXor(values[2], 1u);
    let tmp_324 = subgroupShuffleXor(keys[3], 1u);
    let tmp_325 = subgroupShuffleXor(values[3], 1u);
    let tmp_326 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_327 = keys[0] < tmp_318 || (keys[0] == tmp_318 && values[0] < tmp_319);
    if tmp_326 == tmp_327 { keys[0] = tmp_318; values[0] = tmp_319; }
    let tmp_328 = keys[1] < tmp_320 || (keys[1] == tmp_320 && values[1] < tmp_321);
    if tmp_326 == tmp_328 { keys[1] = tmp_320; values[1] = tmp_321; }
    let tmp_329 = keys[2] < tmp_322 || (keys[2] == tmp_322 && values[2] < tmp_323);
    if tmp_326 == tmp_329 { keys[2] = tmp_322; values[2] = tmp_323; }
    let tmp_330 = keys[3] < tmp_324 || (keys[3] == tmp_324 && values[3] < tmp_325);
    if tmp_326 == tmp_330 { keys[3] = tmp_324; values[3] = tmp_325; }
    }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_331 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_331;let tmp_332 = values[0]; values[0] = values[2]; values[2] = tmp_332; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_333 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_333;let tmp_334 = values[1]; values[1] = values[3]; values[3] = tmp_334; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_335 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_335;let tmp_336 = values[0]; values[0] = values[1]; values[1] = tmp_336; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_337 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_337;let tmp_338 = values[2]; values[2] = values[3]; values[3] = tmp_338; }
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
