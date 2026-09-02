
enable subgroups;

override WG: u32 = 256u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 1024u;
const M: u32 = 256u;
const WPT: u32 = 4u;
const R: u32 = 16u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybrid_sg16_n1024_m256_block(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 10u;

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
    // exch_paral(tmask:4,swbit:2,wpt:4) 
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
    // exch_paral(tmask:2,swbit:1,wpt:4) 
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
    // exch_paral(tmask:1,swbit:0,wpt:4) 
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
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_166 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_166;let tmp_167 = values[0]; values[0] = values[2]; values[2] = tmp_167; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_168 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_168;let tmp_169 = values[1]; values[1] = values[3]; values[3] = tmp_169; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_170 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_170;let tmp_171 = values[0]; values[0] = values[1]; values[1] = tmp_171; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_172 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_172;let tmp_173 = values[2]; values[2] = values[3]; values[3] = tmp_173; }
    }
    // exch_intxn(tmask:31,swbit:4,wpt:4)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_174 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_175 = seg_base + (local_tid ^ 31u); let tmp_176 = smem_keys[tmp_175 * WPT + 3u]; let tmp_177 = smem_vals[tmp_175 * WPT + 3u]; let tmp_178 = keys[0] < tmp_176 || (keys[0] == tmp_176 && values[0] < tmp_177); if tmp_174 == tmp_178 { keys[0] = tmp_176; values[0] = tmp_177; } let tmp_179 = smem_keys[tmp_175 * WPT + 2u]; let tmp_180 = smem_vals[tmp_175 * WPT + 2u]; let tmp_181 = keys[1] < tmp_179 || (keys[1] == tmp_179 && values[1] < tmp_180); if tmp_174 == tmp_181 { keys[1] = tmp_179; values[1] = tmp_180; } let tmp_182 = smem_keys[tmp_175 * WPT + 1u]; let tmp_183 = smem_vals[tmp_175 * WPT + 1u]; let tmp_184 = keys[2] < tmp_182 || (keys[2] == tmp_182 && values[2] < tmp_183); if tmp_174 == tmp_184 { keys[2] = tmp_182; values[2] = tmp_183; } let tmp_185 = smem_keys[tmp_175 * WPT + 0u]; let tmp_186 = smem_vals[tmp_175 * WPT + 0u]; let tmp_187 = keys[3] < tmp_185 || (keys[3] == tmp_185 && values[3] < tmp_186); if tmp_174 == tmp_187 { keys[3] = tmp_185; values[3] = tmp_186; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:4) 
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
    // exch_paral(tmask:4,swbit:2,wpt:4) 
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
    // exch_paral(tmask:2,swbit:1,wpt:4) 
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
    // exch_paral(tmask:1,swbit:0,wpt:4) 
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
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_240 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_240;let tmp_241 = values[0]; values[0] = values[2]; values[2] = tmp_241; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_242 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_242;let tmp_243 = values[1]; values[1] = values[3]; values[3] = tmp_243; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_244 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_244;let tmp_245 = values[0]; values[0] = values[1]; values[1] = tmp_245; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_246 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_246;let tmp_247 = values[2]; values[2] = values[3]; values[3] = tmp_247; }
    }
    // exch_intxn(tmask:63,swbit:5,wpt:4)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_248 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_249 = seg_base + (local_tid ^ 63u); let tmp_250 = smem_keys[tmp_249 * WPT + 3u]; let tmp_251 = smem_vals[tmp_249 * WPT + 3u]; let tmp_252 = keys[0] < tmp_250 || (keys[0] == tmp_250 && values[0] < tmp_251); if tmp_248 == tmp_252 { keys[0] = tmp_250; values[0] = tmp_251; } let tmp_253 = smem_keys[tmp_249 * WPT + 2u]; let tmp_254 = smem_vals[tmp_249 * WPT + 2u]; let tmp_255 = keys[1] < tmp_253 || (keys[1] == tmp_253 && values[1] < tmp_254); if tmp_248 == tmp_255 { keys[1] = tmp_253; values[1] = tmp_254; } let tmp_256 = smem_keys[tmp_249 * WPT + 1u]; let tmp_257 = smem_vals[tmp_249 * WPT + 1u]; let tmp_258 = keys[2] < tmp_256 || (keys[2] == tmp_256 && values[2] < tmp_257); if tmp_248 == tmp_258 { keys[2] = tmp_256; values[2] = tmp_257; } let tmp_259 = smem_keys[tmp_249 * WPT + 0u]; let tmp_260 = smem_vals[tmp_249 * WPT + 0u]; let tmp_261 = keys[3] < tmp_259 || (keys[3] == tmp_259 && values[3] < tmp_260); if tmp_248 == tmp_261 { keys[3] = tmp_259; values[3] = tmp_260; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_262 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_263 = seg_base + (local_tid ^ 16u); let tmp_264 = smem_keys[tmp_263 * WPT + 0u]; let tmp_265 = smem_vals[tmp_263 * WPT + 0u]; let tmp_266 = keys[0] < tmp_264 || (keys[0] == tmp_264 && values[0] < tmp_265); if tmp_262 == tmp_266 { keys[0] = tmp_264; values[0] = tmp_265; } let tmp_267 = smem_keys[tmp_263 * WPT + 1u]; let tmp_268 = smem_vals[tmp_263 * WPT + 1u]; let tmp_269 = keys[1] < tmp_267 || (keys[1] == tmp_267 && values[1] < tmp_268); if tmp_262 == tmp_269 { keys[1] = tmp_267; values[1] = tmp_268; } let tmp_270 = smem_keys[tmp_263 * WPT + 2u]; let tmp_271 = smem_vals[tmp_263 * WPT + 2u]; let tmp_272 = keys[2] < tmp_270 || (keys[2] == tmp_270 && values[2] < tmp_271); if tmp_262 == tmp_272 { keys[2] = tmp_270; values[2] = tmp_271; } let tmp_273 = smem_keys[tmp_263 * WPT + 3u]; let tmp_274 = smem_vals[tmp_263 * WPT + 3u]; let tmp_275 = keys[3] < tmp_273 || (keys[3] == tmp_273 && values[3] < tmp_274); if tmp_262 == tmp_275 { keys[3] = tmp_273; values[3] = tmp_274; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:4) 
    {
    let tmp_276 = subgroupShuffleXor(keys[0], 8u);
    let tmp_277 = subgroupShuffleXor(values[0], 8u);
    let tmp_278 = subgroupShuffleXor(keys[1], 8u);
    let tmp_279 = subgroupShuffleXor(values[1], 8u);
    let tmp_280 = subgroupShuffleXor(keys[2], 8u);
    let tmp_281 = subgroupShuffleXor(values[2], 8u);
    let tmp_282 = subgroupShuffleXor(keys[3], 8u);
    let tmp_283 = subgroupShuffleXor(values[3], 8u);
    let tmp_284 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_285 = keys[0] < tmp_276 || (keys[0] == tmp_276 && values[0] < tmp_277);
    if tmp_284 == tmp_285 { keys[0] = tmp_276; values[0] = tmp_277; }
    let tmp_286 = keys[1] < tmp_278 || (keys[1] == tmp_278 && values[1] < tmp_279);
    if tmp_284 == tmp_286 { keys[1] = tmp_278; values[1] = tmp_279; }
    let tmp_287 = keys[2] < tmp_280 || (keys[2] == tmp_280 && values[2] < tmp_281);
    if tmp_284 == tmp_287 { keys[2] = tmp_280; values[2] = tmp_281; }
    let tmp_288 = keys[3] < tmp_282 || (keys[3] == tmp_282 && values[3] < tmp_283);
    if tmp_284 == tmp_288 { keys[3] = tmp_282; values[3] = tmp_283; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:4) 
    {
    let tmp_289 = subgroupShuffleXor(keys[0], 4u);
    let tmp_290 = subgroupShuffleXor(values[0], 4u);
    let tmp_291 = subgroupShuffleXor(keys[1], 4u);
    let tmp_292 = subgroupShuffleXor(values[1], 4u);
    let tmp_293 = subgroupShuffleXor(keys[2], 4u);
    let tmp_294 = subgroupShuffleXor(values[2], 4u);
    let tmp_295 = subgroupShuffleXor(keys[3], 4u);
    let tmp_296 = subgroupShuffleXor(values[3], 4u);
    let tmp_297 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_298 = keys[0] < tmp_289 || (keys[0] == tmp_289 && values[0] < tmp_290);
    if tmp_297 == tmp_298 { keys[0] = tmp_289; values[0] = tmp_290; }
    let tmp_299 = keys[1] < tmp_291 || (keys[1] == tmp_291 && values[1] < tmp_292);
    if tmp_297 == tmp_299 { keys[1] = tmp_291; values[1] = tmp_292; }
    let tmp_300 = keys[2] < tmp_293 || (keys[2] == tmp_293 && values[2] < tmp_294);
    if tmp_297 == tmp_300 { keys[2] = tmp_293; values[2] = tmp_294; }
    let tmp_301 = keys[3] < tmp_295 || (keys[3] == tmp_295 && values[3] < tmp_296);
    if tmp_297 == tmp_301 { keys[3] = tmp_295; values[3] = tmp_296; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:4) 
    {
    let tmp_302 = subgroupShuffleXor(keys[0], 2u);
    let tmp_303 = subgroupShuffleXor(values[0], 2u);
    let tmp_304 = subgroupShuffleXor(keys[1], 2u);
    let tmp_305 = subgroupShuffleXor(values[1], 2u);
    let tmp_306 = subgroupShuffleXor(keys[2], 2u);
    let tmp_307 = subgroupShuffleXor(values[2], 2u);
    let tmp_308 = subgroupShuffleXor(keys[3], 2u);
    let tmp_309 = subgroupShuffleXor(values[3], 2u);
    let tmp_310 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_311 = keys[0] < tmp_302 || (keys[0] == tmp_302 && values[0] < tmp_303);
    if tmp_310 == tmp_311 { keys[0] = tmp_302; values[0] = tmp_303; }
    let tmp_312 = keys[1] < tmp_304 || (keys[1] == tmp_304 && values[1] < tmp_305);
    if tmp_310 == tmp_312 { keys[1] = tmp_304; values[1] = tmp_305; }
    let tmp_313 = keys[2] < tmp_306 || (keys[2] == tmp_306 && values[2] < tmp_307);
    if tmp_310 == tmp_313 { keys[2] = tmp_306; values[2] = tmp_307; }
    let tmp_314 = keys[3] < tmp_308 || (keys[3] == tmp_308 && values[3] < tmp_309);
    if tmp_310 == tmp_314 { keys[3] = tmp_308; values[3] = tmp_309; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:4) 
    {
    let tmp_315 = subgroupShuffleXor(keys[0], 1u);
    let tmp_316 = subgroupShuffleXor(values[0], 1u);
    let tmp_317 = subgroupShuffleXor(keys[1], 1u);
    let tmp_318 = subgroupShuffleXor(values[1], 1u);
    let tmp_319 = subgroupShuffleXor(keys[2], 1u);
    let tmp_320 = subgroupShuffleXor(values[2], 1u);
    let tmp_321 = subgroupShuffleXor(keys[3], 1u);
    let tmp_322 = subgroupShuffleXor(values[3], 1u);
    let tmp_323 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_324 = keys[0] < tmp_315 || (keys[0] == tmp_315 && values[0] < tmp_316);
    if tmp_323 == tmp_324 { keys[0] = tmp_315; values[0] = tmp_316; }
    let tmp_325 = keys[1] < tmp_317 || (keys[1] == tmp_317 && values[1] < tmp_318);
    if tmp_323 == tmp_325 { keys[1] = tmp_317; values[1] = tmp_318; }
    let tmp_326 = keys[2] < tmp_319 || (keys[2] == tmp_319 && values[2] < tmp_320);
    if tmp_323 == tmp_326 { keys[2] = tmp_319; values[2] = tmp_320; }
    let tmp_327 = keys[3] < tmp_321 || (keys[3] == tmp_321 && values[3] < tmp_322);
    if tmp_323 == tmp_327 { keys[3] = tmp_321; values[3] = tmp_322; }
    }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_328 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_328;let tmp_329 = values[0]; values[0] = values[2]; values[2] = tmp_329; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_330 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_330;let tmp_331 = values[1]; values[1] = values[3]; values[3] = tmp_331; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_332 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_332;let tmp_333 = values[0]; values[0] = values[1]; values[1] = tmp_333; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_334 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_334;let tmp_335 = values[2]; values[2] = values[3]; values[3] = tmp_335; }
    }
    // exch_intxn(tmask:127,swbit:6,wpt:4)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_336 = extractBits(local_tid, 6u, 1u) != 0u; let tmp_337 = seg_base + (local_tid ^ 127u); let tmp_338 = smem_keys[tmp_337 * WPT + 3u]; let tmp_339 = smem_vals[tmp_337 * WPT + 3u]; let tmp_340 = keys[0] < tmp_338 || (keys[0] == tmp_338 && values[0] < tmp_339); if tmp_336 == tmp_340 { keys[0] = tmp_338; values[0] = tmp_339; } let tmp_341 = smem_keys[tmp_337 * WPT + 2u]; let tmp_342 = smem_vals[tmp_337 * WPT + 2u]; let tmp_343 = keys[1] < tmp_341 || (keys[1] == tmp_341 && values[1] < tmp_342); if tmp_336 == tmp_343 { keys[1] = tmp_341; values[1] = tmp_342; } let tmp_344 = smem_keys[tmp_337 * WPT + 1u]; let tmp_345 = smem_vals[tmp_337 * WPT + 1u]; let tmp_346 = keys[2] < tmp_344 || (keys[2] == tmp_344 && values[2] < tmp_345); if tmp_336 == tmp_346 { keys[2] = tmp_344; values[2] = tmp_345; } let tmp_347 = smem_keys[tmp_337 * WPT + 0u]; let tmp_348 = smem_vals[tmp_337 * WPT + 0u]; let tmp_349 = keys[3] < tmp_347 || (keys[3] == tmp_347 && values[3] < tmp_348); if tmp_336 == tmp_349 { keys[3] = tmp_347; values[3] = tmp_348; } workgroupBarrier(); }
    // exch_paral(tmask:32,swbit:5,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_350 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_351 = seg_base + (local_tid ^ 32u); let tmp_352 = smem_keys[tmp_351 * WPT + 0u]; let tmp_353 = smem_vals[tmp_351 * WPT + 0u]; let tmp_354 = keys[0] < tmp_352 || (keys[0] == tmp_352 && values[0] < tmp_353); if tmp_350 == tmp_354 { keys[0] = tmp_352; values[0] = tmp_353; } let tmp_355 = smem_keys[tmp_351 * WPT + 1u]; let tmp_356 = smem_vals[tmp_351 * WPT + 1u]; let tmp_357 = keys[1] < tmp_355 || (keys[1] == tmp_355 && values[1] < tmp_356); if tmp_350 == tmp_357 { keys[1] = tmp_355; values[1] = tmp_356; } let tmp_358 = smem_keys[tmp_351 * WPT + 2u]; let tmp_359 = smem_vals[tmp_351 * WPT + 2u]; let tmp_360 = keys[2] < tmp_358 || (keys[2] == tmp_358 && values[2] < tmp_359); if tmp_350 == tmp_360 { keys[2] = tmp_358; values[2] = tmp_359; } let tmp_361 = smem_keys[tmp_351 * WPT + 3u]; let tmp_362 = smem_vals[tmp_351 * WPT + 3u]; let tmp_363 = keys[3] < tmp_361 || (keys[3] == tmp_361 && values[3] < tmp_362); if tmp_350 == tmp_363 { keys[3] = tmp_361; values[3] = tmp_362; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_364 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_365 = seg_base + (local_tid ^ 16u); let tmp_366 = smem_keys[tmp_365 * WPT + 0u]; let tmp_367 = smem_vals[tmp_365 * WPT + 0u]; let tmp_368 = keys[0] < tmp_366 || (keys[0] == tmp_366 && values[0] < tmp_367); if tmp_364 == tmp_368 { keys[0] = tmp_366; values[0] = tmp_367; } let tmp_369 = smem_keys[tmp_365 * WPT + 1u]; let tmp_370 = smem_vals[tmp_365 * WPT + 1u]; let tmp_371 = keys[1] < tmp_369 || (keys[1] == tmp_369 && values[1] < tmp_370); if tmp_364 == tmp_371 { keys[1] = tmp_369; values[1] = tmp_370; } let tmp_372 = smem_keys[tmp_365 * WPT + 2u]; let tmp_373 = smem_vals[tmp_365 * WPT + 2u]; let tmp_374 = keys[2] < tmp_372 || (keys[2] == tmp_372 && values[2] < tmp_373); if tmp_364 == tmp_374 { keys[2] = tmp_372; values[2] = tmp_373; } let tmp_375 = smem_keys[tmp_365 * WPT + 3u]; let tmp_376 = smem_vals[tmp_365 * WPT + 3u]; let tmp_377 = keys[3] < tmp_375 || (keys[3] == tmp_375 && values[3] < tmp_376); if tmp_364 == tmp_377 { keys[3] = tmp_375; values[3] = tmp_376; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:4) 
    {
    let tmp_378 = subgroupShuffleXor(keys[0], 8u);
    let tmp_379 = subgroupShuffleXor(values[0], 8u);
    let tmp_380 = subgroupShuffleXor(keys[1], 8u);
    let tmp_381 = subgroupShuffleXor(values[1], 8u);
    let tmp_382 = subgroupShuffleXor(keys[2], 8u);
    let tmp_383 = subgroupShuffleXor(values[2], 8u);
    let tmp_384 = subgroupShuffleXor(keys[3], 8u);
    let tmp_385 = subgroupShuffleXor(values[3], 8u);
    let tmp_386 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_387 = keys[0] < tmp_378 || (keys[0] == tmp_378 && values[0] < tmp_379);
    if tmp_386 == tmp_387 { keys[0] = tmp_378; values[0] = tmp_379; }
    let tmp_388 = keys[1] < tmp_380 || (keys[1] == tmp_380 && values[1] < tmp_381);
    if tmp_386 == tmp_388 { keys[1] = tmp_380; values[1] = tmp_381; }
    let tmp_389 = keys[2] < tmp_382 || (keys[2] == tmp_382 && values[2] < tmp_383);
    if tmp_386 == tmp_389 { keys[2] = tmp_382; values[2] = tmp_383; }
    let tmp_390 = keys[3] < tmp_384 || (keys[3] == tmp_384 && values[3] < tmp_385);
    if tmp_386 == tmp_390 { keys[3] = tmp_384; values[3] = tmp_385; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:4) 
    {
    let tmp_391 = subgroupShuffleXor(keys[0], 4u);
    let tmp_392 = subgroupShuffleXor(values[0], 4u);
    let tmp_393 = subgroupShuffleXor(keys[1], 4u);
    let tmp_394 = subgroupShuffleXor(values[1], 4u);
    let tmp_395 = subgroupShuffleXor(keys[2], 4u);
    let tmp_396 = subgroupShuffleXor(values[2], 4u);
    let tmp_397 = subgroupShuffleXor(keys[3], 4u);
    let tmp_398 = subgroupShuffleXor(values[3], 4u);
    let tmp_399 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_400 = keys[0] < tmp_391 || (keys[0] == tmp_391 && values[0] < tmp_392);
    if tmp_399 == tmp_400 { keys[0] = tmp_391; values[0] = tmp_392; }
    let tmp_401 = keys[1] < tmp_393 || (keys[1] == tmp_393 && values[1] < tmp_394);
    if tmp_399 == tmp_401 { keys[1] = tmp_393; values[1] = tmp_394; }
    let tmp_402 = keys[2] < tmp_395 || (keys[2] == tmp_395 && values[2] < tmp_396);
    if tmp_399 == tmp_402 { keys[2] = tmp_395; values[2] = tmp_396; }
    let tmp_403 = keys[3] < tmp_397 || (keys[3] == tmp_397 && values[3] < tmp_398);
    if tmp_399 == tmp_403 { keys[3] = tmp_397; values[3] = tmp_398; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:4) 
    {
    let tmp_404 = subgroupShuffleXor(keys[0], 2u);
    let tmp_405 = subgroupShuffleXor(values[0], 2u);
    let tmp_406 = subgroupShuffleXor(keys[1], 2u);
    let tmp_407 = subgroupShuffleXor(values[1], 2u);
    let tmp_408 = subgroupShuffleXor(keys[2], 2u);
    let tmp_409 = subgroupShuffleXor(values[2], 2u);
    let tmp_410 = subgroupShuffleXor(keys[3], 2u);
    let tmp_411 = subgroupShuffleXor(values[3], 2u);
    let tmp_412 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_413 = keys[0] < tmp_404 || (keys[0] == tmp_404 && values[0] < tmp_405);
    if tmp_412 == tmp_413 { keys[0] = tmp_404; values[0] = tmp_405; }
    let tmp_414 = keys[1] < tmp_406 || (keys[1] == tmp_406 && values[1] < tmp_407);
    if tmp_412 == tmp_414 { keys[1] = tmp_406; values[1] = tmp_407; }
    let tmp_415 = keys[2] < tmp_408 || (keys[2] == tmp_408 && values[2] < tmp_409);
    if tmp_412 == tmp_415 { keys[2] = tmp_408; values[2] = tmp_409; }
    let tmp_416 = keys[3] < tmp_410 || (keys[3] == tmp_410 && values[3] < tmp_411);
    if tmp_412 == tmp_416 { keys[3] = tmp_410; values[3] = tmp_411; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:4) 
    {
    let tmp_417 = subgroupShuffleXor(keys[0], 1u);
    let tmp_418 = subgroupShuffleXor(values[0], 1u);
    let tmp_419 = subgroupShuffleXor(keys[1], 1u);
    let tmp_420 = subgroupShuffleXor(values[1], 1u);
    let tmp_421 = subgroupShuffleXor(keys[2], 1u);
    let tmp_422 = subgroupShuffleXor(values[2], 1u);
    let tmp_423 = subgroupShuffleXor(keys[3], 1u);
    let tmp_424 = subgroupShuffleXor(values[3], 1u);
    let tmp_425 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_426 = keys[0] < tmp_417 || (keys[0] == tmp_417 && values[0] < tmp_418);
    if tmp_425 == tmp_426 { keys[0] = tmp_417; values[0] = tmp_418; }
    let tmp_427 = keys[1] < tmp_419 || (keys[1] == tmp_419 && values[1] < tmp_420);
    if tmp_425 == tmp_427 { keys[1] = tmp_419; values[1] = tmp_420; }
    let tmp_428 = keys[2] < tmp_421 || (keys[2] == tmp_421 && values[2] < tmp_422);
    if tmp_425 == tmp_428 { keys[2] = tmp_421; values[2] = tmp_422; }
    let tmp_429 = keys[3] < tmp_423 || (keys[3] == tmp_423 && values[3] < tmp_424);
    if tmp_425 == tmp_429 { keys[3] = tmp_423; values[3] = tmp_424; }
    }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_430 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_430;let tmp_431 = values[0]; values[0] = values[2]; values[2] = tmp_431; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_432 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_432;let tmp_433 = values[1]; values[1] = values[3]; values[3] = tmp_433; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_434 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_434;let tmp_435 = values[0]; values[0] = values[1]; values[1] = tmp_435; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_436 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_436;let tmp_437 = values[2]; values[2] = values[3]; values[3] = tmp_437; }
    }
    // exch_intxn(tmask:255,swbit:7,wpt:4)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_438 = extractBits(local_tid, 7u, 1u) != 0u; let tmp_439 = seg_base + (local_tid ^ 255u); let tmp_440 = smem_keys[tmp_439 * WPT + 3u]; let tmp_441 = smem_vals[tmp_439 * WPT + 3u]; let tmp_442 = keys[0] < tmp_440 || (keys[0] == tmp_440 && values[0] < tmp_441); if tmp_438 == tmp_442 { keys[0] = tmp_440; values[0] = tmp_441; } let tmp_443 = smem_keys[tmp_439 * WPT + 2u]; let tmp_444 = smem_vals[tmp_439 * WPT + 2u]; let tmp_445 = keys[1] < tmp_443 || (keys[1] == tmp_443 && values[1] < tmp_444); if tmp_438 == tmp_445 { keys[1] = tmp_443; values[1] = tmp_444; } let tmp_446 = smem_keys[tmp_439 * WPT + 1u]; let tmp_447 = smem_vals[tmp_439 * WPT + 1u]; let tmp_448 = keys[2] < tmp_446 || (keys[2] == tmp_446 && values[2] < tmp_447); if tmp_438 == tmp_448 { keys[2] = tmp_446; values[2] = tmp_447; } let tmp_449 = smem_keys[tmp_439 * WPT + 0u]; let tmp_450 = smem_vals[tmp_439 * WPT + 0u]; let tmp_451 = keys[3] < tmp_449 || (keys[3] == tmp_449 && values[3] < tmp_450); if tmp_438 == tmp_451 { keys[3] = tmp_449; values[3] = tmp_450; } workgroupBarrier(); }
    // exch_paral(tmask:64,swbit:6,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_452 = extractBits(local_tid, 6u, 1u) != 0u; let tmp_453 = seg_base + (local_tid ^ 64u); let tmp_454 = smem_keys[tmp_453 * WPT + 0u]; let tmp_455 = smem_vals[tmp_453 * WPT + 0u]; let tmp_456 = keys[0] < tmp_454 || (keys[0] == tmp_454 && values[0] < tmp_455); if tmp_452 == tmp_456 { keys[0] = tmp_454; values[0] = tmp_455; } let tmp_457 = smem_keys[tmp_453 * WPT + 1u]; let tmp_458 = smem_vals[tmp_453 * WPT + 1u]; let tmp_459 = keys[1] < tmp_457 || (keys[1] == tmp_457 && values[1] < tmp_458); if tmp_452 == tmp_459 { keys[1] = tmp_457; values[1] = tmp_458; } let tmp_460 = smem_keys[tmp_453 * WPT + 2u]; let tmp_461 = smem_vals[tmp_453 * WPT + 2u]; let tmp_462 = keys[2] < tmp_460 || (keys[2] == tmp_460 && values[2] < tmp_461); if tmp_452 == tmp_462 { keys[2] = tmp_460; values[2] = tmp_461; } let tmp_463 = smem_keys[tmp_453 * WPT + 3u]; let tmp_464 = smem_vals[tmp_453 * WPT + 3u]; let tmp_465 = keys[3] < tmp_463 || (keys[3] == tmp_463 && values[3] < tmp_464); if tmp_452 == tmp_465 { keys[3] = tmp_463; values[3] = tmp_464; } workgroupBarrier(); }
    // exch_paral(tmask:32,swbit:5,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_466 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_467 = seg_base + (local_tid ^ 32u); let tmp_468 = smem_keys[tmp_467 * WPT + 0u]; let tmp_469 = smem_vals[tmp_467 * WPT + 0u]; let tmp_470 = keys[0] < tmp_468 || (keys[0] == tmp_468 && values[0] < tmp_469); if tmp_466 == tmp_470 { keys[0] = tmp_468; values[0] = tmp_469; } let tmp_471 = smem_keys[tmp_467 * WPT + 1u]; let tmp_472 = smem_vals[tmp_467 * WPT + 1u]; let tmp_473 = keys[1] < tmp_471 || (keys[1] == tmp_471 && values[1] < tmp_472); if tmp_466 == tmp_473 { keys[1] = tmp_471; values[1] = tmp_472; } let tmp_474 = smem_keys[tmp_467 * WPT + 2u]; let tmp_475 = smem_vals[tmp_467 * WPT + 2u]; let tmp_476 = keys[2] < tmp_474 || (keys[2] == tmp_474 && values[2] < tmp_475); if tmp_466 == tmp_476 { keys[2] = tmp_474; values[2] = tmp_475; } let tmp_477 = smem_keys[tmp_467 * WPT + 3u]; let tmp_478 = smem_vals[tmp_467 * WPT + 3u]; let tmp_479 = keys[3] < tmp_477 || (keys[3] == tmp_477 && values[3] < tmp_478); if tmp_466 == tmp_479 { keys[3] = tmp_477; values[3] = tmp_478; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_480 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_481 = seg_base + (local_tid ^ 16u); let tmp_482 = smem_keys[tmp_481 * WPT + 0u]; let tmp_483 = smem_vals[tmp_481 * WPT + 0u]; let tmp_484 = keys[0] < tmp_482 || (keys[0] == tmp_482 && values[0] < tmp_483); if tmp_480 == tmp_484 { keys[0] = tmp_482; values[0] = tmp_483; } let tmp_485 = smem_keys[tmp_481 * WPT + 1u]; let tmp_486 = smem_vals[tmp_481 * WPT + 1u]; let tmp_487 = keys[1] < tmp_485 || (keys[1] == tmp_485 && values[1] < tmp_486); if tmp_480 == tmp_487 { keys[1] = tmp_485; values[1] = tmp_486; } let tmp_488 = smem_keys[tmp_481 * WPT + 2u]; let tmp_489 = smem_vals[tmp_481 * WPT + 2u]; let tmp_490 = keys[2] < tmp_488 || (keys[2] == tmp_488 && values[2] < tmp_489); if tmp_480 == tmp_490 { keys[2] = tmp_488; values[2] = tmp_489; } let tmp_491 = smem_keys[tmp_481 * WPT + 3u]; let tmp_492 = smem_vals[tmp_481 * WPT + 3u]; let tmp_493 = keys[3] < tmp_491 || (keys[3] == tmp_491 && values[3] < tmp_492); if tmp_480 == tmp_493 { keys[3] = tmp_491; values[3] = tmp_492; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:4) 
    {
    let tmp_494 = subgroupShuffleXor(keys[0], 8u);
    let tmp_495 = subgroupShuffleXor(values[0], 8u);
    let tmp_496 = subgroupShuffleXor(keys[1], 8u);
    let tmp_497 = subgroupShuffleXor(values[1], 8u);
    let tmp_498 = subgroupShuffleXor(keys[2], 8u);
    let tmp_499 = subgroupShuffleXor(values[2], 8u);
    let tmp_500 = subgroupShuffleXor(keys[3], 8u);
    let tmp_501 = subgroupShuffleXor(values[3], 8u);
    let tmp_502 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_503 = keys[0] < tmp_494 || (keys[0] == tmp_494 && values[0] < tmp_495);
    if tmp_502 == tmp_503 { keys[0] = tmp_494; values[0] = tmp_495; }
    let tmp_504 = keys[1] < tmp_496 || (keys[1] == tmp_496 && values[1] < tmp_497);
    if tmp_502 == tmp_504 { keys[1] = tmp_496; values[1] = tmp_497; }
    let tmp_505 = keys[2] < tmp_498 || (keys[2] == tmp_498 && values[2] < tmp_499);
    if tmp_502 == tmp_505 { keys[2] = tmp_498; values[2] = tmp_499; }
    let tmp_506 = keys[3] < tmp_500 || (keys[3] == tmp_500 && values[3] < tmp_501);
    if tmp_502 == tmp_506 { keys[3] = tmp_500; values[3] = tmp_501; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:4) 
    {
    let tmp_507 = subgroupShuffleXor(keys[0], 4u);
    let tmp_508 = subgroupShuffleXor(values[0], 4u);
    let tmp_509 = subgroupShuffleXor(keys[1], 4u);
    let tmp_510 = subgroupShuffleXor(values[1], 4u);
    let tmp_511 = subgroupShuffleXor(keys[2], 4u);
    let tmp_512 = subgroupShuffleXor(values[2], 4u);
    let tmp_513 = subgroupShuffleXor(keys[3], 4u);
    let tmp_514 = subgroupShuffleXor(values[3], 4u);
    let tmp_515 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_516 = keys[0] < tmp_507 || (keys[0] == tmp_507 && values[0] < tmp_508);
    if tmp_515 == tmp_516 { keys[0] = tmp_507; values[0] = tmp_508; }
    let tmp_517 = keys[1] < tmp_509 || (keys[1] == tmp_509 && values[1] < tmp_510);
    if tmp_515 == tmp_517 { keys[1] = tmp_509; values[1] = tmp_510; }
    let tmp_518 = keys[2] < tmp_511 || (keys[2] == tmp_511 && values[2] < tmp_512);
    if tmp_515 == tmp_518 { keys[2] = tmp_511; values[2] = tmp_512; }
    let tmp_519 = keys[3] < tmp_513 || (keys[3] == tmp_513 && values[3] < tmp_514);
    if tmp_515 == tmp_519 { keys[3] = tmp_513; values[3] = tmp_514; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:4) 
    {
    let tmp_520 = subgroupShuffleXor(keys[0], 2u);
    let tmp_521 = subgroupShuffleXor(values[0], 2u);
    let tmp_522 = subgroupShuffleXor(keys[1], 2u);
    let tmp_523 = subgroupShuffleXor(values[1], 2u);
    let tmp_524 = subgroupShuffleXor(keys[2], 2u);
    let tmp_525 = subgroupShuffleXor(values[2], 2u);
    let tmp_526 = subgroupShuffleXor(keys[3], 2u);
    let tmp_527 = subgroupShuffleXor(values[3], 2u);
    let tmp_528 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_529 = keys[0] < tmp_520 || (keys[0] == tmp_520 && values[0] < tmp_521);
    if tmp_528 == tmp_529 { keys[0] = tmp_520; values[0] = tmp_521; }
    let tmp_530 = keys[1] < tmp_522 || (keys[1] == tmp_522 && values[1] < tmp_523);
    if tmp_528 == tmp_530 { keys[1] = tmp_522; values[1] = tmp_523; }
    let tmp_531 = keys[2] < tmp_524 || (keys[2] == tmp_524 && values[2] < tmp_525);
    if tmp_528 == tmp_531 { keys[2] = tmp_524; values[2] = tmp_525; }
    let tmp_532 = keys[3] < tmp_526 || (keys[3] == tmp_526 && values[3] < tmp_527);
    if tmp_528 == tmp_532 { keys[3] = tmp_526; values[3] = tmp_527; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:4) 
    {
    let tmp_533 = subgroupShuffleXor(keys[0], 1u);
    let tmp_534 = subgroupShuffleXor(values[0], 1u);
    let tmp_535 = subgroupShuffleXor(keys[1], 1u);
    let tmp_536 = subgroupShuffleXor(values[1], 1u);
    let tmp_537 = subgroupShuffleXor(keys[2], 1u);
    let tmp_538 = subgroupShuffleXor(values[2], 1u);
    let tmp_539 = subgroupShuffleXor(keys[3], 1u);
    let tmp_540 = subgroupShuffleXor(values[3], 1u);
    let tmp_541 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_542 = keys[0] < tmp_533 || (keys[0] == tmp_533 && values[0] < tmp_534);
    if tmp_541 == tmp_542 { keys[0] = tmp_533; values[0] = tmp_534; }
    let tmp_543 = keys[1] < tmp_535 || (keys[1] == tmp_535 && values[1] < tmp_536);
    if tmp_541 == tmp_543 { keys[1] = tmp_535; values[1] = tmp_536; }
    let tmp_544 = keys[2] < tmp_537 || (keys[2] == tmp_537 && values[2] < tmp_538);
    if tmp_541 == tmp_544 { keys[2] = tmp_537; values[2] = tmp_538; }
    let tmp_545 = keys[3] < tmp_539 || (keys[3] == tmp_539 && values[3] < tmp_540);
    if tmp_541 == tmp_545 { keys[3] = tmp_539; values[3] = tmp_540; }
    }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_546 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_546;let tmp_547 = values[0]; values[0] = values[2]; values[2] = tmp_547; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_548 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_548;let tmp_549 = values[1]; values[1] = values[3]; values[3] = tmp_549; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_550 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_550;let tmp_551 = values[0]; values[0] = values[1]; values[1] = tmp_551; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_552 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_552;let tmp_553 = values[2]; values[2] = values[3]; values[3] = tmp_553; }
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
