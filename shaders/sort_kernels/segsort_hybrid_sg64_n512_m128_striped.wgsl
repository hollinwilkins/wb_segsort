
enable subgroups;

override WG: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 512u;
const M: u32 = 128u;
const WPT: u32 = 4u;
const R: u32 = 64u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybrid_sg64_n512_m128_striped(
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
    {
    let tmp_174 = subgroupShuffleXor(keys[3], 31u);
    let tmp_175 = subgroupShuffleXor(values[3], 31u);
    let tmp_176 = subgroupShuffleXor(keys[2], 31u);
    let tmp_177 = subgroupShuffleXor(values[2], 31u);
    let tmp_178 = subgroupShuffleXor(keys[1], 31u);
    let tmp_179 = subgroupShuffleXor(values[1], 31u);
    let tmp_180 = subgroupShuffleXor(keys[0], 31u);
    let tmp_181 = subgroupShuffleXor(values[0], 31u);
    let tmp_182 = extractBits(local_tid, 4u, 1u) != 0u;
    let tmp_183 = keys[0] < tmp_174 || (keys[0] == tmp_174 && values[0] < tmp_175);
    if tmp_182 == tmp_183 { keys[0] = tmp_174; values[0] = tmp_175; }
    let tmp_184 = keys[1] < tmp_176 || (keys[1] == tmp_176 && values[1] < tmp_177);
    if tmp_182 == tmp_184 { keys[1] = tmp_176; values[1] = tmp_177; }
    let tmp_185 = keys[2] < tmp_178 || (keys[2] == tmp_178 && values[2] < tmp_179);
    if tmp_182 == tmp_185 { keys[2] = tmp_178; values[2] = tmp_179; }
    let tmp_186 = keys[3] < tmp_180 || (keys[3] == tmp_180 && values[3] < tmp_181);
    if tmp_182 == tmp_186 { keys[3] = tmp_180; values[3] = tmp_181; }
    }
    // exch_paral(tmask:8,swbit:3,wpt:4) 
    {
    let tmp_187 = subgroupShuffleXor(keys[0], 8u);
    let tmp_188 = subgroupShuffleXor(values[0], 8u);
    let tmp_189 = subgroupShuffleXor(keys[1], 8u);
    let tmp_190 = subgroupShuffleXor(values[1], 8u);
    let tmp_191 = subgroupShuffleXor(keys[2], 8u);
    let tmp_192 = subgroupShuffleXor(values[2], 8u);
    let tmp_193 = subgroupShuffleXor(keys[3], 8u);
    let tmp_194 = subgroupShuffleXor(values[3], 8u);
    let tmp_195 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_196 = keys[0] < tmp_187 || (keys[0] == tmp_187 && values[0] < tmp_188);
    if tmp_195 == tmp_196 { keys[0] = tmp_187; values[0] = tmp_188; }
    let tmp_197 = keys[1] < tmp_189 || (keys[1] == tmp_189 && values[1] < tmp_190);
    if tmp_195 == tmp_197 { keys[1] = tmp_189; values[1] = tmp_190; }
    let tmp_198 = keys[2] < tmp_191 || (keys[2] == tmp_191 && values[2] < tmp_192);
    if tmp_195 == tmp_198 { keys[2] = tmp_191; values[2] = tmp_192; }
    let tmp_199 = keys[3] < tmp_193 || (keys[3] == tmp_193 && values[3] < tmp_194);
    if tmp_195 == tmp_199 { keys[3] = tmp_193; values[3] = tmp_194; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:4) 
    {
    let tmp_200 = subgroupShuffleXor(keys[0], 4u);
    let tmp_201 = subgroupShuffleXor(values[0], 4u);
    let tmp_202 = subgroupShuffleXor(keys[1], 4u);
    let tmp_203 = subgroupShuffleXor(values[1], 4u);
    let tmp_204 = subgroupShuffleXor(keys[2], 4u);
    let tmp_205 = subgroupShuffleXor(values[2], 4u);
    let tmp_206 = subgroupShuffleXor(keys[3], 4u);
    let tmp_207 = subgroupShuffleXor(values[3], 4u);
    let tmp_208 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_209 = keys[0] < tmp_200 || (keys[0] == tmp_200 && values[0] < tmp_201);
    if tmp_208 == tmp_209 { keys[0] = tmp_200; values[0] = tmp_201; }
    let tmp_210 = keys[1] < tmp_202 || (keys[1] == tmp_202 && values[1] < tmp_203);
    if tmp_208 == tmp_210 { keys[1] = tmp_202; values[1] = tmp_203; }
    let tmp_211 = keys[2] < tmp_204 || (keys[2] == tmp_204 && values[2] < tmp_205);
    if tmp_208 == tmp_211 { keys[2] = tmp_204; values[2] = tmp_205; }
    let tmp_212 = keys[3] < tmp_206 || (keys[3] == tmp_206 && values[3] < tmp_207);
    if tmp_208 == tmp_212 { keys[3] = tmp_206; values[3] = tmp_207; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:4) 
    {
    let tmp_213 = subgroupShuffleXor(keys[0], 2u);
    let tmp_214 = subgroupShuffleXor(values[0], 2u);
    let tmp_215 = subgroupShuffleXor(keys[1], 2u);
    let tmp_216 = subgroupShuffleXor(values[1], 2u);
    let tmp_217 = subgroupShuffleXor(keys[2], 2u);
    let tmp_218 = subgroupShuffleXor(values[2], 2u);
    let tmp_219 = subgroupShuffleXor(keys[3], 2u);
    let tmp_220 = subgroupShuffleXor(values[3], 2u);
    let tmp_221 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_222 = keys[0] < tmp_213 || (keys[0] == tmp_213 && values[0] < tmp_214);
    if tmp_221 == tmp_222 { keys[0] = tmp_213; values[0] = tmp_214; }
    let tmp_223 = keys[1] < tmp_215 || (keys[1] == tmp_215 && values[1] < tmp_216);
    if tmp_221 == tmp_223 { keys[1] = tmp_215; values[1] = tmp_216; }
    let tmp_224 = keys[2] < tmp_217 || (keys[2] == tmp_217 && values[2] < tmp_218);
    if tmp_221 == tmp_224 { keys[2] = tmp_217; values[2] = tmp_218; }
    let tmp_225 = keys[3] < tmp_219 || (keys[3] == tmp_219 && values[3] < tmp_220);
    if tmp_221 == tmp_225 { keys[3] = tmp_219; values[3] = tmp_220; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:4) 
    {
    let tmp_226 = subgroupShuffleXor(keys[0], 1u);
    let tmp_227 = subgroupShuffleXor(values[0], 1u);
    let tmp_228 = subgroupShuffleXor(keys[1], 1u);
    let tmp_229 = subgroupShuffleXor(values[1], 1u);
    let tmp_230 = subgroupShuffleXor(keys[2], 1u);
    let tmp_231 = subgroupShuffleXor(values[2], 1u);
    let tmp_232 = subgroupShuffleXor(keys[3], 1u);
    let tmp_233 = subgroupShuffleXor(values[3], 1u);
    let tmp_234 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_235 = keys[0] < tmp_226 || (keys[0] == tmp_226 && values[0] < tmp_227);
    if tmp_234 == tmp_235 { keys[0] = tmp_226; values[0] = tmp_227; }
    let tmp_236 = keys[1] < tmp_228 || (keys[1] == tmp_228 && values[1] < tmp_229);
    if tmp_234 == tmp_236 { keys[1] = tmp_228; values[1] = tmp_229; }
    let tmp_237 = keys[2] < tmp_230 || (keys[2] == tmp_230 && values[2] < tmp_231);
    if tmp_234 == tmp_237 { keys[2] = tmp_230; values[2] = tmp_231; }
    let tmp_238 = keys[3] < tmp_232 || (keys[3] == tmp_232 && values[3] < tmp_233);
    if tmp_234 == tmp_238 { keys[3] = tmp_232; values[3] = tmp_233; }
    }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_239 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_239;let tmp_240 = values[0]; values[0] = values[2]; values[2] = tmp_240; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_241 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_241;let tmp_242 = values[1]; values[1] = values[3]; values[3] = tmp_242; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_243 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_243;let tmp_244 = values[0]; values[0] = values[1]; values[1] = tmp_244; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_245 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_245;let tmp_246 = values[2]; values[2] = values[3]; values[3] = tmp_246; }
    }
    // exch_intxn(tmask:63,swbit:5,wpt:4)
    {
    let tmp_247 = subgroupShuffleXor(keys[3], 63u);
    let tmp_248 = subgroupShuffleXor(values[3], 63u);
    let tmp_249 = subgroupShuffleXor(keys[2], 63u);
    let tmp_250 = subgroupShuffleXor(values[2], 63u);
    let tmp_251 = subgroupShuffleXor(keys[1], 63u);
    let tmp_252 = subgroupShuffleXor(values[1], 63u);
    let tmp_253 = subgroupShuffleXor(keys[0], 63u);
    let tmp_254 = subgroupShuffleXor(values[0], 63u);
    let tmp_255 = extractBits(local_tid, 5u, 1u) != 0u;
    let tmp_256 = keys[0] < tmp_247 || (keys[0] == tmp_247 && values[0] < tmp_248);
    if tmp_255 == tmp_256 { keys[0] = tmp_247; values[0] = tmp_248; }
    let tmp_257 = keys[1] < tmp_249 || (keys[1] == tmp_249 && values[1] < tmp_250);
    if tmp_255 == tmp_257 { keys[1] = tmp_249; values[1] = tmp_250; }
    let tmp_258 = keys[2] < tmp_251 || (keys[2] == tmp_251 && values[2] < tmp_252);
    if tmp_255 == tmp_258 { keys[2] = tmp_251; values[2] = tmp_252; }
    let tmp_259 = keys[3] < tmp_253 || (keys[3] == tmp_253 && values[3] < tmp_254);
    if tmp_255 == tmp_259 { keys[3] = tmp_253; values[3] = tmp_254; }
    }
    // exch_paral(tmask:16,swbit:4,wpt:4) 
    {
    let tmp_260 = subgroupShuffleXor(keys[0], 16u);
    let tmp_261 = subgroupShuffleXor(values[0], 16u);
    let tmp_262 = subgroupShuffleXor(keys[1], 16u);
    let tmp_263 = subgroupShuffleXor(values[1], 16u);
    let tmp_264 = subgroupShuffleXor(keys[2], 16u);
    let tmp_265 = subgroupShuffleXor(values[2], 16u);
    let tmp_266 = subgroupShuffleXor(keys[3], 16u);
    let tmp_267 = subgroupShuffleXor(values[3], 16u);
    let tmp_268 = extractBits(local_tid, 4u, 1u) != 0u;
    let tmp_269 = keys[0] < tmp_260 || (keys[0] == tmp_260 && values[0] < tmp_261);
    if tmp_268 == tmp_269 { keys[0] = tmp_260; values[0] = tmp_261; }
    let tmp_270 = keys[1] < tmp_262 || (keys[1] == tmp_262 && values[1] < tmp_263);
    if tmp_268 == tmp_270 { keys[1] = tmp_262; values[1] = tmp_263; }
    let tmp_271 = keys[2] < tmp_264 || (keys[2] == tmp_264 && values[2] < tmp_265);
    if tmp_268 == tmp_271 { keys[2] = tmp_264; values[2] = tmp_265; }
    let tmp_272 = keys[3] < tmp_266 || (keys[3] == tmp_266 && values[3] < tmp_267);
    if tmp_268 == tmp_272 { keys[3] = tmp_266; values[3] = tmp_267; }
    }
    // exch_paral(tmask:8,swbit:3,wpt:4) 
    {
    let tmp_273 = subgroupShuffleXor(keys[0], 8u);
    let tmp_274 = subgroupShuffleXor(values[0], 8u);
    let tmp_275 = subgroupShuffleXor(keys[1], 8u);
    let tmp_276 = subgroupShuffleXor(values[1], 8u);
    let tmp_277 = subgroupShuffleXor(keys[2], 8u);
    let tmp_278 = subgroupShuffleXor(values[2], 8u);
    let tmp_279 = subgroupShuffleXor(keys[3], 8u);
    let tmp_280 = subgroupShuffleXor(values[3], 8u);
    let tmp_281 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_282 = keys[0] < tmp_273 || (keys[0] == tmp_273 && values[0] < tmp_274);
    if tmp_281 == tmp_282 { keys[0] = tmp_273; values[0] = tmp_274; }
    let tmp_283 = keys[1] < tmp_275 || (keys[1] == tmp_275 && values[1] < tmp_276);
    if tmp_281 == tmp_283 { keys[1] = tmp_275; values[1] = tmp_276; }
    let tmp_284 = keys[2] < tmp_277 || (keys[2] == tmp_277 && values[2] < tmp_278);
    if tmp_281 == tmp_284 { keys[2] = tmp_277; values[2] = tmp_278; }
    let tmp_285 = keys[3] < tmp_279 || (keys[3] == tmp_279 && values[3] < tmp_280);
    if tmp_281 == tmp_285 { keys[3] = tmp_279; values[3] = tmp_280; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:4) 
    {
    let tmp_286 = subgroupShuffleXor(keys[0], 4u);
    let tmp_287 = subgroupShuffleXor(values[0], 4u);
    let tmp_288 = subgroupShuffleXor(keys[1], 4u);
    let tmp_289 = subgroupShuffleXor(values[1], 4u);
    let tmp_290 = subgroupShuffleXor(keys[2], 4u);
    let tmp_291 = subgroupShuffleXor(values[2], 4u);
    let tmp_292 = subgroupShuffleXor(keys[3], 4u);
    let tmp_293 = subgroupShuffleXor(values[3], 4u);
    let tmp_294 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_295 = keys[0] < tmp_286 || (keys[0] == tmp_286 && values[0] < tmp_287);
    if tmp_294 == tmp_295 { keys[0] = tmp_286; values[0] = tmp_287; }
    let tmp_296 = keys[1] < tmp_288 || (keys[1] == tmp_288 && values[1] < tmp_289);
    if tmp_294 == tmp_296 { keys[1] = tmp_288; values[1] = tmp_289; }
    let tmp_297 = keys[2] < tmp_290 || (keys[2] == tmp_290 && values[2] < tmp_291);
    if tmp_294 == tmp_297 { keys[2] = tmp_290; values[2] = tmp_291; }
    let tmp_298 = keys[3] < tmp_292 || (keys[3] == tmp_292 && values[3] < tmp_293);
    if tmp_294 == tmp_298 { keys[3] = tmp_292; values[3] = tmp_293; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:4) 
    {
    let tmp_299 = subgroupShuffleXor(keys[0], 2u);
    let tmp_300 = subgroupShuffleXor(values[0], 2u);
    let tmp_301 = subgroupShuffleXor(keys[1], 2u);
    let tmp_302 = subgroupShuffleXor(values[1], 2u);
    let tmp_303 = subgroupShuffleXor(keys[2], 2u);
    let tmp_304 = subgroupShuffleXor(values[2], 2u);
    let tmp_305 = subgroupShuffleXor(keys[3], 2u);
    let tmp_306 = subgroupShuffleXor(values[3], 2u);
    let tmp_307 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_308 = keys[0] < tmp_299 || (keys[0] == tmp_299 && values[0] < tmp_300);
    if tmp_307 == tmp_308 { keys[0] = tmp_299; values[0] = tmp_300; }
    let tmp_309 = keys[1] < tmp_301 || (keys[1] == tmp_301 && values[1] < tmp_302);
    if tmp_307 == tmp_309 { keys[1] = tmp_301; values[1] = tmp_302; }
    let tmp_310 = keys[2] < tmp_303 || (keys[2] == tmp_303 && values[2] < tmp_304);
    if tmp_307 == tmp_310 { keys[2] = tmp_303; values[2] = tmp_304; }
    let tmp_311 = keys[3] < tmp_305 || (keys[3] == tmp_305 && values[3] < tmp_306);
    if tmp_307 == tmp_311 { keys[3] = tmp_305; values[3] = tmp_306; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:4) 
    {
    let tmp_312 = subgroupShuffleXor(keys[0], 1u);
    let tmp_313 = subgroupShuffleXor(values[0], 1u);
    let tmp_314 = subgroupShuffleXor(keys[1], 1u);
    let tmp_315 = subgroupShuffleXor(values[1], 1u);
    let tmp_316 = subgroupShuffleXor(keys[2], 1u);
    let tmp_317 = subgroupShuffleXor(values[2], 1u);
    let tmp_318 = subgroupShuffleXor(keys[3], 1u);
    let tmp_319 = subgroupShuffleXor(values[3], 1u);
    let tmp_320 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_321 = keys[0] < tmp_312 || (keys[0] == tmp_312 && values[0] < tmp_313);
    if tmp_320 == tmp_321 { keys[0] = tmp_312; values[0] = tmp_313; }
    let tmp_322 = keys[1] < tmp_314 || (keys[1] == tmp_314 && values[1] < tmp_315);
    if tmp_320 == tmp_322 { keys[1] = tmp_314; values[1] = tmp_315; }
    let tmp_323 = keys[2] < tmp_316 || (keys[2] == tmp_316 && values[2] < tmp_317);
    if tmp_320 == tmp_323 { keys[2] = tmp_316; values[2] = tmp_317; }
    let tmp_324 = keys[3] < tmp_318 || (keys[3] == tmp_318 && values[3] < tmp_319);
    if tmp_320 == tmp_324 { keys[3] = tmp_318; values[3] = tmp_319; }
    }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_325 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_325;let tmp_326 = values[0]; values[0] = values[2]; values[2] = tmp_326; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_327 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_327;let tmp_328 = values[1]; values[1] = values[3]; values[3] = tmp_328; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_329 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_329;let tmp_330 = values[0]; values[0] = values[1]; values[1] = tmp_330; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_331 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_331;let tmp_332 = values[2]; values[2] = values[3]; values[3] = tmp_332; }
    }
    // exch_intxn(tmask:127,swbit:6,wpt:4)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_333 = extractBits(local_tid, 6u, 1u) != 0u; let tmp_334 = seg_base + (local_tid ^ 127u); let tmp_335 = smem_keys[tmp_334 * WPT + 3u]; let tmp_336 = smem_vals[tmp_334 * WPT + 3u]; let tmp_337 = keys[0] < tmp_335 || (keys[0] == tmp_335 && values[0] < tmp_336); if tmp_333 == tmp_337 { keys[0] = tmp_335; values[0] = tmp_336; } let tmp_338 = smem_keys[tmp_334 * WPT + 2u]; let tmp_339 = smem_vals[tmp_334 * WPT + 2u]; let tmp_340 = keys[1] < tmp_338 || (keys[1] == tmp_338 && values[1] < tmp_339); if tmp_333 == tmp_340 { keys[1] = tmp_338; values[1] = tmp_339; } let tmp_341 = smem_keys[tmp_334 * WPT + 1u]; let tmp_342 = smem_vals[tmp_334 * WPT + 1u]; let tmp_343 = keys[2] < tmp_341 || (keys[2] == tmp_341 && values[2] < tmp_342); if tmp_333 == tmp_343 { keys[2] = tmp_341; values[2] = tmp_342; } let tmp_344 = smem_keys[tmp_334 * WPT + 0u]; let tmp_345 = smem_vals[tmp_334 * WPT + 0u]; let tmp_346 = keys[3] < tmp_344 || (keys[3] == tmp_344 && values[3] < tmp_345); if tmp_333 == tmp_346 { keys[3] = tmp_344; values[3] = tmp_345; } workgroupBarrier(); }
    // exch_paral(tmask:32,swbit:5,wpt:4) 
    {
    let tmp_347 = subgroupShuffleXor(keys[0], 32u);
    let tmp_348 = subgroupShuffleXor(values[0], 32u);
    let tmp_349 = subgroupShuffleXor(keys[1], 32u);
    let tmp_350 = subgroupShuffleXor(values[1], 32u);
    let tmp_351 = subgroupShuffleXor(keys[2], 32u);
    let tmp_352 = subgroupShuffleXor(values[2], 32u);
    let tmp_353 = subgroupShuffleXor(keys[3], 32u);
    let tmp_354 = subgroupShuffleXor(values[3], 32u);
    let tmp_355 = extractBits(local_tid, 5u, 1u) != 0u;
    let tmp_356 = keys[0] < tmp_347 || (keys[0] == tmp_347 && values[0] < tmp_348);
    if tmp_355 == tmp_356 { keys[0] = tmp_347; values[0] = tmp_348; }
    let tmp_357 = keys[1] < tmp_349 || (keys[1] == tmp_349 && values[1] < tmp_350);
    if tmp_355 == tmp_357 { keys[1] = tmp_349; values[1] = tmp_350; }
    let tmp_358 = keys[2] < tmp_351 || (keys[2] == tmp_351 && values[2] < tmp_352);
    if tmp_355 == tmp_358 { keys[2] = tmp_351; values[2] = tmp_352; }
    let tmp_359 = keys[3] < tmp_353 || (keys[3] == tmp_353 && values[3] < tmp_354);
    if tmp_355 == tmp_359 { keys[3] = tmp_353; values[3] = tmp_354; }
    }
    // exch_paral(tmask:16,swbit:4,wpt:4) 
    {
    let tmp_360 = subgroupShuffleXor(keys[0], 16u);
    let tmp_361 = subgroupShuffleXor(values[0], 16u);
    let tmp_362 = subgroupShuffleXor(keys[1], 16u);
    let tmp_363 = subgroupShuffleXor(values[1], 16u);
    let tmp_364 = subgroupShuffleXor(keys[2], 16u);
    let tmp_365 = subgroupShuffleXor(values[2], 16u);
    let tmp_366 = subgroupShuffleXor(keys[3], 16u);
    let tmp_367 = subgroupShuffleXor(values[3], 16u);
    let tmp_368 = extractBits(local_tid, 4u, 1u) != 0u;
    let tmp_369 = keys[0] < tmp_360 || (keys[0] == tmp_360 && values[0] < tmp_361);
    if tmp_368 == tmp_369 { keys[0] = tmp_360; values[0] = tmp_361; }
    let tmp_370 = keys[1] < tmp_362 || (keys[1] == tmp_362 && values[1] < tmp_363);
    if tmp_368 == tmp_370 { keys[1] = tmp_362; values[1] = tmp_363; }
    let tmp_371 = keys[2] < tmp_364 || (keys[2] == tmp_364 && values[2] < tmp_365);
    if tmp_368 == tmp_371 { keys[2] = tmp_364; values[2] = tmp_365; }
    let tmp_372 = keys[3] < tmp_366 || (keys[3] == tmp_366 && values[3] < tmp_367);
    if tmp_368 == tmp_372 { keys[3] = tmp_366; values[3] = tmp_367; }
    }
    // exch_paral(tmask:8,swbit:3,wpt:4) 
    {
    let tmp_373 = subgroupShuffleXor(keys[0], 8u);
    let tmp_374 = subgroupShuffleXor(values[0], 8u);
    let tmp_375 = subgroupShuffleXor(keys[1], 8u);
    let tmp_376 = subgroupShuffleXor(values[1], 8u);
    let tmp_377 = subgroupShuffleXor(keys[2], 8u);
    let tmp_378 = subgroupShuffleXor(values[2], 8u);
    let tmp_379 = subgroupShuffleXor(keys[3], 8u);
    let tmp_380 = subgroupShuffleXor(values[3], 8u);
    let tmp_381 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_382 = keys[0] < tmp_373 || (keys[0] == tmp_373 && values[0] < tmp_374);
    if tmp_381 == tmp_382 { keys[0] = tmp_373; values[0] = tmp_374; }
    let tmp_383 = keys[1] < tmp_375 || (keys[1] == tmp_375 && values[1] < tmp_376);
    if tmp_381 == tmp_383 { keys[1] = tmp_375; values[1] = tmp_376; }
    let tmp_384 = keys[2] < tmp_377 || (keys[2] == tmp_377 && values[2] < tmp_378);
    if tmp_381 == tmp_384 { keys[2] = tmp_377; values[2] = tmp_378; }
    let tmp_385 = keys[3] < tmp_379 || (keys[3] == tmp_379 && values[3] < tmp_380);
    if tmp_381 == tmp_385 { keys[3] = tmp_379; values[3] = tmp_380; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:4) 
    {
    let tmp_386 = subgroupShuffleXor(keys[0], 4u);
    let tmp_387 = subgroupShuffleXor(values[0], 4u);
    let tmp_388 = subgroupShuffleXor(keys[1], 4u);
    let tmp_389 = subgroupShuffleXor(values[1], 4u);
    let tmp_390 = subgroupShuffleXor(keys[2], 4u);
    let tmp_391 = subgroupShuffleXor(values[2], 4u);
    let tmp_392 = subgroupShuffleXor(keys[3], 4u);
    let tmp_393 = subgroupShuffleXor(values[3], 4u);
    let tmp_394 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_395 = keys[0] < tmp_386 || (keys[0] == tmp_386 && values[0] < tmp_387);
    if tmp_394 == tmp_395 { keys[0] = tmp_386; values[0] = tmp_387; }
    let tmp_396 = keys[1] < tmp_388 || (keys[1] == tmp_388 && values[1] < tmp_389);
    if tmp_394 == tmp_396 { keys[1] = tmp_388; values[1] = tmp_389; }
    let tmp_397 = keys[2] < tmp_390 || (keys[2] == tmp_390 && values[2] < tmp_391);
    if tmp_394 == tmp_397 { keys[2] = tmp_390; values[2] = tmp_391; }
    let tmp_398 = keys[3] < tmp_392 || (keys[3] == tmp_392 && values[3] < tmp_393);
    if tmp_394 == tmp_398 { keys[3] = tmp_392; values[3] = tmp_393; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:4) 
    {
    let tmp_399 = subgroupShuffleXor(keys[0], 2u);
    let tmp_400 = subgroupShuffleXor(values[0], 2u);
    let tmp_401 = subgroupShuffleXor(keys[1], 2u);
    let tmp_402 = subgroupShuffleXor(values[1], 2u);
    let tmp_403 = subgroupShuffleXor(keys[2], 2u);
    let tmp_404 = subgroupShuffleXor(values[2], 2u);
    let tmp_405 = subgroupShuffleXor(keys[3], 2u);
    let tmp_406 = subgroupShuffleXor(values[3], 2u);
    let tmp_407 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_408 = keys[0] < tmp_399 || (keys[0] == tmp_399 && values[0] < tmp_400);
    if tmp_407 == tmp_408 { keys[0] = tmp_399; values[0] = tmp_400; }
    let tmp_409 = keys[1] < tmp_401 || (keys[1] == tmp_401 && values[1] < tmp_402);
    if tmp_407 == tmp_409 { keys[1] = tmp_401; values[1] = tmp_402; }
    let tmp_410 = keys[2] < tmp_403 || (keys[2] == tmp_403 && values[2] < tmp_404);
    if tmp_407 == tmp_410 { keys[2] = tmp_403; values[2] = tmp_404; }
    let tmp_411 = keys[3] < tmp_405 || (keys[3] == tmp_405 && values[3] < tmp_406);
    if tmp_407 == tmp_411 { keys[3] = tmp_405; values[3] = tmp_406; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:4) 
    {
    let tmp_412 = subgroupShuffleXor(keys[0], 1u);
    let tmp_413 = subgroupShuffleXor(values[0], 1u);
    let tmp_414 = subgroupShuffleXor(keys[1], 1u);
    let tmp_415 = subgroupShuffleXor(values[1], 1u);
    let tmp_416 = subgroupShuffleXor(keys[2], 1u);
    let tmp_417 = subgroupShuffleXor(values[2], 1u);
    let tmp_418 = subgroupShuffleXor(keys[3], 1u);
    let tmp_419 = subgroupShuffleXor(values[3], 1u);
    let tmp_420 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_421 = keys[0] < tmp_412 || (keys[0] == tmp_412 && values[0] < tmp_413);
    if tmp_420 == tmp_421 { keys[0] = tmp_412; values[0] = tmp_413; }
    let tmp_422 = keys[1] < tmp_414 || (keys[1] == tmp_414 && values[1] < tmp_415);
    if tmp_420 == tmp_422 { keys[1] = tmp_414; values[1] = tmp_415; }
    let tmp_423 = keys[2] < tmp_416 || (keys[2] == tmp_416 && values[2] < tmp_417);
    if tmp_420 == tmp_423 { keys[2] = tmp_416; values[2] = tmp_417; }
    let tmp_424 = keys[3] < tmp_418 || (keys[3] == tmp_418 && values[3] < tmp_419);
    if tmp_420 == tmp_424 { keys[3] = tmp_418; values[3] = tmp_419; }
    }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_425 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_425;let tmp_426 = values[0]; values[0] = values[2]; values[2] = tmp_426; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_427 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_427;let tmp_428 = values[1]; values[1] = values[3]; values[3] = tmp_428; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_429 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_429;let tmp_430 = values[0]; values[0] = values[1]; values[1] = tmp_430; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_431 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_431;let tmp_432 = values[2]; values[2] = values[3]; values[3] = tmp_432; }
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
