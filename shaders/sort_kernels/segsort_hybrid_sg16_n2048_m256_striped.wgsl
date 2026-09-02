
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
const R: u32 = 16u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybrid_sg16_n2048_m256_striped(
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
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_4 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_4;let tmp_5 = values[4]; values[4] = values[5]; values[5] = tmp_5; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_6 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_6;let tmp_7 = values[6]; values[6] = values[7]; values[7] = tmp_7; }
    }
    // exch_local(3,8) 
    // cmp_swap(0,3)
    if keys[0] > keys[3] || (keys[0] == keys[3] && values[0] > values[3]) {
    // swap(0,3) 
    { let tmp_8 = keys[0]; keys[0] = keys[3]; keys[3] = tmp_8;let tmp_9 = values[0]; values[0] = values[3]; values[3] = tmp_9; }
    }
    // cmp_swap(1,2)
    if keys[1] > keys[2] || (keys[1] == keys[2] && values[1] > values[2]) {
    // swap(1,2) 
    { let tmp_10 = keys[1]; keys[1] = keys[2]; keys[2] = tmp_10;let tmp_11 = values[1]; values[1] = values[2]; values[2] = tmp_11; }
    }
    // cmp_swap(4,7)
    if keys[4] > keys[7] || (keys[4] == keys[7] && values[4] > values[7]) {
    // swap(4,7) 
    { let tmp_12 = keys[4]; keys[4] = keys[7]; keys[7] = tmp_12;let tmp_13 = values[4]; values[4] = values[7]; values[7] = tmp_13; }
    }
    // cmp_swap(5,6)
    if keys[5] > keys[6] || (keys[5] == keys[6] && values[5] > values[6]) {
    // swap(5,6) 
    { let tmp_14 = keys[5]; keys[5] = keys[6]; keys[6] = tmp_14;let tmp_15 = values[5]; values[5] = values[6]; values[6] = tmp_15; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_16 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_16;let tmp_17 = values[0]; values[0] = values[1]; values[1] = tmp_17; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_18 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_18;let tmp_19 = values[2]; values[2] = values[3]; values[3] = tmp_19; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_20 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_20;let tmp_21 = values[4]; values[4] = values[5]; values[5] = tmp_21; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_22 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_22;let tmp_23 = values[6]; values[6] = values[7]; values[7] = tmp_23; }
    }
    // exch_local(7,8) 
    // cmp_swap(0,7)
    if keys[0] > keys[7] || (keys[0] == keys[7] && values[0] > values[7]) {
    // swap(0,7) 
    { let tmp_24 = keys[0]; keys[0] = keys[7]; keys[7] = tmp_24;let tmp_25 = values[0]; values[0] = values[7]; values[7] = tmp_25; }
    }
    // cmp_swap(1,6)
    if keys[1] > keys[6] || (keys[1] == keys[6] && values[1] > values[6]) {
    // swap(1,6) 
    { let tmp_26 = keys[1]; keys[1] = keys[6]; keys[6] = tmp_26;let tmp_27 = values[1]; values[1] = values[6]; values[6] = tmp_27; }
    }
    // cmp_swap(2,5)
    if keys[2] > keys[5] || (keys[2] == keys[5] && values[2] > values[5]) {
    // swap(2,5) 
    { let tmp_28 = keys[2]; keys[2] = keys[5]; keys[5] = tmp_28;let tmp_29 = values[2]; values[2] = values[5]; values[5] = tmp_29; }
    }
    // cmp_swap(3,4)
    if keys[3] > keys[4] || (keys[3] == keys[4] && values[3] > values[4]) {
    // swap(3,4) 
    { let tmp_30 = keys[3]; keys[3] = keys[4]; keys[4] = tmp_30;let tmp_31 = values[3]; values[3] = values[4]; values[4] = tmp_31; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_32 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_32;let tmp_33 = values[0]; values[0] = values[2]; values[2] = tmp_33; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_34 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_34;let tmp_35 = values[1]; values[1] = values[3]; values[3] = tmp_35; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_36 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_36;let tmp_37 = values[4]; values[4] = values[6]; values[6] = tmp_37; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_38 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_38;let tmp_39 = values[5]; values[5] = values[7]; values[7] = tmp_39; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_40 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_40;let tmp_41 = values[0]; values[0] = values[1]; values[1] = tmp_41; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_42 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_42;let tmp_43 = values[2]; values[2] = values[3]; values[3] = tmp_43; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_44 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_44;let tmp_45 = values[4]; values[4] = values[5]; values[5] = tmp_45; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_46 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_46;let tmp_47 = values[6]; values[6] = values[7]; values[7] = tmp_47; }
    }
    // exch_intxn(tmask:1,swbit:0,wpt:8)
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
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_73 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_73;let tmp_74 = values[0]; values[0] = values[4]; values[4] = tmp_74; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_75 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_75;let tmp_76 = values[1]; values[1] = values[5]; values[5] = tmp_76; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_77 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_77;let tmp_78 = values[2]; values[2] = values[6]; values[6] = tmp_78; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_79 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_79;let tmp_80 = values[3]; values[3] = values[7]; values[7] = tmp_80; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_81 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_81;let tmp_82 = values[0]; values[0] = values[2]; values[2] = tmp_82; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_83 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_83;let tmp_84 = values[1]; values[1] = values[3]; values[3] = tmp_84; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_85 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_85;let tmp_86 = values[4]; values[4] = values[6]; values[6] = tmp_86; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_87 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_87;let tmp_88 = values[5]; values[5] = values[7]; values[7] = tmp_88; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_89 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_89;let tmp_90 = values[0]; values[0] = values[1]; values[1] = tmp_90; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_91 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_91;let tmp_92 = values[2]; values[2] = values[3]; values[3] = tmp_92; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_93 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_93;let tmp_94 = values[4]; values[4] = values[5]; values[5] = tmp_94; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_95 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_95;let tmp_96 = values[6]; values[6] = values[7]; values[7] = tmp_96; }
    }
    // exch_intxn(tmask:3,swbit:1,wpt:8)
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
    // exch_paral(tmask:1,swbit:0,wpt:8) 
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
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_147 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_147;let tmp_148 = values[0]; values[0] = values[4]; values[4] = tmp_148; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_149 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_149;let tmp_150 = values[1]; values[1] = values[5]; values[5] = tmp_150; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_151 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_151;let tmp_152 = values[2]; values[2] = values[6]; values[6] = tmp_152; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_153 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_153;let tmp_154 = values[3]; values[3] = values[7]; values[7] = tmp_154; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_155 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_155;let tmp_156 = values[0]; values[0] = values[2]; values[2] = tmp_156; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_157 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_157;let tmp_158 = values[1]; values[1] = values[3]; values[3] = tmp_158; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_159 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_159;let tmp_160 = values[4]; values[4] = values[6]; values[6] = tmp_160; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_161 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_161;let tmp_162 = values[5]; values[5] = values[7]; values[7] = tmp_162; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_163 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_163;let tmp_164 = values[0]; values[0] = values[1]; values[1] = tmp_164; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_165 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_165;let tmp_166 = values[2]; values[2] = values[3]; values[3] = tmp_166; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_167 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_167;let tmp_168 = values[4]; values[4] = values[5]; values[5] = tmp_168; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_169 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_169;let tmp_170 = values[6]; values[6] = values[7]; values[7] = tmp_170; }
    }
    // exch_intxn(tmask:7,swbit:2,wpt:8)
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
    // exch_paral(tmask:2,swbit:1,wpt:8) 
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
    // exch_paral(tmask:1,swbit:0,wpt:8) 
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
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_246 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_246;let tmp_247 = values[0]; values[0] = values[4]; values[4] = tmp_247; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_248 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_248;let tmp_249 = values[1]; values[1] = values[5]; values[5] = tmp_249; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_250 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_250;let tmp_251 = values[2]; values[2] = values[6]; values[6] = tmp_251; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_252 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_252;let tmp_253 = values[3]; values[3] = values[7]; values[7] = tmp_253; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_254 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_254;let tmp_255 = values[0]; values[0] = values[2]; values[2] = tmp_255; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_256 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_256;let tmp_257 = values[1]; values[1] = values[3]; values[3] = tmp_257; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_258 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_258;let tmp_259 = values[4]; values[4] = values[6]; values[6] = tmp_259; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_260 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_260;let tmp_261 = values[5]; values[5] = values[7]; values[7] = tmp_261; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_262 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_262;let tmp_263 = values[0]; values[0] = values[1]; values[1] = tmp_263; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_264 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_264;let tmp_265 = values[2]; values[2] = values[3]; values[3] = tmp_265; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_266 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_266;let tmp_267 = values[4]; values[4] = values[5]; values[5] = tmp_267; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_268 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_268;let tmp_269 = values[6]; values[6] = values[7]; values[7] = tmp_269; }
    }
    // exch_intxn(tmask:15,swbit:3,wpt:8)
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
    // exch_paral(tmask:4,swbit:2,wpt:8) 
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
    // exch_paral(tmask:2,swbit:1,wpt:8) 
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
    // exch_paral(tmask:1,swbit:0,wpt:8) 
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
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_370 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_370;let tmp_371 = values[0]; values[0] = values[4]; values[4] = tmp_371; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_372 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_372;let tmp_373 = values[1]; values[1] = values[5]; values[5] = tmp_373; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_374 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_374;let tmp_375 = values[2]; values[2] = values[6]; values[6] = tmp_375; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_376 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_376;let tmp_377 = values[3]; values[3] = values[7]; values[7] = tmp_377; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_378 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_378;let tmp_379 = values[0]; values[0] = values[2]; values[2] = tmp_379; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_380 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_380;let tmp_381 = values[1]; values[1] = values[3]; values[3] = tmp_381; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_382 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_382;let tmp_383 = values[4]; values[4] = values[6]; values[6] = tmp_383; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_384 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_384;let tmp_385 = values[5]; values[5] = values[7]; values[7] = tmp_385; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_386 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_386;let tmp_387 = values[0]; values[0] = values[1]; values[1] = tmp_387; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_388 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_388;let tmp_389 = values[2]; values[2] = values[3]; values[3] = tmp_389; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_390 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_390;let tmp_391 = values[4]; values[4] = values[5]; values[5] = tmp_391; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_392 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_392;let tmp_393 = values[6]; values[6] = values[7]; values[7] = tmp_393; }
    }
    // exch_intxn(tmask:31,swbit:4,wpt:8)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_394 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_395 = seg_base + (local_tid ^ 31u); let tmp_396 = smem_keys[tmp_395 * WPT + 7u]; let tmp_397 = smem_vals[tmp_395 * WPT + 7u]; let tmp_398 = keys[0] < tmp_396 || (keys[0] == tmp_396 && values[0] < tmp_397); if tmp_394 == tmp_398 { keys[0] = tmp_396; values[0] = tmp_397; } let tmp_399 = smem_keys[tmp_395 * WPT + 6u]; let tmp_400 = smem_vals[tmp_395 * WPT + 6u]; let tmp_401 = keys[1] < tmp_399 || (keys[1] == tmp_399 && values[1] < tmp_400); if tmp_394 == tmp_401 { keys[1] = tmp_399; values[1] = tmp_400; } let tmp_402 = smem_keys[tmp_395 * WPT + 5u]; let tmp_403 = smem_vals[tmp_395 * WPT + 5u]; let tmp_404 = keys[2] < tmp_402 || (keys[2] == tmp_402 && values[2] < tmp_403); if tmp_394 == tmp_404 { keys[2] = tmp_402; values[2] = tmp_403; } let tmp_405 = smem_keys[tmp_395 * WPT + 4u]; let tmp_406 = smem_vals[tmp_395 * WPT + 4u]; let tmp_407 = keys[3] < tmp_405 || (keys[3] == tmp_405 && values[3] < tmp_406); if tmp_394 == tmp_407 { keys[3] = tmp_405; values[3] = tmp_406; } let tmp_408 = smem_keys[tmp_395 * WPT + 3u]; let tmp_409 = smem_vals[tmp_395 * WPT + 3u]; let tmp_410 = keys[4] < tmp_408 || (keys[4] == tmp_408 && values[4] < tmp_409); if tmp_394 == tmp_410 { keys[4] = tmp_408; values[4] = tmp_409; } let tmp_411 = smem_keys[tmp_395 * WPT + 2u]; let tmp_412 = smem_vals[tmp_395 * WPT + 2u]; let tmp_413 = keys[5] < tmp_411 || (keys[5] == tmp_411 && values[5] < tmp_412); if tmp_394 == tmp_413 { keys[5] = tmp_411; values[5] = tmp_412; } let tmp_414 = smem_keys[tmp_395 * WPT + 1u]; let tmp_415 = smem_vals[tmp_395 * WPT + 1u]; let tmp_416 = keys[6] < tmp_414 || (keys[6] == tmp_414 && values[6] < tmp_415); if tmp_394 == tmp_416 { keys[6] = tmp_414; values[6] = tmp_415; } let tmp_417 = smem_keys[tmp_395 * WPT + 0u]; let tmp_418 = smem_vals[tmp_395 * WPT + 0u]; let tmp_419 = keys[7] < tmp_417 || (keys[7] == tmp_417 && values[7] < tmp_418); if tmp_394 == tmp_419 { keys[7] = tmp_417; values[7] = tmp_418; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:8) 
    {
    let tmp_420 = subgroupShuffleXor(keys[0], 8u);
    let tmp_421 = subgroupShuffleXor(values[0], 8u);
    let tmp_422 = subgroupShuffleXor(keys[1], 8u);
    let tmp_423 = subgroupShuffleXor(values[1], 8u);
    let tmp_424 = subgroupShuffleXor(keys[2], 8u);
    let tmp_425 = subgroupShuffleXor(values[2], 8u);
    let tmp_426 = subgroupShuffleXor(keys[3], 8u);
    let tmp_427 = subgroupShuffleXor(values[3], 8u);
    let tmp_428 = subgroupShuffleXor(keys[4], 8u);
    let tmp_429 = subgroupShuffleXor(values[4], 8u);
    let tmp_430 = subgroupShuffleXor(keys[5], 8u);
    let tmp_431 = subgroupShuffleXor(values[5], 8u);
    let tmp_432 = subgroupShuffleXor(keys[6], 8u);
    let tmp_433 = subgroupShuffleXor(values[6], 8u);
    let tmp_434 = subgroupShuffleXor(keys[7], 8u);
    let tmp_435 = subgroupShuffleXor(values[7], 8u);
    let tmp_436 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_437 = keys[0] < tmp_420 || (keys[0] == tmp_420 && values[0] < tmp_421);
    if tmp_436 == tmp_437 { keys[0] = tmp_420; values[0] = tmp_421; }
    let tmp_438 = keys[1] < tmp_422 || (keys[1] == tmp_422 && values[1] < tmp_423);
    if tmp_436 == tmp_438 { keys[1] = tmp_422; values[1] = tmp_423; }
    let tmp_439 = keys[2] < tmp_424 || (keys[2] == tmp_424 && values[2] < tmp_425);
    if tmp_436 == tmp_439 { keys[2] = tmp_424; values[2] = tmp_425; }
    let tmp_440 = keys[3] < tmp_426 || (keys[3] == tmp_426 && values[3] < tmp_427);
    if tmp_436 == tmp_440 { keys[3] = tmp_426; values[3] = tmp_427; }
    let tmp_441 = keys[4] < tmp_428 || (keys[4] == tmp_428 && values[4] < tmp_429);
    if tmp_436 == tmp_441 { keys[4] = tmp_428; values[4] = tmp_429; }
    let tmp_442 = keys[5] < tmp_430 || (keys[5] == tmp_430 && values[5] < tmp_431);
    if tmp_436 == tmp_442 { keys[5] = tmp_430; values[5] = tmp_431; }
    let tmp_443 = keys[6] < tmp_432 || (keys[6] == tmp_432 && values[6] < tmp_433);
    if tmp_436 == tmp_443 { keys[6] = tmp_432; values[6] = tmp_433; }
    let tmp_444 = keys[7] < tmp_434 || (keys[7] == tmp_434 && values[7] < tmp_435);
    if tmp_436 == tmp_444 { keys[7] = tmp_434; values[7] = tmp_435; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:8) 
    {
    let tmp_445 = subgroupShuffleXor(keys[0], 4u);
    let tmp_446 = subgroupShuffleXor(values[0], 4u);
    let tmp_447 = subgroupShuffleXor(keys[1], 4u);
    let tmp_448 = subgroupShuffleXor(values[1], 4u);
    let tmp_449 = subgroupShuffleXor(keys[2], 4u);
    let tmp_450 = subgroupShuffleXor(values[2], 4u);
    let tmp_451 = subgroupShuffleXor(keys[3], 4u);
    let tmp_452 = subgroupShuffleXor(values[3], 4u);
    let tmp_453 = subgroupShuffleXor(keys[4], 4u);
    let tmp_454 = subgroupShuffleXor(values[4], 4u);
    let tmp_455 = subgroupShuffleXor(keys[5], 4u);
    let tmp_456 = subgroupShuffleXor(values[5], 4u);
    let tmp_457 = subgroupShuffleXor(keys[6], 4u);
    let tmp_458 = subgroupShuffleXor(values[6], 4u);
    let tmp_459 = subgroupShuffleXor(keys[7], 4u);
    let tmp_460 = subgroupShuffleXor(values[7], 4u);
    let tmp_461 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_462 = keys[0] < tmp_445 || (keys[0] == tmp_445 && values[0] < tmp_446);
    if tmp_461 == tmp_462 { keys[0] = tmp_445; values[0] = tmp_446; }
    let tmp_463 = keys[1] < tmp_447 || (keys[1] == tmp_447 && values[1] < tmp_448);
    if tmp_461 == tmp_463 { keys[1] = tmp_447; values[1] = tmp_448; }
    let tmp_464 = keys[2] < tmp_449 || (keys[2] == tmp_449 && values[2] < tmp_450);
    if tmp_461 == tmp_464 { keys[2] = tmp_449; values[2] = tmp_450; }
    let tmp_465 = keys[3] < tmp_451 || (keys[3] == tmp_451 && values[3] < tmp_452);
    if tmp_461 == tmp_465 { keys[3] = tmp_451; values[3] = tmp_452; }
    let tmp_466 = keys[4] < tmp_453 || (keys[4] == tmp_453 && values[4] < tmp_454);
    if tmp_461 == tmp_466 { keys[4] = tmp_453; values[4] = tmp_454; }
    let tmp_467 = keys[5] < tmp_455 || (keys[5] == tmp_455 && values[5] < tmp_456);
    if tmp_461 == tmp_467 { keys[5] = tmp_455; values[5] = tmp_456; }
    let tmp_468 = keys[6] < tmp_457 || (keys[6] == tmp_457 && values[6] < tmp_458);
    if tmp_461 == tmp_468 { keys[6] = tmp_457; values[6] = tmp_458; }
    let tmp_469 = keys[7] < tmp_459 || (keys[7] == tmp_459 && values[7] < tmp_460);
    if tmp_461 == tmp_469 { keys[7] = tmp_459; values[7] = tmp_460; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    {
    let tmp_470 = subgroupShuffleXor(keys[0], 2u);
    let tmp_471 = subgroupShuffleXor(values[0], 2u);
    let tmp_472 = subgroupShuffleXor(keys[1], 2u);
    let tmp_473 = subgroupShuffleXor(values[1], 2u);
    let tmp_474 = subgroupShuffleXor(keys[2], 2u);
    let tmp_475 = subgroupShuffleXor(values[2], 2u);
    let tmp_476 = subgroupShuffleXor(keys[3], 2u);
    let tmp_477 = subgroupShuffleXor(values[3], 2u);
    let tmp_478 = subgroupShuffleXor(keys[4], 2u);
    let tmp_479 = subgroupShuffleXor(values[4], 2u);
    let tmp_480 = subgroupShuffleXor(keys[5], 2u);
    let tmp_481 = subgroupShuffleXor(values[5], 2u);
    let tmp_482 = subgroupShuffleXor(keys[6], 2u);
    let tmp_483 = subgroupShuffleXor(values[6], 2u);
    let tmp_484 = subgroupShuffleXor(keys[7], 2u);
    let tmp_485 = subgroupShuffleXor(values[7], 2u);
    let tmp_486 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_487 = keys[0] < tmp_470 || (keys[0] == tmp_470 && values[0] < tmp_471);
    if tmp_486 == tmp_487 { keys[0] = tmp_470; values[0] = tmp_471; }
    let tmp_488 = keys[1] < tmp_472 || (keys[1] == tmp_472 && values[1] < tmp_473);
    if tmp_486 == tmp_488 { keys[1] = tmp_472; values[1] = tmp_473; }
    let tmp_489 = keys[2] < tmp_474 || (keys[2] == tmp_474 && values[2] < tmp_475);
    if tmp_486 == tmp_489 { keys[2] = tmp_474; values[2] = tmp_475; }
    let tmp_490 = keys[3] < tmp_476 || (keys[3] == tmp_476 && values[3] < tmp_477);
    if tmp_486 == tmp_490 { keys[3] = tmp_476; values[3] = tmp_477; }
    let tmp_491 = keys[4] < tmp_478 || (keys[4] == tmp_478 && values[4] < tmp_479);
    if tmp_486 == tmp_491 { keys[4] = tmp_478; values[4] = tmp_479; }
    let tmp_492 = keys[5] < tmp_480 || (keys[5] == tmp_480 && values[5] < tmp_481);
    if tmp_486 == tmp_492 { keys[5] = tmp_480; values[5] = tmp_481; }
    let tmp_493 = keys[6] < tmp_482 || (keys[6] == tmp_482 && values[6] < tmp_483);
    if tmp_486 == tmp_493 { keys[6] = tmp_482; values[6] = tmp_483; }
    let tmp_494 = keys[7] < tmp_484 || (keys[7] == tmp_484 && values[7] < tmp_485);
    if tmp_486 == tmp_494 { keys[7] = tmp_484; values[7] = tmp_485; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    {
    let tmp_495 = subgroupShuffleXor(keys[0], 1u);
    let tmp_496 = subgroupShuffleXor(values[0], 1u);
    let tmp_497 = subgroupShuffleXor(keys[1], 1u);
    let tmp_498 = subgroupShuffleXor(values[1], 1u);
    let tmp_499 = subgroupShuffleXor(keys[2], 1u);
    let tmp_500 = subgroupShuffleXor(values[2], 1u);
    let tmp_501 = subgroupShuffleXor(keys[3], 1u);
    let tmp_502 = subgroupShuffleXor(values[3], 1u);
    let tmp_503 = subgroupShuffleXor(keys[4], 1u);
    let tmp_504 = subgroupShuffleXor(values[4], 1u);
    let tmp_505 = subgroupShuffleXor(keys[5], 1u);
    let tmp_506 = subgroupShuffleXor(values[5], 1u);
    let tmp_507 = subgroupShuffleXor(keys[6], 1u);
    let tmp_508 = subgroupShuffleXor(values[6], 1u);
    let tmp_509 = subgroupShuffleXor(keys[7], 1u);
    let tmp_510 = subgroupShuffleXor(values[7], 1u);
    let tmp_511 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_512 = keys[0] < tmp_495 || (keys[0] == tmp_495 && values[0] < tmp_496);
    if tmp_511 == tmp_512 { keys[0] = tmp_495; values[0] = tmp_496; }
    let tmp_513 = keys[1] < tmp_497 || (keys[1] == tmp_497 && values[1] < tmp_498);
    if tmp_511 == tmp_513 { keys[1] = tmp_497; values[1] = tmp_498; }
    let tmp_514 = keys[2] < tmp_499 || (keys[2] == tmp_499 && values[2] < tmp_500);
    if tmp_511 == tmp_514 { keys[2] = tmp_499; values[2] = tmp_500; }
    let tmp_515 = keys[3] < tmp_501 || (keys[3] == tmp_501 && values[3] < tmp_502);
    if tmp_511 == tmp_515 { keys[3] = tmp_501; values[3] = tmp_502; }
    let tmp_516 = keys[4] < tmp_503 || (keys[4] == tmp_503 && values[4] < tmp_504);
    if tmp_511 == tmp_516 { keys[4] = tmp_503; values[4] = tmp_504; }
    let tmp_517 = keys[5] < tmp_505 || (keys[5] == tmp_505 && values[5] < tmp_506);
    if tmp_511 == tmp_517 { keys[5] = tmp_505; values[5] = tmp_506; }
    let tmp_518 = keys[6] < tmp_507 || (keys[6] == tmp_507 && values[6] < tmp_508);
    if tmp_511 == tmp_518 { keys[6] = tmp_507; values[6] = tmp_508; }
    let tmp_519 = keys[7] < tmp_509 || (keys[7] == tmp_509 && values[7] < tmp_510);
    if tmp_511 == tmp_519 { keys[7] = tmp_509; values[7] = tmp_510; }
    }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_520 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_520;let tmp_521 = values[0]; values[0] = values[4]; values[4] = tmp_521; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_522 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_522;let tmp_523 = values[1]; values[1] = values[5]; values[5] = tmp_523; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_524 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_524;let tmp_525 = values[2]; values[2] = values[6]; values[6] = tmp_525; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_526 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_526;let tmp_527 = values[3]; values[3] = values[7]; values[7] = tmp_527; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_528 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_528;let tmp_529 = values[0]; values[0] = values[2]; values[2] = tmp_529; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_530 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_530;let tmp_531 = values[1]; values[1] = values[3]; values[3] = tmp_531; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_532 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_532;let tmp_533 = values[4]; values[4] = values[6]; values[6] = tmp_533; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_534 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_534;let tmp_535 = values[5]; values[5] = values[7]; values[7] = tmp_535; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_536 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_536;let tmp_537 = values[0]; values[0] = values[1]; values[1] = tmp_537; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_538 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_538;let tmp_539 = values[2]; values[2] = values[3]; values[3] = tmp_539; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_540 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_540;let tmp_541 = values[4]; values[4] = values[5]; values[5] = tmp_541; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_542 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_542;let tmp_543 = values[6]; values[6] = values[7]; values[7] = tmp_543; }
    }
    // exch_intxn(tmask:63,swbit:5,wpt:8)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_544 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_545 = seg_base + (local_tid ^ 63u); let tmp_546 = smem_keys[tmp_545 * WPT + 7u]; let tmp_547 = smem_vals[tmp_545 * WPT + 7u]; let tmp_548 = keys[0] < tmp_546 || (keys[0] == tmp_546 && values[0] < tmp_547); if tmp_544 == tmp_548 { keys[0] = tmp_546; values[0] = tmp_547; } let tmp_549 = smem_keys[tmp_545 * WPT + 6u]; let tmp_550 = smem_vals[tmp_545 * WPT + 6u]; let tmp_551 = keys[1] < tmp_549 || (keys[1] == tmp_549 && values[1] < tmp_550); if tmp_544 == tmp_551 { keys[1] = tmp_549; values[1] = tmp_550; } let tmp_552 = smem_keys[tmp_545 * WPT + 5u]; let tmp_553 = smem_vals[tmp_545 * WPT + 5u]; let tmp_554 = keys[2] < tmp_552 || (keys[2] == tmp_552 && values[2] < tmp_553); if tmp_544 == tmp_554 { keys[2] = tmp_552; values[2] = tmp_553; } let tmp_555 = smem_keys[tmp_545 * WPT + 4u]; let tmp_556 = smem_vals[tmp_545 * WPT + 4u]; let tmp_557 = keys[3] < tmp_555 || (keys[3] == tmp_555 && values[3] < tmp_556); if tmp_544 == tmp_557 { keys[3] = tmp_555; values[3] = tmp_556; } let tmp_558 = smem_keys[tmp_545 * WPT + 3u]; let tmp_559 = smem_vals[tmp_545 * WPT + 3u]; let tmp_560 = keys[4] < tmp_558 || (keys[4] == tmp_558 && values[4] < tmp_559); if tmp_544 == tmp_560 { keys[4] = tmp_558; values[4] = tmp_559; } let tmp_561 = smem_keys[tmp_545 * WPT + 2u]; let tmp_562 = smem_vals[tmp_545 * WPT + 2u]; let tmp_563 = keys[5] < tmp_561 || (keys[5] == tmp_561 && values[5] < tmp_562); if tmp_544 == tmp_563 { keys[5] = tmp_561; values[5] = tmp_562; } let tmp_564 = smem_keys[tmp_545 * WPT + 1u]; let tmp_565 = smem_vals[tmp_545 * WPT + 1u]; let tmp_566 = keys[6] < tmp_564 || (keys[6] == tmp_564 && values[6] < tmp_565); if tmp_544 == tmp_566 { keys[6] = tmp_564; values[6] = tmp_565; } let tmp_567 = smem_keys[tmp_545 * WPT + 0u]; let tmp_568 = smem_vals[tmp_545 * WPT + 0u]; let tmp_569 = keys[7] < tmp_567 || (keys[7] == tmp_567 && values[7] < tmp_568); if tmp_544 == tmp_569 { keys[7] = tmp_567; values[7] = tmp_568; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_570 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_571 = seg_base + (local_tid ^ 16u); let tmp_572 = smem_keys[tmp_571 * WPT + 0u]; let tmp_573 = smem_vals[tmp_571 * WPT + 0u]; let tmp_574 = keys[0] < tmp_572 || (keys[0] == tmp_572 && values[0] < tmp_573); if tmp_570 == tmp_574 { keys[0] = tmp_572; values[0] = tmp_573; } let tmp_575 = smem_keys[tmp_571 * WPT + 1u]; let tmp_576 = smem_vals[tmp_571 * WPT + 1u]; let tmp_577 = keys[1] < tmp_575 || (keys[1] == tmp_575 && values[1] < tmp_576); if tmp_570 == tmp_577 { keys[1] = tmp_575; values[1] = tmp_576; } let tmp_578 = smem_keys[tmp_571 * WPT + 2u]; let tmp_579 = smem_vals[tmp_571 * WPT + 2u]; let tmp_580 = keys[2] < tmp_578 || (keys[2] == tmp_578 && values[2] < tmp_579); if tmp_570 == tmp_580 { keys[2] = tmp_578; values[2] = tmp_579; } let tmp_581 = smem_keys[tmp_571 * WPT + 3u]; let tmp_582 = smem_vals[tmp_571 * WPT + 3u]; let tmp_583 = keys[3] < tmp_581 || (keys[3] == tmp_581 && values[3] < tmp_582); if tmp_570 == tmp_583 { keys[3] = tmp_581; values[3] = tmp_582; } let tmp_584 = smem_keys[tmp_571 * WPT + 4u]; let tmp_585 = smem_vals[tmp_571 * WPT + 4u]; let tmp_586 = keys[4] < tmp_584 || (keys[4] == tmp_584 && values[4] < tmp_585); if tmp_570 == tmp_586 { keys[4] = tmp_584; values[4] = tmp_585; } let tmp_587 = smem_keys[tmp_571 * WPT + 5u]; let tmp_588 = smem_vals[tmp_571 * WPT + 5u]; let tmp_589 = keys[5] < tmp_587 || (keys[5] == tmp_587 && values[5] < tmp_588); if tmp_570 == tmp_589 { keys[5] = tmp_587; values[5] = tmp_588; } let tmp_590 = smem_keys[tmp_571 * WPT + 6u]; let tmp_591 = smem_vals[tmp_571 * WPT + 6u]; let tmp_592 = keys[6] < tmp_590 || (keys[6] == tmp_590 && values[6] < tmp_591); if tmp_570 == tmp_592 { keys[6] = tmp_590; values[6] = tmp_591; } let tmp_593 = smem_keys[tmp_571 * WPT + 7u]; let tmp_594 = smem_vals[tmp_571 * WPT + 7u]; let tmp_595 = keys[7] < tmp_593 || (keys[7] == tmp_593 && values[7] < tmp_594); if tmp_570 == tmp_595 { keys[7] = tmp_593; values[7] = tmp_594; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:8) 
    {
    let tmp_596 = subgroupShuffleXor(keys[0], 8u);
    let tmp_597 = subgroupShuffleXor(values[0], 8u);
    let tmp_598 = subgroupShuffleXor(keys[1], 8u);
    let tmp_599 = subgroupShuffleXor(values[1], 8u);
    let tmp_600 = subgroupShuffleXor(keys[2], 8u);
    let tmp_601 = subgroupShuffleXor(values[2], 8u);
    let tmp_602 = subgroupShuffleXor(keys[3], 8u);
    let tmp_603 = subgroupShuffleXor(values[3], 8u);
    let tmp_604 = subgroupShuffleXor(keys[4], 8u);
    let tmp_605 = subgroupShuffleXor(values[4], 8u);
    let tmp_606 = subgroupShuffleXor(keys[5], 8u);
    let tmp_607 = subgroupShuffleXor(values[5], 8u);
    let tmp_608 = subgroupShuffleXor(keys[6], 8u);
    let tmp_609 = subgroupShuffleXor(values[6], 8u);
    let tmp_610 = subgroupShuffleXor(keys[7], 8u);
    let tmp_611 = subgroupShuffleXor(values[7], 8u);
    let tmp_612 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_613 = keys[0] < tmp_596 || (keys[0] == tmp_596 && values[0] < tmp_597);
    if tmp_612 == tmp_613 { keys[0] = tmp_596; values[0] = tmp_597; }
    let tmp_614 = keys[1] < tmp_598 || (keys[1] == tmp_598 && values[1] < tmp_599);
    if tmp_612 == tmp_614 { keys[1] = tmp_598; values[1] = tmp_599; }
    let tmp_615 = keys[2] < tmp_600 || (keys[2] == tmp_600 && values[2] < tmp_601);
    if tmp_612 == tmp_615 { keys[2] = tmp_600; values[2] = tmp_601; }
    let tmp_616 = keys[3] < tmp_602 || (keys[3] == tmp_602 && values[3] < tmp_603);
    if tmp_612 == tmp_616 { keys[3] = tmp_602; values[3] = tmp_603; }
    let tmp_617 = keys[4] < tmp_604 || (keys[4] == tmp_604 && values[4] < tmp_605);
    if tmp_612 == tmp_617 { keys[4] = tmp_604; values[4] = tmp_605; }
    let tmp_618 = keys[5] < tmp_606 || (keys[5] == tmp_606 && values[5] < tmp_607);
    if tmp_612 == tmp_618 { keys[5] = tmp_606; values[5] = tmp_607; }
    let tmp_619 = keys[6] < tmp_608 || (keys[6] == tmp_608 && values[6] < tmp_609);
    if tmp_612 == tmp_619 { keys[6] = tmp_608; values[6] = tmp_609; }
    let tmp_620 = keys[7] < tmp_610 || (keys[7] == tmp_610 && values[7] < tmp_611);
    if tmp_612 == tmp_620 { keys[7] = tmp_610; values[7] = tmp_611; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:8) 
    {
    let tmp_621 = subgroupShuffleXor(keys[0], 4u);
    let tmp_622 = subgroupShuffleXor(values[0], 4u);
    let tmp_623 = subgroupShuffleXor(keys[1], 4u);
    let tmp_624 = subgroupShuffleXor(values[1], 4u);
    let tmp_625 = subgroupShuffleXor(keys[2], 4u);
    let tmp_626 = subgroupShuffleXor(values[2], 4u);
    let tmp_627 = subgroupShuffleXor(keys[3], 4u);
    let tmp_628 = subgroupShuffleXor(values[3], 4u);
    let tmp_629 = subgroupShuffleXor(keys[4], 4u);
    let tmp_630 = subgroupShuffleXor(values[4], 4u);
    let tmp_631 = subgroupShuffleXor(keys[5], 4u);
    let tmp_632 = subgroupShuffleXor(values[5], 4u);
    let tmp_633 = subgroupShuffleXor(keys[6], 4u);
    let tmp_634 = subgroupShuffleXor(values[6], 4u);
    let tmp_635 = subgroupShuffleXor(keys[7], 4u);
    let tmp_636 = subgroupShuffleXor(values[7], 4u);
    let tmp_637 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_638 = keys[0] < tmp_621 || (keys[0] == tmp_621 && values[0] < tmp_622);
    if tmp_637 == tmp_638 { keys[0] = tmp_621; values[0] = tmp_622; }
    let tmp_639 = keys[1] < tmp_623 || (keys[1] == tmp_623 && values[1] < tmp_624);
    if tmp_637 == tmp_639 { keys[1] = tmp_623; values[1] = tmp_624; }
    let tmp_640 = keys[2] < tmp_625 || (keys[2] == tmp_625 && values[2] < tmp_626);
    if tmp_637 == tmp_640 { keys[2] = tmp_625; values[2] = tmp_626; }
    let tmp_641 = keys[3] < tmp_627 || (keys[3] == tmp_627 && values[3] < tmp_628);
    if tmp_637 == tmp_641 { keys[3] = tmp_627; values[3] = tmp_628; }
    let tmp_642 = keys[4] < tmp_629 || (keys[4] == tmp_629 && values[4] < tmp_630);
    if tmp_637 == tmp_642 { keys[4] = tmp_629; values[4] = tmp_630; }
    let tmp_643 = keys[5] < tmp_631 || (keys[5] == tmp_631 && values[5] < tmp_632);
    if tmp_637 == tmp_643 { keys[5] = tmp_631; values[5] = tmp_632; }
    let tmp_644 = keys[6] < tmp_633 || (keys[6] == tmp_633 && values[6] < tmp_634);
    if tmp_637 == tmp_644 { keys[6] = tmp_633; values[6] = tmp_634; }
    let tmp_645 = keys[7] < tmp_635 || (keys[7] == tmp_635 && values[7] < tmp_636);
    if tmp_637 == tmp_645 { keys[7] = tmp_635; values[7] = tmp_636; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    {
    let tmp_646 = subgroupShuffleXor(keys[0], 2u);
    let tmp_647 = subgroupShuffleXor(values[0], 2u);
    let tmp_648 = subgroupShuffleXor(keys[1], 2u);
    let tmp_649 = subgroupShuffleXor(values[1], 2u);
    let tmp_650 = subgroupShuffleXor(keys[2], 2u);
    let tmp_651 = subgroupShuffleXor(values[2], 2u);
    let tmp_652 = subgroupShuffleXor(keys[3], 2u);
    let tmp_653 = subgroupShuffleXor(values[3], 2u);
    let tmp_654 = subgroupShuffleXor(keys[4], 2u);
    let tmp_655 = subgroupShuffleXor(values[4], 2u);
    let tmp_656 = subgroupShuffleXor(keys[5], 2u);
    let tmp_657 = subgroupShuffleXor(values[5], 2u);
    let tmp_658 = subgroupShuffleXor(keys[6], 2u);
    let tmp_659 = subgroupShuffleXor(values[6], 2u);
    let tmp_660 = subgroupShuffleXor(keys[7], 2u);
    let tmp_661 = subgroupShuffleXor(values[7], 2u);
    let tmp_662 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_663 = keys[0] < tmp_646 || (keys[0] == tmp_646 && values[0] < tmp_647);
    if tmp_662 == tmp_663 { keys[0] = tmp_646; values[0] = tmp_647; }
    let tmp_664 = keys[1] < tmp_648 || (keys[1] == tmp_648 && values[1] < tmp_649);
    if tmp_662 == tmp_664 { keys[1] = tmp_648; values[1] = tmp_649; }
    let tmp_665 = keys[2] < tmp_650 || (keys[2] == tmp_650 && values[2] < tmp_651);
    if tmp_662 == tmp_665 { keys[2] = tmp_650; values[2] = tmp_651; }
    let tmp_666 = keys[3] < tmp_652 || (keys[3] == tmp_652 && values[3] < tmp_653);
    if tmp_662 == tmp_666 { keys[3] = tmp_652; values[3] = tmp_653; }
    let tmp_667 = keys[4] < tmp_654 || (keys[4] == tmp_654 && values[4] < tmp_655);
    if tmp_662 == tmp_667 { keys[4] = tmp_654; values[4] = tmp_655; }
    let tmp_668 = keys[5] < tmp_656 || (keys[5] == tmp_656 && values[5] < tmp_657);
    if tmp_662 == tmp_668 { keys[5] = tmp_656; values[5] = tmp_657; }
    let tmp_669 = keys[6] < tmp_658 || (keys[6] == tmp_658 && values[6] < tmp_659);
    if tmp_662 == tmp_669 { keys[6] = tmp_658; values[6] = tmp_659; }
    let tmp_670 = keys[7] < tmp_660 || (keys[7] == tmp_660 && values[7] < tmp_661);
    if tmp_662 == tmp_670 { keys[7] = tmp_660; values[7] = tmp_661; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    {
    let tmp_671 = subgroupShuffleXor(keys[0], 1u);
    let tmp_672 = subgroupShuffleXor(values[0], 1u);
    let tmp_673 = subgroupShuffleXor(keys[1], 1u);
    let tmp_674 = subgroupShuffleXor(values[1], 1u);
    let tmp_675 = subgroupShuffleXor(keys[2], 1u);
    let tmp_676 = subgroupShuffleXor(values[2], 1u);
    let tmp_677 = subgroupShuffleXor(keys[3], 1u);
    let tmp_678 = subgroupShuffleXor(values[3], 1u);
    let tmp_679 = subgroupShuffleXor(keys[4], 1u);
    let tmp_680 = subgroupShuffleXor(values[4], 1u);
    let tmp_681 = subgroupShuffleXor(keys[5], 1u);
    let tmp_682 = subgroupShuffleXor(values[5], 1u);
    let tmp_683 = subgroupShuffleXor(keys[6], 1u);
    let tmp_684 = subgroupShuffleXor(values[6], 1u);
    let tmp_685 = subgroupShuffleXor(keys[7], 1u);
    let tmp_686 = subgroupShuffleXor(values[7], 1u);
    let tmp_687 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_688 = keys[0] < tmp_671 || (keys[0] == tmp_671 && values[0] < tmp_672);
    if tmp_687 == tmp_688 { keys[0] = tmp_671; values[0] = tmp_672; }
    let tmp_689 = keys[1] < tmp_673 || (keys[1] == tmp_673 && values[1] < tmp_674);
    if tmp_687 == tmp_689 { keys[1] = tmp_673; values[1] = tmp_674; }
    let tmp_690 = keys[2] < tmp_675 || (keys[2] == tmp_675 && values[2] < tmp_676);
    if tmp_687 == tmp_690 { keys[2] = tmp_675; values[2] = tmp_676; }
    let tmp_691 = keys[3] < tmp_677 || (keys[3] == tmp_677 && values[3] < tmp_678);
    if tmp_687 == tmp_691 { keys[3] = tmp_677; values[3] = tmp_678; }
    let tmp_692 = keys[4] < tmp_679 || (keys[4] == tmp_679 && values[4] < tmp_680);
    if tmp_687 == tmp_692 { keys[4] = tmp_679; values[4] = tmp_680; }
    let tmp_693 = keys[5] < tmp_681 || (keys[5] == tmp_681 && values[5] < tmp_682);
    if tmp_687 == tmp_693 { keys[5] = tmp_681; values[5] = tmp_682; }
    let tmp_694 = keys[6] < tmp_683 || (keys[6] == tmp_683 && values[6] < tmp_684);
    if tmp_687 == tmp_694 { keys[6] = tmp_683; values[6] = tmp_684; }
    let tmp_695 = keys[7] < tmp_685 || (keys[7] == tmp_685 && values[7] < tmp_686);
    if tmp_687 == tmp_695 { keys[7] = tmp_685; values[7] = tmp_686; }
    }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_696 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_696;let tmp_697 = values[0]; values[0] = values[4]; values[4] = tmp_697; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_698 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_698;let tmp_699 = values[1]; values[1] = values[5]; values[5] = tmp_699; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_700 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_700;let tmp_701 = values[2]; values[2] = values[6]; values[6] = tmp_701; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_702 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_702;let tmp_703 = values[3]; values[3] = values[7]; values[7] = tmp_703; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_704 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_704;let tmp_705 = values[0]; values[0] = values[2]; values[2] = tmp_705; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_706 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_706;let tmp_707 = values[1]; values[1] = values[3]; values[3] = tmp_707; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_708 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_708;let tmp_709 = values[4]; values[4] = values[6]; values[6] = tmp_709; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_710 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_710;let tmp_711 = values[5]; values[5] = values[7]; values[7] = tmp_711; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_712 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_712;let tmp_713 = values[0]; values[0] = values[1]; values[1] = tmp_713; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_714 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_714;let tmp_715 = values[2]; values[2] = values[3]; values[3] = tmp_715; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_716 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_716;let tmp_717 = values[4]; values[4] = values[5]; values[5] = tmp_717; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_718 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_718;let tmp_719 = values[6]; values[6] = values[7]; values[7] = tmp_719; }
    }
    // exch_intxn(tmask:127,swbit:6,wpt:8)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_720 = extractBits(local_tid, 6u, 1u) != 0u; let tmp_721 = seg_base + (local_tid ^ 127u); let tmp_722 = smem_keys[tmp_721 * WPT + 7u]; let tmp_723 = smem_vals[tmp_721 * WPT + 7u]; let tmp_724 = keys[0] < tmp_722 || (keys[0] == tmp_722 && values[0] < tmp_723); if tmp_720 == tmp_724 { keys[0] = tmp_722; values[0] = tmp_723; } let tmp_725 = smem_keys[tmp_721 * WPT + 6u]; let tmp_726 = smem_vals[tmp_721 * WPT + 6u]; let tmp_727 = keys[1] < tmp_725 || (keys[1] == tmp_725 && values[1] < tmp_726); if tmp_720 == tmp_727 { keys[1] = tmp_725; values[1] = tmp_726; } let tmp_728 = smem_keys[tmp_721 * WPT + 5u]; let tmp_729 = smem_vals[tmp_721 * WPT + 5u]; let tmp_730 = keys[2] < tmp_728 || (keys[2] == tmp_728 && values[2] < tmp_729); if tmp_720 == tmp_730 { keys[2] = tmp_728; values[2] = tmp_729; } let tmp_731 = smem_keys[tmp_721 * WPT + 4u]; let tmp_732 = smem_vals[tmp_721 * WPT + 4u]; let tmp_733 = keys[3] < tmp_731 || (keys[3] == tmp_731 && values[3] < tmp_732); if tmp_720 == tmp_733 { keys[3] = tmp_731; values[3] = tmp_732; } let tmp_734 = smem_keys[tmp_721 * WPT + 3u]; let tmp_735 = smem_vals[tmp_721 * WPT + 3u]; let tmp_736 = keys[4] < tmp_734 || (keys[4] == tmp_734 && values[4] < tmp_735); if tmp_720 == tmp_736 { keys[4] = tmp_734; values[4] = tmp_735; } let tmp_737 = smem_keys[tmp_721 * WPT + 2u]; let tmp_738 = smem_vals[tmp_721 * WPT + 2u]; let tmp_739 = keys[5] < tmp_737 || (keys[5] == tmp_737 && values[5] < tmp_738); if tmp_720 == tmp_739 { keys[5] = tmp_737; values[5] = tmp_738; } let tmp_740 = smem_keys[tmp_721 * WPT + 1u]; let tmp_741 = smem_vals[tmp_721 * WPT + 1u]; let tmp_742 = keys[6] < tmp_740 || (keys[6] == tmp_740 && values[6] < tmp_741); if tmp_720 == tmp_742 { keys[6] = tmp_740; values[6] = tmp_741; } let tmp_743 = smem_keys[tmp_721 * WPT + 0u]; let tmp_744 = smem_vals[tmp_721 * WPT + 0u]; let tmp_745 = keys[7] < tmp_743 || (keys[7] == tmp_743 && values[7] < tmp_744); if tmp_720 == tmp_745 { keys[7] = tmp_743; values[7] = tmp_744; } workgroupBarrier(); }
    // exch_paral(tmask:32,swbit:5,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_746 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_747 = seg_base + (local_tid ^ 32u); let tmp_748 = smem_keys[tmp_747 * WPT + 0u]; let tmp_749 = smem_vals[tmp_747 * WPT + 0u]; let tmp_750 = keys[0] < tmp_748 || (keys[0] == tmp_748 && values[0] < tmp_749); if tmp_746 == tmp_750 { keys[0] = tmp_748; values[0] = tmp_749; } let tmp_751 = smem_keys[tmp_747 * WPT + 1u]; let tmp_752 = smem_vals[tmp_747 * WPT + 1u]; let tmp_753 = keys[1] < tmp_751 || (keys[1] == tmp_751 && values[1] < tmp_752); if tmp_746 == tmp_753 { keys[1] = tmp_751; values[1] = tmp_752; } let tmp_754 = smem_keys[tmp_747 * WPT + 2u]; let tmp_755 = smem_vals[tmp_747 * WPT + 2u]; let tmp_756 = keys[2] < tmp_754 || (keys[2] == tmp_754 && values[2] < tmp_755); if tmp_746 == tmp_756 { keys[2] = tmp_754; values[2] = tmp_755; } let tmp_757 = smem_keys[tmp_747 * WPT + 3u]; let tmp_758 = smem_vals[tmp_747 * WPT + 3u]; let tmp_759 = keys[3] < tmp_757 || (keys[3] == tmp_757 && values[3] < tmp_758); if tmp_746 == tmp_759 { keys[3] = tmp_757; values[3] = tmp_758; } let tmp_760 = smem_keys[tmp_747 * WPT + 4u]; let tmp_761 = smem_vals[tmp_747 * WPT + 4u]; let tmp_762 = keys[4] < tmp_760 || (keys[4] == tmp_760 && values[4] < tmp_761); if tmp_746 == tmp_762 { keys[4] = tmp_760; values[4] = tmp_761; } let tmp_763 = smem_keys[tmp_747 * WPT + 5u]; let tmp_764 = smem_vals[tmp_747 * WPT + 5u]; let tmp_765 = keys[5] < tmp_763 || (keys[5] == tmp_763 && values[5] < tmp_764); if tmp_746 == tmp_765 { keys[5] = tmp_763; values[5] = tmp_764; } let tmp_766 = smem_keys[tmp_747 * WPT + 6u]; let tmp_767 = smem_vals[tmp_747 * WPT + 6u]; let tmp_768 = keys[6] < tmp_766 || (keys[6] == tmp_766 && values[6] < tmp_767); if tmp_746 == tmp_768 { keys[6] = tmp_766; values[6] = tmp_767; } let tmp_769 = smem_keys[tmp_747 * WPT + 7u]; let tmp_770 = smem_vals[tmp_747 * WPT + 7u]; let tmp_771 = keys[7] < tmp_769 || (keys[7] == tmp_769 && values[7] < tmp_770); if tmp_746 == tmp_771 { keys[7] = tmp_769; values[7] = tmp_770; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_772 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_773 = seg_base + (local_tid ^ 16u); let tmp_774 = smem_keys[tmp_773 * WPT + 0u]; let tmp_775 = smem_vals[tmp_773 * WPT + 0u]; let tmp_776 = keys[0] < tmp_774 || (keys[0] == tmp_774 && values[0] < tmp_775); if tmp_772 == tmp_776 { keys[0] = tmp_774; values[0] = tmp_775; } let tmp_777 = smem_keys[tmp_773 * WPT + 1u]; let tmp_778 = smem_vals[tmp_773 * WPT + 1u]; let tmp_779 = keys[1] < tmp_777 || (keys[1] == tmp_777 && values[1] < tmp_778); if tmp_772 == tmp_779 { keys[1] = tmp_777; values[1] = tmp_778; } let tmp_780 = smem_keys[tmp_773 * WPT + 2u]; let tmp_781 = smem_vals[tmp_773 * WPT + 2u]; let tmp_782 = keys[2] < tmp_780 || (keys[2] == tmp_780 && values[2] < tmp_781); if tmp_772 == tmp_782 { keys[2] = tmp_780; values[2] = tmp_781; } let tmp_783 = smem_keys[tmp_773 * WPT + 3u]; let tmp_784 = smem_vals[tmp_773 * WPT + 3u]; let tmp_785 = keys[3] < tmp_783 || (keys[3] == tmp_783 && values[3] < tmp_784); if tmp_772 == tmp_785 { keys[3] = tmp_783; values[3] = tmp_784; } let tmp_786 = smem_keys[tmp_773 * WPT + 4u]; let tmp_787 = smem_vals[tmp_773 * WPT + 4u]; let tmp_788 = keys[4] < tmp_786 || (keys[4] == tmp_786 && values[4] < tmp_787); if tmp_772 == tmp_788 { keys[4] = tmp_786; values[4] = tmp_787; } let tmp_789 = smem_keys[tmp_773 * WPT + 5u]; let tmp_790 = smem_vals[tmp_773 * WPT + 5u]; let tmp_791 = keys[5] < tmp_789 || (keys[5] == tmp_789 && values[5] < tmp_790); if tmp_772 == tmp_791 { keys[5] = tmp_789; values[5] = tmp_790; } let tmp_792 = smem_keys[tmp_773 * WPT + 6u]; let tmp_793 = smem_vals[tmp_773 * WPT + 6u]; let tmp_794 = keys[6] < tmp_792 || (keys[6] == tmp_792 && values[6] < tmp_793); if tmp_772 == tmp_794 { keys[6] = tmp_792; values[6] = tmp_793; } let tmp_795 = smem_keys[tmp_773 * WPT + 7u]; let tmp_796 = smem_vals[tmp_773 * WPT + 7u]; let tmp_797 = keys[7] < tmp_795 || (keys[7] == tmp_795 && values[7] < tmp_796); if tmp_772 == tmp_797 { keys[7] = tmp_795; values[7] = tmp_796; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:8) 
    {
    let tmp_798 = subgroupShuffleXor(keys[0], 8u);
    let tmp_799 = subgroupShuffleXor(values[0], 8u);
    let tmp_800 = subgroupShuffleXor(keys[1], 8u);
    let tmp_801 = subgroupShuffleXor(values[1], 8u);
    let tmp_802 = subgroupShuffleXor(keys[2], 8u);
    let tmp_803 = subgroupShuffleXor(values[2], 8u);
    let tmp_804 = subgroupShuffleXor(keys[3], 8u);
    let tmp_805 = subgroupShuffleXor(values[3], 8u);
    let tmp_806 = subgroupShuffleXor(keys[4], 8u);
    let tmp_807 = subgroupShuffleXor(values[4], 8u);
    let tmp_808 = subgroupShuffleXor(keys[5], 8u);
    let tmp_809 = subgroupShuffleXor(values[5], 8u);
    let tmp_810 = subgroupShuffleXor(keys[6], 8u);
    let tmp_811 = subgroupShuffleXor(values[6], 8u);
    let tmp_812 = subgroupShuffleXor(keys[7], 8u);
    let tmp_813 = subgroupShuffleXor(values[7], 8u);
    let tmp_814 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_815 = keys[0] < tmp_798 || (keys[0] == tmp_798 && values[0] < tmp_799);
    if tmp_814 == tmp_815 { keys[0] = tmp_798; values[0] = tmp_799; }
    let tmp_816 = keys[1] < tmp_800 || (keys[1] == tmp_800 && values[1] < tmp_801);
    if tmp_814 == tmp_816 { keys[1] = tmp_800; values[1] = tmp_801; }
    let tmp_817 = keys[2] < tmp_802 || (keys[2] == tmp_802 && values[2] < tmp_803);
    if tmp_814 == tmp_817 { keys[2] = tmp_802; values[2] = tmp_803; }
    let tmp_818 = keys[3] < tmp_804 || (keys[3] == tmp_804 && values[3] < tmp_805);
    if tmp_814 == tmp_818 { keys[3] = tmp_804; values[3] = tmp_805; }
    let tmp_819 = keys[4] < tmp_806 || (keys[4] == tmp_806 && values[4] < tmp_807);
    if tmp_814 == tmp_819 { keys[4] = tmp_806; values[4] = tmp_807; }
    let tmp_820 = keys[5] < tmp_808 || (keys[5] == tmp_808 && values[5] < tmp_809);
    if tmp_814 == tmp_820 { keys[5] = tmp_808; values[5] = tmp_809; }
    let tmp_821 = keys[6] < tmp_810 || (keys[6] == tmp_810 && values[6] < tmp_811);
    if tmp_814 == tmp_821 { keys[6] = tmp_810; values[6] = tmp_811; }
    let tmp_822 = keys[7] < tmp_812 || (keys[7] == tmp_812 && values[7] < tmp_813);
    if tmp_814 == tmp_822 { keys[7] = tmp_812; values[7] = tmp_813; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:8) 
    {
    let tmp_823 = subgroupShuffleXor(keys[0], 4u);
    let tmp_824 = subgroupShuffleXor(values[0], 4u);
    let tmp_825 = subgroupShuffleXor(keys[1], 4u);
    let tmp_826 = subgroupShuffleXor(values[1], 4u);
    let tmp_827 = subgroupShuffleXor(keys[2], 4u);
    let tmp_828 = subgroupShuffleXor(values[2], 4u);
    let tmp_829 = subgroupShuffleXor(keys[3], 4u);
    let tmp_830 = subgroupShuffleXor(values[3], 4u);
    let tmp_831 = subgroupShuffleXor(keys[4], 4u);
    let tmp_832 = subgroupShuffleXor(values[4], 4u);
    let tmp_833 = subgroupShuffleXor(keys[5], 4u);
    let tmp_834 = subgroupShuffleXor(values[5], 4u);
    let tmp_835 = subgroupShuffleXor(keys[6], 4u);
    let tmp_836 = subgroupShuffleXor(values[6], 4u);
    let tmp_837 = subgroupShuffleXor(keys[7], 4u);
    let tmp_838 = subgroupShuffleXor(values[7], 4u);
    let tmp_839 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_840 = keys[0] < tmp_823 || (keys[0] == tmp_823 && values[0] < tmp_824);
    if tmp_839 == tmp_840 { keys[0] = tmp_823; values[0] = tmp_824; }
    let tmp_841 = keys[1] < tmp_825 || (keys[1] == tmp_825 && values[1] < tmp_826);
    if tmp_839 == tmp_841 { keys[1] = tmp_825; values[1] = tmp_826; }
    let tmp_842 = keys[2] < tmp_827 || (keys[2] == tmp_827 && values[2] < tmp_828);
    if tmp_839 == tmp_842 { keys[2] = tmp_827; values[2] = tmp_828; }
    let tmp_843 = keys[3] < tmp_829 || (keys[3] == tmp_829 && values[3] < tmp_830);
    if tmp_839 == tmp_843 { keys[3] = tmp_829; values[3] = tmp_830; }
    let tmp_844 = keys[4] < tmp_831 || (keys[4] == tmp_831 && values[4] < tmp_832);
    if tmp_839 == tmp_844 { keys[4] = tmp_831; values[4] = tmp_832; }
    let tmp_845 = keys[5] < tmp_833 || (keys[5] == tmp_833 && values[5] < tmp_834);
    if tmp_839 == tmp_845 { keys[5] = tmp_833; values[5] = tmp_834; }
    let tmp_846 = keys[6] < tmp_835 || (keys[6] == tmp_835 && values[6] < tmp_836);
    if tmp_839 == tmp_846 { keys[6] = tmp_835; values[6] = tmp_836; }
    let tmp_847 = keys[7] < tmp_837 || (keys[7] == tmp_837 && values[7] < tmp_838);
    if tmp_839 == tmp_847 { keys[7] = tmp_837; values[7] = tmp_838; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    {
    let tmp_848 = subgroupShuffleXor(keys[0], 2u);
    let tmp_849 = subgroupShuffleXor(values[0], 2u);
    let tmp_850 = subgroupShuffleXor(keys[1], 2u);
    let tmp_851 = subgroupShuffleXor(values[1], 2u);
    let tmp_852 = subgroupShuffleXor(keys[2], 2u);
    let tmp_853 = subgroupShuffleXor(values[2], 2u);
    let tmp_854 = subgroupShuffleXor(keys[3], 2u);
    let tmp_855 = subgroupShuffleXor(values[3], 2u);
    let tmp_856 = subgroupShuffleXor(keys[4], 2u);
    let tmp_857 = subgroupShuffleXor(values[4], 2u);
    let tmp_858 = subgroupShuffleXor(keys[5], 2u);
    let tmp_859 = subgroupShuffleXor(values[5], 2u);
    let tmp_860 = subgroupShuffleXor(keys[6], 2u);
    let tmp_861 = subgroupShuffleXor(values[6], 2u);
    let tmp_862 = subgroupShuffleXor(keys[7], 2u);
    let tmp_863 = subgroupShuffleXor(values[7], 2u);
    let tmp_864 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_865 = keys[0] < tmp_848 || (keys[0] == tmp_848 && values[0] < tmp_849);
    if tmp_864 == tmp_865 { keys[0] = tmp_848; values[0] = tmp_849; }
    let tmp_866 = keys[1] < tmp_850 || (keys[1] == tmp_850 && values[1] < tmp_851);
    if tmp_864 == tmp_866 { keys[1] = tmp_850; values[1] = tmp_851; }
    let tmp_867 = keys[2] < tmp_852 || (keys[2] == tmp_852 && values[2] < tmp_853);
    if tmp_864 == tmp_867 { keys[2] = tmp_852; values[2] = tmp_853; }
    let tmp_868 = keys[3] < tmp_854 || (keys[3] == tmp_854 && values[3] < tmp_855);
    if tmp_864 == tmp_868 { keys[3] = tmp_854; values[3] = tmp_855; }
    let tmp_869 = keys[4] < tmp_856 || (keys[4] == tmp_856 && values[4] < tmp_857);
    if tmp_864 == tmp_869 { keys[4] = tmp_856; values[4] = tmp_857; }
    let tmp_870 = keys[5] < tmp_858 || (keys[5] == tmp_858 && values[5] < tmp_859);
    if tmp_864 == tmp_870 { keys[5] = tmp_858; values[5] = tmp_859; }
    let tmp_871 = keys[6] < tmp_860 || (keys[6] == tmp_860 && values[6] < tmp_861);
    if tmp_864 == tmp_871 { keys[6] = tmp_860; values[6] = tmp_861; }
    let tmp_872 = keys[7] < tmp_862 || (keys[7] == tmp_862 && values[7] < tmp_863);
    if tmp_864 == tmp_872 { keys[7] = tmp_862; values[7] = tmp_863; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    {
    let tmp_873 = subgroupShuffleXor(keys[0], 1u);
    let tmp_874 = subgroupShuffleXor(values[0], 1u);
    let tmp_875 = subgroupShuffleXor(keys[1], 1u);
    let tmp_876 = subgroupShuffleXor(values[1], 1u);
    let tmp_877 = subgroupShuffleXor(keys[2], 1u);
    let tmp_878 = subgroupShuffleXor(values[2], 1u);
    let tmp_879 = subgroupShuffleXor(keys[3], 1u);
    let tmp_880 = subgroupShuffleXor(values[3], 1u);
    let tmp_881 = subgroupShuffleXor(keys[4], 1u);
    let tmp_882 = subgroupShuffleXor(values[4], 1u);
    let tmp_883 = subgroupShuffleXor(keys[5], 1u);
    let tmp_884 = subgroupShuffleXor(values[5], 1u);
    let tmp_885 = subgroupShuffleXor(keys[6], 1u);
    let tmp_886 = subgroupShuffleXor(values[6], 1u);
    let tmp_887 = subgroupShuffleXor(keys[7], 1u);
    let tmp_888 = subgroupShuffleXor(values[7], 1u);
    let tmp_889 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_890 = keys[0] < tmp_873 || (keys[0] == tmp_873 && values[0] < tmp_874);
    if tmp_889 == tmp_890 { keys[0] = tmp_873; values[0] = tmp_874; }
    let tmp_891 = keys[1] < tmp_875 || (keys[1] == tmp_875 && values[1] < tmp_876);
    if tmp_889 == tmp_891 { keys[1] = tmp_875; values[1] = tmp_876; }
    let tmp_892 = keys[2] < tmp_877 || (keys[2] == tmp_877 && values[2] < tmp_878);
    if tmp_889 == tmp_892 { keys[2] = tmp_877; values[2] = tmp_878; }
    let tmp_893 = keys[3] < tmp_879 || (keys[3] == tmp_879 && values[3] < tmp_880);
    if tmp_889 == tmp_893 { keys[3] = tmp_879; values[3] = tmp_880; }
    let tmp_894 = keys[4] < tmp_881 || (keys[4] == tmp_881 && values[4] < tmp_882);
    if tmp_889 == tmp_894 { keys[4] = tmp_881; values[4] = tmp_882; }
    let tmp_895 = keys[5] < tmp_883 || (keys[5] == tmp_883 && values[5] < tmp_884);
    if tmp_889 == tmp_895 { keys[5] = tmp_883; values[5] = tmp_884; }
    let tmp_896 = keys[6] < tmp_885 || (keys[6] == tmp_885 && values[6] < tmp_886);
    if tmp_889 == tmp_896 { keys[6] = tmp_885; values[6] = tmp_886; }
    let tmp_897 = keys[7] < tmp_887 || (keys[7] == tmp_887 && values[7] < tmp_888);
    if tmp_889 == tmp_897 { keys[7] = tmp_887; values[7] = tmp_888; }
    }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_898 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_898;let tmp_899 = values[0]; values[0] = values[4]; values[4] = tmp_899; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_900 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_900;let tmp_901 = values[1]; values[1] = values[5]; values[5] = tmp_901; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_902 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_902;let tmp_903 = values[2]; values[2] = values[6]; values[6] = tmp_903; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_904 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_904;let tmp_905 = values[3]; values[3] = values[7]; values[7] = tmp_905; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_906 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_906;let tmp_907 = values[0]; values[0] = values[2]; values[2] = tmp_907; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_908 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_908;let tmp_909 = values[1]; values[1] = values[3]; values[3] = tmp_909; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_910 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_910;let tmp_911 = values[4]; values[4] = values[6]; values[6] = tmp_911; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_912 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_912;let tmp_913 = values[5]; values[5] = values[7]; values[7] = tmp_913; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_914 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_914;let tmp_915 = values[0]; values[0] = values[1]; values[1] = tmp_915; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_916 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_916;let tmp_917 = values[2]; values[2] = values[3]; values[3] = tmp_917; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_918 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_918;let tmp_919 = values[4]; values[4] = values[5]; values[5] = tmp_919; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_920 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_920;let tmp_921 = values[6]; values[6] = values[7]; values[7] = tmp_921; }
    }
    // exch_intxn(tmask:255,swbit:7,wpt:8)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_922 = extractBits(local_tid, 7u, 1u) != 0u; let tmp_923 = seg_base + (local_tid ^ 255u); let tmp_924 = smem_keys[tmp_923 * WPT + 7u]; let tmp_925 = smem_vals[tmp_923 * WPT + 7u]; let tmp_926 = keys[0] < tmp_924 || (keys[0] == tmp_924 && values[0] < tmp_925); if tmp_922 == tmp_926 { keys[0] = tmp_924; values[0] = tmp_925; } let tmp_927 = smem_keys[tmp_923 * WPT + 6u]; let tmp_928 = smem_vals[tmp_923 * WPT + 6u]; let tmp_929 = keys[1] < tmp_927 || (keys[1] == tmp_927 && values[1] < tmp_928); if tmp_922 == tmp_929 { keys[1] = tmp_927; values[1] = tmp_928; } let tmp_930 = smem_keys[tmp_923 * WPT + 5u]; let tmp_931 = smem_vals[tmp_923 * WPT + 5u]; let tmp_932 = keys[2] < tmp_930 || (keys[2] == tmp_930 && values[2] < tmp_931); if tmp_922 == tmp_932 { keys[2] = tmp_930; values[2] = tmp_931; } let tmp_933 = smem_keys[tmp_923 * WPT + 4u]; let tmp_934 = smem_vals[tmp_923 * WPT + 4u]; let tmp_935 = keys[3] < tmp_933 || (keys[3] == tmp_933 && values[3] < tmp_934); if tmp_922 == tmp_935 { keys[3] = tmp_933; values[3] = tmp_934; } let tmp_936 = smem_keys[tmp_923 * WPT + 3u]; let tmp_937 = smem_vals[tmp_923 * WPT + 3u]; let tmp_938 = keys[4] < tmp_936 || (keys[4] == tmp_936 && values[4] < tmp_937); if tmp_922 == tmp_938 { keys[4] = tmp_936; values[4] = tmp_937; } let tmp_939 = smem_keys[tmp_923 * WPT + 2u]; let tmp_940 = smem_vals[tmp_923 * WPT + 2u]; let tmp_941 = keys[5] < tmp_939 || (keys[5] == tmp_939 && values[5] < tmp_940); if tmp_922 == tmp_941 { keys[5] = tmp_939; values[5] = tmp_940; } let tmp_942 = smem_keys[tmp_923 * WPT + 1u]; let tmp_943 = smem_vals[tmp_923 * WPT + 1u]; let tmp_944 = keys[6] < tmp_942 || (keys[6] == tmp_942 && values[6] < tmp_943); if tmp_922 == tmp_944 { keys[6] = tmp_942; values[6] = tmp_943; } let tmp_945 = smem_keys[tmp_923 * WPT + 0u]; let tmp_946 = smem_vals[tmp_923 * WPT + 0u]; let tmp_947 = keys[7] < tmp_945 || (keys[7] == tmp_945 && values[7] < tmp_946); if tmp_922 == tmp_947 { keys[7] = tmp_945; values[7] = tmp_946; } workgroupBarrier(); }
    // exch_paral(tmask:64,swbit:6,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_948 = extractBits(local_tid, 6u, 1u) != 0u; let tmp_949 = seg_base + (local_tid ^ 64u); let tmp_950 = smem_keys[tmp_949 * WPT + 0u]; let tmp_951 = smem_vals[tmp_949 * WPT + 0u]; let tmp_952 = keys[0] < tmp_950 || (keys[0] == tmp_950 && values[0] < tmp_951); if tmp_948 == tmp_952 { keys[0] = tmp_950; values[0] = tmp_951; } let tmp_953 = smem_keys[tmp_949 * WPT + 1u]; let tmp_954 = smem_vals[tmp_949 * WPT + 1u]; let tmp_955 = keys[1] < tmp_953 || (keys[1] == tmp_953 && values[1] < tmp_954); if tmp_948 == tmp_955 { keys[1] = tmp_953; values[1] = tmp_954; } let tmp_956 = smem_keys[tmp_949 * WPT + 2u]; let tmp_957 = smem_vals[tmp_949 * WPT + 2u]; let tmp_958 = keys[2] < tmp_956 || (keys[2] == tmp_956 && values[2] < tmp_957); if tmp_948 == tmp_958 { keys[2] = tmp_956; values[2] = tmp_957; } let tmp_959 = smem_keys[tmp_949 * WPT + 3u]; let tmp_960 = smem_vals[tmp_949 * WPT + 3u]; let tmp_961 = keys[3] < tmp_959 || (keys[3] == tmp_959 && values[3] < tmp_960); if tmp_948 == tmp_961 { keys[3] = tmp_959; values[3] = tmp_960; } let tmp_962 = smem_keys[tmp_949 * WPT + 4u]; let tmp_963 = smem_vals[tmp_949 * WPT + 4u]; let tmp_964 = keys[4] < tmp_962 || (keys[4] == tmp_962 && values[4] < tmp_963); if tmp_948 == tmp_964 { keys[4] = tmp_962; values[4] = tmp_963; } let tmp_965 = smem_keys[tmp_949 * WPT + 5u]; let tmp_966 = smem_vals[tmp_949 * WPT + 5u]; let tmp_967 = keys[5] < tmp_965 || (keys[5] == tmp_965 && values[5] < tmp_966); if tmp_948 == tmp_967 { keys[5] = tmp_965; values[5] = tmp_966; } let tmp_968 = smem_keys[tmp_949 * WPT + 6u]; let tmp_969 = smem_vals[tmp_949 * WPT + 6u]; let tmp_970 = keys[6] < tmp_968 || (keys[6] == tmp_968 && values[6] < tmp_969); if tmp_948 == tmp_970 { keys[6] = tmp_968; values[6] = tmp_969; } let tmp_971 = smem_keys[tmp_949 * WPT + 7u]; let tmp_972 = smem_vals[tmp_949 * WPT + 7u]; let tmp_973 = keys[7] < tmp_971 || (keys[7] == tmp_971 && values[7] < tmp_972); if tmp_948 == tmp_973 { keys[7] = tmp_971; values[7] = tmp_972; } workgroupBarrier(); }
    // exch_paral(tmask:32,swbit:5,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_974 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_975 = seg_base + (local_tid ^ 32u); let tmp_976 = smem_keys[tmp_975 * WPT + 0u]; let tmp_977 = smem_vals[tmp_975 * WPT + 0u]; let tmp_978 = keys[0] < tmp_976 || (keys[0] == tmp_976 && values[0] < tmp_977); if tmp_974 == tmp_978 { keys[0] = tmp_976; values[0] = tmp_977; } let tmp_979 = smem_keys[tmp_975 * WPT + 1u]; let tmp_980 = smem_vals[tmp_975 * WPT + 1u]; let tmp_981 = keys[1] < tmp_979 || (keys[1] == tmp_979 && values[1] < tmp_980); if tmp_974 == tmp_981 { keys[1] = tmp_979; values[1] = tmp_980; } let tmp_982 = smem_keys[tmp_975 * WPT + 2u]; let tmp_983 = smem_vals[tmp_975 * WPT + 2u]; let tmp_984 = keys[2] < tmp_982 || (keys[2] == tmp_982 && values[2] < tmp_983); if tmp_974 == tmp_984 { keys[2] = tmp_982; values[2] = tmp_983; } let tmp_985 = smem_keys[tmp_975 * WPT + 3u]; let tmp_986 = smem_vals[tmp_975 * WPT + 3u]; let tmp_987 = keys[3] < tmp_985 || (keys[3] == tmp_985 && values[3] < tmp_986); if tmp_974 == tmp_987 { keys[3] = tmp_985; values[3] = tmp_986; } let tmp_988 = smem_keys[tmp_975 * WPT + 4u]; let tmp_989 = smem_vals[tmp_975 * WPT + 4u]; let tmp_990 = keys[4] < tmp_988 || (keys[4] == tmp_988 && values[4] < tmp_989); if tmp_974 == tmp_990 { keys[4] = tmp_988; values[4] = tmp_989; } let tmp_991 = smem_keys[tmp_975 * WPT + 5u]; let tmp_992 = smem_vals[tmp_975 * WPT + 5u]; let tmp_993 = keys[5] < tmp_991 || (keys[5] == tmp_991 && values[5] < tmp_992); if tmp_974 == tmp_993 { keys[5] = tmp_991; values[5] = tmp_992; } let tmp_994 = smem_keys[tmp_975 * WPT + 6u]; let tmp_995 = smem_vals[tmp_975 * WPT + 6u]; let tmp_996 = keys[6] < tmp_994 || (keys[6] == tmp_994 && values[6] < tmp_995); if tmp_974 == tmp_996 { keys[6] = tmp_994; values[6] = tmp_995; } let tmp_997 = smem_keys[tmp_975 * WPT + 7u]; let tmp_998 = smem_vals[tmp_975 * WPT + 7u]; let tmp_999 = keys[7] < tmp_997 || (keys[7] == tmp_997 && values[7] < tmp_998); if tmp_974 == tmp_999 { keys[7] = tmp_997; values[7] = tmp_998; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_1000 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_1001 = seg_base + (local_tid ^ 16u); let tmp_1002 = smem_keys[tmp_1001 * WPT + 0u]; let tmp_1003 = smem_vals[tmp_1001 * WPT + 0u]; let tmp_1004 = keys[0] < tmp_1002 || (keys[0] == tmp_1002 && values[0] < tmp_1003); if tmp_1000 == tmp_1004 { keys[0] = tmp_1002; values[0] = tmp_1003; } let tmp_1005 = smem_keys[tmp_1001 * WPT + 1u]; let tmp_1006 = smem_vals[tmp_1001 * WPT + 1u]; let tmp_1007 = keys[1] < tmp_1005 || (keys[1] == tmp_1005 && values[1] < tmp_1006); if tmp_1000 == tmp_1007 { keys[1] = tmp_1005; values[1] = tmp_1006; } let tmp_1008 = smem_keys[tmp_1001 * WPT + 2u]; let tmp_1009 = smem_vals[tmp_1001 * WPT + 2u]; let tmp_1010 = keys[2] < tmp_1008 || (keys[2] == tmp_1008 && values[2] < tmp_1009); if tmp_1000 == tmp_1010 { keys[2] = tmp_1008; values[2] = tmp_1009; } let tmp_1011 = smem_keys[tmp_1001 * WPT + 3u]; let tmp_1012 = smem_vals[tmp_1001 * WPT + 3u]; let tmp_1013 = keys[3] < tmp_1011 || (keys[3] == tmp_1011 && values[3] < tmp_1012); if tmp_1000 == tmp_1013 { keys[3] = tmp_1011; values[3] = tmp_1012; } let tmp_1014 = smem_keys[tmp_1001 * WPT + 4u]; let tmp_1015 = smem_vals[tmp_1001 * WPT + 4u]; let tmp_1016 = keys[4] < tmp_1014 || (keys[4] == tmp_1014 && values[4] < tmp_1015); if tmp_1000 == tmp_1016 { keys[4] = tmp_1014; values[4] = tmp_1015; } let tmp_1017 = smem_keys[tmp_1001 * WPT + 5u]; let tmp_1018 = smem_vals[tmp_1001 * WPT + 5u]; let tmp_1019 = keys[5] < tmp_1017 || (keys[5] == tmp_1017 && values[5] < tmp_1018); if tmp_1000 == tmp_1019 { keys[5] = tmp_1017; values[5] = tmp_1018; } let tmp_1020 = smem_keys[tmp_1001 * WPT + 6u]; let tmp_1021 = smem_vals[tmp_1001 * WPT + 6u]; let tmp_1022 = keys[6] < tmp_1020 || (keys[6] == tmp_1020 && values[6] < tmp_1021); if tmp_1000 == tmp_1022 { keys[6] = tmp_1020; values[6] = tmp_1021; } let tmp_1023 = smem_keys[tmp_1001 * WPT + 7u]; let tmp_1024 = smem_vals[tmp_1001 * WPT + 7u]; let tmp_1025 = keys[7] < tmp_1023 || (keys[7] == tmp_1023 && values[7] < tmp_1024); if tmp_1000 == tmp_1025 { keys[7] = tmp_1023; values[7] = tmp_1024; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:8) 
    {
    let tmp_1026 = subgroupShuffleXor(keys[0], 8u);
    let tmp_1027 = subgroupShuffleXor(values[0], 8u);
    let tmp_1028 = subgroupShuffleXor(keys[1], 8u);
    let tmp_1029 = subgroupShuffleXor(values[1], 8u);
    let tmp_1030 = subgroupShuffleXor(keys[2], 8u);
    let tmp_1031 = subgroupShuffleXor(values[2], 8u);
    let tmp_1032 = subgroupShuffleXor(keys[3], 8u);
    let tmp_1033 = subgroupShuffleXor(values[3], 8u);
    let tmp_1034 = subgroupShuffleXor(keys[4], 8u);
    let tmp_1035 = subgroupShuffleXor(values[4], 8u);
    let tmp_1036 = subgroupShuffleXor(keys[5], 8u);
    let tmp_1037 = subgroupShuffleXor(values[5], 8u);
    let tmp_1038 = subgroupShuffleXor(keys[6], 8u);
    let tmp_1039 = subgroupShuffleXor(values[6], 8u);
    let tmp_1040 = subgroupShuffleXor(keys[7], 8u);
    let tmp_1041 = subgroupShuffleXor(values[7], 8u);
    let tmp_1042 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_1043 = keys[0] < tmp_1026 || (keys[0] == tmp_1026 && values[0] < tmp_1027);
    if tmp_1042 == tmp_1043 { keys[0] = tmp_1026; values[0] = tmp_1027; }
    let tmp_1044 = keys[1] < tmp_1028 || (keys[1] == tmp_1028 && values[1] < tmp_1029);
    if tmp_1042 == tmp_1044 { keys[1] = tmp_1028; values[1] = tmp_1029; }
    let tmp_1045 = keys[2] < tmp_1030 || (keys[2] == tmp_1030 && values[2] < tmp_1031);
    if tmp_1042 == tmp_1045 { keys[2] = tmp_1030; values[2] = tmp_1031; }
    let tmp_1046 = keys[3] < tmp_1032 || (keys[3] == tmp_1032 && values[3] < tmp_1033);
    if tmp_1042 == tmp_1046 { keys[3] = tmp_1032; values[3] = tmp_1033; }
    let tmp_1047 = keys[4] < tmp_1034 || (keys[4] == tmp_1034 && values[4] < tmp_1035);
    if tmp_1042 == tmp_1047 { keys[4] = tmp_1034; values[4] = tmp_1035; }
    let tmp_1048 = keys[5] < tmp_1036 || (keys[5] == tmp_1036 && values[5] < tmp_1037);
    if tmp_1042 == tmp_1048 { keys[5] = tmp_1036; values[5] = tmp_1037; }
    let tmp_1049 = keys[6] < tmp_1038 || (keys[6] == tmp_1038 && values[6] < tmp_1039);
    if tmp_1042 == tmp_1049 { keys[6] = tmp_1038; values[6] = tmp_1039; }
    let tmp_1050 = keys[7] < tmp_1040 || (keys[7] == tmp_1040 && values[7] < tmp_1041);
    if tmp_1042 == tmp_1050 { keys[7] = tmp_1040; values[7] = tmp_1041; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:8) 
    {
    let tmp_1051 = subgroupShuffleXor(keys[0], 4u);
    let tmp_1052 = subgroupShuffleXor(values[0], 4u);
    let tmp_1053 = subgroupShuffleXor(keys[1], 4u);
    let tmp_1054 = subgroupShuffleXor(values[1], 4u);
    let tmp_1055 = subgroupShuffleXor(keys[2], 4u);
    let tmp_1056 = subgroupShuffleXor(values[2], 4u);
    let tmp_1057 = subgroupShuffleXor(keys[3], 4u);
    let tmp_1058 = subgroupShuffleXor(values[3], 4u);
    let tmp_1059 = subgroupShuffleXor(keys[4], 4u);
    let tmp_1060 = subgroupShuffleXor(values[4], 4u);
    let tmp_1061 = subgroupShuffleXor(keys[5], 4u);
    let tmp_1062 = subgroupShuffleXor(values[5], 4u);
    let tmp_1063 = subgroupShuffleXor(keys[6], 4u);
    let tmp_1064 = subgroupShuffleXor(values[6], 4u);
    let tmp_1065 = subgroupShuffleXor(keys[7], 4u);
    let tmp_1066 = subgroupShuffleXor(values[7], 4u);
    let tmp_1067 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_1068 = keys[0] < tmp_1051 || (keys[0] == tmp_1051 && values[0] < tmp_1052);
    if tmp_1067 == tmp_1068 { keys[0] = tmp_1051; values[0] = tmp_1052; }
    let tmp_1069 = keys[1] < tmp_1053 || (keys[1] == tmp_1053 && values[1] < tmp_1054);
    if tmp_1067 == tmp_1069 { keys[1] = tmp_1053; values[1] = tmp_1054; }
    let tmp_1070 = keys[2] < tmp_1055 || (keys[2] == tmp_1055 && values[2] < tmp_1056);
    if tmp_1067 == tmp_1070 { keys[2] = tmp_1055; values[2] = tmp_1056; }
    let tmp_1071 = keys[3] < tmp_1057 || (keys[3] == tmp_1057 && values[3] < tmp_1058);
    if tmp_1067 == tmp_1071 { keys[3] = tmp_1057; values[3] = tmp_1058; }
    let tmp_1072 = keys[4] < tmp_1059 || (keys[4] == tmp_1059 && values[4] < tmp_1060);
    if tmp_1067 == tmp_1072 { keys[4] = tmp_1059; values[4] = tmp_1060; }
    let tmp_1073 = keys[5] < tmp_1061 || (keys[5] == tmp_1061 && values[5] < tmp_1062);
    if tmp_1067 == tmp_1073 { keys[5] = tmp_1061; values[5] = tmp_1062; }
    let tmp_1074 = keys[6] < tmp_1063 || (keys[6] == tmp_1063 && values[6] < tmp_1064);
    if tmp_1067 == tmp_1074 { keys[6] = tmp_1063; values[6] = tmp_1064; }
    let tmp_1075 = keys[7] < tmp_1065 || (keys[7] == tmp_1065 && values[7] < tmp_1066);
    if tmp_1067 == tmp_1075 { keys[7] = tmp_1065; values[7] = tmp_1066; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    {
    let tmp_1076 = subgroupShuffleXor(keys[0], 2u);
    let tmp_1077 = subgroupShuffleXor(values[0], 2u);
    let tmp_1078 = subgroupShuffleXor(keys[1], 2u);
    let tmp_1079 = subgroupShuffleXor(values[1], 2u);
    let tmp_1080 = subgroupShuffleXor(keys[2], 2u);
    let tmp_1081 = subgroupShuffleXor(values[2], 2u);
    let tmp_1082 = subgroupShuffleXor(keys[3], 2u);
    let tmp_1083 = subgroupShuffleXor(values[3], 2u);
    let tmp_1084 = subgroupShuffleXor(keys[4], 2u);
    let tmp_1085 = subgroupShuffleXor(values[4], 2u);
    let tmp_1086 = subgroupShuffleXor(keys[5], 2u);
    let tmp_1087 = subgroupShuffleXor(values[5], 2u);
    let tmp_1088 = subgroupShuffleXor(keys[6], 2u);
    let tmp_1089 = subgroupShuffleXor(values[6], 2u);
    let tmp_1090 = subgroupShuffleXor(keys[7], 2u);
    let tmp_1091 = subgroupShuffleXor(values[7], 2u);
    let tmp_1092 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_1093 = keys[0] < tmp_1076 || (keys[0] == tmp_1076 && values[0] < tmp_1077);
    if tmp_1092 == tmp_1093 { keys[0] = tmp_1076; values[0] = tmp_1077; }
    let tmp_1094 = keys[1] < tmp_1078 || (keys[1] == tmp_1078 && values[1] < tmp_1079);
    if tmp_1092 == tmp_1094 { keys[1] = tmp_1078; values[1] = tmp_1079; }
    let tmp_1095 = keys[2] < tmp_1080 || (keys[2] == tmp_1080 && values[2] < tmp_1081);
    if tmp_1092 == tmp_1095 { keys[2] = tmp_1080; values[2] = tmp_1081; }
    let tmp_1096 = keys[3] < tmp_1082 || (keys[3] == tmp_1082 && values[3] < tmp_1083);
    if tmp_1092 == tmp_1096 { keys[3] = tmp_1082; values[3] = tmp_1083; }
    let tmp_1097 = keys[4] < tmp_1084 || (keys[4] == tmp_1084 && values[4] < tmp_1085);
    if tmp_1092 == tmp_1097 { keys[4] = tmp_1084; values[4] = tmp_1085; }
    let tmp_1098 = keys[5] < tmp_1086 || (keys[5] == tmp_1086 && values[5] < tmp_1087);
    if tmp_1092 == tmp_1098 { keys[5] = tmp_1086; values[5] = tmp_1087; }
    let tmp_1099 = keys[6] < tmp_1088 || (keys[6] == tmp_1088 && values[6] < tmp_1089);
    if tmp_1092 == tmp_1099 { keys[6] = tmp_1088; values[6] = tmp_1089; }
    let tmp_1100 = keys[7] < tmp_1090 || (keys[7] == tmp_1090 && values[7] < tmp_1091);
    if tmp_1092 == tmp_1100 { keys[7] = tmp_1090; values[7] = tmp_1091; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    {
    let tmp_1101 = subgroupShuffleXor(keys[0], 1u);
    let tmp_1102 = subgroupShuffleXor(values[0], 1u);
    let tmp_1103 = subgroupShuffleXor(keys[1], 1u);
    let tmp_1104 = subgroupShuffleXor(values[1], 1u);
    let tmp_1105 = subgroupShuffleXor(keys[2], 1u);
    let tmp_1106 = subgroupShuffleXor(values[2], 1u);
    let tmp_1107 = subgroupShuffleXor(keys[3], 1u);
    let tmp_1108 = subgroupShuffleXor(values[3], 1u);
    let tmp_1109 = subgroupShuffleXor(keys[4], 1u);
    let tmp_1110 = subgroupShuffleXor(values[4], 1u);
    let tmp_1111 = subgroupShuffleXor(keys[5], 1u);
    let tmp_1112 = subgroupShuffleXor(values[5], 1u);
    let tmp_1113 = subgroupShuffleXor(keys[6], 1u);
    let tmp_1114 = subgroupShuffleXor(values[6], 1u);
    let tmp_1115 = subgroupShuffleXor(keys[7], 1u);
    let tmp_1116 = subgroupShuffleXor(values[7], 1u);
    let tmp_1117 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_1118 = keys[0] < tmp_1101 || (keys[0] == tmp_1101 && values[0] < tmp_1102);
    if tmp_1117 == tmp_1118 { keys[0] = tmp_1101; values[0] = tmp_1102; }
    let tmp_1119 = keys[1] < tmp_1103 || (keys[1] == tmp_1103 && values[1] < tmp_1104);
    if tmp_1117 == tmp_1119 { keys[1] = tmp_1103; values[1] = tmp_1104; }
    let tmp_1120 = keys[2] < tmp_1105 || (keys[2] == tmp_1105 && values[2] < tmp_1106);
    if tmp_1117 == tmp_1120 { keys[2] = tmp_1105; values[2] = tmp_1106; }
    let tmp_1121 = keys[3] < tmp_1107 || (keys[3] == tmp_1107 && values[3] < tmp_1108);
    if tmp_1117 == tmp_1121 { keys[3] = tmp_1107; values[3] = tmp_1108; }
    let tmp_1122 = keys[4] < tmp_1109 || (keys[4] == tmp_1109 && values[4] < tmp_1110);
    if tmp_1117 == tmp_1122 { keys[4] = tmp_1109; values[4] = tmp_1110; }
    let tmp_1123 = keys[5] < tmp_1111 || (keys[5] == tmp_1111 && values[5] < tmp_1112);
    if tmp_1117 == tmp_1123 { keys[5] = tmp_1111; values[5] = tmp_1112; }
    let tmp_1124 = keys[6] < tmp_1113 || (keys[6] == tmp_1113 && values[6] < tmp_1114);
    if tmp_1117 == tmp_1124 { keys[6] = tmp_1113; values[6] = tmp_1114; }
    let tmp_1125 = keys[7] < tmp_1115 || (keys[7] == tmp_1115 && values[7] < tmp_1116);
    if tmp_1117 == tmp_1125 { keys[7] = tmp_1115; values[7] = tmp_1116; }
    }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_1126 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_1126;let tmp_1127 = values[0]; values[0] = values[4]; values[4] = tmp_1127; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_1128 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_1128;let tmp_1129 = values[1]; values[1] = values[5]; values[5] = tmp_1129; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_1130 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_1130;let tmp_1131 = values[2]; values[2] = values[6]; values[6] = tmp_1131; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_1132 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_1132;let tmp_1133 = values[3]; values[3] = values[7]; values[7] = tmp_1133; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_1134 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_1134;let tmp_1135 = values[0]; values[0] = values[2]; values[2] = tmp_1135; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_1136 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_1136;let tmp_1137 = values[1]; values[1] = values[3]; values[3] = tmp_1137; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_1138 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_1138;let tmp_1139 = values[4]; values[4] = values[6]; values[6] = tmp_1139; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_1140 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_1140;let tmp_1141 = values[5]; values[5] = values[7]; values[7] = tmp_1141; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_1142 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_1142;let tmp_1143 = values[0]; values[0] = values[1]; values[1] = tmp_1143; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_1144 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_1144;let tmp_1145 = values[2]; values[2] = values[3]; values[3] = tmp_1145; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_1146 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_1146;let tmp_1147 = values[4]; values[4] = values[5]; values[5] = tmp_1147; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_1148 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_1148;let tmp_1149 = values[6]; values[6] = values[7]; values[7] = tmp_1149; }
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
