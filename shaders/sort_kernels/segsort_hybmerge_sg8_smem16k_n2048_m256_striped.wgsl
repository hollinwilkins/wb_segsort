
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
const SG: u32 = 8u;      // register run spans one subgroup: RUN = SG*WPT = 64

var<workgroup> smem_keys: array<u32, N>;
var<workgroup> smem_vals: array<u32, N>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybmerge_sg8_smem16k_n2048_m256_striped(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 11u;

    let bin_base = select(bin_offsets[BIN - 1u], 0u, BIN == 0u);
    let bin_count = bin_offsets[BIN] - bin_base;

    let local_tid = tid_g % M;
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

    // ---- phase 1: per-subgroup register run-sort (RUN = SG*WPT elements) ----
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

    // stage the sorted runs into shared memory (blocked layout)
    for (var r = 0u; r < WPT; r = r + 1u) {
        smem_keys[local_tid * WPT + r] = keys[r];
        smem_vals[local_tid * WPT + r] = values[r];
    }
    workgroupBarrier();

    let base = local_tid * WPT;   // this thread's blocked output range [base, base+WPT)

    // ---- phase 2: recursive merge-path merges through shared memory ----
    // merge pass 0: two sorted runs of 64 -> 128 (register-staged)
    {
        let group_base = (base / 128u) * 128u;
        let diag = base - group_base;
        let a_base = group_base;
        let b_base = group_base + 64u;
        // merge-path: binary search the diagonal for this thread's A/B split
        var lo = select(0u, diag - 64u, diag > 64u);
        var hi = min(diag, 64u);
        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            let ak = smem_keys[a_base + mid];
            let av = smem_vals[a_base + mid];
            let bpos = b_base + (diag - 1u - mid);
            let bk = smem_keys[bpos];
            let bv = smem_vals[bpos];
            if ak < bk || (ak == bk && av <= bv) { lo = mid + 1u; } else { hi = mid; }
        }
        var ai = lo;
        var bi = diag - lo;
        // merge this thread's WPT outputs into registers (the pong)
        var out_keys: array<u32, 8>;
        var out_vals: array<u32, 8>;
        for (var k = 0u; k < WPT; k = k + 1u) {
            let take_a = bi >= 64u || (ai < 64u &&
                (smem_keys[a_base + ai] < smem_keys[b_base + bi] ||
                 (smem_keys[a_base + ai] == smem_keys[b_base + bi] &&
                  smem_vals[a_base + ai] <= smem_vals[b_base + bi])));
            if take_a {
                out_keys[k] = smem_keys[a_base + ai];
                out_vals[k] = smem_vals[a_base + ai];
                ai = ai + 1u;
            } else {
                out_keys[k] = smem_keys[b_base + bi];
                out_vals[k] = smem_vals[b_base + bi];
                bi = bi + 1u;
            }
        }
        workgroupBarrier();   // every read is done before any write-back
        storageBarrier();     // device-scope fence: workgroupBarrier alone under-orders
                              // the in-place write-back for single-SIMD-group WGs
        for (var k = 0u; k < WPT; k = k + 1u) {
            smem_keys[base + k] = out_keys[k];
            smem_vals[base + k] = out_vals[k];
        }
    }
    workgroupBarrier();
    // merge pass 1: two sorted runs of 128 -> 256 (register-staged)
    {
        let group_base = (base / 256u) * 256u;
        let diag = base - group_base;
        let a_base = group_base;
        let b_base = group_base + 128u;
        // merge-path: binary search the diagonal for this thread's A/B split
        var lo = select(0u, diag - 128u, diag > 128u);
        var hi = min(diag, 128u);
        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            let ak = smem_keys[a_base + mid];
            let av = smem_vals[a_base + mid];
            let bpos = b_base + (diag - 1u - mid);
            let bk = smem_keys[bpos];
            let bv = smem_vals[bpos];
            if ak < bk || (ak == bk && av <= bv) { lo = mid + 1u; } else { hi = mid; }
        }
        var ai = lo;
        var bi = diag - lo;
        // merge this thread's WPT outputs into registers (the pong)
        var out_keys: array<u32, 8>;
        var out_vals: array<u32, 8>;
        for (var k = 0u; k < WPT; k = k + 1u) {
            let take_a = bi >= 128u || (ai < 128u &&
                (smem_keys[a_base + ai] < smem_keys[b_base + bi] ||
                 (smem_keys[a_base + ai] == smem_keys[b_base + bi] &&
                  smem_vals[a_base + ai] <= smem_vals[b_base + bi])));
            if take_a {
                out_keys[k] = smem_keys[a_base + ai];
                out_vals[k] = smem_vals[a_base + ai];
                ai = ai + 1u;
            } else {
                out_keys[k] = smem_keys[b_base + bi];
                out_vals[k] = smem_vals[b_base + bi];
                bi = bi + 1u;
            }
        }
        workgroupBarrier();   // every read is done before any write-back
        storageBarrier();     // device-scope fence: workgroupBarrier alone under-orders
                              // the in-place write-back for single-SIMD-group WGs
        for (var k = 0u; k < WPT; k = k + 1u) {
            smem_keys[base + k] = out_keys[k];
            smem_vals[base + k] = out_vals[k];
        }
    }
    workgroupBarrier();
    // merge pass 2: two sorted runs of 256 -> 512 (register-staged)
    {
        let group_base = (base / 512u) * 512u;
        let diag = base - group_base;
        let a_base = group_base;
        let b_base = group_base + 256u;
        // merge-path: binary search the diagonal for this thread's A/B split
        var lo = select(0u, diag - 256u, diag > 256u);
        var hi = min(diag, 256u);
        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            let ak = smem_keys[a_base + mid];
            let av = smem_vals[a_base + mid];
            let bpos = b_base + (diag - 1u - mid);
            let bk = smem_keys[bpos];
            let bv = smem_vals[bpos];
            if ak < bk || (ak == bk && av <= bv) { lo = mid + 1u; } else { hi = mid; }
        }
        var ai = lo;
        var bi = diag - lo;
        // merge this thread's WPT outputs into registers (the pong)
        var out_keys: array<u32, 8>;
        var out_vals: array<u32, 8>;
        for (var k = 0u; k < WPT; k = k + 1u) {
            let take_a = bi >= 256u || (ai < 256u &&
                (smem_keys[a_base + ai] < smem_keys[b_base + bi] ||
                 (smem_keys[a_base + ai] == smem_keys[b_base + bi] &&
                  smem_vals[a_base + ai] <= smem_vals[b_base + bi])));
            if take_a {
                out_keys[k] = smem_keys[a_base + ai];
                out_vals[k] = smem_vals[a_base + ai];
                ai = ai + 1u;
            } else {
                out_keys[k] = smem_keys[b_base + bi];
                out_vals[k] = smem_vals[b_base + bi];
                bi = bi + 1u;
            }
        }
        workgroupBarrier();   // every read is done before any write-back
        storageBarrier();     // device-scope fence: workgroupBarrier alone under-orders
                              // the in-place write-back for single-SIMD-group WGs
        for (var k = 0u; k < WPT; k = k + 1u) {
            smem_keys[base + k] = out_keys[k];
            smem_vals[base + k] = out_vals[k];
        }
    }
    workgroupBarrier();
    // merge pass 3: two sorted runs of 512 -> 1024 (register-staged)
    {
        let group_base = (base / 1024u) * 1024u;
        let diag = base - group_base;
        let a_base = group_base;
        let b_base = group_base + 512u;
        // merge-path: binary search the diagonal for this thread's A/B split
        var lo = select(0u, diag - 512u, diag > 512u);
        var hi = min(diag, 512u);
        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            let ak = smem_keys[a_base + mid];
            let av = smem_vals[a_base + mid];
            let bpos = b_base + (diag - 1u - mid);
            let bk = smem_keys[bpos];
            let bv = smem_vals[bpos];
            if ak < bk || (ak == bk && av <= bv) { lo = mid + 1u; } else { hi = mid; }
        }
        var ai = lo;
        var bi = diag - lo;
        // merge this thread's WPT outputs into registers (the pong)
        var out_keys: array<u32, 8>;
        var out_vals: array<u32, 8>;
        for (var k = 0u; k < WPT; k = k + 1u) {
            let take_a = bi >= 512u || (ai < 512u &&
                (smem_keys[a_base + ai] < smem_keys[b_base + bi] ||
                 (smem_keys[a_base + ai] == smem_keys[b_base + bi] &&
                  smem_vals[a_base + ai] <= smem_vals[b_base + bi])));
            if take_a {
                out_keys[k] = smem_keys[a_base + ai];
                out_vals[k] = smem_vals[a_base + ai];
                ai = ai + 1u;
            } else {
                out_keys[k] = smem_keys[b_base + bi];
                out_vals[k] = smem_vals[b_base + bi];
                bi = bi + 1u;
            }
        }
        workgroupBarrier();   // every read is done before any write-back
        storageBarrier();     // device-scope fence: workgroupBarrier alone under-orders
                              // the in-place write-back for single-SIMD-group WGs
        for (var k = 0u; k < WPT; k = k + 1u) {
            smem_keys[base + k] = out_keys[k];
            smem_vals[base + k] = out_vals[k];
        }
    }
    workgroupBarrier();
    // merge pass 4: two sorted runs of 1024 -> 2048 (register-staged)
    {
        let group_base = (base / 2048u) * 2048u;
        let diag = base - group_base;
        let a_base = group_base;
        let b_base = group_base + 1024u;
        // merge-path: binary search the diagonal for this thread's A/B split
        var lo = select(0u, diag - 1024u, diag > 1024u);
        var hi = min(diag, 1024u);
        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            let ak = smem_keys[a_base + mid];
            let av = smem_vals[a_base + mid];
            let bpos = b_base + (diag - 1u - mid);
            let bk = smem_keys[bpos];
            let bv = smem_vals[bpos];
            if ak < bk || (ak == bk && av <= bv) { lo = mid + 1u; } else { hi = mid; }
        }
        var ai = lo;
        var bi = diag - lo;
        // merge this thread's WPT outputs into registers (the pong)
        var out_keys: array<u32, 8>;
        var out_vals: array<u32, 8>;
        for (var k = 0u; k < WPT; k = k + 1u) {
            let take_a = bi >= 1024u || (ai < 1024u &&
                (smem_keys[a_base + ai] < smem_keys[b_base + bi] ||
                 (smem_keys[a_base + ai] == smem_keys[b_base + bi] &&
                  smem_vals[a_base + ai] <= smem_vals[b_base + bi])));
            if take_a {
                out_keys[k] = smem_keys[a_base + ai];
                out_vals[k] = smem_vals[a_base + ai];
                ai = ai + 1u;
            } else {
                out_keys[k] = smem_keys[b_base + bi];
                out_vals[k] = smem_vals[b_base + bi];
                bi = bi + 1u;
            }
        }
        workgroupBarrier();   // every read is done before any write-back
        storageBarrier();     // device-scope fence: workgroupBarrier alone under-orders
                              // the in-place write-back for single-SIMD-group WGs
        for (var k = 0u; k < WPT; k = k + 1u) {
            smem_keys[base + k] = out_keys[k];
            smem_vals[base + k] = out_vals[k];
        }
    }
    workgroupBarrier();

    // ---- phase 3: coalesced store from the final buffer ----
    for (var c = 0u; c < WPT; c = c + 1u) {
        let j = c * M + local_tid;
        if is_active && j < seg_size {
            global_keys[seg_start + j] = smem_keys[j];
            global_value_indices[seg_start + j] = smem_vals[j];
        }
    }
}
