
enable subgroups;

override WG: u32 = 64u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 512u;
const M: u32 = 64u;
const WPT: u32 = 8u;
const R: u32 = 8u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybrid_sg8_n512_m64_striped(
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
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_270 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_271 = seg_base + (local_tid ^ 15u); let tmp_272 = smem_keys[tmp_271 * WPT + 7u]; let tmp_273 = smem_vals[tmp_271 * WPT + 7u]; let tmp_274 = keys[0] < tmp_272 || (keys[0] == tmp_272 && values[0] < tmp_273); if tmp_270 == tmp_274 { keys[0] = tmp_272; values[0] = tmp_273; } let tmp_275 = smem_keys[tmp_271 * WPT + 6u]; let tmp_276 = smem_vals[tmp_271 * WPT + 6u]; let tmp_277 = keys[1] < tmp_275 || (keys[1] == tmp_275 && values[1] < tmp_276); if tmp_270 == tmp_277 { keys[1] = tmp_275; values[1] = tmp_276; } let tmp_278 = smem_keys[tmp_271 * WPT + 5u]; let tmp_279 = smem_vals[tmp_271 * WPT + 5u]; let tmp_280 = keys[2] < tmp_278 || (keys[2] == tmp_278 && values[2] < tmp_279); if tmp_270 == tmp_280 { keys[2] = tmp_278; values[2] = tmp_279; } let tmp_281 = smem_keys[tmp_271 * WPT + 4u]; let tmp_282 = smem_vals[tmp_271 * WPT + 4u]; let tmp_283 = keys[3] < tmp_281 || (keys[3] == tmp_281 && values[3] < tmp_282); if tmp_270 == tmp_283 { keys[3] = tmp_281; values[3] = tmp_282; } let tmp_284 = smem_keys[tmp_271 * WPT + 3u]; let tmp_285 = smem_vals[tmp_271 * WPT + 3u]; let tmp_286 = keys[4] < tmp_284 || (keys[4] == tmp_284 && values[4] < tmp_285); if tmp_270 == tmp_286 { keys[4] = tmp_284; values[4] = tmp_285; } let tmp_287 = smem_keys[tmp_271 * WPT + 2u]; let tmp_288 = smem_vals[tmp_271 * WPT + 2u]; let tmp_289 = keys[5] < tmp_287 || (keys[5] == tmp_287 && values[5] < tmp_288); if tmp_270 == tmp_289 { keys[5] = tmp_287; values[5] = tmp_288; } let tmp_290 = smem_keys[tmp_271 * WPT + 1u]; let tmp_291 = smem_vals[tmp_271 * WPT + 1u]; let tmp_292 = keys[6] < tmp_290 || (keys[6] == tmp_290 && values[6] < tmp_291); if tmp_270 == tmp_292 { keys[6] = tmp_290; values[6] = tmp_291; } let tmp_293 = smem_keys[tmp_271 * WPT + 0u]; let tmp_294 = smem_vals[tmp_271 * WPT + 0u]; let tmp_295 = keys[7] < tmp_293 || (keys[7] == tmp_293 && values[7] < tmp_294); if tmp_270 == tmp_295 { keys[7] = tmp_293; values[7] = tmp_294; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:8) 
    {
    let tmp_296 = subgroupShuffleXor(keys[0], 4u);
    let tmp_297 = subgroupShuffleXor(values[0], 4u);
    let tmp_298 = subgroupShuffleXor(keys[1], 4u);
    let tmp_299 = subgroupShuffleXor(values[1], 4u);
    let tmp_300 = subgroupShuffleXor(keys[2], 4u);
    let tmp_301 = subgroupShuffleXor(values[2], 4u);
    let tmp_302 = subgroupShuffleXor(keys[3], 4u);
    let tmp_303 = subgroupShuffleXor(values[3], 4u);
    let tmp_304 = subgroupShuffleXor(keys[4], 4u);
    let tmp_305 = subgroupShuffleXor(values[4], 4u);
    let tmp_306 = subgroupShuffleXor(keys[5], 4u);
    let tmp_307 = subgroupShuffleXor(values[5], 4u);
    let tmp_308 = subgroupShuffleXor(keys[6], 4u);
    let tmp_309 = subgroupShuffleXor(values[6], 4u);
    let tmp_310 = subgroupShuffleXor(keys[7], 4u);
    let tmp_311 = subgroupShuffleXor(values[7], 4u);
    let tmp_312 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_313 = keys[0] < tmp_296 || (keys[0] == tmp_296 && values[0] < tmp_297);
    if tmp_312 == tmp_313 { keys[0] = tmp_296; values[0] = tmp_297; }
    let tmp_314 = keys[1] < tmp_298 || (keys[1] == tmp_298 && values[1] < tmp_299);
    if tmp_312 == tmp_314 { keys[1] = tmp_298; values[1] = tmp_299; }
    let tmp_315 = keys[2] < tmp_300 || (keys[2] == tmp_300 && values[2] < tmp_301);
    if tmp_312 == tmp_315 { keys[2] = tmp_300; values[2] = tmp_301; }
    let tmp_316 = keys[3] < tmp_302 || (keys[3] == tmp_302 && values[3] < tmp_303);
    if tmp_312 == tmp_316 { keys[3] = tmp_302; values[3] = tmp_303; }
    let tmp_317 = keys[4] < tmp_304 || (keys[4] == tmp_304 && values[4] < tmp_305);
    if tmp_312 == tmp_317 { keys[4] = tmp_304; values[4] = tmp_305; }
    let tmp_318 = keys[5] < tmp_306 || (keys[5] == tmp_306 && values[5] < tmp_307);
    if tmp_312 == tmp_318 { keys[5] = tmp_306; values[5] = tmp_307; }
    let tmp_319 = keys[6] < tmp_308 || (keys[6] == tmp_308 && values[6] < tmp_309);
    if tmp_312 == tmp_319 { keys[6] = tmp_308; values[6] = tmp_309; }
    let tmp_320 = keys[7] < tmp_310 || (keys[7] == tmp_310 && values[7] < tmp_311);
    if tmp_312 == tmp_320 { keys[7] = tmp_310; values[7] = tmp_311; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    {
    let tmp_321 = subgroupShuffleXor(keys[0], 2u);
    let tmp_322 = subgroupShuffleXor(values[0], 2u);
    let tmp_323 = subgroupShuffleXor(keys[1], 2u);
    let tmp_324 = subgroupShuffleXor(values[1], 2u);
    let tmp_325 = subgroupShuffleXor(keys[2], 2u);
    let tmp_326 = subgroupShuffleXor(values[2], 2u);
    let tmp_327 = subgroupShuffleXor(keys[3], 2u);
    let tmp_328 = subgroupShuffleXor(values[3], 2u);
    let tmp_329 = subgroupShuffleXor(keys[4], 2u);
    let tmp_330 = subgroupShuffleXor(values[4], 2u);
    let tmp_331 = subgroupShuffleXor(keys[5], 2u);
    let tmp_332 = subgroupShuffleXor(values[5], 2u);
    let tmp_333 = subgroupShuffleXor(keys[6], 2u);
    let tmp_334 = subgroupShuffleXor(values[6], 2u);
    let tmp_335 = subgroupShuffleXor(keys[7], 2u);
    let tmp_336 = subgroupShuffleXor(values[7], 2u);
    let tmp_337 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_338 = keys[0] < tmp_321 || (keys[0] == tmp_321 && values[0] < tmp_322);
    if tmp_337 == tmp_338 { keys[0] = tmp_321; values[0] = tmp_322; }
    let tmp_339 = keys[1] < tmp_323 || (keys[1] == tmp_323 && values[1] < tmp_324);
    if tmp_337 == tmp_339 { keys[1] = tmp_323; values[1] = tmp_324; }
    let tmp_340 = keys[2] < tmp_325 || (keys[2] == tmp_325 && values[2] < tmp_326);
    if tmp_337 == tmp_340 { keys[2] = tmp_325; values[2] = tmp_326; }
    let tmp_341 = keys[3] < tmp_327 || (keys[3] == tmp_327 && values[3] < tmp_328);
    if tmp_337 == tmp_341 { keys[3] = tmp_327; values[3] = tmp_328; }
    let tmp_342 = keys[4] < tmp_329 || (keys[4] == tmp_329 && values[4] < tmp_330);
    if tmp_337 == tmp_342 { keys[4] = tmp_329; values[4] = tmp_330; }
    let tmp_343 = keys[5] < tmp_331 || (keys[5] == tmp_331 && values[5] < tmp_332);
    if tmp_337 == tmp_343 { keys[5] = tmp_331; values[5] = tmp_332; }
    let tmp_344 = keys[6] < tmp_333 || (keys[6] == tmp_333 && values[6] < tmp_334);
    if tmp_337 == tmp_344 { keys[6] = tmp_333; values[6] = tmp_334; }
    let tmp_345 = keys[7] < tmp_335 || (keys[7] == tmp_335 && values[7] < tmp_336);
    if tmp_337 == tmp_345 { keys[7] = tmp_335; values[7] = tmp_336; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    {
    let tmp_346 = subgroupShuffleXor(keys[0], 1u);
    let tmp_347 = subgroupShuffleXor(values[0], 1u);
    let tmp_348 = subgroupShuffleXor(keys[1], 1u);
    let tmp_349 = subgroupShuffleXor(values[1], 1u);
    let tmp_350 = subgroupShuffleXor(keys[2], 1u);
    let tmp_351 = subgroupShuffleXor(values[2], 1u);
    let tmp_352 = subgroupShuffleXor(keys[3], 1u);
    let tmp_353 = subgroupShuffleXor(values[3], 1u);
    let tmp_354 = subgroupShuffleXor(keys[4], 1u);
    let tmp_355 = subgroupShuffleXor(values[4], 1u);
    let tmp_356 = subgroupShuffleXor(keys[5], 1u);
    let tmp_357 = subgroupShuffleXor(values[5], 1u);
    let tmp_358 = subgroupShuffleXor(keys[6], 1u);
    let tmp_359 = subgroupShuffleXor(values[6], 1u);
    let tmp_360 = subgroupShuffleXor(keys[7], 1u);
    let tmp_361 = subgroupShuffleXor(values[7], 1u);
    let tmp_362 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_363 = keys[0] < tmp_346 || (keys[0] == tmp_346 && values[0] < tmp_347);
    if tmp_362 == tmp_363 { keys[0] = tmp_346; values[0] = tmp_347; }
    let tmp_364 = keys[1] < tmp_348 || (keys[1] == tmp_348 && values[1] < tmp_349);
    if tmp_362 == tmp_364 { keys[1] = tmp_348; values[1] = tmp_349; }
    let tmp_365 = keys[2] < tmp_350 || (keys[2] == tmp_350 && values[2] < tmp_351);
    if tmp_362 == tmp_365 { keys[2] = tmp_350; values[2] = tmp_351; }
    let tmp_366 = keys[3] < tmp_352 || (keys[3] == tmp_352 && values[3] < tmp_353);
    if tmp_362 == tmp_366 { keys[3] = tmp_352; values[3] = tmp_353; }
    let tmp_367 = keys[4] < tmp_354 || (keys[4] == tmp_354 && values[4] < tmp_355);
    if tmp_362 == tmp_367 { keys[4] = tmp_354; values[4] = tmp_355; }
    let tmp_368 = keys[5] < tmp_356 || (keys[5] == tmp_356 && values[5] < tmp_357);
    if tmp_362 == tmp_368 { keys[5] = tmp_356; values[5] = tmp_357; }
    let tmp_369 = keys[6] < tmp_358 || (keys[6] == tmp_358 && values[6] < tmp_359);
    if tmp_362 == tmp_369 { keys[6] = tmp_358; values[6] = tmp_359; }
    let tmp_370 = keys[7] < tmp_360 || (keys[7] == tmp_360 && values[7] < tmp_361);
    if tmp_362 == tmp_370 { keys[7] = tmp_360; values[7] = tmp_361; }
    }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_371 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_371;let tmp_372 = values[0]; values[0] = values[4]; values[4] = tmp_372; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_373 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_373;let tmp_374 = values[1]; values[1] = values[5]; values[5] = tmp_374; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_375 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_375;let tmp_376 = values[2]; values[2] = values[6]; values[6] = tmp_376; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_377 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_377;let tmp_378 = values[3]; values[3] = values[7]; values[7] = tmp_378; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_379 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_379;let tmp_380 = values[0]; values[0] = values[2]; values[2] = tmp_380; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_381 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_381;let tmp_382 = values[1]; values[1] = values[3]; values[3] = tmp_382; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_383 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_383;let tmp_384 = values[4]; values[4] = values[6]; values[6] = tmp_384; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_385 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_385;let tmp_386 = values[5]; values[5] = values[7]; values[7] = tmp_386; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_387 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_387;let tmp_388 = values[0]; values[0] = values[1]; values[1] = tmp_388; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_389 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_389;let tmp_390 = values[2]; values[2] = values[3]; values[3] = tmp_390; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_391 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_391;let tmp_392 = values[4]; values[4] = values[5]; values[5] = tmp_392; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_393 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_393;let tmp_394 = values[6]; values[6] = values[7]; values[7] = tmp_394; }
    }
    // exch_intxn(tmask:31,swbit:4,wpt:8)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_395 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_396 = seg_base + (local_tid ^ 31u); let tmp_397 = smem_keys[tmp_396 * WPT + 7u]; let tmp_398 = smem_vals[tmp_396 * WPT + 7u]; let tmp_399 = keys[0] < tmp_397 || (keys[0] == tmp_397 && values[0] < tmp_398); if tmp_395 == tmp_399 { keys[0] = tmp_397; values[0] = tmp_398; } let tmp_400 = smem_keys[tmp_396 * WPT + 6u]; let tmp_401 = smem_vals[tmp_396 * WPT + 6u]; let tmp_402 = keys[1] < tmp_400 || (keys[1] == tmp_400 && values[1] < tmp_401); if tmp_395 == tmp_402 { keys[1] = tmp_400; values[1] = tmp_401; } let tmp_403 = smem_keys[tmp_396 * WPT + 5u]; let tmp_404 = smem_vals[tmp_396 * WPT + 5u]; let tmp_405 = keys[2] < tmp_403 || (keys[2] == tmp_403 && values[2] < tmp_404); if tmp_395 == tmp_405 { keys[2] = tmp_403; values[2] = tmp_404; } let tmp_406 = smem_keys[tmp_396 * WPT + 4u]; let tmp_407 = smem_vals[tmp_396 * WPT + 4u]; let tmp_408 = keys[3] < tmp_406 || (keys[3] == tmp_406 && values[3] < tmp_407); if tmp_395 == tmp_408 { keys[3] = tmp_406; values[3] = tmp_407; } let tmp_409 = smem_keys[tmp_396 * WPT + 3u]; let tmp_410 = smem_vals[tmp_396 * WPT + 3u]; let tmp_411 = keys[4] < tmp_409 || (keys[4] == tmp_409 && values[4] < tmp_410); if tmp_395 == tmp_411 { keys[4] = tmp_409; values[4] = tmp_410; } let tmp_412 = smem_keys[tmp_396 * WPT + 2u]; let tmp_413 = smem_vals[tmp_396 * WPT + 2u]; let tmp_414 = keys[5] < tmp_412 || (keys[5] == tmp_412 && values[5] < tmp_413); if tmp_395 == tmp_414 { keys[5] = tmp_412; values[5] = tmp_413; } let tmp_415 = smem_keys[tmp_396 * WPT + 1u]; let tmp_416 = smem_vals[tmp_396 * WPT + 1u]; let tmp_417 = keys[6] < tmp_415 || (keys[6] == tmp_415 && values[6] < tmp_416); if tmp_395 == tmp_417 { keys[6] = tmp_415; values[6] = tmp_416; } let tmp_418 = smem_keys[tmp_396 * WPT + 0u]; let tmp_419 = smem_vals[tmp_396 * WPT + 0u]; let tmp_420 = keys[7] < tmp_418 || (keys[7] == tmp_418 && values[7] < tmp_419); if tmp_395 == tmp_420 { keys[7] = tmp_418; values[7] = tmp_419; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_421 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_422 = seg_base + (local_tid ^ 8u); let tmp_423 = smem_keys[tmp_422 * WPT + 0u]; let tmp_424 = smem_vals[tmp_422 * WPT + 0u]; let tmp_425 = keys[0] < tmp_423 || (keys[0] == tmp_423 && values[0] < tmp_424); if tmp_421 == tmp_425 { keys[0] = tmp_423; values[0] = tmp_424; } let tmp_426 = smem_keys[tmp_422 * WPT + 1u]; let tmp_427 = smem_vals[tmp_422 * WPT + 1u]; let tmp_428 = keys[1] < tmp_426 || (keys[1] == tmp_426 && values[1] < tmp_427); if tmp_421 == tmp_428 { keys[1] = tmp_426; values[1] = tmp_427; } let tmp_429 = smem_keys[tmp_422 * WPT + 2u]; let tmp_430 = smem_vals[tmp_422 * WPT + 2u]; let tmp_431 = keys[2] < tmp_429 || (keys[2] == tmp_429 && values[2] < tmp_430); if tmp_421 == tmp_431 { keys[2] = tmp_429; values[2] = tmp_430; } let tmp_432 = smem_keys[tmp_422 * WPT + 3u]; let tmp_433 = smem_vals[tmp_422 * WPT + 3u]; let tmp_434 = keys[3] < tmp_432 || (keys[3] == tmp_432 && values[3] < tmp_433); if tmp_421 == tmp_434 { keys[3] = tmp_432; values[3] = tmp_433; } let tmp_435 = smem_keys[tmp_422 * WPT + 4u]; let tmp_436 = smem_vals[tmp_422 * WPT + 4u]; let tmp_437 = keys[4] < tmp_435 || (keys[4] == tmp_435 && values[4] < tmp_436); if tmp_421 == tmp_437 { keys[4] = tmp_435; values[4] = tmp_436; } let tmp_438 = smem_keys[tmp_422 * WPT + 5u]; let tmp_439 = smem_vals[tmp_422 * WPT + 5u]; let tmp_440 = keys[5] < tmp_438 || (keys[5] == tmp_438 && values[5] < tmp_439); if tmp_421 == tmp_440 { keys[5] = tmp_438; values[5] = tmp_439; } let tmp_441 = smem_keys[tmp_422 * WPT + 6u]; let tmp_442 = smem_vals[tmp_422 * WPT + 6u]; let tmp_443 = keys[6] < tmp_441 || (keys[6] == tmp_441 && values[6] < tmp_442); if tmp_421 == tmp_443 { keys[6] = tmp_441; values[6] = tmp_442; } let tmp_444 = smem_keys[tmp_422 * WPT + 7u]; let tmp_445 = smem_vals[tmp_422 * WPT + 7u]; let tmp_446 = keys[7] < tmp_444 || (keys[7] == tmp_444 && values[7] < tmp_445); if tmp_421 == tmp_446 { keys[7] = tmp_444; values[7] = tmp_445; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:8) 
    {
    let tmp_447 = subgroupShuffleXor(keys[0], 4u);
    let tmp_448 = subgroupShuffleXor(values[0], 4u);
    let tmp_449 = subgroupShuffleXor(keys[1], 4u);
    let tmp_450 = subgroupShuffleXor(values[1], 4u);
    let tmp_451 = subgroupShuffleXor(keys[2], 4u);
    let tmp_452 = subgroupShuffleXor(values[2], 4u);
    let tmp_453 = subgroupShuffleXor(keys[3], 4u);
    let tmp_454 = subgroupShuffleXor(values[3], 4u);
    let tmp_455 = subgroupShuffleXor(keys[4], 4u);
    let tmp_456 = subgroupShuffleXor(values[4], 4u);
    let tmp_457 = subgroupShuffleXor(keys[5], 4u);
    let tmp_458 = subgroupShuffleXor(values[5], 4u);
    let tmp_459 = subgroupShuffleXor(keys[6], 4u);
    let tmp_460 = subgroupShuffleXor(values[6], 4u);
    let tmp_461 = subgroupShuffleXor(keys[7], 4u);
    let tmp_462 = subgroupShuffleXor(values[7], 4u);
    let tmp_463 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_464 = keys[0] < tmp_447 || (keys[0] == tmp_447 && values[0] < tmp_448);
    if tmp_463 == tmp_464 { keys[0] = tmp_447; values[0] = tmp_448; }
    let tmp_465 = keys[1] < tmp_449 || (keys[1] == tmp_449 && values[1] < tmp_450);
    if tmp_463 == tmp_465 { keys[1] = tmp_449; values[1] = tmp_450; }
    let tmp_466 = keys[2] < tmp_451 || (keys[2] == tmp_451 && values[2] < tmp_452);
    if tmp_463 == tmp_466 { keys[2] = tmp_451; values[2] = tmp_452; }
    let tmp_467 = keys[3] < tmp_453 || (keys[3] == tmp_453 && values[3] < tmp_454);
    if tmp_463 == tmp_467 { keys[3] = tmp_453; values[3] = tmp_454; }
    let tmp_468 = keys[4] < tmp_455 || (keys[4] == tmp_455 && values[4] < tmp_456);
    if tmp_463 == tmp_468 { keys[4] = tmp_455; values[4] = tmp_456; }
    let tmp_469 = keys[5] < tmp_457 || (keys[5] == tmp_457 && values[5] < tmp_458);
    if tmp_463 == tmp_469 { keys[5] = tmp_457; values[5] = tmp_458; }
    let tmp_470 = keys[6] < tmp_459 || (keys[6] == tmp_459 && values[6] < tmp_460);
    if tmp_463 == tmp_470 { keys[6] = tmp_459; values[6] = tmp_460; }
    let tmp_471 = keys[7] < tmp_461 || (keys[7] == tmp_461 && values[7] < tmp_462);
    if tmp_463 == tmp_471 { keys[7] = tmp_461; values[7] = tmp_462; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    {
    let tmp_472 = subgroupShuffleXor(keys[0], 2u);
    let tmp_473 = subgroupShuffleXor(values[0], 2u);
    let tmp_474 = subgroupShuffleXor(keys[1], 2u);
    let tmp_475 = subgroupShuffleXor(values[1], 2u);
    let tmp_476 = subgroupShuffleXor(keys[2], 2u);
    let tmp_477 = subgroupShuffleXor(values[2], 2u);
    let tmp_478 = subgroupShuffleXor(keys[3], 2u);
    let tmp_479 = subgroupShuffleXor(values[3], 2u);
    let tmp_480 = subgroupShuffleXor(keys[4], 2u);
    let tmp_481 = subgroupShuffleXor(values[4], 2u);
    let tmp_482 = subgroupShuffleXor(keys[5], 2u);
    let tmp_483 = subgroupShuffleXor(values[5], 2u);
    let tmp_484 = subgroupShuffleXor(keys[6], 2u);
    let tmp_485 = subgroupShuffleXor(values[6], 2u);
    let tmp_486 = subgroupShuffleXor(keys[7], 2u);
    let tmp_487 = subgroupShuffleXor(values[7], 2u);
    let tmp_488 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_489 = keys[0] < tmp_472 || (keys[0] == tmp_472 && values[0] < tmp_473);
    if tmp_488 == tmp_489 { keys[0] = tmp_472; values[0] = tmp_473; }
    let tmp_490 = keys[1] < tmp_474 || (keys[1] == tmp_474 && values[1] < tmp_475);
    if tmp_488 == tmp_490 { keys[1] = tmp_474; values[1] = tmp_475; }
    let tmp_491 = keys[2] < tmp_476 || (keys[2] == tmp_476 && values[2] < tmp_477);
    if tmp_488 == tmp_491 { keys[2] = tmp_476; values[2] = tmp_477; }
    let tmp_492 = keys[3] < tmp_478 || (keys[3] == tmp_478 && values[3] < tmp_479);
    if tmp_488 == tmp_492 { keys[3] = tmp_478; values[3] = tmp_479; }
    let tmp_493 = keys[4] < tmp_480 || (keys[4] == tmp_480 && values[4] < tmp_481);
    if tmp_488 == tmp_493 { keys[4] = tmp_480; values[4] = tmp_481; }
    let tmp_494 = keys[5] < tmp_482 || (keys[5] == tmp_482 && values[5] < tmp_483);
    if tmp_488 == tmp_494 { keys[5] = tmp_482; values[5] = tmp_483; }
    let tmp_495 = keys[6] < tmp_484 || (keys[6] == tmp_484 && values[6] < tmp_485);
    if tmp_488 == tmp_495 { keys[6] = tmp_484; values[6] = tmp_485; }
    let tmp_496 = keys[7] < tmp_486 || (keys[7] == tmp_486 && values[7] < tmp_487);
    if tmp_488 == tmp_496 { keys[7] = tmp_486; values[7] = tmp_487; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    {
    let tmp_497 = subgroupShuffleXor(keys[0], 1u);
    let tmp_498 = subgroupShuffleXor(values[0], 1u);
    let tmp_499 = subgroupShuffleXor(keys[1], 1u);
    let tmp_500 = subgroupShuffleXor(values[1], 1u);
    let tmp_501 = subgroupShuffleXor(keys[2], 1u);
    let tmp_502 = subgroupShuffleXor(values[2], 1u);
    let tmp_503 = subgroupShuffleXor(keys[3], 1u);
    let tmp_504 = subgroupShuffleXor(values[3], 1u);
    let tmp_505 = subgroupShuffleXor(keys[4], 1u);
    let tmp_506 = subgroupShuffleXor(values[4], 1u);
    let tmp_507 = subgroupShuffleXor(keys[5], 1u);
    let tmp_508 = subgroupShuffleXor(values[5], 1u);
    let tmp_509 = subgroupShuffleXor(keys[6], 1u);
    let tmp_510 = subgroupShuffleXor(values[6], 1u);
    let tmp_511 = subgroupShuffleXor(keys[7], 1u);
    let tmp_512 = subgroupShuffleXor(values[7], 1u);
    let tmp_513 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_514 = keys[0] < tmp_497 || (keys[0] == tmp_497 && values[0] < tmp_498);
    if tmp_513 == tmp_514 { keys[0] = tmp_497; values[0] = tmp_498; }
    let tmp_515 = keys[1] < tmp_499 || (keys[1] == tmp_499 && values[1] < tmp_500);
    if tmp_513 == tmp_515 { keys[1] = tmp_499; values[1] = tmp_500; }
    let tmp_516 = keys[2] < tmp_501 || (keys[2] == tmp_501 && values[2] < tmp_502);
    if tmp_513 == tmp_516 { keys[2] = tmp_501; values[2] = tmp_502; }
    let tmp_517 = keys[3] < tmp_503 || (keys[3] == tmp_503 && values[3] < tmp_504);
    if tmp_513 == tmp_517 { keys[3] = tmp_503; values[3] = tmp_504; }
    let tmp_518 = keys[4] < tmp_505 || (keys[4] == tmp_505 && values[4] < tmp_506);
    if tmp_513 == tmp_518 { keys[4] = tmp_505; values[4] = tmp_506; }
    let tmp_519 = keys[5] < tmp_507 || (keys[5] == tmp_507 && values[5] < tmp_508);
    if tmp_513 == tmp_519 { keys[5] = tmp_507; values[5] = tmp_508; }
    let tmp_520 = keys[6] < tmp_509 || (keys[6] == tmp_509 && values[6] < tmp_510);
    if tmp_513 == tmp_520 { keys[6] = tmp_509; values[6] = tmp_510; }
    let tmp_521 = keys[7] < tmp_511 || (keys[7] == tmp_511 && values[7] < tmp_512);
    if tmp_513 == tmp_521 { keys[7] = tmp_511; values[7] = tmp_512; }
    }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_522 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_522;let tmp_523 = values[0]; values[0] = values[4]; values[4] = tmp_523; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_524 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_524;let tmp_525 = values[1]; values[1] = values[5]; values[5] = tmp_525; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_526 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_526;let tmp_527 = values[2]; values[2] = values[6]; values[6] = tmp_527; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_528 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_528;let tmp_529 = values[3]; values[3] = values[7]; values[7] = tmp_529; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_530 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_530;let tmp_531 = values[0]; values[0] = values[2]; values[2] = tmp_531; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_532 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_532;let tmp_533 = values[1]; values[1] = values[3]; values[3] = tmp_533; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_534 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_534;let tmp_535 = values[4]; values[4] = values[6]; values[6] = tmp_535; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_536 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_536;let tmp_537 = values[5]; values[5] = values[7]; values[7] = tmp_537; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_538 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_538;let tmp_539 = values[0]; values[0] = values[1]; values[1] = tmp_539; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_540 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_540;let tmp_541 = values[2]; values[2] = values[3]; values[3] = tmp_541; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_542 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_542;let tmp_543 = values[4]; values[4] = values[5]; values[5] = tmp_543; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_544 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_544;let tmp_545 = values[6]; values[6] = values[7]; values[7] = tmp_545; }
    }
    // exch_intxn(tmask:63,swbit:5,wpt:8)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_546 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_547 = seg_base + (local_tid ^ 63u); let tmp_548 = smem_keys[tmp_547 * WPT + 7u]; let tmp_549 = smem_vals[tmp_547 * WPT + 7u]; let tmp_550 = keys[0] < tmp_548 || (keys[0] == tmp_548 && values[0] < tmp_549); if tmp_546 == tmp_550 { keys[0] = tmp_548; values[0] = tmp_549; } let tmp_551 = smem_keys[tmp_547 * WPT + 6u]; let tmp_552 = smem_vals[tmp_547 * WPT + 6u]; let tmp_553 = keys[1] < tmp_551 || (keys[1] == tmp_551 && values[1] < tmp_552); if tmp_546 == tmp_553 { keys[1] = tmp_551; values[1] = tmp_552; } let tmp_554 = smem_keys[tmp_547 * WPT + 5u]; let tmp_555 = smem_vals[tmp_547 * WPT + 5u]; let tmp_556 = keys[2] < tmp_554 || (keys[2] == tmp_554 && values[2] < tmp_555); if tmp_546 == tmp_556 { keys[2] = tmp_554; values[2] = tmp_555; } let tmp_557 = smem_keys[tmp_547 * WPT + 4u]; let tmp_558 = smem_vals[tmp_547 * WPT + 4u]; let tmp_559 = keys[3] < tmp_557 || (keys[3] == tmp_557 && values[3] < tmp_558); if tmp_546 == tmp_559 { keys[3] = tmp_557; values[3] = tmp_558; } let tmp_560 = smem_keys[tmp_547 * WPT + 3u]; let tmp_561 = smem_vals[tmp_547 * WPT + 3u]; let tmp_562 = keys[4] < tmp_560 || (keys[4] == tmp_560 && values[4] < tmp_561); if tmp_546 == tmp_562 { keys[4] = tmp_560; values[4] = tmp_561; } let tmp_563 = smem_keys[tmp_547 * WPT + 2u]; let tmp_564 = smem_vals[tmp_547 * WPT + 2u]; let tmp_565 = keys[5] < tmp_563 || (keys[5] == tmp_563 && values[5] < tmp_564); if tmp_546 == tmp_565 { keys[5] = tmp_563; values[5] = tmp_564; } let tmp_566 = smem_keys[tmp_547 * WPT + 1u]; let tmp_567 = smem_vals[tmp_547 * WPT + 1u]; let tmp_568 = keys[6] < tmp_566 || (keys[6] == tmp_566 && values[6] < tmp_567); if tmp_546 == tmp_568 { keys[6] = tmp_566; values[6] = tmp_567; } let tmp_569 = smem_keys[tmp_547 * WPT + 0u]; let tmp_570 = smem_vals[tmp_547 * WPT + 0u]; let tmp_571 = keys[7] < tmp_569 || (keys[7] == tmp_569 && values[7] < tmp_570); if tmp_546 == tmp_571 { keys[7] = tmp_569; values[7] = tmp_570; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_572 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_573 = seg_base + (local_tid ^ 16u); let tmp_574 = smem_keys[tmp_573 * WPT + 0u]; let tmp_575 = smem_vals[tmp_573 * WPT + 0u]; let tmp_576 = keys[0] < tmp_574 || (keys[0] == tmp_574 && values[0] < tmp_575); if tmp_572 == tmp_576 { keys[0] = tmp_574; values[0] = tmp_575; } let tmp_577 = smem_keys[tmp_573 * WPT + 1u]; let tmp_578 = smem_vals[tmp_573 * WPT + 1u]; let tmp_579 = keys[1] < tmp_577 || (keys[1] == tmp_577 && values[1] < tmp_578); if tmp_572 == tmp_579 { keys[1] = tmp_577; values[1] = tmp_578; } let tmp_580 = smem_keys[tmp_573 * WPT + 2u]; let tmp_581 = smem_vals[tmp_573 * WPT + 2u]; let tmp_582 = keys[2] < tmp_580 || (keys[2] == tmp_580 && values[2] < tmp_581); if tmp_572 == tmp_582 { keys[2] = tmp_580; values[2] = tmp_581; } let tmp_583 = smem_keys[tmp_573 * WPT + 3u]; let tmp_584 = smem_vals[tmp_573 * WPT + 3u]; let tmp_585 = keys[3] < tmp_583 || (keys[3] == tmp_583 && values[3] < tmp_584); if tmp_572 == tmp_585 { keys[3] = tmp_583; values[3] = tmp_584; } let tmp_586 = smem_keys[tmp_573 * WPT + 4u]; let tmp_587 = smem_vals[tmp_573 * WPT + 4u]; let tmp_588 = keys[4] < tmp_586 || (keys[4] == tmp_586 && values[4] < tmp_587); if tmp_572 == tmp_588 { keys[4] = tmp_586; values[4] = tmp_587; } let tmp_589 = smem_keys[tmp_573 * WPT + 5u]; let tmp_590 = smem_vals[tmp_573 * WPT + 5u]; let tmp_591 = keys[5] < tmp_589 || (keys[5] == tmp_589 && values[5] < tmp_590); if tmp_572 == tmp_591 { keys[5] = tmp_589; values[5] = tmp_590; } let tmp_592 = smem_keys[tmp_573 * WPT + 6u]; let tmp_593 = smem_vals[tmp_573 * WPT + 6u]; let tmp_594 = keys[6] < tmp_592 || (keys[6] == tmp_592 && values[6] < tmp_593); if tmp_572 == tmp_594 { keys[6] = tmp_592; values[6] = tmp_593; } let tmp_595 = smem_keys[tmp_573 * WPT + 7u]; let tmp_596 = smem_vals[tmp_573 * WPT + 7u]; let tmp_597 = keys[7] < tmp_595 || (keys[7] == tmp_595 && values[7] < tmp_596); if tmp_572 == tmp_597 { keys[7] = tmp_595; values[7] = tmp_596; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_598 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_599 = seg_base + (local_tid ^ 8u); let tmp_600 = smem_keys[tmp_599 * WPT + 0u]; let tmp_601 = smem_vals[tmp_599 * WPT + 0u]; let tmp_602 = keys[0] < tmp_600 || (keys[0] == tmp_600 && values[0] < tmp_601); if tmp_598 == tmp_602 { keys[0] = tmp_600; values[0] = tmp_601; } let tmp_603 = smem_keys[tmp_599 * WPT + 1u]; let tmp_604 = smem_vals[tmp_599 * WPT + 1u]; let tmp_605 = keys[1] < tmp_603 || (keys[1] == tmp_603 && values[1] < tmp_604); if tmp_598 == tmp_605 { keys[1] = tmp_603; values[1] = tmp_604; } let tmp_606 = smem_keys[tmp_599 * WPT + 2u]; let tmp_607 = smem_vals[tmp_599 * WPT + 2u]; let tmp_608 = keys[2] < tmp_606 || (keys[2] == tmp_606 && values[2] < tmp_607); if tmp_598 == tmp_608 { keys[2] = tmp_606; values[2] = tmp_607; } let tmp_609 = smem_keys[tmp_599 * WPT + 3u]; let tmp_610 = smem_vals[tmp_599 * WPT + 3u]; let tmp_611 = keys[3] < tmp_609 || (keys[3] == tmp_609 && values[3] < tmp_610); if tmp_598 == tmp_611 { keys[3] = tmp_609; values[3] = tmp_610; } let tmp_612 = smem_keys[tmp_599 * WPT + 4u]; let tmp_613 = smem_vals[tmp_599 * WPT + 4u]; let tmp_614 = keys[4] < tmp_612 || (keys[4] == tmp_612 && values[4] < tmp_613); if tmp_598 == tmp_614 { keys[4] = tmp_612; values[4] = tmp_613; } let tmp_615 = smem_keys[tmp_599 * WPT + 5u]; let tmp_616 = smem_vals[tmp_599 * WPT + 5u]; let tmp_617 = keys[5] < tmp_615 || (keys[5] == tmp_615 && values[5] < tmp_616); if tmp_598 == tmp_617 { keys[5] = tmp_615; values[5] = tmp_616; } let tmp_618 = smem_keys[tmp_599 * WPT + 6u]; let tmp_619 = smem_vals[tmp_599 * WPT + 6u]; let tmp_620 = keys[6] < tmp_618 || (keys[6] == tmp_618 && values[6] < tmp_619); if tmp_598 == tmp_620 { keys[6] = tmp_618; values[6] = tmp_619; } let tmp_621 = smem_keys[tmp_599 * WPT + 7u]; let tmp_622 = smem_vals[tmp_599 * WPT + 7u]; let tmp_623 = keys[7] < tmp_621 || (keys[7] == tmp_621 && values[7] < tmp_622); if tmp_598 == tmp_623 { keys[7] = tmp_621; values[7] = tmp_622; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:8) 
    {
    let tmp_624 = subgroupShuffleXor(keys[0], 4u);
    let tmp_625 = subgroupShuffleXor(values[0], 4u);
    let tmp_626 = subgroupShuffleXor(keys[1], 4u);
    let tmp_627 = subgroupShuffleXor(values[1], 4u);
    let tmp_628 = subgroupShuffleXor(keys[2], 4u);
    let tmp_629 = subgroupShuffleXor(values[2], 4u);
    let tmp_630 = subgroupShuffleXor(keys[3], 4u);
    let tmp_631 = subgroupShuffleXor(values[3], 4u);
    let tmp_632 = subgroupShuffleXor(keys[4], 4u);
    let tmp_633 = subgroupShuffleXor(values[4], 4u);
    let tmp_634 = subgroupShuffleXor(keys[5], 4u);
    let tmp_635 = subgroupShuffleXor(values[5], 4u);
    let tmp_636 = subgroupShuffleXor(keys[6], 4u);
    let tmp_637 = subgroupShuffleXor(values[6], 4u);
    let tmp_638 = subgroupShuffleXor(keys[7], 4u);
    let tmp_639 = subgroupShuffleXor(values[7], 4u);
    let tmp_640 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_641 = keys[0] < tmp_624 || (keys[0] == tmp_624 && values[0] < tmp_625);
    if tmp_640 == tmp_641 { keys[0] = tmp_624; values[0] = tmp_625; }
    let tmp_642 = keys[1] < tmp_626 || (keys[1] == tmp_626 && values[1] < tmp_627);
    if tmp_640 == tmp_642 { keys[1] = tmp_626; values[1] = tmp_627; }
    let tmp_643 = keys[2] < tmp_628 || (keys[2] == tmp_628 && values[2] < tmp_629);
    if tmp_640 == tmp_643 { keys[2] = tmp_628; values[2] = tmp_629; }
    let tmp_644 = keys[3] < tmp_630 || (keys[3] == tmp_630 && values[3] < tmp_631);
    if tmp_640 == tmp_644 { keys[3] = tmp_630; values[3] = tmp_631; }
    let tmp_645 = keys[4] < tmp_632 || (keys[4] == tmp_632 && values[4] < tmp_633);
    if tmp_640 == tmp_645 { keys[4] = tmp_632; values[4] = tmp_633; }
    let tmp_646 = keys[5] < tmp_634 || (keys[5] == tmp_634 && values[5] < tmp_635);
    if tmp_640 == tmp_646 { keys[5] = tmp_634; values[5] = tmp_635; }
    let tmp_647 = keys[6] < tmp_636 || (keys[6] == tmp_636 && values[6] < tmp_637);
    if tmp_640 == tmp_647 { keys[6] = tmp_636; values[6] = tmp_637; }
    let tmp_648 = keys[7] < tmp_638 || (keys[7] == tmp_638 && values[7] < tmp_639);
    if tmp_640 == tmp_648 { keys[7] = tmp_638; values[7] = tmp_639; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    {
    let tmp_649 = subgroupShuffleXor(keys[0], 2u);
    let tmp_650 = subgroupShuffleXor(values[0], 2u);
    let tmp_651 = subgroupShuffleXor(keys[1], 2u);
    let tmp_652 = subgroupShuffleXor(values[1], 2u);
    let tmp_653 = subgroupShuffleXor(keys[2], 2u);
    let tmp_654 = subgroupShuffleXor(values[2], 2u);
    let tmp_655 = subgroupShuffleXor(keys[3], 2u);
    let tmp_656 = subgroupShuffleXor(values[3], 2u);
    let tmp_657 = subgroupShuffleXor(keys[4], 2u);
    let tmp_658 = subgroupShuffleXor(values[4], 2u);
    let tmp_659 = subgroupShuffleXor(keys[5], 2u);
    let tmp_660 = subgroupShuffleXor(values[5], 2u);
    let tmp_661 = subgroupShuffleXor(keys[6], 2u);
    let tmp_662 = subgroupShuffleXor(values[6], 2u);
    let tmp_663 = subgroupShuffleXor(keys[7], 2u);
    let tmp_664 = subgroupShuffleXor(values[7], 2u);
    let tmp_665 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_666 = keys[0] < tmp_649 || (keys[0] == tmp_649 && values[0] < tmp_650);
    if tmp_665 == tmp_666 { keys[0] = tmp_649; values[0] = tmp_650; }
    let tmp_667 = keys[1] < tmp_651 || (keys[1] == tmp_651 && values[1] < tmp_652);
    if tmp_665 == tmp_667 { keys[1] = tmp_651; values[1] = tmp_652; }
    let tmp_668 = keys[2] < tmp_653 || (keys[2] == tmp_653 && values[2] < tmp_654);
    if tmp_665 == tmp_668 { keys[2] = tmp_653; values[2] = tmp_654; }
    let tmp_669 = keys[3] < tmp_655 || (keys[3] == tmp_655 && values[3] < tmp_656);
    if tmp_665 == tmp_669 { keys[3] = tmp_655; values[3] = tmp_656; }
    let tmp_670 = keys[4] < tmp_657 || (keys[4] == tmp_657 && values[4] < tmp_658);
    if tmp_665 == tmp_670 { keys[4] = tmp_657; values[4] = tmp_658; }
    let tmp_671 = keys[5] < tmp_659 || (keys[5] == tmp_659 && values[5] < tmp_660);
    if tmp_665 == tmp_671 { keys[5] = tmp_659; values[5] = tmp_660; }
    let tmp_672 = keys[6] < tmp_661 || (keys[6] == tmp_661 && values[6] < tmp_662);
    if tmp_665 == tmp_672 { keys[6] = tmp_661; values[6] = tmp_662; }
    let tmp_673 = keys[7] < tmp_663 || (keys[7] == tmp_663 && values[7] < tmp_664);
    if tmp_665 == tmp_673 { keys[7] = tmp_663; values[7] = tmp_664; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    {
    let tmp_674 = subgroupShuffleXor(keys[0], 1u);
    let tmp_675 = subgroupShuffleXor(values[0], 1u);
    let tmp_676 = subgroupShuffleXor(keys[1], 1u);
    let tmp_677 = subgroupShuffleXor(values[1], 1u);
    let tmp_678 = subgroupShuffleXor(keys[2], 1u);
    let tmp_679 = subgroupShuffleXor(values[2], 1u);
    let tmp_680 = subgroupShuffleXor(keys[3], 1u);
    let tmp_681 = subgroupShuffleXor(values[3], 1u);
    let tmp_682 = subgroupShuffleXor(keys[4], 1u);
    let tmp_683 = subgroupShuffleXor(values[4], 1u);
    let tmp_684 = subgroupShuffleXor(keys[5], 1u);
    let tmp_685 = subgroupShuffleXor(values[5], 1u);
    let tmp_686 = subgroupShuffleXor(keys[6], 1u);
    let tmp_687 = subgroupShuffleXor(values[6], 1u);
    let tmp_688 = subgroupShuffleXor(keys[7], 1u);
    let tmp_689 = subgroupShuffleXor(values[7], 1u);
    let tmp_690 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_691 = keys[0] < tmp_674 || (keys[0] == tmp_674 && values[0] < tmp_675);
    if tmp_690 == tmp_691 { keys[0] = tmp_674; values[0] = tmp_675; }
    let tmp_692 = keys[1] < tmp_676 || (keys[1] == tmp_676 && values[1] < tmp_677);
    if tmp_690 == tmp_692 { keys[1] = tmp_676; values[1] = tmp_677; }
    let tmp_693 = keys[2] < tmp_678 || (keys[2] == tmp_678 && values[2] < tmp_679);
    if tmp_690 == tmp_693 { keys[2] = tmp_678; values[2] = tmp_679; }
    let tmp_694 = keys[3] < tmp_680 || (keys[3] == tmp_680 && values[3] < tmp_681);
    if tmp_690 == tmp_694 { keys[3] = tmp_680; values[3] = tmp_681; }
    let tmp_695 = keys[4] < tmp_682 || (keys[4] == tmp_682 && values[4] < tmp_683);
    if tmp_690 == tmp_695 { keys[4] = tmp_682; values[4] = tmp_683; }
    let tmp_696 = keys[5] < tmp_684 || (keys[5] == tmp_684 && values[5] < tmp_685);
    if tmp_690 == tmp_696 { keys[5] = tmp_684; values[5] = tmp_685; }
    let tmp_697 = keys[6] < tmp_686 || (keys[6] == tmp_686 && values[6] < tmp_687);
    if tmp_690 == tmp_697 { keys[6] = tmp_686; values[6] = tmp_687; }
    let tmp_698 = keys[7] < tmp_688 || (keys[7] == tmp_688 && values[7] < tmp_689);
    if tmp_690 == tmp_698 { keys[7] = tmp_688; values[7] = tmp_689; }
    }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_699 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_699;let tmp_700 = values[0]; values[0] = values[4]; values[4] = tmp_700; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_701 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_701;let tmp_702 = values[1]; values[1] = values[5]; values[5] = tmp_702; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_703 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_703;let tmp_704 = values[2]; values[2] = values[6]; values[6] = tmp_704; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_705 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_705;let tmp_706 = values[3]; values[3] = values[7]; values[7] = tmp_706; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_707 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_707;let tmp_708 = values[0]; values[0] = values[2]; values[2] = tmp_708; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_709 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_709;let tmp_710 = values[1]; values[1] = values[3]; values[3] = tmp_710; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_711 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_711;let tmp_712 = values[4]; values[4] = values[6]; values[6] = tmp_712; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_713 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_713;let tmp_714 = values[5]; values[5] = values[7]; values[7] = tmp_714; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_715 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_715;let tmp_716 = values[0]; values[0] = values[1]; values[1] = tmp_716; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_717 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_717;let tmp_718 = values[2]; values[2] = values[3]; values[3] = tmp_718; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_719 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_719;let tmp_720 = values[4]; values[4] = values[5]; values[5] = tmp_720; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_721 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_721;let tmp_722 = values[6]; values[6] = values[7]; values[7] = tmp_722; }
    }

    // striped (coalesced) store via shared memory
    for (var r = 0u; r < WPT; r = r + 1u) {
        smem_keys[local_tid * WPT + r] = keys[r];
        smem_vals[local_tid * WPT + r] = values[r];
    }
    workgroupBarrier();
    for (var c = 0u; c < WPT; c = c + 1u) {
        let j = c * M + local_tid;
        if is_active && j < seg_size {
            global_keys[seg_start + j] = smem_keys[j];
            global_value_indices[seg_start + j] = smem_vals[j];
        }
    }
}
