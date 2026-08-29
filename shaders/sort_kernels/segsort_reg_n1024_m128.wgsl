
enable subgroups;

override WG: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 1024u;
const M: u32 = 128u;
const WPT: u32 = 8u;

@compute @workgroup_size(WG, 1, 1)
fn segsort_reg_n1024_m128(
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    const BIN: u32 = 10u

    let bin_base = select(bin_offsets[BIN - 1u], 0u, BIN == 0u);
    let bin_count = bin_offsets[BIN] - bin_base;

    let local_tid = sid & (M - 1u);
    let global_seg = (wg_id.x * WG + lid) / M;

    let active = global_seg < bin_count;
    let slot = bin_base + select(0u, global_seg, active);   // clamp so the read is in-range
    let seg_id = bin_indices[slot];
    let seg_start = select(segments[seg_id - 1u], 0u, seg_id == 0u);
    let seg_end = segments[seg_id];
    let seg_size = select(0u, seg_end - seg_start, active);

    var keys: array<u32, 8>;
    var values: array<u32, 8>;

    for (var r = 0u; r < WPT; r = r + 1u) {
        let pos = local_tid * WPT + r;
        if active && pos < seg_size {
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
    {
    let tmp_394 = subgroupShuffleXor(keys[7], 31u);
    let tmp_395 = subgroupShuffleXor(values[7], 31u);
    let tmp_396 = subgroupShuffleXor(keys[6], 31u);
    let tmp_397 = subgroupShuffleXor(values[6], 31u);
    let tmp_398 = subgroupShuffleXor(keys[5], 31u);
    let tmp_399 = subgroupShuffleXor(values[5], 31u);
    let tmp_400 = subgroupShuffleXor(keys[4], 31u);
    let tmp_401 = subgroupShuffleXor(values[4], 31u);
    let tmp_402 = subgroupShuffleXor(keys[3], 31u);
    let tmp_403 = subgroupShuffleXor(values[3], 31u);
    let tmp_404 = subgroupShuffleXor(keys[2], 31u);
    let tmp_405 = subgroupShuffleXor(values[2], 31u);
    let tmp_406 = subgroupShuffleXor(keys[1], 31u);
    let tmp_407 = subgroupShuffleXor(values[1], 31u);
    let tmp_408 = subgroupShuffleXor(keys[0], 31u);
    let tmp_409 = subgroupShuffleXor(values[0], 31u);
    let tmp_410 = extractBits(local_tid, 4u, 1u) != 0u;
    let tmp_411 = keys[0] < tmp_394 || (keys[0] == tmp_394 && values[0] < tmp_395);
    if tmp_410 == tmp_411 { keys[0] = tmp_394; values[0] = tmp_395; }
    let tmp_412 = keys[1] < tmp_396 || (keys[1] == tmp_396 && values[1] < tmp_397);
    if tmp_410 == tmp_412 { keys[1] = tmp_396; values[1] = tmp_397; }
    let tmp_413 = keys[2] < tmp_398 || (keys[2] == tmp_398 && values[2] < tmp_399);
    if tmp_410 == tmp_413 { keys[2] = tmp_398; values[2] = tmp_399; }
    let tmp_414 = keys[3] < tmp_400 || (keys[3] == tmp_400 && values[3] < tmp_401);
    if tmp_410 == tmp_414 { keys[3] = tmp_400; values[3] = tmp_401; }
    let tmp_415 = keys[4] < tmp_402 || (keys[4] == tmp_402 && values[4] < tmp_403);
    if tmp_410 == tmp_415 { keys[4] = tmp_402; values[4] = tmp_403; }
    let tmp_416 = keys[5] < tmp_404 || (keys[5] == tmp_404 && values[5] < tmp_405);
    if tmp_410 == tmp_416 { keys[5] = tmp_404; values[5] = tmp_405; }
    let tmp_417 = keys[6] < tmp_406 || (keys[6] == tmp_406 && values[6] < tmp_407);
    if tmp_410 == tmp_417 { keys[6] = tmp_406; values[6] = tmp_407; }
    let tmp_418 = keys[7] < tmp_408 || (keys[7] == tmp_408 && values[7] < tmp_409);
    if tmp_410 == tmp_418 { keys[7] = tmp_408; values[7] = tmp_409; }
    }
    // exch_paral(tmask:8,swbit:3,wpt:8) 
    {
    let tmp_419 = subgroupShuffleXor(keys[0], 8u);
    let tmp_420 = subgroupShuffleXor(values[0], 8u);
    let tmp_421 = subgroupShuffleXor(keys[1], 8u);
    let tmp_422 = subgroupShuffleXor(values[1], 8u);
    let tmp_423 = subgroupShuffleXor(keys[2], 8u);
    let tmp_424 = subgroupShuffleXor(values[2], 8u);
    let tmp_425 = subgroupShuffleXor(keys[3], 8u);
    let tmp_426 = subgroupShuffleXor(values[3], 8u);
    let tmp_427 = subgroupShuffleXor(keys[4], 8u);
    let tmp_428 = subgroupShuffleXor(values[4], 8u);
    let tmp_429 = subgroupShuffleXor(keys[5], 8u);
    let tmp_430 = subgroupShuffleXor(values[5], 8u);
    let tmp_431 = subgroupShuffleXor(keys[6], 8u);
    let tmp_432 = subgroupShuffleXor(values[6], 8u);
    let tmp_433 = subgroupShuffleXor(keys[7], 8u);
    let tmp_434 = subgroupShuffleXor(values[7], 8u);
    let tmp_435 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_436 = keys[0] < tmp_419 || (keys[0] == tmp_419 && values[0] < tmp_420);
    if tmp_435 == tmp_436 { keys[0] = tmp_419; values[0] = tmp_420; }
    let tmp_437 = keys[1] < tmp_421 || (keys[1] == tmp_421 && values[1] < tmp_422);
    if tmp_435 == tmp_437 { keys[1] = tmp_421; values[1] = tmp_422; }
    let tmp_438 = keys[2] < tmp_423 || (keys[2] == tmp_423 && values[2] < tmp_424);
    if tmp_435 == tmp_438 { keys[2] = tmp_423; values[2] = tmp_424; }
    let tmp_439 = keys[3] < tmp_425 || (keys[3] == tmp_425 && values[3] < tmp_426);
    if tmp_435 == tmp_439 { keys[3] = tmp_425; values[3] = tmp_426; }
    let tmp_440 = keys[4] < tmp_427 || (keys[4] == tmp_427 && values[4] < tmp_428);
    if tmp_435 == tmp_440 { keys[4] = tmp_427; values[4] = tmp_428; }
    let tmp_441 = keys[5] < tmp_429 || (keys[5] == tmp_429 && values[5] < tmp_430);
    if tmp_435 == tmp_441 { keys[5] = tmp_429; values[5] = tmp_430; }
    let tmp_442 = keys[6] < tmp_431 || (keys[6] == tmp_431 && values[6] < tmp_432);
    if tmp_435 == tmp_442 { keys[6] = tmp_431; values[6] = tmp_432; }
    let tmp_443 = keys[7] < tmp_433 || (keys[7] == tmp_433 && values[7] < tmp_434);
    if tmp_435 == tmp_443 { keys[7] = tmp_433; values[7] = tmp_434; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:8) 
    {
    let tmp_444 = subgroupShuffleXor(keys[0], 4u);
    let tmp_445 = subgroupShuffleXor(values[0], 4u);
    let tmp_446 = subgroupShuffleXor(keys[1], 4u);
    let tmp_447 = subgroupShuffleXor(values[1], 4u);
    let tmp_448 = subgroupShuffleXor(keys[2], 4u);
    let tmp_449 = subgroupShuffleXor(values[2], 4u);
    let tmp_450 = subgroupShuffleXor(keys[3], 4u);
    let tmp_451 = subgroupShuffleXor(values[3], 4u);
    let tmp_452 = subgroupShuffleXor(keys[4], 4u);
    let tmp_453 = subgroupShuffleXor(values[4], 4u);
    let tmp_454 = subgroupShuffleXor(keys[5], 4u);
    let tmp_455 = subgroupShuffleXor(values[5], 4u);
    let tmp_456 = subgroupShuffleXor(keys[6], 4u);
    let tmp_457 = subgroupShuffleXor(values[6], 4u);
    let tmp_458 = subgroupShuffleXor(keys[7], 4u);
    let tmp_459 = subgroupShuffleXor(values[7], 4u);
    let tmp_460 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_461 = keys[0] < tmp_444 || (keys[0] == tmp_444 && values[0] < tmp_445);
    if tmp_460 == tmp_461 { keys[0] = tmp_444; values[0] = tmp_445; }
    let tmp_462 = keys[1] < tmp_446 || (keys[1] == tmp_446 && values[1] < tmp_447);
    if tmp_460 == tmp_462 { keys[1] = tmp_446; values[1] = tmp_447; }
    let tmp_463 = keys[2] < tmp_448 || (keys[2] == tmp_448 && values[2] < tmp_449);
    if tmp_460 == tmp_463 { keys[2] = tmp_448; values[2] = tmp_449; }
    let tmp_464 = keys[3] < tmp_450 || (keys[3] == tmp_450 && values[3] < tmp_451);
    if tmp_460 == tmp_464 { keys[3] = tmp_450; values[3] = tmp_451; }
    let tmp_465 = keys[4] < tmp_452 || (keys[4] == tmp_452 && values[4] < tmp_453);
    if tmp_460 == tmp_465 { keys[4] = tmp_452; values[4] = tmp_453; }
    let tmp_466 = keys[5] < tmp_454 || (keys[5] == tmp_454 && values[5] < tmp_455);
    if tmp_460 == tmp_466 { keys[5] = tmp_454; values[5] = tmp_455; }
    let tmp_467 = keys[6] < tmp_456 || (keys[6] == tmp_456 && values[6] < tmp_457);
    if tmp_460 == tmp_467 { keys[6] = tmp_456; values[6] = tmp_457; }
    let tmp_468 = keys[7] < tmp_458 || (keys[7] == tmp_458 && values[7] < tmp_459);
    if tmp_460 == tmp_468 { keys[7] = tmp_458; values[7] = tmp_459; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    {
    let tmp_469 = subgroupShuffleXor(keys[0], 2u);
    let tmp_470 = subgroupShuffleXor(values[0], 2u);
    let tmp_471 = subgroupShuffleXor(keys[1], 2u);
    let tmp_472 = subgroupShuffleXor(values[1], 2u);
    let tmp_473 = subgroupShuffleXor(keys[2], 2u);
    let tmp_474 = subgroupShuffleXor(values[2], 2u);
    let tmp_475 = subgroupShuffleXor(keys[3], 2u);
    let tmp_476 = subgroupShuffleXor(values[3], 2u);
    let tmp_477 = subgroupShuffleXor(keys[4], 2u);
    let tmp_478 = subgroupShuffleXor(values[4], 2u);
    let tmp_479 = subgroupShuffleXor(keys[5], 2u);
    let tmp_480 = subgroupShuffleXor(values[5], 2u);
    let tmp_481 = subgroupShuffleXor(keys[6], 2u);
    let tmp_482 = subgroupShuffleXor(values[6], 2u);
    let tmp_483 = subgroupShuffleXor(keys[7], 2u);
    let tmp_484 = subgroupShuffleXor(values[7], 2u);
    let tmp_485 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_486 = keys[0] < tmp_469 || (keys[0] == tmp_469 && values[0] < tmp_470);
    if tmp_485 == tmp_486 { keys[0] = tmp_469; values[0] = tmp_470; }
    let tmp_487 = keys[1] < tmp_471 || (keys[1] == tmp_471 && values[1] < tmp_472);
    if tmp_485 == tmp_487 { keys[1] = tmp_471; values[1] = tmp_472; }
    let tmp_488 = keys[2] < tmp_473 || (keys[2] == tmp_473 && values[2] < tmp_474);
    if tmp_485 == tmp_488 { keys[2] = tmp_473; values[2] = tmp_474; }
    let tmp_489 = keys[3] < tmp_475 || (keys[3] == tmp_475 && values[3] < tmp_476);
    if tmp_485 == tmp_489 { keys[3] = tmp_475; values[3] = tmp_476; }
    let tmp_490 = keys[4] < tmp_477 || (keys[4] == tmp_477 && values[4] < tmp_478);
    if tmp_485 == tmp_490 { keys[4] = tmp_477; values[4] = tmp_478; }
    let tmp_491 = keys[5] < tmp_479 || (keys[5] == tmp_479 && values[5] < tmp_480);
    if tmp_485 == tmp_491 { keys[5] = tmp_479; values[5] = tmp_480; }
    let tmp_492 = keys[6] < tmp_481 || (keys[6] == tmp_481 && values[6] < tmp_482);
    if tmp_485 == tmp_492 { keys[6] = tmp_481; values[6] = tmp_482; }
    let tmp_493 = keys[7] < tmp_483 || (keys[7] == tmp_483 && values[7] < tmp_484);
    if tmp_485 == tmp_493 { keys[7] = tmp_483; values[7] = tmp_484; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    {
    let tmp_494 = subgroupShuffleXor(keys[0], 1u);
    let tmp_495 = subgroupShuffleXor(values[0], 1u);
    let tmp_496 = subgroupShuffleXor(keys[1], 1u);
    let tmp_497 = subgroupShuffleXor(values[1], 1u);
    let tmp_498 = subgroupShuffleXor(keys[2], 1u);
    let tmp_499 = subgroupShuffleXor(values[2], 1u);
    let tmp_500 = subgroupShuffleXor(keys[3], 1u);
    let tmp_501 = subgroupShuffleXor(values[3], 1u);
    let tmp_502 = subgroupShuffleXor(keys[4], 1u);
    let tmp_503 = subgroupShuffleXor(values[4], 1u);
    let tmp_504 = subgroupShuffleXor(keys[5], 1u);
    let tmp_505 = subgroupShuffleXor(values[5], 1u);
    let tmp_506 = subgroupShuffleXor(keys[6], 1u);
    let tmp_507 = subgroupShuffleXor(values[6], 1u);
    let tmp_508 = subgroupShuffleXor(keys[7], 1u);
    let tmp_509 = subgroupShuffleXor(values[7], 1u);
    let tmp_510 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_511 = keys[0] < tmp_494 || (keys[0] == tmp_494 && values[0] < tmp_495);
    if tmp_510 == tmp_511 { keys[0] = tmp_494; values[0] = tmp_495; }
    let tmp_512 = keys[1] < tmp_496 || (keys[1] == tmp_496 && values[1] < tmp_497);
    if tmp_510 == tmp_512 { keys[1] = tmp_496; values[1] = tmp_497; }
    let tmp_513 = keys[2] < tmp_498 || (keys[2] == tmp_498 && values[2] < tmp_499);
    if tmp_510 == tmp_513 { keys[2] = tmp_498; values[2] = tmp_499; }
    let tmp_514 = keys[3] < tmp_500 || (keys[3] == tmp_500 && values[3] < tmp_501);
    if tmp_510 == tmp_514 { keys[3] = tmp_500; values[3] = tmp_501; }
    let tmp_515 = keys[4] < tmp_502 || (keys[4] == tmp_502 && values[4] < tmp_503);
    if tmp_510 == tmp_515 { keys[4] = tmp_502; values[4] = tmp_503; }
    let tmp_516 = keys[5] < tmp_504 || (keys[5] == tmp_504 && values[5] < tmp_505);
    if tmp_510 == tmp_516 { keys[5] = tmp_504; values[5] = tmp_505; }
    let tmp_517 = keys[6] < tmp_506 || (keys[6] == tmp_506 && values[6] < tmp_507);
    if tmp_510 == tmp_517 { keys[6] = tmp_506; values[6] = tmp_507; }
    let tmp_518 = keys[7] < tmp_508 || (keys[7] == tmp_508 && values[7] < tmp_509);
    if tmp_510 == tmp_518 { keys[7] = tmp_508; values[7] = tmp_509; }
    }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_519 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_519;let tmp_520 = values[0]; values[0] = values[4]; values[4] = tmp_520; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_521 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_521;let tmp_522 = values[1]; values[1] = values[5]; values[5] = tmp_522; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_523 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_523;let tmp_524 = values[2]; values[2] = values[6]; values[6] = tmp_524; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_525 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_525;let tmp_526 = values[3]; values[3] = values[7]; values[7] = tmp_526; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_527 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_527;let tmp_528 = values[0]; values[0] = values[2]; values[2] = tmp_528; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_529 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_529;let tmp_530 = values[1]; values[1] = values[3]; values[3] = tmp_530; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_531 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_531;let tmp_532 = values[4]; values[4] = values[6]; values[6] = tmp_532; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_533 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_533;let tmp_534 = values[5]; values[5] = values[7]; values[7] = tmp_534; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_535 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_535;let tmp_536 = values[0]; values[0] = values[1]; values[1] = tmp_536; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_537 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_537;let tmp_538 = values[2]; values[2] = values[3]; values[3] = tmp_538; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_539 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_539;let tmp_540 = values[4]; values[4] = values[5]; values[5] = tmp_540; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_541 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_541;let tmp_542 = values[6]; values[6] = values[7]; values[7] = tmp_542; }
    }
    // exch_intxn(tmask:63,swbit:5,wpt:8)
    {
    let tmp_543 = subgroupShuffleXor(keys[7], 63u);
    let tmp_544 = subgroupShuffleXor(values[7], 63u);
    let tmp_545 = subgroupShuffleXor(keys[6], 63u);
    let tmp_546 = subgroupShuffleXor(values[6], 63u);
    let tmp_547 = subgroupShuffleXor(keys[5], 63u);
    let tmp_548 = subgroupShuffleXor(values[5], 63u);
    let tmp_549 = subgroupShuffleXor(keys[4], 63u);
    let tmp_550 = subgroupShuffleXor(values[4], 63u);
    let tmp_551 = subgroupShuffleXor(keys[3], 63u);
    let tmp_552 = subgroupShuffleXor(values[3], 63u);
    let tmp_553 = subgroupShuffleXor(keys[2], 63u);
    let tmp_554 = subgroupShuffleXor(values[2], 63u);
    let tmp_555 = subgroupShuffleXor(keys[1], 63u);
    let tmp_556 = subgroupShuffleXor(values[1], 63u);
    let tmp_557 = subgroupShuffleXor(keys[0], 63u);
    let tmp_558 = subgroupShuffleXor(values[0], 63u);
    let tmp_559 = extractBits(local_tid, 5u, 1u) != 0u;
    let tmp_560 = keys[0] < tmp_543 || (keys[0] == tmp_543 && values[0] < tmp_544);
    if tmp_559 == tmp_560 { keys[0] = tmp_543; values[0] = tmp_544; }
    let tmp_561 = keys[1] < tmp_545 || (keys[1] == tmp_545 && values[1] < tmp_546);
    if tmp_559 == tmp_561 { keys[1] = tmp_545; values[1] = tmp_546; }
    let tmp_562 = keys[2] < tmp_547 || (keys[2] == tmp_547 && values[2] < tmp_548);
    if tmp_559 == tmp_562 { keys[2] = tmp_547; values[2] = tmp_548; }
    let tmp_563 = keys[3] < tmp_549 || (keys[3] == tmp_549 && values[3] < tmp_550);
    if tmp_559 == tmp_563 { keys[3] = tmp_549; values[3] = tmp_550; }
    let tmp_564 = keys[4] < tmp_551 || (keys[4] == tmp_551 && values[4] < tmp_552);
    if tmp_559 == tmp_564 { keys[4] = tmp_551; values[4] = tmp_552; }
    let tmp_565 = keys[5] < tmp_553 || (keys[5] == tmp_553 && values[5] < tmp_554);
    if tmp_559 == tmp_565 { keys[5] = tmp_553; values[5] = tmp_554; }
    let tmp_566 = keys[6] < tmp_555 || (keys[6] == tmp_555 && values[6] < tmp_556);
    if tmp_559 == tmp_566 { keys[6] = tmp_555; values[6] = tmp_556; }
    let tmp_567 = keys[7] < tmp_557 || (keys[7] == tmp_557 && values[7] < tmp_558);
    if tmp_559 == tmp_567 { keys[7] = tmp_557; values[7] = tmp_558; }
    }
    // exch_paral(tmask:16,swbit:4,wpt:8) 
    {
    let tmp_568 = subgroupShuffleXor(keys[0], 16u);
    let tmp_569 = subgroupShuffleXor(values[0], 16u);
    let tmp_570 = subgroupShuffleXor(keys[1], 16u);
    let tmp_571 = subgroupShuffleXor(values[1], 16u);
    let tmp_572 = subgroupShuffleXor(keys[2], 16u);
    let tmp_573 = subgroupShuffleXor(values[2], 16u);
    let tmp_574 = subgroupShuffleXor(keys[3], 16u);
    let tmp_575 = subgroupShuffleXor(values[3], 16u);
    let tmp_576 = subgroupShuffleXor(keys[4], 16u);
    let tmp_577 = subgroupShuffleXor(values[4], 16u);
    let tmp_578 = subgroupShuffleXor(keys[5], 16u);
    let tmp_579 = subgroupShuffleXor(values[5], 16u);
    let tmp_580 = subgroupShuffleXor(keys[6], 16u);
    let tmp_581 = subgroupShuffleXor(values[6], 16u);
    let tmp_582 = subgroupShuffleXor(keys[7], 16u);
    let tmp_583 = subgroupShuffleXor(values[7], 16u);
    let tmp_584 = extractBits(local_tid, 4u, 1u) != 0u;
    let tmp_585 = keys[0] < tmp_568 || (keys[0] == tmp_568 && values[0] < tmp_569);
    if tmp_584 == tmp_585 { keys[0] = tmp_568; values[0] = tmp_569; }
    let tmp_586 = keys[1] < tmp_570 || (keys[1] == tmp_570 && values[1] < tmp_571);
    if tmp_584 == tmp_586 { keys[1] = tmp_570; values[1] = tmp_571; }
    let tmp_587 = keys[2] < tmp_572 || (keys[2] == tmp_572 && values[2] < tmp_573);
    if tmp_584 == tmp_587 { keys[2] = tmp_572; values[2] = tmp_573; }
    let tmp_588 = keys[3] < tmp_574 || (keys[3] == tmp_574 && values[3] < tmp_575);
    if tmp_584 == tmp_588 { keys[3] = tmp_574; values[3] = tmp_575; }
    let tmp_589 = keys[4] < tmp_576 || (keys[4] == tmp_576 && values[4] < tmp_577);
    if tmp_584 == tmp_589 { keys[4] = tmp_576; values[4] = tmp_577; }
    let tmp_590 = keys[5] < tmp_578 || (keys[5] == tmp_578 && values[5] < tmp_579);
    if tmp_584 == tmp_590 { keys[5] = tmp_578; values[5] = tmp_579; }
    let tmp_591 = keys[6] < tmp_580 || (keys[6] == tmp_580 && values[6] < tmp_581);
    if tmp_584 == tmp_591 { keys[6] = tmp_580; values[6] = tmp_581; }
    let tmp_592 = keys[7] < tmp_582 || (keys[7] == tmp_582 && values[7] < tmp_583);
    if tmp_584 == tmp_592 { keys[7] = tmp_582; values[7] = tmp_583; }
    }
    // exch_paral(tmask:8,swbit:3,wpt:8) 
    {
    let tmp_593 = subgroupShuffleXor(keys[0], 8u);
    let tmp_594 = subgroupShuffleXor(values[0], 8u);
    let tmp_595 = subgroupShuffleXor(keys[1], 8u);
    let tmp_596 = subgroupShuffleXor(values[1], 8u);
    let tmp_597 = subgroupShuffleXor(keys[2], 8u);
    let tmp_598 = subgroupShuffleXor(values[2], 8u);
    let tmp_599 = subgroupShuffleXor(keys[3], 8u);
    let tmp_600 = subgroupShuffleXor(values[3], 8u);
    let tmp_601 = subgroupShuffleXor(keys[4], 8u);
    let tmp_602 = subgroupShuffleXor(values[4], 8u);
    let tmp_603 = subgroupShuffleXor(keys[5], 8u);
    let tmp_604 = subgroupShuffleXor(values[5], 8u);
    let tmp_605 = subgroupShuffleXor(keys[6], 8u);
    let tmp_606 = subgroupShuffleXor(values[6], 8u);
    let tmp_607 = subgroupShuffleXor(keys[7], 8u);
    let tmp_608 = subgroupShuffleXor(values[7], 8u);
    let tmp_609 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_610 = keys[0] < tmp_593 || (keys[0] == tmp_593 && values[0] < tmp_594);
    if tmp_609 == tmp_610 { keys[0] = tmp_593; values[0] = tmp_594; }
    let tmp_611 = keys[1] < tmp_595 || (keys[1] == tmp_595 && values[1] < tmp_596);
    if tmp_609 == tmp_611 { keys[1] = tmp_595; values[1] = tmp_596; }
    let tmp_612 = keys[2] < tmp_597 || (keys[2] == tmp_597 && values[2] < tmp_598);
    if tmp_609 == tmp_612 { keys[2] = tmp_597; values[2] = tmp_598; }
    let tmp_613 = keys[3] < tmp_599 || (keys[3] == tmp_599 && values[3] < tmp_600);
    if tmp_609 == tmp_613 { keys[3] = tmp_599; values[3] = tmp_600; }
    let tmp_614 = keys[4] < tmp_601 || (keys[4] == tmp_601 && values[4] < tmp_602);
    if tmp_609 == tmp_614 { keys[4] = tmp_601; values[4] = tmp_602; }
    let tmp_615 = keys[5] < tmp_603 || (keys[5] == tmp_603 && values[5] < tmp_604);
    if tmp_609 == tmp_615 { keys[5] = tmp_603; values[5] = tmp_604; }
    let tmp_616 = keys[6] < tmp_605 || (keys[6] == tmp_605 && values[6] < tmp_606);
    if tmp_609 == tmp_616 { keys[6] = tmp_605; values[6] = tmp_606; }
    let tmp_617 = keys[7] < tmp_607 || (keys[7] == tmp_607 && values[7] < tmp_608);
    if tmp_609 == tmp_617 { keys[7] = tmp_607; values[7] = tmp_608; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:8) 
    {
    let tmp_618 = subgroupShuffleXor(keys[0], 4u);
    let tmp_619 = subgroupShuffleXor(values[0], 4u);
    let tmp_620 = subgroupShuffleXor(keys[1], 4u);
    let tmp_621 = subgroupShuffleXor(values[1], 4u);
    let tmp_622 = subgroupShuffleXor(keys[2], 4u);
    let tmp_623 = subgroupShuffleXor(values[2], 4u);
    let tmp_624 = subgroupShuffleXor(keys[3], 4u);
    let tmp_625 = subgroupShuffleXor(values[3], 4u);
    let tmp_626 = subgroupShuffleXor(keys[4], 4u);
    let tmp_627 = subgroupShuffleXor(values[4], 4u);
    let tmp_628 = subgroupShuffleXor(keys[5], 4u);
    let tmp_629 = subgroupShuffleXor(values[5], 4u);
    let tmp_630 = subgroupShuffleXor(keys[6], 4u);
    let tmp_631 = subgroupShuffleXor(values[6], 4u);
    let tmp_632 = subgroupShuffleXor(keys[7], 4u);
    let tmp_633 = subgroupShuffleXor(values[7], 4u);
    let tmp_634 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_635 = keys[0] < tmp_618 || (keys[0] == tmp_618 && values[0] < tmp_619);
    if tmp_634 == tmp_635 { keys[0] = tmp_618; values[0] = tmp_619; }
    let tmp_636 = keys[1] < tmp_620 || (keys[1] == tmp_620 && values[1] < tmp_621);
    if tmp_634 == tmp_636 { keys[1] = tmp_620; values[1] = tmp_621; }
    let tmp_637 = keys[2] < tmp_622 || (keys[2] == tmp_622 && values[2] < tmp_623);
    if tmp_634 == tmp_637 { keys[2] = tmp_622; values[2] = tmp_623; }
    let tmp_638 = keys[3] < tmp_624 || (keys[3] == tmp_624 && values[3] < tmp_625);
    if tmp_634 == tmp_638 { keys[3] = tmp_624; values[3] = tmp_625; }
    let tmp_639 = keys[4] < tmp_626 || (keys[4] == tmp_626 && values[4] < tmp_627);
    if tmp_634 == tmp_639 { keys[4] = tmp_626; values[4] = tmp_627; }
    let tmp_640 = keys[5] < tmp_628 || (keys[5] == tmp_628 && values[5] < tmp_629);
    if tmp_634 == tmp_640 { keys[5] = tmp_628; values[5] = tmp_629; }
    let tmp_641 = keys[6] < tmp_630 || (keys[6] == tmp_630 && values[6] < tmp_631);
    if tmp_634 == tmp_641 { keys[6] = tmp_630; values[6] = tmp_631; }
    let tmp_642 = keys[7] < tmp_632 || (keys[7] == tmp_632 && values[7] < tmp_633);
    if tmp_634 == tmp_642 { keys[7] = tmp_632; values[7] = tmp_633; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    {
    let tmp_643 = subgroupShuffleXor(keys[0], 2u);
    let tmp_644 = subgroupShuffleXor(values[0], 2u);
    let tmp_645 = subgroupShuffleXor(keys[1], 2u);
    let tmp_646 = subgroupShuffleXor(values[1], 2u);
    let tmp_647 = subgroupShuffleXor(keys[2], 2u);
    let tmp_648 = subgroupShuffleXor(values[2], 2u);
    let tmp_649 = subgroupShuffleXor(keys[3], 2u);
    let tmp_650 = subgroupShuffleXor(values[3], 2u);
    let tmp_651 = subgroupShuffleXor(keys[4], 2u);
    let tmp_652 = subgroupShuffleXor(values[4], 2u);
    let tmp_653 = subgroupShuffleXor(keys[5], 2u);
    let tmp_654 = subgroupShuffleXor(values[5], 2u);
    let tmp_655 = subgroupShuffleXor(keys[6], 2u);
    let tmp_656 = subgroupShuffleXor(values[6], 2u);
    let tmp_657 = subgroupShuffleXor(keys[7], 2u);
    let tmp_658 = subgroupShuffleXor(values[7], 2u);
    let tmp_659 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_660 = keys[0] < tmp_643 || (keys[0] == tmp_643 && values[0] < tmp_644);
    if tmp_659 == tmp_660 { keys[0] = tmp_643; values[0] = tmp_644; }
    let tmp_661 = keys[1] < tmp_645 || (keys[1] == tmp_645 && values[1] < tmp_646);
    if tmp_659 == tmp_661 { keys[1] = tmp_645; values[1] = tmp_646; }
    let tmp_662 = keys[2] < tmp_647 || (keys[2] == tmp_647 && values[2] < tmp_648);
    if tmp_659 == tmp_662 { keys[2] = tmp_647; values[2] = tmp_648; }
    let tmp_663 = keys[3] < tmp_649 || (keys[3] == tmp_649 && values[3] < tmp_650);
    if tmp_659 == tmp_663 { keys[3] = tmp_649; values[3] = tmp_650; }
    let tmp_664 = keys[4] < tmp_651 || (keys[4] == tmp_651 && values[4] < tmp_652);
    if tmp_659 == tmp_664 { keys[4] = tmp_651; values[4] = tmp_652; }
    let tmp_665 = keys[5] < tmp_653 || (keys[5] == tmp_653 && values[5] < tmp_654);
    if tmp_659 == tmp_665 { keys[5] = tmp_653; values[5] = tmp_654; }
    let tmp_666 = keys[6] < tmp_655 || (keys[6] == tmp_655 && values[6] < tmp_656);
    if tmp_659 == tmp_666 { keys[6] = tmp_655; values[6] = tmp_656; }
    let tmp_667 = keys[7] < tmp_657 || (keys[7] == tmp_657 && values[7] < tmp_658);
    if tmp_659 == tmp_667 { keys[7] = tmp_657; values[7] = tmp_658; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    {
    let tmp_668 = subgroupShuffleXor(keys[0], 1u);
    let tmp_669 = subgroupShuffleXor(values[0], 1u);
    let tmp_670 = subgroupShuffleXor(keys[1], 1u);
    let tmp_671 = subgroupShuffleXor(values[1], 1u);
    let tmp_672 = subgroupShuffleXor(keys[2], 1u);
    let tmp_673 = subgroupShuffleXor(values[2], 1u);
    let tmp_674 = subgroupShuffleXor(keys[3], 1u);
    let tmp_675 = subgroupShuffleXor(values[3], 1u);
    let tmp_676 = subgroupShuffleXor(keys[4], 1u);
    let tmp_677 = subgroupShuffleXor(values[4], 1u);
    let tmp_678 = subgroupShuffleXor(keys[5], 1u);
    let tmp_679 = subgroupShuffleXor(values[5], 1u);
    let tmp_680 = subgroupShuffleXor(keys[6], 1u);
    let tmp_681 = subgroupShuffleXor(values[6], 1u);
    let tmp_682 = subgroupShuffleXor(keys[7], 1u);
    let tmp_683 = subgroupShuffleXor(values[7], 1u);
    let tmp_684 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_685 = keys[0] < tmp_668 || (keys[0] == tmp_668 && values[0] < tmp_669);
    if tmp_684 == tmp_685 { keys[0] = tmp_668; values[0] = tmp_669; }
    let tmp_686 = keys[1] < tmp_670 || (keys[1] == tmp_670 && values[1] < tmp_671);
    if tmp_684 == tmp_686 { keys[1] = tmp_670; values[1] = tmp_671; }
    let tmp_687 = keys[2] < tmp_672 || (keys[2] == tmp_672 && values[2] < tmp_673);
    if tmp_684 == tmp_687 { keys[2] = tmp_672; values[2] = tmp_673; }
    let tmp_688 = keys[3] < tmp_674 || (keys[3] == tmp_674 && values[3] < tmp_675);
    if tmp_684 == tmp_688 { keys[3] = tmp_674; values[3] = tmp_675; }
    let tmp_689 = keys[4] < tmp_676 || (keys[4] == tmp_676 && values[4] < tmp_677);
    if tmp_684 == tmp_689 { keys[4] = tmp_676; values[4] = tmp_677; }
    let tmp_690 = keys[5] < tmp_678 || (keys[5] == tmp_678 && values[5] < tmp_679);
    if tmp_684 == tmp_690 { keys[5] = tmp_678; values[5] = tmp_679; }
    let tmp_691 = keys[6] < tmp_680 || (keys[6] == tmp_680 && values[6] < tmp_681);
    if tmp_684 == tmp_691 { keys[6] = tmp_680; values[6] = tmp_681; }
    let tmp_692 = keys[7] < tmp_682 || (keys[7] == tmp_682 && values[7] < tmp_683);
    if tmp_684 == tmp_692 { keys[7] = tmp_682; values[7] = tmp_683; }
    }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_693 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_693;let tmp_694 = values[0]; values[0] = values[4]; values[4] = tmp_694; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_695 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_695;let tmp_696 = values[1]; values[1] = values[5]; values[5] = tmp_696; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_697 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_697;let tmp_698 = values[2]; values[2] = values[6]; values[6] = tmp_698; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_699 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_699;let tmp_700 = values[3]; values[3] = values[7]; values[7] = tmp_700; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_701 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_701;let tmp_702 = values[0]; values[0] = values[2]; values[2] = tmp_702; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_703 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_703;let tmp_704 = values[1]; values[1] = values[3]; values[3] = tmp_704; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_705 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_705;let tmp_706 = values[4]; values[4] = values[6]; values[6] = tmp_706; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_707 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_707;let tmp_708 = values[5]; values[5] = values[7]; values[7] = tmp_708; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_709 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_709;let tmp_710 = values[0]; values[0] = values[1]; values[1] = tmp_710; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_711 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_711;let tmp_712 = values[2]; values[2] = values[3]; values[3] = tmp_712; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_713 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_713;let tmp_714 = values[4]; values[4] = values[5]; values[5] = tmp_714; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_715 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_715;let tmp_716 = values[6]; values[6] = values[7]; values[7] = tmp_716; }
    }
    // exch_intxn(tmask:127,swbit:6,wpt:8)
    {
    let tmp_717 = subgroupShuffleXor(keys[7], 127u);
    let tmp_718 = subgroupShuffleXor(values[7], 127u);
    let tmp_719 = subgroupShuffleXor(keys[6], 127u);
    let tmp_720 = subgroupShuffleXor(values[6], 127u);
    let tmp_721 = subgroupShuffleXor(keys[5], 127u);
    let tmp_722 = subgroupShuffleXor(values[5], 127u);
    let tmp_723 = subgroupShuffleXor(keys[4], 127u);
    let tmp_724 = subgroupShuffleXor(values[4], 127u);
    let tmp_725 = subgroupShuffleXor(keys[3], 127u);
    let tmp_726 = subgroupShuffleXor(values[3], 127u);
    let tmp_727 = subgroupShuffleXor(keys[2], 127u);
    let tmp_728 = subgroupShuffleXor(values[2], 127u);
    let tmp_729 = subgroupShuffleXor(keys[1], 127u);
    let tmp_730 = subgroupShuffleXor(values[1], 127u);
    let tmp_731 = subgroupShuffleXor(keys[0], 127u);
    let tmp_732 = subgroupShuffleXor(values[0], 127u);
    let tmp_733 = extractBits(local_tid, 6u, 1u) != 0u;
    let tmp_734 = keys[0] < tmp_717 || (keys[0] == tmp_717 && values[0] < tmp_718);
    if tmp_733 == tmp_734 { keys[0] = tmp_717; values[0] = tmp_718; }
    let tmp_735 = keys[1] < tmp_719 || (keys[1] == tmp_719 && values[1] < tmp_720);
    if tmp_733 == tmp_735 { keys[1] = tmp_719; values[1] = tmp_720; }
    let tmp_736 = keys[2] < tmp_721 || (keys[2] == tmp_721 && values[2] < tmp_722);
    if tmp_733 == tmp_736 { keys[2] = tmp_721; values[2] = tmp_722; }
    let tmp_737 = keys[3] < tmp_723 || (keys[3] == tmp_723 && values[3] < tmp_724);
    if tmp_733 == tmp_737 { keys[3] = tmp_723; values[3] = tmp_724; }
    let tmp_738 = keys[4] < tmp_725 || (keys[4] == tmp_725 && values[4] < tmp_726);
    if tmp_733 == tmp_738 { keys[4] = tmp_725; values[4] = tmp_726; }
    let tmp_739 = keys[5] < tmp_727 || (keys[5] == tmp_727 && values[5] < tmp_728);
    if tmp_733 == tmp_739 { keys[5] = tmp_727; values[5] = tmp_728; }
    let tmp_740 = keys[6] < tmp_729 || (keys[6] == tmp_729 && values[6] < tmp_730);
    if tmp_733 == tmp_740 { keys[6] = tmp_729; values[6] = tmp_730; }
    let tmp_741 = keys[7] < tmp_731 || (keys[7] == tmp_731 && values[7] < tmp_732);
    if tmp_733 == tmp_741 { keys[7] = tmp_731; values[7] = tmp_732; }
    }
    // exch_paral(tmask:32,swbit:5,wpt:8) 
    {
    let tmp_742 = subgroupShuffleXor(keys[0], 32u);
    let tmp_743 = subgroupShuffleXor(values[0], 32u);
    let tmp_744 = subgroupShuffleXor(keys[1], 32u);
    let tmp_745 = subgroupShuffleXor(values[1], 32u);
    let tmp_746 = subgroupShuffleXor(keys[2], 32u);
    let tmp_747 = subgroupShuffleXor(values[2], 32u);
    let tmp_748 = subgroupShuffleXor(keys[3], 32u);
    let tmp_749 = subgroupShuffleXor(values[3], 32u);
    let tmp_750 = subgroupShuffleXor(keys[4], 32u);
    let tmp_751 = subgroupShuffleXor(values[4], 32u);
    let tmp_752 = subgroupShuffleXor(keys[5], 32u);
    let tmp_753 = subgroupShuffleXor(values[5], 32u);
    let tmp_754 = subgroupShuffleXor(keys[6], 32u);
    let tmp_755 = subgroupShuffleXor(values[6], 32u);
    let tmp_756 = subgroupShuffleXor(keys[7], 32u);
    let tmp_757 = subgroupShuffleXor(values[7], 32u);
    let tmp_758 = extractBits(local_tid, 5u, 1u) != 0u;
    let tmp_759 = keys[0] < tmp_742 || (keys[0] == tmp_742 && values[0] < tmp_743);
    if tmp_758 == tmp_759 { keys[0] = tmp_742; values[0] = tmp_743; }
    let tmp_760 = keys[1] < tmp_744 || (keys[1] == tmp_744 && values[1] < tmp_745);
    if tmp_758 == tmp_760 { keys[1] = tmp_744; values[1] = tmp_745; }
    let tmp_761 = keys[2] < tmp_746 || (keys[2] == tmp_746 && values[2] < tmp_747);
    if tmp_758 == tmp_761 { keys[2] = tmp_746; values[2] = tmp_747; }
    let tmp_762 = keys[3] < tmp_748 || (keys[3] == tmp_748 && values[3] < tmp_749);
    if tmp_758 == tmp_762 { keys[3] = tmp_748; values[3] = tmp_749; }
    let tmp_763 = keys[4] < tmp_750 || (keys[4] == tmp_750 && values[4] < tmp_751);
    if tmp_758 == tmp_763 { keys[4] = tmp_750; values[4] = tmp_751; }
    let tmp_764 = keys[5] < tmp_752 || (keys[5] == tmp_752 && values[5] < tmp_753);
    if tmp_758 == tmp_764 { keys[5] = tmp_752; values[5] = tmp_753; }
    let tmp_765 = keys[6] < tmp_754 || (keys[6] == tmp_754 && values[6] < tmp_755);
    if tmp_758 == tmp_765 { keys[6] = tmp_754; values[6] = tmp_755; }
    let tmp_766 = keys[7] < tmp_756 || (keys[7] == tmp_756 && values[7] < tmp_757);
    if tmp_758 == tmp_766 { keys[7] = tmp_756; values[7] = tmp_757; }
    }
    // exch_paral(tmask:16,swbit:4,wpt:8) 
    {
    let tmp_767 = subgroupShuffleXor(keys[0], 16u);
    let tmp_768 = subgroupShuffleXor(values[0], 16u);
    let tmp_769 = subgroupShuffleXor(keys[1], 16u);
    let tmp_770 = subgroupShuffleXor(values[1], 16u);
    let tmp_771 = subgroupShuffleXor(keys[2], 16u);
    let tmp_772 = subgroupShuffleXor(values[2], 16u);
    let tmp_773 = subgroupShuffleXor(keys[3], 16u);
    let tmp_774 = subgroupShuffleXor(values[3], 16u);
    let tmp_775 = subgroupShuffleXor(keys[4], 16u);
    let tmp_776 = subgroupShuffleXor(values[4], 16u);
    let tmp_777 = subgroupShuffleXor(keys[5], 16u);
    let tmp_778 = subgroupShuffleXor(values[5], 16u);
    let tmp_779 = subgroupShuffleXor(keys[6], 16u);
    let tmp_780 = subgroupShuffleXor(values[6], 16u);
    let tmp_781 = subgroupShuffleXor(keys[7], 16u);
    let tmp_782 = subgroupShuffleXor(values[7], 16u);
    let tmp_783 = extractBits(local_tid, 4u, 1u) != 0u;
    let tmp_784 = keys[0] < tmp_767 || (keys[0] == tmp_767 && values[0] < tmp_768);
    if tmp_783 == tmp_784 { keys[0] = tmp_767; values[0] = tmp_768; }
    let tmp_785 = keys[1] < tmp_769 || (keys[1] == tmp_769 && values[1] < tmp_770);
    if tmp_783 == tmp_785 { keys[1] = tmp_769; values[1] = tmp_770; }
    let tmp_786 = keys[2] < tmp_771 || (keys[2] == tmp_771 && values[2] < tmp_772);
    if tmp_783 == tmp_786 { keys[2] = tmp_771; values[2] = tmp_772; }
    let tmp_787 = keys[3] < tmp_773 || (keys[3] == tmp_773 && values[3] < tmp_774);
    if tmp_783 == tmp_787 { keys[3] = tmp_773; values[3] = tmp_774; }
    let tmp_788 = keys[4] < tmp_775 || (keys[4] == tmp_775 && values[4] < tmp_776);
    if tmp_783 == tmp_788 { keys[4] = tmp_775; values[4] = tmp_776; }
    let tmp_789 = keys[5] < tmp_777 || (keys[5] == tmp_777 && values[5] < tmp_778);
    if tmp_783 == tmp_789 { keys[5] = tmp_777; values[5] = tmp_778; }
    let tmp_790 = keys[6] < tmp_779 || (keys[6] == tmp_779 && values[6] < tmp_780);
    if tmp_783 == tmp_790 { keys[6] = tmp_779; values[6] = tmp_780; }
    let tmp_791 = keys[7] < tmp_781 || (keys[7] == tmp_781 && values[7] < tmp_782);
    if tmp_783 == tmp_791 { keys[7] = tmp_781; values[7] = tmp_782; }
    }
    // exch_paral(tmask:8,swbit:3,wpt:8) 
    {
    let tmp_792 = subgroupShuffleXor(keys[0], 8u);
    let tmp_793 = subgroupShuffleXor(values[0], 8u);
    let tmp_794 = subgroupShuffleXor(keys[1], 8u);
    let tmp_795 = subgroupShuffleXor(values[1], 8u);
    let tmp_796 = subgroupShuffleXor(keys[2], 8u);
    let tmp_797 = subgroupShuffleXor(values[2], 8u);
    let tmp_798 = subgroupShuffleXor(keys[3], 8u);
    let tmp_799 = subgroupShuffleXor(values[3], 8u);
    let tmp_800 = subgroupShuffleXor(keys[4], 8u);
    let tmp_801 = subgroupShuffleXor(values[4], 8u);
    let tmp_802 = subgroupShuffleXor(keys[5], 8u);
    let tmp_803 = subgroupShuffleXor(values[5], 8u);
    let tmp_804 = subgroupShuffleXor(keys[6], 8u);
    let tmp_805 = subgroupShuffleXor(values[6], 8u);
    let tmp_806 = subgroupShuffleXor(keys[7], 8u);
    let tmp_807 = subgroupShuffleXor(values[7], 8u);
    let tmp_808 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_809 = keys[0] < tmp_792 || (keys[0] == tmp_792 && values[0] < tmp_793);
    if tmp_808 == tmp_809 { keys[0] = tmp_792; values[0] = tmp_793; }
    let tmp_810 = keys[1] < tmp_794 || (keys[1] == tmp_794 && values[1] < tmp_795);
    if tmp_808 == tmp_810 { keys[1] = tmp_794; values[1] = tmp_795; }
    let tmp_811 = keys[2] < tmp_796 || (keys[2] == tmp_796 && values[2] < tmp_797);
    if tmp_808 == tmp_811 { keys[2] = tmp_796; values[2] = tmp_797; }
    let tmp_812 = keys[3] < tmp_798 || (keys[3] == tmp_798 && values[3] < tmp_799);
    if tmp_808 == tmp_812 { keys[3] = tmp_798; values[3] = tmp_799; }
    let tmp_813 = keys[4] < tmp_800 || (keys[4] == tmp_800 && values[4] < tmp_801);
    if tmp_808 == tmp_813 { keys[4] = tmp_800; values[4] = tmp_801; }
    let tmp_814 = keys[5] < tmp_802 || (keys[5] == tmp_802 && values[5] < tmp_803);
    if tmp_808 == tmp_814 { keys[5] = tmp_802; values[5] = tmp_803; }
    let tmp_815 = keys[6] < tmp_804 || (keys[6] == tmp_804 && values[6] < tmp_805);
    if tmp_808 == tmp_815 { keys[6] = tmp_804; values[6] = tmp_805; }
    let tmp_816 = keys[7] < tmp_806 || (keys[7] == tmp_806 && values[7] < tmp_807);
    if tmp_808 == tmp_816 { keys[7] = tmp_806; values[7] = tmp_807; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:8) 
    {
    let tmp_817 = subgroupShuffleXor(keys[0], 4u);
    let tmp_818 = subgroupShuffleXor(values[0], 4u);
    let tmp_819 = subgroupShuffleXor(keys[1], 4u);
    let tmp_820 = subgroupShuffleXor(values[1], 4u);
    let tmp_821 = subgroupShuffleXor(keys[2], 4u);
    let tmp_822 = subgroupShuffleXor(values[2], 4u);
    let tmp_823 = subgroupShuffleXor(keys[3], 4u);
    let tmp_824 = subgroupShuffleXor(values[3], 4u);
    let tmp_825 = subgroupShuffleXor(keys[4], 4u);
    let tmp_826 = subgroupShuffleXor(values[4], 4u);
    let tmp_827 = subgroupShuffleXor(keys[5], 4u);
    let tmp_828 = subgroupShuffleXor(values[5], 4u);
    let tmp_829 = subgroupShuffleXor(keys[6], 4u);
    let tmp_830 = subgroupShuffleXor(values[6], 4u);
    let tmp_831 = subgroupShuffleXor(keys[7], 4u);
    let tmp_832 = subgroupShuffleXor(values[7], 4u);
    let tmp_833 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_834 = keys[0] < tmp_817 || (keys[0] == tmp_817 && values[0] < tmp_818);
    if tmp_833 == tmp_834 { keys[0] = tmp_817; values[0] = tmp_818; }
    let tmp_835 = keys[1] < tmp_819 || (keys[1] == tmp_819 && values[1] < tmp_820);
    if tmp_833 == tmp_835 { keys[1] = tmp_819; values[1] = tmp_820; }
    let tmp_836 = keys[2] < tmp_821 || (keys[2] == tmp_821 && values[2] < tmp_822);
    if tmp_833 == tmp_836 { keys[2] = tmp_821; values[2] = tmp_822; }
    let tmp_837 = keys[3] < tmp_823 || (keys[3] == tmp_823 && values[3] < tmp_824);
    if tmp_833 == tmp_837 { keys[3] = tmp_823; values[3] = tmp_824; }
    let tmp_838 = keys[4] < tmp_825 || (keys[4] == tmp_825 && values[4] < tmp_826);
    if tmp_833 == tmp_838 { keys[4] = tmp_825; values[4] = tmp_826; }
    let tmp_839 = keys[5] < tmp_827 || (keys[5] == tmp_827 && values[5] < tmp_828);
    if tmp_833 == tmp_839 { keys[5] = tmp_827; values[5] = tmp_828; }
    let tmp_840 = keys[6] < tmp_829 || (keys[6] == tmp_829 && values[6] < tmp_830);
    if tmp_833 == tmp_840 { keys[6] = tmp_829; values[6] = tmp_830; }
    let tmp_841 = keys[7] < tmp_831 || (keys[7] == tmp_831 && values[7] < tmp_832);
    if tmp_833 == tmp_841 { keys[7] = tmp_831; values[7] = tmp_832; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    {
    let tmp_842 = subgroupShuffleXor(keys[0], 2u);
    let tmp_843 = subgroupShuffleXor(values[0], 2u);
    let tmp_844 = subgroupShuffleXor(keys[1], 2u);
    let tmp_845 = subgroupShuffleXor(values[1], 2u);
    let tmp_846 = subgroupShuffleXor(keys[2], 2u);
    let tmp_847 = subgroupShuffleXor(values[2], 2u);
    let tmp_848 = subgroupShuffleXor(keys[3], 2u);
    let tmp_849 = subgroupShuffleXor(values[3], 2u);
    let tmp_850 = subgroupShuffleXor(keys[4], 2u);
    let tmp_851 = subgroupShuffleXor(values[4], 2u);
    let tmp_852 = subgroupShuffleXor(keys[5], 2u);
    let tmp_853 = subgroupShuffleXor(values[5], 2u);
    let tmp_854 = subgroupShuffleXor(keys[6], 2u);
    let tmp_855 = subgroupShuffleXor(values[6], 2u);
    let tmp_856 = subgroupShuffleXor(keys[7], 2u);
    let tmp_857 = subgroupShuffleXor(values[7], 2u);
    let tmp_858 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_859 = keys[0] < tmp_842 || (keys[0] == tmp_842 && values[0] < tmp_843);
    if tmp_858 == tmp_859 { keys[0] = tmp_842; values[0] = tmp_843; }
    let tmp_860 = keys[1] < tmp_844 || (keys[1] == tmp_844 && values[1] < tmp_845);
    if tmp_858 == tmp_860 { keys[1] = tmp_844; values[1] = tmp_845; }
    let tmp_861 = keys[2] < tmp_846 || (keys[2] == tmp_846 && values[2] < tmp_847);
    if tmp_858 == tmp_861 { keys[2] = tmp_846; values[2] = tmp_847; }
    let tmp_862 = keys[3] < tmp_848 || (keys[3] == tmp_848 && values[3] < tmp_849);
    if tmp_858 == tmp_862 { keys[3] = tmp_848; values[3] = tmp_849; }
    let tmp_863 = keys[4] < tmp_850 || (keys[4] == tmp_850 && values[4] < tmp_851);
    if tmp_858 == tmp_863 { keys[4] = tmp_850; values[4] = tmp_851; }
    let tmp_864 = keys[5] < tmp_852 || (keys[5] == tmp_852 && values[5] < tmp_853);
    if tmp_858 == tmp_864 { keys[5] = tmp_852; values[5] = tmp_853; }
    let tmp_865 = keys[6] < tmp_854 || (keys[6] == tmp_854 && values[6] < tmp_855);
    if tmp_858 == tmp_865 { keys[6] = tmp_854; values[6] = tmp_855; }
    let tmp_866 = keys[7] < tmp_856 || (keys[7] == tmp_856 && values[7] < tmp_857);
    if tmp_858 == tmp_866 { keys[7] = tmp_856; values[7] = tmp_857; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    {
    let tmp_867 = subgroupShuffleXor(keys[0], 1u);
    let tmp_868 = subgroupShuffleXor(values[0], 1u);
    let tmp_869 = subgroupShuffleXor(keys[1], 1u);
    let tmp_870 = subgroupShuffleXor(values[1], 1u);
    let tmp_871 = subgroupShuffleXor(keys[2], 1u);
    let tmp_872 = subgroupShuffleXor(values[2], 1u);
    let tmp_873 = subgroupShuffleXor(keys[3], 1u);
    let tmp_874 = subgroupShuffleXor(values[3], 1u);
    let tmp_875 = subgroupShuffleXor(keys[4], 1u);
    let tmp_876 = subgroupShuffleXor(values[4], 1u);
    let tmp_877 = subgroupShuffleXor(keys[5], 1u);
    let tmp_878 = subgroupShuffleXor(values[5], 1u);
    let tmp_879 = subgroupShuffleXor(keys[6], 1u);
    let tmp_880 = subgroupShuffleXor(values[6], 1u);
    let tmp_881 = subgroupShuffleXor(keys[7], 1u);
    let tmp_882 = subgroupShuffleXor(values[7], 1u);
    let tmp_883 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_884 = keys[0] < tmp_867 || (keys[0] == tmp_867 && values[0] < tmp_868);
    if tmp_883 == tmp_884 { keys[0] = tmp_867; values[0] = tmp_868; }
    let tmp_885 = keys[1] < tmp_869 || (keys[1] == tmp_869 && values[1] < tmp_870);
    if tmp_883 == tmp_885 { keys[1] = tmp_869; values[1] = tmp_870; }
    let tmp_886 = keys[2] < tmp_871 || (keys[2] == tmp_871 && values[2] < tmp_872);
    if tmp_883 == tmp_886 { keys[2] = tmp_871; values[2] = tmp_872; }
    let tmp_887 = keys[3] < tmp_873 || (keys[3] == tmp_873 && values[3] < tmp_874);
    if tmp_883 == tmp_887 { keys[3] = tmp_873; values[3] = tmp_874; }
    let tmp_888 = keys[4] < tmp_875 || (keys[4] == tmp_875 && values[4] < tmp_876);
    if tmp_883 == tmp_888 { keys[4] = tmp_875; values[4] = tmp_876; }
    let tmp_889 = keys[5] < tmp_877 || (keys[5] == tmp_877 && values[5] < tmp_878);
    if tmp_883 == tmp_889 { keys[5] = tmp_877; values[5] = tmp_878; }
    let tmp_890 = keys[6] < tmp_879 || (keys[6] == tmp_879 && values[6] < tmp_880);
    if tmp_883 == tmp_890 { keys[6] = tmp_879; values[6] = tmp_880; }
    let tmp_891 = keys[7] < tmp_881 || (keys[7] == tmp_881 && values[7] < tmp_882);
    if tmp_883 == tmp_891 { keys[7] = tmp_881; values[7] = tmp_882; }
    }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_892 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_892;let tmp_893 = values[0]; values[0] = values[4]; values[4] = tmp_893; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_894 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_894;let tmp_895 = values[1]; values[1] = values[5]; values[5] = tmp_895; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_896 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_896;let tmp_897 = values[2]; values[2] = values[6]; values[6] = tmp_897; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_898 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_898;let tmp_899 = values[3]; values[3] = values[7]; values[7] = tmp_899; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_900 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_900;let tmp_901 = values[0]; values[0] = values[2]; values[2] = tmp_901; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_902 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_902;let tmp_903 = values[1]; values[1] = values[3]; values[3] = tmp_903; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_904 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_904;let tmp_905 = values[4]; values[4] = values[6]; values[6] = tmp_905; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_906 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_906;let tmp_907 = values[5]; values[5] = values[7]; values[7] = tmp_907; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_908 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_908;let tmp_909 = values[0]; values[0] = values[1]; values[1] = tmp_909; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_910 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_910;let tmp_911 = values[2]; values[2] = values[3]; values[3] = tmp_911; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_912 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_912;let tmp_913 = values[4]; values[4] = values[5]; values[5] = tmp_913; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_914 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_914;let tmp_915 = values[6]; values[6] = values[7]; values[7] = tmp_915; }
    }

    // blocked store
    for (var r = 0u; r < WPT; r = r + 1u) {
        let pos = local_tid * WPT + r;
        if active && pos < seg_size {
            global_keys[seg_start + pos] = keys[r];
            global_value_indices[seg_start + pos] = values[r];
        }
    }
}
