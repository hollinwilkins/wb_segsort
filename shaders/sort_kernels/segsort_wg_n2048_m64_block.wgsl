
override WG: u32 = 64u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 2048u;
const M: u32 = 64u;
const WPT: u32 = 32u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n2048_m64_block(
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

    var keys: array<u32, 32>;
    var values: array<u32, 32>;

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

    // exch_local(1,32) 
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
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_8 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_8;let tmp_9 = values[8]; values[8] = values[9]; values[9] = tmp_9; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_10 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_10;let tmp_11 = values[10]; values[10] = values[11]; values[11] = tmp_11; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_12 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_12;let tmp_13 = values[12]; values[12] = values[13]; values[13] = tmp_13; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_14 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_14;let tmp_15 = values[14]; values[14] = values[15]; values[15] = tmp_15; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_16 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_16;let tmp_17 = values[16]; values[16] = values[17]; values[17] = tmp_17; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_18 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_18;let tmp_19 = values[18]; values[18] = values[19]; values[19] = tmp_19; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_20 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_20;let tmp_21 = values[20]; values[20] = values[21]; values[21] = tmp_21; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_22 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_22;let tmp_23 = values[22]; values[22] = values[23]; values[23] = tmp_23; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_24 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_24;let tmp_25 = values[24]; values[24] = values[25]; values[25] = tmp_25; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_26 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_26;let tmp_27 = values[26]; values[26] = values[27]; values[27] = tmp_27; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_28 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_28;let tmp_29 = values[28]; values[28] = values[29]; values[29] = tmp_29; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_30 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_30;let tmp_31 = values[30]; values[30] = values[31]; values[31] = tmp_31; }
    }
    // exch_local(3,32) 
    // cmp_swap(0,3)
    if keys[0] > keys[3] || (keys[0] == keys[3] && values[0] > values[3]) {
    // swap(0,3) 
    { let tmp_32 = keys[0]; keys[0] = keys[3]; keys[3] = tmp_32;let tmp_33 = values[0]; values[0] = values[3]; values[3] = tmp_33; }
    }
    // cmp_swap(1,2)
    if keys[1] > keys[2] || (keys[1] == keys[2] && values[1] > values[2]) {
    // swap(1,2) 
    { let tmp_34 = keys[1]; keys[1] = keys[2]; keys[2] = tmp_34;let tmp_35 = values[1]; values[1] = values[2]; values[2] = tmp_35; }
    }
    // cmp_swap(4,7)
    if keys[4] > keys[7] || (keys[4] == keys[7] && values[4] > values[7]) {
    // swap(4,7) 
    { let tmp_36 = keys[4]; keys[4] = keys[7]; keys[7] = tmp_36;let tmp_37 = values[4]; values[4] = values[7]; values[7] = tmp_37; }
    }
    // cmp_swap(5,6)
    if keys[5] > keys[6] || (keys[5] == keys[6] && values[5] > values[6]) {
    // swap(5,6) 
    { let tmp_38 = keys[5]; keys[5] = keys[6]; keys[6] = tmp_38;let tmp_39 = values[5]; values[5] = values[6]; values[6] = tmp_39; }
    }
    // cmp_swap(8,11)
    if keys[8] > keys[11] || (keys[8] == keys[11] && values[8] > values[11]) {
    // swap(8,11) 
    { let tmp_40 = keys[8]; keys[8] = keys[11]; keys[11] = tmp_40;let tmp_41 = values[8]; values[8] = values[11]; values[11] = tmp_41; }
    }
    // cmp_swap(9,10)
    if keys[9] > keys[10] || (keys[9] == keys[10] && values[9] > values[10]) {
    // swap(9,10) 
    { let tmp_42 = keys[9]; keys[9] = keys[10]; keys[10] = tmp_42;let tmp_43 = values[9]; values[9] = values[10]; values[10] = tmp_43; }
    }
    // cmp_swap(12,15)
    if keys[12] > keys[15] || (keys[12] == keys[15] && values[12] > values[15]) {
    // swap(12,15) 
    { let tmp_44 = keys[12]; keys[12] = keys[15]; keys[15] = tmp_44;let tmp_45 = values[12]; values[12] = values[15]; values[15] = tmp_45; }
    }
    // cmp_swap(13,14)
    if keys[13] > keys[14] || (keys[13] == keys[14] && values[13] > values[14]) {
    // swap(13,14) 
    { let tmp_46 = keys[13]; keys[13] = keys[14]; keys[14] = tmp_46;let tmp_47 = values[13]; values[13] = values[14]; values[14] = tmp_47; }
    }
    // cmp_swap(16,19)
    if keys[16] > keys[19] || (keys[16] == keys[19] && values[16] > values[19]) {
    // swap(16,19) 
    { let tmp_48 = keys[16]; keys[16] = keys[19]; keys[19] = tmp_48;let tmp_49 = values[16]; values[16] = values[19]; values[19] = tmp_49; }
    }
    // cmp_swap(17,18)
    if keys[17] > keys[18] || (keys[17] == keys[18] && values[17] > values[18]) {
    // swap(17,18) 
    { let tmp_50 = keys[17]; keys[17] = keys[18]; keys[18] = tmp_50;let tmp_51 = values[17]; values[17] = values[18]; values[18] = tmp_51; }
    }
    // cmp_swap(20,23)
    if keys[20] > keys[23] || (keys[20] == keys[23] && values[20] > values[23]) {
    // swap(20,23) 
    { let tmp_52 = keys[20]; keys[20] = keys[23]; keys[23] = tmp_52;let tmp_53 = values[20]; values[20] = values[23]; values[23] = tmp_53; }
    }
    // cmp_swap(21,22)
    if keys[21] > keys[22] || (keys[21] == keys[22] && values[21] > values[22]) {
    // swap(21,22) 
    { let tmp_54 = keys[21]; keys[21] = keys[22]; keys[22] = tmp_54;let tmp_55 = values[21]; values[21] = values[22]; values[22] = tmp_55; }
    }
    // cmp_swap(24,27)
    if keys[24] > keys[27] || (keys[24] == keys[27] && values[24] > values[27]) {
    // swap(24,27) 
    { let tmp_56 = keys[24]; keys[24] = keys[27]; keys[27] = tmp_56;let tmp_57 = values[24]; values[24] = values[27]; values[27] = tmp_57; }
    }
    // cmp_swap(25,26)
    if keys[25] > keys[26] || (keys[25] == keys[26] && values[25] > values[26]) {
    // swap(25,26) 
    { let tmp_58 = keys[25]; keys[25] = keys[26]; keys[26] = tmp_58;let tmp_59 = values[25]; values[25] = values[26]; values[26] = tmp_59; }
    }
    // cmp_swap(28,31)
    if keys[28] > keys[31] || (keys[28] == keys[31] && values[28] > values[31]) {
    // swap(28,31) 
    { let tmp_60 = keys[28]; keys[28] = keys[31]; keys[31] = tmp_60;let tmp_61 = values[28]; values[28] = values[31]; values[31] = tmp_61; }
    }
    // cmp_swap(29,30)
    if keys[29] > keys[30] || (keys[29] == keys[30] && values[29] > values[30]) {
    // swap(29,30) 
    { let tmp_62 = keys[29]; keys[29] = keys[30]; keys[30] = tmp_62;let tmp_63 = values[29]; values[29] = values[30]; values[30] = tmp_63; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_64 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_64;let tmp_65 = values[0]; values[0] = values[1]; values[1] = tmp_65; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_66 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_66;let tmp_67 = values[2]; values[2] = values[3]; values[3] = tmp_67; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_68 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_68;let tmp_69 = values[4]; values[4] = values[5]; values[5] = tmp_69; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_70 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_70;let tmp_71 = values[6]; values[6] = values[7]; values[7] = tmp_71; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_72 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_72;let tmp_73 = values[8]; values[8] = values[9]; values[9] = tmp_73; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_74 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_74;let tmp_75 = values[10]; values[10] = values[11]; values[11] = tmp_75; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_76 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_76;let tmp_77 = values[12]; values[12] = values[13]; values[13] = tmp_77; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_78 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_78;let tmp_79 = values[14]; values[14] = values[15]; values[15] = tmp_79; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_80 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_80;let tmp_81 = values[16]; values[16] = values[17]; values[17] = tmp_81; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_82 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_82;let tmp_83 = values[18]; values[18] = values[19]; values[19] = tmp_83; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_84 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_84;let tmp_85 = values[20]; values[20] = values[21]; values[21] = tmp_85; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_86 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_86;let tmp_87 = values[22]; values[22] = values[23]; values[23] = tmp_87; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_88 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_88;let tmp_89 = values[24]; values[24] = values[25]; values[25] = tmp_89; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_90 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_90;let tmp_91 = values[26]; values[26] = values[27]; values[27] = tmp_91; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_92 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_92;let tmp_93 = values[28]; values[28] = values[29]; values[29] = tmp_93; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_94 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_94;let tmp_95 = values[30]; values[30] = values[31]; values[31] = tmp_95; }
    }
    // exch_local(7,32) 
    // cmp_swap(0,7)
    if keys[0] > keys[7] || (keys[0] == keys[7] && values[0] > values[7]) {
    // swap(0,7) 
    { let tmp_96 = keys[0]; keys[0] = keys[7]; keys[7] = tmp_96;let tmp_97 = values[0]; values[0] = values[7]; values[7] = tmp_97; }
    }
    // cmp_swap(1,6)
    if keys[1] > keys[6] || (keys[1] == keys[6] && values[1] > values[6]) {
    // swap(1,6) 
    { let tmp_98 = keys[1]; keys[1] = keys[6]; keys[6] = tmp_98;let tmp_99 = values[1]; values[1] = values[6]; values[6] = tmp_99; }
    }
    // cmp_swap(2,5)
    if keys[2] > keys[5] || (keys[2] == keys[5] && values[2] > values[5]) {
    // swap(2,5) 
    { let tmp_100 = keys[2]; keys[2] = keys[5]; keys[5] = tmp_100;let tmp_101 = values[2]; values[2] = values[5]; values[5] = tmp_101; }
    }
    // cmp_swap(3,4)
    if keys[3] > keys[4] || (keys[3] == keys[4] && values[3] > values[4]) {
    // swap(3,4) 
    { let tmp_102 = keys[3]; keys[3] = keys[4]; keys[4] = tmp_102;let tmp_103 = values[3]; values[3] = values[4]; values[4] = tmp_103; }
    }
    // cmp_swap(8,15)
    if keys[8] > keys[15] || (keys[8] == keys[15] && values[8] > values[15]) {
    // swap(8,15) 
    { let tmp_104 = keys[8]; keys[8] = keys[15]; keys[15] = tmp_104;let tmp_105 = values[8]; values[8] = values[15]; values[15] = tmp_105; }
    }
    // cmp_swap(9,14)
    if keys[9] > keys[14] || (keys[9] == keys[14] && values[9] > values[14]) {
    // swap(9,14) 
    { let tmp_106 = keys[9]; keys[9] = keys[14]; keys[14] = tmp_106;let tmp_107 = values[9]; values[9] = values[14]; values[14] = tmp_107; }
    }
    // cmp_swap(10,13)
    if keys[10] > keys[13] || (keys[10] == keys[13] && values[10] > values[13]) {
    // swap(10,13) 
    { let tmp_108 = keys[10]; keys[10] = keys[13]; keys[13] = tmp_108;let tmp_109 = values[10]; values[10] = values[13]; values[13] = tmp_109; }
    }
    // cmp_swap(11,12)
    if keys[11] > keys[12] || (keys[11] == keys[12] && values[11] > values[12]) {
    // swap(11,12) 
    { let tmp_110 = keys[11]; keys[11] = keys[12]; keys[12] = tmp_110;let tmp_111 = values[11]; values[11] = values[12]; values[12] = tmp_111; }
    }
    // cmp_swap(16,23)
    if keys[16] > keys[23] || (keys[16] == keys[23] && values[16] > values[23]) {
    // swap(16,23) 
    { let tmp_112 = keys[16]; keys[16] = keys[23]; keys[23] = tmp_112;let tmp_113 = values[16]; values[16] = values[23]; values[23] = tmp_113; }
    }
    // cmp_swap(17,22)
    if keys[17] > keys[22] || (keys[17] == keys[22] && values[17] > values[22]) {
    // swap(17,22) 
    { let tmp_114 = keys[17]; keys[17] = keys[22]; keys[22] = tmp_114;let tmp_115 = values[17]; values[17] = values[22]; values[22] = tmp_115; }
    }
    // cmp_swap(18,21)
    if keys[18] > keys[21] || (keys[18] == keys[21] && values[18] > values[21]) {
    // swap(18,21) 
    { let tmp_116 = keys[18]; keys[18] = keys[21]; keys[21] = tmp_116;let tmp_117 = values[18]; values[18] = values[21]; values[21] = tmp_117; }
    }
    // cmp_swap(19,20)
    if keys[19] > keys[20] || (keys[19] == keys[20] && values[19] > values[20]) {
    // swap(19,20) 
    { let tmp_118 = keys[19]; keys[19] = keys[20]; keys[20] = tmp_118;let tmp_119 = values[19]; values[19] = values[20]; values[20] = tmp_119; }
    }
    // cmp_swap(24,31)
    if keys[24] > keys[31] || (keys[24] == keys[31] && values[24] > values[31]) {
    // swap(24,31) 
    { let tmp_120 = keys[24]; keys[24] = keys[31]; keys[31] = tmp_120;let tmp_121 = values[24]; values[24] = values[31]; values[31] = tmp_121; }
    }
    // cmp_swap(25,30)
    if keys[25] > keys[30] || (keys[25] == keys[30] && values[25] > values[30]) {
    // swap(25,30) 
    { let tmp_122 = keys[25]; keys[25] = keys[30]; keys[30] = tmp_122;let tmp_123 = values[25]; values[25] = values[30]; values[30] = tmp_123; }
    }
    // cmp_swap(26,29)
    if keys[26] > keys[29] || (keys[26] == keys[29] && values[26] > values[29]) {
    // swap(26,29) 
    { let tmp_124 = keys[26]; keys[26] = keys[29]; keys[29] = tmp_124;let tmp_125 = values[26]; values[26] = values[29]; values[29] = tmp_125; }
    }
    // cmp_swap(27,28)
    if keys[27] > keys[28] || (keys[27] == keys[28] && values[27] > values[28]) {
    // swap(27,28) 
    { let tmp_126 = keys[27]; keys[27] = keys[28]; keys[28] = tmp_126;let tmp_127 = values[27]; values[27] = values[28]; values[28] = tmp_127; }
    }
    // exch_local(2,32) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_128 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_128;let tmp_129 = values[0]; values[0] = values[2]; values[2] = tmp_129; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_130 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_130;let tmp_131 = values[1]; values[1] = values[3]; values[3] = tmp_131; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_132 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_132;let tmp_133 = values[4]; values[4] = values[6]; values[6] = tmp_133; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_134 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_134;let tmp_135 = values[5]; values[5] = values[7]; values[7] = tmp_135; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_136 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_136;let tmp_137 = values[8]; values[8] = values[10]; values[10] = tmp_137; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_138 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_138;let tmp_139 = values[9]; values[9] = values[11]; values[11] = tmp_139; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_140 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_140;let tmp_141 = values[12]; values[12] = values[14]; values[14] = tmp_141; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_142 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_142;let tmp_143 = values[13]; values[13] = values[15]; values[15] = tmp_143; }
    }
    // cmp_swap(16,18)
    if keys[16] > keys[18] || (keys[16] == keys[18] && values[16] > values[18]) {
    // swap(16,18) 
    { let tmp_144 = keys[16]; keys[16] = keys[18]; keys[18] = tmp_144;let tmp_145 = values[16]; values[16] = values[18]; values[18] = tmp_145; }
    }
    // cmp_swap(17,19)
    if keys[17] > keys[19] || (keys[17] == keys[19] && values[17] > values[19]) {
    // swap(17,19) 
    { let tmp_146 = keys[17]; keys[17] = keys[19]; keys[19] = tmp_146;let tmp_147 = values[17]; values[17] = values[19]; values[19] = tmp_147; }
    }
    // cmp_swap(20,22)
    if keys[20] > keys[22] || (keys[20] == keys[22] && values[20] > values[22]) {
    // swap(20,22) 
    { let tmp_148 = keys[20]; keys[20] = keys[22]; keys[22] = tmp_148;let tmp_149 = values[20]; values[20] = values[22]; values[22] = tmp_149; }
    }
    // cmp_swap(21,23)
    if keys[21] > keys[23] || (keys[21] == keys[23] && values[21] > values[23]) {
    // swap(21,23) 
    { let tmp_150 = keys[21]; keys[21] = keys[23]; keys[23] = tmp_150;let tmp_151 = values[21]; values[21] = values[23]; values[23] = tmp_151; }
    }
    // cmp_swap(24,26)
    if keys[24] > keys[26] || (keys[24] == keys[26] && values[24] > values[26]) {
    // swap(24,26) 
    { let tmp_152 = keys[24]; keys[24] = keys[26]; keys[26] = tmp_152;let tmp_153 = values[24]; values[24] = values[26]; values[26] = tmp_153; }
    }
    // cmp_swap(25,27)
    if keys[25] > keys[27] || (keys[25] == keys[27] && values[25] > values[27]) {
    // swap(25,27) 
    { let tmp_154 = keys[25]; keys[25] = keys[27]; keys[27] = tmp_154;let tmp_155 = values[25]; values[25] = values[27]; values[27] = tmp_155; }
    }
    // cmp_swap(28,30)
    if keys[28] > keys[30] || (keys[28] == keys[30] && values[28] > values[30]) {
    // swap(28,30) 
    { let tmp_156 = keys[28]; keys[28] = keys[30]; keys[30] = tmp_156;let tmp_157 = values[28]; values[28] = values[30]; values[30] = tmp_157; }
    }
    // cmp_swap(29,31)
    if keys[29] > keys[31] || (keys[29] == keys[31] && values[29] > values[31]) {
    // swap(29,31) 
    { let tmp_158 = keys[29]; keys[29] = keys[31]; keys[31] = tmp_158;let tmp_159 = values[29]; values[29] = values[31]; values[31] = tmp_159; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_160 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_160;let tmp_161 = values[0]; values[0] = values[1]; values[1] = tmp_161; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_162 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_162;let tmp_163 = values[2]; values[2] = values[3]; values[3] = tmp_163; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_164 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_164;let tmp_165 = values[4]; values[4] = values[5]; values[5] = tmp_165; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_166 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_166;let tmp_167 = values[6]; values[6] = values[7]; values[7] = tmp_167; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_168 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_168;let tmp_169 = values[8]; values[8] = values[9]; values[9] = tmp_169; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_170 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_170;let tmp_171 = values[10]; values[10] = values[11]; values[11] = tmp_171; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_172 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_172;let tmp_173 = values[12]; values[12] = values[13]; values[13] = tmp_173; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_174 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_174;let tmp_175 = values[14]; values[14] = values[15]; values[15] = tmp_175; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_176 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_176;let tmp_177 = values[16]; values[16] = values[17]; values[17] = tmp_177; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_178 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_178;let tmp_179 = values[18]; values[18] = values[19]; values[19] = tmp_179; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_180 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_180;let tmp_181 = values[20]; values[20] = values[21]; values[21] = tmp_181; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_182 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_182;let tmp_183 = values[22]; values[22] = values[23]; values[23] = tmp_183; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_184 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_184;let tmp_185 = values[24]; values[24] = values[25]; values[25] = tmp_185; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_186 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_186;let tmp_187 = values[26]; values[26] = values[27]; values[27] = tmp_187; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_188 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_188;let tmp_189 = values[28]; values[28] = values[29]; values[29] = tmp_189; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_190 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_190;let tmp_191 = values[30]; values[30] = values[31]; values[31] = tmp_191; }
    }
    // exch_local(15,32) 
    // cmp_swap(0,15)
    if keys[0] > keys[15] || (keys[0] == keys[15] && values[0] > values[15]) {
    // swap(0,15) 
    { let tmp_192 = keys[0]; keys[0] = keys[15]; keys[15] = tmp_192;let tmp_193 = values[0]; values[0] = values[15]; values[15] = tmp_193; }
    }
    // cmp_swap(1,14)
    if keys[1] > keys[14] || (keys[1] == keys[14] && values[1] > values[14]) {
    // swap(1,14) 
    { let tmp_194 = keys[1]; keys[1] = keys[14]; keys[14] = tmp_194;let tmp_195 = values[1]; values[1] = values[14]; values[14] = tmp_195; }
    }
    // cmp_swap(2,13)
    if keys[2] > keys[13] || (keys[2] == keys[13] && values[2] > values[13]) {
    // swap(2,13) 
    { let tmp_196 = keys[2]; keys[2] = keys[13]; keys[13] = tmp_196;let tmp_197 = values[2]; values[2] = values[13]; values[13] = tmp_197; }
    }
    // cmp_swap(3,12)
    if keys[3] > keys[12] || (keys[3] == keys[12] && values[3] > values[12]) {
    // swap(3,12) 
    { let tmp_198 = keys[3]; keys[3] = keys[12]; keys[12] = tmp_198;let tmp_199 = values[3]; values[3] = values[12]; values[12] = tmp_199; }
    }
    // cmp_swap(4,11)
    if keys[4] > keys[11] || (keys[4] == keys[11] && values[4] > values[11]) {
    // swap(4,11) 
    { let tmp_200 = keys[4]; keys[4] = keys[11]; keys[11] = tmp_200;let tmp_201 = values[4]; values[4] = values[11]; values[11] = tmp_201; }
    }
    // cmp_swap(5,10)
    if keys[5] > keys[10] || (keys[5] == keys[10] && values[5] > values[10]) {
    // swap(5,10) 
    { let tmp_202 = keys[5]; keys[5] = keys[10]; keys[10] = tmp_202;let tmp_203 = values[5]; values[5] = values[10]; values[10] = tmp_203; }
    }
    // cmp_swap(6,9)
    if keys[6] > keys[9] || (keys[6] == keys[9] && values[6] > values[9]) {
    // swap(6,9) 
    { let tmp_204 = keys[6]; keys[6] = keys[9]; keys[9] = tmp_204;let tmp_205 = values[6]; values[6] = values[9]; values[9] = tmp_205; }
    }
    // cmp_swap(7,8)
    if keys[7] > keys[8] || (keys[7] == keys[8] && values[7] > values[8]) {
    // swap(7,8) 
    { let tmp_206 = keys[7]; keys[7] = keys[8]; keys[8] = tmp_206;let tmp_207 = values[7]; values[7] = values[8]; values[8] = tmp_207; }
    }
    // cmp_swap(16,31)
    if keys[16] > keys[31] || (keys[16] == keys[31] && values[16] > values[31]) {
    // swap(16,31) 
    { let tmp_208 = keys[16]; keys[16] = keys[31]; keys[31] = tmp_208;let tmp_209 = values[16]; values[16] = values[31]; values[31] = tmp_209; }
    }
    // cmp_swap(17,30)
    if keys[17] > keys[30] || (keys[17] == keys[30] && values[17] > values[30]) {
    // swap(17,30) 
    { let tmp_210 = keys[17]; keys[17] = keys[30]; keys[30] = tmp_210;let tmp_211 = values[17]; values[17] = values[30]; values[30] = tmp_211; }
    }
    // cmp_swap(18,29)
    if keys[18] > keys[29] || (keys[18] == keys[29] && values[18] > values[29]) {
    // swap(18,29) 
    { let tmp_212 = keys[18]; keys[18] = keys[29]; keys[29] = tmp_212;let tmp_213 = values[18]; values[18] = values[29]; values[29] = tmp_213; }
    }
    // cmp_swap(19,28)
    if keys[19] > keys[28] || (keys[19] == keys[28] && values[19] > values[28]) {
    // swap(19,28) 
    { let tmp_214 = keys[19]; keys[19] = keys[28]; keys[28] = tmp_214;let tmp_215 = values[19]; values[19] = values[28]; values[28] = tmp_215; }
    }
    // cmp_swap(20,27)
    if keys[20] > keys[27] || (keys[20] == keys[27] && values[20] > values[27]) {
    // swap(20,27) 
    { let tmp_216 = keys[20]; keys[20] = keys[27]; keys[27] = tmp_216;let tmp_217 = values[20]; values[20] = values[27]; values[27] = tmp_217; }
    }
    // cmp_swap(21,26)
    if keys[21] > keys[26] || (keys[21] == keys[26] && values[21] > values[26]) {
    // swap(21,26) 
    { let tmp_218 = keys[21]; keys[21] = keys[26]; keys[26] = tmp_218;let tmp_219 = values[21]; values[21] = values[26]; values[26] = tmp_219; }
    }
    // cmp_swap(22,25)
    if keys[22] > keys[25] || (keys[22] == keys[25] && values[22] > values[25]) {
    // swap(22,25) 
    { let tmp_220 = keys[22]; keys[22] = keys[25]; keys[25] = tmp_220;let tmp_221 = values[22]; values[22] = values[25]; values[25] = tmp_221; }
    }
    // cmp_swap(23,24)
    if keys[23] > keys[24] || (keys[23] == keys[24] && values[23] > values[24]) {
    // swap(23,24) 
    { let tmp_222 = keys[23]; keys[23] = keys[24]; keys[24] = tmp_222;let tmp_223 = values[23]; values[23] = values[24]; values[24] = tmp_223; }
    }
    // exch_local(4,32) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_224 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_224;let tmp_225 = values[0]; values[0] = values[4]; values[4] = tmp_225; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_226 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_226;let tmp_227 = values[1]; values[1] = values[5]; values[5] = tmp_227; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_228 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_228;let tmp_229 = values[2]; values[2] = values[6]; values[6] = tmp_229; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_230 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_230;let tmp_231 = values[3]; values[3] = values[7]; values[7] = tmp_231; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_232 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_232;let tmp_233 = values[8]; values[8] = values[12]; values[12] = tmp_233; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_234 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_234;let tmp_235 = values[9]; values[9] = values[13]; values[13] = tmp_235; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_236 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_236;let tmp_237 = values[10]; values[10] = values[14]; values[14] = tmp_237; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_238 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_238;let tmp_239 = values[11]; values[11] = values[15]; values[15] = tmp_239; }
    }
    // cmp_swap(16,20)
    if keys[16] > keys[20] || (keys[16] == keys[20] && values[16] > values[20]) {
    // swap(16,20) 
    { let tmp_240 = keys[16]; keys[16] = keys[20]; keys[20] = tmp_240;let tmp_241 = values[16]; values[16] = values[20]; values[20] = tmp_241; }
    }
    // cmp_swap(17,21)
    if keys[17] > keys[21] || (keys[17] == keys[21] && values[17] > values[21]) {
    // swap(17,21) 
    { let tmp_242 = keys[17]; keys[17] = keys[21]; keys[21] = tmp_242;let tmp_243 = values[17]; values[17] = values[21]; values[21] = tmp_243; }
    }
    // cmp_swap(18,22)
    if keys[18] > keys[22] || (keys[18] == keys[22] && values[18] > values[22]) {
    // swap(18,22) 
    { let tmp_244 = keys[18]; keys[18] = keys[22]; keys[22] = tmp_244;let tmp_245 = values[18]; values[18] = values[22]; values[22] = tmp_245; }
    }
    // cmp_swap(19,23)
    if keys[19] > keys[23] || (keys[19] == keys[23] && values[19] > values[23]) {
    // swap(19,23) 
    { let tmp_246 = keys[19]; keys[19] = keys[23]; keys[23] = tmp_246;let tmp_247 = values[19]; values[19] = values[23]; values[23] = tmp_247; }
    }
    // cmp_swap(24,28)
    if keys[24] > keys[28] || (keys[24] == keys[28] && values[24] > values[28]) {
    // swap(24,28) 
    { let tmp_248 = keys[24]; keys[24] = keys[28]; keys[28] = tmp_248;let tmp_249 = values[24]; values[24] = values[28]; values[28] = tmp_249; }
    }
    // cmp_swap(25,29)
    if keys[25] > keys[29] || (keys[25] == keys[29] && values[25] > values[29]) {
    // swap(25,29) 
    { let tmp_250 = keys[25]; keys[25] = keys[29]; keys[29] = tmp_250;let tmp_251 = values[25]; values[25] = values[29]; values[29] = tmp_251; }
    }
    // cmp_swap(26,30)
    if keys[26] > keys[30] || (keys[26] == keys[30] && values[26] > values[30]) {
    // swap(26,30) 
    { let tmp_252 = keys[26]; keys[26] = keys[30]; keys[30] = tmp_252;let tmp_253 = values[26]; values[26] = values[30]; values[30] = tmp_253; }
    }
    // cmp_swap(27,31)
    if keys[27] > keys[31] || (keys[27] == keys[31] && values[27] > values[31]) {
    // swap(27,31) 
    { let tmp_254 = keys[27]; keys[27] = keys[31]; keys[31] = tmp_254;let tmp_255 = values[27]; values[27] = values[31]; values[31] = tmp_255; }
    }
    // exch_local(2,32) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_256 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_256;let tmp_257 = values[0]; values[0] = values[2]; values[2] = tmp_257; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_258 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_258;let tmp_259 = values[1]; values[1] = values[3]; values[3] = tmp_259; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_260 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_260;let tmp_261 = values[4]; values[4] = values[6]; values[6] = tmp_261; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_262 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_262;let tmp_263 = values[5]; values[5] = values[7]; values[7] = tmp_263; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_264 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_264;let tmp_265 = values[8]; values[8] = values[10]; values[10] = tmp_265; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_266 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_266;let tmp_267 = values[9]; values[9] = values[11]; values[11] = tmp_267; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_268 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_268;let tmp_269 = values[12]; values[12] = values[14]; values[14] = tmp_269; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_270 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_270;let tmp_271 = values[13]; values[13] = values[15]; values[15] = tmp_271; }
    }
    // cmp_swap(16,18)
    if keys[16] > keys[18] || (keys[16] == keys[18] && values[16] > values[18]) {
    // swap(16,18) 
    { let tmp_272 = keys[16]; keys[16] = keys[18]; keys[18] = tmp_272;let tmp_273 = values[16]; values[16] = values[18]; values[18] = tmp_273; }
    }
    // cmp_swap(17,19)
    if keys[17] > keys[19] || (keys[17] == keys[19] && values[17] > values[19]) {
    // swap(17,19) 
    { let tmp_274 = keys[17]; keys[17] = keys[19]; keys[19] = tmp_274;let tmp_275 = values[17]; values[17] = values[19]; values[19] = tmp_275; }
    }
    // cmp_swap(20,22)
    if keys[20] > keys[22] || (keys[20] == keys[22] && values[20] > values[22]) {
    // swap(20,22) 
    { let tmp_276 = keys[20]; keys[20] = keys[22]; keys[22] = tmp_276;let tmp_277 = values[20]; values[20] = values[22]; values[22] = tmp_277; }
    }
    // cmp_swap(21,23)
    if keys[21] > keys[23] || (keys[21] == keys[23] && values[21] > values[23]) {
    // swap(21,23) 
    { let tmp_278 = keys[21]; keys[21] = keys[23]; keys[23] = tmp_278;let tmp_279 = values[21]; values[21] = values[23]; values[23] = tmp_279; }
    }
    // cmp_swap(24,26)
    if keys[24] > keys[26] || (keys[24] == keys[26] && values[24] > values[26]) {
    // swap(24,26) 
    { let tmp_280 = keys[24]; keys[24] = keys[26]; keys[26] = tmp_280;let tmp_281 = values[24]; values[24] = values[26]; values[26] = tmp_281; }
    }
    // cmp_swap(25,27)
    if keys[25] > keys[27] || (keys[25] == keys[27] && values[25] > values[27]) {
    // swap(25,27) 
    { let tmp_282 = keys[25]; keys[25] = keys[27]; keys[27] = tmp_282;let tmp_283 = values[25]; values[25] = values[27]; values[27] = tmp_283; }
    }
    // cmp_swap(28,30)
    if keys[28] > keys[30] || (keys[28] == keys[30] && values[28] > values[30]) {
    // swap(28,30) 
    { let tmp_284 = keys[28]; keys[28] = keys[30]; keys[30] = tmp_284;let tmp_285 = values[28]; values[28] = values[30]; values[30] = tmp_285; }
    }
    // cmp_swap(29,31)
    if keys[29] > keys[31] || (keys[29] == keys[31] && values[29] > values[31]) {
    // swap(29,31) 
    { let tmp_286 = keys[29]; keys[29] = keys[31]; keys[31] = tmp_286;let tmp_287 = values[29]; values[29] = values[31]; values[31] = tmp_287; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_288 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_288;let tmp_289 = values[0]; values[0] = values[1]; values[1] = tmp_289; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_290 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_290;let tmp_291 = values[2]; values[2] = values[3]; values[3] = tmp_291; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_292 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_292;let tmp_293 = values[4]; values[4] = values[5]; values[5] = tmp_293; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_294 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_294;let tmp_295 = values[6]; values[6] = values[7]; values[7] = tmp_295; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_296 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_296;let tmp_297 = values[8]; values[8] = values[9]; values[9] = tmp_297; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_298 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_298;let tmp_299 = values[10]; values[10] = values[11]; values[11] = tmp_299; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_300 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_300;let tmp_301 = values[12]; values[12] = values[13]; values[13] = tmp_301; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_302 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_302;let tmp_303 = values[14]; values[14] = values[15]; values[15] = tmp_303; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_304 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_304;let tmp_305 = values[16]; values[16] = values[17]; values[17] = tmp_305; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_306 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_306;let tmp_307 = values[18]; values[18] = values[19]; values[19] = tmp_307; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_308 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_308;let tmp_309 = values[20]; values[20] = values[21]; values[21] = tmp_309; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_310 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_310;let tmp_311 = values[22]; values[22] = values[23]; values[23] = tmp_311; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_312 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_312;let tmp_313 = values[24]; values[24] = values[25]; values[25] = tmp_313; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_314 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_314;let tmp_315 = values[26]; values[26] = values[27]; values[27] = tmp_315; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_316 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_316;let tmp_317 = values[28]; values[28] = values[29]; values[29] = tmp_317; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_318 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_318;let tmp_319 = values[30]; values[30] = values[31]; values[31] = tmp_319; }
    }
    // exch_local(31,32) 
    // cmp_swap(0,31)
    if keys[0] > keys[31] || (keys[0] == keys[31] && values[0] > values[31]) {
    // swap(0,31) 
    { let tmp_320 = keys[0]; keys[0] = keys[31]; keys[31] = tmp_320;let tmp_321 = values[0]; values[0] = values[31]; values[31] = tmp_321; }
    }
    // cmp_swap(1,30)
    if keys[1] > keys[30] || (keys[1] == keys[30] && values[1] > values[30]) {
    // swap(1,30) 
    { let tmp_322 = keys[1]; keys[1] = keys[30]; keys[30] = tmp_322;let tmp_323 = values[1]; values[1] = values[30]; values[30] = tmp_323; }
    }
    // cmp_swap(2,29)
    if keys[2] > keys[29] || (keys[2] == keys[29] && values[2] > values[29]) {
    // swap(2,29) 
    { let tmp_324 = keys[2]; keys[2] = keys[29]; keys[29] = tmp_324;let tmp_325 = values[2]; values[2] = values[29]; values[29] = tmp_325; }
    }
    // cmp_swap(3,28)
    if keys[3] > keys[28] || (keys[3] == keys[28] && values[3] > values[28]) {
    // swap(3,28) 
    { let tmp_326 = keys[3]; keys[3] = keys[28]; keys[28] = tmp_326;let tmp_327 = values[3]; values[3] = values[28]; values[28] = tmp_327; }
    }
    // cmp_swap(4,27)
    if keys[4] > keys[27] || (keys[4] == keys[27] && values[4] > values[27]) {
    // swap(4,27) 
    { let tmp_328 = keys[4]; keys[4] = keys[27]; keys[27] = tmp_328;let tmp_329 = values[4]; values[4] = values[27]; values[27] = tmp_329; }
    }
    // cmp_swap(5,26)
    if keys[5] > keys[26] || (keys[5] == keys[26] && values[5] > values[26]) {
    // swap(5,26) 
    { let tmp_330 = keys[5]; keys[5] = keys[26]; keys[26] = tmp_330;let tmp_331 = values[5]; values[5] = values[26]; values[26] = tmp_331; }
    }
    // cmp_swap(6,25)
    if keys[6] > keys[25] || (keys[6] == keys[25] && values[6] > values[25]) {
    // swap(6,25) 
    { let tmp_332 = keys[6]; keys[6] = keys[25]; keys[25] = tmp_332;let tmp_333 = values[6]; values[6] = values[25]; values[25] = tmp_333; }
    }
    // cmp_swap(7,24)
    if keys[7] > keys[24] || (keys[7] == keys[24] && values[7] > values[24]) {
    // swap(7,24) 
    { let tmp_334 = keys[7]; keys[7] = keys[24]; keys[24] = tmp_334;let tmp_335 = values[7]; values[7] = values[24]; values[24] = tmp_335; }
    }
    // cmp_swap(8,23)
    if keys[8] > keys[23] || (keys[8] == keys[23] && values[8] > values[23]) {
    // swap(8,23) 
    { let tmp_336 = keys[8]; keys[8] = keys[23]; keys[23] = tmp_336;let tmp_337 = values[8]; values[8] = values[23]; values[23] = tmp_337; }
    }
    // cmp_swap(9,22)
    if keys[9] > keys[22] || (keys[9] == keys[22] && values[9] > values[22]) {
    // swap(9,22) 
    { let tmp_338 = keys[9]; keys[9] = keys[22]; keys[22] = tmp_338;let tmp_339 = values[9]; values[9] = values[22]; values[22] = tmp_339; }
    }
    // cmp_swap(10,21)
    if keys[10] > keys[21] || (keys[10] == keys[21] && values[10] > values[21]) {
    // swap(10,21) 
    { let tmp_340 = keys[10]; keys[10] = keys[21]; keys[21] = tmp_340;let tmp_341 = values[10]; values[10] = values[21]; values[21] = tmp_341; }
    }
    // cmp_swap(11,20)
    if keys[11] > keys[20] || (keys[11] == keys[20] && values[11] > values[20]) {
    // swap(11,20) 
    { let tmp_342 = keys[11]; keys[11] = keys[20]; keys[20] = tmp_342;let tmp_343 = values[11]; values[11] = values[20]; values[20] = tmp_343; }
    }
    // cmp_swap(12,19)
    if keys[12] > keys[19] || (keys[12] == keys[19] && values[12] > values[19]) {
    // swap(12,19) 
    { let tmp_344 = keys[12]; keys[12] = keys[19]; keys[19] = tmp_344;let tmp_345 = values[12]; values[12] = values[19]; values[19] = tmp_345; }
    }
    // cmp_swap(13,18)
    if keys[13] > keys[18] || (keys[13] == keys[18] && values[13] > values[18]) {
    // swap(13,18) 
    { let tmp_346 = keys[13]; keys[13] = keys[18]; keys[18] = tmp_346;let tmp_347 = values[13]; values[13] = values[18]; values[18] = tmp_347; }
    }
    // cmp_swap(14,17)
    if keys[14] > keys[17] || (keys[14] == keys[17] && values[14] > values[17]) {
    // swap(14,17) 
    { let tmp_348 = keys[14]; keys[14] = keys[17]; keys[17] = tmp_348;let tmp_349 = values[14]; values[14] = values[17]; values[17] = tmp_349; }
    }
    // cmp_swap(15,16)
    if keys[15] > keys[16] || (keys[15] == keys[16] && values[15] > values[16]) {
    // swap(15,16) 
    { let tmp_350 = keys[15]; keys[15] = keys[16]; keys[16] = tmp_350;let tmp_351 = values[15]; values[15] = values[16]; values[16] = tmp_351; }
    }
    // exch_local(8,32) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_352 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_352;let tmp_353 = values[0]; values[0] = values[8]; values[8] = tmp_353; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_354 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_354;let tmp_355 = values[1]; values[1] = values[9]; values[9] = tmp_355; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_356 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_356;let tmp_357 = values[2]; values[2] = values[10]; values[10] = tmp_357; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_358 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_358;let tmp_359 = values[3]; values[3] = values[11]; values[11] = tmp_359; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_360 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_360;let tmp_361 = values[4]; values[4] = values[12]; values[12] = tmp_361; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_362 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_362;let tmp_363 = values[5]; values[5] = values[13]; values[13] = tmp_363; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_364 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_364;let tmp_365 = values[6]; values[6] = values[14]; values[14] = tmp_365; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_366 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_366;let tmp_367 = values[7]; values[7] = values[15]; values[15] = tmp_367; }
    }
    // cmp_swap(16,24)
    if keys[16] > keys[24] || (keys[16] == keys[24] && values[16] > values[24]) {
    // swap(16,24) 
    { let tmp_368 = keys[16]; keys[16] = keys[24]; keys[24] = tmp_368;let tmp_369 = values[16]; values[16] = values[24]; values[24] = tmp_369; }
    }
    // cmp_swap(17,25)
    if keys[17] > keys[25] || (keys[17] == keys[25] && values[17] > values[25]) {
    // swap(17,25) 
    { let tmp_370 = keys[17]; keys[17] = keys[25]; keys[25] = tmp_370;let tmp_371 = values[17]; values[17] = values[25]; values[25] = tmp_371; }
    }
    // cmp_swap(18,26)
    if keys[18] > keys[26] || (keys[18] == keys[26] && values[18] > values[26]) {
    // swap(18,26) 
    { let tmp_372 = keys[18]; keys[18] = keys[26]; keys[26] = tmp_372;let tmp_373 = values[18]; values[18] = values[26]; values[26] = tmp_373; }
    }
    // cmp_swap(19,27)
    if keys[19] > keys[27] || (keys[19] == keys[27] && values[19] > values[27]) {
    // swap(19,27) 
    { let tmp_374 = keys[19]; keys[19] = keys[27]; keys[27] = tmp_374;let tmp_375 = values[19]; values[19] = values[27]; values[27] = tmp_375; }
    }
    // cmp_swap(20,28)
    if keys[20] > keys[28] || (keys[20] == keys[28] && values[20] > values[28]) {
    // swap(20,28) 
    { let tmp_376 = keys[20]; keys[20] = keys[28]; keys[28] = tmp_376;let tmp_377 = values[20]; values[20] = values[28]; values[28] = tmp_377; }
    }
    // cmp_swap(21,29)
    if keys[21] > keys[29] || (keys[21] == keys[29] && values[21] > values[29]) {
    // swap(21,29) 
    { let tmp_378 = keys[21]; keys[21] = keys[29]; keys[29] = tmp_378;let tmp_379 = values[21]; values[21] = values[29]; values[29] = tmp_379; }
    }
    // cmp_swap(22,30)
    if keys[22] > keys[30] || (keys[22] == keys[30] && values[22] > values[30]) {
    // swap(22,30) 
    { let tmp_380 = keys[22]; keys[22] = keys[30]; keys[30] = tmp_380;let tmp_381 = values[22]; values[22] = values[30]; values[30] = tmp_381; }
    }
    // cmp_swap(23,31)
    if keys[23] > keys[31] || (keys[23] == keys[31] && values[23] > values[31]) {
    // swap(23,31) 
    { let tmp_382 = keys[23]; keys[23] = keys[31]; keys[31] = tmp_382;let tmp_383 = values[23]; values[23] = values[31]; values[31] = tmp_383; }
    }
    // exch_local(4,32) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_384 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_384;let tmp_385 = values[0]; values[0] = values[4]; values[4] = tmp_385; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_386 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_386;let tmp_387 = values[1]; values[1] = values[5]; values[5] = tmp_387; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_388 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_388;let tmp_389 = values[2]; values[2] = values[6]; values[6] = tmp_389; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_390 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_390;let tmp_391 = values[3]; values[3] = values[7]; values[7] = tmp_391; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_392 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_392;let tmp_393 = values[8]; values[8] = values[12]; values[12] = tmp_393; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_394 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_394;let tmp_395 = values[9]; values[9] = values[13]; values[13] = tmp_395; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_396 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_396;let tmp_397 = values[10]; values[10] = values[14]; values[14] = tmp_397; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_398 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_398;let tmp_399 = values[11]; values[11] = values[15]; values[15] = tmp_399; }
    }
    // cmp_swap(16,20)
    if keys[16] > keys[20] || (keys[16] == keys[20] && values[16] > values[20]) {
    // swap(16,20) 
    { let tmp_400 = keys[16]; keys[16] = keys[20]; keys[20] = tmp_400;let tmp_401 = values[16]; values[16] = values[20]; values[20] = tmp_401; }
    }
    // cmp_swap(17,21)
    if keys[17] > keys[21] || (keys[17] == keys[21] && values[17] > values[21]) {
    // swap(17,21) 
    { let tmp_402 = keys[17]; keys[17] = keys[21]; keys[21] = tmp_402;let tmp_403 = values[17]; values[17] = values[21]; values[21] = tmp_403; }
    }
    // cmp_swap(18,22)
    if keys[18] > keys[22] || (keys[18] == keys[22] && values[18] > values[22]) {
    // swap(18,22) 
    { let tmp_404 = keys[18]; keys[18] = keys[22]; keys[22] = tmp_404;let tmp_405 = values[18]; values[18] = values[22]; values[22] = tmp_405; }
    }
    // cmp_swap(19,23)
    if keys[19] > keys[23] || (keys[19] == keys[23] && values[19] > values[23]) {
    // swap(19,23) 
    { let tmp_406 = keys[19]; keys[19] = keys[23]; keys[23] = tmp_406;let tmp_407 = values[19]; values[19] = values[23]; values[23] = tmp_407; }
    }
    // cmp_swap(24,28)
    if keys[24] > keys[28] || (keys[24] == keys[28] && values[24] > values[28]) {
    // swap(24,28) 
    { let tmp_408 = keys[24]; keys[24] = keys[28]; keys[28] = tmp_408;let tmp_409 = values[24]; values[24] = values[28]; values[28] = tmp_409; }
    }
    // cmp_swap(25,29)
    if keys[25] > keys[29] || (keys[25] == keys[29] && values[25] > values[29]) {
    // swap(25,29) 
    { let tmp_410 = keys[25]; keys[25] = keys[29]; keys[29] = tmp_410;let tmp_411 = values[25]; values[25] = values[29]; values[29] = tmp_411; }
    }
    // cmp_swap(26,30)
    if keys[26] > keys[30] || (keys[26] == keys[30] && values[26] > values[30]) {
    // swap(26,30) 
    { let tmp_412 = keys[26]; keys[26] = keys[30]; keys[30] = tmp_412;let tmp_413 = values[26]; values[26] = values[30]; values[30] = tmp_413; }
    }
    // cmp_swap(27,31)
    if keys[27] > keys[31] || (keys[27] == keys[31] && values[27] > values[31]) {
    // swap(27,31) 
    { let tmp_414 = keys[27]; keys[27] = keys[31]; keys[31] = tmp_414;let tmp_415 = values[27]; values[27] = values[31]; values[31] = tmp_415; }
    }
    // exch_local(2,32) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_416 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_416;let tmp_417 = values[0]; values[0] = values[2]; values[2] = tmp_417; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_418 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_418;let tmp_419 = values[1]; values[1] = values[3]; values[3] = tmp_419; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_420 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_420;let tmp_421 = values[4]; values[4] = values[6]; values[6] = tmp_421; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_422 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_422;let tmp_423 = values[5]; values[5] = values[7]; values[7] = tmp_423; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_424 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_424;let tmp_425 = values[8]; values[8] = values[10]; values[10] = tmp_425; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_426 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_426;let tmp_427 = values[9]; values[9] = values[11]; values[11] = tmp_427; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_428 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_428;let tmp_429 = values[12]; values[12] = values[14]; values[14] = tmp_429; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_430 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_430;let tmp_431 = values[13]; values[13] = values[15]; values[15] = tmp_431; }
    }
    // cmp_swap(16,18)
    if keys[16] > keys[18] || (keys[16] == keys[18] && values[16] > values[18]) {
    // swap(16,18) 
    { let tmp_432 = keys[16]; keys[16] = keys[18]; keys[18] = tmp_432;let tmp_433 = values[16]; values[16] = values[18]; values[18] = tmp_433; }
    }
    // cmp_swap(17,19)
    if keys[17] > keys[19] || (keys[17] == keys[19] && values[17] > values[19]) {
    // swap(17,19) 
    { let tmp_434 = keys[17]; keys[17] = keys[19]; keys[19] = tmp_434;let tmp_435 = values[17]; values[17] = values[19]; values[19] = tmp_435; }
    }
    // cmp_swap(20,22)
    if keys[20] > keys[22] || (keys[20] == keys[22] && values[20] > values[22]) {
    // swap(20,22) 
    { let tmp_436 = keys[20]; keys[20] = keys[22]; keys[22] = tmp_436;let tmp_437 = values[20]; values[20] = values[22]; values[22] = tmp_437; }
    }
    // cmp_swap(21,23)
    if keys[21] > keys[23] || (keys[21] == keys[23] && values[21] > values[23]) {
    // swap(21,23) 
    { let tmp_438 = keys[21]; keys[21] = keys[23]; keys[23] = tmp_438;let tmp_439 = values[21]; values[21] = values[23]; values[23] = tmp_439; }
    }
    // cmp_swap(24,26)
    if keys[24] > keys[26] || (keys[24] == keys[26] && values[24] > values[26]) {
    // swap(24,26) 
    { let tmp_440 = keys[24]; keys[24] = keys[26]; keys[26] = tmp_440;let tmp_441 = values[24]; values[24] = values[26]; values[26] = tmp_441; }
    }
    // cmp_swap(25,27)
    if keys[25] > keys[27] || (keys[25] == keys[27] && values[25] > values[27]) {
    // swap(25,27) 
    { let tmp_442 = keys[25]; keys[25] = keys[27]; keys[27] = tmp_442;let tmp_443 = values[25]; values[25] = values[27]; values[27] = tmp_443; }
    }
    // cmp_swap(28,30)
    if keys[28] > keys[30] || (keys[28] == keys[30] && values[28] > values[30]) {
    // swap(28,30) 
    { let tmp_444 = keys[28]; keys[28] = keys[30]; keys[30] = tmp_444;let tmp_445 = values[28]; values[28] = values[30]; values[30] = tmp_445; }
    }
    // cmp_swap(29,31)
    if keys[29] > keys[31] || (keys[29] == keys[31] && values[29] > values[31]) {
    // swap(29,31) 
    { let tmp_446 = keys[29]; keys[29] = keys[31]; keys[31] = tmp_446;let tmp_447 = values[29]; values[29] = values[31]; values[31] = tmp_447; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_448 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_448;let tmp_449 = values[0]; values[0] = values[1]; values[1] = tmp_449; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_450 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_450;let tmp_451 = values[2]; values[2] = values[3]; values[3] = tmp_451; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_452 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_452;let tmp_453 = values[4]; values[4] = values[5]; values[5] = tmp_453; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_454 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_454;let tmp_455 = values[6]; values[6] = values[7]; values[7] = tmp_455; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_456 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_456;let tmp_457 = values[8]; values[8] = values[9]; values[9] = tmp_457; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_458 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_458;let tmp_459 = values[10]; values[10] = values[11]; values[11] = tmp_459; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_460 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_460;let tmp_461 = values[12]; values[12] = values[13]; values[13] = tmp_461; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_462 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_462;let tmp_463 = values[14]; values[14] = values[15]; values[15] = tmp_463; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_464 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_464;let tmp_465 = values[16]; values[16] = values[17]; values[17] = tmp_465; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_466 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_466;let tmp_467 = values[18]; values[18] = values[19]; values[19] = tmp_467; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_468 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_468;let tmp_469 = values[20]; values[20] = values[21]; values[21] = tmp_469; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_470 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_470;let tmp_471 = values[22]; values[22] = values[23]; values[23] = tmp_471; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_472 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_472;let tmp_473 = values[24]; values[24] = values[25]; values[25] = tmp_473; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_474 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_474;let tmp_475 = values[26]; values[26] = values[27]; values[27] = tmp_475; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_476 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_476;let tmp_477 = values[28]; values[28] = values[29]; values[29] = tmp_477; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_478 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_478;let tmp_479 = values[30]; values[30] = values[31]; values[31] = tmp_479; }
    }
    // exch_intxn(tmask:1,swbit:0,wpt:32)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_480 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_481 = seg_base + (local_tid ^ 1u); let tmp_482 = smem_keys[tmp_481 * WPT + 31u]; let tmp_483 = smem_vals[tmp_481 * WPT + 31u]; let tmp_484 = keys[0] < tmp_482 || (keys[0] == tmp_482 && values[0] < tmp_483); if tmp_480 == tmp_484 { keys[0] = tmp_482; values[0] = tmp_483; } let tmp_485 = smem_keys[tmp_481 * WPT + 30u]; let tmp_486 = smem_vals[tmp_481 * WPT + 30u]; let tmp_487 = keys[1] < tmp_485 || (keys[1] == tmp_485 && values[1] < tmp_486); if tmp_480 == tmp_487 { keys[1] = tmp_485; values[1] = tmp_486; } let tmp_488 = smem_keys[tmp_481 * WPT + 29u]; let tmp_489 = smem_vals[tmp_481 * WPT + 29u]; let tmp_490 = keys[2] < tmp_488 || (keys[2] == tmp_488 && values[2] < tmp_489); if tmp_480 == tmp_490 { keys[2] = tmp_488; values[2] = tmp_489; } let tmp_491 = smem_keys[tmp_481 * WPT + 28u]; let tmp_492 = smem_vals[tmp_481 * WPT + 28u]; let tmp_493 = keys[3] < tmp_491 || (keys[3] == tmp_491 && values[3] < tmp_492); if tmp_480 == tmp_493 { keys[3] = tmp_491; values[3] = tmp_492; } let tmp_494 = smem_keys[tmp_481 * WPT + 27u]; let tmp_495 = smem_vals[tmp_481 * WPT + 27u]; let tmp_496 = keys[4] < tmp_494 || (keys[4] == tmp_494 && values[4] < tmp_495); if tmp_480 == tmp_496 { keys[4] = tmp_494; values[4] = tmp_495; } let tmp_497 = smem_keys[tmp_481 * WPT + 26u]; let tmp_498 = smem_vals[tmp_481 * WPT + 26u]; let tmp_499 = keys[5] < tmp_497 || (keys[5] == tmp_497 && values[5] < tmp_498); if tmp_480 == tmp_499 { keys[5] = tmp_497; values[5] = tmp_498; } let tmp_500 = smem_keys[tmp_481 * WPT + 25u]; let tmp_501 = smem_vals[tmp_481 * WPT + 25u]; let tmp_502 = keys[6] < tmp_500 || (keys[6] == tmp_500 && values[6] < tmp_501); if tmp_480 == tmp_502 { keys[6] = tmp_500; values[6] = tmp_501; } let tmp_503 = smem_keys[tmp_481 * WPT + 24u]; let tmp_504 = smem_vals[tmp_481 * WPT + 24u]; let tmp_505 = keys[7] < tmp_503 || (keys[7] == tmp_503 && values[7] < tmp_504); if tmp_480 == tmp_505 { keys[7] = tmp_503; values[7] = tmp_504; } let tmp_506 = smem_keys[tmp_481 * WPT + 23u]; let tmp_507 = smem_vals[tmp_481 * WPT + 23u]; let tmp_508 = keys[8] < tmp_506 || (keys[8] == tmp_506 && values[8] < tmp_507); if tmp_480 == tmp_508 { keys[8] = tmp_506; values[8] = tmp_507; } let tmp_509 = smem_keys[tmp_481 * WPT + 22u]; let tmp_510 = smem_vals[tmp_481 * WPT + 22u]; let tmp_511 = keys[9] < tmp_509 || (keys[9] == tmp_509 && values[9] < tmp_510); if tmp_480 == tmp_511 { keys[9] = tmp_509; values[9] = tmp_510; } let tmp_512 = smem_keys[tmp_481 * WPT + 21u]; let tmp_513 = smem_vals[tmp_481 * WPT + 21u]; let tmp_514 = keys[10] < tmp_512 || (keys[10] == tmp_512 && values[10] < tmp_513); if tmp_480 == tmp_514 { keys[10] = tmp_512; values[10] = tmp_513; } let tmp_515 = smem_keys[tmp_481 * WPT + 20u]; let tmp_516 = smem_vals[tmp_481 * WPT + 20u]; let tmp_517 = keys[11] < tmp_515 || (keys[11] == tmp_515 && values[11] < tmp_516); if tmp_480 == tmp_517 { keys[11] = tmp_515; values[11] = tmp_516; } let tmp_518 = smem_keys[tmp_481 * WPT + 19u]; let tmp_519 = smem_vals[tmp_481 * WPT + 19u]; let tmp_520 = keys[12] < tmp_518 || (keys[12] == tmp_518 && values[12] < tmp_519); if tmp_480 == tmp_520 { keys[12] = tmp_518; values[12] = tmp_519; } let tmp_521 = smem_keys[tmp_481 * WPT + 18u]; let tmp_522 = smem_vals[tmp_481 * WPT + 18u]; let tmp_523 = keys[13] < tmp_521 || (keys[13] == tmp_521 && values[13] < tmp_522); if tmp_480 == tmp_523 { keys[13] = tmp_521; values[13] = tmp_522; } let tmp_524 = smem_keys[tmp_481 * WPT + 17u]; let tmp_525 = smem_vals[tmp_481 * WPT + 17u]; let tmp_526 = keys[14] < tmp_524 || (keys[14] == tmp_524 && values[14] < tmp_525); if tmp_480 == tmp_526 { keys[14] = tmp_524; values[14] = tmp_525; } let tmp_527 = smem_keys[tmp_481 * WPT + 16u]; let tmp_528 = smem_vals[tmp_481 * WPT + 16u]; let tmp_529 = keys[15] < tmp_527 || (keys[15] == tmp_527 && values[15] < tmp_528); if tmp_480 == tmp_529 { keys[15] = tmp_527; values[15] = tmp_528; } let tmp_530 = smem_keys[tmp_481 * WPT + 15u]; let tmp_531 = smem_vals[tmp_481 * WPT + 15u]; let tmp_532 = keys[16] < tmp_530 || (keys[16] == tmp_530 && values[16] < tmp_531); if tmp_480 == tmp_532 { keys[16] = tmp_530; values[16] = tmp_531; } let tmp_533 = smem_keys[tmp_481 * WPT + 14u]; let tmp_534 = smem_vals[tmp_481 * WPT + 14u]; let tmp_535 = keys[17] < tmp_533 || (keys[17] == tmp_533 && values[17] < tmp_534); if tmp_480 == tmp_535 { keys[17] = tmp_533; values[17] = tmp_534; } let tmp_536 = smem_keys[tmp_481 * WPT + 13u]; let tmp_537 = smem_vals[tmp_481 * WPT + 13u]; let tmp_538 = keys[18] < tmp_536 || (keys[18] == tmp_536 && values[18] < tmp_537); if tmp_480 == tmp_538 { keys[18] = tmp_536; values[18] = tmp_537; } let tmp_539 = smem_keys[tmp_481 * WPT + 12u]; let tmp_540 = smem_vals[tmp_481 * WPT + 12u]; let tmp_541 = keys[19] < tmp_539 || (keys[19] == tmp_539 && values[19] < tmp_540); if tmp_480 == tmp_541 { keys[19] = tmp_539; values[19] = tmp_540; } let tmp_542 = smem_keys[tmp_481 * WPT + 11u]; let tmp_543 = smem_vals[tmp_481 * WPT + 11u]; let tmp_544 = keys[20] < tmp_542 || (keys[20] == tmp_542 && values[20] < tmp_543); if tmp_480 == tmp_544 { keys[20] = tmp_542; values[20] = tmp_543; } let tmp_545 = smem_keys[tmp_481 * WPT + 10u]; let tmp_546 = smem_vals[tmp_481 * WPT + 10u]; let tmp_547 = keys[21] < tmp_545 || (keys[21] == tmp_545 && values[21] < tmp_546); if tmp_480 == tmp_547 { keys[21] = tmp_545; values[21] = tmp_546; } let tmp_548 = smem_keys[tmp_481 * WPT + 9u]; let tmp_549 = smem_vals[tmp_481 * WPT + 9u]; let tmp_550 = keys[22] < tmp_548 || (keys[22] == tmp_548 && values[22] < tmp_549); if tmp_480 == tmp_550 { keys[22] = tmp_548; values[22] = tmp_549; } let tmp_551 = smem_keys[tmp_481 * WPT + 8u]; let tmp_552 = smem_vals[tmp_481 * WPT + 8u]; let tmp_553 = keys[23] < tmp_551 || (keys[23] == tmp_551 && values[23] < tmp_552); if tmp_480 == tmp_553 { keys[23] = tmp_551; values[23] = tmp_552; } let tmp_554 = smem_keys[tmp_481 * WPT + 7u]; let tmp_555 = smem_vals[tmp_481 * WPT + 7u]; let tmp_556 = keys[24] < tmp_554 || (keys[24] == tmp_554 && values[24] < tmp_555); if tmp_480 == tmp_556 { keys[24] = tmp_554; values[24] = tmp_555; } let tmp_557 = smem_keys[tmp_481 * WPT + 6u]; let tmp_558 = smem_vals[tmp_481 * WPT + 6u]; let tmp_559 = keys[25] < tmp_557 || (keys[25] == tmp_557 && values[25] < tmp_558); if tmp_480 == tmp_559 { keys[25] = tmp_557; values[25] = tmp_558; } let tmp_560 = smem_keys[tmp_481 * WPT + 5u]; let tmp_561 = smem_vals[tmp_481 * WPT + 5u]; let tmp_562 = keys[26] < tmp_560 || (keys[26] == tmp_560 && values[26] < tmp_561); if tmp_480 == tmp_562 { keys[26] = tmp_560; values[26] = tmp_561; } let tmp_563 = smem_keys[tmp_481 * WPT + 4u]; let tmp_564 = smem_vals[tmp_481 * WPT + 4u]; let tmp_565 = keys[27] < tmp_563 || (keys[27] == tmp_563 && values[27] < tmp_564); if tmp_480 == tmp_565 { keys[27] = tmp_563; values[27] = tmp_564; } let tmp_566 = smem_keys[tmp_481 * WPT + 3u]; let tmp_567 = smem_vals[tmp_481 * WPT + 3u]; let tmp_568 = keys[28] < tmp_566 || (keys[28] == tmp_566 && values[28] < tmp_567); if tmp_480 == tmp_568 { keys[28] = tmp_566; values[28] = tmp_567; } let tmp_569 = smem_keys[tmp_481 * WPT + 2u]; let tmp_570 = smem_vals[tmp_481 * WPT + 2u]; let tmp_571 = keys[29] < tmp_569 || (keys[29] == tmp_569 && values[29] < tmp_570); if tmp_480 == tmp_571 { keys[29] = tmp_569; values[29] = tmp_570; } let tmp_572 = smem_keys[tmp_481 * WPT + 1u]; let tmp_573 = smem_vals[tmp_481 * WPT + 1u]; let tmp_574 = keys[30] < tmp_572 || (keys[30] == tmp_572 && values[30] < tmp_573); if tmp_480 == tmp_574 { keys[30] = tmp_572; values[30] = tmp_573; } let tmp_575 = smem_keys[tmp_481 * WPT + 0u]; let tmp_576 = smem_vals[tmp_481 * WPT + 0u]; let tmp_577 = keys[31] < tmp_575 || (keys[31] == tmp_575 && values[31] < tmp_576); if tmp_480 == tmp_577 { keys[31] = tmp_575; values[31] = tmp_576; } workgroupBarrier(); }
    // exch_local(16,32) 
    // cmp_swap(0,16)
    if keys[0] > keys[16] || (keys[0] == keys[16] && values[0] > values[16]) {
    // swap(0,16) 
    { let tmp_578 = keys[0]; keys[0] = keys[16]; keys[16] = tmp_578;let tmp_579 = values[0]; values[0] = values[16]; values[16] = tmp_579; }
    }
    // cmp_swap(1,17)
    if keys[1] > keys[17] || (keys[1] == keys[17] && values[1] > values[17]) {
    // swap(1,17) 
    { let tmp_580 = keys[1]; keys[1] = keys[17]; keys[17] = tmp_580;let tmp_581 = values[1]; values[1] = values[17]; values[17] = tmp_581; }
    }
    // cmp_swap(2,18)
    if keys[2] > keys[18] || (keys[2] == keys[18] && values[2] > values[18]) {
    // swap(2,18) 
    { let tmp_582 = keys[2]; keys[2] = keys[18]; keys[18] = tmp_582;let tmp_583 = values[2]; values[2] = values[18]; values[18] = tmp_583; }
    }
    // cmp_swap(3,19)
    if keys[3] > keys[19] || (keys[3] == keys[19] && values[3] > values[19]) {
    // swap(3,19) 
    { let tmp_584 = keys[3]; keys[3] = keys[19]; keys[19] = tmp_584;let tmp_585 = values[3]; values[3] = values[19]; values[19] = tmp_585; }
    }
    // cmp_swap(4,20)
    if keys[4] > keys[20] || (keys[4] == keys[20] && values[4] > values[20]) {
    // swap(4,20) 
    { let tmp_586 = keys[4]; keys[4] = keys[20]; keys[20] = tmp_586;let tmp_587 = values[4]; values[4] = values[20]; values[20] = tmp_587; }
    }
    // cmp_swap(5,21)
    if keys[5] > keys[21] || (keys[5] == keys[21] && values[5] > values[21]) {
    // swap(5,21) 
    { let tmp_588 = keys[5]; keys[5] = keys[21]; keys[21] = tmp_588;let tmp_589 = values[5]; values[5] = values[21]; values[21] = tmp_589; }
    }
    // cmp_swap(6,22)
    if keys[6] > keys[22] || (keys[6] == keys[22] && values[6] > values[22]) {
    // swap(6,22) 
    { let tmp_590 = keys[6]; keys[6] = keys[22]; keys[22] = tmp_590;let tmp_591 = values[6]; values[6] = values[22]; values[22] = tmp_591; }
    }
    // cmp_swap(7,23)
    if keys[7] > keys[23] || (keys[7] == keys[23] && values[7] > values[23]) {
    // swap(7,23) 
    { let tmp_592 = keys[7]; keys[7] = keys[23]; keys[23] = tmp_592;let tmp_593 = values[7]; values[7] = values[23]; values[23] = tmp_593; }
    }
    // cmp_swap(8,24)
    if keys[8] > keys[24] || (keys[8] == keys[24] && values[8] > values[24]) {
    // swap(8,24) 
    { let tmp_594 = keys[8]; keys[8] = keys[24]; keys[24] = tmp_594;let tmp_595 = values[8]; values[8] = values[24]; values[24] = tmp_595; }
    }
    // cmp_swap(9,25)
    if keys[9] > keys[25] || (keys[9] == keys[25] && values[9] > values[25]) {
    // swap(9,25) 
    { let tmp_596 = keys[9]; keys[9] = keys[25]; keys[25] = tmp_596;let tmp_597 = values[9]; values[9] = values[25]; values[25] = tmp_597; }
    }
    // cmp_swap(10,26)
    if keys[10] > keys[26] || (keys[10] == keys[26] && values[10] > values[26]) {
    // swap(10,26) 
    { let tmp_598 = keys[10]; keys[10] = keys[26]; keys[26] = tmp_598;let tmp_599 = values[10]; values[10] = values[26]; values[26] = tmp_599; }
    }
    // cmp_swap(11,27)
    if keys[11] > keys[27] || (keys[11] == keys[27] && values[11] > values[27]) {
    // swap(11,27) 
    { let tmp_600 = keys[11]; keys[11] = keys[27]; keys[27] = tmp_600;let tmp_601 = values[11]; values[11] = values[27]; values[27] = tmp_601; }
    }
    // cmp_swap(12,28)
    if keys[12] > keys[28] || (keys[12] == keys[28] && values[12] > values[28]) {
    // swap(12,28) 
    { let tmp_602 = keys[12]; keys[12] = keys[28]; keys[28] = tmp_602;let tmp_603 = values[12]; values[12] = values[28]; values[28] = tmp_603; }
    }
    // cmp_swap(13,29)
    if keys[13] > keys[29] || (keys[13] == keys[29] && values[13] > values[29]) {
    // swap(13,29) 
    { let tmp_604 = keys[13]; keys[13] = keys[29]; keys[29] = tmp_604;let tmp_605 = values[13]; values[13] = values[29]; values[29] = tmp_605; }
    }
    // cmp_swap(14,30)
    if keys[14] > keys[30] || (keys[14] == keys[30] && values[14] > values[30]) {
    // swap(14,30) 
    { let tmp_606 = keys[14]; keys[14] = keys[30]; keys[30] = tmp_606;let tmp_607 = values[14]; values[14] = values[30]; values[30] = tmp_607; }
    }
    // cmp_swap(15,31)
    if keys[15] > keys[31] || (keys[15] == keys[31] && values[15] > values[31]) {
    // swap(15,31) 
    { let tmp_608 = keys[15]; keys[15] = keys[31]; keys[31] = tmp_608;let tmp_609 = values[15]; values[15] = values[31]; values[31] = tmp_609; }
    }
    // exch_local(8,32) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_610 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_610;let tmp_611 = values[0]; values[0] = values[8]; values[8] = tmp_611; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_612 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_612;let tmp_613 = values[1]; values[1] = values[9]; values[9] = tmp_613; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_614 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_614;let tmp_615 = values[2]; values[2] = values[10]; values[10] = tmp_615; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_616 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_616;let tmp_617 = values[3]; values[3] = values[11]; values[11] = tmp_617; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_618 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_618;let tmp_619 = values[4]; values[4] = values[12]; values[12] = tmp_619; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_620 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_620;let tmp_621 = values[5]; values[5] = values[13]; values[13] = tmp_621; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_622 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_622;let tmp_623 = values[6]; values[6] = values[14]; values[14] = tmp_623; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_624 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_624;let tmp_625 = values[7]; values[7] = values[15]; values[15] = tmp_625; }
    }
    // cmp_swap(16,24)
    if keys[16] > keys[24] || (keys[16] == keys[24] && values[16] > values[24]) {
    // swap(16,24) 
    { let tmp_626 = keys[16]; keys[16] = keys[24]; keys[24] = tmp_626;let tmp_627 = values[16]; values[16] = values[24]; values[24] = tmp_627; }
    }
    // cmp_swap(17,25)
    if keys[17] > keys[25] || (keys[17] == keys[25] && values[17] > values[25]) {
    // swap(17,25) 
    { let tmp_628 = keys[17]; keys[17] = keys[25]; keys[25] = tmp_628;let tmp_629 = values[17]; values[17] = values[25]; values[25] = tmp_629; }
    }
    // cmp_swap(18,26)
    if keys[18] > keys[26] || (keys[18] == keys[26] && values[18] > values[26]) {
    // swap(18,26) 
    { let tmp_630 = keys[18]; keys[18] = keys[26]; keys[26] = tmp_630;let tmp_631 = values[18]; values[18] = values[26]; values[26] = tmp_631; }
    }
    // cmp_swap(19,27)
    if keys[19] > keys[27] || (keys[19] == keys[27] && values[19] > values[27]) {
    // swap(19,27) 
    { let tmp_632 = keys[19]; keys[19] = keys[27]; keys[27] = tmp_632;let tmp_633 = values[19]; values[19] = values[27]; values[27] = tmp_633; }
    }
    // cmp_swap(20,28)
    if keys[20] > keys[28] || (keys[20] == keys[28] && values[20] > values[28]) {
    // swap(20,28) 
    { let tmp_634 = keys[20]; keys[20] = keys[28]; keys[28] = tmp_634;let tmp_635 = values[20]; values[20] = values[28]; values[28] = tmp_635; }
    }
    // cmp_swap(21,29)
    if keys[21] > keys[29] || (keys[21] == keys[29] && values[21] > values[29]) {
    // swap(21,29) 
    { let tmp_636 = keys[21]; keys[21] = keys[29]; keys[29] = tmp_636;let tmp_637 = values[21]; values[21] = values[29]; values[29] = tmp_637; }
    }
    // cmp_swap(22,30)
    if keys[22] > keys[30] || (keys[22] == keys[30] && values[22] > values[30]) {
    // swap(22,30) 
    { let tmp_638 = keys[22]; keys[22] = keys[30]; keys[30] = tmp_638;let tmp_639 = values[22]; values[22] = values[30]; values[30] = tmp_639; }
    }
    // cmp_swap(23,31)
    if keys[23] > keys[31] || (keys[23] == keys[31] && values[23] > values[31]) {
    // swap(23,31) 
    { let tmp_640 = keys[23]; keys[23] = keys[31]; keys[31] = tmp_640;let tmp_641 = values[23]; values[23] = values[31]; values[31] = tmp_641; }
    }
    // exch_local(4,32) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_642 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_642;let tmp_643 = values[0]; values[0] = values[4]; values[4] = tmp_643; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_644 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_644;let tmp_645 = values[1]; values[1] = values[5]; values[5] = tmp_645; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_646 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_646;let tmp_647 = values[2]; values[2] = values[6]; values[6] = tmp_647; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_648 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_648;let tmp_649 = values[3]; values[3] = values[7]; values[7] = tmp_649; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_650 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_650;let tmp_651 = values[8]; values[8] = values[12]; values[12] = tmp_651; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_652 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_652;let tmp_653 = values[9]; values[9] = values[13]; values[13] = tmp_653; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_654 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_654;let tmp_655 = values[10]; values[10] = values[14]; values[14] = tmp_655; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_656 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_656;let tmp_657 = values[11]; values[11] = values[15]; values[15] = tmp_657; }
    }
    // cmp_swap(16,20)
    if keys[16] > keys[20] || (keys[16] == keys[20] && values[16] > values[20]) {
    // swap(16,20) 
    { let tmp_658 = keys[16]; keys[16] = keys[20]; keys[20] = tmp_658;let tmp_659 = values[16]; values[16] = values[20]; values[20] = tmp_659; }
    }
    // cmp_swap(17,21)
    if keys[17] > keys[21] || (keys[17] == keys[21] && values[17] > values[21]) {
    // swap(17,21) 
    { let tmp_660 = keys[17]; keys[17] = keys[21]; keys[21] = tmp_660;let tmp_661 = values[17]; values[17] = values[21]; values[21] = tmp_661; }
    }
    // cmp_swap(18,22)
    if keys[18] > keys[22] || (keys[18] == keys[22] && values[18] > values[22]) {
    // swap(18,22) 
    { let tmp_662 = keys[18]; keys[18] = keys[22]; keys[22] = tmp_662;let tmp_663 = values[18]; values[18] = values[22]; values[22] = tmp_663; }
    }
    // cmp_swap(19,23)
    if keys[19] > keys[23] || (keys[19] == keys[23] && values[19] > values[23]) {
    // swap(19,23) 
    { let tmp_664 = keys[19]; keys[19] = keys[23]; keys[23] = tmp_664;let tmp_665 = values[19]; values[19] = values[23]; values[23] = tmp_665; }
    }
    // cmp_swap(24,28)
    if keys[24] > keys[28] || (keys[24] == keys[28] && values[24] > values[28]) {
    // swap(24,28) 
    { let tmp_666 = keys[24]; keys[24] = keys[28]; keys[28] = tmp_666;let tmp_667 = values[24]; values[24] = values[28]; values[28] = tmp_667; }
    }
    // cmp_swap(25,29)
    if keys[25] > keys[29] || (keys[25] == keys[29] && values[25] > values[29]) {
    // swap(25,29) 
    { let tmp_668 = keys[25]; keys[25] = keys[29]; keys[29] = tmp_668;let tmp_669 = values[25]; values[25] = values[29]; values[29] = tmp_669; }
    }
    // cmp_swap(26,30)
    if keys[26] > keys[30] || (keys[26] == keys[30] && values[26] > values[30]) {
    // swap(26,30) 
    { let tmp_670 = keys[26]; keys[26] = keys[30]; keys[30] = tmp_670;let tmp_671 = values[26]; values[26] = values[30]; values[30] = tmp_671; }
    }
    // cmp_swap(27,31)
    if keys[27] > keys[31] || (keys[27] == keys[31] && values[27] > values[31]) {
    // swap(27,31) 
    { let tmp_672 = keys[27]; keys[27] = keys[31]; keys[31] = tmp_672;let tmp_673 = values[27]; values[27] = values[31]; values[31] = tmp_673; }
    }
    // exch_local(2,32) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_674 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_674;let tmp_675 = values[0]; values[0] = values[2]; values[2] = tmp_675; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_676 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_676;let tmp_677 = values[1]; values[1] = values[3]; values[3] = tmp_677; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_678 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_678;let tmp_679 = values[4]; values[4] = values[6]; values[6] = tmp_679; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_680 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_680;let tmp_681 = values[5]; values[5] = values[7]; values[7] = tmp_681; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_682 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_682;let tmp_683 = values[8]; values[8] = values[10]; values[10] = tmp_683; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_684 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_684;let tmp_685 = values[9]; values[9] = values[11]; values[11] = tmp_685; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_686 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_686;let tmp_687 = values[12]; values[12] = values[14]; values[14] = tmp_687; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_688 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_688;let tmp_689 = values[13]; values[13] = values[15]; values[15] = tmp_689; }
    }
    // cmp_swap(16,18)
    if keys[16] > keys[18] || (keys[16] == keys[18] && values[16] > values[18]) {
    // swap(16,18) 
    { let tmp_690 = keys[16]; keys[16] = keys[18]; keys[18] = tmp_690;let tmp_691 = values[16]; values[16] = values[18]; values[18] = tmp_691; }
    }
    // cmp_swap(17,19)
    if keys[17] > keys[19] || (keys[17] == keys[19] && values[17] > values[19]) {
    // swap(17,19) 
    { let tmp_692 = keys[17]; keys[17] = keys[19]; keys[19] = tmp_692;let tmp_693 = values[17]; values[17] = values[19]; values[19] = tmp_693; }
    }
    // cmp_swap(20,22)
    if keys[20] > keys[22] || (keys[20] == keys[22] && values[20] > values[22]) {
    // swap(20,22) 
    { let tmp_694 = keys[20]; keys[20] = keys[22]; keys[22] = tmp_694;let tmp_695 = values[20]; values[20] = values[22]; values[22] = tmp_695; }
    }
    // cmp_swap(21,23)
    if keys[21] > keys[23] || (keys[21] == keys[23] && values[21] > values[23]) {
    // swap(21,23) 
    { let tmp_696 = keys[21]; keys[21] = keys[23]; keys[23] = tmp_696;let tmp_697 = values[21]; values[21] = values[23]; values[23] = tmp_697; }
    }
    // cmp_swap(24,26)
    if keys[24] > keys[26] || (keys[24] == keys[26] && values[24] > values[26]) {
    // swap(24,26) 
    { let tmp_698 = keys[24]; keys[24] = keys[26]; keys[26] = tmp_698;let tmp_699 = values[24]; values[24] = values[26]; values[26] = tmp_699; }
    }
    // cmp_swap(25,27)
    if keys[25] > keys[27] || (keys[25] == keys[27] && values[25] > values[27]) {
    // swap(25,27) 
    { let tmp_700 = keys[25]; keys[25] = keys[27]; keys[27] = tmp_700;let tmp_701 = values[25]; values[25] = values[27]; values[27] = tmp_701; }
    }
    // cmp_swap(28,30)
    if keys[28] > keys[30] || (keys[28] == keys[30] && values[28] > values[30]) {
    // swap(28,30) 
    { let tmp_702 = keys[28]; keys[28] = keys[30]; keys[30] = tmp_702;let tmp_703 = values[28]; values[28] = values[30]; values[30] = tmp_703; }
    }
    // cmp_swap(29,31)
    if keys[29] > keys[31] || (keys[29] == keys[31] && values[29] > values[31]) {
    // swap(29,31) 
    { let tmp_704 = keys[29]; keys[29] = keys[31]; keys[31] = tmp_704;let tmp_705 = values[29]; values[29] = values[31]; values[31] = tmp_705; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_706 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_706;let tmp_707 = values[0]; values[0] = values[1]; values[1] = tmp_707; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_708 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_708;let tmp_709 = values[2]; values[2] = values[3]; values[3] = tmp_709; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_710 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_710;let tmp_711 = values[4]; values[4] = values[5]; values[5] = tmp_711; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_712 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_712;let tmp_713 = values[6]; values[6] = values[7]; values[7] = tmp_713; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_714 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_714;let tmp_715 = values[8]; values[8] = values[9]; values[9] = tmp_715; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_716 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_716;let tmp_717 = values[10]; values[10] = values[11]; values[11] = tmp_717; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_718 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_718;let tmp_719 = values[12]; values[12] = values[13]; values[13] = tmp_719; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_720 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_720;let tmp_721 = values[14]; values[14] = values[15]; values[15] = tmp_721; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_722 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_722;let tmp_723 = values[16]; values[16] = values[17]; values[17] = tmp_723; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_724 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_724;let tmp_725 = values[18]; values[18] = values[19]; values[19] = tmp_725; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_726 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_726;let tmp_727 = values[20]; values[20] = values[21]; values[21] = tmp_727; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_728 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_728;let tmp_729 = values[22]; values[22] = values[23]; values[23] = tmp_729; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_730 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_730;let tmp_731 = values[24]; values[24] = values[25]; values[25] = tmp_731; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_732 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_732;let tmp_733 = values[26]; values[26] = values[27]; values[27] = tmp_733; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_734 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_734;let tmp_735 = values[28]; values[28] = values[29]; values[29] = tmp_735; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_736 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_736;let tmp_737 = values[30]; values[30] = values[31]; values[31] = tmp_737; }
    }
    // exch_intxn(tmask:3,swbit:1,wpt:32)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_738 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_739 = seg_base + (local_tid ^ 3u); let tmp_740 = smem_keys[tmp_739 * WPT + 31u]; let tmp_741 = smem_vals[tmp_739 * WPT + 31u]; let tmp_742 = keys[0] < tmp_740 || (keys[0] == tmp_740 && values[0] < tmp_741); if tmp_738 == tmp_742 { keys[0] = tmp_740; values[0] = tmp_741; } let tmp_743 = smem_keys[tmp_739 * WPT + 30u]; let tmp_744 = smem_vals[tmp_739 * WPT + 30u]; let tmp_745 = keys[1] < tmp_743 || (keys[1] == tmp_743 && values[1] < tmp_744); if tmp_738 == tmp_745 { keys[1] = tmp_743; values[1] = tmp_744; } let tmp_746 = smem_keys[tmp_739 * WPT + 29u]; let tmp_747 = smem_vals[tmp_739 * WPT + 29u]; let tmp_748 = keys[2] < tmp_746 || (keys[2] == tmp_746 && values[2] < tmp_747); if tmp_738 == tmp_748 { keys[2] = tmp_746; values[2] = tmp_747; } let tmp_749 = smem_keys[tmp_739 * WPT + 28u]; let tmp_750 = smem_vals[tmp_739 * WPT + 28u]; let tmp_751 = keys[3] < tmp_749 || (keys[3] == tmp_749 && values[3] < tmp_750); if tmp_738 == tmp_751 { keys[3] = tmp_749; values[3] = tmp_750; } let tmp_752 = smem_keys[tmp_739 * WPT + 27u]; let tmp_753 = smem_vals[tmp_739 * WPT + 27u]; let tmp_754 = keys[4] < tmp_752 || (keys[4] == tmp_752 && values[4] < tmp_753); if tmp_738 == tmp_754 { keys[4] = tmp_752; values[4] = tmp_753; } let tmp_755 = smem_keys[tmp_739 * WPT + 26u]; let tmp_756 = smem_vals[tmp_739 * WPT + 26u]; let tmp_757 = keys[5] < tmp_755 || (keys[5] == tmp_755 && values[5] < tmp_756); if tmp_738 == tmp_757 { keys[5] = tmp_755; values[5] = tmp_756; } let tmp_758 = smem_keys[tmp_739 * WPT + 25u]; let tmp_759 = smem_vals[tmp_739 * WPT + 25u]; let tmp_760 = keys[6] < tmp_758 || (keys[6] == tmp_758 && values[6] < tmp_759); if tmp_738 == tmp_760 { keys[6] = tmp_758; values[6] = tmp_759; } let tmp_761 = smem_keys[tmp_739 * WPT + 24u]; let tmp_762 = smem_vals[tmp_739 * WPT + 24u]; let tmp_763 = keys[7] < tmp_761 || (keys[7] == tmp_761 && values[7] < tmp_762); if tmp_738 == tmp_763 { keys[7] = tmp_761; values[7] = tmp_762; } let tmp_764 = smem_keys[tmp_739 * WPT + 23u]; let tmp_765 = smem_vals[tmp_739 * WPT + 23u]; let tmp_766 = keys[8] < tmp_764 || (keys[8] == tmp_764 && values[8] < tmp_765); if tmp_738 == tmp_766 { keys[8] = tmp_764; values[8] = tmp_765; } let tmp_767 = smem_keys[tmp_739 * WPT + 22u]; let tmp_768 = smem_vals[tmp_739 * WPT + 22u]; let tmp_769 = keys[9] < tmp_767 || (keys[9] == tmp_767 && values[9] < tmp_768); if tmp_738 == tmp_769 { keys[9] = tmp_767; values[9] = tmp_768; } let tmp_770 = smem_keys[tmp_739 * WPT + 21u]; let tmp_771 = smem_vals[tmp_739 * WPT + 21u]; let tmp_772 = keys[10] < tmp_770 || (keys[10] == tmp_770 && values[10] < tmp_771); if tmp_738 == tmp_772 { keys[10] = tmp_770; values[10] = tmp_771; } let tmp_773 = smem_keys[tmp_739 * WPT + 20u]; let tmp_774 = smem_vals[tmp_739 * WPT + 20u]; let tmp_775 = keys[11] < tmp_773 || (keys[11] == tmp_773 && values[11] < tmp_774); if tmp_738 == tmp_775 { keys[11] = tmp_773; values[11] = tmp_774; } let tmp_776 = smem_keys[tmp_739 * WPT + 19u]; let tmp_777 = smem_vals[tmp_739 * WPT + 19u]; let tmp_778 = keys[12] < tmp_776 || (keys[12] == tmp_776 && values[12] < tmp_777); if tmp_738 == tmp_778 { keys[12] = tmp_776; values[12] = tmp_777; } let tmp_779 = smem_keys[tmp_739 * WPT + 18u]; let tmp_780 = smem_vals[tmp_739 * WPT + 18u]; let tmp_781 = keys[13] < tmp_779 || (keys[13] == tmp_779 && values[13] < tmp_780); if tmp_738 == tmp_781 { keys[13] = tmp_779; values[13] = tmp_780; } let tmp_782 = smem_keys[tmp_739 * WPT + 17u]; let tmp_783 = smem_vals[tmp_739 * WPT + 17u]; let tmp_784 = keys[14] < tmp_782 || (keys[14] == tmp_782 && values[14] < tmp_783); if tmp_738 == tmp_784 { keys[14] = tmp_782; values[14] = tmp_783; } let tmp_785 = smem_keys[tmp_739 * WPT + 16u]; let tmp_786 = smem_vals[tmp_739 * WPT + 16u]; let tmp_787 = keys[15] < tmp_785 || (keys[15] == tmp_785 && values[15] < tmp_786); if tmp_738 == tmp_787 { keys[15] = tmp_785; values[15] = tmp_786; } let tmp_788 = smem_keys[tmp_739 * WPT + 15u]; let tmp_789 = smem_vals[tmp_739 * WPT + 15u]; let tmp_790 = keys[16] < tmp_788 || (keys[16] == tmp_788 && values[16] < tmp_789); if tmp_738 == tmp_790 { keys[16] = tmp_788; values[16] = tmp_789; } let tmp_791 = smem_keys[tmp_739 * WPT + 14u]; let tmp_792 = smem_vals[tmp_739 * WPT + 14u]; let tmp_793 = keys[17] < tmp_791 || (keys[17] == tmp_791 && values[17] < tmp_792); if tmp_738 == tmp_793 { keys[17] = tmp_791; values[17] = tmp_792; } let tmp_794 = smem_keys[tmp_739 * WPT + 13u]; let tmp_795 = smem_vals[tmp_739 * WPT + 13u]; let tmp_796 = keys[18] < tmp_794 || (keys[18] == tmp_794 && values[18] < tmp_795); if tmp_738 == tmp_796 { keys[18] = tmp_794; values[18] = tmp_795; } let tmp_797 = smem_keys[tmp_739 * WPT + 12u]; let tmp_798 = smem_vals[tmp_739 * WPT + 12u]; let tmp_799 = keys[19] < tmp_797 || (keys[19] == tmp_797 && values[19] < tmp_798); if tmp_738 == tmp_799 { keys[19] = tmp_797; values[19] = tmp_798; } let tmp_800 = smem_keys[tmp_739 * WPT + 11u]; let tmp_801 = smem_vals[tmp_739 * WPT + 11u]; let tmp_802 = keys[20] < tmp_800 || (keys[20] == tmp_800 && values[20] < tmp_801); if tmp_738 == tmp_802 { keys[20] = tmp_800; values[20] = tmp_801; } let tmp_803 = smem_keys[tmp_739 * WPT + 10u]; let tmp_804 = smem_vals[tmp_739 * WPT + 10u]; let tmp_805 = keys[21] < tmp_803 || (keys[21] == tmp_803 && values[21] < tmp_804); if tmp_738 == tmp_805 { keys[21] = tmp_803; values[21] = tmp_804; } let tmp_806 = smem_keys[tmp_739 * WPT + 9u]; let tmp_807 = smem_vals[tmp_739 * WPT + 9u]; let tmp_808 = keys[22] < tmp_806 || (keys[22] == tmp_806 && values[22] < tmp_807); if tmp_738 == tmp_808 { keys[22] = tmp_806; values[22] = tmp_807; } let tmp_809 = smem_keys[tmp_739 * WPT + 8u]; let tmp_810 = smem_vals[tmp_739 * WPT + 8u]; let tmp_811 = keys[23] < tmp_809 || (keys[23] == tmp_809 && values[23] < tmp_810); if tmp_738 == tmp_811 { keys[23] = tmp_809; values[23] = tmp_810; } let tmp_812 = smem_keys[tmp_739 * WPT + 7u]; let tmp_813 = smem_vals[tmp_739 * WPT + 7u]; let tmp_814 = keys[24] < tmp_812 || (keys[24] == tmp_812 && values[24] < tmp_813); if tmp_738 == tmp_814 { keys[24] = tmp_812; values[24] = tmp_813; } let tmp_815 = smem_keys[tmp_739 * WPT + 6u]; let tmp_816 = smem_vals[tmp_739 * WPT + 6u]; let tmp_817 = keys[25] < tmp_815 || (keys[25] == tmp_815 && values[25] < tmp_816); if tmp_738 == tmp_817 { keys[25] = tmp_815; values[25] = tmp_816; } let tmp_818 = smem_keys[tmp_739 * WPT + 5u]; let tmp_819 = smem_vals[tmp_739 * WPT + 5u]; let tmp_820 = keys[26] < tmp_818 || (keys[26] == tmp_818 && values[26] < tmp_819); if tmp_738 == tmp_820 { keys[26] = tmp_818; values[26] = tmp_819; } let tmp_821 = smem_keys[tmp_739 * WPT + 4u]; let tmp_822 = smem_vals[tmp_739 * WPT + 4u]; let tmp_823 = keys[27] < tmp_821 || (keys[27] == tmp_821 && values[27] < tmp_822); if tmp_738 == tmp_823 { keys[27] = tmp_821; values[27] = tmp_822; } let tmp_824 = smem_keys[tmp_739 * WPT + 3u]; let tmp_825 = smem_vals[tmp_739 * WPT + 3u]; let tmp_826 = keys[28] < tmp_824 || (keys[28] == tmp_824 && values[28] < tmp_825); if tmp_738 == tmp_826 { keys[28] = tmp_824; values[28] = tmp_825; } let tmp_827 = smem_keys[tmp_739 * WPT + 2u]; let tmp_828 = smem_vals[tmp_739 * WPT + 2u]; let tmp_829 = keys[29] < tmp_827 || (keys[29] == tmp_827 && values[29] < tmp_828); if tmp_738 == tmp_829 { keys[29] = tmp_827; values[29] = tmp_828; } let tmp_830 = smem_keys[tmp_739 * WPT + 1u]; let tmp_831 = smem_vals[tmp_739 * WPT + 1u]; let tmp_832 = keys[30] < tmp_830 || (keys[30] == tmp_830 && values[30] < tmp_831); if tmp_738 == tmp_832 { keys[30] = tmp_830; values[30] = tmp_831; } let tmp_833 = smem_keys[tmp_739 * WPT + 0u]; let tmp_834 = smem_vals[tmp_739 * WPT + 0u]; let tmp_835 = keys[31] < tmp_833 || (keys[31] == tmp_833 && values[31] < tmp_834); if tmp_738 == tmp_835 { keys[31] = tmp_833; values[31] = tmp_834; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_836 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_837 = seg_base + (local_tid ^ 1u); let tmp_838 = smem_keys[tmp_837 * WPT + 0u]; let tmp_839 = smem_vals[tmp_837 * WPT + 0u]; let tmp_840 = keys[0] < tmp_838 || (keys[0] == tmp_838 && values[0] < tmp_839); if tmp_836 == tmp_840 { keys[0] = tmp_838; values[0] = tmp_839; } let tmp_841 = smem_keys[tmp_837 * WPT + 1u]; let tmp_842 = smem_vals[tmp_837 * WPT + 1u]; let tmp_843 = keys[1] < tmp_841 || (keys[1] == tmp_841 && values[1] < tmp_842); if tmp_836 == tmp_843 { keys[1] = tmp_841; values[1] = tmp_842; } let tmp_844 = smem_keys[tmp_837 * WPT + 2u]; let tmp_845 = smem_vals[tmp_837 * WPT + 2u]; let tmp_846 = keys[2] < tmp_844 || (keys[2] == tmp_844 && values[2] < tmp_845); if tmp_836 == tmp_846 { keys[2] = tmp_844; values[2] = tmp_845; } let tmp_847 = smem_keys[tmp_837 * WPT + 3u]; let tmp_848 = smem_vals[tmp_837 * WPT + 3u]; let tmp_849 = keys[3] < tmp_847 || (keys[3] == tmp_847 && values[3] < tmp_848); if tmp_836 == tmp_849 { keys[3] = tmp_847; values[3] = tmp_848; } let tmp_850 = smem_keys[tmp_837 * WPT + 4u]; let tmp_851 = smem_vals[tmp_837 * WPT + 4u]; let tmp_852 = keys[4] < tmp_850 || (keys[4] == tmp_850 && values[4] < tmp_851); if tmp_836 == tmp_852 { keys[4] = tmp_850; values[4] = tmp_851; } let tmp_853 = smem_keys[tmp_837 * WPT + 5u]; let tmp_854 = smem_vals[tmp_837 * WPT + 5u]; let tmp_855 = keys[5] < tmp_853 || (keys[5] == tmp_853 && values[5] < tmp_854); if tmp_836 == tmp_855 { keys[5] = tmp_853; values[5] = tmp_854; } let tmp_856 = smem_keys[tmp_837 * WPT + 6u]; let tmp_857 = smem_vals[tmp_837 * WPT + 6u]; let tmp_858 = keys[6] < tmp_856 || (keys[6] == tmp_856 && values[6] < tmp_857); if tmp_836 == tmp_858 { keys[6] = tmp_856; values[6] = tmp_857; } let tmp_859 = smem_keys[tmp_837 * WPT + 7u]; let tmp_860 = smem_vals[tmp_837 * WPT + 7u]; let tmp_861 = keys[7] < tmp_859 || (keys[7] == tmp_859 && values[7] < tmp_860); if tmp_836 == tmp_861 { keys[7] = tmp_859; values[7] = tmp_860; } let tmp_862 = smem_keys[tmp_837 * WPT + 8u]; let tmp_863 = smem_vals[tmp_837 * WPT + 8u]; let tmp_864 = keys[8] < tmp_862 || (keys[8] == tmp_862 && values[8] < tmp_863); if tmp_836 == tmp_864 { keys[8] = tmp_862; values[8] = tmp_863; } let tmp_865 = smem_keys[tmp_837 * WPT + 9u]; let tmp_866 = smem_vals[tmp_837 * WPT + 9u]; let tmp_867 = keys[9] < tmp_865 || (keys[9] == tmp_865 && values[9] < tmp_866); if tmp_836 == tmp_867 { keys[9] = tmp_865; values[9] = tmp_866; } let tmp_868 = smem_keys[tmp_837 * WPT + 10u]; let tmp_869 = smem_vals[tmp_837 * WPT + 10u]; let tmp_870 = keys[10] < tmp_868 || (keys[10] == tmp_868 && values[10] < tmp_869); if tmp_836 == tmp_870 { keys[10] = tmp_868; values[10] = tmp_869; } let tmp_871 = smem_keys[tmp_837 * WPT + 11u]; let tmp_872 = smem_vals[tmp_837 * WPT + 11u]; let tmp_873 = keys[11] < tmp_871 || (keys[11] == tmp_871 && values[11] < tmp_872); if tmp_836 == tmp_873 { keys[11] = tmp_871; values[11] = tmp_872; } let tmp_874 = smem_keys[tmp_837 * WPT + 12u]; let tmp_875 = smem_vals[tmp_837 * WPT + 12u]; let tmp_876 = keys[12] < tmp_874 || (keys[12] == tmp_874 && values[12] < tmp_875); if tmp_836 == tmp_876 { keys[12] = tmp_874; values[12] = tmp_875; } let tmp_877 = smem_keys[tmp_837 * WPT + 13u]; let tmp_878 = smem_vals[tmp_837 * WPT + 13u]; let tmp_879 = keys[13] < tmp_877 || (keys[13] == tmp_877 && values[13] < tmp_878); if tmp_836 == tmp_879 { keys[13] = tmp_877; values[13] = tmp_878; } let tmp_880 = smem_keys[tmp_837 * WPT + 14u]; let tmp_881 = smem_vals[tmp_837 * WPT + 14u]; let tmp_882 = keys[14] < tmp_880 || (keys[14] == tmp_880 && values[14] < tmp_881); if tmp_836 == tmp_882 { keys[14] = tmp_880; values[14] = tmp_881; } let tmp_883 = smem_keys[tmp_837 * WPT + 15u]; let tmp_884 = smem_vals[tmp_837 * WPT + 15u]; let tmp_885 = keys[15] < tmp_883 || (keys[15] == tmp_883 && values[15] < tmp_884); if tmp_836 == tmp_885 { keys[15] = tmp_883; values[15] = tmp_884; } let tmp_886 = smem_keys[tmp_837 * WPT + 16u]; let tmp_887 = smem_vals[tmp_837 * WPT + 16u]; let tmp_888 = keys[16] < tmp_886 || (keys[16] == tmp_886 && values[16] < tmp_887); if tmp_836 == tmp_888 { keys[16] = tmp_886; values[16] = tmp_887; } let tmp_889 = smem_keys[tmp_837 * WPT + 17u]; let tmp_890 = smem_vals[tmp_837 * WPT + 17u]; let tmp_891 = keys[17] < tmp_889 || (keys[17] == tmp_889 && values[17] < tmp_890); if tmp_836 == tmp_891 { keys[17] = tmp_889; values[17] = tmp_890; } let tmp_892 = smem_keys[tmp_837 * WPT + 18u]; let tmp_893 = smem_vals[tmp_837 * WPT + 18u]; let tmp_894 = keys[18] < tmp_892 || (keys[18] == tmp_892 && values[18] < tmp_893); if tmp_836 == tmp_894 { keys[18] = tmp_892; values[18] = tmp_893; } let tmp_895 = smem_keys[tmp_837 * WPT + 19u]; let tmp_896 = smem_vals[tmp_837 * WPT + 19u]; let tmp_897 = keys[19] < tmp_895 || (keys[19] == tmp_895 && values[19] < tmp_896); if tmp_836 == tmp_897 { keys[19] = tmp_895; values[19] = tmp_896; } let tmp_898 = smem_keys[tmp_837 * WPT + 20u]; let tmp_899 = smem_vals[tmp_837 * WPT + 20u]; let tmp_900 = keys[20] < tmp_898 || (keys[20] == tmp_898 && values[20] < tmp_899); if tmp_836 == tmp_900 { keys[20] = tmp_898; values[20] = tmp_899; } let tmp_901 = smem_keys[tmp_837 * WPT + 21u]; let tmp_902 = smem_vals[tmp_837 * WPT + 21u]; let tmp_903 = keys[21] < tmp_901 || (keys[21] == tmp_901 && values[21] < tmp_902); if tmp_836 == tmp_903 { keys[21] = tmp_901; values[21] = tmp_902; } let tmp_904 = smem_keys[tmp_837 * WPT + 22u]; let tmp_905 = smem_vals[tmp_837 * WPT + 22u]; let tmp_906 = keys[22] < tmp_904 || (keys[22] == tmp_904 && values[22] < tmp_905); if tmp_836 == tmp_906 { keys[22] = tmp_904; values[22] = tmp_905; } let tmp_907 = smem_keys[tmp_837 * WPT + 23u]; let tmp_908 = smem_vals[tmp_837 * WPT + 23u]; let tmp_909 = keys[23] < tmp_907 || (keys[23] == tmp_907 && values[23] < tmp_908); if tmp_836 == tmp_909 { keys[23] = tmp_907; values[23] = tmp_908; } let tmp_910 = smem_keys[tmp_837 * WPT + 24u]; let tmp_911 = smem_vals[tmp_837 * WPT + 24u]; let tmp_912 = keys[24] < tmp_910 || (keys[24] == tmp_910 && values[24] < tmp_911); if tmp_836 == tmp_912 { keys[24] = tmp_910; values[24] = tmp_911; } let tmp_913 = smem_keys[tmp_837 * WPT + 25u]; let tmp_914 = smem_vals[tmp_837 * WPT + 25u]; let tmp_915 = keys[25] < tmp_913 || (keys[25] == tmp_913 && values[25] < tmp_914); if tmp_836 == tmp_915 { keys[25] = tmp_913; values[25] = tmp_914; } let tmp_916 = smem_keys[tmp_837 * WPT + 26u]; let tmp_917 = smem_vals[tmp_837 * WPT + 26u]; let tmp_918 = keys[26] < tmp_916 || (keys[26] == tmp_916 && values[26] < tmp_917); if tmp_836 == tmp_918 { keys[26] = tmp_916; values[26] = tmp_917; } let tmp_919 = smem_keys[tmp_837 * WPT + 27u]; let tmp_920 = smem_vals[tmp_837 * WPT + 27u]; let tmp_921 = keys[27] < tmp_919 || (keys[27] == tmp_919 && values[27] < tmp_920); if tmp_836 == tmp_921 { keys[27] = tmp_919; values[27] = tmp_920; } let tmp_922 = smem_keys[tmp_837 * WPT + 28u]; let tmp_923 = smem_vals[tmp_837 * WPT + 28u]; let tmp_924 = keys[28] < tmp_922 || (keys[28] == tmp_922 && values[28] < tmp_923); if tmp_836 == tmp_924 { keys[28] = tmp_922; values[28] = tmp_923; } let tmp_925 = smem_keys[tmp_837 * WPT + 29u]; let tmp_926 = smem_vals[tmp_837 * WPT + 29u]; let tmp_927 = keys[29] < tmp_925 || (keys[29] == tmp_925 && values[29] < tmp_926); if tmp_836 == tmp_927 { keys[29] = tmp_925; values[29] = tmp_926; } let tmp_928 = smem_keys[tmp_837 * WPT + 30u]; let tmp_929 = smem_vals[tmp_837 * WPT + 30u]; let tmp_930 = keys[30] < tmp_928 || (keys[30] == tmp_928 && values[30] < tmp_929); if tmp_836 == tmp_930 { keys[30] = tmp_928; values[30] = tmp_929; } let tmp_931 = smem_keys[tmp_837 * WPT + 31u]; let tmp_932 = smem_vals[tmp_837 * WPT + 31u]; let tmp_933 = keys[31] < tmp_931 || (keys[31] == tmp_931 && values[31] < tmp_932); if tmp_836 == tmp_933 { keys[31] = tmp_931; values[31] = tmp_932; } workgroupBarrier(); }
    // exch_local(16,32) 
    // cmp_swap(0,16)
    if keys[0] > keys[16] || (keys[0] == keys[16] && values[0] > values[16]) {
    // swap(0,16) 
    { let tmp_934 = keys[0]; keys[0] = keys[16]; keys[16] = tmp_934;let tmp_935 = values[0]; values[0] = values[16]; values[16] = tmp_935; }
    }
    // cmp_swap(1,17)
    if keys[1] > keys[17] || (keys[1] == keys[17] && values[1] > values[17]) {
    // swap(1,17) 
    { let tmp_936 = keys[1]; keys[1] = keys[17]; keys[17] = tmp_936;let tmp_937 = values[1]; values[1] = values[17]; values[17] = tmp_937; }
    }
    // cmp_swap(2,18)
    if keys[2] > keys[18] || (keys[2] == keys[18] && values[2] > values[18]) {
    // swap(2,18) 
    { let tmp_938 = keys[2]; keys[2] = keys[18]; keys[18] = tmp_938;let tmp_939 = values[2]; values[2] = values[18]; values[18] = tmp_939; }
    }
    // cmp_swap(3,19)
    if keys[3] > keys[19] || (keys[3] == keys[19] && values[3] > values[19]) {
    // swap(3,19) 
    { let tmp_940 = keys[3]; keys[3] = keys[19]; keys[19] = tmp_940;let tmp_941 = values[3]; values[3] = values[19]; values[19] = tmp_941; }
    }
    // cmp_swap(4,20)
    if keys[4] > keys[20] || (keys[4] == keys[20] && values[4] > values[20]) {
    // swap(4,20) 
    { let tmp_942 = keys[4]; keys[4] = keys[20]; keys[20] = tmp_942;let tmp_943 = values[4]; values[4] = values[20]; values[20] = tmp_943; }
    }
    // cmp_swap(5,21)
    if keys[5] > keys[21] || (keys[5] == keys[21] && values[5] > values[21]) {
    // swap(5,21) 
    { let tmp_944 = keys[5]; keys[5] = keys[21]; keys[21] = tmp_944;let tmp_945 = values[5]; values[5] = values[21]; values[21] = tmp_945; }
    }
    // cmp_swap(6,22)
    if keys[6] > keys[22] || (keys[6] == keys[22] && values[6] > values[22]) {
    // swap(6,22) 
    { let tmp_946 = keys[6]; keys[6] = keys[22]; keys[22] = tmp_946;let tmp_947 = values[6]; values[6] = values[22]; values[22] = tmp_947; }
    }
    // cmp_swap(7,23)
    if keys[7] > keys[23] || (keys[7] == keys[23] && values[7] > values[23]) {
    // swap(7,23) 
    { let tmp_948 = keys[7]; keys[7] = keys[23]; keys[23] = tmp_948;let tmp_949 = values[7]; values[7] = values[23]; values[23] = tmp_949; }
    }
    // cmp_swap(8,24)
    if keys[8] > keys[24] || (keys[8] == keys[24] && values[8] > values[24]) {
    // swap(8,24) 
    { let tmp_950 = keys[8]; keys[8] = keys[24]; keys[24] = tmp_950;let tmp_951 = values[8]; values[8] = values[24]; values[24] = tmp_951; }
    }
    // cmp_swap(9,25)
    if keys[9] > keys[25] || (keys[9] == keys[25] && values[9] > values[25]) {
    // swap(9,25) 
    { let tmp_952 = keys[9]; keys[9] = keys[25]; keys[25] = tmp_952;let tmp_953 = values[9]; values[9] = values[25]; values[25] = tmp_953; }
    }
    // cmp_swap(10,26)
    if keys[10] > keys[26] || (keys[10] == keys[26] && values[10] > values[26]) {
    // swap(10,26) 
    { let tmp_954 = keys[10]; keys[10] = keys[26]; keys[26] = tmp_954;let tmp_955 = values[10]; values[10] = values[26]; values[26] = tmp_955; }
    }
    // cmp_swap(11,27)
    if keys[11] > keys[27] || (keys[11] == keys[27] && values[11] > values[27]) {
    // swap(11,27) 
    { let tmp_956 = keys[11]; keys[11] = keys[27]; keys[27] = tmp_956;let tmp_957 = values[11]; values[11] = values[27]; values[27] = tmp_957; }
    }
    // cmp_swap(12,28)
    if keys[12] > keys[28] || (keys[12] == keys[28] && values[12] > values[28]) {
    // swap(12,28) 
    { let tmp_958 = keys[12]; keys[12] = keys[28]; keys[28] = tmp_958;let tmp_959 = values[12]; values[12] = values[28]; values[28] = tmp_959; }
    }
    // cmp_swap(13,29)
    if keys[13] > keys[29] || (keys[13] == keys[29] && values[13] > values[29]) {
    // swap(13,29) 
    { let tmp_960 = keys[13]; keys[13] = keys[29]; keys[29] = tmp_960;let tmp_961 = values[13]; values[13] = values[29]; values[29] = tmp_961; }
    }
    // cmp_swap(14,30)
    if keys[14] > keys[30] || (keys[14] == keys[30] && values[14] > values[30]) {
    // swap(14,30) 
    { let tmp_962 = keys[14]; keys[14] = keys[30]; keys[30] = tmp_962;let tmp_963 = values[14]; values[14] = values[30]; values[30] = tmp_963; }
    }
    // cmp_swap(15,31)
    if keys[15] > keys[31] || (keys[15] == keys[31] && values[15] > values[31]) {
    // swap(15,31) 
    { let tmp_964 = keys[15]; keys[15] = keys[31]; keys[31] = tmp_964;let tmp_965 = values[15]; values[15] = values[31]; values[31] = tmp_965; }
    }
    // exch_local(8,32) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_966 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_966;let tmp_967 = values[0]; values[0] = values[8]; values[8] = tmp_967; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_968 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_968;let tmp_969 = values[1]; values[1] = values[9]; values[9] = tmp_969; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_970 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_970;let tmp_971 = values[2]; values[2] = values[10]; values[10] = tmp_971; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_972 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_972;let tmp_973 = values[3]; values[3] = values[11]; values[11] = tmp_973; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_974 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_974;let tmp_975 = values[4]; values[4] = values[12]; values[12] = tmp_975; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_976 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_976;let tmp_977 = values[5]; values[5] = values[13]; values[13] = tmp_977; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_978 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_978;let tmp_979 = values[6]; values[6] = values[14]; values[14] = tmp_979; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_980 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_980;let tmp_981 = values[7]; values[7] = values[15]; values[15] = tmp_981; }
    }
    // cmp_swap(16,24)
    if keys[16] > keys[24] || (keys[16] == keys[24] && values[16] > values[24]) {
    // swap(16,24) 
    { let tmp_982 = keys[16]; keys[16] = keys[24]; keys[24] = tmp_982;let tmp_983 = values[16]; values[16] = values[24]; values[24] = tmp_983; }
    }
    // cmp_swap(17,25)
    if keys[17] > keys[25] || (keys[17] == keys[25] && values[17] > values[25]) {
    // swap(17,25) 
    { let tmp_984 = keys[17]; keys[17] = keys[25]; keys[25] = tmp_984;let tmp_985 = values[17]; values[17] = values[25]; values[25] = tmp_985; }
    }
    // cmp_swap(18,26)
    if keys[18] > keys[26] || (keys[18] == keys[26] && values[18] > values[26]) {
    // swap(18,26) 
    { let tmp_986 = keys[18]; keys[18] = keys[26]; keys[26] = tmp_986;let tmp_987 = values[18]; values[18] = values[26]; values[26] = tmp_987; }
    }
    // cmp_swap(19,27)
    if keys[19] > keys[27] || (keys[19] == keys[27] && values[19] > values[27]) {
    // swap(19,27) 
    { let tmp_988 = keys[19]; keys[19] = keys[27]; keys[27] = tmp_988;let tmp_989 = values[19]; values[19] = values[27]; values[27] = tmp_989; }
    }
    // cmp_swap(20,28)
    if keys[20] > keys[28] || (keys[20] == keys[28] && values[20] > values[28]) {
    // swap(20,28) 
    { let tmp_990 = keys[20]; keys[20] = keys[28]; keys[28] = tmp_990;let tmp_991 = values[20]; values[20] = values[28]; values[28] = tmp_991; }
    }
    // cmp_swap(21,29)
    if keys[21] > keys[29] || (keys[21] == keys[29] && values[21] > values[29]) {
    // swap(21,29) 
    { let tmp_992 = keys[21]; keys[21] = keys[29]; keys[29] = tmp_992;let tmp_993 = values[21]; values[21] = values[29]; values[29] = tmp_993; }
    }
    // cmp_swap(22,30)
    if keys[22] > keys[30] || (keys[22] == keys[30] && values[22] > values[30]) {
    // swap(22,30) 
    { let tmp_994 = keys[22]; keys[22] = keys[30]; keys[30] = tmp_994;let tmp_995 = values[22]; values[22] = values[30]; values[30] = tmp_995; }
    }
    // cmp_swap(23,31)
    if keys[23] > keys[31] || (keys[23] == keys[31] && values[23] > values[31]) {
    // swap(23,31) 
    { let tmp_996 = keys[23]; keys[23] = keys[31]; keys[31] = tmp_996;let tmp_997 = values[23]; values[23] = values[31]; values[31] = tmp_997; }
    }
    // exch_local(4,32) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_998 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_998;let tmp_999 = values[0]; values[0] = values[4]; values[4] = tmp_999; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_1000 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_1000;let tmp_1001 = values[1]; values[1] = values[5]; values[5] = tmp_1001; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_1002 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_1002;let tmp_1003 = values[2]; values[2] = values[6]; values[6] = tmp_1003; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_1004 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_1004;let tmp_1005 = values[3]; values[3] = values[7]; values[7] = tmp_1005; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_1006 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_1006;let tmp_1007 = values[8]; values[8] = values[12]; values[12] = tmp_1007; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_1008 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_1008;let tmp_1009 = values[9]; values[9] = values[13]; values[13] = tmp_1009; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_1010 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_1010;let tmp_1011 = values[10]; values[10] = values[14]; values[14] = tmp_1011; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_1012 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_1012;let tmp_1013 = values[11]; values[11] = values[15]; values[15] = tmp_1013; }
    }
    // cmp_swap(16,20)
    if keys[16] > keys[20] || (keys[16] == keys[20] && values[16] > values[20]) {
    // swap(16,20) 
    { let tmp_1014 = keys[16]; keys[16] = keys[20]; keys[20] = tmp_1014;let tmp_1015 = values[16]; values[16] = values[20]; values[20] = tmp_1015; }
    }
    // cmp_swap(17,21)
    if keys[17] > keys[21] || (keys[17] == keys[21] && values[17] > values[21]) {
    // swap(17,21) 
    { let tmp_1016 = keys[17]; keys[17] = keys[21]; keys[21] = tmp_1016;let tmp_1017 = values[17]; values[17] = values[21]; values[21] = tmp_1017; }
    }
    // cmp_swap(18,22)
    if keys[18] > keys[22] || (keys[18] == keys[22] && values[18] > values[22]) {
    // swap(18,22) 
    { let tmp_1018 = keys[18]; keys[18] = keys[22]; keys[22] = tmp_1018;let tmp_1019 = values[18]; values[18] = values[22]; values[22] = tmp_1019; }
    }
    // cmp_swap(19,23)
    if keys[19] > keys[23] || (keys[19] == keys[23] && values[19] > values[23]) {
    // swap(19,23) 
    { let tmp_1020 = keys[19]; keys[19] = keys[23]; keys[23] = tmp_1020;let tmp_1021 = values[19]; values[19] = values[23]; values[23] = tmp_1021; }
    }
    // cmp_swap(24,28)
    if keys[24] > keys[28] || (keys[24] == keys[28] && values[24] > values[28]) {
    // swap(24,28) 
    { let tmp_1022 = keys[24]; keys[24] = keys[28]; keys[28] = tmp_1022;let tmp_1023 = values[24]; values[24] = values[28]; values[28] = tmp_1023; }
    }
    // cmp_swap(25,29)
    if keys[25] > keys[29] || (keys[25] == keys[29] && values[25] > values[29]) {
    // swap(25,29) 
    { let tmp_1024 = keys[25]; keys[25] = keys[29]; keys[29] = tmp_1024;let tmp_1025 = values[25]; values[25] = values[29]; values[29] = tmp_1025; }
    }
    // cmp_swap(26,30)
    if keys[26] > keys[30] || (keys[26] == keys[30] && values[26] > values[30]) {
    // swap(26,30) 
    { let tmp_1026 = keys[26]; keys[26] = keys[30]; keys[30] = tmp_1026;let tmp_1027 = values[26]; values[26] = values[30]; values[30] = tmp_1027; }
    }
    // cmp_swap(27,31)
    if keys[27] > keys[31] || (keys[27] == keys[31] && values[27] > values[31]) {
    // swap(27,31) 
    { let tmp_1028 = keys[27]; keys[27] = keys[31]; keys[31] = tmp_1028;let tmp_1029 = values[27]; values[27] = values[31]; values[31] = tmp_1029; }
    }
    // exch_local(2,32) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_1030 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_1030;let tmp_1031 = values[0]; values[0] = values[2]; values[2] = tmp_1031; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_1032 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_1032;let tmp_1033 = values[1]; values[1] = values[3]; values[3] = tmp_1033; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_1034 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_1034;let tmp_1035 = values[4]; values[4] = values[6]; values[6] = tmp_1035; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_1036 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_1036;let tmp_1037 = values[5]; values[5] = values[7]; values[7] = tmp_1037; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_1038 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_1038;let tmp_1039 = values[8]; values[8] = values[10]; values[10] = tmp_1039; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_1040 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_1040;let tmp_1041 = values[9]; values[9] = values[11]; values[11] = tmp_1041; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_1042 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_1042;let tmp_1043 = values[12]; values[12] = values[14]; values[14] = tmp_1043; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_1044 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_1044;let tmp_1045 = values[13]; values[13] = values[15]; values[15] = tmp_1045; }
    }
    // cmp_swap(16,18)
    if keys[16] > keys[18] || (keys[16] == keys[18] && values[16] > values[18]) {
    // swap(16,18) 
    { let tmp_1046 = keys[16]; keys[16] = keys[18]; keys[18] = tmp_1046;let tmp_1047 = values[16]; values[16] = values[18]; values[18] = tmp_1047; }
    }
    // cmp_swap(17,19)
    if keys[17] > keys[19] || (keys[17] == keys[19] && values[17] > values[19]) {
    // swap(17,19) 
    { let tmp_1048 = keys[17]; keys[17] = keys[19]; keys[19] = tmp_1048;let tmp_1049 = values[17]; values[17] = values[19]; values[19] = tmp_1049; }
    }
    // cmp_swap(20,22)
    if keys[20] > keys[22] || (keys[20] == keys[22] && values[20] > values[22]) {
    // swap(20,22) 
    { let tmp_1050 = keys[20]; keys[20] = keys[22]; keys[22] = tmp_1050;let tmp_1051 = values[20]; values[20] = values[22]; values[22] = tmp_1051; }
    }
    // cmp_swap(21,23)
    if keys[21] > keys[23] || (keys[21] == keys[23] && values[21] > values[23]) {
    // swap(21,23) 
    { let tmp_1052 = keys[21]; keys[21] = keys[23]; keys[23] = tmp_1052;let tmp_1053 = values[21]; values[21] = values[23]; values[23] = tmp_1053; }
    }
    // cmp_swap(24,26)
    if keys[24] > keys[26] || (keys[24] == keys[26] && values[24] > values[26]) {
    // swap(24,26) 
    { let tmp_1054 = keys[24]; keys[24] = keys[26]; keys[26] = tmp_1054;let tmp_1055 = values[24]; values[24] = values[26]; values[26] = tmp_1055; }
    }
    // cmp_swap(25,27)
    if keys[25] > keys[27] || (keys[25] == keys[27] && values[25] > values[27]) {
    // swap(25,27) 
    { let tmp_1056 = keys[25]; keys[25] = keys[27]; keys[27] = tmp_1056;let tmp_1057 = values[25]; values[25] = values[27]; values[27] = tmp_1057; }
    }
    // cmp_swap(28,30)
    if keys[28] > keys[30] || (keys[28] == keys[30] && values[28] > values[30]) {
    // swap(28,30) 
    { let tmp_1058 = keys[28]; keys[28] = keys[30]; keys[30] = tmp_1058;let tmp_1059 = values[28]; values[28] = values[30]; values[30] = tmp_1059; }
    }
    // cmp_swap(29,31)
    if keys[29] > keys[31] || (keys[29] == keys[31] && values[29] > values[31]) {
    // swap(29,31) 
    { let tmp_1060 = keys[29]; keys[29] = keys[31]; keys[31] = tmp_1060;let tmp_1061 = values[29]; values[29] = values[31]; values[31] = tmp_1061; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_1062 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_1062;let tmp_1063 = values[0]; values[0] = values[1]; values[1] = tmp_1063; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_1064 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_1064;let tmp_1065 = values[2]; values[2] = values[3]; values[3] = tmp_1065; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_1066 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_1066;let tmp_1067 = values[4]; values[4] = values[5]; values[5] = tmp_1067; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_1068 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_1068;let tmp_1069 = values[6]; values[6] = values[7]; values[7] = tmp_1069; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_1070 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_1070;let tmp_1071 = values[8]; values[8] = values[9]; values[9] = tmp_1071; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_1072 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_1072;let tmp_1073 = values[10]; values[10] = values[11]; values[11] = tmp_1073; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_1074 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_1074;let tmp_1075 = values[12]; values[12] = values[13]; values[13] = tmp_1075; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_1076 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_1076;let tmp_1077 = values[14]; values[14] = values[15]; values[15] = tmp_1077; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_1078 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_1078;let tmp_1079 = values[16]; values[16] = values[17]; values[17] = tmp_1079; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_1080 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_1080;let tmp_1081 = values[18]; values[18] = values[19]; values[19] = tmp_1081; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_1082 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_1082;let tmp_1083 = values[20]; values[20] = values[21]; values[21] = tmp_1083; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_1084 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_1084;let tmp_1085 = values[22]; values[22] = values[23]; values[23] = tmp_1085; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_1086 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_1086;let tmp_1087 = values[24]; values[24] = values[25]; values[25] = tmp_1087; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_1088 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_1088;let tmp_1089 = values[26]; values[26] = values[27]; values[27] = tmp_1089; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_1090 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_1090;let tmp_1091 = values[28]; values[28] = values[29]; values[29] = tmp_1091; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_1092 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_1092;let tmp_1093 = values[30]; values[30] = values[31]; values[31] = tmp_1093; }
    }
    // exch_intxn(tmask:7,swbit:2,wpt:32)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_1094 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_1095 = seg_base + (local_tid ^ 7u); let tmp_1096 = smem_keys[tmp_1095 * WPT + 31u]; let tmp_1097 = smem_vals[tmp_1095 * WPT + 31u]; let tmp_1098 = keys[0] < tmp_1096 || (keys[0] == tmp_1096 && values[0] < tmp_1097); if tmp_1094 == tmp_1098 { keys[0] = tmp_1096; values[0] = tmp_1097; } let tmp_1099 = smem_keys[tmp_1095 * WPT + 30u]; let tmp_1100 = smem_vals[tmp_1095 * WPT + 30u]; let tmp_1101 = keys[1] < tmp_1099 || (keys[1] == tmp_1099 && values[1] < tmp_1100); if tmp_1094 == tmp_1101 { keys[1] = tmp_1099; values[1] = tmp_1100; } let tmp_1102 = smem_keys[tmp_1095 * WPT + 29u]; let tmp_1103 = smem_vals[tmp_1095 * WPT + 29u]; let tmp_1104 = keys[2] < tmp_1102 || (keys[2] == tmp_1102 && values[2] < tmp_1103); if tmp_1094 == tmp_1104 { keys[2] = tmp_1102; values[2] = tmp_1103; } let tmp_1105 = smem_keys[tmp_1095 * WPT + 28u]; let tmp_1106 = smem_vals[tmp_1095 * WPT + 28u]; let tmp_1107 = keys[3] < tmp_1105 || (keys[3] == tmp_1105 && values[3] < tmp_1106); if tmp_1094 == tmp_1107 { keys[3] = tmp_1105; values[3] = tmp_1106; } let tmp_1108 = smem_keys[tmp_1095 * WPT + 27u]; let tmp_1109 = smem_vals[tmp_1095 * WPT + 27u]; let tmp_1110 = keys[4] < tmp_1108 || (keys[4] == tmp_1108 && values[4] < tmp_1109); if tmp_1094 == tmp_1110 { keys[4] = tmp_1108; values[4] = tmp_1109; } let tmp_1111 = smem_keys[tmp_1095 * WPT + 26u]; let tmp_1112 = smem_vals[tmp_1095 * WPT + 26u]; let tmp_1113 = keys[5] < tmp_1111 || (keys[5] == tmp_1111 && values[5] < tmp_1112); if tmp_1094 == tmp_1113 { keys[5] = tmp_1111; values[5] = tmp_1112; } let tmp_1114 = smem_keys[tmp_1095 * WPT + 25u]; let tmp_1115 = smem_vals[tmp_1095 * WPT + 25u]; let tmp_1116 = keys[6] < tmp_1114 || (keys[6] == tmp_1114 && values[6] < tmp_1115); if tmp_1094 == tmp_1116 { keys[6] = tmp_1114; values[6] = tmp_1115; } let tmp_1117 = smem_keys[tmp_1095 * WPT + 24u]; let tmp_1118 = smem_vals[tmp_1095 * WPT + 24u]; let tmp_1119 = keys[7] < tmp_1117 || (keys[7] == tmp_1117 && values[7] < tmp_1118); if tmp_1094 == tmp_1119 { keys[7] = tmp_1117; values[7] = tmp_1118; } let tmp_1120 = smem_keys[tmp_1095 * WPT + 23u]; let tmp_1121 = smem_vals[tmp_1095 * WPT + 23u]; let tmp_1122 = keys[8] < tmp_1120 || (keys[8] == tmp_1120 && values[8] < tmp_1121); if tmp_1094 == tmp_1122 { keys[8] = tmp_1120; values[8] = tmp_1121; } let tmp_1123 = smem_keys[tmp_1095 * WPT + 22u]; let tmp_1124 = smem_vals[tmp_1095 * WPT + 22u]; let tmp_1125 = keys[9] < tmp_1123 || (keys[9] == tmp_1123 && values[9] < tmp_1124); if tmp_1094 == tmp_1125 { keys[9] = tmp_1123; values[9] = tmp_1124; } let tmp_1126 = smem_keys[tmp_1095 * WPT + 21u]; let tmp_1127 = smem_vals[tmp_1095 * WPT + 21u]; let tmp_1128 = keys[10] < tmp_1126 || (keys[10] == tmp_1126 && values[10] < tmp_1127); if tmp_1094 == tmp_1128 { keys[10] = tmp_1126; values[10] = tmp_1127; } let tmp_1129 = smem_keys[tmp_1095 * WPT + 20u]; let tmp_1130 = smem_vals[tmp_1095 * WPT + 20u]; let tmp_1131 = keys[11] < tmp_1129 || (keys[11] == tmp_1129 && values[11] < tmp_1130); if tmp_1094 == tmp_1131 { keys[11] = tmp_1129; values[11] = tmp_1130; } let tmp_1132 = smem_keys[tmp_1095 * WPT + 19u]; let tmp_1133 = smem_vals[tmp_1095 * WPT + 19u]; let tmp_1134 = keys[12] < tmp_1132 || (keys[12] == tmp_1132 && values[12] < tmp_1133); if tmp_1094 == tmp_1134 { keys[12] = tmp_1132; values[12] = tmp_1133; } let tmp_1135 = smem_keys[tmp_1095 * WPT + 18u]; let tmp_1136 = smem_vals[tmp_1095 * WPT + 18u]; let tmp_1137 = keys[13] < tmp_1135 || (keys[13] == tmp_1135 && values[13] < tmp_1136); if tmp_1094 == tmp_1137 { keys[13] = tmp_1135; values[13] = tmp_1136; } let tmp_1138 = smem_keys[tmp_1095 * WPT + 17u]; let tmp_1139 = smem_vals[tmp_1095 * WPT + 17u]; let tmp_1140 = keys[14] < tmp_1138 || (keys[14] == tmp_1138 && values[14] < tmp_1139); if tmp_1094 == tmp_1140 { keys[14] = tmp_1138; values[14] = tmp_1139; } let tmp_1141 = smem_keys[tmp_1095 * WPT + 16u]; let tmp_1142 = smem_vals[tmp_1095 * WPT + 16u]; let tmp_1143 = keys[15] < tmp_1141 || (keys[15] == tmp_1141 && values[15] < tmp_1142); if tmp_1094 == tmp_1143 { keys[15] = tmp_1141; values[15] = tmp_1142; } let tmp_1144 = smem_keys[tmp_1095 * WPT + 15u]; let tmp_1145 = smem_vals[tmp_1095 * WPT + 15u]; let tmp_1146 = keys[16] < tmp_1144 || (keys[16] == tmp_1144 && values[16] < tmp_1145); if tmp_1094 == tmp_1146 { keys[16] = tmp_1144; values[16] = tmp_1145; } let tmp_1147 = smem_keys[tmp_1095 * WPT + 14u]; let tmp_1148 = smem_vals[tmp_1095 * WPT + 14u]; let tmp_1149 = keys[17] < tmp_1147 || (keys[17] == tmp_1147 && values[17] < tmp_1148); if tmp_1094 == tmp_1149 { keys[17] = tmp_1147; values[17] = tmp_1148; } let tmp_1150 = smem_keys[tmp_1095 * WPT + 13u]; let tmp_1151 = smem_vals[tmp_1095 * WPT + 13u]; let tmp_1152 = keys[18] < tmp_1150 || (keys[18] == tmp_1150 && values[18] < tmp_1151); if tmp_1094 == tmp_1152 { keys[18] = tmp_1150; values[18] = tmp_1151; } let tmp_1153 = smem_keys[tmp_1095 * WPT + 12u]; let tmp_1154 = smem_vals[tmp_1095 * WPT + 12u]; let tmp_1155 = keys[19] < tmp_1153 || (keys[19] == tmp_1153 && values[19] < tmp_1154); if tmp_1094 == tmp_1155 { keys[19] = tmp_1153; values[19] = tmp_1154; } let tmp_1156 = smem_keys[tmp_1095 * WPT + 11u]; let tmp_1157 = smem_vals[tmp_1095 * WPT + 11u]; let tmp_1158 = keys[20] < tmp_1156 || (keys[20] == tmp_1156 && values[20] < tmp_1157); if tmp_1094 == tmp_1158 { keys[20] = tmp_1156; values[20] = tmp_1157; } let tmp_1159 = smem_keys[tmp_1095 * WPT + 10u]; let tmp_1160 = smem_vals[tmp_1095 * WPT + 10u]; let tmp_1161 = keys[21] < tmp_1159 || (keys[21] == tmp_1159 && values[21] < tmp_1160); if tmp_1094 == tmp_1161 { keys[21] = tmp_1159; values[21] = tmp_1160; } let tmp_1162 = smem_keys[tmp_1095 * WPT + 9u]; let tmp_1163 = smem_vals[tmp_1095 * WPT + 9u]; let tmp_1164 = keys[22] < tmp_1162 || (keys[22] == tmp_1162 && values[22] < tmp_1163); if tmp_1094 == tmp_1164 { keys[22] = tmp_1162; values[22] = tmp_1163; } let tmp_1165 = smem_keys[tmp_1095 * WPT + 8u]; let tmp_1166 = smem_vals[tmp_1095 * WPT + 8u]; let tmp_1167 = keys[23] < tmp_1165 || (keys[23] == tmp_1165 && values[23] < tmp_1166); if tmp_1094 == tmp_1167 { keys[23] = tmp_1165; values[23] = tmp_1166; } let tmp_1168 = smem_keys[tmp_1095 * WPT + 7u]; let tmp_1169 = smem_vals[tmp_1095 * WPT + 7u]; let tmp_1170 = keys[24] < tmp_1168 || (keys[24] == tmp_1168 && values[24] < tmp_1169); if tmp_1094 == tmp_1170 { keys[24] = tmp_1168; values[24] = tmp_1169; } let tmp_1171 = smem_keys[tmp_1095 * WPT + 6u]; let tmp_1172 = smem_vals[tmp_1095 * WPT + 6u]; let tmp_1173 = keys[25] < tmp_1171 || (keys[25] == tmp_1171 && values[25] < tmp_1172); if tmp_1094 == tmp_1173 { keys[25] = tmp_1171; values[25] = tmp_1172; } let tmp_1174 = smem_keys[tmp_1095 * WPT + 5u]; let tmp_1175 = smem_vals[tmp_1095 * WPT + 5u]; let tmp_1176 = keys[26] < tmp_1174 || (keys[26] == tmp_1174 && values[26] < tmp_1175); if tmp_1094 == tmp_1176 { keys[26] = tmp_1174; values[26] = tmp_1175; } let tmp_1177 = smem_keys[tmp_1095 * WPT + 4u]; let tmp_1178 = smem_vals[tmp_1095 * WPT + 4u]; let tmp_1179 = keys[27] < tmp_1177 || (keys[27] == tmp_1177 && values[27] < tmp_1178); if tmp_1094 == tmp_1179 { keys[27] = tmp_1177; values[27] = tmp_1178; } let tmp_1180 = smem_keys[tmp_1095 * WPT + 3u]; let tmp_1181 = smem_vals[tmp_1095 * WPT + 3u]; let tmp_1182 = keys[28] < tmp_1180 || (keys[28] == tmp_1180 && values[28] < tmp_1181); if tmp_1094 == tmp_1182 { keys[28] = tmp_1180; values[28] = tmp_1181; } let tmp_1183 = smem_keys[tmp_1095 * WPT + 2u]; let tmp_1184 = smem_vals[tmp_1095 * WPT + 2u]; let tmp_1185 = keys[29] < tmp_1183 || (keys[29] == tmp_1183 && values[29] < tmp_1184); if tmp_1094 == tmp_1185 { keys[29] = tmp_1183; values[29] = tmp_1184; } let tmp_1186 = smem_keys[tmp_1095 * WPT + 1u]; let tmp_1187 = smem_vals[tmp_1095 * WPT + 1u]; let tmp_1188 = keys[30] < tmp_1186 || (keys[30] == tmp_1186 && values[30] < tmp_1187); if tmp_1094 == tmp_1188 { keys[30] = tmp_1186; values[30] = tmp_1187; } let tmp_1189 = smem_keys[tmp_1095 * WPT + 0u]; let tmp_1190 = smem_vals[tmp_1095 * WPT + 0u]; let tmp_1191 = keys[31] < tmp_1189 || (keys[31] == tmp_1189 && values[31] < tmp_1190); if tmp_1094 == tmp_1191 { keys[31] = tmp_1189; values[31] = tmp_1190; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_1192 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_1193 = seg_base + (local_tid ^ 2u); let tmp_1194 = smem_keys[tmp_1193 * WPT + 0u]; let tmp_1195 = smem_vals[tmp_1193 * WPT + 0u]; let tmp_1196 = keys[0] < tmp_1194 || (keys[0] == tmp_1194 && values[0] < tmp_1195); if tmp_1192 == tmp_1196 { keys[0] = tmp_1194; values[0] = tmp_1195; } let tmp_1197 = smem_keys[tmp_1193 * WPT + 1u]; let tmp_1198 = smem_vals[tmp_1193 * WPT + 1u]; let tmp_1199 = keys[1] < tmp_1197 || (keys[1] == tmp_1197 && values[1] < tmp_1198); if tmp_1192 == tmp_1199 { keys[1] = tmp_1197; values[1] = tmp_1198; } let tmp_1200 = smem_keys[tmp_1193 * WPT + 2u]; let tmp_1201 = smem_vals[tmp_1193 * WPT + 2u]; let tmp_1202 = keys[2] < tmp_1200 || (keys[2] == tmp_1200 && values[2] < tmp_1201); if tmp_1192 == tmp_1202 { keys[2] = tmp_1200; values[2] = tmp_1201; } let tmp_1203 = smem_keys[tmp_1193 * WPT + 3u]; let tmp_1204 = smem_vals[tmp_1193 * WPT + 3u]; let tmp_1205 = keys[3] < tmp_1203 || (keys[3] == tmp_1203 && values[3] < tmp_1204); if tmp_1192 == tmp_1205 { keys[3] = tmp_1203; values[3] = tmp_1204; } let tmp_1206 = smem_keys[tmp_1193 * WPT + 4u]; let tmp_1207 = smem_vals[tmp_1193 * WPT + 4u]; let tmp_1208 = keys[4] < tmp_1206 || (keys[4] == tmp_1206 && values[4] < tmp_1207); if tmp_1192 == tmp_1208 { keys[4] = tmp_1206; values[4] = tmp_1207; } let tmp_1209 = smem_keys[tmp_1193 * WPT + 5u]; let tmp_1210 = smem_vals[tmp_1193 * WPT + 5u]; let tmp_1211 = keys[5] < tmp_1209 || (keys[5] == tmp_1209 && values[5] < tmp_1210); if tmp_1192 == tmp_1211 { keys[5] = tmp_1209; values[5] = tmp_1210; } let tmp_1212 = smem_keys[tmp_1193 * WPT + 6u]; let tmp_1213 = smem_vals[tmp_1193 * WPT + 6u]; let tmp_1214 = keys[6] < tmp_1212 || (keys[6] == tmp_1212 && values[6] < tmp_1213); if tmp_1192 == tmp_1214 { keys[6] = tmp_1212; values[6] = tmp_1213; } let tmp_1215 = smem_keys[tmp_1193 * WPT + 7u]; let tmp_1216 = smem_vals[tmp_1193 * WPT + 7u]; let tmp_1217 = keys[7] < tmp_1215 || (keys[7] == tmp_1215 && values[7] < tmp_1216); if tmp_1192 == tmp_1217 { keys[7] = tmp_1215; values[7] = tmp_1216; } let tmp_1218 = smem_keys[tmp_1193 * WPT + 8u]; let tmp_1219 = smem_vals[tmp_1193 * WPT + 8u]; let tmp_1220 = keys[8] < tmp_1218 || (keys[8] == tmp_1218 && values[8] < tmp_1219); if tmp_1192 == tmp_1220 { keys[8] = tmp_1218; values[8] = tmp_1219; } let tmp_1221 = smem_keys[tmp_1193 * WPT + 9u]; let tmp_1222 = smem_vals[tmp_1193 * WPT + 9u]; let tmp_1223 = keys[9] < tmp_1221 || (keys[9] == tmp_1221 && values[9] < tmp_1222); if tmp_1192 == tmp_1223 { keys[9] = tmp_1221; values[9] = tmp_1222; } let tmp_1224 = smem_keys[tmp_1193 * WPT + 10u]; let tmp_1225 = smem_vals[tmp_1193 * WPT + 10u]; let tmp_1226 = keys[10] < tmp_1224 || (keys[10] == tmp_1224 && values[10] < tmp_1225); if tmp_1192 == tmp_1226 { keys[10] = tmp_1224; values[10] = tmp_1225; } let tmp_1227 = smem_keys[tmp_1193 * WPT + 11u]; let tmp_1228 = smem_vals[tmp_1193 * WPT + 11u]; let tmp_1229 = keys[11] < tmp_1227 || (keys[11] == tmp_1227 && values[11] < tmp_1228); if tmp_1192 == tmp_1229 { keys[11] = tmp_1227; values[11] = tmp_1228; } let tmp_1230 = smem_keys[tmp_1193 * WPT + 12u]; let tmp_1231 = smem_vals[tmp_1193 * WPT + 12u]; let tmp_1232 = keys[12] < tmp_1230 || (keys[12] == tmp_1230 && values[12] < tmp_1231); if tmp_1192 == tmp_1232 { keys[12] = tmp_1230; values[12] = tmp_1231; } let tmp_1233 = smem_keys[tmp_1193 * WPT + 13u]; let tmp_1234 = smem_vals[tmp_1193 * WPT + 13u]; let tmp_1235 = keys[13] < tmp_1233 || (keys[13] == tmp_1233 && values[13] < tmp_1234); if tmp_1192 == tmp_1235 { keys[13] = tmp_1233; values[13] = tmp_1234; } let tmp_1236 = smem_keys[tmp_1193 * WPT + 14u]; let tmp_1237 = smem_vals[tmp_1193 * WPT + 14u]; let tmp_1238 = keys[14] < tmp_1236 || (keys[14] == tmp_1236 && values[14] < tmp_1237); if tmp_1192 == tmp_1238 { keys[14] = tmp_1236; values[14] = tmp_1237; } let tmp_1239 = smem_keys[tmp_1193 * WPT + 15u]; let tmp_1240 = smem_vals[tmp_1193 * WPT + 15u]; let tmp_1241 = keys[15] < tmp_1239 || (keys[15] == tmp_1239 && values[15] < tmp_1240); if tmp_1192 == tmp_1241 { keys[15] = tmp_1239; values[15] = tmp_1240; } let tmp_1242 = smem_keys[tmp_1193 * WPT + 16u]; let tmp_1243 = smem_vals[tmp_1193 * WPT + 16u]; let tmp_1244 = keys[16] < tmp_1242 || (keys[16] == tmp_1242 && values[16] < tmp_1243); if tmp_1192 == tmp_1244 { keys[16] = tmp_1242; values[16] = tmp_1243; } let tmp_1245 = smem_keys[tmp_1193 * WPT + 17u]; let tmp_1246 = smem_vals[tmp_1193 * WPT + 17u]; let tmp_1247 = keys[17] < tmp_1245 || (keys[17] == tmp_1245 && values[17] < tmp_1246); if tmp_1192 == tmp_1247 { keys[17] = tmp_1245; values[17] = tmp_1246; } let tmp_1248 = smem_keys[tmp_1193 * WPT + 18u]; let tmp_1249 = smem_vals[tmp_1193 * WPT + 18u]; let tmp_1250 = keys[18] < tmp_1248 || (keys[18] == tmp_1248 && values[18] < tmp_1249); if tmp_1192 == tmp_1250 { keys[18] = tmp_1248; values[18] = tmp_1249; } let tmp_1251 = smem_keys[tmp_1193 * WPT + 19u]; let tmp_1252 = smem_vals[tmp_1193 * WPT + 19u]; let tmp_1253 = keys[19] < tmp_1251 || (keys[19] == tmp_1251 && values[19] < tmp_1252); if tmp_1192 == tmp_1253 { keys[19] = tmp_1251; values[19] = tmp_1252; } let tmp_1254 = smem_keys[tmp_1193 * WPT + 20u]; let tmp_1255 = smem_vals[tmp_1193 * WPT + 20u]; let tmp_1256 = keys[20] < tmp_1254 || (keys[20] == tmp_1254 && values[20] < tmp_1255); if tmp_1192 == tmp_1256 { keys[20] = tmp_1254; values[20] = tmp_1255; } let tmp_1257 = smem_keys[tmp_1193 * WPT + 21u]; let tmp_1258 = smem_vals[tmp_1193 * WPT + 21u]; let tmp_1259 = keys[21] < tmp_1257 || (keys[21] == tmp_1257 && values[21] < tmp_1258); if tmp_1192 == tmp_1259 { keys[21] = tmp_1257; values[21] = tmp_1258; } let tmp_1260 = smem_keys[tmp_1193 * WPT + 22u]; let tmp_1261 = smem_vals[tmp_1193 * WPT + 22u]; let tmp_1262 = keys[22] < tmp_1260 || (keys[22] == tmp_1260 && values[22] < tmp_1261); if tmp_1192 == tmp_1262 { keys[22] = tmp_1260; values[22] = tmp_1261; } let tmp_1263 = smem_keys[tmp_1193 * WPT + 23u]; let tmp_1264 = smem_vals[tmp_1193 * WPT + 23u]; let tmp_1265 = keys[23] < tmp_1263 || (keys[23] == tmp_1263 && values[23] < tmp_1264); if tmp_1192 == tmp_1265 { keys[23] = tmp_1263; values[23] = tmp_1264; } let tmp_1266 = smem_keys[tmp_1193 * WPT + 24u]; let tmp_1267 = smem_vals[tmp_1193 * WPT + 24u]; let tmp_1268 = keys[24] < tmp_1266 || (keys[24] == tmp_1266 && values[24] < tmp_1267); if tmp_1192 == tmp_1268 { keys[24] = tmp_1266; values[24] = tmp_1267; } let tmp_1269 = smem_keys[tmp_1193 * WPT + 25u]; let tmp_1270 = smem_vals[tmp_1193 * WPT + 25u]; let tmp_1271 = keys[25] < tmp_1269 || (keys[25] == tmp_1269 && values[25] < tmp_1270); if tmp_1192 == tmp_1271 { keys[25] = tmp_1269; values[25] = tmp_1270; } let tmp_1272 = smem_keys[tmp_1193 * WPT + 26u]; let tmp_1273 = smem_vals[tmp_1193 * WPT + 26u]; let tmp_1274 = keys[26] < tmp_1272 || (keys[26] == tmp_1272 && values[26] < tmp_1273); if tmp_1192 == tmp_1274 { keys[26] = tmp_1272; values[26] = tmp_1273; } let tmp_1275 = smem_keys[tmp_1193 * WPT + 27u]; let tmp_1276 = smem_vals[tmp_1193 * WPT + 27u]; let tmp_1277 = keys[27] < tmp_1275 || (keys[27] == tmp_1275 && values[27] < tmp_1276); if tmp_1192 == tmp_1277 { keys[27] = tmp_1275; values[27] = tmp_1276; } let tmp_1278 = smem_keys[tmp_1193 * WPT + 28u]; let tmp_1279 = smem_vals[tmp_1193 * WPT + 28u]; let tmp_1280 = keys[28] < tmp_1278 || (keys[28] == tmp_1278 && values[28] < tmp_1279); if tmp_1192 == tmp_1280 { keys[28] = tmp_1278; values[28] = tmp_1279; } let tmp_1281 = smem_keys[tmp_1193 * WPT + 29u]; let tmp_1282 = smem_vals[tmp_1193 * WPT + 29u]; let tmp_1283 = keys[29] < tmp_1281 || (keys[29] == tmp_1281 && values[29] < tmp_1282); if tmp_1192 == tmp_1283 { keys[29] = tmp_1281; values[29] = tmp_1282; } let tmp_1284 = smem_keys[tmp_1193 * WPT + 30u]; let tmp_1285 = smem_vals[tmp_1193 * WPT + 30u]; let tmp_1286 = keys[30] < tmp_1284 || (keys[30] == tmp_1284 && values[30] < tmp_1285); if tmp_1192 == tmp_1286 { keys[30] = tmp_1284; values[30] = tmp_1285; } let tmp_1287 = smem_keys[tmp_1193 * WPT + 31u]; let tmp_1288 = smem_vals[tmp_1193 * WPT + 31u]; let tmp_1289 = keys[31] < tmp_1287 || (keys[31] == tmp_1287 && values[31] < tmp_1288); if tmp_1192 == tmp_1289 { keys[31] = tmp_1287; values[31] = tmp_1288; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_1290 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_1291 = seg_base + (local_tid ^ 1u); let tmp_1292 = smem_keys[tmp_1291 * WPT + 0u]; let tmp_1293 = smem_vals[tmp_1291 * WPT + 0u]; let tmp_1294 = keys[0] < tmp_1292 || (keys[0] == tmp_1292 && values[0] < tmp_1293); if tmp_1290 == tmp_1294 { keys[0] = tmp_1292; values[0] = tmp_1293; } let tmp_1295 = smem_keys[tmp_1291 * WPT + 1u]; let tmp_1296 = smem_vals[tmp_1291 * WPT + 1u]; let tmp_1297 = keys[1] < tmp_1295 || (keys[1] == tmp_1295 && values[1] < tmp_1296); if tmp_1290 == tmp_1297 { keys[1] = tmp_1295; values[1] = tmp_1296; } let tmp_1298 = smem_keys[tmp_1291 * WPT + 2u]; let tmp_1299 = smem_vals[tmp_1291 * WPT + 2u]; let tmp_1300 = keys[2] < tmp_1298 || (keys[2] == tmp_1298 && values[2] < tmp_1299); if tmp_1290 == tmp_1300 { keys[2] = tmp_1298; values[2] = tmp_1299; } let tmp_1301 = smem_keys[tmp_1291 * WPT + 3u]; let tmp_1302 = smem_vals[tmp_1291 * WPT + 3u]; let tmp_1303 = keys[3] < tmp_1301 || (keys[3] == tmp_1301 && values[3] < tmp_1302); if tmp_1290 == tmp_1303 { keys[3] = tmp_1301; values[3] = tmp_1302; } let tmp_1304 = smem_keys[tmp_1291 * WPT + 4u]; let tmp_1305 = smem_vals[tmp_1291 * WPT + 4u]; let tmp_1306 = keys[4] < tmp_1304 || (keys[4] == tmp_1304 && values[4] < tmp_1305); if tmp_1290 == tmp_1306 { keys[4] = tmp_1304; values[4] = tmp_1305; } let tmp_1307 = smem_keys[tmp_1291 * WPT + 5u]; let tmp_1308 = smem_vals[tmp_1291 * WPT + 5u]; let tmp_1309 = keys[5] < tmp_1307 || (keys[5] == tmp_1307 && values[5] < tmp_1308); if tmp_1290 == tmp_1309 { keys[5] = tmp_1307; values[5] = tmp_1308; } let tmp_1310 = smem_keys[tmp_1291 * WPT + 6u]; let tmp_1311 = smem_vals[tmp_1291 * WPT + 6u]; let tmp_1312 = keys[6] < tmp_1310 || (keys[6] == tmp_1310 && values[6] < tmp_1311); if tmp_1290 == tmp_1312 { keys[6] = tmp_1310; values[6] = tmp_1311; } let tmp_1313 = smem_keys[tmp_1291 * WPT + 7u]; let tmp_1314 = smem_vals[tmp_1291 * WPT + 7u]; let tmp_1315 = keys[7] < tmp_1313 || (keys[7] == tmp_1313 && values[7] < tmp_1314); if tmp_1290 == tmp_1315 { keys[7] = tmp_1313; values[7] = tmp_1314; } let tmp_1316 = smem_keys[tmp_1291 * WPT + 8u]; let tmp_1317 = smem_vals[tmp_1291 * WPT + 8u]; let tmp_1318 = keys[8] < tmp_1316 || (keys[8] == tmp_1316 && values[8] < tmp_1317); if tmp_1290 == tmp_1318 { keys[8] = tmp_1316; values[8] = tmp_1317; } let tmp_1319 = smem_keys[tmp_1291 * WPT + 9u]; let tmp_1320 = smem_vals[tmp_1291 * WPT + 9u]; let tmp_1321 = keys[9] < tmp_1319 || (keys[9] == tmp_1319 && values[9] < tmp_1320); if tmp_1290 == tmp_1321 { keys[9] = tmp_1319; values[9] = tmp_1320; } let tmp_1322 = smem_keys[tmp_1291 * WPT + 10u]; let tmp_1323 = smem_vals[tmp_1291 * WPT + 10u]; let tmp_1324 = keys[10] < tmp_1322 || (keys[10] == tmp_1322 && values[10] < tmp_1323); if tmp_1290 == tmp_1324 { keys[10] = tmp_1322; values[10] = tmp_1323; } let tmp_1325 = smem_keys[tmp_1291 * WPT + 11u]; let tmp_1326 = smem_vals[tmp_1291 * WPT + 11u]; let tmp_1327 = keys[11] < tmp_1325 || (keys[11] == tmp_1325 && values[11] < tmp_1326); if tmp_1290 == tmp_1327 { keys[11] = tmp_1325; values[11] = tmp_1326; } let tmp_1328 = smem_keys[tmp_1291 * WPT + 12u]; let tmp_1329 = smem_vals[tmp_1291 * WPT + 12u]; let tmp_1330 = keys[12] < tmp_1328 || (keys[12] == tmp_1328 && values[12] < tmp_1329); if tmp_1290 == tmp_1330 { keys[12] = tmp_1328; values[12] = tmp_1329; } let tmp_1331 = smem_keys[tmp_1291 * WPT + 13u]; let tmp_1332 = smem_vals[tmp_1291 * WPT + 13u]; let tmp_1333 = keys[13] < tmp_1331 || (keys[13] == tmp_1331 && values[13] < tmp_1332); if tmp_1290 == tmp_1333 { keys[13] = tmp_1331; values[13] = tmp_1332; } let tmp_1334 = smem_keys[tmp_1291 * WPT + 14u]; let tmp_1335 = smem_vals[tmp_1291 * WPT + 14u]; let tmp_1336 = keys[14] < tmp_1334 || (keys[14] == tmp_1334 && values[14] < tmp_1335); if tmp_1290 == tmp_1336 { keys[14] = tmp_1334; values[14] = tmp_1335; } let tmp_1337 = smem_keys[tmp_1291 * WPT + 15u]; let tmp_1338 = smem_vals[tmp_1291 * WPT + 15u]; let tmp_1339 = keys[15] < tmp_1337 || (keys[15] == tmp_1337 && values[15] < tmp_1338); if tmp_1290 == tmp_1339 { keys[15] = tmp_1337; values[15] = tmp_1338; } let tmp_1340 = smem_keys[tmp_1291 * WPT + 16u]; let tmp_1341 = smem_vals[tmp_1291 * WPT + 16u]; let tmp_1342 = keys[16] < tmp_1340 || (keys[16] == tmp_1340 && values[16] < tmp_1341); if tmp_1290 == tmp_1342 { keys[16] = tmp_1340; values[16] = tmp_1341; } let tmp_1343 = smem_keys[tmp_1291 * WPT + 17u]; let tmp_1344 = smem_vals[tmp_1291 * WPT + 17u]; let tmp_1345 = keys[17] < tmp_1343 || (keys[17] == tmp_1343 && values[17] < tmp_1344); if tmp_1290 == tmp_1345 { keys[17] = tmp_1343; values[17] = tmp_1344; } let tmp_1346 = smem_keys[tmp_1291 * WPT + 18u]; let tmp_1347 = smem_vals[tmp_1291 * WPT + 18u]; let tmp_1348 = keys[18] < tmp_1346 || (keys[18] == tmp_1346 && values[18] < tmp_1347); if tmp_1290 == tmp_1348 { keys[18] = tmp_1346; values[18] = tmp_1347; } let tmp_1349 = smem_keys[tmp_1291 * WPT + 19u]; let tmp_1350 = smem_vals[tmp_1291 * WPT + 19u]; let tmp_1351 = keys[19] < tmp_1349 || (keys[19] == tmp_1349 && values[19] < tmp_1350); if tmp_1290 == tmp_1351 { keys[19] = tmp_1349; values[19] = tmp_1350; } let tmp_1352 = smem_keys[tmp_1291 * WPT + 20u]; let tmp_1353 = smem_vals[tmp_1291 * WPT + 20u]; let tmp_1354 = keys[20] < tmp_1352 || (keys[20] == tmp_1352 && values[20] < tmp_1353); if tmp_1290 == tmp_1354 { keys[20] = tmp_1352; values[20] = tmp_1353; } let tmp_1355 = smem_keys[tmp_1291 * WPT + 21u]; let tmp_1356 = smem_vals[tmp_1291 * WPT + 21u]; let tmp_1357 = keys[21] < tmp_1355 || (keys[21] == tmp_1355 && values[21] < tmp_1356); if tmp_1290 == tmp_1357 { keys[21] = tmp_1355; values[21] = tmp_1356; } let tmp_1358 = smem_keys[tmp_1291 * WPT + 22u]; let tmp_1359 = smem_vals[tmp_1291 * WPT + 22u]; let tmp_1360 = keys[22] < tmp_1358 || (keys[22] == tmp_1358 && values[22] < tmp_1359); if tmp_1290 == tmp_1360 { keys[22] = tmp_1358; values[22] = tmp_1359; } let tmp_1361 = smem_keys[tmp_1291 * WPT + 23u]; let tmp_1362 = smem_vals[tmp_1291 * WPT + 23u]; let tmp_1363 = keys[23] < tmp_1361 || (keys[23] == tmp_1361 && values[23] < tmp_1362); if tmp_1290 == tmp_1363 { keys[23] = tmp_1361; values[23] = tmp_1362; } let tmp_1364 = smem_keys[tmp_1291 * WPT + 24u]; let tmp_1365 = smem_vals[tmp_1291 * WPT + 24u]; let tmp_1366 = keys[24] < tmp_1364 || (keys[24] == tmp_1364 && values[24] < tmp_1365); if tmp_1290 == tmp_1366 { keys[24] = tmp_1364; values[24] = tmp_1365; } let tmp_1367 = smem_keys[tmp_1291 * WPT + 25u]; let tmp_1368 = smem_vals[tmp_1291 * WPT + 25u]; let tmp_1369 = keys[25] < tmp_1367 || (keys[25] == tmp_1367 && values[25] < tmp_1368); if tmp_1290 == tmp_1369 { keys[25] = tmp_1367; values[25] = tmp_1368; } let tmp_1370 = smem_keys[tmp_1291 * WPT + 26u]; let tmp_1371 = smem_vals[tmp_1291 * WPT + 26u]; let tmp_1372 = keys[26] < tmp_1370 || (keys[26] == tmp_1370 && values[26] < tmp_1371); if tmp_1290 == tmp_1372 { keys[26] = tmp_1370; values[26] = tmp_1371; } let tmp_1373 = smem_keys[tmp_1291 * WPT + 27u]; let tmp_1374 = smem_vals[tmp_1291 * WPT + 27u]; let tmp_1375 = keys[27] < tmp_1373 || (keys[27] == tmp_1373 && values[27] < tmp_1374); if tmp_1290 == tmp_1375 { keys[27] = tmp_1373; values[27] = tmp_1374; } let tmp_1376 = smem_keys[tmp_1291 * WPT + 28u]; let tmp_1377 = smem_vals[tmp_1291 * WPT + 28u]; let tmp_1378 = keys[28] < tmp_1376 || (keys[28] == tmp_1376 && values[28] < tmp_1377); if tmp_1290 == tmp_1378 { keys[28] = tmp_1376; values[28] = tmp_1377; } let tmp_1379 = smem_keys[tmp_1291 * WPT + 29u]; let tmp_1380 = smem_vals[tmp_1291 * WPT + 29u]; let tmp_1381 = keys[29] < tmp_1379 || (keys[29] == tmp_1379 && values[29] < tmp_1380); if tmp_1290 == tmp_1381 { keys[29] = tmp_1379; values[29] = tmp_1380; } let tmp_1382 = smem_keys[tmp_1291 * WPT + 30u]; let tmp_1383 = smem_vals[tmp_1291 * WPT + 30u]; let tmp_1384 = keys[30] < tmp_1382 || (keys[30] == tmp_1382 && values[30] < tmp_1383); if tmp_1290 == tmp_1384 { keys[30] = tmp_1382; values[30] = tmp_1383; } let tmp_1385 = smem_keys[tmp_1291 * WPT + 31u]; let tmp_1386 = smem_vals[tmp_1291 * WPT + 31u]; let tmp_1387 = keys[31] < tmp_1385 || (keys[31] == tmp_1385 && values[31] < tmp_1386); if tmp_1290 == tmp_1387 { keys[31] = tmp_1385; values[31] = tmp_1386; } workgroupBarrier(); }
    // exch_local(16,32) 
    // cmp_swap(0,16)
    if keys[0] > keys[16] || (keys[0] == keys[16] && values[0] > values[16]) {
    // swap(0,16) 
    { let tmp_1388 = keys[0]; keys[0] = keys[16]; keys[16] = tmp_1388;let tmp_1389 = values[0]; values[0] = values[16]; values[16] = tmp_1389; }
    }
    // cmp_swap(1,17)
    if keys[1] > keys[17] || (keys[1] == keys[17] && values[1] > values[17]) {
    // swap(1,17) 
    { let tmp_1390 = keys[1]; keys[1] = keys[17]; keys[17] = tmp_1390;let tmp_1391 = values[1]; values[1] = values[17]; values[17] = tmp_1391; }
    }
    // cmp_swap(2,18)
    if keys[2] > keys[18] || (keys[2] == keys[18] && values[2] > values[18]) {
    // swap(2,18) 
    { let tmp_1392 = keys[2]; keys[2] = keys[18]; keys[18] = tmp_1392;let tmp_1393 = values[2]; values[2] = values[18]; values[18] = tmp_1393; }
    }
    // cmp_swap(3,19)
    if keys[3] > keys[19] || (keys[3] == keys[19] && values[3] > values[19]) {
    // swap(3,19) 
    { let tmp_1394 = keys[3]; keys[3] = keys[19]; keys[19] = tmp_1394;let tmp_1395 = values[3]; values[3] = values[19]; values[19] = tmp_1395; }
    }
    // cmp_swap(4,20)
    if keys[4] > keys[20] || (keys[4] == keys[20] && values[4] > values[20]) {
    // swap(4,20) 
    { let tmp_1396 = keys[4]; keys[4] = keys[20]; keys[20] = tmp_1396;let tmp_1397 = values[4]; values[4] = values[20]; values[20] = tmp_1397; }
    }
    // cmp_swap(5,21)
    if keys[5] > keys[21] || (keys[5] == keys[21] && values[5] > values[21]) {
    // swap(5,21) 
    { let tmp_1398 = keys[5]; keys[5] = keys[21]; keys[21] = tmp_1398;let tmp_1399 = values[5]; values[5] = values[21]; values[21] = tmp_1399; }
    }
    // cmp_swap(6,22)
    if keys[6] > keys[22] || (keys[6] == keys[22] && values[6] > values[22]) {
    // swap(6,22) 
    { let tmp_1400 = keys[6]; keys[6] = keys[22]; keys[22] = tmp_1400;let tmp_1401 = values[6]; values[6] = values[22]; values[22] = tmp_1401; }
    }
    // cmp_swap(7,23)
    if keys[7] > keys[23] || (keys[7] == keys[23] && values[7] > values[23]) {
    // swap(7,23) 
    { let tmp_1402 = keys[7]; keys[7] = keys[23]; keys[23] = tmp_1402;let tmp_1403 = values[7]; values[7] = values[23]; values[23] = tmp_1403; }
    }
    // cmp_swap(8,24)
    if keys[8] > keys[24] || (keys[8] == keys[24] && values[8] > values[24]) {
    // swap(8,24) 
    { let tmp_1404 = keys[8]; keys[8] = keys[24]; keys[24] = tmp_1404;let tmp_1405 = values[8]; values[8] = values[24]; values[24] = tmp_1405; }
    }
    // cmp_swap(9,25)
    if keys[9] > keys[25] || (keys[9] == keys[25] && values[9] > values[25]) {
    // swap(9,25) 
    { let tmp_1406 = keys[9]; keys[9] = keys[25]; keys[25] = tmp_1406;let tmp_1407 = values[9]; values[9] = values[25]; values[25] = tmp_1407; }
    }
    // cmp_swap(10,26)
    if keys[10] > keys[26] || (keys[10] == keys[26] && values[10] > values[26]) {
    // swap(10,26) 
    { let tmp_1408 = keys[10]; keys[10] = keys[26]; keys[26] = tmp_1408;let tmp_1409 = values[10]; values[10] = values[26]; values[26] = tmp_1409; }
    }
    // cmp_swap(11,27)
    if keys[11] > keys[27] || (keys[11] == keys[27] && values[11] > values[27]) {
    // swap(11,27) 
    { let tmp_1410 = keys[11]; keys[11] = keys[27]; keys[27] = tmp_1410;let tmp_1411 = values[11]; values[11] = values[27]; values[27] = tmp_1411; }
    }
    // cmp_swap(12,28)
    if keys[12] > keys[28] || (keys[12] == keys[28] && values[12] > values[28]) {
    // swap(12,28) 
    { let tmp_1412 = keys[12]; keys[12] = keys[28]; keys[28] = tmp_1412;let tmp_1413 = values[12]; values[12] = values[28]; values[28] = tmp_1413; }
    }
    // cmp_swap(13,29)
    if keys[13] > keys[29] || (keys[13] == keys[29] && values[13] > values[29]) {
    // swap(13,29) 
    { let tmp_1414 = keys[13]; keys[13] = keys[29]; keys[29] = tmp_1414;let tmp_1415 = values[13]; values[13] = values[29]; values[29] = tmp_1415; }
    }
    // cmp_swap(14,30)
    if keys[14] > keys[30] || (keys[14] == keys[30] && values[14] > values[30]) {
    // swap(14,30) 
    { let tmp_1416 = keys[14]; keys[14] = keys[30]; keys[30] = tmp_1416;let tmp_1417 = values[14]; values[14] = values[30]; values[30] = tmp_1417; }
    }
    // cmp_swap(15,31)
    if keys[15] > keys[31] || (keys[15] == keys[31] && values[15] > values[31]) {
    // swap(15,31) 
    { let tmp_1418 = keys[15]; keys[15] = keys[31]; keys[31] = tmp_1418;let tmp_1419 = values[15]; values[15] = values[31]; values[31] = tmp_1419; }
    }
    // exch_local(8,32) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_1420 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_1420;let tmp_1421 = values[0]; values[0] = values[8]; values[8] = tmp_1421; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_1422 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_1422;let tmp_1423 = values[1]; values[1] = values[9]; values[9] = tmp_1423; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_1424 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_1424;let tmp_1425 = values[2]; values[2] = values[10]; values[10] = tmp_1425; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_1426 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_1426;let tmp_1427 = values[3]; values[3] = values[11]; values[11] = tmp_1427; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_1428 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_1428;let tmp_1429 = values[4]; values[4] = values[12]; values[12] = tmp_1429; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_1430 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_1430;let tmp_1431 = values[5]; values[5] = values[13]; values[13] = tmp_1431; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_1432 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_1432;let tmp_1433 = values[6]; values[6] = values[14]; values[14] = tmp_1433; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_1434 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_1434;let tmp_1435 = values[7]; values[7] = values[15]; values[15] = tmp_1435; }
    }
    // cmp_swap(16,24)
    if keys[16] > keys[24] || (keys[16] == keys[24] && values[16] > values[24]) {
    // swap(16,24) 
    { let tmp_1436 = keys[16]; keys[16] = keys[24]; keys[24] = tmp_1436;let tmp_1437 = values[16]; values[16] = values[24]; values[24] = tmp_1437; }
    }
    // cmp_swap(17,25)
    if keys[17] > keys[25] || (keys[17] == keys[25] && values[17] > values[25]) {
    // swap(17,25) 
    { let tmp_1438 = keys[17]; keys[17] = keys[25]; keys[25] = tmp_1438;let tmp_1439 = values[17]; values[17] = values[25]; values[25] = tmp_1439; }
    }
    // cmp_swap(18,26)
    if keys[18] > keys[26] || (keys[18] == keys[26] && values[18] > values[26]) {
    // swap(18,26) 
    { let tmp_1440 = keys[18]; keys[18] = keys[26]; keys[26] = tmp_1440;let tmp_1441 = values[18]; values[18] = values[26]; values[26] = tmp_1441; }
    }
    // cmp_swap(19,27)
    if keys[19] > keys[27] || (keys[19] == keys[27] && values[19] > values[27]) {
    // swap(19,27) 
    { let tmp_1442 = keys[19]; keys[19] = keys[27]; keys[27] = tmp_1442;let tmp_1443 = values[19]; values[19] = values[27]; values[27] = tmp_1443; }
    }
    // cmp_swap(20,28)
    if keys[20] > keys[28] || (keys[20] == keys[28] && values[20] > values[28]) {
    // swap(20,28) 
    { let tmp_1444 = keys[20]; keys[20] = keys[28]; keys[28] = tmp_1444;let tmp_1445 = values[20]; values[20] = values[28]; values[28] = tmp_1445; }
    }
    // cmp_swap(21,29)
    if keys[21] > keys[29] || (keys[21] == keys[29] && values[21] > values[29]) {
    // swap(21,29) 
    { let tmp_1446 = keys[21]; keys[21] = keys[29]; keys[29] = tmp_1446;let tmp_1447 = values[21]; values[21] = values[29]; values[29] = tmp_1447; }
    }
    // cmp_swap(22,30)
    if keys[22] > keys[30] || (keys[22] == keys[30] && values[22] > values[30]) {
    // swap(22,30) 
    { let tmp_1448 = keys[22]; keys[22] = keys[30]; keys[30] = tmp_1448;let tmp_1449 = values[22]; values[22] = values[30]; values[30] = tmp_1449; }
    }
    // cmp_swap(23,31)
    if keys[23] > keys[31] || (keys[23] == keys[31] && values[23] > values[31]) {
    // swap(23,31) 
    { let tmp_1450 = keys[23]; keys[23] = keys[31]; keys[31] = tmp_1450;let tmp_1451 = values[23]; values[23] = values[31]; values[31] = tmp_1451; }
    }
    // exch_local(4,32) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_1452 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_1452;let tmp_1453 = values[0]; values[0] = values[4]; values[4] = tmp_1453; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_1454 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_1454;let tmp_1455 = values[1]; values[1] = values[5]; values[5] = tmp_1455; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_1456 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_1456;let tmp_1457 = values[2]; values[2] = values[6]; values[6] = tmp_1457; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_1458 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_1458;let tmp_1459 = values[3]; values[3] = values[7]; values[7] = tmp_1459; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_1460 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_1460;let tmp_1461 = values[8]; values[8] = values[12]; values[12] = tmp_1461; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_1462 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_1462;let tmp_1463 = values[9]; values[9] = values[13]; values[13] = tmp_1463; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_1464 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_1464;let tmp_1465 = values[10]; values[10] = values[14]; values[14] = tmp_1465; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_1466 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_1466;let tmp_1467 = values[11]; values[11] = values[15]; values[15] = tmp_1467; }
    }
    // cmp_swap(16,20)
    if keys[16] > keys[20] || (keys[16] == keys[20] && values[16] > values[20]) {
    // swap(16,20) 
    { let tmp_1468 = keys[16]; keys[16] = keys[20]; keys[20] = tmp_1468;let tmp_1469 = values[16]; values[16] = values[20]; values[20] = tmp_1469; }
    }
    // cmp_swap(17,21)
    if keys[17] > keys[21] || (keys[17] == keys[21] && values[17] > values[21]) {
    // swap(17,21) 
    { let tmp_1470 = keys[17]; keys[17] = keys[21]; keys[21] = tmp_1470;let tmp_1471 = values[17]; values[17] = values[21]; values[21] = tmp_1471; }
    }
    // cmp_swap(18,22)
    if keys[18] > keys[22] || (keys[18] == keys[22] && values[18] > values[22]) {
    // swap(18,22) 
    { let tmp_1472 = keys[18]; keys[18] = keys[22]; keys[22] = tmp_1472;let tmp_1473 = values[18]; values[18] = values[22]; values[22] = tmp_1473; }
    }
    // cmp_swap(19,23)
    if keys[19] > keys[23] || (keys[19] == keys[23] && values[19] > values[23]) {
    // swap(19,23) 
    { let tmp_1474 = keys[19]; keys[19] = keys[23]; keys[23] = tmp_1474;let tmp_1475 = values[19]; values[19] = values[23]; values[23] = tmp_1475; }
    }
    // cmp_swap(24,28)
    if keys[24] > keys[28] || (keys[24] == keys[28] && values[24] > values[28]) {
    // swap(24,28) 
    { let tmp_1476 = keys[24]; keys[24] = keys[28]; keys[28] = tmp_1476;let tmp_1477 = values[24]; values[24] = values[28]; values[28] = tmp_1477; }
    }
    // cmp_swap(25,29)
    if keys[25] > keys[29] || (keys[25] == keys[29] && values[25] > values[29]) {
    // swap(25,29) 
    { let tmp_1478 = keys[25]; keys[25] = keys[29]; keys[29] = tmp_1478;let tmp_1479 = values[25]; values[25] = values[29]; values[29] = tmp_1479; }
    }
    // cmp_swap(26,30)
    if keys[26] > keys[30] || (keys[26] == keys[30] && values[26] > values[30]) {
    // swap(26,30) 
    { let tmp_1480 = keys[26]; keys[26] = keys[30]; keys[30] = tmp_1480;let tmp_1481 = values[26]; values[26] = values[30]; values[30] = tmp_1481; }
    }
    // cmp_swap(27,31)
    if keys[27] > keys[31] || (keys[27] == keys[31] && values[27] > values[31]) {
    // swap(27,31) 
    { let tmp_1482 = keys[27]; keys[27] = keys[31]; keys[31] = tmp_1482;let tmp_1483 = values[27]; values[27] = values[31]; values[31] = tmp_1483; }
    }
    // exch_local(2,32) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_1484 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_1484;let tmp_1485 = values[0]; values[0] = values[2]; values[2] = tmp_1485; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_1486 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_1486;let tmp_1487 = values[1]; values[1] = values[3]; values[3] = tmp_1487; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_1488 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_1488;let tmp_1489 = values[4]; values[4] = values[6]; values[6] = tmp_1489; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_1490 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_1490;let tmp_1491 = values[5]; values[5] = values[7]; values[7] = tmp_1491; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_1492 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_1492;let tmp_1493 = values[8]; values[8] = values[10]; values[10] = tmp_1493; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_1494 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_1494;let tmp_1495 = values[9]; values[9] = values[11]; values[11] = tmp_1495; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_1496 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_1496;let tmp_1497 = values[12]; values[12] = values[14]; values[14] = tmp_1497; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_1498 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_1498;let tmp_1499 = values[13]; values[13] = values[15]; values[15] = tmp_1499; }
    }
    // cmp_swap(16,18)
    if keys[16] > keys[18] || (keys[16] == keys[18] && values[16] > values[18]) {
    // swap(16,18) 
    { let tmp_1500 = keys[16]; keys[16] = keys[18]; keys[18] = tmp_1500;let tmp_1501 = values[16]; values[16] = values[18]; values[18] = tmp_1501; }
    }
    // cmp_swap(17,19)
    if keys[17] > keys[19] || (keys[17] == keys[19] && values[17] > values[19]) {
    // swap(17,19) 
    { let tmp_1502 = keys[17]; keys[17] = keys[19]; keys[19] = tmp_1502;let tmp_1503 = values[17]; values[17] = values[19]; values[19] = tmp_1503; }
    }
    // cmp_swap(20,22)
    if keys[20] > keys[22] || (keys[20] == keys[22] && values[20] > values[22]) {
    // swap(20,22) 
    { let tmp_1504 = keys[20]; keys[20] = keys[22]; keys[22] = tmp_1504;let tmp_1505 = values[20]; values[20] = values[22]; values[22] = tmp_1505; }
    }
    // cmp_swap(21,23)
    if keys[21] > keys[23] || (keys[21] == keys[23] && values[21] > values[23]) {
    // swap(21,23) 
    { let tmp_1506 = keys[21]; keys[21] = keys[23]; keys[23] = tmp_1506;let tmp_1507 = values[21]; values[21] = values[23]; values[23] = tmp_1507; }
    }
    // cmp_swap(24,26)
    if keys[24] > keys[26] || (keys[24] == keys[26] && values[24] > values[26]) {
    // swap(24,26) 
    { let tmp_1508 = keys[24]; keys[24] = keys[26]; keys[26] = tmp_1508;let tmp_1509 = values[24]; values[24] = values[26]; values[26] = tmp_1509; }
    }
    // cmp_swap(25,27)
    if keys[25] > keys[27] || (keys[25] == keys[27] && values[25] > values[27]) {
    // swap(25,27) 
    { let tmp_1510 = keys[25]; keys[25] = keys[27]; keys[27] = tmp_1510;let tmp_1511 = values[25]; values[25] = values[27]; values[27] = tmp_1511; }
    }
    // cmp_swap(28,30)
    if keys[28] > keys[30] || (keys[28] == keys[30] && values[28] > values[30]) {
    // swap(28,30) 
    { let tmp_1512 = keys[28]; keys[28] = keys[30]; keys[30] = tmp_1512;let tmp_1513 = values[28]; values[28] = values[30]; values[30] = tmp_1513; }
    }
    // cmp_swap(29,31)
    if keys[29] > keys[31] || (keys[29] == keys[31] && values[29] > values[31]) {
    // swap(29,31) 
    { let tmp_1514 = keys[29]; keys[29] = keys[31]; keys[31] = tmp_1514;let tmp_1515 = values[29]; values[29] = values[31]; values[31] = tmp_1515; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_1516 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_1516;let tmp_1517 = values[0]; values[0] = values[1]; values[1] = tmp_1517; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_1518 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_1518;let tmp_1519 = values[2]; values[2] = values[3]; values[3] = tmp_1519; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_1520 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_1520;let tmp_1521 = values[4]; values[4] = values[5]; values[5] = tmp_1521; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_1522 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_1522;let tmp_1523 = values[6]; values[6] = values[7]; values[7] = tmp_1523; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_1524 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_1524;let tmp_1525 = values[8]; values[8] = values[9]; values[9] = tmp_1525; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_1526 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_1526;let tmp_1527 = values[10]; values[10] = values[11]; values[11] = tmp_1527; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_1528 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_1528;let tmp_1529 = values[12]; values[12] = values[13]; values[13] = tmp_1529; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_1530 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_1530;let tmp_1531 = values[14]; values[14] = values[15]; values[15] = tmp_1531; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_1532 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_1532;let tmp_1533 = values[16]; values[16] = values[17]; values[17] = tmp_1533; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_1534 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_1534;let tmp_1535 = values[18]; values[18] = values[19]; values[19] = tmp_1535; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_1536 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_1536;let tmp_1537 = values[20]; values[20] = values[21]; values[21] = tmp_1537; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_1538 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_1538;let tmp_1539 = values[22]; values[22] = values[23]; values[23] = tmp_1539; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_1540 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_1540;let tmp_1541 = values[24]; values[24] = values[25]; values[25] = tmp_1541; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_1542 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_1542;let tmp_1543 = values[26]; values[26] = values[27]; values[27] = tmp_1543; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_1544 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_1544;let tmp_1545 = values[28]; values[28] = values[29]; values[29] = tmp_1545; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_1546 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_1546;let tmp_1547 = values[30]; values[30] = values[31]; values[31] = tmp_1547; }
    }
    // exch_intxn(tmask:15,swbit:3,wpt:32)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_1548 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_1549 = seg_base + (local_tid ^ 15u); let tmp_1550 = smem_keys[tmp_1549 * WPT + 31u]; let tmp_1551 = smem_vals[tmp_1549 * WPT + 31u]; let tmp_1552 = keys[0] < tmp_1550 || (keys[0] == tmp_1550 && values[0] < tmp_1551); if tmp_1548 == tmp_1552 { keys[0] = tmp_1550; values[0] = tmp_1551; } let tmp_1553 = smem_keys[tmp_1549 * WPT + 30u]; let tmp_1554 = smem_vals[tmp_1549 * WPT + 30u]; let tmp_1555 = keys[1] < tmp_1553 || (keys[1] == tmp_1553 && values[1] < tmp_1554); if tmp_1548 == tmp_1555 { keys[1] = tmp_1553; values[1] = tmp_1554; } let tmp_1556 = smem_keys[tmp_1549 * WPT + 29u]; let tmp_1557 = smem_vals[tmp_1549 * WPT + 29u]; let tmp_1558 = keys[2] < tmp_1556 || (keys[2] == tmp_1556 && values[2] < tmp_1557); if tmp_1548 == tmp_1558 { keys[2] = tmp_1556; values[2] = tmp_1557; } let tmp_1559 = smem_keys[tmp_1549 * WPT + 28u]; let tmp_1560 = smem_vals[tmp_1549 * WPT + 28u]; let tmp_1561 = keys[3] < tmp_1559 || (keys[3] == tmp_1559 && values[3] < tmp_1560); if tmp_1548 == tmp_1561 { keys[3] = tmp_1559; values[3] = tmp_1560; } let tmp_1562 = smem_keys[tmp_1549 * WPT + 27u]; let tmp_1563 = smem_vals[tmp_1549 * WPT + 27u]; let tmp_1564 = keys[4] < tmp_1562 || (keys[4] == tmp_1562 && values[4] < tmp_1563); if tmp_1548 == tmp_1564 { keys[4] = tmp_1562; values[4] = tmp_1563; } let tmp_1565 = smem_keys[tmp_1549 * WPT + 26u]; let tmp_1566 = smem_vals[tmp_1549 * WPT + 26u]; let tmp_1567 = keys[5] < tmp_1565 || (keys[5] == tmp_1565 && values[5] < tmp_1566); if tmp_1548 == tmp_1567 { keys[5] = tmp_1565; values[5] = tmp_1566; } let tmp_1568 = smem_keys[tmp_1549 * WPT + 25u]; let tmp_1569 = smem_vals[tmp_1549 * WPT + 25u]; let tmp_1570 = keys[6] < tmp_1568 || (keys[6] == tmp_1568 && values[6] < tmp_1569); if tmp_1548 == tmp_1570 { keys[6] = tmp_1568; values[6] = tmp_1569; } let tmp_1571 = smem_keys[tmp_1549 * WPT + 24u]; let tmp_1572 = smem_vals[tmp_1549 * WPT + 24u]; let tmp_1573 = keys[7] < tmp_1571 || (keys[7] == tmp_1571 && values[7] < tmp_1572); if tmp_1548 == tmp_1573 { keys[7] = tmp_1571; values[7] = tmp_1572; } let tmp_1574 = smem_keys[tmp_1549 * WPT + 23u]; let tmp_1575 = smem_vals[tmp_1549 * WPT + 23u]; let tmp_1576 = keys[8] < tmp_1574 || (keys[8] == tmp_1574 && values[8] < tmp_1575); if tmp_1548 == tmp_1576 { keys[8] = tmp_1574; values[8] = tmp_1575; } let tmp_1577 = smem_keys[tmp_1549 * WPT + 22u]; let tmp_1578 = smem_vals[tmp_1549 * WPT + 22u]; let tmp_1579 = keys[9] < tmp_1577 || (keys[9] == tmp_1577 && values[9] < tmp_1578); if tmp_1548 == tmp_1579 { keys[9] = tmp_1577; values[9] = tmp_1578; } let tmp_1580 = smem_keys[tmp_1549 * WPT + 21u]; let tmp_1581 = smem_vals[tmp_1549 * WPT + 21u]; let tmp_1582 = keys[10] < tmp_1580 || (keys[10] == tmp_1580 && values[10] < tmp_1581); if tmp_1548 == tmp_1582 { keys[10] = tmp_1580; values[10] = tmp_1581; } let tmp_1583 = smem_keys[tmp_1549 * WPT + 20u]; let tmp_1584 = smem_vals[tmp_1549 * WPT + 20u]; let tmp_1585 = keys[11] < tmp_1583 || (keys[11] == tmp_1583 && values[11] < tmp_1584); if tmp_1548 == tmp_1585 { keys[11] = tmp_1583; values[11] = tmp_1584; } let tmp_1586 = smem_keys[tmp_1549 * WPT + 19u]; let tmp_1587 = smem_vals[tmp_1549 * WPT + 19u]; let tmp_1588 = keys[12] < tmp_1586 || (keys[12] == tmp_1586 && values[12] < tmp_1587); if tmp_1548 == tmp_1588 { keys[12] = tmp_1586; values[12] = tmp_1587; } let tmp_1589 = smem_keys[tmp_1549 * WPT + 18u]; let tmp_1590 = smem_vals[tmp_1549 * WPT + 18u]; let tmp_1591 = keys[13] < tmp_1589 || (keys[13] == tmp_1589 && values[13] < tmp_1590); if tmp_1548 == tmp_1591 { keys[13] = tmp_1589; values[13] = tmp_1590; } let tmp_1592 = smem_keys[tmp_1549 * WPT + 17u]; let tmp_1593 = smem_vals[tmp_1549 * WPT + 17u]; let tmp_1594 = keys[14] < tmp_1592 || (keys[14] == tmp_1592 && values[14] < tmp_1593); if tmp_1548 == tmp_1594 { keys[14] = tmp_1592; values[14] = tmp_1593; } let tmp_1595 = smem_keys[tmp_1549 * WPT + 16u]; let tmp_1596 = smem_vals[tmp_1549 * WPT + 16u]; let tmp_1597 = keys[15] < tmp_1595 || (keys[15] == tmp_1595 && values[15] < tmp_1596); if tmp_1548 == tmp_1597 { keys[15] = tmp_1595; values[15] = tmp_1596; } let tmp_1598 = smem_keys[tmp_1549 * WPT + 15u]; let tmp_1599 = smem_vals[tmp_1549 * WPT + 15u]; let tmp_1600 = keys[16] < tmp_1598 || (keys[16] == tmp_1598 && values[16] < tmp_1599); if tmp_1548 == tmp_1600 { keys[16] = tmp_1598; values[16] = tmp_1599; } let tmp_1601 = smem_keys[tmp_1549 * WPT + 14u]; let tmp_1602 = smem_vals[tmp_1549 * WPT + 14u]; let tmp_1603 = keys[17] < tmp_1601 || (keys[17] == tmp_1601 && values[17] < tmp_1602); if tmp_1548 == tmp_1603 { keys[17] = tmp_1601; values[17] = tmp_1602; } let tmp_1604 = smem_keys[tmp_1549 * WPT + 13u]; let tmp_1605 = smem_vals[tmp_1549 * WPT + 13u]; let tmp_1606 = keys[18] < tmp_1604 || (keys[18] == tmp_1604 && values[18] < tmp_1605); if tmp_1548 == tmp_1606 { keys[18] = tmp_1604; values[18] = tmp_1605; } let tmp_1607 = smem_keys[tmp_1549 * WPT + 12u]; let tmp_1608 = smem_vals[tmp_1549 * WPT + 12u]; let tmp_1609 = keys[19] < tmp_1607 || (keys[19] == tmp_1607 && values[19] < tmp_1608); if tmp_1548 == tmp_1609 { keys[19] = tmp_1607; values[19] = tmp_1608; } let tmp_1610 = smem_keys[tmp_1549 * WPT + 11u]; let tmp_1611 = smem_vals[tmp_1549 * WPT + 11u]; let tmp_1612 = keys[20] < tmp_1610 || (keys[20] == tmp_1610 && values[20] < tmp_1611); if tmp_1548 == tmp_1612 { keys[20] = tmp_1610; values[20] = tmp_1611; } let tmp_1613 = smem_keys[tmp_1549 * WPT + 10u]; let tmp_1614 = smem_vals[tmp_1549 * WPT + 10u]; let tmp_1615 = keys[21] < tmp_1613 || (keys[21] == tmp_1613 && values[21] < tmp_1614); if tmp_1548 == tmp_1615 { keys[21] = tmp_1613; values[21] = tmp_1614; } let tmp_1616 = smem_keys[tmp_1549 * WPT + 9u]; let tmp_1617 = smem_vals[tmp_1549 * WPT + 9u]; let tmp_1618 = keys[22] < tmp_1616 || (keys[22] == tmp_1616 && values[22] < tmp_1617); if tmp_1548 == tmp_1618 { keys[22] = tmp_1616; values[22] = tmp_1617; } let tmp_1619 = smem_keys[tmp_1549 * WPT + 8u]; let tmp_1620 = smem_vals[tmp_1549 * WPT + 8u]; let tmp_1621 = keys[23] < tmp_1619 || (keys[23] == tmp_1619 && values[23] < tmp_1620); if tmp_1548 == tmp_1621 { keys[23] = tmp_1619; values[23] = tmp_1620; } let tmp_1622 = smem_keys[tmp_1549 * WPT + 7u]; let tmp_1623 = smem_vals[tmp_1549 * WPT + 7u]; let tmp_1624 = keys[24] < tmp_1622 || (keys[24] == tmp_1622 && values[24] < tmp_1623); if tmp_1548 == tmp_1624 { keys[24] = tmp_1622; values[24] = tmp_1623; } let tmp_1625 = smem_keys[tmp_1549 * WPT + 6u]; let tmp_1626 = smem_vals[tmp_1549 * WPT + 6u]; let tmp_1627 = keys[25] < tmp_1625 || (keys[25] == tmp_1625 && values[25] < tmp_1626); if tmp_1548 == tmp_1627 { keys[25] = tmp_1625; values[25] = tmp_1626; } let tmp_1628 = smem_keys[tmp_1549 * WPT + 5u]; let tmp_1629 = smem_vals[tmp_1549 * WPT + 5u]; let tmp_1630 = keys[26] < tmp_1628 || (keys[26] == tmp_1628 && values[26] < tmp_1629); if tmp_1548 == tmp_1630 { keys[26] = tmp_1628; values[26] = tmp_1629; } let tmp_1631 = smem_keys[tmp_1549 * WPT + 4u]; let tmp_1632 = smem_vals[tmp_1549 * WPT + 4u]; let tmp_1633 = keys[27] < tmp_1631 || (keys[27] == tmp_1631 && values[27] < tmp_1632); if tmp_1548 == tmp_1633 { keys[27] = tmp_1631; values[27] = tmp_1632; } let tmp_1634 = smem_keys[tmp_1549 * WPT + 3u]; let tmp_1635 = smem_vals[tmp_1549 * WPT + 3u]; let tmp_1636 = keys[28] < tmp_1634 || (keys[28] == tmp_1634 && values[28] < tmp_1635); if tmp_1548 == tmp_1636 { keys[28] = tmp_1634; values[28] = tmp_1635; } let tmp_1637 = smem_keys[tmp_1549 * WPT + 2u]; let tmp_1638 = smem_vals[tmp_1549 * WPT + 2u]; let tmp_1639 = keys[29] < tmp_1637 || (keys[29] == tmp_1637 && values[29] < tmp_1638); if tmp_1548 == tmp_1639 { keys[29] = tmp_1637; values[29] = tmp_1638; } let tmp_1640 = smem_keys[tmp_1549 * WPT + 1u]; let tmp_1641 = smem_vals[tmp_1549 * WPT + 1u]; let tmp_1642 = keys[30] < tmp_1640 || (keys[30] == tmp_1640 && values[30] < tmp_1641); if tmp_1548 == tmp_1642 { keys[30] = tmp_1640; values[30] = tmp_1641; } let tmp_1643 = smem_keys[tmp_1549 * WPT + 0u]; let tmp_1644 = smem_vals[tmp_1549 * WPT + 0u]; let tmp_1645 = keys[31] < tmp_1643 || (keys[31] == tmp_1643 && values[31] < tmp_1644); if tmp_1548 == tmp_1645 { keys[31] = tmp_1643; values[31] = tmp_1644; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_1646 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_1647 = seg_base + (local_tid ^ 4u); let tmp_1648 = smem_keys[tmp_1647 * WPT + 0u]; let tmp_1649 = smem_vals[tmp_1647 * WPT + 0u]; let tmp_1650 = keys[0] < tmp_1648 || (keys[0] == tmp_1648 && values[0] < tmp_1649); if tmp_1646 == tmp_1650 { keys[0] = tmp_1648; values[0] = tmp_1649; } let tmp_1651 = smem_keys[tmp_1647 * WPT + 1u]; let tmp_1652 = smem_vals[tmp_1647 * WPT + 1u]; let tmp_1653 = keys[1] < tmp_1651 || (keys[1] == tmp_1651 && values[1] < tmp_1652); if tmp_1646 == tmp_1653 { keys[1] = tmp_1651; values[1] = tmp_1652; } let tmp_1654 = smem_keys[tmp_1647 * WPT + 2u]; let tmp_1655 = smem_vals[tmp_1647 * WPT + 2u]; let tmp_1656 = keys[2] < tmp_1654 || (keys[2] == tmp_1654 && values[2] < tmp_1655); if tmp_1646 == tmp_1656 { keys[2] = tmp_1654; values[2] = tmp_1655; } let tmp_1657 = smem_keys[tmp_1647 * WPT + 3u]; let tmp_1658 = smem_vals[tmp_1647 * WPT + 3u]; let tmp_1659 = keys[3] < tmp_1657 || (keys[3] == tmp_1657 && values[3] < tmp_1658); if tmp_1646 == tmp_1659 { keys[3] = tmp_1657; values[3] = tmp_1658; } let tmp_1660 = smem_keys[tmp_1647 * WPT + 4u]; let tmp_1661 = smem_vals[tmp_1647 * WPT + 4u]; let tmp_1662 = keys[4] < tmp_1660 || (keys[4] == tmp_1660 && values[4] < tmp_1661); if tmp_1646 == tmp_1662 { keys[4] = tmp_1660; values[4] = tmp_1661; } let tmp_1663 = smem_keys[tmp_1647 * WPT + 5u]; let tmp_1664 = smem_vals[tmp_1647 * WPT + 5u]; let tmp_1665 = keys[5] < tmp_1663 || (keys[5] == tmp_1663 && values[5] < tmp_1664); if tmp_1646 == tmp_1665 { keys[5] = tmp_1663; values[5] = tmp_1664; } let tmp_1666 = smem_keys[tmp_1647 * WPT + 6u]; let tmp_1667 = smem_vals[tmp_1647 * WPT + 6u]; let tmp_1668 = keys[6] < tmp_1666 || (keys[6] == tmp_1666 && values[6] < tmp_1667); if tmp_1646 == tmp_1668 { keys[6] = tmp_1666; values[6] = tmp_1667; } let tmp_1669 = smem_keys[tmp_1647 * WPT + 7u]; let tmp_1670 = smem_vals[tmp_1647 * WPT + 7u]; let tmp_1671 = keys[7] < tmp_1669 || (keys[7] == tmp_1669 && values[7] < tmp_1670); if tmp_1646 == tmp_1671 { keys[7] = tmp_1669; values[7] = tmp_1670; } let tmp_1672 = smem_keys[tmp_1647 * WPT + 8u]; let tmp_1673 = smem_vals[tmp_1647 * WPT + 8u]; let tmp_1674 = keys[8] < tmp_1672 || (keys[8] == tmp_1672 && values[8] < tmp_1673); if tmp_1646 == tmp_1674 { keys[8] = tmp_1672; values[8] = tmp_1673; } let tmp_1675 = smem_keys[tmp_1647 * WPT + 9u]; let tmp_1676 = smem_vals[tmp_1647 * WPT + 9u]; let tmp_1677 = keys[9] < tmp_1675 || (keys[9] == tmp_1675 && values[9] < tmp_1676); if tmp_1646 == tmp_1677 { keys[9] = tmp_1675; values[9] = tmp_1676; } let tmp_1678 = smem_keys[tmp_1647 * WPT + 10u]; let tmp_1679 = smem_vals[tmp_1647 * WPT + 10u]; let tmp_1680 = keys[10] < tmp_1678 || (keys[10] == tmp_1678 && values[10] < tmp_1679); if tmp_1646 == tmp_1680 { keys[10] = tmp_1678; values[10] = tmp_1679; } let tmp_1681 = smem_keys[tmp_1647 * WPT + 11u]; let tmp_1682 = smem_vals[tmp_1647 * WPT + 11u]; let tmp_1683 = keys[11] < tmp_1681 || (keys[11] == tmp_1681 && values[11] < tmp_1682); if tmp_1646 == tmp_1683 { keys[11] = tmp_1681; values[11] = tmp_1682; } let tmp_1684 = smem_keys[tmp_1647 * WPT + 12u]; let tmp_1685 = smem_vals[tmp_1647 * WPT + 12u]; let tmp_1686 = keys[12] < tmp_1684 || (keys[12] == tmp_1684 && values[12] < tmp_1685); if tmp_1646 == tmp_1686 { keys[12] = tmp_1684; values[12] = tmp_1685; } let tmp_1687 = smem_keys[tmp_1647 * WPT + 13u]; let tmp_1688 = smem_vals[tmp_1647 * WPT + 13u]; let tmp_1689 = keys[13] < tmp_1687 || (keys[13] == tmp_1687 && values[13] < tmp_1688); if tmp_1646 == tmp_1689 { keys[13] = tmp_1687; values[13] = tmp_1688; } let tmp_1690 = smem_keys[tmp_1647 * WPT + 14u]; let tmp_1691 = smem_vals[tmp_1647 * WPT + 14u]; let tmp_1692 = keys[14] < tmp_1690 || (keys[14] == tmp_1690 && values[14] < tmp_1691); if tmp_1646 == tmp_1692 { keys[14] = tmp_1690; values[14] = tmp_1691; } let tmp_1693 = smem_keys[tmp_1647 * WPT + 15u]; let tmp_1694 = smem_vals[tmp_1647 * WPT + 15u]; let tmp_1695 = keys[15] < tmp_1693 || (keys[15] == tmp_1693 && values[15] < tmp_1694); if tmp_1646 == tmp_1695 { keys[15] = tmp_1693; values[15] = tmp_1694; } let tmp_1696 = smem_keys[tmp_1647 * WPT + 16u]; let tmp_1697 = smem_vals[tmp_1647 * WPT + 16u]; let tmp_1698 = keys[16] < tmp_1696 || (keys[16] == tmp_1696 && values[16] < tmp_1697); if tmp_1646 == tmp_1698 { keys[16] = tmp_1696; values[16] = tmp_1697; } let tmp_1699 = smem_keys[tmp_1647 * WPT + 17u]; let tmp_1700 = smem_vals[tmp_1647 * WPT + 17u]; let tmp_1701 = keys[17] < tmp_1699 || (keys[17] == tmp_1699 && values[17] < tmp_1700); if tmp_1646 == tmp_1701 { keys[17] = tmp_1699; values[17] = tmp_1700; } let tmp_1702 = smem_keys[tmp_1647 * WPT + 18u]; let tmp_1703 = smem_vals[tmp_1647 * WPT + 18u]; let tmp_1704 = keys[18] < tmp_1702 || (keys[18] == tmp_1702 && values[18] < tmp_1703); if tmp_1646 == tmp_1704 { keys[18] = tmp_1702; values[18] = tmp_1703; } let tmp_1705 = smem_keys[tmp_1647 * WPT + 19u]; let tmp_1706 = smem_vals[tmp_1647 * WPT + 19u]; let tmp_1707 = keys[19] < tmp_1705 || (keys[19] == tmp_1705 && values[19] < tmp_1706); if tmp_1646 == tmp_1707 { keys[19] = tmp_1705; values[19] = tmp_1706; } let tmp_1708 = smem_keys[tmp_1647 * WPT + 20u]; let tmp_1709 = smem_vals[tmp_1647 * WPT + 20u]; let tmp_1710 = keys[20] < tmp_1708 || (keys[20] == tmp_1708 && values[20] < tmp_1709); if tmp_1646 == tmp_1710 { keys[20] = tmp_1708; values[20] = tmp_1709; } let tmp_1711 = smem_keys[tmp_1647 * WPT + 21u]; let tmp_1712 = smem_vals[tmp_1647 * WPT + 21u]; let tmp_1713 = keys[21] < tmp_1711 || (keys[21] == tmp_1711 && values[21] < tmp_1712); if tmp_1646 == tmp_1713 { keys[21] = tmp_1711; values[21] = tmp_1712; } let tmp_1714 = smem_keys[tmp_1647 * WPT + 22u]; let tmp_1715 = smem_vals[tmp_1647 * WPT + 22u]; let tmp_1716 = keys[22] < tmp_1714 || (keys[22] == tmp_1714 && values[22] < tmp_1715); if tmp_1646 == tmp_1716 { keys[22] = tmp_1714; values[22] = tmp_1715; } let tmp_1717 = smem_keys[tmp_1647 * WPT + 23u]; let tmp_1718 = smem_vals[tmp_1647 * WPT + 23u]; let tmp_1719 = keys[23] < tmp_1717 || (keys[23] == tmp_1717 && values[23] < tmp_1718); if tmp_1646 == tmp_1719 { keys[23] = tmp_1717; values[23] = tmp_1718; } let tmp_1720 = smem_keys[tmp_1647 * WPT + 24u]; let tmp_1721 = smem_vals[tmp_1647 * WPT + 24u]; let tmp_1722 = keys[24] < tmp_1720 || (keys[24] == tmp_1720 && values[24] < tmp_1721); if tmp_1646 == tmp_1722 { keys[24] = tmp_1720; values[24] = tmp_1721; } let tmp_1723 = smem_keys[tmp_1647 * WPT + 25u]; let tmp_1724 = smem_vals[tmp_1647 * WPT + 25u]; let tmp_1725 = keys[25] < tmp_1723 || (keys[25] == tmp_1723 && values[25] < tmp_1724); if tmp_1646 == tmp_1725 { keys[25] = tmp_1723; values[25] = tmp_1724; } let tmp_1726 = smem_keys[tmp_1647 * WPT + 26u]; let tmp_1727 = smem_vals[tmp_1647 * WPT + 26u]; let tmp_1728 = keys[26] < tmp_1726 || (keys[26] == tmp_1726 && values[26] < tmp_1727); if tmp_1646 == tmp_1728 { keys[26] = tmp_1726; values[26] = tmp_1727; } let tmp_1729 = smem_keys[tmp_1647 * WPT + 27u]; let tmp_1730 = smem_vals[tmp_1647 * WPT + 27u]; let tmp_1731 = keys[27] < tmp_1729 || (keys[27] == tmp_1729 && values[27] < tmp_1730); if tmp_1646 == tmp_1731 { keys[27] = tmp_1729; values[27] = tmp_1730; } let tmp_1732 = smem_keys[tmp_1647 * WPT + 28u]; let tmp_1733 = smem_vals[tmp_1647 * WPT + 28u]; let tmp_1734 = keys[28] < tmp_1732 || (keys[28] == tmp_1732 && values[28] < tmp_1733); if tmp_1646 == tmp_1734 { keys[28] = tmp_1732; values[28] = tmp_1733; } let tmp_1735 = smem_keys[tmp_1647 * WPT + 29u]; let tmp_1736 = smem_vals[tmp_1647 * WPT + 29u]; let tmp_1737 = keys[29] < tmp_1735 || (keys[29] == tmp_1735 && values[29] < tmp_1736); if tmp_1646 == tmp_1737 { keys[29] = tmp_1735; values[29] = tmp_1736; } let tmp_1738 = smem_keys[tmp_1647 * WPT + 30u]; let tmp_1739 = smem_vals[tmp_1647 * WPT + 30u]; let tmp_1740 = keys[30] < tmp_1738 || (keys[30] == tmp_1738 && values[30] < tmp_1739); if tmp_1646 == tmp_1740 { keys[30] = tmp_1738; values[30] = tmp_1739; } let tmp_1741 = smem_keys[tmp_1647 * WPT + 31u]; let tmp_1742 = smem_vals[tmp_1647 * WPT + 31u]; let tmp_1743 = keys[31] < tmp_1741 || (keys[31] == tmp_1741 && values[31] < tmp_1742); if tmp_1646 == tmp_1743 { keys[31] = tmp_1741; values[31] = tmp_1742; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_1744 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_1745 = seg_base + (local_tid ^ 2u); let tmp_1746 = smem_keys[tmp_1745 * WPT + 0u]; let tmp_1747 = smem_vals[tmp_1745 * WPT + 0u]; let tmp_1748 = keys[0] < tmp_1746 || (keys[0] == tmp_1746 && values[0] < tmp_1747); if tmp_1744 == tmp_1748 { keys[0] = tmp_1746; values[0] = tmp_1747; } let tmp_1749 = smem_keys[tmp_1745 * WPT + 1u]; let tmp_1750 = smem_vals[tmp_1745 * WPT + 1u]; let tmp_1751 = keys[1] < tmp_1749 || (keys[1] == tmp_1749 && values[1] < tmp_1750); if tmp_1744 == tmp_1751 { keys[1] = tmp_1749; values[1] = tmp_1750; } let tmp_1752 = smem_keys[tmp_1745 * WPT + 2u]; let tmp_1753 = smem_vals[tmp_1745 * WPT + 2u]; let tmp_1754 = keys[2] < tmp_1752 || (keys[2] == tmp_1752 && values[2] < tmp_1753); if tmp_1744 == tmp_1754 { keys[2] = tmp_1752; values[2] = tmp_1753; } let tmp_1755 = smem_keys[tmp_1745 * WPT + 3u]; let tmp_1756 = smem_vals[tmp_1745 * WPT + 3u]; let tmp_1757 = keys[3] < tmp_1755 || (keys[3] == tmp_1755 && values[3] < tmp_1756); if tmp_1744 == tmp_1757 { keys[3] = tmp_1755; values[3] = tmp_1756; } let tmp_1758 = smem_keys[tmp_1745 * WPT + 4u]; let tmp_1759 = smem_vals[tmp_1745 * WPT + 4u]; let tmp_1760 = keys[4] < tmp_1758 || (keys[4] == tmp_1758 && values[4] < tmp_1759); if tmp_1744 == tmp_1760 { keys[4] = tmp_1758; values[4] = tmp_1759; } let tmp_1761 = smem_keys[tmp_1745 * WPT + 5u]; let tmp_1762 = smem_vals[tmp_1745 * WPT + 5u]; let tmp_1763 = keys[5] < tmp_1761 || (keys[5] == tmp_1761 && values[5] < tmp_1762); if tmp_1744 == tmp_1763 { keys[5] = tmp_1761; values[5] = tmp_1762; } let tmp_1764 = smem_keys[tmp_1745 * WPT + 6u]; let tmp_1765 = smem_vals[tmp_1745 * WPT + 6u]; let tmp_1766 = keys[6] < tmp_1764 || (keys[6] == tmp_1764 && values[6] < tmp_1765); if tmp_1744 == tmp_1766 { keys[6] = tmp_1764; values[6] = tmp_1765; } let tmp_1767 = smem_keys[tmp_1745 * WPT + 7u]; let tmp_1768 = smem_vals[tmp_1745 * WPT + 7u]; let tmp_1769 = keys[7] < tmp_1767 || (keys[7] == tmp_1767 && values[7] < tmp_1768); if tmp_1744 == tmp_1769 { keys[7] = tmp_1767; values[7] = tmp_1768; } let tmp_1770 = smem_keys[tmp_1745 * WPT + 8u]; let tmp_1771 = smem_vals[tmp_1745 * WPT + 8u]; let tmp_1772 = keys[8] < tmp_1770 || (keys[8] == tmp_1770 && values[8] < tmp_1771); if tmp_1744 == tmp_1772 { keys[8] = tmp_1770; values[8] = tmp_1771; } let tmp_1773 = smem_keys[tmp_1745 * WPT + 9u]; let tmp_1774 = smem_vals[tmp_1745 * WPT + 9u]; let tmp_1775 = keys[9] < tmp_1773 || (keys[9] == tmp_1773 && values[9] < tmp_1774); if tmp_1744 == tmp_1775 { keys[9] = tmp_1773; values[9] = tmp_1774; } let tmp_1776 = smem_keys[tmp_1745 * WPT + 10u]; let tmp_1777 = smem_vals[tmp_1745 * WPT + 10u]; let tmp_1778 = keys[10] < tmp_1776 || (keys[10] == tmp_1776 && values[10] < tmp_1777); if tmp_1744 == tmp_1778 { keys[10] = tmp_1776; values[10] = tmp_1777; } let tmp_1779 = smem_keys[tmp_1745 * WPT + 11u]; let tmp_1780 = smem_vals[tmp_1745 * WPT + 11u]; let tmp_1781 = keys[11] < tmp_1779 || (keys[11] == tmp_1779 && values[11] < tmp_1780); if tmp_1744 == tmp_1781 { keys[11] = tmp_1779; values[11] = tmp_1780; } let tmp_1782 = smem_keys[tmp_1745 * WPT + 12u]; let tmp_1783 = smem_vals[tmp_1745 * WPT + 12u]; let tmp_1784 = keys[12] < tmp_1782 || (keys[12] == tmp_1782 && values[12] < tmp_1783); if tmp_1744 == tmp_1784 { keys[12] = tmp_1782; values[12] = tmp_1783; } let tmp_1785 = smem_keys[tmp_1745 * WPT + 13u]; let tmp_1786 = smem_vals[tmp_1745 * WPT + 13u]; let tmp_1787 = keys[13] < tmp_1785 || (keys[13] == tmp_1785 && values[13] < tmp_1786); if tmp_1744 == tmp_1787 { keys[13] = tmp_1785; values[13] = tmp_1786; } let tmp_1788 = smem_keys[tmp_1745 * WPT + 14u]; let tmp_1789 = smem_vals[tmp_1745 * WPT + 14u]; let tmp_1790 = keys[14] < tmp_1788 || (keys[14] == tmp_1788 && values[14] < tmp_1789); if tmp_1744 == tmp_1790 { keys[14] = tmp_1788; values[14] = tmp_1789; } let tmp_1791 = smem_keys[tmp_1745 * WPT + 15u]; let tmp_1792 = smem_vals[tmp_1745 * WPT + 15u]; let tmp_1793 = keys[15] < tmp_1791 || (keys[15] == tmp_1791 && values[15] < tmp_1792); if tmp_1744 == tmp_1793 { keys[15] = tmp_1791; values[15] = tmp_1792; } let tmp_1794 = smem_keys[tmp_1745 * WPT + 16u]; let tmp_1795 = smem_vals[tmp_1745 * WPT + 16u]; let tmp_1796 = keys[16] < tmp_1794 || (keys[16] == tmp_1794 && values[16] < tmp_1795); if tmp_1744 == tmp_1796 { keys[16] = tmp_1794; values[16] = tmp_1795; } let tmp_1797 = smem_keys[tmp_1745 * WPT + 17u]; let tmp_1798 = smem_vals[tmp_1745 * WPT + 17u]; let tmp_1799 = keys[17] < tmp_1797 || (keys[17] == tmp_1797 && values[17] < tmp_1798); if tmp_1744 == tmp_1799 { keys[17] = tmp_1797; values[17] = tmp_1798; } let tmp_1800 = smem_keys[tmp_1745 * WPT + 18u]; let tmp_1801 = smem_vals[tmp_1745 * WPT + 18u]; let tmp_1802 = keys[18] < tmp_1800 || (keys[18] == tmp_1800 && values[18] < tmp_1801); if tmp_1744 == tmp_1802 { keys[18] = tmp_1800; values[18] = tmp_1801; } let tmp_1803 = smem_keys[tmp_1745 * WPT + 19u]; let tmp_1804 = smem_vals[tmp_1745 * WPT + 19u]; let tmp_1805 = keys[19] < tmp_1803 || (keys[19] == tmp_1803 && values[19] < tmp_1804); if tmp_1744 == tmp_1805 { keys[19] = tmp_1803; values[19] = tmp_1804; } let tmp_1806 = smem_keys[tmp_1745 * WPT + 20u]; let tmp_1807 = smem_vals[tmp_1745 * WPT + 20u]; let tmp_1808 = keys[20] < tmp_1806 || (keys[20] == tmp_1806 && values[20] < tmp_1807); if tmp_1744 == tmp_1808 { keys[20] = tmp_1806; values[20] = tmp_1807; } let tmp_1809 = smem_keys[tmp_1745 * WPT + 21u]; let tmp_1810 = smem_vals[tmp_1745 * WPT + 21u]; let tmp_1811 = keys[21] < tmp_1809 || (keys[21] == tmp_1809 && values[21] < tmp_1810); if tmp_1744 == tmp_1811 { keys[21] = tmp_1809; values[21] = tmp_1810; } let tmp_1812 = smem_keys[tmp_1745 * WPT + 22u]; let tmp_1813 = smem_vals[tmp_1745 * WPT + 22u]; let tmp_1814 = keys[22] < tmp_1812 || (keys[22] == tmp_1812 && values[22] < tmp_1813); if tmp_1744 == tmp_1814 { keys[22] = tmp_1812; values[22] = tmp_1813; } let tmp_1815 = smem_keys[tmp_1745 * WPT + 23u]; let tmp_1816 = smem_vals[tmp_1745 * WPT + 23u]; let tmp_1817 = keys[23] < tmp_1815 || (keys[23] == tmp_1815 && values[23] < tmp_1816); if tmp_1744 == tmp_1817 { keys[23] = tmp_1815; values[23] = tmp_1816; } let tmp_1818 = smem_keys[tmp_1745 * WPT + 24u]; let tmp_1819 = smem_vals[tmp_1745 * WPT + 24u]; let tmp_1820 = keys[24] < tmp_1818 || (keys[24] == tmp_1818 && values[24] < tmp_1819); if tmp_1744 == tmp_1820 { keys[24] = tmp_1818; values[24] = tmp_1819; } let tmp_1821 = smem_keys[tmp_1745 * WPT + 25u]; let tmp_1822 = smem_vals[tmp_1745 * WPT + 25u]; let tmp_1823 = keys[25] < tmp_1821 || (keys[25] == tmp_1821 && values[25] < tmp_1822); if tmp_1744 == tmp_1823 { keys[25] = tmp_1821; values[25] = tmp_1822; } let tmp_1824 = smem_keys[tmp_1745 * WPT + 26u]; let tmp_1825 = smem_vals[tmp_1745 * WPT + 26u]; let tmp_1826 = keys[26] < tmp_1824 || (keys[26] == tmp_1824 && values[26] < tmp_1825); if tmp_1744 == tmp_1826 { keys[26] = tmp_1824; values[26] = tmp_1825; } let tmp_1827 = smem_keys[tmp_1745 * WPT + 27u]; let tmp_1828 = smem_vals[tmp_1745 * WPT + 27u]; let tmp_1829 = keys[27] < tmp_1827 || (keys[27] == tmp_1827 && values[27] < tmp_1828); if tmp_1744 == tmp_1829 { keys[27] = tmp_1827; values[27] = tmp_1828; } let tmp_1830 = smem_keys[tmp_1745 * WPT + 28u]; let tmp_1831 = smem_vals[tmp_1745 * WPT + 28u]; let tmp_1832 = keys[28] < tmp_1830 || (keys[28] == tmp_1830 && values[28] < tmp_1831); if tmp_1744 == tmp_1832 { keys[28] = tmp_1830; values[28] = tmp_1831; } let tmp_1833 = smem_keys[tmp_1745 * WPT + 29u]; let tmp_1834 = smem_vals[tmp_1745 * WPT + 29u]; let tmp_1835 = keys[29] < tmp_1833 || (keys[29] == tmp_1833 && values[29] < tmp_1834); if tmp_1744 == tmp_1835 { keys[29] = tmp_1833; values[29] = tmp_1834; } let tmp_1836 = smem_keys[tmp_1745 * WPT + 30u]; let tmp_1837 = smem_vals[tmp_1745 * WPT + 30u]; let tmp_1838 = keys[30] < tmp_1836 || (keys[30] == tmp_1836 && values[30] < tmp_1837); if tmp_1744 == tmp_1838 { keys[30] = tmp_1836; values[30] = tmp_1837; } let tmp_1839 = smem_keys[tmp_1745 * WPT + 31u]; let tmp_1840 = smem_vals[tmp_1745 * WPT + 31u]; let tmp_1841 = keys[31] < tmp_1839 || (keys[31] == tmp_1839 && values[31] < tmp_1840); if tmp_1744 == tmp_1841 { keys[31] = tmp_1839; values[31] = tmp_1840; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_1842 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_1843 = seg_base + (local_tid ^ 1u); let tmp_1844 = smem_keys[tmp_1843 * WPT + 0u]; let tmp_1845 = smem_vals[tmp_1843 * WPT + 0u]; let tmp_1846 = keys[0] < tmp_1844 || (keys[0] == tmp_1844 && values[0] < tmp_1845); if tmp_1842 == tmp_1846 { keys[0] = tmp_1844; values[0] = tmp_1845; } let tmp_1847 = smem_keys[tmp_1843 * WPT + 1u]; let tmp_1848 = smem_vals[tmp_1843 * WPT + 1u]; let tmp_1849 = keys[1] < tmp_1847 || (keys[1] == tmp_1847 && values[1] < tmp_1848); if tmp_1842 == tmp_1849 { keys[1] = tmp_1847; values[1] = tmp_1848; } let tmp_1850 = smem_keys[tmp_1843 * WPT + 2u]; let tmp_1851 = smem_vals[tmp_1843 * WPT + 2u]; let tmp_1852 = keys[2] < tmp_1850 || (keys[2] == tmp_1850 && values[2] < tmp_1851); if tmp_1842 == tmp_1852 { keys[2] = tmp_1850; values[2] = tmp_1851; } let tmp_1853 = smem_keys[tmp_1843 * WPT + 3u]; let tmp_1854 = smem_vals[tmp_1843 * WPT + 3u]; let tmp_1855 = keys[3] < tmp_1853 || (keys[3] == tmp_1853 && values[3] < tmp_1854); if tmp_1842 == tmp_1855 { keys[3] = tmp_1853; values[3] = tmp_1854; } let tmp_1856 = smem_keys[tmp_1843 * WPT + 4u]; let tmp_1857 = smem_vals[tmp_1843 * WPT + 4u]; let tmp_1858 = keys[4] < tmp_1856 || (keys[4] == tmp_1856 && values[4] < tmp_1857); if tmp_1842 == tmp_1858 { keys[4] = tmp_1856; values[4] = tmp_1857; } let tmp_1859 = smem_keys[tmp_1843 * WPT + 5u]; let tmp_1860 = smem_vals[tmp_1843 * WPT + 5u]; let tmp_1861 = keys[5] < tmp_1859 || (keys[5] == tmp_1859 && values[5] < tmp_1860); if tmp_1842 == tmp_1861 { keys[5] = tmp_1859; values[5] = tmp_1860; } let tmp_1862 = smem_keys[tmp_1843 * WPT + 6u]; let tmp_1863 = smem_vals[tmp_1843 * WPT + 6u]; let tmp_1864 = keys[6] < tmp_1862 || (keys[6] == tmp_1862 && values[6] < tmp_1863); if tmp_1842 == tmp_1864 { keys[6] = tmp_1862; values[6] = tmp_1863; } let tmp_1865 = smem_keys[tmp_1843 * WPT + 7u]; let tmp_1866 = smem_vals[tmp_1843 * WPT + 7u]; let tmp_1867 = keys[7] < tmp_1865 || (keys[7] == tmp_1865 && values[7] < tmp_1866); if tmp_1842 == tmp_1867 { keys[7] = tmp_1865; values[7] = tmp_1866; } let tmp_1868 = smem_keys[tmp_1843 * WPT + 8u]; let tmp_1869 = smem_vals[tmp_1843 * WPT + 8u]; let tmp_1870 = keys[8] < tmp_1868 || (keys[8] == tmp_1868 && values[8] < tmp_1869); if tmp_1842 == tmp_1870 { keys[8] = tmp_1868; values[8] = tmp_1869; } let tmp_1871 = smem_keys[tmp_1843 * WPT + 9u]; let tmp_1872 = smem_vals[tmp_1843 * WPT + 9u]; let tmp_1873 = keys[9] < tmp_1871 || (keys[9] == tmp_1871 && values[9] < tmp_1872); if tmp_1842 == tmp_1873 { keys[9] = tmp_1871; values[9] = tmp_1872; } let tmp_1874 = smem_keys[tmp_1843 * WPT + 10u]; let tmp_1875 = smem_vals[tmp_1843 * WPT + 10u]; let tmp_1876 = keys[10] < tmp_1874 || (keys[10] == tmp_1874 && values[10] < tmp_1875); if tmp_1842 == tmp_1876 { keys[10] = tmp_1874; values[10] = tmp_1875; } let tmp_1877 = smem_keys[tmp_1843 * WPT + 11u]; let tmp_1878 = smem_vals[tmp_1843 * WPT + 11u]; let tmp_1879 = keys[11] < tmp_1877 || (keys[11] == tmp_1877 && values[11] < tmp_1878); if tmp_1842 == tmp_1879 { keys[11] = tmp_1877; values[11] = tmp_1878; } let tmp_1880 = smem_keys[tmp_1843 * WPT + 12u]; let tmp_1881 = smem_vals[tmp_1843 * WPT + 12u]; let tmp_1882 = keys[12] < tmp_1880 || (keys[12] == tmp_1880 && values[12] < tmp_1881); if tmp_1842 == tmp_1882 { keys[12] = tmp_1880; values[12] = tmp_1881; } let tmp_1883 = smem_keys[tmp_1843 * WPT + 13u]; let tmp_1884 = smem_vals[tmp_1843 * WPT + 13u]; let tmp_1885 = keys[13] < tmp_1883 || (keys[13] == tmp_1883 && values[13] < tmp_1884); if tmp_1842 == tmp_1885 { keys[13] = tmp_1883; values[13] = tmp_1884; } let tmp_1886 = smem_keys[tmp_1843 * WPT + 14u]; let tmp_1887 = smem_vals[tmp_1843 * WPT + 14u]; let tmp_1888 = keys[14] < tmp_1886 || (keys[14] == tmp_1886 && values[14] < tmp_1887); if tmp_1842 == tmp_1888 { keys[14] = tmp_1886; values[14] = tmp_1887; } let tmp_1889 = smem_keys[tmp_1843 * WPT + 15u]; let tmp_1890 = smem_vals[tmp_1843 * WPT + 15u]; let tmp_1891 = keys[15] < tmp_1889 || (keys[15] == tmp_1889 && values[15] < tmp_1890); if tmp_1842 == tmp_1891 { keys[15] = tmp_1889; values[15] = tmp_1890; } let tmp_1892 = smem_keys[tmp_1843 * WPT + 16u]; let tmp_1893 = smem_vals[tmp_1843 * WPT + 16u]; let tmp_1894 = keys[16] < tmp_1892 || (keys[16] == tmp_1892 && values[16] < tmp_1893); if tmp_1842 == tmp_1894 { keys[16] = tmp_1892; values[16] = tmp_1893; } let tmp_1895 = smem_keys[tmp_1843 * WPT + 17u]; let tmp_1896 = smem_vals[tmp_1843 * WPT + 17u]; let tmp_1897 = keys[17] < tmp_1895 || (keys[17] == tmp_1895 && values[17] < tmp_1896); if tmp_1842 == tmp_1897 { keys[17] = tmp_1895; values[17] = tmp_1896; } let tmp_1898 = smem_keys[tmp_1843 * WPT + 18u]; let tmp_1899 = smem_vals[tmp_1843 * WPT + 18u]; let tmp_1900 = keys[18] < tmp_1898 || (keys[18] == tmp_1898 && values[18] < tmp_1899); if tmp_1842 == tmp_1900 { keys[18] = tmp_1898; values[18] = tmp_1899; } let tmp_1901 = smem_keys[tmp_1843 * WPT + 19u]; let tmp_1902 = smem_vals[tmp_1843 * WPT + 19u]; let tmp_1903 = keys[19] < tmp_1901 || (keys[19] == tmp_1901 && values[19] < tmp_1902); if tmp_1842 == tmp_1903 { keys[19] = tmp_1901; values[19] = tmp_1902; } let tmp_1904 = smem_keys[tmp_1843 * WPT + 20u]; let tmp_1905 = smem_vals[tmp_1843 * WPT + 20u]; let tmp_1906 = keys[20] < tmp_1904 || (keys[20] == tmp_1904 && values[20] < tmp_1905); if tmp_1842 == tmp_1906 { keys[20] = tmp_1904; values[20] = tmp_1905; } let tmp_1907 = smem_keys[tmp_1843 * WPT + 21u]; let tmp_1908 = smem_vals[tmp_1843 * WPT + 21u]; let tmp_1909 = keys[21] < tmp_1907 || (keys[21] == tmp_1907 && values[21] < tmp_1908); if tmp_1842 == tmp_1909 { keys[21] = tmp_1907; values[21] = tmp_1908; } let tmp_1910 = smem_keys[tmp_1843 * WPT + 22u]; let tmp_1911 = smem_vals[tmp_1843 * WPT + 22u]; let tmp_1912 = keys[22] < tmp_1910 || (keys[22] == tmp_1910 && values[22] < tmp_1911); if tmp_1842 == tmp_1912 { keys[22] = tmp_1910; values[22] = tmp_1911; } let tmp_1913 = smem_keys[tmp_1843 * WPT + 23u]; let tmp_1914 = smem_vals[tmp_1843 * WPT + 23u]; let tmp_1915 = keys[23] < tmp_1913 || (keys[23] == tmp_1913 && values[23] < tmp_1914); if tmp_1842 == tmp_1915 { keys[23] = tmp_1913; values[23] = tmp_1914; } let tmp_1916 = smem_keys[tmp_1843 * WPT + 24u]; let tmp_1917 = smem_vals[tmp_1843 * WPT + 24u]; let tmp_1918 = keys[24] < tmp_1916 || (keys[24] == tmp_1916 && values[24] < tmp_1917); if tmp_1842 == tmp_1918 { keys[24] = tmp_1916; values[24] = tmp_1917; } let tmp_1919 = smem_keys[tmp_1843 * WPT + 25u]; let tmp_1920 = smem_vals[tmp_1843 * WPT + 25u]; let tmp_1921 = keys[25] < tmp_1919 || (keys[25] == tmp_1919 && values[25] < tmp_1920); if tmp_1842 == tmp_1921 { keys[25] = tmp_1919; values[25] = tmp_1920; } let tmp_1922 = smem_keys[tmp_1843 * WPT + 26u]; let tmp_1923 = smem_vals[tmp_1843 * WPT + 26u]; let tmp_1924 = keys[26] < tmp_1922 || (keys[26] == tmp_1922 && values[26] < tmp_1923); if tmp_1842 == tmp_1924 { keys[26] = tmp_1922; values[26] = tmp_1923; } let tmp_1925 = smem_keys[tmp_1843 * WPT + 27u]; let tmp_1926 = smem_vals[tmp_1843 * WPT + 27u]; let tmp_1927 = keys[27] < tmp_1925 || (keys[27] == tmp_1925 && values[27] < tmp_1926); if tmp_1842 == tmp_1927 { keys[27] = tmp_1925; values[27] = tmp_1926; } let tmp_1928 = smem_keys[tmp_1843 * WPT + 28u]; let tmp_1929 = smem_vals[tmp_1843 * WPT + 28u]; let tmp_1930 = keys[28] < tmp_1928 || (keys[28] == tmp_1928 && values[28] < tmp_1929); if tmp_1842 == tmp_1930 { keys[28] = tmp_1928; values[28] = tmp_1929; } let tmp_1931 = smem_keys[tmp_1843 * WPT + 29u]; let tmp_1932 = smem_vals[tmp_1843 * WPT + 29u]; let tmp_1933 = keys[29] < tmp_1931 || (keys[29] == tmp_1931 && values[29] < tmp_1932); if tmp_1842 == tmp_1933 { keys[29] = tmp_1931; values[29] = tmp_1932; } let tmp_1934 = smem_keys[tmp_1843 * WPT + 30u]; let tmp_1935 = smem_vals[tmp_1843 * WPT + 30u]; let tmp_1936 = keys[30] < tmp_1934 || (keys[30] == tmp_1934 && values[30] < tmp_1935); if tmp_1842 == tmp_1936 { keys[30] = tmp_1934; values[30] = tmp_1935; } let tmp_1937 = smem_keys[tmp_1843 * WPT + 31u]; let tmp_1938 = smem_vals[tmp_1843 * WPT + 31u]; let tmp_1939 = keys[31] < tmp_1937 || (keys[31] == tmp_1937 && values[31] < tmp_1938); if tmp_1842 == tmp_1939 { keys[31] = tmp_1937; values[31] = tmp_1938; } workgroupBarrier(); }
    // exch_local(16,32) 
    // cmp_swap(0,16)
    if keys[0] > keys[16] || (keys[0] == keys[16] && values[0] > values[16]) {
    // swap(0,16) 
    { let tmp_1940 = keys[0]; keys[0] = keys[16]; keys[16] = tmp_1940;let tmp_1941 = values[0]; values[0] = values[16]; values[16] = tmp_1941; }
    }
    // cmp_swap(1,17)
    if keys[1] > keys[17] || (keys[1] == keys[17] && values[1] > values[17]) {
    // swap(1,17) 
    { let tmp_1942 = keys[1]; keys[1] = keys[17]; keys[17] = tmp_1942;let tmp_1943 = values[1]; values[1] = values[17]; values[17] = tmp_1943; }
    }
    // cmp_swap(2,18)
    if keys[2] > keys[18] || (keys[2] == keys[18] && values[2] > values[18]) {
    // swap(2,18) 
    { let tmp_1944 = keys[2]; keys[2] = keys[18]; keys[18] = tmp_1944;let tmp_1945 = values[2]; values[2] = values[18]; values[18] = tmp_1945; }
    }
    // cmp_swap(3,19)
    if keys[3] > keys[19] || (keys[3] == keys[19] && values[3] > values[19]) {
    // swap(3,19) 
    { let tmp_1946 = keys[3]; keys[3] = keys[19]; keys[19] = tmp_1946;let tmp_1947 = values[3]; values[3] = values[19]; values[19] = tmp_1947; }
    }
    // cmp_swap(4,20)
    if keys[4] > keys[20] || (keys[4] == keys[20] && values[4] > values[20]) {
    // swap(4,20) 
    { let tmp_1948 = keys[4]; keys[4] = keys[20]; keys[20] = tmp_1948;let tmp_1949 = values[4]; values[4] = values[20]; values[20] = tmp_1949; }
    }
    // cmp_swap(5,21)
    if keys[5] > keys[21] || (keys[5] == keys[21] && values[5] > values[21]) {
    // swap(5,21) 
    { let tmp_1950 = keys[5]; keys[5] = keys[21]; keys[21] = tmp_1950;let tmp_1951 = values[5]; values[5] = values[21]; values[21] = tmp_1951; }
    }
    // cmp_swap(6,22)
    if keys[6] > keys[22] || (keys[6] == keys[22] && values[6] > values[22]) {
    // swap(6,22) 
    { let tmp_1952 = keys[6]; keys[6] = keys[22]; keys[22] = tmp_1952;let tmp_1953 = values[6]; values[6] = values[22]; values[22] = tmp_1953; }
    }
    // cmp_swap(7,23)
    if keys[7] > keys[23] || (keys[7] == keys[23] && values[7] > values[23]) {
    // swap(7,23) 
    { let tmp_1954 = keys[7]; keys[7] = keys[23]; keys[23] = tmp_1954;let tmp_1955 = values[7]; values[7] = values[23]; values[23] = tmp_1955; }
    }
    // cmp_swap(8,24)
    if keys[8] > keys[24] || (keys[8] == keys[24] && values[8] > values[24]) {
    // swap(8,24) 
    { let tmp_1956 = keys[8]; keys[8] = keys[24]; keys[24] = tmp_1956;let tmp_1957 = values[8]; values[8] = values[24]; values[24] = tmp_1957; }
    }
    // cmp_swap(9,25)
    if keys[9] > keys[25] || (keys[9] == keys[25] && values[9] > values[25]) {
    // swap(9,25) 
    { let tmp_1958 = keys[9]; keys[9] = keys[25]; keys[25] = tmp_1958;let tmp_1959 = values[9]; values[9] = values[25]; values[25] = tmp_1959; }
    }
    // cmp_swap(10,26)
    if keys[10] > keys[26] || (keys[10] == keys[26] && values[10] > values[26]) {
    // swap(10,26) 
    { let tmp_1960 = keys[10]; keys[10] = keys[26]; keys[26] = tmp_1960;let tmp_1961 = values[10]; values[10] = values[26]; values[26] = tmp_1961; }
    }
    // cmp_swap(11,27)
    if keys[11] > keys[27] || (keys[11] == keys[27] && values[11] > values[27]) {
    // swap(11,27) 
    { let tmp_1962 = keys[11]; keys[11] = keys[27]; keys[27] = tmp_1962;let tmp_1963 = values[11]; values[11] = values[27]; values[27] = tmp_1963; }
    }
    // cmp_swap(12,28)
    if keys[12] > keys[28] || (keys[12] == keys[28] && values[12] > values[28]) {
    // swap(12,28) 
    { let tmp_1964 = keys[12]; keys[12] = keys[28]; keys[28] = tmp_1964;let tmp_1965 = values[12]; values[12] = values[28]; values[28] = tmp_1965; }
    }
    // cmp_swap(13,29)
    if keys[13] > keys[29] || (keys[13] == keys[29] && values[13] > values[29]) {
    // swap(13,29) 
    { let tmp_1966 = keys[13]; keys[13] = keys[29]; keys[29] = tmp_1966;let tmp_1967 = values[13]; values[13] = values[29]; values[29] = tmp_1967; }
    }
    // cmp_swap(14,30)
    if keys[14] > keys[30] || (keys[14] == keys[30] && values[14] > values[30]) {
    // swap(14,30) 
    { let tmp_1968 = keys[14]; keys[14] = keys[30]; keys[30] = tmp_1968;let tmp_1969 = values[14]; values[14] = values[30]; values[30] = tmp_1969; }
    }
    // cmp_swap(15,31)
    if keys[15] > keys[31] || (keys[15] == keys[31] && values[15] > values[31]) {
    // swap(15,31) 
    { let tmp_1970 = keys[15]; keys[15] = keys[31]; keys[31] = tmp_1970;let tmp_1971 = values[15]; values[15] = values[31]; values[31] = tmp_1971; }
    }
    // exch_local(8,32) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_1972 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_1972;let tmp_1973 = values[0]; values[0] = values[8]; values[8] = tmp_1973; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_1974 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_1974;let tmp_1975 = values[1]; values[1] = values[9]; values[9] = tmp_1975; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_1976 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_1976;let tmp_1977 = values[2]; values[2] = values[10]; values[10] = tmp_1977; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_1978 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_1978;let tmp_1979 = values[3]; values[3] = values[11]; values[11] = tmp_1979; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_1980 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_1980;let tmp_1981 = values[4]; values[4] = values[12]; values[12] = tmp_1981; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_1982 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_1982;let tmp_1983 = values[5]; values[5] = values[13]; values[13] = tmp_1983; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_1984 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_1984;let tmp_1985 = values[6]; values[6] = values[14]; values[14] = tmp_1985; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_1986 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_1986;let tmp_1987 = values[7]; values[7] = values[15]; values[15] = tmp_1987; }
    }
    // cmp_swap(16,24)
    if keys[16] > keys[24] || (keys[16] == keys[24] && values[16] > values[24]) {
    // swap(16,24) 
    { let tmp_1988 = keys[16]; keys[16] = keys[24]; keys[24] = tmp_1988;let tmp_1989 = values[16]; values[16] = values[24]; values[24] = tmp_1989; }
    }
    // cmp_swap(17,25)
    if keys[17] > keys[25] || (keys[17] == keys[25] && values[17] > values[25]) {
    // swap(17,25) 
    { let tmp_1990 = keys[17]; keys[17] = keys[25]; keys[25] = tmp_1990;let tmp_1991 = values[17]; values[17] = values[25]; values[25] = tmp_1991; }
    }
    // cmp_swap(18,26)
    if keys[18] > keys[26] || (keys[18] == keys[26] && values[18] > values[26]) {
    // swap(18,26) 
    { let tmp_1992 = keys[18]; keys[18] = keys[26]; keys[26] = tmp_1992;let tmp_1993 = values[18]; values[18] = values[26]; values[26] = tmp_1993; }
    }
    // cmp_swap(19,27)
    if keys[19] > keys[27] || (keys[19] == keys[27] && values[19] > values[27]) {
    // swap(19,27) 
    { let tmp_1994 = keys[19]; keys[19] = keys[27]; keys[27] = tmp_1994;let tmp_1995 = values[19]; values[19] = values[27]; values[27] = tmp_1995; }
    }
    // cmp_swap(20,28)
    if keys[20] > keys[28] || (keys[20] == keys[28] && values[20] > values[28]) {
    // swap(20,28) 
    { let tmp_1996 = keys[20]; keys[20] = keys[28]; keys[28] = tmp_1996;let tmp_1997 = values[20]; values[20] = values[28]; values[28] = tmp_1997; }
    }
    // cmp_swap(21,29)
    if keys[21] > keys[29] || (keys[21] == keys[29] && values[21] > values[29]) {
    // swap(21,29) 
    { let tmp_1998 = keys[21]; keys[21] = keys[29]; keys[29] = tmp_1998;let tmp_1999 = values[21]; values[21] = values[29]; values[29] = tmp_1999; }
    }
    // cmp_swap(22,30)
    if keys[22] > keys[30] || (keys[22] == keys[30] && values[22] > values[30]) {
    // swap(22,30) 
    { let tmp_2000 = keys[22]; keys[22] = keys[30]; keys[30] = tmp_2000;let tmp_2001 = values[22]; values[22] = values[30]; values[30] = tmp_2001; }
    }
    // cmp_swap(23,31)
    if keys[23] > keys[31] || (keys[23] == keys[31] && values[23] > values[31]) {
    // swap(23,31) 
    { let tmp_2002 = keys[23]; keys[23] = keys[31]; keys[31] = tmp_2002;let tmp_2003 = values[23]; values[23] = values[31]; values[31] = tmp_2003; }
    }
    // exch_local(4,32) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_2004 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_2004;let tmp_2005 = values[0]; values[0] = values[4]; values[4] = tmp_2005; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_2006 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_2006;let tmp_2007 = values[1]; values[1] = values[5]; values[5] = tmp_2007; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_2008 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_2008;let tmp_2009 = values[2]; values[2] = values[6]; values[6] = tmp_2009; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_2010 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_2010;let tmp_2011 = values[3]; values[3] = values[7]; values[7] = tmp_2011; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_2012 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_2012;let tmp_2013 = values[8]; values[8] = values[12]; values[12] = tmp_2013; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_2014 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_2014;let tmp_2015 = values[9]; values[9] = values[13]; values[13] = tmp_2015; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_2016 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_2016;let tmp_2017 = values[10]; values[10] = values[14]; values[14] = tmp_2017; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_2018 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_2018;let tmp_2019 = values[11]; values[11] = values[15]; values[15] = tmp_2019; }
    }
    // cmp_swap(16,20)
    if keys[16] > keys[20] || (keys[16] == keys[20] && values[16] > values[20]) {
    // swap(16,20) 
    { let tmp_2020 = keys[16]; keys[16] = keys[20]; keys[20] = tmp_2020;let tmp_2021 = values[16]; values[16] = values[20]; values[20] = tmp_2021; }
    }
    // cmp_swap(17,21)
    if keys[17] > keys[21] || (keys[17] == keys[21] && values[17] > values[21]) {
    // swap(17,21) 
    { let tmp_2022 = keys[17]; keys[17] = keys[21]; keys[21] = tmp_2022;let tmp_2023 = values[17]; values[17] = values[21]; values[21] = tmp_2023; }
    }
    // cmp_swap(18,22)
    if keys[18] > keys[22] || (keys[18] == keys[22] && values[18] > values[22]) {
    // swap(18,22) 
    { let tmp_2024 = keys[18]; keys[18] = keys[22]; keys[22] = tmp_2024;let tmp_2025 = values[18]; values[18] = values[22]; values[22] = tmp_2025; }
    }
    // cmp_swap(19,23)
    if keys[19] > keys[23] || (keys[19] == keys[23] && values[19] > values[23]) {
    // swap(19,23) 
    { let tmp_2026 = keys[19]; keys[19] = keys[23]; keys[23] = tmp_2026;let tmp_2027 = values[19]; values[19] = values[23]; values[23] = tmp_2027; }
    }
    // cmp_swap(24,28)
    if keys[24] > keys[28] || (keys[24] == keys[28] && values[24] > values[28]) {
    // swap(24,28) 
    { let tmp_2028 = keys[24]; keys[24] = keys[28]; keys[28] = tmp_2028;let tmp_2029 = values[24]; values[24] = values[28]; values[28] = tmp_2029; }
    }
    // cmp_swap(25,29)
    if keys[25] > keys[29] || (keys[25] == keys[29] && values[25] > values[29]) {
    // swap(25,29) 
    { let tmp_2030 = keys[25]; keys[25] = keys[29]; keys[29] = tmp_2030;let tmp_2031 = values[25]; values[25] = values[29]; values[29] = tmp_2031; }
    }
    // cmp_swap(26,30)
    if keys[26] > keys[30] || (keys[26] == keys[30] && values[26] > values[30]) {
    // swap(26,30) 
    { let tmp_2032 = keys[26]; keys[26] = keys[30]; keys[30] = tmp_2032;let tmp_2033 = values[26]; values[26] = values[30]; values[30] = tmp_2033; }
    }
    // cmp_swap(27,31)
    if keys[27] > keys[31] || (keys[27] == keys[31] && values[27] > values[31]) {
    // swap(27,31) 
    { let tmp_2034 = keys[27]; keys[27] = keys[31]; keys[31] = tmp_2034;let tmp_2035 = values[27]; values[27] = values[31]; values[31] = tmp_2035; }
    }
    // exch_local(2,32) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_2036 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_2036;let tmp_2037 = values[0]; values[0] = values[2]; values[2] = tmp_2037; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_2038 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_2038;let tmp_2039 = values[1]; values[1] = values[3]; values[3] = tmp_2039; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_2040 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_2040;let tmp_2041 = values[4]; values[4] = values[6]; values[6] = tmp_2041; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_2042 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_2042;let tmp_2043 = values[5]; values[5] = values[7]; values[7] = tmp_2043; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_2044 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_2044;let tmp_2045 = values[8]; values[8] = values[10]; values[10] = tmp_2045; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_2046 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_2046;let tmp_2047 = values[9]; values[9] = values[11]; values[11] = tmp_2047; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_2048 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_2048;let tmp_2049 = values[12]; values[12] = values[14]; values[14] = tmp_2049; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_2050 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_2050;let tmp_2051 = values[13]; values[13] = values[15]; values[15] = tmp_2051; }
    }
    // cmp_swap(16,18)
    if keys[16] > keys[18] || (keys[16] == keys[18] && values[16] > values[18]) {
    // swap(16,18) 
    { let tmp_2052 = keys[16]; keys[16] = keys[18]; keys[18] = tmp_2052;let tmp_2053 = values[16]; values[16] = values[18]; values[18] = tmp_2053; }
    }
    // cmp_swap(17,19)
    if keys[17] > keys[19] || (keys[17] == keys[19] && values[17] > values[19]) {
    // swap(17,19) 
    { let tmp_2054 = keys[17]; keys[17] = keys[19]; keys[19] = tmp_2054;let tmp_2055 = values[17]; values[17] = values[19]; values[19] = tmp_2055; }
    }
    // cmp_swap(20,22)
    if keys[20] > keys[22] || (keys[20] == keys[22] && values[20] > values[22]) {
    // swap(20,22) 
    { let tmp_2056 = keys[20]; keys[20] = keys[22]; keys[22] = tmp_2056;let tmp_2057 = values[20]; values[20] = values[22]; values[22] = tmp_2057; }
    }
    // cmp_swap(21,23)
    if keys[21] > keys[23] || (keys[21] == keys[23] && values[21] > values[23]) {
    // swap(21,23) 
    { let tmp_2058 = keys[21]; keys[21] = keys[23]; keys[23] = tmp_2058;let tmp_2059 = values[21]; values[21] = values[23]; values[23] = tmp_2059; }
    }
    // cmp_swap(24,26)
    if keys[24] > keys[26] || (keys[24] == keys[26] && values[24] > values[26]) {
    // swap(24,26) 
    { let tmp_2060 = keys[24]; keys[24] = keys[26]; keys[26] = tmp_2060;let tmp_2061 = values[24]; values[24] = values[26]; values[26] = tmp_2061; }
    }
    // cmp_swap(25,27)
    if keys[25] > keys[27] || (keys[25] == keys[27] && values[25] > values[27]) {
    // swap(25,27) 
    { let tmp_2062 = keys[25]; keys[25] = keys[27]; keys[27] = tmp_2062;let tmp_2063 = values[25]; values[25] = values[27]; values[27] = tmp_2063; }
    }
    // cmp_swap(28,30)
    if keys[28] > keys[30] || (keys[28] == keys[30] && values[28] > values[30]) {
    // swap(28,30) 
    { let tmp_2064 = keys[28]; keys[28] = keys[30]; keys[30] = tmp_2064;let tmp_2065 = values[28]; values[28] = values[30]; values[30] = tmp_2065; }
    }
    // cmp_swap(29,31)
    if keys[29] > keys[31] || (keys[29] == keys[31] && values[29] > values[31]) {
    // swap(29,31) 
    { let tmp_2066 = keys[29]; keys[29] = keys[31]; keys[31] = tmp_2066;let tmp_2067 = values[29]; values[29] = values[31]; values[31] = tmp_2067; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_2068 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_2068;let tmp_2069 = values[0]; values[0] = values[1]; values[1] = tmp_2069; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_2070 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_2070;let tmp_2071 = values[2]; values[2] = values[3]; values[3] = tmp_2071; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_2072 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_2072;let tmp_2073 = values[4]; values[4] = values[5]; values[5] = tmp_2073; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_2074 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_2074;let tmp_2075 = values[6]; values[6] = values[7]; values[7] = tmp_2075; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_2076 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_2076;let tmp_2077 = values[8]; values[8] = values[9]; values[9] = tmp_2077; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_2078 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_2078;let tmp_2079 = values[10]; values[10] = values[11]; values[11] = tmp_2079; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_2080 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_2080;let tmp_2081 = values[12]; values[12] = values[13]; values[13] = tmp_2081; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_2082 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_2082;let tmp_2083 = values[14]; values[14] = values[15]; values[15] = tmp_2083; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_2084 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_2084;let tmp_2085 = values[16]; values[16] = values[17]; values[17] = tmp_2085; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_2086 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_2086;let tmp_2087 = values[18]; values[18] = values[19]; values[19] = tmp_2087; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_2088 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_2088;let tmp_2089 = values[20]; values[20] = values[21]; values[21] = tmp_2089; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_2090 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_2090;let tmp_2091 = values[22]; values[22] = values[23]; values[23] = tmp_2091; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_2092 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_2092;let tmp_2093 = values[24]; values[24] = values[25]; values[25] = tmp_2093; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_2094 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_2094;let tmp_2095 = values[26]; values[26] = values[27]; values[27] = tmp_2095; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_2096 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_2096;let tmp_2097 = values[28]; values[28] = values[29]; values[29] = tmp_2097; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_2098 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_2098;let tmp_2099 = values[30]; values[30] = values[31]; values[31] = tmp_2099; }
    }
    // exch_intxn(tmask:31,swbit:4,wpt:32)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_2100 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_2101 = seg_base + (local_tid ^ 31u); let tmp_2102 = smem_keys[tmp_2101 * WPT + 31u]; let tmp_2103 = smem_vals[tmp_2101 * WPT + 31u]; let tmp_2104 = keys[0] < tmp_2102 || (keys[0] == tmp_2102 && values[0] < tmp_2103); if tmp_2100 == tmp_2104 { keys[0] = tmp_2102; values[0] = tmp_2103; } let tmp_2105 = smem_keys[tmp_2101 * WPT + 30u]; let tmp_2106 = smem_vals[tmp_2101 * WPT + 30u]; let tmp_2107 = keys[1] < tmp_2105 || (keys[1] == tmp_2105 && values[1] < tmp_2106); if tmp_2100 == tmp_2107 { keys[1] = tmp_2105; values[1] = tmp_2106; } let tmp_2108 = smem_keys[tmp_2101 * WPT + 29u]; let tmp_2109 = smem_vals[tmp_2101 * WPT + 29u]; let tmp_2110 = keys[2] < tmp_2108 || (keys[2] == tmp_2108 && values[2] < tmp_2109); if tmp_2100 == tmp_2110 { keys[2] = tmp_2108; values[2] = tmp_2109; } let tmp_2111 = smem_keys[tmp_2101 * WPT + 28u]; let tmp_2112 = smem_vals[tmp_2101 * WPT + 28u]; let tmp_2113 = keys[3] < tmp_2111 || (keys[3] == tmp_2111 && values[3] < tmp_2112); if tmp_2100 == tmp_2113 { keys[3] = tmp_2111; values[3] = tmp_2112; } let tmp_2114 = smem_keys[tmp_2101 * WPT + 27u]; let tmp_2115 = smem_vals[tmp_2101 * WPT + 27u]; let tmp_2116 = keys[4] < tmp_2114 || (keys[4] == tmp_2114 && values[4] < tmp_2115); if tmp_2100 == tmp_2116 { keys[4] = tmp_2114; values[4] = tmp_2115; } let tmp_2117 = smem_keys[tmp_2101 * WPT + 26u]; let tmp_2118 = smem_vals[tmp_2101 * WPT + 26u]; let tmp_2119 = keys[5] < tmp_2117 || (keys[5] == tmp_2117 && values[5] < tmp_2118); if tmp_2100 == tmp_2119 { keys[5] = tmp_2117; values[5] = tmp_2118; } let tmp_2120 = smem_keys[tmp_2101 * WPT + 25u]; let tmp_2121 = smem_vals[tmp_2101 * WPT + 25u]; let tmp_2122 = keys[6] < tmp_2120 || (keys[6] == tmp_2120 && values[6] < tmp_2121); if tmp_2100 == tmp_2122 { keys[6] = tmp_2120; values[6] = tmp_2121; } let tmp_2123 = smem_keys[tmp_2101 * WPT + 24u]; let tmp_2124 = smem_vals[tmp_2101 * WPT + 24u]; let tmp_2125 = keys[7] < tmp_2123 || (keys[7] == tmp_2123 && values[7] < tmp_2124); if tmp_2100 == tmp_2125 { keys[7] = tmp_2123; values[7] = tmp_2124; } let tmp_2126 = smem_keys[tmp_2101 * WPT + 23u]; let tmp_2127 = smem_vals[tmp_2101 * WPT + 23u]; let tmp_2128 = keys[8] < tmp_2126 || (keys[8] == tmp_2126 && values[8] < tmp_2127); if tmp_2100 == tmp_2128 { keys[8] = tmp_2126; values[8] = tmp_2127; } let tmp_2129 = smem_keys[tmp_2101 * WPT + 22u]; let tmp_2130 = smem_vals[tmp_2101 * WPT + 22u]; let tmp_2131 = keys[9] < tmp_2129 || (keys[9] == tmp_2129 && values[9] < tmp_2130); if tmp_2100 == tmp_2131 { keys[9] = tmp_2129; values[9] = tmp_2130; } let tmp_2132 = smem_keys[tmp_2101 * WPT + 21u]; let tmp_2133 = smem_vals[tmp_2101 * WPT + 21u]; let tmp_2134 = keys[10] < tmp_2132 || (keys[10] == tmp_2132 && values[10] < tmp_2133); if tmp_2100 == tmp_2134 { keys[10] = tmp_2132; values[10] = tmp_2133; } let tmp_2135 = smem_keys[tmp_2101 * WPT + 20u]; let tmp_2136 = smem_vals[tmp_2101 * WPT + 20u]; let tmp_2137 = keys[11] < tmp_2135 || (keys[11] == tmp_2135 && values[11] < tmp_2136); if tmp_2100 == tmp_2137 { keys[11] = tmp_2135; values[11] = tmp_2136; } let tmp_2138 = smem_keys[tmp_2101 * WPT + 19u]; let tmp_2139 = smem_vals[tmp_2101 * WPT + 19u]; let tmp_2140 = keys[12] < tmp_2138 || (keys[12] == tmp_2138 && values[12] < tmp_2139); if tmp_2100 == tmp_2140 { keys[12] = tmp_2138; values[12] = tmp_2139; } let tmp_2141 = smem_keys[tmp_2101 * WPT + 18u]; let tmp_2142 = smem_vals[tmp_2101 * WPT + 18u]; let tmp_2143 = keys[13] < tmp_2141 || (keys[13] == tmp_2141 && values[13] < tmp_2142); if tmp_2100 == tmp_2143 { keys[13] = tmp_2141; values[13] = tmp_2142; } let tmp_2144 = smem_keys[tmp_2101 * WPT + 17u]; let tmp_2145 = smem_vals[tmp_2101 * WPT + 17u]; let tmp_2146 = keys[14] < tmp_2144 || (keys[14] == tmp_2144 && values[14] < tmp_2145); if tmp_2100 == tmp_2146 { keys[14] = tmp_2144; values[14] = tmp_2145; } let tmp_2147 = smem_keys[tmp_2101 * WPT + 16u]; let tmp_2148 = smem_vals[tmp_2101 * WPT + 16u]; let tmp_2149 = keys[15] < tmp_2147 || (keys[15] == tmp_2147 && values[15] < tmp_2148); if tmp_2100 == tmp_2149 { keys[15] = tmp_2147; values[15] = tmp_2148; } let tmp_2150 = smem_keys[tmp_2101 * WPT + 15u]; let tmp_2151 = smem_vals[tmp_2101 * WPT + 15u]; let tmp_2152 = keys[16] < tmp_2150 || (keys[16] == tmp_2150 && values[16] < tmp_2151); if tmp_2100 == tmp_2152 { keys[16] = tmp_2150; values[16] = tmp_2151; } let tmp_2153 = smem_keys[tmp_2101 * WPT + 14u]; let tmp_2154 = smem_vals[tmp_2101 * WPT + 14u]; let tmp_2155 = keys[17] < tmp_2153 || (keys[17] == tmp_2153 && values[17] < tmp_2154); if tmp_2100 == tmp_2155 { keys[17] = tmp_2153; values[17] = tmp_2154; } let tmp_2156 = smem_keys[tmp_2101 * WPT + 13u]; let tmp_2157 = smem_vals[tmp_2101 * WPT + 13u]; let tmp_2158 = keys[18] < tmp_2156 || (keys[18] == tmp_2156 && values[18] < tmp_2157); if tmp_2100 == tmp_2158 { keys[18] = tmp_2156; values[18] = tmp_2157; } let tmp_2159 = smem_keys[tmp_2101 * WPT + 12u]; let tmp_2160 = smem_vals[tmp_2101 * WPT + 12u]; let tmp_2161 = keys[19] < tmp_2159 || (keys[19] == tmp_2159 && values[19] < tmp_2160); if tmp_2100 == tmp_2161 { keys[19] = tmp_2159; values[19] = tmp_2160; } let tmp_2162 = smem_keys[tmp_2101 * WPT + 11u]; let tmp_2163 = smem_vals[tmp_2101 * WPT + 11u]; let tmp_2164 = keys[20] < tmp_2162 || (keys[20] == tmp_2162 && values[20] < tmp_2163); if tmp_2100 == tmp_2164 { keys[20] = tmp_2162; values[20] = tmp_2163; } let tmp_2165 = smem_keys[tmp_2101 * WPT + 10u]; let tmp_2166 = smem_vals[tmp_2101 * WPT + 10u]; let tmp_2167 = keys[21] < tmp_2165 || (keys[21] == tmp_2165 && values[21] < tmp_2166); if tmp_2100 == tmp_2167 { keys[21] = tmp_2165; values[21] = tmp_2166; } let tmp_2168 = smem_keys[tmp_2101 * WPT + 9u]; let tmp_2169 = smem_vals[tmp_2101 * WPT + 9u]; let tmp_2170 = keys[22] < tmp_2168 || (keys[22] == tmp_2168 && values[22] < tmp_2169); if tmp_2100 == tmp_2170 { keys[22] = tmp_2168; values[22] = tmp_2169; } let tmp_2171 = smem_keys[tmp_2101 * WPT + 8u]; let tmp_2172 = smem_vals[tmp_2101 * WPT + 8u]; let tmp_2173 = keys[23] < tmp_2171 || (keys[23] == tmp_2171 && values[23] < tmp_2172); if tmp_2100 == tmp_2173 { keys[23] = tmp_2171; values[23] = tmp_2172; } let tmp_2174 = smem_keys[tmp_2101 * WPT + 7u]; let tmp_2175 = smem_vals[tmp_2101 * WPT + 7u]; let tmp_2176 = keys[24] < tmp_2174 || (keys[24] == tmp_2174 && values[24] < tmp_2175); if tmp_2100 == tmp_2176 { keys[24] = tmp_2174; values[24] = tmp_2175; } let tmp_2177 = smem_keys[tmp_2101 * WPT + 6u]; let tmp_2178 = smem_vals[tmp_2101 * WPT + 6u]; let tmp_2179 = keys[25] < tmp_2177 || (keys[25] == tmp_2177 && values[25] < tmp_2178); if tmp_2100 == tmp_2179 { keys[25] = tmp_2177; values[25] = tmp_2178; } let tmp_2180 = smem_keys[tmp_2101 * WPT + 5u]; let tmp_2181 = smem_vals[tmp_2101 * WPT + 5u]; let tmp_2182 = keys[26] < tmp_2180 || (keys[26] == tmp_2180 && values[26] < tmp_2181); if tmp_2100 == tmp_2182 { keys[26] = tmp_2180; values[26] = tmp_2181; } let tmp_2183 = smem_keys[tmp_2101 * WPT + 4u]; let tmp_2184 = smem_vals[tmp_2101 * WPT + 4u]; let tmp_2185 = keys[27] < tmp_2183 || (keys[27] == tmp_2183 && values[27] < tmp_2184); if tmp_2100 == tmp_2185 { keys[27] = tmp_2183; values[27] = tmp_2184; } let tmp_2186 = smem_keys[tmp_2101 * WPT + 3u]; let tmp_2187 = smem_vals[tmp_2101 * WPT + 3u]; let tmp_2188 = keys[28] < tmp_2186 || (keys[28] == tmp_2186 && values[28] < tmp_2187); if tmp_2100 == tmp_2188 { keys[28] = tmp_2186; values[28] = tmp_2187; } let tmp_2189 = smem_keys[tmp_2101 * WPT + 2u]; let tmp_2190 = smem_vals[tmp_2101 * WPT + 2u]; let tmp_2191 = keys[29] < tmp_2189 || (keys[29] == tmp_2189 && values[29] < tmp_2190); if tmp_2100 == tmp_2191 { keys[29] = tmp_2189; values[29] = tmp_2190; } let tmp_2192 = smem_keys[tmp_2101 * WPT + 1u]; let tmp_2193 = smem_vals[tmp_2101 * WPT + 1u]; let tmp_2194 = keys[30] < tmp_2192 || (keys[30] == tmp_2192 && values[30] < tmp_2193); if tmp_2100 == tmp_2194 { keys[30] = tmp_2192; values[30] = tmp_2193; } let tmp_2195 = smem_keys[tmp_2101 * WPT + 0u]; let tmp_2196 = smem_vals[tmp_2101 * WPT + 0u]; let tmp_2197 = keys[31] < tmp_2195 || (keys[31] == tmp_2195 && values[31] < tmp_2196); if tmp_2100 == tmp_2197 { keys[31] = tmp_2195; values[31] = tmp_2196; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_2198 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_2199 = seg_base + (local_tid ^ 8u); let tmp_2200 = smem_keys[tmp_2199 * WPT + 0u]; let tmp_2201 = smem_vals[tmp_2199 * WPT + 0u]; let tmp_2202 = keys[0] < tmp_2200 || (keys[0] == tmp_2200 && values[0] < tmp_2201); if tmp_2198 == tmp_2202 { keys[0] = tmp_2200; values[0] = tmp_2201; } let tmp_2203 = smem_keys[tmp_2199 * WPT + 1u]; let tmp_2204 = smem_vals[tmp_2199 * WPT + 1u]; let tmp_2205 = keys[1] < tmp_2203 || (keys[1] == tmp_2203 && values[1] < tmp_2204); if tmp_2198 == tmp_2205 { keys[1] = tmp_2203; values[1] = tmp_2204; } let tmp_2206 = smem_keys[tmp_2199 * WPT + 2u]; let tmp_2207 = smem_vals[tmp_2199 * WPT + 2u]; let tmp_2208 = keys[2] < tmp_2206 || (keys[2] == tmp_2206 && values[2] < tmp_2207); if tmp_2198 == tmp_2208 { keys[2] = tmp_2206; values[2] = tmp_2207; } let tmp_2209 = smem_keys[tmp_2199 * WPT + 3u]; let tmp_2210 = smem_vals[tmp_2199 * WPT + 3u]; let tmp_2211 = keys[3] < tmp_2209 || (keys[3] == tmp_2209 && values[3] < tmp_2210); if tmp_2198 == tmp_2211 { keys[3] = tmp_2209; values[3] = tmp_2210; } let tmp_2212 = smem_keys[tmp_2199 * WPT + 4u]; let tmp_2213 = smem_vals[tmp_2199 * WPT + 4u]; let tmp_2214 = keys[4] < tmp_2212 || (keys[4] == tmp_2212 && values[4] < tmp_2213); if tmp_2198 == tmp_2214 { keys[4] = tmp_2212; values[4] = tmp_2213; } let tmp_2215 = smem_keys[tmp_2199 * WPT + 5u]; let tmp_2216 = smem_vals[tmp_2199 * WPT + 5u]; let tmp_2217 = keys[5] < tmp_2215 || (keys[5] == tmp_2215 && values[5] < tmp_2216); if tmp_2198 == tmp_2217 { keys[5] = tmp_2215; values[5] = tmp_2216; } let tmp_2218 = smem_keys[tmp_2199 * WPT + 6u]; let tmp_2219 = smem_vals[tmp_2199 * WPT + 6u]; let tmp_2220 = keys[6] < tmp_2218 || (keys[6] == tmp_2218 && values[6] < tmp_2219); if tmp_2198 == tmp_2220 { keys[6] = tmp_2218; values[6] = tmp_2219; } let tmp_2221 = smem_keys[tmp_2199 * WPT + 7u]; let tmp_2222 = smem_vals[tmp_2199 * WPT + 7u]; let tmp_2223 = keys[7] < tmp_2221 || (keys[7] == tmp_2221 && values[7] < tmp_2222); if tmp_2198 == tmp_2223 { keys[7] = tmp_2221; values[7] = tmp_2222; } let tmp_2224 = smem_keys[tmp_2199 * WPT + 8u]; let tmp_2225 = smem_vals[tmp_2199 * WPT + 8u]; let tmp_2226 = keys[8] < tmp_2224 || (keys[8] == tmp_2224 && values[8] < tmp_2225); if tmp_2198 == tmp_2226 { keys[8] = tmp_2224; values[8] = tmp_2225; } let tmp_2227 = smem_keys[tmp_2199 * WPT + 9u]; let tmp_2228 = smem_vals[tmp_2199 * WPT + 9u]; let tmp_2229 = keys[9] < tmp_2227 || (keys[9] == tmp_2227 && values[9] < tmp_2228); if tmp_2198 == tmp_2229 { keys[9] = tmp_2227; values[9] = tmp_2228; } let tmp_2230 = smem_keys[tmp_2199 * WPT + 10u]; let tmp_2231 = smem_vals[tmp_2199 * WPT + 10u]; let tmp_2232 = keys[10] < tmp_2230 || (keys[10] == tmp_2230 && values[10] < tmp_2231); if tmp_2198 == tmp_2232 { keys[10] = tmp_2230; values[10] = tmp_2231; } let tmp_2233 = smem_keys[tmp_2199 * WPT + 11u]; let tmp_2234 = smem_vals[tmp_2199 * WPT + 11u]; let tmp_2235 = keys[11] < tmp_2233 || (keys[11] == tmp_2233 && values[11] < tmp_2234); if tmp_2198 == tmp_2235 { keys[11] = tmp_2233; values[11] = tmp_2234; } let tmp_2236 = smem_keys[tmp_2199 * WPT + 12u]; let tmp_2237 = smem_vals[tmp_2199 * WPT + 12u]; let tmp_2238 = keys[12] < tmp_2236 || (keys[12] == tmp_2236 && values[12] < tmp_2237); if tmp_2198 == tmp_2238 { keys[12] = tmp_2236; values[12] = tmp_2237; } let tmp_2239 = smem_keys[tmp_2199 * WPT + 13u]; let tmp_2240 = smem_vals[tmp_2199 * WPT + 13u]; let tmp_2241 = keys[13] < tmp_2239 || (keys[13] == tmp_2239 && values[13] < tmp_2240); if tmp_2198 == tmp_2241 { keys[13] = tmp_2239; values[13] = tmp_2240; } let tmp_2242 = smem_keys[tmp_2199 * WPT + 14u]; let tmp_2243 = smem_vals[tmp_2199 * WPT + 14u]; let tmp_2244 = keys[14] < tmp_2242 || (keys[14] == tmp_2242 && values[14] < tmp_2243); if tmp_2198 == tmp_2244 { keys[14] = tmp_2242; values[14] = tmp_2243; } let tmp_2245 = smem_keys[tmp_2199 * WPT + 15u]; let tmp_2246 = smem_vals[tmp_2199 * WPT + 15u]; let tmp_2247 = keys[15] < tmp_2245 || (keys[15] == tmp_2245 && values[15] < tmp_2246); if tmp_2198 == tmp_2247 { keys[15] = tmp_2245; values[15] = tmp_2246; } let tmp_2248 = smem_keys[tmp_2199 * WPT + 16u]; let tmp_2249 = smem_vals[tmp_2199 * WPT + 16u]; let tmp_2250 = keys[16] < tmp_2248 || (keys[16] == tmp_2248 && values[16] < tmp_2249); if tmp_2198 == tmp_2250 { keys[16] = tmp_2248; values[16] = tmp_2249; } let tmp_2251 = smem_keys[tmp_2199 * WPT + 17u]; let tmp_2252 = smem_vals[tmp_2199 * WPT + 17u]; let tmp_2253 = keys[17] < tmp_2251 || (keys[17] == tmp_2251 && values[17] < tmp_2252); if tmp_2198 == tmp_2253 { keys[17] = tmp_2251; values[17] = tmp_2252; } let tmp_2254 = smem_keys[tmp_2199 * WPT + 18u]; let tmp_2255 = smem_vals[tmp_2199 * WPT + 18u]; let tmp_2256 = keys[18] < tmp_2254 || (keys[18] == tmp_2254 && values[18] < tmp_2255); if tmp_2198 == tmp_2256 { keys[18] = tmp_2254; values[18] = tmp_2255; } let tmp_2257 = smem_keys[tmp_2199 * WPT + 19u]; let tmp_2258 = smem_vals[tmp_2199 * WPT + 19u]; let tmp_2259 = keys[19] < tmp_2257 || (keys[19] == tmp_2257 && values[19] < tmp_2258); if tmp_2198 == tmp_2259 { keys[19] = tmp_2257; values[19] = tmp_2258; } let tmp_2260 = smem_keys[tmp_2199 * WPT + 20u]; let tmp_2261 = smem_vals[tmp_2199 * WPT + 20u]; let tmp_2262 = keys[20] < tmp_2260 || (keys[20] == tmp_2260 && values[20] < tmp_2261); if tmp_2198 == tmp_2262 { keys[20] = tmp_2260; values[20] = tmp_2261; } let tmp_2263 = smem_keys[tmp_2199 * WPT + 21u]; let tmp_2264 = smem_vals[tmp_2199 * WPT + 21u]; let tmp_2265 = keys[21] < tmp_2263 || (keys[21] == tmp_2263 && values[21] < tmp_2264); if tmp_2198 == tmp_2265 { keys[21] = tmp_2263; values[21] = tmp_2264; } let tmp_2266 = smem_keys[tmp_2199 * WPT + 22u]; let tmp_2267 = smem_vals[tmp_2199 * WPT + 22u]; let tmp_2268 = keys[22] < tmp_2266 || (keys[22] == tmp_2266 && values[22] < tmp_2267); if tmp_2198 == tmp_2268 { keys[22] = tmp_2266; values[22] = tmp_2267; } let tmp_2269 = smem_keys[tmp_2199 * WPT + 23u]; let tmp_2270 = smem_vals[tmp_2199 * WPT + 23u]; let tmp_2271 = keys[23] < tmp_2269 || (keys[23] == tmp_2269 && values[23] < tmp_2270); if tmp_2198 == tmp_2271 { keys[23] = tmp_2269; values[23] = tmp_2270; } let tmp_2272 = smem_keys[tmp_2199 * WPT + 24u]; let tmp_2273 = smem_vals[tmp_2199 * WPT + 24u]; let tmp_2274 = keys[24] < tmp_2272 || (keys[24] == tmp_2272 && values[24] < tmp_2273); if tmp_2198 == tmp_2274 { keys[24] = tmp_2272; values[24] = tmp_2273; } let tmp_2275 = smem_keys[tmp_2199 * WPT + 25u]; let tmp_2276 = smem_vals[tmp_2199 * WPT + 25u]; let tmp_2277 = keys[25] < tmp_2275 || (keys[25] == tmp_2275 && values[25] < tmp_2276); if tmp_2198 == tmp_2277 { keys[25] = tmp_2275; values[25] = tmp_2276; } let tmp_2278 = smem_keys[tmp_2199 * WPT + 26u]; let tmp_2279 = smem_vals[tmp_2199 * WPT + 26u]; let tmp_2280 = keys[26] < tmp_2278 || (keys[26] == tmp_2278 && values[26] < tmp_2279); if tmp_2198 == tmp_2280 { keys[26] = tmp_2278; values[26] = tmp_2279; } let tmp_2281 = smem_keys[tmp_2199 * WPT + 27u]; let tmp_2282 = smem_vals[tmp_2199 * WPT + 27u]; let tmp_2283 = keys[27] < tmp_2281 || (keys[27] == tmp_2281 && values[27] < tmp_2282); if tmp_2198 == tmp_2283 { keys[27] = tmp_2281; values[27] = tmp_2282; } let tmp_2284 = smem_keys[tmp_2199 * WPT + 28u]; let tmp_2285 = smem_vals[tmp_2199 * WPT + 28u]; let tmp_2286 = keys[28] < tmp_2284 || (keys[28] == tmp_2284 && values[28] < tmp_2285); if tmp_2198 == tmp_2286 { keys[28] = tmp_2284; values[28] = tmp_2285; } let tmp_2287 = smem_keys[tmp_2199 * WPT + 29u]; let tmp_2288 = smem_vals[tmp_2199 * WPT + 29u]; let tmp_2289 = keys[29] < tmp_2287 || (keys[29] == tmp_2287 && values[29] < tmp_2288); if tmp_2198 == tmp_2289 { keys[29] = tmp_2287; values[29] = tmp_2288; } let tmp_2290 = smem_keys[tmp_2199 * WPT + 30u]; let tmp_2291 = smem_vals[tmp_2199 * WPT + 30u]; let tmp_2292 = keys[30] < tmp_2290 || (keys[30] == tmp_2290 && values[30] < tmp_2291); if tmp_2198 == tmp_2292 { keys[30] = tmp_2290; values[30] = tmp_2291; } let tmp_2293 = smem_keys[tmp_2199 * WPT + 31u]; let tmp_2294 = smem_vals[tmp_2199 * WPT + 31u]; let tmp_2295 = keys[31] < tmp_2293 || (keys[31] == tmp_2293 && values[31] < tmp_2294); if tmp_2198 == tmp_2295 { keys[31] = tmp_2293; values[31] = tmp_2294; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_2296 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_2297 = seg_base + (local_tid ^ 4u); let tmp_2298 = smem_keys[tmp_2297 * WPT + 0u]; let tmp_2299 = smem_vals[tmp_2297 * WPT + 0u]; let tmp_2300 = keys[0] < tmp_2298 || (keys[0] == tmp_2298 && values[0] < tmp_2299); if tmp_2296 == tmp_2300 { keys[0] = tmp_2298; values[0] = tmp_2299; } let tmp_2301 = smem_keys[tmp_2297 * WPT + 1u]; let tmp_2302 = smem_vals[tmp_2297 * WPT + 1u]; let tmp_2303 = keys[1] < tmp_2301 || (keys[1] == tmp_2301 && values[1] < tmp_2302); if tmp_2296 == tmp_2303 { keys[1] = tmp_2301; values[1] = tmp_2302; } let tmp_2304 = smem_keys[tmp_2297 * WPT + 2u]; let tmp_2305 = smem_vals[tmp_2297 * WPT + 2u]; let tmp_2306 = keys[2] < tmp_2304 || (keys[2] == tmp_2304 && values[2] < tmp_2305); if tmp_2296 == tmp_2306 { keys[2] = tmp_2304; values[2] = tmp_2305; } let tmp_2307 = smem_keys[tmp_2297 * WPT + 3u]; let tmp_2308 = smem_vals[tmp_2297 * WPT + 3u]; let tmp_2309 = keys[3] < tmp_2307 || (keys[3] == tmp_2307 && values[3] < tmp_2308); if tmp_2296 == tmp_2309 { keys[3] = tmp_2307; values[3] = tmp_2308; } let tmp_2310 = smem_keys[tmp_2297 * WPT + 4u]; let tmp_2311 = smem_vals[tmp_2297 * WPT + 4u]; let tmp_2312 = keys[4] < tmp_2310 || (keys[4] == tmp_2310 && values[4] < tmp_2311); if tmp_2296 == tmp_2312 { keys[4] = tmp_2310; values[4] = tmp_2311; } let tmp_2313 = smem_keys[tmp_2297 * WPT + 5u]; let tmp_2314 = smem_vals[tmp_2297 * WPT + 5u]; let tmp_2315 = keys[5] < tmp_2313 || (keys[5] == tmp_2313 && values[5] < tmp_2314); if tmp_2296 == tmp_2315 { keys[5] = tmp_2313; values[5] = tmp_2314; } let tmp_2316 = smem_keys[tmp_2297 * WPT + 6u]; let tmp_2317 = smem_vals[tmp_2297 * WPT + 6u]; let tmp_2318 = keys[6] < tmp_2316 || (keys[6] == tmp_2316 && values[6] < tmp_2317); if tmp_2296 == tmp_2318 { keys[6] = tmp_2316; values[6] = tmp_2317; } let tmp_2319 = smem_keys[tmp_2297 * WPT + 7u]; let tmp_2320 = smem_vals[tmp_2297 * WPT + 7u]; let tmp_2321 = keys[7] < tmp_2319 || (keys[7] == tmp_2319 && values[7] < tmp_2320); if tmp_2296 == tmp_2321 { keys[7] = tmp_2319; values[7] = tmp_2320; } let tmp_2322 = smem_keys[tmp_2297 * WPT + 8u]; let tmp_2323 = smem_vals[tmp_2297 * WPT + 8u]; let tmp_2324 = keys[8] < tmp_2322 || (keys[8] == tmp_2322 && values[8] < tmp_2323); if tmp_2296 == tmp_2324 { keys[8] = tmp_2322; values[8] = tmp_2323; } let tmp_2325 = smem_keys[tmp_2297 * WPT + 9u]; let tmp_2326 = smem_vals[tmp_2297 * WPT + 9u]; let tmp_2327 = keys[9] < tmp_2325 || (keys[9] == tmp_2325 && values[9] < tmp_2326); if tmp_2296 == tmp_2327 { keys[9] = tmp_2325; values[9] = tmp_2326; } let tmp_2328 = smem_keys[tmp_2297 * WPT + 10u]; let tmp_2329 = smem_vals[tmp_2297 * WPT + 10u]; let tmp_2330 = keys[10] < tmp_2328 || (keys[10] == tmp_2328 && values[10] < tmp_2329); if tmp_2296 == tmp_2330 { keys[10] = tmp_2328; values[10] = tmp_2329; } let tmp_2331 = smem_keys[tmp_2297 * WPT + 11u]; let tmp_2332 = smem_vals[tmp_2297 * WPT + 11u]; let tmp_2333 = keys[11] < tmp_2331 || (keys[11] == tmp_2331 && values[11] < tmp_2332); if tmp_2296 == tmp_2333 { keys[11] = tmp_2331; values[11] = tmp_2332; } let tmp_2334 = smem_keys[tmp_2297 * WPT + 12u]; let tmp_2335 = smem_vals[tmp_2297 * WPT + 12u]; let tmp_2336 = keys[12] < tmp_2334 || (keys[12] == tmp_2334 && values[12] < tmp_2335); if tmp_2296 == tmp_2336 { keys[12] = tmp_2334; values[12] = tmp_2335; } let tmp_2337 = smem_keys[tmp_2297 * WPT + 13u]; let tmp_2338 = smem_vals[tmp_2297 * WPT + 13u]; let tmp_2339 = keys[13] < tmp_2337 || (keys[13] == tmp_2337 && values[13] < tmp_2338); if tmp_2296 == tmp_2339 { keys[13] = tmp_2337; values[13] = tmp_2338; } let tmp_2340 = smem_keys[tmp_2297 * WPT + 14u]; let tmp_2341 = smem_vals[tmp_2297 * WPT + 14u]; let tmp_2342 = keys[14] < tmp_2340 || (keys[14] == tmp_2340 && values[14] < tmp_2341); if tmp_2296 == tmp_2342 { keys[14] = tmp_2340; values[14] = tmp_2341; } let tmp_2343 = smem_keys[tmp_2297 * WPT + 15u]; let tmp_2344 = smem_vals[tmp_2297 * WPT + 15u]; let tmp_2345 = keys[15] < tmp_2343 || (keys[15] == tmp_2343 && values[15] < tmp_2344); if tmp_2296 == tmp_2345 { keys[15] = tmp_2343; values[15] = tmp_2344; } let tmp_2346 = smem_keys[tmp_2297 * WPT + 16u]; let tmp_2347 = smem_vals[tmp_2297 * WPT + 16u]; let tmp_2348 = keys[16] < tmp_2346 || (keys[16] == tmp_2346 && values[16] < tmp_2347); if tmp_2296 == tmp_2348 { keys[16] = tmp_2346; values[16] = tmp_2347; } let tmp_2349 = smem_keys[tmp_2297 * WPT + 17u]; let tmp_2350 = smem_vals[tmp_2297 * WPT + 17u]; let tmp_2351 = keys[17] < tmp_2349 || (keys[17] == tmp_2349 && values[17] < tmp_2350); if tmp_2296 == tmp_2351 { keys[17] = tmp_2349; values[17] = tmp_2350; } let tmp_2352 = smem_keys[tmp_2297 * WPT + 18u]; let tmp_2353 = smem_vals[tmp_2297 * WPT + 18u]; let tmp_2354 = keys[18] < tmp_2352 || (keys[18] == tmp_2352 && values[18] < tmp_2353); if tmp_2296 == tmp_2354 { keys[18] = tmp_2352; values[18] = tmp_2353; } let tmp_2355 = smem_keys[tmp_2297 * WPT + 19u]; let tmp_2356 = smem_vals[tmp_2297 * WPT + 19u]; let tmp_2357 = keys[19] < tmp_2355 || (keys[19] == tmp_2355 && values[19] < tmp_2356); if tmp_2296 == tmp_2357 { keys[19] = tmp_2355; values[19] = tmp_2356; } let tmp_2358 = smem_keys[tmp_2297 * WPT + 20u]; let tmp_2359 = smem_vals[tmp_2297 * WPT + 20u]; let tmp_2360 = keys[20] < tmp_2358 || (keys[20] == tmp_2358 && values[20] < tmp_2359); if tmp_2296 == tmp_2360 { keys[20] = tmp_2358; values[20] = tmp_2359; } let tmp_2361 = smem_keys[tmp_2297 * WPT + 21u]; let tmp_2362 = smem_vals[tmp_2297 * WPT + 21u]; let tmp_2363 = keys[21] < tmp_2361 || (keys[21] == tmp_2361 && values[21] < tmp_2362); if tmp_2296 == tmp_2363 { keys[21] = tmp_2361; values[21] = tmp_2362; } let tmp_2364 = smem_keys[tmp_2297 * WPT + 22u]; let tmp_2365 = smem_vals[tmp_2297 * WPT + 22u]; let tmp_2366 = keys[22] < tmp_2364 || (keys[22] == tmp_2364 && values[22] < tmp_2365); if tmp_2296 == tmp_2366 { keys[22] = tmp_2364; values[22] = tmp_2365; } let tmp_2367 = smem_keys[tmp_2297 * WPT + 23u]; let tmp_2368 = smem_vals[tmp_2297 * WPT + 23u]; let tmp_2369 = keys[23] < tmp_2367 || (keys[23] == tmp_2367 && values[23] < tmp_2368); if tmp_2296 == tmp_2369 { keys[23] = tmp_2367; values[23] = tmp_2368; } let tmp_2370 = smem_keys[tmp_2297 * WPT + 24u]; let tmp_2371 = smem_vals[tmp_2297 * WPT + 24u]; let tmp_2372 = keys[24] < tmp_2370 || (keys[24] == tmp_2370 && values[24] < tmp_2371); if tmp_2296 == tmp_2372 { keys[24] = tmp_2370; values[24] = tmp_2371; } let tmp_2373 = smem_keys[tmp_2297 * WPT + 25u]; let tmp_2374 = smem_vals[tmp_2297 * WPT + 25u]; let tmp_2375 = keys[25] < tmp_2373 || (keys[25] == tmp_2373 && values[25] < tmp_2374); if tmp_2296 == tmp_2375 { keys[25] = tmp_2373; values[25] = tmp_2374; } let tmp_2376 = smem_keys[tmp_2297 * WPT + 26u]; let tmp_2377 = smem_vals[tmp_2297 * WPT + 26u]; let tmp_2378 = keys[26] < tmp_2376 || (keys[26] == tmp_2376 && values[26] < tmp_2377); if tmp_2296 == tmp_2378 { keys[26] = tmp_2376; values[26] = tmp_2377; } let tmp_2379 = smem_keys[tmp_2297 * WPT + 27u]; let tmp_2380 = smem_vals[tmp_2297 * WPT + 27u]; let tmp_2381 = keys[27] < tmp_2379 || (keys[27] == tmp_2379 && values[27] < tmp_2380); if tmp_2296 == tmp_2381 { keys[27] = tmp_2379; values[27] = tmp_2380; } let tmp_2382 = smem_keys[tmp_2297 * WPT + 28u]; let tmp_2383 = smem_vals[tmp_2297 * WPT + 28u]; let tmp_2384 = keys[28] < tmp_2382 || (keys[28] == tmp_2382 && values[28] < tmp_2383); if tmp_2296 == tmp_2384 { keys[28] = tmp_2382; values[28] = tmp_2383; } let tmp_2385 = smem_keys[tmp_2297 * WPT + 29u]; let tmp_2386 = smem_vals[tmp_2297 * WPT + 29u]; let tmp_2387 = keys[29] < tmp_2385 || (keys[29] == tmp_2385 && values[29] < tmp_2386); if tmp_2296 == tmp_2387 { keys[29] = tmp_2385; values[29] = tmp_2386; } let tmp_2388 = smem_keys[tmp_2297 * WPT + 30u]; let tmp_2389 = smem_vals[tmp_2297 * WPT + 30u]; let tmp_2390 = keys[30] < tmp_2388 || (keys[30] == tmp_2388 && values[30] < tmp_2389); if tmp_2296 == tmp_2390 { keys[30] = tmp_2388; values[30] = tmp_2389; } let tmp_2391 = smem_keys[tmp_2297 * WPT + 31u]; let tmp_2392 = smem_vals[tmp_2297 * WPT + 31u]; let tmp_2393 = keys[31] < tmp_2391 || (keys[31] == tmp_2391 && values[31] < tmp_2392); if tmp_2296 == tmp_2393 { keys[31] = tmp_2391; values[31] = tmp_2392; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_2394 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_2395 = seg_base + (local_tid ^ 2u); let tmp_2396 = smem_keys[tmp_2395 * WPT + 0u]; let tmp_2397 = smem_vals[tmp_2395 * WPT + 0u]; let tmp_2398 = keys[0] < tmp_2396 || (keys[0] == tmp_2396 && values[0] < tmp_2397); if tmp_2394 == tmp_2398 { keys[0] = tmp_2396; values[0] = tmp_2397; } let tmp_2399 = smem_keys[tmp_2395 * WPT + 1u]; let tmp_2400 = smem_vals[tmp_2395 * WPT + 1u]; let tmp_2401 = keys[1] < tmp_2399 || (keys[1] == tmp_2399 && values[1] < tmp_2400); if tmp_2394 == tmp_2401 { keys[1] = tmp_2399; values[1] = tmp_2400; } let tmp_2402 = smem_keys[tmp_2395 * WPT + 2u]; let tmp_2403 = smem_vals[tmp_2395 * WPT + 2u]; let tmp_2404 = keys[2] < tmp_2402 || (keys[2] == tmp_2402 && values[2] < tmp_2403); if tmp_2394 == tmp_2404 { keys[2] = tmp_2402; values[2] = tmp_2403; } let tmp_2405 = smem_keys[tmp_2395 * WPT + 3u]; let tmp_2406 = smem_vals[tmp_2395 * WPT + 3u]; let tmp_2407 = keys[3] < tmp_2405 || (keys[3] == tmp_2405 && values[3] < tmp_2406); if tmp_2394 == tmp_2407 { keys[3] = tmp_2405; values[3] = tmp_2406; } let tmp_2408 = smem_keys[tmp_2395 * WPT + 4u]; let tmp_2409 = smem_vals[tmp_2395 * WPT + 4u]; let tmp_2410 = keys[4] < tmp_2408 || (keys[4] == tmp_2408 && values[4] < tmp_2409); if tmp_2394 == tmp_2410 { keys[4] = tmp_2408; values[4] = tmp_2409; } let tmp_2411 = smem_keys[tmp_2395 * WPT + 5u]; let tmp_2412 = smem_vals[tmp_2395 * WPT + 5u]; let tmp_2413 = keys[5] < tmp_2411 || (keys[5] == tmp_2411 && values[5] < tmp_2412); if tmp_2394 == tmp_2413 { keys[5] = tmp_2411; values[5] = tmp_2412; } let tmp_2414 = smem_keys[tmp_2395 * WPT + 6u]; let tmp_2415 = smem_vals[tmp_2395 * WPT + 6u]; let tmp_2416 = keys[6] < tmp_2414 || (keys[6] == tmp_2414 && values[6] < tmp_2415); if tmp_2394 == tmp_2416 { keys[6] = tmp_2414; values[6] = tmp_2415; } let tmp_2417 = smem_keys[tmp_2395 * WPT + 7u]; let tmp_2418 = smem_vals[tmp_2395 * WPT + 7u]; let tmp_2419 = keys[7] < tmp_2417 || (keys[7] == tmp_2417 && values[7] < tmp_2418); if tmp_2394 == tmp_2419 { keys[7] = tmp_2417; values[7] = tmp_2418; } let tmp_2420 = smem_keys[tmp_2395 * WPT + 8u]; let tmp_2421 = smem_vals[tmp_2395 * WPT + 8u]; let tmp_2422 = keys[8] < tmp_2420 || (keys[8] == tmp_2420 && values[8] < tmp_2421); if tmp_2394 == tmp_2422 { keys[8] = tmp_2420; values[8] = tmp_2421; } let tmp_2423 = smem_keys[tmp_2395 * WPT + 9u]; let tmp_2424 = smem_vals[tmp_2395 * WPT + 9u]; let tmp_2425 = keys[9] < tmp_2423 || (keys[9] == tmp_2423 && values[9] < tmp_2424); if tmp_2394 == tmp_2425 { keys[9] = tmp_2423; values[9] = tmp_2424; } let tmp_2426 = smem_keys[tmp_2395 * WPT + 10u]; let tmp_2427 = smem_vals[tmp_2395 * WPT + 10u]; let tmp_2428 = keys[10] < tmp_2426 || (keys[10] == tmp_2426 && values[10] < tmp_2427); if tmp_2394 == tmp_2428 { keys[10] = tmp_2426; values[10] = tmp_2427; } let tmp_2429 = smem_keys[tmp_2395 * WPT + 11u]; let tmp_2430 = smem_vals[tmp_2395 * WPT + 11u]; let tmp_2431 = keys[11] < tmp_2429 || (keys[11] == tmp_2429 && values[11] < tmp_2430); if tmp_2394 == tmp_2431 { keys[11] = tmp_2429; values[11] = tmp_2430; } let tmp_2432 = smem_keys[tmp_2395 * WPT + 12u]; let tmp_2433 = smem_vals[tmp_2395 * WPT + 12u]; let tmp_2434 = keys[12] < tmp_2432 || (keys[12] == tmp_2432 && values[12] < tmp_2433); if tmp_2394 == tmp_2434 { keys[12] = tmp_2432; values[12] = tmp_2433; } let tmp_2435 = smem_keys[tmp_2395 * WPT + 13u]; let tmp_2436 = smem_vals[tmp_2395 * WPT + 13u]; let tmp_2437 = keys[13] < tmp_2435 || (keys[13] == tmp_2435 && values[13] < tmp_2436); if tmp_2394 == tmp_2437 { keys[13] = tmp_2435; values[13] = tmp_2436; } let tmp_2438 = smem_keys[tmp_2395 * WPT + 14u]; let tmp_2439 = smem_vals[tmp_2395 * WPT + 14u]; let tmp_2440 = keys[14] < tmp_2438 || (keys[14] == tmp_2438 && values[14] < tmp_2439); if tmp_2394 == tmp_2440 { keys[14] = tmp_2438; values[14] = tmp_2439; } let tmp_2441 = smem_keys[tmp_2395 * WPT + 15u]; let tmp_2442 = smem_vals[tmp_2395 * WPT + 15u]; let tmp_2443 = keys[15] < tmp_2441 || (keys[15] == tmp_2441 && values[15] < tmp_2442); if tmp_2394 == tmp_2443 { keys[15] = tmp_2441; values[15] = tmp_2442; } let tmp_2444 = smem_keys[tmp_2395 * WPT + 16u]; let tmp_2445 = smem_vals[tmp_2395 * WPT + 16u]; let tmp_2446 = keys[16] < tmp_2444 || (keys[16] == tmp_2444 && values[16] < tmp_2445); if tmp_2394 == tmp_2446 { keys[16] = tmp_2444; values[16] = tmp_2445; } let tmp_2447 = smem_keys[tmp_2395 * WPT + 17u]; let tmp_2448 = smem_vals[tmp_2395 * WPT + 17u]; let tmp_2449 = keys[17] < tmp_2447 || (keys[17] == tmp_2447 && values[17] < tmp_2448); if tmp_2394 == tmp_2449 { keys[17] = tmp_2447; values[17] = tmp_2448; } let tmp_2450 = smem_keys[tmp_2395 * WPT + 18u]; let tmp_2451 = smem_vals[tmp_2395 * WPT + 18u]; let tmp_2452 = keys[18] < tmp_2450 || (keys[18] == tmp_2450 && values[18] < tmp_2451); if tmp_2394 == tmp_2452 { keys[18] = tmp_2450; values[18] = tmp_2451; } let tmp_2453 = smem_keys[tmp_2395 * WPT + 19u]; let tmp_2454 = smem_vals[tmp_2395 * WPT + 19u]; let tmp_2455 = keys[19] < tmp_2453 || (keys[19] == tmp_2453 && values[19] < tmp_2454); if tmp_2394 == tmp_2455 { keys[19] = tmp_2453; values[19] = tmp_2454; } let tmp_2456 = smem_keys[tmp_2395 * WPT + 20u]; let tmp_2457 = smem_vals[tmp_2395 * WPT + 20u]; let tmp_2458 = keys[20] < tmp_2456 || (keys[20] == tmp_2456 && values[20] < tmp_2457); if tmp_2394 == tmp_2458 { keys[20] = tmp_2456; values[20] = tmp_2457; } let tmp_2459 = smem_keys[tmp_2395 * WPT + 21u]; let tmp_2460 = smem_vals[tmp_2395 * WPT + 21u]; let tmp_2461 = keys[21] < tmp_2459 || (keys[21] == tmp_2459 && values[21] < tmp_2460); if tmp_2394 == tmp_2461 { keys[21] = tmp_2459; values[21] = tmp_2460; } let tmp_2462 = smem_keys[tmp_2395 * WPT + 22u]; let tmp_2463 = smem_vals[tmp_2395 * WPT + 22u]; let tmp_2464 = keys[22] < tmp_2462 || (keys[22] == tmp_2462 && values[22] < tmp_2463); if tmp_2394 == tmp_2464 { keys[22] = tmp_2462; values[22] = tmp_2463; } let tmp_2465 = smem_keys[tmp_2395 * WPT + 23u]; let tmp_2466 = smem_vals[tmp_2395 * WPT + 23u]; let tmp_2467 = keys[23] < tmp_2465 || (keys[23] == tmp_2465 && values[23] < tmp_2466); if tmp_2394 == tmp_2467 { keys[23] = tmp_2465; values[23] = tmp_2466; } let tmp_2468 = smem_keys[tmp_2395 * WPT + 24u]; let tmp_2469 = smem_vals[tmp_2395 * WPT + 24u]; let tmp_2470 = keys[24] < tmp_2468 || (keys[24] == tmp_2468 && values[24] < tmp_2469); if tmp_2394 == tmp_2470 { keys[24] = tmp_2468; values[24] = tmp_2469; } let tmp_2471 = smem_keys[tmp_2395 * WPT + 25u]; let tmp_2472 = smem_vals[tmp_2395 * WPT + 25u]; let tmp_2473 = keys[25] < tmp_2471 || (keys[25] == tmp_2471 && values[25] < tmp_2472); if tmp_2394 == tmp_2473 { keys[25] = tmp_2471; values[25] = tmp_2472; } let tmp_2474 = smem_keys[tmp_2395 * WPT + 26u]; let tmp_2475 = smem_vals[tmp_2395 * WPT + 26u]; let tmp_2476 = keys[26] < tmp_2474 || (keys[26] == tmp_2474 && values[26] < tmp_2475); if tmp_2394 == tmp_2476 { keys[26] = tmp_2474; values[26] = tmp_2475; } let tmp_2477 = smem_keys[tmp_2395 * WPT + 27u]; let tmp_2478 = smem_vals[tmp_2395 * WPT + 27u]; let tmp_2479 = keys[27] < tmp_2477 || (keys[27] == tmp_2477 && values[27] < tmp_2478); if tmp_2394 == tmp_2479 { keys[27] = tmp_2477; values[27] = tmp_2478; } let tmp_2480 = smem_keys[tmp_2395 * WPT + 28u]; let tmp_2481 = smem_vals[tmp_2395 * WPT + 28u]; let tmp_2482 = keys[28] < tmp_2480 || (keys[28] == tmp_2480 && values[28] < tmp_2481); if tmp_2394 == tmp_2482 { keys[28] = tmp_2480; values[28] = tmp_2481; } let tmp_2483 = smem_keys[tmp_2395 * WPT + 29u]; let tmp_2484 = smem_vals[tmp_2395 * WPT + 29u]; let tmp_2485 = keys[29] < tmp_2483 || (keys[29] == tmp_2483 && values[29] < tmp_2484); if tmp_2394 == tmp_2485 { keys[29] = tmp_2483; values[29] = tmp_2484; } let tmp_2486 = smem_keys[tmp_2395 * WPT + 30u]; let tmp_2487 = smem_vals[tmp_2395 * WPT + 30u]; let tmp_2488 = keys[30] < tmp_2486 || (keys[30] == tmp_2486 && values[30] < tmp_2487); if tmp_2394 == tmp_2488 { keys[30] = tmp_2486; values[30] = tmp_2487; } let tmp_2489 = smem_keys[tmp_2395 * WPT + 31u]; let tmp_2490 = smem_vals[tmp_2395 * WPT + 31u]; let tmp_2491 = keys[31] < tmp_2489 || (keys[31] == tmp_2489 && values[31] < tmp_2490); if tmp_2394 == tmp_2491 { keys[31] = tmp_2489; values[31] = tmp_2490; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_2492 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_2493 = seg_base + (local_tid ^ 1u); let tmp_2494 = smem_keys[tmp_2493 * WPT + 0u]; let tmp_2495 = smem_vals[tmp_2493 * WPT + 0u]; let tmp_2496 = keys[0] < tmp_2494 || (keys[0] == tmp_2494 && values[0] < tmp_2495); if tmp_2492 == tmp_2496 { keys[0] = tmp_2494; values[0] = tmp_2495; } let tmp_2497 = smem_keys[tmp_2493 * WPT + 1u]; let tmp_2498 = smem_vals[tmp_2493 * WPT + 1u]; let tmp_2499 = keys[1] < tmp_2497 || (keys[1] == tmp_2497 && values[1] < tmp_2498); if tmp_2492 == tmp_2499 { keys[1] = tmp_2497; values[1] = tmp_2498; } let tmp_2500 = smem_keys[tmp_2493 * WPT + 2u]; let tmp_2501 = smem_vals[tmp_2493 * WPT + 2u]; let tmp_2502 = keys[2] < tmp_2500 || (keys[2] == tmp_2500 && values[2] < tmp_2501); if tmp_2492 == tmp_2502 { keys[2] = tmp_2500; values[2] = tmp_2501; } let tmp_2503 = smem_keys[tmp_2493 * WPT + 3u]; let tmp_2504 = smem_vals[tmp_2493 * WPT + 3u]; let tmp_2505 = keys[3] < tmp_2503 || (keys[3] == tmp_2503 && values[3] < tmp_2504); if tmp_2492 == tmp_2505 { keys[3] = tmp_2503; values[3] = tmp_2504; } let tmp_2506 = smem_keys[tmp_2493 * WPT + 4u]; let tmp_2507 = smem_vals[tmp_2493 * WPT + 4u]; let tmp_2508 = keys[4] < tmp_2506 || (keys[4] == tmp_2506 && values[4] < tmp_2507); if tmp_2492 == tmp_2508 { keys[4] = tmp_2506; values[4] = tmp_2507; } let tmp_2509 = smem_keys[tmp_2493 * WPT + 5u]; let tmp_2510 = smem_vals[tmp_2493 * WPT + 5u]; let tmp_2511 = keys[5] < tmp_2509 || (keys[5] == tmp_2509 && values[5] < tmp_2510); if tmp_2492 == tmp_2511 { keys[5] = tmp_2509; values[5] = tmp_2510; } let tmp_2512 = smem_keys[tmp_2493 * WPT + 6u]; let tmp_2513 = smem_vals[tmp_2493 * WPT + 6u]; let tmp_2514 = keys[6] < tmp_2512 || (keys[6] == tmp_2512 && values[6] < tmp_2513); if tmp_2492 == tmp_2514 { keys[6] = tmp_2512; values[6] = tmp_2513; } let tmp_2515 = smem_keys[tmp_2493 * WPT + 7u]; let tmp_2516 = smem_vals[tmp_2493 * WPT + 7u]; let tmp_2517 = keys[7] < tmp_2515 || (keys[7] == tmp_2515 && values[7] < tmp_2516); if tmp_2492 == tmp_2517 { keys[7] = tmp_2515; values[7] = tmp_2516; } let tmp_2518 = smem_keys[tmp_2493 * WPT + 8u]; let tmp_2519 = smem_vals[tmp_2493 * WPT + 8u]; let tmp_2520 = keys[8] < tmp_2518 || (keys[8] == tmp_2518 && values[8] < tmp_2519); if tmp_2492 == tmp_2520 { keys[8] = tmp_2518; values[8] = tmp_2519; } let tmp_2521 = smem_keys[tmp_2493 * WPT + 9u]; let tmp_2522 = smem_vals[tmp_2493 * WPT + 9u]; let tmp_2523 = keys[9] < tmp_2521 || (keys[9] == tmp_2521 && values[9] < tmp_2522); if tmp_2492 == tmp_2523 { keys[9] = tmp_2521; values[9] = tmp_2522; } let tmp_2524 = smem_keys[tmp_2493 * WPT + 10u]; let tmp_2525 = smem_vals[tmp_2493 * WPT + 10u]; let tmp_2526 = keys[10] < tmp_2524 || (keys[10] == tmp_2524 && values[10] < tmp_2525); if tmp_2492 == tmp_2526 { keys[10] = tmp_2524; values[10] = tmp_2525; } let tmp_2527 = smem_keys[tmp_2493 * WPT + 11u]; let tmp_2528 = smem_vals[tmp_2493 * WPT + 11u]; let tmp_2529 = keys[11] < tmp_2527 || (keys[11] == tmp_2527 && values[11] < tmp_2528); if tmp_2492 == tmp_2529 { keys[11] = tmp_2527; values[11] = tmp_2528; } let tmp_2530 = smem_keys[tmp_2493 * WPT + 12u]; let tmp_2531 = smem_vals[tmp_2493 * WPT + 12u]; let tmp_2532 = keys[12] < tmp_2530 || (keys[12] == tmp_2530 && values[12] < tmp_2531); if tmp_2492 == tmp_2532 { keys[12] = tmp_2530; values[12] = tmp_2531; } let tmp_2533 = smem_keys[tmp_2493 * WPT + 13u]; let tmp_2534 = smem_vals[tmp_2493 * WPT + 13u]; let tmp_2535 = keys[13] < tmp_2533 || (keys[13] == tmp_2533 && values[13] < tmp_2534); if tmp_2492 == tmp_2535 { keys[13] = tmp_2533; values[13] = tmp_2534; } let tmp_2536 = smem_keys[tmp_2493 * WPT + 14u]; let tmp_2537 = smem_vals[tmp_2493 * WPT + 14u]; let tmp_2538 = keys[14] < tmp_2536 || (keys[14] == tmp_2536 && values[14] < tmp_2537); if tmp_2492 == tmp_2538 { keys[14] = tmp_2536; values[14] = tmp_2537; } let tmp_2539 = smem_keys[tmp_2493 * WPT + 15u]; let tmp_2540 = smem_vals[tmp_2493 * WPT + 15u]; let tmp_2541 = keys[15] < tmp_2539 || (keys[15] == tmp_2539 && values[15] < tmp_2540); if tmp_2492 == tmp_2541 { keys[15] = tmp_2539; values[15] = tmp_2540; } let tmp_2542 = smem_keys[tmp_2493 * WPT + 16u]; let tmp_2543 = smem_vals[tmp_2493 * WPT + 16u]; let tmp_2544 = keys[16] < tmp_2542 || (keys[16] == tmp_2542 && values[16] < tmp_2543); if tmp_2492 == tmp_2544 { keys[16] = tmp_2542; values[16] = tmp_2543; } let tmp_2545 = smem_keys[tmp_2493 * WPT + 17u]; let tmp_2546 = smem_vals[tmp_2493 * WPT + 17u]; let tmp_2547 = keys[17] < tmp_2545 || (keys[17] == tmp_2545 && values[17] < tmp_2546); if tmp_2492 == tmp_2547 { keys[17] = tmp_2545; values[17] = tmp_2546; } let tmp_2548 = smem_keys[tmp_2493 * WPT + 18u]; let tmp_2549 = smem_vals[tmp_2493 * WPT + 18u]; let tmp_2550 = keys[18] < tmp_2548 || (keys[18] == tmp_2548 && values[18] < tmp_2549); if tmp_2492 == tmp_2550 { keys[18] = tmp_2548; values[18] = tmp_2549; } let tmp_2551 = smem_keys[tmp_2493 * WPT + 19u]; let tmp_2552 = smem_vals[tmp_2493 * WPT + 19u]; let tmp_2553 = keys[19] < tmp_2551 || (keys[19] == tmp_2551 && values[19] < tmp_2552); if tmp_2492 == tmp_2553 { keys[19] = tmp_2551; values[19] = tmp_2552; } let tmp_2554 = smem_keys[tmp_2493 * WPT + 20u]; let tmp_2555 = smem_vals[tmp_2493 * WPT + 20u]; let tmp_2556 = keys[20] < tmp_2554 || (keys[20] == tmp_2554 && values[20] < tmp_2555); if tmp_2492 == tmp_2556 { keys[20] = tmp_2554; values[20] = tmp_2555; } let tmp_2557 = smem_keys[tmp_2493 * WPT + 21u]; let tmp_2558 = smem_vals[tmp_2493 * WPT + 21u]; let tmp_2559 = keys[21] < tmp_2557 || (keys[21] == tmp_2557 && values[21] < tmp_2558); if tmp_2492 == tmp_2559 { keys[21] = tmp_2557; values[21] = tmp_2558; } let tmp_2560 = smem_keys[tmp_2493 * WPT + 22u]; let tmp_2561 = smem_vals[tmp_2493 * WPT + 22u]; let tmp_2562 = keys[22] < tmp_2560 || (keys[22] == tmp_2560 && values[22] < tmp_2561); if tmp_2492 == tmp_2562 { keys[22] = tmp_2560; values[22] = tmp_2561; } let tmp_2563 = smem_keys[tmp_2493 * WPT + 23u]; let tmp_2564 = smem_vals[tmp_2493 * WPT + 23u]; let tmp_2565 = keys[23] < tmp_2563 || (keys[23] == tmp_2563 && values[23] < tmp_2564); if tmp_2492 == tmp_2565 { keys[23] = tmp_2563; values[23] = tmp_2564; } let tmp_2566 = smem_keys[tmp_2493 * WPT + 24u]; let tmp_2567 = smem_vals[tmp_2493 * WPT + 24u]; let tmp_2568 = keys[24] < tmp_2566 || (keys[24] == tmp_2566 && values[24] < tmp_2567); if tmp_2492 == tmp_2568 { keys[24] = tmp_2566; values[24] = tmp_2567; } let tmp_2569 = smem_keys[tmp_2493 * WPT + 25u]; let tmp_2570 = smem_vals[tmp_2493 * WPT + 25u]; let tmp_2571 = keys[25] < tmp_2569 || (keys[25] == tmp_2569 && values[25] < tmp_2570); if tmp_2492 == tmp_2571 { keys[25] = tmp_2569; values[25] = tmp_2570; } let tmp_2572 = smem_keys[tmp_2493 * WPT + 26u]; let tmp_2573 = smem_vals[tmp_2493 * WPT + 26u]; let tmp_2574 = keys[26] < tmp_2572 || (keys[26] == tmp_2572 && values[26] < tmp_2573); if tmp_2492 == tmp_2574 { keys[26] = tmp_2572; values[26] = tmp_2573; } let tmp_2575 = smem_keys[tmp_2493 * WPT + 27u]; let tmp_2576 = smem_vals[tmp_2493 * WPT + 27u]; let tmp_2577 = keys[27] < tmp_2575 || (keys[27] == tmp_2575 && values[27] < tmp_2576); if tmp_2492 == tmp_2577 { keys[27] = tmp_2575; values[27] = tmp_2576; } let tmp_2578 = smem_keys[tmp_2493 * WPT + 28u]; let tmp_2579 = smem_vals[tmp_2493 * WPT + 28u]; let tmp_2580 = keys[28] < tmp_2578 || (keys[28] == tmp_2578 && values[28] < tmp_2579); if tmp_2492 == tmp_2580 { keys[28] = tmp_2578; values[28] = tmp_2579; } let tmp_2581 = smem_keys[tmp_2493 * WPT + 29u]; let tmp_2582 = smem_vals[tmp_2493 * WPT + 29u]; let tmp_2583 = keys[29] < tmp_2581 || (keys[29] == tmp_2581 && values[29] < tmp_2582); if tmp_2492 == tmp_2583 { keys[29] = tmp_2581; values[29] = tmp_2582; } let tmp_2584 = smem_keys[tmp_2493 * WPT + 30u]; let tmp_2585 = smem_vals[tmp_2493 * WPT + 30u]; let tmp_2586 = keys[30] < tmp_2584 || (keys[30] == tmp_2584 && values[30] < tmp_2585); if tmp_2492 == tmp_2586 { keys[30] = tmp_2584; values[30] = tmp_2585; } let tmp_2587 = smem_keys[tmp_2493 * WPT + 31u]; let tmp_2588 = smem_vals[tmp_2493 * WPT + 31u]; let tmp_2589 = keys[31] < tmp_2587 || (keys[31] == tmp_2587 && values[31] < tmp_2588); if tmp_2492 == tmp_2589 { keys[31] = tmp_2587; values[31] = tmp_2588; } workgroupBarrier(); }
    // exch_local(16,32) 
    // cmp_swap(0,16)
    if keys[0] > keys[16] || (keys[0] == keys[16] && values[0] > values[16]) {
    // swap(0,16) 
    { let tmp_2590 = keys[0]; keys[0] = keys[16]; keys[16] = tmp_2590;let tmp_2591 = values[0]; values[0] = values[16]; values[16] = tmp_2591; }
    }
    // cmp_swap(1,17)
    if keys[1] > keys[17] || (keys[1] == keys[17] && values[1] > values[17]) {
    // swap(1,17) 
    { let tmp_2592 = keys[1]; keys[1] = keys[17]; keys[17] = tmp_2592;let tmp_2593 = values[1]; values[1] = values[17]; values[17] = tmp_2593; }
    }
    // cmp_swap(2,18)
    if keys[2] > keys[18] || (keys[2] == keys[18] && values[2] > values[18]) {
    // swap(2,18) 
    { let tmp_2594 = keys[2]; keys[2] = keys[18]; keys[18] = tmp_2594;let tmp_2595 = values[2]; values[2] = values[18]; values[18] = tmp_2595; }
    }
    // cmp_swap(3,19)
    if keys[3] > keys[19] || (keys[3] == keys[19] && values[3] > values[19]) {
    // swap(3,19) 
    { let tmp_2596 = keys[3]; keys[3] = keys[19]; keys[19] = tmp_2596;let tmp_2597 = values[3]; values[3] = values[19]; values[19] = tmp_2597; }
    }
    // cmp_swap(4,20)
    if keys[4] > keys[20] || (keys[4] == keys[20] && values[4] > values[20]) {
    // swap(4,20) 
    { let tmp_2598 = keys[4]; keys[4] = keys[20]; keys[20] = tmp_2598;let tmp_2599 = values[4]; values[4] = values[20]; values[20] = tmp_2599; }
    }
    // cmp_swap(5,21)
    if keys[5] > keys[21] || (keys[5] == keys[21] && values[5] > values[21]) {
    // swap(5,21) 
    { let tmp_2600 = keys[5]; keys[5] = keys[21]; keys[21] = tmp_2600;let tmp_2601 = values[5]; values[5] = values[21]; values[21] = tmp_2601; }
    }
    // cmp_swap(6,22)
    if keys[6] > keys[22] || (keys[6] == keys[22] && values[6] > values[22]) {
    // swap(6,22) 
    { let tmp_2602 = keys[6]; keys[6] = keys[22]; keys[22] = tmp_2602;let tmp_2603 = values[6]; values[6] = values[22]; values[22] = tmp_2603; }
    }
    // cmp_swap(7,23)
    if keys[7] > keys[23] || (keys[7] == keys[23] && values[7] > values[23]) {
    // swap(7,23) 
    { let tmp_2604 = keys[7]; keys[7] = keys[23]; keys[23] = tmp_2604;let tmp_2605 = values[7]; values[7] = values[23]; values[23] = tmp_2605; }
    }
    // cmp_swap(8,24)
    if keys[8] > keys[24] || (keys[8] == keys[24] && values[8] > values[24]) {
    // swap(8,24) 
    { let tmp_2606 = keys[8]; keys[8] = keys[24]; keys[24] = tmp_2606;let tmp_2607 = values[8]; values[8] = values[24]; values[24] = tmp_2607; }
    }
    // cmp_swap(9,25)
    if keys[9] > keys[25] || (keys[9] == keys[25] && values[9] > values[25]) {
    // swap(9,25) 
    { let tmp_2608 = keys[9]; keys[9] = keys[25]; keys[25] = tmp_2608;let tmp_2609 = values[9]; values[9] = values[25]; values[25] = tmp_2609; }
    }
    // cmp_swap(10,26)
    if keys[10] > keys[26] || (keys[10] == keys[26] && values[10] > values[26]) {
    // swap(10,26) 
    { let tmp_2610 = keys[10]; keys[10] = keys[26]; keys[26] = tmp_2610;let tmp_2611 = values[10]; values[10] = values[26]; values[26] = tmp_2611; }
    }
    // cmp_swap(11,27)
    if keys[11] > keys[27] || (keys[11] == keys[27] && values[11] > values[27]) {
    // swap(11,27) 
    { let tmp_2612 = keys[11]; keys[11] = keys[27]; keys[27] = tmp_2612;let tmp_2613 = values[11]; values[11] = values[27]; values[27] = tmp_2613; }
    }
    // cmp_swap(12,28)
    if keys[12] > keys[28] || (keys[12] == keys[28] && values[12] > values[28]) {
    // swap(12,28) 
    { let tmp_2614 = keys[12]; keys[12] = keys[28]; keys[28] = tmp_2614;let tmp_2615 = values[12]; values[12] = values[28]; values[28] = tmp_2615; }
    }
    // cmp_swap(13,29)
    if keys[13] > keys[29] || (keys[13] == keys[29] && values[13] > values[29]) {
    // swap(13,29) 
    { let tmp_2616 = keys[13]; keys[13] = keys[29]; keys[29] = tmp_2616;let tmp_2617 = values[13]; values[13] = values[29]; values[29] = tmp_2617; }
    }
    // cmp_swap(14,30)
    if keys[14] > keys[30] || (keys[14] == keys[30] && values[14] > values[30]) {
    // swap(14,30) 
    { let tmp_2618 = keys[14]; keys[14] = keys[30]; keys[30] = tmp_2618;let tmp_2619 = values[14]; values[14] = values[30]; values[30] = tmp_2619; }
    }
    // cmp_swap(15,31)
    if keys[15] > keys[31] || (keys[15] == keys[31] && values[15] > values[31]) {
    // swap(15,31) 
    { let tmp_2620 = keys[15]; keys[15] = keys[31]; keys[31] = tmp_2620;let tmp_2621 = values[15]; values[15] = values[31]; values[31] = tmp_2621; }
    }
    // exch_local(8,32) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_2622 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_2622;let tmp_2623 = values[0]; values[0] = values[8]; values[8] = tmp_2623; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_2624 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_2624;let tmp_2625 = values[1]; values[1] = values[9]; values[9] = tmp_2625; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_2626 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_2626;let tmp_2627 = values[2]; values[2] = values[10]; values[10] = tmp_2627; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_2628 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_2628;let tmp_2629 = values[3]; values[3] = values[11]; values[11] = tmp_2629; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_2630 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_2630;let tmp_2631 = values[4]; values[4] = values[12]; values[12] = tmp_2631; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_2632 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_2632;let tmp_2633 = values[5]; values[5] = values[13]; values[13] = tmp_2633; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_2634 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_2634;let tmp_2635 = values[6]; values[6] = values[14]; values[14] = tmp_2635; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_2636 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_2636;let tmp_2637 = values[7]; values[7] = values[15]; values[15] = tmp_2637; }
    }
    // cmp_swap(16,24)
    if keys[16] > keys[24] || (keys[16] == keys[24] && values[16] > values[24]) {
    // swap(16,24) 
    { let tmp_2638 = keys[16]; keys[16] = keys[24]; keys[24] = tmp_2638;let tmp_2639 = values[16]; values[16] = values[24]; values[24] = tmp_2639; }
    }
    // cmp_swap(17,25)
    if keys[17] > keys[25] || (keys[17] == keys[25] && values[17] > values[25]) {
    // swap(17,25) 
    { let tmp_2640 = keys[17]; keys[17] = keys[25]; keys[25] = tmp_2640;let tmp_2641 = values[17]; values[17] = values[25]; values[25] = tmp_2641; }
    }
    // cmp_swap(18,26)
    if keys[18] > keys[26] || (keys[18] == keys[26] && values[18] > values[26]) {
    // swap(18,26) 
    { let tmp_2642 = keys[18]; keys[18] = keys[26]; keys[26] = tmp_2642;let tmp_2643 = values[18]; values[18] = values[26]; values[26] = tmp_2643; }
    }
    // cmp_swap(19,27)
    if keys[19] > keys[27] || (keys[19] == keys[27] && values[19] > values[27]) {
    // swap(19,27) 
    { let tmp_2644 = keys[19]; keys[19] = keys[27]; keys[27] = tmp_2644;let tmp_2645 = values[19]; values[19] = values[27]; values[27] = tmp_2645; }
    }
    // cmp_swap(20,28)
    if keys[20] > keys[28] || (keys[20] == keys[28] && values[20] > values[28]) {
    // swap(20,28) 
    { let tmp_2646 = keys[20]; keys[20] = keys[28]; keys[28] = tmp_2646;let tmp_2647 = values[20]; values[20] = values[28]; values[28] = tmp_2647; }
    }
    // cmp_swap(21,29)
    if keys[21] > keys[29] || (keys[21] == keys[29] && values[21] > values[29]) {
    // swap(21,29) 
    { let tmp_2648 = keys[21]; keys[21] = keys[29]; keys[29] = tmp_2648;let tmp_2649 = values[21]; values[21] = values[29]; values[29] = tmp_2649; }
    }
    // cmp_swap(22,30)
    if keys[22] > keys[30] || (keys[22] == keys[30] && values[22] > values[30]) {
    // swap(22,30) 
    { let tmp_2650 = keys[22]; keys[22] = keys[30]; keys[30] = tmp_2650;let tmp_2651 = values[22]; values[22] = values[30]; values[30] = tmp_2651; }
    }
    // cmp_swap(23,31)
    if keys[23] > keys[31] || (keys[23] == keys[31] && values[23] > values[31]) {
    // swap(23,31) 
    { let tmp_2652 = keys[23]; keys[23] = keys[31]; keys[31] = tmp_2652;let tmp_2653 = values[23]; values[23] = values[31]; values[31] = tmp_2653; }
    }
    // exch_local(4,32) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_2654 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_2654;let tmp_2655 = values[0]; values[0] = values[4]; values[4] = tmp_2655; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_2656 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_2656;let tmp_2657 = values[1]; values[1] = values[5]; values[5] = tmp_2657; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_2658 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_2658;let tmp_2659 = values[2]; values[2] = values[6]; values[6] = tmp_2659; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_2660 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_2660;let tmp_2661 = values[3]; values[3] = values[7]; values[7] = tmp_2661; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_2662 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_2662;let tmp_2663 = values[8]; values[8] = values[12]; values[12] = tmp_2663; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_2664 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_2664;let tmp_2665 = values[9]; values[9] = values[13]; values[13] = tmp_2665; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_2666 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_2666;let tmp_2667 = values[10]; values[10] = values[14]; values[14] = tmp_2667; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_2668 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_2668;let tmp_2669 = values[11]; values[11] = values[15]; values[15] = tmp_2669; }
    }
    // cmp_swap(16,20)
    if keys[16] > keys[20] || (keys[16] == keys[20] && values[16] > values[20]) {
    // swap(16,20) 
    { let tmp_2670 = keys[16]; keys[16] = keys[20]; keys[20] = tmp_2670;let tmp_2671 = values[16]; values[16] = values[20]; values[20] = tmp_2671; }
    }
    // cmp_swap(17,21)
    if keys[17] > keys[21] || (keys[17] == keys[21] && values[17] > values[21]) {
    // swap(17,21) 
    { let tmp_2672 = keys[17]; keys[17] = keys[21]; keys[21] = tmp_2672;let tmp_2673 = values[17]; values[17] = values[21]; values[21] = tmp_2673; }
    }
    // cmp_swap(18,22)
    if keys[18] > keys[22] || (keys[18] == keys[22] && values[18] > values[22]) {
    // swap(18,22) 
    { let tmp_2674 = keys[18]; keys[18] = keys[22]; keys[22] = tmp_2674;let tmp_2675 = values[18]; values[18] = values[22]; values[22] = tmp_2675; }
    }
    // cmp_swap(19,23)
    if keys[19] > keys[23] || (keys[19] == keys[23] && values[19] > values[23]) {
    // swap(19,23) 
    { let tmp_2676 = keys[19]; keys[19] = keys[23]; keys[23] = tmp_2676;let tmp_2677 = values[19]; values[19] = values[23]; values[23] = tmp_2677; }
    }
    // cmp_swap(24,28)
    if keys[24] > keys[28] || (keys[24] == keys[28] && values[24] > values[28]) {
    // swap(24,28) 
    { let tmp_2678 = keys[24]; keys[24] = keys[28]; keys[28] = tmp_2678;let tmp_2679 = values[24]; values[24] = values[28]; values[28] = tmp_2679; }
    }
    // cmp_swap(25,29)
    if keys[25] > keys[29] || (keys[25] == keys[29] && values[25] > values[29]) {
    // swap(25,29) 
    { let tmp_2680 = keys[25]; keys[25] = keys[29]; keys[29] = tmp_2680;let tmp_2681 = values[25]; values[25] = values[29]; values[29] = tmp_2681; }
    }
    // cmp_swap(26,30)
    if keys[26] > keys[30] || (keys[26] == keys[30] && values[26] > values[30]) {
    // swap(26,30) 
    { let tmp_2682 = keys[26]; keys[26] = keys[30]; keys[30] = tmp_2682;let tmp_2683 = values[26]; values[26] = values[30]; values[30] = tmp_2683; }
    }
    // cmp_swap(27,31)
    if keys[27] > keys[31] || (keys[27] == keys[31] && values[27] > values[31]) {
    // swap(27,31) 
    { let tmp_2684 = keys[27]; keys[27] = keys[31]; keys[31] = tmp_2684;let tmp_2685 = values[27]; values[27] = values[31]; values[31] = tmp_2685; }
    }
    // exch_local(2,32) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_2686 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_2686;let tmp_2687 = values[0]; values[0] = values[2]; values[2] = tmp_2687; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_2688 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_2688;let tmp_2689 = values[1]; values[1] = values[3]; values[3] = tmp_2689; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_2690 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_2690;let tmp_2691 = values[4]; values[4] = values[6]; values[6] = tmp_2691; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_2692 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_2692;let tmp_2693 = values[5]; values[5] = values[7]; values[7] = tmp_2693; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_2694 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_2694;let tmp_2695 = values[8]; values[8] = values[10]; values[10] = tmp_2695; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_2696 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_2696;let tmp_2697 = values[9]; values[9] = values[11]; values[11] = tmp_2697; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_2698 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_2698;let tmp_2699 = values[12]; values[12] = values[14]; values[14] = tmp_2699; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_2700 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_2700;let tmp_2701 = values[13]; values[13] = values[15]; values[15] = tmp_2701; }
    }
    // cmp_swap(16,18)
    if keys[16] > keys[18] || (keys[16] == keys[18] && values[16] > values[18]) {
    // swap(16,18) 
    { let tmp_2702 = keys[16]; keys[16] = keys[18]; keys[18] = tmp_2702;let tmp_2703 = values[16]; values[16] = values[18]; values[18] = tmp_2703; }
    }
    // cmp_swap(17,19)
    if keys[17] > keys[19] || (keys[17] == keys[19] && values[17] > values[19]) {
    // swap(17,19) 
    { let tmp_2704 = keys[17]; keys[17] = keys[19]; keys[19] = tmp_2704;let tmp_2705 = values[17]; values[17] = values[19]; values[19] = tmp_2705; }
    }
    // cmp_swap(20,22)
    if keys[20] > keys[22] || (keys[20] == keys[22] && values[20] > values[22]) {
    // swap(20,22) 
    { let tmp_2706 = keys[20]; keys[20] = keys[22]; keys[22] = tmp_2706;let tmp_2707 = values[20]; values[20] = values[22]; values[22] = tmp_2707; }
    }
    // cmp_swap(21,23)
    if keys[21] > keys[23] || (keys[21] == keys[23] && values[21] > values[23]) {
    // swap(21,23) 
    { let tmp_2708 = keys[21]; keys[21] = keys[23]; keys[23] = tmp_2708;let tmp_2709 = values[21]; values[21] = values[23]; values[23] = tmp_2709; }
    }
    // cmp_swap(24,26)
    if keys[24] > keys[26] || (keys[24] == keys[26] && values[24] > values[26]) {
    // swap(24,26) 
    { let tmp_2710 = keys[24]; keys[24] = keys[26]; keys[26] = tmp_2710;let tmp_2711 = values[24]; values[24] = values[26]; values[26] = tmp_2711; }
    }
    // cmp_swap(25,27)
    if keys[25] > keys[27] || (keys[25] == keys[27] && values[25] > values[27]) {
    // swap(25,27) 
    { let tmp_2712 = keys[25]; keys[25] = keys[27]; keys[27] = tmp_2712;let tmp_2713 = values[25]; values[25] = values[27]; values[27] = tmp_2713; }
    }
    // cmp_swap(28,30)
    if keys[28] > keys[30] || (keys[28] == keys[30] && values[28] > values[30]) {
    // swap(28,30) 
    { let tmp_2714 = keys[28]; keys[28] = keys[30]; keys[30] = tmp_2714;let tmp_2715 = values[28]; values[28] = values[30]; values[30] = tmp_2715; }
    }
    // cmp_swap(29,31)
    if keys[29] > keys[31] || (keys[29] == keys[31] && values[29] > values[31]) {
    // swap(29,31) 
    { let tmp_2716 = keys[29]; keys[29] = keys[31]; keys[31] = tmp_2716;let tmp_2717 = values[29]; values[29] = values[31]; values[31] = tmp_2717; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_2718 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_2718;let tmp_2719 = values[0]; values[0] = values[1]; values[1] = tmp_2719; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_2720 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_2720;let tmp_2721 = values[2]; values[2] = values[3]; values[3] = tmp_2721; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_2722 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_2722;let tmp_2723 = values[4]; values[4] = values[5]; values[5] = tmp_2723; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_2724 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_2724;let tmp_2725 = values[6]; values[6] = values[7]; values[7] = tmp_2725; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_2726 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_2726;let tmp_2727 = values[8]; values[8] = values[9]; values[9] = tmp_2727; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_2728 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_2728;let tmp_2729 = values[10]; values[10] = values[11]; values[11] = tmp_2729; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_2730 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_2730;let tmp_2731 = values[12]; values[12] = values[13]; values[13] = tmp_2731; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_2732 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_2732;let tmp_2733 = values[14]; values[14] = values[15]; values[15] = tmp_2733; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_2734 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_2734;let tmp_2735 = values[16]; values[16] = values[17]; values[17] = tmp_2735; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_2736 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_2736;let tmp_2737 = values[18]; values[18] = values[19]; values[19] = tmp_2737; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_2738 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_2738;let tmp_2739 = values[20]; values[20] = values[21]; values[21] = tmp_2739; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_2740 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_2740;let tmp_2741 = values[22]; values[22] = values[23]; values[23] = tmp_2741; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_2742 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_2742;let tmp_2743 = values[24]; values[24] = values[25]; values[25] = tmp_2743; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_2744 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_2744;let tmp_2745 = values[26]; values[26] = values[27]; values[27] = tmp_2745; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_2746 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_2746;let tmp_2747 = values[28]; values[28] = values[29]; values[29] = tmp_2747; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_2748 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_2748;let tmp_2749 = values[30]; values[30] = values[31]; values[31] = tmp_2749; }
    }
    // exch_intxn(tmask:63,swbit:5,wpt:32)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_2750 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_2751 = seg_base + (local_tid ^ 63u); let tmp_2752 = smem_keys[tmp_2751 * WPT + 31u]; let tmp_2753 = smem_vals[tmp_2751 * WPT + 31u]; let tmp_2754 = keys[0] < tmp_2752 || (keys[0] == tmp_2752 && values[0] < tmp_2753); if tmp_2750 == tmp_2754 { keys[0] = tmp_2752; values[0] = tmp_2753; } let tmp_2755 = smem_keys[tmp_2751 * WPT + 30u]; let tmp_2756 = smem_vals[tmp_2751 * WPT + 30u]; let tmp_2757 = keys[1] < tmp_2755 || (keys[1] == tmp_2755 && values[1] < tmp_2756); if tmp_2750 == tmp_2757 { keys[1] = tmp_2755; values[1] = tmp_2756; } let tmp_2758 = smem_keys[tmp_2751 * WPT + 29u]; let tmp_2759 = smem_vals[tmp_2751 * WPT + 29u]; let tmp_2760 = keys[2] < tmp_2758 || (keys[2] == tmp_2758 && values[2] < tmp_2759); if tmp_2750 == tmp_2760 { keys[2] = tmp_2758; values[2] = tmp_2759; } let tmp_2761 = smem_keys[tmp_2751 * WPT + 28u]; let tmp_2762 = smem_vals[tmp_2751 * WPT + 28u]; let tmp_2763 = keys[3] < tmp_2761 || (keys[3] == tmp_2761 && values[3] < tmp_2762); if tmp_2750 == tmp_2763 { keys[3] = tmp_2761; values[3] = tmp_2762; } let tmp_2764 = smem_keys[tmp_2751 * WPT + 27u]; let tmp_2765 = smem_vals[tmp_2751 * WPT + 27u]; let tmp_2766 = keys[4] < tmp_2764 || (keys[4] == tmp_2764 && values[4] < tmp_2765); if tmp_2750 == tmp_2766 { keys[4] = tmp_2764; values[4] = tmp_2765; } let tmp_2767 = smem_keys[tmp_2751 * WPT + 26u]; let tmp_2768 = smem_vals[tmp_2751 * WPT + 26u]; let tmp_2769 = keys[5] < tmp_2767 || (keys[5] == tmp_2767 && values[5] < tmp_2768); if tmp_2750 == tmp_2769 { keys[5] = tmp_2767; values[5] = tmp_2768; } let tmp_2770 = smem_keys[tmp_2751 * WPT + 25u]; let tmp_2771 = smem_vals[tmp_2751 * WPT + 25u]; let tmp_2772 = keys[6] < tmp_2770 || (keys[6] == tmp_2770 && values[6] < tmp_2771); if tmp_2750 == tmp_2772 { keys[6] = tmp_2770; values[6] = tmp_2771; } let tmp_2773 = smem_keys[tmp_2751 * WPT + 24u]; let tmp_2774 = smem_vals[tmp_2751 * WPT + 24u]; let tmp_2775 = keys[7] < tmp_2773 || (keys[7] == tmp_2773 && values[7] < tmp_2774); if tmp_2750 == tmp_2775 { keys[7] = tmp_2773; values[7] = tmp_2774; } let tmp_2776 = smem_keys[tmp_2751 * WPT + 23u]; let tmp_2777 = smem_vals[tmp_2751 * WPT + 23u]; let tmp_2778 = keys[8] < tmp_2776 || (keys[8] == tmp_2776 && values[8] < tmp_2777); if tmp_2750 == tmp_2778 { keys[8] = tmp_2776; values[8] = tmp_2777; } let tmp_2779 = smem_keys[tmp_2751 * WPT + 22u]; let tmp_2780 = smem_vals[tmp_2751 * WPT + 22u]; let tmp_2781 = keys[9] < tmp_2779 || (keys[9] == tmp_2779 && values[9] < tmp_2780); if tmp_2750 == tmp_2781 { keys[9] = tmp_2779; values[9] = tmp_2780; } let tmp_2782 = smem_keys[tmp_2751 * WPT + 21u]; let tmp_2783 = smem_vals[tmp_2751 * WPT + 21u]; let tmp_2784 = keys[10] < tmp_2782 || (keys[10] == tmp_2782 && values[10] < tmp_2783); if tmp_2750 == tmp_2784 { keys[10] = tmp_2782; values[10] = tmp_2783; } let tmp_2785 = smem_keys[tmp_2751 * WPT + 20u]; let tmp_2786 = smem_vals[tmp_2751 * WPT + 20u]; let tmp_2787 = keys[11] < tmp_2785 || (keys[11] == tmp_2785 && values[11] < tmp_2786); if tmp_2750 == tmp_2787 { keys[11] = tmp_2785; values[11] = tmp_2786; } let tmp_2788 = smem_keys[tmp_2751 * WPT + 19u]; let tmp_2789 = smem_vals[tmp_2751 * WPT + 19u]; let tmp_2790 = keys[12] < tmp_2788 || (keys[12] == tmp_2788 && values[12] < tmp_2789); if tmp_2750 == tmp_2790 { keys[12] = tmp_2788; values[12] = tmp_2789; } let tmp_2791 = smem_keys[tmp_2751 * WPT + 18u]; let tmp_2792 = smem_vals[tmp_2751 * WPT + 18u]; let tmp_2793 = keys[13] < tmp_2791 || (keys[13] == tmp_2791 && values[13] < tmp_2792); if tmp_2750 == tmp_2793 { keys[13] = tmp_2791; values[13] = tmp_2792; } let tmp_2794 = smem_keys[tmp_2751 * WPT + 17u]; let tmp_2795 = smem_vals[tmp_2751 * WPT + 17u]; let tmp_2796 = keys[14] < tmp_2794 || (keys[14] == tmp_2794 && values[14] < tmp_2795); if tmp_2750 == tmp_2796 { keys[14] = tmp_2794; values[14] = tmp_2795; } let tmp_2797 = smem_keys[tmp_2751 * WPT + 16u]; let tmp_2798 = smem_vals[tmp_2751 * WPT + 16u]; let tmp_2799 = keys[15] < tmp_2797 || (keys[15] == tmp_2797 && values[15] < tmp_2798); if tmp_2750 == tmp_2799 { keys[15] = tmp_2797; values[15] = tmp_2798; } let tmp_2800 = smem_keys[tmp_2751 * WPT + 15u]; let tmp_2801 = smem_vals[tmp_2751 * WPT + 15u]; let tmp_2802 = keys[16] < tmp_2800 || (keys[16] == tmp_2800 && values[16] < tmp_2801); if tmp_2750 == tmp_2802 { keys[16] = tmp_2800; values[16] = tmp_2801; } let tmp_2803 = smem_keys[tmp_2751 * WPT + 14u]; let tmp_2804 = smem_vals[tmp_2751 * WPT + 14u]; let tmp_2805 = keys[17] < tmp_2803 || (keys[17] == tmp_2803 && values[17] < tmp_2804); if tmp_2750 == tmp_2805 { keys[17] = tmp_2803; values[17] = tmp_2804; } let tmp_2806 = smem_keys[tmp_2751 * WPT + 13u]; let tmp_2807 = smem_vals[tmp_2751 * WPT + 13u]; let tmp_2808 = keys[18] < tmp_2806 || (keys[18] == tmp_2806 && values[18] < tmp_2807); if tmp_2750 == tmp_2808 { keys[18] = tmp_2806; values[18] = tmp_2807; } let tmp_2809 = smem_keys[tmp_2751 * WPT + 12u]; let tmp_2810 = smem_vals[tmp_2751 * WPT + 12u]; let tmp_2811 = keys[19] < tmp_2809 || (keys[19] == tmp_2809 && values[19] < tmp_2810); if tmp_2750 == tmp_2811 { keys[19] = tmp_2809; values[19] = tmp_2810; } let tmp_2812 = smem_keys[tmp_2751 * WPT + 11u]; let tmp_2813 = smem_vals[tmp_2751 * WPT + 11u]; let tmp_2814 = keys[20] < tmp_2812 || (keys[20] == tmp_2812 && values[20] < tmp_2813); if tmp_2750 == tmp_2814 { keys[20] = tmp_2812; values[20] = tmp_2813; } let tmp_2815 = smem_keys[tmp_2751 * WPT + 10u]; let tmp_2816 = smem_vals[tmp_2751 * WPT + 10u]; let tmp_2817 = keys[21] < tmp_2815 || (keys[21] == tmp_2815 && values[21] < tmp_2816); if tmp_2750 == tmp_2817 { keys[21] = tmp_2815; values[21] = tmp_2816; } let tmp_2818 = smem_keys[tmp_2751 * WPT + 9u]; let tmp_2819 = smem_vals[tmp_2751 * WPT + 9u]; let tmp_2820 = keys[22] < tmp_2818 || (keys[22] == tmp_2818 && values[22] < tmp_2819); if tmp_2750 == tmp_2820 { keys[22] = tmp_2818; values[22] = tmp_2819; } let tmp_2821 = smem_keys[tmp_2751 * WPT + 8u]; let tmp_2822 = smem_vals[tmp_2751 * WPT + 8u]; let tmp_2823 = keys[23] < tmp_2821 || (keys[23] == tmp_2821 && values[23] < tmp_2822); if tmp_2750 == tmp_2823 { keys[23] = tmp_2821; values[23] = tmp_2822; } let tmp_2824 = smem_keys[tmp_2751 * WPT + 7u]; let tmp_2825 = smem_vals[tmp_2751 * WPT + 7u]; let tmp_2826 = keys[24] < tmp_2824 || (keys[24] == tmp_2824 && values[24] < tmp_2825); if tmp_2750 == tmp_2826 { keys[24] = tmp_2824; values[24] = tmp_2825; } let tmp_2827 = smem_keys[tmp_2751 * WPT + 6u]; let tmp_2828 = smem_vals[tmp_2751 * WPT + 6u]; let tmp_2829 = keys[25] < tmp_2827 || (keys[25] == tmp_2827 && values[25] < tmp_2828); if tmp_2750 == tmp_2829 { keys[25] = tmp_2827; values[25] = tmp_2828; } let tmp_2830 = smem_keys[tmp_2751 * WPT + 5u]; let tmp_2831 = smem_vals[tmp_2751 * WPT + 5u]; let tmp_2832 = keys[26] < tmp_2830 || (keys[26] == tmp_2830 && values[26] < tmp_2831); if tmp_2750 == tmp_2832 { keys[26] = tmp_2830; values[26] = tmp_2831; } let tmp_2833 = smem_keys[tmp_2751 * WPT + 4u]; let tmp_2834 = smem_vals[tmp_2751 * WPT + 4u]; let tmp_2835 = keys[27] < tmp_2833 || (keys[27] == tmp_2833 && values[27] < tmp_2834); if tmp_2750 == tmp_2835 { keys[27] = tmp_2833; values[27] = tmp_2834; } let tmp_2836 = smem_keys[tmp_2751 * WPT + 3u]; let tmp_2837 = smem_vals[tmp_2751 * WPT + 3u]; let tmp_2838 = keys[28] < tmp_2836 || (keys[28] == tmp_2836 && values[28] < tmp_2837); if tmp_2750 == tmp_2838 { keys[28] = tmp_2836; values[28] = tmp_2837; } let tmp_2839 = smem_keys[tmp_2751 * WPT + 2u]; let tmp_2840 = smem_vals[tmp_2751 * WPT + 2u]; let tmp_2841 = keys[29] < tmp_2839 || (keys[29] == tmp_2839 && values[29] < tmp_2840); if tmp_2750 == tmp_2841 { keys[29] = tmp_2839; values[29] = tmp_2840; } let tmp_2842 = smem_keys[tmp_2751 * WPT + 1u]; let tmp_2843 = smem_vals[tmp_2751 * WPT + 1u]; let tmp_2844 = keys[30] < tmp_2842 || (keys[30] == tmp_2842 && values[30] < tmp_2843); if tmp_2750 == tmp_2844 { keys[30] = tmp_2842; values[30] = tmp_2843; } let tmp_2845 = smem_keys[tmp_2751 * WPT + 0u]; let tmp_2846 = smem_vals[tmp_2751 * WPT + 0u]; let tmp_2847 = keys[31] < tmp_2845 || (keys[31] == tmp_2845 && values[31] < tmp_2846); if tmp_2750 == tmp_2847 { keys[31] = tmp_2845; values[31] = tmp_2846; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_2848 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_2849 = seg_base + (local_tid ^ 16u); let tmp_2850 = smem_keys[tmp_2849 * WPT + 0u]; let tmp_2851 = smem_vals[tmp_2849 * WPT + 0u]; let tmp_2852 = keys[0] < tmp_2850 || (keys[0] == tmp_2850 && values[0] < tmp_2851); if tmp_2848 == tmp_2852 { keys[0] = tmp_2850; values[0] = tmp_2851; } let tmp_2853 = smem_keys[tmp_2849 * WPT + 1u]; let tmp_2854 = smem_vals[tmp_2849 * WPT + 1u]; let tmp_2855 = keys[1] < tmp_2853 || (keys[1] == tmp_2853 && values[1] < tmp_2854); if tmp_2848 == tmp_2855 { keys[1] = tmp_2853; values[1] = tmp_2854; } let tmp_2856 = smem_keys[tmp_2849 * WPT + 2u]; let tmp_2857 = smem_vals[tmp_2849 * WPT + 2u]; let tmp_2858 = keys[2] < tmp_2856 || (keys[2] == tmp_2856 && values[2] < tmp_2857); if tmp_2848 == tmp_2858 { keys[2] = tmp_2856; values[2] = tmp_2857; } let tmp_2859 = smem_keys[tmp_2849 * WPT + 3u]; let tmp_2860 = smem_vals[tmp_2849 * WPT + 3u]; let tmp_2861 = keys[3] < tmp_2859 || (keys[3] == tmp_2859 && values[3] < tmp_2860); if tmp_2848 == tmp_2861 { keys[3] = tmp_2859; values[3] = tmp_2860; } let tmp_2862 = smem_keys[tmp_2849 * WPT + 4u]; let tmp_2863 = smem_vals[tmp_2849 * WPT + 4u]; let tmp_2864 = keys[4] < tmp_2862 || (keys[4] == tmp_2862 && values[4] < tmp_2863); if tmp_2848 == tmp_2864 { keys[4] = tmp_2862; values[4] = tmp_2863; } let tmp_2865 = smem_keys[tmp_2849 * WPT + 5u]; let tmp_2866 = smem_vals[tmp_2849 * WPT + 5u]; let tmp_2867 = keys[5] < tmp_2865 || (keys[5] == tmp_2865 && values[5] < tmp_2866); if tmp_2848 == tmp_2867 { keys[5] = tmp_2865; values[5] = tmp_2866; } let tmp_2868 = smem_keys[tmp_2849 * WPT + 6u]; let tmp_2869 = smem_vals[tmp_2849 * WPT + 6u]; let tmp_2870 = keys[6] < tmp_2868 || (keys[6] == tmp_2868 && values[6] < tmp_2869); if tmp_2848 == tmp_2870 { keys[6] = tmp_2868; values[6] = tmp_2869; } let tmp_2871 = smem_keys[tmp_2849 * WPT + 7u]; let tmp_2872 = smem_vals[tmp_2849 * WPT + 7u]; let tmp_2873 = keys[7] < tmp_2871 || (keys[7] == tmp_2871 && values[7] < tmp_2872); if tmp_2848 == tmp_2873 { keys[7] = tmp_2871; values[7] = tmp_2872; } let tmp_2874 = smem_keys[tmp_2849 * WPT + 8u]; let tmp_2875 = smem_vals[tmp_2849 * WPT + 8u]; let tmp_2876 = keys[8] < tmp_2874 || (keys[8] == tmp_2874 && values[8] < tmp_2875); if tmp_2848 == tmp_2876 { keys[8] = tmp_2874; values[8] = tmp_2875; } let tmp_2877 = smem_keys[tmp_2849 * WPT + 9u]; let tmp_2878 = smem_vals[tmp_2849 * WPT + 9u]; let tmp_2879 = keys[9] < tmp_2877 || (keys[9] == tmp_2877 && values[9] < tmp_2878); if tmp_2848 == tmp_2879 { keys[9] = tmp_2877; values[9] = tmp_2878; } let tmp_2880 = smem_keys[tmp_2849 * WPT + 10u]; let tmp_2881 = smem_vals[tmp_2849 * WPT + 10u]; let tmp_2882 = keys[10] < tmp_2880 || (keys[10] == tmp_2880 && values[10] < tmp_2881); if tmp_2848 == tmp_2882 { keys[10] = tmp_2880; values[10] = tmp_2881; } let tmp_2883 = smem_keys[tmp_2849 * WPT + 11u]; let tmp_2884 = smem_vals[tmp_2849 * WPT + 11u]; let tmp_2885 = keys[11] < tmp_2883 || (keys[11] == tmp_2883 && values[11] < tmp_2884); if tmp_2848 == tmp_2885 { keys[11] = tmp_2883; values[11] = tmp_2884; } let tmp_2886 = smem_keys[tmp_2849 * WPT + 12u]; let tmp_2887 = smem_vals[tmp_2849 * WPT + 12u]; let tmp_2888 = keys[12] < tmp_2886 || (keys[12] == tmp_2886 && values[12] < tmp_2887); if tmp_2848 == tmp_2888 { keys[12] = tmp_2886; values[12] = tmp_2887; } let tmp_2889 = smem_keys[tmp_2849 * WPT + 13u]; let tmp_2890 = smem_vals[tmp_2849 * WPT + 13u]; let tmp_2891 = keys[13] < tmp_2889 || (keys[13] == tmp_2889 && values[13] < tmp_2890); if tmp_2848 == tmp_2891 { keys[13] = tmp_2889; values[13] = tmp_2890; } let tmp_2892 = smem_keys[tmp_2849 * WPT + 14u]; let tmp_2893 = smem_vals[tmp_2849 * WPT + 14u]; let tmp_2894 = keys[14] < tmp_2892 || (keys[14] == tmp_2892 && values[14] < tmp_2893); if tmp_2848 == tmp_2894 { keys[14] = tmp_2892; values[14] = tmp_2893; } let tmp_2895 = smem_keys[tmp_2849 * WPT + 15u]; let tmp_2896 = smem_vals[tmp_2849 * WPT + 15u]; let tmp_2897 = keys[15] < tmp_2895 || (keys[15] == tmp_2895 && values[15] < tmp_2896); if tmp_2848 == tmp_2897 { keys[15] = tmp_2895; values[15] = tmp_2896; } let tmp_2898 = smem_keys[tmp_2849 * WPT + 16u]; let tmp_2899 = smem_vals[tmp_2849 * WPT + 16u]; let tmp_2900 = keys[16] < tmp_2898 || (keys[16] == tmp_2898 && values[16] < tmp_2899); if tmp_2848 == tmp_2900 { keys[16] = tmp_2898; values[16] = tmp_2899; } let tmp_2901 = smem_keys[tmp_2849 * WPT + 17u]; let tmp_2902 = smem_vals[tmp_2849 * WPT + 17u]; let tmp_2903 = keys[17] < tmp_2901 || (keys[17] == tmp_2901 && values[17] < tmp_2902); if tmp_2848 == tmp_2903 { keys[17] = tmp_2901; values[17] = tmp_2902; } let tmp_2904 = smem_keys[tmp_2849 * WPT + 18u]; let tmp_2905 = smem_vals[tmp_2849 * WPT + 18u]; let tmp_2906 = keys[18] < tmp_2904 || (keys[18] == tmp_2904 && values[18] < tmp_2905); if tmp_2848 == tmp_2906 { keys[18] = tmp_2904; values[18] = tmp_2905; } let tmp_2907 = smem_keys[tmp_2849 * WPT + 19u]; let tmp_2908 = smem_vals[tmp_2849 * WPT + 19u]; let tmp_2909 = keys[19] < tmp_2907 || (keys[19] == tmp_2907 && values[19] < tmp_2908); if tmp_2848 == tmp_2909 { keys[19] = tmp_2907; values[19] = tmp_2908; } let tmp_2910 = smem_keys[tmp_2849 * WPT + 20u]; let tmp_2911 = smem_vals[tmp_2849 * WPT + 20u]; let tmp_2912 = keys[20] < tmp_2910 || (keys[20] == tmp_2910 && values[20] < tmp_2911); if tmp_2848 == tmp_2912 { keys[20] = tmp_2910; values[20] = tmp_2911; } let tmp_2913 = smem_keys[tmp_2849 * WPT + 21u]; let tmp_2914 = smem_vals[tmp_2849 * WPT + 21u]; let tmp_2915 = keys[21] < tmp_2913 || (keys[21] == tmp_2913 && values[21] < tmp_2914); if tmp_2848 == tmp_2915 { keys[21] = tmp_2913; values[21] = tmp_2914; } let tmp_2916 = smem_keys[tmp_2849 * WPT + 22u]; let tmp_2917 = smem_vals[tmp_2849 * WPT + 22u]; let tmp_2918 = keys[22] < tmp_2916 || (keys[22] == tmp_2916 && values[22] < tmp_2917); if tmp_2848 == tmp_2918 { keys[22] = tmp_2916; values[22] = tmp_2917; } let tmp_2919 = smem_keys[tmp_2849 * WPT + 23u]; let tmp_2920 = smem_vals[tmp_2849 * WPT + 23u]; let tmp_2921 = keys[23] < tmp_2919 || (keys[23] == tmp_2919 && values[23] < tmp_2920); if tmp_2848 == tmp_2921 { keys[23] = tmp_2919; values[23] = tmp_2920; } let tmp_2922 = smem_keys[tmp_2849 * WPT + 24u]; let tmp_2923 = smem_vals[tmp_2849 * WPT + 24u]; let tmp_2924 = keys[24] < tmp_2922 || (keys[24] == tmp_2922 && values[24] < tmp_2923); if tmp_2848 == tmp_2924 { keys[24] = tmp_2922; values[24] = tmp_2923; } let tmp_2925 = smem_keys[tmp_2849 * WPT + 25u]; let tmp_2926 = smem_vals[tmp_2849 * WPT + 25u]; let tmp_2927 = keys[25] < tmp_2925 || (keys[25] == tmp_2925 && values[25] < tmp_2926); if tmp_2848 == tmp_2927 { keys[25] = tmp_2925; values[25] = tmp_2926; } let tmp_2928 = smem_keys[tmp_2849 * WPT + 26u]; let tmp_2929 = smem_vals[tmp_2849 * WPT + 26u]; let tmp_2930 = keys[26] < tmp_2928 || (keys[26] == tmp_2928 && values[26] < tmp_2929); if tmp_2848 == tmp_2930 { keys[26] = tmp_2928; values[26] = tmp_2929; } let tmp_2931 = smem_keys[tmp_2849 * WPT + 27u]; let tmp_2932 = smem_vals[tmp_2849 * WPT + 27u]; let tmp_2933 = keys[27] < tmp_2931 || (keys[27] == tmp_2931 && values[27] < tmp_2932); if tmp_2848 == tmp_2933 { keys[27] = tmp_2931; values[27] = tmp_2932; } let tmp_2934 = smem_keys[tmp_2849 * WPT + 28u]; let tmp_2935 = smem_vals[tmp_2849 * WPT + 28u]; let tmp_2936 = keys[28] < tmp_2934 || (keys[28] == tmp_2934 && values[28] < tmp_2935); if tmp_2848 == tmp_2936 { keys[28] = tmp_2934; values[28] = tmp_2935; } let tmp_2937 = smem_keys[tmp_2849 * WPT + 29u]; let tmp_2938 = smem_vals[tmp_2849 * WPT + 29u]; let tmp_2939 = keys[29] < tmp_2937 || (keys[29] == tmp_2937 && values[29] < tmp_2938); if tmp_2848 == tmp_2939 { keys[29] = tmp_2937; values[29] = tmp_2938; } let tmp_2940 = smem_keys[tmp_2849 * WPT + 30u]; let tmp_2941 = smem_vals[tmp_2849 * WPT + 30u]; let tmp_2942 = keys[30] < tmp_2940 || (keys[30] == tmp_2940 && values[30] < tmp_2941); if tmp_2848 == tmp_2942 { keys[30] = tmp_2940; values[30] = tmp_2941; } let tmp_2943 = smem_keys[tmp_2849 * WPT + 31u]; let tmp_2944 = smem_vals[tmp_2849 * WPT + 31u]; let tmp_2945 = keys[31] < tmp_2943 || (keys[31] == tmp_2943 && values[31] < tmp_2944); if tmp_2848 == tmp_2945 { keys[31] = tmp_2943; values[31] = tmp_2944; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_2946 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_2947 = seg_base + (local_tid ^ 8u); let tmp_2948 = smem_keys[tmp_2947 * WPT + 0u]; let tmp_2949 = smem_vals[tmp_2947 * WPT + 0u]; let tmp_2950 = keys[0] < tmp_2948 || (keys[0] == tmp_2948 && values[0] < tmp_2949); if tmp_2946 == tmp_2950 { keys[0] = tmp_2948; values[0] = tmp_2949; } let tmp_2951 = smem_keys[tmp_2947 * WPT + 1u]; let tmp_2952 = smem_vals[tmp_2947 * WPT + 1u]; let tmp_2953 = keys[1] < tmp_2951 || (keys[1] == tmp_2951 && values[1] < tmp_2952); if tmp_2946 == tmp_2953 { keys[1] = tmp_2951; values[1] = tmp_2952; } let tmp_2954 = smem_keys[tmp_2947 * WPT + 2u]; let tmp_2955 = smem_vals[tmp_2947 * WPT + 2u]; let tmp_2956 = keys[2] < tmp_2954 || (keys[2] == tmp_2954 && values[2] < tmp_2955); if tmp_2946 == tmp_2956 { keys[2] = tmp_2954; values[2] = tmp_2955; } let tmp_2957 = smem_keys[tmp_2947 * WPT + 3u]; let tmp_2958 = smem_vals[tmp_2947 * WPT + 3u]; let tmp_2959 = keys[3] < tmp_2957 || (keys[3] == tmp_2957 && values[3] < tmp_2958); if tmp_2946 == tmp_2959 { keys[3] = tmp_2957; values[3] = tmp_2958; } let tmp_2960 = smem_keys[tmp_2947 * WPT + 4u]; let tmp_2961 = smem_vals[tmp_2947 * WPT + 4u]; let tmp_2962 = keys[4] < tmp_2960 || (keys[4] == tmp_2960 && values[4] < tmp_2961); if tmp_2946 == tmp_2962 { keys[4] = tmp_2960; values[4] = tmp_2961; } let tmp_2963 = smem_keys[tmp_2947 * WPT + 5u]; let tmp_2964 = smem_vals[tmp_2947 * WPT + 5u]; let tmp_2965 = keys[5] < tmp_2963 || (keys[5] == tmp_2963 && values[5] < tmp_2964); if tmp_2946 == tmp_2965 { keys[5] = tmp_2963; values[5] = tmp_2964; } let tmp_2966 = smem_keys[tmp_2947 * WPT + 6u]; let tmp_2967 = smem_vals[tmp_2947 * WPT + 6u]; let tmp_2968 = keys[6] < tmp_2966 || (keys[6] == tmp_2966 && values[6] < tmp_2967); if tmp_2946 == tmp_2968 { keys[6] = tmp_2966; values[6] = tmp_2967; } let tmp_2969 = smem_keys[tmp_2947 * WPT + 7u]; let tmp_2970 = smem_vals[tmp_2947 * WPT + 7u]; let tmp_2971 = keys[7] < tmp_2969 || (keys[7] == tmp_2969 && values[7] < tmp_2970); if tmp_2946 == tmp_2971 { keys[7] = tmp_2969; values[7] = tmp_2970; } let tmp_2972 = smem_keys[tmp_2947 * WPT + 8u]; let tmp_2973 = smem_vals[tmp_2947 * WPT + 8u]; let tmp_2974 = keys[8] < tmp_2972 || (keys[8] == tmp_2972 && values[8] < tmp_2973); if tmp_2946 == tmp_2974 { keys[8] = tmp_2972; values[8] = tmp_2973; } let tmp_2975 = smem_keys[tmp_2947 * WPT + 9u]; let tmp_2976 = smem_vals[tmp_2947 * WPT + 9u]; let tmp_2977 = keys[9] < tmp_2975 || (keys[9] == tmp_2975 && values[9] < tmp_2976); if tmp_2946 == tmp_2977 { keys[9] = tmp_2975; values[9] = tmp_2976; } let tmp_2978 = smem_keys[tmp_2947 * WPT + 10u]; let tmp_2979 = smem_vals[tmp_2947 * WPT + 10u]; let tmp_2980 = keys[10] < tmp_2978 || (keys[10] == tmp_2978 && values[10] < tmp_2979); if tmp_2946 == tmp_2980 { keys[10] = tmp_2978; values[10] = tmp_2979; } let tmp_2981 = smem_keys[tmp_2947 * WPT + 11u]; let tmp_2982 = smem_vals[tmp_2947 * WPT + 11u]; let tmp_2983 = keys[11] < tmp_2981 || (keys[11] == tmp_2981 && values[11] < tmp_2982); if tmp_2946 == tmp_2983 { keys[11] = tmp_2981; values[11] = tmp_2982; } let tmp_2984 = smem_keys[tmp_2947 * WPT + 12u]; let tmp_2985 = smem_vals[tmp_2947 * WPT + 12u]; let tmp_2986 = keys[12] < tmp_2984 || (keys[12] == tmp_2984 && values[12] < tmp_2985); if tmp_2946 == tmp_2986 { keys[12] = tmp_2984; values[12] = tmp_2985; } let tmp_2987 = smem_keys[tmp_2947 * WPT + 13u]; let tmp_2988 = smem_vals[tmp_2947 * WPT + 13u]; let tmp_2989 = keys[13] < tmp_2987 || (keys[13] == tmp_2987 && values[13] < tmp_2988); if tmp_2946 == tmp_2989 { keys[13] = tmp_2987; values[13] = tmp_2988; } let tmp_2990 = smem_keys[tmp_2947 * WPT + 14u]; let tmp_2991 = smem_vals[tmp_2947 * WPT + 14u]; let tmp_2992 = keys[14] < tmp_2990 || (keys[14] == tmp_2990 && values[14] < tmp_2991); if tmp_2946 == tmp_2992 { keys[14] = tmp_2990; values[14] = tmp_2991; } let tmp_2993 = smem_keys[tmp_2947 * WPT + 15u]; let tmp_2994 = smem_vals[tmp_2947 * WPT + 15u]; let tmp_2995 = keys[15] < tmp_2993 || (keys[15] == tmp_2993 && values[15] < tmp_2994); if tmp_2946 == tmp_2995 { keys[15] = tmp_2993; values[15] = tmp_2994; } let tmp_2996 = smem_keys[tmp_2947 * WPT + 16u]; let tmp_2997 = smem_vals[tmp_2947 * WPT + 16u]; let tmp_2998 = keys[16] < tmp_2996 || (keys[16] == tmp_2996 && values[16] < tmp_2997); if tmp_2946 == tmp_2998 { keys[16] = tmp_2996; values[16] = tmp_2997; } let tmp_2999 = smem_keys[tmp_2947 * WPT + 17u]; let tmp_3000 = smem_vals[tmp_2947 * WPT + 17u]; let tmp_3001 = keys[17] < tmp_2999 || (keys[17] == tmp_2999 && values[17] < tmp_3000); if tmp_2946 == tmp_3001 { keys[17] = tmp_2999; values[17] = tmp_3000; } let tmp_3002 = smem_keys[tmp_2947 * WPT + 18u]; let tmp_3003 = smem_vals[tmp_2947 * WPT + 18u]; let tmp_3004 = keys[18] < tmp_3002 || (keys[18] == tmp_3002 && values[18] < tmp_3003); if tmp_2946 == tmp_3004 { keys[18] = tmp_3002; values[18] = tmp_3003; } let tmp_3005 = smem_keys[tmp_2947 * WPT + 19u]; let tmp_3006 = smem_vals[tmp_2947 * WPT + 19u]; let tmp_3007 = keys[19] < tmp_3005 || (keys[19] == tmp_3005 && values[19] < tmp_3006); if tmp_2946 == tmp_3007 { keys[19] = tmp_3005; values[19] = tmp_3006; } let tmp_3008 = smem_keys[tmp_2947 * WPT + 20u]; let tmp_3009 = smem_vals[tmp_2947 * WPT + 20u]; let tmp_3010 = keys[20] < tmp_3008 || (keys[20] == tmp_3008 && values[20] < tmp_3009); if tmp_2946 == tmp_3010 { keys[20] = tmp_3008; values[20] = tmp_3009; } let tmp_3011 = smem_keys[tmp_2947 * WPT + 21u]; let tmp_3012 = smem_vals[tmp_2947 * WPT + 21u]; let tmp_3013 = keys[21] < tmp_3011 || (keys[21] == tmp_3011 && values[21] < tmp_3012); if tmp_2946 == tmp_3013 { keys[21] = tmp_3011; values[21] = tmp_3012; } let tmp_3014 = smem_keys[tmp_2947 * WPT + 22u]; let tmp_3015 = smem_vals[tmp_2947 * WPT + 22u]; let tmp_3016 = keys[22] < tmp_3014 || (keys[22] == tmp_3014 && values[22] < tmp_3015); if tmp_2946 == tmp_3016 { keys[22] = tmp_3014; values[22] = tmp_3015; } let tmp_3017 = smem_keys[tmp_2947 * WPT + 23u]; let tmp_3018 = smem_vals[tmp_2947 * WPT + 23u]; let tmp_3019 = keys[23] < tmp_3017 || (keys[23] == tmp_3017 && values[23] < tmp_3018); if tmp_2946 == tmp_3019 { keys[23] = tmp_3017; values[23] = tmp_3018; } let tmp_3020 = smem_keys[tmp_2947 * WPT + 24u]; let tmp_3021 = smem_vals[tmp_2947 * WPT + 24u]; let tmp_3022 = keys[24] < tmp_3020 || (keys[24] == tmp_3020 && values[24] < tmp_3021); if tmp_2946 == tmp_3022 { keys[24] = tmp_3020; values[24] = tmp_3021; } let tmp_3023 = smem_keys[tmp_2947 * WPT + 25u]; let tmp_3024 = smem_vals[tmp_2947 * WPT + 25u]; let tmp_3025 = keys[25] < tmp_3023 || (keys[25] == tmp_3023 && values[25] < tmp_3024); if tmp_2946 == tmp_3025 { keys[25] = tmp_3023; values[25] = tmp_3024; } let tmp_3026 = smem_keys[tmp_2947 * WPT + 26u]; let tmp_3027 = smem_vals[tmp_2947 * WPT + 26u]; let tmp_3028 = keys[26] < tmp_3026 || (keys[26] == tmp_3026 && values[26] < tmp_3027); if tmp_2946 == tmp_3028 { keys[26] = tmp_3026; values[26] = tmp_3027; } let tmp_3029 = smem_keys[tmp_2947 * WPT + 27u]; let tmp_3030 = smem_vals[tmp_2947 * WPT + 27u]; let tmp_3031 = keys[27] < tmp_3029 || (keys[27] == tmp_3029 && values[27] < tmp_3030); if tmp_2946 == tmp_3031 { keys[27] = tmp_3029; values[27] = tmp_3030; } let tmp_3032 = smem_keys[tmp_2947 * WPT + 28u]; let tmp_3033 = smem_vals[tmp_2947 * WPT + 28u]; let tmp_3034 = keys[28] < tmp_3032 || (keys[28] == tmp_3032 && values[28] < tmp_3033); if tmp_2946 == tmp_3034 { keys[28] = tmp_3032; values[28] = tmp_3033; } let tmp_3035 = smem_keys[tmp_2947 * WPT + 29u]; let tmp_3036 = smem_vals[tmp_2947 * WPT + 29u]; let tmp_3037 = keys[29] < tmp_3035 || (keys[29] == tmp_3035 && values[29] < tmp_3036); if tmp_2946 == tmp_3037 { keys[29] = tmp_3035; values[29] = tmp_3036; } let tmp_3038 = smem_keys[tmp_2947 * WPT + 30u]; let tmp_3039 = smem_vals[tmp_2947 * WPT + 30u]; let tmp_3040 = keys[30] < tmp_3038 || (keys[30] == tmp_3038 && values[30] < tmp_3039); if tmp_2946 == tmp_3040 { keys[30] = tmp_3038; values[30] = tmp_3039; } let tmp_3041 = smem_keys[tmp_2947 * WPT + 31u]; let tmp_3042 = smem_vals[tmp_2947 * WPT + 31u]; let tmp_3043 = keys[31] < tmp_3041 || (keys[31] == tmp_3041 && values[31] < tmp_3042); if tmp_2946 == tmp_3043 { keys[31] = tmp_3041; values[31] = tmp_3042; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_3044 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_3045 = seg_base + (local_tid ^ 4u); let tmp_3046 = smem_keys[tmp_3045 * WPT + 0u]; let tmp_3047 = smem_vals[tmp_3045 * WPT + 0u]; let tmp_3048 = keys[0] < tmp_3046 || (keys[0] == tmp_3046 && values[0] < tmp_3047); if tmp_3044 == tmp_3048 { keys[0] = tmp_3046; values[0] = tmp_3047; } let tmp_3049 = smem_keys[tmp_3045 * WPT + 1u]; let tmp_3050 = smem_vals[tmp_3045 * WPT + 1u]; let tmp_3051 = keys[1] < tmp_3049 || (keys[1] == tmp_3049 && values[1] < tmp_3050); if tmp_3044 == tmp_3051 { keys[1] = tmp_3049; values[1] = tmp_3050; } let tmp_3052 = smem_keys[tmp_3045 * WPT + 2u]; let tmp_3053 = smem_vals[tmp_3045 * WPT + 2u]; let tmp_3054 = keys[2] < tmp_3052 || (keys[2] == tmp_3052 && values[2] < tmp_3053); if tmp_3044 == tmp_3054 { keys[2] = tmp_3052; values[2] = tmp_3053; } let tmp_3055 = smem_keys[tmp_3045 * WPT + 3u]; let tmp_3056 = smem_vals[tmp_3045 * WPT + 3u]; let tmp_3057 = keys[3] < tmp_3055 || (keys[3] == tmp_3055 && values[3] < tmp_3056); if tmp_3044 == tmp_3057 { keys[3] = tmp_3055; values[3] = tmp_3056; } let tmp_3058 = smem_keys[tmp_3045 * WPT + 4u]; let tmp_3059 = smem_vals[tmp_3045 * WPT + 4u]; let tmp_3060 = keys[4] < tmp_3058 || (keys[4] == tmp_3058 && values[4] < tmp_3059); if tmp_3044 == tmp_3060 { keys[4] = tmp_3058; values[4] = tmp_3059; } let tmp_3061 = smem_keys[tmp_3045 * WPT + 5u]; let tmp_3062 = smem_vals[tmp_3045 * WPT + 5u]; let tmp_3063 = keys[5] < tmp_3061 || (keys[5] == tmp_3061 && values[5] < tmp_3062); if tmp_3044 == tmp_3063 { keys[5] = tmp_3061; values[5] = tmp_3062; } let tmp_3064 = smem_keys[tmp_3045 * WPT + 6u]; let tmp_3065 = smem_vals[tmp_3045 * WPT + 6u]; let tmp_3066 = keys[6] < tmp_3064 || (keys[6] == tmp_3064 && values[6] < tmp_3065); if tmp_3044 == tmp_3066 { keys[6] = tmp_3064; values[6] = tmp_3065; } let tmp_3067 = smem_keys[tmp_3045 * WPT + 7u]; let tmp_3068 = smem_vals[tmp_3045 * WPT + 7u]; let tmp_3069 = keys[7] < tmp_3067 || (keys[7] == tmp_3067 && values[7] < tmp_3068); if tmp_3044 == tmp_3069 { keys[7] = tmp_3067; values[7] = tmp_3068; } let tmp_3070 = smem_keys[tmp_3045 * WPT + 8u]; let tmp_3071 = smem_vals[tmp_3045 * WPT + 8u]; let tmp_3072 = keys[8] < tmp_3070 || (keys[8] == tmp_3070 && values[8] < tmp_3071); if tmp_3044 == tmp_3072 { keys[8] = tmp_3070; values[8] = tmp_3071; } let tmp_3073 = smem_keys[tmp_3045 * WPT + 9u]; let tmp_3074 = smem_vals[tmp_3045 * WPT + 9u]; let tmp_3075 = keys[9] < tmp_3073 || (keys[9] == tmp_3073 && values[9] < tmp_3074); if tmp_3044 == tmp_3075 { keys[9] = tmp_3073; values[9] = tmp_3074; } let tmp_3076 = smem_keys[tmp_3045 * WPT + 10u]; let tmp_3077 = smem_vals[tmp_3045 * WPT + 10u]; let tmp_3078 = keys[10] < tmp_3076 || (keys[10] == tmp_3076 && values[10] < tmp_3077); if tmp_3044 == tmp_3078 { keys[10] = tmp_3076; values[10] = tmp_3077; } let tmp_3079 = smem_keys[tmp_3045 * WPT + 11u]; let tmp_3080 = smem_vals[tmp_3045 * WPT + 11u]; let tmp_3081 = keys[11] < tmp_3079 || (keys[11] == tmp_3079 && values[11] < tmp_3080); if tmp_3044 == tmp_3081 { keys[11] = tmp_3079; values[11] = tmp_3080; } let tmp_3082 = smem_keys[tmp_3045 * WPT + 12u]; let tmp_3083 = smem_vals[tmp_3045 * WPT + 12u]; let tmp_3084 = keys[12] < tmp_3082 || (keys[12] == tmp_3082 && values[12] < tmp_3083); if tmp_3044 == tmp_3084 { keys[12] = tmp_3082; values[12] = tmp_3083; } let tmp_3085 = smem_keys[tmp_3045 * WPT + 13u]; let tmp_3086 = smem_vals[tmp_3045 * WPT + 13u]; let tmp_3087 = keys[13] < tmp_3085 || (keys[13] == tmp_3085 && values[13] < tmp_3086); if tmp_3044 == tmp_3087 { keys[13] = tmp_3085; values[13] = tmp_3086; } let tmp_3088 = smem_keys[tmp_3045 * WPT + 14u]; let tmp_3089 = smem_vals[tmp_3045 * WPT + 14u]; let tmp_3090 = keys[14] < tmp_3088 || (keys[14] == tmp_3088 && values[14] < tmp_3089); if tmp_3044 == tmp_3090 { keys[14] = tmp_3088; values[14] = tmp_3089; } let tmp_3091 = smem_keys[tmp_3045 * WPT + 15u]; let tmp_3092 = smem_vals[tmp_3045 * WPT + 15u]; let tmp_3093 = keys[15] < tmp_3091 || (keys[15] == tmp_3091 && values[15] < tmp_3092); if tmp_3044 == tmp_3093 { keys[15] = tmp_3091; values[15] = tmp_3092; } let tmp_3094 = smem_keys[tmp_3045 * WPT + 16u]; let tmp_3095 = smem_vals[tmp_3045 * WPT + 16u]; let tmp_3096 = keys[16] < tmp_3094 || (keys[16] == tmp_3094 && values[16] < tmp_3095); if tmp_3044 == tmp_3096 { keys[16] = tmp_3094; values[16] = tmp_3095; } let tmp_3097 = smem_keys[tmp_3045 * WPT + 17u]; let tmp_3098 = smem_vals[tmp_3045 * WPT + 17u]; let tmp_3099 = keys[17] < tmp_3097 || (keys[17] == tmp_3097 && values[17] < tmp_3098); if tmp_3044 == tmp_3099 { keys[17] = tmp_3097; values[17] = tmp_3098; } let tmp_3100 = smem_keys[tmp_3045 * WPT + 18u]; let tmp_3101 = smem_vals[tmp_3045 * WPT + 18u]; let tmp_3102 = keys[18] < tmp_3100 || (keys[18] == tmp_3100 && values[18] < tmp_3101); if tmp_3044 == tmp_3102 { keys[18] = tmp_3100; values[18] = tmp_3101; } let tmp_3103 = smem_keys[tmp_3045 * WPT + 19u]; let tmp_3104 = smem_vals[tmp_3045 * WPT + 19u]; let tmp_3105 = keys[19] < tmp_3103 || (keys[19] == tmp_3103 && values[19] < tmp_3104); if tmp_3044 == tmp_3105 { keys[19] = tmp_3103; values[19] = tmp_3104; } let tmp_3106 = smem_keys[tmp_3045 * WPT + 20u]; let tmp_3107 = smem_vals[tmp_3045 * WPT + 20u]; let tmp_3108 = keys[20] < tmp_3106 || (keys[20] == tmp_3106 && values[20] < tmp_3107); if tmp_3044 == tmp_3108 { keys[20] = tmp_3106; values[20] = tmp_3107; } let tmp_3109 = smem_keys[tmp_3045 * WPT + 21u]; let tmp_3110 = smem_vals[tmp_3045 * WPT + 21u]; let tmp_3111 = keys[21] < tmp_3109 || (keys[21] == tmp_3109 && values[21] < tmp_3110); if tmp_3044 == tmp_3111 { keys[21] = tmp_3109; values[21] = tmp_3110; } let tmp_3112 = smem_keys[tmp_3045 * WPT + 22u]; let tmp_3113 = smem_vals[tmp_3045 * WPT + 22u]; let tmp_3114 = keys[22] < tmp_3112 || (keys[22] == tmp_3112 && values[22] < tmp_3113); if tmp_3044 == tmp_3114 { keys[22] = tmp_3112; values[22] = tmp_3113; } let tmp_3115 = smem_keys[tmp_3045 * WPT + 23u]; let tmp_3116 = smem_vals[tmp_3045 * WPT + 23u]; let tmp_3117 = keys[23] < tmp_3115 || (keys[23] == tmp_3115 && values[23] < tmp_3116); if tmp_3044 == tmp_3117 { keys[23] = tmp_3115; values[23] = tmp_3116; } let tmp_3118 = smem_keys[tmp_3045 * WPT + 24u]; let tmp_3119 = smem_vals[tmp_3045 * WPT + 24u]; let tmp_3120 = keys[24] < tmp_3118 || (keys[24] == tmp_3118 && values[24] < tmp_3119); if tmp_3044 == tmp_3120 { keys[24] = tmp_3118; values[24] = tmp_3119; } let tmp_3121 = smem_keys[tmp_3045 * WPT + 25u]; let tmp_3122 = smem_vals[tmp_3045 * WPT + 25u]; let tmp_3123 = keys[25] < tmp_3121 || (keys[25] == tmp_3121 && values[25] < tmp_3122); if tmp_3044 == tmp_3123 { keys[25] = tmp_3121; values[25] = tmp_3122; } let tmp_3124 = smem_keys[tmp_3045 * WPT + 26u]; let tmp_3125 = smem_vals[tmp_3045 * WPT + 26u]; let tmp_3126 = keys[26] < tmp_3124 || (keys[26] == tmp_3124 && values[26] < tmp_3125); if tmp_3044 == tmp_3126 { keys[26] = tmp_3124; values[26] = tmp_3125; } let tmp_3127 = smem_keys[tmp_3045 * WPT + 27u]; let tmp_3128 = smem_vals[tmp_3045 * WPT + 27u]; let tmp_3129 = keys[27] < tmp_3127 || (keys[27] == tmp_3127 && values[27] < tmp_3128); if tmp_3044 == tmp_3129 { keys[27] = tmp_3127; values[27] = tmp_3128; } let tmp_3130 = smem_keys[tmp_3045 * WPT + 28u]; let tmp_3131 = smem_vals[tmp_3045 * WPT + 28u]; let tmp_3132 = keys[28] < tmp_3130 || (keys[28] == tmp_3130 && values[28] < tmp_3131); if tmp_3044 == tmp_3132 { keys[28] = tmp_3130; values[28] = tmp_3131; } let tmp_3133 = smem_keys[tmp_3045 * WPT + 29u]; let tmp_3134 = smem_vals[tmp_3045 * WPT + 29u]; let tmp_3135 = keys[29] < tmp_3133 || (keys[29] == tmp_3133 && values[29] < tmp_3134); if tmp_3044 == tmp_3135 { keys[29] = tmp_3133; values[29] = tmp_3134; } let tmp_3136 = smem_keys[tmp_3045 * WPT + 30u]; let tmp_3137 = smem_vals[tmp_3045 * WPT + 30u]; let tmp_3138 = keys[30] < tmp_3136 || (keys[30] == tmp_3136 && values[30] < tmp_3137); if tmp_3044 == tmp_3138 { keys[30] = tmp_3136; values[30] = tmp_3137; } let tmp_3139 = smem_keys[tmp_3045 * WPT + 31u]; let tmp_3140 = smem_vals[tmp_3045 * WPT + 31u]; let tmp_3141 = keys[31] < tmp_3139 || (keys[31] == tmp_3139 && values[31] < tmp_3140); if tmp_3044 == tmp_3141 { keys[31] = tmp_3139; values[31] = tmp_3140; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_3142 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_3143 = seg_base + (local_tid ^ 2u); let tmp_3144 = smem_keys[tmp_3143 * WPT + 0u]; let tmp_3145 = smem_vals[tmp_3143 * WPT + 0u]; let tmp_3146 = keys[0] < tmp_3144 || (keys[0] == tmp_3144 && values[0] < tmp_3145); if tmp_3142 == tmp_3146 { keys[0] = tmp_3144; values[0] = tmp_3145; } let tmp_3147 = smem_keys[tmp_3143 * WPT + 1u]; let tmp_3148 = smem_vals[tmp_3143 * WPT + 1u]; let tmp_3149 = keys[1] < tmp_3147 || (keys[1] == tmp_3147 && values[1] < tmp_3148); if tmp_3142 == tmp_3149 { keys[1] = tmp_3147; values[1] = tmp_3148; } let tmp_3150 = smem_keys[tmp_3143 * WPT + 2u]; let tmp_3151 = smem_vals[tmp_3143 * WPT + 2u]; let tmp_3152 = keys[2] < tmp_3150 || (keys[2] == tmp_3150 && values[2] < tmp_3151); if tmp_3142 == tmp_3152 { keys[2] = tmp_3150; values[2] = tmp_3151; } let tmp_3153 = smem_keys[tmp_3143 * WPT + 3u]; let tmp_3154 = smem_vals[tmp_3143 * WPT + 3u]; let tmp_3155 = keys[3] < tmp_3153 || (keys[3] == tmp_3153 && values[3] < tmp_3154); if tmp_3142 == tmp_3155 { keys[3] = tmp_3153; values[3] = tmp_3154; } let tmp_3156 = smem_keys[tmp_3143 * WPT + 4u]; let tmp_3157 = smem_vals[tmp_3143 * WPT + 4u]; let tmp_3158 = keys[4] < tmp_3156 || (keys[4] == tmp_3156 && values[4] < tmp_3157); if tmp_3142 == tmp_3158 { keys[4] = tmp_3156; values[4] = tmp_3157; } let tmp_3159 = smem_keys[tmp_3143 * WPT + 5u]; let tmp_3160 = smem_vals[tmp_3143 * WPT + 5u]; let tmp_3161 = keys[5] < tmp_3159 || (keys[5] == tmp_3159 && values[5] < tmp_3160); if tmp_3142 == tmp_3161 { keys[5] = tmp_3159; values[5] = tmp_3160; } let tmp_3162 = smem_keys[tmp_3143 * WPT + 6u]; let tmp_3163 = smem_vals[tmp_3143 * WPT + 6u]; let tmp_3164 = keys[6] < tmp_3162 || (keys[6] == tmp_3162 && values[6] < tmp_3163); if tmp_3142 == tmp_3164 { keys[6] = tmp_3162; values[6] = tmp_3163; } let tmp_3165 = smem_keys[tmp_3143 * WPT + 7u]; let tmp_3166 = smem_vals[tmp_3143 * WPT + 7u]; let tmp_3167 = keys[7] < tmp_3165 || (keys[7] == tmp_3165 && values[7] < tmp_3166); if tmp_3142 == tmp_3167 { keys[7] = tmp_3165; values[7] = tmp_3166; } let tmp_3168 = smem_keys[tmp_3143 * WPT + 8u]; let tmp_3169 = smem_vals[tmp_3143 * WPT + 8u]; let tmp_3170 = keys[8] < tmp_3168 || (keys[8] == tmp_3168 && values[8] < tmp_3169); if tmp_3142 == tmp_3170 { keys[8] = tmp_3168; values[8] = tmp_3169; } let tmp_3171 = smem_keys[tmp_3143 * WPT + 9u]; let tmp_3172 = smem_vals[tmp_3143 * WPT + 9u]; let tmp_3173 = keys[9] < tmp_3171 || (keys[9] == tmp_3171 && values[9] < tmp_3172); if tmp_3142 == tmp_3173 { keys[9] = tmp_3171; values[9] = tmp_3172; } let tmp_3174 = smem_keys[tmp_3143 * WPT + 10u]; let tmp_3175 = smem_vals[tmp_3143 * WPT + 10u]; let tmp_3176 = keys[10] < tmp_3174 || (keys[10] == tmp_3174 && values[10] < tmp_3175); if tmp_3142 == tmp_3176 { keys[10] = tmp_3174; values[10] = tmp_3175; } let tmp_3177 = smem_keys[tmp_3143 * WPT + 11u]; let tmp_3178 = smem_vals[tmp_3143 * WPT + 11u]; let tmp_3179 = keys[11] < tmp_3177 || (keys[11] == tmp_3177 && values[11] < tmp_3178); if tmp_3142 == tmp_3179 { keys[11] = tmp_3177; values[11] = tmp_3178; } let tmp_3180 = smem_keys[tmp_3143 * WPT + 12u]; let tmp_3181 = smem_vals[tmp_3143 * WPT + 12u]; let tmp_3182 = keys[12] < tmp_3180 || (keys[12] == tmp_3180 && values[12] < tmp_3181); if tmp_3142 == tmp_3182 { keys[12] = tmp_3180; values[12] = tmp_3181; } let tmp_3183 = smem_keys[tmp_3143 * WPT + 13u]; let tmp_3184 = smem_vals[tmp_3143 * WPT + 13u]; let tmp_3185 = keys[13] < tmp_3183 || (keys[13] == tmp_3183 && values[13] < tmp_3184); if tmp_3142 == tmp_3185 { keys[13] = tmp_3183; values[13] = tmp_3184; } let tmp_3186 = smem_keys[tmp_3143 * WPT + 14u]; let tmp_3187 = smem_vals[tmp_3143 * WPT + 14u]; let tmp_3188 = keys[14] < tmp_3186 || (keys[14] == tmp_3186 && values[14] < tmp_3187); if tmp_3142 == tmp_3188 { keys[14] = tmp_3186; values[14] = tmp_3187; } let tmp_3189 = smem_keys[tmp_3143 * WPT + 15u]; let tmp_3190 = smem_vals[tmp_3143 * WPT + 15u]; let tmp_3191 = keys[15] < tmp_3189 || (keys[15] == tmp_3189 && values[15] < tmp_3190); if tmp_3142 == tmp_3191 { keys[15] = tmp_3189; values[15] = tmp_3190; } let tmp_3192 = smem_keys[tmp_3143 * WPT + 16u]; let tmp_3193 = smem_vals[tmp_3143 * WPT + 16u]; let tmp_3194 = keys[16] < tmp_3192 || (keys[16] == tmp_3192 && values[16] < tmp_3193); if tmp_3142 == tmp_3194 { keys[16] = tmp_3192; values[16] = tmp_3193; } let tmp_3195 = smem_keys[tmp_3143 * WPT + 17u]; let tmp_3196 = smem_vals[tmp_3143 * WPT + 17u]; let tmp_3197 = keys[17] < tmp_3195 || (keys[17] == tmp_3195 && values[17] < tmp_3196); if tmp_3142 == tmp_3197 { keys[17] = tmp_3195; values[17] = tmp_3196; } let tmp_3198 = smem_keys[tmp_3143 * WPT + 18u]; let tmp_3199 = smem_vals[tmp_3143 * WPT + 18u]; let tmp_3200 = keys[18] < tmp_3198 || (keys[18] == tmp_3198 && values[18] < tmp_3199); if tmp_3142 == tmp_3200 { keys[18] = tmp_3198; values[18] = tmp_3199; } let tmp_3201 = smem_keys[tmp_3143 * WPT + 19u]; let tmp_3202 = smem_vals[tmp_3143 * WPT + 19u]; let tmp_3203 = keys[19] < tmp_3201 || (keys[19] == tmp_3201 && values[19] < tmp_3202); if tmp_3142 == tmp_3203 { keys[19] = tmp_3201; values[19] = tmp_3202; } let tmp_3204 = smem_keys[tmp_3143 * WPT + 20u]; let tmp_3205 = smem_vals[tmp_3143 * WPT + 20u]; let tmp_3206 = keys[20] < tmp_3204 || (keys[20] == tmp_3204 && values[20] < tmp_3205); if tmp_3142 == tmp_3206 { keys[20] = tmp_3204; values[20] = tmp_3205; } let tmp_3207 = smem_keys[tmp_3143 * WPT + 21u]; let tmp_3208 = smem_vals[tmp_3143 * WPT + 21u]; let tmp_3209 = keys[21] < tmp_3207 || (keys[21] == tmp_3207 && values[21] < tmp_3208); if tmp_3142 == tmp_3209 { keys[21] = tmp_3207; values[21] = tmp_3208; } let tmp_3210 = smem_keys[tmp_3143 * WPT + 22u]; let tmp_3211 = smem_vals[tmp_3143 * WPT + 22u]; let tmp_3212 = keys[22] < tmp_3210 || (keys[22] == tmp_3210 && values[22] < tmp_3211); if tmp_3142 == tmp_3212 { keys[22] = tmp_3210; values[22] = tmp_3211; } let tmp_3213 = smem_keys[tmp_3143 * WPT + 23u]; let tmp_3214 = smem_vals[tmp_3143 * WPT + 23u]; let tmp_3215 = keys[23] < tmp_3213 || (keys[23] == tmp_3213 && values[23] < tmp_3214); if tmp_3142 == tmp_3215 { keys[23] = tmp_3213; values[23] = tmp_3214; } let tmp_3216 = smem_keys[tmp_3143 * WPT + 24u]; let tmp_3217 = smem_vals[tmp_3143 * WPT + 24u]; let tmp_3218 = keys[24] < tmp_3216 || (keys[24] == tmp_3216 && values[24] < tmp_3217); if tmp_3142 == tmp_3218 { keys[24] = tmp_3216; values[24] = tmp_3217; } let tmp_3219 = smem_keys[tmp_3143 * WPT + 25u]; let tmp_3220 = smem_vals[tmp_3143 * WPT + 25u]; let tmp_3221 = keys[25] < tmp_3219 || (keys[25] == tmp_3219 && values[25] < tmp_3220); if tmp_3142 == tmp_3221 { keys[25] = tmp_3219; values[25] = tmp_3220; } let tmp_3222 = smem_keys[tmp_3143 * WPT + 26u]; let tmp_3223 = smem_vals[tmp_3143 * WPT + 26u]; let tmp_3224 = keys[26] < tmp_3222 || (keys[26] == tmp_3222 && values[26] < tmp_3223); if tmp_3142 == tmp_3224 { keys[26] = tmp_3222; values[26] = tmp_3223; } let tmp_3225 = smem_keys[tmp_3143 * WPT + 27u]; let tmp_3226 = smem_vals[tmp_3143 * WPT + 27u]; let tmp_3227 = keys[27] < tmp_3225 || (keys[27] == tmp_3225 && values[27] < tmp_3226); if tmp_3142 == tmp_3227 { keys[27] = tmp_3225; values[27] = tmp_3226; } let tmp_3228 = smem_keys[tmp_3143 * WPT + 28u]; let tmp_3229 = smem_vals[tmp_3143 * WPT + 28u]; let tmp_3230 = keys[28] < tmp_3228 || (keys[28] == tmp_3228 && values[28] < tmp_3229); if tmp_3142 == tmp_3230 { keys[28] = tmp_3228; values[28] = tmp_3229; } let tmp_3231 = smem_keys[tmp_3143 * WPT + 29u]; let tmp_3232 = smem_vals[tmp_3143 * WPT + 29u]; let tmp_3233 = keys[29] < tmp_3231 || (keys[29] == tmp_3231 && values[29] < tmp_3232); if tmp_3142 == tmp_3233 { keys[29] = tmp_3231; values[29] = tmp_3232; } let tmp_3234 = smem_keys[tmp_3143 * WPT + 30u]; let tmp_3235 = smem_vals[tmp_3143 * WPT + 30u]; let tmp_3236 = keys[30] < tmp_3234 || (keys[30] == tmp_3234 && values[30] < tmp_3235); if tmp_3142 == tmp_3236 { keys[30] = tmp_3234; values[30] = tmp_3235; } let tmp_3237 = smem_keys[tmp_3143 * WPT + 31u]; let tmp_3238 = smem_vals[tmp_3143 * WPT + 31u]; let tmp_3239 = keys[31] < tmp_3237 || (keys[31] == tmp_3237 && values[31] < tmp_3238); if tmp_3142 == tmp_3239 { keys[31] = tmp_3237; values[31] = tmp_3238; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_3240 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_3241 = seg_base + (local_tid ^ 1u); let tmp_3242 = smem_keys[tmp_3241 * WPT + 0u]; let tmp_3243 = smem_vals[tmp_3241 * WPT + 0u]; let tmp_3244 = keys[0] < tmp_3242 || (keys[0] == tmp_3242 && values[0] < tmp_3243); if tmp_3240 == tmp_3244 { keys[0] = tmp_3242; values[0] = tmp_3243; } let tmp_3245 = smem_keys[tmp_3241 * WPT + 1u]; let tmp_3246 = smem_vals[tmp_3241 * WPT + 1u]; let tmp_3247 = keys[1] < tmp_3245 || (keys[1] == tmp_3245 && values[1] < tmp_3246); if tmp_3240 == tmp_3247 { keys[1] = tmp_3245; values[1] = tmp_3246; } let tmp_3248 = smem_keys[tmp_3241 * WPT + 2u]; let tmp_3249 = smem_vals[tmp_3241 * WPT + 2u]; let tmp_3250 = keys[2] < tmp_3248 || (keys[2] == tmp_3248 && values[2] < tmp_3249); if tmp_3240 == tmp_3250 { keys[2] = tmp_3248; values[2] = tmp_3249; } let tmp_3251 = smem_keys[tmp_3241 * WPT + 3u]; let tmp_3252 = smem_vals[tmp_3241 * WPT + 3u]; let tmp_3253 = keys[3] < tmp_3251 || (keys[3] == tmp_3251 && values[3] < tmp_3252); if tmp_3240 == tmp_3253 { keys[3] = tmp_3251; values[3] = tmp_3252; } let tmp_3254 = smem_keys[tmp_3241 * WPT + 4u]; let tmp_3255 = smem_vals[tmp_3241 * WPT + 4u]; let tmp_3256 = keys[4] < tmp_3254 || (keys[4] == tmp_3254 && values[4] < tmp_3255); if tmp_3240 == tmp_3256 { keys[4] = tmp_3254; values[4] = tmp_3255; } let tmp_3257 = smem_keys[tmp_3241 * WPT + 5u]; let tmp_3258 = smem_vals[tmp_3241 * WPT + 5u]; let tmp_3259 = keys[5] < tmp_3257 || (keys[5] == tmp_3257 && values[5] < tmp_3258); if tmp_3240 == tmp_3259 { keys[5] = tmp_3257; values[5] = tmp_3258; } let tmp_3260 = smem_keys[tmp_3241 * WPT + 6u]; let tmp_3261 = smem_vals[tmp_3241 * WPT + 6u]; let tmp_3262 = keys[6] < tmp_3260 || (keys[6] == tmp_3260 && values[6] < tmp_3261); if tmp_3240 == tmp_3262 { keys[6] = tmp_3260; values[6] = tmp_3261; } let tmp_3263 = smem_keys[tmp_3241 * WPT + 7u]; let tmp_3264 = smem_vals[tmp_3241 * WPT + 7u]; let tmp_3265 = keys[7] < tmp_3263 || (keys[7] == tmp_3263 && values[7] < tmp_3264); if tmp_3240 == tmp_3265 { keys[7] = tmp_3263; values[7] = tmp_3264; } let tmp_3266 = smem_keys[tmp_3241 * WPT + 8u]; let tmp_3267 = smem_vals[tmp_3241 * WPT + 8u]; let tmp_3268 = keys[8] < tmp_3266 || (keys[8] == tmp_3266 && values[8] < tmp_3267); if tmp_3240 == tmp_3268 { keys[8] = tmp_3266; values[8] = tmp_3267; } let tmp_3269 = smem_keys[tmp_3241 * WPT + 9u]; let tmp_3270 = smem_vals[tmp_3241 * WPT + 9u]; let tmp_3271 = keys[9] < tmp_3269 || (keys[9] == tmp_3269 && values[9] < tmp_3270); if tmp_3240 == tmp_3271 { keys[9] = tmp_3269; values[9] = tmp_3270; } let tmp_3272 = smem_keys[tmp_3241 * WPT + 10u]; let tmp_3273 = smem_vals[tmp_3241 * WPT + 10u]; let tmp_3274 = keys[10] < tmp_3272 || (keys[10] == tmp_3272 && values[10] < tmp_3273); if tmp_3240 == tmp_3274 { keys[10] = tmp_3272; values[10] = tmp_3273; } let tmp_3275 = smem_keys[tmp_3241 * WPT + 11u]; let tmp_3276 = smem_vals[tmp_3241 * WPT + 11u]; let tmp_3277 = keys[11] < tmp_3275 || (keys[11] == tmp_3275 && values[11] < tmp_3276); if tmp_3240 == tmp_3277 { keys[11] = tmp_3275; values[11] = tmp_3276; } let tmp_3278 = smem_keys[tmp_3241 * WPT + 12u]; let tmp_3279 = smem_vals[tmp_3241 * WPT + 12u]; let tmp_3280 = keys[12] < tmp_3278 || (keys[12] == tmp_3278 && values[12] < tmp_3279); if tmp_3240 == tmp_3280 { keys[12] = tmp_3278; values[12] = tmp_3279; } let tmp_3281 = smem_keys[tmp_3241 * WPT + 13u]; let tmp_3282 = smem_vals[tmp_3241 * WPT + 13u]; let tmp_3283 = keys[13] < tmp_3281 || (keys[13] == tmp_3281 && values[13] < tmp_3282); if tmp_3240 == tmp_3283 { keys[13] = tmp_3281; values[13] = tmp_3282; } let tmp_3284 = smem_keys[tmp_3241 * WPT + 14u]; let tmp_3285 = smem_vals[tmp_3241 * WPT + 14u]; let tmp_3286 = keys[14] < tmp_3284 || (keys[14] == tmp_3284 && values[14] < tmp_3285); if tmp_3240 == tmp_3286 { keys[14] = tmp_3284; values[14] = tmp_3285; } let tmp_3287 = smem_keys[tmp_3241 * WPT + 15u]; let tmp_3288 = smem_vals[tmp_3241 * WPT + 15u]; let tmp_3289 = keys[15] < tmp_3287 || (keys[15] == tmp_3287 && values[15] < tmp_3288); if tmp_3240 == tmp_3289 { keys[15] = tmp_3287; values[15] = tmp_3288; } let tmp_3290 = smem_keys[tmp_3241 * WPT + 16u]; let tmp_3291 = smem_vals[tmp_3241 * WPT + 16u]; let tmp_3292 = keys[16] < tmp_3290 || (keys[16] == tmp_3290 && values[16] < tmp_3291); if tmp_3240 == tmp_3292 { keys[16] = tmp_3290; values[16] = tmp_3291; } let tmp_3293 = smem_keys[tmp_3241 * WPT + 17u]; let tmp_3294 = smem_vals[tmp_3241 * WPT + 17u]; let tmp_3295 = keys[17] < tmp_3293 || (keys[17] == tmp_3293 && values[17] < tmp_3294); if tmp_3240 == tmp_3295 { keys[17] = tmp_3293; values[17] = tmp_3294; } let tmp_3296 = smem_keys[tmp_3241 * WPT + 18u]; let tmp_3297 = smem_vals[tmp_3241 * WPT + 18u]; let tmp_3298 = keys[18] < tmp_3296 || (keys[18] == tmp_3296 && values[18] < tmp_3297); if tmp_3240 == tmp_3298 { keys[18] = tmp_3296; values[18] = tmp_3297; } let tmp_3299 = smem_keys[tmp_3241 * WPT + 19u]; let tmp_3300 = smem_vals[tmp_3241 * WPT + 19u]; let tmp_3301 = keys[19] < tmp_3299 || (keys[19] == tmp_3299 && values[19] < tmp_3300); if tmp_3240 == tmp_3301 { keys[19] = tmp_3299; values[19] = tmp_3300; } let tmp_3302 = smem_keys[tmp_3241 * WPT + 20u]; let tmp_3303 = smem_vals[tmp_3241 * WPT + 20u]; let tmp_3304 = keys[20] < tmp_3302 || (keys[20] == tmp_3302 && values[20] < tmp_3303); if tmp_3240 == tmp_3304 { keys[20] = tmp_3302; values[20] = tmp_3303; } let tmp_3305 = smem_keys[tmp_3241 * WPT + 21u]; let tmp_3306 = smem_vals[tmp_3241 * WPT + 21u]; let tmp_3307 = keys[21] < tmp_3305 || (keys[21] == tmp_3305 && values[21] < tmp_3306); if tmp_3240 == tmp_3307 { keys[21] = tmp_3305; values[21] = tmp_3306; } let tmp_3308 = smem_keys[tmp_3241 * WPT + 22u]; let tmp_3309 = smem_vals[tmp_3241 * WPT + 22u]; let tmp_3310 = keys[22] < tmp_3308 || (keys[22] == tmp_3308 && values[22] < tmp_3309); if tmp_3240 == tmp_3310 { keys[22] = tmp_3308; values[22] = tmp_3309; } let tmp_3311 = smem_keys[tmp_3241 * WPT + 23u]; let tmp_3312 = smem_vals[tmp_3241 * WPT + 23u]; let tmp_3313 = keys[23] < tmp_3311 || (keys[23] == tmp_3311 && values[23] < tmp_3312); if tmp_3240 == tmp_3313 { keys[23] = tmp_3311; values[23] = tmp_3312; } let tmp_3314 = smem_keys[tmp_3241 * WPT + 24u]; let tmp_3315 = smem_vals[tmp_3241 * WPT + 24u]; let tmp_3316 = keys[24] < tmp_3314 || (keys[24] == tmp_3314 && values[24] < tmp_3315); if tmp_3240 == tmp_3316 { keys[24] = tmp_3314; values[24] = tmp_3315; } let tmp_3317 = smem_keys[tmp_3241 * WPT + 25u]; let tmp_3318 = smem_vals[tmp_3241 * WPT + 25u]; let tmp_3319 = keys[25] < tmp_3317 || (keys[25] == tmp_3317 && values[25] < tmp_3318); if tmp_3240 == tmp_3319 { keys[25] = tmp_3317; values[25] = tmp_3318; } let tmp_3320 = smem_keys[tmp_3241 * WPT + 26u]; let tmp_3321 = smem_vals[tmp_3241 * WPT + 26u]; let tmp_3322 = keys[26] < tmp_3320 || (keys[26] == tmp_3320 && values[26] < tmp_3321); if tmp_3240 == tmp_3322 { keys[26] = tmp_3320; values[26] = tmp_3321; } let tmp_3323 = smem_keys[tmp_3241 * WPT + 27u]; let tmp_3324 = smem_vals[tmp_3241 * WPT + 27u]; let tmp_3325 = keys[27] < tmp_3323 || (keys[27] == tmp_3323 && values[27] < tmp_3324); if tmp_3240 == tmp_3325 { keys[27] = tmp_3323; values[27] = tmp_3324; } let tmp_3326 = smem_keys[tmp_3241 * WPT + 28u]; let tmp_3327 = smem_vals[tmp_3241 * WPT + 28u]; let tmp_3328 = keys[28] < tmp_3326 || (keys[28] == tmp_3326 && values[28] < tmp_3327); if tmp_3240 == tmp_3328 { keys[28] = tmp_3326; values[28] = tmp_3327; } let tmp_3329 = smem_keys[tmp_3241 * WPT + 29u]; let tmp_3330 = smem_vals[tmp_3241 * WPT + 29u]; let tmp_3331 = keys[29] < tmp_3329 || (keys[29] == tmp_3329 && values[29] < tmp_3330); if tmp_3240 == tmp_3331 { keys[29] = tmp_3329; values[29] = tmp_3330; } let tmp_3332 = smem_keys[tmp_3241 * WPT + 30u]; let tmp_3333 = smem_vals[tmp_3241 * WPT + 30u]; let tmp_3334 = keys[30] < tmp_3332 || (keys[30] == tmp_3332 && values[30] < tmp_3333); if tmp_3240 == tmp_3334 { keys[30] = tmp_3332; values[30] = tmp_3333; } let tmp_3335 = smem_keys[tmp_3241 * WPT + 31u]; let tmp_3336 = smem_vals[tmp_3241 * WPT + 31u]; let tmp_3337 = keys[31] < tmp_3335 || (keys[31] == tmp_3335 && values[31] < tmp_3336); if tmp_3240 == tmp_3337 { keys[31] = tmp_3335; values[31] = tmp_3336; } workgroupBarrier(); }
    // exch_local(16,32) 
    // cmp_swap(0,16)
    if keys[0] > keys[16] || (keys[0] == keys[16] && values[0] > values[16]) {
    // swap(0,16) 
    { let tmp_3338 = keys[0]; keys[0] = keys[16]; keys[16] = tmp_3338;let tmp_3339 = values[0]; values[0] = values[16]; values[16] = tmp_3339; }
    }
    // cmp_swap(1,17)
    if keys[1] > keys[17] || (keys[1] == keys[17] && values[1] > values[17]) {
    // swap(1,17) 
    { let tmp_3340 = keys[1]; keys[1] = keys[17]; keys[17] = tmp_3340;let tmp_3341 = values[1]; values[1] = values[17]; values[17] = tmp_3341; }
    }
    // cmp_swap(2,18)
    if keys[2] > keys[18] || (keys[2] == keys[18] && values[2] > values[18]) {
    // swap(2,18) 
    { let tmp_3342 = keys[2]; keys[2] = keys[18]; keys[18] = tmp_3342;let tmp_3343 = values[2]; values[2] = values[18]; values[18] = tmp_3343; }
    }
    // cmp_swap(3,19)
    if keys[3] > keys[19] || (keys[3] == keys[19] && values[3] > values[19]) {
    // swap(3,19) 
    { let tmp_3344 = keys[3]; keys[3] = keys[19]; keys[19] = tmp_3344;let tmp_3345 = values[3]; values[3] = values[19]; values[19] = tmp_3345; }
    }
    // cmp_swap(4,20)
    if keys[4] > keys[20] || (keys[4] == keys[20] && values[4] > values[20]) {
    // swap(4,20) 
    { let tmp_3346 = keys[4]; keys[4] = keys[20]; keys[20] = tmp_3346;let tmp_3347 = values[4]; values[4] = values[20]; values[20] = tmp_3347; }
    }
    // cmp_swap(5,21)
    if keys[5] > keys[21] || (keys[5] == keys[21] && values[5] > values[21]) {
    // swap(5,21) 
    { let tmp_3348 = keys[5]; keys[5] = keys[21]; keys[21] = tmp_3348;let tmp_3349 = values[5]; values[5] = values[21]; values[21] = tmp_3349; }
    }
    // cmp_swap(6,22)
    if keys[6] > keys[22] || (keys[6] == keys[22] && values[6] > values[22]) {
    // swap(6,22) 
    { let tmp_3350 = keys[6]; keys[6] = keys[22]; keys[22] = tmp_3350;let tmp_3351 = values[6]; values[6] = values[22]; values[22] = tmp_3351; }
    }
    // cmp_swap(7,23)
    if keys[7] > keys[23] || (keys[7] == keys[23] && values[7] > values[23]) {
    // swap(7,23) 
    { let tmp_3352 = keys[7]; keys[7] = keys[23]; keys[23] = tmp_3352;let tmp_3353 = values[7]; values[7] = values[23]; values[23] = tmp_3353; }
    }
    // cmp_swap(8,24)
    if keys[8] > keys[24] || (keys[8] == keys[24] && values[8] > values[24]) {
    // swap(8,24) 
    { let tmp_3354 = keys[8]; keys[8] = keys[24]; keys[24] = tmp_3354;let tmp_3355 = values[8]; values[8] = values[24]; values[24] = tmp_3355; }
    }
    // cmp_swap(9,25)
    if keys[9] > keys[25] || (keys[9] == keys[25] && values[9] > values[25]) {
    // swap(9,25) 
    { let tmp_3356 = keys[9]; keys[9] = keys[25]; keys[25] = tmp_3356;let tmp_3357 = values[9]; values[9] = values[25]; values[25] = tmp_3357; }
    }
    // cmp_swap(10,26)
    if keys[10] > keys[26] || (keys[10] == keys[26] && values[10] > values[26]) {
    // swap(10,26) 
    { let tmp_3358 = keys[10]; keys[10] = keys[26]; keys[26] = tmp_3358;let tmp_3359 = values[10]; values[10] = values[26]; values[26] = tmp_3359; }
    }
    // cmp_swap(11,27)
    if keys[11] > keys[27] || (keys[11] == keys[27] && values[11] > values[27]) {
    // swap(11,27) 
    { let tmp_3360 = keys[11]; keys[11] = keys[27]; keys[27] = tmp_3360;let tmp_3361 = values[11]; values[11] = values[27]; values[27] = tmp_3361; }
    }
    // cmp_swap(12,28)
    if keys[12] > keys[28] || (keys[12] == keys[28] && values[12] > values[28]) {
    // swap(12,28) 
    { let tmp_3362 = keys[12]; keys[12] = keys[28]; keys[28] = tmp_3362;let tmp_3363 = values[12]; values[12] = values[28]; values[28] = tmp_3363; }
    }
    // cmp_swap(13,29)
    if keys[13] > keys[29] || (keys[13] == keys[29] && values[13] > values[29]) {
    // swap(13,29) 
    { let tmp_3364 = keys[13]; keys[13] = keys[29]; keys[29] = tmp_3364;let tmp_3365 = values[13]; values[13] = values[29]; values[29] = tmp_3365; }
    }
    // cmp_swap(14,30)
    if keys[14] > keys[30] || (keys[14] == keys[30] && values[14] > values[30]) {
    // swap(14,30) 
    { let tmp_3366 = keys[14]; keys[14] = keys[30]; keys[30] = tmp_3366;let tmp_3367 = values[14]; values[14] = values[30]; values[30] = tmp_3367; }
    }
    // cmp_swap(15,31)
    if keys[15] > keys[31] || (keys[15] == keys[31] && values[15] > values[31]) {
    // swap(15,31) 
    { let tmp_3368 = keys[15]; keys[15] = keys[31]; keys[31] = tmp_3368;let tmp_3369 = values[15]; values[15] = values[31]; values[31] = tmp_3369; }
    }
    // exch_local(8,32) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_3370 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_3370;let tmp_3371 = values[0]; values[0] = values[8]; values[8] = tmp_3371; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_3372 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_3372;let tmp_3373 = values[1]; values[1] = values[9]; values[9] = tmp_3373; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_3374 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_3374;let tmp_3375 = values[2]; values[2] = values[10]; values[10] = tmp_3375; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_3376 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_3376;let tmp_3377 = values[3]; values[3] = values[11]; values[11] = tmp_3377; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_3378 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_3378;let tmp_3379 = values[4]; values[4] = values[12]; values[12] = tmp_3379; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_3380 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_3380;let tmp_3381 = values[5]; values[5] = values[13]; values[13] = tmp_3381; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_3382 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_3382;let tmp_3383 = values[6]; values[6] = values[14]; values[14] = tmp_3383; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_3384 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_3384;let tmp_3385 = values[7]; values[7] = values[15]; values[15] = tmp_3385; }
    }
    // cmp_swap(16,24)
    if keys[16] > keys[24] || (keys[16] == keys[24] && values[16] > values[24]) {
    // swap(16,24) 
    { let tmp_3386 = keys[16]; keys[16] = keys[24]; keys[24] = tmp_3386;let tmp_3387 = values[16]; values[16] = values[24]; values[24] = tmp_3387; }
    }
    // cmp_swap(17,25)
    if keys[17] > keys[25] || (keys[17] == keys[25] && values[17] > values[25]) {
    // swap(17,25) 
    { let tmp_3388 = keys[17]; keys[17] = keys[25]; keys[25] = tmp_3388;let tmp_3389 = values[17]; values[17] = values[25]; values[25] = tmp_3389; }
    }
    // cmp_swap(18,26)
    if keys[18] > keys[26] || (keys[18] == keys[26] && values[18] > values[26]) {
    // swap(18,26) 
    { let tmp_3390 = keys[18]; keys[18] = keys[26]; keys[26] = tmp_3390;let tmp_3391 = values[18]; values[18] = values[26]; values[26] = tmp_3391; }
    }
    // cmp_swap(19,27)
    if keys[19] > keys[27] || (keys[19] == keys[27] && values[19] > values[27]) {
    // swap(19,27) 
    { let tmp_3392 = keys[19]; keys[19] = keys[27]; keys[27] = tmp_3392;let tmp_3393 = values[19]; values[19] = values[27]; values[27] = tmp_3393; }
    }
    // cmp_swap(20,28)
    if keys[20] > keys[28] || (keys[20] == keys[28] && values[20] > values[28]) {
    // swap(20,28) 
    { let tmp_3394 = keys[20]; keys[20] = keys[28]; keys[28] = tmp_3394;let tmp_3395 = values[20]; values[20] = values[28]; values[28] = tmp_3395; }
    }
    // cmp_swap(21,29)
    if keys[21] > keys[29] || (keys[21] == keys[29] && values[21] > values[29]) {
    // swap(21,29) 
    { let tmp_3396 = keys[21]; keys[21] = keys[29]; keys[29] = tmp_3396;let tmp_3397 = values[21]; values[21] = values[29]; values[29] = tmp_3397; }
    }
    // cmp_swap(22,30)
    if keys[22] > keys[30] || (keys[22] == keys[30] && values[22] > values[30]) {
    // swap(22,30) 
    { let tmp_3398 = keys[22]; keys[22] = keys[30]; keys[30] = tmp_3398;let tmp_3399 = values[22]; values[22] = values[30]; values[30] = tmp_3399; }
    }
    // cmp_swap(23,31)
    if keys[23] > keys[31] || (keys[23] == keys[31] && values[23] > values[31]) {
    // swap(23,31) 
    { let tmp_3400 = keys[23]; keys[23] = keys[31]; keys[31] = tmp_3400;let tmp_3401 = values[23]; values[23] = values[31]; values[31] = tmp_3401; }
    }
    // exch_local(4,32) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_3402 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_3402;let tmp_3403 = values[0]; values[0] = values[4]; values[4] = tmp_3403; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_3404 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_3404;let tmp_3405 = values[1]; values[1] = values[5]; values[5] = tmp_3405; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_3406 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_3406;let tmp_3407 = values[2]; values[2] = values[6]; values[6] = tmp_3407; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_3408 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_3408;let tmp_3409 = values[3]; values[3] = values[7]; values[7] = tmp_3409; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_3410 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_3410;let tmp_3411 = values[8]; values[8] = values[12]; values[12] = tmp_3411; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_3412 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_3412;let tmp_3413 = values[9]; values[9] = values[13]; values[13] = tmp_3413; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_3414 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_3414;let tmp_3415 = values[10]; values[10] = values[14]; values[14] = tmp_3415; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_3416 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_3416;let tmp_3417 = values[11]; values[11] = values[15]; values[15] = tmp_3417; }
    }
    // cmp_swap(16,20)
    if keys[16] > keys[20] || (keys[16] == keys[20] && values[16] > values[20]) {
    // swap(16,20) 
    { let tmp_3418 = keys[16]; keys[16] = keys[20]; keys[20] = tmp_3418;let tmp_3419 = values[16]; values[16] = values[20]; values[20] = tmp_3419; }
    }
    // cmp_swap(17,21)
    if keys[17] > keys[21] || (keys[17] == keys[21] && values[17] > values[21]) {
    // swap(17,21) 
    { let tmp_3420 = keys[17]; keys[17] = keys[21]; keys[21] = tmp_3420;let tmp_3421 = values[17]; values[17] = values[21]; values[21] = tmp_3421; }
    }
    // cmp_swap(18,22)
    if keys[18] > keys[22] || (keys[18] == keys[22] && values[18] > values[22]) {
    // swap(18,22) 
    { let tmp_3422 = keys[18]; keys[18] = keys[22]; keys[22] = tmp_3422;let tmp_3423 = values[18]; values[18] = values[22]; values[22] = tmp_3423; }
    }
    // cmp_swap(19,23)
    if keys[19] > keys[23] || (keys[19] == keys[23] && values[19] > values[23]) {
    // swap(19,23) 
    { let tmp_3424 = keys[19]; keys[19] = keys[23]; keys[23] = tmp_3424;let tmp_3425 = values[19]; values[19] = values[23]; values[23] = tmp_3425; }
    }
    // cmp_swap(24,28)
    if keys[24] > keys[28] || (keys[24] == keys[28] && values[24] > values[28]) {
    // swap(24,28) 
    { let tmp_3426 = keys[24]; keys[24] = keys[28]; keys[28] = tmp_3426;let tmp_3427 = values[24]; values[24] = values[28]; values[28] = tmp_3427; }
    }
    // cmp_swap(25,29)
    if keys[25] > keys[29] || (keys[25] == keys[29] && values[25] > values[29]) {
    // swap(25,29) 
    { let tmp_3428 = keys[25]; keys[25] = keys[29]; keys[29] = tmp_3428;let tmp_3429 = values[25]; values[25] = values[29]; values[29] = tmp_3429; }
    }
    // cmp_swap(26,30)
    if keys[26] > keys[30] || (keys[26] == keys[30] && values[26] > values[30]) {
    // swap(26,30) 
    { let tmp_3430 = keys[26]; keys[26] = keys[30]; keys[30] = tmp_3430;let tmp_3431 = values[26]; values[26] = values[30]; values[30] = tmp_3431; }
    }
    // cmp_swap(27,31)
    if keys[27] > keys[31] || (keys[27] == keys[31] && values[27] > values[31]) {
    // swap(27,31) 
    { let tmp_3432 = keys[27]; keys[27] = keys[31]; keys[31] = tmp_3432;let tmp_3433 = values[27]; values[27] = values[31]; values[31] = tmp_3433; }
    }
    // exch_local(2,32) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_3434 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_3434;let tmp_3435 = values[0]; values[0] = values[2]; values[2] = tmp_3435; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_3436 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_3436;let tmp_3437 = values[1]; values[1] = values[3]; values[3] = tmp_3437; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_3438 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_3438;let tmp_3439 = values[4]; values[4] = values[6]; values[6] = tmp_3439; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_3440 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_3440;let tmp_3441 = values[5]; values[5] = values[7]; values[7] = tmp_3441; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_3442 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_3442;let tmp_3443 = values[8]; values[8] = values[10]; values[10] = tmp_3443; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_3444 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_3444;let tmp_3445 = values[9]; values[9] = values[11]; values[11] = tmp_3445; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_3446 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_3446;let tmp_3447 = values[12]; values[12] = values[14]; values[14] = tmp_3447; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_3448 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_3448;let tmp_3449 = values[13]; values[13] = values[15]; values[15] = tmp_3449; }
    }
    // cmp_swap(16,18)
    if keys[16] > keys[18] || (keys[16] == keys[18] && values[16] > values[18]) {
    // swap(16,18) 
    { let tmp_3450 = keys[16]; keys[16] = keys[18]; keys[18] = tmp_3450;let tmp_3451 = values[16]; values[16] = values[18]; values[18] = tmp_3451; }
    }
    // cmp_swap(17,19)
    if keys[17] > keys[19] || (keys[17] == keys[19] && values[17] > values[19]) {
    // swap(17,19) 
    { let tmp_3452 = keys[17]; keys[17] = keys[19]; keys[19] = tmp_3452;let tmp_3453 = values[17]; values[17] = values[19]; values[19] = tmp_3453; }
    }
    // cmp_swap(20,22)
    if keys[20] > keys[22] || (keys[20] == keys[22] && values[20] > values[22]) {
    // swap(20,22) 
    { let tmp_3454 = keys[20]; keys[20] = keys[22]; keys[22] = tmp_3454;let tmp_3455 = values[20]; values[20] = values[22]; values[22] = tmp_3455; }
    }
    // cmp_swap(21,23)
    if keys[21] > keys[23] || (keys[21] == keys[23] && values[21] > values[23]) {
    // swap(21,23) 
    { let tmp_3456 = keys[21]; keys[21] = keys[23]; keys[23] = tmp_3456;let tmp_3457 = values[21]; values[21] = values[23]; values[23] = tmp_3457; }
    }
    // cmp_swap(24,26)
    if keys[24] > keys[26] || (keys[24] == keys[26] && values[24] > values[26]) {
    // swap(24,26) 
    { let tmp_3458 = keys[24]; keys[24] = keys[26]; keys[26] = tmp_3458;let tmp_3459 = values[24]; values[24] = values[26]; values[26] = tmp_3459; }
    }
    // cmp_swap(25,27)
    if keys[25] > keys[27] || (keys[25] == keys[27] && values[25] > values[27]) {
    // swap(25,27) 
    { let tmp_3460 = keys[25]; keys[25] = keys[27]; keys[27] = tmp_3460;let tmp_3461 = values[25]; values[25] = values[27]; values[27] = tmp_3461; }
    }
    // cmp_swap(28,30)
    if keys[28] > keys[30] || (keys[28] == keys[30] && values[28] > values[30]) {
    // swap(28,30) 
    { let tmp_3462 = keys[28]; keys[28] = keys[30]; keys[30] = tmp_3462;let tmp_3463 = values[28]; values[28] = values[30]; values[30] = tmp_3463; }
    }
    // cmp_swap(29,31)
    if keys[29] > keys[31] || (keys[29] == keys[31] && values[29] > values[31]) {
    // swap(29,31) 
    { let tmp_3464 = keys[29]; keys[29] = keys[31]; keys[31] = tmp_3464;let tmp_3465 = values[29]; values[29] = values[31]; values[31] = tmp_3465; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_3466 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_3466;let tmp_3467 = values[0]; values[0] = values[1]; values[1] = tmp_3467; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_3468 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_3468;let tmp_3469 = values[2]; values[2] = values[3]; values[3] = tmp_3469; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_3470 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_3470;let tmp_3471 = values[4]; values[4] = values[5]; values[5] = tmp_3471; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_3472 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_3472;let tmp_3473 = values[6]; values[6] = values[7]; values[7] = tmp_3473; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_3474 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_3474;let tmp_3475 = values[8]; values[8] = values[9]; values[9] = tmp_3475; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_3476 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_3476;let tmp_3477 = values[10]; values[10] = values[11]; values[11] = tmp_3477; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_3478 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_3478;let tmp_3479 = values[12]; values[12] = values[13]; values[13] = tmp_3479; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_3480 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_3480;let tmp_3481 = values[14]; values[14] = values[15]; values[15] = tmp_3481; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_3482 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_3482;let tmp_3483 = values[16]; values[16] = values[17]; values[17] = tmp_3483; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_3484 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_3484;let tmp_3485 = values[18]; values[18] = values[19]; values[19] = tmp_3485; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_3486 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_3486;let tmp_3487 = values[20]; values[20] = values[21]; values[21] = tmp_3487; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_3488 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_3488;let tmp_3489 = values[22]; values[22] = values[23]; values[23] = tmp_3489; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_3490 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_3490;let tmp_3491 = values[24]; values[24] = values[25]; values[25] = tmp_3491; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_3492 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_3492;let tmp_3493 = values[26]; values[26] = values[27]; values[27] = tmp_3493; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_3494 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_3494;let tmp_3495 = values[28]; values[28] = values[29]; values[29] = tmp_3495; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_3496 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_3496;let tmp_3497 = values[30]; values[30] = values[31]; values[31] = tmp_3497; }
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
