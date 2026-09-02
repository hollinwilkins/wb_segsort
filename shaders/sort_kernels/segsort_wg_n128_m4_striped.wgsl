
override WG: u32 = 4u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 128u;
const M: u32 = 4u;
const WPT: u32 = 32u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n128_m4_striped(
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
