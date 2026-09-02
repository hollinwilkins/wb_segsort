
override WG: u32 = 1u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 32u;
const M: u32 = 1u;
const WPT: u32 = 32u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n32_m1_striped(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 5u;

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
