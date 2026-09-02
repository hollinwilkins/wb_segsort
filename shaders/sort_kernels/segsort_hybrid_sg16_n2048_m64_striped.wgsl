
enable subgroups;

override WG: u32 = 64u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 2048u;
const M: u32 = 64u;
const WPT: u32 = 32u;
const R: u32 = 16u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybrid_sg16_n2048_m64_striped(
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
    {
    let tmp_480 = subgroupShuffleXor(keys[31], 1u);
    let tmp_481 = subgroupShuffleXor(values[31], 1u);
    let tmp_482 = subgroupShuffleXor(keys[30], 1u);
    let tmp_483 = subgroupShuffleXor(values[30], 1u);
    let tmp_484 = subgroupShuffleXor(keys[29], 1u);
    let tmp_485 = subgroupShuffleXor(values[29], 1u);
    let tmp_486 = subgroupShuffleXor(keys[28], 1u);
    let tmp_487 = subgroupShuffleXor(values[28], 1u);
    let tmp_488 = subgroupShuffleXor(keys[27], 1u);
    let tmp_489 = subgroupShuffleXor(values[27], 1u);
    let tmp_490 = subgroupShuffleXor(keys[26], 1u);
    let tmp_491 = subgroupShuffleXor(values[26], 1u);
    let tmp_492 = subgroupShuffleXor(keys[25], 1u);
    let tmp_493 = subgroupShuffleXor(values[25], 1u);
    let tmp_494 = subgroupShuffleXor(keys[24], 1u);
    let tmp_495 = subgroupShuffleXor(values[24], 1u);
    let tmp_496 = subgroupShuffleXor(keys[23], 1u);
    let tmp_497 = subgroupShuffleXor(values[23], 1u);
    let tmp_498 = subgroupShuffleXor(keys[22], 1u);
    let tmp_499 = subgroupShuffleXor(values[22], 1u);
    let tmp_500 = subgroupShuffleXor(keys[21], 1u);
    let tmp_501 = subgroupShuffleXor(values[21], 1u);
    let tmp_502 = subgroupShuffleXor(keys[20], 1u);
    let tmp_503 = subgroupShuffleXor(values[20], 1u);
    let tmp_504 = subgroupShuffleXor(keys[19], 1u);
    let tmp_505 = subgroupShuffleXor(values[19], 1u);
    let tmp_506 = subgroupShuffleXor(keys[18], 1u);
    let tmp_507 = subgroupShuffleXor(values[18], 1u);
    let tmp_508 = subgroupShuffleXor(keys[17], 1u);
    let tmp_509 = subgroupShuffleXor(values[17], 1u);
    let tmp_510 = subgroupShuffleXor(keys[16], 1u);
    let tmp_511 = subgroupShuffleXor(values[16], 1u);
    let tmp_512 = subgroupShuffleXor(keys[15], 1u);
    let tmp_513 = subgroupShuffleXor(values[15], 1u);
    let tmp_514 = subgroupShuffleXor(keys[14], 1u);
    let tmp_515 = subgroupShuffleXor(values[14], 1u);
    let tmp_516 = subgroupShuffleXor(keys[13], 1u);
    let tmp_517 = subgroupShuffleXor(values[13], 1u);
    let tmp_518 = subgroupShuffleXor(keys[12], 1u);
    let tmp_519 = subgroupShuffleXor(values[12], 1u);
    let tmp_520 = subgroupShuffleXor(keys[11], 1u);
    let tmp_521 = subgroupShuffleXor(values[11], 1u);
    let tmp_522 = subgroupShuffleXor(keys[10], 1u);
    let tmp_523 = subgroupShuffleXor(values[10], 1u);
    let tmp_524 = subgroupShuffleXor(keys[9], 1u);
    let tmp_525 = subgroupShuffleXor(values[9], 1u);
    let tmp_526 = subgroupShuffleXor(keys[8], 1u);
    let tmp_527 = subgroupShuffleXor(values[8], 1u);
    let tmp_528 = subgroupShuffleXor(keys[7], 1u);
    let tmp_529 = subgroupShuffleXor(values[7], 1u);
    let tmp_530 = subgroupShuffleXor(keys[6], 1u);
    let tmp_531 = subgroupShuffleXor(values[6], 1u);
    let tmp_532 = subgroupShuffleXor(keys[5], 1u);
    let tmp_533 = subgroupShuffleXor(values[5], 1u);
    let tmp_534 = subgroupShuffleXor(keys[4], 1u);
    let tmp_535 = subgroupShuffleXor(values[4], 1u);
    let tmp_536 = subgroupShuffleXor(keys[3], 1u);
    let tmp_537 = subgroupShuffleXor(values[3], 1u);
    let tmp_538 = subgroupShuffleXor(keys[2], 1u);
    let tmp_539 = subgroupShuffleXor(values[2], 1u);
    let tmp_540 = subgroupShuffleXor(keys[1], 1u);
    let tmp_541 = subgroupShuffleXor(values[1], 1u);
    let tmp_542 = subgroupShuffleXor(keys[0], 1u);
    let tmp_543 = subgroupShuffleXor(values[0], 1u);
    let tmp_544 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_545 = keys[0] < tmp_480 || (keys[0] == tmp_480 && values[0] < tmp_481);
    if tmp_544 == tmp_545 { keys[0] = tmp_480; values[0] = tmp_481; }
    let tmp_546 = keys[1] < tmp_482 || (keys[1] == tmp_482 && values[1] < tmp_483);
    if tmp_544 == tmp_546 { keys[1] = tmp_482; values[1] = tmp_483; }
    let tmp_547 = keys[2] < tmp_484 || (keys[2] == tmp_484 && values[2] < tmp_485);
    if tmp_544 == tmp_547 { keys[2] = tmp_484; values[2] = tmp_485; }
    let tmp_548 = keys[3] < tmp_486 || (keys[3] == tmp_486 && values[3] < tmp_487);
    if tmp_544 == tmp_548 { keys[3] = tmp_486; values[3] = tmp_487; }
    let tmp_549 = keys[4] < tmp_488 || (keys[4] == tmp_488 && values[4] < tmp_489);
    if tmp_544 == tmp_549 { keys[4] = tmp_488; values[4] = tmp_489; }
    let tmp_550 = keys[5] < tmp_490 || (keys[5] == tmp_490 && values[5] < tmp_491);
    if tmp_544 == tmp_550 { keys[5] = tmp_490; values[5] = tmp_491; }
    let tmp_551 = keys[6] < tmp_492 || (keys[6] == tmp_492 && values[6] < tmp_493);
    if tmp_544 == tmp_551 { keys[6] = tmp_492; values[6] = tmp_493; }
    let tmp_552 = keys[7] < tmp_494 || (keys[7] == tmp_494 && values[7] < tmp_495);
    if tmp_544 == tmp_552 { keys[7] = tmp_494; values[7] = tmp_495; }
    let tmp_553 = keys[8] < tmp_496 || (keys[8] == tmp_496 && values[8] < tmp_497);
    if tmp_544 == tmp_553 { keys[8] = tmp_496; values[8] = tmp_497; }
    let tmp_554 = keys[9] < tmp_498 || (keys[9] == tmp_498 && values[9] < tmp_499);
    if tmp_544 == tmp_554 { keys[9] = tmp_498; values[9] = tmp_499; }
    let tmp_555 = keys[10] < tmp_500 || (keys[10] == tmp_500 && values[10] < tmp_501);
    if tmp_544 == tmp_555 { keys[10] = tmp_500; values[10] = tmp_501; }
    let tmp_556 = keys[11] < tmp_502 || (keys[11] == tmp_502 && values[11] < tmp_503);
    if tmp_544 == tmp_556 { keys[11] = tmp_502; values[11] = tmp_503; }
    let tmp_557 = keys[12] < tmp_504 || (keys[12] == tmp_504 && values[12] < tmp_505);
    if tmp_544 == tmp_557 { keys[12] = tmp_504; values[12] = tmp_505; }
    let tmp_558 = keys[13] < tmp_506 || (keys[13] == tmp_506 && values[13] < tmp_507);
    if tmp_544 == tmp_558 { keys[13] = tmp_506; values[13] = tmp_507; }
    let tmp_559 = keys[14] < tmp_508 || (keys[14] == tmp_508 && values[14] < tmp_509);
    if tmp_544 == tmp_559 { keys[14] = tmp_508; values[14] = tmp_509; }
    let tmp_560 = keys[15] < tmp_510 || (keys[15] == tmp_510 && values[15] < tmp_511);
    if tmp_544 == tmp_560 { keys[15] = tmp_510; values[15] = tmp_511; }
    let tmp_561 = keys[16] < tmp_512 || (keys[16] == tmp_512 && values[16] < tmp_513);
    if tmp_544 == tmp_561 { keys[16] = tmp_512; values[16] = tmp_513; }
    let tmp_562 = keys[17] < tmp_514 || (keys[17] == tmp_514 && values[17] < tmp_515);
    if tmp_544 == tmp_562 { keys[17] = tmp_514; values[17] = tmp_515; }
    let tmp_563 = keys[18] < tmp_516 || (keys[18] == tmp_516 && values[18] < tmp_517);
    if tmp_544 == tmp_563 { keys[18] = tmp_516; values[18] = tmp_517; }
    let tmp_564 = keys[19] < tmp_518 || (keys[19] == tmp_518 && values[19] < tmp_519);
    if tmp_544 == tmp_564 { keys[19] = tmp_518; values[19] = tmp_519; }
    let tmp_565 = keys[20] < tmp_520 || (keys[20] == tmp_520 && values[20] < tmp_521);
    if tmp_544 == tmp_565 { keys[20] = tmp_520; values[20] = tmp_521; }
    let tmp_566 = keys[21] < tmp_522 || (keys[21] == tmp_522 && values[21] < tmp_523);
    if tmp_544 == tmp_566 { keys[21] = tmp_522; values[21] = tmp_523; }
    let tmp_567 = keys[22] < tmp_524 || (keys[22] == tmp_524 && values[22] < tmp_525);
    if tmp_544 == tmp_567 { keys[22] = tmp_524; values[22] = tmp_525; }
    let tmp_568 = keys[23] < tmp_526 || (keys[23] == tmp_526 && values[23] < tmp_527);
    if tmp_544 == tmp_568 { keys[23] = tmp_526; values[23] = tmp_527; }
    let tmp_569 = keys[24] < tmp_528 || (keys[24] == tmp_528 && values[24] < tmp_529);
    if tmp_544 == tmp_569 { keys[24] = tmp_528; values[24] = tmp_529; }
    let tmp_570 = keys[25] < tmp_530 || (keys[25] == tmp_530 && values[25] < tmp_531);
    if tmp_544 == tmp_570 { keys[25] = tmp_530; values[25] = tmp_531; }
    let tmp_571 = keys[26] < tmp_532 || (keys[26] == tmp_532 && values[26] < tmp_533);
    if tmp_544 == tmp_571 { keys[26] = tmp_532; values[26] = tmp_533; }
    let tmp_572 = keys[27] < tmp_534 || (keys[27] == tmp_534 && values[27] < tmp_535);
    if tmp_544 == tmp_572 { keys[27] = tmp_534; values[27] = tmp_535; }
    let tmp_573 = keys[28] < tmp_536 || (keys[28] == tmp_536 && values[28] < tmp_537);
    if tmp_544 == tmp_573 { keys[28] = tmp_536; values[28] = tmp_537; }
    let tmp_574 = keys[29] < tmp_538 || (keys[29] == tmp_538 && values[29] < tmp_539);
    if tmp_544 == tmp_574 { keys[29] = tmp_538; values[29] = tmp_539; }
    let tmp_575 = keys[30] < tmp_540 || (keys[30] == tmp_540 && values[30] < tmp_541);
    if tmp_544 == tmp_575 { keys[30] = tmp_540; values[30] = tmp_541; }
    let tmp_576 = keys[31] < tmp_542 || (keys[31] == tmp_542 && values[31] < tmp_543);
    if tmp_544 == tmp_576 { keys[31] = tmp_542; values[31] = tmp_543; }
    }
    // exch_local(16,32) 
    // cmp_swap(0,16)
    if keys[0] > keys[16] || (keys[0] == keys[16] && values[0] > values[16]) {
    // swap(0,16) 
    { let tmp_577 = keys[0]; keys[0] = keys[16]; keys[16] = tmp_577;let tmp_578 = values[0]; values[0] = values[16]; values[16] = tmp_578; }
    }
    // cmp_swap(1,17)
    if keys[1] > keys[17] || (keys[1] == keys[17] && values[1] > values[17]) {
    // swap(1,17) 
    { let tmp_579 = keys[1]; keys[1] = keys[17]; keys[17] = tmp_579;let tmp_580 = values[1]; values[1] = values[17]; values[17] = tmp_580; }
    }
    // cmp_swap(2,18)
    if keys[2] > keys[18] || (keys[2] == keys[18] && values[2] > values[18]) {
    // swap(2,18) 
    { let tmp_581 = keys[2]; keys[2] = keys[18]; keys[18] = tmp_581;let tmp_582 = values[2]; values[2] = values[18]; values[18] = tmp_582; }
    }
    // cmp_swap(3,19)
    if keys[3] > keys[19] || (keys[3] == keys[19] && values[3] > values[19]) {
    // swap(3,19) 
    { let tmp_583 = keys[3]; keys[3] = keys[19]; keys[19] = tmp_583;let tmp_584 = values[3]; values[3] = values[19]; values[19] = tmp_584; }
    }
    // cmp_swap(4,20)
    if keys[4] > keys[20] || (keys[4] == keys[20] && values[4] > values[20]) {
    // swap(4,20) 
    { let tmp_585 = keys[4]; keys[4] = keys[20]; keys[20] = tmp_585;let tmp_586 = values[4]; values[4] = values[20]; values[20] = tmp_586; }
    }
    // cmp_swap(5,21)
    if keys[5] > keys[21] || (keys[5] == keys[21] && values[5] > values[21]) {
    // swap(5,21) 
    { let tmp_587 = keys[5]; keys[5] = keys[21]; keys[21] = tmp_587;let tmp_588 = values[5]; values[5] = values[21]; values[21] = tmp_588; }
    }
    // cmp_swap(6,22)
    if keys[6] > keys[22] || (keys[6] == keys[22] && values[6] > values[22]) {
    // swap(6,22) 
    { let tmp_589 = keys[6]; keys[6] = keys[22]; keys[22] = tmp_589;let tmp_590 = values[6]; values[6] = values[22]; values[22] = tmp_590; }
    }
    // cmp_swap(7,23)
    if keys[7] > keys[23] || (keys[7] == keys[23] && values[7] > values[23]) {
    // swap(7,23) 
    { let tmp_591 = keys[7]; keys[7] = keys[23]; keys[23] = tmp_591;let tmp_592 = values[7]; values[7] = values[23]; values[23] = tmp_592; }
    }
    // cmp_swap(8,24)
    if keys[8] > keys[24] || (keys[8] == keys[24] && values[8] > values[24]) {
    // swap(8,24) 
    { let tmp_593 = keys[8]; keys[8] = keys[24]; keys[24] = tmp_593;let tmp_594 = values[8]; values[8] = values[24]; values[24] = tmp_594; }
    }
    // cmp_swap(9,25)
    if keys[9] > keys[25] || (keys[9] == keys[25] && values[9] > values[25]) {
    // swap(9,25) 
    { let tmp_595 = keys[9]; keys[9] = keys[25]; keys[25] = tmp_595;let tmp_596 = values[9]; values[9] = values[25]; values[25] = tmp_596; }
    }
    // cmp_swap(10,26)
    if keys[10] > keys[26] || (keys[10] == keys[26] && values[10] > values[26]) {
    // swap(10,26) 
    { let tmp_597 = keys[10]; keys[10] = keys[26]; keys[26] = tmp_597;let tmp_598 = values[10]; values[10] = values[26]; values[26] = tmp_598; }
    }
    // cmp_swap(11,27)
    if keys[11] > keys[27] || (keys[11] == keys[27] && values[11] > values[27]) {
    // swap(11,27) 
    { let tmp_599 = keys[11]; keys[11] = keys[27]; keys[27] = tmp_599;let tmp_600 = values[11]; values[11] = values[27]; values[27] = tmp_600; }
    }
    // cmp_swap(12,28)
    if keys[12] > keys[28] || (keys[12] == keys[28] && values[12] > values[28]) {
    // swap(12,28) 
    { let tmp_601 = keys[12]; keys[12] = keys[28]; keys[28] = tmp_601;let tmp_602 = values[12]; values[12] = values[28]; values[28] = tmp_602; }
    }
    // cmp_swap(13,29)
    if keys[13] > keys[29] || (keys[13] == keys[29] && values[13] > values[29]) {
    // swap(13,29) 
    { let tmp_603 = keys[13]; keys[13] = keys[29]; keys[29] = tmp_603;let tmp_604 = values[13]; values[13] = values[29]; values[29] = tmp_604; }
    }
    // cmp_swap(14,30)
    if keys[14] > keys[30] || (keys[14] == keys[30] && values[14] > values[30]) {
    // swap(14,30) 
    { let tmp_605 = keys[14]; keys[14] = keys[30]; keys[30] = tmp_605;let tmp_606 = values[14]; values[14] = values[30]; values[30] = tmp_606; }
    }
    // cmp_swap(15,31)
    if keys[15] > keys[31] || (keys[15] == keys[31] && values[15] > values[31]) {
    // swap(15,31) 
    { let tmp_607 = keys[15]; keys[15] = keys[31]; keys[31] = tmp_607;let tmp_608 = values[15]; values[15] = values[31]; values[31] = tmp_608; }
    }
    // exch_local(8,32) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_609 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_609;let tmp_610 = values[0]; values[0] = values[8]; values[8] = tmp_610; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_611 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_611;let tmp_612 = values[1]; values[1] = values[9]; values[9] = tmp_612; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_613 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_613;let tmp_614 = values[2]; values[2] = values[10]; values[10] = tmp_614; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_615 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_615;let tmp_616 = values[3]; values[3] = values[11]; values[11] = tmp_616; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_617 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_617;let tmp_618 = values[4]; values[4] = values[12]; values[12] = tmp_618; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_619 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_619;let tmp_620 = values[5]; values[5] = values[13]; values[13] = tmp_620; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_621 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_621;let tmp_622 = values[6]; values[6] = values[14]; values[14] = tmp_622; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_623 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_623;let tmp_624 = values[7]; values[7] = values[15]; values[15] = tmp_624; }
    }
    // cmp_swap(16,24)
    if keys[16] > keys[24] || (keys[16] == keys[24] && values[16] > values[24]) {
    // swap(16,24) 
    { let tmp_625 = keys[16]; keys[16] = keys[24]; keys[24] = tmp_625;let tmp_626 = values[16]; values[16] = values[24]; values[24] = tmp_626; }
    }
    // cmp_swap(17,25)
    if keys[17] > keys[25] || (keys[17] == keys[25] && values[17] > values[25]) {
    // swap(17,25) 
    { let tmp_627 = keys[17]; keys[17] = keys[25]; keys[25] = tmp_627;let tmp_628 = values[17]; values[17] = values[25]; values[25] = tmp_628; }
    }
    // cmp_swap(18,26)
    if keys[18] > keys[26] || (keys[18] == keys[26] && values[18] > values[26]) {
    // swap(18,26) 
    { let tmp_629 = keys[18]; keys[18] = keys[26]; keys[26] = tmp_629;let tmp_630 = values[18]; values[18] = values[26]; values[26] = tmp_630; }
    }
    // cmp_swap(19,27)
    if keys[19] > keys[27] || (keys[19] == keys[27] && values[19] > values[27]) {
    // swap(19,27) 
    { let tmp_631 = keys[19]; keys[19] = keys[27]; keys[27] = tmp_631;let tmp_632 = values[19]; values[19] = values[27]; values[27] = tmp_632; }
    }
    // cmp_swap(20,28)
    if keys[20] > keys[28] || (keys[20] == keys[28] && values[20] > values[28]) {
    // swap(20,28) 
    { let tmp_633 = keys[20]; keys[20] = keys[28]; keys[28] = tmp_633;let tmp_634 = values[20]; values[20] = values[28]; values[28] = tmp_634; }
    }
    // cmp_swap(21,29)
    if keys[21] > keys[29] || (keys[21] == keys[29] && values[21] > values[29]) {
    // swap(21,29) 
    { let tmp_635 = keys[21]; keys[21] = keys[29]; keys[29] = tmp_635;let tmp_636 = values[21]; values[21] = values[29]; values[29] = tmp_636; }
    }
    // cmp_swap(22,30)
    if keys[22] > keys[30] || (keys[22] == keys[30] && values[22] > values[30]) {
    // swap(22,30) 
    { let tmp_637 = keys[22]; keys[22] = keys[30]; keys[30] = tmp_637;let tmp_638 = values[22]; values[22] = values[30]; values[30] = tmp_638; }
    }
    // cmp_swap(23,31)
    if keys[23] > keys[31] || (keys[23] == keys[31] && values[23] > values[31]) {
    // swap(23,31) 
    { let tmp_639 = keys[23]; keys[23] = keys[31]; keys[31] = tmp_639;let tmp_640 = values[23]; values[23] = values[31]; values[31] = tmp_640; }
    }
    // exch_local(4,32) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_641 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_641;let tmp_642 = values[0]; values[0] = values[4]; values[4] = tmp_642; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_643 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_643;let tmp_644 = values[1]; values[1] = values[5]; values[5] = tmp_644; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_645 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_645;let tmp_646 = values[2]; values[2] = values[6]; values[6] = tmp_646; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_647 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_647;let tmp_648 = values[3]; values[3] = values[7]; values[7] = tmp_648; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_649 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_649;let tmp_650 = values[8]; values[8] = values[12]; values[12] = tmp_650; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_651 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_651;let tmp_652 = values[9]; values[9] = values[13]; values[13] = tmp_652; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_653 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_653;let tmp_654 = values[10]; values[10] = values[14]; values[14] = tmp_654; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_655 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_655;let tmp_656 = values[11]; values[11] = values[15]; values[15] = tmp_656; }
    }
    // cmp_swap(16,20)
    if keys[16] > keys[20] || (keys[16] == keys[20] && values[16] > values[20]) {
    // swap(16,20) 
    { let tmp_657 = keys[16]; keys[16] = keys[20]; keys[20] = tmp_657;let tmp_658 = values[16]; values[16] = values[20]; values[20] = tmp_658; }
    }
    // cmp_swap(17,21)
    if keys[17] > keys[21] || (keys[17] == keys[21] && values[17] > values[21]) {
    // swap(17,21) 
    { let tmp_659 = keys[17]; keys[17] = keys[21]; keys[21] = tmp_659;let tmp_660 = values[17]; values[17] = values[21]; values[21] = tmp_660; }
    }
    // cmp_swap(18,22)
    if keys[18] > keys[22] || (keys[18] == keys[22] && values[18] > values[22]) {
    // swap(18,22) 
    { let tmp_661 = keys[18]; keys[18] = keys[22]; keys[22] = tmp_661;let tmp_662 = values[18]; values[18] = values[22]; values[22] = tmp_662; }
    }
    // cmp_swap(19,23)
    if keys[19] > keys[23] || (keys[19] == keys[23] && values[19] > values[23]) {
    // swap(19,23) 
    { let tmp_663 = keys[19]; keys[19] = keys[23]; keys[23] = tmp_663;let tmp_664 = values[19]; values[19] = values[23]; values[23] = tmp_664; }
    }
    // cmp_swap(24,28)
    if keys[24] > keys[28] || (keys[24] == keys[28] && values[24] > values[28]) {
    // swap(24,28) 
    { let tmp_665 = keys[24]; keys[24] = keys[28]; keys[28] = tmp_665;let tmp_666 = values[24]; values[24] = values[28]; values[28] = tmp_666; }
    }
    // cmp_swap(25,29)
    if keys[25] > keys[29] || (keys[25] == keys[29] && values[25] > values[29]) {
    // swap(25,29) 
    { let tmp_667 = keys[25]; keys[25] = keys[29]; keys[29] = tmp_667;let tmp_668 = values[25]; values[25] = values[29]; values[29] = tmp_668; }
    }
    // cmp_swap(26,30)
    if keys[26] > keys[30] || (keys[26] == keys[30] && values[26] > values[30]) {
    // swap(26,30) 
    { let tmp_669 = keys[26]; keys[26] = keys[30]; keys[30] = tmp_669;let tmp_670 = values[26]; values[26] = values[30]; values[30] = tmp_670; }
    }
    // cmp_swap(27,31)
    if keys[27] > keys[31] || (keys[27] == keys[31] && values[27] > values[31]) {
    // swap(27,31) 
    { let tmp_671 = keys[27]; keys[27] = keys[31]; keys[31] = tmp_671;let tmp_672 = values[27]; values[27] = values[31]; values[31] = tmp_672; }
    }
    // exch_local(2,32) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_673 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_673;let tmp_674 = values[0]; values[0] = values[2]; values[2] = tmp_674; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_675 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_675;let tmp_676 = values[1]; values[1] = values[3]; values[3] = tmp_676; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_677 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_677;let tmp_678 = values[4]; values[4] = values[6]; values[6] = tmp_678; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_679 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_679;let tmp_680 = values[5]; values[5] = values[7]; values[7] = tmp_680; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_681 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_681;let tmp_682 = values[8]; values[8] = values[10]; values[10] = tmp_682; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_683 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_683;let tmp_684 = values[9]; values[9] = values[11]; values[11] = tmp_684; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_685 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_685;let tmp_686 = values[12]; values[12] = values[14]; values[14] = tmp_686; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_687 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_687;let tmp_688 = values[13]; values[13] = values[15]; values[15] = tmp_688; }
    }
    // cmp_swap(16,18)
    if keys[16] > keys[18] || (keys[16] == keys[18] && values[16] > values[18]) {
    // swap(16,18) 
    { let tmp_689 = keys[16]; keys[16] = keys[18]; keys[18] = tmp_689;let tmp_690 = values[16]; values[16] = values[18]; values[18] = tmp_690; }
    }
    // cmp_swap(17,19)
    if keys[17] > keys[19] || (keys[17] == keys[19] && values[17] > values[19]) {
    // swap(17,19) 
    { let tmp_691 = keys[17]; keys[17] = keys[19]; keys[19] = tmp_691;let tmp_692 = values[17]; values[17] = values[19]; values[19] = tmp_692; }
    }
    // cmp_swap(20,22)
    if keys[20] > keys[22] || (keys[20] == keys[22] && values[20] > values[22]) {
    // swap(20,22) 
    { let tmp_693 = keys[20]; keys[20] = keys[22]; keys[22] = tmp_693;let tmp_694 = values[20]; values[20] = values[22]; values[22] = tmp_694; }
    }
    // cmp_swap(21,23)
    if keys[21] > keys[23] || (keys[21] == keys[23] && values[21] > values[23]) {
    // swap(21,23) 
    { let tmp_695 = keys[21]; keys[21] = keys[23]; keys[23] = tmp_695;let tmp_696 = values[21]; values[21] = values[23]; values[23] = tmp_696; }
    }
    // cmp_swap(24,26)
    if keys[24] > keys[26] || (keys[24] == keys[26] && values[24] > values[26]) {
    // swap(24,26) 
    { let tmp_697 = keys[24]; keys[24] = keys[26]; keys[26] = tmp_697;let tmp_698 = values[24]; values[24] = values[26]; values[26] = tmp_698; }
    }
    // cmp_swap(25,27)
    if keys[25] > keys[27] || (keys[25] == keys[27] && values[25] > values[27]) {
    // swap(25,27) 
    { let tmp_699 = keys[25]; keys[25] = keys[27]; keys[27] = tmp_699;let tmp_700 = values[25]; values[25] = values[27]; values[27] = tmp_700; }
    }
    // cmp_swap(28,30)
    if keys[28] > keys[30] || (keys[28] == keys[30] && values[28] > values[30]) {
    // swap(28,30) 
    { let tmp_701 = keys[28]; keys[28] = keys[30]; keys[30] = tmp_701;let tmp_702 = values[28]; values[28] = values[30]; values[30] = tmp_702; }
    }
    // cmp_swap(29,31)
    if keys[29] > keys[31] || (keys[29] == keys[31] && values[29] > values[31]) {
    // swap(29,31) 
    { let tmp_703 = keys[29]; keys[29] = keys[31]; keys[31] = tmp_703;let tmp_704 = values[29]; values[29] = values[31]; values[31] = tmp_704; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_705 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_705;let tmp_706 = values[0]; values[0] = values[1]; values[1] = tmp_706; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_707 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_707;let tmp_708 = values[2]; values[2] = values[3]; values[3] = tmp_708; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_709 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_709;let tmp_710 = values[4]; values[4] = values[5]; values[5] = tmp_710; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_711 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_711;let tmp_712 = values[6]; values[6] = values[7]; values[7] = tmp_712; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_713 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_713;let tmp_714 = values[8]; values[8] = values[9]; values[9] = tmp_714; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_715 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_715;let tmp_716 = values[10]; values[10] = values[11]; values[11] = tmp_716; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_717 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_717;let tmp_718 = values[12]; values[12] = values[13]; values[13] = tmp_718; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_719 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_719;let tmp_720 = values[14]; values[14] = values[15]; values[15] = tmp_720; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_721 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_721;let tmp_722 = values[16]; values[16] = values[17]; values[17] = tmp_722; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_723 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_723;let tmp_724 = values[18]; values[18] = values[19]; values[19] = tmp_724; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_725 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_725;let tmp_726 = values[20]; values[20] = values[21]; values[21] = tmp_726; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_727 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_727;let tmp_728 = values[22]; values[22] = values[23]; values[23] = tmp_728; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_729 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_729;let tmp_730 = values[24]; values[24] = values[25]; values[25] = tmp_730; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_731 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_731;let tmp_732 = values[26]; values[26] = values[27]; values[27] = tmp_732; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_733 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_733;let tmp_734 = values[28]; values[28] = values[29]; values[29] = tmp_734; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_735 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_735;let tmp_736 = values[30]; values[30] = values[31]; values[31] = tmp_736; }
    }
    // exch_intxn(tmask:3,swbit:1,wpt:32)
    {
    let tmp_737 = subgroupShuffleXor(keys[31], 3u);
    let tmp_738 = subgroupShuffleXor(values[31], 3u);
    let tmp_739 = subgroupShuffleXor(keys[30], 3u);
    let tmp_740 = subgroupShuffleXor(values[30], 3u);
    let tmp_741 = subgroupShuffleXor(keys[29], 3u);
    let tmp_742 = subgroupShuffleXor(values[29], 3u);
    let tmp_743 = subgroupShuffleXor(keys[28], 3u);
    let tmp_744 = subgroupShuffleXor(values[28], 3u);
    let tmp_745 = subgroupShuffleXor(keys[27], 3u);
    let tmp_746 = subgroupShuffleXor(values[27], 3u);
    let tmp_747 = subgroupShuffleXor(keys[26], 3u);
    let tmp_748 = subgroupShuffleXor(values[26], 3u);
    let tmp_749 = subgroupShuffleXor(keys[25], 3u);
    let tmp_750 = subgroupShuffleXor(values[25], 3u);
    let tmp_751 = subgroupShuffleXor(keys[24], 3u);
    let tmp_752 = subgroupShuffleXor(values[24], 3u);
    let tmp_753 = subgroupShuffleXor(keys[23], 3u);
    let tmp_754 = subgroupShuffleXor(values[23], 3u);
    let tmp_755 = subgroupShuffleXor(keys[22], 3u);
    let tmp_756 = subgroupShuffleXor(values[22], 3u);
    let tmp_757 = subgroupShuffleXor(keys[21], 3u);
    let tmp_758 = subgroupShuffleXor(values[21], 3u);
    let tmp_759 = subgroupShuffleXor(keys[20], 3u);
    let tmp_760 = subgroupShuffleXor(values[20], 3u);
    let tmp_761 = subgroupShuffleXor(keys[19], 3u);
    let tmp_762 = subgroupShuffleXor(values[19], 3u);
    let tmp_763 = subgroupShuffleXor(keys[18], 3u);
    let tmp_764 = subgroupShuffleXor(values[18], 3u);
    let tmp_765 = subgroupShuffleXor(keys[17], 3u);
    let tmp_766 = subgroupShuffleXor(values[17], 3u);
    let tmp_767 = subgroupShuffleXor(keys[16], 3u);
    let tmp_768 = subgroupShuffleXor(values[16], 3u);
    let tmp_769 = subgroupShuffleXor(keys[15], 3u);
    let tmp_770 = subgroupShuffleXor(values[15], 3u);
    let tmp_771 = subgroupShuffleXor(keys[14], 3u);
    let tmp_772 = subgroupShuffleXor(values[14], 3u);
    let tmp_773 = subgroupShuffleXor(keys[13], 3u);
    let tmp_774 = subgroupShuffleXor(values[13], 3u);
    let tmp_775 = subgroupShuffleXor(keys[12], 3u);
    let tmp_776 = subgroupShuffleXor(values[12], 3u);
    let tmp_777 = subgroupShuffleXor(keys[11], 3u);
    let tmp_778 = subgroupShuffleXor(values[11], 3u);
    let tmp_779 = subgroupShuffleXor(keys[10], 3u);
    let tmp_780 = subgroupShuffleXor(values[10], 3u);
    let tmp_781 = subgroupShuffleXor(keys[9], 3u);
    let tmp_782 = subgroupShuffleXor(values[9], 3u);
    let tmp_783 = subgroupShuffleXor(keys[8], 3u);
    let tmp_784 = subgroupShuffleXor(values[8], 3u);
    let tmp_785 = subgroupShuffleXor(keys[7], 3u);
    let tmp_786 = subgroupShuffleXor(values[7], 3u);
    let tmp_787 = subgroupShuffleXor(keys[6], 3u);
    let tmp_788 = subgroupShuffleXor(values[6], 3u);
    let tmp_789 = subgroupShuffleXor(keys[5], 3u);
    let tmp_790 = subgroupShuffleXor(values[5], 3u);
    let tmp_791 = subgroupShuffleXor(keys[4], 3u);
    let tmp_792 = subgroupShuffleXor(values[4], 3u);
    let tmp_793 = subgroupShuffleXor(keys[3], 3u);
    let tmp_794 = subgroupShuffleXor(values[3], 3u);
    let tmp_795 = subgroupShuffleXor(keys[2], 3u);
    let tmp_796 = subgroupShuffleXor(values[2], 3u);
    let tmp_797 = subgroupShuffleXor(keys[1], 3u);
    let tmp_798 = subgroupShuffleXor(values[1], 3u);
    let tmp_799 = subgroupShuffleXor(keys[0], 3u);
    let tmp_800 = subgroupShuffleXor(values[0], 3u);
    let tmp_801 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_802 = keys[0] < tmp_737 || (keys[0] == tmp_737 && values[0] < tmp_738);
    if tmp_801 == tmp_802 { keys[0] = tmp_737; values[0] = tmp_738; }
    let tmp_803 = keys[1] < tmp_739 || (keys[1] == tmp_739 && values[1] < tmp_740);
    if tmp_801 == tmp_803 { keys[1] = tmp_739; values[1] = tmp_740; }
    let tmp_804 = keys[2] < tmp_741 || (keys[2] == tmp_741 && values[2] < tmp_742);
    if tmp_801 == tmp_804 { keys[2] = tmp_741; values[2] = tmp_742; }
    let tmp_805 = keys[3] < tmp_743 || (keys[3] == tmp_743 && values[3] < tmp_744);
    if tmp_801 == tmp_805 { keys[3] = tmp_743; values[3] = tmp_744; }
    let tmp_806 = keys[4] < tmp_745 || (keys[4] == tmp_745 && values[4] < tmp_746);
    if tmp_801 == tmp_806 { keys[4] = tmp_745; values[4] = tmp_746; }
    let tmp_807 = keys[5] < tmp_747 || (keys[5] == tmp_747 && values[5] < tmp_748);
    if tmp_801 == tmp_807 { keys[5] = tmp_747; values[5] = tmp_748; }
    let tmp_808 = keys[6] < tmp_749 || (keys[6] == tmp_749 && values[6] < tmp_750);
    if tmp_801 == tmp_808 { keys[6] = tmp_749; values[6] = tmp_750; }
    let tmp_809 = keys[7] < tmp_751 || (keys[7] == tmp_751 && values[7] < tmp_752);
    if tmp_801 == tmp_809 { keys[7] = tmp_751; values[7] = tmp_752; }
    let tmp_810 = keys[8] < tmp_753 || (keys[8] == tmp_753 && values[8] < tmp_754);
    if tmp_801 == tmp_810 { keys[8] = tmp_753; values[8] = tmp_754; }
    let tmp_811 = keys[9] < tmp_755 || (keys[9] == tmp_755 && values[9] < tmp_756);
    if tmp_801 == tmp_811 { keys[9] = tmp_755; values[9] = tmp_756; }
    let tmp_812 = keys[10] < tmp_757 || (keys[10] == tmp_757 && values[10] < tmp_758);
    if tmp_801 == tmp_812 { keys[10] = tmp_757; values[10] = tmp_758; }
    let tmp_813 = keys[11] < tmp_759 || (keys[11] == tmp_759 && values[11] < tmp_760);
    if tmp_801 == tmp_813 { keys[11] = tmp_759; values[11] = tmp_760; }
    let tmp_814 = keys[12] < tmp_761 || (keys[12] == tmp_761 && values[12] < tmp_762);
    if tmp_801 == tmp_814 { keys[12] = tmp_761; values[12] = tmp_762; }
    let tmp_815 = keys[13] < tmp_763 || (keys[13] == tmp_763 && values[13] < tmp_764);
    if tmp_801 == tmp_815 { keys[13] = tmp_763; values[13] = tmp_764; }
    let tmp_816 = keys[14] < tmp_765 || (keys[14] == tmp_765 && values[14] < tmp_766);
    if tmp_801 == tmp_816 { keys[14] = tmp_765; values[14] = tmp_766; }
    let tmp_817 = keys[15] < tmp_767 || (keys[15] == tmp_767 && values[15] < tmp_768);
    if tmp_801 == tmp_817 { keys[15] = tmp_767; values[15] = tmp_768; }
    let tmp_818 = keys[16] < tmp_769 || (keys[16] == tmp_769 && values[16] < tmp_770);
    if tmp_801 == tmp_818 { keys[16] = tmp_769; values[16] = tmp_770; }
    let tmp_819 = keys[17] < tmp_771 || (keys[17] == tmp_771 && values[17] < tmp_772);
    if tmp_801 == tmp_819 { keys[17] = tmp_771; values[17] = tmp_772; }
    let tmp_820 = keys[18] < tmp_773 || (keys[18] == tmp_773 && values[18] < tmp_774);
    if tmp_801 == tmp_820 { keys[18] = tmp_773; values[18] = tmp_774; }
    let tmp_821 = keys[19] < tmp_775 || (keys[19] == tmp_775 && values[19] < tmp_776);
    if tmp_801 == tmp_821 { keys[19] = tmp_775; values[19] = tmp_776; }
    let tmp_822 = keys[20] < tmp_777 || (keys[20] == tmp_777 && values[20] < tmp_778);
    if tmp_801 == tmp_822 { keys[20] = tmp_777; values[20] = tmp_778; }
    let tmp_823 = keys[21] < tmp_779 || (keys[21] == tmp_779 && values[21] < tmp_780);
    if tmp_801 == tmp_823 { keys[21] = tmp_779; values[21] = tmp_780; }
    let tmp_824 = keys[22] < tmp_781 || (keys[22] == tmp_781 && values[22] < tmp_782);
    if tmp_801 == tmp_824 { keys[22] = tmp_781; values[22] = tmp_782; }
    let tmp_825 = keys[23] < tmp_783 || (keys[23] == tmp_783 && values[23] < tmp_784);
    if tmp_801 == tmp_825 { keys[23] = tmp_783; values[23] = tmp_784; }
    let tmp_826 = keys[24] < tmp_785 || (keys[24] == tmp_785 && values[24] < tmp_786);
    if tmp_801 == tmp_826 { keys[24] = tmp_785; values[24] = tmp_786; }
    let tmp_827 = keys[25] < tmp_787 || (keys[25] == tmp_787 && values[25] < tmp_788);
    if tmp_801 == tmp_827 { keys[25] = tmp_787; values[25] = tmp_788; }
    let tmp_828 = keys[26] < tmp_789 || (keys[26] == tmp_789 && values[26] < tmp_790);
    if tmp_801 == tmp_828 { keys[26] = tmp_789; values[26] = tmp_790; }
    let tmp_829 = keys[27] < tmp_791 || (keys[27] == tmp_791 && values[27] < tmp_792);
    if tmp_801 == tmp_829 { keys[27] = tmp_791; values[27] = tmp_792; }
    let tmp_830 = keys[28] < tmp_793 || (keys[28] == tmp_793 && values[28] < tmp_794);
    if tmp_801 == tmp_830 { keys[28] = tmp_793; values[28] = tmp_794; }
    let tmp_831 = keys[29] < tmp_795 || (keys[29] == tmp_795 && values[29] < tmp_796);
    if tmp_801 == tmp_831 { keys[29] = tmp_795; values[29] = tmp_796; }
    let tmp_832 = keys[30] < tmp_797 || (keys[30] == tmp_797 && values[30] < tmp_798);
    if tmp_801 == tmp_832 { keys[30] = tmp_797; values[30] = tmp_798; }
    let tmp_833 = keys[31] < tmp_799 || (keys[31] == tmp_799 && values[31] < tmp_800);
    if tmp_801 == tmp_833 { keys[31] = tmp_799; values[31] = tmp_800; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:32) 
    {
    let tmp_834 = subgroupShuffleXor(keys[0], 1u);
    let tmp_835 = subgroupShuffleXor(values[0], 1u);
    let tmp_836 = subgroupShuffleXor(keys[1], 1u);
    let tmp_837 = subgroupShuffleXor(values[1], 1u);
    let tmp_838 = subgroupShuffleXor(keys[2], 1u);
    let tmp_839 = subgroupShuffleXor(values[2], 1u);
    let tmp_840 = subgroupShuffleXor(keys[3], 1u);
    let tmp_841 = subgroupShuffleXor(values[3], 1u);
    let tmp_842 = subgroupShuffleXor(keys[4], 1u);
    let tmp_843 = subgroupShuffleXor(values[4], 1u);
    let tmp_844 = subgroupShuffleXor(keys[5], 1u);
    let tmp_845 = subgroupShuffleXor(values[5], 1u);
    let tmp_846 = subgroupShuffleXor(keys[6], 1u);
    let tmp_847 = subgroupShuffleXor(values[6], 1u);
    let tmp_848 = subgroupShuffleXor(keys[7], 1u);
    let tmp_849 = subgroupShuffleXor(values[7], 1u);
    let tmp_850 = subgroupShuffleXor(keys[8], 1u);
    let tmp_851 = subgroupShuffleXor(values[8], 1u);
    let tmp_852 = subgroupShuffleXor(keys[9], 1u);
    let tmp_853 = subgroupShuffleXor(values[9], 1u);
    let tmp_854 = subgroupShuffleXor(keys[10], 1u);
    let tmp_855 = subgroupShuffleXor(values[10], 1u);
    let tmp_856 = subgroupShuffleXor(keys[11], 1u);
    let tmp_857 = subgroupShuffleXor(values[11], 1u);
    let tmp_858 = subgroupShuffleXor(keys[12], 1u);
    let tmp_859 = subgroupShuffleXor(values[12], 1u);
    let tmp_860 = subgroupShuffleXor(keys[13], 1u);
    let tmp_861 = subgroupShuffleXor(values[13], 1u);
    let tmp_862 = subgroupShuffleXor(keys[14], 1u);
    let tmp_863 = subgroupShuffleXor(values[14], 1u);
    let tmp_864 = subgroupShuffleXor(keys[15], 1u);
    let tmp_865 = subgroupShuffleXor(values[15], 1u);
    let tmp_866 = subgroupShuffleXor(keys[16], 1u);
    let tmp_867 = subgroupShuffleXor(values[16], 1u);
    let tmp_868 = subgroupShuffleXor(keys[17], 1u);
    let tmp_869 = subgroupShuffleXor(values[17], 1u);
    let tmp_870 = subgroupShuffleXor(keys[18], 1u);
    let tmp_871 = subgroupShuffleXor(values[18], 1u);
    let tmp_872 = subgroupShuffleXor(keys[19], 1u);
    let tmp_873 = subgroupShuffleXor(values[19], 1u);
    let tmp_874 = subgroupShuffleXor(keys[20], 1u);
    let tmp_875 = subgroupShuffleXor(values[20], 1u);
    let tmp_876 = subgroupShuffleXor(keys[21], 1u);
    let tmp_877 = subgroupShuffleXor(values[21], 1u);
    let tmp_878 = subgroupShuffleXor(keys[22], 1u);
    let tmp_879 = subgroupShuffleXor(values[22], 1u);
    let tmp_880 = subgroupShuffleXor(keys[23], 1u);
    let tmp_881 = subgroupShuffleXor(values[23], 1u);
    let tmp_882 = subgroupShuffleXor(keys[24], 1u);
    let tmp_883 = subgroupShuffleXor(values[24], 1u);
    let tmp_884 = subgroupShuffleXor(keys[25], 1u);
    let tmp_885 = subgroupShuffleXor(values[25], 1u);
    let tmp_886 = subgroupShuffleXor(keys[26], 1u);
    let tmp_887 = subgroupShuffleXor(values[26], 1u);
    let tmp_888 = subgroupShuffleXor(keys[27], 1u);
    let tmp_889 = subgroupShuffleXor(values[27], 1u);
    let tmp_890 = subgroupShuffleXor(keys[28], 1u);
    let tmp_891 = subgroupShuffleXor(values[28], 1u);
    let tmp_892 = subgroupShuffleXor(keys[29], 1u);
    let tmp_893 = subgroupShuffleXor(values[29], 1u);
    let tmp_894 = subgroupShuffleXor(keys[30], 1u);
    let tmp_895 = subgroupShuffleXor(values[30], 1u);
    let tmp_896 = subgroupShuffleXor(keys[31], 1u);
    let tmp_897 = subgroupShuffleXor(values[31], 1u);
    let tmp_898 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_899 = keys[0] < tmp_834 || (keys[0] == tmp_834 && values[0] < tmp_835);
    if tmp_898 == tmp_899 { keys[0] = tmp_834; values[0] = tmp_835; }
    let tmp_900 = keys[1] < tmp_836 || (keys[1] == tmp_836 && values[1] < tmp_837);
    if tmp_898 == tmp_900 { keys[1] = tmp_836; values[1] = tmp_837; }
    let tmp_901 = keys[2] < tmp_838 || (keys[2] == tmp_838 && values[2] < tmp_839);
    if tmp_898 == tmp_901 { keys[2] = tmp_838; values[2] = tmp_839; }
    let tmp_902 = keys[3] < tmp_840 || (keys[3] == tmp_840 && values[3] < tmp_841);
    if tmp_898 == tmp_902 { keys[3] = tmp_840; values[3] = tmp_841; }
    let tmp_903 = keys[4] < tmp_842 || (keys[4] == tmp_842 && values[4] < tmp_843);
    if tmp_898 == tmp_903 { keys[4] = tmp_842; values[4] = tmp_843; }
    let tmp_904 = keys[5] < tmp_844 || (keys[5] == tmp_844 && values[5] < tmp_845);
    if tmp_898 == tmp_904 { keys[5] = tmp_844; values[5] = tmp_845; }
    let tmp_905 = keys[6] < tmp_846 || (keys[6] == tmp_846 && values[6] < tmp_847);
    if tmp_898 == tmp_905 { keys[6] = tmp_846; values[6] = tmp_847; }
    let tmp_906 = keys[7] < tmp_848 || (keys[7] == tmp_848 && values[7] < tmp_849);
    if tmp_898 == tmp_906 { keys[7] = tmp_848; values[7] = tmp_849; }
    let tmp_907 = keys[8] < tmp_850 || (keys[8] == tmp_850 && values[8] < tmp_851);
    if tmp_898 == tmp_907 { keys[8] = tmp_850; values[8] = tmp_851; }
    let tmp_908 = keys[9] < tmp_852 || (keys[9] == tmp_852 && values[9] < tmp_853);
    if tmp_898 == tmp_908 { keys[9] = tmp_852; values[9] = tmp_853; }
    let tmp_909 = keys[10] < tmp_854 || (keys[10] == tmp_854 && values[10] < tmp_855);
    if tmp_898 == tmp_909 { keys[10] = tmp_854; values[10] = tmp_855; }
    let tmp_910 = keys[11] < tmp_856 || (keys[11] == tmp_856 && values[11] < tmp_857);
    if tmp_898 == tmp_910 { keys[11] = tmp_856; values[11] = tmp_857; }
    let tmp_911 = keys[12] < tmp_858 || (keys[12] == tmp_858 && values[12] < tmp_859);
    if tmp_898 == tmp_911 { keys[12] = tmp_858; values[12] = tmp_859; }
    let tmp_912 = keys[13] < tmp_860 || (keys[13] == tmp_860 && values[13] < tmp_861);
    if tmp_898 == tmp_912 { keys[13] = tmp_860; values[13] = tmp_861; }
    let tmp_913 = keys[14] < tmp_862 || (keys[14] == tmp_862 && values[14] < tmp_863);
    if tmp_898 == tmp_913 { keys[14] = tmp_862; values[14] = tmp_863; }
    let tmp_914 = keys[15] < tmp_864 || (keys[15] == tmp_864 && values[15] < tmp_865);
    if tmp_898 == tmp_914 { keys[15] = tmp_864; values[15] = tmp_865; }
    let tmp_915 = keys[16] < tmp_866 || (keys[16] == tmp_866 && values[16] < tmp_867);
    if tmp_898 == tmp_915 { keys[16] = tmp_866; values[16] = tmp_867; }
    let tmp_916 = keys[17] < tmp_868 || (keys[17] == tmp_868 && values[17] < tmp_869);
    if tmp_898 == tmp_916 { keys[17] = tmp_868; values[17] = tmp_869; }
    let tmp_917 = keys[18] < tmp_870 || (keys[18] == tmp_870 && values[18] < tmp_871);
    if tmp_898 == tmp_917 { keys[18] = tmp_870; values[18] = tmp_871; }
    let tmp_918 = keys[19] < tmp_872 || (keys[19] == tmp_872 && values[19] < tmp_873);
    if tmp_898 == tmp_918 { keys[19] = tmp_872; values[19] = tmp_873; }
    let tmp_919 = keys[20] < tmp_874 || (keys[20] == tmp_874 && values[20] < tmp_875);
    if tmp_898 == tmp_919 { keys[20] = tmp_874; values[20] = tmp_875; }
    let tmp_920 = keys[21] < tmp_876 || (keys[21] == tmp_876 && values[21] < tmp_877);
    if tmp_898 == tmp_920 { keys[21] = tmp_876; values[21] = tmp_877; }
    let tmp_921 = keys[22] < tmp_878 || (keys[22] == tmp_878 && values[22] < tmp_879);
    if tmp_898 == tmp_921 { keys[22] = tmp_878; values[22] = tmp_879; }
    let tmp_922 = keys[23] < tmp_880 || (keys[23] == tmp_880 && values[23] < tmp_881);
    if tmp_898 == tmp_922 { keys[23] = tmp_880; values[23] = tmp_881; }
    let tmp_923 = keys[24] < tmp_882 || (keys[24] == tmp_882 && values[24] < tmp_883);
    if tmp_898 == tmp_923 { keys[24] = tmp_882; values[24] = tmp_883; }
    let tmp_924 = keys[25] < tmp_884 || (keys[25] == tmp_884 && values[25] < tmp_885);
    if tmp_898 == tmp_924 { keys[25] = tmp_884; values[25] = tmp_885; }
    let tmp_925 = keys[26] < tmp_886 || (keys[26] == tmp_886 && values[26] < tmp_887);
    if tmp_898 == tmp_925 { keys[26] = tmp_886; values[26] = tmp_887; }
    let tmp_926 = keys[27] < tmp_888 || (keys[27] == tmp_888 && values[27] < tmp_889);
    if tmp_898 == tmp_926 { keys[27] = tmp_888; values[27] = tmp_889; }
    let tmp_927 = keys[28] < tmp_890 || (keys[28] == tmp_890 && values[28] < tmp_891);
    if tmp_898 == tmp_927 { keys[28] = tmp_890; values[28] = tmp_891; }
    let tmp_928 = keys[29] < tmp_892 || (keys[29] == tmp_892 && values[29] < tmp_893);
    if tmp_898 == tmp_928 { keys[29] = tmp_892; values[29] = tmp_893; }
    let tmp_929 = keys[30] < tmp_894 || (keys[30] == tmp_894 && values[30] < tmp_895);
    if tmp_898 == tmp_929 { keys[30] = tmp_894; values[30] = tmp_895; }
    let tmp_930 = keys[31] < tmp_896 || (keys[31] == tmp_896 && values[31] < tmp_897);
    if tmp_898 == tmp_930 { keys[31] = tmp_896; values[31] = tmp_897; }
    }
    // exch_local(16,32) 
    // cmp_swap(0,16)
    if keys[0] > keys[16] || (keys[0] == keys[16] && values[0] > values[16]) {
    // swap(0,16) 
    { let tmp_931 = keys[0]; keys[0] = keys[16]; keys[16] = tmp_931;let tmp_932 = values[0]; values[0] = values[16]; values[16] = tmp_932; }
    }
    // cmp_swap(1,17)
    if keys[1] > keys[17] || (keys[1] == keys[17] && values[1] > values[17]) {
    // swap(1,17) 
    { let tmp_933 = keys[1]; keys[1] = keys[17]; keys[17] = tmp_933;let tmp_934 = values[1]; values[1] = values[17]; values[17] = tmp_934; }
    }
    // cmp_swap(2,18)
    if keys[2] > keys[18] || (keys[2] == keys[18] && values[2] > values[18]) {
    // swap(2,18) 
    { let tmp_935 = keys[2]; keys[2] = keys[18]; keys[18] = tmp_935;let tmp_936 = values[2]; values[2] = values[18]; values[18] = tmp_936; }
    }
    // cmp_swap(3,19)
    if keys[3] > keys[19] || (keys[3] == keys[19] && values[3] > values[19]) {
    // swap(3,19) 
    { let tmp_937 = keys[3]; keys[3] = keys[19]; keys[19] = tmp_937;let tmp_938 = values[3]; values[3] = values[19]; values[19] = tmp_938; }
    }
    // cmp_swap(4,20)
    if keys[4] > keys[20] || (keys[4] == keys[20] && values[4] > values[20]) {
    // swap(4,20) 
    { let tmp_939 = keys[4]; keys[4] = keys[20]; keys[20] = tmp_939;let tmp_940 = values[4]; values[4] = values[20]; values[20] = tmp_940; }
    }
    // cmp_swap(5,21)
    if keys[5] > keys[21] || (keys[5] == keys[21] && values[5] > values[21]) {
    // swap(5,21) 
    { let tmp_941 = keys[5]; keys[5] = keys[21]; keys[21] = tmp_941;let tmp_942 = values[5]; values[5] = values[21]; values[21] = tmp_942; }
    }
    // cmp_swap(6,22)
    if keys[6] > keys[22] || (keys[6] == keys[22] && values[6] > values[22]) {
    // swap(6,22) 
    { let tmp_943 = keys[6]; keys[6] = keys[22]; keys[22] = tmp_943;let tmp_944 = values[6]; values[6] = values[22]; values[22] = tmp_944; }
    }
    // cmp_swap(7,23)
    if keys[7] > keys[23] || (keys[7] == keys[23] && values[7] > values[23]) {
    // swap(7,23) 
    { let tmp_945 = keys[7]; keys[7] = keys[23]; keys[23] = tmp_945;let tmp_946 = values[7]; values[7] = values[23]; values[23] = tmp_946; }
    }
    // cmp_swap(8,24)
    if keys[8] > keys[24] || (keys[8] == keys[24] && values[8] > values[24]) {
    // swap(8,24) 
    { let tmp_947 = keys[8]; keys[8] = keys[24]; keys[24] = tmp_947;let tmp_948 = values[8]; values[8] = values[24]; values[24] = tmp_948; }
    }
    // cmp_swap(9,25)
    if keys[9] > keys[25] || (keys[9] == keys[25] && values[9] > values[25]) {
    // swap(9,25) 
    { let tmp_949 = keys[9]; keys[9] = keys[25]; keys[25] = tmp_949;let tmp_950 = values[9]; values[9] = values[25]; values[25] = tmp_950; }
    }
    // cmp_swap(10,26)
    if keys[10] > keys[26] || (keys[10] == keys[26] && values[10] > values[26]) {
    // swap(10,26) 
    { let tmp_951 = keys[10]; keys[10] = keys[26]; keys[26] = tmp_951;let tmp_952 = values[10]; values[10] = values[26]; values[26] = tmp_952; }
    }
    // cmp_swap(11,27)
    if keys[11] > keys[27] || (keys[11] == keys[27] && values[11] > values[27]) {
    // swap(11,27) 
    { let tmp_953 = keys[11]; keys[11] = keys[27]; keys[27] = tmp_953;let tmp_954 = values[11]; values[11] = values[27]; values[27] = tmp_954; }
    }
    // cmp_swap(12,28)
    if keys[12] > keys[28] || (keys[12] == keys[28] && values[12] > values[28]) {
    // swap(12,28) 
    { let tmp_955 = keys[12]; keys[12] = keys[28]; keys[28] = tmp_955;let tmp_956 = values[12]; values[12] = values[28]; values[28] = tmp_956; }
    }
    // cmp_swap(13,29)
    if keys[13] > keys[29] || (keys[13] == keys[29] && values[13] > values[29]) {
    // swap(13,29) 
    { let tmp_957 = keys[13]; keys[13] = keys[29]; keys[29] = tmp_957;let tmp_958 = values[13]; values[13] = values[29]; values[29] = tmp_958; }
    }
    // cmp_swap(14,30)
    if keys[14] > keys[30] || (keys[14] == keys[30] && values[14] > values[30]) {
    // swap(14,30) 
    { let tmp_959 = keys[14]; keys[14] = keys[30]; keys[30] = tmp_959;let tmp_960 = values[14]; values[14] = values[30]; values[30] = tmp_960; }
    }
    // cmp_swap(15,31)
    if keys[15] > keys[31] || (keys[15] == keys[31] && values[15] > values[31]) {
    // swap(15,31) 
    { let tmp_961 = keys[15]; keys[15] = keys[31]; keys[31] = tmp_961;let tmp_962 = values[15]; values[15] = values[31]; values[31] = tmp_962; }
    }
    // exch_local(8,32) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_963 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_963;let tmp_964 = values[0]; values[0] = values[8]; values[8] = tmp_964; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_965 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_965;let tmp_966 = values[1]; values[1] = values[9]; values[9] = tmp_966; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_967 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_967;let tmp_968 = values[2]; values[2] = values[10]; values[10] = tmp_968; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_969 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_969;let tmp_970 = values[3]; values[3] = values[11]; values[11] = tmp_970; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_971 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_971;let tmp_972 = values[4]; values[4] = values[12]; values[12] = tmp_972; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_973 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_973;let tmp_974 = values[5]; values[5] = values[13]; values[13] = tmp_974; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_975 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_975;let tmp_976 = values[6]; values[6] = values[14]; values[14] = tmp_976; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_977 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_977;let tmp_978 = values[7]; values[7] = values[15]; values[15] = tmp_978; }
    }
    // cmp_swap(16,24)
    if keys[16] > keys[24] || (keys[16] == keys[24] && values[16] > values[24]) {
    // swap(16,24) 
    { let tmp_979 = keys[16]; keys[16] = keys[24]; keys[24] = tmp_979;let tmp_980 = values[16]; values[16] = values[24]; values[24] = tmp_980; }
    }
    // cmp_swap(17,25)
    if keys[17] > keys[25] || (keys[17] == keys[25] && values[17] > values[25]) {
    // swap(17,25) 
    { let tmp_981 = keys[17]; keys[17] = keys[25]; keys[25] = tmp_981;let tmp_982 = values[17]; values[17] = values[25]; values[25] = tmp_982; }
    }
    // cmp_swap(18,26)
    if keys[18] > keys[26] || (keys[18] == keys[26] && values[18] > values[26]) {
    // swap(18,26) 
    { let tmp_983 = keys[18]; keys[18] = keys[26]; keys[26] = tmp_983;let tmp_984 = values[18]; values[18] = values[26]; values[26] = tmp_984; }
    }
    // cmp_swap(19,27)
    if keys[19] > keys[27] || (keys[19] == keys[27] && values[19] > values[27]) {
    // swap(19,27) 
    { let tmp_985 = keys[19]; keys[19] = keys[27]; keys[27] = tmp_985;let tmp_986 = values[19]; values[19] = values[27]; values[27] = tmp_986; }
    }
    // cmp_swap(20,28)
    if keys[20] > keys[28] || (keys[20] == keys[28] && values[20] > values[28]) {
    // swap(20,28) 
    { let tmp_987 = keys[20]; keys[20] = keys[28]; keys[28] = tmp_987;let tmp_988 = values[20]; values[20] = values[28]; values[28] = tmp_988; }
    }
    // cmp_swap(21,29)
    if keys[21] > keys[29] || (keys[21] == keys[29] && values[21] > values[29]) {
    // swap(21,29) 
    { let tmp_989 = keys[21]; keys[21] = keys[29]; keys[29] = tmp_989;let tmp_990 = values[21]; values[21] = values[29]; values[29] = tmp_990; }
    }
    // cmp_swap(22,30)
    if keys[22] > keys[30] || (keys[22] == keys[30] && values[22] > values[30]) {
    // swap(22,30) 
    { let tmp_991 = keys[22]; keys[22] = keys[30]; keys[30] = tmp_991;let tmp_992 = values[22]; values[22] = values[30]; values[30] = tmp_992; }
    }
    // cmp_swap(23,31)
    if keys[23] > keys[31] || (keys[23] == keys[31] && values[23] > values[31]) {
    // swap(23,31) 
    { let tmp_993 = keys[23]; keys[23] = keys[31]; keys[31] = tmp_993;let tmp_994 = values[23]; values[23] = values[31]; values[31] = tmp_994; }
    }
    // exch_local(4,32) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_995 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_995;let tmp_996 = values[0]; values[0] = values[4]; values[4] = tmp_996; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_997 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_997;let tmp_998 = values[1]; values[1] = values[5]; values[5] = tmp_998; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_999 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_999;let tmp_1000 = values[2]; values[2] = values[6]; values[6] = tmp_1000; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_1001 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_1001;let tmp_1002 = values[3]; values[3] = values[7]; values[7] = tmp_1002; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_1003 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_1003;let tmp_1004 = values[8]; values[8] = values[12]; values[12] = tmp_1004; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_1005 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_1005;let tmp_1006 = values[9]; values[9] = values[13]; values[13] = tmp_1006; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_1007 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_1007;let tmp_1008 = values[10]; values[10] = values[14]; values[14] = tmp_1008; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_1009 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_1009;let tmp_1010 = values[11]; values[11] = values[15]; values[15] = tmp_1010; }
    }
    // cmp_swap(16,20)
    if keys[16] > keys[20] || (keys[16] == keys[20] && values[16] > values[20]) {
    // swap(16,20) 
    { let tmp_1011 = keys[16]; keys[16] = keys[20]; keys[20] = tmp_1011;let tmp_1012 = values[16]; values[16] = values[20]; values[20] = tmp_1012; }
    }
    // cmp_swap(17,21)
    if keys[17] > keys[21] || (keys[17] == keys[21] && values[17] > values[21]) {
    // swap(17,21) 
    { let tmp_1013 = keys[17]; keys[17] = keys[21]; keys[21] = tmp_1013;let tmp_1014 = values[17]; values[17] = values[21]; values[21] = tmp_1014; }
    }
    // cmp_swap(18,22)
    if keys[18] > keys[22] || (keys[18] == keys[22] && values[18] > values[22]) {
    // swap(18,22) 
    { let tmp_1015 = keys[18]; keys[18] = keys[22]; keys[22] = tmp_1015;let tmp_1016 = values[18]; values[18] = values[22]; values[22] = tmp_1016; }
    }
    // cmp_swap(19,23)
    if keys[19] > keys[23] || (keys[19] == keys[23] && values[19] > values[23]) {
    // swap(19,23) 
    { let tmp_1017 = keys[19]; keys[19] = keys[23]; keys[23] = tmp_1017;let tmp_1018 = values[19]; values[19] = values[23]; values[23] = tmp_1018; }
    }
    // cmp_swap(24,28)
    if keys[24] > keys[28] || (keys[24] == keys[28] && values[24] > values[28]) {
    // swap(24,28) 
    { let tmp_1019 = keys[24]; keys[24] = keys[28]; keys[28] = tmp_1019;let tmp_1020 = values[24]; values[24] = values[28]; values[28] = tmp_1020; }
    }
    // cmp_swap(25,29)
    if keys[25] > keys[29] || (keys[25] == keys[29] && values[25] > values[29]) {
    // swap(25,29) 
    { let tmp_1021 = keys[25]; keys[25] = keys[29]; keys[29] = tmp_1021;let tmp_1022 = values[25]; values[25] = values[29]; values[29] = tmp_1022; }
    }
    // cmp_swap(26,30)
    if keys[26] > keys[30] || (keys[26] == keys[30] && values[26] > values[30]) {
    // swap(26,30) 
    { let tmp_1023 = keys[26]; keys[26] = keys[30]; keys[30] = tmp_1023;let tmp_1024 = values[26]; values[26] = values[30]; values[30] = tmp_1024; }
    }
    // cmp_swap(27,31)
    if keys[27] > keys[31] || (keys[27] == keys[31] && values[27] > values[31]) {
    // swap(27,31) 
    { let tmp_1025 = keys[27]; keys[27] = keys[31]; keys[31] = tmp_1025;let tmp_1026 = values[27]; values[27] = values[31]; values[31] = tmp_1026; }
    }
    // exch_local(2,32) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_1027 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_1027;let tmp_1028 = values[0]; values[0] = values[2]; values[2] = tmp_1028; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_1029 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_1029;let tmp_1030 = values[1]; values[1] = values[3]; values[3] = tmp_1030; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_1031 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_1031;let tmp_1032 = values[4]; values[4] = values[6]; values[6] = tmp_1032; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_1033 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_1033;let tmp_1034 = values[5]; values[5] = values[7]; values[7] = tmp_1034; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_1035 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_1035;let tmp_1036 = values[8]; values[8] = values[10]; values[10] = tmp_1036; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_1037 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_1037;let tmp_1038 = values[9]; values[9] = values[11]; values[11] = tmp_1038; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_1039 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_1039;let tmp_1040 = values[12]; values[12] = values[14]; values[14] = tmp_1040; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_1041 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_1041;let tmp_1042 = values[13]; values[13] = values[15]; values[15] = tmp_1042; }
    }
    // cmp_swap(16,18)
    if keys[16] > keys[18] || (keys[16] == keys[18] && values[16] > values[18]) {
    // swap(16,18) 
    { let tmp_1043 = keys[16]; keys[16] = keys[18]; keys[18] = tmp_1043;let tmp_1044 = values[16]; values[16] = values[18]; values[18] = tmp_1044; }
    }
    // cmp_swap(17,19)
    if keys[17] > keys[19] || (keys[17] == keys[19] && values[17] > values[19]) {
    // swap(17,19) 
    { let tmp_1045 = keys[17]; keys[17] = keys[19]; keys[19] = tmp_1045;let tmp_1046 = values[17]; values[17] = values[19]; values[19] = tmp_1046; }
    }
    // cmp_swap(20,22)
    if keys[20] > keys[22] || (keys[20] == keys[22] && values[20] > values[22]) {
    // swap(20,22) 
    { let tmp_1047 = keys[20]; keys[20] = keys[22]; keys[22] = tmp_1047;let tmp_1048 = values[20]; values[20] = values[22]; values[22] = tmp_1048; }
    }
    // cmp_swap(21,23)
    if keys[21] > keys[23] || (keys[21] == keys[23] && values[21] > values[23]) {
    // swap(21,23) 
    { let tmp_1049 = keys[21]; keys[21] = keys[23]; keys[23] = tmp_1049;let tmp_1050 = values[21]; values[21] = values[23]; values[23] = tmp_1050; }
    }
    // cmp_swap(24,26)
    if keys[24] > keys[26] || (keys[24] == keys[26] && values[24] > values[26]) {
    // swap(24,26) 
    { let tmp_1051 = keys[24]; keys[24] = keys[26]; keys[26] = tmp_1051;let tmp_1052 = values[24]; values[24] = values[26]; values[26] = tmp_1052; }
    }
    // cmp_swap(25,27)
    if keys[25] > keys[27] || (keys[25] == keys[27] && values[25] > values[27]) {
    // swap(25,27) 
    { let tmp_1053 = keys[25]; keys[25] = keys[27]; keys[27] = tmp_1053;let tmp_1054 = values[25]; values[25] = values[27]; values[27] = tmp_1054; }
    }
    // cmp_swap(28,30)
    if keys[28] > keys[30] || (keys[28] == keys[30] && values[28] > values[30]) {
    // swap(28,30) 
    { let tmp_1055 = keys[28]; keys[28] = keys[30]; keys[30] = tmp_1055;let tmp_1056 = values[28]; values[28] = values[30]; values[30] = tmp_1056; }
    }
    // cmp_swap(29,31)
    if keys[29] > keys[31] || (keys[29] == keys[31] && values[29] > values[31]) {
    // swap(29,31) 
    { let tmp_1057 = keys[29]; keys[29] = keys[31]; keys[31] = tmp_1057;let tmp_1058 = values[29]; values[29] = values[31]; values[31] = tmp_1058; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_1059 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_1059;let tmp_1060 = values[0]; values[0] = values[1]; values[1] = tmp_1060; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_1061 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_1061;let tmp_1062 = values[2]; values[2] = values[3]; values[3] = tmp_1062; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_1063 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_1063;let tmp_1064 = values[4]; values[4] = values[5]; values[5] = tmp_1064; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_1065 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_1065;let tmp_1066 = values[6]; values[6] = values[7]; values[7] = tmp_1066; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_1067 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_1067;let tmp_1068 = values[8]; values[8] = values[9]; values[9] = tmp_1068; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_1069 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_1069;let tmp_1070 = values[10]; values[10] = values[11]; values[11] = tmp_1070; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_1071 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_1071;let tmp_1072 = values[12]; values[12] = values[13]; values[13] = tmp_1072; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_1073 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_1073;let tmp_1074 = values[14]; values[14] = values[15]; values[15] = tmp_1074; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_1075 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_1075;let tmp_1076 = values[16]; values[16] = values[17]; values[17] = tmp_1076; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_1077 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_1077;let tmp_1078 = values[18]; values[18] = values[19]; values[19] = tmp_1078; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_1079 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_1079;let tmp_1080 = values[20]; values[20] = values[21]; values[21] = tmp_1080; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_1081 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_1081;let tmp_1082 = values[22]; values[22] = values[23]; values[23] = tmp_1082; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_1083 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_1083;let tmp_1084 = values[24]; values[24] = values[25]; values[25] = tmp_1084; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_1085 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_1085;let tmp_1086 = values[26]; values[26] = values[27]; values[27] = tmp_1086; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_1087 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_1087;let tmp_1088 = values[28]; values[28] = values[29]; values[29] = tmp_1088; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_1089 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_1089;let tmp_1090 = values[30]; values[30] = values[31]; values[31] = tmp_1090; }
    }
    // exch_intxn(tmask:7,swbit:2,wpt:32)
    {
    let tmp_1091 = subgroupShuffleXor(keys[31], 7u);
    let tmp_1092 = subgroupShuffleXor(values[31], 7u);
    let tmp_1093 = subgroupShuffleXor(keys[30], 7u);
    let tmp_1094 = subgroupShuffleXor(values[30], 7u);
    let tmp_1095 = subgroupShuffleXor(keys[29], 7u);
    let tmp_1096 = subgroupShuffleXor(values[29], 7u);
    let tmp_1097 = subgroupShuffleXor(keys[28], 7u);
    let tmp_1098 = subgroupShuffleXor(values[28], 7u);
    let tmp_1099 = subgroupShuffleXor(keys[27], 7u);
    let tmp_1100 = subgroupShuffleXor(values[27], 7u);
    let tmp_1101 = subgroupShuffleXor(keys[26], 7u);
    let tmp_1102 = subgroupShuffleXor(values[26], 7u);
    let tmp_1103 = subgroupShuffleXor(keys[25], 7u);
    let tmp_1104 = subgroupShuffleXor(values[25], 7u);
    let tmp_1105 = subgroupShuffleXor(keys[24], 7u);
    let tmp_1106 = subgroupShuffleXor(values[24], 7u);
    let tmp_1107 = subgroupShuffleXor(keys[23], 7u);
    let tmp_1108 = subgroupShuffleXor(values[23], 7u);
    let tmp_1109 = subgroupShuffleXor(keys[22], 7u);
    let tmp_1110 = subgroupShuffleXor(values[22], 7u);
    let tmp_1111 = subgroupShuffleXor(keys[21], 7u);
    let tmp_1112 = subgroupShuffleXor(values[21], 7u);
    let tmp_1113 = subgroupShuffleXor(keys[20], 7u);
    let tmp_1114 = subgroupShuffleXor(values[20], 7u);
    let tmp_1115 = subgroupShuffleXor(keys[19], 7u);
    let tmp_1116 = subgroupShuffleXor(values[19], 7u);
    let tmp_1117 = subgroupShuffleXor(keys[18], 7u);
    let tmp_1118 = subgroupShuffleXor(values[18], 7u);
    let tmp_1119 = subgroupShuffleXor(keys[17], 7u);
    let tmp_1120 = subgroupShuffleXor(values[17], 7u);
    let tmp_1121 = subgroupShuffleXor(keys[16], 7u);
    let tmp_1122 = subgroupShuffleXor(values[16], 7u);
    let tmp_1123 = subgroupShuffleXor(keys[15], 7u);
    let tmp_1124 = subgroupShuffleXor(values[15], 7u);
    let tmp_1125 = subgroupShuffleXor(keys[14], 7u);
    let tmp_1126 = subgroupShuffleXor(values[14], 7u);
    let tmp_1127 = subgroupShuffleXor(keys[13], 7u);
    let tmp_1128 = subgroupShuffleXor(values[13], 7u);
    let tmp_1129 = subgroupShuffleXor(keys[12], 7u);
    let tmp_1130 = subgroupShuffleXor(values[12], 7u);
    let tmp_1131 = subgroupShuffleXor(keys[11], 7u);
    let tmp_1132 = subgroupShuffleXor(values[11], 7u);
    let tmp_1133 = subgroupShuffleXor(keys[10], 7u);
    let tmp_1134 = subgroupShuffleXor(values[10], 7u);
    let tmp_1135 = subgroupShuffleXor(keys[9], 7u);
    let tmp_1136 = subgroupShuffleXor(values[9], 7u);
    let tmp_1137 = subgroupShuffleXor(keys[8], 7u);
    let tmp_1138 = subgroupShuffleXor(values[8], 7u);
    let tmp_1139 = subgroupShuffleXor(keys[7], 7u);
    let tmp_1140 = subgroupShuffleXor(values[7], 7u);
    let tmp_1141 = subgroupShuffleXor(keys[6], 7u);
    let tmp_1142 = subgroupShuffleXor(values[6], 7u);
    let tmp_1143 = subgroupShuffleXor(keys[5], 7u);
    let tmp_1144 = subgroupShuffleXor(values[5], 7u);
    let tmp_1145 = subgroupShuffleXor(keys[4], 7u);
    let tmp_1146 = subgroupShuffleXor(values[4], 7u);
    let tmp_1147 = subgroupShuffleXor(keys[3], 7u);
    let tmp_1148 = subgroupShuffleXor(values[3], 7u);
    let tmp_1149 = subgroupShuffleXor(keys[2], 7u);
    let tmp_1150 = subgroupShuffleXor(values[2], 7u);
    let tmp_1151 = subgroupShuffleXor(keys[1], 7u);
    let tmp_1152 = subgroupShuffleXor(values[1], 7u);
    let tmp_1153 = subgroupShuffleXor(keys[0], 7u);
    let tmp_1154 = subgroupShuffleXor(values[0], 7u);
    let tmp_1155 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_1156 = keys[0] < tmp_1091 || (keys[0] == tmp_1091 && values[0] < tmp_1092);
    if tmp_1155 == tmp_1156 { keys[0] = tmp_1091; values[0] = tmp_1092; }
    let tmp_1157 = keys[1] < tmp_1093 || (keys[1] == tmp_1093 && values[1] < tmp_1094);
    if tmp_1155 == tmp_1157 { keys[1] = tmp_1093; values[1] = tmp_1094; }
    let tmp_1158 = keys[2] < tmp_1095 || (keys[2] == tmp_1095 && values[2] < tmp_1096);
    if tmp_1155 == tmp_1158 { keys[2] = tmp_1095; values[2] = tmp_1096; }
    let tmp_1159 = keys[3] < tmp_1097 || (keys[3] == tmp_1097 && values[3] < tmp_1098);
    if tmp_1155 == tmp_1159 { keys[3] = tmp_1097; values[3] = tmp_1098; }
    let tmp_1160 = keys[4] < tmp_1099 || (keys[4] == tmp_1099 && values[4] < tmp_1100);
    if tmp_1155 == tmp_1160 { keys[4] = tmp_1099; values[4] = tmp_1100; }
    let tmp_1161 = keys[5] < tmp_1101 || (keys[5] == tmp_1101 && values[5] < tmp_1102);
    if tmp_1155 == tmp_1161 { keys[5] = tmp_1101; values[5] = tmp_1102; }
    let tmp_1162 = keys[6] < tmp_1103 || (keys[6] == tmp_1103 && values[6] < tmp_1104);
    if tmp_1155 == tmp_1162 { keys[6] = tmp_1103; values[6] = tmp_1104; }
    let tmp_1163 = keys[7] < tmp_1105 || (keys[7] == tmp_1105 && values[7] < tmp_1106);
    if tmp_1155 == tmp_1163 { keys[7] = tmp_1105; values[7] = tmp_1106; }
    let tmp_1164 = keys[8] < tmp_1107 || (keys[8] == tmp_1107 && values[8] < tmp_1108);
    if tmp_1155 == tmp_1164 { keys[8] = tmp_1107; values[8] = tmp_1108; }
    let tmp_1165 = keys[9] < tmp_1109 || (keys[9] == tmp_1109 && values[9] < tmp_1110);
    if tmp_1155 == tmp_1165 { keys[9] = tmp_1109; values[9] = tmp_1110; }
    let tmp_1166 = keys[10] < tmp_1111 || (keys[10] == tmp_1111 && values[10] < tmp_1112);
    if tmp_1155 == tmp_1166 { keys[10] = tmp_1111; values[10] = tmp_1112; }
    let tmp_1167 = keys[11] < tmp_1113 || (keys[11] == tmp_1113 && values[11] < tmp_1114);
    if tmp_1155 == tmp_1167 { keys[11] = tmp_1113; values[11] = tmp_1114; }
    let tmp_1168 = keys[12] < tmp_1115 || (keys[12] == tmp_1115 && values[12] < tmp_1116);
    if tmp_1155 == tmp_1168 { keys[12] = tmp_1115; values[12] = tmp_1116; }
    let tmp_1169 = keys[13] < tmp_1117 || (keys[13] == tmp_1117 && values[13] < tmp_1118);
    if tmp_1155 == tmp_1169 { keys[13] = tmp_1117; values[13] = tmp_1118; }
    let tmp_1170 = keys[14] < tmp_1119 || (keys[14] == tmp_1119 && values[14] < tmp_1120);
    if tmp_1155 == tmp_1170 { keys[14] = tmp_1119; values[14] = tmp_1120; }
    let tmp_1171 = keys[15] < tmp_1121 || (keys[15] == tmp_1121 && values[15] < tmp_1122);
    if tmp_1155 == tmp_1171 { keys[15] = tmp_1121; values[15] = tmp_1122; }
    let tmp_1172 = keys[16] < tmp_1123 || (keys[16] == tmp_1123 && values[16] < tmp_1124);
    if tmp_1155 == tmp_1172 { keys[16] = tmp_1123; values[16] = tmp_1124; }
    let tmp_1173 = keys[17] < tmp_1125 || (keys[17] == tmp_1125 && values[17] < tmp_1126);
    if tmp_1155 == tmp_1173 { keys[17] = tmp_1125; values[17] = tmp_1126; }
    let tmp_1174 = keys[18] < tmp_1127 || (keys[18] == tmp_1127 && values[18] < tmp_1128);
    if tmp_1155 == tmp_1174 { keys[18] = tmp_1127; values[18] = tmp_1128; }
    let tmp_1175 = keys[19] < tmp_1129 || (keys[19] == tmp_1129 && values[19] < tmp_1130);
    if tmp_1155 == tmp_1175 { keys[19] = tmp_1129; values[19] = tmp_1130; }
    let tmp_1176 = keys[20] < tmp_1131 || (keys[20] == tmp_1131 && values[20] < tmp_1132);
    if tmp_1155 == tmp_1176 { keys[20] = tmp_1131; values[20] = tmp_1132; }
    let tmp_1177 = keys[21] < tmp_1133 || (keys[21] == tmp_1133 && values[21] < tmp_1134);
    if tmp_1155 == tmp_1177 { keys[21] = tmp_1133; values[21] = tmp_1134; }
    let tmp_1178 = keys[22] < tmp_1135 || (keys[22] == tmp_1135 && values[22] < tmp_1136);
    if tmp_1155 == tmp_1178 { keys[22] = tmp_1135; values[22] = tmp_1136; }
    let tmp_1179 = keys[23] < tmp_1137 || (keys[23] == tmp_1137 && values[23] < tmp_1138);
    if tmp_1155 == tmp_1179 { keys[23] = tmp_1137; values[23] = tmp_1138; }
    let tmp_1180 = keys[24] < tmp_1139 || (keys[24] == tmp_1139 && values[24] < tmp_1140);
    if tmp_1155 == tmp_1180 { keys[24] = tmp_1139; values[24] = tmp_1140; }
    let tmp_1181 = keys[25] < tmp_1141 || (keys[25] == tmp_1141 && values[25] < tmp_1142);
    if tmp_1155 == tmp_1181 { keys[25] = tmp_1141; values[25] = tmp_1142; }
    let tmp_1182 = keys[26] < tmp_1143 || (keys[26] == tmp_1143 && values[26] < tmp_1144);
    if tmp_1155 == tmp_1182 { keys[26] = tmp_1143; values[26] = tmp_1144; }
    let tmp_1183 = keys[27] < tmp_1145 || (keys[27] == tmp_1145 && values[27] < tmp_1146);
    if tmp_1155 == tmp_1183 { keys[27] = tmp_1145; values[27] = tmp_1146; }
    let tmp_1184 = keys[28] < tmp_1147 || (keys[28] == tmp_1147 && values[28] < tmp_1148);
    if tmp_1155 == tmp_1184 { keys[28] = tmp_1147; values[28] = tmp_1148; }
    let tmp_1185 = keys[29] < tmp_1149 || (keys[29] == tmp_1149 && values[29] < tmp_1150);
    if tmp_1155 == tmp_1185 { keys[29] = tmp_1149; values[29] = tmp_1150; }
    let tmp_1186 = keys[30] < tmp_1151 || (keys[30] == tmp_1151 && values[30] < tmp_1152);
    if tmp_1155 == tmp_1186 { keys[30] = tmp_1151; values[30] = tmp_1152; }
    let tmp_1187 = keys[31] < tmp_1153 || (keys[31] == tmp_1153 && values[31] < tmp_1154);
    if tmp_1155 == tmp_1187 { keys[31] = tmp_1153; values[31] = tmp_1154; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:32) 
    {
    let tmp_1188 = subgroupShuffleXor(keys[0], 2u);
    let tmp_1189 = subgroupShuffleXor(values[0], 2u);
    let tmp_1190 = subgroupShuffleXor(keys[1], 2u);
    let tmp_1191 = subgroupShuffleXor(values[1], 2u);
    let tmp_1192 = subgroupShuffleXor(keys[2], 2u);
    let tmp_1193 = subgroupShuffleXor(values[2], 2u);
    let tmp_1194 = subgroupShuffleXor(keys[3], 2u);
    let tmp_1195 = subgroupShuffleXor(values[3], 2u);
    let tmp_1196 = subgroupShuffleXor(keys[4], 2u);
    let tmp_1197 = subgroupShuffleXor(values[4], 2u);
    let tmp_1198 = subgroupShuffleXor(keys[5], 2u);
    let tmp_1199 = subgroupShuffleXor(values[5], 2u);
    let tmp_1200 = subgroupShuffleXor(keys[6], 2u);
    let tmp_1201 = subgroupShuffleXor(values[6], 2u);
    let tmp_1202 = subgroupShuffleXor(keys[7], 2u);
    let tmp_1203 = subgroupShuffleXor(values[7], 2u);
    let tmp_1204 = subgroupShuffleXor(keys[8], 2u);
    let tmp_1205 = subgroupShuffleXor(values[8], 2u);
    let tmp_1206 = subgroupShuffleXor(keys[9], 2u);
    let tmp_1207 = subgroupShuffleXor(values[9], 2u);
    let tmp_1208 = subgroupShuffleXor(keys[10], 2u);
    let tmp_1209 = subgroupShuffleXor(values[10], 2u);
    let tmp_1210 = subgroupShuffleXor(keys[11], 2u);
    let tmp_1211 = subgroupShuffleXor(values[11], 2u);
    let tmp_1212 = subgroupShuffleXor(keys[12], 2u);
    let tmp_1213 = subgroupShuffleXor(values[12], 2u);
    let tmp_1214 = subgroupShuffleXor(keys[13], 2u);
    let tmp_1215 = subgroupShuffleXor(values[13], 2u);
    let tmp_1216 = subgroupShuffleXor(keys[14], 2u);
    let tmp_1217 = subgroupShuffleXor(values[14], 2u);
    let tmp_1218 = subgroupShuffleXor(keys[15], 2u);
    let tmp_1219 = subgroupShuffleXor(values[15], 2u);
    let tmp_1220 = subgroupShuffleXor(keys[16], 2u);
    let tmp_1221 = subgroupShuffleXor(values[16], 2u);
    let tmp_1222 = subgroupShuffleXor(keys[17], 2u);
    let tmp_1223 = subgroupShuffleXor(values[17], 2u);
    let tmp_1224 = subgroupShuffleXor(keys[18], 2u);
    let tmp_1225 = subgroupShuffleXor(values[18], 2u);
    let tmp_1226 = subgroupShuffleXor(keys[19], 2u);
    let tmp_1227 = subgroupShuffleXor(values[19], 2u);
    let tmp_1228 = subgroupShuffleXor(keys[20], 2u);
    let tmp_1229 = subgroupShuffleXor(values[20], 2u);
    let tmp_1230 = subgroupShuffleXor(keys[21], 2u);
    let tmp_1231 = subgroupShuffleXor(values[21], 2u);
    let tmp_1232 = subgroupShuffleXor(keys[22], 2u);
    let tmp_1233 = subgroupShuffleXor(values[22], 2u);
    let tmp_1234 = subgroupShuffleXor(keys[23], 2u);
    let tmp_1235 = subgroupShuffleXor(values[23], 2u);
    let tmp_1236 = subgroupShuffleXor(keys[24], 2u);
    let tmp_1237 = subgroupShuffleXor(values[24], 2u);
    let tmp_1238 = subgroupShuffleXor(keys[25], 2u);
    let tmp_1239 = subgroupShuffleXor(values[25], 2u);
    let tmp_1240 = subgroupShuffleXor(keys[26], 2u);
    let tmp_1241 = subgroupShuffleXor(values[26], 2u);
    let tmp_1242 = subgroupShuffleXor(keys[27], 2u);
    let tmp_1243 = subgroupShuffleXor(values[27], 2u);
    let tmp_1244 = subgroupShuffleXor(keys[28], 2u);
    let tmp_1245 = subgroupShuffleXor(values[28], 2u);
    let tmp_1246 = subgroupShuffleXor(keys[29], 2u);
    let tmp_1247 = subgroupShuffleXor(values[29], 2u);
    let tmp_1248 = subgroupShuffleXor(keys[30], 2u);
    let tmp_1249 = subgroupShuffleXor(values[30], 2u);
    let tmp_1250 = subgroupShuffleXor(keys[31], 2u);
    let tmp_1251 = subgroupShuffleXor(values[31], 2u);
    let tmp_1252 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_1253 = keys[0] < tmp_1188 || (keys[0] == tmp_1188 && values[0] < tmp_1189);
    if tmp_1252 == tmp_1253 { keys[0] = tmp_1188; values[0] = tmp_1189; }
    let tmp_1254 = keys[1] < tmp_1190 || (keys[1] == tmp_1190 && values[1] < tmp_1191);
    if tmp_1252 == tmp_1254 { keys[1] = tmp_1190; values[1] = tmp_1191; }
    let tmp_1255 = keys[2] < tmp_1192 || (keys[2] == tmp_1192 && values[2] < tmp_1193);
    if tmp_1252 == tmp_1255 { keys[2] = tmp_1192; values[2] = tmp_1193; }
    let tmp_1256 = keys[3] < tmp_1194 || (keys[3] == tmp_1194 && values[3] < tmp_1195);
    if tmp_1252 == tmp_1256 { keys[3] = tmp_1194; values[3] = tmp_1195; }
    let tmp_1257 = keys[4] < tmp_1196 || (keys[4] == tmp_1196 && values[4] < tmp_1197);
    if tmp_1252 == tmp_1257 { keys[4] = tmp_1196; values[4] = tmp_1197; }
    let tmp_1258 = keys[5] < tmp_1198 || (keys[5] == tmp_1198 && values[5] < tmp_1199);
    if tmp_1252 == tmp_1258 { keys[5] = tmp_1198; values[5] = tmp_1199; }
    let tmp_1259 = keys[6] < tmp_1200 || (keys[6] == tmp_1200 && values[6] < tmp_1201);
    if tmp_1252 == tmp_1259 { keys[6] = tmp_1200; values[6] = tmp_1201; }
    let tmp_1260 = keys[7] < tmp_1202 || (keys[7] == tmp_1202 && values[7] < tmp_1203);
    if tmp_1252 == tmp_1260 { keys[7] = tmp_1202; values[7] = tmp_1203; }
    let tmp_1261 = keys[8] < tmp_1204 || (keys[8] == tmp_1204 && values[8] < tmp_1205);
    if tmp_1252 == tmp_1261 { keys[8] = tmp_1204; values[8] = tmp_1205; }
    let tmp_1262 = keys[9] < tmp_1206 || (keys[9] == tmp_1206 && values[9] < tmp_1207);
    if tmp_1252 == tmp_1262 { keys[9] = tmp_1206; values[9] = tmp_1207; }
    let tmp_1263 = keys[10] < tmp_1208 || (keys[10] == tmp_1208 && values[10] < tmp_1209);
    if tmp_1252 == tmp_1263 { keys[10] = tmp_1208; values[10] = tmp_1209; }
    let tmp_1264 = keys[11] < tmp_1210 || (keys[11] == tmp_1210 && values[11] < tmp_1211);
    if tmp_1252 == tmp_1264 { keys[11] = tmp_1210; values[11] = tmp_1211; }
    let tmp_1265 = keys[12] < tmp_1212 || (keys[12] == tmp_1212 && values[12] < tmp_1213);
    if tmp_1252 == tmp_1265 { keys[12] = tmp_1212; values[12] = tmp_1213; }
    let tmp_1266 = keys[13] < tmp_1214 || (keys[13] == tmp_1214 && values[13] < tmp_1215);
    if tmp_1252 == tmp_1266 { keys[13] = tmp_1214; values[13] = tmp_1215; }
    let tmp_1267 = keys[14] < tmp_1216 || (keys[14] == tmp_1216 && values[14] < tmp_1217);
    if tmp_1252 == tmp_1267 { keys[14] = tmp_1216; values[14] = tmp_1217; }
    let tmp_1268 = keys[15] < tmp_1218 || (keys[15] == tmp_1218 && values[15] < tmp_1219);
    if tmp_1252 == tmp_1268 { keys[15] = tmp_1218; values[15] = tmp_1219; }
    let tmp_1269 = keys[16] < tmp_1220 || (keys[16] == tmp_1220 && values[16] < tmp_1221);
    if tmp_1252 == tmp_1269 { keys[16] = tmp_1220; values[16] = tmp_1221; }
    let tmp_1270 = keys[17] < tmp_1222 || (keys[17] == tmp_1222 && values[17] < tmp_1223);
    if tmp_1252 == tmp_1270 { keys[17] = tmp_1222; values[17] = tmp_1223; }
    let tmp_1271 = keys[18] < tmp_1224 || (keys[18] == tmp_1224 && values[18] < tmp_1225);
    if tmp_1252 == tmp_1271 { keys[18] = tmp_1224; values[18] = tmp_1225; }
    let tmp_1272 = keys[19] < tmp_1226 || (keys[19] == tmp_1226 && values[19] < tmp_1227);
    if tmp_1252 == tmp_1272 { keys[19] = tmp_1226; values[19] = tmp_1227; }
    let tmp_1273 = keys[20] < tmp_1228 || (keys[20] == tmp_1228 && values[20] < tmp_1229);
    if tmp_1252 == tmp_1273 { keys[20] = tmp_1228; values[20] = tmp_1229; }
    let tmp_1274 = keys[21] < tmp_1230 || (keys[21] == tmp_1230 && values[21] < tmp_1231);
    if tmp_1252 == tmp_1274 { keys[21] = tmp_1230; values[21] = tmp_1231; }
    let tmp_1275 = keys[22] < tmp_1232 || (keys[22] == tmp_1232 && values[22] < tmp_1233);
    if tmp_1252 == tmp_1275 { keys[22] = tmp_1232; values[22] = tmp_1233; }
    let tmp_1276 = keys[23] < tmp_1234 || (keys[23] == tmp_1234 && values[23] < tmp_1235);
    if tmp_1252 == tmp_1276 { keys[23] = tmp_1234; values[23] = tmp_1235; }
    let tmp_1277 = keys[24] < tmp_1236 || (keys[24] == tmp_1236 && values[24] < tmp_1237);
    if tmp_1252 == tmp_1277 { keys[24] = tmp_1236; values[24] = tmp_1237; }
    let tmp_1278 = keys[25] < tmp_1238 || (keys[25] == tmp_1238 && values[25] < tmp_1239);
    if tmp_1252 == tmp_1278 { keys[25] = tmp_1238; values[25] = tmp_1239; }
    let tmp_1279 = keys[26] < tmp_1240 || (keys[26] == tmp_1240 && values[26] < tmp_1241);
    if tmp_1252 == tmp_1279 { keys[26] = tmp_1240; values[26] = tmp_1241; }
    let tmp_1280 = keys[27] < tmp_1242 || (keys[27] == tmp_1242 && values[27] < tmp_1243);
    if tmp_1252 == tmp_1280 { keys[27] = tmp_1242; values[27] = tmp_1243; }
    let tmp_1281 = keys[28] < tmp_1244 || (keys[28] == tmp_1244 && values[28] < tmp_1245);
    if tmp_1252 == tmp_1281 { keys[28] = tmp_1244; values[28] = tmp_1245; }
    let tmp_1282 = keys[29] < tmp_1246 || (keys[29] == tmp_1246 && values[29] < tmp_1247);
    if tmp_1252 == tmp_1282 { keys[29] = tmp_1246; values[29] = tmp_1247; }
    let tmp_1283 = keys[30] < tmp_1248 || (keys[30] == tmp_1248 && values[30] < tmp_1249);
    if tmp_1252 == tmp_1283 { keys[30] = tmp_1248; values[30] = tmp_1249; }
    let tmp_1284 = keys[31] < tmp_1250 || (keys[31] == tmp_1250 && values[31] < tmp_1251);
    if tmp_1252 == tmp_1284 { keys[31] = tmp_1250; values[31] = tmp_1251; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:32) 
    {
    let tmp_1285 = subgroupShuffleXor(keys[0], 1u);
    let tmp_1286 = subgroupShuffleXor(values[0], 1u);
    let tmp_1287 = subgroupShuffleXor(keys[1], 1u);
    let tmp_1288 = subgroupShuffleXor(values[1], 1u);
    let tmp_1289 = subgroupShuffleXor(keys[2], 1u);
    let tmp_1290 = subgroupShuffleXor(values[2], 1u);
    let tmp_1291 = subgroupShuffleXor(keys[3], 1u);
    let tmp_1292 = subgroupShuffleXor(values[3], 1u);
    let tmp_1293 = subgroupShuffleXor(keys[4], 1u);
    let tmp_1294 = subgroupShuffleXor(values[4], 1u);
    let tmp_1295 = subgroupShuffleXor(keys[5], 1u);
    let tmp_1296 = subgroupShuffleXor(values[5], 1u);
    let tmp_1297 = subgroupShuffleXor(keys[6], 1u);
    let tmp_1298 = subgroupShuffleXor(values[6], 1u);
    let tmp_1299 = subgroupShuffleXor(keys[7], 1u);
    let tmp_1300 = subgroupShuffleXor(values[7], 1u);
    let tmp_1301 = subgroupShuffleXor(keys[8], 1u);
    let tmp_1302 = subgroupShuffleXor(values[8], 1u);
    let tmp_1303 = subgroupShuffleXor(keys[9], 1u);
    let tmp_1304 = subgroupShuffleXor(values[9], 1u);
    let tmp_1305 = subgroupShuffleXor(keys[10], 1u);
    let tmp_1306 = subgroupShuffleXor(values[10], 1u);
    let tmp_1307 = subgroupShuffleXor(keys[11], 1u);
    let tmp_1308 = subgroupShuffleXor(values[11], 1u);
    let tmp_1309 = subgroupShuffleXor(keys[12], 1u);
    let tmp_1310 = subgroupShuffleXor(values[12], 1u);
    let tmp_1311 = subgroupShuffleXor(keys[13], 1u);
    let tmp_1312 = subgroupShuffleXor(values[13], 1u);
    let tmp_1313 = subgroupShuffleXor(keys[14], 1u);
    let tmp_1314 = subgroupShuffleXor(values[14], 1u);
    let tmp_1315 = subgroupShuffleXor(keys[15], 1u);
    let tmp_1316 = subgroupShuffleXor(values[15], 1u);
    let tmp_1317 = subgroupShuffleXor(keys[16], 1u);
    let tmp_1318 = subgroupShuffleXor(values[16], 1u);
    let tmp_1319 = subgroupShuffleXor(keys[17], 1u);
    let tmp_1320 = subgroupShuffleXor(values[17], 1u);
    let tmp_1321 = subgroupShuffleXor(keys[18], 1u);
    let tmp_1322 = subgroupShuffleXor(values[18], 1u);
    let tmp_1323 = subgroupShuffleXor(keys[19], 1u);
    let tmp_1324 = subgroupShuffleXor(values[19], 1u);
    let tmp_1325 = subgroupShuffleXor(keys[20], 1u);
    let tmp_1326 = subgroupShuffleXor(values[20], 1u);
    let tmp_1327 = subgroupShuffleXor(keys[21], 1u);
    let tmp_1328 = subgroupShuffleXor(values[21], 1u);
    let tmp_1329 = subgroupShuffleXor(keys[22], 1u);
    let tmp_1330 = subgroupShuffleXor(values[22], 1u);
    let tmp_1331 = subgroupShuffleXor(keys[23], 1u);
    let tmp_1332 = subgroupShuffleXor(values[23], 1u);
    let tmp_1333 = subgroupShuffleXor(keys[24], 1u);
    let tmp_1334 = subgroupShuffleXor(values[24], 1u);
    let tmp_1335 = subgroupShuffleXor(keys[25], 1u);
    let tmp_1336 = subgroupShuffleXor(values[25], 1u);
    let tmp_1337 = subgroupShuffleXor(keys[26], 1u);
    let tmp_1338 = subgroupShuffleXor(values[26], 1u);
    let tmp_1339 = subgroupShuffleXor(keys[27], 1u);
    let tmp_1340 = subgroupShuffleXor(values[27], 1u);
    let tmp_1341 = subgroupShuffleXor(keys[28], 1u);
    let tmp_1342 = subgroupShuffleXor(values[28], 1u);
    let tmp_1343 = subgroupShuffleXor(keys[29], 1u);
    let tmp_1344 = subgroupShuffleXor(values[29], 1u);
    let tmp_1345 = subgroupShuffleXor(keys[30], 1u);
    let tmp_1346 = subgroupShuffleXor(values[30], 1u);
    let tmp_1347 = subgroupShuffleXor(keys[31], 1u);
    let tmp_1348 = subgroupShuffleXor(values[31], 1u);
    let tmp_1349 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_1350 = keys[0] < tmp_1285 || (keys[0] == tmp_1285 && values[0] < tmp_1286);
    if tmp_1349 == tmp_1350 { keys[0] = tmp_1285; values[0] = tmp_1286; }
    let tmp_1351 = keys[1] < tmp_1287 || (keys[1] == tmp_1287 && values[1] < tmp_1288);
    if tmp_1349 == tmp_1351 { keys[1] = tmp_1287; values[1] = tmp_1288; }
    let tmp_1352 = keys[2] < tmp_1289 || (keys[2] == tmp_1289 && values[2] < tmp_1290);
    if tmp_1349 == tmp_1352 { keys[2] = tmp_1289; values[2] = tmp_1290; }
    let tmp_1353 = keys[3] < tmp_1291 || (keys[3] == tmp_1291 && values[3] < tmp_1292);
    if tmp_1349 == tmp_1353 { keys[3] = tmp_1291; values[3] = tmp_1292; }
    let tmp_1354 = keys[4] < tmp_1293 || (keys[4] == tmp_1293 && values[4] < tmp_1294);
    if tmp_1349 == tmp_1354 { keys[4] = tmp_1293; values[4] = tmp_1294; }
    let tmp_1355 = keys[5] < tmp_1295 || (keys[5] == tmp_1295 && values[5] < tmp_1296);
    if tmp_1349 == tmp_1355 { keys[5] = tmp_1295; values[5] = tmp_1296; }
    let tmp_1356 = keys[6] < tmp_1297 || (keys[6] == tmp_1297 && values[6] < tmp_1298);
    if tmp_1349 == tmp_1356 { keys[6] = tmp_1297; values[6] = tmp_1298; }
    let tmp_1357 = keys[7] < tmp_1299 || (keys[7] == tmp_1299 && values[7] < tmp_1300);
    if tmp_1349 == tmp_1357 { keys[7] = tmp_1299; values[7] = tmp_1300; }
    let tmp_1358 = keys[8] < tmp_1301 || (keys[8] == tmp_1301 && values[8] < tmp_1302);
    if tmp_1349 == tmp_1358 { keys[8] = tmp_1301; values[8] = tmp_1302; }
    let tmp_1359 = keys[9] < tmp_1303 || (keys[9] == tmp_1303 && values[9] < tmp_1304);
    if tmp_1349 == tmp_1359 { keys[9] = tmp_1303; values[9] = tmp_1304; }
    let tmp_1360 = keys[10] < tmp_1305 || (keys[10] == tmp_1305 && values[10] < tmp_1306);
    if tmp_1349 == tmp_1360 { keys[10] = tmp_1305; values[10] = tmp_1306; }
    let tmp_1361 = keys[11] < tmp_1307 || (keys[11] == tmp_1307 && values[11] < tmp_1308);
    if tmp_1349 == tmp_1361 { keys[11] = tmp_1307; values[11] = tmp_1308; }
    let tmp_1362 = keys[12] < tmp_1309 || (keys[12] == tmp_1309 && values[12] < tmp_1310);
    if tmp_1349 == tmp_1362 { keys[12] = tmp_1309; values[12] = tmp_1310; }
    let tmp_1363 = keys[13] < tmp_1311 || (keys[13] == tmp_1311 && values[13] < tmp_1312);
    if tmp_1349 == tmp_1363 { keys[13] = tmp_1311; values[13] = tmp_1312; }
    let tmp_1364 = keys[14] < tmp_1313 || (keys[14] == tmp_1313 && values[14] < tmp_1314);
    if tmp_1349 == tmp_1364 { keys[14] = tmp_1313; values[14] = tmp_1314; }
    let tmp_1365 = keys[15] < tmp_1315 || (keys[15] == tmp_1315 && values[15] < tmp_1316);
    if tmp_1349 == tmp_1365 { keys[15] = tmp_1315; values[15] = tmp_1316; }
    let tmp_1366 = keys[16] < tmp_1317 || (keys[16] == tmp_1317 && values[16] < tmp_1318);
    if tmp_1349 == tmp_1366 { keys[16] = tmp_1317; values[16] = tmp_1318; }
    let tmp_1367 = keys[17] < tmp_1319 || (keys[17] == tmp_1319 && values[17] < tmp_1320);
    if tmp_1349 == tmp_1367 { keys[17] = tmp_1319; values[17] = tmp_1320; }
    let tmp_1368 = keys[18] < tmp_1321 || (keys[18] == tmp_1321 && values[18] < tmp_1322);
    if tmp_1349 == tmp_1368 { keys[18] = tmp_1321; values[18] = tmp_1322; }
    let tmp_1369 = keys[19] < tmp_1323 || (keys[19] == tmp_1323 && values[19] < tmp_1324);
    if tmp_1349 == tmp_1369 { keys[19] = tmp_1323; values[19] = tmp_1324; }
    let tmp_1370 = keys[20] < tmp_1325 || (keys[20] == tmp_1325 && values[20] < tmp_1326);
    if tmp_1349 == tmp_1370 { keys[20] = tmp_1325; values[20] = tmp_1326; }
    let tmp_1371 = keys[21] < tmp_1327 || (keys[21] == tmp_1327 && values[21] < tmp_1328);
    if tmp_1349 == tmp_1371 { keys[21] = tmp_1327; values[21] = tmp_1328; }
    let tmp_1372 = keys[22] < tmp_1329 || (keys[22] == tmp_1329 && values[22] < tmp_1330);
    if tmp_1349 == tmp_1372 { keys[22] = tmp_1329; values[22] = tmp_1330; }
    let tmp_1373 = keys[23] < tmp_1331 || (keys[23] == tmp_1331 && values[23] < tmp_1332);
    if tmp_1349 == tmp_1373 { keys[23] = tmp_1331; values[23] = tmp_1332; }
    let tmp_1374 = keys[24] < tmp_1333 || (keys[24] == tmp_1333 && values[24] < tmp_1334);
    if tmp_1349 == tmp_1374 { keys[24] = tmp_1333; values[24] = tmp_1334; }
    let tmp_1375 = keys[25] < tmp_1335 || (keys[25] == tmp_1335 && values[25] < tmp_1336);
    if tmp_1349 == tmp_1375 { keys[25] = tmp_1335; values[25] = tmp_1336; }
    let tmp_1376 = keys[26] < tmp_1337 || (keys[26] == tmp_1337 && values[26] < tmp_1338);
    if tmp_1349 == tmp_1376 { keys[26] = tmp_1337; values[26] = tmp_1338; }
    let tmp_1377 = keys[27] < tmp_1339 || (keys[27] == tmp_1339 && values[27] < tmp_1340);
    if tmp_1349 == tmp_1377 { keys[27] = tmp_1339; values[27] = tmp_1340; }
    let tmp_1378 = keys[28] < tmp_1341 || (keys[28] == tmp_1341 && values[28] < tmp_1342);
    if tmp_1349 == tmp_1378 { keys[28] = tmp_1341; values[28] = tmp_1342; }
    let tmp_1379 = keys[29] < tmp_1343 || (keys[29] == tmp_1343 && values[29] < tmp_1344);
    if tmp_1349 == tmp_1379 { keys[29] = tmp_1343; values[29] = tmp_1344; }
    let tmp_1380 = keys[30] < tmp_1345 || (keys[30] == tmp_1345 && values[30] < tmp_1346);
    if tmp_1349 == tmp_1380 { keys[30] = tmp_1345; values[30] = tmp_1346; }
    let tmp_1381 = keys[31] < tmp_1347 || (keys[31] == tmp_1347 && values[31] < tmp_1348);
    if tmp_1349 == tmp_1381 { keys[31] = tmp_1347; values[31] = tmp_1348; }
    }
    // exch_local(16,32) 
    // cmp_swap(0,16)
    if keys[0] > keys[16] || (keys[0] == keys[16] && values[0] > values[16]) {
    // swap(0,16) 
    { let tmp_1382 = keys[0]; keys[0] = keys[16]; keys[16] = tmp_1382;let tmp_1383 = values[0]; values[0] = values[16]; values[16] = tmp_1383; }
    }
    // cmp_swap(1,17)
    if keys[1] > keys[17] || (keys[1] == keys[17] && values[1] > values[17]) {
    // swap(1,17) 
    { let tmp_1384 = keys[1]; keys[1] = keys[17]; keys[17] = tmp_1384;let tmp_1385 = values[1]; values[1] = values[17]; values[17] = tmp_1385; }
    }
    // cmp_swap(2,18)
    if keys[2] > keys[18] || (keys[2] == keys[18] && values[2] > values[18]) {
    // swap(2,18) 
    { let tmp_1386 = keys[2]; keys[2] = keys[18]; keys[18] = tmp_1386;let tmp_1387 = values[2]; values[2] = values[18]; values[18] = tmp_1387; }
    }
    // cmp_swap(3,19)
    if keys[3] > keys[19] || (keys[3] == keys[19] && values[3] > values[19]) {
    // swap(3,19) 
    { let tmp_1388 = keys[3]; keys[3] = keys[19]; keys[19] = tmp_1388;let tmp_1389 = values[3]; values[3] = values[19]; values[19] = tmp_1389; }
    }
    // cmp_swap(4,20)
    if keys[4] > keys[20] || (keys[4] == keys[20] && values[4] > values[20]) {
    // swap(4,20) 
    { let tmp_1390 = keys[4]; keys[4] = keys[20]; keys[20] = tmp_1390;let tmp_1391 = values[4]; values[4] = values[20]; values[20] = tmp_1391; }
    }
    // cmp_swap(5,21)
    if keys[5] > keys[21] || (keys[5] == keys[21] && values[5] > values[21]) {
    // swap(5,21) 
    { let tmp_1392 = keys[5]; keys[5] = keys[21]; keys[21] = tmp_1392;let tmp_1393 = values[5]; values[5] = values[21]; values[21] = tmp_1393; }
    }
    // cmp_swap(6,22)
    if keys[6] > keys[22] || (keys[6] == keys[22] && values[6] > values[22]) {
    // swap(6,22) 
    { let tmp_1394 = keys[6]; keys[6] = keys[22]; keys[22] = tmp_1394;let tmp_1395 = values[6]; values[6] = values[22]; values[22] = tmp_1395; }
    }
    // cmp_swap(7,23)
    if keys[7] > keys[23] || (keys[7] == keys[23] && values[7] > values[23]) {
    // swap(7,23) 
    { let tmp_1396 = keys[7]; keys[7] = keys[23]; keys[23] = tmp_1396;let tmp_1397 = values[7]; values[7] = values[23]; values[23] = tmp_1397; }
    }
    // cmp_swap(8,24)
    if keys[8] > keys[24] || (keys[8] == keys[24] && values[8] > values[24]) {
    // swap(8,24) 
    { let tmp_1398 = keys[8]; keys[8] = keys[24]; keys[24] = tmp_1398;let tmp_1399 = values[8]; values[8] = values[24]; values[24] = tmp_1399; }
    }
    // cmp_swap(9,25)
    if keys[9] > keys[25] || (keys[9] == keys[25] && values[9] > values[25]) {
    // swap(9,25) 
    { let tmp_1400 = keys[9]; keys[9] = keys[25]; keys[25] = tmp_1400;let tmp_1401 = values[9]; values[9] = values[25]; values[25] = tmp_1401; }
    }
    // cmp_swap(10,26)
    if keys[10] > keys[26] || (keys[10] == keys[26] && values[10] > values[26]) {
    // swap(10,26) 
    { let tmp_1402 = keys[10]; keys[10] = keys[26]; keys[26] = tmp_1402;let tmp_1403 = values[10]; values[10] = values[26]; values[26] = tmp_1403; }
    }
    // cmp_swap(11,27)
    if keys[11] > keys[27] || (keys[11] == keys[27] && values[11] > values[27]) {
    // swap(11,27) 
    { let tmp_1404 = keys[11]; keys[11] = keys[27]; keys[27] = tmp_1404;let tmp_1405 = values[11]; values[11] = values[27]; values[27] = tmp_1405; }
    }
    // cmp_swap(12,28)
    if keys[12] > keys[28] || (keys[12] == keys[28] && values[12] > values[28]) {
    // swap(12,28) 
    { let tmp_1406 = keys[12]; keys[12] = keys[28]; keys[28] = tmp_1406;let tmp_1407 = values[12]; values[12] = values[28]; values[28] = tmp_1407; }
    }
    // cmp_swap(13,29)
    if keys[13] > keys[29] || (keys[13] == keys[29] && values[13] > values[29]) {
    // swap(13,29) 
    { let tmp_1408 = keys[13]; keys[13] = keys[29]; keys[29] = tmp_1408;let tmp_1409 = values[13]; values[13] = values[29]; values[29] = tmp_1409; }
    }
    // cmp_swap(14,30)
    if keys[14] > keys[30] || (keys[14] == keys[30] && values[14] > values[30]) {
    // swap(14,30) 
    { let tmp_1410 = keys[14]; keys[14] = keys[30]; keys[30] = tmp_1410;let tmp_1411 = values[14]; values[14] = values[30]; values[30] = tmp_1411; }
    }
    // cmp_swap(15,31)
    if keys[15] > keys[31] || (keys[15] == keys[31] && values[15] > values[31]) {
    // swap(15,31) 
    { let tmp_1412 = keys[15]; keys[15] = keys[31]; keys[31] = tmp_1412;let tmp_1413 = values[15]; values[15] = values[31]; values[31] = tmp_1413; }
    }
    // exch_local(8,32) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_1414 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_1414;let tmp_1415 = values[0]; values[0] = values[8]; values[8] = tmp_1415; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_1416 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_1416;let tmp_1417 = values[1]; values[1] = values[9]; values[9] = tmp_1417; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_1418 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_1418;let tmp_1419 = values[2]; values[2] = values[10]; values[10] = tmp_1419; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_1420 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_1420;let tmp_1421 = values[3]; values[3] = values[11]; values[11] = tmp_1421; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_1422 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_1422;let tmp_1423 = values[4]; values[4] = values[12]; values[12] = tmp_1423; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_1424 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_1424;let tmp_1425 = values[5]; values[5] = values[13]; values[13] = tmp_1425; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_1426 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_1426;let tmp_1427 = values[6]; values[6] = values[14]; values[14] = tmp_1427; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_1428 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_1428;let tmp_1429 = values[7]; values[7] = values[15]; values[15] = tmp_1429; }
    }
    // cmp_swap(16,24)
    if keys[16] > keys[24] || (keys[16] == keys[24] && values[16] > values[24]) {
    // swap(16,24) 
    { let tmp_1430 = keys[16]; keys[16] = keys[24]; keys[24] = tmp_1430;let tmp_1431 = values[16]; values[16] = values[24]; values[24] = tmp_1431; }
    }
    // cmp_swap(17,25)
    if keys[17] > keys[25] || (keys[17] == keys[25] && values[17] > values[25]) {
    // swap(17,25) 
    { let tmp_1432 = keys[17]; keys[17] = keys[25]; keys[25] = tmp_1432;let tmp_1433 = values[17]; values[17] = values[25]; values[25] = tmp_1433; }
    }
    // cmp_swap(18,26)
    if keys[18] > keys[26] || (keys[18] == keys[26] && values[18] > values[26]) {
    // swap(18,26) 
    { let tmp_1434 = keys[18]; keys[18] = keys[26]; keys[26] = tmp_1434;let tmp_1435 = values[18]; values[18] = values[26]; values[26] = tmp_1435; }
    }
    // cmp_swap(19,27)
    if keys[19] > keys[27] || (keys[19] == keys[27] && values[19] > values[27]) {
    // swap(19,27) 
    { let tmp_1436 = keys[19]; keys[19] = keys[27]; keys[27] = tmp_1436;let tmp_1437 = values[19]; values[19] = values[27]; values[27] = tmp_1437; }
    }
    // cmp_swap(20,28)
    if keys[20] > keys[28] || (keys[20] == keys[28] && values[20] > values[28]) {
    // swap(20,28) 
    { let tmp_1438 = keys[20]; keys[20] = keys[28]; keys[28] = tmp_1438;let tmp_1439 = values[20]; values[20] = values[28]; values[28] = tmp_1439; }
    }
    // cmp_swap(21,29)
    if keys[21] > keys[29] || (keys[21] == keys[29] && values[21] > values[29]) {
    // swap(21,29) 
    { let tmp_1440 = keys[21]; keys[21] = keys[29]; keys[29] = tmp_1440;let tmp_1441 = values[21]; values[21] = values[29]; values[29] = tmp_1441; }
    }
    // cmp_swap(22,30)
    if keys[22] > keys[30] || (keys[22] == keys[30] && values[22] > values[30]) {
    // swap(22,30) 
    { let tmp_1442 = keys[22]; keys[22] = keys[30]; keys[30] = tmp_1442;let tmp_1443 = values[22]; values[22] = values[30]; values[30] = tmp_1443; }
    }
    // cmp_swap(23,31)
    if keys[23] > keys[31] || (keys[23] == keys[31] && values[23] > values[31]) {
    // swap(23,31) 
    { let tmp_1444 = keys[23]; keys[23] = keys[31]; keys[31] = tmp_1444;let tmp_1445 = values[23]; values[23] = values[31]; values[31] = tmp_1445; }
    }
    // exch_local(4,32) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_1446 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_1446;let tmp_1447 = values[0]; values[0] = values[4]; values[4] = tmp_1447; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_1448 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_1448;let tmp_1449 = values[1]; values[1] = values[5]; values[5] = tmp_1449; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_1450 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_1450;let tmp_1451 = values[2]; values[2] = values[6]; values[6] = tmp_1451; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_1452 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_1452;let tmp_1453 = values[3]; values[3] = values[7]; values[7] = tmp_1453; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_1454 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_1454;let tmp_1455 = values[8]; values[8] = values[12]; values[12] = tmp_1455; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_1456 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_1456;let tmp_1457 = values[9]; values[9] = values[13]; values[13] = tmp_1457; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_1458 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_1458;let tmp_1459 = values[10]; values[10] = values[14]; values[14] = tmp_1459; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_1460 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_1460;let tmp_1461 = values[11]; values[11] = values[15]; values[15] = tmp_1461; }
    }
    // cmp_swap(16,20)
    if keys[16] > keys[20] || (keys[16] == keys[20] && values[16] > values[20]) {
    // swap(16,20) 
    { let tmp_1462 = keys[16]; keys[16] = keys[20]; keys[20] = tmp_1462;let tmp_1463 = values[16]; values[16] = values[20]; values[20] = tmp_1463; }
    }
    // cmp_swap(17,21)
    if keys[17] > keys[21] || (keys[17] == keys[21] && values[17] > values[21]) {
    // swap(17,21) 
    { let tmp_1464 = keys[17]; keys[17] = keys[21]; keys[21] = tmp_1464;let tmp_1465 = values[17]; values[17] = values[21]; values[21] = tmp_1465; }
    }
    // cmp_swap(18,22)
    if keys[18] > keys[22] || (keys[18] == keys[22] && values[18] > values[22]) {
    // swap(18,22) 
    { let tmp_1466 = keys[18]; keys[18] = keys[22]; keys[22] = tmp_1466;let tmp_1467 = values[18]; values[18] = values[22]; values[22] = tmp_1467; }
    }
    // cmp_swap(19,23)
    if keys[19] > keys[23] || (keys[19] == keys[23] && values[19] > values[23]) {
    // swap(19,23) 
    { let tmp_1468 = keys[19]; keys[19] = keys[23]; keys[23] = tmp_1468;let tmp_1469 = values[19]; values[19] = values[23]; values[23] = tmp_1469; }
    }
    // cmp_swap(24,28)
    if keys[24] > keys[28] || (keys[24] == keys[28] && values[24] > values[28]) {
    // swap(24,28) 
    { let tmp_1470 = keys[24]; keys[24] = keys[28]; keys[28] = tmp_1470;let tmp_1471 = values[24]; values[24] = values[28]; values[28] = tmp_1471; }
    }
    // cmp_swap(25,29)
    if keys[25] > keys[29] || (keys[25] == keys[29] && values[25] > values[29]) {
    // swap(25,29) 
    { let tmp_1472 = keys[25]; keys[25] = keys[29]; keys[29] = tmp_1472;let tmp_1473 = values[25]; values[25] = values[29]; values[29] = tmp_1473; }
    }
    // cmp_swap(26,30)
    if keys[26] > keys[30] || (keys[26] == keys[30] && values[26] > values[30]) {
    // swap(26,30) 
    { let tmp_1474 = keys[26]; keys[26] = keys[30]; keys[30] = tmp_1474;let tmp_1475 = values[26]; values[26] = values[30]; values[30] = tmp_1475; }
    }
    // cmp_swap(27,31)
    if keys[27] > keys[31] || (keys[27] == keys[31] && values[27] > values[31]) {
    // swap(27,31) 
    { let tmp_1476 = keys[27]; keys[27] = keys[31]; keys[31] = tmp_1476;let tmp_1477 = values[27]; values[27] = values[31]; values[31] = tmp_1477; }
    }
    // exch_local(2,32) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_1478 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_1478;let tmp_1479 = values[0]; values[0] = values[2]; values[2] = tmp_1479; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_1480 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_1480;let tmp_1481 = values[1]; values[1] = values[3]; values[3] = tmp_1481; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_1482 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_1482;let tmp_1483 = values[4]; values[4] = values[6]; values[6] = tmp_1483; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_1484 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_1484;let tmp_1485 = values[5]; values[5] = values[7]; values[7] = tmp_1485; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_1486 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_1486;let tmp_1487 = values[8]; values[8] = values[10]; values[10] = tmp_1487; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_1488 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_1488;let tmp_1489 = values[9]; values[9] = values[11]; values[11] = tmp_1489; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_1490 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_1490;let tmp_1491 = values[12]; values[12] = values[14]; values[14] = tmp_1491; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_1492 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_1492;let tmp_1493 = values[13]; values[13] = values[15]; values[15] = tmp_1493; }
    }
    // cmp_swap(16,18)
    if keys[16] > keys[18] || (keys[16] == keys[18] && values[16] > values[18]) {
    // swap(16,18) 
    { let tmp_1494 = keys[16]; keys[16] = keys[18]; keys[18] = tmp_1494;let tmp_1495 = values[16]; values[16] = values[18]; values[18] = tmp_1495; }
    }
    // cmp_swap(17,19)
    if keys[17] > keys[19] || (keys[17] == keys[19] && values[17] > values[19]) {
    // swap(17,19) 
    { let tmp_1496 = keys[17]; keys[17] = keys[19]; keys[19] = tmp_1496;let tmp_1497 = values[17]; values[17] = values[19]; values[19] = tmp_1497; }
    }
    // cmp_swap(20,22)
    if keys[20] > keys[22] || (keys[20] == keys[22] && values[20] > values[22]) {
    // swap(20,22) 
    { let tmp_1498 = keys[20]; keys[20] = keys[22]; keys[22] = tmp_1498;let tmp_1499 = values[20]; values[20] = values[22]; values[22] = tmp_1499; }
    }
    // cmp_swap(21,23)
    if keys[21] > keys[23] || (keys[21] == keys[23] && values[21] > values[23]) {
    // swap(21,23) 
    { let tmp_1500 = keys[21]; keys[21] = keys[23]; keys[23] = tmp_1500;let tmp_1501 = values[21]; values[21] = values[23]; values[23] = tmp_1501; }
    }
    // cmp_swap(24,26)
    if keys[24] > keys[26] || (keys[24] == keys[26] && values[24] > values[26]) {
    // swap(24,26) 
    { let tmp_1502 = keys[24]; keys[24] = keys[26]; keys[26] = tmp_1502;let tmp_1503 = values[24]; values[24] = values[26]; values[26] = tmp_1503; }
    }
    // cmp_swap(25,27)
    if keys[25] > keys[27] || (keys[25] == keys[27] && values[25] > values[27]) {
    // swap(25,27) 
    { let tmp_1504 = keys[25]; keys[25] = keys[27]; keys[27] = tmp_1504;let tmp_1505 = values[25]; values[25] = values[27]; values[27] = tmp_1505; }
    }
    // cmp_swap(28,30)
    if keys[28] > keys[30] || (keys[28] == keys[30] && values[28] > values[30]) {
    // swap(28,30) 
    { let tmp_1506 = keys[28]; keys[28] = keys[30]; keys[30] = tmp_1506;let tmp_1507 = values[28]; values[28] = values[30]; values[30] = tmp_1507; }
    }
    // cmp_swap(29,31)
    if keys[29] > keys[31] || (keys[29] == keys[31] && values[29] > values[31]) {
    // swap(29,31) 
    { let tmp_1508 = keys[29]; keys[29] = keys[31]; keys[31] = tmp_1508;let tmp_1509 = values[29]; values[29] = values[31]; values[31] = tmp_1509; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_1510 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_1510;let tmp_1511 = values[0]; values[0] = values[1]; values[1] = tmp_1511; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_1512 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_1512;let tmp_1513 = values[2]; values[2] = values[3]; values[3] = tmp_1513; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_1514 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_1514;let tmp_1515 = values[4]; values[4] = values[5]; values[5] = tmp_1515; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_1516 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_1516;let tmp_1517 = values[6]; values[6] = values[7]; values[7] = tmp_1517; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_1518 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_1518;let tmp_1519 = values[8]; values[8] = values[9]; values[9] = tmp_1519; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_1520 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_1520;let tmp_1521 = values[10]; values[10] = values[11]; values[11] = tmp_1521; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_1522 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_1522;let tmp_1523 = values[12]; values[12] = values[13]; values[13] = tmp_1523; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_1524 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_1524;let tmp_1525 = values[14]; values[14] = values[15]; values[15] = tmp_1525; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_1526 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_1526;let tmp_1527 = values[16]; values[16] = values[17]; values[17] = tmp_1527; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_1528 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_1528;let tmp_1529 = values[18]; values[18] = values[19]; values[19] = tmp_1529; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_1530 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_1530;let tmp_1531 = values[20]; values[20] = values[21]; values[21] = tmp_1531; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_1532 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_1532;let tmp_1533 = values[22]; values[22] = values[23]; values[23] = tmp_1533; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_1534 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_1534;let tmp_1535 = values[24]; values[24] = values[25]; values[25] = tmp_1535; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_1536 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_1536;let tmp_1537 = values[26]; values[26] = values[27]; values[27] = tmp_1537; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_1538 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_1538;let tmp_1539 = values[28]; values[28] = values[29]; values[29] = tmp_1539; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_1540 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_1540;let tmp_1541 = values[30]; values[30] = values[31]; values[31] = tmp_1541; }
    }
    // exch_intxn(tmask:15,swbit:3,wpt:32)
    {
    let tmp_1542 = subgroupShuffleXor(keys[31], 15u);
    let tmp_1543 = subgroupShuffleXor(values[31], 15u);
    let tmp_1544 = subgroupShuffleXor(keys[30], 15u);
    let tmp_1545 = subgroupShuffleXor(values[30], 15u);
    let tmp_1546 = subgroupShuffleXor(keys[29], 15u);
    let tmp_1547 = subgroupShuffleXor(values[29], 15u);
    let tmp_1548 = subgroupShuffleXor(keys[28], 15u);
    let tmp_1549 = subgroupShuffleXor(values[28], 15u);
    let tmp_1550 = subgroupShuffleXor(keys[27], 15u);
    let tmp_1551 = subgroupShuffleXor(values[27], 15u);
    let tmp_1552 = subgroupShuffleXor(keys[26], 15u);
    let tmp_1553 = subgroupShuffleXor(values[26], 15u);
    let tmp_1554 = subgroupShuffleXor(keys[25], 15u);
    let tmp_1555 = subgroupShuffleXor(values[25], 15u);
    let tmp_1556 = subgroupShuffleXor(keys[24], 15u);
    let tmp_1557 = subgroupShuffleXor(values[24], 15u);
    let tmp_1558 = subgroupShuffleXor(keys[23], 15u);
    let tmp_1559 = subgroupShuffleXor(values[23], 15u);
    let tmp_1560 = subgroupShuffleXor(keys[22], 15u);
    let tmp_1561 = subgroupShuffleXor(values[22], 15u);
    let tmp_1562 = subgroupShuffleXor(keys[21], 15u);
    let tmp_1563 = subgroupShuffleXor(values[21], 15u);
    let tmp_1564 = subgroupShuffleXor(keys[20], 15u);
    let tmp_1565 = subgroupShuffleXor(values[20], 15u);
    let tmp_1566 = subgroupShuffleXor(keys[19], 15u);
    let tmp_1567 = subgroupShuffleXor(values[19], 15u);
    let tmp_1568 = subgroupShuffleXor(keys[18], 15u);
    let tmp_1569 = subgroupShuffleXor(values[18], 15u);
    let tmp_1570 = subgroupShuffleXor(keys[17], 15u);
    let tmp_1571 = subgroupShuffleXor(values[17], 15u);
    let tmp_1572 = subgroupShuffleXor(keys[16], 15u);
    let tmp_1573 = subgroupShuffleXor(values[16], 15u);
    let tmp_1574 = subgroupShuffleXor(keys[15], 15u);
    let tmp_1575 = subgroupShuffleXor(values[15], 15u);
    let tmp_1576 = subgroupShuffleXor(keys[14], 15u);
    let tmp_1577 = subgroupShuffleXor(values[14], 15u);
    let tmp_1578 = subgroupShuffleXor(keys[13], 15u);
    let tmp_1579 = subgroupShuffleXor(values[13], 15u);
    let tmp_1580 = subgroupShuffleXor(keys[12], 15u);
    let tmp_1581 = subgroupShuffleXor(values[12], 15u);
    let tmp_1582 = subgroupShuffleXor(keys[11], 15u);
    let tmp_1583 = subgroupShuffleXor(values[11], 15u);
    let tmp_1584 = subgroupShuffleXor(keys[10], 15u);
    let tmp_1585 = subgroupShuffleXor(values[10], 15u);
    let tmp_1586 = subgroupShuffleXor(keys[9], 15u);
    let tmp_1587 = subgroupShuffleXor(values[9], 15u);
    let tmp_1588 = subgroupShuffleXor(keys[8], 15u);
    let tmp_1589 = subgroupShuffleXor(values[8], 15u);
    let tmp_1590 = subgroupShuffleXor(keys[7], 15u);
    let tmp_1591 = subgroupShuffleXor(values[7], 15u);
    let tmp_1592 = subgroupShuffleXor(keys[6], 15u);
    let tmp_1593 = subgroupShuffleXor(values[6], 15u);
    let tmp_1594 = subgroupShuffleXor(keys[5], 15u);
    let tmp_1595 = subgroupShuffleXor(values[5], 15u);
    let tmp_1596 = subgroupShuffleXor(keys[4], 15u);
    let tmp_1597 = subgroupShuffleXor(values[4], 15u);
    let tmp_1598 = subgroupShuffleXor(keys[3], 15u);
    let tmp_1599 = subgroupShuffleXor(values[3], 15u);
    let tmp_1600 = subgroupShuffleXor(keys[2], 15u);
    let tmp_1601 = subgroupShuffleXor(values[2], 15u);
    let tmp_1602 = subgroupShuffleXor(keys[1], 15u);
    let tmp_1603 = subgroupShuffleXor(values[1], 15u);
    let tmp_1604 = subgroupShuffleXor(keys[0], 15u);
    let tmp_1605 = subgroupShuffleXor(values[0], 15u);
    let tmp_1606 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_1607 = keys[0] < tmp_1542 || (keys[0] == tmp_1542 && values[0] < tmp_1543);
    if tmp_1606 == tmp_1607 { keys[0] = tmp_1542; values[0] = tmp_1543; }
    let tmp_1608 = keys[1] < tmp_1544 || (keys[1] == tmp_1544 && values[1] < tmp_1545);
    if tmp_1606 == tmp_1608 { keys[1] = tmp_1544; values[1] = tmp_1545; }
    let tmp_1609 = keys[2] < tmp_1546 || (keys[2] == tmp_1546 && values[2] < tmp_1547);
    if tmp_1606 == tmp_1609 { keys[2] = tmp_1546; values[2] = tmp_1547; }
    let tmp_1610 = keys[3] < tmp_1548 || (keys[3] == tmp_1548 && values[3] < tmp_1549);
    if tmp_1606 == tmp_1610 { keys[3] = tmp_1548; values[3] = tmp_1549; }
    let tmp_1611 = keys[4] < tmp_1550 || (keys[4] == tmp_1550 && values[4] < tmp_1551);
    if tmp_1606 == tmp_1611 { keys[4] = tmp_1550; values[4] = tmp_1551; }
    let tmp_1612 = keys[5] < tmp_1552 || (keys[5] == tmp_1552 && values[5] < tmp_1553);
    if tmp_1606 == tmp_1612 { keys[5] = tmp_1552; values[5] = tmp_1553; }
    let tmp_1613 = keys[6] < tmp_1554 || (keys[6] == tmp_1554 && values[6] < tmp_1555);
    if tmp_1606 == tmp_1613 { keys[6] = tmp_1554; values[6] = tmp_1555; }
    let tmp_1614 = keys[7] < tmp_1556 || (keys[7] == tmp_1556 && values[7] < tmp_1557);
    if tmp_1606 == tmp_1614 { keys[7] = tmp_1556; values[7] = tmp_1557; }
    let tmp_1615 = keys[8] < tmp_1558 || (keys[8] == tmp_1558 && values[8] < tmp_1559);
    if tmp_1606 == tmp_1615 { keys[8] = tmp_1558; values[8] = tmp_1559; }
    let tmp_1616 = keys[9] < tmp_1560 || (keys[9] == tmp_1560 && values[9] < tmp_1561);
    if tmp_1606 == tmp_1616 { keys[9] = tmp_1560; values[9] = tmp_1561; }
    let tmp_1617 = keys[10] < tmp_1562 || (keys[10] == tmp_1562 && values[10] < tmp_1563);
    if tmp_1606 == tmp_1617 { keys[10] = tmp_1562; values[10] = tmp_1563; }
    let tmp_1618 = keys[11] < tmp_1564 || (keys[11] == tmp_1564 && values[11] < tmp_1565);
    if tmp_1606 == tmp_1618 { keys[11] = tmp_1564; values[11] = tmp_1565; }
    let tmp_1619 = keys[12] < tmp_1566 || (keys[12] == tmp_1566 && values[12] < tmp_1567);
    if tmp_1606 == tmp_1619 { keys[12] = tmp_1566; values[12] = tmp_1567; }
    let tmp_1620 = keys[13] < tmp_1568 || (keys[13] == tmp_1568 && values[13] < tmp_1569);
    if tmp_1606 == tmp_1620 { keys[13] = tmp_1568; values[13] = tmp_1569; }
    let tmp_1621 = keys[14] < tmp_1570 || (keys[14] == tmp_1570 && values[14] < tmp_1571);
    if tmp_1606 == tmp_1621 { keys[14] = tmp_1570; values[14] = tmp_1571; }
    let tmp_1622 = keys[15] < tmp_1572 || (keys[15] == tmp_1572 && values[15] < tmp_1573);
    if tmp_1606 == tmp_1622 { keys[15] = tmp_1572; values[15] = tmp_1573; }
    let tmp_1623 = keys[16] < tmp_1574 || (keys[16] == tmp_1574 && values[16] < tmp_1575);
    if tmp_1606 == tmp_1623 { keys[16] = tmp_1574; values[16] = tmp_1575; }
    let tmp_1624 = keys[17] < tmp_1576 || (keys[17] == tmp_1576 && values[17] < tmp_1577);
    if tmp_1606 == tmp_1624 { keys[17] = tmp_1576; values[17] = tmp_1577; }
    let tmp_1625 = keys[18] < tmp_1578 || (keys[18] == tmp_1578 && values[18] < tmp_1579);
    if tmp_1606 == tmp_1625 { keys[18] = tmp_1578; values[18] = tmp_1579; }
    let tmp_1626 = keys[19] < tmp_1580 || (keys[19] == tmp_1580 && values[19] < tmp_1581);
    if tmp_1606 == tmp_1626 { keys[19] = tmp_1580; values[19] = tmp_1581; }
    let tmp_1627 = keys[20] < tmp_1582 || (keys[20] == tmp_1582 && values[20] < tmp_1583);
    if tmp_1606 == tmp_1627 { keys[20] = tmp_1582; values[20] = tmp_1583; }
    let tmp_1628 = keys[21] < tmp_1584 || (keys[21] == tmp_1584 && values[21] < tmp_1585);
    if tmp_1606 == tmp_1628 { keys[21] = tmp_1584; values[21] = tmp_1585; }
    let tmp_1629 = keys[22] < tmp_1586 || (keys[22] == tmp_1586 && values[22] < tmp_1587);
    if tmp_1606 == tmp_1629 { keys[22] = tmp_1586; values[22] = tmp_1587; }
    let tmp_1630 = keys[23] < tmp_1588 || (keys[23] == tmp_1588 && values[23] < tmp_1589);
    if tmp_1606 == tmp_1630 { keys[23] = tmp_1588; values[23] = tmp_1589; }
    let tmp_1631 = keys[24] < tmp_1590 || (keys[24] == tmp_1590 && values[24] < tmp_1591);
    if tmp_1606 == tmp_1631 { keys[24] = tmp_1590; values[24] = tmp_1591; }
    let tmp_1632 = keys[25] < tmp_1592 || (keys[25] == tmp_1592 && values[25] < tmp_1593);
    if tmp_1606 == tmp_1632 { keys[25] = tmp_1592; values[25] = tmp_1593; }
    let tmp_1633 = keys[26] < tmp_1594 || (keys[26] == tmp_1594 && values[26] < tmp_1595);
    if tmp_1606 == tmp_1633 { keys[26] = tmp_1594; values[26] = tmp_1595; }
    let tmp_1634 = keys[27] < tmp_1596 || (keys[27] == tmp_1596 && values[27] < tmp_1597);
    if tmp_1606 == tmp_1634 { keys[27] = tmp_1596; values[27] = tmp_1597; }
    let tmp_1635 = keys[28] < tmp_1598 || (keys[28] == tmp_1598 && values[28] < tmp_1599);
    if tmp_1606 == tmp_1635 { keys[28] = tmp_1598; values[28] = tmp_1599; }
    let tmp_1636 = keys[29] < tmp_1600 || (keys[29] == tmp_1600 && values[29] < tmp_1601);
    if tmp_1606 == tmp_1636 { keys[29] = tmp_1600; values[29] = tmp_1601; }
    let tmp_1637 = keys[30] < tmp_1602 || (keys[30] == tmp_1602 && values[30] < tmp_1603);
    if tmp_1606 == tmp_1637 { keys[30] = tmp_1602; values[30] = tmp_1603; }
    let tmp_1638 = keys[31] < tmp_1604 || (keys[31] == tmp_1604 && values[31] < tmp_1605);
    if tmp_1606 == tmp_1638 { keys[31] = tmp_1604; values[31] = tmp_1605; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:32) 
    {
    let tmp_1639 = subgroupShuffleXor(keys[0], 4u);
    let tmp_1640 = subgroupShuffleXor(values[0], 4u);
    let tmp_1641 = subgroupShuffleXor(keys[1], 4u);
    let tmp_1642 = subgroupShuffleXor(values[1], 4u);
    let tmp_1643 = subgroupShuffleXor(keys[2], 4u);
    let tmp_1644 = subgroupShuffleXor(values[2], 4u);
    let tmp_1645 = subgroupShuffleXor(keys[3], 4u);
    let tmp_1646 = subgroupShuffleXor(values[3], 4u);
    let tmp_1647 = subgroupShuffleXor(keys[4], 4u);
    let tmp_1648 = subgroupShuffleXor(values[4], 4u);
    let tmp_1649 = subgroupShuffleXor(keys[5], 4u);
    let tmp_1650 = subgroupShuffleXor(values[5], 4u);
    let tmp_1651 = subgroupShuffleXor(keys[6], 4u);
    let tmp_1652 = subgroupShuffleXor(values[6], 4u);
    let tmp_1653 = subgroupShuffleXor(keys[7], 4u);
    let tmp_1654 = subgroupShuffleXor(values[7], 4u);
    let tmp_1655 = subgroupShuffleXor(keys[8], 4u);
    let tmp_1656 = subgroupShuffleXor(values[8], 4u);
    let tmp_1657 = subgroupShuffleXor(keys[9], 4u);
    let tmp_1658 = subgroupShuffleXor(values[9], 4u);
    let tmp_1659 = subgroupShuffleXor(keys[10], 4u);
    let tmp_1660 = subgroupShuffleXor(values[10], 4u);
    let tmp_1661 = subgroupShuffleXor(keys[11], 4u);
    let tmp_1662 = subgroupShuffleXor(values[11], 4u);
    let tmp_1663 = subgroupShuffleXor(keys[12], 4u);
    let tmp_1664 = subgroupShuffleXor(values[12], 4u);
    let tmp_1665 = subgroupShuffleXor(keys[13], 4u);
    let tmp_1666 = subgroupShuffleXor(values[13], 4u);
    let tmp_1667 = subgroupShuffleXor(keys[14], 4u);
    let tmp_1668 = subgroupShuffleXor(values[14], 4u);
    let tmp_1669 = subgroupShuffleXor(keys[15], 4u);
    let tmp_1670 = subgroupShuffleXor(values[15], 4u);
    let tmp_1671 = subgroupShuffleXor(keys[16], 4u);
    let tmp_1672 = subgroupShuffleXor(values[16], 4u);
    let tmp_1673 = subgroupShuffleXor(keys[17], 4u);
    let tmp_1674 = subgroupShuffleXor(values[17], 4u);
    let tmp_1675 = subgroupShuffleXor(keys[18], 4u);
    let tmp_1676 = subgroupShuffleXor(values[18], 4u);
    let tmp_1677 = subgroupShuffleXor(keys[19], 4u);
    let tmp_1678 = subgroupShuffleXor(values[19], 4u);
    let tmp_1679 = subgroupShuffleXor(keys[20], 4u);
    let tmp_1680 = subgroupShuffleXor(values[20], 4u);
    let tmp_1681 = subgroupShuffleXor(keys[21], 4u);
    let tmp_1682 = subgroupShuffleXor(values[21], 4u);
    let tmp_1683 = subgroupShuffleXor(keys[22], 4u);
    let tmp_1684 = subgroupShuffleXor(values[22], 4u);
    let tmp_1685 = subgroupShuffleXor(keys[23], 4u);
    let tmp_1686 = subgroupShuffleXor(values[23], 4u);
    let tmp_1687 = subgroupShuffleXor(keys[24], 4u);
    let tmp_1688 = subgroupShuffleXor(values[24], 4u);
    let tmp_1689 = subgroupShuffleXor(keys[25], 4u);
    let tmp_1690 = subgroupShuffleXor(values[25], 4u);
    let tmp_1691 = subgroupShuffleXor(keys[26], 4u);
    let tmp_1692 = subgroupShuffleXor(values[26], 4u);
    let tmp_1693 = subgroupShuffleXor(keys[27], 4u);
    let tmp_1694 = subgroupShuffleXor(values[27], 4u);
    let tmp_1695 = subgroupShuffleXor(keys[28], 4u);
    let tmp_1696 = subgroupShuffleXor(values[28], 4u);
    let tmp_1697 = subgroupShuffleXor(keys[29], 4u);
    let tmp_1698 = subgroupShuffleXor(values[29], 4u);
    let tmp_1699 = subgroupShuffleXor(keys[30], 4u);
    let tmp_1700 = subgroupShuffleXor(values[30], 4u);
    let tmp_1701 = subgroupShuffleXor(keys[31], 4u);
    let tmp_1702 = subgroupShuffleXor(values[31], 4u);
    let tmp_1703 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_1704 = keys[0] < tmp_1639 || (keys[0] == tmp_1639 && values[0] < tmp_1640);
    if tmp_1703 == tmp_1704 { keys[0] = tmp_1639; values[0] = tmp_1640; }
    let tmp_1705 = keys[1] < tmp_1641 || (keys[1] == tmp_1641 && values[1] < tmp_1642);
    if tmp_1703 == tmp_1705 { keys[1] = tmp_1641; values[1] = tmp_1642; }
    let tmp_1706 = keys[2] < tmp_1643 || (keys[2] == tmp_1643 && values[2] < tmp_1644);
    if tmp_1703 == tmp_1706 { keys[2] = tmp_1643; values[2] = tmp_1644; }
    let tmp_1707 = keys[3] < tmp_1645 || (keys[3] == tmp_1645 && values[3] < tmp_1646);
    if tmp_1703 == tmp_1707 { keys[3] = tmp_1645; values[3] = tmp_1646; }
    let tmp_1708 = keys[4] < tmp_1647 || (keys[4] == tmp_1647 && values[4] < tmp_1648);
    if tmp_1703 == tmp_1708 { keys[4] = tmp_1647; values[4] = tmp_1648; }
    let tmp_1709 = keys[5] < tmp_1649 || (keys[5] == tmp_1649 && values[5] < tmp_1650);
    if tmp_1703 == tmp_1709 { keys[5] = tmp_1649; values[5] = tmp_1650; }
    let tmp_1710 = keys[6] < tmp_1651 || (keys[6] == tmp_1651 && values[6] < tmp_1652);
    if tmp_1703 == tmp_1710 { keys[6] = tmp_1651; values[6] = tmp_1652; }
    let tmp_1711 = keys[7] < tmp_1653 || (keys[7] == tmp_1653 && values[7] < tmp_1654);
    if tmp_1703 == tmp_1711 { keys[7] = tmp_1653; values[7] = tmp_1654; }
    let tmp_1712 = keys[8] < tmp_1655 || (keys[8] == tmp_1655 && values[8] < tmp_1656);
    if tmp_1703 == tmp_1712 { keys[8] = tmp_1655; values[8] = tmp_1656; }
    let tmp_1713 = keys[9] < tmp_1657 || (keys[9] == tmp_1657 && values[9] < tmp_1658);
    if tmp_1703 == tmp_1713 { keys[9] = tmp_1657; values[9] = tmp_1658; }
    let tmp_1714 = keys[10] < tmp_1659 || (keys[10] == tmp_1659 && values[10] < tmp_1660);
    if tmp_1703 == tmp_1714 { keys[10] = tmp_1659; values[10] = tmp_1660; }
    let tmp_1715 = keys[11] < tmp_1661 || (keys[11] == tmp_1661 && values[11] < tmp_1662);
    if tmp_1703 == tmp_1715 { keys[11] = tmp_1661; values[11] = tmp_1662; }
    let tmp_1716 = keys[12] < tmp_1663 || (keys[12] == tmp_1663 && values[12] < tmp_1664);
    if tmp_1703 == tmp_1716 { keys[12] = tmp_1663; values[12] = tmp_1664; }
    let tmp_1717 = keys[13] < tmp_1665 || (keys[13] == tmp_1665 && values[13] < tmp_1666);
    if tmp_1703 == tmp_1717 { keys[13] = tmp_1665; values[13] = tmp_1666; }
    let tmp_1718 = keys[14] < tmp_1667 || (keys[14] == tmp_1667 && values[14] < tmp_1668);
    if tmp_1703 == tmp_1718 { keys[14] = tmp_1667; values[14] = tmp_1668; }
    let tmp_1719 = keys[15] < tmp_1669 || (keys[15] == tmp_1669 && values[15] < tmp_1670);
    if tmp_1703 == tmp_1719 { keys[15] = tmp_1669; values[15] = tmp_1670; }
    let tmp_1720 = keys[16] < tmp_1671 || (keys[16] == tmp_1671 && values[16] < tmp_1672);
    if tmp_1703 == tmp_1720 { keys[16] = tmp_1671; values[16] = tmp_1672; }
    let tmp_1721 = keys[17] < tmp_1673 || (keys[17] == tmp_1673 && values[17] < tmp_1674);
    if tmp_1703 == tmp_1721 { keys[17] = tmp_1673; values[17] = tmp_1674; }
    let tmp_1722 = keys[18] < tmp_1675 || (keys[18] == tmp_1675 && values[18] < tmp_1676);
    if tmp_1703 == tmp_1722 { keys[18] = tmp_1675; values[18] = tmp_1676; }
    let tmp_1723 = keys[19] < tmp_1677 || (keys[19] == tmp_1677 && values[19] < tmp_1678);
    if tmp_1703 == tmp_1723 { keys[19] = tmp_1677; values[19] = tmp_1678; }
    let tmp_1724 = keys[20] < tmp_1679 || (keys[20] == tmp_1679 && values[20] < tmp_1680);
    if tmp_1703 == tmp_1724 { keys[20] = tmp_1679; values[20] = tmp_1680; }
    let tmp_1725 = keys[21] < tmp_1681 || (keys[21] == tmp_1681 && values[21] < tmp_1682);
    if tmp_1703 == tmp_1725 { keys[21] = tmp_1681; values[21] = tmp_1682; }
    let tmp_1726 = keys[22] < tmp_1683 || (keys[22] == tmp_1683 && values[22] < tmp_1684);
    if tmp_1703 == tmp_1726 { keys[22] = tmp_1683; values[22] = tmp_1684; }
    let tmp_1727 = keys[23] < tmp_1685 || (keys[23] == tmp_1685 && values[23] < tmp_1686);
    if tmp_1703 == tmp_1727 { keys[23] = tmp_1685; values[23] = tmp_1686; }
    let tmp_1728 = keys[24] < tmp_1687 || (keys[24] == tmp_1687 && values[24] < tmp_1688);
    if tmp_1703 == tmp_1728 { keys[24] = tmp_1687; values[24] = tmp_1688; }
    let tmp_1729 = keys[25] < tmp_1689 || (keys[25] == tmp_1689 && values[25] < tmp_1690);
    if tmp_1703 == tmp_1729 { keys[25] = tmp_1689; values[25] = tmp_1690; }
    let tmp_1730 = keys[26] < tmp_1691 || (keys[26] == tmp_1691 && values[26] < tmp_1692);
    if tmp_1703 == tmp_1730 { keys[26] = tmp_1691; values[26] = tmp_1692; }
    let tmp_1731 = keys[27] < tmp_1693 || (keys[27] == tmp_1693 && values[27] < tmp_1694);
    if tmp_1703 == tmp_1731 { keys[27] = tmp_1693; values[27] = tmp_1694; }
    let tmp_1732 = keys[28] < tmp_1695 || (keys[28] == tmp_1695 && values[28] < tmp_1696);
    if tmp_1703 == tmp_1732 { keys[28] = tmp_1695; values[28] = tmp_1696; }
    let tmp_1733 = keys[29] < tmp_1697 || (keys[29] == tmp_1697 && values[29] < tmp_1698);
    if tmp_1703 == tmp_1733 { keys[29] = tmp_1697; values[29] = tmp_1698; }
    let tmp_1734 = keys[30] < tmp_1699 || (keys[30] == tmp_1699 && values[30] < tmp_1700);
    if tmp_1703 == tmp_1734 { keys[30] = tmp_1699; values[30] = tmp_1700; }
    let tmp_1735 = keys[31] < tmp_1701 || (keys[31] == tmp_1701 && values[31] < tmp_1702);
    if tmp_1703 == tmp_1735 { keys[31] = tmp_1701; values[31] = tmp_1702; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:32) 
    {
    let tmp_1736 = subgroupShuffleXor(keys[0], 2u);
    let tmp_1737 = subgroupShuffleXor(values[0], 2u);
    let tmp_1738 = subgroupShuffleXor(keys[1], 2u);
    let tmp_1739 = subgroupShuffleXor(values[1], 2u);
    let tmp_1740 = subgroupShuffleXor(keys[2], 2u);
    let tmp_1741 = subgroupShuffleXor(values[2], 2u);
    let tmp_1742 = subgroupShuffleXor(keys[3], 2u);
    let tmp_1743 = subgroupShuffleXor(values[3], 2u);
    let tmp_1744 = subgroupShuffleXor(keys[4], 2u);
    let tmp_1745 = subgroupShuffleXor(values[4], 2u);
    let tmp_1746 = subgroupShuffleXor(keys[5], 2u);
    let tmp_1747 = subgroupShuffleXor(values[5], 2u);
    let tmp_1748 = subgroupShuffleXor(keys[6], 2u);
    let tmp_1749 = subgroupShuffleXor(values[6], 2u);
    let tmp_1750 = subgroupShuffleXor(keys[7], 2u);
    let tmp_1751 = subgroupShuffleXor(values[7], 2u);
    let tmp_1752 = subgroupShuffleXor(keys[8], 2u);
    let tmp_1753 = subgroupShuffleXor(values[8], 2u);
    let tmp_1754 = subgroupShuffleXor(keys[9], 2u);
    let tmp_1755 = subgroupShuffleXor(values[9], 2u);
    let tmp_1756 = subgroupShuffleXor(keys[10], 2u);
    let tmp_1757 = subgroupShuffleXor(values[10], 2u);
    let tmp_1758 = subgroupShuffleXor(keys[11], 2u);
    let tmp_1759 = subgroupShuffleXor(values[11], 2u);
    let tmp_1760 = subgroupShuffleXor(keys[12], 2u);
    let tmp_1761 = subgroupShuffleXor(values[12], 2u);
    let tmp_1762 = subgroupShuffleXor(keys[13], 2u);
    let tmp_1763 = subgroupShuffleXor(values[13], 2u);
    let tmp_1764 = subgroupShuffleXor(keys[14], 2u);
    let tmp_1765 = subgroupShuffleXor(values[14], 2u);
    let tmp_1766 = subgroupShuffleXor(keys[15], 2u);
    let tmp_1767 = subgroupShuffleXor(values[15], 2u);
    let tmp_1768 = subgroupShuffleXor(keys[16], 2u);
    let tmp_1769 = subgroupShuffleXor(values[16], 2u);
    let tmp_1770 = subgroupShuffleXor(keys[17], 2u);
    let tmp_1771 = subgroupShuffleXor(values[17], 2u);
    let tmp_1772 = subgroupShuffleXor(keys[18], 2u);
    let tmp_1773 = subgroupShuffleXor(values[18], 2u);
    let tmp_1774 = subgroupShuffleXor(keys[19], 2u);
    let tmp_1775 = subgroupShuffleXor(values[19], 2u);
    let tmp_1776 = subgroupShuffleXor(keys[20], 2u);
    let tmp_1777 = subgroupShuffleXor(values[20], 2u);
    let tmp_1778 = subgroupShuffleXor(keys[21], 2u);
    let tmp_1779 = subgroupShuffleXor(values[21], 2u);
    let tmp_1780 = subgroupShuffleXor(keys[22], 2u);
    let tmp_1781 = subgroupShuffleXor(values[22], 2u);
    let tmp_1782 = subgroupShuffleXor(keys[23], 2u);
    let tmp_1783 = subgroupShuffleXor(values[23], 2u);
    let tmp_1784 = subgroupShuffleXor(keys[24], 2u);
    let tmp_1785 = subgroupShuffleXor(values[24], 2u);
    let tmp_1786 = subgroupShuffleXor(keys[25], 2u);
    let tmp_1787 = subgroupShuffleXor(values[25], 2u);
    let tmp_1788 = subgroupShuffleXor(keys[26], 2u);
    let tmp_1789 = subgroupShuffleXor(values[26], 2u);
    let tmp_1790 = subgroupShuffleXor(keys[27], 2u);
    let tmp_1791 = subgroupShuffleXor(values[27], 2u);
    let tmp_1792 = subgroupShuffleXor(keys[28], 2u);
    let tmp_1793 = subgroupShuffleXor(values[28], 2u);
    let tmp_1794 = subgroupShuffleXor(keys[29], 2u);
    let tmp_1795 = subgroupShuffleXor(values[29], 2u);
    let tmp_1796 = subgroupShuffleXor(keys[30], 2u);
    let tmp_1797 = subgroupShuffleXor(values[30], 2u);
    let tmp_1798 = subgroupShuffleXor(keys[31], 2u);
    let tmp_1799 = subgroupShuffleXor(values[31], 2u);
    let tmp_1800 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_1801 = keys[0] < tmp_1736 || (keys[0] == tmp_1736 && values[0] < tmp_1737);
    if tmp_1800 == tmp_1801 { keys[0] = tmp_1736; values[0] = tmp_1737; }
    let tmp_1802 = keys[1] < tmp_1738 || (keys[1] == tmp_1738 && values[1] < tmp_1739);
    if tmp_1800 == tmp_1802 { keys[1] = tmp_1738; values[1] = tmp_1739; }
    let tmp_1803 = keys[2] < tmp_1740 || (keys[2] == tmp_1740 && values[2] < tmp_1741);
    if tmp_1800 == tmp_1803 { keys[2] = tmp_1740; values[2] = tmp_1741; }
    let tmp_1804 = keys[3] < tmp_1742 || (keys[3] == tmp_1742 && values[3] < tmp_1743);
    if tmp_1800 == tmp_1804 { keys[3] = tmp_1742; values[3] = tmp_1743; }
    let tmp_1805 = keys[4] < tmp_1744 || (keys[4] == tmp_1744 && values[4] < tmp_1745);
    if tmp_1800 == tmp_1805 { keys[4] = tmp_1744; values[4] = tmp_1745; }
    let tmp_1806 = keys[5] < tmp_1746 || (keys[5] == tmp_1746 && values[5] < tmp_1747);
    if tmp_1800 == tmp_1806 { keys[5] = tmp_1746; values[5] = tmp_1747; }
    let tmp_1807 = keys[6] < tmp_1748 || (keys[6] == tmp_1748 && values[6] < tmp_1749);
    if tmp_1800 == tmp_1807 { keys[6] = tmp_1748; values[6] = tmp_1749; }
    let tmp_1808 = keys[7] < tmp_1750 || (keys[7] == tmp_1750 && values[7] < tmp_1751);
    if tmp_1800 == tmp_1808 { keys[7] = tmp_1750; values[7] = tmp_1751; }
    let tmp_1809 = keys[8] < tmp_1752 || (keys[8] == tmp_1752 && values[8] < tmp_1753);
    if tmp_1800 == tmp_1809 { keys[8] = tmp_1752; values[8] = tmp_1753; }
    let tmp_1810 = keys[9] < tmp_1754 || (keys[9] == tmp_1754 && values[9] < tmp_1755);
    if tmp_1800 == tmp_1810 { keys[9] = tmp_1754; values[9] = tmp_1755; }
    let tmp_1811 = keys[10] < tmp_1756 || (keys[10] == tmp_1756 && values[10] < tmp_1757);
    if tmp_1800 == tmp_1811 { keys[10] = tmp_1756; values[10] = tmp_1757; }
    let tmp_1812 = keys[11] < tmp_1758 || (keys[11] == tmp_1758 && values[11] < tmp_1759);
    if tmp_1800 == tmp_1812 { keys[11] = tmp_1758; values[11] = tmp_1759; }
    let tmp_1813 = keys[12] < tmp_1760 || (keys[12] == tmp_1760 && values[12] < tmp_1761);
    if tmp_1800 == tmp_1813 { keys[12] = tmp_1760; values[12] = tmp_1761; }
    let tmp_1814 = keys[13] < tmp_1762 || (keys[13] == tmp_1762 && values[13] < tmp_1763);
    if tmp_1800 == tmp_1814 { keys[13] = tmp_1762; values[13] = tmp_1763; }
    let tmp_1815 = keys[14] < tmp_1764 || (keys[14] == tmp_1764 && values[14] < tmp_1765);
    if tmp_1800 == tmp_1815 { keys[14] = tmp_1764; values[14] = tmp_1765; }
    let tmp_1816 = keys[15] < tmp_1766 || (keys[15] == tmp_1766 && values[15] < tmp_1767);
    if tmp_1800 == tmp_1816 { keys[15] = tmp_1766; values[15] = tmp_1767; }
    let tmp_1817 = keys[16] < tmp_1768 || (keys[16] == tmp_1768 && values[16] < tmp_1769);
    if tmp_1800 == tmp_1817 { keys[16] = tmp_1768; values[16] = tmp_1769; }
    let tmp_1818 = keys[17] < tmp_1770 || (keys[17] == tmp_1770 && values[17] < tmp_1771);
    if tmp_1800 == tmp_1818 { keys[17] = tmp_1770; values[17] = tmp_1771; }
    let tmp_1819 = keys[18] < tmp_1772 || (keys[18] == tmp_1772 && values[18] < tmp_1773);
    if tmp_1800 == tmp_1819 { keys[18] = tmp_1772; values[18] = tmp_1773; }
    let tmp_1820 = keys[19] < tmp_1774 || (keys[19] == tmp_1774 && values[19] < tmp_1775);
    if tmp_1800 == tmp_1820 { keys[19] = tmp_1774; values[19] = tmp_1775; }
    let tmp_1821 = keys[20] < tmp_1776 || (keys[20] == tmp_1776 && values[20] < tmp_1777);
    if tmp_1800 == tmp_1821 { keys[20] = tmp_1776; values[20] = tmp_1777; }
    let tmp_1822 = keys[21] < tmp_1778 || (keys[21] == tmp_1778 && values[21] < tmp_1779);
    if tmp_1800 == tmp_1822 { keys[21] = tmp_1778; values[21] = tmp_1779; }
    let tmp_1823 = keys[22] < tmp_1780 || (keys[22] == tmp_1780 && values[22] < tmp_1781);
    if tmp_1800 == tmp_1823 { keys[22] = tmp_1780; values[22] = tmp_1781; }
    let tmp_1824 = keys[23] < tmp_1782 || (keys[23] == tmp_1782 && values[23] < tmp_1783);
    if tmp_1800 == tmp_1824 { keys[23] = tmp_1782; values[23] = tmp_1783; }
    let tmp_1825 = keys[24] < tmp_1784 || (keys[24] == tmp_1784 && values[24] < tmp_1785);
    if tmp_1800 == tmp_1825 { keys[24] = tmp_1784; values[24] = tmp_1785; }
    let tmp_1826 = keys[25] < tmp_1786 || (keys[25] == tmp_1786 && values[25] < tmp_1787);
    if tmp_1800 == tmp_1826 { keys[25] = tmp_1786; values[25] = tmp_1787; }
    let tmp_1827 = keys[26] < tmp_1788 || (keys[26] == tmp_1788 && values[26] < tmp_1789);
    if tmp_1800 == tmp_1827 { keys[26] = tmp_1788; values[26] = tmp_1789; }
    let tmp_1828 = keys[27] < tmp_1790 || (keys[27] == tmp_1790 && values[27] < tmp_1791);
    if tmp_1800 == tmp_1828 { keys[27] = tmp_1790; values[27] = tmp_1791; }
    let tmp_1829 = keys[28] < tmp_1792 || (keys[28] == tmp_1792 && values[28] < tmp_1793);
    if tmp_1800 == tmp_1829 { keys[28] = tmp_1792; values[28] = tmp_1793; }
    let tmp_1830 = keys[29] < tmp_1794 || (keys[29] == tmp_1794 && values[29] < tmp_1795);
    if tmp_1800 == tmp_1830 { keys[29] = tmp_1794; values[29] = tmp_1795; }
    let tmp_1831 = keys[30] < tmp_1796 || (keys[30] == tmp_1796 && values[30] < tmp_1797);
    if tmp_1800 == tmp_1831 { keys[30] = tmp_1796; values[30] = tmp_1797; }
    let tmp_1832 = keys[31] < tmp_1798 || (keys[31] == tmp_1798 && values[31] < tmp_1799);
    if tmp_1800 == tmp_1832 { keys[31] = tmp_1798; values[31] = tmp_1799; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:32) 
    {
    let tmp_1833 = subgroupShuffleXor(keys[0], 1u);
    let tmp_1834 = subgroupShuffleXor(values[0], 1u);
    let tmp_1835 = subgroupShuffleXor(keys[1], 1u);
    let tmp_1836 = subgroupShuffleXor(values[1], 1u);
    let tmp_1837 = subgroupShuffleXor(keys[2], 1u);
    let tmp_1838 = subgroupShuffleXor(values[2], 1u);
    let tmp_1839 = subgroupShuffleXor(keys[3], 1u);
    let tmp_1840 = subgroupShuffleXor(values[3], 1u);
    let tmp_1841 = subgroupShuffleXor(keys[4], 1u);
    let tmp_1842 = subgroupShuffleXor(values[4], 1u);
    let tmp_1843 = subgroupShuffleXor(keys[5], 1u);
    let tmp_1844 = subgroupShuffleXor(values[5], 1u);
    let tmp_1845 = subgroupShuffleXor(keys[6], 1u);
    let tmp_1846 = subgroupShuffleXor(values[6], 1u);
    let tmp_1847 = subgroupShuffleXor(keys[7], 1u);
    let tmp_1848 = subgroupShuffleXor(values[7], 1u);
    let tmp_1849 = subgroupShuffleXor(keys[8], 1u);
    let tmp_1850 = subgroupShuffleXor(values[8], 1u);
    let tmp_1851 = subgroupShuffleXor(keys[9], 1u);
    let tmp_1852 = subgroupShuffleXor(values[9], 1u);
    let tmp_1853 = subgroupShuffleXor(keys[10], 1u);
    let tmp_1854 = subgroupShuffleXor(values[10], 1u);
    let tmp_1855 = subgroupShuffleXor(keys[11], 1u);
    let tmp_1856 = subgroupShuffleXor(values[11], 1u);
    let tmp_1857 = subgroupShuffleXor(keys[12], 1u);
    let tmp_1858 = subgroupShuffleXor(values[12], 1u);
    let tmp_1859 = subgroupShuffleXor(keys[13], 1u);
    let tmp_1860 = subgroupShuffleXor(values[13], 1u);
    let tmp_1861 = subgroupShuffleXor(keys[14], 1u);
    let tmp_1862 = subgroupShuffleXor(values[14], 1u);
    let tmp_1863 = subgroupShuffleXor(keys[15], 1u);
    let tmp_1864 = subgroupShuffleXor(values[15], 1u);
    let tmp_1865 = subgroupShuffleXor(keys[16], 1u);
    let tmp_1866 = subgroupShuffleXor(values[16], 1u);
    let tmp_1867 = subgroupShuffleXor(keys[17], 1u);
    let tmp_1868 = subgroupShuffleXor(values[17], 1u);
    let tmp_1869 = subgroupShuffleXor(keys[18], 1u);
    let tmp_1870 = subgroupShuffleXor(values[18], 1u);
    let tmp_1871 = subgroupShuffleXor(keys[19], 1u);
    let tmp_1872 = subgroupShuffleXor(values[19], 1u);
    let tmp_1873 = subgroupShuffleXor(keys[20], 1u);
    let tmp_1874 = subgroupShuffleXor(values[20], 1u);
    let tmp_1875 = subgroupShuffleXor(keys[21], 1u);
    let tmp_1876 = subgroupShuffleXor(values[21], 1u);
    let tmp_1877 = subgroupShuffleXor(keys[22], 1u);
    let tmp_1878 = subgroupShuffleXor(values[22], 1u);
    let tmp_1879 = subgroupShuffleXor(keys[23], 1u);
    let tmp_1880 = subgroupShuffleXor(values[23], 1u);
    let tmp_1881 = subgroupShuffleXor(keys[24], 1u);
    let tmp_1882 = subgroupShuffleXor(values[24], 1u);
    let tmp_1883 = subgroupShuffleXor(keys[25], 1u);
    let tmp_1884 = subgroupShuffleXor(values[25], 1u);
    let tmp_1885 = subgroupShuffleXor(keys[26], 1u);
    let tmp_1886 = subgroupShuffleXor(values[26], 1u);
    let tmp_1887 = subgroupShuffleXor(keys[27], 1u);
    let tmp_1888 = subgroupShuffleXor(values[27], 1u);
    let tmp_1889 = subgroupShuffleXor(keys[28], 1u);
    let tmp_1890 = subgroupShuffleXor(values[28], 1u);
    let tmp_1891 = subgroupShuffleXor(keys[29], 1u);
    let tmp_1892 = subgroupShuffleXor(values[29], 1u);
    let tmp_1893 = subgroupShuffleXor(keys[30], 1u);
    let tmp_1894 = subgroupShuffleXor(values[30], 1u);
    let tmp_1895 = subgroupShuffleXor(keys[31], 1u);
    let tmp_1896 = subgroupShuffleXor(values[31], 1u);
    let tmp_1897 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_1898 = keys[0] < tmp_1833 || (keys[0] == tmp_1833 && values[0] < tmp_1834);
    if tmp_1897 == tmp_1898 { keys[0] = tmp_1833; values[0] = tmp_1834; }
    let tmp_1899 = keys[1] < tmp_1835 || (keys[1] == tmp_1835 && values[1] < tmp_1836);
    if tmp_1897 == tmp_1899 { keys[1] = tmp_1835; values[1] = tmp_1836; }
    let tmp_1900 = keys[2] < tmp_1837 || (keys[2] == tmp_1837 && values[2] < tmp_1838);
    if tmp_1897 == tmp_1900 { keys[2] = tmp_1837; values[2] = tmp_1838; }
    let tmp_1901 = keys[3] < tmp_1839 || (keys[3] == tmp_1839 && values[3] < tmp_1840);
    if tmp_1897 == tmp_1901 { keys[3] = tmp_1839; values[3] = tmp_1840; }
    let tmp_1902 = keys[4] < tmp_1841 || (keys[4] == tmp_1841 && values[4] < tmp_1842);
    if tmp_1897 == tmp_1902 { keys[4] = tmp_1841; values[4] = tmp_1842; }
    let tmp_1903 = keys[5] < tmp_1843 || (keys[5] == tmp_1843 && values[5] < tmp_1844);
    if tmp_1897 == tmp_1903 { keys[5] = tmp_1843; values[5] = tmp_1844; }
    let tmp_1904 = keys[6] < tmp_1845 || (keys[6] == tmp_1845 && values[6] < tmp_1846);
    if tmp_1897 == tmp_1904 { keys[6] = tmp_1845; values[6] = tmp_1846; }
    let tmp_1905 = keys[7] < tmp_1847 || (keys[7] == tmp_1847 && values[7] < tmp_1848);
    if tmp_1897 == tmp_1905 { keys[7] = tmp_1847; values[7] = tmp_1848; }
    let tmp_1906 = keys[8] < tmp_1849 || (keys[8] == tmp_1849 && values[8] < tmp_1850);
    if tmp_1897 == tmp_1906 { keys[8] = tmp_1849; values[8] = tmp_1850; }
    let tmp_1907 = keys[9] < tmp_1851 || (keys[9] == tmp_1851 && values[9] < tmp_1852);
    if tmp_1897 == tmp_1907 { keys[9] = tmp_1851; values[9] = tmp_1852; }
    let tmp_1908 = keys[10] < tmp_1853 || (keys[10] == tmp_1853 && values[10] < tmp_1854);
    if tmp_1897 == tmp_1908 { keys[10] = tmp_1853; values[10] = tmp_1854; }
    let tmp_1909 = keys[11] < tmp_1855 || (keys[11] == tmp_1855 && values[11] < tmp_1856);
    if tmp_1897 == tmp_1909 { keys[11] = tmp_1855; values[11] = tmp_1856; }
    let tmp_1910 = keys[12] < tmp_1857 || (keys[12] == tmp_1857 && values[12] < tmp_1858);
    if tmp_1897 == tmp_1910 { keys[12] = tmp_1857; values[12] = tmp_1858; }
    let tmp_1911 = keys[13] < tmp_1859 || (keys[13] == tmp_1859 && values[13] < tmp_1860);
    if tmp_1897 == tmp_1911 { keys[13] = tmp_1859; values[13] = tmp_1860; }
    let tmp_1912 = keys[14] < tmp_1861 || (keys[14] == tmp_1861 && values[14] < tmp_1862);
    if tmp_1897 == tmp_1912 { keys[14] = tmp_1861; values[14] = tmp_1862; }
    let tmp_1913 = keys[15] < tmp_1863 || (keys[15] == tmp_1863 && values[15] < tmp_1864);
    if tmp_1897 == tmp_1913 { keys[15] = tmp_1863; values[15] = tmp_1864; }
    let tmp_1914 = keys[16] < tmp_1865 || (keys[16] == tmp_1865 && values[16] < tmp_1866);
    if tmp_1897 == tmp_1914 { keys[16] = tmp_1865; values[16] = tmp_1866; }
    let tmp_1915 = keys[17] < tmp_1867 || (keys[17] == tmp_1867 && values[17] < tmp_1868);
    if tmp_1897 == tmp_1915 { keys[17] = tmp_1867; values[17] = tmp_1868; }
    let tmp_1916 = keys[18] < tmp_1869 || (keys[18] == tmp_1869 && values[18] < tmp_1870);
    if tmp_1897 == tmp_1916 { keys[18] = tmp_1869; values[18] = tmp_1870; }
    let tmp_1917 = keys[19] < tmp_1871 || (keys[19] == tmp_1871 && values[19] < tmp_1872);
    if tmp_1897 == tmp_1917 { keys[19] = tmp_1871; values[19] = tmp_1872; }
    let tmp_1918 = keys[20] < tmp_1873 || (keys[20] == tmp_1873 && values[20] < tmp_1874);
    if tmp_1897 == tmp_1918 { keys[20] = tmp_1873; values[20] = tmp_1874; }
    let tmp_1919 = keys[21] < tmp_1875 || (keys[21] == tmp_1875 && values[21] < tmp_1876);
    if tmp_1897 == tmp_1919 { keys[21] = tmp_1875; values[21] = tmp_1876; }
    let tmp_1920 = keys[22] < tmp_1877 || (keys[22] == tmp_1877 && values[22] < tmp_1878);
    if tmp_1897 == tmp_1920 { keys[22] = tmp_1877; values[22] = tmp_1878; }
    let tmp_1921 = keys[23] < tmp_1879 || (keys[23] == tmp_1879 && values[23] < tmp_1880);
    if tmp_1897 == tmp_1921 { keys[23] = tmp_1879; values[23] = tmp_1880; }
    let tmp_1922 = keys[24] < tmp_1881 || (keys[24] == tmp_1881 && values[24] < tmp_1882);
    if tmp_1897 == tmp_1922 { keys[24] = tmp_1881; values[24] = tmp_1882; }
    let tmp_1923 = keys[25] < tmp_1883 || (keys[25] == tmp_1883 && values[25] < tmp_1884);
    if tmp_1897 == tmp_1923 { keys[25] = tmp_1883; values[25] = tmp_1884; }
    let tmp_1924 = keys[26] < tmp_1885 || (keys[26] == tmp_1885 && values[26] < tmp_1886);
    if tmp_1897 == tmp_1924 { keys[26] = tmp_1885; values[26] = tmp_1886; }
    let tmp_1925 = keys[27] < tmp_1887 || (keys[27] == tmp_1887 && values[27] < tmp_1888);
    if tmp_1897 == tmp_1925 { keys[27] = tmp_1887; values[27] = tmp_1888; }
    let tmp_1926 = keys[28] < tmp_1889 || (keys[28] == tmp_1889 && values[28] < tmp_1890);
    if tmp_1897 == tmp_1926 { keys[28] = tmp_1889; values[28] = tmp_1890; }
    let tmp_1927 = keys[29] < tmp_1891 || (keys[29] == tmp_1891 && values[29] < tmp_1892);
    if tmp_1897 == tmp_1927 { keys[29] = tmp_1891; values[29] = tmp_1892; }
    let tmp_1928 = keys[30] < tmp_1893 || (keys[30] == tmp_1893 && values[30] < tmp_1894);
    if tmp_1897 == tmp_1928 { keys[30] = tmp_1893; values[30] = tmp_1894; }
    let tmp_1929 = keys[31] < tmp_1895 || (keys[31] == tmp_1895 && values[31] < tmp_1896);
    if tmp_1897 == tmp_1929 { keys[31] = tmp_1895; values[31] = tmp_1896; }
    }
    // exch_local(16,32) 
    // cmp_swap(0,16)
    if keys[0] > keys[16] || (keys[0] == keys[16] && values[0] > values[16]) {
    // swap(0,16) 
    { let tmp_1930 = keys[0]; keys[0] = keys[16]; keys[16] = tmp_1930;let tmp_1931 = values[0]; values[0] = values[16]; values[16] = tmp_1931; }
    }
    // cmp_swap(1,17)
    if keys[1] > keys[17] || (keys[1] == keys[17] && values[1] > values[17]) {
    // swap(1,17) 
    { let tmp_1932 = keys[1]; keys[1] = keys[17]; keys[17] = tmp_1932;let tmp_1933 = values[1]; values[1] = values[17]; values[17] = tmp_1933; }
    }
    // cmp_swap(2,18)
    if keys[2] > keys[18] || (keys[2] == keys[18] && values[2] > values[18]) {
    // swap(2,18) 
    { let tmp_1934 = keys[2]; keys[2] = keys[18]; keys[18] = tmp_1934;let tmp_1935 = values[2]; values[2] = values[18]; values[18] = tmp_1935; }
    }
    // cmp_swap(3,19)
    if keys[3] > keys[19] || (keys[3] == keys[19] && values[3] > values[19]) {
    // swap(3,19) 
    { let tmp_1936 = keys[3]; keys[3] = keys[19]; keys[19] = tmp_1936;let tmp_1937 = values[3]; values[3] = values[19]; values[19] = tmp_1937; }
    }
    // cmp_swap(4,20)
    if keys[4] > keys[20] || (keys[4] == keys[20] && values[4] > values[20]) {
    // swap(4,20) 
    { let tmp_1938 = keys[4]; keys[4] = keys[20]; keys[20] = tmp_1938;let tmp_1939 = values[4]; values[4] = values[20]; values[20] = tmp_1939; }
    }
    // cmp_swap(5,21)
    if keys[5] > keys[21] || (keys[5] == keys[21] && values[5] > values[21]) {
    // swap(5,21) 
    { let tmp_1940 = keys[5]; keys[5] = keys[21]; keys[21] = tmp_1940;let tmp_1941 = values[5]; values[5] = values[21]; values[21] = tmp_1941; }
    }
    // cmp_swap(6,22)
    if keys[6] > keys[22] || (keys[6] == keys[22] && values[6] > values[22]) {
    // swap(6,22) 
    { let tmp_1942 = keys[6]; keys[6] = keys[22]; keys[22] = tmp_1942;let tmp_1943 = values[6]; values[6] = values[22]; values[22] = tmp_1943; }
    }
    // cmp_swap(7,23)
    if keys[7] > keys[23] || (keys[7] == keys[23] && values[7] > values[23]) {
    // swap(7,23) 
    { let tmp_1944 = keys[7]; keys[7] = keys[23]; keys[23] = tmp_1944;let tmp_1945 = values[7]; values[7] = values[23]; values[23] = tmp_1945; }
    }
    // cmp_swap(8,24)
    if keys[8] > keys[24] || (keys[8] == keys[24] && values[8] > values[24]) {
    // swap(8,24) 
    { let tmp_1946 = keys[8]; keys[8] = keys[24]; keys[24] = tmp_1946;let tmp_1947 = values[8]; values[8] = values[24]; values[24] = tmp_1947; }
    }
    // cmp_swap(9,25)
    if keys[9] > keys[25] || (keys[9] == keys[25] && values[9] > values[25]) {
    // swap(9,25) 
    { let tmp_1948 = keys[9]; keys[9] = keys[25]; keys[25] = tmp_1948;let tmp_1949 = values[9]; values[9] = values[25]; values[25] = tmp_1949; }
    }
    // cmp_swap(10,26)
    if keys[10] > keys[26] || (keys[10] == keys[26] && values[10] > values[26]) {
    // swap(10,26) 
    { let tmp_1950 = keys[10]; keys[10] = keys[26]; keys[26] = tmp_1950;let tmp_1951 = values[10]; values[10] = values[26]; values[26] = tmp_1951; }
    }
    // cmp_swap(11,27)
    if keys[11] > keys[27] || (keys[11] == keys[27] && values[11] > values[27]) {
    // swap(11,27) 
    { let tmp_1952 = keys[11]; keys[11] = keys[27]; keys[27] = tmp_1952;let tmp_1953 = values[11]; values[11] = values[27]; values[27] = tmp_1953; }
    }
    // cmp_swap(12,28)
    if keys[12] > keys[28] || (keys[12] == keys[28] && values[12] > values[28]) {
    // swap(12,28) 
    { let tmp_1954 = keys[12]; keys[12] = keys[28]; keys[28] = tmp_1954;let tmp_1955 = values[12]; values[12] = values[28]; values[28] = tmp_1955; }
    }
    // cmp_swap(13,29)
    if keys[13] > keys[29] || (keys[13] == keys[29] && values[13] > values[29]) {
    // swap(13,29) 
    { let tmp_1956 = keys[13]; keys[13] = keys[29]; keys[29] = tmp_1956;let tmp_1957 = values[13]; values[13] = values[29]; values[29] = tmp_1957; }
    }
    // cmp_swap(14,30)
    if keys[14] > keys[30] || (keys[14] == keys[30] && values[14] > values[30]) {
    // swap(14,30) 
    { let tmp_1958 = keys[14]; keys[14] = keys[30]; keys[30] = tmp_1958;let tmp_1959 = values[14]; values[14] = values[30]; values[30] = tmp_1959; }
    }
    // cmp_swap(15,31)
    if keys[15] > keys[31] || (keys[15] == keys[31] && values[15] > values[31]) {
    // swap(15,31) 
    { let tmp_1960 = keys[15]; keys[15] = keys[31]; keys[31] = tmp_1960;let tmp_1961 = values[15]; values[15] = values[31]; values[31] = tmp_1961; }
    }
    // exch_local(8,32) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_1962 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_1962;let tmp_1963 = values[0]; values[0] = values[8]; values[8] = tmp_1963; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_1964 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_1964;let tmp_1965 = values[1]; values[1] = values[9]; values[9] = tmp_1965; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_1966 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_1966;let tmp_1967 = values[2]; values[2] = values[10]; values[10] = tmp_1967; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_1968 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_1968;let tmp_1969 = values[3]; values[3] = values[11]; values[11] = tmp_1969; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_1970 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_1970;let tmp_1971 = values[4]; values[4] = values[12]; values[12] = tmp_1971; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_1972 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_1972;let tmp_1973 = values[5]; values[5] = values[13]; values[13] = tmp_1973; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_1974 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_1974;let tmp_1975 = values[6]; values[6] = values[14]; values[14] = tmp_1975; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_1976 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_1976;let tmp_1977 = values[7]; values[7] = values[15]; values[15] = tmp_1977; }
    }
    // cmp_swap(16,24)
    if keys[16] > keys[24] || (keys[16] == keys[24] && values[16] > values[24]) {
    // swap(16,24) 
    { let tmp_1978 = keys[16]; keys[16] = keys[24]; keys[24] = tmp_1978;let tmp_1979 = values[16]; values[16] = values[24]; values[24] = tmp_1979; }
    }
    // cmp_swap(17,25)
    if keys[17] > keys[25] || (keys[17] == keys[25] && values[17] > values[25]) {
    // swap(17,25) 
    { let tmp_1980 = keys[17]; keys[17] = keys[25]; keys[25] = tmp_1980;let tmp_1981 = values[17]; values[17] = values[25]; values[25] = tmp_1981; }
    }
    // cmp_swap(18,26)
    if keys[18] > keys[26] || (keys[18] == keys[26] && values[18] > values[26]) {
    // swap(18,26) 
    { let tmp_1982 = keys[18]; keys[18] = keys[26]; keys[26] = tmp_1982;let tmp_1983 = values[18]; values[18] = values[26]; values[26] = tmp_1983; }
    }
    // cmp_swap(19,27)
    if keys[19] > keys[27] || (keys[19] == keys[27] && values[19] > values[27]) {
    // swap(19,27) 
    { let tmp_1984 = keys[19]; keys[19] = keys[27]; keys[27] = tmp_1984;let tmp_1985 = values[19]; values[19] = values[27]; values[27] = tmp_1985; }
    }
    // cmp_swap(20,28)
    if keys[20] > keys[28] || (keys[20] == keys[28] && values[20] > values[28]) {
    // swap(20,28) 
    { let tmp_1986 = keys[20]; keys[20] = keys[28]; keys[28] = tmp_1986;let tmp_1987 = values[20]; values[20] = values[28]; values[28] = tmp_1987; }
    }
    // cmp_swap(21,29)
    if keys[21] > keys[29] || (keys[21] == keys[29] && values[21] > values[29]) {
    // swap(21,29) 
    { let tmp_1988 = keys[21]; keys[21] = keys[29]; keys[29] = tmp_1988;let tmp_1989 = values[21]; values[21] = values[29]; values[29] = tmp_1989; }
    }
    // cmp_swap(22,30)
    if keys[22] > keys[30] || (keys[22] == keys[30] && values[22] > values[30]) {
    // swap(22,30) 
    { let tmp_1990 = keys[22]; keys[22] = keys[30]; keys[30] = tmp_1990;let tmp_1991 = values[22]; values[22] = values[30]; values[30] = tmp_1991; }
    }
    // cmp_swap(23,31)
    if keys[23] > keys[31] || (keys[23] == keys[31] && values[23] > values[31]) {
    // swap(23,31) 
    { let tmp_1992 = keys[23]; keys[23] = keys[31]; keys[31] = tmp_1992;let tmp_1993 = values[23]; values[23] = values[31]; values[31] = tmp_1993; }
    }
    // exch_local(4,32) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_1994 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_1994;let tmp_1995 = values[0]; values[0] = values[4]; values[4] = tmp_1995; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_1996 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_1996;let tmp_1997 = values[1]; values[1] = values[5]; values[5] = tmp_1997; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_1998 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_1998;let tmp_1999 = values[2]; values[2] = values[6]; values[6] = tmp_1999; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_2000 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_2000;let tmp_2001 = values[3]; values[3] = values[7]; values[7] = tmp_2001; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_2002 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_2002;let tmp_2003 = values[8]; values[8] = values[12]; values[12] = tmp_2003; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_2004 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_2004;let tmp_2005 = values[9]; values[9] = values[13]; values[13] = tmp_2005; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_2006 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_2006;let tmp_2007 = values[10]; values[10] = values[14]; values[14] = tmp_2007; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_2008 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_2008;let tmp_2009 = values[11]; values[11] = values[15]; values[15] = tmp_2009; }
    }
    // cmp_swap(16,20)
    if keys[16] > keys[20] || (keys[16] == keys[20] && values[16] > values[20]) {
    // swap(16,20) 
    { let tmp_2010 = keys[16]; keys[16] = keys[20]; keys[20] = tmp_2010;let tmp_2011 = values[16]; values[16] = values[20]; values[20] = tmp_2011; }
    }
    // cmp_swap(17,21)
    if keys[17] > keys[21] || (keys[17] == keys[21] && values[17] > values[21]) {
    // swap(17,21) 
    { let tmp_2012 = keys[17]; keys[17] = keys[21]; keys[21] = tmp_2012;let tmp_2013 = values[17]; values[17] = values[21]; values[21] = tmp_2013; }
    }
    // cmp_swap(18,22)
    if keys[18] > keys[22] || (keys[18] == keys[22] && values[18] > values[22]) {
    // swap(18,22) 
    { let tmp_2014 = keys[18]; keys[18] = keys[22]; keys[22] = tmp_2014;let tmp_2015 = values[18]; values[18] = values[22]; values[22] = tmp_2015; }
    }
    // cmp_swap(19,23)
    if keys[19] > keys[23] || (keys[19] == keys[23] && values[19] > values[23]) {
    // swap(19,23) 
    { let tmp_2016 = keys[19]; keys[19] = keys[23]; keys[23] = tmp_2016;let tmp_2017 = values[19]; values[19] = values[23]; values[23] = tmp_2017; }
    }
    // cmp_swap(24,28)
    if keys[24] > keys[28] || (keys[24] == keys[28] && values[24] > values[28]) {
    // swap(24,28) 
    { let tmp_2018 = keys[24]; keys[24] = keys[28]; keys[28] = tmp_2018;let tmp_2019 = values[24]; values[24] = values[28]; values[28] = tmp_2019; }
    }
    // cmp_swap(25,29)
    if keys[25] > keys[29] || (keys[25] == keys[29] && values[25] > values[29]) {
    // swap(25,29) 
    { let tmp_2020 = keys[25]; keys[25] = keys[29]; keys[29] = tmp_2020;let tmp_2021 = values[25]; values[25] = values[29]; values[29] = tmp_2021; }
    }
    // cmp_swap(26,30)
    if keys[26] > keys[30] || (keys[26] == keys[30] && values[26] > values[30]) {
    // swap(26,30) 
    { let tmp_2022 = keys[26]; keys[26] = keys[30]; keys[30] = tmp_2022;let tmp_2023 = values[26]; values[26] = values[30]; values[30] = tmp_2023; }
    }
    // cmp_swap(27,31)
    if keys[27] > keys[31] || (keys[27] == keys[31] && values[27] > values[31]) {
    // swap(27,31) 
    { let tmp_2024 = keys[27]; keys[27] = keys[31]; keys[31] = tmp_2024;let tmp_2025 = values[27]; values[27] = values[31]; values[31] = tmp_2025; }
    }
    // exch_local(2,32) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_2026 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_2026;let tmp_2027 = values[0]; values[0] = values[2]; values[2] = tmp_2027; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_2028 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_2028;let tmp_2029 = values[1]; values[1] = values[3]; values[3] = tmp_2029; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_2030 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_2030;let tmp_2031 = values[4]; values[4] = values[6]; values[6] = tmp_2031; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_2032 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_2032;let tmp_2033 = values[5]; values[5] = values[7]; values[7] = tmp_2033; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_2034 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_2034;let tmp_2035 = values[8]; values[8] = values[10]; values[10] = tmp_2035; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_2036 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_2036;let tmp_2037 = values[9]; values[9] = values[11]; values[11] = tmp_2037; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_2038 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_2038;let tmp_2039 = values[12]; values[12] = values[14]; values[14] = tmp_2039; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_2040 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_2040;let tmp_2041 = values[13]; values[13] = values[15]; values[15] = tmp_2041; }
    }
    // cmp_swap(16,18)
    if keys[16] > keys[18] || (keys[16] == keys[18] && values[16] > values[18]) {
    // swap(16,18) 
    { let tmp_2042 = keys[16]; keys[16] = keys[18]; keys[18] = tmp_2042;let tmp_2043 = values[16]; values[16] = values[18]; values[18] = tmp_2043; }
    }
    // cmp_swap(17,19)
    if keys[17] > keys[19] || (keys[17] == keys[19] && values[17] > values[19]) {
    // swap(17,19) 
    { let tmp_2044 = keys[17]; keys[17] = keys[19]; keys[19] = tmp_2044;let tmp_2045 = values[17]; values[17] = values[19]; values[19] = tmp_2045; }
    }
    // cmp_swap(20,22)
    if keys[20] > keys[22] || (keys[20] == keys[22] && values[20] > values[22]) {
    // swap(20,22) 
    { let tmp_2046 = keys[20]; keys[20] = keys[22]; keys[22] = tmp_2046;let tmp_2047 = values[20]; values[20] = values[22]; values[22] = tmp_2047; }
    }
    // cmp_swap(21,23)
    if keys[21] > keys[23] || (keys[21] == keys[23] && values[21] > values[23]) {
    // swap(21,23) 
    { let tmp_2048 = keys[21]; keys[21] = keys[23]; keys[23] = tmp_2048;let tmp_2049 = values[21]; values[21] = values[23]; values[23] = tmp_2049; }
    }
    // cmp_swap(24,26)
    if keys[24] > keys[26] || (keys[24] == keys[26] && values[24] > values[26]) {
    // swap(24,26) 
    { let tmp_2050 = keys[24]; keys[24] = keys[26]; keys[26] = tmp_2050;let tmp_2051 = values[24]; values[24] = values[26]; values[26] = tmp_2051; }
    }
    // cmp_swap(25,27)
    if keys[25] > keys[27] || (keys[25] == keys[27] && values[25] > values[27]) {
    // swap(25,27) 
    { let tmp_2052 = keys[25]; keys[25] = keys[27]; keys[27] = tmp_2052;let tmp_2053 = values[25]; values[25] = values[27]; values[27] = tmp_2053; }
    }
    // cmp_swap(28,30)
    if keys[28] > keys[30] || (keys[28] == keys[30] && values[28] > values[30]) {
    // swap(28,30) 
    { let tmp_2054 = keys[28]; keys[28] = keys[30]; keys[30] = tmp_2054;let tmp_2055 = values[28]; values[28] = values[30]; values[30] = tmp_2055; }
    }
    // cmp_swap(29,31)
    if keys[29] > keys[31] || (keys[29] == keys[31] && values[29] > values[31]) {
    // swap(29,31) 
    { let tmp_2056 = keys[29]; keys[29] = keys[31]; keys[31] = tmp_2056;let tmp_2057 = values[29]; values[29] = values[31]; values[31] = tmp_2057; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_2058 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_2058;let tmp_2059 = values[0]; values[0] = values[1]; values[1] = tmp_2059; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_2060 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_2060;let tmp_2061 = values[2]; values[2] = values[3]; values[3] = tmp_2061; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_2062 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_2062;let tmp_2063 = values[4]; values[4] = values[5]; values[5] = tmp_2063; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_2064 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_2064;let tmp_2065 = values[6]; values[6] = values[7]; values[7] = tmp_2065; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_2066 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_2066;let tmp_2067 = values[8]; values[8] = values[9]; values[9] = tmp_2067; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_2068 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_2068;let tmp_2069 = values[10]; values[10] = values[11]; values[11] = tmp_2069; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_2070 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_2070;let tmp_2071 = values[12]; values[12] = values[13]; values[13] = tmp_2071; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_2072 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_2072;let tmp_2073 = values[14]; values[14] = values[15]; values[15] = tmp_2073; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_2074 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_2074;let tmp_2075 = values[16]; values[16] = values[17]; values[17] = tmp_2075; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_2076 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_2076;let tmp_2077 = values[18]; values[18] = values[19]; values[19] = tmp_2077; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_2078 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_2078;let tmp_2079 = values[20]; values[20] = values[21]; values[21] = tmp_2079; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_2080 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_2080;let tmp_2081 = values[22]; values[22] = values[23]; values[23] = tmp_2081; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_2082 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_2082;let tmp_2083 = values[24]; values[24] = values[25]; values[25] = tmp_2083; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_2084 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_2084;let tmp_2085 = values[26]; values[26] = values[27]; values[27] = tmp_2085; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_2086 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_2086;let tmp_2087 = values[28]; values[28] = values[29]; values[29] = tmp_2087; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_2088 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_2088;let tmp_2089 = values[30]; values[30] = values[31]; values[31] = tmp_2089; }
    }
    // exch_intxn(tmask:31,swbit:4,wpt:32)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_2090 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_2091 = seg_base + (local_tid ^ 31u); let tmp_2092 = smem_keys[tmp_2091 * WPT + 31u]; let tmp_2093 = smem_vals[tmp_2091 * WPT + 31u]; let tmp_2094 = keys[0] < tmp_2092 || (keys[0] == tmp_2092 && values[0] < tmp_2093); if tmp_2090 == tmp_2094 { keys[0] = tmp_2092; values[0] = tmp_2093; } let tmp_2095 = smem_keys[tmp_2091 * WPT + 30u]; let tmp_2096 = smem_vals[tmp_2091 * WPT + 30u]; let tmp_2097 = keys[1] < tmp_2095 || (keys[1] == tmp_2095 && values[1] < tmp_2096); if tmp_2090 == tmp_2097 { keys[1] = tmp_2095; values[1] = tmp_2096; } let tmp_2098 = smem_keys[tmp_2091 * WPT + 29u]; let tmp_2099 = smem_vals[tmp_2091 * WPT + 29u]; let tmp_2100 = keys[2] < tmp_2098 || (keys[2] == tmp_2098 && values[2] < tmp_2099); if tmp_2090 == tmp_2100 { keys[2] = tmp_2098; values[2] = tmp_2099; } let tmp_2101 = smem_keys[tmp_2091 * WPT + 28u]; let tmp_2102 = smem_vals[tmp_2091 * WPT + 28u]; let tmp_2103 = keys[3] < tmp_2101 || (keys[3] == tmp_2101 && values[3] < tmp_2102); if tmp_2090 == tmp_2103 { keys[3] = tmp_2101; values[3] = tmp_2102; } let tmp_2104 = smem_keys[tmp_2091 * WPT + 27u]; let tmp_2105 = smem_vals[tmp_2091 * WPT + 27u]; let tmp_2106 = keys[4] < tmp_2104 || (keys[4] == tmp_2104 && values[4] < tmp_2105); if tmp_2090 == tmp_2106 { keys[4] = tmp_2104; values[4] = tmp_2105; } let tmp_2107 = smem_keys[tmp_2091 * WPT + 26u]; let tmp_2108 = smem_vals[tmp_2091 * WPT + 26u]; let tmp_2109 = keys[5] < tmp_2107 || (keys[5] == tmp_2107 && values[5] < tmp_2108); if tmp_2090 == tmp_2109 { keys[5] = tmp_2107; values[5] = tmp_2108; } let tmp_2110 = smem_keys[tmp_2091 * WPT + 25u]; let tmp_2111 = smem_vals[tmp_2091 * WPT + 25u]; let tmp_2112 = keys[6] < tmp_2110 || (keys[6] == tmp_2110 && values[6] < tmp_2111); if tmp_2090 == tmp_2112 { keys[6] = tmp_2110; values[6] = tmp_2111; } let tmp_2113 = smem_keys[tmp_2091 * WPT + 24u]; let tmp_2114 = smem_vals[tmp_2091 * WPT + 24u]; let tmp_2115 = keys[7] < tmp_2113 || (keys[7] == tmp_2113 && values[7] < tmp_2114); if tmp_2090 == tmp_2115 { keys[7] = tmp_2113; values[7] = tmp_2114; } let tmp_2116 = smem_keys[tmp_2091 * WPT + 23u]; let tmp_2117 = smem_vals[tmp_2091 * WPT + 23u]; let tmp_2118 = keys[8] < tmp_2116 || (keys[8] == tmp_2116 && values[8] < tmp_2117); if tmp_2090 == tmp_2118 { keys[8] = tmp_2116; values[8] = tmp_2117; } let tmp_2119 = smem_keys[tmp_2091 * WPT + 22u]; let tmp_2120 = smem_vals[tmp_2091 * WPT + 22u]; let tmp_2121 = keys[9] < tmp_2119 || (keys[9] == tmp_2119 && values[9] < tmp_2120); if tmp_2090 == tmp_2121 { keys[9] = tmp_2119; values[9] = tmp_2120; } let tmp_2122 = smem_keys[tmp_2091 * WPT + 21u]; let tmp_2123 = smem_vals[tmp_2091 * WPT + 21u]; let tmp_2124 = keys[10] < tmp_2122 || (keys[10] == tmp_2122 && values[10] < tmp_2123); if tmp_2090 == tmp_2124 { keys[10] = tmp_2122; values[10] = tmp_2123; } let tmp_2125 = smem_keys[tmp_2091 * WPT + 20u]; let tmp_2126 = smem_vals[tmp_2091 * WPT + 20u]; let tmp_2127 = keys[11] < tmp_2125 || (keys[11] == tmp_2125 && values[11] < tmp_2126); if tmp_2090 == tmp_2127 { keys[11] = tmp_2125; values[11] = tmp_2126; } let tmp_2128 = smem_keys[tmp_2091 * WPT + 19u]; let tmp_2129 = smem_vals[tmp_2091 * WPT + 19u]; let tmp_2130 = keys[12] < tmp_2128 || (keys[12] == tmp_2128 && values[12] < tmp_2129); if tmp_2090 == tmp_2130 { keys[12] = tmp_2128; values[12] = tmp_2129; } let tmp_2131 = smem_keys[tmp_2091 * WPT + 18u]; let tmp_2132 = smem_vals[tmp_2091 * WPT + 18u]; let tmp_2133 = keys[13] < tmp_2131 || (keys[13] == tmp_2131 && values[13] < tmp_2132); if tmp_2090 == tmp_2133 { keys[13] = tmp_2131; values[13] = tmp_2132; } let tmp_2134 = smem_keys[tmp_2091 * WPT + 17u]; let tmp_2135 = smem_vals[tmp_2091 * WPT + 17u]; let tmp_2136 = keys[14] < tmp_2134 || (keys[14] == tmp_2134 && values[14] < tmp_2135); if tmp_2090 == tmp_2136 { keys[14] = tmp_2134; values[14] = tmp_2135; } let tmp_2137 = smem_keys[tmp_2091 * WPT + 16u]; let tmp_2138 = smem_vals[tmp_2091 * WPT + 16u]; let tmp_2139 = keys[15] < tmp_2137 || (keys[15] == tmp_2137 && values[15] < tmp_2138); if tmp_2090 == tmp_2139 { keys[15] = tmp_2137; values[15] = tmp_2138; } let tmp_2140 = smem_keys[tmp_2091 * WPT + 15u]; let tmp_2141 = smem_vals[tmp_2091 * WPT + 15u]; let tmp_2142 = keys[16] < tmp_2140 || (keys[16] == tmp_2140 && values[16] < tmp_2141); if tmp_2090 == tmp_2142 { keys[16] = tmp_2140; values[16] = tmp_2141; } let tmp_2143 = smem_keys[tmp_2091 * WPT + 14u]; let tmp_2144 = smem_vals[tmp_2091 * WPT + 14u]; let tmp_2145 = keys[17] < tmp_2143 || (keys[17] == tmp_2143 && values[17] < tmp_2144); if tmp_2090 == tmp_2145 { keys[17] = tmp_2143; values[17] = tmp_2144; } let tmp_2146 = smem_keys[tmp_2091 * WPT + 13u]; let tmp_2147 = smem_vals[tmp_2091 * WPT + 13u]; let tmp_2148 = keys[18] < tmp_2146 || (keys[18] == tmp_2146 && values[18] < tmp_2147); if tmp_2090 == tmp_2148 { keys[18] = tmp_2146; values[18] = tmp_2147; } let tmp_2149 = smem_keys[tmp_2091 * WPT + 12u]; let tmp_2150 = smem_vals[tmp_2091 * WPT + 12u]; let tmp_2151 = keys[19] < tmp_2149 || (keys[19] == tmp_2149 && values[19] < tmp_2150); if tmp_2090 == tmp_2151 { keys[19] = tmp_2149; values[19] = tmp_2150; } let tmp_2152 = smem_keys[tmp_2091 * WPT + 11u]; let tmp_2153 = smem_vals[tmp_2091 * WPT + 11u]; let tmp_2154 = keys[20] < tmp_2152 || (keys[20] == tmp_2152 && values[20] < tmp_2153); if tmp_2090 == tmp_2154 { keys[20] = tmp_2152; values[20] = tmp_2153; } let tmp_2155 = smem_keys[tmp_2091 * WPT + 10u]; let tmp_2156 = smem_vals[tmp_2091 * WPT + 10u]; let tmp_2157 = keys[21] < tmp_2155 || (keys[21] == tmp_2155 && values[21] < tmp_2156); if tmp_2090 == tmp_2157 { keys[21] = tmp_2155; values[21] = tmp_2156; } let tmp_2158 = smem_keys[tmp_2091 * WPT + 9u]; let tmp_2159 = smem_vals[tmp_2091 * WPT + 9u]; let tmp_2160 = keys[22] < tmp_2158 || (keys[22] == tmp_2158 && values[22] < tmp_2159); if tmp_2090 == tmp_2160 { keys[22] = tmp_2158; values[22] = tmp_2159; } let tmp_2161 = smem_keys[tmp_2091 * WPT + 8u]; let tmp_2162 = smem_vals[tmp_2091 * WPT + 8u]; let tmp_2163 = keys[23] < tmp_2161 || (keys[23] == tmp_2161 && values[23] < tmp_2162); if tmp_2090 == tmp_2163 { keys[23] = tmp_2161; values[23] = tmp_2162; } let tmp_2164 = smem_keys[tmp_2091 * WPT + 7u]; let tmp_2165 = smem_vals[tmp_2091 * WPT + 7u]; let tmp_2166 = keys[24] < tmp_2164 || (keys[24] == tmp_2164 && values[24] < tmp_2165); if tmp_2090 == tmp_2166 { keys[24] = tmp_2164; values[24] = tmp_2165; } let tmp_2167 = smem_keys[tmp_2091 * WPT + 6u]; let tmp_2168 = smem_vals[tmp_2091 * WPT + 6u]; let tmp_2169 = keys[25] < tmp_2167 || (keys[25] == tmp_2167 && values[25] < tmp_2168); if tmp_2090 == tmp_2169 { keys[25] = tmp_2167; values[25] = tmp_2168; } let tmp_2170 = smem_keys[tmp_2091 * WPT + 5u]; let tmp_2171 = smem_vals[tmp_2091 * WPT + 5u]; let tmp_2172 = keys[26] < tmp_2170 || (keys[26] == tmp_2170 && values[26] < tmp_2171); if tmp_2090 == tmp_2172 { keys[26] = tmp_2170; values[26] = tmp_2171; } let tmp_2173 = smem_keys[tmp_2091 * WPT + 4u]; let tmp_2174 = smem_vals[tmp_2091 * WPT + 4u]; let tmp_2175 = keys[27] < tmp_2173 || (keys[27] == tmp_2173 && values[27] < tmp_2174); if tmp_2090 == tmp_2175 { keys[27] = tmp_2173; values[27] = tmp_2174; } let tmp_2176 = smem_keys[tmp_2091 * WPT + 3u]; let tmp_2177 = smem_vals[tmp_2091 * WPT + 3u]; let tmp_2178 = keys[28] < tmp_2176 || (keys[28] == tmp_2176 && values[28] < tmp_2177); if tmp_2090 == tmp_2178 { keys[28] = tmp_2176; values[28] = tmp_2177; } let tmp_2179 = smem_keys[tmp_2091 * WPT + 2u]; let tmp_2180 = smem_vals[tmp_2091 * WPT + 2u]; let tmp_2181 = keys[29] < tmp_2179 || (keys[29] == tmp_2179 && values[29] < tmp_2180); if tmp_2090 == tmp_2181 { keys[29] = tmp_2179; values[29] = tmp_2180; } let tmp_2182 = smem_keys[tmp_2091 * WPT + 1u]; let tmp_2183 = smem_vals[tmp_2091 * WPT + 1u]; let tmp_2184 = keys[30] < tmp_2182 || (keys[30] == tmp_2182 && values[30] < tmp_2183); if tmp_2090 == tmp_2184 { keys[30] = tmp_2182; values[30] = tmp_2183; } let tmp_2185 = smem_keys[tmp_2091 * WPT + 0u]; let tmp_2186 = smem_vals[tmp_2091 * WPT + 0u]; let tmp_2187 = keys[31] < tmp_2185 || (keys[31] == tmp_2185 && values[31] < tmp_2186); if tmp_2090 == tmp_2187 { keys[31] = tmp_2185; values[31] = tmp_2186; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:32) 
    {
    let tmp_2188 = subgroupShuffleXor(keys[0], 8u);
    let tmp_2189 = subgroupShuffleXor(values[0], 8u);
    let tmp_2190 = subgroupShuffleXor(keys[1], 8u);
    let tmp_2191 = subgroupShuffleXor(values[1], 8u);
    let tmp_2192 = subgroupShuffleXor(keys[2], 8u);
    let tmp_2193 = subgroupShuffleXor(values[2], 8u);
    let tmp_2194 = subgroupShuffleXor(keys[3], 8u);
    let tmp_2195 = subgroupShuffleXor(values[3], 8u);
    let tmp_2196 = subgroupShuffleXor(keys[4], 8u);
    let tmp_2197 = subgroupShuffleXor(values[4], 8u);
    let tmp_2198 = subgroupShuffleXor(keys[5], 8u);
    let tmp_2199 = subgroupShuffleXor(values[5], 8u);
    let tmp_2200 = subgroupShuffleXor(keys[6], 8u);
    let tmp_2201 = subgroupShuffleXor(values[6], 8u);
    let tmp_2202 = subgroupShuffleXor(keys[7], 8u);
    let tmp_2203 = subgroupShuffleXor(values[7], 8u);
    let tmp_2204 = subgroupShuffleXor(keys[8], 8u);
    let tmp_2205 = subgroupShuffleXor(values[8], 8u);
    let tmp_2206 = subgroupShuffleXor(keys[9], 8u);
    let tmp_2207 = subgroupShuffleXor(values[9], 8u);
    let tmp_2208 = subgroupShuffleXor(keys[10], 8u);
    let tmp_2209 = subgroupShuffleXor(values[10], 8u);
    let tmp_2210 = subgroupShuffleXor(keys[11], 8u);
    let tmp_2211 = subgroupShuffleXor(values[11], 8u);
    let tmp_2212 = subgroupShuffleXor(keys[12], 8u);
    let tmp_2213 = subgroupShuffleXor(values[12], 8u);
    let tmp_2214 = subgroupShuffleXor(keys[13], 8u);
    let tmp_2215 = subgroupShuffleXor(values[13], 8u);
    let tmp_2216 = subgroupShuffleXor(keys[14], 8u);
    let tmp_2217 = subgroupShuffleXor(values[14], 8u);
    let tmp_2218 = subgroupShuffleXor(keys[15], 8u);
    let tmp_2219 = subgroupShuffleXor(values[15], 8u);
    let tmp_2220 = subgroupShuffleXor(keys[16], 8u);
    let tmp_2221 = subgroupShuffleXor(values[16], 8u);
    let tmp_2222 = subgroupShuffleXor(keys[17], 8u);
    let tmp_2223 = subgroupShuffleXor(values[17], 8u);
    let tmp_2224 = subgroupShuffleXor(keys[18], 8u);
    let tmp_2225 = subgroupShuffleXor(values[18], 8u);
    let tmp_2226 = subgroupShuffleXor(keys[19], 8u);
    let tmp_2227 = subgroupShuffleXor(values[19], 8u);
    let tmp_2228 = subgroupShuffleXor(keys[20], 8u);
    let tmp_2229 = subgroupShuffleXor(values[20], 8u);
    let tmp_2230 = subgroupShuffleXor(keys[21], 8u);
    let tmp_2231 = subgroupShuffleXor(values[21], 8u);
    let tmp_2232 = subgroupShuffleXor(keys[22], 8u);
    let tmp_2233 = subgroupShuffleXor(values[22], 8u);
    let tmp_2234 = subgroupShuffleXor(keys[23], 8u);
    let tmp_2235 = subgroupShuffleXor(values[23], 8u);
    let tmp_2236 = subgroupShuffleXor(keys[24], 8u);
    let tmp_2237 = subgroupShuffleXor(values[24], 8u);
    let tmp_2238 = subgroupShuffleXor(keys[25], 8u);
    let tmp_2239 = subgroupShuffleXor(values[25], 8u);
    let tmp_2240 = subgroupShuffleXor(keys[26], 8u);
    let tmp_2241 = subgroupShuffleXor(values[26], 8u);
    let tmp_2242 = subgroupShuffleXor(keys[27], 8u);
    let tmp_2243 = subgroupShuffleXor(values[27], 8u);
    let tmp_2244 = subgroupShuffleXor(keys[28], 8u);
    let tmp_2245 = subgroupShuffleXor(values[28], 8u);
    let tmp_2246 = subgroupShuffleXor(keys[29], 8u);
    let tmp_2247 = subgroupShuffleXor(values[29], 8u);
    let tmp_2248 = subgroupShuffleXor(keys[30], 8u);
    let tmp_2249 = subgroupShuffleXor(values[30], 8u);
    let tmp_2250 = subgroupShuffleXor(keys[31], 8u);
    let tmp_2251 = subgroupShuffleXor(values[31], 8u);
    let tmp_2252 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_2253 = keys[0] < tmp_2188 || (keys[0] == tmp_2188 && values[0] < tmp_2189);
    if tmp_2252 == tmp_2253 { keys[0] = tmp_2188; values[0] = tmp_2189; }
    let tmp_2254 = keys[1] < tmp_2190 || (keys[1] == tmp_2190 && values[1] < tmp_2191);
    if tmp_2252 == tmp_2254 { keys[1] = tmp_2190; values[1] = tmp_2191; }
    let tmp_2255 = keys[2] < tmp_2192 || (keys[2] == tmp_2192 && values[2] < tmp_2193);
    if tmp_2252 == tmp_2255 { keys[2] = tmp_2192; values[2] = tmp_2193; }
    let tmp_2256 = keys[3] < tmp_2194 || (keys[3] == tmp_2194 && values[3] < tmp_2195);
    if tmp_2252 == tmp_2256 { keys[3] = tmp_2194; values[3] = tmp_2195; }
    let tmp_2257 = keys[4] < tmp_2196 || (keys[4] == tmp_2196 && values[4] < tmp_2197);
    if tmp_2252 == tmp_2257 { keys[4] = tmp_2196; values[4] = tmp_2197; }
    let tmp_2258 = keys[5] < tmp_2198 || (keys[5] == tmp_2198 && values[5] < tmp_2199);
    if tmp_2252 == tmp_2258 { keys[5] = tmp_2198; values[5] = tmp_2199; }
    let tmp_2259 = keys[6] < tmp_2200 || (keys[6] == tmp_2200 && values[6] < tmp_2201);
    if tmp_2252 == tmp_2259 { keys[6] = tmp_2200; values[6] = tmp_2201; }
    let tmp_2260 = keys[7] < tmp_2202 || (keys[7] == tmp_2202 && values[7] < tmp_2203);
    if tmp_2252 == tmp_2260 { keys[7] = tmp_2202; values[7] = tmp_2203; }
    let tmp_2261 = keys[8] < tmp_2204 || (keys[8] == tmp_2204 && values[8] < tmp_2205);
    if tmp_2252 == tmp_2261 { keys[8] = tmp_2204; values[8] = tmp_2205; }
    let tmp_2262 = keys[9] < tmp_2206 || (keys[9] == tmp_2206 && values[9] < tmp_2207);
    if tmp_2252 == tmp_2262 { keys[9] = tmp_2206; values[9] = tmp_2207; }
    let tmp_2263 = keys[10] < tmp_2208 || (keys[10] == tmp_2208 && values[10] < tmp_2209);
    if tmp_2252 == tmp_2263 { keys[10] = tmp_2208; values[10] = tmp_2209; }
    let tmp_2264 = keys[11] < tmp_2210 || (keys[11] == tmp_2210 && values[11] < tmp_2211);
    if tmp_2252 == tmp_2264 { keys[11] = tmp_2210; values[11] = tmp_2211; }
    let tmp_2265 = keys[12] < tmp_2212 || (keys[12] == tmp_2212 && values[12] < tmp_2213);
    if tmp_2252 == tmp_2265 { keys[12] = tmp_2212; values[12] = tmp_2213; }
    let tmp_2266 = keys[13] < tmp_2214 || (keys[13] == tmp_2214 && values[13] < tmp_2215);
    if tmp_2252 == tmp_2266 { keys[13] = tmp_2214; values[13] = tmp_2215; }
    let tmp_2267 = keys[14] < tmp_2216 || (keys[14] == tmp_2216 && values[14] < tmp_2217);
    if tmp_2252 == tmp_2267 { keys[14] = tmp_2216; values[14] = tmp_2217; }
    let tmp_2268 = keys[15] < tmp_2218 || (keys[15] == tmp_2218 && values[15] < tmp_2219);
    if tmp_2252 == tmp_2268 { keys[15] = tmp_2218; values[15] = tmp_2219; }
    let tmp_2269 = keys[16] < tmp_2220 || (keys[16] == tmp_2220 && values[16] < tmp_2221);
    if tmp_2252 == tmp_2269 { keys[16] = tmp_2220; values[16] = tmp_2221; }
    let tmp_2270 = keys[17] < tmp_2222 || (keys[17] == tmp_2222 && values[17] < tmp_2223);
    if tmp_2252 == tmp_2270 { keys[17] = tmp_2222; values[17] = tmp_2223; }
    let tmp_2271 = keys[18] < tmp_2224 || (keys[18] == tmp_2224 && values[18] < tmp_2225);
    if tmp_2252 == tmp_2271 { keys[18] = tmp_2224; values[18] = tmp_2225; }
    let tmp_2272 = keys[19] < tmp_2226 || (keys[19] == tmp_2226 && values[19] < tmp_2227);
    if tmp_2252 == tmp_2272 { keys[19] = tmp_2226; values[19] = tmp_2227; }
    let tmp_2273 = keys[20] < tmp_2228 || (keys[20] == tmp_2228 && values[20] < tmp_2229);
    if tmp_2252 == tmp_2273 { keys[20] = tmp_2228; values[20] = tmp_2229; }
    let tmp_2274 = keys[21] < tmp_2230 || (keys[21] == tmp_2230 && values[21] < tmp_2231);
    if tmp_2252 == tmp_2274 { keys[21] = tmp_2230; values[21] = tmp_2231; }
    let tmp_2275 = keys[22] < tmp_2232 || (keys[22] == tmp_2232 && values[22] < tmp_2233);
    if tmp_2252 == tmp_2275 { keys[22] = tmp_2232; values[22] = tmp_2233; }
    let tmp_2276 = keys[23] < tmp_2234 || (keys[23] == tmp_2234 && values[23] < tmp_2235);
    if tmp_2252 == tmp_2276 { keys[23] = tmp_2234; values[23] = tmp_2235; }
    let tmp_2277 = keys[24] < tmp_2236 || (keys[24] == tmp_2236 && values[24] < tmp_2237);
    if tmp_2252 == tmp_2277 { keys[24] = tmp_2236; values[24] = tmp_2237; }
    let tmp_2278 = keys[25] < tmp_2238 || (keys[25] == tmp_2238 && values[25] < tmp_2239);
    if tmp_2252 == tmp_2278 { keys[25] = tmp_2238; values[25] = tmp_2239; }
    let tmp_2279 = keys[26] < tmp_2240 || (keys[26] == tmp_2240 && values[26] < tmp_2241);
    if tmp_2252 == tmp_2279 { keys[26] = tmp_2240; values[26] = tmp_2241; }
    let tmp_2280 = keys[27] < tmp_2242 || (keys[27] == tmp_2242 && values[27] < tmp_2243);
    if tmp_2252 == tmp_2280 { keys[27] = tmp_2242; values[27] = tmp_2243; }
    let tmp_2281 = keys[28] < tmp_2244 || (keys[28] == tmp_2244 && values[28] < tmp_2245);
    if tmp_2252 == tmp_2281 { keys[28] = tmp_2244; values[28] = tmp_2245; }
    let tmp_2282 = keys[29] < tmp_2246 || (keys[29] == tmp_2246 && values[29] < tmp_2247);
    if tmp_2252 == tmp_2282 { keys[29] = tmp_2246; values[29] = tmp_2247; }
    let tmp_2283 = keys[30] < tmp_2248 || (keys[30] == tmp_2248 && values[30] < tmp_2249);
    if tmp_2252 == tmp_2283 { keys[30] = tmp_2248; values[30] = tmp_2249; }
    let tmp_2284 = keys[31] < tmp_2250 || (keys[31] == tmp_2250 && values[31] < tmp_2251);
    if tmp_2252 == tmp_2284 { keys[31] = tmp_2250; values[31] = tmp_2251; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:32) 
    {
    let tmp_2285 = subgroupShuffleXor(keys[0], 4u);
    let tmp_2286 = subgroupShuffleXor(values[0], 4u);
    let tmp_2287 = subgroupShuffleXor(keys[1], 4u);
    let tmp_2288 = subgroupShuffleXor(values[1], 4u);
    let tmp_2289 = subgroupShuffleXor(keys[2], 4u);
    let tmp_2290 = subgroupShuffleXor(values[2], 4u);
    let tmp_2291 = subgroupShuffleXor(keys[3], 4u);
    let tmp_2292 = subgroupShuffleXor(values[3], 4u);
    let tmp_2293 = subgroupShuffleXor(keys[4], 4u);
    let tmp_2294 = subgroupShuffleXor(values[4], 4u);
    let tmp_2295 = subgroupShuffleXor(keys[5], 4u);
    let tmp_2296 = subgroupShuffleXor(values[5], 4u);
    let tmp_2297 = subgroupShuffleXor(keys[6], 4u);
    let tmp_2298 = subgroupShuffleXor(values[6], 4u);
    let tmp_2299 = subgroupShuffleXor(keys[7], 4u);
    let tmp_2300 = subgroupShuffleXor(values[7], 4u);
    let tmp_2301 = subgroupShuffleXor(keys[8], 4u);
    let tmp_2302 = subgroupShuffleXor(values[8], 4u);
    let tmp_2303 = subgroupShuffleXor(keys[9], 4u);
    let tmp_2304 = subgroupShuffleXor(values[9], 4u);
    let tmp_2305 = subgroupShuffleXor(keys[10], 4u);
    let tmp_2306 = subgroupShuffleXor(values[10], 4u);
    let tmp_2307 = subgroupShuffleXor(keys[11], 4u);
    let tmp_2308 = subgroupShuffleXor(values[11], 4u);
    let tmp_2309 = subgroupShuffleXor(keys[12], 4u);
    let tmp_2310 = subgroupShuffleXor(values[12], 4u);
    let tmp_2311 = subgroupShuffleXor(keys[13], 4u);
    let tmp_2312 = subgroupShuffleXor(values[13], 4u);
    let tmp_2313 = subgroupShuffleXor(keys[14], 4u);
    let tmp_2314 = subgroupShuffleXor(values[14], 4u);
    let tmp_2315 = subgroupShuffleXor(keys[15], 4u);
    let tmp_2316 = subgroupShuffleXor(values[15], 4u);
    let tmp_2317 = subgroupShuffleXor(keys[16], 4u);
    let tmp_2318 = subgroupShuffleXor(values[16], 4u);
    let tmp_2319 = subgroupShuffleXor(keys[17], 4u);
    let tmp_2320 = subgroupShuffleXor(values[17], 4u);
    let tmp_2321 = subgroupShuffleXor(keys[18], 4u);
    let tmp_2322 = subgroupShuffleXor(values[18], 4u);
    let tmp_2323 = subgroupShuffleXor(keys[19], 4u);
    let tmp_2324 = subgroupShuffleXor(values[19], 4u);
    let tmp_2325 = subgroupShuffleXor(keys[20], 4u);
    let tmp_2326 = subgroupShuffleXor(values[20], 4u);
    let tmp_2327 = subgroupShuffleXor(keys[21], 4u);
    let tmp_2328 = subgroupShuffleXor(values[21], 4u);
    let tmp_2329 = subgroupShuffleXor(keys[22], 4u);
    let tmp_2330 = subgroupShuffleXor(values[22], 4u);
    let tmp_2331 = subgroupShuffleXor(keys[23], 4u);
    let tmp_2332 = subgroupShuffleXor(values[23], 4u);
    let tmp_2333 = subgroupShuffleXor(keys[24], 4u);
    let tmp_2334 = subgroupShuffleXor(values[24], 4u);
    let tmp_2335 = subgroupShuffleXor(keys[25], 4u);
    let tmp_2336 = subgroupShuffleXor(values[25], 4u);
    let tmp_2337 = subgroupShuffleXor(keys[26], 4u);
    let tmp_2338 = subgroupShuffleXor(values[26], 4u);
    let tmp_2339 = subgroupShuffleXor(keys[27], 4u);
    let tmp_2340 = subgroupShuffleXor(values[27], 4u);
    let tmp_2341 = subgroupShuffleXor(keys[28], 4u);
    let tmp_2342 = subgroupShuffleXor(values[28], 4u);
    let tmp_2343 = subgroupShuffleXor(keys[29], 4u);
    let tmp_2344 = subgroupShuffleXor(values[29], 4u);
    let tmp_2345 = subgroupShuffleXor(keys[30], 4u);
    let tmp_2346 = subgroupShuffleXor(values[30], 4u);
    let tmp_2347 = subgroupShuffleXor(keys[31], 4u);
    let tmp_2348 = subgroupShuffleXor(values[31], 4u);
    let tmp_2349 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_2350 = keys[0] < tmp_2285 || (keys[0] == tmp_2285 && values[0] < tmp_2286);
    if tmp_2349 == tmp_2350 { keys[0] = tmp_2285; values[0] = tmp_2286; }
    let tmp_2351 = keys[1] < tmp_2287 || (keys[1] == tmp_2287 && values[1] < tmp_2288);
    if tmp_2349 == tmp_2351 { keys[1] = tmp_2287; values[1] = tmp_2288; }
    let tmp_2352 = keys[2] < tmp_2289 || (keys[2] == tmp_2289 && values[2] < tmp_2290);
    if tmp_2349 == tmp_2352 { keys[2] = tmp_2289; values[2] = tmp_2290; }
    let tmp_2353 = keys[3] < tmp_2291 || (keys[3] == tmp_2291 && values[3] < tmp_2292);
    if tmp_2349 == tmp_2353 { keys[3] = tmp_2291; values[3] = tmp_2292; }
    let tmp_2354 = keys[4] < tmp_2293 || (keys[4] == tmp_2293 && values[4] < tmp_2294);
    if tmp_2349 == tmp_2354 { keys[4] = tmp_2293; values[4] = tmp_2294; }
    let tmp_2355 = keys[5] < tmp_2295 || (keys[5] == tmp_2295 && values[5] < tmp_2296);
    if tmp_2349 == tmp_2355 { keys[5] = tmp_2295; values[5] = tmp_2296; }
    let tmp_2356 = keys[6] < tmp_2297 || (keys[6] == tmp_2297 && values[6] < tmp_2298);
    if tmp_2349 == tmp_2356 { keys[6] = tmp_2297; values[6] = tmp_2298; }
    let tmp_2357 = keys[7] < tmp_2299 || (keys[7] == tmp_2299 && values[7] < tmp_2300);
    if tmp_2349 == tmp_2357 { keys[7] = tmp_2299; values[7] = tmp_2300; }
    let tmp_2358 = keys[8] < tmp_2301 || (keys[8] == tmp_2301 && values[8] < tmp_2302);
    if tmp_2349 == tmp_2358 { keys[8] = tmp_2301; values[8] = tmp_2302; }
    let tmp_2359 = keys[9] < tmp_2303 || (keys[9] == tmp_2303 && values[9] < tmp_2304);
    if tmp_2349 == tmp_2359 { keys[9] = tmp_2303; values[9] = tmp_2304; }
    let tmp_2360 = keys[10] < tmp_2305 || (keys[10] == tmp_2305 && values[10] < tmp_2306);
    if tmp_2349 == tmp_2360 { keys[10] = tmp_2305; values[10] = tmp_2306; }
    let tmp_2361 = keys[11] < tmp_2307 || (keys[11] == tmp_2307 && values[11] < tmp_2308);
    if tmp_2349 == tmp_2361 { keys[11] = tmp_2307; values[11] = tmp_2308; }
    let tmp_2362 = keys[12] < tmp_2309 || (keys[12] == tmp_2309 && values[12] < tmp_2310);
    if tmp_2349 == tmp_2362 { keys[12] = tmp_2309; values[12] = tmp_2310; }
    let tmp_2363 = keys[13] < tmp_2311 || (keys[13] == tmp_2311 && values[13] < tmp_2312);
    if tmp_2349 == tmp_2363 { keys[13] = tmp_2311; values[13] = tmp_2312; }
    let tmp_2364 = keys[14] < tmp_2313 || (keys[14] == tmp_2313 && values[14] < tmp_2314);
    if tmp_2349 == tmp_2364 { keys[14] = tmp_2313; values[14] = tmp_2314; }
    let tmp_2365 = keys[15] < tmp_2315 || (keys[15] == tmp_2315 && values[15] < tmp_2316);
    if tmp_2349 == tmp_2365 { keys[15] = tmp_2315; values[15] = tmp_2316; }
    let tmp_2366 = keys[16] < tmp_2317 || (keys[16] == tmp_2317 && values[16] < tmp_2318);
    if tmp_2349 == tmp_2366 { keys[16] = tmp_2317; values[16] = tmp_2318; }
    let tmp_2367 = keys[17] < tmp_2319 || (keys[17] == tmp_2319 && values[17] < tmp_2320);
    if tmp_2349 == tmp_2367 { keys[17] = tmp_2319; values[17] = tmp_2320; }
    let tmp_2368 = keys[18] < tmp_2321 || (keys[18] == tmp_2321 && values[18] < tmp_2322);
    if tmp_2349 == tmp_2368 { keys[18] = tmp_2321; values[18] = tmp_2322; }
    let tmp_2369 = keys[19] < tmp_2323 || (keys[19] == tmp_2323 && values[19] < tmp_2324);
    if tmp_2349 == tmp_2369 { keys[19] = tmp_2323; values[19] = tmp_2324; }
    let tmp_2370 = keys[20] < tmp_2325 || (keys[20] == tmp_2325 && values[20] < tmp_2326);
    if tmp_2349 == tmp_2370 { keys[20] = tmp_2325; values[20] = tmp_2326; }
    let tmp_2371 = keys[21] < tmp_2327 || (keys[21] == tmp_2327 && values[21] < tmp_2328);
    if tmp_2349 == tmp_2371 { keys[21] = tmp_2327; values[21] = tmp_2328; }
    let tmp_2372 = keys[22] < tmp_2329 || (keys[22] == tmp_2329 && values[22] < tmp_2330);
    if tmp_2349 == tmp_2372 { keys[22] = tmp_2329; values[22] = tmp_2330; }
    let tmp_2373 = keys[23] < tmp_2331 || (keys[23] == tmp_2331 && values[23] < tmp_2332);
    if tmp_2349 == tmp_2373 { keys[23] = tmp_2331; values[23] = tmp_2332; }
    let tmp_2374 = keys[24] < tmp_2333 || (keys[24] == tmp_2333 && values[24] < tmp_2334);
    if tmp_2349 == tmp_2374 { keys[24] = tmp_2333; values[24] = tmp_2334; }
    let tmp_2375 = keys[25] < tmp_2335 || (keys[25] == tmp_2335 && values[25] < tmp_2336);
    if tmp_2349 == tmp_2375 { keys[25] = tmp_2335; values[25] = tmp_2336; }
    let tmp_2376 = keys[26] < tmp_2337 || (keys[26] == tmp_2337 && values[26] < tmp_2338);
    if tmp_2349 == tmp_2376 { keys[26] = tmp_2337; values[26] = tmp_2338; }
    let tmp_2377 = keys[27] < tmp_2339 || (keys[27] == tmp_2339 && values[27] < tmp_2340);
    if tmp_2349 == tmp_2377 { keys[27] = tmp_2339; values[27] = tmp_2340; }
    let tmp_2378 = keys[28] < tmp_2341 || (keys[28] == tmp_2341 && values[28] < tmp_2342);
    if tmp_2349 == tmp_2378 { keys[28] = tmp_2341; values[28] = tmp_2342; }
    let tmp_2379 = keys[29] < tmp_2343 || (keys[29] == tmp_2343 && values[29] < tmp_2344);
    if tmp_2349 == tmp_2379 { keys[29] = tmp_2343; values[29] = tmp_2344; }
    let tmp_2380 = keys[30] < tmp_2345 || (keys[30] == tmp_2345 && values[30] < tmp_2346);
    if tmp_2349 == tmp_2380 { keys[30] = tmp_2345; values[30] = tmp_2346; }
    let tmp_2381 = keys[31] < tmp_2347 || (keys[31] == tmp_2347 && values[31] < tmp_2348);
    if tmp_2349 == tmp_2381 { keys[31] = tmp_2347; values[31] = tmp_2348; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:32) 
    {
    let tmp_2382 = subgroupShuffleXor(keys[0], 2u);
    let tmp_2383 = subgroupShuffleXor(values[0], 2u);
    let tmp_2384 = subgroupShuffleXor(keys[1], 2u);
    let tmp_2385 = subgroupShuffleXor(values[1], 2u);
    let tmp_2386 = subgroupShuffleXor(keys[2], 2u);
    let tmp_2387 = subgroupShuffleXor(values[2], 2u);
    let tmp_2388 = subgroupShuffleXor(keys[3], 2u);
    let tmp_2389 = subgroupShuffleXor(values[3], 2u);
    let tmp_2390 = subgroupShuffleXor(keys[4], 2u);
    let tmp_2391 = subgroupShuffleXor(values[4], 2u);
    let tmp_2392 = subgroupShuffleXor(keys[5], 2u);
    let tmp_2393 = subgroupShuffleXor(values[5], 2u);
    let tmp_2394 = subgroupShuffleXor(keys[6], 2u);
    let tmp_2395 = subgroupShuffleXor(values[6], 2u);
    let tmp_2396 = subgroupShuffleXor(keys[7], 2u);
    let tmp_2397 = subgroupShuffleXor(values[7], 2u);
    let tmp_2398 = subgroupShuffleXor(keys[8], 2u);
    let tmp_2399 = subgroupShuffleXor(values[8], 2u);
    let tmp_2400 = subgroupShuffleXor(keys[9], 2u);
    let tmp_2401 = subgroupShuffleXor(values[9], 2u);
    let tmp_2402 = subgroupShuffleXor(keys[10], 2u);
    let tmp_2403 = subgroupShuffleXor(values[10], 2u);
    let tmp_2404 = subgroupShuffleXor(keys[11], 2u);
    let tmp_2405 = subgroupShuffleXor(values[11], 2u);
    let tmp_2406 = subgroupShuffleXor(keys[12], 2u);
    let tmp_2407 = subgroupShuffleXor(values[12], 2u);
    let tmp_2408 = subgroupShuffleXor(keys[13], 2u);
    let tmp_2409 = subgroupShuffleXor(values[13], 2u);
    let tmp_2410 = subgroupShuffleXor(keys[14], 2u);
    let tmp_2411 = subgroupShuffleXor(values[14], 2u);
    let tmp_2412 = subgroupShuffleXor(keys[15], 2u);
    let tmp_2413 = subgroupShuffleXor(values[15], 2u);
    let tmp_2414 = subgroupShuffleXor(keys[16], 2u);
    let tmp_2415 = subgroupShuffleXor(values[16], 2u);
    let tmp_2416 = subgroupShuffleXor(keys[17], 2u);
    let tmp_2417 = subgroupShuffleXor(values[17], 2u);
    let tmp_2418 = subgroupShuffleXor(keys[18], 2u);
    let tmp_2419 = subgroupShuffleXor(values[18], 2u);
    let tmp_2420 = subgroupShuffleXor(keys[19], 2u);
    let tmp_2421 = subgroupShuffleXor(values[19], 2u);
    let tmp_2422 = subgroupShuffleXor(keys[20], 2u);
    let tmp_2423 = subgroupShuffleXor(values[20], 2u);
    let tmp_2424 = subgroupShuffleXor(keys[21], 2u);
    let tmp_2425 = subgroupShuffleXor(values[21], 2u);
    let tmp_2426 = subgroupShuffleXor(keys[22], 2u);
    let tmp_2427 = subgroupShuffleXor(values[22], 2u);
    let tmp_2428 = subgroupShuffleXor(keys[23], 2u);
    let tmp_2429 = subgroupShuffleXor(values[23], 2u);
    let tmp_2430 = subgroupShuffleXor(keys[24], 2u);
    let tmp_2431 = subgroupShuffleXor(values[24], 2u);
    let tmp_2432 = subgroupShuffleXor(keys[25], 2u);
    let tmp_2433 = subgroupShuffleXor(values[25], 2u);
    let tmp_2434 = subgroupShuffleXor(keys[26], 2u);
    let tmp_2435 = subgroupShuffleXor(values[26], 2u);
    let tmp_2436 = subgroupShuffleXor(keys[27], 2u);
    let tmp_2437 = subgroupShuffleXor(values[27], 2u);
    let tmp_2438 = subgroupShuffleXor(keys[28], 2u);
    let tmp_2439 = subgroupShuffleXor(values[28], 2u);
    let tmp_2440 = subgroupShuffleXor(keys[29], 2u);
    let tmp_2441 = subgroupShuffleXor(values[29], 2u);
    let tmp_2442 = subgroupShuffleXor(keys[30], 2u);
    let tmp_2443 = subgroupShuffleXor(values[30], 2u);
    let tmp_2444 = subgroupShuffleXor(keys[31], 2u);
    let tmp_2445 = subgroupShuffleXor(values[31], 2u);
    let tmp_2446 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_2447 = keys[0] < tmp_2382 || (keys[0] == tmp_2382 && values[0] < tmp_2383);
    if tmp_2446 == tmp_2447 { keys[0] = tmp_2382; values[0] = tmp_2383; }
    let tmp_2448 = keys[1] < tmp_2384 || (keys[1] == tmp_2384 && values[1] < tmp_2385);
    if tmp_2446 == tmp_2448 { keys[1] = tmp_2384; values[1] = tmp_2385; }
    let tmp_2449 = keys[2] < tmp_2386 || (keys[2] == tmp_2386 && values[2] < tmp_2387);
    if tmp_2446 == tmp_2449 { keys[2] = tmp_2386; values[2] = tmp_2387; }
    let tmp_2450 = keys[3] < tmp_2388 || (keys[3] == tmp_2388 && values[3] < tmp_2389);
    if tmp_2446 == tmp_2450 { keys[3] = tmp_2388; values[3] = tmp_2389; }
    let tmp_2451 = keys[4] < tmp_2390 || (keys[4] == tmp_2390 && values[4] < tmp_2391);
    if tmp_2446 == tmp_2451 { keys[4] = tmp_2390; values[4] = tmp_2391; }
    let tmp_2452 = keys[5] < tmp_2392 || (keys[5] == tmp_2392 && values[5] < tmp_2393);
    if tmp_2446 == tmp_2452 { keys[5] = tmp_2392; values[5] = tmp_2393; }
    let tmp_2453 = keys[6] < tmp_2394 || (keys[6] == tmp_2394 && values[6] < tmp_2395);
    if tmp_2446 == tmp_2453 { keys[6] = tmp_2394; values[6] = tmp_2395; }
    let tmp_2454 = keys[7] < tmp_2396 || (keys[7] == tmp_2396 && values[7] < tmp_2397);
    if tmp_2446 == tmp_2454 { keys[7] = tmp_2396; values[7] = tmp_2397; }
    let tmp_2455 = keys[8] < tmp_2398 || (keys[8] == tmp_2398 && values[8] < tmp_2399);
    if tmp_2446 == tmp_2455 { keys[8] = tmp_2398; values[8] = tmp_2399; }
    let tmp_2456 = keys[9] < tmp_2400 || (keys[9] == tmp_2400 && values[9] < tmp_2401);
    if tmp_2446 == tmp_2456 { keys[9] = tmp_2400; values[9] = tmp_2401; }
    let tmp_2457 = keys[10] < tmp_2402 || (keys[10] == tmp_2402 && values[10] < tmp_2403);
    if tmp_2446 == tmp_2457 { keys[10] = tmp_2402; values[10] = tmp_2403; }
    let tmp_2458 = keys[11] < tmp_2404 || (keys[11] == tmp_2404 && values[11] < tmp_2405);
    if tmp_2446 == tmp_2458 { keys[11] = tmp_2404; values[11] = tmp_2405; }
    let tmp_2459 = keys[12] < tmp_2406 || (keys[12] == tmp_2406 && values[12] < tmp_2407);
    if tmp_2446 == tmp_2459 { keys[12] = tmp_2406; values[12] = tmp_2407; }
    let tmp_2460 = keys[13] < tmp_2408 || (keys[13] == tmp_2408 && values[13] < tmp_2409);
    if tmp_2446 == tmp_2460 { keys[13] = tmp_2408; values[13] = tmp_2409; }
    let tmp_2461 = keys[14] < tmp_2410 || (keys[14] == tmp_2410 && values[14] < tmp_2411);
    if tmp_2446 == tmp_2461 { keys[14] = tmp_2410; values[14] = tmp_2411; }
    let tmp_2462 = keys[15] < tmp_2412 || (keys[15] == tmp_2412 && values[15] < tmp_2413);
    if tmp_2446 == tmp_2462 { keys[15] = tmp_2412; values[15] = tmp_2413; }
    let tmp_2463 = keys[16] < tmp_2414 || (keys[16] == tmp_2414 && values[16] < tmp_2415);
    if tmp_2446 == tmp_2463 { keys[16] = tmp_2414; values[16] = tmp_2415; }
    let tmp_2464 = keys[17] < tmp_2416 || (keys[17] == tmp_2416 && values[17] < tmp_2417);
    if tmp_2446 == tmp_2464 { keys[17] = tmp_2416; values[17] = tmp_2417; }
    let tmp_2465 = keys[18] < tmp_2418 || (keys[18] == tmp_2418 && values[18] < tmp_2419);
    if tmp_2446 == tmp_2465 { keys[18] = tmp_2418; values[18] = tmp_2419; }
    let tmp_2466 = keys[19] < tmp_2420 || (keys[19] == tmp_2420 && values[19] < tmp_2421);
    if tmp_2446 == tmp_2466 { keys[19] = tmp_2420; values[19] = tmp_2421; }
    let tmp_2467 = keys[20] < tmp_2422 || (keys[20] == tmp_2422 && values[20] < tmp_2423);
    if tmp_2446 == tmp_2467 { keys[20] = tmp_2422; values[20] = tmp_2423; }
    let tmp_2468 = keys[21] < tmp_2424 || (keys[21] == tmp_2424 && values[21] < tmp_2425);
    if tmp_2446 == tmp_2468 { keys[21] = tmp_2424; values[21] = tmp_2425; }
    let tmp_2469 = keys[22] < tmp_2426 || (keys[22] == tmp_2426 && values[22] < tmp_2427);
    if tmp_2446 == tmp_2469 { keys[22] = tmp_2426; values[22] = tmp_2427; }
    let tmp_2470 = keys[23] < tmp_2428 || (keys[23] == tmp_2428 && values[23] < tmp_2429);
    if tmp_2446 == tmp_2470 { keys[23] = tmp_2428; values[23] = tmp_2429; }
    let tmp_2471 = keys[24] < tmp_2430 || (keys[24] == tmp_2430 && values[24] < tmp_2431);
    if tmp_2446 == tmp_2471 { keys[24] = tmp_2430; values[24] = tmp_2431; }
    let tmp_2472 = keys[25] < tmp_2432 || (keys[25] == tmp_2432 && values[25] < tmp_2433);
    if tmp_2446 == tmp_2472 { keys[25] = tmp_2432; values[25] = tmp_2433; }
    let tmp_2473 = keys[26] < tmp_2434 || (keys[26] == tmp_2434 && values[26] < tmp_2435);
    if tmp_2446 == tmp_2473 { keys[26] = tmp_2434; values[26] = tmp_2435; }
    let tmp_2474 = keys[27] < tmp_2436 || (keys[27] == tmp_2436 && values[27] < tmp_2437);
    if tmp_2446 == tmp_2474 { keys[27] = tmp_2436; values[27] = tmp_2437; }
    let tmp_2475 = keys[28] < tmp_2438 || (keys[28] == tmp_2438 && values[28] < tmp_2439);
    if tmp_2446 == tmp_2475 { keys[28] = tmp_2438; values[28] = tmp_2439; }
    let tmp_2476 = keys[29] < tmp_2440 || (keys[29] == tmp_2440 && values[29] < tmp_2441);
    if tmp_2446 == tmp_2476 { keys[29] = tmp_2440; values[29] = tmp_2441; }
    let tmp_2477 = keys[30] < tmp_2442 || (keys[30] == tmp_2442 && values[30] < tmp_2443);
    if tmp_2446 == tmp_2477 { keys[30] = tmp_2442; values[30] = tmp_2443; }
    let tmp_2478 = keys[31] < tmp_2444 || (keys[31] == tmp_2444 && values[31] < tmp_2445);
    if tmp_2446 == tmp_2478 { keys[31] = tmp_2444; values[31] = tmp_2445; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:32) 
    {
    let tmp_2479 = subgroupShuffleXor(keys[0], 1u);
    let tmp_2480 = subgroupShuffleXor(values[0], 1u);
    let tmp_2481 = subgroupShuffleXor(keys[1], 1u);
    let tmp_2482 = subgroupShuffleXor(values[1], 1u);
    let tmp_2483 = subgroupShuffleXor(keys[2], 1u);
    let tmp_2484 = subgroupShuffleXor(values[2], 1u);
    let tmp_2485 = subgroupShuffleXor(keys[3], 1u);
    let tmp_2486 = subgroupShuffleXor(values[3], 1u);
    let tmp_2487 = subgroupShuffleXor(keys[4], 1u);
    let tmp_2488 = subgroupShuffleXor(values[4], 1u);
    let tmp_2489 = subgroupShuffleXor(keys[5], 1u);
    let tmp_2490 = subgroupShuffleXor(values[5], 1u);
    let tmp_2491 = subgroupShuffleXor(keys[6], 1u);
    let tmp_2492 = subgroupShuffleXor(values[6], 1u);
    let tmp_2493 = subgroupShuffleXor(keys[7], 1u);
    let tmp_2494 = subgroupShuffleXor(values[7], 1u);
    let tmp_2495 = subgroupShuffleXor(keys[8], 1u);
    let tmp_2496 = subgroupShuffleXor(values[8], 1u);
    let tmp_2497 = subgroupShuffleXor(keys[9], 1u);
    let tmp_2498 = subgroupShuffleXor(values[9], 1u);
    let tmp_2499 = subgroupShuffleXor(keys[10], 1u);
    let tmp_2500 = subgroupShuffleXor(values[10], 1u);
    let tmp_2501 = subgroupShuffleXor(keys[11], 1u);
    let tmp_2502 = subgroupShuffleXor(values[11], 1u);
    let tmp_2503 = subgroupShuffleXor(keys[12], 1u);
    let tmp_2504 = subgroupShuffleXor(values[12], 1u);
    let tmp_2505 = subgroupShuffleXor(keys[13], 1u);
    let tmp_2506 = subgroupShuffleXor(values[13], 1u);
    let tmp_2507 = subgroupShuffleXor(keys[14], 1u);
    let tmp_2508 = subgroupShuffleXor(values[14], 1u);
    let tmp_2509 = subgroupShuffleXor(keys[15], 1u);
    let tmp_2510 = subgroupShuffleXor(values[15], 1u);
    let tmp_2511 = subgroupShuffleXor(keys[16], 1u);
    let tmp_2512 = subgroupShuffleXor(values[16], 1u);
    let tmp_2513 = subgroupShuffleXor(keys[17], 1u);
    let tmp_2514 = subgroupShuffleXor(values[17], 1u);
    let tmp_2515 = subgroupShuffleXor(keys[18], 1u);
    let tmp_2516 = subgroupShuffleXor(values[18], 1u);
    let tmp_2517 = subgroupShuffleXor(keys[19], 1u);
    let tmp_2518 = subgroupShuffleXor(values[19], 1u);
    let tmp_2519 = subgroupShuffleXor(keys[20], 1u);
    let tmp_2520 = subgroupShuffleXor(values[20], 1u);
    let tmp_2521 = subgroupShuffleXor(keys[21], 1u);
    let tmp_2522 = subgroupShuffleXor(values[21], 1u);
    let tmp_2523 = subgroupShuffleXor(keys[22], 1u);
    let tmp_2524 = subgroupShuffleXor(values[22], 1u);
    let tmp_2525 = subgroupShuffleXor(keys[23], 1u);
    let tmp_2526 = subgroupShuffleXor(values[23], 1u);
    let tmp_2527 = subgroupShuffleXor(keys[24], 1u);
    let tmp_2528 = subgroupShuffleXor(values[24], 1u);
    let tmp_2529 = subgroupShuffleXor(keys[25], 1u);
    let tmp_2530 = subgroupShuffleXor(values[25], 1u);
    let tmp_2531 = subgroupShuffleXor(keys[26], 1u);
    let tmp_2532 = subgroupShuffleXor(values[26], 1u);
    let tmp_2533 = subgroupShuffleXor(keys[27], 1u);
    let tmp_2534 = subgroupShuffleXor(values[27], 1u);
    let tmp_2535 = subgroupShuffleXor(keys[28], 1u);
    let tmp_2536 = subgroupShuffleXor(values[28], 1u);
    let tmp_2537 = subgroupShuffleXor(keys[29], 1u);
    let tmp_2538 = subgroupShuffleXor(values[29], 1u);
    let tmp_2539 = subgroupShuffleXor(keys[30], 1u);
    let tmp_2540 = subgroupShuffleXor(values[30], 1u);
    let tmp_2541 = subgroupShuffleXor(keys[31], 1u);
    let tmp_2542 = subgroupShuffleXor(values[31], 1u);
    let tmp_2543 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_2544 = keys[0] < tmp_2479 || (keys[0] == tmp_2479 && values[0] < tmp_2480);
    if tmp_2543 == tmp_2544 { keys[0] = tmp_2479; values[0] = tmp_2480; }
    let tmp_2545 = keys[1] < tmp_2481 || (keys[1] == tmp_2481 && values[1] < tmp_2482);
    if tmp_2543 == tmp_2545 { keys[1] = tmp_2481; values[1] = tmp_2482; }
    let tmp_2546 = keys[2] < tmp_2483 || (keys[2] == tmp_2483 && values[2] < tmp_2484);
    if tmp_2543 == tmp_2546 { keys[2] = tmp_2483; values[2] = tmp_2484; }
    let tmp_2547 = keys[3] < tmp_2485 || (keys[3] == tmp_2485 && values[3] < tmp_2486);
    if tmp_2543 == tmp_2547 { keys[3] = tmp_2485; values[3] = tmp_2486; }
    let tmp_2548 = keys[4] < tmp_2487 || (keys[4] == tmp_2487 && values[4] < tmp_2488);
    if tmp_2543 == tmp_2548 { keys[4] = tmp_2487; values[4] = tmp_2488; }
    let tmp_2549 = keys[5] < tmp_2489 || (keys[5] == tmp_2489 && values[5] < tmp_2490);
    if tmp_2543 == tmp_2549 { keys[5] = tmp_2489; values[5] = tmp_2490; }
    let tmp_2550 = keys[6] < tmp_2491 || (keys[6] == tmp_2491 && values[6] < tmp_2492);
    if tmp_2543 == tmp_2550 { keys[6] = tmp_2491; values[6] = tmp_2492; }
    let tmp_2551 = keys[7] < tmp_2493 || (keys[7] == tmp_2493 && values[7] < tmp_2494);
    if tmp_2543 == tmp_2551 { keys[7] = tmp_2493; values[7] = tmp_2494; }
    let tmp_2552 = keys[8] < tmp_2495 || (keys[8] == tmp_2495 && values[8] < tmp_2496);
    if tmp_2543 == tmp_2552 { keys[8] = tmp_2495; values[8] = tmp_2496; }
    let tmp_2553 = keys[9] < tmp_2497 || (keys[9] == tmp_2497 && values[9] < tmp_2498);
    if tmp_2543 == tmp_2553 { keys[9] = tmp_2497; values[9] = tmp_2498; }
    let tmp_2554 = keys[10] < tmp_2499 || (keys[10] == tmp_2499 && values[10] < tmp_2500);
    if tmp_2543 == tmp_2554 { keys[10] = tmp_2499; values[10] = tmp_2500; }
    let tmp_2555 = keys[11] < tmp_2501 || (keys[11] == tmp_2501 && values[11] < tmp_2502);
    if tmp_2543 == tmp_2555 { keys[11] = tmp_2501; values[11] = tmp_2502; }
    let tmp_2556 = keys[12] < tmp_2503 || (keys[12] == tmp_2503 && values[12] < tmp_2504);
    if tmp_2543 == tmp_2556 { keys[12] = tmp_2503; values[12] = tmp_2504; }
    let tmp_2557 = keys[13] < tmp_2505 || (keys[13] == tmp_2505 && values[13] < tmp_2506);
    if tmp_2543 == tmp_2557 { keys[13] = tmp_2505; values[13] = tmp_2506; }
    let tmp_2558 = keys[14] < tmp_2507 || (keys[14] == tmp_2507 && values[14] < tmp_2508);
    if tmp_2543 == tmp_2558 { keys[14] = tmp_2507; values[14] = tmp_2508; }
    let tmp_2559 = keys[15] < tmp_2509 || (keys[15] == tmp_2509 && values[15] < tmp_2510);
    if tmp_2543 == tmp_2559 { keys[15] = tmp_2509; values[15] = tmp_2510; }
    let tmp_2560 = keys[16] < tmp_2511 || (keys[16] == tmp_2511 && values[16] < tmp_2512);
    if tmp_2543 == tmp_2560 { keys[16] = tmp_2511; values[16] = tmp_2512; }
    let tmp_2561 = keys[17] < tmp_2513 || (keys[17] == tmp_2513 && values[17] < tmp_2514);
    if tmp_2543 == tmp_2561 { keys[17] = tmp_2513; values[17] = tmp_2514; }
    let tmp_2562 = keys[18] < tmp_2515 || (keys[18] == tmp_2515 && values[18] < tmp_2516);
    if tmp_2543 == tmp_2562 { keys[18] = tmp_2515; values[18] = tmp_2516; }
    let tmp_2563 = keys[19] < tmp_2517 || (keys[19] == tmp_2517 && values[19] < tmp_2518);
    if tmp_2543 == tmp_2563 { keys[19] = tmp_2517; values[19] = tmp_2518; }
    let tmp_2564 = keys[20] < tmp_2519 || (keys[20] == tmp_2519 && values[20] < tmp_2520);
    if tmp_2543 == tmp_2564 { keys[20] = tmp_2519; values[20] = tmp_2520; }
    let tmp_2565 = keys[21] < tmp_2521 || (keys[21] == tmp_2521 && values[21] < tmp_2522);
    if tmp_2543 == tmp_2565 { keys[21] = tmp_2521; values[21] = tmp_2522; }
    let tmp_2566 = keys[22] < tmp_2523 || (keys[22] == tmp_2523 && values[22] < tmp_2524);
    if tmp_2543 == tmp_2566 { keys[22] = tmp_2523; values[22] = tmp_2524; }
    let tmp_2567 = keys[23] < tmp_2525 || (keys[23] == tmp_2525 && values[23] < tmp_2526);
    if tmp_2543 == tmp_2567 { keys[23] = tmp_2525; values[23] = tmp_2526; }
    let tmp_2568 = keys[24] < tmp_2527 || (keys[24] == tmp_2527 && values[24] < tmp_2528);
    if tmp_2543 == tmp_2568 { keys[24] = tmp_2527; values[24] = tmp_2528; }
    let tmp_2569 = keys[25] < tmp_2529 || (keys[25] == tmp_2529 && values[25] < tmp_2530);
    if tmp_2543 == tmp_2569 { keys[25] = tmp_2529; values[25] = tmp_2530; }
    let tmp_2570 = keys[26] < tmp_2531 || (keys[26] == tmp_2531 && values[26] < tmp_2532);
    if tmp_2543 == tmp_2570 { keys[26] = tmp_2531; values[26] = tmp_2532; }
    let tmp_2571 = keys[27] < tmp_2533 || (keys[27] == tmp_2533 && values[27] < tmp_2534);
    if tmp_2543 == tmp_2571 { keys[27] = tmp_2533; values[27] = tmp_2534; }
    let tmp_2572 = keys[28] < tmp_2535 || (keys[28] == tmp_2535 && values[28] < tmp_2536);
    if tmp_2543 == tmp_2572 { keys[28] = tmp_2535; values[28] = tmp_2536; }
    let tmp_2573 = keys[29] < tmp_2537 || (keys[29] == tmp_2537 && values[29] < tmp_2538);
    if tmp_2543 == tmp_2573 { keys[29] = tmp_2537; values[29] = tmp_2538; }
    let tmp_2574 = keys[30] < tmp_2539 || (keys[30] == tmp_2539 && values[30] < tmp_2540);
    if tmp_2543 == tmp_2574 { keys[30] = tmp_2539; values[30] = tmp_2540; }
    let tmp_2575 = keys[31] < tmp_2541 || (keys[31] == tmp_2541 && values[31] < tmp_2542);
    if tmp_2543 == tmp_2575 { keys[31] = tmp_2541; values[31] = tmp_2542; }
    }
    // exch_local(16,32) 
    // cmp_swap(0,16)
    if keys[0] > keys[16] || (keys[0] == keys[16] && values[0] > values[16]) {
    // swap(0,16) 
    { let tmp_2576 = keys[0]; keys[0] = keys[16]; keys[16] = tmp_2576;let tmp_2577 = values[0]; values[0] = values[16]; values[16] = tmp_2577; }
    }
    // cmp_swap(1,17)
    if keys[1] > keys[17] || (keys[1] == keys[17] && values[1] > values[17]) {
    // swap(1,17) 
    { let tmp_2578 = keys[1]; keys[1] = keys[17]; keys[17] = tmp_2578;let tmp_2579 = values[1]; values[1] = values[17]; values[17] = tmp_2579; }
    }
    // cmp_swap(2,18)
    if keys[2] > keys[18] || (keys[2] == keys[18] && values[2] > values[18]) {
    // swap(2,18) 
    { let tmp_2580 = keys[2]; keys[2] = keys[18]; keys[18] = tmp_2580;let tmp_2581 = values[2]; values[2] = values[18]; values[18] = tmp_2581; }
    }
    // cmp_swap(3,19)
    if keys[3] > keys[19] || (keys[3] == keys[19] && values[3] > values[19]) {
    // swap(3,19) 
    { let tmp_2582 = keys[3]; keys[3] = keys[19]; keys[19] = tmp_2582;let tmp_2583 = values[3]; values[3] = values[19]; values[19] = tmp_2583; }
    }
    // cmp_swap(4,20)
    if keys[4] > keys[20] || (keys[4] == keys[20] && values[4] > values[20]) {
    // swap(4,20) 
    { let tmp_2584 = keys[4]; keys[4] = keys[20]; keys[20] = tmp_2584;let tmp_2585 = values[4]; values[4] = values[20]; values[20] = tmp_2585; }
    }
    // cmp_swap(5,21)
    if keys[5] > keys[21] || (keys[5] == keys[21] && values[5] > values[21]) {
    // swap(5,21) 
    { let tmp_2586 = keys[5]; keys[5] = keys[21]; keys[21] = tmp_2586;let tmp_2587 = values[5]; values[5] = values[21]; values[21] = tmp_2587; }
    }
    // cmp_swap(6,22)
    if keys[6] > keys[22] || (keys[6] == keys[22] && values[6] > values[22]) {
    // swap(6,22) 
    { let tmp_2588 = keys[6]; keys[6] = keys[22]; keys[22] = tmp_2588;let tmp_2589 = values[6]; values[6] = values[22]; values[22] = tmp_2589; }
    }
    // cmp_swap(7,23)
    if keys[7] > keys[23] || (keys[7] == keys[23] && values[7] > values[23]) {
    // swap(7,23) 
    { let tmp_2590 = keys[7]; keys[7] = keys[23]; keys[23] = tmp_2590;let tmp_2591 = values[7]; values[7] = values[23]; values[23] = tmp_2591; }
    }
    // cmp_swap(8,24)
    if keys[8] > keys[24] || (keys[8] == keys[24] && values[8] > values[24]) {
    // swap(8,24) 
    { let tmp_2592 = keys[8]; keys[8] = keys[24]; keys[24] = tmp_2592;let tmp_2593 = values[8]; values[8] = values[24]; values[24] = tmp_2593; }
    }
    // cmp_swap(9,25)
    if keys[9] > keys[25] || (keys[9] == keys[25] && values[9] > values[25]) {
    // swap(9,25) 
    { let tmp_2594 = keys[9]; keys[9] = keys[25]; keys[25] = tmp_2594;let tmp_2595 = values[9]; values[9] = values[25]; values[25] = tmp_2595; }
    }
    // cmp_swap(10,26)
    if keys[10] > keys[26] || (keys[10] == keys[26] && values[10] > values[26]) {
    // swap(10,26) 
    { let tmp_2596 = keys[10]; keys[10] = keys[26]; keys[26] = tmp_2596;let tmp_2597 = values[10]; values[10] = values[26]; values[26] = tmp_2597; }
    }
    // cmp_swap(11,27)
    if keys[11] > keys[27] || (keys[11] == keys[27] && values[11] > values[27]) {
    // swap(11,27) 
    { let tmp_2598 = keys[11]; keys[11] = keys[27]; keys[27] = tmp_2598;let tmp_2599 = values[11]; values[11] = values[27]; values[27] = tmp_2599; }
    }
    // cmp_swap(12,28)
    if keys[12] > keys[28] || (keys[12] == keys[28] && values[12] > values[28]) {
    // swap(12,28) 
    { let tmp_2600 = keys[12]; keys[12] = keys[28]; keys[28] = tmp_2600;let tmp_2601 = values[12]; values[12] = values[28]; values[28] = tmp_2601; }
    }
    // cmp_swap(13,29)
    if keys[13] > keys[29] || (keys[13] == keys[29] && values[13] > values[29]) {
    // swap(13,29) 
    { let tmp_2602 = keys[13]; keys[13] = keys[29]; keys[29] = tmp_2602;let tmp_2603 = values[13]; values[13] = values[29]; values[29] = tmp_2603; }
    }
    // cmp_swap(14,30)
    if keys[14] > keys[30] || (keys[14] == keys[30] && values[14] > values[30]) {
    // swap(14,30) 
    { let tmp_2604 = keys[14]; keys[14] = keys[30]; keys[30] = tmp_2604;let tmp_2605 = values[14]; values[14] = values[30]; values[30] = tmp_2605; }
    }
    // cmp_swap(15,31)
    if keys[15] > keys[31] || (keys[15] == keys[31] && values[15] > values[31]) {
    // swap(15,31) 
    { let tmp_2606 = keys[15]; keys[15] = keys[31]; keys[31] = tmp_2606;let tmp_2607 = values[15]; values[15] = values[31]; values[31] = tmp_2607; }
    }
    // exch_local(8,32) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_2608 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_2608;let tmp_2609 = values[0]; values[0] = values[8]; values[8] = tmp_2609; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_2610 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_2610;let tmp_2611 = values[1]; values[1] = values[9]; values[9] = tmp_2611; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_2612 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_2612;let tmp_2613 = values[2]; values[2] = values[10]; values[10] = tmp_2613; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_2614 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_2614;let tmp_2615 = values[3]; values[3] = values[11]; values[11] = tmp_2615; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_2616 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_2616;let tmp_2617 = values[4]; values[4] = values[12]; values[12] = tmp_2617; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_2618 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_2618;let tmp_2619 = values[5]; values[5] = values[13]; values[13] = tmp_2619; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_2620 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_2620;let tmp_2621 = values[6]; values[6] = values[14]; values[14] = tmp_2621; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_2622 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_2622;let tmp_2623 = values[7]; values[7] = values[15]; values[15] = tmp_2623; }
    }
    // cmp_swap(16,24)
    if keys[16] > keys[24] || (keys[16] == keys[24] && values[16] > values[24]) {
    // swap(16,24) 
    { let tmp_2624 = keys[16]; keys[16] = keys[24]; keys[24] = tmp_2624;let tmp_2625 = values[16]; values[16] = values[24]; values[24] = tmp_2625; }
    }
    // cmp_swap(17,25)
    if keys[17] > keys[25] || (keys[17] == keys[25] && values[17] > values[25]) {
    // swap(17,25) 
    { let tmp_2626 = keys[17]; keys[17] = keys[25]; keys[25] = tmp_2626;let tmp_2627 = values[17]; values[17] = values[25]; values[25] = tmp_2627; }
    }
    // cmp_swap(18,26)
    if keys[18] > keys[26] || (keys[18] == keys[26] && values[18] > values[26]) {
    // swap(18,26) 
    { let tmp_2628 = keys[18]; keys[18] = keys[26]; keys[26] = tmp_2628;let tmp_2629 = values[18]; values[18] = values[26]; values[26] = tmp_2629; }
    }
    // cmp_swap(19,27)
    if keys[19] > keys[27] || (keys[19] == keys[27] && values[19] > values[27]) {
    // swap(19,27) 
    { let tmp_2630 = keys[19]; keys[19] = keys[27]; keys[27] = tmp_2630;let tmp_2631 = values[19]; values[19] = values[27]; values[27] = tmp_2631; }
    }
    // cmp_swap(20,28)
    if keys[20] > keys[28] || (keys[20] == keys[28] && values[20] > values[28]) {
    // swap(20,28) 
    { let tmp_2632 = keys[20]; keys[20] = keys[28]; keys[28] = tmp_2632;let tmp_2633 = values[20]; values[20] = values[28]; values[28] = tmp_2633; }
    }
    // cmp_swap(21,29)
    if keys[21] > keys[29] || (keys[21] == keys[29] && values[21] > values[29]) {
    // swap(21,29) 
    { let tmp_2634 = keys[21]; keys[21] = keys[29]; keys[29] = tmp_2634;let tmp_2635 = values[21]; values[21] = values[29]; values[29] = tmp_2635; }
    }
    // cmp_swap(22,30)
    if keys[22] > keys[30] || (keys[22] == keys[30] && values[22] > values[30]) {
    // swap(22,30) 
    { let tmp_2636 = keys[22]; keys[22] = keys[30]; keys[30] = tmp_2636;let tmp_2637 = values[22]; values[22] = values[30]; values[30] = tmp_2637; }
    }
    // cmp_swap(23,31)
    if keys[23] > keys[31] || (keys[23] == keys[31] && values[23] > values[31]) {
    // swap(23,31) 
    { let tmp_2638 = keys[23]; keys[23] = keys[31]; keys[31] = tmp_2638;let tmp_2639 = values[23]; values[23] = values[31]; values[31] = tmp_2639; }
    }
    // exch_local(4,32) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_2640 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_2640;let tmp_2641 = values[0]; values[0] = values[4]; values[4] = tmp_2641; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_2642 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_2642;let tmp_2643 = values[1]; values[1] = values[5]; values[5] = tmp_2643; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_2644 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_2644;let tmp_2645 = values[2]; values[2] = values[6]; values[6] = tmp_2645; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_2646 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_2646;let tmp_2647 = values[3]; values[3] = values[7]; values[7] = tmp_2647; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_2648 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_2648;let tmp_2649 = values[8]; values[8] = values[12]; values[12] = tmp_2649; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_2650 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_2650;let tmp_2651 = values[9]; values[9] = values[13]; values[13] = tmp_2651; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_2652 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_2652;let tmp_2653 = values[10]; values[10] = values[14]; values[14] = tmp_2653; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_2654 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_2654;let tmp_2655 = values[11]; values[11] = values[15]; values[15] = tmp_2655; }
    }
    // cmp_swap(16,20)
    if keys[16] > keys[20] || (keys[16] == keys[20] && values[16] > values[20]) {
    // swap(16,20) 
    { let tmp_2656 = keys[16]; keys[16] = keys[20]; keys[20] = tmp_2656;let tmp_2657 = values[16]; values[16] = values[20]; values[20] = tmp_2657; }
    }
    // cmp_swap(17,21)
    if keys[17] > keys[21] || (keys[17] == keys[21] && values[17] > values[21]) {
    // swap(17,21) 
    { let tmp_2658 = keys[17]; keys[17] = keys[21]; keys[21] = tmp_2658;let tmp_2659 = values[17]; values[17] = values[21]; values[21] = tmp_2659; }
    }
    // cmp_swap(18,22)
    if keys[18] > keys[22] || (keys[18] == keys[22] && values[18] > values[22]) {
    // swap(18,22) 
    { let tmp_2660 = keys[18]; keys[18] = keys[22]; keys[22] = tmp_2660;let tmp_2661 = values[18]; values[18] = values[22]; values[22] = tmp_2661; }
    }
    // cmp_swap(19,23)
    if keys[19] > keys[23] || (keys[19] == keys[23] && values[19] > values[23]) {
    // swap(19,23) 
    { let tmp_2662 = keys[19]; keys[19] = keys[23]; keys[23] = tmp_2662;let tmp_2663 = values[19]; values[19] = values[23]; values[23] = tmp_2663; }
    }
    // cmp_swap(24,28)
    if keys[24] > keys[28] || (keys[24] == keys[28] && values[24] > values[28]) {
    // swap(24,28) 
    { let tmp_2664 = keys[24]; keys[24] = keys[28]; keys[28] = tmp_2664;let tmp_2665 = values[24]; values[24] = values[28]; values[28] = tmp_2665; }
    }
    // cmp_swap(25,29)
    if keys[25] > keys[29] || (keys[25] == keys[29] && values[25] > values[29]) {
    // swap(25,29) 
    { let tmp_2666 = keys[25]; keys[25] = keys[29]; keys[29] = tmp_2666;let tmp_2667 = values[25]; values[25] = values[29]; values[29] = tmp_2667; }
    }
    // cmp_swap(26,30)
    if keys[26] > keys[30] || (keys[26] == keys[30] && values[26] > values[30]) {
    // swap(26,30) 
    { let tmp_2668 = keys[26]; keys[26] = keys[30]; keys[30] = tmp_2668;let tmp_2669 = values[26]; values[26] = values[30]; values[30] = tmp_2669; }
    }
    // cmp_swap(27,31)
    if keys[27] > keys[31] || (keys[27] == keys[31] && values[27] > values[31]) {
    // swap(27,31) 
    { let tmp_2670 = keys[27]; keys[27] = keys[31]; keys[31] = tmp_2670;let tmp_2671 = values[27]; values[27] = values[31]; values[31] = tmp_2671; }
    }
    // exch_local(2,32) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_2672 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_2672;let tmp_2673 = values[0]; values[0] = values[2]; values[2] = tmp_2673; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_2674 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_2674;let tmp_2675 = values[1]; values[1] = values[3]; values[3] = tmp_2675; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_2676 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_2676;let tmp_2677 = values[4]; values[4] = values[6]; values[6] = tmp_2677; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_2678 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_2678;let tmp_2679 = values[5]; values[5] = values[7]; values[7] = tmp_2679; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_2680 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_2680;let tmp_2681 = values[8]; values[8] = values[10]; values[10] = tmp_2681; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_2682 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_2682;let tmp_2683 = values[9]; values[9] = values[11]; values[11] = tmp_2683; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_2684 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_2684;let tmp_2685 = values[12]; values[12] = values[14]; values[14] = tmp_2685; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_2686 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_2686;let tmp_2687 = values[13]; values[13] = values[15]; values[15] = tmp_2687; }
    }
    // cmp_swap(16,18)
    if keys[16] > keys[18] || (keys[16] == keys[18] && values[16] > values[18]) {
    // swap(16,18) 
    { let tmp_2688 = keys[16]; keys[16] = keys[18]; keys[18] = tmp_2688;let tmp_2689 = values[16]; values[16] = values[18]; values[18] = tmp_2689; }
    }
    // cmp_swap(17,19)
    if keys[17] > keys[19] || (keys[17] == keys[19] && values[17] > values[19]) {
    // swap(17,19) 
    { let tmp_2690 = keys[17]; keys[17] = keys[19]; keys[19] = tmp_2690;let tmp_2691 = values[17]; values[17] = values[19]; values[19] = tmp_2691; }
    }
    // cmp_swap(20,22)
    if keys[20] > keys[22] || (keys[20] == keys[22] && values[20] > values[22]) {
    // swap(20,22) 
    { let tmp_2692 = keys[20]; keys[20] = keys[22]; keys[22] = tmp_2692;let tmp_2693 = values[20]; values[20] = values[22]; values[22] = tmp_2693; }
    }
    // cmp_swap(21,23)
    if keys[21] > keys[23] || (keys[21] == keys[23] && values[21] > values[23]) {
    // swap(21,23) 
    { let tmp_2694 = keys[21]; keys[21] = keys[23]; keys[23] = tmp_2694;let tmp_2695 = values[21]; values[21] = values[23]; values[23] = tmp_2695; }
    }
    // cmp_swap(24,26)
    if keys[24] > keys[26] || (keys[24] == keys[26] && values[24] > values[26]) {
    // swap(24,26) 
    { let tmp_2696 = keys[24]; keys[24] = keys[26]; keys[26] = tmp_2696;let tmp_2697 = values[24]; values[24] = values[26]; values[26] = tmp_2697; }
    }
    // cmp_swap(25,27)
    if keys[25] > keys[27] || (keys[25] == keys[27] && values[25] > values[27]) {
    // swap(25,27) 
    { let tmp_2698 = keys[25]; keys[25] = keys[27]; keys[27] = tmp_2698;let tmp_2699 = values[25]; values[25] = values[27]; values[27] = tmp_2699; }
    }
    // cmp_swap(28,30)
    if keys[28] > keys[30] || (keys[28] == keys[30] && values[28] > values[30]) {
    // swap(28,30) 
    { let tmp_2700 = keys[28]; keys[28] = keys[30]; keys[30] = tmp_2700;let tmp_2701 = values[28]; values[28] = values[30]; values[30] = tmp_2701; }
    }
    // cmp_swap(29,31)
    if keys[29] > keys[31] || (keys[29] == keys[31] && values[29] > values[31]) {
    // swap(29,31) 
    { let tmp_2702 = keys[29]; keys[29] = keys[31]; keys[31] = tmp_2702;let tmp_2703 = values[29]; values[29] = values[31]; values[31] = tmp_2703; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_2704 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_2704;let tmp_2705 = values[0]; values[0] = values[1]; values[1] = tmp_2705; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_2706 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_2706;let tmp_2707 = values[2]; values[2] = values[3]; values[3] = tmp_2707; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_2708 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_2708;let tmp_2709 = values[4]; values[4] = values[5]; values[5] = tmp_2709; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_2710 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_2710;let tmp_2711 = values[6]; values[6] = values[7]; values[7] = tmp_2711; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_2712 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_2712;let tmp_2713 = values[8]; values[8] = values[9]; values[9] = tmp_2713; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_2714 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_2714;let tmp_2715 = values[10]; values[10] = values[11]; values[11] = tmp_2715; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_2716 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_2716;let tmp_2717 = values[12]; values[12] = values[13]; values[13] = tmp_2717; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_2718 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_2718;let tmp_2719 = values[14]; values[14] = values[15]; values[15] = tmp_2719; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_2720 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_2720;let tmp_2721 = values[16]; values[16] = values[17]; values[17] = tmp_2721; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_2722 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_2722;let tmp_2723 = values[18]; values[18] = values[19]; values[19] = tmp_2723; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_2724 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_2724;let tmp_2725 = values[20]; values[20] = values[21]; values[21] = tmp_2725; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_2726 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_2726;let tmp_2727 = values[22]; values[22] = values[23]; values[23] = tmp_2727; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_2728 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_2728;let tmp_2729 = values[24]; values[24] = values[25]; values[25] = tmp_2729; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_2730 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_2730;let tmp_2731 = values[26]; values[26] = values[27]; values[27] = tmp_2731; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_2732 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_2732;let tmp_2733 = values[28]; values[28] = values[29]; values[29] = tmp_2733; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_2734 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_2734;let tmp_2735 = values[30]; values[30] = values[31]; values[31] = tmp_2735; }
    }
    // exch_intxn(tmask:63,swbit:5,wpt:32)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_2736 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_2737 = seg_base + (local_tid ^ 63u); let tmp_2738 = smem_keys[tmp_2737 * WPT + 31u]; let tmp_2739 = smem_vals[tmp_2737 * WPT + 31u]; let tmp_2740 = keys[0] < tmp_2738 || (keys[0] == tmp_2738 && values[0] < tmp_2739); if tmp_2736 == tmp_2740 { keys[0] = tmp_2738; values[0] = tmp_2739; } let tmp_2741 = smem_keys[tmp_2737 * WPT + 30u]; let tmp_2742 = smem_vals[tmp_2737 * WPT + 30u]; let tmp_2743 = keys[1] < tmp_2741 || (keys[1] == tmp_2741 && values[1] < tmp_2742); if tmp_2736 == tmp_2743 { keys[1] = tmp_2741; values[1] = tmp_2742; } let tmp_2744 = smem_keys[tmp_2737 * WPT + 29u]; let tmp_2745 = smem_vals[tmp_2737 * WPT + 29u]; let tmp_2746 = keys[2] < tmp_2744 || (keys[2] == tmp_2744 && values[2] < tmp_2745); if tmp_2736 == tmp_2746 { keys[2] = tmp_2744; values[2] = tmp_2745; } let tmp_2747 = smem_keys[tmp_2737 * WPT + 28u]; let tmp_2748 = smem_vals[tmp_2737 * WPT + 28u]; let tmp_2749 = keys[3] < tmp_2747 || (keys[3] == tmp_2747 && values[3] < tmp_2748); if tmp_2736 == tmp_2749 { keys[3] = tmp_2747; values[3] = tmp_2748; } let tmp_2750 = smem_keys[tmp_2737 * WPT + 27u]; let tmp_2751 = smem_vals[tmp_2737 * WPT + 27u]; let tmp_2752 = keys[4] < tmp_2750 || (keys[4] == tmp_2750 && values[4] < tmp_2751); if tmp_2736 == tmp_2752 { keys[4] = tmp_2750; values[4] = tmp_2751; } let tmp_2753 = smem_keys[tmp_2737 * WPT + 26u]; let tmp_2754 = smem_vals[tmp_2737 * WPT + 26u]; let tmp_2755 = keys[5] < tmp_2753 || (keys[5] == tmp_2753 && values[5] < tmp_2754); if tmp_2736 == tmp_2755 { keys[5] = tmp_2753; values[5] = tmp_2754; } let tmp_2756 = smem_keys[tmp_2737 * WPT + 25u]; let tmp_2757 = smem_vals[tmp_2737 * WPT + 25u]; let tmp_2758 = keys[6] < tmp_2756 || (keys[6] == tmp_2756 && values[6] < tmp_2757); if tmp_2736 == tmp_2758 { keys[6] = tmp_2756; values[6] = tmp_2757; } let tmp_2759 = smem_keys[tmp_2737 * WPT + 24u]; let tmp_2760 = smem_vals[tmp_2737 * WPT + 24u]; let tmp_2761 = keys[7] < tmp_2759 || (keys[7] == tmp_2759 && values[7] < tmp_2760); if tmp_2736 == tmp_2761 { keys[7] = tmp_2759; values[7] = tmp_2760; } let tmp_2762 = smem_keys[tmp_2737 * WPT + 23u]; let tmp_2763 = smem_vals[tmp_2737 * WPT + 23u]; let tmp_2764 = keys[8] < tmp_2762 || (keys[8] == tmp_2762 && values[8] < tmp_2763); if tmp_2736 == tmp_2764 { keys[8] = tmp_2762; values[8] = tmp_2763; } let tmp_2765 = smem_keys[tmp_2737 * WPT + 22u]; let tmp_2766 = smem_vals[tmp_2737 * WPT + 22u]; let tmp_2767 = keys[9] < tmp_2765 || (keys[9] == tmp_2765 && values[9] < tmp_2766); if tmp_2736 == tmp_2767 { keys[9] = tmp_2765; values[9] = tmp_2766; } let tmp_2768 = smem_keys[tmp_2737 * WPT + 21u]; let tmp_2769 = smem_vals[tmp_2737 * WPT + 21u]; let tmp_2770 = keys[10] < tmp_2768 || (keys[10] == tmp_2768 && values[10] < tmp_2769); if tmp_2736 == tmp_2770 { keys[10] = tmp_2768; values[10] = tmp_2769; } let tmp_2771 = smem_keys[tmp_2737 * WPT + 20u]; let tmp_2772 = smem_vals[tmp_2737 * WPT + 20u]; let tmp_2773 = keys[11] < tmp_2771 || (keys[11] == tmp_2771 && values[11] < tmp_2772); if tmp_2736 == tmp_2773 { keys[11] = tmp_2771; values[11] = tmp_2772; } let tmp_2774 = smem_keys[tmp_2737 * WPT + 19u]; let tmp_2775 = smem_vals[tmp_2737 * WPT + 19u]; let tmp_2776 = keys[12] < tmp_2774 || (keys[12] == tmp_2774 && values[12] < tmp_2775); if tmp_2736 == tmp_2776 { keys[12] = tmp_2774; values[12] = tmp_2775; } let tmp_2777 = smem_keys[tmp_2737 * WPT + 18u]; let tmp_2778 = smem_vals[tmp_2737 * WPT + 18u]; let tmp_2779 = keys[13] < tmp_2777 || (keys[13] == tmp_2777 && values[13] < tmp_2778); if tmp_2736 == tmp_2779 { keys[13] = tmp_2777; values[13] = tmp_2778; } let tmp_2780 = smem_keys[tmp_2737 * WPT + 17u]; let tmp_2781 = smem_vals[tmp_2737 * WPT + 17u]; let tmp_2782 = keys[14] < tmp_2780 || (keys[14] == tmp_2780 && values[14] < tmp_2781); if tmp_2736 == tmp_2782 { keys[14] = tmp_2780; values[14] = tmp_2781; } let tmp_2783 = smem_keys[tmp_2737 * WPT + 16u]; let tmp_2784 = smem_vals[tmp_2737 * WPT + 16u]; let tmp_2785 = keys[15] < tmp_2783 || (keys[15] == tmp_2783 && values[15] < tmp_2784); if tmp_2736 == tmp_2785 { keys[15] = tmp_2783; values[15] = tmp_2784; } let tmp_2786 = smem_keys[tmp_2737 * WPT + 15u]; let tmp_2787 = smem_vals[tmp_2737 * WPT + 15u]; let tmp_2788 = keys[16] < tmp_2786 || (keys[16] == tmp_2786 && values[16] < tmp_2787); if tmp_2736 == tmp_2788 { keys[16] = tmp_2786; values[16] = tmp_2787; } let tmp_2789 = smem_keys[tmp_2737 * WPT + 14u]; let tmp_2790 = smem_vals[tmp_2737 * WPT + 14u]; let tmp_2791 = keys[17] < tmp_2789 || (keys[17] == tmp_2789 && values[17] < tmp_2790); if tmp_2736 == tmp_2791 { keys[17] = tmp_2789; values[17] = tmp_2790; } let tmp_2792 = smem_keys[tmp_2737 * WPT + 13u]; let tmp_2793 = smem_vals[tmp_2737 * WPT + 13u]; let tmp_2794 = keys[18] < tmp_2792 || (keys[18] == tmp_2792 && values[18] < tmp_2793); if tmp_2736 == tmp_2794 { keys[18] = tmp_2792; values[18] = tmp_2793; } let tmp_2795 = smem_keys[tmp_2737 * WPT + 12u]; let tmp_2796 = smem_vals[tmp_2737 * WPT + 12u]; let tmp_2797 = keys[19] < tmp_2795 || (keys[19] == tmp_2795 && values[19] < tmp_2796); if tmp_2736 == tmp_2797 { keys[19] = tmp_2795; values[19] = tmp_2796; } let tmp_2798 = smem_keys[tmp_2737 * WPT + 11u]; let tmp_2799 = smem_vals[tmp_2737 * WPT + 11u]; let tmp_2800 = keys[20] < tmp_2798 || (keys[20] == tmp_2798 && values[20] < tmp_2799); if tmp_2736 == tmp_2800 { keys[20] = tmp_2798; values[20] = tmp_2799; } let tmp_2801 = smem_keys[tmp_2737 * WPT + 10u]; let tmp_2802 = smem_vals[tmp_2737 * WPT + 10u]; let tmp_2803 = keys[21] < tmp_2801 || (keys[21] == tmp_2801 && values[21] < tmp_2802); if tmp_2736 == tmp_2803 { keys[21] = tmp_2801; values[21] = tmp_2802; } let tmp_2804 = smem_keys[tmp_2737 * WPT + 9u]; let tmp_2805 = smem_vals[tmp_2737 * WPT + 9u]; let tmp_2806 = keys[22] < tmp_2804 || (keys[22] == tmp_2804 && values[22] < tmp_2805); if tmp_2736 == tmp_2806 { keys[22] = tmp_2804; values[22] = tmp_2805; } let tmp_2807 = smem_keys[tmp_2737 * WPT + 8u]; let tmp_2808 = smem_vals[tmp_2737 * WPT + 8u]; let tmp_2809 = keys[23] < tmp_2807 || (keys[23] == tmp_2807 && values[23] < tmp_2808); if tmp_2736 == tmp_2809 { keys[23] = tmp_2807; values[23] = tmp_2808; } let tmp_2810 = smem_keys[tmp_2737 * WPT + 7u]; let tmp_2811 = smem_vals[tmp_2737 * WPT + 7u]; let tmp_2812 = keys[24] < tmp_2810 || (keys[24] == tmp_2810 && values[24] < tmp_2811); if tmp_2736 == tmp_2812 { keys[24] = tmp_2810; values[24] = tmp_2811; } let tmp_2813 = smem_keys[tmp_2737 * WPT + 6u]; let tmp_2814 = smem_vals[tmp_2737 * WPT + 6u]; let tmp_2815 = keys[25] < tmp_2813 || (keys[25] == tmp_2813 && values[25] < tmp_2814); if tmp_2736 == tmp_2815 { keys[25] = tmp_2813; values[25] = tmp_2814; } let tmp_2816 = smem_keys[tmp_2737 * WPT + 5u]; let tmp_2817 = smem_vals[tmp_2737 * WPT + 5u]; let tmp_2818 = keys[26] < tmp_2816 || (keys[26] == tmp_2816 && values[26] < tmp_2817); if tmp_2736 == tmp_2818 { keys[26] = tmp_2816; values[26] = tmp_2817; } let tmp_2819 = smem_keys[tmp_2737 * WPT + 4u]; let tmp_2820 = smem_vals[tmp_2737 * WPT + 4u]; let tmp_2821 = keys[27] < tmp_2819 || (keys[27] == tmp_2819 && values[27] < tmp_2820); if tmp_2736 == tmp_2821 { keys[27] = tmp_2819; values[27] = tmp_2820; } let tmp_2822 = smem_keys[tmp_2737 * WPT + 3u]; let tmp_2823 = smem_vals[tmp_2737 * WPT + 3u]; let tmp_2824 = keys[28] < tmp_2822 || (keys[28] == tmp_2822 && values[28] < tmp_2823); if tmp_2736 == tmp_2824 { keys[28] = tmp_2822; values[28] = tmp_2823; } let tmp_2825 = smem_keys[tmp_2737 * WPT + 2u]; let tmp_2826 = smem_vals[tmp_2737 * WPT + 2u]; let tmp_2827 = keys[29] < tmp_2825 || (keys[29] == tmp_2825 && values[29] < tmp_2826); if tmp_2736 == tmp_2827 { keys[29] = tmp_2825; values[29] = tmp_2826; } let tmp_2828 = smem_keys[tmp_2737 * WPT + 1u]; let tmp_2829 = smem_vals[tmp_2737 * WPT + 1u]; let tmp_2830 = keys[30] < tmp_2828 || (keys[30] == tmp_2828 && values[30] < tmp_2829); if tmp_2736 == tmp_2830 { keys[30] = tmp_2828; values[30] = tmp_2829; } let tmp_2831 = smem_keys[tmp_2737 * WPT + 0u]; let tmp_2832 = smem_vals[tmp_2737 * WPT + 0u]; let tmp_2833 = keys[31] < tmp_2831 || (keys[31] == tmp_2831 && values[31] < tmp_2832); if tmp_2736 == tmp_2833 { keys[31] = tmp_2831; values[31] = tmp_2832; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:32) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; smem_keys[tid_g * WPT + 16u] = keys[16]; smem_vals[tid_g * WPT + 16u] = values[16]; smem_keys[tid_g * WPT + 17u] = keys[17]; smem_vals[tid_g * WPT + 17u] = values[17]; smem_keys[tid_g * WPT + 18u] = keys[18]; smem_vals[tid_g * WPT + 18u] = values[18]; smem_keys[tid_g * WPT + 19u] = keys[19]; smem_vals[tid_g * WPT + 19u] = values[19]; smem_keys[tid_g * WPT + 20u] = keys[20]; smem_vals[tid_g * WPT + 20u] = values[20]; smem_keys[tid_g * WPT + 21u] = keys[21]; smem_vals[tid_g * WPT + 21u] = values[21]; smem_keys[tid_g * WPT + 22u] = keys[22]; smem_vals[tid_g * WPT + 22u] = values[22]; smem_keys[tid_g * WPT + 23u] = keys[23]; smem_vals[tid_g * WPT + 23u] = values[23]; smem_keys[tid_g * WPT + 24u] = keys[24]; smem_vals[tid_g * WPT + 24u] = values[24]; smem_keys[tid_g * WPT + 25u] = keys[25]; smem_vals[tid_g * WPT + 25u] = values[25]; smem_keys[tid_g * WPT + 26u] = keys[26]; smem_vals[tid_g * WPT + 26u] = values[26]; smem_keys[tid_g * WPT + 27u] = keys[27]; smem_vals[tid_g * WPT + 27u] = values[27]; smem_keys[tid_g * WPT + 28u] = keys[28]; smem_vals[tid_g * WPT + 28u] = values[28]; smem_keys[tid_g * WPT + 29u] = keys[29]; smem_vals[tid_g * WPT + 29u] = values[29]; smem_keys[tid_g * WPT + 30u] = keys[30]; smem_vals[tid_g * WPT + 30u] = values[30]; smem_keys[tid_g * WPT + 31u] = keys[31]; smem_vals[tid_g * WPT + 31u] = values[31]; workgroupBarrier(); let tmp_2834 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_2835 = seg_base + (local_tid ^ 16u); let tmp_2836 = smem_keys[tmp_2835 * WPT + 0u]; let tmp_2837 = smem_vals[tmp_2835 * WPT + 0u]; let tmp_2838 = keys[0] < tmp_2836 || (keys[0] == tmp_2836 && values[0] < tmp_2837); if tmp_2834 == tmp_2838 { keys[0] = tmp_2836; values[0] = tmp_2837; } let tmp_2839 = smem_keys[tmp_2835 * WPT + 1u]; let tmp_2840 = smem_vals[tmp_2835 * WPT + 1u]; let tmp_2841 = keys[1] < tmp_2839 || (keys[1] == tmp_2839 && values[1] < tmp_2840); if tmp_2834 == tmp_2841 { keys[1] = tmp_2839; values[1] = tmp_2840; } let tmp_2842 = smem_keys[tmp_2835 * WPT + 2u]; let tmp_2843 = smem_vals[tmp_2835 * WPT + 2u]; let tmp_2844 = keys[2] < tmp_2842 || (keys[2] == tmp_2842 && values[2] < tmp_2843); if tmp_2834 == tmp_2844 { keys[2] = tmp_2842; values[2] = tmp_2843; } let tmp_2845 = smem_keys[tmp_2835 * WPT + 3u]; let tmp_2846 = smem_vals[tmp_2835 * WPT + 3u]; let tmp_2847 = keys[3] < tmp_2845 || (keys[3] == tmp_2845 && values[3] < tmp_2846); if tmp_2834 == tmp_2847 { keys[3] = tmp_2845; values[3] = tmp_2846; } let tmp_2848 = smem_keys[tmp_2835 * WPT + 4u]; let tmp_2849 = smem_vals[tmp_2835 * WPT + 4u]; let tmp_2850 = keys[4] < tmp_2848 || (keys[4] == tmp_2848 && values[4] < tmp_2849); if tmp_2834 == tmp_2850 { keys[4] = tmp_2848; values[4] = tmp_2849; } let tmp_2851 = smem_keys[tmp_2835 * WPT + 5u]; let tmp_2852 = smem_vals[tmp_2835 * WPT + 5u]; let tmp_2853 = keys[5] < tmp_2851 || (keys[5] == tmp_2851 && values[5] < tmp_2852); if tmp_2834 == tmp_2853 { keys[5] = tmp_2851; values[5] = tmp_2852; } let tmp_2854 = smem_keys[tmp_2835 * WPT + 6u]; let tmp_2855 = smem_vals[tmp_2835 * WPT + 6u]; let tmp_2856 = keys[6] < tmp_2854 || (keys[6] == tmp_2854 && values[6] < tmp_2855); if tmp_2834 == tmp_2856 { keys[6] = tmp_2854; values[6] = tmp_2855; } let tmp_2857 = smem_keys[tmp_2835 * WPT + 7u]; let tmp_2858 = smem_vals[tmp_2835 * WPT + 7u]; let tmp_2859 = keys[7] < tmp_2857 || (keys[7] == tmp_2857 && values[7] < tmp_2858); if tmp_2834 == tmp_2859 { keys[7] = tmp_2857; values[7] = tmp_2858; } let tmp_2860 = smem_keys[tmp_2835 * WPT + 8u]; let tmp_2861 = smem_vals[tmp_2835 * WPT + 8u]; let tmp_2862 = keys[8] < tmp_2860 || (keys[8] == tmp_2860 && values[8] < tmp_2861); if tmp_2834 == tmp_2862 { keys[8] = tmp_2860; values[8] = tmp_2861; } let tmp_2863 = smem_keys[tmp_2835 * WPT + 9u]; let tmp_2864 = smem_vals[tmp_2835 * WPT + 9u]; let tmp_2865 = keys[9] < tmp_2863 || (keys[9] == tmp_2863 && values[9] < tmp_2864); if tmp_2834 == tmp_2865 { keys[9] = tmp_2863; values[9] = tmp_2864; } let tmp_2866 = smem_keys[tmp_2835 * WPT + 10u]; let tmp_2867 = smem_vals[tmp_2835 * WPT + 10u]; let tmp_2868 = keys[10] < tmp_2866 || (keys[10] == tmp_2866 && values[10] < tmp_2867); if tmp_2834 == tmp_2868 { keys[10] = tmp_2866; values[10] = tmp_2867; } let tmp_2869 = smem_keys[tmp_2835 * WPT + 11u]; let tmp_2870 = smem_vals[tmp_2835 * WPT + 11u]; let tmp_2871 = keys[11] < tmp_2869 || (keys[11] == tmp_2869 && values[11] < tmp_2870); if tmp_2834 == tmp_2871 { keys[11] = tmp_2869; values[11] = tmp_2870; } let tmp_2872 = smem_keys[tmp_2835 * WPT + 12u]; let tmp_2873 = smem_vals[tmp_2835 * WPT + 12u]; let tmp_2874 = keys[12] < tmp_2872 || (keys[12] == tmp_2872 && values[12] < tmp_2873); if tmp_2834 == tmp_2874 { keys[12] = tmp_2872; values[12] = tmp_2873; } let tmp_2875 = smem_keys[tmp_2835 * WPT + 13u]; let tmp_2876 = smem_vals[tmp_2835 * WPT + 13u]; let tmp_2877 = keys[13] < tmp_2875 || (keys[13] == tmp_2875 && values[13] < tmp_2876); if tmp_2834 == tmp_2877 { keys[13] = tmp_2875; values[13] = tmp_2876; } let tmp_2878 = smem_keys[tmp_2835 * WPT + 14u]; let tmp_2879 = smem_vals[tmp_2835 * WPT + 14u]; let tmp_2880 = keys[14] < tmp_2878 || (keys[14] == tmp_2878 && values[14] < tmp_2879); if tmp_2834 == tmp_2880 { keys[14] = tmp_2878; values[14] = tmp_2879; } let tmp_2881 = smem_keys[tmp_2835 * WPT + 15u]; let tmp_2882 = smem_vals[tmp_2835 * WPT + 15u]; let tmp_2883 = keys[15] < tmp_2881 || (keys[15] == tmp_2881 && values[15] < tmp_2882); if tmp_2834 == tmp_2883 { keys[15] = tmp_2881; values[15] = tmp_2882; } let tmp_2884 = smem_keys[tmp_2835 * WPT + 16u]; let tmp_2885 = smem_vals[tmp_2835 * WPT + 16u]; let tmp_2886 = keys[16] < tmp_2884 || (keys[16] == tmp_2884 && values[16] < tmp_2885); if tmp_2834 == tmp_2886 { keys[16] = tmp_2884; values[16] = tmp_2885; } let tmp_2887 = smem_keys[tmp_2835 * WPT + 17u]; let tmp_2888 = smem_vals[tmp_2835 * WPT + 17u]; let tmp_2889 = keys[17] < tmp_2887 || (keys[17] == tmp_2887 && values[17] < tmp_2888); if tmp_2834 == tmp_2889 { keys[17] = tmp_2887; values[17] = tmp_2888; } let tmp_2890 = smem_keys[tmp_2835 * WPT + 18u]; let tmp_2891 = smem_vals[tmp_2835 * WPT + 18u]; let tmp_2892 = keys[18] < tmp_2890 || (keys[18] == tmp_2890 && values[18] < tmp_2891); if tmp_2834 == tmp_2892 { keys[18] = tmp_2890; values[18] = tmp_2891; } let tmp_2893 = smem_keys[tmp_2835 * WPT + 19u]; let tmp_2894 = smem_vals[tmp_2835 * WPT + 19u]; let tmp_2895 = keys[19] < tmp_2893 || (keys[19] == tmp_2893 && values[19] < tmp_2894); if tmp_2834 == tmp_2895 { keys[19] = tmp_2893; values[19] = tmp_2894; } let tmp_2896 = smem_keys[tmp_2835 * WPT + 20u]; let tmp_2897 = smem_vals[tmp_2835 * WPT + 20u]; let tmp_2898 = keys[20] < tmp_2896 || (keys[20] == tmp_2896 && values[20] < tmp_2897); if tmp_2834 == tmp_2898 { keys[20] = tmp_2896; values[20] = tmp_2897; } let tmp_2899 = smem_keys[tmp_2835 * WPT + 21u]; let tmp_2900 = smem_vals[tmp_2835 * WPT + 21u]; let tmp_2901 = keys[21] < tmp_2899 || (keys[21] == tmp_2899 && values[21] < tmp_2900); if tmp_2834 == tmp_2901 { keys[21] = tmp_2899; values[21] = tmp_2900; } let tmp_2902 = smem_keys[tmp_2835 * WPT + 22u]; let tmp_2903 = smem_vals[tmp_2835 * WPT + 22u]; let tmp_2904 = keys[22] < tmp_2902 || (keys[22] == tmp_2902 && values[22] < tmp_2903); if tmp_2834 == tmp_2904 { keys[22] = tmp_2902; values[22] = tmp_2903; } let tmp_2905 = smem_keys[tmp_2835 * WPT + 23u]; let tmp_2906 = smem_vals[tmp_2835 * WPT + 23u]; let tmp_2907 = keys[23] < tmp_2905 || (keys[23] == tmp_2905 && values[23] < tmp_2906); if tmp_2834 == tmp_2907 { keys[23] = tmp_2905; values[23] = tmp_2906; } let tmp_2908 = smem_keys[tmp_2835 * WPT + 24u]; let tmp_2909 = smem_vals[tmp_2835 * WPT + 24u]; let tmp_2910 = keys[24] < tmp_2908 || (keys[24] == tmp_2908 && values[24] < tmp_2909); if tmp_2834 == tmp_2910 { keys[24] = tmp_2908; values[24] = tmp_2909; } let tmp_2911 = smem_keys[tmp_2835 * WPT + 25u]; let tmp_2912 = smem_vals[tmp_2835 * WPT + 25u]; let tmp_2913 = keys[25] < tmp_2911 || (keys[25] == tmp_2911 && values[25] < tmp_2912); if tmp_2834 == tmp_2913 { keys[25] = tmp_2911; values[25] = tmp_2912; } let tmp_2914 = smem_keys[tmp_2835 * WPT + 26u]; let tmp_2915 = smem_vals[tmp_2835 * WPT + 26u]; let tmp_2916 = keys[26] < tmp_2914 || (keys[26] == tmp_2914 && values[26] < tmp_2915); if tmp_2834 == tmp_2916 { keys[26] = tmp_2914; values[26] = tmp_2915; } let tmp_2917 = smem_keys[tmp_2835 * WPT + 27u]; let tmp_2918 = smem_vals[tmp_2835 * WPT + 27u]; let tmp_2919 = keys[27] < tmp_2917 || (keys[27] == tmp_2917 && values[27] < tmp_2918); if tmp_2834 == tmp_2919 { keys[27] = tmp_2917; values[27] = tmp_2918; } let tmp_2920 = smem_keys[tmp_2835 * WPT + 28u]; let tmp_2921 = smem_vals[tmp_2835 * WPT + 28u]; let tmp_2922 = keys[28] < tmp_2920 || (keys[28] == tmp_2920 && values[28] < tmp_2921); if tmp_2834 == tmp_2922 { keys[28] = tmp_2920; values[28] = tmp_2921; } let tmp_2923 = smem_keys[tmp_2835 * WPT + 29u]; let tmp_2924 = smem_vals[tmp_2835 * WPT + 29u]; let tmp_2925 = keys[29] < tmp_2923 || (keys[29] == tmp_2923 && values[29] < tmp_2924); if tmp_2834 == tmp_2925 { keys[29] = tmp_2923; values[29] = tmp_2924; } let tmp_2926 = smem_keys[tmp_2835 * WPT + 30u]; let tmp_2927 = smem_vals[tmp_2835 * WPT + 30u]; let tmp_2928 = keys[30] < tmp_2926 || (keys[30] == tmp_2926 && values[30] < tmp_2927); if tmp_2834 == tmp_2928 { keys[30] = tmp_2926; values[30] = tmp_2927; } let tmp_2929 = smem_keys[tmp_2835 * WPT + 31u]; let tmp_2930 = smem_vals[tmp_2835 * WPT + 31u]; let tmp_2931 = keys[31] < tmp_2929 || (keys[31] == tmp_2929 && values[31] < tmp_2930); if tmp_2834 == tmp_2931 { keys[31] = tmp_2929; values[31] = tmp_2930; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:32) 
    {
    let tmp_2932 = subgroupShuffleXor(keys[0], 8u);
    let tmp_2933 = subgroupShuffleXor(values[0], 8u);
    let tmp_2934 = subgroupShuffleXor(keys[1], 8u);
    let tmp_2935 = subgroupShuffleXor(values[1], 8u);
    let tmp_2936 = subgroupShuffleXor(keys[2], 8u);
    let tmp_2937 = subgroupShuffleXor(values[2], 8u);
    let tmp_2938 = subgroupShuffleXor(keys[3], 8u);
    let tmp_2939 = subgroupShuffleXor(values[3], 8u);
    let tmp_2940 = subgroupShuffleXor(keys[4], 8u);
    let tmp_2941 = subgroupShuffleXor(values[4], 8u);
    let tmp_2942 = subgroupShuffleXor(keys[5], 8u);
    let tmp_2943 = subgroupShuffleXor(values[5], 8u);
    let tmp_2944 = subgroupShuffleXor(keys[6], 8u);
    let tmp_2945 = subgroupShuffleXor(values[6], 8u);
    let tmp_2946 = subgroupShuffleXor(keys[7], 8u);
    let tmp_2947 = subgroupShuffleXor(values[7], 8u);
    let tmp_2948 = subgroupShuffleXor(keys[8], 8u);
    let tmp_2949 = subgroupShuffleXor(values[8], 8u);
    let tmp_2950 = subgroupShuffleXor(keys[9], 8u);
    let tmp_2951 = subgroupShuffleXor(values[9], 8u);
    let tmp_2952 = subgroupShuffleXor(keys[10], 8u);
    let tmp_2953 = subgroupShuffleXor(values[10], 8u);
    let tmp_2954 = subgroupShuffleXor(keys[11], 8u);
    let tmp_2955 = subgroupShuffleXor(values[11], 8u);
    let tmp_2956 = subgroupShuffleXor(keys[12], 8u);
    let tmp_2957 = subgroupShuffleXor(values[12], 8u);
    let tmp_2958 = subgroupShuffleXor(keys[13], 8u);
    let tmp_2959 = subgroupShuffleXor(values[13], 8u);
    let tmp_2960 = subgroupShuffleXor(keys[14], 8u);
    let tmp_2961 = subgroupShuffleXor(values[14], 8u);
    let tmp_2962 = subgroupShuffleXor(keys[15], 8u);
    let tmp_2963 = subgroupShuffleXor(values[15], 8u);
    let tmp_2964 = subgroupShuffleXor(keys[16], 8u);
    let tmp_2965 = subgroupShuffleXor(values[16], 8u);
    let tmp_2966 = subgroupShuffleXor(keys[17], 8u);
    let tmp_2967 = subgroupShuffleXor(values[17], 8u);
    let tmp_2968 = subgroupShuffleXor(keys[18], 8u);
    let tmp_2969 = subgroupShuffleXor(values[18], 8u);
    let tmp_2970 = subgroupShuffleXor(keys[19], 8u);
    let tmp_2971 = subgroupShuffleXor(values[19], 8u);
    let tmp_2972 = subgroupShuffleXor(keys[20], 8u);
    let tmp_2973 = subgroupShuffleXor(values[20], 8u);
    let tmp_2974 = subgroupShuffleXor(keys[21], 8u);
    let tmp_2975 = subgroupShuffleXor(values[21], 8u);
    let tmp_2976 = subgroupShuffleXor(keys[22], 8u);
    let tmp_2977 = subgroupShuffleXor(values[22], 8u);
    let tmp_2978 = subgroupShuffleXor(keys[23], 8u);
    let tmp_2979 = subgroupShuffleXor(values[23], 8u);
    let tmp_2980 = subgroupShuffleXor(keys[24], 8u);
    let tmp_2981 = subgroupShuffleXor(values[24], 8u);
    let tmp_2982 = subgroupShuffleXor(keys[25], 8u);
    let tmp_2983 = subgroupShuffleXor(values[25], 8u);
    let tmp_2984 = subgroupShuffleXor(keys[26], 8u);
    let tmp_2985 = subgroupShuffleXor(values[26], 8u);
    let tmp_2986 = subgroupShuffleXor(keys[27], 8u);
    let tmp_2987 = subgroupShuffleXor(values[27], 8u);
    let tmp_2988 = subgroupShuffleXor(keys[28], 8u);
    let tmp_2989 = subgroupShuffleXor(values[28], 8u);
    let tmp_2990 = subgroupShuffleXor(keys[29], 8u);
    let tmp_2991 = subgroupShuffleXor(values[29], 8u);
    let tmp_2992 = subgroupShuffleXor(keys[30], 8u);
    let tmp_2993 = subgroupShuffleXor(values[30], 8u);
    let tmp_2994 = subgroupShuffleXor(keys[31], 8u);
    let tmp_2995 = subgroupShuffleXor(values[31], 8u);
    let tmp_2996 = extractBits(local_tid, 3u, 1u) != 0u;
    let tmp_2997 = keys[0] < tmp_2932 || (keys[0] == tmp_2932 && values[0] < tmp_2933);
    if tmp_2996 == tmp_2997 { keys[0] = tmp_2932; values[0] = tmp_2933; }
    let tmp_2998 = keys[1] < tmp_2934 || (keys[1] == tmp_2934 && values[1] < tmp_2935);
    if tmp_2996 == tmp_2998 { keys[1] = tmp_2934; values[1] = tmp_2935; }
    let tmp_2999 = keys[2] < tmp_2936 || (keys[2] == tmp_2936 && values[2] < tmp_2937);
    if tmp_2996 == tmp_2999 { keys[2] = tmp_2936; values[2] = tmp_2937; }
    let tmp_3000 = keys[3] < tmp_2938 || (keys[3] == tmp_2938 && values[3] < tmp_2939);
    if tmp_2996 == tmp_3000 { keys[3] = tmp_2938; values[3] = tmp_2939; }
    let tmp_3001 = keys[4] < tmp_2940 || (keys[4] == tmp_2940 && values[4] < tmp_2941);
    if tmp_2996 == tmp_3001 { keys[4] = tmp_2940; values[4] = tmp_2941; }
    let tmp_3002 = keys[5] < tmp_2942 || (keys[5] == tmp_2942 && values[5] < tmp_2943);
    if tmp_2996 == tmp_3002 { keys[5] = tmp_2942; values[5] = tmp_2943; }
    let tmp_3003 = keys[6] < tmp_2944 || (keys[6] == tmp_2944 && values[6] < tmp_2945);
    if tmp_2996 == tmp_3003 { keys[6] = tmp_2944; values[6] = tmp_2945; }
    let tmp_3004 = keys[7] < tmp_2946 || (keys[7] == tmp_2946 && values[7] < tmp_2947);
    if tmp_2996 == tmp_3004 { keys[7] = tmp_2946; values[7] = tmp_2947; }
    let tmp_3005 = keys[8] < tmp_2948 || (keys[8] == tmp_2948 && values[8] < tmp_2949);
    if tmp_2996 == tmp_3005 { keys[8] = tmp_2948; values[8] = tmp_2949; }
    let tmp_3006 = keys[9] < tmp_2950 || (keys[9] == tmp_2950 && values[9] < tmp_2951);
    if tmp_2996 == tmp_3006 { keys[9] = tmp_2950; values[9] = tmp_2951; }
    let tmp_3007 = keys[10] < tmp_2952 || (keys[10] == tmp_2952 && values[10] < tmp_2953);
    if tmp_2996 == tmp_3007 { keys[10] = tmp_2952; values[10] = tmp_2953; }
    let tmp_3008 = keys[11] < tmp_2954 || (keys[11] == tmp_2954 && values[11] < tmp_2955);
    if tmp_2996 == tmp_3008 { keys[11] = tmp_2954; values[11] = tmp_2955; }
    let tmp_3009 = keys[12] < tmp_2956 || (keys[12] == tmp_2956 && values[12] < tmp_2957);
    if tmp_2996 == tmp_3009 { keys[12] = tmp_2956; values[12] = tmp_2957; }
    let tmp_3010 = keys[13] < tmp_2958 || (keys[13] == tmp_2958 && values[13] < tmp_2959);
    if tmp_2996 == tmp_3010 { keys[13] = tmp_2958; values[13] = tmp_2959; }
    let tmp_3011 = keys[14] < tmp_2960 || (keys[14] == tmp_2960 && values[14] < tmp_2961);
    if tmp_2996 == tmp_3011 { keys[14] = tmp_2960; values[14] = tmp_2961; }
    let tmp_3012 = keys[15] < tmp_2962 || (keys[15] == tmp_2962 && values[15] < tmp_2963);
    if tmp_2996 == tmp_3012 { keys[15] = tmp_2962; values[15] = tmp_2963; }
    let tmp_3013 = keys[16] < tmp_2964 || (keys[16] == tmp_2964 && values[16] < tmp_2965);
    if tmp_2996 == tmp_3013 { keys[16] = tmp_2964; values[16] = tmp_2965; }
    let tmp_3014 = keys[17] < tmp_2966 || (keys[17] == tmp_2966 && values[17] < tmp_2967);
    if tmp_2996 == tmp_3014 { keys[17] = tmp_2966; values[17] = tmp_2967; }
    let tmp_3015 = keys[18] < tmp_2968 || (keys[18] == tmp_2968 && values[18] < tmp_2969);
    if tmp_2996 == tmp_3015 { keys[18] = tmp_2968; values[18] = tmp_2969; }
    let tmp_3016 = keys[19] < tmp_2970 || (keys[19] == tmp_2970 && values[19] < tmp_2971);
    if tmp_2996 == tmp_3016 { keys[19] = tmp_2970; values[19] = tmp_2971; }
    let tmp_3017 = keys[20] < tmp_2972 || (keys[20] == tmp_2972 && values[20] < tmp_2973);
    if tmp_2996 == tmp_3017 { keys[20] = tmp_2972; values[20] = tmp_2973; }
    let tmp_3018 = keys[21] < tmp_2974 || (keys[21] == tmp_2974 && values[21] < tmp_2975);
    if tmp_2996 == tmp_3018 { keys[21] = tmp_2974; values[21] = tmp_2975; }
    let tmp_3019 = keys[22] < tmp_2976 || (keys[22] == tmp_2976 && values[22] < tmp_2977);
    if tmp_2996 == tmp_3019 { keys[22] = tmp_2976; values[22] = tmp_2977; }
    let tmp_3020 = keys[23] < tmp_2978 || (keys[23] == tmp_2978 && values[23] < tmp_2979);
    if tmp_2996 == tmp_3020 { keys[23] = tmp_2978; values[23] = tmp_2979; }
    let tmp_3021 = keys[24] < tmp_2980 || (keys[24] == tmp_2980 && values[24] < tmp_2981);
    if tmp_2996 == tmp_3021 { keys[24] = tmp_2980; values[24] = tmp_2981; }
    let tmp_3022 = keys[25] < tmp_2982 || (keys[25] == tmp_2982 && values[25] < tmp_2983);
    if tmp_2996 == tmp_3022 { keys[25] = tmp_2982; values[25] = tmp_2983; }
    let tmp_3023 = keys[26] < tmp_2984 || (keys[26] == tmp_2984 && values[26] < tmp_2985);
    if tmp_2996 == tmp_3023 { keys[26] = tmp_2984; values[26] = tmp_2985; }
    let tmp_3024 = keys[27] < tmp_2986 || (keys[27] == tmp_2986 && values[27] < tmp_2987);
    if tmp_2996 == tmp_3024 { keys[27] = tmp_2986; values[27] = tmp_2987; }
    let tmp_3025 = keys[28] < tmp_2988 || (keys[28] == tmp_2988 && values[28] < tmp_2989);
    if tmp_2996 == tmp_3025 { keys[28] = tmp_2988; values[28] = tmp_2989; }
    let tmp_3026 = keys[29] < tmp_2990 || (keys[29] == tmp_2990 && values[29] < tmp_2991);
    if tmp_2996 == tmp_3026 { keys[29] = tmp_2990; values[29] = tmp_2991; }
    let tmp_3027 = keys[30] < tmp_2992 || (keys[30] == tmp_2992 && values[30] < tmp_2993);
    if tmp_2996 == tmp_3027 { keys[30] = tmp_2992; values[30] = tmp_2993; }
    let tmp_3028 = keys[31] < tmp_2994 || (keys[31] == tmp_2994 && values[31] < tmp_2995);
    if tmp_2996 == tmp_3028 { keys[31] = tmp_2994; values[31] = tmp_2995; }
    }
    // exch_paral(tmask:4,swbit:2,wpt:32) 
    {
    let tmp_3029 = subgroupShuffleXor(keys[0], 4u);
    let tmp_3030 = subgroupShuffleXor(values[0], 4u);
    let tmp_3031 = subgroupShuffleXor(keys[1], 4u);
    let tmp_3032 = subgroupShuffleXor(values[1], 4u);
    let tmp_3033 = subgroupShuffleXor(keys[2], 4u);
    let tmp_3034 = subgroupShuffleXor(values[2], 4u);
    let tmp_3035 = subgroupShuffleXor(keys[3], 4u);
    let tmp_3036 = subgroupShuffleXor(values[3], 4u);
    let tmp_3037 = subgroupShuffleXor(keys[4], 4u);
    let tmp_3038 = subgroupShuffleXor(values[4], 4u);
    let tmp_3039 = subgroupShuffleXor(keys[5], 4u);
    let tmp_3040 = subgroupShuffleXor(values[5], 4u);
    let tmp_3041 = subgroupShuffleXor(keys[6], 4u);
    let tmp_3042 = subgroupShuffleXor(values[6], 4u);
    let tmp_3043 = subgroupShuffleXor(keys[7], 4u);
    let tmp_3044 = subgroupShuffleXor(values[7], 4u);
    let tmp_3045 = subgroupShuffleXor(keys[8], 4u);
    let tmp_3046 = subgroupShuffleXor(values[8], 4u);
    let tmp_3047 = subgroupShuffleXor(keys[9], 4u);
    let tmp_3048 = subgroupShuffleXor(values[9], 4u);
    let tmp_3049 = subgroupShuffleXor(keys[10], 4u);
    let tmp_3050 = subgroupShuffleXor(values[10], 4u);
    let tmp_3051 = subgroupShuffleXor(keys[11], 4u);
    let tmp_3052 = subgroupShuffleXor(values[11], 4u);
    let tmp_3053 = subgroupShuffleXor(keys[12], 4u);
    let tmp_3054 = subgroupShuffleXor(values[12], 4u);
    let tmp_3055 = subgroupShuffleXor(keys[13], 4u);
    let tmp_3056 = subgroupShuffleXor(values[13], 4u);
    let tmp_3057 = subgroupShuffleXor(keys[14], 4u);
    let tmp_3058 = subgroupShuffleXor(values[14], 4u);
    let tmp_3059 = subgroupShuffleXor(keys[15], 4u);
    let tmp_3060 = subgroupShuffleXor(values[15], 4u);
    let tmp_3061 = subgroupShuffleXor(keys[16], 4u);
    let tmp_3062 = subgroupShuffleXor(values[16], 4u);
    let tmp_3063 = subgroupShuffleXor(keys[17], 4u);
    let tmp_3064 = subgroupShuffleXor(values[17], 4u);
    let tmp_3065 = subgroupShuffleXor(keys[18], 4u);
    let tmp_3066 = subgroupShuffleXor(values[18], 4u);
    let tmp_3067 = subgroupShuffleXor(keys[19], 4u);
    let tmp_3068 = subgroupShuffleXor(values[19], 4u);
    let tmp_3069 = subgroupShuffleXor(keys[20], 4u);
    let tmp_3070 = subgroupShuffleXor(values[20], 4u);
    let tmp_3071 = subgroupShuffleXor(keys[21], 4u);
    let tmp_3072 = subgroupShuffleXor(values[21], 4u);
    let tmp_3073 = subgroupShuffleXor(keys[22], 4u);
    let tmp_3074 = subgroupShuffleXor(values[22], 4u);
    let tmp_3075 = subgroupShuffleXor(keys[23], 4u);
    let tmp_3076 = subgroupShuffleXor(values[23], 4u);
    let tmp_3077 = subgroupShuffleXor(keys[24], 4u);
    let tmp_3078 = subgroupShuffleXor(values[24], 4u);
    let tmp_3079 = subgroupShuffleXor(keys[25], 4u);
    let tmp_3080 = subgroupShuffleXor(values[25], 4u);
    let tmp_3081 = subgroupShuffleXor(keys[26], 4u);
    let tmp_3082 = subgroupShuffleXor(values[26], 4u);
    let tmp_3083 = subgroupShuffleXor(keys[27], 4u);
    let tmp_3084 = subgroupShuffleXor(values[27], 4u);
    let tmp_3085 = subgroupShuffleXor(keys[28], 4u);
    let tmp_3086 = subgroupShuffleXor(values[28], 4u);
    let tmp_3087 = subgroupShuffleXor(keys[29], 4u);
    let tmp_3088 = subgroupShuffleXor(values[29], 4u);
    let tmp_3089 = subgroupShuffleXor(keys[30], 4u);
    let tmp_3090 = subgroupShuffleXor(values[30], 4u);
    let tmp_3091 = subgroupShuffleXor(keys[31], 4u);
    let tmp_3092 = subgroupShuffleXor(values[31], 4u);
    let tmp_3093 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_3094 = keys[0] < tmp_3029 || (keys[0] == tmp_3029 && values[0] < tmp_3030);
    if tmp_3093 == tmp_3094 { keys[0] = tmp_3029; values[0] = tmp_3030; }
    let tmp_3095 = keys[1] < tmp_3031 || (keys[1] == tmp_3031 && values[1] < tmp_3032);
    if tmp_3093 == tmp_3095 { keys[1] = tmp_3031; values[1] = tmp_3032; }
    let tmp_3096 = keys[2] < tmp_3033 || (keys[2] == tmp_3033 && values[2] < tmp_3034);
    if tmp_3093 == tmp_3096 { keys[2] = tmp_3033; values[2] = tmp_3034; }
    let tmp_3097 = keys[3] < tmp_3035 || (keys[3] == tmp_3035 && values[3] < tmp_3036);
    if tmp_3093 == tmp_3097 { keys[3] = tmp_3035; values[3] = tmp_3036; }
    let tmp_3098 = keys[4] < tmp_3037 || (keys[4] == tmp_3037 && values[4] < tmp_3038);
    if tmp_3093 == tmp_3098 { keys[4] = tmp_3037; values[4] = tmp_3038; }
    let tmp_3099 = keys[5] < tmp_3039 || (keys[5] == tmp_3039 && values[5] < tmp_3040);
    if tmp_3093 == tmp_3099 { keys[5] = tmp_3039; values[5] = tmp_3040; }
    let tmp_3100 = keys[6] < tmp_3041 || (keys[6] == tmp_3041 && values[6] < tmp_3042);
    if tmp_3093 == tmp_3100 { keys[6] = tmp_3041; values[6] = tmp_3042; }
    let tmp_3101 = keys[7] < tmp_3043 || (keys[7] == tmp_3043 && values[7] < tmp_3044);
    if tmp_3093 == tmp_3101 { keys[7] = tmp_3043; values[7] = tmp_3044; }
    let tmp_3102 = keys[8] < tmp_3045 || (keys[8] == tmp_3045 && values[8] < tmp_3046);
    if tmp_3093 == tmp_3102 { keys[8] = tmp_3045; values[8] = tmp_3046; }
    let tmp_3103 = keys[9] < tmp_3047 || (keys[9] == tmp_3047 && values[9] < tmp_3048);
    if tmp_3093 == tmp_3103 { keys[9] = tmp_3047; values[9] = tmp_3048; }
    let tmp_3104 = keys[10] < tmp_3049 || (keys[10] == tmp_3049 && values[10] < tmp_3050);
    if tmp_3093 == tmp_3104 { keys[10] = tmp_3049; values[10] = tmp_3050; }
    let tmp_3105 = keys[11] < tmp_3051 || (keys[11] == tmp_3051 && values[11] < tmp_3052);
    if tmp_3093 == tmp_3105 { keys[11] = tmp_3051; values[11] = tmp_3052; }
    let tmp_3106 = keys[12] < tmp_3053 || (keys[12] == tmp_3053 && values[12] < tmp_3054);
    if tmp_3093 == tmp_3106 { keys[12] = tmp_3053; values[12] = tmp_3054; }
    let tmp_3107 = keys[13] < tmp_3055 || (keys[13] == tmp_3055 && values[13] < tmp_3056);
    if tmp_3093 == tmp_3107 { keys[13] = tmp_3055; values[13] = tmp_3056; }
    let tmp_3108 = keys[14] < tmp_3057 || (keys[14] == tmp_3057 && values[14] < tmp_3058);
    if tmp_3093 == tmp_3108 { keys[14] = tmp_3057; values[14] = tmp_3058; }
    let tmp_3109 = keys[15] < tmp_3059 || (keys[15] == tmp_3059 && values[15] < tmp_3060);
    if tmp_3093 == tmp_3109 { keys[15] = tmp_3059; values[15] = tmp_3060; }
    let tmp_3110 = keys[16] < tmp_3061 || (keys[16] == tmp_3061 && values[16] < tmp_3062);
    if tmp_3093 == tmp_3110 { keys[16] = tmp_3061; values[16] = tmp_3062; }
    let tmp_3111 = keys[17] < tmp_3063 || (keys[17] == tmp_3063 && values[17] < tmp_3064);
    if tmp_3093 == tmp_3111 { keys[17] = tmp_3063; values[17] = tmp_3064; }
    let tmp_3112 = keys[18] < tmp_3065 || (keys[18] == tmp_3065 && values[18] < tmp_3066);
    if tmp_3093 == tmp_3112 { keys[18] = tmp_3065; values[18] = tmp_3066; }
    let tmp_3113 = keys[19] < tmp_3067 || (keys[19] == tmp_3067 && values[19] < tmp_3068);
    if tmp_3093 == tmp_3113 { keys[19] = tmp_3067; values[19] = tmp_3068; }
    let tmp_3114 = keys[20] < tmp_3069 || (keys[20] == tmp_3069 && values[20] < tmp_3070);
    if tmp_3093 == tmp_3114 { keys[20] = tmp_3069; values[20] = tmp_3070; }
    let tmp_3115 = keys[21] < tmp_3071 || (keys[21] == tmp_3071 && values[21] < tmp_3072);
    if tmp_3093 == tmp_3115 { keys[21] = tmp_3071; values[21] = tmp_3072; }
    let tmp_3116 = keys[22] < tmp_3073 || (keys[22] == tmp_3073 && values[22] < tmp_3074);
    if tmp_3093 == tmp_3116 { keys[22] = tmp_3073; values[22] = tmp_3074; }
    let tmp_3117 = keys[23] < tmp_3075 || (keys[23] == tmp_3075 && values[23] < tmp_3076);
    if tmp_3093 == tmp_3117 { keys[23] = tmp_3075; values[23] = tmp_3076; }
    let tmp_3118 = keys[24] < tmp_3077 || (keys[24] == tmp_3077 && values[24] < tmp_3078);
    if tmp_3093 == tmp_3118 { keys[24] = tmp_3077; values[24] = tmp_3078; }
    let tmp_3119 = keys[25] < tmp_3079 || (keys[25] == tmp_3079 && values[25] < tmp_3080);
    if tmp_3093 == tmp_3119 { keys[25] = tmp_3079; values[25] = tmp_3080; }
    let tmp_3120 = keys[26] < tmp_3081 || (keys[26] == tmp_3081 && values[26] < tmp_3082);
    if tmp_3093 == tmp_3120 { keys[26] = tmp_3081; values[26] = tmp_3082; }
    let tmp_3121 = keys[27] < tmp_3083 || (keys[27] == tmp_3083 && values[27] < tmp_3084);
    if tmp_3093 == tmp_3121 { keys[27] = tmp_3083; values[27] = tmp_3084; }
    let tmp_3122 = keys[28] < tmp_3085 || (keys[28] == tmp_3085 && values[28] < tmp_3086);
    if tmp_3093 == tmp_3122 { keys[28] = tmp_3085; values[28] = tmp_3086; }
    let tmp_3123 = keys[29] < tmp_3087 || (keys[29] == tmp_3087 && values[29] < tmp_3088);
    if tmp_3093 == tmp_3123 { keys[29] = tmp_3087; values[29] = tmp_3088; }
    let tmp_3124 = keys[30] < tmp_3089 || (keys[30] == tmp_3089 && values[30] < tmp_3090);
    if tmp_3093 == tmp_3124 { keys[30] = tmp_3089; values[30] = tmp_3090; }
    let tmp_3125 = keys[31] < tmp_3091 || (keys[31] == tmp_3091 && values[31] < tmp_3092);
    if tmp_3093 == tmp_3125 { keys[31] = tmp_3091; values[31] = tmp_3092; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:32) 
    {
    let tmp_3126 = subgroupShuffleXor(keys[0], 2u);
    let tmp_3127 = subgroupShuffleXor(values[0], 2u);
    let tmp_3128 = subgroupShuffleXor(keys[1], 2u);
    let tmp_3129 = subgroupShuffleXor(values[1], 2u);
    let tmp_3130 = subgroupShuffleXor(keys[2], 2u);
    let tmp_3131 = subgroupShuffleXor(values[2], 2u);
    let tmp_3132 = subgroupShuffleXor(keys[3], 2u);
    let tmp_3133 = subgroupShuffleXor(values[3], 2u);
    let tmp_3134 = subgroupShuffleXor(keys[4], 2u);
    let tmp_3135 = subgroupShuffleXor(values[4], 2u);
    let tmp_3136 = subgroupShuffleXor(keys[5], 2u);
    let tmp_3137 = subgroupShuffleXor(values[5], 2u);
    let tmp_3138 = subgroupShuffleXor(keys[6], 2u);
    let tmp_3139 = subgroupShuffleXor(values[6], 2u);
    let tmp_3140 = subgroupShuffleXor(keys[7], 2u);
    let tmp_3141 = subgroupShuffleXor(values[7], 2u);
    let tmp_3142 = subgroupShuffleXor(keys[8], 2u);
    let tmp_3143 = subgroupShuffleXor(values[8], 2u);
    let tmp_3144 = subgroupShuffleXor(keys[9], 2u);
    let tmp_3145 = subgroupShuffleXor(values[9], 2u);
    let tmp_3146 = subgroupShuffleXor(keys[10], 2u);
    let tmp_3147 = subgroupShuffleXor(values[10], 2u);
    let tmp_3148 = subgroupShuffleXor(keys[11], 2u);
    let tmp_3149 = subgroupShuffleXor(values[11], 2u);
    let tmp_3150 = subgroupShuffleXor(keys[12], 2u);
    let tmp_3151 = subgroupShuffleXor(values[12], 2u);
    let tmp_3152 = subgroupShuffleXor(keys[13], 2u);
    let tmp_3153 = subgroupShuffleXor(values[13], 2u);
    let tmp_3154 = subgroupShuffleXor(keys[14], 2u);
    let tmp_3155 = subgroupShuffleXor(values[14], 2u);
    let tmp_3156 = subgroupShuffleXor(keys[15], 2u);
    let tmp_3157 = subgroupShuffleXor(values[15], 2u);
    let tmp_3158 = subgroupShuffleXor(keys[16], 2u);
    let tmp_3159 = subgroupShuffleXor(values[16], 2u);
    let tmp_3160 = subgroupShuffleXor(keys[17], 2u);
    let tmp_3161 = subgroupShuffleXor(values[17], 2u);
    let tmp_3162 = subgroupShuffleXor(keys[18], 2u);
    let tmp_3163 = subgroupShuffleXor(values[18], 2u);
    let tmp_3164 = subgroupShuffleXor(keys[19], 2u);
    let tmp_3165 = subgroupShuffleXor(values[19], 2u);
    let tmp_3166 = subgroupShuffleXor(keys[20], 2u);
    let tmp_3167 = subgroupShuffleXor(values[20], 2u);
    let tmp_3168 = subgroupShuffleXor(keys[21], 2u);
    let tmp_3169 = subgroupShuffleXor(values[21], 2u);
    let tmp_3170 = subgroupShuffleXor(keys[22], 2u);
    let tmp_3171 = subgroupShuffleXor(values[22], 2u);
    let tmp_3172 = subgroupShuffleXor(keys[23], 2u);
    let tmp_3173 = subgroupShuffleXor(values[23], 2u);
    let tmp_3174 = subgroupShuffleXor(keys[24], 2u);
    let tmp_3175 = subgroupShuffleXor(values[24], 2u);
    let tmp_3176 = subgroupShuffleXor(keys[25], 2u);
    let tmp_3177 = subgroupShuffleXor(values[25], 2u);
    let tmp_3178 = subgroupShuffleXor(keys[26], 2u);
    let tmp_3179 = subgroupShuffleXor(values[26], 2u);
    let tmp_3180 = subgroupShuffleXor(keys[27], 2u);
    let tmp_3181 = subgroupShuffleXor(values[27], 2u);
    let tmp_3182 = subgroupShuffleXor(keys[28], 2u);
    let tmp_3183 = subgroupShuffleXor(values[28], 2u);
    let tmp_3184 = subgroupShuffleXor(keys[29], 2u);
    let tmp_3185 = subgroupShuffleXor(values[29], 2u);
    let tmp_3186 = subgroupShuffleXor(keys[30], 2u);
    let tmp_3187 = subgroupShuffleXor(values[30], 2u);
    let tmp_3188 = subgroupShuffleXor(keys[31], 2u);
    let tmp_3189 = subgroupShuffleXor(values[31], 2u);
    let tmp_3190 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_3191 = keys[0] < tmp_3126 || (keys[0] == tmp_3126 && values[0] < tmp_3127);
    if tmp_3190 == tmp_3191 { keys[0] = tmp_3126; values[0] = tmp_3127; }
    let tmp_3192 = keys[1] < tmp_3128 || (keys[1] == tmp_3128 && values[1] < tmp_3129);
    if tmp_3190 == tmp_3192 { keys[1] = tmp_3128; values[1] = tmp_3129; }
    let tmp_3193 = keys[2] < tmp_3130 || (keys[2] == tmp_3130 && values[2] < tmp_3131);
    if tmp_3190 == tmp_3193 { keys[2] = tmp_3130; values[2] = tmp_3131; }
    let tmp_3194 = keys[3] < tmp_3132 || (keys[3] == tmp_3132 && values[3] < tmp_3133);
    if tmp_3190 == tmp_3194 { keys[3] = tmp_3132; values[3] = tmp_3133; }
    let tmp_3195 = keys[4] < tmp_3134 || (keys[4] == tmp_3134 && values[4] < tmp_3135);
    if tmp_3190 == tmp_3195 { keys[4] = tmp_3134; values[4] = tmp_3135; }
    let tmp_3196 = keys[5] < tmp_3136 || (keys[5] == tmp_3136 && values[5] < tmp_3137);
    if tmp_3190 == tmp_3196 { keys[5] = tmp_3136; values[5] = tmp_3137; }
    let tmp_3197 = keys[6] < tmp_3138 || (keys[6] == tmp_3138 && values[6] < tmp_3139);
    if tmp_3190 == tmp_3197 { keys[6] = tmp_3138; values[6] = tmp_3139; }
    let tmp_3198 = keys[7] < tmp_3140 || (keys[7] == tmp_3140 && values[7] < tmp_3141);
    if tmp_3190 == tmp_3198 { keys[7] = tmp_3140; values[7] = tmp_3141; }
    let tmp_3199 = keys[8] < tmp_3142 || (keys[8] == tmp_3142 && values[8] < tmp_3143);
    if tmp_3190 == tmp_3199 { keys[8] = tmp_3142; values[8] = tmp_3143; }
    let tmp_3200 = keys[9] < tmp_3144 || (keys[9] == tmp_3144 && values[9] < tmp_3145);
    if tmp_3190 == tmp_3200 { keys[9] = tmp_3144; values[9] = tmp_3145; }
    let tmp_3201 = keys[10] < tmp_3146 || (keys[10] == tmp_3146 && values[10] < tmp_3147);
    if tmp_3190 == tmp_3201 { keys[10] = tmp_3146; values[10] = tmp_3147; }
    let tmp_3202 = keys[11] < tmp_3148 || (keys[11] == tmp_3148 && values[11] < tmp_3149);
    if tmp_3190 == tmp_3202 { keys[11] = tmp_3148; values[11] = tmp_3149; }
    let tmp_3203 = keys[12] < tmp_3150 || (keys[12] == tmp_3150 && values[12] < tmp_3151);
    if tmp_3190 == tmp_3203 { keys[12] = tmp_3150; values[12] = tmp_3151; }
    let tmp_3204 = keys[13] < tmp_3152 || (keys[13] == tmp_3152 && values[13] < tmp_3153);
    if tmp_3190 == tmp_3204 { keys[13] = tmp_3152; values[13] = tmp_3153; }
    let tmp_3205 = keys[14] < tmp_3154 || (keys[14] == tmp_3154 && values[14] < tmp_3155);
    if tmp_3190 == tmp_3205 { keys[14] = tmp_3154; values[14] = tmp_3155; }
    let tmp_3206 = keys[15] < tmp_3156 || (keys[15] == tmp_3156 && values[15] < tmp_3157);
    if tmp_3190 == tmp_3206 { keys[15] = tmp_3156; values[15] = tmp_3157; }
    let tmp_3207 = keys[16] < tmp_3158 || (keys[16] == tmp_3158 && values[16] < tmp_3159);
    if tmp_3190 == tmp_3207 { keys[16] = tmp_3158; values[16] = tmp_3159; }
    let tmp_3208 = keys[17] < tmp_3160 || (keys[17] == tmp_3160 && values[17] < tmp_3161);
    if tmp_3190 == tmp_3208 { keys[17] = tmp_3160; values[17] = tmp_3161; }
    let tmp_3209 = keys[18] < tmp_3162 || (keys[18] == tmp_3162 && values[18] < tmp_3163);
    if tmp_3190 == tmp_3209 { keys[18] = tmp_3162; values[18] = tmp_3163; }
    let tmp_3210 = keys[19] < tmp_3164 || (keys[19] == tmp_3164 && values[19] < tmp_3165);
    if tmp_3190 == tmp_3210 { keys[19] = tmp_3164; values[19] = tmp_3165; }
    let tmp_3211 = keys[20] < tmp_3166 || (keys[20] == tmp_3166 && values[20] < tmp_3167);
    if tmp_3190 == tmp_3211 { keys[20] = tmp_3166; values[20] = tmp_3167; }
    let tmp_3212 = keys[21] < tmp_3168 || (keys[21] == tmp_3168 && values[21] < tmp_3169);
    if tmp_3190 == tmp_3212 { keys[21] = tmp_3168; values[21] = tmp_3169; }
    let tmp_3213 = keys[22] < tmp_3170 || (keys[22] == tmp_3170 && values[22] < tmp_3171);
    if tmp_3190 == tmp_3213 { keys[22] = tmp_3170; values[22] = tmp_3171; }
    let tmp_3214 = keys[23] < tmp_3172 || (keys[23] == tmp_3172 && values[23] < tmp_3173);
    if tmp_3190 == tmp_3214 { keys[23] = tmp_3172; values[23] = tmp_3173; }
    let tmp_3215 = keys[24] < tmp_3174 || (keys[24] == tmp_3174 && values[24] < tmp_3175);
    if tmp_3190 == tmp_3215 { keys[24] = tmp_3174; values[24] = tmp_3175; }
    let tmp_3216 = keys[25] < tmp_3176 || (keys[25] == tmp_3176 && values[25] < tmp_3177);
    if tmp_3190 == tmp_3216 { keys[25] = tmp_3176; values[25] = tmp_3177; }
    let tmp_3217 = keys[26] < tmp_3178 || (keys[26] == tmp_3178 && values[26] < tmp_3179);
    if tmp_3190 == tmp_3217 { keys[26] = tmp_3178; values[26] = tmp_3179; }
    let tmp_3218 = keys[27] < tmp_3180 || (keys[27] == tmp_3180 && values[27] < tmp_3181);
    if tmp_3190 == tmp_3218 { keys[27] = tmp_3180; values[27] = tmp_3181; }
    let tmp_3219 = keys[28] < tmp_3182 || (keys[28] == tmp_3182 && values[28] < tmp_3183);
    if tmp_3190 == tmp_3219 { keys[28] = tmp_3182; values[28] = tmp_3183; }
    let tmp_3220 = keys[29] < tmp_3184 || (keys[29] == tmp_3184 && values[29] < tmp_3185);
    if tmp_3190 == tmp_3220 { keys[29] = tmp_3184; values[29] = tmp_3185; }
    let tmp_3221 = keys[30] < tmp_3186 || (keys[30] == tmp_3186 && values[30] < tmp_3187);
    if tmp_3190 == tmp_3221 { keys[30] = tmp_3186; values[30] = tmp_3187; }
    let tmp_3222 = keys[31] < tmp_3188 || (keys[31] == tmp_3188 && values[31] < tmp_3189);
    if tmp_3190 == tmp_3222 { keys[31] = tmp_3188; values[31] = tmp_3189; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:32) 
    {
    let tmp_3223 = subgroupShuffleXor(keys[0], 1u);
    let tmp_3224 = subgroupShuffleXor(values[0], 1u);
    let tmp_3225 = subgroupShuffleXor(keys[1], 1u);
    let tmp_3226 = subgroupShuffleXor(values[1], 1u);
    let tmp_3227 = subgroupShuffleXor(keys[2], 1u);
    let tmp_3228 = subgroupShuffleXor(values[2], 1u);
    let tmp_3229 = subgroupShuffleXor(keys[3], 1u);
    let tmp_3230 = subgroupShuffleXor(values[3], 1u);
    let tmp_3231 = subgroupShuffleXor(keys[4], 1u);
    let tmp_3232 = subgroupShuffleXor(values[4], 1u);
    let tmp_3233 = subgroupShuffleXor(keys[5], 1u);
    let tmp_3234 = subgroupShuffleXor(values[5], 1u);
    let tmp_3235 = subgroupShuffleXor(keys[6], 1u);
    let tmp_3236 = subgroupShuffleXor(values[6], 1u);
    let tmp_3237 = subgroupShuffleXor(keys[7], 1u);
    let tmp_3238 = subgroupShuffleXor(values[7], 1u);
    let tmp_3239 = subgroupShuffleXor(keys[8], 1u);
    let tmp_3240 = subgroupShuffleXor(values[8], 1u);
    let tmp_3241 = subgroupShuffleXor(keys[9], 1u);
    let tmp_3242 = subgroupShuffleXor(values[9], 1u);
    let tmp_3243 = subgroupShuffleXor(keys[10], 1u);
    let tmp_3244 = subgroupShuffleXor(values[10], 1u);
    let tmp_3245 = subgroupShuffleXor(keys[11], 1u);
    let tmp_3246 = subgroupShuffleXor(values[11], 1u);
    let tmp_3247 = subgroupShuffleXor(keys[12], 1u);
    let tmp_3248 = subgroupShuffleXor(values[12], 1u);
    let tmp_3249 = subgroupShuffleXor(keys[13], 1u);
    let tmp_3250 = subgroupShuffleXor(values[13], 1u);
    let tmp_3251 = subgroupShuffleXor(keys[14], 1u);
    let tmp_3252 = subgroupShuffleXor(values[14], 1u);
    let tmp_3253 = subgroupShuffleXor(keys[15], 1u);
    let tmp_3254 = subgroupShuffleXor(values[15], 1u);
    let tmp_3255 = subgroupShuffleXor(keys[16], 1u);
    let tmp_3256 = subgroupShuffleXor(values[16], 1u);
    let tmp_3257 = subgroupShuffleXor(keys[17], 1u);
    let tmp_3258 = subgroupShuffleXor(values[17], 1u);
    let tmp_3259 = subgroupShuffleXor(keys[18], 1u);
    let tmp_3260 = subgroupShuffleXor(values[18], 1u);
    let tmp_3261 = subgroupShuffleXor(keys[19], 1u);
    let tmp_3262 = subgroupShuffleXor(values[19], 1u);
    let tmp_3263 = subgroupShuffleXor(keys[20], 1u);
    let tmp_3264 = subgroupShuffleXor(values[20], 1u);
    let tmp_3265 = subgroupShuffleXor(keys[21], 1u);
    let tmp_3266 = subgroupShuffleXor(values[21], 1u);
    let tmp_3267 = subgroupShuffleXor(keys[22], 1u);
    let tmp_3268 = subgroupShuffleXor(values[22], 1u);
    let tmp_3269 = subgroupShuffleXor(keys[23], 1u);
    let tmp_3270 = subgroupShuffleXor(values[23], 1u);
    let tmp_3271 = subgroupShuffleXor(keys[24], 1u);
    let tmp_3272 = subgroupShuffleXor(values[24], 1u);
    let tmp_3273 = subgroupShuffleXor(keys[25], 1u);
    let tmp_3274 = subgroupShuffleXor(values[25], 1u);
    let tmp_3275 = subgroupShuffleXor(keys[26], 1u);
    let tmp_3276 = subgroupShuffleXor(values[26], 1u);
    let tmp_3277 = subgroupShuffleXor(keys[27], 1u);
    let tmp_3278 = subgroupShuffleXor(values[27], 1u);
    let tmp_3279 = subgroupShuffleXor(keys[28], 1u);
    let tmp_3280 = subgroupShuffleXor(values[28], 1u);
    let tmp_3281 = subgroupShuffleXor(keys[29], 1u);
    let tmp_3282 = subgroupShuffleXor(values[29], 1u);
    let tmp_3283 = subgroupShuffleXor(keys[30], 1u);
    let tmp_3284 = subgroupShuffleXor(values[30], 1u);
    let tmp_3285 = subgroupShuffleXor(keys[31], 1u);
    let tmp_3286 = subgroupShuffleXor(values[31], 1u);
    let tmp_3287 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_3288 = keys[0] < tmp_3223 || (keys[0] == tmp_3223 && values[0] < tmp_3224);
    if tmp_3287 == tmp_3288 { keys[0] = tmp_3223; values[0] = tmp_3224; }
    let tmp_3289 = keys[1] < tmp_3225 || (keys[1] == tmp_3225 && values[1] < tmp_3226);
    if tmp_3287 == tmp_3289 { keys[1] = tmp_3225; values[1] = tmp_3226; }
    let tmp_3290 = keys[2] < tmp_3227 || (keys[2] == tmp_3227 && values[2] < tmp_3228);
    if tmp_3287 == tmp_3290 { keys[2] = tmp_3227; values[2] = tmp_3228; }
    let tmp_3291 = keys[3] < tmp_3229 || (keys[3] == tmp_3229 && values[3] < tmp_3230);
    if tmp_3287 == tmp_3291 { keys[3] = tmp_3229; values[3] = tmp_3230; }
    let tmp_3292 = keys[4] < tmp_3231 || (keys[4] == tmp_3231 && values[4] < tmp_3232);
    if tmp_3287 == tmp_3292 { keys[4] = tmp_3231; values[4] = tmp_3232; }
    let tmp_3293 = keys[5] < tmp_3233 || (keys[5] == tmp_3233 && values[5] < tmp_3234);
    if tmp_3287 == tmp_3293 { keys[5] = tmp_3233; values[5] = tmp_3234; }
    let tmp_3294 = keys[6] < tmp_3235 || (keys[6] == tmp_3235 && values[6] < tmp_3236);
    if tmp_3287 == tmp_3294 { keys[6] = tmp_3235; values[6] = tmp_3236; }
    let tmp_3295 = keys[7] < tmp_3237 || (keys[7] == tmp_3237 && values[7] < tmp_3238);
    if tmp_3287 == tmp_3295 { keys[7] = tmp_3237; values[7] = tmp_3238; }
    let tmp_3296 = keys[8] < tmp_3239 || (keys[8] == tmp_3239 && values[8] < tmp_3240);
    if tmp_3287 == tmp_3296 { keys[8] = tmp_3239; values[8] = tmp_3240; }
    let tmp_3297 = keys[9] < tmp_3241 || (keys[9] == tmp_3241 && values[9] < tmp_3242);
    if tmp_3287 == tmp_3297 { keys[9] = tmp_3241; values[9] = tmp_3242; }
    let tmp_3298 = keys[10] < tmp_3243 || (keys[10] == tmp_3243 && values[10] < tmp_3244);
    if tmp_3287 == tmp_3298 { keys[10] = tmp_3243; values[10] = tmp_3244; }
    let tmp_3299 = keys[11] < tmp_3245 || (keys[11] == tmp_3245 && values[11] < tmp_3246);
    if tmp_3287 == tmp_3299 { keys[11] = tmp_3245; values[11] = tmp_3246; }
    let tmp_3300 = keys[12] < tmp_3247 || (keys[12] == tmp_3247 && values[12] < tmp_3248);
    if tmp_3287 == tmp_3300 { keys[12] = tmp_3247; values[12] = tmp_3248; }
    let tmp_3301 = keys[13] < tmp_3249 || (keys[13] == tmp_3249 && values[13] < tmp_3250);
    if tmp_3287 == tmp_3301 { keys[13] = tmp_3249; values[13] = tmp_3250; }
    let tmp_3302 = keys[14] < tmp_3251 || (keys[14] == tmp_3251 && values[14] < tmp_3252);
    if tmp_3287 == tmp_3302 { keys[14] = tmp_3251; values[14] = tmp_3252; }
    let tmp_3303 = keys[15] < tmp_3253 || (keys[15] == tmp_3253 && values[15] < tmp_3254);
    if tmp_3287 == tmp_3303 { keys[15] = tmp_3253; values[15] = tmp_3254; }
    let tmp_3304 = keys[16] < tmp_3255 || (keys[16] == tmp_3255 && values[16] < tmp_3256);
    if tmp_3287 == tmp_3304 { keys[16] = tmp_3255; values[16] = tmp_3256; }
    let tmp_3305 = keys[17] < tmp_3257 || (keys[17] == tmp_3257 && values[17] < tmp_3258);
    if tmp_3287 == tmp_3305 { keys[17] = tmp_3257; values[17] = tmp_3258; }
    let tmp_3306 = keys[18] < tmp_3259 || (keys[18] == tmp_3259 && values[18] < tmp_3260);
    if tmp_3287 == tmp_3306 { keys[18] = tmp_3259; values[18] = tmp_3260; }
    let tmp_3307 = keys[19] < tmp_3261 || (keys[19] == tmp_3261 && values[19] < tmp_3262);
    if tmp_3287 == tmp_3307 { keys[19] = tmp_3261; values[19] = tmp_3262; }
    let tmp_3308 = keys[20] < tmp_3263 || (keys[20] == tmp_3263 && values[20] < tmp_3264);
    if tmp_3287 == tmp_3308 { keys[20] = tmp_3263; values[20] = tmp_3264; }
    let tmp_3309 = keys[21] < tmp_3265 || (keys[21] == tmp_3265 && values[21] < tmp_3266);
    if tmp_3287 == tmp_3309 { keys[21] = tmp_3265; values[21] = tmp_3266; }
    let tmp_3310 = keys[22] < tmp_3267 || (keys[22] == tmp_3267 && values[22] < tmp_3268);
    if tmp_3287 == tmp_3310 { keys[22] = tmp_3267; values[22] = tmp_3268; }
    let tmp_3311 = keys[23] < tmp_3269 || (keys[23] == tmp_3269 && values[23] < tmp_3270);
    if tmp_3287 == tmp_3311 { keys[23] = tmp_3269; values[23] = tmp_3270; }
    let tmp_3312 = keys[24] < tmp_3271 || (keys[24] == tmp_3271 && values[24] < tmp_3272);
    if tmp_3287 == tmp_3312 { keys[24] = tmp_3271; values[24] = tmp_3272; }
    let tmp_3313 = keys[25] < tmp_3273 || (keys[25] == tmp_3273 && values[25] < tmp_3274);
    if tmp_3287 == tmp_3313 { keys[25] = tmp_3273; values[25] = tmp_3274; }
    let tmp_3314 = keys[26] < tmp_3275 || (keys[26] == tmp_3275 && values[26] < tmp_3276);
    if tmp_3287 == tmp_3314 { keys[26] = tmp_3275; values[26] = tmp_3276; }
    let tmp_3315 = keys[27] < tmp_3277 || (keys[27] == tmp_3277 && values[27] < tmp_3278);
    if tmp_3287 == tmp_3315 { keys[27] = tmp_3277; values[27] = tmp_3278; }
    let tmp_3316 = keys[28] < tmp_3279 || (keys[28] == tmp_3279 && values[28] < tmp_3280);
    if tmp_3287 == tmp_3316 { keys[28] = tmp_3279; values[28] = tmp_3280; }
    let tmp_3317 = keys[29] < tmp_3281 || (keys[29] == tmp_3281 && values[29] < tmp_3282);
    if tmp_3287 == tmp_3317 { keys[29] = tmp_3281; values[29] = tmp_3282; }
    let tmp_3318 = keys[30] < tmp_3283 || (keys[30] == tmp_3283 && values[30] < tmp_3284);
    if tmp_3287 == tmp_3318 { keys[30] = tmp_3283; values[30] = tmp_3284; }
    let tmp_3319 = keys[31] < tmp_3285 || (keys[31] == tmp_3285 && values[31] < tmp_3286);
    if tmp_3287 == tmp_3319 { keys[31] = tmp_3285; values[31] = tmp_3286; }
    }
    // exch_local(16,32) 
    // cmp_swap(0,16)
    if keys[0] > keys[16] || (keys[0] == keys[16] && values[0] > values[16]) {
    // swap(0,16) 
    { let tmp_3320 = keys[0]; keys[0] = keys[16]; keys[16] = tmp_3320;let tmp_3321 = values[0]; values[0] = values[16]; values[16] = tmp_3321; }
    }
    // cmp_swap(1,17)
    if keys[1] > keys[17] || (keys[1] == keys[17] && values[1] > values[17]) {
    // swap(1,17) 
    { let tmp_3322 = keys[1]; keys[1] = keys[17]; keys[17] = tmp_3322;let tmp_3323 = values[1]; values[1] = values[17]; values[17] = tmp_3323; }
    }
    // cmp_swap(2,18)
    if keys[2] > keys[18] || (keys[2] == keys[18] && values[2] > values[18]) {
    // swap(2,18) 
    { let tmp_3324 = keys[2]; keys[2] = keys[18]; keys[18] = tmp_3324;let tmp_3325 = values[2]; values[2] = values[18]; values[18] = tmp_3325; }
    }
    // cmp_swap(3,19)
    if keys[3] > keys[19] || (keys[3] == keys[19] && values[3] > values[19]) {
    // swap(3,19) 
    { let tmp_3326 = keys[3]; keys[3] = keys[19]; keys[19] = tmp_3326;let tmp_3327 = values[3]; values[3] = values[19]; values[19] = tmp_3327; }
    }
    // cmp_swap(4,20)
    if keys[4] > keys[20] || (keys[4] == keys[20] && values[4] > values[20]) {
    // swap(4,20) 
    { let tmp_3328 = keys[4]; keys[4] = keys[20]; keys[20] = tmp_3328;let tmp_3329 = values[4]; values[4] = values[20]; values[20] = tmp_3329; }
    }
    // cmp_swap(5,21)
    if keys[5] > keys[21] || (keys[5] == keys[21] && values[5] > values[21]) {
    // swap(5,21) 
    { let tmp_3330 = keys[5]; keys[5] = keys[21]; keys[21] = tmp_3330;let tmp_3331 = values[5]; values[5] = values[21]; values[21] = tmp_3331; }
    }
    // cmp_swap(6,22)
    if keys[6] > keys[22] || (keys[6] == keys[22] && values[6] > values[22]) {
    // swap(6,22) 
    { let tmp_3332 = keys[6]; keys[6] = keys[22]; keys[22] = tmp_3332;let tmp_3333 = values[6]; values[6] = values[22]; values[22] = tmp_3333; }
    }
    // cmp_swap(7,23)
    if keys[7] > keys[23] || (keys[7] == keys[23] && values[7] > values[23]) {
    // swap(7,23) 
    { let tmp_3334 = keys[7]; keys[7] = keys[23]; keys[23] = tmp_3334;let tmp_3335 = values[7]; values[7] = values[23]; values[23] = tmp_3335; }
    }
    // cmp_swap(8,24)
    if keys[8] > keys[24] || (keys[8] == keys[24] && values[8] > values[24]) {
    // swap(8,24) 
    { let tmp_3336 = keys[8]; keys[8] = keys[24]; keys[24] = tmp_3336;let tmp_3337 = values[8]; values[8] = values[24]; values[24] = tmp_3337; }
    }
    // cmp_swap(9,25)
    if keys[9] > keys[25] || (keys[9] == keys[25] && values[9] > values[25]) {
    // swap(9,25) 
    { let tmp_3338 = keys[9]; keys[9] = keys[25]; keys[25] = tmp_3338;let tmp_3339 = values[9]; values[9] = values[25]; values[25] = tmp_3339; }
    }
    // cmp_swap(10,26)
    if keys[10] > keys[26] || (keys[10] == keys[26] && values[10] > values[26]) {
    // swap(10,26) 
    { let tmp_3340 = keys[10]; keys[10] = keys[26]; keys[26] = tmp_3340;let tmp_3341 = values[10]; values[10] = values[26]; values[26] = tmp_3341; }
    }
    // cmp_swap(11,27)
    if keys[11] > keys[27] || (keys[11] == keys[27] && values[11] > values[27]) {
    // swap(11,27) 
    { let tmp_3342 = keys[11]; keys[11] = keys[27]; keys[27] = tmp_3342;let tmp_3343 = values[11]; values[11] = values[27]; values[27] = tmp_3343; }
    }
    // cmp_swap(12,28)
    if keys[12] > keys[28] || (keys[12] == keys[28] && values[12] > values[28]) {
    // swap(12,28) 
    { let tmp_3344 = keys[12]; keys[12] = keys[28]; keys[28] = tmp_3344;let tmp_3345 = values[12]; values[12] = values[28]; values[28] = tmp_3345; }
    }
    // cmp_swap(13,29)
    if keys[13] > keys[29] || (keys[13] == keys[29] && values[13] > values[29]) {
    // swap(13,29) 
    { let tmp_3346 = keys[13]; keys[13] = keys[29]; keys[29] = tmp_3346;let tmp_3347 = values[13]; values[13] = values[29]; values[29] = tmp_3347; }
    }
    // cmp_swap(14,30)
    if keys[14] > keys[30] || (keys[14] == keys[30] && values[14] > values[30]) {
    // swap(14,30) 
    { let tmp_3348 = keys[14]; keys[14] = keys[30]; keys[30] = tmp_3348;let tmp_3349 = values[14]; values[14] = values[30]; values[30] = tmp_3349; }
    }
    // cmp_swap(15,31)
    if keys[15] > keys[31] || (keys[15] == keys[31] && values[15] > values[31]) {
    // swap(15,31) 
    { let tmp_3350 = keys[15]; keys[15] = keys[31]; keys[31] = tmp_3350;let tmp_3351 = values[15]; values[15] = values[31]; values[31] = tmp_3351; }
    }
    // exch_local(8,32) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_3352 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_3352;let tmp_3353 = values[0]; values[0] = values[8]; values[8] = tmp_3353; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_3354 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_3354;let tmp_3355 = values[1]; values[1] = values[9]; values[9] = tmp_3355; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_3356 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_3356;let tmp_3357 = values[2]; values[2] = values[10]; values[10] = tmp_3357; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_3358 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_3358;let tmp_3359 = values[3]; values[3] = values[11]; values[11] = tmp_3359; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_3360 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_3360;let tmp_3361 = values[4]; values[4] = values[12]; values[12] = tmp_3361; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_3362 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_3362;let tmp_3363 = values[5]; values[5] = values[13]; values[13] = tmp_3363; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_3364 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_3364;let tmp_3365 = values[6]; values[6] = values[14]; values[14] = tmp_3365; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_3366 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_3366;let tmp_3367 = values[7]; values[7] = values[15]; values[15] = tmp_3367; }
    }
    // cmp_swap(16,24)
    if keys[16] > keys[24] || (keys[16] == keys[24] && values[16] > values[24]) {
    // swap(16,24) 
    { let tmp_3368 = keys[16]; keys[16] = keys[24]; keys[24] = tmp_3368;let tmp_3369 = values[16]; values[16] = values[24]; values[24] = tmp_3369; }
    }
    // cmp_swap(17,25)
    if keys[17] > keys[25] || (keys[17] == keys[25] && values[17] > values[25]) {
    // swap(17,25) 
    { let tmp_3370 = keys[17]; keys[17] = keys[25]; keys[25] = tmp_3370;let tmp_3371 = values[17]; values[17] = values[25]; values[25] = tmp_3371; }
    }
    // cmp_swap(18,26)
    if keys[18] > keys[26] || (keys[18] == keys[26] && values[18] > values[26]) {
    // swap(18,26) 
    { let tmp_3372 = keys[18]; keys[18] = keys[26]; keys[26] = tmp_3372;let tmp_3373 = values[18]; values[18] = values[26]; values[26] = tmp_3373; }
    }
    // cmp_swap(19,27)
    if keys[19] > keys[27] || (keys[19] == keys[27] && values[19] > values[27]) {
    // swap(19,27) 
    { let tmp_3374 = keys[19]; keys[19] = keys[27]; keys[27] = tmp_3374;let tmp_3375 = values[19]; values[19] = values[27]; values[27] = tmp_3375; }
    }
    // cmp_swap(20,28)
    if keys[20] > keys[28] || (keys[20] == keys[28] && values[20] > values[28]) {
    // swap(20,28) 
    { let tmp_3376 = keys[20]; keys[20] = keys[28]; keys[28] = tmp_3376;let tmp_3377 = values[20]; values[20] = values[28]; values[28] = tmp_3377; }
    }
    // cmp_swap(21,29)
    if keys[21] > keys[29] || (keys[21] == keys[29] && values[21] > values[29]) {
    // swap(21,29) 
    { let tmp_3378 = keys[21]; keys[21] = keys[29]; keys[29] = tmp_3378;let tmp_3379 = values[21]; values[21] = values[29]; values[29] = tmp_3379; }
    }
    // cmp_swap(22,30)
    if keys[22] > keys[30] || (keys[22] == keys[30] && values[22] > values[30]) {
    // swap(22,30) 
    { let tmp_3380 = keys[22]; keys[22] = keys[30]; keys[30] = tmp_3380;let tmp_3381 = values[22]; values[22] = values[30]; values[30] = tmp_3381; }
    }
    // cmp_swap(23,31)
    if keys[23] > keys[31] || (keys[23] == keys[31] && values[23] > values[31]) {
    // swap(23,31) 
    { let tmp_3382 = keys[23]; keys[23] = keys[31]; keys[31] = tmp_3382;let tmp_3383 = values[23]; values[23] = values[31]; values[31] = tmp_3383; }
    }
    // exch_local(4,32) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_3384 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_3384;let tmp_3385 = values[0]; values[0] = values[4]; values[4] = tmp_3385; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_3386 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_3386;let tmp_3387 = values[1]; values[1] = values[5]; values[5] = tmp_3387; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_3388 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_3388;let tmp_3389 = values[2]; values[2] = values[6]; values[6] = tmp_3389; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_3390 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_3390;let tmp_3391 = values[3]; values[3] = values[7]; values[7] = tmp_3391; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_3392 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_3392;let tmp_3393 = values[8]; values[8] = values[12]; values[12] = tmp_3393; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_3394 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_3394;let tmp_3395 = values[9]; values[9] = values[13]; values[13] = tmp_3395; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_3396 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_3396;let tmp_3397 = values[10]; values[10] = values[14]; values[14] = tmp_3397; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_3398 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_3398;let tmp_3399 = values[11]; values[11] = values[15]; values[15] = tmp_3399; }
    }
    // cmp_swap(16,20)
    if keys[16] > keys[20] || (keys[16] == keys[20] && values[16] > values[20]) {
    // swap(16,20) 
    { let tmp_3400 = keys[16]; keys[16] = keys[20]; keys[20] = tmp_3400;let tmp_3401 = values[16]; values[16] = values[20]; values[20] = tmp_3401; }
    }
    // cmp_swap(17,21)
    if keys[17] > keys[21] || (keys[17] == keys[21] && values[17] > values[21]) {
    // swap(17,21) 
    { let tmp_3402 = keys[17]; keys[17] = keys[21]; keys[21] = tmp_3402;let tmp_3403 = values[17]; values[17] = values[21]; values[21] = tmp_3403; }
    }
    // cmp_swap(18,22)
    if keys[18] > keys[22] || (keys[18] == keys[22] && values[18] > values[22]) {
    // swap(18,22) 
    { let tmp_3404 = keys[18]; keys[18] = keys[22]; keys[22] = tmp_3404;let tmp_3405 = values[18]; values[18] = values[22]; values[22] = tmp_3405; }
    }
    // cmp_swap(19,23)
    if keys[19] > keys[23] || (keys[19] == keys[23] && values[19] > values[23]) {
    // swap(19,23) 
    { let tmp_3406 = keys[19]; keys[19] = keys[23]; keys[23] = tmp_3406;let tmp_3407 = values[19]; values[19] = values[23]; values[23] = tmp_3407; }
    }
    // cmp_swap(24,28)
    if keys[24] > keys[28] || (keys[24] == keys[28] && values[24] > values[28]) {
    // swap(24,28) 
    { let tmp_3408 = keys[24]; keys[24] = keys[28]; keys[28] = tmp_3408;let tmp_3409 = values[24]; values[24] = values[28]; values[28] = tmp_3409; }
    }
    // cmp_swap(25,29)
    if keys[25] > keys[29] || (keys[25] == keys[29] && values[25] > values[29]) {
    // swap(25,29) 
    { let tmp_3410 = keys[25]; keys[25] = keys[29]; keys[29] = tmp_3410;let tmp_3411 = values[25]; values[25] = values[29]; values[29] = tmp_3411; }
    }
    // cmp_swap(26,30)
    if keys[26] > keys[30] || (keys[26] == keys[30] && values[26] > values[30]) {
    // swap(26,30) 
    { let tmp_3412 = keys[26]; keys[26] = keys[30]; keys[30] = tmp_3412;let tmp_3413 = values[26]; values[26] = values[30]; values[30] = tmp_3413; }
    }
    // cmp_swap(27,31)
    if keys[27] > keys[31] || (keys[27] == keys[31] && values[27] > values[31]) {
    // swap(27,31) 
    { let tmp_3414 = keys[27]; keys[27] = keys[31]; keys[31] = tmp_3414;let tmp_3415 = values[27]; values[27] = values[31]; values[31] = tmp_3415; }
    }
    // exch_local(2,32) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_3416 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_3416;let tmp_3417 = values[0]; values[0] = values[2]; values[2] = tmp_3417; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_3418 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_3418;let tmp_3419 = values[1]; values[1] = values[3]; values[3] = tmp_3419; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_3420 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_3420;let tmp_3421 = values[4]; values[4] = values[6]; values[6] = tmp_3421; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_3422 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_3422;let tmp_3423 = values[5]; values[5] = values[7]; values[7] = tmp_3423; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_3424 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_3424;let tmp_3425 = values[8]; values[8] = values[10]; values[10] = tmp_3425; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_3426 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_3426;let tmp_3427 = values[9]; values[9] = values[11]; values[11] = tmp_3427; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_3428 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_3428;let tmp_3429 = values[12]; values[12] = values[14]; values[14] = tmp_3429; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_3430 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_3430;let tmp_3431 = values[13]; values[13] = values[15]; values[15] = tmp_3431; }
    }
    // cmp_swap(16,18)
    if keys[16] > keys[18] || (keys[16] == keys[18] && values[16] > values[18]) {
    // swap(16,18) 
    { let tmp_3432 = keys[16]; keys[16] = keys[18]; keys[18] = tmp_3432;let tmp_3433 = values[16]; values[16] = values[18]; values[18] = tmp_3433; }
    }
    // cmp_swap(17,19)
    if keys[17] > keys[19] || (keys[17] == keys[19] && values[17] > values[19]) {
    // swap(17,19) 
    { let tmp_3434 = keys[17]; keys[17] = keys[19]; keys[19] = tmp_3434;let tmp_3435 = values[17]; values[17] = values[19]; values[19] = tmp_3435; }
    }
    // cmp_swap(20,22)
    if keys[20] > keys[22] || (keys[20] == keys[22] && values[20] > values[22]) {
    // swap(20,22) 
    { let tmp_3436 = keys[20]; keys[20] = keys[22]; keys[22] = tmp_3436;let tmp_3437 = values[20]; values[20] = values[22]; values[22] = tmp_3437; }
    }
    // cmp_swap(21,23)
    if keys[21] > keys[23] || (keys[21] == keys[23] && values[21] > values[23]) {
    // swap(21,23) 
    { let tmp_3438 = keys[21]; keys[21] = keys[23]; keys[23] = tmp_3438;let tmp_3439 = values[21]; values[21] = values[23]; values[23] = tmp_3439; }
    }
    // cmp_swap(24,26)
    if keys[24] > keys[26] || (keys[24] == keys[26] && values[24] > values[26]) {
    // swap(24,26) 
    { let tmp_3440 = keys[24]; keys[24] = keys[26]; keys[26] = tmp_3440;let tmp_3441 = values[24]; values[24] = values[26]; values[26] = tmp_3441; }
    }
    // cmp_swap(25,27)
    if keys[25] > keys[27] || (keys[25] == keys[27] && values[25] > values[27]) {
    // swap(25,27) 
    { let tmp_3442 = keys[25]; keys[25] = keys[27]; keys[27] = tmp_3442;let tmp_3443 = values[25]; values[25] = values[27]; values[27] = tmp_3443; }
    }
    // cmp_swap(28,30)
    if keys[28] > keys[30] || (keys[28] == keys[30] && values[28] > values[30]) {
    // swap(28,30) 
    { let tmp_3444 = keys[28]; keys[28] = keys[30]; keys[30] = tmp_3444;let tmp_3445 = values[28]; values[28] = values[30]; values[30] = tmp_3445; }
    }
    // cmp_swap(29,31)
    if keys[29] > keys[31] || (keys[29] == keys[31] && values[29] > values[31]) {
    // swap(29,31) 
    { let tmp_3446 = keys[29]; keys[29] = keys[31]; keys[31] = tmp_3446;let tmp_3447 = values[29]; values[29] = values[31]; values[31] = tmp_3447; }
    }
    // exch_local(1,32) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_3448 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_3448;let tmp_3449 = values[0]; values[0] = values[1]; values[1] = tmp_3449; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_3450 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_3450;let tmp_3451 = values[2]; values[2] = values[3]; values[3] = tmp_3451; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_3452 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_3452;let tmp_3453 = values[4]; values[4] = values[5]; values[5] = tmp_3453; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_3454 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_3454;let tmp_3455 = values[6]; values[6] = values[7]; values[7] = tmp_3455; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_3456 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_3456;let tmp_3457 = values[8]; values[8] = values[9]; values[9] = tmp_3457; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_3458 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_3458;let tmp_3459 = values[10]; values[10] = values[11]; values[11] = tmp_3459; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_3460 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_3460;let tmp_3461 = values[12]; values[12] = values[13]; values[13] = tmp_3461; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_3462 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_3462;let tmp_3463 = values[14]; values[14] = values[15]; values[15] = tmp_3463; }
    }
    // cmp_swap(16,17)
    if keys[16] > keys[17] || (keys[16] == keys[17] && values[16] > values[17]) {
    // swap(16,17) 
    { let tmp_3464 = keys[16]; keys[16] = keys[17]; keys[17] = tmp_3464;let tmp_3465 = values[16]; values[16] = values[17]; values[17] = tmp_3465; }
    }
    // cmp_swap(18,19)
    if keys[18] > keys[19] || (keys[18] == keys[19] && values[18] > values[19]) {
    // swap(18,19) 
    { let tmp_3466 = keys[18]; keys[18] = keys[19]; keys[19] = tmp_3466;let tmp_3467 = values[18]; values[18] = values[19]; values[19] = tmp_3467; }
    }
    // cmp_swap(20,21)
    if keys[20] > keys[21] || (keys[20] == keys[21] && values[20] > values[21]) {
    // swap(20,21) 
    { let tmp_3468 = keys[20]; keys[20] = keys[21]; keys[21] = tmp_3468;let tmp_3469 = values[20]; values[20] = values[21]; values[21] = tmp_3469; }
    }
    // cmp_swap(22,23)
    if keys[22] > keys[23] || (keys[22] == keys[23] && values[22] > values[23]) {
    // swap(22,23) 
    { let tmp_3470 = keys[22]; keys[22] = keys[23]; keys[23] = tmp_3470;let tmp_3471 = values[22]; values[22] = values[23]; values[23] = tmp_3471; }
    }
    // cmp_swap(24,25)
    if keys[24] > keys[25] || (keys[24] == keys[25] && values[24] > values[25]) {
    // swap(24,25) 
    { let tmp_3472 = keys[24]; keys[24] = keys[25]; keys[25] = tmp_3472;let tmp_3473 = values[24]; values[24] = values[25]; values[25] = tmp_3473; }
    }
    // cmp_swap(26,27)
    if keys[26] > keys[27] || (keys[26] == keys[27] && values[26] > values[27]) {
    // swap(26,27) 
    { let tmp_3474 = keys[26]; keys[26] = keys[27]; keys[27] = tmp_3474;let tmp_3475 = values[26]; values[26] = values[27]; values[27] = tmp_3475; }
    }
    // cmp_swap(28,29)
    if keys[28] > keys[29] || (keys[28] == keys[29] && values[28] > values[29]) {
    // swap(28,29) 
    { let tmp_3476 = keys[28]; keys[28] = keys[29]; keys[29] = tmp_3476;let tmp_3477 = values[28]; values[28] = values[29]; values[29] = tmp_3477; }
    }
    // cmp_swap(30,31)
    if keys[30] > keys[31] || (keys[30] == keys[31] && values[30] > values[31]) {
    // swap(30,31) 
    { let tmp_3478 = keys[30]; keys[30] = keys[31]; keys[31] = tmp_3478;let tmp_3479 = values[30]; values[30] = values[31]; values[31] = tmp_3479; }
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
