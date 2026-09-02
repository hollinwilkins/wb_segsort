
enable subgroups;

override WG: u32 = 16u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 512u;
const M: u32 = 16u;
const WPT: u32 = 32u;
const SG: u32 = 8u;      // register run spans one subgroup: RUN = SG*WPT = 256

var<workgroup> smem_keys: array<u32, N>;
var<workgroup> smem_vals: array<u32, N>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybmerge_sg8_smem16k_n512_m16_block(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 9u;

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

    // ---- phase 1: per-subgroup register run-sort (RUN = SG*WPT elements) ----
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

    // stage the sorted runs into shared memory (blocked layout)
    for (var r = 0u; r < WPT; r = r + 1u) {
        smem_keys[local_tid * WPT + r] = keys[r];
        smem_vals[local_tid * WPT + r] = values[r];
    }
    workgroupBarrier();

    let base = local_tid * WPT;   // this thread's blocked output range [base, base+WPT)

    // ---- phase 2: recursive merge-path merges through shared memory ----
    // merge pass 0: two sorted runs of 256 -> 512 (register-staged)
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
        var out_keys: array<u32, 32>;
        var out_vals: array<u32, 32>;
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

    // ---- phase 3: coalesced store from the final buffer ----
    for (var r = 0u; r < WPT; r = r + 1u) {
        let pos = local_tid * WPT + r;
        if is_active && pos < seg_size {
            global_keys[seg_start + pos] = smem_keys[pos];
            global_value_indices[seg_start + pos] = smem_vals[pos];
        }
    }
}
