
override WG: u32 = 16u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 256u;
const M: u32 = 16u;
const WPT: u32 = 16u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n256_m16_block(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 8u;

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

    var keys: array<u32, 16>;
    var values: array<u32, 16>;

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

    // exch_local(1,16) 
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
    // exch_local(3,16) 
    // cmp_swap(0,3)
    if keys[0] > keys[3] || (keys[0] == keys[3] && values[0] > values[3]) {
    // swap(0,3) 
    { let tmp_16 = keys[0]; keys[0] = keys[3]; keys[3] = tmp_16;let tmp_17 = values[0]; values[0] = values[3]; values[3] = tmp_17; }
    }
    // cmp_swap(1,2)
    if keys[1] > keys[2] || (keys[1] == keys[2] && values[1] > values[2]) {
    // swap(1,2) 
    { let tmp_18 = keys[1]; keys[1] = keys[2]; keys[2] = tmp_18;let tmp_19 = values[1]; values[1] = values[2]; values[2] = tmp_19; }
    }
    // cmp_swap(4,7)
    if keys[4] > keys[7] || (keys[4] == keys[7] && values[4] > values[7]) {
    // swap(4,7) 
    { let tmp_20 = keys[4]; keys[4] = keys[7]; keys[7] = tmp_20;let tmp_21 = values[4]; values[4] = values[7]; values[7] = tmp_21; }
    }
    // cmp_swap(5,6)
    if keys[5] > keys[6] || (keys[5] == keys[6] && values[5] > values[6]) {
    // swap(5,6) 
    { let tmp_22 = keys[5]; keys[5] = keys[6]; keys[6] = tmp_22;let tmp_23 = values[5]; values[5] = values[6]; values[6] = tmp_23; }
    }
    // cmp_swap(8,11)
    if keys[8] > keys[11] || (keys[8] == keys[11] && values[8] > values[11]) {
    // swap(8,11) 
    { let tmp_24 = keys[8]; keys[8] = keys[11]; keys[11] = tmp_24;let tmp_25 = values[8]; values[8] = values[11]; values[11] = tmp_25; }
    }
    // cmp_swap(9,10)
    if keys[9] > keys[10] || (keys[9] == keys[10] && values[9] > values[10]) {
    // swap(9,10) 
    { let tmp_26 = keys[9]; keys[9] = keys[10]; keys[10] = tmp_26;let tmp_27 = values[9]; values[9] = values[10]; values[10] = tmp_27; }
    }
    // cmp_swap(12,15)
    if keys[12] > keys[15] || (keys[12] == keys[15] && values[12] > values[15]) {
    // swap(12,15) 
    { let tmp_28 = keys[12]; keys[12] = keys[15]; keys[15] = tmp_28;let tmp_29 = values[12]; values[12] = values[15]; values[15] = tmp_29; }
    }
    // cmp_swap(13,14)
    if keys[13] > keys[14] || (keys[13] == keys[14] && values[13] > values[14]) {
    // swap(13,14) 
    { let tmp_30 = keys[13]; keys[13] = keys[14]; keys[14] = tmp_30;let tmp_31 = values[13]; values[13] = values[14]; values[14] = tmp_31; }
    }
    // exch_local(1,16) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_32 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_32;let tmp_33 = values[0]; values[0] = values[1]; values[1] = tmp_33; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_34 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_34;let tmp_35 = values[2]; values[2] = values[3]; values[3] = tmp_35; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_36 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_36;let tmp_37 = values[4]; values[4] = values[5]; values[5] = tmp_37; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_38 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_38;let tmp_39 = values[6]; values[6] = values[7]; values[7] = tmp_39; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_40 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_40;let tmp_41 = values[8]; values[8] = values[9]; values[9] = tmp_41; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_42 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_42;let tmp_43 = values[10]; values[10] = values[11]; values[11] = tmp_43; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_44 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_44;let tmp_45 = values[12]; values[12] = values[13]; values[13] = tmp_45; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_46 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_46;let tmp_47 = values[14]; values[14] = values[15]; values[15] = tmp_47; }
    }
    // exch_local(7,16) 
    // cmp_swap(0,7)
    if keys[0] > keys[7] || (keys[0] == keys[7] && values[0] > values[7]) {
    // swap(0,7) 
    { let tmp_48 = keys[0]; keys[0] = keys[7]; keys[7] = tmp_48;let tmp_49 = values[0]; values[0] = values[7]; values[7] = tmp_49; }
    }
    // cmp_swap(1,6)
    if keys[1] > keys[6] || (keys[1] == keys[6] && values[1] > values[6]) {
    // swap(1,6) 
    { let tmp_50 = keys[1]; keys[1] = keys[6]; keys[6] = tmp_50;let tmp_51 = values[1]; values[1] = values[6]; values[6] = tmp_51; }
    }
    // cmp_swap(2,5)
    if keys[2] > keys[5] || (keys[2] == keys[5] && values[2] > values[5]) {
    // swap(2,5) 
    { let tmp_52 = keys[2]; keys[2] = keys[5]; keys[5] = tmp_52;let tmp_53 = values[2]; values[2] = values[5]; values[5] = tmp_53; }
    }
    // cmp_swap(3,4)
    if keys[3] > keys[4] || (keys[3] == keys[4] && values[3] > values[4]) {
    // swap(3,4) 
    { let tmp_54 = keys[3]; keys[3] = keys[4]; keys[4] = tmp_54;let tmp_55 = values[3]; values[3] = values[4]; values[4] = tmp_55; }
    }
    // cmp_swap(8,15)
    if keys[8] > keys[15] || (keys[8] == keys[15] && values[8] > values[15]) {
    // swap(8,15) 
    { let tmp_56 = keys[8]; keys[8] = keys[15]; keys[15] = tmp_56;let tmp_57 = values[8]; values[8] = values[15]; values[15] = tmp_57; }
    }
    // cmp_swap(9,14)
    if keys[9] > keys[14] || (keys[9] == keys[14] && values[9] > values[14]) {
    // swap(9,14) 
    { let tmp_58 = keys[9]; keys[9] = keys[14]; keys[14] = tmp_58;let tmp_59 = values[9]; values[9] = values[14]; values[14] = tmp_59; }
    }
    // cmp_swap(10,13)
    if keys[10] > keys[13] || (keys[10] == keys[13] && values[10] > values[13]) {
    // swap(10,13) 
    { let tmp_60 = keys[10]; keys[10] = keys[13]; keys[13] = tmp_60;let tmp_61 = values[10]; values[10] = values[13]; values[13] = tmp_61; }
    }
    // cmp_swap(11,12)
    if keys[11] > keys[12] || (keys[11] == keys[12] && values[11] > values[12]) {
    // swap(11,12) 
    { let tmp_62 = keys[11]; keys[11] = keys[12]; keys[12] = tmp_62;let tmp_63 = values[11]; values[11] = values[12]; values[12] = tmp_63; }
    }
    // exch_local(2,16) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_64 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_64;let tmp_65 = values[0]; values[0] = values[2]; values[2] = tmp_65; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_66 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_66;let tmp_67 = values[1]; values[1] = values[3]; values[3] = tmp_67; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_68 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_68;let tmp_69 = values[4]; values[4] = values[6]; values[6] = tmp_69; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_70 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_70;let tmp_71 = values[5]; values[5] = values[7]; values[7] = tmp_71; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_72 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_72;let tmp_73 = values[8]; values[8] = values[10]; values[10] = tmp_73; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_74 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_74;let tmp_75 = values[9]; values[9] = values[11]; values[11] = tmp_75; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_76 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_76;let tmp_77 = values[12]; values[12] = values[14]; values[14] = tmp_77; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_78 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_78;let tmp_79 = values[13]; values[13] = values[15]; values[15] = tmp_79; }
    }
    // exch_local(1,16) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_80 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_80;let tmp_81 = values[0]; values[0] = values[1]; values[1] = tmp_81; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_82 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_82;let tmp_83 = values[2]; values[2] = values[3]; values[3] = tmp_83; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_84 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_84;let tmp_85 = values[4]; values[4] = values[5]; values[5] = tmp_85; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_86 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_86;let tmp_87 = values[6]; values[6] = values[7]; values[7] = tmp_87; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_88 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_88;let tmp_89 = values[8]; values[8] = values[9]; values[9] = tmp_89; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_90 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_90;let tmp_91 = values[10]; values[10] = values[11]; values[11] = tmp_91; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_92 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_92;let tmp_93 = values[12]; values[12] = values[13]; values[13] = tmp_93; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_94 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_94;let tmp_95 = values[14]; values[14] = values[15]; values[15] = tmp_95; }
    }
    // exch_local(15,16) 
    // cmp_swap(0,15)
    if keys[0] > keys[15] || (keys[0] == keys[15] && values[0] > values[15]) {
    // swap(0,15) 
    { let tmp_96 = keys[0]; keys[0] = keys[15]; keys[15] = tmp_96;let tmp_97 = values[0]; values[0] = values[15]; values[15] = tmp_97; }
    }
    // cmp_swap(1,14)
    if keys[1] > keys[14] || (keys[1] == keys[14] && values[1] > values[14]) {
    // swap(1,14) 
    { let tmp_98 = keys[1]; keys[1] = keys[14]; keys[14] = tmp_98;let tmp_99 = values[1]; values[1] = values[14]; values[14] = tmp_99; }
    }
    // cmp_swap(2,13)
    if keys[2] > keys[13] || (keys[2] == keys[13] && values[2] > values[13]) {
    // swap(2,13) 
    { let tmp_100 = keys[2]; keys[2] = keys[13]; keys[13] = tmp_100;let tmp_101 = values[2]; values[2] = values[13]; values[13] = tmp_101; }
    }
    // cmp_swap(3,12)
    if keys[3] > keys[12] || (keys[3] == keys[12] && values[3] > values[12]) {
    // swap(3,12) 
    { let tmp_102 = keys[3]; keys[3] = keys[12]; keys[12] = tmp_102;let tmp_103 = values[3]; values[3] = values[12]; values[12] = tmp_103; }
    }
    // cmp_swap(4,11)
    if keys[4] > keys[11] || (keys[4] == keys[11] && values[4] > values[11]) {
    // swap(4,11) 
    { let tmp_104 = keys[4]; keys[4] = keys[11]; keys[11] = tmp_104;let tmp_105 = values[4]; values[4] = values[11]; values[11] = tmp_105; }
    }
    // cmp_swap(5,10)
    if keys[5] > keys[10] || (keys[5] == keys[10] && values[5] > values[10]) {
    // swap(5,10) 
    { let tmp_106 = keys[5]; keys[5] = keys[10]; keys[10] = tmp_106;let tmp_107 = values[5]; values[5] = values[10]; values[10] = tmp_107; }
    }
    // cmp_swap(6,9)
    if keys[6] > keys[9] || (keys[6] == keys[9] && values[6] > values[9]) {
    // swap(6,9) 
    { let tmp_108 = keys[6]; keys[6] = keys[9]; keys[9] = tmp_108;let tmp_109 = values[6]; values[6] = values[9]; values[9] = tmp_109; }
    }
    // cmp_swap(7,8)
    if keys[7] > keys[8] || (keys[7] == keys[8] && values[7] > values[8]) {
    // swap(7,8) 
    { let tmp_110 = keys[7]; keys[7] = keys[8]; keys[8] = tmp_110;let tmp_111 = values[7]; values[7] = values[8]; values[8] = tmp_111; }
    }
    // exch_local(4,16) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_112 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_112;let tmp_113 = values[0]; values[0] = values[4]; values[4] = tmp_113; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_114 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_114;let tmp_115 = values[1]; values[1] = values[5]; values[5] = tmp_115; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_116 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_116;let tmp_117 = values[2]; values[2] = values[6]; values[6] = tmp_117; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_118 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_118;let tmp_119 = values[3]; values[3] = values[7]; values[7] = tmp_119; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_120 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_120;let tmp_121 = values[8]; values[8] = values[12]; values[12] = tmp_121; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_122 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_122;let tmp_123 = values[9]; values[9] = values[13]; values[13] = tmp_123; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_124 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_124;let tmp_125 = values[10]; values[10] = values[14]; values[14] = tmp_125; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_126 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_126;let tmp_127 = values[11]; values[11] = values[15]; values[15] = tmp_127; }
    }
    // exch_local(2,16) 
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
    // exch_local(1,16) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_144 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_144;let tmp_145 = values[0]; values[0] = values[1]; values[1] = tmp_145; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_146 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_146;let tmp_147 = values[2]; values[2] = values[3]; values[3] = tmp_147; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_148 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_148;let tmp_149 = values[4]; values[4] = values[5]; values[5] = tmp_149; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_150 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_150;let tmp_151 = values[6]; values[6] = values[7]; values[7] = tmp_151; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_152 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_152;let tmp_153 = values[8]; values[8] = values[9]; values[9] = tmp_153; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_154 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_154;let tmp_155 = values[10]; values[10] = values[11]; values[11] = tmp_155; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_156 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_156;let tmp_157 = values[12]; values[12] = values[13]; values[13] = tmp_157; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_158 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_158;let tmp_159 = values[14]; values[14] = values[15]; values[15] = tmp_159; }
    }
    // exch_intxn(tmask:1,swbit:0,wpt:16)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_160 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_161 = seg_base + (local_tid ^ 1u); let tmp_162 = smem_keys[tmp_161 * WPT + 15u]; let tmp_163 = smem_vals[tmp_161 * WPT + 15u]; let tmp_164 = keys[0] < tmp_162 || (keys[0] == tmp_162 && values[0] < tmp_163); if tmp_160 == tmp_164 { keys[0] = tmp_162; values[0] = tmp_163; } let tmp_165 = smem_keys[tmp_161 * WPT + 14u]; let tmp_166 = smem_vals[tmp_161 * WPT + 14u]; let tmp_167 = keys[1] < tmp_165 || (keys[1] == tmp_165 && values[1] < tmp_166); if tmp_160 == tmp_167 { keys[1] = tmp_165; values[1] = tmp_166; } let tmp_168 = smem_keys[tmp_161 * WPT + 13u]; let tmp_169 = smem_vals[tmp_161 * WPT + 13u]; let tmp_170 = keys[2] < tmp_168 || (keys[2] == tmp_168 && values[2] < tmp_169); if tmp_160 == tmp_170 { keys[2] = tmp_168; values[2] = tmp_169; } let tmp_171 = smem_keys[tmp_161 * WPT + 12u]; let tmp_172 = smem_vals[tmp_161 * WPT + 12u]; let tmp_173 = keys[3] < tmp_171 || (keys[3] == tmp_171 && values[3] < tmp_172); if tmp_160 == tmp_173 { keys[3] = tmp_171; values[3] = tmp_172; } let tmp_174 = smem_keys[tmp_161 * WPT + 11u]; let tmp_175 = smem_vals[tmp_161 * WPT + 11u]; let tmp_176 = keys[4] < tmp_174 || (keys[4] == tmp_174 && values[4] < tmp_175); if tmp_160 == tmp_176 { keys[4] = tmp_174; values[4] = tmp_175; } let tmp_177 = smem_keys[tmp_161 * WPT + 10u]; let tmp_178 = smem_vals[tmp_161 * WPT + 10u]; let tmp_179 = keys[5] < tmp_177 || (keys[5] == tmp_177 && values[5] < tmp_178); if tmp_160 == tmp_179 { keys[5] = tmp_177; values[5] = tmp_178; } let tmp_180 = smem_keys[tmp_161 * WPT + 9u]; let tmp_181 = smem_vals[tmp_161 * WPT + 9u]; let tmp_182 = keys[6] < tmp_180 || (keys[6] == tmp_180 && values[6] < tmp_181); if tmp_160 == tmp_182 { keys[6] = tmp_180; values[6] = tmp_181; } let tmp_183 = smem_keys[tmp_161 * WPT + 8u]; let tmp_184 = smem_vals[tmp_161 * WPT + 8u]; let tmp_185 = keys[7] < tmp_183 || (keys[7] == tmp_183 && values[7] < tmp_184); if tmp_160 == tmp_185 { keys[7] = tmp_183; values[7] = tmp_184; } let tmp_186 = smem_keys[tmp_161 * WPT + 7u]; let tmp_187 = smem_vals[tmp_161 * WPT + 7u]; let tmp_188 = keys[8] < tmp_186 || (keys[8] == tmp_186 && values[8] < tmp_187); if tmp_160 == tmp_188 { keys[8] = tmp_186; values[8] = tmp_187; } let tmp_189 = smem_keys[tmp_161 * WPT + 6u]; let tmp_190 = smem_vals[tmp_161 * WPT + 6u]; let tmp_191 = keys[9] < tmp_189 || (keys[9] == tmp_189 && values[9] < tmp_190); if tmp_160 == tmp_191 { keys[9] = tmp_189; values[9] = tmp_190; } let tmp_192 = smem_keys[tmp_161 * WPT + 5u]; let tmp_193 = smem_vals[tmp_161 * WPT + 5u]; let tmp_194 = keys[10] < tmp_192 || (keys[10] == tmp_192 && values[10] < tmp_193); if tmp_160 == tmp_194 { keys[10] = tmp_192; values[10] = tmp_193; } let tmp_195 = smem_keys[tmp_161 * WPT + 4u]; let tmp_196 = smem_vals[tmp_161 * WPT + 4u]; let tmp_197 = keys[11] < tmp_195 || (keys[11] == tmp_195 && values[11] < tmp_196); if tmp_160 == tmp_197 { keys[11] = tmp_195; values[11] = tmp_196; } let tmp_198 = smem_keys[tmp_161 * WPT + 3u]; let tmp_199 = smem_vals[tmp_161 * WPT + 3u]; let tmp_200 = keys[12] < tmp_198 || (keys[12] == tmp_198 && values[12] < tmp_199); if tmp_160 == tmp_200 { keys[12] = tmp_198; values[12] = tmp_199; } let tmp_201 = smem_keys[tmp_161 * WPT + 2u]; let tmp_202 = smem_vals[tmp_161 * WPT + 2u]; let tmp_203 = keys[13] < tmp_201 || (keys[13] == tmp_201 && values[13] < tmp_202); if tmp_160 == tmp_203 { keys[13] = tmp_201; values[13] = tmp_202; } let tmp_204 = smem_keys[tmp_161 * WPT + 1u]; let tmp_205 = smem_vals[tmp_161 * WPT + 1u]; let tmp_206 = keys[14] < tmp_204 || (keys[14] == tmp_204 && values[14] < tmp_205); if tmp_160 == tmp_206 { keys[14] = tmp_204; values[14] = tmp_205; } let tmp_207 = smem_keys[tmp_161 * WPT + 0u]; let tmp_208 = smem_vals[tmp_161 * WPT + 0u]; let tmp_209 = keys[15] < tmp_207 || (keys[15] == tmp_207 && values[15] < tmp_208); if tmp_160 == tmp_209 { keys[15] = tmp_207; values[15] = tmp_208; } workgroupBarrier(); }
    // exch_local(8,16) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_210 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_210;let tmp_211 = values[0]; values[0] = values[8]; values[8] = tmp_211; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_212 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_212;let tmp_213 = values[1]; values[1] = values[9]; values[9] = tmp_213; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_214 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_214;let tmp_215 = values[2]; values[2] = values[10]; values[10] = tmp_215; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_216 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_216;let tmp_217 = values[3]; values[3] = values[11]; values[11] = tmp_217; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_218 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_218;let tmp_219 = values[4]; values[4] = values[12]; values[12] = tmp_219; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_220 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_220;let tmp_221 = values[5]; values[5] = values[13]; values[13] = tmp_221; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_222 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_222;let tmp_223 = values[6]; values[6] = values[14]; values[14] = tmp_223; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_224 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_224;let tmp_225 = values[7]; values[7] = values[15]; values[15] = tmp_225; }
    }
    // exch_local(4,16) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_226 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_226;let tmp_227 = values[0]; values[0] = values[4]; values[4] = tmp_227; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_228 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_228;let tmp_229 = values[1]; values[1] = values[5]; values[5] = tmp_229; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_230 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_230;let tmp_231 = values[2]; values[2] = values[6]; values[6] = tmp_231; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_232 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_232;let tmp_233 = values[3]; values[3] = values[7]; values[7] = tmp_233; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_234 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_234;let tmp_235 = values[8]; values[8] = values[12]; values[12] = tmp_235; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_236 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_236;let tmp_237 = values[9]; values[9] = values[13]; values[13] = tmp_237; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_238 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_238;let tmp_239 = values[10]; values[10] = values[14]; values[14] = tmp_239; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_240 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_240;let tmp_241 = values[11]; values[11] = values[15]; values[15] = tmp_241; }
    }
    // exch_local(2,16) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_242 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_242;let tmp_243 = values[0]; values[0] = values[2]; values[2] = tmp_243; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_244 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_244;let tmp_245 = values[1]; values[1] = values[3]; values[3] = tmp_245; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_246 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_246;let tmp_247 = values[4]; values[4] = values[6]; values[6] = tmp_247; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_248 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_248;let tmp_249 = values[5]; values[5] = values[7]; values[7] = tmp_249; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_250 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_250;let tmp_251 = values[8]; values[8] = values[10]; values[10] = tmp_251; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_252 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_252;let tmp_253 = values[9]; values[9] = values[11]; values[11] = tmp_253; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_254 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_254;let tmp_255 = values[12]; values[12] = values[14]; values[14] = tmp_255; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_256 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_256;let tmp_257 = values[13]; values[13] = values[15]; values[15] = tmp_257; }
    }
    // exch_local(1,16) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_258 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_258;let tmp_259 = values[0]; values[0] = values[1]; values[1] = tmp_259; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_260 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_260;let tmp_261 = values[2]; values[2] = values[3]; values[3] = tmp_261; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_262 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_262;let tmp_263 = values[4]; values[4] = values[5]; values[5] = tmp_263; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_264 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_264;let tmp_265 = values[6]; values[6] = values[7]; values[7] = tmp_265; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_266 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_266;let tmp_267 = values[8]; values[8] = values[9]; values[9] = tmp_267; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_268 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_268;let tmp_269 = values[10]; values[10] = values[11]; values[11] = tmp_269; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_270 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_270;let tmp_271 = values[12]; values[12] = values[13]; values[13] = tmp_271; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_272 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_272;let tmp_273 = values[14]; values[14] = values[15]; values[15] = tmp_273; }
    }
    // exch_intxn(tmask:3,swbit:1,wpt:16)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_274 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_275 = seg_base + (local_tid ^ 3u); let tmp_276 = smem_keys[tmp_275 * WPT + 15u]; let tmp_277 = smem_vals[tmp_275 * WPT + 15u]; let tmp_278 = keys[0] < tmp_276 || (keys[0] == tmp_276 && values[0] < tmp_277); if tmp_274 == tmp_278 { keys[0] = tmp_276; values[0] = tmp_277; } let tmp_279 = smem_keys[tmp_275 * WPT + 14u]; let tmp_280 = smem_vals[tmp_275 * WPT + 14u]; let tmp_281 = keys[1] < tmp_279 || (keys[1] == tmp_279 && values[1] < tmp_280); if tmp_274 == tmp_281 { keys[1] = tmp_279; values[1] = tmp_280; } let tmp_282 = smem_keys[tmp_275 * WPT + 13u]; let tmp_283 = smem_vals[tmp_275 * WPT + 13u]; let tmp_284 = keys[2] < tmp_282 || (keys[2] == tmp_282 && values[2] < tmp_283); if tmp_274 == tmp_284 { keys[2] = tmp_282; values[2] = tmp_283; } let tmp_285 = smem_keys[tmp_275 * WPT + 12u]; let tmp_286 = smem_vals[tmp_275 * WPT + 12u]; let tmp_287 = keys[3] < tmp_285 || (keys[3] == tmp_285 && values[3] < tmp_286); if tmp_274 == tmp_287 { keys[3] = tmp_285; values[3] = tmp_286; } let tmp_288 = smem_keys[tmp_275 * WPT + 11u]; let tmp_289 = smem_vals[tmp_275 * WPT + 11u]; let tmp_290 = keys[4] < tmp_288 || (keys[4] == tmp_288 && values[4] < tmp_289); if tmp_274 == tmp_290 { keys[4] = tmp_288; values[4] = tmp_289; } let tmp_291 = smem_keys[tmp_275 * WPT + 10u]; let tmp_292 = smem_vals[tmp_275 * WPT + 10u]; let tmp_293 = keys[5] < tmp_291 || (keys[5] == tmp_291 && values[5] < tmp_292); if tmp_274 == tmp_293 { keys[5] = tmp_291; values[5] = tmp_292; } let tmp_294 = smem_keys[tmp_275 * WPT + 9u]; let tmp_295 = smem_vals[tmp_275 * WPT + 9u]; let tmp_296 = keys[6] < tmp_294 || (keys[6] == tmp_294 && values[6] < tmp_295); if tmp_274 == tmp_296 { keys[6] = tmp_294; values[6] = tmp_295; } let tmp_297 = smem_keys[tmp_275 * WPT + 8u]; let tmp_298 = smem_vals[tmp_275 * WPT + 8u]; let tmp_299 = keys[7] < tmp_297 || (keys[7] == tmp_297 && values[7] < tmp_298); if tmp_274 == tmp_299 { keys[7] = tmp_297; values[7] = tmp_298; } let tmp_300 = smem_keys[tmp_275 * WPT + 7u]; let tmp_301 = smem_vals[tmp_275 * WPT + 7u]; let tmp_302 = keys[8] < tmp_300 || (keys[8] == tmp_300 && values[8] < tmp_301); if tmp_274 == tmp_302 { keys[8] = tmp_300; values[8] = tmp_301; } let tmp_303 = smem_keys[tmp_275 * WPT + 6u]; let tmp_304 = smem_vals[tmp_275 * WPT + 6u]; let tmp_305 = keys[9] < tmp_303 || (keys[9] == tmp_303 && values[9] < tmp_304); if tmp_274 == tmp_305 { keys[9] = tmp_303; values[9] = tmp_304; } let tmp_306 = smem_keys[tmp_275 * WPT + 5u]; let tmp_307 = smem_vals[tmp_275 * WPT + 5u]; let tmp_308 = keys[10] < tmp_306 || (keys[10] == tmp_306 && values[10] < tmp_307); if tmp_274 == tmp_308 { keys[10] = tmp_306; values[10] = tmp_307; } let tmp_309 = smem_keys[tmp_275 * WPT + 4u]; let tmp_310 = smem_vals[tmp_275 * WPT + 4u]; let tmp_311 = keys[11] < tmp_309 || (keys[11] == tmp_309 && values[11] < tmp_310); if tmp_274 == tmp_311 { keys[11] = tmp_309; values[11] = tmp_310; } let tmp_312 = smem_keys[tmp_275 * WPT + 3u]; let tmp_313 = smem_vals[tmp_275 * WPT + 3u]; let tmp_314 = keys[12] < tmp_312 || (keys[12] == tmp_312 && values[12] < tmp_313); if tmp_274 == tmp_314 { keys[12] = tmp_312; values[12] = tmp_313; } let tmp_315 = smem_keys[tmp_275 * WPT + 2u]; let tmp_316 = smem_vals[tmp_275 * WPT + 2u]; let tmp_317 = keys[13] < tmp_315 || (keys[13] == tmp_315 && values[13] < tmp_316); if tmp_274 == tmp_317 { keys[13] = tmp_315; values[13] = tmp_316; } let tmp_318 = smem_keys[tmp_275 * WPT + 1u]; let tmp_319 = smem_vals[tmp_275 * WPT + 1u]; let tmp_320 = keys[14] < tmp_318 || (keys[14] == tmp_318 && values[14] < tmp_319); if tmp_274 == tmp_320 { keys[14] = tmp_318; values[14] = tmp_319; } let tmp_321 = smem_keys[tmp_275 * WPT + 0u]; let tmp_322 = smem_vals[tmp_275 * WPT + 0u]; let tmp_323 = keys[15] < tmp_321 || (keys[15] == tmp_321 && values[15] < tmp_322); if tmp_274 == tmp_323 { keys[15] = tmp_321; values[15] = tmp_322; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:16) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_324 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_325 = seg_base + (local_tid ^ 1u); let tmp_326 = smem_keys[tmp_325 * WPT + 0u]; let tmp_327 = smem_vals[tmp_325 * WPT + 0u]; let tmp_328 = keys[0] < tmp_326 || (keys[0] == tmp_326 && values[0] < tmp_327); if tmp_324 == tmp_328 { keys[0] = tmp_326; values[0] = tmp_327; } let tmp_329 = smem_keys[tmp_325 * WPT + 1u]; let tmp_330 = smem_vals[tmp_325 * WPT + 1u]; let tmp_331 = keys[1] < tmp_329 || (keys[1] == tmp_329 && values[1] < tmp_330); if tmp_324 == tmp_331 { keys[1] = tmp_329; values[1] = tmp_330; } let tmp_332 = smem_keys[tmp_325 * WPT + 2u]; let tmp_333 = smem_vals[tmp_325 * WPT + 2u]; let tmp_334 = keys[2] < tmp_332 || (keys[2] == tmp_332 && values[2] < tmp_333); if tmp_324 == tmp_334 { keys[2] = tmp_332; values[2] = tmp_333; } let tmp_335 = smem_keys[tmp_325 * WPT + 3u]; let tmp_336 = smem_vals[tmp_325 * WPT + 3u]; let tmp_337 = keys[3] < tmp_335 || (keys[3] == tmp_335 && values[3] < tmp_336); if tmp_324 == tmp_337 { keys[3] = tmp_335; values[3] = tmp_336; } let tmp_338 = smem_keys[tmp_325 * WPT + 4u]; let tmp_339 = smem_vals[tmp_325 * WPT + 4u]; let tmp_340 = keys[4] < tmp_338 || (keys[4] == tmp_338 && values[4] < tmp_339); if tmp_324 == tmp_340 { keys[4] = tmp_338; values[4] = tmp_339; } let tmp_341 = smem_keys[tmp_325 * WPT + 5u]; let tmp_342 = smem_vals[tmp_325 * WPT + 5u]; let tmp_343 = keys[5] < tmp_341 || (keys[5] == tmp_341 && values[5] < tmp_342); if tmp_324 == tmp_343 { keys[5] = tmp_341; values[5] = tmp_342; } let tmp_344 = smem_keys[tmp_325 * WPT + 6u]; let tmp_345 = smem_vals[tmp_325 * WPT + 6u]; let tmp_346 = keys[6] < tmp_344 || (keys[6] == tmp_344 && values[6] < tmp_345); if tmp_324 == tmp_346 { keys[6] = tmp_344; values[6] = tmp_345; } let tmp_347 = smem_keys[tmp_325 * WPT + 7u]; let tmp_348 = smem_vals[tmp_325 * WPT + 7u]; let tmp_349 = keys[7] < tmp_347 || (keys[7] == tmp_347 && values[7] < tmp_348); if tmp_324 == tmp_349 { keys[7] = tmp_347; values[7] = tmp_348; } let tmp_350 = smem_keys[tmp_325 * WPT + 8u]; let tmp_351 = smem_vals[tmp_325 * WPT + 8u]; let tmp_352 = keys[8] < tmp_350 || (keys[8] == tmp_350 && values[8] < tmp_351); if tmp_324 == tmp_352 { keys[8] = tmp_350; values[8] = tmp_351; } let tmp_353 = smem_keys[tmp_325 * WPT + 9u]; let tmp_354 = smem_vals[tmp_325 * WPT + 9u]; let tmp_355 = keys[9] < tmp_353 || (keys[9] == tmp_353 && values[9] < tmp_354); if tmp_324 == tmp_355 { keys[9] = tmp_353; values[9] = tmp_354; } let tmp_356 = smem_keys[tmp_325 * WPT + 10u]; let tmp_357 = smem_vals[tmp_325 * WPT + 10u]; let tmp_358 = keys[10] < tmp_356 || (keys[10] == tmp_356 && values[10] < tmp_357); if tmp_324 == tmp_358 { keys[10] = tmp_356; values[10] = tmp_357; } let tmp_359 = smem_keys[tmp_325 * WPT + 11u]; let tmp_360 = smem_vals[tmp_325 * WPT + 11u]; let tmp_361 = keys[11] < tmp_359 || (keys[11] == tmp_359 && values[11] < tmp_360); if tmp_324 == tmp_361 { keys[11] = tmp_359; values[11] = tmp_360; } let tmp_362 = smem_keys[tmp_325 * WPT + 12u]; let tmp_363 = smem_vals[tmp_325 * WPT + 12u]; let tmp_364 = keys[12] < tmp_362 || (keys[12] == tmp_362 && values[12] < tmp_363); if tmp_324 == tmp_364 { keys[12] = tmp_362; values[12] = tmp_363; } let tmp_365 = smem_keys[tmp_325 * WPT + 13u]; let tmp_366 = smem_vals[tmp_325 * WPT + 13u]; let tmp_367 = keys[13] < tmp_365 || (keys[13] == tmp_365 && values[13] < tmp_366); if tmp_324 == tmp_367 { keys[13] = tmp_365; values[13] = tmp_366; } let tmp_368 = smem_keys[tmp_325 * WPT + 14u]; let tmp_369 = smem_vals[tmp_325 * WPT + 14u]; let tmp_370 = keys[14] < tmp_368 || (keys[14] == tmp_368 && values[14] < tmp_369); if tmp_324 == tmp_370 { keys[14] = tmp_368; values[14] = tmp_369; } let tmp_371 = smem_keys[tmp_325 * WPT + 15u]; let tmp_372 = smem_vals[tmp_325 * WPT + 15u]; let tmp_373 = keys[15] < tmp_371 || (keys[15] == tmp_371 && values[15] < tmp_372); if tmp_324 == tmp_373 { keys[15] = tmp_371; values[15] = tmp_372; } workgroupBarrier(); }
    // exch_local(8,16) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_374 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_374;let tmp_375 = values[0]; values[0] = values[8]; values[8] = tmp_375; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_376 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_376;let tmp_377 = values[1]; values[1] = values[9]; values[9] = tmp_377; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_378 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_378;let tmp_379 = values[2]; values[2] = values[10]; values[10] = tmp_379; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_380 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_380;let tmp_381 = values[3]; values[3] = values[11]; values[11] = tmp_381; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_382 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_382;let tmp_383 = values[4]; values[4] = values[12]; values[12] = tmp_383; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_384 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_384;let tmp_385 = values[5]; values[5] = values[13]; values[13] = tmp_385; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_386 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_386;let tmp_387 = values[6]; values[6] = values[14]; values[14] = tmp_387; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_388 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_388;let tmp_389 = values[7]; values[7] = values[15]; values[15] = tmp_389; }
    }
    // exch_local(4,16) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_390 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_390;let tmp_391 = values[0]; values[0] = values[4]; values[4] = tmp_391; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_392 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_392;let tmp_393 = values[1]; values[1] = values[5]; values[5] = tmp_393; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_394 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_394;let tmp_395 = values[2]; values[2] = values[6]; values[6] = tmp_395; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_396 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_396;let tmp_397 = values[3]; values[3] = values[7]; values[7] = tmp_397; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_398 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_398;let tmp_399 = values[8]; values[8] = values[12]; values[12] = tmp_399; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_400 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_400;let tmp_401 = values[9]; values[9] = values[13]; values[13] = tmp_401; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_402 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_402;let tmp_403 = values[10]; values[10] = values[14]; values[14] = tmp_403; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_404 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_404;let tmp_405 = values[11]; values[11] = values[15]; values[15] = tmp_405; }
    }
    // exch_local(2,16) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_406 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_406;let tmp_407 = values[0]; values[0] = values[2]; values[2] = tmp_407; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_408 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_408;let tmp_409 = values[1]; values[1] = values[3]; values[3] = tmp_409; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_410 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_410;let tmp_411 = values[4]; values[4] = values[6]; values[6] = tmp_411; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_412 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_412;let tmp_413 = values[5]; values[5] = values[7]; values[7] = tmp_413; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_414 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_414;let tmp_415 = values[8]; values[8] = values[10]; values[10] = tmp_415; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_416 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_416;let tmp_417 = values[9]; values[9] = values[11]; values[11] = tmp_417; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_418 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_418;let tmp_419 = values[12]; values[12] = values[14]; values[14] = tmp_419; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_420 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_420;let tmp_421 = values[13]; values[13] = values[15]; values[15] = tmp_421; }
    }
    // exch_local(1,16) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_422 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_422;let tmp_423 = values[0]; values[0] = values[1]; values[1] = tmp_423; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_424 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_424;let tmp_425 = values[2]; values[2] = values[3]; values[3] = tmp_425; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_426 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_426;let tmp_427 = values[4]; values[4] = values[5]; values[5] = tmp_427; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_428 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_428;let tmp_429 = values[6]; values[6] = values[7]; values[7] = tmp_429; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_430 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_430;let tmp_431 = values[8]; values[8] = values[9]; values[9] = tmp_431; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_432 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_432;let tmp_433 = values[10]; values[10] = values[11]; values[11] = tmp_433; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_434 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_434;let tmp_435 = values[12]; values[12] = values[13]; values[13] = tmp_435; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_436 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_436;let tmp_437 = values[14]; values[14] = values[15]; values[15] = tmp_437; }
    }
    // exch_intxn(tmask:7,swbit:2,wpt:16)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_438 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_439 = seg_base + (local_tid ^ 7u); let tmp_440 = smem_keys[tmp_439 * WPT + 15u]; let tmp_441 = smem_vals[tmp_439 * WPT + 15u]; let tmp_442 = keys[0] < tmp_440 || (keys[0] == tmp_440 && values[0] < tmp_441); if tmp_438 == tmp_442 { keys[0] = tmp_440; values[0] = tmp_441; } let tmp_443 = smem_keys[tmp_439 * WPT + 14u]; let tmp_444 = smem_vals[tmp_439 * WPT + 14u]; let tmp_445 = keys[1] < tmp_443 || (keys[1] == tmp_443 && values[1] < tmp_444); if tmp_438 == tmp_445 { keys[1] = tmp_443; values[1] = tmp_444; } let tmp_446 = smem_keys[tmp_439 * WPT + 13u]; let tmp_447 = smem_vals[tmp_439 * WPT + 13u]; let tmp_448 = keys[2] < tmp_446 || (keys[2] == tmp_446 && values[2] < tmp_447); if tmp_438 == tmp_448 { keys[2] = tmp_446; values[2] = tmp_447; } let tmp_449 = smem_keys[tmp_439 * WPT + 12u]; let tmp_450 = smem_vals[tmp_439 * WPT + 12u]; let tmp_451 = keys[3] < tmp_449 || (keys[3] == tmp_449 && values[3] < tmp_450); if tmp_438 == tmp_451 { keys[3] = tmp_449; values[3] = tmp_450; } let tmp_452 = smem_keys[tmp_439 * WPT + 11u]; let tmp_453 = smem_vals[tmp_439 * WPT + 11u]; let tmp_454 = keys[4] < tmp_452 || (keys[4] == tmp_452 && values[4] < tmp_453); if tmp_438 == tmp_454 { keys[4] = tmp_452; values[4] = tmp_453; } let tmp_455 = smem_keys[tmp_439 * WPT + 10u]; let tmp_456 = smem_vals[tmp_439 * WPT + 10u]; let tmp_457 = keys[5] < tmp_455 || (keys[5] == tmp_455 && values[5] < tmp_456); if tmp_438 == tmp_457 { keys[5] = tmp_455; values[5] = tmp_456; } let tmp_458 = smem_keys[tmp_439 * WPT + 9u]; let tmp_459 = smem_vals[tmp_439 * WPT + 9u]; let tmp_460 = keys[6] < tmp_458 || (keys[6] == tmp_458 && values[6] < tmp_459); if tmp_438 == tmp_460 { keys[6] = tmp_458; values[6] = tmp_459; } let tmp_461 = smem_keys[tmp_439 * WPT + 8u]; let tmp_462 = smem_vals[tmp_439 * WPT + 8u]; let tmp_463 = keys[7] < tmp_461 || (keys[7] == tmp_461 && values[7] < tmp_462); if tmp_438 == tmp_463 { keys[7] = tmp_461; values[7] = tmp_462; } let tmp_464 = smem_keys[tmp_439 * WPT + 7u]; let tmp_465 = smem_vals[tmp_439 * WPT + 7u]; let tmp_466 = keys[8] < tmp_464 || (keys[8] == tmp_464 && values[8] < tmp_465); if tmp_438 == tmp_466 { keys[8] = tmp_464; values[8] = tmp_465; } let tmp_467 = smem_keys[tmp_439 * WPT + 6u]; let tmp_468 = smem_vals[tmp_439 * WPT + 6u]; let tmp_469 = keys[9] < tmp_467 || (keys[9] == tmp_467 && values[9] < tmp_468); if tmp_438 == tmp_469 { keys[9] = tmp_467; values[9] = tmp_468; } let tmp_470 = smem_keys[tmp_439 * WPT + 5u]; let tmp_471 = smem_vals[tmp_439 * WPT + 5u]; let tmp_472 = keys[10] < tmp_470 || (keys[10] == tmp_470 && values[10] < tmp_471); if tmp_438 == tmp_472 { keys[10] = tmp_470; values[10] = tmp_471; } let tmp_473 = smem_keys[tmp_439 * WPT + 4u]; let tmp_474 = smem_vals[tmp_439 * WPT + 4u]; let tmp_475 = keys[11] < tmp_473 || (keys[11] == tmp_473 && values[11] < tmp_474); if tmp_438 == tmp_475 { keys[11] = tmp_473; values[11] = tmp_474; } let tmp_476 = smem_keys[tmp_439 * WPT + 3u]; let tmp_477 = smem_vals[tmp_439 * WPT + 3u]; let tmp_478 = keys[12] < tmp_476 || (keys[12] == tmp_476 && values[12] < tmp_477); if tmp_438 == tmp_478 { keys[12] = tmp_476; values[12] = tmp_477; } let tmp_479 = smem_keys[tmp_439 * WPT + 2u]; let tmp_480 = smem_vals[tmp_439 * WPT + 2u]; let tmp_481 = keys[13] < tmp_479 || (keys[13] == tmp_479 && values[13] < tmp_480); if tmp_438 == tmp_481 { keys[13] = tmp_479; values[13] = tmp_480; } let tmp_482 = smem_keys[tmp_439 * WPT + 1u]; let tmp_483 = smem_vals[tmp_439 * WPT + 1u]; let tmp_484 = keys[14] < tmp_482 || (keys[14] == tmp_482 && values[14] < tmp_483); if tmp_438 == tmp_484 { keys[14] = tmp_482; values[14] = tmp_483; } let tmp_485 = smem_keys[tmp_439 * WPT + 0u]; let tmp_486 = smem_vals[tmp_439 * WPT + 0u]; let tmp_487 = keys[15] < tmp_485 || (keys[15] == tmp_485 && values[15] < tmp_486); if tmp_438 == tmp_487 { keys[15] = tmp_485; values[15] = tmp_486; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:16) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_488 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_489 = seg_base + (local_tid ^ 2u); let tmp_490 = smem_keys[tmp_489 * WPT + 0u]; let tmp_491 = smem_vals[tmp_489 * WPT + 0u]; let tmp_492 = keys[0] < tmp_490 || (keys[0] == tmp_490 && values[0] < tmp_491); if tmp_488 == tmp_492 { keys[0] = tmp_490; values[0] = tmp_491; } let tmp_493 = smem_keys[tmp_489 * WPT + 1u]; let tmp_494 = smem_vals[tmp_489 * WPT + 1u]; let tmp_495 = keys[1] < tmp_493 || (keys[1] == tmp_493 && values[1] < tmp_494); if tmp_488 == tmp_495 { keys[1] = tmp_493; values[1] = tmp_494; } let tmp_496 = smem_keys[tmp_489 * WPT + 2u]; let tmp_497 = smem_vals[tmp_489 * WPT + 2u]; let tmp_498 = keys[2] < tmp_496 || (keys[2] == tmp_496 && values[2] < tmp_497); if tmp_488 == tmp_498 { keys[2] = tmp_496; values[2] = tmp_497; } let tmp_499 = smem_keys[tmp_489 * WPT + 3u]; let tmp_500 = smem_vals[tmp_489 * WPT + 3u]; let tmp_501 = keys[3] < tmp_499 || (keys[3] == tmp_499 && values[3] < tmp_500); if tmp_488 == tmp_501 { keys[3] = tmp_499; values[3] = tmp_500; } let tmp_502 = smem_keys[tmp_489 * WPT + 4u]; let tmp_503 = smem_vals[tmp_489 * WPT + 4u]; let tmp_504 = keys[4] < tmp_502 || (keys[4] == tmp_502 && values[4] < tmp_503); if tmp_488 == tmp_504 { keys[4] = tmp_502; values[4] = tmp_503; } let tmp_505 = smem_keys[tmp_489 * WPT + 5u]; let tmp_506 = smem_vals[tmp_489 * WPT + 5u]; let tmp_507 = keys[5] < tmp_505 || (keys[5] == tmp_505 && values[5] < tmp_506); if tmp_488 == tmp_507 { keys[5] = tmp_505; values[5] = tmp_506; } let tmp_508 = smem_keys[tmp_489 * WPT + 6u]; let tmp_509 = smem_vals[tmp_489 * WPT + 6u]; let tmp_510 = keys[6] < tmp_508 || (keys[6] == tmp_508 && values[6] < tmp_509); if tmp_488 == tmp_510 { keys[6] = tmp_508; values[6] = tmp_509; } let tmp_511 = smem_keys[tmp_489 * WPT + 7u]; let tmp_512 = smem_vals[tmp_489 * WPT + 7u]; let tmp_513 = keys[7] < tmp_511 || (keys[7] == tmp_511 && values[7] < tmp_512); if tmp_488 == tmp_513 { keys[7] = tmp_511; values[7] = tmp_512; } let tmp_514 = smem_keys[tmp_489 * WPT + 8u]; let tmp_515 = smem_vals[tmp_489 * WPT + 8u]; let tmp_516 = keys[8] < tmp_514 || (keys[8] == tmp_514 && values[8] < tmp_515); if tmp_488 == tmp_516 { keys[8] = tmp_514; values[8] = tmp_515; } let tmp_517 = smem_keys[tmp_489 * WPT + 9u]; let tmp_518 = smem_vals[tmp_489 * WPT + 9u]; let tmp_519 = keys[9] < tmp_517 || (keys[9] == tmp_517 && values[9] < tmp_518); if tmp_488 == tmp_519 { keys[9] = tmp_517; values[9] = tmp_518; } let tmp_520 = smem_keys[tmp_489 * WPT + 10u]; let tmp_521 = smem_vals[tmp_489 * WPT + 10u]; let tmp_522 = keys[10] < tmp_520 || (keys[10] == tmp_520 && values[10] < tmp_521); if tmp_488 == tmp_522 { keys[10] = tmp_520; values[10] = tmp_521; } let tmp_523 = smem_keys[tmp_489 * WPT + 11u]; let tmp_524 = smem_vals[tmp_489 * WPT + 11u]; let tmp_525 = keys[11] < tmp_523 || (keys[11] == tmp_523 && values[11] < tmp_524); if tmp_488 == tmp_525 { keys[11] = tmp_523; values[11] = tmp_524; } let tmp_526 = smem_keys[tmp_489 * WPT + 12u]; let tmp_527 = smem_vals[tmp_489 * WPT + 12u]; let tmp_528 = keys[12] < tmp_526 || (keys[12] == tmp_526 && values[12] < tmp_527); if tmp_488 == tmp_528 { keys[12] = tmp_526; values[12] = tmp_527; } let tmp_529 = smem_keys[tmp_489 * WPT + 13u]; let tmp_530 = smem_vals[tmp_489 * WPT + 13u]; let tmp_531 = keys[13] < tmp_529 || (keys[13] == tmp_529 && values[13] < tmp_530); if tmp_488 == tmp_531 { keys[13] = tmp_529; values[13] = tmp_530; } let tmp_532 = smem_keys[tmp_489 * WPT + 14u]; let tmp_533 = smem_vals[tmp_489 * WPT + 14u]; let tmp_534 = keys[14] < tmp_532 || (keys[14] == tmp_532 && values[14] < tmp_533); if tmp_488 == tmp_534 { keys[14] = tmp_532; values[14] = tmp_533; } let tmp_535 = smem_keys[tmp_489 * WPT + 15u]; let tmp_536 = smem_vals[tmp_489 * WPT + 15u]; let tmp_537 = keys[15] < tmp_535 || (keys[15] == tmp_535 && values[15] < tmp_536); if tmp_488 == tmp_537 { keys[15] = tmp_535; values[15] = tmp_536; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:16) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_538 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_539 = seg_base + (local_tid ^ 1u); let tmp_540 = smem_keys[tmp_539 * WPT + 0u]; let tmp_541 = smem_vals[tmp_539 * WPT + 0u]; let tmp_542 = keys[0] < tmp_540 || (keys[0] == tmp_540 && values[0] < tmp_541); if tmp_538 == tmp_542 { keys[0] = tmp_540; values[0] = tmp_541; } let tmp_543 = smem_keys[tmp_539 * WPT + 1u]; let tmp_544 = smem_vals[tmp_539 * WPT + 1u]; let tmp_545 = keys[1] < tmp_543 || (keys[1] == tmp_543 && values[1] < tmp_544); if tmp_538 == tmp_545 { keys[1] = tmp_543; values[1] = tmp_544; } let tmp_546 = smem_keys[tmp_539 * WPT + 2u]; let tmp_547 = smem_vals[tmp_539 * WPT + 2u]; let tmp_548 = keys[2] < tmp_546 || (keys[2] == tmp_546 && values[2] < tmp_547); if tmp_538 == tmp_548 { keys[2] = tmp_546; values[2] = tmp_547; } let tmp_549 = smem_keys[tmp_539 * WPT + 3u]; let tmp_550 = smem_vals[tmp_539 * WPT + 3u]; let tmp_551 = keys[3] < tmp_549 || (keys[3] == tmp_549 && values[3] < tmp_550); if tmp_538 == tmp_551 { keys[3] = tmp_549; values[3] = tmp_550; } let tmp_552 = smem_keys[tmp_539 * WPT + 4u]; let tmp_553 = smem_vals[tmp_539 * WPT + 4u]; let tmp_554 = keys[4] < tmp_552 || (keys[4] == tmp_552 && values[4] < tmp_553); if tmp_538 == tmp_554 { keys[4] = tmp_552; values[4] = tmp_553; } let tmp_555 = smem_keys[tmp_539 * WPT + 5u]; let tmp_556 = smem_vals[tmp_539 * WPT + 5u]; let tmp_557 = keys[5] < tmp_555 || (keys[5] == tmp_555 && values[5] < tmp_556); if tmp_538 == tmp_557 { keys[5] = tmp_555; values[5] = tmp_556; } let tmp_558 = smem_keys[tmp_539 * WPT + 6u]; let tmp_559 = smem_vals[tmp_539 * WPT + 6u]; let tmp_560 = keys[6] < tmp_558 || (keys[6] == tmp_558 && values[6] < tmp_559); if tmp_538 == tmp_560 { keys[6] = tmp_558; values[6] = tmp_559; } let tmp_561 = smem_keys[tmp_539 * WPT + 7u]; let tmp_562 = smem_vals[tmp_539 * WPT + 7u]; let tmp_563 = keys[7] < tmp_561 || (keys[7] == tmp_561 && values[7] < tmp_562); if tmp_538 == tmp_563 { keys[7] = tmp_561; values[7] = tmp_562; } let tmp_564 = smem_keys[tmp_539 * WPT + 8u]; let tmp_565 = smem_vals[tmp_539 * WPT + 8u]; let tmp_566 = keys[8] < tmp_564 || (keys[8] == tmp_564 && values[8] < tmp_565); if tmp_538 == tmp_566 { keys[8] = tmp_564; values[8] = tmp_565; } let tmp_567 = smem_keys[tmp_539 * WPT + 9u]; let tmp_568 = smem_vals[tmp_539 * WPT + 9u]; let tmp_569 = keys[9] < tmp_567 || (keys[9] == tmp_567 && values[9] < tmp_568); if tmp_538 == tmp_569 { keys[9] = tmp_567; values[9] = tmp_568; } let tmp_570 = smem_keys[tmp_539 * WPT + 10u]; let tmp_571 = smem_vals[tmp_539 * WPT + 10u]; let tmp_572 = keys[10] < tmp_570 || (keys[10] == tmp_570 && values[10] < tmp_571); if tmp_538 == tmp_572 { keys[10] = tmp_570; values[10] = tmp_571; } let tmp_573 = smem_keys[tmp_539 * WPT + 11u]; let tmp_574 = smem_vals[tmp_539 * WPT + 11u]; let tmp_575 = keys[11] < tmp_573 || (keys[11] == tmp_573 && values[11] < tmp_574); if tmp_538 == tmp_575 { keys[11] = tmp_573; values[11] = tmp_574; } let tmp_576 = smem_keys[tmp_539 * WPT + 12u]; let tmp_577 = smem_vals[tmp_539 * WPT + 12u]; let tmp_578 = keys[12] < tmp_576 || (keys[12] == tmp_576 && values[12] < tmp_577); if tmp_538 == tmp_578 { keys[12] = tmp_576; values[12] = tmp_577; } let tmp_579 = smem_keys[tmp_539 * WPT + 13u]; let tmp_580 = smem_vals[tmp_539 * WPT + 13u]; let tmp_581 = keys[13] < tmp_579 || (keys[13] == tmp_579 && values[13] < tmp_580); if tmp_538 == tmp_581 { keys[13] = tmp_579; values[13] = tmp_580; } let tmp_582 = smem_keys[tmp_539 * WPT + 14u]; let tmp_583 = smem_vals[tmp_539 * WPT + 14u]; let tmp_584 = keys[14] < tmp_582 || (keys[14] == tmp_582 && values[14] < tmp_583); if tmp_538 == tmp_584 { keys[14] = tmp_582; values[14] = tmp_583; } let tmp_585 = smem_keys[tmp_539 * WPT + 15u]; let tmp_586 = smem_vals[tmp_539 * WPT + 15u]; let tmp_587 = keys[15] < tmp_585 || (keys[15] == tmp_585 && values[15] < tmp_586); if tmp_538 == tmp_587 { keys[15] = tmp_585; values[15] = tmp_586; } workgroupBarrier(); }
    // exch_local(8,16) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_588 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_588;let tmp_589 = values[0]; values[0] = values[8]; values[8] = tmp_589; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_590 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_590;let tmp_591 = values[1]; values[1] = values[9]; values[9] = tmp_591; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_592 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_592;let tmp_593 = values[2]; values[2] = values[10]; values[10] = tmp_593; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_594 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_594;let tmp_595 = values[3]; values[3] = values[11]; values[11] = tmp_595; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_596 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_596;let tmp_597 = values[4]; values[4] = values[12]; values[12] = tmp_597; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_598 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_598;let tmp_599 = values[5]; values[5] = values[13]; values[13] = tmp_599; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_600 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_600;let tmp_601 = values[6]; values[6] = values[14]; values[14] = tmp_601; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_602 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_602;let tmp_603 = values[7]; values[7] = values[15]; values[15] = tmp_603; }
    }
    // exch_local(4,16) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_604 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_604;let tmp_605 = values[0]; values[0] = values[4]; values[4] = tmp_605; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_606 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_606;let tmp_607 = values[1]; values[1] = values[5]; values[5] = tmp_607; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_608 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_608;let tmp_609 = values[2]; values[2] = values[6]; values[6] = tmp_609; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_610 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_610;let tmp_611 = values[3]; values[3] = values[7]; values[7] = tmp_611; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_612 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_612;let tmp_613 = values[8]; values[8] = values[12]; values[12] = tmp_613; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_614 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_614;let tmp_615 = values[9]; values[9] = values[13]; values[13] = tmp_615; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_616 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_616;let tmp_617 = values[10]; values[10] = values[14]; values[14] = tmp_617; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_618 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_618;let tmp_619 = values[11]; values[11] = values[15]; values[15] = tmp_619; }
    }
    // exch_local(2,16) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_620 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_620;let tmp_621 = values[0]; values[0] = values[2]; values[2] = tmp_621; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_622 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_622;let tmp_623 = values[1]; values[1] = values[3]; values[3] = tmp_623; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_624 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_624;let tmp_625 = values[4]; values[4] = values[6]; values[6] = tmp_625; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_626 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_626;let tmp_627 = values[5]; values[5] = values[7]; values[7] = tmp_627; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_628 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_628;let tmp_629 = values[8]; values[8] = values[10]; values[10] = tmp_629; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_630 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_630;let tmp_631 = values[9]; values[9] = values[11]; values[11] = tmp_631; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_632 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_632;let tmp_633 = values[12]; values[12] = values[14]; values[14] = tmp_633; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_634 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_634;let tmp_635 = values[13]; values[13] = values[15]; values[15] = tmp_635; }
    }
    // exch_local(1,16) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_636 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_636;let tmp_637 = values[0]; values[0] = values[1]; values[1] = tmp_637; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_638 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_638;let tmp_639 = values[2]; values[2] = values[3]; values[3] = tmp_639; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_640 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_640;let tmp_641 = values[4]; values[4] = values[5]; values[5] = tmp_641; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_642 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_642;let tmp_643 = values[6]; values[6] = values[7]; values[7] = tmp_643; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_644 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_644;let tmp_645 = values[8]; values[8] = values[9]; values[9] = tmp_645; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_646 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_646;let tmp_647 = values[10]; values[10] = values[11]; values[11] = tmp_647; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_648 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_648;let tmp_649 = values[12]; values[12] = values[13]; values[13] = tmp_649; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_650 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_650;let tmp_651 = values[14]; values[14] = values[15]; values[15] = tmp_651; }
    }
    // exch_intxn(tmask:15,swbit:3,wpt:16)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_652 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_653 = seg_base + (local_tid ^ 15u); let tmp_654 = smem_keys[tmp_653 * WPT + 15u]; let tmp_655 = smem_vals[tmp_653 * WPT + 15u]; let tmp_656 = keys[0] < tmp_654 || (keys[0] == tmp_654 && values[0] < tmp_655); if tmp_652 == tmp_656 { keys[0] = tmp_654; values[0] = tmp_655; } let tmp_657 = smem_keys[tmp_653 * WPT + 14u]; let tmp_658 = smem_vals[tmp_653 * WPT + 14u]; let tmp_659 = keys[1] < tmp_657 || (keys[1] == tmp_657 && values[1] < tmp_658); if tmp_652 == tmp_659 { keys[1] = tmp_657; values[1] = tmp_658; } let tmp_660 = smem_keys[tmp_653 * WPT + 13u]; let tmp_661 = smem_vals[tmp_653 * WPT + 13u]; let tmp_662 = keys[2] < tmp_660 || (keys[2] == tmp_660 && values[2] < tmp_661); if tmp_652 == tmp_662 { keys[2] = tmp_660; values[2] = tmp_661; } let tmp_663 = smem_keys[tmp_653 * WPT + 12u]; let tmp_664 = smem_vals[tmp_653 * WPT + 12u]; let tmp_665 = keys[3] < tmp_663 || (keys[3] == tmp_663 && values[3] < tmp_664); if tmp_652 == tmp_665 { keys[3] = tmp_663; values[3] = tmp_664; } let tmp_666 = smem_keys[tmp_653 * WPT + 11u]; let tmp_667 = smem_vals[tmp_653 * WPT + 11u]; let tmp_668 = keys[4] < tmp_666 || (keys[4] == tmp_666 && values[4] < tmp_667); if tmp_652 == tmp_668 { keys[4] = tmp_666; values[4] = tmp_667; } let tmp_669 = smem_keys[tmp_653 * WPT + 10u]; let tmp_670 = smem_vals[tmp_653 * WPT + 10u]; let tmp_671 = keys[5] < tmp_669 || (keys[5] == tmp_669 && values[5] < tmp_670); if tmp_652 == tmp_671 { keys[5] = tmp_669; values[5] = tmp_670; } let tmp_672 = smem_keys[tmp_653 * WPT + 9u]; let tmp_673 = smem_vals[tmp_653 * WPT + 9u]; let tmp_674 = keys[6] < tmp_672 || (keys[6] == tmp_672 && values[6] < tmp_673); if tmp_652 == tmp_674 { keys[6] = tmp_672; values[6] = tmp_673; } let tmp_675 = smem_keys[tmp_653 * WPT + 8u]; let tmp_676 = smem_vals[tmp_653 * WPT + 8u]; let tmp_677 = keys[7] < tmp_675 || (keys[7] == tmp_675 && values[7] < tmp_676); if tmp_652 == tmp_677 { keys[7] = tmp_675; values[7] = tmp_676; } let tmp_678 = smem_keys[tmp_653 * WPT + 7u]; let tmp_679 = smem_vals[tmp_653 * WPT + 7u]; let tmp_680 = keys[8] < tmp_678 || (keys[8] == tmp_678 && values[8] < tmp_679); if tmp_652 == tmp_680 { keys[8] = tmp_678; values[8] = tmp_679; } let tmp_681 = smem_keys[tmp_653 * WPT + 6u]; let tmp_682 = smem_vals[tmp_653 * WPT + 6u]; let tmp_683 = keys[9] < tmp_681 || (keys[9] == tmp_681 && values[9] < tmp_682); if tmp_652 == tmp_683 { keys[9] = tmp_681; values[9] = tmp_682; } let tmp_684 = smem_keys[tmp_653 * WPT + 5u]; let tmp_685 = smem_vals[tmp_653 * WPT + 5u]; let tmp_686 = keys[10] < tmp_684 || (keys[10] == tmp_684 && values[10] < tmp_685); if tmp_652 == tmp_686 { keys[10] = tmp_684; values[10] = tmp_685; } let tmp_687 = smem_keys[tmp_653 * WPT + 4u]; let tmp_688 = smem_vals[tmp_653 * WPT + 4u]; let tmp_689 = keys[11] < tmp_687 || (keys[11] == tmp_687 && values[11] < tmp_688); if tmp_652 == tmp_689 { keys[11] = tmp_687; values[11] = tmp_688; } let tmp_690 = smem_keys[tmp_653 * WPT + 3u]; let tmp_691 = smem_vals[tmp_653 * WPT + 3u]; let tmp_692 = keys[12] < tmp_690 || (keys[12] == tmp_690 && values[12] < tmp_691); if tmp_652 == tmp_692 { keys[12] = tmp_690; values[12] = tmp_691; } let tmp_693 = smem_keys[tmp_653 * WPT + 2u]; let tmp_694 = smem_vals[tmp_653 * WPT + 2u]; let tmp_695 = keys[13] < tmp_693 || (keys[13] == tmp_693 && values[13] < tmp_694); if tmp_652 == tmp_695 { keys[13] = tmp_693; values[13] = tmp_694; } let tmp_696 = smem_keys[tmp_653 * WPT + 1u]; let tmp_697 = smem_vals[tmp_653 * WPT + 1u]; let tmp_698 = keys[14] < tmp_696 || (keys[14] == tmp_696 && values[14] < tmp_697); if tmp_652 == tmp_698 { keys[14] = tmp_696; values[14] = tmp_697; } let tmp_699 = smem_keys[tmp_653 * WPT + 0u]; let tmp_700 = smem_vals[tmp_653 * WPT + 0u]; let tmp_701 = keys[15] < tmp_699 || (keys[15] == tmp_699 && values[15] < tmp_700); if tmp_652 == tmp_701 { keys[15] = tmp_699; values[15] = tmp_700; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:16) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_702 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_703 = seg_base + (local_tid ^ 4u); let tmp_704 = smem_keys[tmp_703 * WPT + 0u]; let tmp_705 = smem_vals[tmp_703 * WPT + 0u]; let tmp_706 = keys[0] < tmp_704 || (keys[0] == tmp_704 && values[0] < tmp_705); if tmp_702 == tmp_706 { keys[0] = tmp_704; values[0] = tmp_705; } let tmp_707 = smem_keys[tmp_703 * WPT + 1u]; let tmp_708 = smem_vals[tmp_703 * WPT + 1u]; let tmp_709 = keys[1] < tmp_707 || (keys[1] == tmp_707 && values[1] < tmp_708); if tmp_702 == tmp_709 { keys[1] = tmp_707; values[1] = tmp_708; } let tmp_710 = smem_keys[tmp_703 * WPT + 2u]; let tmp_711 = smem_vals[tmp_703 * WPT + 2u]; let tmp_712 = keys[2] < tmp_710 || (keys[2] == tmp_710 && values[2] < tmp_711); if tmp_702 == tmp_712 { keys[2] = tmp_710; values[2] = tmp_711; } let tmp_713 = smem_keys[tmp_703 * WPT + 3u]; let tmp_714 = smem_vals[tmp_703 * WPT + 3u]; let tmp_715 = keys[3] < tmp_713 || (keys[3] == tmp_713 && values[3] < tmp_714); if tmp_702 == tmp_715 { keys[3] = tmp_713; values[3] = tmp_714; } let tmp_716 = smem_keys[tmp_703 * WPT + 4u]; let tmp_717 = smem_vals[tmp_703 * WPT + 4u]; let tmp_718 = keys[4] < tmp_716 || (keys[4] == tmp_716 && values[4] < tmp_717); if tmp_702 == tmp_718 { keys[4] = tmp_716; values[4] = tmp_717; } let tmp_719 = smem_keys[tmp_703 * WPT + 5u]; let tmp_720 = smem_vals[tmp_703 * WPT + 5u]; let tmp_721 = keys[5] < tmp_719 || (keys[5] == tmp_719 && values[5] < tmp_720); if tmp_702 == tmp_721 { keys[5] = tmp_719; values[5] = tmp_720; } let tmp_722 = smem_keys[tmp_703 * WPT + 6u]; let tmp_723 = smem_vals[tmp_703 * WPT + 6u]; let tmp_724 = keys[6] < tmp_722 || (keys[6] == tmp_722 && values[6] < tmp_723); if tmp_702 == tmp_724 { keys[6] = tmp_722; values[6] = tmp_723; } let tmp_725 = smem_keys[tmp_703 * WPT + 7u]; let tmp_726 = smem_vals[tmp_703 * WPT + 7u]; let tmp_727 = keys[7] < tmp_725 || (keys[7] == tmp_725 && values[7] < tmp_726); if tmp_702 == tmp_727 { keys[7] = tmp_725; values[7] = tmp_726; } let tmp_728 = smem_keys[tmp_703 * WPT + 8u]; let tmp_729 = smem_vals[tmp_703 * WPT + 8u]; let tmp_730 = keys[8] < tmp_728 || (keys[8] == tmp_728 && values[8] < tmp_729); if tmp_702 == tmp_730 { keys[8] = tmp_728; values[8] = tmp_729; } let tmp_731 = smem_keys[tmp_703 * WPT + 9u]; let tmp_732 = smem_vals[tmp_703 * WPT + 9u]; let tmp_733 = keys[9] < tmp_731 || (keys[9] == tmp_731 && values[9] < tmp_732); if tmp_702 == tmp_733 { keys[9] = tmp_731; values[9] = tmp_732; } let tmp_734 = smem_keys[tmp_703 * WPT + 10u]; let tmp_735 = smem_vals[tmp_703 * WPT + 10u]; let tmp_736 = keys[10] < tmp_734 || (keys[10] == tmp_734 && values[10] < tmp_735); if tmp_702 == tmp_736 { keys[10] = tmp_734; values[10] = tmp_735; } let tmp_737 = smem_keys[tmp_703 * WPT + 11u]; let tmp_738 = smem_vals[tmp_703 * WPT + 11u]; let tmp_739 = keys[11] < tmp_737 || (keys[11] == tmp_737 && values[11] < tmp_738); if tmp_702 == tmp_739 { keys[11] = tmp_737; values[11] = tmp_738; } let tmp_740 = smem_keys[tmp_703 * WPT + 12u]; let tmp_741 = smem_vals[tmp_703 * WPT + 12u]; let tmp_742 = keys[12] < tmp_740 || (keys[12] == tmp_740 && values[12] < tmp_741); if tmp_702 == tmp_742 { keys[12] = tmp_740; values[12] = tmp_741; } let tmp_743 = smem_keys[tmp_703 * WPT + 13u]; let tmp_744 = smem_vals[tmp_703 * WPT + 13u]; let tmp_745 = keys[13] < tmp_743 || (keys[13] == tmp_743 && values[13] < tmp_744); if tmp_702 == tmp_745 { keys[13] = tmp_743; values[13] = tmp_744; } let tmp_746 = smem_keys[tmp_703 * WPT + 14u]; let tmp_747 = smem_vals[tmp_703 * WPT + 14u]; let tmp_748 = keys[14] < tmp_746 || (keys[14] == tmp_746 && values[14] < tmp_747); if tmp_702 == tmp_748 { keys[14] = tmp_746; values[14] = tmp_747; } let tmp_749 = smem_keys[tmp_703 * WPT + 15u]; let tmp_750 = smem_vals[tmp_703 * WPT + 15u]; let tmp_751 = keys[15] < tmp_749 || (keys[15] == tmp_749 && values[15] < tmp_750); if tmp_702 == tmp_751 { keys[15] = tmp_749; values[15] = tmp_750; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:16) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_752 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_753 = seg_base + (local_tid ^ 2u); let tmp_754 = smem_keys[tmp_753 * WPT + 0u]; let tmp_755 = smem_vals[tmp_753 * WPT + 0u]; let tmp_756 = keys[0] < tmp_754 || (keys[0] == tmp_754 && values[0] < tmp_755); if tmp_752 == tmp_756 { keys[0] = tmp_754; values[0] = tmp_755; } let tmp_757 = smem_keys[tmp_753 * WPT + 1u]; let tmp_758 = smem_vals[tmp_753 * WPT + 1u]; let tmp_759 = keys[1] < tmp_757 || (keys[1] == tmp_757 && values[1] < tmp_758); if tmp_752 == tmp_759 { keys[1] = tmp_757; values[1] = tmp_758; } let tmp_760 = smem_keys[tmp_753 * WPT + 2u]; let tmp_761 = smem_vals[tmp_753 * WPT + 2u]; let tmp_762 = keys[2] < tmp_760 || (keys[2] == tmp_760 && values[2] < tmp_761); if tmp_752 == tmp_762 { keys[2] = tmp_760; values[2] = tmp_761; } let tmp_763 = smem_keys[tmp_753 * WPT + 3u]; let tmp_764 = smem_vals[tmp_753 * WPT + 3u]; let tmp_765 = keys[3] < tmp_763 || (keys[3] == tmp_763 && values[3] < tmp_764); if tmp_752 == tmp_765 { keys[3] = tmp_763; values[3] = tmp_764; } let tmp_766 = smem_keys[tmp_753 * WPT + 4u]; let tmp_767 = smem_vals[tmp_753 * WPT + 4u]; let tmp_768 = keys[4] < tmp_766 || (keys[4] == tmp_766 && values[4] < tmp_767); if tmp_752 == tmp_768 { keys[4] = tmp_766; values[4] = tmp_767; } let tmp_769 = smem_keys[tmp_753 * WPT + 5u]; let tmp_770 = smem_vals[tmp_753 * WPT + 5u]; let tmp_771 = keys[5] < tmp_769 || (keys[5] == tmp_769 && values[5] < tmp_770); if tmp_752 == tmp_771 { keys[5] = tmp_769; values[5] = tmp_770; } let tmp_772 = smem_keys[tmp_753 * WPT + 6u]; let tmp_773 = smem_vals[tmp_753 * WPT + 6u]; let tmp_774 = keys[6] < tmp_772 || (keys[6] == tmp_772 && values[6] < tmp_773); if tmp_752 == tmp_774 { keys[6] = tmp_772; values[6] = tmp_773; } let tmp_775 = smem_keys[tmp_753 * WPT + 7u]; let tmp_776 = smem_vals[tmp_753 * WPT + 7u]; let tmp_777 = keys[7] < tmp_775 || (keys[7] == tmp_775 && values[7] < tmp_776); if tmp_752 == tmp_777 { keys[7] = tmp_775; values[7] = tmp_776; } let tmp_778 = smem_keys[tmp_753 * WPT + 8u]; let tmp_779 = smem_vals[tmp_753 * WPT + 8u]; let tmp_780 = keys[8] < tmp_778 || (keys[8] == tmp_778 && values[8] < tmp_779); if tmp_752 == tmp_780 { keys[8] = tmp_778; values[8] = tmp_779; } let tmp_781 = smem_keys[tmp_753 * WPT + 9u]; let tmp_782 = smem_vals[tmp_753 * WPT + 9u]; let tmp_783 = keys[9] < tmp_781 || (keys[9] == tmp_781 && values[9] < tmp_782); if tmp_752 == tmp_783 { keys[9] = tmp_781; values[9] = tmp_782; } let tmp_784 = smem_keys[tmp_753 * WPT + 10u]; let tmp_785 = smem_vals[tmp_753 * WPT + 10u]; let tmp_786 = keys[10] < tmp_784 || (keys[10] == tmp_784 && values[10] < tmp_785); if tmp_752 == tmp_786 { keys[10] = tmp_784; values[10] = tmp_785; } let tmp_787 = smem_keys[tmp_753 * WPT + 11u]; let tmp_788 = smem_vals[tmp_753 * WPT + 11u]; let tmp_789 = keys[11] < tmp_787 || (keys[11] == tmp_787 && values[11] < tmp_788); if tmp_752 == tmp_789 { keys[11] = tmp_787; values[11] = tmp_788; } let tmp_790 = smem_keys[tmp_753 * WPT + 12u]; let tmp_791 = smem_vals[tmp_753 * WPT + 12u]; let tmp_792 = keys[12] < tmp_790 || (keys[12] == tmp_790 && values[12] < tmp_791); if tmp_752 == tmp_792 { keys[12] = tmp_790; values[12] = tmp_791; } let tmp_793 = smem_keys[tmp_753 * WPT + 13u]; let tmp_794 = smem_vals[tmp_753 * WPT + 13u]; let tmp_795 = keys[13] < tmp_793 || (keys[13] == tmp_793 && values[13] < tmp_794); if tmp_752 == tmp_795 { keys[13] = tmp_793; values[13] = tmp_794; } let tmp_796 = smem_keys[tmp_753 * WPT + 14u]; let tmp_797 = smem_vals[tmp_753 * WPT + 14u]; let tmp_798 = keys[14] < tmp_796 || (keys[14] == tmp_796 && values[14] < tmp_797); if tmp_752 == tmp_798 { keys[14] = tmp_796; values[14] = tmp_797; } let tmp_799 = smem_keys[tmp_753 * WPT + 15u]; let tmp_800 = smem_vals[tmp_753 * WPT + 15u]; let tmp_801 = keys[15] < tmp_799 || (keys[15] == tmp_799 && values[15] < tmp_800); if tmp_752 == tmp_801 { keys[15] = tmp_799; values[15] = tmp_800; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:16) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_802 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_803 = seg_base + (local_tid ^ 1u); let tmp_804 = smem_keys[tmp_803 * WPT + 0u]; let tmp_805 = smem_vals[tmp_803 * WPT + 0u]; let tmp_806 = keys[0] < tmp_804 || (keys[0] == tmp_804 && values[0] < tmp_805); if tmp_802 == tmp_806 { keys[0] = tmp_804; values[0] = tmp_805; } let tmp_807 = smem_keys[tmp_803 * WPT + 1u]; let tmp_808 = smem_vals[tmp_803 * WPT + 1u]; let tmp_809 = keys[1] < tmp_807 || (keys[1] == tmp_807 && values[1] < tmp_808); if tmp_802 == tmp_809 { keys[1] = tmp_807; values[1] = tmp_808; } let tmp_810 = smem_keys[tmp_803 * WPT + 2u]; let tmp_811 = smem_vals[tmp_803 * WPT + 2u]; let tmp_812 = keys[2] < tmp_810 || (keys[2] == tmp_810 && values[2] < tmp_811); if tmp_802 == tmp_812 { keys[2] = tmp_810; values[2] = tmp_811; } let tmp_813 = smem_keys[tmp_803 * WPT + 3u]; let tmp_814 = smem_vals[tmp_803 * WPT + 3u]; let tmp_815 = keys[3] < tmp_813 || (keys[3] == tmp_813 && values[3] < tmp_814); if tmp_802 == tmp_815 { keys[3] = tmp_813; values[3] = tmp_814; } let tmp_816 = smem_keys[tmp_803 * WPT + 4u]; let tmp_817 = smem_vals[tmp_803 * WPT + 4u]; let tmp_818 = keys[4] < tmp_816 || (keys[4] == tmp_816 && values[4] < tmp_817); if tmp_802 == tmp_818 { keys[4] = tmp_816; values[4] = tmp_817; } let tmp_819 = smem_keys[tmp_803 * WPT + 5u]; let tmp_820 = smem_vals[tmp_803 * WPT + 5u]; let tmp_821 = keys[5] < tmp_819 || (keys[5] == tmp_819 && values[5] < tmp_820); if tmp_802 == tmp_821 { keys[5] = tmp_819; values[5] = tmp_820; } let tmp_822 = smem_keys[tmp_803 * WPT + 6u]; let tmp_823 = smem_vals[tmp_803 * WPT + 6u]; let tmp_824 = keys[6] < tmp_822 || (keys[6] == tmp_822 && values[6] < tmp_823); if tmp_802 == tmp_824 { keys[6] = tmp_822; values[6] = tmp_823; } let tmp_825 = smem_keys[tmp_803 * WPT + 7u]; let tmp_826 = smem_vals[tmp_803 * WPT + 7u]; let tmp_827 = keys[7] < tmp_825 || (keys[7] == tmp_825 && values[7] < tmp_826); if tmp_802 == tmp_827 { keys[7] = tmp_825; values[7] = tmp_826; } let tmp_828 = smem_keys[tmp_803 * WPT + 8u]; let tmp_829 = smem_vals[tmp_803 * WPT + 8u]; let tmp_830 = keys[8] < tmp_828 || (keys[8] == tmp_828 && values[8] < tmp_829); if tmp_802 == tmp_830 { keys[8] = tmp_828; values[8] = tmp_829; } let tmp_831 = smem_keys[tmp_803 * WPT + 9u]; let tmp_832 = smem_vals[tmp_803 * WPT + 9u]; let tmp_833 = keys[9] < tmp_831 || (keys[9] == tmp_831 && values[9] < tmp_832); if tmp_802 == tmp_833 { keys[9] = tmp_831; values[9] = tmp_832; } let tmp_834 = smem_keys[tmp_803 * WPT + 10u]; let tmp_835 = smem_vals[tmp_803 * WPT + 10u]; let tmp_836 = keys[10] < tmp_834 || (keys[10] == tmp_834 && values[10] < tmp_835); if tmp_802 == tmp_836 { keys[10] = tmp_834; values[10] = tmp_835; } let tmp_837 = smem_keys[tmp_803 * WPT + 11u]; let tmp_838 = smem_vals[tmp_803 * WPT + 11u]; let tmp_839 = keys[11] < tmp_837 || (keys[11] == tmp_837 && values[11] < tmp_838); if tmp_802 == tmp_839 { keys[11] = tmp_837; values[11] = tmp_838; } let tmp_840 = smem_keys[tmp_803 * WPT + 12u]; let tmp_841 = smem_vals[tmp_803 * WPT + 12u]; let tmp_842 = keys[12] < tmp_840 || (keys[12] == tmp_840 && values[12] < tmp_841); if tmp_802 == tmp_842 { keys[12] = tmp_840; values[12] = tmp_841; } let tmp_843 = smem_keys[tmp_803 * WPT + 13u]; let tmp_844 = smem_vals[tmp_803 * WPT + 13u]; let tmp_845 = keys[13] < tmp_843 || (keys[13] == tmp_843 && values[13] < tmp_844); if tmp_802 == tmp_845 { keys[13] = tmp_843; values[13] = tmp_844; } let tmp_846 = smem_keys[tmp_803 * WPT + 14u]; let tmp_847 = smem_vals[tmp_803 * WPT + 14u]; let tmp_848 = keys[14] < tmp_846 || (keys[14] == tmp_846 && values[14] < tmp_847); if tmp_802 == tmp_848 { keys[14] = tmp_846; values[14] = tmp_847; } let tmp_849 = smem_keys[tmp_803 * WPT + 15u]; let tmp_850 = smem_vals[tmp_803 * WPT + 15u]; let tmp_851 = keys[15] < tmp_849 || (keys[15] == tmp_849 && values[15] < tmp_850); if tmp_802 == tmp_851 { keys[15] = tmp_849; values[15] = tmp_850; } workgroupBarrier(); }
    // exch_local(8,16) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_852 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_852;let tmp_853 = values[0]; values[0] = values[8]; values[8] = tmp_853; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_854 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_854;let tmp_855 = values[1]; values[1] = values[9]; values[9] = tmp_855; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_856 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_856;let tmp_857 = values[2]; values[2] = values[10]; values[10] = tmp_857; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_858 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_858;let tmp_859 = values[3]; values[3] = values[11]; values[11] = tmp_859; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_860 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_860;let tmp_861 = values[4]; values[4] = values[12]; values[12] = tmp_861; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_862 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_862;let tmp_863 = values[5]; values[5] = values[13]; values[13] = tmp_863; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_864 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_864;let tmp_865 = values[6]; values[6] = values[14]; values[14] = tmp_865; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_866 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_866;let tmp_867 = values[7]; values[7] = values[15]; values[15] = tmp_867; }
    }
    // exch_local(4,16) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_868 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_868;let tmp_869 = values[0]; values[0] = values[4]; values[4] = tmp_869; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_870 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_870;let tmp_871 = values[1]; values[1] = values[5]; values[5] = tmp_871; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_872 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_872;let tmp_873 = values[2]; values[2] = values[6]; values[6] = tmp_873; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_874 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_874;let tmp_875 = values[3]; values[3] = values[7]; values[7] = tmp_875; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_876 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_876;let tmp_877 = values[8]; values[8] = values[12]; values[12] = tmp_877; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_878 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_878;let tmp_879 = values[9]; values[9] = values[13]; values[13] = tmp_879; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_880 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_880;let tmp_881 = values[10]; values[10] = values[14]; values[14] = tmp_881; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_882 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_882;let tmp_883 = values[11]; values[11] = values[15]; values[15] = tmp_883; }
    }
    // exch_local(2,16) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_884 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_884;let tmp_885 = values[0]; values[0] = values[2]; values[2] = tmp_885; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_886 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_886;let tmp_887 = values[1]; values[1] = values[3]; values[3] = tmp_887; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_888 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_888;let tmp_889 = values[4]; values[4] = values[6]; values[6] = tmp_889; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_890 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_890;let tmp_891 = values[5]; values[5] = values[7]; values[7] = tmp_891; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_892 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_892;let tmp_893 = values[8]; values[8] = values[10]; values[10] = tmp_893; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_894 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_894;let tmp_895 = values[9]; values[9] = values[11]; values[11] = tmp_895; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_896 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_896;let tmp_897 = values[12]; values[12] = values[14]; values[14] = tmp_897; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_898 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_898;let tmp_899 = values[13]; values[13] = values[15]; values[15] = tmp_899; }
    }
    // exch_local(1,16) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_900 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_900;let tmp_901 = values[0]; values[0] = values[1]; values[1] = tmp_901; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_902 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_902;let tmp_903 = values[2]; values[2] = values[3]; values[3] = tmp_903; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_904 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_904;let tmp_905 = values[4]; values[4] = values[5]; values[5] = tmp_905; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_906 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_906;let tmp_907 = values[6]; values[6] = values[7]; values[7] = tmp_907; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_908 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_908;let tmp_909 = values[8]; values[8] = values[9]; values[9] = tmp_909; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_910 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_910;let tmp_911 = values[10]; values[10] = values[11]; values[11] = tmp_911; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_912 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_912;let tmp_913 = values[12]; values[12] = values[13]; values[13] = tmp_913; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_914 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_914;let tmp_915 = values[14]; values[14] = values[15]; values[15] = tmp_915; }
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
