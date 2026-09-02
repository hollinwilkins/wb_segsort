
override WG: u32 = 1u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 16u;
const M: u32 = 1u;
const WPT: u32 = 16u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n16_m1_striped(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 4u;

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
