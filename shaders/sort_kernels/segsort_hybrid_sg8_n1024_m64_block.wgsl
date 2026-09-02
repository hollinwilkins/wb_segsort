
enable subgroups;

override WG: u32 = 64u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 1024u;
const M: u32 = 64u;
const WPT: u32 = 16u;
const R: u32 = 8u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybrid_sg8_n1024_m64_block(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 10u;

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
    {
    let tmp_160 = subgroupShuffleXor(keys[15], 1u);
    let tmp_161 = subgroupShuffleXor(values[15], 1u);
    let tmp_162 = subgroupShuffleXor(keys[14], 1u);
    let tmp_163 = subgroupShuffleXor(values[14], 1u);
    let tmp_164 = subgroupShuffleXor(keys[13], 1u);
    let tmp_165 = subgroupShuffleXor(values[13], 1u);
    let tmp_166 = subgroupShuffleXor(keys[12], 1u);
    let tmp_167 = subgroupShuffleXor(values[12], 1u);
    let tmp_168 = subgroupShuffleXor(keys[11], 1u);
    let tmp_169 = subgroupShuffleXor(values[11], 1u);
    let tmp_170 = subgroupShuffleXor(keys[10], 1u);
    let tmp_171 = subgroupShuffleXor(values[10], 1u);
    let tmp_172 = subgroupShuffleXor(keys[9], 1u);
    let tmp_173 = subgroupShuffleXor(values[9], 1u);
    let tmp_174 = subgroupShuffleXor(keys[8], 1u);
    let tmp_175 = subgroupShuffleXor(values[8], 1u);
    let tmp_176 = subgroupShuffleXor(keys[7], 1u);
    let tmp_177 = subgroupShuffleXor(values[7], 1u);
    let tmp_178 = subgroupShuffleXor(keys[6], 1u);
    let tmp_179 = subgroupShuffleXor(values[6], 1u);
    let tmp_180 = subgroupShuffleXor(keys[5], 1u);
    let tmp_181 = subgroupShuffleXor(values[5], 1u);
    let tmp_182 = subgroupShuffleXor(keys[4], 1u);
    let tmp_183 = subgroupShuffleXor(values[4], 1u);
    let tmp_184 = subgroupShuffleXor(keys[3], 1u);
    let tmp_185 = subgroupShuffleXor(values[3], 1u);
    let tmp_186 = subgroupShuffleXor(keys[2], 1u);
    let tmp_187 = subgroupShuffleXor(values[2], 1u);
    let tmp_188 = subgroupShuffleXor(keys[1], 1u);
    let tmp_189 = subgroupShuffleXor(values[1], 1u);
    let tmp_190 = subgroupShuffleXor(keys[0], 1u);
    let tmp_191 = subgroupShuffleXor(values[0], 1u);
    let tmp_192 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_193 = keys[0] < tmp_160 || (keys[0] == tmp_160 && values[0] < tmp_161);
    if tmp_192 == tmp_193 { keys[0] = tmp_160; values[0] = tmp_161; }
    let tmp_194 = keys[1] < tmp_162 || (keys[1] == tmp_162 && values[1] < tmp_163);
    if tmp_192 == tmp_194 { keys[1] = tmp_162; values[1] = tmp_163; }
    let tmp_195 = keys[2] < tmp_164 || (keys[2] == tmp_164 && values[2] < tmp_165);
    if tmp_192 == tmp_195 { keys[2] = tmp_164; values[2] = tmp_165; }
    let tmp_196 = keys[3] < tmp_166 || (keys[3] == tmp_166 && values[3] < tmp_167);
    if tmp_192 == tmp_196 { keys[3] = tmp_166; values[3] = tmp_167; }
    let tmp_197 = keys[4] < tmp_168 || (keys[4] == tmp_168 && values[4] < tmp_169);
    if tmp_192 == tmp_197 { keys[4] = tmp_168; values[4] = tmp_169; }
    let tmp_198 = keys[5] < tmp_170 || (keys[5] == tmp_170 && values[5] < tmp_171);
    if tmp_192 == tmp_198 { keys[5] = tmp_170; values[5] = tmp_171; }
    let tmp_199 = keys[6] < tmp_172 || (keys[6] == tmp_172 && values[6] < tmp_173);
    if tmp_192 == tmp_199 { keys[6] = tmp_172; values[6] = tmp_173; }
    let tmp_200 = keys[7] < tmp_174 || (keys[7] == tmp_174 && values[7] < tmp_175);
    if tmp_192 == tmp_200 { keys[7] = tmp_174; values[7] = tmp_175; }
    let tmp_201 = keys[8] < tmp_176 || (keys[8] == tmp_176 && values[8] < tmp_177);
    if tmp_192 == tmp_201 { keys[8] = tmp_176; values[8] = tmp_177; }
    let tmp_202 = keys[9] < tmp_178 || (keys[9] == tmp_178 && values[9] < tmp_179);
    if tmp_192 == tmp_202 { keys[9] = tmp_178; values[9] = tmp_179; }
    let tmp_203 = keys[10] < tmp_180 || (keys[10] == tmp_180 && values[10] < tmp_181);
    if tmp_192 == tmp_203 { keys[10] = tmp_180; values[10] = tmp_181; }
    let tmp_204 = keys[11] < tmp_182 || (keys[11] == tmp_182 && values[11] < tmp_183);
    if tmp_192 == tmp_204 { keys[11] = tmp_182; values[11] = tmp_183; }
    let tmp_205 = keys[12] < tmp_184 || (keys[12] == tmp_184 && values[12] < tmp_185);
    if tmp_192 == tmp_205 { keys[12] = tmp_184; values[12] = tmp_185; }
    let tmp_206 = keys[13] < tmp_186 || (keys[13] == tmp_186 && values[13] < tmp_187);
    if tmp_192 == tmp_206 { keys[13] = tmp_186; values[13] = tmp_187; }
    let tmp_207 = keys[14] < tmp_188 || (keys[14] == tmp_188 && values[14] < tmp_189);
    if tmp_192 == tmp_207 { keys[14] = tmp_188; values[14] = tmp_189; }
    let tmp_208 = keys[15] < tmp_190 || (keys[15] == tmp_190 && values[15] < tmp_191);
    if tmp_192 == tmp_208 { keys[15] = tmp_190; values[15] = tmp_191; }
    }
    // exch_local(8,16) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_209 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_209;let tmp_210 = values[0]; values[0] = values[8]; values[8] = tmp_210; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_211 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_211;let tmp_212 = values[1]; values[1] = values[9]; values[9] = tmp_212; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_213 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_213;let tmp_214 = values[2]; values[2] = values[10]; values[10] = tmp_214; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_215 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_215;let tmp_216 = values[3]; values[3] = values[11]; values[11] = tmp_216; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_217 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_217;let tmp_218 = values[4]; values[4] = values[12]; values[12] = tmp_218; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_219 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_219;let tmp_220 = values[5]; values[5] = values[13]; values[13] = tmp_220; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_221 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_221;let tmp_222 = values[6]; values[6] = values[14]; values[14] = tmp_222; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_223 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_223;let tmp_224 = values[7]; values[7] = values[15]; values[15] = tmp_224; }
    }
    // exch_local(4,16) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_225 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_225;let tmp_226 = values[0]; values[0] = values[4]; values[4] = tmp_226; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_227 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_227;let tmp_228 = values[1]; values[1] = values[5]; values[5] = tmp_228; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_229 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_229;let tmp_230 = values[2]; values[2] = values[6]; values[6] = tmp_230; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_231 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_231;let tmp_232 = values[3]; values[3] = values[7]; values[7] = tmp_232; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_233 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_233;let tmp_234 = values[8]; values[8] = values[12]; values[12] = tmp_234; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_235 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_235;let tmp_236 = values[9]; values[9] = values[13]; values[13] = tmp_236; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_237 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_237;let tmp_238 = values[10]; values[10] = values[14]; values[14] = tmp_238; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_239 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_239;let tmp_240 = values[11]; values[11] = values[15]; values[15] = tmp_240; }
    }
    // exch_local(2,16) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_241 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_241;let tmp_242 = values[0]; values[0] = values[2]; values[2] = tmp_242; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_243 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_243;let tmp_244 = values[1]; values[1] = values[3]; values[3] = tmp_244; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_245 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_245;let tmp_246 = values[4]; values[4] = values[6]; values[6] = tmp_246; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_247 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_247;let tmp_248 = values[5]; values[5] = values[7]; values[7] = tmp_248; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_249 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_249;let tmp_250 = values[8]; values[8] = values[10]; values[10] = tmp_250; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_251 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_251;let tmp_252 = values[9]; values[9] = values[11]; values[11] = tmp_252; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_253 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_253;let tmp_254 = values[12]; values[12] = values[14]; values[14] = tmp_254; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_255 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_255;let tmp_256 = values[13]; values[13] = values[15]; values[15] = tmp_256; }
    }
    // exch_local(1,16) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_257 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_257;let tmp_258 = values[0]; values[0] = values[1]; values[1] = tmp_258; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_259 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_259;let tmp_260 = values[2]; values[2] = values[3]; values[3] = tmp_260; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_261 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_261;let tmp_262 = values[4]; values[4] = values[5]; values[5] = tmp_262; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_263 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_263;let tmp_264 = values[6]; values[6] = values[7]; values[7] = tmp_264; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_265 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_265;let tmp_266 = values[8]; values[8] = values[9]; values[9] = tmp_266; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_267 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_267;let tmp_268 = values[10]; values[10] = values[11]; values[11] = tmp_268; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_269 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_269;let tmp_270 = values[12]; values[12] = values[13]; values[13] = tmp_270; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_271 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_271;let tmp_272 = values[14]; values[14] = values[15]; values[15] = tmp_272; }
    }
    // exch_intxn(tmask:3,swbit:1,wpt:16)
    {
    let tmp_273 = subgroupShuffleXor(keys[15], 3u);
    let tmp_274 = subgroupShuffleXor(values[15], 3u);
    let tmp_275 = subgroupShuffleXor(keys[14], 3u);
    let tmp_276 = subgroupShuffleXor(values[14], 3u);
    let tmp_277 = subgroupShuffleXor(keys[13], 3u);
    let tmp_278 = subgroupShuffleXor(values[13], 3u);
    let tmp_279 = subgroupShuffleXor(keys[12], 3u);
    let tmp_280 = subgroupShuffleXor(values[12], 3u);
    let tmp_281 = subgroupShuffleXor(keys[11], 3u);
    let tmp_282 = subgroupShuffleXor(values[11], 3u);
    let tmp_283 = subgroupShuffleXor(keys[10], 3u);
    let tmp_284 = subgroupShuffleXor(values[10], 3u);
    let tmp_285 = subgroupShuffleXor(keys[9], 3u);
    let tmp_286 = subgroupShuffleXor(values[9], 3u);
    let tmp_287 = subgroupShuffleXor(keys[8], 3u);
    let tmp_288 = subgroupShuffleXor(values[8], 3u);
    let tmp_289 = subgroupShuffleXor(keys[7], 3u);
    let tmp_290 = subgroupShuffleXor(values[7], 3u);
    let tmp_291 = subgroupShuffleXor(keys[6], 3u);
    let tmp_292 = subgroupShuffleXor(values[6], 3u);
    let tmp_293 = subgroupShuffleXor(keys[5], 3u);
    let tmp_294 = subgroupShuffleXor(values[5], 3u);
    let tmp_295 = subgroupShuffleXor(keys[4], 3u);
    let tmp_296 = subgroupShuffleXor(values[4], 3u);
    let tmp_297 = subgroupShuffleXor(keys[3], 3u);
    let tmp_298 = subgroupShuffleXor(values[3], 3u);
    let tmp_299 = subgroupShuffleXor(keys[2], 3u);
    let tmp_300 = subgroupShuffleXor(values[2], 3u);
    let tmp_301 = subgroupShuffleXor(keys[1], 3u);
    let tmp_302 = subgroupShuffleXor(values[1], 3u);
    let tmp_303 = subgroupShuffleXor(keys[0], 3u);
    let tmp_304 = subgroupShuffleXor(values[0], 3u);
    let tmp_305 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_306 = keys[0] < tmp_273 || (keys[0] == tmp_273 && values[0] < tmp_274);
    if tmp_305 == tmp_306 { keys[0] = tmp_273; values[0] = tmp_274; }
    let tmp_307 = keys[1] < tmp_275 || (keys[1] == tmp_275 && values[1] < tmp_276);
    if tmp_305 == tmp_307 { keys[1] = tmp_275; values[1] = tmp_276; }
    let tmp_308 = keys[2] < tmp_277 || (keys[2] == tmp_277 && values[2] < tmp_278);
    if tmp_305 == tmp_308 { keys[2] = tmp_277; values[2] = tmp_278; }
    let tmp_309 = keys[3] < tmp_279 || (keys[3] == tmp_279 && values[3] < tmp_280);
    if tmp_305 == tmp_309 { keys[3] = tmp_279; values[3] = tmp_280; }
    let tmp_310 = keys[4] < tmp_281 || (keys[4] == tmp_281 && values[4] < tmp_282);
    if tmp_305 == tmp_310 { keys[4] = tmp_281; values[4] = tmp_282; }
    let tmp_311 = keys[5] < tmp_283 || (keys[5] == tmp_283 && values[5] < tmp_284);
    if tmp_305 == tmp_311 { keys[5] = tmp_283; values[5] = tmp_284; }
    let tmp_312 = keys[6] < tmp_285 || (keys[6] == tmp_285 && values[6] < tmp_286);
    if tmp_305 == tmp_312 { keys[6] = tmp_285; values[6] = tmp_286; }
    let tmp_313 = keys[7] < tmp_287 || (keys[7] == tmp_287 && values[7] < tmp_288);
    if tmp_305 == tmp_313 { keys[7] = tmp_287; values[7] = tmp_288; }
    let tmp_314 = keys[8] < tmp_289 || (keys[8] == tmp_289 && values[8] < tmp_290);
    if tmp_305 == tmp_314 { keys[8] = tmp_289; values[8] = tmp_290; }
    let tmp_315 = keys[9] < tmp_291 || (keys[9] == tmp_291 && values[9] < tmp_292);
    if tmp_305 == tmp_315 { keys[9] = tmp_291; values[9] = tmp_292; }
    let tmp_316 = keys[10] < tmp_293 || (keys[10] == tmp_293 && values[10] < tmp_294);
    if tmp_305 == tmp_316 { keys[10] = tmp_293; values[10] = tmp_294; }
    let tmp_317 = keys[11] < tmp_295 || (keys[11] == tmp_295 && values[11] < tmp_296);
    if tmp_305 == tmp_317 { keys[11] = tmp_295; values[11] = tmp_296; }
    let tmp_318 = keys[12] < tmp_297 || (keys[12] == tmp_297 && values[12] < tmp_298);
    if tmp_305 == tmp_318 { keys[12] = tmp_297; values[12] = tmp_298; }
    let tmp_319 = keys[13] < tmp_299 || (keys[13] == tmp_299 && values[13] < tmp_300);
    if tmp_305 == tmp_319 { keys[13] = tmp_299; values[13] = tmp_300; }
    let tmp_320 = keys[14] < tmp_301 || (keys[14] == tmp_301 && values[14] < tmp_302);
    if tmp_305 == tmp_320 { keys[14] = tmp_301; values[14] = tmp_302; }
    let tmp_321 = keys[15] < tmp_303 || (keys[15] == tmp_303 && values[15] < tmp_304);
    if tmp_305 == tmp_321 { keys[15] = tmp_303; values[15] = tmp_304; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:16) 
    {
    let tmp_322 = subgroupShuffleXor(keys[0], 1u);
    let tmp_323 = subgroupShuffleXor(values[0], 1u);
    let tmp_324 = subgroupShuffleXor(keys[1], 1u);
    let tmp_325 = subgroupShuffleXor(values[1], 1u);
    let tmp_326 = subgroupShuffleXor(keys[2], 1u);
    let tmp_327 = subgroupShuffleXor(values[2], 1u);
    let tmp_328 = subgroupShuffleXor(keys[3], 1u);
    let tmp_329 = subgroupShuffleXor(values[3], 1u);
    let tmp_330 = subgroupShuffleXor(keys[4], 1u);
    let tmp_331 = subgroupShuffleXor(values[4], 1u);
    let tmp_332 = subgroupShuffleXor(keys[5], 1u);
    let tmp_333 = subgroupShuffleXor(values[5], 1u);
    let tmp_334 = subgroupShuffleXor(keys[6], 1u);
    let tmp_335 = subgroupShuffleXor(values[6], 1u);
    let tmp_336 = subgroupShuffleXor(keys[7], 1u);
    let tmp_337 = subgroupShuffleXor(values[7], 1u);
    let tmp_338 = subgroupShuffleXor(keys[8], 1u);
    let tmp_339 = subgroupShuffleXor(values[8], 1u);
    let tmp_340 = subgroupShuffleXor(keys[9], 1u);
    let tmp_341 = subgroupShuffleXor(values[9], 1u);
    let tmp_342 = subgroupShuffleXor(keys[10], 1u);
    let tmp_343 = subgroupShuffleXor(values[10], 1u);
    let tmp_344 = subgroupShuffleXor(keys[11], 1u);
    let tmp_345 = subgroupShuffleXor(values[11], 1u);
    let tmp_346 = subgroupShuffleXor(keys[12], 1u);
    let tmp_347 = subgroupShuffleXor(values[12], 1u);
    let tmp_348 = subgroupShuffleXor(keys[13], 1u);
    let tmp_349 = subgroupShuffleXor(values[13], 1u);
    let tmp_350 = subgroupShuffleXor(keys[14], 1u);
    let tmp_351 = subgroupShuffleXor(values[14], 1u);
    let tmp_352 = subgroupShuffleXor(keys[15], 1u);
    let tmp_353 = subgroupShuffleXor(values[15], 1u);
    let tmp_354 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_355 = keys[0] < tmp_322 || (keys[0] == tmp_322 && values[0] < tmp_323);
    if tmp_354 == tmp_355 { keys[0] = tmp_322; values[0] = tmp_323; }
    let tmp_356 = keys[1] < tmp_324 || (keys[1] == tmp_324 && values[1] < tmp_325);
    if tmp_354 == tmp_356 { keys[1] = tmp_324; values[1] = tmp_325; }
    let tmp_357 = keys[2] < tmp_326 || (keys[2] == tmp_326 && values[2] < tmp_327);
    if tmp_354 == tmp_357 { keys[2] = tmp_326; values[2] = tmp_327; }
    let tmp_358 = keys[3] < tmp_328 || (keys[3] == tmp_328 && values[3] < tmp_329);
    if tmp_354 == tmp_358 { keys[3] = tmp_328; values[3] = tmp_329; }
    let tmp_359 = keys[4] < tmp_330 || (keys[4] == tmp_330 && values[4] < tmp_331);
    if tmp_354 == tmp_359 { keys[4] = tmp_330; values[4] = tmp_331; }
    let tmp_360 = keys[5] < tmp_332 || (keys[5] == tmp_332 && values[5] < tmp_333);
    if tmp_354 == tmp_360 { keys[5] = tmp_332; values[5] = tmp_333; }
    let tmp_361 = keys[6] < tmp_334 || (keys[6] == tmp_334 && values[6] < tmp_335);
    if tmp_354 == tmp_361 { keys[6] = tmp_334; values[6] = tmp_335; }
    let tmp_362 = keys[7] < tmp_336 || (keys[7] == tmp_336 && values[7] < tmp_337);
    if tmp_354 == tmp_362 { keys[7] = tmp_336; values[7] = tmp_337; }
    let tmp_363 = keys[8] < tmp_338 || (keys[8] == tmp_338 && values[8] < tmp_339);
    if tmp_354 == tmp_363 { keys[8] = tmp_338; values[8] = tmp_339; }
    let tmp_364 = keys[9] < tmp_340 || (keys[9] == tmp_340 && values[9] < tmp_341);
    if tmp_354 == tmp_364 { keys[9] = tmp_340; values[9] = tmp_341; }
    let tmp_365 = keys[10] < tmp_342 || (keys[10] == tmp_342 && values[10] < tmp_343);
    if tmp_354 == tmp_365 { keys[10] = tmp_342; values[10] = tmp_343; }
    let tmp_366 = keys[11] < tmp_344 || (keys[11] == tmp_344 && values[11] < tmp_345);
    if tmp_354 == tmp_366 { keys[11] = tmp_344; values[11] = tmp_345; }
    let tmp_367 = keys[12] < tmp_346 || (keys[12] == tmp_346 && values[12] < tmp_347);
    if tmp_354 == tmp_367 { keys[12] = tmp_346; values[12] = tmp_347; }
    let tmp_368 = keys[13] < tmp_348 || (keys[13] == tmp_348 && values[13] < tmp_349);
    if tmp_354 == tmp_368 { keys[13] = tmp_348; values[13] = tmp_349; }
    let tmp_369 = keys[14] < tmp_350 || (keys[14] == tmp_350 && values[14] < tmp_351);
    if tmp_354 == tmp_369 { keys[14] = tmp_350; values[14] = tmp_351; }
    let tmp_370 = keys[15] < tmp_352 || (keys[15] == tmp_352 && values[15] < tmp_353);
    if tmp_354 == tmp_370 { keys[15] = tmp_352; values[15] = tmp_353; }
    }
    // exch_local(8,16) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_371 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_371;let tmp_372 = values[0]; values[0] = values[8]; values[8] = tmp_372; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_373 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_373;let tmp_374 = values[1]; values[1] = values[9]; values[9] = tmp_374; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_375 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_375;let tmp_376 = values[2]; values[2] = values[10]; values[10] = tmp_376; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_377 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_377;let tmp_378 = values[3]; values[3] = values[11]; values[11] = tmp_378; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_379 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_379;let tmp_380 = values[4]; values[4] = values[12]; values[12] = tmp_380; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_381 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_381;let tmp_382 = values[5]; values[5] = values[13]; values[13] = tmp_382; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_383 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_383;let tmp_384 = values[6]; values[6] = values[14]; values[14] = tmp_384; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_385 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_385;let tmp_386 = values[7]; values[7] = values[15]; values[15] = tmp_386; }
    }
    // exch_local(4,16) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_387 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_387;let tmp_388 = values[0]; values[0] = values[4]; values[4] = tmp_388; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_389 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_389;let tmp_390 = values[1]; values[1] = values[5]; values[5] = tmp_390; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_391 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_391;let tmp_392 = values[2]; values[2] = values[6]; values[6] = tmp_392; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_393 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_393;let tmp_394 = values[3]; values[3] = values[7]; values[7] = tmp_394; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_395 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_395;let tmp_396 = values[8]; values[8] = values[12]; values[12] = tmp_396; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_397 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_397;let tmp_398 = values[9]; values[9] = values[13]; values[13] = tmp_398; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_399 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_399;let tmp_400 = values[10]; values[10] = values[14]; values[14] = tmp_400; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_401 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_401;let tmp_402 = values[11]; values[11] = values[15]; values[15] = tmp_402; }
    }
    // exch_local(2,16) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_403 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_403;let tmp_404 = values[0]; values[0] = values[2]; values[2] = tmp_404; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_405 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_405;let tmp_406 = values[1]; values[1] = values[3]; values[3] = tmp_406; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_407 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_407;let tmp_408 = values[4]; values[4] = values[6]; values[6] = tmp_408; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_409 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_409;let tmp_410 = values[5]; values[5] = values[7]; values[7] = tmp_410; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_411 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_411;let tmp_412 = values[8]; values[8] = values[10]; values[10] = tmp_412; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_413 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_413;let tmp_414 = values[9]; values[9] = values[11]; values[11] = tmp_414; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_415 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_415;let tmp_416 = values[12]; values[12] = values[14]; values[14] = tmp_416; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_417 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_417;let tmp_418 = values[13]; values[13] = values[15]; values[15] = tmp_418; }
    }
    // exch_local(1,16) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_419 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_419;let tmp_420 = values[0]; values[0] = values[1]; values[1] = tmp_420; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_421 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_421;let tmp_422 = values[2]; values[2] = values[3]; values[3] = tmp_422; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_423 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_423;let tmp_424 = values[4]; values[4] = values[5]; values[5] = tmp_424; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_425 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_425;let tmp_426 = values[6]; values[6] = values[7]; values[7] = tmp_426; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_427 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_427;let tmp_428 = values[8]; values[8] = values[9]; values[9] = tmp_428; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_429 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_429;let tmp_430 = values[10]; values[10] = values[11]; values[11] = tmp_430; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_431 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_431;let tmp_432 = values[12]; values[12] = values[13]; values[13] = tmp_432; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_433 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_433;let tmp_434 = values[14]; values[14] = values[15]; values[15] = tmp_434; }
    }
    // exch_intxn(tmask:7,swbit:2,wpt:16)
    {
    let tmp_435 = subgroupShuffleXor(keys[15], 7u);
    let tmp_436 = subgroupShuffleXor(values[15], 7u);
    let tmp_437 = subgroupShuffleXor(keys[14], 7u);
    let tmp_438 = subgroupShuffleXor(values[14], 7u);
    let tmp_439 = subgroupShuffleXor(keys[13], 7u);
    let tmp_440 = subgroupShuffleXor(values[13], 7u);
    let tmp_441 = subgroupShuffleXor(keys[12], 7u);
    let tmp_442 = subgroupShuffleXor(values[12], 7u);
    let tmp_443 = subgroupShuffleXor(keys[11], 7u);
    let tmp_444 = subgroupShuffleXor(values[11], 7u);
    let tmp_445 = subgroupShuffleXor(keys[10], 7u);
    let tmp_446 = subgroupShuffleXor(values[10], 7u);
    let tmp_447 = subgroupShuffleXor(keys[9], 7u);
    let tmp_448 = subgroupShuffleXor(values[9], 7u);
    let tmp_449 = subgroupShuffleXor(keys[8], 7u);
    let tmp_450 = subgroupShuffleXor(values[8], 7u);
    let tmp_451 = subgroupShuffleXor(keys[7], 7u);
    let tmp_452 = subgroupShuffleXor(values[7], 7u);
    let tmp_453 = subgroupShuffleXor(keys[6], 7u);
    let tmp_454 = subgroupShuffleXor(values[6], 7u);
    let tmp_455 = subgroupShuffleXor(keys[5], 7u);
    let tmp_456 = subgroupShuffleXor(values[5], 7u);
    let tmp_457 = subgroupShuffleXor(keys[4], 7u);
    let tmp_458 = subgroupShuffleXor(values[4], 7u);
    let tmp_459 = subgroupShuffleXor(keys[3], 7u);
    let tmp_460 = subgroupShuffleXor(values[3], 7u);
    let tmp_461 = subgroupShuffleXor(keys[2], 7u);
    let tmp_462 = subgroupShuffleXor(values[2], 7u);
    let tmp_463 = subgroupShuffleXor(keys[1], 7u);
    let tmp_464 = subgroupShuffleXor(values[1], 7u);
    let tmp_465 = subgroupShuffleXor(keys[0], 7u);
    let tmp_466 = subgroupShuffleXor(values[0], 7u);
    let tmp_467 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_468 = keys[0] < tmp_435 || (keys[0] == tmp_435 && values[0] < tmp_436);
    if tmp_467 == tmp_468 { keys[0] = tmp_435; values[0] = tmp_436; }
    let tmp_469 = keys[1] < tmp_437 || (keys[1] == tmp_437 && values[1] < tmp_438);
    if tmp_467 == tmp_469 { keys[1] = tmp_437; values[1] = tmp_438; }
    let tmp_470 = keys[2] < tmp_439 || (keys[2] == tmp_439 && values[2] < tmp_440);
    if tmp_467 == tmp_470 { keys[2] = tmp_439; values[2] = tmp_440; }
    let tmp_471 = keys[3] < tmp_441 || (keys[3] == tmp_441 && values[3] < tmp_442);
    if tmp_467 == tmp_471 { keys[3] = tmp_441; values[3] = tmp_442; }
    let tmp_472 = keys[4] < tmp_443 || (keys[4] == tmp_443 && values[4] < tmp_444);
    if tmp_467 == tmp_472 { keys[4] = tmp_443; values[4] = tmp_444; }
    let tmp_473 = keys[5] < tmp_445 || (keys[5] == tmp_445 && values[5] < tmp_446);
    if tmp_467 == tmp_473 { keys[5] = tmp_445; values[5] = tmp_446; }
    let tmp_474 = keys[6] < tmp_447 || (keys[6] == tmp_447 && values[6] < tmp_448);
    if tmp_467 == tmp_474 { keys[6] = tmp_447; values[6] = tmp_448; }
    let tmp_475 = keys[7] < tmp_449 || (keys[7] == tmp_449 && values[7] < tmp_450);
    if tmp_467 == tmp_475 { keys[7] = tmp_449; values[7] = tmp_450; }
    let tmp_476 = keys[8] < tmp_451 || (keys[8] == tmp_451 && values[8] < tmp_452);
    if tmp_467 == tmp_476 { keys[8] = tmp_451; values[8] = tmp_452; }
    let tmp_477 = keys[9] < tmp_453 || (keys[9] == tmp_453 && values[9] < tmp_454);
    if tmp_467 == tmp_477 { keys[9] = tmp_453; values[9] = tmp_454; }
    let tmp_478 = keys[10] < tmp_455 || (keys[10] == tmp_455 && values[10] < tmp_456);
    if tmp_467 == tmp_478 { keys[10] = tmp_455; values[10] = tmp_456; }
    let tmp_479 = keys[11] < tmp_457 || (keys[11] == tmp_457 && values[11] < tmp_458);
    if tmp_467 == tmp_479 { keys[11] = tmp_457; values[11] = tmp_458; }
    let tmp_480 = keys[12] < tmp_459 || (keys[12] == tmp_459 && values[12] < tmp_460);
    if tmp_467 == tmp_480 { keys[12] = tmp_459; values[12] = tmp_460; }
    let tmp_481 = keys[13] < tmp_461 || (keys[13] == tmp_461 && values[13] < tmp_462);
    if tmp_467 == tmp_481 { keys[13] = tmp_461; values[13] = tmp_462; }
    let tmp_482 = keys[14] < tmp_463 || (keys[14] == tmp_463 && values[14] < tmp_464);
    if tmp_467 == tmp_482 { keys[14] = tmp_463; values[14] = tmp_464; }
    let tmp_483 = keys[15] < tmp_465 || (keys[15] == tmp_465 && values[15] < tmp_466);
    if tmp_467 == tmp_483 { keys[15] = tmp_465; values[15] = tmp_466; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:16) 
    {
    let tmp_484 = subgroupShuffleXor(keys[0], 2u);
    let tmp_485 = subgroupShuffleXor(values[0], 2u);
    let tmp_486 = subgroupShuffleXor(keys[1], 2u);
    let tmp_487 = subgroupShuffleXor(values[1], 2u);
    let tmp_488 = subgroupShuffleXor(keys[2], 2u);
    let tmp_489 = subgroupShuffleXor(values[2], 2u);
    let tmp_490 = subgroupShuffleXor(keys[3], 2u);
    let tmp_491 = subgroupShuffleXor(values[3], 2u);
    let tmp_492 = subgroupShuffleXor(keys[4], 2u);
    let tmp_493 = subgroupShuffleXor(values[4], 2u);
    let tmp_494 = subgroupShuffleXor(keys[5], 2u);
    let tmp_495 = subgroupShuffleXor(values[5], 2u);
    let tmp_496 = subgroupShuffleXor(keys[6], 2u);
    let tmp_497 = subgroupShuffleXor(values[6], 2u);
    let tmp_498 = subgroupShuffleXor(keys[7], 2u);
    let tmp_499 = subgroupShuffleXor(values[7], 2u);
    let tmp_500 = subgroupShuffleXor(keys[8], 2u);
    let tmp_501 = subgroupShuffleXor(values[8], 2u);
    let tmp_502 = subgroupShuffleXor(keys[9], 2u);
    let tmp_503 = subgroupShuffleXor(values[9], 2u);
    let tmp_504 = subgroupShuffleXor(keys[10], 2u);
    let tmp_505 = subgroupShuffleXor(values[10], 2u);
    let tmp_506 = subgroupShuffleXor(keys[11], 2u);
    let tmp_507 = subgroupShuffleXor(values[11], 2u);
    let tmp_508 = subgroupShuffleXor(keys[12], 2u);
    let tmp_509 = subgroupShuffleXor(values[12], 2u);
    let tmp_510 = subgroupShuffleXor(keys[13], 2u);
    let tmp_511 = subgroupShuffleXor(values[13], 2u);
    let tmp_512 = subgroupShuffleXor(keys[14], 2u);
    let tmp_513 = subgroupShuffleXor(values[14], 2u);
    let tmp_514 = subgroupShuffleXor(keys[15], 2u);
    let tmp_515 = subgroupShuffleXor(values[15], 2u);
    let tmp_516 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_517 = keys[0] < tmp_484 || (keys[0] == tmp_484 && values[0] < tmp_485);
    if tmp_516 == tmp_517 { keys[0] = tmp_484; values[0] = tmp_485; }
    let tmp_518 = keys[1] < tmp_486 || (keys[1] == tmp_486 && values[1] < tmp_487);
    if tmp_516 == tmp_518 { keys[1] = tmp_486; values[1] = tmp_487; }
    let tmp_519 = keys[2] < tmp_488 || (keys[2] == tmp_488 && values[2] < tmp_489);
    if tmp_516 == tmp_519 { keys[2] = tmp_488; values[2] = tmp_489; }
    let tmp_520 = keys[3] < tmp_490 || (keys[3] == tmp_490 && values[3] < tmp_491);
    if tmp_516 == tmp_520 { keys[3] = tmp_490; values[3] = tmp_491; }
    let tmp_521 = keys[4] < tmp_492 || (keys[4] == tmp_492 && values[4] < tmp_493);
    if tmp_516 == tmp_521 { keys[4] = tmp_492; values[4] = tmp_493; }
    let tmp_522 = keys[5] < tmp_494 || (keys[5] == tmp_494 && values[5] < tmp_495);
    if tmp_516 == tmp_522 { keys[5] = tmp_494; values[5] = tmp_495; }
    let tmp_523 = keys[6] < tmp_496 || (keys[6] == tmp_496 && values[6] < tmp_497);
    if tmp_516 == tmp_523 { keys[6] = tmp_496; values[6] = tmp_497; }
    let tmp_524 = keys[7] < tmp_498 || (keys[7] == tmp_498 && values[7] < tmp_499);
    if tmp_516 == tmp_524 { keys[7] = tmp_498; values[7] = tmp_499; }
    let tmp_525 = keys[8] < tmp_500 || (keys[8] == tmp_500 && values[8] < tmp_501);
    if tmp_516 == tmp_525 { keys[8] = tmp_500; values[8] = tmp_501; }
    let tmp_526 = keys[9] < tmp_502 || (keys[9] == tmp_502 && values[9] < tmp_503);
    if tmp_516 == tmp_526 { keys[9] = tmp_502; values[9] = tmp_503; }
    let tmp_527 = keys[10] < tmp_504 || (keys[10] == tmp_504 && values[10] < tmp_505);
    if tmp_516 == tmp_527 { keys[10] = tmp_504; values[10] = tmp_505; }
    let tmp_528 = keys[11] < tmp_506 || (keys[11] == tmp_506 && values[11] < tmp_507);
    if tmp_516 == tmp_528 { keys[11] = tmp_506; values[11] = tmp_507; }
    let tmp_529 = keys[12] < tmp_508 || (keys[12] == tmp_508 && values[12] < tmp_509);
    if tmp_516 == tmp_529 { keys[12] = tmp_508; values[12] = tmp_509; }
    let tmp_530 = keys[13] < tmp_510 || (keys[13] == tmp_510 && values[13] < tmp_511);
    if tmp_516 == tmp_530 { keys[13] = tmp_510; values[13] = tmp_511; }
    let tmp_531 = keys[14] < tmp_512 || (keys[14] == tmp_512 && values[14] < tmp_513);
    if tmp_516 == tmp_531 { keys[14] = tmp_512; values[14] = tmp_513; }
    let tmp_532 = keys[15] < tmp_514 || (keys[15] == tmp_514 && values[15] < tmp_515);
    if tmp_516 == tmp_532 { keys[15] = tmp_514; values[15] = tmp_515; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:16) 
    {
    let tmp_533 = subgroupShuffleXor(keys[0], 1u);
    let tmp_534 = subgroupShuffleXor(values[0], 1u);
    let tmp_535 = subgroupShuffleXor(keys[1], 1u);
    let tmp_536 = subgroupShuffleXor(values[1], 1u);
    let tmp_537 = subgroupShuffleXor(keys[2], 1u);
    let tmp_538 = subgroupShuffleXor(values[2], 1u);
    let tmp_539 = subgroupShuffleXor(keys[3], 1u);
    let tmp_540 = subgroupShuffleXor(values[3], 1u);
    let tmp_541 = subgroupShuffleXor(keys[4], 1u);
    let tmp_542 = subgroupShuffleXor(values[4], 1u);
    let tmp_543 = subgroupShuffleXor(keys[5], 1u);
    let tmp_544 = subgroupShuffleXor(values[5], 1u);
    let tmp_545 = subgroupShuffleXor(keys[6], 1u);
    let tmp_546 = subgroupShuffleXor(values[6], 1u);
    let tmp_547 = subgroupShuffleXor(keys[7], 1u);
    let tmp_548 = subgroupShuffleXor(values[7], 1u);
    let tmp_549 = subgroupShuffleXor(keys[8], 1u);
    let tmp_550 = subgroupShuffleXor(values[8], 1u);
    let tmp_551 = subgroupShuffleXor(keys[9], 1u);
    let tmp_552 = subgroupShuffleXor(values[9], 1u);
    let tmp_553 = subgroupShuffleXor(keys[10], 1u);
    let tmp_554 = subgroupShuffleXor(values[10], 1u);
    let tmp_555 = subgroupShuffleXor(keys[11], 1u);
    let tmp_556 = subgroupShuffleXor(values[11], 1u);
    let tmp_557 = subgroupShuffleXor(keys[12], 1u);
    let tmp_558 = subgroupShuffleXor(values[12], 1u);
    let tmp_559 = subgroupShuffleXor(keys[13], 1u);
    let tmp_560 = subgroupShuffleXor(values[13], 1u);
    let tmp_561 = subgroupShuffleXor(keys[14], 1u);
    let tmp_562 = subgroupShuffleXor(values[14], 1u);
    let tmp_563 = subgroupShuffleXor(keys[15], 1u);
    let tmp_564 = subgroupShuffleXor(values[15], 1u);
    let tmp_565 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_566 = keys[0] < tmp_533 || (keys[0] == tmp_533 && values[0] < tmp_534);
    if tmp_565 == tmp_566 { keys[0] = tmp_533; values[0] = tmp_534; }
    let tmp_567 = keys[1] < tmp_535 || (keys[1] == tmp_535 && values[1] < tmp_536);
    if tmp_565 == tmp_567 { keys[1] = tmp_535; values[1] = tmp_536; }
    let tmp_568 = keys[2] < tmp_537 || (keys[2] == tmp_537 && values[2] < tmp_538);
    if tmp_565 == tmp_568 { keys[2] = tmp_537; values[2] = tmp_538; }
    let tmp_569 = keys[3] < tmp_539 || (keys[3] == tmp_539 && values[3] < tmp_540);
    if tmp_565 == tmp_569 { keys[3] = tmp_539; values[3] = tmp_540; }
    let tmp_570 = keys[4] < tmp_541 || (keys[4] == tmp_541 && values[4] < tmp_542);
    if tmp_565 == tmp_570 { keys[4] = tmp_541; values[4] = tmp_542; }
    let tmp_571 = keys[5] < tmp_543 || (keys[5] == tmp_543 && values[5] < tmp_544);
    if tmp_565 == tmp_571 { keys[5] = tmp_543; values[5] = tmp_544; }
    let tmp_572 = keys[6] < tmp_545 || (keys[6] == tmp_545 && values[6] < tmp_546);
    if tmp_565 == tmp_572 { keys[6] = tmp_545; values[6] = tmp_546; }
    let tmp_573 = keys[7] < tmp_547 || (keys[7] == tmp_547 && values[7] < tmp_548);
    if tmp_565 == tmp_573 { keys[7] = tmp_547; values[7] = tmp_548; }
    let tmp_574 = keys[8] < tmp_549 || (keys[8] == tmp_549 && values[8] < tmp_550);
    if tmp_565 == tmp_574 { keys[8] = tmp_549; values[8] = tmp_550; }
    let tmp_575 = keys[9] < tmp_551 || (keys[9] == tmp_551 && values[9] < tmp_552);
    if tmp_565 == tmp_575 { keys[9] = tmp_551; values[9] = tmp_552; }
    let tmp_576 = keys[10] < tmp_553 || (keys[10] == tmp_553 && values[10] < tmp_554);
    if tmp_565 == tmp_576 { keys[10] = tmp_553; values[10] = tmp_554; }
    let tmp_577 = keys[11] < tmp_555 || (keys[11] == tmp_555 && values[11] < tmp_556);
    if tmp_565 == tmp_577 { keys[11] = tmp_555; values[11] = tmp_556; }
    let tmp_578 = keys[12] < tmp_557 || (keys[12] == tmp_557 && values[12] < tmp_558);
    if tmp_565 == tmp_578 { keys[12] = tmp_557; values[12] = tmp_558; }
    let tmp_579 = keys[13] < tmp_559 || (keys[13] == tmp_559 && values[13] < tmp_560);
    if tmp_565 == tmp_579 { keys[13] = tmp_559; values[13] = tmp_560; }
    let tmp_580 = keys[14] < tmp_561 || (keys[14] == tmp_561 && values[14] < tmp_562);
    if tmp_565 == tmp_580 { keys[14] = tmp_561; values[14] = tmp_562; }
    let tmp_581 = keys[15] < tmp_563 || (keys[15] == tmp_563 && values[15] < tmp_564);
    if tmp_565 == tmp_581 { keys[15] = tmp_563; values[15] = tmp_564; }
    }
    // exch_local(8,16) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_582 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_582;let tmp_583 = values[0]; values[0] = values[8]; values[8] = tmp_583; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_584 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_584;let tmp_585 = values[1]; values[1] = values[9]; values[9] = tmp_585; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_586 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_586;let tmp_587 = values[2]; values[2] = values[10]; values[10] = tmp_587; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_588 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_588;let tmp_589 = values[3]; values[3] = values[11]; values[11] = tmp_589; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_590 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_590;let tmp_591 = values[4]; values[4] = values[12]; values[12] = tmp_591; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_592 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_592;let tmp_593 = values[5]; values[5] = values[13]; values[13] = tmp_593; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_594 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_594;let tmp_595 = values[6]; values[6] = values[14]; values[14] = tmp_595; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_596 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_596;let tmp_597 = values[7]; values[7] = values[15]; values[15] = tmp_597; }
    }
    // exch_local(4,16) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_598 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_598;let tmp_599 = values[0]; values[0] = values[4]; values[4] = tmp_599; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_600 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_600;let tmp_601 = values[1]; values[1] = values[5]; values[5] = tmp_601; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_602 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_602;let tmp_603 = values[2]; values[2] = values[6]; values[6] = tmp_603; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_604 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_604;let tmp_605 = values[3]; values[3] = values[7]; values[7] = tmp_605; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_606 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_606;let tmp_607 = values[8]; values[8] = values[12]; values[12] = tmp_607; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_608 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_608;let tmp_609 = values[9]; values[9] = values[13]; values[13] = tmp_609; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_610 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_610;let tmp_611 = values[10]; values[10] = values[14]; values[14] = tmp_611; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_612 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_612;let tmp_613 = values[11]; values[11] = values[15]; values[15] = tmp_613; }
    }
    // exch_local(2,16) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_614 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_614;let tmp_615 = values[0]; values[0] = values[2]; values[2] = tmp_615; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_616 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_616;let tmp_617 = values[1]; values[1] = values[3]; values[3] = tmp_617; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_618 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_618;let tmp_619 = values[4]; values[4] = values[6]; values[6] = tmp_619; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_620 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_620;let tmp_621 = values[5]; values[5] = values[7]; values[7] = tmp_621; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_622 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_622;let tmp_623 = values[8]; values[8] = values[10]; values[10] = tmp_623; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_624 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_624;let tmp_625 = values[9]; values[9] = values[11]; values[11] = tmp_625; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_626 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_626;let tmp_627 = values[12]; values[12] = values[14]; values[14] = tmp_627; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_628 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_628;let tmp_629 = values[13]; values[13] = values[15]; values[15] = tmp_629; }
    }
    // exch_local(1,16) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_630 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_630;let tmp_631 = values[0]; values[0] = values[1]; values[1] = tmp_631; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_632 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_632;let tmp_633 = values[2]; values[2] = values[3]; values[3] = tmp_633; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_634 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_634;let tmp_635 = values[4]; values[4] = values[5]; values[5] = tmp_635; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_636 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_636;let tmp_637 = values[6]; values[6] = values[7]; values[7] = tmp_637; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_638 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_638;let tmp_639 = values[8]; values[8] = values[9]; values[9] = tmp_639; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_640 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_640;let tmp_641 = values[10]; values[10] = values[11]; values[11] = tmp_641; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_642 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_642;let tmp_643 = values[12]; values[12] = values[13]; values[13] = tmp_643; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_644 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_644;let tmp_645 = values[14]; values[14] = values[15]; values[15] = tmp_645; }
    }
    // exch_intxn(tmask:15,swbit:3,wpt:16)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_646 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_647 = seg_base + (local_tid ^ 15u); let tmp_648 = smem_keys[tmp_647 * WPT + 15u]; let tmp_649 = smem_vals[tmp_647 * WPT + 15u]; let tmp_650 = keys[0] < tmp_648 || (keys[0] == tmp_648 && values[0] < tmp_649); if tmp_646 == tmp_650 { keys[0] = tmp_648; values[0] = tmp_649; } let tmp_651 = smem_keys[tmp_647 * WPT + 14u]; let tmp_652 = smem_vals[tmp_647 * WPT + 14u]; let tmp_653 = keys[1] < tmp_651 || (keys[1] == tmp_651 && values[1] < tmp_652); if tmp_646 == tmp_653 { keys[1] = tmp_651; values[1] = tmp_652; } let tmp_654 = smem_keys[tmp_647 * WPT + 13u]; let tmp_655 = smem_vals[tmp_647 * WPT + 13u]; let tmp_656 = keys[2] < tmp_654 || (keys[2] == tmp_654 && values[2] < tmp_655); if tmp_646 == tmp_656 { keys[2] = tmp_654; values[2] = tmp_655; } let tmp_657 = smem_keys[tmp_647 * WPT + 12u]; let tmp_658 = smem_vals[tmp_647 * WPT + 12u]; let tmp_659 = keys[3] < tmp_657 || (keys[3] == tmp_657 && values[3] < tmp_658); if tmp_646 == tmp_659 { keys[3] = tmp_657; values[3] = tmp_658; } let tmp_660 = smem_keys[tmp_647 * WPT + 11u]; let tmp_661 = smem_vals[tmp_647 * WPT + 11u]; let tmp_662 = keys[4] < tmp_660 || (keys[4] == tmp_660 && values[4] < tmp_661); if tmp_646 == tmp_662 { keys[4] = tmp_660; values[4] = tmp_661; } let tmp_663 = smem_keys[tmp_647 * WPT + 10u]; let tmp_664 = smem_vals[tmp_647 * WPT + 10u]; let tmp_665 = keys[5] < tmp_663 || (keys[5] == tmp_663 && values[5] < tmp_664); if tmp_646 == tmp_665 { keys[5] = tmp_663; values[5] = tmp_664; } let tmp_666 = smem_keys[tmp_647 * WPT + 9u]; let tmp_667 = smem_vals[tmp_647 * WPT + 9u]; let tmp_668 = keys[6] < tmp_666 || (keys[6] == tmp_666 && values[6] < tmp_667); if tmp_646 == tmp_668 { keys[6] = tmp_666; values[6] = tmp_667; } let tmp_669 = smem_keys[tmp_647 * WPT + 8u]; let tmp_670 = smem_vals[tmp_647 * WPT + 8u]; let tmp_671 = keys[7] < tmp_669 || (keys[7] == tmp_669 && values[7] < tmp_670); if tmp_646 == tmp_671 { keys[7] = tmp_669; values[7] = tmp_670; } let tmp_672 = smem_keys[tmp_647 * WPT + 7u]; let tmp_673 = smem_vals[tmp_647 * WPT + 7u]; let tmp_674 = keys[8] < tmp_672 || (keys[8] == tmp_672 && values[8] < tmp_673); if tmp_646 == tmp_674 { keys[8] = tmp_672; values[8] = tmp_673; } let tmp_675 = smem_keys[tmp_647 * WPT + 6u]; let tmp_676 = smem_vals[tmp_647 * WPT + 6u]; let tmp_677 = keys[9] < tmp_675 || (keys[9] == tmp_675 && values[9] < tmp_676); if tmp_646 == tmp_677 { keys[9] = tmp_675; values[9] = tmp_676; } let tmp_678 = smem_keys[tmp_647 * WPT + 5u]; let tmp_679 = smem_vals[tmp_647 * WPT + 5u]; let tmp_680 = keys[10] < tmp_678 || (keys[10] == tmp_678 && values[10] < tmp_679); if tmp_646 == tmp_680 { keys[10] = tmp_678; values[10] = tmp_679; } let tmp_681 = smem_keys[tmp_647 * WPT + 4u]; let tmp_682 = smem_vals[tmp_647 * WPT + 4u]; let tmp_683 = keys[11] < tmp_681 || (keys[11] == tmp_681 && values[11] < tmp_682); if tmp_646 == tmp_683 { keys[11] = tmp_681; values[11] = tmp_682; } let tmp_684 = smem_keys[tmp_647 * WPT + 3u]; let tmp_685 = smem_vals[tmp_647 * WPT + 3u]; let tmp_686 = keys[12] < tmp_684 || (keys[12] == tmp_684 && values[12] < tmp_685); if tmp_646 == tmp_686 { keys[12] = tmp_684; values[12] = tmp_685; } let tmp_687 = smem_keys[tmp_647 * WPT + 2u]; let tmp_688 = smem_vals[tmp_647 * WPT + 2u]; let tmp_689 = keys[13] < tmp_687 || (keys[13] == tmp_687 && values[13] < tmp_688); if tmp_646 == tmp_689 { keys[13] = tmp_687; values[13] = tmp_688; } let tmp_690 = smem_keys[tmp_647 * WPT + 1u]; let tmp_691 = smem_vals[tmp_647 * WPT + 1u]; let tmp_692 = keys[14] < tmp_690 || (keys[14] == tmp_690 && values[14] < tmp_691); if tmp_646 == tmp_692 { keys[14] = tmp_690; values[14] = tmp_691; } let tmp_693 = smem_keys[tmp_647 * WPT + 0u]; let tmp_694 = smem_vals[tmp_647 * WPT + 0u]; let tmp_695 = keys[15] < tmp_693 || (keys[15] == tmp_693 && values[15] < tmp_694); if tmp_646 == tmp_695 { keys[15] = tmp_693; values[15] = tmp_694; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:16) 
    {
    let tmp_696 = subgroupShuffleXor(keys[0], 4u);
    let tmp_697 = subgroupShuffleXor(values[0], 4u);
    let tmp_698 = subgroupShuffleXor(keys[1], 4u);
    let tmp_699 = subgroupShuffleXor(values[1], 4u);
    let tmp_700 = subgroupShuffleXor(keys[2], 4u);
    let tmp_701 = subgroupShuffleXor(values[2], 4u);
    let tmp_702 = subgroupShuffleXor(keys[3], 4u);
    let tmp_703 = subgroupShuffleXor(values[3], 4u);
    let tmp_704 = subgroupShuffleXor(keys[4], 4u);
    let tmp_705 = subgroupShuffleXor(values[4], 4u);
    let tmp_706 = subgroupShuffleXor(keys[5], 4u);
    let tmp_707 = subgroupShuffleXor(values[5], 4u);
    let tmp_708 = subgroupShuffleXor(keys[6], 4u);
    let tmp_709 = subgroupShuffleXor(values[6], 4u);
    let tmp_710 = subgroupShuffleXor(keys[7], 4u);
    let tmp_711 = subgroupShuffleXor(values[7], 4u);
    let tmp_712 = subgroupShuffleXor(keys[8], 4u);
    let tmp_713 = subgroupShuffleXor(values[8], 4u);
    let tmp_714 = subgroupShuffleXor(keys[9], 4u);
    let tmp_715 = subgroupShuffleXor(values[9], 4u);
    let tmp_716 = subgroupShuffleXor(keys[10], 4u);
    let tmp_717 = subgroupShuffleXor(values[10], 4u);
    let tmp_718 = subgroupShuffleXor(keys[11], 4u);
    let tmp_719 = subgroupShuffleXor(values[11], 4u);
    let tmp_720 = subgroupShuffleXor(keys[12], 4u);
    let tmp_721 = subgroupShuffleXor(values[12], 4u);
    let tmp_722 = subgroupShuffleXor(keys[13], 4u);
    let tmp_723 = subgroupShuffleXor(values[13], 4u);
    let tmp_724 = subgroupShuffleXor(keys[14], 4u);
    let tmp_725 = subgroupShuffleXor(values[14], 4u);
    let tmp_726 = subgroupShuffleXor(keys[15], 4u);
    let tmp_727 = subgroupShuffleXor(values[15], 4u);
    let tmp_728 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_729 = keys[0] < tmp_696 || (keys[0] == tmp_696 && values[0] < tmp_697);
    if tmp_728 == tmp_729 { keys[0] = tmp_696; values[0] = tmp_697; }
    let tmp_730 = keys[1] < tmp_698 || (keys[1] == tmp_698 && values[1] < tmp_699);
    if tmp_728 == tmp_730 { keys[1] = tmp_698; values[1] = tmp_699; }
    let tmp_731 = keys[2] < tmp_700 || (keys[2] == tmp_700 && values[2] < tmp_701);
    if tmp_728 == tmp_731 { keys[2] = tmp_700; values[2] = tmp_701; }
    let tmp_732 = keys[3] < tmp_702 || (keys[3] == tmp_702 && values[3] < tmp_703);
    if tmp_728 == tmp_732 { keys[3] = tmp_702; values[3] = tmp_703; }
    let tmp_733 = keys[4] < tmp_704 || (keys[4] == tmp_704 && values[4] < tmp_705);
    if tmp_728 == tmp_733 { keys[4] = tmp_704; values[4] = tmp_705; }
    let tmp_734 = keys[5] < tmp_706 || (keys[5] == tmp_706 && values[5] < tmp_707);
    if tmp_728 == tmp_734 { keys[5] = tmp_706; values[5] = tmp_707; }
    let tmp_735 = keys[6] < tmp_708 || (keys[6] == tmp_708 && values[6] < tmp_709);
    if tmp_728 == tmp_735 { keys[6] = tmp_708; values[6] = tmp_709; }
    let tmp_736 = keys[7] < tmp_710 || (keys[7] == tmp_710 && values[7] < tmp_711);
    if tmp_728 == tmp_736 { keys[7] = tmp_710; values[7] = tmp_711; }
    let tmp_737 = keys[8] < tmp_712 || (keys[8] == tmp_712 && values[8] < tmp_713);
    if tmp_728 == tmp_737 { keys[8] = tmp_712; values[8] = tmp_713; }
    let tmp_738 = keys[9] < tmp_714 || (keys[9] == tmp_714 && values[9] < tmp_715);
    if tmp_728 == tmp_738 { keys[9] = tmp_714; values[9] = tmp_715; }
    let tmp_739 = keys[10] < tmp_716 || (keys[10] == tmp_716 && values[10] < tmp_717);
    if tmp_728 == tmp_739 { keys[10] = tmp_716; values[10] = tmp_717; }
    let tmp_740 = keys[11] < tmp_718 || (keys[11] == tmp_718 && values[11] < tmp_719);
    if tmp_728 == tmp_740 { keys[11] = tmp_718; values[11] = tmp_719; }
    let tmp_741 = keys[12] < tmp_720 || (keys[12] == tmp_720 && values[12] < tmp_721);
    if tmp_728 == tmp_741 { keys[12] = tmp_720; values[12] = tmp_721; }
    let tmp_742 = keys[13] < tmp_722 || (keys[13] == tmp_722 && values[13] < tmp_723);
    if tmp_728 == tmp_742 { keys[13] = tmp_722; values[13] = tmp_723; }
    let tmp_743 = keys[14] < tmp_724 || (keys[14] == tmp_724 && values[14] < tmp_725);
    if tmp_728 == tmp_743 { keys[14] = tmp_724; values[14] = tmp_725; }
    let tmp_744 = keys[15] < tmp_726 || (keys[15] == tmp_726 && values[15] < tmp_727);
    if tmp_728 == tmp_744 { keys[15] = tmp_726; values[15] = tmp_727; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:16) 
    {
    let tmp_745 = subgroupShuffleXor(keys[0], 2u);
    let tmp_746 = subgroupShuffleXor(values[0], 2u);
    let tmp_747 = subgroupShuffleXor(keys[1], 2u);
    let tmp_748 = subgroupShuffleXor(values[1], 2u);
    let tmp_749 = subgroupShuffleXor(keys[2], 2u);
    let tmp_750 = subgroupShuffleXor(values[2], 2u);
    let tmp_751 = subgroupShuffleXor(keys[3], 2u);
    let tmp_752 = subgroupShuffleXor(values[3], 2u);
    let tmp_753 = subgroupShuffleXor(keys[4], 2u);
    let tmp_754 = subgroupShuffleXor(values[4], 2u);
    let tmp_755 = subgroupShuffleXor(keys[5], 2u);
    let tmp_756 = subgroupShuffleXor(values[5], 2u);
    let tmp_757 = subgroupShuffleXor(keys[6], 2u);
    let tmp_758 = subgroupShuffleXor(values[6], 2u);
    let tmp_759 = subgroupShuffleXor(keys[7], 2u);
    let tmp_760 = subgroupShuffleXor(values[7], 2u);
    let tmp_761 = subgroupShuffleXor(keys[8], 2u);
    let tmp_762 = subgroupShuffleXor(values[8], 2u);
    let tmp_763 = subgroupShuffleXor(keys[9], 2u);
    let tmp_764 = subgroupShuffleXor(values[9], 2u);
    let tmp_765 = subgroupShuffleXor(keys[10], 2u);
    let tmp_766 = subgroupShuffleXor(values[10], 2u);
    let tmp_767 = subgroupShuffleXor(keys[11], 2u);
    let tmp_768 = subgroupShuffleXor(values[11], 2u);
    let tmp_769 = subgroupShuffleXor(keys[12], 2u);
    let tmp_770 = subgroupShuffleXor(values[12], 2u);
    let tmp_771 = subgroupShuffleXor(keys[13], 2u);
    let tmp_772 = subgroupShuffleXor(values[13], 2u);
    let tmp_773 = subgroupShuffleXor(keys[14], 2u);
    let tmp_774 = subgroupShuffleXor(values[14], 2u);
    let tmp_775 = subgroupShuffleXor(keys[15], 2u);
    let tmp_776 = subgroupShuffleXor(values[15], 2u);
    let tmp_777 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_778 = keys[0] < tmp_745 || (keys[0] == tmp_745 && values[0] < tmp_746);
    if tmp_777 == tmp_778 { keys[0] = tmp_745; values[0] = tmp_746; }
    let tmp_779 = keys[1] < tmp_747 || (keys[1] == tmp_747 && values[1] < tmp_748);
    if tmp_777 == tmp_779 { keys[1] = tmp_747; values[1] = tmp_748; }
    let tmp_780 = keys[2] < tmp_749 || (keys[2] == tmp_749 && values[2] < tmp_750);
    if tmp_777 == tmp_780 { keys[2] = tmp_749; values[2] = tmp_750; }
    let tmp_781 = keys[3] < tmp_751 || (keys[3] == tmp_751 && values[3] < tmp_752);
    if tmp_777 == tmp_781 { keys[3] = tmp_751; values[3] = tmp_752; }
    let tmp_782 = keys[4] < tmp_753 || (keys[4] == tmp_753 && values[4] < tmp_754);
    if tmp_777 == tmp_782 { keys[4] = tmp_753; values[4] = tmp_754; }
    let tmp_783 = keys[5] < tmp_755 || (keys[5] == tmp_755 && values[5] < tmp_756);
    if tmp_777 == tmp_783 { keys[5] = tmp_755; values[5] = tmp_756; }
    let tmp_784 = keys[6] < tmp_757 || (keys[6] == tmp_757 && values[6] < tmp_758);
    if tmp_777 == tmp_784 { keys[6] = tmp_757; values[6] = tmp_758; }
    let tmp_785 = keys[7] < tmp_759 || (keys[7] == tmp_759 && values[7] < tmp_760);
    if tmp_777 == tmp_785 { keys[7] = tmp_759; values[7] = tmp_760; }
    let tmp_786 = keys[8] < tmp_761 || (keys[8] == tmp_761 && values[8] < tmp_762);
    if tmp_777 == tmp_786 { keys[8] = tmp_761; values[8] = tmp_762; }
    let tmp_787 = keys[9] < tmp_763 || (keys[9] == tmp_763 && values[9] < tmp_764);
    if tmp_777 == tmp_787 { keys[9] = tmp_763; values[9] = tmp_764; }
    let tmp_788 = keys[10] < tmp_765 || (keys[10] == tmp_765 && values[10] < tmp_766);
    if tmp_777 == tmp_788 { keys[10] = tmp_765; values[10] = tmp_766; }
    let tmp_789 = keys[11] < tmp_767 || (keys[11] == tmp_767 && values[11] < tmp_768);
    if tmp_777 == tmp_789 { keys[11] = tmp_767; values[11] = tmp_768; }
    let tmp_790 = keys[12] < tmp_769 || (keys[12] == tmp_769 && values[12] < tmp_770);
    if tmp_777 == tmp_790 { keys[12] = tmp_769; values[12] = tmp_770; }
    let tmp_791 = keys[13] < tmp_771 || (keys[13] == tmp_771 && values[13] < tmp_772);
    if tmp_777 == tmp_791 { keys[13] = tmp_771; values[13] = tmp_772; }
    let tmp_792 = keys[14] < tmp_773 || (keys[14] == tmp_773 && values[14] < tmp_774);
    if tmp_777 == tmp_792 { keys[14] = tmp_773; values[14] = tmp_774; }
    let tmp_793 = keys[15] < tmp_775 || (keys[15] == tmp_775 && values[15] < tmp_776);
    if tmp_777 == tmp_793 { keys[15] = tmp_775; values[15] = tmp_776; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:16) 
    {
    let tmp_794 = subgroupShuffleXor(keys[0], 1u);
    let tmp_795 = subgroupShuffleXor(values[0], 1u);
    let tmp_796 = subgroupShuffleXor(keys[1], 1u);
    let tmp_797 = subgroupShuffleXor(values[1], 1u);
    let tmp_798 = subgroupShuffleXor(keys[2], 1u);
    let tmp_799 = subgroupShuffleXor(values[2], 1u);
    let tmp_800 = subgroupShuffleXor(keys[3], 1u);
    let tmp_801 = subgroupShuffleXor(values[3], 1u);
    let tmp_802 = subgroupShuffleXor(keys[4], 1u);
    let tmp_803 = subgroupShuffleXor(values[4], 1u);
    let tmp_804 = subgroupShuffleXor(keys[5], 1u);
    let tmp_805 = subgroupShuffleXor(values[5], 1u);
    let tmp_806 = subgroupShuffleXor(keys[6], 1u);
    let tmp_807 = subgroupShuffleXor(values[6], 1u);
    let tmp_808 = subgroupShuffleXor(keys[7], 1u);
    let tmp_809 = subgroupShuffleXor(values[7], 1u);
    let tmp_810 = subgroupShuffleXor(keys[8], 1u);
    let tmp_811 = subgroupShuffleXor(values[8], 1u);
    let tmp_812 = subgroupShuffleXor(keys[9], 1u);
    let tmp_813 = subgroupShuffleXor(values[9], 1u);
    let tmp_814 = subgroupShuffleXor(keys[10], 1u);
    let tmp_815 = subgroupShuffleXor(values[10], 1u);
    let tmp_816 = subgroupShuffleXor(keys[11], 1u);
    let tmp_817 = subgroupShuffleXor(values[11], 1u);
    let tmp_818 = subgroupShuffleXor(keys[12], 1u);
    let tmp_819 = subgroupShuffleXor(values[12], 1u);
    let tmp_820 = subgroupShuffleXor(keys[13], 1u);
    let tmp_821 = subgroupShuffleXor(values[13], 1u);
    let tmp_822 = subgroupShuffleXor(keys[14], 1u);
    let tmp_823 = subgroupShuffleXor(values[14], 1u);
    let tmp_824 = subgroupShuffleXor(keys[15], 1u);
    let tmp_825 = subgroupShuffleXor(values[15], 1u);
    let tmp_826 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_827 = keys[0] < tmp_794 || (keys[0] == tmp_794 && values[0] < tmp_795);
    if tmp_826 == tmp_827 { keys[0] = tmp_794; values[0] = tmp_795; }
    let tmp_828 = keys[1] < tmp_796 || (keys[1] == tmp_796 && values[1] < tmp_797);
    if tmp_826 == tmp_828 { keys[1] = tmp_796; values[1] = tmp_797; }
    let tmp_829 = keys[2] < tmp_798 || (keys[2] == tmp_798 && values[2] < tmp_799);
    if tmp_826 == tmp_829 { keys[2] = tmp_798; values[2] = tmp_799; }
    let tmp_830 = keys[3] < tmp_800 || (keys[3] == tmp_800 && values[3] < tmp_801);
    if tmp_826 == tmp_830 { keys[3] = tmp_800; values[3] = tmp_801; }
    let tmp_831 = keys[4] < tmp_802 || (keys[4] == tmp_802 && values[4] < tmp_803);
    if tmp_826 == tmp_831 { keys[4] = tmp_802; values[4] = tmp_803; }
    let tmp_832 = keys[5] < tmp_804 || (keys[5] == tmp_804 && values[5] < tmp_805);
    if tmp_826 == tmp_832 { keys[5] = tmp_804; values[5] = tmp_805; }
    let tmp_833 = keys[6] < tmp_806 || (keys[6] == tmp_806 && values[6] < tmp_807);
    if tmp_826 == tmp_833 { keys[6] = tmp_806; values[6] = tmp_807; }
    let tmp_834 = keys[7] < tmp_808 || (keys[7] == tmp_808 && values[7] < tmp_809);
    if tmp_826 == tmp_834 { keys[7] = tmp_808; values[7] = tmp_809; }
    let tmp_835 = keys[8] < tmp_810 || (keys[8] == tmp_810 && values[8] < tmp_811);
    if tmp_826 == tmp_835 { keys[8] = tmp_810; values[8] = tmp_811; }
    let tmp_836 = keys[9] < tmp_812 || (keys[9] == tmp_812 && values[9] < tmp_813);
    if tmp_826 == tmp_836 { keys[9] = tmp_812; values[9] = tmp_813; }
    let tmp_837 = keys[10] < tmp_814 || (keys[10] == tmp_814 && values[10] < tmp_815);
    if tmp_826 == tmp_837 { keys[10] = tmp_814; values[10] = tmp_815; }
    let tmp_838 = keys[11] < tmp_816 || (keys[11] == tmp_816 && values[11] < tmp_817);
    if tmp_826 == tmp_838 { keys[11] = tmp_816; values[11] = tmp_817; }
    let tmp_839 = keys[12] < tmp_818 || (keys[12] == tmp_818 && values[12] < tmp_819);
    if tmp_826 == tmp_839 { keys[12] = tmp_818; values[12] = tmp_819; }
    let tmp_840 = keys[13] < tmp_820 || (keys[13] == tmp_820 && values[13] < tmp_821);
    if tmp_826 == tmp_840 { keys[13] = tmp_820; values[13] = tmp_821; }
    let tmp_841 = keys[14] < tmp_822 || (keys[14] == tmp_822 && values[14] < tmp_823);
    if tmp_826 == tmp_841 { keys[14] = tmp_822; values[14] = tmp_823; }
    let tmp_842 = keys[15] < tmp_824 || (keys[15] == tmp_824 && values[15] < tmp_825);
    if tmp_826 == tmp_842 { keys[15] = tmp_824; values[15] = tmp_825; }
    }
    // exch_local(8,16) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_843 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_843;let tmp_844 = values[0]; values[0] = values[8]; values[8] = tmp_844; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_845 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_845;let tmp_846 = values[1]; values[1] = values[9]; values[9] = tmp_846; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_847 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_847;let tmp_848 = values[2]; values[2] = values[10]; values[10] = tmp_848; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_849 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_849;let tmp_850 = values[3]; values[3] = values[11]; values[11] = tmp_850; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_851 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_851;let tmp_852 = values[4]; values[4] = values[12]; values[12] = tmp_852; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_853 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_853;let tmp_854 = values[5]; values[5] = values[13]; values[13] = tmp_854; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_855 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_855;let tmp_856 = values[6]; values[6] = values[14]; values[14] = tmp_856; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_857 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_857;let tmp_858 = values[7]; values[7] = values[15]; values[15] = tmp_858; }
    }
    // exch_local(4,16) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_859 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_859;let tmp_860 = values[0]; values[0] = values[4]; values[4] = tmp_860; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_861 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_861;let tmp_862 = values[1]; values[1] = values[5]; values[5] = tmp_862; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_863 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_863;let tmp_864 = values[2]; values[2] = values[6]; values[6] = tmp_864; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_865 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_865;let tmp_866 = values[3]; values[3] = values[7]; values[7] = tmp_866; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_867 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_867;let tmp_868 = values[8]; values[8] = values[12]; values[12] = tmp_868; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_869 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_869;let tmp_870 = values[9]; values[9] = values[13]; values[13] = tmp_870; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_871 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_871;let tmp_872 = values[10]; values[10] = values[14]; values[14] = tmp_872; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_873 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_873;let tmp_874 = values[11]; values[11] = values[15]; values[15] = tmp_874; }
    }
    // exch_local(2,16) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_875 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_875;let tmp_876 = values[0]; values[0] = values[2]; values[2] = tmp_876; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_877 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_877;let tmp_878 = values[1]; values[1] = values[3]; values[3] = tmp_878; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_879 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_879;let tmp_880 = values[4]; values[4] = values[6]; values[6] = tmp_880; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_881 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_881;let tmp_882 = values[5]; values[5] = values[7]; values[7] = tmp_882; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_883 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_883;let tmp_884 = values[8]; values[8] = values[10]; values[10] = tmp_884; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_885 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_885;let tmp_886 = values[9]; values[9] = values[11]; values[11] = tmp_886; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_887 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_887;let tmp_888 = values[12]; values[12] = values[14]; values[14] = tmp_888; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_889 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_889;let tmp_890 = values[13]; values[13] = values[15]; values[15] = tmp_890; }
    }
    // exch_local(1,16) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_891 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_891;let tmp_892 = values[0]; values[0] = values[1]; values[1] = tmp_892; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_893 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_893;let tmp_894 = values[2]; values[2] = values[3]; values[3] = tmp_894; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_895 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_895;let tmp_896 = values[4]; values[4] = values[5]; values[5] = tmp_896; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_897 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_897;let tmp_898 = values[6]; values[6] = values[7]; values[7] = tmp_898; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_899 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_899;let tmp_900 = values[8]; values[8] = values[9]; values[9] = tmp_900; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_901 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_901;let tmp_902 = values[10]; values[10] = values[11]; values[11] = tmp_902; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_903 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_903;let tmp_904 = values[12]; values[12] = values[13]; values[13] = tmp_904; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_905 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_905;let tmp_906 = values[14]; values[14] = values[15]; values[15] = tmp_906; }
    }
    // exch_intxn(tmask:31,swbit:4,wpt:16)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_907 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_908 = seg_base + (local_tid ^ 31u); let tmp_909 = smem_keys[tmp_908 * WPT + 15u]; let tmp_910 = smem_vals[tmp_908 * WPT + 15u]; let tmp_911 = keys[0] < tmp_909 || (keys[0] == tmp_909 && values[0] < tmp_910); if tmp_907 == tmp_911 { keys[0] = tmp_909; values[0] = tmp_910; } let tmp_912 = smem_keys[tmp_908 * WPT + 14u]; let tmp_913 = smem_vals[tmp_908 * WPT + 14u]; let tmp_914 = keys[1] < tmp_912 || (keys[1] == tmp_912 && values[1] < tmp_913); if tmp_907 == tmp_914 { keys[1] = tmp_912; values[1] = tmp_913; } let tmp_915 = smem_keys[tmp_908 * WPT + 13u]; let tmp_916 = smem_vals[tmp_908 * WPT + 13u]; let tmp_917 = keys[2] < tmp_915 || (keys[2] == tmp_915 && values[2] < tmp_916); if tmp_907 == tmp_917 { keys[2] = tmp_915; values[2] = tmp_916; } let tmp_918 = smem_keys[tmp_908 * WPT + 12u]; let tmp_919 = smem_vals[tmp_908 * WPT + 12u]; let tmp_920 = keys[3] < tmp_918 || (keys[3] == tmp_918 && values[3] < tmp_919); if tmp_907 == tmp_920 { keys[3] = tmp_918; values[3] = tmp_919; } let tmp_921 = smem_keys[tmp_908 * WPT + 11u]; let tmp_922 = smem_vals[tmp_908 * WPT + 11u]; let tmp_923 = keys[4] < tmp_921 || (keys[4] == tmp_921 && values[4] < tmp_922); if tmp_907 == tmp_923 { keys[4] = tmp_921; values[4] = tmp_922; } let tmp_924 = smem_keys[tmp_908 * WPT + 10u]; let tmp_925 = smem_vals[tmp_908 * WPT + 10u]; let tmp_926 = keys[5] < tmp_924 || (keys[5] == tmp_924 && values[5] < tmp_925); if tmp_907 == tmp_926 { keys[5] = tmp_924; values[5] = tmp_925; } let tmp_927 = smem_keys[tmp_908 * WPT + 9u]; let tmp_928 = smem_vals[tmp_908 * WPT + 9u]; let tmp_929 = keys[6] < tmp_927 || (keys[6] == tmp_927 && values[6] < tmp_928); if tmp_907 == tmp_929 { keys[6] = tmp_927; values[6] = tmp_928; } let tmp_930 = smem_keys[tmp_908 * WPT + 8u]; let tmp_931 = smem_vals[tmp_908 * WPT + 8u]; let tmp_932 = keys[7] < tmp_930 || (keys[7] == tmp_930 && values[7] < tmp_931); if tmp_907 == tmp_932 { keys[7] = tmp_930; values[7] = tmp_931; } let tmp_933 = smem_keys[tmp_908 * WPT + 7u]; let tmp_934 = smem_vals[tmp_908 * WPT + 7u]; let tmp_935 = keys[8] < tmp_933 || (keys[8] == tmp_933 && values[8] < tmp_934); if tmp_907 == tmp_935 { keys[8] = tmp_933; values[8] = tmp_934; } let tmp_936 = smem_keys[tmp_908 * WPT + 6u]; let tmp_937 = smem_vals[tmp_908 * WPT + 6u]; let tmp_938 = keys[9] < tmp_936 || (keys[9] == tmp_936 && values[9] < tmp_937); if tmp_907 == tmp_938 { keys[9] = tmp_936; values[9] = tmp_937; } let tmp_939 = smem_keys[tmp_908 * WPT + 5u]; let tmp_940 = smem_vals[tmp_908 * WPT + 5u]; let tmp_941 = keys[10] < tmp_939 || (keys[10] == tmp_939 && values[10] < tmp_940); if tmp_907 == tmp_941 { keys[10] = tmp_939; values[10] = tmp_940; } let tmp_942 = smem_keys[tmp_908 * WPT + 4u]; let tmp_943 = smem_vals[tmp_908 * WPT + 4u]; let tmp_944 = keys[11] < tmp_942 || (keys[11] == tmp_942 && values[11] < tmp_943); if tmp_907 == tmp_944 { keys[11] = tmp_942; values[11] = tmp_943; } let tmp_945 = smem_keys[tmp_908 * WPT + 3u]; let tmp_946 = smem_vals[tmp_908 * WPT + 3u]; let tmp_947 = keys[12] < tmp_945 || (keys[12] == tmp_945 && values[12] < tmp_946); if tmp_907 == tmp_947 { keys[12] = tmp_945; values[12] = tmp_946; } let tmp_948 = smem_keys[tmp_908 * WPT + 2u]; let tmp_949 = smem_vals[tmp_908 * WPT + 2u]; let tmp_950 = keys[13] < tmp_948 || (keys[13] == tmp_948 && values[13] < tmp_949); if tmp_907 == tmp_950 { keys[13] = tmp_948; values[13] = tmp_949; } let tmp_951 = smem_keys[tmp_908 * WPT + 1u]; let tmp_952 = smem_vals[tmp_908 * WPT + 1u]; let tmp_953 = keys[14] < tmp_951 || (keys[14] == tmp_951 && values[14] < tmp_952); if tmp_907 == tmp_953 { keys[14] = tmp_951; values[14] = tmp_952; } let tmp_954 = smem_keys[tmp_908 * WPT + 0u]; let tmp_955 = smem_vals[tmp_908 * WPT + 0u]; let tmp_956 = keys[15] < tmp_954 || (keys[15] == tmp_954 && values[15] < tmp_955); if tmp_907 == tmp_956 { keys[15] = tmp_954; values[15] = tmp_955; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:16) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_957 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_958 = seg_base + (local_tid ^ 8u); let tmp_959 = smem_keys[tmp_958 * WPT + 0u]; let tmp_960 = smem_vals[tmp_958 * WPT + 0u]; let tmp_961 = keys[0] < tmp_959 || (keys[0] == tmp_959 && values[0] < tmp_960); if tmp_957 == tmp_961 { keys[0] = tmp_959; values[0] = tmp_960; } let tmp_962 = smem_keys[tmp_958 * WPT + 1u]; let tmp_963 = smem_vals[tmp_958 * WPT + 1u]; let tmp_964 = keys[1] < tmp_962 || (keys[1] == tmp_962 && values[1] < tmp_963); if tmp_957 == tmp_964 { keys[1] = tmp_962; values[1] = tmp_963; } let tmp_965 = smem_keys[tmp_958 * WPT + 2u]; let tmp_966 = smem_vals[tmp_958 * WPT + 2u]; let tmp_967 = keys[2] < tmp_965 || (keys[2] == tmp_965 && values[2] < tmp_966); if tmp_957 == tmp_967 { keys[2] = tmp_965; values[2] = tmp_966; } let tmp_968 = smem_keys[tmp_958 * WPT + 3u]; let tmp_969 = smem_vals[tmp_958 * WPT + 3u]; let tmp_970 = keys[3] < tmp_968 || (keys[3] == tmp_968 && values[3] < tmp_969); if tmp_957 == tmp_970 { keys[3] = tmp_968; values[3] = tmp_969; } let tmp_971 = smem_keys[tmp_958 * WPT + 4u]; let tmp_972 = smem_vals[tmp_958 * WPT + 4u]; let tmp_973 = keys[4] < tmp_971 || (keys[4] == tmp_971 && values[4] < tmp_972); if tmp_957 == tmp_973 { keys[4] = tmp_971; values[4] = tmp_972; } let tmp_974 = smem_keys[tmp_958 * WPT + 5u]; let tmp_975 = smem_vals[tmp_958 * WPT + 5u]; let tmp_976 = keys[5] < tmp_974 || (keys[5] == tmp_974 && values[5] < tmp_975); if tmp_957 == tmp_976 { keys[5] = tmp_974; values[5] = tmp_975; } let tmp_977 = smem_keys[tmp_958 * WPT + 6u]; let tmp_978 = smem_vals[tmp_958 * WPT + 6u]; let tmp_979 = keys[6] < tmp_977 || (keys[6] == tmp_977 && values[6] < tmp_978); if tmp_957 == tmp_979 { keys[6] = tmp_977; values[6] = tmp_978; } let tmp_980 = smem_keys[tmp_958 * WPT + 7u]; let tmp_981 = smem_vals[tmp_958 * WPT + 7u]; let tmp_982 = keys[7] < tmp_980 || (keys[7] == tmp_980 && values[7] < tmp_981); if tmp_957 == tmp_982 { keys[7] = tmp_980; values[7] = tmp_981; } let tmp_983 = smem_keys[tmp_958 * WPT + 8u]; let tmp_984 = smem_vals[tmp_958 * WPT + 8u]; let tmp_985 = keys[8] < tmp_983 || (keys[8] == tmp_983 && values[8] < tmp_984); if tmp_957 == tmp_985 { keys[8] = tmp_983; values[8] = tmp_984; } let tmp_986 = smem_keys[tmp_958 * WPT + 9u]; let tmp_987 = smem_vals[tmp_958 * WPT + 9u]; let tmp_988 = keys[9] < tmp_986 || (keys[9] == tmp_986 && values[9] < tmp_987); if tmp_957 == tmp_988 { keys[9] = tmp_986; values[9] = tmp_987; } let tmp_989 = smem_keys[tmp_958 * WPT + 10u]; let tmp_990 = smem_vals[tmp_958 * WPT + 10u]; let tmp_991 = keys[10] < tmp_989 || (keys[10] == tmp_989 && values[10] < tmp_990); if tmp_957 == tmp_991 { keys[10] = tmp_989; values[10] = tmp_990; } let tmp_992 = smem_keys[tmp_958 * WPT + 11u]; let tmp_993 = smem_vals[tmp_958 * WPT + 11u]; let tmp_994 = keys[11] < tmp_992 || (keys[11] == tmp_992 && values[11] < tmp_993); if tmp_957 == tmp_994 { keys[11] = tmp_992; values[11] = tmp_993; } let tmp_995 = smem_keys[tmp_958 * WPT + 12u]; let tmp_996 = smem_vals[tmp_958 * WPT + 12u]; let tmp_997 = keys[12] < tmp_995 || (keys[12] == tmp_995 && values[12] < tmp_996); if tmp_957 == tmp_997 { keys[12] = tmp_995; values[12] = tmp_996; } let tmp_998 = smem_keys[tmp_958 * WPT + 13u]; let tmp_999 = smem_vals[tmp_958 * WPT + 13u]; let tmp_1000 = keys[13] < tmp_998 || (keys[13] == tmp_998 && values[13] < tmp_999); if tmp_957 == tmp_1000 { keys[13] = tmp_998; values[13] = tmp_999; } let tmp_1001 = smem_keys[tmp_958 * WPT + 14u]; let tmp_1002 = smem_vals[tmp_958 * WPT + 14u]; let tmp_1003 = keys[14] < tmp_1001 || (keys[14] == tmp_1001 && values[14] < tmp_1002); if tmp_957 == tmp_1003 { keys[14] = tmp_1001; values[14] = tmp_1002; } let tmp_1004 = smem_keys[tmp_958 * WPT + 15u]; let tmp_1005 = smem_vals[tmp_958 * WPT + 15u]; let tmp_1006 = keys[15] < tmp_1004 || (keys[15] == tmp_1004 && values[15] < tmp_1005); if tmp_957 == tmp_1006 { keys[15] = tmp_1004; values[15] = tmp_1005; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:16) 
    {
    let tmp_1007 = subgroupShuffleXor(keys[0], 4u);
    let tmp_1008 = subgroupShuffleXor(values[0], 4u);
    let tmp_1009 = subgroupShuffleXor(keys[1], 4u);
    let tmp_1010 = subgroupShuffleXor(values[1], 4u);
    let tmp_1011 = subgroupShuffleXor(keys[2], 4u);
    let tmp_1012 = subgroupShuffleXor(values[2], 4u);
    let tmp_1013 = subgroupShuffleXor(keys[3], 4u);
    let tmp_1014 = subgroupShuffleXor(values[3], 4u);
    let tmp_1015 = subgroupShuffleXor(keys[4], 4u);
    let tmp_1016 = subgroupShuffleXor(values[4], 4u);
    let tmp_1017 = subgroupShuffleXor(keys[5], 4u);
    let tmp_1018 = subgroupShuffleXor(values[5], 4u);
    let tmp_1019 = subgroupShuffleXor(keys[6], 4u);
    let tmp_1020 = subgroupShuffleXor(values[6], 4u);
    let tmp_1021 = subgroupShuffleXor(keys[7], 4u);
    let tmp_1022 = subgroupShuffleXor(values[7], 4u);
    let tmp_1023 = subgroupShuffleXor(keys[8], 4u);
    let tmp_1024 = subgroupShuffleXor(values[8], 4u);
    let tmp_1025 = subgroupShuffleXor(keys[9], 4u);
    let tmp_1026 = subgroupShuffleXor(values[9], 4u);
    let tmp_1027 = subgroupShuffleXor(keys[10], 4u);
    let tmp_1028 = subgroupShuffleXor(values[10], 4u);
    let tmp_1029 = subgroupShuffleXor(keys[11], 4u);
    let tmp_1030 = subgroupShuffleXor(values[11], 4u);
    let tmp_1031 = subgroupShuffleXor(keys[12], 4u);
    let tmp_1032 = subgroupShuffleXor(values[12], 4u);
    let tmp_1033 = subgroupShuffleXor(keys[13], 4u);
    let tmp_1034 = subgroupShuffleXor(values[13], 4u);
    let tmp_1035 = subgroupShuffleXor(keys[14], 4u);
    let tmp_1036 = subgroupShuffleXor(values[14], 4u);
    let tmp_1037 = subgroupShuffleXor(keys[15], 4u);
    let tmp_1038 = subgroupShuffleXor(values[15], 4u);
    let tmp_1039 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_1040 = keys[0] < tmp_1007 || (keys[0] == tmp_1007 && values[0] < tmp_1008);
    if tmp_1039 == tmp_1040 { keys[0] = tmp_1007; values[0] = tmp_1008; }
    let tmp_1041 = keys[1] < tmp_1009 || (keys[1] == tmp_1009 && values[1] < tmp_1010);
    if tmp_1039 == tmp_1041 { keys[1] = tmp_1009; values[1] = tmp_1010; }
    let tmp_1042 = keys[2] < tmp_1011 || (keys[2] == tmp_1011 && values[2] < tmp_1012);
    if tmp_1039 == tmp_1042 { keys[2] = tmp_1011; values[2] = tmp_1012; }
    let tmp_1043 = keys[3] < tmp_1013 || (keys[3] == tmp_1013 && values[3] < tmp_1014);
    if tmp_1039 == tmp_1043 { keys[3] = tmp_1013; values[3] = tmp_1014; }
    let tmp_1044 = keys[4] < tmp_1015 || (keys[4] == tmp_1015 && values[4] < tmp_1016);
    if tmp_1039 == tmp_1044 { keys[4] = tmp_1015; values[4] = tmp_1016; }
    let tmp_1045 = keys[5] < tmp_1017 || (keys[5] == tmp_1017 && values[5] < tmp_1018);
    if tmp_1039 == tmp_1045 { keys[5] = tmp_1017; values[5] = tmp_1018; }
    let tmp_1046 = keys[6] < tmp_1019 || (keys[6] == tmp_1019 && values[6] < tmp_1020);
    if tmp_1039 == tmp_1046 { keys[6] = tmp_1019; values[6] = tmp_1020; }
    let tmp_1047 = keys[7] < tmp_1021 || (keys[7] == tmp_1021 && values[7] < tmp_1022);
    if tmp_1039 == tmp_1047 { keys[7] = tmp_1021; values[7] = tmp_1022; }
    let tmp_1048 = keys[8] < tmp_1023 || (keys[8] == tmp_1023 && values[8] < tmp_1024);
    if tmp_1039 == tmp_1048 { keys[8] = tmp_1023; values[8] = tmp_1024; }
    let tmp_1049 = keys[9] < tmp_1025 || (keys[9] == tmp_1025 && values[9] < tmp_1026);
    if tmp_1039 == tmp_1049 { keys[9] = tmp_1025; values[9] = tmp_1026; }
    let tmp_1050 = keys[10] < tmp_1027 || (keys[10] == tmp_1027 && values[10] < tmp_1028);
    if tmp_1039 == tmp_1050 { keys[10] = tmp_1027; values[10] = tmp_1028; }
    let tmp_1051 = keys[11] < tmp_1029 || (keys[11] == tmp_1029 && values[11] < tmp_1030);
    if tmp_1039 == tmp_1051 { keys[11] = tmp_1029; values[11] = tmp_1030; }
    let tmp_1052 = keys[12] < tmp_1031 || (keys[12] == tmp_1031 && values[12] < tmp_1032);
    if tmp_1039 == tmp_1052 { keys[12] = tmp_1031; values[12] = tmp_1032; }
    let tmp_1053 = keys[13] < tmp_1033 || (keys[13] == tmp_1033 && values[13] < tmp_1034);
    if tmp_1039 == tmp_1053 { keys[13] = tmp_1033; values[13] = tmp_1034; }
    let tmp_1054 = keys[14] < tmp_1035 || (keys[14] == tmp_1035 && values[14] < tmp_1036);
    if tmp_1039 == tmp_1054 { keys[14] = tmp_1035; values[14] = tmp_1036; }
    let tmp_1055 = keys[15] < tmp_1037 || (keys[15] == tmp_1037 && values[15] < tmp_1038);
    if tmp_1039 == tmp_1055 { keys[15] = tmp_1037; values[15] = tmp_1038; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:16) 
    {
    let tmp_1056 = subgroupShuffleXor(keys[0], 2u);
    let tmp_1057 = subgroupShuffleXor(values[0], 2u);
    let tmp_1058 = subgroupShuffleXor(keys[1], 2u);
    let tmp_1059 = subgroupShuffleXor(values[1], 2u);
    let tmp_1060 = subgroupShuffleXor(keys[2], 2u);
    let tmp_1061 = subgroupShuffleXor(values[2], 2u);
    let tmp_1062 = subgroupShuffleXor(keys[3], 2u);
    let tmp_1063 = subgroupShuffleXor(values[3], 2u);
    let tmp_1064 = subgroupShuffleXor(keys[4], 2u);
    let tmp_1065 = subgroupShuffleXor(values[4], 2u);
    let tmp_1066 = subgroupShuffleXor(keys[5], 2u);
    let tmp_1067 = subgroupShuffleXor(values[5], 2u);
    let tmp_1068 = subgroupShuffleXor(keys[6], 2u);
    let tmp_1069 = subgroupShuffleXor(values[6], 2u);
    let tmp_1070 = subgroupShuffleXor(keys[7], 2u);
    let tmp_1071 = subgroupShuffleXor(values[7], 2u);
    let tmp_1072 = subgroupShuffleXor(keys[8], 2u);
    let tmp_1073 = subgroupShuffleXor(values[8], 2u);
    let tmp_1074 = subgroupShuffleXor(keys[9], 2u);
    let tmp_1075 = subgroupShuffleXor(values[9], 2u);
    let tmp_1076 = subgroupShuffleXor(keys[10], 2u);
    let tmp_1077 = subgroupShuffleXor(values[10], 2u);
    let tmp_1078 = subgroupShuffleXor(keys[11], 2u);
    let tmp_1079 = subgroupShuffleXor(values[11], 2u);
    let tmp_1080 = subgroupShuffleXor(keys[12], 2u);
    let tmp_1081 = subgroupShuffleXor(values[12], 2u);
    let tmp_1082 = subgroupShuffleXor(keys[13], 2u);
    let tmp_1083 = subgroupShuffleXor(values[13], 2u);
    let tmp_1084 = subgroupShuffleXor(keys[14], 2u);
    let tmp_1085 = subgroupShuffleXor(values[14], 2u);
    let tmp_1086 = subgroupShuffleXor(keys[15], 2u);
    let tmp_1087 = subgroupShuffleXor(values[15], 2u);
    let tmp_1088 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_1089 = keys[0] < tmp_1056 || (keys[0] == tmp_1056 && values[0] < tmp_1057);
    if tmp_1088 == tmp_1089 { keys[0] = tmp_1056; values[0] = tmp_1057; }
    let tmp_1090 = keys[1] < tmp_1058 || (keys[1] == tmp_1058 && values[1] < tmp_1059);
    if tmp_1088 == tmp_1090 { keys[1] = tmp_1058; values[1] = tmp_1059; }
    let tmp_1091 = keys[2] < tmp_1060 || (keys[2] == tmp_1060 && values[2] < tmp_1061);
    if tmp_1088 == tmp_1091 { keys[2] = tmp_1060; values[2] = tmp_1061; }
    let tmp_1092 = keys[3] < tmp_1062 || (keys[3] == tmp_1062 && values[3] < tmp_1063);
    if tmp_1088 == tmp_1092 { keys[3] = tmp_1062; values[3] = tmp_1063; }
    let tmp_1093 = keys[4] < tmp_1064 || (keys[4] == tmp_1064 && values[4] < tmp_1065);
    if tmp_1088 == tmp_1093 { keys[4] = tmp_1064; values[4] = tmp_1065; }
    let tmp_1094 = keys[5] < tmp_1066 || (keys[5] == tmp_1066 && values[5] < tmp_1067);
    if tmp_1088 == tmp_1094 { keys[5] = tmp_1066; values[5] = tmp_1067; }
    let tmp_1095 = keys[6] < tmp_1068 || (keys[6] == tmp_1068 && values[6] < tmp_1069);
    if tmp_1088 == tmp_1095 { keys[6] = tmp_1068; values[6] = tmp_1069; }
    let tmp_1096 = keys[7] < tmp_1070 || (keys[7] == tmp_1070 && values[7] < tmp_1071);
    if tmp_1088 == tmp_1096 { keys[7] = tmp_1070; values[7] = tmp_1071; }
    let tmp_1097 = keys[8] < tmp_1072 || (keys[8] == tmp_1072 && values[8] < tmp_1073);
    if tmp_1088 == tmp_1097 { keys[8] = tmp_1072; values[8] = tmp_1073; }
    let tmp_1098 = keys[9] < tmp_1074 || (keys[9] == tmp_1074 && values[9] < tmp_1075);
    if tmp_1088 == tmp_1098 { keys[9] = tmp_1074; values[9] = tmp_1075; }
    let tmp_1099 = keys[10] < tmp_1076 || (keys[10] == tmp_1076 && values[10] < tmp_1077);
    if tmp_1088 == tmp_1099 { keys[10] = tmp_1076; values[10] = tmp_1077; }
    let tmp_1100 = keys[11] < tmp_1078 || (keys[11] == tmp_1078 && values[11] < tmp_1079);
    if tmp_1088 == tmp_1100 { keys[11] = tmp_1078; values[11] = tmp_1079; }
    let tmp_1101 = keys[12] < tmp_1080 || (keys[12] == tmp_1080 && values[12] < tmp_1081);
    if tmp_1088 == tmp_1101 { keys[12] = tmp_1080; values[12] = tmp_1081; }
    let tmp_1102 = keys[13] < tmp_1082 || (keys[13] == tmp_1082 && values[13] < tmp_1083);
    if tmp_1088 == tmp_1102 { keys[13] = tmp_1082; values[13] = tmp_1083; }
    let tmp_1103 = keys[14] < tmp_1084 || (keys[14] == tmp_1084 && values[14] < tmp_1085);
    if tmp_1088 == tmp_1103 { keys[14] = tmp_1084; values[14] = tmp_1085; }
    let tmp_1104 = keys[15] < tmp_1086 || (keys[15] == tmp_1086 && values[15] < tmp_1087);
    if tmp_1088 == tmp_1104 { keys[15] = tmp_1086; values[15] = tmp_1087; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:16) 
    {
    let tmp_1105 = subgroupShuffleXor(keys[0], 1u);
    let tmp_1106 = subgroupShuffleXor(values[0], 1u);
    let tmp_1107 = subgroupShuffleXor(keys[1], 1u);
    let tmp_1108 = subgroupShuffleXor(values[1], 1u);
    let tmp_1109 = subgroupShuffleXor(keys[2], 1u);
    let tmp_1110 = subgroupShuffleXor(values[2], 1u);
    let tmp_1111 = subgroupShuffleXor(keys[3], 1u);
    let tmp_1112 = subgroupShuffleXor(values[3], 1u);
    let tmp_1113 = subgroupShuffleXor(keys[4], 1u);
    let tmp_1114 = subgroupShuffleXor(values[4], 1u);
    let tmp_1115 = subgroupShuffleXor(keys[5], 1u);
    let tmp_1116 = subgroupShuffleXor(values[5], 1u);
    let tmp_1117 = subgroupShuffleXor(keys[6], 1u);
    let tmp_1118 = subgroupShuffleXor(values[6], 1u);
    let tmp_1119 = subgroupShuffleXor(keys[7], 1u);
    let tmp_1120 = subgroupShuffleXor(values[7], 1u);
    let tmp_1121 = subgroupShuffleXor(keys[8], 1u);
    let tmp_1122 = subgroupShuffleXor(values[8], 1u);
    let tmp_1123 = subgroupShuffleXor(keys[9], 1u);
    let tmp_1124 = subgroupShuffleXor(values[9], 1u);
    let tmp_1125 = subgroupShuffleXor(keys[10], 1u);
    let tmp_1126 = subgroupShuffleXor(values[10], 1u);
    let tmp_1127 = subgroupShuffleXor(keys[11], 1u);
    let tmp_1128 = subgroupShuffleXor(values[11], 1u);
    let tmp_1129 = subgroupShuffleXor(keys[12], 1u);
    let tmp_1130 = subgroupShuffleXor(values[12], 1u);
    let tmp_1131 = subgroupShuffleXor(keys[13], 1u);
    let tmp_1132 = subgroupShuffleXor(values[13], 1u);
    let tmp_1133 = subgroupShuffleXor(keys[14], 1u);
    let tmp_1134 = subgroupShuffleXor(values[14], 1u);
    let tmp_1135 = subgroupShuffleXor(keys[15], 1u);
    let tmp_1136 = subgroupShuffleXor(values[15], 1u);
    let tmp_1137 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_1138 = keys[0] < tmp_1105 || (keys[0] == tmp_1105 && values[0] < tmp_1106);
    if tmp_1137 == tmp_1138 { keys[0] = tmp_1105; values[0] = tmp_1106; }
    let tmp_1139 = keys[1] < tmp_1107 || (keys[1] == tmp_1107 && values[1] < tmp_1108);
    if tmp_1137 == tmp_1139 { keys[1] = tmp_1107; values[1] = tmp_1108; }
    let tmp_1140 = keys[2] < tmp_1109 || (keys[2] == tmp_1109 && values[2] < tmp_1110);
    if tmp_1137 == tmp_1140 { keys[2] = tmp_1109; values[2] = tmp_1110; }
    let tmp_1141 = keys[3] < tmp_1111 || (keys[3] == tmp_1111 && values[3] < tmp_1112);
    if tmp_1137 == tmp_1141 { keys[3] = tmp_1111; values[3] = tmp_1112; }
    let tmp_1142 = keys[4] < tmp_1113 || (keys[4] == tmp_1113 && values[4] < tmp_1114);
    if tmp_1137 == tmp_1142 { keys[4] = tmp_1113; values[4] = tmp_1114; }
    let tmp_1143 = keys[5] < tmp_1115 || (keys[5] == tmp_1115 && values[5] < tmp_1116);
    if tmp_1137 == tmp_1143 { keys[5] = tmp_1115; values[5] = tmp_1116; }
    let tmp_1144 = keys[6] < tmp_1117 || (keys[6] == tmp_1117 && values[6] < tmp_1118);
    if tmp_1137 == tmp_1144 { keys[6] = tmp_1117; values[6] = tmp_1118; }
    let tmp_1145 = keys[7] < tmp_1119 || (keys[7] == tmp_1119 && values[7] < tmp_1120);
    if tmp_1137 == tmp_1145 { keys[7] = tmp_1119; values[7] = tmp_1120; }
    let tmp_1146 = keys[8] < tmp_1121 || (keys[8] == tmp_1121 && values[8] < tmp_1122);
    if tmp_1137 == tmp_1146 { keys[8] = tmp_1121; values[8] = tmp_1122; }
    let tmp_1147 = keys[9] < tmp_1123 || (keys[9] == tmp_1123 && values[9] < tmp_1124);
    if tmp_1137 == tmp_1147 { keys[9] = tmp_1123; values[9] = tmp_1124; }
    let tmp_1148 = keys[10] < tmp_1125 || (keys[10] == tmp_1125 && values[10] < tmp_1126);
    if tmp_1137 == tmp_1148 { keys[10] = tmp_1125; values[10] = tmp_1126; }
    let tmp_1149 = keys[11] < tmp_1127 || (keys[11] == tmp_1127 && values[11] < tmp_1128);
    if tmp_1137 == tmp_1149 { keys[11] = tmp_1127; values[11] = tmp_1128; }
    let tmp_1150 = keys[12] < tmp_1129 || (keys[12] == tmp_1129 && values[12] < tmp_1130);
    if tmp_1137 == tmp_1150 { keys[12] = tmp_1129; values[12] = tmp_1130; }
    let tmp_1151 = keys[13] < tmp_1131 || (keys[13] == tmp_1131 && values[13] < tmp_1132);
    if tmp_1137 == tmp_1151 { keys[13] = tmp_1131; values[13] = tmp_1132; }
    let tmp_1152 = keys[14] < tmp_1133 || (keys[14] == tmp_1133 && values[14] < tmp_1134);
    if tmp_1137 == tmp_1152 { keys[14] = tmp_1133; values[14] = tmp_1134; }
    let tmp_1153 = keys[15] < tmp_1135 || (keys[15] == tmp_1135 && values[15] < tmp_1136);
    if tmp_1137 == tmp_1153 { keys[15] = tmp_1135; values[15] = tmp_1136; }
    }
    // exch_local(8,16) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_1154 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_1154;let tmp_1155 = values[0]; values[0] = values[8]; values[8] = tmp_1155; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_1156 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_1156;let tmp_1157 = values[1]; values[1] = values[9]; values[9] = tmp_1157; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_1158 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_1158;let tmp_1159 = values[2]; values[2] = values[10]; values[10] = tmp_1159; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_1160 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_1160;let tmp_1161 = values[3]; values[3] = values[11]; values[11] = tmp_1161; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_1162 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_1162;let tmp_1163 = values[4]; values[4] = values[12]; values[12] = tmp_1163; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_1164 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_1164;let tmp_1165 = values[5]; values[5] = values[13]; values[13] = tmp_1165; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_1166 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_1166;let tmp_1167 = values[6]; values[6] = values[14]; values[14] = tmp_1167; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_1168 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_1168;let tmp_1169 = values[7]; values[7] = values[15]; values[15] = tmp_1169; }
    }
    // exch_local(4,16) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_1170 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_1170;let tmp_1171 = values[0]; values[0] = values[4]; values[4] = tmp_1171; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_1172 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_1172;let tmp_1173 = values[1]; values[1] = values[5]; values[5] = tmp_1173; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_1174 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_1174;let tmp_1175 = values[2]; values[2] = values[6]; values[6] = tmp_1175; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_1176 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_1176;let tmp_1177 = values[3]; values[3] = values[7]; values[7] = tmp_1177; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_1178 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_1178;let tmp_1179 = values[8]; values[8] = values[12]; values[12] = tmp_1179; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_1180 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_1180;let tmp_1181 = values[9]; values[9] = values[13]; values[13] = tmp_1181; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_1182 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_1182;let tmp_1183 = values[10]; values[10] = values[14]; values[14] = tmp_1183; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_1184 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_1184;let tmp_1185 = values[11]; values[11] = values[15]; values[15] = tmp_1185; }
    }
    // exch_local(2,16) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_1186 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_1186;let tmp_1187 = values[0]; values[0] = values[2]; values[2] = tmp_1187; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_1188 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_1188;let tmp_1189 = values[1]; values[1] = values[3]; values[3] = tmp_1189; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_1190 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_1190;let tmp_1191 = values[4]; values[4] = values[6]; values[6] = tmp_1191; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_1192 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_1192;let tmp_1193 = values[5]; values[5] = values[7]; values[7] = tmp_1193; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_1194 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_1194;let tmp_1195 = values[8]; values[8] = values[10]; values[10] = tmp_1195; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_1196 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_1196;let tmp_1197 = values[9]; values[9] = values[11]; values[11] = tmp_1197; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_1198 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_1198;let tmp_1199 = values[12]; values[12] = values[14]; values[14] = tmp_1199; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_1200 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_1200;let tmp_1201 = values[13]; values[13] = values[15]; values[15] = tmp_1201; }
    }
    // exch_local(1,16) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_1202 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_1202;let tmp_1203 = values[0]; values[0] = values[1]; values[1] = tmp_1203; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_1204 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_1204;let tmp_1205 = values[2]; values[2] = values[3]; values[3] = tmp_1205; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_1206 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_1206;let tmp_1207 = values[4]; values[4] = values[5]; values[5] = tmp_1207; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_1208 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_1208;let tmp_1209 = values[6]; values[6] = values[7]; values[7] = tmp_1209; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_1210 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_1210;let tmp_1211 = values[8]; values[8] = values[9]; values[9] = tmp_1211; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_1212 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_1212;let tmp_1213 = values[10]; values[10] = values[11]; values[11] = tmp_1213; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_1214 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_1214;let tmp_1215 = values[12]; values[12] = values[13]; values[13] = tmp_1215; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_1216 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_1216;let tmp_1217 = values[14]; values[14] = values[15]; values[15] = tmp_1217; }
    }
    // exch_intxn(tmask:63,swbit:5,wpt:16)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_1218 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_1219 = seg_base + (local_tid ^ 63u); let tmp_1220 = smem_keys[tmp_1219 * WPT + 15u]; let tmp_1221 = smem_vals[tmp_1219 * WPT + 15u]; let tmp_1222 = keys[0] < tmp_1220 || (keys[0] == tmp_1220 && values[0] < tmp_1221); if tmp_1218 == tmp_1222 { keys[0] = tmp_1220; values[0] = tmp_1221; } let tmp_1223 = smem_keys[tmp_1219 * WPT + 14u]; let tmp_1224 = smem_vals[tmp_1219 * WPT + 14u]; let tmp_1225 = keys[1] < tmp_1223 || (keys[1] == tmp_1223 && values[1] < tmp_1224); if tmp_1218 == tmp_1225 { keys[1] = tmp_1223; values[1] = tmp_1224; } let tmp_1226 = smem_keys[tmp_1219 * WPT + 13u]; let tmp_1227 = smem_vals[tmp_1219 * WPT + 13u]; let tmp_1228 = keys[2] < tmp_1226 || (keys[2] == tmp_1226 && values[2] < tmp_1227); if tmp_1218 == tmp_1228 { keys[2] = tmp_1226; values[2] = tmp_1227; } let tmp_1229 = smem_keys[tmp_1219 * WPT + 12u]; let tmp_1230 = smem_vals[tmp_1219 * WPT + 12u]; let tmp_1231 = keys[3] < tmp_1229 || (keys[3] == tmp_1229 && values[3] < tmp_1230); if tmp_1218 == tmp_1231 { keys[3] = tmp_1229; values[3] = tmp_1230; } let tmp_1232 = smem_keys[tmp_1219 * WPT + 11u]; let tmp_1233 = smem_vals[tmp_1219 * WPT + 11u]; let tmp_1234 = keys[4] < tmp_1232 || (keys[4] == tmp_1232 && values[4] < tmp_1233); if tmp_1218 == tmp_1234 { keys[4] = tmp_1232; values[4] = tmp_1233; } let tmp_1235 = smem_keys[tmp_1219 * WPT + 10u]; let tmp_1236 = smem_vals[tmp_1219 * WPT + 10u]; let tmp_1237 = keys[5] < tmp_1235 || (keys[5] == tmp_1235 && values[5] < tmp_1236); if tmp_1218 == tmp_1237 { keys[5] = tmp_1235; values[5] = tmp_1236; } let tmp_1238 = smem_keys[tmp_1219 * WPT + 9u]; let tmp_1239 = smem_vals[tmp_1219 * WPT + 9u]; let tmp_1240 = keys[6] < tmp_1238 || (keys[6] == tmp_1238 && values[6] < tmp_1239); if tmp_1218 == tmp_1240 { keys[6] = tmp_1238; values[6] = tmp_1239; } let tmp_1241 = smem_keys[tmp_1219 * WPT + 8u]; let tmp_1242 = smem_vals[tmp_1219 * WPT + 8u]; let tmp_1243 = keys[7] < tmp_1241 || (keys[7] == tmp_1241 && values[7] < tmp_1242); if tmp_1218 == tmp_1243 { keys[7] = tmp_1241; values[7] = tmp_1242; } let tmp_1244 = smem_keys[tmp_1219 * WPT + 7u]; let tmp_1245 = smem_vals[tmp_1219 * WPT + 7u]; let tmp_1246 = keys[8] < tmp_1244 || (keys[8] == tmp_1244 && values[8] < tmp_1245); if tmp_1218 == tmp_1246 { keys[8] = tmp_1244; values[8] = tmp_1245; } let tmp_1247 = smem_keys[tmp_1219 * WPT + 6u]; let tmp_1248 = smem_vals[tmp_1219 * WPT + 6u]; let tmp_1249 = keys[9] < tmp_1247 || (keys[9] == tmp_1247 && values[9] < tmp_1248); if tmp_1218 == tmp_1249 { keys[9] = tmp_1247; values[9] = tmp_1248; } let tmp_1250 = smem_keys[tmp_1219 * WPT + 5u]; let tmp_1251 = smem_vals[tmp_1219 * WPT + 5u]; let tmp_1252 = keys[10] < tmp_1250 || (keys[10] == tmp_1250 && values[10] < tmp_1251); if tmp_1218 == tmp_1252 { keys[10] = tmp_1250; values[10] = tmp_1251; } let tmp_1253 = smem_keys[tmp_1219 * WPT + 4u]; let tmp_1254 = smem_vals[tmp_1219 * WPT + 4u]; let tmp_1255 = keys[11] < tmp_1253 || (keys[11] == tmp_1253 && values[11] < tmp_1254); if tmp_1218 == tmp_1255 { keys[11] = tmp_1253; values[11] = tmp_1254; } let tmp_1256 = smem_keys[tmp_1219 * WPT + 3u]; let tmp_1257 = smem_vals[tmp_1219 * WPT + 3u]; let tmp_1258 = keys[12] < tmp_1256 || (keys[12] == tmp_1256 && values[12] < tmp_1257); if tmp_1218 == tmp_1258 { keys[12] = tmp_1256; values[12] = tmp_1257; } let tmp_1259 = smem_keys[tmp_1219 * WPT + 2u]; let tmp_1260 = smem_vals[tmp_1219 * WPT + 2u]; let tmp_1261 = keys[13] < tmp_1259 || (keys[13] == tmp_1259 && values[13] < tmp_1260); if tmp_1218 == tmp_1261 { keys[13] = tmp_1259; values[13] = tmp_1260; } let tmp_1262 = smem_keys[tmp_1219 * WPT + 1u]; let tmp_1263 = smem_vals[tmp_1219 * WPT + 1u]; let tmp_1264 = keys[14] < tmp_1262 || (keys[14] == tmp_1262 && values[14] < tmp_1263); if tmp_1218 == tmp_1264 { keys[14] = tmp_1262; values[14] = tmp_1263; } let tmp_1265 = smem_keys[tmp_1219 * WPT + 0u]; let tmp_1266 = smem_vals[tmp_1219 * WPT + 0u]; let tmp_1267 = keys[15] < tmp_1265 || (keys[15] == tmp_1265 && values[15] < tmp_1266); if tmp_1218 == tmp_1267 { keys[15] = tmp_1265; values[15] = tmp_1266; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:16) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_1268 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_1269 = seg_base + (local_tid ^ 16u); let tmp_1270 = smem_keys[tmp_1269 * WPT + 0u]; let tmp_1271 = smem_vals[tmp_1269 * WPT + 0u]; let tmp_1272 = keys[0] < tmp_1270 || (keys[0] == tmp_1270 && values[0] < tmp_1271); if tmp_1268 == tmp_1272 { keys[0] = tmp_1270; values[0] = tmp_1271; } let tmp_1273 = smem_keys[tmp_1269 * WPT + 1u]; let tmp_1274 = smem_vals[tmp_1269 * WPT + 1u]; let tmp_1275 = keys[1] < tmp_1273 || (keys[1] == tmp_1273 && values[1] < tmp_1274); if tmp_1268 == tmp_1275 { keys[1] = tmp_1273; values[1] = tmp_1274; } let tmp_1276 = smem_keys[tmp_1269 * WPT + 2u]; let tmp_1277 = smem_vals[tmp_1269 * WPT + 2u]; let tmp_1278 = keys[2] < tmp_1276 || (keys[2] == tmp_1276 && values[2] < tmp_1277); if tmp_1268 == tmp_1278 { keys[2] = tmp_1276; values[2] = tmp_1277; } let tmp_1279 = smem_keys[tmp_1269 * WPT + 3u]; let tmp_1280 = smem_vals[tmp_1269 * WPT + 3u]; let tmp_1281 = keys[3] < tmp_1279 || (keys[3] == tmp_1279 && values[3] < tmp_1280); if tmp_1268 == tmp_1281 { keys[3] = tmp_1279; values[3] = tmp_1280; } let tmp_1282 = smem_keys[tmp_1269 * WPT + 4u]; let tmp_1283 = smem_vals[tmp_1269 * WPT + 4u]; let tmp_1284 = keys[4] < tmp_1282 || (keys[4] == tmp_1282 && values[4] < tmp_1283); if tmp_1268 == tmp_1284 { keys[4] = tmp_1282; values[4] = tmp_1283; } let tmp_1285 = smem_keys[tmp_1269 * WPT + 5u]; let tmp_1286 = smem_vals[tmp_1269 * WPT + 5u]; let tmp_1287 = keys[5] < tmp_1285 || (keys[5] == tmp_1285 && values[5] < tmp_1286); if tmp_1268 == tmp_1287 { keys[5] = tmp_1285; values[5] = tmp_1286; } let tmp_1288 = smem_keys[tmp_1269 * WPT + 6u]; let tmp_1289 = smem_vals[tmp_1269 * WPT + 6u]; let tmp_1290 = keys[6] < tmp_1288 || (keys[6] == tmp_1288 && values[6] < tmp_1289); if tmp_1268 == tmp_1290 { keys[6] = tmp_1288; values[6] = tmp_1289; } let tmp_1291 = smem_keys[tmp_1269 * WPT + 7u]; let tmp_1292 = smem_vals[tmp_1269 * WPT + 7u]; let tmp_1293 = keys[7] < tmp_1291 || (keys[7] == tmp_1291 && values[7] < tmp_1292); if tmp_1268 == tmp_1293 { keys[7] = tmp_1291; values[7] = tmp_1292; } let tmp_1294 = smem_keys[tmp_1269 * WPT + 8u]; let tmp_1295 = smem_vals[tmp_1269 * WPT + 8u]; let tmp_1296 = keys[8] < tmp_1294 || (keys[8] == tmp_1294 && values[8] < tmp_1295); if tmp_1268 == tmp_1296 { keys[8] = tmp_1294; values[8] = tmp_1295; } let tmp_1297 = smem_keys[tmp_1269 * WPT + 9u]; let tmp_1298 = smem_vals[tmp_1269 * WPT + 9u]; let tmp_1299 = keys[9] < tmp_1297 || (keys[9] == tmp_1297 && values[9] < tmp_1298); if tmp_1268 == tmp_1299 { keys[9] = tmp_1297; values[9] = tmp_1298; } let tmp_1300 = smem_keys[tmp_1269 * WPT + 10u]; let tmp_1301 = smem_vals[tmp_1269 * WPT + 10u]; let tmp_1302 = keys[10] < tmp_1300 || (keys[10] == tmp_1300 && values[10] < tmp_1301); if tmp_1268 == tmp_1302 { keys[10] = tmp_1300; values[10] = tmp_1301; } let tmp_1303 = smem_keys[tmp_1269 * WPT + 11u]; let tmp_1304 = smem_vals[tmp_1269 * WPT + 11u]; let tmp_1305 = keys[11] < tmp_1303 || (keys[11] == tmp_1303 && values[11] < tmp_1304); if tmp_1268 == tmp_1305 { keys[11] = tmp_1303; values[11] = tmp_1304; } let tmp_1306 = smem_keys[tmp_1269 * WPT + 12u]; let tmp_1307 = smem_vals[tmp_1269 * WPT + 12u]; let tmp_1308 = keys[12] < tmp_1306 || (keys[12] == tmp_1306 && values[12] < tmp_1307); if tmp_1268 == tmp_1308 { keys[12] = tmp_1306; values[12] = tmp_1307; } let tmp_1309 = smem_keys[tmp_1269 * WPT + 13u]; let tmp_1310 = smem_vals[tmp_1269 * WPT + 13u]; let tmp_1311 = keys[13] < tmp_1309 || (keys[13] == tmp_1309 && values[13] < tmp_1310); if tmp_1268 == tmp_1311 { keys[13] = tmp_1309; values[13] = tmp_1310; } let tmp_1312 = smem_keys[tmp_1269 * WPT + 14u]; let tmp_1313 = smem_vals[tmp_1269 * WPT + 14u]; let tmp_1314 = keys[14] < tmp_1312 || (keys[14] == tmp_1312 && values[14] < tmp_1313); if tmp_1268 == tmp_1314 { keys[14] = tmp_1312; values[14] = tmp_1313; } let tmp_1315 = smem_keys[tmp_1269 * WPT + 15u]; let tmp_1316 = smem_vals[tmp_1269 * WPT + 15u]; let tmp_1317 = keys[15] < tmp_1315 || (keys[15] == tmp_1315 && values[15] < tmp_1316); if tmp_1268 == tmp_1317 { keys[15] = tmp_1315; values[15] = tmp_1316; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:16) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; smem_keys[tid_g * WPT + 8u] = keys[8]; smem_vals[tid_g * WPT + 8u] = values[8]; smem_keys[tid_g * WPT + 9u] = keys[9]; smem_vals[tid_g * WPT + 9u] = values[9]; smem_keys[tid_g * WPT + 10u] = keys[10]; smem_vals[tid_g * WPT + 10u] = values[10]; smem_keys[tid_g * WPT + 11u] = keys[11]; smem_vals[tid_g * WPT + 11u] = values[11]; smem_keys[tid_g * WPT + 12u] = keys[12]; smem_vals[tid_g * WPT + 12u] = values[12]; smem_keys[tid_g * WPT + 13u] = keys[13]; smem_vals[tid_g * WPT + 13u] = values[13]; smem_keys[tid_g * WPT + 14u] = keys[14]; smem_vals[tid_g * WPT + 14u] = values[14]; smem_keys[tid_g * WPT + 15u] = keys[15]; smem_vals[tid_g * WPT + 15u] = values[15]; workgroupBarrier(); let tmp_1318 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_1319 = seg_base + (local_tid ^ 8u); let tmp_1320 = smem_keys[tmp_1319 * WPT + 0u]; let tmp_1321 = smem_vals[tmp_1319 * WPT + 0u]; let tmp_1322 = keys[0] < tmp_1320 || (keys[0] == tmp_1320 && values[0] < tmp_1321); if tmp_1318 == tmp_1322 { keys[0] = tmp_1320; values[0] = tmp_1321; } let tmp_1323 = smem_keys[tmp_1319 * WPT + 1u]; let tmp_1324 = smem_vals[tmp_1319 * WPT + 1u]; let tmp_1325 = keys[1] < tmp_1323 || (keys[1] == tmp_1323 && values[1] < tmp_1324); if tmp_1318 == tmp_1325 { keys[1] = tmp_1323; values[1] = tmp_1324; } let tmp_1326 = smem_keys[tmp_1319 * WPT + 2u]; let tmp_1327 = smem_vals[tmp_1319 * WPT + 2u]; let tmp_1328 = keys[2] < tmp_1326 || (keys[2] == tmp_1326 && values[2] < tmp_1327); if tmp_1318 == tmp_1328 { keys[2] = tmp_1326; values[2] = tmp_1327; } let tmp_1329 = smem_keys[tmp_1319 * WPT + 3u]; let tmp_1330 = smem_vals[tmp_1319 * WPT + 3u]; let tmp_1331 = keys[3] < tmp_1329 || (keys[3] == tmp_1329 && values[3] < tmp_1330); if tmp_1318 == tmp_1331 { keys[3] = tmp_1329; values[3] = tmp_1330; } let tmp_1332 = smem_keys[tmp_1319 * WPT + 4u]; let tmp_1333 = smem_vals[tmp_1319 * WPT + 4u]; let tmp_1334 = keys[4] < tmp_1332 || (keys[4] == tmp_1332 && values[4] < tmp_1333); if tmp_1318 == tmp_1334 { keys[4] = tmp_1332; values[4] = tmp_1333; } let tmp_1335 = smem_keys[tmp_1319 * WPT + 5u]; let tmp_1336 = smem_vals[tmp_1319 * WPT + 5u]; let tmp_1337 = keys[5] < tmp_1335 || (keys[5] == tmp_1335 && values[5] < tmp_1336); if tmp_1318 == tmp_1337 { keys[5] = tmp_1335; values[5] = tmp_1336; } let tmp_1338 = smem_keys[tmp_1319 * WPT + 6u]; let tmp_1339 = smem_vals[tmp_1319 * WPT + 6u]; let tmp_1340 = keys[6] < tmp_1338 || (keys[6] == tmp_1338 && values[6] < tmp_1339); if tmp_1318 == tmp_1340 { keys[6] = tmp_1338; values[6] = tmp_1339; } let tmp_1341 = smem_keys[tmp_1319 * WPT + 7u]; let tmp_1342 = smem_vals[tmp_1319 * WPT + 7u]; let tmp_1343 = keys[7] < tmp_1341 || (keys[7] == tmp_1341 && values[7] < tmp_1342); if tmp_1318 == tmp_1343 { keys[7] = tmp_1341; values[7] = tmp_1342; } let tmp_1344 = smem_keys[tmp_1319 * WPT + 8u]; let tmp_1345 = smem_vals[tmp_1319 * WPT + 8u]; let tmp_1346 = keys[8] < tmp_1344 || (keys[8] == tmp_1344 && values[8] < tmp_1345); if tmp_1318 == tmp_1346 { keys[8] = tmp_1344; values[8] = tmp_1345; } let tmp_1347 = smem_keys[tmp_1319 * WPT + 9u]; let tmp_1348 = smem_vals[tmp_1319 * WPT + 9u]; let tmp_1349 = keys[9] < tmp_1347 || (keys[9] == tmp_1347 && values[9] < tmp_1348); if tmp_1318 == tmp_1349 { keys[9] = tmp_1347; values[9] = tmp_1348; } let tmp_1350 = smem_keys[tmp_1319 * WPT + 10u]; let tmp_1351 = smem_vals[tmp_1319 * WPT + 10u]; let tmp_1352 = keys[10] < tmp_1350 || (keys[10] == tmp_1350 && values[10] < tmp_1351); if tmp_1318 == tmp_1352 { keys[10] = tmp_1350; values[10] = tmp_1351; } let tmp_1353 = smem_keys[tmp_1319 * WPT + 11u]; let tmp_1354 = smem_vals[tmp_1319 * WPT + 11u]; let tmp_1355 = keys[11] < tmp_1353 || (keys[11] == tmp_1353 && values[11] < tmp_1354); if tmp_1318 == tmp_1355 { keys[11] = tmp_1353; values[11] = tmp_1354; } let tmp_1356 = smem_keys[tmp_1319 * WPT + 12u]; let tmp_1357 = smem_vals[tmp_1319 * WPT + 12u]; let tmp_1358 = keys[12] < tmp_1356 || (keys[12] == tmp_1356 && values[12] < tmp_1357); if tmp_1318 == tmp_1358 { keys[12] = tmp_1356; values[12] = tmp_1357; } let tmp_1359 = smem_keys[tmp_1319 * WPT + 13u]; let tmp_1360 = smem_vals[tmp_1319 * WPT + 13u]; let tmp_1361 = keys[13] < tmp_1359 || (keys[13] == tmp_1359 && values[13] < tmp_1360); if tmp_1318 == tmp_1361 { keys[13] = tmp_1359; values[13] = tmp_1360; } let tmp_1362 = smem_keys[tmp_1319 * WPT + 14u]; let tmp_1363 = smem_vals[tmp_1319 * WPT + 14u]; let tmp_1364 = keys[14] < tmp_1362 || (keys[14] == tmp_1362 && values[14] < tmp_1363); if tmp_1318 == tmp_1364 { keys[14] = tmp_1362; values[14] = tmp_1363; } let tmp_1365 = smem_keys[tmp_1319 * WPT + 15u]; let tmp_1366 = smem_vals[tmp_1319 * WPT + 15u]; let tmp_1367 = keys[15] < tmp_1365 || (keys[15] == tmp_1365 && values[15] < tmp_1366); if tmp_1318 == tmp_1367 { keys[15] = tmp_1365; values[15] = tmp_1366; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:16) 
    {
    let tmp_1368 = subgroupShuffleXor(keys[0], 4u);
    let tmp_1369 = subgroupShuffleXor(values[0], 4u);
    let tmp_1370 = subgroupShuffleXor(keys[1], 4u);
    let tmp_1371 = subgroupShuffleXor(values[1], 4u);
    let tmp_1372 = subgroupShuffleXor(keys[2], 4u);
    let tmp_1373 = subgroupShuffleXor(values[2], 4u);
    let tmp_1374 = subgroupShuffleXor(keys[3], 4u);
    let tmp_1375 = subgroupShuffleXor(values[3], 4u);
    let tmp_1376 = subgroupShuffleXor(keys[4], 4u);
    let tmp_1377 = subgroupShuffleXor(values[4], 4u);
    let tmp_1378 = subgroupShuffleXor(keys[5], 4u);
    let tmp_1379 = subgroupShuffleXor(values[5], 4u);
    let tmp_1380 = subgroupShuffleXor(keys[6], 4u);
    let tmp_1381 = subgroupShuffleXor(values[6], 4u);
    let tmp_1382 = subgroupShuffleXor(keys[7], 4u);
    let tmp_1383 = subgroupShuffleXor(values[7], 4u);
    let tmp_1384 = subgroupShuffleXor(keys[8], 4u);
    let tmp_1385 = subgroupShuffleXor(values[8], 4u);
    let tmp_1386 = subgroupShuffleXor(keys[9], 4u);
    let tmp_1387 = subgroupShuffleXor(values[9], 4u);
    let tmp_1388 = subgroupShuffleXor(keys[10], 4u);
    let tmp_1389 = subgroupShuffleXor(values[10], 4u);
    let tmp_1390 = subgroupShuffleXor(keys[11], 4u);
    let tmp_1391 = subgroupShuffleXor(values[11], 4u);
    let tmp_1392 = subgroupShuffleXor(keys[12], 4u);
    let tmp_1393 = subgroupShuffleXor(values[12], 4u);
    let tmp_1394 = subgroupShuffleXor(keys[13], 4u);
    let tmp_1395 = subgroupShuffleXor(values[13], 4u);
    let tmp_1396 = subgroupShuffleXor(keys[14], 4u);
    let tmp_1397 = subgroupShuffleXor(values[14], 4u);
    let tmp_1398 = subgroupShuffleXor(keys[15], 4u);
    let tmp_1399 = subgroupShuffleXor(values[15], 4u);
    let tmp_1400 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_1401 = keys[0] < tmp_1368 || (keys[0] == tmp_1368 && values[0] < tmp_1369);
    if tmp_1400 == tmp_1401 { keys[0] = tmp_1368; values[0] = tmp_1369; }
    let tmp_1402 = keys[1] < tmp_1370 || (keys[1] == tmp_1370 && values[1] < tmp_1371);
    if tmp_1400 == tmp_1402 { keys[1] = tmp_1370; values[1] = tmp_1371; }
    let tmp_1403 = keys[2] < tmp_1372 || (keys[2] == tmp_1372 && values[2] < tmp_1373);
    if tmp_1400 == tmp_1403 { keys[2] = tmp_1372; values[2] = tmp_1373; }
    let tmp_1404 = keys[3] < tmp_1374 || (keys[3] == tmp_1374 && values[3] < tmp_1375);
    if tmp_1400 == tmp_1404 { keys[3] = tmp_1374; values[3] = tmp_1375; }
    let tmp_1405 = keys[4] < tmp_1376 || (keys[4] == tmp_1376 && values[4] < tmp_1377);
    if tmp_1400 == tmp_1405 { keys[4] = tmp_1376; values[4] = tmp_1377; }
    let tmp_1406 = keys[5] < tmp_1378 || (keys[5] == tmp_1378 && values[5] < tmp_1379);
    if tmp_1400 == tmp_1406 { keys[5] = tmp_1378; values[5] = tmp_1379; }
    let tmp_1407 = keys[6] < tmp_1380 || (keys[6] == tmp_1380 && values[6] < tmp_1381);
    if tmp_1400 == tmp_1407 { keys[6] = tmp_1380; values[6] = tmp_1381; }
    let tmp_1408 = keys[7] < tmp_1382 || (keys[7] == tmp_1382 && values[7] < tmp_1383);
    if tmp_1400 == tmp_1408 { keys[7] = tmp_1382; values[7] = tmp_1383; }
    let tmp_1409 = keys[8] < tmp_1384 || (keys[8] == tmp_1384 && values[8] < tmp_1385);
    if tmp_1400 == tmp_1409 { keys[8] = tmp_1384; values[8] = tmp_1385; }
    let tmp_1410 = keys[9] < tmp_1386 || (keys[9] == tmp_1386 && values[9] < tmp_1387);
    if tmp_1400 == tmp_1410 { keys[9] = tmp_1386; values[9] = tmp_1387; }
    let tmp_1411 = keys[10] < tmp_1388 || (keys[10] == tmp_1388 && values[10] < tmp_1389);
    if tmp_1400 == tmp_1411 { keys[10] = tmp_1388; values[10] = tmp_1389; }
    let tmp_1412 = keys[11] < tmp_1390 || (keys[11] == tmp_1390 && values[11] < tmp_1391);
    if tmp_1400 == tmp_1412 { keys[11] = tmp_1390; values[11] = tmp_1391; }
    let tmp_1413 = keys[12] < tmp_1392 || (keys[12] == tmp_1392 && values[12] < tmp_1393);
    if tmp_1400 == tmp_1413 { keys[12] = tmp_1392; values[12] = tmp_1393; }
    let tmp_1414 = keys[13] < tmp_1394 || (keys[13] == tmp_1394 && values[13] < tmp_1395);
    if tmp_1400 == tmp_1414 { keys[13] = tmp_1394; values[13] = tmp_1395; }
    let tmp_1415 = keys[14] < tmp_1396 || (keys[14] == tmp_1396 && values[14] < tmp_1397);
    if tmp_1400 == tmp_1415 { keys[14] = tmp_1396; values[14] = tmp_1397; }
    let tmp_1416 = keys[15] < tmp_1398 || (keys[15] == tmp_1398 && values[15] < tmp_1399);
    if tmp_1400 == tmp_1416 { keys[15] = tmp_1398; values[15] = tmp_1399; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:16) 
    {
    let tmp_1417 = subgroupShuffleXor(keys[0], 2u);
    let tmp_1418 = subgroupShuffleXor(values[0], 2u);
    let tmp_1419 = subgroupShuffleXor(keys[1], 2u);
    let tmp_1420 = subgroupShuffleXor(values[1], 2u);
    let tmp_1421 = subgroupShuffleXor(keys[2], 2u);
    let tmp_1422 = subgroupShuffleXor(values[2], 2u);
    let tmp_1423 = subgroupShuffleXor(keys[3], 2u);
    let tmp_1424 = subgroupShuffleXor(values[3], 2u);
    let tmp_1425 = subgroupShuffleXor(keys[4], 2u);
    let tmp_1426 = subgroupShuffleXor(values[4], 2u);
    let tmp_1427 = subgroupShuffleXor(keys[5], 2u);
    let tmp_1428 = subgroupShuffleXor(values[5], 2u);
    let tmp_1429 = subgroupShuffleXor(keys[6], 2u);
    let tmp_1430 = subgroupShuffleXor(values[6], 2u);
    let tmp_1431 = subgroupShuffleXor(keys[7], 2u);
    let tmp_1432 = subgroupShuffleXor(values[7], 2u);
    let tmp_1433 = subgroupShuffleXor(keys[8], 2u);
    let tmp_1434 = subgroupShuffleXor(values[8], 2u);
    let tmp_1435 = subgroupShuffleXor(keys[9], 2u);
    let tmp_1436 = subgroupShuffleXor(values[9], 2u);
    let tmp_1437 = subgroupShuffleXor(keys[10], 2u);
    let tmp_1438 = subgroupShuffleXor(values[10], 2u);
    let tmp_1439 = subgroupShuffleXor(keys[11], 2u);
    let tmp_1440 = subgroupShuffleXor(values[11], 2u);
    let tmp_1441 = subgroupShuffleXor(keys[12], 2u);
    let tmp_1442 = subgroupShuffleXor(values[12], 2u);
    let tmp_1443 = subgroupShuffleXor(keys[13], 2u);
    let tmp_1444 = subgroupShuffleXor(values[13], 2u);
    let tmp_1445 = subgroupShuffleXor(keys[14], 2u);
    let tmp_1446 = subgroupShuffleXor(values[14], 2u);
    let tmp_1447 = subgroupShuffleXor(keys[15], 2u);
    let tmp_1448 = subgroupShuffleXor(values[15], 2u);
    let tmp_1449 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_1450 = keys[0] < tmp_1417 || (keys[0] == tmp_1417 && values[0] < tmp_1418);
    if tmp_1449 == tmp_1450 { keys[0] = tmp_1417; values[0] = tmp_1418; }
    let tmp_1451 = keys[1] < tmp_1419 || (keys[1] == tmp_1419 && values[1] < tmp_1420);
    if tmp_1449 == tmp_1451 { keys[1] = tmp_1419; values[1] = tmp_1420; }
    let tmp_1452 = keys[2] < tmp_1421 || (keys[2] == tmp_1421 && values[2] < tmp_1422);
    if tmp_1449 == tmp_1452 { keys[2] = tmp_1421; values[2] = tmp_1422; }
    let tmp_1453 = keys[3] < tmp_1423 || (keys[3] == tmp_1423 && values[3] < tmp_1424);
    if tmp_1449 == tmp_1453 { keys[3] = tmp_1423; values[3] = tmp_1424; }
    let tmp_1454 = keys[4] < tmp_1425 || (keys[4] == tmp_1425 && values[4] < tmp_1426);
    if tmp_1449 == tmp_1454 { keys[4] = tmp_1425; values[4] = tmp_1426; }
    let tmp_1455 = keys[5] < tmp_1427 || (keys[5] == tmp_1427 && values[5] < tmp_1428);
    if tmp_1449 == tmp_1455 { keys[5] = tmp_1427; values[5] = tmp_1428; }
    let tmp_1456 = keys[6] < tmp_1429 || (keys[6] == tmp_1429 && values[6] < tmp_1430);
    if tmp_1449 == tmp_1456 { keys[6] = tmp_1429; values[6] = tmp_1430; }
    let tmp_1457 = keys[7] < tmp_1431 || (keys[7] == tmp_1431 && values[7] < tmp_1432);
    if tmp_1449 == tmp_1457 { keys[7] = tmp_1431; values[7] = tmp_1432; }
    let tmp_1458 = keys[8] < tmp_1433 || (keys[8] == tmp_1433 && values[8] < tmp_1434);
    if tmp_1449 == tmp_1458 { keys[8] = tmp_1433; values[8] = tmp_1434; }
    let tmp_1459 = keys[9] < tmp_1435 || (keys[9] == tmp_1435 && values[9] < tmp_1436);
    if tmp_1449 == tmp_1459 { keys[9] = tmp_1435; values[9] = tmp_1436; }
    let tmp_1460 = keys[10] < tmp_1437 || (keys[10] == tmp_1437 && values[10] < tmp_1438);
    if tmp_1449 == tmp_1460 { keys[10] = tmp_1437; values[10] = tmp_1438; }
    let tmp_1461 = keys[11] < tmp_1439 || (keys[11] == tmp_1439 && values[11] < tmp_1440);
    if tmp_1449 == tmp_1461 { keys[11] = tmp_1439; values[11] = tmp_1440; }
    let tmp_1462 = keys[12] < tmp_1441 || (keys[12] == tmp_1441 && values[12] < tmp_1442);
    if tmp_1449 == tmp_1462 { keys[12] = tmp_1441; values[12] = tmp_1442; }
    let tmp_1463 = keys[13] < tmp_1443 || (keys[13] == tmp_1443 && values[13] < tmp_1444);
    if tmp_1449 == tmp_1463 { keys[13] = tmp_1443; values[13] = tmp_1444; }
    let tmp_1464 = keys[14] < tmp_1445 || (keys[14] == tmp_1445 && values[14] < tmp_1446);
    if tmp_1449 == tmp_1464 { keys[14] = tmp_1445; values[14] = tmp_1446; }
    let tmp_1465 = keys[15] < tmp_1447 || (keys[15] == tmp_1447 && values[15] < tmp_1448);
    if tmp_1449 == tmp_1465 { keys[15] = tmp_1447; values[15] = tmp_1448; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:16) 
    {
    let tmp_1466 = subgroupShuffleXor(keys[0], 1u);
    let tmp_1467 = subgroupShuffleXor(values[0], 1u);
    let tmp_1468 = subgroupShuffleXor(keys[1], 1u);
    let tmp_1469 = subgroupShuffleXor(values[1], 1u);
    let tmp_1470 = subgroupShuffleXor(keys[2], 1u);
    let tmp_1471 = subgroupShuffleXor(values[2], 1u);
    let tmp_1472 = subgroupShuffleXor(keys[3], 1u);
    let tmp_1473 = subgroupShuffleXor(values[3], 1u);
    let tmp_1474 = subgroupShuffleXor(keys[4], 1u);
    let tmp_1475 = subgroupShuffleXor(values[4], 1u);
    let tmp_1476 = subgroupShuffleXor(keys[5], 1u);
    let tmp_1477 = subgroupShuffleXor(values[5], 1u);
    let tmp_1478 = subgroupShuffleXor(keys[6], 1u);
    let tmp_1479 = subgroupShuffleXor(values[6], 1u);
    let tmp_1480 = subgroupShuffleXor(keys[7], 1u);
    let tmp_1481 = subgroupShuffleXor(values[7], 1u);
    let tmp_1482 = subgroupShuffleXor(keys[8], 1u);
    let tmp_1483 = subgroupShuffleXor(values[8], 1u);
    let tmp_1484 = subgroupShuffleXor(keys[9], 1u);
    let tmp_1485 = subgroupShuffleXor(values[9], 1u);
    let tmp_1486 = subgroupShuffleXor(keys[10], 1u);
    let tmp_1487 = subgroupShuffleXor(values[10], 1u);
    let tmp_1488 = subgroupShuffleXor(keys[11], 1u);
    let tmp_1489 = subgroupShuffleXor(values[11], 1u);
    let tmp_1490 = subgroupShuffleXor(keys[12], 1u);
    let tmp_1491 = subgroupShuffleXor(values[12], 1u);
    let tmp_1492 = subgroupShuffleXor(keys[13], 1u);
    let tmp_1493 = subgroupShuffleXor(values[13], 1u);
    let tmp_1494 = subgroupShuffleXor(keys[14], 1u);
    let tmp_1495 = subgroupShuffleXor(values[14], 1u);
    let tmp_1496 = subgroupShuffleXor(keys[15], 1u);
    let tmp_1497 = subgroupShuffleXor(values[15], 1u);
    let tmp_1498 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_1499 = keys[0] < tmp_1466 || (keys[0] == tmp_1466 && values[0] < tmp_1467);
    if tmp_1498 == tmp_1499 { keys[0] = tmp_1466; values[0] = tmp_1467; }
    let tmp_1500 = keys[1] < tmp_1468 || (keys[1] == tmp_1468 && values[1] < tmp_1469);
    if tmp_1498 == tmp_1500 { keys[1] = tmp_1468; values[1] = tmp_1469; }
    let tmp_1501 = keys[2] < tmp_1470 || (keys[2] == tmp_1470 && values[2] < tmp_1471);
    if tmp_1498 == tmp_1501 { keys[2] = tmp_1470; values[2] = tmp_1471; }
    let tmp_1502 = keys[3] < tmp_1472 || (keys[3] == tmp_1472 && values[3] < tmp_1473);
    if tmp_1498 == tmp_1502 { keys[3] = tmp_1472; values[3] = tmp_1473; }
    let tmp_1503 = keys[4] < tmp_1474 || (keys[4] == tmp_1474 && values[4] < tmp_1475);
    if tmp_1498 == tmp_1503 { keys[4] = tmp_1474; values[4] = tmp_1475; }
    let tmp_1504 = keys[5] < tmp_1476 || (keys[5] == tmp_1476 && values[5] < tmp_1477);
    if tmp_1498 == tmp_1504 { keys[5] = tmp_1476; values[5] = tmp_1477; }
    let tmp_1505 = keys[6] < tmp_1478 || (keys[6] == tmp_1478 && values[6] < tmp_1479);
    if tmp_1498 == tmp_1505 { keys[6] = tmp_1478; values[6] = tmp_1479; }
    let tmp_1506 = keys[7] < tmp_1480 || (keys[7] == tmp_1480 && values[7] < tmp_1481);
    if tmp_1498 == tmp_1506 { keys[7] = tmp_1480; values[7] = tmp_1481; }
    let tmp_1507 = keys[8] < tmp_1482 || (keys[8] == tmp_1482 && values[8] < tmp_1483);
    if tmp_1498 == tmp_1507 { keys[8] = tmp_1482; values[8] = tmp_1483; }
    let tmp_1508 = keys[9] < tmp_1484 || (keys[9] == tmp_1484 && values[9] < tmp_1485);
    if tmp_1498 == tmp_1508 { keys[9] = tmp_1484; values[9] = tmp_1485; }
    let tmp_1509 = keys[10] < tmp_1486 || (keys[10] == tmp_1486 && values[10] < tmp_1487);
    if tmp_1498 == tmp_1509 { keys[10] = tmp_1486; values[10] = tmp_1487; }
    let tmp_1510 = keys[11] < tmp_1488 || (keys[11] == tmp_1488 && values[11] < tmp_1489);
    if tmp_1498 == tmp_1510 { keys[11] = tmp_1488; values[11] = tmp_1489; }
    let tmp_1511 = keys[12] < tmp_1490 || (keys[12] == tmp_1490 && values[12] < tmp_1491);
    if tmp_1498 == tmp_1511 { keys[12] = tmp_1490; values[12] = tmp_1491; }
    let tmp_1512 = keys[13] < tmp_1492 || (keys[13] == tmp_1492 && values[13] < tmp_1493);
    if tmp_1498 == tmp_1512 { keys[13] = tmp_1492; values[13] = tmp_1493; }
    let tmp_1513 = keys[14] < tmp_1494 || (keys[14] == tmp_1494 && values[14] < tmp_1495);
    if tmp_1498 == tmp_1513 { keys[14] = tmp_1494; values[14] = tmp_1495; }
    let tmp_1514 = keys[15] < tmp_1496 || (keys[15] == tmp_1496 && values[15] < tmp_1497);
    if tmp_1498 == tmp_1514 { keys[15] = tmp_1496; values[15] = tmp_1497; }
    }
    // exch_local(8,16) 
    // cmp_swap(0,8)
    if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
    // swap(0,8) 
    { let tmp_1515 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_1515;let tmp_1516 = values[0]; values[0] = values[8]; values[8] = tmp_1516; }
    }
    // cmp_swap(1,9)
    if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
    // swap(1,9) 
    { let tmp_1517 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_1517;let tmp_1518 = values[1]; values[1] = values[9]; values[9] = tmp_1518; }
    }
    // cmp_swap(2,10)
    if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
    // swap(2,10) 
    { let tmp_1519 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_1519;let tmp_1520 = values[2]; values[2] = values[10]; values[10] = tmp_1520; }
    }
    // cmp_swap(3,11)
    if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
    // swap(3,11) 
    { let tmp_1521 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_1521;let tmp_1522 = values[3]; values[3] = values[11]; values[11] = tmp_1522; }
    }
    // cmp_swap(4,12)
    if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
    // swap(4,12) 
    { let tmp_1523 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_1523;let tmp_1524 = values[4]; values[4] = values[12]; values[12] = tmp_1524; }
    }
    // cmp_swap(5,13)
    if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
    // swap(5,13) 
    { let tmp_1525 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_1525;let tmp_1526 = values[5]; values[5] = values[13]; values[13] = tmp_1526; }
    }
    // cmp_swap(6,14)
    if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
    // swap(6,14) 
    { let tmp_1527 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_1527;let tmp_1528 = values[6]; values[6] = values[14]; values[14] = tmp_1528; }
    }
    // cmp_swap(7,15)
    if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
    // swap(7,15) 
    { let tmp_1529 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_1529;let tmp_1530 = values[7]; values[7] = values[15]; values[15] = tmp_1530; }
    }
    // exch_local(4,16) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_1531 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_1531;let tmp_1532 = values[0]; values[0] = values[4]; values[4] = tmp_1532; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_1533 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_1533;let tmp_1534 = values[1]; values[1] = values[5]; values[5] = tmp_1534; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_1535 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_1535;let tmp_1536 = values[2]; values[2] = values[6]; values[6] = tmp_1536; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_1537 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_1537;let tmp_1538 = values[3]; values[3] = values[7]; values[7] = tmp_1538; }
    }
    // cmp_swap(8,12)
    if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
    // swap(8,12) 
    { let tmp_1539 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_1539;let tmp_1540 = values[8]; values[8] = values[12]; values[12] = tmp_1540; }
    }
    // cmp_swap(9,13)
    if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
    // swap(9,13) 
    { let tmp_1541 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_1541;let tmp_1542 = values[9]; values[9] = values[13]; values[13] = tmp_1542; }
    }
    // cmp_swap(10,14)
    if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
    // swap(10,14) 
    { let tmp_1543 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_1543;let tmp_1544 = values[10]; values[10] = values[14]; values[14] = tmp_1544; }
    }
    // cmp_swap(11,15)
    if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
    // swap(11,15) 
    { let tmp_1545 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_1545;let tmp_1546 = values[11]; values[11] = values[15]; values[15] = tmp_1546; }
    }
    // exch_local(2,16) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_1547 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_1547;let tmp_1548 = values[0]; values[0] = values[2]; values[2] = tmp_1548; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_1549 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_1549;let tmp_1550 = values[1]; values[1] = values[3]; values[3] = tmp_1550; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_1551 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_1551;let tmp_1552 = values[4]; values[4] = values[6]; values[6] = tmp_1552; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_1553 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_1553;let tmp_1554 = values[5]; values[5] = values[7]; values[7] = tmp_1554; }
    }
    // cmp_swap(8,10)
    if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
    // swap(8,10) 
    { let tmp_1555 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_1555;let tmp_1556 = values[8]; values[8] = values[10]; values[10] = tmp_1556; }
    }
    // cmp_swap(9,11)
    if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
    // swap(9,11) 
    { let tmp_1557 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_1557;let tmp_1558 = values[9]; values[9] = values[11]; values[11] = tmp_1558; }
    }
    // cmp_swap(12,14)
    if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
    // swap(12,14) 
    { let tmp_1559 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_1559;let tmp_1560 = values[12]; values[12] = values[14]; values[14] = tmp_1560; }
    }
    // cmp_swap(13,15)
    if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
    // swap(13,15) 
    { let tmp_1561 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_1561;let tmp_1562 = values[13]; values[13] = values[15]; values[15] = tmp_1562; }
    }
    // exch_local(1,16) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_1563 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_1563;let tmp_1564 = values[0]; values[0] = values[1]; values[1] = tmp_1564; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_1565 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_1565;let tmp_1566 = values[2]; values[2] = values[3]; values[3] = tmp_1566; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_1567 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_1567;let tmp_1568 = values[4]; values[4] = values[5]; values[5] = tmp_1568; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_1569 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_1569;let tmp_1570 = values[6]; values[6] = values[7]; values[7] = tmp_1570; }
    }
    // cmp_swap(8,9)
    if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
    // swap(8,9) 
    { let tmp_1571 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_1571;let tmp_1572 = values[8]; values[8] = values[9]; values[9] = tmp_1572; }
    }
    // cmp_swap(10,11)
    if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
    // swap(10,11) 
    { let tmp_1573 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_1573;let tmp_1574 = values[10]; values[10] = values[11]; values[11] = tmp_1574; }
    }
    // cmp_swap(12,13)
    if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
    // swap(12,13) 
    { let tmp_1575 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_1575;let tmp_1576 = values[12]; values[12] = values[13]; values[13] = tmp_1576; }
    }
    // cmp_swap(14,15)
    if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
    // swap(14,15) 
    { let tmp_1577 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_1577;let tmp_1578 = values[14]; values[14] = values[15]; values[15] = tmp_1578; }
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
