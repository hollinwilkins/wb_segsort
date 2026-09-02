
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
const SG: u32 = 8u;      // register run spans one subgroup: RUN = SG*WPT = 128

var<workgroup> smem_keys: array<u32, N>;
var<workgroup> smem_vals: array<u32, N>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybmerge_sg8_smem16k_n1024_m64_block(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 10u;

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

    // ---- phase 1: per-subgroup register run-sort (RUN = SG*WPT elements) ----
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

    // stage the sorted runs into shared memory (blocked layout)
    for (var r = 0u; r < WPT; r = r + 1u) {
        smem_keys[local_tid * WPT + r] = keys[r];
        smem_vals[local_tid * WPT + r] = values[r];
    }
    workgroupBarrier();

    let base = local_tid * WPT;   // this thread's blocked output range [base, base+WPT)

    // ---- phase 2: recursive merge-path merges through shared memory ----
    // merge pass 0: two sorted runs of 128 -> 256 (register-staged)
    {
        let group_base = (base / 256u) * 256u;
        let diag = base - group_base;
        let a_base = group_base;
        let b_base = group_base + 128u;
        // merge-path: binary search the diagonal for this thread's A/B split
        var lo = select(0u, diag - 128u, diag > 128u);
        var hi = min(diag, 128u);
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
        var out_keys: array<u32, 16>;
        var out_vals: array<u32, 16>;
        for (var k = 0u; k < WPT; k = k + 1u) {
            let take_a = bi >= 128u || (ai < 128u &&
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
        for (var k = 0u; k < WPT; k = k + 1u) {
            smem_keys[base + k] = out_keys[k];
            smem_vals[base + k] = out_vals[k];
        }
    }
    workgroupBarrier();
    // merge pass 1: two sorted runs of 256 -> 512 (register-staged)
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
        var out_keys: array<u32, 16>;
        var out_vals: array<u32, 16>;
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
        for (var k = 0u; k < WPT; k = k + 1u) {
            smem_keys[base + k] = out_keys[k];
            smem_vals[base + k] = out_vals[k];
        }
    }
    workgroupBarrier();
    // merge pass 2: two sorted runs of 512 -> 1024 (register-staged)
    {
        let group_base = (base / 1024u) * 1024u;
        let diag = base - group_base;
        let a_base = group_base;
        let b_base = group_base + 512u;
        // merge-path: binary search the diagonal for this thread's A/B split
        var lo = select(0u, diag - 512u, diag > 512u);
        var hi = min(diag, 512u);
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
        var out_keys: array<u32, 16>;
        var out_vals: array<u32, 16>;
        for (var k = 0u; k < WPT; k = k + 1u) {
            let take_a = bi >= 512u || (ai < 512u &&
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
