
override WG: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 2048u;
const M: u32 = 128u;
const WPT: u32 = 16u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n2048_m128_block(
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
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_0 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_0;
                let tmp_1 = values[0]; values[0] = values[1]; values[1] = tmp_1;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_2 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_2;
                let tmp_3 = values[2]; values[2] = values[3]; values[3] = tmp_3;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_4 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_4;
                let tmp_5 = values[4]; values[4] = values[5]; values[5] = tmp_5;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_6 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_6;
                let tmp_7 = values[6]; values[6] = values[7]; values[7] = tmp_7;
            }
        }
        // cmp_swap(8,9)
        if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
            // swap(8,9)
            {
                let tmp_8 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_8;
                let tmp_9 = values[8]; values[8] = values[9]; values[9] = tmp_9;
            }
        }
        // cmp_swap(10,11)
        if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
            // swap(10,11)
            {
                let tmp_10 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_10;
                let tmp_11 = values[10]; values[10] = values[11]; values[11] = tmp_11;
            }
        }
        // cmp_swap(12,13)
        if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
            // swap(12,13)
            {
                let tmp_12 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_12;
                let tmp_13 = values[12]; values[12] = values[13]; values[13] = tmp_13;
            }
        }
        // cmp_swap(14,15)
        if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
            // swap(14,15)
            {
                let tmp_14 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_14;
                let tmp_15 = values[14]; values[14] = values[15]; values[15] = tmp_15;
            }
        }
    }

    // exch_local(3,16)
    {
        // cmp_swap(0,3)
        if keys[0] > keys[3] || (keys[0] == keys[3] && values[0] > values[3]) {
            // swap(0,3)
            {
                let tmp_16 = keys[0]; keys[0] = keys[3]; keys[3] = tmp_16;
                let tmp_17 = values[0]; values[0] = values[3]; values[3] = tmp_17;
            }
        }
        // cmp_swap(1,2)
        if keys[1] > keys[2] || (keys[1] == keys[2] && values[1] > values[2]) {
            // swap(1,2)
            {
                let tmp_18 = keys[1]; keys[1] = keys[2]; keys[2] = tmp_18;
                let tmp_19 = values[1]; values[1] = values[2]; values[2] = tmp_19;
            }
        }
        // cmp_swap(4,7)
        if keys[4] > keys[7] || (keys[4] == keys[7] && values[4] > values[7]) {
            // swap(4,7)
            {
                let tmp_20 = keys[4]; keys[4] = keys[7]; keys[7] = tmp_20;
                let tmp_21 = values[4]; values[4] = values[7]; values[7] = tmp_21;
            }
        }
        // cmp_swap(5,6)
        if keys[5] > keys[6] || (keys[5] == keys[6] && values[5] > values[6]) {
            // swap(5,6)
            {
                let tmp_22 = keys[5]; keys[5] = keys[6]; keys[6] = tmp_22;
                let tmp_23 = values[5]; values[5] = values[6]; values[6] = tmp_23;
            }
        }
        // cmp_swap(8,11)
        if keys[8] > keys[11] || (keys[8] == keys[11] && values[8] > values[11]) {
            // swap(8,11)
            {
                let tmp_24 = keys[8]; keys[8] = keys[11]; keys[11] = tmp_24;
                let tmp_25 = values[8]; values[8] = values[11]; values[11] = tmp_25;
            }
        }
        // cmp_swap(9,10)
        if keys[9] > keys[10] || (keys[9] == keys[10] && values[9] > values[10]) {
            // swap(9,10)
            {
                let tmp_26 = keys[9]; keys[9] = keys[10]; keys[10] = tmp_26;
                let tmp_27 = values[9]; values[9] = values[10]; values[10] = tmp_27;
            }
        }
        // cmp_swap(12,15)
        if keys[12] > keys[15] || (keys[12] == keys[15] && values[12] > values[15]) {
            // swap(12,15)
            {
                let tmp_28 = keys[12]; keys[12] = keys[15]; keys[15] = tmp_28;
                let tmp_29 = values[12]; values[12] = values[15]; values[15] = tmp_29;
            }
        }
        // cmp_swap(13,14)
        if keys[13] > keys[14] || (keys[13] == keys[14] && values[13] > values[14]) {
            // swap(13,14)
            {
                let tmp_30 = keys[13]; keys[13] = keys[14]; keys[14] = tmp_30;
                let tmp_31 = values[13]; values[13] = values[14]; values[14] = tmp_31;
            }
        }
    }

    // exch_local(1,16)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_32 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_32;
                let tmp_33 = values[0]; values[0] = values[1]; values[1] = tmp_33;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_34 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_34;
                let tmp_35 = values[2]; values[2] = values[3]; values[3] = tmp_35;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_36 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_36;
                let tmp_37 = values[4]; values[4] = values[5]; values[5] = tmp_37;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_38 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_38;
                let tmp_39 = values[6]; values[6] = values[7]; values[7] = tmp_39;
            }
        }
        // cmp_swap(8,9)
        if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
            // swap(8,9)
            {
                let tmp_40 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_40;
                let tmp_41 = values[8]; values[8] = values[9]; values[9] = tmp_41;
            }
        }
        // cmp_swap(10,11)
        if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
            // swap(10,11)
            {
                let tmp_42 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_42;
                let tmp_43 = values[10]; values[10] = values[11]; values[11] = tmp_43;
            }
        }
        // cmp_swap(12,13)
        if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
            // swap(12,13)
            {
                let tmp_44 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_44;
                let tmp_45 = values[12]; values[12] = values[13]; values[13] = tmp_45;
            }
        }
        // cmp_swap(14,15)
        if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
            // swap(14,15)
            {
                let tmp_46 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_46;
                let tmp_47 = values[14]; values[14] = values[15]; values[15] = tmp_47;
            }
        }
    }

    // exch_local(7,16)
    {
        // cmp_swap(0,7)
        if keys[0] > keys[7] || (keys[0] == keys[7] && values[0] > values[7]) {
            // swap(0,7)
            {
                let tmp_48 = keys[0]; keys[0] = keys[7]; keys[7] = tmp_48;
                let tmp_49 = values[0]; values[0] = values[7]; values[7] = tmp_49;
            }
        }
        // cmp_swap(1,6)
        if keys[1] > keys[6] || (keys[1] == keys[6] && values[1] > values[6]) {
            // swap(1,6)
            {
                let tmp_50 = keys[1]; keys[1] = keys[6]; keys[6] = tmp_50;
                let tmp_51 = values[1]; values[1] = values[6]; values[6] = tmp_51;
            }
        }
        // cmp_swap(2,5)
        if keys[2] > keys[5] || (keys[2] == keys[5] && values[2] > values[5]) {
            // swap(2,5)
            {
                let tmp_52 = keys[2]; keys[2] = keys[5]; keys[5] = tmp_52;
                let tmp_53 = values[2]; values[2] = values[5]; values[5] = tmp_53;
            }
        }
        // cmp_swap(3,4)
        if keys[3] > keys[4] || (keys[3] == keys[4] && values[3] > values[4]) {
            // swap(3,4)
            {
                let tmp_54 = keys[3]; keys[3] = keys[4]; keys[4] = tmp_54;
                let tmp_55 = values[3]; values[3] = values[4]; values[4] = tmp_55;
            }
        }
        // cmp_swap(8,15)
        if keys[8] > keys[15] || (keys[8] == keys[15] && values[8] > values[15]) {
            // swap(8,15)
            {
                let tmp_56 = keys[8]; keys[8] = keys[15]; keys[15] = tmp_56;
                let tmp_57 = values[8]; values[8] = values[15]; values[15] = tmp_57;
            }
        }
        // cmp_swap(9,14)
        if keys[9] > keys[14] || (keys[9] == keys[14] && values[9] > values[14]) {
            // swap(9,14)
            {
                let tmp_58 = keys[9]; keys[9] = keys[14]; keys[14] = tmp_58;
                let tmp_59 = values[9]; values[9] = values[14]; values[14] = tmp_59;
            }
        }
        // cmp_swap(10,13)
        if keys[10] > keys[13] || (keys[10] == keys[13] && values[10] > values[13]) {
            // swap(10,13)
            {
                let tmp_60 = keys[10]; keys[10] = keys[13]; keys[13] = tmp_60;
                let tmp_61 = values[10]; values[10] = values[13]; values[13] = tmp_61;
            }
        }
        // cmp_swap(11,12)
        if keys[11] > keys[12] || (keys[11] == keys[12] && values[11] > values[12]) {
            // swap(11,12)
            {
                let tmp_62 = keys[11]; keys[11] = keys[12]; keys[12] = tmp_62;
                let tmp_63 = values[11]; values[11] = values[12]; values[12] = tmp_63;
            }
        }
    }

    // exch_local(2,16)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_64 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_64;
                let tmp_65 = values[0]; values[0] = values[2]; values[2] = tmp_65;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_66 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_66;
                let tmp_67 = values[1]; values[1] = values[3]; values[3] = tmp_67;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_68 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_68;
                let tmp_69 = values[4]; values[4] = values[6]; values[6] = tmp_69;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_70 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_70;
                let tmp_71 = values[5]; values[5] = values[7]; values[7] = tmp_71;
            }
        }
        // cmp_swap(8,10)
        if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
            // swap(8,10)
            {
                let tmp_72 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_72;
                let tmp_73 = values[8]; values[8] = values[10]; values[10] = tmp_73;
            }
        }
        // cmp_swap(9,11)
        if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
            // swap(9,11)
            {
                let tmp_74 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_74;
                let tmp_75 = values[9]; values[9] = values[11]; values[11] = tmp_75;
            }
        }
        // cmp_swap(12,14)
        if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
            // swap(12,14)
            {
                let tmp_76 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_76;
                let tmp_77 = values[12]; values[12] = values[14]; values[14] = tmp_77;
            }
        }
        // cmp_swap(13,15)
        if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
            // swap(13,15)
            {
                let tmp_78 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_78;
                let tmp_79 = values[13]; values[13] = values[15]; values[15] = tmp_79;
            }
        }
    }

    // exch_local(1,16)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_80 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_80;
                let tmp_81 = values[0]; values[0] = values[1]; values[1] = tmp_81;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_82 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_82;
                let tmp_83 = values[2]; values[2] = values[3]; values[3] = tmp_83;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_84 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_84;
                let tmp_85 = values[4]; values[4] = values[5]; values[5] = tmp_85;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_86 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_86;
                let tmp_87 = values[6]; values[6] = values[7]; values[7] = tmp_87;
            }
        }
        // cmp_swap(8,9)
        if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
            // swap(8,9)
            {
                let tmp_88 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_88;
                let tmp_89 = values[8]; values[8] = values[9]; values[9] = tmp_89;
            }
        }
        // cmp_swap(10,11)
        if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
            // swap(10,11)
            {
                let tmp_90 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_90;
                let tmp_91 = values[10]; values[10] = values[11]; values[11] = tmp_91;
            }
        }
        // cmp_swap(12,13)
        if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
            // swap(12,13)
            {
                let tmp_92 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_92;
                let tmp_93 = values[12]; values[12] = values[13]; values[13] = tmp_93;
            }
        }
        // cmp_swap(14,15)
        if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
            // swap(14,15)
            {
                let tmp_94 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_94;
                let tmp_95 = values[14]; values[14] = values[15]; values[15] = tmp_95;
            }
        }
    }

    // exch_local(15,16)
    {
        // cmp_swap(0,15)
        if keys[0] > keys[15] || (keys[0] == keys[15] && values[0] > values[15]) {
            // swap(0,15)
            {
                let tmp_96 = keys[0]; keys[0] = keys[15]; keys[15] = tmp_96;
                let tmp_97 = values[0]; values[0] = values[15]; values[15] = tmp_97;
            }
        }
        // cmp_swap(1,14)
        if keys[1] > keys[14] || (keys[1] == keys[14] && values[1] > values[14]) {
            // swap(1,14)
            {
                let tmp_98 = keys[1]; keys[1] = keys[14]; keys[14] = tmp_98;
                let tmp_99 = values[1]; values[1] = values[14]; values[14] = tmp_99;
            }
        }
        // cmp_swap(2,13)
        if keys[2] > keys[13] || (keys[2] == keys[13] && values[2] > values[13]) {
            // swap(2,13)
            {
                let tmp_100 = keys[2]; keys[2] = keys[13]; keys[13] = tmp_100;
                let tmp_101 = values[2]; values[2] = values[13]; values[13] = tmp_101;
            }
        }
        // cmp_swap(3,12)
        if keys[3] > keys[12] || (keys[3] == keys[12] && values[3] > values[12]) {
            // swap(3,12)
            {
                let tmp_102 = keys[3]; keys[3] = keys[12]; keys[12] = tmp_102;
                let tmp_103 = values[3]; values[3] = values[12]; values[12] = tmp_103;
            }
        }
        // cmp_swap(4,11)
        if keys[4] > keys[11] || (keys[4] == keys[11] && values[4] > values[11]) {
            // swap(4,11)
            {
                let tmp_104 = keys[4]; keys[4] = keys[11]; keys[11] = tmp_104;
                let tmp_105 = values[4]; values[4] = values[11]; values[11] = tmp_105;
            }
        }
        // cmp_swap(5,10)
        if keys[5] > keys[10] || (keys[5] == keys[10] && values[5] > values[10]) {
            // swap(5,10)
            {
                let tmp_106 = keys[5]; keys[5] = keys[10]; keys[10] = tmp_106;
                let tmp_107 = values[5]; values[5] = values[10]; values[10] = tmp_107;
            }
        }
        // cmp_swap(6,9)
        if keys[6] > keys[9] || (keys[6] == keys[9] && values[6] > values[9]) {
            // swap(6,9)
            {
                let tmp_108 = keys[6]; keys[6] = keys[9]; keys[9] = tmp_108;
                let tmp_109 = values[6]; values[6] = values[9]; values[9] = tmp_109;
            }
        }
        // cmp_swap(7,8)
        if keys[7] > keys[8] || (keys[7] == keys[8] && values[7] > values[8]) {
            // swap(7,8)
            {
                let tmp_110 = keys[7]; keys[7] = keys[8]; keys[8] = tmp_110;
                let tmp_111 = values[7]; values[7] = values[8]; values[8] = tmp_111;
            }
        }
    }

    // exch_local(4,16)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_112 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_112;
                let tmp_113 = values[0]; values[0] = values[4]; values[4] = tmp_113;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_114 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_114;
                let tmp_115 = values[1]; values[1] = values[5]; values[5] = tmp_115;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_116 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_116;
                let tmp_117 = values[2]; values[2] = values[6]; values[6] = tmp_117;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_118 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_118;
                let tmp_119 = values[3]; values[3] = values[7]; values[7] = tmp_119;
            }
        }
        // cmp_swap(8,12)
        if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
            // swap(8,12)
            {
                let tmp_120 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_120;
                let tmp_121 = values[8]; values[8] = values[12]; values[12] = tmp_121;
            }
        }
        // cmp_swap(9,13)
        if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
            // swap(9,13)
            {
                let tmp_122 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_122;
                let tmp_123 = values[9]; values[9] = values[13]; values[13] = tmp_123;
            }
        }
        // cmp_swap(10,14)
        if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
            // swap(10,14)
            {
                let tmp_124 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_124;
                let tmp_125 = values[10]; values[10] = values[14]; values[14] = tmp_125;
            }
        }
        // cmp_swap(11,15)
        if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
            // swap(11,15)
            {
                let tmp_126 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_126;
                let tmp_127 = values[11]; values[11] = values[15]; values[15] = tmp_127;
            }
        }
    }

    // exch_local(2,16)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_128 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_128;
                let tmp_129 = values[0]; values[0] = values[2]; values[2] = tmp_129;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_130 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_130;
                let tmp_131 = values[1]; values[1] = values[3]; values[3] = tmp_131;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_132 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_132;
                let tmp_133 = values[4]; values[4] = values[6]; values[6] = tmp_133;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_134 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_134;
                let tmp_135 = values[5]; values[5] = values[7]; values[7] = tmp_135;
            }
        }
        // cmp_swap(8,10)
        if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
            // swap(8,10)
            {
                let tmp_136 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_136;
                let tmp_137 = values[8]; values[8] = values[10]; values[10] = tmp_137;
            }
        }
        // cmp_swap(9,11)
        if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
            // swap(9,11)
            {
                let tmp_138 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_138;
                let tmp_139 = values[9]; values[9] = values[11]; values[11] = tmp_139;
            }
        }
        // cmp_swap(12,14)
        if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
            // swap(12,14)
            {
                let tmp_140 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_140;
                let tmp_141 = values[12]; values[12] = values[14]; values[14] = tmp_141;
            }
        }
        // cmp_swap(13,15)
        if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
            // swap(13,15)
            {
                let tmp_142 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_142;
                let tmp_143 = values[13]; values[13] = values[15]; values[15] = tmp_143;
            }
        }
    }

    // exch_local(1,16)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_144 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_144;
                let tmp_145 = values[0]; values[0] = values[1]; values[1] = tmp_145;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_146 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_146;
                let tmp_147 = values[2]; values[2] = values[3]; values[3] = tmp_147;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_148 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_148;
                let tmp_149 = values[4]; values[4] = values[5]; values[5] = tmp_149;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_150 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_150;
                let tmp_151 = values[6]; values[6] = values[7]; values[7] = tmp_151;
            }
        }
        // cmp_swap(8,9)
        if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
            // swap(8,9)
            {
                let tmp_152 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_152;
                let tmp_153 = values[8]; values[8] = values[9]; values[9] = tmp_153;
            }
        }
        // cmp_swap(10,11)
        if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
            // swap(10,11)
            {
                let tmp_154 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_154;
                let tmp_155 = values[10]; values[10] = values[11]; values[11] = tmp_155;
            }
        }
        // cmp_swap(12,13)
        if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
            // swap(12,13)
            {
                let tmp_156 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_156;
                let tmp_157 = values[12]; values[12] = values[13]; values[13] = tmp_157;
            }
        }
        // cmp_swap(14,15)
        if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
            // swap(14,15)
            {
                let tmp_158 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_158;
                let tmp_159 = values[14]; values[14] = values[15]; values[15] = tmp_159;
            }
        }
    }

    // exch_intxn(tmask:1,swbit:0,wpt:16)
    {
        // _exch_workgroup([(0, 15), (1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8), (8, 7), (9, 6), (10, 5), (11, 4), (12, 3), (13, 2), (14, 1), (15, 0)],1,0)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_160 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_161 = seg_base + (local_tid ^ 1u);
            let tmp_162 = smem_keys[tmp_161 * WPT + 15u];
            let tmp_163 = smem_vals[tmp_161 * WPT + 15u];
            let tmp_164 = keys[0] < tmp_162 || (keys[0] == tmp_162 && values[0] < tmp_163);
            if tmp_160 == tmp_164 { keys[0] = tmp_162; values[0] = tmp_163; }
            let tmp_165 = smem_keys[tmp_161 * WPT + 14u];
            let tmp_166 = smem_vals[tmp_161 * WPT + 14u];
            let tmp_167 = keys[1] < tmp_165 || (keys[1] == tmp_165 && values[1] < tmp_166);
            if tmp_160 == tmp_167 { keys[1] = tmp_165; values[1] = tmp_166; }
            let tmp_168 = smem_keys[tmp_161 * WPT + 13u];
            let tmp_169 = smem_vals[tmp_161 * WPT + 13u];
            let tmp_170 = keys[2] < tmp_168 || (keys[2] == tmp_168 && values[2] < tmp_169);
            if tmp_160 == tmp_170 { keys[2] = tmp_168; values[2] = tmp_169; }
            let tmp_171 = smem_keys[tmp_161 * WPT + 12u];
            let tmp_172 = smem_vals[tmp_161 * WPT + 12u];
            let tmp_173 = keys[3] < tmp_171 || (keys[3] == tmp_171 && values[3] < tmp_172);
            if tmp_160 == tmp_173 { keys[3] = tmp_171; values[3] = tmp_172; }
            let tmp_174 = smem_keys[tmp_161 * WPT + 11u];
            let tmp_175 = smem_vals[tmp_161 * WPT + 11u];
            let tmp_176 = keys[4] < tmp_174 || (keys[4] == tmp_174 && values[4] < tmp_175);
            if tmp_160 == tmp_176 { keys[4] = tmp_174; values[4] = tmp_175; }
            let tmp_177 = smem_keys[tmp_161 * WPT + 10u];
            let tmp_178 = smem_vals[tmp_161 * WPT + 10u];
            let tmp_179 = keys[5] < tmp_177 || (keys[5] == tmp_177 && values[5] < tmp_178);
            if tmp_160 == tmp_179 { keys[5] = tmp_177; values[5] = tmp_178; }
            let tmp_180 = smem_keys[tmp_161 * WPT + 9u];
            let tmp_181 = smem_vals[tmp_161 * WPT + 9u];
            let tmp_182 = keys[6] < tmp_180 || (keys[6] == tmp_180 && values[6] < tmp_181);
            if tmp_160 == tmp_182 { keys[6] = tmp_180; values[6] = tmp_181; }
            let tmp_183 = smem_keys[tmp_161 * WPT + 8u];
            let tmp_184 = smem_vals[tmp_161 * WPT + 8u];
            let tmp_185 = keys[7] < tmp_183 || (keys[7] == tmp_183 && values[7] < tmp_184);
            if tmp_160 == tmp_185 { keys[7] = tmp_183; values[7] = tmp_184; }
            let tmp_186 = smem_keys[tmp_161 * WPT + 7u];
            let tmp_187 = smem_vals[tmp_161 * WPT + 7u];
            let tmp_188 = keys[8] < tmp_186 || (keys[8] == tmp_186 && values[8] < tmp_187);
            if tmp_160 == tmp_188 { keys[8] = tmp_186; values[8] = tmp_187; }
            let tmp_189 = smem_keys[tmp_161 * WPT + 6u];
            let tmp_190 = smem_vals[tmp_161 * WPT + 6u];
            let tmp_191 = keys[9] < tmp_189 || (keys[9] == tmp_189 && values[9] < tmp_190);
            if tmp_160 == tmp_191 { keys[9] = tmp_189; values[9] = tmp_190; }
            let tmp_192 = smem_keys[tmp_161 * WPT + 5u];
            let tmp_193 = smem_vals[tmp_161 * WPT + 5u];
            let tmp_194 = keys[10] < tmp_192 || (keys[10] == tmp_192 && values[10] < tmp_193);
            if tmp_160 == tmp_194 { keys[10] = tmp_192; values[10] = tmp_193; }
            let tmp_195 = smem_keys[tmp_161 * WPT + 4u];
            let tmp_196 = smem_vals[tmp_161 * WPT + 4u];
            let tmp_197 = keys[11] < tmp_195 || (keys[11] == tmp_195 && values[11] < tmp_196);
            if tmp_160 == tmp_197 { keys[11] = tmp_195; values[11] = tmp_196; }
            let tmp_198 = smem_keys[tmp_161 * WPT + 3u];
            let tmp_199 = smem_vals[tmp_161 * WPT + 3u];
            let tmp_200 = keys[12] < tmp_198 || (keys[12] == tmp_198 && values[12] < tmp_199);
            if tmp_160 == tmp_200 { keys[12] = tmp_198; values[12] = tmp_199; }
            let tmp_201 = smem_keys[tmp_161 * WPT + 2u];
            let tmp_202 = smem_vals[tmp_161 * WPT + 2u];
            let tmp_203 = keys[13] < tmp_201 || (keys[13] == tmp_201 && values[13] < tmp_202);
            if tmp_160 == tmp_203 { keys[13] = tmp_201; values[13] = tmp_202; }
            let tmp_204 = smem_keys[tmp_161 * WPT + 1u];
            let tmp_205 = smem_vals[tmp_161 * WPT + 1u];
            let tmp_206 = keys[14] < tmp_204 || (keys[14] == tmp_204 && values[14] < tmp_205);
            if tmp_160 == tmp_206 { keys[14] = tmp_204; values[14] = tmp_205; }
            let tmp_207 = smem_keys[tmp_161 * WPT + 0u];
            let tmp_208 = smem_vals[tmp_161 * WPT + 0u];
            let tmp_209 = keys[15] < tmp_207 || (keys[15] == tmp_207 && values[15] < tmp_208);
            if tmp_160 == tmp_209 { keys[15] = tmp_207; values[15] = tmp_208; }
            workgroupBarrier();
        }
    }

    // exch_local(8,16)
    {
        // cmp_swap(0,8)
        if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
            // swap(0,8)
            {
                let tmp_210 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_210;
                let tmp_211 = values[0]; values[0] = values[8]; values[8] = tmp_211;
            }
        }
        // cmp_swap(1,9)
        if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
            // swap(1,9)
            {
                let tmp_212 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_212;
                let tmp_213 = values[1]; values[1] = values[9]; values[9] = tmp_213;
            }
        }
        // cmp_swap(2,10)
        if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
            // swap(2,10)
            {
                let tmp_214 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_214;
                let tmp_215 = values[2]; values[2] = values[10]; values[10] = tmp_215;
            }
        }
        // cmp_swap(3,11)
        if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
            // swap(3,11)
            {
                let tmp_216 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_216;
                let tmp_217 = values[3]; values[3] = values[11]; values[11] = tmp_217;
            }
        }
        // cmp_swap(4,12)
        if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
            // swap(4,12)
            {
                let tmp_218 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_218;
                let tmp_219 = values[4]; values[4] = values[12]; values[12] = tmp_219;
            }
        }
        // cmp_swap(5,13)
        if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
            // swap(5,13)
            {
                let tmp_220 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_220;
                let tmp_221 = values[5]; values[5] = values[13]; values[13] = tmp_221;
            }
        }
        // cmp_swap(6,14)
        if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
            // swap(6,14)
            {
                let tmp_222 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_222;
                let tmp_223 = values[6]; values[6] = values[14]; values[14] = tmp_223;
            }
        }
        // cmp_swap(7,15)
        if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
            // swap(7,15)
            {
                let tmp_224 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_224;
                let tmp_225 = values[7]; values[7] = values[15]; values[15] = tmp_225;
            }
        }
    }

    // exch_local(4,16)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_226 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_226;
                let tmp_227 = values[0]; values[0] = values[4]; values[4] = tmp_227;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_228 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_228;
                let tmp_229 = values[1]; values[1] = values[5]; values[5] = tmp_229;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_230 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_230;
                let tmp_231 = values[2]; values[2] = values[6]; values[6] = tmp_231;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_232 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_232;
                let tmp_233 = values[3]; values[3] = values[7]; values[7] = tmp_233;
            }
        }
        // cmp_swap(8,12)
        if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
            // swap(8,12)
            {
                let tmp_234 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_234;
                let tmp_235 = values[8]; values[8] = values[12]; values[12] = tmp_235;
            }
        }
        // cmp_swap(9,13)
        if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
            // swap(9,13)
            {
                let tmp_236 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_236;
                let tmp_237 = values[9]; values[9] = values[13]; values[13] = tmp_237;
            }
        }
        // cmp_swap(10,14)
        if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
            // swap(10,14)
            {
                let tmp_238 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_238;
                let tmp_239 = values[10]; values[10] = values[14]; values[14] = tmp_239;
            }
        }
        // cmp_swap(11,15)
        if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
            // swap(11,15)
            {
                let tmp_240 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_240;
                let tmp_241 = values[11]; values[11] = values[15]; values[15] = tmp_241;
            }
        }
    }

    // exch_local(2,16)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_242 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_242;
                let tmp_243 = values[0]; values[0] = values[2]; values[2] = tmp_243;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_244 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_244;
                let tmp_245 = values[1]; values[1] = values[3]; values[3] = tmp_245;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_246 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_246;
                let tmp_247 = values[4]; values[4] = values[6]; values[6] = tmp_247;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_248 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_248;
                let tmp_249 = values[5]; values[5] = values[7]; values[7] = tmp_249;
            }
        }
        // cmp_swap(8,10)
        if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
            // swap(8,10)
            {
                let tmp_250 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_250;
                let tmp_251 = values[8]; values[8] = values[10]; values[10] = tmp_251;
            }
        }
        // cmp_swap(9,11)
        if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
            // swap(9,11)
            {
                let tmp_252 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_252;
                let tmp_253 = values[9]; values[9] = values[11]; values[11] = tmp_253;
            }
        }
        // cmp_swap(12,14)
        if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
            // swap(12,14)
            {
                let tmp_254 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_254;
                let tmp_255 = values[12]; values[12] = values[14]; values[14] = tmp_255;
            }
        }
        // cmp_swap(13,15)
        if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
            // swap(13,15)
            {
                let tmp_256 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_256;
                let tmp_257 = values[13]; values[13] = values[15]; values[15] = tmp_257;
            }
        }
    }

    // exch_local(1,16)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_258 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_258;
                let tmp_259 = values[0]; values[0] = values[1]; values[1] = tmp_259;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_260 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_260;
                let tmp_261 = values[2]; values[2] = values[3]; values[3] = tmp_261;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_262 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_262;
                let tmp_263 = values[4]; values[4] = values[5]; values[5] = tmp_263;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_264 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_264;
                let tmp_265 = values[6]; values[6] = values[7]; values[7] = tmp_265;
            }
        }
        // cmp_swap(8,9)
        if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
            // swap(8,9)
            {
                let tmp_266 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_266;
                let tmp_267 = values[8]; values[8] = values[9]; values[9] = tmp_267;
            }
        }
        // cmp_swap(10,11)
        if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
            // swap(10,11)
            {
                let tmp_268 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_268;
                let tmp_269 = values[10]; values[10] = values[11]; values[11] = tmp_269;
            }
        }
        // cmp_swap(12,13)
        if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
            // swap(12,13)
            {
                let tmp_270 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_270;
                let tmp_271 = values[12]; values[12] = values[13]; values[13] = tmp_271;
            }
        }
        // cmp_swap(14,15)
        if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
            // swap(14,15)
            {
                let tmp_272 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_272;
                let tmp_273 = values[14]; values[14] = values[15]; values[15] = tmp_273;
            }
        }
    }

    // exch_intxn(tmask:3,swbit:1,wpt:16)
    {
        // _exch_workgroup([(0, 15), (1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8), (8, 7), (9, 6), (10, 5), (11, 4), (12, 3), (13, 2), (14, 1), (15, 0)],3,1)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_274 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_275 = seg_base + (local_tid ^ 3u);
            let tmp_276 = smem_keys[tmp_275 * WPT + 15u];
            let tmp_277 = smem_vals[tmp_275 * WPT + 15u];
            let tmp_278 = keys[0] < tmp_276 || (keys[0] == tmp_276 && values[0] < tmp_277);
            if tmp_274 == tmp_278 { keys[0] = tmp_276; values[0] = tmp_277; }
            let tmp_279 = smem_keys[tmp_275 * WPT + 14u];
            let tmp_280 = smem_vals[tmp_275 * WPT + 14u];
            let tmp_281 = keys[1] < tmp_279 || (keys[1] == tmp_279 && values[1] < tmp_280);
            if tmp_274 == tmp_281 { keys[1] = tmp_279; values[1] = tmp_280; }
            let tmp_282 = smem_keys[tmp_275 * WPT + 13u];
            let tmp_283 = smem_vals[tmp_275 * WPT + 13u];
            let tmp_284 = keys[2] < tmp_282 || (keys[2] == tmp_282 && values[2] < tmp_283);
            if tmp_274 == tmp_284 { keys[2] = tmp_282; values[2] = tmp_283; }
            let tmp_285 = smem_keys[tmp_275 * WPT + 12u];
            let tmp_286 = smem_vals[tmp_275 * WPT + 12u];
            let tmp_287 = keys[3] < tmp_285 || (keys[3] == tmp_285 && values[3] < tmp_286);
            if tmp_274 == tmp_287 { keys[3] = tmp_285; values[3] = tmp_286; }
            let tmp_288 = smem_keys[tmp_275 * WPT + 11u];
            let tmp_289 = smem_vals[tmp_275 * WPT + 11u];
            let tmp_290 = keys[4] < tmp_288 || (keys[4] == tmp_288 && values[4] < tmp_289);
            if tmp_274 == tmp_290 { keys[4] = tmp_288; values[4] = tmp_289; }
            let tmp_291 = smem_keys[tmp_275 * WPT + 10u];
            let tmp_292 = smem_vals[tmp_275 * WPT + 10u];
            let tmp_293 = keys[5] < tmp_291 || (keys[5] == tmp_291 && values[5] < tmp_292);
            if tmp_274 == tmp_293 { keys[5] = tmp_291; values[5] = tmp_292; }
            let tmp_294 = smem_keys[tmp_275 * WPT + 9u];
            let tmp_295 = smem_vals[tmp_275 * WPT + 9u];
            let tmp_296 = keys[6] < tmp_294 || (keys[6] == tmp_294 && values[6] < tmp_295);
            if tmp_274 == tmp_296 { keys[6] = tmp_294; values[6] = tmp_295; }
            let tmp_297 = smem_keys[tmp_275 * WPT + 8u];
            let tmp_298 = smem_vals[tmp_275 * WPT + 8u];
            let tmp_299 = keys[7] < tmp_297 || (keys[7] == tmp_297 && values[7] < tmp_298);
            if tmp_274 == tmp_299 { keys[7] = tmp_297; values[7] = tmp_298; }
            let tmp_300 = smem_keys[tmp_275 * WPT + 7u];
            let tmp_301 = smem_vals[tmp_275 * WPT + 7u];
            let tmp_302 = keys[8] < tmp_300 || (keys[8] == tmp_300 && values[8] < tmp_301);
            if tmp_274 == tmp_302 { keys[8] = tmp_300; values[8] = tmp_301; }
            let tmp_303 = smem_keys[tmp_275 * WPT + 6u];
            let tmp_304 = smem_vals[tmp_275 * WPT + 6u];
            let tmp_305 = keys[9] < tmp_303 || (keys[9] == tmp_303 && values[9] < tmp_304);
            if tmp_274 == tmp_305 { keys[9] = tmp_303; values[9] = tmp_304; }
            let tmp_306 = smem_keys[tmp_275 * WPT + 5u];
            let tmp_307 = smem_vals[tmp_275 * WPT + 5u];
            let tmp_308 = keys[10] < tmp_306 || (keys[10] == tmp_306 && values[10] < tmp_307);
            if tmp_274 == tmp_308 { keys[10] = tmp_306; values[10] = tmp_307; }
            let tmp_309 = smem_keys[tmp_275 * WPT + 4u];
            let tmp_310 = smem_vals[tmp_275 * WPT + 4u];
            let tmp_311 = keys[11] < tmp_309 || (keys[11] == tmp_309 && values[11] < tmp_310);
            if tmp_274 == tmp_311 { keys[11] = tmp_309; values[11] = tmp_310; }
            let tmp_312 = smem_keys[tmp_275 * WPT + 3u];
            let tmp_313 = smem_vals[tmp_275 * WPT + 3u];
            let tmp_314 = keys[12] < tmp_312 || (keys[12] == tmp_312 && values[12] < tmp_313);
            if tmp_274 == tmp_314 { keys[12] = tmp_312; values[12] = tmp_313; }
            let tmp_315 = smem_keys[tmp_275 * WPT + 2u];
            let tmp_316 = smem_vals[tmp_275 * WPT + 2u];
            let tmp_317 = keys[13] < tmp_315 || (keys[13] == tmp_315 && values[13] < tmp_316);
            if tmp_274 == tmp_317 { keys[13] = tmp_315; values[13] = tmp_316; }
            let tmp_318 = smem_keys[tmp_275 * WPT + 1u];
            let tmp_319 = smem_vals[tmp_275 * WPT + 1u];
            let tmp_320 = keys[14] < tmp_318 || (keys[14] == tmp_318 && values[14] < tmp_319);
            if tmp_274 == tmp_320 { keys[14] = tmp_318; values[14] = tmp_319; }
            let tmp_321 = smem_keys[tmp_275 * WPT + 0u];
            let tmp_322 = smem_vals[tmp_275 * WPT + 0u];
            let tmp_323 = keys[15] < tmp_321 || (keys[15] == tmp_321 && values[15] < tmp_322);
            if tmp_274 == tmp_323 { keys[15] = tmp_321; values[15] = tmp_322; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],1,0)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_324 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_325 = seg_base + (local_tid ^ 1u);
            let tmp_326 = smem_keys[tmp_325 * WPT + 0u];
            let tmp_327 = smem_vals[tmp_325 * WPT + 0u];
            let tmp_328 = keys[0] < tmp_326 || (keys[0] == tmp_326 && values[0] < tmp_327);
            if tmp_324 == tmp_328 { keys[0] = tmp_326; values[0] = tmp_327; }
            let tmp_329 = smem_keys[tmp_325 * WPT + 1u];
            let tmp_330 = smem_vals[tmp_325 * WPT + 1u];
            let tmp_331 = keys[1] < tmp_329 || (keys[1] == tmp_329 && values[1] < tmp_330);
            if tmp_324 == tmp_331 { keys[1] = tmp_329; values[1] = tmp_330; }
            let tmp_332 = smem_keys[tmp_325 * WPT + 2u];
            let tmp_333 = smem_vals[tmp_325 * WPT + 2u];
            let tmp_334 = keys[2] < tmp_332 || (keys[2] == tmp_332 && values[2] < tmp_333);
            if tmp_324 == tmp_334 { keys[2] = tmp_332; values[2] = tmp_333; }
            let tmp_335 = smem_keys[tmp_325 * WPT + 3u];
            let tmp_336 = smem_vals[tmp_325 * WPT + 3u];
            let tmp_337 = keys[3] < tmp_335 || (keys[3] == tmp_335 && values[3] < tmp_336);
            if tmp_324 == tmp_337 { keys[3] = tmp_335; values[3] = tmp_336; }
            let tmp_338 = smem_keys[tmp_325 * WPT + 4u];
            let tmp_339 = smem_vals[tmp_325 * WPT + 4u];
            let tmp_340 = keys[4] < tmp_338 || (keys[4] == tmp_338 && values[4] < tmp_339);
            if tmp_324 == tmp_340 { keys[4] = tmp_338; values[4] = tmp_339; }
            let tmp_341 = smem_keys[tmp_325 * WPT + 5u];
            let tmp_342 = smem_vals[tmp_325 * WPT + 5u];
            let tmp_343 = keys[5] < tmp_341 || (keys[5] == tmp_341 && values[5] < tmp_342);
            if tmp_324 == tmp_343 { keys[5] = tmp_341; values[5] = tmp_342; }
            let tmp_344 = smem_keys[tmp_325 * WPT + 6u];
            let tmp_345 = smem_vals[tmp_325 * WPT + 6u];
            let tmp_346 = keys[6] < tmp_344 || (keys[6] == tmp_344 && values[6] < tmp_345);
            if tmp_324 == tmp_346 { keys[6] = tmp_344; values[6] = tmp_345; }
            let tmp_347 = smem_keys[tmp_325 * WPT + 7u];
            let tmp_348 = smem_vals[tmp_325 * WPT + 7u];
            let tmp_349 = keys[7] < tmp_347 || (keys[7] == tmp_347 && values[7] < tmp_348);
            if tmp_324 == tmp_349 { keys[7] = tmp_347; values[7] = tmp_348; }
            let tmp_350 = smem_keys[tmp_325 * WPT + 8u];
            let tmp_351 = smem_vals[tmp_325 * WPT + 8u];
            let tmp_352 = keys[8] < tmp_350 || (keys[8] == tmp_350 && values[8] < tmp_351);
            if tmp_324 == tmp_352 { keys[8] = tmp_350; values[8] = tmp_351; }
            let tmp_353 = smem_keys[tmp_325 * WPT + 9u];
            let tmp_354 = smem_vals[tmp_325 * WPT + 9u];
            let tmp_355 = keys[9] < tmp_353 || (keys[9] == tmp_353 && values[9] < tmp_354);
            if tmp_324 == tmp_355 { keys[9] = tmp_353; values[9] = tmp_354; }
            let tmp_356 = smem_keys[tmp_325 * WPT + 10u];
            let tmp_357 = smem_vals[tmp_325 * WPT + 10u];
            let tmp_358 = keys[10] < tmp_356 || (keys[10] == tmp_356 && values[10] < tmp_357);
            if tmp_324 == tmp_358 { keys[10] = tmp_356; values[10] = tmp_357; }
            let tmp_359 = smem_keys[tmp_325 * WPT + 11u];
            let tmp_360 = smem_vals[tmp_325 * WPT + 11u];
            let tmp_361 = keys[11] < tmp_359 || (keys[11] == tmp_359 && values[11] < tmp_360);
            if tmp_324 == tmp_361 { keys[11] = tmp_359; values[11] = tmp_360; }
            let tmp_362 = smem_keys[tmp_325 * WPT + 12u];
            let tmp_363 = smem_vals[tmp_325 * WPT + 12u];
            let tmp_364 = keys[12] < tmp_362 || (keys[12] == tmp_362 && values[12] < tmp_363);
            if tmp_324 == tmp_364 { keys[12] = tmp_362; values[12] = tmp_363; }
            let tmp_365 = smem_keys[tmp_325 * WPT + 13u];
            let tmp_366 = smem_vals[tmp_325 * WPT + 13u];
            let tmp_367 = keys[13] < tmp_365 || (keys[13] == tmp_365 && values[13] < tmp_366);
            if tmp_324 == tmp_367 { keys[13] = tmp_365; values[13] = tmp_366; }
            let tmp_368 = smem_keys[tmp_325 * WPT + 14u];
            let tmp_369 = smem_vals[tmp_325 * WPT + 14u];
            let tmp_370 = keys[14] < tmp_368 || (keys[14] == tmp_368 && values[14] < tmp_369);
            if tmp_324 == tmp_370 { keys[14] = tmp_368; values[14] = tmp_369; }
            let tmp_371 = smem_keys[tmp_325 * WPT + 15u];
            let tmp_372 = smem_vals[tmp_325 * WPT + 15u];
            let tmp_373 = keys[15] < tmp_371 || (keys[15] == tmp_371 && values[15] < tmp_372);
            if tmp_324 == tmp_373 { keys[15] = tmp_371; values[15] = tmp_372; }
            workgroupBarrier();
        }
    }

    // exch_local(8,16)
    {
        // cmp_swap(0,8)
        if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
            // swap(0,8)
            {
                let tmp_374 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_374;
                let tmp_375 = values[0]; values[0] = values[8]; values[8] = tmp_375;
            }
        }
        // cmp_swap(1,9)
        if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
            // swap(1,9)
            {
                let tmp_376 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_376;
                let tmp_377 = values[1]; values[1] = values[9]; values[9] = tmp_377;
            }
        }
        // cmp_swap(2,10)
        if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
            // swap(2,10)
            {
                let tmp_378 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_378;
                let tmp_379 = values[2]; values[2] = values[10]; values[10] = tmp_379;
            }
        }
        // cmp_swap(3,11)
        if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
            // swap(3,11)
            {
                let tmp_380 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_380;
                let tmp_381 = values[3]; values[3] = values[11]; values[11] = tmp_381;
            }
        }
        // cmp_swap(4,12)
        if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
            // swap(4,12)
            {
                let tmp_382 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_382;
                let tmp_383 = values[4]; values[4] = values[12]; values[12] = tmp_383;
            }
        }
        // cmp_swap(5,13)
        if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
            // swap(5,13)
            {
                let tmp_384 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_384;
                let tmp_385 = values[5]; values[5] = values[13]; values[13] = tmp_385;
            }
        }
        // cmp_swap(6,14)
        if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
            // swap(6,14)
            {
                let tmp_386 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_386;
                let tmp_387 = values[6]; values[6] = values[14]; values[14] = tmp_387;
            }
        }
        // cmp_swap(7,15)
        if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
            // swap(7,15)
            {
                let tmp_388 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_388;
                let tmp_389 = values[7]; values[7] = values[15]; values[15] = tmp_389;
            }
        }
    }

    // exch_local(4,16)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_390 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_390;
                let tmp_391 = values[0]; values[0] = values[4]; values[4] = tmp_391;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_392 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_392;
                let tmp_393 = values[1]; values[1] = values[5]; values[5] = tmp_393;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_394 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_394;
                let tmp_395 = values[2]; values[2] = values[6]; values[6] = tmp_395;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_396 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_396;
                let tmp_397 = values[3]; values[3] = values[7]; values[7] = tmp_397;
            }
        }
        // cmp_swap(8,12)
        if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
            // swap(8,12)
            {
                let tmp_398 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_398;
                let tmp_399 = values[8]; values[8] = values[12]; values[12] = tmp_399;
            }
        }
        // cmp_swap(9,13)
        if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
            // swap(9,13)
            {
                let tmp_400 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_400;
                let tmp_401 = values[9]; values[9] = values[13]; values[13] = tmp_401;
            }
        }
        // cmp_swap(10,14)
        if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
            // swap(10,14)
            {
                let tmp_402 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_402;
                let tmp_403 = values[10]; values[10] = values[14]; values[14] = tmp_403;
            }
        }
        // cmp_swap(11,15)
        if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
            // swap(11,15)
            {
                let tmp_404 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_404;
                let tmp_405 = values[11]; values[11] = values[15]; values[15] = tmp_405;
            }
        }
    }

    // exch_local(2,16)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_406 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_406;
                let tmp_407 = values[0]; values[0] = values[2]; values[2] = tmp_407;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_408 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_408;
                let tmp_409 = values[1]; values[1] = values[3]; values[3] = tmp_409;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_410 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_410;
                let tmp_411 = values[4]; values[4] = values[6]; values[6] = tmp_411;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_412 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_412;
                let tmp_413 = values[5]; values[5] = values[7]; values[7] = tmp_413;
            }
        }
        // cmp_swap(8,10)
        if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
            // swap(8,10)
            {
                let tmp_414 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_414;
                let tmp_415 = values[8]; values[8] = values[10]; values[10] = tmp_415;
            }
        }
        // cmp_swap(9,11)
        if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
            // swap(9,11)
            {
                let tmp_416 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_416;
                let tmp_417 = values[9]; values[9] = values[11]; values[11] = tmp_417;
            }
        }
        // cmp_swap(12,14)
        if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
            // swap(12,14)
            {
                let tmp_418 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_418;
                let tmp_419 = values[12]; values[12] = values[14]; values[14] = tmp_419;
            }
        }
        // cmp_swap(13,15)
        if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
            // swap(13,15)
            {
                let tmp_420 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_420;
                let tmp_421 = values[13]; values[13] = values[15]; values[15] = tmp_421;
            }
        }
    }

    // exch_local(1,16)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_422 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_422;
                let tmp_423 = values[0]; values[0] = values[1]; values[1] = tmp_423;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_424 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_424;
                let tmp_425 = values[2]; values[2] = values[3]; values[3] = tmp_425;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_426 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_426;
                let tmp_427 = values[4]; values[4] = values[5]; values[5] = tmp_427;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_428 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_428;
                let tmp_429 = values[6]; values[6] = values[7]; values[7] = tmp_429;
            }
        }
        // cmp_swap(8,9)
        if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
            // swap(8,9)
            {
                let tmp_430 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_430;
                let tmp_431 = values[8]; values[8] = values[9]; values[9] = tmp_431;
            }
        }
        // cmp_swap(10,11)
        if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
            // swap(10,11)
            {
                let tmp_432 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_432;
                let tmp_433 = values[10]; values[10] = values[11]; values[11] = tmp_433;
            }
        }
        // cmp_swap(12,13)
        if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
            // swap(12,13)
            {
                let tmp_434 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_434;
                let tmp_435 = values[12]; values[12] = values[13]; values[13] = tmp_435;
            }
        }
        // cmp_swap(14,15)
        if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
            // swap(14,15)
            {
                let tmp_436 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_436;
                let tmp_437 = values[14]; values[14] = values[15]; values[15] = tmp_437;
            }
        }
    }

    // exch_intxn(tmask:7,swbit:2,wpt:16)
    {
        // _exch_workgroup([(0, 15), (1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8), (8, 7), (9, 6), (10, 5), (11, 4), (12, 3), (13, 2), (14, 1), (15, 0)],7,2)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_438 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_439 = seg_base + (local_tid ^ 7u);
            let tmp_440 = smem_keys[tmp_439 * WPT + 15u];
            let tmp_441 = smem_vals[tmp_439 * WPT + 15u];
            let tmp_442 = keys[0] < tmp_440 || (keys[0] == tmp_440 && values[0] < tmp_441);
            if tmp_438 == tmp_442 { keys[0] = tmp_440; values[0] = tmp_441; }
            let tmp_443 = smem_keys[tmp_439 * WPT + 14u];
            let tmp_444 = smem_vals[tmp_439 * WPT + 14u];
            let tmp_445 = keys[1] < tmp_443 || (keys[1] == tmp_443 && values[1] < tmp_444);
            if tmp_438 == tmp_445 { keys[1] = tmp_443; values[1] = tmp_444; }
            let tmp_446 = smem_keys[tmp_439 * WPT + 13u];
            let tmp_447 = smem_vals[tmp_439 * WPT + 13u];
            let tmp_448 = keys[2] < tmp_446 || (keys[2] == tmp_446 && values[2] < tmp_447);
            if tmp_438 == tmp_448 { keys[2] = tmp_446; values[2] = tmp_447; }
            let tmp_449 = smem_keys[tmp_439 * WPT + 12u];
            let tmp_450 = smem_vals[tmp_439 * WPT + 12u];
            let tmp_451 = keys[3] < tmp_449 || (keys[3] == tmp_449 && values[3] < tmp_450);
            if tmp_438 == tmp_451 { keys[3] = tmp_449; values[3] = tmp_450; }
            let tmp_452 = smem_keys[tmp_439 * WPT + 11u];
            let tmp_453 = smem_vals[tmp_439 * WPT + 11u];
            let tmp_454 = keys[4] < tmp_452 || (keys[4] == tmp_452 && values[4] < tmp_453);
            if tmp_438 == tmp_454 { keys[4] = tmp_452; values[4] = tmp_453; }
            let tmp_455 = smem_keys[tmp_439 * WPT + 10u];
            let tmp_456 = smem_vals[tmp_439 * WPT + 10u];
            let tmp_457 = keys[5] < tmp_455 || (keys[5] == tmp_455 && values[5] < tmp_456);
            if tmp_438 == tmp_457 { keys[5] = tmp_455; values[5] = tmp_456; }
            let tmp_458 = smem_keys[tmp_439 * WPT + 9u];
            let tmp_459 = smem_vals[tmp_439 * WPT + 9u];
            let tmp_460 = keys[6] < tmp_458 || (keys[6] == tmp_458 && values[6] < tmp_459);
            if tmp_438 == tmp_460 { keys[6] = tmp_458; values[6] = tmp_459; }
            let tmp_461 = smem_keys[tmp_439 * WPT + 8u];
            let tmp_462 = smem_vals[tmp_439 * WPT + 8u];
            let tmp_463 = keys[7] < tmp_461 || (keys[7] == tmp_461 && values[7] < tmp_462);
            if tmp_438 == tmp_463 { keys[7] = tmp_461; values[7] = tmp_462; }
            let tmp_464 = smem_keys[tmp_439 * WPT + 7u];
            let tmp_465 = smem_vals[tmp_439 * WPT + 7u];
            let tmp_466 = keys[8] < tmp_464 || (keys[8] == tmp_464 && values[8] < tmp_465);
            if tmp_438 == tmp_466 { keys[8] = tmp_464; values[8] = tmp_465; }
            let tmp_467 = smem_keys[tmp_439 * WPT + 6u];
            let tmp_468 = smem_vals[tmp_439 * WPT + 6u];
            let tmp_469 = keys[9] < tmp_467 || (keys[9] == tmp_467 && values[9] < tmp_468);
            if tmp_438 == tmp_469 { keys[9] = tmp_467; values[9] = tmp_468; }
            let tmp_470 = smem_keys[tmp_439 * WPT + 5u];
            let tmp_471 = smem_vals[tmp_439 * WPT + 5u];
            let tmp_472 = keys[10] < tmp_470 || (keys[10] == tmp_470 && values[10] < tmp_471);
            if tmp_438 == tmp_472 { keys[10] = tmp_470; values[10] = tmp_471; }
            let tmp_473 = smem_keys[tmp_439 * WPT + 4u];
            let tmp_474 = smem_vals[tmp_439 * WPT + 4u];
            let tmp_475 = keys[11] < tmp_473 || (keys[11] == tmp_473 && values[11] < tmp_474);
            if tmp_438 == tmp_475 { keys[11] = tmp_473; values[11] = tmp_474; }
            let tmp_476 = smem_keys[tmp_439 * WPT + 3u];
            let tmp_477 = smem_vals[tmp_439 * WPT + 3u];
            let tmp_478 = keys[12] < tmp_476 || (keys[12] == tmp_476 && values[12] < tmp_477);
            if tmp_438 == tmp_478 { keys[12] = tmp_476; values[12] = tmp_477; }
            let tmp_479 = smem_keys[tmp_439 * WPT + 2u];
            let tmp_480 = smem_vals[tmp_439 * WPT + 2u];
            let tmp_481 = keys[13] < tmp_479 || (keys[13] == tmp_479 && values[13] < tmp_480);
            if tmp_438 == tmp_481 { keys[13] = tmp_479; values[13] = tmp_480; }
            let tmp_482 = smem_keys[tmp_439 * WPT + 1u];
            let tmp_483 = smem_vals[tmp_439 * WPT + 1u];
            let tmp_484 = keys[14] < tmp_482 || (keys[14] == tmp_482 && values[14] < tmp_483);
            if tmp_438 == tmp_484 { keys[14] = tmp_482; values[14] = tmp_483; }
            let tmp_485 = smem_keys[tmp_439 * WPT + 0u];
            let tmp_486 = smem_vals[tmp_439 * WPT + 0u];
            let tmp_487 = keys[15] < tmp_485 || (keys[15] == tmp_485 && values[15] < tmp_486);
            if tmp_438 == tmp_487 { keys[15] = tmp_485; values[15] = tmp_486; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],2,1)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_488 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_489 = seg_base + (local_tid ^ 2u);
            let tmp_490 = smem_keys[tmp_489 * WPT + 0u];
            let tmp_491 = smem_vals[tmp_489 * WPT + 0u];
            let tmp_492 = keys[0] < tmp_490 || (keys[0] == tmp_490 && values[0] < tmp_491);
            if tmp_488 == tmp_492 { keys[0] = tmp_490; values[0] = tmp_491; }
            let tmp_493 = smem_keys[tmp_489 * WPT + 1u];
            let tmp_494 = smem_vals[tmp_489 * WPT + 1u];
            let tmp_495 = keys[1] < tmp_493 || (keys[1] == tmp_493 && values[1] < tmp_494);
            if tmp_488 == tmp_495 { keys[1] = tmp_493; values[1] = tmp_494; }
            let tmp_496 = smem_keys[tmp_489 * WPT + 2u];
            let tmp_497 = smem_vals[tmp_489 * WPT + 2u];
            let tmp_498 = keys[2] < tmp_496 || (keys[2] == tmp_496 && values[2] < tmp_497);
            if tmp_488 == tmp_498 { keys[2] = tmp_496; values[2] = tmp_497; }
            let tmp_499 = smem_keys[tmp_489 * WPT + 3u];
            let tmp_500 = smem_vals[tmp_489 * WPT + 3u];
            let tmp_501 = keys[3] < tmp_499 || (keys[3] == tmp_499 && values[3] < tmp_500);
            if tmp_488 == tmp_501 { keys[3] = tmp_499; values[3] = tmp_500; }
            let tmp_502 = smem_keys[tmp_489 * WPT + 4u];
            let tmp_503 = smem_vals[tmp_489 * WPT + 4u];
            let tmp_504 = keys[4] < tmp_502 || (keys[4] == tmp_502 && values[4] < tmp_503);
            if tmp_488 == tmp_504 { keys[4] = tmp_502; values[4] = tmp_503; }
            let tmp_505 = smem_keys[tmp_489 * WPT + 5u];
            let tmp_506 = smem_vals[tmp_489 * WPT + 5u];
            let tmp_507 = keys[5] < tmp_505 || (keys[5] == tmp_505 && values[5] < tmp_506);
            if tmp_488 == tmp_507 { keys[5] = tmp_505; values[5] = tmp_506; }
            let tmp_508 = smem_keys[tmp_489 * WPT + 6u];
            let tmp_509 = smem_vals[tmp_489 * WPT + 6u];
            let tmp_510 = keys[6] < tmp_508 || (keys[6] == tmp_508 && values[6] < tmp_509);
            if tmp_488 == tmp_510 { keys[6] = tmp_508; values[6] = tmp_509; }
            let tmp_511 = smem_keys[tmp_489 * WPT + 7u];
            let tmp_512 = smem_vals[tmp_489 * WPT + 7u];
            let tmp_513 = keys[7] < tmp_511 || (keys[7] == tmp_511 && values[7] < tmp_512);
            if tmp_488 == tmp_513 { keys[7] = tmp_511; values[7] = tmp_512; }
            let tmp_514 = smem_keys[tmp_489 * WPT + 8u];
            let tmp_515 = smem_vals[tmp_489 * WPT + 8u];
            let tmp_516 = keys[8] < tmp_514 || (keys[8] == tmp_514 && values[8] < tmp_515);
            if tmp_488 == tmp_516 { keys[8] = tmp_514; values[8] = tmp_515; }
            let tmp_517 = smem_keys[tmp_489 * WPT + 9u];
            let tmp_518 = smem_vals[tmp_489 * WPT + 9u];
            let tmp_519 = keys[9] < tmp_517 || (keys[9] == tmp_517 && values[9] < tmp_518);
            if tmp_488 == tmp_519 { keys[9] = tmp_517; values[9] = tmp_518; }
            let tmp_520 = smem_keys[tmp_489 * WPT + 10u];
            let tmp_521 = smem_vals[tmp_489 * WPT + 10u];
            let tmp_522 = keys[10] < tmp_520 || (keys[10] == tmp_520 && values[10] < tmp_521);
            if tmp_488 == tmp_522 { keys[10] = tmp_520; values[10] = tmp_521; }
            let tmp_523 = smem_keys[tmp_489 * WPT + 11u];
            let tmp_524 = smem_vals[tmp_489 * WPT + 11u];
            let tmp_525 = keys[11] < tmp_523 || (keys[11] == tmp_523 && values[11] < tmp_524);
            if tmp_488 == tmp_525 { keys[11] = tmp_523; values[11] = tmp_524; }
            let tmp_526 = smem_keys[tmp_489 * WPT + 12u];
            let tmp_527 = smem_vals[tmp_489 * WPT + 12u];
            let tmp_528 = keys[12] < tmp_526 || (keys[12] == tmp_526 && values[12] < tmp_527);
            if tmp_488 == tmp_528 { keys[12] = tmp_526; values[12] = tmp_527; }
            let tmp_529 = smem_keys[tmp_489 * WPT + 13u];
            let tmp_530 = smem_vals[tmp_489 * WPT + 13u];
            let tmp_531 = keys[13] < tmp_529 || (keys[13] == tmp_529 && values[13] < tmp_530);
            if tmp_488 == tmp_531 { keys[13] = tmp_529; values[13] = tmp_530; }
            let tmp_532 = smem_keys[tmp_489 * WPT + 14u];
            let tmp_533 = smem_vals[tmp_489 * WPT + 14u];
            let tmp_534 = keys[14] < tmp_532 || (keys[14] == tmp_532 && values[14] < tmp_533);
            if tmp_488 == tmp_534 { keys[14] = tmp_532; values[14] = tmp_533; }
            let tmp_535 = smem_keys[tmp_489 * WPT + 15u];
            let tmp_536 = smem_vals[tmp_489 * WPT + 15u];
            let tmp_537 = keys[15] < tmp_535 || (keys[15] == tmp_535 && values[15] < tmp_536);
            if tmp_488 == tmp_537 { keys[15] = tmp_535; values[15] = tmp_536; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],1,0)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_538 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_539 = seg_base + (local_tid ^ 1u);
            let tmp_540 = smem_keys[tmp_539 * WPT + 0u];
            let tmp_541 = smem_vals[tmp_539 * WPT + 0u];
            let tmp_542 = keys[0] < tmp_540 || (keys[0] == tmp_540 && values[0] < tmp_541);
            if tmp_538 == tmp_542 { keys[0] = tmp_540; values[0] = tmp_541; }
            let tmp_543 = smem_keys[tmp_539 * WPT + 1u];
            let tmp_544 = smem_vals[tmp_539 * WPT + 1u];
            let tmp_545 = keys[1] < tmp_543 || (keys[1] == tmp_543 && values[1] < tmp_544);
            if tmp_538 == tmp_545 { keys[1] = tmp_543; values[1] = tmp_544; }
            let tmp_546 = smem_keys[tmp_539 * WPT + 2u];
            let tmp_547 = smem_vals[tmp_539 * WPT + 2u];
            let tmp_548 = keys[2] < tmp_546 || (keys[2] == tmp_546 && values[2] < tmp_547);
            if tmp_538 == tmp_548 { keys[2] = tmp_546; values[2] = tmp_547; }
            let tmp_549 = smem_keys[tmp_539 * WPT + 3u];
            let tmp_550 = smem_vals[tmp_539 * WPT + 3u];
            let tmp_551 = keys[3] < tmp_549 || (keys[3] == tmp_549 && values[3] < tmp_550);
            if tmp_538 == tmp_551 { keys[3] = tmp_549; values[3] = tmp_550; }
            let tmp_552 = smem_keys[tmp_539 * WPT + 4u];
            let tmp_553 = smem_vals[tmp_539 * WPT + 4u];
            let tmp_554 = keys[4] < tmp_552 || (keys[4] == tmp_552 && values[4] < tmp_553);
            if tmp_538 == tmp_554 { keys[4] = tmp_552; values[4] = tmp_553; }
            let tmp_555 = smem_keys[tmp_539 * WPT + 5u];
            let tmp_556 = smem_vals[tmp_539 * WPT + 5u];
            let tmp_557 = keys[5] < tmp_555 || (keys[5] == tmp_555 && values[5] < tmp_556);
            if tmp_538 == tmp_557 { keys[5] = tmp_555; values[5] = tmp_556; }
            let tmp_558 = smem_keys[tmp_539 * WPT + 6u];
            let tmp_559 = smem_vals[tmp_539 * WPT + 6u];
            let tmp_560 = keys[6] < tmp_558 || (keys[6] == tmp_558 && values[6] < tmp_559);
            if tmp_538 == tmp_560 { keys[6] = tmp_558; values[6] = tmp_559; }
            let tmp_561 = smem_keys[tmp_539 * WPT + 7u];
            let tmp_562 = smem_vals[tmp_539 * WPT + 7u];
            let tmp_563 = keys[7] < tmp_561 || (keys[7] == tmp_561 && values[7] < tmp_562);
            if tmp_538 == tmp_563 { keys[7] = tmp_561; values[7] = tmp_562; }
            let tmp_564 = smem_keys[tmp_539 * WPT + 8u];
            let tmp_565 = smem_vals[tmp_539 * WPT + 8u];
            let tmp_566 = keys[8] < tmp_564 || (keys[8] == tmp_564 && values[8] < tmp_565);
            if tmp_538 == tmp_566 { keys[8] = tmp_564; values[8] = tmp_565; }
            let tmp_567 = smem_keys[tmp_539 * WPT + 9u];
            let tmp_568 = smem_vals[tmp_539 * WPT + 9u];
            let tmp_569 = keys[9] < tmp_567 || (keys[9] == tmp_567 && values[9] < tmp_568);
            if tmp_538 == tmp_569 { keys[9] = tmp_567; values[9] = tmp_568; }
            let tmp_570 = smem_keys[tmp_539 * WPT + 10u];
            let tmp_571 = smem_vals[tmp_539 * WPT + 10u];
            let tmp_572 = keys[10] < tmp_570 || (keys[10] == tmp_570 && values[10] < tmp_571);
            if tmp_538 == tmp_572 { keys[10] = tmp_570; values[10] = tmp_571; }
            let tmp_573 = smem_keys[tmp_539 * WPT + 11u];
            let tmp_574 = smem_vals[tmp_539 * WPT + 11u];
            let tmp_575 = keys[11] < tmp_573 || (keys[11] == tmp_573 && values[11] < tmp_574);
            if tmp_538 == tmp_575 { keys[11] = tmp_573; values[11] = tmp_574; }
            let tmp_576 = smem_keys[tmp_539 * WPT + 12u];
            let tmp_577 = smem_vals[tmp_539 * WPT + 12u];
            let tmp_578 = keys[12] < tmp_576 || (keys[12] == tmp_576 && values[12] < tmp_577);
            if tmp_538 == tmp_578 { keys[12] = tmp_576; values[12] = tmp_577; }
            let tmp_579 = smem_keys[tmp_539 * WPT + 13u];
            let tmp_580 = smem_vals[tmp_539 * WPT + 13u];
            let tmp_581 = keys[13] < tmp_579 || (keys[13] == tmp_579 && values[13] < tmp_580);
            if tmp_538 == tmp_581 { keys[13] = tmp_579; values[13] = tmp_580; }
            let tmp_582 = smem_keys[tmp_539 * WPT + 14u];
            let tmp_583 = smem_vals[tmp_539 * WPT + 14u];
            let tmp_584 = keys[14] < tmp_582 || (keys[14] == tmp_582 && values[14] < tmp_583);
            if tmp_538 == tmp_584 { keys[14] = tmp_582; values[14] = tmp_583; }
            let tmp_585 = smem_keys[tmp_539 * WPT + 15u];
            let tmp_586 = smem_vals[tmp_539 * WPT + 15u];
            let tmp_587 = keys[15] < tmp_585 || (keys[15] == tmp_585 && values[15] < tmp_586);
            if tmp_538 == tmp_587 { keys[15] = tmp_585; values[15] = tmp_586; }
            workgroupBarrier();
        }
    }

    // exch_local(8,16)
    {
        // cmp_swap(0,8)
        if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
            // swap(0,8)
            {
                let tmp_588 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_588;
                let tmp_589 = values[0]; values[0] = values[8]; values[8] = tmp_589;
            }
        }
        // cmp_swap(1,9)
        if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
            // swap(1,9)
            {
                let tmp_590 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_590;
                let tmp_591 = values[1]; values[1] = values[9]; values[9] = tmp_591;
            }
        }
        // cmp_swap(2,10)
        if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
            // swap(2,10)
            {
                let tmp_592 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_592;
                let tmp_593 = values[2]; values[2] = values[10]; values[10] = tmp_593;
            }
        }
        // cmp_swap(3,11)
        if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
            // swap(3,11)
            {
                let tmp_594 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_594;
                let tmp_595 = values[3]; values[3] = values[11]; values[11] = tmp_595;
            }
        }
        // cmp_swap(4,12)
        if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
            // swap(4,12)
            {
                let tmp_596 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_596;
                let tmp_597 = values[4]; values[4] = values[12]; values[12] = tmp_597;
            }
        }
        // cmp_swap(5,13)
        if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
            // swap(5,13)
            {
                let tmp_598 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_598;
                let tmp_599 = values[5]; values[5] = values[13]; values[13] = tmp_599;
            }
        }
        // cmp_swap(6,14)
        if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
            // swap(6,14)
            {
                let tmp_600 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_600;
                let tmp_601 = values[6]; values[6] = values[14]; values[14] = tmp_601;
            }
        }
        // cmp_swap(7,15)
        if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
            // swap(7,15)
            {
                let tmp_602 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_602;
                let tmp_603 = values[7]; values[7] = values[15]; values[15] = tmp_603;
            }
        }
    }

    // exch_local(4,16)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_604 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_604;
                let tmp_605 = values[0]; values[0] = values[4]; values[4] = tmp_605;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_606 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_606;
                let tmp_607 = values[1]; values[1] = values[5]; values[5] = tmp_607;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_608 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_608;
                let tmp_609 = values[2]; values[2] = values[6]; values[6] = tmp_609;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_610 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_610;
                let tmp_611 = values[3]; values[3] = values[7]; values[7] = tmp_611;
            }
        }
        // cmp_swap(8,12)
        if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
            // swap(8,12)
            {
                let tmp_612 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_612;
                let tmp_613 = values[8]; values[8] = values[12]; values[12] = tmp_613;
            }
        }
        // cmp_swap(9,13)
        if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
            // swap(9,13)
            {
                let tmp_614 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_614;
                let tmp_615 = values[9]; values[9] = values[13]; values[13] = tmp_615;
            }
        }
        // cmp_swap(10,14)
        if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
            // swap(10,14)
            {
                let tmp_616 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_616;
                let tmp_617 = values[10]; values[10] = values[14]; values[14] = tmp_617;
            }
        }
        // cmp_swap(11,15)
        if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
            // swap(11,15)
            {
                let tmp_618 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_618;
                let tmp_619 = values[11]; values[11] = values[15]; values[15] = tmp_619;
            }
        }
    }

    // exch_local(2,16)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_620 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_620;
                let tmp_621 = values[0]; values[0] = values[2]; values[2] = tmp_621;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_622 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_622;
                let tmp_623 = values[1]; values[1] = values[3]; values[3] = tmp_623;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_624 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_624;
                let tmp_625 = values[4]; values[4] = values[6]; values[6] = tmp_625;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_626 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_626;
                let tmp_627 = values[5]; values[5] = values[7]; values[7] = tmp_627;
            }
        }
        // cmp_swap(8,10)
        if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
            // swap(8,10)
            {
                let tmp_628 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_628;
                let tmp_629 = values[8]; values[8] = values[10]; values[10] = tmp_629;
            }
        }
        // cmp_swap(9,11)
        if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
            // swap(9,11)
            {
                let tmp_630 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_630;
                let tmp_631 = values[9]; values[9] = values[11]; values[11] = tmp_631;
            }
        }
        // cmp_swap(12,14)
        if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
            // swap(12,14)
            {
                let tmp_632 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_632;
                let tmp_633 = values[12]; values[12] = values[14]; values[14] = tmp_633;
            }
        }
        // cmp_swap(13,15)
        if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
            // swap(13,15)
            {
                let tmp_634 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_634;
                let tmp_635 = values[13]; values[13] = values[15]; values[15] = tmp_635;
            }
        }
    }

    // exch_local(1,16)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_636 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_636;
                let tmp_637 = values[0]; values[0] = values[1]; values[1] = tmp_637;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_638 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_638;
                let tmp_639 = values[2]; values[2] = values[3]; values[3] = tmp_639;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_640 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_640;
                let tmp_641 = values[4]; values[4] = values[5]; values[5] = tmp_641;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_642 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_642;
                let tmp_643 = values[6]; values[6] = values[7]; values[7] = tmp_643;
            }
        }
        // cmp_swap(8,9)
        if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
            // swap(8,9)
            {
                let tmp_644 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_644;
                let tmp_645 = values[8]; values[8] = values[9]; values[9] = tmp_645;
            }
        }
        // cmp_swap(10,11)
        if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
            // swap(10,11)
            {
                let tmp_646 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_646;
                let tmp_647 = values[10]; values[10] = values[11]; values[11] = tmp_647;
            }
        }
        // cmp_swap(12,13)
        if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
            // swap(12,13)
            {
                let tmp_648 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_648;
                let tmp_649 = values[12]; values[12] = values[13]; values[13] = tmp_649;
            }
        }
        // cmp_swap(14,15)
        if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
            // swap(14,15)
            {
                let tmp_650 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_650;
                let tmp_651 = values[14]; values[14] = values[15]; values[15] = tmp_651;
            }
        }
    }

    // exch_intxn(tmask:15,swbit:3,wpt:16)
    {
        // _exch_workgroup([(0, 15), (1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8), (8, 7), (9, 6), (10, 5), (11, 4), (12, 3), (13, 2), (14, 1), (15, 0)],15,3)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_652 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_653 = seg_base + (local_tid ^ 15u);
            let tmp_654 = smem_keys[tmp_653 * WPT + 15u];
            let tmp_655 = smem_vals[tmp_653 * WPT + 15u];
            let tmp_656 = keys[0] < tmp_654 || (keys[0] == tmp_654 && values[0] < tmp_655);
            if tmp_652 == tmp_656 { keys[0] = tmp_654; values[0] = tmp_655; }
            let tmp_657 = smem_keys[tmp_653 * WPT + 14u];
            let tmp_658 = smem_vals[tmp_653 * WPT + 14u];
            let tmp_659 = keys[1] < tmp_657 || (keys[1] == tmp_657 && values[1] < tmp_658);
            if tmp_652 == tmp_659 { keys[1] = tmp_657; values[1] = tmp_658; }
            let tmp_660 = smem_keys[tmp_653 * WPT + 13u];
            let tmp_661 = smem_vals[tmp_653 * WPT + 13u];
            let tmp_662 = keys[2] < tmp_660 || (keys[2] == tmp_660 && values[2] < tmp_661);
            if tmp_652 == tmp_662 { keys[2] = tmp_660; values[2] = tmp_661; }
            let tmp_663 = smem_keys[tmp_653 * WPT + 12u];
            let tmp_664 = smem_vals[tmp_653 * WPT + 12u];
            let tmp_665 = keys[3] < tmp_663 || (keys[3] == tmp_663 && values[3] < tmp_664);
            if tmp_652 == tmp_665 { keys[3] = tmp_663; values[3] = tmp_664; }
            let tmp_666 = smem_keys[tmp_653 * WPT + 11u];
            let tmp_667 = smem_vals[tmp_653 * WPT + 11u];
            let tmp_668 = keys[4] < tmp_666 || (keys[4] == tmp_666 && values[4] < tmp_667);
            if tmp_652 == tmp_668 { keys[4] = tmp_666; values[4] = tmp_667; }
            let tmp_669 = smem_keys[tmp_653 * WPT + 10u];
            let tmp_670 = smem_vals[tmp_653 * WPT + 10u];
            let tmp_671 = keys[5] < tmp_669 || (keys[5] == tmp_669 && values[5] < tmp_670);
            if tmp_652 == tmp_671 { keys[5] = tmp_669; values[5] = tmp_670; }
            let tmp_672 = smem_keys[tmp_653 * WPT + 9u];
            let tmp_673 = smem_vals[tmp_653 * WPT + 9u];
            let tmp_674 = keys[6] < tmp_672 || (keys[6] == tmp_672 && values[6] < tmp_673);
            if tmp_652 == tmp_674 { keys[6] = tmp_672; values[6] = tmp_673; }
            let tmp_675 = smem_keys[tmp_653 * WPT + 8u];
            let tmp_676 = smem_vals[tmp_653 * WPT + 8u];
            let tmp_677 = keys[7] < tmp_675 || (keys[7] == tmp_675 && values[7] < tmp_676);
            if tmp_652 == tmp_677 { keys[7] = tmp_675; values[7] = tmp_676; }
            let tmp_678 = smem_keys[tmp_653 * WPT + 7u];
            let tmp_679 = smem_vals[tmp_653 * WPT + 7u];
            let tmp_680 = keys[8] < tmp_678 || (keys[8] == tmp_678 && values[8] < tmp_679);
            if tmp_652 == tmp_680 { keys[8] = tmp_678; values[8] = tmp_679; }
            let tmp_681 = smem_keys[tmp_653 * WPT + 6u];
            let tmp_682 = smem_vals[tmp_653 * WPT + 6u];
            let tmp_683 = keys[9] < tmp_681 || (keys[9] == tmp_681 && values[9] < tmp_682);
            if tmp_652 == tmp_683 { keys[9] = tmp_681; values[9] = tmp_682; }
            let tmp_684 = smem_keys[tmp_653 * WPT + 5u];
            let tmp_685 = smem_vals[tmp_653 * WPT + 5u];
            let tmp_686 = keys[10] < tmp_684 || (keys[10] == tmp_684 && values[10] < tmp_685);
            if tmp_652 == tmp_686 { keys[10] = tmp_684; values[10] = tmp_685; }
            let tmp_687 = smem_keys[tmp_653 * WPT + 4u];
            let tmp_688 = smem_vals[tmp_653 * WPT + 4u];
            let tmp_689 = keys[11] < tmp_687 || (keys[11] == tmp_687 && values[11] < tmp_688);
            if tmp_652 == tmp_689 { keys[11] = tmp_687; values[11] = tmp_688; }
            let tmp_690 = smem_keys[tmp_653 * WPT + 3u];
            let tmp_691 = smem_vals[tmp_653 * WPT + 3u];
            let tmp_692 = keys[12] < tmp_690 || (keys[12] == tmp_690 && values[12] < tmp_691);
            if tmp_652 == tmp_692 { keys[12] = tmp_690; values[12] = tmp_691; }
            let tmp_693 = smem_keys[tmp_653 * WPT + 2u];
            let tmp_694 = smem_vals[tmp_653 * WPT + 2u];
            let tmp_695 = keys[13] < tmp_693 || (keys[13] == tmp_693 && values[13] < tmp_694);
            if tmp_652 == tmp_695 { keys[13] = tmp_693; values[13] = tmp_694; }
            let tmp_696 = smem_keys[tmp_653 * WPT + 1u];
            let tmp_697 = smem_vals[tmp_653 * WPT + 1u];
            let tmp_698 = keys[14] < tmp_696 || (keys[14] == tmp_696 && values[14] < tmp_697);
            if tmp_652 == tmp_698 { keys[14] = tmp_696; values[14] = tmp_697; }
            let tmp_699 = smem_keys[tmp_653 * WPT + 0u];
            let tmp_700 = smem_vals[tmp_653 * WPT + 0u];
            let tmp_701 = keys[15] < tmp_699 || (keys[15] == tmp_699 && values[15] < tmp_700);
            if tmp_652 == tmp_701 { keys[15] = tmp_699; values[15] = tmp_700; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],4,2)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_702 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_703 = seg_base + (local_tid ^ 4u);
            let tmp_704 = smem_keys[tmp_703 * WPT + 0u];
            let tmp_705 = smem_vals[tmp_703 * WPT + 0u];
            let tmp_706 = keys[0] < tmp_704 || (keys[0] == tmp_704 && values[0] < tmp_705);
            if tmp_702 == tmp_706 { keys[0] = tmp_704; values[0] = tmp_705; }
            let tmp_707 = smem_keys[tmp_703 * WPT + 1u];
            let tmp_708 = smem_vals[tmp_703 * WPT + 1u];
            let tmp_709 = keys[1] < tmp_707 || (keys[1] == tmp_707 && values[1] < tmp_708);
            if tmp_702 == tmp_709 { keys[1] = tmp_707; values[1] = tmp_708; }
            let tmp_710 = smem_keys[tmp_703 * WPT + 2u];
            let tmp_711 = smem_vals[tmp_703 * WPT + 2u];
            let tmp_712 = keys[2] < tmp_710 || (keys[2] == tmp_710 && values[2] < tmp_711);
            if tmp_702 == tmp_712 { keys[2] = tmp_710; values[2] = tmp_711; }
            let tmp_713 = smem_keys[tmp_703 * WPT + 3u];
            let tmp_714 = smem_vals[tmp_703 * WPT + 3u];
            let tmp_715 = keys[3] < tmp_713 || (keys[3] == tmp_713 && values[3] < tmp_714);
            if tmp_702 == tmp_715 { keys[3] = tmp_713; values[3] = tmp_714; }
            let tmp_716 = smem_keys[tmp_703 * WPT + 4u];
            let tmp_717 = smem_vals[tmp_703 * WPT + 4u];
            let tmp_718 = keys[4] < tmp_716 || (keys[4] == tmp_716 && values[4] < tmp_717);
            if tmp_702 == tmp_718 { keys[4] = tmp_716; values[4] = tmp_717; }
            let tmp_719 = smem_keys[tmp_703 * WPT + 5u];
            let tmp_720 = smem_vals[tmp_703 * WPT + 5u];
            let tmp_721 = keys[5] < tmp_719 || (keys[5] == tmp_719 && values[5] < tmp_720);
            if tmp_702 == tmp_721 { keys[5] = tmp_719; values[5] = tmp_720; }
            let tmp_722 = smem_keys[tmp_703 * WPT + 6u];
            let tmp_723 = smem_vals[tmp_703 * WPT + 6u];
            let tmp_724 = keys[6] < tmp_722 || (keys[6] == tmp_722 && values[6] < tmp_723);
            if tmp_702 == tmp_724 { keys[6] = tmp_722; values[6] = tmp_723; }
            let tmp_725 = smem_keys[tmp_703 * WPT + 7u];
            let tmp_726 = smem_vals[tmp_703 * WPT + 7u];
            let tmp_727 = keys[7] < tmp_725 || (keys[7] == tmp_725 && values[7] < tmp_726);
            if tmp_702 == tmp_727 { keys[7] = tmp_725; values[7] = tmp_726; }
            let tmp_728 = smem_keys[tmp_703 * WPT + 8u];
            let tmp_729 = smem_vals[tmp_703 * WPT + 8u];
            let tmp_730 = keys[8] < tmp_728 || (keys[8] == tmp_728 && values[8] < tmp_729);
            if tmp_702 == tmp_730 { keys[8] = tmp_728; values[8] = tmp_729; }
            let tmp_731 = smem_keys[tmp_703 * WPT + 9u];
            let tmp_732 = smem_vals[tmp_703 * WPT + 9u];
            let tmp_733 = keys[9] < tmp_731 || (keys[9] == tmp_731 && values[9] < tmp_732);
            if tmp_702 == tmp_733 { keys[9] = tmp_731; values[9] = tmp_732; }
            let tmp_734 = smem_keys[tmp_703 * WPT + 10u];
            let tmp_735 = smem_vals[tmp_703 * WPT + 10u];
            let tmp_736 = keys[10] < tmp_734 || (keys[10] == tmp_734 && values[10] < tmp_735);
            if tmp_702 == tmp_736 { keys[10] = tmp_734; values[10] = tmp_735; }
            let tmp_737 = smem_keys[tmp_703 * WPT + 11u];
            let tmp_738 = smem_vals[tmp_703 * WPT + 11u];
            let tmp_739 = keys[11] < tmp_737 || (keys[11] == tmp_737 && values[11] < tmp_738);
            if tmp_702 == tmp_739 { keys[11] = tmp_737; values[11] = tmp_738; }
            let tmp_740 = smem_keys[tmp_703 * WPT + 12u];
            let tmp_741 = smem_vals[tmp_703 * WPT + 12u];
            let tmp_742 = keys[12] < tmp_740 || (keys[12] == tmp_740 && values[12] < tmp_741);
            if tmp_702 == tmp_742 { keys[12] = tmp_740; values[12] = tmp_741; }
            let tmp_743 = smem_keys[tmp_703 * WPT + 13u];
            let tmp_744 = smem_vals[tmp_703 * WPT + 13u];
            let tmp_745 = keys[13] < tmp_743 || (keys[13] == tmp_743 && values[13] < tmp_744);
            if tmp_702 == tmp_745 { keys[13] = tmp_743; values[13] = tmp_744; }
            let tmp_746 = smem_keys[tmp_703 * WPT + 14u];
            let tmp_747 = smem_vals[tmp_703 * WPT + 14u];
            let tmp_748 = keys[14] < tmp_746 || (keys[14] == tmp_746 && values[14] < tmp_747);
            if tmp_702 == tmp_748 { keys[14] = tmp_746; values[14] = tmp_747; }
            let tmp_749 = smem_keys[tmp_703 * WPT + 15u];
            let tmp_750 = smem_vals[tmp_703 * WPT + 15u];
            let tmp_751 = keys[15] < tmp_749 || (keys[15] == tmp_749 && values[15] < tmp_750);
            if tmp_702 == tmp_751 { keys[15] = tmp_749; values[15] = tmp_750; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],2,1)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_752 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_753 = seg_base + (local_tid ^ 2u);
            let tmp_754 = smem_keys[tmp_753 * WPT + 0u];
            let tmp_755 = smem_vals[tmp_753 * WPT + 0u];
            let tmp_756 = keys[0] < tmp_754 || (keys[0] == tmp_754 && values[0] < tmp_755);
            if tmp_752 == tmp_756 { keys[0] = tmp_754; values[0] = tmp_755; }
            let tmp_757 = smem_keys[tmp_753 * WPT + 1u];
            let tmp_758 = smem_vals[tmp_753 * WPT + 1u];
            let tmp_759 = keys[1] < tmp_757 || (keys[1] == tmp_757 && values[1] < tmp_758);
            if tmp_752 == tmp_759 { keys[1] = tmp_757; values[1] = tmp_758; }
            let tmp_760 = smem_keys[tmp_753 * WPT + 2u];
            let tmp_761 = smem_vals[tmp_753 * WPT + 2u];
            let tmp_762 = keys[2] < tmp_760 || (keys[2] == tmp_760 && values[2] < tmp_761);
            if tmp_752 == tmp_762 { keys[2] = tmp_760; values[2] = tmp_761; }
            let tmp_763 = smem_keys[tmp_753 * WPT + 3u];
            let tmp_764 = smem_vals[tmp_753 * WPT + 3u];
            let tmp_765 = keys[3] < tmp_763 || (keys[3] == tmp_763 && values[3] < tmp_764);
            if tmp_752 == tmp_765 { keys[3] = tmp_763; values[3] = tmp_764; }
            let tmp_766 = smem_keys[tmp_753 * WPT + 4u];
            let tmp_767 = smem_vals[tmp_753 * WPT + 4u];
            let tmp_768 = keys[4] < tmp_766 || (keys[4] == tmp_766 && values[4] < tmp_767);
            if tmp_752 == tmp_768 { keys[4] = tmp_766; values[4] = tmp_767; }
            let tmp_769 = smem_keys[tmp_753 * WPT + 5u];
            let tmp_770 = smem_vals[tmp_753 * WPT + 5u];
            let tmp_771 = keys[5] < tmp_769 || (keys[5] == tmp_769 && values[5] < tmp_770);
            if tmp_752 == tmp_771 { keys[5] = tmp_769; values[5] = tmp_770; }
            let tmp_772 = smem_keys[tmp_753 * WPT + 6u];
            let tmp_773 = smem_vals[tmp_753 * WPT + 6u];
            let tmp_774 = keys[6] < tmp_772 || (keys[6] == tmp_772 && values[6] < tmp_773);
            if tmp_752 == tmp_774 { keys[6] = tmp_772; values[6] = tmp_773; }
            let tmp_775 = smem_keys[tmp_753 * WPT + 7u];
            let tmp_776 = smem_vals[tmp_753 * WPT + 7u];
            let tmp_777 = keys[7] < tmp_775 || (keys[7] == tmp_775 && values[7] < tmp_776);
            if tmp_752 == tmp_777 { keys[7] = tmp_775; values[7] = tmp_776; }
            let tmp_778 = smem_keys[tmp_753 * WPT + 8u];
            let tmp_779 = smem_vals[tmp_753 * WPT + 8u];
            let tmp_780 = keys[8] < tmp_778 || (keys[8] == tmp_778 && values[8] < tmp_779);
            if tmp_752 == tmp_780 { keys[8] = tmp_778; values[8] = tmp_779; }
            let tmp_781 = smem_keys[tmp_753 * WPT + 9u];
            let tmp_782 = smem_vals[tmp_753 * WPT + 9u];
            let tmp_783 = keys[9] < tmp_781 || (keys[9] == tmp_781 && values[9] < tmp_782);
            if tmp_752 == tmp_783 { keys[9] = tmp_781; values[9] = tmp_782; }
            let tmp_784 = smem_keys[tmp_753 * WPT + 10u];
            let tmp_785 = smem_vals[tmp_753 * WPT + 10u];
            let tmp_786 = keys[10] < tmp_784 || (keys[10] == tmp_784 && values[10] < tmp_785);
            if tmp_752 == tmp_786 { keys[10] = tmp_784; values[10] = tmp_785; }
            let tmp_787 = smem_keys[tmp_753 * WPT + 11u];
            let tmp_788 = smem_vals[tmp_753 * WPT + 11u];
            let tmp_789 = keys[11] < tmp_787 || (keys[11] == tmp_787 && values[11] < tmp_788);
            if tmp_752 == tmp_789 { keys[11] = tmp_787; values[11] = tmp_788; }
            let tmp_790 = smem_keys[tmp_753 * WPT + 12u];
            let tmp_791 = smem_vals[tmp_753 * WPT + 12u];
            let tmp_792 = keys[12] < tmp_790 || (keys[12] == tmp_790 && values[12] < tmp_791);
            if tmp_752 == tmp_792 { keys[12] = tmp_790; values[12] = tmp_791; }
            let tmp_793 = smem_keys[tmp_753 * WPT + 13u];
            let tmp_794 = smem_vals[tmp_753 * WPT + 13u];
            let tmp_795 = keys[13] < tmp_793 || (keys[13] == tmp_793 && values[13] < tmp_794);
            if tmp_752 == tmp_795 { keys[13] = tmp_793; values[13] = tmp_794; }
            let tmp_796 = smem_keys[tmp_753 * WPT + 14u];
            let tmp_797 = smem_vals[tmp_753 * WPT + 14u];
            let tmp_798 = keys[14] < tmp_796 || (keys[14] == tmp_796 && values[14] < tmp_797);
            if tmp_752 == tmp_798 { keys[14] = tmp_796; values[14] = tmp_797; }
            let tmp_799 = smem_keys[tmp_753 * WPT + 15u];
            let tmp_800 = smem_vals[tmp_753 * WPT + 15u];
            let tmp_801 = keys[15] < tmp_799 || (keys[15] == tmp_799 && values[15] < tmp_800);
            if tmp_752 == tmp_801 { keys[15] = tmp_799; values[15] = tmp_800; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],1,0)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_802 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_803 = seg_base + (local_tid ^ 1u);
            let tmp_804 = smem_keys[tmp_803 * WPT + 0u];
            let tmp_805 = smem_vals[tmp_803 * WPT + 0u];
            let tmp_806 = keys[0] < tmp_804 || (keys[0] == tmp_804 && values[0] < tmp_805);
            if tmp_802 == tmp_806 { keys[0] = tmp_804; values[0] = tmp_805; }
            let tmp_807 = smem_keys[tmp_803 * WPT + 1u];
            let tmp_808 = smem_vals[tmp_803 * WPT + 1u];
            let tmp_809 = keys[1] < tmp_807 || (keys[1] == tmp_807 && values[1] < tmp_808);
            if tmp_802 == tmp_809 { keys[1] = tmp_807; values[1] = tmp_808; }
            let tmp_810 = smem_keys[tmp_803 * WPT + 2u];
            let tmp_811 = smem_vals[tmp_803 * WPT + 2u];
            let tmp_812 = keys[2] < tmp_810 || (keys[2] == tmp_810 && values[2] < tmp_811);
            if tmp_802 == tmp_812 { keys[2] = tmp_810; values[2] = tmp_811; }
            let tmp_813 = smem_keys[tmp_803 * WPT + 3u];
            let tmp_814 = smem_vals[tmp_803 * WPT + 3u];
            let tmp_815 = keys[3] < tmp_813 || (keys[3] == tmp_813 && values[3] < tmp_814);
            if tmp_802 == tmp_815 { keys[3] = tmp_813; values[3] = tmp_814; }
            let tmp_816 = smem_keys[tmp_803 * WPT + 4u];
            let tmp_817 = smem_vals[tmp_803 * WPT + 4u];
            let tmp_818 = keys[4] < tmp_816 || (keys[4] == tmp_816 && values[4] < tmp_817);
            if tmp_802 == tmp_818 { keys[4] = tmp_816; values[4] = tmp_817; }
            let tmp_819 = smem_keys[tmp_803 * WPT + 5u];
            let tmp_820 = smem_vals[tmp_803 * WPT + 5u];
            let tmp_821 = keys[5] < tmp_819 || (keys[5] == tmp_819 && values[5] < tmp_820);
            if tmp_802 == tmp_821 { keys[5] = tmp_819; values[5] = tmp_820; }
            let tmp_822 = smem_keys[tmp_803 * WPT + 6u];
            let tmp_823 = smem_vals[tmp_803 * WPT + 6u];
            let tmp_824 = keys[6] < tmp_822 || (keys[6] == tmp_822 && values[6] < tmp_823);
            if tmp_802 == tmp_824 { keys[6] = tmp_822; values[6] = tmp_823; }
            let tmp_825 = smem_keys[tmp_803 * WPT + 7u];
            let tmp_826 = smem_vals[tmp_803 * WPT + 7u];
            let tmp_827 = keys[7] < tmp_825 || (keys[7] == tmp_825 && values[7] < tmp_826);
            if tmp_802 == tmp_827 { keys[7] = tmp_825; values[7] = tmp_826; }
            let tmp_828 = smem_keys[tmp_803 * WPT + 8u];
            let tmp_829 = smem_vals[tmp_803 * WPT + 8u];
            let tmp_830 = keys[8] < tmp_828 || (keys[8] == tmp_828 && values[8] < tmp_829);
            if tmp_802 == tmp_830 { keys[8] = tmp_828; values[8] = tmp_829; }
            let tmp_831 = smem_keys[tmp_803 * WPT + 9u];
            let tmp_832 = smem_vals[tmp_803 * WPT + 9u];
            let tmp_833 = keys[9] < tmp_831 || (keys[9] == tmp_831 && values[9] < tmp_832);
            if tmp_802 == tmp_833 { keys[9] = tmp_831; values[9] = tmp_832; }
            let tmp_834 = smem_keys[tmp_803 * WPT + 10u];
            let tmp_835 = smem_vals[tmp_803 * WPT + 10u];
            let tmp_836 = keys[10] < tmp_834 || (keys[10] == tmp_834 && values[10] < tmp_835);
            if tmp_802 == tmp_836 { keys[10] = tmp_834; values[10] = tmp_835; }
            let tmp_837 = smem_keys[tmp_803 * WPT + 11u];
            let tmp_838 = smem_vals[tmp_803 * WPT + 11u];
            let tmp_839 = keys[11] < tmp_837 || (keys[11] == tmp_837 && values[11] < tmp_838);
            if tmp_802 == tmp_839 { keys[11] = tmp_837; values[11] = tmp_838; }
            let tmp_840 = smem_keys[tmp_803 * WPT + 12u];
            let tmp_841 = smem_vals[tmp_803 * WPT + 12u];
            let tmp_842 = keys[12] < tmp_840 || (keys[12] == tmp_840 && values[12] < tmp_841);
            if tmp_802 == tmp_842 { keys[12] = tmp_840; values[12] = tmp_841; }
            let tmp_843 = smem_keys[tmp_803 * WPT + 13u];
            let tmp_844 = smem_vals[tmp_803 * WPT + 13u];
            let tmp_845 = keys[13] < tmp_843 || (keys[13] == tmp_843 && values[13] < tmp_844);
            if tmp_802 == tmp_845 { keys[13] = tmp_843; values[13] = tmp_844; }
            let tmp_846 = smem_keys[tmp_803 * WPT + 14u];
            let tmp_847 = smem_vals[tmp_803 * WPT + 14u];
            let tmp_848 = keys[14] < tmp_846 || (keys[14] == tmp_846 && values[14] < tmp_847);
            if tmp_802 == tmp_848 { keys[14] = tmp_846; values[14] = tmp_847; }
            let tmp_849 = smem_keys[tmp_803 * WPT + 15u];
            let tmp_850 = smem_vals[tmp_803 * WPT + 15u];
            let tmp_851 = keys[15] < tmp_849 || (keys[15] == tmp_849 && values[15] < tmp_850);
            if tmp_802 == tmp_851 { keys[15] = tmp_849; values[15] = tmp_850; }
            workgroupBarrier();
        }
    }

    // exch_local(8,16)
    {
        // cmp_swap(0,8)
        if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
            // swap(0,8)
            {
                let tmp_852 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_852;
                let tmp_853 = values[0]; values[0] = values[8]; values[8] = tmp_853;
            }
        }
        // cmp_swap(1,9)
        if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
            // swap(1,9)
            {
                let tmp_854 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_854;
                let tmp_855 = values[1]; values[1] = values[9]; values[9] = tmp_855;
            }
        }
        // cmp_swap(2,10)
        if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
            // swap(2,10)
            {
                let tmp_856 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_856;
                let tmp_857 = values[2]; values[2] = values[10]; values[10] = tmp_857;
            }
        }
        // cmp_swap(3,11)
        if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
            // swap(3,11)
            {
                let tmp_858 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_858;
                let tmp_859 = values[3]; values[3] = values[11]; values[11] = tmp_859;
            }
        }
        // cmp_swap(4,12)
        if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
            // swap(4,12)
            {
                let tmp_860 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_860;
                let tmp_861 = values[4]; values[4] = values[12]; values[12] = tmp_861;
            }
        }
        // cmp_swap(5,13)
        if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
            // swap(5,13)
            {
                let tmp_862 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_862;
                let tmp_863 = values[5]; values[5] = values[13]; values[13] = tmp_863;
            }
        }
        // cmp_swap(6,14)
        if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
            // swap(6,14)
            {
                let tmp_864 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_864;
                let tmp_865 = values[6]; values[6] = values[14]; values[14] = tmp_865;
            }
        }
        // cmp_swap(7,15)
        if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
            // swap(7,15)
            {
                let tmp_866 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_866;
                let tmp_867 = values[7]; values[7] = values[15]; values[15] = tmp_867;
            }
        }
    }

    // exch_local(4,16)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_868 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_868;
                let tmp_869 = values[0]; values[0] = values[4]; values[4] = tmp_869;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_870 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_870;
                let tmp_871 = values[1]; values[1] = values[5]; values[5] = tmp_871;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_872 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_872;
                let tmp_873 = values[2]; values[2] = values[6]; values[6] = tmp_873;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_874 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_874;
                let tmp_875 = values[3]; values[3] = values[7]; values[7] = tmp_875;
            }
        }
        // cmp_swap(8,12)
        if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
            // swap(8,12)
            {
                let tmp_876 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_876;
                let tmp_877 = values[8]; values[8] = values[12]; values[12] = tmp_877;
            }
        }
        // cmp_swap(9,13)
        if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
            // swap(9,13)
            {
                let tmp_878 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_878;
                let tmp_879 = values[9]; values[9] = values[13]; values[13] = tmp_879;
            }
        }
        // cmp_swap(10,14)
        if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
            // swap(10,14)
            {
                let tmp_880 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_880;
                let tmp_881 = values[10]; values[10] = values[14]; values[14] = tmp_881;
            }
        }
        // cmp_swap(11,15)
        if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
            // swap(11,15)
            {
                let tmp_882 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_882;
                let tmp_883 = values[11]; values[11] = values[15]; values[15] = tmp_883;
            }
        }
    }

    // exch_local(2,16)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_884 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_884;
                let tmp_885 = values[0]; values[0] = values[2]; values[2] = tmp_885;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_886 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_886;
                let tmp_887 = values[1]; values[1] = values[3]; values[3] = tmp_887;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_888 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_888;
                let tmp_889 = values[4]; values[4] = values[6]; values[6] = tmp_889;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_890 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_890;
                let tmp_891 = values[5]; values[5] = values[7]; values[7] = tmp_891;
            }
        }
        // cmp_swap(8,10)
        if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
            // swap(8,10)
            {
                let tmp_892 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_892;
                let tmp_893 = values[8]; values[8] = values[10]; values[10] = tmp_893;
            }
        }
        // cmp_swap(9,11)
        if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
            // swap(9,11)
            {
                let tmp_894 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_894;
                let tmp_895 = values[9]; values[9] = values[11]; values[11] = tmp_895;
            }
        }
        // cmp_swap(12,14)
        if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
            // swap(12,14)
            {
                let tmp_896 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_896;
                let tmp_897 = values[12]; values[12] = values[14]; values[14] = tmp_897;
            }
        }
        // cmp_swap(13,15)
        if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
            // swap(13,15)
            {
                let tmp_898 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_898;
                let tmp_899 = values[13]; values[13] = values[15]; values[15] = tmp_899;
            }
        }
    }

    // exch_local(1,16)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_900 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_900;
                let tmp_901 = values[0]; values[0] = values[1]; values[1] = tmp_901;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_902 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_902;
                let tmp_903 = values[2]; values[2] = values[3]; values[3] = tmp_903;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_904 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_904;
                let tmp_905 = values[4]; values[4] = values[5]; values[5] = tmp_905;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_906 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_906;
                let tmp_907 = values[6]; values[6] = values[7]; values[7] = tmp_907;
            }
        }
        // cmp_swap(8,9)
        if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
            // swap(8,9)
            {
                let tmp_908 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_908;
                let tmp_909 = values[8]; values[8] = values[9]; values[9] = tmp_909;
            }
        }
        // cmp_swap(10,11)
        if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
            // swap(10,11)
            {
                let tmp_910 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_910;
                let tmp_911 = values[10]; values[10] = values[11]; values[11] = tmp_911;
            }
        }
        // cmp_swap(12,13)
        if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
            // swap(12,13)
            {
                let tmp_912 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_912;
                let tmp_913 = values[12]; values[12] = values[13]; values[13] = tmp_913;
            }
        }
        // cmp_swap(14,15)
        if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
            // swap(14,15)
            {
                let tmp_914 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_914;
                let tmp_915 = values[14]; values[14] = values[15]; values[15] = tmp_915;
            }
        }
    }

    // exch_intxn(tmask:31,swbit:4,wpt:16)
    {
        // _exch_workgroup([(0, 15), (1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8), (8, 7), (9, 6), (10, 5), (11, 4), (12, 3), (13, 2), (14, 1), (15, 0)],31,4)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_916 = extractBits(local_tid, 4u, 1u) != 0u;
            let tmp_917 = seg_base + (local_tid ^ 31u);
            let tmp_918 = smem_keys[tmp_917 * WPT + 15u];
            let tmp_919 = smem_vals[tmp_917 * WPT + 15u];
            let tmp_920 = keys[0] < tmp_918 || (keys[0] == tmp_918 && values[0] < tmp_919);
            if tmp_916 == tmp_920 { keys[0] = tmp_918; values[0] = tmp_919; }
            let tmp_921 = smem_keys[tmp_917 * WPT + 14u];
            let tmp_922 = smem_vals[tmp_917 * WPT + 14u];
            let tmp_923 = keys[1] < tmp_921 || (keys[1] == tmp_921 && values[1] < tmp_922);
            if tmp_916 == tmp_923 { keys[1] = tmp_921; values[1] = tmp_922; }
            let tmp_924 = smem_keys[tmp_917 * WPT + 13u];
            let tmp_925 = smem_vals[tmp_917 * WPT + 13u];
            let tmp_926 = keys[2] < tmp_924 || (keys[2] == tmp_924 && values[2] < tmp_925);
            if tmp_916 == tmp_926 { keys[2] = tmp_924; values[2] = tmp_925; }
            let tmp_927 = smem_keys[tmp_917 * WPT + 12u];
            let tmp_928 = smem_vals[tmp_917 * WPT + 12u];
            let tmp_929 = keys[3] < tmp_927 || (keys[3] == tmp_927 && values[3] < tmp_928);
            if tmp_916 == tmp_929 { keys[3] = tmp_927; values[3] = tmp_928; }
            let tmp_930 = smem_keys[tmp_917 * WPT + 11u];
            let tmp_931 = smem_vals[tmp_917 * WPT + 11u];
            let tmp_932 = keys[4] < tmp_930 || (keys[4] == tmp_930 && values[4] < tmp_931);
            if tmp_916 == tmp_932 { keys[4] = tmp_930; values[4] = tmp_931; }
            let tmp_933 = smem_keys[tmp_917 * WPT + 10u];
            let tmp_934 = smem_vals[tmp_917 * WPT + 10u];
            let tmp_935 = keys[5] < tmp_933 || (keys[5] == tmp_933 && values[5] < tmp_934);
            if tmp_916 == tmp_935 { keys[5] = tmp_933; values[5] = tmp_934; }
            let tmp_936 = smem_keys[tmp_917 * WPT + 9u];
            let tmp_937 = smem_vals[tmp_917 * WPT + 9u];
            let tmp_938 = keys[6] < tmp_936 || (keys[6] == tmp_936 && values[6] < tmp_937);
            if tmp_916 == tmp_938 { keys[6] = tmp_936; values[6] = tmp_937; }
            let tmp_939 = smem_keys[tmp_917 * WPT + 8u];
            let tmp_940 = smem_vals[tmp_917 * WPT + 8u];
            let tmp_941 = keys[7] < tmp_939 || (keys[7] == tmp_939 && values[7] < tmp_940);
            if tmp_916 == tmp_941 { keys[7] = tmp_939; values[7] = tmp_940; }
            let tmp_942 = smem_keys[tmp_917 * WPT + 7u];
            let tmp_943 = smem_vals[tmp_917 * WPT + 7u];
            let tmp_944 = keys[8] < tmp_942 || (keys[8] == tmp_942 && values[8] < tmp_943);
            if tmp_916 == tmp_944 { keys[8] = tmp_942; values[8] = tmp_943; }
            let tmp_945 = smem_keys[tmp_917 * WPT + 6u];
            let tmp_946 = smem_vals[tmp_917 * WPT + 6u];
            let tmp_947 = keys[9] < tmp_945 || (keys[9] == tmp_945 && values[9] < tmp_946);
            if tmp_916 == tmp_947 { keys[9] = tmp_945; values[9] = tmp_946; }
            let tmp_948 = smem_keys[tmp_917 * WPT + 5u];
            let tmp_949 = smem_vals[tmp_917 * WPT + 5u];
            let tmp_950 = keys[10] < tmp_948 || (keys[10] == tmp_948 && values[10] < tmp_949);
            if tmp_916 == tmp_950 { keys[10] = tmp_948; values[10] = tmp_949; }
            let tmp_951 = smem_keys[tmp_917 * WPT + 4u];
            let tmp_952 = smem_vals[tmp_917 * WPT + 4u];
            let tmp_953 = keys[11] < tmp_951 || (keys[11] == tmp_951 && values[11] < tmp_952);
            if tmp_916 == tmp_953 { keys[11] = tmp_951; values[11] = tmp_952; }
            let tmp_954 = smem_keys[tmp_917 * WPT + 3u];
            let tmp_955 = smem_vals[tmp_917 * WPT + 3u];
            let tmp_956 = keys[12] < tmp_954 || (keys[12] == tmp_954 && values[12] < tmp_955);
            if tmp_916 == tmp_956 { keys[12] = tmp_954; values[12] = tmp_955; }
            let tmp_957 = smem_keys[tmp_917 * WPT + 2u];
            let tmp_958 = smem_vals[tmp_917 * WPT + 2u];
            let tmp_959 = keys[13] < tmp_957 || (keys[13] == tmp_957 && values[13] < tmp_958);
            if tmp_916 == tmp_959 { keys[13] = tmp_957; values[13] = tmp_958; }
            let tmp_960 = smem_keys[tmp_917 * WPT + 1u];
            let tmp_961 = smem_vals[tmp_917 * WPT + 1u];
            let tmp_962 = keys[14] < tmp_960 || (keys[14] == tmp_960 && values[14] < tmp_961);
            if tmp_916 == tmp_962 { keys[14] = tmp_960; values[14] = tmp_961; }
            let tmp_963 = smem_keys[tmp_917 * WPT + 0u];
            let tmp_964 = smem_vals[tmp_917 * WPT + 0u];
            let tmp_965 = keys[15] < tmp_963 || (keys[15] == tmp_963 && values[15] < tmp_964);
            if tmp_916 == tmp_965 { keys[15] = tmp_963; values[15] = tmp_964; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:8,swbit:3,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],8,3)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_966 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_967 = seg_base + (local_tid ^ 8u);
            let tmp_968 = smem_keys[tmp_967 * WPT + 0u];
            let tmp_969 = smem_vals[tmp_967 * WPT + 0u];
            let tmp_970 = keys[0] < tmp_968 || (keys[0] == tmp_968 && values[0] < tmp_969);
            if tmp_966 == tmp_970 { keys[0] = tmp_968; values[0] = tmp_969; }
            let tmp_971 = smem_keys[tmp_967 * WPT + 1u];
            let tmp_972 = smem_vals[tmp_967 * WPT + 1u];
            let tmp_973 = keys[1] < tmp_971 || (keys[1] == tmp_971 && values[1] < tmp_972);
            if tmp_966 == tmp_973 { keys[1] = tmp_971; values[1] = tmp_972; }
            let tmp_974 = smem_keys[tmp_967 * WPT + 2u];
            let tmp_975 = smem_vals[tmp_967 * WPT + 2u];
            let tmp_976 = keys[2] < tmp_974 || (keys[2] == tmp_974 && values[2] < tmp_975);
            if tmp_966 == tmp_976 { keys[2] = tmp_974; values[2] = tmp_975; }
            let tmp_977 = smem_keys[tmp_967 * WPT + 3u];
            let tmp_978 = smem_vals[tmp_967 * WPT + 3u];
            let tmp_979 = keys[3] < tmp_977 || (keys[3] == tmp_977 && values[3] < tmp_978);
            if tmp_966 == tmp_979 { keys[3] = tmp_977; values[3] = tmp_978; }
            let tmp_980 = smem_keys[tmp_967 * WPT + 4u];
            let tmp_981 = smem_vals[tmp_967 * WPT + 4u];
            let tmp_982 = keys[4] < tmp_980 || (keys[4] == tmp_980 && values[4] < tmp_981);
            if tmp_966 == tmp_982 { keys[4] = tmp_980; values[4] = tmp_981; }
            let tmp_983 = smem_keys[tmp_967 * WPT + 5u];
            let tmp_984 = smem_vals[tmp_967 * WPT + 5u];
            let tmp_985 = keys[5] < tmp_983 || (keys[5] == tmp_983 && values[5] < tmp_984);
            if tmp_966 == tmp_985 { keys[5] = tmp_983; values[5] = tmp_984; }
            let tmp_986 = smem_keys[tmp_967 * WPT + 6u];
            let tmp_987 = smem_vals[tmp_967 * WPT + 6u];
            let tmp_988 = keys[6] < tmp_986 || (keys[6] == tmp_986 && values[6] < tmp_987);
            if tmp_966 == tmp_988 { keys[6] = tmp_986; values[6] = tmp_987; }
            let tmp_989 = smem_keys[tmp_967 * WPT + 7u];
            let tmp_990 = smem_vals[tmp_967 * WPT + 7u];
            let tmp_991 = keys[7] < tmp_989 || (keys[7] == tmp_989 && values[7] < tmp_990);
            if tmp_966 == tmp_991 { keys[7] = tmp_989; values[7] = tmp_990; }
            let tmp_992 = smem_keys[tmp_967 * WPT + 8u];
            let tmp_993 = smem_vals[tmp_967 * WPT + 8u];
            let tmp_994 = keys[8] < tmp_992 || (keys[8] == tmp_992 && values[8] < tmp_993);
            if tmp_966 == tmp_994 { keys[8] = tmp_992; values[8] = tmp_993; }
            let tmp_995 = smem_keys[tmp_967 * WPT + 9u];
            let tmp_996 = smem_vals[tmp_967 * WPT + 9u];
            let tmp_997 = keys[9] < tmp_995 || (keys[9] == tmp_995 && values[9] < tmp_996);
            if tmp_966 == tmp_997 { keys[9] = tmp_995; values[9] = tmp_996; }
            let tmp_998 = smem_keys[tmp_967 * WPT + 10u];
            let tmp_999 = smem_vals[tmp_967 * WPT + 10u];
            let tmp_1000 = keys[10] < tmp_998 || (keys[10] == tmp_998 && values[10] < tmp_999);
            if tmp_966 == tmp_1000 { keys[10] = tmp_998; values[10] = tmp_999; }
            let tmp_1001 = smem_keys[tmp_967 * WPT + 11u];
            let tmp_1002 = smem_vals[tmp_967 * WPT + 11u];
            let tmp_1003 = keys[11] < tmp_1001 || (keys[11] == tmp_1001 && values[11] < tmp_1002);
            if tmp_966 == tmp_1003 { keys[11] = tmp_1001; values[11] = tmp_1002; }
            let tmp_1004 = smem_keys[tmp_967 * WPT + 12u];
            let tmp_1005 = smem_vals[tmp_967 * WPT + 12u];
            let tmp_1006 = keys[12] < tmp_1004 || (keys[12] == tmp_1004 && values[12] < tmp_1005);
            if tmp_966 == tmp_1006 { keys[12] = tmp_1004; values[12] = tmp_1005; }
            let tmp_1007 = smem_keys[tmp_967 * WPT + 13u];
            let tmp_1008 = smem_vals[tmp_967 * WPT + 13u];
            let tmp_1009 = keys[13] < tmp_1007 || (keys[13] == tmp_1007 && values[13] < tmp_1008);
            if tmp_966 == tmp_1009 { keys[13] = tmp_1007; values[13] = tmp_1008; }
            let tmp_1010 = smem_keys[tmp_967 * WPT + 14u];
            let tmp_1011 = smem_vals[tmp_967 * WPT + 14u];
            let tmp_1012 = keys[14] < tmp_1010 || (keys[14] == tmp_1010 && values[14] < tmp_1011);
            if tmp_966 == tmp_1012 { keys[14] = tmp_1010; values[14] = tmp_1011; }
            let tmp_1013 = smem_keys[tmp_967 * WPT + 15u];
            let tmp_1014 = smem_vals[tmp_967 * WPT + 15u];
            let tmp_1015 = keys[15] < tmp_1013 || (keys[15] == tmp_1013 && values[15] < tmp_1014);
            if tmp_966 == tmp_1015 { keys[15] = tmp_1013; values[15] = tmp_1014; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],4,2)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1016 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_1017 = seg_base + (local_tid ^ 4u);
            let tmp_1018 = smem_keys[tmp_1017 * WPT + 0u];
            let tmp_1019 = smem_vals[tmp_1017 * WPT + 0u];
            let tmp_1020 = keys[0] < tmp_1018 || (keys[0] == tmp_1018 && values[0] < tmp_1019);
            if tmp_1016 == tmp_1020 { keys[0] = tmp_1018; values[0] = tmp_1019; }
            let tmp_1021 = smem_keys[tmp_1017 * WPT + 1u];
            let tmp_1022 = smem_vals[tmp_1017 * WPT + 1u];
            let tmp_1023 = keys[1] < tmp_1021 || (keys[1] == tmp_1021 && values[1] < tmp_1022);
            if tmp_1016 == tmp_1023 { keys[1] = tmp_1021; values[1] = tmp_1022; }
            let tmp_1024 = smem_keys[tmp_1017 * WPT + 2u];
            let tmp_1025 = smem_vals[tmp_1017 * WPT + 2u];
            let tmp_1026 = keys[2] < tmp_1024 || (keys[2] == tmp_1024 && values[2] < tmp_1025);
            if tmp_1016 == tmp_1026 { keys[2] = tmp_1024; values[2] = tmp_1025; }
            let tmp_1027 = smem_keys[tmp_1017 * WPT + 3u];
            let tmp_1028 = smem_vals[tmp_1017 * WPT + 3u];
            let tmp_1029 = keys[3] < tmp_1027 || (keys[3] == tmp_1027 && values[3] < tmp_1028);
            if tmp_1016 == tmp_1029 { keys[3] = tmp_1027; values[3] = tmp_1028; }
            let tmp_1030 = smem_keys[tmp_1017 * WPT + 4u];
            let tmp_1031 = smem_vals[tmp_1017 * WPT + 4u];
            let tmp_1032 = keys[4] < tmp_1030 || (keys[4] == tmp_1030 && values[4] < tmp_1031);
            if tmp_1016 == tmp_1032 { keys[4] = tmp_1030; values[4] = tmp_1031; }
            let tmp_1033 = smem_keys[tmp_1017 * WPT + 5u];
            let tmp_1034 = smem_vals[tmp_1017 * WPT + 5u];
            let tmp_1035 = keys[5] < tmp_1033 || (keys[5] == tmp_1033 && values[5] < tmp_1034);
            if tmp_1016 == tmp_1035 { keys[5] = tmp_1033; values[5] = tmp_1034; }
            let tmp_1036 = smem_keys[tmp_1017 * WPT + 6u];
            let tmp_1037 = smem_vals[tmp_1017 * WPT + 6u];
            let tmp_1038 = keys[6] < tmp_1036 || (keys[6] == tmp_1036 && values[6] < tmp_1037);
            if tmp_1016 == tmp_1038 { keys[6] = tmp_1036; values[6] = tmp_1037; }
            let tmp_1039 = smem_keys[tmp_1017 * WPT + 7u];
            let tmp_1040 = smem_vals[tmp_1017 * WPT + 7u];
            let tmp_1041 = keys[7] < tmp_1039 || (keys[7] == tmp_1039 && values[7] < tmp_1040);
            if tmp_1016 == tmp_1041 { keys[7] = tmp_1039; values[7] = tmp_1040; }
            let tmp_1042 = smem_keys[tmp_1017 * WPT + 8u];
            let tmp_1043 = smem_vals[tmp_1017 * WPT + 8u];
            let tmp_1044 = keys[8] < tmp_1042 || (keys[8] == tmp_1042 && values[8] < tmp_1043);
            if tmp_1016 == tmp_1044 { keys[8] = tmp_1042; values[8] = tmp_1043; }
            let tmp_1045 = smem_keys[tmp_1017 * WPT + 9u];
            let tmp_1046 = smem_vals[tmp_1017 * WPT + 9u];
            let tmp_1047 = keys[9] < tmp_1045 || (keys[9] == tmp_1045 && values[9] < tmp_1046);
            if tmp_1016 == tmp_1047 { keys[9] = tmp_1045; values[9] = tmp_1046; }
            let tmp_1048 = smem_keys[tmp_1017 * WPT + 10u];
            let tmp_1049 = smem_vals[tmp_1017 * WPT + 10u];
            let tmp_1050 = keys[10] < tmp_1048 || (keys[10] == tmp_1048 && values[10] < tmp_1049);
            if tmp_1016 == tmp_1050 { keys[10] = tmp_1048; values[10] = tmp_1049; }
            let tmp_1051 = smem_keys[tmp_1017 * WPT + 11u];
            let tmp_1052 = smem_vals[tmp_1017 * WPT + 11u];
            let tmp_1053 = keys[11] < tmp_1051 || (keys[11] == tmp_1051 && values[11] < tmp_1052);
            if tmp_1016 == tmp_1053 { keys[11] = tmp_1051; values[11] = tmp_1052; }
            let tmp_1054 = smem_keys[tmp_1017 * WPT + 12u];
            let tmp_1055 = smem_vals[tmp_1017 * WPT + 12u];
            let tmp_1056 = keys[12] < tmp_1054 || (keys[12] == tmp_1054 && values[12] < tmp_1055);
            if tmp_1016 == tmp_1056 { keys[12] = tmp_1054; values[12] = tmp_1055; }
            let tmp_1057 = smem_keys[tmp_1017 * WPT + 13u];
            let tmp_1058 = smem_vals[tmp_1017 * WPT + 13u];
            let tmp_1059 = keys[13] < tmp_1057 || (keys[13] == tmp_1057 && values[13] < tmp_1058);
            if tmp_1016 == tmp_1059 { keys[13] = tmp_1057; values[13] = tmp_1058; }
            let tmp_1060 = smem_keys[tmp_1017 * WPT + 14u];
            let tmp_1061 = smem_vals[tmp_1017 * WPT + 14u];
            let tmp_1062 = keys[14] < tmp_1060 || (keys[14] == tmp_1060 && values[14] < tmp_1061);
            if tmp_1016 == tmp_1062 { keys[14] = tmp_1060; values[14] = tmp_1061; }
            let tmp_1063 = smem_keys[tmp_1017 * WPT + 15u];
            let tmp_1064 = smem_vals[tmp_1017 * WPT + 15u];
            let tmp_1065 = keys[15] < tmp_1063 || (keys[15] == tmp_1063 && values[15] < tmp_1064);
            if tmp_1016 == tmp_1065 { keys[15] = tmp_1063; values[15] = tmp_1064; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],2,1)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1066 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_1067 = seg_base + (local_tid ^ 2u);
            let tmp_1068 = smem_keys[tmp_1067 * WPT + 0u];
            let tmp_1069 = smem_vals[tmp_1067 * WPT + 0u];
            let tmp_1070 = keys[0] < tmp_1068 || (keys[0] == tmp_1068 && values[0] < tmp_1069);
            if tmp_1066 == tmp_1070 { keys[0] = tmp_1068; values[0] = tmp_1069; }
            let tmp_1071 = smem_keys[tmp_1067 * WPT + 1u];
            let tmp_1072 = smem_vals[tmp_1067 * WPT + 1u];
            let tmp_1073 = keys[1] < tmp_1071 || (keys[1] == tmp_1071 && values[1] < tmp_1072);
            if tmp_1066 == tmp_1073 { keys[1] = tmp_1071; values[1] = tmp_1072; }
            let tmp_1074 = smem_keys[tmp_1067 * WPT + 2u];
            let tmp_1075 = smem_vals[tmp_1067 * WPT + 2u];
            let tmp_1076 = keys[2] < tmp_1074 || (keys[2] == tmp_1074 && values[2] < tmp_1075);
            if tmp_1066 == tmp_1076 { keys[2] = tmp_1074; values[2] = tmp_1075; }
            let tmp_1077 = smem_keys[tmp_1067 * WPT + 3u];
            let tmp_1078 = smem_vals[tmp_1067 * WPT + 3u];
            let tmp_1079 = keys[3] < tmp_1077 || (keys[3] == tmp_1077 && values[3] < tmp_1078);
            if tmp_1066 == tmp_1079 { keys[3] = tmp_1077; values[3] = tmp_1078; }
            let tmp_1080 = smem_keys[tmp_1067 * WPT + 4u];
            let tmp_1081 = smem_vals[tmp_1067 * WPT + 4u];
            let tmp_1082 = keys[4] < tmp_1080 || (keys[4] == tmp_1080 && values[4] < tmp_1081);
            if tmp_1066 == tmp_1082 { keys[4] = tmp_1080; values[4] = tmp_1081; }
            let tmp_1083 = smem_keys[tmp_1067 * WPT + 5u];
            let tmp_1084 = smem_vals[tmp_1067 * WPT + 5u];
            let tmp_1085 = keys[5] < tmp_1083 || (keys[5] == tmp_1083 && values[5] < tmp_1084);
            if tmp_1066 == tmp_1085 { keys[5] = tmp_1083; values[5] = tmp_1084; }
            let tmp_1086 = smem_keys[tmp_1067 * WPT + 6u];
            let tmp_1087 = smem_vals[tmp_1067 * WPT + 6u];
            let tmp_1088 = keys[6] < tmp_1086 || (keys[6] == tmp_1086 && values[6] < tmp_1087);
            if tmp_1066 == tmp_1088 { keys[6] = tmp_1086; values[6] = tmp_1087; }
            let tmp_1089 = smem_keys[tmp_1067 * WPT + 7u];
            let tmp_1090 = smem_vals[tmp_1067 * WPT + 7u];
            let tmp_1091 = keys[7] < tmp_1089 || (keys[7] == tmp_1089 && values[7] < tmp_1090);
            if tmp_1066 == tmp_1091 { keys[7] = tmp_1089; values[7] = tmp_1090; }
            let tmp_1092 = smem_keys[tmp_1067 * WPT + 8u];
            let tmp_1093 = smem_vals[tmp_1067 * WPT + 8u];
            let tmp_1094 = keys[8] < tmp_1092 || (keys[8] == tmp_1092 && values[8] < tmp_1093);
            if tmp_1066 == tmp_1094 { keys[8] = tmp_1092; values[8] = tmp_1093; }
            let tmp_1095 = smem_keys[tmp_1067 * WPT + 9u];
            let tmp_1096 = smem_vals[tmp_1067 * WPT + 9u];
            let tmp_1097 = keys[9] < tmp_1095 || (keys[9] == tmp_1095 && values[9] < tmp_1096);
            if tmp_1066 == tmp_1097 { keys[9] = tmp_1095; values[9] = tmp_1096; }
            let tmp_1098 = smem_keys[tmp_1067 * WPT + 10u];
            let tmp_1099 = smem_vals[tmp_1067 * WPT + 10u];
            let tmp_1100 = keys[10] < tmp_1098 || (keys[10] == tmp_1098 && values[10] < tmp_1099);
            if tmp_1066 == tmp_1100 { keys[10] = tmp_1098; values[10] = tmp_1099; }
            let tmp_1101 = smem_keys[tmp_1067 * WPT + 11u];
            let tmp_1102 = smem_vals[tmp_1067 * WPT + 11u];
            let tmp_1103 = keys[11] < tmp_1101 || (keys[11] == tmp_1101 && values[11] < tmp_1102);
            if tmp_1066 == tmp_1103 { keys[11] = tmp_1101; values[11] = tmp_1102; }
            let tmp_1104 = smem_keys[tmp_1067 * WPT + 12u];
            let tmp_1105 = smem_vals[tmp_1067 * WPT + 12u];
            let tmp_1106 = keys[12] < tmp_1104 || (keys[12] == tmp_1104 && values[12] < tmp_1105);
            if tmp_1066 == tmp_1106 { keys[12] = tmp_1104; values[12] = tmp_1105; }
            let tmp_1107 = smem_keys[tmp_1067 * WPT + 13u];
            let tmp_1108 = smem_vals[tmp_1067 * WPT + 13u];
            let tmp_1109 = keys[13] < tmp_1107 || (keys[13] == tmp_1107 && values[13] < tmp_1108);
            if tmp_1066 == tmp_1109 { keys[13] = tmp_1107; values[13] = tmp_1108; }
            let tmp_1110 = smem_keys[tmp_1067 * WPT + 14u];
            let tmp_1111 = smem_vals[tmp_1067 * WPT + 14u];
            let tmp_1112 = keys[14] < tmp_1110 || (keys[14] == tmp_1110 && values[14] < tmp_1111);
            if tmp_1066 == tmp_1112 { keys[14] = tmp_1110; values[14] = tmp_1111; }
            let tmp_1113 = smem_keys[tmp_1067 * WPT + 15u];
            let tmp_1114 = smem_vals[tmp_1067 * WPT + 15u];
            let tmp_1115 = keys[15] < tmp_1113 || (keys[15] == tmp_1113 && values[15] < tmp_1114);
            if tmp_1066 == tmp_1115 { keys[15] = tmp_1113; values[15] = tmp_1114; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],1,0)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1116 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_1117 = seg_base + (local_tid ^ 1u);
            let tmp_1118 = smem_keys[tmp_1117 * WPT + 0u];
            let tmp_1119 = smem_vals[tmp_1117 * WPT + 0u];
            let tmp_1120 = keys[0] < tmp_1118 || (keys[0] == tmp_1118 && values[0] < tmp_1119);
            if tmp_1116 == tmp_1120 { keys[0] = tmp_1118; values[0] = tmp_1119; }
            let tmp_1121 = smem_keys[tmp_1117 * WPT + 1u];
            let tmp_1122 = smem_vals[tmp_1117 * WPT + 1u];
            let tmp_1123 = keys[1] < tmp_1121 || (keys[1] == tmp_1121 && values[1] < tmp_1122);
            if tmp_1116 == tmp_1123 { keys[1] = tmp_1121; values[1] = tmp_1122; }
            let tmp_1124 = smem_keys[tmp_1117 * WPT + 2u];
            let tmp_1125 = smem_vals[tmp_1117 * WPT + 2u];
            let tmp_1126 = keys[2] < tmp_1124 || (keys[2] == tmp_1124 && values[2] < tmp_1125);
            if tmp_1116 == tmp_1126 { keys[2] = tmp_1124; values[2] = tmp_1125; }
            let tmp_1127 = smem_keys[tmp_1117 * WPT + 3u];
            let tmp_1128 = smem_vals[tmp_1117 * WPT + 3u];
            let tmp_1129 = keys[3] < tmp_1127 || (keys[3] == tmp_1127 && values[3] < tmp_1128);
            if tmp_1116 == tmp_1129 { keys[3] = tmp_1127; values[3] = tmp_1128; }
            let tmp_1130 = smem_keys[tmp_1117 * WPT + 4u];
            let tmp_1131 = smem_vals[tmp_1117 * WPT + 4u];
            let tmp_1132 = keys[4] < tmp_1130 || (keys[4] == tmp_1130 && values[4] < tmp_1131);
            if tmp_1116 == tmp_1132 { keys[4] = tmp_1130; values[4] = tmp_1131; }
            let tmp_1133 = smem_keys[tmp_1117 * WPT + 5u];
            let tmp_1134 = smem_vals[tmp_1117 * WPT + 5u];
            let tmp_1135 = keys[5] < tmp_1133 || (keys[5] == tmp_1133 && values[5] < tmp_1134);
            if tmp_1116 == tmp_1135 { keys[5] = tmp_1133; values[5] = tmp_1134; }
            let tmp_1136 = smem_keys[tmp_1117 * WPT + 6u];
            let tmp_1137 = smem_vals[tmp_1117 * WPT + 6u];
            let tmp_1138 = keys[6] < tmp_1136 || (keys[6] == tmp_1136 && values[6] < tmp_1137);
            if tmp_1116 == tmp_1138 { keys[6] = tmp_1136; values[6] = tmp_1137; }
            let tmp_1139 = smem_keys[tmp_1117 * WPT + 7u];
            let tmp_1140 = smem_vals[tmp_1117 * WPT + 7u];
            let tmp_1141 = keys[7] < tmp_1139 || (keys[7] == tmp_1139 && values[7] < tmp_1140);
            if tmp_1116 == tmp_1141 { keys[7] = tmp_1139; values[7] = tmp_1140; }
            let tmp_1142 = smem_keys[tmp_1117 * WPT + 8u];
            let tmp_1143 = smem_vals[tmp_1117 * WPT + 8u];
            let tmp_1144 = keys[8] < tmp_1142 || (keys[8] == tmp_1142 && values[8] < tmp_1143);
            if tmp_1116 == tmp_1144 { keys[8] = tmp_1142; values[8] = tmp_1143; }
            let tmp_1145 = smem_keys[tmp_1117 * WPT + 9u];
            let tmp_1146 = smem_vals[tmp_1117 * WPT + 9u];
            let tmp_1147 = keys[9] < tmp_1145 || (keys[9] == tmp_1145 && values[9] < tmp_1146);
            if tmp_1116 == tmp_1147 { keys[9] = tmp_1145; values[9] = tmp_1146; }
            let tmp_1148 = smem_keys[tmp_1117 * WPT + 10u];
            let tmp_1149 = smem_vals[tmp_1117 * WPT + 10u];
            let tmp_1150 = keys[10] < tmp_1148 || (keys[10] == tmp_1148 && values[10] < tmp_1149);
            if tmp_1116 == tmp_1150 { keys[10] = tmp_1148; values[10] = tmp_1149; }
            let tmp_1151 = smem_keys[tmp_1117 * WPT + 11u];
            let tmp_1152 = smem_vals[tmp_1117 * WPT + 11u];
            let tmp_1153 = keys[11] < tmp_1151 || (keys[11] == tmp_1151 && values[11] < tmp_1152);
            if tmp_1116 == tmp_1153 { keys[11] = tmp_1151; values[11] = tmp_1152; }
            let tmp_1154 = smem_keys[tmp_1117 * WPT + 12u];
            let tmp_1155 = smem_vals[tmp_1117 * WPT + 12u];
            let tmp_1156 = keys[12] < tmp_1154 || (keys[12] == tmp_1154 && values[12] < tmp_1155);
            if tmp_1116 == tmp_1156 { keys[12] = tmp_1154; values[12] = tmp_1155; }
            let tmp_1157 = smem_keys[tmp_1117 * WPT + 13u];
            let tmp_1158 = smem_vals[tmp_1117 * WPT + 13u];
            let tmp_1159 = keys[13] < tmp_1157 || (keys[13] == tmp_1157 && values[13] < tmp_1158);
            if tmp_1116 == tmp_1159 { keys[13] = tmp_1157; values[13] = tmp_1158; }
            let tmp_1160 = smem_keys[tmp_1117 * WPT + 14u];
            let tmp_1161 = smem_vals[tmp_1117 * WPT + 14u];
            let tmp_1162 = keys[14] < tmp_1160 || (keys[14] == tmp_1160 && values[14] < tmp_1161);
            if tmp_1116 == tmp_1162 { keys[14] = tmp_1160; values[14] = tmp_1161; }
            let tmp_1163 = smem_keys[tmp_1117 * WPT + 15u];
            let tmp_1164 = smem_vals[tmp_1117 * WPT + 15u];
            let tmp_1165 = keys[15] < tmp_1163 || (keys[15] == tmp_1163 && values[15] < tmp_1164);
            if tmp_1116 == tmp_1165 { keys[15] = tmp_1163; values[15] = tmp_1164; }
            workgroupBarrier();
        }
    }

    // exch_local(8,16)
    {
        // cmp_swap(0,8)
        if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
            // swap(0,8)
            {
                let tmp_1166 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_1166;
                let tmp_1167 = values[0]; values[0] = values[8]; values[8] = tmp_1167;
            }
        }
        // cmp_swap(1,9)
        if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
            // swap(1,9)
            {
                let tmp_1168 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_1168;
                let tmp_1169 = values[1]; values[1] = values[9]; values[9] = tmp_1169;
            }
        }
        // cmp_swap(2,10)
        if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
            // swap(2,10)
            {
                let tmp_1170 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_1170;
                let tmp_1171 = values[2]; values[2] = values[10]; values[10] = tmp_1171;
            }
        }
        // cmp_swap(3,11)
        if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
            // swap(3,11)
            {
                let tmp_1172 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_1172;
                let tmp_1173 = values[3]; values[3] = values[11]; values[11] = tmp_1173;
            }
        }
        // cmp_swap(4,12)
        if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
            // swap(4,12)
            {
                let tmp_1174 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_1174;
                let tmp_1175 = values[4]; values[4] = values[12]; values[12] = tmp_1175;
            }
        }
        // cmp_swap(5,13)
        if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
            // swap(5,13)
            {
                let tmp_1176 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_1176;
                let tmp_1177 = values[5]; values[5] = values[13]; values[13] = tmp_1177;
            }
        }
        // cmp_swap(6,14)
        if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
            // swap(6,14)
            {
                let tmp_1178 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_1178;
                let tmp_1179 = values[6]; values[6] = values[14]; values[14] = tmp_1179;
            }
        }
        // cmp_swap(7,15)
        if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
            // swap(7,15)
            {
                let tmp_1180 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_1180;
                let tmp_1181 = values[7]; values[7] = values[15]; values[15] = tmp_1181;
            }
        }
    }

    // exch_local(4,16)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_1182 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_1182;
                let tmp_1183 = values[0]; values[0] = values[4]; values[4] = tmp_1183;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_1184 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_1184;
                let tmp_1185 = values[1]; values[1] = values[5]; values[5] = tmp_1185;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_1186 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_1186;
                let tmp_1187 = values[2]; values[2] = values[6]; values[6] = tmp_1187;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_1188 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_1188;
                let tmp_1189 = values[3]; values[3] = values[7]; values[7] = tmp_1189;
            }
        }
        // cmp_swap(8,12)
        if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
            // swap(8,12)
            {
                let tmp_1190 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_1190;
                let tmp_1191 = values[8]; values[8] = values[12]; values[12] = tmp_1191;
            }
        }
        // cmp_swap(9,13)
        if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
            // swap(9,13)
            {
                let tmp_1192 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_1192;
                let tmp_1193 = values[9]; values[9] = values[13]; values[13] = tmp_1193;
            }
        }
        // cmp_swap(10,14)
        if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
            // swap(10,14)
            {
                let tmp_1194 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_1194;
                let tmp_1195 = values[10]; values[10] = values[14]; values[14] = tmp_1195;
            }
        }
        // cmp_swap(11,15)
        if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
            // swap(11,15)
            {
                let tmp_1196 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_1196;
                let tmp_1197 = values[11]; values[11] = values[15]; values[15] = tmp_1197;
            }
        }
    }

    // exch_local(2,16)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_1198 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_1198;
                let tmp_1199 = values[0]; values[0] = values[2]; values[2] = tmp_1199;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_1200 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_1200;
                let tmp_1201 = values[1]; values[1] = values[3]; values[3] = tmp_1201;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_1202 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_1202;
                let tmp_1203 = values[4]; values[4] = values[6]; values[6] = tmp_1203;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_1204 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_1204;
                let tmp_1205 = values[5]; values[5] = values[7]; values[7] = tmp_1205;
            }
        }
        // cmp_swap(8,10)
        if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
            // swap(8,10)
            {
                let tmp_1206 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_1206;
                let tmp_1207 = values[8]; values[8] = values[10]; values[10] = tmp_1207;
            }
        }
        // cmp_swap(9,11)
        if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
            // swap(9,11)
            {
                let tmp_1208 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_1208;
                let tmp_1209 = values[9]; values[9] = values[11]; values[11] = tmp_1209;
            }
        }
        // cmp_swap(12,14)
        if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
            // swap(12,14)
            {
                let tmp_1210 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_1210;
                let tmp_1211 = values[12]; values[12] = values[14]; values[14] = tmp_1211;
            }
        }
        // cmp_swap(13,15)
        if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
            // swap(13,15)
            {
                let tmp_1212 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_1212;
                let tmp_1213 = values[13]; values[13] = values[15]; values[15] = tmp_1213;
            }
        }
    }

    // exch_local(1,16)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_1214 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_1214;
                let tmp_1215 = values[0]; values[0] = values[1]; values[1] = tmp_1215;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_1216 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_1216;
                let tmp_1217 = values[2]; values[2] = values[3]; values[3] = tmp_1217;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_1218 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_1218;
                let tmp_1219 = values[4]; values[4] = values[5]; values[5] = tmp_1219;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_1220 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_1220;
                let tmp_1221 = values[6]; values[6] = values[7]; values[7] = tmp_1221;
            }
        }
        // cmp_swap(8,9)
        if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
            // swap(8,9)
            {
                let tmp_1222 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_1222;
                let tmp_1223 = values[8]; values[8] = values[9]; values[9] = tmp_1223;
            }
        }
        // cmp_swap(10,11)
        if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
            // swap(10,11)
            {
                let tmp_1224 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_1224;
                let tmp_1225 = values[10]; values[10] = values[11]; values[11] = tmp_1225;
            }
        }
        // cmp_swap(12,13)
        if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
            // swap(12,13)
            {
                let tmp_1226 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_1226;
                let tmp_1227 = values[12]; values[12] = values[13]; values[13] = tmp_1227;
            }
        }
        // cmp_swap(14,15)
        if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
            // swap(14,15)
            {
                let tmp_1228 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_1228;
                let tmp_1229 = values[14]; values[14] = values[15]; values[15] = tmp_1229;
            }
        }
    }

    // exch_intxn(tmask:63,swbit:5,wpt:16)
    {
        // _exch_workgroup([(0, 15), (1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8), (8, 7), (9, 6), (10, 5), (11, 4), (12, 3), (13, 2), (14, 1), (15, 0)],63,5)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1230 = extractBits(local_tid, 5u, 1u) != 0u;
            let tmp_1231 = seg_base + (local_tid ^ 63u);
            let tmp_1232 = smem_keys[tmp_1231 * WPT + 15u];
            let tmp_1233 = smem_vals[tmp_1231 * WPT + 15u];
            let tmp_1234 = keys[0] < tmp_1232 || (keys[0] == tmp_1232 && values[0] < tmp_1233);
            if tmp_1230 == tmp_1234 { keys[0] = tmp_1232; values[0] = tmp_1233; }
            let tmp_1235 = smem_keys[tmp_1231 * WPT + 14u];
            let tmp_1236 = smem_vals[tmp_1231 * WPT + 14u];
            let tmp_1237 = keys[1] < tmp_1235 || (keys[1] == tmp_1235 && values[1] < tmp_1236);
            if tmp_1230 == tmp_1237 { keys[1] = tmp_1235; values[1] = tmp_1236; }
            let tmp_1238 = smem_keys[tmp_1231 * WPT + 13u];
            let tmp_1239 = smem_vals[tmp_1231 * WPT + 13u];
            let tmp_1240 = keys[2] < tmp_1238 || (keys[2] == tmp_1238 && values[2] < tmp_1239);
            if tmp_1230 == tmp_1240 { keys[2] = tmp_1238; values[2] = tmp_1239; }
            let tmp_1241 = smem_keys[tmp_1231 * WPT + 12u];
            let tmp_1242 = smem_vals[tmp_1231 * WPT + 12u];
            let tmp_1243 = keys[3] < tmp_1241 || (keys[3] == tmp_1241 && values[3] < tmp_1242);
            if tmp_1230 == tmp_1243 { keys[3] = tmp_1241; values[3] = tmp_1242; }
            let tmp_1244 = smem_keys[tmp_1231 * WPT + 11u];
            let tmp_1245 = smem_vals[tmp_1231 * WPT + 11u];
            let tmp_1246 = keys[4] < tmp_1244 || (keys[4] == tmp_1244 && values[4] < tmp_1245);
            if tmp_1230 == tmp_1246 { keys[4] = tmp_1244; values[4] = tmp_1245; }
            let tmp_1247 = smem_keys[tmp_1231 * WPT + 10u];
            let tmp_1248 = smem_vals[tmp_1231 * WPT + 10u];
            let tmp_1249 = keys[5] < tmp_1247 || (keys[5] == tmp_1247 && values[5] < tmp_1248);
            if tmp_1230 == tmp_1249 { keys[5] = tmp_1247; values[5] = tmp_1248; }
            let tmp_1250 = smem_keys[tmp_1231 * WPT + 9u];
            let tmp_1251 = smem_vals[tmp_1231 * WPT + 9u];
            let tmp_1252 = keys[6] < tmp_1250 || (keys[6] == tmp_1250 && values[6] < tmp_1251);
            if tmp_1230 == tmp_1252 { keys[6] = tmp_1250; values[6] = tmp_1251; }
            let tmp_1253 = smem_keys[tmp_1231 * WPT + 8u];
            let tmp_1254 = smem_vals[tmp_1231 * WPT + 8u];
            let tmp_1255 = keys[7] < tmp_1253 || (keys[7] == tmp_1253 && values[7] < tmp_1254);
            if tmp_1230 == tmp_1255 { keys[7] = tmp_1253; values[7] = tmp_1254; }
            let tmp_1256 = smem_keys[tmp_1231 * WPT + 7u];
            let tmp_1257 = smem_vals[tmp_1231 * WPT + 7u];
            let tmp_1258 = keys[8] < tmp_1256 || (keys[8] == tmp_1256 && values[8] < tmp_1257);
            if tmp_1230 == tmp_1258 { keys[8] = tmp_1256; values[8] = tmp_1257; }
            let tmp_1259 = smem_keys[tmp_1231 * WPT + 6u];
            let tmp_1260 = smem_vals[tmp_1231 * WPT + 6u];
            let tmp_1261 = keys[9] < tmp_1259 || (keys[9] == tmp_1259 && values[9] < tmp_1260);
            if tmp_1230 == tmp_1261 { keys[9] = tmp_1259; values[9] = tmp_1260; }
            let tmp_1262 = smem_keys[tmp_1231 * WPT + 5u];
            let tmp_1263 = smem_vals[tmp_1231 * WPT + 5u];
            let tmp_1264 = keys[10] < tmp_1262 || (keys[10] == tmp_1262 && values[10] < tmp_1263);
            if tmp_1230 == tmp_1264 { keys[10] = tmp_1262; values[10] = tmp_1263; }
            let tmp_1265 = smem_keys[tmp_1231 * WPT + 4u];
            let tmp_1266 = smem_vals[tmp_1231 * WPT + 4u];
            let tmp_1267 = keys[11] < tmp_1265 || (keys[11] == tmp_1265 && values[11] < tmp_1266);
            if tmp_1230 == tmp_1267 { keys[11] = tmp_1265; values[11] = tmp_1266; }
            let tmp_1268 = smem_keys[tmp_1231 * WPT + 3u];
            let tmp_1269 = smem_vals[tmp_1231 * WPT + 3u];
            let tmp_1270 = keys[12] < tmp_1268 || (keys[12] == tmp_1268 && values[12] < tmp_1269);
            if tmp_1230 == tmp_1270 { keys[12] = tmp_1268; values[12] = tmp_1269; }
            let tmp_1271 = smem_keys[tmp_1231 * WPT + 2u];
            let tmp_1272 = smem_vals[tmp_1231 * WPT + 2u];
            let tmp_1273 = keys[13] < tmp_1271 || (keys[13] == tmp_1271 && values[13] < tmp_1272);
            if tmp_1230 == tmp_1273 { keys[13] = tmp_1271; values[13] = tmp_1272; }
            let tmp_1274 = smem_keys[tmp_1231 * WPT + 1u];
            let tmp_1275 = smem_vals[tmp_1231 * WPT + 1u];
            let tmp_1276 = keys[14] < tmp_1274 || (keys[14] == tmp_1274 && values[14] < tmp_1275);
            if tmp_1230 == tmp_1276 { keys[14] = tmp_1274; values[14] = tmp_1275; }
            let tmp_1277 = smem_keys[tmp_1231 * WPT + 0u];
            let tmp_1278 = smem_vals[tmp_1231 * WPT + 0u];
            let tmp_1279 = keys[15] < tmp_1277 || (keys[15] == tmp_1277 && values[15] < tmp_1278);
            if tmp_1230 == tmp_1279 { keys[15] = tmp_1277; values[15] = tmp_1278; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:16,swbit:4,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],16,4)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1280 = extractBits(local_tid, 4u, 1u) != 0u;
            let tmp_1281 = seg_base + (local_tid ^ 16u);
            let tmp_1282 = smem_keys[tmp_1281 * WPT + 0u];
            let tmp_1283 = smem_vals[tmp_1281 * WPT + 0u];
            let tmp_1284 = keys[0] < tmp_1282 || (keys[0] == tmp_1282 && values[0] < tmp_1283);
            if tmp_1280 == tmp_1284 { keys[0] = tmp_1282; values[0] = tmp_1283; }
            let tmp_1285 = smem_keys[tmp_1281 * WPT + 1u];
            let tmp_1286 = smem_vals[tmp_1281 * WPT + 1u];
            let tmp_1287 = keys[1] < tmp_1285 || (keys[1] == tmp_1285 && values[1] < tmp_1286);
            if tmp_1280 == tmp_1287 { keys[1] = tmp_1285; values[1] = tmp_1286; }
            let tmp_1288 = smem_keys[tmp_1281 * WPT + 2u];
            let tmp_1289 = smem_vals[tmp_1281 * WPT + 2u];
            let tmp_1290 = keys[2] < tmp_1288 || (keys[2] == tmp_1288 && values[2] < tmp_1289);
            if tmp_1280 == tmp_1290 { keys[2] = tmp_1288; values[2] = tmp_1289; }
            let tmp_1291 = smem_keys[tmp_1281 * WPT + 3u];
            let tmp_1292 = smem_vals[tmp_1281 * WPT + 3u];
            let tmp_1293 = keys[3] < tmp_1291 || (keys[3] == tmp_1291 && values[3] < tmp_1292);
            if tmp_1280 == tmp_1293 { keys[3] = tmp_1291; values[3] = tmp_1292; }
            let tmp_1294 = smem_keys[tmp_1281 * WPT + 4u];
            let tmp_1295 = smem_vals[tmp_1281 * WPT + 4u];
            let tmp_1296 = keys[4] < tmp_1294 || (keys[4] == tmp_1294 && values[4] < tmp_1295);
            if tmp_1280 == tmp_1296 { keys[4] = tmp_1294; values[4] = tmp_1295; }
            let tmp_1297 = smem_keys[tmp_1281 * WPT + 5u];
            let tmp_1298 = smem_vals[tmp_1281 * WPT + 5u];
            let tmp_1299 = keys[5] < tmp_1297 || (keys[5] == tmp_1297 && values[5] < tmp_1298);
            if tmp_1280 == tmp_1299 { keys[5] = tmp_1297; values[5] = tmp_1298; }
            let tmp_1300 = smem_keys[tmp_1281 * WPT + 6u];
            let tmp_1301 = smem_vals[tmp_1281 * WPT + 6u];
            let tmp_1302 = keys[6] < tmp_1300 || (keys[6] == tmp_1300 && values[6] < tmp_1301);
            if tmp_1280 == tmp_1302 { keys[6] = tmp_1300; values[6] = tmp_1301; }
            let tmp_1303 = smem_keys[tmp_1281 * WPT + 7u];
            let tmp_1304 = smem_vals[tmp_1281 * WPT + 7u];
            let tmp_1305 = keys[7] < tmp_1303 || (keys[7] == tmp_1303 && values[7] < tmp_1304);
            if tmp_1280 == tmp_1305 { keys[7] = tmp_1303; values[7] = tmp_1304; }
            let tmp_1306 = smem_keys[tmp_1281 * WPT + 8u];
            let tmp_1307 = smem_vals[tmp_1281 * WPT + 8u];
            let tmp_1308 = keys[8] < tmp_1306 || (keys[8] == tmp_1306 && values[8] < tmp_1307);
            if tmp_1280 == tmp_1308 { keys[8] = tmp_1306; values[8] = tmp_1307; }
            let tmp_1309 = smem_keys[tmp_1281 * WPT + 9u];
            let tmp_1310 = smem_vals[tmp_1281 * WPT + 9u];
            let tmp_1311 = keys[9] < tmp_1309 || (keys[9] == tmp_1309 && values[9] < tmp_1310);
            if tmp_1280 == tmp_1311 { keys[9] = tmp_1309; values[9] = tmp_1310; }
            let tmp_1312 = smem_keys[tmp_1281 * WPT + 10u];
            let tmp_1313 = smem_vals[tmp_1281 * WPT + 10u];
            let tmp_1314 = keys[10] < tmp_1312 || (keys[10] == tmp_1312 && values[10] < tmp_1313);
            if tmp_1280 == tmp_1314 { keys[10] = tmp_1312; values[10] = tmp_1313; }
            let tmp_1315 = smem_keys[tmp_1281 * WPT + 11u];
            let tmp_1316 = smem_vals[tmp_1281 * WPT + 11u];
            let tmp_1317 = keys[11] < tmp_1315 || (keys[11] == tmp_1315 && values[11] < tmp_1316);
            if tmp_1280 == tmp_1317 { keys[11] = tmp_1315; values[11] = tmp_1316; }
            let tmp_1318 = smem_keys[tmp_1281 * WPT + 12u];
            let tmp_1319 = smem_vals[tmp_1281 * WPT + 12u];
            let tmp_1320 = keys[12] < tmp_1318 || (keys[12] == tmp_1318 && values[12] < tmp_1319);
            if tmp_1280 == tmp_1320 { keys[12] = tmp_1318; values[12] = tmp_1319; }
            let tmp_1321 = smem_keys[tmp_1281 * WPT + 13u];
            let tmp_1322 = smem_vals[tmp_1281 * WPT + 13u];
            let tmp_1323 = keys[13] < tmp_1321 || (keys[13] == tmp_1321 && values[13] < tmp_1322);
            if tmp_1280 == tmp_1323 { keys[13] = tmp_1321; values[13] = tmp_1322; }
            let tmp_1324 = smem_keys[tmp_1281 * WPT + 14u];
            let tmp_1325 = smem_vals[tmp_1281 * WPT + 14u];
            let tmp_1326 = keys[14] < tmp_1324 || (keys[14] == tmp_1324 && values[14] < tmp_1325);
            if tmp_1280 == tmp_1326 { keys[14] = tmp_1324; values[14] = tmp_1325; }
            let tmp_1327 = smem_keys[tmp_1281 * WPT + 15u];
            let tmp_1328 = smem_vals[tmp_1281 * WPT + 15u];
            let tmp_1329 = keys[15] < tmp_1327 || (keys[15] == tmp_1327 && values[15] < tmp_1328);
            if tmp_1280 == tmp_1329 { keys[15] = tmp_1327; values[15] = tmp_1328; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:8,swbit:3,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],8,3)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1330 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_1331 = seg_base + (local_tid ^ 8u);
            let tmp_1332 = smem_keys[tmp_1331 * WPT + 0u];
            let tmp_1333 = smem_vals[tmp_1331 * WPT + 0u];
            let tmp_1334 = keys[0] < tmp_1332 || (keys[0] == tmp_1332 && values[0] < tmp_1333);
            if tmp_1330 == tmp_1334 { keys[0] = tmp_1332; values[0] = tmp_1333; }
            let tmp_1335 = smem_keys[tmp_1331 * WPT + 1u];
            let tmp_1336 = smem_vals[tmp_1331 * WPT + 1u];
            let tmp_1337 = keys[1] < tmp_1335 || (keys[1] == tmp_1335 && values[1] < tmp_1336);
            if tmp_1330 == tmp_1337 { keys[1] = tmp_1335; values[1] = tmp_1336; }
            let tmp_1338 = smem_keys[tmp_1331 * WPT + 2u];
            let tmp_1339 = smem_vals[tmp_1331 * WPT + 2u];
            let tmp_1340 = keys[2] < tmp_1338 || (keys[2] == tmp_1338 && values[2] < tmp_1339);
            if tmp_1330 == tmp_1340 { keys[2] = tmp_1338; values[2] = tmp_1339; }
            let tmp_1341 = smem_keys[tmp_1331 * WPT + 3u];
            let tmp_1342 = smem_vals[tmp_1331 * WPT + 3u];
            let tmp_1343 = keys[3] < tmp_1341 || (keys[3] == tmp_1341 && values[3] < tmp_1342);
            if tmp_1330 == tmp_1343 { keys[3] = tmp_1341; values[3] = tmp_1342; }
            let tmp_1344 = smem_keys[tmp_1331 * WPT + 4u];
            let tmp_1345 = smem_vals[tmp_1331 * WPT + 4u];
            let tmp_1346 = keys[4] < tmp_1344 || (keys[4] == tmp_1344 && values[4] < tmp_1345);
            if tmp_1330 == tmp_1346 { keys[4] = tmp_1344; values[4] = tmp_1345; }
            let tmp_1347 = smem_keys[tmp_1331 * WPT + 5u];
            let tmp_1348 = smem_vals[tmp_1331 * WPT + 5u];
            let tmp_1349 = keys[5] < tmp_1347 || (keys[5] == tmp_1347 && values[5] < tmp_1348);
            if tmp_1330 == tmp_1349 { keys[5] = tmp_1347; values[5] = tmp_1348; }
            let tmp_1350 = smem_keys[tmp_1331 * WPT + 6u];
            let tmp_1351 = smem_vals[tmp_1331 * WPT + 6u];
            let tmp_1352 = keys[6] < tmp_1350 || (keys[6] == tmp_1350 && values[6] < tmp_1351);
            if tmp_1330 == tmp_1352 { keys[6] = tmp_1350; values[6] = tmp_1351; }
            let tmp_1353 = smem_keys[tmp_1331 * WPT + 7u];
            let tmp_1354 = smem_vals[tmp_1331 * WPT + 7u];
            let tmp_1355 = keys[7] < tmp_1353 || (keys[7] == tmp_1353 && values[7] < tmp_1354);
            if tmp_1330 == tmp_1355 { keys[7] = tmp_1353; values[7] = tmp_1354; }
            let tmp_1356 = smem_keys[tmp_1331 * WPT + 8u];
            let tmp_1357 = smem_vals[tmp_1331 * WPT + 8u];
            let tmp_1358 = keys[8] < tmp_1356 || (keys[8] == tmp_1356 && values[8] < tmp_1357);
            if tmp_1330 == tmp_1358 { keys[8] = tmp_1356; values[8] = tmp_1357; }
            let tmp_1359 = smem_keys[tmp_1331 * WPT + 9u];
            let tmp_1360 = smem_vals[tmp_1331 * WPT + 9u];
            let tmp_1361 = keys[9] < tmp_1359 || (keys[9] == tmp_1359 && values[9] < tmp_1360);
            if tmp_1330 == tmp_1361 { keys[9] = tmp_1359; values[9] = tmp_1360; }
            let tmp_1362 = smem_keys[tmp_1331 * WPT + 10u];
            let tmp_1363 = smem_vals[tmp_1331 * WPT + 10u];
            let tmp_1364 = keys[10] < tmp_1362 || (keys[10] == tmp_1362 && values[10] < tmp_1363);
            if tmp_1330 == tmp_1364 { keys[10] = tmp_1362; values[10] = tmp_1363; }
            let tmp_1365 = smem_keys[tmp_1331 * WPT + 11u];
            let tmp_1366 = smem_vals[tmp_1331 * WPT + 11u];
            let tmp_1367 = keys[11] < tmp_1365 || (keys[11] == tmp_1365 && values[11] < tmp_1366);
            if tmp_1330 == tmp_1367 { keys[11] = tmp_1365; values[11] = tmp_1366; }
            let tmp_1368 = smem_keys[tmp_1331 * WPT + 12u];
            let tmp_1369 = smem_vals[tmp_1331 * WPT + 12u];
            let tmp_1370 = keys[12] < tmp_1368 || (keys[12] == tmp_1368 && values[12] < tmp_1369);
            if tmp_1330 == tmp_1370 { keys[12] = tmp_1368; values[12] = tmp_1369; }
            let tmp_1371 = smem_keys[tmp_1331 * WPT + 13u];
            let tmp_1372 = smem_vals[tmp_1331 * WPT + 13u];
            let tmp_1373 = keys[13] < tmp_1371 || (keys[13] == tmp_1371 && values[13] < tmp_1372);
            if tmp_1330 == tmp_1373 { keys[13] = tmp_1371; values[13] = tmp_1372; }
            let tmp_1374 = smem_keys[tmp_1331 * WPT + 14u];
            let tmp_1375 = smem_vals[tmp_1331 * WPT + 14u];
            let tmp_1376 = keys[14] < tmp_1374 || (keys[14] == tmp_1374 && values[14] < tmp_1375);
            if tmp_1330 == tmp_1376 { keys[14] = tmp_1374; values[14] = tmp_1375; }
            let tmp_1377 = smem_keys[tmp_1331 * WPT + 15u];
            let tmp_1378 = smem_vals[tmp_1331 * WPT + 15u];
            let tmp_1379 = keys[15] < tmp_1377 || (keys[15] == tmp_1377 && values[15] < tmp_1378);
            if tmp_1330 == tmp_1379 { keys[15] = tmp_1377; values[15] = tmp_1378; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],4,2)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1380 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_1381 = seg_base + (local_tid ^ 4u);
            let tmp_1382 = smem_keys[tmp_1381 * WPT + 0u];
            let tmp_1383 = smem_vals[tmp_1381 * WPT + 0u];
            let tmp_1384 = keys[0] < tmp_1382 || (keys[0] == tmp_1382 && values[0] < tmp_1383);
            if tmp_1380 == tmp_1384 { keys[0] = tmp_1382; values[0] = tmp_1383; }
            let tmp_1385 = smem_keys[tmp_1381 * WPT + 1u];
            let tmp_1386 = smem_vals[tmp_1381 * WPT + 1u];
            let tmp_1387 = keys[1] < tmp_1385 || (keys[1] == tmp_1385 && values[1] < tmp_1386);
            if tmp_1380 == tmp_1387 { keys[1] = tmp_1385; values[1] = tmp_1386; }
            let tmp_1388 = smem_keys[tmp_1381 * WPT + 2u];
            let tmp_1389 = smem_vals[tmp_1381 * WPT + 2u];
            let tmp_1390 = keys[2] < tmp_1388 || (keys[2] == tmp_1388 && values[2] < tmp_1389);
            if tmp_1380 == tmp_1390 { keys[2] = tmp_1388; values[2] = tmp_1389; }
            let tmp_1391 = smem_keys[tmp_1381 * WPT + 3u];
            let tmp_1392 = smem_vals[tmp_1381 * WPT + 3u];
            let tmp_1393 = keys[3] < tmp_1391 || (keys[3] == tmp_1391 && values[3] < tmp_1392);
            if tmp_1380 == tmp_1393 { keys[3] = tmp_1391; values[3] = tmp_1392; }
            let tmp_1394 = smem_keys[tmp_1381 * WPT + 4u];
            let tmp_1395 = smem_vals[tmp_1381 * WPT + 4u];
            let tmp_1396 = keys[4] < tmp_1394 || (keys[4] == tmp_1394 && values[4] < tmp_1395);
            if tmp_1380 == tmp_1396 { keys[4] = tmp_1394; values[4] = tmp_1395; }
            let tmp_1397 = smem_keys[tmp_1381 * WPT + 5u];
            let tmp_1398 = smem_vals[tmp_1381 * WPT + 5u];
            let tmp_1399 = keys[5] < tmp_1397 || (keys[5] == tmp_1397 && values[5] < tmp_1398);
            if tmp_1380 == tmp_1399 { keys[5] = tmp_1397; values[5] = tmp_1398; }
            let tmp_1400 = smem_keys[tmp_1381 * WPT + 6u];
            let tmp_1401 = smem_vals[tmp_1381 * WPT + 6u];
            let tmp_1402 = keys[6] < tmp_1400 || (keys[6] == tmp_1400 && values[6] < tmp_1401);
            if tmp_1380 == tmp_1402 { keys[6] = tmp_1400; values[6] = tmp_1401; }
            let tmp_1403 = smem_keys[tmp_1381 * WPT + 7u];
            let tmp_1404 = smem_vals[tmp_1381 * WPT + 7u];
            let tmp_1405 = keys[7] < tmp_1403 || (keys[7] == tmp_1403 && values[7] < tmp_1404);
            if tmp_1380 == tmp_1405 { keys[7] = tmp_1403; values[7] = tmp_1404; }
            let tmp_1406 = smem_keys[tmp_1381 * WPT + 8u];
            let tmp_1407 = smem_vals[tmp_1381 * WPT + 8u];
            let tmp_1408 = keys[8] < tmp_1406 || (keys[8] == tmp_1406 && values[8] < tmp_1407);
            if tmp_1380 == tmp_1408 { keys[8] = tmp_1406; values[8] = tmp_1407; }
            let tmp_1409 = smem_keys[tmp_1381 * WPT + 9u];
            let tmp_1410 = smem_vals[tmp_1381 * WPT + 9u];
            let tmp_1411 = keys[9] < tmp_1409 || (keys[9] == tmp_1409 && values[9] < tmp_1410);
            if tmp_1380 == tmp_1411 { keys[9] = tmp_1409; values[9] = tmp_1410; }
            let tmp_1412 = smem_keys[tmp_1381 * WPT + 10u];
            let tmp_1413 = smem_vals[tmp_1381 * WPT + 10u];
            let tmp_1414 = keys[10] < tmp_1412 || (keys[10] == tmp_1412 && values[10] < tmp_1413);
            if tmp_1380 == tmp_1414 { keys[10] = tmp_1412; values[10] = tmp_1413; }
            let tmp_1415 = smem_keys[tmp_1381 * WPT + 11u];
            let tmp_1416 = smem_vals[tmp_1381 * WPT + 11u];
            let tmp_1417 = keys[11] < tmp_1415 || (keys[11] == tmp_1415 && values[11] < tmp_1416);
            if tmp_1380 == tmp_1417 { keys[11] = tmp_1415; values[11] = tmp_1416; }
            let tmp_1418 = smem_keys[tmp_1381 * WPT + 12u];
            let tmp_1419 = smem_vals[tmp_1381 * WPT + 12u];
            let tmp_1420 = keys[12] < tmp_1418 || (keys[12] == tmp_1418 && values[12] < tmp_1419);
            if tmp_1380 == tmp_1420 { keys[12] = tmp_1418; values[12] = tmp_1419; }
            let tmp_1421 = smem_keys[tmp_1381 * WPT + 13u];
            let tmp_1422 = smem_vals[tmp_1381 * WPT + 13u];
            let tmp_1423 = keys[13] < tmp_1421 || (keys[13] == tmp_1421 && values[13] < tmp_1422);
            if tmp_1380 == tmp_1423 { keys[13] = tmp_1421; values[13] = tmp_1422; }
            let tmp_1424 = smem_keys[tmp_1381 * WPT + 14u];
            let tmp_1425 = smem_vals[tmp_1381 * WPT + 14u];
            let tmp_1426 = keys[14] < tmp_1424 || (keys[14] == tmp_1424 && values[14] < tmp_1425);
            if tmp_1380 == tmp_1426 { keys[14] = tmp_1424; values[14] = tmp_1425; }
            let tmp_1427 = smem_keys[tmp_1381 * WPT + 15u];
            let tmp_1428 = smem_vals[tmp_1381 * WPT + 15u];
            let tmp_1429 = keys[15] < tmp_1427 || (keys[15] == tmp_1427 && values[15] < tmp_1428);
            if tmp_1380 == tmp_1429 { keys[15] = tmp_1427; values[15] = tmp_1428; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],2,1)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1430 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_1431 = seg_base + (local_tid ^ 2u);
            let tmp_1432 = smem_keys[tmp_1431 * WPT + 0u];
            let tmp_1433 = smem_vals[tmp_1431 * WPT + 0u];
            let tmp_1434 = keys[0] < tmp_1432 || (keys[0] == tmp_1432 && values[0] < tmp_1433);
            if tmp_1430 == tmp_1434 { keys[0] = tmp_1432; values[0] = tmp_1433; }
            let tmp_1435 = smem_keys[tmp_1431 * WPT + 1u];
            let tmp_1436 = smem_vals[tmp_1431 * WPT + 1u];
            let tmp_1437 = keys[1] < tmp_1435 || (keys[1] == tmp_1435 && values[1] < tmp_1436);
            if tmp_1430 == tmp_1437 { keys[1] = tmp_1435; values[1] = tmp_1436; }
            let tmp_1438 = smem_keys[tmp_1431 * WPT + 2u];
            let tmp_1439 = smem_vals[tmp_1431 * WPT + 2u];
            let tmp_1440 = keys[2] < tmp_1438 || (keys[2] == tmp_1438 && values[2] < tmp_1439);
            if tmp_1430 == tmp_1440 { keys[2] = tmp_1438; values[2] = tmp_1439; }
            let tmp_1441 = smem_keys[tmp_1431 * WPT + 3u];
            let tmp_1442 = smem_vals[tmp_1431 * WPT + 3u];
            let tmp_1443 = keys[3] < tmp_1441 || (keys[3] == tmp_1441 && values[3] < tmp_1442);
            if tmp_1430 == tmp_1443 { keys[3] = tmp_1441; values[3] = tmp_1442; }
            let tmp_1444 = smem_keys[tmp_1431 * WPT + 4u];
            let tmp_1445 = smem_vals[tmp_1431 * WPT + 4u];
            let tmp_1446 = keys[4] < tmp_1444 || (keys[4] == tmp_1444 && values[4] < tmp_1445);
            if tmp_1430 == tmp_1446 { keys[4] = tmp_1444; values[4] = tmp_1445; }
            let tmp_1447 = smem_keys[tmp_1431 * WPT + 5u];
            let tmp_1448 = smem_vals[tmp_1431 * WPT + 5u];
            let tmp_1449 = keys[5] < tmp_1447 || (keys[5] == tmp_1447 && values[5] < tmp_1448);
            if tmp_1430 == tmp_1449 { keys[5] = tmp_1447; values[5] = tmp_1448; }
            let tmp_1450 = smem_keys[tmp_1431 * WPT + 6u];
            let tmp_1451 = smem_vals[tmp_1431 * WPT + 6u];
            let tmp_1452 = keys[6] < tmp_1450 || (keys[6] == tmp_1450 && values[6] < tmp_1451);
            if tmp_1430 == tmp_1452 { keys[6] = tmp_1450; values[6] = tmp_1451; }
            let tmp_1453 = smem_keys[tmp_1431 * WPT + 7u];
            let tmp_1454 = smem_vals[tmp_1431 * WPT + 7u];
            let tmp_1455 = keys[7] < tmp_1453 || (keys[7] == tmp_1453 && values[7] < tmp_1454);
            if tmp_1430 == tmp_1455 { keys[7] = tmp_1453; values[7] = tmp_1454; }
            let tmp_1456 = smem_keys[tmp_1431 * WPT + 8u];
            let tmp_1457 = smem_vals[tmp_1431 * WPT + 8u];
            let tmp_1458 = keys[8] < tmp_1456 || (keys[8] == tmp_1456 && values[8] < tmp_1457);
            if tmp_1430 == tmp_1458 { keys[8] = tmp_1456; values[8] = tmp_1457; }
            let tmp_1459 = smem_keys[tmp_1431 * WPT + 9u];
            let tmp_1460 = smem_vals[tmp_1431 * WPT + 9u];
            let tmp_1461 = keys[9] < tmp_1459 || (keys[9] == tmp_1459 && values[9] < tmp_1460);
            if tmp_1430 == tmp_1461 { keys[9] = tmp_1459; values[9] = tmp_1460; }
            let tmp_1462 = smem_keys[tmp_1431 * WPT + 10u];
            let tmp_1463 = smem_vals[tmp_1431 * WPT + 10u];
            let tmp_1464 = keys[10] < tmp_1462 || (keys[10] == tmp_1462 && values[10] < tmp_1463);
            if tmp_1430 == tmp_1464 { keys[10] = tmp_1462; values[10] = tmp_1463; }
            let tmp_1465 = smem_keys[tmp_1431 * WPT + 11u];
            let tmp_1466 = smem_vals[tmp_1431 * WPT + 11u];
            let tmp_1467 = keys[11] < tmp_1465 || (keys[11] == tmp_1465 && values[11] < tmp_1466);
            if tmp_1430 == tmp_1467 { keys[11] = tmp_1465; values[11] = tmp_1466; }
            let tmp_1468 = smem_keys[tmp_1431 * WPT + 12u];
            let tmp_1469 = smem_vals[tmp_1431 * WPT + 12u];
            let tmp_1470 = keys[12] < tmp_1468 || (keys[12] == tmp_1468 && values[12] < tmp_1469);
            if tmp_1430 == tmp_1470 { keys[12] = tmp_1468; values[12] = tmp_1469; }
            let tmp_1471 = smem_keys[tmp_1431 * WPT + 13u];
            let tmp_1472 = smem_vals[tmp_1431 * WPT + 13u];
            let tmp_1473 = keys[13] < tmp_1471 || (keys[13] == tmp_1471 && values[13] < tmp_1472);
            if tmp_1430 == tmp_1473 { keys[13] = tmp_1471; values[13] = tmp_1472; }
            let tmp_1474 = smem_keys[tmp_1431 * WPT + 14u];
            let tmp_1475 = smem_vals[tmp_1431 * WPT + 14u];
            let tmp_1476 = keys[14] < tmp_1474 || (keys[14] == tmp_1474 && values[14] < tmp_1475);
            if tmp_1430 == tmp_1476 { keys[14] = tmp_1474; values[14] = tmp_1475; }
            let tmp_1477 = smem_keys[tmp_1431 * WPT + 15u];
            let tmp_1478 = smem_vals[tmp_1431 * WPT + 15u];
            let tmp_1479 = keys[15] < tmp_1477 || (keys[15] == tmp_1477 && values[15] < tmp_1478);
            if tmp_1430 == tmp_1479 { keys[15] = tmp_1477; values[15] = tmp_1478; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],1,0)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1480 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_1481 = seg_base + (local_tid ^ 1u);
            let tmp_1482 = smem_keys[tmp_1481 * WPT + 0u];
            let tmp_1483 = smem_vals[tmp_1481 * WPT + 0u];
            let tmp_1484 = keys[0] < tmp_1482 || (keys[0] == tmp_1482 && values[0] < tmp_1483);
            if tmp_1480 == tmp_1484 { keys[0] = tmp_1482; values[0] = tmp_1483; }
            let tmp_1485 = smem_keys[tmp_1481 * WPT + 1u];
            let tmp_1486 = smem_vals[tmp_1481 * WPT + 1u];
            let tmp_1487 = keys[1] < tmp_1485 || (keys[1] == tmp_1485 && values[1] < tmp_1486);
            if tmp_1480 == tmp_1487 { keys[1] = tmp_1485; values[1] = tmp_1486; }
            let tmp_1488 = smem_keys[tmp_1481 * WPT + 2u];
            let tmp_1489 = smem_vals[tmp_1481 * WPT + 2u];
            let tmp_1490 = keys[2] < tmp_1488 || (keys[2] == tmp_1488 && values[2] < tmp_1489);
            if tmp_1480 == tmp_1490 { keys[2] = tmp_1488; values[2] = tmp_1489; }
            let tmp_1491 = smem_keys[tmp_1481 * WPT + 3u];
            let tmp_1492 = smem_vals[tmp_1481 * WPT + 3u];
            let tmp_1493 = keys[3] < tmp_1491 || (keys[3] == tmp_1491 && values[3] < tmp_1492);
            if tmp_1480 == tmp_1493 { keys[3] = tmp_1491; values[3] = tmp_1492; }
            let tmp_1494 = smem_keys[tmp_1481 * WPT + 4u];
            let tmp_1495 = smem_vals[tmp_1481 * WPT + 4u];
            let tmp_1496 = keys[4] < tmp_1494 || (keys[4] == tmp_1494 && values[4] < tmp_1495);
            if tmp_1480 == tmp_1496 { keys[4] = tmp_1494; values[4] = tmp_1495; }
            let tmp_1497 = smem_keys[tmp_1481 * WPT + 5u];
            let tmp_1498 = smem_vals[tmp_1481 * WPT + 5u];
            let tmp_1499 = keys[5] < tmp_1497 || (keys[5] == tmp_1497 && values[5] < tmp_1498);
            if tmp_1480 == tmp_1499 { keys[5] = tmp_1497; values[5] = tmp_1498; }
            let tmp_1500 = smem_keys[tmp_1481 * WPT + 6u];
            let tmp_1501 = smem_vals[tmp_1481 * WPT + 6u];
            let tmp_1502 = keys[6] < tmp_1500 || (keys[6] == tmp_1500 && values[6] < tmp_1501);
            if tmp_1480 == tmp_1502 { keys[6] = tmp_1500; values[6] = tmp_1501; }
            let tmp_1503 = smem_keys[tmp_1481 * WPT + 7u];
            let tmp_1504 = smem_vals[tmp_1481 * WPT + 7u];
            let tmp_1505 = keys[7] < tmp_1503 || (keys[7] == tmp_1503 && values[7] < tmp_1504);
            if tmp_1480 == tmp_1505 { keys[7] = tmp_1503; values[7] = tmp_1504; }
            let tmp_1506 = smem_keys[tmp_1481 * WPT + 8u];
            let tmp_1507 = smem_vals[tmp_1481 * WPT + 8u];
            let tmp_1508 = keys[8] < tmp_1506 || (keys[8] == tmp_1506 && values[8] < tmp_1507);
            if tmp_1480 == tmp_1508 { keys[8] = tmp_1506; values[8] = tmp_1507; }
            let tmp_1509 = smem_keys[tmp_1481 * WPT + 9u];
            let tmp_1510 = smem_vals[tmp_1481 * WPT + 9u];
            let tmp_1511 = keys[9] < tmp_1509 || (keys[9] == tmp_1509 && values[9] < tmp_1510);
            if tmp_1480 == tmp_1511 { keys[9] = tmp_1509; values[9] = tmp_1510; }
            let tmp_1512 = smem_keys[tmp_1481 * WPT + 10u];
            let tmp_1513 = smem_vals[tmp_1481 * WPT + 10u];
            let tmp_1514 = keys[10] < tmp_1512 || (keys[10] == tmp_1512 && values[10] < tmp_1513);
            if tmp_1480 == tmp_1514 { keys[10] = tmp_1512; values[10] = tmp_1513; }
            let tmp_1515 = smem_keys[tmp_1481 * WPT + 11u];
            let tmp_1516 = smem_vals[tmp_1481 * WPT + 11u];
            let tmp_1517 = keys[11] < tmp_1515 || (keys[11] == tmp_1515 && values[11] < tmp_1516);
            if tmp_1480 == tmp_1517 { keys[11] = tmp_1515; values[11] = tmp_1516; }
            let tmp_1518 = smem_keys[tmp_1481 * WPT + 12u];
            let tmp_1519 = smem_vals[tmp_1481 * WPT + 12u];
            let tmp_1520 = keys[12] < tmp_1518 || (keys[12] == tmp_1518 && values[12] < tmp_1519);
            if tmp_1480 == tmp_1520 { keys[12] = tmp_1518; values[12] = tmp_1519; }
            let tmp_1521 = smem_keys[tmp_1481 * WPT + 13u];
            let tmp_1522 = smem_vals[tmp_1481 * WPT + 13u];
            let tmp_1523 = keys[13] < tmp_1521 || (keys[13] == tmp_1521 && values[13] < tmp_1522);
            if tmp_1480 == tmp_1523 { keys[13] = tmp_1521; values[13] = tmp_1522; }
            let tmp_1524 = smem_keys[tmp_1481 * WPT + 14u];
            let tmp_1525 = smem_vals[tmp_1481 * WPT + 14u];
            let tmp_1526 = keys[14] < tmp_1524 || (keys[14] == tmp_1524 && values[14] < tmp_1525);
            if tmp_1480 == tmp_1526 { keys[14] = tmp_1524; values[14] = tmp_1525; }
            let tmp_1527 = smem_keys[tmp_1481 * WPT + 15u];
            let tmp_1528 = smem_vals[tmp_1481 * WPT + 15u];
            let tmp_1529 = keys[15] < tmp_1527 || (keys[15] == tmp_1527 && values[15] < tmp_1528);
            if tmp_1480 == tmp_1529 { keys[15] = tmp_1527; values[15] = tmp_1528; }
            workgroupBarrier();
        }
    }

    // exch_local(8,16)
    {
        // cmp_swap(0,8)
        if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
            // swap(0,8)
            {
                let tmp_1530 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_1530;
                let tmp_1531 = values[0]; values[0] = values[8]; values[8] = tmp_1531;
            }
        }
        // cmp_swap(1,9)
        if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
            // swap(1,9)
            {
                let tmp_1532 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_1532;
                let tmp_1533 = values[1]; values[1] = values[9]; values[9] = tmp_1533;
            }
        }
        // cmp_swap(2,10)
        if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
            // swap(2,10)
            {
                let tmp_1534 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_1534;
                let tmp_1535 = values[2]; values[2] = values[10]; values[10] = tmp_1535;
            }
        }
        // cmp_swap(3,11)
        if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
            // swap(3,11)
            {
                let tmp_1536 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_1536;
                let tmp_1537 = values[3]; values[3] = values[11]; values[11] = tmp_1537;
            }
        }
        // cmp_swap(4,12)
        if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
            // swap(4,12)
            {
                let tmp_1538 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_1538;
                let tmp_1539 = values[4]; values[4] = values[12]; values[12] = tmp_1539;
            }
        }
        // cmp_swap(5,13)
        if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
            // swap(5,13)
            {
                let tmp_1540 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_1540;
                let tmp_1541 = values[5]; values[5] = values[13]; values[13] = tmp_1541;
            }
        }
        // cmp_swap(6,14)
        if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
            // swap(6,14)
            {
                let tmp_1542 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_1542;
                let tmp_1543 = values[6]; values[6] = values[14]; values[14] = tmp_1543;
            }
        }
        // cmp_swap(7,15)
        if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
            // swap(7,15)
            {
                let tmp_1544 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_1544;
                let tmp_1545 = values[7]; values[7] = values[15]; values[15] = tmp_1545;
            }
        }
    }

    // exch_local(4,16)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_1546 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_1546;
                let tmp_1547 = values[0]; values[0] = values[4]; values[4] = tmp_1547;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_1548 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_1548;
                let tmp_1549 = values[1]; values[1] = values[5]; values[5] = tmp_1549;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_1550 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_1550;
                let tmp_1551 = values[2]; values[2] = values[6]; values[6] = tmp_1551;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_1552 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_1552;
                let tmp_1553 = values[3]; values[3] = values[7]; values[7] = tmp_1553;
            }
        }
        // cmp_swap(8,12)
        if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
            // swap(8,12)
            {
                let tmp_1554 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_1554;
                let tmp_1555 = values[8]; values[8] = values[12]; values[12] = tmp_1555;
            }
        }
        // cmp_swap(9,13)
        if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
            // swap(9,13)
            {
                let tmp_1556 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_1556;
                let tmp_1557 = values[9]; values[9] = values[13]; values[13] = tmp_1557;
            }
        }
        // cmp_swap(10,14)
        if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
            // swap(10,14)
            {
                let tmp_1558 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_1558;
                let tmp_1559 = values[10]; values[10] = values[14]; values[14] = tmp_1559;
            }
        }
        // cmp_swap(11,15)
        if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
            // swap(11,15)
            {
                let tmp_1560 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_1560;
                let tmp_1561 = values[11]; values[11] = values[15]; values[15] = tmp_1561;
            }
        }
    }

    // exch_local(2,16)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_1562 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_1562;
                let tmp_1563 = values[0]; values[0] = values[2]; values[2] = tmp_1563;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_1564 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_1564;
                let tmp_1565 = values[1]; values[1] = values[3]; values[3] = tmp_1565;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_1566 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_1566;
                let tmp_1567 = values[4]; values[4] = values[6]; values[6] = tmp_1567;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_1568 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_1568;
                let tmp_1569 = values[5]; values[5] = values[7]; values[7] = tmp_1569;
            }
        }
        // cmp_swap(8,10)
        if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
            // swap(8,10)
            {
                let tmp_1570 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_1570;
                let tmp_1571 = values[8]; values[8] = values[10]; values[10] = tmp_1571;
            }
        }
        // cmp_swap(9,11)
        if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
            // swap(9,11)
            {
                let tmp_1572 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_1572;
                let tmp_1573 = values[9]; values[9] = values[11]; values[11] = tmp_1573;
            }
        }
        // cmp_swap(12,14)
        if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
            // swap(12,14)
            {
                let tmp_1574 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_1574;
                let tmp_1575 = values[12]; values[12] = values[14]; values[14] = tmp_1575;
            }
        }
        // cmp_swap(13,15)
        if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
            // swap(13,15)
            {
                let tmp_1576 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_1576;
                let tmp_1577 = values[13]; values[13] = values[15]; values[15] = tmp_1577;
            }
        }
    }

    // exch_local(1,16)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_1578 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_1578;
                let tmp_1579 = values[0]; values[0] = values[1]; values[1] = tmp_1579;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_1580 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_1580;
                let tmp_1581 = values[2]; values[2] = values[3]; values[3] = tmp_1581;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_1582 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_1582;
                let tmp_1583 = values[4]; values[4] = values[5]; values[5] = tmp_1583;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_1584 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_1584;
                let tmp_1585 = values[6]; values[6] = values[7]; values[7] = tmp_1585;
            }
        }
        // cmp_swap(8,9)
        if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
            // swap(8,9)
            {
                let tmp_1586 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_1586;
                let tmp_1587 = values[8]; values[8] = values[9]; values[9] = tmp_1587;
            }
        }
        // cmp_swap(10,11)
        if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
            // swap(10,11)
            {
                let tmp_1588 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_1588;
                let tmp_1589 = values[10]; values[10] = values[11]; values[11] = tmp_1589;
            }
        }
        // cmp_swap(12,13)
        if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
            // swap(12,13)
            {
                let tmp_1590 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_1590;
                let tmp_1591 = values[12]; values[12] = values[13]; values[13] = tmp_1591;
            }
        }
        // cmp_swap(14,15)
        if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
            // swap(14,15)
            {
                let tmp_1592 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_1592;
                let tmp_1593 = values[14]; values[14] = values[15]; values[15] = tmp_1593;
            }
        }
    }

    // exch_intxn(tmask:127,swbit:6,wpt:16)
    {
        // _exch_workgroup([(0, 15), (1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8), (8, 7), (9, 6), (10, 5), (11, 4), (12, 3), (13, 2), (14, 1), (15, 0)],127,6)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1594 = extractBits(local_tid, 6u, 1u) != 0u;
            let tmp_1595 = seg_base + (local_tid ^ 127u);
            let tmp_1596 = smem_keys[tmp_1595 * WPT + 15u];
            let tmp_1597 = smem_vals[tmp_1595 * WPT + 15u];
            let tmp_1598 = keys[0] < tmp_1596 || (keys[0] == tmp_1596 && values[0] < tmp_1597);
            if tmp_1594 == tmp_1598 { keys[0] = tmp_1596; values[0] = tmp_1597; }
            let tmp_1599 = smem_keys[tmp_1595 * WPT + 14u];
            let tmp_1600 = smem_vals[tmp_1595 * WPT + 14u];
            let tmp_1601 = keys[1] < tmp_1599 || (keys[1] == tmp_1599 && values[1] < tmp_1600);
            if tmp_1594 == tmp_1601 { keys[1] = tmp_1599; values[1] = tmp_1600; }
            let tmp_1602 = smem_keys[tmp_1595 * WPT + 13u];
            let tmp_1603 = smem_vals[tmp_1595 * WPT + 13u];
            let tmp_1604 = keys[2] < tmp_1602 || (keys[2] == tmp_1602 && values[2] < tmp_1603);
            if tmp_1594 == tmp_1604 { keys[2] = tmp_1602; values[2] = tmp_1603; }
            let tmp_1605 = smem_keys[tmp_1595 * WPT + 12u];
            let tmp_1606 = smem_vals[tmp_1595 * WPT + 12u];
            let tmp_1607 = keys[3] < tmp_1605 || (keys[3] == tmp_1605 && values[3] < tmp_1606);
            if tmp_1594 == tmp_1607 { keys[3] = tmp_1605; values[3] = tmp_1606; }
            let tmp_1608 = smem_keys[tmp_1595 * WPT + 11u];
            let tmp_1609 = smem_vals[tmp_1595 * WPT + 11u];
            let tmp_1610 = keys[4] < tmp_1608 || (keys[4] == tmp_1608 && values[4] < tmp_1609);
            if tmp_1594 == tmp_1610 { keys[4] = tmp_1608; values[4] = tmp_1609; }
            let tmp_1611 = smem_keys[tmp_1595 * WPT + 10u];
            let tmp_1612 = smem_vals[tmp_1595 * WPT + 10u];
            let tmp_1613 = keys[5] < tmp_1611 || (keys[5] == tmp_1611 && values[5] < tmp_1612);
            if tmp_1594 == tmp_1613 { keys[5] = tmp_1611; values[5] = tmp_1612; }
            let tmp_1614 = smem_keys[tmp_1595 * WPT + 9u];
            let tmp_1615 = smem_vals[tmp_1595 * WPT + 9u];
            let tmp_1616 = keys[6] < tmp_1614 || (keys[6] == tmp_1614 && values[6] < tmp_1615);
            if tmp_1594 == tmp_1616 { keys[6] = tmp_1614; values[6] = tmp_1615; }
            let tmp_1617 = smem_keys[tmp_1595 * WPT + 8u];
            let tmp_1618 = smem_vals[tmp_1595 * WPT + 8u];
            let tmp_1619 = keys[7] < tmp_1617 || (keys[7] == tmp_1617 && values[7] < tmp_1618);
            if tmp_1594 == tmp_1619 { keys[7] = tmp_1617; values[7] = tmp_1618; }
            let tmp_1620 = smem_keys[tmp_1595 * WPT + 7u];
            let tmp_1621 = smem_vals[tmp_1595 * WPT + 7u];
            let tmp_1622 = keys[8] < tmp_1620 || (keys[8] == tmp_1620 && values[8] < tmp_1621);
            if tmp_1594 == tmp_1622 { keys[8] = tmp_1620; values[8] = tmp_1621; }
            let tmp_1623 = smem_keys[tmp_1595 * WPT + 6u];
            let tmp_1624 = smem_vals[tmp_1595 * WPT + 6u];
            let tmp_1625 = keys[9] < tmp_1623 || (keys[9] == tmp_1623 && values[9] < tmp_1624);
            if tmp_1594 == tmp_1625 { keys[9] = tmp_1623; values[9] = tmp_1624; }
            let tmp_1626 = smem_keys[tmp_1595 * WPT + 5u];
            let tmp_1627 = smem_vals[tmp_1595 * WPT + 5u];
            let tmp_1628 = keys[10] < tmp_1626 || (keys[10] == tmp_1626 && values[10] < tmp_1627);
            if tmp_1594 == tmp_1628 { keys[10] = tmp_1626; values[10] = tmp_1627; }
            let tmp_1629 = smem_keys[tmp_1595 * WPT + 4u];
            let tmp_1630 = smem_vals[tmp_1595 * WPT + 4u];
            let tmp_1631 = keys[11] < tmp_1629 || (keys[11] == tmp_1629 && values[11] < tmp_1630);
            if tmp_1594 == tmp_1631 { keys[11] = tmp_1629; values[11] = tmp_1630; }
            let tmp_1632 = smem_keys[tmp_1595 * WPT + 3u];
            let tmp_1633 = smem_vals[tmp_1595 * WPT + 3u];
            let tmp_1634 = keys[12] < tmp_1632 || (keys[12] == tmp_1632 && values[12] < tmp_1633);
            if tmp_1594 == tmp_1634 { keys[12] = tmp_1632; values[12] = tmp_1633; }
            let tmp_1635 = smem_keys[tmp_1595 * WPT + 2u];
            let tmp_1636 = smem_vals[tmp_1595 * WPT + 2u];
            let tmp_1637 = keys[13] < tmp_1635 || (keys[13] == tmp_1635 && values[13] < tmp_1636);
            if tmp_1594 == tmp_1637 { keys[13] = tmp_1635; values[13] = tmp_1636; }
            let tmp_1638 = smem_keys[tmp_1595 * WPT + 1u];
            let tmp_1639 = smem_vals[tmp_1595 * WPT + 1u];
            let tmp_1640 = keys[14] < tmp_1638 || (keys[14] == tmp_1638 && values[14] < tmp_1639);
            if tmp_1594 == tmp_1640 { keys[14] = tmp_1638; values[14] = tmp_1639; }
            let tmp_1641 = smem_keys[tmp_1595 * WPT + 0u];
            let tmp_1642 = smem_vals[tmp_1595 * WPT + 0u];
            let tmp_1643 = keys[15] < tmp_1641 || (keys[15] == tmp_1641 && values[15] < tmp_1642);
            if tmp_1594 == tmp_1643 { keys[15] = tmp_1641; values[15] = tmp_1642; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:32,swbit:5,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],32,5)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1644 = extractBits(local_tid, 5u, 1u) != 0u;
            let tmp_1645 = seg_base + (local_tid ^ 32u);
            let tmp_1646 = smem_keys[tmp_1645 * WPT + 0u];
            let tmp_1647 = smem_vals[tmp_1645 * WPT + 0u];
            let tmp_1648 = keys[0] < tmp_1646 || (keys[0] == tmp_1646 && values[0] < tmp_1647);
            if tmp_1644 == tmp_1648 { keys[0] = tmp_1646; values[0] = tmp_1647; }
            let tmp_1649 = smem_keys[tmp_1645 * WPT + 1u];
            let tmp_1650 = smem_vals[tmp_1645 * WPT + 1u];
            let tmp_1651 = keys[1] < tmp_1649 || (keys[1] == tmp_1649 && values[1] < tmp_1650);
            if tmp_1644 == tmp_1651 { keys[1] = tmp_1649; values[1] = tmp_1650; }
            let tmp_1652 = smem_keys[tmp_1645 * WPT + 2u];
            let tmp_1653 = smem_vals[tmp_1645 * WPT + 2u];
            let tmp_1654 = keys[2] < tmp_1652 || (keys[2] == tmp_1652 && values[2] < tmp_1653);
            if tmp_1644 == tmp_1654 { keys[2] = tmp_1652; values[2] = tmp_1653; }
            let tmp_1655 = smem_keys[tmp_1645 * WPT + 3u];
            let tmp_1656 = smem_vals[tmp_1645 * WPT + 3u];
            let tmp_1657 = keys[3] < tmp_1655 || (keys[3] == tmp_1655 && values[3] < tmp_1656);
            if tmp_1644 == tmp_1657 { keys[3] = tmp_1655; values[3] = tmp_1656; }
            let tmp_1658 = smem_keys[tmp_1645 * WPT + 4u];
            let tmp_1659 = smem_vals[tmp_1645 * WPT + 4u];
            let tmp_1660 = keys[4] < tmp_1658 || (keys[4] == tmp_1658 && values[4] < tmp_1659);
            if tmp_1644 == tmp_1660 { keys[4] = tmp_1658; values[4] = tmp_1659; }
            let tmp_1661 = smem_keys[tmp_1645 * WPT + 5u];
            let tmp_1662 = smem_vals[tmp_1645 * WPT + 5u];
            let tmp_1663 = keys[5] < tmp_1661 || (keys[5] == tmp_1661 && values[5] < tmp_1662);
            if tmp_1644 == tmp_1663 { keys[5] = tmp_1661; values[5] = tmp_1662; }
            let tmp_1664 = smem_keys[tmp_1645 * WPT + 6u];
            let tmp_1665 = smem_vals[tmp_1645 * WPT + 6u];
            let tmp_1666 = keys[6] < tmp_1664 || (keys[6] == tmp_1664 && values[6] < tmp_1665);
            if tmp_1644 == tmp_1666 { keys[6] = tmp_1664; values[6] = tmp_1665; }
            let tmp_1667 = smem_keys[tmp_1645 * WPT + 7u];
            let tmp_1668 = smem_vals[tmp_1645 * WPT + 7u];
            let tmp_1669 = keys[7] < tmp_1667 || (keys[7] == tmp_1667 && values[7] < tmp_1668);
            if tmp_1644 == tmp_1669 { keys[7] = tmp_1667; values[7] = tmp_1668; }
            let tmp_1670 = smem_keys[tmp_1645 * WPT + 8u];
            let tmp_1671 = smem_vals[tmp_1645 * WPT + 8u];
            let tmp_1672 = keys[8] < tmp_1670 || (keys[8] == tmp_1670 && values[8] < tmp_1671);
            if tmp_1644 == tmp_1672 { keys[8] = tmp_1670; values[8] = tmp_1671; }
            let tmp_1673 = smem_keys[tmp_1645 * WPT + 9u];
            let tmp_1674 = smem_vals[tmp_1645 * WPT + 9u];
            let tmp_1675 = keys[9] < tmp_1673 || (keys[9] == tmp_1673 && values[9] < tmp_1674);
            if tmp_1644 == tmp_1675 { keys[9] = tmp_1673; values[9] = tmp_1674; }
            let tmp_1676 = smem_keys[tmp_1645 * WPT + 10u];
            let tmp_1677 = smem_vals[tmp_1645 * WPT + 10u];
            let tmp_1678 = keys[10] < tmp_1676 || (keys[10] == tmp_1676 && values[10] < tmp_1677);
            if tmp_1644 == tmp_1678 { keys[10] = tmp_1676; values[10] = tmp_1677; }
            let tmp_1679 = smem_keys[tmp_1645 * WPT + 11u];
            let tmp_1680 = smem_vals[tmp_1645 * WPT + 11u];
            let tmp_1681 = keys[11] < tmp_1679 || (keys[11] == tmp_1679 && values[11] < tmp_1680);
            if tmp_1644 == tmp_1681 { keys[11] = tmp_1679; values[11] = tmp_1680; }
            let tmp_1682 = smem_keys[tmp_1645 * WPT + 12u];
            let tmp_1683 = smem_vals[tmp_1645 * WPT + 12u];
            let tmp_1684 = keys[12] < tmp_1682 || (keys[12] == tmp_1682 && values[12] < tmp_1683);
            if tmp_1644 == tmp_1684 { keys[12] = tmp_1682; values[12] = tmp_1683; }
            let tmp_1685 = smem_keys[tmp_1645 * WPT + 13u];
            let tmp_1686 = smem_vals[tmp_1645 * WPT + 13u];
            let tmp_1687 = keys[13] < tmp_1685 || (keys[13] == tmp_1685 && values[13] < tmp_1686);
            if tmp_1644 == tmp_1687 { keys[13] = tmp_1685; values[13] = tmp_1686; }
            let tmp_1688 = smem_keys[tmp_1645 * WPT + 14u];
            let tmp_1689 = smem_vals[tmp_1645 * WPT + 14u];
            let tmp_1690 = keys[14] < tmp_1688 || (keys[14] == tmp_1688 && values[14] < tmp_1689);
            if tmp_1644 == tmp_1690 { keys[14] = tmp_1688; values[14] = tmp_1689; }
            let tmp_1691 = smem_keys[tmp_1645 * WPT + 15u];
            let tmp_1692 = smem_vals[tmp_1645 * WPT + 15u];
            let tmp_1693 = keys[15] < tmp_1691 || (keys[15] == tmp_1691 && values[15] < tmp_1692);
            if tmp_1644 == tmp_1693 { keys[15] = tmp_1691; values[15] = tmp_1692; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:16,swbit:4,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],16,4)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1694 = extractBits(local_tid, 4u, 1u) != 0u;
            let tmp_1695 = seg_base + (local_tid ^ 16u);
            let tmp_1696 = smem_keys[tmp_1695 * WPT + 0u];
            let tmp_1697 = smem_vals[tmp_1695 * WPT + 0u];
            let tmp_1698 = keys[0] < tmp_1696 || (keys[0] == tmp_1696 && values[0] < tmp_1697);
            if tmp_1694 == tmp_1698 { keys[0] = tmp_1696; values[0] = tmp_1697; }
            let tmp_1699 = smem_keys[tmp_1695 * WPT + 1u];
            let tmp_1700 = smem_vals[tmp_1695 * WPT + 1u];
            let tmp_1701 = keys[1] < tmp_1699 || (keys[1] == tmp_1699 && values[1] < tmp_1700);
            if tmp_1694 == tmp_1701 { keys[1] = tmp_1699; values[1] = tmp_1700; }
            let tmp_1702 = smem_keys[tmp_1695 * WPT + 2u];
            let tmp_1703 = smem_vals[tmp_1695 * WPT + 2u];
            let tmp_1704 = keys[2] < tmp_1702 || (keys[2] == tmp_1702 && values[2] < tmp_1703);
            if tmp_1694 == tmp_1704 { keys[2] = tmp_1702; values[2] = tmp_1703; }
            let tmp_1705 = smem_keys[tmp_1695 * WPT + 3u];
            let tmp_1706 = smem_vals[tmp_1695 * WPT + 3u];
            let tmp_1707 = keys[3] < tmp_1705 || (keys[3] == tmp_1705 && values[3] < tmp_1706);
            if tmp_1694 == tmp_1707 { keys[3] = tmp_1705; values[3] = tmp_1706; }
            let tmp_1708 = smem_keys[tmp_1695 * WPT + 4u];
            let tmp_1709 = smem_vals[tmp_1695 * WPT + 4u];
            let tmp_1710 = keys[4] < tmp_1708 || (keys[4] == tmp_1708 && values[4] < tmp_1709);
            if tmp_1694 == tmp_1710 { keys[4] = tmp_1708; values[4] = tmp_1709; }
            let tmp_1711 = smem_keys[tmp_1695 * WPT + 5u];
            let tmp_1712 = smem_vals[tmp_1695 * WPT + 5u];
            let tmp_1713 = keys[5] < tmp_1711 || (keys[5] == tmp_1711 && values[5] < tmp_1712);
            if tmp_1694 == tmp_1713 { keys[5] = tmp_1711; values[5] = tmp_1712; }
            let tmp_1714 = smem_keys[tmp_1695 * WPT + 6u];
            let tmp_1715 = smem_vals[tmp_1695 * WPT + 6u];
            let tmp_1716 = keys[6] < tmp_1714 || (keys[6] == tmp_1714 && values[6] < tmp_1715);
            if tmp_1694 == tmp_1716 { keys[6] = tmp_1714; values[6] = tmp_1715; }
            let tmp_1717 = smem_keys[tmp_1695 * WPT + 7u];
            let tmp_1718 = smem_vals[tmp_1695 * WPT + 7u];
            let tmp_1719 = keys[7] < tmp_1717 || (keys[7] == tmp_1717 && values[7] < tmp_1718);
            if tmp_1694 == tmp_1719 { keys[7] = tmp_1717; values[7] = tmp_1718; }
            let tmp_1720 = smem_keys[tmp_1695 * WPT + 8u];
            let tmp_1721 = smem_vals[tmp_1695 * WPT + 8u];
            let tmp_1722 = keys[8] < tmp_1720 || (keys[8] == tmp_1720 && values[8] < tmp_1721);
            if tmp_1694 == tmp_1722 { keys[8] = tmp_1720; values[8] = tmp_1721; }
            let tmp_1723 = smem_keys[tmp_1695 * WPT + 9u];
            let tmp_1724 = smem_vals[tmp_1695 * WPT + 9u];
            let tmp_1725 = keys[9] < tmp_1723 || (keys[9] == tmp_1723 && values[9] < tmp_1724);
            if tmp_1694 == tmp_1725 { keys[9] = tmp_1723; values[9] = tmp_1724; }
            let tmp_1726 = smem_keys[tmp_1695 * WPT + 10u];
            let tmp_1727 = smem_vals[tmp_1695 * WPT + 10u];
            let tmp_1728 = keys[10] < tmp_1726 || (keys[10] == tmp_1726 && values[10] < tmp_1727);
            if tmp_1694 == tmp_1728 { keys[10] = tmp_1726; values[10] = tmp_1727; }
            let tmp_1729 = smem_keys[tmp_1695 * WPT + 11u];
            let tmp_1730 = smem_vals[tmp_1695 * WPT + 11u];
            let tmp_1731 = keys[11] < tmp_1729 || (keys[11] == tmp_1729 && values[11] < tmp_1730);
            if tmp_1694 == tmp_1731 { keys[11] = tmp_1729; values[11] = tmp_1730; }
            let tmp_1732 = smem_keys[tmp_1695 * WPT + 12u];
            let tmp_1733 = smem_vals[tmp_1695 * WPT + 12u];
            let tmp_1734 = keys[12] < tmp_1732 || (keys[12] == tmp_1732 && values[12] < tmp_1733);
            if tmp_1694 == tmp_1734 { keys[12] = tmp_1732; values[12] = tmp_1733; }
            let tmp_1735 = smem_keys[tmp_1695 * WPT + 13u];
            let tmp_1736 = smem_vals[tmp_1695 * WPT + 13u];
            let tmp_1737 = keys[13] < tmp_1735 || (keys[13] == tmp_1735 && values[13] < tmp_1736);
            if tmp_1694 == tmp_1737 { keys[13] = tmp_1735; values[13] = tmp_1736; }
            let tmp_1738 = smem_keys[tmp_1695 * WPT + 14u];
            let tmp_1739 = smem_vals[tmp_1695 * WPT + 14u];
            let tmp_1740 = keys[14] < tmp_1738 || (keys[14] == tmp_1738 && values[14] < tmp_1739);
            if tmp_1694 == tmp_1740 { keys[14] = tmp_1738; values[14] = tmp_1739; }
            let tmp_1741 = smem_keys[tmp_1695 * WPT + 15u];
            let tmp_1742 = smem_vals[tmp_1695 * WPT + 15u];
            let tmp_1743 = keys[15] < tmp_1741 || (keys[15] == tmp_1741 && values[15] < tmp_1742);
            if tmp_1694 == tmp_1743 { keys[15] = tmp_1741; values[15] = tmp_1742; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:8,swbit:3,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],8,3)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1744 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_1745 = seg_base + (local_tid ^ 8u);
            let tmp_1746 = smem_keys[tmp_1745 * WPT + 0u];
            let tmp_1747 = smem_vals[tmp_1745 * WPT + 0u];
            let tmp_1748 = keys[0] < tmp_1746 || (keys[0] == tmp_1746 && values[0] < tmp_1747);
            if tmp_1744 == tmp_1748 { keys[0] = tmp_1746; values[0] = tmp_1747; }
            let tmp_1749 = smem_keys[tmp_1745 * WPT + 1u];
            let tmp_1750 = smem_vals[tmp_1745 * WPT + 1u];
            let tmp_1751 = keys[1] < tmp_1749 || (keys[1] == tmp_1749 && values[1] < tmp_1750);
            if tmp_1744 == tmp_1751 { keys[1] = tmp_1749; values[1] = tmp_1750; }
            let tmp_1752 = smem_keys[tmp_1745 * WPT + 2u];
            let tmp_1753 = smem_vals[tmp_1745 * WPT + 2u];
            let tmp_1754 = keys[2] < tmp_1752 || (keys[2] == tmp_1752 && values[2] < tmp_1753);
            if tmp_1744 == tmp_1754 { keys[2] = tmp_1752; values[2] = tmp_1753; }
            let tmp_1755 = smem_keys[tmp_1745 * WPT + 3u];
            let tmp_1756 = smem_vals[tmp_1745 * WPT + 3u];
            let tmp_1757 = keys[3] < tmp_1755 || (keys[3] == tmp_1755 && values[3] < tmp_1756);
            if tmp_1744 == tmp_1757 { keys[3] = tmp_1755; values[3] = tmp_1756; }
            let tmp_1758 = smem_keys[tmp_1745 * WPT + 4u];
            let tmp_1759 = smem_vals[tmp_1745 * WPT + 4u];
            let tmp_1760 = keys[4] < tmp_1758 || (keys[4] == tmp_1758 && values[4] < tmp_1759);
            if tmp_1744 == tmp_1760 { keys[4] = tmp_1758; values[4] = tmp_1759; }
            let tmp_1761 = smem_keys[tmp_1745 * WPT + 5u];
            let tmp_1762 = smem_vals[tmp_1745 * WPT + 5u];
            let tmp_1763 = keys[5] < tmp_1761 || (keys[5] == tmp_1761 && values[5] < tmp_1762);
            if tmp_1744 == tmp_1763 { keys[5] = tmp_1761; values[5] = tmp_1762; }
            let tmp_1764 = smem_keys[tmp_1745 * WPT + 6u];
            let tmp_1765 = smem_vals[tmp_1745 * WPT + 6u];
            let tmp_1766 = keys[6] < tmp_1764 || (keys[6] == tmp_1764 && values[6] < tmp_1765);
            if tmp_1744 == tmp_1766 { keys[6] = tmp_1764; values[6] = tmp_1765; }
            let tmp_1767 = smem_keys[tmp_1745 * WPT + 7u];
            let tmp_1768 = smem_vals[tmp_1745 * WPT + 7u];
            let tmp_1769 = keys[7] < tmp_1767 || (keys[7] == tmp_1767 && values[7] < tmp_1768);
            if tmp_1744 == tmp_1769 { keys[7] = tmp_1767; values[7] = tmp_1768; }
            let tmp_1770 = smem_keys[tmp_1745 * WPT + 8u];
            let tmp_1771 = smem_vals[tmp_1745 * WPT + 8u];
            let tmp_1772 = keys[8] < tmp_1770 || (keys[8] == tmp_1770 && values[8] < tmp_1771);
            if tmp_1744 == tmp_1772 { keys[8] = tmp_1770; values[8] = tmp_1771; }
            let tmp_1773 = smem_keys[tmp_1745 * WPT + 9u];
            let tmp_1774 = smem_vals[tmp_1745 * WPT + 9u];
            let tmp_1775 = keys[9] < tmp_1773 || (keys[9] == tmp_1773 && values[9] < tmp_1774);
            if tmp_1744 == tmp_1775 { keys[9] = tmp_1773; values[9] = tmp_1774; }
            let tmp_1776 = smem_keys[tmp_1745 * WPT + 10u];
            let tmp_1777 = smem_vals[tmp_1745 * WPT + 10u];
            let tmp_1778 = keys[10] < tmp_1776 || (keys[10] == tmp_1776 && values[10] < tmp_1777);
            if tmp_1744 == tmp_1778 { keys[10] = tmp_1776; values[10] = tmp_1777; }
            let tmp_1779 = smem_keys[tmp_1745 * WPT + 11u];
            let tmp_1780 = smem_vals[tmp_1745 * WPT + 11u];
            let tmp_1781 = keys[11] < tmp_1779 || (keys[11] == tmp_1779 && values[11] < tmp_1780);
            if tmp_1744 == tmp_1781 { keys[11] = tmp_1779; values[11] = tmp_1780; }
            let tmp_1782 = smem_keys[tmp_1745 * WPT + 12u];
            let tmp_1783 = smem_vals[tmp_1745 * WPT + 12u];
            let tmp_1784 = keys[12] < tmp_1782 || (keys[12] == tmp_1782 && values[12] < tmp_1783);
            if tmp_1744 == tmp_1784 { keys[12] = tmp_1782; values[12] = tmp_1783; }
            let tmp_1785 = smem_keys[tmp_1745 * WPT + 13u];
            let tmp_1786 = smem_vals[tmp_1745 * WPT + 13u];
            let tmp_1787 = keys[13] < tmp_1785 || (keys[13] == tmp_1785 && values[13] < tmp_1786);
            if tmp_1744 == tmp_1787 { keys[13] = tmp_1785; values[13] = tmp_1786; }
            let tmp_1788 = smem_keys[tmp_1745 * WPT + 14u];
            let tmp_1789 = smem_vals[tmp_1745 * WPT + 14u];
            let tmp_1790 = keys[14] < tmp_1788 || (keys[14] == tmp_1788 && values[14] < tmp_1789);
            if tmp_1744 == tmp_1790 { keys[14] = tmp_1788; values[14] = tmp_1789; }
            let tmp_1791 = smem_keys[tmp_1745 * WPT + 15u];
            let tmp_1792 = smem_vals[tmp_1745 * WPT + 15u];
            let tmp_1793 = keys[15] < tmp_1791 || (keys[15] == tmp_1791 && values[15] < tmp_1792);
            if tmp_1744 == tmp_1793 { keys[15] = tmp_1791; values[15] = tmp_1792; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],4,2)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1794 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_1795 = seg_base + (local_tid ^ 4u);
            let tmp_1796 = smem_keys[tmp_1795 * WPT + 0u];
            let tmp_1797 = smem_vals[tmp_1795 * WPT + 0u];
            let tmp_1798 = keys[0] < tmp_1796 || (keys[0] == tmp_1796 && values[0] < tmp_1797);
            if tmp_1794 == tmp_1798 { keys[0] = tmp_1796; values[0] = tmp_1797; }
            let tmp_1799 = smem_keys[tmp_1795 * WPT + 1u];
            let tmp_1800 = smem_vals[tmp_1795 * WPT + 1u];
            let tmp_1801 = keys[1] < tmp_1799 || (keys[1] == tmp_1799 && values[1] < tmp_1800);
            if tmp_1794 == tmp_1801 { keys[1] = tmp_1799; values[1] = tmp_1800; }
            let tmp_1802 = smem_keys[tmp_1795 * WPT + 2u];
            let tmp_1803 = smem_vals[tmp_1795 * WPT + 2u];
            let tmp_1804 = keys[2] < tmp_1802 || (keys[2] == tmp_1802 && values[2] < tmp_1803);
            if tmp_1794 == tmp_1804 { keys[2] = tmp_1802; values[2] = tmp_1803; }
            let tmp_1805 = smem_keys[tmp_1795 * WPT + 3u];
            let tmp_1806 = smem_vals[tmp_1795 * WPT + 3u];
            let tmp_1807 = keys[3] < tmp_1805 || (keys[3] == tmp_1805 && values[3] < tmp_1806);
            if tmp_1794 == tmp_1807 { keys[3] = tmp_1805; values[3] = tmp_1806; }
            let tmp_1808 = smem_keys[tmp_1795 * WPT + 4u];
            let tmp_1809 = smem_vals[tmp_1795 * WPT + 4u];
            let tmp_1810 = keys[4] < tmp_1808 || (keys[4] == tmp_1808 && values[4] < tmp_1809);
            if tmp_1794 == tmp_1810 { keys[4] = tmp_1808; values[4] = tmp_1809; }
            let tmp_1811 = smem_keys[tmp_1795 * WPT + 5u];
            let tmp_1812 = smem_vals[tmp_1795 * WPT + 5u];
            let tmp_1813 = keys[5] < tmp_1811 || (keys[5] == tmp_1811 && values[5] < tmp_1812);
            if tmp_1794 == tmp_1813 { keys[5] = tmp_1811; values[5] = tmp_1812; }
            let tmp_1814 = smem_keys[tmp_1795 * WPT + 6u];
            let tmp_1815 = smem_vals[tmp_1795 * WPT + 6u];
            let tmp_1816 = keys[6] < tmp_1814 || (keys[6] == tmp_1814 && values[6] < tmp_1815);
            if tmp_1794 == tmp_1816 { keys[6] = tmp_1814; values[6] = tmp_1815; }
            let tmp_1817 = smem_keys[tmp_1795 * WPT + 7u];
            let tmp_1818 = smem_vals[tmp_1795 * WPT + 7u];
            let tmp_1819 = keys[7] < tmp_1817 || (keys[7] == tmp_1817 && values[7] < tmp_1818);
            if tmp_1794 == tmp_1819 { keys[7] = tmp_1817; values[7] = tmp_1818; }
            let tmp_1820 = smem_keys[tmp_1795 * WPT + 8u];
            let tmp_1821 = smem_vals[tmp_1795 * WPT + 8u];
            let tmp_1822 = keys[8] < tmp_1820 || (keys[8] == tmp_1820 && values[8] < tmp_1821);
            if tmp_1794 == tmp_1822 { keys[8] = tmp_1820; values[8] = tmp_1821; }
            let tmp_1823 = smem_keys[tmp_1795 * WPT + 9u];
            let tmp_1824 = smem_vals[tmp_1795 * WPT + 9u];
            let tmp_1825 = keys[9] < tmp_1823 || (keys[9] == tmp_1823 && values[9] < tmp_1824);
            if tmp_1794 == tmp_1825 { keys[9] = tmp_1823; values[9] = tmp_1824; }
            let tmp_1826 = smem_keys[tmp_1795 * WPT + 10u];
            let tmp_1827 = smem_vals[tmp_1795 * WPT + 10u];
            let tmp_1828 = keys[10] < tmp_1826 || (keys[10] == tmp_1826 && values[10] < tmp_1827);
            if tmp_1794 == tmp_1828 { keys[10] = tmp_1826; values[10] = tmp_1827; }
            let tmp_1829 = smem_keys[tmp_1795 * WPT + 11u];
            let tmp_1830 = smem_vals[tmp_1795 * WPT + 11u];
            let tmp_1831 = keys[11] < tmp_1829 || (keys[11] == tmp_1829 && values[11] < tmp_1830);
            if tmp_1794 == tmp_1831 { keys[11] = tmp_1829; values[11] = tmp_1830; }
            let tmp_1832 = smem_keys[tmp_1795 * WPT + 12u];
            let tmp_1833 = smem_vals[tmp_1795 * WPT + 12u];
            let tmp_1834 = keys[12] < tmp_1832 || (keys[12] == tmp_1832 && values[12] < tmp_1833);
            if tmp_1794 == tmp_1834 { keys[12] = tmp_1832; values[12] = tmp_1833; }
            let tmp_1835 = smem_keys[tmp_1795 * WPT + 13u];
            let tmp_1836 = smem_vals[tmp_1795 * WPT + 13u];
            let tmp_1837 = keys[13] < tmp_1835 || (keys[13] == tmp_1835 && values[13] < tmp_1836);
            if tmp_1794 == tmp_1837 { keys[13] = tmp_1835; values[13] = tmp_1836; }
            let tmp_1838 = smem_keys[tmp_1795 * WPT + 14u];
            let tmp_1839 = smem_vals[tmp_1795 * WPT + 14u];
            let tmp_1840 = keys[14] < tmp_1838 || (keys[14] == tmp_1838 && values[14] < tmp_1839);
            if tmp_1794 == tmp_1840 { keys[14] = tmp_1838; values[14] = tmp_1839; }
            let tmp_1841 = smem_keys[tmp_1795 * WPT + 15u];
            let tmp_1842 = smem_vals[tmp_1795 * WPT + 15u];
            let tmp_1843 = keys[15] < tmp_1841 || (keys[15] == tmp_1841 && values[15] < tmp_1842);
            if tmp_1794 == tmp_1843 { keys[15] = tmp_1841; values[15] = tmp_1842; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],2,1)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1844 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_1845 = seg_base + (local_tid ^ 2u);
            let tmp_1846 = smem_keys[tmp_1845 * WPT + 0u];
            let tmp_1847 = smem_vals[tmp_1845 * WPT + 0u];
            let tmp_1848 = keys[0] < tmp_1846 || (keys[0] == tmp_1846 && values[0] < tmp_1847);
            if tmp_1844 == tmp_1848 { keys[0] = tmp_1846; values[0] = tmp_1847; }
            let tmp_1849 = smem_keys[tmp_1845 * WPT + 1u];
            let tmp_1850 = smem_vals[tmp_1845 * WPT + 1u];
            let tmp_1851 = keys[1] < tmp_1849 || (keys[1] == tmp_1849 && values[1] < tmp_1850);
            if tmp_1844 == tmp_1851 { keys[1] = tmp_1849; values[1] = tmp_1850; }
            let tmp_1852 = smem_keys[tmp_1845 * WPT + 2u];
            let tmp_1853 = smem_vals[tmp_1845 * WPT + 2u];
            let tmp_1854 = keys[2] < tmp_1852 || (keys[2] == tmp_1852 && values[2] < tmp_1853);
            if tmp_1844 == tmp_1854 { keys[2] = tmp_1852; values[2] = tmp_1853; }
            let tmp_1855 = smem_keys[tmp_1845 * WPT + 3u];
            let tmp_1856 = smem_vals[tmp_1845 * WPT + 3u];
            let tmp_1857 = keys[3] < tmp_1855 || (keys[3] == tmp_1855 && values[3] < tmp_1856);
            if tmp_1844 == tmp_1857 { keys[3] = tmp_1855; values[3] = tmp_1856; }
            let tmp_1858 = smem_keys[tmp_1845 * WPT + 4u];
            let tmp_1859 = smem_vals[tmp_1845 * WPT + 4u];
            let tmp_1860 = keys[4] < tmp_1858 || (keys[4] == tmp_1858 && values[4] < tmp_1859);
            if tmp_1844 == tmp_1860 { keys[4] = tmp_1858; values[4] = tmp_1859; }
            let tmp_1861 = smem_keys[tmp_1845 * WPT + 5u];
            let tmp_1862 = smem_vals[tmp_1845 * WPT + 5u];
            let tmp_1863 = keys[5] < tmp_1861 || (keys[5] == tmp_1861 && values[5] < tmp_1862);
            if tmp_1844 == tmp_1863 { keys[5] = tmp_1861; values[5] = tmp_1862; }
            let tmp_1864 = smem_keys[tmp_1845 * WPT + 6u];
            let tmp_1865 = smem_vals[tmp_1845 * WPT + 6u];
            let tmp_1866 = keys[6] < tmp_1864 || (keys[6] == tmp_1864 && values[6] < tmp_1865);
            if tmp_1844 == tmp_1866 { keys[6] = tmp_1864; values[6] = tmp_1865; }
            let tmp_1867 = smem_keys[tmp_1845 * WPT + 7u];
            let tmp_1868 = smem_vals[tmp_1845 * WPT + 7u];
            let tmp_1869 = keys[7] < tmp_1867 || (keys[7] == tmp_1867 && values[7] < tmp_1868);
            if tmp_1844 == tmp_1869 { keys[7] = tmp_1867; values[7] = tmp_1868; }
            let tmp_1870 = smem_keys[tmp_1845 * WPT + 8u];
            let tmp_1871 = smem_vals[tmp_1845 * WPT + 8u];
            let tmp_1872 = keys[8] < tmp_1870 || (keys[8] == tmp_1870 && values[8] < tmp_1871);
            if tmp_1844 == tmp_1872 { keys[8] = tmp_1870; values[8] = tmp_1871; }
            let tmp_1873 = smem_keys[tmp_1845 * WPT + 9u];
            let tmp_1874 = smem_vals[tmp_1845 * WPT + 9u];
            let tmp_1875 = keys[9] < tmp_1873 || (keys[9] == tmp_1873 && values[9] < tmp_1874);
            if tmp_1844 == tmp_1875 { keys[9] = tmp_1873; values[9] = tmp_1874; }
            let tmp_1876 = smem_keys[tmp_1845 * WPT + 10u];
            let tmp_1877 = smem_vals[tmp_1845 * WPT + 10u];
            let tmp_1878 = keys[10] < tmp_1876 || (keys[10] == tmp_1876 && values[10] < tmp_1877);
            if tmp_1844 == tmp_1878 { keys[10] = tmp_1876; values[10] = tmp_1877; }
            let tmp_1879 = smem_keys[tmp_1845 * WPT + 11u];
            let tmp_1880 = smem_vals[tmp_1845 * WPT + 11u];
            let tmp_1881 = keys[11] < tmp_1879 || (keys[11] == tmp_1879 && values[11] < tmp_1880);
            if tmp_1844 == tmp_1881 { keys[11] = tmp_1879; values[11] = tmp_1880; }
            let tmp_1882 = smem_keys[tmp_1845 * WPT + 12u];
            let tmp_1883 = smem_vals[tmp_1845 * WPT + 12u];
            let tmp_1884 = keys[12] < tmp_1882 || (keys[12] == tmp_1882 && values[12] < tmp_1883);
            if tmp_1844 == tmp_1884 { keys[12] = tmp_1882; values[12] = tmp_1883; }
            let tmp_1885 = smem_keys[tmp_1845 * WPT + 13u];
            let tmp_1886 = smem_vals[tmp_1845 * WPT + 13u];
            let tmp_1887 = keys[13] < tmp_1885 || (keys[13] == tmp_1885 && values[13] < tmp_1886);
            if tmp_1844 == tmp_1887 { keys[13] = tmp_1885; values[13] = tmp_1886; }
            let tmp_1888 = smem_keys[tmp_1845 * WPT + 14u];
            let tmp_1889 = smem_vals[tmp_1845 * WPT + 14u];
            let tmp_1890 = keys[14] < tmp_1888 || (keys[14] == tmp_1888 && values[14] < tmp_1889);
            if tmp_1844 == tmp_1890 { keys[14] = tmp_1888; values[14] = tmp_1889; }
            let tmp_1891 = smem_keys[tmp_1845 * WPT + 15u];
            let tmp_1892 = smem_vals[tmp_1845 * WPT + 15u];
            let tmp_1893 = keys[15] < tmp_1891 || (keys[15] == tmp_1891 && values[15] < tmp_1892);
            if tmp_1844 == tmp_1893 { keys[15] = tmp_1891; values[15] = tmp_1892; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:16)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)],1,0)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            smem_keys[tid_g * WPT + 4u] = keys[4];
            smem_vals[tid_g * WPT + 4u] = values[4];
            smem_keys[tid_g * WPT + 5u] = keys[5];
            smem_vals[tid_g * WPT + 5u] = values[5];
            smem_keys[tid_g * WPT + 6u] = keys[6];
            smem_vals[tid_g * WPT + 6u] = values[6];
            smem_keys[tid_g * WPT + 7u] = keys[7];
            smem_vals[tid_g * WPT + 7u] = values[7];
            smem_keys[tid_g * WPT + 8u] = keys[8];
            smem_vals[tid_g * WPT + 8u] = values[8];
            smem_keys[tid_g * WPT + 9u] = keys[9];
            smem_vals[tid_g * WPT + 9u] = values[9];
            smem_keys[tid_g * WPT + 10u] = keys[10];
            smem_vals[tid_g * WPT + 10u] = values[10];
            smem_keys[tid_g * WPT + 11u] = keys[11];
            smem_vals[tid_g * WPT + 11u] = values[11];
            smem_keys[tid_g * WPT + 12u] = keys[12];
            smem_vals[tid_g * WPT + 12u] = values[12];
            smem_keys[tid_g * WPT + 13u] = keys[13];
            smem_vals[tid_g * WPT + 13u] = values[13];
            smem_keys[tid_g * WPT + 14u] = keys[14];
            smem_vals[tid_g * WPT + 14u] = values[14];
            smem_keys[tid_g * WPT + 15u] = keys[15];
            smem_vals[tid_g * WPT + 15u] = values[15];
            workgroupBarrier();
            let tmp_1894 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_1895 = seg_base + (local_tid ^ 1u);
            let tmp_1896 = smem_keys[tmp_1895 * WPT + 0u];
            let tmp_1897 = smem_vals[tmp_1895 * WPT + 0u];
            let tmp_1898 = keys[0] < tmp_1896 || (keys[0] == tmp_1896 && values[0] < tmp_1897);
            if tmp_1894 == tmp_1898 { keys[0] = tmp_1896; values[0] = tmp_1897; }
            let tmp_1899 = smem_keys[tmp_1895 * WPT + 1u];
            let tmp_1900 = smem_vals[tmp_1895 * WPT + 1u];
            let tmp_1901 = keys[1] < tmp_1899 || (keys[1] == tmp_1899 && values[1] < tmp_1900);
            if tmp_1894 == tmp_1901 { keys[1] = tmp_1899; values[1] = tmp_1900; }
            let tmp_1902 = smem_keys[tmp_1895 * WPT + 2u];
            let tmp_1903 = smem_vals[tmp_1895 * WPT + 2u];
            let tmp_1904 = keys[2] < tmp_1902 || (keys[2] == tmp_1902 && values[2] < tmp_1903);
            if tmp_1894 == tmp_1904 { keys[2] = tmp_1902; values[2] = tmp_1903; }
            let tmp_1905 = smem_keys[tmp_1895 * WPT + 3u];
            let tmp_1906 = smem_vals[tmp_1895 * WPT + 3u];
            let tmp_1907 = keys[3] < tmp_1905 || (keys[3] == tmp_1905 && values[3] < tmp_1906);
            if tmp_1894 == tmp_1907 { keys[3] = tmp_1905; values[3] = tmp_1906; }
            let tmp_1908 = smem_keys[tmp_1895 * WPT + 4u];
            let tmp_1909 = smem_vals[tmp_1895 * WPT + 4u];
            let tmp_1910 = keys[4] < tmp_1908 || (keys[4] == tmp_1908 && values[4] < tmp_1909);
            if tmp_1894 == tmp_1910 { keys[4] = tmp_1908; values[4] = tmp_1909; }
            let tmp_1911 = smem_keys[tmp_1895 * WPT + 5u];
            let tmp_1912 = smem_vals[tmp_1895 * WPT + 5u];
            let tmp_1913 = keys[5] < tmp_1911 || (keys[5] == tmp_1911 && values[5] < tmp_1912);
            if tmp_1894 == tmp_1913 { keys[5] = tmp_1911; values[5] = tmp_1912; }
            let tmp_1914 = smem_keys[tmp_1895 * WPT + 6u];
            let tmp_1915 = smem_vals[tmp_1895 * WPT + 6u];
            let tmp_1916 = keys[6] < tmp_1914 || (keys[6] == tmp_1914 && values[6] < tmp_1915);
            if tmp_1894 == tmp_1916 { keys[6] = tmp_1914; values[6] = tmp_1915; }
            let tmp_1917 = smem_keys[tmp_1895 * WPT + 7u];
            let tmp_1918 = smem_vals[tmp_1895 * WPT + 7u];
            let tmp_1919 = keys[7] < tmp_1917 || (keys[7] == tmp_1917 && values[7] < tmp_1918);
            if tmp_1894 == tmp_1919 { keys[7] = tmp_1917; values[7] = tmp_1918; }
            let tmp_1920 = smem_keys[tmp_1895 * WPT + 8u];
            let tmp_1921 = smem_vals[tmp_1895 * WPT + 8u];
            let tmp_1922 = keys[8] < tmp_1920 || (keys[8] == tmp_1920 && values[8] < tmp_1921);
            if tmp_1894 == tmp_1922 { keys[8] = tmp_1920; values[8] = tmp_1921; }
            let tmp_1923 = smem_keys[tmp_1895 * WPT + 9u];
            let tmp_1924 = smem_vals[tmp_1895 * WPT + 9u];
            let tmp_1925 = keys[9] < tmp_1923 || (keys[9] == tmp_1923 && values[9] < tmp_1924);
            if tmp_1894 == tmp_1925 { keys[9] = tmp_1923; values[9] = tmp_1924; }
            let tmp_1926 = smem_keys[tmp_1895 * WPT + 10u];
            let tmp_1927 = smem_vals[tmp_1895 * WPT + 10u];
            let tmp_1928 = keys[10] < tmp_1926 || (keys[10] == tmp_1926 && values[10] < tmp_1927);
            if tmp_1894 == tmp_1928 { keys[10] = tmp_1926; values[10] = tmp_1927; }
            let tmp_1929 = smem_keys[tmp_1895 * WPT + 11u];
            let tmp_1930 = smem_vals[tmp_1895 * WPT + 11u];
            let tmp_1931 = keys[11] < tmp_1929 || (keys[11] == tmp_1929 && values[11] < tmp_1930);
            if tmp_1894 == tmp_1931 { keys[11] = tmp_1929; values[11] = tmp_1930; }
            let tmp_1932 = smem_keys[tmp_1895 * WPT + 12u];
            let tmp_1933 = smem_vals[tmp_1895 * WPT + 12u];
            let tmp_1934 = keys[12] < tmp_1932 || (keys[12] == tmp_1932 && values[12] < tmp_1933);
            if tmp_1894 == tmp_1934 { keys[12] = tmp_1932; values[12] = tmp_1933; }
            let tmp_1935 = smem_keys[tmp_1895 * WPT + 13u];
            let tmp_1936 = smem_vals[tmp_1895 * WPT + 13u];
            let tmp_1937 = keys[13] < tmp_1935 || (keys[13] == tmp_1935 && values[13] < tmp_1936);
            if tmp_1894 == tmp_1937 { keys[13] = tmp_1935; values[13] = tmp_1936; }
            let tmp_1938 = smem_keys[tmp_1895 * WPT + 14u];
            let tmp_1939 = smem_vals[tmp_1895 * WPT + 14u];
            let tmp_1940 = keys[14] < tmp_1938 || (keys[14] == tmp_1938 && values[14] < tmp_1939);
            if tmp_1894 == tmp_1940 { keys[14] = tmp_1938; values[14] = tmp_1939; }
            let tmp_1941 = smem_keys[tmp_1895 * WPT + 15u];
            let tmp_1942 = smem_vals[tmp_1895 * WPT + 15u];
            let tmp_1943 = keys[15] < tmp_1941 || (keys[15] == tmp_1941 && values[15] < tmp_1942);
            if tmp_1894 == tmp_1943 { keys[15] = tmp_1941; values[15] = tmp_1942; }
            workgroupBarrier();
        }
    }

    // exch_local(8,16)
    {
        // cmp_swap(0,8)
        if keys[0] > keys[8] || (keys[0] == keys[8] && values[0] > values[8]) {
            // swap(0,8)
            {
                let tmp_1944 = keys[0]; keys[0] = keys[8]; keys[8] = tmp_1944;
                let tmp_1945 = values[0]; values[0] = values[8]; values[8] = tmp_1945;
            }
        }
        // cmp_swap(1,9)
        if keys[1] > keys[9] || (keys[1] == keys[9] && values[1] > values[9]) {
            // swap(1,9)
            {
                let tmp_1946 = keys[1]; keys[1] = keys[9]; keys[9] = tmp_1946;
                let tmp_1947 = values[1]; values[1] = values[9]; values[9] = tmp_1947;
            }
        }
        // cmp_swap(2,10)
        if keys[2] > keys[10] || (keys[2] == keys[10] && values[2] > values[10]) {
            // swap(2,10)
            {
                let tmp_1948 = keys[2]; keys[2] = keys[10]; keys[10] = tmp_1948;
                let tmp_1949 = values[2]; values[2] = values[10]; values[10] = tmp_1949;
            }
        }
        // cmp_swap(3,11)
        if keys[3] > keys[11] || (keys[3] == keys[11] && values[3] > values[11]) {
            // swap(3,11)
            {
                let tmp_1950 = keys[3]; keys[3] = keys[11]; keys[11] = tmp_1950;
                let tmp_1951 = values[3]; values[3] = values[11]; values[11] = tmp_1951;
            }
        }
        // cmp_swap(4,12)
        if keys[4] > keys[12] || (keys[4] == keys[12] && values[4] > values[12]) {
            // swap(4,12)
            {
                let tmp_1952 = keys[4]; keys[4] = keys[12]; keys[12] = tmp_1952;
                let tmp_1953 = values[4]; values[4] = values[12]; values[12] = tmp_1953;
            }
        }
        // cmp_swap(5,13)
        if keys[5] > keys[13] || (keys[5] == keys[13] && values[5] > values[13]) {
            // swap(5,13)
            {
                let tmp_1954 = keys[5]; keys[5] = keys[13]; keys[13] = tmp_1954;
                let tmp_1955 = values[5]; values[5] = values[13]; values[13] = tmp_1955;
            }
        }
        // cmp_swap(6,14)
        if keys[6] > keys[14] || (keys[6] == keys[14] && values[6] > values[14]) {
            // swap(6,14)
            {
                let tmp_1956 = keys[6]; keys[6] = keys[14]; keys[14] = tmp_1956;
                let tmp_1957 = values[6]; values[6] = values[14]; values[14] = tmp_1957;
            }
        }
        // cmp_swap(7,15)
        if keys[7] > keys[15] || (keys[7] == keys[15] && values[7] > values[15]) {
            // swap(7,15)
            {
                let tmp_1958 = keys[7]; keys[7] = keys[15]; keys[15] = tmp_1958;
                let tmp_1959 = values[7]; values[7] = values[15]; values[15] = tmp_1959;
            }
        }
    }

    // exch_local(4,16)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_1960 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_1960;
                let tmp_1961 = values[0]; values[0] = values[4]; values[4] = tmp_1961;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_1962 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_1962;
                let tmp_1963 = values[1]; values[1] = values[5]; values[5] = tmp_1963;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_1964 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_1964;
                let tmp_1965 = values[2]; values[2] = values[6]; values[6] = tmp_1965;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_1966 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_1966;
                let tmp_1967 = values[3]; values[3] = values[7]; values[7] = tmp_1967;
            }
        }
        // cmp_swap(8,12)
        if keys[8] > keys[12] || (keys[8] == keys[12] && values[8] > values[12]) {
            // swap(8,12)
            {
                let tmp_1968 = keys[8]; keys[8] = keys[12]; keys[12] = tmp_1968;
                let tmp_1969 = values[8]; values[8] = values[12]; values[12] = tmp_1969;
            }
        }
        // cmp_swap(9,13)
        if keys[9] > keys[13] || (keys[9] == keys[13] && values[9] > values[13]) {
            // swap(9,13)
            {
                let tmp_1970 = keys[9]; keys[9] = keys[13]; keys[13] = tmp_1970;
                let tmp_1971 = values[9]; values[9] = values[13]; values[13] = tmp_1971;
            }
        }
        // cmp_swap(10,14)
        if keys[10] > keys[14] || (keys[10] == keys[14] && values[10] > values[14]) {
            // swap(10,14)
            {
                let tmp_1972 = keys[10]; keys[10] = keys[14]; keys[14] = tmp_1972;
                let tmp_1973 = values[10]; values[10] = values[14]; values[14] = tmp_1973;
            }
        }
        // cmp_swap(11,15)
        if keys[11] > keys[15] || (keys[11] == keys[15] && values[11] > values[15]) {
            // swap(11,15)
            {
                let tmp_1974 = keys[11]; keys[11] = keys[15]; keys[15] = tmp_1974;
                let tmp_1975 = values[11]; values[11] = values[15]; values[15] = tmp_1975;
            }
        }
    }

    // exch_local(2,16)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_1976 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_1976;
                let tmp_1977 = values[0]; values[0] = values[2]; values[2] = tmp_1977;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_1978 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_1978;
                let tmp_1979 = values[1]; values[1] = values[3]; values[3] = tmp_1979;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_1980 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_1980;
                let tmp_1981 = values[4]; values[4] = values[6]; values[6] = tmp_1981;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_1982 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_1982;
                let tmp_1983 = values[5]; values[5] = values[7]; values[7] = tmp_1983;
            }
        }
        // cmp_swap(8,10)
        if keys[8] > keys[10] || (keys[8] == keys[10] && values[8] > values[10]) {
            // swap(8,10)
            {
                let tmp_1984 = keys[8]; keys[8] = keys[10]; keys[10] = tmp_1984;
                let tmp_1985 = values[8]; values[8] = values[10]; values[10] = tmp_1985;
            }
        }
        // cmp_swap(9,11)
        if keys[9] > keys[11] || (keys[9] == keys[11] && values[9] > values[11]) {
            // swap(9,11)
            {
                let tmp_1986 = keys[9]; keys[9] = keys[11]; keys[11] = tmp_1986;
                let tmp_1987 = values[9]; values[9] = values[11]; values[11] = tmp_1987;
            }
        }
        // cmp_swap(12,14)
        if keys[12] > keys[14] || (keys[12] == keys[14] && values[12] > values[14]) {
            // swap(12,14)
            {
                let tmp_1988 = keys[12]; keys[12] = keys[14]; keys[14] = tmp_1988;
                let tmp_1989 = values[12]; values[12] = values[14]; values[14] = tmp_1989;
            }
        }
        // cmp_swap(13,15)
        if keys[13] > keys[15] || (keys[13] == keys[15] && values[13] > values[15]) {
            // swap(13,15)
            {
                let tmp_1990 = keys[13]; keys[13] = keys[15]; keys[15] = tmp_1990;
                let tmp_1991 = values[13]; values[13] = values[15]; values[15] = tmp_1991;
            }
        }
    }

    // exch_local(1,16)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_1992 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_1992;
                let tmp_1993 = values[0]; values[0] = values[1]; values[1] = tmp_1993;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_1994 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_1994;
                let tmp_1995 = values[2]; values[2] = values[3]; values[3] = tmp_1995;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_1996 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_1996;
                let tmp_1997 = values[4]; values[4] = values[5]; values[5] = tmp_1997;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_1998 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_1998;
                let tmp_1999 = values[6]; values[6] = values[7]; values[7] = tmp_1999;
            }
        }
        // cmp_swap(8,9)
        if keys[8] > keys[9] || (keys[8] == keys[9] && values[8] > values[9]) {
            // swap(8,9)
            {
                let tmp_2000 = keys[8]; keys[8] = keys[9]; keys[9] = tmp_2000;
                let tmp_2001 = values[8]; values[8] = values[9]; values[9] = tmp_2001;
            }
        }
        // cmp_swap(10,11)
        if keys[10] > keys[11] || (keys[10] == keys[11] && values[10] > values[11]) {
            // swap(10,11)
            {
                let tmp_2002 = keys[10]; keys[10] = keys[11]; keys[11] = tmp_2002;
                let tmp_2003 = values[10]; values[10] = values[11]; values[11] = tmp_2003;
            }
        }
        // cmp_swap(12,13)
        if keys[12] > keys[13] || (keys[12] == keys[13] && values[12] > values[13]) {
            // swap(12,13)
            {
                let tmp_2004 = keys[12]; keys[12] = keys[13]; keys[13] = tmp_2004;
                let tmp_2005 = values[12]; values[12] = values[13]; values[13] = tmp_2005;
            }
        }
        // cmp_swap(14,15)
        if keys[14] > keys[15] || (keys[14] == keys[15] && values[14] > values[15]) {
            // swap(14,15)
            {
                let tmp_2006 = keys[14]; keys[14] = keys[15]; keys[15] = tmp_2006;
                let tmp_2007 = values[14]; values[14] = values[15]; values[15] = tmp_2007;
            }
        }
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
