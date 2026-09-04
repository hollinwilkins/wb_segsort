
override WG: u32 = 4u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 32u;
const M: u32 = 4u;
const WPT: u32 = 8u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n32_m4_striped(
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

    var keys: array<u32, 8>;
    var values: array<u32, 8>;

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

    // exch_local(1,8)
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
    }

    // exch_local(3,8)
    {
        // cmp_swap(0,3)
        if keys[0] > keys[3] || (keys[0] == keys[3] && values[0] > values[3]) {
            // swap(0,3)
            {
                let tmp_8 = keys[0]; keys[0] = keys[3]; keys[3] = tmp_8;
                let tmp_9 = values[0]; values[0] = values[3]; values[3] = tmp_9;
            }
        }
        // cmp_swap(1,2)
        if keys[1] > keys[2] || (keys[1] == keys[2] && values[1] > values[2]) {
            // swap(1,2)
            {
                let tmp_10 = keys[1]; keys[1] = keys[2]; keys[2] = tmp_10;
                let tmp_11 = values[1]; values[1] = values[2]; values[2] = tmp_11;
            }
        }
        // cmp_swap(4,7)
        if keys[4] > keys[7] || (keys[4] == keys[7] && values[4] > values[7]) {
            // swap(4,7)
            {
                let tmp_12 = keys[4]; keys[4] = keys[7]; keys[7] = tmp_12;
                let tmp_13 = values[4]; values[4] = values[7]; values[7] = tmp_13;
            }
        }
        // cmp_swap(5,6)
        if keys[5] > keys[6] || (keys[5] == keys[6] && values[5] > values[6]) {
            // swap(5,6)
            {
                let tmp_14 = keys[5]; keys[5] = keys[6]; keys[6] = tmp_14;
                let tmp_15 = values[5]; values[5] = values[6]; values[6] = tmp_15;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_16 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_16;
                let tmp_17 = values[0]; values[0] = values[1]; values[1] = tmp_17;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_18 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_18;
                let tmp_19 = values[2]; values[2] = values[3]; values[3] = tmp_19;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_20 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_20;
                let tmp_21 = values[4]; values[4] = values[5]; values[5] = tmp_21;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_22 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_22;
                let tmp_23 = values[6]; values[6] = values[7]; values[7] = tmp_23;
            }
        }
    }

    // exch_local(7,8)
    {
        // cmp_swap(0,7)
        if keys[0] > keys[7] || (keys[0] == keys[7] && values[0] > values[7]) {
            // swap(0,7)
            {
                let tmp_24 = keys[0]; keys[0] = keys[7]; keys[7] = tmp_24;
                let tmp_25 = values[0]; values[0] = values[7]; values[7] = tmp_25;
            }
        }
        // cmp_swap(1,6)
        if keys[1] > keys[6] || (keys[1] == keys[6] && values[1] > values[6]) {
            // swap(1,6)
            {
                let tmp_26 = keys[1]; keys[1] = keys[6]; keys[6] = tmp_26;
                let tmp_27 = values[1]; values[1] = values[6]; values[6] = tmp_27;
            }
        }
        // cmp_swap(2,5)
        if keys[2] > keys[5] || (keys[2] == keys[5] && values[2] > values[5]) {
            // swap(2,5)
            {
                let tmp_28 = keys[2]; keys[2] = keys[5]; keys[5] = tmp_28;
                let tmp_29 = values[2]; values[2] = values[5]; values[5] = tmp_29;
            }
        }
        // cmp_swap(3,4)
        if keys[3] > keys[4] || (keys[3] == keys[4] && values[3] > values[4]) {
            // swap(3,4)
            {
                let tmp_30 = keys[3]; keys[3] = keys[4]; keys[4] = tmp_30;
                let tmp_31 = values[3]; values[3] = values[4]; values[4] = tmp_31;
            }
        }
    }

    // exch_local(2,8)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_32 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_32;
                let tmp_33 = values[0]; values[0] = values[2]; values[2] = tmp_33;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_34 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_34;
                let tmp_35 = values[1]; values[1] = values[3]; values[3] = tmp_35;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_36 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_36;
                let tmp_37 = values[4]; values[4] = values[6]; values[6] = tmp_37;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_38 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_38;
                let tmp_39 = values[5]; values[5] = values[7]; values[7] = tmp_39;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_40 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_40;
                let tmp_41 = values[0]; values[0] = values[1]; values[1] = tmp_41;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_42 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_42;
                let tmp_43 = values[2]; values[2] = values[3]; values[3] = tmp_43;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_44 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_44;
                let tmp_45 = values[4]; values[4] = values[5]; values[5] = tmp_45;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_46 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_46;
                let tmp_47 = values[6]; values[6] = values[7]; values[7] = tmp_47;
            }
        }
    }

    // exch_intxn(tmask:1,swbit:0,wpt:8)
    {
        // _exch_workgroup([(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)],1,0)
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
            workgroupBarrier();
            let tmp_48 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_49 = seg_base + (local_tid ^ 1u);
            let tmp_50 = smem_keys[tmp_49 * WPT + 7u];
            let tmp_51 = smem_vals[tmp_49 * WPT + 7u];
            let tmp_52 = keys[0] < tmp_50 || (keys[0] == tmp_50 && values[0] < tmp_51);
            if tmp_48 == tmp_52 { keys[0] = tmp_50; values[0] = tmp_51; }
            let tmp_53 = smem_keys[tmp_49 * WPT + 6u];
            let tmp_54 = smem_vals[tmp_49 * WPT + 6u];
            let tmp_55 = keys[1] < tmp_53 || (keys[1] == tmp_53 && values[1] < tmp_54);
            if tmp_48 == tmp_55 { keys[1] = tmp_53; values[1] = tmp_54; }
            let tmp_56 = smem_keys[tmp_49 * WPT + 5u];
            let tmp_57 = smem_vals[tmp_49 * WPT + 5u];
            let tmp_58 = keys[2] < tmp_56 || (keys[2] == tmp_56 && values[2] < tmp_57);
            if tmp_48 == tmp_58 { keys[2] = tmp_56; values[2] = tmp_57; }
            let tmp_59 = smem_keys[tmp_49 * WPT + 4u];
            let tmp_60 = smem_vals[tmp_49 * WPT + 4u];
            let tmp_61 = keys[3] < tmp_59 || (keys[3] == tmp_59 && values[3] < tmp_60);
            if tmp_48 == tmp_61 { keys[3] = tmp_59; values[3] = tmp_60; }
            let tmp_62 = smem_keys[tmp_49 * WPT + 3u];
            let tmp_63 = smem_vals[tmp_49 * WPT + 3u];
            let tmp_64 = keys[4] < tmp_62 || (keys[4] == tmp_62 && values[4] < tmp_63);
            if tmp_48 == tmp_64 { keys[4] = tmp_62; values[4] = tmp_63; }
            let tmp_65 = smem_keys[tmp_49 * WPT + 2u];
            let tmp_66 = smem_vals[tmp_49 * WPT + 2u];
            let tmp_67 = keys[5] < tmp_65 || (keys[5] == tmp_65 && values[5] < tmp_66);
            if tmp_48 == tmp_67 { keys[5] = tmp_65; values[5] = tmp_66; }
            let tmp_68 = smem_keys[tmp_49 * WPT + 1u];
            let tmp_69 = smem_vals[tmp_49 * WPT + 1u];
            let tmp_70 = keys[6] < tmp_68 || (keys[6] == tmp_68 && values[6] < tmp_69);
            if tmp_48 == tmp_70 { keys[6] = tmp_68; values[6] = tmp_69; }
            let tmp_71 = smem_keys[tmp_49 * WPT + 0u];
            let tmp_72 = smem_vals[tmp_49 * WPT + 0u];
            let tmp_73 = keys[7] < tmp_71 || (keys[7] == tmp_71 && values[7] < tmp_72);
            if tmp_48 == tmp_73 { keys[7] = tmp_71; values[7] = tmp_72; }
            workgroupBarrier();
        }
    }

    // exch_local(4,8)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_74 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_74;
                let tmp_75 = values[0]; values[0] = values[4]; values[4] = tmp_75;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_76 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_76;
                let tmp_77 = values[1]; values[1] = values[5]; values[5] = tmp_77;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_78 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_78;
                let tmp_79 = values[2]; values[2] = values[6]; values[6] = tmp_79;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_80 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_80;
                let tmp_81 = values[3]; values[3] = values[7]; values[7] = tmp_81;
            }
        }
    }

    // exch_local(2,8)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_82 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_82;
                let tmp_83 = values[0]; values[0] = values[2]; values[2] = tmp_83;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_84 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_84;
                let tmp_85 = values[1]; values[1] = values[3]; values[3] = tmp_85;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_86 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_86;
                let tmp_87 = values[4]; values[4] = values[6]; values[6] = tmp_87;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_88 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_88;
                let tmp_89 = values[5]; values[5] = values[7]; values[7] = tmp_89;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_90 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_90;
                let tmp_91 = values[0]; values[0] = values[1]; values[1] = tmp_91;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_92 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_92;
                let tmp_93 = values[2]; values[2] = values[3]; values[3] = tmp_93;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_94 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_94;
                let tmp_95 = values[4]; values[4] = values[5]; values[5] = tmp_95;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_96 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_96;
                let tmp_97 = values[6]; values[6] = values[7]; values[7] = tmp_97;
            }
        }
    }

    // exch_intxn(tmask:3,swbit:1,wpt:8)
    {
        // _exch_workgroup([(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)],3,1)
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
            workgroupBarrier();
            let tmp_98 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_99 = seg_base + (local_tid ^ 3u);
            let tmp_100 = smem_keys[tmp_99 * WPT + 7u];
            let tmp_101 = smem_vals[tmp_99 * WPT + 7u];
            let tmp_102 = keys[0] < tmp_100 || (keys[0] == tmp_100 && values[0] < tmp_101);
            if tmp_98 == tmp_102 { keys[0] = tmp_100; values[0] = tmp_101; }
            let tmp_103 = smem_keys[tmp_99 * WPT + 6u];
            let tmp_104 = smem_vals[tmp_99 * WPT + 6u];
            let tmp_105 = keys[1] < tmp_103 || (keys[1] == tmp_103 && values[1] < tmp_104);
            if tmp_98 == tmp_105 { keys[1] = tmp_103; values[1] = tmp_104; }
            let tmp_106 = smem_keys[tmp_99 * WPT + 5u];
            let tmp_107 = smem_vals[tmp_99 * WPT + 5u];
            let tmp_108 = keys[2] < tmp_106 || (keys[2] == tmp_106 && values[2] < tmp_107);
            if tmp_98 == tmp_108 { keys[2] = tmp_106; values[2] = tmp_107; }
            let tmp_109 = smem_keys[tmp_99 * WPT + 4u];
            let tmp_110 = smem_vals[tmp_99 * WPT + 4u];
            let tmp_111 = keys[3] < tmp_109 || (keys[3] == tmp_109 && values[3] < tmp_110);
            if tmp_98 == tmp_111 { keys[3] = tmp_109; values[3] = tmp_110; }
            let tmp_112 = smem_keys[tmp_99 * WPT + 3u];
            let tmp_113 = smem_vals[tmp_99 * WPT + 3u];
            let tmp_114 = keys[4] < tmp_112 || (keys[4] == tmp_112 && values[4] < tmp_113);
            if tmp_98 == tmp_114 { keys[4] = tmp_112; values[4] = tmp_113; }
            let tmp_115 = smem_keys[tmp_99 * WPT + 2u];
            let tmp_116 = smem_vals[tmp_99 * WPT + 2u];
            let tmp_117 = keys[5] < tmp_115 || (keys[5] == tmp_115 && values[5] < tmp_116);
            if tmp_98 == tmp_117 { keys[5] = tmp_115; values[5] = tmp_116; }
            let tmp_118 = smem_keys[tmp_99 * WPT + 1u];
            let tmp_119 = smem_vals[tmp_99 * WPT + 1u];
            let tmp_120 = keys[6] < tmp_118 || (keys[6] == tmp_118 && values[6] < tmp_119);
            if tmp_98 == tmp_120 { keys[6] = tmp_118; values[6] = tmp_119; }
            let tmp_121 = smem_keys[tmp_99 * WPT + 0u];
            let tmp_122 = smem_vals[tmp_99 * WPT + 0u];
            let tmp_123 = keys[7] < tmp_121 || (keys[7] == tmp_121 && values[7] < tmp_122);
            if tmp_98 == tmp_123 { keys[7] = tmp_121; values[7] = tmp_122; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:8)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],1,0)
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
            workgroupBarrier();
            let tmp_124 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_125 = seg_base + (local_tid ^ 1u);
            let tmp_126 = smem_keys[tmp_125 * WPT + 0u];
            let tmp_127 = smem_vals[tmp_125 * WPT + 0u];
            let tmp_128 = keys[0] < tmp_126 || (keys[0] == tmp_126 && values[0] < tmp_127);
            if tmp_124 == tmp_128 { keys[0] = tmp_126; values[0] = tmp_127; }
            let tmp_129 = smem_keys[tmp_125 * WPT + 1u];
            let tmp_130 = smem_vals[tmp_125 * WPT + 1u];
            let tmp_131 = keys[1] < tmp_129 || (keys[1] == tmp_129 && values[1] < tmp_130);
            if tmp_124 == tmp_131 { keys[1] = tmp_129; values[1] = tmp_130; }
            let tmp_132 = smem_keys[tmp_125 * WPT + 2u];
            let tmp_133 = smem_vals[tmp_125 * WPT + 2u];
            let tmp_134 = keys[2] < tmp_132 || (keys[2] == tmp_132 && values[2] < tmp_133);
            if tmp_124 == tmp_134 { keys[2] = tmp_132; values[2] = tmp_133; }
            let tmp_135 = smem_keys[tmp_125 * WPT + 3u];
            let tmp_136 = smem_vals[tmp_125 * WPT + 3u];
            let tmp_137 = keys[3] < tmp_135 || (keys[3] == tmp_135 && values[3] < tmp_136);
            if tmp_124 == tmp_137 { keys[3] = tmp_135; values[3] = tmp_136; }
            let tmp_138 = smem_keys[tmp_125 * WPT + 4u];
            let tmp_139 = smem_vals[tmp_125 * WPT + 4u];
            let tmp_140 = keys[4] < tmp_138 || (keys[4] == tmp_138 && values[4] < tmp_139);
            if tmp_124 == tmp_140 { keys[4] = tmp_138; values[4] = tmp_139; }
            let tmp_141 = smem_keys[tmp_125 * WPT + 5u];
            let tmp_142 = smem_vals[tmp_125 * WPT + 5u];
            let tmp_143 = keys[5] < tmp_141 || (keys[5] == tmp_141 && values[5] < tmp_142);
            if tmp_124 == tmp_143 { keys[5] = tmp_141; values[5] = tmp_142; }
            let tmp_144 = smem_keys[tmp_125 * WPT + 6u];
            let tmp_145 = smem_vals[tmp_125 * WPT + 6u];
            let tmp_146 = keys[6] < tmp_144 || (keys[6] == tmp_144 && values[6] < tmp_145);
            if tmp_124 == tmp_146 { keys[6] = tmp_144; values[6] = tmp_145; }
            let tmp_147 = smem_keys[tmp_125 * WPT + 7u];
            let tmp_148 = smem_vals[tmp_125 * WPT + 7u];
            let tmp_149 = keys[7] < tmp_147 || (keys[7] == tmp_147 && values[7] < tmp_148);
            if tmp_124 == tmp_149 { keys[7] = tmp_147; values[7] = tmp_148; }
            workgroupBarrier();
        }
    }

    // exch_local(4,8)
    {
        // cmp_swap(0,4)
        if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
            // swap(0,4)
            {
                let tmp_150 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_150;
                let tmp_151 = values[0]; values[0] = values[4]; values[4] = tmp_151;
            }
        }
        // cmp_swap(1,5)
        if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
            // swap(1,5)
            {
                let tmp_152 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_152;
                let tmp_153 = values[1]; values[1] = values[5]; values[5] = tmp_153;
            }
        }
        // cmp_swap(2,6)
        if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
            // swap(2,6)
            {
                let tmp_154 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_154;
                let tmp_155 = values[2]; values[2] = values[6]; values[6] = tmp_155;
            }
        }
        // cmp_swap(3,7)
        if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
            // swap(3,7)
            {
                let tmp_156 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_156;
                let tmp_157 = values[3]; values[3] = values[7]; values[7] = tmp_157;
            }
        }
    }

    // exch_local(2,8)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_158 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_158;
                let tmp_159 = values[0]; values[0] = values[2]; values[2] = tmp_159;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_160 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_160;
                let tmp_161 = values[1]; values[1] = values[3]; values[3] = tmp_161;
            }
        }
        // cmp_swap(4,6)
        if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
            // swap(4,6)
            {
                let tmp_162 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_162;
                let tmp_163 = values[4]; values[4] = values[6]; values[6] = tmp_163;
            }
        }
        // cmp_swap(5,7)
        if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
            // swap(5,7)
            {
                let tmp_164 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_164;
                let tmp_165 = values[5]; values[5] = values[7]; values[7] = tmp_165;
            }
        }
    }

    // exch_local(1,8)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_166 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_166;
                let tmp_167 = values[0]; values[0] = values[1]; values[1] = tmp_167;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_168 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_168;
                let tmp_169 = values[2]; values[2] = values[3]; values[3] = tmp_169;
            }
        }
        // cmp_swap(4,5)
        if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
            // swap(4,5)
            {
                let tmp_170 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_170;
                let tmp_171 = values[4]; values[4] = values[5]; values[5] = tmp_171;
            }
        }
        // cmp_swap(6,7)
        if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
            // swap(6,7)
            {
                let tmp_172 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_172;
                let tmp_173 = values[6]; values[6] = values[7]; values[7] = tmp_173;
            }
        }
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
