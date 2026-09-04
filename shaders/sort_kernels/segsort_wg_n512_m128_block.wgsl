
override WG: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 512u;
const M: u32 = 128u;
const WPT: u32 = 4u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n512_m128_block(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 9u;

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

    var keys: array<u32, 4>;
    var values: array<u32, 4>;

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

    // exch_local(1,4)
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
    }

    // exch_local(3,4)
    {
        // cmp_swap(0,3)
        if keys[0] > keys[3] || (keys[0] == keys[3] && values[0] > values[3]) {
            // swap(0,3)
            {
                let tmp_4 = keys[0]; keys[0] = keys[3]; keys[3] = tmp_4;
                let tmp_5 = values[0]; values[0] = values[3]; values[3] = tmp_5;
            }
        }
        // cmp_swap(1,2)
        if keys[1] > keys[2] || (keys[1] == keys[2] && values[1] > values[2]) {
            // swap(1,2)
            {
                let tmp_6 = keys[1]; keys[1] = keys[2]; keys[2] = tmp_6;
                let tmp_7 = values[1]; values[1] = values[2]; values[2] = tmp_7;
            }
        }
    }

    // exch_local(1,4)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_8 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_8;
                let tmp_9 = values[0]; values[0] = values[1]; values[1] = tmp_9;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_10 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_10;
                let tmp_11 = values[2]; values[2] = values[3]; values[3] = tmp_11;
            }
        }
    }

    // exch_intxn(tmask:1,swbit:0,wpt:4)
    {
        // _exch_workgroup([(0, 3), (1, 2), (2, 1), (3, 0)],1,0)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_12 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_13 = seg_base + (local_tid ^ 1u);
            let tmp_14 = smem_keys[tmp_13 * WPT + 3u];
            let tmp_15 = smem_vals[tmp_13 * WPT + 3u];
            let tmp_16 = keys[0] < tmp_14 || (keys[0] == tmp_14 && values[0] < tmp_15);
            if tmp_12 == tmp_16 { keys[0] = tmp_14; values[0] = tmp_15; }
            let tmp_17 = smem_keys[tmp_13 * WPT + 2u];
            let tmp_18 = smem_vals[tmp_13 * WPT + 2u];
            let tmp_19 = keys[1] < tmp_17 || (keys[1] == tmp_17 && values[1] < tmp_18);
            if tmp_12 == tmp_19 { keys[1] = tmp_17; values[1] = tmp_18; }
            let tmp_20 = smem_keys[tmp_13 * WPT + 1u];
            let tmp_21 = smem_vals[tmp_13 * WPT + 1u];
            let tmp_22 = keys[2] < tmp_20 || (keys[2] == tmp_20 && values[2] < tmp_21);
            if tmp_12 == tmp_22 { keys[2] = tmp_20; values[2] = tmp_21; }
            let tmp_23 = smem_keys[tmp_13 * WPT + 0u];
            let tmp_24 = smem_vals[tmp_13 * WPT + 0u];
            let tmp_25 = keys[3] < tmp_23 || (keys[3] == tmp_23 && values[3] < tmp_24);
            if tmp_12 == tmp_25 { keys[3] = tmp_23; values[3] = tmp_24; }
            workgroupBarrier();
        }
    }

    // exch_local(2,4)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_26 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_26;
                let tmp_27 = values[0]; values[0] = values[2]; values[2] = tmp_27;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_28 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_28;
                let tmp_29 = values[1]; values[1] = values[3]; values[3] = tmp_29;
            }
        }
    }

    // exch_local(1,4)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_30 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_30;
                let tmp_31 = values[0]; values[0] = values[1]; values[1] = tmp_31;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_32 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_32;
                let tmp_33 = values[2]; values[2] = values[3]; values[3] = tmp_33;
            }
        }
    }

    // exch_intxn(tmask:3,swbit:1,wpt:4)
    {
        // _exch_workgroup([(0, 3), (1, 2), (2, 1), (3, 0)],3,1)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_34 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_35 = seg_base + (local_tid ^ 3u);
            let tmp_36 = smem_keys[tmp_35 * WPT + 3u];
            let tmp_37 = smem_vals[tmp_35 * WPT + 3u];
            let tmp_38 = keys[0] < tmp_36 || (keys[0] == tmp_36 && values[0] < tmp_37);
            if tmp_34 == tmp_38 { keys[0] = tmp_36; values[0] = tmp_37; }
            let tmp_39 = smem_keys[tmp_35 * WPT + 2u];
            let tmp_40 = smem_vals[tmp_35 * WPT + 2u];
            let tmp_41 = keys[1] < tmp_39 || (keys[1] == tmp_39 && values[1] < tmp_40);
            if tmp_34 == tmp_41 { keys[1] = tmp_39; values[1] = tmp_40; }
            let tmp_42 = smem_keys[tmp_35 * WPT + 1u];
            let tmp_43 = smem_vals[tmp_35 * WPT + 1u];
            let tmp_44 = keys[2] < tmp_42 || (keys[2] == tmp_42 && values[2] < tmp_43);
            if tmp_34 == tmp_44 { keys[2] = tmp_42; values[2] = tmp_43; }
            let tmp_45 = smem_keys[tmp_35 * WPT + 0u];
            let tmp_46 = smem_vals[tmp_35 * WPT + 0u];
            let tmp_47 = keys[3] < tmp_45 || (keys[3] == tmp_45 && values[3] < tmp_46);
            if tmp_34 == tmp_47 { keys[3] = tmp_45; values[3] = tmp_46; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],1,0)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_48 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_49 = seg_base + (local_tid ^ 1u);
            let tmp_50 = smem_keys[tmp_49 * WPT + 0u];
            let tmp_51 = smem_vals[tmp_49 * WPT + 0u];
            let tmp_52 = keys[0] < tmp_50 || (keys[0] == tmp_50 && values[0] < tmp_51);
            if tmp_48 == tmp_52 { keys[0] = tmp_50; values[0] = tmp_51; }
            let tmp_53 = smem_keys[tmp_49 * WPT + 1u];
            let tmp_54 = smem_vals[tmp_49 * WPT + 1u];
            let tmp_55 = keys[1] < tmp_53 || (keys[1] == tmp_53 && values[1] < tmp_54);
            if tmp_48 == tmp_55 { keys[1] = tmp_53; values[1] = tmp_54; }
            let tmp_56 = smem_keys[tmp_49 * WPT + 2u];
            let tmp_57 = smem_vals[tmp_49 * WPT + 2u];
            let tmp_58 = keys[2] < tmp_56 || (keys[2] == tmp_56 && values[2] < tmp_57);
            if tmp_48 == tmp_58 { keys[2] = tmp_56; values[2] = tmp_57; }
            let tmp_59 = smem_keys[tmp_49 * WPT + 3u];
            let tmp_60 = smem_vals[tmp_49 * WPT + 3u];
            let tmp_61 = keys[3] < tmp_59 || (keys[3] == tmp_59 && values[3] < tmp_60);
            if tmp_48 == tmp_61 { keys[3] = tmp_59; values[3] = tmp_60; }
            workgroupBarrier();
        }
    }

    // exch_local(2,4)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_62 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_62;
                let tmp_63 = values[0]; values[0] = values[2]; values[2] = tmp_63;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_64 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_64;
                let tmp_65 = values[1]; values[1] = values[3]; values[3] = tmp_65;
            }
        }
    }

    // exch_local(1,4)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_66 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_66;
                let tmp_67 = values[0]; values[0] = values[1]; values[1] = tmp_67;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_68 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_68;
                let tmp_69 = values[2]; values[2] = values[3]; values[3] = tmp_69;
            }
        }
    }

    // exch_intxn(tmask:7,swbit:2,wpt:4)
    {
        // _exch_workgroup([(0, 3), (1, 2), (2, 1), (3, 0)],7,2)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_70 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_71 = seg_base + (local_tid ^ 7u);
            let tmp_72 = smem_keys[tmp_71 * WPT + 3u];
            let tmp_73 = smem_vals[tmp_71 * WPT + 3u];
            let tmp_74 = keys[0] < tmp_72 || (keys[0] == tmp_72 && values[0] < tmp_73);
            if tmp_70 == tmp_74 { keys[0] = tmp_72; values[0] = tmp_73; }
            let tmp_75 = smem_keys[tmp_71 * WPT + 2u];
            let tmp_76 = smem_vals[tmp_71 * WPT + 2u];
            let tmp_77 = keys[1] < tmp_75 || (keys[1] == tmp_75 && values[1] < tmp_76);
            if tmp_70 == tmp_77 { keys[1] = tmp_75; values[1] = tmp_76; }
            let tmp_78 = smem_keys[tmp_71 * WPT + 1u];
            let tmp_79 = smem_vals[tmp_71 * WPT + 1u];
            let tmp_80 = keys[2] < tmp_78 || (keys[2] == tmp_78 && values[2] < tmp_79);
            if tmp_70 == tmp_80 { keys[2] = tmp_78; values[2] = tmp_79; }
            let tmp_81 = smem_keys[tmp_71 * WPT + 0u];
            let tmp_82 = smem_vals[tmp_71 * WPT + 0u];
            let tmp_83 = keys[3] < tmp_81 || (keys[3] == tmp_81 && values[3] < tmp_82);
            if tmp_70 == tmp_83 { keys[3] = tmp_81; values[3] = tmp_82; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],2,1)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_84 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_85 = seg_base + (local_tid ^ 2u);
            let tmp_86 = smem_keys[tmp_85 * WPT + 0u];
            let tmp_87 = smem_vals[tmp_85 * WPT + 0u];
            let tmp_88 = keys[0] < tmp_86 || (keys[0] == tmp_86 && values[0] < tmp_87);
            if tmp_84 == tmp_88 { keys[0] = tmp_86; values[0] = tmp_87; }
            let tmp_89 = smem_keys[tmp_85 * WPT + 1u];
            let tmp_90 = smem_vals[tmp_85 * WPT + 1u];
            let tmp_91 = keys[1] < tmp_89 || (keys[1] == tmp_89 && values[1] < tmp_90);
            if tmp_84 == tmp_91 { keys[1] = tmp_89; values[1] = tmp_90; }
            let tmp_92 = smem_keys[tmp_85 * WPT + 2u];
            let tmp_93 = smem_vals[tmp_85 * WPT + 2u];
            let tmp_94 = keys[2] < tmp_92 || (keys[2] == tmp_92 && values[2] < tmp_93);
            if tmp_84 == tmp_94 { keys[2] = tmp_92; values[2] = tmp_93; }
            let tmp_95 = smem_keys[tmp_85 * WPT + 3u];
            let tmp_96 = smem_vals[tmp_85 * WPT + 3u];
            let tmp_97 = keys[3] < tmp_95 || (keys[3] == tmp_95 && values[3] < tmp_96);
            if tmp_84 == tmp_97 { keys[3] = tmp_95; values[3] = tmp_96; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],1,0)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_98 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_99 = seg_base + (local_tid ^ 1u);
            let tmp_100 = smem_keys[tmp_99 * WPT + 0u];
            let tmp_101 = smem_vals[tmp_99 * WPT + 0u];
            let tmp_102 = keys[0] < tmp_100 || (keys[0] == tmp_100 && values[0] < tmp_101);
            if tmp_98 == tmp_102 { keys[0] = tmp_100; values[0] = tmp_101; }
            let tmp_103 = smem_keys[tmp_99 * WPT + 1u];
            let tmp_104 = smem_vals[tmp_99 * WPT + 1u];
            let tmp_105 = keys[1] < tmp_103 || (keys[1] == tmp_103 && values[1] < tmp_104);
            if tmp_98 == tmp_105 { keys[1] = tmp_103; values[1] = tmp_104; }
            let tmp_106 = smem_keys[tmp_99 * WPT + 2u];
            let tmp_107 = smem_vals[tmp_99 * WPT + 2u];
            let tmp_108 = keys[2] < tmp_106 || (keys[2] == tmp_106 && values[2] < tmp_107);
            if tmp_98 == tmp_108 { keys[2] = tmp_106; values[2] = tmp_107; }
            let tmp_109 = smem_keys[tmp_99 * WPT + 3u];
            let tmp_110 = smem_vals[tmp_99 * WPT + 3u];
            let tmp_111 = keys[3] < tmp_109 || (keys[3] == tmp_109 && values[3] < tmp_110);
            if tmp_98 == tmp_111 { keys[3] = tmp_109; values[3] = tmp_110; }
            workgroupBarrier();
        }
    }

    // exch_local(2,4)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_112 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_112;
                let tmp_113 = values[0]; values[0] = values[2]; values[2] = tmp_113;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_114 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_114;
                let tmp_115 = values[1]; values[1] = values[3]; values[3] = tmp_115;
            }
        }
    }

    // exch_local(1,4)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_116 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_116;
                let tmp_117 = values[0]; values[0] = values[1]; values[1] = tmp_117;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_118 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_118;
                let tmp_119 = values[2]; values[2] = values[3]; values[3] = tmp_119;
            }
        }
    }

    // exch_intxn(tmask:15,swbit:3,wpt:4)
    {
        // _exch_workgroup([(0, 3), (1, 2), (2, 1), (3, 0)],15,3)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_120 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_121 = seg_base + (local_tid ^ 15u);
            let tmp_122 = smem_keys[tmp_121 * WPT + 3u];
            let tmp_123 = smem_vals[tmp_121 * WPT + 3u];
            let tmp_124 = keys[0] < tmp_122 || (keys[0] == tmp_122 && values[0] < tmp_123);
            if tmp_120 == tmp_124 { keys[0] = tmp_122; values[0] = tmp_123; }
            let tmp_125 = smem_keys[tmp_121 * WPT + 2u];
            let tmp_126 = smem_vals[tmp_121 * WPT + 2u];
            let tmp_127 = keys[1] < tmp_125 || (keys[1] == tmp_125 && values[1] < tmp_126);
            if tmp_120 == tmp_127 { keys[1] = tmp_125; values[1] = tmp_126; }
            let tmp_128 = smem_keys[tmp_121 * WPT + 1u];
            let tmp_129 = smem_vals[tmp_121 * WPT + 1u];
            let tmp_130 = keys[2] < tmp_128 || (keys[2] == tmp_128 && values[2] < tmp_129);
            if tmp_120 == tmp_130 { keys[2] = tmp_128; values[2] = tmp_129; }
            let tmp_131 = smem_keys[tmp_121 * WPT + 0u];
            let tmp_132 = smem_vals[tmp_121 * WPT + 0u];
            let tmp_133 = keys[3] < tmp_131 || (keys[3] == tmp_131 && values[3] < tmp_132);
            if tmp_120 == tmp_133 { keys[3] = tmp_131; values[3] = tmp_132; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],4,2)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_134 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_135 = seg_base + (local_tid ^ 4u);
            let tmp_136 = smem_keys[tmp_135 * WPT + 0u];
            let tmp_137 = smem_vals[tmp_135 * WPT + 0u];
            let tmp_138 = keys[0] < tmp_136 || (keys[0] == tmp_136 && values[0] < tmp_137);
            if tmp_134 == tmp_138 { keys[0] = tmp_136; values[0] = tmp_137; }
            let tmp_139 = smem_keys[tmp_135 * WPT + 1u];
            let tmp_140 = smem_vals[tmp_135 * WPT + 1u];
            let tmp_141 = keys[1] < tmp_139 || (keys[1] == tmp_139 && values[1] < tmp_140);
            if tmp_134 == tmp_141 { keys[1] = tmp_139; values[1] = tmp_140; }
            let tmp_142 = smem_keys[tmp_135 * WPT + 2u];
            let tmp_143 = smem_vals[tmp_135 * WPT + 2u];
            let tmp_144 = keys[2] < tmp_142 || (keys[2] == tmp_142 && values[2] < tmp_143);
            if tmp_134 == tmp_144 { keys[2] = tmp_142; values[2] = tmp_143; }
            let tmp_145 = smem_keys[tmp_135 * WPT + 3u];
            let tmp_146 = smem_vals[tmp_135 * WPT + 3u];
            let tmp_147 = keys[3] < tmp_145 || (keys[3] == tmp_145 && values[3] < tmp_146);
            if tmp_134 == tmp_147 { keys[3] = tmp_145; values[3] = tmp_146; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],2,1)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_148 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_149 = seg_base + (local_tid ^ 2u);
            let tmp_150 = smem_keys[tmp_149 * WPT + 0u];
            let tmp_151 = smem_vals[tmp_149 * WPT + 0u];
            let tmp_152 = keys[0] < tmp_150 || (keys[0] == tmp_150 && values[0] < tmp_151);
            if tmp_148 == tmp_152 { keys[0] = tmp_150; values[0] = tmp_151; }
            let tmp_153 = smem_keys[tmp_149 * WPT + 1u];
            let tmp_154 = smem_vals[tmp_149 * WPT + 1u];
            let tmp_155 = keys[1] < tmp_153 || (keys[1] == tmp_153 && values[1] < tmp_154);
            if tmp_148 == tmp_155 { keys[1] = tmp_153; values[1] = tmp_154; }
            let tmp_156 = smem_keys[tmp_149 * WPT + 2u];
            let tmp_157 = smem_vals[tmp_149 * WPT + 2u];
            let tmp_158 = keys[2] < tmp_156 || (keys[2] == tmp_156 && values[2] < tmp_157);
            if tmp_148 == tmp_158 { keys[2] = tmp_156; values[2] = tmp_157; }
            let tmp_159 = smem_keys[tmp_149 * WPT + 3u];
            let tmp_160 = smem_vals[tmp_149 * WPT + 3u];
            let tmp_161 = keys[3] < tmp_159 || (keys[3] == tmp_159 && values[3] < tmp_160);
            if tmp_148 == tmp_161 { keys[3] = tmp_159; values[3] = tmp_160; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],1,0)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_162 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_163 = seg_base + (local_tid ^ 1u);
            let tmp_164 = smem_keys[tmp_163 * WPT + 0u];
            let tmp_165 = smem_vals[tmp_163 * WPT + 0u];
            let tmp_166 = keys[0] < tmp_164 || (keys[0] == tmp_164 && values[0] < tmp_165);
            if tmp_162 == tmp_166 { keys[0] = tmp_164; values[0] = tmp_165; }
            let tmp_167 = smem_keys[tmp_163 * WPT + 1u];
            let tmp_168 = smem_vals[tmp_163 * WPT + 1u];
            let tmp_169 = keys[1] < tmp_167 || (keys[1] == tmp_167 && values[1] < tmp_168);
            if tmp_162 == tmp_169 { keys[1] = tmp_167; values[1] = tmp_168; }
            let tmp_170 = smem_keys[tmp_163 * WPT + 2u];
            let tmp_171 = smem_vals[tmp_163 * WPT + 2u];
            let tmp_172 = keys[2] < tmp_170 || (keys[2] == tmp_170 && values[2] < tmp_171);
            if tmp_162 == tmp_172 { keys[2] = tmp_170; values[2] = tmp_171; }
            let tmp_173 = smem_keys[tmp_163 * WPT + 3u];
            let tmp_174 = smem_vals[tmp_163 * WPT + 3u];
            let tmp_175 = keys[3] < tmp_173 || (keys[3] == tmp_173 && values[3] < tmp_174);
            if tmp_162 == tmp_175 { keys[3] = tmp_173; values[3] = tmp_174; }
            workgroupBarrier();
        }
    }

    // exch_local(2,4)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_176 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_176;
                let tmp_177 = values[0]; values[0] = values[2]; values[2] = tmp_177;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_178 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_178;
                let tmp_179 = values[1]; values[1] = values[3]; values[3] = tmp_179;
            }
        }
    }

    // exch_local(1,4)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_180 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_180;
                let tmp_181 = values[0]; values[0] = values[1]; values[1] = tmp_181;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_182 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_182;
                let tmp_183 = values[2]; values[2] = values[3]; values[3] = tmp_183;
            }
        }
    }

    // exch_intxn(tmask:31,swbit:4,wpt:4)
    {
        // _exch_workgroup([(0, 3), (1, 2), (2, 1), (3, 0)],31,4)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_184 = extractBits(local_tid, 4u, 1u) != 0u;
            let tmp_185 = seg_base + (local_tid ^ 31u);
            let tmp_186 = smem_keys[tmp_185 * WPT + 3u];
            let tmp_187 = smem_vals[tmp_185 * WPT + 3u];
            let tmp_188 = keys[0] < tmp_186 || (keys[0] == tmp_186 && values[0] < tmp_187);
            if tmp_184 == tmp_188 { keys[0] = tmp_186; values[0] = tmp_187; }
            let tmp_189 = smem_keys[tmp_185 * WPT + 2u];
            let tmp_190 = smem_vals[tmp_185 * WPT + 2u];
            let tmp_191 = keys[1] < tmp_189 || (keys[1] == tmp_189 && values[1] < tmp_190);
            if tmp_184 == tmp_191 { keys[1] = tmp_189; values[1] = tmp_190; }
            let tmp_192 = smem_keys[tmp_185 * WPT + 1u];
            let tmp_193 = smem_vals[tmp_185 * WPT + 1u];
            let tmp_194 = keys[2] < tmp_192 || (keys[2] == tmp_192 && values[2] < tmp_193);
            if tmp_184 == tmp_194 { keys[2] = tmp_192; values[2] = tmp_193; }
            let tmp_195 = smem_keys[tmp_185 * WPT + 0u];
            let tmp_196 = smem_vals[tmp_185 * WPT + 0u];
            let tmp_197 = keys[3] < tmp_195 || (keys[3] == tmp_195 && values[3] < tmp_196);
            if tmp_184 == tmp_197 { keys[3] = tmp_195; values[3] = tmp_196; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:8,swbit:3,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],8,3)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_198 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_199 = seg_base + (local_tid ^ 8u);
            let tmp_200 = smem_keys[tmp_199 * WPT + 0u];
            let tmp_201 = smem_vals[tmp_199 * WPT + 0u];
            let tmp_202 = keys[0] < tmp_200 || (keys[0] == tmp_200 && values[0] < tmp_201);
            if tmp_198 == tmp_202 { keys[0] = tmp_200; values[0] = tmp_201; }
            let tmp_203 = smem_keys[tmp_199 * WPT + 1u];
            let tmp_204 = smem_vals[tmp_199 * WPT + 1u];
            let tmp_205 = keys[1] < tmp_203 || (keys[1] == tmp_203 && values[1] < tmp_204);
            if tmp_198 == tmp_205 { keys[1] = tmp_203; values[1] = tmp_204; }
            let tmp_206 = smem_keys[tmp_199 * WPT + 2u];
            let tmp_207 = smem_vals[tmp_199 * WPT + 2u];
            let tmp_208 = keys[2] < tmp_206 || (keys[2] == tmp_206 && values[2] < tmp_207);
            if tmp_198 == tmp_208 { keys[2] = tmp_206; values[2] = tmp_207; }
            let tmp_209 = smem_keys[tmp_199 * WPT + 3u];
            let tmp_210 = smem_vals[tmp_199 * WPT + 3u];
            let tmp_211 = keys[3] < tmp_209 || (keys[3] == tmp_209 && values[3] < tmp_210);
            if tmp_198 == tmp_211 { keys[3] = tmp_209; values[3] = tmp_210; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],4,2)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_212 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_213 = seg_base + (local_tid ^ 4u);
            let tmp_214 = smem_keys[tmp_213 * WPT + 0u];
            let tmp_215 = smem_vals[tmp_213 * WPT + 0u];
            let tmp_216 = keys[0] < tmp_214 || (keys[0] == tmp_214 && values[0] < tmp_215);
            if tmp_212 == tmp_216 { keys[0] = tmp_214; values[0] = tmp_215; }
            let tmp_217 = smem_keys[tmp_213 * WPT + 1u];
            let tmp_218 = smem_vals[tmp_213 * WPT + 1u];
            let tmp_219 = keys[1] < tmp_217 || (keys[1] == tmp_217 && values[1] < tmp_218);
            if tmp_212 == tmp_219 { keys[1] = tmp_217; values[1] = tmp_218; }
            let tmp_220 = smem_keys[tmp_213 * WPT + 2u];
            let tmp_221 = smem_vals[tmp_213 * WPT + 2u];
            let tmp_222 = keys[2] < tmp_220 || (keys[2] == tmp_220 && values[2] < tmp_221);
            if tmp_212 == tmp_222 { keys[2] = tmp_220; values[2] = tmp_221; }
            let tmp_223 = smem_keys[tmp_213 * WPT + 3u];
            let tmp_224 = smem_vals[tmp_213 * WPT + 3u];
            let tmp_225 = keys[3] < tmp_223 || (keys[3] == tmp_223 && values[3] < tmp_224);
            if tmp_212 == tmp_225 { keys[3] = tmp_223; values[3] = tmp_224; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],2,1)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_226 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_227 = seg_base + (local_tid ^ 2u);
            let tmp_228 = smem_keys[tmp_227 * WPT + 0u];
            let tmp_229 = smem_vals[tmp_227 * WPT + 0u];
            let tmp_230 = keys[0] < tmp_228 || (keys[0] == tmp_228 && values[0] < tmp_229);
            if tmp_226 == tmp_230 { keys[0] = tmp_228; values[0] = tmp_229; }
            let tmp_231 = smem_keys[tmp_227 * WPT + 1u];
            let tmp_232 = smem_vals[tmp_227 * WPT + 1u];
            let tmp_233 = keys[1] < tmp_231 || (keys[1] == tmp_231 && values[1] < tmp_232);
            if tmp_226 == tmp_233 { keys[1] = tmp_231; values[1] = tmp_232; }
            let tmp_234 = smem_keys[tmp_227 * WPT + 2u];
            let tmp_235 = smem_vals[tmp_227 * WPT + 2u];
            let tmp_236 = keys[2] < tmp_234 || (keys[2] == tmp_234 && values[2] < tmp_235);
            if tmp_226 == tmp_236 { keys[2] = tmp_234; values[2] = tmp_235; }
            let tmp_237 = smem_keys[tmp_227 * WPT + 3u];
            let tmp_238 = smem_vals[tmp_227 * WPT + 3u];
            let tmp_239 = keys[3] < tmp_237 || (keys[3] == tmp_237 && values[3] < tmp_238);
            if tmp_226 == tmp_239 { keys[3] = tmp_237; values[3] = tmp_238; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],1,0)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_240 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_241 = seg_base + (local_tid ^ 1u);
            let tmp_242 = smem_keys[tmp_241 * WPT + 0u];
            let tmp_243 = smem_vals[tmp_241 * WPT + 0u];
            let tmp_244 = keys[0] < tmp_242 || (keys[0] == tmp_242 && values[0] < tmp_243);
            if tmp_240 == tmp_244 { keys[0] = tmp_242; values[0] = tmp_243; }
            let tmp_245 = smem_keys[tmp_241 * WPT + 1u];
            let tmp_246 = smem_vals[tmp_241 * WPT + 1u];
            let tmp_247 = keys[1] < tmp_245 || (keys[1] == tmp_245 && values[1] < tmp_246);
            if tmp_240 == tmp_247 { keys[1] = tmp_245; values[1] = tmp_246; }
            let tmp_248 = smem_keys[tmp_241 * WPT + 2u];
            let tmp_249 = smem_vals[tmp_241 * WPT + 2u];
            let tmp_250 = keys[2] < tmp_248 || (keys[2] == tmp_248 && values[2] < tmp_249);
            if tmp_240 == tmp_250 { keys[2] = tmp_248; values[2] = tmp_249; }
            let tmp_251 = smem_keys[tmp_241 * WPT + 3u];
            let tmp_252 = smem_vals[tmp_241 * WPT + 3u];
            let tmp_253 = keys[3] < tmp_251 || (keys[3] == tmp_251 && values[3] < tmp_252);
            if tmp_240 == tmp_253 { keys[3] = tmp_251; values[3] = tmp_252; }
            workgroupBarrier();
        }
    }

    // exch_local(2,4)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_254 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_254;
                let tmp_255 = values[0]; values[0] = values[2]; values[2] = tmp_255;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_256 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_256;
                let tmp_257 = values[1]; values[1] = values[3]; values[3] = tmp_257;
            }
        }
    }

    // exch_local(1,4)
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
    }

    // exch_intxn(tmask:63,swbit:5,wpt:4)
    {
        // _exch_workgroup([(0, 3), (1, 2), (2, 1), (3, 0)],63,5)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_262 = extractBits(local_tid, 5u, 1u) != 0u;
            let tmp_263 = seg_base + (local_tid ^ 63u);
            let tmp_264 = smem_keys[tmp_263 * WPT + 3u];
            let tmp_265 = smem_vals[tmp_263 * WPT + 3u];
            let tmp_266 = keys[0] < tmp_264 || (keys[0] == tmp_264 && values[0] < tmp_265);
            if tmp_262 == tmp_266 { keys[0] = tmp_264; values[0] = tmp_265; }
            let tmp_267 = smem_keys[tmp_263 * WPT + 2u];
            let tmp_268 = smem_vals[tmp_263 * WPT + 2u];
            let tmp_269 = keys[1] < tmp_267 || (keys[1] == tmp_267 && values[1] < tmp_268);
            if tmp_262 == tmp_269 { keys[1] = tmp_267; values[1] = tmp_268; }
            let tmp_270 = smem_keys[tmp_263 * WPT + 1u];
            let tmp_271 = smem_vals[tmp_263 * WPT + 1u];
            let tmp_272 = keys[2] < tmp_270 || (keys[2] == tmp_270 && values[2] < tmp_271);
            if tmp_262 == tmp_272 { keys[2] = tmp_270; values[2] = tmp_271; }
            let tmp_273 = smem_keys[tmp_263 * WPT + 0u];
            let tmp_274 = smem_vals[tmp_263 * WPT + 0u];
            let tmp_275 = keys[3] < tmp_273 || (keys[3] == tmp_273 && values[3] < tmp_274);
            if tmp_262 == tmp_275 { keys[3] = tmp_273; values[3] = tmp_274; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:16,swbit:4,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],16,4)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_276 = extractBits(local_tid, 4u, 1u) != 0u;
            let tmp_277 = seg_base + (local_tid ^ 16u);
            let tmp_278 = smem_keys[tmp_277 * WPT + 0u];
            let tmp_279 = smem_vals[tmp_277 * WPT + 0u];
            let tmp_280 = keys[0] < tmp_278 || (keys[0] == tmp_278 && values[0] < tmp_279);
            if tmp_276 == tmp_280 { keys[0] = tmp_278; values[0] = tmp_279; }
            let tmp_281 = smem_keys[tmp_277 * WPT + 1u];
            let tmp_282 = smem_vals[tmp_277 * WPT + 1u];
            let tmp_283 = keys[1] < tmp_281 || (keys[1] == tmp_281 && values[1] < tmp_282);
            if tmp_276 == tmp_283 { keys[1] = tmp_281; values[1] = tmp_282; }
            let tmp_284 = smem_keys[tmp_277 * WPT + 2u];
            let tmp_285 = smem_vals[tmp_277 * WPT + 2u];
            let tmp_286 = keys[2] < tmp_284 || (keys[2] == tmp_284 && values[2] < tmp_285);
            if tmp_276 == tmp_286 { keys[2] = tmp_284; values[2] = tmp_285; }
            let tmp_287 = smem_keys[tmp_277 * WPT + 3u];
            let tmp_288 = smem_vals[tmp_277 * WPT + 3u];
            let tmp_289 = keys[3] < tmp_287 || (keys[3] == tmp_287 && values[3] < tmp_288);
            if tmp_276 == tmp_289 { keys[3] = tmp_287; values[3] = tmp_288; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:8,swbit:3,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],8,3)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_290 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_291 = seg_base + (local_tid ^ 8u);
            let tmp_292 = smem_keys[tmp_291 * WPT + 0u];
            let tmp_293 = smem_vals[tmp_291 * WPT + 0u];
            let tmp_294 = keys[0] < tmp_292 || (keys[0] == tmp_292 && values[0] < tmp_293);
            if tmp_290 == tmp_294 { keys[0] = tmp_292; values[0] = tmp_293; }
            let tmp_295 = smem_keys[tmp_291 * WPT + 1u];
            let tmp_296 = smem_vals[tmp_291 * WPT + 1u];
            let tmp_297 = keys[1] < tmp_295 || (keys[1] == tmp_295 && values[1] < tmp_296);
            if tmp_290 == tmp_297 { keys[1] = tmp_295; values[1] = tmp_296; }
            let tmp_298 = smem_keys[tmp_291 * WPT + 2u];
            let tmp_299 = smem_vals[tmp_291 * WPT + 2u];
            let tmp_300 = keys[2] < tmp_298 || (keys[2] == tmp_298 && values[2] < tmp_299);
            if tmp_290 == tmp_300 { keys[2] = tmp_298; values[2] = tmp_299; }
            let tmp_301 = smem_keys[tmp_291 * WPT + 3u];
            let tmp_302 = smem_vals[tmp_291 * WPT + 3u];
            let tmp_303 = keys[3] < tmp_301 || (keys[3] == tmp_301 && values[3] < tmp_302);
            if tmp_290 == tmp_303 { keys[3] = tmp_301; values[3] = tmp_302; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],4,2)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_304 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_305 = seg_base + (local_tid ^ 4u);
            let tmp_306 = smem_keys[tmp_305 * WPT + 0u];
            let tmp_307 = smem_vals[tmp_305 * WPT + 0u];
            let tmp_308 = keys[0] < tmp_306 || (keys[0] == tmp_306 && values[0] < tmp_307);
            if tmp_304 == tmp_308 { keys[0] = tmp_306; values[0] = tmp_307; }
            let tmp_309 = smem_keys[tmp_305 * WPT + 1u];
            let tmp_310 = smem_vals[tmp_305 * WPT + 1u];
            let tmp_311 = keys[1] < tmp_309 || (keys[1] == tmp_309 && values[1] < tmp_310);
            if tmp_304 == tmp_311 { keys[1] = tmp_309; values[1] = tmp_310; }
            let tmp_312 = smem_keys[tmp_305 * WPT + 2u];
            let tmp_313 = smem_vals[tmp_305 * WPT + 2u];
            let tmp_314 = keys[2] < tmp_312 || (keys[2] == tmp_312 && values[2] < tmp_313);
            if tmp_304 == tmp_314 { keys[2] = tmp_312; values[2] = tmp_313; }
            let tmp_315 = smem_keys[tmp_305 * WPT + 3u];
            let tmp_316 = smem_vals[tmp_305 * WPT + 3u];
            let tmp_317 = keys[3] < tmp_315 || (keys[3] == tmp_315 && values[3] < tmp_316);
            if tmp_304 == tmp_317 { keys[3] = tmp_315; values[3] = tmp_316; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],2,1)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_318 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_319 = seg_base + (local_tid ^ 2u);
            let tmp_320 = smem_keys[tmp_319 * WPT + 0u];
            let tmp_321 = smem_vals[tmp_319 * WPT + 0u];
            let tmp_322 = keys[0] < tmp_320 || (keys[0] == tmp_320 && values[0] < tmp_321);
            if tmp_318 == tmp_322 { keys[0] = tmp_320; values[0] = tmp_321; }
            let tmp_323 = smem_keys[tmp_319 * WPT + 1u];
            let tmp_324 = smem_vals[tmp_319 * WPT + 1u];
            let tmp_325 = keys[1] < tmp_323 || (keys[1] == tmp_323 && values[1] < tmp_324);
            if tmp_318 == tmp_325 { keys[1] = tmp_323; values[1] = tmp_324; }
            let tmp_326 = smem_keys[tmp_319 * WPT + 2u];
            let tmp_327 = smem_vals[tmp_319 * WPT + 2u];
            let tmp_328 = keys[2] < tmp_326 || (keys[2] == tmp_326 && values[2] < tmp_327);
            if tmp_318 == tmp_328 { keys[2] = tmp_326; values[2] = tmp_327; }
            let tmp_329 = smem_keys[tmp_319 * WPT + 3u];
            let tmp_330 = smem_vals[tmp_319 * WPT + 3u];
            let tmp_331 = keys[3] < tmp_329 || (keys[3] == tmp_329 && values[3] < tmp_330);
            if tmp_318 == tmp_331 { keys[3] = tmp_329; values[3] = tmp_330; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],1,0)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_332 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_333 = seg_base + (local_tid ^ 1u);
            let tmp_334 = smem_keys[tmp_333 * WPT + 0u];
            let tmp_335 = smem_vals[tmp_333 * WPT + 0u];
            let tmp_336 = keys[0] < tmp_334 || (keys[0] == tmp_334 && values[0] < tmp_335);
            if tmp_332 == tmp_336 { keys[0] = tmp_334; values[0] = tmp_335; }
            let tmp_337 = smem_keys[tmp_333 * WPT + 1u];
            let tmp_338 = smem_vals[tmp_333 * WPT + 1u];
            let tmp_339 = keys[1] < tmp_337 || (keys[1] == tmp_337 && values[1] < tmp_338);
            if tmp_332 == tmp_339 { keys[1] = tmp_337; values[1] = tmp_338; }
            let tmp_340 = smem_keys[tmp_333 * WPT + 2u];
            let tmp_341 = smem_vals[tmp_333 * WPT + 2u];
            let tmp_342 = keys[2] < tmp_340 || (keys[2] == tmp_340 && values[2] < tmp_341);
            if tmp_332 == tmp_342 { keys[2] = tmp_340; values[2] = tmp_341; }
            let tmp_343 = smem_keys[tmp_333 * WPT + 3u];
            let tmp_344 = smem_vals[tmp_333 * WPT + 3u];
            let tmp_345 = keys[3] < tmp_343 || (keys[3] == tmp_343 && values[3] < tmp_344);
            if tmp_332 == tmp_345 { keys[3] = tmp_343; values[3] = tmp_344; }
            workgroupBarrier();
        }
    }

    // exch_local(2,4)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_346 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_346;
                let tmp_347 = values[0]; values[0] = values[2]; values[2] = tmp_347;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_348 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_348;
                let tmp_349 = values[1]; values[1] = values[3]; values[3] = tmp_349;
            }
        }
    }

    // exch_local(1,4)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_350 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_350;
                let tmp_351 = values[0]; values[0] = values[1]; values[1] = tmp_351;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_352 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_352;
                let tmp_353 = values[2]; values[2] = values[3]; values[3] = tmp_353;
            }
        }
    }

    // exch_intxn(tmask:127,swbit:6,wpt:4)
    {
        // _exch_workgroup([(0, 3), (1, 2), (2, 1), (3, 0)],127,6)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_354 = extractBits(local_tid, 6u, 1u) != 0u;
            let tmp_355 = seg_base + (local_tid ^ 127u);
            let tmp_356 = smem_keys[tmp_355 * WPT + 3u];
            let tmp_357 = smem_vals[tmp_355 * WPT + 3u];
            let tmp_358 = keys[0] < tmp_356 || (keys[0] == tmp_356 && values[0] < tmp_357);
            if tmp_354 == tmp_358 { keys[0] = tmp_356; values[0] = tmp_357; }
            let tmp_359 = smem_keys[tmp_355 * WPT + 2u];
            let tmp_360 = smem_vals[tmp_355 * WPT + 2u];
            let tmp_361 = keys[1] < tmp_359 || (keys[1] == tmp_359 && values[1] < tmp_360);
            if tmp_354 == tmp_361 { keys[1] = tmp_359; values[1] = tmp_360; }
            let tmp_362 = smem_keys[tmp_355 * WPT + 1u];
            let tmp_363 = smem_vals[tmp_355 * WPT + 1u];
            let tmp_364 = keys[2] < tmp_362 || (keys[2] == tmp_362 && values[2] < tmp_363);
            if tmp_354 == tmp_364 { keys[2] = tmp_362; values[2] = tmp_363; }
            let tmp_365 = smem_keys[tmp_355 * WPT + 0u];
            let tmp_366 = smem_vals[tmp_355 * WPT + 0u];
            let tmp_367 = keys[3] < tmp_365 || (keys[3] == tmp_365 && values[3] < tmp_366);
            if tmp_354 == tmp_367 { keys[3] = tmp_365; values[3] = tmp_366; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:32,swbit:5,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],32,5)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_368 = extractBits(local_tid, 5u, 1u) != 0u;
            let tmp_369 = seg_base + (local_tid ^ 32u);
            let tmp_370 = smem_keys[tmp_369 * WPT + 0u];
            let tmp_371 = smem_vals[tmp_369 * WPT + 0u];
            let tmp_372 = keys[0] < tmp_370 || (keys[0] == tmp_370 && values[0] < tmp_371);
            if tmp_368 == tmp_372 { keys[0] = tmp_370; values[0] = tmp_371; }
            let tmp_373 = smem_keys[tmp_369 * WPT + 1u];
            let tmp_374 = smem_vals[tmp_369 * WPT + 1u];
            let tmp_375 = keys[1] < tmp_373 || (keys[1] == tmp_373 && values[1] < tmp_374);
            if tmp_368 == tmp_375 { keys[1] = tmp_373; values[1] = tmp_374; }
            let tmp_376 = smem_keys[tmp_369 * WPT + 2u];
            let tmp_377 = smem_vals[tmp_369 * WPT + 2u];
            let tmp_378 = keys[2] < tmp_376 || (keys[2] == tmp_376 && values[2] < tmp_377);
            if tmp_368 == tmp_378 { keys[2] = tmp_376; values[2] = tmp_377; }
            let tmp_379 = smem_keys[tmp_369 * WPT + 3u];
            let tmp_380 = smem_vals[tmp_369 * WPT + 3u];
            let tmp_381 = keys[3] < tmp_379 || (keys[3] == tmp_379 && values[3] < tmp_380);
            if tmp_368 == tmp_381 { keys[3] = tmp_379; values[3] = tmp_380; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:16,swbit:4,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],16,4)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_382 = extractBits(local_tid, 4u, 1u) != 0u;
            let tmp_383 = seg_base + (local_tid ^ 16u);
            let tmp_384 = smem_keys[tmp_383 * WPT + 0u];
            let tmp_385 = smem_vals[tmp_383 * WPT + 0u];
            let tmp_386 = keys[0] < tmp_384 || (keys[0] == tmp_384 && values[0] < tmp_385);
            if tmp_382 == tmp_386 { keys[0] = tmp_384; values[0] = tmp_385; }
            let tmp_387 = smem_keys[tmp_383 * WPT + 1u];
            let tmp_388 = smem_vals[tmp_383 * WPT + 1u];
            let tmp_389 = keys[1] < tmp_387 || (keys[1] == tmp_387 && values[1] < tmp_388);
            if tmp_382 == tmp_389 { keys[1] = tmp_387; values[1] = tmp_388; }
            let tmp_390 = smem_keys[tmp_383 * WPT + 2u];
            let tmp_391 = smem_vals[tmp_383 * WPT + 2u];
            let tmp_392 = keys[2] < tmp_390 || (keys[2] == tmp_390 && values[2] < tmp_391);
            if tmp_382 == tmp_392 { keys[2] = tmp_390; values[2] = tmp_391; }
            let tmp_393 = smem_keys[tmp_383 * WPT + 3u];
            let tmp_394 = smem_vals[tmp_383 * WPT + 3u];
            let tmp_395 = keys[3] < tmp_393 || (keys[3] == tmp_393 && values[3] < tmp_394);
            if tmp_382 == tmp_395 { keys[3] = tmp_393; values[3] = tmp_394; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:8,swbit:3,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],8,3)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_396 = extractBits(local_tid, 3u, 1u) != 0u;
            let tmp_397 = seg_base + (local_tid ^ 8u);
            let tmp_398 = smem_keys[tmp_397 * WPT + 0u];
            let tmp_399 = smem_vals[tmp_397 * WPT + 0u];
            let tmp_400 = keys[0] < tmp_398 || (keys[0] == tmp_398 && values[0] < tmp_399);
            if tmp_396 == tmp_400 { keys[0] = tmp_398; values[0] = tmp_399; }
            let tmp_401 = smem_keys[tmp_397 * WPT + 1u];
            let tmp_402 = smem_vals[tmp_397 * WPT + 1u];
            let tmp_403 = keys[1] < tmp_401 || (keys[1] == tmp_401 && values[1] < tmp_402);
            if tmp_396 == tmp_403 { keys[1] = tmp_401; values[1] = tmp_402; }
            let tmp_404 = smem_keys[tmp_397 * WPT + 2u];
            let tmp_405 = smem_vals[tmp_397 * WPT + 2u];
            let tmp_406 = keys[2] < tmp_404 || (keys[2] == tmp_404 && values[2] < tmp_405);
            if tmp_396 == tmp_406 { keys[2] = tmp_404; values[2] = tmp_405; }
            let tmp_407 = smem_keys[tmp_397 * WPT + 3u];
            let tmp_408 = smem_vals[tmp_397 * WPT + 3u];
            let tmp_409 = keys[3] < tmp_407 || (keys[3] == tmp_407 && values[3] < tmp_408);
            if tmp_396 == tmp_409 { keys[3] = tmp_407; values[3] = tmp_408; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:4,swbit:2,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],4,2)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_410 = extractBits(local_tid, 2u, 1u) != 0u;
            let tmp_411 = seg_base + (local_tid ^ 4u);
            let tmp_412 = smem_keys[tmp_411 * WPT + 0u];
            let tmp_413 = smem_vals[tmp_411 * WPT + 0u];
            let tmp_414 = keys[0] < tmp_412 || (keys[0] == tmp_412 && values[0] < tmp_413);
            if tmp_410 == tmp_414 { keys[0] = tmp_412; values[0] = tmp_413; }
            let tmp_415 = smem_keys[tmp_411 * WPT + 1u];
            let tmp_416 = smem_vals[tmp_411 * WPT + 1u];
            let tmp_417 = keys[1] < tmp_415 || (keys[1] == tmp_415 && values[1] < tmp_416);
            if tmp_410 == tmp_417 { keys[1] = tmp_415; values[1] = tmp_416; }
            let tmp_418 = smem_keys[tmp_411 * WPT + 2u];
            let tmp_419 = smem_vals[tmp_411 * WPT + 2u];
            let tmp_420 = keys[2] < tmp_418 || (keys[2] == tmp_418 && values[2] < tmp_419);
            if tmp_410 == tmp_420 { keys[2] = tmp_418; values[2] = tmp_419; }
            let tmp_421 = smem_keys[tmp_411 * WPT + 3u];
            let tmp_422 = smem_vals[tmp_411 * WPT + 3u];
            let tmp_423 = keys[3] < tmp_421 || (keys[3] == tmp_421 && values[3] < tmp_422);
            if tmp_410 == tmp_423 { keys[3] = tmp_421; values[3] = tmp_422; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:2,swbit:1,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],2,1)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_424 = extractBits(local_tid, 1u, 1u) != 0u;
            let tmp_425 = seg_base + (local_tid ^ 2u);
            let tmp_426 = smem_keys[tmp_425 * WPT + 0u];
            let tmp_427 = smem_vals[tmp_425 * WPT + 0u];
            let tmp_428 = keys[0] < tmp_426 || (keys[0] == tmp_426 && values[0] < tmp_427);
            if tmp_424 == tmp_428 { keys[0] = tmp_426; values[0] = tmp_427; }
            let tmp_429 = smem_keys[tmp_425 * WPT + 1u];
            let tmp_430 = smem_vals[tmp_425 * WPT + 1u];
            let tmp_431 = keys[1] < tmp_429 || (keys[1] == tmp_429 && values[1] < tmp_430);
            if tmp_424 == tmp_431 { keys[1] = tmp_429; values[1] = tmp_430; }
            let tmp_432 = smem_keys[tmp_425 * WPT + 2u];
            let tmp_433 = smem_vals[tmp_425 * WPT + 2u];
            let tmp_434 = keys[2] < tmp_432 || (keys[2] == tmp_432 && values[2] < tmp_433);
            if tmp_424 == tmp_434 { keys[2] = tmp_432; values[2] = tmp_433; }
            let tmp_435 = smem_keys[tmp_425 * WPT + 3u];
            let tmp_436 = smem_vals[tmp_425 * WPT + 3u];
            let tmp_437 = keys[3] < tmp_435 || (keys[3] == tmp_435 && values[3] < tmp_436);
            if tmp_424 == tmp_437 { keys[3] = tmp_435; values[3] = tmp_436; }
            workgroupBarrier();
        }
    }

    // exch_paral(tmask:1,swbit:0,wpt:4)
    {
        // _exch_workgroup([(0, 0), (1, 1), (2, 2), (3, 3)],1,0)
        {
            smem_keys[tid_g * WPT + 0u] = keys[0];
            smem_vals[tid_g * WPT + 0u] = values[0];
            smem_keys[tid_g * WPT + 1u] = keys[1];
            smem_vals[tid_g * WPT + 1u] = values[1];
            smem_keys[tid_g * WPT + 2u] = keys[2];
            smem_vals[tid_g * WPT + 2u] = values[2];
            smem_keys[tid_g * WPT + 3u] = keys[3];
            smem_vals[tid_g * WPT + 3u] = values[3];
            workgroupBarrier();
            let tmp_438 = extractBits(local_tid, 0u, 1u) != 0u;
            let tmp_439 = seg_base + (local_tid ^ 1u);
            let tmp_440 = smem_keys[tmp_439 * WPT + 0u];
            let tmp_441 = smem_vals[tmp_439 * WPT + 0u];
            let tmp_442 = keys[0] < tmp_440 || (keys[0] == tmp_440 && values[0] < tmp_441);
            if tmp_438 == tmp_442 { keys[0] = tmp_440; values[0] = tmp_441; }
            let tmp_443 = smem_keys[tmp_439 * WPT + 1u];
            let tmp_444 = smem_vals[tmp_439 * WPT + 1u];
            let tmp_445 = keys[1] < tmp_443 || (keys[1] == tmp_443 && values[1] < tmp_444);
            if tmp_438 == tmp_445 { keys[1] = tmp_443; values[1] = tmp_444; }
            let tmp_446 = smem_keys[tmp_439 * WPT + 2u];
            let tmp_447 = smem_vals[tmp_439 * WPT + 2u];
            let tmp_448 = keys[2] < tmp_446 || (keys[2] == tmp_446 && values[2] < tmp_447);
            if tmp_438 == tmp_448 { keys[2] = tmp_446; values[2] = tmp_447; }
            let tmp_449 = smem_keys[tmp_439 * WPT + 3u];
            let tmp_450 = smem_vals[tmp_439 * WPT + 3u];
            let tmp_451 = keys[3] < tmp_449 || (keys[3] == tmp_449 && values[3] < tmp_450);
            if tmp_438 == tmp_451 { keys[3] = tmp_449; values[3] = tmp_450; }
            workgroupBarrier();
        }
    }

    // exch_local(2,4)
    {
        // cmp_swap(0,2)
        if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
            // swap(0,2)
            {
                let tmp_452 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_452;
                let tmp_453 = values[0]; values[0] = values[2]; values[2] = tmp_453;
            }
        }
        // cmp_swap(1,3)
        if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
            // swap(1,3)
            {
                let tmp_454 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_454;
                let tmp_455 = values[1]; values[1] = values[3]; values[3] = tmp_455;
            }
        }
    }

    // exch_local(1,4)
    {
        // cmp_swap(0,1)
        if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
            // swap(0,1)
            {
                let tmp_456 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_456;
                let tmp_457 = values[0]; values[0] = values[1]; values[1] = tmp_457;
            }
        }
        // cmp_swap(2,3)
        if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
            // swap(2,3)
            {
                let tmp_458 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_458;
                let tmp_459 = values[2]; values[2] = values[3]; values[3] = tmp_459;
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
