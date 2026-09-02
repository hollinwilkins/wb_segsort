
enable subgroups;

override WG: u32 = 32u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 64u;
const M: u32 = 32u;
const WPT: u32 = 2u;
const R: u32 = 8u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_hybrid_sg8_n64_m32_striped(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 6u;

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

    var keys: array<u32, 2>;
    var values: array<u32, 2>;

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

    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_0 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_0;let tmp_1 = values[0]; values[0] = values[1]; values[1] = tmp_1; }
    }
    // exch_intxn(tmask:1,swbit:0,wpt:2)
    {
    let tmp_2 = subgroupShuffleXor(keys[1], 1u);
    let tmp_3 = subgroupShuffleXor(values[1], 1u);
    let tmp_4 = subgroupShuffleXor(keys[0], 1u);
    let tmp_5 = subgroupShuffleXor(values[0], 1u);
    let tmp_6 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_7 = keys[0] < tmp_2 || (keys[0] == tmp_2 && values[0] < tmp_3);
    if tmp_6 == tmp_7 { keys[0] = tmp_2; values[0] = tmp_3; }
    let tmp_8 = keys[1] < tmp_4 || (keys[1] == tmp_4 && values[1] < tmp_5);
    if tmp_6 == tmp_8 { keys[1] = tmp_4; values[1] = tmp_5; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_9 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_9;let tmp_10 = values[0]; values[0] = values[1]; values[1] = tmp_10; }
    }
    // exch_intxn(tmask:3,swbit:1,wpt:2)
    {
    let tmp_11 = subgroupShuffleXor(keys[1], 3u);
    let tmp_12 = subgroupShuffleXor(values[1], 3u);
    let tmp_13 = subgroupShuffleXor(keys[0], 3u);
    let tmp_14 = subgroupShuffleXor(values[0], 3u);
    let tmp_15 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_16 = keys[0] < tmp_11 || (keys[0] == tmp_11 && values[0] < tmp_12);
    if tmp_15 == tmp_16 { keys[0] = tmp_11; values[0] = tmp_12; }
    let tmp_17 = keys[1] < tmp_13 || (keys[1] == tmp_13 && values[1] < tmp_14);
    if tmp_15 == tmp_17 { keys[1] = tmp_13; values[1] = tmp_14; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_18 = subgroupShuffleXor(keys[0], 1u);
    let tmp_19 = subgroupShuffleXor(values[0], 1u);
    let tmp_20 = subgroupShuffleXor(keys[1], 1u);
    let tmp_21 = subgroupShuffleXor(values[1], 1u);
    let tmp_22 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_23 = keys[0] < tmp_18 || (keys[0] == tmp_18 && values[0] < tmp_19);
    if tmp_22 == tmp_23 { keys[0] = tmp_18; values[0] = tmp_19; }
    let tmp_24 = keys[1] < tmp_20 || (keys[1] == tmp_20 && values[1] < tmp_21);
    if tmp_22 == tmp_24 { keys[1] = tmp_20; values[1] = tmp_21; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_25 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_25;let tmp_26 = values[0]; values[0] = values[1]; values[1] = tmp_26; }
    }
    // exch_intxn(tmask:7,swbit:2,wpt:2)
    {
    let tmp_27 = subgroupShuffleXor(keys[1], 7u);
    let tmp_28 = subgroupShuffleXor(values[1], 7u);
    let tmp_29 = subgroupShuffleXor(keys[0], 7u);
    let tmp_30 = subgroupShuffleXor(values[0], 7u);
    let tmp_31 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_32 = keys[0] < tmp_27 || (keys[0] == tmp_27 && values[0] < tmp_28);
    if tmp_31 == tmp_32 { keys[0] = tmp_27; values[0] = tmp_28; }
    let tmp_33 = keys[1] < tmp_29 || (keys[1] == tmp_29 && values[1] < tmp_30);
    if tmp_31 == tmp_33 { keys[1] = tmp_29; values[1] = tmp_30; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_34 = subgroupShuffleXor(keys[0], 2u);
    let tmp_35 = subgroupShuffleXor(values[0], 2u);
    let tmp_36 = subgroupShuffleXor(keys[1], 2u);
    let tmp_37 = subgroupShuffleXor(values[1], 2u);
    let tmp_38 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_39 = keys[0] < tmp_34 || (keys[0] == tmp_34 && values[0] < tmp_35);
    if tmp_38 == tmp_39 { keys[0] = tmp_34; values[0] = tmp_35; }
    let tmp_40 = keys[1] < tmp_36 || (keys[1] == tmp_36 && values[1] < tmp_37);
    if tmp_38 == tmp_40 { keys[1] = tmp_36; values[1] = tmp_37; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_41 = subgroupShuffleXor(keys[0], 1u);
    let tmp_42 = subgroupShuffleXor(values[0], 1u);
    let tmp_43 = subgroupShuffleXor(keys[1], 1u);
    let tmp_44 = subgroupShuffleXor(values[1], 1u);
    let tmp_45 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_46 = keys[0] < tmp_41 || (keys[0] == tmp_41 && values[0] < tmp_42);
    if tmp_45 == tmp_46 { keys[0] = tmp_41; values[0] = tmp_42; }
    let tmp_47 = keys[1] < tmp_43 || (keys[1] == tmp_43 && values[1] < tmp_44);
    if tmp_45 == tmp_47 { keys[1] = tmp_43; values[1] = tmp_44; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_48 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_48;let tmp_49 = values[0]; values[0] = values[1]; values[1] = tmp_49; }
    }
    // exch_intxn(tmask:15,swbit:3,wpt:2)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_50 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_51 = seg_base + (local_tid ^ 15u); let tmp_52 = smem_keys[tmp_51 * WPT + 1u]; let tmp_53 = smem_vals[tmp_51 * WPT + 1u]; let tmp_54 = keys[0] < tmp_52 || (keys[0] == tmp_52 && values[0] < tmp_53); if tmp_50 == tmp_54 { keys[0] = tmp_52; values[0] = tmp_53; } let tmp_55 = smem_keys[tmp_51 * WPT + 0u]; let tmp_56 = smem_vals[tmp_51 * WPT + 0u]; let tmp_57 = keys[1] < tmp_55 || (keys[1] == tmp_55 && values[1] < tmp_56); if tmp_50 == tmp_57 { keys[1] = tmp_55; values[1] = tmp_56; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    {
    let tmp_58 = subgroupShuffleXor(keys[0], 4u);
    let tmp_59 = subgroupShuffleXor(values[0], 4u);
    let tmp_60 = subgroupShuffleXor(keys[1], 4u);
    let tmp_61 = subgroupShuffleXor(values[1], 4u);
    let tmp_62 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_63 = keys[0] < tmp_58 || (keys[0] == tmp_58 && values[0] < tmp_59);
    if tmp_62 == tmp_63 { keys[0] = tmp_58; values[0] = tmp_59; }
    let tmp_64 = keys[1] < tmp_60 || (keys[1] == tmp_60 && values[1] < tmp_61);
    if tmp_62 == tmp_64 { keys[1] = tmp_60; values[1] = tmp_61; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_65 = subgroupShuffleXor(keys[0], 2u);
    let tmp_66 = subgroupShuffleXor(values[0], 2u);
    let tmp_67 = subgroupShuffleXor(keys[1], 2u);
    let tmp_68 = subgroupShuffleXor(values[1], 2u);
    let tmp_69 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_70 = keys[0] < tmp_65 || (keys[0] == tmp_65 && values[0] < tmp_66);
    if tmp_69 == tmp_70 { keys[0] = tmp_65; values[0] = tmp_66; }
    let tmp_71 = keys[1] < tmp_67 || (keys[1] == tmp_67 && values[1] < tmp_68);
    if tmp_69 == tmp_71 { keys[1] = tmp_67; values[1] = tmp_68; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_72 = subgroupShuffleXor(keys[0], 1u);
    let tmp_73 = subgroupShuffleXor(values[0], 1u);
    let tmp_74 = subgroupShuffleXor(keys[1], 1u);
    let tmp_75 = subgroupShuffleXor(values[1], 1u);
    let tmp_76 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_77 = keys[0] < tmp_72 || (keys[0] == tmp_72 && values[0] < tmp_73);
    if tmp_76 == tmp_77 { keys[0] = tmp_72; values[0] = tmp_73; }
    let tmp_78 = keys[1] < tmp_74 || (keys[1] == tmp_74 && values[1] < tmp_75);
    if tmp_76 == tmp_78 { keys[1] = tmp_74; values[1] = tmp_75; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_79 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_79;let tmp_80 = values[0]; values[0] = values[1]; values[1] = tmp_80; }
    }
    // exch_intxn(tmask:31,swbit:4,wpt:2)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_81 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_82 = seg_base + (local_tid ^ 31u); let tmp_83 = smem_keys[tmp_82 * WPT + 1u]; let tmp_84 = smem_vals[tmp_82 * WPT + 1u]; let tmp_85 = keys[0] < tmp_83 || (keys[0] == tmp_83 && values[0] < tmp_84); if tmp_81 == tmp_85 { keys[0] = tmp_83; values[0] = tmp_84; } let tmp_86 = smem_keys[tmp_82 * WPT + 0u]; let tmp_87 = smem_vals[tmp_82 * WPT + 0u]; let tmp_88 = keys[1] < tmp_86 || (keys[1] == tmp_86 && values[1] < tmp_87); if tmp_81 == tmp_88 { keys[1] = tmp_86; values[1] = tmp_87; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_89 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_90 = seg_base + (local_tid ^ 8u); let tmp_91 = smem_keys[tmp_90 * WPT + 0u]; let tmp_92 = smem_vals[tmp_90 * WPT + 0u]; let tmp_93 = keys[0] < tmp_91 || (keys[0] == tmp_91 && values[0] < tmp_92); if tmp_89 == tmp_93 { keys[0] = tmp_91; values[0] = tmp_92; } let tmp_94 = smem_keys[tmp_90 * WPT + 1u]; let tmp_95 = smem_vals[tmp_90 * WPT + 1u]; let tmp_96 = keys[1] < tmp_94 || (keys[1] == tmp_94 && values[1] < tmp_95); if tmp_89 == tmp_96 { keys[1] = tmp_94; values[1] = tmp_95; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    {
    let tmp_97 = subgroupShuffleXor(keys[0], 4u);
    let tmp_98 = subgroupShuffleXor(values[0], 4u);
    let tmp_99 = subgroupShuffleXor(keys[1], 4u);
    let tmp_100 = subgroupShuffleXor(values[1], 4u);
    let tmp_101 = extractBits(local_tid, 2u, 1u) != 0u;
    let tmp_102 = keys[0] < tmp_97 || (keys[0] == tmp_97 && values[0] < tmp_98);
    if tmp_101 == tmp_102 { keys[0] = tmp_97; values[0] = tmp_98; }
    let tmp_103 = keys[1] < tmp_99 || (keys[1] == tmp_99 && values[1] < tmp_100);
    if tmp_101 == tmp_103 { keys[1] = tmp_99; values[1] = tmp_100; }
    }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    {
    let tmp_104 = subgroupShuffleXor(keys[0], 2u);
    let tmp_105 = subgroupShuffleXor(values[0], 2u);
    let tmp_106 = subgroupShuffleXor(keys[1], 2u);
    let tmp_107 = subgroupShuffleXor(values[1], 2u);
    let tmp_108 = extractBits(local_tid, 1u, 1u) != 0u;
    let tmp_109 = keys[0] < tmp_104 || (keys[0] == tmp_104 && values[0] < tmp_105);
    if tmp_108 == tmp_109 { keys[0] = tmp_104; values[0] = tmp_105; }
    let tmp_110 = keys[1] < tmp_106 || (keys[1] == tmp_106 && values[1] < tmp_107);
    if tmp_108 == tmp_110 { keys[1] = tmp_106; values[1] = tmp_107; }
    }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    {
    let tmp_111 = subgroupShuffleXor(keys[0], 1u);
    let tmp_112 = subgroupShuffleXor(values[0], 1u);
    let tmp_113 = subgroupShuffleXor(keys[1], 1u);
    let tmp_114 = subgroupShuffleXor(values[1], 1u);
    let tmp_115 = extractBits(local_tid, 0u, 1u) != 0u;
    let tmp_116 = keys[0] < tmp_111 || (keys[0] == tmp_111 && values[0] < tmp_112);
    if tmp_115 == tmp_116 { keys[0] = tmp_111; values[0] = tmp_112; }
    let tmp_117 = keys[1] < tmp_113 || (keys[1] == tmp_113 && values[1] < tmp_114);
    if tmp_115 == tmp_117 { keys[1] = tmp_113; values[1] = tmp_114; }
    }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_118 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_118;let tmp_119 = values[0]; values[0] = values[1]; values[1] = tmp_119; }
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
