
override WG: u32 = 32u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 128u;
const M: u32 = 32u;
const WPT: u32 = 4u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n128_m32_striped(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 7u;

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
    // exch_local(3,4) 
    // cmp_swap(0,3)
    if keys[0] > keys[3] || (keys[0] == keys[3] && values[0] > values[3]) {
    // swap(0,3) 
    { let tmp_4 = keys[0]; keys[0] = keys[3]; keys[3] = tmp_4;let tmp_5 = values[0]; values[0] = values[3]; values[3] = tmp_5; }
    }
    // cmp_swap(1,2)
    if keys[1] > keys[2] || (keys[1] == keys[2] && values[1] > values[2]) {
    // swap(1,2) 
    { let tmp_6 = keys[1]; keys[1] = keys[2]; keys[2] = tmp_6;let tmp_7 = values[1]; values[1] = values[2]; values[2] = tmp_7; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_8 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_8;let tmp_9 = values[0]; values[0] = values[1]; values[1] = tmp_9; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_10 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_10;let tmp_11 = values[2]; values[2] = values[3]; values[3] = tmp_11; }
    }
    // exch_intxn(tmask:1,swbit:0,wpt:4)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_12 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_13 = seg_base + (local_tid ^ 1u); let tmp_14 = smem_keys[tmp_13 * WPT + 3u]; let tmp_15 = smem_vals[tmp_13 * WPT + 3u]; let tmp_16 = keys[0] < tmp_14 || (keys[0] == tmp_14 && values[0] < tmp_15); if tmp_12 == tmp_16 { keys[0] = tmp_14; values[0] = tmp_15; } let tmp_17 = smem_keys[tmp_13 * WPT + 2u]; let tmp_18 = smem_vals[tmp_13 * WPT + 2u]; let tmp_19 = keys[1] < tmp_17 || (keys[1] == tmp_17 && values[1] < tmp_18); if tmp_12 == tmp_19 { keys[1] = tmp_17; values[1] = tmp_18; } let tmp_20 = smem_keys[tmp_13 * WPT + 1u]; let tmp_21 = smem_vals[tmp_13 * WPT + 1u]; let tmp_22 = keys[2] < tmp_20 || (keys[2] == tmp_20 && values[2] < tmp_21); if tmp_12 == tmp_22 { keys[2] = tmp_20; values[2] = tmp_21; } let tmp_23 = smem_keys[tmp_13 * WPT + 0u]; let tmp_24 = smem_vals[tmp_13 * WPT + 0u]; let tmp_25 = keys[3] < tmp_23 || (keys[3] == tmp_23 && values[3] < tmp_24); if tmp_12 == tmp_25 { keys[3] = tmp_23; values[3] = tmp_24; } workgroupBarrier(); }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_26 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_26;let tmp_27 = values[0]; values[0] = values[2]; values[2] = tmp_27; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_28 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_28;let tmp_29 = values[1]; values[1] = values[3]; values[3] = tmp_29; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_30 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_30;let tmp_31 = values[0]; values[0] = values[1]; values[1] = tmp_31; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_32 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_32;let tmp_33 = values[2]; values[2] = values[3]; values[3] = tmp_33; }
    }
    // exch_intxn(tmask:3,swbit:1,wpt:4)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_34 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_35 = seg_base + (local_tid ^ 3u); let tmp_36 = smem_keys[tmp_35 * WPT + 3u]; let tmp_37 = smem_vals[tmp_35 * WPT + 3u]; let tmp_38 = keys[0] < tmp_36 || (keys[0] == tmp_36 && values[0] < tmp_37); if tmp_34 == tmp_38 { keys[0] = tmp_36; values[0] = tmp_37; } let tmp_39 = smem_keys[tmp_35 * WPT + 2u]; let tmp_40 = smem_vals[tmp_35 * WPT + 2u]; let tmp_41 = keys[1] < tmp_39 || (keys[1] == tmp_39 && values[1] < tmp_40); if tmp_34 == tmp_41 { keys[1] = tmp_39; values[1] = tmp_40; } let tmp_42 = smem_keys[tmp_35 * WPT + 1u]; let tmp_43 = smem_vals[tmp_35 * WPT + 1u]; let tmp_44 = keys[2] < tmp_42 || (keys[2] == tmp_42 && values[2] < tmp_43); if tmp_34 == tmp_44 { keys[2] = tmp_42; values[2] = tmp_43; } let tmp_45 = smem_keys[tmp_35 * WPT + 0u]; let tmp_46 = smem_vals[tmp_35 * WPT + 0u]; let tmp_47 = keys[3] < tmp_45 || (keys[3] == tmp_45 && values[3] < tmp_46); if tmp_34 == tmp_47 { keys[3] = tmp_45; values[3] = tmp_46; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_48 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_49 = seg_base + (local_tid ^ 1u); let tmp_50 = smem_keys[tmp_49 * WPT + 0u]; let tmp_51 = smem_vals[tmp_49 * WPT + 0u]; let tmp_52 = keys[0] < tmp_50 || (keys[0] == tmp_50 && values[0] < tmp_51); if tmp_48 == tmp_52 { keys[0] = tmp_50; values[0] = tmp_51; } let tmp_53 = smem_keys[tmp_49 * WPT + 1u]; let tmp_54 = smem_vals[tmp_49 * WPT + 1u]; let tmp_55 = keys[1] < tmp_53 || (keys[1] == tmp_53 && values[1] < tmp_54); if tmp_48 == tmp_55 { keys[1] = tmp_53; values[1] = tmp_54; } let tmp_56 = smem_keys[tmp_49 * WPT + 2u]; let tmp_57 = smem_vals[tmp_49 * WPT + 2u]; let tmp_58 = keys[2] < tmp_56 || (keys[2] == tmp_56 && values[2] < tmp_57); if tmp_48 == tmp_58 { keys[2] = tmp_56; values[2] = tmp_57; } let tmp_59 = smem_keys[tmp_49 * WPT + 3u]; let tmp_60 = smem_vals[tmp_49 * WPT + 3u]; let tmp_61 = keys[3] < tmp_59 || (keys[3] == tmp_59 && values[3] < tmp_60); if tmp_48 == tmp_61 { keys[3] = tmp_59; values[3] = tmp_60; } workgroupBarrier(); }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_62 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_62;let tmp_63 = values[0]; values[0] = values[2]; values[2] = tmp_63; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_64 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_64;let tmp_65 = values[1]; values[1] = values[3]; values[3] = tmp_65; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_66 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_66;let tmp_67 = values[0]; values[0] = values[1]; values[1] = tmp_67; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_68 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_68;let tmp_69 = values[2]; values[2] = values[3]; values[3] = tmp_69; }
    }
    // exch_intxn(tmask:7,swbit:2,wpt:4)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_70 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_71 = seg_base + (local_tid ^ 7u); let tmp_72 = smem_keys[tmp_71 * WPT + 3u]; let tmp_73 = smem_vals[tmp_71 * WPT + 3u]; let tmp_74 = keys[0] < tmp_72 || (keys[0] == tmp_72 && values[0] < tmp_73); if tmp_70 == tmp_74 { keys[0] = tmp_72; values[0] = tmp_73; } let tmp_75 = smem_keys[tmp_71 * WPT + 2u]; let tmp_76 = smem_vals[tmp_71 * WPT + 2u]; let tmp_77 = keys[1] < tmp_75 || (keys[1] == tmp_75 && values[1] < tmp_76); if tmp_70 == tmp_77 { keys[1] = tmp_75; values[1] = tmp_76; } let tmp_78 = smem_keys[tmp_71 * WPT + 1u]; let tmp_79 = smem_vals[tmp_71 * WPT + 1u]; let tmp_80 = keys[2] < tmp_78 || (keys[2] == tmp_78 && values[2] < tmp_79); if tmp_70 == tmp_80 { keys[2] = tmp_78; values[2] = tmp_79; } let tmp_81 = smem_keys[tmp_71 * WPT + 0u]; let tmp_82 = smem_vals[tmp_71 * WPT + 0u]; let tmp_83 = keys[3] < tmp_81 || (keys[3] == tmp_81 && values[3] < tmp_82); if tmp_70 == tmp_83 { keys[3] = tmp_81; values[3] = tmp_82; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_84 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_85 = seg_base + (local_tid ^ 2u); let tmp_86 = smem_keys[tmp_85 * WPT + 0u]; let tmp_87 = smem_vals[tmp_85 * WPT + 0u]; let tmp_88 = keys[0] < tmp_86 || (keys[0] == tmp_86 && values[0] < tmp_87); if tmp_84 == tmp_88 { keys[0] = tmp_86; values[0] = tmp_87; } let tmp_89 = smem_keys[tmp_85 * WPT + 1u]; let tmp_90 = smem_vals[tmp_85 * WPT + 1u]; let tmp_91 = keys[1] < tmp_89 || (keys[1] == tmp_89 && values[1] < tmp_90); if tmp_84 == tmp_91 { keys[1] = tmp_89; values[1] = tmp_90; } let tmp_92 = smem_keys[tmp_85 * WPT + 2u]; let tmp_93 = smem_vals[tmp_85 * WPT + 2u]; let tmp_94 = keys[2] < tmp_92 || (keys[2] == tmp_92 && values[2] < tmp_93); if tmp_84 == tmp_94 { keys[2] = tmp_92; values[2] = tmp_93; } let tmp_95 = smem_keys[tmp_85 * WPT + 3u]; let tmp_96 = smem_vals[tmp_85 * WPT + 3u]; let tmp_97 = keys[3] < tmp_95 || (keys[3] == tmp_95 && values[3] < tmp_96); if tmp_84 == tmp_97 { keys[3] = tmp_95; values[3] = tmp_96; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_98 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_99 = seg_base + (local_tid ^ 1u); let tmp_100 = smem_keys[tmp_99 * WPT + 0u]; let tmp_101 = smem_vals[tmp_99 * WPT + 0u]; let tmp_102 = keys[0] < tmp_100 || (keys[0] == tmp_100 && values[0] < tmp_101); if tmp_98 == tmp_102 { keys[0] = tmp_100; values[0] = tmp_101; } let tmp_103 = smem_keys[tmp_99 * WPT + 1u]; let tmp_104 = smem_vals[tmp_99 * WPT + 1u]; let tmp_105 = keys[1] < tmp_103 || (keys[1] == tmp_103 && values[1] < tmp_104); if tmp_98 == tmp_105 { keys[1] = tmp_103; values[1] = tmp_104; } let tmp_106 = smem_keys[tmp_99 * WPT + 2u]; let tmp_107 = smem_vals[tmp_99 * WPT + 2u]; let tmp_108 = keys[2] < tmp_106 || (keys[2] == tmp_106 && values[2] < tmp_107); if tmp_98 == tmp_108 { keys[2] = tmp_106; values[2] = tmp_107; } let tmp_109 = smem_keys[tmp_99 * WPT + 3u]; let tmp_110 = smem_vals[tmp_99 * WPT + 3u]; let tmp_111 = keys[3] < tmp_109 || (keys[3] == tmp_109 && values[3] < tmp_110); if tmp_98 == tmp_111 { keys[3] = tmp_109; values[3] = tmp_110; } workgroupBarrier(); }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_112 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_112;let tmp_113 = values[0]; values[0] = values[2]; values[2] = tmp_113; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_114 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_114;let tmp_115 = values[1]; values[1] = values[3]; values[3] = tmp_115; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_116 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_116;let tmp_117 = values[0]; values[0] = values[1]; values[1] = tmp_117; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_118 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_118;let tmp_119 = values[2]; values[2] = values[3]; values[3] = tmp_119; }
    }
    // exch_intxn(tmask:15,swbit:3,wpt:4)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_120 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_121 = seg_base + (local_tid ^ 15u); let tmp_122 = smem_keys[tmp_121 * WPT + 3u]; let tmp_123 = smem_vals[tmp_121 * WPT + 3u]; let tmp_124 = keys[0] < tmp_122 || (keys[0] == tmp_122 && values[0] < tmp_123); if tmp_120 == tmp_124 { keys[0] = tmp_122; values[0] = tmp_123; } let tmp_125 = smem_keys[tmp_121 * WPT + 2u]; let tmp_126 = smem_vals[tmp_121 * WPT + 2u]; let tmp_127 = keys[1] < tmp_125 || (keys[1] == tmp_125 && values[1] < tmp_126); if tmp_120 == tmp_127 { keys[1] = tmp_125; values[1] = tmp_126; } let tmp_128 = smem_keys[tmp_121 * WPT + 1u]; let tmp_129 = smem_vals[tmp_121 * WPT + 1u]; let tmp_130 = keys[2] < tmp_128 || (keys[2] == tmp_128 && values[2] < tmp_129); if tmp_120 == tmp_130 { keys[2] = tmp_128; values[2] = tmp_129; } let tmp_131 = smem_keys[tmp_121 * WPT + 0u]; let tmp_132 = smem_vals[tmp_121 * WPT + 0u]; let tmp_133 = keys[3] < tmp_131 || (keys[3] == tmp_131 && values[3] < tmp_132); if tmp_120 == tmp_133 { keys[3] = tmp_131; values[3] = tmp_132; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_134 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_135 = seg_base + (local_tid ^ 4u); let tmp_136 = smem_keys[tmp_135 * WPT + 0u]; let tmp_137 = smem_vals[tmp_135 * WPT + 0u]; let tmp_138 = keys[0] < tmp_136 || (keys[0] == tmp_136 && values[0] < tmp_137); if tmp_134 == tmp_138 { keys[0] = tmp_136; values[0] = tmp_137; } let tmp_139 = smem_keys[tmp_135 * WPT + 1u]; let tmp_140 = smem_vals[tmp_135 * WPT + 1u]; let tmp_141 = keys[1] < tmp_139 || (keys[1] == tmp_139 && values[1] < tmp_140); if tmp_134 == tmp_141 { keys[1] = tmp_139; values[1] = tmp_140; } let tmp_142 = smem_keys[tmp_135 * WPT + 2u]; let tmp_143 = smem_vals[tmp_135 * WPT + 2u]; let tmp_144 = keys[2] < tmp_142 || (keys[2] == tmp_142 && values[2] < tmp_143); if tmp_134 == tmp_144 { keys[2] = tmp_142; values[2] = tmp_143; } let tmp_145 = smem_keys[tmp_135 * WPT + 3u]; let tmp_146 = smem_vals[tmp_135 * WPT + 3u]; let tmp_147 = keys[3] < tmp_145 || (keys[3] == tmp_145 && values[3] < tmp_146); if tmp_134 == tmp_147 { keys[3] = tmp_145; values[3] = tmp_146; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_148 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_149 = seg_base + (local_tid ^ 2u); let tmp_150 = smem_keys[tmp_149 * WPT + 0u]; let tmp_151 = smem_vals[tmp_149 * WPT + 0u]; let tmp_152 = keys[0] < tmp_150 || (keys[0] == tmp_150 && values[0] < tmp_151); if tmp_148 == tmp_152 { keys[0] = tmp_150; values[0] = tmp_151; } let tmp_153 = smem_keys[tmp_149 * WPT + 1u]; let tmp_154 = smem_vals[tmp_149 * WPT + 1u]; let tmp_155 = keys[1] < tmp_153 || (keys[1] == tmp_153 && values[1] < tmp_154); if tmp_148 == tmp_155 { keys[1] = tmp_153; values[1] = tmp_154; } let tmp_156 = smem_keys[tmp_149 * WPT + 2u]; let tmp_157 = smem_vals[tmp_149 * WPT + 2u]; let tmp_158 = keys[2] < tmp_156 || (keys[2] == tmp_156 && values[2] < tmp_157); if tmp_148 == tmp_158 { keys[2] = tmp_156; values[2] = tmp_157; } let tmp_159 = smem_keys[tmp_149 * WPT + 3u]; let tmp_160 = smem_vals[tmp_149 * WPT + 3u]; let tmp_161 = keys[3] < tmp_159 || (keys[3] == tmp_159 && values[3] < tmp_160); if tmp_148 == tmp_161 { keys[3] = tmp_159; values[3] = tmp_160; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_162 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_163 = seg_base + (local_tid ^ 1u); let tmp_164 = smem_keys[tmp_163 * WPT + 0u]; let tmp_165 = smem_vals[tmp_163 * WPT + 0u]; let tmp_166 = keys[0] < tmp_164 || (keys[0] == tmp_164 && values[0] < tmp_165); if tmp_162 == tmp_166 { keys[0] = tmp_164; values[0] = tmp_165; } let tmp_167 = smem_keys[tmp_163 * WPT + 1u]; let tmp_168 = smem_vals[tmp_163 * WPT + 1u]; let tmp_169 = keys[1] < tmp_167 || (keys[1] == tmp_167 && values[1] < tmp_168); if tmp_162 == tmp_169 { keys[1] = tmp_167; values[1] = tmp_168; } let tmp_170 = smem_keys[tmp_163 * WPT + 2u]; let tmp_171 = smem_vals[tmp_163 * WPT + 2u]; let tmp_172 = keys[2] < tmp_170 || (keys[2] == tmp_170 && values[2] < tmp_171); if tmp_162 == tmp_172 { keys[2] = tmp_170; values[2] = tmp_171; } let tmp_173 = smem_keys[tmp_163 * WPT + 3u]; let tmp_174 = smem_vals[tmp_163 * WPT + 3u]; let tmp_175 = keys[3] < tmp_173 || (keys[3] == tmp_173 && values[3] < tmp_174); if tmp_162 == tmp_175 { keys[3] = tmp_173; values[3] = tmp_174; } workgroupBarrier(); }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_176 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_176;let tmp_177 = values[0]; values[0] = values[2]; values[2] = tmp_177; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_178 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_178;let tmp_179 = values[1]; values[1] = values[3]; values[3] = tmp_179; }
    }
    // exch_local(1,4) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_180 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_180;let tmp_181 = values[0]; values[0] = values[1]; values[1] = tmp_181; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_182 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_182;let tmp_183 = values[2]; values[2] = values[3]; values[3] = tmp_183; }
    }
    // exch_intxn(tmask:31,swbit:4,wpt:4)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_184 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_185 = seg_base + (local_tid ^ 31u); let tmp_186 = smem_keys[tmp_185 * WPT + 3u]; let tmp_187 = smem_vals[tmp_185 * WPT + 3u]; let tmp_188 = keys[0] < tmp_186 || (keys[0] == tmp_186 && values[0] < tmp_187); if tmp_184 == tmp_188 { keys[0] = tmp_186; values[0] = tmp_187; } let tmp_189 = smem_keys[tmp_185 * WPT + 2u]; let tmp_190 = smem_vals[tmp_185 * WPT + 2u]; let tmp_191 = keys[1] < tmp_189 || (keys[1] == tmp_189 && values[1] < tmp_190); if tmp_184 == tmp_191 { keys[1] = tmp_189; values[1] = tmp_190; } let tmp_192 = smem_keys[tmp_185 * WPT + 1u]; let tmp_193 = smem_vals[tmp_185 * WPT + 1u]; let tmp_194 = keys[2] < tmp_192 || (keys[2] == tmp_192 && values[2] < tmp_193); if tmp_184 == tmp_194 { keys[2] = tmp_192; values[2] = tmp_193; } let tmp_195 = smem_keys[tmp_185 * WPT + 0u]; let tmp_196 = smem_vals[tmp_185 * WPT + 0u]; let tmp_197 = keys[3] < tmp_195 || (keys[3] == tmp_195 && values[3] < tmp_196); if tmp_184 == tmp_197 { keys[3] = tmp_195; values[3] = tmp_196; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_198 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_199 = seg_base + (local_tid ^ 8u); let tmp_200 = smem_keys[tmp_199 * WPT + 0u]; let tmp_201 = smem_vals[tmp_199 * WPT + 0u]; let tmp_202 = keys[0] < tmp_200 || (keys[0] == tmp_200 && values[0] < tmp_201); if tmp_198 == tmp_202 { keys[0] = tmp_200; values[0] = tmp_201; } let tmp_203 = smem_keys[tmp_199 * WPT + 1u]; let tmp_204 = smem_vals[tmp_199 * WPT + 1u]; let tmp_205 = keys[1] < tmp_203 || (keys[1] == tmp_203 && values[1] < tmp_204); if tmp_198 == tmp_205 { keys[1] = tmp_203; values[1] = tmp_204; } let tmp_206 = smem_keys[tmp_199 * WPT + 2u]; let tmp_207 = smem_vals[tmp_199 * WPT + 2u]; let tmp_208 = keys[2] < tmp_206 || (keys[2] == tmp_206 && values[2] < tmp_207); if tmp_198 == tmp_208 { keys[2] = tmp_206; values[2] = tmp_207; } let tmp_209 = smem_keys[tmp_199 * WPT + 3u]; let tmp_210 = smem_vals[tmp_199 * WPT + 3u]; let tmp_211 = keys[3] < tmp_209 || (keys[3] == tmp_209 && values[3] < tmp_210); if tmp_198 == tmp_211 { keys[3] = tmp_209; values[3] = tmp_210; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_212 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_213 = seg_base + (local_tid ^ 4u); let tmp_214 = smem_keys[tmp_213 * WPT + 0u]; let tmp_215 = smem_vals[tmp_213 * WPT + 0u]; let tmp_216 = keys[0] < tmp_214 || (keys[0] == tmp_214 && values[0] < tmp_215); if tmp_212 == tmp_216 { keys[0] = tmp_214; values[0] = tmp_215; } let tmp_217 = smem_keys[tmp_213 * WPT + 1u]; let tmp_218 = smem_vals[tmp_213 * WPT + 1u]; let tmp_219 = keys[1] < tmp_217 || (keys[1] == tmp_217 && values[1] < tmp_218); if tmp_212 == tmp_219 { keys[1] = tmp_217; values[1] = tmp_218; } let tmp_220 = smem_keys[tmp_213 * WPT + 2u]; let tmp_221 = smem_vals[tmp_213 * WPT + 2u]; let tmp_222 = keys[2] < tmp_220 || (keys[2] == tmp_220 && values[2] < tmp_221); if tmp_212 == tmp_222 { keys[2] = tmp_220; values[2] = tmp_221; } let tmp_223 = smem_keys[tmp_213 * WPT + 3u]; let tmp_224 = smem_vals[tmp_213 * WPT + 3u]; let tmp_225 = keys[3] < tmp_223 || (keys[3] == tmp_223 && values[3] < tmp_224); if tmp_212 == tmp_225 { keys[3] = tmp_223; values[3] = tmp_224; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_226 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_227 = seg_base + (local_tid ^ 2u); let tmp_228 = smem_keys[tmp_227 * WPT + 0u]; let tmp_229 = smem_vals[tmp_227 * WPT + 0u]; let tmp_230 = keys[0] < tmp_228 || (keys[0] == tmp_228 && values[0] < tmp_229); if tmp_226 == tmp_230 { keys[0] = tmp_228; values[0] = tmp_229; } let tmp_231 = smem_keys[tmp_227 * WPT + 1u]; let tmp_232 = smem_vals[tmp_227 * WPT + 1u]; let tmp_233 = keys[1] < tmp_231 || (keys[1] == tmp_231 && values[1] < tmp_232); if tmp_226 == tmp_233 { keys[1] = tmp_231; values[1] = tmp_232; } let tmp_234 = smem_keys[tmp_227 * WPT + 2u]; let tmp_235 = smem_vals[tmp_227 * WPT + 2u]; let tmp_236 = keys[2] < tmp_234 || (keys[2] == tmp_234 && values[2] < tmp_235); if tmp_226 == tmp_236 { keys[2] = tmp_234; values[2] = tmp_235; } let tmp_237 = smem_keys[tmp_227 * WPT + 3u]; let tmp_238 = smem_vals[tmp_227 * WPT + 3u]; let tmp_239 = keys[3] < tmp_237 || (keys[3] == tmp_237 && values[3] < tmp_238); if tmp_226 == tmp_239 { keys[3] = tmp_237; values[3] = tmp_238; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:4) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; workgroupBarrier(); let tmp_240 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_241 = seg_base + (local_tid ^ 1u); let tmp_242 = smem_keys[tmp_241 * WPT + 0u]; let tmp_243 = smem_vals[tmp_241 * WPT + 0u]; let tmp_244 = keys[0] < tmp_242 || (keys[0] == tmp_242 && values[0] < tmp_243); if tmp_240 == tmp_244 { keys[0] = tmp_242; values[0] = tmp_243; } let tmp_245 = smem_keys[tmp_241 * WPT + 1u]; let tmp_246 = smem_vals[tmp_241 * WPT + 1u]; let tmp_247 = keys[1] < tmp_245 || (keys[1] == tmp_245 && values[1] < tmp_246); if tmp_240 == tmp_247 { keys[1] = tmp_245; values[1] = tmp_246; } let tmp_248 = smem_keys[tmp_241 * WPT + 2u]; let tmp_249 = smem_vals[tmp_241 * WPT + 2u]; let tmp_250 = keys[2] < tmp_248 || (keys[2] == tmp_248 && values[2] < tmp_249); if tmp_240 == tmp_250 { keys[2] = tmp_248; values[2] = tmp_249; } let tmp_251 = smem_keys[tmp_241 * WPT + 3u]; let tmp_252 = smem_vals[tmp_241 * WPT + 3u]; let tmp_253 = keys[3] < tmp_251 || (keys[3] == tmp_251 && values[3] < tmp_252); if tmp_240 == tmp_253 { keys[3] = tmp_251; values[3] = tmp_252; } workgroupBarrier(); }
    // exch_local(2,4) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_254 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_254;let tmp_255 = values[0]; values[0] = values[2]; values[2] = tmp_255; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_256 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_256;let tmp_257 = values[1]; values[1] = values[3]; values[3] = tmp_257; }
    }
    // exch_local(1,4) 
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
