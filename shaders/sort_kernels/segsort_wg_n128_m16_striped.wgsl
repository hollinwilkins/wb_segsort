
override WG: u32 = 16u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 128u;
const M: u32 = 16u;
const WPT: u32 = 8u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n128_m16_striped(
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
    // exch_local(3,8) 
    // cmp_swap(0,3)
    if keys[0] > keys[3] || (keys[0] == keys[3] && values[0] > values[3]) {
    // swap(0,3) 
    { let tmp_8 = keys[0]; keys[0] = keys[3]; keys[3] = tmp_8;let tmp_9 = values[0]; values[0] = values[3]; values[3] = tmp_9; }
    }
    // cmp_swap(1,2)
    if keys[1] > keys[2] || (keys[1] == keys[2] && values[1] > values[2]) {
    // swap(1,2) 
    { let tmp_10 = keys[1]; keys[1] = keys[2]; keys[2] = tmp_10;let tmp_11 = values[1]; values[1] = values[2]; values[2] = tmp_11; }
    }
    // cmp_swap(4,7)
    if keys[4] > keys[7] || (keys[4] == keys[7] && values[4] > values[7]) {
    // swap(4,7) 
    { let tmp_12 = keys[4]; keys[4] = keys[7]; keys[7] = tmp_12;let tmp_13 = values[4]; values[4] = values[7]; values[7] = tmp_13; }
    }
    // cmp_swap(5,6)
    if keys[5] > keys[6] || (keys[5] == keys[6] && values[5] > values[6]) {
    // swap(5,6) 
    { let tmp_14 = keys[5]; keys[5] = keys[6]; keys[6] = tmp_14;let tmp_15 = values[5]; values[5] = values[6]; values[6] = tmp_15; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_16 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_16;let tmp_17 = values[0]; values[0] = values[1]; values[1] = tmp_17; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_18 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_18;let tmp_19 = values[2]; values[2] = values[3]; values[3] = tmp_19; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_20 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_20;let tmp_21 = values[4]; values[4] = values[5]; values[5] = tmp_21; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_22 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_22;let tmp_23 = values[6]; values[6] = values[7]; values[7] = tmp_23; }
    }
    // exch_local(7,8) 
    // cmp_swap(0,7)
    if keys[0] > keys[7] || (keys[0] == keys[7] && values[0] > values[7]) {
    // swap(0,7) 
    { let tmp_24 = keys[0]; keys[0] = keys[7]; keys[7] = tmp_24;let tmp_25 = values[0]; values[0] = values[7]; values[7] = tmp_25; }
    }
    // cmp_swap(1,6)
    if keys[1] > keys[6] || (keys[1] == keys[6] && values[1] > values[6]) {
    // swap(1,6) 
    { let tmp_26 = keys[1]; keys[1] = keys[6]; keys[6] = tmp_26;let tmp_27 = values[1]; values[1] = values[6]; values[6] = tmp_27; }
    }
    // cmp_swap(2,5)
    if keys[2] > keys[5] || (keys[2] == keys[5] && values[2] > values[5]) {
    // swap(2,5) 
    { let tmp_28 = keys[2]; keys[2] = keys[5]; keys[5] = tmp_28;let tmp_29 = values[2]; values[2] = values[5]; values[5] = tmp_29; }
    }
    // cmp_swap(3,4)
    if keys[3] > keys[4] || (keys[3] == keys[4] && values[3] > values[4]) {
    // swap(3,4) 
    { let tmp_30 = keys[3]; keys[3] = keys[4]; keys[4] = tmp_30;let tmp_31 = values[3]; values[3] = values[4]; values[4] = tmp_31; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_32 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_32;let tmp_33 = values[0]; values[0] = values[2]; values[2] = tmp_33; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_34 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_34;let tmp_35 = values[1]; values[1] = values[3]; values[3] = tmp_35; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_36 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_36;let tmp_37 = values[4]; values[4] = values[6]; values[6] = tmp_37; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_38 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_38;let tmp_39 = values[5]; values[5] = values[7]; values[7] = tmp_39; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_40 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_40;let tmp_41 = values[0]; values[0] = values[1]; values[1] = tmp_41; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_42 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_42;let tmp_43 = values[2]; values[2] = values[3]; values[3] = tmp_43; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_44 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_44;let tmp_45 = values[4]; values[4] = values[5]; values[5] = tmp_45; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_46 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_46;let tmp_47 = values[6]; values[6] = values[7]; values[7] = tmp_47; }
    }
    // exch_intxn(tmask:1,swbit:0,wpt:8)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_48 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_49 = seg_base + (local_tid ^ 1u); let tmp_50 = smem_keys[tmp_49 * WPT + 7u]; let tmp_51 = smem_vals[tmp_49 * WPT + 7u]; let tmp_52 = keys[0] < tmp_50 || (keys[0] == tmp_50 && values[0] < tmp_51); if tmp_48 == tmp_52 { keys[0] = tmp_50; values[0] = tmp_51; } let tmp_53 = smem_keys[tmp_49 * WPT + 6u]; let tmp_54 = smem_vals[tmp_49 * WPT + 6u]; let tmp_55 = keys[1] < tmp_53 || (keys[1] == tmp_53 && values[1] < tmp_54); if tmp_48 == tmp_55 { keys[1] = tmp_53; values[1] = tmp_54; } let tmp_56 = smem_keys[tmp_49 * WPT + 5u]; let tmp_57 = smem_vals[tmp_49 * WPT + 5u]; let tmp_58 = keys[2] < tmp_56 || (keys[2] == tmp_56 && values[2] < tmp_57); if tmp_48 == tmp_58 { keys[2] = tmp_56; values[2] = tmp_57; } let tmp_59 = smem_keys[tmp_49 * WPT + 4u]; let tmp_60 = smem_vals[tmp_49 * WPT + 4u]; let tmp_61 = keys[3] < tmp_59 || (keys[3] == tmp_59 && values[3] < tmp_60); if tmp_48 == tmp_61 { keys[3] = tmp_59; values[3] = tmp_60; } let tmp_62 = smem_keys[tmp_49 * WPT + 3u]; let tmp_63 = smem_vals[tmp_49 * WPT + 3u]; let tmp_64 = keys[4] < tmp_62 || (keys[4] == tmp_62 && values[4] < tmp_63); if tmp_48 == tmp_64 { keys[4] = tmp_62; values[4] = tmp_63; } let tmp_65 = smem_keys[tmp_49 * WPT + 2u]; let tmp_66 = smem_vals[tmp_49 * WPT + 2u]; let tmp_67 = keys[5] < tmp_65 || (keys[5] == tmp_65 && values[5] < tmp_66); if tmp_48 == tmp_67 { keys[5] = tmp_65; values[5] = tmp_66; } let tmp_68 = smem_keys[tmp_49 * WPT + 1u]; let tmp_69 = smem_vals[tmp_49 * WPT + 1u]; let tmp_70 = keys[6] < tmp_68 || (keys[6] == tmp_68 && values[6] < tmp_69); if tmp_48 == tmp_70 { keys[6] = tmp_68; values[6] = tmp_69; } let tmp_71 = smem_keys[tmp_49 * WPT + 0u]; let tmp_72 = smem_vals[tmp_49 * WPT + 0u]; let tmp_73 = keys[7] < tmp_71 || (keys[7] == tmp_71 && values[7] < tmp_72); if tmp_48 == tmp_73 { keys[7] = tmp_71; values[7] = tmp_72; } workgroupBarrier(); }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_74 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_74;let tmp_75 = values[0]; values[0] = values[4]; values[4] = tmp_75; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_76 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_76;let tmp_77 = values[1]; values[1] = values[5]; values[5] = tmp_77; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_78 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_78;let tmp_79 = values[2]; values[2] = values[6]; values[6] = tmp_79; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_80 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_80;let tmp_81 = values[3]; values[3] = values[7]; values[7] = tmp_81; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_82 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_82;let tmp_83 = values[0]; values[0] = values[2]; values[2] = tmp_83; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_84 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_84;let tmp_85 = values[1]; values[1] = values[3]; values[3] = tmp_85; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_86 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_86;let tmp_87 = values[4]; values[4] = values[6]; values[6] = tmp_87; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_88 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_88;let tmp_89 = values[5]; values[5] = values[7]; values[7] = tmp_89; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_90 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_90;let tmp_91 = values[0]; values[0] = values[1]; values[1] = tmp_91; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_92 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_92;let tmp_93 = values[2]; values[2] = values[3]; values[3] = tmp_93; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_94 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_94;let tmp_95 = values[4]; values[4] = values[5]; values[5] = tmp_95; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_96 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_96;let tmp_97 = values[6]; values[6] = values[7]; values[7] = tmp_97; }
    }
    // exch_intxn(tmask:3,swbit:1,wpt:8)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_98 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_99 = seg_base + (local_tid ^ 3u); let tmp_100 = smem_keys[tmp_99 * WPT + 7u]; let tmp_101 = smem_vals[tmp_99 * WPT + 7u]; let tmp_102 = keys[0] < tmp_100 || (keys[0] == tmp_100 && values[0] < tmp_101); if tmp_98 == tmp_102 { keys[0] = tmp_100; values[0] = tmp_101; } let tmp_103 = smem_keys[tmp_99 * WPT + 6u]; let tmp_104 = smem_vals[tmp_99 * WPT + 6u]; let tmp_105 = keys[1] < tmp_103 || (keys[1] == tmp_103 && values[1] < tmp_104); if tmp_98 == tmp_105 { keys[1] = tmp_103; values[1] = tmp_104; } let tmp_106 = smem_keys[tmp_99 * WPT + 5u]; let tmp_107 = smem_vals[tmp_99 * WPT + 5u]; let tmp_108 = keys[2] < tmp_106 || (keys[2] == tmp_106 && values[2] < tmp_107); if tmp_98 == tmp_108 { keys[2] = tmp_106; values[2] = tmp_107; } let tmp_109 = smem_keys[tmp_99 * WPT + 4u]; let tmp_110 = smem_vals[tmp_99 * WPT + 4u]; let tmp_111 = keys[3] < tmp_109 || (keys[3] == tmp_109 && values[3] < tmp_110); if tmp_98 == tmp_111 { keys[3] = tmp_109; values[3] = tmp_110; } let tmp_112 = smem_keys[tmp_99 * WPT + 3u]; let tmp_113 = smem_vals[tmp_99 * WPT + 3u]; let tmp_114 = keys[4] < tmp_112 || (keys[4] == tmp_112 && values[4] < tmp_113); if tmp_98 == tmp_114 { keys[4] = tmp_112; values[4] = tmp_113; } let tmp_115 = smem_keys[tmp_99 * WPT + 2u]; let tmp_116 = smem_vals[tmp_99 * WPT + 2u]; let tmp_117 = keys[5] < tmp_115 || (keys[5] == tmp_115 && values[5] < tmp_116); if tmp_98 == tmp_117 { keys[5] = tmp_115; values[5] = tmp_116; } let tmp_118 = smem_keys[tmp_99 * WPT + 1u]; let tmp_119 = smem_vals[tmp_99 * WPT + 1u]; let tmp_120 = keys[6] < tmp_118 || (keys[6] == tmp_118 && values[6] < tmp_119); if tmp_98 == tmp_120 { keys[6] = tmp_118; values[6] = tmp_119; } let tmp_121 = smem_keys[tmp_99 * WPT + 0u]; let tmp_122 = smem_vals[tmp_99 * WPT + 0u]; let tmp_123 = keys[7] < tmp_121 || (keys[7] == tmp_121 && values[7] < tmp_122); if tmp_98 == tmp_123 { keys[7] = tmp_121; values[7] = tmp_122; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_124 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_125 = seg_base + (local_tid ^ 1u); let tmp_126 = smem_keys[tmp_125 * WPT + 0u]; let tmp_127 = smem_vals[tmp_125 * WPT + 0u]; let tmp_128 = keys[0] < tmp_126 || (keys[0] == tmp_126 && values[0] < tmp_127); if tmp_124 == tmp_128 { keys[0] = tmp_126; values[0] = tmp_127; } let tmp_129 = smem_keys[tmp_125 * WPT + 1u]; let tmp_130 = smem_vals[tmp_125 * WPT + 1u]; let tmp_131 = keys[1] < tmp_129 || (keys[1] == tmp_129 && values[1] < tmp_130); if tmp_124 == tmp_131 { keys[1] = tmp_129; values[1] = tmp_130; } let tmp_132 = smem_keys[tmp_125 * WPT + 2u]; let tmp_133 = smem_vals[tmp_125 * WPT + 2u]; let tmp_134 = keys[2] < tmp_132 || (keys[2] == tmp_132 && values[2] < tmp_133); if tmp_124 == tmp_134 { keys[2] = tmp_132; values[2] = tmp_133; } let tmp_135 = smem_keys[tmp_125 * WPT + 3u]; let tmp_136 = smem_vals[tmp_125 * WPT + 3u]; let tmp_137 = keys[3] < tmp_135 || (keys[3] == tmp_135 && values[3] < tmp_136); if tmp_124 == tmp_137 { keys[3] = tmp_135; values[3] = tmp_136; } let tmp_138 = smem_keys[tmp_125 * WPT + 4u]; let tmp_139 = smem_vals[tmp_125 * WPT + 4u]; let tmp_140 = keys[4] < tmp_138 || (keys[4] == tmp_138 && values[4] < tmp_139); if tmp_124 == tmp_140 { keys[4] = tmp_138; values[4] = tmp_139; } let tmp_141 = smem_keys[tmp_125 * WPT + 5u]; let tmp_142 = smem_vals[tmp_125 * WPT + 5u]; let tmp_143 = keys[5] < tmp_141 || (keys[5] == tmp_141 && values[5] < tmp_142); if tmp_124 == tmp_143 { keys[5] = tmp_141; values[5] = tmp_142; } let tmp_144 = smem_keys[tmp_125 * WPT + 6u]; let tmp_145 = smem_vals[tmp_125 * WPT + 6u]; let tmp_146 = keys[6] < tmp_144 || (keys[6] == tmp_144 && values[6] < tmp_145); if tmp_124 == tmp_146 { keys[6] = tmp_144; values[6] = tmp_145; } let tmp_147 = smem_keys[tmp_125 * WPT + 7u]; let tmp_148 = smem_vals[tmp_125 * WPT + 7u]; let tmp_149 = keys[7] < tmp_147 || (keys[7] == tmp_147 && values[7] < tmp_148); if tmp_124 == tmp_149 { keys[7] = tmp_147; values[7] = tmp_148; } workgroupBarrier(); }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_150 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_150;let tmp_151 = values[0]; values[0] = values[4]; values[4] = tmp_151; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_152 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_152;let tmp_153 = values[1]; values[1] = values[5]; values[5] = tmp_153; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_154 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_154;let tmp_155 = values[2]; values[2] = values[6]; values[6] = tmp_155; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_156 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_156;let tmp_157 = values[3]; values[3] = values[7]; values[7] = tmp_157; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_158 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_158;let tmp_159 = values[0]; values[0] = values[2]; values[2] = tmp_159; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_160 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_160;let tmp_161 = values[1]; values[1] = values[3]; values[3] = tmp_161; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_162 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_162;let tmp_163 = values[4]; values[4] = values[6]; values[6] = tmp_163; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_164 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_164;let tmp_165 = values[5]; values[5] = values[7]; values[7] = tmp_165; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_166 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_166;let tmp_167 = values[0]; values[0] = values[1]; values[1] = tmp_167; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_168 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_168;let tmp_169 = values[2]; values[2] = values[3]; values[3] = tmp_169; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_170 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_170;let tmp_171 = values[4]; values[4] = values[5]; values[5] = tmp_171; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_172 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_172;let tmp_173 = values[6]; values[6] = values[7]; values[7] = tmp_173; }
    }
    // exch_intxn(tmask:7,swbit:2,wpt:8)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_174 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_175 = seg_base + (local_tid ^ 7u); let tmp_176 = smem_keys[tmp_175 * WPT + 7u]; let tmp_177 = smem_vals[tmp_175 * WPT + 7u]; let tmp_178 = keys[0] < tmp_176 || (keys[0] == tmp_176 && values[0] < tmp_177); if tmp_174 == tmp_178 { keys[0] = tmp_176; values[0] = tmp_177; } let tmp_179 = smem_keys[tmp_175 * WPT + 6u]; let tmp_180 = smem_vals[tmp_175 * WPT + 6u]; let tmp_181 = keys[1] < tmp_179 || (keys[1] == tmp_179 && values[1] < tmp_180); if tmp_174 == tmp_181 { keys[1] = tmp_179; values[1] = tmp_180; } let tmp_182 = smem_keys[tmp_175 * WPT + 5u]; let tmp_183 = smem_vals[tmp_175 * WPT + 5u]; let tmp_184 = keys[2] < tmp_182 || (keys[2] == tmp_182 && values[2] < tmp_183); if tmp_174 == tmp_184 { keys[2] = tmp_182; values[2] = tmp_183; } let tmp_185 = smem_keys[tmp_175 * WPT + 4u]; let tmp_186 = smem_vals[tmp_175 * WPT + 4u]; let tmp_187 = keys[3] < tmp_185 || (keys[3] == tmp_185 && values[3] < tmp_186); if tmp_174 == tmp_187 { keys[3] = tmp_185; values[3] = tmp_186; } let tmp_188 = smem_keys[tmp_175 * WPT + 3u]; let tmp_189 = smem_vals[tmp_175 * WPT + 3u]; let tmp_190 = keys[4] < tmp_188 || (keys[4] == tmp_188 && values[4] < tmp_189); if tmp_174 == tmp_190 { keys[4] = tmp_188; values[4] = tmp_189; } let tmp_191 = smem_keys[tmp_175 * WPT + 2u]; let tmp_192 = smem_vals[tmp_175 * WPT + 2u]; let tmp_193 = keys[5] < tmp_191 || (keys[5] == tmp_191 && values[5] < tmp_192); if tmp_174 == tmp_193 { keys[5] = tmp_191; values[5] = tmp_192; } let tmp_194 = smem_keys[tmp_175 * WPT + 1u]; let tmp_195 = smem_vals[tmp_175 * WPT + 1u]; let tmp_196 = keys[6] < tmp_194 || (keys[6] == tmp_194 && values[6] < tmp_195); if tmp_174 == tmp_196 { keys[6] = tmp_194; values[6] = tmp_195; } let tmp_197 = smem_keys[tmp_175 * WPT + 0u]; let tmp_198 = smem_vals[tmp_175 * WPT + 0u]; let tmp_199 = keys[7] < tmp_197 || (keys[7] == tmp_197 && values[7] < tmp_198); if tmp_174 == tmp_199 { keys[7] = tmp_197; values[7] = tmp_198; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_200 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_201 = seg_base + (local_tid ^ 2u); let tmp_202 = smem_keys[tmp_201 * WPT + 0u]; let tmp_203 = smem_vals[tmp_201 * WPT + 0u]; let tmp_204 = keys[0] < tmp_202 || (keys[0] == tmp_202 && values[0] < tmp_203); if tmp_200 == tmp_204 { keys[0] = tmp_202; values[0] = tmp_203; } let tmp_205 = smem_keys[tmp_201 * WPT + 1u]; let tmp_206 = smem_vals[tmp_201 * WPT + 1u]; let tmp_207 = keys[1] < tmp_205 || (keys[1] == tmp_205 && values[1] < tmp_206); if tmp_200 == tmp_207 { keys[1] = tmp_205; values[1] = tmp_206; } let tmp_208 = smem_keys[tmp_201 * WPT + 2u]; let tmp_209 = smem_vals[tmp_201 * WPT + 2u]; let tmp_210 = keys[2] < tmp_208 || (keys[2] == tmp_208 && values[2] < tmp_209); if tmp_200 == tmp_210 { keys[2] = tmp_208; values[2] = tmp_209; } let tmp_211 = smem_keys[tmp_201 * WPT + 3u]; let tmp_212 = smem_vals[tmp_201 * WPT + 3u]; let tmp_213 = keys[3] < tmp_211 || (keys[3] == tmp_211 && values[3] < tmp_212); if tmp_200 == tmp_213 { keys[3] = tmp_211; values[3] = tmp_212; } let tmp_214 = smem_keys[tmp_201 * WPT + 4u]; let tmp_215 = smem_vals[tmp_201 * WPT + 4u]; let tmp_216 = keys[4] < tmp_214 || (keys[4] == tmp_214 && values[4] < tmp_215); if tmp_200 == tmp_216 { keys[4] = tmp_214; values[4] = tmp_215; } let tmp_217 = smem_keys[tmp_201 * WPT + 5u]; let tmp_218 = smem_vals[tmp_201 * WPT + 5u]; let tmp_219 = keys[5] < tmp_217 || (keys[5] == tmp_217 && values[5] < tmp_218); if tmp_200 == tmp_219 { keys[5] = tmp_217; values[5] = tmp_218; } let tmp_220 = smem_keys[tmp_201 * WPT + 6u]; let tmp_221 = smem_vals[tmp_201 * WPT + 6u]; let tmp_222 = keys[6] < tmp_220 || (keys[6] == tmp_220 && values[6] < tmp_221); if tmp_200 == tmp_222 { keys[6] = tmp_220; values[6] = tmp_221; } let tmp_223 = smem_keys[tmp_201 * WPT + 7u]; let tmp_224 = smem_vals[tmp_201 * WPT + 7u]; let tmp_225 = keys[7] < tmp_223 || (keys[7] == tmp_223 && values[7] < tmp_224); if tmp_200 == tmp_225 { keys[7] = tmp_223; values[7] = tmp_224; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_226 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_227 = seg_base + (local_tid ^ 1u); let tmp_228 = smem_keys[tmp_227 * WPT + 0u]; let tmp_229 = smem_vals[tmp_227 * WPT + 0u]; let tmp_230 = keys[0] < tmp_228 || (keys[0] == tmp_228 && values[0] < tmp_229); if tmp_226 == tmp_230 { keys[0] = tmp_228; values[0] = tmp_229; } let tmp_231 = smem_keys[tmp_227 * WPT + 1u]; let tmp_232 = smem_vals[tmp_227 * WPT + 1u]; let tmp_233 = keys[1] < tmp_231 || (keys[1] == tmp_231 && values[1] < tmp_232); if tmp_226 == tmp_233 { keys[1] = tmp_231; values[1] = tmp_232; } let tmp_234 = smem_keys[tmp_227 * WPT + 2u]; let tmp_235 = smem_vals[tmp_227 * WPT + 2u]; let tmp_236 = keys[2] < tmp_234 || (keys[2] == tmp_234 && values[2] < tmp_235); if tmp_226 == tmp_236 { keys[2] = tmp_234; values[2] = tmp_235; } let tmp_237 = smem_keys[tmp_227 * WPT + 3u]; let tmp_238 = smem_vals[tmp_227 * WPT + 3u]; let tmp_239 = keys[3] < tmp_237 || (keys[3] == tmp_237 && values[3] < tmp_238); if tmp_226 == tmp_239 { keys[3] = tmp_237; values[3] = tmp_238; } let tmp_240 = smem_keys[tmp_227 * WPT + 4u]; let tmp_241 = smem_vals[tmp_227 * WPT + 4u]; let tmp_242 = keys[4] < tmp_240 || (keys[4] == tmp_240 && values[4] < tmp_241); if tmp_226 == tmp_242 { keys[4] = tmp_240; values[4] = tmp_241; } let tmp_243 = smem_keys[tmp_227 * WPT + 5u]; let tmp_244 = smem_vals[tmp_227 * WPT + 5u]; let tmp_245 = keys[5] < tmp_243 || (keys[5] == tmp_243 && values[5] < tmp_244); if tmp_226 == tmp_245 { keys[5] = tmp_243; values[5] = tmp_244; } let tmp_246 = smem_keys[tmp_227 * WPT + 6u]; let tmp_247 = smem_vals[tmp_227 * WPT + 6u]; let tmp_248 = keys[6] < tmp_246 || (keys[6] == tmp_246 && values[6] < tmp_247); if tmp_226 == tmp_248 { keys[6] = tmp_246; values[6] = tmp_247; } let tmp_249 = smem_keys[tmp_227 * WPT + 7u]; let tmp_250 = smem_vals[tmp_227 * WPT + 7u]; let tmp_251 = keys[7] < tmp_249 || (keys[7] == tmp_249 && values[7] < tmp_250); if tmp_226 == tmp_251 { keys[7] = tmp_249; values[7] = tmp_250; } workgroupBarrier(); }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_252 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_252;let tmp_253 = values[0]; values[0] = values[4]; values[4] = tmp_253; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_254 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_254;let tmp_255 = values[1]; values[1] = values[5]; values[5] = tmp_255; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_256 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_256;let tmp_257 = values[2]; values[2] = values[6]; values[6] = tmp_257; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_258 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_258;let tmp_259 = values[3]; values[3] = values[7]; values[7] = tmp_259; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_260 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_260;let tmp_261 = values[0]; values[0] = values[2]; values[2] = tmp_261; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_262 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_262;let tmp_263 = values[1]; values[1] = values[3]; values[3] = tmp_263; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_264 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_264;let tmp_265 = values[4]; values[4] = values[6]; values[6] = tmp_265; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_266 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_266;let tmp_267 = values[5]; values[5] = values[7]; values[7] = tmp_267; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_268 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_268;let tmp_269 = values[0]; values[0] = values[1]; values[1] = tmp_269; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_270 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_270;let tmp_271 = values[2]; values[2] = values[3]; values[3] = tmp_271; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_272 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_272;let tmp_273 = values[4]; values[4] = values[5]; values[5] = tmp_273; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_274 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_274;let tmp_275 = values[6]; values[6] = values[7]; values[7] = tmp_275; }
    }
    // exch_intxn(tmask:15,swbit:3,wpt:8)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_276 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_277 = seg_base + (local_tid ^ 15u); let tmp_278 = smem_keys[tmp_277 * WPT + 7u]; let tmp_279 = smem_vals[tmp_277 * WPT + 7u]; let tmp_280 = keys[0] < tmp_278 || (keys[0] == tmp_278 && values[0] < tmp_279); if tmp_276 == tmp_280 { keys[0] = tmp_278; values[0] = tmp_279; } let tmp_281 = smem_keys[tmp_277 * WPT + 6u]; let tmp_282 = smem_vals[tmp_277 * WPT + 6u]; let tmp_283 = keys[1] < tmp_281 || (keys[1] == tmp_281 && values[1] < tmp_282); if tmp_276 == tmp_283 { keys[1] = tmp_281; values[1] = tmp_282; } let tmp_284 = smem_keys[tmp_277 * WPT + 5u]; let tmp_285 = smem_vals[tmp_277 * WPT + 5u]; let tmp_286 = keys[2] < tmp_284 || (keys[2] == tmp_284 && values[2] < tmp_285); if tmp_276 == tmp_286 { keys[2] = tmp_284; values[2] = tmp_285; } let tmp_287 = smem_keys[tmp_277 * WPT + 4u]; let tmp_288 = smem_vals[tmp_277 * WPT + 4u]; let tmp_289 = keys[3] < tmp_287 || (keys[3] == tmp_287 && values[3] < tmp_288); if tmp_276 == tmp_289 { keys[3] = tmp_287; values[3] = tmp_288; } let tmp_290 = smem_keys[tmp_277 * WPT + 3u]; let tmp_291 = smem_vals[tmp_277 * WPT + 3u]; let tmp_292 = keys[4] < tmp_290 || (keys[4] == tmp_290 && values[4] < tmp_291); if tmp_276 == tmp_292 { keys[4] = tmp_290; values[4] = tmp_291; } let tmp_293 = smem_keys[tmp_277 * WPT + 2u]; let tmp_294 = smem_vals[tmp_277 * WPT + 2u]; let tmp_295 = keys[5] < tmp_293 || (keys[5] == tmp_293 && values[5] < tmp_294); if tmp_276 == tmp_295 { keys[5] = tmp_293; values[5] = tmp_294; } let tmp_296 = smem_keys[tmp_277 * WPT + 1u]; let tmp_297 = smem_vals[tmp_277 * WPT + 1u]; let tmp_298 = keys[6] < tmp_296 || (keys[6] == tmp_296 && values[6] < tmp_297); if tmp_276 == tmp_298 { keys[6] = tmp_296; values[6] = tmp_297; } let tmp_299 = smem_keys[tmp_277 * WPT + 0u]; let tmp_300 = smem_vals[tmp_277 * WPT + 0u]; let tmp_301 = keys[7] < tmp_299 || (keys[7] == tmp_299 && values[7] < tmp_300); if tmp_276 == tmp_301 { keys[7] = tmp_299; values[7] = tmp_300; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_302 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_303 = seg_base + (local_tid ^ 4u); let tmp_304 = smem_keys[tmp_303 * WPT + 0u]; let tmp_305 = smem_vals[tmp_303 * WPT + 0u]; let tmp_306 = keys[0] < tmp_304 || (keys[0] == tmp_304 && values[0] < tmp_305); if tmp_302 == tmp_306 { keys[0] = tmp_304; values[0] = tmp_305; } let tmp_307 = smem_keys[tmp_303 * WPT + 1u]; let tmp_308 = smem_vals[tmp_303 * WPT + 1u]; let tmp_309 = keys[1] < tmp_307 || (keys[1] == tmp_307 && values[1] < tmp_308); if tmp_302 == tmp_309 { keys[1] = tmp_307; values[1] = tmp_308; } let tmp_310 = smem_keys[tmp_303 * WPT + 2u]; let tmp_311 = smem_vals[tmp_303 * WPT + 2u]; let tmp_312 = keys[2] < tmp_310 || (keys[2] == tmp_310 && values[2] < tmp_311); if tmp_302 == tmp_312 { keys[2] = tmp_310; values[2] = tmp_311; } let tmp_313 = smem_keys[tmp_303 * WPT + 3u]; let tmp_314 = smem_vals[tmp_303 * WPT + 3u]; let tmp_315 = keys[3] < tmp_313 || (keys[3] == tmp_313 && values[3] < tmp_314); if tmp_302 == tmp_315 { keys[3] = tmp_313; values[3] = tmp_314; } let tmp_316 = smem_keys[tmp_303 * WPT + 4u]; let tmp_317 = smem_vals[tmp_303 * WPT + 4u]; let tmp_318 = keys[4] < tmp_316 || (keys[4] == tmp_316 && values[4] < tmp_317); if tmp_302 == tmp_318 { keys[4] = tmp_316; values[4] = tmp_317; } let tmp_319 = smem_keys[tmp_303 * WPT + 5u]; let tmp_320 = smem_vals[tmp_303 * WPT + 5u]; let tmp_321 = keys[5] < tmp_319 || (keys[5] == tmp_319 && values[5] < tmp_320); if tmp_302 == tmp_321 { keys[5] = tmp_319; values[5] = tmp_320; } let tmp_322 = smem_keys[tmp_303 * WPT + 6u]; let tmp_323 = smem_vals[tmp_303 * WPT + 6u]; let tmp_324 = keys[6] < tmp_322 || (keys[6] == tmp_322 && values[6] < tmp_323); if tmp_302 == tmp_324 { keys[6] = tmp_322; values[6] = tmp_323; } let tmp_325 = smem_keys[tmp_303 * WPT + 7u]; let tmp_326 = smem_vals[tmp_303 * WPT + 7u]; let tmp_327 = keys[7] < tmp_325 || (keys[7] == tmp_325 && values[7] < tmp_326); if tmp_302 == tmp_327 { keys[7] = tmp_325; values[7] = tmp_326; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_328 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_329 = seg_base + (local_tid ^ 2u); let tmp_330 = smem_keys[tmp_329 * WPT + 0u]; let tmp_331 = smem_vals[tmp_329 * WPT + 0u]; let tmp_332 = keys[0] < tmp_330 || (keys[0] == tmp_330 && values[0] < tmp_331); if tmp_328 == tmp_332 { keys[0] = tmp_330; values[0] = tmp_331; } let tmp_333 = smem_keys[tmp_329 * WPT + 1u]; let tmp_334 = smem_vals[tmp_329 * WPT + 1u]; let tmp_335 = keys[1] < tmp_333 || (keys[1] == tmp_333 && values[1] < tmp_334); if tmp_328 == tmp_335 { keys[1] = tmp_333; values[1] = tmp_334; } let tmp_336 = smem_keys[tmp_329 * WPT + 2u]; let tmp_337 = smem_vals[tmp_329 * WPT + 2u]; let tmp_338 = keys[2] < tmp_336 || (keys[2] == tmp_336 && values[2] < tmp_337); if tmp_328 == tmp_338 { keys[2] = tmp_336; values[2] = tmp_337; } let tmp_339 = smem_keys[tmp_329 * WPT + 3u]; let tmp_340 = smem_vals[tmp_329 * WPT + 3u]; let tmp_341 = keys[3] < tmp_339 || (keys[3] == tmp_339 && values[3] < tmp_340); if tmp_328 == tmp_341 { keys[3] = tmp_339; values[3] = tmp_340; } let tmp_342 = smem_keys[tmp_329 * WPT + 4u]; let tmp_343 = smem_vals[tmp_329 * WPT + 4u]; let tmp_344 = keys[4] < tmp_342 || (keys[4] == tmp_342 && values[4] < tmp_343); if tmp_328 == tmp_344 { keys[4] = tmp_342; values[4] = tmp_343; } let tmp_345 = smem_keys[tmp_329 * WPT + 5u]; let tmp_346 = smem_vals[tmp_329 * WPT + 5u]; let tmp_347 = keys[5] < tmp_345 || (keys[5] == tmp_345 && values[5] < tmp_346); if tmp_328 == tmp_347 { keys[5] = tmp_345; values[5] = tmp_346; } let tmp_348 = smem_keys[tmp_329 * WPT + 6u]; let tmp_349 = smem_vals[tmp_329 * WPT + 6u]; let tmp_350 = keys[6] < tmp_348 || (keys[6] == tmp_348 && values[6] < tmp_349); if tmp_328 == tmp_350 { keys[6] = tmp_348; values[6] = tmp_349; } let tmp_351 = smem_keys[tmp_329 * WPT + 7u]; let tmp_352 = smem_vals[tmp_329 * WPT + 7u]; let tmp_353 = keys[7] < tmp_351 || (keys[7] == tmp_351 && values[7] < tmp_352); if tmp_328 == tmp_353 { keys[7] = tmp_351; values[7] = tmp_352; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_354 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_355 = seg_base + (local_tid ^ 1u); let tmp_356 = smem_keys[tmp_355 * WPT + 0u]; let tmp_357 = smem_vals[tmp_355 * WPT + 0u]; let tmp_358 = keys[0] < tmp_356 || (keys[0] == tmp_356 && values[0] < tmp_357); if tmp_354 == tmp_358 { keys[0] = tmp_356; values[0] = tmp_357; } let tmp_359 = smem_keys[tmp_355 * WPT + 1u]; let tmp_360 = smem_vals[tmp_355 * WPT + 1u]; let tmp_361 = keys[1] < tmp_359 || (keys[1] == tmp_359 && values[1] < tmp_360); if tmp_354 == tmp_361 { keys[1] = tmp_359; values[1] = tmp_360; } let tmp_362 = smem_keys[tmp_355 * WPT + 2u]; let tmp_363 = smem_vals[tmp_355 * WPT + 2u]; let tmp_364 = keys[2] < tmp_362 || (keys[2] == tmp_362 && values[2] < tmp_363); if tmp_354 == tmp_364 { keys[2] = tmp_362; values[2] = tmp_363; } let tmp_365 = smem_keys[tmp_355 * WPT + 3u]; let tmp_366 = smem_vals[tmp_355 * WPT + 3u]; let tmp_367 = keys[3] < tmp_365 || (keys[3] == tmp_365 && values[3] < tmp_366); if tmp_354 == tmp_367 { keys[3] = tmp_365; values[3] = tmp_366; } let tmp_368 = smem_keys[tmp_355 * WPT + 4u]; let tmp_369 = smem_vals[tmp_355 * WPT + 4u]; let tmp_370 = keys[4] < tmp_368 || (keys[4] == tmp_368 && values[4] < tmp_369); if tmp_354 == tmp_370 { keys[4] = tmp_368; values[4] = tmp_369; } let tmp_371 = smem_keys[tmp_355 * WPT + 5u]; let tmp_372 = smem_vals[tmp_355 * WPT + 5u]; let tmp_373 = keys[5] < tmp_371 || (keys[5] == tmp_371 && values[5] < tmp_372); if tmp_354 == tmp_373 { keys[5] = tmp_371; values[5] = tmp_372; } let tmp_374 = smem_keys[tmp_355 * WPT + 6u]; let tmp_375 = smem_vals[tmp_355 * WPT + 6u]; let tmp_376 = keys[6] < tmp_374 || (keys[6] == tmp_374 && values[6] < tmp_375); if tmp_354 == tmp_376 { keys[6] = tmp_374; values[6] = tmp_375; } let tmp_377 = smem_keys[tmp_355 * WPT + 7u]; let tmp_378 = smem_vals[tmp_355 * WPT + 7u]; let tmp_379 = keys[7] < tmp_377 || (keys[7] == tmp_377 && values[7] < tmp_378); if tmp_354 == tmp_379 { keys[7] = tmp_377; values[7] = tmp_378; } workgroupBarrier(); }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_380 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_380;let tmp_381 = values[0]; values[0] = values[4]; values[4] = tmp_381; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_382 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_382;let tmp_383 = values[1]; values[1] = values[5]; values[5] = tmp_383; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_384 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_384;let tmp_385 = values[2]; values[2] = values[6]; values[6] = tmp_385; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_386 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_386;let tmp_387 = values[3]; values[3] = values[7]; values[7] = tmp_387; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_388 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_388;let tmp_389 = values[0]; values[0] = values[2]; values[2] = tmp_389; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_390 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_390;let tmp_391 = values[1]; values[1] = values[3]; values[3] = tmp_391; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_392 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_392;let tmp_393 = values[4]; values[4] = values[6]; values[6] = tmp_393; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_394 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_394;let tmp_395 = values[5]; values[5] = values[7]; values[7] = tmp_395; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_396 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_396;let tmp_397 = values[0]; values[0] = values[1]; values[1] = tmp_397; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_398 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_398;let tmp_399 = values[2]; values[2] = values[3]; values[3] = tmp_399; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_400 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_400;let tmp_401 = values[4]; values[4] = values[5]; values[5] = tmp_401; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_402 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_402;let tmp_403 = values[6]; values[6] = values[7]; values[7] = tmp_403; }
    }

    // striped (coalesced) store via shared memory
    for (var r = 0u; r < WPT; r = r + 1u) {
        smem_keys[local_tid * WPT + r] = keys[r];
        smem_vals[local_tid * WPT + r] = values[r];
    }
    workgroupBarrier();
    for (var c = 0u; c < WPT; c = c + 1u) {
        let j = c * M + local_tid;
        if is_active && j < seg_size {
            global_keys[seg_start + j] = smem_keys[j];
            global_value_indices[seg_start + j] = smem_vals[j];
        }
    }
}
