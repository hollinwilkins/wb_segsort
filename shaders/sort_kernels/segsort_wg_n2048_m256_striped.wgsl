
override WG: u32 = 256u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 2048u;
const M: u32 = 256u;
const WPT: u32 = 8u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n2048_m256_striped(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    const BIN: u32 = 11u;

    let bin_base = select(bin_offsets[BIN - 1u], 0u, BIN == 0u);
    let bin_count = bin_offsets[BIN] - bin_base;

    let local_tid = tid_g % M;
    let seg_base = tid_g - local_tid;
    let global_seg = (wg_id.x * WG + tid_g) / M;

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
    // exch_intxn(tmask:31,swbit:4,wpt:8)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_404 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_405 = seg_base + (local_tid ^ 31u); let tmp_406 = smem_keys[tmp_405 * WPT + 7u]; let tmp_407 = smem_vals[tmp_405 * WPT + 7u]; let tmp_408 = keys[0] < tmp_406 || (keys[0] == tmp_406 && values[0] < tmp_407); if tmp_404 == tmp_408 { keys[0] = tmp_406; values[0] = tmp_407; } let tmp_409 = smem_keys[tmp_405 * WPT + 6u]; let tmp_410 = smem_vals[tmp_405 * WPT + 6u]; let tmp_411 = keys[1] < tmp_409 || (keys[1] == tmp_409 && values[1] < tmp_410); if tmp_404 == tmp_411 { keys[1] = tmp_409; values[1] = tmp_410; } let tmp_412 = smem_keys[tmp_405 * WPT + 5u]; let tmp_413 = smem_vals[tmp_405 * WPT + 5u]; let tmp_414 = keys[2] < tmp_412 || (keys[2] == tmp_412 && values[2] < tmp_413); if tmp_404 == tmp_414 { keys[2] = tmp_412; values[2] = tmp_413; } let tmp_415 = smem_keys[tmp_405 * WPT + 4u]; let tmp_416 = smem_vals[tmp_405 * WPT + 4u]; let tmp_417 = keys[3] < tmp_415 || (keys[3] == tmp_415 && values[3] < tmp_416); if tmp_404 == tmp_417 { keys[3] = tmp_415; values[3] = tmp_416; } let tmp_418 = smem_keys[tmp_405 * WPT + 3u]; let tmp_419 = smem_vals[tmp_405 * WPT + 3u]; let tmp_420 = keys[4] < tmp_418 || (keys[4] == tmp_418 && values[4] < tmp_419); if tmp_404 == tmp_420 { keys[4] = tmp_418; values[4] = tmp_419; } let tmp_421 = smem_keys[tmp_405 * WPT + 2u]; let tmp_422 = smem_vals[tmp_405 * WPT + 2u]; let tmp_423 = keys[5] < tmp_421 || (keys[5] == tmp_421 && values[5] < tmp_422); if tmp_404 == tmp_423 { keys[5] = tmp_421; values[5] = tmp_422; } let tmp_424 = smem_keys[tmp_405 * WPT + 1u]; let tmp_425 = smem_vals[tmp_405 * WPT + 1u]; let tmp_426 = keys[6] < tmp_424 || (keys[6] == tmp_424 && values[6] < tmp_425); if tmp_404 == tmp_426 { keys[6] = tmp_424; values[6] = tmp_425; } let tmp_427 = smem_keys[tmp_405 * WPT + 0u]; let tmp_428 = smem_vals[tmp_405 * WPT + 0u]; let tmp_429 = keys[7] < tmp_427 || (keys[7] == tmp_427 && values[7] < tmp_428); if tmp_404 == tmp_429 { keys[7] = tmp_427; values[7] = tmp_428; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_430 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_431 = seg_base + (local_tid ^ 8u); let tmp_432 = smem_keys[tmp_431 * WPT + 0u]; let tmp_433 = smem_vals[tmp_431 * WPT + 0u]; let tmp_434 = keys[0] < tmp_432 || (keys[0] == tmp_432 && values[0] < tmp_433); if tmp_430 == tmp_434 { keys[0] = tmp_432; values[0] = tmp_433; } let tmp_435 = smem_keys[tmp_431 * WPT + 1u]; let tmp_436 = smem_vals[tmp_431 * WPT + 1u]; let tmp_437 = keys[1] < tmp_435 || (keys[1] == tmp_435 && values[1] < tmp_436); if tmp_430 == tmp_437 { keys[1] = tmp_435; values[1] = tmp_436; } let tmp_438 = smem_keys[tmp_431 * WPT + 2u]; let tmp_439 = smem_vals[tmp_431 * WPT + 2u]; let tmp_440 = keys[2] < tmp_438 || (keys[2] == tmp_438 && values[2] < tmp_439); if tmp_430 == tmp_440 { keys[2] = tmp_438; values[2] = tmp_439; } let tmp_441 = smem_keys[tmp_431 * WPT + 3u]; let tmp_442 = smem_vals[tmp_431 * WPT + 3u]; let tmp_443 = keys[3] < tmp_441 || (keys[3] == tmp_441 && values[3] < tmp_442); if tmp_430 == tmp_443 { keys[3] = tmp_441; values[3] = tmp_442; } let tmp_444 = smem_keys[tmp_431 * WPT + 4u]; let tmp_445 = smem_vals[tmp_431 * WPT + 4u]; let tmp_446 = keys[4] < tmp_444 || (keys[4] == tmp_444 && values[4] < tmp_445); if tmp_430 == tmp_446 { keys[4] = tmp_444; values[4] = tmp_445; } let tmp_447 = smem_keys[tmp_431 * WPT + 5u]; let tmp_448 = smem_vals[tmp_431 * WPT + 5u]; let tmp_449 = keys[5] < tmp_447 || (keys[5] == tmp_447 && values[5] < tmp_448); if tmp_430 == tmp_449 { keys[5] = tmp_447; values[5] = tmp_448; } let tmp_450 = smem_keys[tmp_431 * WPT + 6u]; let tmp_451 = smem_vals[tmp_431 * WPT + 6u]; let tmp_452 = keys[6] < tmp_450 || (keys[6] == tmp_450 && values[6] < tmp_451); if tmp_430 == tmp_452 { keys[6] = tmp_450; values[6] = tmp_451; } let tmp_453 = smem_keys[tmp_431 * WPT + 7u]; let tmp_454 = smem_vals[tmp_431 * WPT + 7u]; let tmp_455 = keys[7] < tmp_453 || (keys[7] == tmp_453 && values[7] < tmp_454); if tmp_430 == tmp_455 { keys[7] = tmp_453; values[7] = tmp_454; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_456 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_457 = seg_base + (local_tid ^ 4u); let tmp_458 = smem_keys[tmp_457 * WPT + 0u]; let tmp_459 = smem_vals[tmp_457 * WPT + 0u]; let tmp_460 = keys[0] < tmp_458 || (keys[0] == tmp_458 && values[0] < tmp_459); if tmp_456 == tmp_460 { keys[0] = tmp_458; values[0] = tmp_459; } let tmp_461 = smem_keys[tmp_457 * WPT + 1u]; let tmp_462 = smem_vals[tmp_457 * WPT + 1u]; let tmp_463 = keys[1] < tmp_461 || (keys[1] == tmp_461 && values[1] < tmp_462); if tmp_456 == tmp_463 { keys[1] = tmp_461; values[1] = tmp_462; } let tmp_464 = smem_keys[tmp_457 * WPT + 2u]; let tmp_465 = smem_vals[tmp_457 * WPT + 2u]; let tmp_466 = keys[2] < tmp_464 || (keys[2] == tmp_464 && values[2] < tmp_465); if tmp_456 == tmp_466 { keys[2] = tmp_464; values[2] = tmp_465; } let tmp_467 = smem_keys[tmp_457 * WPT + 3u]; let tmp_468 = smem_vals[tmp_457 * WPT + 3u]; let tmp_469 = keys[3] < tmp_467 || (keys[3] == tmp_467 && values[3] < tmp_468); if tmp_456 == tmp_469 { keys[3] = tmp_467; values[3] = tmp_468; } let tmp_470 = smem_keys[tmp_457 * WPT + 4u]; let tmp_471 = smem_vals[tmp_457 * WPT + 4u]; let tmp_472 = keys[4] < tmp_470 || (keys[4] == tmp_470 && values[4] < tmp_471); if tmp_456 == tmp_472 { keys[4] = tmp_470; values[4] = tmp_471; } let tmp_473 = smem_keys[tmp_457 * WPT + 5u]; let tmp_474 = smem_vals[tmp_457 * WPT + 5u]; let tmp_475 = keys[5] < tmp_473 || (keys[5] == tmp_473 && values[5] < tmp_474); if tmp_456 == tmp_475 { keys[5] = tmp_473; values[5] = tmp_474; } let tmp_476 = smem_keys[tmp_457 * WPT + 6u]; let tmp_477 = smem_vals[tmp_457 * WPT + 6u]; let tmp_478 = keys[6] < tmp_476 || (keys[6] == tmp_476 && values[6] < tmp_477); if tmp_456 == tmp_478 { keys[6] = tmp_476; values[6] = tmp_477; } let tmp_479 = smem_keys[tmp_457 * WPT + 7u]; let tmp_480 = smem_vals[tmp_457 * WPT + 7u]; let tmp_481 = keys[7] < tmp_479 || (keys[7] == tmp_479 && values[7] < tmp_480); if tmp_456 == tmp_481 { keys[7] = tmp_479; values[7] = tmp_480; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_482 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_483 = seg_base + (local_tid ^ 2u); let tmp_484 = smem_keys[tmp_483 * WPT + 0u]; let tmp_485 = smem_vals[tmp_483 * WPT + 0u]; let tmp_486 = keys[0] < tmp_484 || (keys[0] == tmp_484 && values[0] < tmp_485); if tmp_482 == tmp_486 { keys[0] = tmp_484; values[0] = tmp_485; } let tmp_487 = smem_keys[tmp_483 * WPT + 1u]; let tmp_488 = smem_vals[tmp_483 * WPT + 1u]; let tmp_489 = keys[1] < tmp_487 || (keys[1] == tmp_487 && values[1] < tmp_488); if tmp_482 == tmp_489 { keys[1] = tmp_487; values[1] = tmp_488; } let tmp_490 = smem_keys[tmp_483 * WPT + 2u]; let tmp_491 = smem_vals[tmp_483 * WPT + 2u]; let tmp_492 = keys[2] < tmp_490 || (keys[2] == tmp_490 && values[2] < tmp_491); if tmp_482 == tmp_492 { keys[2] = tmp_490; values[2] = tmp_491; } let tmp_493 = smem_keys[tmp_483 * WPT + 3u]; let tmp_494 = smem_vals[tmp_483 * WPT + 3u]; let tmp_495 = keys[3] < tmp_493 || (keys[3] == tmp_493 && values[3] < tmp_494); if tmp_482 == tmp_495 { keys[3] = tmp_493; values[3] = tmp_494; } let tmp_496 = smem_keys[tmp_483 * WPT + 4u]; let tmp_497 = smem_vals[tmp_483 * WPT + 4u]; let tmp_498 = keys[4] < tmp_496 || (keys[4] == tmp_496 && values[4] < tmp_497); if tmp_482 == tmp_498 { keys[4] = tmp_496; values[4] = tmp_497; } let tmp_499 = smem_keys[tmp_483 * WPT + 5u]; let tmp_500 = smem_vals[tmp_483 * WPT + 5u]; let tmp_501 = keys[5] < tmp_499 || (keys[5] == tmp_499 && values[5] < tmp_500); if tmp_482 == tmp_501 { keys[5] = tmp_499; values[5] = tmp_500; } let tmp_502 = smem_keys[tmp_483 * WPT + 6u]; let tmp_503 = smem_vals[tmp_483 * WPT + 6u]; let tmp_504 = keys[6] < tmp_502 || (keys[6] == tmp_502 && values[6] < tmp_503); if tmp_482 == tmp_504 { keys[6] = tmp_502; values[6] = tmp_503; } let tmp_505 = smem_keys[tmp_483 * WPT + 7u]; let tmp_506 = smem_vals[tmp_483 * WPT + 7u]; let tmp_507 = keys[7] < tmp_505 || (keys[7] == tmp_505 && values[7] < tmp_506); if tmp_482 == tmp_507 { keys[7] = tmp_505; values[7] = tmp_506; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_508 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_509 = seg_base + (local_tid ^ 1u); let tmp_510 = smem_keys[tmp_509 * WPT + 0u]; let tmp_511 = smem_vals[tmp_509 * WPT + 0u]; let tmp_512 = keys[0] < tmp_510 || (keys[0] == tmp_510 && values[0] < tmp_511); if tmp_508 == tmp_512 { keys[0] = tmp_510; values[0] = tmp_511; } let tmp_513 = smem_keys[tmp_509 * WPT + 1u]; let tmp_514 = smem_vals[tmp_509 * WPT + 1u]; let tmp_515 = keys[1] < tmp_513 || (keys[1] == tmp_513 && values[1] < tmp_514); if tmp_508 == tmp_515 { keys[1] = tmp_513; values[1] = tmp_514; } let tmp_516 = smem_keys[tmp_509 * WPT + 2u]; let tmp_517 = smem_vals[tmp_509 * WPT + 2u]; let tmp_518 = keys[2] < tmp_516 || (keys[2] == tmp_516 && values[2] < tmp_517); if tmp_508 == tmp_518 { keys[2] = tmp_516; values[2] = tmp_517; } let tmp_519 = smem_keys[tmp_509 * WPT + 3u]; let tmp_520 = smem_vals[tmp_509 * WPT + 3u]; let tmp_521 = keys[3] < tmp_519 || (keys[3] == tmp_519 && values[3] < tmp_520); if tmp_508 == tmp_521 { keys[3] = tmp_519; values[3] = tmp_520; } let tmp_522 = smem_keys[tmp_509 * WPT + 4u]; let tmp_523 = smem_vals[tmp_509 * WPT + 4u]; let tmp_524 = keys[4] < tmp_522 || (keys[4] == tmp_522 && values[4] < tmp_523); if tmp_508 == tmp_524 { keys[4] = tmp_522; values[4] = tmp_523; } let tmp_525 = smem_keys[tmp_509 * WPT + 5u]; let tmp_526 = smem_vals[tmp_509 * WPT + 5u]; let tmp_527 = keys[5] < tmp_525 || (keys[5] == tmp_525 && values[5] < tmp_526); if tmp_508 == tmp_527 { keys[5] = tmp_525; values[5] = tmp_526; } let tmp_528 = smem_keys[tmp_509 * WPT + 6u]; let tmp_529 = smem_vals[tmp_509 * WPT + 6u]; let tmp_530 = keys[6] < tmp_528 || (keys[6] == tmp_528 && values[6] < tmp_529); if tmp_508 == tmp_530 { keys[6] = tmp_528; values[6] = tmp_529; } let tmp_531 = smem_keys[tmp_509 * WPT + 7u]; let tmp_532 = smem_vals[tmp_509 * WPT + 7u]; let tmp_533 = keys[7] < tmp_531 || (keys[7] == tmp_531 && values[7] < tmp_532); if tmp_508 == tmp_533 { keys[7] = tmp_531; values[7] = tmp_532; } workgroupBarrier(); }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_534 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_534;let tmp_535 = values[0]; values[0] = values[4]; values[4] = tmp_535; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_536 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_536;let tmp_537 = values[1]; values[1] = values[5]; values[5] = tmp_537; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_538 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_538;let tmp_539 = values[2]; values[2] = values[6]; values[6] = tmp_539; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_540 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_540;let tmp_541 = values[3]; values[3] = values[7]; values[7] = tmp_541; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_542 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_542;let tmp_543 = values[0]; values[0] = values[2]; values[2] = tmp_543; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_544 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_544;let tmp_545 = values[1]; values[1] = values[3]; values[3] = tmp_545; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_546 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_546;let tmp_547 = values[4]; values[4] = values[6]; values[6] = tmp_547; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_548 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_548;let tmp_549 = values[5]; values[5] = values[7]; values[7] = tmp_549; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_550 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_550;let tmp_551 = values[0]; values[0] = values[1]; values[1] = tmp_551; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_552 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_552;let tmp_553 = values[2]; values[2] = values[3]; values[3] = tmp_553; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_554 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_554;let tmp_555 = values[4]; values[4] = values[5]; values[5] = tmp_555; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_556 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_556;let tmp_557 = values[6]; values[6] = values[7]; values[7] = tmp_557; }
    }
    // exch_intxn(tmask:63,swbit:5,wpt:8)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_558 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_559 = seg_base + (local_tid ^ 63u); let tmp_560 = smem_keys[tmp_559 * WPT + 7u]; let tmp_561 = smem_vals[tmp_559 * WPT + 7u]; let tmp_562 = keys[0] < tmp_560 || (keys[0] == tmp_560 && values[0] < tmp_561); if tmp_558 == tmp_562 { keys[0] = tmp_560; values[0] = tmp_561; } let tmp_563 = smem_keys[tmp_559 * WPT + 6u]; let tmp_564 = smem_vals[tmp_559 * WPT + 6u]; let tmp_565 = keys[1] < tmp_563 || (keys[1] == tmp_563 && values[1] < tmp_564); if tmp_558 == tmp_565 { keys[1] = tmp_563; values[1] = tmp_564; } let tmp_566 = smem_keys[tmp_559 * WPT + 5u]; let tmp_567 = smem_vals[tmp_559 * WPT + 5u]; let tmp_568 = keys[2] < tmp_566 || (keys[2] == tmp_566 && values[2] < tmp_567); if tmp_558 == tmp_568 { keys[2] = tmp_566; values[2] = tmp_567; } let tmp_569 = smem_keys[tmp_559 * WPT + 4u]; let tmp_570 = smem_vals[tmp_559 * WPT + 4u]; let tmp_571 = keys[3] < tmp_569 || (keys[3] == tmp_569 && values[3] < tmp_570); if tmp_558 == tmp_571 { keys[3] = tmp_569; values[3] = tmp_570; } let tmp_572 = smem_keys[tmp_559 * WPT + 3u]; let tmp_573 = smem_vals[tmp_559 * WPT + 3u]; let tmp_574 = keys[4] < tmp_572 || (keys[4] == tmp_572 && values[4] < tmp_573); if tmp_558 == tmp_574 { keys[4] = tmp_572; values[4] = tmp_573; } let tmp_575 = smem_keys[tmp_559 * WPT + 2u]; let tmp_576 = smem_vals[tmp_559 * WPT + 2u]; let tmp_577 = keys[5] < tmp_575 || (keys[5] == tmp_575 && values[5] < tmp_576); if tmp_558 == tmp_577 { keys[5] = tmp_575; values[5] = tmp_576; } let tmp_578 = smem_keys[tmp_559 * WPT + 1u]; let tmp_579 = smem_vals[tmp_559 * WPT + 1u]; let tmp_580 = keys[6] < tmp_578 || (keys[6] == tmp_578 && values[6] < tmp_579); if tmp_558 == tmp_580 { keys[6] = tmp_578; values[6] = tmp_579; } let tmp_581 = smem_keys[tmp_559 * WPT + 0u]; let tmp_582 = smem_vals[tmp_559 * WPT + 0u]; let tmp_583 = keys[7] < tmp_581 || (keys[7] == tmp_581 && values[7] < tmp_582); if tmp_558 == tmp_583 { keys[7] = tmp_581; values[7] = tmp_582; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_584 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_585 = seg_base + (local_tid ^ 16u); let tmp_586 = smem_keys[tmp_585 * WPT + 0u]; let tmp_587 = smem_vals[tmp_585 * WPT + 0u]; let tmp_588 = keys[0] < tmp_586 || (keys[0] == tmp_586 && values[0] < tmp_587); if tmp_584 == tmp_588 { keys[0] = tmp_586; values[0] = tmp_587; } let tmp_589 = smem_keys[tmp_585 * WPT + 1u]; let tmp_590 = smem_vals[tmp_585 * WPT + 1u]; let tmp_591 = keys[1] < tmp_589 || (keys[1] == tmp_589 && values[1] < tmp_590); if tmp_584 == tmp_591 { keys[1] = tmp_589; values[1] = tmp_590; } let tmp_592 = smem_keys[tmp_585 * WPT + 2u]; let tmp_593 = smem_vals[tmp_585 * WPT + 2u]; let tmp_594 = keys[2] < tmp_592 || (keys[2] == tmp_592 && values[2] < tmp_593); if tmp_584 == tmp_594 { keys[2] = tmp_592; values[2] = tmp_593; } let tmp_595 = smem_keys[tmp_585 * WPT + 3u]; let tmp_596 = smem_vals[tmp_585 * WPT + 3u]; let tmp_597 = keys[3] < tmp_595 || (keys[3] == tmp_595 && values[3] < tmp_596); if tmp_584 == tmp_597 { keys[3] = tmp_595; values[3] = tmp_596; } let tmp_598 = smem_keys[tmp_585 * WPT + 4u]; let tmp_599 = smem_vals[tmp_585 * WPT + 4u]; let tmp_600 = keys[4] < tmp_598 || (keys[4] == tmp_598 && values[4] < tmp_599); if tmp_584 == tmp_600 { keys[4] = tmp_598; values[4] = tmp_599; } let tmp_601 = smem_keys[tmp_585 * WPT + 5u]; let tmp_602 = smem_vals[tmp_585 * WPT + 5u]; let tmp_603 = keys[5] < tmp_601 || (keys[5] == tmp_601 && values[5] < tmp_602); if tmp_584 == tmp_603 { keys[5] = tmp_601; values[5] = tmp_602; } let tmp_604 = smem_keys[tmp_585 * WPT + 6u]; let tmp_605 = smem_vals[tmp_585 * WPT + 6u]; let tmp_606 = keys[6] < tmp_604 || (keys[6] == tmp_604 && values[6] < tmp_605); if tmp_584 == tmp_606 { keys[6] = tmp_604; values[6] = tmp_605; } let tmp_607 = smem_keys[tmp_585 * WPT + 7u]; let tmp_608 = smem_vals[tmp_585 * WPT + 7u]; let tmp_609 = keys[7] < tmp_607 || (keys[7] == tmp_607 && values[7] < tmp_608); if tmp_584 == tmp_609 { keys[7] = tmp_607; values[7] = tmp_608; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_610 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_611 = seg_base + (local_tid ^ 8u); let tmp_612 = smem_keys[tmp_611 * WPT + 0u]; let tmp_613 = smem_vals[tmp_611 * WPT + 0u]; let tmp_614 = keys[0] < tmp_612 || (keys[0] == tmp_612 && values[0] < tmp_613); if tmp_610 == tmp_614 { keys[0] = tmp_612; values[0] = tmp_613; } let tmp_615 = smem_keys[tmp_611 * WPT + 1u]; let tmp_616 = smem_vals[tmp_611 * WPT + 1u]; let tmp_617 = keys[1] < tmp_615 || (keys[1] == tmp_615 && values[1] < tmp_616); if tmp_610 == tmp_617 { keys[1] = tmp_615; values[1] = tmp_616; } let tmp_618 = smem_keys[tmp_611 * WPT + 2u]; let tmp_619 = smem_vals[tmp_611 * WPT + 2u]; let tmp_620 = keys[2] < tmp_618 || (keys[2] == tmp_618 && values[2] < tmp_619); if tmp_610 == tmp_620 { keys[2] = tmp_618; values[2] = tmp_619; } let tmp_621 = smem_keys[tmp_611 * WPT + 3u]; let tmp_622 = smem_vals[tmp_611 * WPT + 3u]; let tmp_623 = keys[3] < tmp_621 || (keys[3] == tmp_621 && values[3] < tmp_622); if tmp_610 == tmp_623 { keys[3] = tmp_621; values[3] = tmp_622; } let tmp_624 = smem_keys[tmp_611 * WPT + 4u]; let tmp_625 = smem_vals[tmp_611 * WPT + 4u]; let tmp_626 = keys[4] < tmp_624 || (keys[4] == tmp_624 && values[4] < tmp_625); if tmp_610 == tmp_626 { keys[4] = tmp_624; values[4] = tmp_625; } let tmp_627 = smem_keys[tmp_611 * WPT + 5u]; let tmp_628 = smem_vals[tmp_611 * WPT + 5u]; let tmp_629 = keys[5] < tmp_627 || (keys[5] == tmp_627 && values[5] < tmp_628); if tmp_610 == tmp_629 { keys[5] = tmp_627; values[5] = tmp_628; } let tmp_630 = smem_keys[tmp_611 * WPT + 6u]; let tmp_631 = smem_vals[tmp_611 * WPT + 6u]; let tmp_632 = keys[6] < tmp_630 || (keys[6] == tmp_630 && values[6] < tmp_631); if tmp_610 == tmp_632 { keys[6] = tmp_630; values[6] = tmp_631; } let tmp_633 = smem_keys[tmp_611 * WPT + 7u]; let tmp_634 = smem_vals[tmp_611 * WPT + 7u]; let tmp_635 = keys[7] < tmp_633 || (keys[7] == tmp_633 && values[7] < tmp_634); if tmp_610 == tmp_635 { keys[7] = tmp_633; values[7] = tmp_634; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_636 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_637 = seg_base + (local_tid ^ 4u); let tmp_638 = smem_keys[tmp_637 * WPT + 0u]; let tmp_639 = smem_vals[tmp_637 * WPT + 0u]; let tmp_640 = keys[0] < tmp_638 || (keys[0] == tmp_638 && values[0] < tmp_639); if tmp_636 == tmp_640 { keys[0] = tmp_638; values[0] = tmp_639; } let tmp_641 = smem_keys[tmp_637 * WPT + 1u]; let tmp_642 = smem_vals[tmp_637 * WPT + 1u]; let tmp_643 = keys[1] < tmp_641 || (keys[1] == tmp_641 && values[1] < tmp_642); if tmp_636 == tmp_643 { keys[1] = tmp_641; values[1] = tmp_642; } let tmp_644 = smem_keys[tmp_637 * WPT + 2u]; let tmp_645 = smem_vals[tmp_637 * WPT + 2u]; let tmp_646 = keys[2] < tmp_644 || (keys[2] == tmp_644 && values[2] < tmp_645); if tmp_636 == tmp_646 { keys[2] = tmp_644; values[2] = tmp_645; } let tmp_647 = smem_keys[tmp_637 * WPT + 3u]; let tmp_648 = smem_vals[tmp_637 * WPT + 3u]; let tmp_649 = keys[3] < tmp_647 || (keys[3] == tmp_647 && values[3] < tmp_648); if tmp_636 == tmp_649 { keys[3] = tmp_647; values[3] = tmp_648; } let tmp_650 = smem_keys[tmp_637 * WPT + 4u]; let tmp_651 = smem_vals[tmp_637 * WPT + 4u]; let tmp_652 = keys[4] < tmp_650 || (keys[4] == tmp_650 && values[4] < tmp_651); if tmp_636 == tmp_652 { keys[4] = tmp_650; values[4] = tmp_651; } let tmp_653 = smem_keys[tmp_637 * WPT + 5u]; let tmp_654 = smem_vals[tmp_637 * WPT + 5u]; let tmp_655 = keys[5] < tmp_653 || (keys[5] == tmp_653 && values[5] < tmp_654); if tmp_636 == tmp_655 { keys[5] = tmp_653; values[5] = tmp_654; } let tmp_656 = smem_keys[tmp_637 * WPT + 6u]; let tmp_657 = smem_vals[tmp_637 * WPT + 6u]; let tmp_658 = keys[6] < tmp_656 || (keys[6] == tmp_656 && values[6] < tmp_657); if tmp_636 == tmp_658 { keys[6] = tmp_656; values[6] = tmp_657; } let tmp_659 = smem_keys[tmp_637 * WPT + 7u]; let tmp_660 = smem_vals[tmp_637 * WPT + 7u]; let tmp_661 = keys[7] < tmp_659 || (keys[7] == tmp_659 && values[7] < tmp_660); if tmp_636 == tmp_661 { keys[7] = tmp_659; values[7] = tmp_660; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_662 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_663 = seg_base + (local_tid ^ 2u); let tmp_664 = smem_keys[tmp_663 * WPT + 0u]; let tmp_665 = smem_vals[tmp_663 * WPT + 0u]; let tmp_666 = keys[0] < tmp_664 || (keys[0] == tmp_664 && values[0] < tmp_665); if tmp_662 == tmp_666 { keys[0] = tmp_664; values[0] = tmp_665; } let tmp_667 = smem_keys[tmp_663 * WPT + 1u]; let tmp_668 = smem_vals[tmp_663 * WPT + 1u]; let tmp_669 = keys[1] < tmp_667 || (keys[1] == tmp_667 && values[1] < tmp_668); if tmp_662 == tmp_669 { keys[1] = tmp_667; values[1] = tmp_668; } let tmp_670 = smem_keys[tmp_663 * WPT + 2u]; let tmp_671 = smem_vals[tmp_663 * WPT + 2u]; let tmp_672 = keys[2] < tmp_670 || (keys[2] == tmp_670 && values[2] < tmp_671); if tmp_662 == tmp_672 { keys[2] = tmp_670; values[2] = tmp_671; } let tmp_673 = smem_keys[tmp_663 * WPT + 3u]; let tmp_674 = smem_vals[tmp_663 * WPT + 3u]; let tmp_675 = keys[3] < tmp_673 || (keys[3] == tmp_673 && values[3] < tmp_674); if tmp_662 == tmp_675 { keys[3] = tmp_673; values[3] = tmp_674; } let tmp_676 = smem_keys[tmp_663 * WPT + 4u]; let tmp_677 = smem_vals[tmp_663 * WPT + 4u]; let tmp_678 = keys[4] < tmp_676 || (keys[4] == tmp_676 && values[4] < tmp_677); if tmp_662 == tmp_678 { keys[4] = tmp_676; values[4] = tmp_677; } let tmp_679 = smem_keys[tmp_663 * WPT + 5u]; let tmp_680 = smem_vals[tmp_663 * WPT + 5u]; let tmp_681 = keys[5] < tmp_679 || (keys[5] == tmp_679 && values[5] < tmp_680); if tmp_662 == tmp_681 { keys[5] = tmp_679; values[5] = tmp_680; } let tmp_682 = smem_keys[tmp_663 * WPT + 6u]; let tmp_683 = smem_vals[tmp_663 * WPT + 6u]; let tmp_684 = keys[6] < tmp_682 || (keys[6] == tmp_682 && values[6] < tmp_683); if tmp_662 == tmp_684 { keys[6] = tmp_682; values[6] = tmp_683; } let tmp_685 = smem_keys[tmp_663 * WPT + 7u]; let tmp_686 = smem_vals[tmp_663 * WPT + 7u]; let tmp_687 = keys[7] < tmp_685 || (keys[7] == tmp_685 && values[7] < tmp_686); if tmp_662 == tmp_687 { keys[7] = tmp_685; values[7] = tmp_686; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_688 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_689 = seg_base + (local_tid ^ 1u); let tmp_690 = smem_keys[tmp_689 * WPT + 0u]; let tmp_691 = smem_vals[tmp_689 * WPT + 0u]; let tmp_692 = keys[0] < tmp_690 || (keys[0] == tmp_690 && values[0] < tmp_691); if tmp_688 == tmp_692 { keys[0] = tmp_690; values[0] = tmp_691; } let tmp_693 = smem_keys[tmp_689 * WPT + 1u]; let tmp_694 = smem_vals[tmp_689 * WPT + 1u]; let tmp_695 = keys[1] < tmp_693 || (keys[1] == tmp_693 && values[1] < tmp_694); if tmp_688 == tmp_695 { keys[1] = tmp_693; values[1] = tmp_694; } let tmp_696 = smem_keys[tmp_689 * WPT + 2u]; let tmp_697 = smem_vals[tmp_689 * WPT + 2u]; let tmp_698 = keys[2] < tmp_696 || (keys[2] == tmp_696 && values[2] < tmp_697); if tmp_688 == tmp_698 { keys[2] = tmp_696; values[2] = tmp_697; } let tmp_699 = smem_keys[tmp_689 * WPT + 3u]; let tmp_700 = smem_vals[tmp_689 * WPT + 3u]; let tmp_701 = keys[3] < tmp_699 || (keys[3] == tmp_699 && values[3] < tmp_700); if tmp_688 == tmp_701 { keys[3] = tmp_699; values[3] = tmp_700; } let tmp_702 = smem_keys[tmp_689 * WPT + 4u]; let tmp_703 = smem_vals[tmp_689 * WPT + 4u]; let tmp_704 = keys[4] < tmp_702 || (keys[4] == tmp_702 && values[4] < tmp_703); if tmp_688 == tmp_704 { keys[4] = tmp_702; values[4] = tmp_703; } let tmp_705 = smem_keys[tmp_689 * WPT + 5u]; let tmp_706 = smem_vals[tmp_689 * WPT + 5u]; let tmp_707 = keys[5] < tmp_705 || (keys[5] == tmp_705 && values[5] < tmp_706); if tmp_688 == tmp_707 { keys[5] = tmp_705; values[5] = tmp_706; } let tmp_708 = smem_keys[tmp_689 * WPT + 6u]; let tmp_709 = smem_vals[tmp_689 * WPT + 6u]; let tmp_710 = keys[6] < tmp_708 || (keys[6] == tmp_708 && values[6] < tmp_709); if tmp_688 == tmp_710 { keys[6] = tmp_708; values[6] = tmp_709; } let tmp_711 = smem_keys[tmp_689 * WPT + 7u]; let tmp_712 = smem_vals[tmp_689 * WPT + 7u]; let tmp_713 = keys[7] < tmp_711 || (keys[7] == tmp_711 && values[7] < tmp_712); if tmp_688 == tmp_713 { keys[7] = tmp_711; values[7] = tmp_712; } workgroupBarrier(); }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_714 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_714;let tmp_715 = values[0]; values[0] = values[4]; values[4] = tmp_715; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_716 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_716;let tmp_717 = values[1]; values[1] = values[5]; values[5] = tmp_717; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_718 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_718;let tmp_719 = values[2]; values[2] = values[6]; values[6] = tmp_719; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_720 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_720;let tmp_721 = values[3]; values[3] = values[7]; values[7] = tmp_721; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_722 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_722;let tmp_723 = values[0]; values[0] = values[2]; values[2] = tmp_723; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_724 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_724;let tmp_725 = values[1]; values[1] = values[3]; values[3] = tmp_725; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_726 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_726;let tmp_727 = values[4]; values[4] = values[6]; values[6] = tmp_727; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_728 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_728;let tmp_729 = values[5]; values[5] = values[7]; values[7] = tmp_729; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_730 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_730;let tmp_731 = values[0]; values[0] = values[1]; values[1] = tmp_731; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_732 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_732;let tmp_733 = values[2]; values[2] = values[3]; values[3] = tmp_733; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_734 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_734;let tmp_735 = values[4]; values[4] = values[5]; values[5] = tmp_735; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_736 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_736;let tmp_737 = values[6]; values[6] = values[7]; values[7] = tmp_737; }
    }
    // exch_intxn(tmask:127,swbit:6,wpt:8)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_738 = extractBits(local_tid, 6u, 1u) != 0u; let tmp_739 = seg_base + (local_tid ^ 127u); let tmp_740 = smem_keys[tmp_739 * WPT + 7u]; let tmp_741 = smem_vals[tmp_739 * WPT + 7u]; let tmp_742 = keys[0] < tmp_740 || (keys[0] == tmp_740 && values[0] < tmp_741); if tmp_738 == tmp_742 { keys[0] = tmp_740; values[0] = tmp_741; } let tmp_743 = smem_keys[tmp_739 * WPT + 6u]; let tmp_744 = smem_vals[tmp_739 * WPT + 6u]; let tmp_745 = keys[1] < tmp_743 || (keys[1] == tmp_743 && values[1] < tmp_744); if tmp_738 == tmp_745 { keys[1] = tmp_743; values[1] = tmp_744; } let tmp_746 = smem_keys[tmp_739 * WPT + 5u]; let tmp_747 = smem_vals[tmp_739 * WPT + 5u]; let tmp_748 = keys[2] < tmp_746 || (keys[2] == tmp_746 && values[2] < tmp_747); if tmp_738 == tmp_748 { keys[2] = tmp_746; values[2] = tmp_747; } let tmp_749 = smem_keys[tmp_739 * WPT + 4u]; let tmp_750 = smem_vals[tmp_739 * WPT + 4u]; let tmp_751 = keys[3] < tmp_749 || (keys[3] == tmp_749 && values[3] < tmp_750); if tmp_738 == tmp_751 { keys[3] = tmp_749; values[3] = tmp_750; } let tmp_752 = smem_keys[tmp_739 * WPT + 3u]; let tmp_753 = smem_vals[tmp_739 * WPT + 3u]; let tmp_754 = keys[4] < tmp_752 || (keys[4] == tmp_752 && values[4] < tmp_753); if tmp_738 == tmp_754 { keys[4] = tmp_752; values[4] = tmp_753; } let tmp_755 = smem_keys[tmp_739 * WPT + 2u]; let tmp_756 = smem_vals[tmp_739 * WPT + 2u]; let tmp_757 = keys[5] < tmp_755 || (keys[5] == tmp_755 && values[5] < tmp_756); if tmp_738 == tmp_757 { keys[5] = tmp_755; values[5] = tmp_756; } let tmp_758 = smem_keys[tmp_739 * WPT + 1u]; let tmp_759 = smem_vals[tmp_739 * WPT + 1u]; let tmp_760 = keys[6] < tmp_758 || (keys[6] == tmp_758 && values[6] < tmp_759); if tmp_738 == tmp_760 { keys[6] = tmp_758; values[6] = tmp_759; } let tmp_761 = smem_keys[tmp_739 * WPT + 0u]; let tmp_762 = smem_vals[tmp_739 * WPT + 0u]; let tmp_763 = keys[7] < tmp_761 || (keys[7] == tmp_761 && values[7] < tmp_762); if tmp_738 == tmp_763 { keys[7] = tmp_761; values[7] = tmp_762; } workgroupBarrier(); }
    // exch_paral(tmask:32,swbit:5,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_764 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_765 = seg_base + (local_tid ^ 32u); let tmp_766 = smem_keys[tmp_765 * WPT + 0u]; let tmp_767 = smem_vals[tmp_765 * WPT + 0u]; let tmp_768 = keys[0] < tmp_766 || (keys[0] == tmp_766 && values[0] < tmp_767); if tmp_764 == tmp_768 { keys[0] = tmp_766; values[0] = tmp_767; } let tmp_769 = smem_keys[tmp_765 * WPT + 1u]; let tmp_770 = smem_vals[tmp_765 * WPT + 1u]; let tmp_771 = keys[1] < tmp_769 || (keys[1] == tmp_769 && values[1] < tmp_770); if tmp_764 == tmp_771 { keys[1] = tmp_769; values[1] = tmp_770; } let tmp_772 = smem_keys[tmp_765 * WPT + 2u]; let tmp_773 = smem_vals[tmp_765 * WPT + 2u]; let tmp_774 = keys[2] < tmp_772 || (keys[2] == tmp_772 && values[2] < tmp_773); if tmp_764 == tmp_774 { keys[2] = tmp_772; values[2] = tmp_773; } let tmp_775 = smem_keys[tmp_765 * WPT + 3u]; let tmp_776 = smem_vals[tmp_765 * WPT + 3u]; let tmp_777 = keys[3] < tmp_775 || (keys[3] == tmp_775 && values[3] < tmp_776); if tmp_764 == tmp_777 { keys[3] = tmp_775; values[3] = tmp_776; } let tmp_778 = smem_keys[tmp_765 * WPT + 4u]; let tmp_779 = smem_vals[tmp_765 * WPT + 4u]; let tmp_780 = keys[4] < tmp_778 || (keys[4] == tmp_778 && values[4] < tmp_779); if tmp_764 == tmp_780 { keys[4] = tmp_778; values[4] = tmp_779; } let tmp_781 = smem_keys[tmp_765 * WPT + 5u]; let tmp_782 = smem_vals[tmp_765 * WPT + 5u]; let tmp_783 = keys[5] < tmp_781 || (keys[5] == tmp_781 && values[5] < tmp_782); if tmp_764 == tmp_783 { keys[5] = tmp_781; values[5] = tmp_782; } let tmp_784 = smem_keys[tmp_765 * WPT + 6u]; let tmp_785 = smem_vals[tmp_765 * WPT + 6u]; let tmp_786 = keys[6] < tmp_784 || (keys[6] == tmp_784 && values[6] < tmp_785); if tmp_764 == tmp_786 { keys[6] = tmp_784; values[6] = tmp_785; } let tmp_787 = smem_keys[tmp_765 * WPT + 7u]; let tmp_788 = smem_vals[tmp_765 * WPT + 7u]; let tmp_789 = keys[7] < tmp_787 || (keys[7] == tmp_787 && values[7] < tmp_788); if tmp_764 == tmp_789 { keys[7] = tmp_787; values[7] = tmp_788; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_790 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_791 = seg_base + (local_tid ^ 16u); let tmp_792 = smem_keys[tmp_791 * WPT + 0u]; let tmp_793 = smem_vals[tmp_791 * WPT + 0u]; let tmp_794 = keys[0] < tmp_792 || (keys[0] == tmp_792 && values[0] < tmp_793); if tmp_790 == tmp_794 { keys[0] = tmp_792; values[0] = tmp_793; } let tmp_795 = smem_keys[tmp_791 * WPT + 1u]; let tmp_796 = smem_vals[tmp_791 * WPT + 1u]; let tmp_797 = keys[1] < tmp_795 || (keys[1] == tmp_795 && values[1] < tmp_796); if tmp_790 == tmp_797 { keys[1] = tmp_795; values[1] = tmp_796; } let tmp_798 = smem_keys[tmp_791 * WPT + 2u]; let tmp_799 = smem_vals[tmp_791 * WPT + 2u]; let tmp_800 = keys[2] < tmp_798 || (keys[2] == tmp_798 && values[2] < tmp_799); if tmp_790 == tmp_800 { keys[2] = tmp_798; values[2] = tmp_799; } let tmp_801 = smem_keys[tmp_791 * WPT + 3u]; let tmp_802 = smem_vals[tmp_791 * WPT + 3u]; let tmp_803 = keys[3] < tmp_801 || (keys[3] == tmp_801 && values[3] < tmp_802); if tmp_790 == tmp_803 { keys[3] = tmp_801; values[3] = tmp_802; } let tmp_804 = smem_keys[tmp_791 * WPT + 4u]; let tmp_805 = smem_vals[tmp_791 * WPT + 4u]; let tmp_806 = keys[4] < tmp_804 || (keys[4] == tmp_804 && values[4] < tmp_805); if tmp_790 == tmp_806 { keys[4] = tmp_804; values[4] = tmp_805; } let tmp_807 = smem_keys[tmp_791 * WPT + 5u]; let tmp_808 = smem_vals[tmp_791 * WPT + 5u]; let tmp_809 = keys[5] < tmp_807 || (keys[5] == tmp_807 && values[5] < tmp_808); if tmp_790 == tmp_809 { keys[5] = tmp_807; values[5] = tmp_808; } let tmp_810 = smem_keys[tmp_791 * WPT + 6u]; let tmp_811 = smem_vals[tmp_791 * WPT + 6u]; let tmp_812 = keys[6] < tmp_810 || (keys[6] == tmp_810 && values[6] < tmp_811); if tmp_790 == tmp_812 { keys[6] = tmp_810; values[6] = tmp_811; } let tmp_813 = smem_keys[tmp_791 * WPT + 7u]; let tmp_814 = smem_vals[tmp_791 * WPT + 7u]; let tmp_815 = keys[7] < tmp_813 || (keys[7] == tmp_813 && values[7] < tmp_814); if tmp_790 == tmp_815 { keys[7] = tmp_813; values[7] = tmp_814; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_816 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_817 = seg_base + (local_tid ^ 8u); let tmp_818 = smem_keys[tmp_817 * WPT + 0u]; let tmp_819 = smem_vals[tmp_817 * WPT + 0u]; let tmp_820 = keys[0] < tmp_818 || (keys[0] == tmp_818 && values[0] < tmp_819); if tmp_816 == tmp_820 { keys[0] = tmp_818; values[0] = tmp_819; } let tmp_821 = smem_keys[tmp_817 * WPT + 1u]; let tmp_822 = smem_vals[tmp_817 * WPT + 1u]; let tmp_823 = keys[1] < tmp_821 || (keys[1] == tmp_821 && values[1] < tmp_822); if tmp_816 == tmp_823 { keys[1] = tmp_821; values[1] = tmp_822; } let tmp_824 = smem_keys[tmp_817 * WPT + 2u]; let tmp_825 = smem_vals[tmp_817 * WPT + 2u]; let tmp_826 = keys[2] < tmp_824 || (keys[2] == tmp_824 && values[2] < tmp_825); if tmp_816 == tmp_826 { keys[2] = tmp_824; values[2] = tmp_825; } let tmp_827 = smem_keys[tmp_817 * WPT + 3u]; let tmp_828 = smem_vals[tmp_817 * WPT + 3u]; let tmp_829 = keys[3] < tmp_827 || (keys[3] == tmp_827 && values[3] < tmp_828); if tmp_816 == tmp_829 { keys[3] = tmp_827; values[3] = tmp_828; } let tmp_830 = smem_keys[tmp_817 * WPT + 4u]; let tmp_831 = smem_vals[tmp_817 * WPT + 4u]; let tmp_832 = keys[4] < tmp_830 || (keys[4] == tmp_830 && values[4] < tmp_831); if tmp_816 == tmp_832 { keys[4] = tmp_830; values[4] = tmp_831; } let tmp_833 = smem_keys[tmp_817 * WPT + 5u]; let tmp_834 = smem_vals[tmp_817 * WPT + 5u]; let tmp_835 = keys[5] < tmp_833 || (keys[5] == tmp_833 && values[5] < tmp_834); if tmp_816 == tmp_835 { keys[5] = tmp_833; values[5] = tmp_834; } let tmp_836 = smem_keys[tmp_817 * WPT + 6u]; let tmp_837 = smem_vals[tmp_817 * WPT + 6u]; let tmp_838 = keys[6] < tmp_836 || (keys[6] == tmp_836 && values[6] < tmp_837); if tmp_816 == tmp_838 { keys[6] = tmp_836; values[6] = tmp_837; } let tmp_839 = smem_keys[tmp_817 * WPT + 7u]; let tmp_840 = smem_vals[tmp_817 * WPT + 7u]; let tmp_841 = keys[7] < tmp_839 || (keys[7] == tmp_839 && values[7] < tmp_840); if tmp_816 == tmp_841 { keys[7] = tmp_839; values[7] = tmp_840; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_842 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_843 = seg_base + (local_tid ^ 4u); let tmp_844 = smem_keys[tmp_843 * WPT + 0u]; let tmp_845 = smem_vals[tmp_843 * WPT + 0u]; let tmp_846 = keys[0] < tmp_844 || (keys[0] == tmp_844 && values[0] < tmp_845); if tmp_842 == tmp_846 { keys[0] = tmp_844; values[0] = tmp_845; } let tmp_847 = smem_keys[tmp_843 * WPT + 1u]; let tmp_848 = smem_vals[tmp_843 * WPT + 1u]; let tmp_849 = keys[1] < tmp_847 || (keys[1] == tmp_847 && values[1] < tmp_848); if tmp_842 == tmp_849 { keys[1] = tmp_847; values[1] = tmp_848; } let tmp_850 = smem_keys[tmp_843 * WPT + 2u]; let tmp_851 = smem_vals[tmp_843 * WPT + 2u]; let tmp_852 = keys[2] < tmp_850 || (keys[2] == tmp_850 && values[2] < tmp_851); if tmp_842 == tmp_852 { keys[2] = tmp_850; values[2] = tmp_851; } let tmp_853 = smem_keys[tmp_843 * WPT + 3u]; let tmp_854 = smem_vals[tmp_843 * WPT + 3u]; let tmp_855 = keys[3] < tmp_853 || (keys[3] == tmp_853 && values[3] < tmp_854); if tmp_842 == tmp_855 { keys[3] = tmp_853; values[3] = tmp_854; } let tmp_856 = smem_keys[tmp_843 * WPT + 4u]; let tmp_857 = smem_vals[tmp_843 * WPT + 4u]; let tmp_858 = keys[4] < tmp_856 || (keys[4] == tmp_856 && values[4] < tmp_857); if tmp_842 == tmp_858 { keys[4] = tmp_856; values[4] = tmp_857; } let tmp_859 = smem_keys[tmp_843 * WPT + 5u]; let tmp_860 = smem_vals[tmp_843 * WPT + 5u]; let tmp_861 = keys[5] < tmp_859 || (keys[5] == tmp_859 && values[5] < tmp_860); if tmp_842 == tmp_861 { keys[5] = tmp_859; values[5] = tmp_860; } let tmp_862 = smem_keys[tmp_843 * WPT + 6u]; let tmp_863 = smem_vals[tmp_843 * WPT + 6u]; let tmp_864 = keys[6] < tmp_862 || (keys[6] == tmp_862 && values[6] < tmp_863); if tmp_842 == tmp_864 { keys[6] = tmp_862; values[6] = tmp_863; } let tmp_865 = smem_keys[tmp_843 * WPT + 7u]; let tmp_866 = smem_vals[tmp_843 * WPT + 7u]; let tmp_867 = keys[7] < tmp_865 || (keys[7] == tmp_865 && values[7] < tmp_866); if tmp_842 == tmp_867 { keys[7] = tmp_865; values[7] = tmp_866; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_868 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_869 = seg_base + (local_tid ^ 2u); let tmp_870 = smem_keys[tmp_869 * WPT + 0u]; let tmp_871 = smem_vals[tmp_869 * WPT + 0u]; let tmp_872 = keys[0] < tmp_870 || (keys[0] == tmp_870 && values[0] < tmp_871); if tmp_868 == tmp_872 { keys[0] = tmp_870; values[0] = tmp_871; } let tmp_873 = smem_keys[tmp_869 * WPT + 1u]; let tmp_874 = smem_vals[tmp_869 * WPT + 1u]; let tmp_875 = keys[1] < tmp_873 || (keys[1] == tmp_873 && values[1] < tmp_874); if tmp_868 == tmp_875 { keys[1] = tmp_873; values[1] = tmp_874; } let tmp_876 = smem_keys[tmp_869 * WPT + 2u]; let tmp_877 = smem_vals[tmp_869 * WPT + 2u]; let tmp_878 = keys[2] < tmp_876 || (keys[2] == tmp_876 && values[2] < tmp_877); if tmp_868 == tmp_878 { keys[2] = tmp_876; values[2] = tmp_877; } let tmp_879 = smem_keys[tmp_869 * WPT + 3u]; let tmp_880 = smem_vals[tmp_869 * WPT + 3u]; let tmp_881 = keys[3] < tmp_879 || (keys[3] == tmp_879 && values[3] < tmp_880); if tmp_868 == tmp_881 { keys[3] = tmp_879; values[3] = tmp_880; } let tmp_882 = smem_keys[tmp_869 * WPT + 4u]; let tmp_883 = smem_vals[tmp_869 * WPT + 4u]; let tmp_884 = keys[4] < tmp_882 || (keys[4] == tmp_882 && values[4] < tmp_883); if tmp_868 == tmp_884 { keys[4] = tmp_882; values[4] = tmp_883; } let tmp_885 = smem_keys[tmp_869 * WPT + 5u]; let tmp_886 = smem_vals[tmp_869 * WPT + 5u]; let tmp_887 = keys[5] < tmp_885 || (keys[5] == tmp_885 && values[5] < tmp_886); if tmp_868 == tmp_887 { keys[5] = tmp_885; values[5] = tmp_886; } let tmp_888 = smem_keys[tmp_869 * WPT + 6u]; let tmp_889 = smem_vals[tmp_869 * WPT + 6u]; let tmp_890 = keys[6] < tmp_888 || (keys[6] == tmp_888 && values[6] < tmp_889); if tmp_868 == tmp_890 { keys[6] = tmp_888; values[6] = tmp_889; } let tmp_891 = smem_keys[tmp_869 * WPT + 7u]; let tmp_892 = smem_vals[tmp_869 * WPT + 7u]; let tmp_893 = keys[7] < tmp_891 || (keys[7] == tmp_891 && values[7] < tmp_892); if tmp_868 == tmp_893 { keys[7] = tmp_891; values[7] = tmp_892; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_894 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_895 = seg_base + (local_tid ^ 1u); let tmp_896 = smem_keys[tmp_895 * WPT + 0u]; let tmp_897 = smem_vals[tmp_895 * WPT + 0u]; let tmp_898 = keys[0] < tmp_896 || (keys[0] == tmp_896 && values[0] < tmp_897); if tmp_894 == tmp_898 { keys[0] = tmp_896; values[0] = tmp_897; } let tmp_899 = smem_keys[tmp_895 * WPT + 1u]; let tmp_900 = smem_vals[tmp_895 * WPT + 1u]; let tmp_901 = keys[1] < tmp_899 || (keys[1] == tmp_899 && values[1] < tmp_900); if tmp_894 == tmp_901 { keys[1] = tmp_899; values[1] = tmp_900; } let tmp_902 = smem_keys[tmp_895 * WPT + 2u]; let tmp_903 = smem_vals[tmp_895 * WPT + 2u]; let tmp_904 = keys[2] < tmp_902 || (keys[2] == tmp_902 && values[2] < tmp_903); if tmp_894 == tmp_904 { keys[2] = tmp_902; values[2] = tmp_903; } let tmp_905 = smem_keys[tmp_895 * WPT + 3u]; let tmp_906 = smem_vals[tmp_895 * WPT + 3u]; let tmp_907 = keys[3] < tmp_905 || (keys[3] == tmp_905 && values[3] < tmp_906); if tmp_894 == tmp_907 { keys[3] = tmp_905; values[3] = tmp_906; } let tmp_908 = smem_keys[tmp_895 * WPT + 4u]; let tmp_909 = smem_vals[tmp_895 * WPT + 4u]; let tmp_910 = keys[4] < tmp_908 || (keys[4] == tmp_908 && values[4] < tmp_909); if tmp_894 == tmp_910 { keys[4] = tmp_908; values[4] = tmp_909; } let tmp_911 = smem_keys[tmp_895 * WPT + 5u]; let tmp_912 = smem_vals[tmp_895 * WPT + 5u]; let tmp_913 = keys[5] < tmp_911 || (keys[5] == tmp_911 && values[5] < tmp_912); if tmp_894 == tmp_913 { keys[5] = tmp_911; values[5] = tmp_912; } let tmp_914 = smem_keys[tmp_895 * WPT + 6u]; let tmp_915 = smem_vals[tmp_895 * WPT + 6u]; let tmp_916 = keys[6] < tmp_914 || (keys[6] == tmp_914 && values[6] < tmp_915); if tmp_894 == tmp_916 { keys[6] = tmp_914; values[6] = tmp_915; } let tmp_917 = smem_keys[tmp_895 * WPT + 7u]; let tmp_918 = smem_vals[tmp_895 * WPT + 7u]; let tmp_919 = keys[7] < tmp_917 || (keys[7] == tmp_917 && values[7] < tmp_918); if tmp_894 == tmp_919 { keys[7] = tmp_917; values[7] = tmp_918; } workgroupBarrier(); }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_920 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_920;let tmp_921 = values[0]; values[0] = values[4]; values[4] = tmp_921; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_922 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_922;let tmp_923 = values[1]; values[1] = values[5]; values[5] = tmp_923; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_924 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_924;let tmp_925 = values[2]; values[2] = values[6]; values[6] = tmp_925; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_926 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_926;let tmp_927 = values[3]; values[3] = values[7]; values[7] = tmp_927; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_928 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_928;let tmp_929 = values[0]; values[0] = values[2]; values[2] = tmp_929; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_930 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_930;let tmp_931 = values[1]; values[1] = values[3]; values[3] = tmp_931; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_932 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_932;let tmp_933 = values[4]; values[4] = values[6]; values[6] = tmp_933; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_934 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_934;let tmp_935 = values[5]; values[5] = values[7]; values[7] = tmp_935; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_936 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_936;let tmp_937 = values[0]; values[0] = values[1]; values[1] = tmp_937; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_938 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_938;let tmp_939 = values[2]; values[2] = values[3]; values[3] = tmp_939; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_940 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_940;let tmp_941 = values[4]; values[4] = values[5]; values[5] = tmp_941; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_942 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_942;let tmp_943 = values[6]; values[6] = values[7]; values[7] = tmp_943; }
    }
    // exch_intxn(tmask:255,swbit:7,wpt:8)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_944 = extractBits(local_tid, 7u, 1u) != 0u; let tmp_945 = seg_base + (local_tid ^ 255u); let tmp_946 = smem_keys[tmp_945 * WPT + 7u]; let tmp_947 = smem_vals[tmp_945 * WPT + 7u]; let tmp_948 = keys[0] < tmp_946 || (keys[0] == tmp_946 && values[0] < tmp_947); if tmp_944 == tmp_948 { keys[0] = tmp_946; values[0] = tmp_947; } let tmp_949 = smem_keys[tmp_945 * WPT + 6u]; let tmp_950 = smem_vals[tmp_945 * WPT + 6u]; let tmp_951 = keys[1] < tmp_949 || (keys[1] == tmp_949 && values[1] < tmp_950); if tmp_944 == tmp_951 { keys[1] = tmp_949; values[1] = tmp_950; } let tmp_952 = smem_keys[tmp_945 * WPT + 5u]; let tmp_953 = smem_vals[tmp_945 * WPT + 5u]; let tmp_954 = keys[2] < tmp_952 || (keys[2] == tmp_952 && values[2] < tmp_953); if tmp_944 == tmp_954 { keys[2] = tmp_952; values[2] = tmp_953; } let tmp_955 = smem_keys[tmp_945 * WPT + 4u]; let tmp_956 = smem_vals[tmp_945 * WPT + 4u]; let tmp_957 = keys[3] < tmp_955 || (keys[3] == tmp_955 && values[3] < tmp_956); if tmp_944 == tmp_957 { keys[3] = tmp_955; values[3] = tmp_956; } let tmp_958 = smem_keys[tmp_945 * WPT + 3u]; let tmp_959 = smem_vals[tmp_945 * WPT + 3u]; let tmp_960 = keys[4] < tmp_958 || (keys[4] == tmp_958 && values[4] < tmp_959); if tmp_944 == tmp_960 { keys[4] = tmp_958; values[4] = tmp_959; } let tmp_961 = smem_keys[tmp_945 * WPT + 2u]; let tmp_962 = smem_vals[tmp_945 * WPT + 2u]; let tmp_963 = keys[5] < tmp_961 || (keys[5] == tmp_961 && values[5] < tmp_962); if tmp_944 == tmp_963 { keys[5] = tmp_961; values[5] = tmp_962; } let tmp_964 = smem_keys[tmp_945 * WPT + 1u]; let tmp_965 = smem_vals[tmp_945 * WPT + 1u]; let tmp_966 = keys[6] < tmp_964 || (keys[6] == tmp_964 && values[6] < tmp_965); if tmp_944 == tmp_966 { keys[6] = tmp_964; values[6] = tmp_965; } let tmp_967 = smem_keys[tmp_945 * WPT + 0u]; let tmp_968 = smem_vals[tmp_945 * WPT + 0u]; let tmp_969 = keys[7] < tmp_967 || (keys[7] == tmp_967 && values[7] < tmp_968); if tmp_944 == tmp_969 { keys[7] = tmp_967; values[7] = tmp_968; } workgroupBarrier(); }
    // exch_paral(tmask:64,swbit:6,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_970 = extractBits(local_tid, 6u, 1u) != 0u; let tmp_971 = seg_base + (local_tid ^ 64u); let tmp_972 = smem_keys[tmp_971 * WPT + 0u]; let tmp_973 = smem_vals[tmp_971 * WPT + 0u]; let tmp_974 = keys[0] < tmp_972 || (keys[0] == tmp_972 && values[0] < tmp_973); if tmp_970 == tmp_974 { keys[0] = tmp_972; values[0] = tmp_973; } let tmp_975 = smem_keys[tmp_971 * WPT + 1u]; let tmp_976 = smem_vals[tmp_971 * WPT + 1u]; let tmp_977 = keys[1] < tmp_975 || (keys[1] == tmp_975 && values[1] < tmp_976); if tmp_970 == tmp_977 { keys[1] = tmp_975; values[1] = tmp_976; } let tmp_978 = smem_keys[tmp_971 * WPT + 2u]; let tmp_979 = smem_vals[tmp_971 * WPT + 2u]; let tmp_980 = keys[2] < tmp_978 || (keys[2] == tmp_978 && values[2] < tmp_979); if tmp_970 == tmp_980 { keys[2] = tmp_978; values[2] = tmp_979; } let tmp_981 = smem_keys[tmp_971 * WPT + 3u]; let tmp_982 = smem_vals[tmp_971 * WPT + 3u]; let tmp_983 = keys[3] < tmp_981 || (keys[3] == tmp_981 && values[3] < tmp_982); if tmp_970 == tmp_983 { keys[3] = tmp_981; values[3] = tmp_982; } let tmp_984 = smem_keys[tmp_971 * WPT + 4u]; let tmp_985 = smem_vals[tmp_971 * WPT + 4u]; let tmp_986 = keys[4] < tmp_984 || (keys[4] == tmp_984 && values[4] < tmp_985); if tmp_970 == tmp_986 { keys[4] = tmp_984; values[4] = tmp_985; } let tmp_987 = smem_keys[tmp_971 * WPT + 5u]; let tmp_988 = smem_vals[tmp_971 * WPT + 5u]; let tmp_989 = keys[5] < tmp_987 || (keys[5] == tmp_987 && values[5] < tmp_988); if tmp_970 == tmp_989 { keys[5] = tmp_987; values[5] = tmp_988; } let tmp_990 = smem_keys[tmp_971 * WPT + 6u]; let tmp_991 = smem_vals[tmp_971 * WPT + 6u]; let tmp_992 = keys[6] < tmp_990 || (keys[6] == tmp_990 && values[6] < tmp_991); if tmp_970 == tmp_992 { keys[6] = tmp_990; values[6] = tmp_991; } let tmp_993 = smem_keys[tmp_971 * WPT + 7u]; let tmp_994 = smem_vals[tmp_971 * WPT + 7u]; let tmp_995 = keys[7] < tmp_993 || (keys[7] == tmp_993 && values[7] < tmp_994); if tmp_970 == tmp_995 { keys[7] = tmp_993; values[7] = tmp_994; } workgroupBarrier(); }
    // exch_paral(tmask:32,swbit:5,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_996 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_997 = seg_base + (local_tid ^ 32u); let tmp_998 = smem_keys[tmp_997 * WPT + 0u]; let tmp_999 = smem_vals[tmp_997 * WPT + 0u]; let tmp_1000 = keys[0] < tmp_998 || (keys[0] == tmp_998 && values[0] < tmp_999); if tmp_996 == tmp_1000 { keys[0] = tmp_998; values[0] = tmp_999; } let tmp_1001 = smem_keys[tmp_997 * WPT + 1u]; let tmp_1002 = smem_vals[tmp_997 * WPT + 1u]; let tmp_1003 = keys[1] < tmp_1001 || (keys[1] == tmp_1001 && values[1] < tmp_1002); if tmp_996 == tmp_1003 { keys[1] = tmp_1001; values[1] = tmp_1002; } let tmp_1004 = smem_keys[tmp_997 * WPT + 2u]; let tmp_1005 = smem_vals[tmp_997 * WPT + 2u]; let tmp_1006 = keys[2] < tmp_1004 || (keys[2] == tmp_1004 && values[2] < tmp_1005); if tmp_996 == tmp_1006 { keys[2] = tmp_1004; values[2] = tmp_1005; } let tmp_1007 = smem_keys[tmp_997 * WPT + 3u]; let tmp_1008 = smem_vals[tmp_997 * WPT + 3u]; let tmp_1009 = keys[3] < tmp_1007 || (keys[3] == tmp_1007 && values[3] < tmp_1008); if tmp_996 == tmp_1009 { keys[3] = tmp_1007; values[3] = tmp_1008; } let tmp_1010 = smem_keys[tmp_997 * WPT + 4u]; let tmp_1011 = smem_vals[tmp_997 * WPT + 4u]; let tmp_1012 = keys[4] < tmp_1010 || (keys[4] == tmp_1010 && values[4] < tmp_1011); if tmp_996 == tmp_1012 { keys[4] = tmp_1010; values[4] = tmp_1011; } let tmp_1013 = smem_keys[tmp_997 * WPT + 5u]; let tmp_1014 = smem_vals[tmp_997 * WPT + 5u]; let tmp_1015 = keys[5] < tmp_1013 || (keys[5] == tmp_1013 && values[5] < tmp_1014); if tmp_996 == tmp_1015 { keys[5] = tmp_1013; values[5] = tmp_1014; } let tmp_1016 = smem_keys[tmp_997 * WPT + 6u]; let tmp_1017 = smem_vals[tmp_997 * WPT + 6u]; let tmp_1018 = keys[6] < tmp_1016 || (keys[6] == tmp_1016 && values[6] < tmp_1017); if tmp_996 == tmp_1018 { keys[6] = tmp_1016; values[6] = tmp_1017; } let tmp_1019 = smem_keys[tmp_997 * WPT + 7u]; let tmp_1020 = smem_vals[tmp_997 * WPT + 7u]; let tmp_1021 = keys[7] < tmp_1019 || (keys[7] == tmp_1019 && values[7] < tmp_1020); if tmp_996 == tmp_1021 { keys[7] = tmp_1019; values[7] = tmp_1020; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_1022 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_1023 = seg_base + (local_tid ^ 16u); let tmp_1024 = smem_keys[tmp_1023 * WPT + 0u]; let tmp_1025 = smem_vals[tmp_1023 * WPT + 0u]; let tmp_1026 = keys[0] < tmp_1024 || (keys[0] == tmp_1024 && values[0] < tmp_1025); if tmp_1022 == tmp_1026 { keys[0] = tmp_1024; values[0] = tmp_1025; } let tmp_1027 = smem_keys[tmp_1023 * WPT + 1u]; let tmp_1028 = smem_vals[tmp_1023 * WPT + 1u]; let tmp_1029 = keys[1] < tmp_1027 || (keys[1] == tmp_1027 && values[1] < tmp_1028); if tmp_1022 == tmp_1029 { keys[1] = tmp_1027; values[1] = tmp_1028; } let tmp_1030 = smem_keys[tmp_1023 * WPT + 2u]; let tmp_1031 = smem_vals[tmp_1023 * WPT + 2u]; let tmp_1032 = keys[2] < tmp_1030 || (keys[2] == tmp_1030 && values[2] < tmp_1031); if tmp_1022 == tmp_1032 { keys[2] = tmp_1030; values[2] = tmp_1031; } let tmp_1033 = smem_keys[tmp_1023 * WPT + 3u]; let tmp_1034 = smem_vals[tmp_1023 * WPT + 3u]; let tmp_1035 = keys[3] < tmp_1033 || (keys[3] == tmp_1033 && values[3] < tmp_1034); if tmp_1022 == tmp_1035 { keys[3] = tmp_1033; values[3] = tmp_1034; } let tmp_1036 = smem_keys[tmp_1023 * WPT + 4u]; let tmp_1037 = smem_vals[tmp_1023 * WPT + 4u]; let tmp_1038 = keys[4] < tmp_1036 || (keys[4] == tmp_1036 && values[4] < tmp_1037); if tmp_1022 == tmp_1038 { keys[4] = tmp_1036; values[4] = tmp_1037; } let tmp_1039 = smem_keys[tmp_1023 * WPT + 5u]; let tmp_1040 = smem_vals[tmp_1023 * WPT + 5u]; let tmp_1041 = keys[5] < tmp_1039 || (keys[5] == tmp_1039 && values[5] < tmp_1040); if tmp_1022 == tmp_1041 { keys[5] = tmp_1039; values[5] = tmp_1040; } let tmp_1042 = smem_keys[tmp_1023 * WPT + 6u]; let tmp_1043 = smem_vals[tmp_1023 * WPT + 6u]; let tmp_1044 = keys[6] < tmp_1042 || (keys[6] == tmp_1042 && values[6] < tmp_1043); if tmp_1022 == tmp_1044 { keys[6] = tmp_1042; values[6] = tmp_1043; } let tmp_1045 = smem_keys[tmp_1023 * WPT + 7u]; let tmp_1046 = smem_vals[tmp_1023 * WPT + 7u]; let tmp_1047 = keys[7] < tmp_1045 || (keys[7] == tmp_1045 && values[7] < tmp_1046); if tmp_1022 == tmp_1047 { keys[7] = tmp_1045; values[7] = tmp_1046; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_1048 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_1049 = seg_base + (local_tid ^ 8u); let tmp_1050 = smem_keys[tmp_1049 * WPT + 0u]; let tmp_1051 = smem_vals[tmp_1049 * WPT + 0u]; let tmp_1052 = keys[0] < tmp_1050 || (keys[0] == tmp_1050 && values[0] < tmp_1051); if tmp_1048 == tmp_1052 { keys[0] = tmp_1050; values[0] = tmp_1051; } let tmp_1053 = smem_keys[tmp_1049 * WPT + 1u]; let tmp_1054 = smem_vals[tmp_1049 * WPT + 1u]; let tmp_1055 = keys[1] < tmp_1053 || (keys[1] == tmp_1053 && values[1] < tmp_1054); if tmp_1048 == tmp_1055 { keys[1] = tmp_1053; values[1] = tmp_1054; } let tmp_1056 = smem_keys[tmp_1049 * WPT + 2u]; let tmp_1057 = smem_vals[tmp_1049 * WPT + 2u]; let tmp_1058 = keys[2] < tmp_1056 || (keys[2] == tmp_1056 && values[2] < tmp_1057); if tmp_1048 == tmp_1058 { keys[2] = tmp_1056; values[2] = tmp_1057; } let tmp_1059 = smem_keys[tmp_1049 * WPT + 3u]; let tmp_1060 = smem_vals[tmp_1049 * WPT + 3u]; let tmp_1061 = keys[3] < tmp_1059 || (keys[3] == tmp_1059 && values[3] < tmp_1060); if tmp_1048 == tmp_1061 { keys[3] = tmp_1059; values[3] = tmp_1060; } let tmp_1062 = smem_keys[tmp_1049 * WPT + 4u]; let tmp_1063 = smem_vals[tmp_1049 * WPT + 4u]; let tmp_1064 = keys[4] < tmp_1062 || (keys[4] == tmp_1062 && values[4] < tmp_1063); if tmp_1048 == tmp_1064 { keys[4] = tmp_1062; values[4] = tmp_1063; } let tmp_1065 = smem_keys[tmp_1049 * WPT + 5u]; let tmp_1066 = smem_vals[tmp_1049 * WPT + 5u]; let tmp_1067 = keys[5] < tmp_1065 || (keys[5] == tmp_1065 && values[5] < tmp_1066); if tmp_1048 == tmp_1067 { keys[5] = tmp_1065; values[5] = tmp_1066; } let tmp_1068 = smem_keys[tmp_1049 * WPT + 6u]; let tmp_1069 = smem_vals[tmp_1049 * WPT + 6u]; let tmp_1070 = keys[6] < tmp_1068 || (keys[6] == tmp_1068 && values[6] < tmp_1069); if tmp_1048 == tmp_1070 { keys[6] = tmp_1068; values[6] = tmp_1069; } let tmp_1071 = smem_keys[tmp_1049 * WPT + 7u]; let tmp_1072 = smem_vals[tmp_1049 * WPT + 7u]; let tmp_1073 = keys[7] < tmp_1071 || (keys[7] == tmp_1071 && values[7] < tmp_1072); if tmp_1048 == tmp_1073 { keys[7] = tmp_1071; values[7] = tmp_1072; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_1074 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_1075 = seg_base + (local_tid ^ 4u); let tmp_1076 = smem_keys[tmp_1075 * WPT + 0u]; let tmp_1077 = smem_vals[tmp_1075 * WPT + 0u]; let tmp_1078 = keys[0] < tmp_1076 || (keys[0] == tmp_1076 && values[0] < tmp_1077); if tmp_1074 == tmp_1078 { keys[0] = tmp_1076; values[0] = tmp_1077; } let tmp_1079 = smem_keys[tmp_1075 * WPT + 1u]; let tmp_1080 = smem_vals[tmp_1075 * WPT + 1u]; let tmp_1081 = keys[1] < tmp_1079 || (keys[1] == tmp_1079 && values[1] < tmp_1080); if tmp_1074 == tmp_1081 { keys[1] = tmp_1079; values[1] = tmp_1080; } let tmp_1082 = smem_keys[tmp_1075 * WPT + 2u]; let tmp_1083 = smem_vals[tmp_1075 * WPT + 2u]; let tmp_1084 = keys[2] < tmp_1082 || (keys[2] == tmp_1082 && values[2] < tmp_1083); if tmp_1074 == tmp_1084 { keys[2] = tmp_1082; values[2] = tmp_1083; } let tmp_1085 = smem_keys[tmp_1075 * WPT + 3u]; let tmp_1086 = smem_vals[tmp_1075 * WPT + 3u]; let tmp_1087 = keys[3] < tmp_1085 || (keys[3] == tmp_1085 && values[3] < tmp_1086); if tmp_1074 == tmp_1087 { keys[3] = tmp_1085; values[3] = tmp_1086; } let tmp_1088 = smem_keys[tmp_1075 * WPT + 4u]; let tmp_1089 = smem_vals[tmp_1075 * WPT + 4u]; let tmp_1090 = keys[4] < tmp_1088 || (keys[4] == tmp_1088 && values[4] < tmp_1089); if tmp_1074 == tmp_1090 { keys[4] = tmp_1088; values[4] = tmp_1089; } let tmp_1091 = smem_keys[tmp_1075 * WPT + 5u]; let tmp_1092 = smem_vals[tmp_1075 * WPT + 5u]; let tmp_1093 = keys[5] < tmp_1091 || (keys[5] == tmp_1091 && values[5] < tmp_1092); if tmp_1074 == tmp_1093 { keys[5] = tmp_1091; values[5] = tmp_1092; } let tmp_1094 = smem_keys[tmp_1075 * WPT + 6u]; let tmp_1095 = smem_vals[tmp_1075 * WPT + 6u]; let tmp_1096 = keys[6] < tmp_1094 || (keys[6] == tmp_1094 && values[6] < tmp_1095); if tmp_1074 == tmp_1096 { keys[6] = tmp_1094; values[6] = tmp_1095; } let tmp_1097 = smem_keys[tmp_1075 * WPT + 7u]; let tmp_1098 = smem_vals[tmp_1075 * WPT + 7u]; let tmp_1099 = keys[7] < tmp_1097 || (keys[7] == tmp_1097 && values[7] < tmp_1098); if tmp_1074 == tmp_1099 { keys[7] = tmp_1097; values[7] = tmp_1098; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_1100 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_1101 = seg_base + (local_tid ^ 2u); let tmp_1102 = smem_keys[tmp_1101 * WPT + 0u]; let tmp_1103 = smem_vals[tmp_1101 * WPT + 0u]; let tmp_1104 = keys[0] < tmp_1102 || (keys[0] == tmp_1102 && values[0] < tmp_1103); if tmp_1100 == tmp_1104 { keys[0] = tmp_1102; values[0] = tmp_1103; } let tmp_1105 = smem_keys[tmp_1101 * WPT + 1u]; let tmp_1106 = smem_vals[tmp_1101 * WPT + 1u]; let tmp_1107 = keys[1] < tmp_1105 || (keys[1] == tmp_1105 && values[1] < tmp_1106); if tmp_1100 == tmp_1107 { keys[1] = tmp_1105; values[1] = tmp_1106; } let tmp_1108 = smem_keys[tmp_1101 * WPT + 2u]; let tmp_1109 = smem_vals[tmp_1101 * WPT + 2u]; let tmp_1110 = keys[2] < tmp_1108 || (keys[2] == tmp_1108 && values[2] < tmp_1109); if tmp_1100 == tmp_1110 { keys[2] = tmp_1108; values[2] = tmp_1109; } let tmp_1111 = smem_keys[tmp_1101 * WPT + 3u]; let tmp_1112 = smem_vals[tmp_1101 * WPT + 3u]; let tmp_1113 = keys[3] < tmp_1111 || (keys[3] == tmp_1111 && values[3] < tmp_1112); if tmp_1100 == tmp_1113 { keys[3] = tmp_1111; values[3] = tmp_1112; } let tmp_1114 = smem_keys[tmp_1101 * WPT + 4u]; let tmp_1115 = smem_vals[tmp_1101 * WPT + 4u]; let tmp_1116 = keys[4] < tmp_1114 || (keys[4] == tmp_1114 && values[4] < tmp_1115); if tmp_1100 == tmp_1116 { keys[4] = tmp_1114; values[4] = tmp_1115; } let tmp_1117 = smem_keys[tmp_1101 * WPT + 5u]; let tmp_1118 = smem_vals[tmp_1101 * WPT + 5u]; let tmp_1119 = keys[5] < tmp_1117 || (keys[5] == tmp_1117 && values[5] < tmp_1118); if tmp_1100 == tmp_1119 { keys[5] = tmp_1117; values[5] = tmp_1118; } let tmp_1120 = smem_keys[tmp_1101 * WPT + 6u]; let tmp_1121 = smem_vals[tmp_1101 * WPT + 6u]; let tmp_1122 = keys[6] < tmp_1120 || (keys[6] == tmp_1120 && values[6] < tmp_1121); if tmp_1100 == tmp_1122 { keys[6] = tmp_1120; values[6] = tmp_1121; } let tmp_1123 = smem_keys[tmp_1101 * WPT + 7u]; let tmp_1124 = smem_vals[tmp_1101 * WPT + 7u]; let tmp_1125 = keys[7] < tmp_1123 || (keys[7] == tmp_1123 && values[7] < tmp_1124); if tmp_1100 == tmp_1125 { keys[7] = tmp_1123; values[7] = tmp_1124; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:8) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; smem_keys[tid_g * WPT + 2u] = keys[2]; smem_vals[tid_g * WPT + 2u] = values[2]; smem_keys[tid_g * WPT + 3u] = keys[3]; smem_vals[tid_g * WPT + 3u] = values[3]; smem_keys[tid_g * WPT + 4u] = keys[4]; smem_vals[tid_g * WPT + 4u] = values[4]; smem_keys[tid_g * WPT + 5u] = keys[5]; smem_vals[tid_g * WPT + 5u] = values[5]; smem_keys[tid_g * WPT + 6u] = keys[6]; smem_vals[tid_g * WPT + 6u] = values[6]; smem_keys[tid_g * WPT + 7u] = keys[7]; smem_vals[tid_g * WPT + 7u] = values[7]; workgroupBarrier(); let tmp_1126 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_1127 = seg_base + (local_tid ^ 1u); let tmp_1128 = smem_keys[tmp_1127 * WPT + 0u]; let tmp_1129 = smem_vals[tmp_1127 * WPT + 0u]; let tmp_1130 = keys[0] < tmp_1128 || (keys[0] == tmp_1128 && values[0] < tmp_1129); if tmp_1126 == tmp_1130 { keys[0] = tmp_1128; values[0] = tmp_1129; } let tmp_1131 = smem_keys[tmp_1127 * WPT + 1u]; let tmp_1132 = smem_vals[tmp_1127 * WPT + 1u]; let tmp_1133 = keys[1] < tmp_1131 || (keys[1] == tmp_1131 && values[1] < tmp_1132); if tmp_1126 == tmp_1133 { keys[1] = tmp_1131; values[1] = tmp_1132; } let tmp_1134 = smem_keys[tmp_1127 * WPT + 2u]; let tmp_1135 = smem_vals[tmp_1127 * WPT + 2u]; let tmp_1136 = keys[2] < tmp_1134 || (keys[2] == tmp_1134 && values[2] < tmp_1135); if tmp_1126 == tmp_1136 { keys[2] = tmp_1134; values[2] = tmp_1135; } let tmp_1137 = smem_keys[tmp_1127 * WPT + 3u]; let tmp_1138 = smem_vals[tmp_1127 * WPT + 3u]; let tmp_1139 = keys[3] < tmp_1137 || (keys[3] == tmp_1137 && values[3] < tmp_1138); if tmp_1126 == tmp_1139 { keys[3] = tmp_1137; values[3] = tmp_1138; } let tmp_1140 = smem_keys[tmp_1127 * WPT + 4u]; let tmp_1141 = smem_vals[tmp_1127 * WPT + 4u]; let tmp_1142 = keys[4] < tmp_1140 || (keys[4] == tmp_1140 && values[4] < tmp_1141); if tmp_1126 == tmp_1142 { keys[4] = tmp_1140; values[4] = tmp_1141; } let tmp_1143 = smem_keys[tmp_1127 * WPT + 5u]; let tmp_1144 = smem_vals[tmp_1127 * WPT + 5u]; let tmp_1145 = keys[5] < tmp_1143 || (keys[5] == tmp_1143 && values[5] < tmp_1144); if tmp_1126 == tmp_1145 { keys[5] = tmp_1143; values[5] = tmp_1144; } let tmp_1146 = smem_keys[tmp_1127 * WPT + 6u]; let tmp_1147 = smem_vals[tmp_1127 * WPT + 6u]; let tmp_1148 = keys[6] < tmp_1146 || (keys[6] == tmp_1146 && values[6] < tmp_1147); if tmp_1126 == tmp_1148 { keys[6] = tmp_1146; values[6] = tmp_1147; } let tmp_1149 = smem_keys[tmp_1127 * WPT + 7u]; let tmp_1150 = smem_vals[tmp_1127 * WPT + 7u]; let tmp_1151 = keys[7] < tmp_1149 || (keys[7] == tmp_1149 && values[7] < tmp_1150); if tmp_1126 == tmp_1151 { keys[7] = tmp_1149; values[7] = tmp_1150; } workgroupBarrier(); }
    // exch_local(4,8) 
    // cmp_swap(0,4)
    if keys[0] > keys[4] || (keys[0] == keys[4] && values[0] > values[4]) {
    // swap(0,4) 
    { let tmp_1152 = keys[0]; keys[0] = keys[4]; keys[4] = tmp_1152;let tmp_1153 = values[0]; values[0] = values[4]; values[4] = tmp_1153; }
    }
    // cmp_swap(1,5)
    if keys[1] > keys[5] || (keys[1] == keys[5] && values[1] > values[5]) {
    // swap(1,5) 
    { let tmp_1154 = keys[1]; keys[1] = keys[5]; keys[5] = tmp_1154;let tmp_1155 = values[1]; values[1] = values[5]; values[5] = tmp_1155; }
    }
    // cmp_swap(2,6)
    if keys[2] > keys[6] || (keys[2] == keys[6] && values[2] > values[6]) {
    // swap(2,6) 
    { let tmp_1156 = keys[2]; keys[2] = keys[6]; keys[6] = tmp_1156;let tmp_1157 = values[2]; values[2] = values[6]; values[6] = tmp_1157; }
    }
    // cmp_swap(3,7)
    if keys[3] > keys[7] || (keys[3] == keys[7] && values[3] > values[7]) {
    // swap(3,7) 
    { let tmp_1158 = keys[3]; keys[3] = keys[7]; keys[7] = tmp_1158;let tmp_1159 = values[3]; values[3] = values[7]; values[7] = tmp_1159; }
    }
    // exch_local(2,8) 
    // cmp_swap(0,2)
    if keys[0] > keys[2] || (keys[0] == keys[2] && values[0] > values[2]) {
    // swap(0,2) 
    { let tmp_1160 = keys[0]; keys[0] = keys[2]; keys[2] = tmp_1160;let tmp_1161 = values[0]; values[0] = values[2]; values[2] = tmp_1161; }
    }
    // cmp_swap(1,3)
    if keys[1] > keys[3] || (keys[1] == keys[3] && values[1] > values[3]) {
    // swap(1,3) 
    { let tmp_1162 = keys[1]; keys[1] = keys[3]; keys[3] = tmp_1162;let tmp_1163 = values[1]; values[1] = values[3]; values[3] = tmp_1163; }
    }
    // cmp_swap(4,6)
    if keys[4] > keys[6] || (keys[4] == keys[6] && values[4] > values[6]) {
    // swap(4,6) 
    { let tmp_1164 = keys[4]; keys[4] = keys[6]; keys[6] = tmp_1164;let tmp_1165 = values[4]; values[4] = values[6]; values[6] = tmp_1165; }
    }
    // cmp_swap(5,7)
    if keys[5] > keys[7] || (keys[5] == keys[7] && values[5] > values[7]) {
    // swap(5,7) 
    { let tmp_1166 = keys[5]; keys[5] = keys[7]; keys[7] = tmp_1166;let tmp_1167 = values[5]; values[5] = values[7]; values[7] = tmp_1167; }
    }
    // exch_local(1,8) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_1168 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_1168;let tmp_1169 = values[0]; values[0] = values[1]; values[1] = tmp_1169; }
    }
    // cmp_swap(2,3)
    if keys[2] > keys[3] || (keys[2] == keys[3] && values[2] > values[3]) {
    // swap(2,3) 
    { let tmp_1170 = keys[2]; keys[2] = keys[3]; keys[3] = tmp_1170;let tmp_1171 = values[2]; values[2] = values[3]; values[3] = tmp_1171; }
    }
    // cmp_swap(4,5)
    if keys[4] > keys[5] || (keys[4] == keys[5] && values[4] > values[5]) {
    // swap(4,5) 
    { let tmp_1172 = keys[4]; keys[4] = keys[5]; keys[5] = tmp_1172;let tmp_1173 = values[4]; values[4] = values[5]; values[5] = tmp_1173; }
    }
    // cmp_swap(6,7)
    if keys[6] > keys[7] || (keys[6] == keys[7] && values[6] > values[7]) {
    // swap(6,7) 
    { let tmp_1174 = keys[6]; keys[6] = keys[7]; keys[7] = tmp_1174;let tmp_1175 = values[6]; values[6] = values[7]; values[7] = tmp_1175; }
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
