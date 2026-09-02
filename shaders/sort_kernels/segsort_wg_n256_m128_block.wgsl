
override WG: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 256u;
const M: u32 = 128u;
const WPT: u32 = 2u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n256_m128_block(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 8u;

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
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_2 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_3 = seg_base + (local_tid ^ 1u); let tmp_4 = smem_keys[tmp_3 * WPT + 1u]; let tmp_5 = smem_vals[tmp_3 * WPT + 1u]; let tmp_6 = keys[0] < tmp_4 || (keys[0] == tmp_4 && values[0] < tmp_5); if tmp_2 == tmp_6 { keys[0] = tmp_4; values[0] = tmp_5; } let tmp_7 = smem_keys[tmp_3 * WPT + 0u]; let tmp_8 = smem_vals[tmp_3 * WPT + 0u]; let tmp_9 = keys[1] < tmp_7 || (keys[1] == tmp_7 && values[1] < tmp_8); if tmp_2 == tmp_9 { keys[1] = tmp_7; values[1] = tmp_8; } workgroupBarrier(); }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_10 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_10;let tmp_11 = values[0]; values[0] = values[1]; values[1] = tmp_11; }
    }
    // exch_intxn(tmask:3,swbit:1,wpt:2)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_12 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_13 = seg_base + (local_tid ^ 3u); let tmp_14 = smem_keys[tmp_13 * WPT + 1u]; let tmp_15 = smem_vals[tmp_13 * WPT + 1u]; let tmp_16 = keys[0] < tmp_14 || (keys[0] == tmp_14 && values[0] < tmp_15); if tmp_12 == tmp_16 { keys[0] = tmp_14; values[0] = tmp_15; } let tmp_17 = smem_keys[tmp_13 * WPT + 0u]; let tmp_18 = smem_vals[tmp_13 * WPT + 0u]; let tmp_19 = keys[1] < tmp_17 || (keys[1] == tmp_17 && values[1] < tmp_18); if tmp_12 == tmp_19 { keys[1] = tmp_17; values[1] = tmp_18; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_20 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_21 = seg_base + (local_tid ^ 1u); let tmp_22 = smem_keys[tmp_21 * WPT + 0u]; let tmp_23 = smem_vals[tmp_21 * WPT + 0u]; let tmp_24 = keys[0] < tmp_22 || (keys[0] == tmp_22 && values[0] < tmp_23); if tmp_20 == tmp_24 { keys[0] = tmp_22; values[0] = tmp_23; } let tmp_25 = smem_keys[tmp_21 * WPT + 1u]; let tmp_26 = smem_vals[tmp_21 * WPT + 1u]; let tmp_27 = keys[1] < tmp_25 || (keys[1] == tmp_25 && values[1] < tmp_26); if tmp_20 == tmp_27 { keys[1] = tmp_25; values[1] = tmp_26; } workgroupBarrier(); }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_28 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_28;let tmp_29 = values[0]; values[0] = values[1]; values[1] = tmp_29; }
    }
    // exch_intxn(tmask:7,swbit:2,wpt:2)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_30 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_31 = seg_base + (local_tid ^ 7u); let tmp_32 = smem_keys[tmp_31 * WPT + 1u]; let tmp_33 = smem_vals[tmp_31 * WPT + 1u]; let tmp_34 = keys[0] < tmp_32 || (keys[0] == tmp_32 && values[0] < tmp_33); if tmp_30 == tmp_34 { keys[0] = tmp_32; values[0] = tmp_33; } let tmp_35 = smem_keys[tmp_31 * WPT + 0u]; let tmp_36 = smem_vals[tmp_31 * WPT + 0u]; let tmp_37 = keys[1] < tmp_35 || (keys[1] == tmp_35 && values[1] < tmp_36); if tmp_30 == tmp_37 { keys[1] = tmp_35; values[1] = tmp_36; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_38 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_39 = seg_base + (local_tid ^ 2u); let tmp_40 = smem_keys[tmp_39 * WPT + 0u]; let tmp_41 = smem_vals[tmp_39 * WPT + 0u]; let tmp_42 = keys[0] < tmp_40 || (keys[0] == tmp_40 && values[0] < tmp_41); if tmp_38 == tmp_42 { keys[0] = tmp_40; values[0] = tmp_41; } let tmp_43 = smem_keys[tmp_39 * WPT + 1u]; let tmp_44 = smem_vals[tmp_39 * WPT + 1u]; let tmp_45 = keys[1] < tmp_43 || (keys[1] == tmp_43 && values[1] < tmp_44); if tmp_38 == tmp_45 { keys[1] = tmp_43; values[1] = tmp_44; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_46 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_47 = seg_base + (local_tid ^ 1u); let tmp_48 = smem_keys[tmp_47 * WPT + 0u]; let tmp_49 = smem_vals[tmp_47 * WPT + 0u]; let tmp_50 = keys[0] < tmp_48 || (keys[0] == tmp_48 && values[0] < tmp_49); if tmp_46 == tmp_50 { keys[0] = tmp_48; values[0] = tmp_49; } let tmp_51 = smem_keys[tmp_47 * WPT + 1u]; let tmp_52 = smem_vals[tmp_47 * WPT + 1u]; let tmp_53 = keys[1] < tmp_51 || (keys[1] == tmp_51 && values[1] < tmp_52); if tmp_46 == tmp_53 { keys[1] = tmp_51; values[1] = tmp_52; } workgroupBarrier(); }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_54 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_54;let tmp_55 = values[0]; values[0] = values[1]; values[1] = tmp_55; }
    }
    // exch_intxn(tmask:15,swbit:3,wpt:2)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_56 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_57 = seg_base + (local_tid ^ 15u); let tmp_58 = smem_keys[tmp_57 * WPT + 1u]; let tmp_59 = smem_vals[tmp_57 * WPT + 1u]; let tmp_60 = keys[0] < tmp_58 || (keys[0] == tmp_58 && values[0] < tmp_59); if tmp_56 == tmp_60 { keys[0] = tmp_58; values[0] = tmp_59; } let tmp_61 = smem_keys[tmp_57 * WPT + 0u]; let tmp_62 = smem_vals[tmp_57 * WPT + 0u]; let tmp_63 = keys[1] < tmp_61 || (keys[1] == tmp_61 && values[1] < tmp_62); if tmp_56 == tmp_63 { keys[1] = tmp_61; values[1] = tmp_62; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_64 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_65 = seg_base + (local_tid ^ 4u); let tmp_66 = smem_keys[tmp_65 * WPT + 0u]; let tmp_67 = smem_vals[tmp_65 * WPT + 0u]; let tmp_68 = keys[0] < tmp_66 || (keys[0] == tmp_66 && values[0] < tmp_67); if tmp_64 == tmp_68 { keys[0] = tmp_66; values[0] = tmp_67; } let tmp_69 = smem_keys[tmp_65 * WPT + 1u]; let tmp_70 = smem_vals[tmp_65 * WPT + 1u]; let tmp_71 = keys[1] < tmp_69 || (keys[1] == tmp_69 && values[1] < tmp_70); if tmp_64 == tmp_71 { keys[1] = tmp_69; values[1] = tmp_70; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_72 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_73 = seg_base + (local_tid ^ 2u); let tmp_74 = smem_keys[tmp_73 * WPT + 0u]; let tmp_75 = smem_vals[tmp_73 * WPT + 0u]; let tmp_76 = keys[0] < tmp_74 || (keys[0] == tmp_74 && values[0] < tmp_75); if tmp_72 == tmp_76 { keys[0] = tmp_74; values[0] = tmp_75; } let tmp_77 = smem_keys[tmp_73 * WPT + 1u]; let tmp_78 = smem_vals[tmp_73 * WPT + 1u]; let tmp_79 = keys[1] < tmp_77 || (keys[1] == tmp_77 && values[1] < tmp_78); if tmp_72 == tmp_79 { keys[1] = tmp_77; values[1] = tmp_78; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_80 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_81 = seg_base + (local_tid ^ 1u); let tmp_82 = smem_keys[tmp_81 * WPT + 0u]; let tmp_83 = smem_vals[tmp_81 * WPT + 0u]; let tmp_84 = keys[0] < tmp_82 || (keys[0] == tmp_82 && values[0] < tmp_83); if tmp_80 == tmp_84 { keys[0] = tmp_82; values[0] = tmp_83; } let tmp_85 = smem_keys[tmp_81 * WPT + 1u]; let tmp_86 = smem_vals[tmp_81 * WPT + 1u]; let tmp_87 = keys[1] < tmp_85 || (keys[1] == tmp_85 && values[1] < tmp_86); if tmp_80 == tmp_87 { keys[1] = tmp_85; values[1] = tmp_86; } workgroupBarrier(); }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_88 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_88;let tmp_89 = values[0]; values[0] = values[1]; values[1] = tmp_89; }
    }
    // exch_intxn(tmask:31,swbit:4,wpt:2)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_90 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_91 = seg_base + (local_tid ^ 31u); let tmp_92 = smem_keys[tmp_91 * WPT + 1u]; let tmp_93 = smem_vals[tmp_91 * WPT + 1u]; let tmp_94 = keys[0] < tmp_92 || (keys[0] == tmp_92 && values[0] < tmp_93); if tmp_90 == tmp_94 { keys[0] = tmp_92; values[0] = tmp_93; } let tmp_95 = smem_keys[tmp_91 * WPT + 0u]; let tmp_96 = smem_vals[tmp_91 * WPT + 0u]; let tmp_97 = keys[1] < tmp_95 || (keys[1] == tmp_95 && values[1] < tmp_96); if tmp_90 == tmp_97 { keys[1] = tmp_95; values[1] = tmp_96; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_98 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_99 = seg_base + (local_tid ^ 8u); let tmp_100 = smem_keys[tmp_99 * WPT + 0u]; let tmp_101 = smem_vals[tmp_99 * WPT + 0u]; let tmp_102 = keys[0] < tmp_100 || (keys[0] == tmp_100 && values[0] < tmp_101); if tmp_98 == tmp_102 { keys[0] = tmp_100; values[0] = tmp_101; } let tmp_103 = smem_keys[tmp_99 * WPT + 1u]; let tmp_104 = smem_vals[tmp_99 * WPT + 1u]; let tmp_105 = keys[1] < tmp_103 || (keys[1] == tmp_103 && values[1] < tmp_104); if tmp_98 == tmp_105 { keys[1] = tmp_103; values[1] = tmp_104; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_106 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_107 = seg_base + (local_tid ^ 4u); let tmp_108 = smem_keys[tmp_107 * WPT + 0u]; let tmp_109 = smem_vals[tmp_107 * WPT + 0u]; let tmp_110 = keys[0] < tmp_108 || (keys[0] == tmp_108 && values[0] < tmp_109); if tmp_106 == tmp_110 { keys[0] = tmp_108; values[0] = tmp_109; } let tmp_111 = smem_keys[tmp_107 * WPT + 1u]; let tmp_112 = smem_vals[tmp_107 * WPT + 1u]; let tmp_113 = keys[1] < tmp_111 || (keys[1] == tmp_111 && values[1] < tmp_112); if tmp_106 == tmp_113 { keys[1] = tmp_111; values[1] = tmp_112; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_114 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_115 = seg_base + (local_tid ^ 2u); let tmp_116 = smem_keys[tmp_115 * WPT + 0u]; let tmp_117 = smem_vals[tmp_115 * WPT + 0u]; let tmp_118 = keys[0] < tmp_116 || (keys[0] == tmp_116 && values[0] < tmp_117); if tmp_114 == tmp_118 { keys[0] = tmp_116; values[0] = tmp_117; } let tmp_119 = smem_keys[tmp_115 * WPT + 1u]; let tmp_120 = smem_vals[tmp_115 * WPT + 1u]; let tmp_121 = keys[1] < tmp_119 || (keys[1] == tmp_119 && values[1] < tmp_120); if tmp_114 == tmp_121 { keys[1] = tmp_119; values[1] = tmp_120; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_122 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_123 = seg_base + (local_tid ^ 1u); let tmp_124 = smem_keys[tmp_123 * WPT + 0u]; let tmp_125 = smem_vals[tmp_123 * WPT + 0u]; let tmp_126 = keys[0] < tmp_124 || (keys[0] == tmp_124 && values[0] < tmp_125); if tmp_122 == tmp_126 { keys[0] = tmp_124; values[0] = tmp_125; } let tmp_127 = smem_keys[tmp_123 * WPT + 1u]; let tmp_128 = smem_vals[tmp_123 * WPT + 1u]; let tmp_129 = keys[1] < tmp_127 || (keys[1] == tmp_127 && values[1] < tmp_128); if tmp_122 == tmp_129 { keys[1] = tmp_127; values[1] = tmp_128; } workgroupBarrier(); }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_130 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_130;let tmp_131 = values[0]; values[0] = values[1]; values[1] = tmp_131; }
    }
    // exch_intxn(tmask:63,swbit:5,wpt:2)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_132 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_133 = seg_base + (local_tid ^ 63u); let tmp_134 = smem_keys[tmp_133 * WPT + 1u]; let tmp_135 = smem_vals[tmp_133 * WPT + 1u]; let tmp_136 = keys[0] < tmp_134 || (keys[0] == tmp_134 && values[0] < tmp_135); if tmp_132 == tmp_136 { keys[0] = tmp_134; values[0] = tmp_135; } let tmp_137 = smem_keys[tmp_133 * WPT + 0u]; let tmp_138 = smem_vals[tmp_133 * WPT + 0u]; let tmp_139 = keys[1] < tmp_137 || (keys[1] == tmp_137 && values[1] < tmp_138); if tmp_132 == tmp_139 { keys[1] = tmp_137; values[1] = tmp_138; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_140 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_141 = seg_base + (local_tid ^ 16u); let tmp_142 = smem_keys[tmp_141 * WPT + 0u]; let tmp_143 = smem_vals[tmp_141 * WPT + 0u]; let tmp_144 = keys[0] < tmp_142 || (keys[0] == tmp_142 && values[0] < tmp_143); if tmp_140 == tmp_144 { keys[0] = tmp_142; values[0] = tmp_143; } let tmp_145 = smem_keys[tmp_141 * WPT + 1u]; let tmp_146 = smem_vals[tmp_141 * WPT + 1u]; let tmp_147 = keys[1] < tmp_145 || (keys[1] == tmp_145 && values[1] < tmp_146); if tmp_140 == tmp_147 { keys[1] = tmp_145; values[1] = tmp_146; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_148 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_149 = seg_base + (local_tid ^ 8u); let tmp_150 = smem_keys[tmp_149 * WPT + 0u]; let tmp_151 = smem_vals[tmp_149 * WPT + 0u]; let tmp_152 = keys[0] < tmp_150 || (keys[0] == tmp_150 && values[0] < tmp_151); if tmp_148 == tmp_152 { keys[0] = tmp_150; values[0] = tmp_151; } let tmp_153 = smem_keys[tmp_149 * WPT + 1u]; let tmp_154 = smem_vals[tmp_149 * WPT + 1u]; let tmp_155 = keys[1] < tmp_153 || (keys[1] == tmp_153 && values[1] < tmp_154); if tmp_148 == tmp_155 { keys[1] = tmp_153; values[1] = tmp_154; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_156 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_157 = seg_base + (local_tid ^ 4u); let tmp_158 = smem_keys[tmp_157 * WPT + 0u]; let tmp_159 = smem_vals[tmp_157 * WPT + 0u]; let tmp_160 = keys[0] < tmp_158 || (keys[0] == tmp_158 && values[0] < tmp_159); if tmp_156 == tmp_160 { keys[0] = tmp_158; values[0] = tmp_159; } let tmp_161 = smem_keys[tmp_157 * WPT + 1u]; let tmp_162 = smem_vals[tmp_157 * WPT + 1u]; let tmp_163 = keys[1] < tmp_161 || (keys[1] == tmp_161 && values[1] < tmp_162); if tmp_156 == tmp_163 { keys[1] = tmp_161; values[1] = tmp_162; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_164 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_165 = seg_base + (local_tid ^ 2u); let tmp_166 = smem_keys[tmp_165 * WPT + 0u]; let tmp_167 = smem_vals[tmp_165 * WPT + 0u]; let tmp_168 = keys[0] < tmp_166 || (keys[0] == tmp_166 && values[0] < tmp_167); if tmp_164 == tmp_168 { keys[0] = tmp_166; values[0] = tmp_167; } let tmp_169 = smem_keys[tmp_165 * WPT + 1u]; let tmp_170 = smem_vals[tmp_165 * WPT + 1u]; let tmp_171 = keys[1] < tmp_169 || (keys[1] == tmp_169 && values[1] < tmp_170); if tmp_164 == tmp_171 { keys[1] = tmp_169; values[1] = tmp_170; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_172 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_173 = seg_base + (local_tid ^ 1u); let tmp_174 = smem_keys[tmp_173 * WPT + 0u]; let tmp_175 = smem_vals[tmp_173 * WPT + 0u]; let tmp_176 = keys[0] < tmp_174 || (keys[0] == tmp_174 && values[0] < tmp_175); if tmp_172 == tmp_176 { keys[0] = tmp_174; values[0] = tmp_175; } let tmp_177 = smem_keys[tmp_173 * WPT + 1u]; let tmp_178 = smem_vals[tmp_173 * WPT + 1u]; let tmp_179 = keys[1] < tmp_177 || (keys[1] == tmp_177 && values[1] < tmp_178); if tmp_172 == tmp_179 { keys[1] = tmp_177; values[1] = tmp_178; } workgroupBarrier(); }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_180 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_180;let tmp_181 = values[0]; values[0] = values[1]; values[1] = tmp_181; }
    }
    // exch_intxn(tmask:127,swbit:6,wpt:2)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_182 = extractBits(local_tid, 6u, 1u) != 0u; let tmp_183 = seg_base + (local_tid ^ 127u); let tmp_184 = smem_keys[tmp_183 * WPT + 1u]; let tmp_185 = smem_vals[tmp_183 * WPT + 1u]; let tmp_186 = keys[0] < tmp_184 || (keys[0] == tmp_184 && values[0] < tmp_185); if tmp_182 == tmp_186 { keys[0] = tmp_184; values[0] = tmp_185; } let tmp_187 = smem_keys[tmp_183 * WPT + 0u]; let tmp_188 = smem_vals[tmp_183 * WPT + 0u]; let tmp_189 = keys[1] < tmp_187 || (keys[1] == tmp_187 && values[1] < tmp_188); if tmp_182 == tmp_189 { keys[1] = tmp_187; values[1] = tmp_188; } workgroupBarrier(); }
    // exch_paral(tmask:32,swbit:5,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_190 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_191 = seg_base + (local_tid ^ 32u); let tmp_192 = smem_keys[tmp_191 * WPT + 0u]; let tmp_193 = smem_vals[tmp_191 * WPT + 0u]; let tmp_194 = keys[0] < tmp_192 || (keys[0] == tmp_192 && values[0] < tmp_193); if tmp_190 == tmp_194 { keys[0] = tmp_192; values[0] = tmp_193; } let tmp_195 = smem_keys[tmp_191 * WPT + 1u]; let tmp_196 = smem_vals[tmp_191 * WPT + 1u]; let tmp_197 = keys[1] < tmp_195 || (keys[1] == tmp_195 && values[1] < tmp_196); if tmp_190 == tmp_197 { keys[1] = tmp_195; values[1] = tmp_196; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_198 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_199 = seg_base + (local_tid ^ 16u); let tmp_200 = smem_keys[tmp_199 * WPT + 0u]; let tmp_201 = smem_vals[tmp_199 * WPT + 0u]; let tmp_202 = keys[0] < tmp_200 || (keys[0] == tmp_200 && values[0] < tmp_201); if tmp_198 == tmp_202 { keys[0] = tmp_200; values[0] = tmp_201; } let tmp_203 = smem_keys[tmp_199 * WPT + 1u]; let tmp_204 = smem_vals[tmp_199 * WPT + 1u]; let tmp_205 = keys[1] < tmp_203 || (keys[1] == tmp_203 && values[1] < tmp_204); if tmp_198 == tmp_205 { keys[1] = tmp_203; values[1] = tmp_204; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_206 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_207 = seg_base + (local_tid ^ 8u); let tmp_208 = smem_keys[tmp_207 * WPT + 0u]; let tmp_209 = smem_vals[tmp_207 * WPT + 0u]; let tmp_210 = keys[0] < tmp_208 || (keys[0] == tmp_208 && values[0] < tmp_209); if tmp_206 == tmp_210 { keys[0] = tmp_208; values[0] = tmp_209; } let tmp_211 = smem_keys[tmp_207 * WPT + 1u]; let tmp_212 = smem_vals[tmp_207 * WPT + 1u]; let tmp_213 = keys[1] < tmp_211 || (keys[1] == tmp_211 && values[1] < tmp_212); if tmp_206 == tmp_213 { keys[1] = tmp_211; values[1] = tmp_212; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_214 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_215 = seg_base + (local_tid ^ 4u); let tmp_216 = smem_keys[tmp_215 * WPT + 0u]; let tmp_217 = smem_vals[tmp_215 * WPT + 0u]; let tmp_218 = keys[0] < tmp_216 || (keys[0] == tmp_216 && values[0] < tmp_217); if tmp_214 == tmp_218 { keys[0] = tmp_216; values[0] = tmp_217; } let tmp_219 = smem_keys[tmp_215 * WPT + 1u]; let tmp_220 = smem_vals[tmp_215 * WPT + 1u]; let tmp_221 = keys[1] < tmp_219 || (keys[1] == tmp_219 && values[1] < tmp_220); if tmp_214 == tmp_221 { keys[1] = tmp_219; values[1] = tmp_220; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_222 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_223 = seg_base + (local_tid ^ 2u); let tmp_224 = smem_keys[tmp_223 * WPT + 0u]; let tmp_225 = smem_vals[tmp_223 * WPT + 0u]; let tmp_226 = keys[0] < tmp_224 || (keys[0] == tmp_224 && values[0] < tmp_225); if tmp_222 == tmp_226 { keys[0] = tmp_224; values[0] = tmp_225; } let tmp_227 = smem_keys[tmp_223 * WPT + 1u]; let tmp_228 = smem_vals[tmp_223 * WPT + 1u]; let tmp_229 = keys[1] < tmp_227 || (keys[1] == tmp_227 && values[1] < tmp_228); if tmp_222 == tmp_229 { keys[1] = tmp_227; values[1] = tmp_228; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:2) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; smem_keys[tid_g * WPT + 1u] = keys[1]; smem_vals[tid_g * WPT + 1u] = values[1]; workgroupBarrier(); let tmp_230 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_231 = seg_base + (local_tid ^ 1u); let tmp_232 = smem_keys[tmp_231 * WPT + 0u]; let tmp_233 = smem_vals[tmp_231 * WPT + 0u]; let tmp_234 = keys[0] < tmp_232 || (keys[0] == tmp_232 && values[0] < tmp_233); if tmp_230 == tmp_234 { keys[0] = tmp_232; values[0] = tmp_233; } let tmp_235 = smem_keys[tmp_231 * WPT + 1u]; let tmp_236 = smem_vals[tmp_231 * WPT + 1u]; let tmp_237 = keys[1] < tmp_235 || (keys[1] == tmp_235 && values[1] < tmp_236); if tmp_230 == tmp_237 { keys[1] = tmp_235; values[1] = tmp_236; } workgroupBarrier(); }
    // exch_local(1,2) 
    // cmp_swap(0,1)
    if keys[0] > keys[1] || (keys[0] == keys[1] && values[0] > values[1]) {
    // swap(0,1) 
    { let tmp_238 = keys[0]; keys[0] = keys[1]; keys[1] = tmp_238;let tmp_239 = values[0]; values[0] = values[1]; values[1] = tmp_239; }
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
