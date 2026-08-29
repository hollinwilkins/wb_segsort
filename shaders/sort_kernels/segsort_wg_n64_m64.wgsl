
override WG: u32 = 64u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 64u;
const M: u32 = 64u;
const WPT: u32 = 1u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n64_m64(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    const BIN: u32 = 6u

    let bin_base = select(bin_offsets[BIN - 1u], 0u, BIN == 0u);
    let bin_count = bin_offsets[BIN] - bin_base;

    let local_tid = tid_g % M;
    let seg_base = tid_g - local_tid;
    let global_seg = (wg_id.x * WG + tid_g) / M;

    let active = global_seg < bin_count;
    let slot = bin_base + select(0u, global_seg, active);   // clamp so the read is in-range
    let seg_id = bin_indices[slot];
    let seg_start = select(segments[seg_id - 1u], 0u, seg_id == 0u);
    let seg_end = segments[seg_id];
    let seg_size = select(0u, seg_end - seg_start, active);

    var keys: array<u32, 1>;
    var values: array<u32, 1>;

    for (var r = 0u; r < WPT; r = r + 1u) {
        let pos = local_tid * WPT + r;
        if active && pos < seg_size {
            keys[r] = global_keys[seg_start + pos];
            values[r] = seg_start + pos;
        } else {
            keys[r] = 0xffffffffu;
            values[r] = 0xffffffffu;
        }
    }

    // exch_intxn(tmask:1,swbit:0,wpt:1)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_0 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_1 = seg_base + (local_tid ^ 1u); let tmp_2 = smem_keys[tmp_1 * WPT + 0u]; let tmp_3 = smem_vals[tmp_1 * WPT + 0u]; let tmp_4 = keys[0] < tmp_2 || (keys[0] == tmp_2 && values[0] < tmp_3); if tmp_0 == tmp_4 { keys[0] = tmp_2; values[0] = tmp_3; } workgroupBarrier(); }
    // exch_intxn(tmask:3,swbit:1,wpt:1)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_5 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_6 = seg_base + (local_tid ^ 3u); let tmp_7 = smem_keys[tmp_6 * WPT + 0u]; let tmp_8 = smem_vals[tmp_6 * WPT + 0u]; let tmp_9 = keys[0] < tmp_7 || (keys[0] == tmp_7 && values[0] < tmp_8); if tmp_5 == tmp_9 { keys[0] = tmp_7; values[0] = tmp_8; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:1) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_10 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_11 = seg_base + (local_tid ^ 1u); let tmp_12 = smem_keys[tmp_11 * WPT + 0u]; let tmp_13 = smem_vals[tmp_11 * WPT + 0u]; let tmp_14 = keys[0] < tmp_12 || (keys[0] == tmp_12 && values[0] < tmp_13); if tmp_10 == tmp_14 { keys[0] = tmp_12; values[0] = tmp_13; } workgroupBarrier(); }
    // exch_intxn(tmask:7,swbit:2,wpt:1)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_15 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_16 = seg_base + (local_tid ^ 7u); let tmp_17 = smem_keys[tmp_16 * WPT + 0u]; let tmp_18 = smem_vals[tmp_16 * WPT + 0u]; let tmp_19 = keys[0] < tmp_17 || (keys[0] == tmp_17 && values[0] < tmp_18); if tmp_15 == tmp_19 { keys[0] = tmp_17; values[0] = tmp_18; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:1) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_20 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_21 = seg_base + (local_tid ^ 2u); let tmp_22 = smem_keys[tmp_21 * WPT + 0u]; let tmp_23 = smem_vals[tmp_21 * WPT + 0u]; let tmp_24 = keys[0] < tmp_22 || (keys[0] == tmp_22 && values[0] < tmp_23); if tmp_20 == tmp_24 { keys[0] = tmp_22; values[0] = tmp_23; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:1) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_25 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_26 = seg_base + (local_tid ^ 1u); let tmp_27 = smem_keys[tmp_26 * WPT + 0u]; let tmp_28 = smem_vals[tmp_26 * WPT + 0u]; let tmp_29 = keys[0] < tmp_27 || (keys[0] == tmp_27 && values[0] < tmp_28); if tmp_25 == tmp_29 { keys[0] = tmp_27; values[0] = tmp_28; } workgroupBarrier(); }
    // exch_intxn(tmask:15,swbit:3,wpt:1)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_30 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_31 = seg_base + (local_tid ^ 15u); let tmp_32 = smem_keys[tmp_31 * WPT + 0u]; let tmp_33 = smem_vals[tmp_31 * WPT + 0u]; let tmp_34 = keys[0] < tmp_32 || (keys[0] == tmp_32 && values[0] < tmp_33); if tmp_30 == tmp_34 { keys[0] = tmp_32; values[0] = tmp_33; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:1) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_35 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_36 = seg_base + (local_tid ^ 4u); let tmp_37 = smem_keys[tmp_36 * WPT + 0u]; let tmp_38 = smem_vals[tmp_36 * WPT + 0u]; let tmp_39 = keys[0] < tmp_37 || (keys[0] == tmp_37 && values[0] < tmp_38); if tmp_35 == tmp_39 { keys[0] = tmp_37; values[0] = tmp_38; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:1) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_40 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_41 = seg_base + (local_tid ^ 2u); let tmp_42 = smem_keys[tmp_41 * WPT + 0u]; let tmp_43 = smem_vals[tmp_41 * WPT + 0u]; let tmp_44 = keys[0] < tmp_42 || (keys[0] == tmp_42 && values[0] < tmp_43); if tmp_40 == tmp_44 { keys[0] = tmp_42; values[0] = tmp_43; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:1) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_45 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_46 = seg_base + (local_tid ^ 1u); let tmp_47 = smem_keys[tmp_46 * WPT + 0u]; let tmp_48 = smem_vals[tmp_46 * WPT + 0u]; let tmp_49 = keys[0] < tmp_47 || (keys[0] == tmp_47 && values[0] < tmp_48); if tmp_45 == tmp_49 { keys[0] = tmp_47; values[0] = tmp_48; } workgroupBarrier(); }
    // exch_intxn(tmask:31,swbit:4,wpt:1)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_50 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_51 = seg_base + (local_tid ^ 31u); let tmp_52 = smem_keys[tmp_51 * WPT + 0u]; let tmp_53 = smem_vals[tmp_51 * WPT + 0u]; let tmp_54 = keys[0] < tmp_52 || (keys[0] == tmp_52 && values[0] < tmp_53); if tmp_50 == tmp_54 { keys[0] = tmp_52; values[0] = tmp_53; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:1) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_55 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_56 = seg_base + (local_tid ^ 8u); let tmp_57 = smem_keys[tmp_56 * WPT + 0u]; let tmp_58 = smem_vals[tmp_56 * WPT + 0u]; let tmp_59 = keys[0] < tmp_57 || (keys[0] == tmp_57 && values[0] < tmp_58); if tmp_55 == tmp_59 { keys[0] = tmp_57; values[0] = tmp_58; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:1) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_60 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_61 = seg_base + (local_tid ^ 4u); let tmp_62 = smem_keys[tmp_61 * WPT + 0u]; let tmp_63 = smem_vals[tmp_61 * WPT + 0u]; let tmp_64 = keys[0] < tmp_62 || (keys[0] == tmp_62 && values[0] < tmp_63); if tmp_60 == tmp_64 { keys[0] = tmp_62; values[0] = tmp_63; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:1) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_65 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_66 = seg_base + (local_tid ^ 2u); let tmp_67 = smem_keys[tmp_66 * WPT + 0u]; let tmp_68 = smem_vals[tmp_66 * WPT + 0u]; let tmp_69 = keys[0] < tmp_67 || (keys[0] == tmp_67 && values[0] < tmp_68); if tmp_65 == tmp_69 { keys[0] = tmp_67; values[0] = tmp_68; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:1) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_70 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_71 = seg_base + (local_tid ^ 1u); let tmp_72 = smem_keys[tmp_71 * WPT + 0u]; let tmp_73 = smem_vals[tmp_71 * WPT + 0u]; let tmp_74 = keys[0] < tmp_72 || (keys[0] == tmp_72 && values[0] < tmp_73); if tmp_70 == tmp_74 { keys[0] = tmp_72; values[0] = tmp_73; } workgroupBarrier(); }
    // exch_intxn(tmask:63,swbit:5,wpt:1)
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_75 = extractBits(local_tid, 5u, 1u) != 0u; let tmp_76 = seg_base + (local_tid ^ 63u); let tmp_77 = smem_keys[tmp_76 * WPT + 0u]; let tmp_78 = smem_vals[tmp_76 * WPT + 0u]; let tmp_79 = keys[0] < tmp_77 || (keys[0] == tmp_77 && values[0] < tmp_78); if tmp_75 == tmp_79 { keys[0] = tmp_77; values[0] = tmp_78; } workgroupBarrier(); }
    // exch_paral(tmask:16,swbit:4,wpt:1) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_80 = extractBits(local_tid, 4u, 1u) != 0u; let tmp_81 = seg_base + (local_tid ^ 16u); let tmp_82 = smem_keys[tmp_81 * WPT + 0u]; let tmp_83 = smem_vals[tmp_81 * WPT + 0u]; let tmp_84 = keys[0] < tmp_82 || (keys[0] == tmp_82 && values[0] < tmp_83); if tmp_80 == tmp_84 { keys[0] = tmp_82; values[0] = tmp_83; } workgroupBarrier(); }
    // exch_paral(tmask:8,swbit:3,wpt:1) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_85 = extractBits(local_tid, 3u, 1u) != 0u; let tmp_86 = seg_base + (local_tid ^ 8u); let tmp_87 = smem_keys[tmp_86 * WPT + 0u]; let tmp_88 = smem_vals[tmp_86 * WPT + 0u]; let tmp_89 = keys[0] < tmp_87 || (keys[0] == tmp_87 && values[0] < tmp_88); if tmp_85 == tmp_89 { keys[0] = tmp_87; values[0] = tmp_88; } workgroupBarrier(); }
    // exch_paral(tmask:4,swbit:2,wpt:1) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_90 = extractBits(local_tid, 2u, 1u) != 0u; let tmp_91 = seg_base + (local_tid ^ 4u); let tmp_92 = smem_keys[tmp_91 * WPT + 0u]; let tmp_93 = smem_vals[tmp_91 * WPT + 0u]; let tmp_94 = keys[0] < tmp_92 || (keys[0] == tmp_92 && values[0] < tmp_93); if tmp_90 == tmp_94 { keys[0] = tmp_92; values[0] = tmp_93; } workgroupBarrier(); }
    // exch_paral(tmask:2,swbit:1,wpt:1) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_95 = extractBits(local_tid, 1u, 1u) != 0u; let tmp_96 = seg_base + (local_tid ^ 2u); let tmp_97 = smem_keys[tmp_96 * WPT + 0u]; let tmp_98 = smem_vals[tmp_96 * WPT + 0u]; let tmp_99 = keys[0] < tmp_97 || (keys[0] == tmp_97 && values[0] < tmp_98); if tmp_95 == tmp_99 { keys[0] = tmp_97; values[0] = tmp_98; } workgroupBarrier(); }
    // exch_paral(tmask:1,swbit:0,wpt:1) 
    { smem_keys[tid_g * WPT + 0u] = keys[0]; smem_vals[tid_g * WPT + 0u] = values[0]; workgroupBarrier(); let tmp_100 = extractBits(local_tid, 0u, 1u) != 0u; let tmp_101 = seg_base + (local_tid ^ 1u); let tmp_102 = smem_keys[tmp_101 * WPT + 0u]; let tmp_103 = smem_vals[tmp_101 * WPT + 0u]; let tmp_104 = keys[0] < tmp_102 || (keys[0] == tmp_102 && values[0] < tmp_103); if tmp_100 == tmp_104 { keys[0] = tmp_102; values[0] = tmp_103; } workgroupBarrier(); }

    // blocked store
    for (var r = 0u; r < WPT; r = r + 1u) {
        let pos = local_tid * WPT + r;
        if active && pos < seg_size {
            global_keys[seg_start + pos] = keys[r];
            global_value_indices[seg_start + pos] = values[r];
        }
    }
}
