
override WG: u32 = 16u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 32u;
const M: u32 = 16u;
const WPT: u32 = 2u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n32_m16_block(
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

    // block store
    for (var r = 0u; r < WPT; r = r + 1u) {
        let pos = local_tid * WPT + r;
        if is_active && pos < seg_size {
            global_keys[seg_start + pos] = keys[r];
            global_value_indices[seg_start + pos] = values[r];
        }
    }
}
