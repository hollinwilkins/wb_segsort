
override WG: u32 = 4u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 4u;
const M: u32 = 4u;
const WPT: u32 = 1u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n4_m4(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    const BIN: u32 = 2u

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

    // blocked store
    for (var r = 0u; r < WPT; r = r + 1u) {
        let pos = local_tid * WPT + r;
        if active && pos < seg_size {
            global_keys[seg_start + pos] = keys[r];
            global_value_indices[seg_start + pos] = values[r];
        }
    }
}
