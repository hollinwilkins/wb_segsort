
enable subgroups;

override WG: u32 = 64u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 512u;
const M: u32 = 64u;
const WPT: u32 = 8u;
const SG: u32 = 8u;      // one CuteSort run spans SG lanes; RUN = SG*WPT = 64

var<workgroup> smem_keys: array<u32, N>;
var<workgroup> smem_vals: array<u32, N>;

fn lane_mask_lt(sid: u32) -> vec4<u32> {
    var m = vec4<u32>(0u, 0u, 0u, 0u);
    if (sid >= 32u) { m.x = 0xffffffffu; } else { m.x = (1u << sid) - 1u; }
    if (sid >= 64u) { m.y = 0xffffffffu; } else if (sid > 32u) { m.y = (1u << (sid - 32u)) - 1u; }
    if (sid >= 96u) { m.z = 0xffffffffu; } else if (sid > 64u) { m.z = (1u << (sid - 64u)) - 1u; }
    if (sid >= 128u) { m.w = 0xffffffffu; } else if (sid > 96u) { m.w = (1u << (sid - 96u)) - 1u; }
    return m;
}

fn ballot_popc(v: vec4<u32>) -> u32 {
    let c = countOneBits(v);
    return c.x + c.y + c.z + c.w;
}

@compute @workgroup_size(WG, 1, 1)
fn segsort_cutemerge_sg8_smem16k_n512_m64_block(
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 9u;

    let bin_base = select(bin_offsets[BIN - 1u], 0u, BIN == 0u);
    let bin_count = bin_offsets[BIN] - bin_base;

    let local_tid = tid_g % M;
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

    // phase 1 (CuteSort): each subgroup sorts its RUN = SG*WPT elements.
    let sub_block = (tid_g / SG) * SG * WPT;   // this subgroup's runs live here
    let seg_lane_base = sid - (sid % SG);
    let bin_mask = lane_mask_lt(seg_lane_base + SG) & ~lane_mask_lt(seg_lane_base);

    {  // cute run for register slot 0
        let key_k = keys[0];
        var ge_mask = lane_mask_lt(sid);
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            let is_zero = (key_k & (1u << bit)) == 0u;
            let ballot0 = subgroupBallot(is_zero);
            ge_mask = select(ge_mask | ballot0, ge_mask & ballot0, is_zero);
        }
        let rank = ballot_popc(ge_mask & bin_mask);
        smem_keys[sub_block + 0u * SG + rank] = key_k;
        smem_vals[sub_block + 0u * SG + rank] = values[0];
    }
    {  // cute run for register slot 1
        let key_k = keys[1];
        var ge_mask = lane_mask_lt(sid);
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            let is_zero = (key_k & (1u << bit)) == 0u;
            let ballot0 = subgroupBallot(is_zero);
            ge_mask = select(ge_mask | ballot0, ge_mask & ballot0, is_zero);
        }
        let rank = ballot_popc(ge_mask & bin_mask);
        smem_keys[sub_block + 1u * SG + rank] = key_k;
        smem_vals[sub_block + 1u * SG + rank] = values[1];
    }
    {  // cute run for register slot 2
        let key_k = keys[2];
        var ge_mask = lane_mask_lt(sid);
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            let is_zero = (key_k & (1u << bit)) == 0u;
            let ballot0 = subgroupBallot(is_zero);
            ge_mask = select(ge_mask | ballot0, ge_mask & ballot0, is_zero);
        }
        let rank = ballot_popc(ge_mask & bin_mask);
        smem_keys[sub_block + 2u * SG + rank] = key_k;
        smem_vals[sub_block + 2u * SG + rank] = values[2];
    }
    {  // cute run for register slot 3
        let key_k = keys[3];
        var ge_mask = lane_mask_lt(sid);
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            let is_zero = (key_k & (1u << bit)) == 0u;
            let ballot0 = subgroupBallot(is_zero);
            ge_mask = select(ge_mask | ballot0, ge_mask & ballot0, is_zero);
        }
        let rank = ballot_popc(ge_mask & bin_mask);
        smem_keys[sub_block + 3u * SG + rank] = key_k;
        smem_vals[sub_block + 3u * SG + rank] = values[3];
    }
    {  // cute run for register slot 4
        let key_k = keys[4];
        var ge_mask = lane_mask_lt(sid);
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            let is_zero = (key_k & (1u << bit)) == 0u;
            let ballot0 = subgroupBallot(is_zero);
            ge_mask = select(ge_mask | ballot0, ge_mask & ballot0, is_zero);
        }
        let rank = ballot_popc(ge_mask & bin_mask);
        smem_keys[sub_block + 4u * SG + rank] = key_k;
        smem_vals[sub_block + 4u * SG + rank] = values[4];
    }
    {  // cute run for register slot 5
        let key_k = keys[5];
        var ge_mask = lane_mask_lt(sid);
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            let is_zero = (key_k & (1u << bit)) == 0u;
            let ballot0 = subgroupBallot(is_zero);
            ge_mask = select(ge_mask | ballot0, ge_mask & ballot0, is_zero);
        }
        let rank = ballot_popc(ge_mask & bin_mask);
        smem_keys[sub_block + 5u * SG + rank] = key_k;
        smem_vals[sub_block + 5u * SG + rank] = values[5];
    }
    {  // cute run for register slot 6
        let key_k = keys[6];
        var ge_mask = lane_mask_lt(sid);
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            let is_zero = (key_k & (1u << bit)) == 0u;
            let ballot0 = subgroupBallot(is_zero);
            ge_mask = select(ge_mask | ballot0, ge_mask & ballot0, is_zero);
        }
        let rank = ballot_popc(ge_mask & bin_mask);
        smem_keys[sub_block + 6u * SG + rank] = key_k;
        smem_vals[sub_block + 6u * SG + rank] = values[6];
    }
    {  // cute run for register slot 7
        let key_k = keys[7];
        var ge_mask = lane_mask_lt(sid);
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            let is_zero = (key_k & (1u << bit)) == 0u;
            let ballot0 = subgroupBallot(is_zero);
            ge_mask = select(ge_mask | ballot0, ge_mask & ballot0, is_zero);
        }
        let rank = ballot_popc(ge_mask & bin_mask);
        smem_keys[sub_block + 7u * SG + rank] = key_k;
        smem_vals[sub_block + 7u * SG + rank] = values[7];
    }
    workgroupBarrier();

    let base = local_tid * WPT;   // this thread's blocked output range [base, base+WPT)

    // merge pass: two sorted runs of 8 -> 16 (register-staged)
    {
        let group_base = (base / 16u) * 16u;
        let diag = base - group_base;
        let a_base = group_base;
        let b_base = group_base + 8u;
        // merge-path: binary search the diagonal for this thread's A/B split
        var lo = select(0u, diag - 8u, diag > 8u);
        var hi = min(diag, 8u);
        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            let ak = smem_keys[a_base + mid];
            let av = smem_vals[a_base + mid];
            let bpos = b_base + (diag - 1u - mid);
            let bk = smem_keys[bpos];
            let bv = smem_vals[bpos];
            if ak < bk || (ak == bk && av <= bv) { lo = mid + 1u; } else { hi = mid; }
        }
        var ai = lo;
        var bi = diag - lo;
        // merge this thread's WPT outputs into registers (the pong)
        var out_keys: array<u32, 8>;
        var out_vals: array<u32, 8>;
        for (var k = 0u; k < WPT; k = k + 1u) {
            let take_a = bi >= 8u || (ai < 8u &&
                (smem_keys[a_base + ai] < smem_keys[b_base + bi] ||
                 (smem_keys[a_base + ai] == smem_keys[b_base + bi] &&
                  smem_vals[a_base + ai] <= smem_vals[b_base + bi])));
            if take_a {
                out_keys[k] = smem_keys[a_base + ai];
                out_vals[k] = smem_vals[a_base + ai];
                ai = ai + 1u;
            } else {
                out_keys[k] = smem_keys[b_base + bi];
                out_vals[k] = smem_vals[b_base + bi];
                bi = bi + 1u;
            }
        }
        workgroupBarrier();   // every read is done before any write-back
        storageBarrier();     // device-scope fence: workgroupBarrier alone under-orders
                              // the in-place write-back for single-SIMD-group WGs
        for (var k = 0u; k < WPT; k = k + 1u) {
            smem_keys[base + k] = out_keys[k];
            smem_vals[base + k] = out_vals[k];
        }
    }
    workgroupBarrier();
    // merge pass: two sorted runs of 16 -> 32 (register-staged)
    {
        let group_base = (base / 32u) * 32u;
        let diag = base - group_base;
        let a_base = group_base;
        let b_base = group_base + 16u;
        // merge-path: binary search the diagonal for this thread's A/B split
        var lo = select(0u, diag - 16u, diag > 16u);
        var hi = min(diag, 16u);
        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            let ak = smem_keys[a_base + mid];
            let av = smem_vals[a_base + mid];
            let bpos = b_base + (diag - 1u - mid);
            let bk = smem_keys[bpos];
            let bv = smem_vals[bpos];
            if ak < bk || (ak == bk && av <= bv) { lo = mid + 1u; } else { hi = mid; }
        }
        var ai = lo;
        var bi = diag - lo;
        // merge this thread's WPT outputs into registers (the pong)
        var out_keys: array<u32, 8>;
        var out_vals: array<u32, 8>;
        for (var k = 0u; k < WPT; k = k + 1u) {
            let take_a = bi >= 16u || (ai < 16u &&
                (smem_keys[a_base + ai] < smem_keys[b_base + bi] ||
                 (smem_keys[a_base + ai] == smem_keys[b_base + bi] &&
                  smem_vals[a_base + ai] <= smem_vals[b_base + bi])));
            if take_a {
                out_keys[k] = smem_keys[a_base + ai];
                out_vals[k] = smem_vals[a_base + ai];
                ai = ai + 1u;
            } else {
                out_keys[k] = smem_keys[b_base + bi];
                out_vals[k] = smem_vals[b_base + bi];
                bi = bi + 1u;
            }
        }
        workgroupBarrier();   // every read is done before any write-back
        storageBarrier();     // device-scope fence: workgroupBarrier alone under-orders
                              // the in-place write-back for single-SIMD-group WGs
        for (var k = 0u; k < WPT; k = k + 1u) {
            smem_keys[base + k] = out_keys[k];
            smem_vals[base + k] = out_vals[k];
        }
    }
    workgroupBarrier();
    // merge pass: two sorted runs of 32 -> 64 (register-staged)
    {
        let group_base = (base / 64u) * 64u;
        let diag = base - group_base;
        let a_base = group_base;
        let b_base = group_base + 32u;
        // merge-path: binary search the diagonal for this thread's A/B split
        var lo = select(0u, diag - 32u, diag > 32u);
        var hi = min(diag, 32u);
        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            let ak = smem_keys[a_base + mid];
            let av = smem_vals[a_base + mid];
            let bpos = b_base + (diag - 1u - mid);
            let bk = smem_keys[bpos];
            let bv = smem_vals[bpos];
            if ak < bk || (ak == bk && av <= bv) { lo = mid + 1u; } else { hi = mid; }
        }
        var ai = lo;
        var bi = diag - lo;
        // merge this thread's WPT outputs into registers (the pong)
        var out_keys: array<u32, 8>;
        var out_vals: array<u32, 8>;
        for (var k = 0u; k < WPT; k = k + 1u) {
            let take_a = bi >= 32u || (ai < 32u &&
                (smem_keys[a_base + ai] < smem_keys[b_base + bi] ||
                 (smem_keys[a_base + ai] == smem_keys[b_base + bi] &&
                  smem_vals[a_base + ai] <= smem_vals[b_base + bi])));
            if take_a {
                out_keys[k] = smem_keys[a_base + ai];
                out_vals[k] = smem_vals[a_base + ai];
                ai = ai + 1u;
            } else {
                out_keys[k] = smem_keys[b_base + bi];
                out_vals[k] = smem_vals[b_base + bi];
                bi = bi + 1u;
            }
        }
        workgroupBarrier();   // every read is done before any write-back
        storageBarrier();     // device-scope fence: workgroupBarrier alone under-orders
                              // the in-place write-back for single-SIMD-group WGs
        for (var k = 0u; k < WPT; k = k + 1u) {
            smem_keys[base + k] = out_keys[k];
            smem_vals[base + k] = out_vals[k];
        }
    }
    workgroupBarrier();
    // merge pass: two sorted runs of 64 -> 128 (register-staged)
    {
        let group_base = (base / 128u) * 128u;
        let diag = base - group_base;
        let a_base = group_base;
        let b_base = group_base + 64u;
        // merge-path: binary search the diagonal for this thread's A/B split
        var lo = select(0u, diag - 64u, diag > 64u);
        var hi = min(diag, 64u);
        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            let ak = smem_keys[a_base + mid];
            let av = smem_vals[a_base + mid];
            let bpos = b_base + (diag - 1u - mid);
            let bk = smem_keys[bpos];
            let bv = smem_vals[bpos];
            if ak < bk || (ak == bk && av <= bv) { lo = mid + 1u; } else { hi = mid; }
        }
        var ai = lo;
        var bi = diag - lo;
        // merge this thread's WPT outputs into registers (the pong)
        var out_keys: array<u32, 8>;
        var out_vals: array<u32, 8>;
        for (var k = 0u; k < WPT; k = k + 1u) {
            let take_a = bi >= 64u || (ai < 64u &&
                (smem_keys[a_base + ai] < smem_keys[b_base + bi] ||
                 (smem_keys[a_base + ai] == smem_keys[b_base + bi] &&
                  smem_vals[a_base + ai] <= smem_vals[b_base + bi])));
            if take_a {
                out_keys[k] = smem_keys[a_base + ai];
                out_vals[k] = smem_vals[a_base + ai];
                ai = ai + 1u;
            } else {
                out_keys[k] = smem_keys[b_base + bi];
                out_vals[k] = smem_vals[b_base + bi];
                bi = bi + 1u;
            }
        }
        workgroupBarrier();   // every read is done before any write-back
        storageBarrier();     // device-scope fence: workgroupBarrier alone under-orders
                              // the in-place write-back for single-SIMD-group WGs
        for (var k = 0u; k < WPT; k = k + 1u) {
            smem_keys[base + k] = out_keys[k];
            smem_vals[base + k] = out_vals[k];
        }
    }
    workgroupBarrier();
    // merge pass: two sorted runs of 128 -> 256 (register-staged)
    {
        let group_base = (base / 256u) * 256u;
        let diag = base - group_base;
        let a_base = group_base;
        let b_base = group_base + 128u;
        // merge-path: binary search the diagonal for this thread's A/B split
        var lo = select(0u, diag - 128u, diag > 128u);
        var hi = min(diag, 128u);
        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            let ak = smem_keys[a_base + mid];
            let av = smem_vals[a_base + mid];
            let bpos = b_base + (diag - 1u - mid);
            let bk = smem_keys[bpos];
            let bv = smem_vals[bpos];
            if ak < bk || (ak == bk && av <= bv) { lo = mid + 1u; } else { hi = mid; }
        }
        var ai = lo;
        var bi = diag - lo;
        // merge this thread's WPT outputs into registers (the pong)
        var out_keys: array<u32, 8>;
        var out_vals: array<u32, 8>;
        for (var k = 0u; k < WPT; k = k + 1u) {
            let take_a = bi >= 128u || (ai < 128u &&
                (smem_keys[a_base + ai] < smem_keys[b_base + bi] ||
                 (smem_keys[a_base + ai] == smem_keys[b_base + bi] &&
                  smem_vals[a_base + ai] <= smem_vals[b_base + bi])));
            if take_a {
                out_keys[k] = smem_keys[a_base + ai];
                out_vals[k] = smem_vals[a_base + ai];
                ai = ai + 1u;
            } else {
                out_keys[k] = smem_keys[b_base + bi];
                out_vals[k] = smem_vals[b_base + bi];
                bi = bi + 1u;
            }
        }
        workgroupBarrier();   // every read is done before any write-back
        storageBarrier();     // device-scope fence: workgroupBarrier alone under-orders
                              // the in-place write-back for single-SIMD-group WGs
        for (var k = 0u; k < WPT; k = k + 1u) {
            smem_keys[base + k] = out_keys[k];
            smem_vals[base + k] = out_vals[k];
        }
    }
    workgroupBarrier();
    // merge pass: two sorted runs of 256 -> 512 (register-staged)
    {
        let group_base = (base / 512u) * 512u;
        let diag = base - group_base;
        let a_base = group_base;
        let b_base = group_base + 256u;
        // merge-path: binary search the diagonal for this thread's A/B split
        var lo = select(0u, diag - 256u, diag > 256u);
        var hi = min(diag, 256u);
        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            let ak = smem_keys[a_base + mid];
            let av = smem_vals[a_base + mid];
            let bpos = b_base + (diag - 1u - mid);
            let bk = smem_keys[bpos];
            let bv = smem_vals[bpos];
            if ak < bk || (ak == bk && av <= bv) { lo = mid + 1u; } else { hi = mid; }
        }
        var ai = lo;
        var bi = diag - lo;
        // merge this thread's WPT outputs into registers (the pong)
        var out_keys: array<u32, 8>;
        var out_vals: array<u32, 8>;
        for (var k = 0u; k < WPT; k = k + 1u) {
            let take_a = bi >= 256u || (ai < 256u &&
                (smem_keys[a_base + ai] < smem_keys[b_base + bi] ||
                 (smem_keys[a_base + ai] == smem_keys[b_base + bi] &&
                  smem_vals[a_base + ai] <= smem_vals[b_base + bi])));
            if take_a {
                out_keys[k] = smem_keys[a_base + ai];
                out_vals[k] = smem_vals[a_base + ai];
                ai = ai + 1u;
            } else {
                out_keys[k] = smem_keys[b_base + bi];
                out_vals[k] = smem_vals[b_base + bi];
                bi = bi + 1u;
            }
        }
        workgroupBarrier();   // every read is done before any write-back
        storageBarrier();     // device-scope fence: workgroupBarrier alone under-orders
                              // the in-place write-back for single-SIMD-group WGs
        for (var k = 0u; k < WPT; k = k + 1u) {
            smem_keys[base + k] = out_keys[k];
            smem_vals[base + k] = out_vals[k];
        }
    }
    workgroupBarrier();
    for (var r = 0u; r < WPT; r = r + 1u) {
        let pos = local_tid * WPT + r;
        if is_active && pos < seg_size {
            global_keys[seg_start + pos] = smem_keys[pos];
            global_value_indices[seg_start + pos] = smem_vals[pos];
        }
    }
}
