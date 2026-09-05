
enable subgroups;

override WG: u32 = 256u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = 1024u;
const M: u32 = 256u;
const WPT: u32 = 4u;
const SG: u32 = 64u;      // one CuteSort run spans SG lanes; RUN = SG*WPT = 256

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
fn segsort_cutemerge_sg64_smem16k_n1024_m256_striped(
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    const BIN: u32 = 10u;

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

    // phase 1 (CuteSort): each subgroup sorts its RUN = SG*WPT elements.
    let sub_block = (tid_g / SG) * SG * WPT;   // this subgroup's runs live here

    {  // CuteSort wide run 0: 2 slot(s) x 64 lanes -> sorted run of 128
        let rbase = sub_block + 0u;
        var ge_0 = lane_mask_lt(0u + sid);
        var ge_1 = lane_mask_lt(64u + sid);
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            let bal_0 = subgroupBallot((keys[0] & (1u << bit)) == 0u);
            let bal_1 = subgroupBallot((keys[1] & (1u << bit)) == 0u);
            var zmask = vec4<u32>(0u, 0u, 0u, 0u);
            zmask.x = bal_0.x;
            zmask.y = bal_0.y;
            zmask.z = bal_1.x;
            zmask.w = bal_1.y;
            let isz_0 = (keys[0] & (1u << bit)) == 0u;
            ge_0 = select(ge_0 | zmask, ge_0 & zmask, isz_0);
            let isz_1 = (keys[1] & (1u << bit)) == 0u;
            ge_1 = select(ge_1 | zmask, ge_1 & zmask, isz_1);
        }
        let r_0 = ballot_popc(ge_0);
        smem_keys[rbase + r_0] = keys[0];
        smem_vals[rbase + r_0] = values[0];
        let r_1 = ballot_popc(ge_1);
        smem_keys[rbase + r_1] = keys[1];
        smem_vals[rbase + r_1] = values[1];
    }
    {  // CuteSort wide run 1: 2 slot(s) x 64 lanes -> sorted run of 128
        let rbase = sub_block + 128u;
        var ge_0 = lane_mask_lt(0u + sid);
        var ge_1 = lane_mask_lt(64u + sid);
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            let bal_0 = subgroupBallot((keys[2] & (1u << bit)) == 0u);
            let bal_1 = subgroupBallot((keys[3] & (1u << bit)) == 0u);
            var zmask = vec4<u32>(0u, 0u, 0u, 0u);
            zmask.x = bal_0.x;
            zmask.y = bal_0.y;
            zmask.z = bal_1.x;
            zmask.w = bal_1.y;
            let isz_0 = (keys[2] & (1u << bit)) == 0u;
            ge_0 = select(ge_0 | zmask, ge_0 & zmask, isz_0);
            let isz_1 = (keys[3] & (1u << bit)) == 0u;
            ge_1 = select(ge_1 | zmask, ge_1 & zmask, isz_1);
        }
        let r_0 = ballot_popc(ge_0);
        smem_keys[rbase + r_0] = keys[2];
        smem_vals[rbase + r_0] = values[2];
        let r_1 = ballot_popc(ge_1);
        smem_keys[rbase + r_1] = keys[3];
        smem_vals[rbase + r_1] = values[3];
    }
    workgroupBarrier();

    let base = local_tid * WPT;   // this thread's blocked output range [base, base+WPT)

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
        var out_keys: array<u32, 4>;
        var out_vals: array<u32, 4>;
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
        var out_keys: array<u32, 4>;
        var out_vals: array<u32, 4>;
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
    // merge pass: two sorted runs of 512 -> 1024 (register-staged)
    {
        let group_base = (base / 1024u) * 1024u;
        let diag = base - group_base;
        let a_base = group_base;
        let b_base = group_base + 512u;
        // merge-path: binary search the diagonal for this thread's A/B split
        var lo = select(0u, diag - 512u, diag > 512u);
        var hi = min(diag, 512u);
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
        var out_keys: array<u32, 4>;
        var out_vals: array<u32, 4>;
        for (var k = 0u; k < WPT; k = k + 1u) {
            let take_a = bi >= 512u || (ai < 512u &&
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
    for (var c = 0u; c < WPT; c = c + 1u) {
        let j = c * M + local_tid;
        if is_active && j < seg_size {
            global_keys[seg_start + j] = smem_keys[j];
            global_value_indices[seg_start + j] = smem_vals[j];
        }
    }
}
