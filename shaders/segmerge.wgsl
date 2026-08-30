override INPUT_TILE_SIZE: u32 = 2048u;
override TILE_SIZE: u32 = 2048u;
override WG: u32 = 256u;

struct TileInfo {
    seg_start: u32,
    seg_size: u32,
    offset: u32
}

struct TileMeta {
    tile_count: u32,
    max_size: u32
}

@group(0) @binding(0) var<storage, read> keys_in:  array<u32>;
@group(0) @binding(1) var<storage, read> value_indices_in:  array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> value_indices_out: array<u32>;
@group(0) @binding(4) var<storage, read> tiles: array<TileInfo>;
@group(0) @binding(5) var<storage, read> meta: TileMeta;

fn less(ak: u32, av: u32, bk: u32, bv: u32) -> bool {
    return ak < bk || (ak == bk && av < bv);
}

fn merge_path(a_base: u32, a_len: u32, b_base: u32, b_len: u32, diag: u32) -> u32 {
    var begin = select(0u, diag - b_len, diag > b_len);
    var end = min(diag, a_len);

    while (begin < end) {
        let mid = (begin + end) >> 1u;
        let ak = keys_in[a_base + mid];
        let av = value_indices_in[a_base + mid];
        let bidx = diag - 1u - mid;
        let bk = keys_in[b_base + bidx];
        let bv = value_indices_in[b_base + bidx];
        if (!less(bk, bv, ak, av)) { begin = mid + 1u; } else { end = mid; }
    }

    return begin;
}

@compute @workgroup_size(WG, 1, 1)
fn segmerge(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    let GID = wg_id.x;
    if (GID >= meta.tile_count) { return; }

    let wg_els = TILE_SIZE / WG;

    let info = tiles[GID];
    let seg_start = info.seg_start;
    let size = info.seg_size;
    let offset = info.offset;

    let w2 = 2u * INPUT_TILE_SIZE;
    let pair_base = (offset / w2) * w2;
    let a_start = seg_start + pair_base;
    let a_end = seg_start + min(pair_base + INPUT_TILE_SIZE, size);
    let b_start = a_end;
    let b_end = seg_start + min(pair_base + w2, size);
    let a_len = a_end - a_start;
    let b_len = b_end - b_start;
    let pair_tot = a_len + b_len;

    let out_lo = offset - pair_base;
    let out_hi = min(out_lo + TILE_SIZE, pair_tot);

    let base_out = out_lo + tid * wg_els;
    if (base_out >= out_hi) { return; }

    let ai = merge_path(a_start, a_len, b_start, b_len, base_out);
    var ap = ai;
    var bp = base_out - ai;

    for (var i = 0u; i < wg_els; i = i + 1u) {
        let d = base_out + i;

        if (d >= out_hi) { break; }

        let take_a = (bp >= b_len) ||
            (ap < a_len && less(keys_in[a_start + ap], value_indices_in[a_start + ap],
                                keys_in[b_start + bp], value_indices_in[b_start + bp]));

        let dst = seg_start + pair_base + d;

        if (take_a) {
            keys_out[dst] = keys_in[a_start + ap];
            value_indices_out[dst] = value_indices_in[a_start + ap];
            ap = ap + 1u;
        } else {
            keys_out[dst] = keys_in[b_start + bp];
            value_indices_out[dst] = value_indices_in[b_start + bp];
            bp = bp + 1u;
        }
    }
}
