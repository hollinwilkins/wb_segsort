override TILE_SIZE: u32 = 2048u;
override WG: u32 = 256u;
override MAX_PASSES: u32 = 22u;

struct TileMeta {
    tile_count: atomic<u32>,
    max_size: atomic<u32>
}

struct TileInfo {
    seg_start: u32,
    seg_size: u32,
    offset: u32
}

struct DispatchSize {
    x: u32,
    y: u32,
    z: u32
}

@group(0) @binding(0) var<storage, read> segments: array<u32>;
@group(0) @binding(1) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> bin_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> tiles: array<TileInfo>;
@group(0) @binding(4) var<storage, read_write> meta: TileMeta;
@group(0) @binding(5) var<storage, read_write> dispatch_tilesort: DispatchSize;
@group(0) @binding(6) var<storage, read_write> dispatch_merge: array<DispatchSize>;

@compute @workgroup_size(WG, 1, 1)
fn main_build_tiles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = bin_offsets[11];
    let count = bin_offsets[12] - base;
    let GID = gid.x;

    if (GID < count) {
        let seg_id = bin_indices[base + GID];
        let seg_start = select(segments[seg_id - 1u], 0u, seg_id == 0u);
        let seg_end = segments[seg_id];
        let size = seg_end - seg_start;

        let tile_count = (size + TILE_SIZE - 1u) / TILE_SIZE;
        let tile_base = atomicAdd(&meta.tile_count, tile_count);
        atomicMax(&meta.max_size, size);

        for (var i = 0u; i < tile_count; i = i + 1u) {
            tiles[tile_base + i] = TileInfo(seg_start, size, i * TILE_SIZE);
        }
    }
}

fn dispatch_for_len(n: u32) -> DispatchSize {
    var x = n;
    var y = 1u;
    if (x > 65535u) {
        y = (n + 65535u - 1u) / 65535u;
        x = 65535u;
    }

    return DispatchSize(x, y, 1u);
}

@compute @workgroup_size(1, 1, 1)
fn main_merge_schedule() {
    let tile_count = atomicLoad(&meta.tile_count);
    let max_size = atomicLoad(&meta.max_size);

    let tile_dispatch = dispatch_for_len(tile_count);
    dispatch_tilesort = select(
        tile_dispatch,
        DispatchSize(0u, 0u, 0u),
        tile_count == 0u
    );

    var pass_count = 0u;
    var w = TILE_SIZE;
    while (w < max_size) {
        pass_count = pass_count + 1u;
        w = w << 1u;
    }

    if ((pass_count & 1u) == 1u) { pass_count = pass_count + 1u; }

    for (var i = 0u; i < MAX_PASSES; i = i + 1u) {
        dispatch_merge[i] = select(
            DispatchSize(0u, 0u, 0u),
            tile_dispatch,
            i < pass_count && tile_count > 0u
        );
    }
}
