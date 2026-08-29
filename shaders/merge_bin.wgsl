override WORKGROUP_SIZE_X: u32 = 16;
override WORKGROUP_SIZE_Y: u32 = 16;
override WORKGROUP_ITEMS: u32 = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y;

struct Config {
    segments_len: u32,
}

struct DispatchSize {
    x: u32,
    y: u32,
    z: u32
}

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> segments: array<u32>;
@group(0) @binding(2) var<storage, read> bin_workgroup_size: array<u32>;
@group(0) @binding(3) var<storage, read_write> bin_histogram: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> bin_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> bin_indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> dispatch: array<DispatchSize>;

var <workgroup> local_bin_counts: array<atomic<u32>, 13>;

fn segment_bucket(index: u32) -> u32 {
    let segment_end = segments[index];
    var segment_start = 0u;
    if index > 0u {
        segment_start = segments[index - 1u];
    }
    let segment_len = segment_end - segment_start;

    let bucket_segment_len = max(segment_len, 1u) - 1u;
    return min(32u - countLeadingZeros(bucket_segment_len), 12u);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main_histogram(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) workgroup_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let ARRAY_LENGTH = config.segments_len;
    let WORKGROUP_ID = workgroup_id.x + workgroup_id.y * workgroup_dim.x;
    let WID = WORKGROUP_ID * WORKGROUP_ITEMS;
    let GID = WID + TID; // Global thread ID

    if TID < 13 {
        atomicStore(&local_bin_counts[TID], 0u);
    }

    workgroupBarrier();

    if GID < ARRAY_LENGTH {
        atomicAdd(&local_bin_counts[segment_bucket(GID)], 1u);
    }

    workgroupBarrier();

    if (TID < 13) {
        atomicAdd(&bin_histogram[TID], local_bin_counts[TID]);
    }
}

@compute @workgroup_size(1, 1, 1)
fn main_schedule() {
    var sum = 0u;
    for (var i = 0u; i < 13; i++) {
        let cur_count = atomicLoad(&bin_histogram[i]);
        let end = cur_count + sum;

        atomicStore(&bin_histogram[i], end);
        bin_offsets[i] = end;
        sum = sum + cur_count;

        let groups = select(
            (cur_count + bin_workgroup_size[i] - 1u) / bin_workgroup_size[i],
            0u,
            i == 0u
        );
        var x = groups;
        var y = 1u;
        if x > 65535u {
            y = (groups + 65535u - 1u) / 65535u;
            x = 65535u;
        }

        dispatch[i] = DispatchSize(x, y, 1u);
    }
}

// Group segment indices in order for dispatch
// After calling this, all indices will be grouped and in order by segment size ascending
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main_group(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) workgroup_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let ARRAY_LENGTH = config.segments_len;
    let WORKGROUP_ID = workgroup_id.x + workgroup_id.y * workgroup_dim.x;
    let WID = WORKGROUP_ID * WORKGROUP_ITEMS;
    let GID = WID + TID; // Global thread ID

    if GID < ARRAY_LENGTH {
        let pos = atomicSub(&bin_histogram[segment_bucket(GID)], 1u) - 1u;
        bin_indices[pos] = GID;
    }
}
