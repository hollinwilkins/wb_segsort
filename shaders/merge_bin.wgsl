override WORKGROUP_SIZE_X: u32 = 16;
override WORKGROUP_SIZE_Y: u32 = 16;

const WORKGROUP_ITEMS: u32 = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y;

@group(0) @binding(0) var<storage, read> segments: array<u32>;

@group(1) @binding(0) var<storage, read_write> bin_counts: array<atomic<u32>>;

var <workgroup> local_bin_counts: array<atomic<u32>, 13>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main_histogram(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) workgroup_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_COUNT = workgroup_dim.x * workgroup_dim.y * workgroup_dim.z;
    let ARRAY_LENGTH = config.path_tags;
    let WORKGROUP_ID = workgroup_id.x + workgroup_id.y * workgroup_dim.x;
    let WID = WORKGROUP_ID * TEMPLATE_THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    if TID < 13 {
        atomicStore(&local_bin_counts[TID], 0u);
    }

    workgroupBarrier();

    var bucket_index = 0u;
    if GID < arrayLength(&segments) {
        let segment_end = segments[GID];
        var segment_start = 0u;
        if GID == 0 {
            segment_start = segments[GID - 1];
        }
        let segment_len = segment_end - segment_start;

        // 0 -> 0 and 1 -> 0, both get bucketed in bin 0
        let bucket_segment_len = max(segment_len, 1u) - 1u;
        let bucket_index = min(32u - countLeadingZeros(bucket_segment_len), 12u);

        atomicAdd(&local_bin_counts[bucket_index], 1u);
    }

    workgroupBarrier();

    if (TID < 13) {
        atomicAdd(&bin_counts[TID], local_bin_counts[TID]);
    }
}
