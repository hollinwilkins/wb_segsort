override WG: u32 = 256u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

@compute @workgroup_size(WG, 1, 1)
fn segsort_wg_n1_m1_block(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {
    let bin_count = bin_offsets[0];

    let local_tid = tid_g;
    let seg_base = tid_g - local_tid;
    let wg_index = wg_id.x + wg_id.y * wg_dim.x;
    let global_seg = wg_index * WG + tid_g;

    if global_seg >= bin_count {
        return;
    }

    let seg_id = bin_indices[global_seg];
    let seg_start = select(segments[seg_id - 1u], 0u, seg_id == 0u);

    global_value_indices[seg_start] = seg_start;
}
