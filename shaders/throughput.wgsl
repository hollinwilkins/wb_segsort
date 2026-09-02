override WG: u32 = 256u;

struct Params {
    n: u32,
    delta: u32,
}

@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> vals: array<u32>;   // value indices (sort2_vals only)

fn linear_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * (nwg.x * WG);
}

@compute @workgroup_size(WG, 1, 1)
fn main_throughput(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = linear_id(gid, nwg);
    if (i < params.n) {
        buf[i] = buf[i] + params.delta;
    }
}

@compute @workgroup_size(WG, 1, 1)
fn main_throughput_add(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = linear_id(gid, nwg);
    if (i < params.n) {
        buf[i] = buf[i] + params.delta + 1u;
    }
}

@compute @workgroup_size(WG, 1, 1)
fn main_sort2(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = linear_id(gid, nwg) * 2u;
    if (i + 1u < params.n) {
        let a = buf[i];
        let b = buf[i + 1u];
        buf[i] = min(a, b);
        buf[i + 1u] = max(a, b);
    }
}

@compute @workgroup_size(WG, 1, 1)
fn main_sort2_vals(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = linear_id(gid, nwg) * 2u;
    if (i + 1u < params.n) {
        let a = buf[i];
        let b = buf[i + 1u];
        let swap = b < a;
        buf[i] = select(a, b, swap);
        buf[i + 1u] = select(b, a, swap);
        vals[i] = select(i, i + 1u, swap);
        vals[i + 1u] = select(i + 1u, i, swap);
    }
}
