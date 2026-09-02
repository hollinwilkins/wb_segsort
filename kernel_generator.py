import sys
from math import log2
from dataclasses import dataclass


class KernelGenerator:
    _tmp_index: int

    def __init__(self, memory: str, store: str):
        self._tmp_index = 0
        self._memory = memory
        self._store = store

    def tmp(self) -> str:
        tmp_index = self._tmp_index
        self._tmp_index += 1
        return f"tmp_{tmp_index}"

    def swap(self, a: int, b: int) -> str:
        tk = self.tmp()
        tv = self.tmp()
        return (f"// swap({a},{b}) \n" +
                f"{{ let {tk} = keys[{a}]; keys[{a}] = keys[{b}]; keys[{b}] = {tk};" +
                f"let {tv} = values[{a}]; values[{a}] = values[{b}]; values[{b}] = {tv}; }}")

    def cmp_swap(self, a: int, b: int) -> str:
        return (f"// cmp_swap({a},{b})\n" +
                f"if keys[{a}] > keys[{b}] || (keys[{a}] == keys[{b}] && values[{a}] > values[{b}]) {{\n" +
                self.swap(a, b) + "\n}")

    def eql_swap(self, a: int, b: int) -> str:
        return (f"// eql_swap({a},{b}) \n" +
                f"if keys[{a}] != keys[{b}] {{\n" +
                self.swap(a, b) + "\n}")

    def exch(self, r: int, tmask: int, swbit: int) -> str:
        pk, pv, less, bit = self.tmp(), self.tmp(), self.tmp(), self.tmp()
        return (f"// exch(r:{r},tmask:{tmask},swbit:{swbit})\n" +
                f"{{\n" +
                f"  let {pk} = subgroupShuffleXor(keys[{r}], {tmask}u);\n" +
                f"  let {pv} = subgroupShuffleXor(values[{r}], {tmask}u);\n" +
                f"  let {less} = keys[{r}] < {pk} || (keys[{r}] == {pk} && values[{r}] < {pv});\n" +
                f"  let {bit} = extractBits(local_tid, {swbit}u, 1u) != 0u;\n" +
                f"  if {bit} == {less} {{ keys[{r}] = {pk}; values[{r}] = {pv}; }}\n" +
                "}}")

    def exch_local(self, rmask: int, wpt: int) -> str:
        out = []
        for i in range(wpt):
            j = i ^ rmask
            if i < j:
                out.append(self.cmp_swap(i, j))
        return (f"// exch_local({rmask},{wpt}) \n" +
                "\n".join(out))

    def _exch_subgroup(self, pairs, tmask: int, swbit: int) -> str:
        reads, lines = [], []
        for dst, src in pairs:
            pk, pv = self.tmp(), self.tmp()
            lines.append(f"let {pk} = subgroupShuffleXor(keys[{src}], {tmask}u);")
            lines.append(f"let {pv} = subgroupShuffleXor(values[{src}], {tmask}u);")
            reads.append((dst, pk, pv))
        bit = self.tmp()
        lines.append(f"let {bit} = extractBits(local_tid, {swbit}u, 1u) != 0u;")
        for dst, pk, pv in reads:
            less = self.tmp()
            lines.append(f"let {less} = keys[{dst}] < {pk} || (keys[{dst}] == {pk} && values[{dst}] < {pv});")
            lines.append(f"if {bit} == {less} {{ keys[{dst}] = {pk}; values[{dst}] = {pv}; }}")
        return "{\n" + "\n".join(lines) + "\n}"

    def _exch_workgroup(self, pairs, tmask: int, swbit: int) -> str:
        srcs = sorted({src for _, src in pairs})
        lines = []
        for s in srcs:
            lines.append(f"smem_keys[tid_g * WPT + {s}u] = keys[{s}];")
            lines.append(f"smem_vals[tid_g * WPT + {s}u] = values[{s}];")
        lines.append("workgroupBarrier();")
        bit, partner = self.tmp(), self.tmp()
        lines.append(f"let {bit} = extractBits(local_tid, {swbit}u, 1u) != 0u;")
        lines.append(f"let {partner} = seg_base + (local_tid ^ {tmask}u);")
        for dst, src in pairs:
            pk, pv, less = self.tmp(), self.tmp(), self.tmp()
            lines.append(f"let {pk} = smem_keys[{partner} * WPT + {src}u];")
            lines.append(f"let {pv} = smem_vals[{partner} * WPT + {src}u];")
            lines.append(f"let {less} = keys[{dst}] < {pk} || (keys[{dst}] == {pk} && values[{dst}] < {pv});")
            lines.append(f"if {bit} == {less} {{ keys[{dst}] = {pk}; values[{dst}] = {pv}; }}")
        lines.append("workgroupBarrier();")
        return "{ " + " ".join(lines) + " }"

    def _exch(self, pairs, tmask: int, swbit: int) -> str:
        if self._memory == "subgroup":
            return self._exch_subgroup(pairs, tmask, swbit)
        else:
            return self._exch_workgroup(pairs, tmask, swbit)


    def exch_intxn(self, tmask: int, swbit: int, wpt: int) -> str:
        return (f"// exch_intxn(tmask:{tmask},swbit:{swbit},wpt:{wpt})\n" +
                self._exch([(r, wpt - 1 - r) for r in range(wpt)], tmask, swbit))

    def exch_paral(self, tmask: int, swbit: int, wpt: int) -> str:
        return (f"// exch_paral(tmask:{tmask},swbit:{swbit},wpt:{wpt}) \n" +
                self._exch([(r, r) for r in range(wpt)], tmask, swbit))

    def reg_sort(self, N: int, M: int) -> str:
        p = int(log2(N))
        pt = int(log2(M))
        wpt = N // M
        stages = []

        def coop_thrd_size(x): return 1 << (pt - min(pt, x - 1))
        def coop_elem_size(x): return 1 << (p - x + 1)
        def swbit_of(cts): return int(log2(cts)) - 1

        for l in range(p, 0, -1):
            cts = coop_thrd_size(l)
            if cts == 1:
                stages.append(self.exch_local(coop_elem_size(l) - 1, wpt))
            else:
                stages.append(self.exch_intxn(cts - 1, swbit_of(cts), wpt))

            for k in range(l + 1, p + 1):
                cts = coop_thrd_size(k)
                if cts == 1:
                    rmask = coop_elem_size(k) - 1
                    stages.append(self.exch_local(rmask ^ (rmask >> 1), wpt))
                else:
                    tmask = cts - 1
                    stages.append(self.exch_paral(tmask ^ (tmask >> 1), swbit_of(cts), wpt))

        return "\n".join(stages)

    def store_back(self, base: str, length: str, active: str | None,
                   N: int, M: int) -> str:
        if self._store == "block":
            return self._block_store(base, length, active)
        if self._memory == "workgroup":
            return self._striped_store_smem(base, length, active)
        return self._striped_store_subgroup(base, length, active, N, M)

    def _block_store(self, base: str, length: str, active: str | None) -> str:
        guard = f"{active} && pos < {length}" if active else f"pos < {length}"
        return (
            "    // block store\n"
            "    for (var r = 0u; r < WPT; r = r + 1u) {\n"
            "        let pos = local_tid * WPT + r;\n"
            f"        if {guard} {{\n"
            f"            global_keys[{base} + pos] = keys[r];\n"
            f"            global_value_indices[{base} + pos] = values[r];\n"
            "        }\n"
            "    }"
        )

    def _striped_store_smem(self, base: str, length: str, active: str | None) -> str:
        guard = f"{active} && j < {length}" if active else f"j < {length}"
        return (
            "    // striped (coalesced) store via shared memory\n"
            "    for (var r = 0u; r < WPT; r = r + 1u) {\n"
            "        smem_keys[local_tid * WPT + r] = keys[r];\n"
            "        smem_vals[local_tid * WPT + r] = values[r];\n"
            "    }\n"
            "    workgroupBarrier();\n"
            "    for (var c = 0u; c < WPT; c = c + 1u) {\n"
            "        let j = c * M + local_tid;\n"
            f"        if {guard} {{\n"
            f"            global_keys[{base} + j] = smem_keys[j];\n"
            f"            global_value_indices[{base} + j] = smem_vals[j];\n"
            "        }\n"
            "    }"
        )

    def _striped_store_subgroup(self, base: str, length: str, active: str | None,
                                N: int, M: int) -> str:
        wpt = N // M
        w = wpt.bit_length() - 1
        guard = f"{active} && j < {length}" if active else f"j < {length}"

        lines = ["    // striped (coalesced) store via subgroup shuffle transpose",
                 "    {",
                 "        let grp_base = sid - local_tid;   // first lane of this segment",
                 "        let want = local_tid & (WPT - 1u);",
                 "        var out_keys: array<u32, WPT>;",
                 "        var out_values: array<u32, WPT>;"]
        for r in range(wpt):
            lines.append(f"        {{ let src = grp_base + (({r}u * M + local_tid) >> {w}u);")
            for s in range(wpt):
                lines.append(
                    f"          {{ let k = subgroupShuffle(keys[{s}], src);"
                    f" let v = subgroupShuffle(values[{s}], src);"
                    f" if want == {s}u {{ out_keys[{r}] = k; out_values[{r}] = v; }} }}")
            lines.append("        }")
        lines.append("        for (var r = 0u; r < WPT; r = r + 1u) {")
        lines.append("            let j = r * M + local_tid;")
        lines.append(f"            if {guard} {{")
        lines.append(f"                global_keys[{base} + j] = out_keys[r];")
        lines.append(f"                global_value_indices[{base} + j] = out_values[r];")
        lines.append("            }")
        lines.append("        }")
        lines.append("    }")
        return "\n".join(lines)

    def sort_kernel_reg(self, name: str, N: int, M: int) -> str:
        wpt = N // M
        stages = "\n".join("    " + ln for ln in self.reg_sort(N, M).split("\n"))
        store = self.store_back("seg_start", "seg_size", "is_active", N, M)
        return f"""
enable subgroups;

override WG: u32 = {M}u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = {N}u;
const M: u32 = {M}u;
const WPT: u32 = {wpt}u;

@compute @workgroup_size(WG, 1, 1)
fn {name}(
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {{
    const BIN: u32 = {N.bit_length() - 1}u;

    let bin_base = select(bin_offsets[BIN - 1u], 0u, BIN == 0u);
    let bin_count = bin_offsets[BIN] - bin_base;

    let local_tid = sid & (M - 1u);
    let wg_index = wg_id.x + wg_id.y * wg_dim.x;
    let global_seg = (wg_index * WG + lid) / M;

    let is_active = global_seg < bin_count;
    let slot = bin_base + select(0u, global_seg, is_active);   // clamp so the read is in-range
    let seg_id = bin_indices[slot];
    let seg_start = select(segments[seg_id - 1u], 0u, seg_id == 0u);
    let seg_end = segments[seg_id];
    let seg_size = select(0u, seg_end - seg_start, is_active);

    var keys: array<u32, {wpt}>;
    var values: array<u32, {wpt}>;

    for (var r = 0u; r < WPT; r = r + 1u) {{
        let pos = local_tid * WPT + r;
        if is_active && pos < seg_size {{
            keys[r] = global_keys[seg_start + pos];
            values[r] = seg_start + pos;
        }} else {{
            keys[r] = 0xffffffffu;
            values[r] = 0xffffffffu;
        }}
    }}

{stages}

{store}
}}
"""

    def sort_kernel_workgroup(self, name: str, N: int, M: int):
        wpt = N // M
        stages = "\n".join("    " + ln for ln in self.reg_sort(N, M).split("\n"))
        store = self.store_back("seg_start", "seg_size", "is_active", N, M)
        return f"""
override WG: u32 = {M}u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = {N}u;
const M: u32 = {M}u;
const WPT: u32 = {wpt}u;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn {name}(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {{
    const BIN: u32 = {N.bit_length() - 1}u;

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

    var keys: array<u32, {wpt}>;
    var values: array<u32, {wpt}>;

    for (var r = 0u; r < WPT; r = r + 1u) {{
        let pos = local_tid * WPT + r;
        if is_active && pos < seg_size {{
            keys[r] = global_keys[seg_start + pos];
            values[r] = seg_start + pos;
        }} else {{
            keys[r] = 0xffffffffu;
            values[r] = 0xffffffffu;
        }}
    }}

{stages}

{store}
}}
"""

    def sort_kernel(self, name: str, N: int, M: int):
        if (self._memory == "subgroup"):
            return self.sort_kernel_reg(name, N, M)
        else:
            return self.sort_kernel_workgroup(name, N, M)

    def tile_sort_kernel(self, name: str, N: int, M: int):
        assert self._memory == "workgroup"

        wpt = N // M
        stages = "\n".join("    " + ln for ln in self.reg_sort(N, M).split("\n"))
        store = self.store_back("tile_lo", "tile_len", None, N, M)
        return f"""
const WG: u32 = {M}u;
const N: u32 = {N}u;
const M: u32 = {M}u;
const WPT: u32 = {wpt}u;
const TILE_SIZE: u32 = {N}u;

struct TileInfo {{ seg_start: u32, seg_size: u32, offset: u32 }}
struct TileMeta {{ tile_count: u32, max_size: u32 }}

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> tiles: array<TileInfo>;
@group(0) @binding(3) var<storage, read> tile_meta: TileMeta;

var<workgroup> smem_keys: array<u32, WG * WPT>;
var<workgroup> smem_vals: array<u32, WG * WPT>;

@compute @workgroup_size(WG, 1, 1)
fn {name}(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {{
    let GID = wg_id.x + wg_id.y * wg_dim.x;
    if GID >= tile_meta.tile_count {{ return; }}

    let info = tiles[GID];
    let tile_lo = info.seg_start + info.offset;
    let tile_len = min(TILE_SIZE, info.seg_size - info.offset);

    let local_tid = tid_g;
    let seg_base = 0u;

    var keys: array<u32, {wpt}>;
    var values: array<u32, {wpt}>;

    for (var r = 0u; r < WPT; r = r + 1u) {{
        let pos = local_tid * WPT + r;
        if pos < tile_len {{
            keys[r] = global_keys[tile_lo + pos];
            values[r] = tile_lo + pos;
        }} else {{
            keys[r] = 0xffffffffu;
            values[r] = 0xffffffffu;
        }}
    }}

{stages}

{store}
}}
"""
    

SUBGROUP_SIZES = [
    8,      # intel
    16,     # intel
    32,     # standard for NVidia, MacOs
    64,     # standard for AMD/adreno
    128,    # adreno
]

SEGMENT_SIZES = [
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048
]

WPT_THRESHOLD = 8

def msg_round_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()

def wg_m(N):
    return min(N, msg_round_pow2((N + WPT_THRESHOLD - 1) // WPT_THRESHOLD))

@dataclass(frozen=True)
class KernelArgs:
    N: int
    M: int
    wpt: int
    is_register: bool
    is_block: bool

    def name(self):
        store = "block" if self.is_block else "striped"
        mem = "reg" if self.is_register else "wg"
        return f"segsort_{mem}_n{self.N}_m{self.M}_{store}"

def main():
    args = sys.argv[1:]
    output_dir = args[0]

    print("Generating all kernels")

    kernels = set()

    for is_block in [False, True]:
        # reg kernels: per subgroup size, only where register sort is viable
        for sg_size in SUBGROUP_SIZES:
            for N in SEGMENT_SIZES:
                M = min(sg_size, N)
                kernels.add(KernelArgs(N, M, N // M, True, is_block))

        # wg kernels: one per N, threshold-driven M (matches runtime wg selection)
        for N in SEGMENT_SIZES:
            M = wg_m(N)
            kernels.add(KernelArgs(N, M, N // M, False, is_block))

    for kernel in kernels:
        print(f"Kernel: {kernel}")

        store = "block" if kernel.is_block else "striped"
        memory = "subgroup" if kernel.is_register else "workgroup"
        generator = KernelGenerator(memory, store)
        source = generator.sort_kernel(kernel.name(), kernel.N, kernel.M)

        with open(f"{output_dir}/{kernel.name()}.wgsl", "w") as f:
            f.write(source)

    for store in ("block", "striped"):
        tile_gen = KernelGenerator("workgroup", store)
        tile_src = tile_gen.tile_sort_kernel("segsort_tile_n2048_m256", 2048, 256)
        with open(f"{output_dir}/segsort_tile_n2048_m256_{store}.wgsl", "w") as f:
            f.write(tile_src)


if __name__ == "__main__":
    main()
