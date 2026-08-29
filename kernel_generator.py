import sys
from math import log2
from dataclasses import dataclass


class KernelGenerator:
    _tmp_index: int

    def __init__(self, memory: str):
        self._tmp_index = 0
        self._memory = memory

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

    def sort_kernel_reg(self, name: str, N: int, M: int) -> str:
        wpt = N // M
        stages = "\n".join("    " + ln for ln in self.reg_sort(N, M).split("\n"))
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
    @builtin(workgroup_id) wg_id: vec3<u32>
) {{
    const BIN: u32 = {N.bit_length() - 1}u

    let bin_base = select(bin_offsets[BIN - 1u], 0u, BIN == 0u);
    let bin_count = bin_offsets[BIN] - bin_base;

    let local_tid = sid & (M - 1u);
    let global_seg = (wg_id.x * WG + lid) / M;

    let active = global_seg < bin_count;
    let slot = bin_base + select(0u, global_seg, active);   // clamp so the read is in-range
    let seg_id = bin_indices[slot];
    let seg_start = select(segments[seg_id - 1u], 0u, seg_id == 0u);
    let seg_end = segments[seg_id];
    let seg_size = select(0u, seg_end - seg_start, active);

    var keys: array<u32, {wpt}>;
    var values: array<u32, {wpt}>;

    for (var r = 0u; r < WPT; r = r + 1u) {{
        let pos = local_tid * WPT + r;
        if active && pos < seg_size {{
            keys[r] = global_keys[seg_start + pos];
            values[r] = seg_start + pos;
        }} else {{
            keys[r] = 0xffffffffu;
            values[r] = 0xffffffffu;
        }}
    }}

{stages}

    // blocked store
    for (var r = 0u; r < WPT; r = r + 1u) {{
        let pos = local_tid * WPT + r;
        if active && pos < seg_size {{
            global_keys[seg_start + pos] = keys[r];
            global_value_indices[seg_start + pos] = values[r];
        }}
    }}
}}
"""

    def sort_kernel_workgroup(self, name: str, N: int, M: int):
        wpt = N // M
        stages = "\n".join("    " + ln for ln in self.reg_sort(N, M).split("\n"))
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
    @builtin(workgroup_id) wg_id: vec3<u32>
) {{
    const BIN: u32 = {N.bit_length() - 1}u

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

    var keys: array<u32, {wpt}>;
    var values: array<u32, {wpt}>;

    for (var r = 0u; r < WPT; r = r + 1u) {{
        let pos = local_tid * WPT + r;
        if active && pos < seg_size {{
            keys[r] = global_keys[seg_start + pos];
            values[r] = seg_start + pos;
        }} else {{
            keys[r] = 0xffffffffu;
            values[r] = 0xffffffffu;
        }}
    }}

{stages}

    // blocked store
    for (var r = 0u; r < WPT; r = r + 1u) {{
        let pos = local_tid * WPT + r;
        if active && pos < seg_size {{
            global_keys[seg_start + pos] = keys[r];
            global_value_indices[seg_start + pos] = values[r];
        }}
    }}
}}
"""

    def sort_kernel(self, name: str, N: int, M: int):
        if (self._memory == "subgroup"):
            return self.sort_kernel_reg(name, N, M)
        else:
            return self.sort_kernel_workgroup(name, N, M)
    

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

@dataclass(frozen=True)
class KernelArgs:
    N: int
    M: int
    wpt: int
    is_register: bool

    def name(self):
        mem = "reg" if self.is_register else "wg"
        return f"segsort_{mem}_n{self.N}_m{self.M}"

def main():
    args = sys.argv[1:]
    output_dir = args[0]

    print("Generating all kernels")

    kernels = []

    for sg_size in SUBGROUP_SIZES:
        for N in SEGMENT_SIZES:
            M = min(sg_size, N)
            wpt = N // M

            kernels.append(KernelArgs(N, M, wpt, False))
            kernels.append(KernelArgs(N, M, wpt, True))

    kernels = list(set(kernels))

    for kernel in kernels:
        print(f"Kernel: {kernel}")

        kernel_name = kernel.name()
        memory = "subgroup" if kernel.is_register else "workgroup"
        generator = KernelGenerator(memory)
        source = generator.sort_kernel(kernel_name, kernel.N, kernel.M)

        with open(f"{output_dir}/{kernel_name}.wgsl", "w") as f:
            f.write(source)


if __name__ == "__main__":
    main()
