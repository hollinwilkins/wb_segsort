import sys
from math import log2
from dataclasses import dataclass


class KernelGenerator:
    _tmp_index: int

    def __init__(self):
        self._tmp_index = 0

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
                f"{{ let {pk} = subgroupShuffleXor(keys[{r}], {tmask}u);"
                f" let {pv} = subgroupShuffleXor(values[{r}], {tmask}u);"
                f" let {less} = keys[{r}] < {pk} || (keys[{r}] == {pk} && values[{r}] < {pv});"
                f" let {bit} = extractBits(local_tid, {swbit}u, 1u) != 0u;"
                f" if {bit} == {less} {{ keys[{r}] = {pk}; values[{r}] = {pv}; }} }}")

    def exch_local(self, rmask: int, wpt: int) -> str:
        out = []
        for i in range(wpt):
            j = i ^ rmask
            if i < j:
                out.append(self.cmp_swap(i, j))
        return (f"// exch_local({rmask},{wpt}) \n" +
                "\n".join(out))

    def _exch(self, pairs, tmask: int, swbit: int) -> str:
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
        return "{ " + " ".join(lines) + " }"

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

    def sort_kernel(self, name: str, N: int, M: int) -> str:
        wpt = N // M
        stages = "\n".join("    " + ln for ln in self.reg_sort(N, M).split("\n"))
        return f"""enable subgroups;

@group(0) @binding(0) var<storage, read_write> keys_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> values_global: array<u32>;

const N: u32 = {N}u;
const M: u32 = {M}u;
const WPT: u32 = {wpt}u;

// Single subgroup, single segment. Requires subgroup_size == M.
@compute @workgroup_size({M}, 1, 1)
fn {name}(@builtin(subgroup_invocation_id) sid: u32) {{
    let local_tid = sid & (M - 1u);
    let seg_start = 0u;
    let seg_size  = min(N, arrayLength(&keys_global));

    var keys:   array<u32, {wpt}>;
    var values: array<u32, {wpt}>;

    // blocked load, sentinel-padded
    for (var r = 0u; r < WPT; r = r + 1u) {{
        let pos = local_tid * WPT + r;
        if pos < seg_size {{
            keys[r]   = keys_global[seg_start + pos];
            values[r] = seg_start + pos;
        }} else {{
            keys[r]   = 0xffffffffu;
            values[r] = 0xffffffffu;
        }}
    }}

{stages}

    // blocked store
    for (var r = 0u; r < WPT; r = r + 1u) {{
        let pos = local_tid * WPT + r;
        if pos < seg_size {{
            keys_global[seg_start + pos] = keys[r];
            values_global[seg_start + pos] = values[r];
        }}
    }}
}}
"""

SUBGROUP_SIZES = [
    8,      # intel
    16,     # intel
    32,     # standard for NVidia, MacOs
    64,     # standard for NVidia
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

def main():
    args = sys.argv[1:]
    if args[0] == "all":
        print("Generating all kernels")

        kernels = []

        for sg_size in SUBGROUP_SIZES:
            for N in SEGMENT_SIZES:
                M = min(sg_size, N)
                wpt = N // M

                if wpt > WPT_THRESHOLD:
                    kernels.append(KernelArgs(N, M, wpt, False))
                else:
                    kernels.append(KernelArgs(N, M, wpt, True))

        kernels = list(set(kernels))

        for kernel in kernels:
            print(f"Kernel: {kernel}")


if __name__ == "__main__":
    main()
