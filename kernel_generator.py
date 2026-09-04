import sys
from math import log2
from dataclasses import dataclass
from pathlib import Path


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

INDENT_SPACES = 4
INDENT1 = " " * INDENT_SPACES

def msg_round_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()

def wg_m(N):
    return min(N, msg_round_pow2((N + WPT_THRESHOLD - 1) // WPT_THRESHOLD))

@dataclass(frozen=True)
class KernelArgs:
    N: int          # segment length
    M: int          # number of threads (lanes) cooperating on a segment
    wpt: int        # work items per thread = N // M
    R: int          # number of threads in subgroup (for register shuffling),
                    # must be <= n_subgroups supported by chip
    mode: str       # "reg" | "wg" | "hybrid" | "hybmerge"
    is_block: bool  # block store if true, otherwise striped store

    def name(self):
        store = "block" if self.is_block else "striped"
        if self.mode == "hybrid":
            return f"segsort_hybrid_sg{self.R}_n{self.N}_m{self.M}_{store}"
        if self.mode == "hybmerge":
            smem_k = 16 if 8 * self.N <= 16384 else 32
            return f"segsort_hybmerge_sg{self.R}_smem{smem_k}k_n{self.N}_m{self.M}_{store}"
        return f"segsort_{self.mode}_n{self.N}_m{self.M}_{store}"


class KernelGenerator:
    _tmp_index: int

    def __init__(self):
        self._tmp_index = 0

    def tmp(self) -> str:
        tmp_index = self._tmp_index
        self._tmp_index += 1
        return f"tmp_{tmp_index}"

    @staticmethod
    def _indent(text: str, indent: int) -> str:
        prefix = " " * indent * INDENT_SPACES
        return "\n".join(prefix + line if line else line
                         for line in text.split("\n"))

    # swap key/value pairs
    def swap(self, a: int, b: int, indent: int) -> str:
        tk = self.tmp()
        tv = self.tmp()

        body = f"""// swap({a},{b})
{{
    let {tk} = keys[{a}]; keys[{a}] = keys[{b}]; keys[{b}] = {tk};
    let {tv} = values[{a}]; values[{a}] = values[{b}]; values[{b}] = {tv};
}}"""
        return self._indent(body, indent)

    # swap key/value pairs if keys[a] > keys[b]
    # tie break with value index
    def cmp_swap(self, a: int, b: int, indent: int) -> str:
        inner = self.swap(a, b, 1)
        body = f"""// cmp_swap({a},{b})
if keys[{a}] > keys[{b}] || (keys[{a}] == keys[{b}] && values[{a}] > values[{b}]) {{
{inner}
}}"""
        return self._indent(body, indent)

    # swap key/value pairs if keys[a] != keys[b]
    def eql_swap(self, a: int, b: int, indent: int) -> str:
        inner = self.swap(a, b, 1)
        body = f"""// eql_swap({a},{b})
if keys[{a}] != keys[{b}] {{
{inner}
}}"""
        return self._indent(body, indent)


#     def exch(self, r: int, tmask: int, swbit: int, indent: int) -> str:
#         pk, pv, less, bit = self.tmp(), self.tmp(), self.tmp(), self.tmp()

#         lines = f"""// exch(r:{r},tmask:{tmask},swbit:{swbit})
# {{
#     let {pk} = subgroupShuffleXor(keys[{r}], {tmask}u);
#     let {pv} = subgroupShuffleXor(values[{r}], {tmask}u);
#     let {less} = keys[{r}] < {pk} || (keys[{r}] == {pk} && values[{r}] < {pv});
#     let {bit} = extractBits(local_tid, {swbit}u, 1u) != 0u;
#     if {bit} == {less} {{ keys[{r}] = {pk}; values[{r}] = {pv}; }}
# }}

# """.split("\n")
        
#         spaces = " " * indent * INDENT_SPACES
#         return "\n".join(spaces + line for line in lines)

    # exchanges registers within a thread
    def exch_local(self, rmask: int, wpt: int, indent: int) -> str:
        swaps = []
        for i in range(wpt):
            j = i ^ rmask
            if i < j:
                swaps.append(self.cmp_swap(i, j, 1))

        inner = "\n".join(swaps)
        body = f"""// exch_local({rmask},{wpt})
{{
{inner}
}}"""
        return self._indent(body, indent)

    # exchanges registers within a subgroup, different threads
    # Called _exch_primitive in Hou et al
    # This operation swaps registers between cooperating threads in a subgroup
    # r - register index to swap (contains key being compared between the two threads)
    # tmask - bit mask to find cooperating thread
    # swbit - 1/0, determines which threads perform the swap
    def _exch_subgroup(self, pairs, tmask: int, swbit: int, indent: int) -> str:
        reads, stmts = [], []

        for dst, src in pairs:
            pk, pv = self.tmp(), self.tmp()
            stmts.append(f"let {pk} = subgroupShuffleXor(keys[{src}], {tmask}u);")
            stmts.append(f"let {pv} = subgroupShuffleXor(values[{src}], {tmask}u);")
            reads.append((dst, pk, pv))
        bit = self.tmp()
        stmts.append(f"let {bit} = extractBits(local_tid, {swbit}u, 1u) != 0u;")
        for dst, pk, pv in reads:
            less = self.tmp()
            stmts.append(f"let {less} = keys[{dst}] < {pk} || (keys[{dst}] == {pk} && values[{dst}] < {pv});")
            stmts.append(f"if {bit} == {less} {{ keys[{dst}] = {pk}; values[{dst}] = {pv}; }}")

        inner = "\n".join(INDENT1 + s for s in stmts)
        body = f"""// _exch_subgroup({pairs},{tmask},{swbit})
{{
{inner}
}}"""
        return self._indent(body, indent)

    # exchanges registers withing a workgroup using shared memory
    # this works to exchange data within a subgroup, but not as efficient as _exch_subgroup
    # avoid this by tuning N, M, and R properly
    # this could fail on Metal, as workgroupBarrier() is not a strong guarantee if workgroup_size <= 32
    # WGSL -> MSL is this: workgroupBarrier() -> threadgroup_barrier(mem_flags::mem_threadgroup)
    def _exch_workgroup(self, pairs, tmask: int, swbit: int, indent: int) -> str:
        srcs = sorted({src for _, src in pairs})
        stmts = []
        for s in srcs:
            stmts.append(f"smem_keys[tid_g * WPT + {s}u] = keys[{s}];")
            stmts.append(f"smem_vals[tid_g * WPT + {s}u] = values[{s}];")
        stmts.append("workgroupBarrier();")
        bit, partner = self.tmp(), self.tmp()
        stmts.append(f"let {bit} = extractBits(local_tid, {swbit}u, 1u) != 0u;")
        stmts.append(f"let {partner} = seg_base + (local_tid ^ {tmask}u);")
        for dst, src in pairs:
            pk, pv, less = self.tmp(), self.tmp(), self.tmp()
            stmts.append(f"let {pk} = smem_keys[{partner} * WPT + {src}u];")
            stmts.append(f"let {pv} = smem_vals[{partner} * WPT + {src}u];")
            stmts.append(f"let {less} = keys[{dst}] < {pk} || (keys[{dst}] == {pk} && values[{dst}] < {pv});")
            stmts.append(f"if {bit} == {less} {{ keys[{dst}] = {pk}; values[{dst}] = {pv}; }}")
        stmts.append("workgroupBarrier();")

        inner = "\n".join(INDENT1 + s for s in stmts)
        body = f"""// _exch_workgroup({pairs},{tmask},{swbit})
{{
{inner}
}}"""
        return self._indent(body, indent)

    # dispatches to subgroup or workgroup variant based on R
    # this is the exchange primitive in Hou et al, extended to work across
    #   subgroups as well
    def _exch(self, kernel: KernelArgs, pairs, tmask: int, swbit: int, indent: int) -> str:
        if tmask < kernel.R:
            return self._exch_subgroup(pairs, tmask, swbit, indent)
        return self._exch_workgroup(pairs, tmask, swbit, indent)

    # the _exch_intxn primitive from Hou et al, extended to work across subgroups
    def exch_intxn(self, kernel: KernelArgs, tmask: int, swbit: int, indent: int) -> str:
        wpt = kernel.wpt

        inner = self._exch(kernel, [(r, wpt - 1 - r) for r in range(wpt)], tmask, swbit, 1)
        body = f"""// exch_intxn(tmask:{tmask},swbit:{swbit},wpt:{wpt})
{{
{inner}
}}"""
        return self._indent(body, indent)

    # the _exch_paral primitive from Hou et al, extended to work across subgroups
    def exch_paral(self, kernel: KernelArgs, tmask: int, swbit: int, indent: int) -> str:
        wpt = kernel.wpt

        inner = self._exch(kernel, [(r, r) for r in range(wpt)], tmask, swbit, 1)
        body = f"""// exch_paral(tmask:{tmask},swbit:{swbit},wpt:{wpt})
{{
{inner}
}}"""
        return self._indent(body, indent)

    # Algorithm 1 from Hou et al
    def reg_sort(self, kernel: KernelArgs, indent: int) -> str:
        N, M = kernel.N, kernel.M
        p = int(log2(N))
        pt = int(log2(M))
        wpt = kernel.wpt
        stages = []

        def coop_thrd_size(x): return 1 << (pt - min(pt, x - 1))
        def coop_elem_size(x): return 1 << (p - x + 1)
        def swbit_of(cts): return int(log2(cts)) - 1

        for l in range(p, 0, -1):
            cts = coop_thrd_size(l)
            if cts == 1:
                stages.append(self.exch_local(coop_elem_size(l) - 1, wpt, 0))
            else:
                stages.append(self.exch_intxn(kernel, cts - 1, swbit_of(cts), 0))

            for k in range(l + 1, p + 1):
                cts = coop_thrd_size(k)
                if cts == 1:
                    rmask = coop_elem_size(k) - 1
                    stages.append(self.exch_local(rmask ^ (rmask >> 1), wpt, 0))
                else:
                    tmask = cts - 1
                    stages.append(self.exch_paral(kernel, tmask ^ (tmask >> 1), swbit_of(cts), 0))

        return self._indent("\n\n".join(stages), indent)

    # storage algorithm back to global memory. two modes are supported: block, striped
    # block stores items back to global memory without any register reordering
    #   Example: Numbers are write indices in global memory after sorting
    #       Thread 1: [10,11,12,13]
    #       Thread 2: [14,15,16,17]
    #       
    #       Because data is written by threads in lock stop, the index write pairs, in sequence are:
    #           [10,14], [11,15], [12,16], [13,17]
    #       When WPT (items per thread) is greater, the distances between indices is greater, which can cause
    #           additional memory traffic to the GPU, because ranges of memory are written, not just single values
    #
    # striped stores transpose the registers or workgoup memory before writing back to global memory.
    #   Example: Numbers are write indices in global memory after sorting
    #       Thread 1: [10,11,12,13] -> transpose -> [10,12,14,16]
    #       Thread 2: [14,15,16,17] -> transpose -> [11,13,15,17]
    #
    #       So now the data write pairs are coalesed and are written back to global memory in sequence:
    #           [10,11], [12,13], [14,15], [16,17]
    #       The cost of the transpose may outweigh the benefit to memory traffic, so this difference needs to be measured
    def store_back(self, kernel: KernelArgs,
                   base: str, length: str,
                   active: str | None, indent: int) -> str:
        if kernel.is_block:
            return self._block_store(base, length, active, indent)

        if kernel.mode in ("wg", "hybrid"):
            return self._striped_store_smem(base, length, active, indent)
        return self._striped_store_shuffle(base, length, active, kernel.N, kernel.M, indent)

    # writes sorted keys back to global memory without any coalescing
    def _block_store(self, base: str, length: str, active: str | None, indent: int) -> str:
        guard = f"{active} && pos < {length}" if active else f"pos < {length}"

        body = f"""// block store
for (var r = 0u; r < WPT; r = r + 1u) {{
    let pos = local_tid * WPT + r;
    if {guard} {{
        global_keys[{base} + pos] = keys[r];
        global_value_indices[{base} + pos] = values[r];
    }}
}}"""
        return self._indent(body, indent)

    def _striped_store_smem(self, base: str, length: str, active: str | None, indent: int) -> str:
        guard = f"{active} && j < {length}" if active else f"j < {length}"

        body = f"""// striped (coalesced) store via shared memory
for (var r = 0u; r < WPT; r = r + 1u) {{
    smem_keys[tid_g * WPT + r] = keys[r];
    smem_vals[tid_g * WPT + r] = values[r];
}}
workgroupBarrier();
for (var c = 0u; c < WPT; c = c + 1u) {{
    let j = c * M + local_tid;
    if {guard} {{
        global_keys[{base} + j] = smem_keys[seg_base * WPT + j];
        global_value_indices[{base} + j] = smem_vals[seg_base * WPT + j];
    }}
}}"""
        return self._indent(body, indent)

    @staticmethod
    def _transpose_plan(M: int, WPT: int):
        m = M.bit_length() - 1
        w = WPT.bit_length() - 1
        lane_bits = [w + j for j in range(m)]
        reg_bits = [i for i in range(w)]
        ops = []

        def do(a, b):
            ops.append((a, b))
            lane_bits[a], reg_bits[b] = reg_bits[b], lane_bits[a]

        for j in range(m):
            if lane_bits[j] == j:
                continue
            if j in reg_bits:
                do(j, reg_bits.index(j))
            else:
                j2 = lane_bits.index(j)
                do(j2, 0)
                do(j, 0)

        remap = []
        for r in range(WPT):
            rp = 0
            for i, ebit in enumerate(reg_bits):
                if (r >> i) & 1:
                    rp |= 1 << (ebit - m)
            remap.append(rp)
        return ops, remap

    def _striped_store_shuffle(self, base: str, length: str, active: str | None,
                               N: int, M: int, indent: int) -> str:
        wpt = N // M
        guard = f"{active} && j < {length}" if active else f"j < {length}"

        if wpt == 1:
            g = f"{active} && local_tid < {length}" if active else f"local_tid < {length}"
            body = f"""// striped (coalesced) store (WPT==1, no transpose)
if {g} {{
    global_keys[{base} + local_tid] = keys[0];
    global_value_indices[{base} + local_tid] = values[0];
}}"""
            return self._indent(body, indent)

        ops, remap = self._transpose_plan(M, wpt)

        lines = ["// striped (coalesced) store via shfl_xor transpose",
                 "{"]
        for (a_bit, b_bit) in ops:
            mask = 1 << a_bit
            for r0 in range(wpt):
                if r0 & (1 << b_bit):
                    continue
                r1 = r0 | (1 << b_bit)
                lines += [
                    f"    {{ let ex_k = subgroupShuffleXor(keys[{r1}], {mask}u);",
                    f"      let ex_v = subgroupShuffleXor(values[{r1}], {mask}u);",
                    "      var t_k = ex_k; var t_v = ex_v;",
                    f"      if ((local_tid & {mask}u) != 0u) {{",
                    f"          t_k = keys[{r0}]; t_v = values[{r0}];",
                    f"          keys[{r0}] = ex_k; values[{r0}] = ex_v;",
                    "      }",
                    f"      keys[{r1}] = subgroupShuffleXor(t_k, {mask}u);",
                    f"      values[{r1}] = subgroupShuffleXor(t_v, {mask}u); }}",
                ]
        remap_lit = ", ".join(f"{v}u" for v in remap)
        lines += [
            f"    var out_reg = array<u32, WPT>({remap_lit});",
            "    for (var r = 0u; r < WPT; r = r + 1u) {",
            "        let j = out_reg[r] * M + local_tid;",
            f"        if {guard} {{",
            f"            global_keys[{base} + j] = keys[r];",
            f"            global_value_indices[{base} + j] = values[r];",
            "        }",
            "    }",
            "}",
        ]
        return self._indent("\n".join(lines), indent)

    # create a register-based sorting kernel
    def sort_kernel_reg(self, kernel: KernelArgs) -> str:
        name = kernel.name()
        N = kernel.N
        M = kernel.M

        wpt = N // M
        stages = self.reg_sort(kernel, 1)
        store = self.store_back(kernel, "seg_start", "seg_size", "is_active", 1)
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

    # create a shared-memory-base sorting kernel
    def sort_kernel_workgroup(self, kernel: KernelArgs):
        name = kernel.name()
        N = kernel.N
        M = kernel.M

        wpt = N // M
        stages = self.reg_sort(kernel, 1)
        store = self.store_back(kernel, "seg_start", "seg_size", "is_active", 1)
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

    # create a register -> shared memory bitonic sorting kernel
    def sort_kernel_hybrid(self, kernel: KernelArgs):
        name = kernel.name()
        N = kernel.N
        M = kernel.M
        R = kernel.R

        wpt = N // M
        stages = self.reg_sort(kernel, 1)
        store = self.store_back(kernel, "seg_start", "seg_size", "is_active", 1)
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
const R: u32 = {R}u;

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

    # register bitonic sorting network -> shared-memory merge kernel
    def sort_kernel_hybrid_merge(self, kernel: KernelArgs):
        name = kernel.name()
        N, M, sg = kernel.N, kernel.M, kernel.R
        wpt = kernel.wpt
        RUN = sg * wpt
        P = int(log2(M // sg))

        run_kernel = KernelArgs(RUN, sg, wpt, sg, "reg", kernel.is_block)
        phase1 = self.reg_sort(run_kernel, 1)

        run = RUN
        passes = []
        for p in range(P):
            merged = 2 * run
            passes.append(f"""    // merge pass {p}: two sorted runs of {run} -> {merged} (register-staged)
    {{
        let group_base = (base / {merged}u) * {merged}u;
        let diag = base - group_base;
        let a_base = group_base;
        let b_base = group_base + {run}u;
        // merge-path: binary search the diagonal for this thread's A/B split
        var lo = select(0u, diag - {run}u, diag > {run}u);
        var hi = min(diag, {run}u);
        while (lo < hi) {{
            let mid = (lo + hi) >> 1u;
            let ak = smem_keys[a_base + mid];
            let av = smem_vals[a_base + mid];
            let bpos = b_base + (diag - 1u - mid);
            let bk = smem_keys[bpos];
            let bv = smem_vals[bpos];
            if ak < bk || (ak == bk && av <= bv) {{ lo = mid + 1u; }} else {{ hi = mid; }}
        }}
        var ai = lo;
        var bi = diag - lo;
        // merge this thread's WPT outputs into registers (the pong)
        var out_keys: array<u32, {wpt}>;
        var out_vals: array<u32, {wpt}>;
        for (var k = 0u; k < WPT; k = k + 1u) {{
            let take_a = bi >= {run}u || (ai < {run}u &&
                (smem_keys[a_base + ai] < smem_keys[b_base + bi] ||
                 (smem_keys[a_base + ai] == smem_keys[b_base + bi] &&
                  smem_vals[a_base + ai] <= smem_vals[b_base + bi])));
            if take_a {{
                out_keys[k] = smem_keys[a_base + ai];
                out_vals[k] = smem_vals[a_base + ai];
                ai = ai + 1u;
            }} else {{
                out_keys[k] = smem_keys[b_base + bi];
                out_vals[k] = smem_vals[b_base + bi];
                bi = bi + 1u;
            }}
        }}
        workgroupBarrier();   // every read is done before any write-back
        storageBarrier();     // device-scope fence: workgroupBarrier alone under-orders
                              // the in-place write-back for single-SIMD-group WGs
        for (var k = 0u; k < WPT; k = k + 1u) {{
            smem_keys[base + k] = out_keys[k];
            smem_vals[base + k] = out_vals[k];
        }}
    }}
    workgroupBarrier();""")
            run = merged
        merge_passes = "\n".join(passes)
        fk, fv = "smem_keys", "smem_vals"    # the fully merged result lives in smem

        if kernel.is_block:
            store = f"""    for (var r = 0u; r < WPT; r = r + 1u) {{
        let pos = local_tid * WPT + r;
        if is_active && pos < seg_size {{
            global_keys[seg_start + pos] = {fk}[pos];
            global_value_indices[seg_start + pos] = {fv}[pos];
        }}
    }}"""
        else:
            store = f"""    for (var c = 0u; c < WPT; c = c + 1u) {{
        let j = c * M + local_tid;
        if is_active && j < seg_size {{
            global_keys[seg_start + j] = {fk}[j];
            global_value_indices[seg_start + j] = {fv}[j];
        }}
    }}"""

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
const SG: u32 = {sg}u;      // register run spans one subgroup: RUN = SG*WPT = {RUN}

var<workgroup> smem_keys: array<u32, N>;
var<workgroup> smem_vals: array<u32, N>;

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

{phase1}

    // stage the sorted runs into shared memory (blocked layout)
    for (var r = 0u; r < WPT; r = r + 1u) {{
        smem_keys[local_tid * WPT + r] = keys[r];
        smem_vals[local_tid * WPT + r] = values[r];
    }}
    workgroupBarrier();

    let base = local_tid * WPT;   // this thread's blocked output range [base, base+WPT)

{merge_passes}
{store}
}}
"""

    def sort_kernel(self, kernel: KernelArgs):
        if kernel.mode == "reg":
            return self.sort_kernel_reg(kernel)
        elif kernel.mode == "wg":
            return self.sort_kernel_workgroup(kernel)
        elif kernel.mode == "hybrid":
            return self.sort_kernel_hybrid(kernel)
        elif kernel.mode == "hybmerge":
            return self.sort_kernel_hybrid_merge(kernel)

    def sort_kernel_n1(self):
        name = "segsort_wg_n1_m1_block"
        return f"""override WG: u32 = 256u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

@compute @workgroup_size(WG, 1, 1)
fn {name}(
    @builtin(local_invocation_index) tid_g: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_dim: vec3<u32>
) {{
    let bin_count = bin_offsets[0];

    let local_tid = tid_g;
    let seg_base = tid_g - local_tid;
    let wg_index = wg_id.x + wg_id.y * wg_dim.x;
    let global_seg = wg_index * WG + tid_g;

    if global_seg >= bin_count {{
        return;
    }}

    let seg_id = bin_indices[global_seg];
    let seg_start = select(segments[seg_id - 1u], 0u, seg_id == 0u);

    global_value_indices[seg_start] = seg_start;
}}
"""

    def tile_sort_kernel(self, kernel: KernelArgs, name: str):
        N, M = kernel.N, kernel.M
        wpt = kernel.wpt
        stages = self.reg_sort(kernel, 1)
        store = self.store_back(kernel, "tile_lo", "tile_len", None, 1)
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

MAX_WG = 256
SMEM_BUDGETS = [16384, 32768]
WPTS = [2, 4, 8, 16, 32]

def main():
    args = sys.argv[1:]
    output_dir = args[0]

    print("Generating all kernels")

    kernels = set()

    for is_block in [False, True]:
        for sg_size in SUBGROUP_SIZES:
            for N in SEGMENT_SIZES:
                M = min(sg_size, N)
                kernels.add(KernelArgs(N, M, N // M, M, "reg", is_block))

        for N in SEGMENT_SIZES:
            for wpt in WPTS:
                if N % wpt != 0:
                    continue
                M = N // wpt
                if M < 1 or M > MAX_WG:
                    continue

                kernels.add(KernelArgs(N, M, wpt, 1, "wg", is_block))

                for sg_size in SUBGROUP_SIZES:
                    if M > sg_size:
                        kernels.add(KernelArgs(N, M, wpt, sg_size, "hybrid", is_block))
                    if M >= 2 * sg_size and 8 * N <= SMEM_BUDGETS[-1]:
                        kernels.add(KernelArgs(N, M, wpt, sg_size, "hybmerge", is_block))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for kernel in kernels:
        print(f"Kernel: {kernel}")

        generator = KernelGenerator()
        source = generator.sort_kernel(kernel)

        with open(f"{output_dir}/{kernel.name()}.wgsl", "w") as f:
            f.write(source)

    for is_block in (False, True):
        tile_k = KernelArgs(2048, 256, 2048 // 256, 1, "wg", is_block)
        tile_gen = KernelGenerator()
        tile_src = tile_gen.tile_sort_kernel(tile_k, "segsort_tile_n2048_m256")
        store = "block" if is_block else "striped"
        with open(f"{output_dir}/segsort_tile_n2048_m256_{store}.wgsl", "w") as f:
            f.write(tile_src)

    with open(f"{output_dir}/segsort_wg_n1_m1_block.wgsl", "w") as f:
        f.write(KernelGenerator().sort_kernel_n1())


if __name__ == "__main__":
    main()
