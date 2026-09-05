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
    mode: str       # "reg" | "cute" | "wg" | "hybrid" | "hybmerge"
    is_block: bool  # block store if true, otherwise striped store

    def name(self):
        store = "block" if self.is_block else "striped"
        if self.mode == "hybrid":
            return f"segsort_hybrid_sg{self.R}_n{self.N}_m{self.M}_{store}"
        if self.mode == "hybmerge":
            smem_k = 16 if 8 * self.N <= 16384 else 32
            return f"segsort_hybmerge_sg{self.R}_smem{smem_k}k_n{self.N}_m{self.M}_{store}"
        if self.mode == "cutemerge":
            smem_k = 16 if 8 * self.N <= 16384 else 32
            return f"segsort_cutemerge_sg{self.R}_smem{smem_k}k_n{self.N}_m{self.M}_{store}"
        return f"segsort_{self.mode}_n{self.N}_m{self.M}_{store}"

class Transposer:
    def __init__(self, N: int, M: int):
        self.N = N
        self.M = M
        self.WPT = N // M
        self.lane_swaps = (M.bit_length() - 1)
        self.register_swaps = (self.WPT.bit_length() - 1)

        # initial state after sorting is:
        # lane_bits | register_bits (lane bits hi, register bits lo)
        # target bit state after transpose is:
        # register bits (unsorted) : lane_bits (sorted) means (register bits hi, lane bits lo)
        # register bits do not need to be sorted because we will build a remap
        #   array that maps the register index to global index
        self.register_bits = [i for i in range(self.register_swaps)]
        self.lane_bits = [i + self.register_swaps for i in range(self.lane_swaps)]

    def swap(self, lane_bit: int, register_bit: int):
        self.lane_bits[lane_bit], self.register_bits[register_bit] = self.register_bits[register_bit], self.lane_bits[lane_bit]

    def build(self):
        # print(f"N: {self.N}, M: {self.M}, WPT: {self.WPT}, paral_swaps lane_swaps: {self.lane_swaps}, register_swaps: {self.register_swaps}")

        if self.lane_swaps == 0 or self.register_swaps == 0:
            return [], [0]

        # print(f"Before Swap: lane_bits: {self.lane_bits}, register_bits: {self.register_bits}")

        # generate swap pairs
        swaps = []

        for lane_bit in range(self.lane_swaps):
            register_bit = lane_bit % self.register_swaps
            self.swap(lane_bit, register_bit)
            swaps.append((lane_bit, register_bit))

        # print(f"After Swap: lane_bits: {self.lane_bits}, register_bits: {self.register_bits}")

        # print(f"swaps: {swaps}")

        # generate register indices
        bit_map = [bit - self.lane_swaps for bit in self.register_bits]
        register_map = []
        # print(f"bit map: {bit_map}")
        for ri in range(self.WPT):
            r = 0
            for j, bit in enumerate(bit_map):
                if ri & (1 << j):
                    r = r | (1 << bit)

            register_map.append(r)

        # print(f"register_map: {register_map}")
        # print()

        return swaps, register_map


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

    def _striped_store_shuffle(self, base: str, length: str, active: str | None,
                               N: int, M: int, indent: int) -> str:
        wpt = N // M

        if wpt == 1:
            g = f"{active} && local_tid < {length}" if active else f"local_tid < {length}"
            body = f"""// striped (coalesced) store (WPT==1, no transpose)
if {g} {{
    global_keys[{base} + local_tid] = keys[0];
    global_value_indices[{base} + local_tid] = values[0];
}}"""
            return self._indent(body, indent)

        transposer = Transposer(N, M)
        swaps, remap = transposer.build()

        lines = []
        for (lane_bit, register_bit) in swaps:
            mask = 1 << lane_bit
            for hi_register_index in range(wpt):
                if (hi_register_index & (1 << register_bit)) == 0: continue # check if register_bit is 1, this is the right-side of the transpose
                lo_register_index = hi_register_index ^ pow(2, register_bit)

                lines += f"""// swap(lane_bit:{lane_bit},register_bit:{register_bit},hi_register_index:{hi_register_index})
    {{
        let lo_ex_k = subgroupShuffleXor(keys[{lo_register_index}], {mask}u);
        let lo_ex_v = subgroupShuffleXor(values[{lo_register_index}], {mask}u);
        let hi_ex_k = subgroupShuffleXor(keys[{hi_register_index}], {mask}u);
        let hi_ex_v = subgroupShuffleXor(values[{hi_register_index}], {mask}u);
        if ((local_tid & {mask}u) == 0u) {{
            // lo lane: swap self hi register with pair lo register
            keys[{hi_register_index}] = lo_ex_k;
            values[{hi_register_index}] = lo_ex_v;
        }} else {{
            // hi lane: swap self lo register with pair hi register
            keys[{lo_register_index}] = hi_ex_k;
            values[{lo_register_index}] = hi_ex_v;
        }}
    }}""".split("\n")

        if active:
            lines.append(f"    if {active} {{")
        else:
            lines.append("    {")
        for i in range(wpt):
            lines.append(f"        {{ let global_offset = {remap[i]} * M + local_tid;")
            lines.append(f"            if global_offset < {length} {{")
            lines.append(f"        global_keys[{base} + global_offset] = keys[{i}];")
            lines.append(f"        global_value_indices[{base} + global_offset] = values[{i}]; }} }}")
        lines.append("    }")

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

    # merge-path pass: merge adjacent sorted runs of `run` into runs of 2*run,
    # staged through smem (register pong write-back). Identical body to the
    # hybmerge merge pass -- shared by hybmerge and cutemerge.
    def _merge_pass(self, run: int, wpt: int) -> str:
        merged = 2 * run
        return f"""    // merge pass: two sorted runs of {run} -> {merged} (register-staged)
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
    workgroupBarrier();"""

    def _cute_base(self, sg: int, wpt: int):
        COMP = ["x", "y", "z", "w"]
        if sg in (32, 64, 128):
            cpb = sg // 32
            cw = min(wpt, 128 // sg)
            crun = cw * sg
            groups = wpt // cw
            blocks = []
            for g in range(groups):
                s0 = g * cw
                L = [f"    {{  // CuteSort wide run {g}: {cw} slot(s) x {sg} lanes -> sorted run of {crun}",
                     f"        let rbase = sub_block + {g * crun}u;"]
                for c in range(cw):
                    L.append(f"        var ge_{c} = lane_mask_lt({c * sg}u + sid);")
                L.append("        for (var bit = 0u; bit < 32u; bit = bit + 1u) {")
                for c in range(cw):
                    L.append(f"            let bal_{c} = subgroupBallot((keys[{s0 + c}] & (1u << bit)) == 0u);")
                L.append("            var zmask = vec4<u32>(0u, 0u, 0u, 0u);")
                for c in range(cw):
                    for j in range(cpb):
                        L.append(f"            zmask.{COMP[c * cpb + j]} = bal_{c}.{COMP[j]};")
                for c in range(cw):
                    L.append(f"            let isz_{c} = (keys[{s0 + c}] & (1u << bit)) == 0u;")
                    L.append(f"            ge_{c} = select(ge_{c} | zmask, ge_{c} & zmask, isz_{c});")
                L.append("        }")
                for c in range(cw):
                    L.append(f"        let r_{c} = ballot_popc(ge_{c});")
                    L.append(f"        smem_keys[rbase + r_{c}] = keys[{s0 + c}];")
                    L.append(f"        smem_vals[rbase + r_{c}] = values[{s0 + c}];")
                L.append("    }")
                blocks.append("\n".join(L))
            return "\n".join(blocks), crun, False

        slots = []
        for k in range(wpt):
            slots.append(f"""    {{  // cute run for register slot {k}
        let key_k = keys[{k}];
        var ge_mask = lane_mask_lt(sid);
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {{
            let is_zero = (key_k & (1u << bit)) == 0u;
            let ballot0 = subgroupBallot(is_zero);
            ge_mask = select(ge_mask | ballot0, ge_mask & ballot0, is_zero);
        }}
        let rank = ballot_popc(ge_mask & bin_mask);
        smem_keys[sub_block + {k}u * SG + rank] = key_k;
        smem_vals[sub_block + {k}u * SG + rank] = values[{k}];
    }}""")
        return "\n".join(slots), sg, True

    def sort_kernel_cute_merge(self, kernel: KernelArgs):
        name = kernel.name()
        N, M, sg = kernel.N, kernel.M, kernel.R
        wpt = kernel.wpt
        RUN = sg * wpt

        base, initial_run, uses_bin_mask = self._cute_base(sg, wpt)

        passes = []
        run = initial_run
        while run < N:
            passes.append(self._merge_pass(run, wpt))
            run *= 2
        merge_passes = "\n".join(passes)

        bin_setup = ("""    let seg_lane_base = sid - (sid % SG);
    let bin_mask = lane_mask_lt(seg_lane_base + SG) & ~lane_mask_lt(seg_lane_base);
""" if uses_bin_mask else "")

        if kernel.is_block:
            store = """    for (var r = 0u; r < WPT; r = r + 1u) {{
        let pos = local_tid * WPT + r;
        if is_active && pos < seg_size {{
            global_keys[seg_start + pos] = smem_keys[pos];
            global_value_indices[seg_start + pos] = smem_vals[pos];
        }}
    }}"""
        else:
            store = """    for (var c = 0u; c < WPT; c = c + 1u) {{
        let j = c * M + local_tid;
        if is_active && j < seg_size {{
            global_keys[seg_start + j] = smem_keys[j];
            global_value_indices[seg_start + j] = smem_vals[j];
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
const SG: u32 = {sg}u;      // one CuteSort run spans SG lanes; RUN = SG*WPT = {RUN}

var<workgroup> smem_keys: array<u32, N>;
var<workgroup> smem_vals: array<u32, N>;

fn lane_mask_lt(sid: u32) -> vec4<u32> {{
    var m = vec4<u32>(0u, 0u, 0u, 0u);
    if (sid >= 32u) {{ m.x = 0xffffffffu; }} else {{ m.x = (1u << sid) - 1u; }}
    if (sid >= 64u) {{ m.y = 0xffffffffu; }} else if (sid > 32u) {{ m.y = (1u << (sid - 32u)) - 1u; }}
    if (sid >= 96u) {{ m.z = 0xffffffffu; }} else if (sid > 64u) {{ m.z = (1u << (sid - 64u)) - 1u; }}
    if (sid >= 128u) {{ m.w = 0xffffffffu; }} else if (sid > 96u) {{ m.w = (1u << (sid - 96u)) - 1u; }}
    return m;
}}

fn ballot_popc(v: vec4<u32>) -> u32 {{
    let c = countOneBits(v);
    return c.x + c.y + c.z + c.w;
}}

@compute @workgroup_size(WG, 1, 1)
fn {name}(
    @builtin(subgroup_invocation_id) sid: u32,
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

    // phase 1 (CuteSort): each subgroup sorts its RUN = SG*WPT elements.
    let sub_block = (tid_g / SG) * SG * WPT;   // this subgroup's runs live here
{bin_setup}
{base}
    workgroupBarrier();

    let base = local_tid * WPT;   // this thread's blocked output range [base, base+WPT)

{merge_passes}
{store}
}}
"""

    def sort_kernel_cute(self, kernel: KernelArgs) -> str:
        name = kernel.name()
        N = kernel.N
        store = self.store_back(kernel, "seg_start", "seg_size", "is_active", 1)
        return f"""
enable subgroups;

override WG: u32 = {N}u;

@group(0) @binding(0) var<storage, read_write> global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_value_indices: array<u32>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bin_indices: array<u32>;

const N: u32 = {N}u;
const M: u32 = {N}u;
const WPT: u32 = 1u;

var<workgroup> smem_keys: array<u32, WG>;
var<workgroup> smem_vals: array<u32, WG>;

fn lane_mask_lt(sid: u32) -> vec4<u32> {{
    var m = vec4<u32>(0u, 0u, 0u, 0u);
    if (sid >= 32u) {{ m.x = 0xffffffffu; }} else {{ m.x = (1u << sid) - 1u; }}
    if (sid >= 64u) {{ m.y = 0xffffffffu; }} else if (sid > 32u) {{ m.y = (1u << (sid - 32u)) - 1u; }}
    if (sid >= 96u) {{ m.z = 0xffffffffu; }} else if (sid > 64u) {{ m.z = (1u << (sid - 64u)) - 1u; }}
    if (sid >= 128u) {{ m.w = 0xffffffffu; }} else if (sid > 96u) {{ m.w = (1u << (sid - 96u)) - 1u; }}
    return m;
}}

fn ballot_popc(v: vec4<u32>) -> u32 {{
    let c = countOneBits(v);
    return c.x + c.y + c.z + c.w;
}}

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

    // WG == subgroup size, so one subgroup covers the whole workgroup and packs
    // WG/M segments (each M consecutive lanes = one segment). This keeps a full
    // subgroup busy even for tiny N, instead of one segment per workgroup.
    let local_tid = sid & (M - 1u);
    let seg_lane_base = sid - local_tid;            // my segment's base lane in the subgroup
    let wg_index = wg_id.x + wg_id.y * wg_dim.x;
    let global_seg = (wg_index * WG + sid) / M;

    let is_active = global_seg < bin_count;
    let slot = bin_base + select(0u, global_seg, is_active);   // clamp so the read is in-range
    let seg_id = bin_indices[slot];
    let seg_start = select(segments[seg_id - 1u], 0u, seg_id == 0u);
    let seg_end = segments[seg_id];
    let seg_size = select(0u, seg_end - seg_start, is_active);

    var key: u32;
    var value: u32;
    if is_active && local_tid < seg_size {{
        key = global_keys[seg_start + local_tid];
        value = seg_start + local_tid;
    }} else {{
        key = 0xffffffffu;                          // sentinels sort to the top, dropped at store
        value = 0xffffffffu;
    }}

    // Multisplit over the full subgroup, then confine the popcount to my
    // segment's lanes (bin_mask) so segments sharing the subgroup don't mix.
    let bin_mask = lane_mask_lt(seg_lane_base + M) & ~lane_mask_lt(seg_lane_base);
    var ge_mask = lane_mask_lt(sid);
    for (var bit = 0u; bit < 32u; bit = bit + 1u) {{
        let is_zero = (key & (1u << bit)) == 0u;
        let ballot0 = subgroupBallot(is_zero);
        ge_mask = select(ge_mask | ballot0, ge_mask & ballot0, is_zero);
    }}
    let rank = ballot_popc(ge_mask & bin_mask);     // my sorted position within the segment

    smem_keys[seg_lane_base + rank] = key;
    smem_vals[seg_lane_base + rank] = value;
    workgroupBarrier();

    var keys: array<u32, 1>;
    var values: array<u32, 1>;
    keys[0] = smem_keys[sid];
    values[0] = smem_vals[sid];

{store}
}}
"""

    def sort_kernel(self, kernel: KernelArgs):
        if kernel.mode == "reg":
            return self.sort_kernel_reg(kernel)
        elif kernel.mode == "cute":
            return self.sort_kernel_cute(kernel)
        elif kernel.mode == "wg":
            return self.sort_kernel_workgroup(kernel)
        elif kernel.mode == "hybrid":
            return self.sort_kernel_hybrid(kernel)
        elif kernel.mode == "hybmerge":
            return self.sort_kernel_hybrid_merge(kernel)
        elif kernel.mode == "cutemerge":
            return self.sort_kernel_cute_merge(kernel)

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
            if N > max(SUBGROUP_SIZES):
                continue
            kernels.add(KernelArgs(N, N, 1, N, "cute", is_block))

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
                        kernels.add(KernelArgs(N, M, wpt, sg_size, "cutemerge", is_block))

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

def test_transpose(N: int, M: int):
    t = Transposer(N, M)

    # original block matrix
    # rows = lanes
    # cols = registers

    matrix = []
    for row in range(M):
        rv = []
        matrix.append(rv)
        for col in range(t.WPT):
            rv.append(row * t.WPT + col)

    # expected write output result
    # rows = lanes
    # cols = registers

    transpose_matrix = []
    for row in range(M):
        rv = []
        transpose_matrix.append(rv)
        for col in range(t.WPT):
            rv.append(col * M + row)

    swaps, register_map = t.build()

    for lane_bit, register_bit in swaps:
        for lane in range(M):
            # simulate lanes
            lane_pair = lane ^ pow(2, lane_bit)

            for r in range(t.WPT):
                # find the right-side register
                if r & (1 << register_bit) == 0: continue # skip registers without bit set

                register_index = r # right-side for lo lane
                exchange_register_index = r ^ pow(2, register_bit) # left-side for hi lane

                if lane & pow(2, lane_bit) == 0: # check this is the low lane
                    matrix[lane][register_index], matrix[lane_pair][exchange_register_index] = matrix[lane_pair][exchange_register_index], matrix[lane][register_index]

    for row in matrix:
        row_copy = row.copy()
        for i in range(t.WPT):
            row[register_map[i]] = row_copy[i]

    # print("matrix:")
    # for row in matrix:
    #     print(f"{row}")

    # print("transpose_matrix:")
    # for row in transpose_matrix:
    #     print(f"{row}")

    return matrix == transpose_matrix

TRANSPOSER_TESTS = [
    (4, 2),
    (8, 2),
    (16, 2),
    (4, 4),
    (8, 4),
    (16, 4),
    (16, 8),
    (32, 8),
    (32, 8),
    (64, 32),
    (128, 16),
    (128, 32),
    (128, 64),
]

def test_transposer():
    total = len(TRANSPOSER_TESTS)
    failed = 0
    passed = 0

    for test in TRANSPOSER_TESTS:
        N, M = test
        if not test_transpose(N, M):
            failed += 1
            print(f"N({N}), M({M}) FAILED!!!")
        else:
            passed += 1

    if failed > 0:
        print(f"failed: {failed}/{total}")
    print(f"passed: {passed}/{total}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test-transposer":
        test_transposer()
    else:
        main()