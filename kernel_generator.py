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
    R: int          # shuffle radius in lanes: a cross-lane sort stage whose
                    # exchange distance tmask < R runs via subgroup shuffle, the
                    # rest go through shared memory. reg: R = M (all shuffle);
                    # wg: R = 1 (all smem); hybrid: R = subgroup size (shuffle
                    # within a subgroup, smem across). Requires M > R to have a seam.
    mode: str       # reg, wg, or hybrid
    is_block: bool

    def name(self):
        store = "block" if self.is_block else "striped"
        if self.mode == "hybrid":
            # R (subgroup size) disambiguates the several hybrids sharing an (N, M)
            return f"segsort_hybrid_sg{self.R}_n{self.N}_m{self.M}_{store}"
        if self.mode == "hybrid-merge":
            # smem tag = smallest budget the single buffer (8*N bytes: keys+vals)
            # fits in, so the host can dispatch by the device's reported
            # maxComputeWorkgroupStorageSize
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

    def _exch(self, kernel: KernelArgs, pairs, tmask: int, swbit: int) -> str:
        if tmask < kernel.R:
            return self._exch_subgroup(pairs, tmask, swbit)
        return self._exch_workgroup(pairs, tmask, swbit)


    def exch_intxn(self, kernel: KernelArgs, tmask: int, swbit: int) -> str:
        wpt = kernel.wpt
        return (f"// exch_intxn(tmask:{tmask},swbit:{swbit},wpt:{wpt})\n" +
                self._exch(kernel, [(r, wpt - 1 - r) for r in range(wpt)], tmask, swbit))

    def exch_paral(self, kernel: KernelArgs, tmask: int, swbit: int) -> str:
        wpt = kernel.wpt
        return (f"// exch_paral(tmask:{tmask},swbit:{swbit},wpt:{wpt}) \n" +
                self._exch(kernel, [(r, r) for r in range(wpt)], tmask, swbit))

    def reg_sort(self, kernel: KernelArgs) -> str:
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
                stages.append(self.exch_local(coop_elem_size(l) - 1, wpt))
            else:
                stages.append(self.exch_intxn(kernel, cts - 1, swbit_of(cts)))

            for k in range(l + 1, p + 1):
                cts = coop_thrd_size(k)
                if cts == 1:
                    rmask = coop_elem_size(k) - 1
                    stages.append(self.exch_local(rmask ^ (rmask >> 1), wpt))
                else:
                    tmask = cts - 1
                    stages.append(self.exch_paral(kernel, tmask ^ (tmask >> 1), swbit_of(cts)))

        return "\n".join(stages)

    def store_back(self, kernel: KernelArgs, base: str, length: str,
                   active: str | None) -> str:
        if kernel.is_block:
            return self._block_store(base, length, active)
        # hybrid shares the smem striped store: a subgroup-shuffle transpose can't
        # reach across subgroups, but hybrid segments span several of them.
        if kernel.mode in ("wg", "hybrid"):
            return self._striped_store_smem(base, length, active)
        return self._striped_store_subgroup(base, length, active, kernel.N, kernel.M)

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
        # smem is shared by every segment in the workgroup (WG can be a multiple of
        # M -> WG/M segments per workgroup), so index it by the WORKGROUP-relative
        # position, not the segment-relative local_tid. Writing/reading at
        # local_tid*WPT would make all segments collide in smem[0, M*WPT). Each
        # segment owns smem[seg_base*WPT, seg_base*WPT + M*WPT); tid_g*WPT ==
        # seg_base*WPT + local_tid*WPT is this thread's blocked slot within it.
        return (
            "    // striped (coalesced) store via shared memory\n"
            "    for (var r = 0u; r < WPT; r = r + 1u) {\n"
            "        smem_keys[tid_g * WPT + r] = keys[r];\n"
            "        smem_vals[tid_g * WPT + r] = values[r];\n"
            "    }\n"
            "    workgroupBarrier();\n"
            "    for (var c = 0u; c < WPT; c = c + 1u) {\n"
            "        let j = c * M + local_tid;\n"
            f"        if {guard} {{\n"
            f"            global_keys[{base} + j] = smem_keys[seg_base * WPT + j];\n"
            f"            global_value_indices[{base} + j] = smem_vals[seg_base * WPT + j];\n"
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

    def sort_kernel_reg(self, kernel: KernelArgs) -> str:
        name = kernel.name()
        N = kernel.N
        M = kernel.M

        wpt = N // M
        stages = "\n".join("    " + ln for ln in self.reg_sort(kernel).split("\n"))
        store = self.store_back(kernel, "seg_start", "seg_size", "is_active")
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

    def sort_kernel_workgroup(self, kernel: KernelArgs):
        name = kernel.name()
        N = kernel.N
        M = kernel.M

        wpt = N // M
        stages = "\n".join("    " + ln for ln in self.reg_sort(kernel).split("\n"))
        store = self.store_back(kernel, "seg_start", "seg_size", "is_active")
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

    def sort_kernel_hybrid(self, kernel: KernelArgs):
        name = kernel.name()
        N = kernel.N
        M = kernel.M
        R = kernel.R

        wpt = N // M
        # reg_sort emits shuffle stages for tmask < R and smem stages beyond, so
        # this one body carries both. WG = M spans several subgroups; the segment
        # is one workgroup (seg_base == 0), each thread holds WPT elems.
        stages = "\n".join("    " + ln for ln in self.reg_sort(kernel).split("\n"))
        store = self.store_back(kernel, "seg_start", "seg_size", "is_active")
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

    def sort_kernel_hybrid_merge(self, kernel: KernelArgs):
        # bb_segsort-style hybrid: each subgroup register-sorts a run of SG*WPT
        # elements (full-subgroup bitonic via shuffle), then those runs are merged
        # pairwise up to the whole segment with MERGE-PATH merges through smem,
        # ping-ponging two buffers. R (== SG) is the register-run width in lanes.
        name = kernel.name()
        N, M, sg = kernel.N, kernel.M, kernel.R
        wpt = kernel.wpt
        RUN = sg * wpt                       # elements in one register-sorted run
        P = int(log2(M // sg))               # number of merge passes (M >= 2*SG => P >= 1)

        # Phase 1 reuses the register sort restricted to one subgroup (SG lanes,
        # R = SG so every stage shuffles -- no smem).
        run_kernel = KernelArgs(RUN, sg, wpt, sg, "reg", kernel.is_block)
        phase1 = "\n".join("    " + ln for ln in self.reg_sort(run_kernel).split("\n"))

        # Phase 2: register-staged merge passes over a SINGLE smem buffer
        # (bb_segsort style). Each thread merge-paths its WPT outputs into
        # registers, barriers so every read is done, then writes them back to the
        # same buffer -- the registers are the "pong", so no second buffer.
        #
        # The read->barrier->write-back is a WAR hazard on smem. A plain
        # workgroupBarrier() (threadgroup-scope memory fence) is NOT sufficient to
        # order the write-back for single-SIMD-group workgroups (M <= subgroup):
        # it fails for M <= 32 & wpt >= 8 on this 32-lane device with a torn read
        # (correct value index, wrong key). A SECOND barrier at the same scope
        # does not help; a storageBarrier() (storage/device-scope fence, Metal
        # mem_device) DOES -- so we emit both. Measured cost of the extra fence is
        # ~1% worst-case (5-pass config) and it keeps the single 8*N buffer and
        # full WG=M utilization. Emitted unconditionally (device-agnostic): large
        # multi-SIMD-group configs don't need it but pay <=1%.
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

    // ---- phase 1: per-subgroup register run-sort (RUN = SG*WPT elements) ----
{phase1}

    // stage the sorted runs into shared memory (blocked layout)
    for (var r = 0u; r < WPT; r = r + 1u) {{
        smem_keys[local_tid * WPT + r] = keys[r];
        smem_vals[local_tid * WPT + r] = values[r];
    }}
    workgroupBarrier();

    let base = local_tid * WPT;   // this thread's blocked output range [base, base+WPT)

    // ---- phase 2: recursive merge-path merges through shared memory ----
{merge_passes}

    // ---- phase 3: coalesced store from the final buffer ----
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
        elif kernel.mode == "hybrid-merge":
            return self.sort_kernel_hybrid_merge(kernel)

    def sort_kernel_n1(self):
        # single-element segments (bin 0, N=1): the sort is a no-op, we only need
        # to write the identity value index. Statically known, but emitted here so
        # a clear-the-folder regen can't drop it.
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
        stages = "\n".join("    " + ln for ln in self.reg_sort(kernel).split("\n"))
        store = self.store_back(kernel, "tile_lo", "tile_len", None)
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

# --- sweep bounds ------------------------------------------------------------
MAX_WG = 256                    # WebGPU maxComputeWorkgroupSizeX / invocations
SMEM_BUDGETS = [16384, 32768]   # supported workgroup-storage budgets (bytes)
WPT_SWEEP = [2, 4, 8, 16, 32]   # registers per thread swept for the smem/hybrid arms

def main():
    args = sys.argv[1:]
    output_dir = args[0]

    print("Generating all kernels")

    kernels = set()

    for is_block in [False, True]:
        # reg kernels: per subgroup size, the widest viable run (M = min(sg, N)).
        # Pure register sort needs M <= subgroup, so WPT = N/M is fixed per (N, sg);
        # sweeping M below this only inflates WPT, so we keep the single best point.
        for sg_size in SUBGROUP_SIZES:
            for N in SEGMENT_SIZES:
                M = min(sg_size, N)
                kernels.add(KernelArgs(N, M, N // M, M, "reg", is_block))

        # swept arms: WPT (registers/thread) is the knob, M = N / WPT.
        for N in SEGMENT_SIZES:
            for wpt in WPT_SWEEP:
                if N % wpt != 0:
                    continue
                M = N // wpt
                if M < 1 or M > MAX_WG:
                    continue

                # wg (pure smem): every cross-lane stage through shared memory.
                kernels.add(KernelArgs(N, M, wpt, 1, "wg", is_block))

                for sg_size in SUBGROUP_SIZES:
                    # bitonic hybrid: needs a shuffle->smem seam, i.e. M > sg.
                    if M > sg_size:
                        kernels.add(KernelArgs(N, M, wpt, sg_size, "hybrid", is_block))
                    # merge hybrid: reg-run per subgroup + register-staged merge
                    # path. Needs >= 1 merge pass (M >= 2*sg) and the single smem
                    # buffer to fit (2 arrays * N * 4B = 8N).
                    if M >= 2 * sg_size and 8 * N <= SMEM_BUDGETS[-1]:
                        kernels.add(KernelArgs(N, M, wpt, sg_size, "hybrid-merge", is_block))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for kernel in kernels:
        print(f"Kernel: {kernel}")

        generator = KernelGenerator()
        source = generator.sort_kernel(kernel)

        with open(f"{output_dir}/{kernel.name()}.wgsl", "w") as f:
            f.write(source)

    for is_block in (False, True):
        # tile kernel is a plain smem sort (R = 1) over a fixed 2048-key tile
        tile_k = KernelArgs(2048, 256, 2048 // 256, 1, "wg", is_block)
        tile_gen = KernelGenerator()
        tile_src = tile_gen.tile_sort_kernel(tile_k, "segsort_tile_n2048_m256")
        store = "block" if is_block else "striped"
        with open(f"{output_dir}/segsort_tile_n2048_m256_{store}.wgsl", "w") as f:
            f.write(tile_src)

    # single-element segments (bin 0): statically known no-op sort
    with open(f"{output_dir}/segsort_wg_n1_m1_block.wgsl", "w") as f:
        f.write(KernelGenerator().sort_kernel_n1())


if __name__ == "__main__":
    main()
