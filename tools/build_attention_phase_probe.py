"""Create a dev-only attention phase diagnostic with an unchanged timed kernel."""
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parent = (ROOT / "muse_attention_decode/submission_v15_vjeux_f35a41e.py").read_text()
tail = r'''

# Development-only import-time measurements on independently generated fixtures.
# Never promote this diagnostic file: the timed entry point above remains the parent.
def _phase_source(source):
    helper = r"""
__device__ __forceinline__ void phase_mark(const Params& p, int slot, int who = 0, bool first = true) {
  if (p.dbg && threadIdx.x == who) {
    const int c = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    volatile long long* dst = p.dbg + c * 16 + slot;
    if (!first || *dst == 0) {
      long long now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
      *dst = now;
    }
  }
}
"""
    def insert(anchor, text, before=False):
        nonlocal source
        assert source.count(anchor) == 1, (anchor[:90], source.count(anchor))
        source = source.replace(anchor, text + anchor if before else anchor + text)
    insert('__device__ __forceinline__ uint32_t smem_u32(const void* p) {', helper, True)
    insert('  const int pair = blockIdx.z;\n', '  phase_mark(p, 0, 0, false);\n')
    insert('  const int ntiles = ONE ? 1 : (end - start + TILE - 1) / TILE;', '\n  phase_mark(p, 1);\n')
    insert('          load_pages_t(x, pg);\n', '          if (x == 0 && q == 0) { asm volatile("" :: "r"(pg[0]), "r"(pg[PK-1])); phase_mark(p, 2, NT); }\n')
    insert('      if constexpr (!TMAPG_ON) load_pages(0, pg);\n', '      if constexpr (!TMAPG_ON) { if (q == 0) { asm volatile("" :: "r"(pg[0]), "r"(pg[PK-1])); phase_mark(p, 2, NT); } }\n')
    insert('        cp_async_mbar_arrive(&S.full[st]);\n', '        if (x == 0) phase_mark(p, 3, NT);\n')
    insert('        cp_wait<0>();\n        __syncthreads();   // matches the consumers\' end-of-segment barrier', '        phase_mark(p, 12, NT, false);\n', True)
    insert('  // ---- consumer state ----\n', '  phase_mark(p, 4);\n', True)
    insert('    const uint8_t* K = S.k + ((tb + i) % STAGES) * STAGE_BYTES;\n', '    if (i == 0) phase_mark(p, 5);\n', True)
    insert('    const int st = st_c;\n', '    if (i == 0) phase_mark(p, 5);\n', True)
    insert('  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----\n', '  phase_mark(p, 7, 0, false);\n', True)
    insert('  if ((FLAT && (L == 8)) && M_COOP) {\n', '  phase_mark(p, 8, 0, false);\n', True)
    insert('    const int split_idx = (int)blockIdx.x;\n', '    phase_mark(p, 9);\n', True)
    insert('    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);\n', '    phase_mark(p, 10);\n', True)
    insert('  } else {\n  if (tid == 0) {\n    const int mine = end - start;\n', '    phase_mark(p, 11, 0, false);\n', True)
    insert('  if (S.flag) {\n', '  phase_mark(p, 10);\n')
    insert('  if (tid == 0) p.ws_cnt[bp] = 0;\n', '  phase_mark(p, 11, 0, false);\n', True)
    insert('  pdl_wait();   // a CTA without segments has not waited yet (the gate rows are qkv)\n', '  phase_mark(p, 9);\n')
    insert('  const int rows = p.B * pairs * L * 16;\n', '  phase_mark(p, 10);\n', True)
    insert('\n}\n\ntemplate <int L, bool FLAT, int MODE = 0>\n__global__', '\n  phase_mark(p, 11, 0, false);\n', True)
    return source


def _phase_fixture(B, L, window, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    longest = 86000 if B > 1 else 80000
    max_pages = longest + L
    seq = ((74000 + torch.rand((B,), generator=gen, device=device) * 12000).long()
           if B > 1 else torch.full((1,), 80000, dtype=torch.long, device=device))
    table = torch.randperm(B * max_pages, generator=gen, device=device).int().view(B, max_pages)
    shape = (B * max_pages, 1, 2, 64)
    k = torch.randn(shape, generator=gen, device=device).to(torch.float8_e4m3fn)
    v = torch.randn(shape, generator=gen, device=device).to(torch.float8_e4m3fn)
    qkv = (2 * torch.randn((B, L, 2304), generator=gen, device=device)).to(torch.bfloat16)
    positions = seq[:, None] + torch.arange(L, device=device)
    return {"qkv": qkv, "seq_lens": seq.int(), "cache_loc": table.long().gather(1, positions).reshape(-1),
            "positions": positions.reshape(-1), "page_table": table, "k_cache": k, "v_cache": v,
            "k_scale": torch.ones((), device=device), "v_scale": torch.ones((), device=device),
            "cache_seqlens": (seq+L).int(), "cu_seqlens_q": None, "scheduler_metadata": None,
            "config": {"batch": B, "q_len": L, "num_q_heads": 16, "num_kv_heads": 2,
                       "head_dim": 64, "page_size": 1, "window": window,
                       "rope_theta": 500000.0 if window else 0.0,
                       "sm_scale": 0.125, "qk_scale_factor": 5.4730064863875, "kv_scale": 1.0}}


def _phase_pair(fn0, fn1, flush, reps=12):
    import statistics
    graphs = []
    for fn in (fn0, fn1):
        fn(); fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn()
        graphs.append((graph, out))
    torch.cuda.synchronize()
    samples = [[], []]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for rep in range(reps):
        for j in ((0, 1) if rep % 2 == 0 else (1, 0)):
            flush.add_(1)
            start.record(); graphs[j][0].replay(); end.record(); end.synchronize()
            samples[j].append(start.elapsed_time(end) * 1000.0)
    return tuple(statistics.median(x) for x in samples)


def _phase_probe():
    import concurrent.futures
    import json
    import time
    from torch.utils.cpp_extension import load_inline
    global _MOD
    started = time.perf_counter()
    traced = _phase_source(CUDA_SRC)
    def build_debug():
        return load_inline(name="okc_attn_v317_phase_debug_90a", cpp_sources=CPP_SRC,
            cuda_sources=traced, functions=["run"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr"],
            verbose=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        normal = pool.submit(_build)
        debug = pool.submit(build_debug)
        _MOD, debug = normal.result(), debug.result()
    print(f"PHASE builds {time.perf_counter()-started:.1f}s", flush=True)
    dev = torch.device("cuda")
    flush = torch.zeros((64 << 20,), dtype=torch.float32, device=dev)
    fields = {0:"entry", 1:"segment_entry", 2:"first_pages_ready", 3:"first_copies_issued",
              4:"query_ready", 5:"first_KV_ready", 7:"consumer_loop_end", 8:"partial_issue_end",
              9:"finalize_enter", 10:"merge_ready", 11:"output_issued", 12:"producer_issue_end"}
    for B, L, window in ((1,1,2048),(1,8,2048),(1,1,0),(1,8,0),(16,1,0),(32,1,0),
                          (16,8,2048),(32,8,2048),(16,8,0),(32,8,0)):
        if time.perf_counter()-started > 360:
            print("PHASE measurement budget reached", flush=True)
            break
        data = _phase_fixture(B,L,window, 39047+31*B+L+window,dev)
        cfg = data["config"]
        plan = _plan(B,1,L,window,data["page_table"].shape[1],1,dev)
        plan.update(_TUNED.get((B,L,bool(window),1), {}))
        count = plan["n_splits"] * (1 if plan["flat"] else B)
        trace = torch.zeros((count,16), dtype=torch.int64, device=dev)
        fn0 = lambda: _launch(data, _mod=_MOD)
        fn1 = lambda: _launch(data, dbg=trace, _mod=debug)
        y0, y1 = fn0().float(), fn1().float()
        relerr = float((y0-y1).norm()/y0.norm().clamp_min(1e-12))
        assert relerr < 0.002, (B,L,window,relerr)
        old, instrumented = _phase_pair(fn0,fn1,flush)
        # Read one independent cold trace. Reset outside the measured call.
        trace.zero_(); flush.add_(1); fn1(); torch.cuda.synchronize()
        stamps = trace.cpu().tolist()
        entry = [row[0] for row in stamps if row[0] > 0]
        origin = min(entry)
        phases = {}
        for col, name in fields.items():
            values = sorted((row[col]-origin)/1000.0 for row in stamps if row[col] > 1)
            if values: phases[name] = [round(values[0],3),round(values[len(values)//2],3),round(values[-1],3)]
        print("PHASE "+json.dumps({"B":B,"L":L,"window":window,"plan":plan,"relerr":relerr,
              "parent_raw_us":old,"instrumented_raw_us":instrumented,"min_median_max_us":phases}), flush=True)
        del data, trace, y0, y1, stamps
    # Scratch is coordination state only. Drop every synthetic input/output/graph.
    for buffers in _WS.values():
        for buffer in buffers: buffer.zero_()
    torch.cuda.synchronize()
    print(f"PHASE done {time.perf_counter()-started:.1f}s; timed entry remains exact parent",flush=True)


if torch.cuda.is_available():
    try:
        _phase_probe()
    except Exception as _phase_error:
        print(f"PHASE failed: {str(_phase_error)[-1500:]}", flush=True)
        # A compile-only failure can leave normal setup to build the parent as usual.
        for _phase_buffers in _WS.values():
            for _phase_buffer in _phase_buffers: _phase_buffer.zero_()
'''
target = "muse_attention_decode/submission_v17_phase_probe.py"
assert not (ROOT / target).exists(), "Do not overwrite an archived experiment"
source = parent.replace("okc_attn_v298{tag}_90a", "okc_attn_v317_phase_parent{tag}_90a") + tail
ast.parse(source)
namespace = {}
nodes = ast.parse(source)
native = next(ast.literal_eval(n.value) for n in nodes.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=="CUDA_SRC" for t in n.targets))
fun = next(n for n in nodes.body if isinstance(n,ast.FunctionDef) and n.name=="_phase_source")
exec(compile(ast.Module(body=[fun], type_ignores=[]),"<trace builder>","exec"),namespace)
instrumented = namespace["_phase_source"](native)
assert instrumented.count("phase_mark(") > 15
assert source.startswith(parent.replace("okc_attn_v298{tag}_90a", "okc_attn_v317_phase_parent{tag}_90a"))
patch = "*** Begin Patch\n*** Add File: " + target + "\n" + "\n".join("+"+s for s in source.splitlines()) + "\n*** End Patch\n"
subprocess.run(["apply_patch",patch],cwd=ROOT,check=True)
print("PASS: exact parent timed prefix; all instrumentation anchors resolved; syntax passes")
