"""Expose the parent's dormant two-score attention pipeline, with controls.

Changes only native dispatch/resource wrappers. The existing PIPE3 body is
already in the pinned Apache-2.0 parent; no FP8 dense projections are ported.
"""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parent = ROOT / "muse_prefill_layer/submission_v20_context_bind.py"
target = ROOT / "muse_prefill_layer/submission_v23_dual_score_probe.py"
source = parent.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == "8b52dedda281b2e642bda1d9333bcf2d002fc8611e25caa10e90ed2a849021a5"
assert not target.exists()
tree = ast.parse(source)
build_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build")
build = ast.get_source_segment(source, build_node)
assert build.count('"attn_set_prof"]') == 1
build = build.replace('"attn_set_prof"]', '"attn_set_prof", "attn_resources"]')
suffix = r'''

# Independent diagnostic. Timed full-layer dispatch remains parent8/43.
import base64 as _DSBase64
import json as _DSJSON
import math as _DSMath
import statistics as _DSStats
import time as _DSTime
import traceback as _DSTrace
import zlib as _DSZlib

_ds_case = "    case 43: okc_launch(okc_attn::attn_fwd_kernel<3, 0, false, true, false, true>, grid, dim3(512), (size_t)smem, st, a); break;"
_ds_attr = "  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 0, false, true, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);"
_ds_templates = {
    43: "<3, 0, false, true, false, true>",
    48: "<2, 0, false, true, false, true>",
    49: "<2, 0, false, false, false, false, false, false, true>",
}
assert GLUE.count(_ds_case) == GLUE.count(_ds_attr) == 1
for _ds_v in (48, 49):
    _ds_launch = _ds_case.replace("case 43:", f"case {_ds_v}:").replace(_ds_templates[43], _ds_templates[_ds_v]).replace("dim3(512)", "dim3(384)")
    GLUE = GLUE.replace(_ds_case, _ds_case + "\n" + _ds_launch)
    GLUE = GLUE.replace(_ds_attr, _ds_attr + "\n" + _ds_attr.replace(_ds_templates[43], _ds_templates[_ds_v]))
    ATTN_VARIANTS[_ds_v] = (2, 0)
    assert not (_ds_v & 8)
_ds_resources = "\n#include <vector>\nstd::vector<int64_t> k_attn_resources(int variant) {\n cudaFuncAttributes a = {}; cudaError_t e = cudaErrorInvalidValue;\n switch (variant) {\n"
for _ds_v, _ds_t in _ds_templates.items():
    _ds_resources += f" case {_ds_v}: e = cudaFuncGetAttributes(&a, okc_attn::attn_fwd_kernel{_ds_t}); break;\n"
_ds_resources += " }\n return {(int64_t)variant, (int64_t)e, (int64_t)a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes, (int64_t)a.maxDynamicSharedSizeBytes, (int64_t)a.maxThreadsPerBlock};\n}\n"
GLUE += _ds_resources
CPP += "\nstd::vector<int64_t> k_attn_resources(int variant);\nstd::vector<int64_t> attn_resources(int64_t v) { return k_attn_resources((int)v); }\n"
'''
suffix += "\n" + build + "\n"
suffix += r'''

def _ds_capture(st, data, cfg, q8, qkv, variant, splits):
    def run():
        return _attn_custom(st, data, q8, qkv, cfg, nsplit=splits, variant=variant)
    run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = run()
    g.replay()
    torch.cuda.synchronize()
    return g, out


def _ds_error(out, ref):
    a, b = out.float(), ref.float()
    assert bool(torch.isfinite(a).all()), "nonfinite diagnostic output"
    return float((a-b).norm()/b.norm().clamp_min(1e-12))


def _ds_measure(graphs, group, pad, flush, sink):
    names = list(graphs)
    times = {k: [] for k in names}
    for rep in range(9):
        shift = (rep + group + pad) % len(names)
        order = names[shift:] + names[:shift]
        if (rep + group) % 2:
            order.reverse()
        for k in order:
            sink.copy_(flush.sum(dtype=torch.float32))
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[k][0].replay()
            end.record()
            end.synchronize()
            times[k].append(round(start.elapsed_time(end)*1000, 4))
    ratios = {k: round(_DSStats.median(_DSMath.sqrt(a*b)/c for a,b,c in zip(times['a0'], times['a1'], times[k])), 6)
              for k in names if k not in ('a0','a1')}
    return dict(us=times, speedup=ratios,
                aa=round(_DSStats.median(a/b for a,b in zip(times['a0'],times['a1'])), 6))


def _ds_probe(st):
    report = dict(status='starting', stage='resources', rows=[])
    start = _DSTime.monotonic()
    try:
        fields = ('variant','cuda_error','numRegs','localSizeBytes','sharedSizeBytes',
                  'maxDynamicSharedSizeBytes','maxThreadsPerBlock')
        report['resources'] = [dict(zip(fields, st['mod'].attn_resources(v))) for v in (43,48,49)]
        assert all(r['cuda_error'] == 0 for r in report['resources'])
        report['stage'] = 'nearby FA3 validation'
        valid = _validate_attn(st, [43,48,49])
        report['valid_small'] = valid
        assert 43 in valid
        flush = torch.ones(128 << 20, dtype=torch.float16, device='cuda')
        sink = torch.zeros((), dtype=torch.float32, device='cuda')
        geometries = (('t1f',1,753,80000), ('t4f',4,753,80000),
                      ('first0',1,16384,0), ('first32768',1,16384,32768),
                      ('first65536',1,16384,65536))
        for group, seed in enumerate((3101,7907)):
            for pad in (0,1,3,7):
                keep_pad = torch.empty((pad << 20,), dtype=torch.uint8, device='cuda')
                for label, batch, length, context in geometries:
                    report['stage'] = f'fixture {group}/{pad}/{label}'
                    data,cfg,q8,qkv = _synthetic_attn_case(batch,length,context,0,1,seed)
                    graphs, splits = {}, {}
                    for name,v in [('a0',43),('a1',43),('48',48),('49',49)]:
                        if v not in valid:
                            continue
                        ns = _attn_splits(cfg,data,v,st['nsm'])
                        splits[str(v)] = ns
                        graphs[name] = _ds_capture(st,data,cfg,q8,qkv,v,ns)
                    ref = graphs['a0'][1].clone()
                    errors = {}
                    for name in ('48','49'):
                        if name in graphs:
                            errors[name] = _ds_error(graphs[name][1],ref)
                            if errors[name] >= .01:
                                del graphs[name]
                    record = dict(case=label,group=group,seed=seed,pad_mib=pad,
                                  splits=splits,errors=errors)
                    report['stage'] = f'timing {group}/{pad}/{label}'
                    record.update(_ds_measure(graphs,group,pad,flush,sink))
                    # Captures must consume changed query, gate, and KV values.
                    q8.copy_((torch.randn_like(q8.float())*.7).to(FP8))
                    qkv.neg_()
                    data['v_cache'].copy_((-data['v_cache'].float()).to(FP8))
                    for graph,out in graphs.values():
                        graph.replay()
                    torch.cuda.synchronize()
                    change = _ds_error(graphs['a0'][1],ref)
                    assert change > .001, 'fresh fixture did not change parent result'
                    fresh = {k:_ds_error(pair[1],graphs['a0'][1]) for k,pair in graphs.items() if k not in ('a0','a1')}
                    assert all(e < .01 for e in fresh.values()), 'fresh replay mismatch'
                    record.update(fresh_error=fresh,parent_change=change)
                    report['rows'].append(record)
                    del graphs,ref,data,cfg,q8,qkv
                del keep_pad
        report.update(status='complete',stage='finished')
    except Exception as exc:
        report.update(status='failed',error=repr(exc),trace=_DSTrace.format_exc())
    report['elapsed_s'] = round(_DSTime.monotonic()-start,3)
    payload = _DSBase64.b64encode(_DSZlib.compress(_DSJSON.dumps(report,separators=(',',':')).encode(),9)).decode()
    print('PREFILL_DUAL_SCORE_Z '+payload,flush=True)


_ds_parent_prepare = prepare
def prepare(weights):
    st = _ds_parent_prepare(weights)
    _ds_probe(st)
    assert st['attn_choice'] == ATTN_FIXED
    return st
'''
result = source + suffix
ast.parse(result)
target.write_text(result)
print(target.relative_to(ROOT), hashlib.sha256(result.encode()).hexdigest())
