"""Native one-GPU Muse MoE validation; synthetic weights, no serving score.

CPU source preparation:
  python tools/benchmark_serving_moe_candidates.py --plan-only --output /tmp/moe-plan.json
On an existing Hopper development node with candidate07 installed in the pinned image:
  python tools/benchmark_serving_moe_candidates.py --benchmark --output /tmp/moe-gpu.json
Defaults use small synthetic dimensions. Local EP ownership is simulated on one
GPU; this does not run inter-rank communication or validate the full model.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import traceback
from types import MethodType, SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
BASE = '297e54fd105e974cdf401e5b4d20253a22c16f90'
FILE = 'python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py'
PATCHES = {'density': '01_ep_route_density.diff', 'map': '04_fused_expert_map.diff',
           'prefix': '06_local_alignment_prefix.diff', 'combined': '07_all_candidates.diff'}
METHODS = ('_marlin_expert_outputs', '_forward_marlin')


def prepare_sources(sg, names):
    sources, manifest = {}, {}
    for name in ['stock', *names]:
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, GIT_INDEX_FILE=tmp+'/index')
            def git(*args):
                return subprocess.check_output(['git', '-C', str(sg), *args], env=env)
            git('read-tree', BASE)
            entry = {}
            if name != 'stock':
                path = ROOT/'serving/candidates'/PATCHES[name]
                data = path.read_bytes()
                entry['patch_sha256'] = hashlib.sha256(data).hexdigest()
                git('apply', '--cached', str(path))
            source = git('show', ':'+FILE).decode()
        tree = ast.parse(source)
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
                   and x.name == 'MuseSpark1xFp4Experts')
        nodes = [next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                      and x.name == method) for method in METHODS]
        # Preserve all body imports/calls and compile exact methods, without
        # invoking model construction or importing CUDA dependencies in plan mode.
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
                              names=[ast.alias(name='annotations')], level=0), *nodes],
                            type_ignores=[])
        ast.fix_missing_locations(module)
        sources[name] = compile(module, FILE+':'+name, 'exec')
        entry.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                     method_ast_sha256={x.name: hashlib.sha256(
                         ast.dump(x, include_attributes=False).encode()).hexdigest() for x in nodes})
        manifest[name] = entry
    forward_hash = manifest['stock']['method_ast_sha256']['_forward_marlin']
    assert all(x['method_ast_sha256']['_forward_marlin'] == forward_hash
               for x in manifest.values()), 'Finalization changed: extend numerical validation first'
    return sources, manifest


def run_cuda(args, codes, report, save):
    import torch
    assert torch.cuda.is_available(), 'A real CUDA GPU is required; no CPU substitution'
    assert torch.cuda.get_device_capability()[0] == 9, 'Hopper required for this component experiment'
    sys.path.insert(0, str(args.sglang/'python'))
    import sglang.srt.layers.muse_spark.muse_spark_fp4_moe as muse
    import sglang.kernels.ops.moe.moe_wna16_marlin as native
    from sglang.srt.layers.muse_spark.muse_spark_layers import (
        MuseSpark1xRMSNorm, MuseSpark1xRMSNormWithInputScale,
    )
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
    from sglang.test.test_marlin_utils import make_nvfp4_weight_and_ref
    assert Path(muse.__file__).resolve() == (args.sglang/FILE).resolve(), 'Wrong installed SGLang'
    frozen = json.loads((ROOT/'serving/PATCH_MANIFEST_LOCAL_PREFIX_2026-11-09.json').read_text())
    combined = next(x for x in frozen['patches'] if x['patch'].endswith('07_all_candidates.diff'))
    for item in combined['files']:
        assert hashlib.sha256((args.sglang/item['path']).read_bytes()).hexdigest() == item['sha256'], (
            'Install the exact frozen candidate07 sources first', item['path'])
    changed = subprocess.check_output(['git', '-C', str(args.sglang), 'diff', '--name-only', BASE]).decode().splitlines()
    assert set(changed) <= {x['path'] for x in combined['files']}, 'Unrelated SGLang modifications'
    # All common ops must be official base sources, except the new map helper.
    support = [FILE, 'python/sglang/kernels/ops/moe/map_marlin_expert_ids.py',
               'python/sglang/kernels/ops/moe/moe_wna16_marlin.py',
               'python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py',
               'python/sglang/kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh',
               'python/sglang/kernels/jit/csrc/gemm/marlin_moe/marlin_template.h']
    report['support_sha256'] = {}
    for path in support:
        data = (args.sglang/path).read_bytes()
        report['support_sha256'][path] = hashlib.sha256(data).hexdigest()
        if path not in (FILE, support[1]):
            stock = subprocess.check_output(['git', '-C', str(args.sglang), 'show', BASE+':'+path])
            assert data == stock, ('Modified common dependency', path)
    report.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  cuda=torch.version.cuda, numerical=[], graph_checks=[], timings=[])
    save()
    torch.manual_seed(11909)
    report.update(native_gpu_executed=True, phase='weight_preparation')
    save()
    dtype = torch.bfloat16
    local = args.experts//args.ep
    weights = torch.nn.Module()
    weights.quant_config = SimpleNamespace(group_size=16)
    weights.moe_runner_config = SimpleNamespace(is_gated=True)
    weights.params_dtype = dtype
    weights.intermediate_size_per_partition = args.intermediate
    refs = []
    for prefix, n, k in [('w13', 2*args.intermediate, args.hidden),
                         ('w2', args.hidden, args.intermediate)]:
        packed, scales, global_scales, reference = zip(*[
            make_nvfp4_weight_and_ref(n, k, dtype) for _ in range(local)])
        for suffix, tensors in [('weight', packed), ('weight_scale', scales),
                                ('weight_scale_2', global_scales)]:
            setattr(weights, prefix+'_'+suffix,
                    torch.nn.Parameter(torch.stack(tensors), requires_grad=False))
        refs.append(torch.stack(reference))
    humming_layer = None
    if args.humming_prototype:
        report.update(phase='humming_preparation', inflight_candidate='humming_prototype')
        save()
        spec = importlib.util.spec_from_file_location('muse_humming_prototype',
                    ROOT/'serving/prototypes/humming_moe.py')
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        humming_layer = torch.nn.Module()
        humming_layer.params_dtype = dtype
        humming_layer.num_local_experts = local
        humming_layer.global_num_experts = args.experts
        humming_layer.intermediate_size_per_partition = args.intermediate
        # Shared raw tensors during preparation only. Each backend then replaces
        # its own parameters. Two layouts coexist solely for this A/B fixture.
        for attr, tensor in weights.named_parameters():
            setattr(humming_layer, attr, tensor)
        adapter.prepare_humming_moe(humming_layer)
        humming_layer._marlin_expert_outputs = MethodType(adapter.humming_expert_outputs, humming_layer)
    prepare_moe_nvfp4_layer_for_marlin(weights)
    mapping = torch.full((args.experts,), -1, device='cuda', dtype=torch.int32)
    start = args.rank*local
    mapping[start:start+local] = torch.arange(local, device='cuda', dtype=torch.int32)
    functions = {}
    for name, code in codes.items():
        scope = dict(vars(muse))
        exec(code, scope)
        layer = torch.nn.Module()
        layer.global_num_experts, layer.num_local_experts = args.experts, local
        layer.intermediate_size_per_partition = args.intermediate
        layer._expert_map = mapping if args.ep > 1 else None
        for attr, tensor in weights.named_parameters():
            setattr(layer, attr, tensor)
        for method in METHODS:
            setattr(layer, method, MethodType(scope[method], layer))
        functions[name] = layer
    if humming_layer is not None:
        scope = dict(vars(muse))
        exec(codes['stock'], scope)
        humming_layer._forward_marlin = MethodType(scope['_forward_marlin'], humming_layer)
        humming_layer._expert_map = mapping if args.ep > 1 else None
        functions['humming_prototype'] = humming_layer
        humming_forward_module = importlib.import_module('humming.forward')
        true_humming_forward = humming_forward_module.humming_forward
        report['humming_precision'] = dict(use_f16_accum=False, use_stream_k=False,
                                         input='bf16', group_scale='fp8', output='bf16')
        save()

    def configure_norm(weight_first, fused):
        cls = MuseSpark1xRMSNormWithInputScale if weight_first else MuseSpark1xRMSNorm
        norm = cls(args.hidden, eps=1e-6).to(device='cuda', dtype=dtype)
        norm.weight.copy_(torch.linspace(.5, 1.5, args.hidden, device='cuda', dtype=dtype))
        for layer in functions.values():
            object.__setattr__(layer, '_post_expert_norm', norm)
            layer._post_expert_norm_weight = norm.weight.detach() if fused else None
            layer._post_expert_norm_eps = norm.eps
            layer._post_expert_norm_weight_first = weight_first
        return norm

    if args.norm_overlap:
        from serving_norm_overlap_cuda import run as run_norm_overlap
        with torch.inference_mode():
            norm = configure_norm(True, True)
            run_norm_overlap(args, functions['stock'], norm, report, save)
        return

    def inputs(rows, pattern):
        ids = torch.rand(rows, args.experts, device='cuda').topk(args.topk, dim=1).indices.int()
        if pattern == 'local':
            ids.copy_((torch.arange(args.topk, device='cuda')+start).expand_as(ids))
        elif pattern == 'peer':
            peer = ((args.rank+1) % args.ep)*local
            ids.copy_((torch.arange(args.topk, device='cuda')+peer).expand_as(ids))
        elif pattern == 'dropped':
            ids.fill_(-1)
        elif pattern == 'skew':
            ids[:, 0] = start
        scores = torch.rand(rows, args.topk, device='cuda', dtype=torch.float32)
        scores /= scores.sum(1, keepdim=True)
        x = torch.randn(rows, args.hidden, device='cuda', dtype=dtype)/20
        local_ids = torch.where(ids < 0, -1, mapping[ids.clamp_min(0)])
        return x, scores, ids, local_ids

    def metrics(actual, expected):
        delta = (actual.float()-expected.float()).abs()
        return {'max_abs': float(delta.max()) if delta.numel() else 0.,
                'rms': float(delta.square().mean().sqrt()) if delta.numel() else 0.,
                'exact': bool(torch.equal(actual, expected))}

    true_gemm = native.moe_wna16_marlin_gemm
    with torch.inference_mode():
        for weight_first, fused in [(True, True), (False, True), (True, False), (False, False)]:
            norm = configure_norm(weight_first, fused)
            for rows in args.rows:
                for pattern in ('random', 'local', 'peer', 'dropped', 'skew'):
                    report.update(phase='numerical', inflight_cell=dict(
                        rows=rows, pattern=pattern, weight_first=weight_first, fused=fused))
                    save()
                    x, scores, ids, local_ids = inputs(rows, pattern)
                    inactive = local_ids < 0
                    scores[inactive] = float('nan')
                    saved_inputs = [tensor.clone() for tensor in (x, scores, ids)]

                    def poisoned_gemm(a, c, *pos, **kw):
                        c.fill_(float('nan'))
                        c[inactive.reshape(-1), 0] = float('inf')
                        return true_gemm(a, c, *pos, **kw)

                    outputs = {}
                    def poisoned_humming(config, *pos, **kw):
                        c = kw['outputs']
                        c.fill_(float('nan'))
                        c[inactive.reshape(-1), 0] = float('inf')
                        return true_humming_forward(config, *pos, **kw)

                    with ExitStack() as hooks:
                        hooks.enter_context(patch.object(native, 'moe_wna16_marlin_gemm', side_effect=poisoned_gemm))
                        if humming_layer is not None:
                            hooks.enter_context(patch.object(humming_forward_module, 'humming_forward',
                                                             side_effect=poisoned_humming))
                        for name, layer in functions.items():
                            report['inflight_candidate'] = name
                            save()
                            projected, mapped = layer._marlin_expert_outputs(x, scores, ids)
                            actual = layer._forward_marlin(x, scores, ids, norm)
                            assert torch.equal(mapped, local_ids), ('mapping', name, rows, pattern)
                            assert torch.isfinite(actual).all(), ('nonfinite', name, rows, pattern)
                            assert torch.count_nonzero(actual[inactive.all(1)]) == 0
                            for tensor, before in zip((x, scores, ids), saved_inputs):
                                torch.testing.assert_close(tensor, before, rtol=0, atol=0, equal_nan=True)
                            outputs[name] = (projected.clone(), actual.clone())
                    stock_p, stock_y = outputs['stock']
                    cell = dict(rows=rows, pattern=pattern, weight_first=weight_first, fused=fused,
                                paired={})
                    for name, (projection, output) in outputs.items():
                        report.update(phase='paired_comparison', inflight_candidate=name)
                        save()
                        # Undefined inactive projection slots are intentionally excluded.
                        torch.testing.assert_close(projection[~inactive], stock_p[~inactive],
                                                   rtol=args.rtol, atol=args.atol)
                        torch.testing.assert_close(output, stock_y, rtol=args.rtol, atol=args.atol)
                        cell['paired'][name] = metrics(output, stock_y)
                    if rows <= 4:
                        dense = torch.zeros(rows, args.topk, args.hidden, device='cuda', dtype=dtype)
                        for row in range(rows):
                            for route in range(args.topk):
                                expert = int(local_ids[row, route])
                                if expert >= 0:
                                    gate, up = (x[row] @ refs[0][expert].T).chunk(2)
                                    dense[row, route] = (torch.nn.functional.silu(gate)*up) @ refs[1][expert].T
                        safe = scores.masked_fill(inactive, 0.)
                        expected = (norm(dense).float()*safe[..., None]).sum(1).to(dtype)
                        # Same dense-reference tolerance as the pinned upstream test.
                        for name, (_, output) in outputs.items():
                            report.update(phase='dense_reference', inflight_candidate=name)
                            save()
                            torch.testing.assert_close(output, expected, rtol=.05, atol=.25)
                        cell['dense_reference'] = True
                    report['numerical'].append(cell)
                    save()
        norm = configure_norm(True, True)
        for rows in args.rows:
            report.update(phase='graph_capture', inflight_cell=dict(rows=rows))
            save()
            x, scores, ids, local_ids = inputs(rows, 'random')
            graphs = {}
            for name, layer in functions.items():
                report['inflight_candidate'] = name
                save()
                for _ in range(3):
                    layer._forward_marlin(x, scores, ids, norm)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = layer._forward_marlin(x, scores, ids, norm)
                graphs[name] = (graph, out)
            for pattern in ('local', 'peer', 'dropped', 'random'):
                report.update(phase='graph_replay', inflight_cell=dict(rows=rows, pattern=pattern))
                save()
                fresh = inputs(rows, pattern)
                x.copy_(fresh[0]); scores.copy_(fresh[1]); ids.copy_(fresh[2])
                for graph, _ in graphs.values():
                    graph.replay()
                torch.cuda.synchronize()
                for name, (_, output) in graphs.items():
                    report['inflight_candidate'] = name
                    save()
                    expected = functions['stock']._forward_marlin(x, scores, ids, norm)
                    torch.testing.assert_close(output, expected, rtol=args.rtol, atol=args.atol)
                    torch.testing.assert_close(output, graphs['stock'][1], rtol=args.rtol, atol=args.atol)
                report['graph_checks'].append(dict(rows=rows, fresh_pattern=pattern, passed=True))
                save()
            if args.benchmark:
                report['phase'] = 'timing'
                save()
                timing = {}
                for name, layer in functions.items():
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(args.repeats):
                            timing_output = layer._forward_marlin(x, scores, ids, norm)
                    timing[name] = graph
                    for _ in range(5):
                        graph.replay()
                torch.cuda.synchronize()
                for name in [key for key in functions if key != 'stock']:
                    report['inflight_candidate'] = name
                    save()
                    samples = []
                    for round_id in range(args.rounds):
                        sample = {}
                        for which in (('stock', name) if round_id % 2 == 0 else (name, 'stock')):
                            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            begin.record(); timing[which].replay(); end.record(); end.synchronize()
                            sample[which] = begin.elapsed_time(end)*1000/args.repeats
                        samples.append(sample)
                    report['timings'].append(dict(rows=rows, candidate=name, samples_us=samples,
                        median_paired_speedup=statistics.median(s['stock']/s[name] for s in samples)))
                    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sglang', type=Path, default=ROOT/'serving/current/sglang')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--humming-prototype', action='store_true',
                        help='Include the separate development adapter; not a submission patch')
    parser.add_argument('--norm-overlap', action='store_true',
                        help='Isolated stock/candidate08 coordinator experiment with stock Marlin')
    parser.add_argument('--norm-overlap-traces', action='store_true',
                        help='Export separate eager CPU/CUDA timelines for that experiment')
    parser.add_argument('--candidates', nargs='+', choices=list(PATCHES), default=list(PATCHES))
    parser.add_argument('--rows', type=int, nargs='+',
                        default=[1, 4, 7, 8, 16, 28, 32, 112, 128, 224, 256])
    parser.add_argument('--experts', type=int, default=256)
    parser.add_argument('--ep', type=int, default=8)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--intermediate', type=int, default=128)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--rtol', type=float, default=.01)
    parser.add_argument('--atol', type=float, default=.01)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve receipts: choose a fresh output path'
    assert args.ep in (1, 2, 4, 8) and args.experts % args.ep == 0
    assert 0 <= args.rank < args.ep and 0 < args.topk <= args.experts//args.ep
    assert args.hidden % 128 == args.intermediate % 128 == 0
    assert args.hidden > 0 and args.intermediate > 0
    assert not (args.norm_overlap and args.humming_prototype), 'Keep backend and scheduling experiments isolated'
    assert not args.norm_overlap_traces or args.norm_overlap
    assert len(set(args.candidates)) == len(args.candidates)
    assert min(args.rows) > 0 and args.rounds >= 2 and args.repeats > 0
    report = dict(scope=__doc__, base=BASE, requested_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  state='preparing', native_gpu_executed=False)
    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        codes, manifest = prepare_sources(args.sglang, args.candidates)
        report['sources'] = manifest
        if args.norm_overlap:
            from preflight_serving_norm_overlap import prepare_sources as prepare_coordinator
            _, report['norm_overlap_sources'] = prepare_coordinator(args.sglang)
            report['norm_overlap_tools'] = {name: hashlib.sha256((ROOT/'tools'/name).read_bytes()).hexdigest()
                for name in ('benchmark_serving_moe_candidates.py', 'preflight_serving_norm_overlap.py',
                             'serving_norm_overlap_cuda.py')}
            compile(ast.parse((ROOT/'tools/serving_norm_overlap_cuda.py').read_text()), '<CUDA coordinator fixture>', 'exec')
        if args.humming_prototype:
            data = (ROOT/'serving/prototypes/humming_moe.py').read_bytes()
            compile(ast.parse(data.decode()), '<development Humming adapter>', 'exec')
            report['humming_prototype'] = dict(sha256=hashlib.sha256(data).hexdigest(),
                scope='Development projection adapter only; no serving integration or GPU pass.')
        if args.plan_only:
            report['state'] = 'source_preparation_passed_only'
        else:
            report['state'] = 'gpu_validation_requested'
            save()
            run_cuda(args, codes, report, save)
            report.update(state='gpu_component_passed', phase='terminal')
    except BaseException:
        report.update(state='failed', error=traceback.format_exc())
        raise
    finally:
        save()
    print(json.dumps({k: report[k] for k in ('state', 'native_gpu_executed')}))


if __name__ == '__main__':
    main()
