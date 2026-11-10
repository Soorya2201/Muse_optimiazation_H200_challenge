"""Execute stock/candidate coordination and actual norm bodies on CPU.

CUDA streams, routing and expert projections are explicit doubles. This tests
dataflow, branch preservation and norm arithmetic; it does not execute CUDA,
Marlin, collectives, graph capture or the full model, and produces no timings.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MethodType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = '297e54fd105e974cdf401e5b4d20253a22c16f90'
MODEL = 'python/sglang/srt/models/muse_spark.py'
NORMS = 'python/sglang/srt/layers/muse_spark/muse_spark_layers.py'
CANDIDATE = ROOT/'serving/candidates/08_marlin_norm_overlap.diff'


def compile_nodes(nodes, filename):
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                       names=[ast.alias(name='annotations')], level=0), *nodes],
                       type_ignores=[])
    return compile(ast.fix_missing_locations(module), filename, 'exec')


def prepare_sources(sg):
    """Read frozen sources through isolated indices; never modify the checkout."""
    codes, manifest = {}, {}
    for name in ('stock', 'overlap'):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, GIT_INDEX_FILE=tmp+'/index')
            def git(*args):
                return subprocess.check_output(['git', '-C', str(sg), *args], env=env)
            git('read-tree', BASE)
            if name == 'overlap':
                git('apply', '--cached', str(CANDIDATE))
            source = git('show', ':'+MODEL).decode()
            norm_source = git('show', ':'+NORMS).decode()
        tree = ast.parse(source)
        methods = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
                   for node in cls.body if isinstance(node, ast.FunctionDef)
                   and node.name == '_forward_experts']
        assert len(methods) == 1
        codes[name] = compile_nodes(methods, MODEL+':'+name)
        manifest[name] = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            method_ast_sha256=hashlib.sha256(ast.dump(methods[0]).encode()).hexdigest())
    norm_tree = ast.parse(norm_source)
    names = {'GainCenterType', 'calculate_gain_center', 'MuseSpark1xRMSNorm',
             'MuseSpark1xRMSNormWithInputScale'}
    nodes = [node for node in norm_tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in names]
    assert len(nodes) == len(names)
    codes['norms'] = compile_nodes(nodes, NORMS)
    manifest['norm_source_sha256'] = hashlib.sha256(norm_source.encode()).hexdigest()
    manifest['patch_sha256'] = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    manifest['python'] = sys.version
    manifest['ast_hash_scope'] = 'Interpreter-specific AST schema; compare source SHA256 across Python versions'
    return codes, manifest


def run_cpu(codes):
    import enum
    import math
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.manual_seed(11010)
    # Execute the stock CUDA-selected norm body on CPU tensors. This bypasses
    # sgl_kernel CPU extensions, not the arithmetic of either norm class.
    norm_scope = dict(torch=torch, nn=torch.nn, F=F, enum=enum, math=math, _is_cpu=False)
    exec(codes['norms'], norm_scope)
    results = dict(paired_cases=0, eligible_nonempty_cases=0, empty_cases=0,
                   fallback_nonempty_cases=0, exact_outputs=0,
                   norm_kind_cases={}, configurations={})
    for mode, backend, sp, norm_kind, projection, split, capture, rows, dtype in itertools.product(
        ('cpu', 'cuda_unavailable', 'simulated_cuda'), ('marlin', 'trt', 'bf16'),
        (False, True), ('none', 'standard', 'weight_first'), (False, True),
        (False, True), (False, True), (0, 1, 7, 32, 129),
        (torch.float32, torch.bfloat16)):
        idx = results['paired_cases']
        x = torch.randn(rows, 16, dtype=dtype)
        supplied = x.float().clone() if idx % 3 == 0 and not split else None
        weight = torch.randn(8, 16, dtype=torch.float32)/8
        pre_weight = torch.randn(16, 16, dtype=dtype)/4
        saved_x = x.clone()
        saved_supplied = None if supplied is None else supplied.clone()
        outputs, runs = {}, {}
        for name in ('stock', 'overlap'):
            events, norm_inputs, routed_inputs, expert_inputs = [], [], [], []
            cuda_active = mode == 'simulated_cuda'
            class Stream:
                def __init__(self, tag):
                    self.tag = tag
                def wait_stream(self, other):
                    events.append(self.tag+'_wait_'+other.tag)
            main, side = Stream('main'), Stream('side')
            active = ['main']
            @contextlib.contextmanager
            def stream_context(stream):
                old = active[0]
                active[0] = stream.tag
                try:
                    yield
                finally:
                    active[0] = old
            class TracedNorm(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    cls = norm_scope['MuseSpark1xRMSNorm' + (
                        'WithInputScale' if norm_kind == 'weight_first' else '')]
                    self.actual = cls(16, eps=1e-6, zero_centered_gamma=True).to(dtype=dtype)
                    self.actual.weight.data.copy_(torch.linspace(-.3, .7, 16, dtype=dtype))
                    if idx % 2:
                        self.actual.precompute_effective_weight()
                def forward(self, value):
                    events.append('norm_'+active[0])
                    before = value.clone()
                    out = self.actual(value)
                    assert torch.equal(value, before), 'Norm mutated an input'
                    assert out.data_ptr() != value.data_ptr(), 'Nonempty norm must be out of place'
                    norm_inputs.append(value.clone())
                    return out
            norm = None if norm_kind == 'none' else TracedNorm()
            class Gate:
                def __init__(self):
                    self.weight = weight
                def __call__(self, value):
                    events.append('router_'+active[0])
                    routed_inputs.append(value.clone())
                    return value.float() @ self.weight.T
                def forward_split2(self, value):
                    events.append('split2')
                    return self(value)
            def routing(_, logits, top_k, renormalize, *, want_packed):
                events.append('topk_'+active[0])
                scores, ids = logits.sigmoid().topk(top_k, dim=1)
                scores = scores/scores.sum(1, keepdim=True)
                # The packed CUDA route handle is a tensor and participates in
                # record_stream; a Python tuple would misrepresent its API.
                return scores, ids, torch.stack((ids.float(), scores), dim=-1)
            def project(value, matrix, bias=None):
                events.append('projection_'+active[0])
                out = F.linear(value, matrix, bias)
                return (out, None) if mode == 'cpu' else out
            def dense_experts(value, scores, ids):
                expert_inputs.append(value.clone())
                # Deliberately synthetic expert arithmetic, shared by both
                # coordinators; this is not an NVFP4/Marlin numerical check.
                factors = (ids.float()+1).unsqueeze(-1)/8
                expanded = torch.tanh(value.float().unsqueeze(1)*factors)
                return (expanded*scores.unsqueeze(-1)).sum(1).to(dtype)
            class Experts:
                use_marlin = backend == 'marlin'
                post_expert_norm = object()
                def prepare_input(self, value, pre_norm):
                    events.append('prepare_'+active[0])
                    return pre_norm(value) if pre_norm else value
                def __call__(self, value, scores, ids, post_norm, *, pre_expert_norm=None,
                             prepared_input=None, packed_topk_ids=None, ar_workspace=None):
                    events.append('experts_'+active[0])
                    assert post_norm is self.post_expert_norm
                    assert ar_workspace is sentinel
                    if backend == 'bf16':
                        assert pre_expert_norm is None and prepared_input is None
                    if prepared_input is not None:
                        assert not self.use_marlin
                        value = prepared_input
                    elif pre_expert_norm is not None:
                        value = pre_expert_norm(value)
                    return dense_experts(value, scores, ids)
            def sp_experts(value, scores, ids, logits):
                events.append('sp_experts')
                if norm is not None:
                    value = norm(value)
                return dense_experts(value, scores, ids)
            def record_stream(tensor, stream):
                assert stream is main
                events.append('record_main')
            class Capturer:
                def capture(self, *, layer_id, topk_indices):
                    events.append('capture')
                    assert layer_id == 2
                    assert topk_indices.shape == (rows, 2)
            capturer = Capturer() if idx % 2 else None
            scope = dict(torch=torch, contextlib=contextlib, _is_cpu=mode == 'cpu',
                _use_split_bf16_router=lambda n: split, get_forward=lambda: SimpleNamespace(sp_active=sp),
                get_parallel=lambda: SimpleNamespace(tp_group=SimpleNamespace(ca_comm=None)),
                bf16_gemm_dispatch=project, get_global_experts_capturer=lambda: capturer,
                layernorm_sp=SimpleNamespace(sp_exit_gather=lambda ids, **kw: ids,
                                            sp_num_tokens=lambda: rows))
            exec(codes[name], scope)
            obj = SimpleNamespace(router_in_fp32=True, gate=Gate(), experts=Experts(),
                nvfp4_checkpoint=backend != 'bf16', expert_hidden_size=16,
                pre_expert_proj=(lambda value: project(value, pre_weight)) if projection else None,
                pre_expert_norm=norm, top_k=2, layer_idx=2, tp_size=8,
                _side_stream=lambda: side, _routing_op=lambda: routing,
                _forward_sp_experts=sp_experts)
            if projection:
                obj.pre_expert_proj = SimpleNamespace(weight=pre_weight)
                if mode == 'cpu':
                    obj.pre_expert_proj = lambda value: project(value, pre_weight)
            sentinel = object()
            with contextlib.ExitStack() as stack:
                for target, value in [('is_available', lambda: cuda_active),
                    ('current_stream', lambda: main), ('stream', stream_context),
                    ('get_device_capability', lambda *a: (9, 0)),
                    ('is_current_stream_capturing', lambda: capture)]:
                    stack.enter_context(patch.object(torch.cuda, target, value))
                stack.enter_context(patch.object(torch.Tensor, 'record_stream', record_stream))
                with torch.inference_mode():
                    outputs[name] = scope['_forward_experts'](obj, x, supplied, ar_workspace=sentinel)
            assert obj.pre_expert_norm is norm, 'Per-call suppression changed module state'
            assert torch.equal(x, saved_x), 'Hidden states were mutated'
            if supplied is not None:
                assert torch.equal(supplied, saved_supplied)
            assert len(norm_inputs) == int(norm is not None and rows > 0)
            if cuda_active and rows:
                assert events.index('side_wait_main') < events.index('router_side')
                assert events.index('topk_side') < events.index('main_wait_side')
                assert events.count('record_main') == (0 if capture else 3)
                if capturer:
                    assert events.index('main_wait_side') < events.index('capture')
            else:
                assert not any('wait' in event or event == 'record_main' for event in events)
            runs[name] = (events, norm_inputs, routed_inputs, expert_inputs)
        assert torch.equal(outputs['stock'], outputs['overlap']), (
            mode, backend, sp, norm_kind, projection, split, capture, rows, dtype)
        # Check both router input and normalized expert input, not only output.
        for stock_values, new_values in zip(runs['stock'][1:], runs['overlap'][1:]):
            assert len(stock_values) == len(new_values)
            assert all(torch.equal(a, b) for a, b in zip(stock_values, new_values))
        eligible = cuda_active and backend == 'marlin' and not sp and norm is not None and rows > 0
        if eligible:
            before, after = runs['stock'][0], runs['overlap'][0]
            assert before.index('main_wait_side') < before.index('norm_main')
            assert after.index('topk_side') < after.index('norm_main') < after.index('main_wait_side')
            results['eligible_nonempty_cases'] += 1
        else:
            assert runs['stock'][0] == runs['overlap'][0], 'Fallback scheduling changed'
            results['empty_cases' if rows == 0 else 'fallback_nonempty_cases'] += 1
        results['paired_cases'] += 1
        results['exact_outputs'] += 1
        results['norm_kind_cases'][norm_kind] = results['norm_kind_cases'].get(norm_kind, 0)+1
        results['configurations'][mode] = results['configurations'].get(mode, 0)+1
    return dict(results, torch=torch.__version__, cuda_executed=False,
                scope='CPU exact coordinator/norm bodies; simulated streams/routing/experts; no GPU math, capture or performance validation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sglang', type=Path, default=ROOT/'serving/current/sglang')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve earlier receipts'
    codes, manifest = prepare_sources(args.sglang)
    report = dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        base=BASE, sources=manifest, tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checks=run_cpu(codes))
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
