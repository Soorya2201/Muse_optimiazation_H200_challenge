"""Exercise the exact candidate06 method with CPU align/GEMM contract doubles.

This validates argument wiring and route coverage, not native CUDA alignment,
Marlin arithmetic, CUDA graph safety or serving quality/performance.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT/'serving/current/sglang'
BASE = '297e54fd105e974cdf401e5b4d20253a22c16f90'
FILE = 'python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py'
PATCH = ROOT/'serving/candidates/06_local_alignment_prefix.diff'
REPORT = ROOT/'serving/LOCAL_PREFIX_CPU_2026-11-09.json'


def source_at_patch():
    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        env['GIT_INDEX_FILE'] = tmp+'/index'
        def git(*args):
            return subprocess.check_output(['git', '-C', str(SG), *args], env=env)
        git('read-tree', BASE)
        git('apply', '--cached', str(PATCH))
        return git('show', ':'+FILE).decode()


def main():
    assert not REPORT.exists(), 'Keep earlier evidence'
    torch.set_num_threads(1)
    source = source_at_patch()
    tree = ast.parse(source)
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
               and any(isinstance(y, ast.FunctionDef) and y.name == '_marlin_expert_outputs'
                       for y in x.body))
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                  and x.name == '_marlin_expert_outputs')
    # Replace imports with instrumented CPU doubles; execute the actual method body.
    class RemoveImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    method = RemoveImports().visit(method)
    method.decorator_list = []
    active = {}

    def align(ids, block, experts, *, return_inactive_prefix_count):
        assert experts == active['local_experts']
        assert return_inactive_prefix_count
        flat = ids.reshape(-1)
        assert ((flat >= -1) & (flat < experts)).all()
        tokens, blocks = [], []
        inactive = 0
        for expert in range(-1, experts):
            indices = torch.where(flat == expert)[0].tolist()
            if not indices:
                continue
            pad = (-len(indices)) % block
            tokens.extend(indices + [flat.numel()] * pad)
            count = (len(indices) + pad) // block
            blocks.extend([expert] * count)
            if expert == -1:
                inactive = count
        active['alignment'] = (torch.tensor(tokens, dtype=torch.int32),
                               torch.tensor(blocks, dtype=torch.int32),
                               torch.tensor([len(tokens), inactive], dtype=torch.int32))
        active['ids'] = ids
        return active['alignment']

    def single(ids, block, *, return_inactive_prefix_count):
        return align(ids, block, active['local_experts'],
                     return_inactive_prefix_count=return_inactive_prefix_count)

    def gemm(a, out, *args, **kw):
        assert not kw['is_ep'] and kw['has_inactive_prefix_count']
        assert kw['expert_ids_are_sorted'] and kw['use_fp32_reduce']
        assert not kw['use_atomic_add'] and not kw['mul_topk_weights']
        sorted_ids, experts, counts = args[8:11]
        block = kw['moe_block_size']
        prefix = int(counts[1])
        assert int(counts[0]) == len(sorted_ids) == len(experts)*block
        assert (experts[:prefix] == -1).all() and (experts[prefix:] >= 0).all()
        flat = active['ids'].reshape(-1)
        visited = []
        for i in range(prefix, len(experts)):
            rows = sorted_ids[i*block:(i+1)*block]
            rows = rows[rows < flat.numel()].long()
            assert (flat[rows] == experts[i]).all()
            visited.extend(rows.tolist())
        expected = torch.where(flat >= 0)[0].tolist()
        assert sorted(visited) == expected
        active['gemms'] += 1
        return out.zero_()

    def map_ids(ids, mapping):
        return torch.where(ids < 0, -1, mapping[ids.clamp_min(0)])

    scope = {'torch': torch, 'silu_and_mul': lambda *args, **kw: None,
             'moe_wna16_marlin_gemm': gemm, 'moe_align_block_size': align,
             'moe_align_single_token': single, '_map_topk_ids_to_local_experts': map_ids,
             'marlin_make_workspace': lambda *args, **kw: None,
             'scalar_types': SimpleNamespace(float4_e2m1f=object()),
             '_MARLIN_BLOCK_SIZE_M_CANDIDATES': (8, 16, 32, 48, 64)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), FILE, 'exec'), scope)
    torch.manual_seed(119)
    cases = 0
    for ep in (1, 2, 4, 8):
        for rank in range(ep):
            local = 256//ep
            mapping = torch.full((256,), -1, dtype=torch.int32)
            mapping[rank*local:(rank+1)*local] = torch.arange(local)
            state = SimpleNamespace(global_num_experts=256, num_local_experts=local,
                                    intermediate_size_per_partition=16,
                                    _expert_map=mapping if ep > 1 else None,
                                    w13_weight=None, w13_weight_scale=None,
                                    w13_weight_scale_2=None, w2_weight=None,
                                    w2_weight_scale=None, w2_weight_scale_2=None)
            for rows in (1, 4, 16, 32, 128, 256, 753):
                for pattern in ('random', 'local', 'dropped', 'peer'):
                    ids = torch.randint(0, 256, (rows, 8), dtype=torch.int32)
                    if pattern == 'local':
                        ids.fill_(rank*local)
                    elif pattern == 'dropped':
                        ids.fill_(-1)
                    elif pattern == 'peer':
                        ids.fill_(((rank+1) % ep)*local)
                    active.update(local_experts=local, gemms=0)
                    result, local_ids = scope['_marlin_expert_outputs'](
                        state, torch.zeros(rows, 16), torch.ones(rows, 8), ids)
                    expected = map_ids(ids, mapping) if ep > 1 else ids
                    assert torch.equal(local_ids, expected)
                    assert result.shape == (rows, 8, 16) and active['gemms'] == 2
                    cases += 1
    report = {'scope': __doc__, 'cases': cases, 'instrumented_gemm_calls': cases*2,
              'patch_sha256': hashlib.sha256(PATCH.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'torch': torch.__version__, 'native_gpu_executed': False}
    REPORT.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
