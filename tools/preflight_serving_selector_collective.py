"""CPU transport and exact candidate_topk flow; no Triton/CUDA execution.

CUDA admission and the two transport kernels are explicit CPU doubles. Local
top-k uses Torch, not FlashInfer. Native kernels, TP capture and performance
are unproven. The released-config/stock-builder check proves source-path
reachability under those inputs, not that an official model was loaded.
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
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = '297e54fd105e974cdf401e5b4d20253a22c16f90'
FILE = 'python/sglang/srt/models/dflash.py'
PATCH = ROOT/'serving/candidates/09_selector_single_collective.diff'


def pack_cpu(values, ids, start):
    bits = values.float().contiguous().view(torch.int32).long() & 0xffffffff
    return ((ids.long()+start) << 32) | bits


def unpack_cpu(packed):
    return packed.to(torch.int32).contiguous().view(torch.float32), packed >> 32


def bit_equal(a, b):
    return a.shape == b.shape and torch.equal(a.float().contiguous().view(torch.int32),
                                             b.float().contiguous().view(torch.int32))


def prepare(sg):
    codes, manifest = {}, {}
    for name in ('stock', 'candidate'):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, GIT_INDEX_FILE=tmp+'/index')
            def git(*args):
                return subprocess.check_output(['git', '-C', str(sg), *args], env=env)
            git('read-tree', BASE)
            if name == 'candidate':
                git('apply', '--cached', str(PATCH))
            data = git('show', ':'+FILE)
            if name == 'candidate':
                helper = git('show', ':python/sglang/kernels/ops/speculative/selector_collective.py')
                assert helper == (ROOT/'serving/prototypes/selector_collective.py').read_bytes()
        node = next(n for n in ast.parse(data.decode()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'candidate_topk')
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
        codes[name] = compile(ast.fix_missing_locations(module), FILE+':'+name, 'exec')
        manifest[name] = dict(source_sha256=hashlib.sha256(data).hexdigest())
    manifest['patch_sha256'] = hashlib.sha256(PATCH.read_bytes()).hexdigest()
    manifest['helper_sha256'] = hashlib.sha256(helper).hexdigest()
    return codes, manifest


def check_reachability(sg):
    config_path = ROOT/'research/serving/DFLASH2_PUBLIC_CONFIG_2026-11-10.json'
    config = json.loads(config_path.read_text())
    assert config['architectures'] == ['DFlash2DraftModel']
    assert config['dflash_config']['selector_rank'] == 256
    assert config['dflash_config']['selector_top_k'] == 16
    assert config['dflash_config']['block_size'] == 8
    path = 'python/sglang/srt/speculative/dflash_worker_v2.py'
    data = subprocess.check_output(['git', '-C', str(sg), 'show', BASE+':'+path])
    node = next(n for n in ast.walk(ast.parse(data.decode()))
                if isinstance(n, ast.FunctionDef) and n.name == '_maybe_build_draft_sampler')
    module = ast.Module(body=[node], type_ignores=[])
    def forbidden(**kw):
        raise AssertionError('Plain greedy sampler is not the selector branch')
    scope = dict(logger=SimpleNamespace(info=lambda *a, **kw: None),
        envs=SimpleNamespace(SGLANG_DFLASH_EAGER_DRAFT_SAMPLER=SimpleNamespace(get=lambda: False)),
        unwrap_lora_layer=lambda x: x, is_dense_head_weight=lambda x: True,
        should_apply_lm_head_quant_method=lambda *a: False,
        _SelectorDraftSampler=lambda **kw: dict(kind='selector', **kw),
        _DflashDraftSampler=forbidden,
        get_exec=lambda: SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(bs=[1, 4, 16, 32])))))
    exec(compile(ast.fix_missing_locations(module), path, 'exec'), scope)
    draft = SimpleNamespace()
    obj = SimpleNamespace(block_size=8, _target_tp_rank=0, selector=object(),
        _target_worker=SimpleNamespace(model_runner=SimpleNamespace(model=SimpleNamespace(
            lm_head=SimpleNamespace(weight=torch.empty(1))))),
        draft_model=draft, device='cpu', _selector_sampling_enabled=True)
    result = scope['_maybe_build_draft_sampler'](obj)
    assert result['kind'] == 'selector' and result['max_bs'] == 32
    return dict(config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        worker_source_sha256=hashlib.sha256(data).hexdigest(),
        expected_builder_branch='selector, not _DflashDraftSampler',
        scope='Pinned configuration facts + actual builder with CPU constructor doubles; no model load')


def run(codes):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(110109)
    protocol = 0
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        raw = (torch.randint(-(1 << 31), 1 << 31, (65536,), generator=generator,
                             dtype=torch.int32).view(dtype) if dtype == torch.float32 else
               torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype))
        ids = torch.arange(raw.numel(), dtype=torch.int64) % 17
        ids[::3] = (1 << 31)-1
        packed = pack_cpu(raw.reshape(-1, 16), ids.reshape(-1, 16), 0)
        values, restored = unpack_cpu(packed)
        assert bit_equal(values, raw.reshape(-1, 16))
        assert torch.equal(restored.reshape(-1), ids)
        protocol += raw.numel()
    helper = SimpleNamespace(pack_selector_pairs=pack_cpu, unpack_selector_pairs=unpack_cpu)
    cases, fast, fallback = 0, 0, 0
    with patch.dict(sys.modules, {'sglang.kernels.ops.speculative.selector_collective': helper}):
        for world, rows, k, dtype, pattern, partition, admit in itertools.product(
            (1, 2, 3, 8), (0, 1, 7, 28, 112, 224), (1, 16),
            (torch.float16, torch.bfloat16, torch.float32, torch.float64),
            ('random', 'exceptional'), (False, True), ('cuda', 'cpu', 'large_vocab')):
            logits = [torch.randn(rows, 32, dtype=dtype, generator=generator) for _ in range(world)]
            if pattern == 'exceptional':
                for shard in logits:
                    shard.zero_()
                    shard[:, :4] = torch.tensor([float('nan'), -float('inf'), float('inf'), -0.], dtype=dtype)
            local = [torch.topk(x, k, dim=-1) for x in logits]
            offsets = [rank*32 + ((1 << 54) if admit == 'large_vocab' else 0) for rank in range(world)]
            enabled = (world > 1 and admit == 'cuda' and not partition and dtype != torch.float64)
            outputs, counts = {}, {}
            for name in ('stock', 'candidate'):
                calls = []
                def gather(x, dim=-1):
                    calls.append(x.dtype)
                    use_packed = name == 'candidate' and enabled
                    if use_packed:
                        assert x.dtype == torch.int64 and torch.equal(x, pack_cpu(*local[0], offsets[0]))
                        shards = [pack_cpu(*item, offset) for item, offset in zip(local, offsets)]
                    elif len(calls) == 1:
                        assert bit_equal(x, local[0].values.float())
                        shards = [item.values.float() for item in local]
                    elif len(calls) == 2:
                        assert torch.equal(x, local[0].indices.long()+offsets[0])
                        shards = [item.indices.long()+offset for item, offset in zip(local, offsets)]
                    else:
                        shards = [torch.logsumexp(shard.float(), -1).unsqueeze(-1) for shard in logits]
                    return torch.cat(shards, dim=dim)
                scope = dict(torch=torch, get_parallel=lambda: SimpleNamespace(tp_size=world),
                    should_apply_lm_head_quant_method=lambda *a: False, is_dense_head_weight=lambda x: True,
                    _project_candidate_logits=lambda *a, **kw: logits[0], _radix_topk=torch.topk,
                    lilicorr_topk_lse=lambda x, count: (*torch.topk(x, count, dim=-1), torch.logsumexp(x.float(), -1)),
                    tensor_model_parallel_all_gather=gather)
                exec(codes[name], scope)
                head = SimpleNamespace(weight=torch.empty(1), org_vocab_size=(world*32 if admit != 'large_vocab' else (1 << 55)),
                    shard_indices=SimpleNamespace(num_org_elements=32, org_vocab_start_index=offsets[0]))
                # Simulate only admission. CPU tensors/operators remain CPU.
                with patch.object(torch.Tensor, 'is_cuda', property(lambda self: admit != 'cpu')):
                    outputs[name] = scope['candidate_topk'](torch.empty(rows, 1), head, k, with_partition=partition)
                counts[name] = len(calls)
            a, b = outputs['stock'], outputs['candidate']
            assert torch.equal(a[0], b[0]) and bit_equal(a[1], b[1])
            assert (a[2] is None and b[2] is None) or bit_equal(a[2], b[2])
            assert counts['stock'] == (0 if world == 1 else 3 if partition else 2)
            assert counts['candidate'] == (1 if enabled else counts['stock'])
            cases += 1
            fast += int(enabled)
            fallback += int(not enabled)
    return dict(protocol_values=protocol, exact_function_cases=cases,
        one_collective_cases=fast, unchanged_fallback_cases=fallback,
        torch=torch.__version__, native_gpu_executed=False, scope=__doc__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sglang', type=Path, default=ROOT/'serving/current/sglang')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve earlier receipts'
    codes, sources = prepare(args.sglang)
    report = dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=sys.version, base=BASE, sources=sources,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reachability=check_reachability(args.sglang), checks=run(codes))
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['checks'], indent=2))


if __name__ == '__main__':
    main()
