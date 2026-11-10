"""CPU policy/protocol tests; no claim of GPU kernel or serving validation."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT / 'serving/current/sglang'
BASE = '297e54fd105e974cdf401e5b4d20253a22c16f90'
torch.set_num_threads(1)


def base_source(path):
    return subprocess.check_output(['git', '-C', str(SG), 'show', f'{BASE}:{path}']).decode()


moe_path = 'python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py'
moe_old, moe_new = base_source(moe_path), (SG / moe_path).read_text()


def heuristic(source):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'MuseSpark1xFp4Experts')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_marlin_expert_outputs')
    loop = next(n for n in method.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'block_size_m')
    assignments = [n for n in method.body[:method.body.index(loop)] if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'routes_per_expert' for t in n.targets)]
    return compile(ast.fix_missing_locations(ast.Module(body=assignments+[loop], type_ignores=[])), '<exact heuristic>', 'exec')


old_loop, new_loop = heuristic(moe_old), heuristic(moe_new)
policy_cases = 0
policy_table = []
for ep in (1, 2, 4, 8):
    for m in range(16385):
        env = dict(self=SimpleNamespace(num_local_experts=256//ep, global_num_experts=256),
                   num_tokens=m, topk=8, _MARLIN_BLOCK_SIZE_M_CANDIDATES=(8, 16, 32, 48, 64))
        exec(old_loop, env)
        before = env['block_size_m']
        exec(new_loop, env)
        after = env['block_size_m']
        expected = next((b for b in (8, 16, 32, 48, 64) if 10*m*8 < 9*256*b), 64)
        assert after == expected, (ep, m, after, expected)
        if ep == 1:
            assert before == after
        if ep == 8 and m in (1, 4, 8, 16, 32, 64, 128, 192, 256, 384, 768, 2048, 16384):
            policy_table.append(dict(tokens=m, stock_block=before, global_density_block=after))
        policy_cases += 1


def pack_cpu(values, indices, pairs, start):
    pairs[:, 0].copy_(values.float().contiguous().view(torch.int32).to(torch.int64))
    pairs[:, 1].copy_(indices+start)


def select_cpu(gathered, output, tp):
    n = output.numel()
    values = gathered[:, 0].to(torch.int32).contiguous().view(torch.float32).reshape(tp, n)
    ids = gathered[:, 1].reshape(tp, n)
    nan = values.isnan()
    maxima = values.masked_fill(nan, -float('inf')).max(0).values
    matches = torch.where(nan.any(0)[None, :], nan, values == maxima[None, :])
    rank = torch.where(matches, torch.arange(tp)[:, None], tp).min(0).values
    output.copy_(ids[rank, torch.arange(n)])


# Full 16-bit floating encodings: finite values, signed zero, infinities,
# signaling/quiet NaNs. IDs intentionally exceed floating-point exact ranges.
transport_cases = 0
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    if dtype == torch.float32:
        values = torch.randint(-(1 << 31), 1 << 31, (65536,), dtype=torch.int32,
                               generator=torch.Generator().manual_seed(123)).view(dtype)
    else:
        values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    ids = torch.arange(65536, dtype=torch.int64)+(1 << 54)
    pairs = torch.empty((65536, 2), dtype=torch.int64)
    pack_cpu(values, ids, pairs, 97)
    assert torch.equal(pairs[:, 1], ids+97)
    restored = pairs[:, 0].to(torch.int32).view(torch.float32)
    torch.testing.assert_close(restored, values.float(), rtol=0, atol=0, equal_nan=True)
    for tp in (2, 3, 4, 8):
        matrix = torch.stack([values.roll(137*r).float() for r in range(tp)])
        matrix[:, :5] = torch.tensor([0., -0., -float('inf'), float('inf'), float('nan')])
        tokens = torch.stack([ids-r*100000 for r in range(tp)])  # rank ties differ from min-ID ties
        wire = torch.empty((tp*65536, 2), dtype=torch.int64)
        for r in range(tp):
            pack_cpu(matrix[r], tokens[r], wire[r*65536:(r+1)*65536], 0)
        out = torch.empty(65536, dtype=torch.int64)
        select_cpu(wire, out, tp)
        expected = tokens[torch.argmax(matrix, dim=0), torch.arange(65536)]
        assert torch.equal(out, expected)
        transport_cases += 65536

worker_path = 'python/sglang/srt/speculative/dflash_worker_v2.py'


class CPUProxy:
    def __getattr__(self, name):
        return getattr(torch, name)

    def empty(self, *args, **kwargs):
        kwargs['device'] = 'cpu'
        return torch.empty(*args, **kwargs)


class DeviceWeight:
    # CPU emulation of device placement only; arithmetic is real CPU torch.
    is_cuda = True
    device = torch.device('cuda')

    def __init__(self, tensor):
        self.tensor, self.dtype = tensor, tensor.dtype

    def __getitem__(self, key):
        return self.tensor[key]


def load_sampler(source):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == '_DflashDraftSampler')
    env = dict(torch=CPUProxy(), is_cuda=lambda: True, pack_greedy_pairs=pack_cpu, select_greedy_pairs=select_cpu)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<exact sampler class>', 'exec'), env)
    return env['_DflashDraftSampler']


old_sampler, new_sampler = load_sampler(base_source(worker_path)), load_sampler((SG / worker_path).read_text())


class Group:
    def __init__(self, tp, rank):
        self.world_size, self.rank, self.calls = tp, rank, 0

    def reset(self, maxima, ids):
        self.maxima, self.ids, self.calls = maxima, ids, 0

    def all_gather_into_tensor(self, output, input_):
        self.calls += 1
        if input_.ndim == 2:
            wire = torch.empty_like(output)
            n = input_.shape[0]
            for r in range(self.world_size):
                pack_cpu(self.maxima[r], self.ids[r], wire[r*n:(r+1)*n], 0)
            torch.testing.assert_close(input_, wire[self.rank*n:(self.rank+1)*n], rtol=0, atol=0)
            output.copy_(wire)
        elif input_.dtype == torch.int64:
            assert torch.equal(input_, self.ids[self.rank])
            output.copy_(self.ids.reshape(-1))
        else:
            torch.testing.assert_close(input_, self.maxima[self.rank], rtol=0, atol=0)
            output.copy_(self.maxima.reshape(-1))


g = torch.Generator().manual_seed(109)
integration_cases = 0
matching_empty_rejections = 0
for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
    for tp in (1, 2, 3, 4, 8):
        weights = [torch.randn((31, 16), generator=g).to(dtype) for _ in range(tp)]
        starts = [(tp-r)*1000 for r in range(tp)]
        for rank in range(tp):
            group = Group(tp, rank)
            kwargs = dict(weight=DeviceWeight(weights[rank]), block_size=8, num_org=31,
                          org_vocab_start=starts[rank], max_bs=32, tp_group=group)
            stock, candidate = old_sampler(**kwargs), new_sampler(**kwargs)
            out_pointer = candidate.out.data_ptr()
            if tp > 1:
                assert candidate._single_gather == (dtype != torch.float64)
            for bs in (0, 1, 4, 16, 32, 1):
                hidden = torch.randn((bs*8, 16), generator=g).to(dtype)
                if bs == 0:
                    errors = []
                    for sampler in (stock, candidate):
                        try:
                            sampler(hidden)
                        except RuntimeError as exc:
                            errors.append(str(exc))
                    assert len(errors) == 2 and errors[0] == errors[1]
                    matching_empty_rejections += 1
                    continue
                hs = hidden.reshape(bs, 8, 16)[:, 1:, :].reshape(-1, 16)
                local = [(hs @ w.T).max(1) for w in weights]
                maxima = torch.stack([a.values for a in local])
                ids = torch.stack([a.indices+start for a, start in zip(local, starts)])
                group.reset(maxima, ids)
                stock(hidden)
                expected = stock.out[:bs*7].clone()
                stock_calls = group.calls
                group.reset(maxima, ids)
                candidate(hidden)
                assert torch.equal(candidate.out[:bs*7], expected), (dtype, tp, rank, bs)
                assert candidate.out.data_ptr() == out_pointer
                expected_calls = 0 if tp == 1 else 2 if dtype == torch.float64 else 1
                assert group.calls == expected_calls and stock_calls == (0 if tp == 1 else 2)
                integration_cases += 1

out = dict(scope='CPU exact Python policy/class flow plus independent lossless-transport/selection models. Triton kernels, NCCL, CUDA graphs, full-model output quality and H200 speed are NOT validated.',
           torch_version=torch.__version__, base=BASE, policy_cases=policy_cases, policy_table=policy_table,
           transport_and_selection_columns=transport_cases, exhaustive_16bit_value_encodings=131072,
           randomized_fp32_encodings=65536, cpu_sampler_flow_cases=integration_cases,
           stock_and_candidate_matching_empty_batch_rejections=matching_empty_rejections,
           fresh_inputs_and_output_pointer_reuse=True,
           patches={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'serving/candidates').glob('0[12]*.diff'))})
target = ROOT/'serving/CPU_PREFLIGHT_2026-11-09.json'
assert not target.exists()
target.write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
