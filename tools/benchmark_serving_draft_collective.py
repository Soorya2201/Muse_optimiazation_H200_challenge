"""Eight-GPU component validation/AB timing; not an official serving score.

Run in the official CUDA image on an existing development node:
torchrun --standalone --nproc-per-node=8 tools/benchmark_serving_draft_collective.py --output /tmp/draft_collective.json
The adapter uses PyTorch NCCL; the full SGLang collective/serving path still
needs official validation. No model weights are required for these fixtures.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
BASE = '297e54fd105e974cdf401e5b4d20253a22c16f90'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sglang', type=Path, default=ROOT/'serving/current/sglang')
    p.add_argument('--batches', type=int, nargs='+', default=[1, 4, 16, 17, 19, 32])
    p.add_argument('--hidden', type=int, default=1024)
    p.add_argument('--local-vocab', type=int, default=4096)
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert torch.cuda.is_available(), 'An existing CUDA development node is required'
    assert all(0 < bs <= 32 for bs in args.batches) and args.rounds > 1
    import os
    device = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device)
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world > 1
    assert not args.output.exists()
    module_path = args.sglang/'python/sglang/kernels/ops/speculative/draft_head_collective.py'
    spec = importlib.util.spec_from_file_location('muse_draft_head_fixture', module_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    class Group:
        world_size = world

        def all_gather_into_tensor(self, output, input_):
            dist.all_gather_into_tensor(output.view(-1), input_.view(-1))

    group = Group()
    # Real GPU exceptional-value transport, int64 ID range, ragged tail masks.
    for n in (1, 7, 28, 127, 128, 129, 224, 257):
        value = torch.arange(n, device='cuda', dtype=torch.float32)+rank
        exceptional = [-0. if rank == 0 else 0., -float('inf'), float('inf'), float('nan')]
        value[:min(4, n)] = torch.tensor(exceptional[:min(4, n)], device='cuda')
        ids = torch.arange(n, device='cuda', dtype=torch.int64)+(1 << 54)-rank*1000
        pairs = torch.empty((n, 2), device='cuda', dtype=torch.int64)
        gathered = torch.empty((world*n, 2), device='cuda', dtype=torch.int64)
        output = torch.empty(n, device='cuda', dtype=torch.int64)
        helper.pack_greedy_pairs(value, ids, pairs, 0)
        group.all_gather_into_tensor(gathered, pairs)
        helper.select_greedy_pairs(gathered, output, world)
        values = torch.empty(world*n, device='cuda', dtype=torch.float32)
        tokens = torch.empty(world*n, device='cuda', dtype=torch.int64)
        group.all_gather_into_tensor(values, value)
        group.all_gather_into_tensor(tokens, ids)
        expected = tokens.view(world, n).gather(0, values.view(world, n).argmax(0)[None, :])[0]
        assert torch.equal(output, expected), ('transport fixture', rank, n)

    path = 'python/sglang/srt/speculative/dflash_worker_v2.py'
    old = subprocess.check_output(['git', '-C', str(args.sglang), 'show', f'{BASE}:{path}']).decode()
    new = (args.sglang/path).read_text()

    def sampler(source):
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == '_DflashDraftSampler')
        env = dict(torch=torch, is_cuda=lambda: True,
                   pack_greedy_pairs=helper.pack_greedy_pairs, select_greedy_pairs=helper.select_greedy_pairs)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<exact sampler class>', 'exec'), env)
        return env['_DflashDraftSampler']

    gen = torch.Generator(device='cuda').manual_seed(109+rank)
    weight = torch.randn((args.local_vocab, args.hidden), generator=gen, device='cuda', dtype=torch.bfloat16)
    kw = dict(weight=weight, block_size=8, num_org=args.local_vocab,
              org_vocab_start=rank*args.local_vocab, max_bs=32, tp_group=group)
    arms = {'stock':sampler(old)(**kw), 'candidate':sampler(new)(**kw)}
    assert arms['candidate']._single_gather
    records = []
    for bs in args.batches:
        input_gen = torch.Generator(device='cuda').manual_seed(109+bs)
        hidden = torch.randn((bs*8, args.hidden), device='cuda', dtype=torch.bfloat16, generator=input_gen)
        graphs = {}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for fn in arms.values():
                for _ in range(3):
                    fn(hidden)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        for name, fn in arms.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn(hidden)
            graphs[name] = graph
        # Same graph addresses replay on new inputs, not cached answers.
        for _ in range(2):
            hidden.normal_(generator=input_gen)
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(arms['stock'].out[:bs*7], arms['candidate'].out[:bs*7])
        raw = {name:[] for name in arms}
        for rep in range(args.rounds):
            for name in (('stock', 'candidate') if rep % 2 == 0 else ('candidate', 'stock')):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                graphs[name].replay()
                b.record()
                b.synchronize()
                raw[name].append(a.elapsed_time(b)*1000)
        node_raw = [None]*world
        dist.all_gather_object(node_raw, raw)
        maxima = {name:[max(r[name][i] for r in node_raw) for i in range(args.rounds)] for name in arms}
        records.append(dict(batch=bs, per_rank_event_us=node_raw, slowest_rank_event_us=maxima,
                            paired_median_ratio=statistics.median(a/b for a, b in zip(maxima['stock'], maxima['candidate']))))
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(scope='Synthetic GPU component test with PyTorch NCCL, warm graph replay, balanced orders, fresh inputs. Not SGLang GroupCoordinator or full-model serving validation; not an official score.',
                                               base=BASE, world_size=world, torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                                               hidden=args.hidden, local_vocab=args.local_vocab, rows=records), indent=2)+'\n')
        print(args.output)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
