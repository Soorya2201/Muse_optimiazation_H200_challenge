"""Eight-Hopper selector candidate_topk validation/timing; GPU-unrun locally.

CPU source preparation:
  python tools/benchmark_serving_selector_collective.py --plan-only --output /tmp/selector-plan.json
Existing official-image Hopper node, pinned SGLang installed:
  torchrun --standalone --nproc-per-node=8 tools/benchmark_serving_selector_collective.py --output /tmp/selector.json
Uses real dense head/radix top-k/Triton and a PyTorch NCCL adapter. Synthetic
weights, no full selector lattice, SGLang GroupCoordinator or serving score.
Each GPU writes a rank-suffixed receipt; no paid hardware is provisioned.
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
import sys
import traceback
from types import SimpleNamespace

from preflight_serving_selector_collective import BASE, FILE, ROOT, prepare


def run_cuda(args, codes, report, save):
    import torch
    import torch.distributed as dist
    assert torch.cuda.is_available(), 'A real CUDA node is required'
    device = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device)
    assert torch.cuda.get_device_capability()[0] == 9, 'Hopper fixture'
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8, 'This component experiment requires eight GPUs'
    report.update(rank=rank, world_size=world, gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  native_gpu_executed=True, phase='setup')
    save()
    sys.path.insert(0, str(args.sglang/'python'))
    import sglang.srt.models.dflash as dflash
    assert Path(dflash.__file__).resolve() == (args.sglang/FILE).resolve()
    assert hashlib.sha256((args.sglang/FILE).read_bytes()).hexdigest() == report['sources']['stock']['source_sha256'], (
        'Install official base dflash.py; candidate functions are compiled separately')
    name = 'sglang.kernels.ops.speculative.selector_collective'
    path = ROOT/'serving/prototypes/selector_collective.py'
    spec = importlib.util.spec_from_file_location(name, path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[name] = helper
    spec.loader.exec_module(helper)

    def same(actual, expected):
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected) if actual.dtype == torch.int64 else torch.equal(
            actual.float().contiguous().view(torch.int32), expected.float().contiguous().view(torch.int32))

    # Native pack/unpack must preserve exceptional-value bits and boundary IDs.
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        report.update(phase='transport_bits', inflight_dtype=str(dtype)); save()
        raw = (torch.randint(-(1 << 31), 1 << 31, (65536,), device='cuda',
                dtype=torch.int32).view(dtype) if dtype == torch.float32 else
               torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16).view(dtype))
        values = raw.reshape(-1, 16)
        ids = (torch.arange(raw.numel(), device='cuda', dtype=torch.int64) % 1024).reshape_as(values)
        start = rank*1024
        ids[0, 0] = (1 << 31)-1-start
        restored, restored_ids = helper.unpack_selector_pairs(helper.pack_selector_pairs(values, ids, start))
        same(restored, values.float()); same(restored_ids, ids+start)
    report['transport_bit_values_per_rank'] = 196608
    save()

    def gather(x, dim=-1):
        assert x.ndim == 2 and dim == -1
        # Match GroupCoordinator's rank-major -> last-dimension concatenation.
        out = torch.empty((world, *x.shape), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out.view(-1), x.contiguous().view(-1))
        return out.movedim(0, 1).reshape(x.shape[0], world*x.shape[1])

    functions = {}
    for name, code in codes.items():
        scope = dict(vars(dflash), get_parallel=lambda: SimpleNamespace(tp_size=world),
                     tensor_model_parallel_all_gather=gather)
        exec(code, scope)
        functions[name] = scope['candidate_topk']
    torch.manual_seed(110109+rank)
    head = SimpleNamespace(weight=torch.nn.Parameter(torch.randn(args.local_vocab, args.hidden,
        device='cuda', dtype=torch.bfloat16)/args.hidden**.5, requires_grad=False),
        org_vocab_size=world*args.local_vocab, shard_indices=SimpleNamespace(
            num_org_elements=args.local_vocab, org_vocab_start_index=rank*args.local_vocab))
    report.update(numerical=[], graph_checks=[], timings=[])
    save()
    with torch.inference_mode():
        for rows in args.rows:
            report.update(phase='head_eager', inflight_rows=rows); save()
            x = torch.randn(rows, args.hidden, device='cuda', dtype=torch.bfloat16)
            dist.broadcast(x, src=0)
            stock = functions['stock'](x, head, args.k)
            candidate = functions['candidate'](x, head, args.k)
            same(candidate[0], stock[0]); same(candidate[1], stock[1])
            assert stock[2] is candidate[2] is None
            report['numerical'].append(dict(rows=rows, exact=True)); save()
            graphs = {}
            for name, fn in functions.items():
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5):
                        fn(x, head, args.k)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize(); dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = fn(x, head, args.k)
                graphs[name] = (graph, output)
            for fresh in range(3):
                report.update(phase='fresh_graph', inflight_replay=fresh); save()
                x.copy_(torch.randn_like(x)); dist.broadcast(x, src=0)
                original = x.clone()
                for graph, _ in graphs.values():
                    graph.replay()
                expected = functions['stock'](x, head, args.k)
                for _, output in graphs.values():
                    same(output[0], expected[0]); same(output[1], expected[1])
                same(x, original)
            report['graph_checks'].append(dict(rows=rows, fresh_replays=3, exact=True)); save()
            timing = {}
            for name, fn in functions.items():
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with torch.cuda.graph(graph):
                    for _ in range(args.repeats):
                        output = fn(x, head, args.k)
                timing[name] = (graph, output)
                for _ in range(5):
                    graph.replay()
            torch.cuda.synchronize()
            for _, output in timing.values():
                same(output[0], expected[0]); same(output[1], expected[1])
            samples = []
            report['phase'] = 'paired_timing'; save()
            for round_id in range(args.rounds):
                sample = {}
                order = ('stock', 'candidate') if round_id % 2 == 0 else ('candidate', 'stock')
                for name in order:
                    dist.barrier()
                    begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    begin.record(); timing[name][0].replay(); end.record(); end.synchronize()
                    sample[name] = begin.elapsed_time(end)*1000/args.repeats
                # Reduce after both measurements; never time this reduction.
                maxima = torch.tensor([sample['stock'], sample['candidate']], device='cuda', dtype=torch.float64)
                dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
                samples.append(dict(zip(('stock', 'candidate'), maxima.tolist())))
            report['timings'].append(dict(rows=rows, worst_rank_samples_us=samples,
                median_paired_speedup=statistics.median(s['stock']/s['candidate'] for s in samples),
                scope='Synthetic dense head + local/global top-k; no full selector lattice or serving result'))
            save()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sglang', type=Path, default=ROOT/'serving/current/sglang')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--rows', type=int, nargs='+', default=[7, 28, 112, 224])
    parser.add_argument('--hidden', type=int, default=8192)
    parser.add_argument('--local-vocab', type=int, default=25256)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    assert min(args.rows) > 0 and args.hidden > 0 and args.local_vocab >= args.k > 0
    assert args.rounds > 1 and args.repeats > 0
    if not args.plan_only:
        rank = int(os.environ['RANK'])
        args.output = args.output.with_name(args.output.stem+f'.rank{rank}'+args.output.suffix)
    assert not args.output.exists(), 'Preserve earlier receipts'
    report = dict(scope=__doc__, requested_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        base=BASE, state='preparing', native_gpu_executed=False)
    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        codes, report['sources'] = prepare(args.sglang)
        helper = ROOT/'serving/prototypes/selector_collective.py'
        compile(ast.parse(helper.read_text()), str(helper), 'exec')
        report['tools'] = {name: hashlib.sha256((ROOT/'tools'/name).read_bytes()).hexdigest()
            for name in ('benchmark_serving_selector_collective.py', 'preflight_serving_selector_collective.py')}
        if args.plan_only:
            report['state'] = 'source_preparation_passed_only'
        else:
            run_cuda(args, codes, report, save)
            report['state'] = 'gpu_component_passed'
    except BaseException:
        report.update(state='failed', error=traceback.format_exc())
        raise
    finally:
        save()
    print(json.dumps({name: report[name] for name in ('state', 'native_gpu_executed')}))


if __name__ == '__main__':
    main()
