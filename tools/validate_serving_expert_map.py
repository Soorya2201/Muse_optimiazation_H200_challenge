"""CPU fallback checks or real CUDA remap/graph checks and paired timing.

CPU does not import/execute Triton. CUDA needs the patched official checkout:
python tools/validate_serving_expert_map.py --device cuda --benchmark --output /tmp/expert_map.json
This component fixture neither loads the model nor establishes serving speed.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'python/sglang/kernels/ops/moe/map_marlin_expert_ids.py'


def stock(ids, mapping):
    cast = ids.to(torch.int32)
    return torch.where(cast < 0, -1, mapping[cast.clamp_min(0)]).contiguous()


def load_function(path, device):
    if device == 'cuda':
        spec = importlib.util.spec_from_file_location('expert_map_candidate', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.map_marlin_expert_ids
    # Execute the exact candidate wrapper; CPU takes its stock fallback branch.
    tree = ast.parse(path.read_text())
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef)
                and x.name == 'map_marlin_expert_ids')
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['map_marlin_expert_ids']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sglang', type=Path, default=ROOT/'serving/current/sglang')
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve evidence: choose a fresh output path'
    assert not args.benchmark or args.device == 'cuda'
    if args.device == 'cuda':
        assert torch.cuda.is_available(), 'Real CUDA device required'
    torch.set_num_threads(1)
    torch.manual_seed(1209)
    path = args.sglang/MODULE
    candidate = load_function(path, args.device)
    cases = 0
    for ep in (1, 2, 4, 8):
        for rank in range(ep):
            mapping = torch.full((256,), -1, dtype=torch.int32, device=args.device)
            start, count = rank * (256 // ep), 256 // ep
            mapping[start:start+count] = torch.arange(count, device=args.device)
            for rows in (0, 1, 4, 16, 32, 128, 256, 753):
                for dtype in (torch.int32, torch.int64):
                    for layout in ('contiguous', 'strided'):
                        width = 8 if layout == 'contiguous' else 16
                        ids = torch.randint(-8, 256, (rows, width), dtype=dtype,
                                            device=args.device)
                        if layout == 'strided':
                            ids = ids[:, ::2]
                        if rows:
                            ids[0, :4] = torch.tensor([-1, -7, start, start+count-1],
                                                      dtype=dtype, device=args.device)
                            if dtype == torch.int64:
                                # Stock narrows before mapping; test wrap to valid/sentinel IDs.
                                ids[0, 4:] = torch.tensor(
                                    [2**32+start, -(2**32)+start, 2**32-1, -(2**32)-2],
                                    dtype=dtype, device=args.device)
                        saved_ids, saved_mapping = ids.clone(), mapping.clone()
                        expected, actual = stock(ids, mapping), candidate(ids, mapping)
                        assert torch.equal(actual, expected), (ep, rank, rows, dtype, layout)
                        assert actual.dtype == torch.int32 and actual.is_contiguous()
                        assert actual.shape == ids.shape and actual.device == ids.device
                        assert torch.equal(ids, saved_ids) and torch.equal(mapping, saved_mapping)
                        # Same addresses, new contents: catch data-dependent cached results.
                        ids.fill_(-7)
                        assert torch.equal(candidate(ids, mapping), stock(ids, mapping))
                        cases += 1
    graphs, timings = [], []
    if args.device == 'cuda':
        for rows in (1, 4, 16, 32, 128, 256, 753, 3012):
            ids = torch.randint(0, 256, (rows, 8), device='cuda', dtype=torch.int64)
            mapping = torch.full((256,), -1, device='cuda', dtype=torch.int32)
            mapping[32:64] = torch.arange(32, device='cuda', dtype=torch.int32)
            for _ in range(3):
                candidate(ids, mapping)
                stock(ids, mapping)
            torch.cuda.synchronize()
            pair = []
            for fn in (stock, candidate):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = fn(ids, mapping)
                pair.append((graph, output))
            for _ in range(3):
                ids.copy_(torch.randint(-4, 256, ids.shape, device='cuda'))
                for graph, _ in pair:
                    graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(pair[0][1], pair[1][1])
            graphs.append({'rows': rows, 'fresh_input_replays': 3, 'exact': True})
            if args.benchmark:
                # Put repeats inside the graph so Python cannot starve the GPU
                # between tiny launches and dominate the event interval.
                timing_graphs = []
                for fn in (stock, candidate):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(100):
                            timing_output = fn(ids, mapping)
                    timing_graphs.append(graph)
                    for _ in range(5):
                        graph.replay()
                torch.cuda.synchronize()
                samples = []
                for round_id in range(20):
                    measured = {}
                    for index in ((0, 1) if round_id % 2 == 0 else (1, 0)):
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        timing_graphs[index].replay()
                        end.record()
                        end.synchronize()
                        measured['stock_us' if index == 0 else 'candidate_us'] = begin.elapsed_time(end)*10
                    samples.append(measured)
                timings.append({'rows': rows, 'paired_samples': samples,
                                'median_paired_speedup': statistics.median(
                                    x['stock_us']/x['candidate_us'] for x in samples)})
    result = {'scope': ('Real CUDA component correctness/graphs; synthetic routing, not serving.'
                        if args.device == 'cuda' else
                        'Exact CPU fallback wrapper only; Triton branch uncompiled and unexecuted.'),
              'torch': torch.__version__, 'device': args.device,
              'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'cases': cases, 'fresh_input_calls': cases, 'graphs': graphs, 'timings': timings}
    if args.device == 'cuda':
        result['gpu'] = torch.cuda.get_device_name()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
