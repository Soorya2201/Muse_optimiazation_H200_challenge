"""Optional native coordinator experiment for benchmark_serving_moe_candidates.

Uses synthetic weights on one Hopper, the exact stock/candidate coordinator,
native router/projection/norm and stock Marlin methods. No TP collective, model
checkpoint, attention or serving score. This file is GPU-unrun until a receipt
explicitly records otherwise. Source preparation alone does not validate it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import statistics
from types import MethodType, SimpleNamespace

from preflight_serving_norm_overlap import MODEL, NORMS, prepare_sources


def run(args, layer, post_norm, report, save):
    import torch
    import sglang.srt.models.muse_spark as model
    from sglang.srt.layers.muse_spark.muse_spark_layers import MuseSpark1xRMSNorm

    codes, manifest = prepare_sources(args.sglang)
    assert Path(model.__file__).resolve() == (args.sglang/MODEL).resolve()
    assert hashlib.sha256((args.sglang/MODEL).read_bytes()).hexdigest() == manifest['stock']['source_sha256']
    assert hashlib.sha256((args.sglang/NORMS).read_bytes()).hexdigest() == manifest['norm_source_sha256']
    assert report['norm_overlap_sources'] == manifest, 'Sources changed since preparation'
    report.update(phase='norm_overlap_setup', coordinator_numerical=[],
        coordinator_graph_checks=[], coordinator_timings=[],
        coordinator_scope=__doc__)
    save()
    dtype = torch.bfloat16
    pre_norm = MuseSpark1xRMSNorm(args.hidden, eps=1e-6,
        zero_centered_gamma=True).to(device='cuda', dtype=dtype)
    pre_norm.weight.data.copy_(torch.linspace(-.3, .7, args.hidden, device='cuda', dtype=dtype))
    pre_norm.precompute_effective_weight()
    gate = model.MuseSpark1xRouter(args.hidden, args.experts, dtype=torch.float32).cuda()
    gate.weight.data.normal_(0, args.hidden**-.5)
    projection = torch.nn.Linear(args.hidden, args.hidden, bias=False,
                                 device='cuda', dtype=dtype)
    projection.weight.data.normal_(0, args.hidden**-.5)
    local = args.experts//args.ep
    start = args.rank*local
    class Experts:
        use_marlin = True
        post_expert_norm = post_norm
        def __call__(self, x, scores, ids, norm, *, pre_expert_norm=None,
                     prepared_input=None, packed_topk_ids=None, ar_workspace=None):
            assert prepared_input is None and ar_workspace is None
            return layer._forward_marlin(x, scores, ids, norm, pre_expert_norm=pre_expert_norm)
    coordinators = {}
    for name in ('stock', 'overlap'):
        # Pin the execution context to the non-SP fixture. TP8 controls the
        # projection path, but there is no initialized process group or allreduce.
        scope = dict(vars(model),
            get_forward=lambda: SimpleNamespace(sp_active=False),
            get_parallel=lambda: SimpleNamespace(tp_group=SimpleNamespace(ca_comm=None)),
            get_global_experts_capturer=lambda: None)
        exec(codes[name], scope)
        obj = SimpleNamespace(router_in_fp32=True, gate=gate, experts=Experts(),
            nvfp4_checkpoint=True, expert_hidden_size=args.hidden,
            pre_expert_proj=projection, pre_expert_norm=pre_norm, top_k=args.topk,
            layer_idx=0, tp_size=8, route_scale=1., router_multiplier=None,
            route_eps=1e-15, side_stream=torch.cuda.Stream())
        obj._side_stream = lambda obj=obj: obj.side_stream
        obj._routing_op = MethodType(model.MuseSpark1xMoE._routing_op, obj)
        obj.forward = MethodType(scope['_forward_experts'], obj)
        coordinators[name] = obj

    def same(actual, expected):
        torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
        return dict(exact=bool(torch.equal(actual, expected)),
                    max_abs=float((actual.float()-expected.float()).abs().max()))

    def set_bias(pattern):
        gate.e_score_correction_bias.zero_()
        if pattern != 'random':
            peer = start if pattern == 'local' else ((args.rank+1) % args.ep)*local
            gate.e_score_correction_bias[peer:peer+local] = 100

    # A source experiment, not a changed checkpoint: both coordinators share
    # the same synthetic projections, norms, experts and routing parameters.
    with torch.inference_mode():
        for has_projection in (False, True):
            for obj in coordinators.values():
                obj.pre_expert_proj = projection if has_projection else None
            for rows in args.rows:
                cell = dict(rows=rows, projection=has_projection)
                x = torch.randn(rows, args.hidden, device='cuda', dtype=dtype)/20
                for pattern in ('random', 'local', 'peer'):
                    if pattern == 'peer' and args.ep == 1:
                        continue
                    set_bias(pattern)
                    report.update(phase='coordinator_numerical', inflight_cell=dict(cell, pattern=pattern))
                    save()
                    original = x.clone()
                    # Hooks observe real norm input/output and gate input. They
                    # exist only in eager correctness checks, never in timing.
                    observations = {}
                    for name, obj in coordinators.items():
                        norms, routers, routes = [], [], []
                        def observe_norm(m, a, o):
                            assert a[0].data_ptr() != o.data_ptr(), 'Norm aliased its input'
                            norms.append((a[0].clone(), o.clone()))
                        norm_hook = pre_norm.register_forward_hook(observe_norm)
                        gate_hook = gate.register_forward_pre_hook(lambda m, a:
                            routers.append(a[0].clone()))
                        true_routing = obj._routing_op
                        def traced(*a, **kw):
                            output = true_routing()(*a, **kw)
                            routes.append(tuple(t.clone() for t in output[:2]))
                            return output
                        obj._routing_op = lambda: traced
                        try:
                            out = obj.forward(x)
                            torch.cuda.synchronize()
                        finally:
                            obj._routing_op = true_routing
                            norm_hook.remove(); gate_hook.remove()
                        assert len(norms) == 1 and len(routes) == 1
                        # Above the split-router threshold forward_split2
                        # bypasses Module.__call__, so the gate prehook is absent.
                        expected_hooks = 0 if model._use_split_bf16_router(rows) else 1
                        assert len(routers) == expected_hooks
                        assert obj.pre_expert_norm is pre_norm
                        assert torch.equal(x, original)
                        observations[name] = (out, norms[0], routers, routes[0])
                    a, b = observations['stock'], observations['overlap']
                    for u, v in zip(a[1], b[1]):
                        assert torch.equal(u, v), 'Norm input/output changed'
                    for u, v in zip(a[2], b[2]):
                        assert torch.equal(u, v), 'Router input changed'
                    for u, v in zip(a[3], b[3]):
                        assert torch.equal(u, v), 'Routes/scores changed'
                    report['coordinator_numerical'].append(dict(cell, pattern=pattern,
                        passed=True, output=same(b[0], a[0])))
                    save()

                # Fresh-input replay also changes route-bias values in place,
                # retaining their addresses, and exercises each existing bucket.
                set_bias('random')
                graphs = {}
                for name, obj in coordinators.items():
                    warmup = torch.cuda.Stream()
                    warmup.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(warmup):
                        for _ in range(5):
                            obj.forward(x)
                    torch.cuda.current_stream().wait_stream(warmup)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = obj.forward(x)
                    graphs[name] = (graph, output)
                for fresh_pattern in ('random', 'local', 'peer'):
                    if fresh_pattern == 'peer' and args.ep == 1:
                        continue
                    report.update(phase='coordinator_graph_replay',
                                  inflight_cell=dict(cell, pattern=fresh_pattern))
                    save()
                    x.copy_(torch.randn_like(x)/20)
                    set_bias(fresh_pattern)
                    original = x.clone()
                    for graph, _ in graphs.values():
                        graph.replay()
                    expected = coordinators['stock'].forward(x)
                    torch.cuda.synchronize()
                    assert torch.equal(x, original)
                    checks = {name: same(output, expected) for name, (_, output) in graphs.items()}
                    report['coordinator_graph_checks'].append(dict(cell, pattern=fresh_pattern,
                        passed=True, outputs=checks))
                    save()
                if args.norm_overlap_traces:
                    trace_path = args.output.with_name(args.output.stem+
                        f'.rows{rows}.projection{int(has_projection)}.trace.json')
                    assert not trace_path.exists(), 'Preserve earlier traces'
                    set_bias('random')
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                        for name, obj in coordinators.items():
                            with torch.profiler.record_function('coordinator_'+name):
                                obj.forward(x)
                            torch.cuda.synchronize()
                    profile.export_chrome_trace(str(trace_path))
                    report.setdefault('coordinator_traces', []).append(dict(cell,
                        path=str(trace_path), sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                        scope='Instrumented eager timeline; profiler overhead excludes this from paired timings'))
                    save()
                if not args.benchmark:
                    continue
                report.update(phase='coordinator_timing', inflight_cell=cell)
                save()
                set_bias('random')
                timing = {}
                for name, obj in coordinators.items():
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(args.repeats):
                            timing_output = obj.forward(x)
                    timing[name] = (graph, timing_output)
                    for _ in range(5):
                        graph.replay()
                torch.cuda.synchronize()
                for _, output in timing.values():
                    same(output, coordinators['stock'].forward(x))
                samples = []
                for round_id in range(args.rounds):
                    sample = {}
                    order = ('stock', 'overlap') if round_id % 2 == 0 else ('overlap', 'stock')
                    for name in order:
                        begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        begin.record(); timing[name][0].replay(); end.record(); end.synchronize()
                        sample[name] = begin.elapsed_time(end)*1000/args.repeats
                    samples.append(sample)
                report['coordinator_timings'].append(dict(cell, samples_us=samples,
                    median_paired_speedup=statistics.median(s['stock']/s['overlap'] for s in samples),
                    scope='One-GPU synthetic whole coordinator with stock local Marlin; no full-model or serving claim'))
                save()
