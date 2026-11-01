"""Logical traffic and local X-ring dependencies from saved GPU route metadata.

No clocks are used. Logical TMA bytes are not HBM transactions. The dependency
graph models only X-slot retirement, not global/warpgroup concurrency or CUDA
memory ordering. It can reject a ring-capacity cycle, not validate the kernel.
"""
import ast
import bisect
import collections
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / 'muse_moe_w4a16'
source = (TRACK / 'submission.py').read_text()
native = next(ast.literal_eval(n.value) for n in ast.parse(source).body
              if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
TABLES = [[int(x.strip()) for x in re.search(r'^#define ' + name + r' ([0-9, ]+)', native, re.M)[1].split(',')]
          for name in ('OKC_WGSA', 'OKC_WGSB')]
assert TABLES == [[112, 135, 140, 158, 173, 185, 200, 207],
                  [100, 104, 115, 123, 132, 144, 156, 174]]


def jobs_for(counts, ft):
    fat = [(e, min(ft, n-q)) for e, n in enumerate(counts) if n > 8
           for q in range(0, n, ft)]
    skinny = [(e, n) for e, n in enumerate(counts) if 0 < n <= 8]
    return fat + skinny


def cuts_for(jobs, phase, new):
    scale, grid = (2, 264) if new else (1, 132)
    stride = 512 >> phase
    pre = [0]
    for e, count in jobs:
        pre.append(pre[-1] + stride * TABLES[phase][(count+7)//8-1])
    if not jobs:
        return [0]*(grid+1)
    total = pre[-1]
    assert total*grid < 2**32
    result = []
    for c in range(grid):
        x = total*c//grid
        j = bisect.bisect_right(pre, x)-1
        cost = TABLES[phase][(jobs[j][1]+7)//8-1]
        result.append(j*stride*scale + scale*((x-pre[j])//cost))
    result.append(len(jobs)*stride*scale)
    return result


def segments(g0, g1, phase):
    trailing = ((g1-1)//16)*16
    start = trailing if phase == 0 and g1 % 16 and trailing > g0 else g0
    pieces = [(start, g1)] + ([(g0, start)] if start > g0 else [])
    result = []
    for lo, hi in pieces:
        while lo < hi:
            end = min(hi, (lo//16+1)*16)
            result.append((lo//16, lo % 16, (end-1) % 16+1))
            lo = end
    assert sum(b-a for u, a, b in result) == g1-g0
    return result


def xring_acyclic(segment_lengths, slots):
    """First MMA group, segment/stage end, and X-copy issue are distinct nodes."""
    ends = []
    for length in segment_lengths:
        assert length > 0
        ends.extend([False]*(length-1) + [True])
    n = len(ends)
    graph = [[] for _ in range(3*n)]
    indegree = [0]*(3*n)

    def edge(a, b):
        graph[a].append(b)
        indegree[b] += 1

    for i, is_segment_end in enumerate(ends):
        issue, first, end = 3*i, 3*i+1, 3*i+2
        edge(issue, first)
        edge(first, end)
        if i:
            edge(3*(i-1)+2, first)
            edge(3*(i-1), issue)
        if i >= slots:
            old = i-slots
            # Intra-segment last group retires behind the next stage's first
            # group; a segment-ending wait0 retires it immediately instead.
            released_by = 3*old+2 if ends[old] else 3*(old+1)+1
            edge(released_by, issue)
    queue = collections.deque(i for i, d in enumerate(indegree) if not d)
    removed = 0
    while queue:
        i = queue.popleft()
        removed += 1
        for j in graph[i]:
            indegree[j] -= 1
            if not indegree[j]:
                queue.append(j)
    return removed == 3*n


assert not xring_acyclic([2], 1), 'One slot must expose the intra-segment cycle'
assert xring_acyclic([2], 2)
assert xring_acyclic([1, 1], 1), 'A drained segment boundary frees its slot'
phase = json.loads((TRACK / 'PHASE_TRACE_f8208a2105ab4940.json').read_text())
flush = json.loads((TRACK / 'FLUSH_TRACE_b22622ebb61946df.json').read_text())
assert phase['timing_attribution_gate'] and flush['timing_attribution_gate']
rows = []
local_graphs = 0
for t, corroboration in zip(phase['wire']['traces'], flush['wire']['traces']):
    assert all(t[k] == corroboration[k] for k in ('m', 'seed', 'pad_mib', 'jobs'))
    counts = [0]*32
    for e, count in t['jobs']:
        counts[e] += count
    assert jobs_for(counts, 64) == [tuple(x) for x in t['jobs']]
    row = dict(m=t['m'], seed=t['seed'], pad_mib=t['pad_mib'],
               local_expert_counts=counts, local_routes=sum(counts), arms={})
    for name, new in [('parent', False), ('two_cta', True)]:
        jobs = jobs_for(counts, 32 if new else 64)
        units = (64, 32) if new else (32, 16)
        grid, wgs, payload, slots = (264, 2, 18432, 2) if new else (132, 4, 36864, 3)
        padded = sum(((n+7)//8)*8 for e, n in jobs)
        # Each job streams both complete expert matrices once, independent of
        # token count; each output unit streams its own padded X slices.
        weight_bytes = len(jobs)*sum(units)*16*payload
        unique_weight_bytes = sum(n > 0 for n in counts)*27*1024*1024
        x_bytes = padded*sum(units)*16*512
        phases = []
        all_lengths = [[] for _ in range(grid)]
        for pb in (0, 1):
            cuts = cuts_for(jobs, pb, new)
            parts = [segments(lo, hi, pb) for lo, hi in zip(cuts, cuts[1:])]
            # Partitions visit every logical stage exactly once; count split
            # units from actual cuts, not a latency estimate from old clocks.
            ownership = collections.Counter(u for ss in parts for u, a, b in ss)
            assert sum(b-a for ss in parts for u, a, b in ss) == len(jobs)*units[pb]*16
            split_units = sum(n > 1 for n in ownership.values())
            for c, ss in enumerate(parts):
                all_lengths[c].extend(b-a for u, a, b in ss)
            phases.append(dict(phase=pb, total_units=len(ownership),
                               split_units=split_units, split_segments=sum(n for n in ownership.values() if n > 1),
                               consumer_warpgroup_publications_or_finishes=len(ownership)*wgs,
                               empty_cta_ranges=sum(lo == hi for lo, hi in zip(cuts, cuts[1:]))))
        for lengths in all_lengths:
            assert xring_acyclic(lengths, slots)
            local_graphs += 1
        row['arms'][name] = dict(jobs=len(jobs), padded_rows=padded,
                                 weight_tma_bytes=weight_bytes,
                                 unique_weight_payload_bytes=unique_weight_bytes,
                                 x_tma_bytes=x_bytes, phases=phases)
    old, new = row['arms']['parent'], row['arms']['two_cta']
    assert new['padded_rows'] == old['padded_rows']
    assert new['x_tma_bytes'] == 2*old['x_tma_bytes']
    assert new['unique_weight_payload_bytes'] == old['unique_weight_payload_bytes']
    row['logical_weight_read_ratio'] = new['weight_tma_bytes']/old['weight_tma_bytes']
    row['logical_x_read_ratio'] = 2.
    rows.append(row)

out = dict(scope='Offline exact logical traffic/partition arithmetic from two saved synthetic GPU routing fixtures. No new CUDA job, timings, cache counters, HBM measurements or concurrency proof.',
           source_jobs=[phase['job'], flush['job']], canonical_sha256=hashlib.sha256(source.encode()).hexdigest(),
           rows=rows, local_xring_dependency_graphs=local_graphs,
           one_slot_intra_segment_cycle_reproduced=True,
           limitations=['M128 and seed7907 routing metadata are not present in these saved traces.',
                        'Logical repeated loads may hit L2 or share in-flight requests; do not convert these bytes into an HBM latency floor.',
                        'Local graph excludes WGMMA register lifetimes, weight ring, split-finisher waits, global readiness and memory ordering.',
                        'Acyclic local dependencies do not prove actual kernel safety or speed. Original H200 gates remain mandatory.'])
path = TRACK / 'TWO_CTA_TRAFFIC_2026-11-01.json'
assert not path.exists()
path.write_text(json.dumps(out, indent=2)+'\n')
for r in rows:
    print('FIXTURE', r['m'], 'jobs', *(r['arms'][a]['jobs'] for a in ('parent', 'two_cta')),
          'logical_weight_ratio', r['logical_weight_read_ratio'], 'logical_X_ratio', r['logical_x_read_ratio'])
print('LOCAL_X_GRAPHS', local_graphs, 'one_slot_counterexample', True)
print(path.relative_to(ROOT))
