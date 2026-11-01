"""CPU synthetic reports only: check that incomplete evidence cannot qualify.

Never creates a real run receipt. The exact candidate wire function and a copy
of the extractor execute in an automatically removed temporary directory.
"""
import ast
import base64
import copy
import itertools
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'muse_moe_w4a16/submission_v41_two_cta_probe.py'
node = next(n for n in ast.parse(SOURCE.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == '_tc_wire')
env = dict(_TCCopy=copy, _TCJSON=json, _TCStruct=struct,
           _TC64=base64, _TCZlib=zlib)
exec(compile(ast.Module(body=[node], type_ignores=[]), '<exact own wire>', 'exec'), env)


def errors():
    return {name: [0., 0.] for name in ('old', 'new', 'a1')}


def bits(v):
    return struct.unpack('<I', struct.pack('<f', v))[0]


report = dict(status='complete', stage='complete', seconds=1., rows=[],
              nearby=[dict(m=m, mode=mode, error=errors(), fresh=errors(), change=.5)
                      for m, mode in [(17, 'basis'), (33, 'mixed'), (65, 'hot'),
                                      (129, 'mixed'), (257, 'mixed'), (384, 'hot')]],
              empty=[dict(m=m, zero=True, fresh_nonzero=True) for m in (32, 128, 256)],
              fp64={name: [0., 0.] for name in ('a0', 'old', 'new', 'a1')},
              resources={'old': [0, 0, 96, 0, 0, 576, 231168, 0, 1]*2,
                         'new': [0, 0, 96, 0, 0, 320, 110976, 0, 2]*2})
for m, seed, pad in itertools.product((32, 128, 256), (3101, 7907), (0, 1, 3, 7)):
    report['rows'].append(dict(m=m, seed=seed, pad_mib=pad, error=errors(),
                              fresh=errors(), change=.5, scratch_mod32m=0,
                              event_ms_bits={name: [bits(.8 if name == 'new' else 1.)]*8
                                             for name in ('a0', 'old', 'new', 'a1')}))


def variant():
    return copy.deepcopy(report)


cases = [('complete_synthetic_positive_control', variant(), 'qualifies')]
for label, mutate, outcome in [
    ('missing_resources', lambda d: d.update(resources={}), 'malformed'),
    ('missing_fp64', lambda d: d.update(fp64={}), 'malformed'),
    ('duplicate_nearby', lambda d: d['nearby'].__setitem__(1, copy.deepcopy(d['nearby'][0])), 'malformed'),
    ('duplicate_empty', lambda d: d['empty'].__setitem__(1, copy.deepcopy(d['empty'][0])), 'malformed'),
    ('missing_arm', lambda d: d['nearby'][0]['fresh'].pop('new'), 'malformed'),
    ('short_resources', lambda d: d['resources'].__setitem__('new', [0]*9), 'malformed'),
    ('empty_nonzero', lambda d: d['empty'][0].update(zero=False), 'reject'),
    ('empty_stale_replay', lambda d: d['empty'][0].update(fresh_nonzero=False), 'reject'),
    ('insufficient_residency', lambda d: d['resources']['new'].__setitem__(17, 1), 'reject'),
    ('spill', lambda d: d['resources']['new'].__setitem__(3, 16), 'reject'),
    ('occupancy_api_error', lambda d: d['resources']['new'].__setitem__(7, 1), 'reject'),
    ('numeric_error', lambda d: d['rows'][0]['fresh']['new'].__setitem__(0, .003), 'reject'),
    ('nonfinite_error', lambda d: d['nearby'][0]['error']['new'].__setitem__(1, float('nan')), 'reject'),
    ('nonfinite_change', lambda d: d['nearby'][0].update(change=float('inf')), 'reject'),
    ('unchanged_input_output', lambda d: d['nearby'][0].update(change=0.), 'reject'),
    ('fp64_error', lambda d: d['fp64']['new'].__setitem__(0, .016), 'reject'),
    ('aa_drift', lambda d: d['rows'][0]['event_ms_bits'].__setitem__('a1', [bits(1.1)]*8), 'reject'),
    ('matched_control_drift', lambda d: d['rows'][0]['event_ms_bits'].__setitem__('old', [bits(1.1)]*8), 'reject'),
    ('zero_event', lambda d: d['rows'][0]['event_ms_bits']['new'].__setitem__(0, bits(0.)), 'malformed'),
    ('duplicate_timing_group', lambda d: d['rows'].__setitem__(1, copy.deepcopy(d['rows'][0])), 'malformed'),
    ('partial_failure', lambda d: d.update(status='failed', stage='nearby/17', rows=[], nearby=[], error='CPU synthetic failure'), 'partial'),
]:
    value = variant()
    mutate(value)
    cases.append((label, value, outcome))

results = []
with tempfile.TemporaryDirectory(prefix='muse_cpu_synthetic_extractor_') as tmp:
    root = Path(tmp)
    (root / 'tools').mkdir()
    track = root / 'muse_moe_w4a16'
    (track / 'run_receipts').mkdir(parents=True)
    script = root / 'tools/extract_moe_two_cta_probe.py'
    shutil.copyfile(ROOT / 'tools/extract_moe_two_cta_probe.py', script)
    for index, (label, value, expected) in enumerate(cases):
        job = f'cpu_synthetic_fixture_{index}'
        receipt = dict(cluster='CPU_SYNTHETIC_ONLY', source_sha256='NO_GPU_SUBMISSION',
                       log_tail='MOE_TWO_CTA_Z ' + env['_tc_wire'](value) + '\n')
        (track / 'run_receipts' / f'{job}.json').write_text(json.dumps(receipt))
        run = subprocess.run([sys.executable, str(script), job], capture_output=True, text=True)
        if expected == 'malformed':
            assert run.returncode != 0 and not (track / f'TWO_CTA_REPORT_{job}.json').exists(), label
        else:
            assert run.returncode == 0, (label, run.stderr)
            out = json.loads((track / f'TWO_CTA_REPORT_{job}.json').read_text())
            count = sum(r['qualifies'] for r in out['policies'])
            assert (count == 7 if expected == 'qualifies' else count == 0), (label, count)
            if expected == 'partial':
                assert out['diagnostic']['status'] == 'failed' and out['diagnostic']['error'] == value['error']
        results.append(dict(case=label, expected=expected, passed=True))

out = dict(scope='CPU synthetic parser validation only; no GPU receipt, timing or performance evidence.',
           cases=results, passed=len(results))
target = ROOT / 'muse_moe_w4a16/TWO_CTA_EXTRACTOR_PREFLIGHT_2026-11-01.json'
target.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
