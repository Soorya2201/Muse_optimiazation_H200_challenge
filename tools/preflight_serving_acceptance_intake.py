"""Synthetic intake checks using pinned SGLang histogram/metadata bodies.

No server, model or GPU is executed. Fixtures contain metadata only, not
generated tokens. This validates parsing and counter semantics, not a run.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

from inspect_serving_acceptance import inspect, records

ROOT = Path(__file__).resolve().parents[1]
BASE = '297e54fd105e974cdf401e5b4d20253a22c16f90'


def stock_methods(sg):
    wanted = {
        'python/sglang/srt/managers/schedule_batch.py': 'update_spec_correct_drafts_histogram',
        'python/sglang/srt/managers/tokenizer_manager.py': '_calculate_spec_decoding_metrics',
    }
    scope = dict(get_spec=lambda: SimpleNamespace(speculative_num_draft_tokens=8),
                 _ragged_verify_cap_accept=lambda: False)
    manifest = {}
    for path, method in wanted.items():
        data = subprocess.check_output(['git', '-C', str(sg), 'show', BASE+':'+path])
        found = [node for node in ast.walk(ast.parse(data.decode()))
                 if isinstance(node, ast.FunctionDef) and node.name == method]
        assert len(found) == 1
        mod = ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')], level=0), found[0]], type_ignores=[])
        exec(compile(ast.fix_missing_locations(mod), path, 'exec'), scope)
        manifest[path] = dict(source_sha256=hashlib.sha256(data).hexdigest(), method=method)
    return scope, manifest


def body(scope, rid, accepts, completion=220, finish=None):
    req = SimpleNamespace(spec_correct_drafts_histogram=[])
    for accepted in accepts:
        scope['update_spec_correct_drafts_histogram'](req, accepted)
    meta = dict(id=rid, completion_tokens=completion,
                finish_reason=finish or dict(type='length', length=220))
    recv = SimpleNamespace(spec_verify_ct=[len(accepts)],
        spec_num_correct_drafts=[sum(accepts)], completion_tokens=[completion],
        spec_correct_drafts_histogram=[req.spec_correct_drafts_histogram])
    scope['_calculate_spec_decoding_metrics'](None, meta, recv, 0)
    return dict(text='SYNTHETIC METADATA ONLY; no generated tokens', meta_info=meta)


def run(scope):
    tested = 0
    def analyze(items, format='jsonl'):
        if format == 'jsonl':
            data = '\n'.join(json.dumps(x) for x in items)
        elif format == 'json':
            data = json.dumps(items)
        else:
            data = ''.join('data: '+json.dumps(x)+'\n\n' for x in items)+'data: [DONE]\n\n'
        return inspect(records(data, format), 32, 'Synthetic CPU parser fixture; no server or GPU')
    a, b = body(scope, 'synthetic-a', [7]*28), body(scope, 'synthetic-b', [3]*55)
    partial = body(scope, 'synthetic-a', [7]*10, completion=81)
    partial['meta_info']['finish_reason'] = None
    snapshots = [partial, a, b, a]
    for format in ('jsonl', 'json', 'sse'):
        result = analyze(snapshots, format)
        assert result['unique_final_requests'] == 2 and result['verify_request_rounds'] == 83
        assert result['partial_frames_ignored'] == 1
        assert result['identical_final_duplicates_ignored'] == 1
        assert result['client_completion_tokens'] == 440
        assert result['accepted_draft_histogram'] == [0, 0, 0, 55, 0, 0, 0, 28]
        assert result['mean_raw_commit_length_including_bonus'] == float(Fraction(444, 83))
        assert result['counterfactual_raw_yield_by_proposals'] == {
            '1': 2., '3': 4., '7': float(Fraction(444, 83))}
        assert result['actual_active_batch_size'] is None
        tested += 1
    # Source-derived final metadata uses client completion / verify_ct, not
    # mean accepted proposals + bonus: the final accepted run can be clipped.
    assert a['meta_info']['spec_accept_length'] == 220/28
    assert a['meta_info']['spec_accept_length'] != 8
    tested += 1
    rng = random.Random(110102)
    for i in range(200):
        accepts = [rng.randrange(8) for _ in range(rng.randrange(1, 80))]
        fixture = body(scope, 'random-synthetic-'+str(i), accepts)
        result = analyze([fixture])
        assert result['mean_correct_drafts'] == sum(accepts)/len(accepts)
        assert result['verify_request_rounds'] == len(accepts)
        assert result['state'] == 'metadata_consistent'
        tested += 1
    rejections = []
    def reject(label, fn):
        nonlocal tested
        try:
            fn()
        except ValueError:
            rejections.append(label)
            tested += 1
        else:
            raise AssertionError('Accepted invalid input: '+label)
    mutations = {
        'missing histogram': lambda m: m.pop('spec_correct_drafts_histogram'),
        'negative bin': lambda m: m.update(spec_correct_drafts_histogram=[-1]*8),
        'boolean bin': lambda m: m.update(spec_correct_drafts_histogram=[False]*8),
        'fractional bin': lambda m: m.update(spec_correct_drafts_histogram=[1.5]*8),
        'too many bins': lambda m: m.update(spec_correct_drafts_histogram=[0]*9),
        'wrong histogram count': lambda m: m.update(spec_verify_ct=29),
        'wrong weighted count': lambda m: m.update(spec_num_correct_drafts=195),
        'wrong draft cadence': lambda m: m.update(spec_num_proposed_drafts=224),
        'wrong acceptance rate': lambda m: m.update(spec_accept_rate=.5),
        'raw mean substituted for client accept length': lambda m: m.update(spec_accept_length=8),
        'invalid finish object': lambda m: m.update(finish_reason='length'),
        'negative completion': lambda m: m.update(completion_tokens=-1),
        'missing request id': lambda m: m.pop('id'),
        'unknown finality': lambda m: m.pop('finish_reason'),
    }
    for label, mutation in mutations.items():
        changed = copy.deepcopy(a)
        mutation(changed['meta_info'])
        reject(label, lambda changed=changed: analyze([changed]))
    changed = copy.deepcopy(a)
    changed['text'] = 'conflicting final payload'
    reject('conflicting final payload with same id', lambda: analyze([a, changed]))
    reject('partial after final', lambda: analyze([a, partial]))
    reject('non-JSON NaN', lambda: list(records('data: NaN\n\n', 'sse')))
    reject('SSE data after DONE', lambda: list(records(
        'data: [DONE]\n\ndata: '+json.dumps(a)+'\n\n', 'sse')))
    reject('invalid load declaration', lambda: inspect([a], 2, 'synthetic'))
    reject('missing provenance', lambda: inspect([a], 1, ''))
    # A capture ending before a final frame is incomplete, including when
    # other requests have final responses; do not silently qualify the subset.
    for items in ([], [partial], [partial, b]):
        assert analyze(items)['state'] == 'incomplete_or_failed_metadata'
        tested += 1
    for completion, finish in [(219, dict(type='length')), (220, dict(type='abort'))]:
        fixture = body(scope, 'failed-synthetic', [7]*28, completion, finish)
        assert analyze([fixture])['state'] == 'incomplete_or_failed_metadata'
        tested += 1
    # Proper multi-line SSE data, comments and event fields are framing only.
    data = ': comment\nevent: message\nid: ignored\nretry: 20\n'+''.join(
        'data: '+line+'\n' for line in json.dumps(a, indent=2).splitlines())+'\n'
    assert inspect(records(data, 'sse'), 1, 'synthetic')['unique_final_requests'] == 1
    tested += 1
    return snapshots, dict(synthetic_checks=tested, rejected_invalid_cases=rejections,
        scope=__doc__, native_gpu_executed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sglang', type=Path, default=ROOT/'serving/current/sglang')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.fixture.exists(), 'Preserve earlier evidence'
    scope, sources = stock_methods(args.sglang)
    fixtures, checks = run(scope)
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(''.join(json.dumps(x)+'\n' for x in fixtures))
    report = dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        base=BASE, python=sys.version, checks=checks, sources=sources,
        tools={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ('preflight_serving_acceptance_intake.py', 'inspect_serving_acceptance.py',
                            'serving_cost_models.py')},
        fixture=dict(path=str(args.fixture), sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                     scope='Synthetic metadata only, generated by executing official counter methods on CPU'))
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
