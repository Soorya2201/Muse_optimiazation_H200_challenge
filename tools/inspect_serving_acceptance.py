"""Inspect native /generate metadata offline, without GPU or server changes.

Supports a JSON body, JSONL snapshots, or native SSE. Final cumulative
histograms are counted once per request; partial frames are not observations
of independent decode rounds. This cannot establish quality, TTFT, active
batch, latency, a serving score, or shortened-policy steady-state acceptance.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import traceback

from serving_cost_models import expected_committed


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def decode(text):
    def invalid(value):
        raise ValueError('Non-JSON numeric constant: '+value)
    return json.loads(text, parse_constant=invalid)


def records(text, format):
    """Yield response objects, preserving SSE framing and batch response IDs."""
    def flatten(body):
        bodies = body if isinstance(body, list) else [body]
        if not bodies:
            raise ValueError('Empty response batch')
        for item in bodies:
            if not isinstance(item, dict):
                raise ValueError('Expected native /generate response objects')
            yield item
    if format == 'json':
        yield from flatten(decode(text))
    elif format == 'jsonl':
        for line in text.splitlines():
            if line.strip():
                yield from flatten(decode(line))
    elif format == 'sse':
        parts, done = [], False
        def frame():
            return '\n'.join(parts)
        for line in [*text.splitlines(), '']:
            if not line:
                payload = frame()
                parts = []
                if not payload:
                    continue
                if payload == '[DONE]':
                    done = True
                    continue
                if done:
                    raise ValueError('SSE payload after [DONE]')
                yield from flatten(decode(payload))
            elif line.startswith('data:'):
                data = line[5:]
                parts.append(data[1:] if data.startswith(' ') else data)
            elif not (line.startswith(':') or line.startswith(('event:', 'id:', 'retry:'))):
                raise ValueError('Unsupported native SSE line')
    else:
        raise ValueError('Choose json, jsonl or sse')


def final_metadata(body):
    meta = body['meta_info']
    hist = meta.get('spec_correct_drafts_histogram')
    if not isinstance(hist, list) or not 1 <= len(hist) <= 8:
        raise ValueError('Final DFlash metadata needs a 1..8-bin accepted-draft histogram')
    hist = [integer(n, 'histogram count') for n in hist]+[0]*(8-len(hist))
    count = integer(meta.get('spec_verify_ct'), 'spec_verify_ct', 1)
    accepted = integer(meta.get('spec_num_correct_drafts'), 'spec_num_correct_drafts')
    proposed = integer(meta.get('spec_num_proposed_drafts'), 'spec_num_proposed_drafts')
    completion = integer(meta.get('completion_tokens'), 'completion_tokens')
    if sum(hist) != count:
        raise ValueError('Histogram count differs from spec_verify_ct')
    if sum(i*n for i, n in enumerate(hist)) != accepted:
        raise ValueError('Weighted histogram differs from spec_num_correct_drafts')
    if proposed != 7*count:
        raise ValueError('Not the declared fixed 8-position/7-proposal cadence')
    for name, expected in [('spec_accept_rate', accepted/proposed),
                           ('spec_accept_length', completion/count)]:
        if name in meta:
            value = meta[name]
            if (type(value) not in (int, float) or not math.isfinite(value) or
                    not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9)):
                raise ValueError(name+' conflicts with pinned source semantics')
    finish = meta['finish_reason']
    if not isinstance(finish, dict) or not isinstance(finish.get('type'), str):
        raise ValueError('Expected native finish_reason object')
    return dict(request_id_sha256=hashlib.sha256(meta['id'].encode()).hexdigest(),
                accepted_draft_histogram=hist, verify_request_rounds=count,
                correct_drafts=accepted, proposed_drafts=proposed,
                completion_tokens=completion, finish_type=finish['type'],
                normal_finish=finish['type'] in ('length', 'stop'),
                expected_220_tokens=completion == 220,
                raw_commit_tokens_including_bonus=accepted+count,
                client_tokens_minus_raw_commits=completion-accepted-count)


def inspect(bodies, concurrency, provenance):
    if type(concurrency) is not int or concurrency not in (1, 4, 16, 32):
        raise ValueError('Declare the client concurrency: 1, 4, 16 or 32')
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError('Describe the supplied run/source; do not treat fixtures as GPU evidence')
    finals, seen, partials, duplicates = {}, set(), 0, 0
    for body in bodies:
        meta = body.get('meta_info')
        if not isinstance(meta, dict) or not isinstance(meta.get('id'), str) or not meta['id']:
            raise ValueError('Native response requires meta_info.id')
        rid = meta['id']
        seen.add(rid)
        if 'finish_reason' not in meta:
            raise ValueError('Response omits finish_reason; finality is unknown')
        if meta['finish_reason'] is None:
            if rid in finals:
                raise ValueError('Partial snapshot follows final response')
            partials += 1
            continue
        fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if rid in finals:
            if fingerprint != finals[rid][0]:
                raise ValueError('Conflicting final responses share a request ID')
            duplicates += 1
            continue
        finals[rid] = (fingerprint, final_metadata(body))
    rows = [row for _, row in finals.values()]
    hist = [sum(row['accepted_draft_histogram'][i] for row in rows) for i in range(8)]
    count = sum(hist)
    unfinished = len(seen)-len(finals)
    all_final = bool(rows) and unfinished == 0
    gates = dict(all_seen_requests_final=all_final,
                 all_finals_have_220_tokens=bool(rows) and all(r['expected_220_tokens'] for r in rows),
                 all_finals_normal=bool(rows) and all(r['normal_finish'] for r in rows))
    return dict(state='metadata_consistent' if all(gates.values()) else 'incomplete_or_failed_metadata',
        scope=__doc__, provenance=provenance,
        caller_declared_algorithm='DFLASH', caller_declared_block_size=8,
        caller_declared_client_concurrency=concurrency, actual_active_batch_size=None,
        unique_requests_seen=len(seen), unique_final_requests=len(rows),
        unfinished_requests=unfinished, partial_frames_ignored=partials,
        identical_final_duplicates_ignored=duplicates, metadata_gates=gates,
        accepted_draft_histogram=hist, verify_request_rounds=count,
        mean_correct_drafts=None if not count else sum(i*n for i, n in enumerate(hist))/count,
        mean_raw_commit_length_including_bonus=None if not count else
            sum((i+1)*n for i, n in enumerate(hist))/count,
        client_completion_tokens=sum(r['completion_tokens'] for r in rows),
        counterfactual_raw_yield_by_proposals={} if not count else {
            str(k): float(expected_committed(hist, k, 220)) for k in (1, 3, 7)},
        requests=rows,
        limitations=[
            'Caller-supplied bytes and labels are not proof of hardware, flags or official execution.',
            'Client concurrency is not the active or CUDA-graph-padded batch size.',
            'Histograms are request-round weighted and have no cycle timing or per-round remaining-budget association.',
            'Counters precede final output-length clipping; mean(A+1) need not equal completion_tokens/verify_ct.',
            'Raw yields are full-policy counterfactuals, not shortened-policy steady-state acceptance.',
            'No answers, TTFT, speedup, winner or serving score are validated by this tool.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--format', choices=('json', 'jsonl', 'sse'), required=True)
    parser.add_argument('--concurrency', type=int, choices=(1, 4, 16, 32), required=True)
    parser.add_argument('--provenance', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh receipt path')
    data = args.input.read_bytes()
    report = dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cost_tool_sha256=hashlib.sha256(Path(__file__).with_name('serving_cost_models.py').read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(data).hexdigest(), input_path=str(args.input),
        native_gpu_executed_by_this_tool=False, state='parsing')
    try:
        report.update(inspect(records(data.decode(), args.format), args.concurrency, args.provenance))
    except BaseException:
        report.update(state='failed', error=traceback.format_exc())
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({name: report[name] for name in ('state', 'unique_final_requests', 'verify_request_rounds')}))


if __name__ == '__main__':
    main()
