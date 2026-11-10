"""Analytical experiment screening, not a benchmark or runtime policy.

Use --self-check for exact/small-limit math checks, or --input for explicitly
labeled observations. Estimates do not establish future acceptance, output
agreement, graph safety, TTFT or a serving score.
"""
import argparse
import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


def expected_committed(histogram, prefix, remaining=220):
    if len(histogram) != 8 or any(type(x) is not int or x < 0 for x in histogram):
        raise ValueError('Use eight nonnegative integer counts for accepted draft lengths0..7')
    if not sum(histogram) or type(prefix) is not int or not 0 <= prefix <= 7:
        raise ValueError('A nonempty histogram and prefix0..7 are required')
    if type(remaining) is not int or remaining <= 0:
        raise ValueError('remaining must be a positive integer')
    # One target bonus token, with the final request-length cap applied per outcome.
    return Fraction(sum(n*min(min(a, prefix)+1, remaining) for a, n in enumerate(histogram)),
                    sum(histogram))


def amdahl(fraction, local_speedup, added_fraction=0.):
    if not all(math.isfinite(x) for x in (fraction, local_speedup, added_fraction)):
        raise ValueError('Finite inputs required')
    if not 0 <= fraction <= 1 or local_speedup <= 0 or added_fraction < 0:
        raise ValueError('Use fraction0..1, positive speedup and nonnegative added cost')
    return 1 / (1-fraction + fraction/local_speedup + added_fraction)


def norm_overlap_bound(projection, router, norm):
    if any(not math.isfinite(x) or x < 0 for x in (projection, router, norm)):
        raise ValueError('Use finite nonnegative durations')
    before = max(projection, router)+norm
    after = max(projection+norm, router)
    return dict(ideal_before_us=before, ideal_after_us=after,
                ideal_saved_us=before-after,
                scope='Fixed-duration DAG bound only; resource contention and host gaps excluded.')


def bounds(value):
    if (not isinstance(value, list) or len(value) != 2 or
            any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in value) or
            value[0] > value[1]):
        raise ValueError('Cost bounds must be [finite nonnegative low, high]')
    return value


def screen(data):
    origin = data.get('data_origin')
    if origin == 'unmeasured_template':
        return dict(state='needs_measurements', native_gpu_executed=False,
                    message='No costs or acceptance observations supplied; no candidate ranking computed.')
    if origin not in ('illustrative', 'observed_component_trace'):
        raise ValueError('Label data_origin illustrative or observed_component_trace')
    if (data.get('block_size') != 8 or type(data.get('concurrency')) is not int or
            data['concurrency'] not in (1, 4, 16, 32)):
        raise ValueError('This experiment screens block8 at the four challenge loads')
    batch = data.get('active_batch_size')
    if type(batch) is not int or not 0 < batch <= data['concurrency']:
        raise ValueError('Record the actual fixed active batch size, not just client concurrency')
    if not data.get('provenance'):
        raise ValueError('Record the source/provenance of the input observations')
    draft, overhead = bounds(data['full_draft_us']), bounds(data['other_cycle_us'])
    histogram = data['accepted_draft_histogram']
    rows = []
    for prefix_text, cost in data['verify_us_by_prefix'].items():
        prefix = int(prefix_text)
        if str(prefix) != prefix_text or prefix not in (1, 3, 7):
            raise ValueError('Screen only1/3/7proposals (2/4/8logical verifier rows)')
        verify = bounds(cost)
        mu = expected_committed(histogram, prefix, data.get('remaining_tokens', 220))
        total = [draft[i]+verify[i]+overhead[i] for i in (0, 1)]
        if total[0] <= 0:
            raise ValueError('Positive cycle costs required')
        rows.append(dict(prefix=prefix, logical_verify_rows=prefix+1,
                         estimated_committed_tokens=float(mu), cycle_us_bounds=total,
                         per_request_us_per_token_bounds=[x/float(mu) for x in total],
                         aggregate_us_per_output_token_bounds=[x/(batch*float(mu)) for x in total]))
    if not rows:
        raise ValueError('No verification cost observations supplied')
    comparisons = []
    for left in rows:
        for right in rows:
            if left['prefix'] == right['prefix']:
                continue
            comparisons.append(dict(prefix=left['prefix'], compared_with=right['prefix'],
                cost_bounds_separated=left['aggregate_us_per_output_token_bounds'][1] <
                                      right['aggregate_us_per_output_token_bounds'][0]))
    return dict(state='counterfactual_screen_only', data_origin=origin,
                provenance=data['provenance'], concurrency=data['concurrency'], active_batch_size=batch,
                histogram_cycles=sum(histogram), estimates=rows, comparisons=comparisons,
                limitations=[
                    'Histogram belongs to the observed policy, not the proposed steady-state policy.',
                    'Cost bounds are supplied by the caller, not statistical confidence intervals.',
                    'Different graph buckets, KV rollback, RNG and future proposals can change the result.',
                    'No runtime selection, GPU execution, eligibility decision or serving score is produced.'])


def self_check():
    cases = 0
    for accepted in range(8):
        hist = [0]*8; hist[accepted] = 1
        for prefix in range(8):
            actual = expected_committed(hist, prefix)
            # Independent survival-sum identity, without IID acceptance assumptions.
            expected = 1+sum(accepted >= index for index in range(1, prefix+1))
            assert actual == expected
            assert expected_committed(hist, prefix, 1) == 1
            cases += 2
    assert expected_committed([1, 0, 0, 0, 0, 0, 0, 1], 7) == Fraction(9, 2)
    assert amdahl(0., 10.) == 1. and amdahl(1., 2.) == 2.
    assert math.isclose(amdahl(.8, 2.), 5/3)
    assert amdahl(.1, 2., .1) < 1.
    for p in (0., 2., 10.):
        for r in (0., 3., 20.):
            for n in (0., 1., 5.):
                result = norm_overlap_bound(p, r, n)
                assert result['ideal_saved_us'] == min(n, max(0, r-p))
                cases += 1
    fixture = dict(data_origin='illustrative', provenance='Deliberately synthetic analytic self-check only',
                   block_size=8, concurrency=1, accepted_draft_histogram=[0, 0, 0, 0, 0, 0, 0, 10],
                   active_batch_size=1,
                   full_draft_us=[10, 10], other_cycle_us=[0, 0],
                   verify_us_by_prefix={'1': [10, 10], '7': [30, 30]})
    result = screen(fixture)
    assert next(x for x in result['comparisons'] if x['prefix'] == 7)['cost_bounds_separated']
    fixture['verify_us_by_prefix']['7'] = [1, 100]
    result = screen(fixture)
    assert not any(x['cost_bounds_separated'] for x in result['comparisons'])
    assert screen(dict(data_origin='unmeasured_template'))['state'] == 'needs_measurements'
    fixture['accepted_draft_histogram'] = [1, 0, 0, 0, 0, 0, 0, 1]
    fixture['verify_us_by_prefix']['7'] = [30, 30]
    for concurrency in (1, 4, 16, 32):
        for batch in sorted({1, concurrency}):
            fixture.update(concurrency=concurrency, active_batch_size=batch)
            row = next(x for x in screen(fixture)['estimates'] if x['prefix'] == 7)
            # Two equally likely cycles commit one/eight tokens per active request.
            expected_cost = (40+40)/((1+8)*batch)
            assert math.isclose(row['aggregate_us_per_output_token_bounds'][0], expected_cost)
            cases += 1
    return dict(state='analytic_self_checks_passed', identity_and_boundary_checks=cases+8,
                scope='Synthetic mathematics only; no hardware timing, acceptance estimate or runtime policy validated.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--self-check', action='store_true')
    source.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve prior evidence: choose a fresh output path'
    result = self_check() if args.self_check else screen(json.loads(args.input.read_text()))
    result['tool_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['captured_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if args.input:
        result['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
