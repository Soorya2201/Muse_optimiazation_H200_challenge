"""CPU layout/cost preflight of the four-width decoder experiment.

No challenge/reference import, GPU timing, or claim about instruction-cache
latency. The byte counts distinguish shared padding from global traffic.
"""
import ast
import csv
import difflib
import gzip
import hashlib
import json
from pathlib import Path
from decoder_wgs_variants import rounded_widths

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / 'muse_decode_layer'
parent = TRACK / 'submission.py'
parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
assert parent_sha == '8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a'
tree = ast.parse(parent.read_text())
node = next(n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'MOE_CUDA_SRC' for t in n.targets))
native = ast.literal_eval(node.value)
changed = rounded_widths(native)
native_path = TRACK / 'ROUNDED_WIDTHS_NATIVE_2026-10-31.cu.gz'
with gzip.open(native_path, 'wt') as f:
    f.write(changed)
diff = ''.join(difflib.unified_diff(native.splitlines(True), changed.splitlines(True),
                                  fromfile='parent_native.cu', tofile='rounded_widths_native.cu'))
(TRACK / 'ROUNDED_WIDTHS_NATIVE_2026-10-31.diff').write_text(diff)

def wide(n):
    assert 1 <= n <= 8
    return n if n <= 2 else 4 if n <= 4 else 8

# Every valid row/lane stores one aligned 16-byte vector per stage. The
# parent's layout has 32 K groups per n-tile. Widening appends complete
# 4096-byte tiles after the unchanged copied prefix.
addresses = 0
zero_vectors = 0
per_count = []
for count in range(1, 65):
    ntt = (count + 7) // 8
    issued = wide(ntt)
    copied = ntt * 4096
    padded = issued * 4096
    assert padded <= 32768
    valid = set()
    for row in range(count):
        for group in range(32):
            b = (((row // 8) * 32 + group) * 8 + (row % 8)) * 16
            assert b % 16 == 0 and 0 <= b < copied
            assert b not in valid
            valid.add(b)
    addresses += len(valid)
    zeros = []
    for lane in range(32):
        zeros.extend(range(copied + lane * 16, padded, 512))
    assert len(zeros) == len(set(zeros))
    assert set(zeros) == set(range(copied, padded, 16))
    assert not valid.intersection(zeros)
    zero_vectors += len(zeros)
    # Epilogue/partial publication guards use actual ntt. The existing
    # flush consumes and clears exactly those same float4 partial slots.
    published = [n for n in range(issued) if n < ntt]
    assert published == list(range(ntt))
    for stage in range(16):
        base = stage * copied
        assert base + copied <= ntt * 8 * 4096 * 2
        for row in range(count):
            for group in (0, 1, 7, 16, 31):
                before = (((stage * ntt + row // 8) * 32 + group) * 8 + row % 8) * 16
                after = base + (((row // 8) * 32 + group) * 8 + row % 8) * 16
                assert before == after
    per_count.append(dict(count=count, actual_ntt=ntt, issued_ntt=issued,
                          copied_stage_bytes=copied, added_shared_zero_bytes=padded-copied))

unchanged = {}
for name, end_marker in [
    ('wgs_gather', '// Work left from the previous segment'),
    ('wgs_epi_tile', 'template <bool PB>\n__device__ __forceinline__ void wgs_flush'),
    ('wgs_flush', '// one segment (stages'),
]:
    def fragment(s):
        a = s.index('void ' + name + '(')
        b = s.index(end_marker, a)
        return s[a:b]
    assert fragment(native) == fragment(changed)
    unchanged[name] = hashlib.sha256(fragment(native).encode()).hexdigest()
assert native[native.index('// cost-weighted stage ranges:'):] == changed[changed.index('// cost-weighted stage ranges:'):]

rows = []
csv_path = ROOT / 'research/public/trxphxle_72db638/experiments/experts-sched/round4-tuples.csv'
for row in csv.DictReader(csv_path.open()):
    hist = json.loads(row['n_e'])
    jobs = [min(n-j, 64) for n in hist for j in range(0, n, 64)]
    assert len(jobs) == int(row['J64'])
    actual = [(n+7)//8 for n in jobs]
    issued = [wide(n) for n in actual]
    old_sum, new_sum = sum(actual), sum(issued)
    zero_bytes = (new_sum-old_sum) * 4096 * (512+256)
    rows.append(dict(case_id=row['case_id'], jobs=len(jobs),
                     actual_width_histogram={str(n):actual.count(n) for n in range(1,9)},
                     changed_jobs=sum(a!=b for a,b in zip(actual,issued)),
                     old_ntt_sum=old_sum,new_ntt_sum=new_sum,
                     logical_mma_n_expansion=new_sum/old_sum,
                     added_shared_zero_bytes=zero_bytes,
                     added_hbm_activation_bytes=0,
                     baseline_weight_bytes=int(row['planned_weight_bytes'])))
assert len(rows) == 18
out = dict(scope='CPU address/guard assertions and external-fixture work counts only. Not native ISA execution, synchronization proof, numerical validation, code-size measurement or runtime prediction.',
           parent_sha256=parent_sha, native_parent_sha256=hashlib.sha256(native.encode()).hexdigest(),
           native_candidate_sha256=hashlib.sha256(changed.encode()).hexdigest(),
           public_parent='https://github.com/alygg77/okc-muse-spark-kernels/tree/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0',
           license='Derived Apache-2.0 parent; retain existing author and MIT notices in enclosing submission.',
           counts_verified=64,valid_vector_addresses=addresses,
           added_zero_vector_addresses=zero_vectors,per_count=per_count,
           unchanged_native_fragments=unchanged,
           unchanged_from_cost_tables_through_host_wrapper=True,
           extra_scratch_bytes=0,consumer_threads=512,producer_threads=64,
           remaining_widths=[1,2,4,8],original_widths=list(range(1,9)),
           external_csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
           external_fixtures=rows)
(TRACK / 'ROUNDED_WIDTHS_PREFLIGHT_2026-10-31.json').write_text(json.dumps(out,indent=2)+'\n')
print('COUNTS',len(per_count),'VALID_VECTORS',addresses,'ZERO_VECTORS',zero_vectors)
for row in rows:
    print(row['case_id'], 'mma_n', round(row['logical_mma_n_expansion'],4),
          'shared_zero_MiB',round(row['added_shared_zero_bytes']/2**20,3))
