"""Verify the proposed prefill K permutation against current MMA packing.

No torch, GPU execution, numerical tolerance or performance inference.
Reports exact nibble/scale coverage and integer dot invariance only.
"""
import ast
import hashlib
import json
from pathlib import Path
import random
from prefill_scale_permutation import logical_k, packed_k, e2m1_units, e4m3_units

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'muse_prefill_layer/submission.py'
source = PARENT.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == '8b52dedda281b2e642bda1d9333bcf2d002fc8611e25caa10e90ed2a849021a5'
tree = ast.parse(source)
positions = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='_CODE_POS' for t in n.targets))
assert sorted(b for v in positions.values() for b in v) == list(range(32))
assert sorted(logical_k(k) for k in range(64)) == list(range(64))
assert all(packed_k(logical_k(k)) == k and logical_k(packed_k(k)) == k for k in range(4096))


def nibble(row,k):
    return (row*7 + k*13 + (k >> 3)) & 15


def scale_byte(row,group):
    # Arbitrary raw bytes: this part proves representation, not finite arithmetic.
    return (row*31 + group*37) & 255


slots = (3,7,1,5,2,6,0,4)  # current repack's (half,hk,b) ordering
coverage = bytearray(128*4096)
words = byte_values = 0
for ks in range(64):
    for wg in range(2):
        for w in range(4):
            for g in range(8):
                for t in range(4):
                    rows = [wg*64+w*16+half*8+g for half in range(2)]
                    sf = [scale_byte(r,ks*4+t) for r in rows]
                    for j in range(4):
                        codes = [0]*8
                        for half in range(2):
                            for hk in range(2):
                                for b in range(2):
                                    i = 3-(half+2*hk)+4*b
                                    new_k = ks*64+j*16+hk*8+t*2+b
                                    old_k = logical_k(new_k)
                                    assert old_k//16 == ks*4+t
                                    r = rows[half]
                                    codes[i] = nibble(r,old_k)
                                    assert sf[half] == scale_byte(r,old_k//16)
                                    index = r*4096+old_k
                                    assert coverage[index] == 0
                                    coverage[index] = 1
                                    byte_values += 1
                        word = sum(((codes[i]>>bit)&1)<<positions[i][bit]
                                   for i in range(8) for bit in range(4))
                        # Independent inverse proves actual packed bit positions retain all codes.
                        recovered = [sum(((word>>positions[i][bit])&1)<<bit for bit in range(4))
                                     for i in range(8)]
                        assert recovered == codes
                        words += 1
assert all(coverage) and byte_values == 524288 and words == 65536

# Gated repack's row permutation remains bijective: no gate/up remapping is needed
# for weights. The GEMM1 output-to-GEMM2 X layout *does* need the K permutation.
permuted_rows = [(64*nt+8*q+lr if lr<8 else 4096+64*nt+8*q+lr-8)
                 for nt in range(64) for q in range(8) for lr in range(16)]
assert sorted(permuted_rows) == list(range(8192))

rng = random.Random(0x4B5045524D)
dot_cases = 0
seen_scales, seen_codes = set(), set()
for width in (64,128,256,4096):
    for _ in range(16):
        x = [rng.randrange(-32768,32768) for _ in range(width)]
        c = [rng.randrange(16) for _ in range(width)]
        scales = [rng.choice([v for v in range(256) if (v&127)!=127]) for _ in range(width//16)]
        seen_scales.update(scales); seen_codes.update(c)
        before = sum(x[k]*e2m1_units(c[k])*e4m3_units(scales[k//16]) for k in range(width))
        after = sum(x[logical_k(k)]*e2m1_units(c[logical_k(k)])*
                    e4m3_units(scales[logical_k(k)//16]) for k in range(width))
        assert before == after
        dot_cases += 1
assert seen_scales == {v for v in range(256) if (v&127)!=127}
assert seen_codes == set(range(16))
out = dict(scope='CPU-only prospective KPERM-style prefill encoding proof; not GPU correctness or performance.',
           parent_sha256=hashlib.sha256(source.encode()).hexdigest(),
           code_position_map=positions,permutation=[logical_k(k) for k in range(64)],
           inverse=[packed_k(k) for k in range(64)],
           bijective_k_positions=4096,packed_words=words,nibbles_and_scale_assignments=byte_values,
           gated_row_positions=len(permuted_rows),integer_dot_cases=dot_cases,
           finite_scale_patterns_sampled=len(seen_scales),code_patterns_sampled=len(seen_codes),
           per_k64_stage=dict(code_bytes=4096,scale_bytes=512,
                             parent_requested_scale_load_bytes=2048,
                             prospective_requested_scale_load_bytes=512,
                             parent_scale_pair_constructions_per_thread=4,
                             prospective_scale_pair_constructions_per_thread=1),
           missing=['Timed normalized-X gather permutation for GEMM1',
                    'GEMM1 epilogue/staging permutation for GEMM2 inputs',
                    'Full consumer source/compile/resources and pipeline inspection',
                    'FP32 accumulation-order/finite/one-hot/nearby/fresh-input GPU checks',
                    'Paired read-flush in-chain performance and independent repeats'],
           caveats=['Weight/scale values and exact rational products are preserved; FP32 reduction order changes.',
                    'Generic scale-load byte requests are not DRAM byte savings; stored bytes are unchanged.',
                    'Current same-address scale loads can broadcast in shared memory; logical requested-byte count is not measured shared-bus traffic.',
                    'GEMM1 stmatrix currently writes contiguous 8-column chunks. This permutation breaks that simple mapping.',
                    'Instruction-count savings are not cycles; packing/staging overhead can erase them.',
                    'An adaptation of prior KPERM-style scale sharing, not an original codec or measured gain.'])
path=ROOT/'muse_prefill_layer/SCALE_PERMUTATION_CPU_2026-10-31.json'
path.write_text(json.dumps(out,indent=2)+'\n')
print('PASS:',words,'packed words;',byte_values,'nibble/scale assignments;',dot_cases,'exact integer dot cases; no GPU claim')
