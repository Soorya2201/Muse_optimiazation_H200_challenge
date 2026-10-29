"""Offline bit-exact check of the proposed prefill NVFP4 register placement.

No torch/GPU dependency. Judge correctness/timing must still gate promotion.
"""
import ast
import itertools
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "muse_prefill_layer/submission_v18_lee8_dequant.py"
source = SOURCE.read_text()
tree = ast.parse(source)
positions = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_CODE_POS" for t in n.targets))
cuda = next(ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_CUDA_SRC" for t in n.targets))
for instruction in (
    "const uint32_t r0 = q & M;",
    "const uint32_t r1 = __funnelshift_l(q, q, 10) & M;",
    "const uint32_t r2 = __funnelshift_l(q, q, 13) & M;",
    "const uint32_t r3 = ((q & 0x001B001Bu) * 0x1084u) & M;",
):
    assert cuda.count(instruction) == 1, instruction
assert sorted(itertools.chain.from_iterable(positions.values())) == list(range(32))

# Verify the candidate only changes the encoding and its matching CUDA converter.
parent_tree = ast.parse((ROOT / "muse_prefill_layer/submission_v17_vjeux_f35a41e.py").read_text())
assert len(parent_tree.body) == len(tree.body)
parent_cuda = None
for parent_node, candidate_node in zip(parent_tree.body, tree.body):
    names = [t.id for t in parent_node.targets if isinstance(t, ast.Name)] if isinstance(parent_node, ast.Assign) else []
    if names == ["_CODE_POS"]:
        continue
    if names == ["_CUDA_SRC"]:
        parent_cuda = ast.literal_eval(parent_node.value)
        continue
    assert ast.dump(parent_node) == ast.dump(candidate_node), names


def without_comments(code):
    return " ".join(re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", "", code).split())


expected_cuda = parent_cuda.replace("const uint32_t r1 = (q * m16) & M;",
                                   "const uint32_t r1 = __funnelshift_l(q, q, 10) & M;")
expected_cuda = expected_cuda.replace("const uint32_t r2 = __funnelshift_l(q, q, 10) & M;",
                                   "const uint32_t r2 = __funnelshift_l(q, q, 13) & M;")
expected_cuda = expected_cuda.replace("const uint32_t r3 = ((q * m64) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);",
                                   "const uint32_t r3 = ((q & 0x001B001Bu) * 0x1084u) & M;")
assert without_comments(expected_cuda) == without_comments(cuda)


def pack(codes):
    return sum(((code >> bit) & 1) << positions[slot][bit]
               for slot, code in enumerate(codes) for bit in range(4))


def rotate(x, n):
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def verify(codes):
    word = pack(codes)
    mask = 0x81C081C0
    actual = (word & mask, rotate(word, 10) & mask, rotate(word, 13) & mask,
              (((word & 0x001B001B) * 0x1084) & 0xFFFFFFFF) & mask)
    # Original logical fragment order: rows g/g+8, k subgroups 0/2;
    # nibble slots are (3,7), (2,6), (1,5), (0,4), independently of bit layout.
    expected = tuple((((codes[lo] & 8) << 12) | ((codes[lo] & 7) << 6))
                     | ((((codes[hi] & 8) << 12) | ((codes[hi] & 7) << 6)) << 16)
                     for lo, hi in ((3, 7), (2, 6), (1, 5), (0, 4)))
    assert actual == expected, (codes, actual, expected)


checked = 0
for fill in (0, 15):
    for left, right in itertools.combinations(range(8), 2):
        for a, b in itertools.product(range(16), repeat=2):
            codes = [fill] * 8
            codes[left], codes[right] = a, b
            verify(codes)
            checked += 1
# All 256 combinations of the multiply path's eight source bits, with other
# registers both zero and all ones, cover potential cross-bit carry corruption.
for a, b, fill in itertools.product(range(16), range(16), (0, 15)):
    codes = [fill] * 8
    codes[0], codes[4] = a, b
    verify(codes)
    checked += 1
rng = random.Random(0x1084)
for _ in range(20000):
    verify([rng.randrange(16) for _ in range(8)])
    checked += 1
print(f"PASS: {checked} packed words, all16 codes, pair interactions and multiply carries; GPU check still required")
