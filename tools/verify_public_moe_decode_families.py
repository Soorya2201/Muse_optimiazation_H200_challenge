"""Compare public multiply layouts with our previously measured 8-op family.

CPU bit verification only: no imports of upstream submission modules, GPU
timing, full-kernel SASS claim, or change to the canonical kernel.
"""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
old = ROOT / "muse_moe_w4a16/submission_v18_vjeux_f35a41e.py"
lee = ROOT / "muse_moe_w4a16/submission_v20_lee8_dequant.py"
rohan = ROOT / "research/public/rohanarun_018b7a1/moe_mul_decode.py"


def positions(path, name, globals_):
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = dict(globals_)
    exec(compile(ast.Module([fn], []), "<isolated position function>", "exec"), ns)
    return ns[name]()


maps = [positions(old, "_stored_pos", {"DQ7": True}),
        positions(lee, "_stored_pos", {}),
        positions(rohan, "_stored_pos_mul", {})]
for m in maps:
    assert sorted(m.values()) == list(range(32))
for m in maps[1:]:
    assert sum(1 << m[3, u, b] for u, b in itertools.product(range(2), range(4))) == 0x001B001B


def rot(q, n):
    return ((q << n) | (q >> (32 - n))) & 0xFFFFFFFF


def pack(codes, pos):
    return sum(((codes[2 * r + u] >> b) & 1) << pos[r, u, b]
               for r, u, b in itertools.product(range(4), range(2), range(4)))


def decode(codes):
    q0, q1, q2 = [pack(codes, m) for m in maps]
    mask = 0x81C081C0
    expected = tuple(sum((((codes[2*r+u] & 8) << 12) |
                          ((codes[2*r+u] & 7) << 6)) << (16*u)
                         for u in range(2)) for r in range(4))
    parent = (q0 & mask, (q0 * 16) & mask, rot(q0, 10) & mask,
              ((q0 * 64) & 0x80C080C0) | (rot(q0, 14) & 0x01000100))
    lee8 = (q1 & mask, rot(q1, 10) & mask, rot(q1, 13) & mask,
            (((q1 & 0x001B001B) * 0x1084) & 0xFFFFFFFF) & mask)
    public8 = (q2 & mask, rot(q2, 29) & mask, rot(q2, 26) & mask,
               (((q2 & 0x001B001B) * 4228) & 0xFFFFFFFF) & mask)
    assert parent == lee8 == public8 == expected, codes


count = 0
for fill, pair, codes in itertools.product((0, 15), itertools.combinations(range(8), 2),
                                           itertools.product(range(16), repeat=2)):
    word = [fill] * 8
    word[pair[0]], word[pair[1]] = codes
    decode(word)
    count += 1
for a, b, fill in itertools.product(range(16), range(16), (0, 15)):
    word = [fill] * 8
    word[6], word[7] = a, b
    decode(word)
    count += 1
rng = random.Random(0x1084)
for _ in range(20000):
    decode([rng.randrange(16) for _ in range(8)])
    count += 1

# Additional explicit bit basis and complete nonlinear multiply-mask domain.
for bit in range(32):
    word = [0] * 8
    word[bit // 4] = 1 << (bit % 4)
    decode(word)
for a, b in itertools.product(range(16), repeat=2):
    decode([0] * 6 + [a, b])

report = {
    "scope": "CPU fragment-bit/layout equivalence, not a GPU speed result or full-kernel SASS equivalence.",
    "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (old, lee, rohan)},
    "paired_and_random_words": count,
    "additional_basis_words": 32,
    "additional_complete_multiply_domain": 256,
    "all_pass": True,
    "public_vs_lee8": {
        "same_multiply": "4228 == 0x1084",
        "same_output_mask": "0x81C081C0",
        "same_multiply_input_mask": "0x001B001B",
        "difference": "Two funnel-shift constants/directions and their matching lossless packed bit positions",
        "source_operation_classes_per_8_weights": "5 masks/bitwise operations, 2 funnel shifts, 1 integer multiply",
        "limitation": "Equivalent operation classes do not prove identical scheduling, registers, or performance. Prior paired 8-op measurements are counterevidence to assuming a large gain; no new GPU check justified solely by this source.",
    },
    "prior_paired_job": "683c1eb4664a4396",
    "canonical_changed": False,
}
path = ROOT / "muse_moe_w4a16/MOE_PUBLIC_DECODE_EQUIVALENCE_2026-10-30.json"
content = json.dumps(report, indent=2) + "\n"
if path.exists():
    assert path.read_text() == content
else:
    path.write_text(content)
print(f"PASS: {count + 32 + 256} packed words; all three layouts decode identically")
print(path.relative_to(ROOT))
