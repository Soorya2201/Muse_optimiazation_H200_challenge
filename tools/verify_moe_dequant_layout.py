"""Check the isolated MoE 8-op port without importing torch or using a GPU."""
import ast
import itertools
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "muse_moe_w4a16/submission_v18_vjeux_f35a41e.py"
CANDIDATE = ROOT / "muse_moe_w4a16/submission_v20_lee8_dequant.py"
parent, candidate = [ast.parse(p.read_text()) for p in (PARENT, CANDIDATE)]


def positions(tree, old=False):
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_stored_pos")
    namespace = {"DQ7": old}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<packing function>", "exec"), namespace)
    return namespace["_stored_pos"]()


old_pos, new_pos = positions(parent, True), positions(candidate)
assert sorted(new_pos.values()) == list(range(32))
assert len(parent.body) == len(candidate.body)
native = []
for p, c in zip(parent.body, candidate.body):
    if isinstance(p, ast.FunctionDef) and p.name in ("_stored_pos", "_ext"):
        assert c.name == p.name
        continue
    if isinstance(p, ast.Assign):
        names = [t.id for t in p.targets if isinstance(t, ast.Name)]
        if names == ["CUDA_SRC"]:
            native = [ast.literal_eval(n.value) for n in (p, c)]
            continue
        if names == ["DQ7"]:
            assert c.targets[0].id == "DQ8" and ast.literal_eval(c.value) is True
            continue
    assert ast.dump(p) == ast.dump(c)

dq4 = r"__device__ __forceinline__ void dq4\([^)]*\) \{.*?\n\}"
old_fn, new_fn = [re.findall(dq4, text, flags=re.S) for text in native]
assert len(old_fn) == len(new_fn) == 1
for expression in ("r0 = q & M;", "r1 = __funnelshift_l(q, q, 10) & M;",
                   "r2 = __funnelshift_l(q, q, 13) & M;",
                   "r3 = ((q & 0x001B001Bu) * 0x1084u) & M;"):
    assert new_fn[0].count(expression) == 1


def remainder(text):
    text = re.sub(dq4, "DQ4", text, flags=re.S)
    return " ".join(re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", "", text).split())


assert remainder(native[0]) == remainder(native[1]), "another device computation changed"


def rot(x, n):
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def pack(codes, pos):
    return sum(((codes[2 * reg + upper] >> bit) & 1) << pos[reg, upper, bit]
               for reg, upper, bit in itertools.product(range(4), range(2), range(4)))


def expected(codes):
    halves = [((c & 8) << 12) | ((c & 7) << 6) for c in codes]
    return tuple(halves[2 * r] | (halves[2 * r + 1] << 16) for r in range(4))


def verify(codes):
    q9, q8 = pack(codes, old_pos), pack(codes, new_pos)
    mask = 0x81C081C0
    old = (q9 & mask, (q9 * 16) & mask, rot(q9, 10) & mask,
           ((q9 * 64) & 0x80C080C0) | (rot(q9, 14) & 0x01000100))
    new = (q8 & mask, rot(q8, 10) & mask, rot(q8, 13) & mask,
           (((q8 & 0x001B001B) * 0x1084) & 0xFFFFFFFF) & mask)
    assert old == new == expected(codes), codes


count = 0
for fill in (0, 15):
    for left, right in itertools.combinations(range(8), 2):
        for a, b in itertools.product(range(16), repeat=2):
            codes = [fill] * 8
            codes[left], codes[right] = a, b
            verify(codes)
            count += 1
for a, b, fill in itertools.product(range(16), range(16), (0, 15)):
    codes = [fill] * 8
    codes[6], codes[7] = a, b
    verify(codes)
    count += 1
rng = random.Random(0x1084)
for _ in range(20000):
    verify([rng.randrange(16) for _ in range(8)])
    count += 1
print(f"PASS: {count} old/new packed words produce identical BF16 fragment bits; other device code and packers unchanged")
