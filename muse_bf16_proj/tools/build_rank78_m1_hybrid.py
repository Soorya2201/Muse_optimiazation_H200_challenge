"""Emit an apply_patch payload for the pinned Chen parent plus isolated M1 routes.

Usage from the project root: python3 muse_bf16_proj/tools/build_rank78_m1_hybrid.py | apply_patch
"""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'submission_rank78_chen_3d418e1.py'
DONOR = ROOT / 'submission_v30_all_small_qkv32.py'
TARGET = ROOT / 'submission_v33_chen_m1_hybrid.py'
EXPECTED_PARENT = 'bb50079be2ed74d65bb7109f1432fbc7a49bbe70913d247c254717cd0f27f712'
EXPECTED_DONOR = '2d133f8601b005c90d2b380b6334f23307d9a9848b97a8f15e0c1b92664f5d51'


def build():
    parent, donor = PARENT.read_bytes(), DONOR.read_bytes()
    for blob, expected in ((parent, EXPECTED_PARENT), (donor, EXPECTED_DONOR)):
        if hashlib.sha256(blob).hexdigest() != expected:
            raise RuntimeError('Pinned source hash changed; review before rebuilding')
    text = donor.decode()
    # Transfer only the independently compiled, previously judged M1 module.
    start = text.index('_V30M1_CPP = r"""')
    end = text.index('def _v30m1_autotune(', start)
    module = text[start:end]
    module = module.replace('if err>tol:',
        'if not math.isfinite(err) or not bool(torch.isfinite(out).all()) or err>tol:')
    suffix = '''
# v33: Chen Apache-2.0 parent plus vjeux/Madhavan M1 module.
# See THIRD_PARTY_LICENSES/chenwainuo-okc-APACHE-2.0.txt and
# upstream_madhavan_NOTICE for source attribution. Original CUDA is unchanged.
import hashlib
_V33_PARENT_PREPARE, _V33_PARENT_KERNEL = prepare, kernel_fn

def _v33_gate(weights, state):
    bundle = state["v33_m1"]
    choices = set()
    state["v33_choices"] = choices
    gen = torch.Generator(device=weights["qkv_gate"].device).manual_seed(3301029)
    for layer in (*bundle["plans"], "router"):
        if time.time() - _T0 > 275:
            print("[v33] remaining M1 gates skipped: prepare budget", flush=True)
            break
        w = weights[layer]
        n = bundle["plans"][layer]["n"] if layer != "router" else 256
        data = {"x": torch.randn((1, w.shape[1]), dtype=torch.bfloat16,
                                device=w.device, generator=gen),
                "config": {"m": 1, "n": n, "k": w.shape[1], "layer": layer}}
        keepalive = []
        def candidate():
            keepalive.append(_v30m1_launch(layer, data["x"], bundle))
        def baseline():
            keepalive.append(_V33_PARENT_KERNEL(data, state))
        # Use the parent's validated read flush and alternating graph pairs.
        # Require independent agreement from two gates, each with a 3% margin.
        first = _paired(candidate, baseline, pairs=8)
        second = _paired(candidate, baseline, pairs=8)
        ratios = (first[0], second[0])
        if all(math.isfinite(r) and r > 1.0 / 0.97 for r in ratios):
            choices.add(layer)
        print(f"[v33] {layer} m1 ratios={ratios} selected={layer in choices}", flush=True)

def prepare(weights):
    state = _V33_PARENT_PREPARE(weights)
    state["v33_choices"] = set()
    if "cpu" in state or time.time() - _T0 > 245:
        print("[v33] M1 extension skipped: device or prepare budget", flush=True)
        return state
    try:
        ext = _v30m1_build_ext()
        state["v33_m1"] = _v30m1_prepare(weights, ext)
        _v33_gate(weights, state)
    except Exception as error:
        state["v33_choices"] = set()
        print(f"[v33] M1 disabled: {type(error).__name__}: {str(error)[-256:]}", flush=True)
    return state

def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer = cfg["layer"]
    bundle = state.get("v33_m1")
    if bundle is not None and x.shape[0] == 1 and layer in state["v33_choices"]:
        n = 256 if layer == "router" else bundle["plans"][layer]["n"]
        k = 8192 if layer == "router" else bundle["plans"][layer]["k"]
        if (cfg["n"] == n and x.shape[1] == k and x.is_cuda
                and x.dtype == torch.bfloat16 and x.is_contiguous()
                and x.device == bundle["router"].device and x.data_ptr() % 16 == 0):
            return _v30m1_launch(layer, x, bundle)
    return _V33_PARENT_KERNEL(data, state)
'''
    result = parent.decode() + '\n' + module + suffix
    ast.parse(result)
    return result


if __name__ == '__main__':
    source = build()
    if TARGET.exists():
        if TARGET.read_text() != source:
            raise RuntimeError('Existing candidate differs; preserve it for review')
        print('*** Begin Patch\n*** End Patch')
    else:
        print('*** Begin Patch')
        print('*** Add File: muse_bf16_proj/submission_v33_chen_m1_hybrid.py')
        print('\n'.join('+' + line for line in source.splitlines()))
        print('*** End Patch')
