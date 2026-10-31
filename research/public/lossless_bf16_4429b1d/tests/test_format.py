"""CPU tests: the 12-bit format is bit-exact for every BF16 pattern, in every layout."""
import math

import pytest
import torch

from lossless_bf16 import k_permutation, pack, unpack_reference
from lossless_bf16 import format as fmt

LAYOUTS = ("rowtile", "rowpair", "blocked8", "blocked16")


def bf16_from_bits(bits: torch.Tensor) -> torch.Tensor:
    b = bits.to(torch.int32) & 0xFFFF
    return torch.where(b >= 2 ** 15, b - 2 ** 16, b).to(torch.int16).view(torch.bfloat16)


def bits_of(w: torch.Tensor) -> torch.Tensor:
    return w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def assert_roundtrip(w: torch.Tensor, layouts=LAYOUTS):
    p = pack(w, layouts="all")
    n = w.shape[0]
    for layout in layouts:
        if layout.startswith("blocked") and (n // 16) % int(layout[7:]):
            continue
        u = unpack_reference(p, layout)
        assert u.dtype == torch.bfloat16 and u.shape == w.shape
        diff = (bits_of(u) != bits_of(w)).sum().item()
        assert diff == 0, f"{layout}: {diff} weights differ"
    return p


def expected_escapes(w: torch.Tensor) -> torch.Tensor:
    """Independent statement of the escape rule: not normal, or more than 15 binades below the row's largest normal
    exponent."""
    b = bits_of(w)
    e = (b >> 7) & 0xFF
    normal = (e > 0) & (e < 255)
    emax = torch.where(normal, e, torch.zeros_like(e)).amax(dim=1, keepdim=True)
    return ~normal | (e < emax - 15)


def test_all_65536_patterns_sorted():
    w = bf16_from_bits(torch.arange(65536)).view(128, 512)
    p = assert_roundtrip(w)
    assert p.n_escapes == int(expected_escapes(w).sum())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_all_65536_patterns_shuffled(seed):
    g = torch.Generator().manual_seed(seed)
    w = bf16_from_bits(torch.randperm(65536, generator=g)).view(128, 512)
    p = assert_roundtrip(w)
    assert p.n_escapes == int(expected_escapes(w).sum())


def test_all_65536_patterns_every_row():
    # each of 16 rows holds all 65536 patterns once (different column order per row)
    g = torch.Generator().manual_seed(7)
    rows = [torch.randperm(65536, generator=g) for _ in range(16)]
    w = bf16_from_bits(torch.stack(rows))
    assert_roundtrip(w, layouts=("rowtile", "rowpair"))


@pytest.mark.parametrize("scale", [2.0 ** -126, 1e-30, 1e-3, 1.0, 1e3, 1e30, 2.0 ** 120])
def test_random_normal_scales(scale):
    g = torch.Generator().manual_seed(11)
    w = (torch.randn((64, 2048), generator=g, dtype=torch.float64) * scale).to(torch.bfloat16)
    p = assert_roundtrip(w)
    assert p.n_escapes == int(expected_escapes(w).sum())


def test_escape_heavy_rows():
    g = torch.Generator().manual_seed(3)
    n, k = 64, 1024
    # magnitudes log-uniform over 2^-140 .. 2^127 (subnormals included), random signs
    mag = torch.exp2(torch.rand((n, k), generator=g, dtype=torch.float64) * 267 - 140)
    sign = torch.randint(0, 2, (n, k), generator=g) * 2 - 1
    w = (mag * sign).to(torch.bfloat16)
    w[0] = 0
    w[1, ::2] = -0.0
    w[2, :40] = 0
    p = assert_roundtrip(w)
    assert p.n_escapes == int(expected_escapes(w).sum())
    assert p.n_escapes > n * k // 2


def test_special_values():
    specials = torch.tensor([
        0x0000, 0x8000,                  # +0, -0
        0x0001, 0x8001, 0x007F, 0x807F,  # smallest / largest subnormals
        0x0080, 0x8080,                  # smallest normals
        0x7F7F, 0xFF7F,                  # largest finite
        0x7F80, 0xFF80,                  # +-inf
        0x7FC0, 0xFFC0, 0x7F81, 0x7FBF, 0xFF81, 0x7FFF,   # quiet and signaling NaNs with payloads
        0x3F80, 0xBF80, 0x3800, 0x37FF,  # 1, -1, 2^-15, just below 2^-15
    ], dtype=torch.int32)
    g = torch.Generator().manual_seed(5)
    base = (torch.randn((32, 512), generator=g) * 0.02).to(torch.bfloat16)
    bits = bits_of(base)
    pos = torch.randperm(32 * 512, generator=g)[: 22 * 20].view(20, 22)
    for r in range(20):
        bits.view(-1)[pos[r]] = specials
    bits[30] = 0x7FC0          # a row of NaNs
    bits[31] = 0x8000          # a row of -0
    w = bf16_from_bits(bits)
    assert_roundtrip(w)


def test_constant_and_one_binade_rows():
    w = torch.full((32, 512), 0.75, dtype=torch.bfloat16)
    w[1] = -3.0
    w[2] = torch.linspace(1.0, 1.99, 512).to(torch.bfloat16)
    p = assert_roundtrip(w)
    assert p.n_escapes == 0


def test_window_edges():
    # row max 1.0 (exponent 127): exponents 112..127 stored, 111 escaped
    w = torch.zeros((16, 512), dtype=torch.bfloat16)
    w[:, 0] = 1.0
    w[:, 1] = 2.0 ** -15        # bottom of the window
    w[:, 2] = 2.0 ** -16        # one binade below: escape
    w[:, 3] = -(2.0 ** -15) * 1.9921875
    w[:, 4:] = 0.5
    p = assert_roundtrip(w)
    assert p.n_escapes == 16 * 1     # only the 2^-16 column
    tab = p.rowtile["tab"]
    assert int(tab[0, 0, 0]) == 16
    assert set(tab[0, 1:17, 0].tolist()) == {2}


def test_scaled_decode_is_fragment_exact():
    """The register decode of every stored weight equals the row-scaled BF16 pattern of that weight."""
    g = torch.Generator().manual_seed(9)
    w = (torch.randn((48, 256), generator=g) / 16).to(torch.bfloat16)
    w[5, 7] = 0
    enc = fmt.encode_chunk_major(w)
    scaled = fmt.decode_chunk_major(enc["wp"], *w.shape)
    b = bits_of(w)
    s = fmt.row_shift(b)
    ok = fmt.in_window(b, s)
    want = (b & 0x807F) | ((((b >> 7) & 0xFF) + s[:, None]) << 7)
    assert torch.equal(scaled[ok], want[ok])
    assert bool((scaled[~ok] == 0x3800).all())        # escapes decode to +2^-15
    assert ((scaled >> 7) & 0xFF).min() >= 112 and ((scaled >> 7) & 0xFF).max() <= 127


def test_twelve_bits_per_weight():
    w = (torch.randn((128, 1024)) / 32).to(torch.bfloat16)
    p = pack(w, layouts="all")
    assert p.rowtile["wp"].numel() * 8 == 12 * w.numel()
    assert p.rowpair["wp"].numel() * 8 == 12 * w.numel()
    for rt, t in p.blocked["wpb"].items():
        assert t.numel() * 8 == 12 * w.numel()


def test_blocked_is_a_permutation_of_rowtile():
    w = (torch.randn((256, 512)) / 16).to(torch.bfloat16)
    p = pack(w, layouts="all")
    nrt, kcn = 16, 8
    rowtile = p.rowtile["wp"].view(nrt, kcn, 1536)
    for rt, t in p.blocked["wpb"].items():
        blocks = t.view(nrt // rt, kcn, rt, 1536)
        for b in range(nrt // rt):
            for kc in range(kcn):
                for r in range(rt):
                    assert torch.equal(blocks[b, kc, r], rowtile[b * rt + r, kc])


def test_k_permutation_is_a_permutation_with_contiguous_lane_runs():
    k = 512
    perm = k_permutation(k)
    assert torch.equal(torch.sort(perm).values, torch.arange(k))
    # lane t of a chunk reads columns [16t, 16t + 16) across the 4 k-steps: stored positions 16j + 2t + e + 8h
    for kc in range(k // 64):
        for t in range(4):
            stored = [kc * 64 + 16 * j + 2 * t + e + 8 * h for j in range(4) for h in range(2) for e in range(2)]
            assert sorted(perm[stored].tolist()) == list(range(kc * 64 + 16 * t, kc * 64 + 16 * t + 16))


def test_gaussian_escape_rate_matches_gpu_logs():
    """Synthetic N(0, 1/K) weights of the 2304 x 8192 shape: the GPU runs counted 1222-1342 escapes
    (results/escape_counts.csv); a CPU sample from the same distribution should land in that neighbourhood."""
    g = torch.Generator().manual_seed(2026)
    w = (torch.randn((2304, 8192), generator=g) / math.sqrt(8192)).to(torch.bfloat16)
    esc = int(expected_escapes(w).sum())
    p = pack(w)
    assert p.n_escapes == esc
    assert 1100 <= esc <= 1500, esc
