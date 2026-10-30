# Router precision audit, October 30

The scored entry `d857144e59784ad0` (Madhavan c180924, archived as
`submission_v16_madhavan_c180924.py`) rounds the router activation to BF16 at
T <= 8. This conflicts with the baseline's FP32 router activation at these sizes.
It was withdrawn on October 30. Passing numerical tests did not establish compliance.
The prior scored baseline-derived v5-v8 kernels keep `router_dtype=torch.float32`
and `F.linear(router_input, state["router"])` on the small-T branch.

## Evidence

- `baseline.py` lines 13-14 and 316-330: boundary explicitly writes an FP32
  router input for T <= 8, then applies the FP32 linear layer. Above eight,
  BF16 input and high/low BF16 weight products are used.
- Madhavan `_after_attention` lines 18647-18760: allocates only BF16 `h`.
  Both fused `bfpf` and separate `bfp` router paths read that tensor; the
  fallback `dense(h, state["pr"], ...)` also receives it. No alternate FP32
  activation exists on these paths.
- Its native `rgemv_kernel` at line 14738 accepts FP32 weights and a
  `const __nv_bfloat16*` activation. FP32 accumulation cannot recover bits
  discarded by the BF16 input conversion.
- The vjeux decoder check candidate shares this issue; it was never scored
  by us. It is retained only as historical evidence, not a compliant fallback.

## Replacement under test

Public Apache-2.0 source:
https://github.com/alygg77/okc-muse-spark-kernels/tree/6cae49a

Pinned local file: `submission_v17_alygg77_6cae49a.py`.
SHA-256: `8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a`.

- `_pool_router_fp32` sends every T <= 8 fallback through FP32 `hr` and
  FP32 `router_fp32`, using FP32 FMAs/reductions in the native router.
- The fused `_gle_frt` uses separate BF16 pooler input and FP32 router
  input/weight buffers. Its optimized small-T chain is pinned in `prepare`.
- `_gle_b1x_source` corrects the boundary copy: no BF16 rounding of the
  router's normalized branch or `(1 + w)`. BF16 pooler data is separate.
- The unfused base path writes FP32 `hr`; the parallel boundary fallback
  recomputes the FP32 router input from FP32 stream. Input-dependent work
  remains in `kernel_fn`. `prepare` benchmarks independent synthetic inputs.
- PDL data loads were inspected: new fused stages use shared/TMA inputs
  after dependency waits; glue's explicit cache-global loads avoid the
  noncoherent-load hoisting pattern. This source audit does not replace GPU
  correctness checks on fresh graph inputs.
- The source also contains KPERM3 experts for T >= 17 and context-bound,
  deepcopy-safe cuBLASLt holders. Preserve upstream notices and licenses.

First check: `521b039c454045da`, cluster `sfo3`; result pending.
Run a second independent, code-equivalent check and inspect all 12 profiles
before promotion. Correct precision is required even if performance is at parity.
