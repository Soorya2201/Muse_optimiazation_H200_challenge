# Humming feasibility: exact dependency before a larger MoE change

This source audit changes the next experiment, but establishes no speedup.
No Humming package was installed, built, imported or benchmarked.

The official SGLang base declares `humming-kernels[cu13]==0.1.12` in
`python/pyproject.toml:43`. That makes the installed backend a promising
candidate for a small source adapter. It is not proof that the remote image
has the package installed; verify its version and compiler on the actual node.

The previously reviewed vLLM adapter and current upstream Humming tuning code
are newer. To avoid building against the wrong API, downloaded the exact
[0.1.12 source distribution](https://pypi.org/project/humming-kernels/0.1.12/)
using PyPI's metadata URL and verified SHA256:
`c96c37c95f74379067d2d2d0c5154d12ee9a8cfa29a7c3370a7f3e34f207ca72`.
It is 259,323 bytes compressed and contains 291 regular files. Six initial
and nine additional files were downloaded/read selectively, not fully audited.
Exact archive inventory and selected-file hashes are in the two
HUMMING_PINNED reports under `research/serving/`.

## What the pinned API supports

- `humming/schema/modelopt.py:40` defines `ModeloptNvfp4WeightSchema` with
  packed E2M1 weights, E4M3 group scales at group size 16 and outer scales.
  Its conversion retains an outer scalar when one exists per expert; multiple
  scales can instead be folded into the group-scale representation. The choice
  must preserve stock's shared gate/up scale processing and post-expert norm.
- `HummingMethod.prepare_layer_meta`, `transform_humming_layer`,
  `get_default_tuning_configs` and `forward_layer` exist in 0.1.12. The forward
  accepts caller-owned outputs, sorted route IDs, expert IDs, padded token
  counts, top-k and tuning/compute configuration. These are plausible building
  blocks for uncombined per-expert outputs.
- `get_default_tuning_configs` defaults `use_f16_accum=False`. Keep BF16
  activations/output and FP32 accumulation; faster low-precision variants are
  outside this candidate's precision contract.
- SGLang's existing `HummingRunnerCore._run_indexed_gemm` demonstrates local
  alignment, two Humming projections and an activation between them. Its final
  operation weights/sums experts directly. Muse requires normalization before
  that sum, so calling the generic runner unchanged would break the contract.
- `forward_layer` passes `getattr(layer, "locks", None)`. The standalone
  `HummingLayer` constructor creates persistent locks. Graph ownership and
  overlapping-stream behavior need their own audit; the Marlin workspace
  finding must not be repeated through a different API.

## Why the latest tuning code cannot simply be copied

Current public upstream is pinned at
[`3f2fa223`](https://github.com/vllm-project/humming/tree/3f2fa2238cb87fe1ac77aa9d51d3929d143e3c61),
Apache2. Inventoried 309 files and fetched eight files. Read its README overview/support,
pyproject, complete fused-dequant header and selected SM90 policy sections;
other fetched headers were not fully audited or executed.

The current `select_sm90_small_m_schedule` uses eight-row MMA tiles with four
CTAs/SM, and can remove TMA/producer overhead for short expert grids. It also
uses newer scheduling abstractions. The pinned 0.1.12 `Sm90Heuristics` has a
different interface and does not contain that policy function. The latest
repository's reported H200 gains are external microbenchmarks, not evidence
for the old installed package or our serving workload.

All current include files total 533,181 bytes before diff overhead, Python or
licenses. The old package's include files total 328,969 bytes and its entire
`humming/` tree 816,019 bytes. Neither compressed archive size nor wheel size
is the submission source-diff size. Vendoring the whole package is unsuitable
for the 400,000 byte cap. The existing pinned dependency is the practical first
route; a bounded tuning adaptation would need exact old-API validation.

## Concrete next spike

1. Verify 0.1.12 imports, compiler availability and graph-safe allocation on a
   development GPU. Compare its stock indexed BF16/FP32 path with native Marlin
   using the same synthetic NVFP4 tensors, active-route masks and paired timing.
   Keep this separate from official serving results.
2. If worthwhile, add a preparation path before Marlin repacking. Muse's current
   `_process_weights_for_marlin` aliases native names, repacks and deletes the
   original names. An adapter must establish one canonical Humming layout and
   release the originals, rather than retaining both large formats.
3. Match stock's gate/up and global-scale processing. Run two projections and
   retain the exact existing post-expert normalization/finalization. Audit
   inactive-output behavior and use per-capture/per-call lock ownership as
   required by the kernel, not a shared cache justified only by lock reset.
4. Measure decode/verify and chunked-prefill shapes before selecting tactics.
   Default small synthetic dimensions do not establish real-layer throughput;
   stock full-model quality, acceptance, TTFT and all four loads still gate use.

This is an unimplemented candidate. The four existing mechanisms and frozen
patches remain unchanged while the native validation harness is prepared.
