# Prefill public-source and library follow-up

The fresh public code search still returns118 paths/13repos, with zero new
path/blob entries versus the earlier Oct31 capture. The owner metadata
screen now covers752ranked entrants:179above rank180,572below,ourselves.
Four unresolved accounts: Abarbir12#16, Pinoeer-kingxi#163,
accountforgoofystuff#208 and gurrapu21#304. Cached public repo metadata was
reused from the decoder screen only when its original per-account timestamp
was under one hour old; leaderboard positions were freshly requested.
These are metadata/index screens, not all-source/private/all-branch access.

Eleven repo matches are new versus the previous prefill owner screen.
Four SGLang forks' default branches have0commits ahead/0changed files;
other branches are outside that comparison. Alygg77 is already the adopted
public family; NVFP4 utilities are component leads, pocket-muse is a separate
model, and lysithea is a utility fork. Records are
`GLOBAL_CODE_SEARCH_LATE_2026-10-31.json`,
`RANKED_GITHUB_REPO_SCREEN_2026-10-31.json` and
`NEW_REPO_FOLLOWUP_2026-10-31.json`. The code-index refresh helper now accepts
explicit previous/output filenames, preserving all earlier immutable captures.

## New challenge implementation

[trxphxle's prefill source](https://github.com/trxphxle/okc-2026-muse-kernels/blob/72db638fa2718bc0ea5645fdd5c52a5511594579/kernels/prefill_layer.py)
is Apache2 with upstream notices. The460992-byte file's SHA-256
`59177ca6...02f0495` matches its pinned publisher BEST manifest. Reported
ranked1.4774x is below our1.784958x; these are separate jobs, not a paired
causal comparison. The late grouped attach/dispatch, independent validation
and native consumer/packing sections were inspected without execution.
Its grouped persistence/variable widths/conversion family already exists
in our newer prefill parent; no stronger additive measured component found.
This is not a complete native conformance audit. The source remains in
temporary storage, with URL/hash/manifest/scope in
`TRXPHXLE_COMPONENT_SCREEN_2026-10-31.json`; no port or whole-source mirror.

## Hugging Face finegrained-kernels

[Commit5fe9dbb](https://github.com/huggingface/kernels-community/tree/5fe9dbb3d45f6f2652958c1f70e5381891fa7aa0/finegrained-kernels).
Eight selected files Git-blob/SHA verified in temporary research storage.
README/build.toml declare Apache2; no implementation copied or executed.
`FINEGRAINED_COMPONENT_SCREEN_2026-10-31.json` records exact paths/hashes.

The explicit `activation_format="bf16"` grouped path accepts NVFP4 E2M1
nibbles, group16E4M3 scales and separate FP32 per-expert globals. It
upcasts weights in-loop and uses BF16 dot, with persistent scheduling,
optional TMA, gather/scatter maps and gate epilogue. This is a compatible
arithmetic lead in principle. Do not reject it simply because its other
branches quantize activations. Conversely, do not enable its Blackwell
scaled-MMA, activation quantization or intermediate requantization here.

Published tuning/validation is B200; prefill benchmarks are eager. Those
results are not H200 paired cold-graph evidence or a current-parent win.
The inspected path supplies no proven cheaper native decoder than our
specialized packing. Full alternate scheduling/port remains unbenchmarked;
no whole-package check is allocated merely from benchmark headlines.

## SonicMoE

[Primary paper](https://arxiv.org/abs/2512.14080v2) describes IO/MMA overlap,
gather fusion and tile-aware routing. Its training throughput results use
different baselines and do not establish an NVFP4 Muse inference speedup.
Its token-rounding method changes routes; it cannot be substituted for the
challenge's required sigmoid/top8 computation as a generic padding trick.

[Upstream7396f3e](https://github.com/Dao-AILab/sonic-moe/tree/7396f3e604827d8186c2e16e64b28ee33d3defd0)
is Apache2. Four selected unmodified license/docs/forward-dispatch files
are pinned with provenance in`research/public/sonic_moe_7396f3e/`.
The reviewed forward calls ordinary Quack weights/gemm_act/gemm without a
packed-NVFP4 block/global-scale interface. There is no ready task-compatible
replacement in those reviewed paths. Separate downstream NVFP4 integrations
are not covered by this bounded upstream source inspection.

The package also modifies dependency autotuner/config methods at import.
No package was imported or run; a self-contained adaptation would need to
isolate components rather than import that module-changing initialization.
Its softmax router and ordinary expert aggregation also require replacement
with the task's sigmoid routing and post-expert-norm contract. This is an
applicability review, not a claim that the library is generally invalid.

## DeepGEMM-for-sail

[Currenta90b603](https://github.com/t-head/DeepGEMM-for-sail/tree/a90b6037c6a578e55bdd8882b84de85ed155f826)
was screened after search returned its W4A16 API. README/platform/build
requirements explicitly target T-Head PPU, HGCC/HGRTC and PPU SDK; the
weight recipes list MXFP4/E8M0 and INT4/BF16. It is not a directly executable
H200/NVFP4 backend. Metadata/README hash and source URL are in
`SAIL_COMPONENT_SCREEN_2026-10-31.json`. No installation or source port.

## Implementation lead retained

The distinct current work is an independently derived prefill K64
scale-sharing permutation inspired by existing KPERM3. Its65536-word/
524288-assignment CPU proof passes, including all254finite scale byte
patterns across exact integer-dot tests. The activation/gated-intermediate
layout cost remains unresolved; it is not an H200 candidate yet. Read
[SCALE_SHARE_RESEARCH_2026-10-31.md](SCALE_SHARE_RESEARCH_2026-10-31.md).
The new`RESEARCH_LIBRARY.md` links all current/historical methods and
closed experiments. No new GPU run, scored submission or claimed gain.

Verification: all authored edits pass Python syntax/source checks and Git
whitespace checks. All four pinned Sonic files reverify their original
SHA-256/Git-blob hashes. Three trailing spaces in the upstream README are
retained for source fidelity and excluded from the authored-edit whitespace
check. All five canonical submission hashes remain unchanged.
