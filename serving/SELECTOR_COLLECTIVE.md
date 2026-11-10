# Target the released DFlash2 selector

The public draft configuration at revision
`46c90f412549c472941a68382d91a99e5e6372a0` declares `DFlash2DraftModel`,
selector rank256/top-k16, vocabulary202048 and an8-position block. The exact
1,423-byte configuration is pinned with SHA256
`2f88d111418b4018678b8a80ed0bdf83c14b0e395cd6cc163153bce4dd8ba0ae`.
[Primary configuration](https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2/blob/46c90f412549c472941a68382d91a99e5e6372a0/config.json).

This changes the experiment order. At official SGLang297e54fd,
`DFlash2DraftModel.__init__` creates `CandidateSelector`. The worker takes the
selector branch before `_DflashDraftSampler`; its graph calls
`compute_candidates -> candidate_topk`, then builds/walks the selector lattice.
Its eager selector fallback also calls `compute_candidates`. Candidate02
changes the plain greedy sampler, so it is not the expected active path for
this published checkpoint. Earlier first-day guidance01/06/02 was incomplete.
Retain02 as a generic/fallback experiment; do not spend a first-day check on it
without evidence the deployed worker actually uses that class.

The official service's loaded revision, effective TP group, graph eligibility
and head type have not been observed. Confirm these in the first real run.
The exact builder with CPU constructor doubles chooses the selector branch
under the published configuration's inputs; this is source evidence, not a
model load. Draft-parallel resolution preserves the target group absent
attention-DP/CP narrowing, which the judge's listed flags do not enable.

## Candidate09

[09_selector_single_collective.diff](candidates/09_selector_single_collective.diff)
is isolated against the official base:4,170bytes/two source files. It packs
each local candidate's unchanged FP32 value bits and31-bit vocabulary ID
into one int64 word, then uses the existing TP all-gather. One Triton unpack
restores the exact rank-concatenated values/IDs before the unchanged
`torch.topk` and ID gather. Local radix top-k, LM-head projection, selector
weights, softcap, lattice, random draws and verification are unchanged.

At top-k16, the stock payload per candidate is4bytes for the FP32 value plus
8bytes for the ID; packed transport uses8bytes and one collective instead
of two. These are payload/call counts, not network-transaction measurements
or a latency result. Pack/unpack launches can erase the benefit. The global
vocabulary bound, dtype and platform predicates must agree across TP ranks;
there is no rank-dependent ID-range fallback. The existing model's shard
metadata must represent valid IDs within that global vocabulary.

Admission is CUDA/NVIDIA, FP16/BF16/FP32 local values, TP>1, no partition/LSE
request and a globally declared vocabulary in1..2^31. TP1, CPU, HIP, FP64,
missing/large vocabulary metadata and partition/LSE paths retain the original
code. Full-width int64 IDs remain supported there. Global top-k sees exactly
the same value-bit layout, so no new tie-breaking algorithm is introduced.

## Evidence and next gates

- [Offline Hopper compilation](TRITON_OFFLINE_COMPILATION.md) now passes14
  selector specializations in actual Triton3.8.0/ptxas, part of25total cases
  across the five added kernels. Verified CUDA cubins/IR/PTX are archived
  privately. This is compilation/assembly, not CUDA numerical execution or
  proof of compatibility with the organizer's installed image.
- Source packaging/AST checks pass in `PATCH_MANIFEST_SELECTOR_2026-11-10.json`.
- `SELECTOR_COLLECTIVE_CPU_2026-11-10.json`:196,608 protocol values, including
  every FP16/BF16 encoding and random FP32 bits; boundary IDs through2^31-1.
  2,304 exact-function comparisons pass:216 one-collective cases and2,088
  unchanged fallbacks. Includes TP1/2/3/8,0/1/7/28/112/224rows,top-k1/16,
  exceptional values, partition/LSE and IDs above2^54 on the original fallback.
  Projection, collective results, transport kernels and CUDA admission are CPU
  doubles; local top-k is Torch, not FlashInfer. This does not compile/execute
  Triton or prove CUDA capture.
- The first builder fixture missed a logger global and failed with NameError.
  Added the logger double; candidate source did not change. Recorded in the
  reachability audit. No native failure is implied.
- `SELECTOR_NATIVE_PLAN_2026-11-10.json` passes source preparation only.
  The prepared eight-Hopper harness uses actual dense projection/radix top-k,
  candidate Triton kernels and a PyTorch NCCL adapter. It plans native bit
  checks, fresh-input graphs and paired AB/BA head-path timings with per-round
  rank maxima. It does not execute SGLang GroupCoordinator, the selector
  lattice, real checkpoint weights, full-model quality or serving traffic.

```sh
python tools/benchmark_serving_selector_collective.py --plan-only --output /tmp/selector-plan.json
torchrun --standalone --nproc-per-node=8 tools/benchmark_serving_selector_collective.py --output /tmp/selector-native.json
```

Use an existing authorized Hopper node with the official image and base
`dflash.py`. The helper and exact function variants are loaded from frozen
sources by the harness. Its default head dimensions derive from the public
draft config and equal-sized TP8 shards; they are synthetic inputs, not an
observed organizer checkpoint layout. Each rank saves its own receipt.

The next official isolated checks are01/06/09, subject to current rules,
budgets and confirming the deployed path. A check covers one short rung;
quality,220tokens, per-load TTFT and all four scored loads still need full
validation. No combined09 patch has been created. Candidate07/current patch
and08 remain frozen;07 is not a proven strongest serving composition.

Before a larger selector rewrite, also confirm FlashInfer's deterministic
radix top-k is actually available. Stock warns about a Torch fallback, but
that warning's performance estimate is not our measurement. Profile dense
head, local top-k, collectives, lattice and walk separately. Replacing the
released selector, changing its top-k/rank, or skipping its sampling semantics
would change the drafter and is outside this experiment.
