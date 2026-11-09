# Serving checkpoint — Nov9 2026

The project objective is now [winning the serving challenge](OBJECTIVE.md).
There is no official serving run, score or verified rank for this work yet.

Fresh `okc serve init` pins **297e54fd105e974cdf401e5b4d20253a22c16f90**.
Its rules allow400KB of source diff under `python/sglang/`, with no added line
over400characters, and3checks/1scored run per day. The initializer says
**Nov12,9am Pacific** for opening. The cached Nov3 web announcement says Nov10;
direct page fetches returned403. Recheck the live service before any run. The
init notice and current RULES hash are in INITIALIZATION_2026-11-09.json.

The new flags include draft attention `fa4`, CPU multimodal feature transport
and160GB HiCache. The original a55ab7d5/720KB serving plan is historical.
The new working checkout is ignored at `serving/current/sglang`; the source
diffs and downloaded current rules are committed and reproducible.

## Implemented candidates

| Candidate | Mechanism | State |
|---|---|---|
| [01_ep_route_density.diff](candidates/01_ep_route_density.diff) | Correct Marlin's expected expert occupancy using global expert count for the global token batch. Adapted from hariappi's licensed public fix to the new module path. | CPU policy checks pass; no own GPU/serving result |
| [02_draft_single_collective.diff](candidates/02_draft_single_collective.diff) | Transport the draft head's maximum value and full int64 token ID in one all-gather; fuse global selection. Preserve stock first-rank ties/NaNs and unchanged dense GEMM. | CPU protocol/class tests pass; Triton/NCCL/graphs pending |
| [03_combined.diff](candidates/03_combined.diff) | Both mechanisms together | Packaging passes; evaluate only after isolated results |

The combined diff is7,065bytes across3files; longest added line86bytes. It
applies cleanly to the exact official base. No weight copies or external build
dependencies are added. The sampler intentionally sends16bytes per rank/token,
versus stock's10bytes at BF16: one fewer collective is a latency hypothesis,
not a bandwidth reduction or measured speedup.

Validation evidence:

- 65,540 exact Marlin policy cases through16,384tokens, EP1/2/4/8; EP1 unchanged.
- 786,432 CPU transport/selection columns: exhaustive FP16/BF16 encodings,
 random FP32 encodings, NaNs/infinities/signed zeros, non-power-of-two TP sizes,
 first-rank ties with nonmonotonic IDs and token IDs above2^54.
- 360 exact sampler-class CPU flow cases with fresh inputs, shrinking/growing
 batches, output pointer reuse, TP1/2/3/4/8 and FP64 fallback.72 zero-batch cases
 reproduce the same stock/candidate rejection; zero batches are not supported.
- All three patches pass source-path/mode/size/line limits, isolated-index apply
 and Python syntax checks. Reports: CPU_PREFLIGHT_2026-11-09.json and
 PATCH_MANIFEST_2026-11-09.json.

CPU tests use Torch2.15 and emulate the two small Triton operations and the
collective. They do not execute GPU code or validate the deployed image. The
developer GPU component script uses real Triton/PyTorch NCCL and graphs but is
unrun; it still cannot replace full SGLang/serving validation.

## Next execution steps

1. Recheck current rules/opening and available serving checks. With the current
   evidence, check candidate01 first; use the stock comparison supplied by the
   official run. Record the exact handle and poll it to terminal.
2. Check candidate02 independently and then candidate03 if both remain eligible.
   Reject failed quality/TTFT paths and preserve their evidence. Component wins
   need not translate into full-model wins.
3. Use the1/day scored budget for the strongest validated patch, then independently
   repeat its full-node result. Archive all results and compare the actual loads.
   A short check covers only one rung: all-load quality/TTFT/performance remain
   unproven until the full scored run returns. Do not mistake that check for
   complete validation, or make cosmetic resubmissions to fish for scores.
4. Profile the remaining bottleneck before expanding the patch. Higher-impact
   research directions: acceptance/cost-aware prefix verification with the same
   released8-position drafter; single-layout expert fusion without duplicate
   model weights; communication overlap that preserves state/stream correctness.
   These are unimplemented hypotheses, not novelty or performance claims.

No rented hardware has been ordered. An asynchronous question about existing
development hardware is pending; work assumes official runs only unless the
owner supplies an existing machine. That question does not block source work.

The old isolated MoE binding remains non-integrated. Its weight duplication,
post-norm contract and old baseline prevent direct full-model adoption. Its
kernel-track timings are not serving throughput evidence. See MOE_PORT_STATUS.md.
