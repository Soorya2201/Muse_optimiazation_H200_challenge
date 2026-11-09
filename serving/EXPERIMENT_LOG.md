# Serving experiment ledger

## Nov9: initial implementation and source review

- User replaced the kernel objective with serving and revoked earlier AGENTS
  instructions. Goal API replacement failed because the previous goal is
  unfinished. Recorded OBJECTIVE.md; did not falsely complete the old goal.
- Fresh official init succeeds at297e54fd,400KB cap, and announces Nov12opening.
  Old a55ab7d5prototype/rules are retained as historical work. Serving status
  has no priority and jobs is empty. No check/submit POST attempted.
- Web direct rules/update fetches return403. Search cache shows Nov10opening;
  date discrepancy retained. Public discovery expanded to155public repo hits,
  with six selected exact trees and bounded reads, not an exhaustive audit.
- Candidate01 applies the licensed global-route-density fix to the new path.
  Its public H100 layer measurements are external only. CPU65,540policy cases
  pass, including unchanged EP1behavior; arithmetic/GPU/model quality untested.
- Candidate02 combines maximum/value token metadata into one collective.
  The initial patch attempt used an obsolete import anchor and failed atomically;
  re-read the new baseline and corrected the anchor before any test/run.
- CPU preflight initially attempted a zero-sized sampler batch. Stock itself
  rejects its ambiguous reshape. Confirmed the candidate has the same error,
  and retained72matching rejection cases rather than silently dropping the edge.
  All360nonempty sampler cases and786,432transport/selection cases pass.
- Isolated-index application, source restrictions and Python AST checks pass
  for both isolated candidates and their7,065-byte combined patch. No Triton
  compiler, NCCL, CUDA graph or serving-performance result is claimed locally.
- A real multi-GPU component script is prepared and unrun. It uses PyTorch NCCL,
  not the complete SGLang coordinator. Full official validation remains required.

Every future official run must record UTC time, candidate SHA, mode, handle,
service budget, terminal result, quality/TTFT/per-load data and available logs.
Poll an existing handle rather than restarting after an observation timeout.
Preserve invalid/truncated evidence; no success or throughput claim from it.
