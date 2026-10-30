# Fresh-thread context hardening

Candidate `submission_v20_context_bind.py` preserves the exact vjeux f35a41e
parent as its byte-identical prefix, including every native CUDA/C++ string.
SHA-256: `8b52dedda281b2e642bda1d9333bcf2d002fc8611e25caa10e90ed2a849021a5`.
Builder: `tools/build_prefill_context_candidate.py`.

The parent calls a raw cuBLASLt handle from its native wrapper. Fresh host
threads may lack a current CUDA context. The new outer prepare/kernel wrappers
bind the tensor's device primary context before calling the unchanged parent.
The context decision is per-thread/per-device metadata only; it does not read
input tensor values or cache activations/results. This is robustness work,
not a performance optimization or proof the original scored entry is invalid.

The helper follows Matsvei (kernarc user matvey) through the Apache-2.0 source
published by alygg77 at full commit6cae49a639ee6d14b929a5fd4a63c627f96d7dc0.
Names were isolated and tensor field names adapted to this track. Notices and
license are preserved in research/public/alygg77_6cae49a/.

Prepare additionally builds one independent synthetic GEMM fixture for an
already-selected Lt plan. A new host thread checks its initial driver context,
binds the correct device, executes that raw Lt call, and joins before prepare
returns. Output is compared to torch.mm. Probe inputs/output are discarded;
no challenge data or reference module is used. If no Lt plan is selected,
the probe validates driver binding only and reports that narrower coverage.
This does not test every possible concurrent-thread/context scenario.

GPU check `a77a2557e1a24fa7` is running on sfo3. Syntax, prefix/native isolation
and pinned-source checks pass locally. Promote only after the actual probe
and all challenge tests pass; do not interpret tiny timing shifts as a GPU gain.
