# Sampled unit and flush profiling on KPERM3

No production optimization has been promoted in this cycle. The canonical
`submission.py` is still SHA-256
`b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281`,
scored as `ce241acf7ab84c04` at 2.372309331x. All new instrumentation runs
only on independently generated synthetic inputs in `prepare()`; timed
`kernel_fn` remains the incumbent. Diagnostic point scores are not gains.

## V29: detailed unit records fail timing attribution

Check `e038ff14796944dc`, cluster `sfo3`, passed five correctness tests and
six timing cells: 2.374572037x, 5663.547 us, CI 2.338953456–2.410190617.
Preparation took 229.001 seconds. Immutable source:
`submission_v29_unit_trace_probe.py`, SHA-256
`173176c5b14b36a694e8c73f6009561c711b105729eb86eb066fc5872a81394e`.

The complete 90.600-second independent diagnostic retained 24 groups:
M32/128/256, two input seeds, four retained scratch pads, five rotated
cold-L2 rounds. Four arms compare REG112 baseline A0, detailed trace,
baseline A1, and the unchanged canonical allocation. A0/A1 maximum median
offset was 0.84086%; initial and fresh-input agreement errors were zero.
Two full replay traces contain 40 records at M32 and 52 at M256, from
WG0 at CTAs 0/33/66/99, seed3101, pad0. There is no full M128 trace.

|M|Baseline / trace range|Canonical / REG112 baseline range|
|---|---:|---:|
|32|0.892692–0.902949|0.985383–0.999094|
|128|0.876451–0.897917|0.979045–0.997968|
|256|0.865119–0.881463|0.984922–0.999335|

The trace is approximately 11–16% longer. It fails the ±5% perturbation
gate in every group. One control group also misses the ±2% canonical gate.
Full H200 attributes show 8 local bytes in both BIG instantiations, versus
zero for the matched control. Reduced fixed-NTT8 compilation had reported
zero stack/spills after REG112 redistribution: that result did not
generalize to the complete kernel. No timing attribution is accepted.

`UNIT_TRACE_e038ff14796944dc.json` retains every integer timing round,
exact uint64 clock, cut, roster, raw record and derived interval. All six
profiles and the complete available logs were read. Record order, prior
pending-kind chaining, segment bounds, unit-to-roster mapping and the
explicit readback bit validate. These checks establish capture consistency,
not unperturbed kernel-operation costs.

In the perturbed trace, median phase-A whole-publication intervals are
0.640/0.672 us at M32/256. M256's three sampled phase-A readbacks take
2.656–3.392 us. Those values must not be used as exact incumbent latencies
or added up as a causal removable critical-path bound. Quarter-position
CTAs start at whole-unit boundaries for this M32 fixture; its phase-A
readback class is absent from the sample.

## Compiler and telemetry work before V30

The original V29 instrumentation at the 96-register consumer budget spilled
48 frame bytes / 100 spill-store bytes / 284 spill-load bytes in a reduced
NTT8 compilation. It was not sent to H200. REG112 brought that reduced
compile to zero, with a zero-spill recording-disabled control. Both
responses are archived, including the failed original configuration.

V30 removes all per-segment state and clocks only flush entry/exit, using
the pending unit index to address a bounded buffer. It samples adjacent
CTA pairs 0/1 and 66/67 so a whole-boundary sample does not systematically
exclude its neighbor. It records WG0 only. Kind and stage-count labels
are inferred from exported cuts/roster; phase-A partial readback is no
longer independently observed. No inference is described as a sampled tag.

Before queueing, three full-dispatch Compiler Explorer requests failed
inside CE with `RangeError: Invalid string length`. Neither the documented
`skipAsm` option nor a host-assembly filter recovered a result. The failed
requests/responses remain archived; full-dispatch compiler status is
unknown, not a successful compile. The API option comes from CE's
[official API documentation](https://github.com/compiler-explorer/compiler-explorer/blob/main/docs/API.md);
its [NVCC adapter](https://github.com/compiler-explorer/compiler-explorer/blob/main/lib/compilers/nvcc.ts)
also extracts PTX/SASS device views separately. This is a tooling limit,
not evidence of a CUDA syntax error or a kernel improvement.

At the original 96-register budget, the first metadata-bearing V30 build
has zero stack/spills for reduced NTT1 but 16 frame bytes / 20 spill-store
bytes / 100 spill-load bytes at NTT8. Keeping only two timestamps still
has 16 / 20 / 88 bytes at NTT8. Neither was sent to H200. The final REG112
version reports zero frame/spills at reduced NTT8. All reduced SASS views
are clipped at 5000 lines; they do not establish whole-kernel instruction
counts or equivalence. NVIDIA documents `ptxas -v` resource reporting in
the [CUDA compiler driver guide](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-compiler-driver-nvcc/index.html).

The source builders, CPU-only codecs, independent decoders and compressed
CE request/response archives are committed with this report. CPU stress
roundtrips include 24 four-arm timing groups, two traces with 96 frames
each, exact uint64 clocks, zero-width columns and uint32 boundary values.
Final V30 stress payload is 8340 characters, 9940 including a 1600-character
tail allowance, below the service's 16,000-character limit. This tests
telemetry fidelity and retention; it is not GPU performance evidence.

## V30 status

Check `b22622ebb61946df` is queued/running. Immutable source
`submission_v30_flush_trace_probe.py`, SHA-256
`957edc0e485fc5878431a057eb081e6799485a6c5362035883d7d3b8713f91d0`.
This is the fourth of eight reserved MoE checks on Oct31 UTC; zero submits.
The four-arm controls, 24 timing groups and two full traces are retained.
Full native resources and numerical/perturbation/canonical gates are
pending. Do not start another job to replace this observation.

If the complete kernel still spills or the perturbation gate fails,
close this profiling configuration. Do not compensate by treating the
perturbed clocks as parent measurements or relaxing the gates afterward.
A production change still needs a distinct mechanism, a practical model
for at least 3% board gain, fresh/nearby/full correctness checks and an
independent paired repeat before promotion.

## Additional compiler technique screened

CUDA13's `.pragma enable_smem_spilling` is not a drop-in repair here.
NVIDIA explicitly excludes both dynamic shared-memory kernels and dynamic
register redistribution from this feature's supported use cases. This
kernel uses dynamic shared memory and `setmaxnreg`. Its 231168-byte main
shared allocation also leaves only 1280 bytes below the existing 227-KiB
block cap. A static-memory/register-architecture redesign is unimplemented;
no GPU check or speedup is claimed. Keep this idea separate from the
already zero-local production kernel. Source: NVIDIA's
[shared-memory spilling article, limitations section](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/).
