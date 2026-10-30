# Native binding port validation

The previous turn made progress: three higher scored bests, a decoder
precision repair, fresh-thread prefill hardening and an offline serving
prototype. Rank1 is still unachieved. No live GPU job remained on resume.

The external volume is absent from `/Volumes` and `diskutil list external
physical` reports no external device. Continued from a new private-GitHub
checkout at `/Users/soorya/muse_work/continuation_20261030`, starting at
`4ff92f72b40ffe8cbf7a127b8bf4d6e5263217c6`. Uncommitted external-drive
files are unavailable and have not been replaced. The pinned SGLang base
was restored separately into the git-ignored serving checkout.

## What changed

Auditing effective settings found a real omission in the initial offline
port: the donor appends `MID_DEFINES += ("-DOKC_WSH0=1", "@helper0")` after
its main function definitions. The initial generator copied the earlier
constant assignment and only the MAIN CUDA source. Its earlier AST checks
therefore did not prove that the final variant mapping was preserved.

The revised generator ports both native sources and retains this late
augmentation. The verifier now compares final settings as well as all17
packing/setup helper ASTs, dispatch and both device-source sections. Device
changes remain whitespace-only operand-list wraps; PDL stays disabled.
The source-only serving patch is now three files and310171bytes, below
720KiB,with no added line over400characters. No model call site is changed.

`submission_v27_loadjit_validation.py` embeds these exact two headers and
uses SGLang's `load_jit`, with explicit TensorView arguments and the FFI CUDA
stream. It writes only its own compilation sources into a temporary
directory. Packing, variant selection, fresh inputs and local output
semantics are preserved. The existing context binder is retained.

## Measurement gate

This is compatibility/numerical validation, **not an algorithmic speedup
claim**. A GPU binding port must compile and pass the full MoE checks before
it can support serving work. The canonical scored kernel remains unchanged.

- First job: `88b96eec59f34b57`, compile failure on sfo3. The global host
  binding used unqualified bf16_t/fp32_t; pinned SGLang declares these in
  namespace sglang. No tests/timings ran. Full compiler output was read and
  a failed-run receipt archived; the archiver now supports failed jobs
  without profiler JSON when an immutable --source is supplied.
- Source SHA256: `639a5f5a4e86872274ba7f545772538723ad3ae7950d330eedf3c3ad0f54b002`.
- Retry: v28,job `81468b81190440ed`,passed5tests/6timings on sfo3. Both aliases are
  qualified; generated trailing whitespace also removed. Device algorithms,
  packing and final variant mapping unchanged.
- Retry SHA256: `41458546626ef0243a9378d40f86670bee9e35f73bb5548831c09edbb160253f`.
- v28:2.369599741x,5675.431us,CI2.334056–2.405144,prepare50.70s,copies1.
  Full six-profile report,empty local log and remote profiler log read;
  terminal receipt archived. Point score differs−0.114%from canonical,
  inside uncertainty. No performance-gain claim.
- v29:job `11cd023842394fde`,passed5tests/6timings on sfo3 at2.384809928x,
  5639.233us,CI2.349038–2.420582,prepare50.30s,copies1. Exactv28 timed prefix.
  Prepare-only independent probes passed24 shard-offset cases atM1/16/33
  (all eight ep8 offsets,maxrelerr1.416891e-5),remote-only exactzero output,
  real state deepcopy and a fresh thread with initially unbound CUDA
  context. Two graph replays after fresh activation changes had relerr0.0.
  Fixtures/graph discarded before return. All six profiles and both logs
  read;terminal receipt archived.
- Two passing check scores differ0.642%,with overlapping intervals and
  different prepare fixtures/allocator layout. No causal speedup claim.
  Canonical/scored source unchanged;no compatibility-only scored submit.
- Routine MoE budget after this cycle:6/8checks,1/2submits on Oct30UTC.

## Additional source-analysis negative

A generic proposal to overlap unit boundaries was checked against the actual
WGS loop before allocating another run. The current source already alternates
`af[ci & 1]`,uses`wg_wait<1>()`,and flushes deferred prior-segment work in the
first stage of the next segment at`OKC_WGSFCI`. Merely adding a second operand
buffer or moving that release later repeats existing machinery. No new
experiment was justified. A genuinely different accumulator-bank design
would need an exposed-time bound,register-budget evidence and an independent
GPU prototype;this source inspection proves no new speedup.

The serving norm/scale, return-contract, sharding and full-model resident
memory gates in `../serving/MOE_PORT_STATUS.md` still apply. A kernel-track
check cannot establish eight-GPU model agreement or serving performance.

## Public-source follow-up

At resume, latest commits of alygg77, Madhavan113, vjeux, NathenFernandes,
chenwainuo and lee101 match the already reviewed revisions. Exact quoted
GitHub repository searches found no newly pushed Oct30 standalone decoder
source. This is a bounded search, not proof of absence of unpublished code.

Closed an older pending source review:
[xharshvardhan/open-kernel-challenge](https://github.com/xharshvardhan/open-kernel-challenge/tree/4958d3141aa977c4b6b26df7a8d06c90dd094ee9)
publishes a1.3749xMoE with fused local alignment and hot/cold Marlin tile
buckets, while retaining both Marlin GEMMs and post-expert finalization.
Our persistent parent already fuses routing, both phases and finalization;
no additional winning component was established. No code was copied.

[sid19991/open-kernel-challenge](https://github.com/sid19991/open-kernel-challenge)
is an official-material/starter mirror, not a new optimized MoE candidate.
[chinesepowered/hack-argonspeed](https://github.com/chinesepowered/hack-argonspeed)
reports lower historical scores and explicitly does not publish its kernels.
Nibague's README focuses on BF16; no MoE candidate adopted from that read.
