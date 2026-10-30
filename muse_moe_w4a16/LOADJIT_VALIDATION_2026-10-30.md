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
The source-only serving patch is now three files and310131bytes, below
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

- Job: `88b96eec59f34b57`, pending at creation.
- Source SHA256: `639a5f5a4e86872274ba7f545772538723ad3ae7950d330eedf3c3ad0f54b002`.
- Routine MoE budget after enqueue:4/8checks,1/2submits on Oct30UTC.
- Read every profile/log and archive terminal evidence before deciding next.

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
