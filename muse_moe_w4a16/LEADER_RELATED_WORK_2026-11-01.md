# Related public work from the MoE leader

The saved Nov1 19:03 UTC board identifies `dolanzhao` as MoE leader at
2.7642x versus our 2.3723x. This is the matching public GitHub account,
not independent proof of identity or ownership of the winning source.
The fresh owner inventory still contains 15 public repositories. Seven
technical repositories were selected for deeper inspection. Nineteen files,
99,967 bytes, were fetched at exact commits and Git-blob/SHA-256 verified.
The adjacent `LEADER_RELATED_METADATA_2026-11-01.json` records every URL,
hash, read scope and limitation. Downloaded source stayed in temporary
research storage; nothing was imported, executed or ported.

| Public source | Read scope and finding | Application here |
|---|---|---|
| [fmwork at 5a005f09](https://github.com/dolanzhao/fmwork/tree/5a005f09d8c2a546b3dd9394535ea02b76821434) | Complete README, driver and timing utilities. Calls vLLM generation, synchronizes, and reports end-to-end latency/throughput. | A benchmarking wrapper, with no custom MoE kernel in its five-file default tree. Its timing protocol is not our paired cold-L2 graph protocol. |
| [ezconv at cd5e5fe4](https://github.com/dolanzhao/ezconv/tree/cd5e5fe41962b40877465e44657eb41473300d95) | Complete main/default reader, selected profile/metadata parsing. Converts HPCToolkit metric, profile and calling-context databases to an EasyView file. | Despite its name, the inspected code is a profile converter, not a convolution kernel. No NVFP4 arithmetic or expert schedule to adopt. |
| [HPCToolkit authored change 3ea475dd](https://github.com/dolanzhao/hpctoolkit/commit/3ea475dd3f84b557a41f4a0b0e5ebaa0c0931989) | Complete five-file displayed patch. Adds explicit CUDA/CUPTI library paths and fallback resolution. | Helps a profiler locate libraries. It does not accelerate our device computation. No runtime/library patch is introduced in a challenge submission. |
| [XLA at dff319a8](https://github.com/dolanzhao/xla/tree/dff319a8505125bb9b2bd7bd44729527fb4a349e) | README and default-branch comparison metadata. | Zero commits ahead of its immediate parent, mark14wu/xla, at the captured comparison. This is not a full OpenXLA audit or an audit of other branches. No leader-specific optimization found in this screen. |
| [DrCCTProf_SC20 at 31fb6751](https://github.com/dolanzhao/DrCCTProf_SC20/tree/31fb67512bfb8951bfe8f535b3cf12e2bdb366f7) | Complete README and three selected instrumentation clients. They initialize DynamoRIO calling-context/memory instrumentation. | Profiling infrastructure; the inspected clients do not implement the required GPU expert operation. Core library and remaining clients not fully audited. |
| [dead_code at bf97d166](https://github.com/dolanzhao/dead_code/tree/bf97d1668bf081ed2a7d778a050418654c9b35bf) | Complete main and analysis implementation. The latter counts read/write memory references in a trace. | The inspected implementation does not contain a GPU dead-code optimization or a faster MoE kernel. |
| [easyview-artifact at 3a9e3298](https://github.com/dolanzhao/easyview-artifact/tree/3a9e329831e6bafbb9e3fc80dfb60496ff9db696) | Default tree and minimal README. | Large profile/source archives and binary extension not inspected. No winning-source association established. |

The fmwork default comparison is zero commits ahead/98 behind its immediate
parent WarningRan/fmwork. HPCToolkit is two ahead/567 behind its immediate
parent, including the loader-path commit above and a merge. Its comparison
file list is empty, so the authored commit itself was fetched and read;
an empty comparison list was not treated as proof of no authored changes.
These are snapshots of the specified comparisons, not all-branch findings.

Web searches for the exact handle with kernel, GPU, profiling, blog and MoE
terms found community-board references but no verified leader-authored GPU
optimization article or winning submission. This bounded search does not
prove no such public material exists. The other leaders' coverage and gaps
remain in `research/LEADER_PUBLIC_COVERAGE_2026-11-01.md`.

## What this changes in the optimization plan

Profiling tools can improve diagnosis, but they are not evidence of a
particular device schedule or of the leader's winning technique. No source
in this screen supplies an additive replacement for our current persistent
NVFP4 path. Generic claims about fusion, caching or occupancy are not gains.

The independently designed two-resident-CTA experiment remains the next
test: halve the per-block output unit and shared-memory budget, keep four
consumer warpgroups per SM across two blocks, and give them independent
producer/consumer pipelines. Original packed weight pointers, NVFP4/BF16
precision and scratch offsets are preserved. Its compiler checks pass;
actual H200 residency, concurrency, numerics and performance are unknown.
See `TWO_CTA_RESEARCH_2026-11-01.md` for the fixed protocol and failure log.

The saved leader gap requires about 16.5% more whole-problem throughput,
equivalently about 14.2% less time if every load improved uniformly. If only
M32/128/256 improved equally and the other three loads stayed fixed, those
three would need roughly 26.3% less time to close the same gap. This is
planning arithmetic, not a promise that the experiment can achieve it.
The 3% adoption gate is an intermediate gain requirement, not a rank1 claim.

Before any doubled-grid launch, the actual H200 resource query must confirm
two resident new blocks and zero local allocation in both BIG modes. Nearby,
empty-route, fresh-input and FP64 checks must pass before performance can
qualify. The extraction tool now also rejects missing/duplicate evidence and
checks empty-route success explicitly. Twenty-two CPU synthetic parser cases
pass, including positive/negative controls; these are not GPU receipts.

## Service availability

Source/protocol were pushed in e323392 before any GPU attempt. At20:03:58UTC
the read-only OKC jobs command announced maintenance while moving GPU nodes.
Subsequent read-only jobs/problems requests returned API530. No check POST
was attempted, no GPU handle exists, and no run or kernel failure is claimed.
Once the service resumes, read the current budget and run immutable V41
first. The canonical submissions, scores and budgets are unchanged by this
offline work. GitHub repository visibility was freshly verified PRIVATE.
The final read-only budget retry at20:12:10UTC still returned530; its exact
stdout/stderr/status are preserved in
`research/service/OKC_AVAILABILITY_2026-11-01.json`.
