# Prefill public-source audit and optimization campaign

## Oct30 controlled pipeline/exponential follow-up

See [HYBRID_PIPELINE_2026-10-30.md](HYBRID_PIPELINE_2026-10-30.md).
Combined existing FP32 cubic25%/50%FMA with selected full-attention PIPE/RRS.
Two independent seed groups with duplicate controls show16-27% longer
attention time. Follow-up resource queries show zero compiler local bytes;
two-WG controls and hybrids remain slower than the three-WG incumbent on
every timed geometry. FMA25% helps the weaker two-WG control,which does not
make it an improvement over our stronger parent. t1f two-WG arms fail the
chosen1% diagnostic parent-agreement gate and were not timed there.
Both official checks pass all5tests/8timings while timing parent dispatch.
Only unchanged class choices qualify;no promotion/score claim. The report
links primary FA3/FA4 research and retains all numeric failures/regressions.

## Oct30 continuation

Fresh metadata covers662 entrants (117 above/544 below/self at capture),
with timestamped repository reuse within1h and four unresolved accounts.
See `PUBLIC_REPO_FOLLOWUP_2026-10-30.md` and the dated JSON. New/updated
matches were checked; no newly published stronger prefill submission was found
in inspected source. Alygg77 prefill AST/native constants equal originalvjeux.

Context-hardeningv20 preserves the entire parent prefix/native GPU path.
Check `a77a2557e1a24fa7` passed1.805642x on sfo3,all5tests/8timings,
prepare53.23s. A fresh thread initially had no driver context, then bound the
device and executed Lt753router with3.17e-6 relative error. All profiles/logs
were read and archived. Promoted for robustness; scored confirmation
`25a1e0e5df0d4949` passed1.779374x,all5tests/8timings,prepare53.73s,sfo3.
Allprofiles/logs archived/read. Native GPU code is unchanged; no speed gain or
regression is established. Canonical source keeps the context binder; old
eligible1.784958x remains the board best (#118/662).

## Coverage

The2026-10-29T19:47:58UTC repository screen covers all644 ranked accounts:538 above
our then-rank#539,105 below,and ourselves.607 recent repository metadata snapshots were
reused with timestamps; rankings are fresh. Four public accounts returned404:Abarbir12,
Pinoeer-kingxi,accountforgoofystuff,gurrapu21. Raw per-account discovery evidence is in
`RANKED_GITHUB_REPO_SCREEN.json`. This metadata screen does not examine every source file;
private or unlabelled submissions can be missed.

At capture the leader is zzz0906/2.0325x; our incumbent is1.1184x/job b6fbcde2c4aa4418.
Public source teams include vjeux#60/1.7982x,Madhavan113#65/1.7960x,
ruwwww#69/1.7950x,and lee101#294/1.4785x. A repository HEAD is not automatically tied
to its team's scored receipt. Our own unchanged check establishes that association.

## Primary reproduction

- Source: https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_prefill_layer.py
- Local file:`submission_v17_vjeux_f35a41e.py`,5381lines.
- SHA-256:`9a12398e9db70abb59ee3a914e185a5e8010e2557056ece22528f6b0b8c24217`.
- Apache-2.0 license retained as `upstream_vjeux_APACHE_2.0.txt`.
- First routine check:`ce90e6fc5b4f49ed`,**1.785870x**,216927.216us,
  CI[1.759082,1.812658],prepare54.16s;all5tests and8timedcases passed.

The parent uses persistent RS-WGMMA experts with in-register NVFP4 dequantization,
gather+plan fusion,skew-aware expert work order,and separate remainder tile widths.
It replaces our large BF16 expanded-weight scratch path and Marlin at T753. QK norm,
RoPE,FP8 query/cache writes are fused; a staged custom Hopper attention kernel validates
against FA3 during prepare. Full and sliding classes use different validated variants.
cuBLASLt dense choices include a combined hi+lo BF16 router with FP32 output. Large-T
residual reconstruction recomputes the first boundary from its saved RMS factor inside
the timed call, avoiding an FP32 residual-stream round trip.

The reviewed timed chain allocates fresh activations and resets routing/work counters
on every replay. Dynamic local_start stays device-side. Norm weights remain checkpoint
BF16 with1+w formed at use. Codes,block scales,and expert global scales retain their
precision. The scale fast path only applies to exact positive-normal encodings and otherwise
builds the general decoder. Attention loops cover every causal/window-visible key.
Non-Hopper fallback attention synchronizes and is not selected on H200. No runtime network,
input-output cache,reference import,or synthetic bias calibration is used in the scored chain.
Five judge tests and all eight scored shapes passed in the control. Profiles/logs were read;
attention validation's worst relative error was0.0151 for every offered variant.

## Cross-repository candidate

Lower-scoring lee101's [exact8-op dequant layout](https://github.com/lee101/openkernels/blob/c565e373d34575c5f7df862b05f8be7619078b0f/docs/fp4_dequant_survey.md)
is transferred to `submission_v18_lee8_dequant.py`. Both `_CODE_POS` and `convert8`
change together; the four BF16 fragment registers and their scale multiplications are
unchanged. This parent already has7-ALU+2-IMAD dequant,so the incremental comparison is
approximately9 versus8 integer operations,not11 versus8. Compiler scheduling and occupancy
can erase that saving; no performance claim is made before an H200 check.

`tools/verify_prefill_dequant_layout.py` passes34848 packed-word cases: all16 codes,
all pairs of code slots against zero/all-ones backgrounds,the multiply path's256 source-bit
combinations,and20000 deterministic random words. It verifies the candidate source mapping
and device expression strings,then compares reconstructed register bits to the independent
logical fragment order. This is an encoding proof/check,not a GPU runtime test.
The derivative notice records lee101's Apache-2.0 provenance. Check48631378372b4f15
passed all5tests/8timings at **1.801190x**,215082.167us,CI[1.774172,1.828208],prepare44.58s;
source SHA7e7e0793c672bb0e044fb5777c4bcc59916a9b17447056f31821acf4c6fbaee4.
Its0.858% aggregate edge over the unchanged parent is inside overlapping intervals.
The expert profiles are mixed: T753/sliding281.7vs244.1us,T3012/sliding764.3vs731.7,
T16384/sliding3631.7vs3634.1. The baseline also differs across jobs and routing changes,
so raw phase comparisons do not isolate the conversion's effect. No strict improvement
is established;preserve v18 as research and retain exact v17 in submission.py. A future
in-prepare paired comparison must use identical synthetic routes and weights for both
layouts. The parent already moves two shifts to IMAD; the8-op alternative changes pipeline
pressure as well as instruction count. Fewer instructions alone need not reduce runtime.

Scored parent confirmation f62a284779b4447d passed5tests/8timings at **1.784958x**,
217038.147us,CI[1.758183,1.811732],prepare52.97s. New rank **#102/644** from#539.
Normalized time346.381ms->217.038ms(-37.34%);speedup score+59.59%. All profiles/logs
were read. The old local source is exactly archived as submission_v8.py.

`submission_v19_paired_dequant_gate.py` now compares both layouts on identical synthetic
routes/weights inside prepare. It first prepares the original9-op parent,then packs the
8-op candidate separately. A34848-word encoding/source audit also proves its reconstructed
control converter and packing equal the checked parent. Both expert outputs must agree
within1e-5 relative L2 on live routes. Two independent alternating-order,cold-L2 graph
timing groups must each show>2% gain before choosing8-op per token bucket. Compile/time
budget/validation exceptions retain the parent. Timed dispatch uses token geometry only;
synthetic inputs/graphs/outputs do not persist in returned state.

Check75053ef427e34b93 passed5tests/8timings at1.794011x,215942.926us,
CI[1.767100,1.820921],prepare87.45s. Live routed expert outputs match exactly(relerr0).
The two paired speedup factors were T7531.01057/1.01582,T30121.00480/1.00571,
T163841.00065/1.00220. None clears the2% gate;all buckets retain the original converter.
Thus its timed path is the parent everywhere,and the0.51% aggregate edge over the scored
parent is not an8-op gain. The new layout has small positive microbenchmark evidence,
about0.06–1.58% of the expert phase,insufficient for a strict whole-layer improvement.
It remains a candidate component to combine with a larger independent gain. Three of
eight checks and one of two submits used;this reproduction cycle is complete.

## Sources below or close to our former rank

| Repository | What was reviewed | Decision |
|---|---|---|
| https://github.com/titoatwork/qgemm-mx at f83c8511e49fa7fa7feef960fc1cc2e8b7588513 | Then#544/1.1104x;README describes cold/hot cache measurement,packing and occupancy studies; performant Hopper WGMMA/TMA kernel explicitly not implemented | Useful measurement counterevidence; no missing fast kernel to port |
| https://github.com/zhebrak/muse-spark-kernels | Then#608/1.0039x;public tree lists four winner tracks but no prefill winner | No prefill source to reproduce from that tree |
| https://github.com/MadanChaollaPark/open-kernel-experiments at40154c27d4b694c7d641c7959b00d7ff24a7c2e4 | Then#535/1.1206x,near our former rank;README ties exact grouped-TMA source to official762a6d1810654813;627 archived candidates include failures | Grouped TMA/persistent work is already superseded by this control; PRMT FP4 lookup was correct but slower; unfinished mixed-input Hopper proof has no measured speedup |
| https://github.com/lee101/openkernels atc565e373d34575c5f7df862b05f8be7619078b0f | Prefill gap analysis and FP4 conversion survey | Dequant layout is a concrete transferable mechanism; old phase/roofline estimates must be replaced with our passing profile |
| https://github.com/tommyyliu/okc | Above our former rank;public prefill submission is the10692-byte plain-PyTorch challenge starter,despite a higher team score | Public HEAD is not their stronger scored implementation;no optimized prefill mechanism to transfer from it |
| https://github.com/AniMahou/Muse | Above our former rank;README is a Bangla voice/field-intelligence application | Keyword collision;unrelated source |
| https://github.com/zzz0906/sglang | Leader's discovered repository;default HEAD9a7ac3ecefc64bf4b47c70e3446e233ac02773ea dated2026-07-23,inherited SGLang branch names | No verified challenge submission found in reviewed metadata/default branch;do not claim the leader's private implementation was recovered |

Madhavan's source atc1809247 has SHA76fb7501371fd3331858795e462ce5d0877e4b6a789a10a7c189fb7a8e5aaaa4.
It includes a broader experimental history,while the vjeux parent prunes dead device branches.
Their similar reported prefill scores do not justify blindly combining all switches.

## Next measurement decisions

The unchanged parent passed all eight shapes. Measure the isolated8-op
candidate with the same parent configuration. Promote only a passing candidate that clearly
beats the old scored incumbent; judge tiny differences between the two new variants against
their confidence intervals. Read every profile and retain regressions. Rank further work by
the weighted turn1,turn4,and first-turn phase costs,not by repository rank alone.

## Measured bottlenecks after reproduction

The persistent expert profile takes244.1us at T753/sliding,731.7us at T3012/sliding,
and3634.1us at T16384/sliding. Its share is about40%,with dense projections similarly
large at T16384. Full64k-context attention takes6764.5us/about43%. The parent already
eliminates the old materialized-weight expert scratch and uses a single expert launch.
Attention combine kernels can include PDL waits: do not add their full durations to the
producer to infer attention's wall time.

Matching the captured leader2.0325x from this1.785870x control requires12.13% less
normalized time. An optimistic model subtracts a fraction of native phase durations from
the paired shape time and recomputes the three weighted loads. Overlapping waits make
these planning upper bounds,not verified gains:

| Native phase duration cut |10%|20%|30%|
|---|---:|---:|---:|
| Persistent expert kernel |1.8532x|1.9258x|2.0044x|
| Attention forward |1.8148x|1.8448x|1.8758x|
| Dense GEMMs |1.8552x|1.9302x|2.0115x|
| Residual boundaries |1.7984x|1.8112x|1.8241x|

A30% dense reduction is not a credible easy target: these GEMMs already use tuned
cuBLAS at high BF16 throughput. The cheap8-op port is an incremental test,not an expected
rank-one solution. After it, prioritize fused expert scheduling/remainder reuse and
full-attention pipeline efficiency,with whole-load timing as the gate. Lee's older report
that remainder launches reread all experts is not directly transferable: this parent already
fuses heterogeneous tile work into one persistent expert kernel.
