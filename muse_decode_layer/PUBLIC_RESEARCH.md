# Whole-decoder public source audit

## Coverage and current scores

`RANKED_GITHUB_REPO_SCREEN.json` covers all980 ranked accounts at capture:807 above
our then-rank#808,172 below,and ourselves.928 recent repository metadata snapshots were
reused from the MoE audit, with timestamps. Four public accounts returned404:
accountforgoofystuff,Pinoeer-kingxi,cyber199306,gurrapu21. Metadata is a discovery screen;
private/unlabelled source can be missed and matching repositories still need code review.

Our prior scored incumbent is1.232602x,18351.662us,job`bbb6e0110f24457f`. The local v12
check-only candidate reached1.2505x. Public live context: vjeux2.1019x/#131,
Madhavan1132.1388x/#87,ruwwww2.0751x/#164,lee1011.7883x/#314. Current source hashes
are not automatically tied to those receipts; our unchanged reproductions establish evidence.

## First exact control

- Source: https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_decode_layer.py
- Local file: `submission_v15_vjeux_f35a41e.py`.
- SHA-256: `cf4a75ee6e4a5cd9e33b82b5347c47abcdc7b81d80dc6a8b5262dea7388f849c`.
- License: Apache-2.0, preserved in `upstream_vjeux_APACHE_2.0.txt`.
- Check: `aabb75632b31456b`, **2.089048x**,10828.043us,CI[2.057712,2.120384],all13
  tests and12timed cases pass,prepare101.64s. First of eight routine checks today.

The file composes its own embedded C12 projection family, fused paged attention, persistent
NVFP4 MoE, Triton routing/norm/residual glue, and tuned cuBLAS/Marlin fallbacks. Prepare times
backend alternatives on synthetic activations derived independently of timed data. The timed
path computes QKV, attention with current cache writes, O and first residual boundary,
pooler/router/top8/pre-expert norm, local experts, unpooler, and the second boundary; it returns
BF16 hidden,FP32 stream,and int32 expert IDs. Dynamic local_start is consumed on device.

Static review found required precision preserved: BF16 weights use lossless C12 or original
storage; router uses BF16 high+low with FP32 accumulation; NVFP4 weights retain codes/scales;
norm factors1+w are formed in FP32 at use; hidden activations remain BF16; query/cache use
the allowed FP8 formats. Context coverage, causal masks, reset safety,and final residual/routing
accuracy passed all13 judge tests and12 timed cases in the control reproduction.

The source's synthetic tuner temporarily skews its bias for realistic local-expert loads,
then restores `bias0` on the normal return path. It does not intentionally alter timed routing.
An exception before restoration is a robustness concern: inspect tune-failure logs and reject
any candidate that does not restore the original bias. No score may be credited to changed
checkpoint values or dropped expert work. The public prepare wrapper catches tuner errors,
so a future maintained derivative should use an independent bias clone and finally restoration.

The import of os only manages standard compilation/debug settings; Python threads build native
extensions. No timed-path host tensor reads, input-result caching, network, harness import,
or skipped attention values were found in the reviewed call chain. Guarded scratch buffers
are checked after prepare-time tests. A passing judge remains necessary.

## Stronger derivative to inspect next

https://github.com/Madhavan113/kerneloptimizer/blob/c180924742f5ec30ee7434b1d99cf2c16d2aa777/muse_decode_layer/submission.py

This is an Apache-2.0 derivative,960371 bytes,pinned locally as
`submission_v16_madhavan_c180924.py`,SHA-256
`9ff17cb7b57136f8fbff63d6c7c14cdc193936c2f8986174cd7e60d847e56353`. Python syntax and
byte identity pass; the derivative notice is preserved. It embeds the newer attention/MoE modules,
adds CUDA boundary/routing/norm fusion, PDL through128 tokens, stable large-T dense choices,
K-major cuBLAS alternatives,and in-chain pooler/router form selection. Its newer standalone
MoE branch did not beat our vjeux MoE aggregate, so the whole-layer integration must be measured
on its own rather than assuming feature gains add.

Check `9f256608c2df4cc8` passed all13 tests and12timings at **2.119204x**,10673.962us,
CI[2.087416,2.150992],prepare189.92s. No tuner error occurred; original bias was restored.
CUDA glue self-tests pass, but route tails,boundary tails,boundary prologues,and in-MoE routing
report unavailable. The actual tuner selects `agent` MoE for every token count,`gluev2=False`,
and no post-boundary megakernel. Do not credit inactive options with this result. Profiles show
the CUDA route/boundary kernels replacing Triton glue. The1.44% aggregate advantage over the
vjeux control is inside overlapping intervals; it is the stronger measured full candidate,
not proof that an individual added feature wins. Both clearly beat the prior incumbent.
Exact v16 is promoted to `submission.py`; scored reproduction `d857144e59784ad0` is pending.
Two of eight checks and one of two submits used today.

## Why this is the high-value next architecture

Our old layer still pays multi-launch attention for many shapes, 30+small kernels at T1,
Marlin on some token counts,and older three-launch MoE at others. The new standalone source
has already reproduced2.3241x MoE and2.6035x attention in our account. The integrated decoder
control also adds compressed projections and fused boundaries that our layer lacks. These
are substantive changes to weight traffic,expert reuse,and launch overhead; another AP/WP or
grid-wave sweep would preserve the old bottlenecks.

Lower-ranked repositories remain useful as counterevidence. The infraax campaign documents
PDL congestion and ineffective tiny glue fusions; lee101 studies expose attention merge and
dequant chains. Their results do not prove these mechanisms transfer to the current chain.
After a passing whole-layer profile, rank further work by actual phase share and normalized
six-load impact, retaining every regression and actual selected backend.

## Remaining rank-one gap and next experiments

The2.119204x check needs approximately12.35% less normalized time to match the2.4178x
leader at capture. An optimistic phase model subtracts a fraction of each native profile
duration from its paired shape time, then recomputes each weighted load and their geometric
mean. PDL waits overlap, so these are upper-bound planning estimates, not measured gains:

| Phase duration reduction |20% reduction|30% reduction|
|---|---:|---:|
| MoE |2.3381x|2.4659x|
| Attention |2.1846x|2.2189x|
| CUDA routing/boundary glue |2.1518x|2.1685x|

Expert optimization has the largest modeled effect. Lower-ranked lee101's
[FP4 dequant survey](https://github.com/lee101/openkernels/blob/c565e373d34575c5f7df862b05f8be7619078b0f/docs/fp4_dequant_survey.md)
describes an exact8-integer-op bit placement with a corresponding lossless repack, compared
with an11-op consumer. Our embedded MoE still contains the11-op dequant family. A transfer
must change the matching weight pack and every consuming fragment together; changing only
the arithmetic would corrupt weights. This is a concrete next controlled candidate, not a
gain inferred from its author's instruction counts. Preserve BF16 activations and NVFP4
codes/scales, exhaustively verify the16 codes and paired register layouts, then judge the
whole layer. Humming-style RS-WGMMA is already present; re-adding it is not new work.

Other candidates: improve expert scheduling/reuse at T128/256; reduce T1 pooler traffic;
prefetch page metadata before the attention dependency wait. PDL is already enabled through
T128 in the measured derivative. Lee's earlier suggestion to add it globally therefore
does not constitute an unimplemented optimization here. The current cycle first confirms
the large whole-layer gain, then enters prefill before another decoder-only sweep.
