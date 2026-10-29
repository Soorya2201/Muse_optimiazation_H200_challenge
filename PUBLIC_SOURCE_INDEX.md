# Public kernel source research across all five tracks

The owner's scope includes useful public implementations from teams ranked both above and below
our entry. Review happens one track at a time. Rankings are discovery context; actual per-shape
latencies, numerical correctness, compliance, and source provenance determine adoption.

## Coverage status (2026-10-29)

| Track | Research artifacts | Implementation/testing status |
|---|---|---|
| BF16 projections | `muse_bf16_proj/PUBLIC_RESEARCH.md`, `RANKED_GITHUB_REPO_SCREEN.json`, `ABOVE_RANKED_GITHUB_AUDIT.md`, `CHENWAINUO_REPO_DEEP_DIVE.md` | Expanded metadata screen complete for1394 ranked accounts:170 above,1223 below,ourselves;973 recent snapshots reused,eight404s. v18 verified/scored1.521867x;exact Chen and v33 hybrid queued behind8/8routine-check limit until00:00UTC. Historical149-account deeper audit retained. |
| MoE | `muse_moe_w4a16/PUBLIC_RESEARCH.md`, `RANKED_GITHUB_REPO_SCREEN.json` | Completed public metadata screen of1112ranked accounts. Scored vjeux **2.324062x**. Second cycle:8-op dequant and REG112 pass but lack a reliable gain;identical-input gates reject all8-op buckets. Whole-W2 early prefetch regresses2.9-14.3%,also rejected. Cycle complete6/8checks,1/2submits. Deeper zhebrak/operatorstack/tommyyliu/lilaclilac09 source screens documented. |
| Attention | `muse_attention_decode/PUBLIC_RESEARCH.md`, `RANKED_GITHUB_REPO_SCREEN.json` | Full metadata screen of1064 ranked accounts. Scored exact source **2.603505x**. Second-cycle isolated PFL2=1 check passes16tests/12timings but regresses to2.482162x;rejected. Current budget2/8checks,1/2submits. Deeper operatorstack/tommyyliu source sections documented;next priority is measured merge/scheduling cost. |
| Whole decoder | `muse_decode_layer/PUBLIC_RESEARCH.md`, `RANKED_GITHUB_REPO_SCREEN.json` | Metadata screen complete for980 ranked accounts:807 above,172 below,ourselves. Vjeux check2.089048x,Madhavan check2.119204x. Exact Madhavan scored d857144e59784ad0 **2.127754x**,rank#95/984,overall#95/1607. |
| Prefill | `muse_prefill_layer/PUBLIC_RESEARCH.md`, `RANKED_GITHUB_REPO_SCREEN.json` | Full metadata screen of644 ranked accounts:538 above,105 below,ourselves;four404s. Exact vjeux scoredf62a284779b4447d **1.784958x**,rank#102/644. New lee1018-op port passes34848offline cases and5tests/8timings at1.801190x;not promoted. Identical-route paired gates show only0.06–1.58% expert gains,below2%;v19 retains parent everywhere. Cycle complete at3checks/1submit. |

This is a coverage record, not a claim that every repository or private submission was examined.
The account screen paginates publicly owned repositories and matches name/description/topic
signals. Matches still require source review; private, renamed, or unlabelled source may be missed.
Global code search supplements the profile screen, including accounts absent from the board.

## Source families found so far

- https://github.com/vjeux/okc-2026 — Apache-2.0, complete single-file native kernels for all five tracks.
- https://github.com/Madhavan113/kerneloptimizer — Apache-2.0 derivative with per-track extensions and profiles.
- https://github.com/lee101/openkernels — Apache-2.0, many intermediate versions and diagnostic evidence. Its weaker whole scores do not rule out useful individual mechanisms.
- https://github.com/chenwainuo/okc — Apache-2.0, detailed BF16 source/history and broader experimental work.
- https://github.com/NathenFernandes/muse-bf16-proj — MIT, exact BF16 v18 already reproduced by us.
- https://github.com/operatorstack/open-kernel-challenge-solutions — MoE6d9b776 source sections/1.832036x receipt and attention producer/consumer/plan sections/1.822823x receipt reviewed. Retained source concepts already overlap our stronger parents; no additional measured winner yet.
- https://github.com/lilaclilac09/open-kernel-challenge — MoEd9bdcc9 actual source is plain-PyTorch starter;sketch notes are a placeholder. No optimized MoE source found in this inspection.
- https://github.com/PostTrainLLM/posttrainllm — experiments/openkernel sources and provenance; published MoE score below ours, useful mechanism inventory.
- https://github.com/Ranoobaba/okc-campaign — indexed challenge material; source relevance still needs verification.
- https://github.com/tommyyliu/okc — MoEf134725 GEMM/packing/dispatch and attention split/merge/epilogue/plan sections reviewed. Lossless packing,register prefetch,head packing and hierarchical merges overlap our stronger parents. Later board snapshots are not asserted identical to the inspected source. No code copied.
- https://github.com/xharshvardhan/open-kernel-challenge — indexed source; review pending.
- https://github.com/theluckystrike/muse-spark-open-kernels — code search hit, current tree only exposed README; no verified submission yet.
- https://github.com/ruwwww/open-kernel-challenge — current README reports attention 2.6032x, MoE 2.3370x, decoder 2.0751x, and prefill 1.7950x with job IDs. Strong reproduction candidates for the remaining tracks after source/precision review; README mechanism descriptions alone are not proof.
- https://github.com/Aravind-11/muse-moe-w4a16 — historical 1.629x early kernels; README explicitly says current entry is not published.
- https://github.com/infraax/okc-kernel-challenge — lower-ranked archived campaign with measured failures; producer-ring and PDL counterevidence is useful, but no stronger full MoE source than the reproduced parent.
- https://github.com/titoatwork/qgemm-mx — account below us; FP4/Hopper measurement and packing study. README explicitly says performant Hopper WGMMA/TMA kernel is not implemented; use encoding/roofline evidence without claiming speedups.
- https://github.com/zhebrak/muse-spark-kernels — below us on prefill with no published prefill winner, but Apache-2.0 sources/receipts for four other tracks. Describes persistent stream-K expert readiness counters and cluster residual boundaries; possible future component controls, not verified improvements over our promoted sources.
- https://github.com/MadanChaollaPark/open-kernel-experiments — near our old prefill rank; stopped campaign with exact scored sources and627 candidate snapshots. Preserves correct-but-slower PRMT dequant and unfinished Hopper mixed-input proof without a speed claim; useful negative evidence.

## Adoption procedure

Pin the commit, source hash, and license; inspect the timed path and precision; establish an
unchanged-parent OKC result; compare all relevant shapes; port only justified mechanisms;
check the resulting source on current inputs; submit only after the project's improvement gate.
Preserve losing runs and partial/invalid evidence with their actual status. All judge invocations
go into root RESULTS.md. No unmeasured gain or paper's unrelated benchmark becomes a claimed score.
