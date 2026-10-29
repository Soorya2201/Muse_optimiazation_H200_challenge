# Public kernel source research across all five tracks

The owner's scope includes useful public implementations from teams ranked both above and below
our entry. Review happens one track at a time. Rankings are discovery context; actual per-shape
latencies, numerical correctness, compliance, and source provenance determine adoption.

## Coverage status (2026-10-29)

| Track | Research artifacts | Implementation/testing status |
|---|---|---|
| BF16 projections | `muse_bf16_proj/ABOVE_RANKED_GITHUB_AUDIT.md`, `CHENWAINUO_REPO_DEEP_DIVE.md`, `RESEARCH_LIBRARY.md` | v18 is verified/scored; exact Chen and v33 hybrid queued behind the exhausted daily routine budget. Prior account screen covered 149 above-ranked profiles at that snapshot. Below-ranked public sources are in the research library. |
| MoE | `muse_moe_w4a16/PUBLIC_RESEARCH.md`, `RANKED_GITHUB_REPO_SCREEN.json` | Completed public metadata screen of all 1112 ranked accounts: 625 above, 486 below, and ourselves; five 404 accounts. vjeux check2.339526x, Madhavan check2.330435x. Scored vjeux confirmation9835951891204c56 is **2.324062x**, rank#146/1112, overall#447/1599. |
| Attention | `muse_attention_decode/PUBLIC_RESEARCH.md`, `RANKED_GITHUB_REPO_SCREEN.json` | Full metadata screen of1064 ranked accounts:877 above,186 below,ourselves;941 recent repo snapshots reused with timestamps. Exact source check2.597212x, scored confirmation30c669b9702d434f **2.603505x**,rank#90/1066,overall#354/1602. |
| Whole decoder | Existing track `IMPROVISATIONS.md` and profile history | Full above/below public-source audit remains outstanding; depends on standalone component evidence. |
| Prefill | Existing track `IMPROVISATIONS.md` and profile history | Full above/below public-source audit remains outstanding; enter in project order. |

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
- https://github.com/operatorstack/open-kernel-challenge-solutions — additional public kernels; source review and receipt verification needed.
- https://github.com/lilaclilac09/open-kernel-challenge — indexed source; review pending for active MoE scope.
- https://github.com/PostTrainLLM/posttrainllm — experiments/openkernel sources and provenance; published MoE score below ours, useful mechanism inventory.
- https://github.com/Ranoobaba/okc-campaign — indexed challenge material; source relevance still needs verification.
- https://github.com/tommyyliu/okc — indexed source; review pending.
- https://github.com/xharshvardhan/open-kernel-challenge — indexed source; review pending.
- https://github.com/theluckystrike/muse-spark-open-kernels — code search hit, current tree only exposed README; no verified submission yet.
- https://github.com/ruwwww/open-kernel-challenge — current README reports attention 2.6032x, MoE 2.3370x, decoder 2.0751x, and prefill 1.7950x with job IDs. Strong reproduction candidates for the remaining tracks after source/precision review; README mechanism descriptions alone are not proof.
- https://github.com/Aravind-11/muse-moe-w4a16 — historical 1.629x early kernels; README explicitly says current entry is not published.
- https://github.com/infraax/okc-kernel-challenge — lower-ranked archived campaign with measured failures; producer-ring and PDL counterevidence is useful, but no stronger full MoE source than the reproduced parent.
- https://github.com/titoatwork/qgemm-mx — account below us; FP4/Hopper measurement and packing study. README explicitly says performant Hopper WGMMA/TMA kernel is not implemented; use encoding/roofline evidence without claiming speedups.

## Adoption procedure

Pin the commit, source hash, and license; inspect the timed path and precision; establish an
unchanged-parent OKC result; compare all relevant shapes; port only justified mechanisms;
check the resulting source on current inputs; submit only after the project's improvement gate.
Preserve losing runs and partial/invalid evidence with their actual status. All judge invocations
go into root RESULTS.md. No unmeasured gain or paper's unrelated benchmark becomes a claimed score.
