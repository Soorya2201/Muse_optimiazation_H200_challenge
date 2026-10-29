# BF16 Above-Ranked GitHub Audit

Snapshot: 2026-10-29 PT. Our leaderboard position at capture was **#150/1301**, **1.5219x**, with 149 teams ahead. Leader: **colinlikescode**, **1.6367x**.

## Scope and method

Every GitHub profile corresponding to ranks 1 through 149 was queried through the GitHub API. For each resolvable account, all public repositories were paginated and repository names, descriptions, and topics were screened for challenge signals including `muse`, `open-kernel`, `okc`, `H200`, `CUDA kernel`, `kernel challenge`, and `SGLang`. Candidate trees and documentation were then inspected manually.

This is an exhaustive audit of **currently public repositories discoverable from those 149 GitHub accounts**, not proof about private, deleted, renamed, newly-created, or unindexed repositories. A leaderboard job does not expose its source, so techniques are attributed only where public source evidence exists.

## Verified findings

| Board rank | Team | Repository | Verdict |
|---:|---|---|---|
| 1 | colinlikescode | [Open-Inference](https://github.com/colinlikescode/Open-Inference) | Apache-2.0 inference-serving optimizer for vLLM/SGLang. Its tree and README do not contain the Muse BF16 challenge submission, so it does **not** reveal the rank-1 kernel. |
| 56 | Lurrobert | [guild](https://github.com/Lurrobert/guild) | Public encrypted collaboration/forum infrastructure for OKC, not a kernel submission. |
| 78 | chenwainuo | [okc](https://github.com/chenwainuo/okc) | **Verified public `muse_bf16_proj` submission**, Apache-2.0. Audited commit: [`3d418e1`](https://github.com/chenwainuo/okc/commit/3d418e1625f6de2075b5684a27ddbc3d1442422d). Repository reports 3133.1 us / 1.542x. |

No other account above us exposed a repository that could be verified as its Muse BF16 submission at capture time. The SGLang forks are useful upstream context but do not contain evidence tying them to the owners' ranked jobs.

## What the verified rank-78 implementation solved

Source: [README](https://github.com/chenwainuo/okc/blob/3d418e1625f6de2075b5684a27ddbc3d1442422d/README.md), [submission](https://github.com/chenwainuo/okc/blob/3d418e1625f6de2075b5684a27ddbc3d1442422d/muse_bf16_proj/submission.py), and the repository's handover/audit records.

1. **Weight bandwidth:** a lossless 12-bit codec stores sign plus seven mantissa bits and a four-bit exponent offset, with sparse exact-value escapes. This cuts the dominant weight stream to about 75% of BF16 bytes while preserving required accuracy.
2. **Small-m launch and memory cost:** `gv` / `gv_predxor` kernels use register-direct GEMV, packed global loads, shape-specific geometry, and predicated-XOR escape handling for `m <= 32`.
3. **Mid-size tensor-core utilization:** `rq` (“RS128”) uses swap-AB, register-A WGMMA, split-K, two math warpgroups, TMA-fed activations, a five-slot ring, a one-wave CTA grid, and bulk-reduce epilogues. It is used selectively for `m=16/32/128`.
4. **Router graph overhead:** `rtk` fuses high/low FP32 accumulation and split-K into one WGMMA node for router `m=128/256`, avoiding multi-node reduction overhead.
5. **Sparse escape overhead:** a per-thread staged cursor and warp vote skip empty escape blocks; their records attribute roughly a microsecond-scale improvement at `m=128`.
6. **Selection correctness:** preparation benchmarks paired CUDA-graph choices and dispatches by exact shape, retaining cuBLASLt for shapes where custom kernels lose. This prevents a globally attractive kernel from regressing individual routes.

The strongest measured advantage over our current v18 profile is concentrated in QKV at `m=16/32/128` and pooler/unpooler at `m=32/128`. Our incumbent is already competitive or faster on several O-projection, large-router, and tiny-unpooler cells. Therefore the best next experiment is **route-level integration**, not wholesale replacement.

## Implementation decision and queued experiments

Current local implementation work already isolates our reproducible M1 winners in `submission_v30_all_small_qkv32.py`; broad speculative routes were removed after judge evidence did not reproduce them.

The exact verified rank-78 source is now pinned locally as `submission_rank78_chen_3d418e1.py`. It is byte-identical to upstream commit `3d418e1625f6de2075b5684a27ddbc3d1442422d`: SHA-256 `bb50079be2ed74d65bb7109f1432fbc7a49bbe70913d247c254717cd0f27f712`. Its upstream Apache-2.0 license is preserved at `THIRD_PARTY_LICENSES/chenwainuo-okc-APACHE-2.0.txt`. Python syntax validation passes. It has not yet consumed an OKC check; the BF16 reserved check budget for 2026-10-29 was already exhausted before this audit.

Next BF16 sequence, after the reserved daily judge budget resets:

1. Check the exact audited rank-78 Apache-2.0 submission as a provenance-pinned control. Its public judged result is already above ours, but it must pass under our account/current evaluator before promotion.
2. Build a minimal hybrid: keep our exact v18 routes except adopt the public RS128/RS32 selections for QKV `m=16/32/128` and pooler/unpooler `m=32/128`.
3. Compare the hybrid against both exact parents, requiring an improvement in same-run aggregate evidence before submitting.
4. Only then test escape-cursor or geometry changes one at a time. Do not re-open paths the public repo itself records as closed: RS128 at `m=256`, fewer-CTA decompositions, per-warp reduction, and cluster multicast variants.

## Supplemental public BF16 source library

These repositories were found through global GitHub code search. They are useful public references, but they were **not verified as repositories belonging to teams ranked above #150 in this snapshot**:

- [lilaclilac09/open-kernel-challenge](https://github.com/lilaclilac09/open-kernel-challenge)
- [ozgureyilmaz/okc-muse-kernels](https://github.com/ozgureyilmaz/okc-muse-kernels)
- [lee101/openkernels](https://github.com/lee101/openkernels)
- [vjeux/okc-2026](https://github.com/vjeux/okc-2026)
- [PostTrainLLM/posttrainllm](https://github.com/PostTrainLLM/posttrainllm)
- [Ranoobaba/okc-campaign](https://github.com/Ranoobaba/okc-campaign)
- [ruwwww/open-kernel-challenge](https://github.com/ruwwww/open-kernel-challenge)
- [tommyyliu/okc](https://github.com/tommyyliu/okc)
- [rohanarun/Auto-research-agent](https://github.com/rohanarun/Auto-research-agent) (prompt/agent material, not verified ranked kernel source)

## Complete above-ranked account ledger

| Rank | Team/profile | Speedup | Job | Public-repo disposition |
|---:|---|---:|---|---|
| 1 | [colinlikescode](https://github.com/colinlikescode) | 1.6367x | `9fbbc5dca31c427e` | Public hit, unrelated to OKC kernel |
| 2 | [tjiang-fair](https://github.com/tjiang-fair) | 1.6367x | `bae72fd6675a430e` | No relevant public repository found |
| 3 | [naklecha](https://github.com/naklecha) | 1.5972x | `996ed626e1c84150` | No relevant public repository found |
| 4 | [jkbpvsc](https://github.com/jkbpvsc) | 1.5933x | `6b16112fadec4058` | No relevant public repository found |
| 5 | [k9thebeast](https://github.com/k9thebeast) | 1.5915x | `32aac8fac6fe47c1` | No relevant public repository found |
| 6 | [harshagundala](https://github.com/harshagundala) | 1.5899x | `cdc765e57b4a43c3` | No relevant public repository found |
| 7 | [zzz0906](https://github.com/zzz0906) | 1.5893x | `34f03f745b5941e8` | Only matching hit: SGLang fork; no submission found |
| 8 | [sudodaksh](https://github.com/sudodaksh) | 1.5893x | `5326bb1a5f2c4383` | No relevant public repository found |
| 9 | [schrepa](https://github.com/schrepa) | 1.5876x | `5122de1756d949dd` | No relevant public repository found |
| 10 | [mishravikas](https://github.com/mishravikas) | 1.5838x | `58c0c4cf03c84540` | No relevant public repository found |
| 11 | [pjmullin3](https://github.com/pjmullin3) | 1.5836x | `0e986c83e25f46ae` | No relevant public repository found |
| 12 | [olety](https://github.com/olety) | 1.5820x | `2ec57dbff5d54ed3` | No relevant public repository found |
| 13 | [mgpai22](https://github.com/mgpai22) | 1.5796x | `3ba81aec94f14d44` | No relevant public repository found |
| 14 | [0daycloud](https://github.com/0daycloud) | 1.5764x | `53eaddd1bb2a4de1` | No relevant public repository found |
| 15 | [ZhimaoL](https://github.com/ZhimaoL) | 1.5758x | `e53806af316d4305` | Only matching hit: SGLang fork; no submission found |
| 16 | [CapCap](https://github.com/CapCap) | 1.5729x | `00cc03c698cd44ab` | No relevant public repository found |
| 17 | [mdzor](https://github.com/mdzor) | 1.5718x | `48c3f1fffd364855` | No relevant public repository found |
| 18 | [vikram14s](https://github.com/vikram14s) | 1.5716x | `6c081858ee9b4ef0` | No relevant public repository found |
| 19 | [miketung](https://github.com/miketung) | 1.5712x | `69ce14404d4f4316` | Only matching hit: SGLang fork; no submission found |
| 20 | [narayanacharyuluchitroju](https://github.com/narayanacharyuluchitroju) | 1.5709x | `a935ef0eb67744bb` | No relevant public repository found |
| 21 | [spinualexandru](https://github.com/spinualexandru) | 1.5698x | `0a8f6ba411664e6c` | No relevant public repository found |
| 22 | [orisberlo-cell](https://github.com/orisberlo-cell) | 1.5689x | `0823e693fe934492` | No relevant public repository found |
| 23 | [mpeng19](https://github.com/mpeng19) | 1.5687x | `66d1b48514804025` | Only matching hit: SGLang fork; no submission found |
| 24 | [siddhantdubey](https://github.com/siddhantdubey) | 1.5687x | `cce109cae2a9443a` | No relevant public repository found |
| 25 | [shouryamaanjain](https://github.com/shouryamaanjain) | 1.5681x | `5a53e993ae364963` | No relevant public repository found |
| 26 | [2018kguo](https://github.com/2018kguo) | 1.5678x | `47df1227676b4e77` | No relevant public repository found |
| 27 | [reyanshsolis](https://github.com/reyanshsolis) | 1.5673x | `b49b4b312d5d4a7f` | No relevant public repository found |
| 28 | [anencore94](https://github.com/anencore94) | 1.5672x | `4be4ae4d290b4339` | Only matching hit: SGLang fork; no submission found |
| 29 | [concept-guest](https://github.com/concept-guest) | 1.5661x | `e74f31a12a6e4e3e` | No relevant public repository found |
| 30 | [thehimalayanleo](https://github.com/thehimalayanleo) | 1.5651x | `1378a90e27274611` | No relevant public repository found |
| 31 | [abhishekgahlot2](https://github.com/abhishekgahlot2) | 1.5641x | `3d2df758a18549c4` | No relevant public repository found |
| 32 | [franciskim](https://github.com/franciskim) | 1.5639x | `42ab1318d93b4cf1` | No relevant public repository found |
| 33 | [VKothari810](https://github.com/VKothari810) | 1.5635x | `78719448106645de` | GitHub account not publicly resolvable |
| 34 | [devstar222696](https://github.com/devstar222696) | 1.5629x | `d8b5ccc13a624a5e` | No relevant public repository found |
| 35 | [dixit122](https://github.com/dixit122) | 1.5624x | `a4e359863dd94cf9` | Only matching hit: SGLang fork; no submission found |
| 36 | [k4rth33k](https://github.com/k4rth33k) | 1.5617x | `9ec0e3e44de94f3b` | No relevant public repository found |
| 37 | [graadient](https://github.com/graadient) | 1.5616x | `2f1fc0838ca741a3` | No relevant public repository found |
| 38 | [seezatnap](https://github.com/seezatnap) | 1.5616x | `ab4a3e55bfa24310` | No relevant public repository found |
| 39 | [Meganpark980320](https://github.com/Meganpark980320) | 1.5613x | `ce38b52b5ee344fc` | No relevant public repository found |
| 40 | [NewstellerBot](https://github.com/NewstellerBot) | 1.5609x | `fd85d25f7e9843df` | GitHub account not publicly resolvable |
| 41 | [accountforgoofystuff](https://github.com/accountforgoofystuff) | 1.5608x | `5987bf672867464e` | Only matching hit: SGLang fork; no submission found |
| 42 | [saifalharthi](https://github.com/saifalharthi) | 1.5604x | `96792b1167854016` | No relevant public repository found |
| 43 | [Dagmawi-Beyene](https://github.com/Dagmawi-Beyene) | 1.5598x | `22e9c134638d4653` | No relevant public repository found |
| 44 | [jurajselep](https://github.com/jurajselep) | 1.5596x | `b138ec52a6164aa1` | No relevant public repository found |
| 45 | [nl-alonso](https://github.com/nl-alonso) | 1.5593x | `657d506c1ddc4e6f` | No relevant public repository found |
| 46 | [jonastirona](https://github.com/jonastirona) | 1.5592x | `a77fcc3ce68d4ce6` | No relevant public repository found |
| 47 | [rohanarun](https://github.com/rohanarun) | 1.5585x | `d35708a43d0e47a3` | No relevant public repository found |
| 48 | [natexcvi](https://github.com/natexcvi) | 1.5575x | `1fe0d64c9c27414b` | No relevant public repository found |
| 49 | [yxiaoGit](https://github.com/yxiaoGit) | 1.5568x | `3f7b8ea76b9f42f6` | No relevant public repository found |
| 50 | [bill810975](https://github.com/bill810975) | 1.5560x | `dc15f59ab42249ad` | No relevant public repository found |
| 51 | [bishal0922](https://github.com/bishal0922) | 1.5552x | `50293ac8e7164613` | Keyword false positive; no submission found |
| 52 | [almogtavor](https://github.com/almogtavor) | 1.5552x | `9d615dd76fc84998` | No relevant public repository found |
| 53 | [e-shizz](https://github.com/e-shizz) | 1.5550x | `0d9e75c628334ef4` | No relevant public repository found |
| 54 | [Maher4Ever](https://github.com/Maher4Ever) | 1.5549x | `e5f6b95606874ab8` | No relevant public repository found |
| 55 | [EmreKaplaner](https://github.com/EmreKaplaner) | 1.5544x | `f8e680615a0143bf` | No relevant public repository found |
| 56 | [vinod-memfold](https://github.com/vinod-memfold) | 1.5543x | `7afa4e92efdd4580` | Public OKC forum repo, not a submission |
| 57 | [Lurrobert](https://github.com/Lurrobert) | 1.5543x | `e989dbb1e74941ea` | No relevant public repository found |
| 58 | [adjustsquare](https://github.com/adjustsquare) | 1.5536x | `df49543fd71e4857` | No relevant public repository found |
| 59 | [ymiled](https://github.com/ymiled) | 1.5534x | `b3ff9a8b2a3045b6` | Only matching hits: SGLang forks; no submission found |
| 60 | [mottopanikeiku](https://github.com/mottopanikeiku) | 1.5523x | `f99d77e77f5d4689` | No relevant public repository found |
| 61 | [UnderAndOver](https://github.com/UnderAndOver) | 1.5515x | `8f3b5d9de20e494a` | No relevant public repository found |
| 62 | [LDD19](https://github.com/LDD19) | 1.5515x | `caebdd79128944c6` | No relevant public repository found |
| 63 | [moQ83](https://github.com/moQ83) | 1.5514x | `2f147a208bbb4b01` | No relevant public repository found |
| 64 | [booth-algo](https://github.com/booth-algo) | 1.5482x | `bd08cd03e37946f8` | No relevant public repository found |
| 65 | [deepak-foundryai](https://github.com/deepak-foundryai) | 1.5479x | `9da3cf9fa73d4004` | No relevant public repository found |
| 66 | [alygg77](https://github.com/alygg77) | 1.5477x | `56b99bc6bab54cb9` | No relevant public repository found |
| 67 | [Aspect5](https://github.com/Aspect5) | 1.5474x | `bc86b706f0d94576` | Keyword false positive; no submission found |
| 68 | [AkshitPareek](https://github.com/AkshitPareek) | 1.5470x | `9cac16e213844d60` | No relevant public repository found |
| 69 | [Junyi-99](https://github.com/Junyi-99) | 1.5459x | `1b4b7bea59a7497d` | Only matching hit: SGLang-JAX fork; no submission found |
| 70 | [suryuhh](https://github.com/suryuhh) | 1.5454x | `28a7dde9d9e44c8e` | No relevant public repository found |
| 71 | [saahas-kohli](https://github.com/saahas-kohli) | 1.5452x | `4576aa42860c4bc0` | Keyword false positive; no submission found |
| 72 | [khera-shanu](https://github.com/khera-shanu) | 1.5448x | `20b125de9a3741ab` | No relevant public repository found |
| 73 | [Zheruel](https://github.com/Zheruel) | 1.5445x | `3576913f9404449b` | No relevant public repository found |
| 74 | [Sudhendra](https://github.com/Sudhendra) | 1.5438x | `ce21a0d7dc0f47d6` | No relevant public repository found |
| 75 | [syhya](https://github.com/syhya) | 1.5438x | `37e3ff8d9b484f8c` | No relevant public repository found |
| 76 | [defiarbitrage-star](https://github.com/defiarbitrage-star) | 1.5431x | `2ba901b98b1f4fe3` | No relevant public repository found |
| 77 | [willwillbuild](https://github.com/willwillbuild) | 1.5419x | `738b5486d5be481a` | No relevant public repository found |
| 78 | [chenwainuo](https://github.com/chenwainuo) | 1.5419x | `421497ebc4d64f30` | VERIFIED public BF16 submission |
| 79 | [prateekm1007](https://github.com/prateekm1007) | 1.5410x | `3321a1f78bd44a93` | No relevant public repository found |
| 80 | [betitanx](https://github.com/betitanx) | 1.5410x | `d546f26fcfc945e3` | No relevant public repository found |
| 81 | [shashwatgokhe](https://github.com/shashwatgokhe) | 1.5408x | `5217f88e4a274afc` | No relevant public repository found |
| 82 | [fahrenheitjo](https://github.com/fahrenheitjo) | 1.5408x | `d8ecb5c1760c40a8` | No relevant public repository found |
| 83 | [maxgrev](https://github.com/maxgrev) | 1.5406x | `693f1fca2fc04b9d` | No relevant public repository found |
| 84 | [MohamadEidAlAktaa](https://github.com/MohamadEidAlAktaa) | 1.5406x | `137b9d659e3742f8` | No relevant public repository found |
| 85 | [Pinoeer-kingxi](https://github.com/Pinoeer-kingxi) | 1.5401x | `44a2f0505d8943a9` | GitHub account not publicly resolvable |
| 86 | [Raphael-Hao](https://github.com/Raphael-Hao) | 1.5396x | `4fbc0e32bc5c43f7` | Only matching hit: SGLang fork; no submission found |
| 87 | [Nickforall](https://github.com/Nickforall) | 1.5396x | `2002404f908d41f5` | No relevant public repository found |
| 88 | [StarAtNyte](https://github.com/StarAtNyte) | 1.5395x | `beb3e26afb2544c9` | No relevant public repository found |
| 89 | [jschb](https://github.com/jschb) | 1.5395x | `345b1495c0c64730` | No relevant public repository found |
| 90 | [heartkilla](https://github.com/heartkilla) | 1.5385x | `698b469202cf4fb1` | No relevant public repository found |
| 91 | [yousefarifalarif](https://github.com/yousefarifalarif) | 1.5381x | `69aa78b7336640f5` | No relevant public repository found |
| 92 | [AstonMartinRegistry](https://github.com/AstonMartinRegistry) | 1.5380x | `9b0b3337390d4527` | No relevant public repository found |
| 93 | [brianmickel](https://github.com/brianmickel) | 1.5380x | `b808b158212e4b54` | No relevant public repository found |
| 94 | [ashishvasant](https://github.com/ashishvasant) | 1.5376x | `27f37df3095a4e72` | No relevant public repository found |
| 95 | [sanv725](https://github.com/sanv725) | 1.5370x | `bc98ff49961b475e` | No relevant public repository found |
| 96 | [FreshlyCooked](https://github.com/FreshlyCooked) | 1.5368x | `3a19cec588c64e8a` | No relevant public repository found |
| 97 | [pierrelouisdelx](https://github.com/pierrelouisdelx) | 1.5368x | `9232b891dc54409f` | No relevant public repository found |
| 98 | [ronaldstoner](https://github.com/ronaldstoner) | 1.5362x | `b7e24dec030b481a` | No relevant public repository found |
| 99 | [ntovas](https://github.com/ntovas) | 1.5358x | `8b37cd8594ab41d8` | No relevant public repository found |
| 100 | [Krish-Parmar22](https://github.com/Krish-Parmar22) | 1.5358x | `d98fc478fdb14202` | No relevant public repository found |
| 101 | [BoruiXu](https://github.com/BoruiXu) | 1.5357x | `6628065172c449e5` | No relevant public repository found |
| 102 | [mikirejf](https://github.com/mikirejf) | 1.5355x | `254ac05c5705463e` | No relevant public repository found |
| 103 | [shrey-Bish](https://github.com/shrey-Bish) | 1.5344x | `467e94a0ad104f95` | No relevant public repository found |
| 104 | [bsflll](https://github.com/bsflll) | 1.5343x | `23f7b5761b414e57` | No relevant public repository found |
| 105 | [gauravsaini](https://github.com/gauravsaini) | 1.5338x | `371a280e2f3348d5` | No relevant public repository found |
| 106 | [JohnRSandoval](https://github.com/JohnRSandoval) | 1.5331x | `08b041e4ea784ee4` | No relevant public repository found |
| 107 | [nielsgl](https://github.com/nielsgl) | 1.5329x | `c453eaef05234005` | No relevant public repository found |
| 108 | [steelep](https://github.com/steelep) | 1.5329x | `a055e109f76645d4` | No relevant public repository found |
| 109 | [titoatwork](https://github.com/titoatwork) | 1.5321x | `10c29bbc06114703` | No relevant public repository found |
| 110 | [yairbarak22](https://github.com/yairbarak22) | 1.5318x | `4b8dd66de0ef40dc` | No relevant public repository found |
| 111 | [bobforeal](https://github.com/bobforeal) | 1.5311x | `260a66f051834b34` | No relevant public repository found |
| 112 | [1337isnot1337](https://github.com/1337isnot1337) | 1.5310x | `cf9a1d2e3b66469e` | No relevant public repository found |
| 113 | [c-nnr](https://github.com/c-nnr) | 1.5310x | `44edffdd5f5746f2` | No relevant public repository found |
| 114 | [vishaal-drkm](https://github.com/vishaal-drkm) | 1.5310x | `aad191717df04d51` | No relevant public repository found |
| 115 | [BenjiMencer](https://github.com/BenjiMencer) | 1.5307x | `b3ace9f676f2460b` | No relevant public repository found |
| 116 | [sjbaebae](https://github.com/sjbaebae) | 1.5301x | `ff4da6e1af3941cf` | No relevant public repository found |
| 117 | [Amal-David](https://github.com/Amal-David) | 1.5298x | `00a7e3fc42db4ee8` | No relevant public repository found |
| 118 | [pegasus1134](https://github.com/pegasus1134) | 1.5297x | `c31fff0a3f2440a5` | No relevant public repository found |
| 119 | [jsaurabh](https://github.com/jsaurabh) | 1.5297x | `2514cd98567e4afa` | No relevant public repository found |
| 120 | [wustep](https://github.com/wustep) | 1.5296x | `49d2f722fa5147d3` | No relevant public repository found |
| 121 | [burak-bayir](https://github.com/burak-bayir) | 1.5295x | `08020365577c43c5` | No relevant public repository found |
| 122 | [arkintof](https://github.com/arkintof) | 1.5294x | `7d6e27be13174d31` | No relevant public repository found |
| 123 | [ultramathi](https://github.com/ultramathi) | 1.5293x | `7eb720433ba34166` | No relevant public repository found |
| 124 | [EROREx](https://github.com/EROREx) | 1.5287x | `4418139ad34c4b36` | No relevant public repository found |
| 125 | [isaiahbjork](https://github.com/isaiahbjork) | 1.5286x | `f36c6277baac4092` | No relevant public repository found |
| 126 | [CapJack-cloud](https://github.com/CapJack-cloud) | 1.5285x | `4f66869111a941c1` | No relevant public repository found |
| 127 | [plingampally](https://github.com/plingampally) | 1.5280x | `b766f71ae4f4467e` | No relevant public repository found |
| 128 | [vracton](https://github.com/vracton) | 1.5278x | `81d6ce1011f94a53` | No relevant public repository found |
| 129 | [achitaan](https://github.com/achitaan) | 1.5278x | `6131bc96427e4486` | No relevant public repository found |
| 130 | [d42me](https://github.com/d42me) | 1.5273x | `3b082b1bb5dc4e03` | No relevant public repository found |
| 131 | [dfields-1](https://github.com/dfields-1) | 1.5271x | `3251d8f67bb54142` | Keyword false positive; no submission found |
| 132 | [1jmdev](https://github.com/1jmdev) | 1.5271x | `9a430278877348ff` | No relevant public repository found |
| 133 | [olanotolu](https://github.com/olanotolu) | 1.5268x | `ff2cb7e37ea645da` | No relevant public repository found |
| 134 | [zro-inference](https://github.com/zro-inference) | 1.5267x | `a13229bf9cf14c20` | No relevant public repository found |
| 135 | [Akashneelesh](https://github.com/Akashneelesh) | 1.5266x | `058c011ebd3c4b6e` | No relevant public repository found |
| 136 | [hsamrat007](https://github.com/hsamrat007) | 1.5256x | `fb6ba3e8aeec46f7` | No relevant public repository found |
| 137 | [arbazkhan971](https://github.com/arbazkhan971) | 1.5248x | `9b4433de76524036` | No relevant public repository found |
| 138 | [nftboy07](https://github.com/nftboy07) | 1.5245x | `50c7d097923144be` | No relevant public repository found |
| 139 | [GodlyDonuts](https://github.com/GodlyDonuts) | 1.5245x | `69a107021f0d4327` | Only matching hit: SGLang fork; no submission found |
| 140 | [shiv213](https://github.com/shiv213) | 1.5245x | `39db9749f15841ea` | No relevant public repository found |
| 141 | [xneak](https://github.com/xneak) | 1.5240x | `010d9128d8d943dd` | No relevant public repository found |
| 142 | [tanctl](https://github.com/tanctl) | 1.5239x | `b01be780948e4044` | No relevant public repository found |
| 143 | [Arutselvan](https://github.com/Arutselvan) | 1.5239x | `43aacb694c9e4511` | No relevant public repository found |
| 144 | [catminou4](https://github.com/catminou4) | 1.5238x | `93bee95d09b94cfb` | No relevant public repository found |
| 145 | [anthonyfong100](https://github.com/anthonyfong100) | 1.5236x | `0ccdd11f594d45f0` | No relevant public repository found |
| 146 | [aviyanp](https://github.com/aviyanp) | 1.5230x | `6ed63191fd374815` | No relevant public repository found |
| 147 | [argonautica](https://github.com/argonautica) | 1.5225x | `299b9bd4428442ad` | No relevant public repository found |
| 148 | [billy-enrizky](https://github.com/billy-enrizky) | 1.5225x | `3e4ae94ad4444c1e` | No relevant public repository found |
| 149 | [andreilungeanu](https://github.com/andreilungeanu) | 1.5220x | `51daddd901c34807` | No relevant public repository found |
| null | [sglang](https://github.com/sglang) | 1.0000x | `` | No relevant public repository found |


