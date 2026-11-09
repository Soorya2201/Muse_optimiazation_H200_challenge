# Public source and research coverage

This is a bounded review, not a claim to have searched the entire internet or
audited every entrant. A public-only model-name query retrieved155repository
matches across two pages; these include unrelated model projects, not155
challenge entrants. Initial searches also used `okc serve`, `SGLANG_OKC`, the
new base hash, serving/patch paths, MoE occupancy and DFlash gather/lookup terms.
The first authenticated query included our private repo metadata; that result
is excluded from public-source evidence. Zero code-search hits are not absence
proof, especially with index lag or unsupported query qualifiers.

Six selected public trees contain329files;31selected files were fetched at
exact commits with Git-blob/SHA verification. Fetching is not a full source
audit. Discovery queries, repository paths/licenses, file hashes and scope are
under `research/serving/`. No third-party harness was executed.

| Exact public source | Reviewed evidence and applicability |
|---|---|
| [hariappi a3ebb0f](https://github.com/hariappi/okc-muse-spark-kernels/tree/a3ebb0f2f027f07f38b063c6a74ac355a6b38848/serving) | Apache2. Read the complete one-line patch, A/B driver, its14timing cells and selected harness construction/Marlin/reference sections. The global/local expert-denominator bug remains in our new baseline. Adapted as candidate01. |
| [Meta Muse Mega ddb7925](https://github.com/facebookresearch/sgl-muse-mega/tree/ddb7925ff45167250201f35743a19c5152254052) | MIT. Complete README and tree screen; no device-source audit/port. Its reported1,109tok/s/2.1x is batch1 on8GB300, with different flags/KV precision. SM103 kernels and that result do not establish an H200 optimization. Layer fusion, early loads, overlapping communication and whole-step graphs are useful design leads. |
| [arihantbansal a6966ea](https://github.com/arihantbansal/open-kernel-challenge/tree/a6966ea2d68b877bbe527180625c40d8889f46a7) | Complete README/tree screen. Reports a historical peak MoE#1 and a2.37x saved kernel result. No current serving result established. No root license established; no code port/mirror. |
| [ruwwww c035bf9](https://github.com/ruwwww/open-kernel-challenge/tree/c035bf9d5a7d2445d8c8ee5ce49e63b8bb36f49d) | Complete README/tree screen and saved rules. Published kernel timings concern the earlier kernel task; serving folder exposes rules in this tree. No current serving patch/result found in this bounded inventory. No root license established; no source port. |
| [alygg77 6cae49a](https://github.com/alygg77/okc-muse-spark-kernels/tree/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0) | Same Apache2revision already pinned/reviewed in our kernel work; no new serving result established. Existing failed-ring/fence experiments and timing practices remain useful counterevidence. |
| [erkin98 cf977de](https://github.com/erkin98/muse-open-kernel-challenge/tree/cf977def7ea72d870151b0610474e54d9c589479) | Apache2metadata/tree and selected files fetched. Contents not fully audited this turn; do not infer a serving gain from this discovery. |

Hariappi's reported H100 NVL layer speedups span roughly1.34–1.80x for sampled
64–768-token cases. The A/B script alternates in fixed stock/EP order for nine
rounds. Some spreads are large. Its error field compares the EP output with
an FP32 reference, not directly with stock; the report does not establish
bit-identical stock/EP outputs. There is no256-token cell or full-model H200
result in that table. Our rebased policy/source checks are separate evidence.
Four unchanged source/evidence files plus LICENSE/provenance are pinned in
`research/public/hariappi_a3ebb0f/`. Attribute this public fix; do not call it
our novel invention.

The draft single-collective candidate is our new implementation in this patch
set. Packing value/ID data and distributed argmax are established techniques;
historical novelty is not established. It preserves raw FP32 value bits and
int64 IDs, uses the existing SGLang all-gather interface, and handles first-rank
NaNs/ties. Tests support CPU semantics only; GPU latency remains unknown.

Related primary work screens:

- [DFlash/prompt-lookup request](https://github.com/bstnxbt/dflash-mlx/issues/35)
  and [Splash proposal](https://github.com/incoai/splash/issues/186): request-derived
  lookup is prior art, not a new invention. These proposals do not prove
  full-model H200 gains; preserve target verification and assess actual hit rate.
- [Adaptive-draft configuration](https://github.com/nathansutton/chad/blob/main/docs/configuration.md):
  reports cases where full-block greedy verification beats adaptive mode and
  limited lookup replay in reasoning text. Different Apple hardware; useful
  counterevidence against assuming all coding traffic repeats.
- [Official serving update](https://openkernelchallenge.dev/updates): cached
  Nov3announcement prioritizes mergeable patches and a private serving board.
  Its cached Nov10opening conflicts with fresh CLI Nov12notice. Direct refresh
  returned403; use fresh official initialization and recheck at execution.

Next source targets are public serving diffs on the new base and exact H200
component measurements. Extend the inventory with commit/hash/license/read
scope before adopting code. Related implementation ideas are hypotheses until
our own full-node measurements demonstrate a compliant end-to-end benefit.
