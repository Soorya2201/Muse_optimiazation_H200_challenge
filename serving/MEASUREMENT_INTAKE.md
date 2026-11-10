# Use the existing serving observability

The official base already copies accepted counts to CPU with the normal
result transfer. `_resolve_spec_v2_tokens` computes accepted drafts excluding
the bonus and calls `Req.update_spec_correct_drafts_histogram`. The output
streamer carries those cumulative counters to the tokenizer, which exposes
them in native `/generate` response `meta_info`. No new production kernel,
`.item()` or synchronization is needed to obtain them. This does not mean
client capture or profiling has zero overhead; keep diagnostics outside timed
qualification runs. Pinned source hashes/read scopes are in
`research/serving/MEASUREMENT_SOURCE_AUDIT_2026-11-10.json`.

`tools/inspect_serving_acceptance.py` parses saved native JSON, JSONL snapshots
or SSE offline. It never connects to a server. OpenAI-compatible chat streams
have a different schema and are not silently treated as native metadata.

```sh
python tools/inspect_serving_acceptance.py --input /tmp/native-responses.jsonl --format jsonl --concurrency 16 --provenance 'Describe actual node, source SHA, flags, traffic, capture and timing phase' --output /tmp/acceptance-report.json
```

Supply one run/load per capture; request IDs must remain unique. A streaming
histogram is cumulative: partial frames are not extra decode rounds. The tool
counts the final response once, ignores identical final duplicates, rejects
conflicting finals, and flags unfinished captures. It checks sum(hist)=verify
count, weighted sum(hist)=correct drafts, proposed=7×verify count, and declared
stock rate/length formulas. It hashes input/tool provenance and request IDs,
omits text from output, and preserves a failure receipt if parsing fails.
Caller labels and captured bytes do not themselves authenticate a real run.

The synthetic fixture under `serving/fixtures/` contains metadata only and
does not represent generated220-token answers. Its CPU tests execute the
official histogram and response-metric bodies.230 checks pass, including20
invalid-input rejections, truncated streams and request-count weighting. The
sample CLI receipt is `ACCEPTANCE_SYNTHETIC_INTAKE_2026-11-10.json`; it is not
a GPU result despite the deliberately realistic metadata fields.

## Avoid the acceptance-length trap

The histogram counts accepted proposals A=0..7, excluding the bonus. A request
round has raw committed length A+1. Response `spec_accept_length`, however,
is client completion_tokens/verify_ct. The final accepted run can overrun a
length stop, and first-token/prefill accounting also differs. Thus a full
accepted8-token block does not establish an observed8tokens/round at the
client. The fixture deliberately shows these two means differing.

Request-level histograms pool request-rounds across batches, contexts and
remaining budgets. They do not identify real active batch, graph padding,
per-round remaining-token budget or cycle costs. The tool leaves active batch
unknown and does not manufacture cost-model input or a serving score. Its
raw yield screens for1/3/7proposals are full-policy counterfactuals only;
shorter-policy steady-state acceptance remains unmeasured.

## Profile without a new serving patch

The pinned HTTP server already supports `/start_profile` and `/stop_profile`.
On an existing development server, send the following JSON to `/start_profile`
for a separate untimed diagnostic cohort, preserving the judge's launch flags:

```json
{
  "num_steps": 8,
  "activities": ["CPU", "GPU"],
  "profile_by_stage": true,
  "profile_stages": ["prefill", "decode"],
  "with_stack": false,
  "record_shapes": true,
  "detailed_annotations": true,
  "merge_profiles": false
}
```

Use a fresh server-visible output directory/profile ID where needed, and
preserve the resulting files and logs. No profile request was sent here.
The default profiler's environment setting is PROFILE_V2=false; the V2 path
requires by-stage operation and does not support manual start/stop in the
same way. The proposed by-stage configuration follows both paths' source
constraints, but runtime success/export still needs confirmation. Do not
assume a fixed number of files/ranges from num_steps or a successful HTTP
response alone. If a stage is starved, inspect profiler state rather than
starting another capture over it.

The model runner labels target verification as `step[VERIFY ...]` and every
draft runner forward as `step[DRAFT ...]`, even when the draft borrows a
TARGET_VERIFY forward mode. Detailed labels include actual batch and CPU
query/KV aggregates plus a model-runner-local forward ID. CPU spans enclose
launch scheduling; their durations are not GPU compute times. Graph replay
separately rounds actual batch to a captured bucket. Inspect the selected
bucket/row layout rather than treating client concurrency, span batch and
padded graph batch as interchangeable.

Collect per-rank GPU timelines for the four client loads. Attribute GPU work
with valid launch/correlation evidence; asynchronous kernels may outlive
their CPU spans. Keep unassigned activity visible. Overlapping kernel times
cannot be summed into a critical-path fraction. A long NCCL kernel can mean
late peer arrival, not saturated link bandwidth. Compare rank completion
durations using a common measurement protocol; unrelated trace timestamps
are not a clock-calibrated cross-device timeline.

First questions to resolve: whether09's selector path and deterministic
FlashInfer top-k run; where the draft/verify/commit time goes; real graph
buckets; expert skew; and whether08's normalization has router slack to hide
within. Couple histograms to separately observed timing/context cohorts.
Then choose the next experiment from the dominant measured cost. Preserve
all-load stock-output,220-token and TTFT evidence independently; this intake
does not qualify a candidate.
