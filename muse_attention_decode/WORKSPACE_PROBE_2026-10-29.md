# Workspace and finalizer probe, October 29

Check `acf6f11e935049db`, cluster `sfo3`, passed 16 tests and 12 timed shapes
at 2.616172x (CI 2.576929-2.655415), 1439.622 us. The timed path is the parent;
this aggregate is not evidence of an optimization. Import-time probe finished in
48.5 s. All 50 alternative plans passed the synthetic correctness bound, with
maximum relative error 0.000146724. Zero qualified for adoption.

For each shape, this table selects the alternative with the largest minimum
parent/candidate time ratio across six groups. Ratios above one favor the candidate.
This selection is descriptive; the same data selected and evaluated the alternatives.

|Batch|Query length|Trials|Best alternative nc/coop|Minimum ratio|Maximum ratio|
|---:|---:|---:|---:|---:|---:|
|1|1|11|2/0|0.938489|0.962567|
|1|8|11|8/1|0.976119|1.003035|
|16|1|7|8/1|0.974545|0.987775|
|32|1|7|2/0|0.961078|0.980827|
|16|8|7|2/1|0.998952|1.014768|
|32|8|7|2/1|0.982899|0.997508|

B16L8 nc2/coop1 is a mixed, sub-1.5% result, not a consistent win. It is
incorrect to describe every alternative as slower. Some larger copy counts
regress severely: B1L8 nc32/coop0 ratios are 0.217536-0.219366. The fixed split
counts, synthetic normal-random keys/values, and one machine limit generalization.
The probe does not prove the parent is globally optimal.

Per CODEX_NEXT_STEPS.md, close the attention B1 merge/protocol family. Do not
submit this diagnostic file or spend more checks on these copy/finalizer variants.
Raw observations are in `workspace_probe_acf6f11e935049db.json`.
