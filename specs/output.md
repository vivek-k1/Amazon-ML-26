---
paths:
  - "src/s5_infer.py"
  - "make_submission.sh"
  - "code/**"
  - "Documentation_template.md"
---

# S5 test inference, outputs, submission package

## S5
- Test blocking → features → stage-1 predict → (cross-encoder on the band + VAL-fit combiner,
  when `optional.cross_encoder` is on; `thresholds_final.json`) → threshold → injective pass,
  all via `threshold.select` (the exact rule VAL was scored with; ties broken by S1 index, so
  reruns are byte-identical). Stage-1 and xenc scores are checkpointed per S3 part in
  `work/test/s5_scores/<s4_hash>-<s4x_hash>/` (`s5_infer.scores_dir`): keyed by the upstream
  model hashes only, outside the stage dir, so a threshold change or `--force s5_infer` reruns
  select + write (~3 min) and never the cross-encoder pass. The cross-encoder band mask is
  `s4x_xenc.band_mask` (shared with S4x; optional `band_top_k`).
- `output/matching_results.tsv`: header `source1_entity_id\tmatched_entity_ids`; exactly
  one row per test S1 id (left join from test S1); ids comma-joined, no duplicates; empty
  string when none.
- `output/candidate_pairs.tsv`: header `source1_entity_id\tcandidate_entity_ids`; exactly
  the union the model scored; every test S1 id present; matches ⊆ candidates.
- Write both with `separator="\t", quote_style="never"`.
- Sanity report by country: mean predicted matches per S1, empty rate, mean candidates per
  S1 (train GT reference: 3.46 matches, 5.6% empty). If France deviates sharply from
  US/India, suspect normalization first (suffixes, R/RUE, BIS/TER house numbers,
  accents); don't hand-tune a France threshold.
- Validate: `.venv/bin/python data/utils/validate_submission.py --matching
  output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir
  data/dataset/test` must print PASS.

## Submission package (`make_submission.sh`, run via /package)
Build `dist/<team_name>_submission.zip` only after the validator passes:
```
output/matching_results.tsv
output/candidate_pairs.tsv
code/business_entity_resolution/src/          pipeline incl. translit.py (no notebooks/, no tools/extract_notebook.py needed)
code/business_entity_resolution/README.md     exact commands: preflight → run_all.sh → validator
code/business_entity_resolution/requirements.txt   pinned from .venv/bin/pip freeze (only packages used)
code/business_entity_resolution/config.yaml
Documentation_template.md                     filled in (methodology, blocking, model, features)
```
- Never include: `notebooks/`, `.claude/`, `.aws`, ssh config, `logs/preflight.txt` if it
  shows account ids, `work/`, `models/`.
- After building: `unzip -l`, then grep the extracted zip for secrets (`hf_[A-Za-z0-9]{20,}`,
  `AKIA[0-9A-Z]{16}`, `PRIVATE KEY`, `aws_secret_access_key`). It must find nothing.
- Report the zip's size. `candidate_pairs.tsv` is ~1 GB uncompressed at 50 candidates per
  S1. If the portal has an upload limit, lowering k to the recall knee shrinks it.

## Submission gate (2 submissions left; ALL must hold before proposing an upload)
1. Validator PASS on both TSVs; matches ⊆ candidates; one row per test S1.
2. Produced by a timed clean run (`run_all.sh --force all` for run 1; a logged resume for
   run 2), every stage in `logs/timings.tsv`.
3. HOLDOUT F0.5 ± SE overall and per country recorded; it clearly beats 0.908 (the 0.924
   reference solution's honest VAL on the same metric).
4. France sanity (mean matches per S1, empty rate, mean candidates) within reason of US/India.
5. Reproducible: a second S5 run gives identical md5s for both TSVs (delete
   `work/test/s5_infer/_DONE.json` and rerun `s5_infer.py`; the score cache makes this ~3 min).
6. Secret scan of outputs and code clean.
7. Run 2 only: `tools/per_s1_f.py compare submissions/1 work/eval/s4x_full` shows HOLDOUT
   dF > 2 paired SE (dump submission 1's per-S1 F BEFORE `--force all`; it cannot be recomputed
   afterwards).

Archive every submission in `submissions/<n>/`: md5 of both TSVs, config.yaml,
thresholds.json, git commit, VAL/HOLDOUT F ± SE per country, France sanity, and a slot for
the public LB score the user reports (TSVs themselves are gitignored). Log it in PROGRESS.
