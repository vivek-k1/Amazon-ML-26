# Amazon ML Challenge 2026 — Business Entity Resolution

Match every test Source-1 (S1) business record to all its Source-2/3 (S2/S3) records.
Scored by F0.5 per S1, macro-averaged, singletons included: an empty prediction scores
1.0 on a singleton, and any prediction on a singleton scores 0.0.

## Read first
- `docs/PROGRESS.md`: current step, decisions, open issues. Update it after every step.
- `docs/eda-findings.md`: all EDA numbers. Don't re-derive what it establishes.
- `docs/v1_eda.py`: EDA code (the notebook with secrets redacted). Grep it; don't read it whole.
- Each stage's spec is a path-scoped rule in `.claude/rules/`, and it loads when you read
  the matching `src/` file. Before creating or editing a stage, read its rule file:
  S0/S1 `prepare.md`, S2 `blocking.md`, S3 `features.md`, S4 `train.md`,
  S5/packaging `output.md`, run_all/preflight/config/common `infra.md`, optional
  stages `optional-stages.md`.

## Hard constraints
- IMPORTANT: one clean end-to-end run (raw TSVs → both output TSVs → validator, training
  included) must take < `budget_hours` (config: 8) on this VM. Time left in total: 20 h from
  2026-09-26 21:30 = step 3 ≤ 4 h, run 1 ≤ 8 h, user buffer 4 h (edits, VM switch), run 2 ≤ 4 h
  resuming from run-1 checkpoints (S0, S2, embeddings are reused, not recomputed).
- IMPORTANT: only 2 leaderboard submissions remain (run 1 → submission 1, run 2 → submission
  2). The user uploads; never propose a submission that hasn't passed the gate in
  `.claude/rules/output.md`. Final ranking = PRIVATE leaderboard (public is a test subset).
- Reference code `~/lb1_0.924_code` (public LB 0.924) is read-only inspiration: nothing from it
  is copied into `src/`, `code/` or the zip. Its notebook contains a secret: read it only via
  `tools/extract_notebook.py`. Findings: `docs/reference-approach.md`.
- Models: MIT or Apache-2.0 and ≤ 8B params, local weights, no network at inference.
  No external lookups (geocoding, registries, entity DBs, web): that is disqualification.
- `country` is an open set (test adds France, unseen in train). Partition by its value
  dynamically. Never hard-code {US, India}. Never give `country` to the model as a feature.
- Raw data is read-only (`data/` is chmod'ed read-only). Entity IDs are strings.
- No secrets anywhere in `src/`, `code/`, logs, docs or the zip. AWS keys live in `~/.aws`
  (use the `aws` CLI; never read or print them). `notebooks/` holds the raw notebook with a
  token: never read, copy or ship it.

## Environment quirks
- Python: always `.venv/bin/python` (3.12). Verify packages with `.venv/bin/pip freeze`.
- VM: GCP g2-standard-32 (32 vCPU, 128 GB, 1× L4 24 GB, 100 GB SSD). It may change, so
  nothing may assume its size (see `infra.md`).
- Long jobs: anything > 2 min runs detached in tmux, and you wait with
  `bash tools/wait_stage.sh <session> 540` (call it with a 600000 ms tool timeout). Tool
  calls time out; tmux jobs don't. Use `/run-stage` for this.
- Env from `.claude/settings.local.json`: `BER_S3_BUCKET`, `RUN_ID`. If
  `BER_S3_BUCKET` is unset, S3 sync is skipped, never an error.

## Layout
```
data/dataset/{train,test}/*.tsv, data/utils/validate_submission.py   read-only
src/            s0_prepare s1_gt s2_block s3_features s4_train s5_infer common translit threshold (.py)
                optional: s2e_embed s4x_xenc s4l_llm (.py)
.venv-llm/      separate venv for vLLM (never mixed into .venv)
models/         local weights: multilingual-e5-small, bge-m3, Qwen2.5-7B-Instruct (gitignored)
submissions/    <n>/ archive per leaderboard submission (TSVs gitignored)
config.yaml     every tunable; config.v1.yaml = frozen baseline
run_all.sh      resume by default; --from <stage>, --force <stage|all>
preflight.sh    environment + data + AWS + secrets checks
work/           per-stage parquet + _DONE.json (checkpoints), never zipped
output/         matching_results.tsv, candidate_pairs.tsv, artifacts/
logs/           <stage>.log, timings.tsv, preflight.txt
docs/           PROGRESS.md, eda-findings.md, v1_eda.py
tools/          wait_stage.sh, extract_notebook.py
```

## Facts that drive the design
- Train S1 2,206,821 / S2 5,034,616 / S3 5,285,603. Test S1 1,732,544 (India 46.8%,
  US 38.3%, France 15.0%) / S2 4,887,273 / S3 5,082,316.
- GT: 7,638,365 positive pairs, mean 3.46 matches per S1 (max 11), 5.58% singletons,
  100% same-country. Empty GT cell = 0 matches.
- Names are noisy in every way (suffixes, order, typos, domains/hashtags, renames). ~18–28%
  of Indian pool names are Indic script; France and part of US are accented Latin.
- Addresses: containment beats Jaccard (truncation); shared numbers are strong; postal
  codes are mostly absent; leading zeros vary.
- Blocking ceiling ≈ 0.995 (pairs share ≥ 2 tokens). Flat DF-capped keys can't reach 0.95
  in budget (EDA-11). EDA-13's per-S1 Python key search doesn't scale; don't port it.

## Decisions (reasons in the rule files and eda-findings)
1. Blocking = per-country IDF-weighted top-k sparse retrieval (`sparse_dot_topn`), over
   the views W_name, W_addr and C_name, unioned to ≤ 50 candidates per S1.
2. Transliterate: `src/translit.py` (`to_latin`, `skeleton`, `script_class`). It's
   tested; don't replace it with unidecode (GPL) or the notebook's `fold()` (breaks Indic).
3. Matcher = GBM, `model.backend` lightgbm (CPU) | xgboost (GPU). Benchmark once, keep the
   winner. Threshold picked by exact macro-F0.5 on held-out train S1s.
4. Injective pass on the S2/S3 side only (each S2/S3 id keeps its best S1), once GT
   confirms it. Never one-to-one: S1s keep any number of matches.
5. Optional stages (embeddings, cross-encoder, Qwen triage, S1 emit gate, reverse competition)
   are evaluated in step 3 on VAL. One enters run 1 only if its paired VAL ΔF > 2 SE and its
   measured cost fits the budget. BK-trees/SymSpell stay cut.
6. Train S2 blocks ALL 2.2M train S1s and stores up to `store_candidates` (80) per S1 with
   `cand_pos`; S3 applies `max_candidates` (50). Run 2 can change the cap, the train set size
   or add reverse-competition features without re-blocking.

## Target and ceiling
- User target: 0.98 macro-F0.5. Blocking-oracle F0.5 (predict GT ∩ candidates) on the EDA-14
  50K VAL slice: 0.9900 at cap 50 (US .9949, India .9827), 0.9909 at cap 80. The matcher is
  the bottleneck; a blocking change needs oracle-F evidence first.
- Reference point: the 0.924 public-LB solution scored 0.908 on an honest VAL (full GT,
  singletons in). One data point, not a formula.

## Submissions (2 left)
- VAL (100K train S1s) is for choosing: model, features, t, r, optional stages. HOLDOUT (100K
  train S1s, `work/train/s1_gt/holdout_ids.parquet`) is never used for any choice; S4 scores
  it at the chosen settings, and it is the leaderboard predictor.
- Submission 1 = run 1 after the gate. From its public LB score and HOLDOUT per-country F,
  infer France: `F_Fr ≈ (LB − 0.468·F_India − 0.383·F_US) / 0.150`. A low F_Fr means fix
  France normalization before anything else.
- Submission 2 only if a candidate beats submission 1 on HOLDOUT by > 2 paired SE (per-S1 F
  differences over the same S1s) and nothing in it leans on labels France lacks.
- Every submission is archived in `submissions/<n>/` and logged in PROGRESS (output.md).

## Time budget (clean run; the Measured column comes from logs/timings.tsv)
| Stage | Planned | Measured |
|---|---|---|
| S0 prepare (~24.2M records) | 15 min | 4.4 min (train 2.1 + test 2.2) |
| S1 GT checks + split | 2 min | 4 s |
| S2a block all 2.2M train S1 | 25 min | 29.3 min |
| **S2b block 1.73M test S1** (highest risk: India partition) | 30 min | 17.9 min |
| S3a features train+val (~15M pairs) | 10 min | 3.6 min (19.9M pairs) |
| **S3b features test (≤ 87M pairs)** (second risk: memory) | 35 min | 8.9 min (86.0M pairs, peak RSS 19 GB) |
| S2e embeddings (optional) | measure | not in run 1 |
| S4 GBM + threshold sweep | 20 min | 4.1 min |
| S4x cross-encoder (train 17 min + score) | 25 min | 22.1 min (band .01–.99) |
| S5 incl. cross-encoder on the test band | 35–90 min | 34.7 min (xenc 31.7 min, 5.50M pairs = 6.4%) |
| S5 predict + injective + write | 15 min | ~3 min (inside the 34.7) |
| S6 validator + sanity | 5 min | < 1 min |
| **Total (v1 core)** | **≈ 2 h 40 min** | **2 h 07 min wall (run 1, 2026-09-27)** |

## Working rules
- Order of work is in `docs/PROGRESS.md`. Stop and report to the user after each step.
- Every stage first runs on `--limit-s1 50000`. Extrapolate its time to the full run; if
  any stage extrapolates past its kill criterion (rule file), stop and report.
- Checkpoints: stages are resumable via `work/**/_DONE.json` and per-chunk atomic parquet
  (details in `infra.md`). A crash must never force redoing finished work.
- Report measured wall-clock numbers, never estimates. Show evidence (command + output).
- Every F0.5 is reported with its SE (std of per-S1 F / √n). Compare variants with the
  paired SE of per-S1 F differences over the same S1s (`threshold.paired_se`).
- Before marking a newly written stage done, have the `pipeline-reviewer` subagent check
  it against its rule file. Fix correctness gaps only.
- Read big logs/outputs through `tools/wait_stage.sh` or the `log-reader` subagent. Never
  `cat` a whole log, TSV or parquet into the conversation.
- Commit after each working stage (`git add -A && git commit`). Never `git push` without asking.
- Never create cloud instances or spend credits without the user's explicit OK.

## When compacting
Preserve: the current step and stage, measured timings so far, chosen config values
(k, caps, thresholds), open issues, and the list of files modified in this session.

## Done means
Validator PASS; the submission gate passes (output.md); HOLDOUT F ± SE per country recorded; one clean `run_all.sh --force all` under budget with every stage in
`logs/timings.tsv`; report (blocking recall curve, VAL macro-F0.5 and thresholds with
reasoning, per-country, injective on/off, France sanity, what was cut);
`dist/<team>_submission.zip` with BOTH TSVs, code package, and methodology doc, secret scan clean.
