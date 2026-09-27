# Handoff brief — Amazon ML Challenge 2026, Business Entity Resolution

This brief is for an AI reviewer. It says what we built, why we built it that way, what we
measured, and what we want back from you. It stands on its own; the code bundle it ships with
(file map in §13) adds the details. All times are UTC. Numbers are measured unless marked
*estimate*. "±" is the standard error (SE) of a per-record mean. "Paired" means the SE of
per-record differences between two variants on the same records.

**Deadline:** 2026-09-27 17:30 UTC. **One leaderboard submission is left.** Proposals arriving
after about 11:00 UTC can only be small, because run 2 plus the submission checks takes about
3–4 h.

---

## 0. What we want from you

We want a ranked list of at most 8 concrete improvements, ranked by expected leaderboard gain
per hour of work. Use the format in §12. Improvements that help France (§7) or the largest loss
parts (§5) matter most.

- Ground every claim in the numbers here or in the code.
- Point out bugs, leakage or metric mistakes you find. These are as valuable as new ideas.
- Don't re-propose anything in §8 unless you give a specific reason why it would now behave
  differently.

We will re-measure every proposal on our side before adopting it (see §10). We adopt only what
wins on VAL by more than 2 paired SE and fits the time left.

---

## 1. Task and metric

- **Task.** Each test Source-1 record (S1: `entity_id`, `business_name`, `business_address`,
  `country`) must be matched to **all** its records in Sources 2 and 3 (S2/S3; together, "the
  pool"). The output is two files:
  - `matching_results.tsv`: one row per test S1, with a comma-separated list of matched ids.
  - `candidate_pairs.tsv`: the candidate list we scored. It is required but not scored.
- **Metric.** For each S1, F0.5 = 1.25·tp / (0.25·n_gt + n_pred), then the macro average over
  all S1s.
  - A **singleton** is an S1 with no true match. It scores 1 if we predict nothing and 0
    otherwise.
  - Recall counts all true matches, including ones our candidate generation never retrieved.
- **Leaderboard.** The public leaderboard scores a subset of test. The final ranking uses the
  private leaderboard.

## 2. Data facts

| | Train | Test |
|---|---|---|
| S1 records | 2,206,821 (India 40%, US 60%) | 1,732,544 (India 46.8%, US 38.3%, **France 15.0%**) |
| Pool (S2 + S3) | 10,320,219 | 9,969,589 |
| Pool records per S1 | 4.68 | 5.75 (India 5.82, US 5.76, France 5.53) |

- **Ground truth (train only).** 7,638,365 true pairs.
  - Mean 3.46 matches per S1 (max 11). 5.58% of S1s are singletons.
  - All matches are within the same country.
  - Each pool record belongs to at most one S1. This is checked; 0 violations.
- **France appears only in test**, so there are no France labels anywhere.
- **Names are noisy in every way:** legal suffixes, word order, typos, domains and hashtags,
  renames (e.g. "X formerly: Y").
  - About 18–28% of Indian pool names are in Indic scripts.
  - France and parts of the US use accented Latin.
- **Addresses.** Containment works better than Jaccard, because addresses get truncated.
  Shared house numbers are strong evidence. Postal codes are mostly absent, and leading zeros
  vary. About 3% of pool addresses are empty, in every country.
- **Test has 23% more pool records per S1 than train.** Our predicted matches per S1 on test
  (≈3.3) equal those on the labeled HOLDOUT, and the public LB is 0.965. That argues against
  test S1s having many more true matches. Our inference: the extra pool records are mostly
  non-matching distractors.

## 3. Hard constraints (breaking one means disqualification or a failed run)

1. **Models:** MIT or Apache-2.0 license, ≤ 8B parameters, weights stored locally, **no network
   at inference.**
2. **No external lookups:** no geocoding, business registries, entity databases or web. A
   hand-written normalization table in code (like our US-state list) is fine.
3. **`country` is never a model feature.** Countries are an open set (France is unseen), and
   the code partitions by country dynamically. Never hard-code {US, India}, and never tune a
   threshold per country.
4. **One clean end-to-end run** (raw TSVs → both output files → validator, training included)
   must finish in under **8 h** on our VM. Run 2 may reuse run-1 checkpoints and must fit in
   about **4 h**.
5. **Only 1 submission is left.** It is submitted only if it beats submission 1 on HOLDOUT by
   more than 2 paired SE and passes our gate. The gate: validator pass, byte-identical rerun,
   France sanity, secret scan.

**VM:** 32 vCPU, 128 GB RAM, 1× NVIDIA L4 (24 GB), 100 GB SSD. Python 3.12 with polars,
numpy, rapidfuzz, sparse_dot_topn, xgboost (GPU), lightgbm, torch 2.x, transformers.

## 4. Pipeline, stage by stage, with the reason for each choice

The pipeline is `run_all.sh` running stages `src/s0…s5`. Every stage writes checkpoints and a
`_DONE.json` holding a hash of its config and code, and it resumes by default. Run-1 wall time
was **2 h 07 min** in total.

| Stage | What it does | Why | Run-1 time |
|---|---|---|---|
| S0 `s0_prepare.py` | Normalize text. | See the notes below the table. | 2.1 + 2.2 min |
| S1 `s1_gt.py` | Load GT, check injectivity. Split train S1s, stratified by country: TRAIN 180K, ES 20K (early stopping), VAL 100K (all choices), HOLDOUT 100K (report only, the leaderboard predictor). | Honest estimates with error bars. | 4 s |
| S2 `s2_block.py` | Candidate generation (details below). | The table after the notes shows why. | train 29.3 / test 17.9 min |
| S3 `s3_features.py` | 53 pair features (list below). | Hand-crafted evidence for the GBM. | train 3.6 / test 8.9 min (86M pairs) |
| S4 `s4_train.py` | Stage-1 XGBoost on GPU (depth 8, lr 0.05, early stopping on ES; best iteration 1994). Sweeps threshold t, relative threshold r and injective on/off on VAL with exact macro-F0.5; t is the centre of the plateau. | The GBM is the reliable workhorse. | 4.1 min |
| S4x `s4x_xenc.py` | Cross-encoder (details below). | By far our biggest gain (§5). | 22.1 min |
| S5 `s5_infer.py` | Stage 1 → cross-encoder on the band → combiner → threshold → **injective pass** (each pool record keeps only its best S1; S1s keep any number) → both TSVs. Deterministic tie-breaks, so reruns are byte-identical. | Mirrors exactly what VAL scored. | 34.7 min (cross-encoder 31.7 min on 5.50M band pairs) |

**S0 normalization.**
- `to_latin` transliteration (Indic → Latin, custom and tested) and accent folding.
- Name forms:
  - `name_s`: legal forms removed, including LLC/INC/LTD/PVT/…, SAS/SASU/SARL/EURL/SA/SCI/SNC/
    SELARL/EIRL, and romanized Indic forms.
  - `name_sk`: a consonant skeleton.
- `addr_n`: street types and US/Indian states mapped to codes (e.g. R→RUE, CHE→CHEMIN,
  ALL→ALLEE, BD→BLVD, AV→AVE). Also `addr_nums` (house numbers).

**Why S0 looks like this.**
- Unidecode is GPL and breaks Indic.
- Suffixes and states are the most common spurious differences.
- French region/department names are **not** normalized, so "Hauts-de-France" and "Nord" differ.

**S2 candidate generation.**
- Per country, IDF-weighted sparse top-k (`sparse_dot_topn`) over two views:
  - `W_all`: word tokens of name + address, k = 40.
  - `C_all`: name-skeleton character 3-grams + address tokens, k = 40.
- The union is ordered by best view rank and stored up to 80 per S1 (`cand_pos`). S3 caps it at
  50.
- Train blocks **all** 2.2M train S1s, so reverse-competition features see full density.

**Why these views.** From the step-2 blocking study on a 50K VAL sample:

| Views | Recall |
|---|---|
| Name-only + address-only + name-3-gram views, 20 each | .934 |
| W_all at 50 | .961 |
| **W_all 40 + C_all 40 (chosen)** | **.972** (US .985, India .952), 49.7 candidates per S1 |
| All five views uncapped | .980, but 176 candidates per S1, too costly |

Names repeat across businesses (17% of Indian pool records share their exact name with more
than 50 others), so name-only views produce ties.

**S3 features**, grouped:
- **Blocking:** score, rank and gap per view; number of views; number of candidates; `cand_pos`.
- **Name:** exact match; skeleton exact; first token; length difference; script codes and a
  cross-script flag; name frequencies (both sides); rapidfuzz ratio, token_sort, token_set,
  partial and Jaro-Winkler on `name_s` and `name_sk`; space-less ratio and partial; name IDF
  containment.
- **Address:** containment (shared / min size), Jaccard, IDF containment and IDF-shared,
  token_set, partial, missing flags, last-two-token overlap.
- **Numbers:** shared, conflict, longest, first-number match.
- **Group:** rank and gap of name / address similarity within the S1's candidates.
- **Reverse competition (`rc_`):** among all S1s that retrieved this pool record, this S1's
  rank and gap by W_all/C_all score. **These are the top two features by gain, 7× the next one.**
- **Source flag:** `is_s3`. There is never a `country` feature.

**S4x cross-encoder.**
- Model: `multilingual-e5-small` (MIT, 118M), fine-tuned 1 epoch with BCE.
- Training data: 1M TRAIN pairs = 195K pairs with stage-1 p in [.05, .95], plus 402K
  positives, plus 402K negatives.
- Input: raw "name | address" of both records, max 96 tokens.
- It scores every pair with stage-1 p in [.01, .99] (the **band**).
- A logistic combiner on [logit p, xenc, logit p·xenc] is fit on VAL band pairs. It is
  cross-fitted over 2 VAL folds for choosing t, then refit on all of VAL. Chosen t = 0.70, r = 0.

**Why the cross-encoder.** It generalizes: on VAL band pairs whose pool names never appeared in
TRAIN candidates its AUC is .994, against .967 for seen names.

## 5. Results

| System | VAL F0.5 | HOLDOUT F0.5 (report only) | India / US (HOLDOUT) |
|---|---|---|---|
| Blocking oracle (predict GT ∩ candidates) | .9899 ± .0002 | – | .9827 / .9949 (VAL) |
| Stage-1 XGBoost | .94843 ± .00045 | .94836 | .9310 / .9600 |
| **+ e5-small cross-encoder, band .01–.99 (= submission 1)** | **.97682 ± .00029** | **.97642 ± .00029** | **.96772 / .98222** |
| **Submission 1 public LB** | | **0.965** | |

**HOLDOUT loss** (1 − F = .0236), split by where it comes from:

| Part | Loss |
|---|---|
| True match never in the candidates (blocking) | **.0104** |
| True match in the candidates but not predicted | .0080 |
| False positives on S1s that have matches | .0037 |
| Predictions on singletons (emit rate 2.6%) | .0014 |

**Blocking misses.**
- On HOLDOUT, 4.8% of India's true pairs and 1.3% of US's are not retrieved **within 80**.
  Only 0.31% / 0.16% sit at positions 50–79.
- Their causes, from a 200-sample categorization:
  - 50%: ASCII typos, domains or digit-for-letter in the name, with a weak address.
  - 29%: Indic transliteration with a weak address.
  - 12%: names essentially different.
  - 10%: empty pool address with a generic name.

**VAL errors by stage-1 p** (before the injective pass):
- Inside the band [.01, .99]: 7,530 missed true matches and 1,667 false positives.
- Missed true matches outside the band: 820 with p in [.002, .01), 175 with p in [.0005, .002),
  97 below .0005.
- False positives with p above .99: 274.

## 6. Timeline so far

- **Step 3 (≈4 h):** implement S3/S4, the backend benchmark, the optional-stage experiments
  (§8) and the HOLDOUT split.
- **Run 1 (clean `--force all`, 2 h 07 min):** gate passed → submission 1 → **LB 0.965**.
- **Today (after the LB):** the France diagnosis (§7), cross-encoder speed-up (length-sorted
  batches: e5-small end to end 2.7K → 4.4K pairs/s; the GPU part runs at 7.5K pairs/s, and
  tokenization is now about 40% of the time), and the bge reranker experiment (§9).

## 7. The France problem (our main suspect for LB 0.965 vs HOLDOUT 0.976)

**Estimate.** If India and US score on test what they score on HOLDOUT:

F_France ≈ (LB − 0.468·F_India − 0.383·F_US) / 0.150 = (0.965 − 0.8291) / 0.150 ≈ **0.906**,

against .968 for India and .982 for US. France then costs about as much as India (0.0141 vs
0.0151 of the score) at a third of the size.

**Label-free evidence** (`tools/diag_country.py`: HOLDOUT, which has labels, vs test, same
statistics):

| | HOLDOUT India / US | Test India / US | Test France |
|---|---|---|---|
| Mean predicted matches per S1 | 3.24 / 3.34 | 3.24 / 3.35 | **3.12** |
| Empty prediction rate | 6.1% / 5.7% | 6.3% / 5.8% | 6.1% |
| Self-estimated F (from its own probabilities; true F on HOLDOUT is .968 / .982) | .987 / .985 | .986 / .986 | **.975** |
| S1s whose best candidate is uncertain (.2–.9) | 0.6% / 0.7% | 0.6% / 0.6% | **1.3%** |
| Cross-encoder and stage 1 disagree (band pairs) | 14.8% / 13.7% | 15.5% / 12.3% | **19.6%** |
| Kept matches at candidate position ≥ 40 | 0.35% / 0.11% | 0.37% / 0.08% | **0.84%** |

**Clear France pairs are handled fine.** For pairs with the exact same name and the same house
number, 0.6% get stage-1 p < .5 in France, the same as the US. The loss is in hard pairs and in
retrieval. France's kept matches sit deeper in the candidate lists, which suggests more France
true pairs fall beyond position 80.

**What France records look like** (sampled test rows; synthetic-looking, templated):
- **Names:** "{City} {Word} {legal form}": "Tourcoing Culturel SARL", "Lille Comite SARL",
  "Aikido Compagnie SA".
  - Many records share the city and word, so there is a lot of name competition within a city.
  - Legal-form variants: "S.A.R.L.", "S.A.S", "[SAS]", "5AS", "E.U.R.L.", "Sarl".
  - Accent noise: "Çompagnie", "Spôrtive", "Ènglish".
  - Renames: "Tavodrex+ formerly: Association du English".
- **Addresses:**
  - Number variants: "N° 23", "Nº 167", "# 6", "023", "16 -".
  - Abbreviations: "R.", "Q." (quai), "Rte.", "All.", "Blvd"/"BD".
  - Case: all caps on one side, mixed case on the other.
  - Reordered segments: "Lille, Hauts-de-France, 29 RUE X".
  - **Region vs department:** S1 addresses end with the region ("Hauts-de-France", "Pays de la
    Loire", "Nouvelle-Aquitaine"). Pool addresses end with the department ("Nord",
    "Pas-de-Calais", "Gironde", "Loire-Atlantique"), the region, or nothing.
  - The region/department mismatch lowers `a_contain`, `a_jacc` and `a_last2` for true France
    pairs. On clear pairs p stays high anyway.
- **Examples the system gets wrong or unsure** (✓ = predicted):
  - S1 "Tourcoing Culturel SARL | 162 Rue de la Baille, Tourcoing, Hauts-de-France". Candidate
    "Tourcoing Culturel SARL | (empty address)": stage-1 p .03, cross-encoder +1.4, final .16,
    **not predicted**.
    - On HOLDOUT, exact-name pairs with an empty pool address are true only about 10% of the
      time, so this is correct behaviour for US/India. It may be wrong for France's templated
      data.
  - S1 "École primaire Trans | 6 Rue de la Cornouaille, Nantes".
    - "Anciens Trans | 6 R. De La Cornouaille": ✓ (final .75).
    - "école Primaire Trans EURL | 17 R. De La Cornouaille": stage-1 p .77, cross-encoder
      −7.8, final .01, not predicted.

**Alternative hypothesis (can't be excluded with one LB number).** All countries lose a little
on test, for example through the 23% extra distractors, rather than France alone.
- Against it: US/India test behaviour matches HOLDOUT on every statistic above.
- The injective pass is not the cause. Doubling the S1 density on VAL changes its effect only
  from +.00003 to +.00008.

**Leave-one-country-out** (train on one country, test on the other; stage 1 only): India
trained on US only scores .846 (−.085); US trained on India only scores .929 (−.031). An unseen
country costs a lot at stage 1; the cross-encoder is what transfers.

## 8. Already tried (please don't re-propose without a new reason)

| Experiment | Result (paired ΔF on VAL ± SE unless noted) | Decision |
|---|---|---|
| LightGBM vs XGBoost (50K slice) | .9395 vs .9397, tie; XGBoost 3× faster | XGBoost |
| Reverse-competition features (`rc_`) | +.00303 ± .00023 | in |
| Injective pass | +.00005 ± .00002 | in |
| S1-level emit gate (a small GBM predicting "has ≥ 1 match") | +.00008 ± .00013 | out |
| Qwen2.5-7B yes/no log-prob on band pairs (vLLM, 4-shot) | alone +.0041; with the cross-encoder +.0357 vs cross-encoder only +.0364 (4K S1s); 49–61 pairs/s | out |
| e5-small cross-encoder, band .05–.95 | +.0263 ± .0004 | in |
| Band widened to .01–.99 | +.0021 more | in |
| Stage-2 GBM: out-of-fold stage-1 p + S1-group context + similarity to the S1's top candidate | +.0049 over stage 1; **+.0012 on top of the cross-encoder** (band .05–.95) | not implemented (run-2 candidate) |
| Embedding view: e5-small names, exact GPU top-k, as a 3rd blocking view | oracle F +.0008 (top-5) to +.0016 (top-20) | needs re-blocking; deferred |
| Cap 80 instead of 50 | oracle F +.0009 | small |
| Context-aware combiner (LightGBM on [p, xenc, S1-context ranks and gaps]) | +.00001 ± .00010 | out |
| Name embedding cosine as a feature | not measured separately (the cross-encoder covers pairwise semantics) | out |

## 9. In flight right now

**`BAAI/bge-reranker-v2-m3`** (Apache-2.0, 568M, a pre-trained multilingual pair reranker),
fine-tuned as our cross-encoder.
- Training: 300K TRAIN pairs = 195K band pairs + 52K positives + 52K negatives. Batch 64,
  lr 2e-5, 1 epoch.
- Speed: 150 pairs/s, 33.6 min. Batch 128 ran out of GPU memory.
- Evaluation: the same VAL/HOLDOUT harness, paired against the e5-small scores.
- **Result (07:37 UTC):**
  - **VAL .97910** (India .97193, US .98388): **+.00228 ± .00013** paired vs e5-small.
  - HOLDOUT .97882 (India .97053, US .98435): +.0024 ± .00014, report only.
  - t = .71, plateau [.55, .88].
  - Singleton emit rate: 2.6% → 1.8%.
- **HOLDOUT loss with the reranker** (total .0212):

  | Part | Loss |
  |---|---|
  | Blocking misses | **.0104, now half of the loss** |
  | Missed true matches | .0069 |
  | False positives | .0029 |
  | Singleton predictions | .0010 |

- **Cost:** scoring runs at 880 pairs/s, so S5 on the 5.5M-pair test band takes about 104 min.
  Training takes 34 min.
- **Plan:** it goes into run 2 unless something better arrives. Its speed makes a wider band
  (≈ +65% pairs) expensive.

## 10. How we evaluate any proposal

- **Choose on VAL only.** Report HOLDOUT once, never choose on it. Compare variants by paired SE
  over the same S1s (`threshold.paired_se`). Adopt only if ΔF on VAL > 2 paired SE and the
  measured cost fits.
- **France has no labels,** so France-targeted ideas need a label-free argument:
  - leave-one-country-out as a proxy for an unseen country;
  - test-side statistics like §7 (`tools/diag_country.py`, `tools/diag_xenc_country.py`);
  - or evidence that the change is neutral on US/India VAL and fixes a demonstrable France
    mismatch.
- **Harnesses in the bundle:**
  - `tools/exp_xenc.py`: cross-encoder variants.
  - `tools/exp_combiner.py`: combiners.
  - `tools/exp_stage2b.py`: stage 2.
  - `tools/exp_embed_view.py`: embedding view oracle.
  - `tools/diag_*.py`: label-free country checks.
  - `src/threshold.py`: exact metric, sweep, paired SE.

**What rerunning costs** (reusable checkpoints: S0, S1, S2 with 80 stored candidates, S3):

| Change touches | Must rerun | Approximate time |
|---|---|---|
| Combiner, threshold, band only | S4x scoring (partly), S5 | 40–100 min |
| Cross-encoder model | S4x (train + score), S5 | 1.5–2.5 h with the reranker (*estimate*) |
| Features / cap | S3 train + test, S4, S4x, S5 | + 20 min on top of the above |
| Normalization (S0) or blocking (S2) | everything | ≈ 3.5–4.5 h (too late after about 11:00 UTC) |

## 11. Where we most want ideas (ranked by loss share)

1. **France without labels.** Normalization that is neutral for US/India but fixes France
   formats: region vs department, number prefixes like N°/Nº, legal-form variants like
   S.A.R.L./5AS, abbreviations like Q./Rte./All. Also retrieval under heavy
   "{City} {Word}" name collisions, and making the matcher rely less on country-specific
   regularities.
2. **Blocking misses (≈.010, half of the remaining loss once the reranker is in).**
   - Recover true pairs beyond position 80 cheaply: typo- and transliteration-robust keys,
     house number + street keys, an embedding view.
   - We can re-block only if the gain is clearly worth about 1 h.
3. **Matcher errors (≈.010 of false positives + missed matches with the reranker; ≈.012 with
   e5-small).**
   - Better cross-encoder training: hard-negative mining, more epochs or data, input format,
     max length.
   - Better use of the cross-encoder together with the reverse-competition context.
4. **Speed.** Ways to score 5–9M test pairs with a 568M model inside about 90 min on one L4:
   distillation, ONNX/TensorRT, pruning the band by candidate rank. Only if they preserve
   accuracy.

## 12. Response format (please follow exactly; Markdown)

```
## Summary
<3–5 lines: your top recommendation and why>

## Proposals (ranked by expected LB gain per hour)
### P1 — <title>
- Targets: <loss part (§5) and countries>
- Mechanism: <why it should help; cite numbers from this brief or the code>
- Expected gain: <ΔF on VAL / on LB, with a range and your confidence>
- Cost: <stages to rerun (§10), GPU/CPU minutes on our VM, extra memory>
- Risk: <what could go wrong; how it could hurt US/India or France>
- Validation: <exact VAL measurement; the label-free France check if relevant>
- Implementation: <unified diff against bundle paths (src/…, tools/…, config.yaml), or precise
  pseudo-code naming the file and function>
### P2 — …

## Bugs, leakage or metric problems found
<file:line, what's wrong, the fix>

## Things we should stop doing or simplify
<optional>
```

Rules for the answer:
- No external data, lookups or network at inference.
- No models over 8B or outside MIT/Apache-2.0.
- No `country` feature and no per-country thresholds.
- Keep diffs minimal and consistent with the surrounding code style: polars + numpy, config in
  `config.yaml`, checkpointed stages.

## 13. Bundle map

- `HANDOFF.md`: this brief.
- `CLAUDE.md`: project rules and constraints.
- `specs/*.md`: per-stage specs (prepare, blocking, features, train, output, infra,
  optional-stages).
- `src/`:
  - `s0_prepare.py`: normalization.
  - `translit.py`: transliteration.
  - `s1_gt.py`: ground truth and splits.
  - `s2_block.py`: blocking.
  - `s3_features.py`: features.
  - `s4_train.py`: GBM and threshold sweep.
  - `threshold.py`: exact metric, sweep, selection, injective pass.
  - `s4x_xenc.py`: cross-encoder and combiner.
  - `s5_infer.py`: test inference and TSVs.
  - `common.py`: config, checkpoints, S3 sync.
  - Optional stages: `s2e_embed.py`, `s4l_llm.py`.
- `tools/`: experiment harnesses and diagnostics (§10).
- `config.yaml` (current), `config.v1.yaml` (frozen baseline), `run_all.sh`, `preflight.sh`.
- `docs/PROGRESS.md`: step log with every measurement. `docs/eda-findings.md`: EDA numbers.
- `submissions/1/README.md`, `s4x_report.json`: submission-1 record.
- Not included:
  - data, model weights, checkpoints, outputs, logs;
  - a third party's reference code and notes about it (we used it for ideas only).
