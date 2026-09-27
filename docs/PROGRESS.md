# PROGRESS — updated by Claude after every step; injected at every session start

## Order of work (stop and report after each)
1. `/kickoff`: scaffold, preflight, GT injectivity check.
2. S0 + S1 implemented and run (full); S2 on a 50K VAL slice → EDA-14 recall curve + test-time extrapolation.
3. S3 + S4 (slice → full 400K scope): VAL/HOLDOUT F ± SE, loss decomposition, backend benchmark,
   then optional experiments (rc_, emit gate, e5/bge emb_cos, cross-encoder, Qwen) → decide run-1 contents.
4. Run 1: clean `run_all.sh --force all` (≤ 8 h) → submission gate → package → submission 1 (user uploads).
5. User buffer (≤ 4 h): LB feedback → infer France F → edits; VM switch if needed (restore from S3).
6. Run 2: resume from run-1 checkpoints (≤ 4 h) → gate → submission 2 only if > 2 paired SE better on HOLDOUT.
7. `/package <team>` final zip.

## Status (read this first)
- **0.99 is the blocking ceiling, not a tuning target.** With the reranker, HOLDOUT loss is .0212 and
  half of it (.0104) is pairs that are not in the candidate list at all. A perfect matcher on today's
  candidates scores about **.990** on VAL (India's own oracle is .9827, so a test mix that is 47% India
  cannot clear .99 even with a perfect matcher unless retrieval gets better). Public LB sits ~.01 below
  HOLDOUT because France is unseen. **+0.15 is impossible** (the score is already ~.967; the maximum is 1).
- The +0.002 the user saw is the whole measured gain of the reranker (HOLDOUT +.0024). Every other
  measured idea left is smaller: stage-2 GBM +.0012 on top of the cross-encoder, cap-80 oracle +.0009,
  embedding view oracle +.0016.
- **Next change (coded, not run):** number-street reserved slots in S2. See "Retrieval channel" below.
  Run the 50K slice first and abort if it does not rescue a meaningful share of the misses.

## History
- Current step: 4 DONE (2026-09-27 05:41). Run 1 passed the submission gate -> submission-1 candidate in
  submissions/1/ (README has md5s, metrics, France formula). Waiting on: user uploads + reports public LB.
- Run 1: clean `run_all.sh --force all`, 02:56-05:03 UTC = 2 h 07 min wall (stage sum 125.1 min) vs 8 h.
  VAL .97682±.00029 (India .96909, US .98198); HOLDOUT .97642±.00029 (India .96772±.00056, US .98222±.00031).
- Submission 1 public LB 0.965 (HOLDOUT .9764) -> F_Fr ≈ .906 if India/US hold (loss shares: India .0151,
  France .0141, US .0068). Label-free check (tools/diag_country.py): test India/US match HOLDOUT on predicted
  matches, empty rate, self-estimated F; France self-F .975 vs .986, 2x uncertain S1s, xenc/stage-1 disagree
  19.6% vs 12-15%. Obvious France pairs (exact name + same number) are fine (p1<.5: 0.6%, = US). Region vs
  department mismatch lowers a_contain/a_last2 for France but not p on obvious pairs. Gap = hard pairs.
- Next (step 5): run-2 contents = stronger cross-encoder first (the component that transfers), measured on VAL.
- **Run 2 prepared (2026-09-27 ~09:30 UTC, code only; nothing run yet).** Contents and procedure in
  "Run 2" below. Reviewer findings that drove it: France ≈ .906 is the largest loss share; the codebase has
  no correctness bug; the time sinks were infra (S4x retrain cascade, S5 score wipe on any t change).
- Run-2 candidates (measured): stage-2 GBM on top of xenc (+.0012 at band .05/.95), embed view (+.0016 oracle,
  needs re-blocking), max_candidates 80 (+.0009 oracle, S3-only), bge-m3 cross-encoder (unmeasured, ~4x slower).
- AWS session expired during run 1 (every sync failed; the log said exit=0 because `$?` followed `$(date)`;
  fixed in common.s3_sync). RESOLVED 06:11: user re-ran `aws login`; manual sync exit=0; S3 v1/ holds work/
  (179 objects, 13.15 GB; s3_features excluded by design, ~12 min to rebuild) + output/ (1.76 GB).

## Measured results
(stage | split | limit | seconds | extrapolated full | key metric)
- s0_prepare | train | 50K | 100.2 | ~2 min/split | pool 10.32M rows; pool dominates time
- s0_prepare | train | full | 121.4 | - | 12,527,040 rows
- s0_prepare | test | full | 123.3 | - | 11,702,133 rows; France pool 1,434,993
- s0_prepare v2 (reviewer fixes) | train/test | full | 122.8 / 127.7 | - | 0 non-word chars left in names/addresses
- s2_block | train | 50K VAL | 163.3 | S2a 300K ~6 min | recall .9717 @ 49.7 cands (US .985, India .952)
- s2_block | test | 30K | 129.6 | S2b full ~19 min (plan 30, kill 60) | France 49.7 cands/S1
- s1_gt | train | full | 3.7 | - | GT = CLAUDE.md facts exactly; split 180K/100K/20K (India 40%, US 60%)
- s1_gt v2 (HOLDOUT) | train | full | 4.5 | - | + holdout 100K (India 40,021 / US 59,979), old splits identical
- s2_block | train | ALL 2,206,821 S1 | 1768.5 | S2a 29.5 min | 139.5M rows stored (<= 80/S1, cand_pos)
- s2_block | test | 30K | 137.4 | S2b ~25 min (store 80) | 1.87M rows
- s3_features | train | 50K | 131.3 | S3a (400K) 3.6 min measured | ~200K pairs/s + ~1 min prep/country
- s3_features | test | 30K | 98.1 | S3b ~10-12 min (86M pairs) | France pool 1.43M, 55 cols
- s3_features | train | 400K scope | 217.6 | - | 19.9M pairs; recall@50 by split .971-.972
- s4_train | train | 50K | 102.0 | - | benchmark: LGBM VAL .9395±.0014 (19s train, 131s test pred est) vs XGB .9397±.0014 (9s, 45s): tie -> xgboost
- s4_train | train | 400K (XGB) | (crashed in ablation, GPU-check bug; fixed) | - | VAL .94843±.00045 (India .9314, US .9598), oracle .9899; loss: model_miss .026, FP .011, blocking .010, singleton .0044 (emit 7.9%)
- s2e bench (200K names, S2 busy on CPU): e5-small 6,606/s; bge-m3 2,831/s
- vLLM Qwen2.5-7B: 44 pairs/s (166 prompt tokens), load 145 s with the torch sampler. 2026-09-27: CUDA compiler
  set up for FlashInfer (tools/setup_llm_cuda.sh: pip nvcc pinned 13.4 -> 13.0.88 to match cu130 headers +
  .venv-llm/cuda shim): FlashInfer sampler active, 48.7 pairs/s, 64 s load.
- s4_train | train | 400K (XGB, rerun) | 801 (incl. ablation + LOCO) | - | VAL .94843±.00045, HOLDOUT .94836±.00045 (India .9310, US .9600); rc_ ablation +.00303±.00023; injective +.00005±.00002; LOCO: India from US-only .846 (-.085), US from India-only .929 (-.031)
- E4 emit gate (cross-fitted, VAL): +.00008±.00013 -> not significant, stays off (singleton emit 7.9% -> 6.6%, offset elsewhere)
- E5 cross-encoder e5-small (1M TRAIN pairs, 1 epoch, 1,000 pairs/s train = 17 min; score 2,485 pairs/s) + cross-fitted logistic combiner on band p in [.05,.95] (2.2% of pairs): VAL .97471±.00031 = +.02629±.00036 paired; HOLDOUT .9744 (report only). Loss: model_miss .026->.0086, FP .011->.0049, singleton emit 7.9%->3.0%; blocking_miss .0101 is now the largest part.
  Memorization check: band-pair AUC xenc .994 on pool names UNSEEN in TRAIN candidates vs .967 seen (stage-1 p .866/.859) -> generalizes; India .989, US .970.
- Head-to-head on the same 4,000 VAL S1s with band pairs (cross-fitted combiner): xenc +.0364±.0021;
  xenc+Qwen +.0357±.0021; Qwen alone +.0041±.0014 (61 pairs/s with 4-shot prefix cache) -> Qwen OUT.
- Embed view (e5-small names, exact GPU top-k; 8.71M unique train names embedded in 687 s = 12.7K/s):
  blocking-oracle F on EDA-14 50K slice .99002 -> +embed top-5 .99084 (+.0008, 52 cands), top-10 .99119
  (+.0012, 56), top-20 .9916 (+.0016, 65); cap 80 alone +.0009 (63). Deferred to run 2 (needs re-blocking).
- Stage-2 GBM (OOF stage-1 p + S1-group p context + sibling similarity to the S1's top candidate), XGB:
  VAL .95333 = +.00491±.00027 over stage 1; HOLDOUT +.00457 (report only). Stacking with xenc: see below.
- Wide xenc band [.01,.99] (experiment model): VAL .97682±.00029 (+.0284 vs stage 1; +.0021 vs band .05/.95),
  HOLDOUT .97642; singleton emit 2.2%, FP .0038, model_miss .0081 -> ADOPTED (train_band .05/.95, band .01/.99).
- s4x_xenc (production stage, band .05/.95 at that time) | train | full | 1244.6 | - | reproduces the experiment
  exactly: VAL .97471 (India .9649, US .9813), HOLDOUT .9744, t=.71 r=0 plateau [.57,.85]
- s5_infer smoke | test | 30K | 47.6 | - | France 3.32 matches/S1, 4.8% empty; India 3.32/5.9%; US 3.38/5.7%
  (train GT 3.46 / 5.6%). Band share on the test slice: France 8.2%/16.2% (.05-.95/.01-.99), India 6.5/13.7,
  US 5.1/10.2 vs VAL 2.2/5.1 -- inflated by slice-only rc_ context; S5 run-1 xenc time 30-90 min.

- RUN 1 (clean --force all, commit 24af83a) | train+test | full | 7506 stage-sum (wall 2 h 07 min) | - |
  s0 128+133, s1 4, s2a 1758, s3a 219, s2b 1073 (107.96M stored), s3b 536 (85.99M pairs, peak RSS 19 GB),
  s4 244, s4x 1329, s5 2082 (xenc 5,495,967 band pairs = 6.39% in 1902 s = 2,889 pairs/s).
  VAL .97682±.00029, HOLDOUT .97642±.00029 (identical to step 3). Test: France 3.12 matches/S1, 6.1% empty;
  India 3.24/6.3%; US 3.35/5.8%. Validator PASS (also --check-ids). S5 rerun (2089 s): md5 identical.

## Run 2 (prepared 2026-09-27; deadline 17:30 UTC; ONE submission left)
Changes (all committed in this tree; smoke-tested locally on sample rows, not yet run on the VM):
1. **Reranker into production** (`config.yaml`): `bge-reranker-v2-m3`, bs 64, lr 2e-5, 300K pairs = the
   rr300k recipe (VAL +.00228 ± .00013, HOLDOUT +.0024). `model_from: work/eval/xenc_exp_rr300k/model`
   makes S4x COPY those weights instead of retraining (34 min) — legitimate: trained on TRAIN pairs only,
   the combiner is refit on VAL. If the dir is missing, S4x trains the same recipe (preflight WARNs).
2. **France-format normalization, S0 v3** (`s0_prepare.py`, `translit.py`; label-free, both sides, all
   countries): (a) hand-written French region/department → one code (like US_STATES; PARIS/LOT/LOIRE
   excluded) so S1 "…Tourcoing, Hauts-de-France" and pool "…Tourcoing, Nord" agree on a_contain/a_jacc/
   a_last2 as US pairs do; (b) "N° 23"/"Nº 23"/"No 23" → "23" (only before a digit: US "N MAIN ST" kept;
   India "PLOT NO 5" → "PLOT 5" both sides); (c) Q/QU→QUAI, RTE→ROUTE, CRS→COURS, FG→FAUBOURG, BLD/BVD→BLVD,
   CHEM→CHEMIN, SQUARE→SQ, SAINT/SAINTE→ST/STE; (d) digit-for-letter legal forms count as suffixes
   ("5AS"→SAS, "1NC"→INC, "C0"→CO) and `skeleton()` folds digits in mixed tokens (TAV0DREX = TAVODREX) for
   C_all; (e) `_map_tokens` sorts phrases before words — polars `replace_many(leftmost=True)` is
   leftmost-FIRST (verified: "CORSE" listed before "CORSE DU SUD" wins), so order was latent-fragile.
   Smoke test (local, 14 rows): both France variants → `162 RUE DE LA BAILLE TOURCOING RGHDF`; US/India
   rows unchanged except the intended NO-prefix drop.
3. **S5 score cache** (`s5_infer.scores_dir`): per-part stage-1/xenc scores live in
   `work/test/s5_scores/<s4_hash>-<s4x_hash>/`, outside the dir `stage()` wipes. A threshold change or
   `--force s5_infer` now costs select+write (~3 min), not a 104-min reranker pass. Diag tools use
   `s5_infer.find_scores()` (falls back to the run-1 layout).
4. **Gate tooling**: S4x writes `work/eval/s4x_full/per_s1_f_{val,holdout}.parquet`;
   `tools/per_s1_f.py dump|compare` gives the paired SE vs submission 1. `s4x_xenc` resume asserts no
   band pair lacks a score (was `fill_null(0.0)`). `diag_country.py`: countries from data (was hard-coded),
   new `deep` column (kept matches at cand_pos ≥ 40, retrieval proxy), self-F caveat (ignores blocking).
5. Optional `band_top_k` (off): score only each S1's top-k band pairs; `tools/exp_bandcap.py` simulates
   it from the rr300k scores in minutes. Adopt only if dF_val ≥ −1 paired SE.

Procedure (≈ 3 h 10 machine time; every step logged):
```
# 0. BEFORE anything destructive (run 1's eval inputs are overwritten/cleared by --force all)
.venv/bin/python tools/per_s1_f.py dump submissions/1        # per-S1 F of submission 1 (asserts t == .70)
cp -al work work_run1 && cp -a output output_run1            # hardlink snapshot: 5-min fallback to run-1 checkpoints
.venv/bin/python tools/diag_xenc_country.py --exp rr300k     # label-free: does the reranker move France? (~4 min)
.venv/bin/python tools/exp_bandcap.py                        # optional: band_top_k (CPU, minutes)
# 1. run 2 (tmux; wait with tools/wait_stage.sh)
bash run_all.sh --force all
# 2. gate
.venv/bin/python tools/per_s1_f.py compare submissions/1 work/eval/s4x_full   # HOLDOUT dF > 2 paired SE?
.venv/bin/python tools/diag_country.py     # France: top_uncertain 1.3%→?, disagree 19.6%→?, deep 0.84%→?, mean_pred 3.12→?
rm work/test/s5_infer/_DONE.json && .venv/bin/python src/s5_infer.py --split test && md5sum output/*.tsv  # ~3 min
# 3. fallback if S0 v3 hurts HOLDOUT (> 2 SE) or the run is late (> ~13:30 UTC):
#    mv work work_run2 && mv work_run1 work; revert s0_prepare.py/translit.py; run_all.sh (S4x+S5 only, ~2 h)
```
Expected: HOLDOUT ≈ .9788 (reranker) ± neutral S0 change; France label-free stats move toward US/India.
Budget: S0 4.4 + S2 47 + S3 12.5 + S4 4 + S4x ~12 (copy + score 510K band pairs) + S5 ~110 (reranker
5.5M pairs at 880/s) ≈ 3 h 10. Stale `work/test/s5_scores/<tag>/` dirs are not auto-deleted (≈1 GB each).

## Retrieval channel (coded 2026-09-27, not run)
`blocking.num_street` in `src/s2_block.py`. The lexical top 50 stay at cand_pos 0..49. Up to 12 extra
pairs per S1 land at cand_pos 50..61 (8 from the S1's own house number + content street token, 4 from
the top lexical hit's address, i.e. sibling records). `max_candidates` 62, `store_candidates` 92.
Street types and `RG*` region codes do not count. Numbers with df outside [2, 8000] are not indexed
(df 1 already wins IDF; common numbers would be the EDA-11 join). New rows have null view scores and
`from_ns=1`, so S4 retrains. The reranker is still the copied rr300k weights.

Synthetic check (local, no data): a "162 … BAILLE TOURCOING" query retrieves the pool rows that share
162 and BAILLE, not the same number on a different street; df caps hold; cand_pos 0..49 is unchanged;
the sibling of the top hit is added; the top hit is not duplicated.

Abort rule on the 50K slice (`slice recall` log line): lexical top-50 recall must stay ~.9717, and
`rescued` must be large versus the 4,591 misses on that slice. A few hundred rescued pairs is the
cap-80 result we already measured (+.0009) — stop and restore the snapshot. Several thousand rescued
is the only outcome that can move the score by more than the reranker did.

```
cp -al work work_before_ns          # slice --force wipes work/train/s2_block
.venv/bin/python src/s2_block.py --split train --limit-s1 50000 --force
# then, only if rescued is large:
bash run_all.sh --from s2_block     # S0/S1 reused; S2 hash change reruns S2..S5
```
Watch S4x's band-pair count. Run-1 test band was 5.5M pairs / ~104 min at 880/s. If VAL's band is
several times the old ~0.3M, S5 will not finish in budget: set `band_top_k: 8` and rerun S4x+S5.
Do not submit unless HOLDOUT beats the reranker run by > 2 paired SE (`tools/per_s1_f.py compare`).

## Submission log
| # | Date | Contents | HOLDOUT F ± SE (India / US) | Test sanity (France) | md5 matching | Public LB |
|---|---|---|---|---|---|---|
| 1 | 2026-09-27 | run 1: XGB+rc_ + e5-small xenc band .01-.99 + injective, t .70 | .97642±.00029 (.9677 / .9822) | 3.12 / 6.1% empty | e5f67373… | **0.965** |

## Decisions (with reason)
- 2026-09-26 user: 20 h left in total (step 3 ≤ 4 h, run 1 ≤ 8 h, buffer 4 h, run 2 ≤ 4 h resuming);
  only 2 LB submissions left; target 0.98. budget_hours: 8.
- Blocking-oracle F0.5 on EDA-14 slice: 0.9900 @ cap 50 (US .9949, India .9827), 0.9909 @ 80 ->
  the matcher is the bottleneck; step-3 time goes to the matcher.
- HOLDOUT 100K train S1s (disjoint, TRAIN/VAL/ES unchanged: verified row counts + s1_row sums).
- Reference code ~/lb1_0.924_code: ideas only (docs/reference-approach.md). Honest VAL 0.908 <->
  public LB 0.924. Its singleton emit rate 19.5% (~1.1 pt loss) -> emit-gate experiment.
- Run 1 contents: stage-1 XGB + rc_ features + E5 cross-encoder (+.026, 73 SE) + injective. OUT: emit gate
  (n.s.), Qwen (no gain over xenc, 40x slower), embed view (+.0016 oracle; run-2 option), emb_cos.
- Models (all local, offline at inference): multilingual-e5-small (MIT), bge-m3 (MIT),
  Qwen2.5-7B-Instruct (Apache-2.0). vLLM only in .venv-llm.
- v1 plan per CLAUDE.md; embeddings / LLM triage off until evidence.
- GT injectivity: PASSED (0 duplicates) → injective: auto enabled.
- Indic suffix list approved by user 2026-09-26 (PRAIVET PRAIBHET PIRAIVET PRAIVATT LIMITET LIMITTAD
  LIMATID PRA LI ELAELAPI ELELPI); zero-width chars (U+200B-D, FEFF) deleted before punctuation.
- translit: lru_cache on _indic (27.3K -> 35.7K names/s/core); anusvara/candrabindu before p/b/m -> m.
- Tamil K/G: not folded. EDA-14: Tamil = 0.5% of GT pairs, 6/200 misses, no K/G confusion seen.
- S2 views: W_all + C_all (name + address in one vector), k=40 each, cap 50. Name-only views tie on
  duplicated names. embed_view stays off: 0 translated names among misses (EDA-14).
- s3_sync runs in the background (flock, quiet, logs/s3_sync.log). tmux does NOT inherit the Claude
  env: launches must pass `-e BER_S3_BUCKET -e RUN_ID` (run-stage skill updated).

## Kickoff results
- Preflight: 25 PASS, 2 WARN, 0 FAIL. All 7 TSV row counts match CLAUDE.md. AWS put/get OK.
- sparse_dot_topn 1.2.0 installed (preflight checks sp_matmul_topn). Stub _DONE.json removed from S3 (0 objects under v1/).
- GT injectivity: 0 pool ids under more than one S1 -> injective: auto (on).
- run_all.sh resume verified on stubs: rerun skips all; --force s2_block reruns only s2_block.
- data/ holds symlinks into student_resource/ (data/ is gitignored).

## Open issues / waiting on user
- S2 is deterministic for a fixed pool order (verified: identical rerun). A reordered pool changes
  only exact ties at the k boundary (20 pairs / 10 of 5,000 S1s).
