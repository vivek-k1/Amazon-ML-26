---
paths:
  - "src/s2e_embed.py"
  - "src/s4l_llm.py"
  - "models/**"
---

# Optional stages (off in v1; enable only with measured evidence and budget)

Each stage sits behind a flag in `config.optional`, has its own `_DONE.json`, and only
adds candidates or features; it never changes earlier stages.

## embed_view (sentence-transformers) → src/s2e_embed.py
- When: the EDA-14 misses (true pairs no lexical view retrieves) are large enough to
  matter, and the budget has room. Report recall with and without.
- Models (MIT/Apache-2.0, ≤ 8B, loaded from `models/<name>/` with `HF_HUB_OFFLINE=1`):
  `intfloat/multilingual-e5-small` (MIT, ~118M) first; `sentence-transformers/all-MiniLM-L6-v2`
  (Apache-2.0, faster, works on the romanized text); `BAAI/bge-m3` (MIT, 568M) only with a
  ~10 h budget (≈ 2–4 h of L4 time for ~22M records).
- Measure records/s on a 200K sample before committing. fp16, length-sorted batches.
- Per-country faiss-cpu indices on host RAM. Never search across countries. Its top-k
  becomes a 4th view; cosine becomes a feature.
- Run as a separate process, and confirm `nvidia-smi` shows the GPU memory released
  before any other GPU stage.

## llm_triage (Qwen2.5-7B-Instruct, Apache-2.0, vLLM on the L4) → src/s4l_llm.py
- Only on the GBM's uncertain band (`optional.llm_triage.band`) and only if that band is
  ≤ `max_pairs` on test (~30K at tens of pairs/s). Score it by the log-prob of yes/no,
  used as a feature for a second-stage model or a calibrated override, tuned on VAL.
- Separate process; the GPU must be free (no embedding or XGBoost job).

## Other upgrades, in order
1. Reverse-competition features: block all 2.2M train S1s so each pool record knows which
   S1s retrieved it and at what rank (also makes VAL mimic test for the injective pass).
2. A token-synonym table learned from train GT pairs (MH↔MAHARASHTRA), using training
   data only.

## Setup and GPU discipline (2026-09-26)
- Weights in `models/` (multilingual-e5-small MIT, bge-m3 MIT, Qwen2.5-7B-Instruct Apache-2.0),
  fetched once with `hf download`, no token. Runs set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- vLLM lives in `.venv-llm` (its own torch); `llm_triage.python` points at it. The main `.venv`
  never gets vLLM.
- One GPU job at a time: embeddings → cross-encoder → XGBoost → vLLM. Check `nvidia-smi` shows
  no compute apps before each.
- vLLM + FlashInfer JIT: run `bash tools/setup_llm_cuda.sh` once per VM. It pins the pip CUDA
  compiler in `.venv-llm` to torch's CUDA (13.0; CCCL requires nvcc == toolkit headers) and
  builds `.venv-llm/cuda` (a CUDA_HOME of symlinks, incl. lib64 and libcudart.so). s4l_llm.py
  uses it when present, else falls back to `VLLM_USE_FLASHINFER_SAMPLER=0`. No system CUDA.
  Measured Qwen2.5-7B bf16 on the L4, 2,000 prompts of ~166 tokens: FlashInfer sampler 48.7
  pairs/s, 64 s load; torch sampler 44.2 pairs/s, 145 s load. First JIT build ~104 s (cached
  in ~/.cache/flashinfer).

## cross_encoder (fine-tuned, `src/s4x_xenc.py`) — IN for run 1 (EDA-15: +0.0284 VAL)
- Backbone multilingual-e5-small (bge-m3 is a run-2 option). Trained one epoch (no early
  stopping) on TRAIN pairs with stage-1 p in `train_band` + TRAIN positives + random negatives
  (≤ `max_train`); input raw "name | address" of both records.
- Scores every pair with stage-1 p in `band` ([.01,.99]); a logistic combiner on [logit p,
  xenc, logit p·xenc] (cross-fitted over VAL S1s for the (t, r) choice, refit on all VAL for
  test) gives p'. Measured: train 1,000 pairs/s, score ~2,500 pairs/s on the L4.
- The trained weights live in `work/train/s4x_xenc/xenc` (resume) and are copied to
  `output/artifacts/xenc`. CUDA training is not bit-reproducible: run 2 must RESUME S4x (never
  `--force` it) unless it deliberately retrains; S5 reruns are byte-identical.

## emit_gate (S1-level, CPU)
- A small GBM on S1 aggregates (top-1/top-2 p, gap, n above t, n_cands, best name/address
  scores) predicting "≥ 1 GT match among the candidates"; suppress the S1's output below g.
  Cross-fitted over VAL S1s; g swept jointly with t. Targets singleton emissions.
