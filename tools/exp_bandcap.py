#!/usr/bin/env python3
"""Run-2 experiment (CPU, minutes): score only each S1's top-K band pairs (by stage-1 p) with the
cross-encoder; pairs outside keep p. Simulated exactly from an existing score dir, because
s4x_xenc.evaluate takes the band mask as an input. Reports the share of band pairs kept (= S5
reranker time) and the paired dF vs the uncapped band on VAL (HOLDOUT report only).

Adopt a K only if dF_val >= -1 paired SE; then set optional.cross_encoder.band_top_k in config.yaml
(S4x and S5 apply the same mask through s4x_xenc.band_mask).

usage: .venv/bin/python tools/exp_bandcap.py [--scores work/eval/xenc_exp_rr300k] [--k 12 8 6]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import s4x_xenc as xe          # noqa: E402
import threshold as th         # noqa: E402
from common import load_config  # noqa: E402


def main(a):
    cfg = load_config()
    xc = cfg["optional"]["cross_encoder"]
    lo, hi = a.band or xc["band"]
    inj = th.resolve_injective(cfg)
    s4_thr = json.loads(Path("output/artifacts/thresholds.json").read_text())
    scope = pl.read_parquet("work/train/s3_features/s1_scope.parquet")
    n_gt = pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").group_by("s1_row").len("n_gt")
    E = {}
    for name in ("val", "holdout"):
        sc, d = xe.ev_arrays(name, pl.read_parquet(f"work/eval/s4_full/pred_{name}.parquet"), scope, n_gt)
        s = (d.select("s1_row", "pool_row")
              .join(pl.read_parquet(Path(a.scores) / f"scores_{name}.parquet").select("s1_row", "pool_row", "xenc"),
                    on=["s1_row", "pool_row"], how="left", maintain_order="left")["xenc"].fill_null(0.0).to_numpy())
        E[name] = {"sc": sc, "d": d, "s": s, "y": d["label"].to_numpy().astype(np.float64), "p": d["p"].to_numpy()}
    res = {}
    F0 = None
    for K in [None] + a.k:
        Ek = {n: {**E[n], "inb": xe.band_mask(E[n]["d"], lo, hi, K)} for n in E}
        rep, _, F = xe.evaluate(Ek, s4_thr, cfg["threshold"], inj)
        full = xe.band_mask(E["val"]["d"], lo, hi, None)
        kept = float(Ek["val"]["inb"].sum() / full.sum())
        if F0 is None:
            F0 = F
        row = {"K": K, "band_pairs_kept": round(kept, 4), "val": rep["val"]["f05"], "holdout": rep["holdout"]["f05"],
               "t": rep["choice"]["t"], "dF_val": round(float((F["val"] - F0["val"]).mean()), 5),
               "paired_se_val": round(th.paired_se(F["val"], F0["val"]), 5),
               "dF_holdout_report_only": round(float((F["holdout"] - F0["holdout"]).mean()), 5),
               "val_by_country": {c: v["f05"] for c, v in rep["val"]["by_country"].items()}}
        res[str(K)] = row
        print(json.dumps(row), flush=True)
    out = Path("work/eval/bandcap.json")
    out.write_text(json.dumps(res, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="work/eval/xenc_exp_rr300k")
    ap.add_argument("--k", type=int, nargs="+", default=[12, 8, 6])
    ap.add_argument("--band", type=float, nargs=2)
    main(ap.parse_args())
