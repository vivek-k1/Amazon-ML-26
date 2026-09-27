#!/usr/bin/env python3
"""Per-S1 F0.5 of the CURRENT production system, and the paired comparison the submission gate
needs ("beats submission 1 on HOLDOUT by > 2 paired SE").

Run `dump` BEFORE run 2: `run_all.sh --force all` overwrites work/eval/s4_full/pred_*.parquet and
clears work/train/s4x_xenc/, after which run 1's per-S1 F can no longer be recomputed. S4x now
writes the same files for every later run (work/eval/s4x_full/per_s1_f_*.parquet).

usage: .venv/bin/python tools/per_s1_f.py dump submissions/1
       .venv/bin/python tools/per_s1_f.py compare submissions/1 work/eval/s4x_full
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


def dump(out_dir: Path):
    """Reproduces s4x_xenc.evaluate on the production artifacts (deterministic: same VAL folds,
    same sweep) and writes per_s1_f_{val,holdout}.parquet + a summary."""
    cfg = load_config()
    fin = json.loads(Path("output/artifacts/thresholds_final.json").read_text())
    s4_thr = json.loads(Path("output/artifacts/thresholds.json").read_text())
    lo, hi = fin["xenc"]["band"]
    top_k = fin["xenc"].get("band_top_k")
    scope = pl.read_parquet("work/train/s3_features/s1_scope.parquet")
    n_gt = pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").group_by("s1_row").len("n_gt")
    E = {}
    for name in ("val", "holdout"):
        sc, d = xe.ev_arrays(name, pl.read_parquet(f"work/eval/s4_full/pred_{name}.parquet"), scope, n_gt)
        inb = xe.band_mask(d, lo, hi, top_k)
        sj = d.select("s1_row", "pool_row").join(
            pl.read_parquet(f"work/train/s4x_xenc/scores_{name}.parquet").select("s1_row", "pool_row", "xenc"),
            on=["s1_row", "pool_row"], how="left", maintain_order="left")["xenc"]
        assert sj.filter(pl.Series(inb)).null_count() == 0, f"{name}: band pairs without cross-encoder scores"
        E[name] = {"sc": sc, "d": d, "inb": inb, "s": sj.fill_null(0.0).to_numpy(),
                   "y": d["label"].to_numpy().astype(np.float64), "p": d["p"].to_numpy()}
    rep, _, Fs = xe.evaluate(E, s4_thr, cfg["threshold"], fin["injective"])
    assert abs(rep["choice"]["t"] - fin["t"]) < 1e-9, f"re-derived t {rep['choice']['t']} != artifact {fin['t']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, F in Fs.items():
        pl.DataFrame({"s1_row": E[name]["sc"]["s1_row"], "country": E[name]["sc"]["country"], "f05": F}
                     ).write_parquet(out_dir / f"per_s1_f_{name}.parquet")
    summ = {k: {"f05": rep[k]["f05"], "se": rep[k]["se"], "by_country": rep[k]["by_country"]} for k in ("val", "holdout")}
    (out_dir / "per_s1_f_summary.json").write_text(json.dumps({**summ, "t": fin["t"], "r": fin["r"]}, indent=1))
    print(json.dumps(summ, indent=1))


def compare(a: Path, b: Path):
    """b - a by split and country: mean paired dF, paired SE, and whether |dF| > 2 SE."""
    for name in ("val", "holdout"):
        fa, fb = a / f"per_s1_f_{name}.parquet", b / f"per_s1_f_{name}.parquet"
        if not (fa.exists() and fb.exists()):
            print(f"{name}: missing {fa if not fa.exists() else fb}")
            continue
        j = pl.read_parquet(fa).join(pl.read_parquet(fb).select("s1_row", pl.col("f05").alias("f05_b")), on="s1_row")
        assert j.height == pl.read_parquet(fa).height == pl.read_parquet(fb).height, f"{name}: S1 scopes differ"
        rows = [("all", j)] + [(c, g) for (c,), g in j.group_by("country", maintain_order=True)]
        for c, g in sorted(rows, key=lambda x: (x[0] != "all", x[0])):
            d = (g["f05_b"] - g["f05"]).to_numpy()
            se = th.se(d)
            verdict = "PASS (> 2 paired SE)" if d.mean() > 2 * se else ("worse" if d.mean() < -2 * se else "n.s.")
            print(f"{name:8s} {c:8s} n={g.height:7,d}  F_a {g['f05'].mean():.5f}  F_b {g['f05_b'].mean():.5f}  "
                  f"dF {d.mean():+.5f} +/- {se:.5f}  {verdict if c == 'all' else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("dump"); p1.add_argument("out_dir", type=Path)
    p2 = sub.add_parser("compare"); p2.add_argument("a", type=Path); p2.add_argument("b", type=Path)
    a = ap.parse_args()
    dump(a.out_dir) if a.cmd == "dump" else compare(a.a, a.b)
