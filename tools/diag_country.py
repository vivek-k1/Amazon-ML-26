#!/usr/bin/env python3
"""Label-free per-country diagnostics: HOLDOUT (has labels) vs test (none), same statistics.

For each (split, country): band share, S1 confidence profile, cross-encoder vs stage-1
disagreement, predicted matches, and a self-estimated F0.5 from the final probabilities
(E[F] ~ 1.25 sum_{kept} p / (0.25 sum_all p + n_kept)). CAVEAT: the self-estimate sees only the
candidates, never blocking misses, so it OVERSTATES F by the blocking loss (HOLDOUT: India .987 vs
true .968, US .985 vs .982). Read it with `deep` (kept matches at cand_pos >= 40, a retrieval
proxy) next to it; France's deep share is 2-8x India/US's. On HOLDOUT the true F is printed too.

usage: .venv/bin/python tools/diag_country.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import s4x_xenc as xe          # noqa: E402
import threshold as th         # noqa: E402
from s5_infer import find_scores   # noqa: E402

thr = json.loads(open("output/artifacts/thresholds_final.json").read())
LO, HI = thr["xenc"]["band"]
T, R, INJ = thr["t"], thr["r"], thr["injective"]


def final_p(d: pl.DataFrame) -> pl.DataFrame:
    inb = (d["p"].is_between(LO, HI) & d["xenc"].is_not_null()).to_numpy()   # HOLDOUT file stores xenc=0 off-band
    p2 = xe.combine_apply(d["p"].to_numpy(), d["xenc"].fill_null(0.0).to_numpy(), inb, thr["xenc"]["combiner"])
    return d.with_columns(pl.Series("pf", p2), pl.Series("inb", inb))


def deep_share(d: pl.DataFrame, split: str) -> pl.DataFrame:
    """Share of kept matches at cand_pos >= 40 (needs the split's S3 parts; null if deleted)."""
    parts = list(Path(f"work/{split}/s3_features").glob("part-*.parquet"))
    kept = d.filter(pl.col("keep")).select("s1_row", "pool_row", "country")
    if not parts:
        return kept.group_by("country").agg(pl.lit(None, pl.Float64).alias("deep"))
    cp = pl.scan_parquet([str(p) for p in parts]).select("s1_row", "pool_row", "cand_pos")
    return (kept.lazy().join(cp, on=["s1_row", "pool_row"], how="left").collect()
                .group_by("country").agg((pl.col("cand_pos") >= 40).mean().round(4).alias("deep")))


def stats(d: pl.DataFrame, s1: pl.DataFrame, label: str):
    s1 = s1.sort("s1_row").with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("idx"))
    d = d.join(s1.select("s1_row", "idx", "country"), on="s1_row").sort("s1_row", "pool_row")
    keep = th.select(th.Cands(d["idx"].to_numpy(), d["pool_row"].to_numpy(), d["pf"].to_numpy(), s1.height), T, R, INJ)
    d = d.with_columns(pl.Series("keep", keep))
    g = (d.group_by("idx").agg(
            pl.col("pf").max().alias("top"),
            pl.col("keep").sum().alias("n_pred"),
            (pl.col("pf") * pl.col("keep")).sum().alias("sp_kept"),
            pl.col("pf").sum().alias("sp_all"),
            pl.col("inb").sum().alias("n_band"))
         .join(s1.select("idx", "country"), on="idx", how="right")
         .with_columns(pl.col("n_pred", "sp_kept", "sp_all", "n_band", "top").fill_null(0)))
    # self-estimated F per S1 (singleton case: F = 1 if nothing kept, weighted by P(no match))
    ef = pl.when(pl.col("n_pred") == 0).then((-pl.col("sp_all")).exp()).otherwise(
        1.25 * pl.col("sp_kept") / (0.25 * pl.col("sp_all") + pl.col("n_pred")))
    g = g.with_columns(ef.clip(0, 1).alias("selfF"))
    band = d.filter(pl.col("inb"))
    dis = band.group_by("country").agg(
        ((pl.col("xenc") > 0) != (pl.col("p") >= 0.5)).mean().alias("xenc_vs_p1_disagree"),
        (pl.col("xenc") > 0).mean().alias("xenc_pos_rate"))
    out = (g.group_by("country").agg(
              pl.len().alias("s1"),
              pl.col("n_pred").mean().round(3).alias("mean_pred"),
              (pl.col("n_pred") == 0).mean().round(4).alias("empty"),
              ((pl.col("top") > 0.2) & (pl.col("top") < 0.9)).mean().round(4).alias("top_uncertain"),
              (pl.col("n_band") > 0).mean().round(4).alias("s1_with_band"),
              pl.col("selfF").mean().round(4).alias("selfF"))
           .join(d.group_by("country").agg(pl.col("inb").mean().round(4).alias("band_share"),
                                            (pl.len() / pl.col("idx").n_unique()).round(1).alias("cands")),
                 on="country")
           .join(dis, on="country")
           .join(deep_share(d, "train" if label == "holdout" else label), on="country", how="left")
           .sort("country")
           .with_columns(pl.lit(label).alias("split")))
    return out, d, s1


def main():
    rows = []
    # HOLDOUT (labels available)
    ho = pl.read_parquet("work/eval/s4_full/pred_holdout.parquet")
    xs = pl.read_parquet("work/train/s4x_xenc/scores_holdout.parquet")
    s1h = (pl.read_parquet("work/train/s3_features/s1_scope.parquet").filter(pl.col("split") == "holdout")
             .select("s1_row").join(pl.read_parquet("work/train/s0_prepare/s1.parquet", columns=["s1_row", "country"]), on="s1_row"))
    d = final_p(ho.join(xs, on=["s1_row", "pool_row"], how="left"))
    out, dh, s1h = stats(d, s1h, "holdout")
    n_gt = s1h.join(pl.read_parquet("work/train/s1_gt/gt_pairs.parquet").group_by("s1_row").len("n_gt"),
                    on="s1_row", how="left").fill_null(0)["n_gt"].to_numpy()
    F = th.per_s1_f05(n_gt, dh["idx"].to_numpy(), dh["keep"].to_numpy(), dh["label"].to_numpy(), s1h.height)
    cc = s1h["country"].to_numpy()
    trueF = {c: round(float(F[cc == c].mean()), 4) for c in sorted(set(cc.tolist()))}   # countries from the data
    rows.append(out.with_columns(pl.col("country").replace_strict(trueF, default=None).alias("trueF")))
    # TEST (no labels)
    w = find_scores("test")
    pr = pl.read_parquet(f"{w}/pred-part-*.parquet")
    xt = pl.read_parquet(f"{w}/xenc-part-*.parquet")
    s1t = pl.read_parquet("work/test/s0_prepare/s1.parquet", columns=["s1_row", "country"])
    out, _, _ = stats(final_p(pr.join(xt, on=["s1_row", "pool_row"], how="left")), s1t, "test")
    rows.append(out.with_columns(pl.lit(None, pl.Float64).alias("trueF")))
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    print(pl.concat(rows).select("split", "country", "s1", "trueF", "selfF", "deep", "mean_pred", "empty",
                                 "top_uncertain", "s1_with_band", "band_share", "cands", "xenc_pos_rate",
                                 "xenc_vs_p1_disagree"))


if __name__ == "__main__":
    main()
