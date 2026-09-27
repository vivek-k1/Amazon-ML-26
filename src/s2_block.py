#!/usr/bin/env python3
"""S2 blocking: per-country IDF-weighted sparse top-k retrieval. Rule: .claude/rules/blocking.md.

Per country partition: pool = S2+S3 rows, queries = S1 rows. Each view is a binary-tf,
IDF-weighted, L2-normalized sparse matrix; top-k per view by sparse_dot_topn; candidates are
the union of the views, ordered and capped at blocking.store_candidates (cand_pos kept;
S3 applies max_candidates, so run 2 can change it without re-blocking).

Number-street channel (blocking.num_street): EDA-14 misses are mostly a typo'd or
transliterated name with a shared address, and name collisions fill the top 50 before that
address match is reached (cap 80 recovers only +.0009 oracle). Extra slots, placed after
primary_keep so the current top 50 are never dropped, hold pool rows that share a mid-frequency
house number and a content street token with the S1 — or with the S1's top lexical hit, which
catches sibling records once one match was found.
"""

import argparse
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from common import atomic_write_parquet, code_hash, input_hash, load_config, stage
from s1_gt import stratified

log = logging.getLogger("s2_block")
VIEWS = ("W_all", "C_all", "W_name", "W_addr", "C_name")
COLS = ["country", "name_s", "name_sk", "addr_n", "addr_nums", "name_script"]


def view_tokens(df: pl.DataFrame, view: str, vcfg: dict) -> pl.DataFrame:
    """Unique (idx, tok) pairs: binary tf."""
    name = pl.concat_list(pl.col("name_s").str.split(" "), pl.col("name_s").str.extract_all(r"\d+"))
    addr = pl.concat_list(   # numbers only match numbers: '#' prefix
        pl.col("addr_n").str.split(" ").list.eval(pl.element().filter(pl.element().str.contains(r"\p{L}"))),
        pl.col("addr_nums").list.eval(pl.lit("#") + pl.element()))
    if view == "W_name":
        toks = name
    elif view == "W_addr":
        toks = addr
    elif view == "W_all":    # names repeat across businesses: rank by name AND address evidence
        toks = pl.concat_list(name.list.eval(pl.lit("n:") + pl.element()),
                              addr.list.eval(pl.lit("a:") + pl.element()))
    elif view in ("C_name", "C_all"):
        ng = vcfg.get("ngram", 3)
        s = df.select("idx", pl.col("name_sk").str.replace_all(" ", "").alias("s"))
        s = s.filter(pl.col("s").str.len_chars() > 0).with_columns(
            # len_chars is UInt32: cast before subtracting or short names wrap to ~4e9 (OOM)
            pl.int_ranges(0, pl.max_horizontal(pl.col("s").str.len_chars().cast(pl.Int64) - ng + 1, 1))
              .alias("off"))
        grams = s.explode("off").select("idx", pl.col("s").str.slice(pl.col("off"), ng).alias("tok"))
        if view == "C_name":
            return grams.unique()
        # typos / transliteration in the name, anchored by the address
        a = (df.select("idx", addr.alias("tok")).explode("tok")
               .filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")))
        return pl.concat([grams.with_columns((pl.lit("c:") + pl.col("tok")).alias("tok")),
                          a.with_columns((pl.lit("a:") + pl.col("tok")).alias("tok"))]).unique()
    else:
        raise ValueError(view)
    return (df.select("idx", toks.alias("tok")).explode("tok")
              .filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")).unique())


def l2_weights(t: pl.DataFrame) -> pl.DataFrame:
    # sorted so float sums (and so tie-breaks between candidates) are identical run to run
    return t.sort("idx", "fid").with_columns((pl.col("idf") / (pl.col("idf") ** 2).sum().over("idx").sqrt())
                          .cast(pl.Float32).alias("w"))


def build_pool_matrix(ptok: pl.DataFrame, n_pool: int):
    vocab = (ptok.group_by("tok").len("df").sort("tok")      # group_by order is random
                 .with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias("fid"),
                               (np.log(n_pool) - pl.col("df").log()).alias("idf")))
    t = l2_weights(ptok.join(vocab, on="tok"))
    P = sp.csr_matrix((t["w"].to_numpy(), (t["idx"].to_numpy(), t["fid"].to_numpy())),
                      shape=(n_pool, vocab.height), dtype=np.float32)
    PT = P.T.tocsr()
    PT.sort_indices()
    return PT, vocab                      # (features x pool) for A @ B


def query_matrix(qtok, vocab, n_q, cap):
    t = l2_weights(qtok.join(vocab, on="tok"))   # norm over all in-vocab features, then cap
    t = t.filter(pl.col("df") <= cap)
    Q = sp.csr_matrix((t["w"].to_numpy(), (t["idx"].to_numpy(), t["fid"].to_numpy())),
                      shape=(n_q, vocab.height), dtype=np.float32)
    Q.sort_indices()
    return Q, int(t["df"].cast(pl.Int64).sum() or 0)


def order_and_cap(wide: pl.DataFrame, views, max_c: int) -> pl.DataFrame:
    """Rank candidates per S1: best per-view rank, then #views, then best score; keep max_c."""
    ranks = [pl.col(f"rank_{v}") for v in views]
    scores = [pl.col(f"score_{v}") for v in views]
    return (wide.with_columns(pl.min_horizontal(ranks).alias("_best_rank"),
                              pl.sum_horizontal([r.is_not_null() for r in ranks]).alias("_nv"),
                              pl.max_horizontal(scores).alias("_best_score"))
                .sort(["s1_row", "_best_rank", "_nv", "_best_score", "pool_row"],
                      descending=[False, False, True, True, False])
                .with_columns(pl.int_range(pl.len()).over("s1_row").alias("_pos"))
                .filter(pl.col("_pos") < max_c))


# Street-type words. Directions and city names stay: with a mid-frequency house number they
# are real evidence. Region codes (RG*) are the France fold from S0 and are not a street.
NS_STOP = ["ROAD", "STREET", "LANE", "AVENUE", "BOULEVARD", "DRIVE", "PLACE", "COURT",
           "SQUARE", "ROUTE", "CHEMIN", "QUAI", "COURS", "ALLEE", "IMPASSE", "FAUBOURG",
           "NAGAR", "MARG", "COLONY", "LAYOUT", "CROSS", "BLOCK", "SECTOR", "PHASE",
           "FLOOR", "HOUSE", "BUILDING", "COMPLEX", "MARKET", "GANJ", "PETH", "PLOT",
           "FLAT", "SUITE", "UNIT", "NEAR", "OPPOSITE", "BEHIND", "BESIDE"]


def _street_list():
    e = pl.element()
    return (pl.col("addr_n").fill_null("").str.split(" ")
            .list.eval(e.filter((e.str.len_chars() >= 4) & ~e.is_in(NS_STOP) & ~e.str.starts_with("RG"))))


def build_num_street(pc: pl.DataFrame, ns: dict):
    """Postings for house numbers with df in [min_df, max_df], plus content street tokens of
    those pool rows. df 1 already wins the IDF views; very common numbers would be a token join
    of the kind EDA-11 forbids, so they are not indexed."""
    nums = (pc.select("pool_row", pl.col("addr_nums").alias("num")).explode("num")
              .filter(pl.col("num").is_not_null() & (pl.col("num") != "")).unique())
    nums = nums.join(nums.group_by("num").len("df"), on="num").filter(
        pl.col("df").is_between(ns["min_df"], ns["max_df"]))
    st = (pc.select("pool_row", _street_list().alias("tok")).explode("tok")
            .filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")).unique()
            .join(nums.select("pool_row").unique(), on="pool_row"))
    return nums.select("num", "pool_row", "df"), st


def _take(d: pl.DataFrame, k: int) -> pl.DataFrame:
    if d.height == 0 or k <= 0:
        return d.clear()
    return (d.sort(["s1_row", "ns_shared", "tok_len", "pool_row"],
                   descending=[False, True, True, False])
             .with_columns(pl.int_range(pl.len()).over("s1_row").alias("_r"))
             .filter(pl.col("_r") < k))


def num_street_hits(q: pl.DataFrame, pool_nums: pl.DataFrame, pool_st: pl.DataFrame,
                    keep: int, min_street: int, batch: int = 2000) -> pl.DataFrame:
    """Up to `keep` pool rows per S1 sharing one indexed house number and >= min_street content
    street tokens. Batched, and a batch whose postings exceed 5M rows falls back to each S1's
    rarest number, so this cannot become the uncapped token join EDA-11 measured."""
    schema = {"s1_row": q.schema["s1_row"], "pool_row": pool_nums.schema["pool_row"],
              "ns_shared": pl.UInt32, "tok_len": pl.UInt32, "_r": pl.UInt32}
    if q.height == 0 or pool_nums.height == 0 or keep <= 0:
        return pl.DataFrame(schema=schema)
    df_of = pool_nums.select("num", "df").unique()
    out = []
    for off in range(0, q.height, batch):
        qb = q.slice(off, batch)
        qn = (qb.select("s1_row", pl.col("addr_nums").alias("num")).explode("num")
                .filter(pl.col("num").is_not_null() & (pl.col("num") != "")).unique())
        qn = qn.join(df_of, on="num")
        if qn.height == 0:
            continue
        if int(qn["df"].sum()) > 5_000_000:          # rarest number only: bounds the posting join
            qn = qn.sort(["s1_row", "df", "num"]).unique("s1_row", keep="first")
        hits = qn.select("s1_row", "num").join(pool_nums.select("num", "pool_row"), on="num")
        qs = (qb.select("s1_row", _street_list().alias("tok")).explode("tok")
                .filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")).unique())
        if qs.height == 0 or hits.height == 0:
            continue
        ps = pool_st.join(hits.select("pool_row").unique(), on="pool_row")
        sh = (hits.select("s1_row", "pool_row").unique().join(qs, on="s1_row").join(ps, on=["pool_row", "tok"])
                  .group_by("s1_row", "pool_row")
                  .agg(pl.col("tok").n_unique().cast(pl.UInt32).alias("ns_shared"),
                       pl.col("tok").str.len_chars().sum().cast(pl.UInt32).alias("tok_len"))
                  .filter(pl.col("ns_shared") >= min_street))
        if sh.height:
            out.append(_take(sh, keep))
    return pl.concat(out) if out else pl.DataFrame(schema=schema)


def attach_num_street(capped, q, pc, index, views, ns):
    """Insert up to `reserve` number-street hits at cand_pos [primary_keep, primary_keep+reserve).
    The lexical top `primary_keep` are left where they are; the lexical tail shifts up by `reserve`
    so S3's max_candidates keeps the new rows and still drops that tail."""
    primary_keep, reserve = ns["primary_keep"], ns["reserve"]
    n_sib = min(int(ns.get("sibling_top1") or 0), reserve)
    n_q = reserve - n_sib
    pool_nums, pool_st = index
    ban = capped.filter(pl.col("cand_pos") < primary_keep).select("s1_row", "pool_row")
    extra = num_street_hits(q.select("s1_row", "addr_n", "addr_nums"), pool_nums, pool_st,
                            n_q * 5, ns["min_street"]).join(ban, on=["s1_row", "pool_row"], how="anti")
    extra = _take(extra, n_q).with_columns((pl.lit(primary_keep) + pl.col("_r")).alias("cand_pos"))
    if n_sib:
        top1 = capped.filter(pl.col("cand_pos") == 0).select("s1_row", pl.col("pool_row").alias("src"))
        q2 = top1.join(pc.select(pl.col("pool_row").alias("src"), "addr_n", "addr_nums"), on="src")
        sib = num_street_hits(q2.select("s1_row", "addr_n", "addr_nums"), pool_nums, pool_st,
                              n_sib * 5, ns["min_street"])
        sib_ban = pl.concat([ban, extra.select("s1_row", "pool_row"),
                             top1.select("s1_row", pl.col("src").alias("pool_row"))]).unique()
        sib = _take(sib.join(sib_ban, on=["s1_row", "pool_row"], how="anti"), n_sib)
        sib = sib.with_columns((pl.lit(primary_keep + n_q) + pl.col("_r")).alias("cand_pos"))
        extra = pl.concat([extra, sib], how="diagonal_relaxed")
    head = capped.filter(pl.col("cand_pos") < primary_keep).with_columns(pl.lit(0, pl.Int8).alias("ns"))
    tail = (capped.filter(pl.col("cand_pos") >= primary_keep)
                 .with_columns((pl.col("cand_pos") + reserve).cast(pl.Int16).alias("cand_pos"),
                               pl.lit(0, pl.Int8).alias("ns")))
    if extra.height == 0:
        return pl.concat([head, tail], how="diagonal_relaxed"), 0
    add = extra.select(
        "s1_row", "pool_row", pl.col("cand_pos").cast(pl.Int16),
        *[pl.lit(None, dtype=pl.Float32).alias(f"score_{v}") for v in views],
        *[pl.lit(None, dtype=pl.Int16).alias(f"rank_{v}") for v in views],
        pl.lit(1, pl.Int8).alias("ns"))
    return pl.concat([head, add, tail], how="diagonal_relaxed"), add.height


def block_chunk(q: pl.DataFrame, pools: dict, views, cfg_views, k_over, nw, max_c, stats):
    q = q.with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("idx"))
    s1_rows = q["s1_row"].to_numpy()
    wide = None
    for v in views:
        PT, vocab, pool_rows, n_pool = pools[v]
        k = k_over or cfg_views[v]["k"]
        cap = cfg_views[v]["skip_cap_frac"] * n_pool
        t = time.time()
        Q, postings = query_matrix(view_tokens(q, v, cfg_views[v]), vocab, q.height, cap)
        C = sp_matmul_topn(Q, PT, top_n=k, n_threads=nw).tocoo()
        dt = time.time() - t
        stats[v]["seconds"] += dt
        stats[v]["postings"] += postings
        stats[v]["queries"] += q.height
        f = (pl.DataFrame({"s1_row": s1_rows[C.row], "pool_row": pool_rows[C.col],
                           f"score_{v}": C.data.astype(np.float32)})
               .sort(["s1_row", f"score_{v}", "pool_row"], descending=[False, True, False])
               .with_columns((pl.int_range(pl.len()).over("s1_row") + 1).cast(pl.Int16)
                             .alias(f"rank_{v}")))
        wide = f if wide is None else wide.join(f, on=["s1_row", "pool_row"], how="full",
                                                coalesce=True)
    return wide


def select_queries(split, limit_s1, seed):
    s1 = pl.read_parquet(f"work/{split}/s0_prepare/s1.parquet", columns=["s1_row"] + COLS)
    if split == "train" and limit_s1:     # tuning slice: a stratified sample of VAL only
        ids = stratified(pl.read_parquet("work/train/s1_gt/val_ids.parquet"), limit_s1, seed)
        s1 = s1.join(ids.select("s1_row"), on="s1_row")
    elif limit_s1:
        s1 = stratified(s1, limit_s1, seed)
    return s1.sort("s1_row")


def safe(c: str) -> str:
    """File-safe country tag; the hash keeps two countries from sharing part files."""
    return re.sub(r"[^A-Za-z0-9]+", "_", c) + "-" + hashlib.md5(c.encode()).hexdigest()[:6]


def main(split, limit_s1, force, k_override=None):
    cfg = load_config()
    bc = cfg["blocking"]
    views = [v for v in VIEWS if bc["views"][v]["enabled"]]
    inputs = {"s0_prepare": input_hash(split, "s0_prepare")}
    if split == "train" and limit_s1:     # full train blocks every train S1, split-independent
        inputs["s1_gt"] = input_hash("train", "s1_gt")
    # max_candidates is applied in S3 (parts keep store_candidates), so it stays out of the hash;
    # chunk_s1 decides which S1s land in which part, so it must invalidate parts.
    # num_street stays IN the hash: it changes which pairs the parts contain.
    ns = bc.get("num_street") or {}
    reserve = int(ns.get("reserve") or 0) if ns.get("enabled") else 0
    if reserve:
        assert bc["max_candidates"] >= ns["primary_keep"] + reserve, "max_candidates must keep the reserved slots"
        assert bc["store_candidates"] - reserve >= ns["primary_keep"], "store_candidates must hold the lexical tail"
    bc_hash = {k: v for k, v in bc.items() if k != "max_candidates"}
    section = {"blocking": bc_hash, "limit_s1": limit_s1, "k_override": k_override,
               "chunk_s1": cfg["chunk_s1"], "seed": cfg["seed"], "code": code_hash("src/s2_block.py")}
    with stage("s2_block", split, cfg, section, limit_s1=limit_s1, force=force,
               input_stages=inputs) as st:
        if st.skip:
            return
        nw, chunk = cfg["n_workers"], cfg["chunk_s1"]
        # uncapped per-view top-k is kept for the tuning eval; the parts hold the capped union
        keep_uncapped = split == "train" and limit_s1 is not None
        # lexical rows stored before the reserved slots are inserted (the tail shifts by `reserve`)
        max_c = bc["store_candidates"] - reserve
        queries = select_queries(split, limit_s1, cfg["seed"])
        # the exact S1 universe, incl. S1s that get zero candidates (S3/S4 denominators)
        atomic_write_parquet(queries.select("s1_row", "country"), st.work_dir / "queries.parquet")
        pool_all = pl.read_parquet(f"work/{split}/s0_prepare/pool.parquet",
                                   columns=["pool_row"] + COLS)
        log.info(f"queries {queries.height:,}; pool {pool_all.height:,}; views {views}; "
                 f"k={k_override or {v: bc['views'][v]['k'] for v in views}}; cap {max_c}")
        report = {"views": views, "countries": {}}
        total_rows, uncapped = 0, []
        for country in queries["country"].unique().sort().to_list():
            qc = queries.filter(pl.col("country") == country)
            n_chunks = (qc.height + chunk - 1) // chunk
            parts = [st.work_dir / f"part-{safe(country)}-{i:04d}.parquet" for i in range(n_chunks)]
            if all(p.exists() for p in parts) and not keep_uncapped:
                log.info(f"[{country}] all {n_chunks} parts exist, skipping")
                total_rows += sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in parts)
                report["countries"][country] = {"queries": qc.height, "skipped_existing_parts": n_chunks}
                continue
            pc = (pool_all.filter(pl.col("country") == country)
                          .with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("idx")))
            t = time.time()
            pools = {}
            for v in views:
                PT, vocab = build_pool_matrix(view_tokens(pc, v, bc["views"][v]), pc.height)
                pools[v] = (PT, vocab, pc["pool_row"].to_numpy(), pc.height)
            build_s = time.time() - t
            stats = {v: {"seconds": 0.0, "postings": 0, "queries": 0} for v in views}
            log.info(f"[{country}] pool {pc.height:,} rows, matrices built in {build_s:.1f}s "
                     f"(vocab {', '.join(f'{v}={pools[v][1].height:,}' for v in views)})")
            ns_index, ns_added = (build_num_street(pc, ns), 0) if reserve else (None, 0)
            if reserve:
                log.info(f"[{country}] number-street index: {ns_index[0].height:,} number postings, "
                         f"{ns_index[1].height:,} street tokens")
            for i, part in enumerate(parts):
                if part.exists() and not keep_uncapped:
                    total_rows += pl.scan_parquet(part).select(pl.len()).collect().item()
                    continue
                q = qc.slice(i * chunk, chunk)
                t = time.time()
                wide = block_chunk(q, pools, views, bc["views"], k_override, nw, max_c, stats)
                if keep_uncapped:
                    uncapped.append(wide)
                capped = order_and_cap(wide, views, max_c).select(
                    ["s1_row", "pool_row", pl.col("_pos").cast(pl.Int16).alias("cand_pos")]
                    + [f"{m}_{v}" for v in views for m in ("score", "rank")])
                n_add = 0
                if reserve:
                    capped, n_add = attach_num_street(capped, q, pc, ns_index, views, ns)
                    ns_added += n_add
                atomic_write_parquet(capped, part)
                total_rows += capped.height
                log.info(f"[{country}] chunk {i + 1}/{n_chunks}: {q.height:,} S1 -> "
                         f"{capped.height:,} pairs ({capped.height / q.height:.1f}/S1, "
                         f"+{n_add:,} number-street) in {time.time() - t:.1f}s")
            report["countries"][country] = {
                "queries": qc.height, "pool": pc.height, "build_seconds": round(build_s, 1),
                "num_street_added": ns_added,
                "views": {v: {"seconds": round(s["seconds"], 1), "postings": s["postings"],
                              "postings_per_query": round(s["postings"] / max(s["queries"], 1)),
                              "Mpostings_per_s": round(s["postings"] / max(s["seconds"], 1e-9) / 1e6, 1)}
                          for v, s in stats.items()}}
            for v, s in report["countries"][country]["views"].items():
                log.info(f"[{country}] {v}: {s}")
            del pools
        ev = Path(f"work/eval/s2_{split}_{limit_s1 or 'full'}")
        ev.mkdir(parents=True, exist_ok=True)
        (ev / "timing.json").write_text(json.dumps(report, indent=1))
        if keep_uncapped:
            atomic_write_parquet(pl.concat(uncapped, how="diagonal_relaxed"), ev / "uncapped.parquet")
            atomic_write_parquet(queries.select("s1_row", "country", "name_script"), ev / "queries.parquet")
            if reserve:
                _log_slice_recall(st.work_dir, ns["primary_keep"], ns["primary_keep"] + reserve)
        st.rows = total_rows


def _log_slice_recall(work, primary_keep, cap):
    """On the tuning slice only: recall of the lexical top vs the same list plus reserved slots.
    The first number must match the previous run; the difference is the pairs this channel adds."""
    gt_path = Path("work/train/s1_gt/gt_pairs.parquet")
    parts = sorted(Path(work).glob("part-*.parquet"))
    if not gt_path.exists() or not parts:
        return
    q = pl.read_parquet(Path(work) / "queries.parquet").select("s1_row")
    gt = pl.read_parquet(gt_path, columns=["s1_row", "pool_row"]).join(q, on="s1_row")
    c = (pl.concat([pl.read_parquet(p, columns=["s1_row", "pool_row", "cand_pos"]) for p in parts])
           .group_by("s1_row", "pool_row").agg(pl.col("cand_pos").min()))  # tail + slot would double-count
    pos = gt.join(c, on=["s1_row", "pool_row"], how="left")["cand_pos"]
    n = gt.height
    primary = int((pos < primary_keep).fill_null(False).sum())
    both = int((pos < cap).fill_null(False).sum())
    log.info(f"slice recall: lexical top {primary_keep} = {primary / n:.4f} ({primary}/{n}); "
             f"plus number-street cand_pos<{cap} = {both / n:.4f} ({both}/{n}); "
             f"rescued {both - primary}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--limit-s1", type=int)
    p.add_argument("--force", action="store_true")
    p.add_argument("--k-override", type=int, help="tuning: same k for every view")
    a = p.parse_args()
    main(a.split, a.limit_s1, a.force, a.k_override)
