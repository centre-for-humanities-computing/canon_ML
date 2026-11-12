# %%
"""
Novel-level stylistic features from the DaCy token shards (data/dk/memo_pos).

Adapted from src/feature_pipeline.py (newspaper version) for long texts:
  - streams each shard with pyarrow (shard_00000 is ~76M token rows; never
    loaded whole), relying on tokens being stored contiguously per article
  - processes novels in small bounded chunks across a worker pool
  - drops German detection (irrelevant for novels) and semantic-projection
    sentiment (needs a GPU at this scale; add later as a separate step)

LINEBREAK HANDLING
The tagging run kept OCR linebreaks ("\n", "\n\n", ...) as tokens. They are
flagged is_space=True and tagged NOUN, and — more importantly — the parser
treated them as words: real tokens can have a "\n" token as their head, and
some sentences have "\n" as ROOT. So:
  1. all is_space tokens are dropped before any feature is computed
  2. sentences with no remaining tokens are dropped
  3. token positions are re-indexed within the article after dropping, so
     dependency distances are not inflated by invisible linebreak tokens
  4. arcs whose head is a whitespace token are excluded from MDD/NDD, and
     sentences whose ROOT is whitespace are excluded from NDD (root distance
     is undefined)
  5. ws_token_share / ws_head_arc_share / ws_root_sent_share are written out
     as diagnostics so you can check whether any feature tracks them
A clean fix (collapse single "\n" to a space before tagging) belongs in the
next DaCy run; these features should then be recomputed from that run.

Usage:
    python src/novel_features.py --pos-dir data/dk/memo_pos --out data/dk/features/novel_features.parquet
    python src/novel_features.py --limit 5   # quick test on the first 5 novels
"""

import argparse
import bz2
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lexical_diversity import lex_div as ld
from tqdm import tqdm

COLUMNS = [
    "article_id", "token_i", "token_text", "lemma", "is_punct", "is_space",
    "pos", "dep", "morph", "head_i", "sent_id",
]
UPOS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "SCONJ", "VERB", "X", "SYM",
]


# %%
# --- streaming: yield one token dataframe per novel -----------------------


def iter_articles(shard_path, batch_size=500_000):
    """Yield (article_id, df) per novel. Assumes rows are contiguous per article."""
    pf = pq.ParquetFile(shard_path)
    buf, cur = [], None
    for batch in pf.iter_batches(batch_size=batch_size, columns=COLUMNS):
        df = batch.to_pandas()
        ids = df["article_id"].to_numpy()
        # split points where the article id changes inside this batch
        cuts = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.concatenate([[0], cuts])
        ends = np.concatenate([cuts, [len(df)]])
        for s, e in zip(starts, ends):
            aid = ids[s]
            part = df.iloc[s:e]
            if aid != cur and buf:
                yield cur, pd.concat(buf, ignore_index=True)
                buf = []
            cur = aid
            buf.append(part)
    if buf:
        yield cur, pd.concat(buf, ignore_index=True)


# %%
# --- cleaning ------------------------------------------------------------


def clean_tokens(df):
    """Drop linebreak/whitespace tokens, re-index positions, flag broken arcs."""
    df = df.sort_values("token_i")
    ws = df["is_space"].to_numpy() | df["token_text"].fillna("").str.strip().eq("").to_numpy()
    ws_ids = set(df.loc[ws, "token_i"].tolist())
    ws_root_sents = set(df.loc[ws & (df["dep"] == "ROOT").to_numpy(), "sent_id"].tolist())

    d = df.loc[~ws].copy()
    # new contiguous position, used for all distances
    pos_map = pd.Series(np.arange(len(d)), index=d["token_i"].to_numpy())
    d["pos_i"] = np.arange(len(d))
    d["head_is_ws"] = d["head_i"].isin(ws_ids)
    d["head_pos"] = d["head_i"].map(pos_map)  # NaN where head was whitespace
    d["sent_root_ws"] = d["sent_id"].isin(ws_root_sents)

    n_sents_total = d["sent_id"].nunique()
    diag = {
        "n_tokens_raw": len(df),
        "ws_token_share": float(ws.mean()) if len(df) else np.nan,
        "ws_head_arc_share": float(d["head_is_ws"].mean()) if len(d) else np.nan,
        "ws_root_sent_share": (len(ws_root_sents & set(d["sent_id"])) / n_sents_total) if n_sents_total else np.nan,
    }
    return d, diag


# %%
# --- features --------------------------------------------------------------


def pos_derived_features(d):
    """Same definitions as feature_pipeline.get_pos_derived_features (whitespace already removed)."""
    d = d[~d["is_punct"] & (d["pos"] != "NUM")]
    pos = d["pos"]
    morph = d["morph"].fillna("").astype(str)
    low = d["token_text"].str.lower()

    nouns, verbs = d[pos == "NOUN"], d[pos == "VERB"]
    n_adj, n_adv = (pos == "ADJ").sum(), (pos == "ADV").sum()
    n_nominal = pos.isin(["PROPN", "ADJ", "NOUN"]).sum()
    n_func = pos.isin(["ADP", "CCONJ", "SCONJ", "AUX", "PART"]).sum()
    prs = (pos == "PRON") & morph.str.contains("PronType=Prs")
    vm = morph[pos == "VERB"]
    n_voice = vm.str.contains("Voice=").sum()

    N, V = len(d), len(verbs)
    if N == 0:
        return {}
    return {
        "nominal_verb_ratio": n_nominal / (n_nominal + V) if (n_nominal + V) else np.nan,
        "noun_ttr": nouns["lemma"].str.lower().nunique() / len(nouns) if len(nouns) else np.nan,
        "verb_ttr": verbs["lemma"].str.lower().nunique() / V if V else np.nan,
        "personal_pronoun_ratio": prs.sum() / N,
        "first_person_ratio": (prs & morph.str.contains("Person=1")).sum() / N,
        "function_word_ratio": n_func / N,
        "of_ratio": ((pos == "ADP") & low.eq("af")).sum() / N,
        "that_ratio": ((pos == "SCONJ") & low.eq("at")).sum() / N,
        "past_tense_ratio": vm.str.contains("Tense=Past").sum() / V if V else np.nan,
        "present_tense_ratio": vm.str.contains("Tense=Pres").sum() / V if V else np.nan,
        "passive_ratio": vm.str.contains("Voice=Pass").sum() / n_voice if n_voice else np.nan,
        "adjective_adverb_ratio": (n_adj + n_adv) / N,
    }


def upos_profile(d):
    words = d[~d["is_punct"]]
    vc = words["pos"].value_counts(normalize=True)
    out = {f"upos_{p}": float(vc.get(p, 0.0)) for p in UPOS}
    out["punct_per_word"] = d["is_punct"].sum() / len(words) if len(words) else np.nan
    return out


def length_features(d, long_word_chars=6):
    words = d.loc[~d["is_punct"], ["token_text", "sent_id"]]
    wl = words["token_text"].str.len()
    sl = words.groupby("sent_id").size()
    n_w, n_s = len(words), len(sl)
    long_w = (wl > long_word_chars).sum()
    return {
        "n_words": n_w,
        "num_sents": n_s,
        "avg_wordlen": wl.mean() if n_w else np.nan,
        "avg_sentlen": sl.mean() if n_s else np.nan,
        "std_sentlen": sl.std() if n_s > 1 else np.nan,
        "lix": (n_w / n_s) + 100 * long_w / n_w if n_w and n_s else np.nan,
        "rix": long_w / n_s if n_s else np.nan,
    }


def lexical_diversity(d, chunk=1000):
    toks = d.loc[~d["is_punct"], "token_text"].str.lower().tolist()
    n = len(toks)
    if n == 0:
        return {"mtld": np.nan, "msttr_1000": np.nan, "cttr": np.nan}
    # MSTTR over fixed 1000-token windows: length-independent, comparable across novels
    windows = [toks[i:i + chunk] for i in range(0, n - chunk + 1, chunk)]
    msttr = np.mean([len(set(w)) / chunk for w in windows]) if windows else np.nan
    try:
        mtld = ld.mtld(toks)
    except Exception:
        mtld = np.nan
    return {"mtld": mtld, "msttr_1000": msttr, "cttr": len(set(toks)) / np.sqrt(2 * n)}


def dependency_distances(d):
    """MDD / NDD per sentence on whitespace-free positions; broken arcs/sentences excluded."""
    x = d[~d["is_punct"]]
    arcs = x[(x["dep"] != "ROOT") & x["head_pos"].notna()]
    dd = (arcs["pos_i"] - arcs["head_pos"]).abs()
    mdd_sent = dd.groupby(arcs["sent_id"]).mean()

    roots = x[(x["dep"] == "ROOT") & ~x["sent_root_ws"]]
    first = x.groupby("sent_id")["pos_i"].min()
    slen = x.groupby("sent_id").size()
    r = roots.drop_duplicates("sent_id").set_index("sent_id")["pos_i"]
    common = mdd_sent.index.intersection(r.index)
    root_dist = (r.loc[common] - first.loc[common]).clip(lower=0)
    prod = (root_dist + 1) * slen.loc[common]
    m = mdd_sent.loc[common]
    ok = m > 0
    ndd = np.abs(np.log(m[ok] / np.sqrt(prod[ok])))
    return {
        "avg_mdd": mdd_sent.mean(), "std_mdd": mdd_sent.std(),
        "avg_ndd": ndd.mean(), "std_ndd": ndd.std(),
    }


def compression_ratio(d, n_windows=10, window_sents=40, seed=0):
    """bz2 ratio averaged over random 40-sentence windows (the newspaper version used one window)."""
    sents = d.groupby("sent_id")["token_text"].apply(lambda t: " ".join(t)).tolist()
    if len(sents) < window_sents:
        return {"compressrat": np.nan}
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(sents) - window_sents + 1, size=n_windows)
    ratios = []
    for s in starts:
        b = " ".join(sents[s:s + window_sents]).encode("utf-8")
        ratios.append(len(b) / len(bz2.compress(b, compresslevel=9)))
    return {"compressrat": float(np.mean(ratios))}


def process_novel(item):
    aid, df = item
    d, diag = clean_tokens(df)
    row = {"filename": aid, **diag}
    for f in (pos_derived_features, upos_profile, length_features,
              lexical_diversity, dependency_distances, compression_ratio):
        row.update(f(d))
    return row


# %%
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-dir", default="data/dk/memo_pos")
    ap.add_argument("--out", default="data/dk/features/novel_features.parquet")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--chunk", type=int, default=None, help="novels held in memory at once (default 2x workers)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N novels (testing)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    chunk = args.chunk or 2 * args.workers
    shards = sorted(Path(args.pos_dir).glob("shard_*.parquet"))
    print(f"{len(shards)} shards, {args.workers} workers, {chunk} novels per chunk")

    rows, done = [], 0
    with mp.Pool(args.workers) as pool, tqdm(desc="novels") as bar:
        for shard in shards:
            buf = []
            for item in iter_articles(shard):
                buf.append(item)
                if len(buf) >= chunk:
                    rows += pool.map(process_novel, buf)
                    bar.update(len(buf)); done += len(buf); buf = []
                if args.limit and done + len(buf) >= args.limit:
                    break
            if buf:
                rows += pool.map(process_novel, buf)
                bar.update(len(buf)); done += len(buf)
            if args.limit and done >= args.limit:
                break

    res = pd.DataFrame(rows)
    dup = res["filename"].duplicated().sum()
    if dup:
        print(f"WARNING: {dup} duplicate filenames (article split across shards?)")
    res.to_parquet(out, index=False)
    print(f"Wrote {len(res)} rows x {res.shape[1]} cols to {out}")


if __name__ == "__main__":
    main()
