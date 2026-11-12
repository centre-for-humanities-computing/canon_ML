# %%
"""
Sentence-level semantic-projection sentiment for the MeMo novels (GPU script).

Reproduces lauritswl/SemanticProjection's SemanticProjector.project_texts(
texts, concept_vector="Sentiment") without the package's per-call pandas
overhead (it materialises every embedding as a DataFrame — ~18 GB for ~6M
sentences):

    score_i = embedding_i · v / ||v||

  - embedding_i: raw (un-normalised) paraphrase-multilingual-mpnet-base-v2
    sentence embedding, as the package's Embedder produces it
  - v: the package's data/vectors/Sentiment.csv (1 x 768), copied to
    data/vectors/Sentiment.csv in this project
No centering/normalisation of embeddings, same as the package.

Sentences are rebuilt from the DaCy token shards (data/dk/memo_pos) using the
tokens' character offsets, so original spacing is kept ("Hej!" not
"Hej !"). Linebreak tokens (is_space) become a single space. Sentence
boundaries are DaCy's, i.e. identical to those used in novel_features.py.

OUTPUTS (in --out-dir)
  sentences/part_*.parquet  one row per sentence: filename, sent_id,
                            sent_idx (0..n-1 within novel), n_chars,
                            n_tokens, truncated, sentiment_raw
                            -> keeps the full arc for trajectory analyses
  novel_sentiment.parquet   one row per novel (written at the end, after all
                            parts exist): mean/std/quantiles of the RAW score,
                            plus *_z versions standardised ONCE with the
                            corpus-wide sentence mean/std (never per batch;
                            see note in feature_pipeline.py)

Resumable: novels already present in sentences/ are skipped. Re-run with
--aggregate-only to rebuild novel_sentiment.parquet from the parts.

Setup on the GPU machine (no DaCy needed — only reads parquet):
    pip install torch sentence-transformers pandas pyarrow tqdm
Run:
    nohup python -u src/novel_sentiment.py --pos-dir data/dk/memo_pos \
        --out-dir data/dk/sentiment --batch-size 512 > sentiment.log 2>&1 &
Local smoke test (CPU, 1 novel, first 300 sentences):
    python src/novel_sentiment.py --limit 1 --max-sents 300 --out-dir /tmp/sent_test
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL = "paraphrase-multilingual-mpnet-base-v2"
VECTOR_URL = (
    "https://raw.githubusercontent.com/lauritswl/SemanticProjection/main/"
    "src/semanticprojection/data/vectors/Sentiment.csv"
)
COLUMNS = ["article_id", "token_i", "token_text", "is_space", "sent_id", "start_char", "end_char"]


# %%
# --- reading -----------------------------------------------------------------


def iter_articles(shard_path, batch_size=500_000):
    """Yield (article_id, df) per novel; rows are contiguous per article."""
    pf = pq.ParquetFile(shard_path)
    buf, cur = [], None
    for batch in pf.iter_batches(batch_size=batch_size, columns=COLUMNS):
        df = batch.to_pandas()
        ids = df["article_id"].to_numpy()
        cuts = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(df)]):
            aid = ids[s]
            if aid != cur and buf:
                yield cur, pd.concat(buf, ignore_index=True)
                buf = []
            cur = aid
            buf.append(df.iloc[s:e])
    if buf:
        yield cur, pd.concat(buf, ignore_index=True)


def rebuild_sentences(df):
    """Return DataFrame(sent_id, text, n_tokens) using char offsets for spacing."""
    df = df.sort_values("token_i")
    out = []
    for sid, s in df.groupby("sent_id", sort=True):
        parts, prev_end, n_tok = [], None, 0
        for txt, sp, st, en in zip(s["token_text"], s["is_space"], s["start_char"], s["end_char"]):
            if sp or not isinstance(txt, str) or not txt.strip():
                if parts and parts[-1] != " ":
                    parts.append(" ")
                prev_end = en
                continue
            if prev_end is not None and st > prev_end and parts and parts[-1] != " ":
                parts.append(" ")
            parts.append(txt)
            prev_end = en
            n_tok += 1
        text = "".join(parts).strip()
        if text:
            out.append((sid, text, n_tok))
    return pd.DataFrame(out, columns=["sent_id", "text", "n_tokens"])


# %%
# --- model -------------------------------------------------------------------


def load_vector(path):
    path = Path(path)
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading concept vector to {path}")
        urllib.request.urlretrieve(VECTOR_URL, path)
    v = pd.read_csv(path).to_numpy(dtype=np.float64)[0]
    assert v.shape == (768,), v.shape
    return v / np.linalg.norm(v)  # unit vector: score = emb · v / ||v||


def score(model, unit_v, texts, batch_size, device):
    emb = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        show_progress_bar=False, device=device,
    )
    return emb.astype(np.float64) @ unit_v


# %%
# --- aggregation ---------------------------------------------------------------


def aggregate(sent_dir, out_path):
    s = pd.read_parquet(sent_dir)
    mu, sd = s["sentiment_raw"].mean(), s["sentiment_raw"].std()
    s["z"] = (s["sentiment_raw"] - mu) / sd
    g = s.groupby("filename")
    agg = pd.DataFrame({
        "sent_n": g.size(),
        "sent_truncated_share": g["truncated"].mean(),
        "sentiment_mean_raw": g["sentiment_raw"].mean(),
        "sentiment_std_raw": g["sentiment_raw"].std(),
        "sentiment_mean_z": g["z"].mean(),
        "sentiment_std_z": g["z"].std(),
        "sentiment_q10_z": g["z"].quantile(0.10),
        "sentiment_q90_z": g["z"].quantile(0.90),
        "sentiment_pos_share": g["z"].apply(lambda x: (x > 1).mean()),
        "sentiment_neg_share": g["z"].apply(lambda x: (x < -1).mean()),
    })
    # arc shape: mean z in first / last 10% of the novel, and lag-1 autocorrelation
    s["rel_pos"] = s["sent_idx"] / g["sent_idx"].transform("max").clip(lower=1)
    agg["sentiment_first10_z"] = s[s["rel_pos"] <= 0.1].groupby("filename")["z"].mean()
    agg["sentiment_last10_z"] = s[s["rel_pos"] >= 0.9].groupby("filename")["z"].mean()
    agg["sentiment_ac1"] = g["z"].apply(lambda x: x.autocorr(1) if len(x) > 2 else np.nan)
    agg = agg.reset_index()
    agg.attrs["corpus_mean_raw"], agg.attrs["corpus_std_raw"] = float(mu), float(sd)
    agg.to_parquet(out_path, index=False)
    print(f"Corpus sentence mean/std (raw): {mu:.5f} / {sd:.5f}")
    print(f"Wrote {len(agg)} novels to {out_path}")


# %%
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-dir", default="data/dk/memo_pos")
    ap.add_argument("--out-dir", default="data/dk/sentiment")
    ap.add_argument("--vector", default="data/vectors/Sentiment.csv")
    ap.add_argument("--batch-size", type=int, default=256, help="encode batch size (512-1024 is fine on a large GPU)")
    ap.add_argument("--novels-per-part", type=int, default=25)
    ap.add_argument("--no-fp16", action="store_true", help="keep fp32 weights on GPU (fp16 is ~2x faster)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N novels (testing)")
    ap.add_argument("--max-sents", type=int, default=None, help="only first N sentences per novel (testing)")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    sent_dir = out_dir / "sentences"
    sent_dir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
        if device == "cpu":
            print("WARNING: running on CPU — fine for a smoke test, far too slow for the corpus.")
        model = SentenceTransformer(MODEL, device=device)
        if device == "cuda" and not args.no_fp16:
            model.half()
        print("max_seq_length:", model.max_seq_length, "(longer sentences are truncated, flagged in output)")
        unit_v = load_vector(args.vector)

        done = set()
        for p in sent_dir.glob("part_*.parquet"):
            done |= set(pq.read_table(p, columns=["filename"])["filename"].to_pylist())
        part_id = len(list(sent_dir.glob("part_*.parquet")))
        print(f"{len(done)} novels already done; resuming at part {part_id}")

        tok = model.tokenizer
        buf, n_new, t0 = [], 0, time.time()

        def flush():
            nonlocal buf, part_id
            if not buf:
                return
            sents = pd.concat(buf, ignore_index=True)
            enc_len = [len(x) for x in tok(sents["text"].tolist(), add_special_tokens=True)["input_ids"]]
            sents["truncated"] = np.array(enc_len) > model.max_seq_length
            sents["sentiment_raw"] = score(model, unit_v, sents["text"].tolist(), args.batch_size, device)
            sents.drop(columns="text").to_parquet(sent_dir / f"part_{part_id:05d}.parquet", index=False)
            part_id += 1
            buf = []

        shards = sorted(Path(args.pos_dir).glob("shard_*.parquet"))
        with tqdm(desc="novels") as bar:
            for shard in shards:
                for aid, df in iter_articles(shard):
                    if aid in done:
                        continue
                    s = rebuild_sentences(df)
                    if args.max_sents:
                        s = s.head(args.max_sents)
                    s.insert(0, "filename", aid)
                    s["sent_idx"] = np.arange(len(s))
                    s["n_chars"] = s["text"].str.len()
                    buf.append(s)
                    n_new += 1
                    bar.update(1)
                    if len(buf) >= args.novels_per_part:
                        flush()
                    if args.limit and n_new >= args.limit:
                        break
                if args.limit and n_new >= args.limit:
                    break
            flush()
        print(f"Scored {n_new} novels in {(time.time() - t0) / 60:.1f} min")

    aggregate(sent_dir, out_dir / "novel_sentiment.parquet")


if __name__ == "__main__":
    main()
