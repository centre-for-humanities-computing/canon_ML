# %%
"""
Random-forest classification of canon proxies from stylistic features.

For every canon proxy in the corpus config (data/<corpus>/config.py), and for
several feature sets, a random forest is evaluated with:

A repeated grouped, stratified 5-fold CV: grouped by author, so no author is in both train and test;
  repeated N_REPEATS times with a different shuffle of the authors. Every novel is tested once per
  repeat; P/R/F1 are computed per repeat and reported as mean ± SD over the repeats
Class balancing: in every training fold the non-canon novels are randomly undersampled to 1:1 (test folds untouched)

Then:
  - per-proxy, per-feature-set P/R/F1 of the canon class at 0.5 (vs. a coin
    flip), macro-F1, and AP / ROC-AUC as threshold-free secondary metrics
  - a final model per proxy, saved to models/<corpus>/
  - a digitisation-source check (does the model learn the scan batch? relevant for MEMO)
  - performance vs. year, and the most confident errors

Run:
    python src/make_model.py
Set CORPUS in the CONFIG section to "dk" or "de" to pick the corpus.
"""

import importlib.util
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

# %%
# ---------------------------------------------------------------- CONFIG

CWD = Path(__file__).parent.parent.resolve() if "__file__" in globals() else Path.cwd()

CORPUS = "dk"   # "dk" or "de": which data/<corpus>/config.py to use


def load_config(corpus):
    path = CWD / "data" / corpus / "config.py"
    spec = importlib.util.spec_from_file_location(f"config_{corpus}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not mod.HF_REPO or not mod.ID_COL or not mod.CANON_PROXIES:
        raise ValueError(f"{path} is incomplete (HF_REPO / ID_COL / CANON_PROXIES): fill in the TODOs first")
    return mod


cfg = load_config(CORPUS)

FIGS = CWD / "figs" / CORPUS
MODELS = CWD / "models" / CORPUS
RESULTS = CWD / "results" / CORPUS
LOGS = CWD / "logs"
for folder in (FIGS, MODELS, RESULTS, LOGS):
    folder.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
N_REPEATS = 10   # repeats of the whole 5-fold CV, each with a different author shuffle
RANDOM_STATE = 42
N_TREES = 500

logging.basicConfig(
    filename=LOGS / f"make_model_{CORPUS}.txt",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)


# helper for logging
def log(msg):
    print(msg)
    logging.info(msg)


# helper for latex
def latex_int(n):
    return f"{n:,}".replace(",", "{,}")


def label(target):
    return cfg.PROXY_LABELS.get(target, target)


log(f"corpus: {CORPUS} ({cfg.HF_REPO})")

# %%
# ---------------------------------------------------------------- LOAD

# metadata only -- the features were computed from the token shards, so the
# raw text is not needed here
ds = load_dataset(cfg.HF_REPO, revision=cfg.HF_REVISION)["train"]
meta = ds.remove_columns([c for c in ["text"] if c in ds.column_names]).to_pandas()

feats = pd.read_parquet(CWD / cfg.FEATURES_PATH).rename(columns={"filename": cfg.ID_COL})
df = meta.merge(feats, on=cfg.ID_COL, how="inner", validate="one_to_one")
missing = sorted(set(meta[cfg.ID_COL]) - set(feats[cfg.ID_COL]))
log(f"{len(meta)} texts in {cfg.HF_REPO}; {len(feats)} with features; "
    f"modelling the {len(df)} with both ({len(missing)} without features)")
if missing:
    log(f"  no features for: {missing}")

STYLE_FEATURES = [c for c in feats.columns
                  if c != cfg.ID_COL and c not in cfg.NON_FEATURE_COLS
                  and pd.api.types.is_numeric_dtype(feats[c])]

print(f"style features ALL ({len(STYLE_FEATURES)}): {STYLE_FEATURES}")

# set style features manually
STYLE_FEATURES = ['nominal_verb_ratio', 'noun_ttr', 'verb_ttr', 'personal_pronoun_ratio', 'first_person_ratio', 'function_word_ratio', 
                  'of_ratio', 'that_ratio', 'past_tense_ratio', 'present_tense_ratio', 'passive_ratio', 'adjective_adverb_ratio', 
                  #'upos_ADJ', 'upos_ADP', 'upos_ADV', 'upos_AUX', 'upos_CCONJ', 'upos_DET', 'upos_INTJ', 'upos_NOUN', 'upos_NUM', 'upos_PART', 
                  #'upos_PRON', 'upos_PROPN', 'upos_SCONJ', 'upos_VERB', 'upos_X', 'upos_SYM', 'punct_per_word', 
                  #'n_words', 'num_sents', 
                  'avg_wordlen', 'avg_sentlen', 'std_sentlen', 'lix', 'rix', 'mtld', 'msttr_1000', 'cttr', 'avg_mdd', 'std_mdd', 'avg_ndd', 'std_ndd', 'compressrat']


df = df.reset_index(drop=True)
all_idx = np.arange(len(df))
log(f"style features ({len(STYLE_FEATURES)}): {STYLE_FEATURES}")

df.head()

# %%
# ---------------------------------------------------------------- CANON TARGETS

# every proxy is binary: positive if ANY of its source columns is > 0
targets, target_info = {}, []
for name, cols in cfg.CANON_PROXIES.items():
    absent = [c for c in cols if c not in df.columns]
    if absent:
        raise KeyError(f"proxy {name!r}: columns not in the dataset: {absent}")
    yt = (df[cols].astype(float).fillna(0) > 0).any(axis=1).to_numpy().astype(int)
    n_pos = int(yt.sum())
    n_pos_auth = df.loc[yt == 1, cfg.GROUP_COLS[0]].nunique() if cfg.GROUP_COLS else np.nan
    keep = n_pos >= cfg.MIN_POSITIVES
    target_info.append({"target": name, "n_pos": n_pos, "prevalence": yt.mean(),
                        "n_pos_groups": n_pos_auth, "modelled": keep})
    if keep:
        targets[name] = yt

target_info = pd.DataFrame(target_info)
log("\ncanon proxies (positive = any source column > 0):")
log(target_info.round(3).to_string(index=False))
log(f"modelling {len(targets)} proxies; skipped (< {cfg.MIN_POSITIVES} positives): "
    f"{target_info.loc[~target_info.modelled, 'target'].tolist()}")

# how much do the proxies overlap? (Jaccard on positives)
T = pd.DataFrame(targets)
jac = pd.DataFrame({a: {b: ((T[a] & T[b]).sum() / max((T[a] | T[b]).sum(), 1)) for b in T} for a in T})
log("\nproxy overlap (Jaccard of positive sets):")
log(jac.round(2).to_string())

# %%
# canon vs non-canon over time, per proxy (stacked counts per 5-year bin)
YEAR_BIN = 5

_year = df[cfg.YEAR_COL].to_numpy(dtype=float)
_bins = np.arange(np.floor(_year.min() / YEAR_BIN) * YEAR_BIN, _year.max() + YEAR_BIN + 1, YEAR_BIN)
ncol = min(2, len(targets))   # 2 per row: 3 rows of 2 for 6 proxies
nrow = int(np.ceil(len(targets) / ncol))

fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.9 * nrow), squeeze=False, sharex=True)
for ax, (tname, y) in zip(axes.ravel(), targets.items()):
    ax.hist([_year[y == 1], _year[y == 0]], bins=_bins, stacked=True,
            color=["#104281", "#c9c7c1"], label=["canon", "not canon"], edgecolor="white", linewidth=0.5)
    ax.set_title(f"{label(tname)}  ({int(y.sum())}/{len(y)} canon)", fontsize=10, loc="left")
    ax.set_ylabel("novels", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=8)

for ax in axes[-1]:
    ax.set_xlabel("year", fontsize=8)

axes.ravel()[0].legend(fontsize=8, frameon=False)
for ax in axes.ravel()[len(targets):]:
    ax.axis("off")

fig.tight_layout()
fig.savefig(FIGS / f"{CORPUS}_canon_over_time.png", dpi=150, facecolor="white")
plt.close(fig)
log(f"wrote {FIGS / f'{CORPUS}_canon_over_time.png'}")

# %%
# ---------------------------------------------------------------- FEATURES

# Each block returns a dense (n_rows, n_features) array for the given rows plus
# its column names. Random forests need no scaling, so nothing is fit per fold
# here (unlike TF-IDF/scalers) and there is no leakage to guard against.

if cfg.SOURCE_COL:
    source = df[cfg.SOURCE_COL].fillna(cfg.SOURCE_MISSING_LABEL).astype(str)
    SOURCE_DUMMIES = pd.get_dummies(source, prefix="source").astype(float)
else:
    source = pd.Series("all", index=df.index)
    SOURCE_DUMMIES = pd.DataFrame(index=df.index)

BLOCK_COLUMNS = {
    "style": STYLE_FEATURES,
    "year": [cfg.YEAR_COL],
    "source": list(SOURCE_DUMMIES.columns) + [c for c in cfg.SOURCE_DIAGNOSTIC_COLS if c in df.columns],
}

X_ALL = pd.concat([df[STYLE_FEATURES + [cfg.YEAR_COL]
                      + [c for c in cfg.SOURCE_DIAGNOSTIC_COLS if c in df.columns]],
                   SOURCE_DUMMIES], axis=1).astype(float)


def columns_for(blocks):
    cols = []
    for b in blocks:
        cols += [c for c in BLOCK_COLUMNS[b] if c not in cols]
    return cols


# %%
# ---------------------------------------------------------------- CV


def build_groups(df, link_cols=tuple(cfg.GROUP_COLS)):
    """CV groups = connected components over the linking columns: two texts share a group if they
    share a value in ANY link column (transitively). Texts with no link value in any column stay
    solo -- never lumped into one giant 'missing' group."""
    parent = np.arange(len(df))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for col in link_cols:
        key = df[col].astype(str).str.strip().str.lower()
        key = key.where(~key.isin(["nan", "none", ""]))
        first_row = {}
        for i, k in enumerate(key):
            if pd.isna(k):
                continue
            if k in first_row:
                parent[find(i)] = find(first_row[k])
            else:
                first_row[k] = i

    roots = np.array([find(i) for i in range(len(df))])
    sizes = pd.Series(roots).map(pd.Series(roots).value_counts()).values
    return np.array([f"grp_{r}" if s > 1 else f"solo_{a}"
                     for r, s, a in zip(roots, sizes, df[cfg.ID_COL].astype(str))])


groups = build_groups(df)
sizes = pd.Series(groups).value_counts()
log(f"\ngroups: {len(sizes)} distinct (from {len(df)} texts); "
    f"{int((sizes > 1).sum())} multi-text groups, largest = {sizes.iloc[0]} texts")


def make_clf():
    # no class weights: the training rows are already 1:1 after undersample()
    return RandomForestClassifier(
        n_estimators=N_TREES, max_features="sqrt", min_samples_leaf=2,
        n_jobs=-1, random_state=RANDOM_STATE,
    )


def undersample(train_rows, y, seed):
    """CLASS BALANCING: keep all canon novels in the training rows and a random draw of
    equally many non-canon novels (1:1), so P(canon) = 0.5 is a meaningful cut. Only TRAINING
    rows are undersampled; test folds keep the true class ratio, so P/R/F1 reflect the real
    prevalence. The draw is seeded, so every feature set sees the same training rows."""
    pos = train_rows[y[train_rows] == 1]
    neg = train_rows[y[train_rows] == 0]
    rng = np.random.default_rng(seed)
    neg = rng.choice(neg, size=min(len(neg), len(pos)), replace=False)
    return np.sort(np.concatenate([pos, neg]))


def scores(y_true, prob, threshold=0.5):
    """Headline: P/R/F1 for the canon class at the threshold, plus macro-F1.
    AP and ROC-AUC (threshold-free) are kept as secondary metrics."""
    pred = (prob >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f,
            "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
            "flagged_share": pred.mean(),
            "ap": average_precision_score(y_true, prob), "roc_auc": roc_auc_score(y_true, prob)}


def random_f1(prevalence):
    """F1 of a coin flip that labels half the novels canon: P = prevalence, R = 0.5."""
    return 2 * prevalence * 0.5 / (prevalence + 0.5)


def get_folds(y, idx, rep=0):
    """Grouped, stratified folds over the rows `idx`: [(train_pos, test_pos), ...] (positions
    within idx). The seed depends only on the repeat, so for a given target every feature set
    gets the same folds in the same repeat."""
    n_splits = max(2, min(N_SPLITS, int(y[idx].sum())))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + rep)
    return list(cv.split(idx, y[idx], groups[idx]))


def cv_oof(cols, y, idx=all_idx, rep=0):
    """One run of 5-fold CV: out-of-fold P(canon) for the rows `idx` (aligned with idx). Each
    training fold is undersampled to 1:1; the test fold is not."""
    X = X_ALL[cols].to_numpy()
    oof = np.zeros(len(idx))
    for k, (tr_pos, te_pos) in enumerate(get_folds(y, idx, rep)):
        tr, te = undersample(idx[tr_pos], y, seed=RANDOM_STATE + 1000 * rep + k), idx[te_pos]
        clf = make_clf().fit(X[tr], y[tr])
        oof[te_pos] = clf.predict_proba(X[te])[:, 1]
    return oof


def cv_repeated(cols, y, idx=all_idx):
    """N_REPEATS runs of 5-fold CV. Returns the (N_REPEATS, len(idx)) matrix of out-of-fold
    P(canon) and a DataFrame with one row of scores (P/R/F1, ...) per repeat -- each repeat is
    a classification report over all novels, every novel predicted while held out."""
    probs = np.vstack([cv_oof(cols, y, idx, rep) for rep in range(N_REPEATS)])
    per_rep = pd.DataFrame([scores(y[idx], p) for p in probs])
    return probs, per_rep


def mean_sd(per_rep):
    """Mean of every score over the repeats, plus the SD of the headline ones."""
    out = per_rep.mean().to_dict()
    for k in ("precision", "recall", "f1", "macro_f1"):
        out[f"{k}_sd"] = float(per_rep[k].std(ddof=1))
    return out


# %%
# ---------------------------------------------------------------- MODEL COMPARISON

# year_only and source_only are baselines: if they score as well as style, the
# "style" signal may be period or digitisation batch rather than style
VARIANTS = {
    "year_only": ["year"],
    "source_only": ["source"],
    "style": ["style"],
    "style_year": ["style", "year"],
}
if not cfg.SOURCE_COL:
    VARIANTS.pop("source_only")
FINAL_MODEL_VARIANT = "style"

rows, rep_rows, oof_by = [], [], {}
log(f"\nmodel comparison ({N_REPEATS} x {N_SPLITS}-fold CV grouped by {cfg.GROUP_COLS}; "
    f"same folds for every variant within a proxy; mean ± SD over the {N_REPEATS} repeats):")
for tname, y in targets.items():
    log(f"\n{label(tname)}  (n_pos={int(y.sum())}, prevalence={y.mean():.3f}; "
        f"coin-flip F1={random_f1(y.mean()):.3f})")
    for vname, blocks in VARIANTS.items():
        probs, per_rep = cv_repeated(columns_for(blocks), y)
        oof_by[(tname, vname)] = probs.mean(axis=0)   # per-novel score averaged over repeats (plots/errors)
        m = mean_sd(per_rep)
        rows.append({"target": tname, "variant": vname, **m,
                     "prevalence": y.mean(), "random_f1": random_f1(y.mean()), "n_pos": int(y.sum())})
        rep_rows += [{"target": tname, "variant": vname, "repeat": i, **r} for i, r in per_rep.iterrows()]
        log(f"  {vname:24s} P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} ± {m['f1_sd']:.3f}  macro-F1={m['macro_f1']:.3f}  "
            f"| AP={m['ap']:.3f} ROC-AUC={m['roc_auc']:.3f}")

results = pd.DataFrame(rows)
results.to_csv(RESULTS / "model_comparison.csv", index=False)
pd.DataFrame(rep_rows).to_csv(RESULTS / "model_comparison_per_repeat.csv", index=False)
log(f"\nwrote {RESULTS / 'model_comparison.csv'} (+ _per_repeat.csv)")

# %%
# ---------------------------------------------------------------- FINAL MODELS

final_cols = columns_for(VARIANTS[FINAL_MODEL_VARIANT])
importances = {}
for tname, y in targets.items():
    tr = undersample(all_idx, y, seed=RANDOM_STATE)
    clf = make_clf().fit(X_ALL[final_cols].to_numpy()[tr], y[tr])
    importances[tname] = pd.Series(clf.feature_importances_, index=final_cols)
    bundle = {
        "clf": clf, "features": final_cols, "target": tname,
        "target_source_cols": cfg.CANON_PROXIES[tname], "threshold": 0.5,
        "meta": {
            "fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "corpus": CORPUS, "hf_repo": cfg.HF_REPO, "n_train": len(tr), "n_pos": int(y.sum()),
            "balancing": "negatives undersampled to 1:1", "variant": FINAL_MODEL_VARIANT,
            "cv_f1_mean": float(results.query("target == @tname and variant == @FINAL_MODEL_VARIANT")["f1"].iloc[0]),
            "cv": f"{N_REPEATS} x {N_SPLITS}-fold grouped by {cfg.GROUP_COLS}",
            "sklearn": sklearn.__version__, "python": sys.version.split()[0],
        },
    }
    joblib.dump(bundle, MODELS / f"rf_{tname}_{FINAL_MODEL_VARIANT}.joblib", compress=3)

# out-of-fold predictions of the final variant (averaged over the repeats), for downstream scripts
oof_out = df[[cfg.ID_COL, cfg.YEAR_COL]].copy()
for tname, y in targets.items():
    oof_out[f"{tname}__true"] = y
    oof_out[f"{tname}__oof"] = oof_by[(tname, FINAL_MODEL_VARIANT)]
oof_out.to_csv(RESULTS / f"oof_predictions_{FINAL_MODEL_VARIANT}.csv", index=False)

# %%
# ---------------------------------------------------------------- FEATURE IMPORTANCES (final models)

# Built-in random-forest (impurity / mean decrease in Gini) importances of each final model:
# cheap, but biased towards continuous, high-variance features, and they show what the forest
# USES, not whether it helps on unseen novels. Permutation importance on held-out folds
# (separate script) is the check for that.
imp_df = pd.DataFrame(importances)
imp_df.to_csv(RESULTS / f"feature_importance_{FINAL_MODEL_VARIANT}.csv")
log(f"\nfeature importances ({FINAL_MODEL_VARIANT} final models, impurity-based; top 10 per proxy):")
for tname in targets:
    top = imp_df[tname].sort_values(ascending=False).head(10)
    log(f"  {label(tname)}: " + ", ".join(f"{k} ({v:.3f})" for k, v in top.items()))

TOP_N = 15
ncol = min(3, len(targets))
nrow = int(np.ceil(len(targets) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 0.3 * TOP_N * nrow + 1.0 * nrow), squeeze=False)
for ax, tname in zip(axes.ravel(), targets):
    s_imp = imp_df[tname].sort_values().tail(TOP_N)
    ax.barh(s_imp.index, s_imp.values, color="#104281")
    ax.axvline(1 / len(final_cols), color="#52514e", linewidth=0.8, linestyle=(0, (4, 3)))  # equal-share line
    ax.set_title(label(tname), fontsize=10, loc="left")
    ax.set_xlabel("impurity importance (dashed = equal share)", fontsize=8)
    ax.tick_params(labelsize=8)
for ax in axes.ravel()[len(targets):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(FIGS / f"feature_importance_{FINAL_MODEL_VARIANT}.png", dpi=150, facecolor="white")
plt.close(fig)
log(f"wrote {FIGS / f'feature_importance_{FINAL_MODEL_VARIANT}.png'}")


# %%
# ---------------------------------------------------------------- DIGITISATION-SOURCE CHECK

# The later-digitised batches differ in OCR/linebreak handling and are also
# earlier novels on average, so a style model could partly learn the batch.
# (1) source_only above: how far does the batch alone get?
# (2) redo the CV inside the largest source only, where the batch cannot help.
source_rows = []
if cfg.SOURCE_COL:
    log("\npositives per digitisation source:")
    log(pd.DataFrame({t: pd.Series(y).groupby(source.values).sum() for t, y in targets.items()}).to_string())

    main_source = source.value_counts().index[0]
    idx = np.where((source == main_source).values)[0]
    log(f"\nCV within '{main_source}' only ({len(idx)} texts), {FINAL_MODEL_VARIANT}:")
    for tname, y in targets.items():
        if y[idx].sum() < cfg.MIN_POSITIVES // 2:
            log(f"  {label(tname):40s} too few positives ({int(y[idx].sum())})")
            continue
        _, per_rep = cv_repeated(final_cols, y, idx)
        s = mean_sd(per_rep)
        full = results.query("target == @tname and variant == @FINAL_MODEL_VARIANT")["f1"].iloc[0]
        source_rows.append({"target": tname, "n": len(idx), "n_pos": int(y[idx].sum()),
                            "prevalence": y[idx].mean(), "random_f1": random_f1(y[idx].mean()),
                            "f1_full": full, **{f"{k}_within_source": v for k, v in s.items()}})
        log(f"  {label(tname):40s} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} ± {s['f1_sd']:.3f} "
            f"(coin-flip {random_f1(y[idx].mean()):.3f}; full data F1 {full:.3f})")
    pd.DataFrame(source_rows).to_csv(RESULTS / "source_check.csv", index=False)

# %%
# ---------------------------------------------------------------- TABLES FOR THE PAPER

# header + body rows only; \begin{tabular}, caption and label stay hand-written in the paper


def write_comparison_table(path):
    """Canon-class F1 per proxy (rows) and feature set (columns); best feature set per proxy in bold.
    'Canon' = class balance (canon novels / all novels), which sets the coin-flip F1."""
    piv = results.pivot(index="target", columns="variant", values="f1").loc[list(targets), list(VARIANTS)]
    base = results.groupby("target")["random_f1"].first()
    head = " & ".join(rf"\textbf{{{v.replace('_', ' ')}}}" for v in VARIANTS)
    lines = [rf"\textbf{{Proxy}} & \textbf{{Canon}} & \textbf{{Coin flip}} & {head} \\", r"\midrule"]
    for t, y in targets.items():
        best = piv.loc[t].max()
        cells = [(rf"\textbf{{{f:.3f}}}" if np.isclose(f, best) else f"{f:.3f}") for f in piv.loc[t]]
        balance = rf"{int(y.sum())}/{len(y)} ({100 * y.mean():.0f}\%)"
        lines.append(f"{label(t)} & {balance} & {base[t]:.3f} & " + " & ".join(cells) + r" \\")
    Path(path).write_text("\n".join(lines) + "\n")
    log(f"wrote feature-set comparison table to {path}")


def write_final_model_table(path):
    lines = [r"\textbf{Proxy} & \textbf{n pos.} & \textbf{Prev.} & \textbf{P} & \textbf{R} & \textbf{F1} & "
             r"\textbf{SD} & \textbf{Coin flip F1} & \textbf{Macro-F1} & \textbf{AP} \\", r"\midrule"]
    # all scores: mean over the N_REPEATS CV repeats; SD = SD of F1 over the repeats
    for tname, y in targets.items():
        r = results.query("target == @tname and variant == @FINAL_MODEL_VARIANT").iloc[0]
        lines.append(f"{label(tname)} & {int(y.sum())} & {y.mean():.3f} & {r.precision:.3f} & {r.recall:.3f} & "
                     f"{r.f1:.3f} & {r.f1_sd:.3f} & {r.random_f1:.3f} & {r.macro_f1:.3f} & {r.ap:.3f} \\\\")
    Path(path).write_text("\n".join(lines) + "\n")
    log(f"wrote final-model ({FINAL_MODEL_VARIANT}) table to {path} (n={latex_int(len(df))})")


write_comparison_table(RESULTS / "feature_comparison.txt")
write_final_model_table(RESULTS / "final_model_performance.txt")

# %%
# ---------------------------------------------------------------- ERROR ANALYSIS (final model)

# Ranking-based rather than at 0.5: with rare positives and balanced class
# weights the 0.5 cut is arbitrary. FP = highest-ranked negatives, FN = lowest-ranked positives.
show = [c for c in [cfg.ID_COL, cfg.AUTHOR_NAME_COL, cfg.TITLE_COL, cfg.YEAR_COL] if c]
for tname, y in targets.items():
    oof = oof_by[(tname, FINAL_MODEL_VARIANT)]
    tmp = df[show].assign(p=oof, y=y)
    log(f"\n{label(tname)} -- highest-scored NON-canon novels (out-of-fold):")
    log(tmp[tmp.y == 0].nlargest(10, "p").round(3).to_string(index=False))
    log(f"{label(tname)} -- lowest-scored CANON novels:")
    log(tmp[tmp.y == 1].nsmallest(10, "p").round(3).to_string(index=False))

# %%
# ---------------------------------------------------------------- PERFORMANCE VS YEAR

# Is the predicted canon probability tied to period? A strong slope suggests the
# model is partly a period detector.
# "Out-of-fold P(canon)": each novel's probability comes from the CV model whose
# training folds did NOT contain that novel (or any other novel by the same
# author), i.e. a genuine prediction for an unseen author; averaged over the
# N_REPEATS repeats. Training is 1:1
# undersampled, so 0.5 is the decision threshold; it is not a calibrated
# probability of being canon at the real prevalence.
year = df[cfg.YEAR_COL].to_numpy(dtype=float)
ncol = min(3, len(targets))
nrow = int(np.ceil(len(targets) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
log(f"\nP(canon) vs year ({FINAL_MODEL_VARIANT}, out-of-fold):")
for ax, (tname, y) in zip(axes.ravel(), targets.items()):
    oof = oof_by[(tname, FINAL_MODEL_VARIANT)]
    r_all, p_all = spearmanr(year, oof)
    r_pos, p_pos = spearmanr(year[y == 1], oof[y == 1])
    log(f"  {label(tname):40s} all: rho={r_all:+.3f} (p={p_all:.3f}); canon only: rho={r_pos:+.3f} (p={p_pos:.3f})")
    ax.scatter(year[y == 0], oof[y == 0], s=8, alpha=0.25, color="#9d9b95", label="not canon")
    ax.scatter(year[y == 1], oof[y == 1], s=12, alpha=0.7, color="#104281", label="canon")
    ax.axhline(0.5, color="#52514e", linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(0.02, 0.97, f"ρ all = {r_all:+.2f}\nρ canon = {r_pos:+.2f}", transform=ax.transAxes,
            fontsize=8, va="top", ha="left", color="#52514e")
    ax.set_title(label(tname), fontsize=10, loc="left")
    ax.set_xlabel("year", fontsize=8)
    ax.set_ylabel("out-of-fold P(canon)", fontsize=8)
    ax.grid(alpha=0.3)
axes.ravel()[0].legend(fontsize=8, frameon=False)
for ax in axes.ravel()[len(targets):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(FIGS / "p_canon_vs_year.png", dpi=150, facecolor="white", bbox_inches="tight")
plt.close(fig)
log(f"wrote {FIGS / 'p_canon_vs_year.png'}")

# %%
