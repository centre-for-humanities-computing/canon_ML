# %%
"""
Canon-proxy exploration: how do the proxies in the metadata relate to each
other, and is there one latent "canon" dimension behind them?

  1. proxy prep (fill missing with 0, binary proxies as 0/1, continuous proxies
     left on their raw scale -- nothing is scaled -- and the three national canon
     lists merged into one count 0-3)
  2. visualisations: histograms of the continuous proxies, proxy heatmap
  3. proxy diagnostics (Cronbach's alpha, item-total correlations, coverage)
  4. simple binary sum (used as a sanity check for the FA score)
  5. factor analysis: factor-count sweep, KMO/Bartlett, ML + promax fit,
     parallel analysis, bootstrapped loadings heatmap, second-order factor,
     final FA canon score

Run cell by cell (# %%) or as a script. Figures -> figs/<corpus>/,
LaTeX tables -> results/<corpus>/.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo

try:
    from IPython.display import display
except ImportError:          # plain `python` run
    display = print

CWD = Path(__file__).parent.parent.resolve() if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(CWD / "src"))
from factor_utils.factor_sweep import run_factor_sweep  # noqa: E402

# %%
# ---------------------------------------------------------------- CONFIG

CORPUS = "dk"   # "dk" or "de": which data/<corpus>/config.py to use


def load_config(corpus):
    path = CWD / "data" / corpus / "config.py"
    spec = importlib.util.spec_from_file_location(f"config_{corpus}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not mod.HF_REPO or not mod.ID_COL:
        raise ValueError(f"{path} is incomplete (HF_REPO / ID_COL): fill in the TODOs first")
    return mod


cfg = load_config(CORPUS)

FIGS = CWD / "figs" / CORPUS
RESULTS = CWD / "results" / CORPUS
for folder in (FIGS, RESULTS):
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 17
rng = np.random.default_rng(RANDOM_SEED)
ts = time.strftime("%Y%m%d")

# deduplication option (see get_deduplicated_df below)
USE_DEDUPLICATED_VECTORS = False
USED_WHAT = "deduplicated" if USE_DEDUPLICATED_VECTORS else "full"
LATEX_PATH = RESULTS / f"{ts}_proxy_tables_{USED_WHAT}.tex"
LATEX_PATH.write_text("")   # start fresh each run; save_latex_table() appends


def save_latex_table(table, path, caption, label):
    """Append one table (with caption/label) to the run's .tex file."""
    tex = table.to_latex(index=False, caption=caption, label=label, escape=True)
    with open(path, "a") as f:
        f.write(tex + "\n\n")
    print(f"appended table '{label}' to {path}")


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGS / f"{USED_WHAT}_{name}.pdf", dpi=300, bbox_inches="tight")
    plt.show()


# %%
# ---------------------------------------------------------------- LOAD

ds = load_dataset(cfg.HF_REPO, revision=cfg.HF_REVISION)["train"]
df = ds.remove_columns([c for c in ["text"] if c in ds.column_names]).to_pandas()
print(f"{len(df)} texts loaded from {cfg.HF_REPO}")

# %%
# ---------------------------------------------------------------- PROXY PREP

proxy_cols = [
    "wordcount_main_text_lex_auth", "title_mention_lex_authorpage",
    "lex_author_mentioned_in_other_authorpage",
    "n_wikipedia_languages_auth", 
    #"n_wikipedia_sitelinks_auth",
    #"n_library_authorities_auth", 
    #"n_wikidata_awards_auth",
    #"n_wikidata_notable_works_auth", "n_wikidata_works_auth",
    "title_mention_wikidata_notable_works", 
    "wordcount_wikipedia_da_auth",
    "wordcount_wikipedia_en_auth",
    "wikipedia_da_author_mentioned_in_other_authorpage",
    "wikipedia_en_author_mentioned_in_other_authorpage",
    "auth_mention_lithåndb_nameregistry",
    "auth_mention_littveje_nameregistry",
    "auth_mention_litt_find_nameregistry",
    "auth_mention_littmøn_nameregistry",
    "auth_mention_littgenveje_nameregistry",
    "auth_mention_bruglitt_nameregistry",
    "auth_mention_medtiden_nameregistry",
    "auth_mention_litt_intro_nameregistry",
    "auth_mention_trane_nameregistry", 
    "auth_mention_håndb_nameregistry",
    "auth_mention_rød_dan_lithis_nameregistry",
    "auth_mention_grøn_dan_lithis_nameregistry",
    "auth_mention_rød_dan_lithis_titleregistry",
    "title_mention_rød_dan_lithis_titleregistry", 
    "canon_culture_authors",
    "title_mention_canon_culture_authors", "canon_educational_2025",
    "canon_educational_2004", 
    "author_mention_dsl_classic",
    "title_mention_canon_culture_authors_dsl_classic",
]

# missing = not in the source, so 0 is the right fill
for col in proxy_cols:
    print(f"{col}: {df[col].isna().sum()} NaN, {(df[col] == 0).sum()} zeros")
df[proxy_cols] = df[proxy_cols].fillna(0)

# proxy types
NATIONAL_CANON_LISTS = ["canon_culture_authors", "canon_educational_2004", "canon_educational_2025"]
title_proxies = [c for c in proxy_cols if c.startswith("title_")]
auth_proxies = [c for c in proxy_cols if c.startswith("auth_")] + NATIONAL_CANON_LISTS
binary_proxies = title_proxies + auth_proxies
continuous_proxies = [c for c in proxy_cols if c not in binary_proxies]


# no scaling: binary proxies stay 0/1, continuous proxies keep their raw values
df[binary_proxies] = df[binary_proxies].astype(int)
df[continuous_proxies] = df[continuous_proxies].astype(float)   # some arrive as bool/int

# merge near-redundant canon lists: canon_culture_authors, canon_educational_2004 and
# canon_educational_2025 are highly intercorrelated (r = 0.74-0.87), so they become one
# count (0-3) of how many national canon lists an author is on
df["canon_list_count"] = df[NATIONAL_CANON_LISTS].sum(axis=1)
proxy_cols = [c for c in proxy_cols if c not in NATIONAL_CANON_LISTS] + ["canon_list_count"]
binary_proxies = [c for c in binary_proxies if c not in NATIONAL_CANON_LISTS]

PROXY_SETS = {
    "fa": proxy_cols,
    # low-coverage proxies left out
    "wo_low_coverage": [c for c in proxy_cols if c not in [
        "title_mention_canon_culture_authors",
        "title_mention_canon_culture_authors_dsl_classic",
        "title_mention_rød_dan_lithis_titleregistry",
    ]],
}

# %%
# ---------------------------------------------------------------- OPTIONAL: DEDUPLICATE PROXY VECTORS

# Two titles by the same author often have identical proxy vectors (author-level proxies are
# constant per author). Some methods treat such duplicates as mutually reinforcing, inflating
# prolific authors. With USE_DEDUPLICATED_VECTORS = True (CONFIG), exact within-author duplicates
# are collapsed to one row; titles differing on ANY proxy stay distinct.


def get_deduplicated_df(df, cols, author_col="author_id"):
    """Within each author's titles, drop rows that are exact duplicates on cols.
    Titles by different authors are never merged, even if their vectors match."""
    df = df.reset_index(drop=True)
    out = (df.groupby(author_col, group_keys=False)
             .apply(lambda g: g.drop_duplicates(subset=cols, keep="first"))
             .sort_index()
             .reset_index(drop=True))
    print(f"[dedup, within-author only] {len(df)} rows -> {len(out)} unique rows "
          f"({len(df) - len(out)} collapsed)")
    return out


df_analysis = get_deduplicated_df(df, PROXY_SETS["fa"]) if USE_DEDUPLICATED_VECTORS else df
df_analysis.head()

# %%
# ---------------------------------------------------------------- VIS 1: CONTINUOUS PROXY HISTOGRAMS

# for col in continuous_proxies + ["canon_list_count"]:
#     plt.figure(figsize=(5, 4))
#     plt.hist(df_analysis[col], bins=30, color="skyblue", edgecolor="black")
#     plt.title(f"Histogram of {col}")
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
#     plt.grid(axis="y", alpha=0.75)
#     plt.tight_layout()
#     plt.show()

# %%
# ---------------------------------------------------------------- VIS 2: PROXY HEATMAP

# columns grouped by type (continuous | title-level | author-level). The data are not scaled,
# but for DISPLAY only each column is shown relative to its own maximum -- otherwise the
# word counts (thousands) would wash out the 0/1 columns. Rows: number of proxies > 0.
heat_groups = [
    continuous_proxies,
    [c for c in proxy_cols if c in title_proxies],
    [c for c in proxy_cols if c in binary_proxies and c not in title_proxies] + ["canon_list_count"],
]
heat_cols = [c for g in heat_groups for c in g]

mat = df_analysis[heat_cols + ["title", "auth_last"]].copy()
mat["n_proxies_flagged"] = (mat[heat_cols] > 0).sum(axis=1)
mat = mat.sort_values("n_proxies_flagged", ascending=False).reset_index(drop=True)
display_vals = mat[heat_cols] / mat[heat_cols].max().replace(0, 1)

plt.figure(figsize=(10, 12))
sns.heatmap(display_vals, cmap="Greys_r", cbar=False)
plt.xticks(np.arange(len(heat_cols)) + 0.5, heat_cols, rotation=90)

# label every 20th row with "title (author)", truncated to ~30 characters
for i in range(0, len(mat), 20):
    title = str(mat.at[i, "title"]).strip()
    author = mat.at[i, "auth_last"]
    authorlabel = str(author).split(" ")[-1].title() if pd.notna(author) else ""
    if authorlabel == "Gyllembourg-Ehrensvärd-Ehrensvärd":
        authorlabel = "Gyll-Ehrensvärd"
    max_title_len = max(5, 25 - len(authorlabel))
    if len(title) > max_title_len:
        title = title[:max_title_len] + "..."
    plt.text(len(heat_cols) + 0.5, i + 0.5, f"{title} ({authorlabel})", va="center", fontsize=8)

# thick white lines between proxy groups, thin ones between columns
for x in range(len(heat_cols) + 1):
    plt.axvline(x, color="white", linewidth=0.3)
for x in np.cumsum([len(g) for g in heat_groups])[:-1]:
    plt.axvline(x, color="white", linewidth=2)

# y axis: share of titles, 0-100% from the top
plt.yticks([len(mat) * x / 10 for x in range(11)], [f"{x * 10}%" for x in range(11)])
plt.xlabel("Canon proxy")
plt.ylabel("Titles (sorted by number of proxies > 0)")
savefig("proxy_heatmap")

# %%
# ---------------------------------------------------------------- PROXY DIAGNOSTICS


def proxy_diagnostics(df, cols, auth_col="auth_last", weak_itc_threshold=0.3, low_coverage_threshold=5.0):
    """Cronbach's alpha, item-total correlations (Spearman, item vs. sum of the others)
    and coverage per proxy. Returns (table, alpha).
    NOTE: on unscaled data the total (and so alpha and the item-total correlations) is
    dominated by the large-valued continuous proxies (word counts, sitelinks)."""
    X_df = df[cols]
    n_total, n_items = X_df.shape

    item_vars = X_df.var(axis=0, ddof=1)
    total = X_df.sum(axis=1)
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total.var(ddof=1))

    n_auth_total = df[auth_col].nunique()
    rows = []
    for col in cols:
        flagged = df[col] > 0
        rows.append({
            "proxy": col,
            "item_total_corr": X_df[col].corr(total - X_df[col], method="spearman"),
            "pct_flagged_rows": flagged.mean() * 100,
            "pct_unique_authors_flagged": df.loc[flagged, auth_col].nunique() / n_auth_total * 100,
        })
    result = pd.DataFrame(rows).sort_values("item_total_corr").reset_index(drop=True)
    result["weak_itc"] = result["item_total_corr"] < weak_itc_threshold
    result["low_coverage"] = result["pct_flagged_rows"] < low_coverage_threshold
    result["weak_itc_and_low_coverage"] = result["weak_itc"] & result["low_coverage"]

    print(f"Cronbach's alpha: {alpha:.3f}  (n_items={n_items}, n_obs={n_total})")
    return result, alpha


for set_name in ("fa", "wo_low_coverage"):
    print(f"\n--- proxy set: {set_name} ---")
    diagnostics_df, alpha = proxy_diagnostics(df_analysis, PROXY_SETS[set_name])
    print(diagnostics_df.round(3).to_string(index=False))

# %%
# ---------------------------------------------------------------- SIMPLE AGGREGATE SCORE

# number of binary proxies a title is flagged on (+ the canon-list count); no scaling, so the
# continuous proxies are left out of this simple score
df_analysis["simple_binary_sum"] = df_analysis[binary_proxies + ["canon_list_count"]].sum(axis=1)

display(df_analysis[["title", "auth_last", "simple_binary_sum"]]
        .sort_values("simple_binary_sum", ascending=False).head(20))

# %%
# ---------------------------------------------------------------- FACTOR ANALYSIS

# Why FA, not IRT? (both latent-variable models)
# - FA treats all inputs as continuous and works on the correlation matrix, so unscaled
#   continuous proxies and 0/1 binary proxies sit on the same footing without a special
#   binary link function (and without scaling the data first)
# - loadings play the role of IRT discriminations: how strongly each proxy tracks the construct
# - standard diagnostics (KMO, Bartlett) that reviewers recognise
#
# Note: wordcount_main_text_lex_auth and "has a Lex author page" are near-deterministic
# (r = 0.98); only the wordcount is kept.

FA_PROXIES = PROXY_SETS["fa"]
X = df_analysis[FA_PROXIES].to_numpy(dtype=np.float64)

# factor-count sweep: which k give a proper (no Heywood cases), well-identified solution?
sweep_df, sweep_loadings, sweep_uniquenesses, sweep_variances, best_k = run_factor_sweep(
    X, FA_PROXIES, factor_range=range(2, 9))
display(sweep_df)
print(f"best k from sweep: {best_k}")

# %%

# CONFIG!

N_FACTORS = 3   # based on the factor sweep and parallel analysis
F_NAMES = [f"F{i + 1}" for i in range(N_FACTORS)]

kmo_all, kmo_model = calculate_kmo(X)
chi2, p_value = calculate_bartlett_sphericity(X)
print(f"KMO: {kmo_model:.3f} (>0.6 acceptable) | Bartlett chi2={chi2:.1f}, p={p_value:.4e}")

fa = FactorAnalyzer(n_factors=N_FACTORS, method="ml", rotation="promax")
fa.fit(X)

loadings = pd.DataFrame(fa.loadings_, index=FA_PROXIES, columns=F_NAMES)
uniquenesses = pd.Series(fa.get_uniquenesses(), index=FA_PROXIES, name="uniqueness")
variance, _, cumulative = fa.get_factor_variance()

print("Factor loadings:")
print(loadings.round(3))
print("\nVariance explained per factor:", dict(zip(F_NAMES, np.round(variance, 3))))
print("Cumulative variance explained:", dict(zip(F_NAMES, np.round(cumulative, 3))))
print("\nUniquenesses:")
print(uniquenesses.round(3))

# %%
# parallel analysis: real eigenvalues vs. those of random normal data of the same shape


def parallel_analysis(X, fa_fitted, n_iter=10):
    n_obs, n_vars = X.shape
    real_ev, _ = fa_fitted.get_eigenvalues()
    random_evs = np.zeros((n_iter, n_vars))
    for i in range(n_iter):
        fa_random = FactorAnalyzer(n_factors=n_vars, rotation=None)
        fa_random.fit(rng.normal(size=(n_obs, n_vars)))
        random_evs[i], _ = fa_random.get_eigenvalues()
    return real_ev, random_evs.mean(axis=0)


real_ev, random_ev = parallel_analysis(X, fa)

plt.figure(figsize=(6, 3))
plt.plot(range(1, len(real_ev) + 1), real_ev, marker="x", label="Actual data", color="teal")
plt.plot(range(1, len(random_ev) + 1), random_ev, marker="x", linestyle="--", label="Random data (mean)", color="grey")
plt.axhline(1, color="black", linestyle="--", alpha=0.7, linewidth=1)
plt.xticks(range(1, len(real_ev) + 1))
plt.xlabel("Factor number")
plt.ylabel("Eigenvalue")
plt.title("Parallel analysis")
plt.legend()
savefig("parallel_analysis")

# %%
# bootstrapped loadings: SD of each loading over resamples (factors aligned to the reference fit)


def bootstrap_loadings(X, n_factors, n_boot=100, rotation="promax", method="ml"):
    ref = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method).fit(X).loadings_
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), size=len(X), replace=True)
        try:
            L = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method).fit(X[idx]).loadings_
        except Exception:
            continue
        # align factor order and sign to the reference by max |correlation|
        aligned = []
        for ref_col in range(n_factors):
            corrs = [np.corrcoef(ref[:, ref_col], L[:, c])[0, 1] for c in range(n_factors)]
            best = int(np.argmax(np.abs(corrs)))
            aligned.append(L[:, best] * np.sign(corrs[best]))
        boots.append(np.column_stack(aligned))
    return np.array(boots)


boot = bootstrap_loadings(X, N_FACTORS, n_boot=100)
loading_sd = pd.DataFrame(boot.std(axis=0), index=FA_PROXIES, columns=F_NAMES)

# rows grouped by dominant factor
dominant = loadings.abs().idxmax(axis=1)
row_order = dominant.sort_values().index
L_sorted, SD_sorted = loadings.loc[row_order], loading_sd.loc[row_order]

BLANKOUT = 0.1   # loadings below this |value| are not annotated
DARK = 0.5       # above this |value| the cell is dark enough for white text

plt.figure(figsize=(6, 8), dpi=150)
ax = plt.gca()
sns.heatmap(L_sorted, cmap="RdGy_r", center=0, annot=False, cbar=False, ax=ax)
for r, proxy in enumerate(row_order):
    for c, f in enumerate(F_NAMES):
        val, sd = L_sorted.at[proxy, f], SD_sorted.at[proxy, f]
        if abs(val) < BLANKOUT:
            continue
        dark = abs(val) > DARK
        ax.text(c + 0.5, r + 0.35, f"{val:.2f}", ha="center", va="center", fontsize=10,
                color="white" if dark else "black")
        ax.text(c + 0.5, r + 0.7, f"±{sd:.2f}", ha="center", va="center", fontsize=8,
                color="lightgray" if dark else "dimgray")
plt.yticks(np.arange(len(row_order)) + 0.5, row_order, rotation=0)
plt.xticks(np.arange(N_FACTORS) + 0.5, F_NAMES, rotation=0)
savefig("fa_loadings_heatmap")

salient = L_sorted.abs().to_numpy() >= BLANKOUT
print(f"Average bootstrap SD, all loadings: {SD_sorted.to_numpy().mean():.3f}")
print(f"Average bootstrap SD, salient loadings (|loading| >= {BLANKOUT}): {SD_sorted.to_numpy()[salient].mean():.3f}")

# %%
# second-order factor: one general factor fitted on the correlations between the promax factors

phi = fa.phi_
print("Factor correlation matrix (phi):")
print(pd.DataFrame(phi, index=F_NAMES, columns=F_NAMES).round(3))

fa_2nd = FactorAnalyzer(n_factors=1, rotation=None, method="ml", is_corr_matrix=True)
fa_2nd.fit(phi)

# the sign of a single factor is arbitrary: orient it so the loadings are mostly positive
g_raw = fa_2nd.loadings_.flatten()
general_loadings = pd.Series(g_raw * np.sign(g_raw.sum()), index=F_NAMES, name="loading_on_general_factor")
general_var = fa_2nd.get_factor_variance()[2][-1]
print("General-factor loadings:")
print(general_loadings.round(3))
print(f"Variance in factor correlations explained by the general factor: {general_var:.3f}")

save_latex_table(general_loadings.rename_axis("factor").reset_index().round(3), LATEX_PATH,
                 caption=f"Second-order factor loadings on the general factor "
                         f"(explains {general_var:.3f} of the variance in factor correlations).",
                 label=f"tab:{USED_WHAT}_second_order_loadings")

# %%
# final FA canon score: first-order factor scores weighted by their general-factor loadings

factor_scores = fa.transform(X)                      # (n_titles, N_FACTORS)
weights = general_loadings.to_numpy() / general_loadings.sum()
df_analysis["fa_canon_score"] = factor_scores @ weights
df_analysis["fa_canon_percentile"] = df_analysis["fa_canon_score"].rank(pct=True)

# sanity check: higher FA score should mean more canon evidence
check = df_analysis["fa_canon_score"].corr(df_analysis["simple_binary_sum"], method="spearman")
print(f"Spearman FA score vs simple_binary_sum (expect strongly positive): {check:.3f}")

top20 = (df_analysis[["title", "auth_last", "fa_canon_score"]]
         .sort_values("fa_canon_score", ascending=False).head(20).round(3))
display(top20)
save_latex_table(top20, LATEX_PATH, caption="Top 20 titles by factor-analysis canon score.",
                 label=f"tab:{USED_WHAT}_fa_canon_score_top20")

# %%
