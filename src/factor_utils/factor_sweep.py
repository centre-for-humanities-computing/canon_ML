"""
Systematic sweep over factor count (2 through max_factors), checking at
each step whether the solution is statistically PROPER (no Heywood
cases -- no negative or degenerate/boundary uniquenesses) AND
WELL-IDENTIFIED (every factor has at least 3 salient loadings), rather
than just eyeballing loadings one factor-count at a time.

Under method="minres", Heywood cases surface as negative uniquenesses.
Under method="ml", they instead surface as uniquenesses pinned at the
package's internal lower bound (not negative). The n_degenerate_cases
check (uniqueness <= 1e-6) catches this boundary case regardless of
estimation method, so "proper_solution" is method-agnostic.

This turns "we tried 3, 5, and 6 individually and picked 3" into a
reportable, systematic result: either some k>3 also gives a proper,
well-identified solution (in which case that's worth adopting instead),
or NOTHING beyond 3 is statistically valid -- which is itself a strong,
direct rebuttal to "a more generalized metric could be derived via
dimensionality reduction": the data structurally does not support more
dimensions than that, evidenced systematically rather than assumed.

Usage:
    from factor_sweep import run_factor_sweep
    sweep_df, all_loadings, all_uniquenesses, best_k = run_factor_sweep(
        X_final, fa_features_final, factor_range=range(2, 9)
    )
    print(sweep_df.to_string(index=False))
"""

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer


def run_factor_sweep(X, feature_names, factor_range=range(2, 9), rotation="promax", method="minres"):
    """
    X: the same dropna()'d feature matrix used for the main factor analysis.
    feature_names: column names, in the same order as X's columns.
    factor_range: which factor counts to test.
    method: factor extraction method passed to FactorAnalyzer (default "minres").

    Returns a DataFrame with one row per factor count, including whether
    the solution is PROPER (no Heywood/degenerate uniquenesses) and
    WELL-IDENTIFIED (every factor has >=3 salient loadings), plus
    cumulative variance explained.
    """
    results = []
    all_loadings = {}
    all_uniquenesses = {}
    all_factor_variances = {}

    for k in factor_range:
        fa = FactorAnalyzer(n_factors=k, rotation=rotation, method=method)
        try:
            fa.fit(X)
        except Exception as e:
            print(f"Factor count {k} failed to fit: {e}")
            results.append({
                "n_factors": k,
                "cumulative_var": np.nan,
                "incremental_var": np.nan,
                "last_factor_var": np.nan,
                "n_heywood_cases": np.nan,
                "n_degenerate_cases": np.nan,
                "n_weak_factors": np.nan,
                "min_uniqueness": np.nan,
                "proper_solution": False,
                "well_identified": False,
                "fit_error": str(e),
            })
            continue

        var_per_factor, _, cum_var = fa.get_factor_variance()

        uniq = fa.get_uniquenesses()
        n_heywood = int((uniq < 0).sum())
        n_degenerate = int((uniq <= 1e-6).sum())
        n_weak_factors = sum(
            (np.abs(fa.loadings_[:, i]) > 0.4).sum() < 3
            for i in range(k)
        )

        results.append({
            "n_factors": k,
            "cumulative_var": cum_var[-1],
            "incremental_var": np.nan,  # filled in after the loop, needs previous row's cumulative_var
            "last_factor_var": var_per_factor[-1],  # how much the newest factor alone explains
            "n_heywood_cases": n_heywood,
            "n_degenerate_cases": n_degenerate,
            "n_weak_factors": n_weak_factors,
            "min_uniqueness": uniq.min(),
            "proper_solution": (n_heywood == 0) and (n_degenerate == 0),
            "well_identified": (n_heywood == 0) and (n_degenerate == 0) and (n_weak_factors == 0),
            "fit_error": None,
        })

        all_loadings[k] = pd.DataFrame(fa.loadings_, index=feature_names,
                                        columns=[f"F{i+1}" for i in range(k)])
        all_uniquenesses[k] = pd.Series(uniq, index=feature_names)
        all_factor_variances[k] = pd.Series(var_per_factor, index=[f"F{i+1}" for i in range(k)])

    sweep_df = pd.DataFrame(results)
    # incremental variance = how much cumulative_var grew vs. the previous row's factor count
    sweep_df["incremental_var"] = sweep_df["cumulative_var"].diff()

    well_identified = sweep_df[sweep_df["well_identified"]]
    if len(well_identified) > 0:
        best_k = int(well_identified["n_factors"].max())
        print(f"\nLargest WELL-IDENTIFIED solution (no Heywood cases, no weak factors): {best_k} factors")
    else:
        best_k = None
        print("\nNo tested factor count produced a fully well-identified solution.")

    return sweep_df, all_loadings, all_uniquenesses, all_factor_variances, best_k