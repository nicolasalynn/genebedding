"""
TCGA cancer gene enrichment: epistasis metrics vs simple baselines.

Head-to-head comparison: do epistasis-specific metrics (R_singles, magnitude_ratio)
outperform simple double-variant embedding metrics (len_WT_M12, max single effect,
sum of singles) for cancer gene enrichment?

Metrics compared:
  EPISTASIS (require 4 embeddings + additivity computation):
    - log_magnitude_ratio (distance-residualized)
    - epi_R_singles
    - epi_mahal (Mahalanobis-calibrated residual)

  BASELINES (require only 1-2 embeddings, no additivity):
    - len_WT_M12: raw double-mutant effect magnitude
    - max_single: max(len_WT_M1, len_WT_M2)
    - sum_singles: len_WT_M1 + len_WT_M2
    - cos_v1_v2: cosine between single-variant effect vectors

All metrics are distance-residualized before ranking.
Enrichment = hypergeometric test at top X% for oncogenes and TSGs.
"""

# ---------------------------------------------------------------------------
# Cell 1: Setup + data loading
# ---------------------------------------------------------------------------
import warnings; warnings.filterwarnings("ignore")
import sys, os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyarrow.dataset as ds
from scipy.stats import hypergeom, mannwhitneyu, wilcoxon

ROOT = Path.cwd()
for _ in range(5):
    if (ROOT / "notebooks" / "paper_data_config.py").exists():
        break
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from paper_data_config import EPISTASIS_PAPER_ROOT

PARQUET_DIR = EPISTASIS_PAPER_ROOT / "combined_parquets" / "new_embeddings"
ANNOT_DIR = EPISTASIS_PAPER_ROOT / "data" / "annotations"
FIG_DIR = ROOT / "notebooks" / "dirty" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "alphagenome": "AlphaGenome", "borzoi": "Borzoi", "caduceus": "Caduceus",
    "convnova": "ConvNova", "dnabert": "DNABERT", "evo2": "Evo2",
    "hyenadna": "HyenaDNA", "mutbert": "MutBERT",
    "nt50_3mer": "NT-50M-3mer", "nt50_multi": "NT-50M",
    "nt100_multi": "NT-100M", "nt250_multi": "NT-250M",
    "nt500_multi": "NT-500M", "nt500_ref": "NT-500M-ref",
    "nt2500_multi": "NT-2.5B", "nt2500_okgp": "NT-2.5B-1kGP",
    "rinalmo": "RiNALMo", "specieslm": "SpeciesLM",
}
TRACK_MODELS = {"alphagenome", "borzoi"}

# Cancer gene annotations
census = pd.read_csv(ANNOT_DIR / "census_all_genes.csv")
oncokb = pd.read_csv(ANNOT_DIR / "cancerGeneList.tsv", sep="\t")

# Build gene sets
_census_onc = set(census[census["Role in Cancer"].str.contains("oncogene", case=False, na=False)]["Gene Symbol"].str.strip())
_census_tsg = set(census[census["Role in Cancer"].str.contains("TSG", case=False, na=False)]["Gene Symbol"].str.strip())
_oncokb_onc = set(oncokb[oncokb["Is Oncogene"] == "Yes"]["Hugo Symbol"].str.strip()) if "Is Oncogene" in oncokb.columns else set()
_oncokb_tsg = set(oncokb[oncokb["Is Tumor Suppressor Gene"] == "Yes"]["Hugo Symbol"].str.strip()) if "Is Tumor Suppressor Gene" in oncokb.columns else set()

ONCOGENES = _census_onc | _oncokb_onc
TSGS = _census_tsg | _oncokb_tsg
CANCER_GENES = ONCOGENES | TSGS

print(f"Oncogenes: {len(ONCOGENES)}, TSGs: {len(TSGS)}, Total cancer genes: {len(CANCER_GENES)}")
print(f"PARQUET_DIR = {PARQUET_DIR}")
print(f"Available parquets: {len(list(PARQUET_DIR.glob('*.parquet')))}")


# ---------------------------------------------------------------------------
# Cell 2: Load TCGA data for all models
# ---------------------------------------------------------------------------
COLS_NEEDED = [
    "source", "epistasis_id",
    # Epistasis metrics
    "log_magnitude_ratio", "epi_R_singles", "magnitude_ratio",
    # Mahalanobis
    "global_epi_mahal",
    # Baselines (single + double variant magnitudes)
    "len_WT_M1", "len_WT_M2", "len_WT_M12", "len_WT_M12_exp",
    # Cosine
    "cos_v1_v2", "cos_obs_exp",
]

def parse_gene(eid):
    """Extract gene from epistasis_id (first field before ':')."""
    return eid.split(":")[0]

def parse_distance(eid):
    """Extract distance between variant positions."""
    parts = eid.split("|")
    try:
        p1 = int(parts[0].split(":")[2])
        p2 = int(parts[1].split(":")[2])
        return abs(p2 - p1)
    except (IndexError, ValueError):
        return np.nan

# Distance bins for residualization (from paper Methods 2.7)
DIST_BINS = [1, 2, 3, 4, 5, 6, 11, 16, 26, 51, 101, 10000]
DIST_LABELS = ["1", "2", "3", "4", "5", "6-10", "11-15", "16-25", "26-50", "51-100", "101+"]

def distance_residualize(series, distances):
    """Subtract within-bin mean (distance-bin residualization)."""
    bins = pd.cut(distances, bins=DIST_BINS, labels=DIST_LABELS, right=False)
    residuals = series.copy()
    for label in DIST_LABELS:
        mask = bins == label
        if mask.sum() > 0:
            residuals[mask] = series[mask] - series[mask].mean()
    return residuals

all_model_data = {}
for model_key, display in MODELS.items():
    ppath = PARQUET_DIR / f"epistasis_metrics_{model_key}_combined.parquet"
    if not ppath.exists():
        continue
    dset = ds.dataset(ppath)
    avail = {f.name for f in dset.schema}
    cols = [c for c in COLS_NEEDED if c in avail]
    df = dset.to_table(
        columns=cols,
        filter=ds.field("source") == "tcga_doubles",
    ).to_pandas().drop_duplicates(subset=["epistasis_id"])

    if len(df) < 100:
        continue

    # Parse gene and distance
    df["gene"] = df["epistasis_id"].map(parse_gene)
    df["distance"] = df["epistasis_id"].map(parse_distance)

    # Compute baseline metrics
    if "len_WT_M1" in df.columns and "len_WT_M2" in df.columns:
        df["max_single"] = np.maximum(df["len_WT_M1"], df["len_WT_M2"])
        df["sum_singles"] = df["len_WT_M1"] + df["len_WT_M2"]

    # Label cancer genes
    df["is_oncogene"] = df["gene"].isin(ONCOGENES)
    df["is_tsg"] = df["gene"].isin(TSGS)
    df["is_cancer"] = df["gene"].isin(CANCER_GENES)

    all_model_data[model_key] = df

print(f"Loaded TCGA data for {len(all_model_data)} models")
for mk, df in all_model_data.items():
    n_onc = df["is_oncogene"].sum()
    n_tsg = df["is_tsg"].sum()
    print(f"  {mk}: {len(df)} pairs, {n_onc} oncogene, {n_tsg} TSG, "
          f"{len(df) - n_onc - n_tsg} other")


# ---------------------------------------------------------------------------
# Cell 3: Define enrichment test
# ---------------------------------------------------------------------------
def enrichment_at_percentile(
    df, metric_col, gene_set, top_pct, direction="high", distance_residualize_col=True
):
    """
    Hypergeometric enrichment of gene_set at top X% of metric.

    Parameters
    ----------
    df : DataFrame with columns [metric_col, 'gene', 'distance']
    gene_set : set of gene symbols
    top_pct : float (0-100), e.g. 5 for top 5%
    direction : 'high' (top values) or 'low' (bottom values)
    distance_residualize_col : if True, residualize metric by distance bins first

    Returns
    -------
    dict with fold, p_value, k_obs, k_exp, n_top, N_total
    """
    valid = df.dropna(subset=[metric_col, "distance"]).copy()
    if len(valid) < 50:
        return {"fold": np.nan, "p_value": np.nan}

    values = valid[metric_col].values.copy()
    if distance_residualize_col and "distance" in valid.columns:
        values = distance_residualize(pd.Series(values, index=valid.index), valid["distance"]).values

    # Rank
    if direction == "high":
        ranks = np.argsort(np.argsort(-values))  # 0 = highest
    else:
        ranks = np.argsort(np.argsort(values))   # 0 = lowest

    n_top = max(1, int(len(valid) * top_pct / 100))
    in_top = ranks < n_top

    in_set = valid["gene"].isin(gene_set).values
    k_obs = int((in_top & in_set).sum())
    K_total = int(in_set.sum())
    N = len(valid)

    # Expected
    k_exp = K_total * n_top / N

    # Hypergeometric p-value: P(X >= k_obs)
    p_val = hypergeom.sf(k_obs - 1, N, K_total, n_top)

    fold = k_obs / (k_exp + 1e-10)

    return {
        "fold": fold,
        "p_value": p_val,
        "k_obs": k_obs,
        "k_exp": round(k_exp, 1),
        "n_top": n_top,
        "N_total": N,
        "K_in_set": K_total,
    }

print("Enrichment function defined.")


# ---------------------------------------------------------------------------
# Cell 4: Run enrichment for all metrics, all models
# ---------------------------------------------------------------------------
METRICS = {
    # Epistasis-specific metrics
    "log_magnitude_ratio (corrective)": ("log_magnitude_ratio", "low"),
    "log_magnitude_ratio (cumulative)": ("log_magnitude_ratio", "high"),
    "epi_R_singles": ("epi_R_singles", "high"),
    # Mahalanobis (if available)
    "epi_mahal (global)": ("global_epi_mahal", "high"),
    # Baselines
    "len_WT_M12 (double effect)": ("len_WT_M12", "high"),
    "max_single": ("max_single", "high"),
    "sum_singles": ("sum_singles", "high"),
    "cos_v1_v2 (same direction)": ("cos_v1_v2", "high"),
    "cos_v1_v2 (opposite)": ("cos_v1_v2", "low"),
}

PERCENTILES = [1, 2, 5, 10]

# Gene sets to test
GENE_SETS = {
    "TSG": TSGS,
    "Oncogene": ONCOGENES,
    "Cancer (any)": CANCER_GENES,
}

all_results = []
for model_key, df in all_model_data.items():
    display = MODELS.get(model_key, model_key)
    for metric_name, (col, direction) in METRICS.items():
        if col not in df.columns:
            continue
        for gene_label, gene_set in GENE_SETS.items():
            for pct in PERCENTILES:
                result = enrichment_at_percentile(df, col, gene_set, pct, direction)
                result["model"] = display
                result["model_key"] = model_key
                result["metric"] = metric_name
                result["gene_set"] = gene_label
                result["percentile"] = pct
                result["is_track"] = model_key in TRACK_MODELS
                result["is_epistasis"] = "magnitude" in metric_name or "R_singles" in metric_name or "mahal" in metric_name
                all_results.append(result)

df_results = pd.DataFrame(all_results)
print(f"Total enrichment tests: {len(df_results)}")
print(f"Metrics tested: {df_results['metric'].nunique()}")
print(f"Models: {df_results['model'].nunique()}")


# ---------------------------------------------------------------------------
# Cell 5: Summary table — average enrichment across models per metric
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("AVERAGE ENRICHMENT FOLD ACROSS ALL MODELS (top 5%)")
print("=" * 100)

at5 = df_results[df_results["percentile"] == 5].copy()

summary = at5.groupby(["metric", "gene_set"]).agg(
    mean_fold=("fold", "mean"),
    median_fold=("fold", "median"),
    n_sig=("p_value", lambda x: (x < 0.05).sum()),
    n_models=("model", "nunique"),
    mean_p=("p_value", "mean"),
).reset_index()

for gs in ["TSG", "Oncogene", "Cancer (any)"]:
    print(f"\n--- {gs} ---")
    sub = summary[summary["gene_set"] == gs].sort_values("mean_fold", ascending=False)
    print(sub[["metric", "mean_fold", "median_fold", "n_sig", "n_models"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Cell 6: Head-to-head: epistasis vs best baseline per model
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("HEAD-TO-HEAD: EPISTASIS vs BASELINES (per model, top 5%)")
print("=" * 100)

# For each model + gene_set, find best epistasis metric and best baseline
h2h_rows = []
for model_key in all_model_data:
    display = MODELS.get(model_key, model_key)
    for gs in ["TSG", "Oncogene"]:
        sub = at5[(at5["model"] == display) & (at5["gene_set"] == gs)]
        epi = sub[sub["is_epistasis"]]
        base = sub[~sub["is_epistasis"]]

        if len(epi) == 0 or len(base) == 0:
            continue

        best_epi = epi.loc[epi["fold"].idxmax()]
        best_base = base.loc[base["fold"].idxmax()]

        h2h_rows.append({
            "model": display,
            "gene_set": gs,
            "best_epi_metric": best_epi["metric"],
            "epi_fold": round(best_epi["fold"], 2),
            "epi_p": best_epi["p_value"],
            "best_base_metric": best_base["metric"],
            "base_fold": round(best_base["fold"], 2),
            "base_p": best_base["p_value"],
            "epi_wins": best_epi["fold"] > best_base["fold"],
        })

df_h2h = pd.DataFrame(h2h_rows)
for gs in ["TSG", "Oncogene"]:
    sub = df_h2h[df_h2h["gene_set"] == gs]
    n_epi_wins = sub["epi_wins"].sum()
    n_total = len(sub)
    print(f"\n--- {gs} (top 5%) ---")
    print(f"Epistasis metric wins: {n_epi_wins}/{n_total} models")
    print(sub[["model", "best_epi_metric", "epi_fold", "best_base_metric", "base_fold", "epi_wins"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Cell 7: Paired comparison across models (Wilcoxon signed-rank)
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("PAIRED COMPARISON: EPISTASIS vs BASELINES (Wilcoxon signed-rank)")
print("=" * 100)

# For each model: best epistasis fold vs best baseline fold
for gs in ["TSG", "Oncogene", "Cancer (any)"]:
    sub = df_h2h[df_h2h["gene_set"] == gs] if gs != "Cancer (any)" else pd.DataFrame()

    # Also do: specific metric pairs
    for epi_metric, base_metric in [
        ("log_magnitude_ratio (corrective)", "len_WT_M12 (double effect)"),
        ("log_magnitude_ratio (cumulative)", "len_WT_M12 (double effect)"),
        ("epi_R_singles", "max_single"),
        ("log_magnitude_ratio (corrective)", "sum_singles"),
        ("log_magnitude_ratio (cumulative)", "sum_singles"),
    ]:
        epi_folds = []
        base_folds = []
        for model_key in all_model_data:
            display = MODELS.get(model_key, model_key)
            for g_label, g_set in [("TSG", TSGS), ("Oncogene", ONCOGENES)]:
                e = at5[(at5["model"] == display) & (at5["metric"] == epi_metric) & (at5["gene_set"] == g_label)]
                b = at5[(at5["model"] == display) & (at5["metric"] == base_metric) & (at5["gene_set"] == g_label)]
                if len(e) > 0 and len(b) > 0:
                    epi_folds.append(e.iloc[0]["fold"])
                    base_folds.append(b.iloc[0]["fold"])

        if len(epi_folds) >= 5:
            stat, p = wilcoxon(epi_folds, base_folds, alternative="greater")
            mean_diff = np.mean(np.array(epi_folds) - np.array(base_folds))
            print(f"  {epi_metric:45s} vs {base_metric:30s}: "
                  f"mean Δfold={mean_diff:+.3f}, Wilcoxon p={p:.4f}, n={len(epi_folds)}")


# ---------------------------------------------------------------------------
# Cell 8: Visualization — enrichment curves across percentile thresholds
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, gs in zip(axes, ["TSG", "Oncogene", "Cancer (any)"]):
    sub = df_results[df_results["gene_set"] == gs]

    # Average across models for each metric + percentile
    avg = sub.groupby(["metric", "percentile", "is_epistasis"]).agg(
        mean_fold=("fold", "mean"),
    ).reset_index()

    for metric in avg["metric"].unique():
        m = avg[avg["metric"] == metric]
        is_epi = m["is_epistasis"].iloc[0]
        style = "-" if is_epi else "--"
        alpha = 0.9 if is_epi else 0.5
        lw = 2.0 if is_epi else 1.2
        ax.plot(m["percentile"], m["mean_fold"], style, label=metric, alpha=alpha, linewidth=lw)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Top percentile (%)")
    ax.set_ylabel("Mean enrichment fold")
    ax.set_title(f"{gs} enrichment")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(PERCENTILES)

fig.suptitle("Cancer gene enrichment: epistasis metrics (solid) vs baselines (dashed)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "tcga_enrichment_comparison.png", bbox_inches="tight", dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# Cell 9: Ensemble comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("ENSEMBLE ENRICHMENT COMPARISON")
print("=" * 100)

# Build ensemble by averaging percentile ranks across models
# For each metric, rank pairs within each model, then average ranks
def ensemble_enrichment(metric_col, direction, gene_set, top_pct, distance_resid=True):
    """Compute ensemble enrichment by averaging percentile ranks across models."""
    rank_dfs = []
    for model_key, df in all_model_data.items():
        if metric_col not in df.columns:
            continue
        valid = df.dropna(subset=[metric_col, "distance"]).copy()
        if len(valid) < 100:
            continue

        values = valid[metric_col].values.copy()
        if distance_resid:
            values = distance_residualize(pd.Series(values, index=valid.index), valid["distance"]).values

        # Percentile rank (0-100)
        if direction == "high":
            pct_rank = 100 * np.argsort(np.argsort(-values)) / len(values)
        else:
            pct_rank = 100 * np.argsort(np.argsort(values)) / len(values)

        rank_dfs.append(pd.DataFrame({
            "epistasis_id": valid["epistasis_id"].values,
            f"rank_{model_key}": pct_rank,
        }))

    if not rank_dfs:
        return {"fold": np.nan, "p_value": np.nan}

    # Merge all rank columns
    merged = rank_dfs[0]
    for rdf in rank_dfs[1:]:
        merged = merged.merge(rdf, on="epistasis_id", how="inner")

    rank_cols = [c for c in merged.columns if c.startswith("rank_")]
    merged["ensemble_rank"] = merged[rank_cols].mean(axis=1)

    # Add gene info
    merged["gene"] = merged["epistasis_id"].map(parse_gene)

    # Enrichment at top percentile
    n_top = max(1, int(len(merged) * top_pct / 100))
    in_top = merged["ensemble_rank"] < top_pct  # already in percentile space
    in_set = merged["gene"].isin(gene_set)

    k_obs = int((in_top & in_set).sum())
    K_total = int(in_set.sum())
    N = len(merged)
    k_exp = K_total * n_top / N
    p_val = hypergeom.sf(k_obs - 1, N, K_total, n_top)

    return {
        "fold": k_obs / (k_exp + 1e-10),
        "p_value": p_val,
        "k_obs": k_obs,
        "k_exp": round(k_exp, 1),
        "n_top": n_top,
        "N_total": N,
    }

# Run ensemble for each metric
ensemble_rows = []
for metric_name, (col, direction) in METRICS.items():
    for gs_label, gs_set in GENE_SETS.items():
        for pct in [1, 2, 5, 10]:
            result = ensemble_enrichment(col, direction, gs_set, pct)
            result["metric"] = metric_name
            result["gene_set"] = gs_label
            result["percentile"] = pct
            result["is_epistasis"] = "magnitude" in metric_name or "R_singles" in metric_name or "mahal" in metric_name
            ensemble_rows.append(result)

df_ensemble = pd.DataFrame(ensemble_rows)

print("\nEnsemble enrichment at top 5%:")
ens5 = df_ensemble[df_ensemble["percentile"] == 5]
for gs in ["TSG", "Oncogene"]:
    print(f"\n--- {gs} ---")
    sub = ens5[ens5["gene_set"] == gs].sort_values("fold", ascending=False)
    sub_display = sub[["metric", "fold", "p_value", "k_obs", "k_exp"]].copy()
    sub_display["p_value"] = sub_display["p_value"].map(lambda p: f"{p:.4f}" if p >= 0.001 else f"{p:.2e}")
    print(sub_display.to_string(index=False))


# ---------------------------------------------------------------------------
# Cell 10: Final verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("VERDICT")
print("=" * 100)

# Count wins at top 5%
for gs in ["TSG", "Oncogene"]:
    ens_sub = ens5[ens5["gene_set"] == gs].sort_values("fold", ascending=False)
    best = ens_sub.iloc[0]
    is_epi = best["is_epistasis"]
    print(f"\n{gs}: Best metric = '{best['metric']}' (fold={best['fold']:.2f}, p={best['p_value']:.4f})")
    print(f"  {'EPISTASIS WINS' if is_epi else 'BASELINE WINS'}")

    # Compare best epistasis vs best baseline
    epi_best = ens_sub[ens_sub["is_epistasis"]].iloc[0] if ens_sub["is_epistasis"].any() else None
    base_best = ens_sub[~ens_sub["is_epistasis"]].iloc[0] if (~ens_sub["is_epistasis"]).any() else None
    if epi_best is not None and base_best is not None:
        print(f"  Best epistasis: '{epi_best['metric']}' fold={epi_best['fold']:.2f}")
        print(f"  Best baseline:  '{base_best['metric']}' fold={base_best['fold']:.2f}")
        delta = epi_best["fold"] - base_best["fold"]
        print(f"  Delta: {delta:+.2f} ({'epistasis ahead' if delta > 0 else 'baseline ahead'})")

print("\nDone. Figures saved to:", FIG_DIR)
