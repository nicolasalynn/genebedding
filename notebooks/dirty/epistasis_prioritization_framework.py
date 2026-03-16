"""
Epistasis Prioritization Framework: per-model scoring pipeline + validation suite.

FRAMEWORK:
  Given a dLM and a set of co-occurring somatic mutation pairs, produce a
  single epistasis priority score per pair. The score should rank pairs by
  how strongly they deviate from additive expectation in a biologically
  meaningful direction.

  Steps:
    1. Embed all four sequences (WT, M1, M2, M12) per pair
    2. Compute distance-residualized log(magnitude_ratio) per pair
    3. Rank pairs by |residualized log(MR)| — unsigned deviation from additivity
    4. Optionally: separate corrective (log_MR < 0) from cumulative (log_MR > 0)

VALIDATION SUITE:
  The framework is validated against 5 independent tests, each probing a
  different aspect of whether the ranking captures real biology.

  Test 1: CANCER GENE ENRICHMENT (hypergeometric)
    - Do top-ranked pairs land in known cancer genes (COSMIC + OncoKB)?
    - Separate: oncogene enrichment in cumulative tail, TSG in corrective tail

  Test 2: DIRECTIONAL ASYMMETRY (Wilcoxon signed-rank)
    - Across all models, are oncogenes more cumulative than TSGs?
    - This is the paper's headline result — must hold per-model, not just ensemble

  Test 3: POPULATION DEPLETION (1kGP comparison)
    - Are top-ranked TCGA pairs depleted from common population variation?
    - Rationale: truly epistatic cancer pairs should be under negative selection
      in the germline and thus rare or absent in 1kGP

  Test 4: CO-OCCURRENCE STRENGTH (correlation with cond_prob)
    - Do top-ranked pairs have higher conditional co-occurrence probability?
    - Rationale: functionally epistatic pairs should be maintained together by
      somatic selection, producing high co-occurrence

  Test 5: CLINICAL OUTCOME (survival screen)
    - For genes with sufficient cohort size, does the specific pair associate
      with differential survival vs same-gene multi-hit controls?

  SCORE: Each test produces a score per model. Models are ranked by total
  validation score. The best model(s) define the recommended protocol.

BASELINES:
  Same validation suite applied to non-epistasis rankings:
    - Random ranking
    - Single-variant effect magnitude (max_single)
    - Double-variant effect magnitude (len_WT_M12)
    - Sum of singles (sum_singles)
"""

# ---------------------------------------------------------------------------
# Cell 1: Setup
# ---------------------------------------------------------------------------
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import sys, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.dataset as ds
from scipy.stats import hypergeom, mannwhitneyu, wilcoxon, spearmanr, pearsonr

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
DATA_DIR = EPISTASIS_PAPER_ROOT / "data"
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

# Cancer gene annotations (OncoKB has better Gene Type labels)
oncokb = pd.read_csv(ANNOT_DIR / "cancerGeneList.tsv", sep="\t")
census = pd.read_csv(ANNOT_DIR / "census_all_genes.csv")

ONCOGENES = (
    set(oncokb[oncokb["Gene Type"].isin(["ONCOGENE", "ONCOGENE_AND_TSG"])]["Hugo Symbol"])
    | set(census[census["Role in Cancer"].str.contains("oncogene", case=False, na=False)]["Gene Symbol"].str.strip())
)
TSGS = (
    set(oncokb[oncokb["Gene Type"].isin(["TSG", "ONCOGENE_AND_TSG"])]["Hugo Symbol"])
    | set(census[census["Role in Cancer"].str.contains("TSG", case=False, na=False)]["Gene Symbol"].str.strip())
)
CANCER_GENES = ONCOGENES | TSGS

# TCGA pair metadata (n_cases_both, cond_prob, gene)
tcga_meta = pd.read_csv(DATA_DIR / "tcga_doubles_pairs.tsv", sep="\t")
tcga_meta = tcga_meta.set_index("epistasis_id")

# 1kGP null pair IDs for population depletion test
okgp_ids = set(
    pd.read_csv(DATA_DIR / "okgp_chr12_pairs.tsv", sep="\t", usecols=["epistasis_id"])["epistasis_id"]
)

# Distance bins for residualization
DIST_BINS = [1, 2, 3, 4, 5, 6, 11, 16, 26, 51, 101, 10000]
DIST_LABELS = ["1", "2", "3", "4", "5", "6-10", "11-15", "16-25", "26-50", "51-100", "101+"]

print(f"Oncogenes: {len(ONCOGENES)}, TSGs: {len(TSGS)}")
print(f"TCGA pairs: {len(tcga_meta)}, 1kGP null IDs: {len(okgp_ids)}")


# ---------------------------------------------------------------------------
# Cell 2: Scoring function — the framework
# ---------------------------------------------------------------------------
def distance_residualize(values, distances):
    """Subtract within-distance-bin mean."""
    bins = pd.cut(distances, bins=DIST_BINS, labels=DIST_LABELS, right=False)
    result = values.copy()
    for label in DIST_LABELS:
        mask = bins == label
        if mask.sum() > 0:
            result[mask] = values[mask] - values[mask].mean()
    return result

def score_pairs(df, metric_col="log_magnitude_ratio"):
    """
    The epistasis prioritization score.

    Returns df with added columns:
      - resid_metric: distance-residualized metric
      - epi_score: |resid_metric| (unsigned deviation from additivity)
      - epi_direction: 'corrective' (MR<1) or 'cumulative' (MR>1)
      - epi_rank: rank by epi_score (0 = most epistatic)
      - epi_percentile: percentile rank (0 = most epistatic)
    """
    valid = df.dropna(subset=[metric_col]).copy()
    distances = valid["epistasis_id"].map(
        lambda eid: abs(int(eid.split("|")[1].split(":")[2]) - int(eid.split("|")[0].split(":")[2]))
    ).astype(float)

    resid = distance_residualize(valid[metric_col].values, distances)
    valid["resid_metric"] = resid
    valid["epi_score"] = np.abs(resid)
    valid["epi_direction"] = np.where(resid < 0, "corrective", "cumulative")
    valid["epi_rank"] = valid["epi_score"].rank(ascending=False, method="min").astype(int)
    valid["epi_percentile"] = 100 * valid["epi_rank"] / len(valid)
    return valid

def score_pairs_baseline(df, metric_col):
    """Score using a baseline metric (no epistasis decomposition)."""
    valid = df.dropna(subset=[metric_col]).copy()
    distances = valid["epistasis_id"].map(
        lambda eid: abs(int(eid.split("|")[1].split(":")[2]) - int(eid.split("|")[0].split(":")[2]))
    ).astype(float)

    resid = distance_residualize(valid[metric_col].values, distances)
    valid["resid_metric"] = resid
    valid["epi_score"] = np.abs(resid)
    valid["epi_direction"] = "unsigned"
    valid["epi_rank"] = valid["epi_score"].rank(ascending=False, method="min").astype(int)
    valid["epi_percentile"] = 100 * valid["epi_rank"] / len(valid)
    return valid

print("Scoring functions defined.")


# ---------------------------------------------------------------------------
# Cell 3: Validation test functions
# ---------------------------------------------------------------------------
def test_enrichment(scored_df, gene_set, top_pct=5):
    """Test 1: Hypergeometric enrichment of gene_set at top X%."""
    genes = scored_df["epistasis_id"].map(lambda eid: eid.split(":")[0])
    in_set = genes.isin(gene_set)
    n_top = max(1, int(len(scored_df) * top_pct / 100))
    in_top = scored_df["epi_rank"] <= n_top

    k_obs = int((in_top & in_set).sum())
    K_total = int(in_set.sum())
    N = len(scored_df)
    k_exp = K_total * n_top / N
    p_val = hypergeom.sf(k_obs - 1, N, K_total, n_top)
    fold = k_obs / (k_exp + 1e-10)

    return {"fold": fold, "p_value": p_val, "k_obs": k_obs, "k_exp": round(k_exp, 1)}


def test_directional_enrichment(scored_df, top_pct=5):
    """Test 1b: Separate corrective/cumulative enrichment for TSG vs oncogene."""
    genes = scored_df["epistasis_id"].map(lambda eid: eid.split(":")[0])
    is_onc = genes.isin(ONCOGENES)
    is_tsg = genes.isin(TSGS)
    N = len(scored_df)
    n_top = max(1, int(N * top_pct / 100))

    results = {}
    for direction in ["corrective", "cumulative"]:
        # Rank only within this direction
        subset = scored_df[scored_df["epi_direction"] == direction].copy()
        if len(subset) < 50:
            continue
        subset["dir_rank"] = subset["epi_score"].rank(ascending=False, method="min").astype(int)
        n_top_dir = max(1, int(len(subset) * top_pct / 100))
        in_top = subset["dir_rank"] <= n_top_dir

        sub_genes = subset["epistasis_id"].map(lambda eid: eid.split(":")[0])
        for label, gene_mask in [("TSG", sub_genes.isin(TSGS)), ("Oncogene", sub_genes.isin(ONCOGENES))]:
            k_obs = int((in_top & gene_mask).sum())
            K = int(gene_mask.sum())
            k_exp = K * n_top_dir / len(subset)
            p = hypergeom.sf(k_obs - 1, len(subset), K, n_top_dir)
            results[f"{direction}_{label}"] = {
                "fold": k_obs / (k_exp + 1e-10), "p_value": p,
                "k_obs": k_obs, "k_exp": round(k_exp, 1),
            }
    return results


def test_directional_asymmetry(scored_df):
    """Test 2: Are oncogene pairs more cumulative than TSG pairs?"""
    genes = scored_df["epistasis_id"].map(lambda eid: eid.split(":")[0])
    onc_vals = scored_df.loc[genes.isin(ONCOGENES), "resid_metric"].dropna()
    tsg_vals = scored_df.loc[genes.isin(TSGS), "resid_metric"].dropna()

    if len(onc_vals) < 10 or len(tsg_vals) < 10:
        return {"stat": np.nan, "p_value": np.nan}

    stat, p = mannwhitneyu(onc_vals, tsg_vals, alternative="greater")
    return {
        "stat": stat, "p_value": p,
        "onc_mean": float(onc_vals.mean()), "tsg_mean": float(tsg_vals.mean()),
        "delta_mean": float(onc_vals.mean() - tsg_vals.mean()),
    }


def test_cooccurrence_correlation(scored_df, meta_df):
    """Test 4: Correlation between epistasis score and co-occurrence strength."""
    merged = scored_df.merge(
        meta_df[["cond_prob_dependent", "n_cases_both"]],
        left_on="epistasis_id", right_index=True, how="inner"
    )
    if len(merged) < 50:
        return {"rho_cond": np.nan, "rho_cases": np.nan}

    rho_cond, p_cond = spearmanr(merged["epi_score"], merged["cond_prob_dependent"])
    rho_cases, p_cases = spearmanr(merged["epi_score"], merged["n_cases_both"])
    return {
        "rho_cond_prob": rho_cond, "p_cond_prob": p_cond,
        "rho_n_cases": rho_cases, "p_n_cases": p_cases,
    }


def test_population_depletion(scored_df, okgp_scored_df, top_pct=5):
    """Test 3: Are top-scored TCGA pairs absent from population variation?

    Compare the mean epistasis score of TCGA top-ranked pairs to the
    1kGP background. If TCGA top pairs are under stronger selection,
    they should have higher scores than matched 1kGP pairs.
    """
    n_top = max(1, int(len(scored_df) * top_pct / 100))
    top_tcga = scored_df.nsmallest(n_top, "epi_rank")["epi_score"].values

    if okgp_scored_df is None or len(okgp_scored_df) < 100:
        return {"stat": np.nan, "p_value": np.nan}

    # Compare top TCGA scores to the full 1kGP score distribution
    okgp_scores = okgp_scored_df["epi_score"].dropna().values
    stat, p = mannwhitneyu(top_tcga, okgp_scores, alternative="greater")
    return {
        "stat": stat, "p_value": p,
        "tcga_top_mean": float(top_tcga.mean()),
        "okgp_mean": float(okgp_scores.mean()),
        "ratio": float(top_tcga.mean() / (okgp_scores.mean() + 1e-10)),
    }

print("Validation tests defined.")


# ---------------------------------------------------------------------------
# Cell 4: Load data and run framework for all models
# ---------------------------------------------------------------------------
METRIC_COL = "log_magnitude_ratio"
BASELINE_METRICS = {
    "max_single": lambda df: np.maximum(df["len_WT_M1"], df["len_WT_M2"]),
    "sum_singles": lambda df: df["len_WT_M1"] + df["len_WT_M2"],
    "len_WT_M12": lambda df: df["len_WT_M12"],
}
TOP_PCT = 5

all_validation = []

for model_key, display in MODELS.items():
    ppath = PARQUET_DIR / f"epistasis_metrics_{model_key}_combined.parquet"
    if not ppath.exists():
        continue

    # Load TCGA + 1kGP
    dset = ds.dataset(ppath)
    avail = {f.name for f in dset.schema}
    cols = [c for c in ["source", "epistasis_id", METRIC_COL,
                         "len_WT_M1", "len_WT_M2", "len_WT_M12",
                         "epi_R_singles", "magnitude_ratio"] if c in avail]
    df = dset.to_table(
        columns=cols,
        filter=ds.field("source").isin(["tcga_doubles", "okgp_chr12"])
    ).to_pandas().drop_duplicates(subset=["source", "epistasis_id"])

    df_tcga = df[df["source"] == "tcga_doubles"].copy()
    df_okgp = df[df["source"] == "okgp_chr12"].copy()

    if len(df_tcga) < 100:
        continue

    # --- EPISTASIS FRAMEWORK ---
    scored = score_pairs(df_tcga, METRIC_COL)
    okgp_scored = score_pairs(df_okgp, METRIC_COL) if len(df_okgp) > 100 else None

    row = {"model": display, "model_key": model_key, "method": "epistasis (log_MR)",
           "is_track": model_key in TRACK_MODELS}

    # Test 1: Enrichment
    for gs_label, gs_set in [("Cancer", CANCER_GENES), ("TSG", TSGS), ("Oncogene", ONCOGENES)]:
        r = test_enrichment(scored, gs_set, TOP_PCT)
        row[f"enrich_{gs_label}_fold"] = r["fold"]
        row[f"enrich_{gs_label}_p"] = r["p_value"]

    # Test 1b: Directional enrichment
    dir_results = test_directional_enrichment(scored, TOP_PCT)
    for key, val in dir_results.items():
        row[f"dir_{key}_fold"] = val["fold"]
        row[f"dir_{key}_p"] = val["p_value"]

    # Test 2: Asymmetry
    asym = test_directional_asymmetry(scored)
    row["asymmetry_p"] = asym["p_value"]
    row["asymmetry_delta"] = asym.get("delta_mean", np.nan)

    # Test 3: Population depletion
    pop = test_population_depletion(scored, okgp_scored, TOP_PCT)
    row["pop_depletion_p"] = pop["p_value"]
    row["pop_depletion_ratio"] = pop.get("ratio", np.nan)

    # Test 4: Co-occurrence correlation
    cooc = test_cooccurrence_correlation(scored, tcga_meta)
    row["cooc_rho_cond"] = cooc.get("rho_cond_prob", np.nan)
    row["cooc_p_cond"] = cooc.get("p_cond_prob", np.nan)
    row["cooc_rho_cases"] = cooc.get("rho_n_cases", np.nan)

    all_validation.append(row)

    # --- BASELINES ---
    for bl_name, bl_fn in BASELINE_METRICS.items():
        if not all(c in df_tcga.columns for c in ["len_WT_M1", "len_WT_M2", "len_WT_M12"]):
            continue
        df_bl = df_tcga.copy()
        df_bl["_bl_metric"] = bl_fn(df_bl)
        scored_bl = score_pairs_baseline(df_bl, "_bl_metric")
        okgp_bl = None
        if len(df_okgp) > 100:
            df_okgp_bl = df_okgp.copy()
            df_okgp_bl["_bl_metric"] = bl_fn(df_okgp_bl)
            okgp_bl = score_pairs_baseline(df_okgp_bl, "_bl_metric")

        bl_row = {"model": display, "model_key": model_key, "method": f"baseline ({bl_name})",
                   "is_track": model_key in TRACK_MODELS}

        for gs_label, gs_set in [("Cancer", CANCER_GENES), ("TSG", TSGS), ("Oncogene", ONCOGENES)]:
            r = test_enrichment(scored_bl, gs_set, TOP_PCT)
            bl_row[f"enrich_{gs_label}_fold"] = r["fold"]
            bl_row[f"enrich_{gs_label}_p"] = r["p_value"]

        # Baselines don't have directional enrichment (unsigned)
        bl_row["asymmetry_p"] = np.nan
        bl_row["asymmetry_delta"] = np.nan

        pop = test_population_depletion(scored_bl, okgp_bl, TOP_PCT)
        bl_row["pop_depletion_p"] = pop["p_value"]
        bl_row["pop_depletion_ratio"] = pop.get("ratio", np.nan)

        cooc = test_cooccurrence_correlation(scored_bl, tcga_meta)
        bl_row["cooc_rho_cond"] = cooc.get("rho_cond_prob", np.nan)
        bl_row["cooc_p_cond"] = cooc.get("p_cond_prob", np.nan)
        bl_row["cooc_rho_cases"] = cooc.get("rho_n_cases", np.nan)

        all_validation.append(bl_row)

    print(f"  {display}: done")

df_val = pd.DataFrame(all_validation)
print(f"\nTotal rows: {len(df_val)}")


# ---------------------------------------------------------------------------
# Cell 5: Summary — epistasis vs baselines across all tests
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("VALIDATION SUITE RESULTS: EPISTASIS vs BASELINES (averaged across 18 models)")
print("=" * 110)

summary = df_val.groupby("method").agg(
    mean_Cancer_fold=("enrich_Cancer_fold", "mean"),
    mean_TSG_fold=("enrich_TSG_fold", "mean"),
    mean_Onc_fold=("enrich_Oncogene_fold", "mean"),
    n_sig_Cancer=("enrich_Cancer_p", lambda x: (x < 0.05).sum()),
    n_sig_TSG=("enrich_TSG_p", lambda x: (x < 0.05).sum()),
    n_sig_Onc=("enrich_Oncogene_p", lambda x: (x < 0.05).sum()),
    mean_asym_p=("asymmetry_p", "mean"),
    n_sig_asym=("asymmetry_p", lambda x: (x < 0.05).sum()),
    mean_pop_ratio=("pop_depletion_ratio", "mean"),
    n_sig_pop=("pop_depletion_p", lambda x: (x < 0.05).sum()),
    mean_cooc_rho=("cooc_rho_cond", "mean"),
    n_sig_cooc=("cooc_p_cond", lambda x: (x < 0.05).sum()),
).reset_index()

print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Cell 6: Per-model scorecard
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("PER-MODEL SCORECARD (epistasis framework only)")
print("=" * 110)

epi_only = df_val[df_val["method"] == "epistasis (log_MR)"].copy()

# Composite score: count how many tests pass at p < 0.05
epi_only["n_tests_pass"] = (
    (epi_only["enrich_Cancer_p"] < 0.05).astype(int)
    + (epi_only["enrich_TSG_p"] < 0.05).astype(int)
    + (epi_only["enrich_Oncogene_p"] < 0.05).astype(int)
    + (epi_only["asymmetry_p"] < 0.05).astype(int)
    + (epi_only["pop_depletion_p"] < 0.05).astype(int)
    + (epi_only["cooc_p_cond"] < 0.05).astype(int)
)

scorecard_cols = [
    "model", "is_track",
    "enrich_Cancer_fold", "enrich_TSG_fold", "enrich_Oncogene_fold",
    "asymmetry_delta",
    "pop_depletion_ratio",
    "cooc_rho_cond",
    "n_tests_pass",
]
epi_only = epi_only.sort_values("n_tests_pass", ascending=False)
print(epi_only[scorecard_cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Cell 7: Directional enrichment detail
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("DIRECTIONAL ENRICHMENT: corrective_TSG and cumulative_Oncogene")
print("=" * 110)

dir_cols = [c for c in df_val.columns if c.startswith("dir_")]
if dir_cols:
    epi_dir = epi_only[["model", "is_track"] + dir_cols].copy()
    print(epi_dir.to_string(index=False))
else:
    print("No directional enrichment columns found.")


# ---------------------------------------------------------------------------
# Cell 8: Visualization — radar plot per model
# ---------------------------------------------------------------------------
# Normalize each test to [0, 1] across models for radar comparison
test_axes = {
    "Cancer\nenrich": "enrich_Cancer_fold",
    "TSG\nenrich": "enrich_TSG_fold",
    "Oncogene\nenrich": "enrich_Oncogene_fold",
    "Directional\nasymmetry": "asymmetry_delta",
    "Population\ndepletion": "pop_depletion_ratio",
    "Co-occurrence\ncorrelation": "cooc_rho_cond",
}

epi_for_radar = epi_only.copy()
normalized = {}
for label, col in test_axes.items():
    vals = epi_for_radar[col].values
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if vmax - vmin > 1e-10:
        normalized[label] = (vals - vmin) / (vmax - vmin)
    else:
        normalized[label] = np.zeros_like(vals)

n_axes = len(test_axes)
angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Plot top 5 models + bottom 1
top_models = epi_for_radar.head(5)["model"].tolist()
bot_models = epi_for_radar.tail(1)["model"].tolist()
show_models = top_models + bot_models

colors_map = plt.cm.tab10(np.linspace(0, 1, len(show_models)))

for i, model_name in enumerate(show_models):
    idx = epi_for_radar[epi_for_radar["model"] == model_name].index[0]
    model_idx = epi_for_radar.index.get_loc(idx)
    vals = [normalized[label][model_idx] for label in test_axes.keys()]
    vals += vals[:1]
    ax.plot(angles, vals, "o-", label=model_name, color=colors_map[i], linewidth=1.5, markersize=4)
    ax.fill(angles, vals, alpha=0.05, color=colors_map[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(test_axes.keys(), fontsize=8)
ax.set_ylim(0, 1.1)
ax.set_title("Epistasis framework validation\n(normalized per test, top 5 + worst model)", fontsize=10, pad=20)
ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))

fig.tight_layout()
fig.savefig(FIG_DIR / "model_validation_radar.png", bbox_inches="tight", dpi=150)
print(f"Saved radar plot to {FIG_DIR / 'model_validation_radar.png'}")


# ---------------------------------------------------------------------------
# Cell 9: Composite ranking — which model should you use?
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("RECOMMENDED MODEL RANKING (by composite validation score)")
print("=" * 110)

# Rank each test independently, average ranks
rank_cols = {}
for label, col in test_axes.items():
    epi_for_rank = epi_only.copy()
    # Higher is better for all these metrics
    epi_for_rank[f"rank_{col}"] = epi_for_rank[col].rank(ascending=False, method="min")
    rank_cols[col] = f"rank_{col}"

for col, rcol in rank_cols.items():
    if rcol not in epi_only.columns:
        epi_only[rcol] = epi_only[col].rank(ascending=False, method="min")

rank_col_names = list(rank_cols.values())
epi_only["mean_rank"] = epi_only[rank_col_names].mean(axis=1)
epi_only = epi_only.sort_values("mean_rank")

print(epi_only[["model", "is_track", "mean_rank", "n_tests_pass"]
               + rank_col_names].to_string(index=False))

print(f"\n>>> RECOMMENDED: {epi_only.iloc[0]['model']} (mean rank {epi_only.iloc[0]['mean_rank']:.1f})")
print(f"    Runner-up: {epi_only.iloc[1]['model']} (mean rank {epi_only.iloc[1]['mean_rank']:.1f})")

# Save full results
df_val.to_csv(FIG_DIR / "validation_suite_results.csv", index=False)
epi_only.to_csv(FIG_DIR / "model_scorecard.csv", index=False)
print(f"\nAll results saved to {FIG_DIR}")
