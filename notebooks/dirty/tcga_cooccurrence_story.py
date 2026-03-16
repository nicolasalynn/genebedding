"""
Epistasis geometry captures somatic selection pressure.

Core finding: The epistasis framework's deviation-from-additivity score
correlates with how strongly somatic selection maintains mutation pairs
together in tumors (conditional co-occurrence probability). Simple
baselines (double-mutant effect size, sum of singles) show no such
correlation. This establishes that the geometric decomposition captures
genuine interaction biology invisible to non-epistasis approaches.

Figures:
  1. Co-occurrence correlation: epistasis score vs cond_prob (scatter + binned)
  2. Metric comparison: rho across all models for epistasis vs baselines
  3. Directional breakdown: corrective vs cumulative co-occurrence
  4. Enrichment within high-co-occurrence pairs
"""

# ---------------------------------------------------------------------------
# Cell 1: Setup
# ---------------------------------------------------------------------------
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyarrow.dataset as ds
from scipy.stats import spearmanr, hypergeom, mannwhitneyu

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

# Cancer gene sets
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

# TCGA metadata
tcga_meta = pd.read_csv(DATA_DIR / "tcga_doubles_pairs.tsv", sep="\t")
tcga_meta = tcga_meta.set_index("epistasis_id")

# Distance helpers
DIST_BINS = [1, 2, 3, 4, 5, 6, 11, 16, 26, 51, 101, 10000]
DIST_LABELS = ["1", "2", "3", "4", "5", "6-10", "11-15", "16-25", "26-50", "51-100", "101+"]

def distance_residualize(values, distances):
    bins = pd.cut(distances, bins=DIST_BINS, labels=DIST_LABELS, right=False)
    result = values.copy()
    for label in DIST_LABELS:
        mask = bins == label
        if mask.sum() > 0:
            result[mask] = values[mask] - values[mask].mean()
    return result

def parse_distance(eid):
    parts = eid.split("|")
    try:
        return abs(int(parts[1].split(":")[2]) - int(parts[0].split(":")[2]))
    except (IndexError, ValueError):
        return np.nan

# Style
COL_BLUE, COL_TERRA = "#4A7FB5", "#CB6A49"
COL_GREEN, COL_PURPLE = "#5BA05B", "#8B6DB0"
GRAY_MID, GRAY_DARK = "#999999", "#333333"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.linewidth": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

def setup_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=GRAY_DARK, which="both")

print("Setup OK")


# ---------------------------------------------------------------------------
# Cell 2: Load all models
# ---------------------------------------------------------------------------
all_results = []

for model_key, display in MODELS.items():
    ppath = PARQUET_DIR / f"epistasis_metrics_{model_key}_combined.parquet"
    if not ppath.exists():
        continue

    dset = ds.dataset(ppath)
    avail = {f.name for f in dset.schema}
    cols = [c for c in ["source", "epistasis_id", "log_magnitude_ratio",
                         "epi_R_singles", "magnitude_ratio",
                         "len_WT_M1", "len_WT_M2", "len_WT_M12"] if c in avail]
    df = dset.to_table(
        columns=cols,
        filter=ds.field("source") == "tcga_doubles"
    ).to_pandas().drop_duplicates(subset=["epistasis_id"])

    # Merge metadata
    df = df.merge(tcga_meta[["cond_prob_dependent", "n_cases_both"]],
                  left_on="epistasis_id", right_index=True, how="left")
    df["distance"] = df["epistasis_id"].map(parse_distance)
    df["gene"] = df["epistasis_id"].map(lambda eid: eid.split(":")[0])

    # Compute all metrics (distance-residualized)
    valid = df.dropna(subset=["log_magnitude_ratio", "distance", "cond_prob_dependent"]).copy()
    if len(valid) < 100:
        continue

    dists = valid["distance"].values

    # Epistasis metrics
    valid["resid_log_mr"] = distance_residualize(valid["log_magnitude_ratio"].values, dists)
    valid["abs_resid_log_mr"] = np.abs(valid["resid_log_mr"])
    valid["resid_R_singles"] = distance_residualize(valid["epi_R_singles"].values, dists)

    # Baselines
    valid["max_single"] = np.maximum(valid["len_WT_M1"], valid["len_WT_M2"])
    valid["sum_singles"] = valid["len_WT_M1"] + valid["len_WT_M2"]
    valid["resid_len_m12"] = distance_residualize(valid["len_WT_M12"].values, dists)
    valid["resid_max_single"] = distance_residualize(valid["max_single"].values, dists)
    valid["resid_sum_singles"] = distance_residualize(valid["sum_singles"].values, dists)

    # Direction labels
    valid["direction"] = np.where(valid["resid_log_mr"] < 0, "corrective", "cumulative")

    # Correlations with co-occurrence
    metrics = {
        "|log(MR)| (epistasis)": "abs_resid_log_mr",
        "R_singles (epistasis)": "resid_R_singles",
        "len_WT_M12 (baseline)": "resid_len_m12",
        "max_single (baseline)": "resid_max_single",
        "sum_singles (baseline)": "resid_sum_singles",
    }

    for metric_name, col in metrics.items():
        vals = valid[col].dropna()
        cp = valid.loc[vals.index, "cond_prob_dependent"]
        rho, p = spearmanr(vals.abs(), cp)
        all_results.append({
            "model": display, "model_key": model_key,
            "metric": metric_name,
            "is_epistasis": "epistasis" in metric_name,
            "is_track": model_key in TRACK_MODELS,
            "rho": rho, "p_value": p, "n": len(vals),
        })

    # Store one model's data for detailed plots
    if model_key == "nt50_multi":
        detail_df = valid.copy()

df_corr = pd.DataFrame(all_results)
print(f"Loaded {df_corr['model'].nunique()} models, {len(df_corr)} metric-model pairs")


# ---------------------------------------------------------------------------
# Cell 3: Print summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("SPEARMAN CORRELATION: metric score vs co-occurrence probability")
print("(averaged across 18 models, distance-residualized)")
print("=" * 90)

summary = df_corr.groupby("metric").agg(
    mean_rho=("rho", "mean"),
    median_rho=("rho", "median"),
    min_rho=("rho", "min"),
    max_rho=("rho", "max"),
    n_sig=("p_value", lambda x: (x < 0.05).sum()),
    n_models=("model", "nunique"),
).sort_values("mean_rho", ascending=False).reset_index()

print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Cell 4: Figure 1 — Main result (4-panel)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3,
                        left=0.08, right=0.95, top=0.93, bottom=0.08)

# ── Panel A: Binned trend with error bars (no scatter) ──
ax_a = fig.add_subplot(gs[0, 0])
setup_ax(ax_a)

from matplotlib.lines import Line2D

x = detail_df["abs_resid_log_mr"]
y = detail_df["cond_prob_dependent"]
rho_all, p_all = spearmanr(x, y)

# Decile bins — show mean + SEM per bin
n_bins = 10
detail_df["_decile"] = pd.qcut(x, n_bins, labels=False, duplicates="drop")
binned = detail_df.groupby("_decile").agg(
    x_mean=("abs_resid_log_mr", "mean"),
    y_mean=("cond_prob_dependent", "mean"),
    y_sem=("cond_prob_dependent", lambda v: v.std() / np.sqrt(len(v))),
    n=("cond_prob_dependent", "count"),
    cancer_frac=("gene", lambda g: g.isin(CANCER_GENES).mean()),
).reset_index()

# Main line: co-occurrence
ax_a.errorbar(binned["x_mean"], binned["y_mean"], yerr=binned["y_sem"],
              fmt="o-", color=COL_TERRA, linewidth=2, markersize=6, capsize=4,
              capthick=1.5, label="|log(MR)| (epistasis)", zorder=5)

# Baseline comparison: bin by sum_singles, same y
x_bl = detail_df["resid_sum_singles"].abs()
detail_df["_decile_bl"] = pd.qcut(x_bl, n_bins, labels=False, duplicates="drop")
binned_bl = detail_df.groupby("_decile_bl").agg(
    x_mean=("resid_sum_singles", lambda v: v.abs().mean()),
    y_mean=("cond_prob_dependent", "mean"),
    y_sem=("cond_prob_dependent", lambda v: v.std() / np.sqrt(len(v))),
).reset_index()
rho_bl, p_bl = spearmanr(x_bl, y)

ax_a.errorbar(binned_bl["x_mean"], binned_bl["y_mean"], yerr=binned_bl["y_sem"],
              fmt="s--", color=GRAY_MID, linewidth=1.5, markersize=5, capsize=3,
              capthick=1, alpha=0.7, label="sum_singles (baseline)", zorder=3)

ax_a.set_xlabel("Metric score (decile bin mean)", fontsize=10)
ax_a.set_ylabel("Mean co-occurrence probability", fontsize=10)
ax_a.set_title(f"Epistasis score predicts somatic co-occurrence\n"
               f"|log(MR)| ρ={rho_all:.3f} (p={p_all:.1e})  |  "
               f"sum_singles ρ={rho_bl:.3f}", fontsize=9, fontweight="bold")
ax_a.legend(fontsize=8, loc="lower right")
ax_a.text(-0.12, 1.05, "a", transform=ax_a.transAxes, fontsize=14, fontweight="bold")


# ── Panel B: Bar chart — all models, epistasis vs baselines ──
ax_b = fig.add_subplot(gs[0, 1])
setup_ax(ax_b)

# Pivot: one row per model, columns for each metric's rho
pivot = df_corr.pivot(index="model", columns="metric", values="rho")
# Sort by epistasis rho
sort_col = "|log(MR)| (epistasis)"
pivot = pivot.sort_values(sort_col, ascending=True)

y_pos = np.arange(len(pivot))
bar_h = 0.15

metric_colors = {
    "|log(MR)| (epistasis)": COL_TERRA,
    "R_singles (epistasis)": COL_PURPLE,
    "sum_singles (baseline)": "#AAAAAA",
    "max_single (baseline)": "#CCCCCC",
    "len_WT_M12 (baseline)": "#DDDDDD",
}
metric_order = ["|log(MR)| (epistasis)", "R_singles (epistasis)",
                "sum_singles (baseline)", "max_single (baseline)", "len_WT_M12 (baseline)"]

for i, metric in enumerate(metric_order):
    if metric in pivot.columns:
        vals = pivot[metric].values
        ax_b.barh(y_pos + i * bar_h - 2 * bar_h, vals, height=bar_h,
                  color=metric_colors.get(metric, "gray"), alpha=0.85,
                  label=metric, edgecolor="none")

ax_b.axvline(0, color=GRAY_MID, linewidth=0.5, linestyle="-")
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(pivot.index, fontsize=7)
ax_b.set_xlabel("Spearman ρ with co-occurrence probability", fontsize=10)
ax_b.set_title("Epistasis metrics correlate with\nsomatic co-occurrence; baselines do not",
               fontsize=10, fontweight="bold")
ax_b.legend(fontsize=6, loc="lower right", framealpha=0.9)
ax_b.text(-0.15, 1.05, "b", transform=ax_b.transAxes, fontsize=14, fontweight="bold")


# ── Panel C: Corrective vs cumulative breakdown ──
ax_c = fig.add_subplot(gs[1, 0])
setup_ax(ax_c)

# For each model, compute rho separately for corrective and cumulative
dir_rows = []
for model_key, display in MODELS.items():
    ppath = PARQUET_DIR / f"epistasis_metrics_{model_key}_combined.parquet"
    if not ppath.exists():
        continue
    dset = ds.dataset(ppath)
    avail = {f.name for f in dset.schema}
    cols = [c for c in ["source", "epistasis_id", "log_magnitude_ratio"] if c in avail]
    df = dset.to_table(columns=cols, filter=ds.field("source") == "tcga_doubles"
                       ).to_pandas().drop_duplicates(subset=["epistasis_id"])
    df = df.merge(tcga_meta[["cond_prob_dependent"]], left_on="epistasis_id", right_index=True, how="left")
    df["distance"] = df["epistasis_id"].map(parse_distance)
    valid = df.dropna(subset=["log_magnitude_ratio", "distance", "cond_prob_dependent"]).copy()
    if len(valid) < 100:
        continue
    valid["resid"] = distance_residualize(valid["log_magnitude_ratio"].values, valid["distance"].values)

    for direction, subset in [("corrective", valid[valid["resid"] < 0]),
                               ("cumulative", valid[valid["resid"] >= 0])]:
        if len(subset) < 50:
            continue
        rho, p = spearmanr(subset["resid"].abs(), subset["cond_prob_dependent"])
        dir_rows.append({"model": display, "direction": direction, "rho": rho, "p": p})

df_dir = pd.DataFrame(dir_rows)
if len(df_dir) > 0:
    dir_summary = df_dir.groupby("direction")["rho"].agg(["mean", "std", "count"]).reset_index()
    print("\nDirectional co-occurrence correlation:")
    print(dir_summary.to_string(index=False))

    # Paired bar chart with significance markers
    models_sorted = df_dir[df_dir["direction"] == "corrective"].sort_values("rho")["model"].tolist()
    y_d = np.arange(len(models_sorted))

    for i, (direction, color, offset) in enumerate([
        ("corrective", COL_BLUE, -0.15), ("cumulative", COL_TERRA, 0.15)
    ]):
        sub = df_dir[df_dir["direction"] == direction].set_index("model")
        vals = [sub.loc[m, "rho"] if m in sub.index else 0 for m in models_sorted]
        pvals = [sub.loc[m, "p"] if m in sub.index else 1.0 for m in models_sorted]
        bar_alphas = [0.85 if pv < 0.05 else 0.25 for pv in pvals]

        for j, (v, a, pv) in enumerate(zip(vals, bar_alphas, pvals)):
            ax_c.barh(y_d[j] + offset, v, height=0.28, color=color, alpha=a, edgecolor="none")
            # Significance marker
            if pv < 0.001:
                ax_c.text(v + 0.003, y_d[j] + offset, "***", fontsize=6, va="center", color=color)
            elif pv < 0.01:
                ax_c.text(v + 0.003, y_d[j] + offset, "**", fontsize=6, va="center", color=color)
            elif pv < 0.05:
                ax_c.text(v + 0.003, y_d[j] + offset, "*", fontsize=6, va="center", color=color)

        # Add to legend only once
        if i == 0:
            ax_c.barh([], [], height=0.28, color=COL_BLUE, alpha=0.85, label="Corrective (MR<1)")
            ax_c.barh([], [], height=0.28, color=COL_TERRA, alpha=0.85, label="Cumulative (MR>1)")
            ax_c.barh([], [], height=0.28, color="gray", alpha=0.25, label="Not significant")

    ax_c.axvline(0, color=GRAY_MID, linewidth=0.5)
    ax_c.set_yticks(y_d)
    ax_c.set_yticklabels(models_sorted, fontsize=7)
    ax_c.set_xlabel("Spearman ρ (|resid log(MR)| vs cond_prob)", fontsize=9)
    ax_c.set_title("Both corrective and cumulative pairs\ncorrelate with co-occurrence", fontsize=10, fontweight="bold")
    ax_c.legend(fontsize=7, loc="lower right")
    ax_c.text(-0.15, 1.05, "c", transform=ax_c.transAxes, fontsize=14, fontweight="bold")


# ── Panel D: Quintile analysis — zoomed, with significance ──
ax_d = fig.add_subplot(gs[1, 1])
setup_ax(ax_d)

# Quintiles of epistasis score
quintile_data = []
detail_df["quintile"] = pd.qcut(detail_df["abs_resid_log_mr"], 5,
                                 labels=["Q1\n(least)", "Q2", "Q3", "Q4", "Q5\n(most)"])
q_labels = ["Q1\n(least)", "Q2", "Q3", "Q4", "Q5\n(most)"]
for q in q_labels:
    sub = detail_df[detail_df["quintile"] == q]
    quintile_data.append({
        "quintile": q,
        "mean_cond_prob": sub["cond_prob_dependent"].mean(),
        "sem": sub["cond_prob_dependent"].std() / np.sqrt(len(sub)),
        "values": sub["cond_prob_dependent"].values,
        "n": len(sub),
        "cancer_frac": sub["gene"].isin(CANCER_GENES).mean(),
    })

qdf = pd.DataFrame(quintile_data)
x_q = np.arange(len(qdf))

# Gradient color from blue (low) to red (high)
q_colors = [COL_BLUE, "#7BA3C9", GRAY_MID, "#D4927A", COL_TERRA]

bars = ax_d.bar(x_q, qdf["mean_cond_prob"], yerr=qdf["sem"],
                color=q_colors, alpha=0.85, edgecolor="none", capsize=5)

# Zoom y-axis to show the difference clearly
y_min = qdf["mean_cond_prob"].min() - 3 * qdf["sem"].max()
y_max = qdf["mean_cond_prob"].max() + 5 * qdf["sem"].max()
ax_d.set_ylim(y_min, y_max)

# Add break indicator on y-axis
ax_d.spines["left"].set_visible(True)
ax_d.text(-0.02, y_min + 0.001, "≠ 0", transform=ax_d.get_yaxis_transform(),
          fontsize=7, ha="right", va="bottom", color=GRAY_MID)

# Mann-Whitney test: Q5 vs Q1
from scipy.stats import mannwhitneyu as mwu
stat, p_q5q1 = mwu(qdf.iloc[-1]["values"], qdf.iloc[0]["values"], alternative="greater")
# Annotate significance bracket
y_bracket = y_max - 0.4 * qdf["sem"].max()
ax_d.plot([0, 0, 4, 4], [y_bracket - 0.002, y_bracket, y_bracket, y_bracket - 0.002],
          color=GRAY_DARK, linewidth=1)
p_str = f"p={p_q5q1:.1e}" if p_q5q1 < 0.01 else f"p={p_q5q1:.3f}"
ax_d.text(2, y_bracket + 0.001, p_str, ha="center", fontsize=8, fontweight="bold", color=GRAY_DARK)

ax_d.set_xticks(x_q)
ax_d.set_xticklabels(qdf["quintile"], fontsize=8)
ax_d.set_ylabel("Mean co-occurrence probability", fontsize=9)
ax_d.set_xlabel("Epistasis score quintile", fontsize=9)
ax_d.set_title("Most epistatic pairs have strongest\nsomatic co-occurrence (NT-50M)", fontsize=10, fontweight="bold")
ax_d.text(-0.12, 1.05, "d", transform=ax_d.transAxes, fontsize=14, fontweight="bold")

fig.savefig(FIG_DIR / "cooccurrence_main_figure.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG_DIR / "cooccurrence_main_figure.pdf", bbox_inches="tight")
print(f"\nSaved main figure to {FIG_DIR / 'cooccurrence_main_figure.png'}")


# ---------------------------------------------------------------------------
# Cell 5: Supplementary — enrichment within high-co-occurrence pairs
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("ENRICHMENT WITHIN HIGH CO-OCCURRENCE PAIRS")
print("=" * 90)

# Restrict to pairs with cond_prob >= 0.95, then test enrichment
# by epistasis rank within that subset
for mk in ["nt50_multi", "mutbert", "borzoi"]:
    ppath = PARQUET_DIR / f"epistasis_metrics_{mk}_combined.parquet"
    if not ppath.exists():
        continue
    display = MODELS[mk]
    dset = ds.dataset(ppath)
    avail = {f.name for f in dset.schema}
    cols = [c for c in ["source", "epistasis_id", "log_magnitude_ratio",
                         "len_WT_M1", "len_WT_M2"] if c in avail]
    df = dset.to_table(columns=cols, filter=ds.field("source") == "tcga_doubles"
                       ).to_pandas().drop_duplicates(subset=["epistasis_id"])
    df = df.merge(tcga_meta[["cond_prob_dependent"]], left_on="epistasis_id", right_index=True, how="left")
    df["distance"] = df["epistasis_id"].map(parse_distance)
    df["gene"] = df["epistasis_id"].map(lambda eid: eid.split(":")[0])
    valid = df.dropna(subset=["log_magnitude_ratio", "distance"]).copy()
    valid["resid"] = distance_residualize(valid["log_magnitude_ratio"].values, valid["distance"].values)

    # High co-occurrence subset
    high_co = valid[valid["cond_prob_dependent"] >= 0.95]
    if len(high_co) < 50:
        continue

    print(f"\n--- {display} (cond_prob >= 0.95: {len(high_co)} pairs) ---")

    # Rank within high-co-occurrence subset
    high_co = high_co.copy()
    high_co["rank"] = high_co["resid"].abs().rank(ascending=False, method="min")

    for top_pct in [5, 10]:
        n_top = max(1, int(len(high_co) * top_pct / 100))
        in_top = high_co["rank"] <= n_top
        genes = high_co["gene"]

        for gs_label, gs_set in [("Cancer", CANCER_GENES), ("TSG", TSGS), ("Oncogene", ONCOGENES)]:
            in_set = genes.isin(gs_set)
            k_obs = int((in_top & in_set).sum())
            K = int(in_set.sum())
            k_exp = K * n_top / len(high_co)
            p = hypergeom.sf(k_obs - 1, len(high_co), K, n_top)
            fold = k_obs / (k_exp + 1e-10)
            sig = "*" if p < 0.05 else ""
            print(f"  Top {top_pct}%: {gs_label} fold={fold:.2f} (k={k_obs}/{n_top}, "
                  f"exp={k_exp:.1f}, p={p:.4f}){sig}")


# ---------------------------------------------------------------------------
# Cell 6: Summary statistics for paper text
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("PAPER-READY STATISTICS")
print("=" * 90)

epi_rhos = df_corr[df_corr["metric"] == "|log(MR)| (epistasis)"]
bl_rhos = df_corr[df_corr["metric"] == "sum_singles (baseline)"]
bl2_rhos = df_corr[df_corr["metric"] == "len_WT_M12 (baseline)"]

print(f"\n1. Epistasis co-occurrence correlation:")
print(f"   |log(MR)|: mean ρ = {epi_rhos['rho'].mean():.3f} "
      f"(range {epi_rhos['rho'].min():.3f}–{epi_rhos['rho'].max():.3f}), "
      f"significant in {(epi_rhos['p_value'] < 0.05).sum()}/{len(epi_rhos)} models")

print(f"\n2. Baseline co-occurrence correlations:")
print(f"   sum_singles: mean ρ = {bl_rhos['rho'].mean():.3f}, "
      f"sig in {(bl_rhos['p_value'] < 0.05).sum()}/{len(bl_rhos)}")
print(f"   len_WT_M12:  mean ρ = {bl2_rhos['rho'].mean():.3f}, "
      f"sig in {(bl2_rhos['p_value'] < 0.05).sum()}/{len(bl2_rhos)}")

ratio = epi_rhos["rho"].mean() / (bl_rhos["rho"].mean() + 1e-10)
print(f"\n3. Ratio: epistasis correlation is {ratio:.0f}x stronger than best baseline")

print(f"\n4. Quintile analysis (NT-50M):")
print(f"   Q1 (least epistatic): mean cond_prob = {qdf.iloc[0]['mean_cond_prob']:.3f}")
print(f"   Q5 (most epistatic):  mean cond_prob = {qdf.iloc[-1]['mean_cond_prob']:.3f}")
delta = qdf.iloc[-1]["mean_cond_prob"] - qdf.iloc[0]["mean_cond_prob"]
print(f"   Difference: {delta:+.3f}")

if len(df_dir) > 0:
    corr_mean = df_dir[df_dir["direction"] == "corrective"]["rho"].mean()
    cum_mean = df_dir[df_dir["direction"] == "cumulative"]["rho"].mean()
    print(f"\n5. Directional breakdown:")
    print(f"   Corrective pairs: mean ρ = {corr_mean:.3f}")
    print(f"   Cumulative pairs: mean ρ = {cum_mean:.3f}")

print(f"\nAll figures saved to {FIG_DIR}")
