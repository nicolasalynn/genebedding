"""
Epistasis subspace decomposition: what KIND of epistasis does each pair have?

For each TCGA double-variant pair, we decompose the epistasis residual vector
into two components:

  ε = ε_expression + ε_orthogonal

Where ε_expression is the projection onto the top-k eigenvectors of the null
(population) covariance — the "expression-dominated" subspace for track models.

The expression fraction = |ε_expression|² / |ε|² tells you how much of the
pair's epistasis lives in the expression subspace vs orthogonal directions
(splicing, structural, etc).

Key question: do oncogenes and TSGs differ in expression fraction?

If so, the directional asymmetry (corrective vs cumulative) has a mechanistic
explanation: different cancer gene classes interact through different biological
pathways, visible as different subspace projections.

Runs on cluster (needs embedding DBs for residual vectors).
"""

# ---------------------------------------------------------------------------
# Cell 1: Setup
# ---------------------------------------------------------------------------
import sys, os, logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path.cwd()
for _ in range(5):
    if (ROOT / "notebooks" / "paper_data_config.py").exists():
        break
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from paper_data_config import embeddings_dir, EPISTASIS_PAPER_ROOT

OUTPUT_BASE = embeddings_dir()
CACHE_DIR = OUTPUT_BASE / "cache"
ANNOT_DIR = EPISTASIS_PAPER_ROOT / "data" / "annotations"
FIG_DIR = EPISTASIS_PAPER_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

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

print(f"Oncogenes: {len(ONCOGENES)}, TSGs: {len(TSGS)}")


# ---------------------------------------------------------------------------
# Cell 2: Load null eigenvectors + compute subspace projections
# ---------------------------------------------------------------------------
from genebeddings import VariantEmbeddingDB
from genebeddings.epistasis_features import _list_epistasis_ids_from_db, _load_residual_from_db

SOURCES = {
    "tcga_doubles": "TCGA (original)",
    "tcga_high_selection": "TCGA high-sel",
    "tcga_low_selection": "TCGA low-sel",
    "okgp_matched_doubles": "1kGP germline",
    "okgp_chr12": "1kGP chr12 (null)",
}

MODELS = {
    "borzoi": "Borzoi",
    "alphagenome": "AlphaGenome",
    "nt500_multi": "NT-500M",
    "mutbert": "MutBERT",
    "dnabert": "DNABERT",
    "ntv3_100m_post": "NTv3-post",
    "evo2": "Evo2",
}

K_DIMS = 50  # number of top null eigenvectors to define "expression subspace"

all_decomp = []

for mk, display in MODELS.items():
    # Load null covariance eigenvectors
    cov_path = CACHE_DIR / f"{mk}_cov_inv.npz"
    if not cov_path.exists():
        logger.warning("No null cov for %s, skipping", mk)
        continue

    cov_data = np.load(cov_path)
    null_cov = cov_data["cov"]
    d = null_cov.shape[0]

    # Top-k eigenvectors of null covariance = "expression subspace"
    eigenvalues, eigenvectors = np.linalg.eigh(null_cov)
    V_expression = eigenvectors[:, -K_DIMS:]  # (d, k)

    # Process ALL sources
    for source_key, source_label in SOURCES.items():
        db_path = OUTPUT_BASE / source_key / f"{mk}.db"
        if not db_path.exists():
            continue

        db = VariantEmbeddingDB(str(db_path))
        epi_ids = _list_epistasis_ids_from_db(db)
        logger.info("%s/%s: %d pairs, d=%d", mk, source_key, len(epi_ids), d)

        for eid in epi_ids:
            residual = _load_residual_from_db(db, eid)
            if residual is None:
                continue

            r = residual.astype(np.float64)
            r_norm_sq = np.dot(r, r)
            if r_norm_sq < 1e-30:
                continue

            r_proj = V_expression @ (V_expression.T @ r)
            r_proj_norm_sq = np.dot(r_proj, r_proj)

            expression_fraction = r_proj_norm_sq / r_norm_sq
            r_mag = np.sqrt(r_norm_sq)

            gene = eid.split(":")[0]
            parts = eid.split("|")
            distance = abs(int(parts[1].split(":")[2]) - int(parts[0].split(":")[2]))

            all_decomp.append({
                "model": mk,
                "display": display,
                "source": source_key,
                "source_label": source_label,
                "epistasis_id": eid,
                "gene": gene,
                "distance": distance,
                "expression_fraction": float(expression_fraction),
                "orthogonal_fraction": 1.0 - float(expression_fraction),
                "residual_magnitude": float(r_mag),
                "expression_magnitude": float(np.sqrt(r_proj_norm_sq)),
                "orthogonal_magnitude": float(np.sqrt(max(0, r_norm_sq - r_proj_norm_sq))),
                "is_oncogene": gene in ONCOGENES,
                "is_tsg": gene in TSGS,
                "gene_class": "Oncogene" if gene in ONCOGENES else ("TSG" if gene in TSGS else "Other"),
            })

        db.close()

df_decomp = pd.DataFrame(all_decomp)
print(f"\nTotal: {len(df_decomp)} model×pair entries")
print(f"Models: {df_decomp['model'].nunique()}")
print(f"Pairs per model: {df_decomp.groupby('model').size().to_dict()}")


# ---------------------------------------------------------------------------
# Cell 3: Compare expression fraction across sources AND gene classes
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print(f"EXPRESSION FRACTION by source and gene class (top-{K_DIMS} null eigenvectors)")
print("High = epistasis along expression axes | Low = orthogonal (splicing/structural)")
print("=" * 100)

# 3a: Source comparison (cancer vs germline)
print("\n--- SOURCE COMPARISON (all genes) ---")
source_rows = []
for mk in df_decomp["model"].unique():
    sub = df_decomp[df_decomp["model"] == mk]
    display = sub.iloc[0]["display"]

    print(f"\n  {display}:")
    source_vals = {}
    for src in SOURCES:
        vals = sub[sub["source"] == src]["expression_fraction"]
        if len(vals) >= 10:
            source_vals[src] = vals
            print(f"    {SOURCES[src]:25s}: median={vals.median():.4f}, n={len(vals)}")

    # Pairwise comparisons
    for s1, s2 in [("tcga_doubles", "okgp_chr12"),
                    ("tcga_high_selection", "tcga_low_selection"),
                    ("tcga_high_selection", "okgp_matched_doubles")]:
        if s1 in source_vals and s2 in source_vals:
            _, p = mannwhitneyu(source_vals[s1], source_vals[s2], alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            source_rows.append({
                "model": display, "comparison": f"{SOURCES[s1]} vs {SOURCES[s2]}",
                "median_1": source_vals[s1].median(), "median_2": source_vals[s2].median(),
                "p": p,
            })
            print(f"    {SOURCES[s1]} vs {SOURCES[s2]}: p={p:.2e} {sig}")

df_source_comp = pd.DataFrame(source_rows)

# 3b: Gene class comparison within TCGA
print("\n\n--- GENE CLASS COMPARISON (TCGA sources combined) ---")
summary_rows = []
for mk in df_decomp["model"].unique():
    sub = df_decomp[(df_decomp["model"] == mk) &
                     df_decomp["source"].isin(["tcga_doubles", "tcga_high_selection", "tcga_low_selection"])]
    display = sub.iloc[0]["display"]

    onc = sub[sub["gene_class"] == "Oncogene"]["expression_fraction"]
    tsg = sub[sub["gene_class"] == "TSG"]["expression_fraction"]
    other = sub[sub["gene_class"] == "Other"]["expression_fraction"]

    if len(onc) < 10 or len(tsg) < 10:
        continue

    _, p_ot = mannwhitneyu(onc, tsg, alternative="two-sided")
    _, p_oo = mannwhitneyu(onc, other, alternative="two-sided")
    _, p_to = mannwhitneyu(tsg, other, alternative="two-sided")

    sig = "***" if p_ot < 0.001 else "**" if p_ot < 0.01 else "*" if p_ot < 0.05 else ""

    summary_rows.append({
        "model": display,
        "onc_median": onc.median(),
        "tsg_median": tsg.median(),
        "other_median": other.median(),
        "p_onc_vs_tsg": p_ot,
        "p_onc_vs_other": p_oo,
        "p_tsg_vs_other": p_to,
        "n_onc": len(onc),
        "n_tsg": len(tsg),
    })

    print(f"\n  {display}:")
    print(f"    Oncogene: median={onc.median():.4f} (n={len(onc)})")
    print(f"    TSG:      median={tsg.median():.4f} (n={len(tsg)})")
    print(f"    Other:    median={other.median():.4f} (n={len(other)})")
    print(f"    Onc vs TSG: p={p_ot:.2e} {sig}")

df_summary = pd.DataFrame(summary_rows)

# 3c: Null signature matching — what fraction of null pairs look expression-like vs splicing-like?
print("\n\n--- NULL SIGNATURE DISTRIBUTION ---")
for mk in df_decomp["model"].unique():
    sub = df_decomp[df_decomp["model"] == mk]
    display = sub.iloc[0]["display"]

    for src in SOURCES:
        vals = sub[sub["source"] == src]["expression_fraction"]
        if len(vals) < 50:
            continue
        expr_like = (vals > 0.5).mean()  # fraction with >50% expression-axis
        splicing_like = (vals <= 0.5).mean()
        print(f"  {display:15s} {SOURCES[src]:25s}: "
              f"expression-like={expr_like:.1%}, orthogonal={splicing_like:.1%} (n={len(vals)})")


# ---------------------------------------------------------------------------
# Cell 4: Figure — expression fraction distributions by gene class
# ---------------------------------------------------------------------------
models_to_plot = [mk for mk in MODELS if mk in df_decomp["model"].unique()]
n_models = len(models_to_plot)

if n_models > 0:
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    colors = {"Oncogene": "#CB6A49", "TSG": "#4A7FB5", "Other": "#CCCCCC"}

    for ax, mk in zip(axes, models_to_plot):
        sub = df_decomp[df_decomp["model"] == mk]
        display = MODELS[mk]

        for gene_class, color in colors.items():
            vals = sub[sub["gene_class"] == gene_class]["expression_fraction"]
            if len(vals) > 0:
                ax.hist(vals, bins=50, alpha=0.5, color=color, label=gene_class,
                        density=True, edgecolor="none")

        # Medians
        for gene_class, color in colors.items():
            vals = sub[sub["gene_class"] == gene_class]["expression_fraction"]
            if len(vals) > 0:
                ax.axvline(vals.median(), color=color, linestyle="--", linewidth=1.5)

        ax.set_xlabel(f"Expression fraction (top-{K_DIMS} null axes)")
        ax.set_title(display, fontweight="bold")
        ax.legend(fontsize=8)

        # Add p-value annotation
        row = df_summary[df_summary["model"] == display]
        if len(row) > 0:
            p = row.iloc[0]["p_onc_vs_tsg"]
            p_str = f"p={p:.2e}" if p < 0.01 else f"p={p:.3f}"
            ax.text(0.95, 0.95, f"Onc vs TSG:\n{p_str}",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    axes[0].set_ylabel("Density")
    fig.suptitle("Epistasis subspace decomposition: expression vs orthogonal",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_epistasis_subspace_decomposition.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_epistasis_subspace_decomposition.pdf",
                bbox_inches="tight")
    plt.show()
    print(f"\nSaved to {FIG_DIR}")


# ---------------------------------------------------------------------------
# Cell 5: 2D scatter — expression magnitude vs orthogonal magnitude
# ---------------------------------------------------------------------------
if n_models > 0:
    fig2, axes2 = plt.subplots(1, min(n_models, 4), figsize=(5 * min(n_models, 4), 5))
    if min(n_models, 4) == 1:
        axes2 = [axes2]

    for ax, mk in zip(axes2, models_to_plot[:4]):
        sub = df_decomp[df_decomp["model"] == mk]
        display = MODELS[mk]

        for gene_class in ["Other", "TSG", "Oncogene"]:  # plot Other first (background)
            vals = sub[sub["gene_class"] == gene_class]
            color = colors[gene_class]
            alpha = 0.1 if gene_class == "Other" else 0.5
            size = 3 if gene_class == "Other" else 10
            ax.scatter(vals["expression_magnitude"], vals["orthogonal_magnitude"],
                       c=color, s=size, alpha=alpha, label=gene_class, edgecolors="none")

        ax.set_xlabel("Expression-axis epistasis")
        ax.set_ylabel("Orthogonal-axis epistasis")
        ax.set_title(display, fontweight="bold")
        ax.legend(fontsize=7, markerscale=3)

        # Equal aspect
        max_val = max(sub["expression_magnitude"].quantile(0.99),
                      sub["orthogonal_magnitude"].quantile(0.99))
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=0.5)

    fig2.suptitle("Epistasis decomposition: expression vs orthogonal magnitude",
                  fontsize=13, fontweight="bold", y=1.02)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "fig_epistasis_2d_decomposition.png",
                 dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Cell 6: Summary statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("PAPER-READY STATISTICS")
print("=" * 90)

if len(df_summary) > 0:
    print(f"\nExpression fraction (top-{K_DIMS} null eigenvectors):")
    print(df_summary[["model", "onc_median", "tsg_median", "other_median",
                       "p_onc_vs_tsg", "n_onc", "n_tsg"]].to_string(index=False))

    # Across-model summary
    n_sig = (df_summary["p_onc_vs_tsg"] < 0.05).sum()
    mean_onc = df_summary["onc_median"].mean()
    mean_tsg = df_summary["tsg_median"].mean()
    print(f"\nAcross models:")
    print(f"  Mean oncogene expression fraction: {mean_onc:.4f}")
    print(f"  Mean TSG expression fraction: {mean_tsg:.4f}")
    print(f"  Significant (p<0.05): {n_sig}/{len(df_summary)} models")

    if mean_onc > mean_tsg:
        print(f"  → Oncogenes have MORE expression-aligned epistasis")
    else:
        print(f"  → TSGs have MORE expression-aligned epistasis")

# Save decomposition data
df_decomp.to_parquet(FIG_DIR / "epistasis_subspace_decomposition.parquet", index=False)
print(f"\nDecomposition data saved to {FIG_DIR / 'epistasis_subspace_decomposition.parquet'}")
