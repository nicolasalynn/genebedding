"""
Epistasis subspace decomposition: TCGA somatic vs 1kGP germline.

Same anchor mutations, distance-matched. The only difference is the
second mutation: cancer-selected vs population-occurring.

For each pair, decompose the epistasis residual into:
  ε_expression = component along null's dominant axes (expression)
  ε_orthogonal = everything else (splicing, structural, etc.)

Question: do cancer pairs interact through different mechanisms
than germline pairs, even if the overall magnitude is similar?
"""

import sys, os, logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

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

from genebeddings import VariantEmbeddingDB
from genebeddings.epistasis_features import _list_epistasis_ids_from_db, _load_residual_from_db

# Two groups: TCGA somatic vs 1kGP germline (matched anchors)
SOURCES = {
    "tcga_high_selection": "TCGA somatic",
    "okgp_matched_doubles": "1kGP germline",
}

MODELS = {
    "borzoi": "Borzoi",
    "alphagenome": "AlphaGenome",
    "nt500_multi": "NT-500M",
    "mutbert": "MutBERT",
    "dnabert": "DNABERT",
    "ntv3_100m_post": "NTv3-post",
    "rinalmo": "RiNALMo",
}

K_DIMS = 50

# Distance bins for within-bin comparison
DIST_BINS = [(1, 6), (6, 21), (21, 101), (101, 501)]
DIST_LABELS = ["1-5bp", "6-20bp", "21-100bp", "101-500bp"]

print(f"Comparing: {list(SOURCES.values())}")
print(f"Models: {list(MODELS.values())}")
print(f"Expression subspace: top-{K_DIMS} null eigenvectors")


# ---------------------------------------------------------------------------
# Compute decomposition for both sources
# ---------------------------------------------------------------------------
all_rows = []

for mk, display in MODELS.items():
    cov_path = CACHE_DIR / f"{mk}_cov_inv.npz"
    if not cov_path.exists():
        logger.warning("No null cov for %s, skipping", mk)
        continue

    null_cov = np.load(cov_path)["cov"]
    d = null_cov.shape[0]

    eigenvalues, eigenvectors = np.linalg.eigh(null_cov)
    V_expr = eigenvectors[:, -K_DIMS:]  # top-k null eigenvectors

    for source_key, source_label in SOURCES.items():
        db_path = OUTPUT_BASE / source_key / f"{mk}.db"
        if not db_path.exists():
            continue

        db = VariantEmbeddingDB(str(db_path))
        epi_ids = _list_epistasis_ids_from_db(db)
        logger.info("%s / %s: %d pairs", display, source_label, len(epi_ids))

        for eid in epi_ids:
            r = _load_residual_from_db(db, eid)
            if r is None:
                continue
            r = r.astype(np.float64)
            r_norm_sq = np.dot(r, r)
            if r_norm_sq < 1e-30:
                continue

            r_proj = V_expr @ (V_expr.T @ r)
            r_proj_sq = np.dot(r_proj, r_proj)
            expr_frac = r_proj_sq / r_norm_sq

            gene = eid.split(":")[0]
            parts = eid.split("|")
            dist = abs(int(parts[1].split(":")[2]) - int(parts[0].split(":")[2]))

            all_rows.append({
                "model": mk, "display": display,
                "source": source_key, "source_label": source_label,
                "epistasis_id": eid, "gene": gene, "distance": dist,
                "expression_fraction": float(expr_frac),
                "orthogonal_fraction": 1.0 - float(expr_frac),
                "residual_magnitude": float(np.sqrt(r_norm_sq)),
                "expression_magnitude": float(np.sqrt(r_proj_sq)),
                "orthogonal_magnitude": float(np.sqrt(max(0, r_norm_sq - r_proj_sq))),
                "gene_class": ("Oncogene" if gene in ONCOGENES else
                               "TSG" if gene in TSGS else "Other"),
            })

        db.close()

df = pd.DataFrame(all_rows)
print(f"\nTotal: {len(df)} entries")
print(df.groupby(["display", "source_label"]).size().unstack(fill_value=0))


# ---------------------------------------------------------------------------
# Head-to-head: TCGA vs 1kGP expression fraction (within-bin)
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("HEAD-TO-HEAD: expression fraction — TCGA somatic vs 1kGP germline")
print("Within-distance-bin comparison (no residualization)")
print("=" * 100)

for mk, display in MODELS.items():
    sub = df[df["model"] == mk]
    tcga = sub[sub["source"] == "tcga_high_selection"]
    germ = sub[sub["source"] == "okgp_matched_doubles"]

    if len(tcga) < 50 or len(germ) < 50:
        continue

    print(f"\n--- {display} ---")
    print(f"{'bin':>12s} | {'n_TCGA':>7s} {'n_germ':>7s} | "
          f"{'TCGA expr_frac':>15s} {'germ expr_frac':>15s} | "
          f"{'p (two-sided)':>14s} {'dir':>6s}")

    for (lo, hi), label in zip(DIST_BINS, DIST_LABELS):
        t = tcga[(tcga["distance"] >= lo) & (tcga["distance"] < hi)]
        g = germ[(germ["distance"] >= lo) & (germ["distance"] < hi)]

        if len(t) < 10 or len(g) < 10:
            continue

        tv = t["expression_fraction"]
        gv = g["expression_fraction"]
        _, p = mannwhitneyu(tv, gv, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        direction = "TCGA>" if tv.median() > gv.median() else "germ>"

        print(f"{label:>12s} | {len(t):7d} {len(g):7d} | "
              f"{tv.median():15.4f} {gv.median():15.4f} | "
              f"{p:13.2e}{sig:>3s} {direction:>6s}")

    # Pooled
    _, p_pool = mannwhitneyu(tcga["expression_fraction"],
                              germ["expression_fraction"], alternative="two-sided")
    sig = "***" if p_pool < 0.001 else "**" if p_pool < 0.01 else "*" if p_pool < 0.05 else ""
    direction = "TCGA>" if tcga["expression_fraction"].median() > germ["expression_fraction"].median() else "germ>"
    print(f"{'POOLED':>12s} | {len(tcga):7d} {len(germ):7d} | "
          f"{tcga['expression_fraction'].median():15.4f} "
          f"{germ['expression_fraction'].median():15.4f} | "
          f"{p_pool:13.2e}{sig:>3s} {direction:>6s}")


# ---------------------------------------------------------------------------
# Gene class breakdown within TCGA
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("GENE CLASS: expression fraction within TCGA somatic pairs")
print("=" * 100)

for mk, display in MODELS.items():
    tcga = df[(df["model"] == mk) & (df["source"] == "tcga_high_selection")]
    if len(tcga) < 50:
        continue

    onc = tcga[tcga["gene_class"] == "Oncogene"]["expression_fraction"]
    tsg = tcga[tcga["gene_class"] == "TSG"]["expression_fraction"]
    other = tcga[tcga["gene_class"] == "Other"]["expression_fraction"]

    if len(onc) < 5 or len(tsg) < 5:
        continue

    _, p_ot = mannwhitneyu(onc, tsg, alternative="two-sided")
    sig = "***" if p_ot < 0.001 else "**" if p_ot < 0.01 else "*" if p_ot < 0.05 else ""

    print(f"\n  {display}:")
    print(f"    Oncogene:  {onc.median():.4f} (n={len(onc)})")
    print(f"    TSG:       {tsg.median():.4f} (n={len(tsg)})")
    print(f"    Other:     {other.median():.4f} (n={len(other)})")
    print(f"    Onc vs TSG: p={p_ot:.2e} {sig}")


# ---------------------------------------------------------------------------
# Figure: expression fraction by source and gene class
# ---------------------------------------------------------------------------
models_to_plot = [mk for mk in MODELS if mk in df["model"].unique()]
n_m = len(models_to_plot)

if n_m > 0:
    fig, axes = plt.subplots(2, n_m, figsize=(4 * n_m, 8))
    if n_m == 1:
        axes = axes.reshape(2, 1)

    source_colors = {"TCGA somatic": "#d62728", "1kGP germline": "#1f77b4"}
    class_colors = {"Oncogene": "#CB6A49", "TSG": "#4A7FB5", "Other": "#CCCCCC"}

    for j, mk in enumerate(models_to_plot):
        display = MODELS[mk]
        sub = df[df["model"] == mk]

        # Top row: source comparison
        ax = axes[0, j]
        for src_label, color in source_colors.items():
            vals = sub[sub["source_label"] == src_label]["expression_fraction"]
            if len(vals) > 0:
                ax.hist(vals, bins=50, alpha=0.5, color=color,
                        density=True, label=src_label, edgecolor="none")
                ax.axvline(vals.median(), color=color, linestyle="--", linewidth=1.5)
        ax.set_title(display, fontweight="bold", fontsize=10)
        ax.set_xlabel("Expression fraction")
        if j == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7)

        # Add p-value
        tcga = sub[sub["source"] == "tcga_high_selection"]["expression_fraction"]
        germ = sub[sub["source"] == "okgp_matched_doubles"]["expression_fraction"]
        if len(tcga) > 10 and len(germ) > 10:
            _, p = mannwhitneyu(tcga, germ, alternative="two-sided")
            ax.text(0.95, 0.95, f"p={p:.2e}", transform=ax.transAxes,
                    fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        # Bottom row: gene class within TCGA
        ax2 = axes[1, j]
        tcga_sub = sub[sub["source"] == "tcga_high_selection"]
        for gc, color in class_colors.items():
            vals = tcga_sub[tcga_sub["gene_class"] == gc]["expression_fraction"]
            if len(vals) > 0:
                ax2.hist(vals, bins=50, alpha=0.5, color=color,
                         density=True, label=gc, edgecolor="none")
                ax2.axvline(vals.median(), color=color, linestyle="--", linewidth=1.5)
        ax2.set_xlabel("Expression fraction")
        if j == 0:
            ax2.set_ylabel("Density")
        ax2.legend(fontsize=7)

    axes[0, 0].text(-0.15, 1.1, "a", transform=axes[0, 0].transAxes,
                     fontsize=14, fontweight="bold")
    axes[1, 0].text(-0.15, 1.1, "b", transform=axes[1, 0].transAxes,
                     fontsize=14, fontweight="bold")

    fig.suptitle("Epistasis subspace: expression-axis vs orthogonal\n"
                 "(a) TCGA somatic vs 1kGP germline  (b) Gene classes within TCGA",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_epistasis_subspace.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_epistasis_subspace.pdf", bbox_inches="tight")
    print(f"\nFigure saved to {FIG_DIR / 'fig_epistasis_subspace.png'}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

for mk, display in MODELS.items():
    sub = df[df["model"] == mk]
    tcga = sub[sub["source"] == "tcga_high_selection"]
    germ = sub[sub["source"] == "okgp_matched_doubles"]
    if len(tcga) < 50 or len(germ) < 50:
        continue

    _, p_source = mannwhitneyu(tcga["expression_fraction"],
                                germ["expression_fraction"], alternative="two-sided")

    t_expr = (tcga["expression_fraction"] > 0.5).mean()
    g_expr = (germ["expression_fraction"] > 0.5).mean()

    print(f"\n  {display}:")
    print(f"    TCGA expression-dominant: {t_expr:.1%} of pairs")
    print(f"    1kGP expression-dominant: {g_expr:.1%} of pairs")
    print(f"    TCGA vs 1kGP expr_frac: p={p_source:.2e}")

# Save data
df.to_parquet(FIG_DIR / "epistasis_subspace_decomposition.parquet", index=False)
print(f"\nData saved to {FIG_DIR / 'epistasis_subspace_decomposition.parquet'}")
