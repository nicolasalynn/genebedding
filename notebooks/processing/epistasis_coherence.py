"""
Epistasis directional coherence: is cancer epistasis geometrically organized?

The simplest geometric question: do TCGA residual vectors point in more
consistent directions than 1kGP residual vectors?

If cancer mutations interact through a small number of functional pathways,
their residuals should cluster in direction (high pairwise cosine).
If germline pairs interact randomly, their residuals scatter (low cosine).

No subspaces, no eigenvectors, no external references. Just: are cancer
epistasis residuals more organized than germline ones?

Then as a secondary analysis: are the TCGA directions more similar to
FAS/KRAS directions than 1kGP directions are?

Runs on cluster (needs embedding DBs).
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
ANNOT_DIR = EPISTASIS_PAPER_ROOT / "data" / "annotations"
DATA_DIR = EPISTASIS_PAPER_ROOT / "data"
FIG_DIR = EPISTASIS_PAPER_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from genebeddings import VariantEmbeddingDB
from genebeddings.epistasis_features import _list_epistasis_ids_from_db, _load_residual_from_db

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

MODELS = {
    "borzoi": "Borzoi",
    "alphagenome": "AlphaGenome",
    "nt500_multi": "NT-500M",
    "mutbert": "MutBERT",
    "dnabert": "DNABERT",
    "ntv3_100m_post": "NTv3-post",
    "rinalmo": "RiNALMo",
}

SOURCES = {
    "tcga_high_selection": "TCGA somatic",
    "okgp_matched_doubles": "1kGP germline",
}

DIST_BINS = [(1, 6), (6, 21), (21, 101), (101, 501)]
DIST_LABELS = ["1-5bp", "6-20bp", "21-100bp", "101-500bp"]

N_SAMPLE_COSINE = 500  # pairs to sample for pairwise cosine (keep fast)


def pairwise_abs_cosine(vectors, n_sample=500, rng=None):
    """Mean |cosine similarity| among sampled pairs of vectors."""
    if rng is None:
        rng = np.random.RandomState(42)
    n = len(vectors)
    if n < 2:
        return float("nan")

    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-20, None)
    unit = vectors / norms

    # Sample random pairs
    n_pairs = min(n_sample, n * (n - 1) // 2)
    idx1 = rng.randint(0, n, size=n_pairs)
    idx2 = rng.randint(0, n, size=n_pairs)
    # Avoid self-pairs
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    if len(idx1) == 0:
        return float("nan")

    cosines = np.abs(np.sum(unit[idx1] * unit[idx2], axis=1))
    return float(np.mean(cosines))


# =========================================================================
# Load all residuals
# =========================================================================
print("=" * 90)
print("LOADING RESIDUALS")
print("=" * 90)

# Store residuals grouped by (model, source, distance_bin)
residuals = defaultdict(list)  # (mk, source, bin_label) -> list of (residual, gene, eid)

for mk, display in MODELS.items():
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
            if np.linalg.norm(r) < 1e-20:
                continue

            gene = eid.split(":")[0]
            parts = eid.split("|")
            dist = abs(int(parts[1].split(":")[2]) - int(parts[0].split(":")[2]))

            # Assign to distance bin
            for (lo, hi), label in zip(DIST_BINS, DIST_LABELS):
                if lo <= dist < hi:
                    residuals[(mk, source_key, label)].append((r, gene, eid))
                    break

        db.close()

print(f"\nLoaded residuals for {len(residuals)} (model, source, bin) groups")


# =========================================================================
# Analysis 1: Directional coherence — TCGA vs 1kGP
# =========================================================================
print(f"\n{'=' * 90}")
print("ANALYSIS 1: DIRECTIONAL COHERENCE")
print("Mean |cosine similarity| among pairs within each group")
print("Higher = residuals point in more consistent directions = organized epistasis")
print("=" * 90)

rng = np.random.RandomState(42)
coherence_rows = []

for mk, display in MODELS.items():
    print(f"\n--- {display} ---")
    print(f"{'bin':>12s} | {'n_TCGA':>7s} {'n_germ':>7s} | "
          f"{'TCGA coh':>9s} {'germ coh':>9s} | {'diff':>8s} {'p':>12s}")

    for label in DIST_LABELS:
        tcga_group = residuals.get((mk, "tcga_high_selection", label), [])
        germ_group = residuals.get((mk, "okgp_matched_doubles", label), [])

        if len(tcga_group) < 20 or len(germ_group) < 20:
            continue

        tcga_vecs = np.stack([r for r, _, _ in tcga_group])
        germ_vecs = np.stack([r for r, _, _ in germ_group])

        # Coherence
        tcga_coh = pairwise_abs_cosine(tcga_vecs, N_SAMPLE_COSINE, rng)
        germ_coh = pairwise_abs_cosine(germ_vecs, N_SAMPLE_COSINE, rng)

        # Bootstrap p-value: shuffle source labels, recompute difference
        combined = np.vstack([tcga_vecs, germ_vecs])
        n_tcga = len(tcga_vecs)
        observed_diff = tcga_coh - germ_coh

        null_diffs = []
        for _ in range(200):
            perm = rng.permutation(len(combined))
            perm_tcga = combined[perm[:n_tcga]]
            perm_germ = combined[perm[n_tcga:]]
            d = (pairwise_abs_cosine(perm_tcga, N_SAMPLE_COSINE, rng)
                 - pairwise_abs_cosine(perm_germ, N_SAMPLE_COSINE, rng))
            null_diffs.append(d)

        null_diffs = np.array(null_diffs)
        p_value = np.mean(np.abs(null_diffs) >= abs(observed_diff))
        p_value = max(p_value, 1 / 201)  # floor

        sig = "***" if p_value < 0.005 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

        coherence_rows.append({
            "model": display, "bin": label,
            "tcga_coherence": tcga_coh, "germ_coherence": germ_coh,
            "diff": observed_diff, "p": p_value,
            "n_tcga": len(tcga_group), "n_germ": len(germ_group),
        })

        print(f"{label:>12s} | {len(tcga_group):7d} {len(germ_group):7d} | "
              f"{tcga_coh:9.5f} {germ_coh:9.5f} | {observed_diff:+8.5f} "
              f"{p_value:11.3f}{sig:>3s}")

df_coh = pd.DataFrame(coherence_rows)


# =========================================================================
# Analysis 2: Gene class coherence within TCGA
# =========================================================================
print(f"\n{'=' * 90}")
print("ANALYSIS 2: GENE CLASS COHERENCE WITHIN TCGA")
print("Are oncogene pairs more directionally coherent than TSG pairs?")
print("=" * 90)

for mk, display in MODELS.items():
    # Collect all TCGA residuals for this model
    all_tcga = []
    for label in DIST_LABELS:
        all_tcga.extend(residuals.get((mk, "tcga_high_selection", label), []))

    if len(all_tcga) < 50:
        continue

    onc_vecs = np.stack([r for r, g, _ in all_tcga if g in ONCOGENES])
    tsg_vecs = np.stack([r for r, g, _ in all_tcga if g in TSGS])
    other_vecs = np.stack([r for r, g, _ in all_tcga if g not in ONCOGENES and g not in TSGS])

    if len(onc_vecs) < 10 or len(tsg_vecs) < 10:
        continue

    onc_coh = pairwise_abs_cosine(onc_vecs, N_SAMPLE_COSINE, rng)
    tsg_coh = pairwise_abs_cosine(tsg_vecs, N_SAMPLE_COSINE, rng)
    other_coh = pairwise_abs_cosine(other_vecs, N_SAMPLE_COSINE, rng)

    print(f"\n  {display}:")
    print(f"    Oncogene coherence: {onc_coh:.5f} (n={len(onc_vecs)})")
    print(f"    TSG coherence:      {tsg_coh:.5f} (n={len(tsg_vecs)})")
    print(f"    Other coherence:    {other_coh:.5f} (n={len(other_vecs)})")


# =========================================================================
# Figure
# =========================================================================
if len(df_coh) > 0:
    models_in_data = df_coh["model"].unique()
    n_m = len(models_in_data)

    fig, axes = plt.subplots(1, n_m, figsize=(3.5 * n_m, 4), sharey=True)
    if n_m == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models_in_data):
        sub = df_coh[df_coh["model"] == model_name]
        x = np.arange(len(sub))
        w = 0.35

        ax.bar(x - w/2, sub["tcga_coherence"], w, color="#d62728",
               alpha=0.8, label="TCGA somatic")
        ax.bar(x + w/2, sub["germ_coherence"], w, color="#1f77b4",
               alpha=0.8, label="1kGP germline")

        # Significance markers
        for i, (_, row) in enumerate(sub.iterrows()):
            if row["p"] < 0.05:
                higher = max(row["tcga_coherence"], row["germ_coherence"])
                ax.text(i, higher + 0.0002, "*" * (3 if row["p"] < 0.005 else
                        2 if row["p"] < 0.01 else 1),
                        ha="center", fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(sub["bin"], fontsize=7, rotation=30, ha="right")
        ax.set_title(model_name, fontweight="bold", fontsize=10)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Mean |cosine similarity|\n(directional coherence)")
    fig.suptitle("Is cancer epistasis geometrically organized?",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_epistasis_coherence.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_epistasis_coherence.pdf", bbox_inches="tight")
    print(f"\nFigure saved to {FIG_DIR}")


# =========================================================================
# Summary
# =========================================================================
print(f"\n{'=' * 90}")
print("SUMMARY")
print("=" * 90)

if len(df_coh) > 0:
    tcga_higher = (df_coh["diff"] > 0).sum()
    germ_higher = (df_coh["diff"] < 0).sum()
    n_sig = (df_coh["p"] < 0.05).sum()

    print(f"\nAcross {len(df_coh)} model×bin comparisons:")
    print(f"  TCGA more coherent: {tcga_higher}/{len(df_coh)}")
    print(f"  1kGP more coherent: {germ_higher}/{len(df_coh)}")
    print(f"  Significant (p<0.05): {n_sig}/{len(df_coh)}")

    print(f"\n  If TCGA is consistently more coherent:")
    print(f"  → Cancer epistasis is geometrically organized (fewer pathways)")
    print(f"  If 1kGP is more coherent:")
    print(f"  → Population epistasis follows more regular patterns")
    print(f"  If no difference:")
    print(f"  → The direction of epistasis doesn't distinguish cancer from germline")

print("\nDone.")
