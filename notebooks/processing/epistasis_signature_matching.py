"""
Epistasis signature matching with validation chain.

Step 1: Build "splicing epistasis subspace" from top 10% FAS pairs
Step 2: VALIDATE — does KRAS G60G-Q61K rank high in this subspace
        among 13K KRAS neighborhood pairs? If yes, the subspace
        captures real splicing biology, not FAS-specific noise.
Step 3: Apply validated subspace to TCGA vs 1kGP comparison.

Runs on cluster (needs embedding DBs).
"""

import sys, os, logging
from pathlib import Path

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
from genebeddings.epistasis_features import (
    _list_epistasis_ids_from_db, _load_residual_from_db, fit_covariance,
)

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

KRAS_TARGET = "KRAS:12:25227343:G:T:N|KRAS:12:25227344:A:T:N"
FAS_DATA = DATA_DIR / "fas_subset.csv"
K_SUBSPACE = 20  # dims for splicing subspace

MODELS = {
    "borzoi": "Borzoi",
    "alphagenome": "AlphaGenome",
    "nt500_multi": "NT-500M",
    "mutbert": "MutBERT",
    "dnabert": "DNABERT",
    "ntv3_100m_post": "NTv3-post",
    "rinalmo": "RiNALMo",
}

DIST_BINS = [(1, 6), (6, 21), (21, 101), (101, 501)]
DIST_LABELS = ["1-5bp", "6-20bp", "21-100bp", "101-500bp"]


def subspace_projection_fraction(residual, V_subspace):
    """Fraction of residual variance in the subspace: |P·r|²/|r|²"""
    r = residual.astype(np.float64)
    r_sq = np.dot(r, r)
    if r_sq < 1e-30:
        return 0.0
    proj = V_subspace @ (V_subspace.T @ r)
    return float(np.dot(proj, proj) / r_sq)


# =========================================================================
# STEP 1: Build FAS splicing subspace from top 10% epistatic pairs
# =========================================================================
print("=" * 90)
print("STEP 1: Build FAS splicing subspace")
print("=" * 90)

# Load FAS empirical data
fas_df = pd.read_csv(FAS_DATA)
fas_df["abs_epi"] = fas_df["empirical_epistasis"].abs()
threshold = fas_df["abs_epi"].quantile(0.90)
fas_top_ids = set(fas_df[fas_df["abs_epi"] >= threshold]["epistasis_id"])
fas_bottom_ids = set(fas_df[fas_df["abs_epi"] <= fas_df["abs_epi"].quantile(0.50)]["epistasis_id"])
print(f"FAS top 10%: {len(fas_top_ids)} pairs (|epi| >= {threshold:.4f})")
print(f"FAS bottom 50%: {len(fas_bottom_ids)} pairs (for negative control)")

# For each model: load top FAS residuals, fit covariance, extract eigenvectors
fas_subspaces = {}  # model -> V_subspace (d, k)

for mk, display in MODELS.items():
    fas_db = OUTPUT_BASE / "fas_exon" / f"{mk}.db"
    if not fas_db.exists():
        continue

    db = VariantEmbeddingDB(str(fas_db))
    top_residuals = []
    for eid in fas_top_ids:
        r = _load_residual_from_db(db, eid)
        if r is not None:
            top_residuals.append(r.astype(np.float64))
    db.close()

    if len(top_residuals) < 50:
        continue

    # Fit covariance of top FAS residuals
    arr = np.stack(top_residuals)
    cov = np.cov(arr, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    V_fas = eigenvectors[:, -K_SUBSPACE:]  # top-k eigenvectors

    fas_subspaces[mk] = V_fas
    print(f"  {display}: FAS subspace from {len(top_residuals)} residuals, "
          f"d={V_fas.shape[0]}, k={K_SUBSPACE}")


# =========================================================================
# STEP 2: VALIDATE — KRAS G60G-Q61K in FAS subspace
# =========================================================================
print(f"\n{'=' * 90}")
print("STEP 2: VALIDATION — KRAS G60G-Q61K ranking in FAS subspace")
print("If the compensatory pair ranks high, the subspace captures")
print("real splicing biology, not FAS-specific noise.")
print("=" * 90)

for mk, display in MODELS.items():
    if mk not in fas_subspaces:
        continue
    V_fas = fas_subspaces[mk]

    kras_db = OUTPUT_BASE / "kras_neighborhood" / f"{mk}.db"
    if not kras_db.exists():
        continue

    db = VariantEmbeddingDB(str(kras_db))
    kras_ids = _list_epistasis_ids_from_db(db)

    scores = []
    target_score = None
    for eid in kras_ids:
        r = _load_residual_from_db(db, eid)
        if r is None:
            continue
        frac = subspace_projection_fraction(r, V_fas)
        scores.append(frac)
        if eid == KRAS_TARGET:
            target_score = frac
    db.close()

    if target_score is None or len(scores) < 100:
        print(f"  {display}: KRAS target not found or too few pairs")
        continue

    scores = np.array(scores)
    percentile = (scores <= target_score).mean() * 100

    # Also check: do top FAS residuals (from FAS itself) score higher than
    # bottom FAS residuals? (sanity check that the subspace is meaningful)
    fas_db2 = OUTPUT_BASE / "fas_exon" / f"{mk}.db"
    db2 = VariantEmbeddingDB(str(fas_db2))
    top_scores = []
    bottom_scores = []
    for eid in fas_top_ids:
        r = _load_residual_from_db(db2, eid)
        if r is not None:
            top_scores.append(subspace_projection_fraction(r, V_fas))
    for eid in list(fas_bottom_ids)[:500]:  # sample for speed
        r = _load_residual_from_db(db2, eid)
        if r is not None:
            bottom_scores.append(subspace_projection_fraction(r, V_fas))
    db2.close()

    _, p_fas_sanity = mannwhitneyu(top_scores, bottom_scores, alternative="greater")

    print(f"\n  {display}:")
    print(f"    KRAS G60G-Q61K FAS-subspace fraction: {target_score:.4f}")
    print(f"    KRAS neighborhood median: {np.median(scores):.4f}")
    print(f"    KRAS percentile: {percentile:.1f}%")
    print(f"    FAS sanity: top10% median={np.median(top_scores):.4f}, "
          f"bottom50% median={np.median(bottom_scores):.4f}, "
          f"p={p_fas_sanity:.2e}")


# =========================================================================
# STEP 3: TCGA vs 1kGP in validated FAS subspace
# =========================================================================
print(f"\n{'=' * 90}")
print("STEP 3: TCGA somatic vs 1kGP germline — FAS subspace projection")
print("=" * 90)

TEST_SOURCES = {
    "tcga_high_selection": "TCGA somatic",
    "okgp_matched_doubles": "1kGP germline",
}

all_rows = []

for mk, display in MODELS.items():
    if mk not in fas_subspaces:
        continue
    V_fas = fas_subspaces[mk]

    for source_key, source_label in TEST_SOURCES.items():
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

            frac = subspace_projection_fraction(r, V_fas)
            gene = eid.split(":")[0]
            parts = eid.split("|")
            dist = abs(int(parts[1].split(":")[2]) - int(parts[0].split(":")[2]))

            all_rows.append({
                "model": mk, "display": display,
                "source": source_key, "source_label": source_label,
                "epistasis_id": eid, "gene": gene, "distance": dist,
                "fas_subspace_fraction": frac,
                "gene_class": ("Oncogene" if gene in ONCOGENES else
                               "TSG" if gene in TSGS else "Other"),
            })
        db.close()

df = pd.DataFrame(all_rows)
print(f"\nTotal: {len(df)} entries")

# Within-bin comparison
print(f"\n--- FAS subspace projection: TCGA vs 1kGP (within-bin) ---")
for mk, display in MODELS.items():
    sub = df[df["model"] == mk]
    tcga = sub[sub["source"] == "tcga_high_selection"]
    germ = sub[sub["source"] == "okgp_matched_doubles"]
    if len(tcga) < 50 or len(germ) < 50:
        continue

    print(f"\n  {display}:")
    print(f"  {'bin':>12s} | {'n_TCGA':>7s} {'n_germ':>7s} | "
          f"{'TCGA median':>12s} {'germ median':>12s} | {'p':>12s} {'dir':>6s}")

    for (lo, hi), label in zip(DIST_BINS, DIST_LABELS):
        t = tcga[(tcga["distance"] >= lo) & (tcga["distance"] < hi)]
        g = germ[(germ["distance"] >= lo) & (germ["distance"] < hi)]
        if len(t) < 10 or len(g) < 10:
            continue
        tv = t["fas_subspace_fraction"]
        gv = g["fas_subspace_fraction"]
        _, p = mannwhitneyu(tv, gv, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        direction = "TCGA>" if tv.median() > gv.median() else "germ>"
        print(f"  {label:>12s} | {len(t):7d} {len(g):7d} | "
              f"{tv.median():12.4f} {gv.median():12.4f} | "
              f"{p:11.2e}{sig:>3s} {direction:>6s}")

    # Pooled
    tv = tcga["fas_subspace_fraction"]
    gv = germ["fas_subspace_fraction"]
    _, p = mannwhitneyu(tv, gv, alternative="two-sided")
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    direction = "TCGA>" if tv.median() > gv.median() else "germ>"
    print(f"  {'POOLED':>12s} | {len(tv):7d} {len(gv):7d} | "
          f"{tv.median():12.4f} {gv.median():12.4f} | "
          f"{p:11.2e}{sig:>3s} {direction:>6s}")


# Gene class
print(f"\n--- Gene class within TCGA ---")
for mk, display in MODELS.items():
    tcga = df[(df["model"] == mk) & (df["source"] == "tcga_high_selection")]
    if len(tcga) < 50:
        continue
    onc = tcga[tcga["gene_class"] == "Oncogene"]["fas_subspace_fraction"]
    tsg = tcga[tcga["gene_class"] == "TSG"]["fas_subspace_fraction"]
    if len(onc) < 5 or len(tsg) < 5:
        continue
    _, p = mannwhitneyu(onc, tsg, alternative="two-sided")
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {display:15s}: Onc={onc.median():.4f} TSG={tsg.median():.4f} p={p:.2e} {sig}")


# =========================================================================
# Summary
# =========================================================================
print(f"\n{'=' * 90}")
print("SUMMARY")
print("=" * 90)
print(f"\nValidation chain:")
print(f"  1. FAS splicing subspace defined from top 10% experimental epistasis (k={K_SUBSPACE})")
print(f"  2. KRAS G60G-Q61K tested against 13K neighborhood pairs")
print(f"  3. TCGA vs 1kGP compared in validated subspace")
print(f"\nIf KRAS ranks high AND TCGA differs from 1kGP,")
print(f"the finding is: cancer double mutations produce epistasis that")
print(f"resembles experimentally validated splicing interactions.")

print("\nDone.")
