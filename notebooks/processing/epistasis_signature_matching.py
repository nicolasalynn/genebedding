"""
Epistasis signature matching: do TCGA pairs resemble known epistasis events?

Ground truth signatures (defined from experimental data):
  1. FAS splicing signature: average residual of top 10% FAS epistatic pairs
  2. KRAS compensatory signature: residual of the G60G-Q61K pair
  3. MST1R splicing signature: average residual of top 10% MST1R pairs

For each TCGA and 1kGP pair, compute cosine similarity with each signature.
If cancer pairs match these signatures more than germline pairs, the models
are detecting real interaction biology — not just abstract geometry.

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
from genebeddings.epistasis_features import _list_epistasis_ids_from_db, _load_residual_from_db

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

# KRAS target pair
KRAS_TARGET = "KRAS:12:25227343:G:T:N|KRAS:12:25227344:A:T:N"

# Load empirical FAS and MST1R data to find top epistatic pairs
FAS_DATA = DATA_DIR / "fas_subset.csv"
MST1R_CANDIDATES = [DATA_DIR / "mst1r_splicing_pairs.tsv",
                     DATA_DIR / "mst1r_subset.csv"]

MODELS = {
    "borzoi": "Borzoi",
    "alphagenome": "AlphaGenome",
    "nt500_multi": "NT-500M",
    "mutbert": "MutBERT",
    "dnabert": "DNABERT",
    "ntv3_100m_post": "NTv3-post",
    "rinalmo": "RiNALMo",
}

# Sources to compare
TEST_SOURCES = {
    "tcga_high_selection": "TCGA somatic",
    "okgp_matched_doubles": "1kGP germline",
}

# Distance bins
DIST_BINS = [(1, 6), (6, 21), (21, 101), (101, 501)]
DIST_LABELS = ["1-5bp", "6-20bp", "21-100bp", "101-500bp"]


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-20 or nb < 1e-20:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =========================================================================
# Step 1: Build ground truth signatures from FAS, MST1R, KRAS
# =========================================================================
print("=" * 90)
print("BUILDING GROUND TRUTH EPISTASIS SIGNATURES")
print("=" * 90)

# Load FAS empirical data to identify top 10% epistatic pairs
fas_top_ids = set()
if FAS_DATA.exists():
    fas_df = pd.read_csv(FAS_DATA)
    fas_df["abs_epi"] = fas_df["empirical_epistasis"].abs()
    threshold = fas_df["abs_epi"].quantile(0.90)
    fas_top = fas_df[fas_df["abs_epi"] >= threshold]
    fas_top_ids = set(fas_top["epistasis_id"])
    print(f"FAS: {len(fas_top_ids)} top 10% pairs (|empirical_epistasis| >= {threshold:.4f})")
else:
    print(f"WARNING: FAS data not found at {FAS_DATA}")

# MST1R top pairs
mst_top_ids = set()
for mst_path in MST1R_CANDIDATES:
    if mst_path.exists():
        mst_df = pd.read_csv(mst_path, sep="\t" if mst_path.suffix == ".tsv" else ",")
        # Find the epistasis metric column
        for col in ["ae_skipping_pct", "empirical_epistasis", "score"]:
            if col in mst_df.columns:
                mst_df["abs_metric"] = mst_df[col].abs()
                threshold_m = mst_df["abs_metric"].quantile(0.90)
                mst_top = mst_df[mst_df["abs_metric"] >= threshold_m]
                if "epistasis_id" in mst_top.columns:
                    mst_top_ids = set(mst_top["epistasis_id"])
                    print(f"MST1R: {len(mst_top_ids)} top 10% pairs from {mst_path.name}")
                break
        break

signatures = {}  # model -> {sig_name: vector}

for mk, display in MODELS.items():
    sigs = {}

    # FAS signature: average residual of top 10% FAS pairs
    fas_db = OUTPUT_BASE / "fas_exon" / f"{mk}.db"
    if fas_db.exists() and fas_top_ids:
        db = VariantEmbeddingDB(str(fas_db))
        fas_residuals = []
        for eid in fas_top_ids:
            r = _load_residual_from_db(db, eid)
            if r is not None:
                fas_residuals.append(r.astype(np.float64))
        db.close()
        if fas_residuals:
            fas_sig = np.mean(fas_residuals, axis=0)
            fas_sig = fas_sig / (np.linalg.norm(fas_sig) + 1e-20)  # unit vector
            sigs["FAS_splicing"] = fas_sig
            print(f"  {display}: FAS signature from {len(fas_residuals)} residuals, dim={len(fas_sig)}")

    # KRAS signature: single pair residual
    kras_db = OUTPUT_BASE / "kras_neighborhood" / f"{mk}.db"
    if kras_db.exists():
        db = VariantEmbeddingDB(str(kras_db))
        r = _load_residual_from_db(db, KRAS_TARGET)
        db.close()
        if r is not None:
            kras_sig = r.astype(np.float64)
            kras_sig = kras_sig / (np.linalg.norm(kras_sig) + 1e-20)
            sigs["KRAS_compensatory"] = kras_sig
            print(f"  {display}: KRAS signature, dim={len(kras_sig)}")

    # MST1R signature
    mst_db = OUTPUT_BASE / "mst1r_splicing" / f"{mk}.db"
    if mst_db.exists() and mst_top_ids:
        db = VariantEmbeddingDB(str(mst_db))
        mst_residuals = []
        for eid in mst_top_ids:
            r = _load_residual_from_db(db, eid)
            if r is not None:
                mst_residuals.append(r.astype(np.float64))
        db.close()
        if mst_residuals:
            mst_sig = np.mean(mst_residuals, axis=0)
            mst_sig = mst_sig / (np.linalg.norm(mst_sig) + 1e-20)
            sigs["MST1R_splicing"] = mst_sig
            print(f"  {display}: MST1R signature from {len(mst_residuals)} residuals")

    if sigs:
        signatures[mk] = sigs


# =========================================================================
# Step 2: Score every TCGA and 1kGP pair against signatures
# =========================================================================
print(f"\n{'=' * 90}")
print("SCORING PAIRS AGAINST GROUND TRUTH SIGNATURES")
print("=" * 90)

all_rows = []

for mk, display in MODELS.items():
    if mk not in signatures:
        continue
    sigs = signatures[mk]

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

            gene = eid.split(":")[0]
            parts = eid.split("|")
            dist = abs(int(parts[1].split(":")[2]) - int(parts[0].split(":")[2]))

            row = {
                "model": mk, "display": display,
                "source": source_key, "source_label": source_label,
                "epistasis_id": eid, "gene": gene, "distance": dist,
                "residual_magnitude": float(np.linalg.norm(r)),
                "gene_class": ("Oncogene" if gene in ONCOGENES else
                               "TSG" if gene in TSGS else "Other"),
            }

            # Cosine similarity with each signature
            for sig_name, sig_vec in sigs.items():
                if len(sig_vec) == len(r):
                    row[f"cos_{sig_name}"] = cosine_sim(r, sig_vec)
                    row[f"abs_cos_{sig_name}"] = abs(cosine_sim(r, sig_vec))

            all_rows.append(row)

        db.close()

df = pd.DataFrame(all_rows)
print(f"\nTotal: {len(df)} entries")
print(df.groupby(["display", "source_label"]).size().unstack(fill_value=0))


# =========================================================================
# Step 3: Compare TCGA vs 1kGP signature similarity
# =========================================================================
sig_names = [c.replace("abs_cos_", "") for c in df.columns if c.startswith("abs_cos_")]

print(f"\n{'=' * 90}")
print("HEAD-TO-HEAD: signature similarity — TCGA somatic vs 1kGP germline")
print("Within-distance-bin, two-sided Mann-Whitney")
print(f"{'=' * 90}")

for mk, display in MODELS.items():
    sub = df[df["model"] == mk]
    tcga = sub[sub["source"] == "tcga_high_selection"]
    germ = sub[sub["source"] == "okgp_matched_doubles"]

    if len(tcga) < 50 or len(germ) < 50:
        continue

    print(f"\n--- {display} ---")

    for sig_name in sig_names:
        col = f"abs_cos_{sig_name}"
        if col not in df.columns:
            continue

        print(f"\n  Signature: {sig_name}")
        print(f"  {'bin':>12s} | {'n_TCGA':>7s} {'n_germ':>7s} | "
              f"{'TCGA median':>12s} {'germ median':>12s} | {'p':>12s} {'dir':>6s}")

        for (lo, hi), label in zip(DIST_BINS, DIST_LABELS):
            t = tcga[(tcga["distance"] >= lo) & (tcga["distance"] < hi)]
            g = germ[(germ["distance"] >= lo) & (germ["distance"] < hi)]
            if len(t) < 10 or len(g) < 10:
                continue

            tv = t[col].dropna()
            gv = g[col].dropna()
            if len(tv) < 10 or len(gv) < 10:
                continue

            _, p = mannwhitneyu(tv, gv, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            direction = "TCGA>" if tv.median() > gv.median() else "germ>"

            print(f"  {label:>12s} | {len(tv):7d} {len(gv):7d} | "
                  f"{tv.median():12.4f} {gv.median():12.4f} | "
                  f"{p:11.2e}{sig:>3s} {direction:>6s}")

        # Pooled
        tv = tcga[col].dropna()
        gv = germ[col].dropna()
        if len(tv) > 10 and len(gv) > 10:
            _, p = mannwhitneyu(tv, gv, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            direction = "TCGA>" if tv.median() > gv.median() else "germ>"
            print(f"  {'POOLED':>12s} | {len(tv):7d} {len(gv):7d} | "
                  f"{tv.median():12.4f} {gv.median():12.4f} | "
                  f"{p:11.2e}{sig:>3s} {direction:>6s}")


# =========================================================================
# Step 4: Gene class breakdown — do cancer gene types match different signatures?
# =========================================================================
print(f"\n{'=' * 90}")
print("GENE CLASS: do oncogenes vs TSGs match different signatures?")
print(f"{'=' * 90}")

for mk, display in MODELS.items():
    tcga = df[(df["model"] == mk) & (df["source"] == "tcga_high_selection")]
    if len(tcga) < 50:
        continue

    print(f"\n--- {display} ---")

    for sig_name in sig_names:
        col = f"abs_cos_{sig_name}"
        if col not in tcga.columns:
            continue

        onc = tcga[tcga["gene_class"] == "Oncogene"][col].dropna()
        tsg = tcga[tcga["gene_class"] == "TSG"][col].dropna()

        if len(onc) < 5 or len(tsg) < 5:
            continue

        _, p = mannwhitneyu(onc, tsg, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        print(f"  {sig_name:25s}: Onc={onc.median():.4f} (n={len(onc)}), "
              f"TSG={tsg.median():.4f} (n={len(tsg)}), p={p:.2e} {sig}")


# =========================================================================
# Step 5: Figure
# =========================================================================
models_to_plot = [mk for mk in MODELS if mk in df["model"].unique()]
n_m = len(models_to_plot)
n_sigs = len(sig_names)

if n_m > 0 and n_sigs > 0:
    fig, axes = plt.subplots(n_sigs, n_m, figsize=(4 * n_m, 4 * n_sigs),
                              squeeze=False)

    source_colors = {"TCGA somatic": "#d62728", "1kGP germline": "#1f77b4"}

    for j, mk in enumerate(models_to_plot):
        display = MODELS[mk]
        sub = df[df["model"] == mk]

        for i, sig_name in enumerate(sig_names):
            ax = axes[i, j]
            col = f"abs_cos_{sig_name}"
            if col not in sub.columns:
                ax.set_visible(False)
                continue

            for src_label, color in source_colors.items():
                vals = sub[sub["source_label"] == src_label][col].dropna()
                if len(vals) > 0:
                    ax.hist(vals, bins=50, alpha=0.5, color=color,
                            density=True, label=src_label, edgecolor="none")
                    ax.axvline(vals.median(), color=color, linestyle="--", linewidth=1.5)

            if i == 0:
                ax.set_title(display, fontweight="bold", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{sig_name}\nDensity", fontsize=9)
            ax.set_xlabel("|cos similarity|", fontsize=8)
            ax.legend(fontsize=6)

            # p-value
            tcga_v = sub[sub["source"] == "tcga_high_selection"][col].dropna()
            germ_v = sub[sub["source"] == "okgp_matched_doubles"][col].dropna()
            if len(tcga_v) > 10 and len(germ_v) > 10:
                _, p = mannwhitneyu(tcga_v, germ_v, alternative="two-sided")
                ax.text(0.95, 0.95, f"p={p:.1e}", transform=ax.transAxes,
                        fontsize=7, ha="right", va="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.suptitle("Do TCGA pairs resemble known epistasis signatures more than germline?",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_epistasis_signature_matching.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_epistasis_signature_matching.pdf",
                bbox_inches="tight")
    print(f"\nFigure saved to {FIG_DIR}")

print("\nDone.")
