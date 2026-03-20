"""
Find TCGA pairs that resemble the KRAS G60G-Q61K compensatory event.

Simple: take the KRAS G60G-Q61K residual vector as a template.
For each TCGA pair, compute cosine similarity with this template.
Rank. Top hits are pairs the model sees as interacting the same way.

Only use models that actually detect KRAS: AlphaGenome (0.04th %ile)
and Borzoi (1.3rd %ile).

Runs on cluster (needs embedding DBs).
"""

import sys, os, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

KRAS_TARGET = "KRAS:12:25227343:G:T:N|KRAS:12:25227344:A:T:N"

# TCGA metadata
tcga_meta = None
tcga_meta_path = DATA_DIR / "tcga_doubles_pairs.tsv"
if tcga_meta_path.exists():
    tcga_meta = pd.read_csv(tcga_meta_path, sep="\t").set_index("epistasis_id")

# Models that detect KRAS (from paper: AlphaGenome 0.04%ile, Borzoi 1.3%ile)
MODELS = {
    "alphagenome": "AlphaGenome",
    "borzoi": "Borzoi",
}


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-20 or nb < 1e-20:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =========================================================================
# Load KRAS template + score all TCGA pairs
# =========================================================================
for mk, display in MODELS.items():
    print(f"\n{'=' * 90}")
    print(f"{display}: finding TCGA pairs similar to KRAS G60G-Q61K")
    print(f"{'=' * 90}")

    # Load KRAS template
    kras_db = OUTPUT_BASE / "kras_neighborhood" / f"{mk}.db"
    if not kras_db.exists():
        print(f"  No KRAS DB, skipping")
        continue

    db = VariantEmbeddingDB(str(kras_db))
    kras_residual = _load_residual_from_db(db, KRAS_TARGET)
    db.close()

    if kras_residual is None:
        print(f"  KRAS target not found, skipping")
        continue

    kras_vec = kras_residual.astype(np.float64)
    kras_norm = np.linalg.norm(kras_vec)
    print(f"  KRAS template: dim={len(kras_vec)}, |ε|={kras_norm:.6f}")

    # Score all TCGA pairs
    tcga_db = OUTPUT_BASE / "tcga_doubles" / f"{mk}.db"
    if not tcga_db.exists():
        print(f"  No TCGA DB, skipping")
        continue

    db = VariantEmbeddingDB(str(tcga_db))
    epi_ids = _list_epistasis_ids_from_db(db)
    print(f"  Scoring {len(epi_ids)} TCGA pairs...")

    rows = []
    for eid in epi_ids:
        r = _load_residual_from_db(db, eid)
        if r is None:
            continue
        r = r.astype(np.float64)
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-20:
            continue

        cos = cosine_sim(r, kras_vec)

        gene = eid.split(":")[0]
        parts = eid.split("|")
        dist = abs(int(parts[1].split(":")[2]) - int(parts[0].split(":")[2]))

        row = {
            "epistasis_id": eid,
            "gene": gene,
            "distance": dist,
            "cosine_to_kras": cos,
            "abs_cosine_to_kras": abs(cos),
            "residual_magnitude": float(r_norm),
            "gene_class": ("Oncogene" if gene in ONCOGENES else
                           "TSG" if gene in TSGS else "Other"),
        }

        # Add metadata
        if tcga_meta is not None and eid in tcga_meta.index:
            for col in ["n_cases_both", "cond_prob_dependent"]:
                if col in tcga_meta.columns:
                    row[col] = tcga_meta.loc[eid, col]

        rows.append(row)

    db.close()
    df = pd.DataFrame(rows)
    df = df.sort_values("abs_cosine_to_kras", ascending=False).reset_index(drop=True)
    df["percentile"] = (df.index + 1) / len(df) * 100

    # Top 30 most KRAS-like pairs
    print(f"\n  TOP 30 TCGA pairs most similar to KRAS G60G-Q61K ({display}):")
    print(f"  {'rank':>4s} {'gene':>12s} {'class':>10s} {'cos':>7s} {'|ε|':>8s} "
          f"{'dist':>5s} {'n_pts':>5s} {'epistasis_id'}")
    print(f"  " + "-" * 100)

    for i, r in df.head(30).iterrows():
        n_pts = f"{int(r['n_cases_both']):>5d}" if "n_cases_both" in r and pd.notna(r.get("n_cases_both")) else "    ?"
        print(f"  {i+1:>4d} {r['gene']:>12s} {r['gene_class']:>10s} "
              f"{r['cosine_to_kras']:>+7.3f} {r['residual_magnitude']:>8.4f} "
              f"{r['distance']:>5d} {n_pts} {r['epistasis_id']}")

    # Top cancer gene pairs
    cancer = df[df["gene_class"] != "Other"].head(20)
    print(f"\n  TOP 20 CANCER GENE pairs most similar to KRAS ({display}):")
    print(f"  {'rank':>4s} {'gene':>12s} {'class':>10s} {'cos':>7s} {'|ε|':>8s} "
          f"{'dist':>5s} {'%ile':>6s}")
    print(f"  " + "-" * 70)
    for _, r in cancer.iterrows():
        print(f"  {int(r.name)+1:>4d} {r['gene']:>12s} {r['gene_class']:>10s} "
              f"{r['cosine_to_kras']:>+7.3f} {r['residual_magnitude']:>8.4f} "
              f"{r['distance']:>5d} {r['percentile']:>5.1f}%")

    # Stats
    print(f"\n  Summary:")
    print(f"    Mean |cos| all: {df['abs_cosine_to_kras'].mean():.4f}")
    print(f"    Mean |cos| cancer genes: {df[df['gene_class'] != 'Other']['abs_cosine_to_kras'].mean():.4f}")
    print(f"    Mean |cos| oncogenes: {df[df['gene_class'] == 'Oncogene']['abs_cosine_to_kras'].mean():.4f}")
    print(f"    Mean |cos| TSGs: {df[df['gene_class'] == 'TSG']['abs_cosine_to_kras'].mean():.4f}")

    # Are cancer genes more KRAS-like?
    from scipy.stats import mannwhitneyu
    cancer_cos = df[df["gene_class"] != "Other"]["abs_cosine_to_kras"]
    other_cos = df[df["gene_class"] == "Other"]["abs_cosine_to_kras"]
    if len(cancer_cos) > 10:
        _, p = mannwhitneyu(cancer_cos, other_cos, alternative="two-sided")
        print(f"    Cancer vs Other |cos|: p={p:.2e}")

    # Enrichment: are cancer genes overrepresented in top 5%?
    from scipy.stats import hypergeom
    n_top = max(1, int(len(df) * 0.05))
    top = df.head(n_top)
    k_obs = (top["gene_class"] != "Other").sum()
    K = (df["gene_class"] != "Other").sum()
    N = len(df)
    k_exp = K * n_top / N
    p_enrich = hypergeom.sf(k_obs - 1, N, K, n_top)
    fold = k_obs / (k_exp + 1e-10)
    print(f"    Cancer gene enrichment in top 5%: fold={fold:.2f}, "
          f"k={k_obs}/{n_top}, exp={k_exp:.1f}, p={p_enrich:.4f}")

    # Save
    out_path = FIG_DIR / f"tcga_kras_similarity_{mk}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Full ranking saved to {out_path}")

print("\nDone.")
