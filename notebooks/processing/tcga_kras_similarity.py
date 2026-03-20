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

# All sources to score
SOURCES = {
    "tcga_high_selection": "TCGA high-sel",
    "tcga_low_selection": "TCGA low-sel",
    "tcga_doubles": "TCGA original",
    "okgp_matched_doubles": "1kGP germline",
}

DIST_BINS = [(1, 6), (6, 21), (21, 101), (101, 501)]
DIST_LABELS = ["1-5bp", "6-20bp", "21-100bp", "101-500bp"]


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-20 or nb < 1e-20:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =========================================================================
# Score ALL sources against KRAS template
# =========================================================================
for mk, display in MODELS.items():
    print(f"\n{'=' * 90}")
    print(f"{display}: scoring all sources against KRAS G60G-Q61K")
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
    print(f"  KRAS template: dim={len(kras_vec)}, |ε|={np.linalg.norm(kras_vec):.6f}")

    # Score all sources
    all_rows = []
    for source_key, source_label in SOURCES.items():
        src_db = OUTPUT_BASE / source_key / f"{mk}.db"
        if not src_db.exists():
            continue

        db = VariantEmbeddingDB(str(src_db))
        epi_ids = _list_epistasis_ids_from_db(db)
        logger.info("%s / %s: %d pairs", display, source_label, len(epi_ids))

        for eid in epi_ids:
            r = _load_residual_from_db(db, eid)
            if r is None:
                continue
            r = r.astype(np.float64)
            r_norm = np.linalg.norm(r)
            if r_norm < 1e-20:
                continue

            gene = eid.split(":")[0]
            parts = eid.split("|")
            dist = abs(int(parts[1].split(":")[2]) - int(parts[0].split(":")[2]))

            all_rows.append({
                "source": source_key,
                "source_label": source_label,
                "epistasis_id": eid,
                "gene": gene,
                "distance": dist,
                "cosine_to_kras": cosine_sim(r, kras_vec),
                "abs_cosine_to_kras": abs(cosine_sim(r, kras_vec)),
                "residual_magnitude": float(r_norm),
                "gene_class": ("Oncogene" if gene in ONCOGENES else
                               "TSG" if gene in TSGS else "Other"),
            })
        db.close()

    df = pd.DataFrame(all_rows)
    print(f"  Total scored: {len(df)}")
    print(f"  Per source: {df.groupby('source_label').size().to_dict()}")

    # ---- Within-bin comparison: TCGA vs 1kGP ----
    from scipy.stats import mannwhitneyu

    print(f"\n  WITHIN-BIN: |cosine to KRAS| — TCGA sources vs 1kGP germline")
    print(f"  {'bin':>12s} | {'source':>15s} {'n':>6s} {'median':>8s} | "
          f"{'vs 1kGP p':>12s} {'dir':>6s}")

    for (lo, hi), label in zip(DIST_BINS, DIST_LABELS):
        germ = df[(df["source"] == "okgp_matched_doubles") &
                   (df["distance"] >= lo) & (df["distance"] < hi)]
        if len(germ) < 10:
            continue

        gv = germ["abs_cosine_to_kras"]

        for src_key in ["tcga_high_selection", "tcga_low_selection", "tcga_doubles"]:
            src = df[(df["source"] == src_key) &
                      (df["distance"] >= lo) & (df["distance"] < hi)]
            if len(src) < 10:
                continue

            sv = src["abs_cosine_to_kras"]
            _, p = mannwhitneyu(sv, gv, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            direction = "TCGA>" if sv.median() > gv.median() else "germ>"

            print(f"  {label:>12s} | {SOURCES[src_key]:>15s} {len(src):6d} "
                  f"{sv.median():8.4f} | {p:11.2e}{sig:>3s} {direction:>6s}")

        # Print germline reference
        print(f"  {label:>12s} | {'1kGP germline':>15s} {len(germ):6d} "
              f"{gv.median():8.4f} | {'(reference)':>15s}")
        print()

    # ---- Pooled comparison ----
    print(f"\n  POOLED comparison:")
    germ_all = df[df["source"] == "okgp_matched_doubles"]["abs_cosine_to_kras"]
    for src_key in ["tcga_high_selection", "tcga_low_selection", "tcga_doubles"]:
        src_all = df[df["source"] == src_key]["abs_cosine_to_kras"]
        if len(src_all) < 10:
            continue
        _, p = mannwhitneyu(src_all, germ_all, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        direction = "TCGA>" if src_all.median() > germ_all.median() else "germ>"
        print(f"    {SOURCES[src_key]:>15s}: median={src_all.median():.4f} vs "
              f"1kGP={germ_all.median():.4f}, p={p:.2e}{sig} {direction}")

    # ---- Tail enrichment: at top X% of combined ranking, what fraction is TCGA? ----
    print(f"\n  TAIL ENRICHMENT: fraction TCGA among most KRAS-like pairs")

    # Combined TCGA high-sel + 1kGP (the matched pair)
    matched = df[df["source"].isin(["tcga_high_selection", "okgp_matched_doubles"])].copy()
    matched = matched.sort_values("abs_cosine_to_kras", ascending=False).reset_index(drop=True)
    n_tcga_total = (matched["source"] == "tcga_high_selection").sum()
    n_germ_total = (matched["source"] == "okgp_matched_doubles").sum()
    base_rate = n_tcga_total / len(matched)

    print(f"    Base rate TCGA: {base_rate:.1%} ({n_tcga_total}/{len(matched)})")

    from scipy.stats import fisher_exact
    for top_pct in [1, 2, 5, 10]:
        n_top = max(1, int(len(matched) * top_pct / 100))
        top = matched.head(n_top)
        n_tcga_top = (top["source"] == "tcga_high_selection").sum()
        n_germ_top = (top["source"] == "okgp_matched_doubles").sum()
        frac_tcga = n_tcga_top / n_top

        # Fisher's exact
        table = [[n_tcga_top, n_germ_top],
                 [n_tcga_total - n_tcga_top, n_germ_total - n_germ_top]]
        odds, p_fisher = fisher_exact(table, alternative="two-sided")
        sig = "***" if p_fisher < 0.001 else "**" if p_fisher < 0.01 else "*" if p_fisher < 0.05 else ""

        print(f"    Top {top_pct:>2d}% ({n_top:>5d}): TCGA={n_tcga_top}/{n_top} "
              f"({frac_tcga:.1%} vs base {base_rate:.1%}), "
              f"OR={odds:.2f}, p={p_fisher:.2e}{sig}")

    # ---- Top TCGA nominations ----
    tcga_only = df[df["source"] == "tcga_doubles"].sort_values(
        "abs_cosine_to_kras", ascending=False).reset_index(drop=True)

    print(f"\n  TOP 20 CANCER GENE nominations ({display}):")
    cancer = tcga_only[tcga_only["gene_class"] != "Other"].head(20)
    print(f"  {'rank':>4s} {'gene':>12s} {'class':>10s} {'cos':>7s} {'|ε|':>8s} {'dist':>5s}")
    print(f"  " + "-" * 55)
    for _, r in cancer.iterrows():
        print(f"  {int(r.name)+1:>4d} {r['gene']:>12s} {r['gene_class']:>10s} "
              f"{r['cosine_to_kras']:>+7.3f} {r['residual_magnitude']:>8.4f} "
              f"{r['distance']:>5d}")

    # Save
    out_path = FIG_DIR / f"tcga_kras_similarity_{mk}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")

print("\nDone.")
