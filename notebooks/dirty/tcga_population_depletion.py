"""
Population depletion analysis + corrective-subset enrichment.

Two analyses:
1. POPULATION DEPLETION: For each TCGA double-variant pair, check if the
   constituent mutations co-occur in 1000 Genomes. Top epistasis-ranked pairs
   should be depleted from population haplotypes (negative selection).

2. CORRECTIVE SUBSET ENRICHMENT: Restrict to pairs where the double mutant is
   closer to wild type than expected (MR < 1, low len_WT_M12 relative to
   len_WT_M12_exp). Test cancer gene enrichment only within this subset.
   Rationale: corrective epistasis is the uniquely epistatic signal — baselines
   can't identify pairs where two mutations cancel out.
"""

# ---------------------------------------------------------------------------
# Cell 1: Setup
# ---------------------------------------------------------------------------
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import sys, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.dataset as ds
from scipy.stats import hypergeom, mannwhitneyu, spearmanr

ROOT = Path.cwd()
for _ in range(5):
    if (ROOT / "notebooks" / "paper_data_config.py").exists():
        break
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, "/Users/nicolaslynn/Documents/phd/ParseTCGA")

from paper_data_config import EPISTASIS_PAPER_ROOT
from parsegenomes import OKGPData

PARQUET_DIR = EPISTASIS_PAPER_ROOT / "combined_parquets" / "new_embeddings"
ANNOT_DIR = EPISTASIS_PAPER_ROOT / "data" / "annotations"
DATA_DIR = EPISTASIS_PAPER_ROOT / "data"
FIG_DIR = ROOT / "notebooks" / "dirty" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OKGP_DB = "/Users/nicolaslynn/Downloads/likely_garbage/okgp_query/1kg_mutations.duckdb"
kg = OKGPData(OKGP_DB)

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

# Distance residualization
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

def parse_gene(eid):
    return eid.split(":")[0]

def parse_distance(eid):
    parts = eid.split("|")
    try:
        return abs(int(parts[1].split(":")[2]) - int(parts[0].split(":")[2]))
    except (IndexError, ValueError):
        return np.nan

def parse_mut_ids(eid):
    """Extract the two mutation IDs from an epistasis ID, stripping strand."""
    parts = eid.split("|")
    mut1 = parts[0]
    mut2 = parts[1]
    # Strip strand suffix :P or :N
    for suffix in (":P", ":N"):
        if mut1.endswith(suffix):
            mut1 = mut1[:-len(suffix)]
        if mut2.endswith(suffix):
            mut2 = mut2[:-len(suffix)]
    return mut1, mut2

print(f"TCGA pairs: {len(tcga_meta)}")
print(f"1kGP DB loaded from {OKGP_DB}")
print(f"Cancer genes: {len(CANCER_GENES)} (Onc: {len(ONCOGENES)}, TSG: {len(TSGS)})")


# ---------------------------------------------------------------------------
# Cell 2: Population co-occurrence lookup for all TCGA pairs
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("POPULATION CO-OCCURRENCE LOOKUP")
print("=" * 80)

# For each TCGA pair, check if both constituent mutations exist in 1kGP
# and if any individual carries both
pop_rows = []
n_total = len(tcga_meta)

# Cache mutation carrier sets to avoid redundant queries
_carrier_cache = {}

def get_carriers(mut_id):
    if mut_id not in _carrier_cache:
        try:
            _carrier_cache[mut_id] = set(kg.samples_with_mutation(mut_id))
        except Exception:
            _carrier_cache[mut_id] = set()
    return _carrier_cache[mut_id]

print(f"Looking up {n_total} TCGA pairs in 1kGP...")
for i, row in tcga_meta.iterrows():
    eid = row["epistasis_id"]
    mut1, mut2 = parse_mut_ids(eid)

    carriers_1 = get_carriers(mut1)
    carriers_2 = get_carriers(mut2)
    both = carriers_1 & carriers_2

    pop_rows.append({
        "epistasis_id": eid,
        "gene": row["gene"],
        "n_carriers_1": len(carriers_1),
        "n_carriers_2": len(carriers_2),
        "n_carriers_both": len(both),
        "either_in_1kgp": len(carriers_1) > 0 or len(carriers_2) > 0,
        "both_muts_in_1kgp": len(carriers_1) > 0 and len(carriers_2) > 0,
        "cooccur_in_1kgp": len(both) > 0,
    })

    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{n_total}...")

df_pop = pd.DataFrame(pop_rows)
print(f"\nDone. Queried {len(_carrier_cache)} unique mutations.")
print(f"\nPopulation co-occurrence summary:")
print(f"  Pairs where mutation 1 found in 1kGP: {(df_pop['n_carriers_1'] > 0).sum()}")
print(f"  Pairs where mutation 2 found in 1kGP: {(df_pop['n_carriers_2'] > 0).sum()}")
print(f"  Pairs where BOTH mutations found in 1kGP: {df_pop['both_muts_in_1kgp'].sum()}")
print(f"  Pairs where both co-occur in same individual: {df_pop['cooccur_in_1kgp'].sum()}")

# Save for later use
df_pop.to_csv(FIG_DIR / "tcga_population_cooccurrence.csv", index=False)


# ---------------------------------------------------------------------------
# Cell 3: Load model metrics and merge with population data
# ---------------------------------------------------------------------------
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

# Use a representative set of models for speed
TEST_MODELS = ["nt50_multi", "nt50_3mer", "mutbert", "borzoi", "alphagenome", "evo2"]

model_data = {}
for mk in TEST_MODELS:
    ppath = PARQUET_DIR / f"epistasis_metrics_{mk}_combined.parquet"
    if not ppath.exists():
        continue
    dset = ds.dataset(ppath)
    avail = {f.name for f in dset.schema}
    cols = [c for c in ["source", "epistasis_id", "log_magnitude_ratio",
                         "magnitude_ratio", "epi_R_singles",
                         "len_WT_M1", "len_WT_M2", "len_WT_M12", "len_WT_M12_exp"]
            if c in avail]
    df = dset.to_table(
        columns=cols,
        filter=ds.field("source") == "tcga_doubles"
    ).to_pandas().drop_duplicates(subset=["epistasis_id"])

    # Merge with population data
    df = df.merge(df_pop, on="epistasis_id", how="left")
    df["distance"] = df["epistasis_id"].map(parse_distance)

    # Compute derived metrics
    if "len_WT_M1" in df.columns and "len_WT_M2" in df.columns:
        df["max_single"] = np.maximum(df["len_WT_M1"], df["len_WT_M2"])
        df["sum_singles"] = df["len_WT_M1"] + df["len_WT_M2"]

    # Distance-residualized log_MR
    valid = df.dropna(subset=["log_magnitude_ratio", "distance"]).copy()
    valid["resid_log_mr"] = distance_residualize(
        valid["log_magnitude_ratio"].values,
        valid["distance"].values,
    )
    valid["abs_resid_log_mr"] = np.abs(valid["resid_log_mr"])
    valid["epi_rank"] = valid["abs_resid_log_mr"].rank(ascending=False, method="min").astype(int)
    valid["epi_percentile"] = 100 * valid["epi_rank"] / len(valid)

    # Also rank by baseline
    valid["resid_sum_singles"] = distance_residualize(
        valid["sum_singles"].values, valid["distance"].values
    )
    valid["bl_rank"] = valid["resid_sum_singles"].abs().rank(ascending=False, method="min").astype(int)

    model_data[mk] = valid

print(f"Loaded {len(model_data)} models")


# ---------------------------------------------------------------------------
# Cell 4: Population depletion test — proper version
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TEST: POPULATION DEPLETION (proper)")
print("=" * 80)
print("Do top epistasis-ranked TCGA pairs have lower population co-occurrence?")

for mk, df in model_data.items():
    display = MODELS.get(mk, mk)
    print(f"\n--- {display} ---")

    for top_pct in [1, 5, 10, 50]:
        n_top = max(1, int(len(df) * top_pct / 100))
        top = df.nsmallest(n_top, "epi_rank")
        bottom = df.nlargest(n_top, "epi_rank")
        rest = df[df["epi_rank"] > n_top]

        top_cooccur = top["cooccur_in_1kgp"].mean()
        rest_cooccur = rest["cooccur_in_1kgp"].mean()
        bottom_cooccur = bottom["cooccur_in_1kgp"].mean()

        top_both_muts = top["both_muts_in_1kgp"].mean()
        rest_both_muts = rest["both_muts_in_1kgp"].mean()

        print(f"  Top {top_pct}%: co-occur={top_cooccur:.3f}, both_in_1kgp={top_both_muts:.3f} | "
              f"Bottom {top_pct}%: co-occur={bottom_cooccur:.3f} | "
              f"Rest: co-occur={rest_cooccur:.3f}")

    # Overall correlation: epistasis rank vs population prevalence
    has_pop = df[df["both_muts_in_1kgp"]].copy()
    if len(has_pop) > 20:
        rho, p = spearmanr(has_pop["epi_rank"], has_pop["n_carriers_both"])
        print(f"  Spearman(epi_rank, n_carriers_both) among pairs in 1kGP: rho={rho:.3f}, p={p:.4f}")

    # Compare epistasis vs baseline ranking
    for top_pct in [5]:
        n_top = max(1, int(len(df) * top_pct / 100))
        top_epi = df.nsmallest(n_top, "epi_rank")
        top_bl = df.nsmallest(n_top, "bl_rank")
        print(f"  Top 5% co-occur rate: EPISTASIS={top_epi['cooccur_in_1kgp'].mean():.3f}, "
              f"BASELINE(sum_singles)={top_bl['cooccur_in_1kgp'].mean():.3f}, "
              f"ALL={df['cooccur_in_1kgp'].mean():.3f}")


# ---------------------------------------------------------------------------
# Cell 5: Corrective-subset enrichment
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("CORRECTIVE SUBSET ENRICHMENT")
print("=" * 80)
print("Restrict to pairs where double mutant is closer to WT than expected (MR < 1).")
print("These are uniquely epistatic — baselines can't identify cancellation.")

for mk, df in model_data.items():
    display = MODELS.get(mk, mk)
    print(f"\n--- {display} ---")

    # Corrective subset: residualized log_MR < 0 (double mutant closer to WT)
    corrective = df[df["resid_log_mr"] < 0].copy()
    # Strongly corrective: bottom 25% of log_MR
    threshold = df["resid_log_mr"].quantile(0.25)
    strongly_corrective = df[df["resid_log_mr"] < threshold].copy()

    for subset_name, subset_df in [("All corrective (MR<1)", corrective),
                                     ("Strongly corrective (bottom 25%)", strongly_corrective)]:
        if len(subset_df) < 50:
            continue

        genes = subset_df["epistasis_id"].map(parse_gene)
        n = len(subset_df)

        for gs_label, gs_set in [("TSG", TSGS), ("Oncogene", ONCOGENES), ("Cancer", CANCER_GENES)]:
            in_set = genes.isin(gs_set)
            K = int(in_set.sum())
            # Compare to full dataset rate
            all_genes = df["epistasis_id"].map(parse_gene)
            K_all = int(all_genes.isin(gs_set).sum())
            rate_subset = K / n
            rate_all = K_all / len(df)
            fold = rate_subset / (rate_all + 1e-10)
            # Hypergeometric: is the subset enriched vs the full set?
            p = hypergeom.sf(K - 1, len(df), K_all, n)

            if p < 0.1 or gs_label == "TSG":
                print(f"  {subset_name}: {gs_label} fold={fold:.2f} (p={p:.4f}), "
                      f"{K}/{n} in subset vs {K_all}/{len(df)} overall")

    # Also check: within corrective subset, are pairs depleted from population?
    if len(corrective) > 50:
        corr_cooccur = corrective["cooccur_in_1kgp"].mean()
        all_cooccur = df["cooccur_in_1kgp"].mean()
        print(f"  Population co-occur: corrective={corr_cooccur:.3f} vs all={all_cooccur:.3f}")


# ---------------------------------------------------------------------------
# Cell 6: Combined visualization
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (mk, df) in enumerate(model_data.items()):
    if idx >= 6:
        break
    display = MODELS.get(mk, mk)
    ax = axes[idx // 3, idx % 3]

    # Bin by epistasis percentile, compute co-occurrence rate
    df_sorted = df.sort_values("epi_rank")
    n_bins = 20
    bin_size = len(df_sorted) // n_bins
    bin_centers = []
    bin_cooccur = []
    bin_both_muts = []
    for b in range(n_bins):
        start = b * bin_size
        end = start + bin_size if b < n_bins - 1 else len(df_sorted)
        chunk = df_sorted.iloc[start:end]
        pct = (start + end) / 2 / len(df_sorted) * 100
        bin_centers.append(pct)
        bin_cooccur.append(chunk["cooccur_in_1kgp"].mean())
        bin_both_muts.append(chunk["both_muts_in_1kgp"].mean())

    ax.plot(bin_centers, bin_cooccur, "o-", color="tab:red", label="Co-occur in 1kGP", markersize=4)
    ax.plot(bin_centers, bin_both_muts, "s--", color="tab:blue", label="Both muts in 1kGP", markersize=3, alpha=0.6)
    ax.set_xlabel("Epistasis percentile (0=most epistatic)")
    ax.set_ylabel("Fraction of pairs")
    ax.set_title(display, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle("Population prevalence by epistasis rank\n(lower percentile = more epistatic)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "population_depletion_by_rank.png", bbox_inches="tight", dpi=150)
print(f"\nSaved to {FIG_DIR / 'population_depletion_by_rank.png'}")


# ---------------------------------------------------------------------------
# Cell 7: Summary statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print(f"\n1. Population co-occurrence across all TCGA pairs:")
print(f"   Either mutation in 1kGP: {df_pop['either_in_1kgp'].mean():.1%}")
print(f"   Both mutations in 1kGP: {df_pop['both_muts_in_1kgp'].mean():.1%}")
print(f"   Co-occur in same individual: {df_pop['cooccur_in_1kgp'].mean():.1%}")

# Aggregate across models
print(f"\n2. Top 5% epistasis-ranked pairs vs rest (averaged across {len(model_data)} models):")
top_rates = []
rest_rates = []
bl_rates = []
for mk, df in model_data.items():
    n_top = max(1, int(len(df) * 0.05))
    top = df.nsmallest(n_top, "epi_rank")
    rest = df[df["epi_rank"] > n_top]
    top_bl = df.nsmallest(n_top, "bl_rank")
    top_rates.append(top["cooccur_in_1kgp"].mean())
    rest_rates.append(rest["cooccur_in_1kgp"].mean())
    bl_rates.append(top_bl["cooccur_in_1kgp"].mean())

print(f"   Epistasis top 5%: {np.mean(top_rates):.3f} co-occur rate")
print(f"   Baseline top 5%:  {np.mean(bl_rates):.3f} co-occur rate")
print(f"   Remaining 95%:    {np.mean(rest_rates):.3f} co-occur rate")

direction = "DEPLETED" if np.mean(top_rates) < np.mean(rest_rates) else "ENRICHED"
print(f"   Top epistatic pairs are {direction} from population co-occurrence")

print(f"\nDone. All outputs in {FIG_DIR}")
