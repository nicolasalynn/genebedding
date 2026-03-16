"""
Generate selection comparison pairs — v2 with distance matching.

Changes from v1:
- TCGA search radius expanded to 500bp (was 100bp)
- 1kGP partner search expanded to 500bp
- Distance-matched sampling: all 3 sets are sampled to the SAME distance distribution
- More genes covered
- Gene-level matching where possible

Output: selection_comparison_pairs_v2.tsv (pipeline-ready)
"""

import sys, os
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import json

ROOT = Path.cwd()
for _ in range(5):
    if (ROOT / "notebooks" / "paper_data_config.py").exists():
        break
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, "/Users/nicolaslynn/Documents/phd/ParseTCGA")

from paper_data_config import EPISTASIS_PAPER_ROOT
from parsegenomes.ids import normalize_mutation_id

DATA_DIR = EPISTASIS_PAPER_ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
ANNOT_DIR = DATA_DIR / "annotations"
OKGP_DB = "/Users/nicolaslynn/Downloads/likely_garbage/okgp_query/1kg_mutations.duckdb"
TCGA_PARQUET = SOURCE_DIR / "tcga_all.parquet"
OUTPUT_DIR = ROOT / "notebooks" / "dirty" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DIST_BP = 500  # expanded from 100

# Cancer gene sets
oncokb = pd.read_csv(ANNOT_DIR / "cancerGeneList.tsv", sep="\t")
census = pd.read_csv(ANNOT_DIR / "census_all_genes.csv")
CANCER_GENES = (
    set(oncokb["Hugo Symbol"])
    | set(census[census["Role in Cancer"].str.contains("oncogene|TSG", case=False, na=False)]["Gene Symbol"].str.strip())
)

# Strand lookup
STRAND_CACHE = SOURCE_DIR / "gene_strand_cache.json"
strand_lookup = {}
if STRAND_CACHE.exists():
    with open(STRAND_CACHE) as f:
        strand_lookup = json.load(f)


def is_snv(ref, alt):
    return (len(str(ref)) == 1 and len(str(alt)) == 1
            and str(ref) in "ACGT" and str(alt) in "ACGT"
            and str(ref) != str(alt))


def make_eid(gene, chrom, pos1, ref1, alt1, pos2, ref2, alt2):
    chrom = str(chrom).replace("chr", "")
    strand = strand_lookup.get(gene, "P")
    if pos1 <= pos2:
        return f"{gene}:{chrom}:{pos1}:{ref1}:{alt1}:{strand}|{gene}:{chrom}:{pos2}:{ref2}:{alt2}:{strand}"
    else:
        return f"{gene}:{chrom}:{pos2}:{ref2}:{alt2}:{strand}|{gene}:{chrom}:{pos1}:{ref1}:{alt1}:{strand}"


def strip_strand(mut):
    if mut.endswith(":P") or mut.endswith(":N"):
        return mut[:-2]
    return mut


# Distance bins for matching
DIST_BINS = [1, 3, 6, 11, 21, 51, 101, 201, 501]
DIST_LABELS = ["1-2", "3-5", "6-10", "11-20", "21-50", "51-100", "101-200", "201-500"]


# =========================================================================
# STEP 1: Parse TCGA high-selection (cond_prob >= 0.80) at 500bp
# =========================================================================
print("=" * 80)
print("STEP 1: TCGA HIGH-SELECTION (cond_prob >= 0.80, ≤500bp, SNV only)")
print("=" * 80)

MIN_DEPTH = 15; ALT_MIN = 15; MIN_VAF = 0.10
EXCLUDE_TOP_BURDEN_GENES = 20
HIGH_COND_MIN = 0.80; MIN_CASES_BOTH = 3

con_tcga = duckdb.connect()
raw = con_tcga.execute("""
    SELECT case_id, Gene_name, Chromosome, Start_Position,
        (Gene_name || ':' || REPLACE(Chromosome, 'chr', '') || ':'
         || CAST(Start_Position AS VARCHAR) || ':'
         || UPPER(COALESCE(Reference_Allele, '')) || ':'
         || UPPER(COALESCE(Tumor_Seq_Allele2, ''))) AS mutation_id,
        UPPER(COALESCE(Reference_Allele, '')) AS ref_allele,
        UPPER(COALESCE(Tumor_Seq_Allele2, '')) AS alt_allele,
        t_depth, t_alt_count
    FROM read_parquet(?)
    WHERE t_depth >= ? AND COALESCE(t_alt_count, 0) >= ?
""", [str(TCGA_PARQUET), MIN_DEPTH, ALT_MIN]).fetchdf()

raw["pos"] = raw["Start_Position"].astype(int)
raw["vaf"] = (raw["t_alt_count"] / raw["t_depth"].clip(lower=1)).astype(float)
raw["chrom"] = raw["Chromosome"].astype(str).str.replace("chr", "", regex=False)

# Exclude top burden genes + filter SNV
gene_cts = raw["Gene_name"].value_counts()
excluded = set(gene_cts.head(EXCLUDE_TOP_BURDEN_GENES).index)
raw = raw[~raw["Gene_name"].isin(excluded)].copy()
raw_snv = raw[raw.apply(lambda r: is_snv(r["ref_allele"], r["alt_allele"]), axis=1)].copy()
print(f"TCGA SNV mutations (after filters): {len(raw_snv):,}")

# Find all pairs within 500bp, same gene
print("Finding pairs (≤500bp)...")
pairs_high = []
pairs_low = []

for (case_id, chrom), grp in raw_snv.groupby(["case_id", "chrom"]):
    grp = grp.sort_values("pos").reset_index(drop=True)
    if len(grp) < 2:
        continue
    pos = grp["pos"].values
    vafs = grp["vaf"].values
    genes = grp["Gene_name"].values
    mut_ids = grp["mutation_id"].values
    refs = grp["ref_allele"].values
    alts = grp["alt_allele"].values

    for i in range(len(grp)):
        hi = np.searchsorted(pos, pos[i] + MAX_DIST_BP, side="right")
        for j in range(i + 1, hi):
            if pos[j] - pos[i] <= 0 or genes[i] != genes[j]:
                continue
            if vafs[i] < MIN_VAF or vafs[j] < MIN_VAF:
                continue
            rec = {
                "case_id": case_id, "gene": str(genes[i]).strip(),
                "chrom": chrom,
                "pos1": int(pos[i]), "pos2": int(pos[j]),
                "ref1": refs[i], "alt1": alts[i],
                "ref2": refs[j], "alt2": alts[j],
                "mut1_id": mut_ids[i], "mut2_id": mut_ids[j],
                "distance_bp": int(pos[j] - pos[i]),
            }
            pairs_high.append(rec)
            pairs_low.append(rec)  # low-selection will filter by cond_prob later

pairs_all = pd.DataFrame(pairs_high)
pairs_all = pairs_all.drop_duplicates(subset=["mut1_id", "mut2_id", "case_id"])
print(f"Total candidate pairs (≤500bp): {len(pairs_all):,}")

# Co-occurrence stats
all_muts_tcga = list(set(pairs_all["mut1_id"]) | set(pairs_all["mut2_id"]))
print(f"Looking up {len(all_muts_tcga):,} mutations for co-occurrence...")
lookup_parts = []
for i in range(0, len(all_muts_tcga), 50000):
    chunk = all_muts_tcga[i:i+50000]
    lookup_parts.append(con_tcga.execute("""
        SELECT
            (Gene_name || ':' || REPLACE(Chromosome, 'chr', '') || ':'
             || CAST(Start_Position AS VARCHAR) || ':'
             || UPPER(COALESCE(Reference_Allele, '')) || ':'
             || UPPER(COALESCE(Tumor_Seq_Allele2, ''))) AS mutation_id,
            case_id
        FROM read_parquet(?)
        WHERE (Gene_name || ':' || REPLACE(Chromosome, 'chr', '') || ':'
               || CAST(Start_Position AS VARCHAR) || ':'
               || UPPER(COALESCE(Reference_Allele, '')) || ':'
               || UPPER(COALESCE(Tumor_Seq_Allele2, ''))) IN (SELECT unnest(?))
    """, [str(TCGA_PARQUET), chunk]).fetchdf())
con_tcga.close()

lookup_df = pd.concat(lookup_parts, ignore_index=True)
global_m2c = lookup_df.groupby("mutation_id")["case_id"].apply(set).to_dict()

# Compute cond_prob per unique pair
epi_info = pairs_all.groupby(["mut1_id", "mut2_id"], as_index=False).agg(
    gene=("gene", "first"), chrom=("chrom", "first"),
    pos1=("pos1", "first"), pos2=("pos2", "first"),
    ref1=("ref1", "first"), alt1=("alt1", "first"),
    ref2=("ref2", "first"), alt2=("alt2", "first"),
    distance_bp=("distance_bp", "first"),
)

pair_stats = []
for _, r in epi_info.iterrows():
    s1 = global_m2c.get(r["mut1_id"], set())
    s2 = global_m2c.get(r["mut2_id"], set())
    both = s1 & s2
    n1, n2, nb = len(s1), len(s2), len(both)
    cond = max(nb / max(n1, 1), nb / max(n2, 1))
    pair_stats.append({**r.to_dict(), "n_cases_both": nb, "cond_prob": round(cond, 4)})

df_all_pairs = pd.DataFrame(pair_stats)

# Split into high and low selection
set1 = df_all_pairs[(df_all_pairs["cond_prob"] >= HIGH_COND_MIN) &
                     (df_all_pairs["n_cases_both"] >= MIN_CASES_BOTH)].copy()
set3 = df_all_pairs[(df_all_pairs["cond_prob"] >= 0.20) &
                     (df_all_pairs["cond_prob"] <= 0.50) &
                     (df_all_pairs["n_cases_both"] >= 2)].copy()

# Add epistasis_ids
set1["epistasis_id"] = set1.apply(
    lambda r: make_eid(r["gene"], r["chrom"], r["pos1"], r["ref1"], r["alt1"],
                       r["pos2"], r["ref2"], r["alt2"]), axis=1)
set3["epistasis_id"] = set3.apply(
    lambda r: make_eid(r["gene"], r["chrom"], r["pos1"], r["ref1"], r["alt1"],
                       r["pos2"], r["ref2"], r["alt2"]), axis=1)

set1 = set1.drop_duplicates(subset=["epistasis_id"])
set3 = set3.drop_duplicates(subset=["epistasis_id"])

print(f"\nSet 1 (high-selection, cond≥0.80): {len(set1)} pairs")
print(f"Set 3 (low-selection, cond 0.20-0.50): {len(set3)} pairs")


# =========================================================================
# STEP 2: Find 1kGP matched doubles (same anchor, germline partner, ≤500bp)
# =========================================================================
print(f"\n{'=' * 80}")
print("STEP 2: 1kGP MATCHED DOUBLES (≤500bp)")
print("=" * 80)

# Find which Set1 mutations exist in 1kGP
set1_muts = set()
set1_pair_muts = {}
for _, r in set1.iterrows():
    parts = r["epistasis_id"].split("|")
    m1, m2 = strip_strand(parts[0]), strip_strand(parts[1])
    set1_muts.add(m1)
    set1_muts.add(m2)
    set1_pair_muts[r["epistasis_id"]] = (m1, m2)

mut_list = sorted(set1_muts)
con = duckdb.connect(OKGP_DB, read_only=True)

found_rare = con.execute("""
    SELECT mutation_id, len(carriers) as n_carriers, carriers
    FROM mutations WHERE mutation_id IN (SELECT unnest(?))
""", [mut_list]).fetchdf()

found_common = con.execute("""
    SELECT mutation_id, n_carriers
    FROM common_mutations WHERE mutation_id IN (SELECT unnest(?))
""", [mut_list]).fetchdf()

found_all = set(found_rare["mutation_id"]) | set(found_common["mutation_id"])
carrier_map = {}
for _, row in found_rare.iterrows():
    carrier_map[row["mutation_id"]] = set(row["carriers"]) if row["carriers"] is not None else set()

print(f"Set1 mutations found in 1kGP: {len(found_all)} / {len(set1_muts)}")

# For each anchor, find nearby 1kGP partners
anchors_with_carriers = {m: c for m, c in carrier_map.items()
                         if m in set1_muts and len(c) >= 1}
print(f"Anchors with 1kGP carriers: {len(anchors_with_carriers)}")

set2_rows = []
processed_genes = {}  # gene:chrom -> already queried

for anchor_mut, anchor_carriers in sorted(anchors_with_carriers.items(),
                                           key=lambda x: -len(x[1])):
    gene = anchor_mut.split(":")[0]
    chrom = anchor_mut.split(":")[1]
    anchor_pos = int(anchor_mut.split(":")[2])
    gene_key = f"{gene}:{chrom}"

    # Query all mutations in this gene (cache per gene to avoid repeat queries)
    if gene_key not in processed_genes:
        pattern = f"{gene}:{chrom}:%"
        gene_muts = con.execute("""
            SELECT mutation_id, len(carriers) as n_carriers, carriers
            FROM mutations WHERE mutation_id LIKE ?
        """, [pattern]).fetchdf()
        processed_genes[gene_key] = gene_muts
    else:
        gene_muts = processed_genes[gene_key]

    if len(gene_muts) == 0:
        continue

    gene_muts_copy = gene_muts.copy()
    gene_muts_copy["pos"] = gene_muts_copy["mutation_id"].map(lambda m: int(m.split(":")[2]))
    gene_muts_copy["dist"] = (gene_muts_copy["pos"] - anchor_pos).abs()
    nearby = gene_muts_copy[(gene_muts_copy["dist"] > 0) & (gene_muts_copy["dist"] <= MAX_DIST_BP)]

    # Filter to SNV partners
    nearby = nearby.copy()
    nearby["ref"] = nearby["mutation_id"].map(lambda m: m.split(":")[3])
    nearby["alt"] = nearby["mutation_id"].map(lambda m: m.split(":")[4])
    nearby = nearby[nearby.apply(lambda r: is_snv(r["ref"], r["alt"]), axis=1)]

    anchor_ref = anchor_mut.split(":")[3]
    anchor_alt = anchor_mut.split(":")[4]

    for _, nr in nearby.iterrows():
        partner_carriers = set(nr["carriers"]) if nr["carriers"] is not None else set()
        n_cooccur = len(anchor_carriers & partner_carriers)

        partner_mut = nr["mutation_id"]
        partner_pos = int(partner_mut.split(":")[2])
        partner_ref = partner_mut.split(":")[3]
        partner_alt = partner_mut.split(":")[4]

        eid = make_eid(gene, chrom, anchor_pos, anchor_ref, anchor_alt,
                       partner_pos, partner_ref, partner_alt)

        set2_rows.append({
            "epistasis_id": eid,
            "gene": gene, "chr": chrom,
            "pos1": min(anchor_pos, partner_pos),
            "pos2": max(anchor_pos, partner_pos),
            "distance_bp": abs(partner_pos - anchor_pos),
            "anchor_mut": anchor_mut,
            "partner_mut": partner_mut,
            "n_anchor_carriers": len(anchor_carriers),
            "n_partner_carriers": nr["n_carriers"],
            "n_cooccur": n_cooccur,
            "is_cancer_gene": gene in CANCER_GENES,
        })

con.close()

set2 = pd.DataFrame(set2_rows).drop_duplicates(subset=["epistasis_id"])
print(f"Set 2 (1kGP matched): {len(set2)} pairs from {set2['anchor_mut'].nunique()} anchors")


# =========================================================================
# STEP 3: Distance matching across all 3 sets
# =========================================================================
print(f"\n{'=' * 80}")
print("STEP 3: DISTANCE MATCHING")
print("=" * 80)

set1["dist_bin"] = pd.cut(set1["distance_bp"], bins=DIST_BINS, labels=DIST_LABELS, right=False)
set2["dist_bin"] = pd.cut(set2["distance_bp"], bins=DIST_BINS, labels=DIST_LABELS, right=False)
set3["dist_bin"] = pd.cut(set3["distance_bp"], bins=DIST_BINS, labels=DIST_LABELS, right=False)

print("\nBefore matching:")
for name, df in [("Set1 high-sel", set1), ("Set2 OKGP", set2), ("Set3 low-sel", set3)]:
    fracs = df["dist_bin"].value_counts(normalize=True).reindex(DIST_LABELS).fillna(0)
    print(f"  {name:15s} (n={len(df):6d}): " + "  ".join(f"{l}:{f:.2f}" for l, f in fracs.items()))

# Find the bottleneck count per bin (min across all 3 sets)
rng = np.random.RandomState(42)
matched_sets = {"set1": [], "set2": [], "set3": []}

print("\nPer-bin matching:")
for label in DIST_LABELS:
    n1 = (set1["dist_bin"] == label).sum()
    n2 = (set2["dist_bin"] == label).sum()
    n3 = (set3["dist_bin"] == label).sum()
    n_match = min(n1, n2, n3)

    if n_match < 10:
        print(f"  {label:>8s}: n1={n1:5d} n2={n2:5d} n3={n3:5d} -> SKIP (< 10)")
        continue

    print(f"  {label:>8s}: n1={n1:5d} n2={n2:5d} n3={n3:5d} -> sample {n_match} each")
    for name, df in [("set1", set1), ("set2", set2), ("set3", set3)]:
        pool = df[df["dist_bin"] == label]
        sampled = pool.sample(n_match, random_state=42) if len(pool) > n_match else pool
        matched_sets[name].append(sampled)

set1_matched = pd.concat(matched_sets["set1"]).reset_index(drop=True)
set2_matched = pd.concat(matched_sets["set2"]).reset_index(drop=True)
set3_matched = pd.concat(matched_sets["set3"]).reset_index(drop=True)

print(f"\nAfter matching:")
for name, df in [("Set1 high-sel", set1_matched), ("Set2 OKGP", set2_matched), ("Set3 low-sel", set3_matched)]:
    fracs = df["dist_bin"].value_counts(normalize=True).reindex(DIST_LABELS).fillna(0)
    print(f"  {name:15s} (n={len(df):6d}): " + "  ".join(f"{l}:{f:.2f}" for l, f in fracs.items()))


# =========================================================================
# STEP 4: Save outputs
# =========================================================================
print(f"\n{'=' * 80}")
print("SAVING")
print("=" * 80)

# Full sets (unmatched — for flexibility)
set1["source"] = "tcga_high_selection"
set1["label"] = "positive"
set2["source"] = "okgp_matched_doubles"
set2["label"] = "null"
set3["source"] = "tcga_low_selection"
set3["label"] = "negative"

set1.to_csv(OUTPUT_DIR / "set1_tcga_high_selection_v2.tsv", sep="\t", index=False)
set2.to_csv(OUTPUT_DIR / "set2_okgp_matched_doubles_v2.tsv", sep="\t", index=False)
set3.to_csv(OUTPUT_DIR / "set3_tcga_low_selection_v2.tsv", sep="\t", index=False)

# Distance-matched sets
set1_matched["source"] = "tcga_high_selection"
set1_matched["label"] = "positive"
set2_matched["source"] = "okgp_matched_doubles"
set2_matched["label"] = "null"
set3_matched["source"] = "tcga_low_selection"
set3_matched["label"] = "negative"

set1_matched.to_csv(OUTPUT_DIR / "set1_matched_v2.tsv", sep="\t", index=False)
set2_matched.to_csv(OUTPUT_DIR / "set2_matched_v2.tsv", sep="\t", index=False)
set3_matched.to_csv(OUTPUT_DIR / "set3_matched_v2.tsv", sep="\t", index=False)

# Pipeline-ready file: NEW pairs only (Set 2 + Set 3 + Set 1 pairs not in existing tcga_doubles)
existing_eids = set(pd.read_csv(
    ROOT / "notebooks" / "processing" / "data" / "all_pairs_combined.tsv",
    sep="\t", usecols=["epistasis_id"]
)["epistasis_id"])

pipeline_rows = []
for df in [set1, set2, set3]:
    for _, r in df.iterrows():
        if r["epistasis_id"] not in existing_eids:
            pipeline_rows.append({
                "source": r["source"],
                "label": r["label"],
                "epistasis_id": r["epistasis_id"],
            })

pipeline_df = pd.DataFrame(pipeline_rows).drop_duplicates(subset=["epistasis_id"])
pipeline_path = OUTPUT_DIR / "selection_comparison_pairs_v2.tsv"
pipeline_df.to_csv(pipeline_path, sep="\t", index=False)

print(f"\nPipeline-ready: {pipeline_path}")
print(f"  Total NEW pairs to embed: {len(pipeline_df)}")
print(f"  By source: {pipeline_df['source'].value_counts().to_dict()}")
print(f"\nFull sets (unmatched):")
print(f"  Set 1: {len(set1)} | Set 2: {len(set2)} | Set 3: {len(set3)}")
print(f"Matched sets:")
print(f"  Set 1: {len(set1_matched)} | Set 2: {len(set2_matched)} | Set 3: {len(set3_matched)}")
