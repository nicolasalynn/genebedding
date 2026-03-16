"""
Generate three comparison sets for the selection pressure experiment.

Set 1: tcga_high_selection — TCGA SNV doubles with cond_prob >= 0.80
        (already exists, but we re-extract the subset where both mutations
         are found in 1kGP, for matched comparison)

Set 2: okgp_matched_doubles — For each TCGA anchor mutation found in 1kGP,
        find its actual 1kGP partner mutations within 100bp in the same gene.
        These are real germline double variants at the same loci.

Set 3: tcga_low_selection — TCGA SNV doubles with cond_prob 0.20-0.50.
        These co-occur in some patients but NOT under strong selection.
        Re-parsed from raw TCGA parquet with relaxed filters.

Output: single TSV file compatible with all_pairs_combined.tsv (source, label, epistasis_id)
        plus a metadata TSV with full details for analysis.
"""

import sys, os
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

ROOT = Path.cwd()
for _ in range(5):
    if (ROOT / "notebooks" / "paper_data_config.py").exists():
        break
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "scripts" / "data_generation"))
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

# Cancer gene sets for labeling
oncokb = pd.read_csv(ANNOT_DIR / "cancerGeneList.tsv", sep="\t")
census = pd.read_csv(ANNOT_DIR / "census_all_genes.csv")
CANCER_GENES = (
    set(oncokb["Hugo Symbol"])
    | set(census[census["Role in Cancer"].str.contains("oncogene|TSG", case=False, na=False)]["Gene Symbol"].str.strip())
)

# Strand lookup (cached)
import json
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
    """Build epistasis_id with strand, pos1 < pos2."""
    chrom = str(chrom).replace("chr", "")
    strand = strand_lookup.get(gene, "P")
    if pos1 <= pos2:
        return f"{gene}:{chrom}:{pos1}:{ref1}:{alt1}:{strand}|{gene}:{chrom}:{pos2}:{ref2}:{alt2}:{strand}"
    else:
        return f"{gene}:{chrom}:{pos2}:{ref2}:{alt2}:{strand}|{gene}:{chrom}:{pos1}:{ref1}:{alt1}:{strand}"


# =========================================================================
# SET 1: TCGA high-selection pairs (existing, filter to both-in-1kGP)
# =========================================================================
print("=" * 80)
print("SET 1: TCGA high-selection (cond_prob >= 0.80, SNV only)")
print("=" * 80)

tcga_existing = pd.read_csv(DATA_DIR / "tcga_doubles_pairs.tsv", sep="\t")
tcga_snv = tcga_existing[
    tcga_existing.apply(lambda r: is_snv(r["ref1"], r["alt1"]) and is_snv(r["ref2"], r["alt2"]), axis=1)
].copy()
print(f"Existing TCGA SNV pairs: {len(tcga_snv)}")

# Find which mutations exist in 1kGP
def strip_strand(mut):
    if mut.endswith(":P") or mut.endswith(":N"):
        return mut[:-2]
    return mut

all_tcga_muts = set()
tcga_pair_muts = {}
for _, r in tcga_snv.iterrows():
    parts = r["epistasis_id"].split("|")
    m1, m2 = strip_strand(parts[0]), strip_strand(parts[1])
    all_tcga_muts.add(m1)
    all_tcga_muts.add(m2)
    tcga_pair_muts[r["epistasis_id"]] = (m1, m2)

mut_list = sorted(all_tcga_muts)
con = duckdb.connect(OKGP_DB, read_only=True)

# Batch query 1kGP
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

print(f"TCGA mutations found in 1kGP: {len(found_all)} / {len(all_tcga_muts)}")

# Tag pairs
tcga_snv["mut1"] = tcga_snv["epistasis_id"].map(lambda eid: tcga_pair_muts[eid][0])
tcga_snv["mut2"] = tcga_snv["epistasis_id"].map(lambda eid: tcga_pair_muts[eid][1])
tcga_snv["mut1_in_1kgp"] = tcga_snv["mut1"].isin(found_all)
tcga_snv["mut2_in_1kgp"] = tcga_snv["mut2"].isin(found_all)
tcga_snv["both_in_1kgp"] = tcga_snv["mut1_in_1kgp"] & tcga_snv["mut2_in_1kgp"]

# Check co-occurrence
def check_cooccur(m1, m2):
    c1 = carrier_map.get(m1, set())
    c2 = carrier_map.get(m2, set())
    return len(c1 & c2)

tcga_snv["n_cooccur_1kgp"] = tcga_snv.apply(
    lambda r: check_cooccur(r["mut1"], r["mut2"]), axis=1)

set1 = tcga_snv.copy()
set1["source"] = "tcga_high_selection"
set1["label"] = "positive"
print(f"Set 1: {len(set1)} pairs")
print(f"  Both in 1kGP: {set1['both_in_1kgp'].sum()}")
print(f"  Co-occur in 1kGP: {(set1['n_cooccur_1kgp'] > 0).sum()}")


# =========================================================================
# SET 2: 1kGP matched doubles
# =========================================================================
print(f"\n{'=' * 80}")
print("SET 2: 1kGP matched doubles (same anchor mutation, germline partner)")
print("=" * 80)

# For each anchor mutation that's in 1kGP with carriers, find nearby partners
anchors_with_carriers = {m: c for m, c in carrier_map.items()
                         if m in all_tcga_muts and len(c) >= 1}
print(f"Anchor mutations with 1kGP carriers: {len(anchors_with_carriers)}")

set2_rows = []
processed_anchors = set()

for anchor_mut, anchor_carriers in sorted(anchors_with_carriers.items(),
                                           key=lambda x: -len(x[1])):
    gene = anchor_mut.split(":")[0]
    chrom = anchor_mut.split(":")[1]
    anchor_pos = int(anchor_mut.split(":")[2])

    if anchor_mut in processed_anchors:
        continue
    processed_anchors.add(anchor_mut)

    # Find nearby mutations in same gene in 1kGP
    pattern = f"{gene}:{chrom}:%"
    nearby = con.execute("""
        SELECT mutation_id, len(carriers) as n_carriers, carriers
        FROM mutations
        WHERE mutation_id LIKE ?
    """, [pattern]).fetchdf()

    if len(nearby) == 0:
        continue

    nearby["pos"] = nearby["mutation_id"].map(lambda m: int(m.split(":")[2]))
    nearby["dist"] = (nearby["pos"] - anchor_pos).abs()
    nearby = nearby[(nearby["dist"] > 0) & (nearby["dist"] <= 100)]

    # Filter to SNV partners
    nearby["ref"] = nearby["mutation_id"].map(lambda m: m.split(":")[3])
    nearby["alt"] = nearby["mutation_id"].map(lambda m: m.split(":")[4])
    nearby = nearby[nearby.apply(lambda r: is_snv(r["ref"], r["alt"]), axis=1)]

    for _, nr in nearby.iterrows():
        partner_carriers = set(nr["carriers"]) if nr["carriers"] is not None else set()
        n_cooccur = len(anchor_carriers & partner_carriers)

        partner_mut = nr["mutation_id"]
        partner_pos = int(partner_mut.split(":")[2])
        partner_ref = partner_mut.split(":")[3]
        partner_alt = partner_mut.split(":")[4]
        anchor_ref = anchor_mut.split(":")[3]
        anchor_alt = anchor_mut.split(":")[4]

        eid = make_eid(gene, chrom, anchor_pos, anchor_ref, anchor_alt,
                       partner_pos, partner_ref, partner_alt)

        set2_rows.append({
            "source": "okgp_matched_doubles",
            "label": "null",
            "epistasis_id": eid,
            "gene": gene,
            "chr": chrom,
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

set2 = pd.DataFrame(set2_rows).drop_duplicates(subset=["epistasis_id"])
print(f"Set 2: {len(set2)} 1kGP matched double pairs")
print(f"  From {set2['anchor_mut'].nunique()} anchor mutations")
print(f"  Co-occurring (n>0): {(set2['n_cooccur'] > 0).sum()}")
print(f"  In cancer genes: {set2['is_cancer_gene'].sum()}")
print(f"  Distance range: {set2['distance_bp'].min()}-{set2['distance_bp'].max()}bp")


# =========================================================================
# SET 3: TCGA low-selection (cond_prob 0.20-0.50)
# =========================================================================
print(f"\n{'=' * 80}")
print("SET 3: TCGA low-selection (cond_prob 0.20-0.50)")
print("=" * 80)

if not TCGA_PARQUET.exists():
    print(f"WARNING: TCGA parquet not found at {TCGA_PARQUET}")
    print("Set 3 requires the raw TCGA parquet. Skipping.")
    set3 = pd.DataFrame()
else:
    MIN_DEPTH = 15
    ALT_MIN = 15
    MIN_VAF = 0.10
    MAX_DIST_BP = 100
    LOW_COND_PROB_MIN = 0.20
    LOW_COND_PROB_MAX = 0.50
    MIN_CASES_BOTH = 2
    EXCLUDE_TOP_BURDEN_GENES = 20

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
    con_tcga.close()

    raw["pos"] = raw["Start_Position"].astype(int)
    raw["vaf"] = (raw["t_alt_count"] / raw["t_depth"].clip(lower=1)).astype(float)
    raw["chrom"] = raw["Chromosome"].astype(str).str.replace("chr", "", regex=False)
    print(f"Raw TCGA mutations: {len(raw):,}")

    # Exclude top burden genes
    gene_cts = raw["Gene_name"].value_counts()
    excluded = set(gene_cts.head(EXCLUDE_TOP_BURDEN_GENES).index)
    raw = raw[~raw["Gene_name"].isin(excluded)].copy()

    # Filter to SNV only
    raw = raw[raw.apply(lambda r: is_snv(r["ref_allele"], r["alt_allele"]), axis=1)].copy()
    print(f"After SNV filter + burden exclusion: {len(raw):,}")

    # Find pairs
    pairs = []
    for (case_id, chrom), grp in raw.groupby(["case_id", "chrom"]):
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
                if pos[j] - pos[i] <= 0:
                    continue
                if vafs[i] < MIN_VAF or vafs[j] < MIN_VAF:
                    continue
                if genes[i] != genes[j]:
                    continue  # same gene only
                pairs.append({
                    "case_id": case_id,
                    "gene": str(genes[i]).strip(),
                    "chrom": chrom,
                    "pos1": int(pos[i]), "pos2": int(pos[j]),
                    "ref1": refs[i], "alt1": alts[i],
                    "ref2": refs[j], "alt2": alts[j],
                    "mut1_id": mut_ids[i], "mut2_id": mut_ids[j],
                    "distance_bp": int(pos[j] - pos[i]),
                })

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df) == 0:
        print("No low-selection pairs found!")
        set3 = pd.DataFrame()
    else:
        pairs_df = pairs_df.drop_duplicates(subset=["mut1_id", "mut2_id", "case_id"])

        # Co-occurrence stats from full parquet
        con_tcga2 = duckdb.connect()
        all_muts_low = list(set(pairs_df["mut1_id"]) | set(pairs_df["mut2_id"]))
        # Batch lookup
        lookup_parts = []
        for i in range(0, len(all_muts_low), 50000):
            chunk = all_muts_low[i:i+50000]
            lookup_parts.append(con_tcga2.execute("""
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
        con_tcga2.close()
        lookup_df = pd.concat(lookup_parts, ignore_index=True)
        global_m2c = lookup_df.groupby("mutation_id")["case_id"].apply(set).to_dict()

        # Compute co-occurrence per unique pair
        epi_info = pairs_df.groupby(["mut1_id", "mut2_id"], as_index=False).agg(
            gene=("gene", "first"),
            chrom=("chrom", "first"),
            pos1=("pos1", "first"), pos2=("pos2", "first"),
            ref1=("ref1", "first"), alt1=("alt1", "first"),
            ref2=("ref2", "first"), alt2=("alt2", "first"),
            distance_bp=("distance_bp", "first"),
        )

        set3_rows = []
        for _, r in epi_info.iterrows():
            s1 = global_m2c.get(r["mut1_id"], set())
            s2 = global_m2c.get(r["mut2_id"], set())
            both = s1 & s2
            n1, n2, nb = len(s1), len(s2), len(both)
            cond = max(nb / max(n1, 1), nb / max(n2, 1))

            if cond < LOW_COND_PROB_MIN or cond > LOW_COND_PROB_MAX:
                continue
            if nb < MIN_CASES_BOTH:
                continue

            eid = make_eid(r["gene"], r["chrom"], r["pos1"], r["ref1"], r["alt1"],
                           r["pos2"], r["ref2"], r["alt2"])
            set3_rows.append({
                "source": "tcga_low_selection",
                "label": "negative",
                "epistasis_id": eid,
                "gene": r["gene"],
                "chr": r["chrom"],
                "pos1": r["pos1"], "pos2": r["pos2"],
                "ref1": r["ref1"], "alt1": r["alt1"],
                "ref2": r["ref2"], "alt2": r["alt2"],
                "distance_bp": r["distance_bp"],
                "n_cases_both": nb,
                "cond_prob": round(cond, 4),
                "is_cancer_gene": r["gene"] in CANCER_GENES,
            })

        set3 = pd.DataFrame(set3_rows).drop_duplicates(subset=["epistasis_id"])
        print(f"Set 3: {len(set3)} low-selection TCGA pairs")
        print(f"  Cond_prob range: {set3['cond_prob'].min():.2f}-{set3['cond_prob'].max():.2f}")
        print(f"  Distance range: {set3['distance_bp'].min()}-{set3['distance_bp'].max()}bp")
        print(f"  In cancer genes: {set3['is_cancer_gene'].sum()}")

con.close()


# =========================================================================
# Save outputs
# =========================================================================
print(f"\n{'=' * 80}")
print("SAVING")
print("=" * 80)

# Pipeline-ready file (source, label, epistasis_id only)
pipeline_rows = []

# Set 1: already in pipeline as tcga_doubles, but save the annotated version
set1_out = set1[["epistasis_id", "gene", "chr", "pos1", "pos2", "ref1", "alt1",
                  "ref2", "alt2", "distance_bp", "n_cases_both", "cond_prob_dependent",
                  "both_in_1kgp", "n_cooccur_1kgp"]].copy()
set1_out["source"] = "tcga_high_selection"
set1_out.to_csv(OUTPUT_DIR / "set1_tcga_high_selection.tsv", sep="\t", index=False)
# Don't add set1 to pipeline — it's already there as tcga_doubles

# Set 2: new pairs, need embedding
for _, r in set2.iterrows():
    pipeline_rows.append({"source": r["source"], "label": r["label"], "epistasis_id": r["epistasis_id"]})
set2.to_csv(OUTPUT_DIR / "set2_okgp_matched_doubles.tsv", sep="\t", index=False)

# Set 3: new pairs, need embedding
if len(set3) > 0:
    for _, r in set3.iterrows():
        pipeline_rows.append({"source": r["source"], "label": r["label"], "epistasis_id": r["epistasis_id"]})
    set3.to_csv(OUTPUT_DIR / "set3_tcga_low_selection.tsv", sep="\t", index=False)

# Combined pipeline file
pipeline_df = pd.DataFrame(pipeline_rows)
pipeline_path = OUTPUT_DIR / "selection_comparison_pairs.tsv"
pipeline_df.to_csv(pipeline_path, sep="\t", index=False)

print(f"\nPipeline-ready file: {pipeline_path}")
print(f"  Total new pairs to embed: {len(pipeline_df)}")
print(f"  Sources: {pipeline_df['source'].value_counts().to_dict()}")

# Summary
print(f"\nMetadata files:")
print(f"  {OUTPUT_DIR / 'set1_tcga_high_selection.tsv'}: {len(set1_out)} rows")
print(f"  {OUTPUT_DIR / 'set2_okgp_matched_doubles.tsv'}: {len(set2)} rows")
if len(set3) > 0:
    print(f"  {OUTPUT_DIR / 'set3_tcga_low_selection.tsv'}: {len(set3)} rows")

print(f"\nTo embed, append {pipeline_path} to all_pairs_combined.tsv")
print(f"or add these sources to pipeline_config.py and run process_epistasis.")
