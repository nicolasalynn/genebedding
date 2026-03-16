# Expression Doubles — Data Generation Guide

Complete guide to reproducing the variant pair datasets used for benchmarking DNA language models on epistasis prediction.

## Quick Start

```bash
cd /path/to/genebeddings

# Set data root (default: ~/data/epistasis_paper)
export EPISTASIS_PAPER_ROOT=~/data/epistasis_paper

# Step 1: Run the main pipeline (parses all datasets, ~10 min)
python -m scripts.data_generation.aggregate

# Step 2: Add MAF annotations (queries Ensembl VEP, ~3 hrs first run, instant after)
python -m scripts.data_generation.add_maf
```

## Data Layout

All data lives under `EPISTASIS_PAPER_ROOT` (env var, default `~/data/epistasis_paper`):

```
EPISTASIS_PAPER_ROOT/
  data/
    source/                    # Raw input files (Excel, CSV, TAR, parquet)
    all_pairs_combined.tsv     # Main output: all parsed variant pairs
    annotations/               # Cancer gene lists for TCGA analysis
    trna_figures/              # Pre-generated tRNA images
  combined_parquets/           # Model output parquets (per-model, all sources)
    new_embeddings/
    parquets_epi/
  embeddings/                  # Per-source DBs (pipeline output)
  figures/                     # Analysis notebook output
```

## Prerequisites

### Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas openpyxl pyliftover seqmat requests matplotlib numpy
```

Jupyter kernel name: `expression_doubles`

### Required Input Files

All raw data goes in `data/`. The pipeline expects:

| File | Dataset | Source |
|------|---------|--------|
| `pnas.2007049118.sd04.xlsx` | Uebbing MPRA | [PNAS SI](https://www.pnas.org/doi/10.1073/pnas.2007049118) |
| `suppl_table_S1.xlsx`, `suppl_table_S3.xlsx` | Yang 2016 evQTL | Yang et al. 2016 supplementary |
| `elife-01381-supp1-v2.xlsx` | Brown 2014 v-eQTL | [eLife](https://elifesciences.org/articles/01381) |
| `GTEx_Analysis_v8_eQTL_independent.tar` | GTEx null tier 1 | [GTEx Portal](https://gtexportal.org) |
| `GTEx_Analysis_v8_eQTL.tar` | GTEx null tier 2 | [GTEx Portal](https://gtexportal.org) |
| `41467_2016_BFncomms11558_MOESM968_ESM.xlsx` | FAS exon 3 | Nature Comms SI |
| `41467_2018_5748_MOESM4_ESM.xlsx`, `MOESM6` | MST1R splicing | Nature Comms SI |
| `okgp_epistasis.csv` | OKGP chr12 | OKGP project |
| `correlated_variants_27K.csv` | Correlated eQTL (LD null) | Custom generation |
| `Supplemental_Table_S2.xlsx` | Ke SL6 | Ke et al. supplementary |

External dependency (optional):
- `data/source/tcga_all.parquet` — TCGA somatic mutations (from ParseTCGA)

### External Services

- **Ensembl REST API** — gene strand lookups, rsID allele lookups, VEP MAF queries
- **hg38 FASTA** — accessed via `seqmat` for reference allele validation
- **pyliftover** — hg19 → hg38 coordinate conversion

## Pipeline Architecture

```
scripts/
├── aggregate.py          # Main orchestrator (run this)
├── add_maf.py            # Standalone MAF enrichment (run after aggregate)
├── common.py             # Shared utilities, paths, API helpers
├── parse_uebbing_mpra.py # Uebbing MPRA enhancer pairs
├── parse_yang_evqtl.py   # Yang 2016 evQTL
├── parse_brown_veqtl.py  # Brown 2014 v-eQTL
├── parse_gtex_null_tier1.py  # GTEx independent eQTLs (null T1)
├── parse_gtex_null_tier2.py  # GTEx distance-matched controls (null T2)
├── parse_fas_exon.py     # FAS exon 3 combinatorial mutagenesis
├── parse_kras_neighborhood.py  # KRAS synthetic neighborhood
├── parse_mst1r_splicing.py     # MST1R/RON exon 11 splicing
├── parse_tcga_doubles.py       # TCGA somatic co-occurring mutations
├── parse_trna41.py             # tRNA41 cloverleaf (structural ground truth)
├── parse_ke_sl6.py             # Ke SL6 analysis (standalone)
├── parse_ke_sl6_pairs.py       # Ke SL6 variant pairs (structural ground truth)
├── parse_rauscher_cftr.py      # CFTR negative control
├── parse_okgp_chr12.py         # OKGP population pairs
└── parse_correlated_eqtl.py    # High-LD correlated eQTL (LD null)
```

## aggregate.py Pipeline Steps

### Step 1 — Parse All Datasets

Each parser reads its raw input and returns a DataFrame of variant pairs. Parsers handle dataset-specific logic (allele extraction, position mapping, filtering).

### Step 2 — Gene Strand Lookup

Collects all unique gene symbols, queries Ensembl REST API in batches of 1000, caches results in `data/gene_strand_cache.json`. Maps GTEx Ensembl IDs to symbols via egenes files.

### Step 2.5 — Enforce pos1 ≤ pos2 Ordering

All `*1`/`*2` column pairs are swapped when pos1 > pos2. This ensures canonical ordering for epistasis_id construction and deduplication. Liftover datasets order by hg38 positions; native datasets order by pos1/pos2.

### Step 3 — Validate Reference Alleles

Checks each variant's ref allele against the hg38 FASTA (via `seqmat`). Resolution order for mismatches:
1. If alt matches hg38 ref → swap ref/alt
2. If complement(ref) matches hg38 ref → complement both (opposite strand data)
3. Otherwise → log as unresolved mismatch

### Step 4 — Build Epistasis IDs

Format: `gene:chrom:pos:ref:alt:strand|gene:chrom:pos:ref:alt:strand`

Rules:
- Always hg38 positions
- pos1 < pos2 enforced
- Bare chromosome (no `chr` prefix)
- Strand: P (positive/forward) or N (negative/reverse)
- If no gene: literal `GENE`

### Step 5 — Export Per-Dataset TSVs

Each dataset gets a TSV with its specific columns plus standard columns. Saved to `parsed_pairs/`.

### Step 6 — Combined File

All per-dataset exports concatenated → `all_pairs_combined.tsv`

### Step 7 — TopLD Export

Unique variants and pair coordinates for LD clumping → `variants_for_topld_hg38.tsv`, `pairs_for_topld_hg38.tsv`

### Step 8 — Distance Distribution Plot

Histogram of pairwise distances by dataset → `distance_distributions.png`

## add_maf.py Post-Processing

Adds gnomAD genome allele frequencies (maf1/maf2) to all output TSVs.

### Dataset Classification

| Category | Datasets | Position Columns | MAF Source |
|----------|----------|-----------------|------------|
| Liftover | MPRA, Yang, Brown, MST1R | pos1_hg38/pos2_hg38 | Ensembl VEP |
| Native hg38 | GTEx T1/T2, TCGA, Corr. eQTL, Rauscher | pos1/pos2 | Ensembl VEP |
| OKGP | OKGP chr12 | pos1/pos2 | Copies AF1/AF2 |
| Synthetic | FAS, KRAS, tRNA41, Ke SL6 | — | NaN |

### Caching

All VEP results cached in `data/variant_maf_cache.json`. Cache is saved every 100 batches during queries and on completion. First run: ~3 hours (51K variants at 50/batch). Subsequent runs: instant.

## Dataset Details

### Positive Epistasis Sets

| Dataset | Parser | Genome Build | Pairs | Notes |
|---------|--------|-------------|-------|-------|
| Uebbing MPRA | `parse_uebbing_mpra.py` | hg19→hg38 | 100 positive, 3 null | Enhancer variant pairs, dual-luciferase MPRA |
| Yang 2016 | `parse_yang_evqtl.py` | hg19→hg38 | 1,452 | evQTL × peSNP pairs, cis-cis only |
| Brown 2014 | `parse_brown_veqtl.py` | hg19→hg38 | 16 | Replicated v-eQTL pairs, alleles from Ensembl |

### Null Sets

| Dataset | Parser | Type | Pairs | Design |
|---------|--------|------|-------|--------|
| GTEx T1 | `parse_gtex_null_tier1.py` | Unmatched | 1,635 | Genes with ≥2 independent eQTL signals (LCL + Whole Blood) |
| GTEx T2 | `parse_gtex_null_tier2.py` | Distance-matched | 126,413 | For each positive pair's anchor, GTEx eQTLs for same gene within 6kb |
| Correlated eQTL | `parse_correlated_eqtl.py` | LD null | 27,343 | High-LD variant pairs (R², D') |

### Experimental Epistasis

| Dataset | Parser | Type | Pairs | Notes |
|---------|--------|------|-------|-------|
| FAS exon 3 | `parse_fas_exon.py` | Synthetic/experimental | ~16K | Combinatorial mutagenesis, empirical epistasis scores |
| MST1R splicing | `parse_mst1r_splicing.py` | hg19→hg38 | 972 | Minigene splicing assay, exon 11 |
| TCGA doubles | `parse_tcga_doubles.py` | hg38 native | 10,900 | Somatic co-occurring mutations, conditional prob ≥0.80 |
| OKGP chr12 | `parse_okgp_chr12.py` | hg38 native | 100,000 | Population variant pairs with allele frequencies |

### Structural Ground Truth

| Dataset | Parser | Pairs | Notes |
|---------|--------|-------|-------|
| tRNA41 | `parse_trna41.py` | 24,309 | Cloverleaf structure, base-paired labels, contact map |
| Ke SL6 | `parse_ke_sl6_pairs.py` | 36,045 | Stem-loop structure, base-paired labels, contact map |

### Controls

| Dataset | Parser | Pairs | Notes |
|---------|--------|-------|-------|
| KRAS neighborhood | `parse_kras_neighborhood.py` | 13,455 | Synthetic SNV×SNV from FASTA |
| Rauscher CFTR | `parse_rauscher_cftr.py` | 6 | Negative control: missense × synonymous (translational epistasis) |

## Output Files

### Per-Dataset TSVs (`parsed_pairs/`)

```
mpra_positive_pairs.tsv       # 100 rows
mpra_null_pairs.tsv           # 3 rows
yang_positive_pairs.tsv       # 1,452 rows
brown_positive_pairs.tsv      # 16 rows
gtex_null_tier1_pairs.tsv     # 1,635 rows
gtex_null_tier2_pairs.tsv     # 126,413 rows
fas_pairs.tsv                 # ~16,728 rows
kras_neighborhood_pairs.tsv   # 13,455 rows
mst1r_splicing_pairs.tsv      # 972 rows
tcga_doubles_pairs.tsv        # 10,900 rows
trna41_pairs.tsv              # 24,309 rows
rauscher_cftr_pairs.tsv       # 6 rows
okgp_chr12_pairs.tsv          # 100,000 rows
correlated_eqtl_pairs.tsv    # 27,343 rows
```

### Structural Ground Truth Files
```
trna41_singles.tsv            # 222 single-mutation effects
trna41_contact_map.tsv        # 74×74 binary contact matrix
ke_sl6_pairs.tsv              # 36,045 double-mutation pairs
ke_sl6_singles.tsv            # 270 single-mutation effects
ke_sl6_contact_map.tsv        # 90×90 binary contact matrix
ke_sl6_sequence.txt           # WT sequence (90 nt)
ke_sl6_analysis.tsv           # Standalone analysis (not in combined)
```

### Aggregated Files
```
all_pairs_combined.tsv        # All datasets concatenated
epistasis_ids_light.csv       # source, label, epistasis_id, maf1, maf2
distance_distributions.png    # Histogram by dataset
variants_for_topld_hg38.tsv   # Unique variants for LD lookup
pairs_for_topld_hg38.tsv      # Pair coordinates for LD lookup
```

### Cache Files (`data/`)
```
gene_strand_cache.json        # Ensembl gene strand lookups
variant_maf_cache.json        # Ensembl VEP MAF lookups
```

## Standard Column Reference

| Column | Description | Present In |
|--------|-------------|-----------|
| `source` | Dataset identifier | All |
| `pair_id` | Unique pair ID within dataset | All |
| `gene` | Gene symbol (or 'GENE') | Most |
| `chr` | Chromosome (bare, no 'chr' prefix) | All |
| `pos1`, `pos2` | Original positions | All |
| `pos1_hg38`, `pos2_hg38` | Lifted hg38 positions | Liftover datasets |
| `ref1`, `alt1` | Variant 1 alleles (validated vs hg38) | All |
| `ref2`, `alt2` | Variant 2 alleles (validated vs hg38) | All |
| `distance_bp` | Distance between variants | All |
| `label` | positive, null_tier1, null_tier2, null_ld, synthetic, negative_control | All |
| `maf1`, `maf2` | gnomAD genome allele frequencies | After add_maf.py |
| `genome_build` | hg19 or hg38 | All |
| `epistasis_id` | Canonical pair identifier | All |

## Troubleshooting

- **Ensembl API rate limits**: The pipeline includes sleep intervals (0.1–0.5s) between requests. If you hit rate limits, re-run — caches ensure no duplicate work.
- **seqmat errors**: Ensure hg38 FASTA is accessible. `seqmat.SeqMat.from_fasta('hg38', ...)` must resolve.
- **GTEx tar extraction**: Handled automatically by `ensure_gtex_extracted()`. Files extract to `data/GTEx_Analysis_v8_eQTL_independent/` and `data/GTEx_Analysis_v8_eQTL/`.
- **Missing input files**: Datasets with missing files are skipped gracefully (FAS, MST1R, TCGA).
- **TCGA dependency**: Requires separate ParseTCGA project. If parquet not found, TCGA is skipped.
