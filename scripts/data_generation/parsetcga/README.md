# parsetcga

Programmatic access to TCGA clinical (pickle) and mutation (parquet) data with survival analysis.

## Mutation and epistasis IDs

- **Mutation ID**: `gene:chrom:pos:ref:alt` with `chrom` **without** `chr` (e.g. `TP53:17:7577120:G:A`).
  - Built automatically in query results (`mutation_id` column) and via `mutation_id(gene, chrom, pos, ref, alt)`.
- **Epistasis ID**: multiple mutation IDs joined with `|`, sorted (e.g. `KRAS:12:25380275:C:A|TP53:17:7577120:G:A`).
  - Use `epistasis_id(list_of_mutation_ids)`; `patient_mutation_ids(..., add_epistasis_id=True)` adds an `epistasis_id` column per patient.

## Data

- **Clinical** (`data/df_p_all.pkl`): One row per patient; `patient_uuid` (join key `case_id`), `Proj_name`, survival-related columns, therapies.
- **Mutations** (`data/tcga_all.parquet`): One row per mutation; `case_id`, `Gene_name`, `Proj_name`, variant details. Queried via DuckDB (no full load).

## Install

```bash
pip install -r requirements.txt
# or: pip install pandas pyarrow duckdb lifelines
```

From the repo root, use the package as `parsetcga` (no pip install needed if you run from repo).

## Quick start

```python
from parsetcga import TCGAData

tcga = TCGAData()  # uses ./data or TCGADATA_DIR

# --- Clinical ---
clin = tcga.clinical.survival_prepared()   # duration (years), event, case_id, therapies, Proj_name
raw  = tcga.clinical.raw()                 # full pickle

# --- Mutations (lazy, DuckDB) ---
tp53 = tcga.mutations.patients_with_mutation("TP53")
both = tcga.mutations.patients_with_one_or_two_genes("TP53", "KRAS")  # only_a, only_b, both, either
tcga.mutations.project_breakdown()         # n_patients, n_mutations per Proj_name
tcga.mutations.gene_summary(["TP53", "KRAS"], by_project=True)
tmb  = tcga.mutations.tumor_mutation_burden(project="TCGA_BRCA")
df   = tcga.mutations.query(genes=["TP53"], limit=1000)

# --- Survival ---
sa = tcga.survival()
# Cohort: patients with TP53 mutation
cohort_df = sa.cohort_from_mutation(["TP53"])
res = sa.kaplan_meier(cohort_df, target_label="TP53 mutated", control_label="Control", plot=True)
# Two-gene cohorts
cohort_df = sa.cohort_from_two_genes("TP53", "KRAS", cohort="both")
# Segmentation summary by project
from parsetcga import survival_segmentation_summary
survival_segmentation_summary()
```

## API summary

| Use case | Method |
|----------|--------|
| Patients with one gene | `tcga.mutations.patients_with_mutation("TP53")` |
| Patients with two genes (only A, only B, both, either) | `tcga.mutations.patients_with_one_or_two_genes("TP53", "KRAS")` |
| Cancer project breakdown | `tcga.mutations.project_breakdown()` |
| Mutation counts per patient (TMB) | `tcga.mutations.tumor_mutation_burden()` |
| Survival-ready clinical | `tcga.clinical.survival_prepared()` |
| Survival by mutation cohort | `tcga.survival().cohort_from_mutation(["TP53"])` then `kaplan_meier(...)` |
| Segmentation by project | `survival_segmentation_summary()` |
| Mutation ID (canonical) | `mutation_id(gene, chrom, pos, ref, alt)`; `parse_mutation_id(s)` |
| Epistasis ID | `epistasis_id([mutation_id1, ...])`; `tcga.mutations.patient_mutation_ids(..., add_epistasis_id=True)` |
| Double-variant co-occurrence | `tcga.mutations.co_occurrence("mut1\|mut2")` → P(mut2\|mut1), P(mut1\|mut2), contingency, counts |
| Epistasis survival (single vs both) | `tcga.epistasis_survival_segmentation("mut1\|mut2", per_project=True)` → p_value per project |

**Faster co_occurrence / double_variant_case_partition:** The first time you use either function (or after deleting the lookup file), a persistent index is built under `data/tcga_mutations_lookup.duckdb` (one-time, can take a few minutes). After that, lookups are very fast.

**Prebuild the lookup (recommended):** Run once in your environment (only needs `duckdb`):
```bash
python scripts/prebuild_lookup.py
# or from anywhere with TCGADATA_DIR set:
# TCGADATA_DIR=/path/to/data python scripts/prebuild_lookup.py
```
Or from Python: `build_mutation_lookup()` or `tcga.mutations.build_mutation_lookup()`.

## Custom data dir

```python
tcga = TCGAData(data_dir=Path("/path/to/data"))
# or set env: TCGADATA_DIR=/path/to/data
```
