# Notebooks

## Data root (single directory for all outputs)

All embeddings, dataframes, and CSVs are stored under **one root directory**. By default: `/tamir2/nicolaslynn/data/epistasis_paper`. Override with the environment variable `EPISTASIS_PAPER_ROOT` (e.g. on a remote machine).

- **data/** — Input and generated CSVs (subset files, `epistasis_aggregated.csv`, etc.). Use `from notebooks.paper_data_config import data_dir`.
- **embeddings/** — Embedding DBs per source per model, null covariance packs, annotated parquets. Use `embeddings_dir()`.

`run_everything.ipynb` uses `paper_data_config`; run from the repo with this root so all outputs go to the same place. **Data generation** notebooks live in `/Users/nicolaslynn/Documents/phd/projects/genomenet/data_generation` and run only there; their outputs can be pointed at this pipeline via the same data root or custom paths.

## Layout

- **DATA_GENERATION.md** – How each source dataset is created (ClinVar, eQTL, FAS, MST1R, tRNA, TopLD, OKGP, TCGA, null). Read this to reproduce or extend variant/epistasis tables.
- **run_everything.ipynb** – **Run this notebook** to execute the full pipeline: process all sources (null first), then compute and save covariance packs per model. **Environment control:** set `ENV_PROFILE` to `"alphagenome"`, `"evo2"`, `"main"`, or `"all"` so only the models for the current conda/env run (AlphaGenome and Evo2 each have dedicated envs; all others use `"main"`).
- **process_epistasis.py** – Pipeline implementation: `run_sources()` (embeddings + DBs), `run_covariance_and_save()` (cov/cov_inv .npz). All models (including AlphaGenome, Evo2, SpliceAI) are configured; SpliceAI runs only for splicing sources (fas_analysis, mst1r_analysis, kras). Uses `add_epistasis_metrics(..., batch_size=...)` for batched embedding.

## Processing pipeline

**Input:** Either (1) a list of per-source tables, or (2) a single dataframe with all double variant IDs and a `source` column.

- **Per-source list:** Each table has at least an `epistasis_id` column (format: `mut1|mut2` with `GENE:CHROM:POS:REF:ALT:STRAND` per mut). Pass `run_sources([(source_name, path_or_df), ...], output_base=...)`.
- **Single dataframe:** One table with columns `epistasis_id` and `source` (or another name via `source_col`). Rows are grouped by `source`; embedding storage is split by that value: `{output_base}/{source}/{model_key}.db`. Use `run_from_single_dataframe(df, output_base=..., source_col="source")`. In `run_everything.ipynb`, set `USE_SINGLE_DATAFRAME = True` and `SINGLE_DATAFRAME_PATH` to your combined CSV (e.g. `epistasis_aggregated.csv`).

**Output:**

1. One directory per source, each containing one SQLite database per model, e.g.:

```
embeddings/
  null/
    nt500_multi.db
    alphagenome.db
    convnova.db
    borzoi.db
    spliceai.db   # only for splicing sources, see below
    ...
  fas_analysis/
    nt500_multi.db
    alphagenome.db
    spliceai.db
    ...
  mst1r_analysis/
    ...
```

2. **Null covariance** (used in metrics): After the null source is processed, covariance is fitted from `null/*.db` only and saved under `embeddings/null_cov/{model_key}_pack.npz`. That **null cov_inv is passed into add_epistasis_metrics** when processing every non-null source so that epistasis metrics include Mahalanobis terms (epi_mahal, mahal_obs, mahal_add, etc.) relative to the null background.

3. Optional combined covariance: `embeddings/{model_key}_pack.npz` with `cov`, `cov_inv` from all source DBs combined (for other uses).

**Models:** AlphaGenome runs for all sources. SpliceAI runs only for **splicing** sources: `fas_analysis`, `mst1r_analysis`, `kras`. Set `OPENSPLICEAI_MODEL_DIR` or pass `spliceai_model_dir` for SpliceAI. Full list in `notebooks.processing.process_epistasis.FULL_MODEL_CONFIG`.

**Per-source model control:** Pass `source_model_map` (dict: `source_name -> list of model keys`) to run only selected models per source. If a source is missing from the map, the global `model_keys` is used. Example: `{"null": ["nt500_multi", "convnova"], "fas_analysis": ["nt500_multi", "convnova", "spliceai"]}`. When set, the default SpliceAI-on-splicing-sources rule is not applied; the map fully defines which models run where.

**Order:** The pipeline always processes the source named `null` first, then the remaining sources.

**Environment profiles:** Some models require a dedicated environment. Use one profile per run so only the models for the current env execute:
- **alphagenome** — run only AlphaGenome (use in AlphaGenome / JAX env)
- **evo2** — run only Evo2 (use in Evo2 env)
- **main** — run all models that do *not* need a special env (excludes alphagenome, evo2)
- **all** — run every model (only if all dependencies are in one env)

In the notebook: set `ENV_PROFILE = "alphagenome"` (or `"evo2"` / `"main"` / `"all"`); `MODEL_KEYS` is then set via `get_model_keys_for_env(ENV_PROFILE)`. From CLI: `--env-profile alphagenome`.

**Run from project root:**

```bash
# Example: null first, then fas and mst1r
python -m notebooks.processing.process_epistasis \
  --sources null:path/to/null_epistasis.csv fas_analysis:path/to/fas_subset.csv mst1r_analysis:path/to/mst1r.csv \
  --output embeddings \
  --id-col epistasis_id
```

Or from Python:

```python
from pathlib import Path
import pandas as pd
from notebooks.processing.process_epistasis import run_sources, run_from_single_dataframe

# Option A: per-source list
run_sources(
    [
        ("null", Path("data/null_epistasis.csv")),
        ("fas_analysis", Path("data/fas_subset.csv")),
        ("mst1r_analysis", Path("data/mst1r_subset.csv")),
    ],
    output_base=Path("embeddings"),
    model_keys=["nt500_multi", "convnova", "borzoi"],
    id_col="epistasis_id",
)

# Option B: single dataframe with 'source' column (storage split by source)
df = pd.read_csv("data/epistasis_aggregated.csv")  # columns: epistasis_id, source, ...
run_from_single_dataframe(
    df,
    output_base=Path("embeddings"),
    source_col="source",
    id_col="epistasis_id",
    model_keys=["nt500_multi", "convnova", "borzoi"],
)

# Option C: different models per source (source_model_map)
run_sources(
    [("null", Path("data/null_epistasis.csv")), ("fas_analysis", Path("data/fas_subset.csv"))],
    output_base=Path("embeddings"),
    model_keys=["nt500_multi", "convnova", "spliceai"],
    source_model_map={"null": ["nt500_multi", "convnova"], "fas_analysis": ["nt500_multi", "convnova", "spliceai"]},
    id_col="epistasis_id",
)
```
