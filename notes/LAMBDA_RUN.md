# Epistasis pipeline on Lambda (8× A100)

What we run on a Lambda Labs instance with 8× A100 GPUs: full embedding + metrics pipeline for all epistasis pairs, one tool at a time per conda env, with cov_inv from okgp and one parquet sheet per tool.

---

## Purpose

- **Input:** One table of epistasis pairs with columns `epistasis_id` and `source` (bundled: `notebooks/processing/data/all_pairs_combined.tsv`, ~323k rows).
- **Per (source, model):** Compute embeddings and store in SQLite: `{output_base}/{source}/{model_key}.db`.
- **cov_inv:** Fit from rows whose `source` is in `COV_INV_SOURCE_NAMES` (okgp_chr12); save to `{output_base}/null_cov/{model_key}_pack.npz` and use for Mahalanobis (epi_mahal, etc.) on all other sources.
- **Output:** One parquet sheet per tool: `{output_base}/sheets/epistasis_metrics_{tool}.parquet`, same structure each, all metrics including Mahalanobis.

---

## Input data

- **File:** `notebooks/processing/data/all_pairs_combined.tsv` (in repo). Columns: `source`, `epistasis_id`.
- **Sources in file:** brown_veqtl, correlated_eqtl, fas_exon, gtex_independent, gtex_matched_null, kras_neighborhood, mst1r_splicing, okgp_chr12, rauscher_cftr, tcga_doubles, trna41, uebbing_mpra, yang_evqtl.
- **Config** (`notebooks/processing/pipeline_config.py`):
  - `COV_INV_SOURCE_NAMES = ["okgp_chr12"]` — rows used to fit cov_inv.
  - `SPLICEAI_SOURCE_NAMES = ["fas_exon"]` — only these sources get SpliceAI; all other sources get all other tools.

---

## Conda environments and tools

We run **one env at a time**; within each env we run **one tool (model) at a time** over all sources (okgp first so cov_inv can be built).

| Profile      | Conda env   | Tools (model_key) |
|-------------|-------------|-------------------|
| **main**    | `genebeddings_main` | nt50_3mer, nt50_multi, nt100_multi, nt250_multi, nt500_multi, nt500_ref, nt2500_multi, nt2500_okgp, convnova, mutbert, hyenadna, caduceus, borzoi, rinalmo, specieslm, dnabert, spliceai |
| **evo2**    | `evo2`      | evo2 |
| **alphagenome** | `alphagenome` | alphagenome |

Batch sizes (A100-tuned) are in `pipeline_config.BATCH_SIZE_BY_TOOL` (e.g. borzoi 4, nt500_multi 32).

---

## Execution order on Lambda

1. **Prereqs (once per instance)**
   - Clone repo, install conda.
   - Run setup scripts so envs exist: `genebeddings_main`, `evo2`, `alphagenome` (see `scripts/setup_envs/`).
   - **seqmat genome data** (once per instance, any env):
     ```bash
     pip install seqmat  # if not already installed
     seqmat setup --path /path/to/seqmat_data --organism hg38
     export SEQMAT_DATA_DIR=/path/to/seqmat_data
     ```
     Add `export SEQMAT_DATA_DIR=...` to `~/.bashrc` so all envs find the genome.
   - Set `EPISTASIS_PAPER_ROOT` for output paths (or rely on default in `notebooks/paper_data_config`).
   - Optional: `~/.hf_token` for Hugging Face (required for gated models like Nucleotide Transformer).

2. **Smoke-test (recommended first)**
   - ALL tools, 5 rows per source, full round-trip (embed + cov_inv + parquet sheets). Verifies every model produces embeddings, DBs are created, and final parquet output works:
   ```bash
   cd /path/to/genebeddings
   bash scripts/run_pipeline_cluster.sh --smoke-test 2>&1 | tee smoke.log
   ```
   - Or one env at a time:
   ```bash
   bash scripts/run_pipeline_cluster.sh --smoke-test --env-profile main
   bash scripts/run_pipeline_cluster.sh --smoke-test --env-profile evo2
   bash scripts/run_pipeline_cluster.sh --smoke-test --env-profile alphagenome
   bash scripts/run_pipeline_cluster.sh --phase metrics
   ```
   - Quick alternative (single tool, 20 rows, no metrics — just verify GPU works):
   ```bash
   bash scripts/run_pipeline_cluster.sh --quick-test --env-profile main
   ```

3. **Full run**  
   - Embed phase: for each profile (main → evo2 → alphagenome), activate that env and run embed for that profile’s tools over all sources (okgp_chr12 first; after each model, cov_inv is computed from okgp_chr12 and saved).  
   - Metrics phase: load full table, compute cov_inv from okgp_chr12 rows, recompute all metrics with that cov_inv, write one parquet per tool to `output_base/sheets/`.

   Single command (runs all profiles then metrics):
   ```bash
   cd /path/to/genebeddings
   bash scripts/run_pipeline_cluster.sh 2>&1 | tee pipeline.log
   ```

   Or run one profile at a time:
   ```bash
   bash scripts/run_pipeline_cluster.sh --env-profile main
   bash scripts/run_pipeline_cluster.sh --env-profile evo2
   bash scripts/run_pipeline_cluster.sh --env-profile alphagenome
   bash scripts/run_pipeline_cluster.sh --phase metrics
   ```

4. **Monitor (second terminal)**  
   - Progress and GPU utilization:
   ```bash
   bash scripts/monitor_pipeline.sh
   ```
   - Uses `pipeline_status.json` in repo root (or `PIPELINE_STATUS_FILE`); refreshes every 2s with status + `nvidia-smi`.

---

## Output layout

- **Embeddings:** `{output_base}/{source}/{model_key}.db` (e.g. `embeddings/okgp_chr12/nt500_multi.db`).  
- **Covariance:** `{output_base}/null_cov/{model_key}_pack.npz` (cov and cov_inv from okgp_chr12).  
- **Sheets:** `{output_base}/sheets/epistasis_metrics_{tool}.parquet` — one parquet per tool, same columns (epistasis_id, source, metrics including epi_mahal, etc.).

`output_base` defaults to `embeddings_dir()` from `notebooks/paper_data_config` (e.g. `$EPISTASIS_PAPER_ROOT/embeddings`).

---

## Options reference

- `--env-profile PROFILE` — Run only this profile’s embed phase (main | evo2 | alphagenome). Omit to run all three then metrics.
- `--phase metrics` — Run only the metrics phase (cov_inv + recompute + sheets).
- `--skip-metrics` — After embed, do not run metrics.
- `--quick-test` — One tool, 20 rows per source; fast GPU/wrapper sanity check.
- `--smoke-test` — ALL tools, 5 rows per source; full end-to-end verification (embed + metrics + parquet).
- `--smoke-test-full` — ALL tools, 5 rows per source + ALL fas_exon rows (16k); full splicing validation.
- `--dry-run` — Print commands, do not run.

---

## Tuning

- **Batch size:** Edit `notebooks/processing/pipeline_config.BATCH_SIZE_BY_TOOL`. Larger = faster but more VRAM; tune so A100 is well utilized without OOM.
- **Sources for cov_inv / SpliceAI:** Edit `COV_INV_SOURCE_NAMES` and `SPLICEAI_SOURCE_NAMES` in `pipeline_config.py` to match the exact `source` values in your data file.
