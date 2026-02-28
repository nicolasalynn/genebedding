# Codebase Analysis: Optimality, Cleanliness, Modularity

Analysis for paper submission readiness. Focus: **genebeddings** library and **notebooks/processing** pipeline.

---

## 1. Modularity

### 1.1 Single very large module

- **`genebeddings/genebeddings.py`** is ~4,800 lines and mixes:
  - Geometry classes (`SingleVariantGeometry`, `EpistasisGeometry`)
  - Data classes (`EpistasisMetrics`, `DBMetadata`)
  - `VariantEmbeddingDB`
  - Parsing (`parse_single_mut_id`, `parse_epistasis_id`)
  - Embedding helpers (`embed_single_variant`, `embed_epistasis`, `_prepare_epistasis_sequences`)
  - DataFrame integration (`add_single_variant_metrics`, `add_epistasis_metrics`, `add_conditional_nucleotide_probs`)
  - Dependency map (`compute_dependency_map`, `compute_embedding_perturbation_map`, …)

**Recommendation:** Split into submodules, e.g. `genebeddings/db.py`, `genebeddings/geometry.py`, `genebeddings/embedding_io.py`, `genebeddings/metrics.py`, `genebeddings/dependency_map.py`, and re-export from `genebeddings/__init__.py`. This improves navigation, testing, and reuse. For a paper submission, splitting is optional but improves reviewer perception.

### 1.2 Duplicate constants (fixed in follow-up)

- **Epistasis DB key suffixes** are defined in two places:
  - `genebeddings/genebeddings.py`: `KEY_WT`, `KEY_M1`, `KEY_M2`, `KEY_M12`, `KEY_DELTA1`, …
  - `notebooks/processing/process_epistasis.py`: `_EPI_KEY_WT`, `_EPI_KEY_M1`, … (same values)

**Recommendation:** Use a single source of truth. Either import from `genebeddings.genebeddings` in `process_epistasis.py` or define shared constants in a small `genebeddings/constants.py` used by both.

### 1.3 Two EpistasisGeometry implementations

- **`genebeddings/genebeddings.py`**: `EpistasisGeometry` — used by `add_epistasis_metrics` and the main API; additive expectation; cosine/L2 diff.
- **`genebeddings/benchmarks/epistasis_geometry.py`**: Different `EpistasisGeometry` with optional “geometric” and “pythagorean” expectations and MDS/plotting.

**Recommendation:** Document clearly which is canonical (the one in `genebeddings.py`). In README or METHODS, state that the paper pipeline uses the additive expectation only and that `benchmarks/epistasis_geometry.py` is an alternative for benchmarking. No need to merge unless you want one unified class with a switch for expectation type.

### 1.4 Pipeline vs library boundary

- **`notebooks/processing/process_epistasis.py`** correctly depends on `genebeddings` and `genebeddings.epistasis_features` for covariance. It does not reimplement geometry.
- **`notebooks/paper_data_config.py`** is pipeline-specific (paths, data root). Keeping it under `notebooks/` is appropriate.

**Verdict:** Boundary is clear; pipeline is modular relative to the library.

---

## 2. Cleanliness

### 2.1 Exception handling

- **Bare `except:`** appears in:
  - `genebeddings/benchmarks/simple_fine_tune.py` (around line 75): catches all, returns `None, None`.
  - `genebeddings/benchmarks/enhanced_pathogenicity_benchmark.py` (around line 795): catches all, sets `final_auc = 0`.

**Recommendation:** Replace with `except Exception:` and log the exception (e.g. `logger.debug(...)`) so failures are visible during debugging. Bare `except` also catches `KeyboardInterrupt`/`SystemExit`, which is usually undesirable.

- Elsewhere, **`except Exception as e`** with logging or warnings is used consistently — good.

### 2.2 Public API surface

- **`genebeddings/__init__.py`** exports a subset of symbols; many useful names (e.g. `add_epistasis_metrics`, `embed_single_variant`, `parse_epistasis_id`) live in `genebeddings.genebeddings` but are not in the top-level `__all__`.

**Recommendation:** Either (a) add the main entry points to `__init__.py` and `__all__` so that `from genebeddings import add_epistasis_metrics` works, or (b) document in README that “advanced” use is via `from genebeddings.genebeddings import ...`. For a paper, (a) is friendlier for reproducibility.

### 2.3 Type hints and docstrings

- Type hints are used widely in `genebeddings.py`, `process_epistasis.py`, `epistasis_features.py`, and wrappers.
- Public functions and classes have docstrings (often NumPy style). Parameters and returns are described.

**Verdict:** Good for review and reuse.

### 2.4 Naming and style

- Consistent use of `snake_case` for functions/variables and `PascalCase` for classes. Constants in `UPPER` or `CamelCase` where appropriate.
- No TODO/FIXME/HACK found in the codebase — clean.

---

## 3. Optimality

### 3.1 VariantEmbeddingDB

- **WAL mode** and **NORMAL synchronous** — appropriate for throughput and durability.
- **Reusable cursors** (`_cur_has`, `_cur_get`, `_cur_put`) — avoids repeated cursor creation.
- **Index on `mut_id`** — lookups are indexed.

**Verdict:** Design is appropriate for the use case.

### 3.2 add_epistasis_metrics

- **Batched path** when `batch_size > 1`: one forward pass per chunk of pairs; falls back to sequential if the model does not accept lists. Good for GPU utilization.
- **DB check before compute**: skips embedding when all four keys exist and `force=False` — avoids redundant work.
- **Batched Mahalanobis** when `cov_inv` is set — vectorized over rows; no per-row Python loop for the quadratic form.

**Verdict:** Efficient for typical workloads.

### 3.3 Lookup-and-copy (embedding_lookup_bases)

- **Per epistasis_id** loop over candidate DBs: for each id, up to four `has()` and four `load()` calls per candidate. For very large tables (e.g. 100k+ ids) and many bases, consider batching: e.g. for each candidate DB, query which of the requested ids exist (single SQL `WHERE mut_id IN (...)` or chunked), then bulk-load and store. Current approach is simple and correct; optimize only if profiling shows DB I/O as a bottleneck.

**Verdict:** Acceptable as-is; document possible future batch optimization if you mention it in the paper.

### 3.4 Covariance and recompute

- **compute_cov_inv_from_paths_combined** streams residuals from DBs and fits one covariance — memory usage is linear in number of residuals. **epistasis_ids** filter reduces work when only a subset is needed.
- **recompute_metrics_with_cov_inv** opens one DB per (source, model) and runs `add_epistasis_metrics` per source; no unnecessary re-embedding.

**Verdict:** Sensible and scalable.

---

## 4. Paper submission checklist

| Item | Status |
|------|--------|
| Single source of truth for DB key constants | Fix: import from genebeddings in process_epistasis |
| No bare `except:` | Fix: use `except Exception` + log in benchmarks |
| Public API documented or exported | Consider exporting add_epistasis_metrics, parse_*, embed_* in __init__ |
| README / METHODS state which EpistasisGeometry is used | Clarify additive-only in pipeline; benchmarks variant optional |
| Optional: split genebeddings.py | Improves modularity; not strictly required for submission |
| Tests for core path | test_wrappers exists; consider pytest for add_epistasis_metrics + DB round-trip |
| License (e.g. MIT) | Present in pyproject.toml |
| Version and dependencies | pyproject.toml and optional deps are clear |

---

## 5. Summary

- **Modularity:** Pipeline is well separated from the library. The main improvement is removing duplicate constants and, optionally, splitting `genebeddings.py` and clarifying the two EpistasisGeometry roles.
- **Cleanliness:** Code is readable and documented. Fix bare `except` in benchmarks and consider a clearer public API in `__init__.py`.
- **Optimality:** DB usage, batching in `add_epistasis_metrics`, and covariance/recompute flows are sound. Lookup-from-multiple-DBs is fine unless you scale to very large id sets.

Applying the constant fix and the two bare-`except` fixes will address the most visible issues for reviewers; the rest can be done as time allows before submission.
