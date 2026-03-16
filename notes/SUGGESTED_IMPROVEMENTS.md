# Thorough Review: Suggested Improvements for genebeddings

This is a deeper review of the repository structure, core API
(`genebeddings/genebeddings.py`), wrappers in `genebeddings/wrappers/`,
benchmarks in `genebeddings/benchmarks/`, and the surrounding docs/tests.

## Critical/Functional Issues (fix first)

- `genebeddings/__init__.py` exports `GeneBeddings`, but there is no
  `GeneBeddings` class or symbol in `genebeddings/genebeddings.py`. This
  makes `import genebeddings` fail immediately.
- `BorzoiWrapper` does not inherit `BaseWrapper` and does not implement
  `predict_tracks()`. It exposes `tracks()` instead. This breaks the
  standardized wrapper contract and causes `quick_test.py` inheritance
  checks to fail.
- `CaduceusWrapperOld` lives alongside a newer `CaduceusWrapper` in
  `caduceus_wrapper.py`. The old class is still in the file, which risks
  accidental use or confusion. Consolidate to one implementation.
- `GPNMSAWrapper.embed()` expects an MSA tensor/array, not a sequence
  string. This deviates from `BaseWrapper.embed(seq: str, ...)` and
  should be explicitly separated (e.g., `embed_msa(...)`) or the base
  API should be generalized for MSA models.
- Multiple wrappers rely on local assets that are missing from the repo:
  `assets/conformer/` and `assets/convnova/` are referenced but do not
  exist, so defaults will fail at runtime.
- Benchmark scripts expect data in `assets/benchmarks/` (ClinVar subsets,
  gene strand mapping), but this directory is not present. Most benchmark
  paths will fail by default.

## API Consistency and Wrapper Behavior

- `BaseWrapper.embed()` declares `seq: str`, but many wrappers accept
  `Union[str, List[str]]` (NT, DNABERT, Evo2, MutBERT, etc.). Consider
  standardizing batch handling (either always accept lists or add
  `embed_batch()` across wrappers).
- `SpeciesLMWrapper.embed()` uses `layers` while others use `layer`.
  Standardize parameter naming and shapes for consistency.
- `SpliceAIWrapper` provides `forward()` and `predict_splice_sites()` but
  this capability is not discoverable via `BaseWrapper.supports_capability`.
  Either add a new capability key or document the extra API clearly.
- `BorzoiWrapper` exposes `tracks()` and `tracks_by_name()` instead of
  `predict_tracks()`; other modules (e.g., benchmarks) call `tracks()`.
  Decide on one interface and adapt both wrappers and benchmarks.
- Sequence normalization logic is duplicated across wrappers with small
  differences (e.g., N handling, U/T conversion). Centralize in a shared
  utility to reduce drift.
- Device/dtype selection is copied in many wrappers; a shared helper in
  `BaseWrapper` would reduce duplication and improve consistency.

## Packaging and Dependency Management

- There is no `pyproject.toml`/`setup.cfg`. `pip install -e .` in
  `TESTING.md` currently cannot work. Add a proper package definition
  and tie `__version__` to it.
- `pip-requirements.txt` is monolithic and heavy; split into extras
  (`genebeddings[nt]`, `[borzoi]`, `[alphagenome]`, `[evo2]`, etc.) with a
  minimal core dependency set.
- Several wrappers depend on packages not listed in requirements:
  `evo2`, `alphagenome_research` (JAX/Haiku), `spliceai-pytorch`,
  `multimolecule`, `gpn` (from Git), and HyenaDNA dependencies. Add them
  to extras or clearly document install steps per wrapper.
- `spec-file.txt` is Linux-only; add a lighter `environment.yml` or a
  macOS-friendly conda spec.

## Documentation and Project Onboarding

- `README.md` is a placeholder (`# dlm_wrappers`). Replace with a real
  README: project overview, installation, minimal usage, and links to
  wrappers/benchmarks/tests.
- `genebeddings/wrappers/summary.md` is outdated and missing many
  wrappers (AlphaGenome, Evo2, GPN-MSA, HyenaDNA, SpliceAI, SpliceBERT).
  Update it or auto-generate from wrapper metadata.
- Add a concise "wrapper cookbook" section showing input constraints,
  required dependencies, and example calls for each wrapper.
- Document that `dependency_map.py` expects `model.probs()` or
  `model.embedding()` methods, which are not implemented by wrappers
  today. Either add these methods or update the docs to match reality.

## Benchmarks and Data Artifacts

- `genebeddings/benchmarks/` is not a Python package (no `__init__.py`)
  and relies on `sys.path` hacks. Convert it into an importable module.
- Benchmarks duplicate mutation parsing logic separate from
  `genebeddings.py` (`parse_single_mut_id` vs
  `benchmarks/variants.parse_mut_id`). Consolidate into one utility.
- Evaluate script claims "local assets where available" but assets are
  missing for Conformer and ConvNova; update defaults or document.

## Testing and CI

- `quick_test.py` and `test_wrappers.py` only cover a subset of wrappers.
  Add newer wrappers (AlphaGenome, Evo2, HyenaDNA, GPN-MSA, SpliceAI,
  SpliceBERT, MutBERT, GenomeNet) or explicitly document the exclusions.
- Replace `exec()` usage in `test_wrappers.py` with `importlib` to
  improve safety and traceability.
- Add a minimal CI workflow that runs smoke tests and lints; guard
  heavy model tests behind a flag.

## Core Library Structure

- `genebeddings/genebeddings.py` is very large (3k+ lines) and mixes
  geometry, parsing, embedding, DB storage, plotting, and benchmarks.
  Split into submodules (`geometry.py`, `db.py`, `embedding.py`, etc.).
- `VariantEmbeddingDB` stores flattened embeddings only; original shapes
  are lost. Store shape metadata so token-level embeddings can be
  reconstructed, or document that only pooled embeddings are supported.
- `VariantEmbeddingDB.iter_all()` uses LIMIT/OFFSET, which is slow on
  large tables. Consider iterating by rowid or using a streaming cursor.

## Security/Trust Considerations

- Several wrappers use `trust_remote_code=True` (DNABERT, Caduceus,
  HyenaDNA, etc.). Document this clearly and provide a safe mode.

## Suggested Next Steps (Practical)

1. Fix the `GeneBeddings` export issue so the package imports.
2. Make Borzoi conform to `BaseWrapper` or clearly separate it.
3. Add proper packaging (`pyproject.toml`) and minimal install path.
4. Restore missing assets or adjust default paths to avoid runtime errors.
5. Update README + wrapper summary to reflect actual functionality.
6. Normalize wrapper APIs and add unified tests for all wrappers.

