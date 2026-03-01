"""
Unified epistasis processing pipeline.

Given a DataFrame of epistasis IDs and a source name (e.g. fas_analysis,
mst1r_analysis, tcga_analysis, null), computes embeddings for each model
and writes one SQLite database per model under:

    {output_base}/{source_name}/{model_key}.db

Sources in pipeline_config.COV_INV_SOURCE_NAMES are processed first; cov_inv
(Mahalanobis background) is computed from them and used for all other sources.

SpliceAI runs only for sources in pipeline_config.SPLICEAI_SOURCE_NAMES.
All other tools run for every source. AlphaGenome and evo2 use dedicated envs.

**Environment profiles:** Some models require a dedicated conda/env (e.g. AlphaGenome
uses JAX/CUDA, Evo2 has its own stack). Use an env profile so only the models for the
current environment are run:
  - "alphagenome" → run only AlphaGenome (use in AlphaGenome env)
  - "evo2"       → run only Evo2 (use in Evo2 env)
  - "main"       → run all models that do *not* require a special env (excludes alphagenome, evo2)
  - "all"        → run every model (only if all deps are in one env)

After processing, covariance matrices can be computed from combined DBs
and saved as {model_key}_pack.npz (cov, cov_inv, model, pool) for use in
Mahalanobis-based epistasis metrics.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Type for per-source model override: source_name -> list of model keys to run for that source.
# If a source is missing, global model_keys is used. Enables e.g. SpliceAI only for splicing sources.
SourceModelMap = Optional[Dict[str, Sequence[str]]]

import numpy as np
import pandas as pd

# Source name used for background / null distribution; must be processed first
NULL_SOURCE_NAME = "null"

# Source lists from config (exactly as in the data file). Lazy import to avoid circular import.
def _spliceai_run_for_source(source_name: str) -> bool:
    from notebooks.processing.pipeline_config import SPLICEAI_SOURCE_NAMES
    return source_name in SPLICEAI_SOURCE_NAMES


def _is_cov_inv_source(source_name: str) -> bool:
    from notebooks.processing.pipeline_config import COV_INV_SOURCE_NAMES
    return source_name in COV_INV_SOURCE_NAMES

# All model keys and their (context, init_spec).
# init_spec: "nt" => NTWrapper(model=key); "alphagenome" => AlphaGenomeWrapper(fixed_length=8192);
# "spliceai" => SpliceAIWrapper(model="10k", model_dir=...); "evo2" => Evo2Wrapper("7b_base"); else => Wrapper()
FULL_MODEL_CONFIG: Dict[str, Tuple[int, str]] = {
    "convnova": (1000, "default"),
    "mutbert": (256, "default"),
    "hyenadna": (60_000, "default"),
    "caduceus": (60_000, "default"),
    "nt50_3mer": (3000, "nt"),
    "nt50_multi": (3000, "nt"),
    "nt100_multi": (3000, "nt"),
    "nt250_multi": (3000, "nt"),
    "nt500_multi": (3000, "nt"),
    "nt500_ref": (3000, "nt"),
    "nt2500_multi": (3000, "nt"),
    "nt2500_okgp": (3000, "nt"),
    "borzoi": (260_000, "default"),
    "rinalmo": (511, "default"),
    "specieslm": (600, "default"),
    "alphagenome": (8192 // 2, "alphagenome"),  # 4096; fixed_length=8192 in wrapper
    "evo2": (4_000, "evo2"),
    "spliceai": (10_000, "spliceai"),
    "dnabert": (512, "default"),
}

# Default subset for quick runs (override with model_keys= or env_profile=)
DEFAULT_MODEL_KEYS = [
    "nt500_multi",
    "convnova",
    "mutbert",
    "borzoi",
    "alphagenome",
]

# ---------------------------------------------------------------------------
# Environment profiles: built dynamically from pipeline_config.TOOL_TO_ENV
# ---------------------------------------------------------------------------
from collections import defaultdict as _defaultdict
from notebooks.processing.pipeline_config import get_env_for_tool as _get_env_for_tool

_env_to_models: Dict[str, List[str]] = _defaultdict(list)
for _k in FULL_MODEL_CONFIG:
    _env_to_models[_get_env_for_tool(_k)].append(_k)

ENV_PROFILES: Dict[str, List[str]] = dict(_env_to_models)
ENV_PROFILES["all"] = list(FULL_MODEL_CONFIG)


def get_model_keys_for_env(profile: str) -> List[str]:
    """
    Return the list of model keys to run for the given environment profile.

    Profiles are built dynamically from ``pipeline_config.TOOL_TO_ENV``.
    Each unique conda env name becomes a profile, plus ``"all"`` runs every model.

    Parameters
    ----------
    profile : str
        A conda env name (e.g. "nt", "borzoi", "evo2") or "all".

    Returns
    -------
    list of str
        Model keys to run in this environment.
    """
    if profile not in ENV_PROFILES:
        raise ValueError(
            f"Unknown env profile {profile!r}. "
            f"Choose one of: {list(ENV_PROFILES)}"
        )
    return list(ENV_PROFILES[profile])


logger = logging.getLogger(__name__)

# Epistasis embedding key suffixes: single source of truth from genebeddings
from genebeddings.genebeddings import (
    KEY_WT,
    KEY_M1,
    KEY_M2,
    KEY_M12,
    KEY_DELTA1,
    KEY_DELTA2,
    KEY_DELTA12,
)
_EPI_KEYS_OPTIONAL = (KEY_DELTA1, KEY_DELTA2, KEY_DELTA12)


def _copy_embeddings_from_lookup_dbs(
    df: pd.DataFrame,
    id_col: str,
    target_db: Any,
    source_name: str,
    model_key: str,
    embedding_lookup_bases: Sequence[Union[str, Path]],
    embedding_lookup_flat: bool = False,
) -> int:
    """
    Search existing DBs for each epistasis_id in df; if found, copy WT/M1/M2/M12
    (and optional deltas) into target_db so add_epistasis_metrics will load them
    instead of recomputing. Returns number of epistasis IDs copied.

    Layout: if embedding_lookup_flat is False, expect base/source_name/model_key.db
    (same as output_base). If True, expect base/model_key.db (flat: one dir per run,
    .db files directly inside).
    """
    from genebeddings import VariantEmbeddingDB

    epi_ids = df[id_col].dropna().astype(str).unique().tolist()
    n_copied = 0
    for base in embedding_lookup_bases:
        base = Path(base)
        if embedding_lookup_flat:
            candidate_path = base / f"{model_key}.db"
        else:
            candidate_path = base / source_name / f"{model_key}.db"
        if not candidate_path.exists():
            continue
        try:
            candidate_db = VariantEmbeddingDB(str(candidate_path))
        except Exception as e:
            logger.warning("Could not open lookup DB %s: %s", candidate_path, e)
            continue
        try:
            for epi_id in epi_ids:
                wt_key = epi_id + KEY_WT
                if target_db.has(wt_key):
                    continue
                m1_key = epi_id + KEY_M1
                m2_key = epi_id + KEY_M2
                m12_key = epi_id + KEY_M12
                if not (
                    candidate_db.has(wt_key)
                    and candidate_db.has(m1_key)
                    and candidate_db.has(m2_key)
                    and candidate_db.has(m12_key)
                ):
                    continue
                h_wt = np.asarray(candidate_db.load(wt_key, as_torch=False), dtype=np.float32)
                h_m1 = np.asarray(candidate_db.load(m1_key, as_torch=False), dtype=np.float32)
                h_m2 = np.asarray(candidate_db.load(m2_key, as_torch=False), dtype=np.float32)
                h_m12 = np.asarray(candidate_db.load(m12_key, as_torch=False), dtype=np.float32)
                target_db.store(wt_key, h_wt)
                target_db.store(m1_key, h_m1)
                target_db.store(m2_key, h_m2)
                target_db.store(m12_key, h_m12)
                for suf in _EPI_KEYS_OPTIONAL:
                    key = epi_id + suf
                    if candidate_db.has(key):
                        arr = np.asarray(candidate_db.load(key, as_torch=False), dtype=np.float32)
                        target_db.store(key, arr)
                n_copied += 1
        finally:
            candidate_db.close()
    return n_copied


def _build_model(model_key: str, init_spec: str, spliceai_model_dir: Optional[str] = None):
    """Instantiate wrapper with correct constructor args. Returns model or None."""
    if init_spec == "nt":
        from genebeddings.wrappers import NTWrapper
        return NTWrapper(model=model_key)
    if init_spec == "alphagenome":
        from genebeddings.wrappers import AlphaGenomeWrapper
        return AlphaGenomeWrapper(fixed_length=8192)
    if init_spec == "spliceai":
        from genebeddings.wrappers import SpliceAIWrapper
        model_dir = spliceai_model_dir or os.environ.get("OPENSPLICEAI_MODEL_DIR")
        if not model_dir or not os.path.isdir(model_dir):
            raise FileNotFoundError(
                "SpliceAI/OpenSpliceAI model_dir required. Set OPENSPLICEAI_MODEL_DIR or pass spliceai_model_dir."
            )
        return SpliceAIWrapper(model="10k", model_dir=model_dir)
    if init_spec == "evo2":
        try:
            from genebeddings.wrappers import Evo2Wrapper
            return Evo2Wrapper("7b_base")
        except Exception as e:
            logger.warning("Evo2 not available: %s", e)
            return None
    # default — import only the one wrapper we need (each has unique deps)
    _defaults = {
        "convnova": ("convnova_wrapper", "ConvNovaWrapper", {}),
        "mutbert": ("mutbert_wrapper", "MutBERTWrapper", {}),
        "hyenadna": ("hyenadna_wrapper", "HyenaDNAWrapper", {}),
        "caduceus": ("caduceus_wrapper", "CaduceusWrapper", {}),
        "borzoi": ("borzoi_wrapper", "BorzoiWrapper", {}),
        "rinalmo": ("rinalmo_wrapper", "RiNALMoWrapper", {}),
        "specieslm": ("specieslm_wrapper", "SpeciesLMWrapper", {}),
        "dnabert": ("dnabert_wrapper", "DNABERTWrapper", {}),
    }
    entry = _defaults.get(model_key)
    if entry is None:
        return None
    mod_name, cls_name, kwargs = entry
    import importlib
    mod = importlib.import_module(f"genebeddings.wrappers.{mod_name}")
    cls = getattr(mod, cls_name)
    return cls(**kwargs)


def _load_epistasis_df(path: Union[str, Path], id_col: str = "epistasis_id") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Epistasis table not found: {p}")
    df = pd.read_csv(p)
    if id_col not in df.columns:
        raise ValueError(f"Column {id_col!r} not found in {p}. Columns: {list(df.columns)}")
    return df


def _sort_sources_first_null(
    sources: Sequence[Tuple[str, Union[str, Path, pd.DataFrame]]],
) -> List[Tuple[str, Union[str, Path, pd.DataFrame]]]:
    """Return sources with okgp (cov_inv) sources first, then rest. Process okgp first so cov_inv can be built."""
    okgp_first = [(n, p) for n, p in sources if _is_cov_inv_source(n)]
    rest = [(n, p) for n, p in sources if not _is_cov_inv_source(n)]
    return okgp_first + rest


def run_from_single_dataframe(
    df: pd.DataFrame,
    output_base: Union[str, Path],
    source_col: str = "source",
    model_keys: Optional[Sequence[str]] = None,
    env_profile: Optional[str] = None,
    splicing_sources: Optional[set] = None,
    source_model_map: SourceModelMap = None,
    embedding_lookup_bases: Optional[Sequence[Union[str, Path]]] = None,
    embedding_lookup_flat: bool = False,
    spliceai_model_dir: Optional[str] = None,
    id_col: str = "epistasis_id",
    genome: str = "hg38",
    show_progress: bool = True,
    force: bool = False,
    batch_size: int = 8,
    batch_size_by_model: Optional[Dict[str, int]] = None,
    save_annotated: bool = True,
    annotated_format: str = "parquet",
    use_by_tool: bool = False,
    status_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Run the full processing pipeline from a single DataFrame that contains all
    double variant IDs and a column indicating the source. Embedding storage
    directories are split by the value in `source_col`: each source gets
    `{output_base}/{source_value}/{model_key}.db`. Null is processed first if
    present so that null covariance is available for non-null sources.

    If use_by_tool is True, runs one tool at a time over all sources (run_sources_by_tool).

    Parameters
    ----------
    df : pandas.DataFrame
        Must have columns `id_col` (e.g. epistasis_id) and `source_col` (e.g. source).
        Each unique value in `source_col` defines a source; rows are grouped by it.
    output_base : path
        Base directory for outputs. Each source gets a subdir: output_base/source_name/.
    source_col : str, default "source"
        Column name whose values define the source (and thus the storage subdirectory).
    model_keys, env_profile, splicing_sources, source_model_map, embedding_lookup_bases,
    embedding_lookup_flat, spliceai_model_dir, id_col, genome, show_progress, force, batch_size,
    batch_size_by_model, save_annotated, annotated_format
        Passed through to run_sources().
    """
    if id_col not in df.columns:
        raise ValueError(f"DataFrame must have column {id_col!r}. Columns: {list(df.columns)}")
    if source_col not in df.columns:
        raise ValueError(f"DataFrame must have column {source_col!r}. Columns: {list(df.columns)}")

    unique_sources = df[source_col].dropna().astype(str).unique().tolist()
    # Order so null is first; rest in stable order
    ordered_sources = [s for s in unique_sources if s == NULL_SOURCE_NAME]
    ordered_sources += [s for s in unique_sources if s != NULL_SOURCE_NAME]

    sources = [
        (name, df.loc[df[source_col].astype(str) == name].copy())
        for name in ordered_sources
    ]
    logger.info(
        "Single dataframe: %d rows -> %d sources (col=%r): %s",
        len(df),
        len(sources),
        source_col,
        [name for name, _ in sources],
    )
    if use_by_tool:
        if model_keys is None and env_profile is not None:
            model_keys = get_model_keys_for_env(env_profile)
        model_keys = model_keys or DEFAULT_MODEL_KEYS
        run_sources_by_tool(
            sources,
            output_base=output_base,
            model_keys=model_keys,
            env_profile=env_profile,
            splicing_sources=splicing_sources,
            source_model_map=source_model_map,
            embedding_lookup_bases=embedding_lookup_bases,
            embedding_lookup_flat=embedding_lookup_flat,
            spliceai_model_dir=spliceai_model_dir,
            id_col=id_col,
            genome=genome,
            show_progress=show_progress,
            force=force,
            batch_size=batch_size,
            batch_size_by_model=batch_size_by_model,
            save_annotated=save_annotated,
            annotated_format=annotated_format,
            status_path=status_path,
        )
    else:
        run_sources(
            sources,
            output_base=output_base,
            model_keys=model_keys,
            env_profile=env_profile,
            splicing_sources=splicing_sources,
            source_model_map=source_model_map,
            embedding_lookup_bases=embedding_lookup_bases,
            embedding_lookup_flat=embedding_lookup_flat,
            spliceai_model_dir=spliceai_model_dir,
            id_col=id_col,
            genome=genome,
            show_progress=show_progress,
            force=force,
            batch_size=batch_size,
            batch_size_by_model=batch_size_by_model,
            save_annotated=save_annotated,
            annotated_format=annotated_format,
        )


def run_sources(
    sources: Sequence[Tuple[str, Union[str, Path, pd.DataFrame]]],
    output_base: Union[str, Path],
    model_keys: Optional[Sequence[str]] = None,
    env_profile: Optional[str] = None,
    splicing_sources: Optional[set] = None,
    source_model_map: SourceModelMap = None,
    embedding_lookup_bases: Optional[Sequence[Union[str, Path]]] = None,
    embedding_lookup_flat: bool = False,
    spliceai_model_dir: Optional[str] = None,
    id_col: str = "epistasis_id",
    genome: str = "hg38",
    show_progress: bool = True,
    force: bool = False,
    batch_size: int = 8,
    batch_size_by_model: Optional[Dict[str, int]] = None,
    save_annotated: bool = True,
    annotated_format: str = "parquet",
) -> None:
    """
    Process each source: for each model, compute/store epistasis embeddings
    and add metrics. Writes one .db per model under output_base/source_name/,
    and optionally the annotated DataFrame (with metric columns) to
    output_base/source_name/{model_key}_annotated.parquet (or .csv).
    SpliceAI is run only for sources in pipeline_config.SPLICEAI_SOURCE_NAMES.
    Use source_model_map to override which models run per source (see below).

    Parameters
    ----------
    sources : list of (source_name, path_or_dataframe)
    output_base : path
    model_keys : list of str, optional; which models to run. If None, resolved from env_profile or DEFAULT_MODEL_KEYS.
    env_profile : str, optional; if model_keys is None, use get_model_keys_for_env(env_profile). A conda env name (e.g. "nt", "borzoi", "evo2") or "all".
    splicing_sources : set of str, optional; unused (kept for API compat). SpliceAI runs only for sources in SPLICEAI_SOURCE_NAMES. Ignored when source_model_map is set.
    source_model_map : dict, optional; map source_name -> list of model keys for that source. If a source is missing, model_keys is used. Enables e.g. running SpliceAI only for certain sources or restricting null to a subset of models.
    embedding_lookup_bases : sequence of paths, optional; directories to search for existing embeddings. Layout depends on embedding_lookup_flat (see below).
    embedding_lookup_flat : bool, optional; if False (default), each base has layout base/source_name/model_key.db. If True, each base has .db files directly: base/model_key.db (one directory per run, no source subdirs).
    spliceai_model_dir : str, optional; OpenSpliceAI checkpoint dir; else OPENSPLICEAI_MODEL_DIR.
    id_col : str
    genome : str
    show_progress : bool
    force : bool
    batch_size : int
        Batched embedding in add_epistasis_metrics (e.g. 8 = 32 sequences per batch).
    batch_size_by_model : dict, optional
        Per-model override: {model_key: batch_size}. Overrides batch_size for that model.
    save_annotated : bool
        If True, save the DataFrame returned by add_epistasis_metrics to
        output_base/source_name/{model_key}_annotated.{parquet|csv}. Default True.
    annotated_format : str
        "parquet" or "csv". Default "parquet".
    """
    from genebeddings import VariantEmbeddingDB
    from genebeddings.genebeddings import add_epistasis_metrics

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    if model_keys is None and env_profile is not None:
        model_keys = get_model_keys_for_env(env_profile)
        logger.info("Env profile %r -> models: %s", env_profile, model_keys)
    model_keys = model_keys or DEFAULT_MODEL_KEYS
    model_keys = list(model_keys)
    sources = _sort_sources_first_null(sources)

    null_cov_dir = output_base / "null_cov"
    null_cov_computed = False
    okgp_names = [n for n, _ in sources if _is_cov_inv_source(n)]
    seen_okgp: set = set()

    for source_name, payload in sources:
        if isinstance(payload, pd.DataFrame):
            df = payload.copy()
        else:
            df = _load_epistasis_df(payload, id_col=id_col)

        n = len(df)
        logger.info("Source %r: %d rows (id_col=%r)", source_name, n, id_col)
        if n == 0:
            logger.warning("Skipping empty source %r", source_name)
            continue

        out_dir = output_base / source_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # For non-okgp sources we pass cov_inv (from okgp) into add_epistasis_metrics (must be computed after okgp sources are done)
        if not _is_cov_inv_source(source_name) and not null_cov_computed:
            logger.warning("Cov_inv not yet computed; process okgp sources first. Epistasis metrics will not include Mahalanobis (epi_mahal, etc.).")

        models_for_source = (
            list(source_model_map[source_name])
            if source_model_map and source_name in source_model_map
            else model_keys
        )
        for model_key in models_for_source:
            if model_key not in FULL_MODEL_CONFIG:
                logger.warning("Unknown model key %r, skipping", model_key)
                continue
            # SpliceAI only for sources whose name contains "fas" (when not using source_model_map)
            if (
                model_key == "spliceai"
                and source_model_map is None
                and not _spliceai_run_for_source(source_name)
            ):
                continue
            context, init_spec = FULL_MODEL_CONFIG[model_key]
            try:
                model = _build_model(model_key, init_spec, spliceai_model_dir)
            except Exception as e:
                logger.warning("Skip model %r: %s", model_key, e)
                continue
            if model is None:
                continue
            db_path = out_dir / f"{model_key}.db"
            logger.info("Model %s -> %s (context=%s)", model_key, db_path, context)

            # Load cov_inv (from okgp sources) for non-okgp sources. genebeddings.add_epistasis_metrics uses it
            # to compute epi_mahal (Mahalanobis distance of residual) and mahal_obs, mahal_add,
            # mahal_ratio, log_mahal_ratio (see genebeddings.genebeddings.add_epistasis_metrics).
            cov_inv = None
            if not _is_cov_inv_source(source_name):
                null_pack = null_cov_dir / f"{model_key}_pack.npz"
                if null_pack.exists():
                    try:
                        data = np.load(null_pack, allow_pickle=True)
                        cov_inv = np.asarray(data["cov_inv"], dtype=np.float64)
                        if cov_inv.ndim != 2 or cov_inv.shape[0] != cov_inv.shape[1]:
                            logger.warning("Invalid null cov_inv shape for %r: %s", model_key, cov_inv.shape)
                            cov_inv = None
                    except Exception as e:
                        logger.warning("Could not load null cov_inv for %r: %s", model_key, e)

            bs = (batch_size_by_model or {}).get(model_key, batch_size)
            db = VariantEmbeddingDB(str(db_path))
            try:
                if embedding_lookup_bases and not force:
                    n_looked_up = _copy_embeddings_from_lookup_dbs(
                        df, id_col, db, source_name, model_key, embedding_lookup_bases,
                        embedding_lookup_flat=embedding_lookup_flat,
                    )
                    if n_looked_up:
                        logger.info(
                            "Copied %d epistasis IDs from lookup DBs into %s (source=%s, model=%s)",
                            n_looked_up, db_path, source_name, model_key,
                        )
                annotated = add_epistasis_metrics(
                    df,
                    db,
                    model=model,
                    id_col=id_col,
                    context=context,
                    genome=genome,
                    show_progress=show_progress,
                    force=force,
                    batch_size=bs,
                    cov_inv=cov_inv,
                )
            finally:
                db.close()

            if save_annotated and annotated is not None:
                annotated_path = out_dir / f"{model_key}_annotated"
                try:
                    if annotated_format == "parquet":
                        annotated_path = annotated_path.with_suffix(".parquet")
                        annotated.to_parquet(annotated_path, index=False)
                    else:
                        annotated_path = annotated_path.with_suffix(".csv")
                        annotated.to_csv(annotated_path, index=False)
                    logger.info("Saved annotated DataFrame to %s", annotated_path)
                except Exception as e:
                    logger.warning("Could not save annotated DataFrame to %s: %s", annotated_path, e)
                    try:
                        annotated_path = out_dir / f"{model_key}_annotated.csv"
                        annotated.to_csv(annotated_path, index=False)
                        logger.info("Saved annotated DataFrame to %s (fallback)", annotated_path)
                    except Exception as e2:
                        logger.warning("Fallback save also failed: %s", e2)

        # After finishing all okgp sources: compute cov_inv from okgp source DBs and save for non-okgp sources
        if _is_cov_inv_source(source_name):
            seen_okgp.add(source_name)
        if okgp_names and seen_okgp >= set(okgp_names) and not null_cov_computed:
            run_covariance_and_save(
                output_base,
                okgp_names,
                model_keys=model_keys,
                out_npz_dir=null_cov_dir,
                method="ledoit_wolf",
                show_progress=show_progress,
            )
            null_cov_computed = True


def run_sources_by_tool(
    sources: Sequence[Tuple[str, Union[str, Path, pd.DataFrame]]],
    output_base: Union[str, Path],
    model_keys: Optional[Sequence[str]] = None,
    env_profile: Optional[str] = None,
    splicing_sources: Optional[set] = None,
    source_model_map: SourceModelMap = None,
    embedding_lookup_bases: Optional[Sequence[Union[str, Path]]] = None,
    embedding_lookup_flat: bool = False,
    spliceai_model_dir: Optional[str] = None,
    id_col: str = "epistasis_id",
    genome: str = "hg38",
    show_progress: bool = True,
    force: bool = False,
    batch_size: int = 8,
    batch_size_by_model: Optional[Dict[str, int]] = None,
    save_annotated: bool = True,
    annotated_format: str = "parquet",
    status_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Process one tool at a time: for each model, load once and run over all sources (null first).
    Minimizes model load/unload and matches cluster usage (one env, one tool, all sources).
    Same output layout as run_sources: output_base/source_name/model_key.db and null_cov.
    If status_path is set (or PIPELINE_STATUS_FILE), writes progress for monitoring.
    """
    from genebeddings import VariantEmbeddingDB
    from genebeddings.genebeddings import add_epistasis_metrics

    try:
        from notebooks.processing.pipeline_status import write_status as _write_status
    except Exception:
        _write_status = lambda **kw: None  # noqa: E731

    def _status(**kwargs):
        if status_path is not None:
            _write_status(path=Path(status_path), phase="embed", env_profile=env_profile, **kwargs)

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    if model_keys is None and env_profile is not None:
        model_keys = get_model_keys_for_env(env_profile)
        logger.info("Env profile %r -> models: %s", env_profile, model_keys)
    model_keys = model_keys or DEFAULT_MODEL_KEYS
    model_keys = list(model_keys)
    sources_ordered = _sort_sources_first_null(sources)
    null_cov_dir = output_base / "null_cov"
    null_cov_dir.mkdir(parents=True, exist_ok=True)
    batch_size_by_model = batch_size_by_model or {}
    n_tools = len([k for k in model_keys if k in FULL_MODEL_CONFIG])

    for tool_idx, model_key in enumerate(model_keys):
        if model_key not in FULL_MODEL_CONFIG:
            logger.warning("Unknown model key %r, skipping", model_key)
            continue
        context, init_spec = FULL_MODEL_CONFIG[model_key]
        try:
            model = _build_model(model_key, init_spec, spliceai_model_dir)
        except Exception as e:
            logger.warning("Skip model %r: %s", model_key, e)
            continue
        if model is None:
            continue
        bs = batch_size_by_model.get(model_key, batch_size)
        logger.info("Tool %r: processing all sources (batch_size=%s)", model_key, bs)
        _status(
            model_key=model_key,
            source=None,
            tools_done=tool_idx,
            tools_total=n_tools,
            message=f"embed {model_key}",
        )

        for source_name, payload in sources_ordered:
            models_for_source = (
                list(source_model_map[source_name])
                if source_model_map and source_name in source_model_map
                else model_keys
            )
            if model_key not in models_for_source:
                continue
            if (
                model_key == "spliceai"
                and source_model_map is None
                and not _spliceai_run_for_source(source_name)
            ):
                continue
            if isinstance(payload, pd.DataFrame):
                df = payload.copy()
            else:
                df = _load_epistasis_df(payload, id_col=id_col)
            if len(df) == 0:
                continue
            _status(model_key=model_key, source=source_name, n_total=len(df), message=f"embed {model_key} -> {source_name}")
            out_dir = output_base / source_name
            out_dir.mkdir(parents=True, exist_ok=True)
            db_path = out_dir / f"{model_key}.db"
            cov_inv = None
            if not _is_cov_inv_source(source_name):
                null_pack = null_cov_dir / f"{model_key}_pack.npz"
                if null_pack.exists():
                    try:
                        data = np.load(null_pack, allow_pickle=True)
                        cov_inv = np.asarray(data["cov_inv"], dtype=np.float64)
                        if cov_inv.ndim != 2 or cov_inv.shape[0] != cov_inv.shape[1]:
                            cov_inv = None
                    except Exception:
                        cov_inv = None
            db = VariantEmbeddingDB(str(db_path))
            try:
                if embedding_lookup_bases and not force:
                    _copy_embeddings_from_lookup_dbs(
                        df, id_col, db, source_name, model_key, embedding_lookup_bases,
                        embedding_lookup_flat=embedding_lookup_flat,
                    )
                annotated = add_epistasis_metrics(
                    df,
                    db,
                    model=model,
                    id_col=id_col,
                    context=context,
                    genome=genome,
                    show_progress=show_progress,
                    force=force,
                    batch_size=bs,
                    cov_inv=cov_inv,
                )
            finally:
                db.close()
            if save_annotated and annotated is not None:
                ap = out_dir / f"{model_key}_annotated"
                ap = ap.with_suffix(".parquet" if annotated_format == "parquet" else ".csv")
                try:
                    if annotated_format == "parquet":
                        annotated.to_parquet(ap, index=False)
                    else:
                        annotated.to_csv(ap, index=False)
                except Exception as e:
                    logger.warning("Could not save annotated to %s: %s", ap, e)

        # After processing all sources for this model: build cov_inv from okgp sources and save
        okgp_names = [n for n, _ in sources_ordered if _is_cov_inv_source(n)]
        if okgp_names:
            run_covariance_and_save(
                output_base,
                okgp_names,
                model_keys=[model_key],
                out_npz_dir=null_cov_dir,
                method="ledoit_wolf",
                show_progress=show_progress,
            )
        else:
            logger.debug("No okgp sources for cov_inv; skipping null_cov save for %r", model_key)


def run_null_covariance_and_save(
    output_base: Union[str, Path],
    model_keys: Optional[Sequence[str]] = None,
    out_npz_dir: Optional[Union[str, Path]] = None,
    method: str = "ledoit_wolf",
    show_progress: bool = True,
) -> List[Path]:
    """
    Compute covariance from the **null** source DBs only and save per-model packs.
    These are the cov_inv values that should be passed to add_epistasis_metrics
    when processing non-null sources (for epi_mahal, mahal_obs, etc.).

    Writes to out_npz_dir / {model_key}_pack.npz (default: output_base / "null_cov").
    Returns list of saved .npz paths.
    """
    output_base = Path(output_base)
    out_npz_dir = Path(out_npz_dir) if out_npz_dir is not None else output_base / "null_cov"
    return run_covariance_and_save(
        output_base,
        [NULL_SOURCE_NAME],
        model_keys=model_keys,
        out_npz_dir=out_npz_dir,
        method=method,
        show_progress=show_progress,
    )


def run_covariance_and_save(
    output_base: Union[str, Path],
    source_names: Sequence[str],
    model_keys: Optional[Sequence[str]] = None,
    out_npz_dir: Optional[Union[str, Path]] = None,
    method: str = "ledoit_wolf",
    show_progress: bool = True,
) -> List[Path]:
    """
    For each model, load residuals from the given source DBs (output_base/source/model_key.db),
    fit a single cov/cov_inv with compute_cov_inv_from_paths_combined, and save
    {model_key}_pack.npz with keys: cov, cov_inv, model (str), pool (str).

    Use run_null_covariance_and_save to build null-only packs; those cov_inv are
    passed into add_epistasis_metrics when processing non-null sources.

    Returns list of saved .npz paths.
    """
    from genebeddings.epistasis_features import compute_cov_inv_from_paths_combined

    output_base = Path(output_base)
    out_npz_dir = Path(out_npz_dir) if out_npz_dir is not None else output_base
    out_npz_dir.mkdir(parents=True, exist_ok=True)
    model_keys = model_keys or list(FULL_MODEL_CONFIG)

    saved: List[Path] = []
    for model_key in model_keys:
        paths = []
        for src in source_names:
            p = output_base / src / f"{model_key}.db"
            if p.exists():
                paths.append(str(p))
        if not paths:
            logger.warning("No DBs found for model %r", model_key)
            continue
        try:
            cov, cov_inv = compute_cov_inv_from_paths_combined(
                paths,
                method=method,
                show_progress=show_progress,
            )
        except Exception as e:
            logger.warning("Covariance failed for %r: %s", model_key, e)
            continue
        npz_path = out_npz_dir / f"{model_key}_pack.npz"
        np.savez(
            npz_path,
            cov=cov,
            cov_inv=cov_inv,
            model=np.array(model_key),
            pool=np.array("mean"),
        )
        saved.append(npz_path)
        logger.info("Saved %s", npz_path)
    return saved


def compute_cov_inv(
    output_base: Union[str, Path],
    source_names: Optional[Sequence[str]] = None,
    *,
    source_df: Optional[pd.DataFrame] = None,
    source_col: str = "source",
    id_col: str = "epistasis_id",
    epistasis_ids: Optional[Sequence[str]] = None,
    model_keys: Optional[Sequence[str]] = None,
    method: str = "ledoit_wolf",
    show_progress: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute cov and cov_inv from embeddings in the given source DBs and return
    them directly (no disk write). Use a subset of your dataframe to define
    which sources (and optionally which epistasis_ids) to use for the fit.

    Parameters
    ----------
    output_base : path
        Base directory where source DBs live (output_base/source_name/model_key.db).
    source_names : list of str, optional
        Which source subdirs to use (e.g. ["null"] or ["null", "fas_analysis"]).
        Ignored if source_df is provided.
    source_df : pandas.DataFrame, optional
        Subset of your dataframe. Unique values in source_df[source_col] define
        the source names. If id_col is present, source_df[id_col] defines which
        epistasis_ids to use for the covariance fit (otherwise all IDs in those DBs).
    source_col : str, default "source"
        Column in source_df used to get source names.
    id_col : str, default "epistasis_id"
        Column in source_df used to get epistasis_ids when source_df is provided.
        Ignored if epistasis_ids is explicitly provided.
    epistasis_ids : sequence of str, optional
        If provided, fit covariance using only these epistasis IDs (must exist in
        the source DBs). Overrides any ids from source_df.
    model_keys : list of str, optional
        Which models to compute for. Default: all in FULL_MODEL_CONFIG.
    method : str, default "ledoit_wolf"
        Covariance estimator (passed to compute_cov_inv_from_paths_combined).
    show_progress : bool, default True

    Returns
    -------
    dict
        {model_key: (cov, cov_inv)}. Each cov and cov_inv is a numpy array.
    """
    from genebeddings.epistasis_features import compute_cov_inv_from_paths_combined

    output_base = Path(output_base)
    cov_epi_ids: Optional[List[str]] = None
    if source_df is not None:
        if source_col not in source_df.columns:
            raise ValueError(f"source_df must have column {source_col!r}")
        source_names = source_df[source_col].dropna().astype(str).unique().tolist()
        if epistasis_ids is not None:
            cov_epi_ids = list(epistasis_ids)
        elif id_col in source_df.columns:
            cov_epi_ids = source_df[id_col].dropna().astype(str).unique().tolist()
    elif epistasis_ids is not None:
        cov_epi_ids = list(epistasis_ids)
    if not source_names:
        raise ValueError("Provide source_names or source_df with at least one source")

    model_keys = model_keys or list(FULL_MODEL_CONFIG)
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for model_key in model_keys:
        paths = []
        for src in source_names:
            p = output_base / src / f"{model_key}.db"
            if p.exists():
                paths.append(str(p))
        if not paths:
            logger.warning("No DBs found for model %r in sources %s", model_key, source_names)
            continue
        try:
            cov, cov_inv = compute_cov_inv_from_paths_combined(
                paths,
                epistasis_ids=cov_epi_ids,
                method=method,
                show_progress=show_progress,
            )
            result[model_key] = (cov, cov_inv)
        except Exception as e:
            logger.warning("Covariance failed for %r: %s", model_key, e)

    return result


def recompute_metrics_with_cov_inv(
    output_base: Union[str, Path],
    df: pd.DataFrame,
    cov_inv_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
    source_col: str = "source",
    model_keys: Optional[Sequence[str]] = None,
    id_col: str = "epistasis_id",
    genome: str = "hg38",
    spliceai_model_dir: Optional[str] = None,
    batch_size: int = 8,
    show_progress: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Recompute epistasis metric columns using existing embeddings in the new DBs
    and the supplied cov_inv per model. No embedding recomputation; only metrics
    (including Mahalanobis from cov_inv) are updated.

    Returns a series of tables: one DataFrame per model (tool). Each table has
    the same structure: id_col, source_col, and the same metric columns
    (len_WT_M1, epi_R_raw, epi_mahal, etc.). Only the tool differs. Use the
    dict key (model_key) to identify which table is which.

    Use after processing has filled output_base/source/model_key.db and you have
    computed cov_inv (e.g. from a chosen source and set of epistasis_ids via
    compute_cov_inv). Supply the full dataframe with epistasis_id and source.

    Parameters
    ----------
    output_base : path
        Base directory (output_base/source_name/model_key.db).
    df : pandas.DataFrame
        Full table with id_col and source_col (e.g. epistasis_id, source).
    cov_inv_by_model : dict
        {model_key: (cov, cov_inv)} from compute_cov_inv.
    source_col : str, default "source"
        Column that defines the source (and DB subdirectory).
    model_keys : list of str, optional
        Which models to recompute for. Default: keys of cov_inv_by_model.
    id_col : str, default "epistasis_id"
        Epistasis ID column.
    genome : str, default "hg38"
    spliceai_model_dir : str, optional
        For SpliceAI model build.
    batch_size : int, default 8
        Passed to add_epistasis_metrics.
    show_progress : bool, default True

    Returns
    -------
    dict
        {model_key: annotated_df}. Each annotated_df has the same columns
        (id_col, source_col, len_WT_M1, epi_R_raw, epi_mahal, ...); one table
        per tool for easy comparison and downstream use.
    """
    from genebeddings import VariantEmbeddingDB
    from genebeddings.genebeddings import add_epistasis_metrics

    output_base = Path(output_base)
    if source_col not in df.columns or id_col not in df.columns:
        raise ValueError(f"df must have columns {source_col!r} and {id_col!r}")

    model_keys = model_keys or list(cov_inv_by_model)
    result: Dict[str, pd.DataFrame] = {}
    sources = df[source_col].dropna().astype(str).unique().tolist()

    for model_key in model_keys:
        if model_key not in cov_inv_by_model:
            logger.warning("No cov_inv for model %r, skipping", model_key)
            continue
        if model_key not in FULL_MODEL_CONFIG:
            logger.warning("Unknown model key %r, skipping", model_key)
            continue
        cov, cov_inv = cov_inv_by_model[model_key]
        context, init_spec = FULL_MODEL_CONFIG[model_key]
        try:
            model = _build_model(model_key, init_spec, spliceai_model_dir)
        except Exception as e:
            logger.warning("Could not build model %r: %s", model_key, e)
            continue
        if model is None:
            continue

        parts: List[pd.DataFrame] = []
        for source_name in sources:
            db_path = output_base / source_name / f"{model_key}.db"
            if not db_path.exists():
                logger.warning("DB not found for %s / %s, skipping source", source_name, model_key)
                continue
            df_s = df.loc[df[source_col].astype(str) == source_name].copy()
            if len(df_s) == 0:
                continue
            db = VariantEmbeddingDB(str(db_path))
            try:
                annotated_s = add_epistasis_metrics(
                    df_s,
                    db,
                    model=model,
                    id_col=id_col,
                    context=context,
                    genome=genome,
                    cov_inv=cov_inv,
                    force=False,
                    batch_size=batch_size,
                    show_progress=show_progress,
                )
                parts.append(annotated_s)
            finally:
                db.close()

        if parts:
            result[model_key] = pd.concat(parts, axis=0).sort_index()
            logger.info("Recomputed metrics for model %r: %d rows", model_key, len(result[model_key]))

    return result


def main() -> int:
    try:
        from notebooks.paper_data_config import embeddings_dir
        _default_output = str(embeddings_dir())
    except Exception:
        _default_output = "embeddings"

    parser = argparse.ArgumentParser(
        description="Process epistasis IDs per source into per-model databases (null first); optional covariance save."
    )
    parser.add_argument("--sources", type=str, nargs="+", metavar="NAME:PATH")
    parser.add_argument("--output", type=str, default=_default_output, help="Output base dir for DBs (default: EPISTASIS_PAPER_ROOT/embeddings)")
    parser.add_argument(
        "--env-profile",
        type=str,
        default=None,
        help="Conda env profile name (e.g. nt, borzoi, spliceai, evo2, alphagenome, all)",
    )
    parser.add_argument("--models", type=str, nargs="*", default=None, metavar="KEY", help="Override: run these model keys (ignored if --env-profile set)")
    parser.add_argument("--id-col", type=str, default="epistasis_id")
    parser.add_argument("--batch-size", type=int, default=8, help="add_epistasis_metrics batch_size")
    parser.add_argument("--spliceai-dir", type=str, default=None, help="OpenSpliceAI model_dir")
    parser.add_argument(
        "--embedding-lookup-bases",
        type=str,
        nargs="*",
        default=None,
        metavar="DIR",
        help="Directories to search for existing embeddings (layout: base/source/model.db, or base/model.db if --embedding-lookup-flat).",
    )
    parser.add_argument(
        "--embedding-lookup-flat",
        action="store_true",
        help="Lookup bases contain .db files directly (base/model_key.db); no source subdirs.",
    )
    parser.add_argument("--covariance", action="store_true", help="After processing, compute and save cov/cov_inv npz")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-save-annotated", action="store_true", help="Do not save annotated DataFrames (default: save to {model_key}_annotated.parquet)")
    parser.add_argument("--annotated-format", type=str, default="parquet", choices=("parquet", "csv"), help="Format for annotated DataFrames (default: parquet)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.sources:
        parser.error("At least one --sources NAME:PATH is required")

    pairs = []
    for s in args.sources:
        if ":" not in s:
            parser.error(f"Each source must be NAME:PATH, got {s!r}")
        name, path = s.split(":", 1)
        pairs.append((name.strip(), path.strip()))

    run_sources(
        pairs,
        output_base=Path(args.output),
        model_keys=args.models if not args.env_profile else None,
        env_profile=args.env_profile,
        embedding_lookup_bases=args.embedding_lookup_bases,
        embedding_lookup_flat=args.embedding_lookup_flat,
        spliceai_model_dir=args.spliceai_dir,
        id_col=args.id_col,
        show_progress=not args.no_progress,
        force=args.force,
        batch_size=args.batch_size,
        save_annotated=not args.no_save_annotated,
        annotated_format=args.annotated_format,
    )

    if args.covariance:
        source_names = [n for n, _ in _sort_sources_first_null(pairs)]
        cov_model_keys = None
        if args.env_profile:
            cov_model_keys = get_model_keys_for_env(args.env_profile)
        elif args.models:
            cov_model_keys = args.models
        run_covariance_and_save(
            args.output,
            source_names,
            model_keys=cov_model_keys,
            out_npz_dir=Path(args.output),
            show_progress=not args.no_progress,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
