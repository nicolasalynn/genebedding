"""
Unified epistasis processing pipeline.

Given a DataFrame of epistasis IDs and a source name (e.g. fas_analysis,
mst1r_analysis, tcga_analysis, null), computes embeddings for each model
and writes one SQLite database per model under:

    {output_base}/{source_name}/{model_key}.db

The special source name 'null' is always processed first so that a
background distribution of embedding shifts exists for each model before
any other analyses.

SpliceAI runs only for "splicing" sources (fas_analysis, mst1r_analysis, kras).
AlphaGenome and all other models run for every source.

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

# SpliceAI runs only for these sources (splicing-related datasets)
SPLICING_SOURCES = {"fas_analysis", "mst1r_analysis", "kras"}

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
# Environment profiles: run only models that belong to the current environment
# ---------------------------------------------------------------------------
# Models that require the AlphaGenome environment (JAX, etc.). Run ONLY these in that env.
MODELS_ALPHAGENOME_ENV: List[str] = ["alphagenome"]

# Models that require the Evo2 environment. Run ONLY these in that env.
MODELS_EVO2_ENV: List[str] = ["evo2"]

# All other models (run in the shared "main" environment). Excludes alphagenome and evo2.
_MODELS_REQUIRING_SPECIAL_ENV = set(MODELS_ALPHAGENOME_ENV) | set(MODELS_EVO2_ENV)
MODELS_MAIN_ENV: List[str] = [
    k for k in FULL_MODEL_CONFIG
    if k not in _MODELS_REQUIRING_SPECIAL_ENV
]

# Named profiles: use get_model_keys_for_env(profile) or --env-profile in CLI
ENV_PROFILES: Dict[str, List[str]] = {
    "alphagenome": MODELS_ALPHAGENOME_ENV,
    "evo2": MODELS_EVO2_ENV,
    "main": MODELS_MAIN_ENV,
    "all": list(FULL_MODEL_CONFIG),
}


def get_model_keys_for_env(profile: str) -> List[str]:
    """
    Return the list of model keys to run for the given environment profile.

    Use this so the same notebook/script can be run in different conda envs:
    set ENV_PROFILE to "alphagenome" / "evo2" / "main" / "all" and pass
    model_keys=get_model_keys_for_env(ENV_PROFILE).

    Parameters
    ----------
    profile : str
        One of: "alphagenome", "evo2", "main", "all".

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
    # default
    _defaults = {
        "convnova": ("ConvNovaWrapper", {}),
        "mutbert": ("MutBERTWrapper", {}),
        "hyenadna": ("HyenaDNAWrapper", {}),
        "caduceus": ("CaduceusWrapper", {}),
        "borzoi": ("BorzoiWrapper", {}),
        "rinalmo": ("RiNALMoWrapper", {}),
        "specieslm": ("SpeciesLMWrapper", {}),
        "dnabert": ("DNABERTWrapper", {}),
    }
    cls_name, kwargs = _defaults.get(model_key, (None, None))
    if cls_name is None:
        return None
    from genebeddings.wrappers import (
        BorzoiWrapper,
        CaduceusWrapper,
        ConvNovaWrapper,
        DNABERTWrapper,
        HyenaDNAWrapper,
        MutBERTWrapper,
        RiNALMoWrapper,
        SpeciesLMWrapper,
    )
    cls_map = {
        "ConvNovaWrapper": ConvNovaWrapper,
        "MutBERTWrapper": MutBERTWrapper,
        "HyenaDNAWrapper": HyenaDNAWrapper,
        "CaduceusWrapper": CaduceusWrapper,
        "BorzoiWrapper": BorzoiWrapper,
        "RiNALMoWrapper": RiNALMoWrapper,
        "SpeciesLMWrapper": SpeciesLMWrapper,
        "DNABERTWrapper": DNABERTWrapper,
    }
    return cls_map[cls_name](**kwargs)


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
    """Return sources with 'null' first; rest in original order."""
    ordered = []
    null_item = None
    for name, payload in sources:
        if name == NULL_SOURCE_NAME:
            null_item = (name, payload)
        else:
            ordered.append((name, payload))
    if null_item is not None:
        ordered.insert(0, null_item)
    return ordered


def run_from_single_dataframe(
    df: pd.DataFrame,
    output_base: Union[str, Path],
    source_col: str = "source",
    model_keys: Optional[Sequence[str]] = None,
    env_profile: Optional[str] = None,
    splicing_sources: Optional[set] = None,
    source_model_map: SourceModelMap = None,
    spliceai_model_dir: Optional[str] = None,
    id_col: str = "epistasis_id",
    genome: str = "hg38",
    show_progress: bool = True,
    force: bool = False,
    batch_size: int = 8,
    save_annotated: bool = True,
    annotated_format: str = "parquet",
) -> None:
    """
    Run the full processing pipeline from a single DataFrame that contains all
    double variant IDs and a column indicating the source. Embedding storage
    directories are split by the value in `source_col`: each source gets
    `{output_base}/{source_value}/{model_key}.db`. Null is processed first if
    present so that null covariance is available for non-null sources.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have columns `id_col` (e.g. epistasis_id) and `source_col` (e.g. source).
        Each unique value in `source_col` defines a source; rows are grouped by it.
    output_base : path
        Base directory for outputs. Each source gets a subdir: output_base/source_name/.
    source_col : str, default "source"
        Column name whose values define the source (and thus the storage subdirectory).
    model_keys, env_profile, splicing_sources, source_model_map, spliceai_model_dir, id_col, genome,
    show_progress, force, batch_size, save_annotated, annotated_format
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
    run_sources(
        sources,
        output_base=output_base,
        model_keys=model_keys,
        env_profile=env_profile,
        splicing_sources=splicing_sources,
        source_model_map=source_model_map,
        spliceai_model_dir=spliceai_model_dir,
        id_col=id_col,
        genome=genome,
        show_progress=show_progress,
        force=force,
        batch_size=batch_size,
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
    spliceai_model_dir: Optional[str] = None,
    id_col: str = "epistasis_id",
    genome: str = "hg38",
    show_progress: bool = True,
    force: bool = False,
    batch_size: int = 8,
    save_annotated: bool = True,
    annotated_format: str = "parquet",
) -> None:
    """
    Process each source: for each model, compute/store epistasis embeddings
    and add metrics. Writes one .db per model under output_base/source_name/,
    and optionally the annotated DataFrame (with metric columns) to
    output_base/source_name/{model_key}_annotated.parquet (or .csv).
    SpliceAI is run only for sources in splicing_sources (default: fas_analysis, mst1r_analysis, kras).
    Use source_model_map to override which models run per source (see below).

    Parameters
    ----------
    sources : list of (source_name, path_or_dataframe)
    output_base : path
    model_keys : list of str, optional; which models to run. If None, resolved from env_profile or DEFAULT_MODEL_KEYS.
    env_profile : str, optional; if model_keys is None, use get_model_keys_for_env(env_profile). One of: "alphagenome", "evo2", "main", "all".
    splicing_sources : set of str, optional; sources for which SpliceAI is run; default SPLICING_SOURCES. Ignored when source_model_map is set.
    source_model_map : dict, optional; map source_name -> list of model keys for that source. If a source is missing, model_keys is used. Enables e.g. running SpliceAI only for certain sources or restricting null to a subset of models.
    spliceai_model_dir : str, optional; OpenSpliceAI checkpoint dir; else OPENSPLICEAI_MODEL_DIR.
    id_col : str
    genome : str
    show_progress : bool
    force : bool
    batch_size : int
        Batched embedding in add_epistasis_metrics (e.g. 8 = 32 sequences per batch).
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
    splicing_sources = splicing_sources if splicing_sources is not None else SPLICING_SOURCES
    sources = _sort_sources_first_null(sources)

    null_cov_dir = output_base / "null_cov"
    null_cov_computed = False

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

        # For non-null sources we pass null cov_inv into add_epistasis_metrics (must be computed after null is done)
        if source_name != NULL_SOURCE_NAME and not null_cov_computed:
            logger.warning("Null covariance not yet computed; run null source first. Epistasis metrics will not include Mahalanobis (epi_mahal, etc.).")

        models_for_source = (
            list(source_model_map[source_name])
            if source_model_map and source_name in source_model_map
            else model_keys
        )
        for model_key in models_for_source:
            if model_key not in FULL_MODEL_CONFIG:
                logger.warning("Unknown model key %r, skipping", model_key)
                continue
            # SpliceAI only for splicing-related sources (when not using source_model_map)
            if (
                model_key == "spliceai"
                and source_model_map is None
                and source_name not in splicing_sources
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

            # Load null cov_inv for non-null sources. genebeddings.add_epistasis_metrics uses it
            # to compute epi_mahal (Mahalanobis distance of residual) and mahal_obs, mahal_add,
            # mahal_ratio, log_mahal_ratio (see genebeddings.genebeddings.add_epistasis_metrics).
            # The residual is v12_obs - v12_exp (same as in epistasis_features.compute_core_vectors).
            cov_inv = None
            if source_name != NULL_SOURCE_NAME:
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

            db = VariantEmbeddingDB(str(db_path))
            try:
                annotated = add_epistasis_metrics(
                    df,
                    db,
                    model=model,
                    id_col=id_col,
                    context=context,
                    genome=genome,
                    show_progress=show_progress,
                    force=force,
                    batch_size=batch_size,
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

        # After finishing the null source: compute null covariance from null/*.db and save for use in non-null sources
        if source_name == NULL_SOURCE_NAME:
            run_null_covariance_and_save(
                output_base,
                models_for_source,
                out_npz_dir=null_cov_dir,
                method="ledoit_wolf",
                show_progress=show_progress,
            )
            null_cov_computed = True


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
    model_keys: Optional[Sequence[str]] = None,
    method: str = "ledoit_wolf",
    show_progress: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute cov and cov_inv from embeddings in the given source DBs and return
    them directly (no disk write). Use a subset of your dataframe to define
    which sources to use.

    Parameters
    ----------
    output_base : path
        Base directory where source DBs live (output_base/source_name/model_key.db).
    source_names : list of str, optional
        Which source subdirs to use (e.g. ["null"] or ["null", "fas_analysis"]).
        Ignored if source_df is provided.
    source_df : pandas.DataFrame, optional
        Subset of your dataframe. Unique values in source_df[source_col] define
        the source names. Use this to compute cov_inv from "a part of that df".
    source_col : str, default "source"
        Column in source_df used to get source names.
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
    if source_df is not None:
        if source_col not in source_df.columns:
            raise ValueError(f"source_df must have column {source_col!r}")
        source_names = source_df[source_col].dropna().astype(str).unique().tolist()
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
                method=method,
                show_progress=show_progress,
            )
            result[model_key] = (cov, cov_inv)
        except Exception as e:
            logger.warning("Covariance failed for %r: %s", model_key, e)

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
        choices=list(ENV_PROFILES),
        help="Run only models for this environment: alphagenome, evo2, main, or all",
    )
    parser.add_argument("--models", type=str, nargs="*", default=None, metavar="KEY", help="Override: run these model keys (ignored if --env-profile set)")
    parser.add_argument("--id-col", type=str, default="epistasis_id")
    parser.add_argument("--batch-size", type=int, default=8, help="add_epistasis_metrics batch_size")
    parser.add_argument("--spliceai-dir", type=str, default=None, help="OpenSpliceAI model_dir")
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
