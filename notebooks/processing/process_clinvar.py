"""
ClinVar single-variant processing pipeline.

Given a DataFrame of single mutation IDs (mut_id column), computes embeddings
for each model and writes one SQLite database per model under:

    {output_base}/clinvar/{model_key}.db

Each variant gets three keys in the DB:
    {mut_id}       → delta embedding (h_mut - h_wt)
    {mut_id}|WT    → wild-type embedding
    {mut_id}|MUT   → mutant embedding

This mirrors process_epistasis.py but uses add_single_variant_metrics() for
individual variants instead of add_epistasis_metrics() for pairs.

Usage:
    python -m notebooks.processing.process_clinvar --phase embed --env-profile nt
    python -m notebooks.processing.process_clinvar --phase embed --env-profile borzoi
    python -m notebooks.processing.process_clinvar --phase metrics
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from notebooks.processing.process_epistasis import (
    FULL_MODEL_CONFIG,
    ENV_PROFILES,
    get_model_keys_for_env,
    _build_model,
)
from notebooks.processing.pipeline_config import get_batch_size

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bundled ClinVar data
# ---------------------------------------------------------------------------
CLINVAR_SOURCE_NAME = "clinvar"
_BUNDLED_CLINVAR_PATH = Path(__file__).resolve().parent / "data" / "clinvar_variants.tsv"

SOURCE_COL = "source"
ID_COL = "mut_id"
LABEL_COL = "label"


def get_bundled_clinvar_path() -> Optional[Path]:
    """Return path to bundled ClinVar variants file if it exists and has data."""
    if _BUNDLED_CLINVAR_PATH.exists():
        return _BUNDLED_CLINVAR_PATH
    return None


def load_clinvar_df(
    path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load ClinVar variants from a TSV/CSV file.

    Parameters
    ----------
    path : path, optional
        Path to the variants file. If None, uses the bundled file.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least ``mut_id`` column. May also have ``source``
        and ``label`` columns.
    """
    if path is None:
        path = get_bundled_clinvar_path()
        if path is None:
            raise FileNotFoundError(
                "No ClinVar variants file found. Provide a path or populate "
                f"{_BUNDLED_CLINVAR_PATH}"
            )
    path = Path(path)
    df = pd.read_csv(path, sep=None, engine="python")
    if ID_COL not in df.columns:
        raise ValueError(
            f"ClinVar file must have column {ID_COL!r}. "
            f"Columns found: {list(df.columns)}"
        )
    return df


# ---------------------------------------------------------------------------
# Embedding phase: run single-variant embeddings across all models
# ---------------------------------------------------------------------------

def run_clinvar_embeddings(
    df: pd.DataFrame,
    output_base: Union[str, Path],
    model_keys: Optional[Sequence[str]] = None,
    env_profile: Optional[str] = None,
    id_col: str = ID_COL,
    genome: str = "hg38",
    show_progress: bool = True,
    force: bool = False,
    batch_size_by_model: Optional[Dict[str, int]] = None,
    save_annotated: bool = True,
    annotated_format: str = "parquet",
    status_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Compute and store single-variant embeddings for ClinVar mutations.

    For each model, creates ``{output_base}/clinvar/{model_key}.db`` and
    calls ``add_single_variant_metrics()`` which stores WT, MUT, and delta
    embeddings plus geometric metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Must have column ``id_col`` with mutation IDs in
        ``GENE:CHROM:POS:REF:ALT[:STRAND]`` format.
    output_base : path
        Root embeddings directory. ClinVar DBs go under ``output_base/clinvar/``.
    model_keys : list of str, optional
        Which models to run. Resolved from env_profile if None.
    env_profile : str, optional
        Conda env profile name (e.g. "nt", "borzoi"). Used to resolve model_keys.
    id_col : str
        Column with mutation IDs.
    genome : str
        Genome assembly.
    show_progress : bool
        Show tqdm progress bars.
    force : bool
        Recompute even if embeddings exist.
    batch_size_by_model : dict, optional
        Per-model batch size overrides.
    save_annotated : bool
        Save annotated DataFrame with metrics.
    annotated_format : str
        "parquet" or "csv".
    status_path : path, optional
        Progress JSON path for monitoring.
    """
    from genebeddings import VariantEmbeddingDB
    from genebeddings.genebeddings import add_single_variant_metrics

    output_base = Path(output_base)
    out_dir = output_base / CLINVAR_SOURCE_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_keys is None and env_profile is not None:
        model_keys = get_model_keys_for_env(env_profile)
        logger.info("Env profile %r -> models: %s", env_profile, model_keys)
    model_keys = model_keys or list(FULL_MODEL_CONFIG)
    model_keys = list(model_keys)
    batch_size_by_model = batch_size_by_model or {}

    n_variants = len(df)
    logger.info(
        "ClinVar embedding: %d variants, %d models, output=%s",
        n_variants, len(model_keys), out_dir,
    )

    for model_key in model_keys:
        if model_key not in FULL_MODEL_CONFIG:
            logger.warning("Unknown model key %r, skipping", model_key)
            continue
        # Skip SpliceAI for ClinVar (not meaningful for individual SNVs)
        if model_key == "spliceai":
            logger.info("Skipping SpliceAI for ClinVar (not applicable to single SNVs)")
            continue

        context, init_spec = FULL_MODEL_CONFIG[model_key]
        try:
            model = _build_model(model_key, init_spec)
        except Exception as e:
            logger.warning("Skip model %r: %s", model_key, e)
            continue
        if model is None:
            continue

        db_path = out_dir / f"{model_key}.db"
        logger.info("Model %s -> %s (context=%s)", model_key, db_path, context)

        db = VariantEmbeddingDB(str(db_path))
        try:
            annotated = add_single_variant_metrics(
                df,
                db,
                model=model,
                id_col=id_col,
                context=context,
                genome=genome,
                show_progress=show_progress,
                pool="mean",
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
                logger.warning("Could not save annotated to %s: %s", annotated_path, e)
                try:
                    fallback = out_dir / f"{model_key}_annotated.csv"
                    annotated.to_csv(fallback, index=False)
                    logger.info("Saved annotated (fallback) to %s", fallback)
                except Exception as e2:
                    logger.warning("Fallback save also failed: %s", e2)

        # Free model memory before loading next
        del model


# ---------------------------------------------------------------------------
# Metrics phase: recompute geometry from existing DBs (no model needed)
# ---------------------------------------------------------------------------

def recompute_clinvar_metrics(
    output_base: Union[str, Path],
    df: pd.DataFrame,
    model_keys: Optional[Sequence[str]] = None,
    id_col: str = ID_COL,
    show_progress: bool = True,
    sheets_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Recompute single-variant metrics from existing ClinVar DBs (no model needed).

    Loads WT and MUT embeddings from ``{output_base}/clinvar/{model_key}.db``
    and computes geometry. Saves one parquet per model to sheets_dir.

    Parameters
    ----------
    output_base : path
        Root embeddings directory.
    df : pd.DataFrame
        ClinVar variants with ``id_col`` column.
    model_keys : list of str, optional
        Which models. Default: auto-discover from available DBs.
    id_col : str
        Column with mutation IDs.
    show_progress : bool
    sheets_dir : path, optional
        Where to save metrics parquets. Default: ``output_base/sheets``.

    Returns
    -------
    dict
        {model_key: annotated_dataframe}.
    """
    from genebeddings import VariantEmbeddingDB
    from genebeddings.genebeddings import add_single_variant_metrics

    output_base = Path(output_base)
    clinvar_dir = output_base / CLINVAR_SOURCE_NAME

    if not clinvar_dir.exists():
        raise FileNotFoundError(f"ClinVar embedding directory not found: {clinvar_dir}")

    if model_keys is None:
        # Auto-discover from available .db files
        model_keys = [
            p.stem for p in sorted(clinvar_dir.glob("*.db"))
            if not p.stem.endswith("_annotated")
        ]
        logger.info("Auto-discovered models: %s", model_keys)

    if not model_keys:
        logger.warning("No ClinVar DBs found in %s", clinvar_dir)
        return {}

    if sheets_dir is None:
        sheets_dir = output_base / "sheets"
    sheets_dir = Path(sheets_dir)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, pd.DataFrame] = {}
    for model_key in model_keys:
        db_path = clinvar_dir / f"{model_key}.db"
        if not db_path.exists():
            logger.warning("No DB for %r at %s", model_key, db_path)
            continue

        logger.info("Recomputing metrics for %s", model_key)
        db = VariantEmbeddingDB(str(db_path))
        try:
            annotated = add_single_variant_metrics(
                df,
                db,
                model=None,  # No model needed, load from DB
                id_col=id_col,
                show_progress=show_progress,
                pool="mean",
            )
        finally:
            db.close()

        results[model_key] = annotated

        out_path = sheets_dir / f"clinvar_metrics_{model_key}.parquet"
        try:
            annotated.to_parquet(out_path, index=False)
            logger.info("Saved %s (%d rows)", out_path.name, len(annotated))
        except Exception as e:
            logger.warning("Could not save %s: %s", out_path, e)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from notebooks.paper_data_config import embeddings_dir

    parser = argparse.ArgumentParser(
        description="ClinVar single-variant pipeline: embed (per env) or metrics"
    )
    parser.add_argument(
        "--phase", choices=("embed", "metrics"), required=True,
    )
    parser.add_argument(
        "--env-profile", type=str, default=None,
        help="Conda env profile name (e.g. nt, borzoi, evo2, alphagenome, all)",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to ClinVar TSV/CSV file (default: bundled clinvar_variants.tsv)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output base (default: embeddings_dir())")
    parser.add_argument("--sheets-dir", type=str, default=None, help="Where to save metrics parquets")
    parser.add_argument("--model-key", type=str, default=None, help="Run only this model key")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick-test", action="store_true", help="First tool only, 20 rows")
    parser.add_argument("--smoke-test", action="store_true", help="All tools, 5 rows")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    output_base = Path(args.output) if args.output else embeddings_dir()
    sheets_dir = Path(args.sheets_dir) if args.sheets_dir else output_base / "sheets"

    df = load_clinvar_df(args.input)
    logger.info("Loaded %d ClinVar variants", len(df))

    if args.phase == "embed":
        if not args.env_profile:
            parser.error("--phase embed requires --env-profile")

        model_keys = get_model_keys_for_env(args.env_profile)
        if args.model_key:
            if args.model_key not in model_keys:
                parser.error(
                    f"Model key {args.model_key!r} not in env profile {args.env_profile!r}. "
                    f"Available: {model_keys}"
                )
            model_keys = [args.model_key]
        if args.quick_test:
            model_keys = model_keys[:1]
            df = df.head(20)
            logger.info("Quick-test: 1 tool, 20 rows")
        elif args.smoke_test:
            df = df.head(5)
            logger.info("Smoke-test: all tools, 5 rows")

        batch_size_by_model = {k: get_batch_size(k) for k in model_keys}
        if args.batch_size is not None:
            batch_size_by_model = {k: args.batch_size for k in model_keys}

        run_clinvar_embeddings(
            df,
            output_base=output_base,
            model_keys=model_keys,
            env_profile=args.env_profile,
            show_progress=True,
            force=args.force,
            batch_size_by_model=batch_size_by_model,
        )
    else:
        model_keys = None
        if args.model_key:
            model_keys = [args.model_key]
        elif args.env_profile:
            model_keys = get_model_keys_for_env(args.env_profile)

        recompute_clinvar_metrics(
            output_base,
            df,
            model_keys=model_keys,
            show_progress=True,
            sheets_dir=sheets_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
