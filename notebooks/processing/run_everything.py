#!/usr/bin/env python3
"""
Single entry point for the epistasis pipeline.

  Phase embed:  Run one env profile (one tool at a time over all sources).
                Invoke once per conda env (nt, borzoi, evo2, alphagenome, ...).
  Phase metrics: Compute cov_inv from null, recompute all metrics, save one
                 parquet per tool (sheets). Run once after all embed phases.

Usage:
  # On Lambda: one env at a time (run from repo root)
  conda activate nt && python -m notebooks.processing.run_everything --phase embed --env-profile nt
  conda activate evo2 && python -m notebooks.processing.run_everything --phase embed --env-profile evo2
  conda activate alphagenome && python -m notebooks.processing.run_everything --phase embed --env-profile alphagenome
  python -m notebooks.processing.run_everything --phase metrics

  # Or use scripts/run_pipeline_cluster.sh to run all phases.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Repo root on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebooks.paper_data_config import data_dir, embeddings_dir
from notebooks.processing.pipeline_config import (
    SOURCE_COL,
    ID_COL,
    SOURCE_MODEL_MAP,
    COV_INV_SOURCE_NAMES,
    SPLICEAI_SOURCE_NAMES,
    get_batch_size,
    get_single_dataframe_path,
    resolve_sources,
)
from notebooks.processing.process_epistasis import (
    run_sources_by_tool,
    run_from_single_dataframe,
    compute_cov_inv,
    recompute_metrics_with_cov_inv,
    get_model_keys_for_env,
    FULL_MODEL_CONFIG,
)
from notebooks.processing.pipeline_status import write_status as _write_status

logger = logging.getLogger(__name__)


def _status_path(output_base: Path):
    import os
    p = os.environ.get("PIPELINE_STATUS_FILE")
    return Path(p) if p else output_base / "pipeline_status.json"


def _get_batch_size_by_model(model_keys):
    return {k: get_batch_size(k) for k in model_keys}


def _truncate_df(df, n_rows, source_col, full_sources=None):
    """Truncate DataFrame to n_rows per source, keeping full_sources intact.

    Parameters
    ----------
    df : DataFrame
    n_rows : int
        Max rows per source (0 = no truncation).
    source_col : str
    full_sources : set of str, optional
        Source names to keep in full (no truncation).
    """
    import pandas as pd
    if not n_rows:
        return df
    full_sources = set(full_sources or [])
    is_full = df[source_col].isin(full_sources)
    truncated = df[~is_full].groupby(source_col, dropna=False).head(n_rows)
    return pd.concat([df[is_full], truncated]).reset_index(drop=True)


# Sources that are kept in full during --smoke-test-full (all rows, not truncated).
# fas_exon is the primary splicing analysis source; run it fully to validate
# SpliceAI and other models on real splice-site epistasis at scale.
SMOKE_FULL_SOURCES = set(SPLICEAI_SOURCE_NAMES)  # {"fas_exon"}


def run_embed_phase(
    env_profile: str,
    output_base: Path,
    spliceai_model_dir=None,
    force: bool = False,
    quick_test: bool = False,
    smoke_test: bool = False,
    smoke_test_full: bool = False,
    model_key: Optional[str] = None,
    sources_filter: Optional[list] = None,
    status_path: Optional[Path] = None,
    batch_size_override: Optional[int] = None,
) -> None:
    """Run embedding for one env profile (one tool at a time over all sources).

    quick_test:      first tool only, 20 rows per source (fast sanity check).
    smoke_test:      ALL tools, 5 rows per source (verify every model end-to-end).
    smoke_test_full: ALL tools, 5 rows per source + ALL rows for fas_exon.
    model_key:       run only this model key (must belong to env_profile).
    sources_filter:  run only these source names (e.g. ["okgp_chr12", "fas_exon"]).
    """
    status_path = status_path or _status_path(output_base)
    _write_status(path=status_path, phase="embed", env_profile=env_profile, message=f"embed env={env_profile}")

    model_keys = get_model_keys_for_env(env_profile)
    if model_key:
        if model_key not in model_keys:
            raise ValueError(
                f"Model key {model_key!r} not in env profile {env_profile!r}. "
                f"Available: {model_keys}"
            )
        model_keys = [model_key]
        logger.info("Filtering to single model key: %s", model_key)
    if quick_test:
        model_keys = model_keys[:1]
        logger.info("Quick-test: running only first tool %r", model_keys[0] if model_keys else None)
    elif smoke_test or smoke_test_full:
        label = "smoke-test-full" if smoke_test_full else "smoke-test"
        logger.info("%s: running ALL %d tools for env %r", label, len(model_keys), env_profile)

    n_rows = 20 if quick_test else (5 if (smoke_test or smoke_test_full) else 0)
    full_sources = SMOKE_FULL_SOURCES if smoke_test_full else None

    batch_size_by_model = _get_batch_size_by_model(model_keys)
    if batch_size_override is not None:
        batch_size_by_model = {k: batch_size_override for k in model_keys}
        logger.info("Batch size override: %d for all models", batch_size_override)
    single_path = get_single_dataframe_path(data_dir)
    if single_path is not None:
        import pandas as pd
        df = pd.read_csv(single_path, sep=None, engine="python")
        if sources_filter:
            df = df[df[SOURCE_COL].isin(sources_filter)]
            logger.info("Filtered to sources %s (%d rows)", sources_filter, len(df))
        if n_rows:
            df = _truncate_df(df, n_rows, SOURCE_COL, full_sources=full_sources)
            if full_sources:
                logger.info("Truncated to %d rows per source, full for %s (%d rows total)", n_rows, full_sources, len(df))
            else:
                logger.info("Truncated to %d rows per source (%d rows total)", n_rows, len(df))
        run_from_single_dataframe(
            df,
            output_base=output_base,
            source_col=SOURCE_COL,
            model_keys=model_keys,
            env_profile=env_profile,
            source_model_map=SOURCE_MODEL_MAP,
            id_col=ID_COL,
            show_progress=True,
            force=force,
            batch_size=16,
            batch_size_by_model=batch_size_by_model,
            use_by_tool=True,
            status_path=status_path,
        )
        return
    sources = resolve_sources(data_dir)
    if not sources:
        raise ValueError("No sources resolved. Set SOURCES in pipeline_config or use single dataframe.")
    if sources_filter:
        sources = [(name, path) for name, path in sources if name in sources_filter]
        logger.info("Filtered to sources: %s", [s[0] for s in sources])
    if n_rows:
        import pandas as pd
        truncated = []
        for name, path in sources:
            if full_sources and name in full_sources:
                if isinstance(path, pd.DataFrame):
                    truncated.append((name, path))
                else:
                    truncated.append((name, pd.read_csv(path, sep=None, engine="python")))
            elif isinstance(path, pd.DataFrame):
                truncated.append((name, path.head(n_rows)))
            else:
                truncated.append((name, pd.read_csv(path, nrows=n_rows)))
        sources = truncated
        logger.info("Truncated to %d rows per source%s", n_rows,
                     f" (full for {full_sources})" if full_sources else "")

    run_sources_by_tool(
        sources,
        output_base=output_base,
        model_keys=model_keys,
        source_model_map=SOURCE_MODEL_MAP,
        spliceai_model_dir=spliceai_model_dir,
        id_col=ID_COL,
        show_progress=True,
        force=force,
        batch_size=16,
        batch_size_by_model=batch_size_by_model,
        save_annotated=True,
        annotated_format="parquet",
        status_path=status_path,
        env_profile=env_profile,
    )


def run_metrics_phase(
    output_base: Path,
    sheets_dir: Path,
    spliceai_model_dir=None,
    status_path: Optional[Path] = None,
    model_key: Optional[str] = None,
    env_profile: Optional[str] = None,
) -> None:
    """Compute cov_inv from null, recompute metrics, save one parquet per tool.

    Parameters
    ----------
    model_key : str, optional
        If set, only compute metrics for this model.
    env_profile : str, optional
        If set (and model_key is None), only compute metrics for models in this env profile.
    """
    status_path = status_path or _status_path(output_base)
    _write_status(path=status_path, phase="metrics", message="cov_inv + recompute + sheets")
    import pandas as pd
    single_path = get_single_dataframe_path(data_dir)
    if single_path is not None:
        df_full = pd.read_csv(single_path, sep=None, engine="python")
    else:
        sources = resolve_sources(data_dir)
        if not sources:
            raise ValueError("No sources; need single dataframe or SOURCES in config.")
        dfs = []
        for name, path in sources:
            df = pd.read_csv(path, sep=None, engine="python")
            df[SOURCE_COL] = name
            dfs.append(df)
        df_full = pd.concat(dfs, ignore_index=True)
    if SOURCE_COL not in df_full.columns or ID_COL not in df_full.columns:
        raise ValueError(f"DataFrame must have {SOURCE_COL!r} and {ID_COL!r}")
    # cov_inv from rows whose source is in COV_INV_SOURCE_NAMES (see pipeline_config)
    src = df_full[SOURCE_COL].astype(str)
    df_null = df_full[src.isin(COV_INV_SOURCE_NAMES)]
    if len(df_null) == 0:
        logger.warning("No rows with source in COV_INV_SOURCE_NAMES=%s; cov_inv will have no rows.", COV_INV_SOURCE_NAMES)
        df_null = df_full.head(1).copy()
    okgp_sources = [s for s in src.unique() if s in COV_INV_SOURCE_NAMES]

    # Resolve which models to run metrics for
    if model_key:
        model_keys_filter = [model_key]
    elif env_profile:
        model_keys_filter = get_model_keys_for_env(env_profile)
    else:
        model_keys_filter = None  # all models

    model_keys = [
        k for k in FULL_MODEL_CONFIG
        if any((output_base / src / f"{k}.db").exists() for src in okgp_sources)
    ]
    if model_keys_filter:
        model_keys = [k for k in model_keys if k in model_keys_filter]
        logger.info("Metrics phase filtered to models: %s", model_keys)
    if not model_keys:
        logger.warning("No model DBs found under output_base for okgp sources %s", okgp_sources)
        return
    cov_inv_by_model = compute_cov_inv(
        output_base,
        source_df=df_null,
        source_col=SOURCE_COL,
        id_col=ID_COL,
        model_keys=model_keys,
        method="ledoit_wolf",
        show_progress=True,
    )
    metrics_by_tool = recompute_metrics_with_cov_inv(
        output_base,
        df_full,
        cov_inv_by_model,
        source_col=SOURCE_COL,
        model_keys=list(cov_inv_by_model),
        id_col=ID_COL,
        spliceai_model_dir=spliceai_model_dir,
        show_progress=True,
    )
    sheets_dir.mkdir(parents=True, exist_ok=True)
    for tool, tbl in metrics_by_tool.items():
        out = sheets_dir / f"epistasis_metrics_{tool}.parquet"
        tbl.to_parquet(out, index=False)
        logger.info("Saved sheet %s (%d rows)", out.name, len(tbl))


def main() -> int:
    parser = argparse.ArgumentParser(description="Epistasis pipeline: embed (per env) or metrics (sheets)")
    parser.add_argument("--phase", choices=("embed", "metrics"), required=True)
    parser.add_argument("--env-profile", type=str, default=None,
                        help="Conda env profile name (e.g. nt, borzoi, spliceai, evo2, alphagenome, all)")
    parser.add_argument("--output", type=str, default=None, help="Output base for DBs (default: embeddings_dir())")
    parser.add_argument("--sheets-dir", type=str, default=None, help="Where to save parquet sheets (default: output/sheets)")
    parser.add_argument("--spliceai-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick-test", action="store_true", help="One tool, 20 rows per source; verify GPU and wrappers")
    parser.add_argument("--smoke-test", action="store_true", help="ALL tools, 5 rows per source; verify every model end-to-end")
    parser.add_argument("--smoke-test-full", action="store_true", help="ALL tools, 5 rows per source + ALL fas_exon rows; full splicing validation")
    parser.add_argument("--model-key", type=str, default=None, help="Run only this model key (must belong to --env-profile)")
    parser.add_argument("--sources", type=str, nargs="+", default=None, help="Run only these source names (e.g. okgp_chr12 fas_exon)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (default: per-model config in pipeline_config.py)")
    parser.add_argument("--status-file", type=str, default=None, help="Progress JSON path (default: output_base/pipeline_status.json)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    output_base = Path(args.output) if args.output else embeddings_dir()
    sheets_dir = Path(args.sheets_dir) if args.sheets_dir else output_base / "sheets"
    status_path = Path(args.status_file) if args.status_file else None

    if args.phase == "embed":
        if not args.env_profile:
            parser.error("--phase embed requires --env-profile")
        run_embed_phase(
            args.env_profile,
            output_base,
            spliceai_model_dir=args.spliceai_dir,
            force=args.force,
            quick_test=args.quick_test,
            smoke_test=args.smoke_test,
            smoke_test_full=args.smoke_test_full,
            model_key=args.model_key,
            sources_filter=args.sources,
            status_path=status_path,
            batch_size_override=args.batch_size,
        )
    else:
        run_metrics_phase(
            output_base, sheets_dir,
            spliceai_model_dir=args.spliceai_dir,
            status_path=status_path,
            model_key=args.model_key,
            env_profile=args.env_profile,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
