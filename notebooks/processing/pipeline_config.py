"""
Single config for the epistasis pipeline: sources, tools, envs, batch sizes.

Use from run_everything.ipynb or run_everything.py. Output layout:
  {output_base}/{source}/{model_key}.db
  {output_base}/null_cov/{model_key}_pack.npz
  {sheets_dir}/epistasis_metrics_{model_key}.parquet   (one sheet per tool)

Execution: one tool at a time (one conda env per profile). Each tool runs over
all sources (null first) so null cov_inv is available for Mahalanobis in non-null.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Default batch_size for add_epistasis_metrics. Per-tool overrides below.
# batch_size = N => 4*N sequences per batch (WT, M1, M2, M12 per epistasis pair).
# Tune for A100: larger = faster but more VRAM. Conservative defaults.
DEFAULT_BATCH_SIZE = 16

# Per-tool batch_size (optional). Tools not listed use DEFAULT_BATCH_SIZE.
# Reduce for large-context or memory-heavy models (borzoi, alphagenome).
BATCH_SIZE_BY_TOOL: Dict[str, int] = {
    "nt50_3mer": 4,       # 3mer tokenisation → long token seqs, needs small batch
    "nt50_multi": 32,
    "nt100_multi": 32,
    "nt250_multi": 32,
    "nt500_multi": 32,
    "nt500_ref": 32,
    "nt2500_multi": 16,
    "nt2500_okgp": 16,
    "convnova": 24,
    "mutbert": 32,
    "hyenadna": 8,
    "caduceus": 8,
    "borzoi": 6,
    "alphagenome": 8,
    "evo2": 16,
    "rinalmo": 16,
    "specieslm": 16,
    "dnabert": 24,
    "spliceai": 8,
}

# Tool -> conda env name.  Every model must be explicitly mapped.
# NT variants all share the "nt" conda env.
TOOL_TO_ENV: Dict[str, str] = {
    "nt50_3mer": "nt",
    "nt50_multi": "nt",
    "nt100_multi": "nt",
    "nt250_multi": "nt",
    "nt500_multi": "nt",
    "nt500_ref": "nt",
    "nt2500_multi": "nt",
    "nt2500_okgp": "nt",
    "convnova": "convnova",
    "mutbert": "mutbert",
    "hyenadna": "hyenadna",
    "caduceus": "caduceus",
    "borzoi": "borzoi",
    "dnabert": "dnabert",
    "rinalmo": "rinalmo",
    "specieslm": "specieslm",
    "spliceai": "spliceai",
    "alphagenome": "alphagenome",
    "evo2": "evo2",
}


def get_env_for_tool(model_key: str) -> str:
    """Return conda env name for a model key.  Raises on unknown keys."""
    if model_key not in TOOL_TO_ENV:
        raise KeyError(
            f"Unknown model key {model_key!r}. "
            f"Add it to TOOL_TO_ENV in pipeline_config.py. "
            f"Known keys: {sorted(TOOL_TO_ENV)}"
        )
    return TOOL_TO_ENV[model_key]


def get_batch_size(model_key: str, default: Optional[int] = None) -> int:
    """Return batch_size for a model key."""
    if default is None:
        default = DEFAULT_BATCH_SIZE
    return BATCH_SIZE_BY_TOOL.get(model_key, default)


def get_tools_for_env(env_profile: str) -> List[str]:
    """Return list of model keys that run in the given env profile."""
    from notebooks.processing.process_epistasis import get_model_keys_for_env
    return get_model_keys_for_env(env_profile)


# ---------------------------------------------------------------------------
# Pipeline config: sources and optional overrides
# ---------------------------------------------------------------------------
# Bundled file: epistasis_id + source only (shipped with package).
_BUNDLED_PAIRS_PATH = Path(__file__).resolve().parent / "data" / "all_pairs_combined.tsv"

# Either use a single dataframe (all epistasis_ids + source column) or a list
# of (source_name, path_to_csv). Paths can be relative to data_dir().
# Default: use bundled all_pairs_combined.tsv if present.
USE_SINGLE_DATAFRAME = True
SINGLE_DATAFRAME_PATH: Optional[Union[str, Path]] = None  # None = use bundled path below
BUNDLED_PAIRS_PATH: Path = _BUNDLED_PAIRS_PATH

# When USE_SINGLE_DATAFRAME is False, list (source_name, path) with paths under data_dir().
SOURCES: List[Tuple[str, Union[str, Path]]] = [
    ("null", "null_epistasis.csv"),
    ("fas_analysis", "fas_subset.csv"),
    ("mst1r_analysis", "mst1r_subset.csv"),
    ("kras", "kras_subset.csv"),
]

# Optional: restrict which tools run per source. None = use all tools for env (with SpliceAI only on splicing sources).
# Example: {"null": ["nt500_multi", "convnova"], "fas_analysis": ["nt500_multi", "convnova", "spliceai"]}
SOURCE_MODEL_MAP: Optional[Dict[str, Sequence[str]]] = None

# Column names (must match the file)
SOURCE_COL = "source"
ID_COL = "epistasis_id"

# Source names exactly as they appear in the data file (from all_pairs_combined.tsv).
# Rows with source in COV_INV_SOURCE_NAMES are used to fit cov_inv (Mahalanobis background).
COV_INV_SOURCE_NAMES: List[str] = ["okgp_chr12"]
# Run SpliceAI only for rows whose source is in this list.
SPLICEAI_SOURCE_NAMES: List[str] = ["fas_exon"]


def resolve_sources(
    data_dir_fn,
) -> List[Tuple[str, Union[str, Path]]]:
    """Resolve source paths against data_dir. Returns list of (name, path)."""
    data_root = Path(data_dir_fn())
    if USE_SINGLE_DATAFRAME and SINGLE_DATAFRAME_PATH:
        return []  # Caller uses run_from_single_dataframe
    out = []
    for name, path in SOURCES:
        p = Path(path)
        if not p.is_absolute():
            p = data_root / p
        out.append((name, p))
    return out


def get_single_dataframe_path(data_dir_fn):
    """Return path for single dataframe mode, or None. Prefers bundled all_pairs_combined.tsv if present."""
    if not USE_SINGLE_DATAFRAME:
        return None
    if SINGLE_DATAFRAME_PATH is not None:
        p = Path(SINGLE_DATAFRAME_PATH)
        if not p.is_absolute():
            p = data_dir_fn() / p
        return p if p.exists() else None
    if BUNDLED_PAIRS_PATH.exists():
        return BUNDLED_PAIRS_PATH
    return None
