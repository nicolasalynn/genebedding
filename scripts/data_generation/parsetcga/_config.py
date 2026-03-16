"""Path and default configuration for TCGA data."""
import os
from pathlib import Path

# Default data directory: use EPISTASIS_PAPER_ROOT/data/source if available,
# otherwise fall back to sibling data/ directory.
_CLUSTER_ROOT = Path("/tamir2/nicolaslynn/data/epistasis_paper")
_HOME_ROOT = Path.home() / "data" / "epistasis_paper"
_PAPER_ROOT = _CLUSTER_ROOT if _CLUSTER_ROOT.parent.exists() else _HOME_ROOT
_PAPER_SOURCE = Path(os.environ.get("EPISTASIS_PAPER_ROOT", str(_PAPER_ROOT))) / "data" / "source"
_FALLBACK_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_DATA_DIR = _PAPER_SOURCE if _PAPER_SOURCE.exists() else _FALLBACK_DIR

CLINICAL_PICKLE = "df_p_all.pkl"
MUTATIONS_PARQUET = "tcga_all.parquet"


def get_data_dir() -> Path:
    """Return data directory; can be overridden via TCGADATA_DIR env var."""
    return Path(os.environ.get("TCGADATA_DIR", str(_DEFAULT_DATA_DIR)))


def get_clinical_path() -> Path:
    return get_data_dir() / CLINICAL_PICKLE


def get_mutations_path() -> Path:
    return get_data_dir() / MUTATIONS_PARQUET
