"""
Single root for all epistasis paper data (embeddings, dataframes, CSVs).

All notebooks and the processing pipeline use this module so that everything
is stored under one directory. Override via env EPISTASIS_PAPER_ROOT.
Default: /tamir2/nicolaslynn/data/epistasis_paper

Directory layout:
  EPISTASIS_PAPER_ROOT/
    data/                    # data_dir() — input and generated CSVs
      null_epistasis.csv
      fas_subset.csv, mst1r_subset.csv, kras_neighborhood_doubles.csv, ...
      mrna_folding_epistasis.csv, epistasis_aggregated.csv, ...
    embeddings/              # embeddings_dir() — output_base for process_epistasis
      null/                  # null source DBs per model
      null_cov/              # null covariance packs (*_pack.npz)
      fas_analysis/, kras/, mst1r_analysis/, mrna_folding/, ...
        {model_key}.db
        {model_key}_annotated.parquet
"""

import os
from pathlib import Path

_DEFAULT_ROOT = Path("/tamir2/nicolaslynn/data/epistasis_paper")

EPISTASIS_PAPER_ROOT: Path = Path(
    os.environ.get("EPISTASIS_PAPER_ROOT", _DEFAULT_ROOT)
).resolve()


def data_dir() -> Path:
    """Directory for all input and generated CSVs (subset files, aggregated table)."""
    p = EPISTASIS_PAPER_ROOT / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def embeddings_dir() -> Path:
    """Directory for embedding DBs and covariance packs (output_base for process_epistasis)."""
    p = EPISTASIS_PAPER_ROOT / "embeddings"
    p.mkdir(parents=True, exist_ok=True)
    return p
