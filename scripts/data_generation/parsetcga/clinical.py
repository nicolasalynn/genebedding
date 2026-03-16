"""Clinical data loading and survival-ready preprocessing."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ._config import get_clinical_path


# Columns used to compute follow-up duration (days)
_DURATION_COL_PATTERNS = (
    "days_to_followup",
    "days_to_last_followup",
    "days_to_know_alive",
    "days_to_last_known_alive",
    "days_to_death",
)

# Standardized survival columns produced by prepare_clinical
SURVIVAL_COLS = ["duration", "event", "case_id"]
TREATMENT_FEATURES = [
    "chemotherapy",
    "hormone_therapy",
    "immunotherapy",
    "targeted_molecular_therapy",
]


def load_clinical_raw(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw clinical pickle. Uses package default path if path is None."""
    p = path or get_clinical_path()
    if not p.exists():
        raise FileNotFoundError(f"Clinical data not found: {p}")
    with open(p, "rb") as f:
        df = pickle.load(f)
    return df


def _duration_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        for pat in _DURATION_COL_PATTERNS:
            if pat in c:
                cols.append(c)
                break
    return cols


def prepare_clinical(
    df: Optional[pd.DataFrame] = None,
    path: Optional[Path] = None,
    duration_years: bool = True,
    min_duration: float = 0.0,
    subset_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Build a survival-ready clinical dataframe with duration and event.

    - Renames `patient_uuid` -> `case_id` for joining with mutation data.
    - `duration`: max of all follow-up/death day columns, in years if duration_years else days.
    - `event`: 1 if patient has days_to_death (deceased), else 0.
    - Drops rows with duration < min_duration.
    - If subset_columns is given, only those columns are kept (plus duration, event, case_id).
    """
    if df is None:
        df = load_clinical_raw(path).copy()
    else:
        df = df.copy()

    df = df.rename(columns={"patient_uuid": "case_id"})
    duration_cols = _duration_columns(df)
    if not duration_cols:
        raise ValueError("No duration-related columns found in clinical data.")

    # Event: 1 if deceased (has days_to_death)
    event_col = "event"
    death_col = "days_to_death"
    df.insert(1, event_col, (df[death_col].notna() & (df[death_col] >= 0)).astype(int))

    # Duration: max of all time columns
    duration_label = "duration"
    df.insert(1, duration_label, df[duration_cols].max(axis=1))
    if duration_years:
        df[duration_label] = df[duration_label] / 365.0

    df = df[df[duration_label] >= min_duration]

    if subset_columns is not None:
        keep = [c for c in [duration_label, event_col, "case_id"] + list(subset_columns) if c in df.columns]
        df = df[keep]

    return df


def clinical_with_survival(
    path: Optional[Path] = None,
    include_treatments: bool = True,
    min_duration: float = 0.0,
) -> pd.DataFrame:
    """
    Load and prepare clinical data with standard survival + optional treatment columns.
    Ideal for survival analysis and cohort definitions.
    """
    subset = []
    if include_treatments:
        subset.extend(TREATMENT_FEATURES)
        subset.append("Proj_name")
    return prepare_clinical(
        path=path,
        min_duration=min_duration,
        subset_columns=subset,
    )
