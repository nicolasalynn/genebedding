"""Survival analysis: cohort definition from mutations and Kaplan–Meier / log-rank."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Union

import pandas as pd

from .clinical import (
    clinical_with_survival,
    prepare_clinical,
    SURVIVAL_COLS,
    TREATMENT_FEATURES,
    load_clinical_raw,
)
from .mutations import (
    patients_with_mutation,
    patients_with_one_or_two_genes,
    double_variant_case_partition,
)
from ._config import get_clinical_path

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    from lifelines.exceptions import ConvergenceError
    from lifelines import CoxPHFitter
    _HAS_LIFELINES = True
except ImportError:
    _HAS_LIFELINES = False


def _require_lifelines():
    if not _HAS_LIFELINES:
        raise ImportError(
            "Survival analysis requires lifelines. Install with: pip install lifelines"
        )


class SurvivalAnalysis:
    """
    Survival analysis over TCGA clinical data, with cohorts defined by mutation status.
    Use target_cases (e.g. patients with a mutation) and optionally control_cases for comparison.
    """

    def __init__(
        self,
        clinical_df: Optional[pd.DataFrame] = None,
        path=None,
        mutation_path: Optional[Path] = None,
    ):
        if clinical_df is not None:
            self._clinical_raw = clinical_df
        else:
            from .clinical import load_clinical_raw
            self._clinical_raw = load_clinical_raw(path or get_clinical_path())
        self._mutation_path = mutation_path
        self.clindf = prepare_clinical(
            self._clinical_raw,
            subset_columns=TREATMENT_FEATURES + ["Proj_name"],
        )
        self.df = self.clindf.copy()
        self.df["group"] = 0
        self.df = self.df.fillna(0)
        self.treatment_features = list(TREATMENT_FEATURES)

    def cohort_from_mutation(
        self,
        genes: List[str],
        *,
        any_of: bool = True,
        control_cases: Optional[List[str]] = None,
        use_all_others_as_control: bool = True,
    ) -> pd.DataFrame:
        """
        Build survival dataframe with group=1 for patients with mutation in any (or all) of genes,
        and group=0 for control. If control_cases is None and use_all_others_as_control is True,
        all clinical patients not in the target set are used as control; otherwise pass control_cases.
        """
        target = patients_with_mutation(genes, any_of=any_of, path=self._mutation_path)
        if target.empty:
            target_cases = []
        else:
            target_cases = target["case_id"].astype(str).tolist()
        if control_cases is None and use_all_others_as_control:
            all_ids = set(self.df["case_id"].astype(str))
            target_set = set(target_cases)
            control_cases = list(all_ids - target_set)
        return self.generate_clinical_dataframe(
            target_cases=target_cases,
            control_cases=control_cases,
            features_of_interest=[],
        )

    def cohort_from_two_genes(
        self,
        gene_a: str,
        gene_b: str,
        *,
        cohort: str = "either",
        control_cases: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Build survival dataframe for cohorts defined by gene_a / gene_b mutation.
        cohort: one of 'only_a', 'only_b', 'both', 'either'.
        """
        parts = patients_with_one_or_two_genes(gene_a, gene_b, path=self._mutation_path)
        target_cases = list(parts.get(cohort, set()))
        return self.generate_clinical_dataframe(
            target_cases=target_cases,
            control_cases=control_cases,
            features_of_interest=[],
        )

    def generate_clinical_dataframe(
        self,
        target_cases: List[str],
        control_cases: Optional[List[str]] = None,
        features_of_interest: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Subset to target and (optional) control cases; add group 0/1 for survival comparison.
        Drops treatment columns with no variance; keeps duration, event, and variable features.
        """
        features_of_interest = features_of_interest or []
        df = self.df.copy()
        target_set = set(str(c) for c in target_cases)
        df.loc[df["case_id"].astype(str).isin(target_set), "group"] = 2
        if control_cases is not None:
            control_set = set(str(c) for c in control_cases)
            df.loc[df["case_id"].astype(str).isin(control_set), "group"] = 1
        df = df[df["group"] > 0].copy()
        df["group"] = df["group"] - 1  # 0 and 1

        core = ["duration", "event", "group"]
        feats = [f for f in features_of_interest if f in df.columns]
        for col in self.treatment_features:
            if col in df.columns:
                df.loc[df[col] > 0, col] = 1
                if df[col].nunique() > 1 and col not in feats:
                    feats.append(col)
        keep = core + [c for c in feats if c in df.columns and df[c].nunique() > 1]
        return df[[c for c in keep if c in df.columns]].copy()

    def kaplan_meier(
        self,
        df: pd.DataFrame,
        *,
        feature: str = "group",
        control_label: str = "Control",
        target_label: str = "Target",
        plot: bool = False,
        title: Optional[str] = None,
        time_cap: bool = True,
        savepath: Optional[str] = None,
        figsize: tuple = (7, 4),
    ) -> pd.Series:
        """
        Run Kaplan–Meier and optional log-rank test on a two-group survival dataframe.
        Returns a Series with n per group, p_value, and optional AUC metrics.
        """
        _require_lifelines()
        if df[feature].nunique() != 2:
            raise ValueError("Kaplan–Meier requires exactly two groups")

        cap_time = float(df.groupby(feature)["duration"].max().min())
        results = pd.Series(dtype=float)
        ax = None

        for val in [0, 1]:
            g = df[df[feature] == val]
            label = f"{control_label} (n={len(g)})" if val == 0 else f"{target_label} (n={len(g)})"
            results[control_label if val == 0 else target_label] = len(g)
            kmf = KaplanMeierFitter()
            kmf.fit(g["duration"], g["event"], label=label)
            if plot:
                if ax is None:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=figsize)
                    kmf.plot_survival_function(ax=ax, ci_show=True, color="#2430e0", lw=2)
                else:
                    kmf.plot_survival_function(ax=ax, ci_show=True, color="#e60215", lw=2)

        g1 = df[df[feature] == 1]
        g0 = df[df[feature] == 0]
        p_value = logrank_test(
            g1["duration"], g0["duration"],
            event_observed_A=g1["event"],
            event_observed_B=g0["event"],
        ).p_value
        results["p_value"] = p_value

        if plot and ax is not None:
            ax.text(0.6, 0.6, f"Log-rank p = {p_value:.3e}", transform=ax.transAxes, fontsize=10)
            ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if title:
                ax.set_title(title, fontsize=12)
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Survival probability")
            if time_cap:
                ax.set_xlim(0, cap_time)
            if savepath:
                import matplotlib.pyplot as plt
                plt.savefig(savepath, bbox_inches="tight", dpi=300)
            import matplotlib.pyplot as plt
            plt.show()

        return results

    def log_rank(self, group1: pd.DataFrame, group2: pd.DataFrame) -> float:
        """Log-rank p-value between two survival groups (each with duration, event)."""
        _require_lifelines()
        return logrank_test(
            group1["duration"], group2["duration"],
            event_observed_A=group1["event"],
            event_observed_B=group2["event"],
        ).p_value

    def cox_analysis(
        self,
        df: pd.DataFrame,
        features: List[str],
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> pd.Series:
        """Cox proportional hazards; returns p-values for each feature. Converge errors return empty Series."""
        _require_lifelines()
        cols = [c for c in features if c in df.columns and c not in (duration_col, event_col)]
        cols = cols + [duration_col, event_col]
        try:
            return CoxPHFitter().fit(df[cols], duration_col, event_col).summary["p"]
        except ConvergenceError:
            return pd.Series()


def survival_segmentation_summary(
    clinical_df: Optional[pd.DataFrame] = None,
    *,
    by_project: bool = True,
) -> pd.DataFrame:
    """
    High-level summary: for each segment (e.g. Proj_name), count of patients,
    events (deaths), median follow-up, and optional treatment breakdown.
    """
    if clinical_df is None:
        clinical_df = clinical_with_survival()
    df = clinical_df.copy()
    if "Proj_name" not in df.columns or not by_project:
        df["_segment"] = "all"
        by_col = "_segment"
    else:
        by_col = "Proj_name"
    g = df.groupby(by_col)
    out = pd.DataFrame({
        "n_patients": g["case_id"].count(),
        "n_events": g["event"].sum(),
        "median_follow_up_years": g["duration"].median(),
    }).reset_index()
    if by_col == "_segment":
        out = out.drop(columns=["_segment"])
    return out


def epistasis_survival_segmentation(
    epistasis_id_str: str,
    *,
    clinical_df: Optional[pd.DataFrame] = None,
    mutation_path: Optional[Path] = None,
    per_project: bool = True,
    min_n_per_group: int = 5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    For a double-variant epistasis ID (mut1|mut2), test whether survival differs between
    single (mut1 only or mut2 only) vs double (mut1 and mut2). Scalable: pass preloaded
    clinical_df and mutation_path once, then call for many epistasis IDs.

    Segmentation: single = only_mut1 ∪ only_mut2, double = both.
    Returns a DataFrame with one row per project (or one row "all" if per_project=False):
    project, n_single, n_double, p_value (log-rank), median_survival_single, median_survival_double,
    significant (True if p_value < alpha). Skips segments with fewer than min_n_per_group in either group.
    """
    _require_lifelines()
    if clinical_df is None:
        clinical_df = clinical_with_survival()
    clin = clinical_df.copy()
    clin["case_id"] = clin["case_id"].astype(str)
    if "Proj_name" not in clin.columns:
        clin["Proj_name"] = "all"
        per_project = False

    partition = double_variant_case_partition(epistasis_id_str, None, path=mutation_path)
    single_ids = partition["only_mut1"] | partition["only_mut2"]
    double_ids = partition["both"]
    all_ids = single_ids | double_ids
    if not all_ids:
        return pd.DataFrame(
            columns=[
                "project", "n_single", "n_double", "p_value",
                "median_survival_single", "median_survival_double", "significant",
            ]
        )

    clin = clin[clin["case_id"].isin(all_ids)].copy()
    clin["_segment"] = clin["case_id"].map(lambda c: "double" if c in double_ids else "single")
    clin = clin[clin["_segment"].isin(["single", "double"])]

    def run_logrank(sub: pd.DataFrame):
        s_single = sub[sub["_segment"] == "single"]
        s_double = sub[sub["_segment"] == "double"]
        if len(s_single) < min_n_per_group or len(s_double) < min_n_per_group:
            return None
        p = logrank_test(
            s_single["duration"], s_double["duration"],
            event_observed_A=s_single["event"],
            event_observed_B=s_double["event"],
        ).p_value
        return {
            "n_single": len(s_single),
            "n_double": len(s_double),
            "p_value": p,
            "median_survival_single": s_single["duration"].median(),
            "median_survival_double": s_double["duration"].median(),
            "significant": p < alpha,
        }

    if per_project:
        projects = clin["Proj_name"].dropna().unique().tolist()
        if not projects:
            projects = ["all"]
        rows = []
        for proj in projects:
            sub = clin[clin["Proj_name"] == proj]
            res = run_logrank(sub)
            if res is not None:
                rows.append({"project": proj, **res})
        out = pd.DataFrame(rows)
    else:
        res = run_logrank(clin)
        out = pd.DataFrame([{"project": "all", **res}] if res else [])

    if not out.empty:
        out = out[["project", "n_single", "n_double", "p_value", "median_survival_single", "median_survival_double", "significant"]]
    else:
        out = pd.DataFrame(
            columns=[
                "project", "n_single", "n_double", "p_value",
                "median_survival_single", "median_survival_double", "significant",
            ]
        )
    return out
