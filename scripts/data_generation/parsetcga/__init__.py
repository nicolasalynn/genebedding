"""
parsetcga: Programmatic access to TCGA clinical (pickle) and mutation (parquet) data.

Single entry point: TCGAData(data_dir=None) gives access to:
  - .clinical: raw or survival-prepared clinical DataFrame
  - .mutations: lazy mutation queries (patients by gene, project breakdown, TMB, etc.)
  - .survival: SurvivalAnalysis for cohort definition and Kaplan–Meier / log-rank

Example:
    from parsetcga import TCGAData
    tcga = TCGAData()
    # Patients with TP53 mutation
    tp53 = tcga.mutations.patients_with_mutation("TP53")
    # Patients with TP53 and/or KRAS
    both = tcga.mutations.patients_with_one_or_two_genes("TP53", "KRAS")
    # Cancer project breakdown
    tcga.mutations.project_breakdown()
    # Survival-ready clinical
    clin = tcga.clinical.survival_prepared()
    # KM by mutation cohort
    sa = tcga.survival()
    df = sa.cohort_from_mutation(["TP53"])
    sa.kaplan_meier(df, target_label="TP53 mutated", control_label="Control", plot=True)
"""

from pathlib import Path
from typing import Optional

from . import _config
from . import clinical
from . import mutations
from . import survival as survival_module
from .clinical import (
    load_clinical_raw,
    prepare_clinical,
    clinical_with_survival,
    SURVIVAL_COLS,
    TREATMENT_FEATURES,
)
from .mutations import (
    patients_with_mutation,
    patients_with_one_or_two_genes,
    mutation_counts_per_patient,
    project_breakdown,
    gene_summary,
    tumor_mutation_burden,
    query_mutations,
    patient_mutation_ids,
    mutation_id_from_row,
    double_variant_case_partition,
    co_occurrence,
    build_mutation_lookup,
    find_mutations_in_data,
)
from .ids import mutation_id, normalize_mutation_id, epistasis_id, parse_mutation_id
from .survival import SurvivalAnalysis, survival_segmentation_summary, epistasis_survival_segmentation


class TCGAData:
    """
    Unified access to TCGA clinical (pickle) and mutation (parquet) data.
    Uses data_dir if provided, else package default (parsetcga/data) or TCGADATA_DIR env.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is not None:
            self._data_dir = Path(data_dir)
        else:
            self._data_dir = _config.get_data_dir()
        self._clinical_path = self._data_dir / _config.CLINICAL_PICKLE
        self._mutations_path = self._data_dir / _config.MUTATIONS_PARQUET
        self._clinical_raw: Optional[object] = None
        self._clinical_survival: Optional[object] = None

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def clinical(self) -> "ClinicalAccessor":
        return ClinicalAccessor(self)

    @property
    def mutations(self) -> "MutationAccessor":
        return MutationAccessor(self)

    def survival(self, clinical_df=None) -> SurvivalAnalysis:
        """Return a SurvivalAnalysis instance using package clinical data or the given DataFrame."""
        return SurvivalAnalysis(
            clinical_df=clinical_df,
            path=self._clinical_path if self._clinical_path.exists() else None,
            mutation_path=self._mutations_path if self._mutations_path.exists() else None,
        )

    def epistasis_survival_segmentation(
        self,
        epistasis_id_str: str,
        *,
        clinical_df=None,
        per_project: bool = True,
        min_n_per_group: int = 5,
        alpha: float = 0.05,
    ):
        """Run epistasis_survival_segmentation for this data (single vs double variant)."""
        if clinical_df is None:
            clinical_df = clinical_with_survival(path=self._clinical_path)
        return epistasis_survival_segmentation(
            epistasis_id_str,
            clinical_df=clinical_df,
            mutation_path=self._mutations_path if self._mutations_path.exists() else None,
            per_project=per_project,
            min_n_per_group=min_n_per_group,
            alpha=alpha,
        )


class ClinicalAccessor:
    """Access to clinical data for this TCGAData instance."""

    def __init__(self, parent: TCGAData):
        self._parent = parent

    def raw(self):
        """Load raw clinical pickle (patient_uuid, Proj_name, therapies, etc.)."""
        return load_clinical_raw(path=self._parent._clinical_path)

    def survival_prepared(self, min_duration: float = 0.0, include_treatments: bool = True):
        """Clinical with duration (years), event, case_id; optional treatment and Proj_name."""
        return clinical_with_survival(
            path=self._parent._clinical_path,
            min_duration=min_duration,
            include_treatments=include_treatments,
        )

    def prepare(self, df=None, **kwargs):
        """Custom prepare_clinical with optional DataFrame or path from this instance."""
        if df is None and self._parent._clinical_path.exists():
            df = load_clinical_raw(path=self._parent._clinical_path)
        return prepare_clinical(df=df, path=self._parent._clinical_path, **kwargs)


class MutationAccessor:
    """Mutation queries scoped to this TCGAData instance's parquet path."""

    def __init__(self, parent: TCGAData):
        self._parent = parent

    def _path(self) -> Optional[Path]:
        return self._parent._mutations_path if self._parent._mutations_path.exists() else None

    def patients_with_mutation(self, genes, any_of=True, variant_classification=None):
        return patients_with_mutation(
            genes, path=self._path(), any_of=any_of, variant_classification=variant_classification
        )

    def patients_with_one_or_two_genes(self, gene_a, gene_b, variant_classification=None):
        return patients_with_one_or_two_genes(
            gene_a, gene_b, path=self._path(), variant_classification=variant_classification
        )

    def mutation_counts_per_patient(self, genes=None, project=None):
        return mutation_counts_per_patient(
            path=self._path(), genes=genes, project=project
        )

    def project_breakdown(self, genes=None):
        return project_breakdown(path=self._path(), genes=genes)

    def gene_summary(self, genes=None, by_project=False):
        return gene_summary(genes=genes, path=self._path(), by_project=by_project)

    def tumor_mutation_burden(self, project=None):
        return tumor_mutation_burden(path=self._path(), project=project)

    def query(self, case_ids=None, genes=None, projects=None, variant_classification=None, limit=None):
        return query_mutations(
            path=self._path(),
            case_ids=case_ids,
            genes=genes,
            projects=projects,
            variant_classification=variant_classification,
            limit=limit,
        )

    def patient_mutation_ids(
        self,
        genes=None,
        case_ids=None,
        projects=None,
        variant_classification=None,
        add_epistasis_id=True,
    ):
        return patient_mutation_ids(
            path=self._path(),
            genes=genes,
            case_ids=case_ids,
            projects=projects,
            variant_classification=variant_classification,
            add_epistasis_id=add_epistasis_id,
        )

    def double_variant_case_partition(self, mut1_id, mut2_id=None):
        return double_variant_case_partition(mut1_id, mut2_id, path=self._path())

    def co_occurrence(self, mut1_id, mut2_id=None, as_dataframe=False):
        return co_occurrence(mut1_id, mut2_id, path=self._path(), as_dataframe=as_dataframe)

    def build_mutation_lookup(self):
        """Build the persistent lookup DB for fast co_occurrence / double_variant_case_partition (one-time)."""
        return build_mutation_lookup(path=self._path())

    def find_mutations_in_data(self, gene, start_pos=None, end_pos=None):
        """Query parquet for how a gene/position is stored (diagnostic when mutation_id not found)."""
        return find_mutations_in_data(gene, start_pos=start_pos, end_pos=end_pos, path=self._path())


__all__ = [
    "TCGAData",
    "ClinicalAccessor",
    "MutationAccessor",
    "SurvivalAnalysis",
    "load_clinical_raw",
    "prepare_clinical",
    "clinical_with_survival",
    "patients_with_mutation",
    "patients_with_one_or_two_genes",
    "mutation_counts_per_patient",
    "project_breakdown",
    "gene_summary",
    "tumor_mutation_burden",
    "query_mutations",
    "patient_mutation_ids",
    "mutation_id_from_row",
    "double_variant_case_partition",
    "co_occurrence",
    "build_mutation_lookup",
    "find_mutations_in_data",
    "mutation_id",
    "normalize_mutation_id",
    "epistasis_id",
    "parse_mutation_id",
    "survival_segmentation_summary",
    "epistasis_survival_segmentation",
    "SURVIVAL_COLS",
    "TREATMENT_FEATURES",
]
