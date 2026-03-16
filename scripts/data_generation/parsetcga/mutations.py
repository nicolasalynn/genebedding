"""Efficient mutation queries over the TCGA parquet via DuckDB."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ._config import get_mutations_path
from .ids import epistasis_id as _epistasis_id, normalize_mutation_id as _normalize_mutation_id

try:
    import duckdb
except ImportError:
    duckdb = None


# Canonical mutation_id: gene:chrom:pos:ref:alt with chrom without 'chr'; ref/alt uppercased for consistent matching
_MUTATION_ID_SQL = (
    "Gene_name || ':' || REPLACE(Chromosome, 'chr', '') || ':' || CAST(Start_Position AS VARCHAR) || "
    "':' || UPPER(COALESCE(Reference_Allele, '')) || ':' || UPPER(COALESCE(Tumor_Seq_Allele2, ''))"
)

LOOKUP_DB_NAME = "tcga_mutations_lookup.duckdb"
# Bump to force rebuild when mutation_id format changes (e.g. allele case normalisation)
_LOOKUP_SCHEMA_VERSION = 2


def _lookup_db_path(parquet_path: Path) -> Path:
    """Path for the persistent mutation_id -> case_id lookup database."""
    return parquet_path.parent / LOOKUP_DB_NAME


def build_mutation_lookup(path: Optional[Path] = None) -> Path:
    """
    Build (or rebuild) the persistent lookup DB for fast co_occurrence / double_variant_case_partition.
    One-time cost: reads the full parquet and writes data/tcga_mutations_lookup.duckdb (with index).
    After this, lookups by mutation_id are orders of magnitude faster.
    Returns the path to the lookup DB.
    """
    p = path or get_mutations_path()
    if not p.exists():
        raise FileNotFoundError(f"Mutations parquet not found: {p}")
    return _ensure_lookup_db(p)


def _ensure_lookup_db(parquet_path: Path) -> Path:
    """Create or confirm the lookup DB exists; return path. One-time build from parquet."""
    db_path = _lookup_db_path(parquet_path)
    if db_path.exists():
        try:
            con = duckdb.connect(str(db_path))
            try:
                stored = con.execute("SELECT version FROM _parsetcga_lookup_meta").fetchone()
                if stored and stored[0] == _LOOKUP_SCHEMA_VERSION and parquet_path.exists() and os.path.getmtime(db_path) >= os.path.getmtime(parquet_path):
                    return db_path
            finally:
                con.close()
        except Exception:
            pass
    if duckdb is None:
        raise ImportError("parsetcga mutation lookups require duckdb. Install with: pip install duckdb")
    con = duckdb.connect(str(db_path))
    try:
        # Single table: mutation_id, case_id for fast indexed lookups
        con.execute(f"""
            CREATE OR REPLACE TABLE mut_lookup AS
            SELECT ({_MUTATION_ID_SQL}) AS mutation_id, case_id
            FROM read_parquet({repr(str(parquet_path))})
        """)
        try:
            con.execute("CREATE INDEX idx_mut_lookup_mutation_id ON mut_lookup(mutation_id)")
        except Exception:
            pass  # Index may not exist in older DuckDB; table scan still faster than full parquet
        con.execute("CREATE OR REPLACE TABLE _parsetcga_lookup_meta (version INTEGER)")
        con.execute(f"INSERT INTO _parsetcga_lookup_meta VALUES ({_LOOKUP_SCHEMA_VERSION})")
    finally:
        con.close()
    return db_path


def _con(path: Optional[Path] = None):
    if duckdb is None:
        raise ImportError("parsetcga mutation queries require duckdb. Install with: pip install duckdb")
    p = path or get_mutations_path()
    if not p.exists():
        raise FileNotFoundError(f"Mutations parquet not found: {p}")
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW mut AS SELECT *, ({_MUTATION_ID_SQL}) AS mutation_id FROM read_parquet({repr(str(p))})"
    )
    return con


def _ensure_list(x: Union[str, list]) -> list:
    if isinstance(x, str):
        return [x]
    return list(x)


def patients_with_mutation(
    genes: Union[str, list[str]],
    *,
    path: Optional[Path] = None,
    any_of: bool = True,
    variant_classification: Optional[Union[str, list[str]]] = None,
) -> pd.DataFrame:
    """
    Return a dataframe of case_id and which of the requested genes are mutated.

    - genes: one gene symbol or list of gene symbols (e.g. "TP53" or ["TP53", "KRAS"]).
    - any_of: if True, patients with mutation in any of the genes; if False, only patients
      with mutations in all listed genes.
    - variant_classification: optional filter, e.g. "Missense_Mutation" or list of types.
    Returns columns: case_id, plus one column per gene (e.g. has_TP53) with 0/1.
    """
    genes = _ensure_list(genes)
    if not genes:
        return pd.DataFrame(columns=["case_id"])

    con = _con(path)
    try:
        gene_list = ", ".join(repr(g) for g in genes)
        where = f"Gene_name IN ({gene_list})"
        if variant_classification is not None:
            vc = _ensure_list(variant_classification)
            vc_list = ", ".join(repr(v) for v in vc)
            where += f" AND Variant_Classification IN ({vc_list})"
        q = f"""
        SELECT case_id, Gene_name
        FROM mut
        WHERE {where}
        """
        df = con.execute(q).fetchdf()
    finally:
        con.close()

    if df.empty:
        out = pd.DataFrame(columns=["case_id"] + [f"has_{g}" for g in genes])
        return out

    # One row per case_id per gene mutated
    piv = df.drop_duplicates().assign(val=1).pivot_table(
        index="case_id", columns="Gene_name", values="val", fill_value=0
    )
    piv.columns = [f"has_{c}" for c in piv.columns]
    piv = piv.reindex(columns=[f"has_{g}" for g in genes], fill_value=0).reset_index()

    if not any_of and len(genes) > 1:
        # keep only rows where all genes are mutated
        required = [f"has_{g}" for g in genes]
        piv = piv[piv[required].all(axis=1)]

    return piv


def patients_with_one_or_two_genes(
    gene_a: str,
    gene_b: str,
    *,
    path: Optional[Path] = None,
    variant_classification: Optional[Union[str, list[str]]] = None,
) -> dict[str, set[str]]:
    """
    Partition patients into: only A mutated, only B mutated, both mutated, either mutated.
    Returns dict with keys: 'only_a', 'only_b', 'both', 'either' (each set of case_id).
    """
    df = patients_with_mutation(
        [gene_a, gene_b],
        path=path,
        any_of=True,
        variant_classification=variant_classification,
    )
    if df.empty:
        return {"only_a": set(), "only_b": set(), "both": set(), "either": set()}

    ca, cb = f"has_{gene_a}", f"has_{gene_b}"
    if ca not in df.columns:
        df[ca] = 0
    if cb not in df.columns:
        df[cb] = 0

    only_a = set(df.loc[(df[ca] == 1) & (df[cb] == 0), "case_id"].astype(str))
    only_b = set(df.loc[(df[ca] == 0) & (df[cb] == 1), "case_id"].astype(str))
    both = set(df.loc[(df[ca] == 1) & (df[cb] == 1), "case_id"].astype(str))
    either = only_a | only_b | both
    return {"only_a": only_a, "only_b": only_b, "both": both, "either": either}


def mutation_counts_per_patient(
    *,
    path: Optional[Path] = None,
    genes: Optional[Union[str, list[str]]] = None,
    project: Optional[str] = None,
) -> pd.Series:
    """
    Number of mutation events per case_id (optionally restricted to genes or project).
    Returns a Series index by case_id.
    """
    con = _con(path)
    try:
        where = ["1=1"]
        if genes:
            g = _ensure_list(genes)
            where.append("Gene_name IN (" + ", ".join(repr(x) for x in g) + ")")
        if project:
            where.append(f"Proj_name = {repr(project)}")
        q = f"""
        SELECT case_id, COUNT(*) AS n_mutations
        FROM mut
        WHERE {" AND ".join(where)}
        GROUP BY case_id
        """
        df = con.execute(q).fetchdf()
    finally:
        con.close()
    return df.set_index("case_id")["n_mutations"]


def project_breakdown(
    *,
    path: Optional[Path] = None,
    genes: Optional[Union[str, list[str]]] = None,
) -> pd.DataFrame:
    """
    Count of unique patients and total mutation events per cancer project (Proj_name).
    Optionally restrict to a set of genes.
    """
    con = _con(path)
    try:
        where = "1=1"
        if genes:
            g = _ensure_list(genes)
            where += " AND Gene_name IN (" + ", ".join(repr(x) for x in g) + ")"
        q = f"""
        SELECT
            Proj_name,
            COUNT(DISTINCT case_id) AS n_patients,
            COUNT(*) AS n_mutations
        FROM mut
        WHERE {where}
        GROUP BY Proj_name
        ORDER BY n_patients DESC
        """
        return con.execute(q).fetchdf()
    finally:
        con.close()


def gene_summary(
    genes: Optional[Union[str, list[str]]] = None,
    *,
    path: Optional[Path] = None,
    by_project: bool = False,
) -> pd.DataFrame:
    """
    Per-gene counts: number of patients with at least one mutation, and total mutation events.
    If by_project True, breakdown by Proj_name and Gene_name.
    """
    con = _con(path)
    try:
        where = "1=1"
        if genes:
            g = _ensure_list(genes)
            where += " AND Gene_name IN (" + ", ".join(repr(x) for x in g) + ")"
        if by_project:
            q = f"""
            SELECT Proj_name, Gene_name,
                   COUNT(DISTINCT case_id) AS n_patients,
                   COUNT(*) AS n_mutations
            FROM mut WHERE {where}
            GROUP BY Proj_name, Gene_name
            ORDER BY n_patients DESC
            """
        else:
            q = f"""
            SELECT Gene_name,
                   COUNT(DISTINCT case_id) AS n_patients,
                   COUNT(*) AS n_mutations
            FROM mut WHERE {where}
            GROUP BY Gene_name
            ORDER BY n_patients DESC
            """
        return con.execute(q).fetchdf()
    finally:
        con.close()


def tumor_mutation_burden(
    *,
    path: Optional[Path] = None,
    project: Optional[str] = None,
) -> pd.Series:
    """
    Tumor mutation burden (TMB) per patient: total count of mutations per case_id.
    Optionally filter by project. Same as mutation_counts_per_patient with no gene filter.
    """
    return mutation_counts_per_patient(path=path, project=project)


def query_mutations(
    *,
    path: Optional[Path] = None,
    case_ids: Optional[Union[str, list[str]]] = None,
    genes: Optional[Union[str, list[str]]] = None,
    projects: Optional[Union[str, list[str]]] = None,
    variant_classification: Optional[Union[str, list[str]]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run a flexible query on the mutation table. All filters are optional.
    Use limit to cap rows for large result sets.
    """
    con = _con(path)
    try:
        where = ["1=1"]
        if case_ids:
            c = _ensure_list(case_ids)
            where.append("case_id IN (" + ", ".join(repr(x) for x in c) + ")")
        if genes:
            g = _ensure_list(genes)
            where.append("Gene_name IN (" + ", ".join(repr(x) for x in g) + ")")
        if projects:
            p = _ensure_list(projects)
            where.append("Proj_name IN (" + ", ".join(repr(x) for x in p) + ")")
        if variant_classification:
            v = _ensure_list(variant_classification)
            where.append("Variant_Classification IN (" + ", ".join(repr(x) for x in v) + ")")
        limit_cl = f"LIMIT {int(limit)}" if limit is not None else ""
        q = f"SELECT * FROM mut WHERE {' AND '.join(where)} {limit_cl}"
        return con.execute(q).fetchdf()
    finally:
        con.close()


def patient_mutation_ids(
    *,
    path: Optional[Path] = None,
    genes: Optional[Union[str, list[str]]] = None,
    case_ids: Optional[Union[str, list[str]]] = None,
    projects: Optional[Union[str, list[str]]] = None,
    variant_classification: Optional[Union[str, list[str]]] = None,
    add_epistasis_id: bool = True,
) -> pd.DataFrame:
    """
    Per-patient mutation IDs (and optionally epistasis_id).
    Returns DataFrame with case_id, mutation_id; if add_epistasis_id, one row per patient
    with epistasis_id = '|'.join(sorted(mutation_ids)) for that patient.
    """
    con = _con(path)
    try:
        where = ["1=1"]
        if genes:
            g = _ensure_list(genes)
            where.append("Gene_name IN (" + ", ".join(repr(x) for x in g) + ")")
        if case_ids:
            c = _ensure_list(case_ids)
            where.append("case_id IN (" + ", ".join(repr(x) for x in c) + ")")
        if projects:
            p = _ensure_list(projects)
            where.append("Proj_name IN (" + ", ".join(repr(x) for x in p) + ")")
        if variant_classification:
            v = _ensure_list(variant_classification)
            where.append("Variant_Classification IN (" + ", ".join(repr(x) for x in v) + ")")
        q = f"""
        SELECT case_id, mutation_id
        FROM mut
        WHERE {" AND ".join(where)}
        """
        df = con.execute(q).fetchdf()
    finally:
        con.close()

    if df.empty:
        out = pd.DataFrame(columns=["case_id", "mutation_id"])
        if add_epistasis_id:
            out["epistasis_id"] = []
        return out

    if add_epistasis_id:
        agg = df.groupby("case_id", as_index=False)["mutation_id"].agg(
            epistasis_id=lambda x: _epistasis_id(x.tolist())
        )
        # Also return one row per mutation with mutation_id, and merge epistasis_id onto it
        df = df.merge(agg[["case_id", "epistasis_id"]], on="case_id", how="left")
    return df


def _parse_epistasis_pair(epistasis_id_str: str) -> tuple[str, str]:
    """Parse 'mut1|mut2' into (canonical mut1, canonical mut2)."""
    parts = [p.strip() for p in str(epistasis_id_str).split("|") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected exactly two mutation IDs separated by '|', got {len(parts)}: {epistasis_id_str!r}")
    return _normalize_mutation_id(parts[0]), _normalize_mutation_id(parts[1])


def double_variant_case_partition(
    mut1_id: str,
    mut2_id: Optional[str] = None,
    *,
    path: Optional[Path] = None,
) -> dict:
    """
    For a double variant (two mutation IDs), partition patients into:
    only_mut1, only_mut2, both.
    Accepts epistasis_id string 'mut1|mut2' (pass as mut1_id only) or two separate mutation IDs.
    Returns dict with keys: mut1_id, mut2_id (canonical), only_mut1, only_mut2, both (sets of case_id).
    """
    if mut2_id is None and "|" in str(mut1_id):
        mut1_id, mut2_id = _parse_epistasis_pair(mut1_id)
    else:
        if mut2_id is None:
            raise ValueError("Provide either epistasis_id string 'mut1|mut2' or both mut1_id and mut2_id")
        mut1_id = _normalize_mutation_id(mut1_id)
        mut2_id = _normalize_mutation_id(mut2_id)
    p = path or get_mutations_path()
    if not p.exists():
        raise FileNotFoundError(f"Mutations parquet not found: {p}")
    db_path = _ensure_lookup_db(p)
    con = duckdb.connect(str(db_path))
    try:
        q = f"""
        SELECT case_id, mutation_id
        FROM mut_lookup
        WHERE mutation_id IN ({repr(mut1_id)}, {repr(mut2_id)})
        """
        df = con.execute(q).fetchdf()
    finally:
        con.close()
    only_mut1 = set()
    only_mut2 = set()
    both = set()
    if df.empty:
        return {"mut1_id": mut1_id, "mut2_id": mut2_id, "only_mut1": only_mut1, "only_mut2": only_mut2, "both": both}
    for case_id, grp in df.groupby("case_id"):
        ids = set(grp["mutation_id"].astype(str))
        has1 = mut1_id in ids
        has2 = mut2_id in ids
        c = str(case_id)
        if has1 and has2:
            both.add(c)
        elif has1:
            only_mut1.add(c)
        elif has2:
            only_mut2.add(c)
    return {"mut1_id": mut1_id, "mut2_id": mut2_id, "only_mut1": only_mut1, "only_mut2": only_mut2, "both": both}


def co_occurrence(
    mut1_id: str,
    mut2_id: Optional[str] = None,
    *,
    path: Optional[Path] = None,
    as_dataframe: bool = False,
) -> Union[dict, pd.DataFrame]:
    """
    Co-occurrence of two variants (mut1_id, mut2_id or single string 'mut1|mut2').
    Returns counts and conditional probabilities: is one mutation always (or often) in the presence of the other?

    Returns:
        - n_only_mut1, n_only_mut2, n_both, n_either
        - P_mut2_given_mut1 = n_both / (n_only_mut1 + n_both)
        - P_mut1_given_mut2 = n_both / (n_only_mut2 + n_both)
        - contingency_2x2: DataFrame with rows mut1_absent/mut1_present, cols mut2_absent/mut2_present
    """
    if mut2_id is None and "|" in str(mut1_id):
        partition = double_variant_case_partition(mut1_id, None, path=path)
    else:
        partition = double_variant_case_partition(mut1_id, mut2_id, path=path)
    o1, o2, b = partition["only_mut1"], partition["only_mut2"], partition["both"]
    n_only_mut1 = len(o1)
    n_only_mut2 = len(o2)
    n_both = len(b)
    n_either = n_only_mut1 + n_only_mut2 + n_both
    n_mut1 = n_only_mut1 + n_both
    n_mut2 = n_only_mut2 + n_both
    P_mut2_given_mut1 = (n_both / n_mut1) if n_mut1 else 0.0
    P_mut1_given_mut2 = (n_both / n_mut2) if n_mut2 else 0.0
    out = {
        "mut1_id": partition["mut1_id"],
        "mut2_id": partition["mut2_id"],
        "n_only_mut1": n_only_mut1,
        "n_only_mut2": n_only_mut2,
        "n_both": n_both,
        "n_either": n_either,
        "n_with_mut1": n_mut1,
        "n_with_mut2": n_mut2,
        "P_mut2_given_mut1": P_mut2_given_mut1,
        "P_mut1_given_mut2": P_mut1_given_mut2,
    }
    # 2x2 among patients with at least one of the two variants (no "neither" in our query)
    contingency = pd.DataFrame(
        [[0, n_only_mut2], [n_only_mut1, n_both]],
        index=["mut1_absent", "mut1_present"],
        columns=["mut2_absent", "mut2_present"],
    )
    out["contingency_2x2"] = contingency
    if as_dataframe:
        row = {k: v for k, v in out.items() if k != "contingency_2x2" and not isinstance(v, (set, pd.DataFrame))}
        return pd.DataFrame([row])
    return out


def find_mutations_in_data(
    gene: str,
    *,
    start_pos: Optional[int] = None,
    end_pos: Optional[int] = None,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Query the parquet directly (no lookup) for a gene and optional position range.
    Returns DataFrame with mutation_id, Gene_name, Chromosome, Start_Position,
    Reference_Allele, Tumor_Seq_Allele2, and n_cases (count of case_id per mutation).
    Use this to see how mutations are stored when a given mutation_id is not found.
    """
    con = _con(path)
    try:
        where = [f"Gene_name = {repr(gene)}"]
        if start_pos is not None:
            where.append(f"Start_Position >= {int(start_pos)}")
        if end_pos is not None:
            where.append(f"Start_Position <= {int(end_pos)}")
        q = f"""
        SELECT mutation_id, Gene_name, Chromosome, Start_Position,
               Reference_Allele, Tumor_Seq_Allele2,
               COUNT(DISTINCT case_id) AS n_cases
        FROM mut
        WHERE {" AND ".join(where)}
        GROUP BY mutation_id, Gene_name, Chromosome, Start_Position, Reference_Allele, Tumor_Seq_Allele2
        ORDER BY Start_Position
        """
        return con.execute(q).fetchdf()
    finally:
        con.close()


def mutation_id_from_row(row: Union[dict, pd.Series]) -> str:
    """
    Build canonical mutation_id from a row (dict or Series) with keys
    Gene_name, Chromosome, Start_Position, Reference_Allele, Tumor_Seq_Allele2.
    Chromosome is normalized to not contain 'chr'.
    """
    from .ids import mutation_id
    return mutation_id(
        gene=row["Gene_name"],
        chrom=row["Chromosome"],
        pos=row["Start_Position"],
        ref=row.get("Reference_Allele", ""),
        alt=row.get("Tumor_Seq_Allele2", ""),
    )
