"""
Epistasis feature extraction from precomputed embeddings.

This module provides a small, clean interface to compute epistasis metrics
from precomputed embeddings for WT, single A, single B, and double AB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import hashlib
import json
import math
import sqlite3
import os
import random

import numpy as np
import torch

TensorLike = Union[np.ndarray, torch.Tensor]

_HASH_VERSION = "epistasis-case-v1"


def _to_tensor_1d(x: TensorLike, *, name: str) -> torch.Tensor:
    t = torch.as_tensor(x)
    if t.ndim != 1:
        raise ValueError(f"{name} must be 1D vector, got shape {tuple(t.shape)}")
    if not torch.is_floating_point(t):
        raise TypeError(f"{name} must be floating point, got {t.dtype}")
    if t.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} dtype must be float32/float64, got {t.dtype}")
    return t


def _stable_bytes(x: torch.Tensor) -> bytes:
    arr = x.detach().cpu().numpy().astype(np.float32, copy=False)
    arr = np.ascontiguousarray(arr)
    return arr.tobytes()


def _hash_metadata(meta: Optional[Dict[str, Any]]) -> str:
    if not meta:
        return ""
    payload = json.dumps(meta, sort_keys=True, default=str, separators=(",", ":"))
    return payload


@dataclass(frozen=True)
class EpistasisCase:
    """A single epistasis case with four embeddings and metadata."""
    case_id: str
    z_wt: TensorLike
    z_a: TensorLike
    z_b: TensorLike
    z_ab: TensorLike
    mut_a: Optional[str] = None
    mut_b: Optional[str] = None
    model: Optional[str] = None
    pool: str = "mean"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_wt = _to_tensor_1d(self.z_wt, name="z_wt")
        z_a = _to_tensor_1d(self.z_a, name="z_a")
        z_b = _to_tensor_1d(self.z_b, name="z_b")
        z_ab = _to_tensor_1d(self.z_ab, name="z_ab")

        if not (z_wt.shape == z_a.shape == z_b.shape == z_ab.shape):
            raise ValueError(
                "Embeddings must have identical shape: "
                f"z_wt={tuple(z_wt.shape)}, z_a={tuple(z_a.shape)}, "
                f"z_b={tuple(z_b.shape)}, z_ab={tuple(z_ab.shape)}"
            )
        return z_wt, z_a, z_b, z_ab

    def hash(self) -> str:
        z_wt, z_a, z_b, z_ab = self.as_tensors()
        h = hashlib.sha256()
        h.update(_HASH_VERSION.encode("utf-8"))
        h.update(_stable_bytes(z_wt))
        h.update(_stable_bytes(z_a))
        h.update(_stable_bytes(z_b))
        h.update(_stable_bytes(z_ab))
        h.update(str(z_wt.numel()).encode("utf-8"))
        h.update(str(self.model or "").encode("utf-8"))
        h.update(str(self.pool).encode("utf-8"))
        h.update(_hash_metadata(self.metadata).encode("utf-8"))
        return h.hexdigest()


def _safe_norm(v: torch.Tensor, eps: float) -> float:
    return float(torch.norm(v) + eps)


def _cosine(u: torch.Tensor, v: torch.Tensor, eps: float) -> float:
    nu = _safe_norm(u, eps)
    nv = _safe_norm(v, eps)
    if nu < eps or nv < eps:
        return 0.0
    return float(torch.dot(u, v) / (nu * nv))


def compute_core_vectors(
    z_wt: torch.Tensor,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    z_ab: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    delta_a = z_a - z_wt
    delta_b = z_b - z_wt
    delta_add = delta_a + delta_b
    delta_obs = z_ab - z_wt
    residual = delta_obs - delta_add
    return {
        "delta_a": delta_a,
        "delta_b": delta_b,
        "delta_add": delta_add,
        "delta_obs": delta_obs,
        "residual": residual,
    }


def compute_compat_metrics(
    *,
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    delta_add: torch.Tensor,
    delta_obs: torch.Tensor,
    residual: torch.Tensor,
    eps: float,
) -> Dict[str, float]:
    a1 = _safe_norm(delta_a, eps)
    a2 = _safe_norm(delta_b, eps)
    a12 = _safe_norm(delta_obs, eps)
    a12_exp = _safe_norm(delta_add, eps)

    r_raw = _safe_norm(residual, eps)
    single_scale = math.sqrt(a1**2 + a2**2) + eps
    r_singles = r_raw / single_scale
    magnitude_ratio = a12 / (a12_exp + eps)
    log_magnitude_ratio = math.log((a12 + eps) / (a12_exp + eps))

    return {
        "len_WT_M1": a1,
        "len_WT_M2": a2,
        "len_WT_M12": a12,
        "len_WT_M12_exp": a12_exp,
        "epi_R_raw": r_raw,
        "epi_R_singles": r_singles,
        "magnitude_ratio": magnitude_ratio,
        "log_magnitude_ratio": log_magnitude_ratio,
        "cos_v1_v2": _cosine(delta_a, delta_b, eps),
        "cos_exp_to_obs": _cosine(delta_add, residual, eps),
    }


def compute_geometry_metrics(
    *,
    delta_add: torch.Tensor,
    delta_obs: torch.Tensor,
    residual: torch.Tensor,
    eps: float,
) -> Dict[str, float]:
    add_norm = _safe_norm(delta_add, eps)
    if add_norm < eps:
        r_parallel = 0.0
        r_perp = _safe_norm(residual, eps)
    else:
        add_unit = delta_add / add_norm
        r_parallel = float(torch.dot(residual, add_unit))
        r_perp = _safe_norm(residual - r_parallel * add_unit, eps)

    obs_norm = _safe_norm(delta_obs, eps)
    denom = (obs_norm * add_norm) + eps
    cos_theta = float(torch.dot(delta_obs, delta_add) / denom)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    theta = math.acos(cos_theta)

    return {
        "r_parallel": r_parallel,
        "r_perp": r_perp,
        "theta": theta,
    }


def fit_covariance(
    vectors: Sequence[TensorLike],
    *,
    method: str = "ledoit_wolf",
    ridge: float = 1e-6,
    return_variances: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.stack([np.asarray(v, dtype=np.float64) for v in vectors], axis=0)

    if method == "diag":
        var = np.var(arr, axis=0, ddof=1)
        cov = np.diag(var)
    elif method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(arr)
            cov = lw.covariance_
        except Exception:
            cov = np.cov(arr, rowvar=False)
    else:
        cov = np.cov(arr, rowvar=False)

    cov = cov + np.eye(cov.shape[0]) * ridge
    cov_inv = np.linalg.inv(cov)
    if return_variances:
        return cov, cov_inv, np.diag(cov)
    return cov, cov_inv


def compute_mahalanobis(residual: torch.Tensor, cov_inv: np.ndarray) -> float:
    r = residual.detach().cpu().numpy().astype(np.float64, copy=False)
    return float(np.sqrt(r.T @ cov_inv @ r))


def empirical_pvalue(score: float, null_scores: Sequence[float]) -> float:
    null = np.asarray(list(null_scores), dtype=np.float64)
    if null.size == 0:
        return float("nan")
    return float((1 + np.sum(null >= score)) / (1 + null.size))


def bh_fdr(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q_out = np.empty_like(q)
    q_out[order] = q_sorted
    return q_out.tolist()


class EpistasisCache:
    """Lightweight cache for epistasis metrics."""
    def __init__(self, path: Optional[str] = None):
        self._path = path
        self._mem: Dict[str, Dict[str, Any]] = {}
        self._conn: Optional[sqlite3.Connection] = None
        if path:
            self._conn = sqlite3.connect(path, isolation_level=None, timeout=30.0)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS epistasis_cache (key TEXT PRIMARY KEY, value TEXT)"
            )

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if self._conn is None:
            return self._mem.get(key)
        row = self._conn.execute(
            "SELECT value FROM epistasis_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, key: str, value: Dict[str, Any]) -> None:
        payload = json.dumps(value, sort_keys=True, default=float)
        if self._conn is None:
            self._mem[key] = value
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO epistasis_cache (key, value) VALUES (?, ?)",
            (key, payload),
        )


def _list_epistasis_ids_from_db(db) -> List[str]:
    """
    Return epistasis IDs present in DB (based on |WT keys with 2+ pipes).
    """
    keys = db.list_keys(pattern="%|WT")
    epi_ids = []
    for k in keys:
        if k.endswith("|WT") and k.count("|") >= 2:
            epi_ids.append(k[:-3])
    return epi_ids


def _load_residual_from_db(db, epi_id: str) -> Optional[np.ndarray]:
    """Return residual vector for an epistasis ID, or None if missing."""
    try:
        z_wt = db.load(f"{epi_id}|WT", as_torch=True)
        z_m1 = db.load(f"{epi_id}|M1", as_torch=True)
        z_m2 = db.load(f"{epi_id}|M2", as_torch=True)
        z_m12 = db.load(f"{epi_id}|M12", as_torch=True)
    except KeyError:
        return None

    z_wt = torch.as_tensor(z_wt)
    z_m1 = torch.as_tensor(z_m1)
    z_m2 = torch.as_tensor(z_m2)
    z_m12 = torch.as_tensor(z_m12)

    if not (z_wt.ndim == z_m1.ndim == z_m2.ndim == z_m12.ndim == 1):
        return None

    residual = (z_m12 - z_wt) - ((z_m1 - z_wt) + (z_m2 - z_wt))
    return residual.detach().cpu().numpy()


def compute_cov_inv_from_db(
    db_path: str,
    *,
    method: str = "ledoit_wolf",
    ridge: float = 1e-6,
    sample_frac: Optional[float] = None,
    max_samples: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute covariance + inverse from all epistasis residuals in a DB.
    """
    from .genebeddings import VariantEmbeddingDB

    rng = random.Random(random_state)
    db = VariantEmbeddingDB(db_path)
    epi_ids = _list_epistasis_ids_from_db(db)

    if sample_frac is not None:
        k = max(1, int(len(epi_ids) * sample_frac))
        epi_ids = rng.sample(epi_ids, k)
    if max_samples is not None and len(epi_ids) > max_samples:
        epi_ids = rng.sample(epi_ids, max_samples)

    residuals = []
    for epi_id in epi_ids:
        r = _load_residual_from_db(db, epi_id)
        if r is not None:
            residuals.append(r)

    db.close()
    if not residuals:
        raise ValueError(f"No epistasis residuals found in {db_path}")

    return fit_covariance(residuals, method=method, ridge=ridge)


def compute_cov_inv_from_paths(
    paths: Sequence[str],
    *,
    method: str = "ledoit_wolf",
    ridge: float = 1e-6,
    sample_frac: Optional[float] = None,
    max_samples: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute cov_inv for every .db found in provided paths or directories.

    Returns {db_path: (cov, cov_inv)}.
    """
    db_files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for name in os.listdir(p):
                if name.endswith(".db"):
                    db_files.append(os.path.join(p, name))
        else:
            db_files.append(p)

    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for db_path in db_files:
        cov, cov_inv = compute_cov_inv_from_db(
            db_path,
            method=method,
            ridge=ridge,
            sample_frac=sample_frac,
            max_samples=max_samples,
            random_state=random_state,
        )
        results[db_path] = (cov, cov_inv)
    return results


class EpistasisFeatureExtractor:
    """Compute epistasis features for a list of cases."""
    def __init__(
        self,
        *,
        eps: float = 1e-20,
        include_geometry: bool = True,
        include_mahalanobis: bool = False,
        cov_inv: Optional[np.ndarray] = None,
        cache: Optional[EpistasisCache] = None,
        debug: bool = False,
    ):
        self.eps = float(eps)
        self.include_geometry = include_geometry
        self.include_mahalanobis = include_mahalanobis
        self.cov_inv = cov_inv
        self.cache = cache
        self.debug = debug

    def compute_case(self, case: EpistasisCase) -> Dict[str, Any]:
        key = case.hash()
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        z_wt, z_a, z_b, z_ab = case.as_tensors()
        core = compute_core_vectors(z_wt, z_a, z_b, z_ab)
        metrics = compute_compat_metrics(**core, eps=self.eps)

        if self.include_geometry:
            metrics.update(compute_geometry_metrics(
                delta_add=core["delta_add"],
                delta_obs=core["delta_obs"],
                residual=core["residual"],
                eps=self.eps,
            ))

        if self.include_mahalanobis and self.cov_inv is not None:
            metrics["epi_mahal"] = compute_mahalanobis(core["residual"], self.cov_inv)

        row = {
            "case_id": case.case_id,
            "case_hash": key,
            "mut_a": case.mut_a,
            "mut_b": case.mut_b,
            "model": case.model,
            "pool": case.pool,
            **metrics,
        }

        if self.debug:
            row.update({
                "norm_delta_a": _safe_norm(core["delta_a"], self.eps),
                "norm_delta_b": _safe_norm(core["delta_b"], self.eps),
                "norm_delta_add": _safe_norm(core["delta_add"], self.eps),
                "norm_delta_obs": _safe_norm(core["delta_obs"], self.eps),
                "norm_residual": _safe_norm(core["residual"], self.eps),
            })

        if case.metadata:
            row.update({f"meta_{k}": v for k, v in case.metadata.items()})

        if self.cache is not None:
            self.cache.set(key, row)
        return row

    def compute_table(
        self,
        cases: Iterable[EpistasisCase],
        *,
        return_dataframe: bool = True,
        null_scores: Optional[Dict[str, Sequence[float]]] = None,
        fdr: bool = False,
    ):
        rows = [self.compute_case(c) for c in cases]

        if null_scores:
            for metric, null in null_scores.items():
                pvals = [empirical_pvalue(r.get(metric, float("nan")), null) for r in rows]
                for row, p in zip(rows, pvals):
                    row[f"p_{metric}"] = p
                if fdr:
                    qvals = bh_fdr(pvals)
                    for row, q in zip(rows, qvals):
                        row[f"q_{metric}"] = q

        if not return_dataframe:
            return rows

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for return_dataframe=True") from e
        return pd.DataFrame(rows)


def sanity_suite(
    extractor: EpistasisFeatureExtractor,
    cases: Sequence[EpistasisCase],
) -> Dict[str, Any]:
    """
    Run basic sanity checks on a small set of cases.
    """
    results: Dict[str, Any] = {}

    # Symmetry: swapping A/B should not change scalar metrics
    if cases:
        c = cases[0]
        c_swapped = EpistasisCase(
            case_id=f"{c.case_id}|swap",
            z_wt=c.z_wt, z_a=c.z_b, z_b=c.z_a, z_ab=c.z_ab,
            mut_a=c.mut_b, mut_b=c.mut_a, model=c.model, pool=c.pool,
            metadata=c.metadata,
        )
        r1 = extractor.compute_case(c)
        r2 = extractor.compute_case(c_swapped)
        sym_keys = ["epi_R_singles", "magnitude_ratio", "epi_R_raw"]
        results["symmetry_ok"] = all(
            abs(r1.get(k, 0.0) - r2.get(k, 0.0)) < 1e-6 for k in sym_keys
        )

    # Zero-residual: AB == additive should give ~0 residual
    if cases:
        c = cases[0]
        z_wt, z_a, z_b, _ = c.as_tensors()
        z_ab = z_wt + (z_a - z_wt) + (z_b - z_wt)
        c_zero = EpistasisCase(
            case_id=f"{c.case_id}|zero",
            z_wt=z_wt, z_a=z_a, z_b=z_b, z_ab=z_ab,
            mut_a=c.mut_a, mut_b=c.mut_b, model=c.model, pool=c.pool,
            metadata=c.metadata,
        )
        r = extractor.compute_case(c_zero)
        results["zero_residual_ok"] = abs(r.get("epi_R_raw", 1.0)) < 1e-6

    return results
