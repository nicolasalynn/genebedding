"""
Mechanism signatures for epistasis and mutation effects.

Fits probabilistic signatures (direction + low-dim distribution) from
curated sets of effect vectors, and scores new vectors against them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math

import numpy as np

from .epistasis_features import EpistasisCase, compute_core_vectors


def _normalize_rows(x: np.ndarray, eps: float) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def _fit_pca(
    x: np.ndarray,
    *,
    n_components: Optional[int],
    variance_ratio: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (basis, explained_variance) where basis has shape (d, k).
    """
    n, d = x.shape
    if n < 2:
        basis = np.zeros((d, 1), dtype=np.float64)
        basis[0, 0] = 1.0
        return basis, np.array([1.0], dtype=np.float64)

    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=None, svd_solver="full")
        pca.fit(x)
        evr = pca.explained_variance_ratio_
        if n_components is None and variance_ratio is not None:
            k = int(np.searchsorted(np.cumsum(evr), variance_ratio) + 1)
        elif n_components is None:
            k = min(10, d)
        else:
            k = int(n_components)
        k = max(1, min(k, d))
        basis = pca.components_[:k].T
        return basis, evr[:k]
    except Exception:
        # Fallback: SVD
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        var = (s ** 2) / max(n - 1, 1)
        evr = var / max(np.sum(var), 1e-12)
        if n_components is None and variance_ratio is not None:
            k = int(np.searchsorted(np.cumsum(evr), variance_ratio) + 1)
        elif n_components is None:
            k = min(10, d)
        else:
            k = int(n_components)
        k = max(1, min(k, d))
        basis = vt[:k].T
        return basis, evr[:k]


@dataclass(frozen=True)
class MechanismSignature:
    name: str
    direction: np.ndarray
    basis: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    cov_inv: np.ndarray
    center: np.ndarray
    n_samples: int
    normalize: bool = True
    explained_variance: Optional[np.ndarray] = None

    def _prep(self, v: np.ndarray, eps: float) -> np.ndarray:
        x = v.astype(np.float64, copy=False)
        if self.normalize:
            x = x / (np.linalg.norm(x) + eps)
        return x

    def project(self, v: np.ndarray, *, eps: float = 1e-20) -> np.ndarray:
        x = self._prep(v, eps)
        return (x - self.center) @ self.basis

    def alignment(self, v: np.ndarray, *, eps: float = 1e-20) -> float:
        x = self._prep(v, eps)
        return float(np.dot(x, self.direction))

    def mahalanobis(self, v: np.ndarray, *, eps: float = 1e-20) -> float:
        z = self.project(v, eps=eps) - self.mean
        return float(np.sqrt(z.T @ self.cov_inv @ z))

    def log_likelihood(self, v: np.ndarray, *, eps: float = 1e-20) -> float:
        z = self.project(v, eps=eps) - self.mean
        k = z.shape[0]
        det = max(np.linalg.det(self.cov), eps)
        quad = float(z.T @ self.cov_inv @ z)
        return -0.5 * (k * math.log(2 * math.pi) + math.log(det) + quad)

    def p_value(self, v: np.ndarray, *, eps: float = 1e-20) -> float:
        try:
            from scipy.stats import chi2
        except Exception:
            return float("nan")
        z = self.project(v, eps=eps) - self.mean
        k = z.shape[0]
        d2 = float(z.T @ self.cov_inv @ z)
        return float(1.0 - chi2.cdf(d2, df=k))


def fit_signature(
    vectors: Sequence[np.ndarray],
    *,
    name: str,
    normalize: bool = True,
    center: bool = True,
    method: str = "pca",
    n_components: Optional[int] = None,
    variance_ratio: Optional[float] = 0.9,
    ridge: float = 1e-6,
    eps: float = 1e-20,
) -> MechanismSignature:
    x = np.stack([np.asarray(v, dtype=np.float64) for v in vectors], axis=0)
    if normalize:
        x = _normalize_rows(x, eps)
    center_vec = x.mean(axis=0) if center else np.zeros(x.shape[1], dtype=np.float64)
    xc = x - center_vec

    if method == "mean":
        direction = center_vec / (np.linalg.norm(center_vec) + eps)
        basis = direction.reshape(-1, 1)
        evr = np.array([1.0], dtype=np.float64)
    else:
        basis, evr = _fit_pca(xc, n_components=n_components, variance_ratio=variance_ratio)
        direction = basis[:, 0].copy()
        if np.dot(direction, center_vec) < 0:
            direction = -direction

    scores = xc @ basis
    mean = scores.mean(axis=0)
    cov = np.cov(scores, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    cov = cov + np.eye(cov.shape[0]) * ridge
    cov_inv = np.linalg.inv(cov)

    return MechanismSignature(
        name=name,
        direction=direction,
        basis=basis,
        mean=mean,
        cov=cov,
        cov_inv=cov_inv,
        center=center_vec,
        n_samples=x.shape[0],
        normalize=normalize,
        explained_variance=evr,
    )


def fit_signature_from_cases(
    cases: Sequence[EpistasisCase],
    *,
    name: str,
    vector: str = "residual",
    **kwargs: Any,
) -> MechanismSignature:
    vecs: List[np.ndarray] = []
    for c in cases:
        z_wt, z_a, z_b, z_ab = c.as_tensors()
        core = compute_core_vectors(z_wt, z_a, z_b, z_ab)
        v = core.get(vector)
        if v is None:
            raise ValueError(f"Unknown vector {vector!r}")
        vecs.append(v.detach().cpu().numpy())
    return fit_signature(vecs, name=name, **kwargs)


@dataclass
class MechanismSignatureSet:
    signatures: List[MechanismSignature]

    def score(self, v: np.ndarray, *, eps: float = 1e-20) -> List[Dict[str, Any]]:
        out = []
        for sig in self.signatures:
            out.append({
                "name": sig.name,
                "alignment": sig.alignment(v, eps=eps),
                "mahalanobis": sig.mahalanobis(v, eps=eps),
                "log_likelihood": sig.log_likelihood(v, eps=eps),
                "p_value": sig.p_value(v, eps=eps),
            })
        return out

    def attribution(
        self,
        v: np.ndarray,
        *,
        eps: float = 1e-20,
    ) -> Dict[str, Any]:
        if not self.signatures:
            raise ValueError("No signatures provided")
        dirs = np.stack([s.direction for s in self.signatures], axis=1)  # (d, m)
        x = v.astype(np.float64, copy=False)
        coeffs, *_ = np.linalg.lstsq(dirs, x, rcond=None)
        recon = dirs @ coeffs
        residual = x - recon
        denom = np.linalg.norm(x) + eps
        explained = 1.0 - (np.linalg.norm(residual) / denom)
        return {
            "coefficients": {s.name: float(c) for s, c in zip(self.signatures, coeffs)},
            "residual_norm": float(np.linalg.norm(residual)),
            "explained_fraction": float(explained),
        }


def score_cases(
    cases: Sequence[EpistasisCase],
    signatures: MechanismSignatureSet,
    *,
    vector: str = "residual",
    eps: float = 1e-20,
):
    rows = []
    for c in cases:
        z_wt, z_a, z_b, z_ab = c.as_tensors()
        core = compute_core_vectors(z_wt, z_a, z_b, z_ab)
        v = core.get(vector)
        if v is None:
            raise ValueError(f"Unknown vector {vector!r}")
        v_np = v.detach().cpu().numpy()
        scores = signatures.score(v_np, eps=eps)
        row = {
            "case_id": c.case_id,
            "case_hash": c.hash(),
        }
        for s in scores:
            name = s["name"]
            row[f"{name}_alignment"] = s["alignment"]
            row[f"{name}_mahal"] = s["mahalanobis"]
            row[f"{name}_loglik"] = s["log_likelihood"]
            row[f"{name}_pval"] = s["p_value"]
        rows.append(row)
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for score_cases") from e
    return pd.DataFrame(rows)
