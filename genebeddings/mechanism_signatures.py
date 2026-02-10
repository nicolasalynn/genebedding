"""
Mechanism signatures for epistasis and mutation effects.

Fits probabilistic signatures (direction + low-dim distribution) from
curated sets of effect vectors, and scores new vectors against them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import logging
import math

import numpy as np

from .epistasis_features import EpistasisCase, compute_core_vectors

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Delta loading helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONTEXT = 3000
_DEFAULT_GENOME = "hg38"


def _load_delta_from_db(db, mut_id: str) -> Optional[np.ndarray]:
    """
    Load delta (MUT − WT) from database for a single variant.

    Tries ``{mut_id}|WT`` / ``{mut_id}|MUT`` first, then falls back to
    loading ``mut_id`` directly (in case the delta was stored as-is).
    """
    if db is None:
        return None
    wt_key = f"{mut_id}|WT"
    mut_key = f"{mut_id}|MUT"
    try:
        z_wt = np.asarray(db.load(wt_key, as_torch=False), dtype=np.float64)
        z_mut = np.asarray(db.load(mut_key, as_torch=False), dtype=np.float64)
        return z_mut - z_wt
    except Exception:
        pass
    # Fallback: delta stored directly under the mut_id
    try:
        return np.asarray(db.load(mut_id, as_torch=False), dtype=np.float64)
    except Exception:
        return None


def _load_epistasis_delta_from_db(
    db,
    epi_id: str,
    which: str = "M1",
) -> Optional[np.ndarray]:
    """
    Load delta for one leg of an epistasis pair.

    Returns ``embed(which) − embed(WT)`` where *which* is ``"M1"``,
    ``"M2"``, or ``"M12"``.
    """
    if db is None:
        return None
    wt_key = f"{epi_id}|WT"
    mut_key = f"{epi_id}|{which}"
    try:
        z_wt = np.asarray(db.load(wt_key, as_torch=False), dtype=np.float64)
        z_mut = np.asarray(db.load(mut_key, as_torch=False), dtype=np.float64)
        return z_mut - z_wt
    except Exception:
        return None


def _compute_single_variant_delta(
    model,
    mut_id: str,
    *,
    context: int = _DEFAULT_CONTEXT,
    genome: str = _DEFAULT_GENOME,
    pool: str = "mean",
) -> Optional[np.ndarray]:
    """
    Compute delta on the fly from genomic coordinates using a model.

    Parses *mut_id* (``GENE:CHROM:POS:REF:ALT[:STRAND]``), fetches the
    reference sequence, applies the mutation, and returns
    ``embed(mutant) − embed(wildtype)`` as a numpy array.
    """
    try:
        from .genebeddings import parse_single_mut_id, _embed_single_variant_direct
        chrom, pos, ref, alt, rev = parse_single_mut_id(mut_id)
        _h_wt, _h_mut, delta = _embed_single_variant_direct(
            model, chrom, pos, ref, alt,
            reverse_complement=rev,
            context=context,
            genome=genome,
            pool=pool,
        )
        return np.asarray(delta, dtype=np.float64).flatten()
    except Exception as e:
        logger.warning("Failed to compute delta for %s: %s", mut_id, e)
        return None


def _compute_epistasis_delta(
    model,
    epi_id: str,
    which: str = "M1",
    *,
    context: int = _DEFAULT_CONTEXT,
    genome: str = _DEFAULT_GENOME,
    pool: str = "mean",
) -> Optional[np.ndarray]:
    """
    Compute epistasis-leg delta on the fly using a model.

    Parses *epi_id*, embeds WT and the four genotypes, and returns
    ``embed(which) − embed(WT)`` as a numpy array.
    """
    try:
        from .genebeddings import parse_epistasis_id, _embed_epistasis_direct
        (chrom, pos1, ref1, alt1, rev), (_, pos2, ref2, alt2, _) = (
            parse_epistasis_id(epi_id)
        )
        h_wt, h_m1, h_m2, h_m12 = _embed_epistasis_direct(
            model, chrom, pos1, ref1, alt1, pos2, ref2, alt2,
            reverse_complement=rev,
            context=context,
            genome=genome,
            pool=pool,
        )
        which_map = {"WT": h_wt, "M1": h_m1, "M2": h_m2, "M12": h_m12}
        h = which_map.get(which)
        if h is None:
            raise ValueError(f"Unknown which={which!r}")
        delta = np.asarray(h, dtype=np.float64).flatten() - np.asarray(h_wt, dtype=np.float64).flatten()
        return delta
    except Exception as e:
        logger.warning("Failed to compute epistasis delta for %s|%s: %s", epi_id, which, e)
        return None


def _get_delta(
    mut_id: str,
    db=None,
    model=None,
    *,
    context: int = _DEFAULT_CONTEXT,
    genome: str = _DEFAULT_GENOME,
    pool: str = "mean",
    save_to_db: bool = True,
) -> Optional[np.ndarray]:
    """
    Get single-variant delta: try DB first, fall back to model.

    If *model* computes the delta and *save_to_db* is True, the WT and
    MUT embeddings are stored in the DB for next time.
    """
    # Try DB first
    delta = _load_delta_from_db(db, mut_id)
    if delta is not None:
        return delta

    # Compute with model
    if model is None:
        return None

    delta = _compute_single_variant_delta(
        model, mut_id, context=context, genome=genome, pool=pool,
    )
    if delta is None:
        return None

    # Optionally cache in DB
    if save_to_db and db is not None:
        try:
            from .genebeddings import parse_single_mut_id, _embed_single_variant_direct
            chrom, pos, ref, alt, rev = parse_single_mut_id(mut_id)
            h_wt, h_mut, _ = _embed_single_variant_direct(
                model, chrom, pos, ref, alt,
                reverse_complement=rev,
                context=context,
                genome=genome,
                pool=pool,
            )
            db.store(f"{mut_id}|WT", h_wt)
            db.store(f"{mut_id}|MUT", h_mut)
        except Exception:
            pass  # caching is best-effort

    return delta


def _get_epistasis_delta(
    epi_id: str,
    which: str = "M1",
    db=None,
    model=None,
    *,
    context: int = _DEFAULT_CONTEXT,
    genome: str = _DEFAULT_GENOME,
    pool: str = "mean",
    save_to_db: bool = True,
) -> Optional[np.ndarray]:
    """
    Get epistasis-leg delta: try DB first, fall back to model.

    If *model* computes the embeddings and *save_to_db* is True, all
    four embeddings (WT, M1, M2, M12) are stored in the DB.
    """
    delta = _load_epistasis_delta_from_db(db, epi_id, which=which)
    if delta is not None:
        return delta

    if model is None:
        return None

    # Compute all four and cache
    try:
        from .genebeddings import parse_epistasis_id, _embed_epistasis_direct
        (chrom, pos1, ref1, alt1, rev), (_, pos2, ref2, alt2, _) = (
            parse_epistasis_id(epi_id)
        )
        h_wt, h_m1, h_m2, h_m12 = _embed_epistasis_direct(
            model, chrom, pos1, ref1, alt1, pos2, ref2, alt2,
            reverse_complement=rev,
            context=context,
            genome=genome,
            pool=pool,
        )
        if save_to_db and db is not None:
            db.store(f"{epi_id}|WT", h_wt)
            db.store(f"{epi_id}|M1", h_m1)
            db.store(f"{epi_id}|M2", h_m2)
            db.store(f"{epi_id}|M12", h_m12)

        which_map = {"WT": h_wt, "M1": h_m1, "M2": h_m2, "M12": h_m12}
        h = which_map[which]
        delta = np.asarray(h, dtype=np.float64).flatten() - np.asarray(h_wt, dtype=np.float64).flatten()
        return delta
    except Exception as e:
        logger.warning("Failed to compute epistasis delta for %s|%s: %s", epi_id, which, e)
        return None


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    """Result of classifying a variant against mechanism signatures."""

    best_group: str
    scores: Dict[str, Dict[str, float]]
    confidence: float
    delta: Optional[np.ndarray] = field(default=None, repr=False)

    def __repr__(self) -> str:
        lines = [
            f"ClassificationResult(best={self.best_group!r}, "
            f"confidence={self.confidence:.3f})",
        ]
        for name, s in self.scores.items():
            lines.append(
                f"  {name}: LL={s['log_likelihood']:.2f}, "
                f"mahal={s['mahalanobis']:.2f}, "
                f"align={s['alignment']:.3f}, "
                f"p={s['p_value']:.4f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Flatten to a single dict (useful for DataFrame rows)."""
        flat: Dict[str, Any] = {
            "best_group": self.best_group,
            "confidence": self.confidence,
        }
        for name, s in self.scores.items():
            for metric, val in s.items():
                if metric != "name":
                    flat[f"{name}_{metric}"] = val
        return flat


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Summary of how well-separated the mechanism groups are."""

    silhouette: float
    centroid_distances: Dict[Tuple[str, str], float]
    resubstitution_accuracy: float
    per_group_accuracy: Dict[str, float]
    n_per_group: Dict[str, int]
    confusion: Optional[Dict[Tuple[str, str], int]] = field(
        default=None, repr=False
    )
    cv_accuracy: float = float("nan")

    def __repr__(self) -> str:
        lines = [
            f"ValidationResult(",
            f"  silhouette       = {self.silhouette:.3f},",
            f"  resubstitution   = {self.resubstitution_accuracy:.1%},",
        ]
        if not math.isnan(self.cv_accuracy):
            lines.append(
                f"  cv_accuracy      = {self.cv_accuracy:.1%},"
            )
        for name, acc in self.per_group_accuracy.items():
            lines.append(
                f"  {name:20s}: n={self.n_per_group[name]:>4d}, "
                f"acc={acc:.1%}"
            )
        for (a, b), d in self.centroid_distances.items():
            lines.append(f"  centroid_cos_dist({a}, {b}) = {d:.4f}")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MechanismClassifier
# ---------------------------------------------------------------------------


class MechanismClassifier:
    """
    Fit mechanism signatures from labelled mutation groups, validate
    separation, and classify new variants.

    Workflow
    --------
    1. Build from a dict of ``{label: [mut_ids]}``:

       >>> clf = MechanismClassifier.from_mutation_groups(
       ...     {"splicing": splice_ids, "expression": expr_ids},
       ...     db=my_db,
       ...     show_progress=True,
       ... )

    2. Validate separation:

       >>> val = clf.validate()
       >>> print(val)

    3. Visualise:

       >>> clf.plot()

    4. Classify a new variant:

       >>> result = clf.classify_variant("KRAS:12:25227343:G:T:N", db)
       >>> print(result)

    5. Classify an epistasis leg:

       >>> result = clf.classify_epistasis(
       ...     "KRAS:12:25227343:G:T:N|KRAS:12:25227344:A:C:N",
       ...     db, which="M1",
       ... )

    6. Batch classify:

       >>> df = clf.classify_variants(mut_ids, db)
    """

    def __init__(
        self,
        sigset: MechanismSignatureSet,
        group_vectors: Dict[str, List[np.ndarray]],
        group_ids: Dict[str, List[str]],
        normalize: bool = True,
        variance_ratio: float = 0.9,
        context: int = _DEFAULT_CONTEXT,
        genome: str = _DEFAULT_GENOME,
        pool: str = "mean",
    ):
        self.sigset = sigset
        self.group_vectors = group_vectors
        self.group_ids = group_ids
        self.normalize = normalize
        self.variance_ratio = variance_ratio
        self.context = context
        self.genome = genome
        self.pool = pool
        self._labels = [s.name for s in sigset.signatures]

        # Sklearn classifier (populated by fit_classifier)
        self._clf = None
        self._clf_method: Optional[str] = None

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_mutation_groups(
        cls,
        groups: Dict[str, List[str]],
        db=None,
        model=None,
        *,
        normalize: bool = True,
        variance_ratio: float = 0.9,
        context: int = _DEFAULT_CONTEXT,
        genome: str = _DEFAULT_GENOME,
        pool: str = "mean",
        save_to_db: bool = True,
        show_progress: bool = False,
        eps: float = 1e-20,
    ) -> "MechanismClassifier":
        """
        Build a classifier from labelled mutation groups.

        Parameters
        ----------
        groups : dict
            ``{label: [mut_ids]}``, e.g.
            ``{"splicing": [...], "expression": [...]}``.
        db : VariantEmbeddingDB, optional
            Database containing precomputed embeddings. If a variant is
            found here, the model is not called.
        model : object, optional
            Model with ``.embed(seq, pool=...)`` method. Used as fallback
            when a variant is not in *db*. If both *db* and *model* are
            None, raises on any missing variant.
        normalize : bool
            Normalize delta vectors before fitting. Default: True.
        variance_ratio : float
            PCA variance ratio for each signature. Default: 0.9.
        context : int
            Context window for on-the-fly embedding. Default: 3000.
        genome : str
            Genome assembly. Default: ``"hg38"``.
        pool : str
            Pooling strategy. Default: ``"mean"``.
        save_to_db : bool
            If True and *model* computes a new embedding, store it in
            *db* for next time. Default: True.
        show_progress : bool
            Show tqdm progress bar. Default: False.
        """
        group_vectors: Dict[str, List[np.ndarray]] = {}
        group_ids: Dict[str, List[str]] = {}
        signatures: List[MechanismSignature] = []

        for label, mut_ids in groups.items():
            vectors: List[np.ndarray] = []
            valid_ids: List[str] = []

            iterator: Iterable = mut_ids
            if show_progress:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(mut_ids, desc=f"Loading {label}")
                except ImportError:
                    pass

            n_skipped = 0
            n_computed = 0
            n_from_db = 0
            for mid in iterator:
                # Try DB first
                delta = _load_delta_from_db(db, mid)
                if delta is not None:
                    n_from_db += 1
                else:
                    # Compute with model
                    delta = _get_delta(
                        mid, db=db, model=model,
                        context=context, genome=genome, pool=pool,
                        save_to_db=save_to_db,
                    )
                    if delta is not None:
                        n_computed += 1

                if delta is not None:
                    vectors.append(delta)
                    valid_ids.append(mid)
                else:
                    n_skipped += 1

            if not vectors:
                raise ValueError(
                    f"No valid deltas found for group {label!r} "
                    f"({len(mut_ids)} ids provided)"
                )

            if n_skipped:
                logger.warning(
                    "Group %r: %d/%d variants skipped (not in DB and "
                    "could not be computed)",
                    label, n_skipped, len(mut_ids),
                )
            if n_computed:
                logger.info(
                    "Group %r: %d variants computed on the fly",
                    label, n_computed,
                )

            group_vectors[label] = vectors
            group_ids[label] = valid_ids

            sig = fit_signature(
                vectors,
                name=label,
                normalize=normalize,
                variance_ratio=variance_ratio,
            )
            signatures.append(sig)

        sigset = MechanismSignatureSet(signatures)
        return cls(
            sigset, group_vectors, group_ids, normalize, variance_ratio,
            context=context, genome=genome, pool=pool,
        )

    # ------------------------------------------------------------------ #
    #  Classification
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Classifier fitting
    # ------------------------------------------------------------------ #

    def _build_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Stack all group vectors into X, y arrays."""
        all_vecs: List[np.ndarray] = []
        all_labels: List[int] = []
        for i, label in enumerate(self._labels):
            for v in self.group_vectors[label]:
                all_vecs.append(v)
                all_labels.append(i)
        X = np.stack(all_vecs)
        y = np.array(all_labels)
        if self.normalize:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / (norms + 1e-20)
        return X, y

    def fit_classifier(
        self,
        method: str = "lda",
        **kwargs: Any,
    ) -> "MechanismClassifier":
        """
        Fit a proper sklearn classifier on the stored group vectors.

        Once fitted, :meth:`classify` automatically uses this classifier
        instead of the raw signature log-likelihoods (which can be
        unreliable when groups have different PCA dimensionalities).

        Parameters
        ----------
        method : str
            One of:

            - ``"lda"`` — Linear Discriminant Analysis (shared covariance,
              works well even with many features). **Recommended default.**
            - ``"qda"`` — Quadratic Discriminant Analysis (per-class
              covariance, needs enough samples per class).
            - ``"logistic"`` — Logistic Regression (L2-regularised).
            - ``"knn"`` — k-Nearest Neighbours (cosine metric).
            - ``"nearest_centroid"`` — Nearest centroid (cosine metric,
              simplest possible).
        **kwargs
            Extra keyword arguments passed to the sklearn constructor
            (e.g. ``n_neighbors=10`` for knn).

        Returns
        -------
        self
            For chaining: ``clf.fit_classifier("lda").validate()``.
        """
        X, y = self._build_training_data()

        if method == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            clf = LinearDiscriminantAnalysis(**kwargs)
        elif method == "qda":
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            kwargs.setdefault("reg_param", 1e-4)
            clf = QuadraticDiscriminantAnalysis(**kwargs)
        elif method == "logistic":
            from sklearn.linear_model import LogisticRegression
            kwargs.setdefault("max_iter", 2000)
            kwargs.setdefault("C", 1.0)
            clf = LogisticRegression(**kwargs)
        elif method == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            kwargs.setdefault("n_neighbors", 5)
            kwargs.setdefault("metric", "cosine")
            clf = KNeighborsClassifier(**kwargs)
        elif method == "nearest_centroid":
            from sklearn.neighbors import NearestCentroid
            kwargs.setdefault("metric", "cosine")
            clf = NearestCentroid(**kwargs)
        else:
            raise ValueError(
                f"Unknown method {method!r}. Choose from: "
                "lda, qda, logistic, knn, nearest_centroid"
            )

        clf.fit(X, y)
        self._clf = clf
        self._clf_method = method

        logger.info(
            "Fitted %s classifier on %d vectors (%d groups)",
            method, X.shape[0], len(self._labels),
        )
        return self

    # ------------------------------------------------------------------ #
    #  Classification
    # ------------------------------------------------------------------ #

    def _prep_vector(self, delta: np.ndarray) -> np.ndarray:
        """Normalise a query vector (if signatures were built normalised)."""
        v = np.asarray(delta, dtype=np.float64).flatten()
        if self.normalize:
            v = v / (np.linalg.norm(v) + 1e-20)
        return v

    def classify(self, delta: np.ndarray) -> ClassificationResult:
        """
        Classify a delta vector against all mechanism signatures.

        If :meth:`fit_classifier` has been called, uses the sklearn
        classifier for the ``best_group`` decision. Otherwise falls back
        to raw signature log-likelihoods (which can be unreliable when
        groups have different PCA dimensions — see :meth:`fit_classifier`).

        Per-group signature scores (alignment, Mahalanobis, log-likelihood,
        p-value) are always included regardless of classification method.
        """
        # Always compute per-group signature scores for interpretability
        scores_list = self.sigset.score(delta)
        scores = {s["name"]: s for s in scores_list}

        # Determine best group
        if self._clf is not None:
            # Use the fitted sklearn classifier
            v = self._prep_vector(delta).reshape(1, -1)
            pred_idx = int(self._clf.predict(v)[0])
            best = self._labels[pred_idx]

            # Confidence from predict_proba if available
            if hasattr(self._clf, "predict_proba"):
                proba = self._clf.predict_proba(v)[0]
                sorted_proba = sorted(proba, reverse=True)
                confidence = float(sorted_proba[0] - sorted_proba[1])
            else:
                confidence = float("nan")
        else:
            # Fallback: raw log-likelihood (warn on first use)
            sorted_by_ll = sorted(
                scores_list,
                key=lambda s: s["log_likelihood"],
                reverse=True,
            )
            best = sorted_by_ll[0]["name"]
            if len(sorted_by_ll) > 1:
                confidence = (
                    sorted_by_ll[0]["log_likelihood"]
                    - sorted_by_ll[1]["log_likelihood"]
                )
            else:
                confidence = float("inf")

        return ClassificationResult(
            best_group=best,
            scores=scores,
            confidence=confidence,
            delta=delta,
        )

    def classify_variant(
        self,
        mut_id: str,
        db=None,
        model=None,
    ) -> ClassificationResult:
        """
        Load a single-variant delta from DB (or compute with model)
        and classify.
        """
        delta = _get_delta(
            mut_id, db=db, model=model,
            context=self.context, genome=self.genome, pool=self.pool,
        )
        if delta is None:
            raise ValueError(
                f"Could not load or compute delta for {mut_id!r}"
            )
        return self.classify(delta)

    def classify_epistasis(
        self,
        epi_id: str,
        db=None,
        model=None,
        which: str = "M1",
    ) -> ClassificationResult:
        """
        Load an epistasis-leg delta from DB (or compute with model)
        and classify.
        """
        delta = _get_epistasis_delta(
            epi_id, which=which, db=db, model=model,
            context=self.context, genome=self.genome, pool=self.pool,
        )
        if delta is None:
            raise ValueError(
                f"Could not load or compute delta for {epi_id!r}|{which}"
            )
        return self.classify(delta)

    def classify_variants(
        self,
        mut_ids: List[str],
        db=None,
        model=None,
        *,
        show_progress: bool = False,
    ) -> "pd.DataFrame":
        """
        Batch-classify a list of single variants. Returns a DataFrame.
        """
        import pandas as pd

        rows: List[Dict[str, Any]] = []
        iterator: Iterable = mut_ids
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(mut_ids, desc="Classifying")
            except ImportError:
                pass

        for mid in iterator:
            delta = _get_delta(
                mid, db=db, model=model,
                context=self.context, genome=self.genome, pool=self.pool,
            )
            if delta is None:
                logger.warning("Skipping %s (not in DB/model)", mid)
                continue
            result = self.classify(delta)
            row = {"mut_id": mid, **result.to_dict()}
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #

    def validate(self, cv: int = 0) -> ValidationResult:
        """
        Assess how well-separated the mechanism groups are.

        Computes:
        - **Silhouette score** (cosine): global cluster quality [-1, 1].
        - **Pairwise centroid cosine distances**: how far apart group
          centroids are.
        - **Resubstitution accuracy**: fraction of training vectors
          correctly classified (uses fitted sklearn classifier if
          available, else signature log-likelihoods).
        - **Cross-validated accuracy** (if ``cv > 0``): stratified k-fold
          CV using the same classifier method. Much more honest than
          resubstitution.
        - **Per-group accuracy**: breakdown by group.
        - **Confusion counts**: (true_label, predicted_label) → count.

        Parameters
        ----------
        cv : int, optional
            Number of cross-validation folds. If 0 (default), only
            resubstitution accuracy is computed. If > 0, adds
            cross-validated accuracy (requires a fitted classifier via
            :meth:`fit_classifier`).
        """
        X, y = self._build_training_data()

        # 1. Silhouette score
        try:
            from sklearn.metrics import silhouette_score

            if len(set(y)) > 1 and len(y) > len(set(y)):
                sil = float(silhouette_score(X, y, metric="cosine"))
            else:
                sil = float("nan")
        except ImportError:
            logger.warning("sklearn not available; skipping silhouette")
            sil = float("nan")

        # 2. Pairwise centroid cosine distances
        centroids: Dict[str, np.ndarray] = {}
        for label in self._labels:
            vecs = self.group_vectors[label]
            vecs_arr = np.stack(vecs)
            if self.normalize:
                norms = np.linalg.norm(vecs_arr, axis=1, keepdims=True)
                vecs_arr = vecs_arr / (norms + 1e-20)
            centroids[label] = vecs_arr.mean(axis=0)

        centroid_dists: Dict[Tuple[str, str], float] = {}
        for i, a in enumerate(self._labels):
            for b in self._labels[i + 1 :]:
                ca, cb = centroids[a], centroids[b]
                cos_sim = float(
                    np.dot(ca, cb)
                    / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-20)
                )
                centroid_dists[(a, b)] = 1.0 - cos_sim

        # 3. Resubstitution accuracy + confusion
        correct = 0
        total = 0
        per_group_correct: Dict[str, int] = {l: 0 for l in self._labels}
        per_group_total: Dict[str, int] = {l: 0 for l in self._labels}
        confusion: Dict[Tuple[str, str], int] = {}
        for a in self._labels:
            for b in self._labels:
                confusion[(a, b)] = 0

        for label in self._labels:
            for v in self.group_vectors[label]:
                result = self.classify(v)
                predicted = result.best_group
                confusion[(label, predicted)] += 1
                if predicted == label:
                    correct += 1
                    per_group_correct[label] += 1
                total += 1
                per_group_total[label] += 1

        resub_acc = correct / max(total, 1)
        per_group_acc = {
            l: per_group_correct[l] / max(per_group_total[l], 1)
            for l in self._labels
        }
        n_per_group = {
            l: len(self.group_vectors[l]) for l in self._labels
        }

        # 4. Cross-validated accuracy (optional)
        cv_accuracy = float("nan")
        if cv > 0 and self._clf is not None:
            try:
                from sklearn.model_selection import StratifiedKFold, cross_val_score
                skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                # Build a fresh classifier of the same type
                clf_clone = self._clf.__class__(
                    **self._clf.get_params()
                )
                cv_scores = cross_val_score(
                    clf_clone, X, y, cv=skf, scoring="accuracy",
                )
                cv_accuracy = float(cv_scores.mean())
                logger.info(
                    "CV accuracy (%d-fold): %.1f%% (+/- %.1f%%)",
                    cv, cv_accuracy * 100, cv_scores.std() * 100,
                )
            except Exception as e:
                logger.warning("Cross-validation failed: %s", e)

        return ValidationResult(
            silhouette=sil,
            centroid_distances=centroid_dists,
            resubstitution_accuracy=resub_acc,
            per_group_accuracy=per_group_acc,
            n_per_group=n_per_group,
            confusion=confusion,
            cv_accuracy=cv_accuracy,
        )

    # ------------------------------------------------------------------ #
    #  Plotting
    # ------------------------------------------------------------------ #

    def plot(
        self,
        *,
        method: str = "pca",
        query_deltas: Optional[Dict[str, np.ndarray]] = None,
        figsize: Tuple[float, float] = (10, 8),
        title: str = "",
        show: bool = True,
    ):
        """
        Visualise groups in 2-D (PCA or UMAP).

        Parameters
        ----------
        method : str
            ``"pca"`` (default) or ``"umap"``.
        query_deltas : dict, optional
            ``{label: delta_vector}`` — additional query points to overlay
            on the plot (shown as stars).
        figsize : tuple
            Figure size.
        title : str
            Plot title.
        show : bool
            Call ``plt.show()``.
        """
        import matplotlib.pyplot as plt

        # Collect data
        all_vectors: List[np.ndarray] = []
        all_labels: List[str] = []
        for label in self._labels:
            for v in self.group_vectors[label]:
                all_vectors.append(v)
                all_labels.append(label)

        X = np.stack(all_vectors)

        # Normalize if the signatures were fitted normalized
        if self.normalize:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / (norms + 1e-20)

        # Reduce to 2-D
        if method == "umap":
            try:
                from umap import UMAP
                reducer = UMAP(n_components=2, metric="cosine", random_state=42)
                coords = reducer.fit_transform(X)
            except ImportError:
                logger.warning("umap not installed, falling back to PCA")
                method = "pca"

        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(X)

        # Prepare query points
        query_coords = {}
        if query_deltas:
            for qname, qvec in query_deltas.items():
                qv = np.asarray(qvec, dtype=np.float64).reshape(1, -1)
                if self.normalize:
                    qv = qv / (np.linalg.norm(qv) + 1e-20)
                if method == "pca":
                    query_coords[qname] = reducer.transform(qv)[0]
                else:
                    # For UMAP, use transform if available
                    try:
                        query_coords[qname] = reducer.transform(qv)[0]
                    except Exception:
                        query_coords[qname] = None

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.Set1(np.linspace(0, 1, max(len(self._labels), 3)))
        label_to_color = {l: colors[i] for i, l in enumerate(self._labels)}

        for label in self._labels:
            mask = [l == label for l in all_labels]
            pts = coords[mask]
            ax.scatter(
                pts[:, 0], pts[:, 1],
                c=[label_to_color[label]],
                label=f"{label} (n={len(pts)})",
                alpha=0.6,
                s=30,
                edgecolors="white",
                linewidths=0.3,
            )

        # Overlay queries
        if query_coords:
            for qname, qc in query_coords.items():
                if qc is not None:
                    ax.scatter(
                        [qc[0]], [qc[1]],
                        marker="*",
                        s=300,
                        c="black",
                        edgecolors="gold",
                        linewidths=1.5,
                        zorder=10,
                        label=f"query: {qname}",
                    )

        ax.legend(framealpha=0.9)
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.set_title(title or f"Mechanism groups ({method.upper()})")
        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_confusion(
        self,
        validation: Optional[ValidationResult] = None,
        *,
        figsize: Tuple[float, float] = (8, 6),
        title: str = "",
        show: bool = True,
    ):
        """
        Plot the confusion matrix from validation.

        Parameters
        ----------
        validation : ValidationResult, optional
            If None, runs ``self.validate()`` first.
        """
        import matplotlib.pyplot as plt

        if validation is None:
            validation = self.validate()

        n = len(self._labels)
        matrix = np.zeros((n, n), dtype=int)
        for i, true_l in enumerate(self._labels):
            for j, pred_l in enumerate(self._labels):
                matrix[i, j] = validation.confusion.get(
                    (true_l, pred_l), 0
                )

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, cmap="Blues", aspect="equal")
        plt.colorbar(im, ax=ax, label="Count")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(self._labels, rotation=45, ha="right")
        ax.set_yticklabels(self._labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # Annotate cells
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, str(matrix[i, j]),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                    fontsize=12,
                )

        ax.set_title(title or "Resubstitution confusion matrix")
        plt.tight_layout()

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------ #
    #  Repr
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        groups = ", ".join(
            f"{l}(n={len(self.group_vectors[l])})" for l in self._labels
        )
        return f"MechanismClassifier([{groups}])"


# ---------------------------------------------------------------------------
# Hierarchical Mechanism Classifier
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalClassificationResult:
    """Result of a two-level hierarchical classification."""

    mechanism: str
    pathogenicity: str
    label: str
    mechanism_confidence: float
    pathogenicity_confidence: float
    mechanism_scores: Dict[str, Dict[str, float]]
    pathogenicity_scores: Dict[str, Dict[str, float]]
    delta: Optional[np.ndarray] = field(default=None, repr=False)

    def __repr__(self) -> str:
        lines = [
            f"HierarchicalClassificationResult(",
            f"  mechanism     = {self.mechanism!r} "
            f"(confidence={self.mechanism_confidence:.3f})",
            f"  pathogenicity = {self.pathogenicity!r} "
            f"(confidence={self.pathogenicity_confidence:.3f})",
            f"  label         = {self.label!r}",
            f")",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        flat: Dict[str, Any] = {
            "mechanism": self.mechanism,
            "pathogenicity": self.pathogenicity,
            "label": self.label,
            "mechanism_confidence": self.mechanism_confidence,
            "pathogenicity_confidence": self.pathogenicity_confidence,
        }
        for name, s in self.mechanism_scores.items():
            for metric, val in s.items():
                if metric != "name":
                    flat[f"mech_{name}_{metric}"] = val
        for name, s in self.pathogenicity_scores.items():
            for metric, val in s.items():
                if metric != "name":
                    flat[f"path_{name}_{metric}"] = val
        return flat


@dataclass
class HierarchicalValidationResult:
    """Validation results for each level of the hierarchy."""

    mechanism: ValidationResult
    pathogenicity: Dict[str, ValidationResult]

    def __repr__(self) -> str:
        lines = [
            "HierarchicalValidationResult(",
            "",
            "  === Level 1: Mechanism ===",
            f"  {self.mechanism}",
        ]
        for mech, val in self.pathogenicity.items():
            lines.append("")
            lines.append(f"  === Level 2: {mech} pathogenicity ===")
            lines.append(f"  {val}")
        lines.append(")")
        return "\n".join(lines)


class HierarchicalMechanismClassifier:
    """
    Two-level classifier: mechanism type first, then pathogenicity.

    Level 1 separates mutations by mechanism (e.g. splicing vs missense).
    Level 2 separates benign from pathogenic *within* each mechanism.

    This avoids forcing a single classifier to learn both the mechanism
    distinction (large signal) and the pathogenicity distinction (subtle
    signal) simultaneously.

    Parameters
    ----------
    groups : dict
        ``{label: [mut_ids]}``. Labels **must** follow the convention
        ``"mechanism_pathogenicity"`` (e.g. ``"splicing_benign"``,
        ``"missense_pathogenic"``). The part before the *last* underscore
        is the mechanism, the part after is the pathogenicity class.
    separator : str
        Character separating mechanism from pathogenicity in the label.
        Default: ``"_"``.

    Examples
    --------
    >>> hclf = HierarchicalMechanismClassifier.from_mutation_groups(
    ...     {
    ...         "splicing_benign": splice_b,
    ...         "splicing_pathogenic": splice_p,
    ...         "missense_benign": miss_b,
    ...         "missense_pathogenic": miss_p,
    ...     },
    ...     db=db,
    ...     classifier="lda",
    ...     show_progress=True,
    ... )
    >>> val = hclf.validate(cv=5)
    >>> print(val)
    >>> result = hclf.classify_variant("KRAS:12:25227343:G:T:N", db=db)
    >>> print(result)
    """

    def __init__(
        self,
        mechanism_clf: MechanismClassifier,
        pathogenicity_clfs: Dict[str, MechanismClassifier],
        separator: str = "_",
    ):
        self.mechanism_clf = mechanism_clf
        self.pathogenicity_clfs = pathogenicity_clfs
        self.separator = separator
        self._mechanisms = list(pathogenicity_clfs.keys())

    @classmethod
    def from_mutation_groups(
        cls,
        groups: Dict[str, List[str]],
        db=None,
        model=None,
        *,
        separator: str = "_",
        classifier: str = "lda",
        normalize: bool = True,
        variance_ratio: float = 0.9,
        context: int = _DEFAULT_CONTEXT,
        genome: str = _DEFAULT_GENOME,
        pool: str = "mean",
        save_to_db: bool = True,
        show_progress: bool = False,
        **clf_kwargs: Any,
    ) -> "HierarchicalMechanismClassifier":
        """
        Build a hierarchical classifier from labelled groups.

        Parameters
        ----------
        groups : dict
            ``{label: [mut_ids]}`` where each label has the form
            ``"mechanism_pathogenicity"`` (e.g. ``"splicing_benign"``).
        db, model, context, genome, pool, save_to_db
            Passed to :meth:`MechanismClassifier.from_mutation_groups`.
        separator : str
            Splits label into (mechanism, pathogenicity). Default ``"_"``.
        classifier : str
            Sklearn classifier method for both levels. Default ``"lda"``.
        normalize : bool
            Normalize delta vectors. Default True.
        **clf_kwargs
            Extra kwargs passed to ``fit_classifier``.
        """
        # Parse labels into (mechanism, pathogenicity)
        parsed: Dict[str, Dict[str, List[str]]] = {}
        for label, mut_ids in groups.items():
            idx = label.rfind(separator)
            if idx == -1:
                raise ValueError(
                    f"Label {label!r} does not contain separator "
                    f"{separator!r}. Expected format: mechanism{separator}pathogenicity"
                )
            mechanism = label[:idx]
            pathogenicity = label[idx + len(separator):]

            if mechanism not in parsed:
                parsed[mechanism] = {}
            parsed[mechanism][pathogenicity] = mut_ids

        build_kwargs = dict(
            db=db, model=model, normalize=normalize,
            variance_ratio=variance_ratio, context=context,
            genome=genome, pool=pool, save_to_db=save_to_db,
            show_progress=show_progress,
        )

        # Level 1: mechanism classifier (merge pathogenicity classes)
        mechanism_groups: Dict[str, List[str]] = {}
        for mechanism, path_dict in parsed.items():
            all_ids: List[str] = []
            for ids in path_dict.values():
                all_ids.extend(ids)
            mechanism_groups[mechanism] = all_ids

        logger.info("Fitting level 1 (mechanism): %s", list(mechanism_groups.keys()))
        mechanism_clf = MechanismClassifier.from_mutation_groups(
            mechanism_groups, **build_kwargs,
        )
        mechanism_clf.fit_classifier(classifier, **clf_kwargs)

        # Level 2: per-mechanism pathogenicity classifiers
        pathogenicity_clfs: Dict[str, MechanismClassifier] = {}
        for mechanism, path_dict in parsed.items():
            if len(path_dict) < 2:
                logger.warning(
                    "Mechanism %r has only 1 pathogenicity class — "
                    "skipping level-2 classifier",
                    mechanism,
                )
                continue

            logger.info(
                "Fitting level 2 (%s pathogenicity): %s",
                mechanism, list(path_dict.keys()),
            )
            path_clf = MechanismClassifier.from_mutation_groups(
                path_dict, **build_kwargs,
            )
            path_clf.fit_classifier(classifier, **clf_kwargs)
            pathogenicity_clfs[mechanism] = path_clf

        return cls(mechanism_clf, pathogenicity_clfs, separator=separator)

    # ------------------------------------------------------------------ #
    #  Classification
    # ------------------------------------------------------------------ #

    def classify(self, delta: np.ndarray) -> HierarchicalClassificationResult:
        """Classify a delta vector through both hierarchy levels."""
        # Level 1
        mech_result = self.mechanism_clf.classify(delta)
        mechanism = mech_result.best_group

        # Level 2
        path_clf = self.pathogenicity_clfs.get(mechanism)
        if path_clf is not None:
            path_result = path_clf.classify(delta)
            pathogenicity = path_result.best_group
            path_confidence = path_result.confidence
            path_scores = path_result.scores
        else:
            pathogenicity = "unknown"
            path_confidence = float("nan")
            path_scores = {}

        label = f"{mechanism}{self.separator}{pathogenicity}"

        return HierarchicalClassificationResult(
            mechanism=mechanism,
            pathogenicity=pathogenicity,
            label=label,
            mechanism_confidence=mech_result.confidence,
            pathogenicity_confidence=path_confidence,
            mechanism_scores=mech_result.scores,
            pathogenicity_scores=path_scores,
            delta=delta,
        )

    def classify_variant(
        self, mut_id: str, db=None, model=None,
    ) -> HierarchicalClassificationResult:
        """Load delta and classify through the hierarchy."""
        delta = _get_delta(
            mut_id, db=db, model=model,
            context=self.mechanism_clf.context,
            genome=self.mechanism_clf.genome,
            pool=self.mechanism_clf.pool,
        )
        if delta is None:
            raise ValueError(
                f"Could not load or compute delta for {mut_id!r}"
            )
        return self.classify(delta)

    def classify_epistasis(
        self, epi_id: str, db=None, model=None, which: str = "M1",
    ) -> HierarchicalClassificationResult:
        """Load epistasis-leg delta and classify through the hierarchy."""
        delta = _get_epistasis_delta(
            epi_id, which=which, db=db, model=model,
            context=self.mechanism_clf.context,
            genome=self.mechanism_clf.genome,
            pool=self.mechanism_clf.pool,
        )
        if delta is None:
            raise ValueError(
                f"Could not load or compute delta for {epi_id!r}|{which}"
            )
        return self.classify(delta)

    def classify_variants(
        self,
        mut_ids: List[str],
        db=None,
        model=None,
        *,
        show_progress: bool = False,
    ) -> "pd.DataFrame":
        """Batch-classify variants. Returns a DataFrame."""
        import pandas as pd

        rows: List[Dict[str, Any]] = []
        iterator: Iterable = mut_ids
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(mut_ids, desc="Classifying")
            except ImportError:
                pass

        for mid in iterator:
            delta = _get_delta(
                mid, db=db, model=model,
                context=self.mechanism_clf.context,
                genome=self.mechanism_clf.genome,
                pool=self.mechanism_clf.pool,
            )
            if delta is None:
                logger.warning("Skipping %s", mid)
                continue
            result = self.classify(delta)
            row = {"mut_id": mid, **result.to_dict()}
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #

    def validate(self, cv: int = 0) -> HierarchicalValidationResult:
        """
        Validate both levels of the hierarchy.

        Parameters
        ----------
        cv : int
            Cross-validation folds. 0 = resubstitution only.

        Returns
        -------
        HierarchicalValidationResult
            Contains validation for level 1 (mechanism) and level 2
            (pathogenicity within each mechanism).
        """
        mech_val = self.mechanism_clf.validate(cv=cv)

        path_vals: Dict[str, ValidationResult] = {}
        for mechanism, path_clf in self.pathogenicity_clfs.items():
            path_vals[mechanism] = path_clf.validate(cv=cv)

        return HierarchicalValidationResult(
            mechanism=mech_val,
            pathogenicity=path_vals,
        )

    # ------------------------------------------------------------------ #
    #  Plotting
    # ------------------------------------------------------------------ #

    def plot(self, **kwargs: Any):
        """Plot all levels. Returns dict of figures."""
        figs = {}
        figs["mechanism"] = self.mechanism_clf.plot(
            title="Level 1: Mechanism", **kwargs,
        )
        for mechanism, path_clf in self.pathogenicity_clfs.items():
            figs[f"{mechanism}_pathogenicity"] = path_clf.plot(
                title=f"Level 2: {mechanism} pathogenicity", **kwargs,
            )
        return figs

    def __repr__(self) -> str:
        mechs = ", ".join(self._mechanisms)
        paths = {
            m: list(c._labels)
            for m, c in self.pathogenicity_clfs.items()
        }
        return (
            f"HierarchicalMechanismClassifier("
            f"mechanisms=[{mechs}], pathogenicity={paths})"
        )
