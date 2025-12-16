"""
Genebeddings: Geometric Analysis of Variant Embeddings

A model-agnostic framework for computing geometrically interpretable metrics
from sequence embeddings of genetic variants. Supports both single-variant
analysis and double-variant epistasis quantification.

Overview
--------
This module provides tools for:

1. **Single-Variant Geometry**: Analyze the effect of a single mutation by
   comparing wild-type (WT) and mutant (MUT) embeddings. Computes delta vectors,
   distances, and canonical 2D projections.

2. **Epistasis Geometry**: Analyze genetic interactions between two mutations
   using four embeddings (WT, M1, M2, M12). Quantifies epistasis as deviation
   from additive expectation using both high-dimensional and triangle-based
   representations.

3. **Embedding Storage**: Persist and retrieve embeddings using a fast SQLite
   key-value store with metadata tracking.

4. **Embedding Generation**: Helper functions to generate embeddings from
   genomic coordinates using any model with a `.embed(seq, pool="mean")` API.

Key Concepts
------------
- **Effect Vector**: The difference MUT - WT in embedding space, representing
  the "direction" of a mutation's effect.

- **Additive Expectation**: For two mutations, the expected double-mutant
  embedding assuming no interaction: WT + (M1 - WT) + (M2 - WT).

- **Epistasis**: Deviation of the observed double-mutant from the additive
  expectation, quantified as magnitude (how much) and angle (direction).

- **Complex Epistasis**: A representation where epistasis is encoded as a
  complex number ε = Δ∥ + iΔ⊥, with magnitude |ε| and phase angle θ.

Example Usage
-------------
    >>> from genebeddings import SingleVariantGeometry, EpistasisGeometry
    >>> from genebeddings import VariantEmbeddingDB, embed_single_variant
    >>>
    >>> # Single variant analysis
    >>> geom = SingleVariantGeometry(h_wt, h_mut)
    >>> coords = geom.canonical_coords()
    >>> geom.plot()
    >>>
    >>> # Epistasis analysis
    >>> epi = EpistasisGeometry(h_wt, h_m1, h_m2, h_m12, diff="cosine")
    >>> metrics = epi.metrics()
    >>> epi.plot()
    >>>
    >>> # Database storage
    >>> with VariantEmbeddingDB("variants.db") as db:
    ...     db.store("GENE:1:12345:A:G", delta_embedding)
    ...     emb = db.load("GENE:1:12345:A:G")
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import MDS
from tqdm import tqdm

# SeqMat is imported locally in embed functions to avoid hard dependency

# ---------------------------------------------------------------------------
# Module Configuration
# ---------------------------------------------------------------------------

__all__ = [
    # Core geometry classes
    "SingleVariantGeometry",
    "EpistasisGeometry",
    # Data classes for results
    "SingleVariantCoords",
    "EpistasisComplexCoords",
    "EpistasisTriangleCoords",
    "EpistasisMetrics",
    # Database
    "VariantEmbeddingDB",
    # Embedding functions
    "embed_single_variant",
    "embed_epistasis",
    "store_single_variant_embedding",
    "store_epistasis_embeddings",
    "bulk_store_single_variants",
    "bulk_store_epistasis",
    # DataFrame integration
    "add_single_variant_metrics",
    "add_epistasis_metrics",
    # Parsing utilities
    "parse_single_mut_id",
    "parse_epistasis_id",
    # Distance functions
    "cosine_distance",
    "l2_distance",
    # Dependency map
    "DependencyMapResult",
    "compute_pairwise_epistasis",
    "compute_dependency_map",
    # Epistasis expectation model
    "EpistasisDataset",
    "EpistasisExpectationModel",
]

__version__ = "0.1.0"

# Configure module logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

TensorLike = Union[np.ndarray, torch.Tensor]
DiffFn = Callable[[torch.Tensor, torch.Tensor], float]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Database key suffixes for variant storage
KEY_WT = "|WT"
KEY_MUT = "|MUT"
KEY_M1 = "|M1"
KEY_M2 = "|M2"
KEY_M12 = "|M12"
KEY_DELTA1 = "|Δ1"
KEY_DELTA2 = "|Δ2"
KEY_DELTA12 = "|Δ12"

# Default values
DEFAULT_EPS = 1e-20
DEFAULT_CONTEXT = 3000
DEFAULT_GENOME = "hg38"
DEFAULT_RANDOM_STATE = 42

# Plot colors
COLOR_WT = "#1f77b4"
COLOR_OBSERVED = "#d62728"
COLOR_EXPECTED = "#555555"
COLOR_M1 = "#2ca02c"  # green
COLOR_M2 = "#9467bd"  # purple


# ---------------------------------------------------------------------------
# Distance Functions
# ---------------------------------------------------------------------------


def cosine_distance(x: torch.Tensor, y: torch.Tensor, eps: float = DEFAULT_EPS) -> float:
    """
    Compute cosine distance between two vectors: 1 - cosine_similarity.

    Parameters
    ----------
    x : torch.Tensor
        First vector.
    y : torch.Tensor
        Second vector.
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    float
        Cosine distance in range [0, 2].
    """
    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y).float()
    num = torch.dot(x, y)
    denom = torch.norm(x) * torch.norm(y) + eps
    cos = float(num / denom)
    cos = max(min(cos, 1.0), -1.0)  # Clamp for numerical stability
    return 1.0 - cos


def l2_distance(x: torch.Tensor, y: torch.Tensor, eps: float = DEFAULT_EPS) -> float:
    """
    Compute L2 (Euclidean) distance between two vectors.

    Parameters
    ----------
    x : torch.Tensor
        First vector.
    y : torch.Tensor
        Second vector.
    eps : float, optional
        Small constant added to result for numerical stability.

    Returns
    -------
    float
        L2 distance.
    """
    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y).float()
    return float(torch.norm(x - y) + eps)


# ---------------------------------------------------------------------------
# Result Data Classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SingleVariantCoords:
    """
    Canonical 2D coordinates for a single-variant effect vector.

    The effect vector v = MUT - WT is projected into a 2D plane defined by:
    - e_par: parallel axis (along v or a reference direction)
    - e_perp: perpendicular axis (orthogonal to e_par)

    Attributes
    ----------
    v_l2 : float
        L2 norm of the effect vector ||v||.
    v_diff : float
        Distance between WT and MUT using configured metric.
    v_par : float
        Component of v along the parallel axis.
    v_perp : float
        Component of v along the perpendicular axis.
    x : float
        Normalized parallel component (v_par / ||v||).
    y : float
        Normalized perpendicular component (v_perp / ||v||).
    radius : float
        Magnitude sqrt(x² + y²), approximately 1.
    angle : float
        Angle in radians from parallel axis.
    angle_over_pi : float
        Angle normalized by π.
    """

    v_l2: float
    v_diff: float
    v_par: float
    v_perp: float
    x: float
    y: float
    radius: float
    angle: float
    angle_over_pi: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for compatibility."""
        return {
            "v_l2": self.v_l2,
            "v_diff": self.v_diff,
            "v_par": self.v_par,
            "v_perp": self.v_perp,
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
            "angle": self.angle,
            "angle_over_pi": self.angle_over_pi,
        }


@dataclass(frozen=True)
class EpistasisComplexCoords:
    """
    Complex epistasis coordinates in the high-dimensional canonical plane.

    The plane is defined by:
    - e_par: unit vector along WT→expected (additive prediction)
    - e_perp: orthogonal axis derived from (v1 - v2), pointing from M2's
      effect toward M1's effect (after removing the parallel component)

    This construction gives directional meaning to the y-axis:
    - y_obs > 0: epistasis is "M1-like" (double mutant resembles M1 more)
    - y_obs < 0: epistasis is "M2-like" (double mutant resembles M2 more)

    Coordinates are normalized so expected lies at approximately (1, 0).

    Attributes
    ----------
    x_exp : float
        Expected double-mutant x-coordinate (≈ 1).
    y_exp : float
        Expected double-mutant y-coordinate (≈ 0).
    x_obs : float
        Observed double-mutant x-coordinate.
    y_obs : float
        Observed double-mutant y-coordinate. Positive means the epistatic
        deviation is in the M1 direction; negative means M2 direction.
    rho : float
        Magnitude |ε| of complex epistasis.
    theta : float
        Phase angle of complex epistasis (radians).
    theta_over_pi : float
        Phase angle normalized by π.
    """

    x_exp: float
    y_exp: float
    x_obs: float
    y_obs: float
    rho: float
    theta: float
    theta_over_pi: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for compatibility."""
        return {
            "x_exp": self.x_exp,
            "y_exp": self.y_exp,
            "x_obs": self.x_obs,
            "y_obs": self.y_obs,
            "rho": self.rho,
            "theta": self.theta,
            "theta_over_pi": self.theta_over_pi,
        }


@dataclass(frozen=True)
class EpistasisTriangleCoords:
    """
    Triangle-based epistasis coordinates using only pairwise distances.

    Uses the law of cosines to embed the WT-Expected-Observed triangle in 2D,
    with WT at origin and Expected at (1, 0). This representation depends only
    on three distances and is independent of embedding dimensionality.

    Attributes
    ----------
    x_exp : float
        Expected position x-coordinate (always 1.0).
    y_exp : float
        Expected position y-coordinate (always 0.0).
    x_obs : float
        Observed position x-coordinate.
    y_obs : float
        Observed position y-coordinate (≥ 0, upper half-plane).
    rho : float
        Magnitude of triangle-based epistasis.
    theta : float
        Angle of triangle-based epistasis (radians).
    theta_over_pi : float
        Angle normalized by π.
    """

    x_exp: float
    y_exp: float
    x_obs: float
    y_obs: float
    rho: float
    theta: float
    theta_over_pi: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for compatibility."""
        return {
            "x_exp_tri": self.x_exp,
            "y_exp_tri": self.y_exp,
            "x_obs_tri": self.x_obs,
            "y_obs_tri": self.y_obs,
            "rho_tri": self.rho,
            "theta_tri": self.theta,
            "theta_tri_over_pi": self.theta_over_pi,
        }


@dataclass(frozen=True)
class EpistasisMetrics:
    """
    Comprehensive metrics for epistasis analysis.

    Contains both distance-based metrics (using configured distance function)
    and L2 norm-based vector lengths.

    Attributes
    ----------
    dist_WT_M1 : float
        Distance from WT to single-mutant 1.
    dist_WT_M2 : float
        Distance from WT to single-mutant 2.
    dist_WT_M12_obs : float
        Distance from WT to observed double-mutant.
    dist_WT_M12_exp : float
        Distance from WT to expected double-mutant.
    dist_M1_M2 : float
        Distance between single-mutants.
    dist_obs_exp : float
        Distance between observed and expected double-mutants.
    len_WT_M1 : float
        L2 norm of effect vector v1 = M1 - WT.
    len_WT_M2 : float
        L2 norm of effect vector v2 = M2 - WT.
    len_WT_M12 : float
        L2 norm of observed effect vector v12_obs = M12 - WT.
    len_WT_M12_exp : float
        L2 norm of expected effect vector v12_exp = v1 + v2.
    len_M1_M2 : float
        L2 norm of vector between single-mutants.
    epi_R_raw : float
        Raw epistasis residual (distance between observed and expected).
    epi_R_singles : float
        Residual normalized by sqrt(|v1|² + |v2|²).
    epi_R_expected : float
        Residual normalized by distance to expected.
    epi_score : float
        Bounded epistasis score in [0, 1]. Computed as R_raw / (R_raw + d_exp).
        Handles small expected values gracefully. 0 = perfectly additive, 1 = maximal epistasis.
    cos_v12_v1 : float
        Cosine similarity between observed double-mutant effect and single-mutant 1 effect.
    cos_v12_v2 : float
        Cosine similarity between observed double-mutant effect and single-mutant 2 effect.
    cos_v12_vexp : float
        Cosine similarity between observed and expected double-mutant effects.
    cos_v1_v2 : float
        Cosine similarity between single-mutant effects (measures effect alignment).
    magnitude_ratio : float
        Ratio of observed to expected magnitude: |v12_obs| / |v12_exp|.
        <1 = sub-additive, >1 = super-additive.
    radial_deviation : float
        Signed deviation: (|v12_obs| - |v12_exp|) / |v12_exp|.
        Negative = closer to WT than expected, positive = further.
    epi_total : float
        **Primary epistasis metric** capturing both angle AND magnitude deviation.
        Computed as: sqrt((1 - cos_v12_vexp)^2 + (magnitude_ratio - 1)^2).
        This is the Euclidean distance from the "no epistasis" point (cos=1, ratio=1).
        0 = perfectly additive, higher = more epistatic.
    epi_total_bounded : float
        Bounded version of epi_total in [0, 1]: epi_total / (1 + epi_total).
        0 = perfectly additive, approaches 1 for strong epistasis.
    log_rho_tri : float
        Log of magnitude ratio: log(|v12_obs| / |v12_exp|).
        0 = same magnitude as expected (additive).
        Positive = super-additive (observed > expected).
        Negative = sub-additive (observed < expected).
        High dynamic range, good for detecting outliers.
    abs_log_rho_tri : float
        **Recommended filter metric**: |log(|v12_obs| / |v12_exp|)|.
        0 = perfectly additive, larger = more epistatic (either direction).
        Unbounded with high dynamic range. Values:
        0.1 = ~25% magnitude difference
        0.3 = ~2x difference
        0.7 = ~5x difference
        1.0 = ~10x difference
    len_WT_M12_L1 : float
        L1 norm of observed effect vector: sum(|v12_obs|).
    len_WT_M12_exp_L1 : float
        L1 norm of expected effect vector: sum(|v12_exp|).
    epi_R_raw_L1 : float
        L1 norm of residual: sum(|v12_obs - v12_exp|).
    epi_R_expected_L1 : float
        L1 residual normalized by L1 expected: epi_R_raw_L1 / len_WT_M12_exp_L1.
    magnitude_ratio_L1 : float
        L1 magnitude ratio: len_WT_M12_L1 / len_WT_M12_exp_L1.
    log_rho_L1 : float
        Log of L1 magnitude ratio. 0 = additive, + = super, - = sub.
    abs_log_rho_L1 : float
        |log(L1 ratio)|. Recommended L1-based filter metric.
        Symmetric: 2x and 0.5x both give ~0.69.
    """

    dist_WT_M1: float
    dist_WT_M2: float
    dist_WT_M12_obs: float
    dist_WT_M12_exp: float
    dist_M1_M2: float
    dist_obs_exp: float
    len_WT_M1: float
    len_WT_M2: float
    len_WT_M12: float
    len_WT_M12_exp: float
    len_M1_M2: float
    epi_R_raw: float
    epi_R_singles: float
    epi_R_expected: float
    epi_score: float
    cos_v12_v1: float
    cos_v12_v2: float
    cos_v12_vexp: float
    cos_v1_v2: float
    magnitude_ratio: float
    radial_deviation: float
    epi_total: float
    epi_total_bounded: float
    log_rho_tri: float
    abs_log_rho_tri: float
    len_WT_M12_L1: float
    len_WT_M12_exp_L1: float
    epi_R_raw_L1: float
    epi_R_expected_L1: float
    magnitude_ratio_L1: float
    log_rho_L1: float
    abs_log_rho_L1: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for compatibility."""
        return {
            "dist_WT_M1": self.dist_WT_M1,
            "dist_WT_M2": self.dist_WT_M2,
            "dist_WT_M12_obs": self.dist_WT_M12_obs,
            "dist_WT_M12_exp": self.dist_WT_M12_exp,
            "dist_M1_M2": self.dist_M1_M2,
            "dist_obs_exp": self.dist_obs_exp,
            "len_WT_M1": self.len_WT_M1,
            "len_WT_M2": self.len_WT_M2,
            "len_WT_M12": self.len_WT_M12,
            "len_WT_M12_exp": self.len_WT_M12_exp,
            "len_M1_M2": self.len_M1_M2,
            "epi_R_raw": self.epi_R_raw,
            "epi_R_singles": self.epi_R_singles,
            "epi_R_expected": self.epi_R_expected,
            "epi_score": self.epi_score,
            "cos_v12_v1": self.cos_v12_v1,
            "cos_v12_v2": self.cos_v12_v2,
            "cos_v12_vexp": self.cos_v12_vexp,
            "cos_v1_v2": self.cos_v1_v2,
            "magnitude_ratio": self.magnitude_ratio,
            "radial_deviation": self.radial_deviation,
            "epi_total": self.epi_total,
            "epi_total_bounded": self.epi_total_bounded,
            "log_rho_tri": self.log_rho_tri,
            "abs_log_rho_tri": self.abs_log_rho_tri,
            "len_WT_M12_L1": self.len_WT_M12_L1,
            "len_WT_M12_exp_L1": self.len_WT_M12_exp_L1,
            "epi_R_raw_L1": self.epi_R_raw_L1,
            "epi_R_expected_L1": self.epi_R_expected_L1,
            "magnitude_ratio_L1": self.magnitude_ratio_L1,
            "log_rho_L1": self.log_rho_L1,
            "abs_log_rho_L1": self.abs_log_rho_L1,
        }


# ---------------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------------


def _draw_arrow(
    ax: plt.Axes,
    p_from: np.ndarray,
    p_to: np.ndarray,
    color: str,
    linewidth: float = 2.0,
    alpha: float = 0.9,
    linestyle: str = "-",
    zorder: int = 3,
) -> None:
    """Draw an arrow annotation on a matplotlib axes."""
    ax.annotate(
        "",
        xy=p_to,
        xytext=p_from,
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            shrinkA=3,
            shrinkB=3,
        ),
        zorder=zorder,
    )


def _setup_epistasis_axes(ax: plt.Axes, max_radius: float) -> None:
    """Configure axes for epistasis plot."""
    circle = plt.Circle(
        (0, 0),
        1.0,
        edgecolor="0.8",
        facecolor="none",
        linestyle="--",
        linewidth=1.0,
    )
    ax.add_patch(circle)
    ax.axhline(0, color="0.85", linewidth=0.8)
    ax.axvline(0, color="0.85", linewidth=0.8)
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Geometry Base Class
# ---------------------------------------------------------------------------


class _GeometryBase:
    """
    Base class providing shared utilities and configurable distance metrics.

    Parameters
    ----------
    eps : float
        Small constant for numerical stability.
    diff : str or callable
        Distance metric: "cosine", "l2", or a custom function
        with signature (x: Tensor, y: Tensor) -> float.
    """

    def __init__(
        self,
        eps: float = DEFAULT_EPS,
        diff: Union[str, DiffFn] = "cosine",
    ):
        self.eps = eps
        self._diff_fn: DiffFn
        self.diff_name: str

        if callable(diff):
            self._diff_fn = diff
            self.diff_name = getattr(diff, "__name__", "custom")
        else:
            diff_lower = diff.lower()
            if diff_lower == "cosine":
                self._diff_fn = lambda x, y: cosine_distance(x, y, eps)
                self.diff_name = "cosine"
            elif diff_lower == "l2":
                self._diff_fn = lambda x, y: l2_distance(x, y, eps)
                self.diff_name = "l2"
            else:
                raise ValueError(
                    f"Unknown distance metric: {diff!r}. "
                    f"Use 'cosine', 'l2', or a callable."
                )

        logger.debug("Initialized geometry with diff=%s, eps=%e", self.diff_name, eps)

    def _embed_dist(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute distance between embeddings using configured metric."""
        return self._diff_fn(x, y)

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        """Pool sequence embeddings to single vector (mean pooling)."""
        h = torch.as_tensor(h).float()
        return h.mean(dim=0) if h.ndim == 2 else h

    def _safe_norm(self, v: torch.Tensor) -> float:
        """Compute L2 norm with numerical stability."""
        return float(torch.norm(v) + self.eps)

    def _unit(self, v: torch.Tensor) -> torch.Tensor:
        """Normalize vector to unit length, with fallback for zero vectors."""
        n = torch.norm(v)
        if n < self.eps:
            e = torch.zeros_like(v)
            e[0] = 1.0
            return e
        return v / n

    @staticmethod
    def _validate_embedding(emb: torch.Tensor, name: str) -> None:
        """Validate embedding tensor."""
        if not isinstance(emb, (torch.Tensor, np.ndarray)):
            raise TypeError(f"{name} must be a tensor or array, got {type(emb)}")
        if torch.as_tensor(emb).numel() == 0:
            raise ValueError(f"{name} cannot be empty")


# ---------------------------------------------------------------------------
# Single Variant Geometry
# ---------------------------------------------------------------------------


class SingleVariantGeometry(_GeometryBase):
    """
    Geometric analysis of a single genetic variant.

    Computes the effect vector v = MUT - WT and projects it into a canonical
    2D plane for visualization and comparison.

    Parameters
    ----------
    h_ref : torch.Tensor
        Wild-type (reference) embedding. Shape: (dim,) or (seq_len, dim).
    h_mut : torch.Tensor
        Mutant embedding. Shape: (dim,) or (seq_len, dim).
    eps : float, optional
        Numerical stability constant. Default: 1e-20.
    diff : str or callable, optional
        Distance metric. Default: "cosine".
    e_par_ref : torch.Tensor, optional
        Reference axis for parallel direction. If provided, enables
        angle comparisons across different variants. If None, uses
        the variant's own effect direction (angle will be 0).

    Attributes
    ----------
    WT : torch.Tensor
        Pooled wild-type embedding.
    M : torch.Tensor
        Pooled mutant embedding.
    v : torch.Tensor
        Effect vector (M - WT).

    Examples
    --------
    >>> geom = SingleVariantGeometry(h_wt, h_mut)
    >>> coords = geom.canonical_coords()
    >>> print(f"Effect magnitude: {coords.v_l2:.3f}")
    >>> geom.plot()
    """

    def __init__(
        self,
        h_ref: torch.Tensor,
        h_mut: torch.Tensor,
        eps: float = DEFAULT_EPS,
        diff: Union[str, DiffFn] = "cosine",
        e_par_ref: Optional[torch.Tensor] = None,
    ):
        super().__init__(eps=eps, diff=diff)

        # Validate inputs
        self._validate_embedding(h_ref, "h_ref")
        self._validate_embedding(h_mut, "h_mut")

        self.WT = self._pool(h_ref)
        self.M = self._pool(h_mut)
        self.v = self.M - self.WT

        self._e_par_ref = (
            torch.as_tensor(e_par_ref).float() if e_par_ref is not None else None
        )
        self._cached_coords: Optional[SingleVariantCoords] = None

        logger.debug(
            "SingleVariantGeometry initialized: dim=%d, diff=%s",
            self.WT.shape[0],
            self.diff_name,
        )

    def __repr__(self) -> str:
        coords = self.canonical_coords()
        return (
            f"SingleVariantGeometry("
            f"v_l2={coords.v_l2:.4f}, "
            f"v_diff={coords.v_diff:.4f}, "
            f"angle/π={coords.angle_over_pi:.3f}, "
            f"diff={self.diff_name!r})"
        )

    def canonical_coords(self) -> SingleVariantCoords:
        """
        Compute canonical 2D coordinates for the effect vector.

        The effect vector v is projected onto a 2D plane defined by:
        - e_par: parallel axis (reference direction or v's own direction)
        - e_perp: perpendicular axis (orthogonal to e_par)

        Coordinates are normalized by ||v|| so the result lies near the
        unit circle.

        Returns
        -------
        SingleVariantCoords
            Dataclass with all coordinate values.
        """
        if self._cached_coords is not None:
            return self._cached_coords

        v = self.v

        # Determine parallel axis
        if self._e_par_ref is not None:
            e_par = self._unit(self._e_par_ref)
        else:
            e_par = self._unit(v)

        # Build orthogonal direction
        v_perp_vec = v - torch.dot(v, e_par) * e_par
        _ = self._unit(v_perp_vec)  # e_perp computed for completeness

        v_par = float(torch.dot(v, e_par))
        v_perp = float(torch.norm(v - v_par * e_par))

        v_l2 = self._safe_norm(v)
        v_diff = self._embed_dist(self.WT, self.M)

        # Normalized coordinates
        denom = v_l2 if v_l2 > self.eps else self.eps
        x = v_par / denom
        y = v_perp / denom  # Always >= 0 (upper half-plane)

        radius = math.sqrt(x * x + y * y)
        angle = math.atan2(y, x)
        angle_over_pi = angle / math.pi

        self._cached_coords = SingleVariantCoords(
            v_l2=v_l2,
            v_diff=v_diff,
            v_par=v_par,
            v_perp=v_perp,
            x=x,
            y=y,
            radius=radius,
            angle=angle,
            angle_over_pi=angle_over_pi,
        )

        return self._cached_coords

    def metrics(self) -> dict[str, float]:
        """
        Get metrics as dictionary (alias for canonical_coords().to_dict()).

        Returns
        -------
        dict[str, float]
            Dictionary of all metric values.
        """
        return self.canonical_coords().to_dict()

    def plot(
        self,
        figsize: tuple[float, float] = (4, 4),
        annotate: bool = True,
        figure_name: str = "",
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot the single variant in the canonical 2D plane.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default: (4, 4).
        annotate : bool, optional
            Show metric annotations. Default: True.
        figure_name : str, optional
            If provided, save figure to {figure_name}.png.
        show : bool, optional
            Call plt.show(). Default: True.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        c = self.canonical_coords()

        WT_2d = np.array([0.0, 0.0])
        MUT_2d = np.array([c.x, c.y])

        fig, ax = plt.subplots(figsize=figsize)

        # Unit circle reference
        circle = plt.Circle(
            (0, 0),
            1.0,
            edgecolor="0.8",
            facecolor="none",
            linestyle="--",
            linewidth=1.0,
        )
        ax.add_patch(circle)

        # Arrow WT → MUT
        _draw_arrow(ax, WT_2d, MUT_2d, COLOR_OBSERVED, linewidth=2.0, alpha=0.95)

        # Points
        ax.scatter(*WT_2d, s=40, color=COLOR_WT, edgecolor="black", zorder=4)
        ax.scatter(*MUT_2d, s=70, color=COLOR_OBSERVED, edgecolor="black", zorder=5)

        # Labels
        ax.annotate(
            "WT",
            WT_2d,
            textcoords="offset points",
            xytext=(-8, -8),
            fontsize=9,
            fontweight="bold",
        )
        ax.annotate(
            "mut",
            MUT_2d,
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            fontweight="bold",
        )

        if annotate:
            text = (
                r"$\Delta =$ single-mutation vector"
                + "\n"
                + rf"$|\Delta|_2 = {c.v_l2:.3f}$"
                + "\n"
                + rf"$d(\mathrm{{WT}}, \mathrm{{mut}}) = {c.v_diff:.3f}$"
                + "\n"
                + rf"$\angle(\Delta)/\pi = {c.angle_over_pi:.2f}$"
            )
            ax.text(
                0.5,
                0.02,
                text,
                transform=ax.transAxes,
                fontsize=8,
                ha="center",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.95, pad=3),
            )

        # Axes styling
        ax.axhline(0, color="0.85", linewidth=0.8)
        ax.axvline(0, color="0.85", linewidth=0.8)

        max_r = max(1.2, c.radius * 1.2, 1.5)
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.set_aspect("equal")

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xlabel("Δ‖ (normalized)", fontsize=10)
        ax.set_ylabel("Δ⊥ (normalized)", fontsize=10)
        ax.set_title("Single-variant geometry", fontsize=11)

        plt.tight_layout()
        if figure_name:
            fig.savefig(f"{figure_name}.png", dpi=300, bbox_inches="tight")
            logger.info("Saved figure to %s.png", figure_name)
        if show:
            plt.show()

        return fig


# ---------------------------------------------------------------------------
# Epistasis Geometry
# ---------------------------------------------------------------------------


class EpistasisGeometry(_GeometryBase):
    """
    Geometric analysis of epistasis between two genetic variants.

    Analyzes the relationship between four embeddings:
    - WT: wild-type (reference)
    - M1: single-mutant 1
    - M2: single-mutant 2
    - M12: double-mutant

    Quantifies epistasis as deviation from additive expectation using:
    1. High-dimensional canonical complex representation
    2. Triangle-based representation using only distances
    3. MDS-based 2D embedding

    Parameters
    ----------
    h_ref : torch.Tensor
        Wild-type embedding.
    h_m1 : torch.Tensor
        Single-mutant 1 embedding.
    h_m2 : torch.Tensor
        Single-mutant 2 embedding.
    h_m12 : torch.Tensor
        Double-mutant embedding.
    eps : float, optional
        Numerical stability constant. Default: 1e-20.
    diff : str or callable, optional
        Distance metric. Default: "cosine".
    expectation_model : EpistasisExpectationModel, optional
        Trained model for predicting expected Δ₁₂. If provided, v12_exp is
        computed using the model instead of simple additivity (v1 + v2).
        Default: None (use additive expectation).

    Attributes
    ----------
    WT, M1, M2, M12 : torch.Tensor
        Pooled embeddings.
    v1, v2 : torch.Tensor
        Single-mutant effect vectors.
    v12_obs : torch.Tensor
        Observed double-mutant effect vector.
    v12_exp : torch.Tensor
        Expected double-mutant effect vector (additive or model-predicted).

    Examples
    --------
    >>> epi = EpistasisGeometry(h_wt, h_m1, h_m2, h_m12)
    >>> metrics = epi.metrics()
    >>> print(f"Epistasis magnitude: {metrics.epi_R_raw:.4f}")
    >>> epi.plot()
    >>> epi.plot_triangle()
    >>>
    >>> # With trained expectation model
    >>> epi = EpistasisGeometry(h_wt, h_m1, h_m2, h_m12, expectation_model=model)
    """

    def __init__(
        self,
        h_ref: torch.Tensor,
        h_m1: torch.Tensor,
        h_m2: torch.Tensor,
        h_m12: torch.Tensor,
        eps: float = DEFAULT_EPS,
        diff: Union[str, DiffFn] = "cosine",
        expectation_model: Optional["EpistasisExpectationModel"] = None,
    ):
        super().__init__(eps=eps, diff=diff)

        # Validate inputs
        for name, emb in [
            ("h_ref", h_ref),
            ("h_m1", h_m1),
            ("h_m2", h_m2),
            ("h_m12", h_m12),
        ]:
            self._validate_embedding(emb, name)

        # Pool inputs - keep original order (M1 = first mutation, M2 = second mutation)
        self.WT = self._pool(h_ref)
        self.M1 = self._pool(h_m1)
        self.M2 = self._pool(h_m2)
        self.M12 = self._pool(h_m12)

        # Effect vectors
        self.v1 = self.M1 - self.WT
        self.v2 = self.M2 - self.WT
        self.v12_obs = self.M12 - self.WT

        # Expected v12: use model if provided, otherwise simple additive
        self.expectation_model = expectation_model
        if expectation_model is not None:
            # Use trained model to predict expected delta
            wt_input = self.WT if expectation_model.include_wt else None
            self.v12_exp = expectation_model.predict(self.v1, self.v2, wt=wt_input)
            self._uses_learned_expectation = True
        else:
            # Simple additive expectation: v12_exp = v1 + v2
            self.v12_exp = self.v1 + self.v2
            self._uses_learned_expectation = False

        self.v_1to2 = self.M1 - self.M2

        # Cached results
        self._cached_metrics: Optional[EpistasisMetrics] = None
        self._cached_complex: Optional[EpistasisComplexCoords] = None
        self._cached_triangle: Optional[EpistasisTriangleCoords] = None
        self._cached_mds: Optional[tuple[dict[str, np.ndarray], np.ndarray]] = None

        exp_type = "learned" if self._uses_learned_expectation else "additive"
        logger.debug(
            "EpistasisGeometry initialized: dim=%d, diff=%s, expectation=%s",
            self.WT.shape[0],
            self.diff_name,
            exp_type,
        )

    def __repr__(self) -> str:
        m = self.metrics()
        exp_type = "learned" if self._uses_learned_expectation else "additive"
        return (
            f"EpistasisGeometry("
            f"R_raw={m.epi_R_raw:.4f}, "
            f"R_expected={m.epi_R_expected:.4f}, "
            f"diff={self.diff_name!r}, "
            f"expectation={exp_type!r})"
        )

    def complex_coords(self) -> EpistasisComplexCoords:
        """
        Compute canonical complex epistasis coordinates.

        Builds a 2D plane in embedding space:
        - e_par: unit vector along WT→expected (additive direction)
        - e_perp: orthogonal axis derived from (v1 - v2), giving directional
          meaning to the perpendicular component:
            * +y: deviation is "M1-like" (double mutant resembles M1 more)
            * -y: deviation is "M2-like" (double mutant resembles M2 more)

        The perpendicular axis is constructed by taking the difference vector
        (v1 - v2), which points from the M2 effect toward the M1 effect, and
        removing its component along e_par. This gives a geometrically
        meaningful interpretation: epistasis that "cancels" M2's effect will
        appear at +y, while epistasis that "cancels" M1's effect will appear
        at -y.

        Projects WT, expected, and observed into this plane, normalizing
        so expected lies at approximately (1, 0).

        Returns
        -------
        EpistasisComplexCoords
            Dataclass with complex plane coordinates.
        """
        if self._cached_complex is not None:
            return self._cached_complex

        v_exp = self.v12_exp.clone()
        if torch.norm(v_exp) < self.eps:
            v_exp = self.v12_obs.clone()

        e_par = self._unit(v_exp)

        # Perpendicular axis: derived from (v1 - v2) to give directional meaning
        # v1 - v2 points from M2's effect toward M1's effect
        # After removing the parallel component, +y means "M1-like", -y means "M2-like"
        v_diff = self.v1 - self.v2
        v_diff_perp = v_diff - torch.dot(v_diff, e_par) * e_par
        e_perp = self._unit(v_diff_perp)

        x_exp_raw = float(torch.dot(self.v12_exp, e_par))
        y_exp_raw = float(torch.dot(self.v12_exp, e_perp))
        x_obs_raw = float(torch.dot(self.v12_obs, e_par))
        y_obs_raw = float(torch.dot(self.v12_obs, e_perp))

        r_exp_raw = math.sqrt(x_exp_raw**2 + y_exp_raw**2)
        r_obs_raw = math.sqrt(x_obs_raw**2 + y_obs_raw**2)

        # Additive expectation normalization: expected at radius 1
        r_norm = r_exp_raw + self.eps

        # Geometric mean normalization (centers rho around 1 for typical interactions):
        # r_norm = math.sqrt(r_exp_raw * r_obs_raw) + self.eps

        x_exp = x_exp_raw / r_norm
        y_exp = y_exp_raw / r_norm
        x_obs = x_obs_raw / r_norm
        y_obs = y_obs_raw / r_norm

        rho = math.sqrt(x_obs**2 + y_obs**2)
        theta = math.atan2(y_obs, x_obs)
        theta_over_pi = theta / math.pi

        self._cached_complex = EpistasisComplexCoords(
            x_exp=x_exp,
            y_exp=y_exp,
            x_obs=x_obs,
            y_obs=y_obs,
            rho=rho,
            theta=theta,
            theta_over_pi=theta_over_pi,
        )

        return self._cached_complex

    def metrics(self) -> EpistasisMetrics:
        """
        Compute comprehensive epistasis metrics.

        Distances use the configured metric; vector lengths use L2 norm.

        Returns
        -------
        EpistasisMetrics
            Dataclass with all metric values.
        """
        if self._cached_metrics is not None:
            return self._cached_metrics

        v1, v2, v12, v12_exp = self.v1, self.v2, self.v12_obs, self.v12_exp

        # Vector lengths (L2)
        a1 = self._safe_norm(v1)
        a2 = self._safe_norm(v2)
        a12 = self._safe_norm(v12)
        a12_exp = self._safe_norm(v12_exp)
        a_1to2 = self._safe_norm(self.v_1to2)

        # Distances (configured metric)
        d_WT_M1 = self._embed_dist(self.WT, self.M1)
        d_WT_M2 = self._embed_dist(self.WT, self.M2)
        d_WT_M12_obs = self._embed_dist(self.WT, self.M12)

        M12_exp_point = self.WT + v12_exp
        d_WT_M12_exp = self._embed_dist(self.WT, M12_exp_point)
        d_M1_M2 = self._embed_dist(self.M1, self.M2)
        d_obs_exp = self._embed_dist(self.M12, M12_exp_point)

        # Normalized residuals
        R_raw = d_obs_exp
        single_scale = math.sqrt(a1**2 + a2**2) + self.eps
        R_singles = R_raw / single_scale
        R_expected = R_raw / (d_WT_M12_exp + self.eps)

        # Bounded epistasis score: R_raw / (R_raw + d_exp) in [0, 1]
        epi_score = R_raw / (R_raw + d_WT_M12_exp + self.eps)

        # Cosine similarities between effect vectors
        def _cosine(u: torch.Tensor, v: torch.Tensor) -> float:
            nu = self._safe_norm(u)
            nv = self._safe_norm(v)
            if nu < self.eps or nv < self.eps:
                return 0.0
            return float(torch.dot(u, v) / (nu * nv))

        cos_v12_v1 = _cosine(v12, v1)
        cos_v12_v2 = _cosine(v12, v2)
        cos_v12_vexp = _cosine(v12, v12_exp)
        cos_v1_v2 = _cosine(v1, v2)

        # Magnitude ratio and radial deviation
        magnitude_ratio = a12 / (a12_exp + self.eps)
        radial_deviation = (a12 - a12_exp) / (a12_exp + self.eps)

        # Total epistasis: distance from "no epistasis" point (cos=1, ratio=1)
        # Captures both angular AND magnitude deviation in one metric
        angle_dev = 1.0 - cos_v12_vexp  # 0 when aligned, 2 when opposite
        mag_dev = magnitude_ratio - 1.0  # 0 when equal magnitude
        epi_total = math.sqrt(angle_dev**2 + mag_dev**2)
        epi_total_bounded = epi_total / (1.0 + epi_total)

        # Log-scale magnitude ratio: high dynamic range, centered at 0
        # log(ratio) = 0 when additive, positive when super-additive, negative when sub-additive
        log_rho_tri = math.log(magnitude_ratio + self.eps)
        abs_log_rho_tri = abs(log_rho_tri)

        # L1-based metrics (Manhattan distance)
        len_WT_M12_L1 = float(torch.sum(torch.abs(v12)))
        len_WT_M12_exp_L1 = float(torch.sum(torch.abs(v12_exp)))
        residual = v12 - v12_exp
        epi_R_raw_L1 = float(torch.sum(torch.abs(residual)))
        epi_R_expected_L1 = epi_R_raw_L1 / (len_WT_M12_exp_L1 + self.eps)
        magnitude_ratio_L1 = len_WT_M12_L1 / (len_WT_M12_exp_L1 + self.eps)
        log_rho_L1 = math.log(magnitude_ratio_L1 + self.eps)
        abs_log_rho_L1 = abs(log_rho_L1)

        self._cached_metrics = EpistasisMetrics(
            dist_WT_M1=d_WT_M1,
            dist_WT_M2=d_WT_M2,
            dist_WT_M12_obs=d_WT_M12_obs,
            dist_WT_M12_exp=d_WT_M12_exp,
            dist_M1_M2=d_M1_M2,
            dist_obs_exp=d_obs_exp,
            len_WT_M1=a1,
            len_WT_M2=a2,
            len_WT_M12=a12,
            len_WT_M12_exp=a12_exp,
            len_M1_M2=a_1to2,
            epi_R_raw=R_raw,
            epi_R_singles=R_singles,
            epi_R_expected=R_expected,
            epi_score=epi_score,
            cos_v12_v1=cos_v12_v1,
            cos_v12_v2=cos_v12_v2,
            cos_v12_vexp=cos_v12_vexp,
            cos_v1_v2=cos_v1_v2,
            magnitude_ratio=magnitude_ratio,
            radial_deviation=radial_deviation,
            epi_total=epi_total,
            epi_total_bounded=epi_total_bounded,
            log_rho_tri=log_rho_tri,
            abs_log_rho_tri=abs_log_rho_tri,
            len_WT_M12_L1=len_WT_M12_L1,
            len_WT_M12_exp_L1=len_WT_M12_exp_L1,
            epi_R_raw_L1=epi_R_raw_L1,
            epi_R_expected_L1=epi_R_expected_L1,
            magnitude_ratio_L1=magnitude_ratio_L1,
            log_rho_L1=log_rho_L1,
            abs_log_rho_L1=abs_log_rho_L1,
        )

        return self._cached_metrics

    def triangle_coords(self) -> EpistasisTriangleCoords:
        """
        Compute triangle-based epistasis coordinates.

        Uses only three distances to embed the WT-Expected-Observed triangle:
        - A_o = dist(WT, observed)
        - A_e = dist(WT, expected)
        - A_r = dist(expected, observed)

        Places WT at origin, Expected at (1, 0), and Observed in upper
        half-plane using the law of cosines.

        Returns
        -------
        EpistasisTriangleCoords
            Dataclass with triangle coordinates.
        """
        if self._cached_triangle is not None:
            return self._cached_triangle

        m = self.metrics()
        A_o = m.dist_WT_M12_obs
        A_e = m.dist_WT_M12_exp
        A_r = m.dist_obs_exp

        if A_e < self.eps:
            x_exp = 1.0
            y_exp = 0.0
            x_obs = A_o / (A_e + self.eps)
            y_obs = 0.0
        else:
            x_raw = (A_o**2 + A_e**2 - A_r**2) / (2.0 * A_e)
            y_sq = max(A_o**2 - x_raw**2, 0.0)
            y_raw = math.sqrt(y_sq)

            x_exp = 1.0
            y_exp = 0.0
            x_obs = x_raw / (A_e + self.eps)
            y_obs = y_raw / (A_e + self.eps)

        rho = math.sqrt(x_obs**2 + y_obs**2)
        theta = math.atan2(y_obs, x_obs)
        theta_over_pi = theta / math.pi

        self._cached_triangle = EpistasisTriangleCoords(
            x_exp=x_exp,
            y_exp=y_exp,
            x_obs=x_obs,
            y_obs=y_obs,
            rho=rho,
            theta=theta,
            theta_over_pi=theta_over_pi,
        )

        return self._cached_triangle

    def mds_2d(
        self, random_state: int = DEFAULT_RANDOM_STATE
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Compute 2D MDS embedding of all five points.

        Uses the configured distance metric to compute pairwise distances,
        then applies MDS to embed in 2D.

        Parameters
        ----------
        random_state : int, optional
            Random seed for MDS. Default: 42.

        Returns
        -------
        coords : dict[str, np.ndarray]
            Mapping from point names to 2D coordinates.
        D : np.ndarray
            Pairwise distance matrix (5x5).
        """
        if self._cached_mds is not None:
            return self._cached_mds

        WT = self.WT
        M1 = self.M1
        M2 = self.M2
        M12 = self.M12
        # M12_exp = WT + (M1 - WT) + (M2 - WT)
        M12_exp = WT + self.v12_exp
        
        X = torch.stack([WT, M1, M2, M12, M12_exp], dim=0).float()
        n = X.shape[0]
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = self._embed_dist(X[i], X[j])

        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=20,
            max_iter=2000,
        )
        coords = mds.fit_transform(D)
        coords = coords - coords[0:1, :]  # Center on WT

        labels = ["WT", "M1", "M2", "M12_obs", "M12_exp"]
        coords_dict = {lab: coords[i] for i, lab in enumerate(labels)}

        self._cached_mds = (coords_dict, D)
        return coords_dict, D

    def plot(
        self,
        figsize: tuple[float, float] = (6, 6),
        annotate: bool = True,
        figure_name: str = "",
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot epistasis in the high-dimensional canonical complex plane.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default: (6, 6).
        annotate : bool, optional
            Show metric annotations. Default: True.
        figure_name : str, optional
            If provided, save figure to {figure_name}.png.
        show : bool, optional
            Call plt.show(). Default: True.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        c = self.complex_coords()

        WT_2d = np.array([0.0, 0.0])
        EXP_2d = np.array([c.x_exp, c.y_exp])
        OBS_2d = np.array([c.x_obs, c.y_obs])

        fig, ax = plt.subplots(figsize=figsize)

        max_r = max(1.2, c.rho * 1.2, 1.5)
        _setup_epistasis_axes(ax, max_r)

        # Arrows
        _draw_arrow(ax, WT_2d, EXP_2d, COLOR_EXPECTED, linewidth=2.0, linestyle="--")
        _draw_arrow(ax, WT_2d, OBS_2d, COLOR_OBSERVED, linewidth=2.3, alpha=0.95)
        _draw_arrow(ax, EXP_2d, OBS_2d, "black", linewidth=2.3, alpha=1.0, zorder=5)

        # Points
        ax.scatter(*WT_2d, s=50, color=COLOR_WT, edgecolor="black", zorder=4)
        ax.scatter(
            *EXP_2d,
            s=80,
            facecolor="white",
            edgecolor=COLOR_EXPECTED,
            linewidth=1.5,
            zorder=5,
        )
        ax.scatter(
            *OBS_2d, s=90, color=COLOR_OBSERVED, edgecolor="black", linewidth=1.0, zorder=6
        )

        # Labels
        ax.annotate(
            "WT",
            WT_2d,
            textcoords="offset points",
            xytext=(-8, -8),
            fontsize=10,
            fontweight="bold",
        )
        ax.annotate(
            "expected", EXP_2d, textcoords="offset points", xytext=(6, 4), fontsize=10
        )
        ax.annotate(
            "observed",
            OBS_2d,
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=10,
            fontweight="bold",
        )

        if annotate:
            text = (
                r"$\varepsilon =$ observed in canonical plane"
                + "\n"
                + rf"$|\varepsilon| = {c.rho:.2f}$  (radial, rel. to expected)"
                + "\n"
                + rf"$\angle(\varepsilon)/\pi = {c.theta_over_pi:.2f}$"
            )
            ax.text(
                0.5,
                0.02,
                text,
                transform=ax.transAxes,
                fontsize=9,
                ha="center",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.95, pad=3),
            )

        ax.set_xlabel("Δ‖ / |WT→expected|  (expected ≈ 1)", fontsize=11)
        ax.set_ylabel("Δ⊥ / |WT→expected|", fontsize=11)

        title = figure_name or "Epistasis in canonical complex plane"
        ax.set_title(title, fontsize=12)

        plt.tight_layout()
        if figure_name:
            fig.savefig(f"{figure_name}.png", dpi=300, bbox_inches="tight")
            logger.info("Saved figure to %s.png", figure_name)
        if show:
            plt.show()

        return fig

    def plot_triangle(
        self,
        figsize: tuple[float, float] = (6, 6),
        annotate: bool = True,
        figure_name: str = "",
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot triangle-based epistasis representation.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default: (6, 6).
        annotate : bool, optional
            Show metric annotations. Default: True.
        figure_name : str, optional
            If provided, save figure to {figure_name}_triangle.png.
        show : bool, optional
            Call plt.show(). Default: True.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        c = self.triangle_coords()

        WT_2d = np.array([0.0, 0.0])
        EXP_2d = np.array([c.x_exp, c.y_exp])
        OBS_2d = np.array([c.x_obs, c.y_obs])

        fig, ax = plt.subplots(figsize=figsize)

        max_r = max(1.2, c.rho * 1.2, 1.5)
        _setup_epistasis_axes(ax, max_r)

        # Arrows
        _draw_arrow(ax, WT_2d, EXP_2d, COLOR_EXPECTED, linewidth=2.0, linestyle="--")
        _draw_arrow(ax, WT_2d, OBS_2d, COLOR_OBSERVED, linewidth=2.3, alpha=0.95)
        _draw_arrow(ax, EXP_2d, OBS_2d, "black", linewidth=2.3, alpha=1.0, zorder=5)

        # Points
        ax.scatter(*WT_2d, s=50, color=COLOR_WT, edgecolor="black", zorder=4)
        ax.scatter(
            *EXP_2d,
            s=80,
            facecolor="white",
            edgecolor=COLOR_EXPECTED,
            linewidth=1.5,
            zorder=5,
        )
        ax.scatter(
            *OBS_2d, s=90, color=COLOR_OBSERVED, edgecolor="black", linewidth=1.0, zorder=6
        )

        # Labels
        ax.annotate(
            "WT",
            WT_2d,
            textcoords="offset points",
            xytext=(-8, -8),
            fontsize=10,
            fontweight="bold",
        )
        ax.annotate(
            "expected", EXP_2d, textcoords="offset points", xytext=(6, 4), fontsize=10
        )
        ax.annotate(
            "observed",
            OBS_2d,
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=10,
            fontweight="bold",
        )

        if annotate:
            text = (
                r"$\varepsilon_{\triangle} =$ triangle-based"
                + "\n"
                + rf"$|\varepsilon_{{\triangle}}| = {c.rho:.2f}$"
                + "\n"
                + rf"$\angle(\varepsilon_{{\triangle}})/\pi = {c.theta_over_pi:.2f}$"
            )
            ax.text(
                0.5,
                0.02,
                text,
                transform=ax.transAxes,
                fontsize=9,
                ha="center",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.95, pad=3),
            )

        ax.set_xlabel("simplified Δ‖  (expected fixed at 1)", fontsize=11)
        ax.set_ylabel("simplified Δ⊥", fontsize=11)

        title = figure_name or "Triangle-based epistasis"
        ax.set_title(title, fontsize=12)

        plt.tight_layout()
        if figure_name:
            fig.savefig(f"{figure_name}_triangle.png", dpi=300, bbox_inches="tight")
            logger.info("Saved figure to %s_triangle.png", figure_name)
        if show:
            plt.show()

        return fig

    def plot_story(
        self,
        figsize: tuple[float, float] = (12, 10),
        figure_name: str = "",
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot the epistasis story as a visual narrative.

        Shows the journey from WT through individual mutations to the double
        mutant, highlighting the deviation from additive expectation.
        Projects to 2D using PCA, scaled to emphasize the epistasis.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default: (10, 6).
        figure_name : str, optional
            If provided, save figure to {figure_name}.png.
        show : bool, optional
            Call plt.show(). Default: True.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        # Use MDS-like approach: preserve pairwise distances as best as possible
        # Stack the 5 points: WT, M1, M2, M12_exp, M12_obs
        M12_exp = self.WT + self.v12_exp

        points = [self.WT, self.M1, self.M2, M12_exp, self.M12]
        n = len(points)

        # Compute pairwise distance matrix using the configured distance metric
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = self._embed_dist(points[i], points[j])

        # Classical MDS: convert distances to coordinates
        # Center the squared distance matrix
        D_sq = D ** 2
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ D_sq @ H

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(B)
        # Sort by eigenvalue descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Take top 2 dimensions
        coords_2d = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))

        WT_2d = coords_2d[0]
        M1_2d = coords_2d[1]
        M2_2d = coords_2d[2]
        EXP_2d = coords_2d[3]
        OBS_2d = coords_2d[4]

        # Center on WT
        M1_2d = M1_2d - WT_2d
        M2_2d = M2_2d - WT_2d
        EXP_2d = EXP_2d - WT_2d
        OBS_2d = OBS_2d - WT_2d
        WT_2d = np.array([0.0, 0.0])

        # Get metrics
        m = self.metrics()
        c = self.complex_coords()
        epistasis_R = m.epi_R_expected
        rho = c.rho

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#fafafa")

        # Subtle parallelogram fill to show additive expectation region
        parallelogram = plt.Polygon(
            [WT_2d, M1_2d, EXP_2d, M2_2d],
            fill=True,
            facecolor="#e8e8e8",
            edgecolor="0.7",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            zorder=1,
        )
        ax.add_patch(parallelogram)

        # Draw the story arrows with varying thickness to show narrative flow
        # WT -> M1 (first mutation)
        _draw_arrow(ax, WT_2d, M1_2d, COLOR_M1, linewidth=2.5, alpha=0.85)
        # WT -> M2 (second mutation)
        _draw_arrow(ax, WT_2d, M2_2d, COLOR_M2, linewidth=2.5, alpha=0.85)
        # M1 -> Expected (what M2 would add)
        _draw_arrow(ax, M1_2d, EXP_2d, COLOR_M2, linewidth=1.5, alpha=0.4, linestyle="--")
        # M2 -> Expected (what M1 would add)
        _draw_arrow(ax, M2_2d, EXP_2d, COLOR_M1, linewidth=1.5, alpha=0.4, linestyle="--")

        # The key: WT -> Observed (actual double mutant) - bold
        _draw_arrow(ax, WT_2d, OBS_2d, COLOR_OBSERVED, linewidth=3.5, alpha=0.95)

        # Epistasis vector: expected -> observed (the punchline)
        _draw_arrow(ax, EXP_2d, OBS_2d, "black", linewidth=3.0, alpha=1.0, zorder=10)

        # Points - sized by importance in the story
        ax.scatter(*WT_2d, s=180, color=COLOR_WT, edgecolor="white", linewidth=2, zorder=15)
        ax.scatter(*M1_2d, s=120, color=COLOR_M1, edgecolor="white", linewidth=1.5, zorder=14)
        ax.scatter(*M2_2d, s=120, color=COLOR_M2, edgecolor="white", linewidth=1.5, zorder=14)
        ax.scatter(*EXP_2d, s=140, facecolor="white", edgecolor="0.4", linewidth=2.5, zorder=16)
        ax.scatter(*OBS_2d, s=200, color=COLOR_OBSERVED, edgecolor="white", linewidth=2, zorder=17)

        # Clean labels
        fs_label = 11
        ax.annotate("WT", WT_2d, textcoords="offset points", xytext=(-12, -12),
                    fontsize=fs_label, fontweight="bold", color=COLOR_WT)
        ax.annotate("M1", M1_2d, textcoords="offset points", xytext=(8, 8),
                    fontsize=fs_label, fontweight="bold", color=COLOR_M1)
        ax.annotate("M2", M2_2d, textcoords="offset points", xytext=(8, 8),
                    fontsize=fs_label, fontweight="bold", color=COLOR_M2)
        ax.annotate("expected", EXP_2d, textcoords="offset points", xytext=(10, -12),
                    fontsize=10, color="0.4", style="italic")
        ax.annotate("M12", OBS_2d, textcoords="offset points", xytext=(10, 8),
                    fontsize=fs_label, fontweight="bold", color=COLOR_OBSERVED)

        # Scale to fit with minimum spacing - compute first so we can use it for label offset
        all_points = np.array([WT_2d, M1_2d, M2_2d, EXP_2d, OBS_2d])
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range, 1e-9)

        # Padding relative to data range
        pad = max_range * 0.15

        # Center the view
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        half_span = (max_range / 2) + pad

        ax.set_xlim(x_center - half_span, x_center + half_span)
        ax.set_ylim(y_center - half_span, y_center + half_span)
        ax.set_aspect("equal")

        # Epistasis label on the black arrow - offset relative to plot scale
        epi_midpoint = (EXP_2d + OBS_2d) / 2
        epi_vec = OBS_2d - EXP_2d
        # Perpendicular offset scaled to plot size
        perp = np.array([-epi_vec[1], epi_vec[0]])
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 1e-9:
            perp = perp / perp_norm * max_range * 0.05  # 5% of plot range
        ax.annotate(
            "epistasis",
            epi_midpoint + perp,
            fontsize=9,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
        )

        # Minimal axis styling
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Display metrics: R (normalized epistasis) and rho (complex magnitude)
        ax.text(
            0.98, 0.02,
            f"R = {epistasis_R:.3f}\nρ = {rho:.3f}",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="right",
            va="bottom",
            color="black",
            bbox=dict(facecolor="white", edgecolor="0.3", alpha=0.9, pad=6, boxstyle="round,pad=0.4"),
        )

        # Title
        title = figure_name if figure_name else "Epistasis Story"
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

        plt.tight_layout()
        if figure_name:
            fig.savefig(f"{figure_name}.png", dpi=300, bbox_inches="tight")
            logger.info("Saved figure to %s.png", figure_name)
        if show:
            plt.show()

        return fig


# ---------------------------------------------------------------------------
# Variant Embedding Database
# ---------------------------------------------------------------------------


@dataclass
class DBMetadata:
    """Metadata stored in the database."""

    version: str = __version__
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    model_name: Optional[str] = None
    embedding_dim: Optional[int] = None
    description: str = ""


class VariantEmbeddingDB:
    """
    SQLite-backed key-value store for variant embeddings.

    Provides fast random access to stored embeddings with optional metadata
    tracking. Supports both numpy arrays and PyTorch tensors.

    Parameters
    ----------
    db_path : str or Path
        Path to SQLite database file (created if doesn't exist).

    Attributes
    ----------
    db_path : Path
        Database file path.

    Examples
    --------
    >>> # Using as context manager (recommended)
    >>> with VariantEmbeddingDB("variants.db") as db:
    ...     db.store("GENE:1:12345:A:G", embedding)
    ...     emb = db.load("GENE:1:12345:A:G")
    ...
    >>> # Manual management
    >>> db = VariantEmbeddingDB("variants.db")
    >>> db.store("variant_id", embedding)
    >>> db.close()
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()

        logger.info("Opened VariantEmbeddingDB at %s", self.db_path)

    def _connect(self) -> None:
        """Establish database connection and initialize schema."""
        self._conn = sqlite3.connect(
            self.db_path, isolation_level=None, timeout=30.0
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")

        # Main embeddings table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                mut_id TEXT PRIMARY KEY,
                dim    INTEGER NOT NULL,
                dtype  TEXT    NOT NULL,
                data   BLOB    NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mut_id ON embeddings(mut_id);"
        )

        # Metadata table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Reusable cursors for performance
        self._cur_has = self._conn.cursor()
        self._cur_get = self._conn.cursor()
        self._cur_put = self._conn.cursor()

    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection, reconnecting if necessary."""
        if self._conn is None:
            self._connect()
        return self._conn  # type: ignore

    def __enter__(self) -> "VariantEmbeddingDB":
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        count = self.count()
        return f"VariantEmbeddingDB(path={self.db_path!r}, count={count})"

    def __len__(self) -> int:
        """Return number of stored embeddings."""
        return self.count()

    def __contains__(self, mut_id: str) -> bool:
        """Check if mutation ID exists in database."""
        return self.has(mut_id)

    # ---------- Metadata API ----------

    def set_metadata(self, key: str, value: str) -> None:
        """Store a metadata key-value pair."""
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        logger.debug("Set metadata: %s = %s", key, value)

    def get_metadata(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a metadata value."""
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else default

    def get_all_metadata(self) -> dict[str, str]:
        """Retrieve all metadata as a dictionary."""
        rows = self.conn.execute("SELECT key, value FROM metadata").fetchall()
        return dict(rows)

    def store_db_metadata(self, meta: DBMetadata) -> None:
        """Store structured metadata."""
        self.set_metadata("version", meta.version)
        self.set_metadata("created_at", meta.created_at)
        if meta.model_name:
            self.set_metadata("model_name", meta.model_name)
        if meta.embedding_dim:
            self.set_metadata("embedding_dim", str(meta.embedding_dim))
        if meta.description:
            self.set_metadata("description", meta.description)

    # ---------- Core API ----------

    def has(self, mut_id: str) -> bool:
        """
        Check if a mutation ID exists in the database.

        Parameters
        ----------
        mut_id : str
            Mutation identifier.

        Returns
        -------
        bool
            True if the mutation exists.
        """
        row = self._cur_has.execute(
            "SELECT 1 FROM embeddings WHERE mut_id = ? LIMIT 1", (mut_id,)
        ).fetchone()
        return row is not None

    def store(self, mut_id: str, emb: "TensorLike") -> None:
        """
        Store an embedding for a mutation ID.

        Parameters
        ----------
        mut_id : str
            Mutation identifier (used as key).
        emb : array-like
            Embedding vector (numpy array or torch tensor).
            Will be flattened and stored as float32.

        Notes
        -----
        Overwrites any existing embedding with the same mut_id.
        """
        if isinstance(emb, torch.Tensor):
            arr = emb.detach().cpu().numpy()
        else:
            arr = np.asarray(emb)

        arr = np.ascontiguousarray(arr.flatten().astype(np.float32))
        dim = arr.size
        dtype = str(arr.dtype)
        blob = arr.tobytes()

        self._cur_put.execute(
            "INSERT OR REPLACE INTO embeddings (mut_id, dim, dtype, data) "
            "VALUES (?, ?, ?, ?)",
            (mut_id, dim, dtype, blob),
        )
        logger.debug("Stored embedding for %s (dim=%d)", mut_id, dim)

    def load(
        self, mut_id: str, as_torch: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Load an embedding by mutation ID.

        Parameters
        ----------
        mut_id : str
            Mutation identifier.
        as_torch : bool, optional
            Return as torch.Tensor if True, else numpy array. Default: True.

        Returns
        -------
        array-like
            The stored embedding vector.

        Raises
        ------
        KeyError
            If the mutation ID is not found.
        """
        row = self._cur_get.execute(
            "SELECT dim, dtype, data FROM embeddings WHERE mut_id = ?", (mut_id,)
        ).fetchone()

        if row is None:
            raise KeyError(f"Mutation {mut_id!r} not found in database")

        dim, dtype, blob = row
        arr = np.frombuffer(blob, dtype=dtype).reshape(dim).copy()

        if as_torch:
            return torch.from_numpy(arr)
        return arr

    def load_batch(
        self, mut_ids: list[str], as_torch: bool = True
    ) -> dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Load multiple embeddings at once.

        Parameters
        ----------
        mut_ids : list[str]
            List of mutation identifiers.
        as_torch : bool, optional
            Return as torch.Tensor if True. Default: True.

        Returns
        -------
        dict
            Mapping from mut_id to embedding (skips missing IDs).
        """
        result = {}
        for mut_id in mut_ids:
            try:
                result[mut_id] = self.load(mut_id, as_torch=as_torch)
            except KeyError:
                logger.warning("Mutation %s not found, skipping", mut_id)
        return result

    def delete(self, mut_id: str) -> bool:
        """
        Delete an embedding by mutation ID.

        Parameters
        ----------
        mut_id : str
            Mutation identifier.

        Returns
        -------
        bool
            True if an embedding was deleted, False if not found.
        """
        cursor = self.conn.execute(
            "DELETE FROM embeddings WHERE mut_id = ?", (mut_id,)
        )
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("Deleted embedding for %s", mut_id)
        return deleted

    def count(self) -> int:
        """
        Count total number of stored embeddings.

        Returns
        -------
        int
            Number of embeddings in database.
        """
        row = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return row[0] if row else 0

    def iter_all(
        self, batch_size: int = 1024, as_torch: bool = False
    ) -> Iterator[tuple[str, Union[np.ndarray, torch.Tensor]]]:
        """
        Iterate over all stored embeddings.

        Parameters
        ----------
        batch_size : int, optional
            Number of rows to fetch per batch. Default: 1024.
        as_torch : bool, optional
            Yield as torch.Tensor if True. Default: False.

        Yields
        ------
        tuple[str, array-like]
            (mut_id, embedding) pairs.
        """
        cur = self.conn.cursor()
        offset = 0
        while True:
            rows = cur.execute(
                "SELECT mut_id, dim, dtype, data FROM embeddings "
                "LIMIT ? OFFSET ?",
                (batch_size, offset),
            ).fetchall()
            if not rows:
                break
            offset += len(rows)
            for mut_id, dim, dtype, blob in rows:
                arr = np.frombuffer(blob, dtype=dtype).reshape(dim).copy()
                if as_torch:
                    yield mut_id, torch.from_numpy(arr)
                else:
                    yield mut_id, arr

    def list_keys(self, pattern: Optional[str] = None) -> list[str]:
        """
        List all mutation IDs, optionally filtered by pattern.

        Parameters
        ----------
        pattern : str, optional
            SQL LIKE pattern for filtering (e.g., "BRCA1:%").

        Returns
        -------
        list[str]
            List of matching mutation IDs.
        """
        if pattern:
            rows = self.conn.execute(
                "SELECT mut_id FROM embeddings WHERE mut_id LIKE ?", (pattern,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT mut_id FROM embeddings").fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("Closed VariantEmbeddingDB at %s", self.db_path)


# ---------------------------------------------------------------------------
# Mutation ID Parsing
# ---------------------------------------------------------------------------


def parse_single_mut_id(mut_id: str) -> tuple[str, int, str, str, bool]:
    """
    Parse a single mutation identifier.

    Supports formats:
    - GENE:CHROM:POS:REF:ALT
    - GENE:CHROM:POS:REF:ALT:STRAND (P=positive, N=negative)

    Parameters
    ----------
    mut_id : str
        Mutation identifier string.

    Returns
    -------
    tuple
        (chrom, pos, ref, alt, reverse_complement)

    Raises
    ------
    ValueError
        If format is invalid.

    Examples
    --------
    >>> parse_single_mut_id("BRCA1:17:43071077:A:G")
    ('17', 43071077, 'A', 'G', False)
    >>> parse_single_mut_id("KRAS:12:25227342:G:C:N")
    ('12', 25227342, 'G', 'C', True)
    """
    parts = mut_id.split(":")

    if len(parts) == 6:
        _, chrom, pos, ref, alt, strand = parts
        if strand == "P":
            rev = False
        elif strand == "N":
            rev = True
        else:
            logger.warning("Unknown strand %r, defaulting to positive", strand)
            rev = False
    elif len(parts) == 5:
        _, chrom, pos, ref, alt = parts
        rev = False
    else:
        raise ValueError(
            f"Invalid mut_id format: {mut_id!r}. "
            f"Expected GENE:CHROM:POS:REF:ALT or GENE:CHROM:POS:REF:ALT:STRAND"
        )

    return chrom, int(pos), ref, alt, rev


def parse_epistasis_id(
    epi_id: str,
) -> tuple[tuple[str, int, str, str, bool], tuple[str, int, str, str, bool]]:
    """
    Parse an epistasis identifier (two mutations).

    Format: mut1|mut2 where each follows parse_single_mut_id format.

    Parameters
    ----------
    epi_id : str
        Epistasis identifier string.

    Returns
    -------
    tuple
        ((chrom1, pos1, ref1, alt1, rev1), (chrom2, pos2, ref2, alt2, rev2))

    Raises
    ------
    ValueError
        If format is invalid or mutations are on different chromosomes/strands.

    Examples
    --------
    >>> parse_epistasis_id("KRAS:12:25227342:G:C|KRAS:12:25227344:A:T")
    (('12', 25227342, 'G', 'C', False), ('12', 25227344, 'A', 'T', False))
    """
    if "|" not in epi_id:
        raise ValueError(f"Epistasis ID must contain '|': {epi_id!r}")

    mut1_str, mut2_str = epi_id.split("|", 1)
    mut1 = parse_single_mut_id(mut1_str)
    mut2 = parse_single_mut_id(mut2_str)

    chrom1, _, _, _, rev1 = mut1
    chrom2, _, _, _, rev2 = mut2

    if chrom1 != chrom2:
        raise ValueError(
            f"Epistasis across chromosomes not supported: {chrom1} vs {chrom2}"
        )
    if rev1 != rev2:
        raise ValueError(f"Mixed strand info in epistasis_id: {epi_id!r}")

    return mut1, mut2


# ---------------------------------------------------------------------------
# Embedding Generation Functions
# ---------------------------------------------------------------------------


def embed_single_variant(
    model,
    mut_id: str,
    context: int = DEFAULT_CONTEXT,
    genome: str = DEFAULT_GENOME,
    pool: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate embeddings for a single variant.

    Parameters
    ----------
    model : object
        Model with `.embed(sequence: str, pool=...) -> Tensor` method.
    mut_id : str
        Mutation identifier (see parse_single_mut_id).
    context : int, optional
        Context window size (bases on each side). Default: 3000.
    genome : str, optional
        Genome assembly name. Default: "hg38".
    pool : str, optional
        Pooling method for embeddings. "mean" for mean-pooled (default),
        "tokens" for full token embeddings (flattened).

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        (h_ref, h_mut, delta) where delta = h_mut - h_ref.
    """
    from seqmat import SeqMat

    chrom, pos, ref, alt, rev = parse_single_mut_id(mut_id)

    start = pos - context
    end = pos + context - 1

    logger.debug(
        "Embedding variant %s: chr%s:%d-%d (rev=%s)",
        mut_id,
        chrom,
        start,
        end,
        rev,
    )

    s = SeqMat.from_fasta(genome, f"chr{chrom}", start, end)
    if rev:
        s.reverse_complement()

    m = s.clone()
    m.apply_mutations([(pos, ref, alt)])

    h_ref = model.embed(s.seq, pool=pool)
    h_mut = model.embed(m.seq, pool=pool)
    delta = h_mut - h_ref

    return h_ref, h_mut, delta


def embed_epistasis(
    model,
    epistasis_id: str,
    context: int = DEFAULT_CONTEXT,
    genome: str = DEFAULT_GENOME,
    pool: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate embeddings for an epistasis pair.

    Parameters
    ----------
    model : object
        Model with `.embed(sequence: str, pool=...) -> Tensor` method.
    epistasis_id : str
        Epistasis identifier (see parse_epistasis_id).
    context : int, optional
        Context window size. Default: 3000.
    genome : str, optional
        Genome assembly name. Default: "hg38".
    pool : str, optional
        Pooling method for embeddings. "mean" for mean-pooled (default),
        "tokens" for full token embeddings (flattened).

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        (h_ref, h_m1, h_m2, h_m12) embeddings.
    """
    from seqmat import SeqMat

    (chrom, pos1, ref1, alt1, rev), (_, pos2, ref2, alt2, _) = parse_epistasis_id(
        epistasis_id
    )

    p_min = min(pos1, pos2)
    p_max = max(pos1, pos2)
    start = p_min - context
    end = p_max + context - 1

    logger.debug(
        "Embedding epistasis %s: chr%s:%d-%d (rev=%s)",
        epistasis_id,
        chrom,
        start,
        end,
        rev,
    )

    s = SeqMat.from_fasta(genome, f"chr{chrom}", start, end)
    if rev:
        s.reverse_complement()

    var1 = (pos1, ref1, alt1)
    var2 = (pos2, ref2, alt2)

    m1 = s.clone()
    m2 = s.clone()
    m12 = s.clone()

    m1.apply_mutations([var1])
    m2.apply_mutations([var2])
    m12.apply_mutations([var1, var2])

    h_ref = model.embed(s.seq, pool=pool)
    h_m1 = model.embed(m1.seq, pool=pool)
    h_m2 = model.embed(m2.seq, pool=pool)
    h_m12 = model.embed(m12.seq, pool=pool)

    return h_ref, h_m1, h_m2, h_m12



def embed_epistasis_to_gene(
    model,
    epistasis_id: str,
    context: int = DEFAULT_CONTEXT,
    genome: str = DEFAULT_GENOME,
    pool: str = "mean",
    mature_mrna: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate embeddings for an epistasis pair.

    Parameters
    ----------
    model : object
        Model with `.embed(sequence: str, pool=...) -> Tensor` method.
    epistasis_id : str
        Epistasis identifier (see parse_epistasis_id).
    context : int, optional
        Context window size. Default: 3000.
    genome : str, optional
        Genome assembly name. Default: "hg38".
    pool : str, optional
        Pooling method for embeddings. "mean" for mean-pooled (default),
        "tokens" for full token embeddings (flattened).

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        (h_ref, h_m1, h_m2, h_m12) embeddings.
    """
    from seqmat.gene import Gene

    gene = epistasis_id.split(':')[0]
    (chrom, pos1, ref1, alt1, rev), (_, pos2, ref2, alt2, _) = parse_epistasis_id(
        epistasis_id
    )

    p_min = min(pos1, pos2)
    p_max = max(pos1, pos2)
    start = p_min - context
    end = p_max + context - 1

    logger.debug(
        "Embedding epistasis %s: chr%s:%d-%d (rev=%s)",
        epistasis_id,
        chrom,
        start,
        end,
        rev,
    )

    s = Gene.from_file(gene).transcript().generate_pre_mrna().generate_mature_mrna().mature_mrna
    # if rev:
    #     s.reverse_complement()

    var1 = (pos1, ref1, alt1)
    var2 = (pos2, ref2, alt2)

    m1 = s.clone()
    m2 = s.clone()
    m12 = s.clone()

    m1.apply_mutations([var1])
    m2.apply_mutations([var2])
    m12.apply_mutations([var1, var2])

    # seq = s.seq
    # if len(seq) > context * 2:
    #     seq = seq[:context * 2]

    h_ref = model.embed(s.seq, pool=pool)
    h_m1 = model.embed(m1.seq, pool=pool)
    h_m2 = model.embed(m2.seq, pool=pool)
    h_m12 = model.embed(m12.seq, pool=pool)

    return h_ref, h_m1, h_m2, h_m12



# ---------------------------------------------------------------------------
# Database Storage Functions
# ---------------------------------------------------------------------------


def store_single_variant_embedding(
    model,
    mut_id: str,
    db: VariantEmbeddingDB,
    context: int = DEFAULT_CONTEXT,
    genome: str = DEFAULT_GENOME,
    overwrite: bool = False,
    store_wt_mut: bool = False,
) -> bool:
    """
    Compute and store embedding for a single variant.

    Parameters
    ----------
    model : object
        Model with `.embed()` method.
    mut_id : str
        Mutation identifier.
    db : VariantEmbeddingDB
        Database for storage.
    context : int, optional
        Context window size. Default: 3000.
    genome : str, optional
        Genome assembly. Default: "hg38".
    overwrite : bool, optional
        Overwrite existing entries. Default: False.
    store_wt_mut : bool, optional
        Also store WT and MUT embeddings under {mut_id}|WT and {mut_id}|MUT.
        Default: False.

    Returns
    -------
    bool
        True if embedding was computed (not skipped).
    """
    if not overwrite and db.has(mut_id):
        logger.debug("Skipping %s (already exists)", mut_id)
        return False

    h_ref, h_mut, delta = embed_single_variant(
        model=model,
        mut_id=mut_id,
        context=context,
        genome=genome,
    )

    db.store(mut_id, delta)

    if store_wt_mut:
        db.store(f"{mut_id}{KEY_WT}", h_ref)
        db.store(f"{mut_id}{KEY_MUT}", h_mut)

    logger.debug("Stored embedding for %s", mut_id)
    return True


def store_epistasis_embeddings(
    model,
    epistasis_id: str,
    db: VariantEmbeddingDB,
    context: int = DEFAULT_CONTEXT,
    genome: str = DEFAULT_GENOME,
    overwrite: bool = False,
    store_deltas: bool = True,
) -> bool:
    """
    Compute and store embeddings for an epistasis pair.

    Parameters
    ----------
    model : object
        Model with `.embed()` method.
    epistasis_id : str
        Epistasis identifier.
    db : VariantEmbeddingDB
        Database for storage.
    context : int, optional
        Context window size. Default: 3000.
    genome : str, optional
        Genome assembly. Default: "hg38".
    overwrite : bool, optional
        Overwrite existing entries. Default: False.
    store_deltas : bool, optional
        Also store delta embeddings (Δ1, Δ2, Δ12). Default: True.

    Returns
    -------
    bool
        True if embeddings were computed (not skipped).
    """
    keys_abs = [
        f"{epistasis_id}{KEY_WT}",
        f"{epistasis_id}{KEY_M1}",
        f"{epistasis_id}{KEY_M2}",
        f"{epistasis_id}{KEY_M12}",
    ]

    if not overwrite and all(db.has(k) for k in keys_abs):
        logger.debug("Skipping %s (already exists)", epistasis_id)
        return False

    h_ref, h_m1, h_m2, h_m12 = embed_epistasis(
        model=model,
        epistasis_id=epistasis_id,
        context=context,
        genome=genome,
    )

    db.store(f"{epistasis_id}{KEY_WT}", h_ref)
    db.store(f"{epistasis_id}{KEY_M1}", h_m1)
    db.store(f"{epistasis_id}{KEY_M2}", h_m2)
    db.store(f"{epistasis_id}{KEY_M12}", h_m12)

    if store_deltas:
        db.store(f"{epistasis_id}{KEY_DELTA1}", h_m1 - h_ref)
        db.store(f"{epistasis_id}{KEY_DELTA2}", h_m2 - h_ref)
        db.store(f"{epistasis_id}{KEY_DELTA12}", h_m12 - h_ref)

    logger.debug("Stored epistasis embeddings for %s", epistasis_id)
    return True


def bulk_store_single_variants(
    model,
    mut_ids: list[str],
    db: VariantEmbeddingDB,
    context: int = DEFAULT_CONTEXT,
    genome: str = DEFAULT_GENOME,
    overwrite: bool = False,
    store_wt_mut: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> int:
    """
    Compute and store embeddings for multiple single variants.

    Parameters
    ----------
    model : object
        Model with `.embed()` method.
    mut_ids : list[str]
        List of mutation identifiers.
    db : VariantEmbeddingDB
        Database for storage.
    context : int, optional
        Context window size. Default: 3000.
    genome : str, optional
        Genome assembly. Default: "hg38".
    overwrite : bool, optional
        Overwrite existing entries. Default: False.
    store_wt_mut : bool, optional
        Also store WT and MUT embeddings. Default: False.
    progress_callback : callable, optional
        Called with (current_index, total, mut_id) for each variant.

    Returns
    -------
    int
        Number of embeddings actually computed (not skipped).
    """
    total = len(mut_ids)
    computed = 0

    logger.info("Processing %d single variants", total)

    for i, mut_id in enumerate(mut_ids):
        if progress_callback:
            progress_callback(i, total, mut_id)

        was_computed = store_single_variant_embedding(
            model,
            mut_id,
            db,
            context=context,
            genome=genome,
            overwrite=overwrite,
            store_wt_mut=store_wt_mut,
        )
        if was_computed:
            computed += 1

    logger.info("Completed: %d/%d computed (%d skipped)", computed, total, total - computed)
    return computed


def bulk_store_epistasis(
    model,
    epistasis_ids: list[str],
    db: VariantEmbeddingDB,
    context: int = DEFAULT_CONTEXT,
    genome: str = DEFAULT_GENOME,
    overwrite: bool = False,
    store_deltas: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> int:
    """
    Compute and store embeddings for multiple epistasis pairs.

    Parameters
    ----------
    model : object
        Model with `.embed()` method.
    epistasis_ids : list[str]
        List of epistasis identifiers.
    db : VariantEmbeddingDB
        Database for storage.
    context : int, optional
        Context window size. Default: 3000.
    genome : str, optional
        Genome assembly. Default: "hg38".
    overwrite : bool, optional
        Overwrite existing entries. Default: False.
    store_deltas : bool, optional
        Also store delta embeddings. Default: True.
    progress_callback : callable, optional
        Called with (current_index, total, epistasis_id) for each pair.

    Returns
    -------
    int
        Number of epistasis pairs actually computed (not skipped).
    """
    total = len(epistasis_ids)
    computed = 0

    logger.info("Processing %d epistasis pairs", total)

    for i, epi_id in enumerate(epistasis_ids):
        if progress_callback:
            progress_callback(i, total, epi_id)

        was_computed = store_epistasis_embeddings(
            model,
            epi_id,
            db,
            context=context,
            genome=genome,
            overwrite=overwrite,
            store_deltas=store_deltas,
        )
        if was_computed:
            computed += 1

    logger.info("Completed: %d/%d computed (%d skipped)", computed, total, total - computed)
    return computed


# ---------------------------------------------------------------------------
# DataFrame Integration
# ---------------------------------------------------------------------------

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
    pd = None  # type: ignore


def _require_pandas():
    """Raise ImportError if pandas is not available."""
    if not _HAS_PANDAS:
        raise ImportError(
            "pandas is required for DataFrame operations. "
            "Install it with: pip install pandas"
        )


def _parse_strand(value) -> bool:
    """
    Parse strand value to boolean (True = reverse complement).

    Accepts: "-", "N", "negative", False, 0 -> True (reverse)
             "+", "P", "positive", True, 1  -> False (no reverse)
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("-", "n", "negative", "neg", "reverse", "rev", "-1"):
            return True
        if value in ("+", "p", "positive", "pos", "forward", "fwd", "1", ""):
            return False
    raise ValueError(f"Cannot parse strand value: {value!r}")


def _embed_single_variant_direct(
    model,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    reverse_complement: bool = False,
    context: int = DEFAULT_CONTEXT,
    genome: str = DEFAULT_GENOME,
    pool: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute embeddings for a single variant from genomic coordinates.

    Returns (h_wt, h_mut, delta) embeddings with specified pooling.
    """
    from seqmat import SeqMat

    start = pos - context
    end = pos + context - 1

    logger.debug(
        "Embedding variant chr%s:%d %s>%s (rev=%s)",
        chrom, pos, ref, alt, reverse_complement
    )

    s = SeqMat.from_fasta(genome, f"chr{chrom}", start, end)
    if reverse_complement:
        s.reverse_complement()

    m = s.clone()
    m.apply_mutations([(pos, ref, alt)])

    h_wt = model.embed(s.seq, pool=pool)
    h_mut = model.embed(m.seq, pool=pool)
    delta = h_mut - h_wt

    return h_wt, h_mut, delta


def _embed_epistasis_direct(
    model,
    chrom: str,
    pos1: int,
    ref1: str,
    alt1: str,
    pos2: int,
    ref2: str,
    alt2: str,
    reverse_complement: bool = False,
    context: int = DEFAULT_CONTEXT,
    genome: str = DEFAULT_GENOME,
    pool: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute embeddings for epistasis from genomic coordinates.

    Returns (h_wt, h_m1, h_m2, h_m12) embeddings with specified pooling.
    """
    from seqmat import SeqMat

    p_min = min(pos1, pos2)
    p_max = max(pos1, pos2)
    start = p_min - context
    end = p_max + context - 1

    logger.debug(
        "Embedding epistasis chr%s:%d,%d (rev=%s)",
        chrom, pos1, pos2, reverse_complement
    )

    s = SeqMat.from_fasta(genome, f"chr{chrom}", start, end)
    if reverse_complement:
        s.reverse_complement()

    var1 = (pos1, ref1, alt1)
    var2 = (pos2, ref2, alt2)

    m1 = s.clone()
    m2 = s.clone()
    m12 = s.clone()

    m1.apply_mutations([var1])
    m2.apply_mutations([var2])
    m12.apply_mutations([var1, var2])

    h_wt = model.embed(s.seq, pool=pool)
    h_m1 = model.embed(m1.seq, pool=pool)
    h_m2 = model.embed(m2.seq, pool=pool)
    h_m12 = model.embed(m12.seq, pool=pool)

    return h_wt, h_m1, h_m2, h_m12


def add_single_variant_metrics(
    df: "pd.DataFrame",
    db: VariantEmbeddingDB,
    model=None,
    id_col: str = "mut_id",
    chrom_col: Optional[str] = None,
    pos_col: Optional[str] = None,
    ref_col: Optional[str] = None,
    alt_col: Optional[str] = None,
    strand_col: Optional[str] = None,
    reverse_complement: bool = False,
    context: int = DEFAULT_CONTEXT,
    genome: str = DEFAULT_GENOME,
    diff: Union[str, DiffFn] = "cosine",
    prefix: str = "",
    inplace: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    pool: str = "mean",
    show_progress: bool = True,
) -> "pd.DataFrame":
    """
    Add single-variant geometry metrics to a DataFrame.

    For each row:
    1. Check if embeddings exist in database
    2. If not and model is provided, compute and save embeddings
    3. Compute geometric metrics using SingleVariantGeometry

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing mutation data.
    db : VariantEmbeddingDB
        Database for storing/loading embeddings.
    model : object, optional
        Model with `.embed(sequence: str, pool=...) -> Tensor` method.
        Required for computing new embeddings.
    id_col : str, optional
        Column with mutation IDs (format: GENE:CHROM:POS:REF:ALT). Default: "mut_id".
    chrom_col : str, optional
        Column with chromosome. If None, parsed from id_col.
    pos_col : str, optional
        Column with position. If None, parsed from id_col.
    ref_col : str, optional
        Column with reference allele. If None, parsed from id_col.
    alt_col : str, optional
        Column with alternate allele. If None, parsed from id_col.
    strand_col : str, optional
        Column with strand info ("+"/"-" or "P"/"N"). Overrides reverse_complement.
    reverse_complement : bool, optional
        Default strand orientation. True = negative strand. Default: False.
    context : int, optional
        Context window size for embedding. Default: 3000.
    genome : str, optional
        Genome assembly. Default: "hg38".
    diff : str or callable, optional
        Distance metric for geometry. Default: "cosine".
    prefix : str, optional
        Prefix for new column names. Default: "".
    inplace : bool, optional
        Modify DataFrame in place. Default: False.
    progress_callback : callable, optional
        Called with (current_index, total, mut_id) for progress tracking.
    pool : str, optional
        Pooling method for embeddings. "mean" for mean-pooled (default),
        "tokens" for full token embeddings (flattened).
    show_progress : bool, optional
        Show tqdm progress bar. Default: True.

    Returns
    -------
    pd.DataFrame
        DataFrame with added metric columns.

    Examples
    --------
    >>> # With model - computes and saves embeddings
    >>> df = add_single_variant_metrics(df, db, model=borzoi)

    >>> # With explicit columns and strand
    >>> df = add_single_variant_metrics(
    ...     df, db, model=borzoi,
    ...     chrom_col="chr", pos_col="position",
    ...     ref_col="ref", alt_col="alt",
    ...     strand_col="strand"  # column with "+"/"-"
    ... )

    >>> # Negative strand by default
    >>> df = add_single_variant_metrics(
    ...     df, db, model=borzoi,
    ...     reverse_complement=True
    ... )

    >>> # With full token embeddings
    >>> df = add_single_variant_metrics(df, db, model=borzoi, pool="tokens")
    """
    _require_pandas()

    if id_col not in df.columns:
        raise ValueError(f"Column {id_col!r} not found in DataFrame")

    if not inplace:
        df = df.copy()

    # Initialize metric columns with NaN
    metric_names = [
        "v_l2", "v_diff", "v_par", "v_perp",
        "x", "y", "radius", "angle", "angle_over_pi"
    ]
    for name in metric_names:
        df[f"{prefix}{name}"] = float("nan")

    n_processed = 0
    n_computed = 0
    n_loaded = 0
    n_skipped = 0
    total = len(df)

    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=total, desc="Single-variant metrics")

    for i, (idx, row) in enumerate(iterator):
        mut_id = row[id_col]

        if progress_callback:
            progress_callback(i, total, str(mut_id))

        # Determine genomic coordinates
        if chrom_col and pos_col and ref_col and alt_col:
            chrom = str(row[chrom_col])
            pos = int(row[pos_col])
            ref = str(row[ref_col])
            alt = str(row[alt_col])
        else:
            # Parse from mut_id
            try:
                chrom, pos, ref, alt, _ = parse_single_mut_id(mut_id)
            except ValueError as e:
                logger.warning("Cannot parse %s: %s", mut_id, e)
                n_skipped += 1
                continue

        # Determine strand
        if strand_col and strand_col in df.columns:
            rev = _parse_strand(row[strand_col])
        else:
            rev = reverse_complement

        # Check if embeddings exist
        wt_key = f"{mut_id}{KEY_WT}"
        mut_key = f"{mut_id}{KEY_MUT}"

        if db.has(wt_key) and db.has(mut_key):
            # Load existing embeddings
            h_wt = db.load(wt_key)
            h_mut = db.load(mut_key)
            n_loaded += 1
        elif model is not None:
            # Compute and save embeddings
            try:
                h_wt, h_mut, delta = _embed_single_variant_direct(
                    model=model,
                    chrom=chrom,
                    pos=pos,
                    ref=ref,
                    alt=alt,
                    reverse_complement=rev,
                    context=context,
                    genome=genome,
                    pool=pool,
                )
                # Save to database
                db.store(wt_key, h_wt)
                db.store(mut_key, h_mut)
                db.store(mut_id, delta)  # Also store delta
                n_computed += 1
            except Exception as e:
                logger.warning("Failed to embed %s: %s", mut_id, e)
                n_skipped += 1
                continue
        else:
            # No embeddings and no model
            logger.debug("No embeddings for %s and no model provided", mut_id)
            n_skipped += 1
            continue

        # Compute geometry
        geom = SingleVariantGeometry(h_wt, h_mut, diff=diff)
        coords = geom.canonical_coords()

        # Assign metrics to row
        for name in metric_names:
            df.at[idx, f"{prefix}{name}"] = getattr(coords, name)

        n_processed += 1

    logger.info(
        "Single-variant metrics: %d processed (%d computed, %d loaded), %d skipped",
        n_processed, n_computed, n_loaded, n_skipped
    )

    return df


def add_epistasis_metrics(
    df: "pd.DataFrame",
    db: VariantEmbeddingDB,
    model=None,
    id_col: str = "epistasis_id",
    chrom_col: Optional[str] = None,
    pos1_col: Optional[str] = None,
    ref1_col: Optional[str] = None,
    alt1_col: Optional[str] = None,
    pos2_col: Optional[str] = None,
    ref2_col: Optional[str] = None,
    alt2_col: Optional[str] = None,
    strand_col: Optional[str] = None,
    reverse_complement: bool = False,
    context: int = DEFAULT_CONTEXT,
    genome: str = DEFAULT_GENOME,
    diff: Union[str, DiffFn] = "cosine",
    prefix: str = "",
    include_complex: bool = True,
    include_triangle: bool = True,
    include_distances: bool = True,
    full_features: bool = False,
    inplace: bool = False,
    show_progress: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    pool: str = "mean",
    force: bool = False,
) -> "pd.DataFrame":
    """
    Add epistasis geometry metrics to a DataFrame.

    For each row:
    1. Check if embeddings exist in database
    2. If not and model is provided, compute and save embeddings
    3. Compute geometric metrics using EpistasisGeometry

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing epistasis data.
    db : VariantEmbeddingDB
        Database for storing/loading embeddings.
    model : object, optional
        Model with `.embed(sequence: str, pool=...) -> Tensor` method.
    id_col : str, optional
        Column with epistasis IDs (format: mut1|mut2). Default: "epistasis_id".
    chrom_col : str, optional
        Column with chromosome. If None, parsed from id_col.
    pos1_col, pos2_col : str, optional
        Columns with positions. If None, parsed from id_col.
    ref1_col, ref2_col : str, optional
        Columns with reference alleles. If None, parsed from id_col.
    alt1_col, alt2_col : str, optional
        Columns with alternate alleles. If None, parsed from id_col.
    strand_col : str, optional
        Column with strand info ("+"/"-"). Overrides reverse_complement.
    reverse_complement : bool, optional
        Default strand orientation. True = negative strand. Default: False.
    context : int, optional
        Context window size. Default: 3000.
    genome : str, optional
        Genome assembly. Default: "hg38".
    diff : str or callable, optional
        Distance metric. Default: "cosine".
    prefix : str, optional
        Prefix for new column names. Default: "".
    include_complex : bool, optional
        Include complex plane coordinates. Default: True.
    include_triangle : bool, optional
        Include triangle-based coordinates. Default: True.
    include_distances : bool, optional
        Include all distance/length metrics. Default: True.
    full_features : bool, optional
        If True, compute metrics for all three expectation models (additive,
        geometric, pythagorean) with prefixes "add_", "geo_", "pyth_".
        Creates a comprehensive feature space for ML applications.
        Default: False.
    inplace : bool, optional
        Modify DataFrame in place. Default: False.
    show_progress : bool, optional
        Show tqdm progress bar. Default: False.
    progress_callback : callable, optional
        Called with (current_index, total, epistasis_id) for progress.
    pool : str, optional
        Pooling method for embeddings. "mean" for mean-pooled (default),
        "tokens" for full token embeddings (flattened).
    force : bool, optional
        If True, recompute embeddings even if they exist in the database.
        Default: False.

    Returns
    -------
    pd.DataFrame
        DataFrame with added metric columns.

    Examples
    --------
    >>> # With model - computes and saves all embeddings
    >>> df = add_epistasis_metrics(df, db, model=borzoi)

    >>> # Force recomputation of all embeddings
    >>> df = add_epistasis_metrics(df, db, model=borzoi, force=True)

    >>> # Full feature space for ML with all expectation models
    >>> df = add_epistasis_metrics(df, db, model=borzoi, full_features=True)

    >>> # Negative strand genes
    >>> df = add_epistasis_metrics(
    ...     df, db, model=borzoi,
    ...     reverse_complement=True
    ... )

    >>> # Per-row strand from column
    >>> df = add_epistasis_metrics(
    ...     df, db, model=borzoi,
    ...     strand_col="gene_strand"
    ... )

    >>> # With full token embeddings
    >>> df = add_epistasis_metrics(df, db, model=borzoi, pool="tokens")
    """
    _require_pandas()

    if id_col not in df.columns:
        raise ValueError(f"Column {id_col!r} not found in DataFrame")

    if not inplace:
        df = df.copy()

    # Expectation model definitions for full_features mode
    def _geometric_v12_exp(v1: torch.Tensor, v2: torch.Tensor, eps: float) -> torch.Tensor:
        """Geometric mean of effect magnitudes with average direction."""
        mag = torch.sqrt(v1.norm() * v2.norm() + eps)
        direction = v1 + v2
        direction = direction / (direction.norm() + eps)
        return mag * direction

    def _pythagorean_v12_exp(v1: torch.Tensor, v2: torch.Tensor, eps: float) -> torch.Tensor:
        """Pythagorean combination: sqrt(||v1||^2 + ||v2||^2) with average direction."""
        mag = torch.sqrt(v1.norm()**2 + v2.norm()**2)
        direction = v1 + v2
        direction = direction / (direction.norm() + eps)
        return mag * direction

    # Define which expectation models to use
    def _additive_v12_exp(v1: torch.Tensor, v2: torch.Tensor, eps: float) -> torch.Tensor:
        """Additive expectation: v12_exp = v1 + v2."""
        _ = eps  # unused but kept for consistent signature
        return v1 + v2

    if full_features:
        expectation_models = [
            ("add", _additive_v12_exp),     # additive
            ("geo", _geometric_v12_exp),    # geometric
            ("pyth", _pythagorean_v12_exp), # pythagorean
        ]
    else:
        expectation_models = [("", _additive_v12_exp)]  # additive only, no prefix

    # Core metrics that don't depend on expectation model
    base_metrics = [
        "dist_WT_M1", "dist_WT_M2", "dist_M1_M2",
        "len_WT_M1", "len_WT_M2", "len_M1_M2",
        "cos_v1_v2",
    ]

    # Expectation-dependent metrics
    exp_metrics = [
        "dist_WT_M12_obs", "dist_WT_M12_exp", "dist_obs_exp",
        "len_WT_M12", "len_WT_M12_exp",
        "epi_R_raw", "epi_R_singles", "epi_R_expected", "epi_score",
        "cos_v12_v1", "cos_v12_v2", "cos_v12_vexp",
        "magnitude_ratio", "radial_deviation",
        "epi_total", "epi_total_bounded",
        "log_rho_tri", "abs_log_rho_tri",
        # L1-based metrics
        "len_WT_M12_L1", "len_WT_M12_exp_L1",
        "epi_R_raw_L1", "epi_R_expected_L1",
        "magnitude_ratio_L1", "log_rho_L1", "abs_log_rho_L1",
    ]

    # Complex coords metrics (expectation-dependent)
    complex_metrics = [
        "x_exp", "y_exp", "x_obs", "y_obs",
        "rho", "theta", "theta_over_pi"
    ]

    # Triangle coords metrics (expectation-dependent)
    triangle_metrics = [
        "x_exp_tri", "y_exp_tri", "x_obs_tri", "y_obs_tri",
        "rho_tri", "theta_tri", "theta_tri_over_pi"
    ]

    # Build list of metric columns to add
    metric_cols: list[str] = []

    # Base metrics (only once, not per-expectation)
    if include_distances:
        metric_cols.extend(base_metrics)

    # Per-expectation metrics
    for exp_prefix, _ in expectation_models:
        full_prefix = f"{exp_prefix}_" if exp_prefix else ""

        if include_distances:
            metric_cols.extend([f"{full_prefix}{m}" for m in exp_metrics])

        if include_complex:
            metric_cols.extend([f"{full_prefix}{m}" for m in complex_metrics])

        if include_triangle:
            metric_cols.extend([f"{full_prefix}{m}" for m in triangle_metrics])

    # Initialize columns with user prefix
    for name in metric_cols:
        df[f"{prefix}{name}"] = float("nan")

    n_processed = 0
    n_computed = 0
    n_loaded = 0
    n_skipped = 0
    total = len(df)

    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=total, desc="Epistasis metrics")

    for i, (idx, row) in enumerate(iterator):
        epi_id = row[id_col]

        if progress_callback:
            progress_callback(i, total, str(epi_id))

        # Determine genomic coordinates
        use_columns = (
            chrom_col and pos1_col and ref1_col and alt1_col
            and pos2_col and ref2_col and alt2_col
        )

        if use_columns:
            chrom = str(row[chrom_col])
            pos1 = int(row[pos1_col])
            ref1 = str(row[ref1_col])
            alt1 = str(row[alt1_col])
            pos2 = int(row[pos2_col])
            ref2 = str(row[ref2_col])
            alt2 = str(row[alt2_col])
            rev_from_id = None
        else:
            # Parse from epistasis_id
            try:
                (chrom, pos1, ref1, alt1, rev1), (_, pos2, ref2, alt2, rev2) = parse_epistasis_id(epi_id)
                rev_from_id = rev1  # rev1 == rev2 guaranteed by parse_epistasis_id
            except ValueError as e:
                logger.warning("Cannot parse %s: %s", epi_id, e)
                n_skipped += 1
                continue

        # Determine strand: strand_col > parsed from id > reverse_complement default
        if strand_col and strand_col in df.columns:
            rev = _parse_strand(row[strand_col])
        elif rev_from_id is not None:
            rev = rev_from_id
        else:
            rev = reverse_complement

        # Check if embeddings exist
        wt_key = f"{epi_id}{KEY_WT}"
        m1_key = f"{epi_id}{KEY_M1}"
        m2_key = f"{epi_id}{KEY_M2}"
        m12_key = f"{epi_id}{KEY_M12}"

        if not force and db.has(wt_key) and db.has(m1_key) and db.has(m2_key) and db.has(m12_key):
            # Load existing embeddings
            h_wt = db.load(wt_key)
            h_m1 = db.load(m1_key)
            h_m2 = db.load(m2_key)
            h_m12 = db.load(m12_key)
            n_loaded += 1
        elif model is not None:
            # Compute and save embeddings
            try:
                h_wt, h_m1, h_m2, h_m12 = _embed_epistasis_direct(
                    model=model,
                    chrom=chrom,
                    pos1=pos1,
                    ref1=ref1,
                    alt1=alt1,
                    pos2=pos2,
                    ref2=ref2,
                    alt2=alt2,
                    reverse_complement=rev,
                    context=context,
                    genome=genome,
                    pool=pool,
                )
                # Save all embeddings to database
                db.store(wt_key, h_wt)
                db.store(m1_key, h_m1)
                db.store(m2_key, h_m2)
                db.store(m12_key, h_m12)
                # Also store deltas
                db.store(f"{epi_id}{KEY_DELTA1}", h_m1 - h_wt)
                db.store(f"{epi_id}{KEY_DELTA2}", h_m2 - h_wt)
                db.store(f"{epi_id}{KEY_DELTA12}", h_m12 - h_wt)
                n_computed += 1
            except Exception as e:
                logger.warning("Failed to embed %s: %s", epi_id, e)
                n_skipped += 1
                continue
        else:
            # No embeddings and no model
            logger.debug("No embeddings for %s and no model provided", epi_id)
            n_skipped += 1
            continue

        # Compute effect vectors (needed for all expectation models)
        # Convert to tensors for expectation model computations
        v1 = torch.as_tensor(h_m1 - h_wt, dtype=torch.float32)
        v2 = torch.as_tensor(h_m2 - h_wt, dtype=torch.float32)

        # Assign base metrics (only once, not per-expectation)
        if include_distances:
            # Use default geometry for base metrics
            geom_base = EpistasisGeometry(h_wt, h_m1, h_m2, h_m12, diff=diff)
            metrics_base = geom_base.metrics()
            for name in base_metrics:
                df.at[idx, f"{prefix}{name}"] = getattr(metrics_base, name)

        # Compute metrics for each expectation model
        for exp_prefix, exp_fn in expectation_models:
            full_prefix = f"{prefix}{exp_prefix}_" if exp_prefix else prefix

            # Compute expected v12 using this expectation model
            v12_exp = exp_fn(v1, v2, DEFAULT_EPS)

            # Create geometry with custom v12_exp
            # We need to create the geometry manually to use the custom expectation
            geom = EpistasisGeometry(h_wt, h_m1, h_m2, h_m12, diff=diff)
            # Override v12_exp with our computed value
            geom.v12_exp = v12_exp
            geom._cached_metrics = None  # Clear cache to recompute with new v12_exp
            geom._cached_complex = None
            geom._cached_triangle = None

            # Assign expectation-dependent distance metrics
            if include_distances:
                metrics = geom.metrics()
                for name in exp_metrics:
                    df.at[idx, f"{full_prefix}{name}"] = getattr(metrics, name)

            # Assign complex coords
            if include_complex:
                coords = geom.complex_coords()
                for name in complex_metrics:
                    df.at[idx, f"{full_prefix}{name}"] = getattr(coords, name)

            # Assign triangle coords
            if include_triangle:
                tri = geom.triangle_coords()
                # Map column names to dataclass attributes
                tri_attr_map = {
                    "x_exp_tri": "x_exp",
                    "y_exp_tri": "y_exp",
                    "x_obs_tri": "x_obs",
                    "y_obs_tri": "y_obs",
                    "rho_tri": "rho",
                    "theta_tri": "theta",
                    "theta_tri_over_pi": "theta_over_pi",
                }
                for col_name in triangle_metrics:
                    attr_name = tri_attr_map[col_name]
                    df.at[idx, f"{full_prefix}{col_name}"] = getattr(tri, attr_name)

        n_processed += 1

    logger.info(
        "Epistasis metrics: %d processed (%d computed, %d loaded), %d skipped",
        n_processed, n_computed, n_loaded, n_skipped
    )

    return df


# ---------------------------------------------------------------------------
# Dependency Map Analysis
# ---------------------------------------------------------------------------

NUCLEOTIDES = ["A", "C", "G", "T"]


@dataclass
class DependencyMapResult:
    """
    Result of a dependency map computation.

    Attributes
    ----------
    positions : np.ndarray
        Array of genomic positions analyzed.
    matrix : np.ndarray
        NxN matrix where entry (i, j) is the dependency score between
        positions[i] and positions[j]. Symmetric, with NaN on diagonal.
    aggregation : str
        How mutation combinations were aggregated ("max" or "mean").
    metric : str
        Which epistasis metric was used ("epi_R_raw", "epi_R_expected", "rho").
    sequence : str
        The reference sequence used.
    details : dict
        Full results for each position pair, keyed by (pos_i, pos_j).
    """

    positions: np.ndarray
    matrix: np.ndarray
    aggregation: str
    metric: str
    sequence: str
    details: dict = field(default_factory=dict)

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert matrix to a labeled DataFrame."""
        _require_pandas()
        import pandas as pd

        return pd.DataFrame(
            self.matrix,
            index=self.positions,
            columns=self.positions,
        )

    def plot(
        self,
        figsize: tuple[float, float] = (10, 8),
        cmap: str = "viridis",
        title: str = "",
        figure_name: str = "",
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot dependency map as a heatmap.

        Parameters
        ----------
        figsize : tuple
            Figure size.
        cmap : str
            Colormap name.
        title : str
            Plot title.
        figure_name : str
            If provided, save figure to {figure_name}.png.
        show : bool
            Call plt.show().

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Mask diagonal
        mask = np.eye(len(self.positions), dtype=bool)
        masked_matrix = np.ma.array(self.matrix, mask=mask)

        im = ax.imshow(masked_matrix, cmap=cmap, aspect="equal")
        plt.colorbar(im, ax=ax, label=f"Dependency ({self.metric})")

        # Labels
        n = len(self.positions)
        if n <= 50:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(self.positions, rotation=90, fontsize=8)
            ax.set_yticklabels(self.positions, fontsize=8)
        else:
            # Too many positions, use sparse labels
            step = max(1, n // 10)
            ax.set_xticks(range(0, n, step))
            ax.set_yticks(range(0, n, step))
            ax.set_xticklabels(self.positions[::step], rotation=90, fontsize=8)
            ax.set_yticklabels(self.positions[::step], fontsize=8)

        ax.set_xlabel("Position j")
        ax.set_ylabel("Position i")

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Dependency Map ({self.metric}, {self.aggregation})")

        plt.tight_layout()

        if figure_name:
            fig.savefig(f"{figure_name}.png", dpi=300, bbox_inches="tight")
            logger.info("Saved dependency map to %s.png", figure_name)

        if show:
            plt.show()

        return fig

    def top_pairs(self, n: int = 10) -> list[tuple[int, int, float]]:
        """
        Get the top N position pairs by dependency score.

        Returns list of (pos_i, pos_j, score) tuples, sorted descending.
        """
        pairs = []
        m = len(self.positions)
        for i in range(m):
            for j in range(i + 1, m):
                score = self.matrix[i, j]
                if not np.isnan(score):
                    pairs.append((self.positions[i], self.positions[j], score))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:n]


def compute_pairwise_epistasis(
    model,
    sequence: str,
    pos1: int,
    pos2: int,
    diff: Union[str, DiffFn] = "cosine",
    eps: float = DEFAULT_EPS,
) -> dict:
    """
    Compute epistasis for all mutation combinations at two positions.

    For each position, tries all 3 alternative nucleotides (excluding the
    reference). Returns 9 epistasis computations (3 x 3 combinations).

    Parameters
    ----------
    model : object
        Model with `.embed(sequence: str, pool='mean') -> Tensor` method.
    sequence : str
        Reference sequence.
    pos1 : int
        First position (0-indexed into sequence).
    pos2 : int
        Second position (0-indexed into sequence).
    diff : str or callable
        Distance metric for epistasis computation.
    eps : float
        Numerical stability constant.

    Returns
    -------
    dict
        Dictionary with:
        - "ref1", "ref2": reference nucleotides
        - "combinations": list of dicts, each with:
            - "alt1", "alt2": alternative nucleotides
            - "metrics": EpistasisMetrics
            - "complex": EpistasisComplexCoords
    """
    if pos1 >= pos2:
        raise ValueError(f"pos1 ({pos1}) must be < pos2 ({pos2})")
    if pos1 < 0 or pos2 >= len(sequence):
        raise ValueError(f"Positions out of bounds for sequence length {len(sequence)}")

    ref1 = sequence[pos1]
    ref2 = sequence[pos2]

    # Get WT embedding
    h_wt = model.embed(sequence, pool="mean")

    results = {
        "ref1": ref1,
        "ref2": ref2,
        "pos1": pos1,
        "pos2": pos2,
        "combinations": [],
    }

    # All alternative alleles
    alts1 = [n for n in NUCLEOTIDES if n != ref1]
    alts2 = [n for n in NUCLEOTIDES if n != ref2]

    for alt1 in alts1:
        # Create M1 sequence
        seq_m1 = sequence[:pos1] + alt1 + sequence[pos1 + 1:]
        h_m1 = model.embed(seq_m1, pool="mean")

        for alt2 in alts2:
            # Create M2 sequence
            seq_m2 = sequence[:pos2] + alt2 + sequence[pos2 + 1:]
            h_m2 = model.embed(seq_m2, pool="mean")

            # Create M12 sequence (both mutations)
            seq_m12 = sequence[:pos1] + alt1 + sequence[pos1 + 1:pos2] + alt2 + sequence[pos2 + 1:]
            h_m12 = model.embed(seq_m12, pool="mean")

            # Compute epistasis geometry
            geom = EpistasisGeometry(h_wt, h_m1, h_m2, h_m12, eps=eps, diff=diff)

            results["combinations"].append({
                "alt1": alt1,
                "alt2": alt2,
                "metrics": geom.metrics(),
                "complex": geom.complex_coords(),
            })

    return results


def compute_dependency_map(
    model,
    sequence: str,
    positions: Optional[list[int]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    metric: str = "epi_R_expected",
    aggregation: str = "max",
    diff: Union[str, DiffFn] = "cosine",
    eps: float = DEFAULT_EPS,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cache_embeddings: bool = True,
) -> DependencyMapResult:
    """
    Compute a dependency map showing epistasis between all position pairs.

    For each pair of positions (i, j), computes epistasis for all 9 mutation
    combinations and aggregates to a single dependency score.

    Parameters
    ----------
    model : object
        Model with `.embed(sequence: str, pool='mean') -> Tensor` method.
    sequence : str
        Reference sequence.
    positions : list[int], optional
        Specific positions to analyze (0-indexed). If None, uses start/end.
    start : int, optional
        Start position (inclusive). Default: 0.
    end : int, optional
        End position (exclusive). Default: len(sequence).
    step : int, optional
        Step size between positions. Default: 1.
    metric : str, optional
        Epistasis metric to use: "epi_R_raw", "epi_R_expected", "epi_R_singles",
        or "rho" (complex magnitude). Default: "epi_R_expected".
    aggregation : str, optional
        How to aggregate across mutation combinations: "max" or "mean".
        Default: "max".
    diff : str or callable
        Distance metric for epistasis computation. Default: "cosine".
    eps : float
        Numerical stability constant.
    progress_callback : callable, optional
        Called with (current_pair, total_pairs, description) for progress.
    cache_embeddings : bool, optional
        Cache single-mutant embeddings to avoid recomputation. Default: True.

    Returns
    -------
    DependencyMapResult
        Object containing the dependency matrix and metadata.

    Examples
    --------
    >>> # Analyze region around two positions
    >>> result = compute_dependency_map(
    ...     model, sequence,
    ...     positions=[2900, 2950, 3000, 3050, 3100],
    ...     metric="epi_R_expected",
    ... )
    >>> result.plot()
    >>> print(result.top_pairs(5))

    >>> # Scan a range with step size
    >>> result = compute_dependency_map(
    ...     model, sequence,
    ...     start=2800, end=3200, step=10,
    ... )
    """
    # Determine positions to analyze
    if positions is not None:
        pos_array = np.array(sorted(positions))
    else:
        if start is None:
            start = 0
        if end is None:
            end = len(sequence)
        pos_array = np.arange(start, end, step)

    n = len(pos_array)
    if n < 2:
        raise ValueError("Need at least 2 positions for dependency map")

    logger.info(
        "Computing dependency map: %d positions, %d pairs",
        n, n * (n - 1) // 2
    )

    # Validate positions
    for p in pos_array:
        if p < 0 or p >= len(sequence):
            raise ValueError(f"Position {p} out of bounds for sequence length {len(sequence)}")

    # Initialize result matrix
    matrix = np.full((n, n), np.nan)
    details = {}

    # Cache for single-mutant embeddings: (pos, alt) -> embedding
    embedding_cache: dict[tuple[int, str], torch.Tensor] = {}

    # Get WT embedding once
    h_wt = model.embed(sequence, pool="mean")

    def get_single_mutant_embedding(pos: int, alt: str) -> torch.Tensor:
        """Get or compute single-mutant embedding."""
        key = (pos, alt)
        if cache_embeddings and key in embedding_cache:
            return embedding_cache[key]

        seq_mut = sequence[:pos] + alt + sequence[pos + 1:]
        h_mut = model.embed(seq_mut, pool="mean")

        if cache_embeddings:
            embedding_cache[key] = h_mut

        return h_mut

    # Compute all pairs
    total_pairs = n * (n - 1) // 2
    pair_idx = 0

    pbar = tqdm(total=total_pairs, desc="Computing dependency map")
    for i in range(n):
        pos_i = pos_array[i]
        ref_i = sequence[pos_i]
        alts_i = [nuc for nuc in NUCLEOTIDES if nuc != ref_i]

        for j in range(i + 1, n):
            pos_j = pos_array[j]
            ref_j = sequence[pos_j]
            alts_j = [nuc for nuc in NUCLEOTIDES if nuc != ref_j]

            pbar.set_postfix(pair=f"({pos_i}, {pos_j})")
            if progress_callback:
                progress_callback(
                    pair_idx, total_pairs,
                    f"Pair ({pos_i}, {pos_j})"
                )

            # Compute all 9 combinations
            scores = []
            pair_details = []

            for alt_i in alts_i:
                h_m1 = get_single_mutant_embedding(pos_i, alt_i)

                for alt_j in alts_j:
                    h_m2 = get_single_mutant_embedding(pos_j, alt_j)

                    # Double mutant
                    seq_m12 = (
                        sequence[:pos_i] + alt_i +
                        sequence[pos_i + 1:pos_j] + alt_j +
                        sequence[pos_j + 1:]
                    )
                    h_m12 = model.embed(seq_m12, pool="mean")

                    # Epistasis geometry
                    geom = EpistasisGeometry(h_wt, h_m1, h_m2, h_m12, eps=eps, diff=diff)
                    metrics = geom.metrics()
                    complex_coords = geom.complex_coords()

                    # Extract score based on metric
                    if metric == "rho":
                        score = complex_coords.rho
                    elif hasattr(metrics, metric):
                        score = getattr(metrics, metric)
                    else:
                        raise ValueError(f"Unknown metric: {metric}")

                    scores.append(score)
                    pair_details.append({
                        "alt_i": alt_i,
                        "alt_j": alt_j,
                        "score": score,
                        "metrics": metrics,
                        "complex": complex_coords,
                    })

            # Aggregate scores
            if aggregation == "max":
                agg_score = max(scores)
            elif aggregation == "mean":
                agg_score = sum(scores) / len(scores)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            # Store in matrix (symmetric)
            matrix[i, j] = agg_score
            matrix[j, i] = agg_score

            # Store details
            details[(pos_i, pos_j)] = {
                "ref_i": ref_i,
                "ref_j": ref_j,
                "aggregated_score": agg_score,
                "all_combinations": pair_details,
            }

            pair_idx += 1
            pbar.update(1)

    pbar.close()
    logger.info("Dependency map complete: %d pairs computed", total_pairs)

    return DependencyMapResult(
        positions=pos_array,
        matrix=matrix,
        aggregation=aggregation,
        metric=metric,
        sequence=sequence,
        details=details,
    )


# =============================================================================
# EPISTASIS EXPECTATION MODEL
# =============================================================================


@dataclass
class EpistasisDataset:
    """
    Dataset for training epistasis expectation models.

    Stores delta vectors (Δ₁, Δ₂, Δ₁₂) and optionally wild-type embeddings
    for a set of epistasis pairs.

    Attributes
    ----------
    delta1 : torch.Tensor
        Single mutant 1 deltas, shape (N, D).
    delta2 : torch.Tensor
        Single mutant 2 deltas, shape (N, D).
    delta12 : torch.Tensor
        Double mutant deltas (targets), shape (N, D).
    wt : torch.Tensor or None
        Wild-type embeddings, shape (N, D). None if not included.
    epistasis_ids : list[str]
        Original epistasis IDs for each sample.
    embedding_dim : int
        Dimensionality of embeddings.
    """

    delta1: torch.Tensor
    delta2: torch.Tensor
    delta12: torch.Tensor
    wt: Optional[torch.Tensor]
    epistasis_ids: list
    embedding_dim: int

    def __len__(self) -> int:
        return len(self.epistasis_ids)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample as a dictionary."""
        item = {
            "delta1": self.delta1[idx],
            "delta2": self.delta2[idx],
            "delta12": self.delta12[idx],
            "epistasis_id": self.epistasis_ids[idx],
        }
        if self.wt is not None:
            item["wt"] = self.wt[idx]
        return item

    @classmethod
    def from_dataframe(
        cls,
        df: "pd.DataFrame",
        db: VariantEmbeddingDB,
        id_col: str = "epistasis_id",
        include_wt: bool = True,
        show_progress: bool = True,
    ) -> "EpistasisDataset":
        """
        Create dataset from DataFrame and embedding database.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with epistasis IDs.
        db : VariantEmbeddingDB
            Database containing embeddings. Must have Δ₁, Δ₂, Δ₁₂ stored
            (use store_deltas=True when computing embeddings).
        id_col : str, optional
            Column containing epistasis IDs. Default: "epistasis_id".
        include_wt : bool, optional
            Include wild-type embeddings as input features. Default: True.
        show_progress : bool, optional
            Show progress bar. Default: True.

        Returns
        -------
        EpistasisDataset
            Dataset ready for training.

        Examples
        --------
        >>> dataset = EpistasisDataset.from_dataframe(df, db, include_wt=True)
        >>> print(f"Loaded {len(dataset)} samples, dim={dataset.embedding_dim}")
        """
        _require_pandas()

        if id_col not in df.columns:
            raise ValueError(f"Column {id_col!r} not found in DataFrame")

        delta1_list = []
        delta2_list = []
        delta12_list = []
        wt_list = [] if include_wt else None
        valid_ids = []
        skipped = 0

        iterator = df[id_col]
        if show_progress:
            iterator = tqdm(iterator, desc="Loading embeddings")

        for epi_id in iterator:
            # Check for required keys
            d1_key = f"{epi_id}{KEY_DELTA1}"
            d2_key = f"{epi_id}{KEY_DELTA2}"
            d12_key = f"{epi_id}{KEY_DELTA12}"
            wt_key = f"{epi_id}{KEY_WT}"

            required_keys = [d1_key, d2_key, d12_key]
            if include_wt:
                required_keys.append(wt_key)

            if not all(db.has(k) for k in required_keys):
                skipped += 1
                continue

            # Load embeddings
            d1 = db.load(d1_key, as_torch=True)
            d2 = db.load(d2_key, as_torch=True)
            d12 = db.load(d12_key, as_torch=True)

            delta1_list.append(d1)
            delta2_list.append(d2)
            delta12_list.append(d12)

            if include_wt:
                wt = db.load(wt_key, as_torch=True)
                wt_list.append(wt)

            valid_ids.append(epi_id)

        if not valid_ids:
            raise ValueError("No valid epistasis samples found in database")

        if skipped > 0:
            logger.warning("Skipped %d epistasis IDs (missing embeddings)", skipped)

        # Stack into tensors
        delta1 = torch.stack(delta1_list, dim=0)
        delta2 = torch.stack(delta2_list, dim=0)
        delta12 = torch.stack(delta12_list, dim=0)
        wt = torch.stack(wt_list, dim=0) if include_wt else None

        embedding_dim = delta1.shape[1]

        logger.info(
            "Created EpistasisDataset: %d samples, dim=%d, include_wt=%s",
            len(valid_ids),
            embedding_dim,
            include_wt,
        )

        return cls(
            delta1=delta1,
            delta2=delta2,
            delta12=delta12,
            wt=wt,
            epistasis_ids=valid_ids,
            embedding_dim=embedding_dim,
        )

    def to_torch_dataset(self) -> "torch.utils.data.TensorDataset":
        """Convert to PyTorch TensorDataset for DataLoader."""
        if self.wt is not None:
            return torch.utils.data.TensorDataset(
                self.delta1, self.delta2, self.wt, self.delta12
            )
        return torch.utils.data.TensorDataset(
            self.delta1, self.delta2, self.delta12
        )

    def train_val_split(
        self,
        val_fraction: float = 0.2,
        seed: int = DEFAULT_RANDOM_STATE,
    ) -> tuple["EpistasisDataset", "EpistasisDataset"]:
        """
        Split into training and validation datasets.

        Parameters
        ----------
        val_fraction : float, optional
            Fraction for validation. Default: 0.2.
        seed : int, optional
            Random seed. Default: 42.

        Returns
        -------
        tuple[EpistasisDataset, EpistasisDataset]
            (train_dataset, val_dataset)
        """
        n = len(self)
        n_val = int(n * val_fraction)
        n_train = n - n_val

        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n, generator=rng)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        def subset(idx: torch.Tensor) -> "EpistasisDataset":
            return EpistasisDataset(
                delta1=self.delta1[idx],
                delta2=self.delta2[idx],
                delta12=self.delta12[idx],
                wt=self.wt[idx] if self.wt is not None else None,
                epistasis_ids=[self.epistasis_ids[i] for i in idx.tolist()],
                embedding_dim=self.embedding_dim,
            )

        return subset(train_idx), subset(val_idx)


class EpistasisExpectationModel:
    """
    Trainable model for predicting expected double-mutant delta vectors.

    Learns to predict Δ₁₂ from (Δ₁, Δ₂) or (Δ₁, Δ₂, E_WT), enabling
    data-driven epistasis expectations instead of assuming additivity.

    Parameters
    ----------
    architecture : str, optional
        Model architecture: "linear", "mlp", or "bilinear". Default: "linear".
    include_wt : bool, optional
        Include wild-type embedding as input. Default: False.
    hidden_dims : list[int], optional
        Hidden layer dimensions for MLP. Default: [256].
    dropout : float, optional
        Dropout rate for MLP. Default: 0.1.
    activation : str, optional
        Activation function: "relu", "gelu", "tanh". Default: "relu".
    device : str, optional
        Device for training. Default: "cpu".

    Examples
    --------
    >>> # Train a model
    >>> model = EpistasisExpectationModel(architecture="mlp", include_wt=True)
    >>> model.fit(dataset, epochs=100, lr=1e-3)
    >>>
    >>> # Predict expected delta
    >>> delta12_exp = model.predict(delta1, delta2, wt=h_wt)
    >>>
    >>> # Use in EpistasisGeometry
    >>> geom = EpistasisGeometry(h_wt, h_m1, h_m2, h_m12, expectation_model=model)
    """

    def __init__(
        self,
        architecture: str = "linear",
        include_wt: bool = False,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        device: str = "cpu",
    ):
        self.architecture = architecture
        self.include_wt = include_wt
        self.hidden_dims = hidden_dims or [256]
        self.dropout = dropout
        self.activation = activation
        self.device = torch.device(device)
        self._model: Optional[torch.nn.Module] = None
        self._embedding_dim: Optional[int] = None
        self._trained = False

    def _build_model(self, embedding_dim: int) -> torch.nn.Module:
        """Build the neural network architecture."""
        self._embedding_dim = embedding_dim

        # Input size: Δ₁ + Δ₂ (+ E_WT if included)
        input_dim = 2 * embedding_dim
        if self.include_wt:
            input_dim += embedding_dim

        output_dim = embedding_dim

        if self.architecture == "linear":
            # Simple linear combination: W₁·Δ₁ + W₂·Δ₂ + b (or with W_wt·E_WT)
            return torch.nn.Linear(input_dim, output_dim)

        elif self.architecture == "mlp":
            # Multi-layer perceptron
            layers = []
            prev_dim = input_dim

            act_fn = {
                "relu": torch.nn.ReLU,
                "gelu": torch.nn.GELU,
                "tanh": torch.nn.Tanh,
            }.get(self.activation, torch.nn.ReLU)

            for hidden_dim in self.hidden_dims:
                layers.extend([
                    torch.nn.Linear(prev_dim, hidden_dim),
                    act_fn(),
                    torch.nn.Dropout(self.dropout),
                ])
                prev_dim = hidden_dim

            layers.append(torch.nn.Linear(prev_dim, output_dim))
            return torch.nn.Sequential(*layers)

        elif self.architecture == "bilinear":
            # Bilinear interaction between Δ₁ and Δ₂
            class BilinearModel(torch.nn.Module):
                def __init__(self, dim: int, include_wt: bool):
                    super().__init__()
                    self.include_wt = include_wt
                    self.bilinear = torch.nn.Bilinear(dim, dim, dim)
                    self.linear = torch.nn.Linear(dim * 2, dim)
                    if include_wt:
                        self.wt_linear = torch.nn.Linear(dim, dim)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    dim = x.shape[-1] // (3 if self.include_wt else 2)
                    d1 = x[..., :dim]
                    d2 = x[..., dim:2*dim]

                    bilin = self.bilinear(d1, d2)
                    linear = self.linear(torch.cat([d1, d2], dim=-1))
                    out = bilin + linear

                    if self.include_wt:
                        wt = x[..., 2*dim:]
                        out = out + self.wt_linear(wt)

                    return out

            return BilinearModel(embedding_dim, self.include_wt)

        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

    def _prepare_input(
        self,
        delta1: torch.Tensor,
        delta2: torch.Tensor,
        wt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Concatenate inputs for the model."""
        inputs = [delta1, delta2]
        if self.include_wt:
            if wt is None:
                raise ValueError("Model requires wt but none provided")
            inputs.append(wt)
        return torch.cat(inputs, dim=-1)

    def fit(
        self,
        dataset: EpistasisDataset,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        val_fraction: float = 0.2,
        directional_weight: float = 0.5,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> dict:
        """
        Train the model on an epistasis dataset.

        Parameters
        ----------
        dataset : EpistasisDataset
            Training data.
        epochs : int, optional
            Number of training epochs. Default: 100.
        lr : float, optional
            Learning rate. Default: 1e-3.
        weight_decay : float, optional
            L2 regularization. Default: 1e-5.
        batch_size : int, optional
            Batch size. Default: 32.
        val_fraction : float, optional
            Validation split fraction. Default: 0.2.
        directional_weight : float, optional
            Weight for cosine similarity loss (0=MSE only, 1=cosine only).
            Default: 0.5 (equal weight).
        early_stopping_patience : int, optional
            Epochs without improvement before stopping. Default: 10.
        verbose : bool, optional
            Print training progress. Default: True.

        Returns
        -------
        dict
            Training history with 'train_loss', 'val_loss', 'best_epoch'.
        """
        if self.include_wt and dataset.wt is None:
            raise ValueError("Model requires wt but dataset has include_wt=False")

        # Build model
        self._model = self._build_model(dataset.embedding_dim).to(self.device)

        # Split data
        train_data, val_data = dataset.train_val_split(val_fraction)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_data.to_torch_dataset(),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_data.to_torch_dataset(),
            batch_size=batch_size,
            shuffle=False,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Training loop
        history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self._model.train()
            train_losses = []

            for batch in train_loader:
                if self.include_wt:
                    d1, d2, wt, d12_target = [b.to(self.device) for b in batch]
                else:
                    d1, d2, d12_target = [b.to(self.device) for b in batch]
                    wt = None

                optimizer.zero_grad()

                x = self._prepare_input(d1, d2, wt)
                d12_pred = self._model(x)

                loss = self._compute_loss(d12_pred, d12_target, directional_weight)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            self._model.eval()
            val_losses = []

            with torch.no_grad():
                for batch in val_loader:
                    if self.include_wt:
                        d1, d2, wt, d12_target = [b.to(self.device) for b in batch]
                    else:
                        d1, d2, d12_target = [b.to(self.device) for b in batch]
                        wt = None

                    x = self._prepare_input(d1, d2, wt)
                    d12_pred = self._model(x)

                    loss = self._compute_loss(d12_pred, d12_target, directional_weight)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                history["best_epoch"] = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"train: {train_loss:.6f}, val: {val_loss:.6f}"
                )

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._model.to(self.device)
        self._trained = True

        if verbose:
            print(
                f"Training complete. Best val loss: {best_val_loss:.6f} "
                f"at epoch {history['best_epoch']+1}"
            )

        return history

    def _compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        directional_weight: float,
    ) -> torch.Tensor:
        """Compute combined MSE and directional (cosine) loss."""
        # MSE loss (magnitude)
        mse_loss = torch.nn.functional.mse_loss(pred, target)

        if directional_weight == 0:
            return mse_loss

        # Cosine similarity loss (direction)
        cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=-1)
        cosine_loss = (1 - cos_sim).mean()  # 0 when perfect alignment

        # Combined loss
        return (1 - directional_weight) * mse_loss + directional_weight * cosine_loss

    def predict(
        self,
        delta1: TensorLike,
        delta2: TensorLike,
        wt: Optional[TensorLike] = None,
    ) -> torch.Tensor:
        """
        Predict expected Δ₁₂ from single-mutant deltas.

        Parameters
        ----------
        delta1 : array-like
            Single mutant 1 delta vector.
        delta2 : array-like
            Single mutant 2 delta vector.
        wt : array-like, optional
            Wild-type embedding (required if include_wt=True).

        Returns
        -------
        torch.Tensor
            Predicted Δ₁₂ (expected double-mutant delta).
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Convert to tensors
        d1 = torch.as_tensor(delta1, dtype=torch.float32)
        d2 = torch.as_tensor(delta2, dtype=torch.float32)

        # Handle batched vs single input
        squeeze_output = False
        if d1.ndim == 1:
            d1 = d1.unsqueeze(0)
            d2 = d2.unsqueeze(0)
            squeeze_output = True

        if wt is not None:
            wt = torch.as_tensor(wt, dtype=torch.float32)
            if wt.ndim == 1:
                wt = wt.unsqueeze(0)

        # Move to device
        d1 = d1.to(self.device)
        d2 = d2.to(self.device)
        if wt is not None:
            wt = wt.to(self.device)

        # Predict
        self._model.eval()
        with torch.no_grad():
            x = self._prepare_input(d1, d2, wt)
            pred = self._model(x)

        if squeeze_output:
            pred = pred.squeeze(0)

        return pred.cpu()

    def as_expectation_function(self) -> Callable:
        """
        Return a callable that can be used as the `expectation` parameter
        in EpistasisGeometry or classify_epistasis_from_embeddings.

        The returned function takes (WT, M1, M2) embeddings and returns
        the predicted M12_expected embedding.

        Returns
        -------
        callable
            Function with signature (WT, M1, M2) -> M12_expected

        Examples
        --------
        >>> model = EpistasisExpectationModel.load("model.pt")
        >>> expectation_fn = model.as_expectation_function()
        >>>
        >>> # Use with EpistasisGeometry
        >>> geom = EpistasisGeometry(wt, m1, m2, m12, expectation=expectation_fn)
        >>>
        >>> # Use with classify_epistasis_from_embeddings
        >>> result = classify_epistasis_from_embeddings(
        ...     wt, m1, m2, m12, expectation=expectation_fn
        ... )
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call fit() or load() first.")

        def expectation_fn(WT, M1, M2):
            """Predict M12_expected from (WT, M1, M2) embeddings."""
            # Convert to tensors
            WT = torch.as_tensor(WT).float()
            M1 = torch.as_tensor(M1).float()
            M2 = torch.as_tensor(M2).float()

            # Pool if needed (handle 2D inputs)
            if WT.ndim == 2:
                WT = WT.mean(dim=0)
            if M1.ndim == 2:
                M1 = M1.mean(dim=0)
            if M2.ndim == 2:
                M2 = M2.mean(dim=0)

            # Compute deltas
            delta1 = M1 - WT
            delta2 = M2 - WT

            # Predict delta12
            if self.include_wt:
                delta12_pred = self.predict(delta1, delta2, wt=WT)
            else:
                delta12_pred = self.predict(delta1, delta2)

            # Return M12_expected
            return WT + delta12_pred

        return expectation_fn

    def save(self, path: Union[str, Path]) -> None:
        """Save model to file."""
        if not self._trained:
            raise RuntimeError("Cannot save untrained model")

        state = {
            "architecture": self.architecture,
            "include_wt": self.include_wt,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "activation": self.activation,
            "embedding_dim": self._embedding_dim,
            "model_state": self._model.state_dict(),
        }
        torch.save(state, path)
        logger.info("Saved EpistasisExpectationModel to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "EpistasisExpectationModel":
        """Load model from file."""
        state = torch.load(path, map_location=device, weights_only=False)

        model = cls(
            architecture=state["architecture"],
            include_wt=state["include_wt"],
            hidden_dims=state["hidden_dims"],
            dropout=state["dropout"],
            activation=state["activation"],
            device=device,
        )

        model._model = model._build_model(state["embedding_dim"]).to(model.device)
        model._model.load_state_dict(state["model_state"])
        model._trained = True

        logger.info("Loaded EpistasisExpectationModel from %s", path)
        return model

    def __repr__(self) -> str:
        status = "trained" if self._trained else "untrained"
        return (
            f"EpistasisExpectationModel("
            f"architecture={self.architecture!r}, "
            f"include_wt={self.include_wt}, "
            f"status={status})"
        )


# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
# -------------------------------------------------------------------------
# 1. SETUP: Load your model and create a database per model
# -------------------------------------------------------------------------
#
# from genebeddings import (
#     VariantEmbeddingDB,
#     add_single_variant_metrics,
#     add_epistasis_metrics,
#     SingleVariantGeometry,
#     EpistasisGeometry,
# )
#
# # Load your embedding model (must have .embed(seq, pool="mean") method)
# from your_model_library import load_model
# borzoi = load_model("borzoi")
#
# # Create one SQLite database per model for reproducibility
# db = VariantEmbeddingDB("embeddings_borzoi.db")
#
# -------------------------------------------------------------------------
# 2. SINGLE VARIANT ANALYSIS (DataFrame)
# -------------------------------------------------------------------------
#
# import pandas as pd
#
# # DataFrame with mutation IDs (format: GENE:CHROM:POS:REF:ALT)
# df = pd.DataFrame({
#     "mut_id": [
#         "KRAS:12:25227342:G:A",
#         "KRAS:12:25227343:C:T",
#         "BRCA1:17:43071077:A:G",
#     ],
#     "strand": ["-", "-", "-"],  # Optional strand column
# })
#
# # Compute metrics - embeddings are saved automatically
# df = add_single_variant_metrics(
#     df,
#     db,
#     model=borzoi,                    # Model for computing embeddings
#     strand_col="strand",             # Column with "+"/"-" strand info
#     # OR: reverse_complement=True,   # Default strand for all rows
#     context=3000,                    # Context window (bases each side)
#     genome="hg38",                   # Genome assembly
# )
#
# # Result: df now has columns v_l2, v_diff, angle_over_pi, etc.
# print(df[["mut_id", "v_l2", "v_diff", "angle_over_pi"]])
#
# -------------------------------------------------------------------------
# 3. EPISTASIS ANALYSIS (DataFrame)
# -------------------------------------------------------------------------
#
# # DataFrame with epistasis IDs (format: mut1|mut2)
# df_epi = pd.DataFrame({
#     "epistasis_id": [
#         "KRAS:12:25227342:G:A|KRAS:12:25227344:C:T",
#         "KRAS:12:25227342:G:A|KRAS:12:25227346:A:G",
#     ],
#     "gene_strand": ["-", "-"],
# })
#
# # Compute epistasis metrics - all embeddings saved (WT, M1, M2, M12, deltas)
# df_epi = add_epistasis_metrics(
#     df_epi,
#     db,
#     model=borzoi,
#     strand_col="gene_strand",
#     include_complex=True,            # Complex plane coords (rho, theta)
#     include_triangle=True,           # Triangle-based coords
#     include_distances=True,          # All distance metrics
# )
#
# # Key metrics: rho (epistasis magnitude), theta_over_pi (epistasis direction)
# print(df_epi[["epistasis_id", "rho", "theta_over_pi", "epi_R_raw"]])
#
# -------------------------------------------------------------------------
# 4. USING EXPLICIT COORDINATE COLUMNS
# -------------------------------------------------------------------------
#
# # If your DataFrame has separate columns instead of mut_id format:
# df_explicit = pd.DataFrame({
#     "gene": ["KRAS", "KRAS"],
#     "chr": ["12", "12"],
#     "position": [25227342, 25227343],
#     "ref_allele": ["G", "C"],
#     "alt_allele": ["A", "T"],
#     "strand": ["-", "-"],
#     "mut_id": ["KRAS:12:25227342:G:A", "KRAS:12:25227343:C:T"],  # Still needed as key
# })
#
# df_explicit = add_single_variant_metrics(
#     df_explicit,
#     db,
#     model=borzoi,
#     chrom_col="chr",
#     pos_col="position",
#     ref_col="ref_allele",
#     alt_col="alt_allele",
#     strand_col="strand",
# )
#
# -------------------------------------------------------------------------
# 5. MULTIPLE MODELS (separate databases)
# -------------------------------------------------------------------------
#
# db_borzoi = VariantEmbeddingDB("embeddings_borzoi.db")
# db_evo2 = VariantEmbeddingDB("embeddings_evo2.db")
#
# # Same DataFrame, different models with prefixes
# df = add_single_variant_metrics(df, db_borzoi, model=borzoi, prefix="borzoi_")
# df = add_single_variant_metrics(df, db_evo2, model=evo2, prefix="evo2_")
#
# # Compare metrics across models
# print(df[["mut_id", "borzoi_v_l2", "evo2_v_l2"]])
#
# -------------------------------------------------------------------------
# 6. MANUAL GEOMETRY ANALYSIS (without DataFrame)
# -------------------------------------------------------------------------
#
# # Load embeddings from database
# h_wt = db.load("KRAS:12:25227342:G:A|WT")
# h_mut = db.load("KRAS:12:25227342:G:A|MUT")
#
# # Single variant geometry
# geom = SingleVariantGeometry(h_wt, h_mut, diff="cosine")
# coords = geom.canonical_coords()
# print(f"Effect magnitude: {coords.v_l2:.4f}")
# print(f"Distance (cosine): {coords.v_diff:.4f}")
# geom.plot()
#
# # Epistasis geometry
# h_wt = db.load("KRAS:12:25227342:G:A|KRAS:12:25227344:C:T|WT")
# h_m1 = db.load("KRAS:12:25227342:G:A|KRAS:12:25227344:C:T|M1")
# h_m2 = db.load("KRAS:12:25227342:G:A|KRAS:12:25227344:C:T|M2")
# h_m12 = db.load("KRAS:12:25227342:G:A|KRAS:12:25227344:C:T|M12")
#
# epi = EpistasisGeometry(h_wt, h_m1, h_m2, h_m12, diff="cosine")
# metrics = epi.metrics()
# print(f"Epistasis magnitude: {metrics.epi_R_raw:.4f}")
# epi.plot()           # Complex plane visualization
# epi.plot_triangle()  # Triangle-based visualization
#
# -------------------------------------------------------------------------
# 7. DATABASE OPERATIONS
# -------------------------------------------------------------------------
#
# # Using context manager (recommended)
# with VariantEmbeddingDB("embeddings.db") as db:
#     db.store("my_variant", embedding_tensor)
#     emb = db.load("my_variant")
#
# # Check what's stored
# print(f"Total embeddings: {len(db)}")
# print(f"Has variant: {'my_variant' in db}")
#
# # List all keys matching a pattern
# kras_keys = db.list_keys(pattern="KRAS:%")
#
# # Iterate over all embeddings
# for mut_id, embedding in db.iter_all(as_torch=True):
#     print(mut_id, embedding.shape)
#
# # Store metadata about the database
# from genebeddings import DBMetadata
# db.store_db_metadata(DBMetadata(
#     model_name="borzoi",
#     embedding_dim=1024,
#     description="KRAS variant embeddings"
# ))
#
# -------------------------------------------------------------------------
# 8. STRAND SPECIFICATION OPTIONS
# -------------------------------------------------------------------------
#
# # The strand determines whether to reverse complement the sequence.
# # Multiple formats are accepted:
# #
# # Positive strand (no reverse complement):
# #   "+", "P", "positive", "pos", "forward", "fwd", "", 1, True
# #
# # Negative strand (reverse complement):
# #   "-", "N", "negative", "neg", "reverse", "rev", -1, 0, False
# #
# # Option A: Per-row strand from DataFrame column
# df = add_single_variant_metrics(df, db, model=m, strand_col="strand")
#
# # Option B: Global default for all rows
# df = add_single_variant_metrics(df, db, model=m, reverse_complement=True)
#
# # Option C: Encoded in mut_id (GENE:CHR:POS:REF:ALT:STRAND)
# # e.g., "KRAS:12:25227342:G:A:N" for negative strand
#
# -------------------------------------------------------------------------
# 9. PROGRESS TRACKING (with tqdm)
# -------------------------------------------------------------------------
#
# from tqdm import tqdm
#
# pbar = tqdm(total=len(df))
# def update_progress(i, total, mut_id):
#     pbar.update(1)
#     pbar.set_description(f"Processing {mut_id[:20]}...")
#
# df = add_single_variant_metrics(
#     df, db, model=borzoi,
#     progress_callback=update_progress
# )
# pbar.close()
#
# -------------------------------------------------------------------------
# 10. LOGGING
# -------------------------------------------------------------------------
#
# import logging
#
# # Enable debug logging to see what's happening
# logging.basicConfig(level=logging.DEBUG)
#
# # Or just for this module
# logging.getLogger("genebeddings").setLevel(logging.DEBUG)
#
