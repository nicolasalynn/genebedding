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
    - e_perp: orthogonal unit vector derived from single-mutant effect

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
        Observed double-mutant y-coordinate.
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

    Attributes
    ----------
    WT, M1, M2, M12 : torch.Tensor
        Pooled embeddings.
    v1, v2 : torch.Tensor
        Single-mutant effect vectors.
    v12_obs : torch.Tensor
        Observed double-mutant effect vector.
    v12_exp : torch.Tensor
        Expected (additive) double-mutant effect vector.
    mut_swapped : bool
        True if M1 and M2 were swapped to enforce d(WT,M1) <= d(WT,M2).

    Examples
    --------
    >>> epi = EpistasisGeometry(h_wt, h_m1, h_m2, h_m12)
    >>> metrics = epi.metrics()
    >>> print(f"Epistasis magnitude: {metrics.epi_R_raw:.4f}")
    >>> epi.plot()
    >>> epi.plot_triangle()
    """

    def __init__(
        self,
        h_ref: torch.Tensor,
        h_m1: torch.Tensor,
        h_m2: torch.Tensor,
        h_m12: torch.Tensor,
        eps: float = DEFAULT_EPS,
        diff: Union[str, DiffFn] = "cosine",
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

        # Pool inputs
        self.WT = self._pool(h_ref)
        m1_vec = self._pool(h_m1)
        m2_vec = self._pool(h_m2)
        self.M12 = self._pool(h_m12)

        # Enforce convention: M1 is closer to WT than M2
        d1 = self._embed_dist(self.WT, m1_vec)
        d2 = self._embed_dist(self.WT, m2_vec)

        if d1 > d2:
            m1_vec, m2_vec = m2_vec, m1_vec
            self.mut_swapped = True
            logger.debug("Swapped M1 and M2 to enforce d(WT,M1) <= d(WT,M2)")
        else:
            self.mut_swapped = False

        self.M1 = m1_vec
        self.M2 = m2_vec

        # Effect vectors
        self.v1 = self.M1 - self.WT
        self.v2 = self.M2 - self.WT
        self.v12_obs = self.M12 - self.WT
        self.v12_exp = self.v1 + self.v2
        self.v_1to2 = self.M1 - self.M2

        # Cached results
        self._cached_metrics: Optional[EpistasisMetrics] = None
        self._cached_complex: Optional[EpistasisComplexCoords] = None
        self._cached_triangle: Optional[EpistasisTriangleCoords] = None
        self._cached_mds: Optional[tuple[dict[str, np.ndarray], np.ndarray]] = None

        logger.debug(
            "EpistasisGeometry initialized: dim=%d, diff=%s, swapped=%s",
            self.WT.shape[0],
            self.diff_name,
            self.mut_swapped,
        )

    def __repr__(self) -> str:
        m = self.metrics()
        return (
            f"EpistasisGeometry("
            f"R_raw={m.epi_R_raw:.4f}, "
            f"R_expected={m.epi_R_expected:.4f}, "
            f"swapped={self.mut_swapped}, "
            f"diff={self.diff_name!r})"
        )

    def complex_coords(self) -> EpistasisComplexCoords:
        """
        Compute canonical complex epistasis coordinates.

        Builds a 2D plane in embedding space:
        - e_par: unit vector along WT→expected
        - e_perp: orthogonal unit vector from v1

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
        v1_res = self.v1 - torch.dot(self.v1, e_par) * e_par
        e_perp = self._unit(v1_res)

        x_exp_raw = float(torch.dot(self.v12_exp, e_par))
        y_exp_raw = float(torch.dot(self.v12_exp, e_perp))
        x_obs_raw = float(torch.dot(self.v12_obs, e_par))
        y_obs_raw = float(torch.dot(self.v12_obs, e_perp))

        r_exp = math.sqrt(x_exp_raw**2 + y_exp_raw**2) + self.eps

        x_exp = x_exp_raw / r_exp
        y_exp = y_exp_raw / r_exp
        x_obs = x_obs_raw / r_exp
        y_obs = y_obs_raw / r_exp

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
        M12_exp = WT + (M1 - WT) + (M2 - WT)

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate embeddings for a single variant.

    Parameters
    ----------
    model : object
        Model with `.embed(sequence: str, pool='mean') -> Tensor` method.
    mut_id : str
        Mutation identifier (see parse_single_mut_id).
    context : int, optional
        Context window size (bases on each side). Default: 3000.
    genome : str, optional
        Genome assembly name. Default: "hg38".

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

    h_ref = model.embed(s.seq, pool="mean")
    h_mut = model.embed(m.seq, pool="mean")
    delta = h_mut - h_ref

    return h_ref, h_mut, delta


def embed_epistasis(
    model,
    epistasis_id: str,
    context: int = DEFAULT_CONTEXT,
    genome: str = DEFAULT_GENOME,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate embeddings for an epistasis pair.

    Parameters
    ----------
    model : object
        Model with `.embed(sequence: str, pool='mean') -> Tensor` method.
    epistasis_id : str
        Epistasis identifier (see parse_epistasis_id).
    context : int, optional
        Context window size. Default: 3000.
    genome : str, optional
        Genome assembly name. Default: "hg38".

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

    h_ref = model.embed(s.seq, pool="mean")
    h_m1 = model.embed(m1.seq, pool="mean")
    h_m2 = model.embed(m2.seq, pool="mean")
    h_m12 = model.embed(m12.seq, pool="mean")

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute embeddings for a single variant from genomic coordinates.

    Returns (h_wt, h_mut, delta) as mean-pooled embeddings.
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

    h_wt = model.embed(s.seq, pool="mean")
    h_mut = model.embed(m.seq, pool="mean")
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute embeddings for epistasis from genomic coordinates.

    Returns (h_wt, h_m1, h_m2, h_m12) as mean-pooled embeddings.
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

    h_wt = model.embed(s.seq, pool="mean")
    h_m1 = model.embed(m1.seq, pool="mean")
    h_m2 = model.embed(m2.seq, pool="mean")
    h_m12 = model.embed(m12.seq, pool="mean")

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
        Model with `.embed(sequence: str, pool='mean') -> Tensor` method.
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

    for i, (idx, row) in enumerate(df.iterrows()):
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
    inplace: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
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
        Model with `.embed(sequence: str, pool='mean') -> Tensor` method.
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
    inplace : bool, optional
        Modify DataFrame in place. Default: False.
    progress_callback : callable, optional
        Called with (current_index, total, epistasis_id) for progress.

    Returns
    -------
    pd.DataFrame
        DataFrame with added metric columns.

    Examples
    --------
    >>> # With model - computes and saves all embeddings
    >>> df = add_epistasis_metrics(df, db, model=borzoi)

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
    """
    _require_pandas()

    if id_col not in df.columns:
        raise ValueError(f"Column {id_col!r} not found in DataFrame")

    if not inplace:
        df = df.copy()

    # Build list of metric columns to add
    metric_cols: list[str] = ["mut_swapped"]

    if include_complex:
        metric_cols.extend([
            "x_exp", "y_exp", "x_obs", "y_obs",
            "rho", "theta", "theta_over_pi"
        ])

    if include_triangle:
        metric_cols.extend([
            "x_exp_tri", "y_exp_tri", "x_obs_tri", "y_obs_tri",
            "rho_tri", "theta_tri", "theta_tri_over_pi"
        ])

    if include_distances:
        metric_cols.extend([
            "dist_WT_M1", "dist_WT_M2", "dist_WT_M12_obs", "dist_WT_M12_exp",
            "dist_M1_M2", "dist_obs_exp",
            "len_WT_M1", "len_WT_M2", "len_WT_M12", "len_WT_M12_exp", "len_M1_M2",
            "epi_R_raw", "epi_R_singles", "epi_R_expected"
        ])

    # Initialize columns
    for name in metric_cols:
        if name == "mut_swapped":
            df[f"{prefix}{name}"] = None
        else:
            df[f"{prefix}{name}"] = float("nan")

    n_processed = 0
    n_computed = 0
    n_loaded = 0
    n_skipped = 0
    total = len(df)

    for i, (idx, row) in enumerate(df.iterrows()):
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
        else:
            # Parse from epistasis_id
            try:
                (chrom, pos1, ref1, alt1, _), (_, pos2, ref2, alt2, _) = parse_epistasis_id(epi_id)
            except ValueError as e:
                logger.warning("Cannot parse %s: %s", epi_id, e)
                n_skipped += 1
                continue

        # Determine strand
        if strand_col and strand_col in df.columns:
            rev = _parse_strand(row[strand_col])
        else:
            rev = reverse_complement

        # Check if embeddings exist
        wt_key = f"{epi_id}{KEY_WT}"
        m1_key = f"{epi_id}{KEY_M1}"
        m2_key = f"{epi_id}{KEY_M2}"
        m12_key = f"{epi_id}{KEY_M12}"

        if db.has(wt_key) and db.has(m1_key) and db.has(m2_key) and db.has(m12_key):
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

        # Compute geometry
        geom = EpistasisGeometry(h_wt, h_m1, h_m2, h_m12, diff=diff)

        # Assign mut_swapped
        df.at[idx, f"{prefix}mut_swapped"] = geom.mut_swapped

        # Assign complex coords
        if include_complex:
            coords = geom.complex_coords()
            for name in ["x_exp", "y_exp", "x_obs", "y_obs",
                         "rho", "theta", "theta_over_pi"]:
                df.at[idx, f"{prefix}{name}"] = getattr(coords, name)

        # Assign triangle coords
        if include_triangle:
            tri = geom.triangle_coords()
            df.at[idx, f"{prefix}x_exp_tri"] = tri.x_exp
            df.at[idx, f"{prefix}y_exp_tri"] = tri.y_exp
            df.at[idx, f"{prefix}x_obs_tri"] = tri.x_obs
            df.at[idx, f"{prefix}y_obs_tri"] = tri.y_obs
            df.at[idx, f"{prefix}rho_tri"] = tri.rho
            df.at[idx, f"{prefix}theta_tri"] = tri.theta
            df.at[idx, f"{prefix}theta_tri_over_pi"] = tri.theta_over_pi

        # Assign distance metrics
        if include_distances:
            metrics = geom.metrics()
            for name in [
                "dist_WT_M1", "dist_WT_M2", "dist_WT_M12_obs", "dist_WT_M12_exp",
                "dist_M1_M2", "dist_obs_exp",
                "len_WT_M1", "len_WT_M2", "len_WT_M12", "len_WT_M12_exp", "len_M1_M2",
                "epi_R_raw", "epi_R_singles", "epi_R_expected"
            ]:
                df.at[idx, f"{prefix}{name}"] = getattr(metrics, name)

        n_processed += 1

    logger.info(
        "Epistasis metrics: %d processed (%d computed, %d loaded), %d skipped",
        n_processed, n_computed, n_loaded, n_skipped
    )

    return df


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
