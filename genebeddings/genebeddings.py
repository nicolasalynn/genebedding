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
    "add_conditional_nucleotide_probs",
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
    "compute_logodds_dependency_map",
    "compute_embedding_perturbation_map",
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
    Clean, minimal metrics for epistasis analysis.

    Focuses on interpretable metrics with minimal assumptions:
    - Deviation magnitude (how much epistasis)
    - Direction indicator (toward or away from WT)
    - Raw effect sizes (for context)

    Attributes
    ----------
    len_WT_M1 : float
        L2 norm of effect vector v1 = M1 - WT. Single mutation 1 effect size.
    len_WT_M2 : float
        L2 norm of effect vector v2 = M2 - WT. Single mutation 2 effect size.
    len_WT_M12 : float
        L2 norm of observed effect vector v12_obs = M12 - WT. Observed double effect.
    len_WT_M12_exp : float
        L2 norm of expected effect vector v12_exp = v1 + v2. Expected double effect.
    epi_R_raw : float
        Raw epistasis residual: ||M12_obs - M12_exp||.
        Absolute deviation from additive expectation.
    epi_R_singles : float
        Normalized residual: epi_R_raw / sqrt(||v1||² + ||v2||²).
        Allows comparison across events with different single-mutation magnitudes.
        Fewer assumptions than normalizing by expected.
    cos_v1_v2 : float
        Cosine similarity between single-mutant effects v1 and v2.
        +1 = same direction (effects reinforce)
        -1 = opposite directions (effects cancel)
        Useful context for interpreting epistasis.
    cos_exp_to_obs : float
        Cosine similarity between v12_exp and (v12_obs - v12_exp).
        Indicates direction of deviation relative to WT:
        -1 = observed moved back toward WT (corrective/sub-additive)
        0 = orthogonal deviation
        +1 = observed moved further from WT (synergistic/super-additive)
    magnitude_ratio : float
        Ratio of observed to expected magnitude: ||v12_obs|| / ||v12_exp||.
        <1 = closer to WT than expected (sub-additive)
        >1 = further from WT than expected (super-additive)
    log_magnitude_ratio : float
        Log ratio of observed to expected magnitude:
        log((||v12_obs|| + eps) / (||v12_exp|| + eps)).
        0 = additive, negative = sub-additive, positive = super-additive.
    """

    len_WT_M1: float
    len_WT_M2: float
    len_WT_M12: float
    len_WT_M12_exp: float
    epi_R_raw: float
    epi_R_singles: float
    cos_v1_v2: float
    cos_exp_to_obs: float
    magnitude_ratio: float
    log_magnitude_ratio: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for compatibility."""
        return {
            "len_WT_M1": self.len_WT_M1,
            "len_WT_M2": self.len_WT_M2,
            "len_WT_M12": self.len_WT_M12,
            "len_WT_M12_exp": self.len_WT_M12_exp,
            "epi_R_raw": self.epi_R_raw,
            "epi_R_singles": self.epi_R_singles,
            "cos_v1_v2": self.cos_v1_v2,
            "cos_exp_to_obs": self.cos_exp_to_obs,
            "magnitude_ratio": self.magnitude_ratio,
            "log_magnitude_ratio": self.log_magnitude_ratio,
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
        Expected double-mutant effect vector (additive or model-predicted).

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

        # Pool inputs - keep original order (M1 = first mutation, M2 = second mutation)
        self.WT = self._pool(h_ref)
        self.M1 = self._pool(h_m1)
        self.M2 = self._pool(h_m2)
        self.M12 = self._pool(h_m12)

        # Effect vectors
        self.v1 = self.M1 - self.WT
        self.v2 = self.M2 - self.WT
        self.v12_obs = self.M12 - self.WT

        # Additive expectation: v12_exp = v1 + v2
        self.v12_exp = self.v1 + self.v2

        self.v_1to2 = self.M1 - self.M2

        # Cached results
        self._cached_metrics: Optional[EpistasisMetrics] = None
        self._cached_complex: Optional[EpistasisComplexCoords] = None
        self._cached_triangle: Optional[EpistasisTriangleCoords] = None
        self._cached_mds: Optional[tuple[dict[str, np.ndarray], np.ndarray]] = None

        logger.debug(
            "EpistasisGeometry initialized: dim=%d, diff=%s",
            self.WT.shape[0],
            self.diff_name,
        )

    def __repr__(self) -> str:
        m = self.metrics()
        return (
            f"EpistasisGeometry("
            f"R_raw={m.epi_R_raw:.4f}, "
            f"R_expected={m.epi_R_expected:.4f}, "
            f"diff={self.diff_name!r})"
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
        Compute clean, minimal epistasis metrics.

        Returns
        -------
        EpistasisMetrics
            Dataclass with all metric values.
        """
        if self._cached_metrics is not None:
            return self._cached_metrics

        v1, v2, v12, v12_exp = self.v1, self.v2, self.v12_obs, self.v12_exp

        # Vector lengths (L2 norms) - raw effect sizes
        a1 = self._safe_norm(v1)
        a2 = self._safe_norm(v2)
        a12 = self._safe_norm(v12)
        a12_exp = self._safe_norm(v12_exp)

        # Residual: difference between observed and expected
        residual = v12 - v12_exp
        R_raw = self._safe_norm(residual)

        # Normalized by single mutation magnitudes (fewer assumptions)
        single_scale = math.sqrt(a1**2 + a2**2) + self.eps
        R_singles = R_raw / single_scale

        # Cosine helper
        def _cosine(u: torch.Tensor, v: torch.Tensor) -> float:
            nu = self._safe_norm(u)
            nv = self._safe_norm(v)
            if nu < self.eps or nv < self.eps:
                return 0.0
            return float(torch.dot(u, v) / (nu * nv))

        # Single mutation alignment
        cos_v1_v2 = _cosine(v1, v2)

        # Direction indicator: cos(v12_exp, residual)
        # -1 = moved back toward WT (corrective)
        # +1 = moved further from WT (synergistic)
        cos_exp_to_obs = _cosine(v12_exp, residual)

        # Magnitude ratio: how much closer/further from WT than expected
        magnitude_ratio = a12 / (a12_exp + self.eps)
        log_magnitude_ratio = math.log((a12 + self.eps) / (a12_exp + self.eps))

        self._cached_metrics = EpistasisMetrics(
            len_WT_M1=a1,
            len_WT_M2=a2,
            len_WT_M12=a12,
            len_WT_M12_exp=a12_exp,
            epi_R_raw=R_raw,
            epi_R_singles=R_singles,
            cos_v1_v2=cos_v1_v2,
            cos_exp_to_obs=cos_exp_to_obs,
            magnitude_ratio=magnitude_ratio,
            log_magnitude_ratio=log_magnitude_ratio,
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
        A_o = m.len_WT_M12       # observed effect magnitude
        A_e = m.len_WT_M12_exp   # expected effect magnitude
        A_r = m.epi_R_raw        # residual magnitude

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

    # Try batched forward pass (single call, ~2x faster on GPU)
    try:
        embs = model.embed([s.seq, m.seq], pool=pool)
        h_ref, h_mut = embs[0], embs[1]
    except (TypeError, AttributeError):
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

    # Try batched forward pass (single call, ~4x faster on GPU)
    try:
        embs = model.embed([s.seq, m1.seq, m2.seq, m12.seq], pool=pool)
        h_ref, h_m1, h_m2, h_m12 = embs[0], embs[1], embs[2], embs[3]
    except (TypeError, AttributeError):
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

    # Try batched forward pass (single call, ~2x faster on GPU)
    try:
        embs = model.embed([s.seq, m.seq], pool=pool)
        h_wt, h_mut = embs[0], embs[1]
    except (TypeError, AttributeError):
        h_wt = model.embed(s.seq, pool=pool)
        h_mut = model.embed(m.seq, pool=pool)
    delta = h_mut - h_wt

    return h_wt, h_mut, delta


def _model_max_seq_len(model, context: int) -> int:
    """Infer max sequence length from the model, capped at 4*context.

    Uses the model's declared max input length if available and smaller
    than 4*context.  The 4*context cap ensures variant pairs that are
    too far apart to meaningfully interact are skipped regardless of
    the model's theoretical capacity.
    """
    cap = 4 * context
    for attr in ("max_length", "min_input_len", "max_seq_len"):
        val = getattr(model, attr, None)
        if val is not None and isinstance(val, int) and val > 0:
            return min(val, cap)
    return cap


def _prepare_epistasis_sequences(
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
    max_seq_len: int = 0,
) -> tuple[str, str, str, str]:
    """
    Prepare the 4 sequences (WT, M1, M2, M12) for an epistasis pair.

    Returns (wt_seq, m1_seq, m2_seq, m12_seq) as raw strings ready for
    ``model.embed()``.  All four sequences are identical length.

    Parameters
    ----------
    max_seq_len : int, default 0
        Maximum allowed sequence length. If > 0 and the resulting sequence
        would exceed this, raises ValueError (caller skips the pair).
        A good default is ``4 * context`` — variants farther apart than
        ``2 * context`` can't meaningfully "see" each other.
    """
    from seqmat import SeqMat

    p_min = min(pos1, pos2)
    p_max = max(pos1, pos2)
    start = p_min - context
    end = p_max + context - 1
    seq_len = end - start + 1

    if max_seq_len > 0 and seq_len > max_seq_len:
        raise ValueError(
            f"Variant pair too far apart: distance={p_max - p_min}, "
            f"seq_len={seq_len} > max_seq_len={max_seq_len}"
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

    return s.seq, m1.seq, m2.seq, m12.seq


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
    max_seq_len: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute embeddings for epistasis from genomic coordinates.

    Returns (h_wt, h_m1, h_m2, h_m12) embeddings with specified pooling.

    Automatically batches the four sequences into a single forward pass
    when the model supports list input, falling back to sequential
    embedding otherwise.
    """
    logger.debug(
        "Embedding epistasis chr%s:%d,%d (rev=%s)",
        chrom, pos1, pos2, reverse_complement
    )

    wt_seq, m1_seq, m2_seq, m12_seq = _prepare_epistasis_sequences(
        chrom, pos1, ref1, alt1, pos2, ref2, alt2,
        reverse_complement=reverse_complement,
        context=context,
        genome=genome,
        max_seq_len=max_seq_len,
    )

    # Try batched forward pass (single call, ~4x faster on GPU)
    try:
        embs = model.embed([wt_seq, m1_seq, m2_seq, m12_seq], pool=pool)
        h_wt, h_m1, h_m2, h_m12 = embs[0], embs[1], embs[2], embs[3]
    except (TypeError, AttributeError):
        # Fallback for models that don't accept list input
        h_wt = model.embed(wt_seq, pool=pool)
        h_m1 = model.embed(m1_seq, pool=pool)
        h_m2 = model.embed(m2_seq, pool=pool)
        h_m12 = model.embed(m12_seq, pool=pool)

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

    # Initialize metric columns with NaN (all at once to avoid fragmentation)
    metric_names = [
        "v_l2", "v_diff", "v_par", "v_perp",
        "x", "y", "radius", "angle", "angle_over_pi"
    ]
    col_names = [f"{prefix}{name}" for name in metric_names]
    new_cols_df = pd.DataFrame(float("nan"), index=df.index, columns=col_names)
    df = pd.concat([df, new_cols_df], axis=1)

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
    inplace: bool = False,
    show_progress: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    pool: str = "mean",
    force: bool = False,
    cov_inv: Optional[np.ndarray] = None,
    batch_size: int = 1,
) -> "pd.DataFrame":
    """
    Add epistasis geometry metrics to a DataFrame.

    For each row:
    1. Check if embeddings exist in database
    2. If not and model is provided, compute and save embeddings
    3. Compute simplified epistasis metrics using EpistasisGeometry

    Adds the following columns (with optional prefix):
    - len_WT_M1: Single mutation 1 effect size
    - len_WT_M2: Single mutation 2 effect size
    - len_WT_M12: Observed double mutant effect size
    - len_WT_M12_exp: Expected double mutant effect size (additive)
    - epi_R_raw: Raw epistasis residual (absolute deviation)
    - epi_R_singles: Normalized residual (comparable across events)
    - cos_v1_v2: Alignment of single mutation effects
    - cos_exp_to_obs: Direction indicator (-1=toward WT, +1=away from WT)
    - magnitude_ratio: Ratio of observed to expected effect
    - log_magnitude_ratio: Log ratio of observed to expected effect
    - epi_mahal: Mahalanobis distance of residual (if cov_inv provided)
    - mahal_obs: Mahalanobis magnitude of observed effect (if cov_inv provided)
    - mahal_add: Mahalanobis magnitude of additive effect (if cov_inv provided)
    - mahal_ratio: mahal_obs / mahal_add (if cov_inv provided)
    - log_mahal_ratio: log((mahal_obs + eps) / (mahal_add + eps))

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
    cov_inv : np.ndarray, optional
        Precomputed inverse covariance for residuals. If provided, adds
        covariance-aware epistasis metrics (epi_mahal, mahal_*).
    batch_size : int, optional
        Number of epistasis pairs to embed in a single forward pass.
        Each pair requires 4 sequences (WT, M1, M2, M12), so
        ``batch_size=8`` sends 32 sequences through the model at once.
        Requires a model whose ``.embed()`` accepts ``list[str]`` input
        (most wrappers do). Falls back to sequential if the model does
        not support batched input. Default: 1 (original per-pair behavior).

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

    >>> # Batched embedding - ~batch_size x faster on GPU
    >>> df = add_epistasis_metrics(df, db, model=borzoi, batch_size=8)

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

    # Simplified metrics - clean, minimal set with interpretable meaning
    metric_cols = [
        "len_WT_M1",       # Single mutation 1 effect size
        "len_WT_M2",       # Single mutation 2 effect size
        "len_WT_M12",      # Observed double mutant effect size
        "len_WT_M12_exp",  # Expected double mutant effect size (additive)
        "epi_R_raw",       # Raw epistasis residual (absolute deviation)
        "epi_R_singles",   # Normalized residual (comparable across events)
        "cos_v1_v2",       # Alignment of single mutation effects
        "cos_exp_to_obs",  # Direction indicator: -1=toward WT, +1=away from WT
        "magnitude_ratio",     # Ratio of observed to expected effect
        "log_magnitude_ratio", # Log ratio of observed to expected effect
    ]
    if cov_inv is not None:
        metric_cols.append("epi_mahal")
        metric_cols.extend(["mahal_obs", "mahal_add", "mahal_ratio", "log_mahal_ratio"])

    # Initialize columns with user prefix (all at once to avoid fragmentation)
    col_names = [f"{prefix}{name}" for name in metric_cols]
    new_cols_df = pd.DataFrame(float("nan"), index=df.index, columns=col_names)
    df = pd.concat([df, new_cols_df], axis=1)

    # ------------------------------------------------------------------
    # Helper: assign metrics for one row given its four embeddings
    # ------------------------------------------------------------------
    def _assign_metrics(idx, h_wt, h_m1, h_m2, h_m12, skip_mahal_columns=False):
        geom = EpistasisGeometry(h_wt, h_m1, h_m2, h_m12, diff=diff)
        metrics = geom.metrics()
        for name in metric_cols:
            if skip_mahal_columns and (name == "epi_mahal" or name in ("mahal_obs", "mahal_add", "mahal_ratio", "log_mahal_ratio")):
                continue  # already filled by batched Mahalanobis
            if name == "epi_mahal":
                residual = geom.v12_obs - geom.v12_exp
                r = residual.detach().cpu().numpy().astype(np.float64, copy=False)
                if cov_inv.shape[0] != r.shape[0] or cov_inv.shape[1] != r.shape[0]:
                    raise ValueError(
                        f"cov_inv shape {cov_inv.shape} does not match residual dim {r.shape[0]}"
                    )
                df.at[idx, f"{prefix}{name}"] = float(math.sqrt(r.T @ cov_inv @ r))
            elif name in ("mahal_obs", "mahal_add", "mahal_ratio", "log_mahal_ratio"):
                v_obs = geom.v12_obs.detach().cpu().numpy().astype(np.float64, copy=False)
                v_add = geom.v12_exp.detach().cpu().numpy().astype(np.float64, copy=False)
                if cov_inv.shape[0] != v_obs.shape[0] or cov_inv.shape[1] != v_obs.shape[0]:
                    raise ValueError(
                        f"cov_inv shape {cov_inv.shape} does not match residual dim {v_obs.shape[0]}"
                    )
                mahal_obs = float(math.sqrt(v_obs.T @ cov_inv @ v_obs))
                mahal_add = float(math.sqrt(v_add.T @ cov_inv @ v_add))
                if name == "mahal_obs":
                    df.at[idx, f"{prefix}{name}"] = mahal_obs
                elif name == "mahal_add":
                    df.at[idx, f"{prefix}{name}"] = mahal_add
                else:
                    ratio = mahal_obs / (mahal_add + DEFAULT_EPS)
                    df.at[idx, f"{prefix}mahal_ratio"] = ratio
                    df.at[idx, f"{prefix}log_mahal_ratio"] = math.log(
                        (mahal_obs + DEFAULT_EPS) / (mahal_add + DEFAULT_EPS)
                    )
            else:
                df.at[idx, f"{prefix}{name}"] = getattr(metrics, name)

    # ------------------------------------------------------------------
    # Helper: parse one row into coordinates + strand
    # ------------------------------------------------------------------
    def _parse_row(row):
        """Return (chrom, p1, r1, a1, p2, r2, a2, rev) or None on failure."""
        epi_id = row[id_col]
        use_columns = (
            chrom_col and pos1_col and ref1_col and alt1_col
            and pos2_col and ref2_col and alt2_col
        )
        if use_columns:
            chrom = str(row[chrom_col])
            p1 = int(row[pos1_col])
            r1 = str(row[ref1_col])
            a1 = str(row[alt1_col])
            p2 = int(row[pos2_col])
            r2 = str(row[ref2_col])
            a2 = str(row[alt2_col])
            rev_from_id = None
        else:
            try:
                (chrom, p1, r1, a1, rev1), (_, p2, r2, a2, _) = parse_epistasis_id(epi_id)
                rev_from_id = rev1
            except ValueError as e:
                logger.warning("Cannot parse %s: %s", epi_id, e)
                return None
        # Determine strand
        if strand_col and strand_col in df.columns:
            rev = _parse_strand(row[strand_col])
        elif rev_from_id is not None:
            rev = rev_from_id
        else:
            rev = reverse_complement
        return chrom, p1, r1, a1, p2, r2, a2, rev

    # ==================================================================
    # BATCHED PATH — batch_size > 1
    # ==================================================================
    if batch_size > 1 and model is not None:
        n_processed = 0
        n_computed = 0
        n_loaded = 0
        n_skipped = 0
        total = len(df)

        # Phase 1 — parse rows, check DB, partition into loaded / to_compute
        embeddings = {}   # idx -> (h_wt, h_m1, h_m2, h_m12)
        to_compute = []   # (idx, epi_id, chrom, p1, r1, a1, p2, r2, a2, rev)
        # Track duplicate epi_ids: compute once, copy to all duplicate indices
        _seen_to_compute = {}  # epi_id -> first idx queued for computation
        _dup_indices = {}      # epi_id -> [list of extra indices needing the same result]

        parse_iter = df.iterrows()
        if show_progress:
            parse_iter = tqdm(parse_iter, total=total, desc="Parsing / DB lookup")

        for idx, row in parse_iter:
            epi_id = row[id_col]
            parsed = _parse_row(row)
            if parsed is None:
                n_skipped += 1
                continue
            chrom, p1, r1, a1, p2, r2, a2, rev = parsed

            wt_key = f"{epi_id}{KEY_WT}"
            m1_key = f"{epi_id}{KEY_M1}"
            m2_key = f"{epi_id}{KEY_M2}"
            m12_key = f"{epi_id}{KEY_M12}"

            if not force and db.has(wt_key) and db.has(m1_key) and db.has(m2_key) and db.has(m12_key):
                embeddings[idx] = (db.load(wt_key), db.load(m1_key), db.load(m2_key), db.load(m12_key))
                n_loaded += 1
            elif epi_id in _seen_to_compute:
                # Duplicate epi_id already queued for computation — skip model call,
                # will copy embeddings from the first occurrence after Phase 2.
                _dup_indices.setdefault(epi_id, []).append(idx)
            else:
                _seen_to_compute[epi_id] = idx
                to_compute.append((idx, epi_id, chrom, p1, r1, a1, p2, r2, a2, rev))

        n_dedup = sum(len(v) for v in _dup_indices.values())
        if n_dedup:
            logger.info("Deduplicated %d duplicate epistasis_ids (will copy embeddings after compute)", n_dedup)

        # Phase 2 — batch compute missing embeddings
        if to_compute:
            from concurrent.futures import ThreadPoolExecutor

            _max_sl = _model_max_seq_len(model, context)

            def _prep_chunk(chunk):
                """Prepare sequences for a chunk (FASTA reads + mutations)."""
                seqs, items, skipped = [], [], 0
                for item in chunk:
                    (idx_i, epi_id_i, chrom_i,
                     p1_i, r1_i, a1_i, p2_i, r2_i, a2_i, rev_i) = item
                    try:
                        s = _prepare_epistasis_sequences(
                            chrom_i, p1_i, r1_i, a1_i, p2_i, r2_i, a2_i,
                            reverse_complement=rev_i, context=context, genome=genome,
                            max_seq_len=_max_sl,
                        )
                        seqs.extend(s)
                        items.append(item)
                    except Exception as e:
                        logger.warning("Failed to prepare sequences for %s: %s", epi_id_i, e)
                        skipped += 1
                return seqs, items, skipped

            chunks = [
                to_compute[i:i + batch_size]
                for i in range(0, len(to_compute), batch_size)
            ]
            chunk_iter = range(len(chunks))
            if show_progress:
                chunk_iter = tqdm(
                    list(chunk_iter),
                    desc=f"Embedding (batch_size={batch_size})",
                )

            # Prefetch: prepare batch N+1 on a background thread while GPU runs batch N.
            # SeqMat FASTA reads release the GIL (C I/O), so this overlaps with GPU compute.
            with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
                # Submit first chunk prep
                next_future = prefetch_pool.submit(_prep_chunk, chunks[0])

                for ci in chunk_iter:
                    # Get prepped data for current batch
                    all_seqs, valid_items, chunk_skipped = next_future.result()
                    n_skipped += chunk_skipped

                    # Submit next batch prep (overlaps with GPU forward pass below)
                    if ci + 1 < len(chunks):
                        next_future = prefetch_pool.submit(_prep_chunk, chunks[ci + 1])

                    if not all_seqs:
                        continue

                    # Single forward pass for the entire chunk
                    try:
                        all_embs = model.embed(all_seqs, pool=pool)
                    except (TypeError, AttributeError):
                        # Fallback: sequential embedding for models without batch support
                        all_embs = np.stack([model.embed(s, pool=pool) for s in all_seqs])

                    for j, item in enumerate(valid_items):
                        idx_i, epi_id_i = item[0], item[1]
                        h_wt = all_embs[j * 4]
                        h_m1 = all_embs[j * 4 + 1]
                        h_m2 = all_embs[j * 4 + 2]
                        h_m12 = all_embs[j * 4 + 3]

                        # Cache in DB
                        wt_key = f"{epi_id_i}{KEY_WT}"
                        m1_key = f"{epi_id_i}{KEY_M1}"
                        m2_key = f"{epi_id_i}{KEY_M2}"
                        m12_key = f"{epi_id_i}{KEY_M12}"
                        db.store(wt_key, h_wt)
                        db.store(m1_key, h_m1)
                        db.store(m2_key, h_m2)
                        db.store(m12_key, h_m12)
                        db.store(f"{epi_id_i}{KEY_DELTA1}", h_m1 - h_wt)
                        db.store(f"{epi_id_i}{KEY_DELTA2}", h_m2 - h_wt)
                        db.store(f"{epi_id_i}{KEY_DELTA12}", h_m12 - h_wt)

                        embeddings[idx_i] = (h_wt, h_m1, h_m2, h_m12)
                        n_computed += 1

        # Phase 2b — copy embeddings to duplicate indices
        for epi_id, dup_idxs in _dup_indices.items():
            first_idx = _seen_to_compute[epi_id]
            if first_idx in embeddings:
                for dup_idx in dup_idxs:
                    embeddings[dup_idx] = embeddings[first_idx]
                    n_loaded += 1

        # Phase 2.5 — batched Mahalanobis when cov_inv is set (avoids O(N*d^2) row-by-row loop)
        if cov_inv is not None and embeddings:
            _torch = None
            try:
                import torch
                _torch = torch
            except ImportError:
                pass

            def _to_np(x):
                if _torch is not None and isinstance(x, _torch.Tensor):
                    return x.detach().cpu().numpy().astype(np.float64, copy=False)
                return np.asarray(x, dtype=np.float64)

            inds = list(embeddings.keys())
            h_wt = np.stack([_to_np(embeddings[i][0]) for i in inds])
            h_m1 = np.stack([_to_np(embeddings[i][1]) for i in inds])
            h_m2 = np.stack([_to_np(embeddings[i][2]) for i in inds])
            h_m12 = np.stack([_to_np(embeddings[i][3]) for i in inds])
            v12_obs = h_m12 - h_wt
            v12_exp = (h_m1 - h_wt) + (h_m2 - h_wt)
            residual = v12_obs - v12_exp
            # (N, d) @ (d, d) -> (N, d); then (N, d) * (N, d) sum axis=1 -> (N,)
            epi_mahal = np.sqrt(np.maximum(0.0, np.sum((residual @ cov_inv) * residual, axis=1)))
            mahal_obs = np.sqrt(np.maximum(0.0, np.sum((v12_obs @ cov_inv) * v12_obs, axis=1)))
            mahal_add = np.sqrt(np.maximum(0.0, np.sum((v12_exp @ cov_inv) * v12_exp, axis=1)))
            eps = DEFAULT_EPS
            mahal_ratio = mahal_obs / (mahal_add + eps)
            log_mahal_ratio = np.log((mahal_obs + eps) / (mahal_add + eps))
            df.loc[inds, f"{prefix}epi_mahal"] = epi_mahal
            df.loc[inds, f"{prefix}mahal_obs"] = mahal_obs
            df.loc[inds, f"{prefix}mahal_add"] = mahal_add
            df.loc[inds, f"{prefix}mahal_ratio"] = mahal_ratio
            df.loc[inds, f"{prefix}log_mahal_ratio"] = log_mahal_ratio

        # Phase 3 — compute metrics (non-Mahal when batched above)
        metrics_iter = df.iterrows()
        if show_progress:
            metrics_iter = tqdm(metrics_iter, total=total, desc="Computing metrics")

        skip_mahal = cov_inv is not None and bool(embeddings)
        for i, (idx, row) in enumerate(metrics_iter):
            if progress_callback:
                progress_callback(i, total, str(row[id_col]))
            if idx not in embeddings:
                continue
            h_wt, h_m1, h_m2, h_m12 = embeddings[idx]
            _assign_metrics(idx, h_wt, h_m1, h_m2, h_m12, skip_mahal_columns=skip_mahal)
            n_processed += 1

        logger.info(
            "Epistasis metrics (batched, bs=%d): %d processed (%d computed, %d loaded), %d skipped",
            batch_size, n_processed, n_computed, n_loaded, n_skipped
        )
        return df

    # ==================================================================
    # ORIGINAL PATH — batch_size <= 1 (unchanged behaviour)
    # ==================================================================
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
                    max_seq_len=_model_max_seq_len(model, context),
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

        _assign_metrics(idx, h_wt, h_m1, h_m2, h_m12)
        n_processed += 1

    logger.info(
        "Epistasis metrics: %d processed (%d computed, %d loaded), %d skipped",
        n_processed, n_computed, n_loaded, n_skipped
    )

    return df


# ---------------------------------------------------------------------------
# Conditional Nucleotide Probability Analysis (RiNALMo / MLM)
# ---------------------------------------------------------------------------


def add_conditional_nucleotide_probs(
    df: "pd.DataFrame",
    model,
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
    prefix: str = "",
    inplace: bool = False,
    show_progress: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> "pd.DataFrame":
    """
    Probe conditional nucleotide dependencies between epistatic positions.

    For each epistasis pair (pos1, pos2), this function:
    1. Fetches the reference sequence around both positions
    2. For each nucleotide N1 in {A, C, G, T}, sets pos1 = N1
    3. Masks pos2 and queries the model for P(pos2 | pos1 = N1)
    4. Records the full 4-nucleotide probability distribution at pos2

    This reveals how the identity at site 1 influences the model's belief
    about site 2 — a direct probe of learned pairwise dependencies.

    Adds columns
    -------------
    - ``{prefix}ref1_nuc``: Reference nucleotide at pos1 (in sequence orientation)
    - ``{prefix}ref2_nuc``: Reference nucleotide at pos2 (in sequence orientation)
    - ``{prefix}{N1}_{N2}`` for N1, N2 in {A, C, G, T}:
        P(pos2 = N2 | pos1 = N1). 16 columns total forming a 4×4
        conditional probability matrix.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing epistasis data.
    model : object
        Model wrapper with a ``predict_nucleotides(seq, positions)`` method
        (e.g. ``RiNALMoWrapper``).
    id_col : str, optional
        Column with epistasis IDs (format: mut1|mut2). Default: ``"epistasis_id"``.
    chrom_col : str, optional
        Column with chromosome. If None, parsed from *id_col*.
    pos1_col, pos2_col : str, optional
        Columns with genomic positions. If None, parsed from *id_col*.
    ref1_col, ref2_col : str, optional
        Columns with reference alleles. If None, parsed from *id_col*.
    alt1_col, alt2_col : str, optional
        Columns with alternate alleles. If None, parsed from *id_col*.
    strand_col : str, optional
        Column with strand info (``"+"/"-"``). Overrides *reverse_complement*.
    reverse_complement : bool, optional
        Default strand orientation. ``True`` = negative strand. Default: ``False``.
    context : int, optional
        Context window size (bp flanking each position). Default: 3000.
    genome : str, optional
        Genome assembly. Default: ``"hg38"``.
    prefix : str, optional
        Prefix for new column names. Default: ``""``.
    inplace : bool, optional
        Modify DataFrame in place. Default: ``False``.
    show_progress : bool, optional
        Show tqdm progress bar. Default: ``False``.
    progress_callback : callable, optional
        Called with ``(current_index, total, epistasis_id)`` for progress.

    Returns
    -------
    pd.DataFrame
        DataFrame with added conditional probability columns.

    Examples
    --------
    >>> from genebeddings.wrappers.rinalmo_wrapper import RiNALMoWrapper
    >>> model = RiNALMoWrapper("giga-v1")
    >>> df = add_conditional_nucleotide_probs(df, model, show_progress=True)
    >>> # Each row now has 16 probability columns, e.g.:
    >>> df[["A_A", "A_C", "A_G", "A_T"]].head()  # P(pos2 | pos1=A)
    """
    _require_pandas()
    import pandas as pd
    from seqmat import SeqMat

    if id_col not in df.columns:
        raise ValueError(f"Column {id_col!r} not found in DataFrame")

    if not hasattr(model, "predict_nucleotides"):
        raise TypeError(
            f"Model {type(model).__name__} does not have a predict_nucleotides method. "
            "This function requires a masked-language model wrapper (e.g. RiNALMoWrapper)."
        )

    if not inplace:
        df = df.copy()

    nucs = ["A", "C", "G", "T"]

    # Build column names: 16 probability cols + 2 ref nucleotide cols
    prob_col_names = [f"{prefix}{n1}_{n2}" for n1 in nucs for n2 in nucs]
    meta_col_names = [f"{prefix}ref1_nuc", f"{prefix}ref2_nuc"]
    all_new_cols = meta_col_names + prob_col_names

    new_cols_df = pd.DataFrame(
        {c: (np.nan if c in prob_col_names else "") for c in all_new_cols},
        index=df.index,
    )
    df = pd.concat([df, new_cols_df], axis=1)

    n_processed = 0
    n_skipped = 0
    total = len(df)

    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=total, desc="Conditional nucleotide probs")

    for i, (idx, row) in enumerate(iterator):
        epi_id = row[id_col]

        if progress_callback:
            progress_callback(i, total, str(epi_id))

        # ---- Parse coordinates (same logic as add_epistasis_metrics) ----
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
            try:
                (chrom, pos1, ref1, alt1, rev1), (_, pos2, ref2, alt2, rev2) = (
                    parse_epistasis_id(epi_id)
                )
                rev_from_id = rev1
            except ValueError as e:
                logger.warning("Cannot parse %s: %s", epi_id, e)
                n_skipped += 1
                continue

        # Determine strand
        if strand_col and strand_col in df.columns:
            rev = _parse_strand(row[strand_col])
        elif rev_from_id is not None:
            rev = rev_from_id
        else:
            rev = reverse_complement

        # ---- Fetch reference sequence ----
        try:
            p_min = min(pos1, pos2)
            p_max = max(pos1, pos2)
            start = p_min - context
            end = p_max + context - 1

            s = SeqMat.from_fasta(genome, f"chr{chrom}", start, end)
            if rev:
                s.reverse_complement()

            ref_seq = s.seq
        except Exception as e:
            logger.warning("Failed to fetch sequence for %s: %s", epi_id, e)
            n_skipped += 1
            continue

        # ---- Compute 0-based offsets into the sequence string ----
        if rev:
            idx1 = end - pos1
            idx2 = end - pos2
        else:
            idx1 = pos1 - start
            idx2 = pos2 - start

        if not (0 <= idx1 < len(ref_seq)) or not (0 <= idx2 < len(ref_seq)):
            logger.warning(
                "Position out of bounds for %s: idx1=%d, idx2=%d, seq_len=%d",
                epi_id, idx1, idx2, len(ref_seq),
            )
            n_skipped += 1
            continue

        # Record reference nucleotides (in sequence orientation)
        df.at[idx, f"{prefix}ref1_nuc"] = ref_seq[idx1]
        df.at[idx, f"{prefix}ref2_nuc"] = ref_seq[idx2]

        # ---- For each nucleotide at pos1, get distribution at masked pos2 ----
        try:
            for n1 in nucs:
                # Build sequence with n1 at pos1
                seq_list = list(ref_seq)
                seq_list[idx1] = n1
                mutated_seq = "".join(seq_list)

                # Query model: mask pos2 and get nucleotide probabilities
                probs = model.predict_nucleotides(
                    mutated_seq, positions=[idx2], return_dict=True
                )
                # probs is a list of 1 dict: [{'A': p, 'C': p, 'G': p, 'T': p}]
                prob_dict = probs[0]

                for n2 in nucs:
                    df.at[idx, f"{prefix}{n1}_{n2}"] = prob_dict.get(n2, float("nan"))
        except Exception as e:
            logger.warning("Failed predict_nucleotides for %s: %s", epi_id, e)
            n_skipped += 1
            continue

        n_processed += 1

    logger.info(
        "Conditional nucleotide probs: %d processed, %d skipped (of %d total)",
        n_processed, n_skipped, total,
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


def compute_logodds_dependency_map(
    model,
    sequence: str,
    positions: Optional[list[int]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    symmetrize: bool = True,
    eps: float = 1e-10,
    show_progress: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    store_details: bool = False,
) -> DependencyMapResult:
    """
    Compute a dependency map using masked-language-model log-odds.

    For each ordered pair (i, j) the score D(i, j) is computed as follows:

    1. **Baseline**: mask position j in the original sequence and obtain the
       model's predicted nucleotide distribution P_base(j).
    2. **Perturbation**: for each of the 3 alternative nucleotides at
       position i, mutate i, mask j, and obtain the new distribution
       P_mut(j).
    3. **Log-odds**: for every nucleotide n at j, compute
       ``log(P_mut(j, n) / P_base(j, n))``.
    4. **Aggregate**: keep the largest absolute log-odds across the 4
       nucleotides at j, then take the maximum across the 3 mutations at i.

    By default the map is symmetrised: ``D_sym(i, j) = max(D(i, j), D(j, i))``.

    Complexity
    ----------
    The function makes ``1 + 3·N`` calls to ``predict_nucleotides``, each
    batching up to N masked positions internally (one forward pass per call).
    This is much cheaper than the ``O(N²)`` approach of calling per-pair.

    Parameters
    ----------
    model : object
        Model wrapper with a ``predict_nucleotides(seq, positions,
        return_dict=True)`` method (e.g. ``RiNALMoWrapper``).
    sequence : str
        Reference DNA/RNA sequence.
    positions : list[int], optional
        0-indexed positions to analyse. If None, uses *start*/*end*/*step*.
    start : int, optional
        Start position (inclusive). Default: 0.
    end : int, optional
        End position (exclusive). Default: ``len(sequence)``.
    step : int, optional
        Step size between positions. Default: 1.
    symmetrize : bool, optional
        If True (default), return ``max(D(i,j), D(j,i))``. If False,
        return the raw asymmetric matrix.
    eps : float, optional
        Small constant added to probabilities before taking the log to
        avoid ``log(0)``. Default: 1e-10.
    show_progress : bool, optional
        Show a tqdm progress bar. Default: False.
    progress_callback : callable, optional
        Called with ``(current_step, total_steps, description)``.
    store_details : bool, optional
        If True, store per-pair baseline and mutant distributions in
        ``result.details``. Useful for inspection but memory-intensive
        for large position sets. Default: False.

    Returns
    -------
    DependencyMapResult
        Object containing the dependency matrix and metadata.
        ``result.details["D_asymmetric"]`` always contains the raw
        asymmetric matrix before symmetrisation.

    Examples
    --------
    >>> from genebeddings.wrappers.rinalmo_wrapper import RiNALMoWrapper
    >>> model = RiNALMoWrapper("giga-v1")
    >>> result = compute_logodds_dependency_map(
    ...     model, sequence,
    ...     positions=[2900, 2950, 3000, 3050, 3100],
    ...     show_progress=True,
    ... )
    >>> result.plot(title="Log-odds dependency map")
    >>> result.top_pairs(5)
    """
    if not hasattr(model, "predict_nucleotides"):
        raise TypeError(
            f"Model {type(model).__name__} does not have a predict_nucleotides "
            "method. This function requires a masked-language model wrapper "
            "(e.g. RiNALMoWrapper)."
        )

    # ---- Determine positions ----
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
        raise ValueError("Need at least 2 positions for a dependency map")

    pos_list = pos_array.tolist()

    for p in pos_list:
        if p < 0 or p >= len(sequence):
            raise ValueError(
                f"Position {p} out of bounds for sequence length {len(sequence)}"
            )

    nucs = ["A", "C", "G", "T"]

    logger.info(
        "Computing log-odds dependency map: %d positions (%d baseline + "
        "%d mutation calls)",
        n, 1, 3 * n,
    )

    # ---- Step 1: baseline distributions (one batched call) ----
    baseline_dicts = model.predict_nucleotides(
        sequence, positions=pos_list, return_dict=True
    )
    # baseline_dicts: list of N dicts  [{'A': p, 'C': p, 'G': p, 'T': p}, ...]
    baseline_matrix = np.array(
        [[d[nt] for nt in nucs] for d in baseline_dicts]
    )  # (N, 4)

    # ---- Step 2: perturb each query position, read all targets ----
    D_raw = np.full((n, n), np.nan)  # asymmetric D(i, j)

    details: dict = {}

    total_steps = 3 * n  # 3 mutations per query position
    step_count = 0

    iterator = range(n)
    if show_progress:
        iterator = tqdm(iterator, total=n, desc="Log-odds dependency map")

    for i_idx in iterator:
        pos_i = int(pos_array[i_idx])
        ref_i = sequence[pos_i]
        alts_i = [nt for nt in nucs if nt != ref_i]

        # Accumulate max |log-odds| across the 3 mutations at i,
        # separately for each target j.
        max_logodds_per_j = np.zeros(n)

        for alt in alts_i:
            if progress_callback:
                progress_callback(
                    step_count, total_steps,
                    f"pos {pos_i}: {ref_i}>{alt}",
                )
            step_count += 1

            # Build mutated sequence (only pos_i changes)
            mutated_seq = sequence[:pos_i] + alt + sequence[pos_i + 1:]

            # Get distributions at ALL target positions in one call
            mut_dicts = model.predict_nucleotides(
                mutated_seq, positions=pos_list, return_dict=True
            )
            mut_matrix = np.array(
                [[d[nt] for nt in nucs] for d in mut_dicts]
            )  # (N, 4)

            # Log-odds per (target, nucleotide)
            log_odds = np.log(
                (mut_matrix + eps) / (baseline_matrix + eps)
            )  # (N, 4)

            # Max absolute log-odds across the 4 nucleotides at each target
            max_abs = np.max(np.abs(log_odds), axis=1)  # (N,)

            # Keep running max across the 3 mutations at i
            max_logodds_per_j = np.maximum(max_logodds_per_j, max_abs)

            # Optional: store per-mutation detail
            if store_details:
                for j_idx in range(n):
                    if j_idx == i_idx:
                        continue
                    pos_j = int(pos_array[j_idx])
                    key = (pos_i, pos_j)
                    if key not in details:
                        details[key] = {
                            "ref_i": ref_i,
                            "ref_j": sequence[pos_j],
                            "baseline_probs": {
                                nt: float(baseline_matrix[j_idx, k])
                                for k, nt in enumerate(nucs)
                            },
                            "mutations": [],
                        }
                    details[key]["mutations"].append({
                        "alt_i": alt,
                        "mutant_probs": {
                            nt: float(mut_matrix[j_idx, k])
                            for k, nt in enumerate(nucs)
                        },
                        "log_odds": {
                            nt: float(log_odds[j_idx, k])
                            for k, nt in enumerate(nucs)
                        },
                        "max_abs_logodds": float(max_abs[j_idx]),
                    })

        # Write row i of asymmetric matrix
        D_raw[i_idx, :] = max_logodds_per_j
        D_raw[i_idx, i_idx] = np.nan  # no self-dependency

    # ---- Step 3: symmetrise ----
    if symmetrize:
        matrix = np.fmax(D_raw, D_raw.T)  # element-wise max, NaN-safe
    else:
        matrix = D_raw

    # Always store the asymmetric matrix so users can inspect directionality
    details["D_asymmetric"] = D_raw.copy()

    logger.info(
        "Log-odds dependency map complete: %d positions, %d model calls",
        n, 1 + 3 * n,
    )

    return DependencyMapResult(
        positions=pos_array,
        matrix=matrix,
        aggregation="max" if symmetrize else "max_asymmetric",
        metric="logodds",
        sequence=sequence,
        details=details,
    )


def _detect_base_token_offset(model, sequence: str) -> int:
    """
    Detect the token index of the first base in ``pool='tokens'`` output.

    Mutates the first base and finds the token whose embedding changes
    **the most**, which must be the first base token (e.g. after a BOS
    token).  In attention-based models every token representation shifts
    when any input changes, so we cannot simply look for the first
    non-zero difference — we need the *largest* difference.
    """
    alt = "C" if sequence[0] != "C" else "A"
    seq_probe = alt + sequence[1:]

    tokens_orig = model.embed(sequence, pool="tokens", return_numpy=False)
    tokens_probe = model.embed(seq_probe, pool="tokens", return_numpy=False)

    tokens_orig = torch.as_tensor(tokens_orig).float()
    tokens_probe = torch.as_tensor(tokens_probe).float()

    diffs = torch.norm(tokens_orig - tokens_probe, dim=-1)  # (T,)
    offset = int(torch.argmax(diffs).item())
    return offset


def compute_embedding_perturbation_map(
    model,
    sequence: str,
    positions: Optional[list[int]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    diff: Union[str, DiffFn] = "cosine",
    aggregation: str = "max",
    symmetrize: bool = True,
    show_progress: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> DependencyMapResult:
    """
    Compute a dependency map by measuring embedding perturbation at each token.

    This is the **embedding-space analog** of
    :func:`compute_logodds_dependency_map`. Instead of asking *"how do the
    predicted probabilities at j change when I mutate i?"*, it asks *"how
    does the model's internal representation at j change when I mutate i?"*

    For each ordered pair (i, j):

    1. Embed the original sequence with ``pool='tokens'`` to get per-token
       representations.
    2. For each of the 3 alternative nucleotides at position i, mutate i,
       re-embed with ``pool='tokens'``, and measure the distance between
       the baseline and mutated embeddings **at token j**.
    3. Keep the maximum distance across the 3 mutations.

    By default the map is symmetrised:
    ``D(i, j) = max(D_raw(i, j), D_raw(j, i))``.

    Complexity
    ----------
    ``3 + 3·N`` forward passes (3 for offset detection + baseline, then
    one per mutation), each returning the full token-level embedding.
    This is the same ``O(N)`` scaling as the log-odds map and dramatically
    cheaper than the ``O(N²)`` embedding-epistasis map.

    Comparison with other dependency maps
    --------------------------------------
    - **Epistasis map** (``compute_dependency_map``): measures deviation
      from additive expectation — requires the double mutant (M12), so is
      ``O(N²)``. Captures true *epistasis*.
    - **Log-odds map** (``compute_logodds_dependency_map``): measures
      shifts in nucleotide probabilities. Requires an MLM head.
    - **Embedding perturbation map** (this function): measures shifts in
      hidden representations. Works with *any* model that supports
      ``pool='tokens'``, no MLM head required.

    Parameters
    ----------
    model : object
        Model wrapper with ``embed(seq, pool='tokens')`` support.
    sequence : str
        Reference DNA/RNA sequence.
    positions : list[int], optional
        0-indexed positions to analyse. If None, uses *start*/*end*/*step*.
    start : int, optional
        Start position (inclusive). Default: 0.
    end : int, optional
        End position (exclusive). Default: ``len(sequence)``.
    step : int, optional
        Step size between positions. Default: 1.
    diff : str or callable, optional
        Distance metric for comparing token embeddings: ``"cosine"``
        (default) or ``"l2"``, or a callable ``(x, y) -> float``.
    aggregation : str, optional
        How to aggregate across the 3 mutations at a query position:
        ``"max"`` (default) or ``"mean"``.
    symmetrize : bool, optional
        If True (default), return ``max(D(i,j), D(j,i))``.
    show_progress : bool, optional
        Show tqdm progress bar. Default: False.
    progress_callback : callable, optional
        Called with ``(current_step, total_steps, description)``.

    Returns
    -------
    DependencyMapResult
        Object containing the dependency matrix and metadata.
        ``result.details["D_asymmetric"]`` holds the raw asymmetric matrix.

    Examples
    --------
    >>> result = compute_embedding_perturbation_map(
    ...     model, sequence,
    ...     positions=[2900, 2950, 3000, 3050, 3100],
    ...     diff="cosine",
    ...     show_progress=True,
    ... )
    >>> result.plot(title="Embedding perturbation map")
    >>> result.top_pairs(5)
    """
    # ---- Resolve distance function ----
    if isinstance(diff, str):
        diff_lower = diff.lower()
        if diff_lower == "cosine":
            dist_fn = cosine_distance
        elif diff_lower == "l2":
            dist_fn = l2_distance
        else:
            raise ValueError(f"Unknown diff metric: {diff!r}")
        diff_name = diff_lower
    elif callable(diff):
        dist_fn = diff
        diff_name = getattr(diff, "__name__", "custom")
    else:
        raise TypeError(f"diff must be str or callable, got {type(diff)}")

    # ---- Determine positions ----
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
        raise ValueError("Need at least 2 positions for a dependency map")

    pos_list = pos_array.tolist()

    for p in pos_list:
        if p < 0 or p >= len(sequence):
            raise ValueError(
                f"Position {p} out of bounds for sequence length "
                f"{len(sequence)}"
            )

    nucs = ["A", "C", "G", "T"]
    k = getattr(model, "k", 1)

    logger.info(
        "Computing embedding perturbation map: %d positions, diff=%s",
        n, diff_name,
    )

    # ---- Detect base-token offset and get baseline embeddings ----
    offset = _detect_base_token_offset(model, sequence)
    tokens_wt = model.embed(sequence, pool="tokens", return_numpy=False)
    tokens_wt = torch.as_tensor(tokens_wt).float()  # (T, H)

    # Map sequence positions to token indices
    tok_indices = [offset + p // k for p in pos_list]

    T = tokens_wt.shape[0]
    for p, ti in zip(pos_list, tok_indices):
        if ti >= T:
            raise ValueError(
                f"Position {p} maps to token index {ti} but model "
                f"only returns {T} tokens"
            )

    # Extract baseline token embeddings at target positions: (N, H)
    baseline_tokens = tokens_wt[tok_indices]  # (N, H)

    # ---- Perturb each query position ----
    D_raw = np.full((n, n), np.nan)

    total_steps = 3 * n
    step_count = 0

    iterator = range(n)
    if show_progress:
        iterator = tqdm(iterator, total=n, desc="Embedding perturbation map")

    for i_idx in iterator:
        pos_i = int(pos_array[i_idx])
        ref_i = sequence[pos_i]
        alts_i = [nt for nt in nucs if nt != ref_i]

        # Accumulate distances across mutations: (n_mutations, N)
        all_dists = []

        for alt in alts_i:
            if progress_callback:
                progress_callback(
                    step_count, total_steps,
                    f"pos {pos_i}: {ref_i}>{alt}",
                )
            step_count += 1

            mutated_seq = sequence[:pos_i] + alt + sequence[pos_i + 1:]
            tokens_mut = model.embed(mutated_seq, pool="tokens", return_numpy=False)
            tokens_mut = torch.as_tensor(tokens_mut).float()  # (T, H)

            # Extract mutated token embeddings at target positions
            mut_tokens = tokens_mut[tok_indices]  # (N, H)

            # Compute distance for each target position j
            dists_j = np.array([
                dist_fn(baseline_tokens[j], mut_tokens[j])
                for j in range(n)
            ])
            all_dists.append(dists_j)

        all_dists = np.stack(all_dists, axis=0)  # (3, N)

        # Aggregate across mutations
        if aggregation == "max":
            D_raw[i_idx, :] = np.max(all_dists, axis=0)
        elif aggregation == "mean":
            D_raw[i_idx, :] = np.mean(all_dists, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation!r}")

        D_raw[i_idx, i_idx] = np.nan

    # ---- Symmetrise ----
    if symmetrize:
        matrix = np.fmax(D_raw, D_raw.T)
    else:
        matrix = D_raw

    details: dict = {"D_asymmetric": D_raw.copy()}

    logger.info(
        "Embedding perturbation map complete: %d positions, %d forward passes, "
        "diff=%s",
        n, 3 + 3 * n, diff_name,
    )

    return DependencyMapResult(
        positions=pos_array,
        matrix=matrix,
        aggregation=aggregation,
        metric=f"perturbation_{diff_name}",
        sequence=sequence,
        details=details,
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
# )
#
# # Key metrics:
# # - epi_R_raw: absolute deviation from additive expectation
# # - epi_R_singles: normalized (comparable across events)
# # - cos_exp_to_obs: direction indicator (-1=toward WT, +1=away from WT)
# # - magnitude_ratio: observed/expected effect ratio
# print(df_epi[["epistasis_id", "epi_R_raw", "epi_R_singles", "cos_exp_to_obs"]])
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
