import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from typing import Optional, Dict, Union, Callable


class EpistasisGeometry:
    """
    Epistasis geometry for 4 embeddings: WT, M1, M2, M12.

    - Works in the full embedding space:
        * defines effect vectors v1, v2, v12_obs, v12_exp
        * computes epistasis residual and normalized scores
        * gives simple classification labels

    - Builds a 2D MDS embedding of:
        WT, M1, M2, M12_obs, M12_exp

    - Can plot:
        * WT at origin
        * arrows WT→M1, WT→M2, WT→M12_obs, WT→M12_exp
        * distances from WT on arrows
        * a small table with epistasis metrics

    Parameters
    ----------
    h_ref : array-like
        Wild-type embedding
    h_m1 : array-like
        Single mutant 1 embedding
    h_m2 : array-like
        Single mutant 2 embedding
    h_m12 : array-like
        Double mutant embedding
    eps : float
        Small constant for numerical stability
    expectation : str or callable, optional
        How to compute expected double-mutant embedding. Options:
        - "additive": v1 + v2 (default, classical model)
        - "geometric": Geometric mean of effect magnitudes
        - "pythagorean": sqrt(||v1||^2 + ||v2||^2) combination
        - callable: Custom function(WT, M1, M2) -> M12_expected
    """

    # Built-in expectation models
    _EXPECTATION_MODELS = {
        "additive": lambda self: self.v1 + self.v2,
        "geometric": "_geometric_expectation",
        "pythagorean": "_pythagorean_expectation",
    }

    def __init__(
        self,
        h_ref,
        h_m1,
        h_m2,
        h_m12,
        eps: float = 1e-8,
        expectation: Union[str, Callable] = "additive",
    ):
        self.eps = eps
        self.expectation = expectation

        # ---- pool inputs ----
        self.WT  = self._pool(h_ref)
        self.M1  = self._pool(h_m1)
        self.M2  = self._pool(h_m2)
        self.M12 = self._pool(h_m12)

        # ---- effect vectors in embedding space ----
        self.v1       = self.M1  - self.WT          # WT→M1
        self.v2       = self.M2  - self.WT          # WT→M2
        self.v12_obs  = self.M12 - self.WT          # WT→M12 (observed)
        self.v12_exp  = self._compute_v12_exp()     # WT→M12 (expected)
        self.v_1to2   = self.M1 - self.M2          # M1↔M2

        # cached things
        self._metrics: Optional[Dict[str, float]] = None
        self._coords:  Optional[Dict[str, np.ndarray]] = None
        self._D:       Optional[np.ndarray] = None

    def _compute_v12_exp(self) -> torch.Tensor:
        """Compute expected v12 based on the expectation model."""
        if callable(self.expectation):
            # Custom model: expects (WT, M1, M2) -> M12_expected
            M12_exp = self.expectation(self.WT, self.M1, self.M2)
            M12_exp = torch.as_tensor(M12_exp).float()
            return M12_exp - self.WT
        elif isinstance(self.expectation, str):
            if self.expectation == "additive":
                return self.v1 + self.v2
            elif self.expectation == "geometric":
                return self._geometric_expectation()
            elif self.expectation == "pythagorean":
                return self._pythagorean_expectation()
            else:
                raise ValueError(
                    f"Unknown expectation model: {self.expectation}. "
                    f"Options: 'additive', 'geometric', 'pythagorean', or a callable"
                )
        else:
            raise ValueError(f"expectation must be a string or callable, got {type(self.expectation)}")

    def _geometric_expectation(self) -> torch.Tensor:
        """Geometric mean of effect magnitudes with averaged direction."""
        mag = torch.sqrt(self.v1.norm() * self.v2.norm() + self.eps)
        direction = self.v1 + self.v2
        direction = direction / (direction.norm() + self.eps)
        return mag * direction

    def _pythagorean_expectation(self) -> torch.Tensor:
        """Pythagorean combination: sqrt(||v1||^2 + ||v2||^2) with averaged direction."""
        mag = torch.sqrt(self.v1.norm()**2 + self.v2.norm()**2)
        direction = self.v1 + self.v2
        direction = direction / (direction.norm() + self.eps)
        return mag * direction

    # ----------------- helpers ----------------- #

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        h = torch.as_tensor(h).float()
        return h.mean(dim=0) if h.ndim == 2 else h

    def _cos_angle(self, u: torch.Tensor, v: torch.Tensor) -> float:
        num = torch.dot(u, v)
        denom = (torch.norm(u) * torch.norm(v) + self.eps)
        return float(num / denom)

    # ----------------- metrics ----------------- #

    def metrics(self) -> Dict[str, float]:
        """
        Compute and cache all scalar epistasis metrics.
        """
        if self._metrics is not None:
            return self._metrics

        v1, v2, v12, v12_exp = self.v1, self.v2, self.v12_obs, self.v12_exp

        # norms
        a1      = torch.norm(v1).item()
        a2      = torch.norm(v2).item()
        a12     = torch.norm(v12).item()
        a12_exp = torch.norm(v12_exp).item()
        a_1to2  = torch.norm(self.v_1to2).item()

        max_single = max(a1, a2)
        min_single = min(a1, a2)

        # angles vs singles
        c1 = self._cos_angle(v12, v1)
        c2 = self._cos_angle(v12, v2)
        same_direction = (c1 > 0.5) and (c2 > 0.5)

        # qualitative classification vs WT
        if a12 < min_single:
            type_WT = "corrective (closer to WT than either single)"
        elif a12 > max_single and same_direction:
            type_WT = "cumulative (same-direction aggravating)"
        elif a12 > max_single and not same_direction:
            type_WT = "divergent (further but rotated)"
        else:
            type_WT = "intermediate/ambiguous"

        # qualitative classification vs additive radius
        if a12 < a12_exp:
            type_add = "sub-additive (dampened)"
        elif a12 > a12_exp:
            type_add = "super-additive (synergistic)"
        else:
            type_add = "approximately additive"

        # residual epistasis vector
        residual_vec  = v12 - v12_exp
        residual_norm = torch.norm(residual_vec).item()

        # normalized epistasis (magnitude only)
        denom_exp = a12_exp if a12_exp > self.eps else self.eps
        epi_rel_expected = residual_norm / denom_exp

        single_scale = (a1**2 + a2**2) ** 0.5
        if single_scale < self.eps:
            single_scale = self.eps
        epi_rel_singles = residual_norm / single_scale

        # angle between observed and expected effect
        cos_v12_vexp = self._cos_angle(v12, v12_exp)
        cos_v12_vexp_clamped = max(min(cos_v12_vexp, 1.0), -1.0)
        # normalized angle in [0,1]: 0=same, 0.5=orthogonal, 1=opposite
        angle_v12_vexp = math.acos(cos_v12_vexp_clamped) / math.pi

        # radial shift: closer/further from WT than expected
        radial_shift = (a12 - a12_exp) / (a12_exp + self.eps)

        # signed non-linearity: sign says inward/outward vs WT
        epi_signed_radial = epi_rel_expected * np.sign(radial_shift)

        # a single geometric score combining magnitude and angle
        epi_geom = epi_rel_expected * angle_v12_vexp

        self._metrics = {
            # raw distances from WT
            "dist_WT_M1": a1,
            "dist_WT_M2": a2,
            "dist_WT_M12_obs": a12,
            "dist_WT_M12_exp": a12_exp,
            "dist_M1_M2": a_1to2,

            # directionality vs singles
            "cos(v12,v1)": c1,
            "cos(v12,v2)": c2,
            "same_direction": float(same_direction),

            # residual and normalized scores
            "residual_norm": residual_norm,
            "epi_rel_expected": epi_rel_expected,
            "epi_rel_singles": epi_rel_singles,

            # geometry of obs vs expected
            "cos_v12_vexp": cos_v12_vexp,
            "angle_v12_vexp": angle_v12_vexp,  # in units of π, 0–1
            "radial_shift": radial_shift,
            "epi_signed_radial": epi_signed_radial,
            "epi_geom": epi_geom,

            # qualitative labels
            "type_relative_WT": type_WT,
            "type_relative_additivity": type_add,
        }
        return self._metrics

    # ----------------- 2D MDS embedding ----------------- #

    def mds_2d(self):
        """
        MDS embedding of WT, M1, M2, M12_obs, M12_exp into 2D.

        Returns:
            coords_dict: {label -> np.array([x,y])}
            D:          (5,5) pairwise distance matrix in full space
        """
        if self._coords is not None and self._D is not None:
            return self._coords, self._D

        WT = self.WT
        M1 = self.M1
        M2 = self.M2
        M12 = self.M12
        M12_exp = WT + (M1 - WT) + (M2 - WT)

        X = torch.stack([WT, M1, M2, M12, M12_exp], 0).cpu().numpy()  # (5,D)

        diff = X[:, None, :] - X[None, :, :]
        D = np.sqrt((diff ** 2).sum(-1) + self.eps)

        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
            n_init=20,
            max_iter=2000,
        )
        coords = mds.fit_transform(D)  # (5,2)

        # recenter so WT is at origin
        coords = coords - coords[0:1, :]

        labels = ["WT", "M1", "M2", "M12_obs", "M12_exp"]
        coords_dict = {lab: coords[i] for i, lab in enumerate(labels)}

        self._coords = coords_dict
        self._D = D
        return coords_dict, D

    # ----------------- plotting ----------------- #

    def plot(
        self,
        figsize=(12, 12),
        show_legend: bool = True,
        show_table: bool = True,
    ):
        """
        Plot WT at origin and arrows to M1, M2, M12_obs, M12_exp,
        with HD distances on the arrows and a small metrics table.
        """
        coords, D = self.mds_2d()
        m = self.metrics()

        # high-D distances from WT (index 0)
        hd_dists = {
            "M1":      float(D[0, 1]),
            "M2":      float(D[0, 2]),
            "M12_obs": float(D[0, 3]),
            "M12_exp": float(D[0, 4]),
        }

        fig, ax = plt.subplots(figsize=figsize)

        colors = {
            "WT": "black",
            "M1": "#DD8452",
            "M2": "#55A868",
            "M12_obs": "#C44E52",
            "M12_exp": "#4C72B0",
        }

        # points + labels
        for name, (x, y) in coords.items():
            if name == "WT":
                ax.scatter(x, y, s=80, color=colors[name], zorder=5)
            else:
                ax.scatter(x, y, s=60, color=colors[name], edgecolor="k", zorder=5)
            ax.text(
                x + 0.03, y + 0.03,
                name,
                fontsize=9,
                color=colors.get(name, "k"),
            )

        # helper to annotate distance on arrow
        def annotate_arrow_distance(start_name, end_name, label_name, color):
            sx, sy = coords[start_name]
            ex, ey = coords[end_name]

            mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0  # midpoint

            dx, dy = ex - sx, ey - sy
            nx, ny = -dy, dx
            norm = np.sqrt(nx**2 + ny**2) + 1e-8
            nx, ny = nx / norm, ny / norm

            d = hd_dists[label_name]
            ax.text(
                mx + 0.04 * nx,
                my + 0.04 * ny,
                f"d={d:.3f}",
                fontsize=9,
                color=color,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                    pad=1.5,
                ),
                zorder=10,
            )

        # arrows from WT + distance labels
        WT_xy = coords["WT"]
        for name in ["M1", "M2", "M12_obs", "M12_exp"]:
            x, y = coords[name]
            ax.arrow(
                WT_xy[0], WT_xy[1],
                x - WT_xy[0], y - WT_xy[1],
                length_includes_head=True,
                head_width=0.03,
                head_length=0.05,
                linewidth=2,
                alpha=0.9,
                color=colors[name],
            )
            annotate_arrow_distance("WT", name, name, colors[name])

        # axes styling
        ax.axhline(0, color="0.85", linewidth=0.8)
        ax.axvline(0, color="0.85", linewidth=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0.2)

        if show_legend:
            handles = [
                plt.Line2D([0], [0], color=colors["M1"], lw=2, label="M1"),
                plt.Line2D([0], [0], color=colors["M2"], lw=2, label="M2"),
                plt.Line2D([0], [0], color=colors["M12_obs"], lw=2, label="M12_obs"),
                plt.Line2D([0], [0], color=colors["M12_exp"], lw=2, label="M12_exp"),
            ]
            ax.legend(handles=handles, loc="best", frameon=False, fontsize=9)

        # small metrics table
        if show_table:
            rows = []
            for key, nice in [
                ("residual_norm",     "‖ε‖"),
                ("epi_rel_expected",  "ε_rel(exp)"),
                ("epi_rel_singles",   "ε_rel(singles)"),
                ("angle_v12_vexp",    "angle(obs,exp)"),
                ("radial_shift",      "radial_shift"),
                ("epi_signed_radial", "ε_signed_rad"),
                ("epi_geom",          "ε_geom"),
            ]:
                if key in m:
                    try:
                        rows.append([nice, f"{float(m[key]):.3f}"])
                    except (TypeError, ValueError):
                        pass

            if rows:
                table = ax.table(
                    cellText=rows,
                    colLabels=["metric", "value"],
                    loc="lower left",
                    colLoc="center",
                    cellLoc="center",
                    bbox=[0.02, 0.02, 0.45, 0.35],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                for (_, _), cell in table.get_celld().items():
                    cell.set_linewidth(0.3)

        plt.tight_layout()
        plt.show()

