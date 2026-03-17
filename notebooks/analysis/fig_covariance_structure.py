"""
Figure: Embedding space geometry explains phenotype-metric alignment.

Publication-quality figure showing:
(a) Null effective dimensionality by model (participation ratio)
(b) Signal-null subspace overlap: FAS vs eQTL vs TCGA
    - Track models: eQTL aligned with null, FAS orthogonal
    - Self-supervised: both similar
(c) Single-variant vs epistasis: TCGA vs 1kGP separation

Loads precomputed covariance matrices from {embeddings_dir}/signal_covariance/
and null cov from {embeddings_dir}/cache/.
"""

# ---------------------------------------------------------------------------
# Cell 1: Load precomputed covariance matrices
# ---------------------------------------------------------------------------
import sys, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path.cwd()
for _ in range(5):
    if (ROOT / "notebooks" / "paper_data_config.py").exists():
        break
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from paper_data_config import EPISTASIS_PAPER_ROOT, embeddings_dir
from notebooks.processing.process_epistasis import FULL_MODEL_CONFIG

OUTPUT_BASE = embeddings_dir()
CACHE_DIR = OUTPUT_BASE / "cache"
COV_SAVE_DIR = OUTPUT_BASE / "signal_covariance"
FIG_DIR = EPISTASIS_PAPER_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Model display names
MODELS = {
    "alphagenome": "AlphaGenome", "borzoi": "Borzoi", "caduceus": "Caduceus",
    "convnova": "ConvNova", "dnabert": "DNABERT", "evo2": "Evo2",
    "hyenadna": "HyenaDNA", "mutbert": "MutBERT",
    "nt50_3mer": "NT-50M-3mer", "nt50_multi": "NT-50M",
    "nt100_multi": "NT-100M", "nt250_multi": "NT-250M",
    "nt500_multi": "NT-500M", "nt500_ref": "NT-500M-ref",
    "nt2500_multi": "NT-2.5B", "nt2500_okgp": "NT-2.5B-1kGP",
    "rinalmo": "RiNALMo", "specieslm": "SpeciesLM",
}
TRACK_MODELS = {"alphagenome", "borzoi"}

# Colors
COL_TRACK = "#9467bd"    # purple
COL_MLM = "#1f77b4"      # blue
COL_SSM = "#ff7f0e"      # orange
COL_CNN = "#2ca02c"      # green
GRAY_MID = "#999999"

def model_color(mk):
    if mk in TRACK_MODELS:
        return COL_TRACK
    if mk.startswith("nt") or mk in ("dnabert", "mutbert", "rinalmo", "specieslm"):
        return COL_MLM
    if mk in ("hyenadna", "caduceus", "evo2"):
        return COL_SSM
    if mk in ("convnova", "spliceai"):
        return COL_CNN
    return GRAY_MID

# Load null covariance (from cache)
null_cov = {}
for f in CACHE_DIR.glob("*_cov_inv.npz"):
    mk = f.stem.replace("_cov_inv", "")
    if mk in FULL_MODEL_CONFIG:
        null_cov[mk] = np.load(f)["cov"]

# Load signal covariance (from signal_covariance/)
signal_cov = defaultdict(dict)  # source -> {model -> cov}
for f in COV_SAVE_DIR.glob("*.npz"):
    data = np.load(f)
    source = str(data["source"])
    mk = str(data["model"])
    if "null" not in source:
        signal_cov[source][mk] = data["cov"]

MODEL_KEYS = sorted(null_cov.keys())
print(f"Null cov: {len(null_cov)} models")
for src in sorted(signal_cov.keys()):
    print(f"Signal cov ({src}): {len(signal_cov[src])} models")


# ---------------------------------------------------------------------------
# Cell 2: Compute all metrics for figure
# ---------------------------------------------------------------------------
def participation_ratio(cov):
    eigs = np.linalg.eigvalsh(cov)[::-1]
    eigs_pos = eigs[eigs > 0]
    total = eigs_pos.sum()
    return total**2 / (eigs_pos**2).sum()

def subspace_overlap(cov_a, cov_b, k=50):
    d = cov_a.shape[0]
    k = min(k, d)
    _, vecs_a = np.linalg.eigh(cov_a)
    _, vecs_b = np.linalg.eigh(cov_b)
    V_a = vecs_a[:, -k:]
    V_b = vecs_b[:, -k:]
    cos_matrix = np.abs(V_a.T @ V_b)
    return float(cos_matrix.max(axis=0).mean())

# Panel A data: null effective dimensionality
panel_a = []
for mk in MODEL_KEYS:
    d = null_cov[mk].shape[0]
    pr = participation_ratio(null_cov[mk])
    panel_a.append({"model": mk, "display": MODELS.get(mk, mk), "d": d,
                     "pr": pr, "usage": pr / d})

df_a = pd.DataFrame(panel_a).sort_values("usage")

# Panel B data: subspace overlap (FAS, eQTL, TCGA vs null)
panel_b = []
for source_label, source_key in [("FAS (splicing)", "fas_exon"),
                                   ("eQTL (expression)", "yang_evqtl"),
                                   ("TCGA (cancer)", "tcga_doubles")]:
    src_dict = signal_cov.get(source_key, {})
    for mk in MODEL_KEYS:
        if mk not in src_dict or mk not in null_cov:
            continue
        gs = subspace_overlap(null_cov[mk], src_dict[mk], k=50)
        panel_b.append({"model": mk, "display": MODELS.get(mk, mk),
                         "source": source_label, "overlap": gs,
                         "is_track": mk in TRACK_MODELS})

df_b = pd.DataFrame(panel_b)

print(f"\nPanel A: {len(df_a)} models")
print(f"Panel B: {len(df_b)} rows ({df_b['source'].value_counts().to_dict()})")


# ---------------------------------------------------------------------------
# Cell 3: Generate figure
# ---------------------------------------------------------------------------
mm = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

fig = plt.figure(figsize=(183 * mm, 140 * mm))
gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35,
                        left=0.10, right=0.95, top=0.93, bottom=0.08)

def setup_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

# ── Panel A: Null effective dimensionality ──
ax_a = fig.add_subplot(gs[0, 0])
setup_ax(ax_a)

y = np.arange(len(df_a))
colors = [model_color(mk) for mk in df_a["model"]]
ax_a.barh(y, df_a["usage"], color=colors, alpha=0.85, edgecolor="none")
ax_a.set_yticks(y)
ax_a.set_yticklabels(df_a["display"], fontsize=6)
ax_a.set_xlabel("Effective rank / embedding dim", fontsize=8)
ax_a.set_title("Null covariance anisotropy", fontsize=9, fontweight="bold")
ax_a.axvline(1.0, color=GRAY_MID, linestyle="--", alpha=0.4)
ax_a.grid(True, alpha=0.2, axis="x")

# Legend
from matplotlib.patches import Patch
ax_a.legend(handles=[
    Patch(facecolor=COL_TRACK, label="Track"), Patch(facecolor=COL_MLM, label="MLM"),
    Patch(facecolor=COL_SSM, label="SSM"), Patch(facecolor=COL_CNN, label="CNN"),
], fontsize=5, loc="lower right")
ax_a.text(-0.15, 1.05, "a", transform=ax_a.transAxes, fontsize=12, fontweight="bold")

# ── Panel B: Signal-null subspace overlap ──
ax_b = fig.add_subplot(gs[0, 1])
setup_ax(ax_b)

source_colors = {"FAS (splicing)": "#2ca02c", "eQTL (expression)": "#ff7f0e", "TCGA (cancer)": "#d62728"}
source_offsets = {"FAS (splicing)": -0.22, "eQTL (expression)": 0.0, "TCGA (cancer)": 0.22}

# Sort by FAS overlap
fas_order = df_b[df_b["source"] == "FAS (splicing)"].sort_values("overlap")["model"].tolist()
# Add models not in FAS
for mk in MODEL_KEYS:
    if mk not in fas_order and mk in df_b["model"].values:
        fas_order.append(mk)
y_b = np.arange(len(fas_order))

for source, offset in source_offsets.items():
    src = df_b[df_b["source"] == source]
    vals = [src[src["model"] == mk]["overlap"].values[0] if mk in src["model"].values else 0
            for mk in fas_order]
    ax_b.barh(y_b + offset, vals, height=0.2,
              color=source_colors[source], alpha=0.8, label=source, edgecolor="none")

ax_b.set_yticks(y_b)
ax_b.set_yticklabels([MODELS.get(mk, mk) for mk in fas_order], fontsize=6)
ax_b.set_xlabel("Grassmann similarity with null (k=50)", fontsize=8)
ax_b.set_title("Signal-null subspace alignment", fontsize=9, fontweight="bold")
ax_b.legend(fontsize=5, loc="lower right")
ax_b.grid(True, alpha=0.2, axis="x")
ax_b.text(-0.15, 1.05, "b", transform=ax_b.transAxes, fontsize=12, fontweight="bold")

# ── Panel C: Track model detail — FAS vs eQTL overlap ──
ax_c = fig.add_subplot(gs[1, 0])
setup_ax(ax_c)

# Grouped bar: for each model, show FAS and eQTL overlap side by side
# Focus on models where both FAS and eQTL data exist
models_with_both = []
for mk in MODEL_KEYS:
    has_fas = mk in signal_cov.get("fas_exon", {})
    has_eqtl = mk in signal_cov.get("yang_evqtl", {})
    if has_fas and has_eqtl:
        models_with_both.append(mk)

if models_with_both:
    # Compute delta = eQTL_overlap - FAS_overlap (positive = eQTL more aligned with null)
    delta_rows = []
    for mk in models_with_both:
        fas_gs = subspace_overlap(null_cov[mk], signal_cov["fas_exon"][mk], k=50)
        eqtl_gs = subspace_overlap(null_cov[mk], signal_cov["yang_evqtl"][mk], k=50)
        delta_rows.append({
            "model": mk, "display": MODELS.get(mk, mk),
            "fas": fas_gs, "eqtl": eqtl_gs,
            "delta": eqtl_gs - fas_gs,
            "is_track": mk in TRACK_MODELS,
        })
    df_delta = pd.DataFrame(delta_rows).sort_values("delta")

    y_c = np.arange(len(df_delta))
    bar_colors = ["#d62728" if d > 0 else "#1f77b4" for d in df_delta["delta"]]
    edge_colors = ["black" if t else "none" for t in df_delta["is_track"]]
    ax_c.barh(y_c, df_delta["delta"], color=bar_colors, alpha=0.85,
              edgecolor=edge_colors, linewidth=1)
    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels(df_delta["display"], fontsize=6)
    ax_c.axvline(0, color=GRAY_MID, linewidth=0.5)
    ax_c.set_xlabel("eQTL overlap - FAS overlap\n(>0: eQTL more aligned with null)", fontsize=7)
    ax_c.set_title("Phenotype-metric alignment\n(track models outlined)", fontsize=9, fontweight="bold")
    ax_c.grid(True, alpha=0.2, axis="x")

ax_c.text(-0.15, 1.05, "c", transform=ax_c.transAxes, fontsize=12, fontweight="bold")

# ── Panel D: Effective dimensionality — null vs TCGA vs FAS ──
ax_d = fig.add_subplot(gs[1, 1])
setup_ax(ax_d)

dim_rows = []
for mk in MODEL_KEYS:
    if mk not in null_cov:
        continue
    d = null_cov[mk].shape[0]
    null_pr = participation_ratio(null_cov[mk])
    row = {"model": mk, "display": MODELS.get(mk, mk), "d": d, "Null": null_pr / d}
    for src_label, src_key in [("TCGA", "tcga_doubles"), ("FAS", "fas_exon")]:
        if mk in signal_cov.get(src_key, {}):
            row[src_label] = participation_ratio(signal_cov[src_key][mk]) / d
    dim_rows.append(row)

df_dim = pd.DataFrame(dim_rows)
df_dim = df_dim.sort_values("Null")

y_d = np.arange(len(df_dim))
ax_d.barh(y_d - 0.2, df_dim["Null"], height=0.2, color="#1f77b4", alpha=0.7, label="Null (1kGP)")
if "TCGA" in df_dim.columns:
    ax_d.barh(y_d, df_dim["TCGA"].fillna(0), height=0.2, color="#d62728", alpha=0.7, label="TCGA")
if "FAS" in df_dim.columns:
    ax_d.barh(y_d + 0.2, df_dim["FAS"].fillna(0), height=0.2, color="#2ca02c", alpha=0.7, label="FAS")
ax_d.set_yticks(y_d)
ax_d.set_yticklabels(df_dim["display"], fontsize=6)
ax_d.set_xlabel("Effective rank / embedding dim", fontsize=8)
ax_d.set_title("Signal concentration\n(lower = more concentrated)", fontsize=9, fontweight="bold")
ax_d.legend(fontsize=5, loc="lower right")
ax_d.grid(True, alpha=0.2, axis="x")
ax_d.text(-0.15, 1.05, "d", transform=ax_d.transAxes, fontsize=12, fontweight="bold")

for ext in (".png", ".pdf"):
    fig.savefig(FIG_DIR / f"fig_covariance_structure{ext}", dpi=600,
                bbox_inches="tight", facecolor="white")
plt.show()
print(f"Saved to {FIG_DIR / 'fig_covariance_structure.png'}")
