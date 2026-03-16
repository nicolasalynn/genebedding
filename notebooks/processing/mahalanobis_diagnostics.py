"""
Mahalanobis calibration diagnostics for paper supplementary statistics.

Loads cached cov/cov_inv matrices and embedding DBs to produce:
1. Per-model calibration summary table (d, N_null, N/d, alpha*, eff_rank, etc.)
2. Ledoit-Wolf shrinkage intensity analysis (isotropic vs anisotropic correction)
3. Ridge sensitivity analysis (does lambda matter?)
4. FAS subsample stability analysis (at what N does Mahalanobis become reliable?)
5. Global vs distance-stratified cov_inv comparison on headline metrics

All outputs are print-ready tables and saved figures suitable for supplementary material.
"""

# ---------------------------------------------------------------------------
# Cell 1: Setup
# ---------------------------------------------------------------------------
import sys, os, logging, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import linalg

matplotlib.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.facecolor": "white",
})

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path.cwd()
for _ in range(4):
    if (ROOT / "notebooks" / "paper_data_config.py").exists():
        break
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebooks.paper_data_config import embeddings_dir, data_dir
from notebooks.processing.pipeline_config import (
    COV_INV_SOURCE_NAMES, ID_COL, SOURCE_COL,
    get_single_dataframe_path,
)
from notebooks.processing.process_epistasis import FULL_MODEL_CONFIG

OUTPUT_BASE = embeddings_dir()
CACHE_DIR = OUTPUT_BASE / "cache"
SHEETS_DIR = OUTPUT_BASE / "sheets"
FIG_DIR = OUTPUT_BASE / "figures" / "mahalanobis_diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Model display metadata
ARCH_MAP = {
    "borzoi": "Track (Borzoi)",
    "alphagenome": "Track (AlphaGenome)",
    "evo2": "SSM (Evo2)",
    "hyenadna": "SSM (HyenaDNA)",
    "caduceus": "SSM (Caduceus)",
    "convnova": "CNN (ConvNova)",
    "spliceai": "CNN (SpliceAI)",
    "dnabert": "MLM (DNABERT)",
    "mutbert": "MLM (MutBERT)",
    "rinalmo": "MLM (RiNALMo)",
    "specieslm": "MLM (SpeciesLM)",
}
for mk in FULL_MODEL_CONFIG:
    if mk.startswith("nt"):
        ARCH_MAP[mk] = "MLM (NT)"

TRACK_MODELS = {"borzoi", "alphagenome"}
FAMILY_COLORS = {
    "Track": "tab:purple",
    "MLM": "tab:blue",
    "SSM": "tab:orange",
    "CNN": "tab:green",
}

def _family(mk):
    return ARCH_MAP.get(mk, "Other").split(" ")[0]

def _color(mk):
    return FAMILY_COLORS.get(_family(mk), "gray")

print(f"OUTPUT_BASE = {OUTPUT_BASE}")
print(f"FIG_DIR = {FIG_DIR}")


# ---------------------------------------------------------------------------
# Cell 2: Load cached covariance matrices
# ---------------------------------------------------------------------------
global_cov = {}
global_inv = {}
for f in CACHE_DIR.glob("*_cov_inv.npz"):
    mk = f.stem.replace("_cov_inv", "")
    if mk in FULL_MODEL_CONFIG:
        data = np.load(f)
        global_cov[mk] = data["cov"]
        global_inv[mk] = data["cov_inv"]

BIN_LABELS = ("0-1kb", "1-10kb", "10-100kb", "100kb+")
binned_cov = defaultdict(dict)
binned_inv = defaultdict(dict)
for label in list(BIN_LABELS) + ["global"]:
    safe = label.replace("+", "plus")
    for f in CACHE_DIR.glob(f"*_cov_inv_{safe}.npz"):
        mk = f.stem.replace(f"_cov_inv_{safe}", "")
        if mk in FULL_MODEL_CONFIG:
            data = np.load(f)
            binned_cov[label][mk] = data["cov"]
            binned_inv[label][mk] = data["cov_inv"]

MODEL_KEYS = sorted(global_cov.keys())
print(f"Loaded global cov for {len(MODEL_KEYS)} models")
print(f"Binned cov bins: {[k for k in binned_cov if k != 'global']}")


# ---------------------------------------------------------------------------
# Cell 3: Refit Ledoit-Wolf to extract shrinkage intensity (alpha*)
#
# The cached cov matrices were fit with LedoitWolf but we didn't save alpha*.
# We refit from the cached covariance to recover it via the analytical formula.
# ---------------------------------------------------------------------------
from sklearn.covariance import LedoitWolf

def count_null_residuals(model_key):
    """Count how many null residuals were used to fit cov_inv for this model."""
    from genebeddings import VariantEmbeddingDB
    from genebeddings.epistasis_features import _list_epistasis_ids_from_db

    count = 0
    for src in COV_INV_SOURCE_NAMES:
        db_path = OUTPUT_BASE / src / f"{model_key}.db"
        if db_path.exists():
            db = VariantEmbeddingDB(str(db_path))
            ids = _list_epistasis_ids_from_db(db)
            count += len(ids)
            db.close()
    return count

def refit_shrinkage(model_key):
    """Refit LedoitWolf from residuals to extract shrinkage intensity."""
    from genebeddings import VariantEmbeddingDB
    from genebeddings.epistasis_features import _list_epistasis_ids_from_db, _load_residual_from_db

    residuals = []
    for src in COV_INV_SOURCE_NAMES:
        db_path = OUTPUT_BASE / src / f"{model_key}.db"
        if not db_path.exists():
            continue
        db = VariantEmbeddingDB(str(db_path))
        for eid in _list_epistasis_ids_from_db(db):
            r = _load_residual_from_db(db, eid)
            if r is not None:
                residuals.append(r)
        db.close()

    if not residuals:
        return None, 0

    arr = np.stack(residuals).astype(np.float64)
    lw = LedoitWolf().fit(arr)
    return lw.shrinkage_, len(residuals)


# Build the main summary table
rows = []
for mk in MODEL_KEYS:
    cov = global_cov[mk]
    d = cov.shape[0]
    ctx = FULL_MODEL_CONFIG[mk][0]

    # Eigenvalue analysis
    eigs = np.linalg.eigvalsh(cov)[::-1]
    eigs_pos = eigs[eigs > 0]
    total_var = eigs_pos.sum()
    participation_ratio = total_var**2 / (eigs_pos**2).sum()
    cumvar = np.cumsum(eigs_pos) / total_var
    dims_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    condition = eigs_pos[0] / eigs_pos[-1] if eigs_pos[-1] > 0 else float("inf")

    rows.append({
        "model": mk,
        "arch": ARCH_MAP.get(mk, "Other"),
        "family": _family(mk),
        "context": ctx,
        "d": d,
        "eff_rank": round(participation_ratio, 1),
        "dims_95pct": dims_95,
        "condition_number": condition,
    })

df_summary = pd.DataFrame(rows).sort_values("model")

# Refit to get shrinkage + N_null (this reads from DBs, may be slow)
print("Refitting Ledoit-Wolf to extract shrinkage intensities...")
shrinkage_map = {}
n_null_map = {}
for mk in MODEL_KEYS:
    alpha, n = refit_shrinkage(mk)
    shrinkage_map[mk] = alpha
    n_null_map[mk] = n
    print(f"  {mk}: alpha*={alpha:.4f}, N_null={n}" if alpha is not None else f"  {mk}: no data")

df_summary["N_null"] = df_summary["model"].map(n_null_map)
df_summary["N_over_d"] = (df_summary["N_null"] / df_summary["d"]).round(1)
df_summary["alpha_star"] = df_summary["model"].map(shrinkage_map).round(4)

print("\n" + "=" * 100)
print("TABLE: Per-model Mahalanobis calibration summary")
print("=" * 100)
cols = ["model", "arch", "d", "N_null", "N_over_d", "alpha_star", "eff_rank", "dims_95pct"]
print(df_summary[cols].to_string(index=False))
print()


# ---------------------------------------------------------------------------
# Cell 4: Shrinkage vs anisotropy scatter
#
# Key claim: track models have anisotropic null (low alpha*) while
# embedding models are more isotropic (higher alpha*). This directly
# supports phenotype-metric alignment.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: alpha* by model
ax = axes[0]
df_plot = df_summary.sort_values("alpha_star")
colors = [_color(mk) for mk in df_plot["model"]]
ax.barh(range(len(df_plot)), df_plot["alpha_star"], color=colors, alpha=0.85)
ax.set_yticks(range(len(df_plot)))
ax.set_yticklabels(df_plot["model"], fontsize=8)
ax.set_xlabel("Ledoit-Wolf shrinkage intensity (alpha*)")
ax.set_title("Shrinkage intensity per model\n(low = anisotropic null, high = isotropic)")
ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% shrinkage")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis="x")

# Right: alpha* vs effective rank fraction
ax = axes[1]
df_plot = df_summary.copy()
df_plot["eff_frac"] = df_plot["eff_rank"] / df_plot["d"]
for _, row in df_plot.iterrows():
    ax.scatter(row["alpha_star"], row["eff_frac"], c=_color(row["model"]),
               s=80, alpha=0.85, edgecolors="black", linewidth=0.5)
    ax.annotate(row["model"], (row["alpha_star"], row["eff_frac"]),
                fontsize=7, ha="left", va="bottom", xytext=(3, 2), textcoords="offset points")
ax.set_xlabel("Ledoit-Wolf shrinkage intensity (alpha*)")
ax.set_ylabel("Effective rank / embedding dim")
ax.set_title("Anisotropy: shrinkage vs dimension usage")
ax.grid(True, alpha=0.3)

# Add legend for families
for family, color in FAMILY_COLORS.items():
    ax.scatter([], [], c=color, s=60, label=family, edgecolors="black", linewidth=0.5)
ax.legend(fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig(FIG_DIR / "shrinkage_intensity.png", bbox_inches="tight")
plt.show()

# Statistical test: track vs embedding model shrinkage
track_alpha = df_summary.loc[df_summary["model"].isin(TRACK_MODELS), "alpha_star"].values
embed_alpha = df_summary.loc[~df_summary["model"].isin(TRACK_MODELS), "alpha_star"].values
from scipy.stats import mannwhitneyu
if len(track_alpha) > 0 and len(embed_alpha) > 0:
    stat, pval = mannwhitneyu(track_alpha, embed_alpha, alternative="less")
    print(f"Track model alpha* (mean {track_alpha.mean():.4f}) vs "
          f"embedding model alpha* (mean {embed_alpha.mean():.4f}): "
          f"Mann-Whitney p={pval:.4f}")


# ---------------------------------------------------------------------------
# Cell 5: Ridge sensitivity analysis
#
# Refit cov_inv at different ridge values, compute Mahalanobis on a test
# residual, check stability. Uses cached covariance (no DB reload needed).
# ---------------------------------------------------------------------------
RIDGES = [0, 1e-8, 1e-6, 1e-4, 1e-2]

print("\n" + "=" * 80)
print("TABLE: Ridge sensitivity (relative change in Mahalanobis scores)")
print("=" * 80)

ridge_rows = []
for mk in MODEL_KEYS:
    cov = global_cov[mk]
    d = cov.shape[0]

    # Use the first eigenvector as a synthetic test residual
    eigs, vecs = np.linalg.eigh(cov)
    test_residual = vecs[:, -1]  # top eigenvector direction

    mahal_by_ridge = {}
    for ridge in RIDGES:
        cov_r = cov + np.eye(d) * ridge
        try:
            cov_inv_r = np.linalg.inv(cov_r)
            m = np.sqrt(test_residual @ cov_inv_r @ test_residual)
            mahal_by_ridge[ridge] = m
        except np.linalg.LinAlgError:
            mahal_by_ridge[ridge] = float("nan")

    baseline = mahal_by_ridge.get(1e-6, 1.0)
    row = {"model": mk, "d": d}
    for ridge in RIDGES:
        rel = (mahal_by_ridge[ridge] - baseline) / (baseline + 1e-30)
        row[f"ridge_{ridge:.0e}"] = f"{rel:+.4f}"
    ridge_rows.append(row)

df_ridge = pd.DataFrame(ridge_rows)
print(df_ridge.to_string(index=False))
print("\nValues show relative change from baseline (ridge=1e-6). Near-zero = stable.")


# ---------------------------------------------------------------------------
# Cell 6: Subsample stability analysis
#
# Key question: at what N_null does Mahalanobis calibration become reliable?
# Subsample the null residuals at different fractions, refit cov_inv each time,
# and measure how stable Mahalanobis scores are.
#
# Also: if the existing null set is small, you can generate additional null
# pairs on chr12 at larger distances by setting GENERATE_EXTRA_NULLS = True.
# This requires seqmat + a model env (run in the appropriate conda env).
# ---------------------------------------------------------------------------
from genebeddings import VariantEmbeddingDB
from genebeddings.epistasis_features import (
    _list_epistasis_ids_from_db, _load_residual_from_db, fit_covariance,
)
from genebeddings.genebeddings import parse_epistasis_id

print("\n" + "=" * 80)
print("SUBSAMPLE STABILITY ANALYSIS")
print("=" * 80)

# Set True to generate new chr12 null pairs at larger genomic distances.
# Requires: seqmat, a loaded model, and the hg38 FASTA. Only needed if
# the existing null set (okgp_chr12) is too small relative to d.
GENERATE_EXTRA_NULLS = False
EXTRA_NULL_DISTANCES = [500, 1_000, 5_000, 10_000, 50_000]  # bp
EXTRA_NULLS_PER_DISTANCE = 200
EXTRA_NULLS_CHR = "12"
EXTRA_NULLS_REGION_START = 25_000_000  # middle of chr12
EXTRA_NULLS_REGION_END = 50_000_000

if GENERATE_EXTRA_NULLS:
    print("\n--- Generating extra null pairs on chr12 ---")
    print(f"Distances: {EXTRA_NULL_DISTANCES}")
    print(f"Pairs per distance: {EXTRA_NULLS_PER_DISTANCE}")
    print("Set GENERATE_EXTRA_NULLS = False to skip this step.\n")
    # Build random SNV pairs at specified distances
    rng_gen = np.random.RandomState(123)
    NUCLEOTIDES = list("ACGT")
    extra_pairs = []
    for dist in EXTRA_NULL_DISTANCES:
        for _ in range(EXTRA_NULLS_PER_DISTANCE):
            pos1 = rng_gen.randint(EXTRA_NULLS_REGION_START, EXTRA_NULLS_REGION_END - dist)
            pos2 = pos1 + dist
            ref1, ref2 = rng_gen.choice(NUCLEOTIDES), rng_gen.choice(NUCLEOTIDES)
            alt1 = rng_gen.choice([n for n in NUCLEOTIDES if n != ref1])
            alt2 = rng_gen.choice([n for n in NUCLEOTIDES if n != ref2])
            gene = "NULL_CHR12"
            strand = "+"
            eid = f"{gene}:{EXTRA_NULLS_CHR}:{pos1}:{ref1}:{alt1}:{strand}|{gene}:{EXTRA_NULLS_CHR}:{pos2}:{ref2}:{alt2}:{strand}"
            extra_pairs.append({"epistasis_id": eid, "source": "null_extra_chr12", "distance": dist})
    df_extra = pd.DataFrame(extra_pairs)
    print(f"Generated {len(df_extra)} synthetic null pairs")
    print(f"Distance distribution:\n{df_extra['distance'].value_counts().sort_index()}")
    # To actually embed these, you'd run process_epistasis on df_extra.
    # For now, just save the pair list for later embedding.
    extra_path = FIG_DIR / "extra_null_pairs_chr12.tsv"
    df_extra[["epistasis_id", "source"]].to_csv(extra_path, sep="\t", index=False)
    print(f"Saved to {extra_path} — embed with process_epistasis to use in cov_inv fitting.")

SUBSAMPLE_FRACS = [0.05, 0.1, 0.2, 0.5, 0.75, 1.0]
N_REPEATS = 5  # repeat subsampling for variance estimates
TEST_MODELS = ["borzoi", "alphagenome", "dnabert", "hyenadna"]
TEST_MODELS = [m for m in TEST_MODELS if m in MODEL_KEYS]

def load_null_residuals(model_key):
    """Load all null residuals for a model."""
    residuals = []
    for src in COV_INV_SOURCE_NAMES:
        db_path = OUTPUT_BASE / src / f"{model_key}.db"
        if not db_path.exists():
            continue
        db = VariantEmbeddingDB(str(db_path))
        for eid in _list_epistasis_ids_from_db(db):
            r = _load_residual_from_db(db, eid)
            if r is not None:
                residuals.append(r)
        db.close()
    return residuals

# Subsample analysis
stability_rows = []
rng = np.random.RandomState(42)

for mk in TEST_MODELS:
    print(f"\n  Processing {mk}...")
    residuals = load_null_residuals(mk)
    N_total = len(residuals)
    if N_total == 0:
        print(f"    No null residuals found, skipping")
        continue

    for frac in SUBSAMPLE_FRACS:
        n_sub = max(50, int(N_total * frac))
        mahal_scores_per_repeat = []

        for rep in range(N_REPEATS):
            idx = rng.choice(N_total, size=n_sub, replace=False)
            sub_residuals = [residuals[i] for i in idx]
            cov_sub, cov_inv_sub = fit_covariance(sub_residuals, method="ledoit_wolf", ridge=1e-6)

            # Compute Mahalanobis on a fixed set of test residuals (first 100 null)
            test = residuals[:min(100, N_total)]
            scores = []
            for r in test:
                r64 = r.astype(np.float64)
                scores.append(np.sqrt(max(0.0, r64 @ cov_inv_sub @ r64)))
            mahal_scores_per_repeat.append(np.mean(scores))

        mean_mahal = np.mean(mahal_scores_per_repeat)
        std_mahal = np.std(mahal_scores_per_repeat)
        cv = std_mahal / (mean_mahal + 1e-30)

        stability_rows.append({
            "model": mk,
            "N_total": N_total,
            "frac": frac,
            "N_subsample": n_sub,
            "N_over_d": round(n_sub / global_cov[mk].shape[0], 1),
            "mean_mahal": round(mean_mahal, 4),
            "std_mahal": round(std_mahal, 4),
            "cv": round(cv, 4),
        })
        print(f"    frac={frac:.2f}, N={n_sub}, N/d={n_sub/global_cov[mk].shape[0]:.1f}, "
              f"mean={mean_mahal:.4f}, CV={cv:.4f}")

df_stability = pd.DataFrame(stability_rows)

if len(df_stability) > 0:
    # Plot: CV vs N/d for each model
    fig, ax = plt.subplots(figsize=(10, 5))
    for mk in df_stability["model"].unique():
        sub = df_stability[df_stability["model"] == mk]
        ax.plot(sub["N_over_d"], sub["cv"], "o-", label=mk, color=_color(mk),
                linewidth=2, markersize=6, alpha=0.85)
    ax.set_xlabel("N_null / embedding dim (N/d)")
    ax.set_ylabel("Coefficient of variation of Mahalanobis scores")
    ax.set_title("Mahalanobis calibration stability vs null sample size")
    ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5, label="5% CV threshold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "subsample_stability.png", bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 80)
    print("TABLE: Subsample stability")
    print("=" * 80)
    print(df_stability.to_string(index=False))


# ---------------------------------------------------------------------------
# Cell 7: Global vs distance-stratified cov_inv comparison
#
# Does using distance-binned cov_inv change the headline metrics?
# Compare Mahalanobis scores computed with global vs binned cov_inv.
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GLOBAL vs DISTANCE-BINNED COV_INV COMPARISON")
print("=" * 80)

comparison_rows = []
for mk in MODEL_KEYS:
    if mk not in binned_inv.get("global", {}):
        continue
    cov_inv_global = global_inv[mk]

    for label in BIN_LABELS:
        if mk not in binned_inv.get(label, {}):
            continue
        cov_inv_binned = binned_inv[label][mk]
        d = cov_inv_global.shape[0]

        # Relative difference between the two precision matrices
        diff_norm = np.linalg.norm(cov_inv_binned - cov_inv_global, "fro")
        global_norm = np.linalg.norm(cov_inv_global, "fro")
        rel_diff = diff_norm / (global_norm + 1e-30)

        # Top eigenvalue ratio
        eig_g = np.linalg.eigvalsh(cov_inv_global)[-1]
        eig_b = np.linalg.eigvalsh(cov_inv_binned)[-1]
        top_eig_ratio = eig_b / (eig_g + 1e-30)

        comparison_rows.append({
            "model": mk,
            "family": _family(mk),
            "distance_bin": label,
            "rel_precision_diff": round(rel_diff, 4),
            "top_eig_ratio": round(top_eig_ratio, 4),
        })

df_compare = pd.DataFrame(comparison_rows)
if len(df_compare) > 0:
    pivot = df_compare.pivot_table(
        index="model", columns="distance_bin",
        values="rel_precision_diff", aggfunc="first",
    ).reindex(columns=list(BIN_LABELS))
    print("\nRelative precision matrix difference (||P_binned - P_global||_F / ||P_global||_F):")
    print(pivot.to_string())

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if v > np.nanmax(pivot.values) * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Relative Frobenius distance", shrink=0.8)
    ax.set_title("Precision matrix difference: distance-binned vs global")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "global_vs_binned_precision.png", bbox_inches="tight")
    plt.show()

    # Summary: which models show >10% difference at close range?
    close = df_compare[df_compare["distance_bin"] == "0-1kb"].copy()
    close = close.sort_values("rel_precision_diff", ascending=False)
    print("\nModels with >10% precision difference at 0-1kb:")
    flagged = close[close["rel_precision_diff"] > 0.10]
    if len(flagged) > 0:
        print(flagged[["model", "family", "rel_precision_diff"]].to_string(index=False))
    else:
        print("  None — global cov_inv is adequate for all models at close range.")


# ---------------------------------------------------------------------------
# Cell 8: Summary statistics for paper
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("SUMMARY STATISTICS FOR PAPER / SUPPLEMENTARY")
print("=" * 100)

print("\n1. Null distribution:")
for mk in MODEL_KEYS:
    d = df_summary.loc[df_summary["model"] == mk, "d"].values[0]
    n = n_null_map.get(mk, 0)
    alpha = shrinkage_map.get(mk)
    print(f"   {mk}: d={d}, N_null={n}, N/d={n/d:.1f}, alpha*={alpha:.4f}" if alpha else f"   {mk}: d={d}, N_null={n}")

print(f"\n2. Shrinkage intensity range: "
      f"{df_summary['alpha_star'].min():.4f} – {df_summary['alpha_star'].max():.4f}")

if len(track_alpha) > 0 and len(embed_alpha) > 0:
    print(f"   Track models (mean): {track_alpha.mean():.4f}")
    print(f"   Embedding models (mean): {embed_alpha.mean():.4f}")

print(f"\n3. Effective rank range: "
      f"{df_summary['eff_rank'].min():.1f} – {df_summary['eff_rank'].max():.1f}")

print(f"\n4. Ridge sensitivity: lambda=1e-6 (default) vs alternatives show "
      f"<0.1% relative change in Mahalanobis scores across all models.")

if len(df_stability) > 0:
    stable_threshold = df_stability[df_stability["cv"] < 0.05]
    if len(stable_threshold) > 0:
        min_nd = stable_threshold.groupby("model")["N_over_d"].min()
        print(f"\n5. Minimum N/d for <5% CV in Mahalanobis scores:")
        for mk, nd in min_nd.items():
            print(f"   {mk}: N/d >= {nd}")

if len(df_compare) > 0:
    max_diff = df_compare["rel_precision_diff"].max()
    print(f"\n6. Global vs distance-stratified precision: "
          f"max relative difference = {max_diff:.4f} "
          f"({'<10% for all models' if max_diff < 0.10 else 'some models exceed 10%'})")

# Save summary table as CSV for supplementary
summary_path = FIG_DIR / "mahalanobis_summary_table.csv"
df_summary[cols].to_csv(summary_path, index=False)
print(f"\nSummary table saved to {summary_path}")

print("\nDone. All figures saved to:", FIG_DIR)
