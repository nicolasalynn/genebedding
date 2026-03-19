"""
Figure: Selection pressure shapes epistatic interaction structure.

Three distance-matched comparison sets (3,547 pairs each, identical distance distributions):
  Set 1: TCGA high-selection (somatic doubles, cond_prob >= 0.80)
  Set 2: 1kGP matched doubles (same anchor mutation, germline partner)
  Set 3: TCGA low-selection (somatic doubles, cond_prob 0.20-0.50)

All comparisons are WITHIN distance bins — no residualization artifacts.
SNV-only pairs. All chromosomes.

Key questions:
  1. Do high-selection TCGA pairs show more epistasis than low-selection?
  2. Do somatic pairs (TCGA) show more epistasis than germline pairs (1kGP)?
  3. Do individual mutation effects differ, or only the interaction structure?
  4. Is this consistent across models?
"""

# ---------------------------------------------------------------------------
# Cell 1: Setup & data loading
# ---------------------------------------------------------------------------
import warnings; warnings.filterwarnings("ignore")
import sys, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu
from matplotlib.lines import Line2D

ROOT = Path.cwd()
for _ in range(5):
    if (ROOT / "notebooks" / "paper_data_config.py").exists():
        break
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from paper_data_config import EPISTASIS_PAPER_ROOT, embeddings_dir

OUTPUT_BASE = embeddings_dir()
ANNOT_DIR = EPISTASIS_PAPER_ROOT / "data" / "annotations"
DATA_DIR = ROOT / "notebooks" / "dirty" / "data"
FIG_DIR = EPISTASIS_PAPER_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Cancer gene sets
oncokb = pd.read_csv(ANNOT_DIR / "cancerGeneList.tsv", sep="\t")
census = pd.read_csv(ANNOT_DIR / "census_all_genes.csv")
ONCOGENES = (
    set(oncokb[oncokb["Gene Type"].isin(["ONCOGENE", "ONCOGENE_AND_TSG"])]["Hugo Symbol"])
    | set(census[census["Role in Cancer"].str.contains("oncogene", case=False, na=False)]["Gene Symbol"].str.strip())
)
TSGS = (
    set(oncokb[oncokb["Gene Type"].isin(["TSG", "ONCOGENE_AND_TSG"])]["Hugo Symbol"])
    | set(census[census["Role in Cancer"].str.contains("TSG", case=False, na=False)]["Gene Symbol"].str.strip())
)
CANCER_GENES = ONCOGENES | TSGS

# Load distance-matched metadata
set1_meta = pd.read_csv(DATA_DIR / "set1_matched_v2.tsv", sep="\t")
set2_meta = pd.read_csv(DATA_DIR / "set2_matched_v2.tsv", sep="\t")
set3_meta = pd.read_csv(DATA_DIR / "set3_matched_v2.tsv", sep="\t")

set1_ids = set(set1_meta["epistasis_id"])
set2_ids = set(set2_meta["epistasis_id"])
set3_ids = set(set3_meta["epistasis_id"])

# Source labels and colors
SOURCES = {
    "tcga_high_selection": ("TCGA high-selection\n(cond_prob ≥ 0.80)", "#d62728"),
    "tcga_low_selection": ("TCGA low-selection\n(cond_prob 0.20-0.50)", "#ff7f0e"),
    "okgp_matched_doubles": ("1kGP germline\n(matched anchor)", "#1f77b4"),
}

# Distance bins for within-bin comparison
DIST_BINS = [(1, 3), (3, 6), (6, 11), (11, 21), (21, 51), (51, 101), (101, 201), (201, 501)]
DIST_LABELS = ["1-2", "3-5", "6-10", "11-20", "21-50", "51-100", "101-200", "201-500"]

# Models to analyze
MODELS_TO_TEST = ["borzoi", "nt500_multi", "nt2500_okgp", "rinalmo"]

print(f"Matched sets: Set1={len(set1_ids)}, Set2={len(set2_ids)}, Set3={len(set3_ids)}")
print(f"Models: {MODELS_TO_TEST}")


# ---------------------------------------------------------------------------
# Cell 2: Load embedding metrics from DBs
# ---------------------------------------------------------------------------
from genebeddings import VariantEmbeddingDB
from genebeddings.epistasis_features import _list_epistasis_ids_from_db, _load_residual_from_db

def load_metrics_from_db(model_key, source_name, epi_ids_filter=None):
    """Load WT, M1, M2, M12 embeddings and compute metrics directly from DB."""
    db_path = OUTPUT_BASE / source_name / f"{model_key}.db"
    if not db_path.exists():
        return pd.DataFrame()

    db = VariantEmbeddingDB(str(db_path))
    all_epi_ids = _list_epistasis_ids_from_db(db)

    if epi_ids_filter is not None:
        all_epi_ids = [eid for eid in all_epi_ids if eid in epi_ids_filter]

    rows = []
    for eid in all_epi_ids:
        try:
            z_wt = db.load(f"{eid}|WT", as_torch=False)
            z_m1 = db.load(f"{eid}|M1", as_torch=False)
            z_m2 = db.load(f"{eid}|M2", as_torch=False)
            z_m12 = db.load(f"{eid}|M12", as_torch=False)
        except KeyError:
            continue

        z_wt, z_m1, z_m2, z_m12 = [x.astype(np.float64) for x in [z_wt, z_m1, z_m2, z_m12]]

        v1 = z_m1 - z_wt
        v2 = z_m2 - z_wt
        v12_obs = z_m12 - z_wt
        v12_exp = v1 + v2
        residual = v12_obs - v12_exp

        eps = 1e-20
        len_m1 = np.linalg.norm(v1)
        len_m2 = np.linalg.norm(v2)
        len_m12 = np.linalg.norm(v12_obs)
        len_exp = np.linalg.norm(v12_exp)
        r_raw = np.linalg.norm(residual)
        r_singles = r_raw / (np.sqrt(len_m1**2 + len_m2**2) + eps)
        mr = len_m12 / (len_exp + eps)
        log_mr = np.log((len_m12 + eps) / (len_exp + eps))

        gene = eid.split(":")[0]
        parts = eid.split("|")
        p1 = int(parts[0].split(":")[2])
        p2 = int(parts[1].split(":")[2])
        distance = abs(p2 - p1)

        rows.append({
            "epistasis_id": eid,
            "source": source_name,
            "gene": gene,
            "distance": distance,
            "len_WT_M1": len_m1,
            "len_WT_M2": len_m2,
            "len_WT_M12": len_m12,
            "max_single": max(len_m1, len_m2),
            "epi_R_singles": r_singles,
            "log_magnitude_ratio": log_mr,
            "magnitude_ratio": mr,
        })

    db.close()
    return pd.DataFrame(rows)

# Load all data
model_data = {}
for mk in MODELS_TO_TEST:
    dfs = []
    for source, (_, _) in SOURCES.items():
        # Use matched IDs for the right source
        if source == "tcga_high_selection":
            id_filter = set1_ids
        elif source == "okgp_matched_doubles":
            id_filter = set2_ids
        elif source == "tcga_low_selection":
            id_filter = set3_ids
        else:
            id_filter = None

        df = load_metrics_from_db(mk, source, epi_ids_filter=id_filter)
        if len(df) > 0:
            dfs.append(df)
            print(f"  {mk}/{source}: {len(df)} pairs loaded")

    if dfs:
        model_data[mk] = pd.concat(dfs, ignore_index=True)

print(f"\nLoaded {len(model_data)} models")
for mk, df in model_data.items():
    print(f"  {mk}: {df['source'].value_counts().to_dict()}")


# ---------------------------------------------------------------------------
# Cell 3: Within-bin comparison — the clean test
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("WITHIN-BIN COMPARISON: all 3 sets, SNV only, no residualization")
print("=" * 110)

comparison_rows = []

for mk, df in model_data.items():
    print(f"\n--- {mk} ---")
    print(f"{'bin':>10s} | {'n_hi':>5s} {'n_lo':>5s} {'n_germ':>6s} | "
          f"{'|logMR| hi':>10s} {'lo':>10s} {'germ':>10s} | "
          f"{'hi_v_lo':>8s} {'hi_v_germ':>10s} | "
          f"{'maxS hi':>8s} {'lo':>8s} {'germ':>8s}")

    for (lo, hi), label in zip(DIST_BINS, DIST_LABELS):
        hi_sel = df[(df["source"] == "tcga_high_selection") &
                     (df["distance"] >= lo) & (df["distance"] < hi)]
        lo_sel = df[(df["source"] == "tcga_low_selection") &
                     (df["distance"] >= lo) & (df["distance"] < hi)]
        germ = df[(df["source"] == "okgp_matched_doubles") &
                   (df["distance"] >= lo) & (df["distance"] < hi)]

        if len(hi_sel) < 10 or len(lo_sel) < 10 or len(germ) < 10:
            continue

        # Epistasis: |log_MR|
        hi_mr = hi_sel["log_magnitude_ratio"].abs()
        lo_mr = lo_sel["log_magnitude_ratio"].abs()
        germ_mr = germ["log_magnitude_ratio"].abs()

        _, p_hi_lo = mannwhitneyu(hi_mr, lo_mr, alternative="greater")
        _, p_hi_germ = mannwhitneyu(hi_mr, germ_mr, alternative="greater")

        # Singles: max_single
        hi_s = hi_sel["max_single"]
        lo_s = lo_sel["max_single"]
        germ_s = germ["max_single"]

        sig_hl = "***" if p_hi_lo < 0.001 else "**" if p_hi_lo < 0.01 else "*" if p_hi_lo < 0.05 else "n.s."
        sig_hg = "***" if p_hi_germ < 0.001 else "**" if p_hi_germ < 0.01 else "*" if p_hi_germ < 0.05 else "n.s."

        print(f"{label:>10s} | {len(hi_sel):5d} {len(lo_sel):5d} {len(germ):6d} | "
              f"{hi_mr.median():10.4f} {lo_mr.median():10.4f} {germ_mr.median():10.4f} | "
              f"{sig_hl:>8s} {sig_hg:>10s} | "
              f"{hi_s.median():8.4f} {lo_s.median():8.4f} {germ_s.median():8.4f}")

        comparison_rows.append({
            "model": mk, "bin": label, "lo_bp": lo, "hi_bp": hi,
            "n_hi": len(hi_sel), "n_lo": len(lo_sel), "n_germ": len(germ),
            "mr_hi": hi_mr.median(), "mr_lo": lo_mr.median(), "mr_germ": germ_mr.median(),
            "p_hi_vs_lo": p_hi_lo, "p_hi_vs_germ": p_hi_germ,
            "single_hi": hi_s.median(), "single_lo": lo_s.median(), "single_germ": germ_s.median(),
        })

df_comp = pd.DataFrame(comparison_rows)


# ---------------------------------------------------------------------------
# Cell 4: Aggregate across distance bins — pooled within-bin test
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("POOLED WITHIN-BIN TEST (aggregate per model)")
print("=" * 110)

pooled_rows = []
for mk, df in model_data.items():
    # For each pair, subtract its distance-bin mean computed from ALL three sets combined
    df_copy = df.copy()
    df_copy["dist_bin"] = pd.cut(df_copy["distance"], bins=[b[0] for b in DIST_BINS] + [501],
                                  labels=DIST_LABELS, right=False)

    for metric in ["log_magnitude_ratio", "max_single"]:
        # Compute within-bin residuals using ALL data (all 3 sources combined)
        residuals = df_copy[metric].copy()
        for label in DIST_LABELS:
            mask = df_copy["dist_bin"] == label
            if mask.sum() > 0:
                residuals[mask] = df_copy.loc[mask, metric] - df_copy.loc[mask, metric].mean()
        df_copy[f"resid_{metric}"] = residuals

    hi = df_copy[df_copy["source"] == "tcga_high_selection"]
    lo = df_copy[df_copy["source"] == "tcga_low_selection"]
    germ = df_copy[df_copy["source"] == "okgp_matched_doubles"]

    for metric_name, col in [("epistasis |log_MR|", "resid_log_magnitude_ratio"),
                               ("single max_single", "resid_max_single")]:
        hi_vals = hi[col].abs().dropna()
        lo_vals = lo[col].abs().dropna()
        germ_vals = germ[col].abs().dropna()

        if len(hi_vals) < 30 or len(lo_vals) < 30 or len(germ_vals) < 30:
            continue

        _, p_hl = mannwhitneyu(hi_vals, lo_vals, alternative="greater")
        _, p_hg = mannwhitneyu(hi_vals, germ_vals, alternative="greater")
        _, p_lg = mannwhitneyu(lo_vals, germ_vals, alternative="greater")

        r_hl = hi_vals.median() / (lo_vals.median() + 1e-20)
        r_hg = hi_vals.median() / (germ_vals.median() + 1e-20)

        pooled_rows.append({
            "model": mk, "metric": metric_name,
            "hi_median": hi_vals.median(), "lo_median": lo_vals.median(),
            "germ_median": germ_vals.median(),
            "ratio_hi_lo": r_hl, "p_hi_lo": p_hl,
            "ratio_hi_germ": r_hg, "p_hi_germ": p_hg,
            "p_lo_germ": p_lg,
            "n_hi": len(hi_vals), "n_lo": len(lo_vals), "n_germ": len(germ_vals),
        })

        sig_hl = "***" if p_hl < 0.001 else "**" if p_hl < 0.01 else "*" if p_hl < 0.05 else "n.s."
        sig_hg = "***" if p_hg < 0.001 else "**" if p_hg < 0.01 else "*" if p_hg < 0.05 else "n.s."

        print(f"  {mk:15s} {metric_name:25s}: hi/lo={r_hl:.3f} ({sig_hl:>4s}), "
              f"hi/germ={r_hg:.3f} ({sig_hg:>4s})")

df_pooled = pd.DataFrame(pooled_rows)


# ---------------------------------------------------------------------------
# Cell 5: Figure — 4 panel
# ---------------------------------------------------------------------------
mm = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

def setup_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

n_models = len(model_data)
if n_models == 0:
    print("No model data loaded — run on cluster with embedding DBs available.")
else:
    fig = plt.figure(figsize=(183 * mm, 160 * mm))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35,
                            left=0.10, right=0.95, top=0.93, bottom=0.08)

    source_colors = {
        "tcga_high_selection": "#d62728",
        "tcga_low_selection": "#ff7f0e",
        "okgp_matched_doubles": "#1f77b4",
    }
    source_labels = {
        "tcga_high_selection": "TCGA high-sel",
        "tcga_low_selection": "TCGA low-sel",
        "okgp_matched_doubles": "1kGP germline",
    }

    # ── Panel A: |log_MR| by distance bin (one representative model) ──
    ax_a = fig.add_subplot(gs[0, 0])
    setup_ax(ax_a)

    rep_model = MODELS_TO_TEST[0] if MODELS_TO_TEST[0] in model_data else list(model_data.keys())[0]
    df_rep = model_data[rep_model]

    for source, color in source_colors.items():
        medians = []
        centers = []
        for (lo_bp, hi_bp), label in zip(DIST_BINS, DIST_LABELS):
            sub = df_rep[(df_rep["source"] == source) &
                          (df_rep["distance"] >= lo_bp) & (df_rep["distance"] < hi_bp)]
            if len(sub) >= 10:
                medians.append(sub["log_magnitude_ratio"].abs().median())
                centers.append((lo_bp + hi_bp) / 2)
        if medians:
            ax_a.plot(centers, medians, "o-", color=color, linewidth=1.5, markersize=4,
                      label=source_labels[source], alpha=0.85)

    ax_a.set_xlabel("Distance (bp)", fontsize=8)
    ax_a.set_ylabel("Median |log(MR)|", fontsize=8)
    ax_a.set_title(f"Epistasis by distance ({rep_model})", fontsize=9, fontweight="bold")
    ax_a.legend(fontsize=6, loc="upper right")
    ax_a.set_xscale("log")
    ax_a.grid(True, alpha=0.2)
    ax_a.text(-0.12, 1.05, "a", transform=ax_a.transAxes, fontsize=12, fontweight="bold")

    # ── Panel B: Pooled within-bin comparison (all models) ──
    ax_b = fig.add_subplot(gs[0, 1])
    setup_ax(ax_b)

    epi_pooled = df_pooled[df_pooled["metric"] == "epistasis |log_MR|"]
    if len(epi_pooled) > 0:
        models_sorted = epi_pooled.sort_values("ratio_hi_lo", ascending=True)["model"].tolist()
        y_b = np.arange(len(models_sorted))

        for i, mk in enumerate(models_sorted):
            row = epi_pooled[epi_pooled["model"] == mk].iloc[0]
            # Bar showing hi/lo ratio
            color = "#d62728" if row["p_hi_lo"] < 0.05 else "#CCCCCC"
            ax_b.barh(i, row["ratio_hi_lo"], color=color, alpha=0.85, edgecolor="none")
            # Mark hi/germ ratio
            ax_b.plot(row["ratio_hi_germ"], i, "D", color="#1f77b4", markersize=5, zorder=5)

        ax_b.axvline(1.0, color="#999999", linestyle="--", linewidth=0.5)
        ax_b.set_yticks(y_b)
        ax_b.set_yticklabels(models_sorted, fontsize=7)
        ax_b.set_xlabel("Ratio of median |resid log(MR)|", fontsize=8)
        ax_b.set_title("High-sel / Low-sel (bars)\nHigh-sel / Germline (diamonds)",
                       fontsize=9, fontweight="bold")
        ax_b.grid(True, alpha=0.2, axis="x")
    ax_b.text(-0.15, 1.05, "b", transform=ax_b.transAxes, fontsize=12, fontweight="bold")

    # ── Panel C: Singles vs epistasis comparison ──
    ax_c = fig.add_subplot(gs[1, 0])
    setup_ax(ax_c)

    if len(df_pooled) > 0:
        metrics = df_pooled["metric"].unique()
        x_c = np.arange(len(model_data))
        bar_w = 0.35
        metric_colors = {"epistasis |log_MR|": "#d62728", "single max_single": "#999999"}

        for j, metric in enumerate(metrics):
            sub = df_pooled[df_pooled["metric"] == metric]
            vals = [sub[sub["model"] == mk]["ratio_hi_lo"].values[0]
                    if mk in sub["model"].values else 1.0
                    for mk in model_data.keys()]
            sigs = [sub[sub["model"] == mk]["p_hi_lo"].values[0] < 0.05
                    if mk in sub["model"].values else False
                    for mk in model_data.keys()]
            alpha = [0.85 if s else 0.3 for s in sigs]
            for i, (v, a) in enumerate(zip(vals, alpha)):
                ax_c.bar(x_c[i] + j * bar_w - bar_w / 2, v, bar_w,
                         color=metric_colors.get(metric, "gray"), alpha=a, edgecolor="none")

        ax_c.axhline(1.0, color="#999999", linestyle="--", linewidth=0.5)
        ax_c.set_xticks(x_c)
        ax_c.set_xticklabels(list(model_data.keys()), fontsize=7, rotation=45, ha="right")
        ax_c.set_ylabel("High-sel / Low-sel ratio", fontsize=8)
        ax_c.set_title("Epistasis (red) vs singles (gray)\n(faded = n.s.)",
                       fontsize=9, fontweight="bold")
        ax_c.legend(handles=[
            Line2D([0], [0], color="#d62728", linewidth=6, label="Epistasis |log(MR)|"),
            Line2D([0], [0], color="#999999", linewidth=6, label="Single-variant max"),
        ], fontsize=6, loc="upper left")
        ax_c.grid(True, alpha=0.2, axis="y")
    ax_c.text(-0.12, 1.05, "c", transform=ax_c.transAxes, fontsize=12, fontweight="bold")

    # ── Panel D: Distribution violin/box for representative model ──
    ax_d = fig.add_subplot(gs[1, 1])
    setup_ax(ax_d)

    if rep_model in model_data:
        df_rep = model_data[rep_model]
        # Within-bin residualize using combined pool
        df_rep = df_rep.copy()
        df_rep["dist_bin"] = pd.cut(df_rep["distance"],
                                     bins=[b[0] for b in DIST_BINS] + [501],
                                     labels=DIST_LABELS, right=False)
        resid = df_rep["log_magnitude_ratio"].copy()
        for label in DIST_LABELS:
            mask = df_rep["dist_bin"] == label
            if mask.sum() > 0:
                resid[mask] = df_rep.loc[mask, "log_magnitude_ratio"] - df_rep.loc[mask, "log_magnitude_ratio"].mean()
        df_rep["resid_log_mr"] = resid

        plot_data = []
        plot_labels = []
        plot_colors = []
        for source in ["tcga_high_selection", "tcga_low_selection", "okgp_matched_doubles"]:
            vals = df_rep[df_rep["source"] == source]["resid_log_mr"].abs().dropna()
            if len(vals) > 0:
                plot_data.append(vals.values)
                plot_labels.append(source_labels[source])
                plot_colors.append(source_colors[source])

        if plot_data:
            parts = ax_d.violinplot(plot_data, positions=range(len(plot_data)),
                                     showmeans=False, showmedians=True, showextrema=False)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(plot_colors[i])
                pc.set_alpha(0.5)
            parts["cmedians"].set_color("black")

            ax_d.set_xticks(range(len(plot_labels)))
            ax_d.set_xticklabels(plot_labels, fontsize=7)
            ax_d.set_ylabel("|Residualized log(MR)|", fontsize=8)
            ax_d.set_title(f"Epistasis score distribution\n({rep_model})",
                           fontsize=9, fontweight="bold")
            ax_d.grid(True, alpha=0.2, axis="y")
    ax_d.text(-0.12, 1.05, "d", transform=ax_d.transAxes, fontsize=12, fontweight="bold")

    for ext in (".png", ".pdf"):
        fig.savefig(FIG_DIR / f"fig_selection_pressure{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"\nSaved to {FIG_DIR / 'fig_selection_pressure.png'}")


# ---------------------------------------------------------------------------
# Cell 6: Summary statistics for paper
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("PAPER-READY STATISTICS")
print("=" * 110)

if len(df_pooled) > 0:
    epi = df_pooled[df_pooled["metric"] == "epistasis |log_MR|"]
    sing = df_pooled[df_pooled["metric"] == "single max_single"]

    print(f"\n1. Epistasis (|log_MR|) high-sel vs low-sel:")
    for _, r in epi.iterrows():
        sig = "***" if r["p_hi_lo"] < 0.001 else "**" if r["p_hi_lo"] < 0.01 else "*" if r["p_hi_lo"] < 0.05 else "n.s."
        print(f"   {r['model']:15s}: ratio={r['ratio_hi_lo']:.3f}, p={r['p_hi_lo']:.2e} {sig}")

    print(f"\n2. Singles (max_single) high-sel vs low-sel:")
    for _, r in sing.iterrows():
        sig = "***" if r["p_hi_lo"] < 0.001 else "**" if r["p_hi_lo"] < 0.01 else "*" if r["p_hi_lo"] < 0.05 else "n.s."
        print(f"   {r['model']:15s}: ratio={r['ratio_hi_lo']:.3f}, p={r['p_hi_lo']:.2e} {sig}")

    print(f"\n3. Epistasis high-sel vs germline:")
    for _, r in epi.iterrows():
        sig = "***" if r["p_hi_germ"] < 0.001 else "**" if r["p_hi_germ"] < 0.01 else "*" if r["p_hi_germ"] < 0.05 else "n.s."
        print(f"   {r['model']:15s}: ratio={r['ratio_hi_germ']:.3f}, p={r['p_hi_germ']:.2e} {sig}")

    # Key finding
    epi_mean_ratio = epi["ratio_hi_lo"].mean()
    sing_mean_ratio = sing["ratio_hi_lo"].mean() if len(sing) > 0 else float("nan")
    print(f"\n4. Summary:")
    print(f"   Mean epistasis hi/lo ratio: {epi_mean_ratio:.3f}")
    print(f"   Mean singles hi/lo ratio:   {sing_mean_ratio:.3f}")
    if epi_mean_ratio > sing_mean_ratio:
        print(f"   Epistasis separates high/low selection MORE than singles")
    else:
        print(f"   Singles separate high/low selection MORE than epistasis")
else:
    print("No data — run on cluster with embedding DBs available.")


# ---------------------------------------------------------------------------
# Cell 7: Two-sided within-bin test — no directional assumption
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("TWO-SIDED WITHIN-BIN TEST: are the distributions different at all?")
print("=" * 110)

if len(model_data) > 0:
    for mk, df in model_data.items():
        print(f"\n--- {mk} ---")
        print(f"{'bin':>10s} | {'n_hi':>5s} {'n_lo':>5s} {'n_germ':>6s} | "
              f"{'hi_med':>8s} {'lo_med':>8s} {'germ_med':>9s} | "
              f"{'hi_v_lo':>10s} {'hi_v_germ':>12s} {'lo_v_germ':>12s}")

        for (lo, hi), label in zip(DIST_BINS, DIST_LABELS):
            h = df[(df["source"] == "tcga_high_selection") &
                    (df["distance"] >= lo) & (df["distance"] < hi)]
            l = df[(df["source"] == "tcga_low_selection") &
                    (df["distance"] >= lo) & (df["distance"] < hi)]
            g = df[(df["source"] == "okgp_matched_doubles") &
                    (df["distance"] >= lo) & (df["distance"] < hi)]

            if len(h) < 10 or len(l) < 10 or len(g) < 10:
                continue

            hv = h["log_magnitude_ratio"].abs()
            lv = l["log_magnitude_ratio"].abs()
            gv = g["log_magnitude_ratio"].abs()

            _, p_hl = mannwhitneyu(hv, lv, alternative="two-sided")
            _, p_hg = mannwhitneyu(hv, gv, alternative="two-sided")
            _, p_lg = mannwhitneyu(lv, gv, alternative="two-sided")

            sig_hl = "***" if p_hl < 0.001 else "**" if p_hl < 0.01 else "*" if p_hl < 0.05 else ""
            sig_hg = "***" if p_hg < 0.001 else "**" if p_hg < 0.01 else "*" if p_hg < 0.05 else ""
            sig_lg = "***" if p_lg < 0.001 else "**" if p_lg < 0.01 else "*" if p_lg < 0.05 else ""

            print(f"{label:>10s} | {len(h):5d} {len(l):5d} {len(g):6d} | "
                  f"{hv.median():8.4f} {lv.median():8.4f} {gv.median():9.4f} | "
                  f"{p_hl:9.2e}{sig_hl:>3s} {p_hg:11.2e}{sig_hg:>3s} {p_lg:11.2e}{sig_lg:>3s}")
