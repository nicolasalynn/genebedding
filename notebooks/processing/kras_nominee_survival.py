"""
Survival analysis for KRAS-similarity nominated gene pairs.

For each nominated cancer gene pair, compare survival of patients
carrying that specific pair vs patients with one constituent mutation
plus a different second hit in the same gene.

Same controlled design as the paper's NF1/WWC3/CARD11/PABPC1 analysis.

Requires ParseTCGA for clinical data.
"""

import sys, os, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path.cwd()
for _ in range(5):
    if (ROOT / "notebooks" / "paper_data_config.py").exists():
        break
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "scripts" / "data_generation"))

from paper_data_config import EPISTASIS_PAPER_ROOT

FIG_DIR = EPISTASIS_PAPER_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load ParseTCGA
try:
    from parsetcga import TCGAData
    tcga = TCGAData()
    clin = tcga.clinical.survival_prepared()
    surv_patients = set(clin["case_id"])
    print(f"Loaded clinical data: {len(surv_patients)} patients with survival")
except ImportError:
    logger.error("ParseTCGA not available. Install or add to path.")
    sys.exit(1)

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu


def parse_pair(eid):
    """Extract gene, pos1, pos2 from epistasis_id."""
    m1, m2 = eid.split("|")
    p1 = m1.split(":")
    p2 = m2.split(":")
    return p1[0], int(p1[2]), int(p2[2])


def build_cohort(gene, target_positions, project=None):
    """Build matched survival cohorts.

    Target: patients with BOTH mutations at target_positions.
    Control: patients with one target mutation + different second hit in same gene.

    Returns (target_ids, control_ids) intersected with survival patients.
    """
    gm = tcga.mutations.query(genes=[gene])
    if project:
        gm = gm[gm.Proj_name == project]

    if len(gm) == 0:
        return set(), set()

    # Group by patient — get set of positions per patient
    pp = gm.groupby("case_id").agg(
        positions=("Start_Position", lambda x: set(x)))
    multi = pp[pp.positions.apply(len) >= 2]

    target_set = set(target_positions)
    epi_positions = set(target_positions)

    target_ids = set()
    control_ids = set()

    for patient, row in multi.iterrows():
        pos = row.positions
        # Target: has BOTH specific positions
        if all(p in pos for p in target_positions):
            target_ids.add(patient)
        # Control: has one target position + at least one OTHER mutation in gene
        elif bool(pos & epi_positions) and len(pos - epi_positions) >= 1:
            control_ids.add(patient)

    control_ids -= target_ids
    return target_ids & surv_patients, control_ids & surv_patients


# Cross-model nominees (genes appearing in 2+ models' top cancer lists)
# From the KRAS similarity results
NOMINEES = [
    # (gene, epistasis_id, models that nominated it)
    ("WRN", "WRN:8:31072988:G:A:N|WRN:8:31072992:C:T:N", ["AlphaGenome", "Borzoi", "NTv3"]),
    ("TGFBR1", "TGFBR1:9:99137962:A:G:P|TGFBR1:9:99137964:A:G:P", ["AlphaGenome", "Borzoi", "RiNALMo"]),
    ("PDS5B", "PDS5B:13:32853889:C:T:P|PDS5B:13:32853895:C:T:P", ["AlphaGenome", "Borzoi"]),
    ("NUP214", "NUP214:9:131152997:C:T:P|NUP214:9:131152998:G:A:P", ["AlphaGenome", "NTv3", "RiNALMo"]),
    ("AGO1", "AGO1:1:36367447:A:G:N|AGO1:1:36367451:T:C:N", ["AlphaGenome", "Borzoi"]),
    ("MDM4", "MDM4:1:204526040:C:T:P|MDM4:1:204526046:T:C:P", ["AlphaGenome", "Borzoi"]),
    ("PREX2", "PREX2:8:68080778:A:G:P|PREX2:8:68080779:T:C:P", ["AlphaGenome", "NTv3", "MutBERT"]),
    ("ATF1", "ATF1:12:50809588:C:T:P|ATF1:12:50809593:T:C:P", ["Borzoi", "NTv3"]),
    ("BMPR1A", "BMPR1A:10:86756585:T:C:P|BMPR1A:10:86756587:A:G:P", ["AlphaGenome", "Borzoi"]),
    ("PABPC1", "PABPC1:8:100706737:G:A:N|PABPC1:8:100706740:C:T:N", ["AlphaGenome"]),
    ("FANCD2", "FANCD2:3:10068137:C:T:P|FANCD2:3:10068138:C:T:P", ["Borzoi", "RiNALMo"]),
    ("BARD1", "BARD1:2:214744295:G:A:N|BARD1:2:214744303:C:T:N", ["NTv3", "RiNALMo"]),
    ("SPEN", "SPEN:1:15873065:T:G:P|SPEN:1:15873068:A:G:P", ["AlphaGenome"]),
    ("ERBB4", "ERBB4:2:211590834:C:T:P|ERBB4:2:211590840:T:C:P", ["AlphaGenome"]),
    ("MTOR", "MTOR:1:11134915:G:A:P|MTOR:1:11134920:T:C:P", ["AlphaGenome"]),
    ("ACVR1B", "ACVR1B:12:51460764:G:A:N|ACVR1B:12:51460767:C:T:N", ["AlphaGenome", "Borzoi"]),
    ("SETD2", "SETD2:3:47016529:G:A:P|SETD2:3:47016532:T:C:P", ["NTv3"]),
    ("BRAF", "BRAF:7:140800437:C:T:N|BRAF:7:140800438:A:G:N", ["NTv3"]),
    ("POT1", "POT1:7:124858948:A:C:N|POT1:7:124858949:T:A:N", ["Borzoi"]),
    ("DROSHA", "DROSHA:5:31430697:C:T:P|DROSHA:5:31430709:G:A:P", ["Borzoi", "NTv3"]),
]

# Get all TCGA project names for multi-cancer testing
all_projects = sorted(clin["project_id"].dropna().unique()) if "project_id" in clin.columns else []
# Common cancer types
CANCER_PROJECTS = [p for p in all_projects if p.startswith("TCGA")]
if not CANCER_PROJECTS:
    # Try getting from mutations
    try:
        proj_counts = tcga.mutations.query().groupby("Proj_name").size().sort_values(ascending=False)
        CANCER_PROJECTS = proj_counts.head(30).index.tolist()
    except Exception:
        CANCER_PROJECTS = []

print(f"Cancer projects: {len(CANCER_PROJECTS)}")
print(f"Nominees to test: {len(NOMINEES)}")


# =========================================================================
# Test each nominee across cancer types
# =========================================================================
print("\n" + "=" * 100)
print("SURVIVAL ANALYSIS FOR KRAS-SIMILARITY NOMINEES")
print("=" * 100)

results = []

for gene, eid, models in NOMINEES:
    _, pos1, pos2 = parse_pair(eid)
    target_positions = [pos1, pos2]

    print(f"\n--- {gene} ({', '.join(models)}) ---")
    print(f"    Pair: {eid}")

    # Test across all cancer types
    for project in CANCER_PROJECTS:
        t_ids, c_ids = build_cohort(gene, target_positions, project=project)

        if len(t_ids) < 3 or len(c_ids) < 3:
            continue

        tc = clin[clin.case_id.isin(t_ids)]
        cc = clin[clin.case_id.isin(c_ids)]

        try:
            lr = logrank_test(tc["duration"], cc["duration"], tc["event"], cc["event"])
            p = lr.p_value
        except Exception:
            continue

        # Median survival
        t_med = tc["duration"].median()
        c_med = cc["duration"].median()

        result = {
            "gene": gene,
            "epistasis_id": eid,
            "project": project,
            "n_target": len(t_ids),
            "n_control": len(c_ids),
            "target_median_yrs": round(t_med, 2),
            "control_median_yrs": round(c_med, 2),
            "logrank_p": p,
            "direction": "worse" if t_med < c_med else "better",
            "models": ", ".join(models),
        }
        results.append(result)

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if p < 0.1:
            print(f"    {project:15s}: n={len(t_ids)}v{len(c_ids)}, "
                  f"target={t_med:.1f}yr vs control={c_med:.1f}yr, "
                  f"p={p:.3f}{sig} ({result['direction']})")

    # Also test pan-cancer (no project filter)
    t_ids, c_ids = build_cohort(gene, target_positions)
    if len(t_ids) >= 3 and len(c_ids) >= 3:
        tc = clin[clin.case_id.isin(t_ids)]
        cc = clin[clin.case_id.isin(c_ids)]
        try:
            lr = logrank_test(tc["duration"], cc["duration"], tc["event"], cc["event"])
            p = lr.p_value
            t_med = tc["duration"].median()
            c_med = cc["duration"].median()
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {'PAN-CANCER':15s}: n={len(t_ids)}v{len(c_ids)}, "
                  f"target={t_med:.1f}yr vs control={c_med:.1f}yr, "
                  f"p={p:.3f}{sig}")
            results.append({
                "gene": gene, "epistasis_id": eid, "project": "PAN-CANCER",
                "n_target": len(t_ids), "n_control": len(c_ids),
                "target_median_yrs": round(t_med, 2), "control_median_yrs": round(c_med, 2),
                "logrank_p": p, "direction": "worse" if t_med < c_med else "better",
                "models": ", ".join(models),
            })
        except Exception:
            pass

df_results = pd.DataFrame(results)


# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 100)
print("SIGNIFICANT FINDINGS (p < 0.05)")
print("=" * 100)

if len(df_results) > 0:
    sig = df_results[df_results["logrank_p"] < 0.05].sort_values("logrank_p")
    if len(sig) > 0:
        print(sig[["gene", "project", "n_target", "n_control",
                    "target_median_yrs", "control_median_yrs",
                    "logrank_p", "direction", "models"]].to_string(index=False))
    else:
        print("No significant findings at p < 0.05")

    # Save
    out_path = FIG_DIR / "kras_nominee_survival_results.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nAll results saved to {out_path}")

    # KM plots for significant findings
    sig_to_plot = df_results[df_results["logrank_p"] < 0.05].head(8)
    if len(sig_to_plot) > 0:
        n_plots = len(sig_to_plot)
        ncols = min(4, n_plots)
        nrows = (n_plots + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if n_plots == 1:
            axes = np.array([[axes]])
        axes = np.atleast_2d(axes)

        for idx, (_, row) in enumerate(sig_to_plot.iterrows()):
            ax = axes[idx // ncols, idx % ncols]
            gene = row["gene"]
            project = row["project"]
            _, pos1, pos2 = parse_pair(row["epistasis_id"])

            if project == "PAN-CANCER":
                t_ids, c_ids = build_cohort(gene, [pos1, pos2])
            else:
                t_ids, c_ids = build_cohort(gene, [pos1, pos2], project=project)

            tc = clin[clin.case_id.isin(t_ids)]
            cc = clin[clin.case_id.isin(c_ids)]

            kmf_t = KaplanMeierFitter()
            kmf_c = KaplanMeierFitter()
            kmf_t.fit(tc.duration, tc.event, label=f"Specific pair (n={len(tc)})")
            kmf_c.fit(cc.duration, cc.event, label=f"Control (n={len(cc)})")

            kmf_c.plot_survival_function(ax=ax, color="#4A7FB5", linewidth=1)
            kmf_t.plot_survival_function(ax=ax, color="#CB6A49", linewidth=1.5)

            p_str = f"p={row['logrank_p']:.3f}" if row["logrank_p"] >= 0.001 else f"p={row['logrank_p']:.1e}"
            ax.set_title(f"{gene}\n{project}", fontsize=9, fontweight="bold")
            ax.text(0.95, 0.95, p_str, transform=ax.transAxes, fontsize=8,
                    ha="right", va="top", fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
            ax.set_xlabel("Years")
            ax.set_ylabel("Survival")
            ax.legend(fontsize=7)

        for idx in range(n_plots, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.suptitle("Survival: KRAS-similarity nominated epistatic pairs",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_kras_nominee_survival.png", dpi=200, bbox_inches="tight")
        fig.savefig(FIG_DIR / "fig_kras_nominee_survival.pdf", bbox_inches="tight")
        print(f"KM plots saved to {FIG_DIR / 'fig_kras_nominee_survival.png'}")
else:
    print("No results to report.")

print("\nDone.")
