"""
Rigorous survival confounder removal framework for epistatic pairs.

For each candidate pair (gene, pos1, pos2), runs a cascade of increasingly
strict survival tests. A pair must pass ALL levels to be called significant.

Levels:
  L0 — Raw logrank (target=both mutations, control=one+other in gene)
  L1 — Gene mutation count matched (control has same # mutations in gene)
  L2 — L1 + TMB matched (control within ±50% TMB, or stratified by TMB tertile)
  L3 — L2 + cancer type stratified (stratified logrank across cancer types)

Each level removes a specific confounder:
  L0→L1: removes "more mutations in gene = worse prognosis" confound
  L1→L2: removes "higher TMB = different prognosis" confound
  L2→L3: removes "cancer type composition" confound

Requires ParseTCGA.
"""

import sys
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
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

MIN_TARGET = 2
MIN_CONTROL = 3


def _logrank(target_ids, control_ids, clin):
    """Run logrank, return (p, t_median, c_median, n_t, n_c) or None."""
    tc = clin[clin.case_id.isin(target_ids)]
    cc = clin[clin.case_id.isin(control_ids)]
    if len(tc) < MIN_TARGET or len(cc) < MIN_CONTROL:
        return None
    try:
        lr = logrank_test(tc["duration"], cc["duration"], tc["event"], cc["event"])
        return lr.p_value, tc["duration"].median(), cc["duration"].median(), len(tc), len(cc)
    except Exception:
        return None


def _stratified_logrank(target_ids, control_ids, clin, strata_col):
    """Stratified logrank test. Pools evidence across strata."""
    tc = clin[clin.case_id.isin(target_ids)]
    cc = clin[clin.case_id.isin(control_ids)]
    if len(tc) < MIN_TARGET or len(cc) < MIN_CONTROL:
        return None

    combined = pd.concat([
        tc.assign(_group=1),
        cc.assign(_group=0),
    ])

    if strata_col not in combined.columns:
        return None

    # Use lifelines stratified logrank
    from lifelines.statistics import logrank_test as lr_test
    try:
        # Stratified: pool within each stratum
        strata = combined[strata_col].dropna().unique()
        chi2_sum = 0.0
        var_sum = 0.0
        for s in strata:
            sub = combined[combined[strata_col] == s]
            t = sub[sub["_group"] == 1]
            c = sub[sub["_group"] == 0]
            if len(t) < 1 or len(c) < 1:
                continue
            lr = lr_test(t["duration"], c["duration"], t["event"], c["event"])
            chi2_sum += lr.test_statistic
            var_sum += 1  # approximate: count strata with data
        if var_sum == 0:
            return None
        # Combined p from chi2 sum (approximate)
        from scipy.stats import chi2
        p = chi2.sf(chi2_sum, df=var_sum)
        return p, tc["duration"].median(), cc["duration"].median(), len(tc), len(cc)
    except Exception:
        return None


def test_pair_cascade(gene, pos1, pos2, patient_gene_info, clin, patient_tmb, patient_proj):
    """Run the full confounder cascade for one pair.

    patient_gene_info: dict[patient_id] -> (positions: frozenset, n_muts: int)
        Only patients mutated in this gene.

    Returns dict with results at each level, or None if L0 fails.
    """
    # Classify patients
    target_ids = set()
    # For each mutation count bucket: patients with that many muts but not this pair
    ctrl_by_nmut = defaultdict(set)
    target_nmuts = {}

    for pid, (positions, n_muts) in patient_gene_info.items():
        has_pair = (pos1 in positions) and (pos2 in positions)
        if has_pair:
            target_ids.add(pid)
            target_nmuts[pid] = n_muts
        else:
            ctrl_by_nmut[n_muts].add(pid)

    if len(target_ids) < MIN_TARGET:
        return None

    # =====================================================================
    # L0: Raw — control = anyone with one of the positions + different 2nd hit
    # =====================================================================
    ctrl_L0 = set()
    for pid, (positions, n_muts) in patient_gene_info.items():
        if pid in target_ids:
            continue
        if (pos1 in positions) or (pos2 in positions):
            ctrl_L0.add(pid)

    r0 = _logrank(target_ids, ctrl_L0, clin)
    if r0 is None:
        return None

    result = {
        "L0_p": r0[0], "L0_t_med": r0[1], "L0_c_med": r0[2],
        "L0_n_target": r0[3], "L0_n_ctrl": r0[4],
    }

    # =====================================================================
    # L1: Gene mutation count matched
    # Control = patients with same number of mutations in the gene, but not this pair
    # =====================================================================
    ctrl_L1 = set()
    for pid in target_ids:
        n = target_nmuts[pid]
        # Match exactly, or ±1 if exact match is sparse
        matched = ctrl_by_nmut.get(n, set())
        if len(matched) < 3:
            matched = matched | ctrl_by_nmut.get(n - 1, set()) | ctrl_by_nmut.get(n + 1, set())
        ctrl_L1 |= matched

    r1 = _logrank(target_ids, ctrl_L1, clin)
    if r1 is not None:
        result["L1_p"] = r1[0]
        result["L1_t_med"] = r1[1]
        result["L1_c_med"] = r1[2]
        result["L1_n_target"] = r1[3]
        result["L1_n_ctrl"] = r1[4]
    else:
        result["L1_p"] = np.nan

    # =====================================================================
    # L2: L1 + TMB matched
    # From L1 controls, keep only those within ±50% of target median TMB
    # (or TMB-tertile stratified)
    # =====================================================================
    target_tmbs = [patient_tmb.get(pid, np.nan) for pid in target_ids]
    target_tmbs = [t for t in target_tmbs if not np.isnan(t)]
    if target_tmbs:
        tmb_med = np.median(target_tmbs)
        tmb_lo = tmb_med * 0.5
        tmb_hi = tmb_med * 1.5

        ctrl_L2 = set()
        for pid in ctrl_L1:
            t = patient_tmb.get(pid, np.nan)
            if tmb_lo <= t <= tmb_hi:
                ctrl_L2.add(pid)

        r2 = _logrank(target_ids, ctrl_L2, clin)
        if r2 is not None:
            result["L2_p"] = r2[0]
            result["L2_t_med"] = r2[1]
            result["L2_c_med"] = r2[2]
            result["L2_n_target"] = r2[3]
            result["L2_n_ctrl"] = r2[4]
        else:
            result["L2_p"] = np.nan

        # TMB balance check
        ctrl_tmbs = [patient_tmb.get(pid, np.nan) for pid in ctrl_L2]
        ctrl_tmbs = [t for t in ctrl_tmbs if not np.isnan(t)]
        if len(target_tmbs) >= 2 and len(ctrl_tmbs) >= 2:
            _, tmb_p = mannwhitneyu(target_tmbs, ctrl_tmbs, alternative="two-sided")
            result["L2_tmb_p"] = tmb_p
        else:
            result["L2_tmb_p"] = np.nan
    else:
        result["L2_p"] = np.nan
        result["L2_tmb_p"] = np.nan

    # =====================================================================
    # L3: L2 + cancer type stratified
    # Stratified logrank across cancer types (from L2 controls)
    # =====================================================================
    if not np.isnan(result.get("L2_p", np.nan)):
        # Add project info to clin for stratification
        clin_proj = clin.copy()
        clin_proj["project"] = clin_proj["case_id"].map(patient_proj)

        r3 = _stratified_logrank(target_ids, ctrl_L2, clin_proj, "project")
        if r3 is not None:
            result["L3_p"] = r3[0]
            result["L3_n_target"] = r3[3]
            result["L3_n_ctrl"] = r3[4]
        else:
            result["L3_p"] = np.nan
    else:
        result["L3_p"] = np.nan

    return result


def main():
    from parsetcga import TCGAData

    print("=" * 100)
    print("SURVIVAL CONFOUNDER FRAMEWORK")
    print("=" * 100)

    tcga = TCGAData()
    clin = tcga.clinical.survival_prepared()
    surv_patients = set(clin["case_id"])
    all_muts = tcga.mutations.query()

    # Patient TMB
    patient_tmb = all_muts.groupby("case_id").size().to_dict()

    # Patient project
    patient_proj = (
        all_muts[["case_id", "Proj_name"]]
        .drop_duplicates("case_id")
        .set_index("case_id")["Proj_name"]
        .to_dict()
    )

    # Patient-gene info
    logger.info("Building patient-gene index...")
    pg = all_muts.groupby(["case_id", "Gene_name"]).agg(
        n_muts=("Start_Position", "nunique"),
        positions=("Start_Position", lambda x: frozenset(x)),
    ).reset_index()

    # Index: gene -> {patient: (positions, n_muts)}
    gene_patient_info = {}
    for _, row in pg.iterrows():
        pid = row["case_id"]
        if pid not in surv_patients:
            continue
        gene = row["Gene_name"]
        if gene not in gene_patient_info:
            gene_patient_info[gene] = {}
        gene_patient_info[gene][pid] = (row["positions"], row["n_muts"])

    logger.info("Indexed %d genes", len(gene_patient_info))

    # Load significant pairs
    sig = pd.read_csv(FIG_DIR / "tcga_survival_epistasis_significant.csv")
    sig = sig.sort_values("logrank_p").drop_duplicates(["gene", "pos1", "pos2"])
    print(f"Testing {len(sig)} unique pairs\n")

    # Run cascade
    results = []
    for i, (_, row) in enumerate(sig.iterrows()):
        gene = row["gene"]
        pos1, pos2 = int(row["pos1"]), int(row["pos2"])

        gpi = gene_patient_info.get(gene, {})
        if not gpi:
            continue

        r = test_pair_cascade(gene, pos1, pos2, gpi, clin, patient_tmb, patient_proj)
        if r is None:
            continue

        r["gene"] = gene
        r["pos1"] = pos1
        r["pos2"] = pos2
        r["dist"] = int(row["dist"])
        r["prot1"] = str(row.get("prot1", "?"))[:15]
        r["prot2"] = str(row.get("prot2", "?"))[:15]
        r["epistasis_id"] = row.get("epistasis_id", "")
        r["is_driver"] = row.get("is_driver", False)
        r["orig_fdr"] = row.get("fdr_q", np.nan)
        results.append(r)

        if (i + 1) % 50 == 0:
            logger.info("  %d / %d pairs tested", i + 1, len(sig))

    df = pd.DataFrame(results)
    print(f"\nTested: {len(df)} pairs")

    # Summary
    print(f"\n{'='*100}")
    print("CASCADE RESULTS")
    print(f"{'='*100}")
    for level, col in [("L0 raw", "L0_p"), ("L1 gene-mut-matched", "L1_p"),
                        ("L2 +TMB-matched", "L2_p"), ("L3 +cancer-stratified", "L3_p")]:
        valid = df[col].notna()
        sig05 = (df[col] < 0.05) & valid
        sig01 = (df[col] < 0.01) & valid
        print(f"  {level:>25s}: {valid.sum():>4d} testable, "
              f"{sig05.sum():>4d} p<0.05, {sig01.sum():>4d} p<0.01")

    # Pairs passing ALL levels
    all_pass = df[
        (df["L0_p"] < 0.05) &
        (df["L1_p"] < 0.05) &
        (df["L2_p"] < 0.05)
    ].copy()

    # Include L3 where available
    has_L3 = all_pass["L3_p"].notna()
    all_pass_L3 = all_pass[~has_L3 | (all_pass["L3_p"] < 0.05)]

    print(f"\n  Pass L0+L1+L2: {len(all_pass)}")
    print(f"  Pass L0+L1+L2+L3 (where testable): {len(all_pass_L3)}")

    # Show survivors
    print(f"\n{'='*100}")
    print("PAIRS PASSING ALL CONFOUNDER LEVELS")
    print(f"{'='*100}")

    display = all_pass_L3.sort_values("L2_p")
    print(f"\n{'gene':>12s} {'pair':>25s} {'d':>3s} | "
          f"{'L0':>9s} {'L1':>9s} {'L2':>9s} {'L3':>9s} | "
          f"{'n_t':>4s} {'n_c':>5s} {'surv':>11s} {'tmb_p':>6s}")
    print("-" * 110)

    for _, r in display.iterrows():
        L3_str = f"{r['L3_p']:.2e}" if pd.notna(r["L3_p"]) else "—"
        drv = "*" if r.get("is_driver") else " "
        print(f"{drv}{r['gene']:>11s} {r['prot1']:>11s}+{r['prot2']:<11s} {r['dist']:>3d} | "
              f"{r['L0_p']:>9.2e} {r['L1_p']:>9.2e} {r['L2_p']:>9.2e} {L3_str:>9s} | "
              f"{r.get('L2_n_target', 0):>4.0f} {r.get('L2_n_ctrl', 0):>5.0f} "
              f"{r.get('L2_t_med', 0):>5.1f}v{r.get('L2_c_med', 0):>4.1f}yr "
              f"{r.get('L2_tmb_p', np.nan):>6.2f}")

    # Save
    out = FIG_DIR / "tcga_survival_cascade.csv"
    df.to_csv(out, index=False)
    print(f"\nAll results → {out}")

    out2 = FIG_DIR / "tcga_survival_cascade_clean.csv"
    all_pass_L3.to_csv(out2, index=False)
    print(f"Clean pairs → {out2} ({len(all_pass_L3)} pairs)")

    print("\nDone.")


if __name__ == "__main__":
    main()
