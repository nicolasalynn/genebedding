"""
Identify epistatic mutation pairs with survival significance in TCGA.

Pipeline:
  1. ELIGIBLE PAIRS — all protein-coding genes (excl noise families),
     coding variants, 1-500bp distance, ≥2 co-mutated patients, ≤500 patients
  2. QUALITY FILTER — depth-based: flag likely germline positions using
     tumor VAF clustering and FILTER-field germline annotations;
     require ≥1 position per pair to be likely somatic
  3. CO-OCCURRENCE FILTER — Fisher's exact test for enriched co-occurrence
     (O/E ≥ 1, Fisher p < 0.05); removes random co-occurrences
  4. SURVIVAL ANALYSIS — two tracks with BH-FDR correction:
     a) Pan-cancer logrank
     b) Cancer-type-specific logrank
  5. OUTPUT — FDR-significant pairs + full table for null-set construction

Runs locally. Requires ParseTCGA.
"""

import sys
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from lifelines.statistics import logrank_test
from statsmodels.stats.multitest import multipletests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
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
ANNOT_DIR = EPISTASIS_PAPER_ROOT / "data" / "annotations"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MIN_PATIENTS = 2            # minimum target patients (pair carriers)
MAX_PATIENTS = 500          # max target patients (above = likely germline)
MIN_CONTROL = 3             # minimum control patients
MAX_DISTANCE = 500          # bp
CO_OCC_ALPHA = 0.05         # Fisher p threshold for co-occurrence
FDR_THRESHOLD = 0.10        # BH-FDR reporting threshold
GERMLINE_VAF_LO = 0.15      # tumor VAF range indicating germline het
GERMLINE_VAF_HI = 0.35
GERMLINE_VAF_STD = 0.08     # low spread = germline
GERMLINE_FLAG_FRAC = 0.80   # if >80% of calls flagged germline_risk → skip

ELIGIBLE_VC = {
    # Protein-altering
    "Missense_Mutation", "Nonsense_Mutation",
    "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site",
    "In_Frame_Del", "In_Frame_Ins",
    "Nonstop_Mutation", "Translation_Start_Site",
    # Silent & intronic — models work on DNA, not protein
    "Silent",
    "Intron",
    "Splice_Region",
    "5'UTR", "3'UTR",
    "5'Flank", "3'Flank",
}

# Gene families to exclude: noise, repeat-rich, polymorphic
EXCLUDE_GENE_PREFIXES = (
    "OR",       # olfactory receptors
    "TAS2R",    # taste receptors
    "KIR",      # killer immunoglobulin-like receptors
    "PRAMEF",   # PRAME family
    "GAGE",     # GAGE family
    "NBPF",     # neuroblastoma breakpoint family
    "LILR",     # leukocyte immunoglobulin-like receptors
    "TPTE",
    "FRG",
)
EXCLUDE_GENES_EXACT = {
    # Hypermutated / very long
    "TTN", "MUC16", "MUC6", "MUC4", "MUC5B", "MUC17", "MUC3A", "MUC12",
    "OBSCN", "SYNE1", "SYNE2", "RYR1", "RYR2", "RYR3",
    "DNAH5", "DNAH11", "PCLO", "CSMD3", "USH2A", "LRP1B", "FLG",
    "MKI67", "AHNAK2", "PARP4", "HRNR", "PRSS1", "PRSS3",
    "PABPC3", "FCGBP",
    # HLA
    "HLA-A", "HLA-B", "HLA-C", "HLA-DRB1", "HLA-DRB5", "HLA-DQA2",
}


def _is_excluded_gene(gene):
    """Check if gene should be excluded by exact match or prefix."""
    if gene in EXCLUDE_GENES_EXACT:
        return True
    for prefix in EXCLUDE_GENE_PREFIXES:
        if gene.startswith(prefix):
            return True
    return False


# =========================================================================
# DATA LOADING
# =========================================================================
def load_data():
    """Load clinical + mutation data, filter to eligible coding mutations."""
    from parsetcga import TCGAData

    tcga = TCGAData()
    clin = tcga.clinical.survival_prepared()
    surv_patients = set(clin["case_id"])
    logger.info("Patients with survival: %d", len(surv_patients))

    all_muts = tcga.mutations.query()
    logger.info("Total mutations: %d", len(all_muts))

    # Patient → project
    patient_proj = (
        all_muts[["case_id", "Proj_name"]]
        .drop_duplicates("case_id")
        .set_index("case_id")["Proj_name"]
    )

    # Exclude by gene name + hypermutated (>50k mutations)
    gene_counts = all_muts["Gene_name"].value_counts()
    hypermut = set(gene_counts[gene_counts > 50000].index)
    exclude = EXCLUDE_GENES_EXACT | hypermut

    coding = all_muts[
        (~all_muts["Gene_name"].apply(_is_excluded_gene))
        & (~all_muts["Gene_name"].isin(exclude))
        & (all_muts["Variant_Classification"].isin(ELIGIBLE_VC))
    ].copy()
    logger.info("After gene/VC filter: %d mutations in %d genes",
                len(coding), coding["Gene_name"].nunique())

    # Driver gene labels (for annotation, not filtering)
    oncokb = pd.read_csv(ANNOT_DIR / "cancerGeneList.tsv", sep="\t")
    driver_genes = set(
        oncokb[oncokb["Gene Type"].isin(["ONCOGENE", "TSG", "ONCOGENE_AND_TSG"])][
            "Hugo Symbol"
        ]
    )

    return clin, coding, all_muts, surv_patients, patient_proj, driver_genes


# =========================================================================
# QUALITY FILTER — depth & germline flagging
# =========================================================================
def build_position_quality(coding_muts):
    """For each (gene, position), compute quality metrics.

    Returns dict[(gene, pos)] → {
        'median_vaf', 'std_vaf', 'frac_germline_risk',
        'frac_alt_in_normal', 'likely_germline', 'n_calls',
        'median_t_depth',
    }

    Vectorized for speed on large datasets (50M+ rows).
    """
    logger.info("Computing per-position quality metrics (vectorized)...")

    # Vectorized computations — no .apply(lambda)
    vaf = coding_muts["t_alt_count"] / coding_muts["t_depth"].clip(lower=1)
    filt = coding_muts["FILTER"].fillna("")
    has_germ = filt.str.contains("germline_risk", regex=False)
    has_alt_norm = filt.str.contains("alt_allele_in_normal", regex=False)

    # Build a temporary frame with just what we need for groupby
    tmp = pd.DataFrame({
        "gene": coding_muts["Gene_name"].values,
        "pos": coding_muts["Start_Position"].values,
        "vaf": vaf.values,
        "germ": has_germ.values.astype(np.float32),
        "alt_norm": has_alt_norm.values.astype(np.float32),
        "depth": coding_muts["t_depth"].values,
    })

    logger.info("  Aggregating %d rows...", len(tmp))
    agg = tmp.groupby(["gene", "pos"]).agg(
        n_calls=("vaf", "size"),
        med_vaf=("vaf", "median"),
        std_vaf=("vaf", "std"),
        frac_germ=("germ", "mean"),
        frac_alt_norm=("alt_norm", "mean"),
        med_depth=("depth", "median"),
    )
    # Fill NaN std (single-observation groups)
    agg["std_vaf"] = agg["std_vaf"].fillna(0)

    # Germline heuristic — vectorized
    agg["likely_germline"] = (
        (agg["med_vaf"].between(GERMLINE_VAF_LO, GERMLINE_VAF_HI))
        & (agg["std_vaf"] < GERMLINE_VAF_STD)
        & (agg["frac_germ"] > GERMLINE_FLAG_FRAC)
    ) | (
        (agg["frac_germ"] > 0.9) & (agg["frac_alt_norm"] > 0.9)
    )

    # Convert to dict
    quality = {}
    for (gene, pos), row in agg.iterrows():
        quality[(gene, pos)] = {
            "median_vaf": round(row["med_vaf"], 4),
            "std_vaf": round(row["std_vaf"], 4),
            "frac_germline_risk": round(row["frac_germ"], 3),
            "frac_alt_in_normal": round(row["frac_alt_norm"], 3),
            "likely_germline": bool(row["likely_germline"]),
            "n_calls": int(row["n_calls"]),
            "median_t_depth": round(row["med_depth"], 1),
        }

    n_germline = sum(1 for v in quality.values() if v["likely_germline"])
    logger.info("Position quality: %d positions, %d flagged likely germline (%.1f%%)",
                len(quality), n_germline, 100 * n_germline / max(len(quality), 1))
    return quality


# =========================================================================
# ANNOTATION CACHE
# =========================================================================
def build_annotation_cache(coding_muts):
    """Pre-compute (gene, pos) → (vc, hgvsp, ref, alt, chrom)."""
    logger.info("Building annotation cache...")
    cache = {}
    for (gene, pos), grp in coding_muts.groupby(["Gene_name", "Start_Position"]):
        vc = grp["Variant_Classification"].mode()
        vc = str(vc.iloc[0]) if len(vc) > 0 else "?"
        hp = grp["HGVSp_Short"].dropna().mode() if "HGVSp_Short" in grp.columns else pd.Series()
        hgvsp = str(hp.iloc[0])[:20] if len(hp) > 0 else "?"
        ref = str(grp["Reference_Allele"].iloc[0])
        alt = str(grp["Tumor_Seq_Allele2"].iloc[0])
        chrom = str(grp["Chromosome"].iloc[0]).replace("chr", "")
        cache[(gene, pos)] = (vc, hgvsp, ref, alt, chrom)
    logger.info("Annotation cache: %d entries", len(cache))
    return cache


# =========================================================================
# PAIR ENUMERATION
# =========================================================================
def enumerate_pairs(patient_gene, max_dist=MAX_DISTANCE):
    """Enumerate within-gene position pairs, 1 ≤ dist ≤ max_dist.

    Returns dict[(gene, pos1, pos2)] → n_patients.
    """
    logger.info("Enumerating pairs (dist ≤ %d bp)...", max_dist)
    pair_counts = Counter()
    n_rows = len(patient_gene)

    for idx, (_, row) in enumerate(patient_gene.iterrows()):
        if (idx + 1) % 500000 == 0:
            logger.info("  %d / %d rows (%.0f%%)", idx + 1, n_rows,
                        100 * (idx + 1) / n_rows)
        positions = sorted(row["positions"])
        n_pos = len(positions)
        if n_pos < 2 or n_pos > 20:
            continue
        gene = row["Gene_name"]
        for i in range(n_pos):
            for j in range(i + 1, n_pos):
                d = positions[j] - positions[i]
                if d <= max_dist:
                    pair_counts[(gene, positions[i], positions[j])] += 1

    # Apply patient count bounds
    filtered = {
        k: v for k, v in pair_counts.items()
        if MIN_PATIENTS <= v <= MAX_PATIENTS
    }
    logger.info("Pairs: %d raw → %d after patient bounds [%d, %d]",
                len(pair_counts), len(filtered), MIN_PATIENTS, MAX_PATIENTS)
    return filtered


# =========================================================================
# QUALITY + CO-OCCURRENCE FILTER
# =========================================================================
def filter_pairs(pair_counts, patient_gene, pos_quality):
    """Apply quality filter then co-occurrence filter.

    Quality: require at least one position in the pair to NOT be likely germline.
    Co-occurrence: O/E ≥ 1 AND Fisher p < CO_OCC_ALPHA.

    Returns (passed_pairs, oe_values, fisher_pvals).
    """
    # --- Quality filter ---
    quality_passed = {}
    n_germ_dropped = 0
    for (gene, pos1, pos2), n in pair_counts.items():
        q1 = pos_quality.get((gene, pos1), {})
        q2 = pos_quality.get((gene, pos2), {})
        g1 = q1.get("likely_germline", False)
        g2 = q2.get("likely_germline", False)
        if g1 and g2:
            n_germ_dropped += 1
            continue
        quality_passed[(gene, pos1, pos2)] = n

    logger.info("Quality filter: %d → %d pairs (%d dropped, both positions germline)",
                len(pair_counts), len(quality_passed), n_germ_dropped)

    # --- Co-occurrence filter ---
    logger.info("Co-occurrence filter (O/E≥1, Fisher p<%.2f)...", CO_OCC_ALPHA)

    # Pre-index: gene → {patient: positions}
    gene_patient_pos = {}
    for _, row in patient_gene.iterrows():
        gene = row["Gene_name"]
        if gene not in gene_patient_pos:
            gene_patient_pos[gene] = {}
        gene_patient_pos[gene][row["case_id"]] = row["positions"]

    passed = {}
    oe_values = {}
    fisher_pvals = {}

    for (gene, pos1, pos2), n_both in quality_passed.items():
        gpp = gene_patient_pos.get(gene, {})
        n_gene = len(gpp)
        if n_gene < 5:
            continue

        n_a = sum(1 for ps in gpp.values() if pos1 in ps)
        n_b = sum(1 for ps in gpp.values() if pos2 in ps)

        expected = max((n_a / n_gene) * (n_b / n_gene) * n_gene, 0.001)
        oe = n_both / expected
        if oe < 1.0:
            continue

        table = [
            [n_both, n_a - n_both],
            [n_b - n_both, n_gene - n_a - n_b + n_both],
        ]
        try:
            _, p_fisher = fisher_exact(table, alternative="greater")
        except Exception:
            continue

        oe_values[(gene, pos1, pos2)] = oe
        fisher_pvals[(gene, pos1, pos2)] = p_fisher

        if p_fisher < CO_OCC_ALPHA:
            passed[(gene, pos1, pos2)] = n_both

    logger.info("Co-occurrence: %d → %d pairs", len(quality_passed), len(passed))
    return passed, oe_values, fisher_pvals


# =========================================================================
# SURVIVAL TESTING
# =========================================================================
def build_gene_patient_index(patient_gene, surv_patients, patient_proj):
    """gene → {patient: (positions, n_muts, project)}. Only survival patients."""
    gpi = {}
    for _, row in patient_gene.iterrows():
        pid = row["case_id"]
        if pid not in surv_patients:
            continue
        gene = row["Gene_name"]
        proj = patient_proj.get(pid)
        n_muts = len(row["positions"])
        gpi.setdefault(gene, {})[pid] = (row["positions"], n_muts, proj)
    return gpi


def _logrank(target_ids, control_ids, clin):
    """Returns (p, target_median, control_median) or None."""
    tc = clin[clin.case_id.isin(target_ids)]
    cc = clin[clin.case_id.isin(control_ids)]
    if len(tc) < 2 or len(cc) < 2:
        return None
    try:
        lr = logrank_test(tc["duration"], cc["duration"], tc["event"], cc["event"])
        return lr.p_value, tc["duration"].median(), cc["duration"].median()
    except Exception:
        return None


def test_pair(gene, pos1, pos2, gpi, clin, annot_cache, pos_quality, patient_proj):
    """Run pan-cancer + cancer-specific survival for one pair.

    Control group is BURDEN-MATCHED: patients with the same number of
    mutations in this gene, but NOT carrying this specific pair.
    This removes gene-level mutation burden as a confounder BEFORE
    FDR correction, reducing the multiple-testing penalty.

    Returns list of result dicts.
    """
    patients = gpi.get(gene, {})
    target_ids = set()
    target_nmuts = {}
    # Bucket controls by mutation count
    ctrl_by_nmut = {}

    target_by_proj = {}
    control_by_proj = {}

    for pid, (positions, n_muts, proj) in patients.items():
        has1, has2 = pos1 in positions, pos2 in positions
        if has1 and has2:
            target_ids.add(pid)
            target_nmuts[pid] = n_muts
            if proj:
                target_by_proj.setdefault(proj, set()).add(pid)
        else:
            ctrl_by_nmut.setdefault(n_muts, set()).add(pid)

    # Build burden-matched control: for each target patient's mutation count,
    # include controls with the same count (±1 if sparse)
    control_ids = set()
    for pid in target_ids:
        n = target_nmuts[pid]
        matched = ctrl_by_nmut.get(n, set())
        if len(matched) < 3:
            matched = matched | ctrl_by_nmut.get(n - 1, set()) | ctrl_by_nmut.get(n + 1, set())
        control_ids |= matched

    control_ids -= target_ids

    # Build per-project controls from the burden-matched set
    for pid in control_ids:
        _, _, proj = patients.get(pid, (None, None, None))
        if proj:
            control_by_proj.setdefault(proj, set()).add(pid)

    results = []

    # Lazy annotations
    a1 = annot_cache.get((gene, pos1), ("?", "?", "?", "?", "?"))
    a2 = annot_cache.get((gene, pos2), ("?", "?", "?", "?", "?"))
    q1 = pos_quality.get((gene, pos1), {})
    q2 = pos_quality.get((gene, pos2), {})

    base = {
        "gene": gene, "pos1": pos1, "pos2": pos2,
        "dist": abs(pos2 - pos1),
        "prot1": a1[1], "prot2": a2[1],
        "vc1": a1[0], "vc2": a2[0],
        "vaf1": q1.get("median_vaf", np.nan),
        "vaf2": q2.get("median_vaf", np.nan),
        "depth1": q1.get("median_t_depth", np.nan),
        "depth2": q2.get("median_t_depth", np.nan),
        "germ_flag1": q1.get("likely_germline", False),
        "germ_flag2": q2.get("likely_germline", False),
    }

    # --- Pan-cancer ---
    if len(target_ids) >= MIN_PATIENTS and len(control_ids) >= MIN_CONTROL:
        lr = _logrank(target_ids, control_ids, clin)
        if lr is not None:
            p, t_med, c_med = lr
            projs = pd.Series([patient_proj.get(pid, "?") for pid in target_ids])
            top3 = projs.value_counts().head(3)
            results.append({
                **base,
                "track": "pan-cancer", "project": "ALL",
                "n_target": len(target_ids), "n_control": len(control_ids),
                "target_median_yr": round(t_med, 3),
                "control_median_yr": round(c_med, 3),
                "logrank_p": p,
                "direction": "worse" if t_med < c_med else "better",
                "top_cancers": "; ".join(f"{c}({n})" for c, n in top3.items()),
            })

    # --- Cancer-specific ---
    for proj in set(target_by_proj) & set(control_by_proj):
        t = target_by_proj[proj]
        c = control_by_proj.get(proj, set()) - t
        if len(t) < MIN_PATIENTS or len(c) < MIN_CONTROL:
            continue
        lr = _logrank(t, c, clin)
        if lr is None:
            continue
        p, t_med, c_med = lr
        results.append({
            **base,
            "track": "cancer-specific", "project": proj,
            "n_target": len(t), "n_control": len(c),
            "target_median_yr": round(t_med, 3),
            "control_median_yr": round(c_med, 3),
            "logrank_p": p,
            "direction": "worse" if t_med < c_med else "better",
            "top_cancers": proj,
        })

    return results


# =========================================================================
# FDR
# =========================================================================
def apply_fdr(df):
    """BH-FDR correction within each track."""
    df = df.copy()
    df["fdr_q"] = np.nan
    for track in df["track"].unique():
        mask = df["track"] == track
        pvals = df.loc[mask, "logrank_p"].values
        if len(pvals) == 0:
            continue
        _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
        df.loc[mask, "fdr_q"] = qvals
    return df


def format_epistasis_id(row, annot_cache):
    """Build epistasis_id from annotation cache."""
    a1 = annot_cache.get((row["gene"], row["pos1"]))
    a2 = annot_cache.get((row["gene"], row["pos2"]))
    if a1 is None or a2 is None:
        return None
    chrom = a1[4]
    return (
        f"{row['gene']}:{chrom}:{row['pos1']}:{a1[2]}:{a1[3]}:P"
        f"|{row['gene']}:{chrom}:{row['pos2']}:{a2[2]}:{a2[3]}:P"
    )


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("=" * 100)
    print("TCGA SURVIVAL EPISTASIS")
    print("=" * 100)

    # ---- Load ----
    clin, coding, all_muts, surv_patients, patient_proj, driver_genes = load_data()

    # ---- Position quality ----
    pos_quality = build_position_quality(coding)

    # ---- Annotation cache ----
    annot_cache = build_annotation_cache(coding)

    # ---- Patient-gene positions ----
    logger.info("Grouping patient-gene positions...")
    patient_gene = (
        coding.groupby(["case_id", "Gene_name"])
        .agg(positions=("Start_Position", lambda x: frozenset(x)))
        .reset_index()
    )
    logger.info("Patient-gene groups: %d", len(patient_gene))

    # ---- Enumerate pairs ----
    pair_counts = enumerate_pairs(patient_gene)

    # ---- Quality + co-occurrence filter ----
    passed, oe_values, fisher_pvals = filter_pairs(pair_counts, patient_gene, pos_quality)

    # ---- Pre-index for survival ----
    logger.info("Building survival index...")
    gpi = build_gene_patient_index(patient_gene, surv_patients, patient_proj)
    logger.info("Indexed %d genes", len(gpi))

    # ---- Survival tests ----
    print(f"\n{'='*80}")
    print(f"SURVIVAL TESTING: {len(passed)} pairs")
    print(f"{'='*80}")

    all_results = []
    n_pan = 0
    n_specific = 0

    CHECKPOINT_EVERY = 50000  # save partial results every N pairs
    partial_path = FIG_DIR / "tcga_survival_epistasis_PARTIAL.csv"

    for i, ((gene, pos1, pos2), n_both) in enumerate(passed.items()):
        if (i + 1) % 1000 == 0:
            logger.info("  %d / %d pairs (pan=%d, specific=%d)",
                        i + 1, len(passed), n_pan, n_specific)

        # Periodic checkpoint
        if (i + 1) % CHECKPOINT_EVERY == 0 and all_results:
            df_partial = pd.DataFrame(all_results)
            df_partial = apply_fdr(df_partial)
            df_partial.to_csv(partial_path, index=False)
            n_sig = (df_partial["fdr_q"] < FDR_THRESHOLD).sum()
            n_nom = (df_partial["logrank_p"] < 0.05).sum()
            logger.info("  CHECKPOINT %d: %d results, %d nominal p<0.05, %d FDR<%.2f → %s",
                        i + 1, len(df_partial), n_nom, n_sig, FDR_THRESHOLD, partial_path)
            # Show VC breakdown of significant pairs
            sig_partial = df_partial[df_partial["fdr_q"] < FDR_THRESHOLD]
            if len(sig_partial) > 0:
                vc_pairs = sig_partial.apply(lambda r: f"{r.get('vc1','?')}/{r.get('vc2','?')}", axis=1)
                logger.info("    VC breakdown (FDR-sig): %s", vc_pairs.value_counts().head(5).to_dict())

        results = test_pair(gene, pos1, pos2, gpi, clin, annot_cache,
                            pos_quality, patient_proj)
        for r in results:
            r["n_both_total"] = n_both
            r["co_occ_OE"] = oe_values.get((gene, pos1, pos2), np.nan)
            r["co_occ_fisher_p"] = fisher_pvals.get((gene, pos1, pos2), np.nan)
            r["is_driver"] = gene in driver_genes
            all_results.append(r)
            if r["track"] == "pan-cancer":
                n_pan += 1
            else:
                n_specific += 1

    df = pd.DataFrame(all_results)
    logger.info("Total tests: %d (pan=%d, specific=%d)", len(df), n_pan, n_specific)

    if len(df) == 0:
        print("No testable pairs found.")
        return

    # ---- FDR ----
    df = apply_fdr(df)

    # ---- Epistasis IDs ----
    df["epistasis_id"] = df.apply(lambda r: format_epistasis_id(r, annot_cache), axis=1)

    # ---- Report ----
    print(f"\n{'='*100}")
    print("RESULTS")
    print(f"{'='*100}")

    for track in ["pan-cancer", "cancer-specific"]:
        sub = df[df["track"] == track]
        n_tests = len(sub)
        print(f"\n--- {track.upper()} ({n_tests} tests) ---")
        print(f"  Nominal p < 0.05:  {(sub['logrank_p'] < 0.05).sum()}")
        print(f"  Nominal p < 0.01:  {(sub['logrank_p'] < 0.01).sum()}")
        print(f"  FDR q < 0.10:      {(sub['fdr_q'] < 0.10).sum()}")
        print(f"  FDR q < 0.05:      {(sub['fdr_q'] < 0.05).sum()}")

        sig = sub[sub["fdr_q"] < FDR_THRESHOLD].sort_values("logrank_p")
        if len(sig) > 0:
            print(f"\n  Top FDR < {FDR_THRESHOLD}:")
            for _, r in sig.head(40).iterrows():
                proj = r["project"] if track == "cancer-specific" else "PAN"
                drv = "*" if r["is_driver"] else " "
                gf = "G" if r["germ_flag1"] or r["germ_flag2"] else " "
                print(
                    f"  {drv}{gf} {r['gene']:>12s} {r['prot1']:>15s}+{r['prot2']:<15s} "
                    f"d={r['dist']:>4d} {proj:>12s} "
                    f"n={r['n_target']:>4d}v{r['n_control']:>4d} "
                    f"surv={r['target_median_yr']:>5.1f}v{r['control_median_yr']:>5.1f}yr "
                    f"p={r['logrank_p']:.2e} q={r['fdr_q']:.3f} {r['direction']:>6s} "
                    f"O/E={r['co_occ_OE']:>5.1f} "
                    f"VAF={r['vaf1']:.2f}/{r['vaf2']:.2f}"
                )

    # ---- Driver gene enrichment ----
    print(f"\n{'='*80}")
    print("DRIVER GENE ENRICHMENT")
    print(f"{'='*80}")
    for track in ["pan-cancer", "cancer-specific"]:
        sub = df[df["track"] == track]
        sig = sub[sub["fdr_q"] < FDR_THRESHOLD]
        if len(sig) == 0:
            continue
        n_driver_sig = sig["is_driver"].sum()
        n_driver_all = sub["is_driver"].sum()
        frac_sig = n_driver_sig / max(len(sig), 1)
        frac_all = n_driver_all / max(len(sub), 1)
        print(f"  {track}: drivers in FDR-sig = {n_driver_sig}/{len(sig)} ({frac_sig:.1%}) "
              f"vs background {n_driver_all}/{len(sub)} ({frac_all:.1%})")

    # ---- Save ----
    out_all = FIG_DIR / "tcga_survival_epistasis_all.csv"
    df.to_csv(out_all, index=False)
    print(f"\nAll results: {len(df)} rows → {out_all}")

    sig_all = df[df["fdr_q"] < FDR_THRESHOLD].sort_values("logrank_p")
    out_sig = FIG_DIR / "tcga_survival_epistasis_significant.csv"
    sig_all.to_csv(out_sig, index=False)
    print(f"FDR-significant: {len(sig_all)} rows → {out_sig}")

    # Pipeline pairs: unique (gene, pos1, pos2) at nominal p < 0.05
    pipeline = (
        df[df["logrank_p"] < 0.05]
        .drop_duplicates(subset=["gene", "pos1", "pos2"])
        .sort_values("logrank_p")
    )
    out_pipe = FIG_DIR / "tcga_survival_pipeline_pairs.csv"
    pipeline.to_csv(out_pipe, index=False)
    print(f"Pipeline pairs (p<0.05): {len(pipeline)} unique → {out_pipe}")

    # ---- Final stats ----
    print(f"\n{'='*100}")
    print("PIPELINE SUMMARY")
    print(f"{'='*100}")
    print(f"  Eligible pairs:           {len(pair_counts)}")
    print(f"  After quality filter:     (germline pairs removed)")
    print(f"  After co-occurrence:      {len(passed)}")
    print(f"  Tests run:                {len(df)} (pan={n_pan}, specific={n_specific})")
    n_fdr = (df["fdr_q"] < FDR_THRESHOLD).sum()
    n_fdr05 = (df["fdr_q"] < 0.05).sum()
    print(f"  FDR < 0.10:               {n_fdr}")
    print(f"  FDR < 0.05:               {n_fdr05}")
    print(f"  Unique genes (FDR<0.10):  {df[df['fdr_q'] < FDR_THRESHOLD]['gene'].nunique()}")
    print(f"  Nominal p < 0.05:         {(df['logrank_p'] < 0.05).sum()}")
    print(f"  Unique pairs (p<0.05):    {len(pipeline)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
