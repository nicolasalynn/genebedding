"""Orchestrate all dataset parsers and produce final output files."""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from .common import (
    DISTANCE_THRESHOLD, OUTPUT_DIR,
    LCL_TISSUE, WB_TISSUE, GTEX_EQTL_DIR,
    FAS_EXCEL, MST1R_LIBRARY_EXCEL, TCGA_PARQUET,
    load_gene_symbol_mapping, lookup_gene_strands,
    lookup_variant_mafs, assign_maf_columns,
    validate_ref_alleles, make_epistasis_id,
    enforce_pos1_lt_pos2,
)
from .parse_uebbing_mpra import parse as parse_mpra
from .parse_yang_evqtl import parse as parse_yang
from .parse_brown_veqtl import parse as parse_brown
from .parse_gtex_null_tier1 import parse as parse_gtex_t1
from .parse_gtex_null_tier2 import parse as parse_gtex_t2
from .parse_fas_exon import parse as parse_fas
from .parse_kras_neighborhood import parse as parse_kras
from .parse_mst1r_splicing import parse as parse_mst1r
from .parse_tcga_doubles import parse as parse_tcga
from .parse_rauscher_cftr import parse as parse_rauscher
from .parse_ke_sl6 import parse as parse_ke_sl6
from .parse_okgp_chr12 import parse as parse_okgp
from .parse_correlated_eqtl import parse as parse_corr_eqtl
from .parse_trna41 import parse as parse_trna41
from .parse_ke_sl6_pairs import parse as parse_ke_sl6_pairs


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Parse each dataset
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Parse datasets")
    print("=" * 60)

    print("\n--- Uebbing MPRA ---")
    mpra_positive, mpra_null = parse_mpra()

    print("\n--- Yang 2016 ---")
    yang_pairs = parse_yang()

    print("\n--- Brown 2014 ---")
    brown_pairs = parse_brown()

    print("\n--- GTEx Null Tier 1 ---")
    gtex_null_tier1 = parse_gtex_t1()

    print("\n--- GTEx Null Tier 2 ---")
    # Build positive_pairs_df for tier 2 matching
    shared_cols = ['source', 'pair_id', 'gene', 'chr', 'pos1', 'pos2',
                   'pos1_hg38', 'pos2_hg38', 'ref1', 'alt1', 'ref2', 'alt2', 'distance_bp']
    pos_dfs = []
    for df in [yang_pairs, brown_pairs]:
        if len(df) > 0:
            present_cols = [c for c in shared_cols if c in df.columns]
            # Ensure snp_id columns if present
            for extra in ['snp_id1', 'snp_id2']:
                if extra in df.columns:
                    present_cols.append(extra)
            pos_dfs.append(df[present_cols])
    positive_pairs_df = pd.concat(pos_dfs, ignore_index=True) if pos_dfs else pd.DataFrame()
    gtex_null_tier2 = parse_gtex_t2(positive_pairs_df=positive_pairs_df)

    # --- New datasets ---
    fas_pairs = pd.DataFrame()
    if FAS_EXCEL.exists():
        print("\n--- FAS exon 3 ---")
        fas_pairs = parse_fas()
    else:
        print(f"\n--- FAS: skipped ({FAS_EXCEL} not found) ---")

    print("\n--- KRAS neighborhood ---")
    kras_pairs = parse_kras()

    mst1r_pairs = pd.DataFrame()
    if MST1R_LIBRARY_EXCEL.exists():
        print("\n--- MST1R splicing ---")
        mst1r_pairs = parse_mst1r()
    else:
        print(f"\n--- MST1R: skipped ({MST1R_LIBRARY_EXCEL} not found) ---")

    tcga_pairs = pd.DataFrame()
    if TCGA_PARQUET.exists():
        print("\n--- TCGA doubles ---")
        tcga_pairs = parse_tcga()
    else:
        print(f"\n--- TCGA: skipped ({TCGA_PARQUET} not found) ---")

    print("\n--- OKGP chr12 ---")
    okgp_pairs = parse_okgp()

    print("\n--- Correlated eQTL (LD null) ---")
    corr_eqtl_pairs = parse_corr_eqtl()

    print("\n--- tRNA41 (structural ground truth) ---")
    trna41_singles, trna41_pairs, trna41_contact_map = parse_trna41()
    # Save contact map and singles
    import numpy as np
    cmap_positions = list(range(159_141_611, 159_141_684 + 1))
    cmap_df = pd.DataFrame(trna41_contact_map, index=cmap_positions, columns=cmap_positions)
    cmap_path = OUTPUT_DIR / 'trna41_contact_map.tsv'
    cmap_df.to_csv(cmap_path, sep='\t')
    print(f"  Saved contact map to {cmap_path}")
    trna41_singles_path = OUTPUT_DIR / 'trna41_singles.tsv'
    trna41_singles.to_csv(trna41_singles_path, sep='\t', index=False)
    print(f"  Saved {len(trna41_singles)} singles to {trna41_singles_path}")

    print("\n--- Rauscher CFTR (negative control) ---")
    rauscher_pairs = parse_rauscher()

    print("\n--- Ke SL6 (separate analysis) ---")
    ke_sl6 = parse_ke_sl6()
    ke_sl6_path = OUTPUT_DIR / 'ke_sl6_analysis.tsv'
    ke_sl6.to_csv(ke_sl6_path, sep='\t', index=False)
    print(f"Saved {ke_sl6_path}: {len(ke_sl6)} rows (standalone, not in combined)")

    print("\n--- Ke SL6 variant pairs (structural ground truth) ---")
    ke_sl6_singles, ke_sl6_pairs_df, ke_sl6_cmap = parse_ke_sl6_pairs()
    ke_sl6_singles_path = OUTPUT_DIR / 'ke_sl6_singles.tsv'
    ke_sl6_singles.to_csv(ke_sl6_singles_path, sep='\t', index=False)
    print(f"  Saved {len(ke_sl6_singles)} singles to {ke_sl6_singles_path}")
    ke_sl6_pairs_path = OUTPUT_DIR / 'ke_sl6_pairs.tsv'
    ke_sl6_pairs_df.to_csv(ke_sl6_pairs_path, sep='\t', index=False)
    print(f"  Saved {len(ke_sl6_pairs_df):,} doubles to {ke_sl6_pairs_path}")
    ke_sl6_cmap_path = OUTPUT_DIR / 'ke_sl6_contact_map.tsv'
    ke_sl6_cmap_df = pd.DataFrame(ke_sl6_cmap, index=range(90), columns=range(90))
    ke_sl6_cmap_df.to_csv(ke_sl6_cmap_path, sep='\t')
    print(f"  Saved contact map (90x90) to {ke_sl6_cmap_path}")
    ke_sl6_seq_path = OUTPUT_DIR / 'ke_sl6_sequence.txt'
    with open(ke_sl6_seq_path, 'w') as f:
        f.write('CCCCACCTCTTCTTCTTTTCTAGAGTTGCTGCTGGGAGCTCCAGCACAGTGAAATGGACAGAAGGGCAGAGCAAGTGAGTGGACAATGCG\n')
    print(f"  Saved WT sequence to {ke_sl6_seq_path}")

    # ------------------------------------------------------------------
    # 2. Collect gene symbols and batch strand lookup
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Gene strand lookup")
    print("=" * 60)

    _, ensembl_to_symbol = load_gene_symbol_mapping()

    all_gene_symbols = set()
    all_gene_symbols.update(yang_pairs['gene'].dropna().unique())
    all_gene_symbols.update(brown_pairs['gene'].dropna().unique())

    # GTEx T1: map Ensembl IDs to gene symbols
    if len(gtex_null_tier1) > 0:
        t1_symbols = {ensembl_to_symbol[g]
                      for g in gtex_null_tier1['gene_id'].unique()
                      if g in ensembl_to_symbol}
        all_gene_symbols.update(t1_symbols)
        # Add gene_name column
        gtex_null_tier1['gene_name'] = gtex_null_tier1['gene_id'].map(ensembl_to_symbol)

    if len(gtex_null_tier2) > 0:
        all_gene_symbols.update(gtex_null_tier2['gene'].dropna().unique())

    # New datasets: collect gene symbols for strand lookup
    if len(fas_pairs) > 0:
        all_gene_symbols.update(fas_pairs['gene'].dropna().unique())
    if len(mst1r_pairs) > 0:
        all_gene_symbols.update(mst1r_pairs['gene'].dropna().unique())
    if len(tcga_pairs) > 0:
        all_gene_symbols.update(tcga_pairs['gene'].dropna().unique())
        # Also include dependent_gene if different from anchor
        if 'dependent_gene' in tcga_pairs.columns:
            all_gene_symbols.update(tcga_pairs['dependent_gene'].dropna().unique())
    if len(rauscher_pairs) > 0:
        all_gene_symbols.update(rauscher_pairs['gene'].dropna().unique())
    if len(okgp_pairs) > 0:
        okgp_genes = set(okgp_pairs['gene'].dropna().unique()) - {'GENE'}
        all_gene_symbols.update(okgp_genes)

    print(f"Total unique gene symbols to look up: {len(all_gene_symbols)}")
    gene_strand = lookup_gene_strands(all_gene_symbols)
    print(f"Strands resolved: {len(gene_strand)} / {len(all_gene_symbols)}")
    print(f"  Forward (P): {sum(1 for s in gene_strand.values() if s == 'P')}")
    print(f"  Reverse (N): {sum(1 for s in gene_strand.values() if s == 'N')}")

    # ------------------------------------------------------------------
    # 2.5. Enforce pos1 <= pos2 ordering
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2.5: Enforce pos1 <= pos2 ordering")
    print("=" * 60)

    # hg19+hg38 datasets: order by hg38 positions
    if len(mpra_positive) > 0:
        print("MPRA positive:")
        mpra_positive = enforce_pos1_lt_pos2(mpra_positive, pos1_col='pos1_hg38', pos2_col='pos2_hg38')
    if len(mpra_null) > 0:
        print("MPRA null:")
        mpra_null = enforce_pos1_lt_pos2(mpra_null, pos1_col='pos1_hg38', pos2_col='pos2_hg38')
    if len(yang_pairs) > 0:
        print("Yang:")
        yang_pairs = enforce_pos1_lt_pos2(yang_pairs, pos1_col='pos1_hg38', pos2_col='pos2_hg38')
    if len(brown_pairs) > 0:
        print("Brown:")
        brown_pairs = enforce_pos1_lt_pos2(brown_pairs, pos1_col='pos1_hg38', pos2_col='pos2_hg38')
    if len(mst1r_pairs) > 0:
        print("MST1R:")
        mst1r_pairs = enforce_pos1_lt_pos2(mst1r_pairs, pos1_col='pos1_hg38', pos2_col='pos2_hg38')

    # hg38-native datasets: order by pos1/pos2
    if len(gtex_null_tier1) > 0:
        print("GTEx null T1:")
        gtex_null_tier1 = enforce_pos1_lt_pos2(gtex_null_tier1, pos1_col='pos1', pos2_col='pos2')
    if len(gtex_null_tier2) > 0:
        print("GTEx null T2:")
        gtex_null_tier2 = enforce_pos1_lt_pos2(gtex_null_tier2, pos1_col='pos1', pos2_col='pos2')
    if len(fas_pairs) > 0:
        print("FAS:")
        fas_pairs = enforce_pos1_lt_pos2(fas_pairs, pos1_col='pos1', pos2_col='pos2')
    if len(kras_pairs) > 0:
        print("KRAS:")
        kras_pairs = enforce_pos1_lt_pos2(kras_pairs, pos1_col='pos1', pos2_col='pos2')
    if len(tcga_pairs) > 0:
        print("TCGA:")
        tcga_pairs = enforce_pos1_lt_pos2(tcga_pairs, pos1_col='pos1', pos2_col='pos2')
    if len(rauscher_pairs) > 0:
        print("Rauscher:")
        rauscher_pairs = enforce_pos1_lt_pos2(rauscher_pairs, pos1_col='pos1', pos2_col='pos2')
    if len(okgp_pairs) > 0:
        print("OKGP:")
        okgp_pairs = enforce_pos1_lt_pos2(okgp_pairs, pos1_col='pos1', pos2_col='pos2')
    if len(corr_eqtl_pairs) > 0:
        print("Correlated eQTL:")
        corr_eqtl_pairs = enforce_pos1_lt_pos2(corr_eqtl_pairs, pos1_col='pos1', pos2_col='pos2')
    if len(trna41_pairs) > 0:
        print("tRNA41:")
        trna41_pairs = enforce_pos1_lt_pos2(trna41_pairs, pos1_col='pos1', pos2_col='pos2')

    # ------------------------------------------------------------------
    # 3. Validate ref alleles against hg38 FASTA
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Validate reference alleles against hg38 FASTA")
    print("=" * 60)

    # MPRA, Yang, Brown: hg38 positions in pos1_hg38, pos2_hg38
    if len(mpra_positive) > 0:
        validate_ref_alleles(mpra_positive, 'MPRA positive', 'chr', 'pos1_hg38', 'pos2_hg38', 'ref1', 'ref2')
    if len(mpra_null) > 0:
        validate_ref_alleles(mpra_null, 'MPRA null', 'chr', 'pos1_hg38', 'pos2_hg38', 'ref1', 'ref2')
    if len(yang_pairs) > 0:
        validate_ref_alleles(yang_pairs, 'Yang', 'chr', 'pos1_hg38', 'pos2_hg38', 'ref1', 'ref2')
    if len(brown_pairs) > 0:
        validate_ref_alleles(brown_pairs, 'Brown', 'chr', 'pos1_hg38', 'pos2_hg38', 'ref1', 'ref2')

    # GTEx: native hg38 in pos1, pos2
    if len(gtex_null_tier1) > 0:
        validate_ref_alleles(gtex_null_tier1, 'GTEx null T1', 'chr', 'pos1', 'pos2', 'ref1', 'ref2')
    if len(gtex_null_tier2) > 0:
        validate_ref_alleles(gtex_null_tier2, 'GTEx null T2', 'chr', 'pos1', 'pos2', 'ref1', 'ref2')

    # FAS: native hg38 in pos1, pos2
    if len(fas_pairs) > 0:
        validate_ref_alleles(fas_pairs, 'FAS', 'chr', 'pos1', 'pos2', 'ref1', 'ref2')

    # KRAS: skip ref validation (synthetic from FASTA, already correct)

    # MST1R: hg38 positions in pos1_hg38, pos2_hg38
    if len(mst1r_pairs) > 0:
        validate_ref_alleles(mst1r_pairs, 'MST1R', 'chr', 'pos1_hg38', 'pos2_hg38', 'ref1', 'ref2')

    # TCGA: native hg38 in pos1, pos2
    if len(tcga_pairs) > 0:
        validate_ref_alleles(tcga_pairs, 'TCGA', 'chr', 'pos1', 'pos2', 'ref1', 'ref2')

    # Rauscher: native hg38 in pos1, pos2
    if len(rauscher_pairs) > 0:
        validate_ref_alleles(rauscher_pairs, 'Rauscher CFTR', 'chr', 'pos1', 'pos2', 'ref1', 'ref2')

    # OKGP: native hg38 in pos1, pos2
    if len(okgp_pairs) > 0:
        validate_ref_alleles(okgp_pairs, 'OKGP chr12', 'chr', 'pos1', 'pos2', 'ref1', 'ref2')

    # Correlated eQTL: native hg38 in pos1, pos2
    if len(corr_eqtl_pairs) > 0:
        validate_ref_alleles(corr_eqtl_pairs, 'Correlated eQTL', 'chr', 'pos1', 'pos2', 'ref1', 'ref2')

    # ------------------------------------------------------------------
    # 3.5. Variant MAF lookup (SKIPPED — run separately later)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3.5: Variant MAF lookup — SKIPPED (add later)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 4. Build epistasis_ids (after validation so swaps are captured)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Build epistasis IDs")
    print("=" * 60)

    if len(mpra_positive) > 0:
        mpra_positive['epistasis_id'] = mpra_positive.apply(
            lambda r: make_epistasis_id(r, None, 'chr', 'pos1_hg38', 'pos2_hg38',
                                         'ref1', 'alt1', 'ref2', 'alt2', {}), axis=1)
        print(f"MPRA positive: {mpra_positive['epistasis_id'].notna().sum()} / {len(mpra_positive)}")

    if len(mpra_null) > 0:
        mpra_null['epistasis_id'] = mpra_null.apply(
            lambda r: make_epistasis_id(r, None, 'chr', 'pos1_hg38', 'pos2_hg38',
                                         'ref1', 'alt1', 'ref2', 'alt2', {}), axis=1)
        print(f"MPRA null: {mpra_null['epistasis_id'].notna().sum()} / {len(mpra_null)}")

    if len(yang_pairs) > 0:
        yang_pairs['epistasis_id'] = yang_pairs.apply(
            lambda r: make_epistasis_id(r, 'gene', 'chr', 'pos1_hg38', 'pos2_hg38',
                                         'ref1', 'alt1', 'ref2', 'alt2', gene_strand), axis=1)
        print(f"Yang: {yang_pairs['epistasis_id'].notna().sum()} / {len(yang_pairs)}")

    if len(brown_pairs) > 0:
        brown_pairs['epistasis_id'] = brown_pairs.apply(
            lambda r: make_epistasis_id(r, 'gene', 'chr', 'pos1_hg38', 'pos2_hg38',
                                         'ref1', 'alt1', 'ref2', 'alt2', gene_strand), axis=1)
        print(f"Brown: {brown_pairs['epistasis_id'].notna().sum()} / {len(brown_pairs)}")

    if len(gtex_null_tier1) > 0:
        gtex_null_tier1['epistasis_id'] = gtex_null_tier1.apply(
            lambda r: make_epistasis_id(r, 'gene_name', 'chr', 'pos1', 'pos2',
                                         'ref1', 'alt1', 'ref2', 'alt2', gene_strand), axis=1)
        print(f"GTEx T1: {gtex_null_tier1['epistasis_id'].notna().sum()} / {len(gtex_null_tier1)}")

    if len(gtex_null_tier2) > 0:
        gtex_null_tier2['epistasis_id'] = gtex_null_tier2.apply(
            lambda r: make_epistasis_id(r, 'gene', 'chr', 'pos1', 'pos2',
                                         'ref1', 'alt1', 'ref2', 'alt2', gene_strand), axis=1)
        print(f"GTEx T2: {gtex_null_tier2['epistasis_id'].notna().sum()} / {len(gtex_null_tier2)}")

    # FAS: epistasis_id already built in parser, but rebuild after ref validation
    if len(fas_pairs) > 0:
        fas_pairs['epistasis_id'] = fas_pairs.apply(
            lambda r: make_epistasis_id(r, 'gene', 'chr', 'pos1', 'pos2',
                                         'ref1', 'alt1', 'ref2', 'alt2', gene_strand), axis=1)
        print(f"FAS: {fas_pairs['epistasis_id'].notna().sum()} / {len(fas_pairs)}")

    # KRAS: epistasis_id already built in parser (synthetic, no validation changes)
    if len(kras_pairs) > 0:
        print(f"KRAS: {kras_pairs['epistasis_id'].notna().sum()} / {len(kras_pairs)} (built in parser)")

    # MST1R: build epistasis_id using hg38 positions
    if len(mst1r_pairs) > 0:
        mst1r_pairs['epistasis_id'] = mst1r_pairs.apply(
            lambda r: make_epistasis_id(r, 'gene', 'chr', 'pos1_hg38', 'pos2_hg38',
                                         'ref1', 'alt1', 'ref2', 'alt2', gene_strand), axis=1)
        print(f"MST1R: {mst1r_pairs['epistasis_id'].notna().sum()} / {len(mst1r_pairs)}")

    # TCGA: rebuild epistasis_id with strand (ParseTCGA format omits strand)
    if len(tcga_pairs) > 0:
        tcga_pairs['epistasis_id'] = tcga_pairs.apply(
            lambda r: make_epistasis_id(r, 'gene', 'chr', 'pos1', 'pos2',
                                         'ref1', 'alt1', 'ref2', 'alt2', gene_strand), axis=1)
        print(f"TCGA: {tcga_pairs['epistasis_id'].notna().sum()} / {len(tcga_pairs)}")

    # Correlated eQTL: build epistasis_id (source has 5-field IDs without strand)
    if len(corr_eqtl_pairs) > 0:
        corr_eqtl_pairs['epistasis_id'] = corr_eqtl_pairs.apply(
            lambda r: make_epistasis_id(r, 'gene', 'chr', 'pos1', 'pos2',
                                         'ref1', 'alt1', 'ref2', 'alt2', gene_strand), axis=1)
        print(f"Correlated eQTL: {corr_eqtl_pairs['epistasis_id'].notna().sum()} / {len(corr_eqtl_pairs)}")

    # OKGP: rebuild epistasis_id after ref validation
    if len(okgp_pairs) > 0:
        okgp_pairs['epistasis_id'] = okgp_pairs.apply(
            lambda r: make_epistasis_id(r, 'gene', 'chr', 'pos1', 'pos2',
                                         'ref1', 'alt1', 'ref2', 'alt2', gene_strand), axis=1)
        print(f"OKGP: {okgp_pairs['epistasis_id'].notna().sum()} / {len(okgp_pairs)}")

    # Rauscher CFTR: build epistasis_id (hg38 native)
    if len(rauscher_pairs) > 0:
        rauscher_pairs['epistasis_id'] = rauscher_pairs.apply(
            lambda r: make_epistasis_id(r, 'gene', 'chr', 'pos1', 'pos2',
                                         'ref1', 'alt1', 'ref2', 'alt2', gene_strand), axis=1)
        print(f"Rauscher: {rauscher_pairs['epistasis_id'].notna().sum()} / {len(rauscher_pairs)}")

    # ------------------------------------------------------------------
    # 5. Export per-dataset TSVs
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Export files")
    print("=" * 60)

    # MPRA positive
    mpra_pos_cols = [
        'source', 'pair_id', 'enhancer', 'chr', 'pos1', 'pos2', 'pos1_hg38', 'pos2_hg38',
        'ref1', 'alt1', 'ref2', 'alt2', 'distance_bp', 'distance_hg38', 'label', 'pair_ambiguity',
        'effect_size_1', 'effect_size_2', 'interaction_effect_size', 'maf1', 'maf2', 'genome_build', 'epistasis_id',
    ]
    for col in mpra_pos_cols:
        if col not in mpra_positive.columns:
            mpra_positive[col] = pd.NA
    mpra_positive_export = mpra_positive[[c for c in mpra_pos_cols if c in mpra_positive.columns]].copy()

    # MPRA null
    for col in mpra_pos_cols:
        if col not in mpra_null.columns:
            mpra_null[col] = pd.NA
    mpra_null_export = mpra_null[[c for c in mpra_pos_cols if c in mpra_null.columns]].copy()

    # Yang
    yang_cols = [
        'source', 'pair_id', 'gene', 'chr', 'pos1', 'pos2', 'pos1_hg38', 'pos2_hg38',
        'snp_id1', 'snp_id2', 'ref1', 'alt1', 'ref2', 'alt2', 'distance_bp', 'distance_hg38',
        'label', 'p_value', 'effect_size', 'maf1', 'maf2', 'genome_build', 'epistasis_id',
    ]
    for col in yang_cols:
        if col not in yang_pairs.columns:
            yang_pairs[col] = pd.NA
    yang_export = yang_pairs[[c for c in yang_cols if c in yang_pairs.columns]].copy()

    # Brown
    brown_cols = [
        'source', 'pair_id', 'gene', 'chr', 'pos1', 'pos2', 'pos1_hg38', 'pos2_hg38',
        'snp_id1', 'snp_id2', 'ref1', 'alt1', 'ref2', 'alt2', 'distance_bp', 'distance_hg38',
        'label', 'p_value', 'effect_size', 'replicated', 'maf1', 'maf2', 'genome_build', 'epistasis_id',
    ]
    for col in brown_cols:
        if col not in brown_pairs.columns:
            brown_pairs[col] = pd.NA
    brown_export = brown_pairs[[c for c in brown_cols if c in brown_pairs.columns]].copy()

    # GTEx null tier 1
    gtex_null1_cols = [
        'source', 'pair_id', 'gene_id', 'gene_name', 'tissue',
        'chr', 'pos1', 'pos2', 'variant_id1', 'variant_id2', 'ref1', 'alt1', 'ref2', 'alt2',
        'distance_bp', 'rank1', 'rank2', 'beta1', 'beta2', 'pval1', 'pval2',
        'label', 'maf1', 'maf2', 'genome_build', 'epistasis_id',
    ]
    gtex_null1_export = (gtex_null_tier1[[c for c in gtex_null1_cols if c in gtex_null_tier1.columns]].copy()
                         if len(gtex_null_tier1) > 0
                         else pd.DataFrame(columns=gtex_null1_cols))

    # GTEx null tier 2
    gtex_null2_cols = [
        'source', 'pair_id', 'gene', 'positive_pair_id',
        'chr', 'pos1', 'pos2', 'ref1', 'alt1', 'ref2', 'alt2',
        'variant_id1', 'variant_id2', 'distance_bp',
        'label', 'maf1', 'maf2', 'genome_build', 'epistasis_id',
    ]
    gtex_null2_export = (gtex_null_tier2[[c for c in gtex_null2_cols if c in gtex_null_tier2.columns]].copy()
                         if len(gtex_null_tier2) > 0
                         else pd.DataFrame(columns=gtex_null2_cols))

    # FAS
    fas_cols = [
        'source', 'pair_id', 'gene', 'chr', 'pos1', 'pos2',
        'ref1', 'alt1', 'ref2', 'alt2', 'distance_bp', 'label',
        'score1', 'score2', 'experimental_score', 'expected_experimental_score',
        'empirical_epistasis', 'empirical_pval', 'CategoryA', 'CategoryB',
        'maf1', 'maf2', 'genome_build', 'epistasis_id',
    ]
    fas_export = (fas_pairs[[c for c in fas_cols if c in fas_pairs.columns]].copy()
                  if len(fas_pairs) > 0
                  else pd.DataFrame(columns=fas_cols))

    # KRAS
    kras_cols = [
        'source', 'pair_id', 'gene', 'chr', 'pos1', 'pos2',
        'ref1', 'alt1', 'ref2', 'alt2', 'distance_bp', 'label',
        'maf1', 'maf2', 'genome_build', 'epistasis_id',
    ]
    kras_export = (kras_pairs[[c for c in kras_cols if c in kras_pairs.columns]].copy()
                   if len(kras_pairs) > 0
                   else pd.DataFrame(columns=kras_cols))

    # MST1R
    mst1r_cols = [
        'source', 'pair_id', 'gene', 'chr', 'pos1', 'pos2', 'pos1_hg38', 'pos2_hg38',
        'ref1', 'alt1', 'ref2', 'alt2', 'distance_bp', 'label',
        'ae_inclusion_pct', 'ae_skipping_pct',
        'maf1', 'maf2', 'genome_build', 'epistasis_id',
    ]
    mst1r_export = (mst1r_pairs[[c for c in mst1r_cols if c in mst1r_pairs.columns]].copy()
                    if len(mst1r_pairs) > 0
                    else pd.DataFrame(columns=mst1r_cols))

    # TCGA
    tcga_cols = [
        'source', 'pair_id', 'gene', 'chr', 'pos1', 'pos2',
        'ref1', 'alt1', 'ref2', 'alt2', 'distance_bp', 'label',
        'n_cases_both', 'cond_prob_dependent', 'anchor_gene', 'dependent_gene',
        'maf1', 'maf2', 'genome_build', 'epistasis_id',
    ]
    tcga_export = (tcga_pairs[[c for c in tcga_cols if c in tcga_pairs.columns]].copy()
                   if len(tcga_pairs) > 0
                   else pd.DataFrame(columns=tcga_cols))

    # Correlated eQTL
    corr_eqtl_cols = [
        'source', 'pair_id', 'gene', 'chr', 'pos1', 'pos2',
        'ref1', 'alt1', 'ref2', 'alt2', 'distance_bp', 'label',
        'R2', 'Dprime',
        'maf1', 'maf2', 'genome_build', 'epistasis_id',
    ]
    corr_eqtl_export = (corr_eqtl_pairs[[c for c in corr_eqtl_cols if c in corr_eqtl_pairs.columns]].copy()
                         if len(corr_eqtl_pairs) > 0
                         else pd.DataFrame(columns=corr_eqtl_cols))

    # OKGP chr12
    okgp_cols = [
        'source', 'pair_id', 'gene', 'chr', 'pos1', 'pos2',
        'ref1', 'alt1', 'ref2', 'alt2', 'distance_bp', 'label',
        'AF1', 'AF2', 'n_carriers1', 'n_carriers2', 'n_carriers_both',
        'pair_type', 'weight',
        'maf1', 'maf2', 'genome_build', 'epistasis_id',
    ]
    okgp_export = (okgp_pairs[[c for c in okgp_cols if c in okgp_pairs.columns]].copy()
                   if len(okgp_pairs) > 0
                   else pd.DataFrame(columns=okgp_cols))

    # tRNA41
    trna41_cols = [
        'source', 'pair_id', 'gene', 'chr', 'pos1', 'pos2',
        'ref1', 'alt1', 'ref2', 'alt2', 'distance_bp', 'label',
        'base_paired',
        'maf1', 'maf2', 'genome_build', 'epistasis_id',
    ]
    trna41_export = (trna41_pairs[[c for c in trna41_cols if c in trna41_pairs.columns]].copy()
                     if len(trna41_pairs) > 0
                     else pd.DataFrame(columns=trna41_cols))

    # Rauscher CFTR
    rauscher_cols = [
        'source', 'pair_id', 'gene', 'chr', 'pos1', 'pos2',
        'ref1', 'alt1', 'ref2', 'alt2', 'distance_bp', 'label',
        'missense_name', 'cdna_change',
        'maf1', 'maf2', 'genome_build', 'epistasis_id',
    ]
    rauscher_export = (rauscher_pairs[[c for c in rauscher_cols if c in rauscher_pairs.columns]].copy()
                       if len(rauscher_pairs) > 0
                       else pd.DataFrame(columns=rauscher_cols))

    # Save per-dataset files
    for name, df in [('mpra_positive_pairs', mpra_positive_export),
                      ('mpra_null_pairs', mpra_null_export),
                      ('yang_positive_pairs', yang_export),
                      ('brown_positive_pairs', brown_export),
                      ('gtex_null_tier1_pairs', gtex_null1_export),
                      ('gtex_null_tier2_pairs', gtex_null2_export),
                      ('fas_pairs', fas_export),
                      ('kras_neighborhood_pairs', kras_export),
                      ('mst1r_splicing_pairs', mst1r_export),
                      ('tcga_doubles_pairs', tcga_export),
                      ('trna41_pairs', trna41_export),
                      ('rauscher_cftr_pairs', rauscher_export),
                      ('okgp_chr12_pairs', okgp_export),
                      ('correlated_eqtl_pairs', corr_eqtl_export)]:
        fpath = OUTPUT_DIR / f'{name}.tsv'
        df.to_csv(fpath, sep='\t', index=False)
        print(f"Saved {fpath}: {len(df):,} rows")

    # ------------------------------------------------------------------
    # 6. Combined file
    # ------------------------------------------------------------------
    combined_dfs = [df for df in [mpra_positive_export, mpra_null_export, yang_export,
                                   brown_export, gtex_null1_export, gtex_null2_export,
                                   fas_export, kras_export, mst1r_export, tcga_export,
                                   trna41_export, rauscher_export, okgp_export, corr_eqtl_export]
                    if len(df) > 0]
    all_combined = pd.concat(combined_dfs, ignore_index=True, sort=False)
    combined_path = OUTPUT_DIR / 'all_pairs_combined.tsv'
    all_combined.to_csv(combined_path, sep='\t', index=False)
    print(f"\nSaved {combined_path}: {len(all_combined):,} rows")
    print(f"Rows by source:\n{all_combined['source'].value_counts()}")
    print(f"Rows by label:\n{all_combined['label'].value_counts()}")

    # ------------------------------------------------------------------
    # 7. TopLD export
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: TopLD export")
    print("=" * 60)

    variants = set()
    pairs_list = []
    hg38_info = [
        (mpra_positive_export, 'chr', 'pos1_hg38', 'pos2_hg38', None, None, 'pair_id'),
        (mpra_null_export, 'chr', 'pos1_hg38', 'pos2_hg38', None, None, 'pair_id'),
        (yang_export, 'chr', 'pos1_hg38', 'pos2_hg38', 'snp_id1', 'snp_id2', 'pair_id'),
        (brown_export, 'chr', 'pos1_hg38', 'pos2_hg38', 'snp_id1', 'snp_id2', 'pair_id'),
        (gtex_null1_export, 'chr', 'pos1', 'pos2', 'variant_id1', 'variant_id2', 'pair_id'),
        (gtex_null2_export, 'chr', 'pos1', 'pos2', 'variant_id1', 'variant_id2', 'pair_id'),
        # FAS: hg38 native, pos in pos1/pos2
        (fas_export, 'chr', 'pos1', 'pos2', None, None, 'pair_id'),
        # KRAS: hg38 native
        (kras_export, 'chr', 'pos1', 'pos2', None, None, 'pair_id'),
        # MST1R: liftover'd hg38 in pos1_hg38/pos2_hg38
        (mst1r_export, 'chr', 'pos1_hg38', 'pos2_hg38', None, None, 'pair_id'),
        # TCGA: hg38 native
        (tcga_export, 'chr', 'pos1', 'pos2', None, None, 'pair_id'),
        # tRNA41: hg38 native
        (trna41_export, 'chr', 'pos1', 'pos2', None, None, 'pair_id'),
        # Rauscher: hg38 native
        (rauscher_export, 'chr', 'pos1', 'pos2', None, None, 'pair_id'),
        # OKGP: hg38 native
        (okgp_export, 'chr', 'pos1', 'pos2', None, None, 'pair_id'),
        # Correlated eQTL: hg38 native
        (corr_eqtl_export, 'chr', 'pos1', 'pos2', None, None, 'pair_id'),
    ]
    for df, chr_col, pos1_col, pos2_col, id1_col, id2_col, pair_id_col in hg38_info:
        for _, row in df.iterrows():
            chr_val = str(row[chr_col]) if pd.notna(row.get(chr_col)) else None
            p1 = row.get(pos1_col)
            p2 = row.get(pos2_col)
            p1 = p1 if pd.notna(p1) else None
            p2 = p2 if pd.notna(p2) else None
            if chr_val and p1 is not None:
                variants.add((chr_val, int(p1)))
            if chr_val and p2 is not None:
                variants.add((chr_val, int(p2)))
            if chr_val and p1 is not None and p2 is not None:
                pairs_list.append({
                    'chr': chr_val, 'pos1': int(p1), 'pos2': int(p2),
                    'snp_id1': row.get(id1_col, pd.NA) if id1_col else pd.NA,
                    'snp_id2': row.get(id2_col, pd.NA) if id2_col else pd.NA,
                    'pair_id': row[pair_id_col],
                })

    hg38_var_df = pd.DataFrame(sorted(variants), columns=['chr', 'pos'])
    hg38_pairs_df = (pd.DataFrame(pairs_list) if pairs_list
                     else pd.DataFrame(columns=['chr', 'pos1', 'pos2', 'snp_id1', 'snp_id2', 'pair_id']))

    hg38_var_path = OUTPUT_DIR / 'variants_for_topld_hg38.tsv'
    hg38_var_df.to_csv(hg38_var_path, sep='\t', index=False)
    print(f"Saved {hg38_var_path}: {len(hg38_var_df):,} unique variants")

    hg38_pairs_path = OUTPUT_DIR / 'pairs_for_topld_hg38.tsv'
    hg38_pairs_df.to_csv(hg38_pairs_path, sep='\t', index=False)
    print(f"Saved {hg38_pairs_path}: {len(hg38_pairs_df):,} pairs")

    # ------------------------------------------------------------------
    # 8. Distance distribution plot
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: Distance distribution plot")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(10, 5))
    datasets_to_plot = []
    if len(mpra_positive) > 0:
        datasets_to_plot.append(('MPRA positive', mpra_positive['distance_bp'].dropna(), 'red'))
    if len(yang_pairs) > 0:
        datasets_to_plot.append(('Yang positive', yang_pairs['distance_hg38'].dropna(), 'blue'))
    if len(brown_pairs) > 0:
        datasets_to_plot.append(('Brown positive', brown_pairs['distance_hg38'].dropna(), 'green'))
    if len(gtex_null_tier1) > 0:
        lcl_mask = gtex_null_tier1['tissue'] == LCL_TISSUE
        wb_mask = gtex_null_tier1['tissue'] == WB_TISSUE
        datasets_to_plot.append(('GTEx null T1 (LCL)', gtex_null_tier1.loc[lcl_mask, 'distance_bp'].dropna(), 'orange'))
        datasets_to_plot.append(('GTEx null T1 (WB)', gtex_null_tier1.loc[wb_mask, 'distance_bp'].dropna(), 'goldenrod'))
    if len(gtex_null_tier2) > 0:
        datasets_to_plot.append(('GTEx null T2', gtex_null_tier2['distance_bp'].dropna(), 'purple'))
    if len(fas_pairs) > 0:
        datasets_to_plot.append(('FAS exon', fas_pairs['distance_bp'].dropna(), 'cyan'))
    if len(kras_pairs) > 0:
        datasets_to_plot.append(('KRAS neighborhood', kras_pairs['distance_bp'].dropna(), 'magenta'))
    if len(mst1r_pairs) > 0:
        datasets_to_plot.append(('MST1R splicing', mst1r_pairs['distance_bp'].dropna(), 'brown'))
    if len(tcga_pairs) > 0:
        datasets_to_plot.append(('TCGA doubles', tcga_pairs['distance_bp'].dropna(), 'gray'))
    if len(trna41_pairs) > 0:
        datasets_to_plot.append(('tRNA41', trna41_pairs['distance_bp'].dropna(), 'salmon'))
    if len(rauscher_pairs) > 0:
        datasets_to_plot.append(('Rauscher CFTR', rauscher_pairs['distance_bp'].dropna(), 'darkgreen'))
    if len(okgp_pairs) > 0:
        datasets_to_plot.append(('OKGP chr12', okgp_pairs['distance_bp'].dropna(), 'navy'))
    if len(corr_eqtl_pairs) > 0:
        datasets_to_plot.append(('Correlated eQTL', corr_eqtl_pairs['distance_bp'].dropna(), 'teal'))

    for label, data, color in datasets_to_plot:
        if len(data) > 0:
            ax.hist(data / 1000, bins=30, alpha=0.4, label=f'{label} (n={len(data):,})', color=color)

    ax.axvline(x=DISTANCE_THRESHOLD / 1000, color='black', linestyle='--', alpha=0.5,
               label=f'Threshold ({DISTANCE_THRESHOLD/1000:.0f} kb)')
    ax.set_xlabel('Distance (kb)')
    ax.set_ylabel('Count')
    ax.set_title(f'Distance Distribution — All Pairs (≤{DISTANCE_THRESHOLD/1000:.0f} kb)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / 'distance_distributions.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved {plot_path}")
    plt.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  MPRA positive: {len(mpra_positive_export)}")
    print(f"  MPRA null: {len(mpra_null_export)}")
    print(f"  Yang positive: {len(yang_export)}")
    print(f"  Brown positive: {len(brown_export)}")
    print(f"  GTEx null T1: {len(gtex_null1_export)}")
    print(f"  GTEx null T2: {len(gtex_null2_export)}")
    print(f"  FAS: {len(fas_export)}")
    print(f"  KRAS: {len(kras_export)}")
    print(f"  MST1R: {len(mst1r_export)}")
    print(f"  TCGA: {len(tcga_export)}")
    print(f"  tRNA41: {len(trna41_export)}")
    print(f"  Rauscher CFTR: {len(rauscher_export)}")
    print(f"  OKGP chr12: {len(okgp_export)}")
    print(f"  Correlated eQTL: {len(corr_eqtl_export)}")
    print(f"  Ke SL6 analysis (separate): {len(ke_sl6)}")
    print(f"  Ke SL6 singles: {len(ke_sl6_singles)}")
    print(f"  Ke SL6 pairs: {len(ke_sl6_pairs_df):,}")
    print(f"  tRNA41 singles: {len(trna41_singles)}")
    print(f"  Combined: {len(all_combined)}")
    print("\nDone.")


if __name__ == '__main__':
    main()
