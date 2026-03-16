"""Standalone script to add maf1/maf2 columns to parsed_pairs TSVs.

Reads existing output TSVs, collects unique variants, queries Ensembl VEP
for gnomAD genome allele frequencies, and patches maf1/maf2 into each file.

Uses a JSON cache at data/variant_maf_cache.json so subsequent runs are instant.

Usage: python -m scripts.add_maf
"""

import numpy as np
import pandas as pd

from .common import OUTPUT_DIR, lookup_variant_mafs, assign_maf_columns

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
# Liftover datasets: hg38 positions in pos1_hg38/pos2_hg38
LIFTOVER_DATASETS = [
    'mpra_positive_pairs',
    'mpra_null_pairs',
    'yang_positive_pairs',
    'brown_positive_pairs',
    'mst1r_splicing_pairs',
]

# Native hg38 datasets: hg38 positions in pos1/pos2
NATIVE_HG38_DATASETS = [
    'gtex_null_tier1_pairs',
    'gtex_null_tier2_pairs',
    'tcga_doubles_pairs',
    'correlated_eqtl_pairs',
    'rauscher_cftr_pairs',
]

# Special case: OKGP has AF1/AF2 columns already — copy to maf1/maf2
OKGP_DATASET = 'okgp_chr12_pairs'

# Synthetic datasets: no observed variants, maf1/maf2 = NaN
SYNTHETIC_DATASETS = [
    'fas_pairs',
    'kras_neighborhood_pairs',
    'trna41_pairs',
    'ke_sl6_pairs',
]


def collect_variants():
    """Collect unique (chrom, pos, ref, alt) tuples from all non-synthetic TSVs."""
    variants = set()

    for name in LIFTOVER_DATASETS:
        path = OUTPUT_DIR / f'{name}.tsv'
        if not path.exists():
            print(f"  Skipping {name} (file not found)")
            continue
        df = pd.read_csv(path, sep='\t', dtype=str)
        if len(df) == 0:
            continue
        chroms = df['chr'].values
        p1s = df['pos1_hg38'].values
        p2s = df['pos2_hg38'].values
        r1s = df['ref1'].values
        a1s = df['alt1'].values
        r2s = df['ref2'].values
        a2s = df['alt2'].values
        for c, p1, p2, r1, a1, r2, a2 in zip(chroms, p1s, p2s, r1s, a1s, r2s, a2s):
            if pd.notna(c) and pd.notna(p1) and pd.notna(r1) and pd.notna(a1):
                variants.add((str(c), str(p1), str(r1), str(a1)))
            if pd.notna(c) and pd.notna(p2) and pd.notna(r2) and pd.notna(a2):
                variants.add((str(c), str(p2), str(r2), str(a2)))
        print(f"  {name}: {len(df)} rows, running total {len(variants)} unique variants")

    for name in NATIVE_HG38_DATASETS:
        path = OUTPUT_DIR / f'{name}.tsv'
        if not path.exists():
            print(f"  Skipping {name} (file not found)")
            continue
        df = pd.read_csv(path, sep='\t', dtype=str)
        if len(df) == 0:
            continue
        chroms = df['chr'].values
        p1s = df['pos1'].values
        p2s = df['pos2'].values
        r1s = df['ref1'].values
        a1s = df['alt1'].values
        r2s = df['ref2'].values
        a2s = df['alt2'].values
        for c, p1, p2, r1, a1, r2, a2 in zip(chroms, p1s, p2s, r1s, a1s, r2s, a2s):
            if pd.notna(c) and pd.notna(p1) and pd.notna(r1) and pd.notna(a1):
                variants.add((str(c), str(p1), str(r1), str(a1)))
            if pd.notna(c) and pd.notna(p2) and pd.notna(r2) and pd.notna(a2):
                variants.add((str(c), str(p2), str(r2), str(a2)))
        print(f"  {name}: {len(df)} rows, running total {len(variants)} unique variants")

    return variants


def patch_tsv(name, maf_lookup, pos1_col, pos2_col):
    """Read a TSV, assign maf1/maf2 via lookup, overwrite the file."""
    path = OUTPUT_DIR / f'{name}.tsv'
    if not path.exists():
        print(f"  Skipping {name} (file not found)")
        return
    df = pd.read_csv(path, sep='\t')
    if len(df) == 0:
        print(f"  {name}: 0 rows, skipping")
        return

    # Build string-typed lookup keys to match how variants were collected
    keys1 = list(zip(
        df['chr'].astype(str),
        df[pos1_col].astype(str),
        df['ref1'].astype(str),
        df['alt1'].astype(str),
    ))
    keys2 = list(zip(
        df['chr'].astype(str),
        df[pos2_col].astype(str),
        df['ref2'].astype(str),
        df['alt2'].astype(str),
    ))
    df['maf1'] = [maf_lookup.get(k) for k in keys1]
    df['maf2'] = [maf_lookup.get(k) for k in keys2]

    n_found = df['maf1'].notna().sum() + df['maf2'].notna().sum()
    n_total = len(df) * 2
    print(f"  {name}: {n_found}/{n_total} MAF values populated")

    df.to_csv(path, sep='\t', index=False)


def patch_okgp(maf_lookup):
    """OKGP special case: copy AF1/AF2 to maf1/maf2."""
    path = OUTPUT_DIR / f'{OKGP_DATASET}.tsv'
    if not path.exists():
        print(f"  Skipping {OKGP_DATASET} (file not found)")
        return
    df = pd.read_csv(path, sep='\t')
    if 'AF1' in df.columns and 'AF2' in df.columns:
        df['maf1'] = df['AF1']
        df['maf2'] = df['AF2']
        n_found = df['maf1'].notna().sum() + df['maf2'].notna().sum()
        print(f"  {OKGP_DATASET}: copied AF1/AF2 -> maf1/maf2 ({n_found}/{len(df)*2} populated)")
    else:
        df['maf1'] = np.nan
        df['maf2'] = np.nan
        print(f"  {OKGP_DATASET}: AF1/AF2 not found, set maf1/maf2 = NaN")
    df.to_csv(path, sep='\t', index=False)


def patch_synthetic():
    """Set maf1/maf2 = NaN for synthetic datasets."""
    for name in SYNTHETIC_DATASETS:
        path = OUTPUT_DIR / f'{name}.tsv'
        if not path.exists():
            print(f"  Skipping {name} (file not found)")
            continue
        df = pd.read_csv(path, sep='\t')
        df['maf1'] = np.nan
        df['maf2'] = np.nan
        df.to_csv(path, sep='\t', index=False)
        print(f"  {name}: {len(df)} rows, maf1/maf2 = NaN (synthetic)")


def update_combined():
    """Rebuild all_pairs_combined.tsv and epistasis_ids_light.csv with maf1/maf2."""
    all_datasets = (
        LIFTOVER_DATASETS + NATIVE_HG38_DATASETS
        + [OKGP_DATASET] + SYNTHETIC_DATASETS
    )

    dfs = []
    for name in all_datasets:
        path = OUTPUT_DIR / f'{name}.tsv'
        if not path.exists():
            continue
        df = pd.read_csv(path, sep='\t')
        if len(df) > 0:
            dfs.append(df)

    if not dfs:
        print("  No data to combine")
        return

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined_path = OUTPUT_DIR / 'all_pairs_combined.tsv'
    combined.to_csv(combined_path, sep='\t', index=False)
    print(f"  Updated {combined_path}: {len(combined):,} rows")

    # Update epistasis_ids_light.csv
    light_cols = ['source', 'label', 'epistasis_id', 'maf1', 'maf2']
    present = [c for c in light_cols if c in combined.columns]
    light = combined[present].copy()
    light_path = OUTPUT_DIR / 'epistasis_ids_light.csv'
    light.to_csv(light_path, index=False)
    print(f"  Updated {light_path}: {len(light):,} rows")


def main():
    print("=" * 60)
    print("STEP 1: Collect unique variants from TSVs")
    print("=" * 60)
    variants = collect_variants()
    print(f"\nTotal unique variants to query: {len(variants):,}")

    print("\n" + "=" * 60)
    print("STEP 2: Look up MAFs via Ensembl VEP")
    print("=" * 60)
    maf_lookup = lookup_variant_mafs(variants)

    n_with_maf = sum(1 for v in maf_lookup.values() if v is not None)
    print(f"\nMAF lookup complete: {n_with_maf:,} / {len(maf_lookup):,} variants have gnomAD AF")

    print("\n" + "=" * 60)
    print("STEP 3: Patch TSVs with maf1/maf2")
    print("=" * 60)

    # Liftover datasets
    for name in LIFTOVER_DATASETS:
        patch_tsv(name, maf_lookup, pos1_col='pos1_hg38', pos2_col='pos2_hg38')

    # Native hg38 datasets
    for name in NATIVE_HG38_DATASETS:
        patch_tsv(name, maf_lookup, pos1_col='pos1', pos2_col='pos2')

    # OKGP special case
    patch_okgp(maf_lookup)

    # Synthetic datasets
    patch_synthetic()

    print("\n" + "=" * 60)
    print("STEP 4: Update combined files")
    print("=" * 60)
    update_combined()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Variants queried: {len(variants):,}")
    print(f"  Variants with gnomAD AF: {n_with_maf:,}")
    print(f"  Cache hit rate: {len(variants) - len([v for v in variants if v not in maf_lookup]):,} / {len(variants):,}")

    print("\n  MAF coverage per dataset:")
    all_datasets = (
        LIFTOVER_DATASETS + NATIVE_HG38_DATASETS
        + [OKGP_DATASET] + SYNTHETIC_DATASETS
    )
    for name in all_datasets:
        path = OUTPUT_DIR / f'{name}.tsv'
        if not path.exists():
            continue
        df = pd.read_csv(path, sep='\t')
        if len(df) == 0:
            print(f"    {name}: 0 rows")
            continue
        m1 = df['maf1'].notna().sum() if 'maf1' in df.columns else 0
        m2 = df['maf2'].notna().sum() if 'maf2' in df.columns else 0
        print(f"    {name}: maf1 {m1}/{len(df)}, maf2 {m2}/{len(df)}")

    print("\nDone.")


if __name__ == '__main__':
    main()
