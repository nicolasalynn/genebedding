"""Parse Yang 2016 evQTL variant pairs (positive)."""

import pandas as pd

from .common import (
    DISTANCE_THRESHOLD, YANG_S3_FILE, OUTPUT_DIR, liftover_dataframe,
)


def _parse_alleles(allele_str):
    if pd.isna(allele_str):
        return pd.NA, pd.NA
    parts = str(allele_str).split('/')
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return pd.NA, pd.NA


def parse():
    """Return yang_pairs DataFrame.

    Positions are liftover'd to hg38 and distance-filtered.
    No epistasis_id or ref validation (deferred to aggregation).
    """
    yang = pd.read_excel(YANG_S3_FILE, sheet_name='S3_partial_eQTL')
    print(f"Loaded {len(yang)} evQTL-peSNP pairs from Yang 2016")

    # Clean types
    yang['evQTL_pos'] = pd.to_numeric(yang['evQTL_pos'], errors='coerce').astype('Int64')
    yang['partial_eQTL_pos'] = pd.to_numeric(yang['partial_eQTL_pos'], errors='coerce').astype('Int64')
    yang['evQTL_chrid'] = yang['evQTL_chrid'].astype(str)
    yang['partial_eQTL_chrid'] = yang['partial_eQTL_chrid'].astype(str)

    # Remove self-pairs
    same_snp = yang['evQTL_rsid'] == yang['partial_eQTL_rsid']
    yang = yang[~same_snp].copy()
    print(f"After removing self-pairs: {len(yang)}")

    # Cis-cis filter
    cis_mask = yang['evQTL_chrid'] == yang['partial_eQTL_chrid']
    yang_cis = yang[cis_mask].copy()
    print(f"Cis-cis pairs (same chromosome): {len(yang_cis)}")

    # Distance
    yang_cis['distance_bp'] = (yang_cis['evQTL_pos'] - yang_cis['partial_eQTL_pos']).abs()

    # Parse alleles
    yang_cis[['evQTL_ref', 'evQTL_alt_parsed']] = yang_cis['evQTL_alleles'].apply(
        lambda x: pd.Series(_parse_alleles(x))
    )
    yang_cis[['peSNP_ref', 'peSNP_alt_parsed']] = yang_cis['partial_eQTL_alleles'].apply(
        lambda x: pd.Series(_parse_alleles(x))
    )

    # Build output
    yang_pairs = pd.DataFrame({
        'source': 'yang_evqtl',
        'pair_id': [f'yang_pos_{i+1:05d}' for i in range(len(yang_cis))],
        'gene': yang_cis['evQTL_Gene'].values,
        'chr': yang_cis['evQTL_chrid'].values,
        'pos1': yang_cis['evQTL_pos'].values,
        'pos2': yang_cis['partial_eQTL_pos'].values,
        'snp_id1': yang_cis['evQTL_rsid'].values,
        'snp_id2': yang_cis['partial_eQTL_rsid'].values,
        'ref1': yang_cis['evQTL_ref'].values,
        'alt1': yang_cis['evQTL_alt_parsed'].values,
        'ref2': yang_cis['peSNP_ref'].values,
        'alt2': yang_cis['peSNP_alt_parsed'].values,
        'distance_bp': yang_cis['distance_bp'].values,
        'label': 'positive',
        'p_value': yang_cis['partial_eQTL_P_value'].values,
        'effect_size': pd.NA,
        'genome_build': 'hg19',
    })
    print(f"Yang pairs (all cis-cis, pre-liftover): {len(yang_pairs)}")

    # Liftover + filter
    print("\nLifting Yang pairs...")
    yang_pairs = liftover_dataframe(yang_pairs)
    print(f"Unique genes remaining: {yang_pairs['gene'].nunique()}")

    return yang_pairs


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = parse()
    df.to_csv(OUTPUT_DIR / 'yang_positive_pairs.tsv', sep='\t', index=False)
    print(f"\nSaved {len(df)} pairs")
