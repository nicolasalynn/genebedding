"""Parse correlated (high-LD) eQTL variant pairs — LD sanity check / null control."""

import pandas as pd

from .common import DATA_DIR, OUTPUT_DIR

CORR_CSV = DATA_DIR / 'correlated_variants_27K.csv'


def parse():
    """Return DataFrame of correlated eQTL pairs.

    Positions are hg38 native. epistasis_id deferred to aggregation (source
    file has 5-field IDs without strand).
    """
    df = pd.read_csv(CORR_CSV)
    print(f"Loaded {len(df)} correlated eQTL pairs")

    # Parse ref/alt from Uniq_ID columns (format: pos:ref:alt)
    id1_parts = df['Uniq_ID_1'].str.split(':', expand=True)
    id2_parts = df['Uniq_ID_2'].str.split(':', expand=True)

    df['pos1'] = id1_parts[0].astype(int)
    df['ref1'] = id1_parts[1]
    df['alt1'] = id1_parts[2]
    df['pos2'] = id2_parts[0].astype(int)
    df['ref2'] = id2_parts[1]
    df['alt2'] = id2_parts[2]

    df['chr'] = df['chrom'].astype(str)
    df['distance_bp'] = df['distance']
    df['source'] = 'correlated_eqtl'
    df['pair_id'] = [f'correqtl_{i+1:05d}' for i in range(len(df))]
    df['gene'] = 'GENE'
    df['label'] = 'null_ld'
    df['genome_build'] = 'hg38'

    print(f"Correlated eQTL pairs: {len(df)}, distance range: "
          f"{df['distance_bp'].min()}-{df['distance_bp'].max()} bp")

    return df


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    result = parse()
    result.to_csv(OUTPUT_DIR / 'correlated_eqtl_pairs.tsv', sep='\t', index=False)
    print(f"\nSaved {len(result)} correlated eQTL pairs")
