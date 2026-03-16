"""Parse OKGP chr12 population double-variant candidates."""

import pandas as pd

from .common import DATA_DIR, OUTPUT_DIR

OKGP_CSV = DATA_DIR / 'okgp_epistasis.csv'


def parse():
    """Return DataFrame of OKGP chr12 double-variant pairs.

    Positions are hg38 native. epistasis_id already present in source data.
    """
    df = pd.read_csv(OKGP_CSV)
    print(f"Loaded {len(df)} OKGP chr12 pairs")

    df['source'] = 'okgp_chr12'
    df['pair_id'] = [f'okgp_{i+1:06d}' for i in range(len(df))]
    df['chr'] = df['chrom'].astype(str)
    df['distance_bp'] = df['distance']
    df['label'] = 'positive'
    df['genome_build'] = 'hg38'

    print(f"OKGP chr12 pairs: {len(df)}, distance range: "
          f"{df['distance_bp'].min()}-{df['distance_bp'].max()} bp")

    return df


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    result = parse()
    result.to_csv(OUTPUT_DIR / 'okgp_chr12_pairs.tsv', sep='\t', index=False)
    print(f"\nSaved {len(result)} OKGP pairs")
