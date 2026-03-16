"""Parse FAS exon 3 combinatorial mutagenesis doubles (Baeza-Centurion et al. 2016)."""

import pandas as pd

from .common import FAS_EXCEL, OUTPUT_DIR

# FAS exon 3 genomic bounds (hg38, chr10)
FAS_EXON_START = 89_010_753
FAS_EXON_END = 89_010_815
FAS_CHROM = '10'


def _get_fas_pre_mrna():
    """Load FAS pre-mRNA sequence via seqmat Gene."""
    from seqmat.gene import Gene
    return Gene.from_file('FAS').transcript().generate_pre_mrna()


def _convert_local_exon_to_mut(local_id, pre_mrna, strand):
    """Convert local exon ID like '1-A' to 'FAS:10:pos:ref:alt:strand'."""
    pos_str, new_allele = local_id.split('-')
    pos = FAS_EXON_START + int(pos_str) - 1
    old_allele = pre_mrna.pre_mrna[pos][0].decode('utf-8')
    return f'FAS:{FAS_CHROM}:{pos}:{old_allele}:{new_allele}:{strand}'


def _parse_mut_id(mut_id):
    """Parse 'FAS:10:pos:ref:alt:strand' -> (pos, ref, alt)."""
    parts = mut_id.split(':')
    return int(parts[2]), parts[3], parts[4]


def parse():
    """Return DataFrame of FAS double-mutant pairs.

    Positions are hg38 native. No liftover needed.
    epistasis_id is built directly (not deferred to aggregation).
    """
    fas_pre_mrna = _get_fas_pre_mrna()
    # FAS is on positive strand
    strand = 'P'

    df = pd.read_excel(FAS_EXCEL).rename(columns={
        'IDA': 'mut1', 'IDB': 'mut2',
        'Ascore': 'score1', 'Bscore': 'score2',
        'A+B': 'expected_experimental_score',
        'ABscore': 'experimental_score',
        'EmpiricalEpistasis': 'empirical_epistasis',
        'EmpiricalEpistasisRawPvalue': 'empirical_pval',
    })
    keep_cols = ['mut1', 'mut2', 'score1', 'score2', 'experimental_score',
                 'expected_experimental_score', 'empirical_epistasis',
                 'empirical_pval', 'CategoryA', 'CategoryB']
    df = df[[c for c in keep_cols if c in df.columns]].sort_values(['mut1', 'mut2']).copy()
    print(f"Loaded {len(df)} FAS double-mutant pairs from Excel")

    # Convert local IDs to genomic mut IDs
    df['mut1'] = df['mut1'].apply(lambda x: _convert_local_exon_to_mut(x, fas_pre_mrna, strand))
    df['mut2'] = df['mut2'].apply(lambda x: _convert_local_exon_to_mut(x, fas_pre_mrna, strand))

    # Build epistasis_id (ensure pos1 < pos2)
    def _make_epistasis_id(row):
        p1 = _parse_mut_id(row['mut1'])[0]
        p2 = _parse_mut_id(row['mut2'])[0]
        if p1 <= p2:
            return f"{row['mut1']}|{row['mut2']}"
        return f"{row['mut2']}|{row['mut1']}"

    df['epistasis_id'] = df.apply(_make_epistasis_id, axis=1)

    # Extract pos/ref/alt for standard columns
    df[['pos1', 'ref1', 'alt1']] = df['mut1'].apply(
        lambda x: pd.Series(_parse_mut_id(x)))
    df[['pos2', 'ref2', 'alt2']] = df['mut2'].apply(
        lambda x: pd.Series(_parse_mut_id(x)))

    df['distance_bp'] = (df['pos1'] - df['pos2']).abs()
    df['source'] = 'fas_exon'
    df['pair_id'] = [f'fas_{i+1:05d}' for i in range(len(df))]
    df['gene'] = 'FAS'
    df['chr'] = FAS_CHROM
    df['label'] = 'positive'
    df['genome_build'] = 'hg38'

    print(f"FAS pairs: {len(df)}, distance range: {df['distance_bp'].min()}-{df['distance_bp'].max()} bp")
    print(f"  Epistasis IDs: {df['epistasis_id'].notna().sum()} / {len(df)}")

    return df


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    result = parse()
    result.to_csv(OUTPUT_DIR / 'fas_pairs.tsv', sep='\t', index=False)
    print(f"\nSaved {len(result)} FAS pairs")
