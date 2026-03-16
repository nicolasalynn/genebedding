"""Parse MST1R/RON exon 11 splicing double-mutants (Braun et al. 2018)."""

import pandas as pd

from .common import (
    MST1R_LIBRARY_EXCEL, OUTPUT_DIR, liftover_pos,
)

# MST1R minigene → hg19 genomic coordinate mapping (chr3, minus strand)
# Minigene pos 1 corresponds to hg19 chr3:49933838
MST1R_MINIGENE_HG19_BASE = 49933838
MST1R_CHROM = '3'
MST1R_STRAND = 'N'  # MST1R is on minus strand

# HEK293T replicate sheet names
_HEK_SHEETS = ['HEK293T rep1', 'HEK293T rep2', 'HEK293T rep3']


def _parse_snv_mutation(mut_str):
    """Parse mutation like 'T216A' -> (ref, minigene_pos, alt)."""
    ref = mut_str[0]
    alt = mut_str[-1]
    pos = int(mut_str[1:-1])
    return ref, pos, alt


def _minigene_to_hg19(minigene_pos):
    """Convert minigene position to hg19 genomic coordinate (minus strand)."""
    return MST1R_MINIGENE_HG19_BASE - minigene_pos


def parse():
    """Return DataFrame of MST1R double-mutant pairs.

    Positions are hg19 with hg38 liftover. epistasis_id deferred to aggregation.
    """
    # Read HEK293T replicates
    sheets = pd.read_excel(MST1R_LIBRARY_EXCEL, sheet_name=_HEK_SHEETS)
    rep1 = sheets[_HEK_SHEETS[0]]
    rep2 = sheets[_HEK_SHEETS[1]]
    rep3 = sheets[_HEK_SHEETS[2]]
    print(f"Loaded MST1R library: {len(rep1)} rows in rep1")

    # Mutation column is 'mutations'
    mut_col = 'mutations'

    # Filter to rows with exactly 2 comma-separated SNV mutations
    rep1_valid = rep1[rep1[mut_col].notna()].copy()
    rep1_valid['n_muts'] = rep1_valid[mut_col].apply(lambda x: len(str(x).split(',')))
    doubles = rep1_valid[rep1_valid['n_muts'] == 2].copy()
    print(f"Rows with exactly 2 SNV mutations: {len(doubles)}")

    # Splicing readout columns
    ae_inc_col = 'AE inclusion (%)'
    ae_skip_col = 'AE skipping (%)'

    # Index rep2/rep3 by barcode for averaging
    rep2_idx = rep2.set_index('barcode')
    rep3_idx = rep3.set_index('barcode')

    # Build pairs
    rows = []
    for idx, row in doubles.iterrows():
        muts = [m.strip() for m in str(row[mut_col]).split(',')]
        if len(muts) != 2:
            continue

        try:
            ref1, mg_pos1, alt1 = _parse_snv_mutation(muts[0])
            ref2, mg_pos2, alt2 = _parse_snv_mutation(muts[1])
        except (ValueError, IndexError):
            continue

        # Convert to hg19 genomic positions
        hg19_pos1 = _minigene_to_hg19(mg_pos1)
        hg19_pos2 = _minigene_to_hg19(mg_pos2)

        # Ensure pos1 < pos2
        if hg19_pos1 > hg19_pos2:
            hg19_pos1, hg19_pos2 = hg19_pos2, hg19_pos1
            ref1, ref2 = ref2, ref1
            alt1, alt2 = alt2, alt1

        # Liftover to hg38
        hg38_pos1 = liftover_pos(MST1R_CHROM, hg19_pos1)
        hg38_pos2 = liftover_pos(MST1R_CHROM, hg19_pos2)

        # Average splicing readouts across 3 HEK293T replicates
        bc = row['barcode']
        inc_vals = [row[ae_inc_col]]
        skip_vals = [row[ae_skip_col]]
        for rep_idx in [rep2_idx, rep3_idx]:
            if bc in rep_idx.index:
                inc_vals.append(rep_idx.loc[bc, ae_inc_col])
                skip_vals.append(rep_idx.loc[bc, ae_skip_col])

        ae_inc = pd.to_numeric(pd.Series(inc_vals), errors='coerce').mean()
        ae_skip = pd.to_numeric(pd.Series(skip_vals), errors='coerce').mean()

        rows.append({
            'pos1': hg19_pos1,
            'pos2': hg19_pos2,
            'pos1_hg38': hg38_pos1,
            'pos2_hg38': hg38_pos2,
            'ref1': ref1,
            'alt1': alt1,
            'ref2': ref2,
            'alt2': alt2,
            'distance_bp': abs(hg19_pos2 - hg19_pos1),
            'ae_inclusion_pct': ae_inc,
            'ae_skipping_pct': ae_skip,
        })

    df = pd.DataFrame(rows)

    # Filter to rows where liftover succeeded
    df['pos1_hg38'] = pd.to_numeric(df['pos1_hg38'], errors='coerce').astype('Int64')
    df['pos2_hg38'] = pd.to_numeric(df['pos2_hg38'], errors='coerce').astype('Int64')
    n_before = len(df)
    df = df[df['pos1_hg38'].notna() & df['pos2_hg38'].notna()].copy()
    print(f"  After liftover: {len(df)} / {n_before}")

    df['source'] = 'mst1r_splicing'
    df['pair_id'] = [f'mst1r_{i+1:04d}' for i in range(len(df))]
    df['gene'] = 'MST1R'
    df['chr'] = MST1R_CHROM
    df['label'] = 'positive'
    df['genome_build'] = 'hg19'

    print(f"MST1R pairs: {len(df)}, distance range: "
          f"{df['distance_bp'].min()}-{df['distance_bp'].max()} bp")

    return df


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    result = parse()
    result.to_csv(OUTPUT_DIR / 'mst1r_splicing_pairs.tsv', sep='\t', index=False)
    print(f"\nSaved {len(result)} MST1R pairs")
