"""Parse GTEx independent eQTL null pairs (tier 1)."""

import pandas as pd

from .common import (
    DISTANCE_THRESHOLD, OUTPUT_DIR,
    LCL_TISSUE, WB_TISSUE,
    ensure_gtex_extracted, gtex_tissue_paths, parse_gtex_variant_id,
)


def _make_null_tier1(tissue_df, tissue_name, prefix):
    """Generate null pairs from genes with >=2 independent eQTL signals."""
    null_pairs = []
    pair_id_counter = 0
    genes_multi = tissue_df.groupby('gene_id').filter(lambda x: len(x) >= 2)

    for gene_id, group in genes_multi.groupby('gene_id'):
        rows = list(group.iterrows())
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                _, row1 = rows[i]
                _, row2 = rows[j]
                if row1['var_chr'] != row2['var_chr']:
                    continue
                distance = abs(int(row1['var_pos']) - int(row2['var_pos']))
                if distance > DISTANCE_THRESHOLD:
                    continue
                pair_id_counter += 1
                null_pairs.append({
                    'source': 'gtex_independent',
                    'pair_id': f'{prefix}_{pair_id_counter:05d}',
                    'gene_id': gene_id,
                    'gene_name': pd.NA,
                    'tissue': tissue_name,
                    'chr': row1['var_chr'],
                    'pos1': row1['var_pos'],
                    'pos2': row2['var_pos'],
                    'variant_id1': row1['variant_id'],
                    'variant_id2': row2['variant_id'],
                    'ref1': row1['var_ref'],
                    'alt1': row1['var_alt'],
                    'ref2': row2['var_ref'],
                    'alt2': row2['var_alt'],
                    'distance_bp': distance,
                    'rank1': row1['rank'],
                    'rank2': row2['rank'],
                    'beta1': row1['slope'],
                    'beta2': row2['slope'],
                    'pval1': row1['pval_nominal'],
                    'pval2': row2['pval_nominal'],
                    'label': 'null_tier1',
                    'genome_build': 'hg38',
                })
    return pd.DataFrame(null_pairs) if null_pairs else pd.DataFrame()


def _parse_variant_cols(df):
    """Add var_chr, var_pos, var_ref, var_alt columns from variant_id."""
    parsed = df['variant_id'].apply(lambda x: pd.Series(parse_gtex_variant_id(x)))
    parsed.columns = ['var_chr', 'var_pos', 'var_ref', 'var_alt']
    for col in parsed.columns:
        df[col] = parsed[col]
    df['var_pos'] = pd.to_numeric(df['var_pos'], errors='coerce').astype('Int64')


def parse():
    """Return gtex_null_tier1 DataFrame (native hg38, no liftover needed)."""
    ensure_gtex_extracted()

    lcl_indep_path, _ = gtex_tissue_paths(LCL_TISSUE)
    wb_indep_path, _ = gtex_tissue_paths(WB_TISSUE)

    gtex_lcl = pd.read_csv(lcl_indep_path, sep='\t')
    gtex_wb = pd.read_csv(wb_indep_path, sep='\t')
    print(f"GTEx LCL independent: {len(gtex_lcl)} rows")
    print(f"GTEx WB independent:  {len(gtex_wb)} rows")

    _parse_variant_cols(gtex_lcl)
    _parse_variant_cols(gtex_wb)

    lcl_pairs = _make_null_tier1(gtex_lcl, LCL_TISSUE, 'gtex_null1_lcl')
    wb_pairs = _make_null_tier1(gtex_wb, WB_TISSUE, 'gtex_null1_wb')

    gtex_null_tier1 = pd.concat([lcl_pairs, wb_pairs], ignore_index=True)
    print(f"\nLCL null tier 1 (≤{DISTANCE_THRESHOLD:,} bp): {len(lcl_pairs)}")
    print(f"WB null tier 1 (≤{DISTANCE_THRESHOLD:,} bp):  {len(wb_pairs)}")
    print(f"Total: {len(gtex_null_tier1)}")

    return gtex_null_tier1


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = parse()
    df.to_csv(OUTPUT_DIR / 'gtex_null_tier1_pairs.tsv', sep='\t', index=False)
    print(f"\nSaved {len(df)} pairs")
