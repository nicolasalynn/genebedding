"""Parse GTEx matched null pairs (tier 2) — distance-matched controls for positive pairs."""

import pandas as pd

from .common import (
    DISTANCE_THRESHOLD, OUTPUT_DIR,
    LCL_TISSUE, GTEX_EQTL_DIR,
    ensure_gtex_extracted, gtex_tissue_paths,
    parse_gtex_variant_id, load_gene_symbol_mapping,
)


def parse(positive_pairs_df=None):
    """Return gtex_null_tier2 DataFrame (native hg38).

    Args:
        positive_pairs_df: DataFrame with columns [source, pair_id, gene, chr,
            pos1, pos2, pos1_hg38, pos2_hg38, snp_id1, snp_id2, ref1, alt1,
            ref2, alt2, distance_bp]. If None, reads from existing TSVs.
    """
    ensure_gtex_extracted()

    # Load significant eQTLs (LCL)
    _, lcl_signif_path = gtex_tissue_paths(LCL_TISSUE)
    print("Loading GTEx significant variant-gene pairs (LCL)...")
    gtex_sig_lcl = pd.read_csv(lcl_signif_path, sep='\t')
    print(f"Loaded {len(gtex_sig_lcl):,} significant pairs")

    # Parse variant IDs
    parsed_sig = gtex_sig_lcl['variant_id'].apply(lambda x: pd.Series(parse_gtex_variant_id(x)))
    parsed_sig.columns = ['var_chr', 'var_pos', 'var_ref', 'var_alt']
    gtex_sig_lcl = pd.concat([gtex_sig_lcl, parsed_sig], axis=1)
    gtex_sig_lcl['var_pos'] = pd.to_numeric(gtex_sig_lcl['var_pos'], errors='coerce').astype('Int64')

    # Build gene-level lookup
    gtex_sig_by_gene = {}
    for gene_id, group in gtex_sig_lcl.groupby('gene_id'):
        gtex_sig_by_gene[gene_id] = group[
            ['variant_id', 'var_chr', 'var_pos', 'var_ref', 'var_alt', 'slope', 'pval_nominal']
        ].to_dict('records')
    print(f"Genes with significant eQTLs: {len(gtex_sig_by_gene)}")

    # Gene symbol -> Ensembl ID mapping
    symbol_to_ensembl, _ = load_gene_symbol_mapping()

    # Load positive pairs if not provided
    if positive_pairs_df is None:
        yang_path = OUTPUT_DIR / 'yang_positive_pairs.tsv'
        brown_path = OUTPUT_DIR / 'brown_positive_pairs.tsv'
        dfs = []
        shared_cols = ['source', 'pair_id', 'gene', 'chr', 'pos1', 'pos2',
                       'pos1_hg38', 'pos2_hg38', 'snp_id1', 'snp_id2',
                       'ref1', 'alt1', 'ref2', 'alt2', 'distance_bp']
        if yang_path.exists():
            y = pd.read_csv(yang_path, sep='\t')
            dfs.append(y[[c for c in shared_cols if c in y.columns]])
        if brown_path.exists():
            b = pd.read_csv(brown_path, sep='\t')
            dfs.append(b[[c for c in shared_cols if c in b.columns]])
        all_positive = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        all_positive = positive_pairs_df

    print(f"Total positive pairs to match: {len(all_positive)}")

    # Match positive pairs to GTEx eQTLs
    null_tier2_pairs = []
    pair_id_counter = 0
    matched_positive_count = 0

    for _, pos_row in all_positive.iterrows():
        gene = pos_row['gene']
        gene_ensembl = symbol_to_ensembl.get(gene)
        if not gene_ensembl or gene_ensembl not in gtex_sig_by_gene:
            continue

        anchor_hg38 = pos_row['pos1_hg38']
        if pd.isna(anchor_hg38):
            continue

        anchor_hg38 = int(anchor_hg38)
        anchor_ref = pos_row['ref1']
        anchor_alt = pos_row['alt1']
        gene_eqtls = gtex_sig_by_gene[gene_ensembl]

        has_match = False
        for eqtl in gene_eqtls:
            eqtl_pos = eqtl['var_pos']
            if pd.isna(eqtl_pos):
                continue
            eqtl_pos = int(eqtl_pos)

            distance = abs(anchor_hg38 - eqtl_pos)
            if distance > DISTANCE_THRESHOLD or distance == 0:
                continue

            pair_id_counter += 1
            has_match = True

            null_tier2_pairs.append({
                'source': 'gtex_matched_null',
                'pair_id': f'gtex_null2_{pair_id_counter:06d}',
                'gene': gene,
                'positive_pair_id': pos_row['pair_id'],
                'chr': eqtl['var_chr'],
                'pos1': anchor_hg38,
                'pos2': eqtl_pos,
                'ref1': anchor_ref,
                'alt1': anchor_alt,
                'ref2': eqtl['var_ref'],
                'alt2': eqtl['var_alt'],
                'variant_id1': pd.NA,
                'variant_id2': eqtl['variant_id'],
                'distance_bp': distance,
                'label': 'null_tier2',
                'genome_build': 'hg38',
            })

        if has_match:
            matched_positive_count += 1

    gtex_null_tier2_cols = [
        'source', 'pair_id', 'gene', 'positive_pair_id', 'chr', 'pos1', 'pos2',
        'ref1', 'alt1', 'ref2', 'alt2', 'variant_id1', 'variant_id2',
        'distance_bp', 'label', 'genome_build',
    ]
    gtex_null_tier2 = (pd.DataFrame(null_tier2_pairs, columns=gtex_null_tier2_cols)
                       if null_tier2_pairs
                       else pd.DataFrame(columns=gtex_null_tier2_cols))

    print(f"\nPositive pairs with ≥1 matched null: {matched_positive_count} / {len(all_positive)}")
    print(f"Total null tier 2 pairs (≤{DISTANCE_THRESHOLD:,} bp): {len(gtex_null_tier2)}")

    return gtex_null_tier2


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = parse()
    df.to_csv(OUTPUT_DIR / 'gtex_null_tier2_pairs.tsv', sep='\t', index=False)
    print(f"\nSaved {len(df)} pairs")
