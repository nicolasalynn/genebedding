"""Parse Brown 2014 v-eQTL variant pairs (positive)."""

import pandas as pd

from .common import (
    DISTANCE_THRESHOLD, BROWN_FILE, OUTPUT_DIR,
    liftover_dataframe, lookup_rsid_alleles,
)


def parse():
    """Return brown_pairs DataFrame.

    Alleles looked up via Ensembl REST API.
    Positions are liftover'd to hg38 and distance-filtered.
    No epistasis_id or ref validation (deferred to aggregation).
    """
    # Load Sheet 1B with multi-row header
    brown_b_multi = pd.read_excel(BROWN_FILE, sheet_name='Supplementary File 1B', header=[2, 3])
    brown_b = brown_b_multi.copy()
    brown_b.columns = [
        'chromosome', 'gene_name', 'ensembl_id', 'epistasis_snp', 'veqtl_snp',
        'pos_epistasis', 'pos_veqtl',
        'pval_twinsuk', 'pval_dominance_corrected',
        'additive_var_twinsuk', 'interaction_var_twinsuk',
        'additive_var_geuvadis', 'interaction_var_geuvadis',
        'pval_geuvadis', 'pval_haplotype_controlled', 'replicated_geuvadis',
    ]
    print(f"Loaded Sheet 1B: {len(brown_b)} pairs")

    # Clean types
    brown_b['chromosome'] = brown_b['chromosome'].astype(str)
    brown_b['pos_epistasis'] = pd.to_numeric(brown_b['pos_epistasis'], errors='coerce').astype('Int64')
    brown_b['pos_veqtl'] = pd.to_numeric(brown_b['pos_veqtl'], errors='coerce').astype('Int64')

    # Replication
    brown_b['pval_geuvadis_numeric'] = pd.to_numeric(brown_b['pval_geuvadis'], errors='coerce')
    brown_b['replicated'] = brown_b['pval_geuvadis_numeric'] < 0.05

    # Distance
    brown_b['distance_bp'] = (brown_b['pos_epistasis'] - brown_b['pos_veqtl']).abs()

    # Build initial output
    brown_pairs = pd.DataFrame({
        'source': 'brown_veqtl',
        'pair_id': [f'brown_pos_{i+1:04d}' for i in range(len(brown_b))],
        'gene': brown_b['gene_name'].values,
        'chr': brown_b['chromosome'].values,
        'pos1': brown_b['pos_veqtl'].values,
        'pos2': brown_b['pos_epistasis'].values,
        'snp_id1': brown_b['veqtl_snp'].values,
        'snp_id2': brown_b['epistasis_snp'].values,
        'distance_bp': brown_b['distance_bp'].values,
        'label': 'positive',
        'p_value': brown_b['pval_twinsuk'].values,
        'effect_size': brown_b['interaction_var_twinsuk'].values,
        'replicated': brown_b['replicated'].values,
        'genome_build': 'hg19',
    })
    print(f"Brown pairs (all, pre-liftover): {len(brown_pairs)}")

    # Look up ref/alt alleles via Ensembl REST
    brown_rsids = (set(brown_pairs['snp_id1'].dropna().unique()) |
                   set(brown_pairs['snp_id2'].dropna().unique()))
    print(f"\nLooking up {len(brown_rsids)} rsIDs via Ensembl...")
    rsid_alleles = lookup_rsid_alleles(brown_rsids)
    print(f"Successfully looked up: {len(rsid_alleles)} / {len(brown_rsids)}")

    rsid_ref = {rsid: alleles[0] for rsid, alleles in rsid_alleles.items()}
    rsid_alt = {rsid: alleles[1][0] for rsid, alleles in rsid_alleles.items()}

    brown_pairs['ref1'] = brown_pairs['snp_id1'].map(rsid_ref)
    brown_pairs['alt1'] = brown_pairs['snp_id1'].map(rsid_alt)
    brown_pairs['ref2'] = brown_pairs['snp_id2'].map(rsid_ref)
    brown_pairs['alt2'] = brown_pairs['snp_id2'].map(rsid_alt)

    n_complete = (brown_pairs['ref1'].notna() & brown_pairs['alt1'].notna() &
                  brown_pairs['ref2'].notna() & brown_pairs['alt2'].notna()).sum()
    print(f"Pairs with complete alleles: {n_complete} / {len(brown_pairs)}")

    # Liftover + filter
    print("\nLifting Brown pairs...")
    brown_pairs = liftover_dataframe(brown_pairs)

    return brown_pairs


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = parse()
    df.to_csv(OUTPUT_DIR / 'brown_positive_pairs.tsv', sep='\t', index=False)
    print(f"\nSaved {len(df)} pairs")
