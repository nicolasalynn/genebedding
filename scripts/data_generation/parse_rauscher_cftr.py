"""Parse Rauscher et al. CFTR missense × sSNP pairs (negative control).

Six missense mutations paired with synonymous c.2562T>G (chr7:117595001),
testing translational epistasis invisible to DNA sequence models.
Coordinates validated against hg38 FASTA via Ensembl map/cds endpoint.
"""

import pandas as pd

from .common import OUTPUT_DIR

# CFTR is on positive strand (chr7)
CFTR_CHROM = '7'
CFTR_STRAND = 'P'

# Synonymous SNP present in all pairs
SSNP_POS = 117595001  # c.2562T>G, hg38
SSNP_REF = 'T'
SSNP_ALT = 'G'

# Validated missense mutations (hg38 coordinates from Ensembl map/cds)
MISSENSE_VARIANTS = [
    # (name, cDNA_change, genomic_pos_hg38, ref, alt)
    ('G85E',   'c.254G>A',  117509123, 'G', 'A'),
    ('G551D',  'c.1652G>A', 117587806, 'G', 'A'),
    ('D579G',  'c.1736A>G', 117590409, 'A', 'G'),
    ('D614G',  'c.1841A>G', 117592008, 'A', 'G'),
    ('F1074L', 'c.3222T>A', 117611663, 'T', 'A'),
    ('N1303K', 'c.3909C>G', 117652877, 'C', 'G'),
]


def parse():
    """Return DataFrame of 6 CFTR missense × sSNP pairs.

    All positions are hg38. No distance filter applied (these are negative
    controls, most pairs exceed 6kb). epistasis_id deferred to aggregation.
    """
    rows = []
    for name, cdna, mis_pos, mis_ref, mis_alt in MISSENSE_VARIANTS:
        # Ensure pos1 < pos2
        if mis_pos < SSNP_POS:
            p1, r1, a1 = mis_pos, mis_ref, mis_alt
            p2, r2, a2 = SSNP_POS, SSNP_REF, SSNP_ALT
        else:
            p1, r1, a1 = SSNP_POS, SSNP_REF, SSNP_ALT
            p2, r2, a2 = mis_pos, mis_ref, mis_alt

        rows.append({
            'source': 'rauscher_cftr',
            'gene': 'CFTR',
            'chr': CFTR_CHROM,
            'pos1': p1,
            'pos2': p2,
            'ref1': r1,
            'alt1': a1,
            'ref2': r2,
            'alt2': a2,
            'distance_bp': abs(mis_pos - SSNP_POS),
            'label': 'negative_control',
            'missense_name': name,
            'cdna_change': cdna,
            'genome_build': 'hg38',
        })

    df = pd.DataFrame(rows)
    df['pair_id'] = [f'rauscher_{i+1:04d}' for i in range(len(df))]

    print(f"Rauscher CFTR pairs: {len(df)}, distance range: "
          f"{df['distance_bp'].min():,}-{df['distance_bp'].max():,} bp")

    return df


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    result = parse()
    result.to_csv(OUTPUT_DIR / 'rauscher_cftr_pairs.tsv', sep='\t', index=False)
    print(f"\nSaved {len(result)} Rauscher CFTR pairs")
