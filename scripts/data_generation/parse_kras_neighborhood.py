"""Generate KRAS synthetic neighborhood double-SNV pairs from hg38 FASTA."""

from itertools import combinations, product

import pandas as pd

from .common import OUTPUT_DIR

BASES = 'ACGT'
KRAS_CENTER_LO = 25227343
KRAS_CENTER_HI = 25227344
KRAS_WINDOW = 150
KRAS_CHROM = '12'
KRAS_STRAND = 'N'  # KRAS is on negative strand


def _generate_pairs(seq, indices, gene, chrom, strand, delta_dist):
    """Generate all double-SNV epistasis IDs within delta_dist of each other."""
    L = len(seq)
    pairs = []
    for i, j in combinations(range(L), 2):
        pos1, pos2 = int(indices[i]), int(indices[j])
        ref1, ref2 = seq[i], seq[j]
        if ref1 not in BASES or ref2 not in BASES:
            continue
        if pos2 - pos1 > delta_dist:
            continue
        alts1 = [b for b in BASES if b != ref1]
        alts2 = [b for b in BASES if b != ref2]
        for alt1, alt2 in product(alts1, alts2):
            eid = (f'{gene}:{chrom}:{pos1}:{ref1}:{alt1}:{strand}|'
                   f'{gene}:{chrom}:{pos2}:{ref2}:{alt2}:{strand}')
            pairs.append({
                'pos1': pos1, 'pos2': pos2,
                'ref1': ref1, 'alt1': alt1,
                'ref2': ref2, 'alt2': alt2,
                'distance_bp': pos2 - pos1,
                'epistasis_id': eid,
            })
    return pairs


def parse():
    """Return DataFrame of KRAS synthetic neighborhood pairs.

    Positions are hg38 native. epistasis_id is built directly.
    """
    from seqmat import SeqMat

    lo = KRAS_CENTER_LO - KRAS_WINDOW
    hi = KRAS_CENTER_HI + KRAS_WINDOW
    sm = SeqMat.from_fasta('hg38', 'chr12', lo, hi)
    seq, indices = sm.seq, sm.index

    # Adjacent pairs (delta_dist=1) + nearby pairs (delta_dist=5)
    pairs_d1 = _generate_pairs(seq, indices, 'KRAS', KRAS_CHROM, KRAS_STRAND, delta_dist=1)
    pairs_d5 = _generate_pairs(seq, indices, 'KRAS', KRAS_CHROM, KRAS_STRAND, delta_dist=5)

    # Combine and deduplicate
    all_pairs = pairs_d1 + pairs_d5
    df = pd.DataFrame(all_pairs).drop_duplicates(subset='epistasis_id').reset_index(drop=True)

    df['source'] = 'kras_neighborhood'
    df['pair_id'] = [f'kras_{i+1:05d}' for i in range(len(df))]
    df['gene'] = 'KRAS'
    df['chr'] = KRAS_CHROM
    df['label'] = 'synthetic'
    df['genome_build'] = 'hg38'

    print(f"KRAS neighborhood: {len(df)} pairs, distance range: "
          f"{df['distance_bp'].min()}-{df['distance_bp'].max()} bp")

    return df


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    result = parse()
    result.to_csv(OUTPUT_DIR / 'kras_neighborhood_pairs.tsv', sep='\t', index=False)
    print(f"\nSaved {len(result)} KRAS pairs")
