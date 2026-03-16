"""Generate SNV x SNV variant pair epistasis_ids for tRNA41.

tRNA41 (chr1:159141611-159141684, hg38, negative strand) is a 74-bp tRNA
gene with a known cloverleaf secondary structure (21 base pairs).  Unlike
Ke SL6, this gene has real genomic coordinates so it can be included in the
combined variant pair pipeline.

The cloverleaf structure is derived from the published annotation (dot-
bracket with alignment gaps removed).  All SNV x SNV double variants within
the 74bp gene are generated, with a ``base_paired`` column indicating
whether the two positions form a Watson-Crick pair in the true structure.

For model benchmarking we use the DEPENDENCY MAP approach: the model
predicts which position pairs interact, and we evaluate against the known
contact map.  This was validated in the trna_dependency_map notebook where
SpeciesLM achieved Pearson r = 0.85 (perturbation map vs true contacts).

Output files:
  trna41_pairs.tsv       — 24,309 SNV x SNV doubles with base_paired labels
  trna41_singles.tsv     — 222 single SNVs (for additive baselines)
  trna41_contact_map.tsv — 74 x 74 binary contact map from cloverleaf
"""

from itertools import combinations, product

import numpy as np
import pandas as pd
from seqmat import SeqMat

from .common import OUTPUT_DIR

GENE = 'tRNA41'
CHROM = '1'
STRAND = 'N'
START = 159_141_611
END = 159_141_684
BASES = 'ATGC'

# Known cloverleaf secondary structure (from published annotation).
# '>' denotes 5' partner, '<' denotes 3' partner in base-pair.
# Dots in _REFSEQ_RAW mark alignment gaps (introns / variable-loop padding);
# positions where refseq == '.' are removed before mapping to genomic coords.
_SS_RAW = (
    '.>>>>>>>..>>>>..........<<<<.>>>>>'
    '.........................<<<<<'
    '................>>>>>.......<<<<<<<<<<<<.'
)
_REFSEQ_RAW = (
    '.GTCTCTGTGGCGCAATGGAcgA.GCGCGCTGGACTTCTA'
    '..................ATCCAGAG'
    '...........GtTCCGGGTTCGAGTCCCGGCAGAGATG'
)


def _build_contact_map():
    """Build 74x74 binary contact map from dot-bracket structure.

    Returns (contact_matrix, dot_bracket_string_filtered).
    """
    ss = ''.join(s for s, r in zip(_SS_RAW, _REFSEQ_RAW) if r != '.')
    ss = ss.replace('>', '(').replace('<', ')')

    N = len(ss)
    contact = np.zeros((N, N), dtype=int)

    stack = []
    for i, ch in enumerate(ss):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            j = stack.pop()
            contact[j, i] = 1
            contact[i, j] = 1

    return contact, ss


def parse():
    """Generate singles and doubles for tRNA41 with structure labels.

    Returns (singles_df, pairs_df, contact_map_matrix).
    """
    s = SeqMat.from_fasta('hg38', 'chr1', START, END)
    seq = s.seq.upper()
    indices = s.index
    L = len(seq)

    contact_map, ss = _build_contact_map()
    assert L == len(ss) == 74, f"Expected 74bp, got seq={L}, ss={len(ss)}"

    # --- Singles ---
    single_rows = []
    for i in range(L):
        pos = int(indices[i])
        ref = seq[i]
        if ref not in BASES:
            continue
        for alt in BASES:
            if alt == ref:
                continue
            mid = f'{GENE}:{CHROM}:{pos}:{ref}:{alt}:{STRAND}'
            single_rows.append({
                'mutation_id': mid,
                'gene': GENE,
                'chr': CHROM,
                'pos': pos,
                'ref': ref,
                'alt': alt,
            })
    singles_df = pd.DataFrame(single_rows)

    # --- Doubles ---
    rows = []
    for i, j in combinations(range(L), 2):
        pos1, pos2 = int(indices[i]), int(indices[j])
        ref1, ref2 = seq[i], seq[j]

        if ref1 not in BASES or ref2 not in BASES:
            continue

        is_paired = bool(contact_map[i, j])

        alts1 = [b for b in BASES if b != ref1]
        alts2 = [b for b in BASES if b != ref2]

        for alt1, alt2 in product(alts1, alts2):
            eid = (f"{GENE}:{CHROM}:{pos1}:{ref1}:{alt1}:{STRAND}|"
                   f"{GENE}:{CHROM}:{pos2}:{ref2}:{alt2}:{STRAND}")
            rows.append({
                'source': 'trna41',
                'gene': GENE,
                'chr': CHROM,
                'pos1': pos1,
                'pos2': pos2,
                'ref1': ref1,
                'alt1': alt1,
                'ref2': ref2,
                'alt2': alt2,
                'distance_bp': pos2 - pos1,
                'base_paired': is_paired,
                'label': 'synthetic',
                'genome_build': 'hg38',
                'epistasis_id': eid,
            })

    df = pd.DataFrame(rows)
    df['pair_id'] = [f'trna41_{i + 1:06d}' for i in range(len(df))]

    n_paired = df['base_paired'].sum()
    n_total = len(df)
    n_bp_positions = contact_map.sum() // 2
    print(f"tRNA41: {len(singles_df)} single mutations, {n_total:,} variant pairs ({L}bp)")
    print(f"  Base-paired position pairs: {n_bp_positions}")
    print(f"  Variant pairs at base-paired positions: {n_paired:,}")
    print(f"  Variant pairs at non-paired positions: {n_total - n_paired:,}")

    return singles_df, df, contact_map


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    singles, pairs, cmap = parse()

    singles_path = OUTPUT_DIR / 'trna41_singles.tsv'
    singles.to_csv(singles_path, sep='\t', index=False)
    print(f"\nSaved {len(singles)} singles to {singles_path}")

    out_path = OUTPUT_DIR / 'trna41_pairs.tsv'
    pairs.to_csv(out_path, sep='\t', index=False)
    print(f"Saved {len(pairs):,} doubles to {out_path}")

    cmap_path = OUTPUT_DIR / 'trna41_contact_map.tsv'
    positions = list(range(START, END + 1))
    cmap_df = pd.DataFrame(cmap, index=positions, columns=positions)
    cmap_df.to_csv(cmap_path, sep='\t')
    print(f"Saved contact map ({cmap.shape[0]}x{cmap.shape[1]}) to {cmap_path}")
