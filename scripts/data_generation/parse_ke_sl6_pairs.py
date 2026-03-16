"""Generate SNV x SNV variant pair epistasis_ids for the Ke SL6 construct.

Ke et al. (Genome Res 2018) performed saturation mutagenesis of a 51-nt
internal exon (WT1-5) in a three-exon minigene.  Every tandem dinucleotide
(DBS) and every single base (SBS) at exon positions 2-48 was substituted in
10 hexamer backgrounds (HMA-HMJ).  HMA is the true wild-type; HMB-HMJ
replace a 6-mer at exon positions 5-10.

The 90-nt construct (23 nt upstream intron + 51 nt exon + 16 nt downstream
intron) folds into a stem-loop called SL6 (positions 8-22 in exon coords,
string indices ~30-44).  This is the ONLY secondary structure found to
correlate with splicing.  In HMA the SL6 stem is intact; in HMB-HMJ the
hexamer change at positions 5-10 disrupts the upstream arm, destroying the
stem.  Mutations in the downstream arm (exon 17-22) therefore show
structure-dependent (epistatic) effects in HMA but not in other HMs.

IMPORTANT — the Ke data does NOT contain pairwise SNV epistasis
measurements.  Each construct carries exactly ONE mutation (SBS or DBS at a
single position).  The "structural epistasis" in the original analysis
(LEI_HMA - mean_LEI_other) captures the interaction between a point
mutation and the 6-base hexamer context, not between two independent SNVs.

For model benchmarking we therefore use the DEPENDENCY MAP approach:
generate all possible SNV x SNV pairs in the 90-nt WT sequence, label each
pair with whether the two positions are base-paired in the MFE structure,
and ask the model to predict which positions interact.  Singles are also
provided so the model can compute additive expectations.

Output files (all standalone, not in the combined pipeline):
  ke_sl6_pairs.tsv       — 36,045 SNV x SNV doubles with base_paired labels
  ke_sl6_singles.tsv     — 270 single SNVs with experimental LEI where available
  ke_sl6_contact_map.tsv — 90 x 90 binary contact map from MFE structure
  ke_sl6_sequence.txt    — the 90-nt WT (HMA) sequence
"""

from itertools import combinations, product

import numpy as np
import pandas as pd

from .common import DATA_DIR, OUTPUT_DIR

KE_EXCEL = DATA_DIR / 'Supplemental_Table_S2.xlsx'
POSITION_OFFSET = 22  # Ke exon position N = string index N + 22
BASES = 'ATGC'

# 90-nt construct: 23 nt intron + 51 nt exon + 16 nt intron
SL6_WT_SEQ = (
    'CCCCACCTCTTCTTCTTTTCTAGAGTTGCTGCTGGGAGCTCCAGCACAGT'
    'GAAATGGACAGAAGGGCAGAGCAAGTGAGTGGACAATGCG'
)
# MFE dot-bracket (91 chars in data; trailing dot trimmed to match 90-nt seq)
SL6_WT_SS = (
    '((((((((((((((((..((((...((((((((((.....)))))..)))))..)))).))))))..))))...))).).))........'
)

assert len(SL6_WT_SEQ) == 90
assert len(SL6_WT_SS) == 90


# -----------------------------------------------------------------------
# Structure helpers
# -----------------------------------------------------------------------
def _build_contact_map(ss):
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
    return contact


# -----------------------------------------------------------------------
# Experimental data loading
# -----------------------------------------------------------------------
def _load_experimental():
    """Load SBS LEI from Ke Excel (mean across hex backgrounds).

    Returns
    -------
    sbs_lei : dict  (string_index, alt) -> mean LEI across hex backgrounds
    """
    df = pd.read_excel(KE_EXCEL, sheet_name='Main', header=0)

    muts = df[df['Posn #'].notna() & (df['Posn #'] > 0)].copy()
    muts['Posn #'] = muts['Posn #'].astype(int)

    sbs = muts[muts['SBS / DBS'] == 'S']

    # SBS: parse single-base change from the dinucleotide notation
    # base_change "XY-AY" -> first base X->A at position N (string index N+22)
    sbs_lei = {}
    for (posn, bc), grp in sbs.groupby(['Posn #', 'base change(s)']):
        ref_dinuc, mut_dinuc = bc.split('-')
        si = int(posn) + POSITION_OFFSET
        alt = mut_dinuc[0]
        sbs_lei[(si, alt)] = grp['LEI'].mean()

    return sbs_lei


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def parse():
    """Generate singles and doubles tables for SL6.

    Returns (singles_df, doubles_df, contact_map).
    """
    seq = SL6_WT_SEQ
    L = len(seq)
    contact_map = _build_contact_map(SL6_WT_SS)

    sbs_lei = _load_experimental()
    print(f"Experimental SBS entries: {len(sbs_lei)}")

    # --- Singles ---
    single_rows = []
    for i in range(L):
        ref = seq[i]
        if ref not in BASES:
            continue
        for alt in BASES:
            if alt == ref:
                continue
            mid = f'SL6:0:{i}:{ref}:{alt}:P'
            lei = sbs_lei.get((i, alt), np.nan)
            single_rows.append({
                'mutation_id': mid,
                'gene': 'SL6',
                'pos': i,
                'ref': ref,
                'alt': alt,
                'experimental_lei': lei,
            })
    singles_df = pd.DataFrame(single_rows)
    n_with_lei = singles_df['experimental_lei'].notna().sum()
    print(f"Singles: {len(singles_df)} mutations, {n_with_lei} with experimental LEI")

    # --- Doubles ---
    double_rows = []
    for i, j in combinations(range(L), 2):
        ref1, ref2 = seq[i], seq[j]
        if ref1 not in BASES or ref2 not in BASES:
            continue
        is_paired = bool(contact_map[i, j])

        alts1 = [b for b in BASES if b != ref1]
        alts2 = [b for b in BASES if b != ref2]

        for alt1, alt2 in product(alts1, alts2):
            eid = f'SL6:0:{i}:{ref1}:{alt1}:P|SL6:0:{j}:{ref2}:{alt2}:P'

            double_rows.append({
                'source': 'ke_sl6',
                'gene': 'SL6',
                'pos1': i,
                'pos2': j,
                'ref1': ref1,
                'alt1': alt1,
                'ref2': ref2,
                'alt2': alt2,
                'distance': j - i,
                'base_paired': is_paired,
                'label': 'synthetic',
                'epistasis_id': eid,
            })

    doubles_df = pd.DataFrame(double_rows)
    doubles_df['pair_id'] = [f'ke_sl6_{i + 1:06d}' for i in range(len(doubles_df))]

    n_paired = doubles_df['base_paired'].sum()
    print(f"Doubles: {len(doubles_df):,} variant pairs")
    print(f"  Base-paired position pairs: {contact_map.sum() // 2}")
    print(f"  Variant pairs at base-paired positions: {n_paired:,}")
    print(f"  Non-paired: {len(doubles_df) - n_paired:,}")

    return singles_df, doubles_df, contact_map


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    singles, doubles, cmap = parse()

    singles_path = OUTPUT_DIR / 'ke_sl6_singles.tsv'
    singles.to_csv(singles_path, sep='\t', index=False)
    print(f"\nSaved {len(singles)} singles to {singles_path}")

    doubles_path = OUTPUT_DIR / 'ke_sl6_pairs.tsv'
    doubles.to_csv(doubles_path, sep='\t', index=False)
    print(f"Saved {len(doubles):,} doubles to {doubles_path}")

    cmap_path = OUTPUT_DIR / 'ke_sl6_contact_map.tsv'
    cmap_df = pd.DataFrame(cmap, index=range(90), columns=range(90))
    cmap_df.to_csv(cmap_path, sep='\t')
    print(f"Saved contact map (90x90) to {cmap_path}")

    seq_path = OUTPUT_DIR / 'ke_sl6_sequence.txt'
    with open(seq_path, 'w') as f:
        f.write(SL6_WT_SEQ + '\n')
    print(f"Saved WT sequence to {seq_path}")
