"""Parse Uebbing MPRA variant pairs (positive + null)."""

from itertools import combinations

import numpy as np
import pandas as pd

from .common import (
    DISTANCE_THRESHOLD, UEBBING_FILE, OUTPUT_DIR, liftover_dataframe,
)


def _parse_base_change(bc):
    if pd.isna(bc):
        return pd.NA, pd.NA
    parts = str(bc).split('->')
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return pd.NA, pd.NA


def parse():
    """Return (mpra_positive, mpra_null) DataFrames.

    Positions are liftover'd to hg38 and distance-filtered.
    No epistasis_id or ref validation (deferred to aggregation).
    """
    # Load
    ueb = pd.read_excel(UEBBING_FILE, sheet_name='Dataset S4', header=2)
    print(f"Loaded {len(ueb)} hSubs from Uebbing MPRA")

    # Parse alleles
    ueb[['ref_allele', 'alt_allele']] = ueb['baseChange'].apply(
        lambda x: pd.Series(_parse_base_change(x))
    )
    ueb['Pos'] = pd.to_numeric(ueb['Pos'], errors='coerce').astype('Int64')
    ueb['Chr'] = ueb['Chr'].astype(str)

    # Split into groups
    hsub_interacting_mask = ueb['interactions'].str.contains('hSub', na=False)
    additive_mask = ueb['interactions'].isna()

    interactive_hsub = ueb[hsub_interacting_mask].copy()
    additive_only = ueb[additive_mask].copy()

    # ---- Positive pairs ----
    positive_pairs = []
    pair_id_counter = 0
    for enhancer, group in interactive_hsub.groupby('Enhancer'):
        n_hsubs = len(group)
        ambiguity = 'unambiguous' if n_hsubs == 2 else 'ambiguous'
        for (idx1, row1), (idx2, row2) in combinations(group.iterrows(), 2):
            pair_id_counter += 1
            distance = (abs(int(row1['Pos']) - int(row2['Pos']))
                        if pd.notna(row1['Pos']) and pd.notna(row2['Pos']) else pd.NA)
            positive_pairs.append({
                'source': 'uebbing_mpra',
                'pair_id': f'mpra_pos_{pair_id_counter:04d}',
                'enhancer': enhancer,
                'chr': row1['Chr'],
                'pos1': row1['Pos'],
                'pos2': row2['Pos'],
                'ref1': row1['ref_allele'],
                'alt1': row1['alt_allele'],
                'ref2': row2['ref_allele'],
                'alt2': row2['alt_allele'],
                'distance_bp': distance,
                'label': 'positive',
                'pair_ambiguity': ambiguity,
                'effect_size_1': row1.get('max_intera_effect_size', pd.NA),
                'effect_size_2': row2.get('max_intera_effect_size', pd.NA),
                'interaction_effect_size': np.nanmean([
                    row1.get('max_intera_effect_size', np.nan),
                    row2.get('max_intera_effect_size', np.nan)]),
                'genome_build': 'hg19',
            })

    mpra_positive = pd.DataFrame(positive_pairs)
    print(f"Positive pairs (pre-liftover): {len(mpra_positive)}")

    # ---- Null pairs ----
    interacting_enhancers = set(interactive_hsub['Enhancer'].unique())
    additive_in_interacting = additive_only[additive_only['Enhancer'].isin(interacting_enhancers)].copy()

    null_pairs = []
    pair_id_counter = 0

    # Within-enhancer
    for enhancer, group in additive_in_interacting.groupby('Enhancer'):
        if len(group) < 2:
            continue
        for (idx1, row1), (idx2, row2) in combinations(group.iterrows(), 2):
            pair_id_counter += 1
            distance = (abs(int(row1['Pos']) - int(row2['Pos']))
                        if pd.notna(row1['Pos']) and pd.notna(row2['Pos']) else pd.NA)
            null_pairs.append({
                'source': 'uebbing_mpra',
                'pair_id': f'mpra_null_{pair_id_counter:04d}',
                'enhancer': enhancer,
                'chr': row1['Chr'],
                'pos1': row1['Pos'],
                'pos2': row2['Pos'],
                'ref1': row1['ref_allele'],
                'alt1': row1['alt_allele'],
                'ref2': row2['ref_allele'],
                'alt2': row2['alt_allele'],
                'distance_bp': distance,
                'label': 'null',
                'pair_ambiguity': 'within_enhancer',
                'effect_size_1': row1['max_add_effect_size'],
                'effect_size_2': row2['max_add_effect_size'],
                'interaction_effect_size': pd.NA,
                'genome_build': 'hg19',
            })

    within_null_count = len(null_pairs)

    # Cross-enhancer
    additive_by_enhancer = {e: g for e, g in additive_in_interacting.groupby('Enhancer')}
    enhancer_list = sorted(additive_by_enhancer.keys())
    for i in range(len(enhancer_list)):
        e1 = enhancer_list[i]
        g1 = additive_by_enhancer[e1]
        for j in range(i + 1, len(enhancer_list)):
            e2 = enhancer_list[j]
            g2 = additive_by_enhancer[e2]
            for _, row1 in g1.iterrows():
                for _, row2 in g2.iterrows():
                    if row1['Chr'] != row2['Chr']:
                        continue
                    pair_id_counter += 1
                    distance = (abs(int(row1['Pos']) - int(row2['Pos']))
                                if pd.notna(row1['Pos']) and pd.notna(row2['Pos']) else pd.NA)
                    null_pairs.append({
                        'source': 'uebbing_mpra',
                        'pair_id': f'mpra_null_{pair_id_counter:04d}',
                        'enhancer': f'{e1}|{e2}',
                        'chr': row1['Chr'],
                        'pos1': row1['Pos'],
                        'pos2': row2['Pos'],
                        'ref1': row1['ref_allele'],
                        'alt1': row1['alt_allele'],
                        'ref2': row2['ref_allele'],
                        'alt2': row2['alt_allele'],
                        'distance_bp': distance,
                        'label': 'null',
                        'pair_ambiguity': 'cross_enhancer',
                        'effect_size_1': row1['max_add_effect_size'],
                        'effect_size_2': row2['max_add_effect_size'],
                        'interaction_effect_size': pd.NA,
                        'genome_build': 'hg19',
                    })

    mpra_null_cols = [
        'source', 'pair_id', 'enhancer', 'chr', 'pos1', 'pos2', 'ref1', 'alt1',
        'ref2', 'alt2', 'distance_bp', 'label', 'pair_ambiguity',
        'effect_size_1', 'effect_size_2', 'interaction_effect_size', 'genome_build',
    ]
    mpra_null = (pd.DataFrame(null_pairs, columns=mpra_null_cols)
                 if null_pairs else pd.DataFrame(columns=mpra_null_cols))
    print(f"Null pairs (within: {within_null_count}, cross: {len(null_pairs) - within_null_count})")

    # ---- Liftover ----
    print("\nLifting MPRA positive pairs...")
    if len(mpra_positive) > 0:
        mpra_positive = liftover_dataframe(mpra_positive)

    print("Lifting MPRA null pairs...")
    if len(mpra_null) > 0:
        mpra_null = liftover_dataframe(mpra_null)

    return mpra_positive, mpra_null


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    pos, null = parse()
    pos.to_csv(OUTPUT_DIR / 'mpra_positive_pairs.tsv', sep='\t', index=False)
    null.to_csv(OUTPUT_DIR / 'mpra_null_pairs.tsv', sep='\t', index=False)
    print(f"\nSaved {len(pos)} positive, {len(null)} null pairs")
