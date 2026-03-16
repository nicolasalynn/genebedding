"""Parse Ke et al. SL6 stem-loop structural epistasis (standalone analysis).

Computes structural_epistasis = LEI(HexA) - mean(LEI(HexB-J)) for each
unique (position, mutation) in the DBS (double base substitution) data.

SL6 downstream arm positions 17-22 are labeled 'positive' (structure-mediated
epistasis expected), neutral zone 30-48 labeled 'null'.

Output is a standalone analysis table, not merged into the combined pipeline.
"""

import numpy as np
import pandas as pd

from .common import DATA_DIR, OUTPUT_DIR

KE_EXCEL = DATA_DIR / 'Supplemental_Table_S2.xlsx'

# Region definitions
SL6_ARM_POSITIONS = set(range(17, 23))   # 17-22 inclusive
NEUTRAL_POSITIONS = set(range(30, 49))   # 30-48 inclusive

# Hexamer backgrounds
HEX_A = 'A'
HEX_BJ = list('BCDEFGHIJ')


def parse():
    """Return DataFrame of SL6 structural epistasis analysis.

    One row per unique (position, base_change) with structural epistasis
    computed across hexamer backgrounds.
    """
    df = pd.read_excel(KE_EXCEL, sheet_name='Main', header=0)

    # Drop wild-type/header rows (Posn # == 0 or NaN)
    df = df[df['Posn #'].notna() & (df['Posn #'] > 0)].copy()
    df['Posn #'] = df['Posn #'].astype(int)

    # Filter to DBS (double base substitutions = dinucleotide mutations)
    # Column values are 'D' (DBS) and 'S' (SBS), not full words
    dbs = df[df['SBS / DBS'] == 'D'].copy()
    print(f"Ke et al: {len(dbs)} DBS rows, {dbs['Posn #'].nunique()} positions, "
          f"{dbs['Hex-mut #'].nunique()} hex backgrounds")

    # Compute structural epistasis per (position, mutation)
    rows = []
    for (posn, bc), grp in dbs.groupby(['Posn #', 'base change(s)']):
        hex_lei = grp.set_index('Hex-mut #')['LEI'].to_dict()
        hex_leisc = grp.set_index('Hex-mut #')['LEIsc'].to_dict()

        lei_a = hex_lei.get(HEX_A, np.nan)
        lei_bj = [hex_lei.get(h, np.nan) for h in HEX_BJ]
        lei_bj_valid = [v for v in lei_bj if pd.notna(v)]
        mean_lei_bj = np.mean(lei_bj_valid) if lei_bj_valid else np.nan

        leisc_a = hex_leisc.get(HEX_A, np.nan)
        leisc_bj = [hex_leisc.get(h, np.nan) for h in HEX_BJ]
        leisc_bj_valid = [v for v in leisc_bj if pd.notna(v)]
        mean_leisc_bj = np.mean(leisc_bj_valid) if leisc_bj_valid else np.nan

        struct_epi = (lei_a - mean_lei_bj
                      if pd.notna(lei_a) and pd.notna(mean_lei_bj) else np.nan)
        struct_epi_sc = (leisc_a - mean_leisc_bj
                         if pd.notna(leisc_a) and pd.notna(mean_leisc_bj) else np.nan)

        # Region/label
        if posn in SL6_ARM_POSITIONS:
            region, label = 'SL6_arm', 'positive'
        elif posn in NEUTRAL_POSITIONS:
            region, label = 'neutral', 'null'
        else:
            region, label = 'other', 'unlabeled'

        ref_dinuc, mut_dinuc = bc.split('-') if '-' in bc else (bc, '')

        rows.append({
            'posn': posn,
            'base_change': bc,
            'ref_dinuc': ref_dinuc,
            'mut_dinuc': mut_dinuc,
            'LEI_hexA': lei_a,
            'mean_LEI_hexBJ': mean_lei_bj,
            'structural_epistasis': struct_epi,
            'LEIsc_hexA': leisc_a,
            'mean_LEIsc_hexBJ': mean_leisc_bj,
            'structural_epistasis_sc': struct_epi_sc,
            'n_hex_backgrounds': len(lei_bj_valid) + (1 if pd.notna(lei_a) else 0),
            'region': region,
            'label': label,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values(['posn', 'base_change']).reset_index(drop=True)
    result['pair_id'] = [f'ke_sl6_{i+1:04d}' for i in range(len(result))]

    # Summary stats
    sl6 = result[result['region'] == 'SL6_arm']
    neutral = result[result['region'] == 'neutral']
    print(f"Ke SL6 analysis: {len(result)} unique (position, mutation) combinations")
    print(f"  SL6 arm (pos 17-22): {len(sl6)}, "
          f"mean structural epistasis = {sl6['structural_epistasis'].mean():.4f}")
    print(f"  Neutral (pos 30-48): {len(neutral)}, "
          f"mean structural epistasis = {neutral['structural_epistasis'].mean():.4f}")

    return result


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    result = parse()
    out_path = OUTPUT_DIR / 'ke_sl6_analysis.tsv'
    result.to_csv(out_path, sep='\t', index=False)
    print(f"\nSaved {len(result)} rows to {out_path}")
