"""Parse TCGA somatic co-occurring mutation doubles via DuckDB."""

import sys

import numpy as np
import pandas as pd

from .common import TCGA_PARQUET, PARSETCGA_ROOT, OUTPUT_DIR

# Add parent of parsetcga package to path for epistasis_id helper
_parsetcga_parent = str(PARSETCGA_ROOT.parent)
if _parsetcga_parent not in sys.path:
    sys.path.insert(0, _parsetcga_parent)

MAX_DIST_BP = 100
MIN_DEPTH = 15
ALT_MIN = 15
MIN_VAF = 0.10
MIN_COND_PROB = 0.80
MAX_CASES_BOTH = 50
MIN_CASES_BOTH = 3
EXCLUDE_TOP_BURDEN_GENES = 20


def parse():
    """Return DataFrame of TCGA co-occurring mutation pairs.

    Positions are hg38 native. epistasis_id deferred to aggregation.
    """
    import duckdb
    from parsetcga.ids import epistasis_id as make_tcga_epistasis_id

    # Step 1: Load mutations with quality filters
    print("Loading TCGA mutations...")
    con = duckdb.connect()
    q = """
    SELECT
        case_id, Gene_name, Chromosome, Start_Position,
        (Gene_name || ':' || REPLACE(Chromosome, 'chr', '') || ':'
         || CAST(Start_Position AS VARCHAR) || ':'
         || UPPER(COALESCE(Reference_Allele, '')) || ':'
         || UPPER(COALESCE(Tumor_Seq_Allele2, ''))) AS mutation_id,
        UPPER(COALESCE(Reference_Allele, '')) AS ref_allele,
        UPPER(COALESCE(Tumor_Seq_Allele2, '')) AS alt_allele,
        t_depth, t_alt_count
    FROM read_parquet(?)
    WHERE t_depth >= ? AND COALESCE(t_alt_count, 0) >= ?
    """
    raw = con.execute(q, [str(TCGA_PARQUET), MIN_DEPTH, ALT_MIN]).fetchdf()
    con.close()

    raw['pos'] = raw['Start_Position'].astype(int)
    raw['vaf'] = (raw['t_alt_count'] / raw['t_depth'].clip(lower=1)).astype(float)
    raw['chrom'] = raw['Chromosome'].astype(str).str.replace('chr', '', regex=False)
    print(f"  Total mutations (depth>={MIN_DEPTH}, alt>={ALT_MIN}): {len(raw):,}")

    # Exclude top burden genes
    gene_cts = raw['Gene_name'].value_counts()
    excluded = set(gene_cts.head(EXCLUDE_TOP_BURDEN_GENES).index)
    raw = raw[~raw['Gene_name'].isin(excluded)].copy()
    print(f"  Excluded top {EXCLUDE_TOP_BURDEN_GENES} burden genes: "
          f"{', '.join(sorted(excluded)[:5])}...")
    print(f"  Remaining: {len(raw):,} rows")

    # Step 2: Find candidate pairs (same case, same chrom, within MAX_DIST_BP)
    print("Finding candidate pairs...")
    pairs = []
    for (case_id, chrom), grp in raw.groupby(['case_id', 'chrom']):
        grp = grp.sort_values('pos').reset_index(drop=True)
        if len(grp) < 2:
            continue
        pos = grp['pos'].values
        vafs = grp['vaf'].values
        genes = grp['Gene_name'].values
        mut_ids = grp['mutation_id'].values
        refs = grp['ref_allele'].values
        alts = grp['alt_allele'].values

        for i in range(len(grp)):
            hi = np.searchsorted(pos, pos[i] + MAX_DIST_BP, side='right')
            for j in range(i + 1, hi):
                if pos[j] - pos[i] <= 0:
                    continue
                if vafs[i] < MIN_VAF or vafs[j] < MIN_VAF:
                    continue
                pairs.append({
                    'case_id': case_id,
                    'epistasis_id': make_tcga_epistasis_id([mut_ids[i], mut_ids[j]]),
                    'mut1_id': mut_ids[i],
                    'mut2_id': mut_ids[j],
                    'gene1': str(genes[i]).strip(),
                    'gene2': str(genes[j]).strip(),
                    'chrom': chrom,
                    'pos1': int(pos[i]),
                    'pos2': int(pos[j]),
                    'ref1': refs[i],
                    'alt1': alts[i],
                    'ref2': refs[j],
                    'alt2': alts[j],
                    'distance_bp': int(pos[j] - pos[i]),
                })

    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        print("  No candidate pairs found!")
        return pd.DataFrame()
    pairs_df = pairs_df.drop_duplicates(subset=['epistasis_id', 'case_id'])
    print(f"  Candidate pairs: {len(pairs_df):,} case-epistasis, "
          f"{pairs_df['epistasis_id'].nunique():,} unique")

    # Step 3: Co-occurrence statistics
    print("Computing co-occurrence statistics...")
    # Build global mutation -> case_id sets from FULL parquet (no depth/alt filter)
    # to match the notebook's build_mutation_lookup() approach
    all_muts = list(set(pairs_df['mut1_id']) | set(pairs_df['mut2_id']))
    print(f"  Looking up {len(all_muts):,} unique mutations in full parquet...")
    con = duckdb.connect()
    lookup_parts = []
    for i in range(0, len(all_muts), 50_000):
        chunk = all_muts[i:i + 50_000]
        lookup_parts.append(con.execute("""
            SELECT
                (Gene_name || ':' || REPLACE(Chromosome, 'chr', '') || ':'
                 || CAST(Start_Position AS VARCHAR) || ':'
                 || UPPER(COALESCE(Reference_Allele, '')) || ':'
                 || UPPER(COALESCE(Tumor_Seq_Allele2, ''))) AS mutation_id,
                case_id
            FROM read_parquet(?)
            WHERE (Gene_name || ':' || REPLACE(Chromosome, 'chr', '') || ':'
                   || CAST(Start_Position AS VARCHAR) || ':'
                   || UPPER(COALESCE(Reference_Allele, '')) || ':'
                   || UPPER(COALESCE(Tumor_Seq_Allele2, ''))) IN (SELECT unnest(?))
        """, [str(TCGA_PARQUET), chunk]).fetchdf())
    con.close()
    lookup_df = pd.concat(lookup_parts, ignore_index=True)
    global_m2c = lookup_df.groupby('mutation_id')['case_id'].apply(set).to_dict()
    print(f"  Resolved case sets for {len(global_m2c):,} mutations")

    epi_info = pairs_df.groupby('epistasis_id', as_index=False).agg(
        mut1_id=('mut1_id', 'first'),
        mut2_id=('mut2_id', 'first'),
        gene1=('gene1', 'first'),
        gene2=('gene2', 'first'),
        chrom=('chrom', 'first'),
        pos1=('pos1', 'first'),
        pos2=('pos2', 'first'),
        ref1=('ref1', 'first'),
        alt1=('alt1', 'first'),
        ref2=('ref2', 'first'),
        alt2=('alt2', 'first'),
        distance_bp=('distance_bp', 'median'),
    )

    rows = []
    for _, r in epi_info.iterrows():
        s1 = global_m2c.get(r['mut1_id'], set())
        s2 = global_m2c.get(r['mut2_id'], set())
        both = s1 & s2
        only1 = s1 - s2
        only2 = s2 - s1
        nb, no1, no2 = len(both), len(only1), len(only2)

        n_mut1 = nb + no1
        n_mut2 = nb + no2
        p_mut1_given_mut2 = nb / max(n_mut2, 1)
        p_mut2_given_mut1 = nb / max(n_mut1, 1)
        cond_max = max(p_mut1_given_mut2, p_mut2_given_mut1)

        # Label anchor vs dependent
        if p_mut1_given_mut2 >= p_mut2_given_mut1:
            anc_gene, dep_gene = r['gene1'], r['gene2']
        else:
            anc_gene, dep_gene = r['gene2'], r['gene1']

        rows.append({
            'epistasis_id': r['epistasis_id'],
            'chr': r['chrom'],
            'pos1': int(r['pos1']),
            'pos2': int(r['pos2']),
            'ref1': r['ref1'],
            'alt1': r['alt1'],
            'ref2': r['ref2'],
            'alt2': r['alt2'],
            'distance_bp': int(r['distance_bp']) if pd.notna(r['distance_bp']) else 0,
            'n_cases_both': nb,
            'cond_prob_dependent': cond_max,
            'anchor_gene': anc_gene,
            'dependent_gene': dep_gene,
        })

    epi_df = pd.DataFrame(rows)

    # Step 4: Filter
    epi_df = epi_df[
        (epi_df['n_cases_both'] >= MIN_CASES_BOTH)
        & (epi_df['n_cases_both'] <= MAX_CASES_BOTH)
        & (epi_df['cond_prob_dependent'] >= MIN_COND_PROB)
    ].sort_values(
        ['cond_prob_dependent', 'n_cases_both'], ascending=[False, False]
    ).reset_index(drop=True)

    print(f"  After filters ({MIN_CASES_BOTH}-{MAX_CASES_BOTH} patients, "
          f"cond_prob >= {MIN_COND_PROB}): {len(epi_df)} pairs")

    # Build output
    epi_df['source'] = 'tcga_doubles'
    epi_df['pair_id'] = [f'tcga_{i+1:05d}' for i in range(len(epi_df))]
    epi_df['gene'] = epi_df['anchor_gene']
    epi_df['label'] = 'positive'
    epi_df['genome_build'] = 'hg38'

    return epi_df


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    result = parse()
    if len(result) > 0:
        result.to_csv(OUTPUT_DIR / 'tcga_doubles_pairs.tsv', sep='\t', index=False)
        print(f"\nSaved {len(result)} TCGA pairs")
    else:
        print("\nNo TCGA pairs to save")
