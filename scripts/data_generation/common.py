"""Shared utilities for expression doubles variant pair parsing."""

import json as _json
import os
import tarfile
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# ---------------------------------------------------------------------------
# Paths — unified under EPISTASIS_PAPER_ROOT
# ---------------------------------------------------------------------------
# All data lives under EPISTASIS_PAPER_ROOT. Override via env var.
# Default: /tamir2/nicolaslynn/data/epistasis_paper (cluster) or ~/data/epistasis_paper.
_SCRIPT_DIR = Path(__file__).resolve().parent
_CLUSTER_ROOT = Path("/tamir2/nicolaslynn/data/epistasis_paper")
_HOME_ROOT = Path.home() / "data" / "epistasis_paper"
_DEFAULT_ROOT = _CLUSTER_ROOT if _CLUSTER_ROOT.parent.exists() else _HOME_ROOT

EPISTASIS_PAPER_ROOT = Path(os.environ.get("EPISTASIS_PAPER_ROOT", str(_DEFAULT_ROOT))).resolve()

DATA_DIR = EPISTASIS_PAPER_ROOT / 'data' / 'source'
OUTPUT_DIR = EPISTASIS_PAPER_ROOT / 'data'
PARSETCGA_ROOT = _SCRIPT_DIR / 'parsetcga'

DISTANCE_THRESHOLD = 6_000  # 6 kb maximum distance between variants in a pair

# GTEx file paths
LCL_TISSUE = 'Cells_EBV-transformed_lymphocytes'
WB_TISSUE = 'Whole_Blood'

UEBBING_FILE = DATA_DIR / 'pnas.2007049118.sd04.xlsx'
YANG_S3_FILE = DATA_DIR / 'suppl_table_S3.xlsx'
YANG_S1_FILE = DATA_DIR / 'suppl_table_S1.xlsx'
BROWN_FILE = DATA_DIR / 'elife-01381-supp1-v2.xlsx'
GTEX_INDEP_TAR = DATA_DIR / 'GTEx_Analysis_v8_eQTL_independent.tar'
GTEX_EQTL_TAR = DATA_DIR / 'GTEx_Analysis_v8_eQTL.tar'
GTEX_INDEP_DIR = DATA_DIR / 'GTEx_Analysis_v8_eQTL_independent'
GTEX_EQTL_DIR = DATA_DIR / 'GTEx_Analysis_v8_eQTL'

FAS_EXCEL = DATA_DIR / '41467_2016_BFncomms11558_MOESM968_ESM.xlsx'
MST1R_LIBRARY_EXCEL = DATA_DIR / '41467_2018_5748_MOESM4_ESM.xlsx'
MST1R_SINGLES_EXCEL = DATA_DIR / '41467_2018_5748_MOESM6_ESM.xlsx'
TCGA_PARQUET = DATA_DIR / 'tcga_all.parquet'


# ---------------------------------------------------------------------------
# GTEx extraction
# ---------------------------------------------------------------------------
def ensure_gtex_extracted():
    """Extract GTEx tar files if not already extracted."""
    for tar_path, extract_dir in [(GTEX_INDEP_TAR, GTEX_INDEP_DIR),
                                   (GTEX_EQTL_TAR, GTEX_EQTL_DIR)]:
        if not extract_dir.is_dir():
            print(f"Extracting {tar_path}...")
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=DATA_DIR)
            print(f"  -> Extracted to {extract_dir}")
        else:
            print(f"  {extract_dir} already exists, skipping extraction")


def gtex_tissue_paths(tissue):
    """Return (independent_eqtl_path, significant_eqtl_path) for a tissue."""
    indep = GTEX_INDEP_DIR / f'{tissue}.v8.independent_eqtls.txt.gz'
    signif = GTEX_EQTL_DIR / f'{tissue}.v8.signif_variant_gene_pairs.txt.gz'
    return indep, signif


# ---------------------------------------------------------------------------
# Liftover  (lazy init so import is fast)
# ---------------------------------------------------------------------------
_liftover = None


def _get_liftover():
    global _liftover
    if _liftover is None:
        from pyliftover import LiftOver
        _liftover = LiftOver('hg19', 'hg38')
    return _liftover


def liftover_pos(chrom, pos):
    """Lift a single position from hg19 to hg38. Returns hg38 pos or pd.NA."""
    if pd.isna(chrom) or pd.isna(pos):
        return pd.NA
    chrom_str = str(chrom)
    if not chrom_str.startswith('chr'):
        chrom_str = 'chr' + chrom_str
    result = _get_liftover().convert_coordinate(chrom_str, int(pos))
    if result and len(result) > 0:
        return int(result[0][1])
    return pd.NA


def liftover_dataframe(df, chr_col='chr', pos1_col='pos1', pos2_col='pos2'):
    """Add pos1_hg38/pos2_hg38 columns, recompute distance, apply 6kb filter.

    Returns filtered DataFrame copy.
    """
    df = df.copy()
    df['pos1_hg38'] = df.apply(lambda r: liftover_pos(r[chr_col], r[pos1_col]), axis=1)
    df['pos2_hg38'] = df.apply(lambda r: liftover_pos(r[chr_col], r[pos2_col]), axis=1)
    df['pos1_hg38'] = pd.to_numeric(df['pos1_hg38'], errors='coerce').astype('Int64')
    df['pos2_hg38'] = pd.to_numeric(df['pos2_hg38'], errors='coerce').astype('Int64')

    lifted = df['pos1_hg38'].notna() & df['pos2_hg38'].notna()
    n_total = len(df)
    n_lifted = lifted.sum()
    print(f"  Both positions lifted: {n_lifted} / {n_total}")

    df.loc[lifted, 'distance_hg38'] = (
        df.loc[lifted, 'pos1_hg38'].astype(int) - df.loc[lifted, 'pos2_hg38'].astype(int)
    ).abs()

    df = df[lifted & (df['distance_hg38'] <= DISTANCE_THRESHOLD)].copy()
    print(f"  After {DISTANCE_THRESHOLD:,} bp filter: {len(df)} (from {n_total})")
    return df


# ---------------------------------------------------------------------------
# GTEx variant_id parsing
# ---------------------------------------------------------------------------
def parse_gtex_variant_id(variant_id):
    """Parse GTEx variant_id format: chr_pos_ref_alt_b38 -> (chrom, pos, ref, alt)."""
    parts = str(variant_id).split('_')
    if len(parts) >= 4:
        chrom = parts[0].replace('chr', '')
        pos = int(parts[1])
        ref = parts[2]
        alt = parts[3]
        return chrom, pos, ref, alt
    return pd.NA, pd.NA, pd.NA, pd.NA


# ---------------------------------------------------------------------------
# Ensembl REST API helpers
# ---------------------------------------------------------------------------
def lookup_rsid_alleles(rsids):
    """Look up ref/alt alleles for rsIDs via Ensembl REST API.

    Returns dict: rsid -> (ref, [alt_alleles]).
    """
    rsid_alleles = {}
    for rsid in sorted(rsids):
        url = f'https://rest.ensembl.org/variation/human/{rsid}?content-type=application/json'
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read().decode())
            for m in data.get('mappings', []):
                if m.get('assembly_name') == 'GRCh38':
                    alleles = m['allele_string'].split('/')
                    rsid_alleles[rsid] = (alleles[0], alleles[1:])
                    break
        except Exception as e:
            print(f"  Failed for {rsid}: {e}")
        time.sleep(0.1)
    return rsid_alleles


_STRAND_CACHE_PATH = DATA_DIR / 'gene_strand_cache.json'


def lookup_gene_strands(gene_symbols):
    """Batch Ensembl lookup for gene strands, with JSON file cache.

    Returns dict: gene_symbol -> 'P' or 'N'.
    """
    # Load cache
    cached = {}
    if _STRAND_CACHE_PATH.exists():
        with open(_STRAND_CACHE_PATH) as f:
            cached = _json.load(f)
        print(f"  Loaded {len(cached)} cached strand lookups")

    gene_strand = {s: cached[s] for s in gene_symbols if s in cached}
    missing = sorted(set(gene_symbols) - set(gene_strand))

    if missing:
        print(f"  Looking up {len(missing)} new gene symbols via Ensembl...")
        for batch_start in range(0, len(missing), 1000):
            batch = missing[batch_start:batch_start + 1000]
            payload = _json.dumps({"symbols": batch}).encode()
            url = 'https://rest.ensembl.org/lookup/symbol/homo_sapiens'
            req = urllib.request.Request(url, data=payload,
                                         headers={'Content-Type': 'application/json',
                                                  'Accept': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = _json.loads(resp.read().decode())
                for symbol, info in data.items():
                    if isinstance(info, dict) and 'strand' in info:
                        strand = 'P' if info['strand'] == 1 else 'N'
                        gene_strand[symbol] = strand
                        cached[symbol] = strand
            except Exception as e:
                print(f"  Batch {batch_start} failed: {e}")
            time.sleep(0.5)

        # Save updated cache
        with open(_STRAND_CACHE_PATH, 'w') as f:
            _json.dump(cached, f, sort_keys=True)
        print(f"  Saved {len(cached)} entries to strand cache")
    else:
        print(f"  All {len(gene_strand)} strands found in cache")

    return gene_strand


_MAF_CACHE_PATH = DATA_DIR / 'variant_maf_cache.json'


def lookup_variant_mafs(variants):
    """Batch Ensembl VEP lookup for gnomAD genome allele frequencies, with JSON file cache.

    Parameters
    ----------
    variants : set of (chrom, pos, ref, alt) tuples

    Returns
    -------
    dict : (chrom, pos, ref, alt) -> float or None
    """
    # Load cache (keys stored as "chrom:pos:ref:alt")
    cached = {}
    if _MAF_CACHE_PATH.exists():
        with open(_MAF_CACHE_PATH) as f:
            cached = _json.load(f)
        print(f"  Loaded {len(cached)} cached MAF lookups")

    maf_lookup = {}
    to_query = []
    for v in variants:
        key = f"{v[0]}:{v[1]}:{v[2]}:{v[3]}"
        if key in cached:
            maf_lookup[v] = cached[key]
        else:
            to_query.append(v)

    print(f"  {len(maf_lookup)} from cache, {len(to_query)} to query via Ensembl VEP")

    if to_query:
        batch_size = 50
        for batch_start in range(0, len(to_query), batch_size):
            batch = to_query[batch_start:batch_start + batch_size]
            # Build VEP region input: "chrom pos pos ref/alt"  (1-based)
            vep_variants = []
            for chrom, pos, ref, alt in batch:
                c = str(chrom).replace('chr', '')
                # For SNVs: start == end == pos
                # For indels this is approximate but sufficient for AF lookup
                end = int(pos) + len(ref) - 1
                allele_str = f"{ref}/{alt}" if ref != '-' else f"-/{alt}"
                vep_variants.append(f"{c} {int(pos)} {end} {allele_str} +")

            payload = _json.dumps({"variants": vep_variants}).encode()
            url = 'https://rest.ensembl.org/vep/homo_sapiens/region'

            results = None
            for attempt in range(3):
                req = urllib.request.Request(
                    url, data=payload,
                    headers={'Content-Type': 'application/json',
                             'Accept': 'application/json'})
                try:
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        results = _json.loads(resp.read().decode())
                    break
                except Exception as e:
                    if attempt < 2:
                        wait = (attempt + 1) * 5
                        print(f"  VEP batch {batch_start} attempt {attempt+1} failed: {e}, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"  VEP batch {batch_start}-{batch_start + len(batch)} failed after 3 attempts: {e}")

            if results is not None:
                for result in results:
                    # Parse input back to identify the variant
                    inp = result.get('input', '')
                    parts = inp.split()
                    if len(parts) < 4:
                        continue
                    r_chrom = parts[0]
                    r_pos = int(parts[1])
                    alleles = parts[3].split('/')
                    r_ref = alleles[0]
                    r_alt = alleles[1] if len(alleles) > 1 else ''

                    af = None
                    for cv in result.get('colocated_variants', []):
                        freqs = cv.get('frequencies', {})
                        alt_freqs = freqs.get(r_alt, {})
                        if 'gnomadg' in alt_freqs:
                            af = alt_freqs['gnomadg']
                            break

                    # Reconstruct the original tuple key (with original chrom format)
                    for v in batch:
                        vc = str(v[0]).replace('chr', '')
                        if vc == r_chrom and int(v[1]) == r_pos and v[2] == r_ref and v[3] == r_alt:
                            maf_lookup[v] = af
                            cached[f"{v[0]}:{v[1]}:{v[2]}:{v[3]}"] = af
                            break

            if batch_start + batch_size < len(to_query):
                time.sleep(0.5)

            if (batch_start // batch_size) % 100 == 0 and batch_start > 0:
                print(f"  ... processed {batch_start + len(batch)} / {len(to_query)} variants")
                # Periodic cache save
                with open(_MAF_CACHE_PATH, 'w') as f:
                    _json.dump(cached, f, sort_keys=True)

        # Fill any variants that weren't returned by VEP
        for v in to_query:
            if v not in maf_lookup:
                maf_lookup[v] = None
                cached[f"{v[0]}:{v[1]}:{v[2]}:{v[3]}"] = None

        # Save final cache
        with open(_MAF_CACHE_PATH, 'w') as f:
            _json.dump(cached, f, sort_keys=True)
        print(f"  Saved {len(cached)} entries to MAF cache")

    return maf_lookup


def assign_maf_columns(df, maf_lookup, chr_col='chr', pos1_col='pos1', pos2_col='pos2',
                        ref1_col='ref1', alt1_col='alt1', ref2_col='ref2', alt2_col='alt2'):
    """Add maf1/maf2 columns to DataFrame from the MAF lookup dict."""
    keys1 = list(zip(df[chr_col], df[pos1_col], df[ref1_col], df[alt1_col]))
    keys2 = list(zip(df[chr_col], df[pos2_col], df[ref2_col], df[alt2_col]))
    df['maf1'] = [maf_lookup.get(k) for k in keys1]
    df['maf2'] = [maf_lookup.get(k) for k in keys2]
    n_found = df['maf1'].notna().sum() + df['maf2'].notna().sum()
    n_total = len(df) * 2
    print(f"  MAF assigned: {n_found} / {n_total} values populated")
    return df


# ---------------------------------------------------------------------------
# Gene symbol mapping (GTEx egenes files)
# ---------------------------------------------------------------------------
def load_gene_symbol_mapping():
    """Load gene symbol <-> Ensembl ID mappings from GTEx egenes files.

    Returns (symbol_to_ensembl, ensembl_to_symbol) dicts.
    """
    egenes_lcl = pd.read_csv(
        GTEX_EQTL_DIR / f'{LCL_TISSUE}.v8.egenes.txt.gz',
        sep='\t', usecols=['gene_id', 'gene_name'])
    egenes_wb = pd.read_csv(
        GTEX_EQTL_DIR / f'{WB_TISSUE}.v8.egenes.txt.gz',
        sep='\t', usecols=['gene_id', 'gene_name'])
    egenes_all = pd.concat([egenes_lcl, egenes_wb]).drop_duplicates('gene_id')

    symbol_to_ensembl = dict(zip(egenes_all['gene_name'], egenes_all['gene_id']))
    ensembl_to_symbol = dict(zip(egenes_all['gene_id'], egenes_all['gene_name']))
    return symbol_to_ensembl, ensembl_to_symbol


# ---------------------------------------------------------------------------
# Epistasis ID
# ---------------------------------------------------------------------------
def make_epistasis_id(row, gene_col, chrom_col, pos1_col, pos2_col,
                       ref1_col, alt1_col, ref2_col, alt2_col, strand_lookup):
    """Build gene:chrom:pos:ref:alt:strand|gene:chrom:pos:ref:alt:strand with pos1 < pos2."""
    gene = 'GENE' if gene_col is None else str(row.get(gene_col, 'GENE'))
    chrom = str(row[chrom_col]).strip('chr').strip('chrm')
    strand = strand_lookup.get(gene, 'P')

    p1 = row[pos1_col]
    p2 = row[pos2_col]
    r1, a1 = row[ref1_col], row[alt1_col]
    r2, a2 = row[ref2_col], row[alt2_col]

    if pd.isna(p1) or pd.isna(p2) or pd.isna(r1) or pd.isna(a1) or pd.isna(r2) or pd.isna(a2):
        return pd.NA

    p1, p2 = int(p1), int(p2)

    if p1 <= p2:
        return f"{gene}:{chrom}:{p1}:{r1}:{a1}:{strand}|{gene}:{chrom}:{p2}:{r2}:{a2}:{strand}"
    else:
        return f"{gene}:{chrom}:{p2}:{r2}:{a2}:{strand}|{gene}:{chrom}:{p1}:{r1}:{a1}:{strand}"


def enforce_pos1_lt_pos2(df, pos1_col='pos1', pos2_col='pos2'):
    """Swap all *1/*2 column pairs in rows where pos1 > pos2.

    Auto-detects paired columns by suffix (ref1/ref2, alt1/alt2, snp_id1/snp_id2, etc.).
    Returns modified copy. Idempotent — already-ordered rows pass through unchanged.
    """
    df = df.copy()
    swap_mask = df[pos1_col] > df[pos2_col]
    n_swap = swap_mask.sum()

    if n_swap == 0:
        print(f"  enforce_pos1_lt_pos2: 0 rows needed swapping (all ordered)")
        return df

    # Auto-detect all *1/*2 column pairs (handles pos1_hg38/pos2_hg38 etc.)
    pairs = []
    seen = set()
    for c in df.columns:
        if '1' in c and c not in seen:
            c2 = c.replace('1', '2', 1)
            if c2 in df.columns and c2 not in seen:
                pairs.append((c, c2))
                seen.add(c)
                seen.add(c2)

    # Swap all paired columns in affected rows
    for c1, c2 in pairs:
        df.loc[swap_mask, [c1, c2]] = df.loc[swap_mask, [c2, c1]].values

    print(f"  enforce_pos1_lt_pos2: swapped {n_swap} rows ({len(pairs)} column pairs: {pairs})")
    return df


# ---------------------------------------------------------------------------
# Reference allele validation
# ---------------------------------------------------------------------------
_COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}


def _complement(base):
    """Return complement of a single base, or None if not a standard base."""
    return _COMPLEMENT.get(base.upper())


def validate_ref_alleles(df, name, chr_col, pos1_col, pos2_col, ref1_col, ref2_col,
                          alt1_col='alt1', alt2_col='alt2'):
    """Validate ref alleles against hg38 FASTA.

    Resolution order for mismatches:
      1. Swap ref/alt if alt matches hg38 ref
      2. Complement ref+alt if complement(ref) matches hg38 ref (minus-strand data)

    Modifies df in place. Returns (mismatches_pos1, mismatches_pos2, swapped_pos1, swapped_pos2).
    """
    from seqmat import SeqMat

    mismatches_1, mismatches_2 = [], []
    swapped_1, swapped_2 = 0, 0
    complemented_1, complemented_2 = 0, 0
    checked = 0

    for idx, row in df.iterrows():
        chrom = str(row[chr_col])
        chr_str = chrom if chrom.startswith('chr') else f'chr{chrom}'

        for pos_col, ref_col, alt_col, mm_list, label in [
            (pos1_col, ref1_col, alt1_col, mismatches_1, 'ref1'),
            (pos2_col, ref2_col, alt2_col, mismatches_2, 'ref2'),
        ]:
            pos = row[pos_col]
            ref = str(row[ref_col]).upper() if pd.notna(row[ref_col]) else None
            alt = str(row[alt_col]).upper() if pd.notna(row[alt_col]) else None
            if pd.isna(pos) or ref is None:
                continue
            pos = int(pos)

            # Skip insertions (ref is '-' or empty — nothing to validate)
            if ref in ('-', ''):
                continue

            try:
                end_pos = pos + len(ref) - 1
                s = SeqMat.from_fasta('hg38', chr_str, pos, end_pos)
                actual_ref = s.seq.upper()
            except Exception as e:
                mm_list.append((idx, chr_str, pos, ref, f'ERROR: {e}'))
                continue

            if ref != actual_ref:
                if alt is not None and alt == actual_ref:
                    # Alt matches hg38 ref -> swap ref/alt
                    df.at[idx, ref_col] = alt
                    df.at[idx, alt_col] = ref
                    if label == 'ref1':
                        swapped_1 += 1
                    else:
                        swapped_2 += 1
                elif len(ref) == 1 and _complement(ref) == actual_ref:
                    # Complement matches -> data is on opposite strand
                    comp_alt = _complement(alt) if alt and len(alt) == 1 else alt
                    df.at[idx, ref_col] = _complement(ref)
                    if comp_alt:
                        df.at[idx, alt_col] = comp_alt
                    if label == 'ref1':
                        complemented_1 += 1
                    else:
                        complemented_2 += 1
                else:
                    mm_list.append((idx, chr_str, pos, ref, actual_ref))
        checked += 1

    n_mm1 = len(mismatches_1)
    n_mm2 = len(mismatches_2)
    print(f'{name}: checked {checked} pairs')
    if swapped_1 > 0 or swapped_2 > 0:
        print(f'  Swapped ref/alt (alt matched hg38 ref): {swapped_1} at pos1, {swapped_2} at pos2')
    if complemented_1 > 0 or complemented_2 > 0:
        print(f'  Complemented ref+alt (opposite strand): {complemented_1} at pos1, {complemented_2} at pos2')
    if n_mm1 == 0 and n_mm2 == 0:
        print(f'  All reference alleles match hg38')
    else:
        if n_mm1 > 0:
            print(f'  ref1 mismatches: {n_mm1}')
            for idx, c, p, ref, actual in mismatches_1[:5]:
                print(f'    row {idx}: {c}:{p} expected={ref} actual={actual}')
        if n_mm2 > 0:
            print(f'  ref2 mismatches: {n_mm2}')
            for idx, c, p, ref, actual in mismatches_2[:5]:
                print(f'    row {idx}: {c}:{p} expected={ref} actual={actual}')
    return n_mm1, n_mm2, swapped_1, swapped_2
