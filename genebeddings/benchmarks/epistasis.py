"""
Epistasis analysis for double variants.

Computes embedding-based features comparing WT, M1, M2, and M12 contexts.
Returns quantitative metrics without saving raw embeddings.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from seqmat import SeqMat
import pandas as pd
from pathlib import Path


def load_gene_strands(csv_path: Optional[str] = None) -> Dict[str, bool]:
    """
    Load gene strand mapping from CSV file.

    Args:
        csv_path: Path to gene_strands.csv. If None, uses default location.

    Returns:
        Dictionary mapping gene_name -> strand (True for +, False for -)
    """
    if csv_path is None:
        # Default location relative to this file
        csv_path = Path(__file__).parent.parent / "assets" / "benchmarks" / "gene_strands.csv"

    try:
        df = pd.read_csv(csv_path)
        return dict(zip(df['gene_name'], df['Strand']))
    except Exception as e:
        print(f"Warning: Could not load gene_strands.csv from {csv_path}: {e}")
        return {}


def parse_epistasis_id(epi_id: str) -> Dict[str, object]:
    """
    Parse ID: gene:chrom:pos1:ref1:alt1|gene:chrom:pos2:ref2:alt2

    Returns dict with parsed mutation info.
    """
    mut_strs = epi_id.split("|")
    genes = []
    chroms = []
    pos_list = []
    ref_list = []
    alt_list = []

    for m in mut_strs:
        gene, chrm, pos, ref, alt = m.split(":")
        genes.append(gene)
        chroms.append(chrm)
        pos_list.append(int(pos))
        ref_list.append(ref)
        alt_list.append(alt)

    return {
        "epistasis_id": epi_id,
        "gene": genes[0],
        "chrom": chroms[0],
        "positions": pos_list,
        "refs": ref_list,
        "alts": alt_list,
        "n_mut": len(pos_list),
    }


def build_context_seqs(
    epi_id: str,
    flank: int = 3000,
    rc: bool = False,
    genome: str = "hg38",
) -> Tuple[List[str], Dict[str, object]]:
    """
    Build sequences for [WT, M1, M2, M12] contexts.

    Returns:
        seqs: list of 4 sequences
        info: metadata dict
    """
    info = parse_epistasis_id(epi_id)
    positions = info["positions"]
    refs = info["refs"]
    alts = info["alts"]

    if len(positions) != 2:
        raise ValueError(f"Expected 2 mutations for double variant, got {len(positions)}")

    # Sanity checks
    for r, a in zip(refs, alts):
        if r == a:
            raise ValueError(f"No-op mutation (ref==alt) in {epi_id}: {r}->{a}")
        if "-" in (r + a):
            raise ValueError(f"Indel not supported: {r}->{a}")

    chrom = f"chr{info['chrom']}"
    start = min(positions) - flank
    end = max(positions) + flank

    # Get base sequence
    base = SeqMat.from_fasta(genome, chrom, start, end)
    if rc:
        base.reverse_complement()

    mut_list = list(zip(positions, refs, alts))

    # Build 4 contexts: WT, M1, M2, M12
    seqs = [base.seq]  # WT

    # M1 (first mutation only)
    m1 = base.clone()
    m1.apply_mutations([mut_list[0]], permissive_ref=True)
    seqs.append(m1.seq)

    # M2 (second mutation only)
    m2 = base.clone()
    m2.apply_mutations([mut_list[1]], permissive_ref=True)
    seqs.append(m2.seq)

    # M12 (both mutations)
    m12 = base.clone()
    m12.apply_mutations(mut_list, permissive_ref=True)
    seqs.append(m12.seq)

    info.update({
        "chrom_full": chrom,
        "start": start,
        "end": end,
        "rc": rc,
        "pos1": positions[0],
        "pos2": positions[1],
        "ref1": refs[0],
        "ref2": refs[1],
        "alt1": alts[0],
        "alt2": alts[1],
    })

    return seqs, info


def get_embeddings(
    model,
    seqs: List[str],
    pool_type: str = "tokens",
) -> np.ndarray:
    """
    Get embeddings for sequences.

    Returns:
        E: (4, C, L) array for 4 contexts
    """
    emb_list = []

    for s in seqs:
        arr = model.embed(s, pool=pool_type)

        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()

        # Transpose (L, H) -> (H, L) for token embeddings
        if pool_type == "tokens" and arr.ndim == 2:
            arr = arr.T

        arr = arr.astype(np.float32, copy=False)
        emb_list.append(arr)

    # Crop to same length
    min_L = min(a.shape[-1] for a in emb_list)
    emb_list = [a[..., :min_L] for a in emb_list]

    return np.stack(emb_list, axis=0)


def compute_pairwise_cosine(z: torch.Tensor) -> Dict[str, float]:
    """
    Compute all pairwise cosine similarities.

    z: (4, C) tensor for [WT, M1, M2, M12]

    Returns dict with:
        cos_wt_m1, cos_wt_m2, cos_wt_m12
        cos_m1_m2, cos_m1_m12, cos_m2_m12
    """
    z_wt, z_m1, z_m2, z_m12 = z[0], z[1], z[2], z[3]

    def _cos(a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()

    return {
        "cos_wt_m1": _cos(z_wt, z_m1),
        "cos_wt_m2": _cos(z_wt, z_m2),
        "cos_wt_m12": _cos(z_wt, z_m12),
        "cos_m1_m2": _cos(z_m1, z_m2),
        "cos_m1_m12": _cos(z_m1, z_m12),
        "cos_m2_m12": _cos(z_m2, z_m12),
    }


def compute_euclidean_distances(z: torch.Tensor) -> Dict[str, float]:
    """
    Compute all pairwise Euclidean distances.

    z: (4, C) tensor for [WT, M1, M2, M12]
    """
    z_wt, z_m1, z_m2, z_m12 = z[0], z[1], z[2], z[3]

    def _dist(a, b):
        return torch.norm(a - b).item()

    return {
        "dist_wt_m1": _dist(z_wt, z_m1),
        "dist_wt_m2": _dist(z_wt, z_m2),
        "dist_wt_m12": _dist(z_wt, z_m12),
        "dist_m1_m2": _dist(z_m1, z_m2),
        "dist_m1_m12": _dist(z_m1, z_m12),
        "dist_m2_m12": _dist(z_m2, z_m12),
    }


def compute_epistasis_metrics_old(z: torch.Tensor) -> Dict[str, float]:
    """
    Compute epistasis-specific metrics.

    z: (4, C) tensor for [WT, M1, M2, M12]

    Returns:
        eps_mag: magnitude of epistasis vector
        eps_rel: relative epistasis (eps_mag / ||d_add||)
        cos_d12_dadd: cosine between actual and additive directions
        eps_par: signed epistasis along additive direction
        eps_perp_mag: orthogonal epistasis magnitude
        wt_closeness: ||d_add|| - ||d_12|| (positive = closer to WT)
    """
    z_wt, z_m1, z_m2, z_m12 = z[0], z[1], z[2], z[3]

    # Displacement vectors from WT
    d1 = z_m1 - z_wt
    d2 = z_m2 - z_wt
    d12 = z_m12 - z_wt
    d_add = d1 + d2  # Additive expectation

    norm_dadd = torch.norm(d_add) + 1e-8
    norm_d12 = torch.norm(d12) + 1e-8

    # Epistasis vector
    eps_vec = d12 - d_add
    eps_mag = torch.norm(eps_vec).item()
    eps_rel = eps_mag / norm_dadd.item()

    # Cosine between actual and additive
    cos_d12_dadd = F.cosine_similarity(
        d12.unsqueeze(0), d_add.unsqueeze(0), dim=-1
    ).item()

    # Signed epistasis along additive direction
    eps_par = (torch.dot(eps_vec, d_add) / norm_dadd).item()

    # Orthogonal epistasis
    dadd_sq = torch.dot(d_add, d_add) + 1e-8
    eps_vec_par = (torch.dot(eps_vec, d_add) / dadd_sq) * d_add
    eps_perp = eps_vec - eps_vec_par
    eps_perp_mag = torch.norm(eps_perp).item()

    # Distance comparison to WT
    wt_closeness = (norm_dadd - norm_d12).item()

    return {
        "eps_mag": eps_mag,
        "eps_rel": eps_rel,
        "cos_d12_dadd": cos_d12_dadd,
        "eps_par": eps_par,
        "eps_perp_mag": eps_perp_mag,
        "wt_closeness": wt_closeness,
        "norm_d1": torch.norm(d1).item(),
        "norm_d2": torch.norm(d2).item(),
        "norm_d12": norm_d12.item(),
        "norm_dadd": norm_dadd.item(),
    }


def compute_epistasis_metrics(z: torch.Tensor) -> Dict[str, float]:
    """
    Compute epistasis-specific metrics in the same geometric language
    we've been using for the 4-vector epistasis picture.

    z: (4, C) tensor for [WT, M1, M2, M12]

    Returns:
        residual_norm       : ||v12_obs - v12_exp||
        epi_rel_expected    : residual_norm / ||v12_exp||
        epi_rel_singles     : residual_norm / sqrt(||v1||^2 + ||v2||^2)
        cos_d12_dexp        : cos(angle between v12_obs and v12_exp)
        cos_d12_d1          : cos(angle between v12_obs and v1)
        cos_d12_d2          : cos(angle between v12_obs and v2)
        norm_d1, norm_d2,
        norm_d12_obs, norm_d12_exp
        wt_closeness        : ||v12_exp|| - ||v12_obs|| (positive = closer to WT than expected)
    """
    z_wt, z_m1, z_m2, z_m12 = z[0], z[1], z[2], z[3]

    # --- effect vectors from WT (this is the "epistasis space") ---
    v1       = z_m1  - z_wt          # single 1
    v2       = z_m2  - z_wt          # single 2
    v12_obs  = z_m12 - z_wt          # observed double
    v12_exp  = v1 + v2               # additive expectation (M1 + M2 - WT - WT)

    # --- norms ---
    norm_v1      = torch.norm(v1)      + 1e-8
    norm_v2      = torch.norm(v2)      + 1e-8
    norm_v12_obs = torch.norm(v12_obs) + 1e-8
    norm_v12_exp = torch.norm(v12_exp) + 1e-8

    # --- epistasis residual vector (what we care about) ---
    residual_vec  = v12_obs - v12_exp
    residual_norm = torch.norm(residual_vec).item()

    # 1) normalized by expected double-mutant magnitude  ||v12_exp||
    epi_rel_expected = residual_norm / norm_v12_exp.item()

    # 2) normalized by combined single-mutation magnitude sqrt(||v1||^2 + ||v2||^2)
    single_scale = (norm_v1.item()**2 + norm_v2.item()**2) ** 0.5
    if single_scale < 1e-8:
        single_scale = 1e-8
    epi_rel_singles = residual_norm / single_scale

    # --- directionality of the observed double relative to expectations ---
    cos_d12_dexp = torch.nn.functional.cosine_similarity(
        v12_obs.unsqueeze(0), v12_exp.unsqueeze(0), dim=-1
    ).item()

    # angles to each single-effect vector
    cos_d12_d1 = torch.nn.functional.cosine_similarity(
        v12_obs.unsqueeze(0), v1.unsqueeze(0), dim=-1
    ).item()
    cos_d12_d2 = torch.nn.functional.cosine_similarity(
        v12_obs.unsqueeze(0), v2.unsqueeze(0), dim=-1
    ).item()

    # --- "closeness" to WT: is the observed double closer or further than expected? ---
    wt_closeness = (norm_v12_exp - norm_v12_obs).item()

    return {
        # absolute residual in effect space
        "residual_norm":      residual_norm,

        # normalized epistasis scores
        "epi_rel_expected":   epi_rel_expected,
        "epi_rel_singles":    epi_rel_singles,

        # geometry of the double mutant
        "cos_d12_dexp":       cos_d12_dexp,
        "cos_d12_d1":         cos_d12_d1,
        "cos_d12_d2":         cos_d12_d2,

        # magnitudes of effect vectors
        "norm_d1":            norm_v1.item(),
        "norm_d2":            norm_v2.item(),
        "norm_d12_obs":       norm_v12_obs.item(),
        "norm_d12_exp":       norm_v12_exp.item(),

        # is the double closer/further from WT than the additive prediction?
        "wt_closeness":       wt_closeness,
    }


def compute_all_features(E: np.ndarray) -> Dict[str, float]:
    """
    Compute all features from embeddings.

    E: (4, C, L) array for [WT, M1, M2, M12]

    Returns dict with:
        - Pairwise cosine similarities
        - Pairwise Euclidean distances
        - Epistasis metrics
    """
    # Pool over sequence length
    z_np = E.mean(axis=-1)  # (4, C)
    z = torch.from_numpy(z_np.astype(np.float32))

    features = {}
    features.update(compute_pairwise_cosine(z))
    features.update(compute_euclidean_distances(z))
    features.update(compute_epistasis_metrics(z))

    return features


def analyze_double_variant(
    model,
    epi_id: str,
    flank: int = 3000,
    rc: bool = False,
    genome: str = "hg38",
    pool_type: str = "tokens",
) -> Dict[str, object]:
    """
    Analyze a single double variant.

    Returns dict with metadata and all computed features.
    """
    # Build sequences
    seqs, info = build_context_seqs(epi_id, flank=flank, rc=rc, genome=genome)

    # Get embeddings
    E = get_embeddings(model, seqs, pool_type=pool_type)

    # Compute features
    features = compute_all_features(E)

    # Combine info and features
    result = {
        "epistasis_id": info["epistasis_id"],
        "gene": info["gene"],
        "chrom": info["chrom"],
        "pos1": info["pos1"],
        "pos2": info["pos2"],
        "ref1": info["ref1"],
        "alt1": info["alt1"],
        "ref2": info["ref2"],
        "alt2": info["alt2"],
        "rc": info["rc"],
        "start": info["start"],
        "end": info["end"],
    }
    result.update(features)

    return result


def run_epistasis_analysis(
    model,
    epi_ids: List[str],
    *,
    flank: int = 3000,
    rcs: List[bool] = None,
    genome: str = "hg38",
    pool_type: str = "tokens",
    run_both_strands: bool = True,
    gene_strands_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run epistasis analysis for a list of double variants.

    Args:
        model: Embedding model with .embed() method
        epi_ids: List of epistasis IDs
        flank: Flanking sequence length
        rcs: List of reverse complement flags (optional)
        genome: Reference genome
        pool_type: Embedding pooling type
        run_both_strands: If True and rcs not provided, run both orientations
        gene_strands_csv: Path to gene_strands.csv (optional, uses default if None)

    Returns:
        DataFrame with all features for each variant
    """
    records = []

    # Handle rcs
    if rcs is None:
        # Load gene strand mapping
        gene_strands = load_gene_strands(gene_strands_csv)

        # Determine strand for each epistasis ID
        expanded_ids = []
        expanded_rcs = []

        for eid in epi_ids:
            # Extract gene name from epistasis ID
            gene = eid.split(":")[0]

            # Check if gene has a defined strand
            if gene in gene_strands:
                # Use the strand from the CSV
                expanded_ids.append(eid)
                expanded_rcs.append(gene_strands[gene])
            elif run_both_strands:
                # Gene not in CSV: run both orientations
                expanded_ids.extend([eid, eid])
                expanded_rcs.extend([False, True])
            else:
                # Gene not in CSV and run_both_strands=False: use forward strand
                expanded_ids.append(eid)
                expanded_rcs.append(False)

        epi_ids = expanded_ids
        rcs = expanded_rcs

    for eid, rc in tqdm(list(zip(epi_ids, rcs)), desc="Analyzing variants"):
        try:
            result = analyze_double_variant(
                model, eid, flank=flank, rc=rc,
                genome=genome, pool_type=pool_type
            )
            records.append(result)
        except Exception as e:
            print(f"[{eid}] error: {e}")
            continue

    df = pd.DataFrame(records)
    # if not df.empty:
    #     df = df.sort_values("eps_mag", ascending=False).reset_index(drop=True)

    return df


def run_epistasis_for_models(
    model_factories: Dict[str, callable],
    epi_ids: List[str],
    *,
    flank: int = 3000,
    rcs: List[bool] = None,
    genome: str = "hg38",
    pool_type: str = "tokens",
    out_dir: str = "results/epistasis",
    save_combined: bool = True,
    gene_strands_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run epistasis analysis across multiple models.

    Args:
        model_factories: Dict of model_name -> factory function
        epi_ids: List of epistasis IDs
        flank: Default flanking length
        rcs: Reverse complement flags
        genome: Reference genome
        pool_type: Embedding pooling type
        out_dir: Output directory
        save_combined: Save combined results
        gene_strands_csv: Path to gene_strands.csv (optional, uses default if None)

    Returns:
        Combined DataFrame with results from all models
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_name, model in model_factories.items():
        # Adjust flank for specific models
        model_flank = flank
        if model_name == 'borzoi':
            model_flank = 524_000 // 2

        print(f"\n=== Running epistasis analysis: {model_name} ===")

        # Instantiate model
        # model = factory()

        # Run analysis
        df = run_epistasis_analysis(
            model, epi_ids,
            flank=model_flank, rcs=rcs, genome=genome, pool_type=pool_type,
            gene_strands_csv=gene_strands_csv
        )
        df["model_name"] = model_name

        # Save per-model results
        out_path = out_dir / f"epistasis_{model_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path}")

        all_results.append(df)

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_results:
        return pd.DataFrame()

    df_all = pd.concat(all_results, ignore_index=True)

    if save_combined:
        combined_path = out_dir / "epistasis_all_models.parquet"
        df_all.to_parquet(combined_path, index=False)
        print(f"\nSaved combined results to {combined_path}")

    return df_all
