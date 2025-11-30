from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from seqmat import SeqMat

Pool = Literal["tokens", "mean", "cls"]


# -------------------------------------------------------------------
# 1) Parse SINGLE variant ID
#    mut_id: "gene:chrom:pos:ref:alt"
# -------------------------------------------------------------------
def parse_mut_id(mut_id: str) -> Dict[str, object]:
    """
    Parse a single-variant ID of the form:
        gene:chrom:pos:ref:alt

    Returns a dict with:
        mut_id, gene, chrom, pos (int), ref, alt
    """
    gene, chrm, pos, ref, alt = mut_id.split(":")
    pos_i = int(pos)

    if ref == alt:
        raise ValueError(f"No-op mutation (ref==alt) in {mut_id}: {ref}->{alt}")
    if "-" in (ref + alt):
        raise ValueError(f"Indel not supported yet in {mut_id}: {ref}->{alt}")

    return {
        "mut_id": mut_id,
        "gene": gene,
        "chrom": chrm,
        "pos": pos_i,
        "ref": ref,
        "alt": alt,
    }


# -------------------------------------------------------------------
# 2) Build [WT, MUT] sequences around the site
# -------------------------------------------------------------------
def build_wt_mut_seqs(
    mut_id: str,
    flank: int = 3000,
    rc: bool | None = None,
    genome: str = "hg38",
) -> Tuple[List[str], Dict[str, object]]:
    """
    Given a mutation ID, build sequences for:
        [WT, MUT]

    Returns:
        seqs: [wt_seq, mut_seq]
        info: meta dict with window info
    """
    info = parse_mut_id(mut_id)
    chrom = f"chr{info['chrom']}"
    pos   = info["pos"]

    start = pos - flank
    end   = pos + flank

    base = SeqMat.from_fasta(genome, chrom, start, end)
    if rc:
        base.reverse_complement()

    mut = base.clone()
    mut.apply_mutations([(pos, info["ref"], info["alt"])], permissive_ref=True)

    seqs = [base.seq, mut.seq]  # [WT, MUT]

    info.update({
        "chrom_full": chrom,
        "start_full": start,
        "end_full": end,
        "rc": bool(rc),
        "context_labels": ["WT", "MUT"],
    })
    return seqs, info


# -------------------------------------------------------------------
# 3) Run model on [WT, MUT] and normalize to (2, C, L)
# -------------------------------------------------------------------
def run_encoder_2contexts(
    model,
    mut_id: str,
    flank: int = 3000,
    rc: bool | None = None,
    tracks: bool = False,
    pool_type: str = "tokens",
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Run model on [WT, MUT].

    Assumptions:
      - model.tracks(seq) -> (C, L) np.ndarray or torch.Tensor
      - model.embed(seq, pool="tokens") -> (L, H) or (C, L) as torch.Tensor/np.ndarray

    Returns:
      E:    (2, C, L) np.ndarray
      info: meta dict
    """
    seqs, info = build_wt_mut_seqs(mut_id, flank=flank, rc=rc)
    emb_list = []

    for s in seqs:
        if tracks:
            arr = model.tracks(s)
        else:
            arr = model.embed(s, pool=pool_type)

        # to numpy
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()

        # normalize shape to (C, L)
        if (not tracks) and pool_type == "tokens":
            # assume (L, H) -> (H, L)
            if arr.ndim == 2:
                arr = arr.T

        arr = arr.astype(np.float32, copy=False)
        emb_list.append(arr)

    # Make sure last dimension is same length (center / left crop to min)
    min_L = min(a.shape[-1] for a in emb_list)
    emb_list = [a[..., :min_L] for a in emb_list]

    E = np.stack(emb_list, axis=0)  # (2, C, L)
    return E, info


# -------------------------------------------------------------------
# 4) Single-variant "effect" metrics from pooled embeddings
# -------------------------------------------------------------------
def single_effect_from_pooled(z: torch.Tensor) -> Dict[str, float | torch.Tensor]:
    """
    z: (2, C) tensor
    order: [WT, MUT]

    Definitions:
        d      = z_MUT - z_WT
        d_mag  = ||d||
        d_rel  = ||d|| / (||z_WT|| + eps)
        cos_wt = cos(z_WT, z_MUT)

    Returns:
        delta_vec  : (C,) torch.Tensor
        delta_mag  : float
        delta_rel  : float
        cos_wt     : float
    """
    if z.shape[0] != 2:
        raise ValueError("Expected z with shape (2, C): [WT, MUT].")

    z_wt, z_mut = z  # each (C,)

    d = z_mut - z_wt
    d_mag = torch.norm(d).item()
    d_rel = d_mag / (torch.norm(z_wt).item() + 1e-8)

    cos_wt = F.cosine_similarity(
        z_wt.unsqueeze(0),
        z_mut.unsqueeze(0),
        dim=-1,
    ).item()

    return {
        "delta_vec": d,
        "delta_mag": d_mag,
        "delta_rel": d_rel,
        "cos_wt": cos_wt,
    }


def single_effect_from_embed(E: np.ndarray) -> Dict[str, float | torch.Tensor]:
    """
    E: (2, C, L) from run_encoder_2contexts.
    Pools over L and computes single-variant metrics.
    """
    z_np = E.mean(axis=-1)  # (2, C)
    z = torch.from_numpy(z_np.astype(np.float32))
    return single_effect_from_pooled(z)


# -------------------------------------------------------------------
# 5) Run over a list of single variants for ONE model
# -------------------------------------------------------------------
def run_single_effect_for_list(
    model,
    mut_ids: List[str],
    *,
    flank: int = 3000,
    rcs=None,
) -> pd.DataFrame:
    """
    Loop over a list of mut_ids, compute embedding deltas.

    Returns:
        df with columns:
          mut_id, gene, chrom, pos, ref, alt,
          chrom_full, start_full, end_full, rc,
          delta_mag, delta_rel, cos_wt
    """
    records = []

    if rcs is None or len(rcs) != len(mut_ids):
        rcs = [None for _ in range(len(mut_ids))]

    for mid, rc in tqdm(list(zip(mut_ids, rcs))):
        try:
            # if rc is None, do both orientations (rc=True/False)
            if rc is None:
                for rc_flag in (True, False):
                    E, info = run_encoder_2contexts(
                        model,
                        mid,
                        flank=flank,
                        rc=rc_flag,
                    )
                    metrics = single_effect_from_embed(E)
                    row = {
                        **parse_mut_id(mid),
                        **info,
                        **{k: v for k, v in metrics.items() if k != "delta_vec"},
                    }
                    records.append(row)
            else:
                E, info = run_encoder_2contexts(
                    model,
                    mid,
                    flank=flank,
                    rc=rc,
                )
                metrics = single_effect_from_embed(E)
                row = {
                    **parse_mut_id(mid),
                    **info,
                    **{k: v for k, v in metrics.items() if k != "delta_vec"},
                }
                records.append(row)

        except Exception as e:
            print(f"[{mid}] error: {e}")
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("delta_mag", ascending=False).reset_index(drop=True)
    return df


# -------------------------------------------------------------------
# 6) Run the single-variant pipeline for MULTIPLE MODELS
# -------------------------------------------------------------------
def run_single_effect_for_all_models(
    model_factories: dict,
    mut_ids: List[str],
    *,
    flank_default: int = 3000,
    rcs=None,
    out_dir: str = "results/single_effect",
    save_combined: bool = True,
) -> pd.DataFrame:
    """
    model_factories: dict[str, callable] mapping model_name -> () -> model
                     (model must implement .tracks or .embed as above)

    Returns:
        df_all: concatenated DataFrame with column 'model_name'.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_name, factory in model_factories.items():
        # optional special flank for particular models
        if model_name.lower() == "borzoi":
            flank = int(524_000 // 2)
        else:
            flank = flank_default

        print(f"\n=== Running single-variant benchmark for model: {model_name} ===")
        model = factory()

        df = run_single_effect_for_list(
            model,
            mut_ids,
            flank=flank,
            rcs=rcs,
        )
        df["model_name"] = model_name

        out_path = out_dir / f"single_effect_{model_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path}")

        all_results.append(df)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_results:
        return pd.DataFrame()

    df_all = pd.concat(all_results, ignore_index=True)
    if save_combined:
        combined_path = out_dir / "single_effect_all_models.parquet"
        df_all.to_parquet(combined_path, index=False)
        print(f"\nSaved combined results to {combined_path}")

    return df_all