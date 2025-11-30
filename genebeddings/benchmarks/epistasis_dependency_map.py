"""
Epistasis-based dependency map computation.

Computes dependency matrices where each cell (i, j) represents the maximum
epistatic residual across all possible double mutations at positions i and j.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Literal, Dict
from tqdm import tqdm


def classify_epistasis_from_embeddings(h_ref, h_m1, h_m2, h_m12, eps=1e-8):
    """
    Classify epistasis from four pooled embeddings:
        WT, M1, M2, M12

    Inputs can be numpy arrays, lists, or torch tensors of shape (D,) or (L,D).
    If (L,D), they will be mean-pooled automatically.
    """
    def pool(h):
        h = torch.as_tensor(h).float()
        if h.ndim == 2:
            return h.mean(dim=0)
        return h

    # ---- Pool embeddings ----
    WT  = pool(h_ref)
    M1  = pool(h_m1)
    M2  = pool(h_m2)
    M12 = pool(h_m12)

    # ---- Effect vectors ----
    v1  = M1  - WT          # effect of mutation 1
    v2  = M2  - WT          # effect of mutation 2
    v12 = M12 - WT          # effect of double mutant

    # Additive predicted double-mutant
    M12_exp = M1 + M2 - WT
    v12_exp = M12_exp - WT

    # ---- Norms ----
    a1      = v1.norm().item()
    a2      = v2.norm().item()
    a12     = v12.norm().item()
    a12_exp = v12_exp.norm().item()

    max_single = max(a1, a2)
    min_single = min(a1, a2)

    # ---- Angle helper ----
    def cos_angle(u, v):
        return float((u @ v) / (u.norm() * v.norm() + eps))

    c1 = cos_angle(v12, v1)
    c2 = cos_angle(v12, v2)
    same_direction = (c1 > 0.5) and (c2 > 0.5)

    # ---- Classify relative to WT ----
    if a12 < min_single:
        type_WT = "corrective (closer to WT than either single)"
    elif a12 > max_single and same_direction:
        type_WT = "cumulative (same-direction aggravating)"
    elif a12 > max_single and not same_direction:
        type_WT = "divergent (further but rotated)"
    else:
        type_WT = "intermediate/ambiguous"

    # ---- Classify relative to additive expectation ----
    if a12 < a12_exp:
        type_add = "sub-additive (dampened)"
    elif a12 > a12_exp:
        type_add = "super-additive (synergistic)"
    else:
        type_add = "approximately additive"

    # ---- Epistatic residual ----
    residual_vec  = M12 - M12_exp
    residual_norm = residual_vec.norm().item()

    # ======================================================
    # 1) Normalized epistasis score vs expected magnitude
    #    epi_rel_expected = ||M12_obs - M12_exp|| / ||M12_exp - WT||
    # ======================================================
    denom_exp = a12_exp if a12_exp > eps else eps
    epi_rel_expected = residual_norm / denom_exp

    # ======================================================
    # 2) Normalized epistasis vs combined single effects
    #    epi_rel_singles = ||residual|| / sqrt(||v1||^2 + ||v2||^2)
    #    (isolate deviation from what you'd expect given strong/weak singles)
    # ======================================================
    single_scale = (a1**2 + a2**2) ** 0.5
    if single_scale < eps:
        single_scale = eps
    epi_rel_singles = residual_norm / single_scale

    return {
        # Raw distances
        "dist_WT_M1": a1,
        "dist_WT_M2": a2,
        "dist_WT_M12_obs": a12,
        "dist_WT_M12_exp": a12_exp,

        # Directionality
        "cos(v12, v1)": c1,
        "cos(v12, v2)": c2,
        "same_direction": same_direction,

        # Residual (absolute epistasis)
        "residual_norm": residual_norm,

        # Normalized epistasis scores
        # 1) relative to expected double-mutant magnitude
        "epi_rel_expected": epi_rel_expected,
        # 2) relative to combined single-mutation magnitude
        "epi_rel_singles": epi_rel_singles,

        # Qualitative epistasis labels
        "type_relative_WT": type_WT,
        "type_relative_additivity": type_add,

        # Embeddings
        "embed_WT": WT.detach().cpu().numpy(),
        "embed_M1": M1.detach().cpu().numpy(),
        "embed_M2": M2.detach().cpu().numpy(),
        "embed_M12": M12.detach().cpu().numpy(),
    }


BASES = ['A', 'C', 'G', 'T']


def compute_epistasis_dependency_map(
    sequence: str,
    wrapper,
    range_start: Optional[int] = None,
    range_end: Optional[int] = None,
    metric: Literal["residual_norm", "epi_rel_singles", "epi_rel_expected"] = "residual_norm",
    pool: str = "mean",
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute epistasis dependency map where each cell (i, j) is the maximum
    epistatic residual across all possible mutation pairs at positions i and j.

    Args:
        sequence: DNA sequence string (e.g., "ACGT...")
        wrapper: Model wrapper with .embed() method (e.g., NTWrapper)
        range_start: Start position in sequence (default: 0)
        range_end: End position in sequence (default: len(sequence))
        metric: Which epistasis metric to use:
            - "residual_norm": Raw epistatic residual ||M12 - M12_expected||
            - "epi_rel_singles": Normalized by combined single mutation magnitude
            - "epi_rel_expected": Normalized by expected double-mutant magnitude
        pool: Pooling method for embeddings ('mean', 'cls', 'tokens')
        show_progress: Whether to show progress bar

    Returns:
        Dependency matrix [W, W] where W = range_end - range_start
        dep[i, j] = max epistatic residual across all mutation pairs at (i, j)
    """
    sequence = sequence.upper()
    L = len(sequence)

    if range_start is None:
        range_start = 0
    if range_end is None:
        range_end = L

    if range_start < 0 or range_end > L or range_start >= range_end:
        raise ValueError(f"Invalid range [{range_start}, {range_end}) for sequence length {L}")

    W = range_end - range_start
    dep = np.zeros((W, W))

    print(f"Computing epistasis dependency map ({metric}) for {W}x{W} positions...")
    print(f"Total position pairs to evaluate: {W * (W - 1) // 2}")

    # Get reference (WT) embedding once
    ref_emb = wrapper.embed(sequence, pool=pool, return_numpy=False)
    if isinstance(ref_emb, np.ndarray):
        ref_emb = torch.from_numpy(ref_emb)

    # Cache single-mutation embeddings for efficiency
    single_mut_cache: Dict[tuple, torch.Tensor] = {}

    def get_single_mut_embedding(pos: int, alt: str) -> torch.Tensor:
        key = (pos, alt)
        if key not in single_mut_cache:
            mut_seq = list(sequence)
            mut_seq[pos] = alt
            emb = wrapper.embed(''.join(mut_seq), pool=pool, return_numpy=False)
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)
            single_mut_cache[key] = emb
        return single_mut_cache[key]

    # Iterate over all position pairs
    iterator = range(W)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Epistasis ({metric})")

    for i in iterator:
        pos_i = range_start + i
        ref_base_i = sequence[pos_i]

        for j in range(i + 1, W):  # Only upper triangle, then mirror
            pos_j = range_start + j
            ref_base_j = sequence[pos_j]

            max_epistasis = 0.0

            # Try all pairs of mutations at positions i and j
            for alt_i in BASES:
                if alt_i == ref_base_i:
                    continue

                # Get single mutation embedding at i (cached)
                emb_i = get_single_mut_embedding(pos_i, alt_i)

                for alt_j in BASES:
                    if alt_j == ref_base_j:
                        continue

                    try:
                        # Get single mutation embedding at j (cached)
                        emb_j = get_single_mut_embedding(pos_j, alt_j)

                        # Get double mutation embedding
                        mut_ij_seq = list(sequence)
                        mut_ij_seq[pos_i] = alt_i
                        mut_ij_seq[pos_j] = alt_j
                        emb_ij = wrapper.embed(''.join(mut_ij_seq), pool=pool, return_numpy=False)
                        if isinstance(emb_ij, np.ndarray):
                            emb_ij = torch.from_numpy(emb_ij)

                        # Compute epistasis metrics
                        result = classify_epistasis_from_embeddings(
                            h_ref=ref_emb,
                            h_m1=emb_i,
                            h_m2=emb_j,
                            h_m12=emb_ij
                        )

                        epistasis_value = result[metric]
                        max_epistasis = max(max_epistasis, epistasis_value)

                    except Exception as e:
                        continue

            # Fill both (i,j) and (j,i) - symmetric matrix
            dep[i, j] = max_epistasis
            dep[j, i] = max_epistasis

    return dep


def plot_epistasis_dependency_map(
    dep: np.ndarray,
    sequence: Optional[str] = None,
    range_start: int = 0,
    title: str = "Epistasis Dependency Map",
    metric: str = "residual_norm",
    save_path: Optional[str] = None,
    cmap: str = "viridis",
) -> plt.Figure:
    """Plot epistasis dependency map heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(dep, cmap=cmap, origin='upper', aspect='auto')

    W = dep.shape[0]

    # Add sequence labels if provided
    if sequence is not None:
        subseq = sequence[range_start:range_start + W]
        if W <= 50:
            ax.set_xticks(range(W))
            ax.set_xticklabels(list(subseq), fontsize=8, rotation=90)
            ax.set_yticks(range(W))
            ax.set_yticklabels(list(subseq), fontsize=8)
        else:
            step = max(1, W // 20)
            ticks = list(range(0, W, step))
            ax.set_xticks(ticks)
            ax.set_xticklabels([subseq[i] if i < len(subseq) else '' for i in ticks], fontsize=8, rotation=90)
            ax.set_yticks(ticks)
            ax.set_yticklabels([subseq[i] if i < len(subseq) else '' for i in ticks], fontsize=8)

    ax.set_xlabel("Position j")
    ax.set_ylabel("Position i")
    ax.set_title(f"{title}\n(metric: {metric})")

    plt.colorbar(im, ax=ax, label=f"Max {metric}")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


if __name__ == "__main__":
    print("Epistasis dependency map module loaded.")
    print("\nExample usage:")
    print("""
    from epistasis_dependency_map import compute_epistasis_dependency_map, plot_epistasis_dependency_map

    # Using NTWrapper
    from genebeddings.wrappers.nt_wrapper import NTWrapper
    wrapper = NTWrapper(model="v2-500m-multi")

    # Compute dependency map for a region
    sequence = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"

    # Using residual_norm (raw epistatic residual)
    dep = compute_epistasis_dependency_map(
        sequence,
        wrapper,
        range_start=0,
        range_end=20,
        metric="residual_norm"
    )

    # Using epi_rel_singles (normalized by single mutation magnitudes)
    dep = compute_epistasis_dependency_map(
        sequence,
        wrapper,
        range_start=0,
        range_end=20,
        metric="epi_rel_singles"
    )

    # Using epi_rel_expected (normalized by expected double-mutant magnitude)
    dep = compute_epistasis_dependency_map(
        sequence,
        wrapper,
        range_start=0,
        range_end=20,
        metric="epi_rel_expected"
    )

    # Plot the result
    plot_epistasis_dependency_map(
        dep,
        sequence=sequence,
        range_start=0,
        metric="residual_norm",
        save_path="epistasis_map.png"
    )
    """)
