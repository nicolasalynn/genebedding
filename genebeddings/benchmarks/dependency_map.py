"""
Simple dependency map computation for genomic language models.

Computes dependency matrices showing how mutations at position i affect predictions at position j.
Uses global embeddings (no token-level mapping needed).
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Literal
from tqdm import tqdm


def compute_dependency_map(
    sequence: Union[str, torch.Tensor],
    model,
    method: Literal["logprobs", "epistatic", "mutual_info"] = "mutual_info",
    range_start: Optional[int] = None,
    range_end: Optional[int] = None,
    embedding_distance: Literal["cosine", "euclidean"] = "euclidean"
) -> np.ndarray:
    """
    Compute dependency map showing how mutations at position i affect position j.

    Args:
        sequence: DNA sequence string (e.g., "ACGT...")
        model: Model wrapper with .embed() and/or .predict_nucleotides() methods
        method:
            - "logprobs": Log probability changes when masking j and mutating i
            - "epistatic": Epistatic interaction between mutations at i and j
            - "mutual_info": Mutual information via dual masking embedding changes
        range_start: Start position (default: 0)
        range_end: End position (default: len(sequence))
        embedding_distance: Distance metric for embedding-based methods

    Returns:
        Dependency matrix [W, W] where W = range_end - range_start
        dep[i,j] = dependency of position j on mutations at position i
    """
    # Convert to string if needed
    if isinstance(sequence, torch.Tensor):
        vocab = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
        sequence = ''.join(vocab.get(int(x), 'N') for x in sequence.flatten())

    sequence = sequence.upper()
    L = len(sequence)

    # Set range
    if range_start is None:
        range_start = 0
    if range_end is None:
        range_end = L

    if range_start < 0 or range_end > L or range_start >= range_end:
        raise ValueError(f"Invalid range [{range_start}, {range_end}) for sequence length {L}")

    W = range_end - range_start
    print(f"Computing {method} dependency map for {W}x{W} positions...")

    if method == "logprobs":
        dep = _compute_logprobs_dependency(sequence, model, range_start, range_end)
    elif method == "epistatic":
        dep = _compute_epistatic_dependency(sequence, model, range_start, range_end, embedding_distance)
    elif method == "mutual_info":
        dep = _compute_mutual_info_dependency(sequence, model, range_start, range_end, embedding_distance)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Zero diagonal (self-dependency)
    np.fill_diagonal(dep, 0)

    return dep


def _compute_logprobs_dependency(sequence: str, model, range_start: int, range_end: int) -> np.ndarray:
    """
    Method 1: Log probability changes
    For each i,j: mutate i, mask j, compare log probabilities at j
    """
    W = range_end - range_start
    dep = np.zeros((W, W))

    for i in tqdm(range(W), desc="Logprobs dependency"):
        pos_i = range_start + i
        ref_base_i = sequence[pos_i]

        for j in range(W):
            if i == j:
                continue

            pos_j = range_start + j
            max_delta = 0.0

            # Try each alternative at position i
            for alt_base_i in ['A', 'C', 'G', 'T']:
                if alt_base_i == ref_base_i:
                    continue

                # Reference: mask j only
                ref_seq = list(sequence)
                ref_seq[pos_j] = 'N'
                ref_seq_str = ''.join(ref_seq)

                # Mutated: mutate i and mask j
                mut_seq = list(sequence)
                mut_seq[pos_i] = alt_base_i
                mut_seq[pos_j] = 'N'
                mut_seq_str = ''.join(mut_seq)

                try:
                    # Get probabilities at the masked position j
                    ref_probs = model.predict_nucleotides(ref_seq_str, positions=None, return_dict=False)
                    mut_probs = model.predict_nucleotides(mut_seq_str, positions=None, return_dict=False)

                    if len(ref_probs) == 0 or len(mut_probs) == 0:
                        continue

                    ref_p = ref_probs[0][:4]  # [A, C, G, T]
                    mut_p = mut_probs[0][:4]  # [A, C, G, T]

                    # Compute max |Δ log-odds| across all bases
                    for b_idx in range(4):
                        ref_logit = _safe_logit(ref_p[b_idx])
                        mut_logit = _safe_logit(mut_p[b_idx])
                        delta = abs(mut_logit - ref_logit)
                        max_delta = max(max_delta, delta)

                except Exception:
                    continue

            dep[i, j] = max_delta

    return dep


def _compute_epistatic_dependency(sequence: str, model, range_start: int, range_end: int, distance_metric: str) -> np.ndarray:
    """
    Method 2: Epistatic interaction
    For each i,j: compute max nonlinear interaction between all mutations at i and j
    """
    W = range_end - range_start
    dep = np.zeros((W, W))

    # Get reference embedding
    ref_emb = model.embed(sequence, pool='mean', return_numpy=False)
    if not isinstance(ref_emb, torch.Tensor):
        ref_emb = torch.tensor(ref_emb)

    for i in tqdm(range(W), desc="Epistatic dependency"):
        pos_i = range_start + i
        ref_base_i = sequence[pos_i]

        for j in range(W):
            if i == j:
                continue

            pos_j = range_start + j
            ref_base_j = sequence[pos_j]
            max_epistasis = 0.0

            # Try all pairs of mutations at i and j
            for alt_i in ['A', 'C', 'G', 'T']:
                if alt_i == ref_base_i:
                    continue
                for alt_j in ['A', 'C', 'G', 'T']:
                    if alt_j == ref_base_j:
                        continue

                    try:
                        # Single mutation at i
                        mut_i_seq = list(sequence)
                        mut_i_seq[pos_i] = alt_i
                        emb_i = model.embed(''.join(mut_i_seq), pool='mean', return_numpy=False)
                        if not isinstance(emb_i, torch.Tensor):
                            emb_i = torch.tensor(emb_i)

                        # Single mutation at j
                        mut_j_seq = list(sequence)
                        mut_j_seq[pos_j] = alt_j
                        emb_j = model.embed(''.join(mut_j_seq), pool='mean', return_numpy=False)
                        if not isinstance(emb_j, torch.Tensor):
                            emb_j = torch.tensor(emb_j)

                        # Double mutation at i and j
                        mut_ij_seq = list(sequence)
                        mut_ij_seq[pos_i] = alt_i
                        mut_ij_seq[pos_j] = alt_j
                        emb_ij = model.embed(''.join(mut_ij_seq), pool='mean', return_numpy=False)
                        if not isinstance(emb_ij, torch.Tensor):
                            emb_ij = torch.tensor(emb_ij)

                        # Compute distances
                        if distance_metric == "euclidean":
                            d_ref_i = torch.norm(ref_emb - emb_i, p=2).item()
                            d_ref_j = torch.norm(ref_emb - emb_j, p=2).item()
                            d_ref_ij = torch.norm(ref_emb - emb_ij, p=2).item()
                        elif distance_metric == "cosine":
                            d_ref_i = 1.0 - F.cosine_similarity(ref_emb.unsqueeze(0), emb_i.unsqueeze(0)).item()
                            d_ref_j = 1.0 - F.cosine_similarity(ref_emb.unsqueeze(0), emb_j.unsqueeze(0)).item()
                            d_ref_ij = 1.0 - F.cosine_similarity(ref_emb.unsqueeze(0), emb_ij.unsqueeze(0)).item()

                        # Epistatic interaction: nonlinear residual
                        epistasis = abs(d_ref_ij - d_ref_i - d_ref_j)
                        max_epistasis = max(max_epistasis, epistasis)

                    except Exception:
                        continue

            dep[i, j] = max_epistasis

    return dep


def _compute_mutual_info_dependency(sequence: str, model, range_start: int, range_end: int, distance_metric: str) -> np.ndarray:
    """
    Method 3: Mutual information via dual masking
    For each i,j: mask both positions, measure embedding change vs single masking
    """
    W = range_end - range_start
    dep = np.zeros((W, W))

    # Get reference embedding (no masking)
    ref_emb = model.embed(sequence, pool='mean', return_numpy=False)
    if not isinstance(ref_emb, torch.Tensor):
        ref_emb = torch.tensor(ref_emb)

    # Cache single-masked embeddings
    single_mask_cache = {}

    for i in tqdm(range(W), desc="Mutual info dependency"):
        pos_i = range_start + i

        for j in range(W):
            if i == j:
                continue

            pos_j = range_start + j

            try:
                # Get single-masked embedding for position i (cache for efficiency)
                if pos_i not in single_mask_cache:
                    mask_i_seq = list(sequence)
                    mask_i_seq[pos_i] = 'N'
                    emb_i = model.embed(''.join(mask_i_seq), pool='mean', return_numpy=False)
                    if not isinstance(emb_i, torch.Tensor):
                        emb_i = torch.tensor(emb_i)
                    single_mask_cache[pos_i] = emb_i
                else:
                    emb_i = single_mask_cache[pos_i]

                # Get single-masked embedding for position j (cache for efficiency)
                if pos_j not in single_mask_cache:
                    mask_j_seq = list(sequence)
                    mask_j_seq[pos_j] = 'N'
                    emb_j = model.embed(''.join(mask_j_seq), pool='mean', return_numpy=False)
                    if not isinstance(emb_j, torch.Tensor):
                        emb_j = torch.tensor(emb_j)
                    single_mask_cache[pos_j] = emb_j
                else:
                    emb_j = single_mask_cache[pos_j]

                # Get dual-masked embedding (mask both i and j)
                mask_ij_seq = list(sequence)
                mask_ij_seq[pos_i] = 'N'
                mask_ij_seq[pos_j] = 'N'
                emb_ij = model.embed(''.join(mask_ij_seq), pool='mean', return_numpy=False)
                if not isinstance(emb_ij, torch.Tensor):
                    emb_ij = torch.tensor(emb_ij)

                # Compute distances
                if distance_metric == "euclidean":
                    d_ref_i = torch.norm(ref_emb - emb_i, p=2).item()
                    d_ref_j = torch.norm(ref_emb - emb_j, p=2).item()
                    d_ref_ij = torch.norm(ref_emb - emb_ij, p=2).item()
                elif distance_metric == "cosine":
                    d_ref_i = 1.0 - F.cosine_similarity(ref_emb.unsqueeze(0), emb_i.unsqueeze(0)).item()
                    d_ref_j = 1.0 - F.cosine_similarity(ref_emb.unsqueeze(0), emb_j.unsqueeze(0)).item()
                    d_ref_ij = 1.0 - F.cosine_similarity(ref_emb.unsqueeze(0), emb_ij.unsqueeze(0)).item()

                # Mutual information: interaction effect of dual masking
                # Positive = mutual information, Negative = redundancy
                mutual_info = d_ref_ij - d_ref_i - d_ref_j
                dep[i, j] = mutual_info

            except Exception:
                continue

    return dep


def _safe_logit(p: float, eps: float = 1e-6) -> float:
    """Compute log-odds with numerical stability."""
    p = max(min(p, 1.0 - eps), eps)
    return np.log(p) - np.log(1.0 - p)


def plot_dependency_map(
    dep: np.ndarray,
    sequence: Optional[str] = None,
    title: str = "Dependency Map",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot dependency map heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(dep, cmap='viridis', origin='upper', aspect='auto')

    W = dep.shape[0]

    # Add sequence labels if provided
    if sequence is not None and len(sequence) == W:
        if W <= 50:
            ax.set_xticks(range(W))
            ax.set_xticklabels(list(sequence), fontsize=8, rotation=90)
            ax.set_yticks(range(W))
            ax.set_yticklabels(list(sequence), fontsize=8)
        else:
            # Sample positions for large sequences
            step = max(1, W // 20)
            ticks = list(range(0, W, step))
            ax.set_xticks(ticks)
            ax.set_xticklabels([sequence[i] if i < len(sequence) else '' for i in ticks], fontsize=8, rotation=90)
            ax.set_yticks(ticks)
            ax.set_yticklabels([sequence[i] if i < len(sequence) else '' for i in ticks], fontsize=8)

    ax.set_xlabel("Target position (j)")
    ax.set_ylabel("Mutated position (i)")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Dependency")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


if __name__ == "__main__":
    print("Clean dependency map module loaded.")
    print("\nExample usage:")
    print("""
    # Analyze specific region of sequence (e.g., positions 2975-3025)
    dep = compute_dependency_map(
        sequence,                    # Full DNA sequence
        model,                       # Your model wrapper
        method="mutual_info",        # or "logprobs", "epistatic"
        range_start=2975,           # Start position in sequence
        range_end=3025,             # End position in sequence
        embedding_distance="euclidean"
    )

    # This creates a 50x50 dependency matrix for positions 2975-3024
    # dep[i,j] = dependency of position (2975+j) on mutations at position (2975+i)

    # Method examples:
    dep1 = compute_dependency_map(sequence, model, method="logprobs", range_start=100, range_end=150)
    dep2 = compute_dependency_map(sequence, model, method="epistatic", range_start=100, range_end=150)
    dep3 = compute_dependency_map(sequence, model, method="mutual_info", range_start=100, range_end=150)
    """)