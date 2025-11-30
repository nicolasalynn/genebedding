# genomenet_wrapper.py
"""
GenomeNet wrapper for the CheapGenomicMLM model.
Standardized API: embed(), predict_nucleotides()
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Support both package import and direct sys.path import
try:
    from .base_wrapper import BaseWrapper
except ImportError:
    from base_wrapper import BaseWrapper

# ====================== Model Architecture ======================
# New model architecture provided by user

import math
import random

def sequence_mean_pool(hidden: torch.Tensor,
                       pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    hidden:   (B, L, D)
    pad_mask: (B, L) with 1 = valid, 0 = pad  (or None)

    Returns:
        pooled: (B, D) mask-weighted mean over sequence.
    """
    if pad_mask is None:
        return hidden.mean(dim=1)

    mask = pad_mask.unsqueeze(-1).to(hidden.dtype)   # (B, L, 1)
    summed = (hidden * mask).sum(dim=1)              # (B, D)
    denom = mask.sum(dim=1).clamp(min=1.0)           # (B, 1)
    return summed / denom


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


class NucConvTokenizer(nn.Module):
    """
    Nucleotide-level tokenizer + conv front-end.

    Accepts:
        - single string: "ACGTN..."
        - list of strings
        - tensor of IDs (B, L)
    """

    def __init__(
        self,
        vocab_size: int = 7,
        d_embed: int = 64,
        d_model: int = 512,
        kernel_sizes=(3, 7, 15),
        pad_id: int = 0,
    ):
        super().__init__()

        # N = unknown, X = masked, P = pad (id 0)
        self.nt_to_id = {
            "A": 1, "C": 2, "G": 3, "T": 4,
            "N": 5, "X": 6,
            "P": pad_id,
        }
        self.id_to_nt = {v: k for k, v in self.nt_to_id.items()}
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)

        self.convs = nn.ModuleList(
            nn.Conv1d(d_embed, d_model, k, padding=k // 2)
            for k in kernel_sizes
        )

        self.proj = nn.Linear(len(kernel_sizes) * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

    def encode_strings(self, seqs: List[str]) -> torch.Tensor:
        """Encode list[str] → (B, Lmax) ids"""
        B = len(seqs)
        lengths = [len(s) for s in seqs]
        Lmax = max(lengths)

        arr = torch.full((B, Lmax), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            for j, ch in enumerate(s.upper()):
                arr[i, j] = self.nt_to_id.get(ch, self.nt_to_id["N"])
        return arr

    def decode_ids(self, ids: List[int]) -> str:
        """Decode token IDs back to nucleotide sequence."""
        return "".join(self.id_to_nt.get(int(i), "N") for i in ids)

    def forward(
        self,
        x: Union[str, List[str], torch.Tensor],
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            hidden:   (B, L, D)
            pad_mask: (B, L) with 1 = valid, 0 = pad
        """
        # normalise input to (B, L) ids
        if isinstance(x, str):
            input_ids = self.encode_strings([x])
        elif isinstance(x, list) and isinstance(x[0], str):
            input_ids = self.encode_strings(x)
        else:
            input_ids = x  # assume tensor (B, L)

        device = self.embed.weight.device
        input_ids = input_ids.to(device)

        if pad_mask is None:
            pad_mask = (input_ids != self.pad_id).long()
        else:
            pad_mask = pad_mask.to(device)

        emb = self.embed(input_ids)          # (B, L, d_embed)
        x_t = emb.transpose(1, 2)            # (B, d_embed, L)

        feats = []
        for conv in self.convs:
            feats.append(F.gelu(conv(x_t)))  # (B, d_model, L)

        h_cat = torch.cat(feats, dim=1)      # (B, K*d_model, L)
        h = h_cat.transpose(1, 2)            # (B, L, K*d_model)
        h = self.proj(h)                     # (B, L, d_model)
        h = self.norm(h)
        h = self.pos_enc(h)

        return h, pad_mask


class SimpleMLMHead(nn.Module):
    """Simple MLM head: hidden (B, L, D) -> logits (B, L, V)"""
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tie_weights: bool = False,
        embedding_weight: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=True)
        if tie_weights:
            if embedding_weight is None:
                raise ValueError("tie_weights=True but embedding_weight is None.")
            if embedding_weight.size(1) != d_model:
                raise ValueError(f"Embedding dim {embedding_weight.size(1)} != d_model {d_model}")
            self.proj.weight = embedding_weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class SimpleGenomicMLM(nn.Module):
    """Simple genomic MLM model with conv tokenizer."""
    def __init__(
        self,
        vocab_size: int = 7,
        pad_id: int = 0,
        d_embed: int = 64,
        d_model: int = 512,
        kernel_sizes=(3, 7, 15),
        tie_weights: bool = False,
    ):
        super().__init__()
        self.tokenizer = NucConvTokenizer(
            vocab_size=vocab_size,
            d_embed=d_embed,
            d_model=d_model,
            kernel_sizes=kernel_sizes,
            pad_id=pad_id,
        )

        emb_w = self.tokenizer.embed.weight if tie_weights else None
        self.mlm_head = SimpleMLMHead(
            d_model=d_model,
            vocab_size=vocab_size,
            tie_weights=tie_weights,
            embedding_weight=emb_w,
        )

    def forward(
        self,
        x: Union[torch.Tensor, List[str]],
        pad_mask: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
    ):
        hidden, pad_mask = self.tokenizer(x, pad_mask)
        logits = self.mlm_head(hidden)

        loss = None
        if mlm_labels is not None:
            B, L, V = logits.shape
            loss = F.cross_entropy(
                logits.view(-1, V),
                mlm_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss, hidden


# ====================== Wrapper Class ======================

class GenomeNetWrapper(BaseWrapper):
    """
    GenomeNet wrapper for CheapGenomicMLM model.
    Supports both embeddings and nucleotide prediction via MLM.
    """

    def __init__(
        self,
        *,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        # Model architecture parameters
        vocab_size: int = 7,
        pad_id: int = 0,
        d_embed: int = 64,
        d_model: int = 512,
        kernel_sizes: Tuple[int, ...] = (3, 7, 15),
        tie_weights: bool = False,
    ):
        super().__init__()

        # Device setup
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available()
                       else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                             else "cpu"))
        )
        self.dtype = torch.float32

        # Default checkpoint path (updated to sally_25)
        self.default_checkpoint_path = "/tamir2/nicolaslynn/projects/genomenet/genomenet/simple_training_model_states/mlm_encoder_sally_25.pt"
        if checkpoint_path is None:
            checkpoint_path = self.default_checkpoint_path

        # Initialize model
        self.model = SimpleGenomicMLM(
            vocab_size=vocab_size,
            pad_id=pad_id,
            d_embed=d_embed,
            d_model=d_model,
            kernel_sizes=kernel_sizes,
            tie_weights=tie_weights,
        ).to(self.device).eval()

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

        # Store tokenizer reference for convenience
        self.tokenizer = self.model.tokenizer

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint (handles DataParallel models)."""
        try:
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            state_dict = torch.load(checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Handle DataParallel models (strip 'module.' prefix)
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading checkpoint: {unexpected_keys}")

    def find_N_positions(self, seq: str) -> List[int]:
        """Find positions with 'N' nucleotides."""
        return [i for i, c in enumerate(seq) if c.upper() == "N"]

    @torch.no_grad()
    def embed(
        self,
        seq: Union[str, List[str]],
        *,
        pool: str = "mean",
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray]]:
        """
        Generate embeddings for the input sequence(s).

        Parameters
        ----------
        seq : str or List[str]
            Input DNA sequence(s)
        pool : str
            Pooling strategy: 'mean', 'cls', or 'tokens'
        return_numpy : bool
            Whether to return numpy arrays or tensors

        Returns
        -------
        embeddings : np.ndarray or torch.Tensor
            Embeddings with shape depending on pool strategy
        """
        is_batch = isinstance(seq, (list, tuple))

        if not is_batch:
            # Single sequence
            hidden, pad_mask = self.model.tokenizer(seq)  # (1, L, D)

            if pool == "mean":
                # Use sequence_mean_pool utility directly on batch
                emb = sequence_mean_pool(hidden, pad_mask)[0]  # (D,)
            elif pool == "cls":
                emb = hidden[0, 0]  # (D,) - first token of first sequence
            elif pool == "tokens":
                emb = hidden[0]  # (L, D) - tokens of first sequence
            else:
                raise ValueError("pool must be one of {'mean', 'cls', 'tokens'}")

            return emb.detach().cpu().numpy() if return_numpy else emb.detach().cpu()

        # Batch processing
        results = [self.embed(s, pool=pool, return_numpy=False) for s in seq]

        if pool == "tokens":
            # Variable lengths -> return list
            if return_numpy:
                return [r.detach().cpu().numpy() for r in results]
            return [r.detach().cpu() for r in results]

        # Stack for mean/cls
        stacked = torch.stack(results, dim=0)  # (B, D)
        return stacked.detach().cpu().numpy() if return_numpy else stacked.detach().cpu()

    @torch.no_grad()
    def predict_nucleotides(
        self,
        seq: str,
        positions: Optional[List[int]] = None,
        *,
        return_dict: bool = True,
    ) -> Union[List[Dict[str, float]], np.ndarray]:
        """
        Predict nucleotide probabilities at specified positions.

        Parameters
        ----------
        seq : str
            Input DNA sequence
        positions : List[int], optional
            Positions to predict. If None, auto-detects 'N' positions
        return_dict : bool
            Whether to return list of dicts or numpy array

        Returns
        -------
        predictions : List[Dict] or np.ndarray
            Nucleotide probabilities at each position
        """
        # Auto-detect 'N' sites if not provided
        if not positions:
            positions = self.find_N_positions(seq)
        if not positions:
            raise ValueError("No positions provided and no 'N' bases found in seq.")

        # Create masked sequence for MLM prediction
        masked_seq = list(seq.upper())
        for pos in positions:
            if 0 <= pos < len(masked_seq):
                masked_seq[pos] = 'X'  # Use 'X' as mask token
        masked_seq_str = "".join(masked_seq)

        # Get MLM logits
        logits, _, _ = self.model(masked_seq_str)  # (1, L, V)
        logits = logits[0]  # (L, V)

        if return_dict:
            results = []
            for pos in positions:
                if not (0 <= pos < len(seq)):
                    raise IndexError(f"Position {pos} out of range for sequence length {len(seq)}")

                # Get logits for nucleotide tokens only (A=1, C=2, G=3, T=4)
                nt_logits = logits[pos, [1, 2, 3, 4]]  # A, C, G, T
                probs = torch.softmax(nt_logits, dim=0).detach().cpu().numpy()

                results.append({
                    'A': float(probs[0]),
                    'C': float(probs[1]),
                    'G': float(probs[2]),
                    'T': float(probs[3]),
                })
            return results
        else:
            results = np.zeros((len(positions), 4), dtype=np.float32)
            for i, pos in enumerate(positions):
                if not (0 <= pos < len(seq)):
                    raise IndexError(f"Position {pos} out of range for sequence length {len(seq)}")

                nt_logits = logits[pos, [1, 2, 3, 4]]
                probs = torch.softmax(nt_logits, dim=0).detach().cpu().numpy()
                results[i] = probs
            return results


# ====================== Example Usage ======================

if __name__ == "__main__":
    # Initialize wrapper
    wrapper = GenomeNetWrapper()

    # Test sequence
    seq = "ACGTNACGTACGTN"

    # Test embedding
    emb = wrapper.embed(seq, pool="mean")
    print(f"Embedding shape: {emb.shape}")

    # Test nucleotide prediction (will auto-detect N positions)
    preds = wrapper.predict_nucleotides(seq)
    print(f"Predictions: {preds}")