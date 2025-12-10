# evo2_wrapper.py
# Evo2 genomic foundation model wrapper
# Supports Evo2 model variants (7b, 1b, etc.)
# Standardized API: embed(), predict_nucleotides()
#
# Notes:
# * Nucleotide-level tokenization (k=1)
# * Embeddings extracted from specified layers

import re
from typing import Dict, Optional, List, Union, Literal

import numpy as np
import torch

# Support both package import and direct sys.path import
try:
    from .base_wrapper import BaseWrapper, PoolMode
except ImportError:
    from base_wrapper import BaseWrapper, PoolMode

BASES = ("A", "C", "G", "T")
DNA_SET = set("ACGTN")

# Model registry: short name -> model ID for Evo2
EVO2_MODELS: Dict[str, str] = {
    "7b": "evo2_7b",
    "7b_base": "evo2_7b_base",
    "1b": "evo2_1b",
    "evo2_7b": "evo2_7b",
    "evo2_7b_base": "evo2_7b_base",
    "evo2_1b": "evo2_1b",
}

# Default embedding layers for each model size
DEFAULT_EMBEDDING_LAYERS: Dict[str, str] = {
    "evo2_7b": "blocks.28.mlp.l3",
    "evo2_7b_base": "blocks.28.mlp.l3",
    "evo2_1b": "blocks.14.mlp.l3",  # Approximate middle layer for 1b
}

# Type alias for model selection
Evo2ModelName = Literal["7b", "7b_base", "1b", "evo2_7b", "evo2_7b_base", "evo2_1b"]


def list_available_models() -> List[str]:
    """Return list of available model short names."""
    return list(EVO2_MODELS.keys())


class Evo2Wrapper(BaseWrapper):
    """
    Evo2 genomic foundation model wrapper.
    Implements BaseWrapper: embed(), predict_nucleotides().

    Parameters
    ----------
    model : str, default="7b"
        Model to use. Can be:
        - A short name from the registry (e.g., "7b", "1b")
        - A full model ID (e.g., "evo2_7b")
    device : str, optional
        Device to use. Defaults to CUDA if available, else CPU.
        Note: Evo2 typically requires CUDA.
    dtype : torch.dtype, default=torch.bfloat16
        Data type for model weights. bfloat16 recommended for Evo2.
    embedding_layer : str, optional
        Layer to extract embeddings from. If None, uses model-specific default.
        Format: 'blocks.{N}.mlp.l3' where N is the block number.

    Examples
    --------
    >>> wrapper = Evo2Wrapper(model="7b")
    >>> wrapper = Evo2Wrapper(model="1b", dtype=torch.float16)

    >>> # Get embeddings
    >>> emb = wrapper.embed("ACGTACGT", pool="mean")

    >>> # Predict nucleotides at N positions
    >>> preds = wrapper.predict_nucleotides("ACNTNACGT")
    """

    def __init__(
        self,
        model: str = "7b",
        *,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        embedding_layer: Optional[str] = None,
    ):
        super().__init__()

        # Lazy import evo2 to avoid import errors if not installed
        try:
            from evo2 import Evo2
        except ImportError as e:
            raise ImportError(
                "evo2 package not found. Install with: pip install evo2"
            ) from e

        # Resolve model ID
        if model in EVO2_MODELS:
            model_id = EVO2_MODELS[model]
            self.model_name = model
        else:
            model_id = model
            self.model_name = model

        self.model_id = model_id

        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype

        # Load model
        self.model = Evo2(model_id)

        # Set embedding layer
        if embedding_layer is not None:
            self.embedding_layer = embedding_layer
        else:
            self.embedding_layer = DEFAULT_EMBEDDING_LAYERS.get(
                model_id, "blocks.28.mlp.l3"
            )

        # Nucleotide-level tokenization
        self.k = 1

        # Cache for vocab mapping (for nucleotide prediction)
        self._build_vocab_mapping()

    def __repr__(self) -> str:
        return f"Evo2Wrapper(model='{self.model_name}', device={self.device}, embedding_layer='{self.embedding_layer}')"

    def _build_vocab_mapping(self):
        """Build mapping from token IDs to nucleotides."""
        self._base_to_id: Dict[str, int] = {}
        self._id_to_base: Dict[int, str] = {}

        # Tokenize each base to find its token ID
        for base in BASES:
            token_ids = self.model.tokenizer.tokenize(base)
            if len(token_ids) == 1:
                self._base_to_id[base] = token_ids[0]
                self._id_to_base[token_ids[0]] = base

    def _normalize_seq(self, seq: str, *, max_nt: Optional[int] = None) -> str:
        """
        Clean to A/C/G/T/N, optionally truncate.
        """
        s = re.sub(r"[^ACGTN]", "N", (seq or "").upper())
        if max_nt is not None and len(s) > max_nt:
            s = s[:max_nt]
        return s

    def _tokenize(self, seq: str) -> torch.Tensor:
        """Tokenize a sequence and return tensor on device."""
        token_ids = self.model.tokenizer.tokenize(seq)
        return torch.tensor(token_ids, dtype=torch.int).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def embed(
        self,
        seq: Union[str, List[str]],
        *,
        pool: PoolMode = "mean",
        return_numpy: bool = True,
        layer: Optional[str] = None,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray]]:
        """
        Generate embeddings for DNA sequence(s).

        If seq is a string -> returns (H,) for mean/cls, or (L,H) for tokens.
        If seq is a list[str] -> returns (B,H) for mean/cls, or list[(Li,H)] for tokens.

        Parameters
        ----------
        seq : str or list of str
            Input DNA sequence(s)
        pool : {'mean', 'cls', 'tokens'}, default='mean'
            Pooling strategy:
            - 'mean': Average over all tokens
            - 'cls': Use first token (note: Evo2 may not have a true CLS token)
            - 'tokens': Return all token embeddings
        return_numpy : bool, default=True
            If True, return numpy array; if False, return torch.Tensor
        layer : str, optional
            Layer to extract embeddings from. If None, uses default.
            Format: 'blocks.{N}.mlp.l3'
        """
        layer_name = layer if layer is not None else self.embedding_layer

        is_batch = isinstance(seq, (list, tuple))
        if not is_batch:
            seq = [seq]

        results = []
        for s in seq:
            s_norm = self._normalize_seq(s)
            input_ids = self._tokenize(s_norm)

            _, embeddings = self.model(
                input_ids,
                return_embeddings=True,
                layer_names=[layer_name]
            )

            emb = embeddings[layer_name]  # (1, L, H)

            if pool == "tokens":
                emb_out = emb[0]  # (L, H)
            elif pool == "cls":
                emb_out = emb[0, 0]  # (H,)
            elif pool == "mean":
                emb_out = emb[0].mean(dim=0)  # (H,)
            else:
                raise ValueError("pool must be one of {'mean', 'cls', 'tokens'}")

            results.append(emb_out)

        if not is_batch:
            emb_out = results[0]
            return emb_out.detach().cpu().float().numpy() if return_numpy else emb_out.detach().cpu()

        # Batched return
        if pool == "tokens":
            # Variable lengths -> return list
            return [
                r.detach().cpu().float().numpy() if return_numpy else r.detach().cpu()
                for r in results
            ]

        # Stack for mean/cls
        stacked = torch.stack(results, dim=0)  # (B, H)
        return stacked.detach().cpu().float().numpy() if return_numpy else stacked.detach().cpu()

    @torch.no_grad()
    def predict_nucleotides(
        self,
        seq: str,
        positions: Optional[List[int]] = None,
        *,
        return_dict: bool = True,
    ) -> Union[List[Dict[str, float]], np.ndarray]:
        """
        Predict nucleotide probabilities at specified positions using logits.

        Parameters
        ----------
        seq : str
            Input DNA sequence
        positions : list of int, optional
            0-based positions in the sequence to predict.
            If None, auto-detects 'N' positions.
        return_dict : bool, default=True
            If True, return list of dicts with keys 'A', 'C', 'G', 'T'
            If False, return numpy array of shape (len(positions), 4)
        """
        # Auto-detect 'N' sites if not provided
        if positions is None:
            positions = self.find_N_positions(seq)
        if not positions:
            raise ValueError("No positions provided and no 'N' bases found in seq.")

        s = self._normalize_seq(seq)
        L = len(s)
        pos_list = [int(p) for p in positions if 0 <= int(p) < L]

        if not pos_list:
            raise ValueError("No valid positions after filtering.")

        # Get logits from model
        input_ids = self._tokenize(s)
        outputs, _ = self.model(input_ids)
        logits = outputs[0]  # (L, vocab_size)

        out_list: List[Dict[str, float]] = []

        for p in pos_list:
            if p >= logits.shape[0]:
                continue

            # Get logits at position
            pos_logits = logits[p]  # (vocab_size,)

            # Extract probabilities for each base
            base_probs = {}
            for base in BASES:
                if base in self._base_to_id:
                    token_id = self._base_to_id[base]
                    base_probs[base] = float(pos_logits[token_id].item())

            # Convert logits to probabilities via softmax over just the bases
            logit_vals = torch.tensor([base_probs[b] for b in BASES])
            probs = torch.softmax(logit_vals, dim=0)

            prob_dict = {b: float(probs[i].item()) for i, b in enumerate(BASES)}
            out_list.append(prob_dict)

        if return_dict:
            return out_list

        arr = np.zeros((len(out_list), 4), dtype=np.float32)
        for i, d in enumerate(out_list):
            arr[i] = [d["A"], d["C"], d["G"], d["T"]]
        return arr

    def find_N_positions(self, seq: str) -> List[int]:
        """Indices where the input has 'N'."""
        return [i for i, c in enumerate(seq) if c.upper() == "N"]

    def generate(
        self,
        prompt: Union[str, List[str]],
        n_tokens: int = 100,
        *,
        temperature: float = 1.0,
        top_k: int = 4,
    ) -> Union[str, List[str]]:
        """
        Generate DNA sequence continuation.

        Parameters
        ----------
        prompt : str or list of str
            Starting sequence(s) to continue from
        n_tokens : int, default=100
            Number of tokens (nucleotides) to generate
        temperature : float, default=1.0
            Sampling temperature (higher = more random)
        top_k : int, default=4
            Top-k sampling parameter

        Returns
        -------
        sequences : str or list of str
            Generated sequence(s) including the prompt
        """
        is_batch = isinstance(prompt, (list, tuple))
        if not is_batch:
            prompts = [prompt]
        else:
            prompts = list(prompt)

        output = self.model.generate(
            prompt_seqs=prompts,
            n_tokens=n_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        if not is_batch:
            return output.sequences[0]
        return output.sequences

    def get_available_layers(self) -> List[str]:
        """
        Get list of available layer names for embedding extraction.

        Returns
        -------
        layers : list of str
            Available layer names in format 'blocks.{N}.mlp.l3'
        """
        # Evo2 7B has ~32 blocks, 1B has ~16 blocks
        if "7b" in self.model_id.lower():
            n_blocks = 32
        elif "1b" in self.model_id.lower():
            n_blocks = 16
        else:
            n_blocks = 32  # Default assumption

        return [f"blocks.{i}.mlp.l3" for i in range(n_blocks)]
