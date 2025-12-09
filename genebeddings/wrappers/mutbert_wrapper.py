# mutbert_wrapper.py
# MutBERT wrapper for genomic foundation models
# Standardized API: embed()
#
# MutBERT uses one-hot encoded inputs instead of token IDs.
# Models are from the CompBioDSA HuggingFace org.

import re
from typing import Dict, Optional, List, Union, Literal

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Support both package import and direct sys.path import
try:
    from .base_wrapper import BaseWrapper, PoolMode
except ImportError:
    from base_wrapper import BaseWrapper, PoolMode

BASES = ("A", "C", "G", "T")

# Model registry
MUTBERT_MODELS: Dict[str, str] = {
    "mutbert": "CompBioDSA/MutBERT",
    "human-ref": "CompBioDSA/MutBERT-Human-Ref",
    "multi": "CompBioDSA/MutBERT-Multi",
    # Aliases
    "default": "CompBioDSA/MutBERT",
    "human": "CompBioDSA/MutBERT-Human-Ref",
}

MutBERTModelName = Literal["mutbert", "human-ref", "multi", "default", "human"]


def list_available_models() -> List[str]:
    """Return list of available model short names."""
    return list(MUTBERT_MODELS.keys())


class MutBERTWrapper(BaseWrapper):
    """
    MutBERT wrapper with standardized API.

    MutBERT is a DNA language model that uses one-hot encoded inputs
    instead of discrete token IDs. This allows for soft/probabilistic
    inputs and gradient-based analysis.

    Implements BaseWrapper: embed()

    Parameters
    ----------
    model : str, default="mutbert"
        Model to use. Can be:
        - A short name from registry (e.g., "mutbert", "human-ref", "multi")
        - A full HuggingFace model ID (e.g., "CompBioDSA/MutBERT")
    device : str, optional
        Device to use. Defaults to CUDA if available, else MPS, else CPU.
    dtype : torch.dtype, default=torch.float32
        Data type for model weights.

    Examples
    --------
    >>> wrapper = MutBERTWrapper(model="mutbert")
    >>> wrapper = MutBERTWrapper(model="human-ref")

    >>> # Get embeddings
    >>> emb = wrapper.embed("ACGTACGT", pool="mean")

    >>> # Batch embeddings
    >>> embs = wrapper.embed(["ACGT", "GCTA"], pool="mean")

    >>> # Use soft/probabilistic inputs
    >>> soft_input = torch.tensor([[[0.7, 0.1, 0.1, 0.1, 0, 0],  # mostly A
    ...                             [0.1, 0.7, 0.1, 0.1, 0, 0]]]) # mostly C
    >>> emb = wrapper.embed_soft(soft_input, pool="mean")
    """

    def __init__(
        self,
        model: str = "mutbert",
        *,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # Resolve model ID
        if model in MUTBERT_MODELS:
            model_id = MUTBERT_MODELS[model]
            self.model_name = model
        else:
            model_id = model
            self.model_name = model

        self.model_id = model_id

        # Device/dtype
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            )
        self.device = torch.device(device)
        self.dtype = dtype

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device).eval()

        # Vocab size for one-hot encoding
        self.vocab_size = len(self.tokenizer)

    def __repr__(self) -> str:
        return f"MutBERTWrapper(model='{self.model_name}', device={self.device})"

    def _normalize_seq(self, seq: str) -> str:
        """Clean sequence to valid DNA characters."""
        return re.sub(r"[^ACGTNacgtn]", "N", seq or "").upper()

    def _tokenize(self, seq: str) -> torch.Tensor:
        """Tokenize sequence and convert to one-hot."""
        seq = self._normalize_seq(seq)
        input_ids = self.tokenizer(seq, return_tensors='pt')["input_ids"]
        # Convert to one-hot
        one_hot = F.one_hot(input_ids, num_classes=self.vocab_size).float()
        return one_hot.to(self.device)

    def _tokenize_batch(self, seqs: List[str]) -> torch.Tensor:
        """Tokenize batch and convert to one-hot with padding."""
        seqs = [self._normalize_seq(s) for s in seqs]

        # Tokenize with padding
        enc = self.tokenizer(
            seqs,
            return_tensors='pt',
            padding='longest',
            truncation=True,
        )
        input_ids = enc["input_ids"]

        # Convert to one-hot
        one_hot = F.one_hot(input_ids, num_classes=self.vocab_size).float()
        return one_hot.to(self.device), enc.get("attention_mask", None)

    @torch.no_grad()
    def embed(
        self,
        seq: Union[str, List[str]],
        *,
        pool: PoolMode = "mean",
        return_numpy: bool = True,
        layer: Optional[int] = None,
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
            - 'mean': Average over all positions
            - 'cls': Use first position embedding
            - 'tokens': Return all position embeddings
        return_numpy : bool, default=True
            If True, return numpy array; if False, return torch.Tensor
        layer : int, optional
            Which layer to extract. If None, uses last hidden state.
        """
        is_batch = isinstance(seq, (list, tuple))

        if not is_batch:
            # Single sequence
            one_hot = self._tokenize(seq)  # (1, L, V)
            out = self._forward(one_hot, layer=layer)  # (1, L, H)

            if pool == "tokens":
                emb = out[0]  # (L, H)
            elif pool == "cls":
                emb = out[0, 0]  # (H,)
            elif pool == "mean":
                emb = out[0].mean(dim=0)  # (H,)
            else:
                raise ValueError("pool must be one of {'mean', 'cls', 'tokens'}")

            return emb.detach().cpu().numpy() if return_numpy else emb.detach().cpu()

        # Batch processing
        one_hot, attention_mask = self._tokenize_batch(list(seq))
        out = self._forward(one_hot, layer=layer)  # (B, L, H)

        if pool == "tokens":
            # Variable lengths -> return list
            pieces = []
            for b in range(out.size(0)):
                if attention_mask is not None:
                    Lb = int(attention_mask[b].sum().item())
                    emb = out[b, :Lb]
                else:
                    emb = out[b]
                pieces.append(emb.detach().cpu().numpy() if return_numpy else emb.detach().cpu())
            return pieces

        if pool == "cls":
            emb = out[:, 0]  # (B, H)
        elif pool == "mean":
            if attention_mask is not None:
                # Masked mean
                mask = attention_mask.unsqueeze(-1).float().to(self.device)
                emb = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                emb = out.mean(dim=1)  # (B, H)
        else:
            raise ValueError("pool must be one of {'mean', 'cls', 'tokens'}")

        return emb.detach().cpu().numpy() if return_numpy else emb.detach().cpu()

    @torch.no_grad()
    def embed_soft(
        self,
        soft_input: torch.Tensor,
        *,
        pool: PoolMode = "mean",
        return_numpy: bool = True,
        layer: Optional[int] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate embeddings from soft/probabilistic inputs.

        This allows for gradient-based analysis and soft mutations.

        Parameters
        ----------
        soft_input : torch.Tensor
            Soft one-hot input of shape (B, L, vocab_size) or (L, vocab_size)
            Values should sum to 1 along the last dimension.
        pool : {'mean', 'cls', 'tokens'}, default='mean'
            Pooling strategy
        return_numpy : bool, default=True
            If True, return numpy array
        layer : int, optional
            Which layer to extract

        Returns
        -------
        embeddings : np.ndarray or torch.Tensor
        """
        if soft_input.dim() == 2:
            soft_input = soft_input.unsqueeze(0)

        soft_input = soft_input.to(self.device).to(self.dtype)
        out = self._forward(soft_input, layer=layer)

        if pool == "tokens":
            emb = out.squeeze(0) if out.size(0) == 1 else out
        elif pool == "cls":
            emb = out[:, 0]
            if emb.size(0) == 1:
                emb = emb.squeeze(0)
        elif pool == "mean":
            emb = out.mean(dim=1)
            if emb.size(0) == 1:
                emb = emb.squeeze(0)
        else:
            raise ValueError("pool must be one of {'mean', 'cls', 'tokens'}")

        return emb.detach().cpu().numpy() if return_numpy else emb.detach().cpu()

    def _forward(self, one_hot: torch.Tensor, layer: Optional[int] = None) -> torch.Tensor:
        """Run forward pass and extract hidden states."""
        out = self.model(one_hot, output_hidden_states=(layer is not None))

        if layer is not None:
            if hasattr(out, 'hidden_states'):
                return out.hidden_states[layer]
            # Fallback
            return out.last_hidden_state

        return out.last_hidden_state

    def find_N_positions(self, seq: str) -> List[int]:
        """Indices where the input has 'N'."""
        return [i for i, c in enumerate(seq) if c.upper() == "N"]

    def create_soft_mutation(
        self,
        seq: str,
        position: int,
        probs: Dict[str, float],
    ) -> torch.Tensor:
        """
        Create a soft input with a probabilistic mutation at a position.

        Parameters
        ----------
        seq : str
            Original DNA sequence
        position : int
            Position to mutate (0-based, in the sequence not including special tokens)
        probs : dict
            Probability distribution over bases {'A': p_A, 'C': p_C, 'G': p_G, 'T': p_T}

        Returns
        -------
        soft_input : torch.Tensor
            Shape (1, L+special_tokens, vocab_size)
        """
        # First get the hard one-hot encoding
        one_hot = self._tokenize(seq)  # (1, L, V)

        # Adjust position for special tokens (CLS token at start)
        # MutBERT typically adds special tokens
        adjusted_pos = position + 1  # +1 for CLS token

        # Get token IDs for bases
        base_to_idx = {}
        for base in BASES:
            tokens = self.tokenizer(base, add_special_tokens=False)["input_ids"]
            if tokens:
                base_to_idx[base] = tokens[0]

        # Create soft distribution at the position
        one_hot[0, adjusted_pos, :] = 0  # Zero out
        for base, prob in probs.items():
            if base in base_to_idx:
                one_hot[0, adjusted_pos, base_to_idx[base]] = prob

        return one_hot
