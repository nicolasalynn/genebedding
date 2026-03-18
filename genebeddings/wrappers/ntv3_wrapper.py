# ntv3_wrapper.py
# Nucleotide Transformer v3 wrapper
# Standardized API: embed(), predict_nucleotides()
#
# NTv3 uses a U-Net architecture with single-base tokenization and
# convolutional downsampling. Unlike NT v1/v2, it does NOT use k-mer
# tokenization. Sequences must be padded to multiples of 128 (2^7
# downsample stages).
#
# Available models:
#   Pre-trained: NTv3_8M_pre, NTv3_100M_pre, NTv3_650M_pre
#   Post-trained: NTv3_100M_post, NTv3_650M_post
#   Intermediate: NTv3_*_8kb, NTv3_*_131kb

import logging
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

try:
    from .base_wrapper import BaseWrapper, PoolMode
except ImportError:
    from base_wrapper import BaseWrapper, PoolMode

logger = logging.getLogger(__name__)

BASES = ("A", "C", "G", "T")

# Model registry: short name -> HuggingFace model ID
NTV3_MODELS: Dict[str, str] = {
    # Pre-trained (self-supervised only)
    "8m-pre": "InstaDeepAI/NTv3_8M_pre",
    "100m-pre": "InstaDeepAI/NTv3_100M_pre",
    "650m-pre": "InstaDeepAI/NTv3_650M_pre",
    # Post-trained (self-supervised + supervised tracks)
    "100m-post": "InstaDeepAI/NTv3_100M_post",
    "650m-post": "InstaDeepAI/NTv3_650M_post",
    # Intermediate checkpoints (shorter context, for exploration)
    "8m-pre-8kb": "InstaDeepAI/NTv3_8M_pre_8kb",
    "100m-pre-8kb": "InstaDeepAI/NTv3_100M_pre_8kb",
    "650m-pre-8kb": "InstaDeepAI/NTv3_650M_pre_8kb",
    "100m-post-131kb": "InstaDeepAI/NTv3_100M_post_131kb",
    "650m-post-131kb": "InstaDeepAI/NTv3_650M_post_131kb",
    # Pipeline aliases
    "ntv3_100m": "InstaDeepAI/NTv3_100M_pre",
    "ntv3_650m": "InstaDeepAI/NTv3_650M_pre",
    "ntv3_650m_post": "InstaDeepAI/NTv3_650M_post",
}

# Number of downsample stages determines padding requirement
DEFAULT_NUM_DOWNSAMPLES = 7
PAD_MULTIPLE = 2 ** DEFAULT_NUM_DOWNSAMPLES  # 128


def list_available_models() -> List[str]:
    """Return list of available model short names."""
    return list(NTV3_MODELS.keys())


class NTv3Wrapper(BaseWrapper):
    """
    Nucleotide Transformer v3 wrapper.

    NTv3 uses a U-Net architecture with single-base tokenization (A, C, G, T, N),
    convolutional downsampling (7 stages), a transformer core, and deconvolutional
    upsampling with skip connections.

    Parameters
    ----------
    model : str, default="650m-pre"
        Model to use. Can be a short name from the registry or a full HuggingFace ID.
    device : str, optional
        Device to use. Defaults to CUDA if available.
    dtype : torch.dtype, default=torch.float32
        Data type. Use torch.bfloat16 for long sequences to save memory.
    max_length : int, optional
        Maximum sequence length. If None, uses model default.

    Examples
    --------
    >>> wrapper = NTv3Wrapper(model="650m-pre")
    >>> emb = wrapper.embed("ACGTACGT" * 100, pool="mean")  # (1536,)
    >>> emb.shape
    (1536,)

    >>> # Nucleotide prediction
    >>> preds = wrapper.predict_nucleotides("ACNTNACGT")
    """
    TRUST_REMOTE = True

    def __init__(
        self,
        model: str = "650m-pre",
        *,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        max_length: Optional[int] = None,
    ):
        super().__init__()

        # Resolve model ID
        if model in NTV3_MODELS:
            model_id = NTV3_MODELS[model]
            self.model_name = model
        else:
            model_id = model
            self.model_name = model

        self.model_id = model_id

        # Device
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else ("mps" if getattr(torch.backends, "mps", None)
                      and torch.backends.mps.is_available() else "cpu")
            )
        self.device = torch.device(device)
        self.dtype = dtype

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=self.TRUST_REMOTE
        )
        self.max_length = max_length or getattr(
            self.tokenizer, "model_max_length", 1_000_000
        )

        # Model
        logger.info("Loading NTv3 model: %s", model_id)
        self.mlm = AutoModelForMaskedLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=self.TRUST_REMOTE,
        ).to(self.device).eval()
        logger.info("Loaded NTv3 model: %s", model_id)

        # Vocab for nucleotide prediction
        self._vocab = self.tokenizer.get_vocab()
        self._base_ids = {b: self._vocab.get(b) for b in BASES if b in self._vocab}

        # Cache the token mapping to avoid slow added_tokens_encoder rebuild
        # NTv3's tokenizer rebuilds this dict on every call, which is O(vocab) per token
        self._char_to_id = {c: self._vocab[c] for c in "ACGTN" if c in self._vocab}
        pad_tok = self.tokenizer.pad_token
        self._pad_id = self._vocab.get(pad_tok, 0) if pad_tok else 0

    def __repr__(self) -> str:
        return (f"NTv3Wrapper(model='{self.model_name}', device={self.device}, "
                f"max_length={self.max_length:,})")

    def _pad_length(self, length: int) -> int:
        """Round up to nearest multiple of PAD_MULTIPLE (128)."""
        remainder = length % PAD_MULTIPLE
        if remainder == 0:
            return length
        return length + PAD_MULTIPLE - remainder

    def _fast_tokenize(self, seq: str, padded_len: int) -> List[int]:
        """Fast character-level tokenization bypassing slow HF tokenizer."""
        ids = [self._char_to_id.get(c, self._char_to_id.get("N", 0)) for c in seq.upper()]
        # Pad to padded_len
        if len(ids) < padded_len:
            ids.extend([self._pad_id] * (padded_len - len(ids)))
        elif len(ids) > padded_len:
            ids = ids[:padded_len]
        return ids

    def _tokenize(self, seq: str) -> dict:
        """Tokenize a sequence, padding to multiple of 128."""
        padded_len = self._pad_length(len(seq))
        ids = self._fast_tokenize(seq, padded_len)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[0, len(seq):] = 0
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _tokenize_batch(self, seqs: List[str]) -> dict:
        """Tokenize a batch of sequences, padding to same length (multiple of 128)."""
        max_len = max(len(s) for s in seqs)
        padded_len = self._pad_length(max_len)
        batch_ids = [self._fast_tokenize(s, padded_len) for s in seqs]
        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        for i, s in enumerate(seqs):
            attention_mask[i, len(s):] = 0
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def embed(
        self,
        seq: Union[str, List[str]],
        *,
        pool: PoolMode = "mean",
        return_numpy: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray]]:
        """
        Generate embeddings for DNA sequence(s).

        Parameters
        ----------
        seq : str or list of str
            Input DNA sequence(s).
        pool : {'mean', 'cls', 'tokens'}, default='mean'
            Pooling strategy. 'cls' uses the first token.
        return_numpy : bool, default=True
            Return numpy array if True, torch.Tensor if False.
        """
        is_batch = isinstance(seq, (list, tuple))
        seqs = list(seq) if is_batch else [seq]
        seq_lengths = [len(s) for s in seqs]

        enc = self._tokenize_batch(seqs)
        out = self.mlm(**enc, output_hidden_states=True)

        # Last hidden state from hidden_states tuple
        hidden = out.hidden_states[-1]  # (B, L_padded, H)

        results = []
        for i in range(len(seqs)):
            L = seq_lengths[i]
            h = hidden[i, :L, :]  # trim to actual sequence length

            if pool == "mean":
                pooled = h.mean(dim=0)  # (H,)
            elif pool == "cls":
                pooled = h[0]  # (H,)
            elif pool == "tokens":
                pooled = h  # (L, H)
            else:
                raise ValueError(f"Unknown pool mode: {pool!r}")

            if return_numpy:
                results.append(pooled.cpu().float().numpy())
            else:
                results.append(pooled)

        if not is_batch:
            return results[0]
        if pool == "tokens":
            return results  # list of (Li, H) arrays
        return np.stack(results) if return_numpy else torch.stack(results)

    @torch.no_grad()
    def predict_nucleotides(
        self,
        seq: str,
        positions: Optional[List[int]] = None,
        *,
        return_dict: bool = True,
    ) -> Union[List[Dict[str, float]], np.ndarray]:
        """
        Predict nucleotide probabilities using MLM logits.

        Parameters
        ----------
        seq : str
            Input DNA sequence. 'N' positions are automatically detected.
        positions : list of int, optional
            Positions to predict. If None, predicts at all 'N' positions.
        return_dict : bool, default=True
            If True, return list of {base: prob} dicts.
        """
        if positions is None:
            positions = [i for i, c in enumerate(seq) if c.upper() == "N"]

        if not positions:
            return [] if return_dict else np.array([])

        # Replace N with mask token for prediction
        seq_list = list(seq.upper())
        mask_token = self.tokenizer.mask_token or "[MASK]"
        mask_id = self.tokenizer.mask_token_id
        for pos in positions:
            seq_list[pos] = mask_token

        masked_seq = "".join(seq_list)
        enc = self._tokenize(masked_seq)
        out = self.mlm(**enc)
        logits = out.logits[0]  # (L, V)

        results = []
        for pos in positions:
            if pos >= logits.shape[0]:
                results.append({b: 0.25 for b in BASES})
                continue

            pos_logits = logits[pos]
            probs = torch.softmax(pos_logits, dim=-1)

            if return_dict:
                d = {}
                for base, bid in self._base_ids.items():
                    if bid is not None:
                        d[base] = float(probs[bid])
                # Normalize to sum to 1 across ACGT
                total = sum(d.values()) or 1.0
                d = {b: p / total for b, p in d.items()}
                results.append(d)
            else:
                row = [float(probs[self._base_ids.get(b, 0)]) for b in BASES]
                results.append(row)

        if return_dict:
            return results
        return np.array(results)

    def supports_capability(self, capability: str) -> bool:
        """Check if this wrapper supports a given capability."""
        return capability in ("embed", "predict_nucleotides")


if __name__ == "__main__":
    # Quick test
    model = NTv3Wrapper(model="650m-pre-8kb")
    seq = "ACGT" * 64  # 256bp, multiple of 128
    emb = model.embed(seq, pool="mean")
    print(f"Embedding shape: {emb.shape}")
    print(f"Embedding dtype: {emb.dtype}")

    # Token embeddings
    tok_emb = model.embed(seq, pool="tokens")
    print(f"Token embedding shape: {tok_emb.shape}")

    # Nucleotide prediction
    test_seq = "ACGT" * 31 + "NCGT"  # 128bp with one N
    preds = model.predict_nucleotides(test_seq)
    print(f"Predictions at N: {preds}")
