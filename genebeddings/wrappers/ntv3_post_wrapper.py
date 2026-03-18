# ntv3_post_wrapper.py
# Nucleotide Transformer v3 POST-TRAINED wrapper
# Standardized API: embed()
#
# Uses the post-trained NTv3 model which has been further trained on
# ~16,000 functional tracks across 24 species. Embeddings from this
# model encode track-prediction knowledge (expression, chromatin,
# splicing) in addition to the self-supervised pretraining.
#
# Requires species_ids for forward pass — hardcoded to human.
# Sequences must be padded to multiples of 128.

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

try:
    from .base_wrapper import BaseWrapper, PoolMode
except ImportError:
    from base_wrapper import BaseWrapper, PoolMode

logger = logging.getLogger(__name__)

BASES = ("A", "C", "G", "T")

NTV3_POST_MODELS: Dict[str, str] = {
    "100m-post": "InstaDeepAI/NTv3_100M_post",
    "650m-post": "InstaDeepAI/NTv3_650M_post",
    "100m-post-131kb": "InstaDeepAI/NTv3_100M_post_131kb",
    "650m-post-131kb": "InstaDeepAI/NTv3_650M_post_131kb",
    # Pipeline aliases
    "ntv3_100m_post": "InstaDeepAI/NTv3_100M_post",
    "ntv3_650m_post": "InstaDeepAI/NTv3_650M_post",
}

PAD_MULTIPLE = 128


class NTv3PostWrapper(BaseWrapper):
    """
    Nucleotide Transformer v3 post-trained wrapper.

    The post-trained model has been further trained on functional genomic
    tracks. Its embeddings encode track-prediction knowledge, making them
    more informative for variant effect analysis than the pre-trained model.

    Always runs as human (species_ids hardcoded).

    Parameters
    ----------
    model : str, default="100m-post"
        Model variant. See NTV3_POST_MODELS for options.
    device : str, optional
        Device to use.
    dtype : torch.dtype, default=torch.float32
        Data type for model weights.

    Examples
    --------
    >>> wrapper = NTv3PostWrapper(model="100m-post")
    >>> emb = wrapper.embed("ACGT" * 200, pool="mean")
    """
    TRUST_REMOTE = True

    def __init__(
        self,
        model: str = "100m-post",
        *,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if model in NTV3_POST_MODELS:
            model_id = NTV3_POST_MODELS[model]
            self.model_name = model
        else:
            model_id = model
            self.model_name = model

        self.model_id = model_id

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

        # Model — post-trained uses AutoModel, not AutoModelForMaskedLM
        logger.info("Loading NTv3 post-trained model: %s", model_id)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=self.TRUST_REMOTE,
        ).to(self.device).eval()

        # Human species ID (with special token offset)
        species_list = list(self.model.config.bigwigs_per_species.keys())
        num_special = getattr(self.model.config, "num_species_special_tokens", 0)
        self._human_species_id = species_list.index("human") + num_special
        logger.info("Loaded NTv3 post-trained: human species_id=%d", self._human_species_id)

        # Cache token mapping for fast tokenization
        self._vocab = self.tokenizer.get_vocab()
        self._char_to_id = {c: self._vocab[c] for c in "ACGTN" if c in self._vocab}
        pad_tok = self.tokenizer.pad_token
        self._pad_id = self._vocab.get(pad_tok, 0) if pad_tok else 0

    def __repr__(self) -> str:
        return f"NTv3PostWrapper(model='{self.model_name}', device={self.device})"

    def _pad_length(self, length: int) -> int:
        remainder = length % PAD_MULTIPLE
        if remainder == 0:
            return length
        return length + PAD_MULTIPLE - remainder

    def _fast_tokenize(self, seq: str, padded_len: int) -> List[int]:
        ids = [self._char_to_id.get(c, self._char_to_id.get("N", 0)) for c in seq.upper()]
        if len(ids) < padded_len:
            ids.extend([self._pad_id] * (padded_len - len(ids)))
        elif len(ids) > padded_len:
            ids = ids[:padded_len]
        return ids

    def _tokenize_batch(self, seqs: List[str]) -> dict:
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
        Generate embeddings from the post-trained NTv3 model.

        Parameters
        ----------
        seq : str or list of str
            Input DNA sequence(s).
        pool : {'mean', 'cls', 'tokens'}, default='mean'
            Pooling strategy.
        return_numpy : bool, default=True
            Return numpy array if True.
        """
        is_batch = isinstance(seq, (list, tuple))
        seqs = list(seq) if is_batch else [seq]
        seq_lengths = [len(s) for s in seqs]
        batch_size = len(seqs)

        enc = self._tokenize_batch(seqs)
        species_ids = torch.full((batch_size,), self._human_species_id,
                                  dtype=torch.long, device=self.device)

        out = self.model(
            **enc,
            species_ids=species_ids,
            output_hidden_states=True,
        )

        hidden = out.hidden_states[-1]  # (B, L_padded, H)

        results = []
        for i in range(batch_size):
            L = seq_lengths[i]
            h = hidden[i, :L, :]

            if pool == "mean":
                pooled = h.mean(dim=0)
            elif pool == "cls":
                pooled = h[0]
            elif pool == "tokens":
                pooled = h
            else:
                raise ValueError(f"Unknown pool mode: {pool!r}")

            if return_numpy:
                results.append(pooled.cpu().float().numpy())
            else:
                results.append(pooled)

        if not is_batch:
            return results[0]
        if pool == "tokens":
            return results
        return np.stack(results) if return_numpy else torch.stack(results)

    def supports_capability(self, capability: str) -> bool:
        return capability in ("embed",)


if __name__ == "__main__":
    model = NTv3PostWrapper(model="100m-post")
    seq = "ACGT" * 64
    emb = model.embed(seq, pool="mean")
    print(f"Embedding shape: {emb.shape}")
    print(f"Embedding dtype: {emb.dtype}")
