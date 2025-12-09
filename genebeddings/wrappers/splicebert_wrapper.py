# splicebert_wrapper.py
# SpliceBERT wrapper for RNA foundation models
# Standardized API: embed(), predict_nucleotides()
#
# SpliceBERT is a BERT-style model pre-trained on mRNA precursor sequences.
# It uses the multimolecule library and supports RNA sequences (U instead of T).
# Models are from the multimolecule HuggingFace org.

import re
from typing import Dict, Optional, List, Union, Literal

import numpy as np
import torch

# Support both package import and direct sys.path import
try:
    from .base_wrapper import BaseWrapper, PoolMode
except ImportError:
    from base_wrapper import BaseWrapper, PoolMode

BASES_RNA = ("A", "C", "G", "U")
BASES_DNA = ("A", "C", "G", "T")

# Model registry
SPLICEBERT_MODELS: Dict[str, str] = {
    "splicebert": "multimolecule/splicebert",
    "510": "multimolecule/splicebert.510",
    "510nt": "multimolecule/splicebert.510nt",
    "human-510": "multimolecule/splicebert-human.510",
    # Aliases
    "default": "multimolecule/splicebert",
    "human": "multimolecule/splicebert-human.510",
}

SpliceBertModelName = Literal["splicebert", "510", "510nt", "human-510", "default", "human"]


def list_available_models() -> List[str]:
    """Return list of available model short names."""
    return list(SPLICEBERT_MODELS.keys())


class SpliceBertWrapper(BaseWrapper):
    """
    SpliceBERT wrapper with standardized API.

    SpliceBERT is a BERT-style model pre-trained on over 2 million vertebrate
    mRNA precursor sequences. It's designed for RNA splicing and other
    RNA-related tasks.

    Implements BaseWrapper: embed(), predict_nucleotides()

    Parameters
    ----------
    model : str, default="splicebert"
        Model to use. Can be:
        - A short name from registry (e.g., "splicebert", "human-510")
        - A full HuggingFace model ID (e.g., "multimolecule/splicebert")
    device : str, optional
        Device to use. Defaults to CUDA if available, else MPS, else CPU.
    dtype : torch.dtype, default=torch.float32
        Data type for model weights.
    load_mlm : bool, default=True
        Whether to load MLM head for nucleotide prediction.
    convert_dna_to_rna : bool, default=True
        Whether to automatically convert T to U in input sequences.

    Examples
    --------
    >>> wrapper = SpliceBertWrapper(model="splicebert")

    >>> # Get embeddings (DNA input - auto-converted to RNA)
    >>> emb = wrapper.embed("ACGTACGT", pool="mean")

    >>> # RNA input works directly
    >>> emb = wrapper.embed("ACGUACGU", pool="mean")

    >>> # Predict nucleotides at N positions
    >>> probs = wrapper.predict_nucleotides("ACGUNACGU")

    Notes
    -----
    - SpliceBERT works with RNA sequences (A, C, G, U)
    - DNA sequences (with T) are automatically converted to RNA (U) by default
    - Model may not work well on sequences shorter than 64nt
    - Requires the multimolecule package: pip install multimolecule
    """

    def __init__(
        self,
        model: str = "splicebert",
        *,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        load_mlm: bool = True,
        convert_dna_to_rna: bool = True,
    ):
        super().__init__()

        # Resolve model ID
        if model in SPLICEBERT_MODELS:
            model_id = SPLICEBERT_MODELS[model]
            self.model_name = model
        else:
            model_id = model
            self.model_name = model

        self.model_id = model_id
        self.convert_dna_to_rna = convert_dna_to_rna

        # Device/dtype
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            )
        self.device = torch.device(device)
        self.dtype = dtype

        # Import multimolecule (required to register models)
        try:
            import multimolecule  # noqa: F401
            from multimolecule import RnaTokenizer, SpliceBertModel
        except ImportError:
            raise ImportError(
                "SpliceBERT requires the multimolecule package. Install with:\n"
                "pip install multimolecule"
            )

        # Load tokenizer and model
        self.tokenizer = RnaTokenizer.from_pretrained(model_id)
        self.model = SpliceBertModel.from_pretrained(model_id).to(self.device).to(dtype).eval()

        # Load MLM head if requested
        if load_mlm:
            from multimolecule import SpliceBertForMaskedLM
            self.mlm = SpliceBertForMaskedLM.from_pretrained(model_id).to(self.device).to(dtype).eval()
            self.mask_id = self.tokenizer.mask_token_id
        else:
            self.mlm = None
            self.mask_id = None

    def __repr__(self) -> str:
        return f"SpliceBertWrapper(model='{self.model_name}', device={self.device})"

    def _normalize_seq(self, seq: str) -> str:
        """Clean sequence and optionally convert DNA to RNA."""
        seq = (seq or "").upper()
        # Convert T to U if requested
        if self.convert_dna_to_rna:
            seq = seq.replace("T", "U")
        # Replace invalid chars with N
        seq = re.sub(r"[^ACGUN]", "N", seq)
        return seq

    def _encode_one(self, seq: str) -> Dict[str, torch.Tensor]:
        """Encode a single sequence."""
        seq = self._normalize_seq(seq)
        enc = self.tokenizer(seq, return_tensors="pt", padding=False, truncation=True)
        return {k: v.to(self.device) for k, v in enc.items()}

    def _encode_many(self, seqs: List[str]) -> Dict[str, torch.Tensor]:
        """Encode multiple sequences with padding."""
        seqs = [self._normalize_seq(s) for s in seqs]
        enc = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

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
        Generate embeddings for RNA/DNA sequence(s).

        If seq is a string -> returns (H,) for mean/cls, or (L,H) for tokens.
        If seq is a list[str] -> returns (B,H) for mean/cls, or list[(Li,H)] for tokens.

        Parameters
        ----------
        seq : str or list of str
            Input sequence(s). DNA (T) is auto-converted to RNA (U).
        pool : {'mean', 'cls', 'tokens'}, default='mean'
            Pooling strategy:
            - 'mean': Average over all positions
            - 'cls': Use CLS token embedding
            - 'tokens': Return all token embeddings
        return_numpy : bool, default=True
            If True, return numpy array; if False, return torch.Tensor
        layer : int, optional
            Which layer to extract. If None, uses last hidden state.
        """
        is_batch = isinstance(seq, (list, tuple))

        if not is_batch:
            # Single sequence
            enc = self._encode_one(seq)
            out = self._get_hidden_states(enc, layer=layer)  # (1, L, H)

            if pool == "tokens":
                emb = out[0]  # (L, H)
            elif pool == "cls":
                emb = out[0, 0]  # (H,)
            elif pool == "mean":
                mask = enc["attention_mask"].unsqueeze(-1)  # (1, L, 1)
                denom = mask.sum(dim=1).clamp(min=1)
                emb = ((out * mask).sum(dim=1) / denom)[0]  # (H,)
            else:
                raise ValueError("pool must be one of {'mean', 'cls', 'tokens'}")

            return emb.detach().cpu().numpy() if return_numpy else emb.detach().cpu()

        # Batch processing
        enc = self._encode_many(list(seq))
        out = self._get_hidden_states(enc, layer=layer)  # (B, L, H)

        if pool == "tokens":
            # Variable lengths -> return list
            attn = enc["attention_mask"]
            pieces = []
            for b in range(out.size(0)):
                Lb = int(attn[b].sum().item())
                emb = out[b, :Lb]
                pieces.append(emb.detach().cpu().numpy() if return_numpy else emb.detach().cpu())
            return pieces

        if pool == "cls":
            emb = out[:, 0]  # (B, H)
        elif pool == "mean":
            mask = enc["attention_mask"].unsqueeze(-1)  # (B, L, 1)
            denom = mask.sum(dim=1).clamp(min=1)
            emb = (out * mask).sum(dim=1) / denom  # (B, H)
        else:
            raise ValueError("pool must be one of {'mean', 'cls', 'tokens'}")

        return emb.detach().cpu().numpy() if return_numpy else emb.detach().cpu()

    def _get_hidden_states(self, enc: Dict[str, torch.Tensor], layer: Optional[int] = None) -> torch.Tensor:
        """Get hidden states from model."""
        out = self.model(**enc, output_hidden_states=(layer is not None))

        if layer is not None:
            if hasattr(out, 'hidden_states'):
                return out.hidden_states[layer]
            return out.last_hidden_state

        return out.last_hidden_state

    @torch.no_grad()
    def predict_nucleotides(
        self,
        seq: str,
        positions: Optional[List[int]] = None,
        *,
        return_dict: bool = True,
        use_rna_bases: bool = True,
    ) -> Union[List[Dict[str, float]], np.ndarray]:
        """
        Predict nucleotide probabilities at specified positions.

        Parameters
        ----------
        seq : str
            Input sequence. N positions are auto-detected if positions not provided.
        positions : list of int, optional
            0-based positions to predict. If None, auto-detects 'N' positions.
        return_dict : bool, default=True
            If True, return list of dicts with keys 'A', 'C', 'G', 'U' (or 'T')
        use_rna_bases : bool, default=True
            If True, return RNA bases (A, C, G, U); if False, return DNA (A, C, G, T)

        Returns
        -------
        predictions : list of dict or np.ndarray
        """
        if self.mlm is None:
            raise NotImplementedError("MLM head not loaded. Initialize with load_mlm=True")

        seq = self._normalize_seq(seq)

        # Auto-detect N positions if not provided
        if not positions:
            positions = [i for i, c in enumerate(seq) if c == "N"]
        if not positions:
            raise ValueError("No positions provided and no 'N' bases found in seq.")

        # Get token indices for RNA bases
        base_tokens = {}
        for base in BASES_RNA:
            tokens = self.tokenizer(base, add_special_tokens=False)["input_ids"]
            if tokens:
                base_tokens[base] = tokens[0]

        results = []

        for pos in positions:
            # Create masked sequence
            seq_list = list(seq)
            seq_list[pos] = self.tokenizer.mask_token
            masked_seq = "".join(seq_list)

            # Encode and get logits
            enc = self._encode_one(masked_seq)
            logits = self.mlm(**enc).logits  # (1, L, V)

            # Adjust for special tokens (CLS at start)
            adjusted_pos = pos + 1

            # Get probabilities for nucleotides
            pos_logits = logits[0, adjusted_pos, :]
            probs_tensor = torch.softmax(pos_logits, dim=-1)

            probs = {}
            for base, tok_id in base_tokens.items():
                probs[base] = float(probs_tensor[tok_id].cpu())

            # Normalize to sum to 1
            total = sum(probs.values())
            if total > 0:
                probs = {b: p / total for b, p in probs.items()}

            # Convert to DNA bases if requested
            if not use_rna_bases:
                probs = {b.replace("U", "T"): p for b, p in probs.items()}

            if return_dict:
                results.append(probs)
            else:
                bases = BASES_RNA if use_rna_bases else BASES_DNA
                results.append([probs[b] for b in bases])

        if return_dict:
            return results

        return np.array(results, dtype=np.float32)

    def find_N_positions(self, seq: str) -> List[int]:
        """Indices where the input has 'N'."""
        seq = self._normalize_seq(seq)
        return [i for i, c in enumerate(seq) if c == "N"]
