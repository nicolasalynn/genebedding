# splicebert_wrapper.py
# SpliceBERT wrapper for RNA foundation models
# Standardized API: embed(), predict_nucleotides()
#
# SpliceBERT is a BERT-style model pre-trained on mRNA precursor sequences.
# Supports both:
# - Local models using standard HuggingFace transformers
# - HuggingFace models via multimolecule library

import os
import re
from pathlib import Path
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

# Get package assets directory
_PACKAGE_DIR = Path(__file__).parent.parent
ASSETS_DIR = _PACKAGE_DIR / "assets"
DEFAULT_LOCAL_PATH = ASSETS_DIR / "splicebert"

# Model registry for HuggingFace models (via multimolecule)
SPLICEBERT_MODELS: Dict[str, str] = {
    "splicebert": "multimolecule/splicebert",
    "510": "multimolecule/splicebert.510",
    "510nt": "multimolecule/splicebert.510nt",
    "human-510": "multimolecule/splicebert-human.510",
    # Aliases
    "default": "multimolecule/splicebert",
    "human": "multimolecule/splicebert-human.510",
}

SpliceBertModelName = Literal["splicebert", "510", "510nt", "human-510", "default", "human", "local"]


def list_available_models() -> List[str]:
    """Return list of available model short names."""
    models = list(SPLICEBERT_MODELS.keys())
    if DEFAULT_LOCAL_PATH.exists():
        models.append("local")
    return models


class SpliceBertWrapper(BaseWrapper):
    """
    SpliceBERT wrapper with standardized API.

    SpliceBERT is a BERT-style model pre-trained on over 2 million vertebrate
    mRNA precursor sequences. It's designed for RNA splicing and other
    RNA-related tasks.

    Implements BaseWrapper: embed(), predict_nucleotides()

    Parameters
    ----------
    model : str, default="local"
        Model to use. Can be:
        - "local": Load from assets/splicebert directory (default)
        - A path to a local model directory
        - A short name from registry (e.g., "splicebert", "human-510")
        - A full HuggingFace model ID (e.g., "multimolecule/splicebert")
    device : str, optional
        Device to use. Defaults to CUDA if available, else MPS, else CPU.
    dtype : torch.dtype, default=torch.float32
        Data type for model weights.
    load_mlm : bool, default=True
        Whether to load MLM head for nucleotide prediction.

    Examples
    --------
    >>> # Load from local assets/splicebert (default)
    >>> wrapper = SpliceBertWrapper()

    >>> # Load from custom local path
    >>> wrapper = SpliceBertWrapper(model="/path/to/splicebert")

    >>> # Load from HuggingFace via multimolecule
    >>> wrapper = SpliceBertWrapper(model="splicebert")

    >>> # Get embeddings (DNA input - auto-converted)
    >>> emb = wrapper.embed("ACGTACGT", pool="mean")

    >>> # RNA input works directly
    >>> emb = wrapper.embed("ACGUACGU", pool="mean")

    >>> # Predict nucleotides at N positions
    >>> probs = wrapper.predict_nucleotides("ACGUNACGU")

    Notes
    -----
    - SpliceBERT works with RNA sequences (A, C, G, U)
    - DNA sequences (with T) work - local model converts to T, HuggingFace converts to U
    - Model may not work well on sequences shorter than 64nt
    - Local models use whitespace tokenization (char-level)
    - HuggingFace models require multimolecule: pip install multimolecule
    """

    def __init__(
        self,
        model: str = "local",
        *,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        load_mlm: bool = True,
    ):
        super().__init__()

        # Device/dtype
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            )
        self.device = torch.device(device)
        self.dtype = dtype

        # Determine if using local model or HuggingFace
        self._is_local = False
        model_path = None

        if model == "local":
            model_path = DEFAULT_LOCAL_PATH
            self._is_local = True
        elif os.path.isdir(model):
            model_path = Path(model)
            self._is_local = True
        elif model in SPLICEBERT_MODELS:
            model_path = SPLICEBERT_MODELS[model]
        else:
            # Assume it's a HuggingFace model ID
            model_path = model

        self.model_path = model_path
        self.model_name = model

        if self._is_local:
            self._init_local_model(model_path, load_mlm)
        else:
            self._init_huggingface_model(model_path, load_mlm)

    def _init_local_model(self, model_path: Path, load_mlm: bool):
        """Initialize from local model directory using standard transformers."""
        from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Local SpliceBERT model not found at {model_path}. "
                f"Please download and place the model files there."
            )

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModel.from_pretrained(str(model_path)).to(self.device).to(self.dtype).eval()

        # Load MLM head if requested
        if load_mlm:
            self.mlm = AutoModelForMaskedLM.from_pretrained(str(model_path)).to(self.device).to(self.dtype).eval()
            self.mask_id = self.tokenizer.mask_token_id
        else:
            self.mlm = None
            self.mask_id = None

        # Get token IDs for nucleotides (local model uses T, not U)
        self._base_token_ids = {}
        for base in BASES_DNA:
            tokens = self.tokenizer(base, add_special_tokens=False)["input_ids"]
            if tokens:
                self._base_token_ids[base] = tokens[0]

    def _init_huggingface_model(self, model_id: str, load_mlm: bool):
        """Initialize from HuggingFace via multimolecule."""
        try:
            import multimolecule  # noqa: F401
            from multimolecule import RnaTokenizer, SpliceBertModel
        except ImportError:
            raise ImportError(
                "HuggingFace SpliceBERT models require the multimolecule package. Install with:\n"
                "pip install multimolecule\n"
                "Or use a local model: SpliceBertWrapper(model='local')"
            )

        # Load tokenizer and model
        self.tokenizer = RnaTokenizer.from_pretrained(model_id)
        self.model = SpliceBertModel.from_pretrained(model_id).to(self.device).to(self.dtype).eval()

        # Load MLM head if requested
        if load_mlm:
            from multimolecule import SpliceBertForMaskedLM
            self.mlm = SpliceBertForMaskedLM.from_pretrained(model_id).to(self.device).to(self.dtype).eval()
            self.mask_id = self.tokenizer.mask_token_id
        else:
            self.mlm = None
            self.mask_id = None

        # Get token IDs for nucleotides (HuggingFace model uses U)
        self._base_token_ids = {}
        for base in BASES_RNA:
            tokens = self.tokenizer(base, add_special_tokens=False)["input_ids"]
            if tokens:
                self._base_token_ids[base] = tokens[0]

    def __repr__(self) -> str:
        model_type = "local" if self._is_local else "huggingface"
        return f"SpliceBertWrapper(model='{self.model_name}', type={model_type}, device={self.device})"

    def _normalize_seq(self, seq: str) -> str:
        """Clean sequence and convert for model."""
        seq = (seq or "").upper()

        if self._is_local:
            # Local model: convert U to T, add whitespace between chars
            seq = seq.replace("U", "T")
            seq = re.sub(r"[^ACGTN]", "N", seq)
            # Add whitespace between nucleotides for char-level tokenization
            seq = " ".join(list(seq))
        else:
            # HuggingFace model: convert T to U
            seq = seq.replace("T", "U")
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
            Input sequence(s). DNA and RNA both work.
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
        use_dna_bases: bool = True,
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
            If True, return list of dicts with keys 'A', 'C', 'G', 'T' (or 'U')
        use_dna_bases : bool, default=True
            If True, return DNA bases (A, C, G, T); if False, return RNA (A, C, G, U)

        Returns
        -------
        predictions : list of dict or np.ndarray
        """
        if self.mlm is None:
            raise NotImplementedError("MLM head not loaded. Initialize with load_mlm=True")

        # Normalize but keep original for position mapping
        seq_upper = (seq or "").upper()
        seq_clean = seq_upper.replace("U", "T") if self._is_local else seq_upper.replace("T", "U")
        seq_clean = re.sub(r"[^ACGTUN]", "N", seq_clean)

        # Auto-detect N positions if not provided
        if not positions:
            positions = [i for i, c in enumerate(seq_clean) if c == "N"]
        if not positions:
            raise ValueError("No positions provided and no 'N' bases found in seq.")

        results = []

        for pos in positions:
            # Create masked sequence
            seq_list = list(seq_clean)
            seq_list[pos] = "[MASK]" if self._is_local else self.tokenizer.mask_token
            masked_seq = "".join(seq_list)

            # For local model, normalize adds whitespace
            if self._is_local:
                masked_seq = " ".join(list(masked_seq.replace("[MASK]", self.tokenizer.mask_token)))

            # Encode
            enc = self.tokenizer(masked_seq, return_tensors="pt", padding=False, truncation=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}

            # Get logits
            logits = self.mlm(**enc).logits  # (1, L, V)

            # Adjust for special tokens (CLS at start)
            adjusted_pos = pos + 1

            # Get probabilities for nucleotides
            pos_logits = logits[0, adjusted_pos, :]
            probs_tensor = torch.softmax(pos_logits, dim=-1)

            probs = {}
            for base, tok_id in self._base_token_ids.items():
                probs[base] = float(probs_tensor[tok_id].cpu())

            # Normalize to sum to 1
            total = sum(probs.values())
            if total > 0:
                probs = {b: p / total for b, p in probs.items()}

            # Convert bases if needed
            if self._is_local and not use_dna_bases:
                # Local model has T, convert to U
                probs = {b.replace("T", "U"): p for b, p in probs.items()}
            elif not self._is_local and use_dna_bases:
                # HuggingFace model has U, convert to T
                probs = {b.replace("U", "T"): p for b, p in probs.items()}

            if return_dict:
                results.append(probs)
            else:
                bases = BASES_DNA if use_dna_bases else BASES_RNA
                results.append([probs[b] for b in bases])

        if return_dict:
            return results

        return np.array(results, dtype=np.float32)

    def find_N_positions(self, seq: str) -> List[int]:
        """Indices where the input has 'N'."""
        seq = (seq or "").upper()
        return [i for i, c in enumerate(seq) if c == "N"]
