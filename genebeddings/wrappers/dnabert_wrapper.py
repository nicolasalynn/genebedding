# dnabert_wrapper.py
# Standardized API: embed(), predict_nucleotides()
# Note: DNABERT-2 uses BPE tokenization with variable-length tokens

import re
from typing import Dict, Optional, List, Union, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Support both package import and direct sys.path import
try:
    from .base_wrapper import BaseWrapper, PoolMode
except ImportError:
    from base_wrapper import BaseWrapper, PoolMode

BASES = ("A", "C", "G", "T")
DNA_SET = set("ACGTN")


class DNABERTWrapper(BaseWrapper):
    """
    DNABERT-2 wrapper with standardized API.

    Implements BaseWrapper: embed(), predict_nucleotides()

    Note: DNABERT-2 uses BPE tokenization with variable-length tokens.
    The predict_nucleotides method handles this by:
    1. Finding which token(s) cover the position of interest
    2. Masking the token and running MLM prediction
    3. Filtering vocab probabilities to tokens matching the surrounding context
    4. Aggregating base probabilities at the specific position within tokens

    Args:
        model_id: HuggingFace model identifier
        device: Device to load model on (auto-detected if None)
        dtype: Model dtype (default: float32)
        attn_implementation: Attention implementation ("eager" for PyTorch, "flash_attention_2" for FlashAttn)
                           Default is "eager" to avoid Triton compatibility issues with flash attention.
    """
    MODEL_ID = "zhihan1996/DNABERT-2-117M"
    TRUST_REMOTE = True

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        attn_implementation: str = "eager",
    ):
        super().__init__()
        # device/dtype
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            )
        self.device = torch.device(device)
        self.dtype = dtype

        # tokenizer / model (use MLM model for predictions)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.TRUST_REMOTE)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=self.TRUST_REMOTE,
            attn_implementation=attn_implementation,  # Use "eager" to avoid Triton flash-attn issues
        ).to(self.device).eval()

        # Get mask token id
        self.mask_id = self.tokenizer.mask_token_id
        if self.mask_id is None:
            raise ValueError("Tokenizer has no [MASK] token; MLM predictions require it.")

        # Cache for vocab analysis
        self._vocab = self.tokenizer.get_vocab()
        self._id2dna_cache: Dict[int, str] = {}

        # Build index of DNA tokens by length for efficient filtering
        self._build_dna_token_index()

    def _build_dna_token_index(self):
        """Build index of vocabulary tokens that are pure DNA sequences."""
        self._dna_tokens_by_len: Dict[int, List[Tuple[int, str]]] = {}

        for tok_str, tok_id in self._vocab.items():
            # Clean token string to get DNA content
            dna = self._clean_token_to_dna(tok_str)
            if dna and set(dna).issubset(DNA_SET) and 'N' not in dna:
                length = len(dna)
                if length not in self._dna_tokens_by_len:
                    self._dna_tokens_by_len[length] = []
                self._dna_tokens_by_len[length].append((tok_id, dna))

    def _clean_token_to_dna(self, tok_str: str) -> str:
        """Extract DNA sequence from token string (remove BPE markers etc)."""
        # Remove common BPE markers and non-DNA characters
        cleaned = re.sub(r"[^ACGTNacgtn]", "", tok_str)
        return cleaned.upper()

    def _id2dna(self, tok_id: int) -> str:
        """Get DNA string for a token id (cached)."""
        if tok_id in self._id2dna_cache:
            return self._id2dna_cache[tok_id]
        tok_str = self.tokenizer.convert_ids_to_tokens(tok_id)
        dna = self._clean_token_to_dna(tok_str or "")
        self._id2dna_cache[tok_id] = dna
        return dna

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
        layer : int, optional
            Which transformer layer to extract (0 = embedding, -1 = last).
            If None, uses last hidden state.
        """
        is_batch = isinstance(seq, (list, tuple))

        if not is_batch:
            # single sequence
            enc = self._encode_one(seq)
            out = self._get_hidden_states(enc, layer=layer)

            if pool == "tokens":
                emb = out[0]  # (L,H)
            elif pool == "cls":
                emb = out[:, 0][0]  # (H,)
            elif pool == "mean":
                mask = enc["attention_mask"].unsqueeze(-1)  # (1,L,1)
                denom = mask.sum(dim=1).clamp(min=1)
                emb = ((out * mask).sum(dim=1) / denom)[0]  # (H,)
            else:
                raise ValueError("pool must be one of {'mean','cls','tokens'}")

            return emb.detach().cpu().numpy() if return_numpy else emb.detach().cpu()

        # batched path
        enc = self._encode_many(list(seq))
        out = self._get_hidden_states(enc, layer=layer)

        if pool == "tokens":
            # variable lengths -> return list of (Li,H)
            attn = enc["attention_mask"]
            pieces = []
            for b in range(out.size(0)):
                Lb = int(attn[b].sum().item())
                pieces.append(out[b, :Lb].detach().cpu().numpy() if return_numpy
                              else out[b, :Lb].detach().cpu())
            return pieces

        if pool == "cls":
            emb = out[:, 0]  # (B,H)
        elif pool == "mean":
            mask = enc["attention_mask"].unsqueeze(-1)  # (B,L,1)
            denom = mask.sum(dim=1).clamp(min=1)
            emb = ((out * mask).sum(dim=1) / denom)  # (B,H)
        else:
            raise ValueError("pool must be one of {'mean','cls','tokens'}")

        return emb.detach().cpu().numpy() if return_numpy else emb.detach().cpu()

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

        For BPE tokenization, this:
        1. Maps each position to its covering token
        2. Masks that token and runs MLM
        3. Filters vocab to tokens matching surrounding context pattern
        4. Aggregates base probabilities at the position within each matching token

        Parameters
        ----------
        seq : str
            Input DNA sequence
        positions : list of int, optional
            0-based positions to predict. If None, auto-detects 'N' positions.
        return_dict : bool
            If True, returns list of dicts with keys 'A','C','G','T'

        Returns
        -------
        predictions : list of dict or np.ndarray
        """
        s = self._normalize_seq(seq)
        L = len(s)

        # Auto-detect N positions if not provided
        if not positions:
            positions = [i for i, c in enumerate(s) if c == "N"]
        if not positions:
            raise ValueError("No positions provided and no 'N' bases found in seq.")

        # Tokenize to get char-to-token mapping
        enc = self.tokenizer(
            s,
            return_tensors="pt",
            add_special_tokens=True,
            return_offsets_mapping=True,
            padding=False,
            truncation=False,
        )

        input_ids = enc["input_ids"][0].tolist()
        offsets = enc["offset_mapping"][0].tolist()  # [(start, end), ...]

        # Build char_to_token mapping
        char_to_token = [None] * L
        for tok_idx, (start, end) in enumerate(offsets):
            if start == end:  # special token
                continue
            for ci in range(start, min(end, L)):
                char_to_token[ci] = tok_idx

        # Get token strings (without specials for pattern matching)
        token_strings = []
        for tok_idx, tok_id in enumerate(input_ids):
            dna = self._id2dna(tok_id)
            token_strings.append(dna)

        results: List[Dict[str, float]] = []

        for pos in positions:
            if not (0 <= pos < L):
                raise IndexError(f"Position {pos} out of range for sequence length {L}")

            tok_idx = char_to_token[pos]
            if tok_idx is None:
                # Position not covered by any token (shouldn't happen)
                results.append({b: 0.25 for b in BASES})
                continue

            # Get the token info
            start, end = offsets[tok_idx]
            tok_len = end - start
            rel_pos = pos - start  # position within token

            # Get surrounding context pattern
            # The token covers s[start:end], we want to match tokens where
            # the base at rel_pos can be A/C/G/T but other positions must match
            token_seq = s[start:end]
            pattern = list(token_seq)
            pattern[rel_pos] = "N"  # wildcard at position of interest
            pattern = "".join(pattern)

            # Mask the token and get logits
            masked_ids = input_ids.copy()
            masked_ids[tok_idx] = self.mask_id

            batch = {
                "input_ids": torch.tensor([masked_ids], device=self.device),
                "attention_mask": torch.ones(1, len(masked_ids), device=self.device, dtype=torch.long),
            }

            logits = self.model(**batch).logits[0, tok_idx, :]  # (V,)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

            # Filter and aggregate by pattern
            base_probs = self._pattern_filter(probs, pattern, rel_pos, tok_len)
            results.append(base_probs)

        if return_dict:
            return results

        arr = np.zeros((len(results), 4), dtype=np.float32)
        for i, d in enumerate(results):
            arr[i] = [d["A"], d["C"], d["G"], d["T"]]
        return arr

    def _pattern_filter(
        self,
        probs: np.ndarray,
        pattern: str,
        rel_pos: int,
        tok_len: int,
    ) -> Dict[str, float]:
        """
        Filter vocab probabilities by pattern and aggregate base probabilities.

        Parameters
        ----------
        probs : np.ndarray
            Full vocabulary probabilities (V,)
        pattern : str
            Pattern to match, with 'N' as wildcard at position of interest
        rel_pos : int
            Position within token where we want base probability
        tok_len : int
            Length of the token

        Returns
        -------
        dict : {'A': p, 'C': p, 'G': p, 'T': p}
        """
        base_mass = {b: 0.0 for b in BASES}

        # Get tokens of matching length
        if tok_len not in self._dna_tokens_by_len:
            return {b: 0.25 for b in BASES}

        for tok_id, dna in self._dna_tokens_by_len[tok_len]:
            # Check if token matches pattern
            if not self._matches_pattern(dna, pattern):
                continue

            # Get base at position of interest
            base = dna[rel_pos]
            if base in base_mass:
                base_mass[base] += float(probs[tok_id])

        # Normalize
        total = sum(base_mass.values())
        if total <= 0:
            return {b: 0.25 for b in BASES}

        return {b: v / total for b, v in base_mass.items()}

    def _matches_pattern(self, token: str, pattern: str) -> bool:
        """Check if token matches pattern (N is wildcard)."""
        if len(token) != len(pattern):
            return False
        for t, p in zip(token, pattern):
            if p != "N" and t != p:
                return False
        return True

    def _get_hidden_states(self, enc: Dict[str, torch.Tensor], layer: Optional[int] = None) -> torch.Tensor:
        """Get hidden states from model."""
        # Access base model for hidden states
        if hasattr(self.model, "bert"):
            backbone = self.model.bert
        elif hasattr(self.model, "base_model"):
            backbone = self.model.base_model
        else:
            backbone = self.model

        out = backbone(**enc, output_hidden_states=True)

        if layer is None:
            return out.last_hidden_state
        return out.hidden_states[layer]

    def _normalize_seq(self, seq: str) -> str:
        """Clean to A/C/G/T/N, uppercase."""
        return re.sub(r"[^ACGTN]", "N", (seq or "").upper())

    def _encode_one(self, seq: str) -> Dict[str, torch.Tensor]:
        s = self._normalize_seq(seq)
        enc = self.tokenizer(s, return_tensors="pt", add_special_tokens=True, padding=False, truncation=False)
        return {k: v.to(self.device) for k, v in enc.items()}

    def _encode_many(self, seqs: List[str]) -> Dict[str, torch.Tensor]:
        ss = [self._normalize_seq(s) for s in seqs]
        enc = self.tokenizer(
            ss, return_tensors="pt", add_special_tokens=True,
            padding="longest", truncation=False
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    # Legacy compatibility methods
    def tokens_from_seq(self, seq: str) -> List[int]:
        """Get token IDs for sequence (without special tokens)."""
        s = self._normalize_seq(seq)
        enc = self.tokenizer(s, add_special_tokens=False)
        return list(map(int, enc["input_ids"]))

    def find_N_positions(self, seq: str) -> List[int]:
        """Find positions with 'N' in sequence."""
        return [i for i, c in enumerate(seq) if c.upper() == "N"]
