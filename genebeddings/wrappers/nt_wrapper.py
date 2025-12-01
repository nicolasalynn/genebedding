# nt_wrapper.py
# Unified Nucleotide Transformer wrapper (non-overlapping fixed-k)
# Supports all NT model variants via model name selection
# Standardized API: embed(), predict_nucleotides()
#
# Notes:
# * Uses 'N' as wildcard in pattern matching
# * Automatically detects whether to use attn_implementation='eager' for v2 models

import re
from typing import Dict, Optional, List, Tuple, Union, Literal

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

# Model registry: short name -> (HuggingFace model ID, requires_eager_attn)
NT_MODELS: Dict[str, Tuple[str, bool]] = {
    # NT v2 models - 6mer tokenization (require eager attention)
    "v2-500m-multi": ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", True),
    "v2-250m-multi": ("InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", True),
    "v2-100m-multi": ("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", True),
    "v2-50m-multi": ("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", True),

    # NT v2 models - 3mer tokenization (require eager attention)
    "v2-50m-3mer": ("InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species", True),

    # NT v1 models (standard attention)
    "2.5b-multi": ("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", False),
    "2.5b-1000g": ("InstaDeepAI/nucleotide-transformer-2.5b-1000g", False),
    "500m-multi": ("InstaDeepAI/nucleotide-transformer-500m-multi-species", False),
    "500m-1000g": ("InstaDeepAI/nucleotide-transformer-500m-1000g", False),
    "500m-human-ref": ("InstaDeepAI/nucleotide-transformer-500m-human-ref", False),
}

# Type alias for model selection
NTModelName = Literal[
    "v2-500m-multi", "v2-250m-multi", "v2-100m-multi", "v2-50m-multi",
    "v2-50m-3mer",
    "2.5b-multi", "2.5b-1000g", "500m-multi", "500m-1000g", "500m-human-ref",
]


def list_available_models() -> List[str]:
    """Return list of available model short names."""
    return list(NT_MODELS.keys())


class NTWrapper(BaseWrapper):
    """
    Unified Nucleotide Transformer wrapper supporting all NT model variants.
    Implements BaseWrapper: embed(), predict_nucleotides().

    Parameters
    ----------
    model : str, default="v2-500m-multi"
        Model to use. Can be:
        - A short name from the registry (e.g., "v2-500m-multi", "500m-human-ref")
        - A full HuggingFace model ID (e.g., "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    device : str, optional
        Device to use. Defaults to CUDA if available, else MPS, else CPU.
    dtype : torch.dtype, default=torch.float32
        Data type for model weights.
    load_mlm : bool, default=True
        Whether to load the MLM head. Set to False for embeddings-only use.

    Examples
    --------
    >>> wrapper = NTWrapper(model="v2-500m-multi")
    >>> wrapper = NTWrapper(model="500m-human-ref")
    >>> wrapper = NTWrapper(model="2.5b-multi", dtype=torch.float16)

    >>> # Get embeddings
    >>> emb = wrapper.embed("ACGTACGT", pool="mean")

    >>> # Batch embeddings
    >>> embs = wrapper.embed(["ACGT", "GCTA"], pool="mean")

    >>> # Predict nucleotides at N positions
    >>> preds = wrapper.predict_nucleotides("ACNTNACGT")
    """
    TRUST_REMOTE = True

    def __init__(
        self,
        model: str = "v2-500m-multi",
        *,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        load_mlm: bool = True,
    ):
        super().__init__()

        # Resolve model ID and attention implementation
        if model in NT_MODELS:
            model_id, use_eager_attn = NT_MODELS[model]
            self.model_name = model
        else:
            # Assume it's a full HuggingFace model ID
            model_id = model
            # v2 models need eager attention
            use_eager_attn = "-v2-" in model_id.lower()
            self.model_name = model_id

        self.model_id = model_id

        # device/dtype
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            )
        self.device = torch.device(device)
        self.dtype = dtype

        # tokenizer / kmers
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.TRUST_REMOTE)
        self.k = self._infer_k_nonoverlap()
        self.max_length = self.tokenizer.model_max_length

        self.mask_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        # model (MLM used for both logits + last hidden state)
        if load_mlm:
            model_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": self.TRUST_REMOTE,
            }
            if use_eager_attn:
                model_kwargs["attn_implementation"] = "eager"

            self.mlm = AutoModelForMaskedLM.from_pretrained(
                model_id, **model_kwargs
            ).to(self.device).eval()

            if self.mask_id is None:
                raise ValueError("Tokenizer has no [MASK]; MLM probabilities require it.")
        else:
            self.mlm = None

        # small caches
        self._id2dna_cache: Dict[int, str] = {}
        self._vocab = self.tokenizer.get_vocab()  # token_str -> id

    def __repr__(self) -> str:
        return f"NTWrapper(model='{self.model_name}', device={self.device}, k={self.k})"

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
            Pooling strategy
        return_numpy : bool, default=True
            If True, return numpy array; if False, return torch.Tensor
        layer : int, optional
            Which transformer layer to extract (0 = embedding layer, -1 = last layer).
            If None, uses last hidden state.
        """
        is_batch = isinstance(seq, (list, tuple))
        if not is_batch:
            # single sequence
            enc = self._encode_one(seq)
            out = self._get_hidden_states(enc, layer=layer)  # (1,L,H)

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
        out = self._get_hidden_states(enc, layer=layer)  # (B,L,H)

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

    # ================= Optional capability (implemented) =================
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
        positions : list of int, optional
            0-based positions in the sequence to predict.
            If None, auto-detects 'N' positions.
        return_dict : bool, default=True
            If True, return list of dicts with keys 'A', 'C', 'G', 'T'
            If False, return numpy array of shape (len(positions), 4)
        """
        if self.mlm is None:
            raise NotImplementedError("Initialized without MLM head; nucleotide prediction unavailable.")

        # Auto-detect 'N' sites if not provided
        if not positions:
            positions = self.find_N_positions(seq)
        if not positions:
            raise ValueError("No positions provided and no 'N' bases found in seq.")

        # Normalize (trim to multiple of k); keep track of new length
        s = self._normalize_seq(seq)
        L = len(s)
        pos_list = [int(p) for p in positions if 0 <= int(p) < L]
        token_ids = self._tokens_from_seq(s)

        logits_cache: Dict[int, torch.Tensor] = {}
        out_list: List[Dict[str, float]] = []

        for p in pos_list:
            t_idx, rel = p // self.k, p % self.k
            if t_idx >= len(token_ids):
                continue  # position in trimmed tail (rare if not multiple of k)
            if t_idx not in logits_cache:
                logits_cache[t_idx] = self._masked_token_logits(token_ids, t_idx)
            logits_vec = logits_cache[t_idx]

            # Get the actual k-mer from the sequence to use as pattern
            # Per the paper: filter to only k-mers matching reference at all positions EXCEPT rel
            kmer_start = t_idx * self.k
            kmer_end = kmer_start + self.k
            ref_kmer = list(s[kmer_start:kmer_end])
            ref_kmer[rel] = "N"  # wildcard only at the position of interest
            pattern = "".join(ref_kmer)
            out_list.append(self._pattern_filter(logits_vec, pattern, masked_idx_in_token=rel))

        if return_dict:
            return out_list
        arr = np.zeros((len(out_list), 4), dtype=np.float32)
        for i, d in enumerate(out_list):
            arr[i] = [d["A"], d["C"], d["G"], d["T"]]
        return arr

    # ================= Small public helpers =================

    def find_N_positions(self, seq: str) -> List[int]:
        """Indices where the input has 'N'."""
        return [i for i, c in enumerate(seq) if c.upper() == "N"]

    def apply_mutations(self, seq: str, muts: List[Tuple[int, str, str]]) -> str:
        """Apply SNP-like mutations: (pos, ref, alt) with permissive ref=='N'."""
        s = list(seq)
        for pos, ref, alt in muts:
            if 0 <= pos < len(s) and (s[pos].upper() == ref.upper() or ref.upper() == "N"):
                s[pos] = alt.upper()
        return "".join(s)

    def center_token_info(self, seq: str, center_base_index: int) -> Tuple[int, int, str]:
        """
        Return (token_idx, rel_in_token, k-mer chunk) after normalizing/trim-to-k.
        """
        s = self._normalize_seq(seq)
        if not (0 <= center_base_index < len(s)):
            raise IndexError("center_base_index out of range after normalization.")
        t_idx, rel = center_base_index // self.k, center_base_index % self.k
        if (t_idx + 1) * self.k > len(s):
            raise IndexError("center token exceeds normalized sequence length.")
        chunk = s[t_idx * self.k: (t_idx + 1) * self.k]
        return t_idx, rel, chunk

    # ================= Internal helpers =================
    def _normalize_seq(self, seq: str, *, max_nt: Optional[int] = None) -> str:
        """
        Clean to A/C/G/T/N, optionally truncate to max_nt, then trim RIGHT
        to be divisible by k (so non-overlap k-mer math is exact).
        """
        s = re.sub(r"[^ACGTN]", "N", (seq or "").upper())
        if max_nt is not None and len(s) > max_nt:
            s = s[:max_nt]
        # trim right to multiple of k
        if len(s) % self.k != 0:
            s = s[: (len(s) // self.k) * self.k]
        return s

    def _get_max_seq_len(self) -> int:
        """Get maximum sequence length in nucleotides based on model's max token length."""
        # Reserve 2 tokens for special tokens (CLS, SEP/EOS)
        max_tokens = self.max_length - 2
        return max_tokens * self.k

    def _encode_one(self, seq: str) -> Dict[str, torch.Tensor]:
        # Truncate to max sequence length to avoid position embedding overflow
        max_nt = self._get_max_seq_len()
        s = self._normalize_seq(seq, max_nt=max_nt)
        enc = self.tokenizer(s, return_tensors="pt", add_special_tokens=True, padding=False, truncation=True, max_length=self.max_length)
        return {k: v.to(self.device) for k, v in enc.items()}

    def _encode_many(self, seqs: List[str], *, max_nt: Optional[int] = None) -> Dict[str, torch.Tensor]:
        # Use model's max if not specified
        if max_nt is None:
            max_nt = self._get_max_seq_len()
        ss = [self._normalize_seq(s, max_nt=max_nt) for s in seqs]
        enc = self.tokenizer(
            ss, return_tensors="pt", add_special_tokens=True,
            padding="longest", truncation=True, max_length=self.max_length
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _tokens_from_seq(self, seq: str) -> List[int]:
        s = self._normalize_seq(seq)
        ids = self.tokenizer(s, add_special_tokens=False)["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(map(int, ids))

    @torch.no_grad()
    def _masked_token_logits(self, token_ids_no_specials: List[int], token_idx: int) -> torch.Tensor:
        """Mask token_idx (k-mer), run forward, return logits (V,) on CPU."""
        if not (0 <= token_idx < len(token_ids_no_specials)):
            raise IndexError("token_idx out of range.")
        ids = list(token_ids_no_specials)
        ids[token_idx] = self.mask_id
        inp = self.tokenizer.build_inputs_with_special_tokens(ids)
        attn = [1] * len(inp)
        batch = {
            "input_ids": torch.tensor([inp], device=self.device),
            "attention_mask": torch.tensor([attn], device=self.device),
        }
        logits = self.mlm(**batch).logits  # (1,L,V)
        mpos = (batch["input_ids"][0] == self.mask_id).nonzero(as_tuple=True)[0]
        if mpos.numel() != 1:
            raise RuntimeError("Expected exactly one [MASK].")
        vec = logits[0, int(mpos.item()), :]  # (V,)
        return vec.detach().cpu()

    def _pattern_filter(
        self,
        logits_vec: Union[torch.Tensor, np.ndarray],  # (V,)
        pattern: str,  # len=k; 'N' wildcard
        *,
        masked_idx_in_token: int,
    ) -> Dict[str, float]:
        """Aggregate vocab prob mass over tokens matching pattern; bucket by base at masked index."""
        if len(pattern) != self.k:
            raise ValueError(f"Pattern length must be k={self.k}.")
        logits = torch.as_tensor(logits_vec)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        base_mass = {b: 0.0 for b in BASES}

        def clean(s: str) -> str:
            return re.sub(r"[^ACGTN]", "", (s or "").upper())

        def match(tok: str, pat: str) -> bool:
            if len(tok) != len(pat):
                return False
            for t, p in zip(tok, pat):
                if p != "N" and t != p:
                    return False
            return True

        for tok_str, tok_id in self._vocab.items():
            s = self._id2dna(int(tok_id), tok_str, clean)
            if len(s) != self.k or not match(s, pattern):
                continue
            b = s[masked_idx_in_token]
            if b in base_mass:
                base_mass[b] += float(probs[int(tok_id)])

        total = sum(base_mass.values())
        if total <= 0:
            return {b: 0.25 for b in BASES}
        return {b: v / total for b, v in base_mass.items()}

    def _infer_k_nonoverlap(self) -> int:
        test = "A" * 120
        ids = self.tokenizer(test, add_special_tokens=False)["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        for t in ids[:32]:
            s = self.tokenizer.convert_ids_to_tokens(int(t)) or ""
            s = re.sub(r"[^ACGTN]", "", s.upper())
            if s and set(s).issubset(DNA_SET):
                return len(s)
        raise RuntimeError("Could not infer non-overlap k for this tokenizer.")

    def _id2dna(self, tok_id: int, tok_str: Optional[str], cleaner) -> str:
        if tok_id in self._id2dna_cache:
            return self._id2dna_cache[tok_id]
        s = tok_str if tok_str is not None else self.tokenizer.convert_ids_to_tokens(int(tok_id)) or ""
        s = cleaner(s)
        self._id2dna_cache[tok_id] = s
        return s

    def _get_hidden_states(self, enc: dict, layer: Optional[int] = None) -> torch.Tensor:
        """
        Return hidden states from the backbone.

        Parameters
        ----------
        layer : int, optional
            Which layer to extract. If None, returns last hidden state.
            Use negative indices for layers from the end (-1 = last, -2 = second to last).
        """
        need_all_hidden = layer is not None

        # Build attention masks for encoder
        attention_mask = enc["attention_mask"]
        encoder_attention_mask = attention_mask

        # Common HF convention
        if hasattr(self.mlm, "base_model"):
            out = self.mlm.base_model(
                **enc,
                encoder_attention_mask=encoder_attention_mask,
                output_hidden_states=need_all_hidden
            )
            if layer is None:
                return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            hs = out.hidden_states
            return hs[layer]

        # InstaDeep NT models often expose an 'esm' backbone
        if hasattr(self.mlm, "esm"):
            out = self.mlm.esm(
                **enc,
                encoder_attention_mask=encoder_attention_mask,
                output_hidden_states=need_all_hidden
            )
            if layer is None:
                return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            hs = out.hidden_states
            return hs[layer]

        # Fallback: ask MLM to return hidden states
        out = self.mlm(
            **enc,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True
        )
        hs = out.hidden_states  # tuple(layer0..last)
        if layer is None:
            return hs[-1]
        return hs[layer]
