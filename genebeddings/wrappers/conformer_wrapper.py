import re
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import torch
import sys
import os
import torch.nn as nn

# Import SeqMat for sequence manipulation
try:
    from ..seqmat import SeqMat
except ImportError:
    try:
        from seqmat import SeqMat
    except ImportError:
        # Fallback if SeqMat is not available
        SeqMat = None

# Add local assets to path for conformer dependencies
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'conformer')
if _ASSETS_DIR not in sys.path:
    sys.path.insert(0, _ASSETS_DIR)

from anchor_tokenizer import GenomicBPE
from conformer import ConformerGenomicLM

# Default asset paths
_DEFAULT_TOKENIZER_PATH = os.path.join(_ASSETS_DIR, 'grover.tkz')
_DEFAULT_CHECKPOINT_PATH = os.path.join(_ASSETS_DIR, 'ep005.pt')

try:
    from .base_wrapper import BaseWrapper, PoolMode
except ImportError:
    from base_wrapper import BaseWrapper, PoolMode

BASES = ["A", "C", "G", "T"]  # if not already defined



class AnchoredNTLike(BaseWrapper):
    """
    NT-like wrapper over ConformerGenomicLM + GenomicBPE that supports anchoring.
    Implements the BaseWrapper API:
      - embed()
      - predict_nucleotides()
    """

    def __init__(
        self,
        *,
        tokenizer_path: str = None,       # path to .tkz (defaults to local asset)
        checkpoint_path: str = None,      # path to model checkpoint (defaults to local asset)
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        tie_weights: bool = True,
        embed_dim: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
        mlp_dim: int = 768,
        local_window: int = 256,
        dropout: float = 0.1,
        rope_base: float = 50_000.0,
        use_dataparallel: bool = True,
        device_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        self.dtype = dtype

        # Use default asset paths if not provided
        if tokenizer_path is None:
            tokenizer_path = _DEFAULT_TOKENIZER_PATH
        if checkpoint_path is None:
            checkpoint_path = _DEFAULT_CHECKPOINT_PATH

        # --- tokenizer ---
        self.tokenizer = GenomicBPE.load(tokenizer_path)

        # device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)


        # --- model ---
        self.model = ConformerGenomicLM(
            input_vocab_size=len(self.tokenizer.vocab) + 4,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            local_window=local_window,
            dropout=dropout,
            rope_base=rope_base,
            pad_id=self.tokenizer.PAD_ID,
            tie_weights=tie_weights,
        ).to(self.device).to(dtype).eval()

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=True)

        if use_dataparallel and torch.cuda.device_count() > 1:
            if device_ids is None:
                device_ids = list(range(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        # Non-overlapping, base-anchored: treat as k=1 for compatibility
        self.k = 1
        # mask id for MLM
        self.mask_id = self.tokenizer.MASK_ID

        # Map base token IDs to fixed A/C/G/T order
        base_tok_ids_in_head_order = list(self.tokenizer.base_alphabet_ids())
        self._tokid2headidx = {tok_id: i for i, tok_id in enumerate(base_tok_ids_in_head_order)}
        self._base_tokid_by_letter = {b: self.tokenizer.token_to_id(b) for b in BASES}

    # ---------- helpers ----------

    def _normalize_seq(self, seq: str) -> str:
        """Clean to A/C/G/T/N, uppercase."""
        return re.sub(r"[^ACGTN]", "N", (seq or "").upper())

    def find_N_positions(self, seq: str) -> List[int]:
        """Indices where the input has 'N'."""
        return [i for i, c in enumerate(seq) if c.upper() == "N"]

    # ---------- Public API (genomic / NT-ish) ----------

    def center_token_mapping(
        self,
        *,
        chrom: str,
        query_pos: int,
        query_ref: str,
        rc: bool,
        L: int,
        fasta_name: str = "hg38",
        mutations: Optional[List[Tuple[int, str, str]]] = None,
    ) -> Dict[str, Union[str, int, bool, List[int], List[str]]]:
        """
        Tokenize a [+/- L] window around query_pos with anchors at the mutated base.
        Return a dict compatible with the NT code-path.
        """
        # Extract sequence window
        seq = SeqMat.from_fasta(fasta_name, chrom, query_pos - L + 1, query_pos + L + 1)
        if mutations:
            seq.apply_mutations(mutations, permissive_ref=True)

        if rc:
            seq = seq.reverse_complement()

        # For "reference chunk" semantics as in NT: set that base to 'N' in char-space
        seq_ref = seq.clone()
        seq_ref.apply_mutations([(query_pos, query_ref, "N")], permissive_ref=True)

        seq_s = seq.seq
        seq_ref_s = seq_ref.seq

        # Base-level anchoring (k=1) → center nt index is L
        center_nt = L

        # Tokenize with anchors; add_special_tokens=False for clean mapping
        tok = self.tokenizer.tokenize_with_anchors(
            seq_s, keep_base_mask=seq.mutation_vector, add_special_tokens=False
        )
        token_ids = tok["input_ids"]                           # (T,)
        tokens = [self.tokenizer.id_to_token(i) for i in token_ids]
        char_to_token = tok["char_to_token"]                  # list[int|None]
        token_pos = char_to_token[center_nt]
        if token_pos is None:
            token_pos = center_nt
        token_pos = int(token_pos)
        rel_in_token = 0  # k=1

        expected_seq_chunk = seq_ref_s[center_nt:center_nt + self.k]
        token_string = tokens[token_pos] if token_pos < len(tokens) else ""
        token_string_clean = re.sub(r"[^ACGTN]", "", token_string)

        return {
            "seq": seq_s,
            "seq_ref": seq_ref_s,
            "k": self.k,
            "token_ids": token_ids,
            "tokens": tokens,
            "token_position_of_interest": token_pos,
            "token_string": token_string,
            "corresponding_seq_chunk": expected_seq_chunk,    # length 1
            "matches": token_string_clean == expected_seq_chunk,
            "relative_index_of_mutation_in_token": rel_in_token,
            "_tok": tok,
            "_attention_mask": tok.get("attention_mask", None),
        }

    @torch.no_grad()
    def get_masked_position_logits(
        self,
        token_ids_no_specials: List[int],
        token_position_of_interest: int,
        *,
        attention_mask: Optional[List[int]] = None,
        return_probs: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Mask the token at the position of interest, forward once, and return logits (or probs)
        over the base alphabet in fixed (A,C,G,T) order as a 1D torch.Tensor of length 4.
        """
        device = self.device

        input_ids = torch.tensor(token_ids_no_specials, dtype=torch.long, device=device)
        if attention_mask is None:
            attention_mask = [1] * len(token_ids_no_specials)
        pad_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

        masked_ids = input_ids.clone()
        if not (0 <= token_position_of_interest < masked_ids.numel()):
            raise IndexError("token_position_of_interest out of range.")
        masked_ids[token_position_of_interest] = self.mask_id

        out = self.model(
            input_ids=masked_ids.unsqueeze(0),   # (1,L)
            pad_mask=pad_mask.unsqueeze(0),      # (1,L)
        )
        logits_base = out["mlm_logits_base"][0]       # (L, Bbase)
        logits_at_pos = logits_base[token_position_of_interest]  # (Bbase,)

        base_logits = torch.empty(4, dtype=logits_at_pos.dtype, device=logits_at_pos.device)
        for i, base in enumerate(BASES):
            tok_id = self._base_tokid_by_letter[base]
            head_idx = self._tokid2headidx.get(tok_id, None)
            if head_idx is None or head_idx >= logits_at_pos.numel():
                raise IndexError(
                    f"Base '{base}' token_id {tok_id} not present in base head of size {logits_at_pos.numel()}"
                )
            base_logits[i] = logits_at_pos[head_idx]

        if return_probs:
            vec = torch.softmax(base_logits / float(max(1e-6, temperature)), dim=-1)
        else:
            vec = base_logits
        return vec.detach().cpu()  # (4,)

    def pattern_matching_filter(
        self,
        logits_vec: Union[torch.Tensor, np.ndarray],  # (4,) for A,C,G,T
        target_pattern: str,                          # kept for API parity; ignored
        *,
        temperature: float = 1.0,
        return_debug: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], List[Tuple[str, float]]]]:
        """
        For anchored base logits, pattern filtering is unnecessary.
        We simply (optionally) softmax and map to {A,C,G,T}.
        """
        if isinstance(logits_vec, np.ndarray):
            logits = torch.from_numpy(logits_vec)
        else:
            logits = logits_vec
        if logits.dim() != 1 or logits.shape[0] != 4:
            raise ValueError("Expected a 1D tensor of length 4 for A,C,G,T logits.")

        probs = torch.softmax(logits / float(max(1e-6, temperature)), dim=-1).detach().cpu().numpy()
        result = {b: float(probs[i]) for i, b in enumerate(BASES)}
        if return_debug:
            dbg = list(zip(BASES, probs.tolist()))
            return result, dbg
        return result

    # ======================== BaseWrapper: embed ========================

    @torch.no_grad()
    def embed(
        self,
        seq: Union[str, List[str]],
        *,
        pool: PoolMode = "mean",           # 'mean' | 'cls' | 'tokens'
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray]]:
        """
        If seq is a string -> returns (H,) for mean/cls, or (L,H) for tokens.
        If seq is a list[str] -> returns (B,H) for mean/cls, or list[(Li,H)] for tokens.
        """
        # single sequence path
        if isinstance(seq, str):
            return self._embed_one(seq, pool=pool, return_numpy=return_numpy)

        # batch path
        seqs = list(seq)
        embs = [self._embed_one(s, pool=pool, return_numpy=False) for s in seqs]

        if pool == "tokens":
            # variable lengths → list[(Li,H)]
            if return_numpy:
                return [e.detach().cpu().numpy() for e in embs]
            return [e.detach().cpu() for e in embs]

        stacked = torch.stack(embs, dim=0)  # (B,H)
        return stacked.detach().cpu().numpy() if return_numpy else stacked.detach().cpu()

    @torch.no_grad()
    def _embed_one(
        self,
        seq: str,
        *,
        pool: PoolMode = "mean",
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        seq = self._normalize_seq(seq)

        tok = self.tokenizer.tokenize_with_anchors(
            seq
        )
        # print(tok['input_ids'])

        pad_mask  = tok.get("attention_mask", [1] * len(tok["input_ids"]))
        input_ids = torch.tensor(tok['input_ids'], dtype=torch.long, device=self.device)

        out = self.model(input_ids.unsqueeze(0))
        # print(out)

        hidden = None
        for key in ("hidden_states", "sequence_representations", "last_hidden_state", "hidden"):
            if key in out and out[key] is not None:
                hidden = out[key][0]    # (L, H)
                break

        if hidden is None:
            logits_base = out["mlm_logits_base"][0]  # (L, Bbase)
            token_feats = torch.logsumexp(logits_base, dim=-1, keepdim=True)
            hidden = token_feats  # (L,1)

        if pool == "tokens":
            emb = hidden
        elif pool == "cls":
            emb = hidden[0]
        elif pool == "mean":
            mask = torch.tensor(pad_mask, device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)  # (L,1)
            denom = mask.sum(dim=0).clamp(min=1)
            emb = (hidden * mask).sum(dim=0) / denom        # (H,)
        else:
            raise ValueError("pool must be one of {'mean','cls','tokens'}")

        if return_numpy:
            return emb.detach().cpu().numpy()
        return emb

    # =================== BaseWrapper: predict_nucleotides ===================

    @torch.no_grad()
    def predict_nucleotides(
        self,
        seq: str,
        positions: Optional[List[int]] = None,
        *,
        return_dict: bool = True,
    ) -> Union[List[Dict[str, float]], np.ndarray]:
        """
        NT-like nucleotide prediction using base-anchored MLM head.

        seq: raw DNA string
        positions: 0-based indices in seq, or None to auto-detect 'N' positions
        """
        # Auto-detect 'N' sites if not provided
        if not positions:
            positions = self.find_N_positions(seq)
        if not positions:
            raise ValueError("No positions provided and no 'N' bases found in seq.")

        s = self._normalize_seq(seq)
        L = len(s)
        pos_list = [int(p) for p in positions if 0 <= int(p) < L]

        tok = self.tokenizer.tokenize_with_anchors(
            s, keep_base_mask=None, add_special_tokens=False
        )
        token_ids = tok["input_ids"]  # Keep as list for indexing char_to_token
        char_to_token = tok["char_to_token"]
        attention_mask = tok.get("attention_mask", [1] * len(token_ids))

        out_list: List[Dict[str, float]] = []

        for p in pos_list:
            t_idx = char_to_token[p]
            if t_idx is None or t_idx < 0:
                continue
            t_idx = int(t_idx)

            probs_vec = self.get_masked_position_logits(
                token_ids,
                t_idx,
                attention_mask=attention_mask,
                return_probs=True,
            )  # (4,)
            probs = probs_vec.detach().cpu().numpy()
            d = {b: float(probs[i]) for i, b in enumerate(BASES)}
            out_list.append(d)

        if not return_dict:
            arr = np.zeros((len(out_list), 4), dtype=np.float32)
            for i, d in enumerate(out_list):
                arr[i] = [d["A"], d["C"], d["G"], d["T"]]
            return arr

        return out_list