# rinalmo_wrapper.py
# Standardized API: embed(), predict_nucleotides()
# Legacy methods: predict_masked(), predict_masked_single()

from __future__ import annotations
from typing import List, Sequence, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import re
from rinalmo.pretrained import get_pretrained_model

# Support both package import and direct sys.path import
try:
    from .base_wrapper import BaseWrapper
except ImportError:
    from base_wrapper import BaseWrapper



class RiNALMoWrapper(BaseWrapper):
    """
    RiNALMo wrapper that matches the NTWrapper interface/behavior.

    Public API (identical to NTWrapper):
      - embed(seq_or_list, pool={'mean','cls','tokens'}, return_numpy=True)
      - predict_nucleotides(seq, positions, return_dict=True)

    Notes:
      * RiNALMo is character-level -> k = 1.
      * If the model vocab is RNA-only (U but no T), we map T->U at tokenize time,
        and expose 'T' back to the caller if dna_output=True.
      * If a mask id is available -> masked mode; else -> no-mask (read logits at site).
    """
    def __init__(
        self,
        model_name: str = "giga-v1",
        *,
        device: Optional[str] = None,
        autocast: bool = True,
        dna_output: bool = True,    # expose 'T' in outputs when model uses 'U'
    ):
        super().__init__()
        model, alphabet = get_pretrained_model(model_name=model_name)
        self.model = model.eval()
        self.alphabet = alphabet

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device)
        self.dtype = torch.float32
        self.autocast = bool(autocast) and (self.device.type == "cuda")

        # k-mer compatibility with NTWrapper
        self.k = 1

        # vocab helpers (RiNALMo provides an alphabet)
        self.tok_to_id: Dict[str,int] = getattr(self.alphabet, "tok_to_idx", {})
        self.id_to_tok: Dict[int,str] = getattr(self.alphabet, "idx_to_tok", {})

        self.pad_id = self._discover_pad_id()
        self.mask_id = self._discover_mask_id(allow_none=True)

        # discover base slot + ids
        self.base_ids, self.base_slot = self._discover_base_ids_and_slot()

        # RNA-only?
        self._uses_U_only = ("U" in self.base_ids) and ("T" not in self.base_ids)
        self.dna_output = bool(dna_output)

    # ---------------- Standardized API ----------------

    @torch.no_grad()
    def embed(
        self,
        seqs: Union[str, Sequence[str]],
        *,
        pool: str = "mean",        # 'mean'|'cls'|'tokens'
        return_numpy: bool = True,
        layer: Optional[int] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate embeddings for sequences.

        Parameters
        ----------
        layer : int, optional
            Which layer to extract. If None, uses last hidden state.
            Note: RiNALMo may not expose all intermediate layers.
        """
        is_batch = isinstance(seqs, (list, tuple))
        if not is_batch:
            out = self._embed_batch([str(seqs)], pool=pool, layer=layer)   # (1,H) or (1,T,H)
            out0 = out[0]
            return out0.detach().cpu().numpy() if return_numpy else out0
        out = self._embed_batch(list(seqs), pool=pool, layer=layer)
        return out.detach().cpu().numpy() if return_numpy else out

    @torch.no_grad()
    def predict_nucleotides(
        self,
        seq: str,
        positions: Optional[List[int]] = None,   # <-- changed
        *,
        return_dict: bool = True,
    ) -> Union[List[Dict[str,float]], np.ndarray]:
        # --- NT-style auto-detect: use 'N' sites when positions not given ---
        if not positions:
            positions = self.find_N_positions(seq)
        if not positions:
            raise ValueError("No positions provided and no 'N' bases found in seq.")

        # reuse the batch API (NTWrapper-like flow)
        results_nested = self._predict_masked_like([seq], [positions], return_dict=return_dict)
        res = results_nested[0]
        if return_dict:
            return res
        return np.stack([r.cpu().numpy() if isinstance(r, torch.Tensor) else np.asarray(r) for r in res], axis=0)
        # ---------------- Internals: batching ----------------

    @torch.no_grad()
    def _embed_batch(self, seqs: Sequence[str], pool: str = "mean", layer: Optional[int] = None) -> torch.Tensor:
        seqs_norm = self._normalize_batch_for_tokenizer(seqs)
        tokens = torch.tensor(self.alphabet.batch_tokenize(seqs_norm), dtype=torch.long, device=self.device)  # (B,T)

        with torch.cuda.amp.autocast(enabled=self.autocast):
            out = self.model(tokens)

        reps = self._extract_representations(out, layer=layer)  # (B,T,H)

        if pool == "tokens":
            return reps

        if pool == "cls":
            # robust: pick first base token (no explicit [CLS] guaranteed)
            first_idx = self._first_base_token_indices(tokens)
            return torch.stack([reps[i, first_idx[i]] for i in range(reps.size(0))], dim=0)

        if pool == "mean":
            if self.pad_id is not None:
                mask = (tokens != self.pad_id).float().unsqueeze(-1)  # (B,T,1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                return (reps * mask).sum(dim=1) / denom
            return reps.mean(dim=1)

        raise ValueError("pool must be one of {'mean','cls','tokens'}")

    @torch.no_grad()
    def _predict_masked_like(
        self,
        seqs: Sequence[str],
        positions: Sequence[Sequence[int]],
        return_dict: bool = True,
    ) -> List[List[Union[Dict[str,float], torch.Tensor]]]:
        """
        NT-style masked path (k=1) or no-mask fallback. Returns list per sequence.
        """
        assert len(seqs) == len(positions), "len(seqs) must match len(positions)"

        seqs_norm = self._normalize_batch_for_tokenizer(seqs)
        batch_tok = torch.tensor(self.alphabet.batch_tokenize(seqs_norm), dtype=torch.long, device=self.device)  # (B,T)
        B, T = batch_tok.shape
        first_base_idx = self._first_base_token_indices(batch_tok)

        # model-side 4 bases order
        model_four = ["A","C","G", ("U" if self._uses_U_only else "T")]
        model_base_ids = torch.tensor([self.base_ids[b] for b in model_four], dtype=torch.long, device=self.device)

        # expose to caller
        expose_four = ["A","C","G", ("T" if self.dna_output else ("U" if "U" in self.base_ids else "T"))]

        out: List[List[Union[Dict[str,float], torch.Tensor]]] = [[] for _ in seqs]

        # No-mask path
        if self.mask_id is None:
            with torch.cuda.amp.autocast(enabled=self.autocast):
                model_out = self.model(batch_tok)
                logits = self._extract_logits(model_out)  # (B,T,V)

            for i, pos_list in enumerate(positions):
                for p in pos_list:
                    if not (0 <= p < len(seqs[i])):
                        continue
                    tok_idx = first_base_idx[i] + p  # k=1
                    if not (0 <= tok_idx < T):
                        probs = torch.full((4,), float("nan"), device=logits.device)
                    else:
                        row = logits[i, tok_idx].index_select(0, model_base_ids)
                        probs = F.softmax(row, dim=0)
                    if return_dict:
                        out[i].append({expose_four[k]: float(probs[k].item()) for k in range(4)})
                    else:
                        out[i].append(probs.detach().cpu())
            return out

        # Masked path: create one masked row per (seq, pos)
        masked_rows = []
        owners: List[Tuple[int,int]] = []  # (seq_idx, j_idx)
        for i, pos_list in enumerate(positions):
            for j, p in enumerate(pos_list):
                if not (0 <= p < len(seqs[i])):
                    continue
                tok_idx = first_base_idx[i] + p
                if not (0 <= tok_idx < T):
                    continue
                m = batch_tok[i].clone()
                m[tok_idx] = self.mask_id
                masked_rows.append(m)
                owners.append((i, j))

        if not masked_rows:
            return out

        masked_inputs = torch.stack(masked_rows, dim=0).to(self.device)  # (N,T)
        with torch.cuda.amp.autocast(enabled=self.autocast):
            mo = self.model(masked_inputs)
            logits = self._extract_logits(mo)  # (N,T,V)

        for r, (seq_idx, j) in enumerate(owners):
            p = positions[seq_idx][j]
            tok_idx = first_base_idx[seq_idx] + p
            row = logits[r, tok_idx].index_select(0, model_base_ids)
            probs = F.softmax(row, dim=0)
            if return_dict:
                out[seq_idx].append({expose_four[k]: float(probs[k].item()) for k in range(4)})
            else:
                out[seq_idx].append(probs.detach().cpu())
        return out

    # ---------------- Utilities / discovery ----------------

    def _normalize_batch_for_tokenizer(self, seqs: Sequence[str]) -> List[str]:
        # allow only A/C/G/T/U/N; if model is RNA-only, map T->U
        out = []
        for s in seqs:
            s = re.sub(r"[^ACGTUNacgtun]", "N", s or "")
            if self._uses_U_only:
                s = s.replace("T","U").replace("t","u")
            out.append(s.upper())
        return out

    def _extract_representations(self, model_out, layer: Optional[int] = None) -> torch.Tensor:
        """
        Extract representations from RiNALMo output.

        Parameters
        ----------
        layer : int, optional
            Which layer to extract. If None, uses last/default representation.
        """
        # Try to get hidden_states tuple first if layer is specified
        if layer is not None:
            hs = None
            if isinstance(model_out, dict):
                hs = model_out.get("hidden_states")
            else:
                hs = getattr(model_out, "hidden_states", None)
            if hs is not None:
                if isinstance(hs, (tuple, list)):
                    return hs[layer]
                elif hs.dim() == 4:  # (num_layers, B, T, H)
                    return hs[layer]

        # RiNALMo usually returns {'representation': (B,T,H)}; be permissive
        if isinstance(model_out, dict) and "representation" in model_out:
            return model_out["representation"]
        rep = getattr(model_out, "representation", None)
        if rep is not None:
            return rep
        if isinstance(model_out, dict):
            for k in ("hidden_states","last_hidden_state"):
                v = model_out.get(k)
                if v is not None:
                    return v if v.dim()==3 else v[-1]
        for k in ("hidden_states","last_hidden_state"):
            v = getattr(model_out, k, None)
            if v is not None:
                return v if v.dim()==3 else v[-1]
        raise RuntimeError("Token representations not found in RiNALMo output.")

    def _extract_logits(self, model_out) -> torch.Tensor:
        if isinstance(model_out, dict):
            for k in ("logits","lm_logits"):
                v = model_out.get(k)
                if v is not None:
                    return v
            # derive from representation if head exists
            rep = model_out.get("representation")
            head = getattr(self.model, "lm_head", None)
            if rep is not None and head is not None:
                return head(rep)
        logits = getattr(model_out, "logits", None)
        if logits is not None:
            return logits
        rep = getattr(model_out, "representation", None)
        head = getattr(self.model, "lm_head", None)
        if rep is not None and head is not None:
            return head(rep)
        # alternate head names
        for name in ("mlm_head","language_modeling_head"):
            head = getattr(self.model, name, None)
            if head is not None:
                rep = rep if rep is not None else self._extract_representations(model_out)
                return head(rep)
        raise RuntimeError("Could not extract RiNALMo logits; LM head not exposed?")

    def _discover_pad_id(self) -> Optional[int]:
        for attr in ("PAD_ID","pad_id","pad"):
            v = getattr(self.alphabet, attr, None)
            if isinstance(v, int):
                return v
        for tok in ("<pad>","[PAD]","<PAD>"):
            if tok in self.tok_to_id:
                return self.tok_to_id[tok]
        return None

    def _discover_mask_id(self, allow_none: bool=False) -> Optional[int]:
        for attr in ("MASK_ID","mask_id","mask"):
            v = getattr(self.alphabet, attr, None)
            if isinstance(v, int):
                return v
        for tok in ("<mask>","[MASK]","<MASK>","▁<mask>","MASK","<msk>","[MSK]"):
            if tok in self.tok_to_id:
                return self.tok_to_id[tok]
        for k in self.tok_to_id:
            if "mask" in k.lower():
                return self.tok_to_id[k]
        if allow_none:
            return None
        raise RuntimeError("No [MASK] id found for RiNALMo.")

    def _discover_base_ids_and_slot(self) -> Tuple[Dict[str,int], int]:
        # identify which token position corresponds to the base char
        probes = ["AAAA", "CAAA", "GAAA", "UAAA", "TAAA"]
        tok = torch.tensor(self.alphabet.batch_tokenize(probes), dtype=torch.long, device=self.device)  # (5,T)
        a_ids, c_ids, g_ids, u_ids, t_ids = [tok[i].tolist() for i in range(tok.size(0))]
        base_slot = None
        for j, (a, c) in enumerate(zip(a_ids, c_ids)):
            if a != c:
                base_slot = j
                break
        if base_slot is None:
            raise RuntimeError("Could not locate base slot for RiNALMo tokenizer.")
        idA, idC, idG, idU, idT = a_ids[base_slot], c_ids[base_slot], g_ids[base_slot], u_ids[base_slot], t_ids[base_slot]
        base_ids: Dict[str,int] = {"A": idA, "C": idC, "G": idG}
        if idU != idA: base_ids["U"] = idU
        if idT != idA: base_ids["T"] = idT
        if ("U" not in base_ids) and ("T" not in base_ids):
            raise RuntimeError("Neither U nor T discovered in RiNALMo vocab.")
        return base_ids, base_slot

    def _first_base_token_indices(self, batch_tok: torch.Tensor) -> List[int]:
        base_set = set(self.base_ids.values())
        firsts = []
        for row in batch_tok.tolist():
            j = 0
            while j < len(row) and row[j] not in base_set:
                j += 1
            firsts.append(j if j < len(row) else 0)
        return firsts



    # =============== NEW: NT-parity helpers ===============
    def find_N_positions(self, seq: str) -> List[int]:
        """Indices where the input has 'N' (case-insensitive)."""
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
        Return (token_idx, rel_in_token, k-mer chunk) after normalizing.
        For RiNALMo k=1, rel_in_token is always 0 and chunk is the single base.
        """
        s = self._normalize_seq(seq)
        if not (0 <= center_base_index < len(s)):
            raise IndexError("center_base_index out of range after normalization.")
        # For RiNALMo we map base index -> token index via first_base_idx
        toks = torch.tensor(self.alphabet.batch_tokenize([self._normalize_batch_for_tokenizer([s])[0]]),
                            dtype=torch.long, device=self.device)
        first = self._first_base_token_indices(toks)[0]
        t_idx = first + center_base_index
        return t_idx, 0, s[center_base_index:center_base_index+1]

    def _normalize_seq(self, seq: str) -> str:
        """Single-sequence cleaner used by the helpers (NT parity)."""
        s = re.sub(r"[^ACGTUNacgtun]", "N", (seq or ""))
        return s.upper()
    # =============== /helpers ===============