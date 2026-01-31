
# caduceus_wrapper.py
# Standardized API: embed(), predict_nucleotides()
# Legacy methods preserved for backward compatibility:
#   - tokens_from_seq(seq) -> token ids (no specials)
#   - get_masked_position_logits(token_ids_no_specials, i, return_probs=False, temperature=1.0)
#   - pattern_matching_filter(logits_vec, target_pattern_with_N, masked_idx_in_token=None, ...)
#   - center_token_mapping(...): returns k-mer/chunk covering the queried base + indices
#   - build_masked_pattern_and_index(info): returns (pattern_with_N, masked_idx_in_token)

import re
from typing import Dict, Optional, List, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Support both package import and direct sys.path import
try:
    from .base_wrapper import BaseWrapper, PoolMode
except ImportError:
    from base_wrapper import BaseWrapper, PoolMode

BASES = ("A", "C", "G", "T")
DNA_SET = set("ACGTN")



class CaduceusWrapper(BaseWrapper):
    """
    Caduceus wrapper aligned with NTWrapper semantics:
      - embed(seq|list, pool={'mean','cls','tokens'}, return_numpy=True)
      - predict_nucleotides(seq, positions=None|List[int], return_dict=True)
        * if positions is None, auto-detect 'N' positions like NTWrapper
      - Supports long inputs (no truncation); avoids passing attention_mask
    """

    MODEL_ID = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
    TRUST_REMOTE = True

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        trust_remote_code: bool = True,
        attn_impl: str = "eager",
    ):
        super().__init__()

        # device
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            )
        self.device = torch.device(device)
        self.dtype = dtype

        # tokenizer / model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_impl,
        ).to(self.device).eval()

        # backbone for hidden states (HF convention varies)
        self.backbone = getattr(self.model, "base_model", None)
        if self.backbone is None:
            self.backbone = getattr(self.model, "model", self.model)

        self.k = self._infer_k_nonoverlap()
        self.mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if self.mask_id is None:
            raise ValueError("Tokenizer has no [MASK]; MLM probabilities require it.")

        self._id2dna_cache: Dict[int, str] = {}
        self._vocab = self.tokenizer.get_vocab()

    # -------------------- Public API --------------------

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
        Single string:
          - 'tokens' -> (L,H)
          - 'cls'    -> (H,)
          - 'mean'   -> (H,)
        List[str]:
          - 'tokens' -> list[(Li,H)]
          - 'cls'/'mean' -> (B,H)

        Parameters
        ----------
        layer : int, optional
            Which transformer layer to extract (0 = embedding, -1 = last).
            If None, uses last hidden state.
        """
        is_batch = isinstance(seq, (list, tuple))

        if not is_batch:
            enc = self._encode_one(str(seq))
            out = self._last_hidden(enc, layer=layer)  # (1,L,H)

            if pool == "tokens":
                emb = out[0]                               # (L,H)
            elif pool == "cls":
                emb = out[:, 0][0]                         # (H,)
            elif pool == "mean":
                # No padding used => attention_mask = ones
                mask = torch.ones(out.shape[1], 1, device=out.device).unsqueeze(0)
                denom = mask.sum(dim=1).clamp(min=1)
                emb = ((out * mask).sum(dim=1) / denom)[0] # (H,)
            else:
                raise ValueError("pool must be one of {'mean','cls','tokens'}")

            return emb.detach().cpu().numpy() if return_numpy else emb.detach().cpu()

        # batched
        enc = self._encode_many(list(seq))
        out = self._last_hidden(enc, layer=layer)  # (B,L,H)

        if pool == "tokens":
            ids = enc["input_ids"]
            attn = torch.ones_like(ids, device=ids.device)  # we padded to longest below
            pieces: List[Union[np.ndarray, torch.Tensor]] = []
            for b in range(out.size(0)):
                # recover true length by trimming right-padding id 0 if present
                # use tokenizer.pad_token_id when available; else rely on attention to slice
                if self.tokenizer.pad_token_id is not None:
                    pad_id = self.tokenizer.pad_token_id
                    row = ids[b].tolist()
                    Lb = len(row) - row[::-1].index(next((x for x in row[::-1] if x != pad_id), row[-1])) \
                        if any(x == pad_id for x in row) else len(row)
                else:
                    Lb = len(ids[b])
                piece = out[b, :Lb]
                pieces.append(piece.detach().cpu().numpy() if return_numpy else piece.detach().cpu())
            return pieces

        if pool == "cls":
            emb = out[:, 0]  # (B,H)
        elif pool == "mean":
            # build attention mask (1 for non-pad) from pad_token_id
            ids = enc["input_ids"]
            if self.tokenizer.pad_token_id is not None:
                mask = (ids != self.tokenizer.pad_token_id).to(out.dtype).unsqueeze(-1)  # (B,L,1)
            else:
                mask = torch.ones(out.shape[:2], device=out.device, dtype=out.dtype).unsqueeze(-1)
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
        If positions is None -> auto-detect 'N' bases (mirrors NTWrapper).
        For k=1 (char-level), map directly to A/C/G/T token probs.
        For k>1, aggregate prob mass across k-mers matching pattern with 'N' at rel index.
        """
        s = self._normalize_seq(seq)
        L = len(s)

        if not positions:
            positions = [i for i, c in enumerate(s) if c == "N"]
        if not positions:
            raise ValueError("No positions provided and no 'N' bases found in seq.")

        token_ids = self.tokens_from_seq(s)

        def one_pos(pos: int) -> Dict[str, float]:
            if not (0 <= pos < L):
                raise IndexError(f"Position {pos} out of range for sequence length {L}")
            t_idx, rel = pos // self.k, pos % self.k
            if t_idx >= len(token_ids):
                raise IndexError(f"Token index {t_idx} out of range")
            logits = self.get_masked_position_logits(token_ids, t_idx, return_probs=False)

            if self.k == 1:
                vocab = self._vocab
                base_ids = {b: vocab.get(b, vocab.get(b.lower(), -1)) for b in BASES}
                probs = torch.softmax(logits, dim=0)
                d = {b: float(probs[tid].item()) if (tid is not None and tid >= 0 and tid < probs.numel()) else 0.25
                     for b, tid in base_ids.items()}
                tot = sum(d.values())
                return {b: (v / tot) if tot > 0 else 0.25 for b, v in d.items()}
            else:
                pattern = "N" * self.k
                pattern = pattern[:rel] + "N" + pattern[rel + 1 :]
                return self.pattern_matching_filter(logits, pattern, masked_idx_in_token=rel)

        if return_dict:
            return [one_pos(p) for p in positions]
        arr = np.zeros((len(positions), 4), dtype=np.float32)
        for i, p in enumerate(positions):
            d = one_pos(p)
            arr[i] = [d["A"], d["C"], d["G"], d["T"]]
        return arr

    # -------------------- Legacy helpers --------------------

    def tokens_from_seq(self, seq: str) -> List[int]:
        ids = self.tokenizer(seq, add_special_tokens=False)["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(map(int, ids))

    @torch.no_grad()
    def get_masked_position_logits(
        self,
        token_ids_no_specials: List[int],
        token_position_of_interest: int,
        *,
        return_probs: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if not (0 <= token_position_of_interest < len(token_ids_no_specials)):
            raise IndexError("token_position_of_interest out of range.")
        ids = list(token_ids_no_specials)
        ids[token_position_of_interest] = self.mask_id
        input_ids = self.tokenizer.build_inputs_with_special_tokens(ids)

        try:
            mask_index = input_ids.index(self.mask_id)
        except ValueError:
            raise RuntimeError("Failed to find [MASK] after adding special tokens.")

        batch = {"input_ids": torch.tensor([input_ids], device=self.device)}
        vec = self.model(**batch).logits[0, mask_index, :]  # (V,)
        if return_probs:
            vec = torch.softmax(vec / float(max(1e-6, temperature)), dim=-1)
        return vec.detach().cpu()

    def pattern_matching_filter(
        self,
        logits_vec: Union[torch.Tensor, np.ndarray],
        target_pattern_with_N: str,
        *,
        masked_idx_in_token: Optional[int] = None,
        temperature: float = 1.0,
        return_debug: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], List[Tuple[str, float]]]]:
        pat = target_pattern_with_N.upper()
        if len(pat) != self.k:
            raise ValueError(f"target_pattern length must equal k={self.k}.")
        if masked_idx_in_token is None:
            n_count = pat.count("N")
            if n_count == 1:
                masked_idx_in_token = pat.index("N")
            else:
                raise ValueError(
                    "masked_idx_in_token must be provided when pattern contains "
                    f"{n_count} 'N' positions."
                )
        if not (0 <= masked_idx_in_token < self.k):
            raise ValueError("masked_idx_in_token out of range.")

        logits = torch.as_tensor(logits_vec)
        probs = torch.softmax(logits / float(temperature), dim=-1).detach().cpu().numpy()

        buckets = {b: 0.0 for b in BASES}
        debug_pairs: List[Tuple[str, float]] = []

        def clean(s: str) -> str:
            return re.sub(r"[^ACGTN]", "", (s or "").upper())

        def match(tok: str, pattern: str) -> bool:
            if len(tok) != len(pattern):
                return False
            for t, p in zip(tok, pattern):
                if p != "N" and t != p:
                    return False
            return True

        for tok_str, tok_id in self._vocab.items():
            s = self._id2dna(int(tok_id), tok_str, clean)
            if len(s) != self.k or not match(s, pat):
                continue
            b = s[masked_idx_in_token]
            if b in buckets:
                p = float(probs[int(tok_id)])
                buckets[b] += p
                if return_debug:
                    debug_pairs.append((s, p))

        total = sum(buckets.values())
        if total <= 0:
            res = {b: 0.25 for b in BASES}
            return (res, []) if return_debug else res

        res = {b: v / total for b, v in buckets.items()}
        if return_debug:
            debug_pairs.sort(key=lambda x: x[1], reverse=True)
            return res, debug_pairs
        return res

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
        """Mirror NTWrapper: construct window, make divisible by k, expose token + chunk."""
        from seqmat import SeqMat

        seq = SeqMat.from_fasta(fasta_name, chrom, query_pos - L + 1, query_pos + L + 1)
        if mutations:
            seq.apply_mutations(mutations, permissive_ref=True)
        if rc:
            seq = seq.reverse_complement()

        seq_ref = seq.clone()
        seq_ref.apply_mutations([(query_pos, query_ref, "N")], permissive_ref=True)

        seq_s = seq.seq
        seq_ref_s = seq_ref.seq

        # trim right to multiple of k
        if len(seq_s) % self.k != 0:
            seq_s = seq_s[: (len(seq_s) // self.k) * self.k]
            seq_ref_s = seq_ref_s[: len(seq_s)]

        center_nt = L
        token_pos = center_nt // self.k
        rel_in_token = center_nt % self.k

        enc = self.tokenizer(seq_s, add_special_tokens=False)
        token_ids = enc["input_ids"]
        tokens = [self.tokenizer.convert_ids_to_tokens(i) for i in token_ids]

        expected_chunk = seq_ref_s[token_pos * self.k : (token_pos + 1) * self.k]
        token_clean = re.sub(r"[^ACGTN]", "", tokens[token_pos])

        return {
            "seq": seq_s,
            "seq_ref": seq_ref_s,
            "k": self.k,
            "token_ids": token_ids,
            "tokens": tokens,
            "token_position_of_interest": token_pos,
            "token_string": tokens[token_pos],
            "corresponding_seq_chunk": expected_chunk,
            "matches": token_clean == expected_chunk,
            "relative_index_of_mutation_in_token": rel_in_token,
        }

    def build_masked_pattern_and_index(self, info: Dict[str, Union[str, int]]) -> Tuple[str, int]:
        chunk = str(info["corresponding_seq_chunk"])
        rel = int(info["relative_index_of_mutation_in_token"])
        if chunk[rel] != "N":
            chunk = f"{chunk[:rel]}N{chunk[rel+1:]}"
        return chunk, rel

    def parameters(self):
        return self.model.parameters()

    # -------------------- Internals --------------------

    def _encode_one(self, seq: str) -> Dict[str, torch.Tensor]:
        s = self._normalize_seq(seq)
        enc = self.tokenizer(s, return_tensors="pt", add_special_tokens=True, padding=False, truncation=False)
        return {k: v.to(self.device) for k, v in enc.items()}

    def _encode_many(self, seqs: List[str], *, max_nt: Optional[int] = None) -> Dict[str, torch.Tensor]:
        ss = [self._normalize_seq(s, max_nt=max_nt) for s in seqs]
        enc = self.tokenizer(ss, return_tensors="pt", add_special_tokens=True, padding="longest", truncation=False)
        return {k: v.to(self.device) for k, v in enc.items()}

    def _last_hidden(self, enc: Dict[str, torch.Tensor], layer: Optional[int] = None) -> torch.Tensor:
        """
        Return hidden states (B,L,H) from backbone.

        Parameters
        ----------
        layer : int, optional
            Which layer to extract. If None, returns last hidden state.
        """
        inputs = {"input_ids": enc["input_ids"]}
        out = self.backbone(**inputs, output_hidden_states=True, return_dict=True)

        if layer is None:
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                return out.last_hidden_state
            return out.hidden_states[-1]

        return out.hidden_states[layer]

    def _normalize_seq(self, seq: str, *, max_nt: Optional[int] = None) -> str:
        s = re.sub(r"[^ACGTN]", "N", (seq or "").upper())
        if max_nt is not None and len(s) > max_nt:
            s = s[:max_nt]
        if len(s) % self.k != 0:
            s = s[: (len(s) // self.k) * self.k]
        return s

    def _infer_k_nonoverlap(self) -> int:
        test = "A" * 256
        ids = self.tokenizer(test, add_special_tokens=False)["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        for t in ids[:64]:
            tok = self.tokenizer.convert_ids_to_tokens(int(t)) or ""
            s = re.sub(r"[^ACGTN]", "", tok.upper())
            if s and set(s).issubset(DNA_SET):
                return len(s)
        return 1  # fallback: char-level

    def _id2dna(self, tok_id: int, tok_str: Optional[str], cleaner) -> str:
        if tok_id in self._id2dna_cache:
            return self._id2dna_cache[tok_id]
        s = tok_str if tok_str is not None else self.tokenizer.convert_ids_to_tokens(int(tok_id)) or ""
        s = cleaner(s)
        self._id2dna_cache[tok_id] = s
        return s
