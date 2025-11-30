
import math
import itertools
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Support both package import and direct sys.path import
try:
    from .base_wrapper import BaseWrapper
except ImportError:
    from base_wrapper import BaseWrapper

NUCS = ("A", "C", "G", "T")


def _is_dna_kmer(s: str, k: int) -> bool:
    return isinstance(s, str) and len(s) == k and set(s).issubset(set("ACGT"))


class SpeciesLMWrapper(BaseWrapper):
    """
    SpeciesLM wrapper with standardized API.

    Standardized API: embed(), predict_nucleotides()
    Legacy method: acgt_probs() (same as predict_nucleotides but returns full array)

    Notes
    -----
    * Tokenization is whitespace-separated 6-mers with stride=1.
    * For A/C/G/T probabilities, we mask k=6 consecutive tokens per nucleotide position,
      run the MLM once per batch slice, softmax over vocab, then fold k-mer token
      probabilities into per-base probabilities via a precomputed filter tensor.
    """

    def __init__(
        self,
        model_id: str = "gagneurlab/SpeciesLM",
        revision: str = "downstream_species_lm",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        kmer_size: int = 6,
    ):
        super().__init__()
        self.k = int(kmer_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_id, revision=revision, torch_dtype=dtype
        )
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            )
        self.device = torch.device(device)
        self.dtype = dtype
        self.model.to(self.device).eval()

        # IDs
        self.mask_id = self.tokenizer.mask_token_id
        if self.mask_id is None:
            # SpeciesLM should define one; fall back to [MASK] lookup if needed.
            try:
                self.mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
            except Exception as _:
                raise ValueError("Tokenizer has no [MASK] token; MLM probabilities require it.")

        # Discover contiguous "k-mer token block" safely and build:
        #  - self.kmer_ids_sorted: list[int] of vocab ids for all DNA kmers
        #  - self.kmer_id_to_compact: dict[vocab_id] -> [0..N_kmer-1]
        #  - self.non_kmer_cols_keep: LongTensor columns to gather from logits (in vocab id order)
        self._build_kmer_index_and_filter()

        # Cache kmer_id tensor on device for fast lookups
        self._kmer_id_tensor = None


    def _tokenize_seq(
        self,
        seq: str,
        species_proxy: Optional[str],
        *,
        add_special_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize as stride-1 k-mers with optional species prefix token/string."""
        seq = seq.upper()
        # build "A C G T A ..." of stride-1 6-mers
        def kmers_stride1(s: str, k: int) -> List[str]:
            return [s[i : i + k] for i in range(0, len(s) - k + 1)]

        toks = " ".join(kmers_stride1(seq, self.k))
        text = (species_proxy + " " + toks) if species_proxy else toks
        enc = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        return enc

    def _count_specials(self, input_ids: torch.Tensor, where: str) -> int:
        """Heuristically count non-DNA-kmer tokens at left/right."""
        # Cache kmer_id_tensor on the correct device
        if self._kmer_id_tensor is None or self._kmer_id_tensor.device != input_ids.device:
            self._kmer_id_tensor = torch.tensor(list(self._kmer_id_set), device=input_ids.device, dtype=input_ids.dtype)

        kmer_id_tensor = self._kmer_id_tensor

        if where == "left":
            c = 0
            for i in range(len(input_ids)):
                if not torch.any(input_ids[i] == kmer_id_tensor):
                    c += 1
                else:
                    break
            return c
        elif where == "right":
            c = 0
            for i in range(len(input_ids) - 1, -1, -1):
                if not torch.any(input_ids[i] == kmer_id_tensor):
                    c += 1
                else:
                    break
            return c
        else:
            raise ValueError("where must be 'left' or 'right'")

    def _build_kmer_index_and_filter(self):
        """
        Build:
          self.kmer_ids_sorted      : list[int] of vocab ids for DNA k-mers
          self._kmer_id_set         : set[int] for quick membership
          self._gather_vocab_cols   : LongTensor to gather k-mer logits into compact [0..N_kmer)
          self._prb_filter          : (k, Nkmer, 4) tensor mapping k-mer probs -> base probs
        """
        vocab: Dict[str, int] = self.tokenizer.get_vocab()
        # Collect all vocabulary entries that are *pure DNA* kmers of length k
        kmer_entries: List[Tuple[str, int]] = []
        for tok, tid in vocab.items():
            # Some tokenizers wrap words like "A C G ..." exactly; we want plain kmer tokens
            tk = re.sub(r"[^ACGT]", "", tok.upper())
            if _is_dna_kmer(tk, self.k):
                kmer_entries.append((tok, tid))

        if not kmer_entries:
            raise RuntimeError("Could not find any pure DNA k-mer tokens in SpeciesLM vocab.")

        # Sort by vocab id for deterministic gather
        kmer_entries.sort(key=lambda x: x[1])
        self.kmer_ids_sorted = [tid for (_, tid) in kmer_entries]
        self._kmer_id_set = set(self.kmer_ids_sorted)
        self._gather_vocab_cols = torch.tensor(self.kmer_ids_sorted, dtype=torch.long, device=self.device)
        Nkmer = len(self.kmer_ids_sorted)

        # Build prb_filter: (k, Nkmer, 4)
        # For each k-mer token (by vocab id order), mark which base appears at offset o in [0..k-1].
        nt_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
        prb_filter = np.zeros((self.k, Nkmer, 4), dtype=np.float32)
        for compact_j, (_, tid) in enumerate(kmer_entries):
            tok_str = self.tokenizer.convert_ids_to_tokens(tid)
            kmer = re.sub(r"[^ACGT]", "", tok_str.upper())
            if len(kmer) != self.k:
                continue
            for o, nt in enumerate(kmer):
                prb_filter[o, compact_j, nt_to_idx[nt]] = 1.0
        self._prb_filter = torch.from_numpy(prb_filter).to(self.device)


    # --- add these helpers inside SpeciesLMWrapper --------------------------------
    
    def _max_model_tokens(self) -> int:
        # fallbacks across configs/tokenizers; SpeciesLM uses 512
        m = getattr(self.model.config, "max_position_embeddings", None)
        t = getattr(self.tokenizer, "model_max_length", None)
        # HF uses very large default for unknown; clamp to config if available
        if m is None and (t is None or t > 8192):
            return 512
        if m is None:
            return int(t)
        if t is None or t > m:
            return int(m)
        return int(min(m, t))
    
    def _window_nt_len(self) -> int:
        # leave a small margin for specials/species tokenization overhead
        max_tok = self._max_model_tokens()
        safety = 16
        kmer_budget = max(32, max_tok - safety)       # number of k-mer tokens per window
        return int(kmer_budget + self.k - 1)          # nucleotides per window
    
    def _iter_seq_windows(self, seq: str):
        """Yield (start_nt, end_nt) half-open windows with overlap k-1."""
        L = len(seq)
        W = self._window_nt_len()
        step = max(1, W - (self.k - 1))
        s = 0
        while s < L:
            e = min(L, s + W)
            yield s, e
            if e == L:
                break
            s += step
    
    # --- standardized API methods -------------------------------------------------

    @torch.no_grad()
    def predict_nucleotides(
        self,
        seq: str,
        positions: List[int],
        *,
        species_proxy: Optional[str] = None,
        return_dict: bool = True,
    ) -> Union[List[Dict[str, float]], np.ndarray]:
        """
        Predict nucleotide probabilities at specified positions.
        Uses the full acgt_probs method internally and extracts requested positions.
        """
        # Get probabilities for all positions
        all_probs = self.acgt_probs(seq, species_proxy)  # (L, 4) array with [A, C, G, T]

        if return_dict:
            results = []
            for pos in positions:
                if not (0 <= pos < len(seq)):
                    raise IndexError(f"Position {pos} out of range for sequence length {len(seq)}")
                probs = all_probs[pos]
                results.append({
                    'A': float(probs[0]),
                    'C': float(probs[1]),
                    'G': float(probs[2]),
                    'T': float(probs[3]),
                })
            return results
        else:
            results = np.zeros((len(positions), 4), dtype=np.float32)
            for i, pos in enumerate(positions):
                if not (0 <= pos < len(seq)):
                    raise IndexError(f"Position {pos} out of range for sequence length {len(seq)}")
                results[i] = all_probs[pos]
            return results

    # --- modify embed() to chunk when needed --------------------------------------

    @torch.no_grad()
    def embed(self, seq, *, species_proxy=None, pool="mean", layers=None, return_numpy=True):
        if len(seq) <= self._window_nt_len():
            # original single-shot path
            enc = self._tokenize_seq(seq, species_proxy, add_special_tokens=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc, output_hidden_states=True)
            if layers is None:
                if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                    rep = out.last_hidden_state
                else:
                    rep = out.hidden_states[-1]
            else:
                rep = out.hidden_states[layers] if isinstance(layers, int) \
                      else torch.stack(out.hidden_states[layers[0]:layers[1]], 0).mean(0)
                
            rep = rep[0]
            if pool == "mean":
                mask = enc["attention_mask"][0].unsqueeze(-1)
                outv = (rep * mask).sum(0) / mask.sum().clamp(min=1)
            elif pool == "cls":
                outv = rep[0]
            elif pool == "tokens":
                outv = rep
            else:
                raise ValueError("pool must be 'mean'|'cls'|'tokens'")
            return outv.detach().cpu().numpy() if return_numpy else outv
    
        # Chunked path
        if pool == "tokens":
            # total number of stride-1 k-mer tokens across full sequence
            n_total = max(0, len(seq) - self.k + 1)
            if n_total == 0:
                return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        
            # accumulate on CPU to keep GPU mem low
            H = int(getattr(self.model.config, "hidden_size", getattr(self.model.config, "d_model", 768)))
            sum_emb = torch.zeros((n_total, H), dtype=torch.float32)
            cnt = torch.zeros((n_total, 1), dtype=torch.float32)
        
            for s, e in self._iter_seq_windows(seq):
                enc = self._tokenize_seq(seq[s:e], species_proxy, add_special_tokens=True)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc, output_hidden_states=True)
        
                rep = out.last_hidden_state if (hasattr(out, "last_hidden_state") and out.last_hidden_state is not None) \
                      else out.hidden_states[-1]
                rep = rep[0]  # (T_window, H)
        
                # figure out how many non-kmer specials flank the k-mer block in this window
                left_special  = self._count_specials(enc["input_ids"][0], where="left")
                right_special = self._count_specials(enc["input_ids"][0], where="right")
        
                # token embeddings for actual k-mers in this window
                rep_core = rep[left_special : rep.shape[0] - right_special]      # (T_core, H)
                # should correspond to k-mer starts covering seq[s:e]
                n_tok = (e - s) - self.k + 1
                if n_tok <= 0:
                    continue
                # Align: token index 0 in this window corresponds to global token index `s`
                rep_core = rep_core[:n_tok]                                      # (n_tok, H)
                g0, g1 = s, s + n_tok                                            # global token slice
        
                # accumulate (CPU)
                rc = rep_core.detach().cpu()
                sum_emb[g0:g1] += rc
                cnt[g0:g1]    += 1.0
        
            out_tokens = sum_emb / torch.clamp_min(cnt, 1.0)
            return out_tokens.numpy() if return_numpy else out_tokens
        
            
        agg = None
        weight_sum = 0.0
        for s, e in self._iter_seq_windows(seq):
            enc = self._tokenize_seq(seq[s:e], species_proxy, add_special_tokens=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc, output_hidden_states=True)
            if layers is None:
                if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                    rep = out.last_hidden_state
                else:
                    rep = out.hidden_states[-1]
            else:
                rep = out.hidden_states[layers] if isinstance(layers, int) \
                      else torch.stack(out.hidden_states[layers[0]:layers[1]], 0).mean(0)

            rep = rep[0]                           # (T, H)

            if pool == "mean":
                mask = enc["attention_mask"][0].unsqueeze(-1)
                vec = (rep * mask).sum(0) / mask.sum().clamp(min=1)
            elif pool == "cls":
                vec = rep[0]
            w = float(e - s)  # weight by nucleotides covered
            agg = vec * w if agg is None else agg + vec * w
            weight_sum += w
        outv = agg / max(weight_sum, 1.0)
        return outv.detach().cpu().numpy() if return_numpy else outv
    
    # --- modify acgt_probs() to chunk + stitch ------------------------------------
        
    @torch.no_grad()
    def acgt_probs(self, seq: str, species_proxy: Optional[str], *, pred_batch_size: int = 128) -> np.ndarray:
        L = len(seq)
        if L == 0:
            return np.zeros((0, 4), dtype=np.float32)
    
        probs_sum = np.zeros((L, 4), dtype=np.float32)
        probs_cnt = np.zeros((L,), dtype=np.float32)
        k = self.k
        kmers_cols = self._gather_vocab_cols  # (Nkmer,)
    
        for s, e in self._iter_seq_windows(seq):
            # tokenize this window
            enc = self._tokenize_seq(seq[s:e], species_proxy, add_special_tokens=True)
            input_ids = enc["input_ids"][0].to(self.device)         # (T,)
            attn = enc["attention_mask"][0].to(self.device)         # (T,)
    
            # count specials flanking the k-mer block
            left_special  = self._count_specials(input_ids, where="left")
            right_special = self._count_specials(input_ids, where="right")
    
            Lw = e - s                                  # bases in this window
            T = int(input_ids.shape[0])
            Ltok = T - left_special - right_special     # k-mer tokens
    
            if Ltok <= 0:
                continue
    
            # Build masked inputs: one row per base position in the window
            # More efficient: create all rows at once, then mask in-place
            masked_inputs = input_ids.unsqueeze(0).expand(Lw, -1).clone()  # (Lw, T)
            for p in range(Lw):
                t_min = max(0, p - (k - 1))
                t_max = min(p, Ltok - 1)
                if t_min <= t_max:
                    masked_inputs[p, t_min + left_special:t_max + left_special + 1] = self.mask_id
    
            # forward in batches; fold k-mer probs -> base probs
            window_probs = np.zeros((Lw, 4), dtype=np.float32)
            for b in range(0, masked_inputs.shape[0], pred_batch_size):
                mb = masked_inputs[b:b + pred_batch_size]                       # (B, T)
                attn_mb = attn.unsqueeze(0).expand(mb.shape[0], -1)             # (B, T)
                logits = self.model(mb, attention_mask=attn_mb).logits          # (B, T, V)
                kept   = logits[:, left_special: logits.shape[1] - right_special, :]  # (B, Ltok, V)
                gathered = kept.index_select(-1, kmers_cols)                     # (B, Ltok, Nkmer)
                km_probs = torch.softmax(gathered, dim=-1)                       # (B, Ltok, Nk)
    
                B = km_probs.shape[0]
                for i in range(B):
                    p = b + i
                    if p >= Lw:
                        break
                    t_min = max(0, p - (k - 1))
                    t_max = min(p, Ltok - 1)
                    if t_min > t_max:
                        continue
                    sel = km_probs[i, t_min:t_max + 1, :]                        # (m, Nk)
                    # fold: for token start t, base offset o = p - t
                    prb_accum = torch.zeros((4,), device=km_probs.device, dtype=km_probs.dtype)
                    for j, t_start in enumerate(range(t_min, t_max + 1)):
                        o = p - t_start                                         # 0..k-1
                        prb_accum += sel[j] @ self._prb_filter[o]               # (Nk) @ (Nk,4) -> (4,)
                    prb = prb_accum / (prb_accum.sum() + 1e-9)
                    window_probs[p] = prb.detach().cpu().numpy()
    
            # stitch this window back into full-length arrays
            probs_sum[s:e] += window_probs.astype(np.float32)
            probs_cnt[s:e] += 1.0
    
        probs = probs_sum / np.maximum(probs_cnt[:, None], 1.0)
        return probs.astype(np.float32)
