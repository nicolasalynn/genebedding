
from __future__ import annotations
import copy, math, sys, os
from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Support both package import and direct sys.path import
try:
    from .base_wrapper import BaseWrapper
except ImportError:
    from base_wrapper import BaseWrapper

# Default asset paths
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'convnova')
_DEFAULT_CONFIG_YAML = os.path.join(_ASSETS_DIR, 'convnova.yaml')
_DEFAULT_CHECKPOINT_PATH = os.path.join(_ASSETS_DIR, 'last.backbone.pth')

# =============== Simple DNA tokenizer (A/C/G/T/N -> 0..4) ===============

class BaseDNATokenizer:
    ALPHABET = ("A", "C", "G", "T", "N")
    MAP = {c: i for i, c in enumerate(ALPHABET)}
    def __init__(self): self.vocab_size = len(self.ALPHABET); self.PAD_ID = None
    def encode(self, seq: str) -> Dict[str, List[int]]:
        s = seq.upper()
        ids = [self.MAP.get(ch, self.MAP["N"]) for ch in s]
        return {"input_ids": ids, "attention_mask": [1]*len(ids), "char_to_token": list(range(len(ids)))}
    def id_to_token(self, idx: int) -> str:
        return self.ALPHABET[idx] if 0 <= idx < len(self.ALPHABET) else "N"
    def decode_ids(self, ids: List[int]) -> str:
        return "".join(self.id_to_token(int(i)) for i in ids)

# =============== Small helpers ===============

def _unwrap_tuple(x):
    return x[0] if isinstance(x, (tuple, list)) and len(x) > 0 else x

def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # torch>=2.1
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu")

def _strip_prefixes(sd: dict, prefixes=("model.", "module.")):
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p): nk = nk[len(p):]
        out[nk] = v
    return out

def _filter_to_model(sd: dict, model: nn.Module):
    target = model.state_dict()
    keep = {}
    for k, v in sd.items():
        if k in target and getattr(v, "shape", None) == getattr(target[k], "shape", None):
            keep[k] = v
    return keep

def load_backbone_weights(model: nn.Module, ckpt_path: str, *, strict=False) -> Tuple[List[str], List[str]]:
    state = _safe_torch_load(ckpt_path)
    sd = state.get("state_dict", state)
    drop_prefixes = (
        "train_torchmetrics.", "val_torchmetrics.", "test_torchmetrics.",
        "task.", "encoder.", "decoder.", "metrics."
    )
    sd = {k: v for k, v in sd.items() if not any(k.startswith(dp) for dp in drop_prefixes)}
    sd = _strip_prefixes(sd, prefixes=("model.", "module."))
    sd = _filter_to_model(sd, model)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    return list(missing), list(unexpected)

# =============== ConvNova blocks ===============

class LayerNorm(nn.Module):
    """LayerNorm that supports channels_first (B,C,L) or channels_last (B,L,C)."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps; self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None] * x + self.bias[None, :, None]

def get_dilation_schedule(dilation_base, dilation_max, num_layers, small=True):
    if small:
        return [1 if i < 2 else min(dilation_max, dilation_base ** (i - 2))
                for i in range(num_layers)]
    return [min(dilation_max, dilation_base ** i) for i in range(num_layers)]

def build_conv_layers(hidden_dim, kernel_size, num_layers, num_stacks, dilation_base, dilation_max):
    dilations = get_dilation_schedule(dilation_base, dilation_max, num_layers)
    conv_layers = []
    for d in dilations:
        padding = (kernel_size - 1) // 2 * d
        conv_layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                                     dilation=d, padding=padding))
    conv_layers = nn.ModuleList([copy.deepcopy(layer) for layer in conv_layers for _ in range(num_stacks)])
    return conv_layers

class CNNModel(nn.Module):
    """
    ConvNova-ish 1D dilated CNN with RC gating. Produces per-token hidden features or base logits.
    Expected input at inference here: seq: LongTensor (B,L) with A,C,G,T,N -> 0..4.
    """
    def __init__(self,
                 args,                      # must have: hidden_dim, num_cnn_stacks, dropout
                 alphabet_size: int,
                 for_representation: bool = False,
                 pretrain: bool = False,
                 dilation: int = 2,
                 kernel_size: int = 9,
                 num_conv1d: int = 5,
                 d_inner: int = 2,
                 final_conv: bool = False,
                 ffn: bool = True,
                 **kwargs):
        super().__init__()
        self.alphabet_size = int(alphabet_size)
        self.args = args
        self.for_representation = for_representation
        self.d_model = int(args.hidden_dim)
        self.pretrain = pretrain
        self.num_conv1d = int(num_conv1d)
        self.d_inner = int(d_inner)
        self.use_final_conv = bool(final_conv)
        self.num_layers = self.num_conv1d * int(args.num_cnn_stacks)
        self.num_cnn_stacks = int(args.num_cnn_stacks)
        self.hidden_dim = int(1.42 * self.d_model)  # not heavily used, kept for parity
        self.ffn = bool(ffn)

        # token input embedding (one-hot -> channels)
        self.linear   = nn.Conv1d(self.alphabet_size, self.d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.rc_linear= nn.Conv1d(self.alphabet_size, self.d_model, kernel_size=kernel_size, padding=kernel_size//2)

        # gated conv stacks (dilated)
        self.convs = build_conv_layers(self.d_model, kernel_size, self.num_conv1d, self.num_cnn_stacks,
                                       dilation_base=dilation, dilation_max=1024)
        self.gates = build_conv_layers(self.d_model, kernel_size, self.num_conv1d, self.num_cnn_stacks,
                                       dilation_base=dilation, dilation_max=1024)

        if ffn:
            self.mlpgroup2 = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.LayerNorm(self.d_model),
                ) for _ in range(self.num_layers)
            ])

        self.milinear = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*self.d_inner), nn.GELU(),
            nn.Linear(self.d_model*self.d_inner, self.d_model*self.d_inner), nn.LayerNorm(self.d_model*self.d_inner),
            nn.Linear(self.d_model*self.d_inner, self.d_model*self.d_inner), nn.GELU(),
            nn.Linear(self.d_model*self.d_inner, self.d_model), nn.LayerNorm(self.d_model)
        )
        self.norms    = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.num_layers)])
        self.rc_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.num_layers)])

        self.final_conv = None
        if self.use_final_conv:
            self.final_conv = nn.Sequential(
                nn.Conv1d(self.d_model, self.d_model, kernel_size=1), nn.GELU(),
                nn.Conv1d(self.d_model, self.d_model, kernel_size=1)
            )

        # base head (can be replaced by wrapper if needed)
        self.out_linear = nn.Linear(self.d_model, self.alphabet_size)
        self.dropout = nn.Dropout(float(args.dropout))

    def forward(self, seq, t=None, cls=None, return_embedding=False, state=None):
        """
        seq: LongTensor (B,L) with 0..4 (A,C,G,T,N). If pretrain=True, seq may be (ids, mask).
        """
        if self.pretrain:
            mask = seq[1]; seq = seq[0]

        # ACGTN mapping with RC
        Nmask = (seq == 4)
        rc_seq = 3 - seq
        rc_seq = torch.where(Nmask, seq.new_full((), 4), rc_seq)

        # one-hot
        rc_seq = F.one_hot(rc_seq, num_classes=self.alphabet_size).float()  # (B,L,V)
        seq    = F.one_hot(seq,    num_classes=self.alphabet_size).float()

        feat   = F.gelu(self.linear(seq.transpose(1, 2)))      # (B,D,L)
        rc_feat= F.gelu(self.rc_linear(rc_seq.transpose(1, 2)))

        # gated dilated stacks
        for i in range(self.num_layers):
            h    = self.dropout(feat.clone())                  # (B,D,L)
            rc_h = self.dropout(rc_feat.clone())
            h    = self.norms[i](h.transpose(1, 2)).transpose(1, 2)       # (B,D,L)
            rc_h = self.rc_norms[i](rc_h.transpose(1, 2)).transpose(1, 2)
            g    = torch.sigmoid(self.gates[i](rc_h))                        # (B,D,L)
            h    = F.gelu(self.convs[i](h))                                   # (B,D,L)
            feat = h * g + feat
            rc_feat = g + rc_feat
            if self.ffn:
                feat = self.mlpgroup2[i](feat.transpose(1, 2)).transpose(1, 2) + feat

        feat = (self.milinear(feat.transpose(1, 2)).transpose(1, 2)) + feat  # (B,D,L)
        if self.final_conv is not None:
            feat = self.final_conv(feat)                                      # (B,D,L)
        feat = feat.transpose(1, 2)                                           # (B,L,D)

        if not self.pretrain:
            if self.for_representation or return_embedding:
                return feat, None                                             # (B,L,D)
            logits = self.out_linear(feat)                                    # (B,L,V)
            return logits
        else:
            lm_logits = self.out_linear(feat)
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
            return CausalLMOutput(logits=(lm_logits, mask)), None

    @property
    def d_output(self):
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule must set d_output")
        return self.d_model

# =============== Wrapper with NT-like API ===============

class _Args:
    def __init__(self, hidden_dim=384, num_cnn_stacks=4, dropout=0.1):
        self.hidden_dim = hidden_dim
        self.num_cnn_stacks = num_cnn_stacks
        self.dropout = dropout


class ConvNovaWrapper(BaseWrapper):
    def __init__(self,
                 *,
                 # Optional config YAML and checkpoint
                 config_yaml: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 # Model architecture parameters (can override YAML or use defaults)
                 hidden_dim: int = 384,
                 num_cnn_stacks: int = 4,
                 dropout: float = 0.1,
                 kernel_size: int = 9,
                 dilation: int = 2,
                 num_conv1d: int = 5,
                 d_inner: int = 2,
                 final_conv: bool = False,
                 ffn: bool = True,
                 # DP / device
                 device: Optional[str] = None,
                 use_dataparallel: bool = False,
                 device_ids: Optional[List[int]] = None):
        """
        Build ConvNova using the embedded CNNModel implementation.
        Can optionally load config from YAML and/or weights from checkpoint.
        Exposes standardized API: embed(), predict_nucleotides().
        Legacy methods preserved: get_masked_position_logits(), pattern_matching_filter(), center_token_mapping().
        """
        super().__init__()
        # device
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available()
                       else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                             else "cpu"))
        )
        self.dtype = torch.float32

        # tokenizer & base order
        self.tokenizer = BaseDNATokenizer()
        self.base_order = ("A","C","G","T","N")
        self._base_letter2idx = {b: i for i, b in enumerate(self.base_order)}

        # Use default asset paths if not provided
        if config_yaml is None:
            config_yaml = _DEFAULT_CONFIG_YAML
        if checkpoint_path is None:
            checkpoint_path = _DEFAULT_CHECKPOINT_PATH

        # Load config from YAML
        cfg = OmegaConf.load(config_yaml)
        model_kwargs = dict(cfg.model)

        # We want inference path that outputs per-token logits; set flags explicitly.
        model_kwargs.setdefault("alphabet_size", 5)       # A,C,G,T,N
        model_kwargs["for_representation"] = False
        model_kwargs["pretrain"] = False                  # inference logits

        # build model
        core = CNNModel(args=type("Args", (), {
                            "hidden_dim": int(model_kwargs.get("hidden_dim", 384)),
                            "num_cnn_stacks": int(model_kwargs.get("num_cnn_stacks", 4)),
                            "dropout": float(model_kwargs.get("dropout", 0.1))
                         })(),
                         alphabet_size=int(model_kwargs.get("alphabet_size", 5)),
                         for_representation=bool(model_kwargs.get("for_representation", False)),
                         pretrain=bool(model_kwargs.get("pretrain", False)),
                         dilation=int(model_kwargs.get("dilation", 2)),
                         kernel_size=int(model_kwargs.get("kernel_size", 9)),
                         num_conv1d=int(model_kwargs.get("num_conv1d", 5)),
                         d_inner=int(model_kwargs.get("d_inner", 2)),
                         final_conv=bool(model_kwargs.get("final_conv", False)),
                         ffn=bool(model_kwargs.get("ffn", True)),
                         ).to(self.device).eval()

        # optional checkpoint
        if checkpoint_path:
            missing, unexpected = load_backbone_weights(core, checkpoint_path, strict=False)

        # probe embedding dim & ensure base head matches vocab size
        with torch.no_grad():
            prev = getattr(core, "for_representation", False)
            core.for_representation = True
            probe = torch.randint(0, 5, (1, 128), device=self.device)
            feat, _ = core(seq=probe)                 # (1,L,D)
            D = feat.shape[-1]
            core.for_representation = prev

        head_ok = isinstance(getattr(core, "out_linear", None), nn.Linear) \
                  and core.out_linear.in_features == D \
                  and core.out_linear.out_features in (4, 5)
        if not head_ok:
            core.out_linear = nn.Linear(D, 5, bias=True).to(self.device).eval()

        # wrap DP if asked
        if use_dataparallel and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(core, device_ids=device_ids)
        else:
            self.model = core

            
    # ---- internals ----
    def _core(self): return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _tokenize(self, seq: str):
        tok = self.tokenizer.encode(seq)
        ids = torch.as_tensor(tok["input_ids"], dtype=torch.long, device=self.device)
        am  = torch.as_tensor(tok["attention_mask"], dtype=torch.long, device=self.device)
        return ids, am, tok["char_to_token"]

    # ---- public helpers ----
    def find_N_positions(self, seq: str) -> List[int]:
        """Indices where the input has 'N'."""
        return [i for i, c in enumerate(seq) if c.upper() == "N"]

    # ---- public API ----
    @torch.no_grad()
    def embed(
        self,
        seq: Union[str, List[str]],
        *,
        pool: str = "mean",
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray]]:
        """
        Generate embeddings for the input sequence(s).

        If seq is a string -> returns (H,) for mean/cls, or (L,H) for tokens.
        If seq is a list[str] -> returns (B,H) for mean/cls, or list[(Li,H)] for tokens.
        """
        is_batch = isinstance(seq, (list, tuple))

        if not is_batch:
            # single sequence
            ids, am, _ = self._tokenize(seq)
            ids = ids.unsqueeze(0)  # (1,L)
            core = self._core()
            prev = getattr(core, "for_representation", False)
            core.for_representation = True
            try:
                H, _ = core(seq=ids)                    # (1,L,D)
            finally:
                core.for_representation = prev
            H = H[0]                                    # (L,D)

            if pool == "mean":
                feat = (H * am.unsqueeze(-1)).sum(0) / am.sum().clamp(min=1)
            elif pool == "cls":
                feat = H[0]
            elif pool == "tokens":
                feat = H
            else:
                raise ValueError("pool must be one of {'mean','cls','tokens'}")
            return feat.detach().cpu().numpy() if return_numpy else feat

        # batched path - process each sequence
        results = [self.embed(s, pool=pool, return_numpy=False) for s in seq]

        if pool == "tokens":
            # variable lengths -> return list
            if return_numpy:
                return [r.detach().cpu().numpy() for r in results]
            return [r.detach().cpu() for r in results]

        # stack for mean/cls
        stacked = torch.stack(results, dim=0)  # (B, H)
        return stacked.detach().cpu().numpy() if return_numpy else stacked.detach().cpu()

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
        ConvNova is k=1 (base-level), so positions map directly to sequence indices.

        If positions is not provided, automatically detects positions with 'N' in the sequence.
        """
        # Auto-detect 'N' sites if not provided
        if not positions:
            positions = self.find_N_positions(seq)
        if not positions:
            raise ValueError("No positions provided and no 'N' bases found in seq.")

        # Get logits for whole sequence
        logits = self.logits_for_seq(seq)  # (L, 5) - A,C,G,T,N

        if return_dict:
            results = []
            for pos in positions:
                if not (0 <= pos < len(seq)):
                    raise IndexError(f"Position {pos} out of range for sequence length {len(seq)}")

                # Get probs at this position (only A/C/G/T, ignore N)
                vec = logits[pos, :4]  # exclude N at index 4
                probs = torch.softmax(vec, dim=0).detach().cpu().numpy()

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

                vec = logits[pos, :4]
                probs = torch.softmax(vec, dim=0).detach().cpu().numpy()
                results[i] = probs
            return results

    @torch.no_grad()
    def logits_for_seq(self, seq: str) -> torch.Tensor:
        """(L, V) base logits for the whole sequence."""
        ids, _, _ = self._tokenize(seq)
        ids = ids.unsqueeze(0)  # (1,L)
        core = self._core()
        prev = getattr(core, "for_representation", False)
        core.for_representation = False
        try:
            logits = core(seq=ids)                  # (1,L,V)
        finally:
            core.for_representation = prev
        logits = _unwrap_tuple(logits)
        if logits.dim() == 2: logits = logits.unsqueeze(1)  # (1,1,V) edge-case
        return logits[0]                                    # (L,V)

    @torch.no_grad()
    def get_position_base_logits(self, *, seq: str, token_index: int) -> torch.Tensor:
        L = len(seq)
        if token_index < 0 or token_index >= L:
            raise IndexError(f"token_index {token_index} out of range [0,{L})")
        return self.logits_for_seq(seq)[int(token_index), :].detach()

    @torch.no_grad()
    def get_masked_position_logits(self,
                                   *,
                                   token_ids_no_specials: List[int],
                                   token_position_of_interest: int,
                                   attention_mask: Optional[List[int]] = None,
                                   return_probs: bool = False,
                                   temperature: float = 1.0):
        """
        NT-compatible signature: we don't actually mask; we just decode ids back to seq
        and return the base logits at the requested position.
        """
        L = len(token_ids_no_specials)
        if not (0 <= token_position_of_interest < L):
            raise IndexError("token_position_of_interest out of range")
        if attention_mask is not None:
            if len(attention_mask) != L:
                raise ValueError("attention_mask length mismatch")
            if attention_mask[token_position_of_interest] == 0:
                raise RuntimeError("interest position is masked out")
        seq = self.tokenizer.decode_ids(list(map(int, token_ids_no_specials)))
        vec = self.get_position_base_logits(seq=seq, token_index=int(token_position_of_interest))
        if return_probs:
            vec = torch.softmax(vec / float(max(1e-6, temperature)), dim=-1)
        return vec.detach().cpu()  # (V,)

    def pattern_matching_filter(self, logits_vec, target_pattern: str, **_):
        """
        For ConvNova (k=1), return normalized probs over A/C/G/T from provided logits.
        """
        logits = torch.from_numpy(logits_vec) if isinstance(logits_vec, np.ndarray) else logits_vec
        probs = torch.softmax(logits, dim=-1)
        out = {}
        for b in ("A","C","G","T"):
            idx = self._base_letter2idx.get(b, None)
            out[b] = float(probs[idx].item()) if (idx is not None and idx < probs.numel()) else 0.0
        return out

    def center_token_mapping(self,
                             *,
                             chrom: str,
                             query_pos: int,
                             query_ref: str,
                             rc: bool,
                             L: int,
                             fasta_name: str = "hg38",
                             mutations: Optional[List[Tuple[int, str, str]]] = None) -> Dict[str, Union[str, int, bool, List[int]]]:
        """
        For compatibility with your NT tooling. Requires seqmat installed & FASTA available.
        """
        try:
            from seqmat import SeqMat
        except Exception as e:
            raise RuntimeError("seqmat is required for center_token_mapping") from e

        s = SeqMat.from_fasta(fasta_name, chrom, query_pos - L, query_pos + L + 1)
        if mutations: s.apply_mutations(mutations, permissive_ref=True)
        if rc: s = s.reverse_complement()

        s_ref = s.clone()
        s_ref.apply_mutations([(query_pos, query_ref, "N")], permissive_ref=True)

        seq_s, seq_ref_s = s.seq, s_ref.seq
        token_pos = L  # base-level
        ids = self.tokenizer.encode(seq_s)["input_ids"]
        am  = self.tokenizer.encode(seq_s)["attention_mask"]
        return {
            "seq": seq_s,
            "seq_ref": seq_ref_s,
            "k": 1,
            "token_ids": ids,
            "tokens": list(seq_s),
            "token_position_of_interest": token_pos,
            "token_string": seq_s[token_pos],
            "corresponding_seq_chunk": seq_ref_s[token_pos:token_pos+1],
            "matches": True,
            "_attention_mask": am,
        }

# =============== Example usage ===============
if __name__ == "__main__":
    # Example 1: Use with defaults (no config or checkpoint)
    model = ConvNovaWrapper()

    # Example 2: Load from config and checkpoint (if you have them)
    # model = ConvNovaWrapper(
    #     config_yaml="/path/to/config.yaml",
    #     checkpoint_path="/path/to/checkpoint.pth"
    # )

    seq = "ACGT" * 256

    # Test embedding
    E = model.embed(seq, pool="tokens")          # (L, H)
    print(f"Token embeddings shape: {E.shape}")

    # Test logits
    logits = model.logits_for_seq(seq)           # (L, 5)
    print(f"Logits shape: {logits.shape}")

    # Test position-specific prediction
    pos_vec = model.get_position_base_logits(seq=seq, token_index=100)  # (5,)
    print(f"Position logits shape: {pos_vec.shape}")




