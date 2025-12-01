# borzoi_wrapper.py (adaptive min-input wrapper)
from __future__ import annotations
from typing import Optional, Union, Literal, Dict, List, Tuple
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Support both package import and direct sys.path import
try:
    from .base_wrapper import BaseWrapper
except ImportError:
    from base_wrapper import BaseWrapper

# Default asset paths
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'borzoi')
_DEFAULT_TARGETS_PATH = os.path.join(_ASSETS_DIR, 'targets_human.txt')

Pool = Literal["mean", "cls", "tokens"]



def _dna_to_onehot4(seq: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return (1, 4, L) one-hot tensor in the desired dtype."""
    L = len(seq)
    x = np.zeros((1, 4, L), dtype=np.float32)
    for i, ch in enumerate(seq.upper()):
        if   ch == "A": x[0, 0, i] = 1.0
        elif ch == "C": x[0, 1, i] = 1.0
        elif ch == "G": x[0, 2, i] = 1.0
        elif ch == "T": x[0, 3, i] = 1.0
    return torch.from_numpy(x).to(device=device, dtype=dtype)


def _center_slice(big_len: int, small_len: int) -> slice:
    off = max(0, (big_len - small_len) // 2)
    return slice(off, off + small_len)


def _get_target_len(model: nn.Module, default: int = 6144) -> int:
    """Try to learn Borzoi's target length from its crop layer, else default."""
    for name in ("crop",):
        m = getattr(model, name, None)
        if m is not None and hasattr(m, "target_len"):
            return int(getattr(m, "target_len"))
    for m in model.modules():
        if m.__class__.__name__.lower().startswith("targetlengthcrop") and hasattr(m, "target_len"):
            return int(getattr(m, "target_len"))
    return int(default)



class BorzoiWrapper:
    """
    Minimal wrapper:
      - Pads/crops any input to a fixed min_input_len (default 524,288 bp)
      - Returns central Lret=min(len(seq), target_len) positions
      - Provides:
          tracks(seq) -> (C, Lret) np.array
          embed(seq, pool) -> pooled hidden representations
          get_track_names() -> list of track descriptions
          tracks_by_name(seq, names) -> dict of named tracks
    """

    def __init__(
        self,
        repo: str = "johahi/flashzoi-replicate-0",
        required_input_len: int = 524_288,
        targets_path: str = None,
    ):
        from borzoi_pytorch import Borzoi

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # Load model once
        self.model = Borzoi.from_pretrained(repo).to(self.device, dtype=self.dtype).eval()

        # Final heads should work in float32 (Borzoi does x.float() before them)
        self.model.human_head = self.model.human_head.to(torch.float32)
        if repo == "johahi/flashzoi-replicate-0":
            self.model.mouse_head = self.model.mouse_head.to(torch.float32)
        else:
            self.model.mouse_head = None

        # Crop length of Borzoi outputs
        self.target_len = _get_target_len(self.model, default=6144)

        # Input length we enforce for all sequences
        self.min_input_len = int(required_input_len)

        # for embeddings
        self._head: Optional[nn.Conv1d] = None
        self._last_hidden: Optional[torch.Tensor] = None

        # Track metadata from local assets
        if targets_path is None:
            targets_path = _DEFAULT_TARGETS_PATH

        self._targets_df = pd.read_csv(targets_path, delimiter="\t")
        self._targets_df = self._targets_df.rename(columns={"Unnamed: 0": "feat_index"})
        self._targets_df["full_description"] = (
            self._targets_df["feat_index"].astype(str) + "_" + self._targets_df["description"]
        )
        self.track_names: List[str] = self._targets_df["full_description"].tolist()

        # Build name-to-index mapping for fast lookup
        self._track_name_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(self.track_names)
        }
        # Also map by description only (without index prefix)
        self._track_desc_to_idx: Dict[str, int] = {
            desc: i for i, desc in enumerate(self._targets_df["description"].tolist())
        }

    def get_track_names(self) -> List[str]:
        """Return list of all track names/descriptions."""
        return self.track_names.copy()

    def get_track_metadata(self) -> pd.DataFrame:
        """Return full track metadata DataFrame."""
        return self._targets_df.copy()

    def find_tracks(self, pattern: str) -> List[Tuple[int, str]]:
        """
        Find tracks matching a pattern (case-insensitive substring search).
        Returns list of (index, name) tuples.
        """
        pattern_lower = pattern.lower()
        matches = []
        for i, name in enumerate(self.track_names):
            if pattern_lower in name.lower():
                matches.append((i, name))
        return matches

    # ---------- internal prep ----------
    def _prep(self, seq: str) -> tuple[torch.Tensor, int, slice]:
        """
        Pad/crop user seq to self.min_input_len, but plan to return only
        central Lret=min(len(seq), target_len) from Borzoi output.
        """
        L0 = len(seq)
        if L0 == 0:
            raise ValueError("Empty sequence.")

        # center-pad / center-crop
        if L0 < self.min_input_len:
            pad_total = self.min_input_len - L0
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            seq_proc = ("N" * pad_left) + seq + ("N" * pad_right)
        elif L0 > self.min_input_len:
            sl_in = _center_slice(L0, self.min_input_len)
            seq_proc = seq[sl_in]
        else:
            seq_proc = seq

        # central window in model output
        Lret = min(L0, self.target_len)
        sl_out = _center_slice(self.target_len, Lret)

        x = _dna_to_onehot4(seq_proc, self.device, self.dtype)  # (1,4,min_input_len)
        return x, Lret, sl_out

    # ---------- public API ----------
    @torch.no_grad()
    def tracks(self, seq: str) -> np.ndarray:
        """
        Return Borzoi tracks for seq:
            (C, Lret) np.ndarray
        where:
            C     = n_tracks
            Lret  = min(len(seq), target_len)
        """
        x_proc, Lret, sl_out = self._prep(seq)
        y = self.model(x_proc)  # (1, C, target_len)
        if not isinstance(y, torch.Tensor) or y.ndim != 3:
            raise RuntimeError("Unexpected Borzoi output shape.")
        y = y.squeeze(0)[:, sl_out]  # (C, Lret)
        return y.detach().cpu().numpy()

    @torch.no_grad()
    def tracks_by_name(
        self,
        seq: str,
        names: List[str],
        *,
        match_description: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Get specific tracks by name.

        Parameters
        ----------
        seq : str
            Input DNA sequence
        names : list of str
            Track names or descriptions to retrieve
        match_description : bool
            If True, also match against description without index prefix

        Returns
        -------
        dict : {name: (Lret,) np.ndarray}
        """
        all_tracks = self.tracks(seq)  # (C, Lret)

        result = {}
        for name in names:
            idx = self._track_name_to_idx.get(name)
            if idx is None and match_description:
                idx = self._track_desc_to_idx.get(name)
            if idx is None:
                raise KeyError(f"Track '{name}' not found. Use find_tracks() to search.")
            result[name] = all_tracks[idx]

        return result

    @torch.no_grad()
    def embed(self, seq: str, *, pool: Pool = "tokens", return_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Hidden representations near the output head.
        Returns:
          pool='tokens' -> (Lret, H)
          pool='mean'   -> (H,)
          pool='cls'    -> (H,)  (first position)
        """
        x_proc, Lret, sl_out = self._prep(seq)

        # First pass: get output to infer C and locate final 1x1 conv
        y = self.model(x_proc)  # (1, C, target_len)
        if not isinstance(y, torch.Tensor) or y.ndim != 3:
            raise RuntimeError("Unexpected Borzoi output shape.")
        C = int(y.shape[1])

        # Find last 1x1 conv with out_channels == C
        self._head = None
        for m in self.model.modules():
            if isinstance(m, nn.Conv1d):
                k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
                if k == 1 and m.out_channels == C:
                    self._head = m

        feats: Optional[torch.Tensor] = None
        if self._head is not None:
            self._last_hidden = None

            def hook_fn(mod, inp, out):
                # inp[0]: (N, H, L)
                self._last_hidden = inp[0].detach()

            hook = self._head.register_forward_hook(hook_fn)
            try:
                _ = self.model(x_proc)
                feats = self._last_hidden  # (1, H, target_len)
            finally:
                hook.remove()

        if feats is None:
            # fallback to final output as "features"
            feats = y  # (1, C, target_len)

        feats = feats[0, :, sl_out].transpose(0, 1).contiguous()  # -> (Lret, H)

        if pool == "tokens":
            result = feats
        elif pool == "mean":
            result = feats.mean(dim=0)
        elif pool == "cls":
            result = feats[0]
        else:
            raise ValueError("pool must be one of {'tokens','mean','cls'}")

        # Convert to numpy if requested
        if return_numpy:
            return result.cpu().numpy()
        else:
            return result
