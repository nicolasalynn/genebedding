"""
OpenSpliceAI wrapper.

Provides splice site probabilities and optional embeddings from the
penultimate feature map (input to the final 1x1 conv head).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Union, Literal, Tuple
import os
import warnings

import numpy as np
import torch

try:
    from .base_wrapper import BaseWrapper, PoolMode
except ImportError:
    from base_wrapper import BaseWrapper, PoolMode

BASES = ("A", "C", "G", "T")


@dataclass
class OpenSpliceAIOutput:
    """
    Output from OpenSpliceAIWrapper.forward().

    Attributes
    ----------
    acceptor_prob : torch.Tensor
        Splice acceptor probabilities, shape (1, seq_len).
    donor_prob : torch.Tensor
        Splice donor probabilities, shape (1, seq_len).
    neither_prob : torch.Tensor
        Neither splice site probabilities, shape (1, seq_len).
    embeddings : Optional[torch.Tensor]
        Hidden features from penultimate layer, shape (1, seq_len, hidden_dim).
        Only populated if return_embeddings=True.
    """
    acceptor_prob: torch.Tensor
    donor_prob: torch.Tensor
    neither_prob: torch.Tensor
    embeddings: Optional[torch.Tensor] = None


OPENSPLICEAI_MODELS: Dict[str, int] = {
    "80nt": 80,
    "400nt": 400,
    "2k": 2000,
    "10k": 10000,
}

OpenSpliceAIModelName = Literal["80nt", "400nt", "2k", "10k"]


def list_available_models() -> List[str]:
    """Return list of available OpenSpliceAI flanking sizes."""
    return list(OPENSPLICEAI_MODELS.keys())


def _get_torch_device() -> torch.device:
    """Get the best available device for PyTorch."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            torch.tensor([1.0], device="mps")
            return torch.device("mps")
        except RuntimeError:
            return torch.device("cpu")
    return torch.device("cpu")


def _safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, weights_only=True, map_location=device)
    except TypeError:
        return torch.load(path, map_location=device)


def _normalize_seq(seq: str) -> str:
    return "".join(ch if ch in "ACGT" else "N" for ch in (seq or "").upper())


def _infer_model_dir() -> Optional[str]:
    """Try env override, then look inside the openspliceai package."""
    env_dir = os.environ.get("OPENSPLICEAI_MODEL_DIR")
    if env_dir:
        return env_dir
    try:
        import openspliceai
        pkg_dir = os.path.dirname(os.path.abspath(openspliceai.__file__))
        candidate = os.path.join(pkg_dir, "models", "openspliceai-mane", "10000nt")
        if os.path.isdir(candidate):
            return candidate
    except Exception:
        return None
    return None


def _get_arch_params(flanking_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (W, AR) arrays for a given flanking size.
    Mapping is copied from OpenSpliceAI predict.py.
    """
    if flanking_size == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
    elif flanking_size == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
    elif flanking_size == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10])
    elif flanking_size == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                         10, 10, 10, 10, 25, 25, 25, 25])
    else:
        raise ValueError("flanking_size must be one of {80, 400, 2000, 10000}")
    return W, AR


class OpenSpliceAIWrapper(BaseWrapper):
    """
    OpenSpliceAI wrapper with standardized API for splice site prediction.

    Parameters
    ----------
    model : str, default="10k"
        Flanking size. Options: "80nt", "400nt", "2k", "10k".
    model_dir : str, optional
        Directory containing OpenSpliceAI checkpoint(s). If None, tries
        OPENSPLICEAI_MODEL_DIR or package models.
    device : str, optional
        Device to use. Defaults to CUDA if available, else MPS, else CPU.
    dtype : torch.dtype, default=torch.float32
        Data type for computation.
    """

    def __init__(
        self,
        model: OpenSpliceAIModelName = "10k",
        *,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if model not in OPENSPLICEAI_MODELS:
            raise ValueError(f"Unknown model {model!r}. Available: {list(OPENSPLICEAI_MODELS.keys())}")

        self.flanking_size = OPENSPLICEAI_MODELS[model]
        self.model_name = model
        self.model_dir = model_dir or _infer_model_dir()
        if not self.model_dir or not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                "OpenSpliceAI model directory not found. Set OPENSPLICEAI_MODEL_DIR "
                "or pass model_dir explicitly."
            )

        self.device = torch.device(device) if device else _get_torch_device()
        self.dtype = dtype
        self._models: Optional[List[torch.nn.Module]] = None
        self._sl: Optional[int] = None

    def _ensure_models(self) -> List[torch.nn.Module]:
        if self._models is not None:
            return self._models

        from openspliceai.train_base.openspliceai import SpliceAI
        from openspliceai.constants import SL

        self._sl = int(SL)
        W, AR = _get_arch_params(self.flanking_size)

        model_files = [
            os.path.join(self.model_dir, f)
            for f in os.listdir(self.model_dir)
            if f.endswith(".pt") or f.endswith(".pth")
        ]
        if not model_files:
            raise FileNotFoundError(f"No .pt/.pth files found in {self.model_dir!r}")

        models: List[torch.nn.Module] = []
        for model_file in sorted(model_files):
            state = _safe_torch_load(model_file, self.device)
            model = SpliceAI(L=32, W=W, AR=AR, apply_softmax=True).to(self.device)
            try:
                model.load_state_dict(state)
            except RuntimeError as e:
                warnings.warn(
                    f"Skipping {model_file} due to incompatible shapes: {e}"
                )
                continue
            model.eval()
            models.append(model)

        if not models:
            raise RuntimeError("No OpenSpliceAI models loaded successfully.")

        self._models = models
        return self._models

    def _encode(self, seq: str) -> Tuple[torch.Tensor, int]:
        from openspliceai.predict.predict import create_datapoints
        from openspliceai.constants import SL

        seq = _normalize_seq(seq)
        seq_len = len(seq)
        X = create_datapoints(seq, SL=int(SL), CL_max=int(self.flanking_size))
        X = torch.tensor(X, dtype=torch.float32, device=self.device).permute(0, 2, 1)
        return X, seq_len

    def _run_models(
        self,
        X: torch.Tensor,
        seq_len: int,
        *,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        models = self._ensure_models()

        out_sum = None
        emb_sum = None

        for model in models:
            embeddings = None
            hook = None
            if return_embeddings:
                captured: List[torch.Tensor] = []

                def hook_fn(_mod, inp, _out):
                    if inp and isinstance(inp, tuple):
                        captured.append(inp[0].detach())

                hook = model.final_conv.register_forward_hook(hook_fn)

            with torch.no_grad():
                preds = model(X)  # (B, 3, SL)

            if hook is not None:
                hook.remove()
                if captured:
                    # captured shape: (B, H, SL)
                    emb = captured[0].permute(0, 2, 1).contiguous().view(-1, captured[0].shape[1])
                    embeddings = emb[:seq_len]

            preds = preds.permute(0, 2, 1).contiguous().view(-1, preds.shape[1])
            preds = preds[:seq_len]

            if out_sum is None:
                out_sum = preds
            else:
                out_sum = out_sum + preds

            if return_embeddings:
                if embeddings is None:
                    raise RuntimeError("Failed to capture embeddings from OpenSpliceAI.")
                if emb_sum is None:
                    emb_sum = embeddings
                else:
                    emb_sum = emb_sum + embeddings

        out_avg = out_sum / float(len(models))
        emb_avg = emb_sum / float(len(models)) if return_embeddings else None
        return out_avg, emb_avg

    @torch.no_grad()
    def forward(
        self,
        seq: str,
        *,
        return_embeddings: bool = False,
    ) -> OpenSpliceAIOutput:
        X, seq_len = self._encode(seq)
        preds, emb = self._run_models(X, seq_len, return_embeddings=return_embeddings)

        # preds: (L, 3) with columns [neither, acceptor, donor]
        acceptor = preds[:, 1].unsqueeze(0)
        donor = preds[:, 2].unsqueeze(0)
        neither = preds[:, 0].unsqueeze(0)

        if emb is not None:
            emb = emb.unsqueeze(0)

        return OpenSpliceAIOutput(
            acceptor_prob=acceptor,
            donor_prob=donor,
            neither_prob=neither,
            embeddings=emb,
        )

    @torch.no_grad()
    def predict_splice_sites(
        self,
        seq: str,
        *,
        threshold: float = 0.5,
        return_probs: bool = False,
    ) -> Union[Dict[str, List[int]], Dict[str, Union[List[int], np.ndarray]]]:
        out = self.forward(seq)
        acceptor_probs = out.acceptor_prob[0].detach().cpu().numpy()
        donor_probs = out.donor_prob[0].detach().cpu().numpy()

        result = {
            "acceptor_sites": list(np.where(acceptor_probs > threshold)[0]),
            "donor_sites": list(np.where(donor_probs > threshold)[0]),
        }
        if return_probs:
            result["acceptor_probs"] = acceptor_probs
            result["donor_probs"] = donor_probs
        return result

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
        Generate embeddings for DNA sequence(s) from the penultimate layer.
        """
        is_batch = isinstance(seq, (list, tuple))

        if not is_batch:
            out = self.forward(seq, return_embeddings=True)
            emb = out.embeddings
            if emb is None:
                raise RuntimeError("Embeddings not available from OpenSpliceAI.")
            emb = emb[0]  # (L, H)

            if pool == "tokens":
                result = emb
            elif pool == "cls":
                result = emb[0]
            elif pool == "mean":
                result = emb.mean(dim=0)
            else:
                raise ValueError("pool must be one of {'mean','cls','tokens'}")

            return result.detach().cpu().numpy() if return_numpy else result.detach().cpu()

        # Batched: process independently due to variable lengths
        results = [self.embed(s, pool=pool, return_numpy=return_numpy) for s in seq]
        if pool == "tokens":
            return results

        if return_numpy:
            return np.stack(results, axis=0)
        return torch.stack([torch.as_tensor(r) for r in results], dim=0)
