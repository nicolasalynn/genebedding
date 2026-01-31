# spliceai_wrapper.py
# SpliceAI wrapper for splice site prediction CNN models
# Standardized API: embed(), forward(), predict_splice_sites()
#
# SpliceAI is a dilated CNN that predicts splice site probabilities.
# Uses hook-based feature extraction for embeddings (similar to Borzoi).
#
# Supports:
# - spliceai-pytorch package (pip install spliceai-pytorch)
# - Custom PyTorch SpliceAI models with compatible architecture

from dataclasses import dataclass
from typing import Dict, Optional, List, Union, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn

# Support both package import and direct sys.path import
try:
    from .base_wrapper import BaseWrapper, PoolMode
except ImportError:
    from base_wrapper import BaseWrapper, PoolMode

BASES = ("A", "C", "G", "T")


@dataclass
class SpliceAIOutput:
    """
    Output from SpliceAIWrapper.forward().

    Attributes
    ----------
    acceptor_prob : torch.Tensor
        Splice acceptor probabilities, shape (batch_size, seq_len).
    donor_prob : torch.Tensor
        Splice donor probabilities, shape (batch_size, seq_len).
    neither_prob : torch.Tensor
        Neither splice site probabilities, shape (batch_size, seq_len).
    embeddings : Optional[torch.Tensor]
        Hidden features from penultimate layer, shape (batch_size, seq_len, hidden_dim).
        Only populated if return_embeddings=True.
    """
    acceptor_prob: torch.Tensor
    donor_prob: torch.Tensor
    neither_prob: torch.Tensor
    embeddings: Optional[torch.Tensor] = None


# Model configurations for spliceai-pytorch package
SPLICEAI_MODELS: Dict[str, str] = {
    "80nt": "80nt",
    "400nt": "400nt",
    "2k": "2k",
    "10k": "10k",
}

SpliceAIModelName = Literal["80nt", "400nt", "2k", "10k"]


def _dna_to_onehot(seq: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert DNA sequence to one-hot encoding.

    Returns (1, 4, L) tensor where channels are [A, C, G, T].
    """
    L = len(seq)
    x = np.zeros((1, 4, L), dtype=np.float32)
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, ch in enumerate(seq.upper()):
        if ch in mapping:
            x[0, mapping[ch], i] = 1.0
        # N or other ambiguous bases remain as zeros
    return torch.from_numpy(x).to(device=device, dtype=dtype)


def list_available_models() -> List[str]:
    """Return list of available model configurations."""
    return list(SPLICEAI_MODELS.keys())


class SpliceAIWrapper(BaseWrapper):
    """
    SpliceAI wrapper with standardized API for splice site prediction.

    SpliceAI is a dilated convolutional neural network that predicts
    splice acceptor and donor probabilities for each position in a sequence.

    Implements BaseWrapper: embed()
    Also provides: forward(), predict_splice_sites()

    Parameters
    ----------
    model : str, default="10k"
        Model context size. Options: "80nt", "400nt", "2k", "10k".
        Larger context = more accurate but slower.
    model_path : str, optional
        Path to custom model weights. If provided, loads from path instead
        of using pre-configured models.
    device : str, optional
        Device to use. Defaults to CUDA if available, else MPS, else CPU.
    dtype : torch.dtype, default=torch.float32
        Data type for model weights.
    embed_layer : str, default="auto"
        Which layer to extract embeddings from:
        - "auto": Automatically find penultimate layer before output head
        - "last_conv": Use last convolutional layer
        - Layer name for custom models

    Examples
    --------
    >>> # Load pre-configured model
    >>> wrapper = SpliceAIWrapper(model="10k")

    >>> # Get splice site predictions
    >>> out = wrapper.forward("ACGT" * 1000)
    >>> out.acceptor_prob.shape  # (1, 4000)
    >>> out.donor_prob.shape  # (1, 4000)

    >>> # Get embeddings
    >>> emb = wrapper.embed("ACGT" * 1000, pool="mean")
    >>> emb.shape  # (hidden_dim,)

    >>> # Get both predictions and embeddings
    >>> out = wrapper.forward("ACGT" * 1000, return_embeddings=True)
    >>> out.embeddings.shape  # (1, 4000, hidden_dim)

    Notes
    -----
    - SpliceAI works best with sequences >= 64nt
    - Context requirements vary by model (80nt to 10k)
    - Sequences are automatically padded if needed
    """

    def __init__(
        self,
        model: str = "10k",
        *,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        embed_layer: str = "auto",
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

        self.model_name = model
        self._embed_layer_name = embed_layer

        # Load model
        if model_path is not None:
            # Load from custom path
            self.model = torch.load(model_path, map_location=self.device)
            if isinstance(self.model, dict):
                raise ValueError(
                    "model_path should point to a saved model, not a state_dict. "
                    "Use torch.save(model, path) not torch.save(model.state_dict(), path)"
                )
        else:
            # Load from spliceai-pytorch package
            try:
                from spliceai_pytorch import SpliceAI
            except ImportError:
                raise ImportError(
                    "spliceai-pytorch package required. Install with:\n"
                    "pip install spliceai-pytorch\n"
                    "Or provide model_path for custom models."
                )

            if model not in SPLICEAI_MODELS:
                raise ValueError(f"Unknown model: {model}. Available: {list(SPLICEAI_MODELS.keys())}")

            self.model = SpliceAI.from_preconfigured(SPLICEAI_MODELS[model])

        self.model = self.model.to(self.device).to(self.dtype).eval()

        # Find the embedding layer (penultimate conv before 3-channel output)
        self._embed_hook_layer: Optional[nn.Module] = None
        self._cached_embeddings: Optional[torch.Tensor] = None
        self._find_embed_layer()

        # Get hidden dimension from the embedding layer
        self.hidden_dim = self._get_hidden_dim()

    def _find_embed_layer(self):
        """Find the layer to hook for embeddings (input to final 1x1 conv)."""
        if self._embed_layer_name == "auto":
            # Find last Conv1d with out_channels == 3 (the output head)
            # Hook into its input to get penultimate features
            output_head = None
            prev_conv = None

            for module in self.model.modules():
                if isinstance(module, nn.Conv1d):
                    if module.out_channels == 3:
                        output_head = module
                    else:
                        prev_conv = module

            # The output head's input comes from previous layer
            # We'll hook the output head and capture its input
            self._embed_hook_layer = output_head if output_head else prev_conv

        elif self._embed_layer_name == "last_conv":
            # Find last conv before output
            last_conv = None
            for module in self.model.modules():
                if isinstance(module, nn.Conv1d) and module.out_channels != 3:
                    last_conv = module
            self._embed_hook_layer = last_conv

        else:
            # Custom layer name
            self._embed_hook_layer = dict(self.model.named_modules()).get(self._embed_layer_name)

        if self._embed_hook_layer is None:
            # Fallback: no embedding extraction possible
            import warnings
            warnings.warn("Could not find embedding layer. embed() will use model output.")

    def _get_hidden_dim(self) -> int:
        """Infer hidden dimension from embedding layer."""
        if self._embed_hook_layer is not None and isinstance(self._embed_hook_layer, nn.Conv1d):
            # If this is the output head, hidden_dim is in_channels
            if self._embed_hook_layer.out_channels == 3:
                return self._embed_hook_layer.in_channels
            return self._embed_hook_layer.out_channels
        return 32  # Default fallback

    def __repr__(self) -> str:
        return f"SpliceAIWrapper(model='{self.model_name}', device={self.device}, hidden_dim={self.hidden_dim})"

    def _normalize_seq(self, seq: str) -> str:
        """Clean sequence to valid DNA."""
        import re
        return re.sub(r"[^ACGT]", "N", (seq or "").upper())

    @torch.no_grad()
    def forward(
        self,
        seq: Union[str, List[str]],
        *,
        return_embeddings: bool = False,
    ) -> SpliceAIOutput:
        """
        Run forward pass and return splice site predictions.

        Parameters
        ----------
        seq : str or list of str
            Input DNA sequence(s).
        return_embeddings : bool, default=False
            If True, also extract and return embeddings from penultimate layer.

        Returns
        -------
        SpliceAIOutput
            Dataclass with:
            - acceptor_prob: (batch_size, seq_len) splice acceptor probabilities
            - donor_prob: (batch_size, seq_len) splice donor probabilities
            - neither_prob: (batch_size, seq_len) neither probability
            - embeddings: (batch_size, seq_len, hidden_dim) if return_embeddings=True
        """
        is_batch = isinstance(seq, (list, tuple))
        seqs = list(seq) if is_batch else [seq]

        # Encode sequences
        x_list = [_dna_to_onehot(self._normalize_seq(s), self.device, self.dtype) for s in seqs]

        # Pad to same length if batched
        if len(x_list) > 1:
            max_len = max(x.shape[2] for x in x_list)
            x_list = [
                nn.functional.pad(x, (0, max_len - x.shape[2]))
                for x in x_list
            ]

        x = torch.cat(x_list, dim=0)  # (B, 4, L)

        embeddings = None
        if return_embeddings and self._embed_hook_layer is not None:
            # Register hook to capture embeddings
            self._cached_embeddings = None

            def hook_fn(module, inp, out):
                # For output head: inp[0] is (B, H, L)
                # For other conv: out is (B, H, L)
                if module.out_channels == 3:
                    self._cached_embeddings = inp[0].detach()
                else:
                    self._cached_embeddings = out.detach()

            hook = self._embed_hook_layer.register_forward_hook(hook_fn)
            try:
                logits = self.model(x)  # (B, 3, L)
                embeddings = self._cached_embeddings  # (B, H, L)
                if embeddings is not None:
                    embeddings = embeddings.permute(0, 2, 1)  # (B, L, H)
            finally:
                hook.remove()
        else:
            logits = self.model(x)  # (B, 3, L)

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)  # (B, 3, L)

        return SpliceAIOutput(
            acceptor_prob=probs[:, 0, :],  # (B, L)
            donor_prob=probs[:, 1, :],  # (B, L)
            neither_prob=probs[:, 2, :],  # (B, L)
            embeddings=embeddings,
        )

    @torch.no_grad()
    def predict_splice_sites(
        self,
        seq: str,
        *,
        threshold: float = 0.5,
        return_probs: bool = False,
    ) -> Union[Dict[str, List[int]], Dict[str, Union[List[int], np.ndarray]]]:
        """
        Predict splice site positions in a sequence.

        Parameters
        ----------
        seq : str
            Input DNA sequence.
        threshold : float, default=0.5
            Probability threshold for calling a splice site.
        return_probs : bool, default=False
            If True, also return full probability arrays.

        Returns
        -------
        dict
            - 'acceptor_sites': List of 0-based positions predicted as acceptors
            - 'donor_sites': List of 0-based positions predicted as donors
            - 'acceptor_probs': (L,) array if return_probs=True
            - 'donor_probs': (L,) array if return_probs=True
        """
        out = self.forward(seq)

        acceptor_probs = out.acceptor_prob[0].cpu().numpy()
        donor_probs = out.donor_prob[0].cpu().numpy()

        result = {
            'acceptor_sites': list(np.where(acceptor_probs > threshold)[0]),
            'donor_sites': list(np.where(donor_probs > threshold)[0]),
        }

        if return_probs:
            result['acceptor_probs'] = acceptor_probs
            result['donor_probs'] = donor_probs

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
        Generate embeddings for DNA sequence(s) from penultimate layer.

        Parameters
        ----------
        seq : str or list of str
            Input DNA sequence(s).
        pool : {'mean', 'cls', 'tokens'}, default='mean'
            Pooling strategy:
            - 'mean': Average over all positions (returns 1D vector)
            - 'cls': Use first position (returns 1D vector)
            - 'tokens': Return all position embeddings (returns 2D array)
        return_numpy : bool, default=True
            If True, return numpy array; if False, return torch.Tensor.
        layer : int, optional
            Not used for SpliceAI (CNN has no transformer layers).
            Included for API compatibility.

        Returns
        -------
        embeddings : np.ndarray or torch.Tensor
            Shape depends on pool:
            - 'mean' or 'cls': (hidden_dim,) for single seq, (B, hidden_dim) for batch
            - 'tokens': (seq_len, hidden_dim) for single, list of arrays for batch
        """
        is_batch = isinstance(seq, (list, tuple))

        out = self.forward(seq, return_embeddings=True)

        if out.embeddings is None:
            # Fallback: use concatenated probabilities as features
            import warnings
            warnings.warn("No embedding layer found, using splice probabilities as features")
            emb = torch.stack([out.acceptor_prob, out.donor_prob, out.neither_prob], dim=-1)
        else:
            emb = out.embeddings  # (B, L, H)

        if not is_batch:
            emb = emb[0]  # (L, H)

            if pool == "tokens":
                result = emb
            elif pool == "cls":
                result = emb[0]  # (H,)
            elif pool == "mean":
                result = emb.mean(dim=0)  # (H,)
            else:
                raise ValueError(f"pool must be one of {{'mean', 'cls', 'tokens'}}, got {pool}")

            return result.cpu().numpy() if return_numpy else result.cpu()

        # Batch processing
        if pool == "tokens":
            pieces = []
            for b in range(emb.size(0)):
                piece = emb[b]  # (L, H)
                pieces.append(piece.cpu().numpy() if return_numpy else piece.cpu())
            return pieces

        if pool == "cls":
            result = emb[:, 0, :]  # (B, H)
        elif pool == "mean":
            result = emb.mean(dim=1)  # (B, H)
        else:
            raise ValueError(f"pool must be one of {{'mean', 'cls', 'tokens'}}, got {pool}")

        return result.cpu().numpy() if return_numpy else result.cpu()

    @torch.no_grad()
    def get_all_layer_features(
        self,
        seq: str,
        *,
        return_numpy: bool = False,
    ) -> List[Union[np.ndarray, torch.Tensor]]:
        """
        Get features from all convolutional layers.

        Parameters
        ----------
        seq : str
            Input DNA sequence.
        return_numpy : bool, default=False
            If True, return numpy arrays; if False, return torch.Tensors.

        Returns
        -------
        features : list
            List of feature tensors from each conv layer.
            Each element has shape (seq_len, channels).
        """
        x = _dna_to_onehot(self._normalize_seq(seq), self.device, self.dtype)

        features = []

        def make_hook(container):
            def hook_fn(module, inp, out):
                container.append(out.detach())
            return hook_fn

        hooks = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv1d):
                h = module.register_forward_hook(make_hook(features))
                hooks.append(h)

        try:
            _ = self.model(x)
        finally:
            for h in hooks:
                h.remove()

        # Convert from (1, C, L) to (L, C)
        result = []
        for f in features:
            f = f[0].permute(1, 0)  # (L, C)
            result.append(f.cpu().numpy() if return_numpy else f.cpu())

        return result
