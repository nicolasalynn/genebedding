# hyenadna_wrapper.py
# HyenaDNA wrapper for genomic foundation models
# Standardized API: embed()
#
# HyenaDNA uses character-level tokenization and state-space models (Hyena),
# not transformers. Models are loaded from HuggingFace LongSafari org.

import json
import os
import re
import subprocess
from typing import Dict, Optional, List, Union, Literal

import numpy as np
import torch

# Support both package import and direct sys.path import
try:
    from .base_wrapper import BaseWrapper, PoolMode
except ImportError:
    from base_wrapper import BaseWrapper, PoolMode


# Model registry: short name -> (HuggingFace model name, context length)
HYENADNA_MODELS: Dict[str, tuple] = {
    # Flagship models
    "large-1m": ("hyenadna-large-1m-seqlen-hf", 1_000_000),
    "medium-450k": ("hyenadna-medium-450k-seqlen-hf", 450_000),
    "medium-160k": ("hyenadna-medium-160k-seqlen-hf", 160_000),

    # Smaller models
    "small-32k": ("hyenadna-small-32k-seqlen-hf", 32_000),
    "tiny-1k": ("hyenadna-tiny-1k-seqlen-hf", 1_000),

    # Alternative naming
    "large": ("hyenadna-large-1m-seqlen-hf", 1_000_000),
    "medium": ("hyenadna-medium-160k-seqlen-hf", 160_000),
    "small": ("hyenadna-small-32k-seqlen-hf", 32_000),
    "tiny": ("hyenadna-tiny-1k-seqlen-hf", 1_000),
}

HyenaDNAModelName = Literal[
    "large-1m", "medium-450k", "medium-160k", "small-32k", "tiny-1k",
    "large", "medium", "small", "tiny",
]


def list_available_models() -> List[str]:
    """Return list of available model short names."""
    return list(HYENADNA_MODELS.keys())


# ============================================================================
# HyenaDNA Model Loading Utilities (from HuggingFace)
# ============================================================================

def inject_substring(orig_str: str) -> str:
    """Hack to handle matching keys between models trained with and without
    gradient checkpointing."""
    # modify for mixer keys
    pattern = r"\.mixer"
    injection = ".mixer.layer"
    modified_string = re.sub(pattern, injection, orig_str)

    # modify for mlp keys
    pattern = r"\.mlp"
    injection = ".mlp.layer"
    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string


def load_weights(scratch_dict: dict, pretrained_dict: dict, checkpointing: bool = False) -> dict:
    """Loads pretrained (backbone only) weights into the scratch state dict.

    Args:
        scratch_dict: State dict from a newly initialized HyenaDNA model
        pretrained_dict: State dict from the pretrained checkpoint
        checkpointing: Whether gradient checkpoint flag was used in pretrained model

    Returns:
        State dict with pretrained weights loaded (head is scratch)
    """
    for key, value in scratch_dict.items():
        if 'backbone' in key:
            # the state dicts differ by one prefix, '.model', so we add that
            key_loaded = 'model.' + key
            # need to add an extra ".layer" in key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except KeyError:
                raise RuntimeError(f'Key mismatch in state dicts: {key_loaded} not found')

    return scratch_dict


class HyenaDNAWrapper(BaseWrapper):
    """
    HyenaDNA wrapper with standardized API.

    HyenaDNA is a long-range genomic foundation model based on state-space models
    (Hyena operator) rather than transformers. It uses character-level tokenization
    and can handle very long sequences (up to 1M bp).

    Implements BaseWrapper: embed()

    Parameters
    ----------
    model : str, default="medium-160k"
        Model to use. Can be:
        - A short name from registry (e.g., "large-1m", "medium-160k", "small-32k")
        - A full model name (e.g., "hyenadna-large-1m-seqlen-hf")
    device : str, optional
        Device to use. Defaults to CUDA if available, else MPS, else CPU.
    dtype : torch.dtype, default=torch.float32
        Data type for model weights.
    cache_dir : str, optional
        Directory to cache downloaded models. Defaults to ~/.cache/hyenadna

    Examples
    --------
    >>> wrapper = HyenaDNAWrapper(model="medium-160k")
    >>> wrapper = HyenaDNAWrapper(model="large-1m", dtype=torch.float16)

    >>> # Get embeddings
    >>> emb = wrapper.embed("ACGTACGT", pool="mean")

    >>> # Batch embeddings
    >>> embs = wrapper.embed(["ACGT", "GCTA"], pool="mean")
    """

    def __init__(
        self,
        model: str = "medium-160k",
        *,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        # Resolve model name and context length
        if model in HYENADNA_MODELS:
            model_name, self.max_length = HYENADNA_MODELS[model]
            self.model_name = model
        else:
            # Assume it's a full model name
            model_name = model
            self.model_name = model
            # Try to infer max length from name
            self.max_length = self._infer_max_length(model_name)

        # Device/dtype
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            )
        self.device = torch.device(device)
        self.dtype = dtype

        # Cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "hyenadna")
        self.cache_dir = cache_dir

        # Load model
        self.model = self._load_model(model_name)
        self.model = self.model.to(self.device).to(dtype).eval()

        # Character-level tokenization mapping
        self._char_to_id = {
            'A': 7, 'C': 8, 'G': 9, 'T': 10, 'N': 11,
            'a': 7, 'c': 8, 'g': 9, 't': 10, 'n': 11,
        }
        # Padding token
        self._pad_id = 4

    def _infer_max_length(self, model_name: str) -> int:
        """Infer max sequence length from model name."""
        if "1m" in model_name:
            return 1_000_000
        elif "450k" in model_name:
            return 450_000
        elif "160k" in model_name:
            return 160_000
        elif "32k" in model_name:
            return 32_000
        elif "1k" in model_name:
            return 1_000
        else:
            return 160_000  # default

    def _load_model(self, model_name: str):
        """Load HyenaDNA model from HuggingFace."""
        # Try to import the HyenaDNA model class
        try:
            from transformers import AutoModel, AutoConfig

            # HyenaDNA models on HuggingFace
            hf_path = f"LongSafari/{model_name}"

            # Try loading via transformers AutoModel first
            try:
                model = AutoModel.from_pretrained(
                    hf_path,
                    trust_remote_code=True,
                    torch_dtype=self.dtype,
                )
                return model
            except Exception:
                pass

            # Fallback: manual loading
            return self._load_model_manual(model_name)

        except ImportError:
            return self._load_model_manual(model_name)

    def _load_model_manual(self, model_name: str):
        """Manual loading of HyenaDNA model."""
        pretrained_model_path = os.path.join(self.cache_dir, model_name)

        # Download if not present
        if not os.path.isdir(pretrained_model_path):
            hf_url = f'https://huggingface.co/LongSafari/{model_name}'
            os.makedirs(self.cache_dir, exist_ok=True)

            print(f"Downloading HyenaDNA model from {hf_url}...")
            command = f'cd {self.cache_dir} && git lfs install && git clone {hf_url}'
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to download model: {result.stderr}")

        # Load config
        config_path = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Try to import the model class from the downloaded repo
        import sys
        sys.path.insert(0, pretrained_model_path)

        try:
            from standalone_hyenadna import HyenaDNAModel
        except ImportError:
            # Try alternative import
            try:
                from huggingface import HyenaDNAModel
            except ImportError:
                raise ImportError(
                    "Could not import HyenaDNAModel. Please ensure the model files are complete. "
                    f"Check {pretrained_model_path}"
                )
        finally:
            sys.path.pop(0)

        # Create model
        model = HyenaDNAModel(**config, use_head=False, n_classes=2)

        # Load weights
        weights_path = os.path.join(pretrained_model_path, 'weights.ckpt')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found at {weights_path}")

        loaded_ckpt = torch.load(weights_path, map_location='cpu')

        # Handle gradient checkpointing differences
        checkpointing = config.get("checkpoint_mixer", False)

        # Load weights
        state_dict = load_weights(
            model.state_dict(),
            loaded_ckpt['state_dict'],
            checkpointing=checkpointing
        )
        model.load_state_dict(state_dict)

        print(f"Loaded HyenaDNA model: {model_name}")
        return model

    def __repr__(self) -> str:
        return f"HyenaDNAWrapper(model='{self.model_name}', device={self.device}, max_length={self.max_length:,})"

    def _normalize_seq(self, seq: str) -> str:
        """Clean sequence to valid DNA characters."""
        return re.sub(r"[^ACGTNacgtn]", "N", seq or "").upper()

    def _tokenize(self, seq: str) -> torch.Tensor:
        """Character-level tokenization for HyenaDNA."""
        seq = self._normalize_seq(seq)
        # Truncate if needed
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]

        ids = [self._char_to_id.get(c, self._char_to_id['N']) for c in seq]
        return torch.tensor(ids, dtype=torch.long)

    def _tokenize_batch(self, seqs: List[str]) -> tuple:
        """Tokenize batch with padding."""
        seqs = [self._normalize_seq(s) for s in seqs]
        # Truncate
        seqs = [s[:self.max_length] for s in seqs]

        max_len = max(len(s) for s in seqs)

        batch_ids = []
        attention_masks = []

        for s in seqs:
            ids = [self._char_to_id.get(c, self._char_to_id['N']) for c in s]
            # Pad
            pad_len = max_len - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids = ids + [self._pad_id] * pad_len

            batch_ids.append(ids)
            attention_masks.append(mask)

        return (
            torch.tensor(batch_ids, dtype=torch.long),
            torch.tensor(attention_masks, dtype=torch.long),
        )

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
            Pooling strategy:
            - 'mean': Average over all positions
            - 'cls': Use first position embedding
            - 'tokens': Return all position embeddings
        return_numpy : bool, default=True
            If True, return numpy array; if False, return torch.Tensor
        layer : int, optional
            Which layer to extract. If None, uses last hidden state.
            Note: Layer extraction may not be supported for all HyenaDNA variants.
        """
        is_batch = isinstance(seq, (list, tuple))

        if not is_batch:
            # Single sequence
            input_ids = self._tokenize(seq).unsqueeze(0).to(self.device)

            # Forward pass
            out = self._forward(input_ids, layer=layer)  # (1, L, H)

            if pool == "tokens":
                emb = out[0]  # (L, H)
            elif pool == "cls":
                emb = out[0, 0]  # (H,) - first position
            elif pool == "mean":
                emb = out[0].mean(dim=0)  # (H,)
            else:
                raise ValueError("pool must be one of {'mean', 'cls', 'tokens'}")

            return emb.detach().cpu().numpy() if return_numpy else emb.detach().cpu()

        # Batch processing
        input_ids, attention_mask = self._tokenize_batch(list(seq))
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        out = self._forward(input_ids, layer=layer)  # (B, L, H)

        if pool == "tokens":
            # Variable lengths -> return list
            pieces = []
            for b in range(out.size(0)):
                Lb = int(attention_mask[b].sum().item())
                emb = out[b, :Lb]
                pieces.append(emb.detach().cpu().numpy() if return_numpy else emb.detach().cpu())
            return pieces

        if pool == "cls":
            emb = out[:, 0]  # (B, H)
        elif pool == "mean":
            # Masked mean
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            emb = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, H)
        else:
            raise ValueError("pool must be one of {'mean', 'cls', 'tokens'}")

        return emb.detach().cpu().numpy() if return_numpy else emb.detach().cpu()

    def _forward(self, input_ids: torch.Tensor, layer: Optional[int] = None) -> torch.Tensor:
        """Run forward pass and extract hidden states."""
        # HyenaDNA models have different interfaces depending on source

        # Try HuggingFace interface first
        if hasattr(self.model, 'forward'):
            try:
                # Standard HF interface
                out = self.model(input_ids, output_hidden_states=(layer is not None))

                if hasattr(out, 'last_hidden_state'):
                    if layer is None:
                        return out.last_hidden_state
                    return out.hidden_states[layer]

                # Some HyenaDNA variants return tuple
                if isinstance(out, tuple):
                    if layer is None:
                        return out[0]
                    if len(out) > 1 and isinstance(out[1], (list, tuple)):
                        return out[1][layer]
                    return out[0]

                # Direct tensor output
                if isinstance(out, torch.Tensor):
                    return out

            except Exception:
                pass

        # Try backbone interface (standalone model)
        if hasattr(self.model, 'backbone'):
            out = self.model.backbone(input_ids)
            if isinstance(out, tuple):
                return out[0]
            return out

        # Fallback: direct call
        out = self.model(input_ids)
        if isinstance(out, tuple):
            return out[0]
        return out

    def find_N_positions(self, seq: str) -> List[int]:
        """Indices where the input has 'N'."""
        return [i for i, c in enumerate(seq) if c.upper() == "N"]
