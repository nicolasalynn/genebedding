# gpn_msa_wrapper.py
# GPN-MSA wrapper for genomic foundation models
# Standardized API: embed(), predict_nucleotides()
#
# GPN-MSA uses multiple sequence alignment (MSA) data from 100-way vertebrate
# alignment as auxiliary features alongside the human sequence.
# Models are from the songlab HuggingFace org.

import re
from typing import Dict, Optional, List, Union, Literal

import numpy as np
import torch
from transformers import AutoModel, AutoModelForMaskedLM

# Support both package import and direct sys.path import
try:
    from .base_wrapper import BaseWrapper, PoolMode
except ImportError:
    from base_wrapper import BaseWrapper, PoolMode

BASES = ("A", "C", "G", "T")

# Model registry
GPN_MSA_MODELS: Dict[str, str] = {
    "sapiens": "songlab/gpn-msa-sapiens",
    "human": "songlab/gpn-msa-sapiens",  # alias
}

GPNMSAModelName = Literal["sapiens", "human"]


def list_available_models() -> List[str]:
    """Return list of available model short names."""
    return list(GPN_MSA_MODELS.keys())


class GPNMSAWrapper(BaseWrapper):
    """
    GPN-MSA wrapper with standardized API.

    GPN-MSA is a genomic foundation model that uses multiple sequence alignment
    (MSA) data from 100-way vertebrate alignment as auxiliary features. This
    provides evolutionary context for predictions.

    Implements BaseWrapper: embed(), predict_nucleotides()

    Parameters
    ----------
    model : str, default="sapiens"
        Model to use. Can be:
        - A short name from registry (e.g., "sapiens")
        - A full HuggingFace model ID (e.g., "songlab/gpn-msa-sapiens")
    msa_path : str, optional
        Path to the MSA data (zarr archive). If None, uses default HuggingFace path.
        For faster queries, download locally from:
        https://huggingface.co/datasets/songlab/multiz100way
    device : str, optional
        Device to use. Defaults to CUDA if available, else MPS, else CPU.
    dtype : torch.dtype, default=torch.float32
        Data type for model weights.
    load_mlm : bool, default=True
        Whether to load the MLM head for nucleotide prediction.

    Examples
    --------
    >>> wrapper = GPNMSAWrapper(model="sapiens")

    >>> # Get embeddings using genomic coordinates (requires MSA data)
    >>> emb = wrapper.embed_region("6", 31575665, 31575793, strand="+", pool="mean")

    >>> # Get embeddings from pre-tokenized MSA tensor
    >>> emb = wrapper.embed(msa_tensor, pool="mean")

    >>> # Predict nucleotides at masked positions
    >>> probs = wrapper.predict_nucleotides_region("6", 31575665, 31575793, positions=[76])

    Notes
    -----
    GPN-MSA requires MSA data for embedding. The input format is:
    - input_ids: Human sequence tokens (B, L)
    - aux_features: MSA tokens from other species (B, L, 89)

    For direct sequence embedding without MSA, consider using other wrappers.
    """

    def __init__(
        self,
        model: str = "sapiens",
        *,
        msa_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        load_mlm: bool = True,
    ):
        super().__init__()

        # Resolve model ID
        if model in GPN_MSA_MODELS:
            model_id = GPN_MSA_MODELS[model]
            self.model_name = model
        else:
            model_id = model
            self.model_name = model

        self.model_id = model_id

        # Device/dtype
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            )
        self.device = torch.device(device)
        self.dtype = dtype

        # MSA path
        if msa_path is None:
            msa_path = "zip:///::https://huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip"
        self.msa_path = msa_path
        self._genome_msa = None  # Lazy load

        # Load tokenizer
        try:
            from gpn.data import Tokenizer
            self.tokenizer = Tokenizer()
        except ImportError:
            raise ImportError(
                "GPN-MSA requires the gpn package. Install with:\n"
                "pip install git+https://github.com/songlab-cal/gpn.git"
            )

        # Load models
        self.model = AutoModel.from_pretrained(model_id).to(self.device).to(dtype).eval()

        if load_mlm:
            self.mlm = AutoModelForMaskedLM.from_pretrained(model_id).to(self.device).to(dtype).eval()
            self.mask_id = self.tokenizer.mask_token_id()
        else:
            self.mlm = None
            self.mask_id = None

    @property
    def genome_msa(self):
        """Lazy load GenomeMSA (can take a minute or two on first access)."""
        if self._genome_msa is None:
            try:
                from gpn.data import GenomeMSA
                print(f"Loading GenomeMSA from {self.msa_path}...")
                self._genome_msa = GenomeMSA(self.msa_path)
                print("GenomeMSA loaded.")
            except ImportError:
                raise ImportError(
                    "GPN-MSA requires the gpn package. Install with:\n"
                    "pip install git+https://github.com/songlab-cal/gpn.git"
                )
        return self._genome_msa

    def __repr__(self) -> str:
        return f"GPNMSAWrapper(model='{self.model_name}', device={self.device})"

    def get_msa(
        self,
        chrom: str,
        start: int,
        end: int,
        strand: str = "+",
        tokenize: bool = True,
    ) -> np.ndarray:
        """
        Get MSA data for a genomic region.

        Parameters
        ----------
        chrom : str
            Chromosome (e.g., "6" or "chr6")
        start : int
            Start position (0-based)
        end : int
            End position
        strand : str
            Strand ("+" or "-")
        tokenize : bool
            Whether to return tokenized data

        Returns
        -------
        msa : np.ndarray
            Shape (L, 90) if tokenize=True, (L, 90) of bytes if tokenize=False
            First column is human, remaining 89 are other species
        """
        # Normalize chromosome
        chrom = chrom.replace("chr", "")
        return self.genome_msa.get_msa(chrom, start, end, strand=strand, tokenize=tokenize)

    def _prepare_inputs(self, msa: np.ndarray) -> tuple:
        """
        Prepare model inputs from MSA array.

        Parameters
        ----------
        msa : np.ndarray
            MSA data of shape (L, 90) - tokenized

        Returns
        -------
        input_ids : torch.Tensor
            Human sequence tokens (1, L)
        aux_features : torch.Tensor
            Other species tokens (1, L, 89)
        """
        # Convert to tensor
        if isinstance(msa, np.ndarray):
            msa = torch.tensor(np.expand_dims(msa, 0).astype(np.int64))
        elif isinstance(msa, torch.Tensor) and msa.dim() == 2:
            msa = msa.unsqueeze(0)

        # Split human from other species
        input_ids = msa[:, :, 0].to(self.device)  # (B, L)
        aux_features = msa[:, :, 1:].to(self.device)  # (B, L, 89)

        return input_ids, aux_features

    @torch.no_grad()
    def embed(
        self,
        msa: Union[np.ndarray, torch.Tensor],
        *,
        pool: PoolMode = "mean",
        return_numpy: bool = True,
        layer: Optional[int] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate embeddings from pre-tokenized MSA data.

        Parameters
        ----------
        msa : np.ndarray or torch.Tensor
            Tokenized MSA data of shape (L, 90) or (B, L, 90)
            First column/dim is human, remaining 89 are other species
        pool : {'mean', 'cls', 'tokens'}, default='mean'
            Pooling strategy
        return_numpy : bool, default=True
            If True, return numpy array; if False, return torch.Tensor
        layer : int, optional
            Which layer to extract. If None, uses last hidden state.

        Returns
        -------
        embeddings : np.ndarray or torch.Tensor
            Shape depends on pool:
            - 'mean' or 'cls': (hidden_dim,) or (B, hidden_dim)
            - 'tokens': (L, hidden_dim) or (B, L, hidden_dim)
        """
        input_ids, aux_features = self._prepare_inputs(msa)

        # Forward pass
        out = self.model(
            input_ids=input_ids,
            aux_features=aux_features,
            output_hidden_states=(layer is not None),
        )

        if layer is not None:
            hidden = out.hidden_states[layer]
        else:
            hidden = out.last_hidden_state  # (B, L, H)

        # Pool
        if pool == "tokens":
            emb = hidden.squeeze(0) if hidden.size(0) == 1 else hidden
        elif pool == "cls":
            emb = hidden[:, 0]  # (B, H)
            if emb.size(0) == 1:
                emb = emb.squeeze(0)
        elif pool == "mean":
            emb = hidden.mean(dim=1)  # (B, H)
            if emb.size(0) == 1:
                emb = emb.squeeze(0)
        else:
            raise ValueError("pool must be one of {'mean', 'cls', 'tokens'}")

        return emb.detach().cpu().numpy() if return_numpy else emb.detach().cpu()

    @torch.no_grad()
    def embed_region(
        self,
        chrom: str,
        start: int,
        end: int,
        strand: str = "+",
        *,
        pool: PoolMode = "mean",
        return_numpy: bool = True,
        layer: Optional[int] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate embeddings for a genomic region using MSA data.

        Parameters
        ----------
        chrom : str
            Chromosome (e.g., "6" or "chr6")
        start : int
            Start position (0-based)
        end : int
            End position
        strand : str
            Strand ("+" or "-")
        pool : {'mean', 'cls', 'tokens'}, default='mean'
            Pooling strategy
        return_numpy : bool, default=True
            If True, return numpy array
        layer : int, optional
            Which layer to extract

        Returns
        -------
        embeddings : np.ndarray or torch.Tensor
        """
        msa = self.get_msa(chrom, start, end, strand=strand, tokenize=True)
        return self.embed(msa, pool=pool, return_numpy=return_numpy, layer=layer)

    @torch.no_grad()
    def predict_nucleotides(
        self,
        msa: Union[np.ndarray, torch.Tensor],
        positions: List[int],
        *,
        return_dict: bool = True,
    ) -> Union[List[Dict[str, float]], np.ndarray]:
        """
        Predict nucleotide probabilities at specified positions.

        Parameters
        ----------
        msa : np.ndarray or torch.Tensor
            Tokenized MSA data of shape (L, 90)
        positions : list of int
            0-based positions to predict
        return_dict : bool, default=True
            If True, return list of dicts with keys 'A', 'C', 'G', 'T'

        Returns
        -------
        predictions : list of dict or np.ndarray
        """
        if self.mlm is None:
            raise NotImplementedError("MLM head not loaded. Initialize with load_mlm=True")

        input_ids, aux_features = self._prepare_inputs(msa)
        original_input_ids = input_ids.clone()

        results = []
        nucleotide_indices = [self.tokenizer.vocab.index(nc) for nc in BASES]

        for pos in positions:
            # Mask the position
            input_ids_masked = original_input_ids.clone()
            input_ids_masked[0, pos] = self.mask_id

            # Forward pass
            logits = self.mlm(
                input_ids=input_ids_masked,
                aux_features=aux_features,
            ).logits  # (1, L, 6)

            # Get logits for nucleotides at the masked position
            pos_logits = logits[0, pos, nucleotide_indices]
            probs = torch.softmax(pos_logits, dim=0).cpu().numpy()

            if return_dict:
                results.append({b: float(p) for b, p in zip(BASES, probs)})
            else:
                results.append(probs)

        if return_dict:
            return results

        return np.array(results, dtype=np.float32)

    @torch.no_grad()
    def predict_nucleotides_region(
        self,
        chrom: str,
        start: int,
        end: int,
        positions: List[int],
        strand: str = "+",
        *,
        return_dict: bool = True,
    ) -> Union[List[Dict[str, float]], np.ndarray]:
        """
        Predict nucleotide probabilities at positions within a genomic region.

        Parameters
        ----------
        chrom : str
            Chromosome
        start : int
            Region start position
        end : int
            Region end position
        positions : list of int
            0-based positions within the region to predict
        strand : str
            Strand ("+" or "-")
        return_dict : bool
            If True, return list of dicts

        Returns
        -------
        predictions : list of dict or np.ndarray
        """
        msa = self.get_msa(chrom, start, end, strand=strand, tokenize=True)
        return self.predict_nucleotides(msa, positions, return_dict=return_dict)

    def find_N_positions(self, msa: np.ndarray) -> List[int]:
        """Find positions with gaps/unknown in human sequence."""
        # In tokenized MSA, 0 is gap/unknown
        human_seq = msa[:, 0] if msa.ndim == 2 else msa[0, :, 0]
        return [i for i, v in enumerate(human_seq) if v == 0]
