"""
Base wrapper interface for genomic foundation models.

This module defines the standard API that all model wrappers should implement.
Models can support different capabilities (embeddings, nucleotide predictions, tracks).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Literal
import numpy as np
import torch


PoolMode = Literal["mean", "cls", "tokens"]


class BaseWrapper(ABC):
    """
    Base class defining the standard API for all genomic model wrappers.

    All wrappers should implement:
    - embed(): Generate sequence embeddings

    Optional capabilities (implement if model supports):
    - predict_nucleotides(): Get nucleotide probabilities at positions
    - predict_tracks(): Get genomic track predictions (Borzoi-style)
    - supports_capability(): Check what the model can do
    """

    def __init__(self):
        """Initialize the wrapper. Subclasses should set up their models here."""
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None

    # ======================== Required Methods ========================

    @abstractmethod
    def embed(
        self,
        seq: str,
        *,
        pool: PoolMode = "mean",
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate embeddings for a DNA/RNA sequence.

        Parameters
        ----------
        seq : str
            Input DNA or RNA sequence (e.g., "ACGTACGT...")
        pool : {'mean', 'cls', 'tokens'}, default='mean'
            Pooling strategy:
            - 'mean': Average over all tokens (returns 1D vector)
            - 'cls': Use first/CLS token only (returns 1D vector)
            - 'tokens': Return all token embeddings (returns 2D array)
        return_numpy : bool, default=True
            If True, return numpy array; if False, return torch.Tensor

        Returns
        -------
        embeddings : np.ndarray or torch.Tensor
            Shape depends on pool:
            - 'mean' or 'cls': (hidden_dim,)
            - 'tokens': (num_tokens, hidden_dim)
        """
        pass

    # ======================== Optional Methods ========================

    def predict_nucleotides(
        self,
        seq: str,
        positions: List[int],
        *,
        return_dict: bool = True,
    ) -> Union[List[Dict[str, float]], np.ndarray]:
        """
        Predict nucleotide probabilities at specified positions.

        This method should be implemented by models that can predict
        nucleotide identities (masked language models).

        Parameters
        ----------
        seq : str
            Input DNA or RNA sequence
        positions : list of int
            0-based positions in the sequence to predict
        return_dict : bool, default=True
            If True, return list of dicts with keys 'A', 'C', 'G', 'T'
            If False, return numpy array of shape (len(positions), 4)

        Returns
        -------
        predictions : list of dict or np.ndarray
            If return_dict=True:
                List of dicts, one per position: {'A': p_A, 'C': p_C, 'G': p_G, 'T': p_T}
            If return_dict=False:
                Array of shape (len(positions), 4) with columns [A, C, G, T]

        Raises
        ------
        NotImplementedError
            If the model does not support nucleotide prediction
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support nucleotide prediction. "
            f"Supported capabilities: {self.get_capabilities()}"
        )

    def predict_tracks(
        self,
        seq: str,
    ) -> np.ndarray:
        """
        Predict genomic tracks (e.g., chromatin accessibility, TF binding).

        This method should be implemented by models that predict continuous
        genomic signals across positions (e.g., Borzoi, Enformer).

        Parameters
        ----------
        seq : str
            Input DNA sequence

        Returns
        -------
        tracks : np.ndarray
            Shape (num_tracks, num_positions)
            Predictions for each track at each position

        Raises
        ------
        NotImplementedError
            If the model does not support track prediction
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support track prediction. "
            f"Supported capabilities: {self.get_capabilities()}"
        )

    # ======================== Capability Discovery ========================

    def supports_capability(self, capability: str) -> bool:
        """
        Check if this wrapper supports a specific capability.

        Parameters
        ----------
        capability : str
            One of: 'embed', 'predict_nucleotides', 'predict_tracks'

        Returns
        -------
        supported : bool
        """
        capability_map = {
            'embed': True,  # all models support this
            'predict_nucleotides': self._implements_predict_nucleotides(),
            'predict_tracks': self._implements_predict_tracks(),
        }
        return capability_map.get(capability, False)

    def get_capabilities(self) -> List[str]:
        """
        Get list of all capabilities this wrapper supports.

        Returns
        -------
        capabilities : list of str
            List of capability names
        """
        all_caps = ['embed', 'predict_nucleotides', 'predict_tracks']
        return [cap for cap in all_caps if self.supports_capability(cap)]

    def _implements_predict_nucleotides(self) -> bool:
        """Check if predict_nucleotides is actually implemented."""
        try:
            # Try calling with dummy data - if it raises NotImplementedError, not supported
            method = self.__class__.predict_nucleotides
            # Check if it's overridden from base class
            return method is not BaseWrapper.predict_nucleotides
        except Exception:
            return False

    def _implements_predict_tracks(self) -> bool:
        """Check if predict_tracks is actually implemented."""
        try:
            method = self.__class__.predict_tracks
            return method is not BaseWrapper.predict_tracks
        except Exception:
            return False

    # ======================== Utility Methods ========================

    def __repr__(self) -> str:
        """String representation showing capabilities."""
        caps = ", ".join(self.get_capabilities())
        return f"{self.__class__.__name__}(capabilities=[{caps}])"
