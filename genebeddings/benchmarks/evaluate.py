"""
Model factory functions for benchmarking.

All wrappers use self-contained local assets where available.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wrappers.borzoi_wrapper import BorzoiWrapper
from wrappers.caduceus_wrapper import CaduceusWrapper
from wrappers.conformer_wrapper import AnchoredNTLike
from wrappers.nt_wrapper import NTWrapper
from wrappers.rinalmo_wrapper import RiNALMoWrapper
from wrappers.convnova_wrapper import ConvNovaWrapper
from wrappers.dnabert_wrapper import DNABERTWrapper
from wrappers.specieslm_wrapper import SpeciesLMWrapper


def make_borzoi():
    """Factory that returns a BorzoiWrapper model (uses local assets)."""
    return BorzoiWrapper()


def make_caduceus():
    """Factory that returns a CaduceusWrapper model."""
    return CaduceusWrapper()


def make_conformer():
    """Factory that returns a Conformer model (uses local assets)."""
    return AnchoredNTLike()


def make_nt():
    """Factory that returns an NTWrapper model."""
    return NTWrapper()


def make_rinalmo():
    """Factory that returns a RiNALMoWrapper model."""
    return RiNALMoWrapper()


def make_convnova():
    """Factory that returns a ConvNovaWrapper model (uses local assets)."""
    return ConvNovaWrapper()


def make_dnabert():
    """Factory that returns a DNABERTWrapper model."""
    return DNABERTWrapper()


def make_specieslm():
    """Factory that returns a SpeciesLMWrapper model."""
    return SpeciesLMWrapper()


# Model factory registry
MODEL_FACTORIES = {
    "borzoi": make_borzoi,
    "caduceus": make_caduceus,
    "conformer": make_conformer,
    "nt": make_nt,
    "rinalmo": make_rinalmo,
    "convnova": make_convnova,
    "dnabert": make_dnabert,
    "specieslm": make_specieslm,
}


def get_model(name: str):
    """
    Get a model by name.

    Parameters
    ----------
    name : str
        Model name (one of: borzoi, caduceus, conformer, nt, rinalmo, convnova, dnabert, specieslm)

    Returns
    -------
    wrapper : BaseWrapper
        Initialized model wrapper
    """
    if name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_FACTORIES.keys())}")
    return MODEL_FACTORIES[name]()


def list_models():
    """Return list of available model names."""
    return list(MODEL_FACTORIES.keys())
