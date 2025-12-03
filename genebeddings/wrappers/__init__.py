"""
Genomic foundation model wrappers with standardized APIs.

All wrappers inherit from BaseWrapper and provide:
- embed(seq, pool='mean'|'cls'|'tokens', return_numpy=True)

Optional capabilities (check with wrapper.supports_capability('capability_name')):
- predict_nucleotides(seq, positions, return_dict=True)  # For MLM models
- predict_tracks(seq)  # For genomic track prediction models

Usage:
------
>>> from genebeddings.wrappers import NTWrapper, BorzoiWrapper, Evo2Wrapper
>>>
>>> # Embeddings example
>>> nt = NTWrapper()
>>> embedding = nt.embed("ACGTACGT", pool="mean")  # Returns (hidden_dim,) numpy array
>>>
>>> # Nucleotide prediction example
>>> probs = nt.predict_nucleotides("ACGTACGT", positions=[0, 4])
>>> # Returns [{'A': 0.1, 'C': 0.2, 'G': 0.3, 'T': 0.4}, ...]
>>>
>>> # Evo2 example
>>> evo2 = Evo2Wrapper(model="7b")
>>> embedding = evo2.embed("ACGTACGT", pool="mean")
>>> generated = evo2.generate("ACGT", n_tokens=100)
>>>
>>> # Track prediction example
>>> borzoi = BorzoiWrapper()
>>> tracks = borzoi.predict_tracks("ACGT" * 131_072)  # Returns (num_tracks, length) numpy array
"""

# Support both package import and direct sys.path import
try:
    from .base_wrapper import BaseWrapper
    # Import all wrappers
    from .borzoi_wrapper import BorzoiWrapper
    from .caduceus_wrapper import CaduceusWrapper
    from .convnova_wrapper import ConvNovaWrapper
    from .evo2_wrapper import Evo2Wrapper
    from .nt_wrapper import NTWrapper
    from .rinalmo_wrapper import RiNALMoWrapper
    from .specieslm_wrapper import SpeciesLMWrapper
except ImportError:
    from base_wrapper import BaseWrapper
    # Import all wrappers
    from borzoi_wrapper import BorzoiWrapper
    from caduceus_wrapper import CaduceusWrapper
    from convnova_wrapper import ConvNovaWrapper
    from evo2_wrapper import Evo2Wrapper
    from nt_wrapper import NTWrapper
    from rinalmo_wrapper import RiNALMoWrapper
    from specieslm_wrapper import SpeciesLMWrapper

__all__ = [
    "BaseWrapper",
    "BorzoiWrapper",
    "CaduceusWrapper",
    "ConvNovaWrapper",
    "Evo2Wrapper",
    "NTWrapper",
    "RiNALMoWrapper",
    "SpeciesLMWrapper",
]
