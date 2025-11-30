# """
# Genomic foundation model wrappers with standardized APIs.

# All wrappers inherit from BaseWrapper and provide:
# - embed(seq, pool='mean'|'cls'|'tokens', return_numpy=True)

# Optional capabilities (check with wrapper.supports_capability('capability_name')):
# - predict_nucleotides(seq, positions, return_dict=True)  # For MLM models
# - predict_tracks(seq)  # For genomic track prediction models

# Usage:
# ------
# >>> from genebeddings.wrappers import NTWrapper, BorzoiWrapper
# >>>
# >>> # Embeddings example
# >>> nt = NTWrapper()
# >>> embedding = nt.embed("ACGTACGT", pool="mean")  # Returns (hidden_dim,) numpy array
# >>>
# >>> # Nucleotide prediction example
# >>> probs = nt.predict_nucleotides("ACGTACGT", positions=[0, 4])
# >>> # Returns [{'A': 0.1, 'C': 0.2, 'G': 0.3, 'T': 0.4}, ...]
# >>>
# >>> # Track prediction example
# >>> borzoi = BorzoiWrapper()
# >>> tracks = borzoi.predict_tracks("ACGT" * 131_072)  # Returns (num_tracks, length) numpy array
# """

# # Support both package import and direct sys.path import
# try:
#     from .base_wrapper import BaseWrapper
#     # Import all wrappers
#     from .borzoi_wrapper import BorzoiWrapper
#     from .caduceus_wrapper import CaduceusWrapper
#     from .convnova_wrapper import ConvNovaWrapper
#     from .nt_wrapper import NTWrapper
#     from .rinalmo_wrapper import RiNALMoWrapper
#     from .specieslm_wrapper import SpeciesLMWrapper
# except ImportError:
#     from base_wrapper import BaseWrapper
#     # Import all wrappers
#     from borzoi_wrapper import BorzoiWrapper
#     from caduceus_wrapper import CaduceusWrapper
#     from convnova_wrapper import ConvNovaWrapper
#     from nt_wrapper import NTWrapper
#     from rinalmo_wrapper import RiNALMoWrapper
#     from specieslm_wrapper import SpeciesLMWrapper

# __all__ = [
#     "BaseWrapper",
#     "BorzoiWrapper",
#     "CaduceusWrapper",
#     "ConvNovaWrapper",
#     "NTWrapper",
#     "RiNALMoWrapper",
#     "SpeciesLMWrapper",
# ]
