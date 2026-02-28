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

# Lazy imports - wrappers are only loaded when accessed
from .base_wrapper import BaseWrapper

__all__ = [
    "AlphaGenomeWrapper",
    "BaseWrapper",
    "BorzoiWrapper",
    "CaduceusWrapper",
    "ConvNovaWrapper",
    "DNABERTWrapper",
    "Evo2Wrapper",
    "GPNMSAWrapper",
    "HyenaDNAWrapper",
    "MutBERTWrapper",
    "NTWrapper",
    "OpenSpliceAIOutput",
    "OpenSpliceAIWrapper",
    "RiNALMoWrapper",
    "SpeciesLMWrapper",
    "SpliceAIWrapper",
    "SpliceAIOutput",
    "SpliceBertWrapper",
    "SpliceBertOutput",
]

# Mapping of wrapper names to their modules
_WRAPPER_MODULES = {
    "AlphaGenomeWrapper": "alphagenome_wrapper",
    "BorzoiWrapper": "borzoi_wrapper",
    "CaduceusWrapper": "caduceus_wrapper",
    "ConvNovaWrapper": "convnova_wrapper",
    "DNABERTWrapper": "dnabert_wrapper",
    "Evo2Wrapper": "evo2_wrapper",
    "GPNMSAWrapper": "gpn_msa_wrapper",
    "HyenaDNAWrapper": "hyenadna_wrapper",
    "MutBERTWrapper": "mutbert_wrapper",
    "NTWrapper": "nt_wrapper",
    "OpenSpliceAIOutput": "openspliceai_wrapper",
    "OpenSpliceAIWrapper": "openspliceai_wrapper",
    "RiNALMoWrapper": "rinalmo_wrapper",
    "SpeciesLMWrapper": "specieslm_wrapper",
    "SpliceAIWrapper": "spliceai_wrapper",
    "SpliceAIOutput": "spliceai_wrapper",
    "SpliceBertWrapper": "splicebert_wrapper",
    "SpliceBertOutput": "splicebert_wrapper",
}


def __getattr__(name):
    if name in _WRAPPER_MODULES:
        import importlib
        module = importlib.import_module(f".{_WRAPPER_MODULES[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
