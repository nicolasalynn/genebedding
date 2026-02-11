"""
Genebeddings - Genomic foundation model embeddings library.

A unified interface for extracting embeddings from various genomic
foundation models (transformers, CNNs, state-space models).
"""

from .genebeddings import (
    __version__,
    DBMetadata,
    DependencyMapResult,
    EpistasisGeometry,
    EpistasisMetrics,
    SingleVariantGeometry,
    VariantEmbeddingDB,
)

from .delta_classifier import DeltaClassifier

__all__ = [
    "__version__",
    "DBMetadata",
    "DeltaClassifier",
    "DependencyMapResult",
    "EpistasisGeometry",
    "EpistasisMetrics",
    "SingleVariantGeometry",
    "VariantEmbeddingDB",
]
