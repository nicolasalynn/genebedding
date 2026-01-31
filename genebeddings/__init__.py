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

__all__ = [
    "__version__",
    "DBMetadata",
    "DependencyMapResult",
    "EpistasisGeometry",
    "EpistasisMetrics",
    "SingleVariantGeometry",
    "VariantEmbeddingDB",
]
