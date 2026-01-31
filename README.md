# genebeddings

Unified interface for extracting embeddings from genomic foundation models.

## Overview

genebeddings provides:

- **Standardized wrappers** for 16 genomic foundation models (transformers, CNNs, state-space models, track predictors)
- **Geometric analysis** tools for single-variant and epistasis embeddings
- **Embedding storage** via SQLite key-value store
- **Benchmarking** utilities for pathogenicity prediction

## Installation

```bash
pip install -e .
```

Install with model-specific dependencies:

```bash
pip install -e ".[nt]"          # Nucleotide Transformer
pip install -e ".[borzoi]"      # Borzoi
pip install -e ".[alphagenome]" # AlphaGenome (requires JAX + GPU)
pip install -e ".[all]"         # Common models
```

## Quick Start

### Embeddings

```python
from genebeddings.wrappers import NTWrapper

model = NTWrapper()
embedding = model.embed("ACGTACGT" * 100, pool="mean")  # (hidden_dim,) numpy array
```

### Nucleotide Predictions

```python
probs = model.predict_nucleotides("ACGTNACGT", positions=[4])
# [{'A': 0.1, 'C': 0.2, 'G': 0.3, 'T': 0.4}]
```

### Track Predictions

```python
from genebeddings.wrappers import BorzoiWrapper

borzoi = BorzoiWrapper()
tracks = borzoi.predict_tracks("ACGT" * 131_072)  # (num_tracks, length) numpy array
```

### Variant Geometry

```python
from genebeddings import SingleVariantGeometry

geom = SingleVariantGeometry(wt_embedding, mut_embedding)
print(geom.cosine_distance, geom.euclidean_distance)
```

## Supported Models

| Wrapper | Architecture | Max Input | Capabilities |
|---------|-------------|-----------|-------------|
| AlphaGenomeWrapper | Encoder-Transformer-Decoder (JAX) | 1M bp | embed, tracks, variants |
| BorzoiWrapper | CNN (PyTorch) | 524K bp | embed, tracks |
| CaduceusWrapper | Bidirectional SSM | ~131K tokens | embed, nucleotides |
| DNABERTWrapper | Transformer (BPE) | Model-dep. | embed, nucleotides |
| Evo2Wrapper | SSM | Very long | embed, nucleotides, generate |
| GPNMSAWrapper | Transformer + MSA | Model-dep. | embed (MSA), nucleotides |
| HyenaDNAWrapper | Hyena SSM | Up to 1M bp | embed |
| NTWrapper | Transformer (k-mer) | Long | embed, nucleotides |
| RiNALMoWrapper | Transformer (RNA) | Model-dep. | embed, nucleotides |
| SpliceAIWrapper | CNN | Model-dep. | embed, splice sites |
| SpliceBertWrapper | Transformer | Model-dep. | embed, nucleotides |

See [wrappers/summary.md](genebeddings/wrappers/summary.md) for full details.

## Testing

```bash
python quick_test.py
```
