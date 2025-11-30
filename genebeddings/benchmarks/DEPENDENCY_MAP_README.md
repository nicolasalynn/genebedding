# Dependency Map Analysis

A clean, unified module for computing and visualizing dependency maps in genomic language models.

## Overview

Dependency maps show how mutations at position `i` (mutated position) affect predictions at position `j` (target position). This helps understand long-range dependencies and interactions in genomic sequences.

## Methods

### 1. Log-odds Method (`method="logprobs"`)
- **Formula**: `dep[i,j] = max_alt max_b |Δ log-odds(j,b)|`
- **Interpretation**: Maximum change in log-odds of any base at position j when mutating position i
- **Requirements**: Model must have `.probs(tokens)` method or return logits
- **Use case**: Understanding prediction confidence changes

### 2. Embeddings Method (`method="embeddings"`)
- **Formula**: `dep[i,j] = max_alt cosine_distance(emb_ref[j], emb_alt[j])`
- **Interpretation**: Maximum cosine distance between reference and mutated embeddings at position j
- **Requirements**: Model must have `.embedding(tokens)` method
- **Use case**: Understanding representation changes in embedding space

## Quick Start

```python
from dependency_map import compute_dependency_map, plot_dependency_map

# Define sequence
sequence = "ACGTACGT..." * 100  # Your DNA sequence

# Compute dependency map
dep = compute_dependency_map(
    sequence,           # DNA sequence or torch.Tensor of token IDs
    model,              # Your model wrapper
    method="logprobs",  # or "embeddings"
    range_start=0,      # Start position
    range_end=100,      # End position
)

# Plot the results
fig = plot_dependency_map(
    dep,
    sequence=sequence[0:100],
    title="Dependency Map",
    colorbar_label="Max |Δ log-odds|",
    save_path="dependency_map.png"
)
```

## API Reference

### `compute_dependency_map()`

Compute dependency map showing how mutations at position i affect position j.

**Parameters:**
- `sequence` (str or torch.Tensor): DNA sequence or tokenized sequence
- `model`: Model wrapper with `.probs()` or `.embedding()` method
- `method` (str): "logprobs" or "embeddings"
- `range_start` (int, optional): Start position for dependency map (default: 0)
- `range_end` (int, optional): End position (default: len(sequence))
- `vocab_size` (int): Vocabulary size (default: 5 for ACGTN)
- `target_base_ids` (tuple): Base IDs to consider (default: (0,1,2,3) for ACGT)
- `skip_diagonal` (bool): Zero out diagonal (default: True)
- `bfloat16_ok` (bool): Allow bfloat16 autocasting (default: True)
- `device` (str, optional): Device to run on

**Returns:**
- `dep` (np.ndarray): Dependency matrix of shape [W, W] where W = range_end - range_start

### `plot_dependency_map()`

Plot dependency map heatmap.

**Parameters:**
- `dep` (np.ndarray): Dependency matrix from `compute_dependency_map()`
- `sequence` (str, optional): DNA sequence for axis labels
- `title` (str): Plot title
- `cmap` (str): Colormap name (default: "viridis")
- `figsize` (tuple): Figure size (default: (10, 8))
- `vmin` (float, optional): Min value for colormap
- `vmax` (float, optional): Max value for colormap
- `show_sequence_labels` (bool): Show nucleotide labels (default: True)
- `colorbar_label` (str): Colorbar label
- `save_path` (str, optional): Path to save figure

**Returns:**
- `fig` (matplotlib.Figure): Figure object

## Examples

### Example 1: Basic Usage with Genomic Sequence

```python
from seqmat import SeqMat
from dependency_map import compute_dependency_map, plot_dependency_map

# Get genomic sequence
seq_mat = SeqMat.from_fasta("hg38", "chr1", 1000000, 1001000)
sequence = seq_mat.seq

# Compute dependency map for central 200bp region
dep = compute_dependency_map(
    sequence,
    model,
    method="logprobs",
    range_start=400,
    range_end=600,
)

# Plot
fig = plot_dependency_map(
    dep,
    sequence=sequence[400:600],
    title="chr1:1000400-1000600 Dependency Map",
    save_path="results/chr1_dep_map.png"
)
```

### Example 2: Compare Both Methods

```python
# Compute using both methods
dep_logprobs = compute_dependency_map(sequence, model, method="logprobs")
dep_embeddings = compute_dependency_map(sequence, model, method="embeddings")

# Plot side by side
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

im1 = ax1.imshow(dep_logprobs, cmap='viridis')
ax1.set_title("Log-odds Method")
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(dep_embeddings, cmap='plasma')
ax2.set_title("Embeddings Method")
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig("comparison.png")
```

### Example 3: Using Pre-tokenized Input

```python
import torch

# If you already have tokens (A=0, C=1, G=2, T=3, N=4)
tokens = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3] * 100)

dep = compute_dependency_map(
    tokens,  # Pass tokens directly
    model,
    method="embeddings",
    range_start=50,
    range_end=150,
)
```

### Example 4: Custom Colormap and Styling

```python
dep = compute_dependency_map(sequence, model, method="logprobs")

fig = plot_dependency_map(
    dep,
    sequence=sequence[:100],
    title="Custom Styled Dependency Map",
    cmap="RdYlBu_r",  # Red-Yellow-Blue reversed
    figsize=(12, 10),
    vmin=0,
    vmax=5.0,  # Custom scale
    colorbar_label="Custom Label",
    save_path="custom_map.png"
)
```

## Model Requirements

Your model wrapper must implement one or both of:

### For `method="logprobs"`:
```python
class YourModel(torch.nn.Module):
    def probs(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Return per-position nucleotide probabilities.

        Args:
            tokens: [B, L] token IDs

        Returns:
            probs: [B, L, V] probabilities (already softmaxed)
        """
        # Your implementation
        pass
```

### For `method="embeddings"`:
```python
class YourModel(torch.nn.Module):
    def embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Return per-position embeddings.

        Args:
            tokens: [B, L] token IDs

        Returns:
            embeddings: [B, L, D] embeddings
        """
        # Your implementation
        pass
```

## Token Mapping

Default mapping for DNA sequences:
- `A` → 0
- `C` → 1
- `G` → 2
- `T` → 3
- `N` → 4

## Performance Tips

1. **Use smaller ranges**: Computing dependency for large ranges (>200bp) can be slow
2. **Enable bfloat16**: Set `bfloat16_ok=True` for faster computation on compatible hardware
3. **Batch processing**: The code automatically batches alternative mutations for efficiency
4. **Skip diagonal**: Set `skip_diagonal=True` to zero out self-dependencies

## Interpretation Guide

### Reading the Heatmap

- **Rows (i)**: Mutated position - where the mutation occurs
- **Columns (j)**: Target position - where the effect is measured
- **Value [i,j]**: How much mutating position i affects position j

### Common Patterns

- **Diagonal band**: Local dependencies (nearby positions affect each other)
- **Off-diagonal spots**: Long-range dependencies
- **Symmetric patterns**: Bidirectional dependencies
- **Asymmetric patterns**: Directional dependencies (i→j ≠ j→i)

## Troubleshooting

### Error: "Model has no attribute 'probs'"
- Your model doesn't implement the required method
- Use `method="embeddings"` instead, or implement `.probs()` method

### Error: "Model has no attribute 'embedding'"
- Your model doesn't implement the required method
- Use `method="logprobs"` instead, or implement `.embedding()` method

### Low/zero dependency values
- This may indicate your model is untrained or the mutations don't affect predictions
- Try a different genomic region with known functional elements

### Memory issues
- Reduce the range size (use smaller range_end - range_start)
- Enable bfloat16: `bfloat16_ok=True`
- Process in chunks if needed

## Citation

If you use this code in your research, please cite appropriately.
