# Wrapper Summary

This document summarizes key information about each wrapper in the genebeddings package.

## BorzoiWrapper

- **Model Source**: HuggingFace (`johahi/flashzoi-replicate-0`)
- **Requires Local Model States**: No
- **Input Length**: 
  - Required input length: 524,288 bp (default, configurable)
  - Sequences are padded/cropped to this length
- **Output Length**: 
  - Target output length: 6,144 positions (default, inferred from model)
  - Returns central `min(len(seq), target_len)` positions
- **Tokenization**: One-hot encoding (A/C/G/T -> 4 channels)
- **Capabilities**: 
  - `embed()` - Hidden representations
  - `tracks()` - Genomic track predictions (chromatin accessibility, TF binding, etc.)
- **Notes**: Uses `borzoi_pytorch` package for model loading

## CaduceusWrapper

- **Model Source**: HuggingFace (`kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16`)
- **Requires Local Model States**: No
- **Input Length**: 
  - Model max length: ~131k tokens (as indicated by model name)
  - Supports long sequences without truncation by default
- **k-mer Size**: 
  - Inferred from tokenizer (often k=1 for character-level tokenization)
  - Non-overlapping k-mers
- **Tokenization**: Standard Caduceus tokenizer (character-level or k-mer based)
- **Capabilities**: 
  - `embed()` - Sequence embeddings
  - `predict_nucleotides()` - Masked language model predictions
- **Notes**: Uses `trust_remote_code=True` for custom model architecture

## ConformerWrapper (AnchoredNTLike)

- **Model Source**: Local model states (required)
- **Requires Local Model States**: Yes
  - `tokenizer_path`: Path to `.tkz` tokenizer file (GenomicBPE format)
  - `checkpoint_path`: Path to `.pt` model checkpoint file
- **Input Length**: Limited by model architecture (ConformerGenomicLM)
- **k-mer Size**: k=1 (base-anchored, non-overlapping)
- **Tokenization**: GenomicBPE tokenizer with anchoring support
- **Capabilities**: 
  - `embed()` - Sequence embeddings
  - `predict_nucleotides()` - Base-level nucleotide predictions
- **Notes**: 
  - Requires `anchor_tokenizer` and `conformer` modules from genomenet project
  - Supports DataParallel for multi-GPU usage
  - Base-anchored tokenization (k=1)

## ConvNovaWrapper

- **Model Source**: Optional local checkpoint or defaults
- **Requires Local Model States**: Optional
  - `checkpoint_path`: Optional path to model checkpoint
  - `config_yaml`: Optional path to config YAML file
  - Can work with default architecture parameters if no checkpoint provided
- **Input Length**: Limited by model architecture (typically handles long sequences)
- **k-mer Size**: k=1 (base-level via BaseDNATokenizer)
- **Tokenization**: Simple DNA tokenizer (A/C/G/T/N -> 0-4)
- **Capabilities**: 
  - `embed()` - Sequence embeddings
  - `predict_nucleotides()` - Base-level nucleotide predictions
- **Notes**: 
  - Uses embedded CNNModel implementation (dilated convolutions)
  - Supports DataParallel for multi-GPU usage
  - Can work without checkpoint (uses random initialization)

## NTWrapper

- **Model Source**: HuggingFace (`InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`)
- **Requires Local Model States**: No
- **Input Length**: 
  - Model max length: Handles long sequences without truncation
  - No hard limit in wrapper (relies on model's capabilities)
- **k-mer Size**: 
  - Inferred from tokenizer (typically k=6 for NT models)
  - Non-overlapping k-mers
- **Tokenization**: Nucleotide Transformer tokenizer (k-mer based)
- **Capabilities**: 
  - `embed()` - Sequence embeddings (supports batch processing)
  - `predict_nucleotides()` - Masked language model predictions
- **Notes**: 
  - Uses `attn_implementation="eager"` for stability
  - Auto-detects 'N' positions if `positions` parameter is None
  - Supports batch embeddings

## RiNALMoWrapper

- **Model Source**: Pretrained model via `rinalmo` package (`giga-v1` default)
- **Requires Local Model States**: No (uses package's pretrained models)
- **Input Length**: Limited by model architecture
- **k-mer Size**: k=1 (character-level)
- **Tokenization**: RiNALMo alphabet tokenizer (character-level)
- **Capabilities**: 
  - `embed()` - Sequence embeddings (supports batch processing)
  - `predict_nucleotides()` - Nucleotide predictions
- **Notes**: 
  - Can handle RNA-only models (T->U conversion if model uses RNA vocab)
  - Supports both masked and no-mask prediction modes
  - `dna_output=True` by default (exposes 'T' even if model uses 'U')

## SpeciesLMWrapper

- **Model Source**: HuggingFace (`gagneurlab/SpeciesLM`, revision: `downstream_species_lm`)
- **Requires Local Model States**: No
- **Input Length**: 
  - Max model tokens: ~512 tokens
  - Uses window-based chunking for longer sequences
  - Window size: ~496 nucleotides (with safety margin)
- **k-mer Size**: k=6 (stride-1, overlapping k-mers)
- **Tokenization**: 
  - Whitespace-separated 6-mers with stride=1
  - Supports optional species proxy token
- **Capabilities**: 
  - `embed()` - Sequence embeddings (handles chunking automatically)
  - `predict_nucleotides()` - Per-base nucleotide probabilities
  - `acgt_probs()` - Full sequence probability matrix
- **Notes**: 
  - Automatically chunks long sequences into overlapping windows
  - Aggregates results across overlapping windows
  - Supports species conditioning via `species_proxy` parameter

## DNABertWrapper

- **Status**: Currently empty (no implementation)

## Summary Table

| Wrapper | Model Source | Local States Required | Input Length | k-mer Size | Capabilities |
|---------|--------------|------------------------|--------------|------------|--------------|
| BorzoiWrapper | HuggingFace | No | 524,288 bp (fixed) | N/A (one-hot) | embed, tracks |
| CaduceusWrapper | HuggingFace | No | ~131k tokens | k=1 (often) | embed, predict_nucleotides |
| ConformerWrapper | Local | Yes (required) | Model-dependent | k=1 | embed, predict_nucleotides |
| ConvNovaWrapper | Local/Optional | Optional | Model-dependent | k=1 | embed, predict_nucleotides |
| NTWrapper | HuggingFace | No | Long (no truncation) | k=6 (typical) | embed, predict_nucleotides |
| RiNALMoWrapper | Package | No | Model-dependent | k=1 | embed, predict_nucleotides |
| SpeciesLMWrapper | HuggingFace | No | ~512 tokens (chunked) | k=6 (stride-1) | embed, predict_nucleotides |