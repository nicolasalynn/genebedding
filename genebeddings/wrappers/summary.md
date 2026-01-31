# Wrapper Summary

This document summarizes key information about each wrapper in the genebeddings package.

## AlphaGenomeWrapper

- **Model Source**: Kaggle or HuggingFace (`google/alphagenome`)
- **Requires Local Model States**: No (downloads from Kaggle/HF, requires license acceptance)
- **Framework**: JAX (not PyTorch)
- **Input Length**: Up to 1,048,576 bp (1M bp)
- **Tokenization**: One-hot encoding (A/C/G/T -> 4 channels)
- **Embedding Dimensions**: 1536 (1bp resolution), 3072 (128bp resolution)
- **Capabilities**:
  - `embed()` - Sequence embeddings at 1bp or 128bp resolution
  - `predict_tracks()` - Multi-modal genomic track predictions (RNA-seq, ATAC, DNase, ChIP, splice sites, contact maps)
  - `predict_variant()` - Variant effect predictions
  - `score_variant()` - Variant effect scoring
- **Notes**: Requires GPU (H100+ recommended). Uses `alphagenome_research` package.

## BorzoiWrapper

- **Model Source**: HuggingFace (`johahi/flashzoi-replicate-0`)
- **Requires Local Model States**: No
- **Input Length**: 524,288 bp (default, configurable). Sequences are padded/cropped.
- **Output Length**: 6,144 positions (default, inferred from model)
- **Tokenization**: One-hot encoding (A/C/G/T -> 4 channels)
- **Capabilities**:
  - `embed()` - Hidden representations
  - `predict_tracks()` - Genomic track predictions
  - `tracks()` - Alias for predict_tracks
  - `tracks_by_name()` - Named track retrieval
- **Notes**: Uses `borzoi_pytorch` package for model loading

## CaduceusWrapper

- **Model Source**: HuggingFace (`kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16`)
- **Requires Local Model States**: No
- **Input Length**: ~131k tokens
- **k-mer Size**: Inferred from tokenizer (often k=1, character-level)
- **Tokenization**: Standard Caduceus tokenizer
- **Capabilities**:
  - `embed()` - Sequence embeddings (supports batch)
  - `predict_nucleotides()` - Masked language model predictions
- **Notes**: Uses `trust_remote_code=True`

## ConformerWrapper (AnchoredNTLike)

- **Model Source**: Local model states (required)
- **Requires Local Model States**: Yes (tokenizer `.tkz` + checkpoint `.pt`)
- **Input Length**: Model-dependent
- **k-mer Size**: k=1 (base-anchored, non-overlapping)
- **Tokenization**: GenomicBPE tokenizer with anchoring
- **Capabilities**:
  - `embed()` - Sequence embeddings
  - `predict_nucleotides()` - Base-level nucleotide predictions
- **Notes**: Requires `anchor_tokenizer` and `conformer` modules. Supports DataParallel.

## ConvNovaWrapper

- **Model Source**: Optional local checkpoint
- **Requires Local Model States**: Optional
- **Input Length**: Model-dependent (handles long sequences)
- **k-mer Size**: k=1 (base-level via BaseDNATokenizer)
- **Tokenization**: Simple DNA tokenizer (A/C/G/T/N -> 0-4)
- **Capabilities**:
  - `embed()` - Sequence embeddings
  - `predict_nucleotides()` - Base-level nucleotide predictions
- **Notes**: Embedded CNNModel implementation (dilated convolutions). Supports DataParallel.

## DNABERTWrapper

- **Model Source**: HuggingFace (`zhihan1996/DNABERT-2-117M`)
- **Requires Local Model States**: No
- **Input Length**: Model-dependent (BPE tokenization)
- **k-mer Size**: Variable (BPE tokenization)
- **Tokenization**: DNABERT-2 BPE tokenizer
- **Capabilities**:
  - `embed()` - Sequence embeddings (supports batch)
  - `predict_nucleotides()` - BPE pattern-matching predictions
- **Notes**: Uses `trust_remote_code=True`

## Evo2Wrapper

- **Model Source**: Evo2 package (`evo2`)
- **Requires Local Model States**: No
- **Input Length**: Very long context (architecture-dependent)
- **k-mer Size**: k=1 (character-level)
- **Tokenization**: Character-level
- **Capabilities**:
  - `embed()` - Sequence embeddings (supports batch, configurable layers)
  - `predict_nucleotides()` - Logits extraction
  - `generate()` - Sequence generation
- **Notes**: Models: "1b", "7b", "40b". Supports bfloat16.

## GenomeNetWrapper

- **Model Source**: HuggingFace (various GenomeNet models)
- **Requires Local Model States**: No
- **k-mer Size**: k=1 (character-level)
- **Capabilities**:
  - `embed()` - Sequence embeddings (supports batch)
  - `predict_nucleotides()` - MLM-based predictions
- **Notes**: Uses standard HuggingFace MLM interface

## GPNMSAWrapper

- **Model Source**: HuggingFace (`songlab/gpn-msa-sapiens`)
- **Requires Local Model States**: No (but requires MSA data download)
- **Input Length**: Model-dependent
- **k-mer Size**: k=1 (character-level)
- **Tokenization**: GPN tokenizer with MSA auxiliary features
- **Capabilities**:
  - `embed_msa()` - Embeddings from pre-tokenized MSA data
  - `embed_region()` - Embeddings from genomic coordinates (auto-fetches MSA)
  - `predict_nucleotides()` - MSA-based nucleotide predictions
- **Notes**: `embed(seq)` raises NotImplementedError — use `embed_msa()` or `embed_region()`. Requires `gpn` package and 100-way vertebrate alignment data.

## HyenaDNAWrapper

- **Model Source**: HuggingFace (`LongSafari/hyenadna-*-seqlen-hf`)
- **Requires Local Model States**: No
- **Input Length**: Up to 1M bp (model-dependent: 1k, 32k, 160k, 450k, 1M)
- **k-mer Size**: k=1 (character-level)
- **Tokenization**: Character-level (A=7, C=8, G=9, T=10, N=11)
- **Capabilities**:
  - `embed()` - Sequence embeddings (supports batch, layer selection)
- **Notes**: State-space model (Hyena operator), not a transformer. Uses `trust_remote_code=True`.

## MutBERTWrapper

- **Model Source**: HuggingFace or local
- **Requires Local Model States**: Optional
- **k-mer Size**: k=1 (one-hot encoded input)
- **Tokenization**: One-hot encoding (soft/probabilistic input supported)
- **Capabilities**:
  - `embed()` - Sequence embeddings
  - `embed_soft()` - Embeddings from soft/probabilistic one-hot input
- **Notes**: Supports probabilistic/soft mutations via differentiable input.

## NTWrapper

- **Model Source**: HuggingFace (`InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`)
- **Requires Local Model States**: No
- **Input Length**: Long sequences (no truncation)
- **k-mer Size**: Inferred (typically k=6 for NT models, k=3 for smaller)
- **Tokenization**: Nucleotide Transformer tokenizer (k-mer based)
- **Capabilities**:
  - `embed()` - Sequence embeddings (supports batch)
  - `predict_nucleotides()` - Masked language model predictions
- **Notes**: Auto-detects 'N' positions if `positions=None`. Uses `attn_implementation="eager"`.

## RiNALMoWrapper

- **Model Source**: `rinalmo` package (`giga-v1` default)
- **Requires Local Model States**: No
- **Input Length**: Model-dependent
- **k-mer Size**: k=1 (character-level)
- **Tokenization**: RiNALMo alphabet tokenizer
- **Capabilities**:
  - `embed()` - Sequence embeddings (supports batch)
  - `predict_nucleotides()` - Nucleotide predictions
- **Notes**: RNA model; handles T->U conversion. `dna_output=True` by default.

## SpeciesLMWrapper

- **Model Source**: HuggingFace (`gagneurlab/SpeciesLM`, revision: `downstream_species_lm`)
- **Requires Local Model States**: No
- **Input Length**: ~512 tokens (auto-chunked for longer sequences)
- **k-mer Size**: k=6 (stride-1, overlapping)
- **Tokenization**: Whitespace-separated 6-mers with stride=1
- **Capabilities**:
  - `embed()` - Sequence embeddings (auto-chunking)
  - `predict_nucleotides()` - Per-base nucleotide probabilities
  - `acgt_probs()` - Full sequence probability matrix
- **Notes**: Overlapping window aggregation. Supports species conditioning.

## SpliceAIWrapper

- **Model Source**: `spliceai-pytorch` package
- **Requires Local Model States**: No
- **Input Length**: Model-dependent (CNN architecture)
- **Tokenization**: One-hot encoding
- **Capabilities**:
  - `embed()` - CNN feature embeddings
  - `predict_splice_sites()` - Splice site predictions (acceptor/donor probabilities)
  - `forward()` - Returns SpliceAIOutput with embeddings + splice predictions
- **Notes**: Returns `SpliceAIOutput` dataclass. Layer-wise feature extraction available.

## SpliceBertWrapper

- **Model Source**: HuggingFace or local (`genebeddings/assets/splicebert/`)
- **Requires Local Model States**: Optional (local assets available)
- **k-mer Size**: k=1 (character-level) for local, BPE for HF
- **Tokenization**: Character-level or BPE depending on model source
- **Capabilities**:
  - `embed()` - Sequence embeddings (supports batch)
  - `predict_nucleotides()` - MLM-based nucleotide predictions
  - `forward()` - Returns SpliceBertOutput with hidden states + logits
- **Notes**: Supports both RNA and DNA input. Returns `SpliceBertOutput` dataclass.

## Summary Table

| Wrapper | Model Source | Local States | Max Input | k-mer | Capabilities |
|---------|-------------|-------------|-----------|-------|-------------|
| AlphaGenomeWrapper | Kaggle/HF | No | 1M bp | N/A (one-hot) | embed, predict_tracks, predict_variant |
| BorzoiWrapper | HuggingFace | No | 524K bp | N/A (one-hot) | embed, predict_tracks |
| CaduceusWrapper | HuggingFace | No | ~131k tokens | k=1 | embed, predict_nucleotides |
| ConformerWrapper | Local | Yes | Model-dep. | k=1 | embed, predict_nucleotides |
| ConvNovaWrapper | Local | Optional | Model-dep. | k=1 | embed, predict_nucleotides |
| DNABERTWrapper | HuggingFace | No | Model-dep. | BPE | embed, predict_nucleotides |
| Evo2Wrapper | Package | No | Very long | k=1 | embed, predict_nucleotides, generate |
| GenomeNetWrapper | HuggingFace | No | Model-dep. | k=1 | embed, predict_nucleotides |
| GPNMSAWrapper | HuggingFace | No (MSA data) | Model-dep. | k=1 | embed_msa, embed_region, predict_nucleotides |
| HyenaDNAWrapper | HuggingFace | No | Up to 1M bp | k=1 | embed |
| MutBERTWrapper | HuggingFace | Optional | Model-dep. | k=1 | embed, embed_soft |
| NTWrapper | HuggingFace | No | Long | k=6 | embed, predict_nucleotides |
| RiNALMoWrapper | Package | No | Model-dep. | k=1 | embed, predict_nucleotides |
| SpeciesLMWrapper | HuggingFace | No | ~512 tokens | k=6 | embed, predict_nucleotides |
| SpliceAIWrapper | Package | No | Model-dep. | N/A (one-hot) | embed, predict_splice_sites |
| SpliceBertWrapper | HF/Local | Optional | Model-dep. | k=1/BPE | embed, predict_nucleotides |
