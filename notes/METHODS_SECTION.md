# Methods

## Geometric framework for detecting epistasis in foundation model embeddings

### Overview

We embed each mutation pair's four genotypes — wild-type (WT), two single mutants (M1, M2), and double mutant (M12) — into the representation space of a genomic foundation model and measure non-additivity in the resulting embedding geometry. The core intuition is that if two mutations interact (i.e., are epistatic), the double-mutant embedding will deviate from the position predicted by the sum of the two single-mutation effects.

We additionally introduce embedding perturbation maps — token-level cosine distance matrices that capture how mutating one position changes the representation of every other position — as a complementary, structure-aware readout that requires no MLM head.

---

## 1. Notation and setup

Let $f_\theta$ denote a pre-trained genomic foundation model that maps a nucleotide sequence $\mathbf{s}$ to a sequence of token embeddings $\{h_1, h_2, \ldots, h_T\} \in \mathbb{R}^{T \times d}$, where $T$ is the number of tokens and $d$ is the hidden dimension. A mean-pooled sequence embedding is

$$\bar{h}(\mathbf{s}) = \frac{1}{T} \sum_{t=1}^{T} h_t(\mathbf{s})$$

For a pair of variants at genomic positions $i$ and $j$, we construct four sequences that differ only at those positions:

| Sequence | Position $i$ | Position $j$ |
|----------|-------------|-------------|
| WT ($\mathbf{s}_\text{wt}$) | $\text{ref}_i$ | $\text{ref}_j$ |
| M1 ($\mathbf{s}_1$) | $\text{alt}_i$ | $\text{ref}_j$ |
| M2 ($\mathbf{s}_2$) | $\text{ref}_i$ | $\text{alt}_j$ |
| M12 ($\mathbf{s}_{12}$) | $\text{alt}_i$ | $\text{alt}_j$ |

Each sequence is embedded: $\bar{h}_\text{wt} = \bar{h}(\mathbf{s}_\text{wt})$, and similarly for $\bar{h}_1$, $\bar{h}_2$, $\bar{h}_{12}$.

---

## 2. Epistasis geometry metrics

### 2.1 Effect vectors

We define single-mutation effect vectors and the observed double-mutation effect:

$$\mathbf{v}_1 = \bar{h}_1 - \bar{h}_\text{wt}, \quad \mathbf{v}_2 = \bar{h}_2 - \bar{h}_\text{wt}, \quad \mathbf{v}_{12}^\text{obs} = \bar{h}_{12} - \bar{h}_\text{wt}$$

Under an additive (non-epistatic) model, the expected double-mutant effect is:

$$\mathbf{v}_{12}^\text{exp} = \mathbf{v}_1 + \mathbf{v}_2$$

The epistasis residual measures deviation from additivity:

$$\boldsymbol{\varepsilon} = \mathbf{v}_{12}^\text{obs} - \mathbf{v}_{12}^\text{exp} = \bar{h}_{12} - \bar{h}_1 - \bar{h}_2 + \bar{h}_\text{wt}$$

### 2.2 Raw epistasis magnitude

$$R_\text{raw} = \|\boldsymbol{\varepsilon}\|_2$$

This is the absolute magnitude of deviation from additivity in embedding space.

### 2.3 Normalized epistasis (epi_R_singles)

To compare across mutation pairs with different single-mutation effect sizes, we normalize by the scale of the single-mutation effects:

$$R_\text{singles} = \frac{\|\boldsymbol{\varepsilon}\|_2}{\sqrt{\|\mathbf{v}_1\|_2^2 + \|\mathbf{v}_2\|_2^2} + \epsilon}$$

where $\epsilon = 10^{-20}$ prevents division by zero. This metric is analogous to a signal-to-noise ratio: it measures the epistatic residual relative to the individual mutation magnitudes, allowing comparison across pairs where one mutation may be nearly silent and another highly disruptive.

### 2.4 Magnitude ratio

$$\rho = \frac{\|\mathbf{v}_{12}^\text{obs}\|_2}{\|\mathbf{v}_{12}^\text{exp}\|_2 + \epsilon}$$

Values $\rho < 1$ indicate sub-additivity (the double mutant is closer to wild-type than expected), and $\rho > 1$ indicate super-additivity. The log magnitude ratio $\log \rho$ symmetrizes the scale around zero.

### 2.5 Directional metrics

The cosine similarity between single-mutation effect vectors:

$$\cos(\mathbf{v}_1, \mathbf{v}_2) = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \, \|\mathbf{v}_2\|}$$

measures whether two mutations push the embedding in the same direction (+1), opposite directions (−1), or orthogonally (0).

The direction of epistatic deviation relative to the additive expectation:

$$\cos_\text{exp→obs} = \frac{\mathbf{v}_{12}^\text{exp} \cdot \boldsymbol{\varepsilon}}{\|\mathbf{v}_{12}^\text{exp}\| \, \|\boldsymbol{\varepsilon}\|}$$

Values near +1 indicate synergistic epistasis (the double mutant overshoots the additive prediction away from wild-type), and values near −1 indicate compensatory epistasis (the double mutant undershoots, falling back toward wild-type).

---

## 3. Mahalanobis calibration against population-level background

### 3.1 Motivation

Raw Euclidean metrics in embedding space are not comparable across models with different dimensionalities, scales, and anisotropies. To enable cross-model comparison and statistical interpretation, we calibrate epistasis residuals against an empirical null distribution derived from a population-level background set.

### 3.2 Null distribution

We use a set of common variant pairs from the 1000 Genomes Project (chromosome 12, hereafter okgp_chr12) as a background. For each pair in the null set, we compute the epistasis residual vector $\boldsymbol{\varepsilon}_k$ as in Section 2.1 and collect $N_\text{null}$ residuals into a matrix $\mathbf{E} \in \mathbb{R}^{N_\text{null} \times d}$.

### 3.3 Covariance estimation with Ledoit-Wolf shrinkage

Because the embedding dimension $d$ is typically large (640–5120 depending on model) relative to the number of null pairs, the sample covariance matrix is poorly conditioned. We use the Ledoit-Wolf shrinkage estimator, which computes an optimal linear combination of the sample covariance and a structured target (scaled identity):

$$\hat{\Sigma} = (1 - \alpha^*) \, \hat{\Sigma}_\text{sample} + \alpha^* \, \mu \mathbf{I}$$

where $\alpha^* \in [0, 1]$ is the analytically optimal shrinkage intensity and $\mu = \text{tr}(\hat{\Sigma}_\text{sample}) / d$. We add a small ridge $\lambda = 10^{-6}$ before inversion:

$$\hat{\Sigma}^{-1} = (\hat{\Sigma} + \lambda \mathbf{I})^{-1}$$

### 3.4 Mahalanobis epistasis distance

The calibrated epistasis score for a mutation pair with residual $\boldsymbol{\varepsilon}$ is:

$$d_\text{Mahal}(\boldsymbol{\varepsilon}) = \sqrt{\boldsymbol{\varepsilon}^\top \hat{\Sigma}^{-1} \boldsymbol{\varepsilon}}$$

This measures how many "standard deviations" the epistasis residual lies from zero in the null distribution's geometry, accounting for the correlation structure of the embedding space. We similarly compute calibrated magnitudes for the observed and expected double-mutant effects:

$$d_\text{Mahal}^\text{obs} = \sqrt{(\mathbf{v}_{12}^\text{obs})^\top \hat{\Sigma}^{-1} \mathbf{v}_{12}^\text{obs}}, \quad d_\text{Mahal}^\text{exp} = \sqrt{(\mathbf{v}_{12}^\text{exp})^\top \hat{\Sigma}^{-1} \mathbf{v}_{12}^\text{exp}}$$

and their ratio $d_\text{Mahal}^\text{obs} / d_\text{Mahal}^\text{exp}$, which is the Mahalanobis-calibrated analog of the magnitude ratio from Section 2.4.

### 3.5 Distance-binned calibration

Epistasis signatures may vary systematically with genomic distance between variants. To account for this, we optionally stratify the null distribution into distance bins (0–1 kb, 1–10 kb, 10–100 kb, >100 kb) and fit separate covariance matrices per bin, applying each bin's $\hat{\Sigma}^{-1}$ only to test pairs in the corresponding distance range.

---

## 4. Embedding perturbation maps

### 4.1 Motivation

Epistasis geometry (Section 2) uses mean-pooled embeddings to measure global non-additivity between two mutations. Embedding perturbation maps instead operate at the token level, measuring how mutating a single position changes the representation at every other position. This produces a position × position matrix analogous to a contact map, capturing pairwise dependency structure directly from the embedding space — without requiring an MLM head, fine-tuning, or masked inference.

### 4.2 Algorithm

Given a sequence $\mathbf{s}$ of length $N$ and a set of target positions $\mathcal{P} = \{p_1, \ldots, p_N\}$:

1. **Baseline embedding.** Compute token-level embeddings $\{h_t^\text{wt}\}_{t=1}^T$ from the unmodified sequence.

2. **Token alignment.** Identify the offset $\delta$ such that token index $\delta + p$ corresponds to sequence position $p$. We detect this by mutating the first position, embedding the mutant, and finding the token with the largest L2 shift: $\delta = \arg\max_t \|h_t^\text{wt} - h_t^\text{mut}\|_2$. For short sequences where attention effects on special tokens (EOS/CLS) can exceed the signal at the mutated position, we restrict the search to plausible offsets $\delta \in [0, T - N]$.

3. **Perturbation.** For each query position $i \in \mathcal{P}$ with reference nucleotide $r_i$, create three mutant sequences $\mathbf{s}^{i \to b}$ for $b \in \{A, C, G, T\} \setminus \{r_i\}$. Embed each mutant to obtain token embeddings $\{h_t^{i \to b}\}$.

4. **Distance computation.** For each mutant $b$ at position $i$, compute the cosine distance at every target position $j$:

$$d_{ij}^{(b)} = 1 - \frac{h_{\delta+j}^\text{wt} \cdot h_{\delta+j}^{i \to b}}{\|h_{\delta+j}^\text{wt}\| \, \|h_{\delta+j}^{i \to b}\|}$$

5. **Aggregation across mutations.** The perturbation score for the pair $(i, j)$ is:

$$D_{ij} = \max_{b \,\in\, \{A,C,G,T\} \setminus \{r_i\}} d_{ij}^{(b)}$$

We also evaluate mean aggregation: $D_{ij} = \frac{1}{3}\sum_b d_{ij}^{(b)}$.

6. **Symmetrization.** The final matrix is $D_{ij}^{\text{sym}} = \max(D_{ij}, D_{ji})$.

### 4.3 Complexity

The method requires $1 + 3N$ forward passes (one baseline plus three mutants per position) — the same cost as nucleotide dependency (Section 5). It requires only the encoder's hidden states, making it applicable to any model with a `pool='tokens'` interface, including models without an MLM head.

---

## 5. Nucleotide dependency maps (Da Silva et al. baseline)

As a baseline, we implement the nucleotide dependency method of Da Silva et al. (2025, Nature Genetics), which measures how mutations at position $i$ change the predicted nucleotide probabilities at position $j$ through the model's MLM head.

### 5.1 Algorithm

1. **Reference prediction.** Pass the unmasked sequence through the model and record nucleotide probabilities $P_\text{ref}(j, n)$ for each position $j$ and nucleotide $n \in \{A, C, G, T\}$.

2. **Perturbation.** For each position $i$, create three mutant sequences (as in Section 4.2) and record mutant probabilities $P_\text{mut}^{(b)}(j, n)$.

3. **Logit-difference scoring.** Convert probabilities to logits and compute the maximum absolute change:

$$\text{logit}(p) = \log_2(p) - \log_2(1 - p)$$

$$D_{ij} = \max_{b} \max_{n} \left| \text{logit}(P_\text{mut}^{(b)}(j, n)) - \text{logit}(P_\text{ref}(j, n)) \right|$$

A small $\epsilon = 10^{-10}$ is added before the log to ensure numerical stability (probabilities are renormalized after adding $\epsilon$).

4. **Symmetrization.** $D_{ij}^{\text{sym}} = \max(D_{ij}, D_{ji})$.

This method requires an MLM head (`predict_nucleotides` interface) and thus cannot be applied to encoder-only or non-autoregressive models that lack nucleotide-level output predictions.

---

## 6. Structural validation benchmarks

### 6.1 Ground truth structures

We validate the embedding-space methods against known RNA structures at two levels: secondary structure (base-pairing contacts) and tertiary structure (3D spatial proximity from crystal structures).

**tRNA-Arg-TCT-4-1** (*Homo sapiens*, chr1:159,141,611–159,141,684, minus strand; 74 nt). The cloverleaf secondary structure defines 24 Watson-Crick base pairs. The crystal structure (PDB 1EHZ, 1.93 Å) provides 3D atomic coordinates.

**Ribozymes.** To test generalization beyond the well-studied tRNA fold, we benchmark on three self-cleaving ribozymes with high-resolution crystal structures:

| Ribozyme | PDB | Resolution | Length | Chains | Structural features |
|----------|-----|-----------|--------|--------|-------------------|
| Hammerhead (*S. mansoni*) | 3ZP8 | 1.55 Å | 63 nt | 2 (A+B) | Three-way junction, no pseudoknots |
| Twister (env25) | 4OJI | 2.30 Å | 54 nt | 1 | Double pseudoknot |
| HDV (genomic) | 1SJ3 | 2.20 Å | 76 nt | 1 | Nested double pseudoknot |

All sequences are input as bare RNA with no genomic flanking context. For the hammerhead ribozyme, the two RNA chains are concatenated into a single input sequence. Sequences are obtained from the PDB FASTA records; the model wrapper automatically converts between T and U alphabets as needed.

### 6.2 Contact map evaluation (secondary structure)

For tRNA, we convert the dot-bracket secondary structure annotation to a binary contact matrix $\mathbf{C} \in \{0, 1\}^{N \times N}$, where $C_{ij} = 1$ if positions $i$ and $j$ form a base pair. We compute the Pearson correlation $r$ between the upper triangle of the predicted dependency map $\mathbf{D}$ and the binary contact matrix $\mathbf{C}$.

### 6.3 3D proximity evaluation (tertiary structure)

We extract C1' atom coordinates from each PDB structure and compute a proximity matrix:

$$P_{ij} = \frac{1}{\|x_i - x_j\|_2}$$

where $x_i$ is the C1' coordinate of residue $i$. The diagonal is set to zero.

**PDB-to-sequence alignment.** PDB residue numbers are mapped to 0-indexed FASTA positions using the residue numbering within each chain: $\text{fasta\_idx} = \text{chain\_offset} + (\text{resnum} - \text{first\_resnum})$. For multi-chain structures (hammerhead), the chain offset accumulates across chains. Residues missing from the PDB (e.g., disordered loop residues 17–18 in 4OJI) produce NaN entries in the proximity matrix; these pairs are excluded from correlation analysis.

We report Pearson $r$ between $D_{ij}$ and $P_{ij}$ over all upper-triangular pairs with $|i - j| \geq 4$, which excludes trivially adjacent positions along the backbone — standard practice in contact prediction evaluation.

### 6.4 Correction methods

We optionally apply two standard corrections from coevolution analysis:

**Average Product Correction (APC).** Removes row/column biases that create spurious signals from universally "noisy" positions (Dunn et al., 2008):

$$D_{ij}^\text{APC} = D_{ij} - \frac{\bar{D}_{i\cdot} \, \bar{D}_{\cdot j}}{\bar{D}}$$

where $\bar{D}_{i\cdot}$ is the mean of row $i$, $\bar{D}_{\cdot j}$ the mean of column $j$, and $\bar{D}$ the global mean (all computed with diagonal set to zero).

**Distance normalization.** Subtracts the mean score at each sequence separation $|i - j| = d$ to remove the systematic proximity bias inherent to all sequence-based methods:

$$D_{ij}^\text{dist} = D_{ij} - \overline{D}_{|i-j|}$$

---

## 7. Epistasis geometry for structural benchmarks

In addition to the perturbation and nucleotide dependency maps, we compute $R_\text{singles}$ (Section 2.3) for each position pair in the twister ribozyme to produce an epistasis geometry map. For each pair $(i, j)$ with $|i - j| \leq d_\text{max}$, we enumerate all combinations of alternative alleles at both positions (9 combinations per pair: 3 alternatives at $i$ × 3 alternatives at $j$), compute $R_\text{singles}$ for each, and take the maximum:

$$R_\text{singles}^{(i,j)} = \max_{\substack{b_i \neq r_i \\ b_j \neq r_j}} \frac{\|\boldsymbol{\varepsilon}^{(b_i, b_j)}\|_2}{\sqrt{\|\mathbf{v}_1^{(b_i)}\|_2^2 + \|\mathbf{v}_2^{(b_j)}\|_2^2} + \epsilon}$$

This produces a position × position matrix directly comparable to the perturbation and dependency maps, enabling a three-way comparison of embedding perturbation (pairwise dependency), nucleotide dependency (MLM-based dependency), and epistasis geometry (non-additive interaction) against the same 3D proximity ground truth.

---

## 8. Models

We evaluate 18 genomic foundation models spanning three architectural families:

**Transformer encoders (masked language models).** Nucleotide Transformer variants (NT-50M-3mer, NT-50M-multi, NT-100M-multi, NT-250M-multi, NT-500M-multi, NT-500M-ref, NT-2500M-multi, NT-2500M-1000G), DNABERT-2, MutBERT, RiNALMo, SpeciesLM (metazoa, fungi).

**State-space models.** HyenaDNA, Caduceus, Evo2.

**Supervised track predictors.** Borzoi, AlphaGenome.

Context lengths range from 256 nt (MutBERT) to 260 kb (Borzoi). Because these models have incompatible dependency requirements, each runs in a dedicated conda environment. For the dependency map benchmarks (Sections 4–6), we use RiNALMo (an RNA-specialized transformer) and SpeciesLM as representative models from different training paradigms.

---

## 9. Genomic epistasis pipeline

### 9.1 Mutation pair sources

We analyze 322,000+ mutation pairs from six biological systems:
- **tRNA folding** (74 bp, known contacts + PDB crystal structure)
- **CFTR mRNA structure** (3 folding pairs spanning 10 bp to 86 kb genomic)
- **KRAS cancer neighborhoods** (>4,100 variant pairs)
- **TCGA somatic pairs** (10,900 pairs from co-occurring somatic mutations)
- **Splicing variants** (FAS exon 6, MST1R exon 12)
- **eQTL epistasis** (Yang evQTL, GTEx)

### 9.2 Embedding generation

For each mutation pair, we extract the genomic region surrounding both variant positions (±context bp, where context depends on the model's maximum input length), apply the four genotype combinations (WT, M1, M2, M12) using the SeqMat library for coordinate-aware sequence manipulation, and compute mean-pooled embeddings. Embeddings are stored in per-source, per-model SQLite databases with WAL mode for concurrent access.

### 9.3 Metric computation

All metrics from Section 2 are computed for each pair. For datasets with >10,000 pairs, we use vectorized NumPy computation: all four embedding sets ($\bar{h}_\text{wt}$, $\bar{h}_1$, $\bar{h}_2$, $\bar{h}_{12}$) are loaded in a single SQL batch query and stacked into contiguous $(N \times d)$ arrays, enabling broadcasting-based computation of all metrics simultaneously. Mahalanobis metrics (Section 3) are computed via vectorized quadratic forms: $d_\text{Mahal} = \sqrt{\text{diag}(\mathbf{E} \, \hat{\Sigma}^{-1} \, \mathbf{E}^\top)}$ where $\mathbf{E}$ is the $(N \times d)$ matrix of residual vectors.

---

## 10. Software and reproducibility

All methods are implemented in the `genebeddings` Python package. Embedding perturbation and nucleotide dependency maps, epistasis geometry, Mahalanobis calibration, and model wrappers for all 18 models are available through a unified API. The structural validation analyses (tRNA and ribozyme benchmarks) are implemented as Jupyter notebooks.
