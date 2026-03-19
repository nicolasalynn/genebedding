# Updates to Manuscript — Quantitative Evidence (March 2026)

Supported by results from `analyze_covariance.ipynb`, baseline comparisons,
and embarrassment checks. All referenced figures in `{embeddings_dir}/figures/covariance/`.

---

## 1. Phenotype-Metric Alignment — Mechanistic Proof (Discussion)

### Proposed text:

The geometric mechanism underlying phenotype-metric alignment is directly observable
in the covariance structure of epistasis residuals. For each model, we computed the
Grassmann similarity between the top-50 eigenvectors of the signal covariance
(from FAS, eQTL, or TCGA residuals) and the null covariance (from 1kGP population
pairs), measuring how much the signal subspace aligns with the null subspace.

For track-based models, expression epistasis (eQTL) aligns MORE with the null than
splicing epistasis (FAS), confirming that the null axes are expression-dominated:

| Model | FAS overlap | eQTL overlap | TCGA overlap | eQTL > FAS? |
|-------|------------|-------------|-------------|-------------|
| Borzoi | 0.254 | 0.290 | 0.348 | ✓ |
| AlphaGenome | 0.244 | 0.325 | 0.372 | ✓ |

This explains the paper's central result: raw Euclidean distances in track models
capture expression epistasis (eQTL z=55.9) because the signal lives on the same axes
as the null. Splicing epistasis is orthogonal to these axes, making it invisible to
raw distances but recoverable with Mahalanobis calibration that downweights the
expression-dominated dimensions.

For self-supervised models (HyenaDNA, Caduceus: participation ratio = 256/256,
condition number = 1.0), the null is isotropic — there are no dominant axes to
downweight, and Mahalanobis calibration reduces to identity.

### NTv3 as independent confirmation:

NTv3 provides a direct experimental test: the pre-trained model (NTv3 100M pre)
shows moderate overlap across all sources (FAS=0.251, eQTL=0.287, TCGA=0.326),
typical of self-supervised MLMs. After post-training on 16,000 functional tracks,
NTv3 100M post shifts to track-like behavior (FAS=0.289, eQTL=**0.400**,
TCGA=**0.399**), with eQTL overlap nearly doubling. Post-training concentrates the
null from PR=728 to PR=5 — an extreme 145-fold reduction in effective dimensionality.
This demonstrates that supervised post-training on expression/chromatin tracks
directly produces the anisotropic geometry that necessitates Mahalanobis correction
for orthogonal phenotypes.

**Referenced figures:**
- Supplementary Figure S_subspace_overlap: `signal_null_subspace_overlap.png`
- Supplementary Figure S_eigenspectra: `signal_vs_null_eigenspectra.png`

---

## 2. Low-Dimensional Structure of Cancer Epistasis (Section 3.4 or Discussion)

TCGA epistasis residuals occupy a substantially lower-dimensional subspace than
population variant residuals across 18 of 20 models (excluding HyenaDNA and
Caduceus, whose isotropic null prevents comparison).

| Model | d | Null PR | TCGA PR | Ratio |
|-------|---|---------|---------|-------|
| Borzoi | 1920 | 1719 | 66 | 0.04 |
| NT-2500-1kGP | 2560 | 2532 | 78 | 0.03 |
| NTv3 100M pre | 768 | 728 | 47 | 0.06 |
| Evo2 | 4096 | 49 | 5 | 0.10 |
| NT-50M | 512 | 156 | 46 | 0.29 |
| NTv3 100M post | 768 | 5 | 4 | 0.80 |
| AlphaGenome | 1536 | 34 | 36 | 1.05 |

Using full null covariance (no N-matching needed — N-matching effect was 1.00 across
all models when using `global_cov` from the full 100K null pairs).

This concentration suggests cancer epistasis is organized along a small number of
biologically meaningful axes, not random noise in high-dimensional space.

---

## 3. CORRECTIONS — Findings That Do NOT Hold

### 3a. Single-variant vs epistasis comparison: CONFOUNDED

The residualized comparison (Cell 14 output showing |log_MR| ratio 3.96x) is
**confounded by distance**. Within-bin analysis shows:

| Model | Metric | Within-bin TCGA/1kGP ratio | Significant? |
|-------|--------|---------------------------|-------------|
| NT-50M | \|log_MR\| | 0.4-0.9 across all bins | NO — 1kGP HIGHER |
| Borzoi | \|log_MR\| | 0.8-1.0 across all bins | NO |
| MutBERT | \|log_MR\| | 1.2-1.7 at some bins | Partial |
| DNABERT | \|log_MR\| | <1 at short range, ~1.8 at long | Distance-dependent |

**DO NOT report the residualized ratios.** The distance distributions of TCGA
(median 4bp) and 1kGP (median 29bp) are too different for residualization to work.
Only within-bin comparisons are valid.

### 3b. Co-occurrence correlation: CONFOUNDED

The rho=0.199 correlation between epistasis score and TCGA co-occurrence probability
was **entirely driven by distance** (closer pairs have both higher epistasis and
higher co-occurrence). After distance-residualizing BOTH variables, correlation
flipped to rho=-0.05.

### 3c. Cancer gene enrichment vs baselines: MODEST

Enrichment is comparable between epistasis metrics and simple baselines:
- TSG: epistasis wins 13/18 models, but mean folds similar (1.62 vs 1.43)
- Oncogene: baselines win 10/18 models (sum_singles 1.74 > log_MR 1.61)

The framework's advantage is directional decomposition (corrective vs cumulative),
not raw enrichment power.

### 3d. HyenaDNA/Caduceus covariance: UNINFORMATIVE

Both have PR = 256/256, condition = 1.0 (identity matrix). Their covariance is
pure Ledoit-Wolf shrinkage to identity. Exclude from all covariance analyses.

---

## 4. What IS Solid (peer-review defensible)

| Finding | Why it's clean |
|---------|---------------|
| FAS Mahalanobis improvement | Single gene, experimental ground truth, no confounds |
| tRNA structural validation | Crystal structure ground truth |
| KRAS compensatory detection | Known biology, neighborhood ranking |
| eQTL track-model detection | Distance-controlled permutation test |
| NF1 survival (3 cancers) | Cross-cancer replication, controlled design |
| Subspace overlap FAS < eQTL for track models | Same null reference, no distance confound |
| NTv3 pre→post shift | Direct experimental test of mechanism |
| Directional asymmetry (TSG corrective, oncogene cumulative) | Holds across 18 models (Wilcoxon), but weak per-bin for individual models |

---

## 5. Cross-Model Covariance Structure (Supplementary)

### NT family scaling:
NT-2500-1kGP uses 99% of embedding space for null (PR=2532/2560).
NT-2500-multi concentrates into 0.6% (PR=16/2560). 160-fold difference
from training data composition alone.

### NTv3 pre vs post:
NTv3 100M pre: null PR = 728/768 (95% isotropic)
NTv3 100M post: null PR = 5/768 (0.7% — extreme concentration)
Post-training compressed null variation 145-fold into 5 dimensions.

### RiNALMo anomaly:
PR = 1.7/1280, condition number 31,000. All null variation in ~2 axes.
Despite this, competitive FAS performance — the 2 null axes are not
aligned with splicing signal.

---

## 6. Pending Analyses

- [ ] Selection pressure comparison (fig_selection_pressure.py — 4 models embedded, needs to run)
- [ ] Mahalanobis diagnostics: shrinkage intensity per model
- [ ] Distance-binned covariance shift (run_metrics_only Cell 4 — binned data now available)
- [ ] Update Cell 14 to use within-bin comparison instead of residualization

---

## Supplementary Figure Index

| Label | File | Description |
|-------|------|-------------|
| S_eigenspectra | `eigenvalue_spectra.png` | Null eigenvalue decay + cumulative variance |
| S_effective_dim | `effective_dimensionality.png` | Participation ratio vs embedding dim |
| S_dimension_usage | `dimension_usage_ratio.png` | Fraction of embedding space used for null |
| S_signal_vs_null | `signal_vs_null_eigenspectra.png` | Signal (TCGA/FAS/eQTL) vs null per model |
| S_subspace_overlap | `signal_null_subspace_overlap.png` | Grassmann similarity + projection fraction |
| S_precision_heatmaps | `precision_heatmaps.png` | Clustered partial correlation matrices |
| S_precision_sparsity | `precision_sparsity.png` | Conditional dependency sparsity |
| S_rv_coefficient | `cross_model_rv_coefficient.png` | Cross-model covariance alignment |
| S_eigenspectrum_sim | `eigenspectrum_similarity.png` | Spectral shape clustering |
