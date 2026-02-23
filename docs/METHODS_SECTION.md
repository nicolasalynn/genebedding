# Methods Section Outline for Paper

This document outlines the methodology as implemented in the codebase, for use when writing the paper's Methods section. It reflects the actual definitions and pipeline so the text stays consistent with the analysis.

---

## 1. Embedding extraction and variant representation

- **Models:** Any genomic foundation model exposed via the wrapper API with `embed(seq, pool="mean")` (and optionally batched `embed(list_of_seqs, pool="mean")`). Used models in the pipeline include Nucleotide Transformer (NT) variants, ConvNova, MutBERT, Borzoi, AlphaGenome, Evo2, SpliceAI (OpenSpliceAI), etc.; each wrapper normalizes sequence, tokenization, and pooling.
- **Sequence context:** For each variant or epistasis pair, reference sequence is fetched from the genome (e.g. hg38) with a fixed **context window** (e.g. 3 kb) centered on the variant site(s). For pairs, the window spans from min(pos1, pos2) − context to max(pos1, pos2) + context. Reverse complement is applied when strand is negative (parsed from variant ID or from a strand column).
- **Variant IDs:** Single variants: `GENE:CHROM:POS:REF:ALT` or with optional `:STRAND` (P/N). Epistasis pairs: `mut1|mut2` with the same per-mutation format.
- **Genotypes:** For each epistasis pair we obtain four sequences—WT (reference), M1 (mutant 1 only), M2 (mutant 2 only), M12 (both mutations)—and compute one embedding per sequence, then store them under keys `{epistasis_id}|WT`, `|M1`, `|M2`, `|M12` in a SQLite embedding database.

---

## 2. Effect vectors and additive expectation

- **Effect vectors (in embedding space):**
  - **v1** = M1 − WT  
  - **v2** = M2 − WT  
  - **v12_obs** = M12 − WT (observed double-mutant effect)  
  - **v12_exp** = v1 + v2 (additive expectation).

- **Epistasis residual:**  
  **r** = v12_obs − v12_exp = (M12 − WT) − (v1 + v2).  
  This is the deviation of the double-mutant embedding from the additive prediction and is the main quantity used for epistasis strength and for null-background calibration.

All norms and angles below are in the **same embedding space** (L2 for magnitudes; cosines for angles), unless stated otherwise (e.g. dependency maps can use different metrics).

---

## 3. Epistasis metrics (scalar summaries)

The following metrics are computed for each epistasis pair and added to the analysis table. They are interpretable and assumption-light.

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **len_WT_M1** | ‖v1‖ | Single mutation 1 effect size |
| **len_WT_M2** | ‖v2‖ | Single mutation 2 effect size |
| **len_WT_M12** | ‖v12_obs‖ | Observed double-mutant effect size |
| **len_WT_M12_exp** | ‖v12_exp‖ | Expected (additive) effect size |
| **epi_R_raw** | ‖r‖ | Raw epistasis magnitude (deviation from additivity) |
| **epi_R_singles** | ‖r‖ / √(‖v1‖² + ‖v2‖²) | Residual normalized by single-mutation scale; comparable across pairs |
| **cos_v1_v2** | cos(v1, v2) | Alignment of single effects (+1 reinforcing, −1 opposing) |
| **cos_exp_to_obs** | cos(v12_exp, r) | Direction of deviation: −1 toward WT (corrective), +1 away from WT (synergistic) |
| **magnitude_ratio** | ‖v12_obs‖ / ‖v12_exp‖ | &lt;1 sub-additive, &gt;1 super-additive |
| **log_magnitude_ratio** | log((‖v12_obs‖+ε) / (‖v12_exp‖+ε)) | Same on log scale; 0 ≈ additive |

(ε is a small constant for numerical stability, e.g. 1e−20.)

---

## 4. Null background and Mahalanobis metrics

- **Null design:** A dedicated **null** dataset of epistasis pairs (e.g. neutral/synonymous or non-functional pairs) is processed first. For each model, embeddings for WT, M1, M2, M12 are computed and stored; residuals **r** are computed as above.
- **Covariance estimation:** From the null residuals only, a covariance matrix **Σ** is estimated (e.g. Ledoit–Wolf shrinkage), regularized with a small ridge (e.g. 1e−6), and inverted to obtain **Σ⁻¹**. This is saved per model (e.g. `null_cov/{model_key}_pack.npz`).
- **Mahalanobis metrics (non-null sources):** When analysing non-null epistasis (e.g. disease or experimental pairs), the same **Σ⁻¹** from the null is used to define covariance-aware metrics:
  - **epi_mahal** = √(rᵀ Σ⁻¹ r) — Mahalanobis distance of the residual in the null residual distribution.
  - **mahal_obs** = √(v12_obsᵀ Σ⁻¹ v12_obs) — Mahalanobis magnitude of the observed effect.
  - **mahal_add** = √(v12_expᵀ Σ⁻¹ v12_exp) — Mahalanobis magnitude of the additive effect.
  - **mahal_ratio** = mahal_obs / mahal_add; **log_mahal_ratio** = log(mahal_obs / mahal_add).

So the null provides the covariance of “non-interacting” residuals; epi_mahal measures how unusual the residual is relative to that background.

---

## 5. Pipeline and batch processing

- **Order of operations:** (1) Process the **null** source; compute and save null **Σ** and **Σ⁻¹** per model. (2) For each non-null source, load the corresponding **Σ⁻¹** and run the metric step with it so that Mahalanobis columns are filled.
- **Batched embedding:** To reduce runtime, multiple epistasis pairs can be embedded in one forward pass (e.g. `batch_size=8` → 32 sequences per batch). The same four sequences (WT, M1, M2, M12) per pair are built with the same context and genome; embeddings are then assigned back to each pair and stored. Metrics are computed from these stored embeddings.
- **SpliceAI:** Used only for splicing-related sources (e.g. fas_analysis, mst1r_analysis, kras); other models are run for all sources.
- **Environment profiles:** Some models require separate environments (e.g. AlphaGenome with JAX, Evo2 with its stack). The pipeline can be run with an environment profile so that only models available in the current environment are executed (e.g. `main` vs `alphagenome` vs `evo2`).

---

## 6. Optional: triangle and complex representations

- **Triangle representation:** Using only the three distances ‖v12_obs‖, ‖v12_exp‖, and ‖r‖, the WT–Expected–Observed triangle is embedded in 2D (WT at origin, Expected at (1,0), Observed in the upper half-plane via the law of cosines). This gives a dimensionality-free view of epistasis (rho, theta).
- **Complex representation:** In the high-dimensional plane spanned by the additive direction and an orthogonal direction derived from the single-mutation effects, observed and expected positions are projected to 2D; epistasis can be written as a complex number ε = Δ∥ + iΔ⊥ with magnitude and phase. Used for visualization and for dependency maps that aggregate over pairs.

---

## 7. Dependency maps and pairwise epistasis

- **Pairwise epistasis:** For two positions in a sequence, all 3×3 (or 4×4) single-nucleotide combinations can be embedded to get WT, M1, M2, M12 and hence metrics (e.g. epi_R_raw) per combination.
- **Dependency map:** A matrix of positions × positions where each entry is an aggregation (e.g. max or mean) of an epistasis metric (e.g. epi_R_raw) over the mutation combinations at that pair. Used to summarize which position pairs show strong epistasis in the embedding space.

---

## 8. Consistency and implementation notes

- **Single implementation of core metrics:** The scalar epistasis metrics (len_WT_M1, len_WT_M2, len_WT_M12, len_WT_M12_exp, epi_R_raw, epi_R_singles, cos_v1_v2, cos_exp_to_obs, magnitude_ratio, log_magnitude_ratio) are defined in one place in `genebeddings.genebeddings.EpistasisGeometry.metrics()` and duplicated in logic in `add_epistasis_metrics()`. The same formulas appear in `genebeddings.epistasis_features.compute_compat_metrics()` (used for feature extraction and for null covariance from precomputed embeddings). Definitions are aligned: residual = delta_obs − delta_add, with delta_obs = z_ab − z_wt and delta_add = (z_a − z_wt) + (z_b − z_wt).
- **Null residual:** The residual used for null covariance is exactly (M12 − WT) − (M1 − WT + M2 − WT) = M12 − M1 − M2 + WT, i.e. the same definition as in the main metrics.
- **Benchmarks:** The `benchmarks/` folder contains an alternative `EpistasisGeometry` with optional “geometric” and “pythagorean” expectations; the main pipeline and this methods outline use only the **additive** expectation (v12_exp = v1 + v2). For the paper, the canonical pipeline is the one in `genebeddings` and `notebooks/processing/process_epistasis.py`.

---

## Suggested Methods subsection headings (for the paper)

1. **Embedding extraction and sequence context**  
2. **Effect vectors and additive expectation**  
3. **Epistasis metrics**  
4. **Null background and Mahalanobis normalization**  
5. **Processing pipeline and batch embedding**  
6. **Optional: Triangle and complex representations** (if you include those analyses)  
7. **Dependency maps** (if used)

You can paste the definitions and interpretations from above into each subsection and adjust wording to match journal style.

---

## 9. Limitations and framing (for Discussion / end of Methods)

**Scope of the framework.** The framework is designed and validated for (1) quantifying genetic interactions (epistasis) via effect vectors and additive expectation, and (2) comparing effect magnitudes and directions in embedding space. It is **not** designed to reliably predict discrete mutation mechanism (e.g. splicing vs. missense) from the embedding delta alone.

**Supervised mechanism classification from delta.** We attempted to classify mutation mechanism (e.g. missense vs. splicing) from the delta embedding (MUT − WT) using implemented classifiers (e.g. MLP on deltas with optional whitening; mechanism signatures with LDA or other discriminative methods). Performance was limited (e.g. near chance or modest AUC). We interpret this as indicating that the embedding space in the models used does not strongly separate mechanism in a supervised, delta-only setting—likely because deltas encode magnitude and local context more than a categorical mechanism axis, and because mechanism is often confounded with position and effect size. We therefore do not claim that embedding deltas suffice for mechanism classification; mechanism-specific predictors (e.g. splice-effect scores) remain more appropriate for that task. This negative result is reported for transparency.

**How to strengthen the paper.** Emphasise analyses where the framework performs well: null-calibrated epistasis metrics, dependency maps, effect similarity and within-mechanism stratification (e.g. pathogenic vs. benign among missense only). Where relevant, use mechanism-specific tools (e.g. SpliceAI/OpenSpliceAI) to characterise splicing and report correlation with splicing-related outcomes rather than claiming generic-delta-based mechanism prediction.

---

## 10. Mahalanobis “profile” for mechanism (generative scoring)

**Idea.** Instead of training a discriminative classifier (e.g. boundary in delta space), build a **covariance profile** per mechanism: from a set of known missplicing deltas fit a mean and covariance (e.g. μ_splice, Σ_splice); from known missense deltas fit (μ_missense, Σ_missense). For a **new** delta **x**, score how well it fits each profile, e.g. by Mahalanobis distance to each group:

- d_splice(**x**) = √((**x** − μ_splice)′ Σ_splice⁻¹ (**x** − μ_splice))  
- d_missense(**x**) = √((**x** − μ_missense)′ Σ_missense⁻¹ (**x** − μ_missense))

Then assign **x** to the mechanism whose profile it matches best (e.g. smallest Mahalanobis distance, or highest Gaussian likelihood under that group’s distribution). Equivalently, you can use log-likelihood under each group’s normal distribution and choose the class with highest likelihood.

**Does it make sense?** Yes. This is a standard **generative** approach: each mechanism is modelled as a distribution in delta space; a new delta is scored by “typicality” under each distribution. It is closely related to Quadratic Discriminant Analysis (QDA). It is **not** ridiculous; it is a natural and mathematically sound alternative to a discriminative classifier.

**Relation to existing code.** The package already supports this pattern in `genebeddings.mechanism_signatures`: `MechanismSignature` holds a direction, a (e.g. PCA) basis, mean, covariance and cov_inv, and provides `mahalanobis(v)`, `log_likelihood(v)`, and `p_value(v)`. `MechanismClassifier` fits one signature per group (e.g. “splicing”, “missense”) and can classify by comparing these scores (e.g. by log-likelihood or by a downstream LDA/kNN). So “Mahal profile” classification is essentially: fit one signature per mechanism, then classify a new delta to the group for which Mahalanobis distance is smallest (or log-likelihood is largest). You can do this with the current API by using the per-signature `mahalanobis` or `log_likelihood` and assigning to the best-matching group.

**Caveats.** (1) **Dimension vs. sample size:** If the delta dimension is large (e.g. 512–1024) and you have few labelled deltas per mechanism, the sample covariance is singular. Use regularization (e.g. ridge, Ledoit–Wolf) or reduce dimension (e.g. PCA per group, then Mahalanobis in that subspace)—as in the existing signature fitting. (2) **Comparing across groups:** If each group uses a different PCA subspace or dimension, comparing log-likelihoods across groups is comparing different models; using a shared subspace (e.g. joint PCA, then per-class mean/cov in that space) or a fixed number of components per class can make comparisons fair. (3) **Same underlying limitation:** If the delta truly does not carry a strong mechanism signal, the two mechanism distributions will overlap and even this generative approach may give limited accuracy—but it can still provide interpretable “typicality” scores (e.g. “this delta is very typical of missense, atypical of splicing”) and is worth trying alongside or instead of a purely discriminative classifier.
