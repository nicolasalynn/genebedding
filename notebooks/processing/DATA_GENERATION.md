# Data Generation: Epistasis and Variant Sets

This document describes how each **source dataset** of epistasis pairs or single variants is created. These datasets are the inputs to the **processing** pipeline (see `process_epistasis`), which computes embeddings and stores them in per-model databases.

---

## Identifier formats

- **Single variant (mut_id):** `GENE:CHROM:POS:REF:ALT:STRAND` where `STRAND` is `P` (plus) or `N` (minus).
- **Epistasis (epistasis_id):** `mut1|mut2` with each mut in the above format, e.g. `KRAS:12:25227343:G:T:N|KRAS:12:25227344:A:T:N`.

Helpers used in generation:

- **Gene assignment:** PyRanges + GENCODE GTF; assign variant to nearest protein-coding gene TSS; strand from GTF.
- **make_mut_id(row, pos_col, ref_col, var_col, chrom_col, gene_col, rev_col):** Builds `mut_id` from row; uses `GENE_STRAND` (from `assets/benchmarks/gene_strands.csv`) if `rev_col` is None.
- **LiftOver:** UCSC hg19→hg38 chain for any hg19 coordinates.

---

## 1. ClinVar (single variants)

- **Source:** ClinVar VCF with VEP annotation (e.g. `clinvar.vep.vcf`).
- **Parsing:** Parse `CSQ` from INFO; use canonical transcript when available; one row per variant with consequence, impact, gene symbol, SIFT, PolyPhen, etc.
- **Curation:** Balance by consequence and pathogenicity (Pathogenic / Benign); include “ambiguous pathogenicity” subset.
- **Output:** Table with `chrom`, `pos`, `ref`, `alt`, `gene`, `consequence`, `clin_sig_clinvar`, plus `mut_id` and `rev` from `make_mut_id` and gene annotation.

---

## 2. eQTL (epistasis pairs)

- **Functional set (eQTL-connected):** From Cai et al. epistatic eQTL resource (lead evQTL + partial eQTL). Summary stats lifted hg19→hg38 (LiftOver). Retain SNVs with valid ref/alt in hg38; compute inter-variant distance; **exclude pairs with distance > 6 kb** (cis/local). Add identifiers in `GENE:CHROM:POS:REF:ALT:STRAND` format; annotate with gnomAD AF. Final set ≈ 1,450 pairs.
- **Null set (correlated variants):** Same genomic regions and LD as functional set but without reported epistatic eQTL. Anchors = genomic positions of functional pairs (e.g. lead evQTL). Per chromosome, get pairs in LD from TopLD (EUR, R² ≥ 0.2, 1 Mb window). Keep pairs where at least one variant is an anchor; format as above. Optionally **restrict null to distance < 6 kb** (or match/stratify by distance in analysis) to avoid distance confounding. Result ≈ 27K pairs.

---

## 3. Missplicing mutations

- Variants predicted to cause missplicing (e.g. SpliceAI). Curation and exact filters as in project-specific pipeline; output includes `mut_id` (and optionally epistasis pairs if double variants are defined).

---

## 4. FAS exon variants

- **Region:** FAS gene, exon of interest (e.g. exon 6/8).
- **Annotation:** e.g. Ensembl VEP; consequence field used to filter.
- **Filter:** Exclude any variant that has `missense_variant` in Consequence (silent/synonymous-only doubles if building pairs).
- **Pairs:** Double variants (epistasis) from the same exon/region; `epistasis_id` = `mut1|mut2` with `mut_id` format above. Optional Ensembl-style IDs for VEP input (`chrom pos pos ref/alt 1`).

---

## 5. MST1R variants

- Same idea as FAS: define gene/region, annotate (e.g. VEP), apply consequence/distance filters, output table with `mut_id` and, if applicable, `epistasis_id` for pairs.

---

## 6. tRNA sequences

- **Region:** e.g. tRNA loci from reference (e.g. SeqMat `SeqMat.from_fasta('hg38', 'chr1', start, end)` for a tRNA).
- **Epistasis set:** All unordered pairs of positions in the window with `max_distance` (e.g. 100 bp). For each pair, all non-reference ALT combinations (3×3 per pair). Format: `GENE:CHROM:POS1:REF1:ALT1:N|GENE:CHROM:POS2:REF2:ALT2:N` (zygosity `N`). Skip positions with ref not in `ACGT`.
- **Output:** DataFrame with `epistasis_id`, `pos1`, `pos2` (and optionally `mut1`, `mut2`).

---

## 7. TopLD double variants

- **Source:** TopLD output (e.g. EUR, R² ≥ 0.2, 1 Mb), e.g. `EUR_chr*_no_filter_0.2_1000000_LD.csv` with columns such as SNP1, SNP2, etc.
- **Processing:** Add `distance = SNP2 - SNP1`; filter e.g. `distance <= 5` (or 100 bp). Merge with 1kGP-style data if needed for AF. Add `site1_id`, `site2_id`, `pair_site_id`, `pair_source` (e.g. `g1k_chr12`), `pair_type` (e.g. `population_double_candidate`).
- **Epistasis IDs:** Build `mut_id` for each variant (gene from annotation + make_mut_id), then `epistasis_id = mut1|mut2`. Gene annotation via PyRanges/GENCODE as above.
- **Output:** Table with `chrom`, `pos1`, `pos2`, `distance`, `ref1`, `alt1`, `ref2`, `alt2`, `epistasis_id`, etc.

---

## 8. 1000 Genomes Project (OKGP) double variants

- **Source:** e.g. `g1k_chr12_double_variants_max100bp.csv` (or similar); columns e.g. `distance_bp`, `Chromosome` → rename to `distance`, `chrom`.
- **Filter:** e.g. `distance <= 5` (or project-specific). Optionally merge with TopLD for LD info.
- **Identifiers:** Same as TopLD: gene annotation, `make_mut_id`, `epistasis_id = mut1|mut2`.
- **Output:** One table per chromosome or concatenated, with `epistasis_id`, allele frequencies, carrier counts, etc.

---

## 9. TCGA double variants

- **Source:** Per-gene TCGA mutation tables (e.g. via `geney.tcga_utils.TCGAGene`); exclude filter “panel_of_normals”.
- **Constants:** `MAX_DIST` (e.g. 100 bp), `MIN_DIST` (0), `ALT_MIN`, `MIN_DEPTH`, `LINK_VAF_DELTA`, `LINK_ALT_RATIO`, `DIST_BINS`.
- **Per gene and case (tumor sample):** Get variants with sufficient depth; compute VAF; form `site_id` (chrom:pos). For each chromosome, find variant pairs within `MAX_DIST` (e.g. with searchsorted). Classify as linked vs unlinked using VAF and alt ratio.
- **Matching:** Distance bins (e.g. 0–5, 5–10, 10–25, 25–50, 50–100 bp) for downstream matching of linked/unlinked.
- **Output:** Table with `epistasis_id`, `mut1`, `mut2`, `case_id`, `is_linked`, `distance`, `chrom`, `gene`, `pos1`, `pos2`, `rev`, filters, depth, VAF. Optionally restrict to `epistasis_id` with unique `case_id` (one case per pair) for analysis.

---

## 10. Null / background set (`null`)

- **Purpose:** Background distribution of embedding shifts per model; no biological selection.
- **Options:**
  - Random pairs from a reference (e.g. same regions as a functional set, or genome-wide) with distance constraints.
  - Or use one of the “null” sets above (e.g. TopLD correlated pairs, or eQTL null) and label the **source** as `'null'` when running the processing pipeline.
- **Critical:** The processing pipeline must run the **`null`** source **first** so that null embeddings exist before any downstream comparison or background calibration.

---

## Summary table

| Source           | Type    | Key filters / notes                               | Output columns (min)     |
|------------------|---------|---------------------------------------------------|--------------------------|
| ClinVar          | Single  | Balanced pathogenicity, consequence               | mut_id, chrom, pos, ref, alt, gene, rev |
| eQTL functional  | Pairs   | Distance ≤ 6 kb, liftover hg38                    | epistasis_id, …          |
| eQTL null        | Pairs   | TopLD anchors, optional distance ≤ 6 kb            | epistasis_id, …          |
| Missplicing      | Single/Pairs | SpliceAI-based                            | mut_id / epistasis_id    |
| FAS exon         | Pairs   | Exon, silent-only (no missense)                   | epistasis_id, mut1, mut2 |
| MST1R            | Pairs   | Gene/region, consequence filters                  | epistasis_id, …           |
| tRNA             | Pairs   | All ALT combos, max_distance                      | epistasis_id, pos1, pos2  |
| TopLD            | Pairs   | distance ≤ 5 (or 100), LD from TopLD              | epistasis_id, pair_site_id, pair_source |
| OKGP             | Pairs   | distance ≤ 5, 1kGP + optional TopLD               | epistasis_id, AF, …      |
| TCGA             | Pairs   | MAX_DIST, depth/VAF, linked/unlinked, distance bins | epistasis_id, case_id, is_linked, distance |
| null             | Pairs   | Background; no functional selection               | epistasis_id             |

---

## Dependencies

- **seqmat:** Gene, SeqMat, genome sequence and indices.
- **pyranges:** GTF overlap for gene assignment.
- **pyliftover:** hg19→hg38.
- **pandas, numpy:** Tables and identifiers.
- **geney (optional):** TCGA (e.g. `TCGAGene`).
- **GENCODE GTF, UCSC chain file, TopLD/1kGP outputs:** As per paths in project.

Once a dataset is generated, it is passed to the **processing** step with a **source name** (e.g. `fas_analysis`, `mst1r_analysis`, `tcga_analysis`, `okgp_analysis`, `null`). The pipeline writes one directory per source, each containing one `.db` file per model; **`null` must be processed first.**
