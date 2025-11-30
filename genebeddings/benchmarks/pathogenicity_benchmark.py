"""
Enhanced Pathogenicity Prediction Benchmarker

A robust and comprehensive benchmarking framework for evaluating model performance
on ClinVar pathogenicity prediction using multiple scoring methods.

Features:
- Three prediction methods: embedding differences, log odds, probability scoring
- Multiple geometric distance metrics for embeddings
- Comprehensive evaluation metrics and statistical testing
- Efficient batching and error handling
- Cross-validation support
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional, Union, Callable
from seqmat import SeqMat
from tqdm import tqdm
import warnings
from dataclasses import dataclass
from pathlib import Path
import json

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# Constants
NUCLEOTIDES = ("A", "C", "G", "T")
DEFAULT_CONTEXT_SIZE = 3000
DEFAULT_BATCH_SIZE = 8


@dataclass
class BenchmarkConfig:
    """Configuration for pathogenicity benchmarking"""
    context_size: int = DEFAULT_CONTEXT_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    fasta_name: str = "hg38"

    # Column mappings
    chrom_col: str = "chrom"
    pos_col: str = "pos"
    ref_col: str = "ref"
    alt_col: str = "alt"
    label_col: str = "clin_sig_clinvar"

    # Evaluation settings
    n_folds: int = 5
    random_state: int = 42
    min_samples_per_class: int = 10

    # Filtering settings
    max_indel_size: int = 50  # Skip very large indels
    skip_complex_variants: bool = True


class GeometricMetrics:
    """Collection of geometric distance/similarity metrics for embeddings"""

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors"""
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / max(denom, 1e-12))

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance (1 - cosine_similarity)"""
        return 1.0 - GeometricMetrics.cosine_similarity(a, b)

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """L2 Euclidean distance"""
        return float(np.linalg.norm(a - b))

    @staticmethod
    def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
        """L1 Manhattan distance"""
        return float(np.sum(np.abs(a - b)))

    @staticmethod
    def chebyshev_distance(a: np.ndarray, b: np.ndarray) -> float:
        """L∞ Chebyshev distance"""
        return float(np.max(np.abs(a - b)))

    @staticmethod
    def minkowski_distance(a: np.ndarray, b: np.ndarray, p: float = 3.0) -> float:
        """Lp Minkowski distance"""
        return float(np.sum(np.abs(a - b) ** p) ** (1.0 / p))

    @staticmethod
    def canberra_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Canberra distance"""
        numerator = np.abs(a - b)
        denominator = np.abs(a) + np.abs(b) + 1e-12
        return float(np.sum(numerator / denominator))


class PathogenicityBenchmarker:
    """
    Comprehensive pathogenicity prediction benchmarker with multiple scoring methods
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.geometric_metrics = {
            'cosine_similarity': GeometricMetrics.cosine_similarity,
            'cosine_distance': GeometricMetrics.cosine_distance,
            'euclidean_distance': GeometricMetrics.euclidean_distance,
            'manhattan_distance': GeometricMetrics.manhattan_distance,
            'chebyshev_distance': GeometricMetrics.chebyshev_distance,
            'minkowski_distance': GeometricMetrics.minkowski_distance,
            'canberra_distance': GeometricMetrics.canberra_distance,
        }

    def _validate_variant(self, chrom: str, pos: int, ref: str, alt: str) -> bool:
        """Validate if variant should be processed"""
        try:
            # Check for valid nucleotides
            ref, alt = ref.upper(), alt.upper()

            # Skip if either contains invalid characters (beyond ATCG)
            valid_chars = set(NUCLEOTIDES + ('N', '-'))
            if not (set(ref) <= valid_chars and set(alt) <= valid_chars):
                return False

            # Skip very large indels
            if max(len(ref), len(alt)) > self.config.max_indel_size:
                return False

            # Skip complex variants if configured
            if self.config.skip_complex_variants:
                # Define complex as having multiple changes or unusual patterns
                if len(ref) > 1 and len(alt) > 1 and ref != alt:
                    return False

            return True

        except Exception:
            return False

    def _get_sequence_context(self, chrom: str, pos: int, ref: str, alt: str) -> Tuple[str, str, int]:
        """
        Get reference and mutated sequence contexts

        Returns:
            ref_seq: Reference sequence
            alt_seq: Mutated sequence
            center_idx: Index of mutation site in sequences
        """
        # try:
        # Build reference sequence
        context_start = pos - self.config.context_size
        context_end = pos + self.config.context_size

        ref_seqmat = SeqMat.from_fasta(
            self.config.fasta_name,
            f'chr{chrom}',
            context_start,
            context_end
        )
        ref_seq = ref_seqmat.seq
        center_idx = self.config.context_size

        # Build mutated sequence
        alt_seqmat = SeqMat.from_fasta(
            self.config.fasta_name,
            f'chr{chrom}',
            context_start,
            context_end
        )
        alt_seqmat.apply_mutations([(pos, ref, alt)], permissive_ref=True)
        alt_seq = alt_seqmat.seq

        return ref_seq, alt_seq, center_idx

        # except Exception as e:
        #     raise ValueError(f"Failed to get sequence context for {chrom}:{pos} {ref}>{alt}: {str(e)}")

    def _compute_embedding_metrics(self, model, ref_seq: str, alt_seq: str) -> Dict[str, float]:
        """Compute all embedding-based distance/similarity metrics"""
        try:
            # Get embeddings (mean pooled)
            ref_emb = model.embed(ref_seq, pool="mean", return_numpy=True)
            alt_emb = model.embed(alt_seq, pool="mean", return_numpy=True)

            # Compute all geometric metrics
            metrics = {}
            for name, metric_func in self.geometric_metrics.items():
                try:
                    metrics[f"emb_{name}"] = metric_func(ref_emb, alt_emb)
                except Exception as e:
                    warnings.warn(f"Failed to compute {name}: {str(e)}")
                    metrics[f"emb_{name}"] = np.nan

            return metrics

        except Exception as e:
            warnings.warn(f"Failed to compute embedding metrics: {str(e)}")
            return {f"emb_{name}": np.nan for name in self.geometric_metrics.keys()}

    def _compute_probability_metrics(self, model, ref_seq: str, alt_seq: str,
                                   center_idx: int, ref: str, alt: str) -> Dict[str, float]:
        """Compute probability-based metrics"""
        try:
            metrics = {}

            # Create masked sequence at center position for probability prediction
            masked_seq = ref_seq[:center_idx] + 'N' + ref_seq[center_idx + len(ref):]

            # Get nucleotide probabilities at masked position
            probs_dict = model.predict_nucleotides(masked_seq, return_dict=True)[0]

            # Extract probabilities
            p_ref = float(probs_dict.get(ref, 0.0))
            p_alt = float(probs_dict.get(alt, 0.0))

            # Probability metrics
            metrics['prob_ref'] = p_ref
            metrics['prob_alt'] = p_alt
            metrics['prob_diff'] = p_ref - p_alt
            metrics['prob_ratio'] = p_ref / (p_alt + 1e-9)

            # Log-based metrics (handle zeros)
            log_p_ref = np.log(max(p_ref, 1e-9))
            log_p_alt = np.log(max(p_alt, 1e-9))

            metrics['log_prob_ref'] = log_p_ref
            metrics['log_prob_alt'] = log_p_alt
            metrics['log_odds_ratio'] = log_p_ref - log_p_alt

            # Normalized probabilities
            total_prob = sum(probs_dict.values())
            if total_prob > 0:
                metrics['prob_ref_norm'] = p_ref / total_prob
                metrics['prob_alt_norm'] = p_alt / total_prob
            else:
                metrics['prob_ref_norm'] = np.nan
                metrics['prob_alt_norm'] = np.nan

            return metrics

        except Exception as e:
            warnings.warn(f"Failed to compute probability metrics: {str(e)}")
            return {
                'prob_ref': np.nan, 'prob_alt': np.nan, 'prob_diff': np.nan,
                'prob_ratio': np.nan, 'log_prob_ref': np.nan, 'log_prob_alt': np.nan,
                'log_odds_ratio': np.nan, 'prob_ref_norm': np.nan, 'prob_alt_norm': np.nan
            }

    def _compute_contextual_probability_metrics(self, model, ref_seq: str, alt_seq: str,
                                              center_idx: int, ref: str, alt: str) -> Dict[str, float]:
        """Compute probability metrics comparing ref vs alt contexts"""
        try:
            metrics = {}

            # Mask the same position in both reference and altered contexts
            ref_masked = ref_seq[:center_idx] + 'N' + ref_seq[center_idx + len(ref):]
            alt_masked = alt_seq[:center_idx] + 'N' + alt_seq[center_idx + len(alt):]

            # Get probabilities in both contexts
            ref_probs = model.predict_nucleotides(ref_masked, return_dict=True)[0]
            alt_probs = model.predict_nucleotides(alt_masked, return_dict=True)[0]

            # Compare how the mutation affects the probability landscape
            prob_shifts = {}
            for nuc in NUCLEOTIDES:
                p_ref_ctx = ref_probs.get(nuc, 0.0)
                p_alt_ctx = alt_probs.get(nuc, 0.0)
                prob_shifts[f'ctx_shift_{nuc}'] = p_alt_ctx - p_ref_ctx

            metrics.update(prob_shifts)

            # Summary metrics
            metrics['ctx_total_shift'] = sum(abs(v) for v in prob_shifts.values())
            metrics['ctx_max_shift'] = max(abs(v) for v in prob_shifts.values()) if prob_shifts else 0.0

            # Specific focus on ref/alt alleles in different contexts
            ref_in_ref_ctx = ref_probs.get(ref, 0.0)
            ref_in_alt_ctx = alt_probs.get(ref, 0.0)
            alt_in_ref_ctx = ref_probs.get(alt, 0.0)
            alt_in_alt_ctx = alt_probs.get(alt, 0.0)

            metrics['ref_allele_ctx_shift'] = ref_in_alt_ctx - ref_in_ref_ctx
            metrics['alt_allele_ctx_shift'] = alt_in_alt_ctx - alt_in_ref_ctx

            return metrics

        except Exception as e:
            warnings.warn(f"Failed to compute contextual probability metrics: {str(e)}")
            base_names = [f'ctx_shift_{nuc}' for nuc in NUCLEOTIDES]
            return {name: np.nan for name in base_names +
                   ['ctx_total_shift', 'ctx_max_shift', 'ref_allele_ctx_shift', 'alt_allele_ctx_shift']}

    def compute_variant_metrics(self, model, chrom: str, pos: int, ref: str, alt: str) -> Dict[str, float]:
        """
        Compute all pathogenicity prediction metrics for a single variant

        Returns dictionary with all computed metrics
        """
        # Validate variant
        if not self._validate_variant(chrom, pos, ref, alt):
            return {'error': 'invalid_variant'}

        try:
            # Get sequence contexts
            ref_seq, alt_seq, center_idx = self._get_sequence_context(chrom, pos, ref, alt)

            # Compute all metric types
            metrics = {}

            # 1. Embedding-based metrics (multiple geometric measures)
            emb_metrics = self._compute_embedding_metrics(model, ref_seq, alt_seq)
            metrics.update(emb_metrics)

            # 2. Basic probability metrics
            prob_metrics = self._compute_probability_metrics(
                model, ref_seq, alt_seq, center_idx, ref, alt
            )
            metrics.update(prob_metrics)

            # 3. Contextual probability comparison
            ctx_metrics = self._compute_contextual_probability_metrics(
                model, ref_seq, alt_seq, center_idx, ref, alt
            )
            metrics.update(ctx_metrics)

            return metrics

        except Exception as e:
            warnings.warn(f"Failed to process variant {chrom}:{pos} {ref}>{alt}: {str(e)}")
            return {'error': str(e)}

    def benchmark_dataset(self, model, df: pd.DataFrame,
                         sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Benchmark model on a dataset of variants

        Args:
            model: Model to benchmark
            df: DataFrame with variants (must have chrom, pos, ref, alt columns)
            sample_size: If provided, randomly sample this many variants

        Returns:
            DataFrame with original data plus computed metrics
        """
        # Prepare data
        df = df.copy()

        # Sample if requested
        if sample_size and len(df) > sample_size:
            df_path = df[df.clin_sig_clinvar == 'Pathogenic'].sample(n=int(sample_size // 2), random_state=self.config.random_state)
            df_benign = df[df.clin_sig_clinvar == 'Benign'].sample(n=int(sample_size // 2), random_state=self.config.random_state)
            df = pd.concat([df_path, df_benign])
            print(f"Sampled {sample_size} variants from {len(df)} total")

        # Filter to valid labels
        if self.config.label_col in df.columns:
            valid_labels = {'Benign', 'Pathogenic'}
            df = df[df[self.config.label_col].isin(valid_labels)]
            print(f"Filtered to {len(df)} variants with valid labels")

        results = []

        print(f"Processing {len(df)} variants...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                chrom = str(row[self.config.chrom_col])
                pos = int(row[self.config.pos_col])
                ref = str(row[self.config.ref_col]).upper()
                alt = str(row[self.config.alt_col]).upper()

                # Compute metrics
                metrics = self.compute_variant_metrics(model, chrom, pos, ref, alt)

                # Combine with original row data
                result_row = dict(row)
                result_row.update(metrics)
                results.append(result_row)

            except Exception as e:
                warnings.warn(f"Failed to process row {idx}: {str(e)}")
                result_row = dict(row)
                result_row.update({'error': str(e)})
                results.append(result_row)

        return pd.DataFrame(results)

    def evaluate_predictions(self, df: pd.DataFrame,
                           output_dir: Optional[Union[str, Path]] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of prediction metrics

        Args:
            df: DataFrame with metrics and labels
            output_dir: Directory to save evaluation plots and reports

        Returns:
            Dictionary with evaluation results
        """
        # Prepare labels
        df = df.copy()
        df = df.dropna(subset=[self.config.label_col])
        df['label_binary'] = df[self.config.label_col].map({'Benign': 0, 'Pathogenic': 1})
        df = df.dropna(subset=['label_binary'])

        if len(df) < self.config.min_samples_per_class:
            raise ValueError(f"Insufficient data: need at least {self.config.min_samples_per_class} samples")

        y_true = df['label_binary'].values.astype(int)

        # Get all computed metric columns (exclude metadata)
        exclude_cols = {
            self.config.chrom_col, self.config.pos_col, self.config.ref_col,
            self.config.alt_col, self.config.label_col, 'label_binary', 'error'
        }
        metric_cols = [col for col in df.columns
                      if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]

        evaluation_results = {}
        best_metrics = {}

        # Evaluate each metric
        for metric_col in metric_cols:
            values = df[metric_col].astype(float)

            # Skip if all NaN or constant
            if values.isna().all() or values.nunique() <= 1:
                continue

            # Handle NaN values
            mask = ~values.isna()
            if mask.sum() < self.config.min_samples_per_class:
                continue

            y_subset = y_true[mask]
            x_subset = values[mask]

            # Determine if we should invert the metric (lower = more pathogenic)
            # Based on naming patterns and biological intuition
            invert_metric = any(term in metric_col.lower() for term in [
                'similarity', 'prob_ref', 'log_prob_ref'
            ])

            if invert_metric:
                x_subset = -x_subset

            try:
                # ROC analysis
                fpr, tpr, _ = roc_curve(y_subset, x_subset)
                roc_auc = auc(fpr, tpr)

                # Precision-Recall
                precision, recall, _ = precision_recall_curve(y_subset, x_subset)
                pr_auc = average_precision_score(y_subset, x_subset)

                evaluation_results[f'{metric_col}_roc_auc'] = roc_auc
                evaluation_results[f'{metric_col}_pr_auc'] = pr_auc

                # Track best performing metric
                if not best_metrics or roc_auc > best_metrics.get('roc_auc', 0):
                    best_metrics = {
                        'metric_name': metric_col,
                        'roc_auc': roc_auc,
                        'pr_auc': pr_auc,
                        'inverted': invert_metric
                    }

            except Exception as e:
                warnings.warn(f"Failed to evaluate {metric_col}: {str(e)}")

        # Overall summary
        evaluation_results['best_metric'] = best_metrics.get('metric_name', 'none')
        evaluation_results['best_roc_auc'] = best_metrics.get('roc_auc', 0.0)
        evaluation_results['n_variants_evaluated'] = len(df)
        evaluation_results['n_pathogenic'] = (y_true == 1).sum()
        evaluation_results['n_benign'] = (y_true == 0).sum()

        # Save detailed results if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # # Save evaluation metrics
            # with open(output_path / 'evaluation_results.json', 'w') as f:
            #     json.dump(evaluation_results, f, indent=2)

            # Generate plots
            self._generate_evaluation_plots(df, metric_cols, output_path)

        return evaluation_results

    def _generate_evaluation_plots(self, df: pd.DataFrame, metric_cols: List[str],
                                 output_dir: Path):
        """Generate comprehensive evaluation plots"""
        y_true = df['label_binary'].values

        # ROC curves for top metrics
        top_metrics = []
        for col in metric_cols:
            values = df[col].astype(float)
            if values.isna().all() or values.nunique() <= 1:
                continue

            mask = ~values.isna()
            if mask.sum() < 10:
                continue

            x_subset = values[mask]
            y_subset = y_true[mask]

            # Check if should invert
            invert = any(term in col.lower() for term in [
                'similarity', 'prob_ref', 'log_prob_ref'
            ])
            if invert:
                x_subset = -x_subset

            try:
                auc_score = roc_auc_score(y_subset, x_subset)
                top_metrics.append((col, auc_score, invert))
            except:
                continue

        # Sort and take top 10
        top_metrics.sort(key=lambda x: x[1], reverse=True)
        top_metrics = top_metrics[:10]

        # Plot ROC curves
        plt.figure(figsize=(12, 8))

        for i, (metric_col, auc_score, inverted) in enumerate(top_metrics):
            values = df[metric_col].astype(float)
            mask = ~values.isna()
            x_subset = values[mask]
            y_subset = y_true[mask]

            if inverted:
                x_subset = -x_subset

            fpr, tpr, _ = roc_curve(y_subset, x_subset)

            label = f"{metric_col} (AUC={auc_score:.3f})"
            if inverted:
                label += " *"

            plt.plot(fpr, tpr, linewidth=2, label=label)

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Top Pathogenicity Prediction Metrics')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curves_top_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Correlation heatmap
        numeric_df = df[metric_cols].select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(14, 12))
            correlation_matrix = numeric_df.corr()

            # Mask upper triangle for cleaner visualization
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

            sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm',
                       center=0, square=True, linewidths=0.1, cbar_kws={"shrink": .8})
            plt.title('Correlation Matrix of Pathogenicity Prediction Metrics')
            plt.tight_layout()
            plt.savefig(output_dir / 'metric_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()

    def cross_validate(self, model, df: pd.DataFrame,
                      output_dir: Optional[Union[str, Path]] = None) -> Dict[str, float]:
        """
        Perform cross-validation evaluation

        Args:
            model: Model to evaluate
            df: Dataset with variants
            output_dir: Directory to save results

        Returns:
            Cross-validation results
        """
        df = df.copy()
        df = df.dropna(subset=[self.config.label_col])
        df['label_binary'] = df[self.config.label_col].map({'Benign': 0, 'Pathogenic': 1})
        df = df.dropna(subset=['label_binary'])

        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=self.config.n_folds,
                             shuffle=True,
                             random_state=self.config.random_state)

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(df, df['label_binary'])):
            print(f"Processing fold {fold + 1}/{self.config.n_folds}")

            test_df = df.iloc[test_idx]

            # Compute metrics on test set
            test_results = self.benchmark_dataset(model, test_df)

            # Evaluate
            fold_eval = self.evaluate_predictions(test_results)
            fold_eval['fold'] = fold
            fold_results.append(fold_eval)

        # Aggregate results
        cv_results = {}

        # Average across folds
        metric_keys = [k for k in fold_results[0].keys()
                      if k not in ['fold', 'best_metric'] and isinstance(fold_results[0][k], (int, float))]

        for key in metric_keys:
            values = [r[key] for r in fold_results if not np.isnan(r.get(key, np.nan))]
            if values:
                cv_results[f'{key}_mean'] = np.mean(values)
                cv_results[f'{key}_std'] = np.std(values)

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            with open(output_path / 'cross_validation_results.json', 'w') as f:
                json.dump({
                    'cv_summary': cv_results,
                    'fold_details': fold_results
                }, f, indent=2)

        return cv_results


def load_clinvar_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load and preprocess ClinVar data"""
    df = pd.read_csv(filepath)

    # Standardize column names to match expected format
    column_mapping = {
        'clin_sig_clinvar': 'clin_sig_clinvar',  # already correct
        # Add other mappings if needed
    }

    df = df.rename(columns=column_mapping)

    # Basic preprocessing
    df = df.dropna(subset=['chrom', 'pos', 'ref', 'alt'])
    df['chrom'] = df['chrom'].astype(str)
    df['pos'] = df['pos'].astype(int)
    df['ref'] = df['ref'].astype(str).str.upper()
    df['alt'] = df['alt'].astype(str).str.upper()

    return df


def run_full_benchmark(model,
                      output_dir: Union[str, Path],
                      sample_size: Optional[int] = None,
                      data_path: Union[str, Path] = '/tamir2/nicolaslynn/projects/dlm_wrappers/genebeddings/assets/benchmarks/clinvar_vep_subset.csv',
                      config: Optional[BenchmarkConfig] = None):
    """
    Run complete benchmarking pipeline

    Args:
        model: Model to benchmark
        data_path: Path to ClinVar CSV file
        output_dir: Directory for outputs
        sample_size: Number of variants to sample (None for all)
        config: Benchmark configuration
    """
    # Setup
    config = config or BenchmarkConfig()
    benchmarker = PathogenicityBenchmarker(config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading ClinVar data...")
    df = load_clinvar_data(data_path)
    df = df[df.chrom.isin(['1', '2', '3' , '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y'])]
    
    print(f"Loaded {len(df)} variants")
    display(df.head())

    # Run benchmarking
    print("Computing pathogenicity metrics...")
    results_df = benchmarker.benchmark_dataset(model, df, sample_size=sample_size)

    # Save raw results
    results_df.to_csv(output_path / 'pathogenicity_results.csv', index=False)
    print(f"Saved raw results to {output_path / 'pathogenicity_results.csv'}")

    # Evaluation
    print("Evaluating predictions...")
    evaluation = benchmarker.evaluate_predictions(results_df, output_dir=output_path)

    print("\nBenchmark Results:")
    print(f"Best metric: {evaluation.get('best_metric', 'none')}")
    print(f"Best ROC-AUC: {evaluation.get('best_roc_auc', 0.0):.3f}")
    print(f"Variants evaluated: {evaluation.get('n_variants_evaluated', 0)}")
    print(f"Pathogenic: {evaluation.get('n_pathogenic', 0)}")
    print(f"Benign: {evaluation.get('n_benign', 0)}")

    # Cross-validation (on subset if data is large)
    cv_sample_size = min(1000, len(results_df)) if sample_size is None else sample_size
    if cv_sample_size and cv_sample_size < len(results_df):
        print(f"\nRunning cross-validation on {cv_sample_size} variants...")
        cv_df = df.sample(n=cv_sample_size, random_state=config.random_state)
        cv_results = benchmarker.cross_validate(model, cv_df, output_dir=output_path)

        print("Cross-validation results:")
        for key, value in cv_results.items():
            if 'mean' in key:
                print(f"{key}: {value:.3f} ± {cv_results.get(key.replace('mean', 'std'), 0.0):.3f}")

    return results_df, evaluation


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Run pathogenicity prediction benchmark')
    parser.add_argument('--data', required=True, help='Path to ClinVar CSV file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--sample_size', type=int, help='Number of variants to sample')
    parser.add_argument('--model_name', default='nt', help='Model name/identifier')

    args = parser.parse_args()

    # This would need to be adapted based on your model loading setup
    # model = load_your_model(args.model_name)

    print(f"Note: This script requires a model object to run.")
    print(f"Please load your model and call run_full_benchmark(model, '{args.data}', '{args.output}', {args.sample_size})")