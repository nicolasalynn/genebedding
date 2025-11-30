#!/usr/bin/env python3
"""
Runner script for pathogenicity benchmarking with multiple models.

This script integrates the existing pathogenicity benchmark with the model
loading infrastructure from evaluate.py to benchmark multiple models on
ClinVar pathogenicity prediction.
"""

import sys
from pathlib import Path
import argparse
from typing import List, Dict, Optional
import json
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate import get_model, list_models
from pathogenicity_benchmark import (
    run_full_benchmark,
    BenchmarkConfig,
    PathogenicityBenchmarker,
    load_clinvar_data
)

DEFAULT_DATA_PATH = "../assets/benchmarks/clinvar_vep_subset.csv"
DEFAULT_OUTPUT_DIR = "./pathogenicity_results"

def benchmark_single_model(model_name: str, output_dir: Path, sample_size: Optional[int] = None,
                          data_path: str = DEFAULT_DATA_PATH, config: Optional[BenchmarkConfig] = None) -> Dict:
    """
    Benchmark a single model on pathogenicity prediction.

    Parameters
    ----------
    model_name : str
        Name of the model to benchmark
    output_dir : Path
        Output directory for results
    sample_size : int, optional
        Number of variants to sample for testing
    data_path : str
        Path to ClinVar data
    config : BenchmarkConfig, optional
        Benchmark configuration

    Returns
    -------
    dict
        Benchmark results summary
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking model: {model_name}")
    print(f"{'='*60}")

    try:
        # Load model
        model = get_model(model_name)
        print(f"✓ Successfully loaded model: {model_name}")

        # Create model-specific output directory
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Run benchmark
        print(f"Running pathogenicity benchmark...")
        run_full_benchmark(
            model=model,
            output_dir=model_output_dir,
            sample_size=sample_size,
            data_path=data_path,
            config=config
        )

        # Load and return summary results
        results_file = model_output_dir / 'pathogenicity_results.csv'
        evaluation_file = model_output_dir / 'evaluation_summary.json'

        summary = {
            'model_name': model_name,
            'status': 'success',
            'results_file': str(results_file),
            'output_dir': str(model_output_dir)
        }

        # Add evaluation metrics if available
        if evaluation_file.exists():
            try:
                with open(evaluation_file, 'r') as f:
                    eval_data = json.load(f)
                    summary.update(eval_data)
            except Exception as e:
                print(f"Warning: Could not load evaluation summary: {e}")

        print(f"✓ Benchmark completed for {model_name}")
        return summary

    except Exception as e:
        print(f"✗ Error benchmarking {model_name}: {e}")
        import traceback
        traceback.print_exc()

        return {
            'model_name': model_name,
            'status': 'error',
            'error': str(e)
        }

def benchmark_multiple_models(model_names: List[str], output_dir: str = DEFAULT_OUTPUT_DIR,
                             sample_size: Optional[int] = None, data_path: str = DEFAULT_DATA_PATH,
                             config: Optional[BenchmarkConfig] = None) -> Dict:
    """
    Benchmark multiple models on pathogenicity prediction.

    Parameters
    ----------
    model_names : list of str
        Names of models to benchmark
    output_dir : str
        Output directory for all results
    sample_size : int, optional
        Number of variants to sample for testing
    data_path : str
        Path to ClinVar data
    config : BenchmarkConfig, optional
        Benchmark configuration

    Returns
    -------
    dict
        Combined results from all models
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting pathogenicity benchmark for {len(model_names)} models")
    print(f"Models: {', '.join(model_names)}")
    print(f"Output directory: {output_path}")
    print(f"Sample size: {sample_size if sample_size else 'All variants'}")

    # Load and validate data first
    try:
        df = load_clinvar_data(data_path)
        print(f"✓ Loaded {len(df)} variants from ClinVar data")

        # Filter to standard chromosomes
        df = df[df.chrom.isin([str(i) for i in range(1, 23)] + ['X', 'Y'])]
        print(f"✓ Filtered to {len(df)} variants on standard chromosomes")

        # Check class balance
        class_dist = df['clin_sig_clinvar'].value_counts()
        print(f"Class distribution: {class_dist.to_dict()}")

    except Exception as e:
        raise RuntimeError(f"Failed to load or validate ClinVar data: {e}")

    # Benchmark each model
    results = {}

    for model_name in model_names:
        results[model_name] = benchmark_single_model(
            model_name=model_name,
            output_dir=output_path,
            sample_size=sample_size,
            data_path=data_path,
            config=config
        )

    # Generate comparison summary
    summary_file = output_path / 'benchmark_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    successful_models = [name for name, res in results.items() if res['status'] == 'success']
    failed_models = [name for name, res in results.items() if res['status'] == 'error']

    print(f"✓ Successful: {len(successful_models)} models: {', '.join(successful_models)}")
    if failed_models:
        print(f"✗ Failed: {len(failed_models)} models: {', '.join(failed_models)}")

    # Print metrics comparison for successful models
    metrics_comparison = []
    for model_name in successful_models:
        model_results = results[model_name]
        if 'best_roc_auc' in model_results:
            metrics_comparison.append({
                'Model': model_name,
                'ROC-AUC': model_results.get('best_roc_auc', 0.0),
                'Best Metric': model_results.get('best_metric', 'unknown'),
                'Variants': model_results.get('n_variants_evaluated', 0)
            })

    if metrics_comparison:
        comparison_df = pd.DataFrame(metrics_comparison)
        comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
        print(f"\nPerformance Comparison:")
        print(comparison_df.to_string(index=False, float_format='%.3f'))

        # Save comparison
        comparison_df.to_csv(output_path / 'model_comparison.csv', index=False)

    print(f"\nResults saved to: {output_path}")
    return results

def main():
    """Main function for running pathogenicity benchmarks."""
    parser = argparse.ArgumentParser(
        description='Benchmark genomic models on ClinVar pathogenicity prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=['nt', 'conformer'],
        help='Model names to benchmark'
    )
    parser.add_argument(
        '--data',
        default=DEFAULT_DATA_PATH,
        help='Path to ClinVar CSV file'
    )
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for results'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        help='Number of variants to sample (default: use all)'
    )
    parser.add_argument(
        '--context_length',
        type=int,
        default=1000,
        help='Context length around variants'
    )
    parser.add_argument(
        '--list_models',
        action='store_true',
        help='List available models and exit'
    )

    args = parser.parse_args()

    if args.list_models:
        available_models = list_models()
        print("Available models:")
        for model in available_models:
            print(f"  - {model}")
        return

    # Create benchmark configuration
    config = BenchmarkConfig(
        context_length=args.context_length,
        n_cv_folds=3 if args.sample_size and args.sample_size < 1000 else 5
    )

    # Validate models
    available_models = list_models()
    invalid_models = [m for m in args.models if m not in available_models]
    if invalid_models:
        print(f"Error: Invalid models: {invalid_models}")
        print(f"Available models: {available_models}")
        return 1

    try:
        # Run benchmark
        results = benchmark_multiple_models(
            model_names=args.models,
            output_dir=args.output,
            sample_size=args.sample_size,
            data_path=args.data,
            config=config
        )

        return 0

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())