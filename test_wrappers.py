#!/usr/bin/env python3
"""
Test script for genomic model wrappers.

Command-line usage:
    python test_wrappers.py                    # Interactive mode - choose models
    python test_wrappers.py --all              # Test all available models
    python test_wrappers.py --models nt borzoi # Test specific models
    python test_wrappers.py --quick            # Quick test (shorter sequences)

Notebook usage:
    # Option 1: If genebeddings package is installed
    from test_wrappers import run_tests
    results = run_tests(models=['nt', 'borzoi'])

    # Option 2: If using wrappers directory directly (add to sys.path first)
    import sys
    sys.path.append('/path/to/genebeddings/wrappers')
    from test_wrappers import run_tests
    results = run_tests(models=['nt'], quick=True)

    # Show available models
    run_tests()
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch

# Add the parent directory to sys.path to allow imports from genebeddings
# This allows the script to work when run from any directory
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")


def print_success(text: str):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_info(text: str):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def print_warning(text: str):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


# Model configurations
MODEL_CONFIGS = {
    'nt': {
        'name': 'Nucleotide Transformer',
        'class': 'NTWrapper',
        'module': 'nt_wrapper',  # For direct import when wrappers dir is in sys.path
        'import': 'from genebeddings.wrappers import NTWrapper',
        'init': 'NTWrapper(model_id="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")',
        'requires': ['transformers'],
        'seq_len': 1000,
        'capabilities': ['embed', 'predict_nucleotides'],
    },
    'caduceus': {
        'name': 'Caduceus',
        'class': 'CaduceusWrapper',
        'module': 'caduceus_wrapper',
        'import': 'from genebeddings.wrappers import CaduceusWrapper',
        'init': 'CaduceusWrapper(model_id="kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16")',
        'requires': ['transformers'],
        'seq_len': 1000,
        'capabilities': ['embed', 'predict_nucleotides'],
    },
    'specieslm': {
        'name': 'SpeciesLM',
        'class': 'SpeciesLMWrapper',
        'module': 'specieslm_wrapper',
        'import': 'from genebeddings.wrappers import SpeciesLMWrapper',
        'init': 'SpeciesLMWrapper()',
        'requires': ['transformers'],
        'seq_len': 1000,
        'capabilities': ['embed', 'predict_nucleotides'],
    },
    'rinalmo': {
        'name': 'RiNALMo',
        'class': 'RiNALMoWrapper',
        'module': 'rinalmo_wrapper',
        'import': 'from genebeddings.wrappers import RiNALMoWrapper',
        'init': 'RiNALMoWrapper(model_name="giga-v1")',
        'requires': ['rinalmo'],
        'seq_len': 1000,
        'capabilities': ['embed', 'predict_nucleotides'],
    },
    'borzoi': {
        'name': 'Borzoi',
        'class': 'BorzoiWrapper',
        'module': 'borzoi_wrapper',
        'import': 'from genebeddings.wrappers import BorzoiWrapper',
        'init': 'BorzoiWrapper(repo="johahi/borzoi-replicate-0")',
        'requires': ['borzoi_pytorch'],
        'seq_len': 524288,  # Borzoi needs long sequences
        'capabilities': ['embed', 'predict_tracks'],
    },
    'convnova': {
        'name': 'ConvNova',
        'class': 'ConvNovaWrapper',
        'module': 'convnova_wrapper',
        'import': 'from genebeddings.wrappers import ConvNovaWrapper',
        'init': 'ConvNovaWrapper()',  # Uses embedded model with defaults
        'requires': [],
        'seq_len': 1000,
        'capabilities': ['embed', 'predict_nucleotides'],
    },
}


def generate_test_sequence(length: int, quick: bool = False) -> str:
    """Generate a random DNA sequence for testing."""
    if quick:
        # For quick tests, use shorter sequences
        length = min(length, 500)

    np.random.seed(42)  # For reproducibility
    bases = ['A', 'C', 'G', 'T']
    return ''.join(np.random.choice(bases, size=length))


def test_embed(model: Any, seq: str, model_name: str) -> bool:
    """Test embedding functionality."""
    print_info(f"Testing embed() for {model_name}...")

    try:
        # Test mean pooling
        emb_mean = model.embed(seq, pool='mean', return_numpy=True)
        assert isinstance(emb_mean, np.ndarray), "embed() should return numpy array"
        assert emb_mean.ndim == 1, "Mean pooled embedding should be 1D"
        print_success(f"  Mean pooling: shape {emb_mean.shape}")

        # Test cls pooling
        emb_cls = model.embed(seq, pool='cls', return_numpy=True)
        assert isinstance(emb_cls, np.ndarray), "embed() should return numpy array"
        assert emb_cls.ndim == 1, "CLS pooled embedding should be 1D"
        print_success(f"  CLS pooling: shape {emb_cls.shape}")

        # Test token pooling
        emb_tokens = model.embed(seq, pool='tokens', return_numpy=True)
        assert isinstance(emb_tokens, np.ndarray), "embed() should return numpy array"
        assert emb_tokens.ndim == 2, "Token embeddings should be 2D"
        print_success(f"  Token embeddings: shape {emb_tokens.shape}")

        # Test return_numpy=False
        emb_torch = model.embed(seq, pool='mean', return_numpy=False)
        assert isinstance(emb_torch, torch.Tensor), "Should return torch.Tensor when return_numpy=False"
        print_success(f"  Torch tensor output: shape {emb_torch.shape}")

        return True
    except Exception as e:
        print_error(f"  embed() failed: {e}")
        return False


def test_predict_nucleotides(model: Any, seq: str, model_name: str) -> bool:
    """Test nucleotide prediction functionality."""
    print_info(f"Testing predict_nucleotides() for {model_name}...")

    try:
        # Test with a few positions
        positions = [10, 50, 100] if len(seq) > 100 else [5, 10, 15]

        # Test dict output
        results_dict = model.predict_nucleotides(seq, positions, return_dict=True)
        assert isinstance(results_dict, list), "Should return list when return_dict=True"
        assert len(results_dict) == len(positions), "Should return one result per position"

        for i, result in enumerate(results_dict):
            assert isinstance(result, dict), f"Result {i} should be dict"
            assert set(result.keys()) == {'A', 'C', 'G', 'T'}, f"Should have keys A, C, G, T but got {set(result.keys())}"
            total = sum(result.values())
            assert abs(total - 1.0) < 0.01, f"Probabilities should sum to 1, got {total}"

        print_success(f"  Dict output: {len(results_dict)} positions predicted")
        print_info(f"    Example at pos {positions[0]}: A={results_dict[0]['A']:.3f}, "
                   f"C={results_dict[0]['C']:.3f}, G={results_dict[0]['G']:.3f}, T={results_dict[0]['T']:.3f}")

        # Test array output
        results_array = model.predict_nucleotides(seq, positions, return_dict=False)
        assert isinstance(results_array, np.ndarray), "Should return numpy array when return_dict=False"
        assert results_array.shape == (len(positions), 4), f"Array should be shape ({len(positions)}, 4)"

        print_success(f"  Array output: shape {results_array.shape}")

        return True
    except Exception as e:
        print_error(f"  predict_nucleotides() failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predict_tracks(model: Any, seq: str, model_name: str) -> bool:
    """Test track prediction functionality."""
    print_info(f"Testing predict_tracks() for {model_name}...")

    try:
        tracks = model.predict_tracks(seq)
        assert isinstance(tracks, np.ndarray), "predict_tracks() should return numpy array"
        assert tracks.ndim == 2, "Tracks should be 2D (num_tracks, num_positions)"

        num_tracks, num_positions = tracks.shape
        print_success(f"  Predicted {num_tracks} tracks across {num_positions} positions")
        print_info(f"    Track value range: [{tracks.min():.3f}, {tracks.max():.3f}]")

        return True
    except Exception as e:
        print_error(f"  predict_tracks() failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_capability_discovery(model: Any, model_name: str, expected_caps: List[str]) -> bool:
    """Test capability discovery methods."""
    print_info(f"Testing capability discovery for {model_name}...")

    try:
        # Test get_capabilities()
        caps = model.get_capabilities()
        assert isinstance(caps, list), "get_capabilities() should return list"
        print_success(f"  Capabilities: {caps}")

        # Test supports_capability()
        for cap in expected_caps:
            supported = model.supports_capability(cap)
            assert supported, f"Should support '{cap}'"
            print_success(f"  Supports '{cap}': {supported}")

        # Test unsupported capability
        if 'predict_tracks' not in expected_caps:
            supported = model.supports_capability('predict_tracks')
            assert not supported, "Should not support 'predict_tracks'"
            print_success(f"  Does not support 'predict_tracks': {supported}")

        return True
    except Exception as e:
        print_error(f"  Capability discovery failed: {e}")
        return False


def test_model(model_key: str, quick: bool = False) -> Dict[str, bool]:
    """Test a single model."""
    config = MODEL_CONFIGS[model_key]

    print_header(f"Testing {config['name']}")

    # Check if manual setup required
    if config.get('manual_setup'):
        print_warning(f"{config['name']} requires manual setup (config files, checkpoints, etc.)")
        print_warning("Skipping this model. Please configure it manually to test.")
        return {'skipped': True}

    # Check dependencies
    missing_deps = []
    for dep in config['requires']:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)

    if missing_deps:
        print_warning(f"Missing dependencies: {', '.join(missing_deps)}")
        print_warning(f"Install with: pip install {' '.join(missing_deps)}")
        return {'skipped': True, 'reason': f"Missing: {', '.join(missing_deps)}"}

    # Initialize model
    print_info("Initializing model...")
    try:
        # Try the primary import statement (package import)
        exec(config['import'])
        model = eval(config['init'])
        print_success(f"Model initialized: {model}")
    except (ImportError, ModuleNotFoundError) as import_err:
        # If package import fails, try direct import (for when wrappers dir is in sys.path)
        try:
            print_info("Trying alternative import method (direct from wrappers dir)...")
            # Use the module name to import directly
            module_name = config['module']
            class_name = config['class']
            exec(f"from {module_name} import {class_name}")
            model = eval(config['init'])
            print_success(f"Model initialized via alternative import: {model}")
        except Exception as e2:
            print_error(f"Failed to initialize model with package import: {import_err}")
            print_error(f"Failed with alternative import (direct): {e2}")
            print_info("Hint: Make sure either:")
            print_info("  1. genebeddings package is installed, OR")
            print_info("  2. wrappers directory is added to sys.path (sys.path.append('/path/to/wrappers'))")
            import traceback
            traceback.print_exc()
            return {'init_failed': True, 'error': str(import_err)}
    except Exception as e:
        print_error(f"Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return {'init_failed': True, 'error': str(e)}

    # Generate test sequence
    seq_len = config['seq_len']
    if quick:
        seq_len = min(seq_len, 1000)
    seq = generate_test_sequence(seq_len, quick=quick)
    print_info(f"Generated test sequence of length {len(seq)}")

    # Run tests
    results = {}

    # Test capability discovery
    results['capability_discovery'] = test_capability_discovery(
        model, config['name'], config['capabilities']
    )

    # Test embed (always supported)
    if 'embed' in config['capabilities']:
        results['embed'] = test_embed(model, seq, config['name'])

    # Test predict_nucleotides if supported
    if 'predict_nucleotides' in config['capabilities']:
        results['predict_nucleotides'] = test_predict_nucleotides(model, seq, config['name'])

    # Test predict_tracks if supported
    if 'predict_tracks' in config['capabilities']:
        results['predict_tracks'] = test_predict_tracks(model, seq, config['name'])

    # Print summary
    print()
    passed = sum(1 for v in results.values() if v is True)
    total = len(results)

    if passed == total:
        print_success(f"All {total} tests passed for {config['name']}!")
    else:
        print_warning(f"Passed {passed}/{total} tests for {config['name']}")

    return results


def interactive_select_models() -> List[str]:
    """Interactively select models to test."""
    print_header("Available Models")

    available = []
    for i, (key, config) in enumerate(MODEL_CONFIGS.items(), 1):
        status = ""
        if config.get('manual_setup'):
            status = f"{Colors.WARNING}(requires manual setup){Colors.ENDC}"

        print(f"{i}. {config['name']} ({key}) {status}")
        available.append(key)

    print(f"\n{len(available) + 1}. Test all models")
    print("0. Exit")

    while True:
        try:
            choice = input(f"\n{Colors.BOLD}Select models to test (comma-separated numbers or 'all'): {Colors.ENDC}")
            choice = choice.strip().lower()

            if choice == '0':
                return []

            if choice == 'all' or choice == str(len(available) + 1):
                return available

            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected = [available[i] for i in indices if 0 <= i < len(available)]

            if selected:
                return selected
            else:
                print_error("Invalid selection. Please try again.")
        except (ValueError, IndexError):
            print_error("Invalid input. Please enter numbers separated by commas.")


def run_tests(models: Optional[List[str]] = None, test_all: bool = False, quick: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Run tests on genomic model wrappers (notebook-friendly version).

    Args:
        models: List of model keys to test (e.g., ['nt', 'borzoi']).
                If None and test_all is False, will show available models.
        test_all: If True, test all available models.
        quick: If True, use shorter sequences for faster testing.

    Returns:
        Dictionary mapping model keys to their test results.

    Examples:
        # Test specific models
        >>> results = run_tests(models=['nt', 'caduceus'])

        # Test all models
        >>> results = run_tests(test_all=True)

        # Quick test of specific model
        >>> results = run_tests(models=['borzoi'], quick=True)

        # Show available models (no arguments)
        >>> run_tests()
    """
    # Determine which models to test
    if test_all:
        models_to_test = list(MODEL_CONFIGS.keys())
    elif models:
        # Validate model names
        invalid = [m for m in models if m not in MODEL_CONFIGS]
        if invalid:
            print_error(f"Invalid model(s): {', '.join(invalid)}")
            print_info(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
            return {}
        models_to_test = models
    else:
        # Show available models
        print_header("Available Models")
        print_info("Call run_tests() with specific parameters:\n")
        print("  run_tests(models=['nt', 'borzoi'])  # Test specific models")
        print("  run_tests(test_all=True)            # Test all models")
        print("  run_tests(models=['nt'], quick=True) # Quick test\n")

        print("Available model keys:")
        for key, config in MODEL_CONFIGS.items():
            status = " (requires manual setup)" if config.get('manual_setup') else ""
            print(f"  - '{key}': {config['name']}{status}")
        return {}

    print_header(f"Testing {len(models_to_test)} Model(s)")
    if quick:
        print_info("Quick mode: using shorter sequences")

    # Test each model
    all_results = {}
    for model_key in models_to_test:
        results = test_model(model_key, quick=quick)
        all_results[model_key] = results

    # Print final summary
    print_header("Test Summary")

    for model_key, results in all_results.items():
        config = MODEL_CONFIGS[model_key]

        if results.get('skipped'):
            reason = results.get('reason', 'Manual setup required')
            print_warning(f"{config['name']}: Skipped ({reason})")
        elif results.get('init_failed'):
            print_error(f"{config['name']}: Initialization failed")
        else:
            passed = sum(1 for v in results.values() if v is True)
            total = len(results)

            if passed == total:
                print_success(f"{config['name']}: ✓ All {total} tests passed")
            else:
                print_warning(f"{config['name']}: {passed}/{total} tests passed")

    print()

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Test genomic model wrappers')
    parser.add_argument('--all', action='store_true', help='Test all available models')
    parser.add_argument('--models', nargs='+', choices=list(MODEL_CONFIGS.keys()),
                       help='Specific models to test')
    parser.add_argument('--quick', action='store_true', help='Quick test with shorter sequences')

    args = parser.parse_args()

    # Determine which models to test
    if args.all:
        models_to_test = list(MODEL_CONFIGS.keys())
    elif args.models:
        models_to_test = args.models
    else:
        # Interactive mode
        models_to_test = interactive_select_models()
        if not models_to_test:
            print_info("No models selected. Exiting.")
            return

    print_header(f"Testing {len(models_to_test)} Model(s)")
    if args.quick:
        print_info("Quick mode: using shorter sequences")

    # Test each model
    all_results = {}
    for model_key in models_to_test:
        results = test_model(model_key, quick=args.quick)
        all_results[model_key] = results

    # Print final summary
    print_header("Test Summary")

    for model_key, results in all_results.items():
        config = MODEL_CONFIGS[model_key]

        if results.get('skipped'):
            reason = results.get('reason', 'Manual setup required')
            print_warning(f"{config['name']}: Skipped ({reason})")
        elif results.get('init_failed'):
            print_error(f"{config['name']}: Initialization failed")
        else:
            passed = sum(1 for v in results.values() if v is True)
            total = len(results)

            if passed == total:
                print_success(f"{config['name']}: ✓ All {total} tests passed")
            else:
                print_warning(f"{config['name']}: {passed}/{total} tests passed")

    print()


if __name__ == '__main__':
    main()
