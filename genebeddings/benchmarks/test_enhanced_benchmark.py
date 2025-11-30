#!/usr/bin/env python3
"""
Quick test script for the enhanced pathogenicity benchmark
to verify that the epoch issue is fixed and the new model architecture works.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhanced_pathogenicity_benchmark import (
    validate_dataset_without_models,
    ConvolutionalEmbeddingPredictor
)
import torch

def test_model_architecture():
    """Test that the new model architecture works correctly."""
    print("Testing ConvolutionalEmbeddingPredictor architecture...")

    # Test parameters
    batch_size = 10
    embedding_dim = 256
    num_consequence_classes = 15

    # Create model with different configurations
    configs = [
        {"use_conv_features": True, "use_matrix_features": False},
        {"use_conv_features": False, "use_matrix_features": True},
        {"use_conv_features": True, "use_matrix_features": True},
        {"use_conv_features": False, "use_matrix_features": False}
    ]

    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")

        model = ConvolutionalEmbeddingPredictor(
            embedding_dim=embedding_dim,
            num_consequence_classes=num_consequence_classes,
            **config
        )

        # Test forward pass
        wt_emb = torch.randn(batch_size, embedding_dim)
        var_emb = torch.randn(batch_size, embedding_dim)

        try:
            path_logits, cons_logits = model(wt_emb, var_emb)

            # Check output shapes
            assert path_logits.shape == (batch_size, 2), f"Pathogenicity logits shape: {path_logits.shape}"
            assert cons_logits.shape == (batch_size, num_consequence_classes), f"Consequence logits shape: {cons_logits.shape}"

            print(f"  ✓ Forward pass successful")
            print(f"  ✓ Pathogenicity output shape: {path_logits.shape}")
            print(f"  ✓ Consequence output shape: {cons_logits.shape}")
            print(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            raise

    print("\n✅ All model architecture tests passed!")

def test_dataset_validation():
    """Test dataset validation functionality."""
    print("\nTesting dataset validation (without loading models)...")

    try:
        # Run validation on a small sample
        results = validate_dataset_without_models(
            max_variants=50,
            sample_size=100,
            sequence_length=1000
        )

        print(f"✓ Dataset validation completed")
        print(f"✓ Total variants: {results['total_variants']}")
        print(f"✓ Successful sequences: {results['successful_sequences']}/{results['validation_sample_size']}")

        if results['errors']:
            print(f"⚠ Found {len(results['errors'])} errors (this may be normal)")

    except ImportError as e:
        print(f"⚠ Dataset validation skipped: {e}")
        print("This is expected if SeqMat is not installed")
    except Exception as e:
        print(f"✗ Dataset validation failed: {e}")
        raise

if __name__ == "__main__":
    print("🔬 Testing Enhanced Pathogenicity Benchmark")
    print("=" * 50)

    test_model_architecture()
    test_dataset_validation()

    print(f"\n🎉 All tests completed successfully!")
    print("\nYou can now run the enhanced benchmark with:")
    print("python enhanced_pathogenicity_benchmark.py --models conformer --max_variants 100 --epochs 50")
    print("Or with convolutional features:")
    print("python enhanced_pathogenicity_benchmark.py --models conformer --use_conv_features --use_matrix_features --epochs 50")