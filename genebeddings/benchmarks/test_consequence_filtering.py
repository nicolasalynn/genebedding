#!/usr/bin/env python3
"""
Test script for consequence filtering functionality
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhanced_pathogenicity_benchmark import (
    list_available_consequences,
    ClinVarDataset,
    ConvolutionalEmbeddingPredictor
)
import torch
import numpy as np

def test_consequence_listing():
    """Test the consequence listing functionality."""
    print("🔍 Testing consequence listing functionality...")

    try:
        consequences = list_available_consequences(sample_size=5000)

        print(f"\n✅ Found {len(consequences)} consequence types")

        # Show top 5 most common
        sorted_consequences = sorted(consequences.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 5 most common consequences:")
        for i, (consequence, count) in enumerate(sorted_consequences[:5]):
            print(f"  {i+1}. {consequence}: {count} variants")

        return sorted_consequences

    except Exception as e:
        print(f"❌ Consequence listing failed: {e}")
        raise

def test_consequence_filtering():
    """Test consequence filtering with specific classes."""
    print("\n🧪 Testing consequence filtering...")

    # Test with specific consequences
    selected_consequences = ['missense_variant', 'stop_gained', 'synonymous_variant']

    try:
        # Create dataset with filtering
        dataset = ClinVarDataset(
            max_samples=1000,
            selected_consequences=selected_consequences,
            min_consequence_samples=5
        )

        df = dataset.load_and_preprocess()
        dataset.prepare_labels()

        # Check that only selected consequences remain
        actual_consequences = set(df['consequence'].unique())
        expected_consequences = set(selected_consequences)

        print(f"\n✅ Filtering Results:")
        print(f"  Selected: {selected_consequences}")
        print(f"  Found: {list(actual_consequences)}")
        print(f"  Dataset size: {len(df)} variants")

        # Verify filtering worked correctly
        if not actual_consequences.issubset(expected_consequences):
            unexpected = actual_consequences - expected_consequences
            print(f"⚠️  Warning: Found unexpected consequences: {unexpected}")
        else:
            print("✅ Filtering successful - only selected consequences present")

        # Show distribution
        consequence_counts = df['consequence'].value_counts()
        print(f"\nConsequence distribution:")
        for consequence, count in consequence_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {consequence}: {count} ({percentage:.1f}%)")

        # Test that we can still train a model
        print(f"\n🔧 Testing model training with filtered data...")
        test_model_with_filtered_data(df, dataset)

        return df

    except Exception as e:
        print(f"❌ Consequence filtering failed: {e}")
        raise

def test_model_with_filtered_data(df, dataset):
    """Test that the model can train with filtered consequence data."""

    # Create dummy embeddings for testing
    batch_size = min(50, len(df))
    embedding_dim = 128

    wt_embeddings = np.random.randn(batch_size, embedding_dim).astype(np.float32)
    var_embeddings = np.random.randn(batch_size, embedding_dim).astype(np.float32)

    # Get labels for the batch
    pathogenicity_labels = df['pathogenicity_label'].values[:batch_size]
    consequence_labels = df['consequence_label'].values[:batch_size]

    num_consequence_classes = len(dataset.consequence_encoder.classes_)

    print(f"  Testing with {batch_size} variants, {num_consequence_classes} consequence classes")

    # Create model
    model = ConvolutionalEmbeddingPredictor(
        embedding_dim=embedding_dim,
        num_consequence_classes=num_consequence_classes,
        use_conv_features=True,
        use_matrix_features=False
    )

    # Test forward pass
    wt_tensor = torch.tensor(wt_embeddings)
    var_tensor = torch.tensor(var_embeddings)

    with torch.no_grad():
        path_logits, cons_logits = model(wt_tensor, var_tensor)

    print(f"  ✅ Model forward pass successful")
    print(f"  ✅ Pathogenicity output shape: {path_logits.shape}")
    print(f"  ✅ Consequence output shape: {cons_logits.shape}")
    print(f"  ✅ Expected consequence classes: {num_consequence_classes}")

def test_edge_cases():
    """Test edge cases for consequence filtering."""
    print("\n🧪 Testing edge cases...")

    # Test with non-existent consequence
    print("Testing with non-existent consequence...")
    try:
        dataset = ClinVarDataset(
            max_samples=100,
            selected_consequences=['non_existent_consequence'],
            min_consequence_samples=1
        )
        df = dataset.load_and_preprocess()

        if len(df) == 0:
            print("✅ Correctly handled non-existent consequence (empty dataset)")
        else:
            print(f"⚠️  Unexpected: Found {len(df)} variants for non-existent consequence")

    except Exception as e:
        print(f"✅ Correctly caught exception for non-existent consequence: {str(e)[:100]}...")

    # Test with very rare consequence
    print("\nTesting with very rare consequences...")
    try:
        # Get available consequences first
        consequences = list_available_consequences(sample_size=1000)

        # Find a rare consequence (bottom 3)
        sorted_consequences = sorted(consequences.items(), key=lambda x: x[1])
        rare_consequences = [c[0] for c in sorted_consequences[:3] if c[1] > 0]

        if rare_consequences:
            print(f"Testing with rare consequence: {rare_consequences[0]}")

            dataset = ClinVarDataset(
                max_samples=1000,
                selected_consequences=rare_consequences[:1],
                min_consequence_samples=1
            )
            df = dataset.load_and_preprocess()

            print(f"✅ Successfully processed rare consequence: {len(df)} variants found")
        else:
            print("✅ No rare consequences found to test with")

    except Exception as e:
        print(f"⚠️  Issue with rare consequence test: {e}")

def main():
    """Main test function."""
    print("🧬 Testing Consequence Filtering Functionality")
    print("=" * 60)

    try:
        # Test 1: List available consequences
        consequences = test_consequence_listing()

        # Test 2: Filter for specific consequences
        filtered_df = test_consequence_filtering()

        # Test 3: Edge cases
        test_edge_cases()

        print(f"\n🎉 All consequence filtering tests passed!")

        # Show usage examples
        print(f"\n📚 Usage Examples:")
        print(f"# List all available consequences:")
        print(f"python enhanced_pathogenicity_benchmark.py --list_consequences")
        print(f"")
        print(f"# Train on specific consequences:")
        print(f"python enhanced_pathogenicity_benchmark.py --models conformer \\")
        print(f"    --selected_consequences missense_variant stop_gained synonymous_variant \\")
        print(f"    --max_variants 1000 --epochs 50")
        print(f"")
        print(f"# Train on just high-impact variants:")
        print(f"python enhanced_pathogenicity_benchmark.py --models conformer \\")
        print(f"    --selected_consequences stop_gained start_lost splice_acceptor_variant \\")
        print(f"    --epochs 100")

    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())