#!/usr/bin/env python3
"""
Simple real-time fine-tuning for pathogenicity prediction.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from seqmat import SeqMat
except ImportError:
    SeqMat = None

from enhanced_pathogenicity_benchmark import (
    ClinVarDataset,
    ConvolutionalEmbeddingPredictor,
    MODEL_SEQUENCE_LENGTHS,
    DEFAULT_CLINVAR_PATH
)

def extract_single_variant_embeddings(model, variant_row, sequence_length: int, jitter: int = 0):
    """Extract WT and variant embeddings for a single variant with real-time jitter."""
    if SeqMat is None:
        raise ImportError("SeqMat is required")

    chrom = str(variant_row['chrom'])
    pos = int(variant_row['pos'])
    ref = variant_row['ref']
    alt = variant_row['alt']

    # Add 'chr' prefix if needed
    if not chrom.startswith('chr'):
        chrom = f'chr{chrom}'

    # Apply jitter (different each time this is called)
    half_length = sequence_length // 2
    if jitter > 0:
        offset = np.random.randint(-jitter, jitter + 1)
    else:
        offset = 0

    start_pos = max(1, pos - half_length + offset)
    end_pos = pos + half_length + offset

    try:
        # Extract sequences
        wt_seqmat = SeqMat.from_fasta('hg38', chrom, start_pos, end_pos)
        var_seqmat = wt_seqmat.clone()
        var_seqmat.apply_mutations([(pos, ref, alt)], permissive_ref=True)

        wt_seq = wt_seqmat.seq
        var_seq = var_seqmat.seq

        # Validate
        if len(wt_seq) != len(var_seq) or len(wt_seq) < sequence_length * 0.8:
            return None, None

        # Get embeddings
        wt_emb = model.embed(wt_seq, pool='mean', return_numpy=True)
        var_emb = model.embed(var_seq, pool='mean', return_numpy=True)

        return wt_emb, var_emb
    except:
        return None, None

def fine_tune_model(model, model_name: str, max_variants: int = 1000, epochs: int = 5,
                   jitter: int = 500, use_matrix_features: bool = True,
                   sequence_length: int = 2000, balance_consequences: bool = True,
                   dropout: float = 0.1, selected_consequences=None,
                   data_path: str = None, batch_size: int = 16):
    """Simple fine-tuning with real-time embeddings."""

    if data_path is None:
        data_path = DEFAULT_CLINVAR_PATH

    # Load data
    dataset = ClinVarDataset(
        data_path,
        max_samples=max_variants,
        balance_consequences=balance_consequences,
        selected_consequences=selected_consequences
    )
    df = dataset.load_and_preprocess()
    dataset.prepare_labels()

    print(f"Fine-tuning {model_name} on {len(df)} variants")

    # Split data
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['pathogenicity_label'], random_state=42
    )

    # Get first embedding to determine dimension
    for _, row in train_df.iterrows():
        wt_emb, var_emb = extract_single_variant_embeddings(model, row, sequence_length, jitter)
        if wt_emb is not None:
            embedding_dim = wt_emb.shape[0]
            break
    else:
        raise ValueError("Could not extract any embeddings")

    num_consequence_classes = len(df['consequence_label'].unique())

    # Create prediction model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred_model = ConvolutionalEmbeddingPredictor(
        embedding_dim=embedding_dim,
        num_consequence_classes=num_consequence_classes,
        use_matrix_features=use_matrix_features,
        use_conv_features=True,
        dropout=dropout
    ).to(device)

    # Training setup
    optimizer = torch.optim.Adam(pred_model.parameters(), lr=1e-4)
    path_criterion = nn.CrossEntropyLoss()
    cons_criterion = nn.CrossEntropyLoss()

    print(f"Training on {device} for {epochs} epochs")
    print(f"Model parameters: {sum(p.numel() for p in pred_model.parameters()):,}")

    # Training loop
    best_acc = 0.0

    for epoch in range(epochs):
        pred_model.train()
        train_loss = 0.0
        train_count = 0

        # Shuffle and batch
        shuffled_train = train_df.sample(frac=1).reset_index(drop=True)

        pbar = tqdm(range(0, len(shuffled_train), batch_size), desc=f"Epoch {epoch+1}")
        for i in pbar:
            batch_df = shuffled_train.iloc[i:i+batch_size]

            wt_batch, var_batch, path_batch, cons_batch = [], [], [], []

            # Real-time embedding with jitter
            for _, row in batch_df.iterrows():
                wt_emb, var_emb = extract_single_variant_embeddings(model, row, sequence_length, jitter)
                if wt_emb is not None:
                    wt_batch.append(wt_emb)
                    var_batch.append(var_emb)
                    path_batch.append(row['pathogenicity_label'])
                    cons_batch.append(row['consequence_label'])

            if len(wt_batch) == 0:
                continue

            # Convert to tensors
            wt_tensor = torch.tensor(np.array(wt_batch), dtype=torch.float32).to(device)
            var_tensor = torch.tensor(np.array(var_batch), dtype=torch.float32).to(device)
            path_tensor = torch.tensor(path_batch, dtype=torch.long).to(device)
            cons_tensor = torch.tensor(cons_batch, dtype=torch.long).to(device)

            optimizer.zero_grad()

            # Forward pass
            path_logits, cons_logits = pred_model(wt_tensor, var_tensor)

            # Loss
            path_loss = path_criterion(path_logits, path_tensor)
            cons_loss = cons_criterion(cons_logits, cons_tensor)
            total_loss = 2.0 * path_loss + cons_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(pred_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()
            train_count += 1

            pbar.set_postfix({'loss': f'{train_loss/train_count:.3f}'})

        # Validation
        pred_model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for i in range(0, len(test_df), batch_size):
                batch_df = test_df.iloc[i:i+batch_size]

                wt_batch, var_batch, path_batch = [], [], []

                for _, row in batch_df.iterrows():
                    wt_emb, var_emb = extract_single_variant_embeddings(model, row, sequence_length, jitter=0)  # No jitter for eval
                    if wt_emb is not None:
                        wt_batch.append(wt_emb)
                        var_batch.append(var_emb)
                        path_batch.append(row['pathogenicity_label'])

                if len(wt_batch) == 0:
                    continue

                wt_tensor = torch.tensor(np.array(wt_batch), dtype=torch.float32).to(device)
                var_tensor = torch.tensor(np.array(var_batch), dtype=torch.float32).to(device)

                path_logits, _ = pred_model(wt_tensor, var_tensor)
                path_preds = torch.argmax(path_logits, dim=1)

                val_preds.extend(path_preds.cpu().numpy())
                val_labels.extend(path_batch)

        # Metrics
        if val_preds:
            acc = accuracy_score(val_labels, val_preds)
            if acc > best_acc:
                best_acc = acc

            print(f"Epoch {epoch+1}: Loss={train_loss/train_count:.3f}, Acc={acc:.3f}, Best={best_acc:.3f}")

    return {
        'model_name': model_name,
        'pathogenicity_accuracy': best_acc,
        'sequence_length': sequence_length,
        'embedding_dim': embedding_dim,
        'num_variants_tested': len(test_df)
    }

if __name__ == "__main__":
    # Example usage matching your function call
    from evaluate import get_model

    results = fine_tune_model(
        model=get_model('convnova'),
        model_name='convnova',
        max_variants=1000,
        epochs=5,
        jitter=500,
        use_matrix_features=True,
        sequence_length=2000,
        balance_consequences=True,
        dropout=0.1,
        selected_consequences=['stop_gained', 'missense_variant', 'synonymous_variant']
    )

    print(f"\nFinal results: {results}")