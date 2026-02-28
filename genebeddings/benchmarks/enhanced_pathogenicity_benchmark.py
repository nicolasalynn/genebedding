"""
Enhanced Pathogenicity Benchmark for Fine-tuning Prediction Heads

This benchmark fine-tunes prediction heads on model embeddings to predict variant consequences
from ClinVar data. Each model uses its optimal sequence length:
- ConvNova: 1,000 nucleotides
- NT: 6,000 nucleotides
- Borzoi: 520,000 nucleotides
- Caduceus: 100,000 nucleotides

The prediction head takes both wild-type and variant embeddings to predict:
1. Variant consequence (multi-class)
2. Clinical significance (pathogenic/benign)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from typing import List, Dict, Tuple, Optional, Union
import sys
from pathlib import Path
from tqdm import tqdm
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from seqmat import SeqMat
except ImportError:
    print("Warning: SeqMat not available. Some functionality may be limited.")
    SeqMat = None

# Import model loading function
try:
    from evaluate import get_model
except ImportError:
    print("Warning: evaluate.py not available. get_model function may not work.")
    def get_model(model_name):
        raise ImportError("get_model function not available - evaluate.py not imported")


# Default data path - works from any directory
DEFAULT_CLINVAR_PATH = str(Path(__file__).parent.parent / 'assets' / 'benchmarks' / 'clinvar_vep_subset.csv')

# Model-specific sequence lengths (nucleotides around mutation site)
MODEL_SEQUENCE_LENGTHS = {
    'convnova': 1000,
    'nt': 6000,
    'borzoi': 520_000,
    'caduceus': 100_000,
    'conformer': 3000,  # Adding conformer with reasonable length
    'dnabert': 1000,
    'specieslm': 2000,
    'rinalmo': 1000
}

class ClinVarDataset:
    """Dataset handler for ClinVar pathogenicity data."""

    def __init__(self, data_path: str = None, max_samples: Optional[int] = None,
                 balance_consequences: bool = False, target_samples_per_consequence: Optional[int] = None,
                 min_consequence_samples: int = 10, selected_consequences: Optional[List[str]] = None):
        self.data_path = data_path
        self.max_samples = max_samples
        self.balance_consequences = balance_consequences
        self.target_samples_per_consequence = target_samples_per_consequence
        self.min_consequence_samples = min_consequence_samples
        self.selected_consequences = selected_consequences
        self.df = None
        self.pathogenicity_encoder = None
        self.consequence_encoder = None

    def load_and_preprocess(self) -> pd.DataFrame:
        """Load and preprocess ClinVar data."""
        if self.data_path is None:
            self.data_path = DEFAULT_CLINVAR_PATH
        print(f"Loading ClinVar data from {self.data_path}")
        df = pd.read_csv(self.data_path)

        print(f"Initial dataset size: {len(df)}")

        # Filter for clear pathogenic/benign cases
        df = df[df['clin_sig_clinvar'].isin(['Pathogenic', 'Benign'])].copy()
        print(f"After pathogenicity filtering: {len(df)}")

        # Filter for SNVs (single nucleotide variants) to start
        df = df[(df['ref'].str.len() == 1) & (df['alt'].str.len() == 1)].copy()
        print(f"After SNV filtering: {len(df)}")

        # Remove variants with missing information
        df = df.dropna(subset=['gene', 'consequence', 'chrom', 'pos']).copy()
        print(f"After removing missing data: {len(df)}")

        # Filter to standard chromosomes
        standard_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y']
        df = df[df['chrom'].astype(str).isin(standard_chroms)].copy()
        print(f"After chromosome filtering: {len(df)}")

        # Filter for specific consequence classes if requested
        if self.selected_consequences is not None:
            print(f"Filtering for specific consequences: {self.selected_consequences}")
            initial_count = len(df)
            df = df[df['consequence'].isin(self.selected_consequences)].copy()
            print(f"After consequence filtering: {len(df)} (removed {initial_count - len(df)} variants)")

            # Check if we have enough data for each selected consequence
            consequence_counts = df['consequence'].value_counts()
            print("Selected consequence distributions:")
            for consequence in self.selected_consequences:
                count = consequence_counts.get(consequence, 0)
                print(f"  {consequence}: {count} variants")
                if count == 0:
                    print(f"  ⚠️  Warning: No variants found for consequence '{consequence}'")
                elif count < self.min_consequence_samples:
                    print(f"  ⚠️  Warning: Only {count} variants for '{consequence}' (min: {self.min_consequence_samples})")

        # Sample if requested
        if self.max_samples and len(df) > self.max_samples:
            # Stratified sampling to maintain class balance
            df = df.groupby('clin_sig_clinvar').apply(
                lambda x: x.sample(min(len(x), self.max_samples // 2), random_state=42)
            ).reset_index(drop=True)
            print(f"After sampling: {len(df)}")

        # Print class distributions
        print("\nClass distributions:")
        print("Pathogenicity:", df['clin_sig_clinvar'].value_counts().to_dict())
        print("Top consequences:", df['consequence'].value_counts().head(10).to_dict())

        # Balance consequences if requested
        if self.balance_consequences:
            df = self._balance_consequences(df)
            print(f"After balancing consequences: {len(df)}")

        self.df = df
        return df

    def _balance_consequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the dataframe by consequences using resampling.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            Balanced dataframe
        """
        print("\nBalancing consequences...")

        # Get consequence counts before balancing
        consequence_counts = df['consequence'].value_counts()
        print(f"Original consequences: {len(consequence_counts)} types")
        print(f"Original size range: {consequence_counts.min()} to {consequence_counts.max()}")

        # Filter out consequences with too few samples
        valid_consequences = consequence_counts[consequence_counts >= self.min_consequence_samples].index
        df_filtered = df[df['consequence'].isin(valid_consequences)].copy()

        removed_consequences = len(consequence_counts) - len(valid_consequences)
        if removed_consequences > 0:
            print(f"Filtered out {removed_consequences} consequences with < {self.min_consequence_samples} samples")

        # Determine target sample size per consequence
        if self.target_samples_per_consequence is None:
            # Use median as target to balance between under/over sampling
            valid_counts = consequence_counts[valid_consequences]
            target_size = int(valid_counts.median())
        else:
            target_size = self.target_samples_per_consequence

        print(f"Target samples per consequence: {target_size}")

        # Resample each consequence to target size
        balanced_dfs = []

        for consequence in valid_consequences:
            consequence_df = df_filtered[df_filtered['consequence'] == consequence]
            current_size = len(consequence_df)

            if current_size > target_size:
                # Downsample
                resampled_df = resample(
                    consequence_df,
                    n_samples=target_size,
                    random_state=42,
                    replace=False
                )
            elif current_size < target_size:
                # Upsample with replacement
                resampled_df = resample(
                    consequence_df,
                    n_samples=target_size,
                    random_state=42,
                    replace=True
                )
            else:
                # Perfect size, keep as is
                resampled_df = consequence_df

            balanced_dfs.append(resampled_df)

        # Combine all balanced consequences
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)

        # Shuffle the final dataframe
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Print final balance report
        final_counts = balanced_df['consequence'].value_counts()
        print(f"Balanced consequences: {len(final_counts)} types")
        print(f"Balanced size range: {final_counts.min()} to {final_counts.max()}")
        print(f"Final dataset size: {len(balanced_df)}")

        # Maintain original pathogenicity balance within each consequence
        print("\nPathogenicity balance in balanced dataset:")
        print(balanced_df['clin_sig_clinvar'].value_counts().to_dict())

        return balanced_df

    def prepare_labels(self):
        """Prepare label encoders for pathogenicity and consequence prediction."""
        if self.df is None:
            raise ValueError("Must load data first")

        # Encode pathogenicity (binary: 0=Benign, 1=Pathogenic)
        self.pathogenicity_encoder = LabelEncoder()
        self.df['pathogenicity_label'] = self.pathogenicity_encoder.fit_transform(self.df['clin_sig_clinvar'])

        # Encode consequences (multi-class)
        self.consequence_encoder = LabelEncoder()
        self.df['consequence_label'] = self.consequence_encoder.fit_transform(self.df['consequence'])

        print(f"Pathogenicity classes: {list(self.pathogenicity_encoder.classes_)}")
        print(f"Number of consequence classes: {len(self.consequence_encoder.classes_)}")
        print(f"All consequence classes: {list(self.consequence_encoder.classes_)}")

def list_available_consequences(data_path: str = None, sample_size: int = 10000) -> Dict[str, int]:
    """
    List all available consequence types in the dataset with their counts.

    Parameters
    ----------
    data_path : str, optional
        Path to ClinVar data
    sample_size : int
        Number of samples to examine (for faster analysis)

    Returns
    -------
    dict
        Dictionary mapping consequence type to count
    """
    if data_path is None:
        data_path = DEFAULT_CLINVAR_PATH

    print(f"📊 Analyzing consequence types in: {data_path}")

    # Create dataset instance for preprocessing
    dataset = ClinVarDataset(data_path, max_samples=sample_size)
    df = dataset.load_and_preprocess()

    # Get consequence counts
    consequence_counts = df['consequence'].value_counts()

    print(f"\n📋 Available Consequence Types ({len(consequence_counts)} total):")
    print("=" * 60)
    for i, (consequence, count) in enumerate(consequence_counts.items()):
        percentage = (count / len(df)) * 100
        print(f"{i+1:2d}. {consequence:<40} {count:>6} ({percentage:4.1f}%)")

    print(f"\nTotal variants: {len(df):,}")
    print(f"Pathogenicity distribution: {df['clin_sig_clinvar'].value_counts().to_dict()}")

    return consequence_counts.to_dict()

class SimpleEmbeddingPredictor(nn.Module):
    """Simple, lightweight neural network for pathogenicity prediction."""

    def __init__(self, embedding_dim: int, num_consequence_classes: int,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Simple feature engineering: just basic differences and ratios
        # Input: [wt_emb, var_emb, diff_emb, ratio_emb] = 4 * embedding_dim
        input_dim = embedding_dim * 4

        print(f"SimpleEmbeddingPredictor: embedding_dim={embedding_dim}")
        print(f"Input dimension: {input_dim}")

        # Simple 3-layer network
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Prediction heads
        self.pathogenicity_head = nn.Linear(hidden_dim // 2, 2)
        self.consequence_head = nn.Linear(hidden_dim // 2, num_consequence_classes)

    def forward(self, wt_emb, var_emb):
        # Simple feature engineering
        diff_emb = var_emb - wt_emb
        ratio_emb = torch.where(
            torch.abs(wt_emb) > 1e-8,
            var_emb / (wt_emb + 1e-8),
            torch.zeros_like(var_emb)
        )

        # Concatenate features
        features = torch.cat([wt_emb, var_emb, diff_emb, ratio_emb], dim=1)

        # Forward pass
        x = self.feature_net(features)

        path_logits = self.pathogenicity_head(x)
        cons_logits = self.consequence_head(x)

        return path_logits, cons_logits

class ConvolutionalEmbeddingPredictor(nn.Module):
    """
    Advanced neural network that predicts variant consequences and pathogenicity
    from wild-type and variant embeddings using convolutional feature extraction
    for capturing local patterns and spatial relationships.
    """

    def __init__(self, embedding_dim: int, num_consequence_classes: int,
                 hidden_dim: int = 512, dropout: float = 0.3, use_matrix_features: bool = False,
                 use_conv_features: bool = True, conv_channels: List[int] = None,
                 conv_kernel_sizes: List[int] = None):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.use_matrix_features = use_matrix_features
        self.use_conv_features = use_conv_features

        # Default convolutional parameters
        if conv_channels is None:
            conv_channels = [64, 128, 256, 512]
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [3, 5, 7, 11]  # Different kernel sizes for multi-scale features

        self.conv_channels = conv_channels
        self.conv_kernel_sizes = conv_kernel_sizes

        # Calculate input dimension for base features
        base_features = embedding_dim * 6  # wt_emb, var_emb, diff_emb, ratio_emb, hadamard_emb, cosine_sim

        print(f"ConvolutionalEmbeddingPredictor: embedding_dim={embedding_dim}")
        print(f"Features: conv={use_conv_features}, matrix={use_matrix_features}")
        print(f"Base features dimension: {base_features}")

        # Convolutional feature extraction (if enabled)
        if use_conv_features:
            # Create convolutional layers for different scales
            self.conv_layers = nn.ModuleList()
            self.conv_pools = nn.ModuleList()

            for i, (channels, kernel_size) in enumerate(zip(conv_channels, conv_kernel_sizes)):
                # Conv1d layers to capture local patterns in embedding space
                conv_block = nn.Sequential(
                    nn.Conv1d(in_channels=2, out_channels=channels, kernel_size=kernel_size,
                             padding=kernel_size//2),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(in_channels=channels, out_channels=channels//2, kernel_size=3,
                             padding=1),
                    nn.BatchNorm1d(channels//2),
                    nn.ReLU()
                )
                self.conv_layers.append(conv_block)

                # Adaptive pooling to get fixed-size outputs
                self.conv_pools.append(nn.AdaptiveAvgPool1d(embedding_dim // 4))

            # Calculate conv features dimension
            conv_features = sum([ch//2 * (embedding_dim//4) for ch in conv_channels])
            print(f"Convolutional features dimension: {conv_features}")
        else:
            conv_features = 0

        # Matrix interaction features (if enabled)
        if use_matrix_features:
            # Process the outer product interaction matrix using convolutional layers
            matrix_dim = min(embedding_dim * embedding_dim, 2048)  # Cap matrix size

            # Use 2D convolutions on the interaction matrix
            matrix_size = int(embedding_dim ** 0.5) if embedding_dim >= 16 else embedding_dim
            self.matrix_conv = nn.Sequential(
                # Reshape matrix and apply 2D convolutions
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((8, 8))  # Fixed output size
            )
            matrix_features = 64 * 8 * 8  # 4096 features
            print(f"Matrix conv features dimension: {matrix_features}")
        else:
            matrix_features = 0

        # Calculate total input dimension
        total_input_dim = base_features + conv_features + matrix_features
        print(f"Total input dimension: {total_input_dim}")

        # Sophisticated feature extraction network with residual connections
        self.feature_extractor = nn.ModuleList([
            # Layer 1
            nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            # Layer 2 with residual
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            # Layer 3
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            # Layer 4
            nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        ])

        # Project for residual connections
        self.residual_projection = nn.Linear(total_input_dim, hidden_dim)

        # Task-specific prediction heads with deeper architecture
        # Pathogenicity head (binary classification)
        self.pathogenicity_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 8, 2)
        )

        # Consequence head (multi-class classification)
        self.consequence_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 8, num_consequence_classes)
        )

    def forward(self, wt_emb, var_emb):
        batch_size = wt_emb.shape[0]

        # Enhanced feature engineering
        # 1. Basic difference and ratio features
        diff_emb = var_emb - wt_emb

        # Improved ratio computation with better numerical stability
        wt_norm = torch.norm(wt_emb, dim=1, keepdim=True) + 1e-8
        var_norm = torch.norm(var_emb, dim=1, keepdim=True) + 1e-8
        ratio_emb = var_emb / (wt_emb + 1e-8)

        # 2. Hadamard (element-wise) product
        hadamard_emb = wt_emb * var_emb

        # 3. Cosine similarity features
        cosine_sim = F.cosine_similarity(wt_emb, var_emb, dim=1, eps=1e-8).unsqueeze(1)
        cosine_sim_expanded = cosine_sim.expand(-1, self.embedding_dim)

        # Concatenate base features
        base_features = torch.cat([
            wt_emb, var_emb, diff_emb, ratio_emb, hadamard_emb, cosine_sim_expanded
        ], dim=1)

        features_list = [base_features]

        # Convolutional interaction features
        if self.use_conv_features:
            # Stack embeddings for convolutional processing: [wt_emb, var_emb]
            stacked_embs = torch.stack([wt_emb, var_emb], dim=1)  # (batch_size, 2, embedding_dim)

            # Apply different convolutional layers with different kernel sizes
            conv_outputs = []
            for conv_layer, conv_pool in zip(self.conv_layers, self.conv_pools):
                # Apply convolution
                conv_out = conv_layer(stacked_embs)  # (batch_size, channels//2, embedding_dim)

                # Apply adaptive pooling to get consistent size
                pooled_out = conv_pool(conv_out)  # (batch_size, channels//2, embedding_dim//4)

                # Flatten the output
                conv_outputs.append(pooled_out.flatten(1))

            # Concatenate all convolutional features
            if conv_outputs:
                conv_features = torch.cat(conv_outputs, dim=1)
                features_list.append(conv_features)

        # Matrix interaction features with 2D convolutions
        if self.use_matrix_features:
            # Compute outer product matrix
            interaction_matrix = torch.bmm(
                wt_emb.unsqueeze(-1),      # (batch_size, embedding_dim, 1)
                var_emb.unsqueeze(1)       # (batch_size, 1, embedding_dim)
            )
            # interaction_matrix shape: (batch_size, embedding_dim, embedding_dim)

            # For 2D convolution, just use the matrix as is (embedding_dim x embedding_dim)
            matrix_2d = interaction_matrix.unsqueeze(1)  # Add channel dimension
            # Shape: (batch_size, 1, embedding_dim, embedding_dim)

            # Apply 2D convolutions
            matrix_features = self.matrix_conv(matrix_2d)
            matrix_features_flat = matrix_features.view(batch_size, -1)
            features_list.append(matrix_features_flat)

        # Combine all features
        combined_features = torch.cat(features_list, dim=1)

        # Feature extraction with residual connections
        x = combined_features
        residual = self.residual_projection(x)

        # First layer
        x = self.feature_extractor[0](x)

        # Second layer with residual connection
        x_layer2 = self.feature_extractor[1](x)
        x = x_layer2 + residual  # Residual connection

        # Remaining layers
        x = self.feature_extractor[2](x)
        x = self.feature_extractor[3](x)

        # Task-specific predictions
        pathogenicity_logits = self.pathogenicity_head(x)
        consequence_logits = self.consequence_head(x)

        return pathogenicity_logits, consequence_logits


# Keep the original class for backwards compatibility
AdvancedEmbeddingPredictor = ConvolutionalEmbeddingPredictor  # Transition alias
DualEmbeddingPredictor = ConvolutionalEmbeddingPredictor  # Original alias

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

def train_prediction_head_realtime(model, df, model_name: str, sequence_length: int, jitter: int = 0,
                                   epochs: int = 100, batch_size: int = 32, lr: float = 1e-3,
                                   test_size: float = 0.2, device: str = 'auto',
                                   use_matrix_features: bool = False, use_conv_features: bool = True,
                                   hidden_dim: int = 512, dropout: float = 0.3, simple_model: bool = True) -> Dict:
    """Train with real-time embedding generation during training."""

    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Split data
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['pathogenicity_label'], random_state=42)

    # Get first embedding to determine dimension
    first_row = train_df.iloc[0]
    wt_emb, var_emb = extract_single_variant_embeddings(model, first_row, sequence_length, jitter)
    if wt_emb is None:
        raise ValueError("Could not extract embeddings for first variant")

    embedding_dim = wt_emb.shape[0]
    num_consequence_classes = len(df['consequence_label'].unique())

    # Initialize prediction model - use simple model by default
    if simple_model and not (use_matrix_features or use_conv_features):
        pred_model = SimpleEmbeddingPredictor(
            embedding_dim=embedding_dim,
            num_consequence_classes=num_consequence_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(device)
    else:
        pred_model = ConvolutionalEmbeddingPredictor(
            embedding_dim=embedding_dim,
            num_consequence_classes=num_consequence_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_matrix_features=use_matrix_features,
            use_conv_features=use_conv_features
        ).to(device)

    # Loss and optimizer
    path_criterion = nn.CrossEntropyLoss()
    cons_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pred_model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Training on {len(train_df)} variants, testing on {len(test_df)}")
    print(f"Embedding dim: {embedding_dim}, Consequence classes: {num_consequence_classes}")
    print(f"Model parameters: {sum(p.numel() for p in pred_model.parameters()):,}")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        pred_model.train()
        train_loss = 0.0
        train_count = 0

        # Shuffle training data
        shuffled_train = train_df.sample(frac=1).reset_index(drop=True)

        # Process in batches
        for i in range(0, len(shuffled_train), batch_size):
            batch_df = shuffled_train.iloc[i:i+batch_size]

            wt_batch, var_batch, path_batch, cons_batch = [], [], [], []

            # Extract embeddings for this batch (real-time with different jitter each time)
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

            # Losses
            path_loss = path_criterion(path_logits, path_tensor)
            cons_loss = cons_criterion(cons_logits, cons_tensor)
            total_loss = 2.0 * path_loss + 1.0 * cons_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(pred_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()
            train_count += 1

        # Validation (no jitter for consistency)
        pred_model.eval()
        val_loss = 0.0
        val_count = 0
        all_path_preds, all_path_labels = [], []

        with torch.no_grad():
            for i in range(0, len(test_df), batch_size):
                batch_df = test_df.iloc[i:i+batch_size]

                wt_batch, var_batch, path_batch, cons_batch = [], [], [], []

                for _, row in batch_df.iterrows():
                    wt_emb, var_emb = extract_single_variant_embeddings(model, row, sequence_length, jitter=0)  # No jitter for eval
                    if wt_emb is not None:
                        wt_batch.append(wt_emb)
                        var_batch.append(var_emb)
                        path_batch.append(row['pathogenicity_label'])
                        cons_batch.append(row['consequence_label'])

                if len(wt_batch) == 0:
                    continue

                wt_tensor = torch.tensor(np.array(wt_batch), dtype=torch.float32).to(device)
                var_tensor = torch.tensor(np.array(var_batch), dtype=torch.float32).to(device)
                path_tensor = torch.tensor(path_batch, dtype=torch.long).to(device)
                cons_tensor = torch.tensor(cons_batch, dtype=torch.long).to(device)

                path_logits, cons_logits = pred_model(wt_tensor, var_tensor)

                path_loss = path_criterion(path_logits, path_tensor)
                cons_loss = cons_criterion(cons_logits, cons_tensor)
                total_loss = 2.0 * path_loss + 1.0 * cons_loss

                val_loss += total_loss.item()
                val_count += 1

                # Store predictions
                path_preds = torch.argmax(path_logits, dim=1)
                all_path_preds.extend(path_preds.cpu().numpy())
                all_path_labels.extend(path_tensor.cpu().numpy())

        avg_train_loss = train_loss / max(train_count, 1)
        avg_val_loss = val_loss / max(val_count, 1)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if epoch % 10 == 0 or epoch < 5:
            acc = accuracy_score(all_path_labels, all_path_preds) if all_path_labels else 0
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.3f}, Val={avg_val_loss:.3f}, Acc={acc:.3f}")

    # Final evaluation
    final_acc = accuracy_score(all_path_labels, all_path_preds) if all_path_labels else 0
    try:
        final_auc = roc_auc_score(all_path_labels, all_path_preds) if len(set(all_path_labels)) > 1 else 0
    except Exception:
        # e.g. single class in y_true; roc_auc_score undefined
        final_auc = 0

    return {
        'model_name': model_name,
        'pathogenicity_accuracy': final_acc,
        'pathogenicity_auc': final_auc,
        'sequence_length': sequence_length,
        'num_variants_tested': len(test_df),
        'embedding_dim': embedding_dim,
        'trained_model': pred_model
    }
    """
    Train the dual embedding prediction head.

    Parameters
    ----------
    embeddings_data : dict
        Dictionary containing WT and variant embeddings
    labels_data : dict
        Dictionary containing pathogenicity and consequence labels
    model_name : str
        Name of the model being fine-tuned
    epochs : int
        Number of training epochs
    batch_size : int
        Training batch size
    lr : float
        Learning rate
    test_size : float
        Fraction for test set
    device : str
        Training device ('auto', 'cuda', 'cpu')
    use_matrix_features : bool
        Whether to use matrix features from WT and variant embedding outer product
    use_conv_features : bool
        Whether to use convolutional features for embedding interactions
    hidden_dim : int
        Hidden dimension size for the neural network
    dropout : float
        Dropout rate for regularization
    selected_consequences : list of str, optional
        Specific consequence types to include in training. If None, uses all consequences.

    Returns
    -------
    dict
        Training results and metrics
    """
    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Training on device: {device}")

    # Prepare data
    wt_emb = embeddings_data['wt_embeddings'].astype(np.float32)
    var_emb = embeddings_data['var_embeddings'].astype(np.float32)
    valid_indices = embeddings_data['valid_indices']

    # Get labels for valid indices
    pathogenicity_labels = labels_data['pathogenicity_labels'][valid_indices]
    consequence_labels = labels_data['consequence_labels'][valid_indices]

    print(f"Training data shape: {wt_emb.shape}")
    print(f"Pathogenicity distribution: {np.bincount(pathogenicity_labels)}")
    print(f"Number of consequence classes: {len(np.unique(consequence_labels))}")

    # Split data
    indices = np.arange(len(wt_emb))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=pathogenicity_labels, random_state=42
    )

    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(wt_emb[train_idx]),
        torch.tensor(var_emb[train_idx]),
        torch.tensor(pathogenicity_labels[train_idx], dtype=torch.long),
        torch.tensor(consequence_labels[train_idx], dtype=torch.long)
    )

    test_dataset = TensorDataset(
        torch.tensor(wt_emb[test_idx]),
        torch.tensor(var_emb[test_idx]),
        torch.tensor(pathogenicity_labels[test_idx], dtype=torch.long),
        torch.tensor(consequence_labels[test_idx], dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    embedding_dim = wt_emb.shape[1]
    num_consequence_classes = len(np.unique(consequence_labels))

    model = ConvolutionalEmbeddingPredictor(
        embedding_dim=embedding_dim,
        num_consequence_classes=num_consequence_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_matrix_features=use_matrix_features,
        use_conv_features=use_conv_features
    ).to(device)

    # Loss functions and optimizer
    pathogenicity_criterion = nn.CrossEntropyLoss()
    consequence_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30  # Increased patience

    print(f"Starting training for {epochs} epochs with patience={patience}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(test_loader.dataset)}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        batch_count = 0

        for wt_emb_batch, var_emb_batch, path_labels, cons_labels in tqdm(train_loader):
            wt_emb_batch = wt_emb_batch.to(device)
            var_emb_batch = var_emb_batch.to(device)
            path_labels = path_labels.to(device)
            cons_labels = cons_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            path_logits, cons_logits = model(wt_emb_batch, var_emb_batch)

            # Compute losses
            path_loss = pathogenicity_criterion(path_logits, path_labels)
            cons_loss = consequence_criterion(cons_logits, cons_labels)

            # Combined loss (weight pathogenicity higher)
            total_loss = 2.0 * path_loss + 1.0 * cons_loss

            # Check for NaN losses
            if torch.isnan(total_loss):
                print(f"Warning: NaN loss detected at epoch {epoch}, batch {batch_count}")
                continue

            total_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_train_loss += total_loss.item()
            batch_count += 1

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for wt_emb_batch, var_emb_batch, path_labels, cons_labels in test_loader:
                wt_emb_batch = wt_emb_batch.to(device)
                var_emb_batch = var_emb_batch.to(device)
                path_labels = path_labels.to(device)
                cons_labels = cons_labels.to(device)

                path_logits, cons_logits = model(wt_emb_batch, var_emb_batch)

                path_loss = pathogenicity_criterion(path_logits, path_labels)
                cons_loss = consequence_criterion(cons_logits, cons_labels)
                total_loss = 2.0 * path_loss + 1.0 * cons_loss

                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # Early stopping with minimum epochs requirement
        min_epochs = 20  # Minimum epochs before early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Only apply early stopping after minimum epochs
        if epoch >= min_epochs and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience exceeded)")
            break

        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Patience = {patience_counter}")

    # Final evaluation
    model.eval()
    all_path_preds, all_cons_preds = [], []
    all_path_labels, all_cons_labels = [], []
    all_path_probs = []

    with torch.no_grad():
        for wt_emb_batch, var_emb_batch, path_labels, cons_labels in test_loader:
            wt_emb_batch = wt_emb_batch.to(device)
            var_emb_batch = var_emb_batch.to(device)

            path_logits, cons_logits = model(wt_emb_batch, var_emb_batch)

            path_probs = torch.softmax(path_logits, dim=1)
            path_preds = torch.argmax(path_logits, dim=1)
            cons_preds = torch.argmax(cons_logits, dim=1)

            all_path_preds.extend(path_preds.cpu().numpy())
            all_cons_preds.extend(cons_preds.cpu().numpy())
            all_path_labels.extend(path_labels.cpu().numpy())
            all_cons_labels.extend(cons_labels.cpu().numpy())
            all_path_probs.extend(path_probs[:, 1].cpu().numpy())  # Probability of pathogenic

    # Compute metrics
    path_accuracy = accuracy_score(all_path_labels, all_path_preds)
    cons_accuracy = accuracy_score(all_cons_labels, all_cons_preds)
    path_auc = roc_auc_score(all_path_labels, all_path_probs)

    path_precision, path_recall, path_f1, _ = precision_recall_fscore_support(
        all_path_labels, all_path_preds, average='binary'
    )

    results = {
        'model_name': model_name,
        'trained_model': model,
        'train_losses': train_losses,
        'embedding_dim': embedding_dim,
        'num_variants_trained': len(train_idx),
        'num_variants_tested': len(test_idx),
        'pathogenicity_accuracy': path_accuracy,
        'pathogenicity_auc': path_auc,
        'pathogenicity_precision': path_precision,
        'pathogenicity_recall': path_recall,
        'pathogenicity_f1': path_f1,
        'consequence_accuracy': cons_accuracy,
        'num_consequence_classes': num_consequence_classes,
        'sequence_length': MODEL_SEQUENCE_LENGTHS.get(model_name, 3000)
    }

    return results

def fine_tune_model(model, model_name: str, data_path: str = None,
                   max_variants: Optional[int] = None, epochs: int = 100,
                   sequence_length: Optional[int] = None, jitter: int = 0,
                   balance_consequences: bool = False, target_samples_per_consequence: Optional[int] = None,
                   min_consequence_samples: int = 10, use_matrix_features: bool = False,
                   use_conv_features: bool = False, hidden_dim: int = 256, dropout: float = 0.3,
                   selected_consequences: Optional[List[str]] = None, simple_model: bool = True) -> Dict:
    """
    Fine-tune a single model (already loaded) on pathogenicity prediction.

    Parameters
    ----------
    model : BaseWrapper
        Already loaded model wrapper
    model_name : str
        Name/identifier for the model (determines sequence length if not provided)
    data_path : str, optional
        Path to ClinVar data (uses default if None)
    max_variants : int, optional
        Maximum number of variants to use
    epochs : int
        Number of training epochs
    sequence_length : int, optional
        Length of sequence context around variant (overrides model default)
    jitter : int
        Maximum random offset from center position (default 0 = variant at center)
    balance_consequences : bool
        Whether to balance the dataset by consequence types (default False)
    target_samples_per_consequence : int, optional
        Target number of samples per consequence (uses median if None)
    min_consequence_samples : int
        Minimum samples required to include a consequence type (default 10)
    use_matrix_features : bool
        Whether to use matrix features from WT and variant embedding outer product

    Returns
    -------
    dict
        Training results and metrics for the model
    """
    # Use default path if none provided
    if data_path is None:
        data_path = DEFAULT_CLINVAR_PATH

    # Load and prepare data
    print(f"Loading and preprocessing ClinVar data for {model_name}...")
    dataset = ClinVarDataset(
        data_path,
        max_samples=max_variants,
        balance_consequences=balance_consequences,
        target_samples_per_consequence=target_samples_per_consequence,
        min_consequence_samples=min_consequence_samples,
        selected_consequences=selected_consequences
    )
    df = dataset.load_and_preprocess()
    dataset.prepare_labels()

    # Prepare label arrays
    labels_data = {
        'pathogenicity_labels': df['pathogenicity_label'].values,
        'consequence_labels': df['consequence_label'].values,
        'pathogenicity_encoder': dataset.pathogenicity_encoder,
        'consequence_encoder': dataset.consequence_encoder
    }

    print(f"{'='*60}")
    print(f"Fine-tuning prediction head for model: {model_name}")
    print(f"{'='*60}")

    # Use custom sequence length or model default
    seq_length = sequence_length if sequence_length is not None else MODEL_SEQUENCE_LENGTHS.get(model_name, 3000)

    # Train with real-time embeddings
    print(f"Training with real-time embeddings using {seq_length:,} nucleotide context with jitter={jitter}...")
    model_results = train_prediction_head_realtime(
        model, df, model_name, seq_length, jitter=jitter, epochs=epochs,
        use_matrix_features=use_matrix_features, use_conv_features=use_conv_features,
        hidden_dim=hidden_dim, dropout=dropout, simple_model=simple_model
    )

    # Print summary
    print(f"\nResults for {model_name}:")
    print(f"  Sequence length: {model_results['sequence_length']:,} nucleotides")
    print(f"  Embedding dimension: {model_results['embedding_dim']}")
    print(f"  Variants processed: {model_results['num_variants_tested']}")
    print(f"  Pathogenicity Accuracy: {model_results['pathogenicity_accuracy']:.3f}")
    print(f"  Pathogenicity AUC: {model_results['pathogenicity_auc']:.3f}")
    print(f"  Pathogenicity F1: {model_results['pathogenicity_f1']:.3f}")
    print(f"  Consequence Accuracy: {model_results['consequence_accuracy']:.3f}")

    return model_results

def benchmark_model_fine_tuning(model_names: List[str], data_path: str = None,
                               max_variants: Optional[int] = None, epochs: int = 100,
                               balance_consequences: bool = False, target_samples_per_consequence: Optional[int] = None,
                               min_consequence_samples: int = 10, use_matrix_features: bool = False,
                               use_conv_features: bool = True, hidden_dim: int = 512, dropout: float = 0.3,
                               selected_consequences: Optional[List[str]] = None) -> Dict:
    """
    Benchmark fine-tuning of prediction heads for multiple models.

    Parameters
    ----------
    model_names : list of str
        Names of models to benchmark
    data_path : str
        Path to ClinVar data
    max_variants : int, optional
        Maximum number of variants to use
    epochs : int
        Number of training epochs
    balance_consequences : bool
        Whether to balance the dataset by consequence types (default False)
    target_samples_per_consequence : int, optional
        Target number of samples per consequence (uses median if None)
    min_consequence_samples : int
        Minimum samples required to include a consequence type (default 10)
    use_matrix_features : bool
        Whether to use matrix features from WT and variant embedding outer product

    Returns
    -------
    dict
        Results for all models
    """
    # Use default path if none provided
    if data_path is None:
        data_path = DEFAULT_CLINVAR_PATH

    # Load and prepare data once
    print("Loading and preprocessing ClinVar data...")
    dataset = ClinVarDataset(
        data_path,
        max_samples=max_variants,
        balance_consequences=balance_consequences,
        target_samples_per_consequence=target_samples_per_consequence,
        min_consequence_samples=min_consequence_samples,
        selected_consequences=selected_consequences
    )
    df = dataset.load_and_preprocess()
    dataset.prepare_labels()

    # Prepare label arrays
    labels_data = {
        'pathogenicity_labels': df['pathogenicity_label'].values,
        'consequence_labels': df['consequence_label'].values,
        'pathogenicity_encoder': dataset.pathogenicity_encoder,
        'consequence_encoder': dataset.consequence_encoder
    }

    results = {}

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Fine-tuning prediction head for model: {model_name}")
        print(f"{'='*60}")

        try:
            # Load model
            print(f"Loading model: {model_name}")
            model = get_model(model_name)

            # Extract embeddings
            print(f"Extracting embeddings using {MODEL_SEQUENCE_LENGTHS.get(model_name, 3000):,} nucleotide context...")
            embeddings_data = extract_sequences_and_embeddings(model, model_name, df, max_variants)

            if len(embeddings_data['wt_embeddings']) == 0:
                print(f"No valid embeddings extracted for {model_name}")
                continue

            # Train prediction head
            print(f"Training prediction head...")
            model_results = train_prediction_head(
                embeddings_data, labels_data, model_name, epochs=epochs,
                use_matrix_features=use_matrix_features, use_conv_features=use_conv_features,
                hidden_dim=hidden_dim, dropout=dropout
            )

            results[model_name] = model_results

            # Print summary
            print(f"\nResults for {model_name}:")
            print(f"  Sequence length: {model_results['sequence_length']:,} nucleotides")
            print(f"  Embedding dimension: {model_results['embedding_dim']}")
            print(f"  Variants processed: {model_results['num_variants_tested']}")
            print(f"  Pathogenicity Accuracy: {model_results['pathogenicity_accuracy']:.3f}")
            print(f"  Pathogenicity AUC: {model_results['pathogenicity_auc']:.3f}")
            print(f"  Pathogenicity F1: {model_results['pathogenicity_f1']:.3f}")
            print(f"  Consequence Accuracy: {model_results['consequence_accuracy']:.3f}")

        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results

def validate_dataset_without_models(data_path: str = None, model_name: str = 'nt',
                                   max_variants: int = 100, sample_size: Optional[int] = None,
                                   sequence_length: Optional[int] = None, jitter: int = 0) -> Dict:
    """
    Validate the dataset without loading models - check sequences, labels, and lengths.

    Parameters
    ----------
    data_path : str
        Path to ClinVar data
    model_name : str
        Model name to determine sequence length context (if not provided)
    max_variants : int
        Maximum variants to validate sequences for
    sample_size : int, optional
        Maximum samples to load from dataset
    sequence_length : int, optional
        Length of sequence context around variant (overrides model default)
    jitter : int
        Maximum random offset from center position (default 0 = variant at center)

    Returns
    -------
    dict
        Validation results
    """
    if SeqMat is None:
        raise ImportError("SeqMat is required for sequence validation")

    # Use default path if none provided
    if data_path is None:
        data_path = DEFAULT_CLINVAR_PATH

    print("🔍 VALIDATING DATASET WITHOUT MODELS")
    print("="*50)
    print(f"Using data file: {data_path}")

    # Load and process data
    dataset = ClinVarDataset(data_path, max_samples=sample_size)
    df = dataset.load_and_preprocess()
    dataset.prepare_labels()

    if sequence_length is None:
        sequence_length = MODEL_SEQUENCE_LENGTHS.get(model_name, 3000)

    print(f"\nUsing {model_name} sequence context: {sequence_length:,} nucleotides")
    if jitter > 0:
        print(f"Using jitter: ±{jitter} nucleotides from center")

    # Sample for sequence validation
    validation_df = df.sample(n=min(max_variants, len(df)), random_state=42)

    results = {
        'total_variants': len(df),
        'validation_sample_size': len(validation_df),
        'sequence_length': sequence_length,
        'successful_sequences': 0,
        'failed_sequences': 0,
        'examples': [],
        'errors': []
    }

    print(f"\nValidating {len(validation_df)} sample variants...")

    for idx, row in tqdm(validation_df.iterrows(), total=len(validation_df), desc="Validating"):
        try:
            chrom = str(row['chrom'])
            pos = int(row['pos'])
            ref = row['ref']
            alt = row['alt']

            # Add 'chr' prefix if not present (SeqMat expects chr1, chr2, etc.)
            if not chrom.startswith('chr'):
                chrom = f'chr{chrom}'

            # Calculate sequence window with jitter
            half_length = sequence_length // 2

            # Apply random jitter to offset variant from center
            if jitter > 0:
                offset = np.random.randint(-jitter, jitter + 1)
            else:
                offset = 0

            start_pos = max(1, pos - half_length + offset)
            end_pos = pos + half_length + offset

            # Extract sequences
            wt_seqmat = SeqMat.from_fasta('hg38', chrom, start_pos, end_pos)
            var_seqmat = wt_seqmat.clone()
            mutations = [(pos, ref, alt)]
            var_seqmat.apply_mutations(mutations, permissive_ref=True)

            wt_seq = wt_seqmat.seq
            var_seq = var_seqmat.seq

            # Basic validation
            if len(wt_seq) != len(var_seq):
                results['errors'].append(f"{chrom}:{pos} - Length mismatch")
                continue

            if len(wt_seq) < sequence_length * 0.8:
                results['errors'].append(f"{chrom}:{pos} - Sequence too short: {len(wt_seq)}")
                continue

            # Find mutation position and validate
            mut_pos = pos - start_pos
            if 0 <= mut_pos < len(wt_seq):
                wt_base = wt_seq[mut_pos].upper()
                var_base = var_seq[mut_pos].upper()

                if wt_base != ref.upper():
                    results['errors'].append(f"{chrom}:{pos} - Ref mismatch: expected {ref}, got {wt_base}")
                    continue

                if var_base != alt.upper():
                    results['errors'].append(f"{chrom}:{pos} - Alt mismatch: expected {alt}, got {var_base}")
                    continue

            # Success!
            results['successful_sequences'] += 1

            # Store example
            if len(results['examples']) < 5:
                context_start = max(0, mut_pos - 15)
                context_end = min(len(wt_seq), mut_pos + 16)

                results['examples'].append({
                    'variant': f"{chrom}:{pos} {ref}>{alt}",
                    'pathogenicity': row['clin_sig_clinvar'],
                    'consequence': row['consequence'],
                    'wt_context': wt_seq[context_start:context_end],
                    'var_context': var_seq[context_start:context_end],
                    'sequence_length': len(wt_seq),
                    'mutation_position': mut_pos
                })

        except Exception as e:
            results['failed_sequences'] += 1
            results['errors'].append(f"{chrom}:{pos} - Exception: {str(e)}")

    # Print results
    success_rate = results['successful_sequences'] / len(validation_df) * 100
    print(f"\n✅ VALIDATION RESULTS:")
    print(f"Dataset size: {results['total_variants']:,} variants")
    print(f"Sequence validation: {results['successful_sequences']}/{len(validation_df)} ({success_rate:.1f}%)")
    print(f"Pathogenicity distribution: {df['clin_sig_clinvar'].value_counts().to_dict()}")
    print(f"Top consequences: {dict(df['consequence'].value_counts().head(5))}")

    # Show examples
    print(f"\n📝 EXAMPLE SEQUENCES:")
    for i, example in enumerate(results['examples']):
        print(f"\n{i+1}. {example['variant']} - {example['pathogenicity']}")
        print(f"   Consequence: {example['consequence']}")
        print(f"   Length: {example['sequence_length']} nt, Mutation at position: {example['mutation_position']}")
        print(f"   WT:  {example['wt_context']}")
        print(f"   VAR: {example['var_context']}")

    # Show errors
    if results['errors']:
        print(f"\n⚠️  ERRORS ({len(results['errors'])} total):")
        for error in results['errors'][:3]:
            print(f"   {error}")
        if len(results['errors']) > 3:
            print(f"   ... and {len(results['errors']) - 3} more")

    return results

def save_benchmark_results(results: Dict, output_dir: str):
    """Save benchmark results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare summary data
    summary_data = []
    for model_name, model_results in results.items():
        summary_data.append({
            'Model': model_name,
            'Sequence_Length': model_results['sequence_length'],
            'Embedding_Dim': model_results['embedding_dim'],
            'Variants_Tested': model_results['num_variants_tested'],
            'Pathogenicity_Accuracy': model_results['pathogenicity_accuracy'],
            'Pathogenicity_AUC': model_results['pathogenicity_auc'],
            'Pathogenicity_F1': model_results['pathogenicity_f1'],
            'Consequence_Accuracy': model_results['consequence_accuracy'],
            'Num_Consequence_Classes': model_results['num_consequence_classes']
        })

    # Save summary table
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Pathogenicity_AUC', ascending=False)
    summary_df.to_csv(output_path / 'model_comparison.csv', index=False)

    # Save detailed results (excluding trained models)
    detailed_results = {}
    for model_name, model_results in results.items():
        detailed_results[model_name] = {k: v for k, v in model_results.items()
                                      if k != 'trained_model'}

    with open(output_path / 'detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print("\nModel Comparison:")
    print(summary_df.round(3).to_string(index=False))

def main():
    """Main function for running the enhanced pathogenicity benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Pathogenicity Benchmark with Fine-tuning')
    parser.add_argument('--models', nargs='+', default=['nt', 'conformer'],
                       help='Models to benchmark')
    parser.add_argument('--data', default='../assets/benchmarks/clinvar_vep_subset.csv',
                       help='Path to ClinVar data')
    parser.add_argument('--output', default='./pathogenicity_finetuning_results',
                       help='Output directory')
    parser.add_argument('--max_variants', type=int, help='Maximum variants to use')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--balance_consequences', action='store_true',
                       help='Balance dataset by consequence types')
    parser.add_argument('--target_samples_per_consequence', type=int,
                       help='Target number of samples per consequence (default: use median)')
    parser.add_argument('--min_consequence_samples', type=int, default=10,
                       help='Minimum samples required to include a consequence type (default: 10)')
    parser.add_argument('--use_matrix_features', action='store_true',
                       help='Use matrix features from WT and variant embedding outer product')
    parser.add_argument('--use_conv_features', action='store_true', default=True,
                       help='Use convolutional features for embedding interactions (default: True)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension size for neural network (default: 512)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate for regularization (default: 0.3)')
    parser.add_argument('--no_conv_features', action='store_true',
                       help='Disable convolutional features (overrides --use_conv_features)')
    parser.add_argument('--selected_consequences', nargs='+',
                       help='Specific consequence types to include (e.g., missense_variant stop_gained)')
    parser.add_argument('--list_consequences', action='store_true',
                       help='List all available consequence types and exit')

    args = parser.parse_args()

    # List consequences and exit if requested
    if args.list_consequences:
        list_available_consequences(args.data)
        return

    # Handle convolutional features flag
    use_conv_features = args.use_conv_features and not args.no_conv_features

    # Run benchmark
    results = benchmark_model_fine_tuning(
        model_names=args.models,
        data_path=args.data,
        max_variants=args.max_variants,
        epochs=args.epochs,
        balance_consequences=args.balance_consequences,
        target_samples_per_consequence=args.target_samples_per_consequence,
        min_consequence_samples=args.min_consequence_samples,
        use_matrix_features=args.use_matrix_features,
        use_conv_features=use_conv_features,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        selected_consequences=args.selected_consequences
    )

    # Save results
    save_benchmark_results(results, args.output)

if __name__ == "__main__":
    main()