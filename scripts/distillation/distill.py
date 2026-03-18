"""
Distill any genebeddings teacher model into a student model.

Usage:
    python -m scripts.distillation.distill \
        --teacher ntv3_100m_post \
        --student-config scripts/distillation/configs/unet_100m.yaml \
        --data-fasta /path/to/hg38.fa \
        --output-dir /path/to/distilled_model \
        --n-sequences 500000 \
        --seq-length 8192 \
        --batch-size 16 \
        --epochs 20

Steps:
    1. Load teacher model via genebeddings wrapper (any model in our system)
    2. Generate random genomic sequences from reference FASTA
    3. Run teacher to get embeddings (cached to disk)
    4. Train student to match teacher embeddings via MSE loss
    5. Save student weights in HuggingFace-compatible format

The student can be any architecture — the only requirement is that it
produces a fixed-dim embedding from a DNA sequence.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset: teacher embeddings cached to disk
# ---------------------------------------------------------------------------
class TeacherEmbeddingDataset(Dataset):
    """Dataset of (sequence_tokens, teacher_embedding) pairs."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No cached embeddings in {cache_dir}")
        # Load first to get dimensions
        sample = np.load(self.files[0])
        self.embed_dim = sample["embedding"].shape[0]
        logger.info("Dataset: %d samples, embed_dim=%d", len(self.files), self.embed_dim)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        tokens = torch.tensor(data["tokens"], dtype=torch.long)
        embedding = torch.tensor(data["embedding"], dtype=torch.float32)
        seq_len = int(data["seq_len"])
        return tokens, embedding, seq_len


# ---------------------------------------------------------------------------
# Step 1: Generate teacher embeddings
# ---------------------------------------------------------------------------
def generate_teacher_embeddings(
    teacher_key: str,
    fasta_path: str,
    cache_dir: Path,
    n_sequences: int = 100_000,
    seq_length: int = 8192,
    pool: str = "mean",
    seed: int = 42,
):
    """Run teacher model on random genomic windows, cache embeddings."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if already cached
    existing = list(cache_dir.glob("*.npz"))
    if len(existing) >= n_sequences:
        logger.info("Teacher embeddings already cached: %d files", len(existing))
        return

    # Load teacher via our wrapper system
    logger.info("Loading teacher model: %s", teacher_key)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from notebooks.processing.process_epistasis import FULL_MODEL_CONFIG, _build_model
    if teacher_key not in FULL_MODEL_CONFIG:
        raise ValueError(f"Unknown teacher: {teacher_key}. Available: {list(FULL_MODEL_CONFIG.keys())}")

    _, init_spec = FULL_MODEL_CONFIG[teacher_key]
    teacher = _build_model(teacher_key, init_spec)
    if teacher is None:
        raise RuntimeError(f"Failed to load teacher model: {teacher_key}")

    # Generate random genomic windows from FASTA
    logger.info("Sampling %d sequences of length %d from %s", n_sequences, seq_length, fasta_path)
    from seqmat import SeqMat

    rng = np.random.RandomState(seed)

    # Get chromosome lengths
    import pysam
    fasta = pysam.FastaFile(fasta_path)
    chroms = [(name, length) for name, length in
              zip(fasta.references, fasta.lengths)
              if name.startswith("chr") and name[3:].isdigit() and length > seq_length * 2]
    chrom_weights = np.array([l for _, l in chroms], dtype=float)
    chrom_weights /= chrom_weights.sum()
    fasta.close()

    n_cached = len(existing)
    logger.info("Starting from sample %d", n_cached)

    for i in range(n_cached, n_sequences):
        # Sample random chromosome and position
        chrom_idx = rng.choice(len(chroms), p=chrom_weights)
        chrom_name, chrom_len = chroms[chrom_idx]
        start = rng.randint(0, chrom_len - seq_length)

        try:
            sm = SeqMat.from_fasta("hg38", chrom_name, start, start + seq_length - 1)
            seq = sm.seq.upper()

            # Skip sequences with too many Ns
            n_count = seq.count("N")
            if n_count > seq_length * 0.1:
                continue

            # Get teacher embedding
            emb = teacher.embed(seq, pool=pool, return_numpy=True)

            # Tokenize (simple char-to-int for storage)
            token_map = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
            tokens = np.array([token_map.get(c, 4) for c in seq], dtype=np.int8)

            # Save
            np.savez(
                cache_dir / f"sample_{i:07d}.npz",
                tokens=tokens,
                embedding=emb.astype(np.float32),
                seq_len=len(seq),
                chrom=chrom_name,
                start=start,
            )

            if (i + 1) % 1000 == 0:
                logger.info("  Cached %d/%d embeddings", i + 1, n_sequences)

        except Exception as e:
            logger.warning("Failed sample %d: %s", i, e)
            continue

    logger.info("Teacher embedding generation complete: %d samples", n_sequences)


# ---------------------------------------------------------------------------
# Step 2: Simple student models
# ---------------------------------------------------------------------------
class ConvStudentModel(nn.Module):
    """Simple conv-based student that maps DNA tokens to embeddings."""

    def __init__(self, vocab_size=5, embed_dim=512, hidden_dim=256, n_layers=6,
                 kernel_size=9):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        layers = []
        for i in range(n_layers):
            in_ch = hidden_dim
            out_ch = hidden_dim
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.BatchNorm1d(out_ch),
            ])
        self.conv_tower = nn.Sequential(*layers)
        self.projection = nn.Linear(hidden_dim, embed_dim)

    def forward(self, tokens, seq_len=None):
        """tokens: (B, L) -> embedding: (B, embed_dim)"""
        x = self.token_embed(tokens)  # (B, L, H)
        x = x.transpose(1, 2)  # (B, H, L)
        x = self.conv_tower(x)  # (B, H, L)
        x = x.transpose(1, 2)  # (B, L, H)

        # Mean pool (respecting actual seq_len)
        if seq_len is not None:
            mask = torch.arange(x.size(1), device=x.device)[None, :] < seq_len[:, None]
            x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
        else:
            x = x.mean(dim=1)

        return self.projection(x)  # (B, embed_dim)


# ---------------------------------------------------------------------------
# Step 3: Training loop
# ---------------------------------------------------------------------------
def train_student(
    student: nn.Module,
    dataset: TeacherEmbeddingDataset,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cuda",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    student = student.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        n_batches = 0

        for tokens, target_emb, seq_lens in loader:
            tokens = tokens.to(device)
            target_emb = target_emb.to(device)
            seq_lens = seq_lens.to(device)

            pred_emb = student(tokens, seq_len=seq_lens)
            loss = loss_fn(pred_emb, target_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        logger.info("Epoch %d/%d: loss=%.6f, lr=%.2e",
                     epoch + 1, epochs, avg_loss, scheduler.get_last_lr()[0])

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "loss": best_loss,
                "embed_dim": dataset.embed_dim,
            }, output_dir / "best_student.pt")

    # Save final
    torch.save({
        "epoch": epochs,
        "model_state_dict": student.state_dict(),
        "loss": avg_loss,
        "embed_dim": dataset.embed_dim,
    }, output_dir / "final_student.pt")

    logger.info("Training complete. Best loss: %.6f", best_loss)
    return best_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Distill a genebeddings teacher model")
    parser.add_argument("--teacher", required=True,
                        help="Teacher model key (e.g. ntv3_100m_post, borzoi, nt500_multi)")
    parser.add_argument("--data-fasta", required=True,
                        help="Path to reference genome FASTA (e.g. hg38.fa)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for cached embeddings and student model")
    parser.add_argument("--n-sequences", type=int, default=500_000,
                        help="Number of sequences to sample for distillation")
    parser.add_argument("--seq-length", type=int, default=8192,
                        help="Length of each sequence")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--student-hidden", type=int, default=256)
    parser.add_argument("--student-layers", type=int, default=6)
    parser.add_argument("--pool", default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate teacher embeddings, don't train student")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "teacher_cache"

    # Step 1: Generate teacher embeddings
    generate_teacher_embeddings(
        teacher_key=args.teacher,
        fasta_path=args.data_fasta,
        cache_dir=cache_dir,
        n_sequences=args.n_sequences,
        seq_length=args.seq_length,
        pool=args.pool,
        seed=args.seed,
    )

    if args.generate_only:
        logger.info("Teacher embeddings generated. Exiting (--generate-only).")
        return

    # Step 2: Load dataset
    dataset = TeacherEmbeddingDataset(cache_dir)

    # Step 3: Create student
    student = ConvStudentModel(
        vocab_size=5,
        embed_dim=dataset.embed_dim,
        hidden_dim=args.student_hidden,
        n_layers=args.student_layers,
    )
    n_params = sum(p.numel() for p in student.parameters())
    logger.info("Student model: %d parameters (%.1fM)", n_params, n_params / 1e6)

    # Step 4: Train
    train_student(
        student=student,
        dataset=dataset,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
