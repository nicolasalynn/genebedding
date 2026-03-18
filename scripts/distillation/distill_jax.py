"""
Distill any genebeddings teacher model into a JAX student.

The teacher can be any PyTorch model in our wrapper system. Teacher
inference runs in PyTorch, embeddings are cached to disk, then a JAX
student is trained to match them.

Student architecture: convolutional U-Net in Haiku (same family as
NTv3/AlphaGenome). Configurable depth, width, and whether to include
transformer layers in the bottleneck.

Usage:
    # Step 1: Cache teacher embeddings (PyTorch, GPU)
    python -m scripts.distillation.distill_jax generate \
        --teacher borzoi \
        --fasta /path/to/hg38.fa \
        --cache-dir /path/to/cache \
        --n-sequences 500000 \
        --seq-length 8192

    # Step 2: Train JAX student (JAX, GPU/TPU)
    python -m scripts.distillation.distill_jax train \
        --cache-dir /path/to/cache \
        --output-dir /path/to/student \
        --hidden-dim 256 \
        --n-downsamples 5 \
        --n-transformer-layers 4 \
        --epochs 30 \
        --batch-size 64
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================================
# Step 1: Generate teacher embeddings (PyTorch side)
# =========================================================================
def cmd_generate(args):
    """Cache teacher embeddings to disk using PyTorch wrappers."""
    import torch

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = sorted(cache_dir.glob("*.npz"))
    n_existing = len(existing)
    if n_existing >= args.n_sequences:
        logger.info("Already cached %d embeddings, skipping.", n_existing)
        return

    # Load teacher
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from notebooks.processing.process_epistasis import FULL_MODEL_CONFIG, _build_model

    if args.teacher not in FULL_MODEL_CONFIG:
        raise ValueError(f"Unknown teacher: {args.teacher}. "
                         f"Available: {sorted(FULL_MODEL_CONFIG.keys())}")

    _, init_spec = FULL_MODEL_CONFIG[args.teacher]
    teacher = _build_model(args.teacher, init_spec)
    if teacher is None:
        raise RuntimeError(f"Failed to load teacher: {args.teacher}")

    # Sample genomic windows
    import pysam
    fasta = pysam.FastaFile(args.fasta)
    chroms = [(name, length) for name, length in
              zip(fasta.references, fasta.lengths)
              if name.startswith("chr") and name[3:].isdigit() and length > args.seq_length * 2]
    chrom_weights = np.array([l for _, l in chroms], dtype=float)
    chrom_weights /= chrom_weights.sum()
    fasta.close()

    rng = np.random.RandomState(args.seed)
    token_map = {c: i for i, c in enumerate("ACGTN")}

    # Save metadata on first run
    meta_path = cache_dir / "metadata.npz"
    if not meta_path.exists():
        # Get embed dim from a test sequence
        test_seq = "ACGT" * (args.seq_length // 4)
        test_emb = teacher.embed(test_seq, pool="mean", return_numpy=True)
        np.savez(meta_path,
                 teacher=args.teacher,
                 embed_dim=test_emb.shape[0],
                 seq_length=args.seq_length,
                 pool="mean")
        logger.info("Teacher %s: embed_dim=%d", args.teacher, test_emb.shape[0])

    from seqmat import SeqMat

    i = n_existing
    failures = 0
    while i < args.n_sequences:
        chrom_idx = rng.choice(len(chroms), p=chrom_weights)
        chrom_name, chrom_len = chroms[chrom_idx]
        start = rng.randint(0, chrom_len - args.seq_length)

        try:
            sm = SeqMat.from_fasta("hg38", chrom_name, start, start + args.seq_length - 1)
            seq = sm.seq.upper()
            if seq.count("N") > args.seq_length * 0.1:
                continue

            emb = teacher.embed(seq, pool="mean", return_numpy=True)
            tokens = np.array([token_map.get(c, 4) for c in seq], dtype=np.int8)

            np.savez(cache_dir / f"s_{i:07d}.npz",
                     tokens=tokens,
                     embedding=emb.astype(np.float32),
                     seq_len=np.int32(len(seq)))

            i += 1
            failures = 0
            if i % 1000 == 0:
                logger.info("  %d/%d cached", i, args.n_sequences)

        except Exception as e:
            failures += 1
            if failures > 100:
                logger.error("Too many failures, aborting.")
                break
            continue

    logger.info("Done: %d embeddings cached to %s", i, cache_dir)


# =========================================================================
# Step 2: JAX student model (Haiku)
# =========================================================================
def cmd_train(args):
    """Train JAX student to match cached teacher embeddings."""
    import jax
    import jax.numpy as jnp
    import haiku as hk
    import optax

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta = np.load(cache_dir / "metadata.npz")
    embed_dim = int(meta["embed_dim"])
    seq_length = int(meta["seq_length"])
    teacher_name = str(meta["teacher"])
    logger.info("Teacher: %s, embed_dim=%d, seq_length=%d",
                teacher_name, embed_dim, seq_length)

    # Load all cached embeddings into memory
    files = sorted(cache_dir.glob("s_*.npz"))
    logger.info("Loading %d cached samples...", len(files))

    all_tokens = []
    all_embeddings = []
    for f in files:
        data = np.load(f)
        all_tokens.append(data["tokens"])
        all_embeddings.append(data["embedding"])

    all_tokens = np.stack(all_tokens)        # (N, L)
    all_embeddings = np.stack(all_embeddings)  # (N, D)
    n_samples = len(all_tokens)
    logger.info("Loaded %d samples: tokens %s, embeddings %s",
                n_samples, all_tokens.shape, all_embeddings.shape)

    # -------------------------------------------------------------------
    # Student model definition (Haiku)
    # -------------------------------------------------------------------
    hidden_dim = args.hidden_dim
    n_down = args.n_downsamples
    n_transformer = args.n_transformer_layers

    def student_fn(tokens):
        """tokens: (B, L) int -> embedding: (B, embed_dim)"""
        B, L = tokens.shape

        # Token embedding
        x = hk.Embed(vocab_size=5, embed_dim=hidden_dim)(tokens)  # (B, L, H)

        # Downsampling conv tower with skip connections
        skips = []
        for i in range(n_down):
            x = hk.Conv1D(hidden_dim, kernel_shape=7, padding="SAME",
                          name=f"down_conv_{i}")(x)
            x = jax.nn.gelu(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name=f"down_ln_{i}")(x)
            skips.append(x)
            # Downsample by 2 via strided conv
            x = hk.Conv1D(hidden_dim, kernel_shape=2, stride=2, padding="VALID",
                          name=f"down_stride_{i}")(x)

        # Bottleneck: optional transformer layers
        for i in range(n_transformer):
            # Self-attention
            residual = x
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name=f"attn_ln_{i}")(x)
            x = hk.MultiHeadAttention(
                num_heads=max(1, hidden_dim // 64),
                key_size=64,
                model_size=hidden_dim,
                name=f"mha_{i}",
            )(x, x, x)
            x = x + residual

            # FFN
            residual = x
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name=f"ffn_ln_{i}")(x)
            x = hk.Linear(hidden_dim * 4, name=f"ffn_up_{i}")(x)
            x = jax.nn.gelu(x)
            x = hk.Linear(hidden_dim, name=f"ffn_down_{i}")(x)
            x = x + residual

        # Upsampling conv tower
        for i in range(n_down - 1, -1, -1):
            # Upsample by 2 via repeat
            x = jnp.repeat(x, 2, axis=1)
            # Trim to match skip size
            skip = skips[i]
            x = x[:, :skip.shape[1], :]
            # Add skip connection
            x = x + skip
            x = hk.Conv1D(hidden_dim, kernel_shape=7, padding="SAME",
                          name=f"up_conv_{i}")(x)
            x = jax.nn.gelu(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name=f"up_ln_{i}")(x)

        # Mean pool -> projection
        x = jnp.mean(x, axis=1)  # (B, H)
        x = hk.Linear(embed_dim, name="projection")(x)
        return x

    # -------------------------------------------------------------------
    # Init
    # -------------------------------------------------------------------
    model = hk.without_apply_rng(hk.transform(student_fn))

    rng_key = jax.random.PRNGKey(args.seed)
    dummy_tokens = jnp.zeros((1, seq_length), dtype=jnp.int32)
    params = model.init(rng_key, dummy_tokens)

    n_params = sum(x.size for x in jax.tree.leaves(params))
    logger.info("Student: %d parameters (%.1fM)", n_params, n_params / 1e6)

    # Optimizer
    schedule = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=args.epochs * (n_samples // args.batch_size),
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=0.01)
    opt_state = optimizer.init(params)

    # -------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------
    @jax.jit
    def train_step(params, opt_state, batch_tokens, batch_targets):
        def loss_fn(params):
            pred = model.apply(params, batch_tokens)
            return jnp.mean((pred - batch_targets) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    rng = np.random.RandomState(args.seed)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        # Shuffle
        perm = rng.permutation(n_samples)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples - args.batch_size, args.batch_size):
            idx = perm[start:start + args.batch_size]
            batch_tok = jnp.array(all_tokens[idx], dtype=jnp.int32)
            batch_emb = jnp.array(all_embeddings[idx], dtype=jnp.float32)

            params, opt_state, loss = train_step(params, opt_state, batch_tok, batch_emb)
            total_loss += float(loss)
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info("Epoch %d/%d: loss=%.6f", epoch + 1, args.epochs, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save params
            flat_params = {
                f"{'/'.join(str(k) for k in path)}": np.array(leaf)
                for path, leaf in jax.tree_util.tree_leaves_with_path(params)
            }
            np.savez(output_dir / "best_params.npz", **flat_params)
            # Save config
            np.savez(output_dir / "config.npz",
                     teacher=teacher_name,
                     embed_dim=embed_dim,
                     seq_length=seq_length,
                     hidden_dim=hidden_dim,
                     n_downsamples=n_down,
                     n_transformer_layers=n_transformer,
                     n_params=n_params,
                     best_loss=best_loss)

    logger.info("Training complete. Best loss: %.6f", best_loss)
    logger.info("Saved to %s", output_dir)


# =========================================================================
# CLI
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Distill genebeddings model to JAX student")
    subparsers = parser.add_subparsers(dest="command")

    # Generate
    gen = subparsers.add_parser("generate", help="Cache teacher embeddings")
    gen.add_argument("--teacher", required=True)
    gen.add_argument("--fasta", required=True)
    gen.add_argument("--cache-dir", required=True)
    gen.add_argument("--n-sequences", type=int, default=500_000)
    gen.add_argument("--seq-length", type=int, default=8192)
    gen.add_argument("--seed", type=int, default=42)

    # Train
    tr = subparsers.add_parser("train", help="Train JAX student")
    tr.add_argument("--cache-dir", required=True)
    tr.add_argument("--output-dir", required=True)
    tr.add_argument("--hidden-dim", type=int, default=256)
    tr.add_argument("--n-downsamples", type=int, default=5)
    tr.add_argument("--n-transformer-layers", type=int, default=4)
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
