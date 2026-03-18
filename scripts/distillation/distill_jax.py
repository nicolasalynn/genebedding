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

    BASES = list("ACGT")

    # Mutation regime: for each window, generate WT + variants with 1..max_muts mutations
    max_muts = args.max_mutations
    variants_per_window = args.variants_per_window
    logger.info("Mutation regime: up to %d mutations, %d variants per window",
                max_muts, variants_per_window)

    i = n_existing
    window_idx = i // (1 + variants_per_window)  # approximate resume
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

            # ---- Wild-type embedding ----
            emb_wt = teacher.embed(seq, pool="mean", return_numpy=True)
            tokens_wt = np.array([token_map.get(c, 4) for c in seq], dtype=np.int8)

            np.savez(cache_dir / f"s_{i:07d}.npz",
                     tokens=tokens_wt,
                     embedding=emb_wt.astype(np.float32),
                     seq_len=np.int32(len(seq)),
                     n_mutations=np.int32(0),
                     window_id=np.int32(window_idx))
            i += 1

            # ---- Mutant embeddings ----
            # For each variant: pick random number of mutations (1..max_muts),
            # at random positions, with random alt alleles.
            # This teaches the student to model the full perturbation landscape:
            # single mutations, double mutations, and higher-order combinations.
            seq_list = list(seq)
            valid_positions = [p for p in range(len(seq)) if seq[p] in BASES]

            for v in range(variants_per_window):
                if i >= args.n_sequences:
                    break

                # Random number of mutations: geometric distribution favoring fewer
                # P(k) ∝ 0.5^k, so singles are most common, higher-order are rarer
                n_muts = min(
                    rng.geometric(p=0.4),  # mean ~2.5 mutations
                    max_muts,
                    len(valid_positions),
                )

                # Pick random positions
                mut_positions = rng.choice(valid_positions, size=n_muts, replace=False)
                mut_positions.sort()

                # Apply mutations
                mut_seq = seq_list.copy()
                mut_info = []
                for pos in mut_positions:
                    ref = mut_seq[pos]
                    alt = rng.choice([b for b in BASES if b != ref])
                    mut_seq[pos] = alt
                    mut_info.append((int(pos), ref, alt))

                mut_str = "".join(mut_seq)
                emb_mut = teacher.embed(mut_str, pool="mean", return_numpy=True)
                tokens_mut = np.array([token_map.get(c, 4) for c in mut_str], dtype=np.int8)

                # Store mutation positions as array for potential use in training
                mut_positions_arr = np.array([m[0] for m in mut_info], dtype=np.int32)

                np.savez(cache_dir / f"s_{i:07d}.npz",
                         tokens=tokens_mut,
                         embedding=emb_mut.astype(np.float32),
                         seq_len=np.int32(len(mut_str)),
                         n_mutations=np.int32(n_muts),
                         mutation_positions=mut_positions_arr,
                         window_id=np.int32(window_idx))
                i += 1

            window_idx += 1
            failures = 0
            if i % 1000 == 0:
                logger.info("  %d/%d cached (%d windows)", i, args.n_sequences, window_idx)

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
    steps_per_epoch = n_samples // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    schedule = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=total_steps,
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=0.01)
    opt_state = optimizer.init(params)

    # -------------------------------------------------------------------
    # HuggingFace Hub setup
    # -------------------------------------------------------------------
    hf_repo = getattr(args, "hf_repo", None)
    hf_api = None
    if hf_repo:
        try:
            from huggingface_hub import HfApi, create_repo
            hf_api = HfApi()
            create_repo(hf_repo, exist_ok=True, repo_type="model")
            logger.info("HuggingFace repo: %s", hf_repo)

            # Push initial model card
            model_card = f"""---
tags:
  - genomics
  - distillation
  - jax
  - dna-language-model
license: apache-2.0
---

# Distilled Genomic Foundation Model

Distilled from **{teacher_name}** using genebeddings distillation framework.

## Architecture
- **Type**: U-Net (conv downsample + transformer bottleneck + conv upsample)
- **Parameters**: {n_params:,} ({n_params/1e6:.1f}M)
- **Hidden dim**: {hidden_dim}
- **Downsamples**: {n_down} (sequence compressed {2**n_down}x)
- **Transformer layers**: {n_transformer}
- **Input**: DNA sequence up to {seq_length:,} bp
- **Output**: {embed_dim}-dim embedding (mean-pooled)

## Training
- **Teacher**: {teacher_name}
- **Framework**: JAX/Haiku
- **Data**: {n_samples:,} sequences (WT + mutant variants from hg38)
- **Loss**: MSE on teacher embeddings
- **Mutation-aware**: trained on WT, single, double, and multi-mutant sequences

## Usage
```python
# Load and use (requires JAX + Haiku)
import jax, haiku as hk, numpy as np
params = np.load("best_params.npz")
config = np.load("config.npz")
```
"""
            card_path = output_dir / "README.md"
            card_path.write_text(model_card)
            hf_api.upload_file(
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                repo_id=hf_repo,
                commit_message="Initial model card",
            )
        except Exception as e:
            logger.warning("HuggingFace Hub setup failed: %s. Training without HF.", e)
            hf_api = None

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
    global_step = 0

    for epoch in range(args.epochs):
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
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        lr_now = float(schedule(global_step))
        logger.info("Epoch %d/%d: loss=%.6f, lr=%.2e, step=%d",
                     epoch + 1, args.epochs, avg_loss, lr_now, global_step)

        # Save best
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        # Always save latest; save best separately
        for tag in (["latest"] + (["best"] if is_best else [])):
            flat_params = {
                f"{'/'.join(str(k) for k in path)}": np.array(leaf)
                for path, leaf in jax.tree_util.tree_leaves_with_path(params)
            }
            np.savez(output_dir / f"{tag}_params.npz", **flat_params)

        # Save config (always update with latest stats)
        np.savez(output_dir / "config.npz",
                 teacher=teacher_name,
                 embed_dim=embed_dim,
                 seq_length=seq_length,
                 hidden_dim=hidden_dim,
                 n_downsamples=n_down,
                 n_transformer_layers=n_transformer,
                 n_params=n_params,
                 best_loss=best_loss,
                 last_epoch=epoch + 1,
                 last_loss=avg_loss,
                 total_steps=global_step)

        # Push to HuggingFace
        if hf_api and hf_repo:
            try:
                # Upload checkpoint
                files_to_upload = [
                    (str(output_dir / "latest_params.npz"), "latest_params.npz"),
                    (str(output_dir / "config.npz"), "config.npz"),
                ]
                if is_best:
                    files_to_upload.append(
                        (str(output_dir / "best_params.npz"), "best_params.npz"))

                for local, remote in files_to_upload:
                    hf_api.upload_file(
                        path_or_fileobj=local,
                        path_in_repo=remote,
                        repo_id=hf_repo,
                        commit_message=f"Epoch {epoch+1}: loss={avg_loss:.6f}"
                                       f"{' (best)' if is_best else ''}",
                    )

                # Push training log as JSON
                import json
                log_path = output_dir / "training_log.json"
                log_entries = []
                if log_path.exists():
                    log_entries = json.loads(log_path.read_text())
                log_entries.append({
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "best_loss": best_loss,
                    "lr": lr_now,
                    "step": global_step,
                })
                log_path.write_text(json.dumps(log_entries, indent=2))
                hf_api.upload_file(
                    path_or_fileobj=str(log_path),
                    path_in_repo="training_log.json",
                    repo_id=hf_repo,
                    commit_message=f"Training log epoch {epoch+1}",
                )

                logger.info("  Pushed to HuggingFace: %s", hf_repo)
            except Exception as e:
                logger.warning("  HF push failed: %s", e)

    logger.info("Training complete. Best loss: %.6f", best_loss)
    logger.info("Saved to %s", output_dir)
    if hf_repo:
        logger.info("HuggingFace: https://huggingface.co/%s", hf_repo)


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
    gen.add_argument("--n-sequences", type=int, default=500_000,
                     help="Total sequences (WT + all variants)")
    gen.add_argument("--seq-length", type=int, default=15_000)
    gen.add_argument("--max-mutations", type=int, default=10,
                     help="Max mutations per variant (actual count drawn from geometric dist)")
    gen.add_argument("--variants-per-window", type=int, default=6,
                     help="Number of mutant variants per WT window (so 1 WT + N variants per window)")
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
    tr.add_argument("--hf-repo", default=None,
                    help="HuggingFace repo ID to push checkpoints (e.g. nicolasalynn/distilled-borzoi)")

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
