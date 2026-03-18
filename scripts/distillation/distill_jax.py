"""
Online distillation: stream teacher (PyTorch) → student (JAX).

No caching — teacher and student run simultaneously. Teacher produces
full token-level embeddings (B, L, D), student learns to match them.

Mutation-aware: each batch contains a genomic window + variants with
1-5 mutations. The student must learn how the teacher's token-level
representations change under perturbation.

Usage:
    python -m scripts.distillation.distill_jax \
        --teacher borzoi \
        --fasta /path/to/hg38.fa \
        --output-dir /path/to/student \
        --hf-repo nicolasalynn/distilled-borzoi \
        --seq-length 15000 \
        --hidden-dim 256 \
        --epochs 30
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASES = list("ACGT")


# =========================================================================
# Sequence sampler — generates WT + mutant batches from reference FASTA
# =========================================================================
class GenomicWindowSampler:
    """Sample random genomic windows + mutant variants from a FASTA."""

    def __init__(self, fasta_path, seq_length, max_mutations=5,
                 variants_per_window=6, seed=42):
        import pysam
        self.seq_length = seq_length
        self.max_mutations = max_mutations
        self.variants_per_window = variants_per_window
        self.rng = np.random.RandomState(seed)
        self.token_map = {c: i for i, c in enumerate("ACGTN")}

        fasta = pysam.FastaFile(fasta_path)
        self.chroms = [(name, length) for name, length in
                       zip(fasta.references, fasta.lengths)
                       if name.startswith("chr") and name[3:].isdigit()
                       and length > seq_length * 2]
        weights = np.array([l for _, l in self.chroms], dtype=float)
        self.chrom_weights = weights / weights.sum()
        self.fasta_path = fasta_path
        fasta.close()

    def _tokenize(self, seq):
        return np.array([self.token_map.get(c, 4) for c in seq], dtype=np.int32)

    def sample_batch(self, batch_size):
        """Sample batch_size windows, each with WT + variants.

        Returns:
            tokens: (B * (1 + V), L) int32 — all sequences tokenized
            n_per_window: int — 1 + variants_per_window
            seq_lengths: (B * (1 + V),) int32
        """
        from seqmat import SeqMat

        all_tokens = []
        all_lengths = []
        n_per = 1 + self.variants_per_window

        for _ in range(batch_size):
            # Sample window
            for _attempt in range(20):
                ci = self.rng.choice(len(self.chroms), p=self.chrom_weights)
                chrom_name, chrom_len = self.chroms[ci]
                start = self.rng.randint(0, chrom_len - self.seq_length)
                try:
                    sm = SeqMat.from_fasta("hg38", chrom_name,
                                           start, start + self.seq_length - 1)
                    seq = sm.seq.upper()
                    if seq.count("N") < self.seq_length * 0.1:
                        break
                except Exception:
                    continue
            else:
                # Fallback: random sequence
                seq = "".join(self.rng.choice(BASES, size=self.seq_length))

            valid_pos = [p for p in range(len(seq)) if seq[p] in BASES]

            # WT
            all_tokens.append(self._tokenize(seq))
            all_lengths.append(len(seq))

            # Variants
            seq_list = list(seq)
            for _ in range(self.variants_per_window):
                n_muts = min(
                    self.rng.geometric(p=0.4),
                    self.max_mutations,
                    len(valid_pos),
                )
                positions = self.rng.choice(valid_pos, size=n_muts, replace=False)
                mut_seq = seq_list.copy()
                for pos in positions:
                    ref = mut_seq[pos]
                    alt = self.rng.choice([b for b in BASES if b != ref])
                    mut_seq[pos] = alt
                all_tokens.append(self._tokenize("".join(mut_seq)))
                all_lengths.append(len(seq))

        tokens = np.stack(all_tokens)  # (B * n_per, L)
        lengths = np.array(all_lengths, dtype=np.int32)
        return tokens, n_per, lengths


# =========================================================================
# Teacher wrapper — extracts full token embeddings via PyTorch
# =========================================================================
class TeacherExtractor:
    """Run teacher model, return full token-level hidden states as numpy."""

    def __init__(self, teacher_key):
        import torch
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from notebooks.processing.process_epistasis import FULL_MODEL_CONFIG, _build_model

        if teacher_key not in FULL_MODEL_CONFIG:
            raise ValueError(f"Unknown teacher: {teacher_key}")

        _, init_spec = FULL_MODEL_CONFIG[teacher_key]
        self.model = _build_model(teacher_key, init_spec)
        if self.model is None:
            raise RuntimeError(f"Failed to load: {teacher_key}")

        self.teacher_key = teacher_key

        # Get embedding dimension
        test_emb = self.model.embed("ACGT" * 100, pool="tokens", return_numpy=True)
        self.embed_dim = test_emb.shape[-1]
        logger.info("Teacher %s: embed_dim=%d", teacher_key, self.embed_dim)

    def get_token_embeddings(self, sequences, seq_lengths):
        """Get full token embeddings for a batch of sequences.

        Args:
            sequences: list of str or (B, L) token array
            seq_lengths: (B,) actual lengths

        Returns:
            embeddings: (B, max_L, D) numpy float32
        """
        import torch

        if isinstance(sequences, np.ndarray):
            # Convert token IDs back to sequences
            id_to_char = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}
            str_seqs = []
            for i in range(len(sequences)):
                L = int(seq_lengths[i])
                chars = [id_to_char.get(int(t), "N") for t in sequences[i, :L]]
                str_seqs.append("".join(chars))
        else:
            str_seqs = sequences

        # Run teacher one sequence at a time (safest for variable architectures)
        all_embs = []
        max_L = max(seq_lengths)
        for seq in str_seqs:
            emb = self.model.embed(seq, pool="tokens", return_numpy=True)
            # emb shape: (L_tokens, D) — may differ from input length for k-mer models
            if emb.ndim == 1:
                emb = emb[np.newaxis, :]  # (1, D)
            # Pad to max_L
            if emb.shape[0] < max_L:
                pad = np.zeros((max_L - emb.shape[0], emb.shape[-1]), dtype=np.float32)
                emb = np.concatenate([emb, pad], axis=0)
            elif emb.shape[0] > max_L:
                emb = emb[:max_L]
            all_embs.append(emb)

        return np.stack(all_embs).astype(np.float32)  # (B, max_L, D)


# =========================================================================
# JAX Student model
# =========================================================================
def build_student(hidden_dim, n_downsamples, n_transformer_layers, embed_dim,
                  seq_length):
    """Build Haiku student model that outputs (B, L, embed_dim) token embeddings."""
    import jax
    import jax.numpy as jnp
    import haiku as hk

    def student_fn(tokens):
        B, L = tokens.shape

        x = hk.Embed(vocab_size=5, embed_dim=hidden_dim)(tokens)  # (B, L, H)

        # Downsample tower
        skips = []
        for i in range(n_downsamples):
            x = hk.Conv1D(hidden_dim, kernel_shape=7, padding="SAME",
                          name=f"down_conv_{i}")(x)
            x = jax.nn.gelu(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name=f"down_ln_{i}")(x)
            skips.append(x)
            x = hk.Conv1D(hidden_dim, kernel_shape=2, stride=2, padding="VALID",
                          name=f"down_stride_{i}")(x)

        # Transformer bottleneck
        for i in range(n_transformer_layers):
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

            residual = x
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name=f"ffn_ln_{i}")(x)
            x = hk.Linear(hidden_dim * 4, name=f"ffn_up_{i}")(x)
            x = jax.nn.gelu(x)
            x = hk.Linear(hidden_dim, name=f"ffn_down_{i}")(x)
            x = x + residual

        # Upsample tower
        for i in range(n_downsamples - 1, -1, -1):
            x = jnp.repeat(x, 2, axis=1)
            skip = skips[i]
            x = x[:, :skip.shape[1], :]
            x = x + skip
            x = hk.Conv1D(hidden_dim, kernel_shape=7, padding="SAME",
                          name=f"up_conv_{i}")(x)
            x = jax.nn.gelu(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name=f"up_ln_{i}")(x)

        # Project to teacher's embedding dim — FULL token-level output
        x = hk.Linear(embed_dim, name="projection")(x)  # (B, L, D)
        return x

    return hk.without_apply_rng(hk.transform(student_fn))


# =========================================================================
# Training
# =========================================================================
def train(args):
    import jax
    import jax.numpy as jnp
    import optax

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Teacher
    teacher = TeacherExtractor(args.teacher)
    embed_dim = teacher.embed_dim

    # Sampler
    sampler = GenomicWindowSampler(
        args.fasta, args.seq_length,
        max_mutations=args.max_mutations,
        variants_per_window=args.variants_per_window,
        seed=args.seed,
    )

    # Student
    model = build_student(
        args.hidden_dim, args.n_downsamples,
        args.n_transformer_layers, embed_dim, args.seq_length,
    )

    rng_key = jax.random.PRNGKey(args.seed)
    # Pad seq_length to multiple of 2^n_downsamples
    pad_mult = 2 ** args.n_downsamples
    padded_len = ((args.seq_length + pad_mult - 1) // pad_mult) * pad_mult
    dummy = jnp.zeros((1, padded_len), dtype=jnp.int32)
    params = model.init(rng_key, dummy)

    n_params = sum(x.size for x in jax.tree.leaves(params))
    logger.info("Student: %d params (%.1fM), teacher embed_dim=%d",
                n_params, n_params / 1e6, embed_dim)

    # Optimizer
    steps_per_epoch = args.steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    schedule = optax.cosine_decay_schedule(init_value=args.lr, decay_steps=total_steps)
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=0.01)
    opt_state = optimizer.init(params)

    # HuggingFace setup
    hf_api = None
    if args.hf_repo:
        try:
            from huggingface_hub import HfApi, create_repo
            hf_api = HfApi()
            create_repo(args.hf_repo, exist_ok=True, repo_type="model")

            card = f"""---
tags: [genomics, distillation, jax, dna-language-model]
license: apache-2.0
---
# Distilled {args.teacher}
- **Student**: {n_params:,} params ({n_params/1e6:.1f}M) U-Net (Haiku/JAX)
- **Teacher**: {args.teacher} ({embed_dim}d embeddings)
- **Training**: Online distillation on full token-level embeddings
- **Data**: Streaming from hg38, WT + 1-{args.max_mutations} mutation variants
- **Context**: {args.seq_length:,} bp
"""
            card_path = output_dir / "README.md"
            card_path.write_text(card)
            hf_api.upload_file(path_or_fileobj=str(card_path),
                               path_in_repo="README.md", repo_id=args.hf_repo,
                               commit_message="Init")
            logger.info("HF repo: %s", args.hf_repo)
        except Exception as e:
            logger.warning("HF setup failed: %s", e)
            hf_api = None

    # JIT training step
    @jax.jit
    def train_step(params, opt_state, student_tokens, teacher_embs, mask):
        def loss_fn(params):
            pred = model.apply(params, student_tokens)  # (B, L, D)
            # MSE only on non-padded positions
            diff = (pred - teacher_embs) ** 2  # (B, L, D)
            # mask: (B, L) — 1 for real, 0 for pad
            diff = diff * mask[:, :, None]
            return diff.sum() / (mask.sum() * embed_dim + 1e-10)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Training loop
    log_entries = []
    best_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_steps = 0

        for step in range(steps_per_epoch):
            # Sample batch
            tokens_np, n_per, lengths = sampler.sample_batch(args.batch_size)
            # tokens_np: (B * n_per, L)

            # Teacher forward (PyTorch)
            teacher_embs_np = teacher.get_token_embeddings(tokens_np, lengths)
            # teacher_embs_np: (B * n_per, L_teacher, D)

            # Pad student input to padded_len
            B_total = tokens_np.shape[0]
            if tokens_np.shape[1] < padded_len:
                pad_width = padded_len - tokens_np.shape[1]
                tokens_np = np.pad(tokens_np, ((0, 0), (0, pad_width)),
                                   constant_values=4)  # pad with N token

            # Match teacher embedding length to padded_len
            L_teacher = teacher_embs_np.shape[1]
            if L_teacher < padded_len:
                pad_emb = np.zeros((B_total, padded_len - L_teacher, embed_dim),
                                   dtype=np.float32)
                teacher_embs_np = np.concatenate([teacher_embs_np, pad_emb], axis=1)
            elif L_teacher > padded_len:
                teacher_embs_np = teacher_embs_np[:, :padded_len, :]

            # Mask: 1 for real positions, 0 for padding
            mask_np = np.zeros((B_total, padded_len), dtype=np.float32)
            for i in range(B_total):
                real_len = min(int(lengths[i]), padded_len)
                # Also limited by teacher token count
                real_len = min(real_len, L_teacher)
                mask_np[i, :real_len] = 1.0

            # To JAX
            tokens_jax = jnp.array(tokens_np, dtype=jnp.int32)
            teacher_jax = jnp.array(teacher_embs_np, dtype=jnp.float32)
            mask_jax = jnp.array(mask_np, dtype=jnp.float32)

            params, opt_state, loss = train_step(
                params, opt_state, tokens_jax, teacher_jax, mask_jax)

            total_loss += float(loss)
            n_steps += 1
            global_step += 1

        avg_loss = total_loss / max(n_steps, 1)
        lr_now = float(schedule(global_step))
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        logger.info("Epoch %d/%d: loss=%.6f, lr=%.2e, step=%d%s",
                     epoch + 1, args.epochs, avg_loss, lr_now, global_step,
                     " *best*" if is_best else "")

        # Save checkpoints
        for tag in (["latest"] + (["best"] if is_best else [])):
            flat = {f"{'/'.join(str(k) for k in p)}": np.array(v)
                    for p, v in jax.tree_util.tree_leaves_with_path(params)}
            np.savez(output_dir / f"{tag}_params.npz", **flat)

        np.savez(output_dir / "config.npz",
                 teacher=args.teacher, embed_dim=embed_dim,
                 seq_length=args.seq_length, padded_len=padded_len,
                 hidden_dim=args.hidden_dim,
                 n_downsamples=args.n_downsamples,
                 n_transformer_layers=args.n_transformer_layers,
                 n_params=n_params, best_loss=best_loss,
                 last_epoch=epoch + 1, total_steps=global_step)

        log_entries.append({
            "epoch": epoch + 1, "loss": avg_loss,
            "best_loss": best_loss, "lr": lr_now, "step": global_step,
        })
        log_path = output_dir / "training_log.json"
        log_path.write_text(json.dumps(log_entries, indent=2))

        # Push to HF
        if hf_api and args.hf_repo:
            try:
                files = [
                    ("latest_params.npz", "latest_params.npz"),
                    ("config.npz", "config.npz"),
                    ("training_log.json", "training_log.json"),
                ]
                if is_best:
                    files.append(("best_params.npz", "best_params.npz"))
                for local, remote in files:
                    hf_api.upload_file(
                        path_or_fileobj=str(output_dir / local),
                        path_in_repo=remote, repo_id=args.hf_repo,
                        commit_message=f"Epoch {epoch+1}: loss={avg_loss:.6f}"
                                       f"{' (best)' if is_best else ''}",
                    )
                logger.info("  → pushed to HF")
            except Exception as e:
                logger.warning("  HF push failed: %s", e)

    logger.info("Done. Best loss: %.6f", best_loss)
    if args.hf_repo:
        logger.info("https://huggingface.co/%s", args.hf_repo)


# =========================================================================
# CLI
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Online distillation: any PyTorch teacher → JAX student")
    parser.add_argument("--teacher", required=True,
                        help="Teacher model key (e.g. borzoi, ntv3_100m_post)")
    parser.add_argument("--fasta", required=True,
                        help="Reference FASTA (e.g. hg38.fa)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-length", type=int, default=15_000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-downsamples", type=int, default=5)
    parser.add_argument("--n-transformer-layers", type=int, default=4)
    parser.add_argument("--max-mutations", type=int, default=5)
    parser.add_argument("--variants-per-window", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Windows per batch (total seqs = batch_size * (1+variants))")
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-repo", default=None,
                        help="HuggingFace repo for progress tracking")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
