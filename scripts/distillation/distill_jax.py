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
    """Sample genomic windows with structured epistasis-aware mutations.

    For each window, generates:
      1. WT (reference)
      2. Singles: each focal position mutated alone (all 3 alt alleles)
      3. Doubles: pairs at controlled distances (1, 5, 20, 100, 500bp)
      4. Higher-order: 3-5 mutations at focal positions

    This teaches the student that embedding(double) ≠ embedding(single1) +
    embedding(single2) - embedding(WT), which is exactly epistasis.
    """

    # Distances at which to sample double-mutation pairs
    PAIR_DISTANCES = [1, 2, 5, 10, 20, 50, 100, 500]

    def __init__(self, fasta_path, seq_length, n_focal_positions=5,
                 max_mutations=5, seed=42):
        import pysam
        self.seq_length = seq_length
        self.n_focal = n_focal_positions
        self.max_mutations = max_mutations
        self.rng = np.random.RandomState(seed)
        self.token_map = {c: i for i, c in enumerate("ACGTN")}

        fasta = pysam.FastaFile(fasta_path)
        self.chroms = [(name, length) for name, length in
                       zip(fasta.references, fasta.lengths)
                       if name.startswith("chr") and name[3:].isdigit()
                       and length > seq_length * 2]
        weights = np.array([l for _, l in self.chroms], dtype=float)
        self.chrom_weights = weights / weights.sum()
        fasta.close()

    def _tokenize(self, seq):
        return np.array([self.token_map.get(c, 4) for c in seq], dtype=np.int32)

    def _mutate(self, seq_list, positions_and_alts):
        """Apply mutations to a sequence list. Returns new list."""
        out = seq_list.copy()
        for pos, alt in positions_and_alts:
            out[pos] = alt
        return out

    def _random_alt(self, ref):
        return self.rng.choice([b for b in BASES if b != ref])

    def sample_batch(self, batch_size):
        """Sample batch_size windows with structured mutation sets.

        Returns:
            tokens: (N, L) int32 — all sequences tokenized
            n_per_window: int — sequences per window (variable but padded)
            seq_lengths: (N,) int32
        """
        from seqmat import SeqMat

        all_tokens = []
        all_lengths = []

        for _ in range(batch_size):
            # Sample a genomic window
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
                seq = "".join(self.rng.choice(BASES, size=self.seq_length))

            seq_list = list(seq)
            L = len(seq)
            valid_pos = [p for p in range(L) if seq[p] in BASES]

            # Pick focal positions — center of the window ± spread
            center = L // 2
            spread = min(L // 4, 2000)
            focal_candidates = [p for p in valid_pos
                                if center - spread <= p <= center + spread]
            n_focal = min(self.n_focal, len(focal_candidates))
            focal_positions = sorted(
                self.rng.choice(focal_candidates, size=n_focal, replace=False))

            # ---- 1. WT ----
            all_tokens.append(self._tokenize(seq))
            all_lengths.append(L)

            # ---- 2. Singles: each focal position × all 3 alt alleles ----
            for fp in focal_positions:
                ref = seq_list[fp]
                for alt in [b for b in BASES if b != ref]:
                    mut = self._mutate(seq_list, [(fp, alt)])
                    all_tokens.append(self._tokenize("".join(mut)))
                    all_lengths.append(L)

            # ---- 3. Doubles: pairs at controlled distances ----
            for dist in self.PAIR_DISTANCES:
                # Find a focal position that has a valid partner at this distance
                candidates = [(fp, fp + dist) for fp in focal_positions
                              if fp + dist < L and seq[fp + dist] in BASES]
                if not candidates:
                    candidates = [(fp, fp - dist) for fp in focal_positions
                                  if fp - dist >= 0 and seq[fp - dist] in BASES]
                if not candidates:
                    continue

                pos1, pos2 = candidates[self.rng.randint(len(candidates))]
                ref1, ref2 = seq_list[pos1], seq_list[pos2]
                alt1, alt2 = self._random_alt(ref1), self._random_alt(ref2)

                # Double mutant
                mut = self._mutate(seq_list, [(pos1, alt1), (pos2, alt2)])
                all_tokens.append(self._tokenize("".join(mut)))
                all_lengths.append(L)

                # Also add each single from this pair (if not already a focal)
                # so the student sees WT, single1, single2, double for this pair
                for p, a in [(pos1, alt1), (pos2, alt2)]:
                    if p not in focal_positions:
                        mut_s = self._mutate(seq_list, [(p, a)])
                        all_tokens.append(self._tokenize("".join(mut_s)))
                        all_lengths.append(L)

            # ---- 4. Higher-order: 3-5 mutations at focal positions ----
            for n_muts in range(3, min(self.max_mutations + 1, n_focal + 1)):
                chosen = self.rng.choice(focal_positions, size=n_muts, replace=False)
                muts = [(int(p), self._random_alt(seq_list[p])) for p in chosen]
                mut = self._mutate(seq_list, muts)
                all_tokens.append(self._tokenize("".join(mut)))
                all_lengths.append(L)

            # ---- 5. Random mutations (for diversity) ----
            for _ in range(3):
                n_muts = self.rng.geometric(p=0.4)
                n_muts = min(n_muts, self.max_mutations, len(valid_pos))
                positions = self.rng.choice(valid_pos, size=n_muts, replace=False)
                muts = [(int(p), self._random_alt(seq_list[p])) for p in positions]
                mut = self._mutate(seq_list, muts)
                all_tokens.append(self._tokenize("".join(mut)))
                all_lengths.append(L)

        tokens = np.stack(all_tokens)
        lengths = np.array(all_lengths, dtype=np.int32)
        n_per = len(all_tokens) // batch_size
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
                w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
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
        n_focal_positions=args.n_focal_positions,
        max_mutations=args.max_mutations,
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

    # -------------------------------------------------------------------
    # Load covariance weighting (if available)
    # -------------------------------------------------------------------
    # The null covariance tells us which dimensions carry signal vs noise.
    # We use the diagonal of cov_inv as per-dimension weights:
    # high cov_inv diagonal = low null variance = signal dimension = upweight
    dim_weights = None
    cov_inv_path = args.cov_inv
    if cov_inv_path is None:
        # Try auto-detect from embeddings_dir
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        try:
            from notebooks.paper_data_config import embeddings_dir
            auto_path = embeddings_dir() / "cache" / f"{args.teacher}_cov_inv.npz"
            if auto_path.exists():
                cov_inv_path = str(auto_path)
                logger.info("Auto-detected cov_inv: %s", auto_path)
        except Exception:
            pass

    if cov_inv_path and os.path.exists(cov_inv_path):
        cov_data = np.load(cov_inv_path)
        cov_inv = cov_data["cov_inv"]  # (D, D)
        # Use diagonal as per-dimension importance weight
        # High diagonal = low null variance in this dim = important for signal
        raw_weights = np.diag(cov_inv).astype(np.float32)
        # Normalize so mean weight = 1 (doesn't change loss scale)
        raw_weights = raw_weights / (raw_weights.mean() + 1e-10)
        # Clip extremes to avoid instability
        raw_weights = np.clip(raw_weights, 0.01, 100.0)
        dim_weights = jnp.array(raw_weights)  # (D,)
        logger.info("Covariance weighting: %d dims, weight range [%.2f, %.2f], "
                     "effective dims (>1.0): %d",
                     len(raw_weights), raw_weights.min(), raw_weights.max(),
                     (raw_weights > 1.0).sum())
    else:
        logger.info("No covariance weighting — using uniform MSE. "
                     "Pass --cov-inv to enable.")

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

    # JIT training step — covariance-weighted MSE
    _dim_weights = dim_weights  # capture for JIT closure

    @jax.jit
    def train_step(params, opt_state, student_tokens, teacher_embs, mask):
        def loss_fn(params):
            pred = model.apply(params, student_tokens)  # (B, L, D)
            diff = (pred - teacher_embs) ** 2  # (B, L, D)

            # Apply covariance weighting per dimension
            # Upweights dimensions where null has low variance (signal dims)
            if _dim_weights is not None:
                diff = diff * _dim_weights[None, None, :]  # (B, L, D) * (1, 1, D)

            # Mask out padding positions
            diff = diff * mask[:, :, None]  # (B, L, 1)
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

            step_loss = float(loss)
            total_loss += step_loss
            n_steps += 1
            global_step += 1

            # Per-step logging every 10 steps
            if (step + 1) % 10 == 0:
                running_avg = total_loss / n_steps
                logger.info("  step %d/%d: loss=%.6f (running avg=%.6f)",
                             step + 1, steps_per_epoch, step_loss, running_avg)

            # Mid-epoch HF push every 100 steps
            if hf_api and args.hf_repo and (step + 1) % 100 == 0:
                try:
                    mid_log = {
                        "epoch": epoch + 1, "step": global_step,
                        "step_in_epoch": step + 1,
                        "running_loss": total_loss / n_steps,
                    }
                    mid_log_path = output_dir / "progress.json"
                    mid_log_path.write_text(json.dumps(mid_log, indent=2))
                    hf_api.upload_file(
                        path_or_fileobj=str(mid_log_path),
                        path_in_repo="progress.json",
                        repo_id=args.hf_repo,
                        commit_message=f"Step {global_step}: loss={total_loss/n_steps:.6f}",
                    )
                except Exception:
                    pass  # don't interrupt training for HF issues

        avg_loss = total_loss / max(n_steps, 1)
        lr_now = float(schedule(global_step))
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        # -----------------------------------------------------------
        # Validation: does the student preserve perturbation structure?
        # Sample a few WT+mutant pairs, check if student's delta
        # correlates with teacher's delta (mean-pooled).
        # -----------------------------------------------------------
        val_corrs = []
        try:
            for _v in range(10):  # 10 validation windows
                val_tokens, val_nper, val_lens = sampler.sample_batch(1)
                val_teacher = teacher.get_token_embeddings(val_tokens, val_lens)

                # Pad for student
                vB = val_tokens.shape[0]
                if val_tokens.shape[1] < padded_len:
                    val_tokens = np.pad(val_tokens,
                        ((0, 0), (0, padded_len - val_tokens.shape[1])),
                        constant_values=4)
                val_student = np.array(
                    model.apply(params, jnp.array(val_tokens, dtype=jnp.int32)))

                # Mean-pool both (over real positions only)
                L_real = min(int(val_lens[0]), val_teacher.shape[1], val_student.shape[1])
                t_pooled = val_teacher[:, :L_real, :].mean(axis=1)  # (N, D)
                s_pooled = val_student[:, :L_real, :].mean(axis=1)  # (N, D)

                # WT is first sequence, rest are mutants
                t_wt = t_pooled[0]
                s_wt = s_pooled[0]

                for j in range(1, len(t_pooled)):
                    t_delta = t_pooled[j] - t_wt  # teacher perturbation
                    s_delta = s_pooled[j] - s_wt  # student perturbation
                    # Correlation between teacher and student deltas
                    t_flat = t_delta.flatten()
                    s_flat = s_delta.flatten()
                    denom = (np.linalg.norm(t_flat) * np.linalg.norm(s_flat) + 1e-20)
                    corr = float(np.dot(t_flat, s_flat) / denom)
                    val_corrs.append(corr)

            mean_corr = np.mean(val_corrs)
            median_corr = np.median(val_corrs)
        except Exception as e:
            mean_corr = float("nan")
            median_corr = float("nan")
            logger.warning("Validation failed: %s", e)

        logger.info("Epoch %d/%d: loss=%.6f, lr=%.2e, step=%d, "
                     "delta_corr=%.3f (median=%.3f)%s",
                     epoch + 1, args.epochs, avg_loss, lr_now, global_step,
                     mean_corr, median_corr,
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
            "delta_corr_mean": float(mean_corr),
            "delta_corr_median": float(median_corr),
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
    parser.add_argument("--n-downsamples", type=int, default=3)
    parser.add_argument("--n-transformer-layers", type=int, default=8)
    parser.add_argument("--max-mutations", type=int, default=5)
    parser.add_argument("--n-focal-positions", type=int, default=5,
                        help="Focal positions per window for structured mutations")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Windows per batch (each produces ~40-60 sequences)")
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cov-inv", default=None,
                        help="Path to null cov_inv .npz (auto-detected if not set). "
                             "Used to weight loss: signal dimensions get higher weight.")
    parser.add_argument("--hf-repo", default=None,
                        help="HuggingFace repo for progress tracking")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
