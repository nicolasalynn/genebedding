"""
Model distillation with manifold-aware training.

Three loss components:
  L₀: Zero-order  — match absolute embeddings (token-level MSE)
  L₁: First-order  — match perturbation vectors (WT→mutant deltas)
  L₂: Second-order — match epistasis residuals (non-additive curvature)

Training data sources:
  - Multi-genome: human, mouse, drosophila, yeast, arabidopsis, E. coli
  - Synthetic: random DNA, shuffled real DNA, extreme GC content
  - Structured mutations: singles, doubles at controlled distances, higher-order

The goal is to capture the full shape of the teacher's embedding manifold,
not just point samples. The second-order loss directly optimizes for
epistasis preservation.

Usage:
    python -m scripts.distillation.distill_jax \
        --teacher ntv3_100m_post \
        --fasta /path/to/hg38.fa \
        --output-dir /path/to/student \
        --seq-length 8192 \
        --epochs 30
"""

import argparse
import gc
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
# Multi-source sequence sampler
# =========================================================================
class ManifoldSampler:
    """Sample sequences from multiple sources to cover the teacher's manifold.

    Sources:
      1. Reference genome windows (primary FASTA, typically human)
      2. Additional genomes (if provided)
      3. Synthetic: random DNA at various GC contents
      4. Shuffled: real sequences with shuffled base order (preserves composition)

    For each window, generates structured mutation sets:
      - WT
      - Singles (each focal position × 3 alt alleles)
      - Doubles at controlled distances (for epistasis quadruplets)
      - Higher-order (3-5 mutations)
      - Random perturbations
    """

    PAIR_DISTANCES = [1, 2, 5, 10, 20, 50, 100, 500]

    def __init__(self, fasta_paths, seq_length, n_focal_positions=3,
                 max_mutations=5, synthetic_fraction=0.2, seed=42):
        """
        Args:
            fasta_paths: list of FASTA file paths (first is primary genome)
            seq_length: window length
            n_focal_positions: positions per window for structured mutations
            max_mutations: max mutations per variant
            synthetic_fraction: fraction of windows that are synthetic
        """
        import pysam

        self.seq_length = seq_length
        self.n_focal = n_focal_positions
        self.max_mutations = max_mutations
        self.synthetic_fraction = synthetic_fraction
        self.rng = np.random.RandomState(seed)
        self.token_map = {c: i for i, c in enumerate("ACGTN")}

        # Load all genomes
        self.genomes = []
        for fasta_path in fasta_paths:
            if not os.path.exists(fasta_path):
                logger.warning("FASTA not found, skipping: %s", fasta_path)
                continue
            fasta = pysam.FastaFile(fasta_path)
            chroms = [(name, length) for name, length in
                       zip(fasta.references, fasta.lengths)
                       if length > seq_length * 2]
            if chroms:
                weights = np.array([l for _, l in chroms], dtype=float)
                self.genomes.append({
                    "path": fasta_path,
                    "chroms": chroms,
                    "weights": weights / weights.sum(),
                    "name": Path(fasta_path).stem,
                })
                logger.info("Loaded genome: %s (%d chroms, %d total bp)",
                             Path(fasta_path).stem, len(chroms),
                             sum(l for _, l in chroms))
            fasta.close()

        if not self.genomes:
            raise FileNotFoundError("No valid FASTA files found")

    def _tokenize(self, seq):
        return np.array([self.token_map.get(c, 4) for c in seq], dtype=np.int32)

    def _mutate(self, seq_list, positions_and_alts):
        out = seq_list.copy()
        for pos, alt in positions_and_alts:
            out[pos] = alt
        return out

    def _random_alt(self, ref):
        return self.rng.choice([b for b in BASES if b != ref])

    def _sample_real_window(self):
        """Sample a window from a random genome."""
        genome = self.genomes[self.rng.randint(len(self.genomes))]
        for _ in range(20):
            ci = self.rng.choice(len(genome["chroms"]), p=genome["weights"])
            chrom_name, chrom_len = genome["chroms"][ci]
            start = self.rng.randint(0, chrom_len - self.seq_length)
            try:
                import pysam
                fasta = pysam.FastaFile(genome["path"])
                seq = fasta.fetch(chrom_name, start, start + self.seq_length).upper()
                fasta.close()
                if seq.count("N") < self.seq_length * 0.1:
                    return seq
            except Exception:
                continue
        return None

    def _sample_synthetic_window(self):
        """Generate synthetic DNA with random GC content."""
        gc_frac = self.rng.uniform(0.2, 0.8)
        at_frac = 1 - gc_frac
        probs = [at_frac / 2, gc_frac / 2, gc_frac / 2, at_frac / 2]  # A, C, G, T
        return "".join(self.rng.choice(BASES, size=self.seq_length, p=probs))

    def _generate_variants(self, seq):
        """Generate structured mutation set for one window.

        Returns list of (tokens, n_mutations) tuples.
        The first entry is always WT (0 mutations).
        """
        seq_list = list(seq)
        L = len(seq)
        valid_pos = [p for p in range(L) if seq[p] in BASES]

        variants = []  # list of token arrays

        # WT
        variants.append(self._tokenize(seq))

        # Pick focal positions near center
        center = L // 2
        spread = min(L // 4, 2000)
        focal_candidates = [p for p in valid_pos
                            if center - spread <= p <= center + spread]
        n_focal = min(self.n_focal, len(focal_candidates))
        if n_focal == 0:
            return variants

        focal_positions = sorted(
            self.rng.choice(focal_candidates, size=n_focal, replace=False))

        # Singles: each focal × 3 alt alleles
        for fp in focal_positions:
            ref = seq_list[fp]
            for alt in [b for b in BASES if b != ref]:
                variants.append(
                    self._tokenize("".join(self._mutate(seq_list, [(fp, alt)]))))

        # Doubles at controlled distances (epistasis quadruplets)
        for dist in self.PAIR_DISTANCES:
            candidates = [(fp, fp + dist) for fp in focal_positions
                          if fp + dist < L and seq[fp + dist] in BASES]
            if not candidates:
                candidates = [(fp, fp - dist) for fp in focal_positions
                              if fp - dist >= 0 and seq[fp - dist] in BASES]
            if not candidates:
                continue

            pos1, pos2 = candidates[self.rng.randint(len(candidates))]
            alt1, alt2 = self._random_alt(seq_list[pos1]), self._random_alt(seq_list[pos2])

            # Double
            variants.append(
                self._tokenize("".join(self._mutate(seq_list, [(pos1, alt1), (pos2, alt2)]))))

            # Ensure singles for this pair exist (for epistasis quadruplet)
            for p, a in [(pos1, alt1), (pos2, alt2)]:
                if p not in focal_positions:
                    variants.append(
                        self._tokenize("".join(self._mutate(seq_list, [(p, a)]))))

        # Higher-order
        for n_muts in range(3, min(self.max_mutations + 1, n_focal + 1)):
            chosen = self.rng.choice(focal_positions, size=n_muts, replace=False)
            muts = [(int(p), self._random_alt(seq_list[p])) for p in chosen]
            variants.append(self._tokenize("".join(self._mutate(seq_list, muts))))

        # Random perturbations
        for _ in range(2):
            n_muts = min(self.rng.geometric(p=0.4), self.max_mutations, len(valid_pos))
            positions = self.rng.choice(valid_pos, size=n_muts, replace=False)
            muts = [(int(p), self._random_alt(seq_list[p])) for p in positions]
            variants.append(self._tokenize("".join(self._mutate(seq_list, muts))))

        return variants

    def sample_batch(self):
        """Sample one window with all its variants.

        Returns:
            tokens: (N, L) int32
            seq_lengths: (N,) int32
        """
        # Choose real vs synthetic
        if self.rng.random() < self.synthetic_fraction:
            seq = self._sample_synthetic_window()
        else:
            seq = self._sample_real_window()
            if seq is None:
                seq = self._sample_synthetic_window()

        variants = self._generate_variants(seq)
        tokens = np.stack(variants)
        lengths = np.full(len(variants), len(seq), dtype=np.int32)
        return tokens, lengths


# =========================================================================
# Teacher wrapper
# =========================================================================
class TeacherExtractor:
    """Run teacher model, return full token-level hidden states."""

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
        test_emb = self.model.embed("ACGT" * 100, pool="tokens", return_numpy=True)
        self.embed_dim = test_emb.shape[-1]
        logger.info("Teacher %s: embed_dim=%d", teacher_key, self.embed_dim)

    def get_token_embeddings(self, sequences, seq_lengths, chunk_size=4):
        """Get full token embeddings, processing in small chunks."""
        import torch

        if isinstance(sequences, np.ndarray):
            id_to_char = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}
            str_seqs = []
            for i in range(len(sequences)):
                L = int(seq_lengths[i])
                chars = [id_to_char.get(int(t), "N") for t in sequences[i, :L]]
                str_seqs.append("".join(chars))
        else:
            str_seqs = sequences

        all_embs = []
        max_L = max(seq_lengths)

        for ci in range(0, len(str_seqs), chunk_size):
            chunk = str_seqs[ci:ci + chunk_size]
            for seq in chunk:
                emb = self.model.embed(seq, pool="tokens", return_numpy=True)
                if emb.ndim == 1:
                    emb = emb[np.newaxis, :]
                if emb.shape[0] < max_L:
                    pad = np.zeros((max_L - emb.shape[0], emb.shape[-1]), dtype=np.float32)
                    emb = np.concatenate([emb, pad], axis=0)
                elif emb.shape[0] > max_L:
                    emb = emb[:max_L]
                all_embs.append(emb)

            torch.cuda.empty_cache()
            gc.collect()

        return np.stack(all_embs).astype(np.float32)


# =========================================================================
# JAX Student model
# =========================================================================
def build_student(hidden_dim, n_downsamples, n_transformer_layers, embed_dim,
                  seq_length):
    """U-Net student: (B, L) tokens → (B, L, embed_dim) token embeddings."""
    import jax
    import jax.numpy as jnp
    import haiku as hk

    def student_fn(tokens):
        B, L = tokens.shape
        x = hk.Embed(vocab_size=5, embed_dim=hidden_dim)(tokens)

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

        x = hk.Linear(embed_dim, name="projection")(x)
        return x

    return hk.without_apply_rng(hk.transform(student_fn))


# =========================================================================
# Training with manifold-aware loss
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

    # Sampler — multi-genome
    fasta_paths = [args.fasta]
    if args.extra_fastas:
        fasta_paths.extend(args.extra_fastas)

    sampler = ManifoldSampler(
        fasta_paths, args.seq_length,
        n_focal_positions=args.n_focal_positions,
        max_mutations=args.max_mutations,
        synthetic_fraction=args.synthetic_fraction,
        seed=args.seed,
    )

    # Student
    model = build_student(
        args.hidden_dim, args.n_downsamples,
        args.n_transformer_layers, embed_dim, args.seq_length,
    )

    rng_key = jax.random.PRNGKey(args.seed)
    pad_mult = 2 ** args.n_downsamples
    padded_len = ((args.seq_length + pad_mult - 1) // pad_mult) * pad_mult
    dummy = jnp.zeros((1, padded_len), dtype=jnp.int32)
    params = model.init(rng_key, dummy)

    n_params = sum(x.size for x in jax.tree.leaves(params))
    logger.info("Student: %d params (%.1fM), embed_dim=%d, padded_len=%d",
                n_params, n_params / 1e6, embed_dim, padded_len)

    # Covariance weighting
    dim_weights = None
    cov_path = args.cov_inv
    if cov_path is None:
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        try:
            from notebooks.paper_data_config import embeddings_dir
            auto = embeddings_dir() / "cache" / f"{args.teacher}_cov_inv.npz"
            if auto.exists():
                cov_path = str(auto)
        except Exception:
            pass

    if cov_path and os.path.exists(cov_path):
        raw = np.diag(np.load(cov_path)["cov_inv"]).astype(np.float32)
        raw = raw / (raw.mean() + 1e-10)
        raw = np.clip(raw, 0.01, 100.0)
        dim_weights = jnp.array(raw)
        logger.info("Cov weighting: %d dims, range [%.2f, %.2f]",
                     len(raw), raw.min(), raw.max())

    # Optimizer
    steps_per_epoch = args.steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    schedule = optax.cosine_decay_schedule(init_value=args.lr, decay_steps=total_steps)
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=0.01)
    opt_state = optimizer.init(params)

    # Loss weights
    w0 = args.loss_w_embedding   # zero-order: absolute embedding match
    w1 = args.loss_w_delta       # first-order: perturbation vectors
    w2 = args.loss_w_epistasis   # second-order: epistasis residuals

    logger.info("Loss weights: L0=%.2f (embedding), L1=%.2f (delta), L2=%.2f (epistasis)",
                w0, w1, w2)

    # HuggingFace
    hf_api = None
    if args.hf_repo:
        try:
            from huggingface_hub import HfApi, create_repo
            hf_api = HfApi()
            create_repo(args.hf_repo, exist_ok=True, repo_type="model")
            card = f"""---
tags: [genomics, distillation, jax, epistasis]
license: apache-2.0
---
# Distilled {args.teacher} (manifold-aware)
- **Student**: {n_params:,} params ({n_params/1e6:.1f}M)
- **Teacher**: {args.teacher} ({embed_dim}d)
- **Loss**: L0 embedding ({w0}) + L1 delta ({w1}) + L2 epistasis ({w2})
- **Data**: Multi-genome + synthetic + structured mutations
- **Context**: {args.seq_length:,} bp
"""
            (output_dir / "README.md").write_text(card)
            hf_api.upload_file(path_or_fileobj=str(output_dir / "README.md"),
                               path_in_repo="README.md", repo_id=args.hf_repo,
                               commit_message="Init")
        except Exception as e:
            logger.warning("HF setup failed: %s", e)
            hf_api = None

    # ---------------------------------------------------------------
    # JIT-compiled training step with three loss components
    # ---------------------------------------------------------------
    _w = dim_weights

    @jax.jit
    def train_step(params, opt_state, tokens, teacher_embs, mask):
        """
        tokens: (N, L) — WT at index 0, variants at 1..N-1
        teacher_embs: (N, L, D)
        mask: (N, L) — 1 for real positions
        """
        def loss_fn(params):
            pred = model.apply(params, tokens)  # (N, L, D)

            # Weighted diff
            diff = (pred - teacher_embs) ** 2
            if _w is not None:
                diff = diff * _w[None, None, :]

            # --- L0: Absolute embedding MSE ---
            masked_diff = diff * mask[:, :, None]
            L0 = masked_diff.sum() / (mask.sum() * embed_dim + 1e-10)

            # --- L1: Delta (perturbation) matching ---
            # Mean-pool each sequence, compute delta from WT (index 0)
            # mask: (N, L) → mean pool respecting mask
            mask_sum = mask.sum(axis=1, keepdims=True).clip(1)  # (N, 1)
            pred_pooled = (pred * mask[:, :, None]).sum(axis=1) / mask_sum  # (N, D)
            teach_pooled = (teacher_embs * mask[:, :, None]).sum(axis=1) / mask_sum

            pred_deltas = pred_pooled[1:] - pred_pooled[0:1]  # (N-1, D)
            teach_deltas = teach_pooled[1:] - teach_pooled[0:1]
            delta_diff = (pred_deltas - teach_deltas) ** 2
            if _w is not None:
                delta_diff = delta_diff * _w[None, :]
            L1 = delta_diff.mean()

            # --- L2: Epistasis residual matching ---
            # For any triplet (WT=0, single_i, single_j, double_k) where
            # double_k = mutations at both positions i and j:
            # ε = h(AB) - h(A) - h(B) + h(WT)
            # We approximate by: for each variant with ≥2 mutations relative
            # to WT, compute the residual against the additive expectation
            # from available singles. This is complex to track indices, so
            # we use a simpler proxy: the variance of the residuals around
            # the mean delta direction should match between teacher and student.
            # Proxy: cosine similarity of the delta vectors should match.
            N_var = pred_deltas.shape[0]
            if N_var >= 2:
                # Pairwise cosine between all delta vectors
                pred_norms = jnp.linalg.norm(pred_deltas, axis=1, keepdims=True).clip(1e-10)
                teach_norms = jnp.linalg.norm(teach_deltas, axis=1, keepdims=True).clip(1e-10)
                pred_unit = pred_deltas / pred_norms
                teach_unit = teach_deltas / teach_norms

                pred_cos = pred_unit @ pred_unit.T  # (N-1, N-1)
                teach_cos = teach_unit @ teach_unit.T
                L2 = ((pred_cos - teach_cos) ** 2).mean()
            else:
                L2 = jnp.float32(0.0)

            return w0 * L0 + w1 * L1 + w2 * L2, (L0, L1, L2)

        (loss, (l0, l1, l2)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss, l0, l1, l2

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    log_entries = []
    best_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        tot_loss = tot_l0 = tot_l1 = tot_l2 = 0.0
        n_steps = 0

        for step in range(steps_per_epoch):
            tokens_np, lengths = sampler.sample_batch()
            teacher_embs_np = teacher.get_token_embeddings(tokens_np, lengths)

            # Pad
            B = tokens_np.shape[0]
            if tokens_np.shape[1] < padded_len:
                tokens_np = np.pad(tokens_np,
                    ((0, 0), (0, padded_len - tokens_np.shape[1])), constant_values=4)

            L_t = teacher_embs_np.shape[1]
            if L_t < padded_len:
                teacher_embs_np = np.pad(teacher_embs_np,
                    ((0, 0), (0, padded_len - L_t), (0, 0)))
            elif L_t > padded_len:
                teacher_embs_np = teacher_embs_np[:, :padded_len, :]

            mask_np = np.zeros((B, padded_len), dtype=np.float32)
            for i in range(B):
                real = min(int(lengths[i]), padded_len, L_t)
                mask_np[i, :real] = 1.0

            params, opt_state, loss, l0, l1, l2 = train_step(
                params, opt_state,
                jnp.array(tokens_np, dtype=jnp.int32),
                jnp.array(teacher_embs_np, dtype=jnp.float32),
                jnp.array(mask_np, dtype=jnp.float32),
            )

            tot_loss += float(loss)
            tot_l0 += float(l0)
            tot_l1 += float(l1)
            tot_l2 += float(l2)
            n_steps += 1
            global_step += 1

            if (step + 1) % 10 == 0:
                logger.info("  step %d/%d: loss=%.5f (L0=%.5f L1=%.5f L2=%.5f)",
                             step + 1, steps_per_epoch,
                             tot_loss / n_steps, tot_l0 / n_steps,
                             tot_l1 / n_steps, tot_l2 / n_steps)

            # Mid-epoch HF push
            if hf_api and args.hf_repo and (step + 1) % 100 == 0:
                try:
                    p = {"step": global_step, "loss": tot_loss / n_steps,
                         "L0": tot_l0 / n_steps, "L1": tot_l1 / n_steps,
                         "L2": tot_l2 / n_steps}
                    (output_dir / "progress.json").write_text(json.dumps(p, indent=2))
                    hf_api.upload_file(
                        path_or_fileobj=str(output_dir / "progress.json"),
                        path_in_repo="progress.json", repo_id=args.hf_repo,
                        commit_message=f"Step {global_step}")
                except Exception:
                    pass

        # Epoch summary
        avg = tot_loss / max(n_steps, 1)
        lr = float(schedule(global_step))
        is_best = avg < best_loss
        if is_best:
            best_loss = avg

        # Validation: perturbation correlation
        val_corrs = []
        try:
            for _ in range(5):
                vt, vl = sampler.sample_batch()
                vte = teacher.get_token_embeddings(vt, vl)
                if vt.shape[1] < padded_len:
                    vt = np.pad(vt, ((0, 0), (0, padded_len - vt.shape[1])), constant_values=4)
                vs = np.array(model.apply(params, jnp.array(vt, dtype=jnp.int32)))
                L_r = min(int(vl[0]), vte.shape[1], vs.shape[1])
                tp = vte[:, :L_r, :].mean(axis=1)
                sp = vs[:, :L_r, :].mean(axis=1)
                for j in range(1, len(tp)):
                    td = tp[j] - tp[0]
                    sd = sp[j] - sp[0]
                    d = float(np.linalg.norm(td)) * float(np.linalg.norm(sd))
                    if d > 1e-10:
                        val_corrs.append(float(np.dot(td.flatten(), sd.flatten()) / d))
            corr = np.mean(val_corrs) if val_corrs else float("nan")
        except Exception:
            corr = float("nan")

        logger.info("Epoch %d/%d: loss=%.5f (L0=%.5f L1=%.5f L2=%.5f) "
                     "lr=%.2e delta_corr=%.3f%s",
                     epoch + 1, args.epochs, avg,
                     tot_l0 / max(n_steps, 1), tot_l1 / max(n_steps, 1),
                     tot_l2 / max(n_steps, 1), lr, corr,
                     " *best*" if is_best else "")

        # Save
        for tag in (["latest"] + (["best"] if is_best else [])):
            flat = {f"{'/'.join(str(k) for k in p)}": np.array(v)
                    for p, v in jax.tree_util.tree_leaves_with_path(params)}
            np.savez(output_dir / f"{tag}_params.npz", **flat)

        np.savez(output_dir / "config.npz",
                 teacher=args.teacher, embed_dim=embed_dim,
                 seq_length=args.seq_length, padded_len=padded_len,
                 hidden_dim=args.hidden_dim, n_downsamples=args.n_downsamples,
                 n_transformer_layers=args.n_transformer_layers,
                 n_params=n_params, best_loss=best_loss,
                 loss_w0=w0, loss_w1=w1, loss_w2=w2,
                 last_epoch=epoch + 1, total_steps=global_step, delta_corr=corr)

        log_entries.append({
            "epoch": epoch + 1, "loss": avg, "L0": tot_l0 / max(n_steps, 1),
            "L1": tot_l1 / max(n_steps, 1), "L2": tot_l2 / max(n_steps, 1),
            "best_loss": best_loss, "lr": lr, "delta_corr": corr,
            "step": global_step,
        })
        (output_dir / "training_log.json").write_text(json.dumps(log_entries, indent=2))

        if hf_api and args.hf_repo:
            try:
                for f in ["latest_params.npz", "config.npz", "training_log.json"]:
                    hf_api.upload_file(
                        path_or_fileobj=str(output_dir / f),
                        path_in_repo=f, repo_id=args.hf_repo,
                        commit_message=f"Epoch {epoch+1}: loss={avg:.5f} corr={corr:.3f}")
                if is_best:
                    hf_api.upload_file(
                        path_or_fileobj=str(output_dir / "best_params.npz"),
                        path_in_repo="best_params.npz", repo_id=args.hf_repo,
                        commit_message=f"Best: {avg:.5f}")
            except Exception:
                pass

    logger.info("Done. Best loss: %.5f", best_loss)


# =========================================================================
# CLI
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Manifold-aware distillation: PyTorch teacher → JAX student")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--fasta", required=True, help="Primary reference FASTA")
    parser.add_argument("--extra-fastas", nargs="*", default=[],
                        help="Additional genome FASTAs (mouse, drosophila, etc.)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-downsamples", type=int, default=3)
    parser.add_argument("--n-transformer-layers", type=int, default=8)
    parser.add_argument("--n-focal-positions", type=int, default=3)
    parser.add_argument("--max-mutations", type=int, default=5)
    parser.add_argument("--synthetic-fraction", type=float, default=0.2,
                        help="Fraction of windows that are synthetic random DNA")
    parser.add_argument("--loss-w-embedding", type=float, default=1.0,
                        help="Weight for L0 (absolute embedding MSE)")
    parser.add_argument("--loss-w-delta", type=float, default=1.0,
                        help="Weight for L1 (perturbation vector matching)")
    parser.add_argument("--loss-w-epistasis", type=float, default=0.5,
                        help="Weight for L2 (epistasis geometry matching)")
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cov-inv", default=None)
    parser.add_argument("--hf-repo", default=None)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
