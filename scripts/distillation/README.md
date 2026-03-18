# Model Distillation

Distill any genebeddings teacher model into a smaller/open student model.

## Quick Start

```bash
# Step 1: Generate teacher embeddings (GPU required, ~hours)
python -m scripts.distillation.distill \
    --teacher ntv3_100m_post \
    --data-fasta /path/to/hg38.fa \
    --output-dir /path/to/distill_ntv3 \
    --n-sequences 500000 \
    --seq-length 8192 \
    --generate-only

# Step 2: Train student (GPU required, ~hours)
python -m scripts.distillation.distill \
    --teacher ntv3_100m_post \
    --data-fasta /path/to/hg38.fa \
    --output-dir /path/to/distill_ntv3 \
    --n-sequences 500000 \
    --epochs 20 \
    --batch-size 32
```

## How It Works

1. **Sample** random genomic windows from a reference FASTA
2. **Run teacher** on each window, cache mean-pooled embeddings to disk
3. **Train student** (conv tower + projection) to match teacher embeddings via MSE
4. **Save** student weights for downstream use

## Available Teachers

Any model registered in `process_epistasis.py FULL_MODEL_CONFIG`:

| Teacher key | Model | License |
|-------------|-------|---------|
| `ntv3_100m_post` | NTv3 100M post-trained | Non-commercial |
| `ntv3_100m` | NTv3 100M pre-trained | Non-commercial |
| `borzoi` | Borzoi | Apache 2.0 |
| `alphagenome` | AlphaGenome | Non-commercial |
| `nt500_multi` | NT v2 500M | CC-BY-SA |
| `evo2` | Evo2 7B | Apache 2.0 |
| ... | Any wrapper in genebeddings | Varies |

## Cost Estimates

| Sequences | Teacher inference | Student training | Total |
|-----------|-------------------|-----------------|-------|
| 100K | ~$50 (A100) | ~$100 | ~$150 |
| 500K | ~$250 | ~$500 | ~$750 |
| 1M | ~$500 | ~$1,000 | ~$1,500 |

## Legal Notes

- Distillation from non-commercial models (NTv3, AlphaGenome) is fine for academic use
- For commercial use, distill from open-license models (Borzoi, Evo2, NT v1/v2) OR
  train from scratch on OpenGenome2 (Apache 2.0) using the validated architecture
