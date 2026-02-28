#!/usr/bin/env bash
# Single conda env for all "main" profile models (everything except AlphaGenome and Evo2).
# Use this to run the full pipeline with env_profile=main from one environment.
set -e
CONDA_ENV="${CONDA_ENV:-genebeddings_main}"
CUDA_VERSION="${CUDA_VERSION:-121}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "  Main (all non–AlphaGenome/Evo2) env: $CONDA_ENV"
echo "  CUDA: $CUDA_VERSION"
echo "============================================"

conda create -n "$CONDA_ENV" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pip install --upgrade pip setuptools wheel
# torch>=2.6 for NT (torch.load CVE) and borzoi (wrap_triton)
pip install "torch>=2.6" --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
pip install "transformers>=4.30,<4.46" "borzoi-pytorch>=0.4" spliceai-pytorch omegaconf
# RiNALMo has no PyPI package; flash-attn must see torch (--no-build-isolation)
pip install flash-attn==2.3.2 --no-build-isolation
pip install "git+https://github.com/lbcb-sci/RiNALMo.git"
pip install -e .

echo ""
echo ">>> Smoke test: one model per family (NT, ConvNova, Borzoi, RiNALMo)"
python -c "
from genebeddings.wrappers import NTWrapper, ConvNovaWrapper, BorzoiWrapper, RiNALMoWrapper
for name, w in [
    ('NT', NTWrapper(model='nt2500_multi')),
    ('ConvNova', ConvNovaWrapper()),
    ('RiNALMo', RiNALMoWrapper(model_name='giga-v1')),
]:
    e = w.embed('ACGT' * 50, pool='mean')
    print(f'  {name}: embed shape {e.shape}')
print('  Borzoi (slow): skipping embed test; import OK')
print('  OK')
"

echo ""
echo "Done. Activate with: conda activate $CONDA_ENV"
echo "Run pipeline with: env_profile=main (or --env-profile main)"
