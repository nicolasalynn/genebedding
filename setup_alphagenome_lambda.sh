#!/bin/bash
set -e

echo "============================================"
echo "  AlphaGenome Lambda Setup"
echo "============================================"

# 1. Print system info
echo ""
echo ">>> SYSTEM INFO"
nvidia-smi | head -6
echo ""
nvcc --version 2>/dev/null || echo "nvcc not found (ok - JAX ships its own CUDA)"
echo ""

# 2. Install Python 3.11 (AlphaGenome requires >=3.11)
echo ">>> Installing Python 3.11..."
sudo apt update -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# 3. Create venv with Python 3.11
echo ""
echo ">>> Creating virtual environment..."
rm -rf ~/alphagenome_env
python3.11 -m venv ~/alphagenome_env
source ~/alphagenome_env/bin/activate
pip install --upgrade pip setuptools wheel

# 4. Install JAX with CUDA 12
echo ""
echo ">>> Installing JAX..."
pip install "jax[cuda12]==0.6.0"

# 5. Install AlphaGenome dependencies
echo ""
echo ">>> Installing AlphaGenome dependencies..."
pip install \
    dm-haiku==0.0.14 \
    chex==0.1.88 \
    jmp \
    orbax-checkpoint==0.11.32 \
    einshape \
    "etils[epath]" \
    huggingface_hub \
    jaxtyping \
    kagglehub \
    anndata \
    pyfaidx \
    pyranges \
    pandas \
    pyarrow \
    tensorflow \
    numpy \
    aiohttp \
    requests \
    fsspec \
    alphagenome \
    hatchling

# 6. Clone and install alphagenome_research
echo ""
echo ">>> Cloning alphagenome_research..."
cd ~
rm -rf ~/alphagenome_research
git clone https://github.com/google-deepmind/alphagenome_research.git
cd ~/alphagenome_research
pip install .

# 7. Print all versions
echo ""
echo "============================================"
echo "  INSTALLED VERSIONS"
echo "============================================"
python3 << 'PYEOF'
import jax
import jaxlib
import haiku as hk
import chex
import numpy as np
print(f"jax:      {jax.__version__}")
print(f"jaxlib:   {jaxlib.__version__}")
print(f"haiku:    {hk.__version__}")
print(f"chex:     {chex.__version__}")
print(f"numpy:    {np.__version__}")
print(f"devices:  {jax.devices()}")
print(f"backend:  {jax.default_backend()}")
PYEOF
echo ""

# 8. Test AlphaGenome
echo "============================================"
echo "  TESTING ALPHAGENOME"
echo "============================================"
python3 << 'PYEOF'
from alphagenome_research.model import dna_model
print("import OK")

model = dna_model.create_from_huggingface("fold_0")
print("model loaded")

out = model.predict_sequence(
    "ACGT" * 512,
    organism=dna_model.Organism.HOMO_SAPIENS,
    requested_outputs=[],
    ontology_terms=[],
)
print("predict_sequence: SUCCESS")
PYEOF

# 9. Print full environment snapshot
echo ""
echo "============================================"
echo "  FULL ENVIRONMENT SNAPSHOT"
echo "============================================"
echo "NVIDIA Driver:"
nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader
echo ""
echo "CUDA toolkit (if available):"
nvcc --version 2>/dev/null || echo "N/A (using JAX-bundled CUDA)"
echo ""
echo "Key packages:"
pip list | grep -i -E "jax|haiku|chex|orbax|numpy|tensorflow|cuda"
echo ""
echo "============================================"
echo "  DONE"
echo "============================================"
