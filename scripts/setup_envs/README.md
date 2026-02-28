# Environment setup scripts (CUDA/GPU)

One script per model (or per family, e.g. Nucleotide Transformer). Each script:

1. Creates a **conda** environment with the required Python version
2. Installs **PyTorch** (or JAX for AlphaGenome) with **CUDA** support
3. Installs the model package and any weights/repos as needed
4. Installs **genebeddings** (editable) with the matching extra(s)

Run from the **repo root** so `pip install -e .[extra]` finds `pyproject.toml`.  
Requires: conda, CUDA-capable GPU, NVIDIA drivers.

| Script | Env name (default) | Models covered |
|--------|--------------------|----------------|
| `setup_nucleotide_transformer.sh` | nt | nt50_3mer, nt50_multi, nt100_multi, nt250_multi, nt500_multi, nt500_ref, nt2500_multi, nt2500_okgp, v2-* |
| `setup_alphagenome.sh` | alphagenome | alphagenome |
| `setup_evo2.sh` | evo2 | evo2 |
| `setup_spliceai.sh` | spliceai | spliceai (OpenSpliceAI) |
| `setup_convnova.sh` | convnova | convnova |
| `setup_borzoi.sh` | borzoi | borzoi |
| `setup_mutbert.sh` | mutbert | mutbert |
| `setup_hyenadna.sh` | hyenadna | hyenadna |
| `setup_caduceus.sh` | caduceus | caduceus |
| `setup_dnabert.sh` | dnabert | dnabert |
| `setup_rinalmo.sh` | rinalmo | rinalmo |
| `setup_specieslm.sh` | specieslm | specieslm |
| `setup_main.sh` | genebeddings_main | All “main” profile models (nt, convnova, mutbert, hyenadna, caduceus, borzoi, rinalmo, specieslm, dnabert) in one env |

**Usage (example):**

```bash
cd /path/to/genebeddings
bash scripts/setup_envs/setup_nucleotide_transformer.sh
conda activate nt
python -c "from genebeddings.wrappers import NTWrapper; w = NTWrapper(model='nt500_multi'); print(w.embed('ACGT'*100, pool='mean').shape)"
```

Override defaults with env vars (see top of each script), e.g.:

```bash
CONDA_ENV=my_nt CUDA_VERSION=12.4 bash scripts/setup_envs/setup_nucleotide_transformer.sh
```

**Notes:**

- **AlphaGenome:** Installs `alphagenome_research` from GitHub (no PyPI package). Clone dir defaults to `$HOME/alphagenome_research`; override with `ALPHAGENOME_CLONE_DIR`.
- **ConvNova:** For pretrained weights, add `genebeddings/assets/convnova/convnova.yaml` and `last.backbone.pth`; otherwise the wrapper uses random init.
- **SpliceAI:** Set `OPENSPLICEAI_MODEL_DIR` to a directory containing OpenSpliceAI checkpoints if using the OpenSpliceAI backend.
- **CUDA:** Default is CUDA 12.1 (`cu121`). Set `CUDA_VERSION=124` (or your driver version) if needed.

**Testing (e.g. on Lambda Labs GPU instances):** Clone the repo and run the full test from the repo root:

```bash
git clone https://github.com/.../genebeddings.git && cd genebeddings
bash scripts/setup_envs/test_all_envs.sh
```

Use `--skip-setup` to only run wrapper tests in already-created envs. When the package is made pip-installable, the setup and test scripts will be included so the same workflow can be run from an install.
