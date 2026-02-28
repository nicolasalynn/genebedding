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
python -c "from genebeddings.wrappers import NTWrapper; w = NTWrapper(model='nt2500_multi'); print(w.embed('ACGT'*100, pool='mean').shape)"
```

Override defaults with env vars (see top of each script), e.g.:

```bash
CONDA_ENV=my_nt CUDA_VERSION=12.4 bash scripts/setup_envs/setup_nucleotide_transformer.sh
```

**Notes:**

- **Nucleotide Transformer:** Tests use the 2.5b multi-species model (`nt2500_multi` → [InstaDeepAI/nucleotide-transformer-2.5b-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species)). Gated; requires a HuggingFace token (see below).
- **AlphaGenome:** Installs `alphagenome_research` from [GitHub](https://github.com/google-deepmind/alphagenome_research). Clone dir defaults to `$HOME/alphagenome_research`; override with `ALPHAGENOME_CLONE_DIR`. In this env, genebeddings is installed **non-editable** (`pip install .`) so the env works even when the build backend lacks the editable hook (“No module named 'genebeddings'” is then avoided).
- **Evo2:** Per [ArcInstitute/evo2](https://github.com/ArcInstitute/evo2): Python 3.12, [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) (conda `transformer-engine-torch=2.3.0`), Flash Attention 2.8.0.post2, then `pip install evo2`.
- **RiNALMo:** Per [lbcb-sci/RiNALMo](https://github.com/lbcb-sci/RiNALMo): clone repo, `pip install .`, then `pip install flash-attn==2.3.2`. Clone dir defaults to `$HOME/RiNALMo`; override with `RINALMO_CLONE_DIR`.
- **ConvNova:** For pretrained weights, add `genebeddings/assets/convnova/convnova.yaml` and `last.backbone.pth`; otherwise the wrapper uses random init.
- **Borzoi (flash-attn):** The default FlashZoi checkpoint requires `flash_attn`. The setup script tries a pre-built wheel from [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/releases) (Linux, Python 3.10, CUDA 12, PyTorch 2.4/2.5) so you don’t need to compile. If that fails, it falls back to building from source (needs nvcc, ninja, ~5 min).
- **SpliceAI:** Set `OPENSPLICEAI_MODEL_DIR` to a directory containing OpenSpliceAI checkpoints if using the OpenSpliceAI backend.
- **CUDA:** Default is CUDA 12.1 (`cu121`). Set `CUDA_VERSION=124` (or your driver version) if needed.

**Testing (e.g. on Lambda Labs GPU instances):** Clone the repo and run the full test from the repo root:

```bash
git clone https://github.com/.../genebeddings.git && cd genebeddings
bash scripts/setup_envs/test_all_envs.sh
```

Use `--skip-setup` to only run wrapper tests in already-created envs. When the package is made pip-installable, the setup and test scripts will be included so the same workflow can be run from an install.

**HuggingFace token (for NT and other gated models):** On the remote host (e.g. Lambda instance), create a file so the token is available when the test runs:

```bash
# On the remote (e.g. ssh ubuntu@<ip>):
echo 'YOUR_HF_TOKEN' > ~/.hf_token && chmod 600 ~/.hf_token
```

The remote script sources `~/.hf_token` and sets `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` before running tests. Use a token with read access to the models you need (e.g. InstaDeepAI/nucleotide-transformer-*, AlphaGenome, etc.).

**Remote testing (e.g. Lambda Labs):** From your laptop, run the full test on a GPU host via SSH:

```bash
# From repo root (or scripts/setup_envs)
bash scripts/setup_envs/run_tests_remote.sh ubuntu@<instance-ip>
```

Options: `--repo-url URL`, `--repo-path PATH` (on remote, default `~/genebeddings`), `--branch BRANCH`, `--skip-setup` (only run wrapper tests; envs must already exist), `--get-report` (scp the report file into the current directory). Repo URL and branch default to the current git origin and branch.

```bash
# Re-run only tests (no setup), fetch report
bash scripts/setup_envs/run_tests_remote.sh --skip-setup --get-report ubuntu@10.0.0.5
```
