# Debugging wrapper setup and tests

Goal: **get every wrapper (nt, alphagenome, evo2, spliceai, convnova, borzoi, mutbert, hyenadna, caduceus, dnabert, rinalmo, specieslm) running** on your Lambda (or other GPU) host.

When a setup fails, the test run now writes the **last 100 lines** of that setup script’s output to `scripts/setup_envs/setup_logs/setup_<env>_last100.txt`. Use that (and the checklist below) to fix failures.

---

## 1. One-time checks on the Lambda host

**SSH in and confirm:**

| Check | Command | What we need |
|-------|---------|--------------|
| HuggingFace token (for NT, AlphaGenome, gated models) | `cat ~/.hf_token` | A file with a single line: your **valid** HF token (create at huggingface.co/settings/tokens). If invalid, replace it. `test_all_envs.sh` auto-loads it when present. |
| git-lfs (for HyenaDNA and some HF models) | `git lfs version` | If missing: `sudo apt-get update && sudo apt-get install -y git-lfs && git lfs install` |
| GPU visible | `nvidia-smi` | At least one GPU listed |
| Conda | `conda --version` | e.g. 24.x or 25.x |
| Repo and branch | `cd ~/genebeddings && git branch && git log -1 --oneline` | Branch should match what you push (e.g. `epistasis_framework_upgrade`). **Push your latest changes before running the remote test** so Lambda gets the fixes. |

---

## 2. How to run and what to send back

**From your laptop (repo root):**

```bash
# Full run: create/update all envs, run all wrapper tests, get report + failure logs
bash scripts/setup_envs/run_tests_remote.sh --get-report ubuntu@64.181.228.169
```

- Use your actual Lambda IP/host instead of `64.181.228.169` if different.
- When it finishes you’ll have:
  - `./test_all_envs_report.txt` – which envs/wrappers passed or failed
  - `./setup_logs/` – for each **failed** setup, `setup_<env>_last100.txt` with the last 100 lines of that setup script

**What to send back for debugging:**

1. **Full report:** contents of `test_all_envs_report.txt`.
2. **For each env that shows `SETUP_FAIL`:** paste the contents of the corresponding `setup_logs/setup_<env>_last100.txt` (or at least the **last 30–40 lines**, where the actual error usually is).
3. **For any wrapper that is not `SETUP_FAIL` but fails the test (e.g. `FAIL` with an exception):** paste the DETAIL column from the report (that’s the exception message).

That’s enough to fix the remaining failures.

---

## 3. Per-wrapper checklist (what’s unclear / what to try)

Use this when a wrapper is failing: run the suggested command **on the Lambda host** (after `ssh ubuntu@...` and `cd ~/genebeddings`), then paste the output.

| Wrapper | If it fails, run this on Lambda and paste output | Notes |
|---------|---------------------------------------------------|--------|
| **nt** | `conda activate nt && python -c "import os; from huggingface_hub import login; login(token=os.environ.get('HF_TOKEN')); from genebeddings.wrappers import NTWrapper; w=NTWrapper(model='nt2500_multi'); print(w.embed('ACGT'*50, pool='mean').shape)"` | Confirms HF token and 2.5b model load. If 401/403, token or gated access. |
| **alphagenome** | `conda activate alphagenome && python -c "from genebeddings.wrappers import AlphaGenomeWrapper; w=AlphaGenomeWrapper(model_version='fold_0', source='huggingface'); print(w.embed('ACGT'*100, pool='mean').shape)"` | If “No module named 'genebeddings'”, the non-editable install didn’t run; check that setup uses `pip install .` in this env. |
| **evo2** | `conda activate evo2 && python -c "import evo2; from evo2 import Evo2; m=Evo2('evo2_7b'); print('evo2 OK')"` | If “transformer-engine” or “empty meta package”, install Transformer Engine + flash-attn per Evo2 README before evo2. |
| **rinalmo** | `conda activate rinalmo && python -c "from rinalmo.pretrained import get_pretrained_model; m,a=get_pretrained_model('giga-v1'); print('rinalmo OK')"` | If “No module named 'rinalmo'”, install from GitHub (clone + `pip install .` + flash-attn==2.3.2). |
| **borzoi** | `conda activate borzoi && python -c "import torch; print(torch.__version__); import torch.library; getattr(torch.library, 'wrap_triton', None)"` | If `wrap_triton` is missing, PyTorch is too old for the flash-attn wheel; we need a compatible PyTorch/triton combo. |
| **caduceus** | `conda activate caduceus && python -c "import mamba_ssm; print('mamba_ssm OK')"` | If “selective_scan_cuda” or undefined symbol, mamba_ssm wasn’t built for this PyTorch/CUDA; rebuild or match versions. |
| **dnabert** | `conda activate dnabert && python -c "from transformers import BertConfig; c=BertConfig(); getattr(c,'is_decoder',None)"` | If `is_decoder` missing, pin `transformers<4.46` (or whatever version still has it). |
| **hyenadna** | `conda activate hyenadna && python -c "from genebeddings.wrappers import HyenaDNAWrapper; w=HyenaDNAWrapper(model='medium-160k'); print(w.embed('ACGT'*50, pool='mean').shape)"` | Paste the full traceback if it fails. |
| **specieslm** | `conda activate specieslm && python -c "from genebeddings.wrappers import SpeciesLMWrapper; w=SpeciesLMWrapper(); print(w.embed('ACGT'*50, pool='mean').shape)"` | Same: paste full traceback. |
| **spliceai** | Set `OPENSPLICEAI_MODEL_DIR` to the dir with OpenSpliceAI checkpoints, or leave unset to SKIP (not a failure). | No extra command; either you have the model dir or you skip. |
| **convnova** | Already often OK with random weights; if you want real weights, add `genebeddings/assets/convnova/convnova.yaml` and `last.backbone.pth`. | Optional. |

---

## 4. What we need from you (summary)

1. **Push** your latest genebeddings (or dlm_wrappers) changes so the remote run uses the current scripts.
2. **Run** once:  
   `bash scripts/setup_envs/run_tests_remote.sh --get-report ubuntu@YOUR_LAMBDA_IP`
3. **Send:**
   - `test_all_envs_report.txt`
   - For every `SETUP_FAIL`: the last 30–40 lines (or full file) of `setup_logs/setup_<env>_last100.txt`
   - For any non–SETUP_FAIL test failure: the FAIL line from the report (DETAIL column)

With that we can fix the remaining setups and get **all** wrappers running.

---

## 5. Quick fixes when you only ran `--skip-setup`

If you ran `test_all_envs.sh --skip-setup` and some envs were never set up, install the missing bits and re-run tests:

```bash
# On Lambda, from repo root (~/genebeddings). Pull latest first: git pull origin epistasis_framework_upgrade

# 1) HF token (required for NT, AlphaGenome). Use a valid token from https://huggingface.co/settings/tokens
echo 'YOUR_HF_TOKEN' > ~/.hf_token && chmod 600 ~/.hf_token

# 2) git-lfs (required for HyenaDNA and some HF models)
sudo apt-get update && sudo apt-get install -y git-lfs && git lfs install

# 3) Missing packages in envs that were never set up
conda activate evo2    && pip install evo2
conda activate specieslm && pip install "transformers>=4.30,<4.46"

# 4) rinalmo: torch must be installed first; flash-attn must build with --no-build-isolation so it sees torch
conda activate rinalmo && pip install "torch>=2.0" --index-url "https://download.pytorch.org/whl/cu121" && \
  pip install flash-attn==2.3.2 --no-build-isolation && \
  pip install "git+https://github.com/lbcb-sci/RiNALMo.git"

# 5) caduceus / dnabert: force older transformers (tie_weights/recompute_mapping and is_decoder removed in 4.46+)
conda activate caduceus && pip install "transformers>=4.30,<4.46" --force-reinstall
conda activate dnabert  && pip install "transformers>=4.30,<4.46" --force-reinstall

# 6) Borzoi needs PyTorch >=2.5 for wrap_triton
conda activate borzoi && pip install "torch>=2.5" --index-url "https://download.pytorch.org/whl/cu121"

# 7) genebeddings_main: install in order (torch first, then flash-attn with --no-build-isolation, then rinalmo)
conda activate genebeddings_main && \
  pip install "torch>=2.5" --index-url "https://download.pytorch.org/whl/cu121" && \
  pip install "transformers>=4.30,<4.46" omegaconf && \
  pip install flash-attn==2.3.2 --no-build-isolation && \
  pip install "git+https://github.com/lbcb-sci/RiNALMo.git" borzoi-pytorch spliceai-pytorch
```

Then run again: `bash scripts/setup_envs/test_all_envs.sh --skip-setup`.

**Or run the single fix script** (does all of the above in order):

```bash
bash scripts/setup_envs/fix_envs_after_skip_setup.sh
```
