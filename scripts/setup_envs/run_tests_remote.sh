#!/usr/bin/env bash
# Run env setup + wrapper tests on a remote GPU host (e.g. Lambda Labs instance).
# SSHs to the host, clones/pulls the repo, runs test_all_envs.sh, optionally fetches the report.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_NAME="test_all_envs_report.txt"

usage() {
  echo "Usage: $0 [OPTIONS] SSH_TARGET"
  echo ""
  echo "  SSH_TARGET    ssh target (e.g. ubuntu@1.2.3.4 or lambda-instance)"
  echo ""
  echo "Options:"
  echo "  --repo-url URL    Clone URL (default: current git origin or github.com/nicolasalynn/genebedding.git)"
  echo "  --repo-path PATH  Path on remote (default: ~/genebeddings)"
  echo "  --branch BRANCH   Branch to checkout (default: current branch or epistasis_framework_upgrade)"
  echo "  --skip-setup      Only run wrapper tests; do not run setup scripts (envs must exist)"
  echo "  --get-report      After tests, scp report file into current directory"
  echo "  -h, --help        Show this help"
  echo ""
  echo "Examples:"
  echo "  $0 ubuntu@10.0.0.5"
  echo "  $0 --skip-setup --get-report ubuntu@10.0.0.5"
  echo "  $0 --branch main lambda  # if 'lambda' is a Host in ~/.ssh/config"
  exit 0
}

REPO_URL=""
REPO_PATH="~/genebeddings"
BRANCH=""
SKIP_SETUP=""
GET_REPORT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --repo-url)   REPO_URL="$2";  shift 2 ;;
    --repo-path)  REPO_PATH="$2"; shift 2 ;;
    --branch)     BRANCH="$2";    shift 2 ;;
    --skip-setup) SKIP_SETUP="--skip-setup"; shift ;;
    --get-report) GET_REPORT=1; shift ;;
    -h|--help)    usage ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [ $# -lt 1 ]; then
  echo "Error: SSH_TARGET required" >&2
  usage
fi
SSH_TARGET="$1"

if [ -z "$REPO_URL" ]; then
  REPO_URL=$(git -C "$SCRIPT_DIR/../.." remote get-url origin 2>/dev/null) || true
  if [ -z "$REPO_URL" ]; then
    REPO_URL="https://github.com/nicolasalynn/genebedding.git"
  fi
  # Normalize to HTTPS for clone if it's SSH
  REPO_URL="${REPO_URL//git@github.com:/https://github.com/}"
  REPO_URL="${REPO_URL%.git}.git"
fi

if [ -z "$BRANCH" ]; then
  BRANCH=$(git -C "$SCRIPT_DIR/../.." branch --show-current 2>/dev/null) || true
  [ -z "$BRANCH" ] && BRANCH="epistasis_framework_upgrade"
fi

echo "=============================================="
echo "  Remote test: $SSH_TARGET"
echo "  Repo: $REPO_URL"
echo "  Path: $REPO_PATH   Branch: $BRANCH"
echo "  Skip setup: ${SKIP_SETUP:-no}   Get report: ${GET_REPORT:-no}"
echo "=============================================="

# Use a path the remote can expand (avoid literal ~ in single-quoted cd)
# Default: $HOME/genebeddings so ~ expands on remote
if [ "$REPO_PATH" = "~/genebeddings" ]; then
  REMOTE_REPO_PATH='$HOME/genebeddings'
else
  REMOTE_REPO_PATH=$(printf '%s' "$REPO_PATH" | sed "s/'/'\\\\''/g")
fi
REMOTE_REPORT_PATH="${REMOTE_REPO_PATH}/scripts/setup_envs/${REPORT_NAME}"
# For scp we need a path the remote can expand; avoid $HOME so local shell does not expand it
if [ "$REPO_PATH" = "~/genebeddings" ]; then
  SCP_REPORT_PATH="~/genebeddings/scripts/setup_envs/${REPORT_NAME}"
else
  SCP_REPORT_PATH="${REMOTE_REPORT_PATH}"
fi
RU=$(printf '%s' "$REPO_URL" | sed "s/'/'\\\\''/g")

# Build remote commands: clone or pull, source conda, load HF token, then run test script.
REMOTE_SCRIPT="set -e
REPO_PATH=$REMOTE_REPO_PATH
if [ -f ~/.hf_token ]; then export HF_TOKEN=\$(cat ~/.hf_token); export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN; fi
if [ ! -d \"\$REPO_PATH\" ]; then git clone '$RU' \"\$REPO_PATH\"; fi
cd \"\$REPO_PATH\" && git fetch origin && git checkout $BRANCH && git pull origin $BRANCH
for c in ~/miniconda3 ~/anaconda3 /opt/conda; do
  [ -f \"\$c/etc/profile.d/conda.sh\" ] && . \"\$c/etc/profile.d/conda.sh\" && break
done
if ! command -v conda &>/dev/null; then
  echo 'Conda not found; installing Miniconda to ~/miniconda3 ...'
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh || true
  if [ -f /tmp/miniconda.sh ]; then
    bash /tmp/miniconda.sh -b -p \"\$HOME/miniconda3\"
    . \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
    conda config --set allow_conda_downgrades true 2>/dev/null || true
    rm -f /tmp/miniconda.sh
    echo 'Miniconda installed.'
  else
    echo 'Failed to download Miniconda.' >&2; exit 1
  fi
fi
# Accept Conda ToS so conda create works non-interactively (new Miniconda default)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
# When --skip-setup: refresh editable install and env-specific deps in each existing env
if [ -n \"$SKIP_SETUP\" ]; then
  for env in nt alphagenome evo2 spliceai convnova borzoi mutbert hyenadna caduceus dnabert rinalmo specieslm genebeddings_main; do
    if conda activate \"\$env\" 2>/dev/null; then
      echo \"Refreshing \$env ...\"
      pip install --upgrade setuptools wheel -q 2>/dev/null || true
      if [ \"\$env\" = alphagenome ]; then
        pip install . -q 2>/dev/null || true
      else
        pip install -e . -q 2>/dev/null || true
      fi
      case \"\$env\" in
        evo2) conda install -c conda-forge transformer-engine-torch=2.3.0 -y -q 2>/dev/null || true; pip install flash-attn==2.8.0.post2 --no-build-isolation -q 2>/dev/null || true; pip install evo2 -q 2>/dev/null || true ;;
        rinalmo) pip install \"git+https://github.com/lbcb-sci/RiNALMo.git\" flash-attn==2.3.2 -q 2>/dev/null || true ;;
        borzoi) pip install \"https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl\" -q 2>/dev/null || pip install \"https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl\" -q 2>/dev/null || pip install flash-attn -q 2>/dev/null || true ;;
        caduceus) pip install mamba_ssm -q 2>/dev/null || true ;;
        dnabert) pip install einops -q 2>/dev/null || true ;;
        specieslm) pip install \"torch>=2.6\" -q 2>/dev/null || true ;;
        mutbert) pip install \"transformers>=4.30,<4.46\" -q 2>/dev/null || true ;;
        genebeddings_main) pip install transformers omegaconf rinalmo borzoi-pytorch spliceai-pytorch -q 2>/dev/null || true ;;
      esac
      conda deactivate 2>/dev/null || true
    fi
  done
fi
# Run in same shell so conda stays in PATH
. scripts/setup_envs/test_all_envs.sh $SKIP_SETUP
echo ''
echo \"Report at: \$REPO_PATH/scripts/setup_envs/$REPORT_NAME\"
"

ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 "$SSH_TARGET" "$REMOTE_SCRIPT"

if [ -n "$GET_REPORT" ]; then
  local_report="./${REPORT_NAME}"
  scp -o StrictHostKeyChecking=accept-new "$SSH_TARGET:$SCP_REPORT_PATH" "$local_report" 2>/dev/null || {
    echo "Warning: could not scp report. Try: scp $SSH_TARGET:$SCP_REPORT_PATH ." >&2
    exit 0
  }
  echo "Report saved to $local_report"
  # Fetch setup failure logs for debugging (last 100 lines per failed env)
  scp -o StrictHostKeyChecking=accept-new -r "$SSH_TARGET:~/genebeddings/scripts/setup_envs/setup_logs" ./ 2>/dev/null || true
  [ -d ./setup_logs ] && [ -n "$(ls -A ./setup_logs 2>/dev/null)" ] && echo "Setup failure logs in ./setup_logs/"
fi
