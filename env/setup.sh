#!/bin/bash
# ============================================================================
# env/setup.sh — reproducible single-cell + DL environment for the osteoclast
# de novo discovery project.
# Target: Miyabi-G (aarch64 / NVIDIA GH200, CUDA 12.x/13.x driver), PBS node.
# Creates a project-local conda env at $PROJ/env/conda (py3.10) and installs the
# analysis stack. Idempotent & RESILIENT: independent best-effort pip stages so
# one failing source-build cannot abort the rest. Re-running only adds what's
# missing.
#
# Run on a COMPUTE NODE (never the login node), e.g. inside a PBS/Slurm job.
# ============================================================================
set -uo pipefail

# Project root: override with OC_PROJ, else default to this script's parent dir.
PROJ="${OC_PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_PREFIX="$PROJ/env/conda"
PY_VER=3.10
export PIP_CACHE_DIR="$PROJ/env/.pip_cache"
export CONDA_PKGS_DIRS="$PROJ/env/.conda_pkgs"
export HF_HOME="$PROJ/.hf_home"
export TORCH_HOME="$PROJ/.torch_home"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME" "$TORCH_HOME"

echo "===== [setup] host=$(hostname) date=$(date) ====="

module load miniforge3/24.11.0-0 2>/dev/null || true
CONDA_BASE="${CONDA_BASE:-/work/share/miniforge3-aarch64}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  echo "[setup] creating conda env at $ENV_PREFIX (python=$PY_VER)"
  conda create -y -p "$ENV_PREFIX" "python=$PY_VER" pip || { echo "CONDA_CREATE_FAIL"; exit 1; }
else
  echo "[setup] reusing existing env at $ENV_PREFIX"
fi
conda activate "$ENV_PREFIX"
python --version; which python pip

pip_try () { echo "[setup] pip install $*"; pip install "$@" || echo "[setup] WARN: failed: $*"; }

# --- PyTorch (aarch64 + CUDA) ------------------------------------------------
if ! python -c "import torch" 2>/dev/null; then
  pip install --upgrade pip
  pip install torch --index-url https://download.pytorch.org/whl/cu126 \
    || pip install torch --index-url https://download.pytorch.org/whl/cu124 \
    || pip install torch
fi

# --- core scientific + single-cell stack (independent stages) ----------------
pip_try "numpy<2.0" scipy pandas "scikit-learn" statsmodels scikit-misc
pip_try matplotlib seaborn tqdm pyyaml h5py pyarrow joblib openpyxl
pip_try anndata scanpy leidenalg igraph python-igraph
pip_try harmonypy scrublet
pip_try scvi-tools                       # VAE integration / scANVI label transfer
pip_try scvelo cellrank                  # RNA velocity / fate
pip_try decoupler omnipath gseapy        # TF/pathway activity + enrichment
pip_try cellxgene-census                 # programmatic human bone/marrow data
pip_try squidpy                          # spatial (if spatial data used)
pip_try pertpy                           # perturbation / scGen / augur utilities

# --- GRN + in-silico perturbation (the discovery engine; best-effort) --------
# pyscenic = SCENIC regulons; arboreto/ctxcore are its deps.
pip_try arboreto ctxcore pyscenic
# CellOracle = GRN + TF KO simulation (heaviest deps; may fail on aarch64).
pip_try celloracle

# --- verification ------------------------------------------------------------
echo "===== [setup] VERIFY ====="
python - <<'PY'
import importlib
mods = ["torch","numpy","scipy","pandas","sklearn","matplotlib","anndata","scanpy",
        "harmonypy","scrublet","scvi","scvelo","cellrank","decoupler","gseapy",
        "cellxgene_census","squidpy","pyscenic","celloracle"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m:20s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"MISS {m:20s} {type(e).__name__}")
import torch
print("torch.cuda.is_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
echo "===== [setup] DONE ====="; date
