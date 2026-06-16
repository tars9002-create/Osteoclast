# Reproducibility Notes

## Scope

The repository is designed as a manuscript code record. It documents the end-to-end analysis
workflow and includes final figure exports, but it does not bundle raw data or large generated
single-cell objects.

## Expected Compute

The full workflow is computationally heavy. It was written for a Linux/HPC environment with
Python 3.10, a project-local conda environment, and CUDA-capable PyTorch. Some steps require
substantial memory and GPU resources because they build and integrate a multi-cohort single-cell
atlas.

## Minimal Repository Check

The lightweight check does not require single-cell dependencies:

```bash
python scripts/check_repository.py
find . -name '*.py' -print0 | xargs -0 python -m py_compile
```

This validates repository structure, obvious identity/path leaks, large-file mistakes, and
Python syntax. It does not prove that the full biological pipeline can run without the public
datasets and the expected compute environment.

## Full Pipeline

Use the run order in `README.md`. The essential setup is:

```bash
export OC_PROJ=/path/to/Osteoclast
export OC_DOWNLOADS=$OC_PROJ/downloads
bash env/setup.sh
```

Then run each stage in order from data discovery through figure generation. The main figure
script writes PNG and PDF outputs to `figures/main/`.

## Randomness

Scripts set fixed random seeds where the analysis uses stochastic methods. Exact numeric
reproduction may still depend on library versions, BLAS/CUDA behavior, and hardware.
