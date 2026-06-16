# Osteoclast convergent-program analysis

Code and figure outputs for a single-cell analysis of a convergent invasive-resorptive
osteoclast program across bone tumors.

This repository is organized for manuscript review and release on GitHub. It contains
the analysis scripts, environment notes, and manuscript figure exports. It does not
include raw data, large intermediate `.h5ad` objects, manuscript documents, or local
cluster outputs.

## Scientific Scope

The pipeline supports the following analysis chain:

1. Public-data discovery and osteoclast-content checks from GEO and CELLxGENE Census.
2. Single-cell atlas construction, scVI integration, and strict mature-osteoclast annotation.
3. Ambient-deconfounded discovery of a 74-gene tumor osteoclast program.
4. Robustness, specificity, doublet, adult-reference, and external-validation controls.
5. In-silico transcription-factor perturbation with a Ridge-GRN signal-propagation model.
6. Downstream signaling, pathway-activity, trajectory, and publication-figure generation.

Public datasets referenced by the code include `GSE266330`, `GSE254672`, `GSE168664`,
`GSE162454`, `GSE152048`, `GSE268835`, `GSE212341`, `GSE143791`, and CELLxGENE Census
osteoclast reference cells.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── CODE_AVAILABILITY.md
├── DATA_AVAILABILITY.md
├── REPRODUCIBILITY.md
├── LICENSE
├── CITATION.cff
├── env/
│   └── setup.sh
├── data_discovery/
│   ├── census_osteoclast.py
│   └── verify_oc_content.py
├── data_engineering/
│   ├── build_master.py
│   └── integrate_annotate.py
├── discovery/
│   ├── convergence_discovery_v2.py
│   └── robust_and_driver.py
├── modeling/
│   └── insilico_perturb.py
├── validation/
│   ├── supp_compartment.py
│   ├── supp_external.py
│   ├── correction_adult_ref.py
│   ├── doublet_control.py
│   ├── reanalysis_harden.py
│   └── build_evidence_matrix.py
├── advanced/
│   ├── run_adv_A.py
│   ├── run_adv_B.py
│   ├── adv_helpers.py
│   └── make_figs_v2.py
├── report/
│   ├── pub_style.py
│   └── tables/
└── figures/
    ├── main/
    └── supplementary/
```

## Scripts

| Stage | Script | Purpose |
|---|---|---|
| Data discovery | `data_discovery/census_osteoclast.py` | Pull human osteoclast reference cells from CELLxGENE Census. |
| Data discovery | `data_discovery/verify_oc_content.py` | Check raw integer counts and mature-osteoclast marker content for GEO 10x inputs. |
| Atlas build | `data_engineering/build_master.py` | Load cohorts, run per-sample QC, harmonize Ensembl IDs, and build `master_raw.h5ad`. |
| Atlas build | `data_engineering/integrate_annotate.py` | Run scVI integration and strict osteoclast-lineage annotation. |
| Discovery | `discovery/convergence_discovery_v2.py` | Discover the ambient-deconfounded convergent tumor-osteoclast program. |
| Discovery | `discovery/robust_and_driver.py` | Run robustness summaries and GRNBoost2-based driver nomination. |
| Modeling | `modeling/insilico_perturb.py` | Run Ridge-GRN in-silico transcription-factor knockouts. |
| Validation | `validation/supp_compartment.py` | Quantify compartment specificity and module behavior. |
| Validation | `validation/supp_external.py` | Run independent osteosarcoma and prostate bone-metastasis validation checks. |
| Validation | `validation/correction_adult_ref.py` | Run adult-reference, depth, and maturation controls. |
| Validation | `validation/doublet_control.py` | Test robustness after scrublet doublet control and specific-core scoring. |
| Validation | `validation/reanalysis_harden.py` | Build bootstrap and confound-hardening summaries. |
| Validation | `validation/build_evidence_matrix.py` | Build the 74-gene evidence matrix under `report/tables/`. |
| Advanced | `advanced/run_adv_A.py` | Run LIANA, decoupler/DoRothEA, PROGENy, and GSEA summaries. |
| Advanced | `advanced/run_adv_B.py` | Run PAGA, diffusion pseudotime, and pseudotime gene heatmap summaries. |
| Figures | `advanced/make_figs_v2.py` | Render main figure panels into `figures/main/`. |

## Environment

The reference environment targets Python 3.10 on a Linux/HPC machine with CUDA-capable
PyTorch. The `env/setup.sh` installer is written for a project-local conda environment
and can be adapted by changing `CONDA_BASE` and the PyTorch wheel index.

```bash
export OC_PROJ=/path/to/Osteoclast
bash env/setup.sh
```

For a generic Python environment, install PyTorch first from the matching CUDA or CPU
wheel index, then install the remaining dependencies:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

`celloracle` is not required. The in-silico perturbation step uses the Ridge-GRN
implementation in `modeling/insilico_perturb.py`.

## Runtime Paths

Scripts resolve the project root from `OC_PROJ` when set. If `OC_PROJ` is not set, each
script falls back to the repository root inferred from its own file path.

Common variables:

| Variable | Meaning | Default |
|---|---|---|
| `OC_PROJ` | Project/data root containing the repository layout | repository root |
| `OC_DOWNLOADS` | Raw download directory | `$OC_PROJ/downloads` |
| `CONDA_BASE` | Conda installation root used by `env/setup.sh` | `/work/share/miniforge3-aarch64` |

## Run Order

The full analysis requires the public raw data and substantial compute. Run the scripts
from the repository root or with `OC_PROJ` set to a data root that mirrors this layout.

```bash
# 1. Data discovery and input checks
python data_discovery/census_osteoclast.py
python data_discovery/verify_oc_content.py <accession> <extracted_dir> <verification_dir>

# 2. Atlas construction
python data_engineering/build_master.py
python data_engineering/integrate_annotate.py

# 3. Program discovery and driver analysis
python discovery/convergence_discovery_v2.py
python discovery/robust_and_driver.py

# 4. In-silico perturbation model
python modeling/insilico_perturb.py

# 5. Validation and controls
python validation/supp_compartment.py
python validation/supp_external.py
python validation/correction_adult_ref.py
python validation/doublet_control.py
python validation/reanalysis_harden.py
python validation/build_evidence_matrix.py

# 6. Advanced downstream summaries
python advanced/run_adv_A.py
python advanced/run_adv_B.py

# 7. Main figures
python advanced/make_figs_v2.py
```

## Included Figure Exports

| File stem | Paper figure | Content |
|---|---|---|
| `figures/main/G1_atlas` | Fig. 1 | Osteoclast-rich pan-tumor atlas, QC, and lineage summaries. |
| `figures/main/G2_deconfounding` | Fig. 2 | Ambient deconfounding and convergent-program definition. |
| `figures/main/G3_program` | Fig. 3 | The 74-gene program and module structure. |
| `figures/main/G4_specificity` | Fig. 4 | Compartment and module specificity. |
| `figures/main/G5_communication` | Fig. 5 | Cell-cell communication summaries. |
| `figures/main/G6_regulation_trajectory` | Fig. 6 | TF/pathway activity, trajectory, and driver analysis. |
| `figures/main/G7_validation` | Fig. 7 | Held-out and external validation. |
| `figures/supplementary/F6_translation_novelty` | Supplementary figure | Translation/novelty summary panel exported with the manuscript figures. |

PDF exports are tracked in this repository. PNG previews may be generated locally from the
same figure scripts but are not tracked to keep the GitHub repository lightweight.

## Notes for Reviewers

- The repository intentionally excludes raw downloads, generated `.h5ad` files, large
  intermediate matrices, and manuscript documents.
- The currently included code is intended to document and reproduce the computational
  workflow when the referenced public datasets are downloaded into the expected layout.
- The figure-generation script depends on upstream generated CSV/JSON/H5AD artifacts; it
  is not a standalone plotting demo.
- A lightweight repository smoke check is available through GitHub Actions and can also be
  run locally with `python scripts/check_repository.py`.

## Citation and License

If you reuse this code, cite the accompanying manuscript. Citation metadata is provided in
`CITATION.cff`; update author and DOI fields after publication if needed.

The code is released under the MIT License unless replaced by a journal- or institution-
specific license before public release.
