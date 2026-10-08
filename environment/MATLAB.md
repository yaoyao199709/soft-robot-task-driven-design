# MATLAB environment

## Requirements

- **MATLAB:** R2022b–R2025a (releases used for the research workflows)
- **Additional MathWorks toolboxes:** None
- **COMSOL Multiphysics:** Not required

The MATLAB workflows process precomputed FEM response data, fit polynomial surrogate models, reconstruct optimized designs, and produce analysis figures. The required `.mat` and `.csv` datasets are included under [`surrogate/data/`](../surrogate/data/).

## Running the workflows

See [`surrogate/README.md`](../surrogate/README.md) for the preprocessing and modeling entry points. Run scripts in MATLAB with the repository's included datasets. Python task simulations use the separate [`environment.yml`](environment.yml) environment.
