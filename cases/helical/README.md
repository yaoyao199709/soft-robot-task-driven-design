# Helical actuator

The helical actuator supports **sim-to-real validation** under gravity and tip loading, plus **shape-matching design optimization** using design-conditioned surrogate meta-models.

## Structure and workflow

| Location | Purpose |
|---|---|
| `mesh/` | Visualization geometry for segment and module assemblies |
| `Sim2Real/surrogate_models/` | Axis-wise FEM data, pretrained TFLite models, scalers, and training code |
| `Sim2Real/run_helical_validation.py` | PyBullet validation with gravity and selected tip loads |
| `Shape-matching_opt/optimization/data/` | FEM response data across candidate designs |
| `Shape-matching_opt/optimization/train_shape_meta_models.py` | Train design-conditioned axis-wise behavior models |
| `Shape-matching_opt/optimization/optimize_shape_cmaes.py` | CMA-ES search for designs matching a target helix |
| `Shape-matching_opt/result/data/` | FEM data for selected optimized designs |
| `Shape-matching_opt/result/train_optimized_surrogates.py` | Fit design-specific surrogates |
| `Shape-matching_opt/result/evaluate_optimized_shapes.py` | Simulate and compare resulting shapes |

## Data sources

The upstream MATLAB datasets and reconstruction code are organized under [`../../surrogate/data/helical/`](../../surrogate/data/helical/) and [`../../surrogate/matlab/helical/`](../../surrogate/matlab/helical/). The preprocessed CSVs supplied in this case are used directly by the training scripts. COMSOL models are not required to run the provided surrogate-based tasks.

## Running the workflows

The relative commands below are intended to be run from `cases/helical/`. From the repository root, prefix each script path with `cases/helical/`. Dependencies include `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`, `joblib`, `pybullet`, and `cma` as appropriate to each step; see the repository's `environment/` directory.

1. **Validation:** `python Sim2Real/run_helical_validation.py`. Adjust the prescribed load in `external_load_fns`. The gravity-only mode and tip-load modes use different step counts and a code-section toggle; inspect the script's marked configuration before launching.
2. **Meta-model training:** `python Shape-matching_opt/optimization/train_shape_meta_models.py` (one axis/configuration per run, as defined near the top of the file).
3. **Optimization:** `python Shape-matching_opt/optimization/optimize_shape_cmaes.py`.
4. **Optimized-design reconstruction:** `python Shape-matching_opt/result/train_optimized_surrogates.py` followed by `python Shape-matching_opt/result/evaluate_optimized_shapes.py`.

Training and optimization may overwrite model files or logs; run in a copy of the repository when preserving packaged results matters. The packaged models can be used without retraining.

## Modeling notes

The validation implementation loads learned per-axis actuator response models. Shape matching additionally conditions surrogate response on design parameters, searches over segment arrangements and design variables, and validates selected geometries using reconstructed actuator models. The `mesh/` files are visual representations and do not replace surrogate physics.


## Surrogate data and model interface

The task simulation loads per-axis (x, y, z) neural surrogate models for two model segments from `Sim2Real/surrogate_models/`. Each TFLite model is paired with the corresponding input and output `.joblib` scalers; preserve the axis and segment suffixes when replacing assets.

The shape-matching optimizer uses `Shape-matching_opt/optimization/data/` (design parameters and x/y/z response matrices) and `models_x/`, `models_y/`, `models_z/` (TFLite meta-models and `.npy` normalization arrays). The result evaluation loads the design-specific models from `Shape-matching_opt/result/opt_models/` and normalization arrays from `opt_scalers/`. FEM-derived datasets and MATLAB processing are documented in [`../../surrogate/README.md`](../../surrogate/README.md).

The validation script exports `helical_end_pos.csv` and `helix_position.csv`; run in a separate working directory when retaining existing outputs. Training or optimization may update model files and logs.
