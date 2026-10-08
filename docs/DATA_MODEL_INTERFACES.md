# Data and model interfaces

The repository starts from **precomputed FEM-derived datasets**. COMSOL project files and FEM generation scripts are not distributed. The case-specific Python modules use the supplied model weights, scalers and geometry.

## Gripper

- `surrogate/data/gripper/` stores FEM-derived MATLAB and CSV data, including polynomial coefficients and design-sweep datasets.
- `surrogate/matlab/gripper/` documents processing and model analysis.
- `cases/gripper/Sim2Real/` runs grasp simulation; `cases/gripper/RL/` contains the meta-model, PPO environment, training and evaluation.

## Helical

- `surrogate/data/helical/design_sweep/sm_dataset.mat` is the MATLAB input for `surrogate/matlab/helical/preprocessing/SMDatasetReconstruct.m`.
- The design-sweep CSVs feed `cases/helical/Shape-matching_opt/optimization/train_shape_meta_models.py`. Already processed copies are shipped under that case's `data/` folder.
- The optimizer loads `models_x/`, `models_y/`, `models_z/`, each containing a `.tflite` model and x/y normalization arrays.
- `surrogate/data/helical/optimized_design/sm_result.mat` is the input for `SMReconstruct_results.m`. Its generated CSVs are written to `surrogate/matlab/helical/preprocessing/outputs/` to protect supplied datasets.
- `cases/helical/Shape-matching_opt/result/` contains design-specific surrogate training and shape evaluation.
- `cases/helical/Sim2Real/surrogate_models/` provides the per-axis validation models and scalers.

## Tendon-driven

- `surrogate/data/tendon_driven/` holds FEM-derived training data; a corresponding CSV is provided within `cases/tendon_driven/`.
- `train_joint_surrogate.py` exports `tendon_driven_surrogate.tflite`, whereas `run_tendon_validation.py` reads the supplied `tendon_driven_surrogate_model.tflite`.
- These two model files must not be treated as interchangeable without checking tensor shapes and scaler compatibility. Training may overwrite scaler files; preserve distributed inference assets before training.

## Safe execution

Training, optimization and simulation can create or overwrite models, CSV files, logs and URDFs. Use a working copy for experiments intended to preserve the packaged results. Keep pretrained model and scaler files together when evaluating supplied cases.
