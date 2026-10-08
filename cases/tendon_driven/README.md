# Tendon-driven actuator

This case validates the task-level motion of a tendon-driven soft actuator, modeled with a neural surrogate of FEM-derived joint torque response.

## Files

| File | Role |
|---|---|
| `tendon_driven_FEM_data.csv` | FEM-derived joint training data |
| `train_joint_surrogate.py` | Train the joint-torque surrogate and feature scalers |
| `tendon_driven_surrogate_model.tflite` | Pretrained model used by the simulation |
| `x_scaler.joblib`, `y_scaler.joblib` | Pretrained scaling transformations |
| `run_tendon_validation.py` | PyBullet simulation and loading cases |
| `link/` | Geometry meshes for visualization |

A copy of the FEM training data is also catalogued at [`../../surrogate/data/tendon_driven/validation/`](../../surrogate/data/tendon_driven/validation/).

## Usage

From this directory, run `python run_tendon_validation.py` to simulate the selected load case. The `load_mass` parameter controls the tip load in grams (typical cases: 0, 20, or 50).

Run `python train_joint_surrogate.py` only when regenerating model files. It writes scalers and exports a model named `tendon_driven_surrogate.tflite`, whereas the packaged simulation reads `tendon_driven_surrogate_model.tflite`. The training output and the packaged inference model have different filenames. Keep the inference model and its matching scalers together when running validation.

Requires relevant packages in `environment/`, including PyBullet for simulation and TensorFlow / scikit-learn / joblib for model training and inference.


## Model interface and output files

The simulation loads `tendon_driven_surrogate_model.tflite` with `x_scaler.joblib` and `y_scaler.joblib`. The training script currently writes `tendon_driven_surrogate.tflite`, and writes the scaler files under the same directory. These two model names are distinct; do not replace the distributed inference model without checking input/output tensor shapes, scaler compatibility, and the validation results. Train on a separate copy if preserving the packaged models.

The FEM-derived training CSV is also catalogued in `surrogate/data/tendon_driven/`. The simulation uses the packaged `link/` meshes and supports selectable loading conditions.
