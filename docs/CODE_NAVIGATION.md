# Code navigation

## Shape-matching optimizer

`cases/helical/Shape-matching_opt/optimization/optimize_shape_cmaes.py` is the CMA-ES and simulation entry point (1046 source lines). It is intentionally maintained as one entry point to preserve its stateful initialization, model loading and optimization behavior.

Its imports include: `cma`, `joblib`, `logging`, `math`, `matplotlib`, `numpy`, `os`, `pandas`, `pathlib`, `pybullet`, `pybullet_data`, `tensorflow`, `time`, `warnings`.

Key definitions (line numbers refer to this repository revision):

- `import_tensorflow` — lines 22–34
- `_load_axis_model` — lines 58–88
- `quaternion_difference` — lines 110–112
- `quaternion_to_euler` — lines 115–117
- `wrap_angle` — lines 120–122
- `quaternion_rotation_batch` — lines 125–141
- `compute_module_mass` — lines 143–171
- `compute_segment_phis` — lines 173–180
- `build_soft_robot` — lines 183–384
- `torque_model` — lines 386–417
- `run_simulation` — lines 420–549
- `sample_target_helix_by_arclength` — lines 551–577
- `shape_matching_loss_true_arclength` — lines 580–606
- `denorm` — lines 622–623
- `decode_theta_with_pressure_normalized` — lines 625–671
- `init_log` — lines 676–683
- `log_attempt` — lines 684–712
- `compute_target_helix_length` — lines 715–717
- `infer_total_module_range` — lines 720–736
- `partition_total_into_4` — lines 738–751
- `generate_module_number_structures` — lines 755–806
- `optimize_geometry_cmaes` — lines 812–972
- `fitness` — lines 827–934
- `full_design_optimization` — lines 978–1029

The optimizer resolves axis-specific TFLite models and normalization arrays relative to the script location, but also calls `os.chdir()` at import time. This working-directory side effect is part of the current execution behavior and should be considered before turning the script into an importable library. The included output and model files should be treated as reproducibility assets.

## Relationship to MATLAB

The MATLAB helical preprocessing scripts reconstruct x/y/z response tables. These datasets support the design-conditioned surrogate-training path; the already exported files are stored under `surrogate/data/helical/` and relevant case directories. Follow `DATA_MODEL_INTERFACES.md` when replacing any trained artifact.
