# Surrogate modeling and FEM-derived data

This directory contains finite-element response datasets and scripts for data transformation, polynomial model fitting, stiffness/force analysis, and optimization-result reconstruction. COMSOL models and scripts requiring COMSOL model files are not included.

## Structure

```text
data/
  gripper/
    design_sweep/         FEM response dataset and polynomial coefficients
    optimized_design/     FEM response for the optimized gripper
  helical/
    design_sweep/         Shape-matching training data
    optimized_design/     Optimized-design response data
  tendon_driven/
    validation/           Tendon-driven response dataset
matlab/
  gripper/
    preprocessing/       Process FEM response datasets
    model_analysis/       Fit stiffness and actuation response
    result_analysis/      Analyze grasping and optimized-design results
  helical/                Helical analysis and data reconstruction
  translation/            Translation stiffness analysis
```

## Gripper MATLAB workflows

| Script | Purpose | Primary inputs / outputs |
|---|---|---|
| `matlab/gripper/model_analysis/gripper_stiffness.m` | Fit and visualize actuator stiffness and pressure–torque relationships | FEM values embedded in script; figures and fitted parameters in MATLAB session |
| `matlab/gripper/preprocessing/gripper_dataset_process.m` | Convert design-sweep FEM response into polynomial coefficient rows | `data/gripper/design_sweep/gripper_dataset.mat` → `matlab/gripper/preprocessing/outputs/poly_surrogate_coefficient.csv` |
| `matlab/gripper/preprocessing/opt_design_process.m` | Analyze fitted response for the optimized gripper design | `data/gripper/optimized_design/gripper_dataset_opt.mat`; MATLAB workspace/figures |
| `matlab/gripper/result_analysis/grasping_results_plot.m` | Compare grasping validation results | Experimental/simulation values selected in script; figures |
| `matlab/gripper/result_analysis/opt_design_simulation_results.m` | Visualize optimized-gripper simulation results | Scenario values in script; figures |

The packaged `data/gripper/design_sweep/poly_surrogate_coefficient.csv` is the reference coefficient dataset. Running `gripper_dataset_process.m` writes a new copy under its `outputs/` directory, rather than overwriting that packaged dataset. The downstream Python meta-model entry point is [`../cases/gripper/RL/train_surrogate_meta_model.py`](../cases/gripper/RL/train_surrogate_meta_model.py); it uses the corresponding coefficient dataset in its own workflow directory.

## MATLAB requirements

MATLAB is required to execute `.m` scripts. Gripper stiffness and preprocessing scripts use standard `polyfit`/`polyval`, array operations, and MAT file loading. Dependencies and MATLAB release compatibility should be confirmed on the desired platform before execution. Scripts in other actuator sections may have additional requirements; their processing and runtime behavior have not been independently revalidated in this package.

Use the per-case [task simulation guide](../cases/README.md) for model consumption and optimization.

## Helical and tendon-driven data

The helical workflows comprise baseline axis-wise stiffness analyses (`matlab/helical/model_analysis/`) and two data-reconstruction stages (`matlab/helical/preprocessing/`): design-sweep data in `data/helical/design_sweep/` and optimized-design reconstruction data in `data/helical/optimized_design/`. These are consumed by the respective Python workflows under `cases/helical/`. MATLAB scripts use the precomputed `.mat` inputs; the included CSV files provide the inputs used by the Python training workflows.

The tendon-driven case uses `data/tendon_driven/validation/tendon_driven_FEM_data.csv`, with Python training and simulation under `cases/tendon_driven/`.
