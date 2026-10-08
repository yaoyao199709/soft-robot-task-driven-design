# Generalized Task-Driven Design of Soft Robots

Research code, precomputed FEM data, pretrained models, and project website for the paper published in *Advanced Intelligent Systems*.

[Project website](website/) · [Journal article](https://doi.org/10.1002/aisy.70406) · [arXiv](https://arxiv.org/abs/2603.19794)

## Overview

The framework uses FEM-derived actuator response data to construct efficient surrogate models and design-conditioned meta-models, then incorporates them into task-level simulation and design optimization.

**Workflow:** FEM-derived data → surrogate modeling → task simulation → task evaluation → optimized design.

## Repository map

| Directory | Contents |
|---|---|
| [`surrogate/`](surrogate/) | Precomputed FEM data, MATLAB processing and surrogate analyses |
| [`cases/`](cases/) | Simulation, learning, validation and optimization workflows |
| [`cases/gripper/`](cases/gripper/) | Grasp validation and PPO-based morphology/control co-design |
| [`cases/helical/`](cases/helical/) | Helical-actuator validation and shape-matching optimization |
| [`cases/tendon_driven/`](cases/tendon_driven/) | Tendon-driven actuator simulation |
| [`environment/`](environment/) | Python and MATLAB software requirements |
| [`website/`](website/) | Research project website |

## Workflow documentation

Read [surrogate/README.md](surrogate/README.md) to understand the data and model-generation layer, then [cases/README.md](cases/README.md) for the task-level workflows and their inputs. Case-specific commands and configuration details are documented inside each case directory.

For the gripper, the MATLAB coefficient dataset is stored in `surrogate/data/gripper/design_sweep/` and the Python meta-model training code in `cases/gripper/RL/` uses the corresponding coefficient data packaged with that workflow. The baseline grasp simulator and the RL simulator are separate implementations with distinct purposes.

## Environments and reproducibility

The Python environment specification and MATLAB requirements are under [`environment/`](environment/); see individual workflows for their additional dependencies. MATLAB is used for FEM-response processing and analysis; COMSOL model files and FEM-generation scripts are not distributed. Dataset-driven surrogate and task simulation workflows do not require COMSOL.

Training scripts and some simulation utilities write files to disk. Run write-producing workflows in a disposable working copy if you need to preserve packaged models, URDF files, and results. Check input paths and required model assets before running each workflow.

## Validation scope

Automated GitHub Actions checks validate Python source syntax and repository resources, installation of the documented Linux/CPU environment, import of key dependencies, TFLite model loading and sample inference, packaged scalers, Gripper PPO checkpoint loading and deterministic policy prediction, and headless PyBullet mesh loading and physics stepping.

These lightweight checks do not constitute full reproduction of the paper's numerical results or end-to-end execution of all Gripper, Helical, and Tendon-driven training, validation, and optimization workflows. MATLAB scripts were not executed in GitHub Actions. See [the automated checks](.github/workflows/repository-checks.yml) and [smoke tests](tests/check_research_smoke.py).

## Citation

Use the [journal DOI](https://doi.org/10.1002/aisy.70406) when citing this work.

## Technical documentation

- [Data and model interfaces](docs/DATA_MODEL_INTERFACES.md)
- [MATLAB requirements](environment/MATLAB.md)
- [Code navigation](docs/CODE_NAVIGATION.md)

## Environments

See [`environment/README.md`](environment/README.md) for the Python dependency specification, MATLAB prerequisites, and runtime compatibility notes.
