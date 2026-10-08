# Soft Gripper

Task-level grasp simulation and reinforcement-learning co-design for a soft robotic gripper.

For FEM response data, stiffness fitting and polynomial coefficient preprocessing, see [Surrogate modeling](../../surrogate/README.md).

## Research workflows

### 1. Grasp simulation and validation

The `Sim2Real/` directory contains the PyBullet gripper model, grasp trial scripts and simulation results. Mesh and URDF resources are located in the sibling `link/`, `objects/` and `urdf/` directories.

| Script or resource | Function |
|---|---|
| `Sim2Real/run_grasp_simulation.py` | Interactive grasping simulation |
| `Sim2Real/run_grasp_trials.py` | Repeated grasp trials and CSV generation |
| `Sim2Real/mid_air_sim_results/` | Grasp-trial result files |
| `urdf/` | Object definitions and mesh-selection utilities |
| `objects/` | Grasped object meshes |
| `link/` | Gripper link meshes |

The grasping-condition parameters are set in the configuration block near the end of the interactive simulation script.

### 2. RL co-design

`RL/` contains surrogate meta-model training, the Gym environment, PPO training and policy evaluation.

| Script or resource | Function |
|---|---|
| `RL/train_surrogate_meta_model.py` | Fit the design-conditioned surrogate meta-model |
| `RL/gripper_simulation_rl.py` | Simulator used by the RL environment |
| `RL/gripper_rl_env.py` | Observations, actions and rewards |
| `RL/train_ppo_codesign.py` | PPO training and checkpoint generation |
| `RL/evaluate_ppo_policy.py` | Policy evaluation using a saved checkpoint |
| `RL/poly_meta-model*` | Trained meta-model and scaling objects |
| `RL/models/` | PPO checkpoints |

## Environment

Use `../../environment/environment.yml` as the Python environment specification. Dependencies include PyBullet, NumPy, TensorFlow/Keras, scikit-learn/joblib, Gym, Stable-Baselines3 and Weights & Biases for tracked training.

## Running the workflows

Activate a compatible Python environment first. Run these entry points separately from the repository root:

```bash
python cases/gripper/Sim2Real/run_grasp_simulation.py
python cases/gripper/Sim2Real/run_grasp_trials.py
python cases/gripper/RL/train_surrogate_meta_model.py
python cases/gripper/RL/train_ppo_codesign.py
python cases/gripper/RL/evaluate_ppo_policy.py
```

PPO evaluation uses `RL/models/ywk5xiqs/final_model.zip` by default. Set `GRIPPER_PPO_MODEL_PATH` to select another checkpoint. PPO training requires an appropriate Weights & Biases configuration.

**Write safety:** Model training may overwrite trained assets, and trial scripts may write to result directories. Make a working copy before running write-producing workflows.

## Resource preflight

```bash
python cases/gripper/preflight.py
```

This check inspects entry points and packaged resources without importing PyBullet or TensorFlow; it is not a numerical simulation test. On case-sensitive systems, verify that mesh paths resolve to the intended files.
