# Task-level simulations and design applications

Three actuator case studies use FEM-derived surrogate response models in task-level simulation. Each case has its own README with entry points and resources.

| Case | Validation | Design application |
|---|---|---|
| [Gripper](gripper/) | Simulated grasping vs. experiments | PPO morphology/control co-design |
| [Helical actuator](helical/) | Gravity and tip-load deformation | CMA-ES shape-matching optimization |
| [Tendon-driven actuator](tendon_driven/) | Free-load and tip-load deformation | — |

Precomputed datasets and MATLAB-based FEM-response processing are described in [the surrogate layer](../surrogate/README.md). Check each case's README before training: model fitting and simulation utilities may write into their working directories.
