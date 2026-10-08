# Research workflows

| Case | Objective | Key code |
|---|---|---|
| Gripper | Grasping simulation and sim-to-real comparison | `cases/gripper/Sim2Real/run_grasp_simulation.py`, `run_grasp_trials.py` |
| Gripper | RL co-design | `cases/gripper/RL/train_surrogate_meta_model.py`, `train_ppo_codesign.py`, `evaluate_ppo_policy.py` |
| Helical actuator | Deformation simulation and surrogate modeling | `cases/helical/Sim2Real/` |
| Helical actuator | Shape-matching design optimization | `cases/helical/Shape-matching_opt/` |
| Tendon-driven actuator | Deformation simulation | `cases/tendon_driven/` |
| Translation | Stiffness/generalization analysis | `surrogate/matlab/` |

Start with the README in each case directory for experiment-specific inputs and settings.
