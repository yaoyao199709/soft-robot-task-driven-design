# Software environment

The research code uses one Python dependency specification and a separate MATLAB installation.

## Python

From the repository root:

```bash
conda env create -f environment/environment.yml
conda activate soft_robotics_sim
```

`environment.yml` includes packages used across the gripper, helical, and tendon-driven task simulations, surrogate learning, and design optimization. Python 3.11 and NumPy 1.26 are specified for compatibility with the Gym 0.26 and TensorFlow 2.18 workflows. The environment file is an installation specification, not a platform-specific lockfile; package resolution and runtime compatibility may vary by operating system and hardware.

## MATLAB

See [MATLAB.md](MATLAB.md) for the MATLAB release range, toolbox requirements, and included FEM-data processing workflows.

MATLAB is not required for running the provided Python task simulations from existing model and data files.
