# Repository navigation for code agents

Read [README.md](README.md), [surrogate/README.md](surrogate/README.md), and [cases/README.md](cases/README.md) before making changes. Each case also provides a technical README.

## Research boundaries

- The provided MAT/CSV datasets represent precomputed FEM responses. COMSOL generation is outside this repository.
- Keep physics, numerical formulas, design variables, scalers, trained weights, and training hyperparameters stable unless explicitly tasked otherwise.
- Treat geometry meshes, model weights, dataset CSVs and trained checkpoints as immutable inputs unless the requested operation explicitly generates replacements.
- Use paths relative to script location, not machine-specific absolute paths. Do not assume the current working directory.
- Prefer documentation or minimal safe edits to rewriting an algorithm solely for style.
- Do not run scripts that overwrite packaged reference outputs in place; use a disposable copy.

## Entry points

- Gripper: [cases/gripper/README.md](cases/gripper/README.md)
- Helical: [cases/helical/README.md](cases/helical/README.md)
- Tendon-driven: [cases/tendon_driven/README.md](cases/tendon_driven/README.md)
- MATLAB: [surrogate/README.md](surrogate/README.md)

Report the boundary between static validation and real PyBullet, MATLAB, and TensorFlow execution; do not claim results were reproduced unless run.
