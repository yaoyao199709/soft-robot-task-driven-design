# Code and data maintenance guidelines

1. Keep one cohesive repository, with a root overview and focused README files for each layer and actuator.
2. Organize by research role: FEM-derived inputs and surrogate analysis under `surrogate/`, task simulations and optimization under `cases/`.
3. Give each entry point a descriptive name and a module docstring stating its task, inputs, and outputs. Add comments for modeling assumptions and units; avoid commented-out historical code.
4. Preserve algorithm and data semantics during readability refactors. Use AST comparisons for comment/docstring-only edits and content hashes for data, meshes, trained models, and checkpoints.
5. Audit imports and filesystem references when renaming or moving files. Resolve resources from the script path and write generated outputs to clearly defined locations.
6. Record external dependencies and version/toolbox requirements without claiming untested minimum versions. Do not assume a local MATLAB, PyBullet, or training environment exists.
7. Validate in layers: syntax, static resource existence, data integrity, deterministic lightweight functions, then runtime simulation and training when an appropriate environment is available.
8. Keep migration diffs, source-file mappings, and internal regression logs outside the public repository. Publish user-facing documentation, not implementation history.
