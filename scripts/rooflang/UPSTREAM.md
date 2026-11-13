# Rooflang source subset

Source: https://github.com/yzygitzh/rooflang
Revision: 5cf122bb260fb53dd85688cd88001627e33b018c
License: MIT; see LICENSE.txt and preserved source headers.

The Python source files are copied unchanged from this revision. Included:
graphs, tensors, hardware components, placement, forward/identity/communication
kernels, dtype utilities, simulator and trace export, with their package files.
Programs, experiments, optimization passes and training kernels are omitted.
Package docstrings may describe upstream modules outside this subset.
Simthesizer-specific model definitions and adapters live in ../rooflang_bridge.
No separate Rooflang checkout or installation is needed. NetworkX is declared
in the repository's pyproject.toml and uv.lock.
