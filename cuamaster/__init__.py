"""CUAMaster: Scaling Computer-Use Agents through Open-World Recursive Improvement.

Package layout mirrors the method described in the paper:

- :mod:`cuamaster.action_space` -- the hybrid atomic action space (GUI + CLI + tools).
- :mod:`cuamaster.runtime` -- benchmark-independent canonical CUA runtime.
- :mod:`cuamaster.seeding` -- open-world seeding: corpus harvesting, seed task
  mining, and visual-semantic dual-path trajectory reconstruction.
- :mod:`cuamaster.propagation` -- autocatalytic propagation: primitive extraction
  and primitive-based composable task synthesis.
- :mod:`cuamaster.collection` -- resilient trajectory collection with the
  feedback-guided snapshot-rollback retry mechanism.
- :mod:`cuamaster.benchmarks` -- benchmark interaction harnesses.
"""

__version__ = "0.1.0"
