"""Resilient snapshot-and-rollback trajectory collection (paper Sec. 2.4.2).

Rolls out verified tasks in sandboxed environments, detects failures or
inefficiencies, localizes the divergence step, and retries from the latest
valid snapshot instead of restarting the whole rollout.
"""

from .export import export_result_tree
from .profiles import MatrixConfig, VMProfile, load_matrix_config
from .schemas import AgentSpec, EpisodeStatus, StepRecord, TargetEnvironment, TaskSpec

__all__ = [
    "AgentSpec",
    "EpisodeStatus",
    "MatrixConfig",
    "StepRecord",
    "TargetEnvironment",
    "TaskSpec",
    "VMProfile",
    "export_result_tree",
    "load_matrix_config",
]
