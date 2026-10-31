# Annotator Role Prompts

CUAMaster employs a general-purpose annotator model with eight specialized roles
throughout the data construction pipeline (paper Table 8). Each file below is the
full prompt for one role; the paper figures show typeset (compressed) versions of
the same prompts.

| File | Role (Table 8) | Function | Paper figure |
| --- | --- | --- | --- |
| `video_suitability_scoring.txt` | Score Evaluator | Video suitability scoring and filtering | Fig. 11 |
| `seed_task_mining.txt` | Task Generator | Seed task mining and temporal segmentation | Fig. 12 |
| `action_boundary_refinement.txt` | Action Rechecker | Visual-semantic action boundary refinement | Fig. 13 |
| `action_annotation.txt` | Action Annotator | Atomic GUI action annotation | Fig. 14 |
| `trajectory_verification.txt` | Trajectory Verifier | Trajectory-level verification and filtering | Fig. 15 |
| `primitive_extraction.txt` | Primitive Extractor | Reusable ability primitive extraction | Fig. 16 |
| `task_synthesis.txt` | Task Synthesizer | Composable task and verifier synthesis | Fig. 17 |
| `failure_detection.txt` | Detector | Failure localization and diagnostic feedback | Fig. 18 |

Placeholder conventions:

- `<TRANSCRIPT>`, `<TASK>`, `<APPLICATION>`, … — substituted with per-item text at request time.
- `<*_IMAGE_i>` / `{VIDEO_FRAME_i}` — substituted with the corresponding image inputs.
- `<*_JSON>` — substituted with the serialized JSON payload described by the surrounding pipeline stage.
- `<ACTION_SPACE>` — the allowed action-type list from `cuamaster.action_space` (paper Table 7).

Pointer coordinates are always normalized to `[0, 1000]` (resolution-independent).
