# Prompt templates

Reusable prompt / workflow templates for building long-horizon agent training
pipelines. Each template is a self-contained briefing you can paste into an LLM
or use to structure a sub-agent.

| file | purpose |
|------|---------|
| 01_env_synthesis | build a verifiable environment |
| 02_reward_shaping | design a progress potential (PPRS) |
| 03_task_decomposition | split a task into verifiable subgoals |
| 04_progress_annotation | annotate dense per-step rewards |
| 05_rollout_reflection | reflect on a failed rollout |
| 06_verifier_design | write a deterministic completion verifier |
| 07_curriculum | order tasks by horizon |
| 08_ablation | prove PPRS beats sparse |
