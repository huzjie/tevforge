# Recipes

Ready-to-use config / workflow recipes for different training regimes.

| recipe | shaping | use case |
|--------|---------|----------|
| pprs_terminal | pprs | terminal / SWE long-horizon |
| dense_reward | dense | shortest, densest signal |
| grpo_group | pprs | group size / LR tuning |
| long_horizon | pprs | 24-64 step tasks |
| curriculum | pprs | horizon curriculum |
| verifier | pprs | strict verifier checklist |

All recipes are plain YAML consumed by `Config.from_dict` (or the built-in
`utils/yamlish.py` fallback). They only change fields, not the pipeline.
