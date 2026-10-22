# Troubleshooting

**`No module named yaml` (only in exotic runtimes)**
The managed runtime ships a hand-rolled YAML parser (`utils/yamlish.py`) as a
fallback. Install `pyyaml` only if you want the full parser.

**Training curve flat at skill = 0.5**
Check you are using a dense shaped reward (PPRS), not the sparse reward. Sparse
reward at the random baseline yields all-zero advantages, so the group carries
no gradient.

**Benchmark numbers look "too perfect"**
They are deterministic by design (single oracle + md5-seeded). That is the
point: it makes the reward math auditable and regression-testable.

**`completion_rate` is 0 at skill = 0.5**
Expected: `0.5^L ~ 1.5e-5` for a 16-step task. This is exactly the failure
sparse reward suffers and PPRS fixes.
