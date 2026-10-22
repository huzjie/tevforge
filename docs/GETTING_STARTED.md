# Getting Started

`tevforge` is pure Python with **zero dependencies** (Python 3.8+).

## Install & verify

```bash
python -m tevforge doctor     # smoke-test every component
python -m tevforge bench      # run the benchmark suite
python -m tevforge train --steps 40 --tasks 8
python -m tevforge serve      # HTTP server on :8090
```

## Train a long-horizon agent

```python
from tevforge.config import Config
from tevforge.engine.train_engine import TrainEngine

engine = TrainEngine(Config())
result = engine.fit(n_tasks=8, max_steps=40)
print(result["final_skill"])          # 0.5 -> ~1.0
```

## Run tests

```bash
python -m unittest discover -s tests -t .
```
