PY ?= python3

.PHONY: doctor bench train test compile serve

doctor:
	$(PY) -m tevforge doctor

bench:
	$(PY) -m tevforge bench

train:
	$(PY) -m tevforge train --steps 40 --tasks 8

test:
	$(PY) -m unittest discover -s tests -t .

compile:
	$(PY) -m compileall -q tevforge tests examples

serve:
	$(PY) -m tevforge serve
