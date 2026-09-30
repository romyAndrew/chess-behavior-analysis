PYTHON ?= python
CONFIG ?= config/config.yaml

.PHONY: help install test multi v13 clean

help:
	@echo "make install  - install pinned dependencies"
	@echo "make test     - run pytest"
	@echo "make multi    - run the final multi-player research pipeline"
	@echo "make v13      - run the synthetic-null and opponent-adjusted analysis"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	PYTHONPATH=. pytest -q

multi:
	PYTHONPATH=. $(PYTHON) -m src.pipeline multi --config $(CONFIG)

v13:
	PYTHONPATH=. $(PYTHON) -m src.pipeline v13 --config $(CONFIG)

clean:
	rm -f data/processed/*.csv data/processed/multi_player/*.csv figures/*.png results/*.json results/*.csv results/*.md
