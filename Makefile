PYTHON ?= python
CONFIG ?= config/config.yaml

.PHONY: help install test collect parse features analyze model viz heterogeneity v13 multi pipeline clean

help:
	@echo "make install        - install pinned dependencies"
	@echo "make test           - run pytest"
	@echo "make pipeline       - run the original single-player baseline"
	@echo "make multi          - run the final multi-player research pipeline"
	@echo "make collect        - download baseline Lichess PGN"
	@echo "make parse          - parse baseline PGN"
	@echo "make features       - build baseline behavioral features"
	@echo "make analyze        - run baseline statistical analysis"
	@echo "make model          - run baseline predictive models"
	@echo "make viz            - generate baseline figures"
	@echo "make heterogeneity  - run v12 player-level heterogeneity analysis"
	@echo "make v13             - run v13 null-model and opponent-adjusted analysis"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	PYTHONPATH=. pytest -q

collect:
	PYTHONPATH=. $(PYTHON) -m src.pipeline collect --config $(CONFIG)

parse:
	PYTHONPATH=. $(PYTHON) -m src.pipeline parse --config $(CONFIG)

features:
	PYTHONPATH=. $(PYTHON) -m src.pipeline features --config $(CONFIG)

analyze:
	PYTHONPATH=. $(PYTHON) -m src.pipeline analyze --config $(CONFIG)

model:
	PYTHONPATH=. $(PYTHON) -m src.pipeline model --config $(CONFIG)

viz:
	PYTHONPATH=. $(PYTHON) -m src.pipeline viz --config $(CONFIG)

heterogeneity:
	PYTHONPATH=. $(PYTHON) -m src.pipeline heterogeneity --config $(CONFIG)

v13:
	PYTHONPATH=. $(PYTHON) -m src.pipeline v13 --config $(CONFIG)

pipeline: collect parse features analyze model viz

multi:
	PYTHONPATH=. $(PYTHON) -m src.pipeline multi --config $(CONFIG)

clean:
	rm -f data/processed/*.csv results/*.json results/*.csv results/*.md figures/*.png figures/*.svg
