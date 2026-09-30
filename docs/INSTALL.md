# Reproduction

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

To run the v13 analytical layer on the verified multi-player feature file:

```bash
python -m src.pipeline v13 --config config/config.yaml
```

This command does not recollect data. It uses `data/processed/multi_player_games_features.csv`.
