"""Run the complete chess behavior analysis pipeline."""

from __future__ import annotations

from src.pipeline import run


CONFIG_PATH = "config/config.yaml"


def main() -> None:
    """Run all analysis stages in order."""
    run("all", CONFIG_PATH)


if __name__ == "__main__":
    main()
