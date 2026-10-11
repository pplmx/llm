"""Allow ``python -m llm ...`` to run the unified CLI (same as ``llm ...``)."""

from llm.cli.main import app

if __name__ == "__main__":
    app()
