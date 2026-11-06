import argparse
import json

from app.config import settings
from app.engine import run
from app.storage import save_run


def main():
    parser = argparse.ArgumentParser(description="Run AgentBench.")
    parser.add_argument(
        "--provider",
        choices=["groq", "mock", "ollama"],
        default=settings.default_provider,
    )
    parser.add_argument("--model", default=settings.default_model)
    args = parser.parse_args()
    result = save_run(run(args.provider, args.model))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
