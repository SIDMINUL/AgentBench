import argparse
import json

from app.engine import run
from app.storage import save_run


def main():
    parser = argparse.ArgumentParser(description="Run AgentBench.")
    parser.add_argument("--provider", choices=["mock", "ollama"], default="mock")
    parser.add_argument("--model", default="llama3.2:3b")
    args = parser.parse_args()
    result = save_run(run(args.provider, args.model))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
