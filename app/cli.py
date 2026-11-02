import argparse
import json
from app.engine import run

def main():
    parser = argparse.ArgumentParser(description="Run AgentBench.")
    parser.add_argument("--provider", choices=["mock", "ollama"], default="mock")
    parser.add_argument("--model", default="llama3.2:3b")
    args = parser.parse_args()
    print(json.dumps(run(args.provider, args.model), indent=2))

if __name__ == "__main__":
    main()
