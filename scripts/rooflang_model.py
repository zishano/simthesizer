"""Export a Python model's structure and scheduler metadata to Rust."""

import argparse
import importlib
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    args = parser.parse_args()
    request = json.load(sys.stdin)
    if request.get("version") != 1:
        raise ValueError("unsupported model protocol")
    model = importlib.import_module(args.module)
    print(json.dumps(model.describe(request["parallel"]), allow_nan=False))


if __name__ == "__main__":
    main()
