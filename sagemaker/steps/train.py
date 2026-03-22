"""
Training dispatcher for the ED Triage pipeline.

SageMaker always invokes this file as the training entry point.
It resolves the architecture-specific training module and delegates to it.

How it works:
  1. SageMaker calls: python steps/train.py --architecture mock --epochs 1
  2. This script extracts --architecture, leaving remaining args in sys.argv
  3. It imports models/<arch>/train.py and calls its main()
  4. The model's main() sees only its own args (e.g. --epochs 1)

Directory layout:
  sagemaker/
    steps/train.py          <-- you are here (always the entry_point)
    models/
      mock/train.py         <-- resolved when --architecture=mock
      arch4/train.py        <-- resolved when --architecture=arch4

To add a new architecture:
  1. Create sagemaker/models/<name>/train.py with a main() function
  2. Run the pipeline with --architecture <name>
"""

import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Dispatch training to the appropriate architecture module.",
    )
    parser.add_argument(
        "--architecture",
        required=True,
        help="Model architecture to train (e.g. mock, arch4)",
    )
    args, remaining = parser.parse_known_args()

    # Strip --architecture so the model's argparse only sees its own args
    sys.argv = [sys.argv[0]] + remaining

    print(f"[dispatcher] Loading architecture: {args.architecture}")
    module = importlib.import_module(f"models.{args.architecture}.train")
    module.main()


if __name__ == "__main__":
    main()
