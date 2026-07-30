import argparse
import sys
from pathlib import Path

from llm.tokenization.bpe_tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on text files.")
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="One or more paths to text files for training.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tokenizer.json",
        help="Path to save the trained tokenizer.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=5000,
        help="The desired vocabulary size.",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="The minimum frequency for a pair to be merged.",
    )
    parser.add_argument(
        "--special_tokens",
        nargs="+",
        default=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        help="List of special tokens to include.",
    )

    args = parser.parse_args()

    # Verify files exist
    training_files = []
    for f in args.files:
        if Path(f).exists():
            training_files.append(f)
        else:
            print(f"Warning: File {f} does not exist. Skipping.")  # noqa: T201 - CLI user-facing

    if not training_files:
        print("Error: No valid training files provided.")  # noqa: T201 - CLI
        sys.exit(1)

    print(f"Training BPE tokenizer on {len(training_files)} files...")  # noqa: T201
    print(f"Vocab size: {args.vocab_size}")  # noqa: T201
    print(f"Output path: {args.output}")  # noqa: T201

    tokenizer = BPETokenizer.train(
        files=training_files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=args.special_tokens,
    )

    tokenizer.save(args.output)
    print(f"Tokenizer saved to {args.output}")  # noqa: T201
    print(f"Final vocab size: {tokenizer.vocab_size}")  # noqa: T201


if __name__ == "__main__":
    main()
