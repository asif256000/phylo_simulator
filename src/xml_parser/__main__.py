import argparse
from collections.abc import Sequence
from pathlib import Path

from .parser import XMLParser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse PhyloXML outputs into structured datasets.")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/generation.yaml"),
        help="Path to the YAML/JSON generation configuration file.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    import time
    args = parse_args(argv)
    parser = XMLParser.from_config_file(args.config)
    t0 = time.time()
    examples = parser.parse_examples()
    output_path = parser.write_dataset(examples)
    t1 = time.time()
    print(f"Wrote NPY dataset to {output_path}")
    print(f"Time taken: {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    main()
