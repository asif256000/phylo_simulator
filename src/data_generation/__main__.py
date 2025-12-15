from __future__ import annotations

import argparse
from pathlib import Path

from .tree_sequence_generator import TreeSequenceGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate phylogenetic trees and sequences using the YAML/JSON configuration file.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/generation.yaml"),
        help="Path to the configuration file (default: config/generation.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    import time
    args = parse_args()
    generator = TreeSequenceGenerator.from_config_file(args.config)
    t0 = time.time()
    output_path = generator.write_xml()
    t1 = time.time()
    print(
        f"Generated {generator.config.dataset.tree_count} trees with backend"
        f" {generator.config.simulation.backend} -> {output_path}"
    )
    print(f"Time taken: {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    main()
