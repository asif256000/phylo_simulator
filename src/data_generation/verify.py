"""Utilities for mirroring generated PhyloXML datasets to Newick text dumps."""

from __future__ import annotations

import argparse
from pathlib import Path

from Bio import Phylo

from src.data_generation.config import load_generation_config


def verify_from_config(config_path: Path | str, *, output_path: Path | None = None) -> Path:
    """Load *config_path*, locate its XML dataset, and write a Newick dump."""

    config = load_generation_config(config_path)
    xml_path = config.dataset.xml_path()
    if not xml_path.exists():
        raise FileNotFoundError(f"PhyloXML file not found: {xml_path}")

    destination = Path(output_path) if output_path else _default_verify_path(xml_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_newick_dump(xml_path, destination)
    return destination


def _default_verify_path(xml_path: Path) -> Path:
    xml_directory = xml_path.parent
    verify_directory = xml_directory / "verify"
    verify_directory.mkdir(parents=True, exist_ok=True)
    return verify_directory / f"{xml_path.stem}.txt"


def _write_newick_dump(xml_path: Path, output_path: Path) -> None:
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for count, phylogeny in enumerate(Phylo.parse(str(xml_path), "phyloxml"), start=1):
            newick_str = phylogeny.format("newick").strip()
            if not newick_str.endswith(";"):
                newick_str += ";"
            handle.write(newick_str)
            handle.write("\n")
    if count == 0:
        output_path.write_text("", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to the generation configuration file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional destination path for the Newick dump",
    )
    return parser.parse_args()


def main() -> None:
    import time
    args = _parse_args()
    t0 = time.time()
    output_path = verify_from_config(args.config, output_path=args.output)
    t1 = time.time()
    print(f"Wrote Newick dump to {output_path}")
    print(f"Time taken: {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    main()
