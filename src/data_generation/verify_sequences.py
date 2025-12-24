"""Dump simulated sequences from PhyloXML datasets into a FASTA file."""

from __future__ import annotations

import argparse
from pathlib import Path

from Bio import Phylo

from src.data_generation.config import load_generation_config


def verify_sequences_from_config(config_path: Path | str) -> Path:
    """Load *config_path*, locate its XML dataset, and write a FASTA dump.

    Sequences are written in the order of the configured taxa, with the tree
    index appended to each taxon label (e.g., A_1, B_1, A_2...).
    The output FASTA file is written to <xml_directory>/verify/<output_name>_sequences.fasta.
    """

    config = load_generation_config(config_path)
    xml_path = config.dataset.xml_path()
    if not xml_path.exists():
        raise FileNotFoundError(f"PhyloXML file not found: {xml_path}")

    destination = _default_sequence_path(xml_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    _write_fasta_dump(xml_path, destination, config.tree.taxa_labels)
    return destination


def _default_sequence_path(xml_path: Path) -> Path:
    xml_directory = xml_path.parent
    verify_directory = xml_directory / "verify"
    verify_directory.mkdir(parents=True, exist_ok=True)
    return verify_directory / f"{xml_path.stem}_sequences.fasta"


def _write_fasta_dump(xml_path: Path, output_path: Path, taxa_labels: tuple[str, ...]) -> None:
    tree_index = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for tree_index, phylogeny in enumerate(Phylo.parse(str(xml_path), "phyloxml"), start=1):
            sequence_map = _extract_sequences(phylogeny, taxa_labels)
            for taxon in taxa_labels:
                header = f">{taxon}_{tree_index}"
                handle.write(header)
                handle.write("\n")
                handle.write(sequence_map[taxon])
                handle.write("\n")
    if tree_index == 0:
        output_path.write_text("", encoding="utf-8")


def _extract_sequences(phylogeny, taxa_labels: tuple[str, ...]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    for taxon in taxa_labels:
        clade = phylogeny.find_any(name=taxon)
        if clade is None:
            raise ValueError(f"Phylogeny missing taxon '{taxon}'")

        entries = getattr(clade, "sequences", None) or []
        if not entries:
            raise ValueError(f"No sequences attached to taxon '{taxon}'")
        sequence_obj = entries[0]
        mol_seq = getattr(sequence_obj, "mol_seq", None)
        sequence_value = getattr(mol_seq, "value", None) if mol_seq is not None else None
        if sequence_value is None:
            raise ValueError(f"Sequence missing mol_seq value for taxon '{taxon}'")

        cleaned_sequence = "".join(str(sequence_value).split()).upper()
        if not cleaned_sequence:
            raise ValueError(f"Empty sequence for taxon '{taxon}'")
        sequences[taxon] = cleaned_sequence
    return sequences


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to the generation configuration file",
    )
    return parser.parse_args()


def main() -> None:
    import time

    args = _parse_args()
    t0 = time.time()
    output_path = verify_sequences_from_config(args.config)
    t1 = time.time()
    print(f"Wrote FASTA dump to {output_path}")
    print(f"Time taken: {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    main()
