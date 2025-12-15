from __future__ import annotations

from pathlib import Path

from Bio.Phylo._io import parse as phylo_parse

from src.data_generation.config import GenerationConfig

from .base import CladeRecord, TreeExample


class PhyloExampleExtractor:
    """Extracts tree examples from a PhyloXML file based on a generation config."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.clade_names = tuple(config.tree.taxa_labels)
        self.sequence_length = config.sequence.length
        self.xml_path = config.dataset.xml_path()

    def extract(self) -> list[TreeExample]:
        xml_path = self.xml_path
        if not xml_path.exists():
            raise FileNotFoundError(f"PhyloXML file not found: {xml_path}")

        examples: list[TreeExample] = []
        for tree_index, tree in enumerate(phylo_parse(str(xml_path), "phyloxml")):
            clade_records: list[CladeRecord] = []
            missing_required_data = False
            for clade_name in self.clade_names:
                clade = tree.find_any(name=clade_name)
                if clade is None:
                    missing_required_data = True
                    break
                branch_length = clade.branch_length
                if branch_length is None:
                    missing_required_data = True
                    break
                if not getattr(clade, "sequences", None):
                    missing_required_data = True
                    break
                sequence_obj = clade.sequences[0]
                if getattr(sequence_obj, "mol_seq", None) is None:
                    missing_required_data = True
                    break
                sequence_value = sequence_obj.mol_seq.value
                if sequence_value is None:
                    missing_required_data = True
                    break
                cleaned_sequence = "".join(sequence_value.split()).upper()
                clade_records.append(
                    CladeRecord(
                        name=clade_name,
                        sequence=cleaned_sequence,
                        branch_length=float(branch_length),
                    )
                )
            if missing_required_data:
                continue
            
            branches = {}
            clade_splits = {}
            for clade in tree.find_clades(order="postorder"):
                if clade.is_terminal():
                    split = frozenset([clade.name])
                else:
                    split = frozenset()
                    for child in clade.clades:
                        if child in clade_splits:
                            split = split | clade_splits[child]
                
                clade_splits[clade] = split
                
                if clade is not tree.root and clade.branch_length is not None:
                    branches[split] = float(clade.branch_length)

            metadata = _extract_metadata_annotations(tree)
            examples.append(
                TreeExample(
                    tree_index=tree_index,
                    clades=tuple(clade_records),
                    branches=branches,
                    metadata=metadata or None,
                )
            )
        return examples

    def xml_path_exists(self) -> bool:
        return Path(self.xml_path).exists()


def _extract_metadata_annotations(tree) -> dict[str, str]:
    annotations = {}
    entries = getattr(tree, "other", None) or []
    for entry in entries:
        tag = getattr(entry, "tag", None)
        value = getattr(entry, "value", None)
        if not tag or value is None:
            continue
        annotations[str(tag)] = str(value)
    return annotations
