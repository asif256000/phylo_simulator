from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from multiprocessing.pool import ThreadPool

from src.data_generation.config import GenerationConfig
from src.utils.labels import get_topology_map

from ..base import TreeExample, nucleotide_channel_count, one_hot_encode
from .base import DatasetWriter


class NpyDatasetWriter(DatasetWriter):
    """Writes parsed PhyloXML examples to a structured NumPy .npy file."""

    def __init__(self, config: GenerationConfig, output_path: Path | None = None) -> None:
        super().__init__(config, output_path)
        self.clade_names = tuple(config.tree.taxa_labels)
        self.sequence_length = config.sequence.length
        self.config.dataset.ensure_npy_directory()
        self.include_gap_channel = bool(config.simulation.indel.enabled)
        self.channel_count = nucleotide_channel_count(self.include_gap_channel)

        self.topology_map = get_topology_map(config.tree.topologies)
        self.use_special_branch_order = True  # only 2–4 taxa supported
        self.num_branches = _branch_vector_length(self.clade_names, rooted=config.tree.rooted)
        self.num_topologies = len(self.topology_map)

    def write(self, examples: Sequence[TreeExample]) -> Path:
        example_count = len(examples)
        if example_count == 0:
            raise ValueError("No usable phylogenies were found in the input file")

        output_path = self.output_path or self.config.dataset.output_npy_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        dtype = np.dtype(
            [
                ("X", np.uint8, (len(self.clade_names), self.sequence_length, self.channel_count)),
                ("y_br", np.float32, (self.num_branches,)),
                ("branch_mask", np.bool_, (self.num_branches,)),
                ("y_top", np.float32, (self.num_topologies,)),
                ("tree_index", np.int32),
            ]
        )

        dataset = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=dtype,
            shape=(example_count,),
        )

        jobs: Iterable[tuple[int, TreeExample, tuple[str, ...], int, int, bool, bool, dict[str, int], bool, int]] = (
            (
                row_index,
                example,
                self.clade_names,
                self.sequence_length,
                self.channel_count,
                self.include_gap_channel,
                self.use_special_branch_order,
                self.topology_map,
                self.config.tree.rooted,
                self.num_branches,
            )
            for row_index, example in enumerate(examples)
        )

        try:
            if self.parallel_cores <= 1:
                for row_index, tree_index, encoded, y_br, branch_mask, y_top in map(_encode_example, jobs):
                    dataset[row_index]["tree_index"] = np.int32(tree_index)
                    dataset[row_index]["X"][...] = encoded
                    dataset[row_index]["y_br"][...] = y_br
                    dataset[row_index]["branch_mask"][...] = branch_mask
                    dataset[row_index]["y_top"][...] = y_top
            else:
                with ThreadPool(processes=self.parallel_cores) as pool:
                    for row_index, tree_index, encoded, y_br, branch_mask, y_top in pool.imap(_encode_example, jobs):
                        dataset[row_index]["tree_index"] = np.int32(tree_index)
                        dataset[row_index]["X"][...] = encoded
                        dataset[row_index]["y_br"][...] = y_br
                        dataset[row_index]["branch_mask"][...] = branch_mask
                        dataset[row_index]["y_top"][...] = y_top
        finally:
            dataset.flush()
            del dataset

        return output_path


def _encode_example(
    payload: tuple[int, TreeExample, tuple[str, ...], int, int, bool, bool, dict[str, int], bool, int]
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        row_index,
        example,
        clade_names,
        sequence_length,
        channel_count,
        include_gap,
        use_special_branch_order,
        topology_map,
        rooted,
        expected_branches,
    ) = payload

    expected_taxa = len(clade_names)
    if len(example.clades) != expected_taxa:
        raise ValueError(
            "Example does not contain the expected number of clades: "
            f"expected {expected_taxa}, observed {len(example.clades)}"
        )

    encoded = np.empty((expected_taxa, sequence_length, channel_count), dtype=np.uint8)

    for clade_index, clade in enumerate(example.clades):
        try:
            encoding = one_hot_encode(clade.sequence, sequence_length, include_gap=include_gap)
            encoded[clade_index] = encoding
        except ValueError as exc:
            raise ValueError(
                f"Tree index {example.tree_index} clade '{clade.name}' failed encoding: {exc}"
            ) from exc

    # Encode y_br and branch_mask
    if use_special_branch_order:
        y_br, branch_mask = _encode_branches_special(example.branches, clade_names, rooted=rooted)
    else:
        raise ValueError("Unsupported taxa count; only 2–4 taxa are supported in this writer.")

    if not rooted and use_special_branch_order:
        filtered = [(value, mask_val) for value, mask_val in zip(y_br, branch_mask) if mask_val]
        y_br = np.array([value for value, _ in filtered], dtype=np.float32)
        branch_mask = np.ones(len(y_br), dtype=np.bool_)
    
    if len(y_br) != expected_branches:
        raise ValueError(
            f"Branch vector length mismatch: expected {expected_branches}, observed {len(y_br)}"
        )

    # Encode y_top
    y_top = np.zeros(len(topology_map), dtype=np.float32)
    if example.metadata and "topology" in example.metadata:
        topo_str = example.metadata["topology"]
        if topo_str in topology_map:
            y_top[topology_map[topo_str]] = 1.0

    return row_index, example.tree_index, encoded, y_br, branch_mask, y_top


def _branch_vector_length(
    clade_names: Sequence[str],
    *,
    rooted: bool,
) -> int:
    num_taxa = len(clade_names)
    if rooted:
        if num_taxa == 2:
            return 2
        if num_taxa == 3:
            return 6
        if num_taxa == 4:
            return 10
        raise ValueError("Only 2–4 taxa are supported for rooted datasets.")

    # Unrooted: emit only present edges in order (no doubling)
    if num_taxa == 2:
        return 1
    if num_taxa == 3:
        return 3
    if num_taxa == 4:
        return 5
    raise ValueError("Only 2–4 taxa are supported for unrooted datasets.")


def _encode_branches_special(
    branches: Mapping[frozenset[str], float],
    clade_names: Sequence[str],
    *,
    rooted: bool,
) -> tuple[np.ndarray, np.ndarray]:
    num_taxa = len(clade_names)
    taxa_set = set(clade_names)

    if rooted:
        if num_taxa == 2:
            y_br = np.zeros(2, dtype=np.float32)
            mask = np.zeros(2, dtype=np.bool_)
            for idx, taxon in enumerate(clade_names):
                split = frozenset([taxon])
                if split in branches:
                    y_br[idx] = branches[split]
                    mask[idx] = True
            return y_br, mask

        if num_taxa == 3:
            y_br = np.zeros(6, dtype=np.float32)
            mask = np.zeros(6, dtype=np.bool_)
            for idx, taxon in enumerate(clade_names):
                primary = frozenset([taxon])
                complement = frozenset(taxa_set - {taxon})
                base = idx * 2
                if primary in branches:
                    y_br[base] = branches[primary]
                    mask[base] = True
                if complement in branches:
                    y_br[base + 1] = branches[complement]
                    mask[base + 1] = True
            return y_br, mask

        if num_taxa == 4:
            y_br = np.zeros(10, dtype=np.float32)
            mask = np.zeros(10, dtype=np.bool_)

            # Leaf branches (a1/a2, b1/b2, c1/c2, d1/d2)
            for idx, taxon in enumerate(clade_names):
                primary = frozenset([taxon])
                complement = frozenset(taxa_set - {taxon})
                base = idx * 2
                if primary in branches:
                    y_br[base] = branches[primary]
                    mask[base] = True
                if complement in branches:
                    y_br[base + 1] = branches[complement]
                    mask[base + 1] = True

            # Internal branch (i1/i2)
            size_two = sorted(
                [split for split in branches.keys() if len(split) == 2 and split.issubset(taxa_set)],
                key=lambda s: sorted(s),
            )
            internal_base = 8
            if size_two:
                primary_internal = size_two[0]
                complement_internal = frozenset(taxa_set - set(primary_internal))
                y_br[internal_base] = branches[primary_internal]
                mask[internal_base] = True
                if complement_internal in branches:
                    y_br[internal_base + 1] = branches[complement_internal]
                    mask[internal_base + 1] = True
            return y_br, mask

        return _encode_branches_canonical(branches, canonical_slots=[])

    # Unrooted (no doubling): emit only present edges in deterministic order
    if num_taxa == 2:
        y_br = np.zeros(1, dtype=np.float32)
        mask = np.zeros(1, dtype=np.bool_)
        split = frozenset([clade_names[0]])
        if split in branches:
            y_br[0] = branches[split]
            mask[0] = True
        return y_br, mask

    if num_taxa == 3:
        y_br = np.zeros(3, dtype=np.float32)
        mask = np.zeros(3, dtype=np.bool_)
        for idx, taxon in enumerate(clade_names):
            split = frozenset([taxon])
            if split in branches:
                y_br[idx] = branches[split]
                mask[idx] = True
        return y_br, mask

    if num_taxa == 4:
        y_br = np.zeros(5, dtype=np.float32)
        mask = np.zeros(5, dtype=np.bool_)

        # Leaf branches first
        for idx, taxon in enumerate(clade_names):
            split = frozenset([taxon])
            if split in branches:
                y_br[idx] = branches[split]
                mask[idx] = True

        # Internal branch last
        size_two = sorted(
            [split for split in branches.keys() if len(split) == 2 and split.issubset(taxa_set)],
            key=lambda s: sorted(s),
        )
        internal_idx = 4
        if size_two:
            y_br[internal_idx] = branches[size_two[0]]
            mask[internal_idx] = True
        return y_br, mask

    return _encode_branches_canonical(branches, canonical_slots=[])


def _encode_branches_canonical(
    branches: Mapping[frozenset[str], float],
    canonical_slots: Sequence[frozenset[str]],
) -> tuple[np.ndarray, np.ndarray]:
    raise ValueError("Canonical encoding is not used; only 2–4 taxa are supported.")
