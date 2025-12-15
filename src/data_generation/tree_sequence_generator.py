from __future__ import annotations

import os
import random
import subprocess
import tempfile
import multiprocessing as mp
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterator

from Bio import SeqIO
from Bio.Phylo import PhyloXML
from Bio.Phylo._io import write as phylo_write
from Bio.Phylo.BaseTree import Clade, Tree as BaseTree
from Bio.Phylo.PhyloXML import Phylogeny, Other

from src.utils import flatten_topology, format_topology, infer_branch_output_count

from .config import GenerationConfig, TopologySpec


@dataclass
class TreeSequenceResult:
    """Container for the generated tree, sequences, and alignment metadata."""

    tree: BaseTree
    sequences: dict[str, str]
    aligned: bool
    topology: TopologySpec


class TreeSequenceGenerator:
    """Generate phylogenetic trees and associated sequences using reusable classes."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self.parallel_cores = max(1, config.parallel_cores)

    @classmethod
    def from_config_file(cls, config_path: Path | str) -> "TreeSequenceGenerator":
        from .config import load_generation_config

        config = load_generation_config(config_path)
        return cls(config)

    def generate_tree_and_sequences(self, topology: TopologySpec | None = None) -> TreeSequenceResult:
        tree, used_topology = self._build_tree(topology_override=topology)
        newick_str = self._tree_to_newick(tree)
        sequences, aligned = self._simulate_sequences(newick_str)
        return TreeSequenceResult(tree=tree, sequences=sequences, aligned=aligned, topology=used_topology)

    def generate_phylogeny(self, topology: TopologySpec | None = None) -> tuple[Phylogeny, bool]:
        result = self.generate_tree_and_sequences(topology=topology)
        phylogeny = self._attach_sequences(result.tree, result.sequences, result.aligned)
        self._annotate_topology(phylogeny, result.topology)
        return phylogeny, result.aligned

    def generate_phylogenies(self) -> tuple[list[Phylogeny], bool]:
        tree_count = self.config.dataset.tree_count
        seeds = [self._rng.randint(0, 2**32 - 1) for _ in range(tree_count)]
        schedule = self._topology_schedule(tree_count)

        phylogenies: list[Phylogeny] = []
        all_aligned = True

        if self.parallel_cores <= 1:
            for seed, topology in zip(seeds, schedule):
                phylogeny, aligned = _generate_phylogeny_worker((self.config, seed, topology))
                phylogenies.append(phylogeny)
                all_aligned = all_aligned and aligned
        else:
            payloads: Iterable[tuple[GenerationConfig, int, TopologySpec]] = (
                (self.config, seed, topology)
                for seed, topology in zip(seeds, schedule)
            )
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=self.parallel_cores) as pool:
                for phylogeny, aligned in pool.imap(_generate_phylogeny_worker, payloads):
                    phylogenies.append(phylogeny)
                    all_aligned = all_aligned and aligned

        return phylogenies, all_aligned

    def write_xml(self) -> Path:
        phylogenies, aligned = self.generate_phylogenies()
        dataset_settings = self.config.dataset
        dataset_settings.ensure_xml_directory()
        output_path = dataset_settings.xml_path()
        phyloxml = PhyloXML.Phyloxml({})
        phyloxml.phylogenies = phylogenies
        with output_path.open("w", encoding="utf-8") as handle:
            phylo_write(phyloxml, handle, "phyloxml")
        return output_path

    def _build_tree(self, topology_override: TopologySpec | None = None) -> tuple[BaseTree, TopologySpec]:
        taxa_count = self.config.tree.taxa_count
        topology = topology_override or self._select_topology(taxa_count)

        if len(flatten_topology(topology)) != taxa_count:
            raise ValueError("Provided topology does not match configured taxa count")

        if taxa_count == 2:
            tree = self._build_two_taxa_tree(topology)
        elif taxa_count == 3:
            tree = self._build_three_taxa_tree(topology)
        elif taxa_count == 4:
            tree = self._build_four_taxa_tree(topology)
        else:
            tree = self._build_tree_from_topology(topology)
        return tree, topology

    def _build_two_taxa_tree(self, topology: TopologySpec) -> BaseTree:
        if len(flatten_topology(topology)) != 2:
            raise ValueError("Two-taxa configurations must reference exactly two taxa")
        return self._build_tree_from_topology(topology)

    def _build_three_taxa_tree(self, topology: TopologySpec) -> BaseTree:
        if len(flatten_topology(topology)) != 3:
            raise ValueError("Three-taxa configurations must reference exactly three taxa")
        return self._build_tree_from_topology(topology)

    def _build_four_taxa_tree(self, topology: TopologySpec) -> BaseTree:
        if len(flatten_topology(topology)) != 4:
            raise ValueError("Four-taxa configurations must reference exactly four taxa")
        return self._build_tree_from_topology(topology)

    def _select_topology(self, taxa_count: int) -> TopologySpec:
        candidates = self._topology_candidates(taxa_count)
        return self._rng.choice(candidates)

    def _topology_candidates(self, taxa_count: int) -> list[TopologySpec]:
        configured = self.config.tree.topologies
        candidates = [topology for topology in configured if len(flatten_topology(topology)) == taxa_count]
        if not candidates:
            raise ValueError(
                "No configured topologies match the requested taxa count of " f"{taxa_count}."
            )
        return candidates

    def _topology_schedule(self, tree_count: int) -> list[TopologySpec]:
        candidates = self._topology_candidates(self.config.tree.taxa_count)
        count = len(candidates)
        return [candidates[index % count] for index in range(tree_count)]

    def _build_tree_from_topology(self, topology: TopologySpec) -> BaseTree:
        if not topology:
            raise ValueError("Topology definitions must include at least one group")

        group_clades = [self._build_group_clade(group) for group in topology]

        if self.config.tree.rooted:
            if topology.root_index is None:
                raise ValueError("Rooted tree configurations require ':' in the topology definition")
            left_groups = tuple(group_clades[: topology.root_index + 1])
            right_groups = tuple(group_clades[topology.root_index + 1 :])
            if not left_groups or not right_groups:
                raise ValueError("Rooted topologies must include taxa on both sides of ':'")
            left_subtree = self._build_chain_subtree(left_groups)
            right_subtree = self._build_chain_subtree(right_groups)
            root_clade = Clade(clades=[left_subtree, right_subtree])
        else:
            root_clade = self._build_chain_subtree(tuple(group_clades))

        tree = BaseTree(root=root_clade)
        tree.rooted = self.config.tree.rooted
        self._assign_branch_lengths(tree.root, topology)
        return tree

    def _build_group_clade(self, group: tuple[str, ...]) -> Clade:
        if len(group) == 1:
            return Clade(name=group[0])
        if len(group) == 2:
            left = Clade(name=group[0])
            right = Clade(name=group[1])
            return Clade(clades=[left, right])
        raise ValueError("Topology groups can contain at most two taxa")

    def _build_chain_subtree(self, groups: Sequence[Clade]) -> Clade:
        if not groups:
            raise ValueError("Topology definitions must include at least one group")
        subtree = groups[-1]
        for group in reversed(groups[:-1]):
            parent = Clade(clades=[group, subtree])
            subtree = parent
        return subtree

    def _assign_branch_lengths(self, root: Clade, topology: TopologySpec) -> None:
        num_taxa = self.config.tree.taxa_count
        segment_count = infer_branch_output_count(num_taxa, rooted=False)
        if segment_count <= 0:
            return

        samples = [self._sample_branch_length() for _ in range(segment_count)]
        length_iter = iter(samples)
        flattened = flatten_topology(topology)
        first_taxon = flattened[0] if flattened else None

        if self.config.tree.rooted and len(root.clades) >= 2:
            self._assign_rooted_branch_lengths(root, length_iter)
        else:
            self._assign_unrooted_branch_lengths(root, length_iter, first_taxon)

    def _assign_rooted_branch_lengths(self, root: Clade, length_iter: Iterator[float]) -> None:
        children = list(root.clades)
        if len(children) != 2:
            # Degenerate rooted trees (e.g., single taxon) fall back to the unrooted logic.
            self._assign_unrooted_branch_lengths(root, length_iter, None)
            return

        try:
            connector_length = next(length_iter)
        except StopIteration as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("Insufficient branch length samples for root connector") from exc

        min_len, _ = self.config.tree.branch_length_range
        lower_bound = min_len
        upper_bound = connector_length
        if upper_bound < lower_bound:
            lower_bound = upper_bound
        split_value = upper_bound if upper_bound == lower_bound else self._rng.uniform(lower_bound, upper_bound)
        remainder = max(connector_length - split_value, 0.0)

        if self._rng.random() < 0.5:
            children[0].branch_length = split_value
            children[1].branch_length = remainder
        else:
            children[0].branch_length = remainder
            children[1].branch_length = split_value

        for child in children:
            self._populate_branch_lengths(child, length_iter)

    def _assign_unrooted_branch_lengths(
        self,
        root: Clade,
        length_iter: Iterator[float],
        first_taxon: str | None,
    ) -> None:
        if self.config.tree.taxa_count == 2 and len(root.clades) >= 1:
            target_child = self._child_containing_taxon(root, first_taxon)
            if target_child is None:
                target_child = root.clades[0]
            try:
                target_child.branch_length = next(length_iter)
            except StopIteration as exc:  # pragma: no cover - defensive guard
                raise RuntimeError("Missing branch length sample for two-taxa unrooted tree") from exc
            return

        children = list(root.clades)
        if len(children) == 2:
            try:
                connector_length = next(length_iter)
            except StopIteration as exc:  # pragma: no cover - defensive guard
                raise RuntimeError("Missing branch length sample for unrooted connector") from exc

            if connector_length <= 0:
                split_value = 0.0
            else:
                split_value = self._rng.uniform(0.0, connector_length)
            remainder = max(connector_length - split_value, 0.0)

            if self._rng.random() < 0.5:
                children[0].branch_length = split_value
                children[1].branch_length = remainder
            else:
                children[0].branch_length = remainder
                children[1].branch_length = split_value

        self._populate_branch_lengths(root, length_iter)

    def _populate_branch_lengths(self, clade: Clade, length_iter: Iterator[float]) -> None:
        for child in clade.clades:
            if child.branch_length is None:
                try:
                    child.branch_length = next(length_iter)
                except StopIteration as exc:  # pragma: no cover - defensive guard
                    raise RuntimeError("Ran out of sampled branch lengths while assigning the tree") from exc
            self._populate_branch_lengths(child, length_iter)

    def _child_containing_taxon(self, root: Clade, taxon: str | None) -> Clade | None:
        if taxon is None:
            return None
        for child in root.clades:
            if any(leaf.name == taxon for leaf in child.get_terminals()):
                return child
        return None

    def _sample_branch_length(self) -> float:
        min_len, max_len = self.config.tree.branch_length_range
        distribution = self.config.tree.branch_length_distribution
        if distribution == "uniform":
            return self._rng.uniform(min_len, max_len)
        raise ValueError(f"Unsupported branch length distribution '{distribution}'")

    def _tree_to_newick(self, tree: BaseTree) -> str:
        with StringIO() as handle:
            phylo_write([tree], handle, "newick")
            return handle.getvalue()

    def _simulate_sequences(self, newick_tree: str) -> tuple[dict[str, str], bool]:
        simulation = self.config.simulation
        indel_rates = simulation.indel.rates if simulation.indel.enabled else None

        simulator = simulation.backend
        if simulator == "iqtree":
            sequences = self._simulate_with_iqtree(
                newick_tree,
                seq_length=self.config.sequence.length,
                model=self.config.sequence.model,
                indel_rate=indel_rates,
                iqtree_path=simulation.iqtree_path,
            )
        elif simulator == "seqgen":
            if indel_rates is not None:
                raise ValueError("Seq-Gen simulation does not support indel parameters")
            sequences = self._simulate_with_seqgen(
                newick_tree,
                seq_length=self.config.sequence.length,
                seqgen_path=simulation.seqgen_path,
                seqgen_kwargs=simulation.seqgen_kwargs,
            )
        else:  # pragma: no cover - guarded during config parsing
            raise ValueError(f"Unsupported simulator '{simulator}'")

        taxa = self.config.tree.taxa_labels
        ordered_sequences: dict[str, str] = {}
        for taxon in taxa:
            seq_value = sequences.get(taxon)
            if seq_value is None:
                raise RuntimeError(f"Simulator output missing sequence for taxon '{taxon}'")
            ordered_sequences[taxon] = seq_value

        unique_lengths = {len(value) for value in ordered_sequences.values()}
        aligned = len(unique_lengths) == 1
        return ordered_sequences, aligned

    def _attach_sequences(
        self,
        tree: BaseTree,
        sequences: Mapping[str, str],
        aligned: bool,
    ) -> Phylogeny:
        phylogeny = Phylogeny.from_tree(tree)
        for clade in phylogeny.get_terminals():
            seq_value = sequences.get(clade.name)
            if seq_value is None:
                continue
            phyloxml_sequence = PhyloXML.Sequence(type="dna")
            phyloxml_sequence.mol_seq = PhyloXML.MolSeq(seq_value, is_aligned=aligned)
            clade.sequences.append(phyloxml_sequence)
        return phylogeny

    def _annotate_topology(self, phylogeny: Phylogeny, topology: TopologySpec) -> None:
        topology_str = format_topology(topology)
        other_entry = Other(tag="topology", value=topology_str)
        existing = getattr(phylogeny, "other", None)
        if existing is None:
            phylogeny.other = [other_entry]
        else:
            phylogeny.other.append(other_entry)

    def _simulate_with_iqtree(
        self,
        newick_tree: str,
        seq_length: int,
        model: str,
        indel_rate: tuple[float, float] | None,
        iqtree_path: str | None,
    ) -> dict[str, str]:
        iqtree_exec = iqtree_path or "iqtree3"
        try:
            subprocess.run([iqtree_exec, "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            raise RuntimeError("IQ-TREE is not installed or not available at the specified path.") from error

        with tempfile.TemporaryDirectory(prefix="iqtree_sim_") as tmp_dir:
            tree_file = os.path.join(tmp_dir, "tree.nwk")
            with open(tree_file, "w", encoding="utf-8") as handle:
                handle.write(newick_tree)

            dummy_alignment = os.path.join(tmp_dir, "dummy.fa")
            with open(dummy_alignment, "w", encoding="utf-8") as handle:
                handle.write(">A\nA\n>B\nA\n")

            prefix = os.path.join(tmp_dir, "simulation")
            command = [
                iqtree_exec,
                "--alisim",
                f"{prefix}",
                "-t",
                tree_file,
                "--length",
                str(seq_length),
                "-m",
                model,
                "--seqtype",
                "DNA",
                "-af",
                "fasta",
                "--quiet",
            ]
            if indel_rate:
                ins_rate, del_rate = indel_rate
                command.extend(["--indel", f"{ins_rate},{del_rate}"])

            try:
                subprocess.run(command, check=True, capture_output=True)
            except subprocess.CalledProcessError as error:
                stderr = error.stderr.decode() if error.stderr else ""
                stdout = error.stdout.decode() if error.stdout else ""
                raise RuntimeError(
                    f"IQ-TREE simulation failed with exit code {error.returncode}.\n"
                    f"Command: {' '.join(command)}\n"
                    f"Stdout:\n{stdout}\nStderr:\n{stderr}"
                ) from error

            supported_ext = {
                ".fa": "fasta",
                ".fasta": "fasta",
                ".fas": "fasta",
                ".phy": "phylip",
                ".phylip": "phylip",
            }
            candidate_paths: list[tuple[str, str]] = []
            for filename in os.listdir(tmp_dir):
                if not filename.startswith(os.path.basename(prefix)):
                    continue
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported_ext:
                    candidate_paths.append((os.path.join(tmp_dir, filename), supported_ext[ext]))

            if not candidate_paths:
                raise RuntimeError(f"IQ-TREE did not produce a simulated alignment file in {tmp_dir}")

            aligned_sequences = None
            for path, fmt in sorted(candidate_paths, key=lambda item: item[0]):
                records = list(SeqIO.parse(path, fmt))
                if not records:
                    continue
                if aligned_sequences is None:
                    aligned_sequences = records
                if any("-" in str(record.seq) for record in records):
                    aligned_sequences = records
                    break

            if not aligned_sequences:
                raise RuntimeError("Unable to read aligned sequences produced by IQ-TREE")

            taxa = self.config.tree.taxa_labels
            seq_map = {record.id: str(record.seq) for record in aligned_sequences}
            fallback_iter = iter(aligned_sequences)
            ordered: dict[str, str] = {}
            for taxon in taxa:
                if taxon in seq_map:
                    ordered[taxon] = seq_map[taxon]
                    continue
                try:
                    ordered[taxon] = str(next(fallback_iter).seq)
                except StopIteration as exc:  # pragma: no cover - defensive guard
                    raise RuntimeError("IQ-TREE output does not contain enough sequences") from exc
            return ordered

    def _simulate_with_seqgen(
        self,
        newick_tree: str,
        seq_length: int,
        seqgen_path: str | None,
        seqgen_kwargs: Mapping[str, Any],
    ) -> dict[str, str]:
        seqgen_exec = seqgen_path or "seq-gen"
        if not os.path.isfile(seqgen_exec):
            raise RuntimeError(f"Seq-Gen executable not found at {seqgen_exec}")

        config = dict(seqgen_kwargs or {})
        ts_tv_ratio = float(config.get("ts_tv_ratio", config.get("tstv", 0.5)))
        frequencies = config.get("frequencies", (0.25, 0.25, 0.25, 0.25))
        replicates = int(config.get("replicates", 1))
        seed = config.get("seed")
        additional_args = config.get("additional_args", [])
        if isinstance(additional_args, (str, bytes)):
            raise ValueError("Seq-Gen additional_args must be an iterable of arguments, not a string.")
        additional_args_list = [str(arg) for arg in additional_args]

        if replicates != 1:
            raise ValueError("Seq-Gen simulation currently supports replicates=1 when streaming output.")

        if isinstance(frequencies, str):
            freq_arg = frequencies
        else:
            freq_values = tuple(frequencies)
            if len(freq_values) != 4:
                raise ValueError("Seq-Gen frequencies must contain exactly four values.")
            freq_arg = ",".join(str(value) for value in freq_values)

        command = [
            seqgen_exec,
            "-m",
            "HKY",
            f"-t{ts_tv_ratio}",
            f"-f{freq_arg}",
            "-l",
            str(seq_length),
            "-n",
            str(replicates),
            "-of",
        ]
        if seed is not None:
            command.extend(["-z", str(seed)])
        command.extend(additional_args_list)

        tree_input = newick_tree.strip()
        if not tree_input.endswith(";"):
            tree_input += ";"
        tree_input += "\n"

        with tempfile.TemporaryDirectory(prefix="seqgen_sim_") as tmp_dir:
            tree_path = Path(tmp_dir, "tree.nwk")
            tree_path.write_text(tree_input)

            command_with_tree = command + [str(tree_path)]

            try:
                result = subprocess.run(
                    command_with_tree,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=tmp_dir,
                )
            except subprocess.CalledProcessError as error:
                stderr = error.stderr or ""
                stdout = error.stdout or ""
                raise RuntimeError(
                    f"Seq-Gen simulation failed with exit code {error.returncode}.\n"
                    f"Command: {' '.join(command_with_tree)}\n"
                    f"Stdout:\n{stdout}\nStderr:\n{stderr}"
                ) from error

            fasta_output = result.stdout.strip()
            if not fasta_output:
                fasta_candidates = sorted(
                    path
                    for path in Path(tmp_dir).iterdir()
                    if path.suffix.lower() in {".fa", ".fasta", ".fas", ".fna"}
                )
                if not fasta_candidates:
                    raise RuntimeError("Seq-Gen did not produce any FASTA output.")
                fasta_output = fasta_candidates[0].read_text().strip()

            records = list(SeqIO.parse(StringIO(fasta_output), "fasta"))
            taxa = self.config.tree.taxa_labels
            if len(records) < len(taxa):
                raise RuntimeError("Seq-Gen output does not contain the expected sequences.")

            seq_map = {record.id.split()[0]: str(record.seq) for record in records}
            ordered: dict[str, str] = {}
            fallback_iter = iter(records)
            for taxon in taxa:
                if taxon in seq_map:
                    ordered[taxon] = seq_map[taxon]
                    continue
                try:
                    ordered[taxon] = str(next(fallback_iter).seq)
                except StopIteration as exc:  # pragma: no cover - defensive guard
                    raise RuntimeError("Seq-Gen output does not contain enough sequences.") from exc
            return ordered


__all__ = ["TreeSequenceGenerator", "TreeSequenceResult"]


def _generate_phylogeny_worker(payload: tuple[GenerationConfig, int, TopologySpec]) -> tuple[Phylogeny, bool]:
    config, seed, topology = payload
    seeded_config = config.with_seed(seed)
    generator = TreeSequenceGenerator(seeded_config)
    return generator.generate_phylogeny(topology=topology)
