from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import random
import re
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterator, Optional

from Bio import SeqIO
from Bio.Phylo import PhyloXML
from Bio.Phylo._io import write as phylo_write
from Bio.Phylo.BaseTree import Clade
from Bio.Phylo.BaseTree import Tree as BaseTree
from Bio.Phylo.PhyloXML import Other, Phylogeny

from src.utils import flatten_topology, format_topology, infer_branch_output_count

from .config import (
    GenerationConfig,
    ModelParameterDistributionSettings,
    SequenceModelParameters,
    TopologySpec,
)


@dataclass
class TreeSequenceResult:
    """Container for the generated tree, sequences, and alignment metadata."""

    tree: BaseTree
    sequences: dict[str, str]
    aligned: bool
    topology: TopologySpec
    debug_metadata: dict[str, str] | None = None


class TreeSequenceGenerator:
    """Generate phylogenetic trees and associated sequences using reusable classes."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self.parallel_cores = (
            mp.cpu_count()
            if config.parallel_cores == 0
            else max(1, config.parallel_cores)
        )
        self._active_distribution: str | None = None
        self._last_sequence_command: str | None = None
        self._last_iqtree_log_metadata: dict[str, str] | None = None

    @classmethod
    def from_config_file(cls, config_path: Path | str) -> "TreeSequenceGenerator":
        from .config import load_generation_config

        config = load_generation_config(config_path)
        return cls(config)

    def generate_tree_and_sequences(
        self,
        topology: TopologySpec | None = None,
        distribution: str | None = None,
    ) -> TreeSequenceResult:
        previous = self._active_distribution
        self._active_distribution = distribution
        try:
            tree, used_topology = self._build_tree(topology_override=topology)
            newick_str = self._tree_to_newick(tree)
            sequences, aligned, debug_metadata = self._simulate_sequences(
                newick_str,
                distribution=distribution,
            )
            return TreeSequenceResult(
                tree=tree,
                sequences=sequences,
                aligned=aligned,
                topology=used_topology,
                debug_metadata=debug_metadata,
            )
        finally:
            self._active_distribution = previous

    def generate_phylogeny(
        self,
        topology: TopologySpec | None = None,
        distribution: str | None = None,
    ) -> tuple[Phylogeny, bool]:
        result = self.generate_tree_and_sequences(
            topology=topology, distribution=distribution
        )
        phylogeny = self._attach_sequences(
            result.tree, result.sequences, result.aligned
        )
        self._annotate_topology(phylogeny, result.topology)
        if result.debug_metadata:
            self._annotate_debug_metadata(phylogeny, result.debug_metadata)
        return phylogeny, result.aligned

    def generate_phylogenies(self) -> tuple[list[Phylogeny], bool]:
        tree_count = self.config.dataset.tree_count
        schedule = self._distribution_topology_schedule(tree_count)
        seeds = [self._rng.randint(0, 2**32 - 1) for _ in range(len(schedule))]

        phylogenies: list[Phylogeny] = []
        all_aligned = True

        if self.parallel_cores <= 1:
            for seed, (topology, distribution) in zip(seeds, schedule):
                phylogeny, aligned = _generate_phylogeny_worker(
                    (self.config, seed, topology, distribution)
                )
                phylogenies.append(phylogeny)
                all_aligned = all_aligned and aligned
        else:
            payloads: Iterable[tuple[GenerationConfig, int, TopologySpec, str]] = (
                (self.config, seed, topology, distribution)
                for seed, (topology, distribution) in zip(seeds, schedule)
            )
            # Cap pool size to avoid resource exhaustion on large systems.
            pool_size = min(self.parallel_cores, mp.cpu_count(), 64)
            ctx = mp.get_context("spawn")
            try:
                with ctx.Pool(processes=pool_size, maxtasksperchild=1) as pool:
                    for phylogeny, aligned in pool.imap(
                        _generate_phylogeny_worker, payloads, chunksize=1
                    ):
                        phylogenies.append(phylogeny)
                        all_aligned = all_aligned and aligned
            except Exception as exc:
                raise RuntimeError(
                    "Multiprocessing failed during phylogeny generation. "
                    "Try reducing 'parallel_cores' or running with parallel_cores=1."
                ) from exc

        return phylogenies, all_aligned

    def write_xml(self) -> Path:
        dataset_settings = self.config.dataset
        dataset_settings.ensure_xml_directory()
        output_path = dataset_settings.xml_path()
        chunk_size = dataset_settings.tree_chunk_size
        if chunk_size <= 0:
            raise ValueError("'dataset.tree_chunk_size' must be positive")

        footer: str | None = None
        with output_path.open("w", encoding="utf-8") as handle:
            for index, (phylogenies, _aligned) in enumerate(
                self._iter_phylogeny_chunks(chunk_size)
            ):
                phyloxml = PhyloXML.Phyloxml({})
                phyloxml.phylogenies = phylogenies
                xml_text = self._render_phyloxml(phyloxml)
                header, body, chunk_footer = self._split_phyloxml_document(xml_text)
                if index == 0:
                    handle.write(header)
                    handle.write(body)
                    footer = chunk_footer
                else:
                    handle.write(body)

            if footer is not None:
                handle.write(footer)
        return output_path

    def _iter_phylogeny_chunks(
        self, chunk_size: int
    ) -> Iterator[tuple[list[Phylogeny], bool]]:
        tree_count = self.config.dataset.tree_count
        schedule = self._distribution_topology_schedule(tree_count)
        seeds = [self._rng.randint(0, 2**32 - 1) for _ in range(len(schedule))]

        if self.parallel_cores <= 1:
            for start in range(0, len(schedule), chunk_size):
                end = start + chunk_size
                phylogenies: list[Phylogeny] = []
                chunk_aligned = True
                for seed, (topology, distribution) in zip(
                    seeds[start:end], schedule[start:end]
                ):
                    phylogeny, aligned = _generate_phylogeny_worker(
                        (self.config, seed, topology, distribution)
                    )
                    phylogenies.append(phylogeny)
                    chunk_aligned = chunk_aligned and aligned
                yield phylogenies, chunk_aligned
            return

        payloads_all = [
            (self.config, seed, topology, distribution)
            for seed, (topology, distribution) in zip(seeds, schedule)
        ]
        pool_size = min(self.parallel_cores, mp.cpu_count(), 64)
        ctx = mp.get_context("spawn")
        try:
            with ctx.Pool(processes=pool_size, maxtasksperchild=1) as pool:
                for start in range(0, len(payloads_all), chunk_size):
                    chunk_payloads = payloads_all[start : start + chunk_size]
                    phylogenies = []
                    chunk_aligned = True
                    for phylogeny, aligned in pool.imap(
                        _generate_phylogeny_worker, chunk_payloads, chunksize=1
                    ):
                        phylogenies.append(phylogeny)
                        chunk_aligned = chunk_aligned and aligned
                    yield phylogenies, chunk_aligned
        except Exception as exc:
            raise RuntimeError(
                "Multiprocessing failed during phylogeny generation. "
                "Try reducing 'parallel_cores' or running with parallel_cores=1."
            ) from exc

    @staticmethod
    def _render_phyloxml(phyloxml: PhyloXML.Phyloxml) -> str:
        buffer = StringIO()
        phylo_write(phyloxml, buffer, "phyloxml")
        return buffer.getvalue()

    @staticmethod
    def _split_phyloxml_document(xml_text: str) -> tuple[str, str, str]:
        open_match = re.search(r"<phyloxml[^>]*>", xml_text)
        close_match = re.search(r"</phyloxml\s*>", xml_text)
        if not open_match or not close_match:
            raise ValueError("Unable to locate phyloxml document boundaries")
        header = xml_text[: open_match.end()]
        body = xml_text[open_match.end() : close_match.start()]
        footer = xml_text[close_match.start() :]
        return header, body, footer

    def _build_tree(
        self, topology_override: TopologySpec | None = None
    ) -> tuple[BaseTree, TopologySpec]:
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
            raise ValueError(
                "Three-taxa configurations must reference exactly three taxa"
            )
        return self._build_tree_from_topology(topology)

    def _build_four_taxa_tree(self, topology: TopologySpec) -> BaseTree:
        if len(flatten_topology(topology)) != 4:
            raise ValueError(
                "Four-taxa configurations must reference exactly four taxa"
            )
        return self._build_tree_from_topology(topology)

    def _select_topology(self, taxa_count: int) -> TopologySpec:
        candidates = self._topology_candidates(taxa_count)
        return self._rng.choice(candidates)

    def _topology_candidates(self, taxa_count: int) -> list[TopologySpec]:
        configured = self.config.tree.topologies
        candidates = [
            topology
            for topology in configured
            if len(flatten_topology(topology)) == taxa_count
        ]
        if not candidates:
            raise ValueError(
                "No configured topologies match the requested taxa count of "
                f"{taxa_count}."
            )
        return candidates

    def _topology_schedule(self, tree_count: int) -> list[TopologySpec]:
        candidates = self._topology_candidates(self.config.tree.taxa_count)
        count = len(candidates)
        return [candidates[index % count] for index in range(tree_count)]

    def _distribution_topology_schedule(
        self, tree_count: int
    ) -> list[tuple[TopologySpec, str]]:
        candidates = self._topology_candidates(self.config.tree.taxa_count)
        if not candidates:
            return []

        distributions = list(self.config.tree.branch_length_distributions)
        if not distributions:
            raise ValueError("No branch length distributions configured")

        topology_count = len(candidates)
        scheduled: list[tuple[TopologySpec, str]] = []

        # Round each distribution target up, then round again to full topology cycles.
        # This keeps distribution-topology combinations balanced and can exceed tree_count.
        for dist_name, weight in distributions:
            target = max(1, math.ceil(weight * tree_count))
            per_topology = math.ceil(target / topology_count)
            for topology in candidates:
                for _ in range(per_topology):
                    scheduled.append((topology, dist_name))

        self._rng.shuffle(scheduled)
        return scheduled

    def _build_tree_from_topology(self, topology: TopologySpec) -> BaseTree:
        if not topology:
            raise ValueError("Topology definitions must include at least one group")

        group_clades = [self._build_group_clade(group) for group in topology]

        if self.config.tree.rooted:
            if topology.root_index is None:
                raise ValueError(
                    "Rooted tree configurations require ':' in the topology definition"
                )
            left_groups = tuple(group_clades[: topology.root_index + 1])
            right_groups = tuple(group_clades[topology.root_index + 1 :])
            if not left_groups or not right_groups:
                raise ValueError(
                    "Rooted topologies must include taxa on both sides of ':'"
                )
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
        if self.config.tree.rooted and not self.config.tree.split_root_branch:
            segment_count = infer_branch_output_count(num_taxa, rooted=True)
            if segment_count <= 0:
                return
            samples = [self._sample_branch_length() for _ in range(segment_count)]
            self._assign_rooted_no_split(root, iter(samples))
            return

        segment_count = infer_branch_output_count(num_taxa, rooted=False)
        if segment_count <= 0:
            return

        samples = [self._sample_branch_length() for _ in range(segment_count)]
        length_iter = iter(samples)
        flattened = flatten_topology(topology)
        first_taxon = flattened[0] if flattened else None

        if self.config.tree.rooted:
            self._assign_rooted_from_unrooted(root, topology, length_iter, first_taxon)
            return

        self._assign_unrooted_branch_lengths(root, length_iter, first_taxon)

    def _assign_rooted_from_unrooted(
        self,
        root: Clade,
        topology: TopologySpec,
        length_iter: Iterator[float],
        first_taxon: str | None,
    ) -> None:
        children = list(root.clades)
        if len(children) != 2:
            self._assign_unrooted_branch_lengths(root, length_iter, first_taxon)
            return

        try:
            connector_length = next(length_iter)
        except StopIteration as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(
                "Missing branch length sample for rooted split connector"
            ) from exc

        target_child = self._root_side_child(root, topology)
        if target_child is None:
            target_child = self._child_containing_taxon(root, first_taxon)

        if target_child is children[0]:
            children[0].branch_length = connector_length
            children[1].branch_length = None
        elif target_child is children[1]:
            children[0].branch_length = None
            children[1].branch_length = connector_length
        else:
            children[0].branch_length = connector_length
            children[1].branch_length = None

        for child in children:
            self._populate_branch_lengths(child, length_iter)

        self._split_root_children(
            root,
            connector_length,
            target_child=target_child,
            enforce_min=True,
        )

    def _assign_small_tree_branch_lengths(
        self, root: Clade, topology: TopologySpec, segment_count: int
    ) -> None:
        samples = [self._sample_branch_length() for _ in range(segment_count)]

        if self.config.tree.taxa_count == 2:
            self._assign_two_taxa_branch_lengths(root, samples)
            return

        if not self.config.tree.rooted:
            flattened = flatten_topology(topology)
            first_taxon = flattened[0] if flattened else None
            self._assign_unrooted_branch_lengths(root, iter(samples), first_taxon)
            return

        connector_length, *remaining = samples
        target_child = self._root_side_child(root, topology)
        self._split_root_children(
            root,
            connector_length,
            target_child=target_child,
            enforce_min=True,
        )

        if remaining:
            self._populate_branch_lengths(root, iter(remaining))

    def _assign_two_taxa_branch_lengths(
        self, root: Clade, samples: list[float]
    ) -> None:
        if not samples:
            return
        connector_length = samples[0]
        children = list(root.clades)
        if not children:
            return

        if len(children) == 1:
            children[0].branch_length = connector_length
            return

        preferred = self._child_containing_taxon(root, self.config.tree.taxa_labels[0])
        if preferred is None:
            preferred = children[0]

        if self.config.tree.rooted:
            target_child = preferred if preferred in children else None
            self._split_root_children(
                root,
                connector_length,
                target_child=target_child,
                enforce_min=True,
            )
            return

        preferred.branch_length = connector_length

    def _split_root_children(
        self,
        root: Clade,
        connector_length: float,
        *,
        target_child: Clade | None,
        enforce_min: bool,
    ) -> None:
        children = list(root.clades)
        if not children:
            return

        if len(children) == 1:
            children[0].branch_length = connector_length
            return

        split_value, remainder = self._split_length(
            connector_length, enforce_min=enforce_min
        )

        if target_child is None:
            if self._rng.random() < 0.5:
                children[0].branch_length = split_value
                children[1].branch_length = remainder
            else:
                children[0].branch_length = remainder
                children[1].branch_length = split_value
            return

        if target_child is children[0]:
            children[0].branch_length = split_value
            children[1].branch_length = remainder
        elif target_child is children[1]:
            children[0].branch_length = remainder
            children[1].branch_length = split_value
        else:
            children[0].branch_length = remainder
            children[1].branch_length = split_value

    def _split_length(self, total: float, *, enforce_min: bool) -> tuple[float, float]:
        min_len = self.config.tree.min_branch_length
        lower_bound = min_len if enforce_min else 0.0
        upper_bound = total
        if upper_bound < lower_bound:
            lower_bound = upper_bound
        if upper_bound == lower_bound:
            first = upper_bound
        else:
            first = self._sample_uniform(lower_bound, upper_bound)
        second = max(total - first, 0.0)
        return first, second

    def _root_side_child(self, root: Clade, topology: TopologySpec) -> Clade | None:
        if topology.root_index is None:
            return None
        children = list(root.clades)
        if len(children) < 2:
            return None

        right_taxa = set(flatten_topology(topology.tokens[topology.root_index + 1 :]))
        for child in children:
            if any(leaf.name in right_taxa for leaf in child.get_terminals()):
                return child
        return None

    def _assign_rooted_branch_lengths(
        self, root: Clade, length_iter: Iterator[float]
    ) -> None:
        children = list(root.clades)
        if len(children) != 2:
            # Degenerate rooted trees (e.g., single taxon) fall back to the unrooted logic.
            self._assign_unrooted_branch_lengths(root, length_iter, None)
            return

        try:
            connector_length = next(length_iter)
        except StopIteration as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(
                "Insufficient branch length samples for root connector"
            ) from exc

        min_len = self.config.tree.min_branch_length
        lower_bound = min_len
        upper_bound = connector_length
        if upper_bound < lower_bound:
            lower_bound = upper_bound
        split_value = (
            upper_bound
            if upper_bound == lower_bound
            else self._sample_uniform(lower_bound, upper_bound)
        )
        remainder = max(connector_length - split_value, 0.0)

        if self._rng.random() < 0.5:
            children[0].branch_length = split_value
            children[1].branch_length = remainder
        else:
            children[0].branch_length = remainder
            children[1].branch_length = split_value

        for child in children:
            self._populate_branch_lengths(child, length_iter)

    def _assign_rooted_no_split(
        self, root: Clade, length_iter: Iterator[float]
    ) -> None:
        children = list(root.clades)
        if len(children) < 2:
            self._assign_unrooted_branch_lengths(root, length_iter, None)
            return

        for child in children:
            try:
                child.branch_length = next(length_iter)
            except StopIteration as exc:  # pragma: no cover - defensive guard
                raise RuntimeError(
                    "Missing branch length sample for rooted tree"
                ) from exc

        self._populate_branch_lengths(root, length_iter)

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
                raise RuntimeError(
                    "Missing branch length sample for two-taxa unrooted tree"
                ) from exc
            return

        children = list(root.clades)
        if len(children) == 2:
            try:
                connector_length = next(length_iter)
            except StopIteration as exc:  # pragma: no cover - defensive guard
                raise RuntimeError(
                    "Missing branch length sample for unrooted connector"
                ) from exc

            implicit_child = self._select_unrooted_implicit_child(root, first_taxon)
            if implicit_child is children[0]:
                children[0].branch_length = None
                children[1].branch_length = connector_length
            elif implicit_child is children[1]:
                children[0].branch_length = connector_length
                children[1].branch_length = None
            else:
                # If selection fails unexpectedly, keep behavior deterministic.
                children[0].branch_length = connector_length
                children[1].branch_length = None

            # For unrooted trees represented with two root children, keep one side
            # implicit (no branch length) and only populate descendant edges.
            for child in children:
                self._populate_branch_lengths(child, length_iter)
            return

        self._populate_branch_lengths(root, length_iter)

    def _select_unrooted_implicit_child(
        self, root: Clade, first_taxon: str | None
    ) -> Clade | None:
        children = list(root.clades)
        if len(children) != 2:
            return None

        left_count = len(children[0].get_terminals())
        right_count = len(children[1].get_terminals())
        if left_count > right_count:
            return children[0]
        if right_count > left_count:
            return children[1]

        preferred = self._child_containing_taxon(root, first_taxon)
        return preferred if preferred is not None else children[0]

    def _populate_branch_lengths(
        self, clade: Clade, length_iter: Iterator[float]
    ) -> None:
        for child in clade.clades:
            if child.branch_length is None:
                try:
                    child.branch_length = next(length_iter)
                except StopIteration as exc:  # pragma: no cover - defensive guard
                    raise RuntimeError(
                        "Ran out of sampled branch lengths while assigning the tree"
                    ) from exc
            self._populate_branch_lengths(child, length_iter)

    def _child_containing_taxon(self, root: Clade, taxon: str | None) -> Clade | None:
        if taxon is None:
            return None
        for child in root.clades:
            if any(leaf.name == taxon for leaf in child.get_terminals()):
                return child
        return None

    def _sample_branch_length(self) -> float:
        """Sample a branch length from the configured distribution mix."""
        distributions = self.config.tree.branch_length_distributions
        params = self.config.tree.branch_length_params

        if not distributions:
            raise ValueError("No branch length distributions configured")

        selected_dist = self._active_distribution
        if selected_dist is None:
            roll = self._rng.random()
            cumulative = 0.0
            for dist_name, weight in distributions:
                cumulative += weight
                if roll <= cumulative:
                    selected_dist = dist_name
                    break
            if selected_dist is None:
                selected_dist = distributions[-1][0]

        dist_params = params.get(selected_dist, {})

        if selected_dist == "uniform":
            range_vals = dist_params.get("range")
            if not range_vals or len(range_vals) != 2:
                raise ValueError(
                    "uniform distribution requires 'range' parameter with two values"
                )
            min_val, max_val = range_vals
            return self._sample_uniform(min_val, max_val)

        if selected_dist == "exponential":
            rate = dist_params.get("rate")
            if rate is None or rate <= 0:
                raise ValueError(
                    "exponential distribution requires positive 'rate' parameter"
                )
            return self._sample_exponential(rate)

        if selected_dist == "truncated_exponential":
            return self._sample_truncated_exponential(dist_params)

        if selected_dist == "normal":
            mean = dist_params.get("mean")
            variance = dist_params.get("variance")
            if mean is None or variance is None:
                raise ValueError(
                    "normal distribution requires 'mean' and 'variance' parameters"
                )
            min_bound = dist_params.get("min")
            max_bound = dist_params.get("max")
            min_bound_f = None if min_bound is None else float(min_bound)
            max_bound_f = None if max_bound is None else float(max_bound)
            return self._sample_normal(
                float(mean),
                float(variance),
                min_bound_f,
                max_bound_f,
            )

        raise ValueError(f"Unsupported branch length distribution '{selected_dist}'")

    def _sample_truncated_exponential(self, params: Mapping[str, Any]) -> float:
        rate = params.get("rate")
        min_val = params.get("min", 0.0)
        max_val = params.get("max")

        if rate is None or rate <= 0:
            raise ValueError("truncated_exponential requires positive 'rate' parameter")
        if max_val is None or max_val <= 0:
            raise ValueError("truncated_exponential requires positive 'max' parameter")
        if min_val < 0 or min_val >= max_val:
            raise ValueError("truncated_exponential 'min' must be >= 0 and < 'max'")

        span = max_val - min_val
        u = self._rng.random()
        scale = 1.0 - math.exp(-rate * span)
        if scale <= 0:
            return min_val
        return min_val - (1.0 / rate) * math.log(1.0 - u * scale)

    def _sample_uniform(self, lower: float, upper: float) -> float:
        if upper <= lower:
            raise ValueError("uniform draw requires upper > lower")
        return self._rng.uniform(lower, upper)

    def _sample_exponential(self, rate: float) -> float:
        if rate <= 0:
            raise ValueError("exponential requires positive rate")
        return self._rng.expovariate(rate)

    def _sample_normal(
        self,
        mean: float,
        variance: float,
        min_bound: Optional[float] = None,
        max_bound: Optional[float] = None,
    ) -> float:
        if variance < 0:
            raise ValueError("variance must be non-negative")
        std_dev = math.sqrt(variance)

        # Degenerate case: zero variance
        if std_dev == 0:
            if (min_bound is not None and mean < min_bound) or (
                max_bound is not None and mean > max_bound
            ):
                raise ValueError(
                    "variance is zero and mean lies outside the provided bounds (no possible sample)."
                )
            return mean

        max_rounds = 10_000
        rounds = 0
        while True:
            val = self._rng.gauss(mean, std_dev)
            if (min_bound is None or val >= min_bound) and (
                max_bound is None or val <= max_bound
            ):
                return val
            rounds += 1
            if rounds >= max_rounds:
                raise RuntimeError(
                    f"Exceeded maximum resampling rounds ({max_rounds}). Bounds too restrictive."
                )

    # Model-parameter drawing uses the single-sample `_sample_*` helpers.

    def _sample_model_parameters(
        self, model_parameters: SequenceModelParameters
    ) -> tuple[float, ...]:
        if model_parameters.fixed_parameters is not None:
            return model_parameters.fixed_parameters

        distribution = model_parameters.parameter_distribution
        if distribution is None:
            return tuple()

        return tuple(self._draw_model_parameter_values(distribution))

    def _draw_model_parameter_values(
        self, distribution: ModelParameterDistributionSettings
    ) -> list[float]:
        draw_count = distribution.draw_count
        params = distribution.parameters

        if distribution.distribution_name == "uniform":
            lower, upper = params["range"]
            return [
                self._sample_uniform(float(lower), float(upper))
                for _ in range(draw_count)
            ]

        if distribution.distribution_name == "exponential":
            rate = float(params["rate"])
            return [self._sample_exponential(rate) for _ in range(draw_count)]

        if distribution.distribution_name == "truncated_exponential":
            return [
                self._sample_truncated_exponential(params) for _ in range(draw_count)
            ]

        if distribution.distribution_name == "normal":
            mean = float(params["mean"])
            variance = float(params["variance"])
            min_b = params.get("min")
            max_b = params.get("max")
            min_b_f = None if min_b is None else float(min_b)
            max_b_f = None if max_b is None else float(max_b)
            return [
                self._sample_normal(mean, variance, min_b_f, max_b_f)
                for _ in range(draw_count)
            ]

        raise ValueError(
            f"Unsupported model parameter distribution '{distribution.distribution_name}'"
        )

    def _format_iqtree_model(
        self,
        model: str,
        model_parameters: SequenceModelParameters | None,
        *,
        parameter_values: tuple[float, ...] | None = None,
    ) -> str:
        if model_parameters is None:
            return model

        if parameter_values is None:
            parameter_values = self._sample_model_parameters(model_parameters)
        if not parameter_values:
            return model

        rounded_values = [round(value, 6) for value in parameter_values]
        parameters_string = "/".join(str(value) for value in rounded_values)
        return f"{model}{{{parameters_string}}}"

    def _tree_to_newick(self, tree: BaseTree) -> str:
        with StringIO() as handle:
            phylo_write([tree], handle, "newick")
            return handle.getvalue()

    def _simulate_sequences(
        self,
        newick_tree: str,
        *,
        distribution: str | None = None,
    ) -> tuple[dict[str, str], bool, dict[str, str] | None]:
        simulation = self.config.simulation
        indel_rates = simulation.indel.rates if simulation.indel.enabled else None
        indel_sizes = simulation.indel.sizes if simulation.indel.enabled else None
        debug_metadata: dict[str, str] | None = None

        simulator = simulation.backend
        if simulator == "iqtree":
            model_parameter_values = None
            if self.config.debug and self.config.sequence.model_parameters is not None:
                model_parameter_values = self._sample_model_parameters(
                    self.config.sequence.model_parameters
                )
            result = self._simulate_with_iqtree(
                newick_tree,
                seq_length=self.config.sequence.length,
                model=self.config.sequence.model,
                model_parameters=self.config.sequence.model_parameters,
                model_parameter_values=model_parameter_values,
                indel_rate=indel_rates,
                indel_size=indel_sizes,
                iqtree_path=simulation.iqtree_path,
            )
            if isinstance(result, tuple):
                sequences, sequence_command = result
            else:
                sequences = result
                sequence_command = None

            if self.config.debug:
                debug_metadata = self._build_debug_metadata(
                    distribution=distribution,
                    newick=newick_tree.strip(),
                    sequence_command=sequence_command,
                    iqtree_log_metadata=getattr(
                        self, "_last_iqtree_log_metadata", None
                    ),
                )
        elif simulator == "seqgen":
            if indel_rates is not None or indel_sizes is not None:
                raise ValueError("Seq-Gen simulation does not support indel parameters")
            result = self._simulate_with_seqgen(
                newick_tree,
                seq_length=self.config.sequence.length,
                seqgen_path=simulation.seqgen_path,
                seqgen_kwargs=simulation.seqgen_kwargs,
            )
            if isinstance(result, tuple):
                sequences, sequence_command = result
            else:
                sequences = result
                sequence_command = getattr(self, "_last_sequence_command", None)

            if self.config.debug:
                debug_metadata = self._build_debug_metadata(
                    distribution=distribution,
                    newick=newick_tree.strip(),
                    sequence_command=sequence_command,
                )
        else:  # pragma: no cover - guarded during config parsing
            raise ValueError(f"Unsupported simulator '{simulator}'")

        taxa = self.config.tree.taxa_labels
        ordered_sequences: dict[str, str] = {}
        for taxon in taxa:
            seq_value = sequences.get(taxon)
            if seq_value is None:
                raise RuntimeError(
                    f"Simulator output missing sequence for taxon '{taxon}'"
                )
            ordered_sequences[taxon] = seq_value

        unique_lengths = {len(value) for value in ordered_sequences.values()}
        aligned = len(unique_lengths) == 1
        return ordered_sequences, aligned, debug_metadata

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

    def _annotate_debug_metadata(
        self, phylogeny: Phylogeny, metadata: Mapping[str, str]
    ) -> None:
        entries = [Other(tag=tag, value=value) for tag, value in metadata.items()]
        existing = getattr(phylogeny, "other", None)
        if existing is None:
            phylogeny.other = entries
        else:
            phylogeny.other.extend(entries)

    def _build_debug_metadata(
        self,
        *,
        distribution: str | None,
        newick: str | None = None,
        sequence_command: str | None = None,
        iqtree_log_metadata: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        metadata: dict[str, str] = {}

        if distribution is not None:
            metadata["branch_length_distribution"] = distribution

        if newick is not None:
            metadata["newick"] = newick

        if sequence_command is not None:
            metadata["sequence_command"] = sequence_command

        if iqtree_log_metadata:
            metadata.update(iqtree_log_metadata)

        return metadata

    @staticmethod
    def _serialize_debug_value(value: Any) -> str:
        return json.dumps(_jsonify(value), separators=(",", ":"), sort_keys=True)

    def _simulate_with_iqtree(
        self,
        newick_tree: str,
        seq_length: int,
        model: str,
        model_parameters: SequenceModelParameters | None,
        model_parameter_values: tuple[float, ...] | None = None,
        indel_rate: tuple[float, float] | None = None,
        indel_size: tuple[str, str] | None = None,
        iqtree_path: str | None = None,
    ) -> tuple[dict[str, str], str]:
        iqtree_exec = iqtree_path or "iqtree3"
        try:
            subprocess.run([iqtree_exec, "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            raise RuntimeError(
                "IQ-TREE is not installed or not available at the specified path."
            ) from error

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
                self._format_iqtree_model(
                    model,
                    model_parameters,
                    parameter_values=model_parameter_values,
                ),
                "--seqtype",
                "DNA",
                "-af",
                "fasta",
                "--quiet",
            ]
            if indel_rate:
                ins_rate, del_rate = indel_rate
                command.extend(["--indel", f"{ins_rate},{del_rate}"])
            if indel_size:
                ins_size, del_size = indel_size
                command.extend(["--indel-size", f"{ins_size},{del_size}"])

            command_str = " ".join(command)
            try:
                subprocess.run(command, check=True, capture_output=True)
            except subprocess.CalledProcessError as error:
                stderr = error.stderr.decode() if error.stderr else ""
                stdout = error.stdout.decode() if error.stdout else ""
                raise RuntimeError(
                    f"IQ-TREE simulation failed with exit code {error.returncode}.\n"
                    f"Command: {command_str}\n"
                    f"Stdout:\n{stdout}\nStderr:\n{stderr}"
                ) from error

            log_path = Path(f"{tree_file}.log")
            if log_path.exists():
                self._last_iqtree_log_metadata = self._parse_iqtree_log_metadata(
                    log_path.read_text(encoding="utf-8")
                )
            else:
                self._last_iqtree_log_metadata = None

            self._last_sequence_command = command_str

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
                    candidate_paths.append(
                        (os.path.join(tmp_dir, filename), supported_ext[ext])
                    )

            if not candidate_paths:
                raise RuntimeError(
                    f"IQ-TREE did not produce a simulated alignment file in {tmp_dir}"
                )

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
                raise RuntimeError(
                    "Unable to read aligned sequences produced by IQ-TREE"
                )

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
                    raise RuntimeError(
                        "IQ-TREE output does not contain enough sequences"
                    ) from exc
            return ordered, command_str

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
            raise ValueError(
                "Seq-Gen additional_args must be an iterable of arguments, not a string."
            )
        additional_args_list = [str(arg) for arg in additional_args]

        if replicates != 1:
            raise ValueError(
                "Seq-Gen simulation currently supports replicates=1 when streaming output."
            )

        if isinstance(frequencies, str):
            freq_arg = frequencies
        else:
            freq_values = tuple(frequencies)
            if len(freq_values) != 4:
                raise ValueError(
                    "Seq-Gen frequencies must contain exactly four values."
                )
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
                raise RuntimeError(
                    "Seq-Gen output does not contain the expected sequences."
                )

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
                    raise RuntimeError(
                        "Seq-Gen output does not contain enough sequences."
                    ) from exc
            self._last_sequence_command = " ".join(command_with_tree)
            return ordered

    @staticmethod
    def _parse_iqtree_log_metadata(log_text: str) -> dict[str, str]:
        metadata: dict[str, str] = {}

        model_match = re.search(r"^\s*-\s*Model:\s*(.+)$", log_text, re.MULTILINE)
        if model_match:
            metadata["model"] = model_match.group(1).strip()

        seed_match = re.search(r"^Seed:\s+(\d+)\b", log_text, re.MULTILINE)
        if seed_match:
            metadata["seed"] = seed_match.group(1)

        state_frequencies = TreeSequenceGenerator._parse_iqtree_state_frequencies(
            log_text
        )
        if state_frequencies:
            metadata["state_frequencies"] = (
                TreeSequenceGenerator._serialize_debug_value(state_frequencies)
            )

        rate_matrix = TreeSequenceGenerator._parse_iqtree_rate_matrix(log_text)
        if rate_matrix:
            metadata["rate_matrix"] = TreeSequenceGenerator._serialize_debug_value(
                rate_matrix
            )

        return metadata

    @staticmethod
    def _parse_iqtree_state_frequencies(log_text: str) -> dict[str, float]:
        lines = log_text.splitlines()
        in_block = False
        frequencies: dict[str, float] = {}
        for line in lines:
            if line.strip().startswith("State frequencies:"):
                in_block = True
                continue
            if in_block and line.strip().startswith("Rate matrix Q:"):
                break
            if not in_block:
                continue
            match = re.match(
                r"^\s*pi\(([A-Za-z])\)\s*=\s*([+-]?\d+(?:\.\d+)?)\s*$", line
            )
            if match:
                frequencies[match.group(1)] = float(match.group(2))
        return frequencies

    @staticmethod
    def _parse_iqtree_rate_matrix(log_text: str) -> dict[str, dict[str, float]]:
        lines = log_text.splitlines()
        in_block = False
        row_labels = ["A", "C", "G", "T"]
        matrix: dict[str, dict[str, float]] = {}

        for line in lines:
            if line.strip().startswith("Rate matrix Q:"):
                in_block = True
                continue
            if not in_block:
                continue

            if line.strip().startswith("Model of rate heterogeneity:"):
                break

            match = re.match(r"^\s*([ACGT])\s+(.+?)\s*$", line)
            if not match:
                continue

            row_label = match.group(1)
            values = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", match.group(2))
            if len(values) != 4:
                continue
            matrix[row_label] = {
                col_label: float(values[index])
                for index, col_label in enumerate(row_labels)
            }

        return matrix


__all__ = ["TreeSequenceGenerator", "TreeSequenceResult"]


def _generate_phylogeny_worker(
    payload: tuple[GenerationConfig, int, TopologySpec, str],
) -> tuple[Phylogeny, bool]:
    config, seed, topology, distribution = payload
    seeded_config = config.with_seed(seed)
    generator = TreeSequenceGenerator(seeded_config)
    return generator.generate_phylogeny(topology=topology, distribution=distribution)


def _jsonify(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    return value
