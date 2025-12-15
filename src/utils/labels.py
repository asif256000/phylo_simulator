from __future__ import annotations

from collections.abc import Sequence, Iterable
from typing import Any


from src.utils.phylo import format_topology


def get_clade_set(clade_tokens: Any) -> frozenset[str]:
    """Recursively extract all leaf labels from a nested topology structure."""
    if isinstance(clade_tokens, str):
        return frozenset([clade_tokens])

    leaves = []
    if isinstance(clade_tokens, Iterable):
        for child in clade_tokens:
            leaves.extend(get_clade_set(child))
    return frozenset(leaves)


def extract_splits_from_topology(topology_tokens: Any) -> set[frozenset[str]]:
    """
    Extracts all splits (clades) from a nested topology structure.
    Returns a set of frozensets, where each frozenset represents a clade (set of leaves).
    Excludes the full set (root) if it appears.
    """
    splits = set()

    def _recurse(tokens):
        current_leaves = set()
        for child in tokens:
            if isinstance(child, str):
                child_leaves = frozenset([child])
            else:
                child_leaves = _recurse(child)

            splits.add(child_leaves)
            current_leaves.update(child_leaves)
        return frozenset(current_leaves)

    # If the top level is a tuple of groups (as in TopologySpec.tokens)
    if isinstance(topology_tokens, tuple):
        all_leaves = set()
        for group in topology_tokens:
            # Each group in the top-level tuple is a clade under the root
            # But wait, the structure in config seems to be:
            # tokens: (('A', 'B'), ('C',))
            # This implies root has children (A,B) and C.
            # So (A,B) is a split. C is a split.
            # And A, B are splits.
            group_leaves = _recurse(group)
            splits.add(group_leaves)
            all_leaves.update(group_leaves)
        # We don't add all_leaves because that's the root
    else:
        _recurse(topology_tokens)

    return splits


def canonical_branch_ordering(
    taxa: Sequence[str],
    topologies: Sequence[Any],
) -> list[frozenset[str]]:
    """
    Defines a canonical ordering of branches (splits) based on the provided topologies.
    Always includes the trivial splits (leaves) first, in the order of `taxa`.
    Then includes all unique internal splits found in `topologies`, sorted by size and then lexicographically.
    """
    # 1. Trivial splits (leaves)
    ordered_slots = [frozenset([t]) for t in taxa]
    seen_splits = set(ordered_slots)

    # 2. Internal splits from topologies
    internal_splits = set()
    for topo in topologies:
        # topo is expected to be TopologySpec or similar, having 'tokens'
        tokens = getattr(topo, "tokens", topo)
        splits = extract_splits_from_topology(tokens)
        for s in splits:
            if s not in seen_splits:
                internal_splits.add(s)

    # Sort internal splits
    # Sort by number of taxa (size), then by the names of taxa (joined string)
    sorted_internal = sorted(
        list(internal_splits),
        key=lambda s: (len(s), sorted(list(s)))
    )

    return ordered_slots + sorted_internal


def get_topology_map(topologies: Sequence[Any]) -> dict[str, int]:
    """
    Returns a mapping from topology string representation to index.
    """
    mapping = {}
    for idx, topo in enumerate(topologies):
        # We assume the topology string in XML matches format_topology(topo)
        topo_str = format_topology(topo)
        mapping[topo_str] = idx
    return mapping
 
