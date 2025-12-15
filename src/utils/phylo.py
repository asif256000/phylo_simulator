from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def infer_branch_output_count(num_taxa: int, *, rooted: bool = True) -> int:
    """Infer the number of branch-length targets for a tree with ``num_taxa`` tips."""
    if num_taxa <= 0:
        raise ValueError("num_taxa must be positive when inferring branch outputs")
    if num_taxa == 1:
        return 1
    if num_taxa == 2:
        return 2 if rooted else 1
    if rooted:
        return 2 * num_taxa - 2
    return 2 * num_taxa - 3


def infer_num_outputs(
    num_taxa: int,
    *,
    rooted: bool = True,
    include_topologies: bool = False,
    topology_classes: int | None = None,
) -> int:
    """Infer regression + optional topology-class outputs for a tree.

    Parameters
    ----------
    num_taxa:
        Total number of taxa/leaves in the dataset.
    rooted:
        Whether the underlying tree is rooted. Controls the number of branch-length
        parameters inferred for the regression head.
    include_topologies:
        When ``True``, add ``topology_classes`` units reserved for topology
        classification logits.
    topology_classes:
        Number of topology categories to append when ``include_topologies`` is set.
    """

    outputs = infer_branch_output_count(num_taxa, rooted=rooted)
    if include_topologies:
        if topology_classes is None or topology_classes <= 0:
            raise ValueError("topology_classes must be a positive integer when include_topologies=True")
        outputs += topology_classes
    return outputs


def flatten_topology(topology: Any) -> tuple[str, ...]:
    """Flatten a ``TopologySpec`` (or compatible iterable) into a tuple of taxa labels."""
    groups = _topology_groups(topology)
    return tuple(taxon for group in groups for taxon in group)


def format_topology(topology: Any) -> str:
    """Render a topology as a Newick-like string with explicit root markers."""
    groups = _topology_groups(topology)
    if not groups:
        return ""

    root_index = getattr(topology, "root_index", None)
    if root_index is None:
        return _chain_to_string(groups)

    if root_index < 0 or root_index >= len(groups) - 1:
        raise ValueError("Invalid root_index stored on TopologySpec")

    left_groups = groups[: root_index + 1]
    right_groups = groups[root_index + 1 :]
    left_payload = _chain_to_string(left_groups)
    right_payload = _chain_to_string(right_groups)
    return f"({left_payload},:{right_payload})"


def _topology_groups(topology: Any) -> tuple[tuple[str, ...], ...]:
    if hasattr(topology, "tokens"):
        tokens = getattr(topology, "tokens")
    else:
        tokens = topology

    if isinstance(tokens, tuple):
        return tuple(tuple(group) for group in tokens)

    if not isinstance(tokens, Iterable):
        raise TypeError("Topology tokens must be iterable")

    groups: list[tuple[str, ...]] = []
    for group in tokens:
        if not isinstance(group, Iterable):
            raise TypeError("Topology groups must be iterable")
        groups.append(tuple(str(label) for label in group))
    return tuple(groups)


def _chain_to_string(groups: Sequence[tuple[str, ...]]) -> str:
    if not groups:
        raise ValueError("Topology groups cannot be empty when formatting")

    rendered = _group_to_string(groups[-1])
    for group in reversed(groups[:-1]):
        rendered = f"({_group_to_string(group)},{rendered})"
    return rendered


def _group_to_string(group: tuple[str, ...]) -> str:
    if not group:
        raise ValueError("Empty topology group encountered")
    if len(group) == 1:
        return group[0]
    if len(group) == 2:
        return f"({group[0]},{group[1]})"
    raise ValueError("Topology groups may only contain one or two taxa")


__all__ = [
    "infer_branch_output_count",
    "infer_num_outputs",
    "flatten_topology",
    "format_topology",
]
