from __future__ import annotations

from collections.abc import Sequence
from typing import Any


from src.utils.phylo import format_topology


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
 

