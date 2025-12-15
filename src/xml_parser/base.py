from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

NUCLEOTIDE_ORDER: tuple[str, ...] = ("A", "T", "G", "C")
GAP_SYMBOL: str = "-"
NUCLEOTIDE_TO_INDEX: Mapping[str, int] = {nt: idx for idx, nt in enumerate(NUCLEOTIDE_ORDER)}
NUCLEOTIDE_TO_INDEX_WITH_GAP: Mapping[str, int] = {
    **NUCLEOTIDE_TO_INDEX,
    GAP_SYMBOL: len(NUCLEOTIDE_ORDER),
}


def nucleotide_channel_count(include_gap: bool) -> int:
    return len(NUCLEOTIDE_ORDER) + (1 if include_gap else 0)


@dataclass
class CladeRecord:
    """Container for clade metadata extracted from a phylogeny."""

    name: str
    sequence: str
    branch_length: float


@dataclass
class TreeExample:
    """Represents one training example derived from a single phylogeny."""

    tree_index: int
    clades: Sequence[CladeRecord]
    branches: Mapping[frozenset[str], float]
    metadata: Mapping[str, str] | None = None


def one_hot_encode(sequence: str, seq_length: int, *, include_gap: bool = False) -> np.ndarray:
    """Convert a nucleotide string into an LxC one-hot matrix using PyTorch."""
    if len(sequence) != seq_length:
        raise ValueError(f"Sequence length mismatch: expected {seq_length}, observed {len(sequence)}")

    mapping = NUCLEOTIDE_TO_INDEX_WITH_GAP if include_gap else NUCLEOTIDE_TO_INDEX
    num_classes = nucleotide_channel_count(include_gap)
    try:
        indices = [mapping[symbol] for symbol in sequence]
    except KeyError as exc:
        raise ValueError(f"Unsupported nucleotide encountered: {exc.args[0]!r}") from exc

    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    one_hot = F.one_hot(index_tensor, num_classes=num_classes)
    return one_hot.to(dtype=torch.uint8).cpu().numpy()
