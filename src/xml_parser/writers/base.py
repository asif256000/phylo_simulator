from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from src.data_generation.config import GenerationConfig

from ..base import TreeExample


class DatasetWriter(ABC):
    """Abstract base class for dataset writers that persist parsed phylogenies."""

    def __init__(self, config: GenerationConfig, output_path: Path | None = None) -> None:
        self.config = config
        self.output_path = output_path
        self.parallel_cores = max(1, config.parallel_cores)

    @abstractmethod
    def write(self, examples: Sequence[TreeExample]) -> Path:
        """Persist the provided examples and return the materialized path."""
        raise NotImplementedError
