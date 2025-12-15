from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from src.data_generation.config import GenerationConfig, load_generation_config

from .base import TreeExample
from .extractor import PhyloExampleExtractor
from .writers.base import DatasetWriter
from .writers.npy_writer import NpyDatasetWriter


class XMLParser:
    """Coordinates extraction of phylogenies and persistence to various formats."""

    def __init__(self, config: GenerationConfig, *, output_path: Path | None = None) -> None:
        self.config = config
        self.clade_names = tuple(config.tree.taxa_labels)
        self.sequence_length = config.sequence.length
        self.xml_path = config.dataset.xml_path()
        self.output_path = output_path or config.dataset.output_npy_path()
        self.parallel_cores = max(1, config.parallel_cores)
        self._extractor = PhyloExampleExtractor(config)

    @classmethod
    def from_config_file(cls, config_path: Path | str) -> "XMLParser":
        config = load_generation_config(config_path)
        return cls(config)

    def parse_examples(self) -> list[TreeExample]:
        """Extract tree examples from the configured PhyloXML file."""
        return self._extractor.extract()

    def write_dataset(
        self,
        examples: Sequence[TreeExample] | None = None,
        *,
        writer: DatasetWriter | None = None,
    ) -> Path:
        """Persist parsed examples using the provided writer (defaults to NPY)."""
        material = list(examples) if examples is not None else self.parse_examples()
        selected_writer = writer or NpyDatasetWriter(self.config, output_path=self.output_path)
        return selected_writer.write(material)


class PhyloXMLDatasetBuilder(XMLParser):
    """Backward-compatible adapter exposing the legacy builder API."""

    def __init__(self, config: GenerationConfig) -> None:
        super().__init__(config)

    @classmethod
    def from_config_file(cls, config_path: Path | str) -> "PhyloXMLDatasetBuilder":
        config = load_generation_config(config_path)
        return cls(config)

    def build_dataset(self, examples: Sequence[TreeExample]) -> Path:
        return self.write_dataset(examples=examples)
