from pathlib import Path

import numpy as np
import pytest

from src.data_generation.config import GenerationConfig
from src.data_generation.tree_sequence_generator import TreeSequenceGenerator
from src.xml_parser import CladeRecord, TreeExample, XMLParser, one_hot_encode
from src.xml_parser.writers.npy_writer import NpyDatasetWriter


@pytest.fixture
def config(tmp_path_factory: pytest.TempPathFactory) -> GenerationConfig:
    base_dir = tmp_path_factory.mktemp("config")
    payload = {
        "seed": 7,
        "parallel_cores": 1,
        "tree": {
            "taxa_labels": ["A", "B"],
            "branch_length_range": [0.1, 0.2],
            "rooted": True,
            "topologies": ["(A,:B)"],
        },
        "sequence": {"length": 6, "model": "JC"},
        "simulation": {
            "backend": "iqtree",
            "iqtree_path": "/fake/iqtree",
            "seqgen_path": "/fake/seq-gen",
            "seqgen_kwargs": {},
            "indel": {"enabled": False},
        },
        "dataset": {"tree_count": 1, "output_name": "generated"},
    }
    return GenerationConfig.from_mapping(payload, base_path=base_dir)


@pytest.fixture
def phyloxml_file(monkeypatch: pytest.MonkeyPatch, config: GenerationConfig) -> Path:
    generator = TreeSequenceGenerator(config)
    monkeypatch.setattr(
        TreeSequenceGenerator,
        "_simulate_with_iqtree",
        lambda self, *args, **kwargs: {
            "A": "A" * config.sequence.length,
            "B": "C" * config.sequence.length,
        },
    )
    return generator.write_xml()


def test_parse_examples_extracts_clades(phyloxml_file: Path, config: GenerationConfig) -> None:
    parser = XMLParser(config)
    assert parser.output_path.parent.name == "npy_data"
    assert parser.xml_path.parent.name == "xml_data"
    examples = parser.parse_examples()
    assert len(examples) == 1

    example = examples[0]
    assert example.tree_index == 0
    assert [clade.name for clade in example.clades] == ["A", "B"]
    assert all(len(clade.sequence) == len(example.clades[0].sequence) for clade in example.clades)
    assert example.metadata and example.metadata.get("topology") == "(A,:B)"
    
    assert example.branches
    assert frozenset(["A"]) in example.branches
    assert frozenset(["B"]) in example.branches


def test_build_dataset_creates_structured_npy(tmp_path: Path, config: GenerationConfig) -> None:
    parser = XMLParser(config, output_path=tmp_path / "dataset.npy")

    examples = [
        TreeExample(
            tree_index=3,
            clades=[
                CladeRecord(name="A", sequence="ATGCAA", branch_length=0.5),
                CladeRecord(name="B", sequence="TACGTT", branch_length=0.8),
            ],
            branches={
                frozenset(["A"]): 0.5,
                frozenset(["B"]): 0.8,
            },
            metadata={"topology": "(A,:B)"},
        )
    ]

    output_path = parser.write_dataset(examples)

    dataset = np.load(output_path, mmap_mode="r")
    assert dataset.shape == (1,)
    record = dataset[0]

    assert record["tree_index"] == 3
    
    assert "y_br" in record.dtype.names
    assert "branch_mask" in record.dtype.names
    assert "y_top" in record.dtype.names

    # For 2 taxa A, B, canonical slots are {A}, {B}
    assert record["y_br"].shape == (2,)
    # Order depends on canonical_branch_ordering, which sorts by size then name.
    # {A}, {B} are size 1. A comes before B.
    assert record["y_br"][0] == pytest.approx(0.5)
    assert record["y_br"][1] == pytest.approx(0.8)
    
    assert np.all(record["branch_mask"])

    assert record["y_top"].shape == (1,)
    assert record["y_top"][0] == 1.0

    assert record["X"].shape == (2, config.sequence.length, 4)
    assert np.all(record["X"][0].sum(axis=1) == 1)
    assert np.all(record["X"][1].sum(axis=1) == 1)


def test_one_hot_encode_validates_length(config: GenerationConfig) -> None:
    with pytest.raises(ValueError):
        one_hot_encode("ATGC", config.sequence.length)


def test_one_hot_encode_supports_gap_when_enabled() -> None:
    sequence = "AT-C"
    encoding = one_hot_encode(sequence, len(sequence), include_gap=True)
    assert encoding.shape == (4, 5)
    assert encoding[2, 4] == 1  # gap occupies the final channel
    with pytest.raises(ValueError):
        one_hot_encode(sequence, len(sequence))


def test_write_dataset_uses_gap_channel(tmp_path: Path) -> None:
    payload = {
        "seed": 11,
        "parallel_cores": 1,
        "tree": {
            "taxa_labels": ["A", "B"],
            "branch_length_range": [0.1, 0.2],
            "rooted": True,
            "topologies": ["(A,:B)"],
        },
        "sequence": {"length": 4, "model": "JC"},
        "simulation": {
            "backend": "iqtree",
            "iqtree_path": "/fake/iqtree",
            "seqgen_path": "/fake/seq-gen",
            "seqgen_kwargs": {},
            "indel": {"enabled": True, "rates": [0.02, 0.02]},
        },
        "dataset": {"tree_count": 1, "output_name": "gapped"},
    }
    config_with_indels = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    parser = XMLParser(config_with_indels, output_path=tmp_path / "gapped.npy")
    examples = [
        TreeExample(
            tree_index=0,
            clades=[
                CladeRecord(name="A", sequence="A-A-", branch_length=0.5),
                CladeRecord(name="B", sequence="TT-T", branch_length=0.7),
            ],
            branches={frozenset(["A"]): 0.5, frozenset(["B"]): 0.7},
            metadata={"topology": "(A,:B)"},
        )
    ]
    dataset_path = parser.write_dataset(examples=examples)
    dataset = np.load(dataset_path, mmap_mode="r")
    record = dataset[0]
    assert record["X"].shape == (2, 4, 5)
    assert record["X"][0, 1, 4] == 1


def test_branch_mapping_three_taxa(tmp_path: Path) -> None:
    payload = {
        "seed": 5,
        "parallel_cores": 1,
        "tree": {
            "taxa_labels": ["A", "B", "C"],
            "branch_length_range": [0.1, 0.2],
            "rooted": True,
            "topologies": ["((A,B),:C)"]
        },
        "sequence": {"length": 4, "model": "JC"},
        "simulation": {
            "backend": "iqtree",
            "iqtree_path": "/fake/iqtree",
            "seqgen_path": "/fake/seq-gen",
            "seqgen_kwargs": {},
            "indel": {"enabled": False},
        },
        "dataset": {"tree_count": 1, "output_name": "three_taxa"},
    }

    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    writer = NpyDatasetWriter(config, output_path=tmp_path / "three.npy")

    branches = {
        frozenset(["A"]): 0.1,
        frozenset(["B"]): 0.2,
        frozenset(["C"]): 0.3,
        frozenset(["B", "C"]): 0.9,
    }

    examples = [
        TreeExample(
            tree_index=0,
            clades=[
                CladeRecord(name="A", sequence="AAAA", branch_length=0.1),
                CladeRecord(name="B", sequence="CCCC", branch_length=0.2),
                CladeRecord(name="C", sequence="GGGG", branch_length=0.3),
            ],
            branches=branches,
            metadata={"topology": "((A,B),:C)"},
        )
    ]

    path = writer.write(examples)
    record = np.load(path)[0]

    assert record["y_br"].shape == (6,)
    assert record["branch_mask"].tolist() == [True, True, True, False, True, False]
    assert record["y_br"].tolist() == pytest.approx([0.1, 0.9, 0.2, 0.0, 0.3, 0.0])


def test_branch_mapping_four_taxa(tmp_path: Path) -> None:
    payload = {
        "seed": 9,
        "parallel_cores": 1,
        "tree": {
            "taxa_labels": ["A", "B", "C", "D"],
            "branch_length_range": [0.1, 0.2],
            "rooted": True,
            "topologies": ["((A,B),:(C,D))"],
        },
        "sequence": {"length": 4, "model": "JC"},
        "simulation": {
            "backend": "iqtree",
            "iqtree_path": "/fake/iqtree",
            "seqgen_path": "/fake/seq-gen",
            "seqgen_kwargs": {},
            "indel": {"enabled": False},
        },
        "dataset": {"tree_count": 1, "output_name": "four_taxa"},
    }

    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    writer = NpyDatasetWriter(config, output_path=tmp_path / "four.npy")

    branches = {
        frozenset(["A"]): 0.1,
        frozenset(["B"]): 0.2,
        frozenset(["C"]): 0.3,
        frozenset(["D"]): 0.4,
        frozenset(["B", "C", "D"]): 0.7,
        frozenset(["A", "B"]): 0.5,
        frozenset(["C", "D"]): 0.6,
    }

    examples = [
        TreeExample(
            tree_index=0,
            clades=[
                CladeRecord(name="A", sequence="AAAA", branch_length=0.1),
                CladeRecord(name="B", sequence="CCCC", branch_length=0.2),
                CladeRecord(name="C", sequence="GGGG", branch_length=0.3),
                CladeRecord(name="D", sequence="TTTT", branch_length=0.4),
            ],
            branches=branches,
            metadata={"topology": "((A,B),:(C,D))"},
        )
    ]

    path = writer.write(examples)
    record = np.load(path)[0]

    assert record["y_br"].shape == (10,)
    assert record["y_br"].tolist() == pytest.approx([0.1, 0.7, 0.2, 0.0, 0.3, 0.0, 0.4, 0.0, 0.5, 0.6])
    assert record["branch_mask"].tolist() == [True, True, True, False, True, False, True, False, True, True]


@pytest.mark.parametrize(
    ("taxa_labels", "topologies"),
    (
        (("A", "B", "C"), ("((A,B),:C)", "((A,C),:B)")),
        (("A", "B", "C", "D"), ("((A,B),:(C,D))", "((A,C),:(B,D))")),
    ),
)
def test_xml_parser_supports_multi_taxa_shapes(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    taxa_labels: tuple[str, ...],
    topologies: tuple[str, ...],
) -> None:
    base_dir = tmp_path_factory.mktemp("multi_taxa")
    payload = {
        "seed": 13,
        "parallel_cores": 1,
        "tree": {
            "taxa_labels": list(taxa_labels),
            "branch_length_range": [0.2, 0.4],
            "rooted": True,
            "topologies": list(topologies),
        },
        "sequence": {"length": 12, "model": "JC"},
        "simulation": {
            "backend": "iqtree",
            "iqtree_path": "/fake/iqtree",
            "seqgen_path": "/fake/seq-gen",
            "seqgen_kwargs": {},
            "indel": {"enabled": False},
        },
        "dataset": {"tree_count": 2, "output_name": "multi_dataset"},
    }
    config = GenerationConfig.from_mapping(payload, base_path=base_dir)
    allowed_nucleotides = ("A", "T", "G", "C")
    sequences = {
        label: allowed_nucleotides[index % len(allowed_nucleotides)] * config.sequence.length
        for index, label in enumerate(taxa_labels)
    }

    monkeypatch.setattr(
        TreeSequenceGenerator,
        "_simulate_with_iqtree",
        lambda self, *args, **kwargs: dict(sequences),
    )

    generator = TreeSequenceGenerator(config)
    generator.write_xml()

    output_path = base_dir / f"dataset_{len(taxa_labels)}.npy"
    parser = XMLParser(config, output_path=output_path)
    emitted_path = parser.write_dataset()

    dataset = np.load(emitted_path)
    expected_taxa = len(taxa_labels)
    assert dataset.shape == (config.dataset.tree_count,)
    assert dataset["X"].shape == (
        config.dataset.tree_count,
        expected_taxa,
        config.sequence.length,
        4,
    )
    assert dataset["y_br"].shape[0] == config.dataset.tree_count
    assert dataset["y_br"].shape[1] >= expected_taxa
    assert "y_top" in dataset.dtype.names
