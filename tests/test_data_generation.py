from __future__ import annotations

from pathlib import Path
import math

import pytest
from Bio import Phylo
import yaml

from src.data_generation import verify_from_config, verify_sequences_from_config
from src.data_generation.config import ConfigurationError, GenerationConfig
from src.data_generation.tree_sequence_generator import TreeSequenceGenerator
from src.utils import infer_branch_output_count

from tests.test_inputs import (
    branch_lengths,
    build_payload,
    generation_config,
    iqtree_simulation,
    seqgen_simulation,
)


def test_generate_tree_and_sequences(monkeypatch: pytest.MonkeyPatch, generation_config: GenerationConfig) -> None:
    generator = TreeSequenceGenerator(generation_config)

    monkeypatch.setattr(
        TreeSequenceGenerator,
        "_simulate_with_iqtree",
        lambda self, *args, **kwargs: {
            "A": "A" * generation_config.sequence.length,
            "B": "C" * generation_config.sequence.length,
        },
    )

    result = generator.generate_tree_and_sequences()
    assert len(result.tree.get_terminals()) == 2
    lengths = branch_lengths(result.tree)
    assert lengths
    min_len, max_len = generation_config.tree.branch_length_params["uniform"]["range"]
    assert all(0 <= length <= max_len for length in lengths)
    assert any(length >= min_len for length in lengths)
    assert len(lengths) == infer_branch_output_count(len(generation_config.tree.taxa_labels), rooted=True)
    assert result.sequences == {
        "A": "A" * generation_config.sequence.length,
        "B": "C" * generation_config.sequence.length,
    }
    assert result.aligned
    assert result.topology in generation_config.tree.topologies


def test_generate_phylogenies_respects_count(monkeypatch: pytest.MonkeyPatch, generation_config: GenerationConfig) -> None:
    generator = TreeSequenceGenerator(generation_config)

    def fake_sim(*args, **kwargs):  # pragma: no cover - patched within test
        return {
            "A": "A" * generation_config.sequence.length,
            "B": "C" * generation_config.sequence.length,
        }

    monkeypatch.setattr(TreeSequenceGenerator, "_simulate_with_iqtree", fake_sim)

    phylogenies, aligned = generator.generate_phylogenies()
    assert len(phylogenies) == generation_config.dataset.tree_count
    assert aligned
    for phylogeny in phylogenies:
        assert isinstance(phylogeny, Phylo.PhyloXML.Phylogeny)
        assert phylogeny.other and phylogeny.other[0].value == "(A,:B)"


def test_write_xml_creates_expected_phyloxml(monkeypatch: pytest.MonkeyPatch, generation_config: GenerationConfig) -> None:
    generator = TreeSequenceGenerator(generation_config)

    monkeypatch.setattr(
        TreeSequenceGenerator,
        "_simulate_with_iqtree",
        lambda self, *args, **kwargs: {
            "A": "A" * generation_config.sequence.length,
            "B": "C" * generation_config.sequence.length,
        },
    )

    output_path = generator.write_xml()
    assert output_path.exists()

    phyloxml_entries = list(Phylo.parse(str(output_path), "phyloxml"))
    assert len(phyloxml_entries) == generation_config.dataset.tree_count
    assert phyloxml_entries[0].other[0].value == "(A,:B)"


def test_verify_module_emits_newick_dump(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = build_payload(
        seed=7,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=6,
        tree_count=2,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    monkeypatch.setattr(
        TreeSequenceGenerator,
        "_simulate_with_iqtree",
        lambda self, *args, **kwargs: {
            "A": "A" * config.sequence.length,
            "B": "C" * config.sequence.length,
        },
    )

    generator.write_xml()
    output_path = verify_from_config(config_path)

    expected_dir = tmp_path / "xml_data" / "verify"
    assert output_path == expected_dir / "generated.txt"
    assert output_path.exists()
    contents = output_path.read_text().strip().splitlines()
    assert len(contents) == config.dataset.tree_count
    assert all(line.endswith(";") for line in contents)


def test_verify_module_with_custom_output_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = build_payload(
        seed=8,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=6,
        tree_count=2,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    monkeypatch.setattr(
        TreeSequenceGenerator,
        "_simulate_with_iqtree",
        lambda self, *args, **kwargs: {
            "A": "A" * config.sequence.length,
            "B": "C" * config.sequence.length,
        },
    )

    generator.write_xml()
    custom_output = tmp_path / "custom" / "out.txt"
    output_path = verify_from_config(config_path, output_path=custom_output)

    assert output_path == custom_output
    assert output_path.exists()
    contents = output_path.read_text().strip().splitlines()
    assert len(contents) == config.dataset.tree_count
    assert all(line.endswith(";") for line in contents)


def test_verify_module_raises_when_xml_missing(tmp_path: Path) -> None:
    payload = build_payload(
        seed=8,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=6,
        tree_count=2,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        verify_from_config(config_path)


def test_indel_sizes_parsed_from_config(tmp_path: Path) -> None:
    payload = build_payload(
        seed=21,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=4,
        tree_count=1,
        output_name="indel_sizes",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
        simulation=iqtree_simulation(
            indel_enabled=True,
            indel_rates=[0.02, 0.03],
            indel_sizes=["POW{1.5/50}", "GEO{5}"],
        ),
    )

    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    assert config.simulation.indel.sizes == ("POW{1.5/50}", "GEO{5}")


def test_verify_module_with_custom_xml_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that verify module uses custom xml_directory when specified."""
    custom_xml_dir = str(tmp_path / "my_custom_xml")
    payload = build_payload(
        seed=7,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=6,
        tree_count=2,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
        xml_directory=custom_xml_dir,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    monkeypatch.setattr(
        TreeSequenceGenerator,
        "_simulate_with_iqtree",
        lambda self, *args, **kwargs: {
            "A": "A" * config.sequence.length,
            "B": "C" * config.sequence.length,
        },
    )

    generator.write_xml()
    output_path = verify_from_config(config_path)

    expected_dir = Path(custom_xml_dir) / "verify"
    assert output_path == expected_dir / "generated.txt"
    assert output_path.exists()
    contents = output_path.read_text().strip().splitlines()
    assert len(contents) == config.dataset.tree_count
    assert all(line.endswith(";") for line in contents)


def test_verify_sequences_module_emits_fasta_dump(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = build_payload(
        seed=11,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=5,
        tree_count=2,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    monkeypatch.setattr(
        TreeSequenceGenerator,
        "_simulate_with_iqtree",
        lambda self, *args, **kwargs: {
            "A": "A" * config.sequence.length,
            "B": "C" * config.sequence.length,
        },
    )

    generator.write_xml()
    output_path = verify_sequences_from_config(config_path)

    expected_dir = tmp_path / "xml_data" / "verify"
    assert output_path == expected_dir / "generated_sequences.fasta"
    assert output_path.exists()
    contents = output_path.read_text().strip().splitlines()
    assert contents == [
        ">A_1",
        "A" * config.sequence.length,
        ">B_1",
        "C" * config.sequence.length,
        ">A_2",
        "A" * config.sequence.length,
        ">B_2",
        "C" * config.sequence.length,
    ]


def test_verify_sequences_preserves_gaps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = build_payload(
        seed=12,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=4,
        tree_count=1,
        output_name="gapped",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
        simulation=iqtree_simulation(indel_enabled=True, indel_rates=[0.02, 0.02]),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    monkeypatch.setattr(
        TreeSequenceGenerator,
        "_simulate_with_iqtree",
        lambda self, *args, **kwargs: {
            "A": "A--T",
            "B": "TT-A",
        },
    )

    generator.write_xml()
    output_path = verify_sequences_from_config(config_path)
    contents = output_path.read_text().strip().splitlines()
    assert contents == [
        ">A_1",
        "A--T",
        ">B_1",
        "TT-A",
    ]


def test_verify_sequences_module_with_custom_xml_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom_xml_dir = str(tmp_path / "custom_xml")
    payload = build_payload(
        seed=13,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=4,
        tree_count=2,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
        xml_directory=custom_xml_dir,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    monkeypatch.setattr(
        TreeSequenceGenerator,
        "_simulate_with_iqtree",
        lambda self, *args, **kwargs: {
            "A": "A" * config.sequence.length,
            "B": "C" * config.sequence.length,
        },
    )

    generator.write_xml()
    output_path = verify_sequences_from_config(config_path)

    expected_dir = Path(custom_xml_dir) / "verify"
    assert output_path == expected_dir / "generated_sequences.fasta"
    assert output_path.exists()
    contents = output_path.read_text().strip().splitlines()
    assert contents == [
        ">A_1",
        "A" * config.sequence.length,
        ">B_1",
        "C" * config.sequence.length,
        ">A_2",
        "A" * config.sequence.length,
        ">B_2",
        "C" * config.sequence.length,
    ]


def test_seqgen_stdout_parsing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = build_payload(
        seed=5,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=4,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 0.2]}},
        simulation=seqgen_simulation(),
    )
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    dummy_fasta = ">A\nAAAA\n>B\nCCCC\n"
    captured_command: dict[str, object] = {}

    class DummyResult:
        stdout = dummy_fasta
        stderr = ""

    def fake_run(command, **kwargs):  # pragma: no cover - exercised through monkeypatch
        captured_command["cmd"] = command
        cwd = kwargs.get("cwd")
        assert kwargs.get("check")
        assert kwargs.get("capture_output")
        assert kwargs.get("text")
        assert cwd and Path(cwd).exists()
        tree_path = Path(command[-1])
        assert tree_path.exists()
        assert tree_path.parent == Path(cwd)
        return DummyResult()

    monkeypatch.setattr("src.data_generation.tree_sequence_generator.os.path.isfile", lambda path: True)
    monkeypatch.setattr("src.data_generation.tree_sequence_generator.subprocess.run", fake_run)

    seq_map = generator._simulate_with_seqgen(
        "(A:0.1,B:0.1);",
        seq_length=4,
        seqgen_path="/fake/seq-gen",
        seqgen_kwargs={},
    )

    assert seq_map == {"A": "AAAA", "B": "CCCC"}
    issued_cmd = captured_command.get("cmd", [])
    assert issued_cmd and issued_cmd[0] == "/fake/seq-gen"
    assert "-of" in issued_cmd
    assert issued_cmd[-1].endswith(".nwk")


def test_seqgen_reads_output_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = build_payload(
        seed=5,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=4,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 0.2]}},
        simulation=seqgen_simulation(),
    )
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    dummy_fasta = ">A\nGGGG\n>B\nTTTT\n"

    class DummyResult:
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):  # pragma: no cover - exercised through monkeypatch
        cwd = Path(kwargs.get("cwd"))
        assert cwd.exists()
        tree_path = Path(command[-1])
        assert tree_path.exists()
        output_path = cwd / "seqgen_1.fasta"
        output_path.write_text(dummy_fasta)
        return DummyResult()

    monkeypatch.setattr("src.data_generation.tree_sequence_generator.os.path.isfile", lambda path: True)
    monkeypatch.setattr("src.data_generation.tree_sequence_generator.subprocess.run", fake_run)

    seq_map = generator._simulate_with_seqgen(
        "(A:0.1,B:0.1);",
        seq_length=4,
        seqgen_path="/fake/seq-gen",
        seqgen_kwargs={},
    )

    assert seq_map == {"A": "GGGG", "B": "TTTT"}


def test_seqgen_rejects_multiple_replicates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = build_payload(
        seed=5,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=4,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 0.2]}},
        simulation=seqgen_simulation(),
    )
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    monkeypatch.setattr("src.data_generation.tree_sequence_generator.os.path.isfile", lambda path: True)

    with pytest.raises(ValueError):
        generator._simulate_with_seqgen(
            "(A:0.1,B:0.1);",
            seq_length=4,
            seqgen_path="/fake/seq-gen",
            seqgen_kwargs={"replicates": 2},
        )


def test_topologies_required(tmp_path: Path) -> None:
    payload = build_payload(
        seed=5,
        taxa_labels=["A", "B"],
        topologies=[],
        sequence_length=4,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 0.2]}},
    )
    payload["tree"].pop("topologies")
    with pytest.raises(ConfigurationError):
        GenerationConfig.from_mapping(payload, base_path=tmp_path)


def test_rooted_topology_requires_colon(tmp_path: Path) -> None:
    payload = build_payload(
        seed=3,
        taxa_labels=["A", "B"],
        topologies=["(A,B)"],
        sequence_length=4,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 0.2]}},
    )
    with pytest.raises(ConfigurationError):
        GenerationConfig.from_mapping(payload, base_path=tmp_path)


def test_unrooted_topology_ignores_colon(tmp_path: Path) -> None:
    payload = build_payload(
        seed=3,
        taxa_labels=["A", "B", "C"],
        topologies=["(A,:(B,C))"],
        sequence_length=4,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 0.2]}},
        rooted=False,
    )
    with pytest.warns(RuntimeWarning):
        config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    assert not config.tree.rooted


def test_topology_rejects_duplicate_taxa(tmp_path: Path) -> None:
    payload = build_payload(
        seed=5,
        taxa_labels=["A", "B", "C"],
        topologies=["((A,A),:B)"],
        sequence_length=4,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 0.2]}},
    )
    with pytest.raises(ConfigurationError, match=r"Duplicate taxa"):
        GenerationConfig.from_mapping(payload, base_path=tmp_path)


def test_branch_length_distribution_validation(tmp_path: Path) -> None:
    """Test that invalid distribution names are rejected."""
    payload = build_payload(
        seed=12,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=4,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"invalid_distribution": 1.0},
        branch_length_params={"invalid_distribution": {"param": 0.5}},
    )
    with pytest.raises(ConfigurationError):
        GenerationConfig.from_mapping(payload, base_path=tmp_path)


def test_exponential_distribution_uses_rate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = {
        "seed": 12,
        "tree": {
            "taxa_labels": ["A", "B"],
            "branch_length_distributions": {"exponential": 1.0},
            "branch_length_params": {"exponential": {"rate": 2.5}},
            "rooted": True,
            "topologies": ["(A,:B)"],
        },
        "sequence": {"length": 4, "model": "JC"},
        "simulation": {
            "backend": "iqtree",
            "iqtree_path": "/fake/iqtree",
            "seqgen_path": "/fake/seq-gen",
            "seqgen_kwargs": {},
            "indel": {"enabled": False},
        },
        "dataset": {"tree_count": 1, "output_name": "generated"},
    }
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    captured: list[float] = []

    def fake_expovariate(rate: float) -> float:
        captured.append(rate)
        return 0.123

    monkeypatch.setattr(generator._rng, "expovariate", fake_expovariate)
    value = generator._sample_branch_length()
    assert captured == [2.5]
    assert value == pytest.approx(0.123)


def test_truncated_exponential_bounds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = {
        "seed": 12,
        "tree": {
            "taxa_labels": ["A", "B"],
            "branch_length_distributions": {"truncated_exponential": 1.0},
            "branch_length_params": {
                "truncated_exponential": {"rate": 3.0, "min": 0.1, "max": 0.5}
            },
            "rooted": True,
            "topologies": ["(A,:B)"],
        },
        "sequence": {"length": 4, "model": "JC"},
        "simulation": {
            "backend": "iqtree",
            "iqtree_path": "/fake/iqtree",
            "seqgen_path": "/fake/seq-gen",
            "seqgen_kwargs": {},
            "indel": {"enabled": False},
        },
        "dataset": {"tree_count": 1, "output_name": "generated"},
    }
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    monkeypatch.setattr(generator._rng, "random", lambda: 0.0)
    value = generator._sample_branch_length()
    assert value == pytest.approx(0.1)


def test_indel_sizes_passed_to_iqtree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = {
        "seed": 17,
        "parallel_cores": 1,
        "tree": {
            "taxa_labels": ["A", "B"],
            "branch_length_distributions": {"uniform": 1.0},
            "branch_length_params": {"uniform": {"range": [0.1, 0.2]}},
            "rooted": True,
            "topologies": ["(A,:B)"],
        },
        "sequence": {"length": 4, "model": "JC"},
        "simulation": {
            "backend": "iqtree",
            "iqtree_path": "/fake/iqtree",
            "seqgen_path": "/fake/seq-gen",
            "seqgen_kwargs": {},
            "indel": {
                "enabled": True,
                "rates": [0.02, 0.03],
                "sizes": ["POW{1.5/50}", "GEO{5}"],
            },
        },
        "dataset": {"tree_count": 1, "output_name": "generated"},
    }
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    captured: dict[str, object] = {}

    def fake_sim(*args, **kwargs):
        captured["indel_rate"] = kwargs.get("indel_rate")
        captured["indel_size"] = kwargs.get("indel_size")
        return {
            "A": "A" * config.sequence.length,
            "B": "C" * config.sequence.length,
        }

    monkeypatch.setattr(TreeSequenceGenerator, "_simulate_with_iqtree", fake_sim)
    generator.generate_tree_and_sequences()

    assert captured["indel_rate"] == (0.02, 0.03)
    assert captured["indel_size"] == ("POW{1.5/50}", "GEO{5}")


def test_split_root_branch_flag_parsing(tmp_path: Path) -> None:
    payload = build_payload(
        seed=12,
        taxa_labels=["A", "B", "C"],
        topologies=["((A,B),:C)"],
        sequence_length=4,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 0.3]}},
        split_root_branch=False,
    )

    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    assert config.tree.split_root_branch is False


def test_topology_cycle_even_distribution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = build_payload(
        seed=9,
        taxa_labels=["A", "B", "C"],
        topologies=["((A,B),:C)", "((A,C),:B)"],
        sequence_length=6,
        tree_count=5,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 0.2]}},
    )
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    def fake_sim(*args, **kwargs):
        return {
            "A": "A" * config.sequence.length,
            "B": "C" * config.sequence.length,
            "C": "G" * config.sequence.length,
        }

    monkeypatch.setattr(TreeSequenceGenerator, "_simulate_with_iqtree", fake_sim)

    phylogenies, _ = generator.generate_phylogenies()
    observed = [phylogeny.other[0].value for phylogeny in phylogenies]
    assert len(observed) == 5
    assert observed.count("((A,B),:C)") == 3
    assert observed.count("((A,C),:B)") == 2


def test_root_insertion_preserves_neighbor_pairs(tmp_path: Path) -> None:
    payload = build_payload(
        seed=17,
        taxa_labels=["A", "B", "C"],
        topologies=["((A,B),:C)", "((A,C),:B)", "((B,C),:A)"],
        sequence_length=10,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 0.5]}},
    )
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    for topology in config.tree.topologies:
        expected_pairs = {tuple(sorted(group)) for group in topology if len(group) == 2}
        for _ in range(10):
            tree, _ = generator._build_tree(topology_override=topology)
            cherries = {
                tuple(sorted(clade.name for clade in node.get_terminals()))
                for node in tree.get_nonterminals()
                if len(node.get_terminals()) == 2
            }
            assert expected_pairs <= cherries


def test_branch_sampling_uses_unrooted_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = build_payload(
        seed=3,
        taxa_labels=["A", "B", "C"],
        topologies=["((A,B),:C)"],
        sequence_length=8,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 0.9]}},
    )
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)
    samples: list[float] = []

    def fake_sample(self):  # type: ignore[override]
        value = 0.25
        samples.append(value)
        return value

    monkeypatch.setattr(TreeSequenceGenerator, "_sample_branch_length", fake_sample)
    generator._build_tree(topology_override=config.tree.topologies[0])

    expected = infer_branch_output_count(config.tree.taxa_count, rooted=False)
    assert len(samples) == expected


def test_root_split_preserves_total_length(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = build_payload(
        seed=5,
        taxa_labels=["A", "B", "C"],
        topologies=["((A,B),:C)"],
        sequence_length=6,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.0, 1.0]}},
    )
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    samples = iter([0.5, 0.4, 0.4])

    def fake_sample(self):  # type: ignore[override]
        return next(samples)

    monkeypatch.setattr(TreeSequenceGenerator, "_sample_branch_length", fake_sample)
    monkeypatch.setattr(generator._rng, "uniform", lambda low, high: (low + high) / 2 if high > low else high)

    tree, _ = generator._build_tree(topology_override=config.tree.topologies[0])
    left_child, right_child = tree.root.clades
    assert math.isclose(left_child.branch_length + right_child.branch_length, 0.5, rel_tol=1e-9)
    assert len(branch_lengths(tree)) == infer_branch_output_count(3, rooted=True)


def test_rooted_no_split_draws_independent_edges(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = build_payload(
        seed=5,
        taxa_labels=["A", "B", "C"],
        topologies=["((A,B),:C)"],
        sequence_length=6,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
        split_root_branch=False,
    )
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    generator = TreeSequenceGenerator(config)

    samples = iter([0.1, 0.2, 0.3, 0.4])

    def fake_sample(self):  # type: ignore[override]
        return next(samples)

    monkeypatch.setattr(TreeSequenceGenerator, "_sample_branch_length", fake_sample)

    tree, _ = generator._build_tree(topology_override=config.tree.topologies[0])

    lengths = sorted(branch_lengths(tree))
    assert len(lengths) == infer_branch_output_count(3, rooted=True)
    assert lengths == sorted([0.1, 0.2, 0.3, 0.4])


def test_unrooted_two_taxa_assigns_single_branch(tmp_path: Path) -> None:
    payload = build_payload(
        seed=12,
        taxa_labels=["taxon_1", "taxon_2"],
        topologies=["(taxon_1,taxon_2)"],
        sequence_length=5,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 0.3]}},
        rooted=False,
    )
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    tree, _ = TreeSequenceGenerator(config)._build_tree(topology_override=config.tree.topologies[0])

    lengths = branch_lengths(tree)
    assert len(lengths) == 1
    root = tree.root
    first_taxon = config.tree.taxa_labels[0]
    first_child = next(child for child in root.clades if child.name == first_taxon)
    other_child = next(child for child in root.clades if child.name != first_taxon)
    assert first_child.branch_length is not None
    assert other_child.branch_length is None
    assert math.isclose(lengths[0], first_child.branch_length, rel_tol=1e-9)


def test_three_taxa_tree_respects_topology(tmp_path: Path) -> None:
    payload = build_payload(
        seed=11,
        taxa_labels=["sp1", "sp2", "sp3"],
        topologies=["((sp1,sp2),:sp3)"],
        sequence_length=8,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.5, 1.0]}},
    )
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    tree, _ = TreeSequenceGenerator(config)._build_tree(topology_override=config.tree.topologies[0])

    assert len(tree.get_terminals()) == 3
    internal_term_sets = {frozenset(clade.name for clade in node.get_terminals()) for node in tree.get_nonterminals()}
    assert frozenset({"sp1", "sp2"}) in internal_term_sets
    lengths = branch_lengths(tree)
    assert lengths
    assert all(0 <= length <= 1.0 for length in lengths)
    assert any(length >= 0.5 for length in lengths)
    assert len(lengths) == infer_branch_output_count(3, rooted=True)
    assert tree.rooted


def test_four_taxa_tree_supports_double_cherries(tmp_path: Path) -> None:
    payload = build_payload(
        seed=21,
        taxa_labels=["sp1", "sp2", "sp3", "sp4"],
        topologies=["((sp1,sp2),(sp3,sp4))"],
        sequence_length=8,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.2, 0.6]}},
        rooted=False,
    )
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    tree, _ = TreeSequenceGenerator(config)._build_tree(topology_override=config.tree.topologies[0])

    assert len(tree.get_terminals()) == 4
    internal_term_sets = {frozenset(clade.name for clade in node.get_terminals()) for node in tree.get_nonterminals()}
    assert frozenset({"sp1", "sp2"}) in internal_term_sets
    assert frozenset({"sp3", "sp4"}) in internal_term_sets
    lengths = branch_lengths(tree)
    assert lengths
    assert all(0 <= length <= 0.6 for length in lengths)
    assert any(length >= 0.2 for length in lengths)
    assert not tree.rooted


def test_phylogeny_stores_newick_metadata(monkeypatch: pytest.MonkeyPatch, generation_config: GenerationConfig) -> None:
    generator = TreeSequenceGenerator(generation_config)

    monkeypatch.setattr(
        TreeSequenceGenerator,
        "_simulate_with_iqtree",
        lambda self, *args, **kwargs: {
            "A": "A" * generation_config.sequence.length,
            "B": "C" * generation_config.sequence.length,
        },
    )

    phylogeny, aligned = generator.generate_phylogeny()
    assert aligned
    assert phylogeny.other is not None
    tags = {entry.tag: entry.value for entry in phylogeny.other}
    assert tags.get("topology") == "(A,:B)"
    assert tags.get("newick") is not None
    assert tags["newick"].strip().endswith(";")


def test_topology_validation_requires_all_taxa(tmp_path: Path) -> None:
    payload = build_payload(
        seed=15,
        taxa_labels=["A", "B", "C", "D"],
        topologies=["((A,B),:C)"],
        sequence_length=4,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
    )
    with pytest.raises(ConfigurationError):
        GenerationConfig.from_mapping(payload, base_path=tmp_path)


def test_custom_xml_directory(tmp_path: Path) -> None:
    """Test that custom xml_directory is respected."""
    custom_xml_dir = str(tmp_path / "custom_xml")
    payload = build_payload(
        seed=42,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=8,
        tree_count=3,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
        xml_directory=custom_xml_dir,
    )
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    assert config.dataset.xml_directory == custom_xml_dir
    assert config.dataset.xml_path() == Path(custom_xml_dir) / "generated.xml"


def test_custom_npy_directory(tmp_path: Path) -> None:
    """Test that custom npy_directory is respected."""
    custom_npy_dir = str(tmp_path / "custom_npy")
    payload = build_payload(
        seed=42,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=8,
        tree_count=3,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
        npy_directory=custom_npy_dir,
    )
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    assert config.dataset.npy_directory == custom_npy_dir
    assert config.dataset.output_npy_path() == Path(custom_npy_dir) / "generated.npy"


def test_both_custom_directories(tmp_path: Path) -> None:
    """Test that both custom xml_directory and npy_directory are respected."""
    custom_xml_dir = str(tmp_path / "custom_xml")
    custom_npy_dir = str(tmp_path / "custom_npy")
    payload = build_payload(
        seed=42,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=8,
        tree_count=3,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
        xml_directory=custom_xml_dir,
        npy_directory=custom_npy_dir,
    )
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    assert config.dataset.xml_directory == custom_xml_dir
    assert config.dataset.npy_directory == custom_npy_dir
    assert config.dataset.xml_path() == Path(custom_xml_dir) / "generated.xml"
    assert config.dataset.output_npy_path() == Path(custom_npy_dir) / "generated.npy"


def test_default_directories_when_not_specified(tmp_path: Path) -> None:
    """Test that default directories are used when custom directories are not specified."""
    payload = build_payload(
        seed=42,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=8,
        tree_count=3,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
    )
    config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    assert config.dataset.xml_directory is None
    assert config.dataset.npy_directory is None
    assert config.dataset.xml_path() == tmp_path / "xml_data" / "generated.xml"
    assert config.dataset.output_npy_path() == tmp_path / "npy_data" / "generated.npy"


def test_tree_chunk_size_must_be_positive(tmp_path: Path) -> None:
    payload = build_payload(
        seed=42,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=8,
        tree_count=3,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
        tree_chunk_size=0,
    )
    with pytest.raises(ConfigurationError, match="'dataset.tree_chunk_size' must be positive"):
        GenerationConfig.from_mapping(payload, base_path=tmp_path)


def test_empty_xml_directory_raises_error(tmp_path: Path) -> None:
    """Test that empty xml_directory string raises ConfigurationError."""
    payload = build_payload(
        seed=42,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=8,
        tree_count=3,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
        xml_directory="",
    )
    with pytest.raises(ConfigurationError, match="'dataset.xml_directory' must be a non-empty string"):
        GenerationConfig.from_mapping(payload, base_path=tmp_path)


def test_empty_npy_directory_raises_error(tmp_path: Path) -> None:
    """Test that empty npy_directory string raises ConfigurationError."""
    payload = build_payload(
        seed=42,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=8,
        tree_count=3,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
        npy_directory="",
    )
    with pytest.raises(ConfigurationError, match="'dataset.npy_directory' must be a non-empty string"):
        GenerationConfig.from_mapping(payload, base_path=tmp_path)

