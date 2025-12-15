from __future__ import annotations

from pathlib import Path
import math

import pytest
from Bio import Phylo
import yaml

from src.data_generation import verify_from_config
from src.data_generation.config import ConfigurationError, GenerationConfig
from src.data_generation.tree_sequence_generator import TreeSequenceGenerator
from src.utils import infer_branch_output_count


def _branch_lengths(tree: Phylo.BaseTree.Tree) -> list[float]:
    lengths: list[float] = []
    for clade in tree.find_clades(order="level"):
        for child in clade.clades:
            if child.branch_length is not None:
                lengths.append(child.branch_length)
    return lengths


@pytest.fixture()
def generation_config(tmp_path: Path) -> GenerationConfig:
    payload: dict[str, object] = {
        "seed": 42,
        "tree": {
            "taxa_labels": ["A", "B"],
            "branch_length_range": [0.1, 1.0],
            "rooted": True,
            "topologies": ["(A,:B)"],
        },
        "sequence": {"length": 8, "model": "JC"},
        "simulation": {
            "backend": "iqtree",
            "iqtree_path": "/fake/iqtree",
            "seqgen_path": "/fake/seq-gen",
            "seqgen_kwargs": {},
            "indel": {"enabled": False},
        },
        "dataset": {
            "tree_count": 3,
            "output_name": "generated",
        },
    }
    return GenerationConfig.from_mapping(payload, base_path=tmp_path)


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
    lengths = _branch_lengths(result.tree)
    assert lengths
    min_len, max_len = generation_config.tree.branch_length_range
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
    payload = {
        "seed": 7,
        "tree": {
            "taxa_labels": ["A", "B"],
            "branch_length_range": [0.1, 1.0],
            "rooted": True,
            "topologies": ["(A,:B)"]
        },
        "sequence": {"length": 6, "model": "JC"},
        "simulation": {
            "backend": "iqtree",
            "iqtree_path": "/fake/iqtree",
            "seqgen_path": "/fake/seq-gen",
            "seqgen_kwargs": {},
            "indel": {"enabled": False},
        },
        "dataset": {"tree_count": 2, "output_name": "generated"},
    }
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

    expected_dir = tmp_path / "xml_verify"
    assert output_path == expected_dir / "generated.txt"
    assert output_path.exists()
    contents = output_path.read_text().strip().splitlines()
    assert len(contents) == config.dataset.tree_count
    assert all(line.endswith(";") for line in contents)


def test_seqgen_stdout_parsing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = {
        "seed": 5,
        "tree": {
            "taxa_labels": ["A", "B"],
            "branch_length_range": [0.1, 0.2],
            "rooted": True,
            "topologies": ["(A,:B)"]
        },
        "sequence": {"length": 4, "model": "JC"},
        "simulation": {
            "backend": "seqgen",
            "seqgen_path": "/fake/seq-gen",
            "seqgen_kwargs": {},
            "indel": {"enabled": False},
        },
        "dataset": {"tree_count": 1, "output_name": "generated"},
    }
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
    payload = {
        "seed": 5,
        "tree": {
            "taxa_labels": ["A", "B"],
            "branch_length_range": [0.1, 0.2],
            "rooted": True,
            "topologies": ["(A,:B)"]
        },
        "sequence": {"length": 4, "model": "JC"},
        "simulation": {
            "backend": "seqgen",
            "seqgen_path": "/fake/seq-gen",
            "seqgen_kwargs": {},
            "indel": {"enabled": False},
        },
        "dataset": {"tree_count": 1, "output_name": "generated"},
    }
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
    payload = {
        "seed": 5,
        "tree": {
            "taxa_labels": ["A", "B"],
            "branch_length_range": [0.1, 0.2],
            "rooted": True,
            "topologies": ["(A,:B)"]
        },
        "sequence": {"length": 4, "model": "JC"},
        "simulation": {
            "backend": "seqgen",
            "seqgen_path": "/fake/seq-gen",
            "seqgen_kwargs": {},
            "indel": {"enabled": False},
        },
        "dataset": {"tree_count": 1, "output_name": "generated"},
    }
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
    payload = {
        "seed": 5,
        "tree": {
            "taxa_labels": ["A", "B"],
            "branch_length_range": [0.1, 0.2],
            "rooted": True,
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
    with pytest.raises(ConfigurationError):
        GenerationConfig.from_mapping(payload, base_path=tmp_path)


def test_rooted_topology_requires_colon(tmp_path: Path) -> None:
    payload = {
        "seed": 3,
        "tree": {
            "taxa_labels": ["A", "B"],
            "branch_length_range": [0.1, 0.2],
            "rooted": True,
            "topologies": ["(A,B)"]
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
    with pytest.raises(ConfigurationError):
        GenerationConfig.from_mapping(payload, base_path=tmp_path)


def test_unrooted_topology_ignores_colon(tmp_path: Path) -> None:
    payload = {
        "seed": 3,
        "tree": {
            "taxa_labels": ["A", "B", "C"],
            "branch_length_range": [0.1, 0.2],
            "rooted": False,
            "topologies": ["(A,:(B,C))"],
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
    with pytest.warns(RuntimeWarning):
        config = GenerationConfig.from_mapping(payload, base_path=tmp_path)
    assert not config.tree.rooted


def test_topology_rejects_duplicate_taxa(tmp_path: Path) -> None:
    payload = {
        "seed": 5,
        "tree": {
            "taxa_labels": ["A", "B", "C"],
            "branch_length_range": [0.1, 0.2],
            "rooted": True,
            "topologies": ["((A,A),:B)"],
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
    with pytest.raises(ConfigurationError, match=r"Duplicate taxa"):
        GenerationConfig.from_mapping(payload, base_path=tmp_path)


def test_branch_length_distribution_validation(tmp_path: Path) -> None:
    payload = {
        "seed": 12,
        "tree": {
            "taxa_labels": ["A", "B"],
            "branch_length_range": [0.1, 0.2],
            "rooted": True,
            "branch_length_distribution": "normal",
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
    with pytest.raises(ConfigurationError):
        GenerationConfig.from_mapping(payload, base_path=tmp_path)


def test_topology_cycle_even_distribution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = {
        "seed": 9,
        "tree": {
            "taxa_labels": ["A", "B", "C"],
            "branch_length_range": [0.1, 0.2],
            "rooted": True,
            "topologies": ["((A,B),:C)", "((A,C),:B)"]
        },
        "sequence": {"length": 6, "model": "JC"},
        "simulation": {
            "backend": "iqtree",
            "iqtree_path": "/fake/iqtree",
            "seqgen_path": "/fake/seq-gen",
            "seqgen_kwargs": {},
            "indel": {"enabled": False},
        },
        "dataset": {"tree_count": 5, "output_name": "generated"},
    }
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
    assert observed == ["((A,B),:C)", "((A,C),:B)", "((A,B),:C)", "((A,C),:B)", "((A,B),:C)"]


def test_root_insertion_preserves_neighbor_pairs(tmp_path: Path) -> None:
    payload = {
        "seed": 17,
        "tree": {
            "taxa_labels": ["A", "B", "C"],
            "branch_length_range": [0.1, 0.5],
            "rooted": True,
            "topologies": ["((A,B),:C)", "((A,C),:B)", "((B,C),:A)"],
        },
        "sequence": {"length": 10, "model": "JC"},
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
    payload = {
        "seed": 3,
        "tree": {
            "taxa_labels": ["A", "B", "C"],
            "branch_length_range": [0.1, 0.9],
            "rooted": True,
            "topologies": ["((A,B),:C)"],
        },
        "sequence": {"length": 8, "model": "JC"},
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
    payload = {
        "seed": 5,
        "tree": {
            "taxa_labels": ["A", "B", "C"],
            "branch_length_range": [0.0, 1.0],
            "rooted": True,
            "topologies": ["((A,B),:C)"],
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
    assert len(_branch_lengths(tree)) == infer_branch_output_count(3, rooted=True)


def test_unrooted_two_taxa_assigns_single_branch(tmp_path: Path) -> None:
    payload = {
        "seed": 12,
        "tree": {
            "taxa_labels": ["taxon_1", "taxon_2"],
            "branch_length_range": [0.1, 0.3],
            "rooted": False,
            "topologies": ["(taxon_1,taxon_2)"],
        },
        "sequence": {"length": 5, "model": "JC"},
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
    tree, _ = TreeSequenceGenerator(config)._build_tree(topology_override=config.tree.topologies[0])

    lengths = _branch_lengths(tree)
    assert len(lengths) == 1
    root = tree.root
    first_taxon = config.tree.taxa_labels[0]
    first_child = next(child for child in root.clades if child.name == first_taxon)
    other_child = next(child for child in root.clades if child.name != first_taxon)
    assert first_child.branch_length is not None
    assert other_child.branch_length is None
    assert math.isclose(lengths[0], first_child.branch_length, rel_tol=1e-9)


def test_three_taxa_tree_respects_topology(tmp_path: Path) -> None:
    payload = {
        "seed": 11,
        "tree": {
            "taxa_labels": ["sp1", "sp2", "sp3"],
            "branch_length_range": [0.5, 1.0],
            "rooted": True,
            "topologies": ["((sp1,sp2),:sp3)"],
        },
        "sequence": {"length": 8, "model": "JC"},
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
    tree, _ = TreeSequenceGenerator(config)._build_tree(topology_override=config.tree.topologies[0])

    assert len(tree.get_terminals()) == 3
    internal_term_sets = {frozenset(clade.name for clade in node.get_terminals()) for node in tree.get_nonterminals()}
    assert frozenset({"sp1", "sp2"}) in internal_term_sets
    lengths = _branch_lengths(tree)
    assert lengths
    assert all(0 <= length <= 1.0 for length in lengths)
    assert any(length >= 0.5 for length in lengths)
    assert len(lengths) == infer_branch_output_count(3, rooted=True)
    assert tree.rooted


def test_four_taxa_tree_supports_double_cherries(tmp_path: Path) -> None:
    payload = {
        "seed": 21,
        "tree": {
            "taxa_labels": ["sp1", "sp2", "sp3", "sp4"],
            "branch_length_range": [0.2, 0.6],
            "rooted": False,
            "topologies": ["((sp1,sp2),(sp3,sp4))"],
        },
        "sequence": {"length": 8, "model": "JC"},
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
    tree, _ = TreeSequenceGenerator(config)._build_tree(topology_override=config.tree.topologies[0])

    assert len(tree.get_terminals()) == 4
    internal_term_sets = {frozenset(clade.name for clade in node.get_terminals()) for node in tree.get_nonterminals()}
    assert frozenset({"sp1", "sp2"}) in internal_term_sets
    assert frozenset({"sp3", "sp4"}) in internal_term_sets
    lengths = _branch_lengths(tree)
    assert lengths
    assert all(0 <= length <= 0.6 for length in lengths)
    assert any(length >= 0.2 for length in lengths)
    assert not tree.rooted


def test_topology_validation_requires_all_taxa(tmp_path: Path) -> None:
    payload = {
        "seed": 15,
        "tree": {
            "taxa_labels": ["A", "B", "C", "D"],
            "branch_length_range": [0.1, 1.0],
            "rooted": True,
            "topologies": ["((A,B),:C)"],
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
    with pytest.raises(ConfigurationError):
        GenerationConfig.from_mapping(payload, base_path=tmp_path)
