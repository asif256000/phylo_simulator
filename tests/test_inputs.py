from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
from Bio import Phylo

from src.data_generation.config import GenerationConfig
from src.data_generation.tree_sequence_generator import TreeSequenceGenerator


def branch_lengths(tree: Phylo.BaseTree.Tree) -> list[float]:
    lengths: list[float] = []
    for clade in tree.find_clades(order="level"):
        for child in clade.clades:
            if child.branch_length is not None:
                lengths.append(child.branch_length)
    return lengths


def iqtree_simulation(
    *,
    indel_enabled: bool = False,
    indel_rates: list[float] | None = None,
    indel_sizes: list[str] | None = None,
    iqtree_path: str = "/fake/iqtree",
    seqgen_path: str = "/fake/seq-gen",
    seqgen_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    indel: dict[str, Any] = {"enabled": indel_enabled}
    if indel_rates is not None:
        indel["rates"] = indel_rates
    if indel_sizes is not None:
        indel["sizes"] = indel_sizes

    return {
        "backend": "iqtree",
        "iqtree_path": iqtree_path,
        "seqgen_path": seqgen_path,
        "seqgen_kwargs": dict(seqgen_kwargs or {}),
        "indel": indel,
    }


def seqgen_simulation(
    *,
    seqgen_path: str = "/fake/seq-gen",
    seqgen_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "backend": "seqgen",
        "iqtree_path": None,
        "seqgen_path": seqgen_path,
        "seqgen_kwargs": dict(seqgen_kwargs or {}),
        "indel": {"enabled": False},
    }


def build_payload(
    *,
    seed: int,
    taxa_labels: list[str],
    topologies: list[str],
    sequence_length: int,
    model: str = "JC",
    tree_count: int,
    output_name: str,
    branch_length_distributions: Mapping[str, float],
    branch_length_params: Mapping[str, Mapping[str, Any]],
    rooted: bool = True,
    split_root_branch: bool = True,
    parallel_cores: int = 1,
    simulation: Mapping[str, Any] | None = None,
    verify_padding_for_fasta: bool | None = None,
    xml_directory: str | None = None,
    npy_directory: str | None = None,
    tree_chunk_size: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "seed": seed,
        "parallel_cores": parallel_cores,
        "tree": {
            "taxa_labels": taxa_labels,
            "branch_length_distributions": dict(branch_length_distributions),
            "branch_length_params": {name: dict(params) for name, params in branch_length_params.items()},
            "rooted": rooted,
            "topologies": topologies,
            "split_root_branch": split_root_branch,
        },
        "sequence": {"length": sequence_length, "model": model},
        "simulation": dict(simulation or iqtree_simulation()),
        "dataset": {"tree_count": tree_count, "output_name": output_name},
    }

    if xml_directory is not None:
        payload["dataset"]["xml_directory"] = xml_directory
    if npy_directory is not None:
        payload["dataset"]["npy_directory"] = npy_directory
    if tree_chunk_size is not None:
        payload["dataset"]["tree_chunk_size"] = tree_chunk_size
    if verify_padding_for_fasta is not None:
        payload["verify"] = {"padding_for_fasta": verify_padding_for_fasta}

    return payload


def uniform_payload(
    *,
    seed: int,
    taxa_labels: list[str],
    topologies: list[str],
    sequence_length: int,
    tree_count: int,
    output_name: str,
    uniform_range: tuple[float, float],
    rooted: bool = True,
    split_root_branch: bool = True,
    parallel_cores: int = 1,
    model: str = "JC",
    simulation: Mapping[str, Any] | None = None,
    verify_padding_for_fasta: bool | None = None,
    xml_directory: str | None = None,
    npy_directory: str | None = None,
    tree_chunk_size: int | None = None,
) -> dict[str, Any]:
    return build_payload(
        seed=seed,
        taxa_labels=taxa_labels,
        topologies=topologies,
        sequence_length=sequence_length,
        model=model,
        tree_count=tree_count,
        output_name=output_name,
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": list(uniform_range)}},
        rooted=rooted,
        split_root_branch=split_root_branch,
        parallel_cores=parallel_cores,
        simulation=simulation,
        verify_padding_for_fasta=verify_padding_for_fasta,
        xml_directory=xml_directory,
        npy_directory=npy_directory,
        tree_chunk_size=tree_chunk_size,
    )


@pytest.fixture()
def generation_config(tmp_path: Path) -> GenerationConfig:
    payload = build_payload(
        seed=42,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=8,
        tree_count=3,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 1.0]}},
        tree_chunk_size=2,
    )
    return GenerationConfig.from_mapping(payload, base_path=tmp_path)


@pytest.fixture
def config(tmp_path_factory: pytest.TempPathFactory) -> GenerationConfig:
    base_dir = tmp_path_factory.mktemp("config")
    payload = build_payload(
        seed=7,
        taxa_labels=["A", "B"],
        topologies=["(A,:B)"],
        sequence_length=6,
        tree_count=1,
        output_name="generated",
        branch_length_distributions={"uniform": 1.0},
        branch_length_params={"uniform": {"range": [0.1, 0.2]}},
    )
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
